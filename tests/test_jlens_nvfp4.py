#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_jlens_nvfp4", SCRIPTS / "run_jlens_nvfp4.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class JacobianLensHelpersTest(unittest.TestCase):
    def test_report_schema_attests_unrounded_float32_scores(self):
        self.assertEqual(MODULE.SCHEMA_VERSION, 3)
        self.assertEqual(MODULE.SCORE_ENCODING, "unrounded-float32")

    def test_public_lens_provenance_does_not_claim_fit_precision(self):
        self.assertEqual(MODULE.PUBLIC_FIT_TIME_MODEL_PRECISION, "unpublished")
        self.assertEqual(MODULE.PUBLIC_FIT_TIME_QUANTIZATION, "unpublished")
        self.assertIn("FP16 lens", MODULE.PUBLIC_LENS_APPLICATION)
        self.assertIn("unpublished fit-time precision", MODULE.PUBLIC_LENS_APPLICATION)
        self.assertNotIn("BF16-fitted", MODULE.PUBLIC_LENS_APPLICATION)

    def test_parse_all_layers(self):
        self.assertEqual(MODULE.parse_integer_list("all", allow_all=True), list(range(63)))

    def test_parse_integer_list_rejects_duplicates(self):
        with self.assertRaises(Exception):
            MODULE.parse_integer_list("1,2,1")

    def test_validate_layers_sorts_and_bounds_checks(self):
        self.assertEqual(MODULE.validate_layers([62, 0, 31]), [0, 31, 62])
        with self.assertRaises(ValueError):
            MODULE.validate_layers([63])

    def test_resolve_negative_positions(self):
        self.assertEqual(MODULE.resolve_positions([-1, -3, 0], 8), [7, 5, 0])

    def test_resolve_positions_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            MODULE.resolve_positions([-9], 8)

    def test_target_tokens_use_teacher_forcing_and_generation(self):
        self.assertEqual(
            MODULE.target_token_ids_for_positions([10, 20, 30], [0, 2], 99),
            (20, 99),
        )

    def test_target_tokens_require_final_position(self):
        with self.assertRaises(ValueError):
            MODULE.target_token_ids_for_positions([10, 20, 30], [0, 1], 99)

    def test_target_token_override_applies_only_to_final_position(self):
        self.assertEqual(
            MODULE.target_token_ids_for_positions(
                [10, 20, 30],
                [0, 2],
                99,
                target_token_id_override=77,
            ),
            (20, 77),
        )

    def test_capture_positions_add_final_implicitly(self):
        self.assertEqual(MODULE.capture_positions_with_final([1, 3], 6), [1, 3, 5])

    def test_capture_positions_do_not_duplicate_final(self):
        self.assertEqual(MODULE.capture_positions_with_final([1, 5], 6), [1, 5])

    def test_post_block_reconstructs_branch_plus_residual(self):
        self.assertEqual(MODULE.reconstruct_post_block((7, 11)), 18)

    def test_transport_uses_transposed_jacobian(self):
        transpose_marker = object()

        class Jacobian:
            @property
            def T(self):
                return transpose_marker

        class Residual:
            def __matmul__(self, operand):
                self.operand = operand
                return "transported"

        residual = Residual()
        self.assertEqual(MODULE.transport_residual(residual, Jacobian()), "transported")
        self.assertIs(residual.operand, transpose_marker)

    def test_compact_topk_preserves_float32_scores_without_decimal_rounding(self):
        import torch

        logits = torch.tensor([0.123456791, 1.987654328, -2.0], dtype=torch.float32)
        result = MODULE._compact_topk(logits, top_k=2, target_token_id=0)
        self.assertEqual(result["scores"][0], float(logits[1]))
        self.assertEqual(result["target_score"], float(logits[0]))
        self.assertNotEqual(result["target_score"], round(float(logits[0]), 6))

    def test_compact_topk_records_exact_float32_target_logprob(self):
        import torch

        logits = torch.tensor([0.123456791, 1.987654328, -2.0], dtype=torch.float32)
        result = MODULE._compact_topk(logits, top_k=2, target_token_id=0)
        expected = torch.log_softmax(logits.float(), dim=-1)[0]
        self.assertEqual(result["target_logprob"], float(expected))
        self.assertNotEqual(result["target_logprob"], round(float(expected), 6))

    def test_compact_topk_records_requested_vocabulary_scores(self):
        import torch

        logits = torch.tensor([0.5, 1.5, -0.5], dtype=torch.float32)
        result = MODULE._compact_topk(
            logits,
            top_k=2,
            target_token_id=1,
            score_token_ids=(0, 2),
        )
        expected_logprobs = torch.log_softmax(logits, dim=-1)
        self.assertEqual(
            result["scored_tokens"],
            [
                {
                    "token_id": 0,
                    "score": float(logits[0]),
                    "logprob": float(expected_logprobs[0]),
                    "rank": 2,
                },
                {
                    "token_id": 2,
                    "score": float(logits[2]),
                    "logprob": float(expected_logprobs[2]),
                    "rank": 3,
                },
            ],
        )

    def test_scored_vocabulary_ids_are_range_checked(self):
        class Tokenizer:
            def __len__(self):
                return 3

        MODULE._validate_vocabulary_ids(
            Tokenizer(),
            prompt_id="valid",
            token_ids=[0, 1],
            target_token_id=2,
            score_token_ids=(0, 2),
        )
        with self.assertRaisesRegex(ValueError, "scored token IDs"):
            MODULE._validate_vocabulary_ids(
                Tokenizer(),
                prompt_id="invalid",
                token_ids=[0, 1],
                target_token_id=2,
                score_token_ids=(3,),
            )

    def test_distribution_fidelity_is_zero_for_identical_logits(self):
        import torch

        logits = torch.tensor([3.0, 1.0, -2.0], dtype=torch.float32)
        result = MODULE._distribution_fidelity(
            logits,
            logits.clone(),
            reference_top_ids=[0, 1, 2],
            candidate_top_ids=[0, 1, 2],
        )
        self.assertEqual(result["kl_final_to_readout"], 0.0)
        self.assertEqual(result["kl_readout_to_final"], 0.0)
        self.assertAlmostEqual(result["jensen_shannon_divergence"], 0.0, places=7)
        self.assertEqual(result["total_variation_distance"], 0.0)
        self.assertTrue(result["top1_matches_final"])
        self.assertEqual(result["top_k_overlap_fraction"], 1.0)

    def test_distribution_fidelity_detects_divergence_and_topk_mismatch(self):
        import torch

        reference = torch.tensor([5.0, 2.0, 0.0, -1.0], dtype=torch.float32)
        candidate = torch.tensor([-1.0, 0.0, 2.0, 5.0], dtype=torch.float32)
        result = MODULE._distribution_fidelity(
            reference,
            candidate,
            reference_top_ids=[0, 1, 2, 3],
            candidate_top_ids=[3, 2, 1, 0],
            top_k=2,
        )
        self.assertGreater(result["kl_final_to_readout"], 0.0)
        self.assertGreater(result["kl_readout_to_final"], 0.0)
        self.assertGreater(result["jensen_shannon_divergence"], 0.0)
        self.assertGreater(result["total_variation_distance"], 0.0)
        self.assertFalse(result["top1_matches_final"])
        self.assertEqual(result["top_k_overlap_fraction"], 0.0)

    def test_stream_capture_selects_each_chunk_tail(self):
        import torch

        first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        second = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100
        captured = MODULE._capture_rows(
            first, positions=(4,), stream_final_only=True
        )
        captured = MODULE._capture_rows(
            second, positions=(4,), stream_final_only=True
        )
        self.assertTrue(torch.equal(captured, second[-1:]))

    def test_default_capture_still_uses_absolute_forward_rows(self):
        import torch

        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        captured = MODULE._capture_rows(
            tensor, positions=(0, 2), stream_final_only=False
        )
        self.assertTrue(torch.equal(captured, tensor[[0, 2]]))

    def test_installed_stream_hooks_overwrite_with_latest_chunk_tail(self):
        import torch

        class Handle:
            def remove(self):
                return None

        class Hookable:
            def __init__(self):
                self.hooks = []

            def register_forward_hook(self, hook):
                self.hooks.append(hook)
                return Handle()

        class Config:
            hidden_size = 4

        class TextModel:
            def __init__(self):
                self.layers = [Hookable() for _ in range(64)]
                self.norm = Hookable()
                self.config = Config()

        class LanguageModel:
            def __init__(self):
                self.model = TextModel()
                self.lm_head = object()

        class Model:
            def __init__(self):
                self.language_model = LanguageModel()

        model = Model()
        grad_enabled = torch.is_grad_enabled()
        try:
            MODULE._install_capture_hooks(model)
            MODULE._prepare_capture(
                model, positions=(99,), stream_final_only=True
            )
            first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            second = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100
            zero_first = torch.zeros_like(first)
            zero_second = torch.zeros_like(second)
            layer_hook = model.language_model.model.layers[0].hooks[0]
            norm_hook = model.language_model.model.norm.hooks[0]
            layer_hook(None, None, (first, zero_first))
            norm_hook(None, None, first)
            layer_hook(None, None, (second, zero_second))
            norm_hook(None, None, second)
            self.assertTrue(
                torch.equal(model._jlens_captures[0], second[-1:])
            )
            self.assertTrue(
                torch.equal(model._jlens_final_normalized, second[-1:])
            )
        finally:
            torch.set_grad_enabled(grad_enabled)

    def test_stream_capture_rejects_multiple_positions(self):
        import torch

        with self.assertRaisesRegex(ValueError, "exactly one position"):
            MODULE._capture_rows(
                torch.zeros(2, 4),
                positions=(2, 3),
                stream_final_only=True,
            )

    def test_stream_mode_requires_final_requested_position_only(self):
        MODULE._require_stream_final_position([7], 8)
        with self.assertRaisesRegex(ValueError, "final prompt position 7"):
            MODULE._require_stream_final_position([3, 7], 8)
        with self.assertRaisesRegex(ValueError, "final prompt position 7"):
            MODULE._require_stream_final_position([6], 8)

    def test_prompt_file_preserves_exact_ids_target_and_metadata(self):
        payload = [
            {
                "id": "swe-task",
                "text": "display text",
                "token_ids": [3, 5, 8],
                "target_token_id": 13,
                "score_token_ids": [21, 34],
                "metadata": {"instance_id": "sympy__sympy-123"},
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(json.dumps(payload))
            args = MODULE.build_parser().parse_args(["--prompts-file", str(path)])
            self.assertEqual(MODULE._load_prompts(args), payload)

    def test_exact_prompt_ids_are_authoritative_over_text_tokenization(self):
        class Tokenizer:
            def encode(self, text, *, add_special_tokens):
                raise AssertionError("exact token IDs must bypass tokenization")

            def decode(self, token_ids, **kwargs):
                return "decoded"

        token_ids, text = MODULE._resolve_prompt_input(
            Tokenizer(), {"id": "p", "text": "original", "token_ids": [7, 9]}
        )
        self.assertEqual(token_ids, [7, 9])
        self.assertEqual(text, "original")

    def test_exact_prompt_without_text_uses_deterministic_decode(self):
        calls = []

        class Tokenizer:
            def decode(self, token_ids, **kwargs):
                calls.append((token_ids, kwargs))
                return "decoded exact IDs"

        token_ids, text = MODULE._resolve_prompt_input(
            Tokenizer(), {"id": "p", "token_ids": [7, 9]}
        )
        self.assertEqual(token_ids, [7, 9])
        self.assertEqual(text, "decoded exact IDs")
        self.assertEqual(
            calls,
            [
                (
                    [7, 9],
                    {
                        "skip_special_tokens": False,
                        "clean_up_tokenization_spaces": False,
                    },
                )
            ],
        )

    def test_prompt_file_rejects_invalid_exact_ids_and_targets(self):
        invalid_entries = (
            {"token_ids": []},
            {"token_ids": [1, True]},
            {"token_ids": [1], "target_token_id": -1},
            {"token_ids": [1], "score_token_ids": []},
            {"token_ids": [1], "score_token_ids": [2, 2]},
            {"token_ids": [1], "score_token_ids": [True]},
            {"metadata": {"missing": "input"}},
        )
        for entry in invalid_entries:
            with self.subTest(
                entry=entry
            ), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "prompts.json"
                path.write_text(json.dumps([entry]))
                args = MODULE.build_parser().parse_args(
                    ["--prompts-file", str(path)]
                )
                with self.assertRaises(ValueError):
                    MODULE._load_prompts(args)

    def test_prompt_score_ids_extend_global_ids_without_duplicates(self):
        self.assertEqual(
            MODULE._prompt_score_token_ids(
                (2, 3), {"score_token_ids": [3, 5, 8]}
            ),
            (2, 3, 5, 8),
        )
        self.assertEqual(MODULE._prompt_score_token_ids((2, 3), {}), (2, 3))

    def test_runtime_pin_defaults_disable_prefix_cache_settings(self):
        args = MODULE.build_parser().parse_args([])
        self.assertEqual(
            MODULE._runtime_pins(args),
            {
                "max_model_len": 256,
                "max_num_batched_tokens": 256,
                "mamba_block_size": None,
                "enable_prefix_caching": False,
                "kv_cache_dtype": "auto",
                "kv_offloading_size": None,
                "kv_offloading_backend": "native",
                "stream_final_only": False,
            },
        )

    def test_runtime_pins_accept_long_context_chunking_values(self):
        args = MODULE.build_parser().parse_args(
            [
                "--max-model-len",
                "32768",
                "--max-num-batched-tokens",
                "4096",
                "--mamba-block-size",
                "4096",
                "--enable-prefix-caching",
                "--kv-cache-dtype",
                "fp8",
                "--kv-offloading-size",
                "8",
                "--kv-offloading-backend",
                "native",
                "--stream-final-only",
            ]
        )
        self.assertEqual(
            MODULE._runtime_pins(args),
            {
                "max_model_len": 32768,
                "max_num_batched_tokens": 4096,
                "mamba_block_size": 4096,
                "enable_prefix_caching": True,
                "kv_cache_dtype": "fp8",
                "kv_offloading_size": 8.0,
                "kv_offloading_backend": "native",
                "stream_final_only": True,
            },
        )

    def test_mamba_block_size_requires_prefix_caching(self):
        args = MODULE.build_parser().parse_args(["--mamba-block-size", "1024"])
        with self.assertRaisesRegex(ValueError, "requires --enable-prefix-caching"):
            MODULE._runtime_pins(args)

    def test_kv_offloading_size_must_be_positive(self):
        args = MODULE.build_parser().parse_args(["--kv-offloading-size", "0"])
        with self.assertRaisesRegex(ValueError, "must be positive"):
            MODULE._runtime_pins(args)

    def test_launcher_disables_expandable_segments_for_kv_offload(self):
        launcher = (ROOT / "scripts" / "run_jlens_nvfp4.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('"$argument" == --kv-offloading-size', launcher)
        self.assertIn("unset PYTORCH_CUDA_ALLOC_CONF", launcher)

    def test_residual_capture_manifest_binds_positions_and_bytes(self):
        import torch

        captures = {
            0: torch.arange(8, dtype=torch.float32).reshape(2, 4),
            1: torch.arange(8, dtype=torch.float32).reshape(2, 4) + 10,
        }
        first = MODULE.captured_residual_manifest(
            captures, token_positions=(3, 7), layers=(0, 1)
        )
        second = MODULE.captured_residual_manifest(
            captures, token_positions=(3, 7), layers=(0, 1)
        )
        self.assertEqual(first, second)
        self.assertEqual(first["tensor_count"], 2)
        self.assertEqual(first["logical_bytes"], 64)
        changed = {key: value.clone() for key, value in captures.items()}
        changed[1][0, 0] += 1
        self.assertNotEqual(
            first["sha256"],
            MODULE.captured_residual_manifest(
                changed, token_positions=(3, 7), layers=(0, 1)
            )["sha256"],
        )
        self.assertNotEqual(
            first["sha256"],
            MODULE.captured_residual_manifest(
                captures, token_positions=(3, 6), layers=(0, 1)
            )["sha256"],
        )

    def test_model_checkpoint_is_revalidated_after_evaluation(self):
        calls = []

        class Checkpoint:
            def validate_pinned_integrity(self):
                calls.append("after")

        def factory(path, *, strict_pinned):
            calls.append((path, strict_pinned))
            return Checkpoint()

        checkpoint, record = MODULE.open_pinned_model_checkpoint(
            Path("snapshot"), checkpoint_factory=factory
        )
        self.assertEqual(calls, [(Path("snapshot"), True)])
        self.assertFalse(record["validated_after_evaluation"])
        MODULE.revalidate_pinned_model_checkpoint(checkpoint, record)
        self.assertEqual(calls[-1], "after")
        self.assertTrue(record["validated_after_evaluation"])

    def test_model_checkpoint_revalidation_failure_is_not_marked_valid(self):
        class Checkpoint:
            def validate_pinned_integrity(self):
                raise ValueError("shard SHA-256 mismatch")

        record = {"validated_after_evaluation": False}
        with self.assertRaisesRegex(ValueError, "shard SHA-256 mismatch"):
            MODULE.revalidate_pinned_model_checkpoint(Checkpoint(), record)
        self.assertFalse(record["validated_after_evaluation"])

    def test_default_prompt_matches_reference(self):
        self.assertEqual(
            MODULE.DEFAULT_PROMPT,
            "Fact: The currency used in the country shaped like a boot is",
        )

    def test_default_and_path_only_lenses_use_public_verifier(self):
        parser = MODULE.build_parser()
        self.assertEqual(
            MODULE.lens_artifact_mode(parser.parse_args([])), "public"
        )
        self.assertEqual(
            MODULE.lens_artifact_mode(
                parser.parse_args(["--lens-path", "public.pt"])
            ),
            "public",
        )

    def test_legacy_namespace_without_lens_kind_remains_auto(self):
        legacy = MODULE.argparse.Namespace(
            lens_path=Path("local.pt"),
            lens_sha256="a" * 64,
            lens_provenance=Path("local.provenance.json"),
        )
        self.assertEqual(MODULE.lens_artifact_mode(legacy), "local_fit")

    def test_local_lens_requires_path_hash_and_provenance(self):
        parser = MODULE.build_parser()
        local = parser.parse_args(
            [
                "--lens-path",
                "local.pt",
                "--lens-sha256",
                "a" * 64,
                "--lens-provenance",
                "local.pt.provenance.json",
            ]
        )
        self.assertEqual(MODULE.lens_artifact_mode(local), "local_fit")

        incomplete = (
            ["--lens-path", "local.pt", "--lens-sha256", "a" * 64],
            [
                "--lens-path",
                "local.pt",
                "--lens-provenance",
                "local.pt.provenance.json",
            ],
            [
                "--lens-sha256",
                "a" * 64,
                "--lens-provenance",
                "local.pt.provenance.json",
            ],
        )
        for arguments in incomplete:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                MODULE.lens_artifact_mode(parser.parse_args(arguments))

    def test_explicit_native_nvfp4_ste_lens_mode(self):
        parser = MODULE.build_parser()
        arguments = [
            "--lens-kind",
            "nvfp4-ste",
            "--lens-path",
            "native.pt",
            "--lens-sha256",
            "a" * 64,
            "--lens-provenance",
            "native.final.json",
            "--lens-state",
            "state.json",
            "--lens-state-sha256",
            "b" * 64,
        ]
        self.assertEqual(
            MODULE.lens_artifact_mode(parser.parse_args(arguments)),
            "native_nvfp4_ste",
        )

    def test_explicit_nf4_lens_mode_preserves_local_fit(self):
        parser = MODULE.build_parser()
        arguments = [
            "--lens-kind",
            "nf4",
            "--lens-path",
            "nf4.pt",
            "--lens-sha256",
            "b" * 64,
            "--lens-provenance",
            "nf4.provenance.json",
        ]
        self.assertEqual(
            MODULE.lens_artifact_mode(parser.parse_args(arguments)), "local_fit"
        )

    def test_explicit_local_lens_modes_require_all_artifacts(self):
        parser = MODULE.build_parser()
        for lens_kind in ("nf4", "nvfp4-ste"):
            with self.subTest(lens_kind=lens_kind), self.assertRaisesRegex(
                ValueError, "require --lens-path"
            ):
                MODULE.lens_artifact_mode(
                    parser.parse_args(
                        ["--lens-kind", lens_kind, "--lens-path", "lens.pt"]
                    )
                )

    def test_explicit_public_mode_rejects_local_metadata(self):
        parser = MODULE.build_parser()
        with self.assertRaisesRegex(ValueError, "public lenses do not accept"):
            MODULE.lens_artifact_mode(
                parser.parse_args(
                    [
                        "--lens-kind",
                        "public",
                        "--lens-path",
                        "public.pt",
                        "--lens-sha256",
                        "a" * 64,
                    ]
                )
            )


if __name__ == "__main__":
    unittest.main()
