#!/usr/bin/env python3

import importlib.util
import sys
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
