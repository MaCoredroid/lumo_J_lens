"""Focused unit and mutation tests for paired NVFP4 report comparison."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "compare_jlens_nvfp4_reports",
    SCRIPTS / "compare_jlens_nvfp4_reports.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def readout(target: int, target_token: str, token_ids: list[int]) -> dict[str, object]:
    scores = [float(10 - index) for index in range(len(token_ids))]
    if target in token_ids:
        target_index = token_ids.index(target)
        target_rank = target_index + 1
        target_score = scores[target_index]
    else:
        target_rank = len(token_ids) + 1
        target_score = scores[-1] - 1.0
    return {
        "token_ids": token_ids,
        "tokens": [target_token if token == target else f"token-{token}" for token in token_ids],
        "scores": scores,
        "target_token_id": target,
        "target_token": target_token,
        "target_rank": target_rank,
        "target_score": target_score,
        "target_logprob": -float(target_rank),
    }


def jacobian_ids(side: str, target: int, observation: int) -> list[int]:
    candidates = [100_000 + observation * 10 + index for index in range(6)]
    candidates = [value for value in candidates if value != target]
    if observation % 2 == 0:
        if side == "native":
            return [target, candidates[0], candidates[1], candidates[2], candidates[3]]
        return [candidates[0], target, candidates[1], candidates[2], candidates[4]]
    if side == "native":
        return [candidates[0], candidates[1], target, candidates[2], candidates[3]]
    return [candidates[0], candidates[1], target, candidates[2], candidates[4]]


def make_experiment(
    side: str,
    *,
    prompt_id: str = "prompt-0",
    prompt_token_ids: list[int] | None = None,
    requested_positions: list[int] | None = None,
) -> dict[str, object]:
    prompt_token_ids = list(prompt_token_ids or [100, 101, 102])
    requested_positions = list(requested_positions or [0, 1])
    resolved_positions = [
        position + len(prompt_token_ids) if position < 0 else position
        for position in requested_positions
    ]
    final_position = len(prompt_token_ids) - 1
    capture_positions = list(resolved_positions)
    if final_position not in capture_positions:
        capture_positions.append(final_position)
    prompt_tokens = [f"token-{token}" for token in prompt_token_ids]
    generated_token_id = max(prompt_token_ids) + 1
    generated_token = f"token-{generated_token_id}"

    def target(position: int) -> tuple[int, str]:
        if position == final_position:
            return generated_token_id, generated_token
        return prompt_token_ids[position + 1], prompt_tokens[position + 1]

    positions = []
    for observation, token_position in enumerate(resolved_positions):
        target_id, target_token = target(token_position)
        baseline_ids = [
            target_id,
            200_000 + observation * 10,
            200_001 + observation * 10,
            200_002 + observation * 10,
            200_003 + observation * 10,
        ]
        positions.append(
            {
                "capture_index": capture_positions.index(token_position),
                "token_position": token_position,
                "jacobian_lens": readout(
                    target_id,
                    target_token,
                    jacobian_ids(side, target_id, observation),
                ),
                "logit_lens": readout(target_id, target_token, baseline_ids),
            }
        )

    final_readouts = []
    for observation, token_position in enumerate(capture_positions):
        target_id, target_token = target(token_position)
        final_readouts.append(
            readout(
                target_id,
                target_token,
                [
                    target_id,
                    300_000 + observation * 10,
                    300_001 + observation * 10,
                    300_002 + observation * 10,
                    300_003 + observation * 10,
                ],
            )
        )
    residual_manifest = {
        "algorithm": MODULE.RESIDUAL_CAPTURE_ALGORITHM,
        "sha256": hashlib.sha256(
            json.dumps(
                {"prompt_id": prompt_id, "positions": capture_positions},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "tensor_count": MODULE.TARGET_LAYER + 1,
        "logical_bytes": (
            (MODULE.TARGET_LAYER + 1)
            * len(capture_positions)
            * MODULE.D_MODEL
            * 4
        ),
        "token_positions": capture_positions,
    }

    return {
        "id": prompt_id,
        "prompt": f"synthetic {prompt_id}",
        "prompt_token_ids": prompt_token_ids,
        "prompt_tokens": prompt_tokens,
        "positions_requested": requested_positions,
        "positions_resolved": resolved_positions,
        "capture_positions_resolved": capture_positions,
        "final_validation_position": final_position,
        "position_tokens": [prompt_tokens[position] for position in resolved_positions],
        "generated_token_id": generated_token_id,
        "generated_token": generated_token,
        "generated_text": generated_token,
        "final_layer_top1_matches_greedy": True,
        "final_norm_reconstruction": {
            "max_abs_error": 0.1,
            "rms_error": 0.005,
            "reference_rms": 1.0,
            "relative_rms_error": 0.005,
            "max_abs_tolerance": MODULE.FINAL_NORM_MAX_ABS_TOLERANCE,
            "rms_tolerance": MODULE.FINAL_NORM_RMS_TOLERANCE,
            "within_tolerance": True,
        },
        "final_logits_reconstruction": {
            "max_abs_error": 0.07,
            "max_abs_tolerance": MODULE.FINAL_LOGIT_MAX_ABS_TOLERANCE,
            "rms_error": 0.005,
            "rms_tolerance": MODULE.FINAL_LOGIT_RMS_TOLERANCE,
            "top_k_prefix": MODULE.FINAL_TOPK_PARITY_K,
            "top_k_prefix_token_ids_match": True,
            "within_tolerance": False,
        },
        "final_model_readout": copy.deepcopy(final_readouts),
        "captured_final_model_readout": copy.deepcopy(final_readouts),
        "residual_capture_manifest": residual_manifest,
        "layers": [
            {
                "layer": layer,
                "layer_type": "full_attention" if layer % 4 == 3 else "linear_attention",
                "positions": copy.deepcopy(positions),
            }
            for layer in MODULE.SOURCE_LAYERS
        ],
    }


def native_lens() -> dict[str, object]:
    return {
        "kind": MODULE.NATIVE_LENS_KIND,
        "application": "native NVFP4/FP8 STE lens",
        "checkpoint_keys": ["J", "d_model", "n_prompts", "source_layers"],
        "contract_sha256": MODULE.PRODUCTION_CONTRACT_SHA256,
        "d_model": MODULE.D_MODEL,
        "finite_checked": True,
        "fit_estimator": MODULE.FIT_ESTIMATOR_LABEL,
        "fit_model": MODULE.MODEL_REPO,
        "fit_model_revision": MODULE.MODEL_REVISION,
        "fit_quantization": MODULE.FIT_QUANTIZATION_LABEL,
        "n_prompts": MODULE.N_PROMPTS,
        "provenance_sha256": "b" * 64,
        "provenance_size_bytes": 1234,
        "verification_scope": "exact pinned production run; not a generic portable fit",
        "state_sha256": "e" * 64,
        "state_size_bytes": 5678,
        "run_id": "synthetic-native-run",
        "surrogate_backward": (
            "identity STE; not the literal derivative of quantized rounding"
        ),
        "layer_aggregate_sha256": "c" * 64,
        "committed_prompts_sha256": "d" * 64,
        "sha256": "a" * 64,
        "size_bytes": 6_606_000_000,
        "source_layers": list(MODULE.SOURCE_LAYERS),
        "target_layer": MODULE.TARGET_LAYER,
        "tensor_dtype": "torch.float32",
        "tensor_shape": [MODULE.D_MODEL, MODULE.D_MODEL],
    }


def public_lens() -> dict[str, object]:
    return {
        "kind": MODULE.PUBLIC_LENS_KIND,
        "application": MODULE.PUBLIC_LENS_APPLICATION,
        "checkpoint_keys": ["J", "d_model", "n_prompts", "source_layers"],
        "d_model": MODULE.D_MODEL,
        "filename": MODULE.LENS_FILENAME,
        "finite_checked": False,
        "fit_time_model_precision": MODULE.PUBLIC_FIT_TIME_MODEL_PRECISION,
        "fit_time_quantization": MODULE.PUBLIC_FIT_TIME_QUANTIZATION,
        "n_prompts": MODULE.PUBLIC_N_PROMPTS,
        "repo_id": MODULE.LENS_REPO,
        "revision": MODULE.LENS_REVISION,
        "sha256": MODULE.LENS_SHA256,
        "size_bytes": MODULE.LENS_SIZE,
        "source_layers": list(MODULE.SOURCE_LAYERS),
        "tensor_dtype": "torch.float16",
        "tensor_shape": [MODULE.D_MODEL, MODULE.D_MODEL],
    }


def report(side: str) -> dict[str, object]:
    return {
        "schema_version": MODULE.INPUT_SCHEMA_VERSION,
        "score_encoding": MODULE.SCORE_ENCODING,
        "status": "failed",
        "assertions": {
            "all_final_adapter_reconstructions_within_tolerance": False,
            "all_final_layer_top1_match_greedy": True,
            "lens_hash_matches": True,
            "lens_metadata_matches": True,
            "model_architecture_matches": True,
        },
        "lens": native_lens() if side == "native" else public_lens(),
        "model": {
            "repo_id": MODULE.MODEL_REPO,
            "revision": MODULE.MODEL_REVISION,
            "config_sha256": MODULE.MODEL_CONFIG_SHA256,
            "index_sha256": MODULE.MODEL_INDEX_SHA256,
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "model_info": {
                "hidden_size": MODULE.D_MODEL,
                "layer_count": MODULE.TARGET_LAYER + 1,
                "root_class": "Qwen3_5ForConditionalGeneration",
            },
            "checkpoint_integrity": copy.deepcopy(
                MODULE.EXPECTED_CHECKPOINT_INTEGRITY
            ),
        },
        "runtime": {
            "mtp_enabled": False,
            "enforce_eager": True,
            "language_model_only": True,
            "max_model_len": 256,
            "gpu_memory_utilization": 0.82,
            "capture_adapter": "vLLM apply_model forward hooks",
            "transport_dtype": "torch.float32",
            "readout_dtype": "torch.bfloat16",
        },
        "host": {
            "platform": MODULE.EXPECTED_PLATFORM,
            "python": MODULE.EXPECTED_PYTHON,
            "gpu": {
                **MODULE.EXPECTED_GPU_IDENTITY,
                "memory_used_mib": "20000",
            },
            "packages": copy.deepcopy(MODULE.EXPECTED_PACKAGES),
        },
        "experiments": [make_experiment(side)],
    }


class PairedReportUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.native = report("native")
        self.public = report("public")

    def compare(self) -> dict[str, object]:
        return MODULE.compare_reports(self.native, self.public)

    def position(self, side: str, *, layer: int = 0, position: int = 0):
        source = self.native if side == "native" else self.public
        return source["experiments"][0]["layers"][layer]["positions"][position]

    def test_metrics_preserve_matching_documented_adapter_failure(self) -> None:
        result = self.compare()
        self.assertEqual(result["pairing"]["observation_count"], 126)
        self.assertEqual(result["pairing"]["top_k"], 5)
        self.assertTrue(result["pairing"]["logit_baseline_present"])
        self.assertEqual(result["pairing"]["score_encoding"], MODULE.SCORE_ENCODING)
        self.assertEqual(
            result["adapter_certificates"]["native"]["report_status"], "failed"
        )
        self.assertTrue(result["adapter_certificates"]["paired_diagnostics_identical"])
        comparison = result["metrics"]["overall"]["comparisons"][
            "native_vs_public_jacobian_lens"
        ]
        self.assertEqual(comparison["top1_agreement_count"], 63)
        self.assertAlmostEqual(comparison["top5_overlap_mean_fraction"], 0.8)
        self.assertIn("target_score", comparison["target_values"])
        baseline = result["metrics"]["overall"]["comparisons"][
            "native_vs_public_logit_lens"
        ]
        self.assertEqual(baseline["top1_agreement_rate"], 1.0)
        self.assertEqual(baseline["top5_exact_set_agreement_rate"], 1.0)
        self.assertEqual(
            result["pairing"]["residual_capture_manifests"]["prompt-0"],
            self.native["experiments"][0]["residual_capture_manifest"],
        )

    def test_schema3_requires_explicit_unrounded_score_encoding(self) -> None:
        self.assertEqual(MODULE.INPUT_SCHEMA_VERSION, 3)
        for side in ("native", "public"):
            self.setUp()
            with self.subTest(side=side, mutation="missing"), self.assertRaisesRegex(
                ValueError, "score_encoding"
            ):
                del getattr(self, side)["score_encoding"]
                self.compare()

            self.setUp()
            with self.subTest(side=side, mutation="rounded"), self.assertRaisesRegex(
                ValueError, "score_encoding"
            ):
                getattr(self, side)["score_encoding"] = "rounded-decimal6"
                self.compare()

        self.setUp()
        self.native["schema_version"] = 2
        self.public["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema_version must be 3"):
            self.compare()

    def test_runtime_semantics_are_pinned(self) -> None:
        mutations = (
            ("mtp_enabled", True),
            ("enforce_eager", False),
            ("language_model_only", False),
            ("capture_adapter", "arbitrary hooks"),
            ("transport_dtype", "torch.bfloat16"),
            ("readout_dtype", "torch.float32"),
        )
        for field, value in mutations:
            self.setUp()
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, f"runtime.{field}"
            ):
                self.native["runtime"][field] = value
                self.public["runtime"][field] = value
                self.compare()

    def test_runtime_capacity_and_gpu_utilization_are_validated(self) -> None:
        for report_value in (self.native, self.public):
            report_value["runtime"]["max_model_len"] = 3
        with self.assertRaisesRegex(ValueError, "no generation slot"):
            self.compare()

        self.setUp()
        for report_value in (self.native, self.public):
            report_value["runtime"]["gpu_memory_utilization"] = 0.69
        with self.assertRaisesRegex(ValueError, "0.70..0.90"):
            self.compare()

    def test_complete_long_context_runtime_identity_is_paired(self) -> None:
        long_context_runtime = {
            "max_num_batched_tokens": 4096,
            "mamba_block_size": 1024,
            "enable_prefix_caching": True,
            "kv_cache_dtype": "fp8_e4m3",
            "stream_final_only": True,
        }
        for report_value in (self.native, self.public):
            report_value["runtime"].update(long_context_runtime)

        result = self.compare()
        for field, value in long_context_runtime.items():
            self.assertEqual(result["pairing"]["runtime"][field], value)

        for field, value in (
            ("max_num_batched_tokens", 2048),
            ("mamba_block_size", 512),
            ("enable_prefix_caching", False),
            ("kv_cache_dtype", "auto"),
            ("stream_final_only", False),
        ):
            self.setUp()
            for report_value in (self.native, self.public):
                report_value["runtime"].update(long_context_runtime)
            self.public["runtime"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "paired runtime identity mismatch|enable_prefix_caching"
            ):
                self.compare()

    def test_partial_or_invalid_long_context_runtime_identity_is_rejected(self) -> None:
        for report_value in (self.native, self.public):
            report_value["runtime"]["max_num_batched_tokens"] = 4096
        with self.assertRaisesRegex(ValueError, "complete long-context runtime identity"):
            self.compare()

        long_context_runtime = {
            "max_num_batched_tokens": 4096,
            "mamba_block_size": 1024,
            "enable_prefix_caching": True,
            "kv_cache_dtype": "fp8_e4m3",
            "stream_final_only": True,
        }
        invalid_values = (
            ("max_num_batched_tokens", 0, "positive integer"),
            ("mamba_block_size", 0, "positive integer"),
            ("enable_prefix_caching", 1, "boolean"),
            ("kv_cache_dtype", "", "nonempty string"),
            ("stream_final_only", 1, "boolean"),
        )
        for field, value, message in invalid_values:
            self.setUp()
            for report_value in (self.native, self.public):
                report_value["runtime"].update(long_context_runtime)
                report_value["runtime"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.compare()

    def test_host_gpu_driver_and_packages_are_pinned(self) -> None:
        mutations = (
            ("gpu", "name", "arbitrary GPU", "host.gpu.name"),
            ("gpu", "driver_version", "0.0", "host.gpu.driver_version"),
            ("packages", "vllm", "0.0", "package identity"),
        )
        for section, field, value, message in mutations:
            self.setUp()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.native["host"][section][field] = value
                self.public["host"][section][field] = value
                self.compare()

        self.setUp()
        self.native["host"]["gpu"]["memory_used_mib"] = "999999"
        with self.assertRaisesRegex(ValueError, "memory_used_mib"):
            self.compare()

    def test_residual_capture_manifest_must_match_and_validate(self) -> None:
        self.public["experiments"][0]["residual_capture_manifest"]["sha256"] = (
            "e" * 64
        )
        with self.assertRaisesRegex(ValueError, "lens-independent diagnostic mismatch"):
            self.compare()

        mutations = (
            ("sha256", "invalid", "lowercase SHA-256"),
            ("tensor_count", 63, "cover all model residual layers"),
            ("logical_bytes", 1, "captured FP32 geometry"),
            ("token_positions", [0], "capture grid"),
            ("algorithm", "wrong", "pinned digest algorithm"),
        )
        for field, value, message in mutations:
            self.setUp()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                for report_value in (self.native, self.public):
                    report_value["experiments"][0]["residual_capture_manifest"][
                        field
                    ] = copy.deepcopy(value)
                self.compare()

    def test_rounded_tie_rank_interval_is_supported_but_bounded(self) -> None:
        target = 42
        record = readout(target, "token-42", [1, 2, 3, 4, target])
        record["scores"] = [10.0, 9.0, 8.0, 8.0, 8.0]
        record["target_score"] = 8.0
        record["target_rank"] = 4
        MODULE._validate_readout(
            record,
            "legacy-rounded",
            expected_target_id=target,
            expected_target_token="token-42",
            expected_top_k=5,
            allow_legacy_rounded_ties=True,
        )

        with self.assertRaisesRegex(ValueError, "target_rank/target_score"):
            MODULE._validate_readout(
                record,
                "unrounded-contract",
                expected_target_id=target,
                expected_target_token="token-42",
                expected_top_k=5,
            )

        record["target_rank"] = 2
        with self.assertRaisesRegex(ValueError, "target_rank/target_score"):
            MODULE._validate_readout(
                record,
                "impossible-rounded",
                expected_target_id=target,
                expected_target_token="token-42",
                expected_top_k=5,
                allow_legacy_rounded_ties=True,
            )

    def test_rounded_tie_can_exclude_target_at_top_k_boundary(self) -> None:
        record = readout(42, "token-42", [1, 2, 3, 4, 5])
        record["target_score"] = record["scores"][-1]
        record["target_rank"] = 6
        MODULE._validate_readout(
            record,
            "legacy-boundary",
            expected_target_id=42,
            expected_target_token="token-42",
            expected_top_k=5,
            allow_legacy_rounded_ties=True,
        )

    def test_final_requested_position_and_variable_length_prompts_are_supported(self) -> None:
        self.native["experiments"].append(
            make_experiment(
                "native",
                prompt_id="prompt-long",
                prompt_token_ids=[200, 201, 202, 203, 204],
                requested_positions=[1, -1],
            )
        )
        self.public["experiments"].append(
            make_experiment(
                "public",
                prompt_id="prompt-long",
                prompt_token_ids=[200, 201, 202, 203, 204],
                requested_positions=[1, -1],
            )
        )
        result = self.compare()
        self.assertEqual(
            result["pairing"]["positions_by_prompt"],
            {"prompt-0": [0, 1], "prompt-long": [1, 4]},
        )
        self.assertEqual(result["pairing"]["observation_count"], 252)

    def test_native_role_rejects_nf4_identity(self) -> None:
        self.native["lens"]["kind"] = "local_fit"
        with self.assertRaisesRegex(ValueError, "native.lens.kind"):
            self.compare()

    def test_swapped_and_same_roles_are_rejected(self) -> None:
        for native, public in (
            (self.public, self.native),
            (self.native, self.native),
            (self.public, self.public),
        ):
            with self.subTest(native=native["lens"].get("kind")), self.assertRaises(
                ValueError
            ):
                MODULE.compare_reports(native, public)

    def test_native_fit_and_public_pin_mutations_fail(self) -> None:
        mutations = (
            (self.native["lens"], "fit_quantization", "nf4"),
            (self.native["lens"], "contract_sha256", "0" * 64),
            (self.public["lens"], "sha256", "0" * 64),
            (self.public["lens"], "revision", "wrong"),
            (self.public["lens"], "application", "verified BF16 fit"),
            (self.public["lens"], "fit_time_model_precision", "bfloat16"),
            (self.public["lens"], "fit_time_quantization", "none"),
        )
        for record, field, value in mutations:
            original = record[field]
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "pinned identity"
            ):
                record[field] = value
                try:
                    self.compare()
                finally:
                    record[field] = original

    def test_native_exact_state_evidence_is_required_and_preserved(self) -> None:
        result = self.compare()
        preserved = result["inputs"]["native_lens"]
        for field in (
            "verification_scope",
            "state_sha256",
            "state_size_bytes",
            "run_id",
            "surrogate_backward",
        ):
            self.assertEqual(preserved[field], self.native["lens"][field])

        mutations = (
            ("verification_scope", "generic fit", "verification_scope"),
            ("state_sha256", "invalid", "state_sha256"),
            ("state_size_bytes", 0, "state_size_bytes"),
            ("run_id", "", "run_id"),
            ("surrogate_backward", "literal derivative", "surrogate_backward"),
        )
        for field, value, message in mutations:
            self.setUp()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.native["lens"][field] = value
                self.compare()

    def test_exact_checkpoint_integrity_is_required_and_preserved(self) -> None:
        result = self.compare()
        self.assertEqual(
            result["pairing"]["model"]["checkpoint_integrity"],
            MODULE.EXPECTED_CHECKPOINT_INTEGRITY,
        )

        shard = next(iter(MODULE.EXPECTED_CHECKPOINT_INTEGRITY["shards"]))

        def set_before_false(value):
            value["validated_before_model_load"] = False

        def set_after_false(value):
            value["validated_after_evaluation"] = False

        def mutate_metadata(value):
            value["metadata_sha256"]["config.json"] = "0" * 64

        def remove_shard(value):
            del value["shards"][shard]

        def mutate_shard_hash(value):
            value["shards"][shard]["sha256"] = "0" * 64

        def mutate_shard_size(value):
            value["shards"][shard]["bytes"] += 1

        for name, mutate in (
            ("before", set_before_false),
            ("after", set_after_false),
            ("metadata", mutate_metadata),
            ("missing-shard", remove_shard),
            ("shard-hash", mutate_shard_hash),
            ("shard-size", mutate_shard_size),
        ):
            self.setUp()
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "checkpoint_integrity"
            ):
                mutate(self.native["model"]["checkpoint_integrity"])
                self.compare()

    def test_required_artifact_and_model_assertions_must_be_true(self) -> None:
        for name in MODULE.REQUIRED_TRUE_ASSERTIONS:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.native["assertions"][name] = False
                try:
                    self.compare()
                finally:
                    self.native["assertions"][name] = True

    def test_pinned_model_identity_is_required(self) -> None:
        self.public["model"]["revision"] = "wrong"
        with self.assertRaisesRegex(ValueError, "public.model.revision"):
            self.compare()

    def test_lens_geometry_and_observed_grid_are_bound(self) -> None:
        self.native["lens"]["tensor_shape"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "tensor_shape"):
            self.compare()
        self.native["lens"]["tensor_shape"] = [MODULE.D_MODEL, MODULE.D_MODEL]

        self.native["experiments"][0]["layers"].pop()
        self.public["experiments"][0]["layers"].pop()
        with self.assertRaisesRegex(ValueError, "observed layer grid"):
            self.compare()

    def test_missing_or_changed_logit_baseline_fails(self) -> None:
        del self.position("public")["logit_lens"]
        with self.assertRaisesRegex(ValueError, "exact logit baseline is required"):
            self.compare()

        self.setUp()
        public_logit = self.position("public")["logit_lens"]
        public_logit["token_ids"][-1] += 1
        public_logit["tokens"][-1] = f"token-{public_logit['token_ids'][-1]}"
        with self.assertRaisesRegex(ValueError, "paired logit baseline differs"):
            self.compare()

    def test_lens_independent_diagnostics_must_match_exactly(self) -> None:
        self.public["experiments"][0]["final_norm_reconstruction"][
            "max_abs_error"
        ] += 0.01
        with self.assertRaisesRegex(ValueError, "lens-independent diagnostic mismatch"):
            self.compare()

    def test_adapter_assertion_and_status_must_match_diagnostics(self) -> None:
        self.native["assertions"][
            "all_final_adapter_reconstructions_within_tolerance"
        ] = True
        with self.assertRaisesRegex(ValueError, "adapter assertion"):
            self.compare()

        self.setUp()
        self.native["status"] = "passed"
        with self.assertRaisesRegex(ValueError, "status does not match"):
            self.compare()

    def test_adapter_pass_is_derived_from_errors_not_reported_booleans(self) -> None:
        for field in ("max_abs_error", "rms_error"):
            self.setUp()
            for report_value in (self.native, self.public):
                diagnostic = report_value["experiments"][0][
                    "final_norm_reconstruction"
                ]
                diagnostic[field] = 100.0
                if field == "rms_error":
                    diagnostic["relative_rms_error"] = (
                        diagnostic[field] / diagnostic["reference_rms"]
                    )
                diagnostic["within_tolerance"] = True
            with self.subTest(kind="norm", field=field), self.assertRaisesRegex(
                ValueError, "within_tolerance does not match the derived result"
            ):
                self.compare()

            self.setUp()
            for report_value in (self.native, self.public):
                diagnostic = report_value["experiments"][0][
                    "final_logits_reconstruction"
                ]
                diagnostic[field] = 100.0
                diagnostic["within_tolerance"] = True
            with self.subTest(kind="logits", field=field), self.assertRaisesRegex(
                ValueError, "within_tolerance does not match the derived result"
            ):
                self.compare()

    def test_adapter_boundaries_use_unrounded_schema3_values(self) -> None:
        for report_value in (self.native, self.public):
            experiment = report_value["experiments"][0]
            norm = experiment["final_norm_reconstruction"]
            norm["max_abs_error"] = MODULE.FINAL_NORM_MAX_ABS_TOLERANCE
            norm["rms_error"] = MODULE.FINAL_NORM_RMS_TOLERANCE
            norm["relative_rms_error"] = norm["rms_error"] / norm["reference_rms"]
            norm["within_tolerance"] = True
            logits = experiment["final_logits_reconstruction"]
            logits["max_abs_error"] = MODULE.FINAL_LOGIT_MAX_ABS_TOLERANCE
            logits["rms_error"] = MODULE.FINAL_LOGIT_RMS_TOLERANCE
            logits["within_tolerance"] = True
            report_value["assertions"][
                "all_final_adapter_reconstructions_within_tolerance"
            ] = True
            report_value["status"] = "passed"
        result = self.compare()
        self.assertEqual(
            result["adapter_certificates"]["native"]["report_status"], "passed"
        )

        boundary_mutations = (
            (
                "final_norm_reconstruction",
                "max_abs_error",
                math.nextafter(MODULE.FINAL_NORM_MAX_ABS_TOLERANCE, math.inf),
            ),
            (
                "final_norm_reconstruction",
                "rms_error",
                math.nextafter(MODULE.FINAL_NORM_RMS_TOLERANCE, math.inf),
            ),
            (
                "final_logits_reconstruction",
                "max_abs_error",
                math.nextafter(MODULE.FINAL_LOGIT_MAX_ABS_TOLERANCE, math.inf),
            ),
            (
                "final_logits_reconstruction",
                "rms_error",
                math.nextafter(MODULE.FINAL_LOGIT_RMS_TOLERANCE, math.inf),
            ),
        )
        for diagnostic_name, field, value in boundary_mutations:
            self.setUp()
            for report_value in (self.native, self.public):
                diagnostic = report_value["experiments"][0][diagnostic_name]
                if diagnostic_name == "final_logits_reconstruction":
                    diagnostic["max_abs_error"] = (
                        MODULE.FINAL_LOGIT_MAX_ABS_TOLERANCE
                    )
                    diagnostic["rms_error"] = MODULE.FINAL_LOGIT_RMS_TOLERANCE
                diagnostic[field] = value
                if (
                    diagnostic_name == "final_norm_reconstruction"
                    and field == "rms_error"
                ):
                    diagnostic["relative_rms_error"] = (
                        diagnostic[field] / diagnostic["reference_rms"]
                    )
                diagnostic["within_tolerance"] = True
            with self.subTest(
                diagnostic=diagnostic_name, field=field
            ), self.assertRaisesRegex(
                ValueError, "within_tolerance does not match the derived result"
            ):
                self.compare()

    def test_relative_rms_error_is_derived_from_unrounded_operands(self) -> None:
        for report_value in (self.native, self.public):
            diagnostic = report_value["experiments"][0][
                "final_norm_reconstruction"
            ]
            diagnostic["relative_rms_error"] = math.nextafter(
                diagnostic["rms_error"] / diagnostic["reference_rms"] + 1e-12,
                math.inf,
            )
        with self.assertRaisesRegex(ValueError, "rms_error/reference_rms"):
            self.compare()

    def test_adapter_tolerances_are_pinned(self) -> None:
        mutations = (
            ("final_norm_reconstruction", "max_abs_tolerance", 0.5),
            ("final_norm_reconstruction", "rms_tolerance", 0.5),
            ("final_logits_reconstruction", "max_abs_tolerance", 0.5),
            ("final_logits_reconstruction", "rms_tolerance", 0.5),
            ("final_logits_reconstruction", "top_k_prefix", 4),
        )
        for diagnostic_name, field, value in mutations:
            self.setUp()
            for report_value in (self.native, self.public):
                report_value["experiments"][0][diagnostic_name][field] = value
            with self.subTest(diagnostic=diagnostic_name, field=field), self.assertRaisesRegex(
                ValueError, "tolerance|top_k_prefix"
            ):
                self.compare()

    def test_top_k_prefix_flag_is_derived_from_final_readout_ids(self) -> None:
        for report_value in (self.native, self.public):
            report_value["experiments"][0]["final_logits_reconstruction"][
                "top_k_prefix_token_ids_match"
            ] = False
        with self.assertRaisesRegex(ValueError, "derived token-ID prefixes"):
            self.compare()

        self.setUp()
        for report_value in (self.native, self.public):
            readout_record = report_value["experiments"][0][
                "captured_final_model_readout"
            ][0]
            readout_record["token_ids"][1] = 999_999
            readout_record["tokens"][1] = "token-999999"
        with self.assertRaisesRegex(ValueError, "derived token-ID prefixes"):
            self.compare()

    def test_final_top1_is_derived_from_both_final_readouts(self) -> None:
        for report_value in (self.native, self.public):
            report_value["experiments"][0]["final_layer_top1_matches_greedy"] = False
        with self.assertRaisesRegex(ValueError, "derived final-position token IDs"):
            self.compare()

        self.setUp()
        for report_value in (self.native, self.public):
            experiment = report_value["experiments"][0]
            final_index = experiment["capture_positions_resolved"].index(
                experiment["final_validation_position"]
            )
            for field in ("final_model_readout", "captured_final_model_readout"):
                readout_record = experiment[field][final_index]
                readout_record["token_ids"][0], readout_record["token_ids"][1] = (
                    readout_record["token_ids"][1],
                    readout_record["token_ids"][0],
                )
                readout_record["tokens"][0], readout_record["tokens"][1] = (
                    readout_record["tokens"][1],
                    readout_record["tokens"][0],
                )
                readout_record["target_rank"] = 2
                readout_record["target_score"] = readout_record["scores"][1]
        with self.assertRaisesRegex(ValueError, "derived final-position token IDs"):
            self.compare()

    def test_rank_score_and_top_k_mutations_fail(self) -> None:
        readout_record = self.position("native")["jacobian_lens"]
        readout_record["target_rank"] += 1
        with self.assertRaisesRegex(ValueError, "target_rank/target_score"):
            self.compare()

        self.setUp()
        readout_record = self.position("native")["jacobian_lens"]
        readout_record["target_score"] -= 0.5
        with self.assertRaisesRegex(ValueError, "target_rank/target_score|target entry"):
            self.compare()

        self.setUp()
        readout_record = self.position("native")["jacobian_lens"]
        for field in ("token_ids", "tokens", "scores"):
            value = readout_record[field][-1] if field != "token_ids" else 999_999
            readout_record[field].append(value)
        with self.assertRaisesRegex(ValueError, "top-k differs"):
            self.compare()

    def test_token_id_score_order_and_target_token_mutations_fail(self) -> None:
        mutations = (
            ("token_ids", lambda value: value.__setitem__(1, value[0]), "duplicates"),
            ("token_ids", lambda value: value.__setitem__(1, -1), "nonnegative"),
            ("scores", lambda value: value.__setitem__(1, value[0] + 1), "descending"),
        )
        for field, mutate, message in mutations:
            self.setUp()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                mutate(self.position("native")["jacobian_lens"][field])
                self.compare()

        self.setUp()
        self.position("native")["jacobian_lens"]["target_token"] = "wrong"
        with self.assertRaisesRegex(ValueError, "target_token"):
            self.compare()

    def test_final_position_uses_generated_target(self) -> None:
        self.native["experiments"] = [
            make_experiment("native", requested_positions=[-1])
        ]
        self.public["experiments"] = [
            make_experiment("public", requested_positions=[-1])
        ]
        final_position = self.native["experiments"][0]["layers"][0]["positions"][0]
        final_position["jacobian_lens"]["target_token_id"] = 102
        with self.assertRaisesRegex(ValueError, "target_token_id"):
            self.compare()


class ReportFileBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.native_path = self.root / "native.json"
        self.public_path = self.root / "public.json"
        self.native_bytes = json.dumps(report("native"), sort_keys=True).encode("utf-8")
        self.public_bytes = json.dumps(report("public"), sort_keys=True).encode("utf-8")
        self.native_path.write_bytes(self.native_bytes)
        self.public_path.write_bytes(self.public_bytes)

    def test_comparison_records_hashes_of_the_parsed_bytes(self) -> None:
        original_compare = MODULE.compare_reports

        def mutate_after_read(native, public):
            self.native_path.write_text("changed after parsing", encoding="utf-8")
            return original_compare(native, public)

        with mock.patch.object(
            MODULE, "compare_reports", side_effect=mutate_after_read
        ):
            result = MODULE.compare_report_files(self.native_path, self.public_path)
        self.assertEqual(
            result["input_files"]["native"]["sha256"],
            hashlib.sha256(self.native_bytes).hexdigest(),
        )
        self.assertEqual(
            result["input_files"]["native"]["size_bytes"], len(self.native_bytes)
        )

    def test_same_path_hardlink_and_same_bytes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "different files"):
            MODULE.compare_report_files(self.native_path, self.native_path)

        hardlink = self.root / "native-hardlink.json"
        os.link(self.native_path, hardlink)
        with self.assertRaisesRegex(ValueError, "different files"):
            MODULE.compare_report_files(self.native_path, hardlink)

        same_bytes = self.root / "native-copy.json"
        same_bytes.write_bytes(self.native_bytes)
        with self.assertRaisesRegex(ValueError, "different bytes"):
            MODULE.compare_report_files(self.native_path, same_bytes)

    def test_symlink_and_nonregular_inputs_are_rejected(self) -> None:
        link = self.root / "native-link.json"
        link.symlink_to(self.native_path)
        with self.assertRaisesRegex(ValueError, "regular non-symlink"):
            MODULE.load_report(link)
        with self.assertRaisesRegex(ValueError, "regular non-symlink"):
            MODULE.load_report(self.root)

    def test_atomic_file_comparison_records_pinned_roles(self) -> None:
        result = MODULE.compare_report_files(self.native_path, self.public_path)
        self.assertEqual(
            result["inputs"]["native_lens"]["kind"], MODULE.NATIVE_LENS_KIND
        )
        self.assertEqual(
            result["inputs"]["public_lens"]["sha256"], MODULE.LENS_SHA256
        )


class CommittedReportIntegrationTest(unittest.TestCase):
    def test_existing_schema2_nf4_report_is_rejected_before_publication(self) -> None:
        native = MODULE.load_report(
            ROOT / "validation" / "jlens-nf4-on-nvfp4-2026-07-16.json"
        )
        public = MODULE.load_report(
            ROOT / "validation" / "jlens-public-on-nvfp4-heldout-2026-07-16.json"
        )
        with self.assertRaisesRegex(ValueError, "schema_version must be 3"):
            MODULE.compare_reports(native, public)


if __name__ == "__main__":
    unittest.main()
