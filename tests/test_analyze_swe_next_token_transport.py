#!/usr/bin/env python3
"""Focused tests for the frozen greedy-next-token transport supplement."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_swe_next_token_transport",
    ROOT / "scripts" / "analyze_swe_next_token_transport.py",
)
assert SPEC and SPEC.loader
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)
PROTOCOL_BYTES = (ROOT / "configs" / "swe_next_token_transport_protocol.json").read_bytes()
PROTOCOL = ANALYZE.validate_protocol(
    json.loads(PROTOCOL_BYTES), protocol_sha256=ANALYZE.PROTOCOL_SHA256
)
COHORT_SHA = PROTOCOL["scope"]["cohort_manifest_sha256"]
VOCABULARY_SIZE = PROTOCOL["readout"]["scored_vocabulary_size"]


def method_layers(rank: int, target: int = 7) -> list[dict]:
    return [
        {
            "layer": layer,
            "target_token_id": target,
            "target_rank": rank,
            "target_logprob": -float(rank) / VOCABULARY_SIZE,
            "record_sha256": f"{layer:064x}",
        }
        for layer in range(24, 48)
    ]


def paired_rows(
    *,
    public_rank: int = 1,
    ordinary_rank: int = VOCABULARY_SIZE,
    nf4_rank: int = 1,
    native_rank: int = VOCABULARY_SIZE,
) -> list[dict]:
    result = []
    index = 0
    for task_index in range(20):
        for ordinal in range(8):
            result.append(
                {
                    "checkpoint_index": index,
                    "checkpoint_ordinal": ordinal,
                    "id": f"prompt-{task_index:02d}-{ordinal}",
                    "task_id": f"project__task-{task_index:02d}",
                    "repo": f"owner/repo-{task_index % 11:02d}",
                    "cohort_id": "development" if task_index < 10 else "replication",
                    "generated_token_id": 7,
                    "prompt_identity_sha256": "a" * 64,
                    "metadata_identity_sha256": "b" * 64,
                    "strict_eligible": True,
                    "sensitivity_eligible": True,
                    "report_diagnostics": {},
                    "methods": {
                        "ordinary_logit": method_layers(ordinary_rank),
                        "public_jacobian": method_layers(public_rank),
                        "nf4_jacobian": method_layers(nf4_rank),
                        "native_jacobian": method_layers(native_rank),
                    },
                }
            )
            index += 1
    return result


def compact_reports(rows: list[dict]) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for report_label in ANALYZE.REPORT_LABELS:
        report_rows = []
        method = ANALYZE.JACOBIAN_METHOD[report_label]
        for row in rows:
            layers = {}
            for index, layer in enumerate(range(24, 48)):
                layers[str(layer)] = {
                    "ordinary_logit": copy.deepcopy(row["methods"]["ordinary_logit"][index]),
                    "jacobian": copy.deepcopy(row["methods"][method][index]),
                }
            report_rows.append(
                {
                    "id": row["id"],
                    "task_id": row["task_id"],
                    "repo": row["repo"],
                    "cohort_id": row["cohort_id"],
                    "cohort_manifest_sha256": COHORT_SHA,
                    "metadata_identity_sha256": "0" * 64,
                    "generated_token_id": row["generated_token_id"],
                    "prompt_identity_sha256": "1" * 64,
                    "residual_identity_sha256": "2" * 64,
                    "diagnostics": {
                        "strict_certified": row["strict_eligible"],
                        "sensitivity_certified": row["sensitivity_eligible"],
                    },
                    "layers": layers,
                }
            )
        reports[report_label] = {
            "model_identity": {"model": "same"},
            "runtime_identity": {"runtime": "same"},
            "rows": report_rows,
        }
    return reports


def compact_readout(target: int = 7, rank: int = 1) -> dict:
    return {
        "token_ids": [7, 8, 9, 10, 11],
        "target_token_id": target,
        "target_rank": rank,
        "target_logprob": -0.1,
    }


def raw_experiment() -> dict:
    return {
        "id": "prompt-00-0",
        "prompt": "short prompt",
        "prompt_token_ids": [1, 2],
        "positions_requested": [-1],
        "positions_resolved": [1],
        "capture_positions_resolved": [1],
        "final_validation_position": 1,
        "generated_token_id": 7,
        "final_layer_top1_matches_greedy": True,
        "final_model_readout": [compact_readout()],
        "captured_final_model_readout": [compact_readout()],
        "final_norm_reconstruction": {
            "max_abs_error": 0.01,
            "rms_error": 0.001,
            "max_abs_tolerance": 0.125,
            "rms_tolerance": 0.006,
            "within_tolerance": True,
        },
        "final_logits_reconstruction": {
            "max_abs_error": 0.01,
            "rms_error": 0.001,
            "max_abs_tolerance": 0.0625,
            "rms_tolerance": 0.01,
            "top_k_prefix": 5,
            "top_k_prefix_token_ids_match": True,
            "within_tolerance": True,
        },
        "residual_capture_manifest": {
            "sha256": "3" * 64,
            "tensor_count": 64,
            "token_positions": [1],
        },
        "metadata": {
            "task": {"instance_id": "project__task-00", "repo": "owner/repo-00"},
            "cohort": {"id": "development", "cohort_manifest_sha256": COHORT_SHA},
        },
        "layers": [
            {
                "layer": layer,
                "positions": [
                    {
                        "capture_index": 0,
                        "token_position": 1,
                        "logit_lens": compact_readout(rank=20),
                        "jacobian_lens": compact_readout(rank=2),
                    }
                ],
            }
            for layer in range(24, 48)
        ],
    }


class TransportAnalysisTests(unittest.TestCase):
    def test_strict_track_detects_public_control_and_native_deficit(self) -> None:
        result = ANALYZE.build_track(
            paired_rows(), PROTOCOL, sensitivity=False, bootstrap_samples=200
        )
        self.assertTrue(result["support"]["all_gates_pass"])
        self.assertEqual(
            result["classification"]["classification"],
            "native_refit_capacity_candidate",
        )
        comparison = result["comparisons"]["public_jacobian_minus_native_jacobian"]
        self.assertAlmostEqual(comparison["estimate"], 1.0)
        self.assertGreater(comparison["bootstrap"]["confidence_interval"]["lower"], 0)
        emergence = result["descriptive_transport_emergence"]
        self.assertEqual(len(emergence["by_fixed_layer"]), 24)
        self.assertEqual(len(emergence["by_checkpoint_ordinal"]), 8)
        self.assertFalse(emergence["used_by_classification"])

    def test_sensitivity_prefix_and_no_material_native_deficit(self) -> None:
        result = ANALYZE.build_track(
            paired_rows(native_rank=1),
            PROTOCOL,
            sensitivity=True,
            bootstrap_samples=100,
        )
        self.assertEqual(
            result["classification"]["classification"],
            "sensitivity_no_material_native_deficit",
        )
        self.assertIn("post_public_numerical_diagnostic", result["role"])

    def test_sensitivity_readout_control_failure_and_seeded_interval(self) -> None:
        result = ANALYZE.build_track(
            paired_rows(
                public_rank=VOCABULARY_SIZE,
                ordinary_rank=1,
                native_rank=VOCABULARY_SIZE,
            ),
            PROTOCOL,
            sensitivity=True,
            bootstrap_samples=200,
        )
        self.assertEqual(
            result["classification"]["classification"],
            "sensitivity_readout_control_failure",
        )
        self.assertEqual(
            result["classification"]["reason_codes"],
            ["public_minus_logit_ci_upper_nonpositive"],
        )
        comparison = result["comparisons"]["public_jacobian_minus_ordinary_logit"]
        self.assertAlmostEqual(comparison["estimate"], -1.0)
        self.assertEqual(
            comparison["bootstrap"]["confidence_interval"],
            {"lower": -1.0, "upper": -1.0},
        )

    def test_numerical_exclusions_fail_support_instead_of_becoming_samples(self) -> None:
        rows = paired_rows()
        for row in rows[:40]:
            row["strict_eligible"] = False
        result = ANALYZE.build_track(
            rows, PROTOCOL, sensitivity=False, bootstrap_samples=50
        )
        self.assertFalse(result["support"]["all_gates_pass"])
        self.assertEqual(
            result["classification"]["classification"], "insufficient_support"
        )
        self.assertEqual(
            result["nf4_diagnostic"]["classification"],
            "insufficient_support_not_interpretable",
        )

    def test_pairing_fails_closed_on_residual_and_ordinary_logit_changes(self) -> None:
        rows = paired_rows()
        reports = compact_reports(rows)
        reports["native"]["rows"][0]["residual_identity_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "residual_identity_sha256"):
            ANALYZE.pair_reports(reports, PROTOCOL)

        reports = compact_reports(rows)
        reports["nf4"]["rows"][0]["layers"]["24"]["ordinary_logit"][
            "target_rank"
        ] += 1
        with self.assertRaisesRegex(ValueError, "ordinary-logit pairing"):
            ANALYZE.pair_reports(reports, PROTOCOL)

        reports = compact_reports(rows)
        reports["native"]["rows"][0]["generated_token_id"] = 8
        with self.assertRaisesRegex(ValueError, "generated_token_id"):
            ANALYZE.pair_reports(reports, PROTOCOL)

        reports = compact_reports(rows)
        reports["native"]["rows"][0]["layers"]["24"]["jacobian"][
            "target_token_id"
        ] = 8
        with self.assertRaisesRegex(ValueError, "target identity failed"):
            ANALYZE.pair_reports(reports, PROTOCOL)

    def test_streaming_report_reduction_and_target_identity(self) -> None:
        report = {
            "schema_version": 3,
            "score_encoding": "unrounded-float32",
            "status": "passed",
            "experiments": [raw_experiment()],
            "model": {
                **ANALYZE.MODEL_PIN,
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
            },
            "runtime": {
                "mtp_enabled": False,
                "enforce_eager": True,
                "language_model_only": True,
                "stream_final_only": True,
                "transport_dtype": "torch.float32",
                "readout_dtype": "torch.bfloat16",
                "model_load_seconds": 1.0,
            },
            "assertions": {
                "lens_hash_matches": True,
                "lens_metadata_matches": True,
                "model_architecture_matches": True,
            },
            "lens": {
                **ANALYZE.LENS_PINS["public"],
                "d_model": 5120,
                "source_layers": list(range(63)),
                "tensor_shape": [5120, 5120],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            compact = ANALYZE.load_compact_report(path, label="public")
            self.assertEqual(len(compact["rows"]), 1)
            self.assertTrue(compact["rows"][0]["diagnostics"]["strict_certified"])

            report["experiments"][0]["final_logits_reconstruction"].update(
                {"max_abs_error": 0.125, "within_tolerance": False}
            )
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            compact = ANALYZE.load_compact_report(path, label="public")
            diagnostics = compact["rows"][0]["diagnostics"]
            self.assertFalse(diagnostics["strict_certified"])
            self.assertTrue(diagnostics["sensitivity_certified"])

            report["experiments"][0]["layers"][0]["positions"][0][
                "jacobian_lens"
            ]["target_token_id"] = 99
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "target differs from generated token"):
                ANALYZE.load_compact_report(path, label="public")

            report = {
                **report,
                "experiments": [raw_experiment()],
                "lens": {**report["lens"], "sha256": "f" * 64},
            }
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "lens pin changed"):
                ANALYZE.load_compact_report(path, label="public")

            report["lens"] = {
                **report["lens"],
                "sha256": ANALYZE.LENS_PINS["public"]["sha256"],
            }
            report["experiments"][0]["generated_token_id"] = VOCABULARY_SIZE
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "generated token ID exceeds vocabulary"):
                ANALYZE.load_compact_report(path, label="public")

    def test_frozen_prompt_scope_is_checked_row_for_row(self) -> None:
        rows = paired_rows()
        expected = [
            {
                key: row[key]
                for key in (
                    "id",
                    "task_id",
                    "repo",
                    "cohort_id",
                    "prompt_identity_sha256",
                    "metadata_identity_sha256",
                )
            }
            for row in rows
        ]
        ANALYZE.validate_prompt_scope(rows, expected)
        expected[0]["prompt_identity_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "frozen prompt field"):
            ANALYZE.validate_prompt_scope(rows, expected)


if __name__ == "__main__":
    unittest.main()
