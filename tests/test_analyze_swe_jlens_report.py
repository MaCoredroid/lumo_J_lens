#!/usr/bin/env python3
"""Tests for the compact certified-SWE Jacobian-Lens analyzer."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_swe_jlens_report",
    ROOT / "scripts" / "analyze_swe_jlens_report.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AnalyzeSweJlensReportTest(unittest.TestCase):
    def test_spearman_supports_ties(self) -> None:
        self.assertAlmostEqual(MODULE.spearman([1, 2, 2, 4], [10, 20, 20, 40]), 1.0)
        self.assertAlmostEqual(MODULE.spearman([1, 2, 3], [3, 2, 1]), -1.0)

    def test_spearman_rejects_constant_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "undefined Spearman"):
            MODULE.spearman([1, 1, 1], [1, 2, 3])

    def test_adapter_summary_preserves_partial_failure(self) -> None:
        def experiment(identifier: str, *, norm: bool, logits: bool) -> dict[str, object]:
            return {
                "id": identifier,
                "final_norm_reconstruction": {"within_tolerance": norm},
                "final_logits_reconstruction": {
                    "within_tolerance": logits,
                    "top_k_prefix_token_ids_match": True,
                },
                "final_layer_top1_matches_greedy": True,
            }

        report = {
            "status": "failed",
            "experiments": [
                experiment("ok", norm=True, logits=True),
                experiment("bad", norm=True, logits=False),
            ],
        }
        summary = MODULE.adapter_summary(report)
        self.assertEqual(summary["combined_strict_pass_count"], 1)
        self.assertEqual(summary["final_norm_pass_count"], 2)
        self.assertEqual(summary["final_logits_pass_count"], 1)
        self.assertEqual(summary["failed_experiment_ids"], ["bad"])

    def test_margin_summary_uses_only_fixed_reporting_layers(self) -> None:
        rows = []
        for layer in range(63):
            rows.append(
                {
                    "layer": layer,
                    "native_jacobian": {
                        "correct_minus_buggy_logprob": float(layer),
                        "correct_rank": 1,
                        "buggy_rank": 2,
                    },
                }
            )
        summary = MODULE.margin_summary(rows, "native_jacobian")
        expected = sum(MODULE.FIXED_MIDDLE_LAYERS) / len(MODULE.FIXED_MIDDLE_LAYERS)
        self.assertEqual(summary["layers"], list(MODULE.FIXED_MIDDLE_LAYERS))
        self.assertAlmostEqual(summary["mean_correct_minus_buggy_logprob"], expected)
        self.assertEqual(summary["positive_layer_count"], 9)

    def test_average_ranks_assigns_one_based_midranks(self) -> None:
        self.assertEqual(MODULE.average_ranks([20, 10, 10, 40]), [3.0, 1.5, 1.5, 4.0])

    def test_materializer_hash_is_sorted_pretty_json_with_newline(self) -> None:
        value = [{"z": 1, "a": "x"}]
        expected = MODULE.sha256_bytes(b'[\n  {\n    "a": "x",\n    "z": 1\n  }\n]\n')
        self.assertEqual(MODULE.materializer_json_sha256(value), expected)

    def test_candidate_pair_rejects_model_runtime_and_lens_role_mutations(self) -> None:
        native_path = (
            ROOT
            / "validation"
            / "jlens-swe-qwen-code-candidate-probe-2026-07-17.json"
        )
        public_path = (
            ROOT
            / "validation"
            / "jlens-swe-qwen-code-candidate-probe-public-2026-07-17.json"
        )
        native = json.loads(native_path.read_text(encoding="utf-8"))
        public = json.loads(public_path.read_text(encoding="utf-8"))

        mutations = (
            (
                "model",
                lambda left, right: left["model"].__setitem__("repo_id", "wrong"),
                "native.model.repo_id",
            ),
            (
                "runtime",
                lambda left, right: right["runtime"].__setitem__("mtp_enabled", True),
                "public.runtime.mtp_enabled",
            ),
            (
                "lens-role",
                lambda left, right: right.__setitem__("lens", copy.deepcopy(left["lens"])),
                "public.lens",
            ),
        )
        for name, mutate, message in mutations:
            candidate_native = copy.copy(native)
            candidate_public = copy.copy(public)
            candidate_native["model"] = copy.deepcopy(native["model"])
            candidate_native["runtime"] = copy.deepcopy(native["runtime"])
            candidate_native["lens"] = copy.deepcopy(native["lens"])
            candidate_public["model"] = copy.deepcopy(public["model"])
            candidate_public["runtime"] = copy.deepcopy(public["runtime"])
            candidate_public["lens"] = copy.deepcopy(public["lens"])
            mutate(candidate_native, candidate_public)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                MODULE.validate_candidate_pair(candidate_native, candidate_public)

    def test_preflight_binds_exact_request_and_runtime_identity(self) -> None:
        public_stage = json.loads(
            (
                ROOT
                / "validation"
                / "jlens-swe-qwen-code-public-2026-07-17.json"
            ).read_text(encoding="utf-8")
        )
        preflight = json.loads(
            (
                ROOT
                / "validation"
                / "jlens-swe-qwen-code-longest-preflight-2026-07-17.json"
            ).read_text(encoding="utf-8")
        )
        stage_identity = MODULE.validate_paired_report_identities(
            json.loads(
                (
                    ROOT
                    / "validation"
                    / "jlens-swe-qwen-code-native-nvfp4-ste-2026-07-17.json"
                ).read_text(encoding="utf-8")
            ),
            public_stage,
        )
        expected_stage = public_stage["experiments"][-1]
        result = MODULE.validate_preflight(
            preflight, expected_stage, public_stage, stage_identity
        )
        self.assertTrue(result["exact_stage_context_fields_match"])
        self.assertFalse(result["capture_manifest_matches_all_layer_stage"])

        changed = copy.copy(preflight)
        changed["runtime"] = copy.deepcopy(preflight["runtime"])
        changed["runtime"]["max_num_batched_tokens"] = 2048
        with self.assertRaisesRegex(ValueError, "runtime identity mismatch"):
            MODULE.validate_preflight(
                changed, expected_stage, public_stage, stage_identity
            )

        changed = copy.copy(preflight)
        changed["experiments"] = copy.deepcopy(preflight["experiments"])
        changed["experiments"][0]["prompt_token_ids"][-1] += 1
        with self.assertRaisesRegex(ValueError, "prompt_token_ids"):
            MODULE.validate_preflight(
                changed, expected_stage, public_stage, stage_identity
            )


if __name__ == "__main__":
    unittest.main()
