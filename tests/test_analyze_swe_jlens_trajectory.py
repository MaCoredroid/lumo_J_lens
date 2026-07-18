#!/usr/bin/env python3
"""Focused tests for dense SWE J-lens trajectory analysis."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_swe_jlens_trajectory",
    ROOT / "scripts" / "analyze_swe_jlens_trajectory.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fidelity(kl: float, overlap: float) -> dict[str, object]:
    count = round(overlap * 5)
    return {
        "reference": "captured_block_63_final_model",
        "kl_final_to_readout": kl,
        "kl_readout_to_final": kl,
        "jensen_shannon_divergence": kl / 4,
        "total_variation_distance": min(1.0, kl / 2),
        "top1_matches_final": True,
        "top_k": 5,
        "top_k_overlap_count": count,
        "top_k_overlap_fraction": count / 5,
    }


def readout(target: int, *, kl: float, logprob: float, rank: int, overlap: float) -> dict[str, object]:
    return {
        "token_ids": [target, 1, 2, 3, 4],
        "tokens": ["target", "a", "b", "c", "d"],
        "scores": [5.0, 4.0, 3.0, 2.0, 1.0],
        "target_token_id": target,
        "target_token": "target",
        "target_score": 5.0,
        "target_logprob": logprob,
        "target_rank": rank,
        "final_distribution_fidelity": fidelity(kl, overlap),
    }


def experiment(request: int, *, jacobian_kl: float = 0.8, logit_kl: float = 1.0) -> dict[str, object]:
    identifier = f"trajectory-request-{request:02d}-offset-0000"
    target = 100 + request
    prompt_ids = [10, 20, 30 + request]
    layers = []
    for layer in MODULE.FIXED_MIDDLE_LAYERS:
        layers.append(
            {
                "layer": layer,
                "layer_type": "linear_attention",
                "positions": [
                    {
                        "capture_index": 0,
                        "token_position": 2,
                        "jacobian_lens": readout(
                            target, kl=jacobian_kl, logprob=-1.0, rank=2, overlap=0.8
                        ),
                        "logit_lens": readout(
                            target, kl=logit_kl, logprob=-1.05, rank=3, overlap=0.6
                        ),
                    }
                ],
            }
        )
    final = {
        "token_ids": [target, 1, 2, 3, 4],
        "target_token_id": target,
        "target_token": "target",
        "target_logprob": -0.8,
        "target_rank": 1,
    }
    return {
        "id": identifier,
        "prompt": "prompt",
        "prompt_token_ids": prompt_ids,
        "prompt_tokens": ["x", "y", "z"],
        "positions_requested": [-1],
        "positions_resolved": [2],
        "capture_positions_resolved": [2],
        "final_validation_position": 2,
        "position_tokens": ["z"],
        "target_token_id_override": target,
        "generated_token_id": target,
        "generated_token": "target",
        "generated_text": "target",
        "metadata": {
            "trajectory": {
                "request_index": request,
                "offset": 0,
                "region": "sampled_reasoning",
                "events": [],
                "target_token_id": target,
            }
        },
        "layers": layers,
        "final_model_readout": [copy.deepcopy(final)],
        "captured_final_model_readout": [copy.deepcopy(final)],
        "residual_capture_manifest": {"sha256": f"{request:064x}"},
        "final_layer_top1_matches_greedy": True,
        "final_norm_reconstruction": {
            "max_abs_error": 0.0625,
            "max_abs_tolerance": 0.125,
            "rms_error": 0.005,
            "rms_tolerance": 0.006,
            "within_tolerance": True,
        },
        "final_logits_reconstruction": {
            "max_abs_error": 0.0625,
            "max_abs_tolerance": 0.0625,
            "rms_error": 0.007,
            "rms_tolerance": 0.01,
            "top_k_prefix": 5,
            "top_k_prefix_token_ids_match": True,
            "within_tolerance": True,
        },
    }


def report() -> dict[str, object]:
    return {
        "schema_version": 3,
        "score_encoding": "unrounded-float32",
        "status": "passed",
        "model": {
            "repo_id": MODULE.MODEL_REPO,
            "revision": MODULE.MODEL_REVISION,
            "config_sha256": MODULE.MODEL_CONFIG_SHA256,
            "index_sha256": MODULE.MODEL_INDEX_SHA256,
        },
        "lens": {"n_prompts": 1000},
        "runtime": {"mtp_enabled": False, "stream_final_only": True},
        "assertions": {
            "all_final_layer_top1_match_greedy": True,
            "all_final_adapter_reconstructions_within_tolerance": True,
        },
        "experiments": [experiment(request) for request in range(1, 10)],
    }


class AnalyzeSweJlensTrajectoryTest(unittest.TestCase):
    def test_primary_gate_uses_equal_request_macro_and_deterministic_statistics(self) -> None:
        first = MODULE.analyze(report(), seed=7, samples=200)
        second = MODULE.analyze(report(), seed=7, samples=200)
        self.assertEqual(first, second)
        primary = first["primary"]
        macro = primary["fixed_middle_primary"]["macro_equal_request"]
        self.assertAlmostEqual(macro["relative_kl_reduction"], 0.2)
        self.assertEqual(
            macro["kl_gain_exact_one_sided_sign_flip_p"], 1 / 512
        )
        self.assertTrue(primary["predictive_improvement_diagnostic"]["passed"])
        self.assertTrue(primary["next_token_noninferiority_diagnostic"]["passed"])
        self.assertTrue(primary["predictive_diagnostic_passed"])
        self.assertIn("CALIBRATION DIAGNOSTIC", first["label"])

    def test_rejects_unpinned_model(self) -> None:
        value = report()
        value["model"]["revision"] = "wrong"
        with self.assertRaisesRegex(ValueError, "pinned model mismatch"):
            MODULE.analyze(value, samples=20)

    def test_rejects_unsorted_ids(self) -> None:
        value = report()
        value["experiments"][0], value["experiments"][1] = (
            value["experiments"][1],
            value["experiments"][0],
        )
        with self.assertRaisesRegex(ValueError, "IDs must be unique and sorted"):
            MODULE.analyze(value, samples=20)

    def test_rejects_target_override_disagreement(self) -> None:
        value = report()
        value["experiments"][0]["target_token_id_override"] += 1
        with self.assertRaisesRegex(ValueError, "target override mismatch"):
            MODULE.analyze(value, samples=20)

    def test_rejects_adapter_status_inconsistency(self) -> None:
        value = report()
        value["experiments"][0]["final_logits_reconstruction"]["within_tolerance"] = False
        with self.assertRaisesRegex(ValueError, "logits status mismatch"):
            MODULE.analyze(value, samples=20)

    def test_optional_native_requires_exact_residual_pairing(self) -> None:
        primary = report()
        native = copy.deepcopy(primary)
        native["experiments"][3]["residual_capture_manifest"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "residual_capture_manifest|manifest"):
            MODULE.analyze(primary, native, samples=20)

    def test_optional_native_reports_gain_delta(self) -> None:
        primary = report()
        native = report()
        for item in native["experiments"]:
            for layer in item["layers"]:
                layer["positions"][0]["jacobian_lens"]["final_distribution_fidelity"][
                    "kl_final_to_readout"
                ] = 0.7
        result = MODULE.analyze(primary, native, seed=9, samples=100)
        self.assertAlmostEqual(
            result["native_comparison"]["native_minus_primary_kl_gain_nats"],
            0.1,
        )
        self.assertEqual(result["native_comparison"]["positive_request_count"], 9)


if __name__ == "__main__":
    unittest.main()
