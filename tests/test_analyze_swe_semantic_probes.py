#!/usr/bin/env python3
"""Focused tests for semantic SWE contrast analysis."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_swe_semantic_probes",
    ROOT / "scripts" / "analyze_swe_semantic_probes.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CONFIG_PATH = ROOT / "configs/swe_semantic_probes.json"
CONFIG_RAW = CONFIG_PATH.read_bytes()
CONFIG = json.loads(CONFIG_RAW)
CONFIG_SHA = hashlib.sha256(CONFIG_RAW).hexdigest()


def vocabulary() -> dict[int, str]:
    return {
        record["token_id"]: record["text"]
        for probe in CONFIG["probes"]
        for group in ("positive", "negative")
        for record in probe[group]
    }


def scored(probe: dict[str, object], positive_score: float) -> list[dict[str, object]]:
    positive = {record["token_id"] for record in probe["positive"]}
    values = vocabulary()
    return [
        {
            "token_id": token_id,
            "token": values[token_id],
            "score": positive_score if token_id in positive else 0.0,
            "logprob": -10.0,
        }
        for token_id in sorted(values)
    ]


def readout(probe: dict[str, object], positive_score: float) -> dict[str, object]:
    return {"scored_tokens": scored(probe, positive_score)}


def make_report() -> dict[str, object]:
    values = vocabulary()
    experiments = []
    for index, probe in enumerate(CONFIG["probes"]):
        experiments.append(
            {
                "id": f"swe-semantic-{probe['id']}",
                "prompt_token_ids": [10, 20, 30 + index],
                "metadata": {
                    "request_index": probe["request_index"],
                    "trajectory": {"offset": probe["offset"]},
                    "semantic_probe": {
                        "id": probe["id"],
                        "positive_token_ids": [item["token_id"] for item in probe["positive"]],
                        "negative_token_ids": [item["token_id"] for item in probe["negative"]],
                        "config_sha256": CONFIG_SHA,
                        "trajectory_bundle_sha256": "a" * 64,
                    },
                },
                "layers": [
                    {
                        "layer": layer,
                        "positions": [
                            {
                                "jacobian_lens": readout(probe, 1.8),
                                "logit_lens": readout(probe, -0.5),
                            }
                        ],
                    }
                    for layer in MODULE.PRIMARY_LAYERS
                ],
                "captured_final_model_readout": [readout(probe, 2.0)],
                "residual_capture_manifest": {"sha256": f"{index:064x}"},
            }
        )
    return {
        "schema_version": 3,
        "model": {"repo_id": MODULE.MODEL_REPO, "revision": MODULE.MODEL_REVISION},
        "scored_vocabulary": {
            "token_ids": sorted(values),
            "tokens": [values[token] for token in sorted(values)],
        },
        "experiments": experiments,
    }


class AnalyzeSweSemanticProbesTest(unittest.TestCase):
    def test_gate_passes_for_positive_closer_jacobian_margins(self) -> None:
        result = MODULE.analyze(
            CONFIG,
            make_report(),
            config_sha256=CONFIG_SHA,
            native_report=make_report(),
        )
        primary = result["primary"]
        self.assertEqual(primary["observation_count"], 10)
        self.assertEqual(primary["jacobian_positive_count"], 10)
        self.assertEqual(primary["logit_positive_count"], 0)
        self.assertEqual(primary["jacobian_closer_count"], 10)
        self.assertGreater(primary["mean_absolute_error_reduction"], 0.25)
        self.assertTrue(primary["final_margin_calibration_diagnostic"]["passed"])
        self.assertEqual(result["paired_sign_agreement"]["jacobian_margin_sign_agreement_count"], 10)
        self.assertTrue(result["paired_calibration_diagnostic"]["passed"])
        self.assertIn("EXPLORATORY", result["label"])

    def test_single_report_leaves_pairing_gate_unevaluated(self) -> None:
        result = MODULE.analyze(CONFIG, make_report(), config_sha256=CONFIG_SHA)
        self.assertIsNone(result["paired_sign_agreement"]["passed"])
        self.assertFalse(result["paired_calibration_diagnostic"]["passed"])

    def test_rejects_config_hash_mismatch(self) -> None:
        value = make_report()
        value["experiments"][0]["metadata"]["semantic_probe"]["config_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "config hash mismatch"):
            MODULE.analyze(CONFIG, value, config_sha256=CONFIG_SHA)

    def test_rejects_request_offset_mismatch(self) -> None:
        value = make_report()
        value["experiments"][0]["metadata"]["trajectory"]["offset"] += 1
        with self.assertRaisesRegex(ValueError, "request/offset mismatch"):
            MODULE.analyze(CONFIG, value, config_sha256=CONFIG_SHA)

    def test_rejects_incomplete_vocabulary_scores(self) -> None:
        value = make_report()
        value["experiments"][0]["layers"][0]["positions"][0]["jacobian_lens"]["scored_tokens"].pop()
        with self.assertRaisesRegex(ValueError, "configured vocabulary"):
            MODULE.analyze(CONFIG, value, config_sha256=CONFIG_SHA)

    def test_rejects_paired_residual_mismatch(self) -> None:
        primary = make_report()
        native = copy.deepcopy(primary)
        native["experiments"][2]["residual_capture_manifest"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "residual_capture_manifest"):
            MODULE.analyze(
                CONFIG,
                primary,
                config_sha256=CONFIG_SHA,
                native_report=native,
            )


if __name__ == "__main__":
    unittest.main()
