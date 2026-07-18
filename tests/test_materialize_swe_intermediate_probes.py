#!/usr/bin/env python3
"""Tests for the frozen SWE intermediate-concept probe bundle."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_swe_intermediate_probes",
    ROOT / "scripts" / "materialize_swe_intermediate_probes.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/"
    "snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404"
)
HAS_PINNED_INTEGRATION = (
    importlib.util.find_spec("transformers") is not None
    and SNAPSHOT.exists()
    and MODULE.DEFAULT_CONFIG.exists()
    and MODULE.DEFAULT_TRAJECTORY.exists()
)


class PinTokenizer:
    def __init__(self, pins: dict[str, int], *, drift: str | None = None) -> None:
        self.pins = pins
        self.reverse = {token_id: text for text, token_id in pins.items()}
        self.drift = drift

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("probe token pins must disable special tokens")
        if text == self.drift:
            return [999999]
        return [self.pins[text]]

    def decode(self, token_ids: list[int], **_: object) -> str:
        return self.reverse.get(token_ids[0], "not-the-pinned-surface")


def intermediate(key: str, ordinal: int) -> dict[str, object]:
    return {
        "key": key,
        "forms": [{"text": f" concept{ordinal}", "token_id": 1000 + ordinal}],
    }


def item(ordinal: int) -> dict[str, object]:
    key = f"concept_{ordinal}"
    return {
        "id": f"item-{ordinal}",
        "event_family": "test",
        "request_index": ordinal,
        "offset": 0,
        "state": f"state-{ordinal}",
        "rationale": f"rationale-{ordinal}",
        "leakage_class": "tool_outcome_explicit",
        "request_sha256": str(ordinal) * 64,
        "evidence": [
            {
                "kind": "tool_result",
                "content_sha256": "e" * 64,
                "supports": [key],
            }
        ],
        "intermediates": [intermediate(key, ordinal)],
    }


def config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "swe_verified_intermediate_concept_eval",
        "adaptation": {
            "status": "exploratory_one_task_adaptation",
            "lens_outputs_used_for_selection": False,
        },
        "model": {
            "repo_id": MODULE.MODEL_REPO,
            "revision": MODULE.MODEL_REVISION,
            "tokenizer_json_sha256": MODULE.TOKENIZER_JSON_SHA256,
        },
        "task": {"instance_id": "test"},
        "source": {
            "trajectory_bundle_sha256": "a" * 64,
            "trajectory_prompt_count": MODULE.EXPECTED_TRAJECTORY_COUNT,
            "trace_sha256": "b" * 64,
            "dataset_sha256": "d" * 64,
            "prompt_provenance_id": "c" * 64,
        },
        "middle_band": {
            "layers": list(MODULE.EXPECTED_LAYERS),
            "fixed_before_scoring": True,
        },
        "metric": {
            "name": "intermediate_pass_at_k",
            "accepted_target_token_scored": False,
            "pass_at_k": [1, 10, 100],
        },
        "items": [item(ordinal) for ordinal in range(1, 9)],
    }


def trajectory_prompt(
    request_index: int,
    offset: int,
    *,
    request_sha256: str,
    target_token_id: int = 900000,
    prompt_id: str | None = None,
) -> dict[str, object]:
    return {
        "id": prompt_id or f"request-{request_index}-offset-{offset}",
        "token_ids": [request_index, offset, 123],
        "target_token_id": target_token_id,
        "metadata": {
            "provenance_id": "c" * 64,
            "request_index": request_index,
            "source_hashes": {
                "request_sha256": request_sha256,
                "trace_sha256": "b" * 64,
                "tokenizer_json_sha256": MODULE.TOKENIZER_JSON_SHA256,
            },
            "trajectory": {
                "offset": offset,
                "region": "reasoning",
                "events": ["event"],
                "target_token_id": target_token_id,
            },
            "nested": {"preserve": [1, 2, 3]},
        },
    }


def trajectory() -> list[dict[str, object]]:
    selected = [
        trajectory_prompt(ordinal, 0, request_sha256=str(ordinal) * 64)
        for ordinal in range(1, 9)
    ]
    filler = [
        trajectory_prompt(9, offset, request_sha256="9" * 64)
        for offset in range(MODULE.EXPECTED_TRAJECTORY_COUNT - len(selected))
    ]
    return selected + filler


class MaterializeSweIntermediateProbesTest(unittest.TestCase):
    def test_config_rejects_duplicate_items_points_concepts_and_forms(self) -> None:
        cases: list[tuple[dict[str, object], str]] = []

        duplicate_id = config()
        duplicate_id["items"][1]["id"] = duplicate_id["items"][0]["id"]
        cases.append((duplicate_id, "duplicate ID"))

        duplicate_point = config()
        duplicate_point["items"][1]["request_index"] = 1
        cases.append((duplicate_point, "duplicate intermediate probe point"))

        duplicate_concept = config()
        duplicate_concept["items"][1]["intermediates"][0]["key"] = "concept_1"
        duplicate_concept["items"][1]["evidence"][0]["supports"] = ["concept_1"]
        cases.append((duplicate_concept, "duplicate concept key"))

        duplicate_form = config()
        form = duplicate_form["items"][0]["intermediates"][0]["forms"][0]
        duplicate_form["items"][0]["intermediates"][0]["forms"].append(copy.deepcopy(form))
        cases.append((duplicate_form, "duplicate or invalid forms"))

        for value, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    MODULE.validate_config(value)

    def test_config_rejects_unfrozen_method_and_ungrounded_concepts(self) -> None:
        wrong_band = config()
        wrong_band["middle_band"]["layers"] = list(range(17, 48))
        with self.assertRaisesRegex(ValueError, "fixed contiguous layers"):
            MODULE.validate_config(wrong_band)

        lens_selected = config()
        lens_selected["adaptation"]["lens_outputs_used_for_selection"] = True
        with self.assertRaisesRegex(ValueError, "independent of lens outputs"):
            MODULE.validate_config(lens_selected)

        ungrounded = config()
        ungrounded["items"][0]["evidence"][0]["supports"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "unknown concept"):
            MODULE.validate_config(ungrounded)

    def test_config_checks_exact_single_token_round_trip(self) -> None:
        value = config()
        pins = {
            form["text"]: form["token_id"]
            for probe_item in value["items"]
            for concept in probe_item["intermediates"]
            for form in concept["forms"]
        }
        items, token_ids, pass_at_k = MODULE.validate_config(
            value, tokenizer=PinTokenizer(pins)
        )
        self.assertEqual(len(items), 8)
        self.assertEqual(token_ids, tuple(range(1001, 1009)))
        self.assertEqual(pass_at_k, (1, 10, 100))
        with self.assertRaisesRegex(ValueError, "token pin changed"):
            MODULE.validate_config(
                value,
                tokenizer=PinTokenizer(pins, drift=" concept4"),
            )

    def test_selects_exact_points_and_preserves_source_by_deep_copy(self) -> None:
        value = config()
        source = trajectory()
        original = copy.deepcopy(source)
        bundle, summary = MODULE.build_probe_bundle(
            source,
            value,
            config_sha256="c" * 64,
            trajectory_sha256="a" * 64,
        )
        self.assertEqual(len(bundle), 8)
        self.assertEqual(
            [probe["id"] for probe in bundle],
            [f"swe-intermediate-item-{ordinal}" for ordinal in range(1, 9)],
        )
        self.assertEqual([MODULE.trajectory_point(probe) for probe in bundle], [(i, 0) for i in range(1, 9)])
        metadata = bundle[0]["metadata"]["intermediate_probe"]
        self.assertEqual(metadata["middle_band_layers"], list(range(16, 48)))
        self.assertEqual(metadata["intermediates"][0]["key"], "concept_1")
        self.assertFalse(metadata["lens_outputs_used_for_selection"])
        bundle[0]["metadata"]["nested"]["preserve"].append(99)
        bundle[0]["metadata"]["intermediate_probe"]["evidence"].append({})
        self.assertEqual(source, original)
        self.assertEqual(summary["item_count"], 8)
        self.assertEqual(summary["intermediate_count"], 8)
        self.assertEqual(summary["scored_token_ids"], list(range(1001, 1009)))

    def test_rejects_source_drift_missing_points_and_accepted_target_leakage(self) -> None:
        value = config()
        source = trajectory()
        with self.assertRaisesRegex(ValueError, "bundle hash"):
            MODULE.build_probe_bundle(
                source,
                value,
                config_sha256="c" * 64,
                trajectory_sha256="x" * 64,
            )

        missing = copy.deepcopy(source)
        missing[0]["metadata"]["trajectory"]["offset"] = 999
        with self.assertRaisesRegex(ValueError, "does not contain intermediate probe point"):
            MODULE.build_probe_bundle(
                missing,
                value,
                config_sha256="c" * 64,
                trajectory_sha256="a" * 64,
            )

        leaking = copy.deepcopy(source)
        leaking[0]["target_token_id"] = 1001
        leaking[0]["metadata"]["trajectory"]["target_token_id"] = 1001
        with self.assertRaisesRegex(ValueError, "included in scored forms"):
            MODULE.build_probe_bundle(
                leaking,
                value,
                config_sha256="c" * 64,
                trajectory_sha256="a" * 64,
            )

    def test_rejects_duplicate_trajectory_points_and_request_hash_drift(self) -> None:
        value = config()
        source = trajectory()
        duplicate = copy.deepcopy(source)
        duplicate[-1]["metadata"]["trajectory"]["offset"] = 0
        with self.assertRaisesRegex(ValueError, "duplicate point"):
            MODULE.build_probe_bundle(
                duplicate,
                value,
                config_sha256="c" * 64,
                trajectory_sha256="a" * 64,
            )

        drifted = copy.deepcopy(source)
        drifted[0]["metadata"]["source_hashes"]["request_sha256"] = "x" * 64
        with self.assertRaisesRegex(ValueError, "request hash"):
            MODULE.build_probe_bundle(
                drifted,
                value,
                config_sha256="c" * 64,
                trajectory_sha256="a" * 64,
            )

    @unittest.skipUnless(
        HAS_PINNED_INTEGRATION,
        "pinned tokenizer, config, and trajectory bundle are required",
    )
    def test_real_config_token_pins_and_frozen_points_validate(self) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
        real_config = MODULE.require_mapping(
            json.loads(MODULE.DEFAULT_CONFIG.read_text(encoding="utf-8")),
            "intermediate config",
        )
        items, scored_token_ids, pass_at_k = MODULE.validate_config(
            real_config, tokenizer=tokenizer
        )
        self.assertEqual(len(items), 10)
        self.assertEqual(sum(len(item["intermediates"]) for item in items), 17)
        self.assertEqual(len(scored_token_ids), 58)
        self.assertEqual(scored_token_ids[0], 999)
        self.assertEqual(scored_token_ids[-1], 82170)
        self.assertEqual(pass_at_k, (1, 5, 10, 50, 100, 1000))

        real_trajectory = json.loads(
            MODULE.DEFAULT_TRAJECTORY.read_text(encoding="utf-8")
        )
        bundle, summary = MODULE.build_probe_bundle(
            real_trajectory,
            real_config,
            config_sha256=MODULE.sha256_file(MODULE.DEFAULT_CONFIG),
            trajectory_sha256=MODULE.sha256_file(MODULE.DEFAULT_TRAJECTORY),
        )
        self.assertEqual(len(bundle), 10)
        self.assertEqual(summary["intermediate_count"], 17)
        self.assertEqual(summary["scored_token_ids"], list(scored_token_ids))
        self.assertEqual(
            [MODULE.trajectory_point(probe) for probe in bundle],
            [(1, 0), (2, 0), (3, 32), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (9, 60)],
        )


if __name__ == "__main__":
    unittest.main()
