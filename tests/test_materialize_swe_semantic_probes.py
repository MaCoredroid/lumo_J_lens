#!/usr/bin/env python3
"""Tests for certified SWE semantic-probe selection."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_swe_semantic_probes",
    ROOT / "scripts" / "materialize_swe_semantic_probes.py",
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


class ByteTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("semantic token pins must disable special tokens")
        return list(text.encode("ascii"))

    def decode(self, token_ids: list[int], **_: object) -> str:
        return bytes(token_ids).decode("ascii")


class DriftTokenizer(ByteTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if text == "a":
            return [999]
        return super().encode(text, add_special_tokens=add_special_tokens)


def token(text: str, token_id: int | None = None) -> dict[str, object]:
    return {
        "text": text,
        "token_id": ord(text) if token_id is None else token_id,
    }


def probe(
    probe_id: str,
    request_index: int,
    offset: int,
    *,
    positive: list[dict[str, object]] | None = None,
    negative: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "id": probe_id,
        "request_index": request_index,
        "offset": offset,
        "state": f"state-{probe_id}",
        "positive": positive or [token("a")],
        "negative": negative or [token("b")],
    }


def config(*probes: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "test_semantic_contrasts",
        "model": {
            "repo_id": MODULE.MODEL_REPO,
            "revision": MODULE.MODEL_REVISION,
        },
        "task": "task-1",
        "primary_layers": [39, 40],
        "selection_note": "fixed before scoring",
        "probes": list(probes) or [probe("probe-1", 1, 0)],
    }


def trajectory_prompt(
    request_index: int,
    offset: int,
    *,
    prompt_id: str | None = None,
) -> dict[str, object]:
    return {
        "id": prompt_id or f"request-{request_index}-offset-{offset}",
        "token_ids": [request_index, offset, 123],
        "target_token_id": 456,
        "metadata": {
            "request_index": request_index,
            "stage_name": f"stage-{request_index}",
            "trajectory": {
                "offset": offset,
                "region": "reasoning",
                "events": ["event"],
            },
            "nested": {"preserve": [1, 2, 3]},
        },
    }


class MaterializeSweSemanticProbesTest(unittest.TestCase):
    def test_config_rejects_duplicate_ids_points_and_group_token_ids(self) -> None:
        cases = (
            (
                config(probe("same", 1, 0), probe("same", 2, 0)),
                "duplicate id",
            ),
            (
                config(probe("one", 1, 0), probe("two", 1, 0)),
                "duplicate semantic probe point",
            ),
            (
                config(
                    probe(
                        "tokens",
                        1,
                        0,
                        positive=[token("a"), token("c", ord("a"))],
                    )
                ),
                "invalid token IDs",
            ),
        )
        for value, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    MODULE.validate_config(value)

    def test_config_rejects_positive_negative_overlap(self) -> None:
        value = config(
            probe(
                "overlap",
                1,
                0,
                positive=[token("a")],
                negative=[token("x", ord("a"))],
            )
        )
        with self.assertRaisesRegex(ValueError, "overlapping contrast tokens"):
            MODULE.validate_config(value)

    def test_config_rejects_tokenizer_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "token pin changed"):
            MODULE.validate_config(config(probe("drift", 1, 0)), tokenizer=DriftTokenizer())
        probes, token_ids = MODULE.validate_config(
            config(probe("stable", 1, 0)), tokenizer=ByteTokenizer()
        )
        self.assertEqual([item["id"] for item in probes], ["stable"])
        self.assertEqual(token_ids, (ord("a"), ord("b")))

    def test_selects_exact_points_and_deep_copies_source_metadata(self) -> None:
        first = trajectory_prompt(1, 4)
        second = trajectory_prompt(2, 7)
        trajectory = [second, first]
        original = copy.deepcopy(trajectory)
        value = config(probe("first", 1, 4), probe("second", 2, 7))
        bundle, summary = MODULE.build_probe_bundle(
            trajectory,
            value,
            config_sha256="c" * 64,
            trajectory_sha256="t" * 64,
        )
        self.assertEqual(
            [item["id"] for item in bundle],
            ["swe-semantic-first", "swe-semantic-second"],
        )
        self.assertEqual(bundle[0]["token_ids"], first["token_ids"])
        semantic = bundle[0]["metadata"]["semantic_probe"]
        self.assertEqual(semantic["id"], "first")
        self.assertEqual(semantic["config_sha256"], "c" * 64)
        self.assertEqual(semantic["trajectory_bundle_sha256"], "t" * 64)
        bundle[0]["metadata"]["nested"]["preserve"].append(99)
        bundle[0]["metadata"]["semantic_probe"]["positive_token_ids"].append(999)
        self.assertEqual(trajectory, original)
        self.assertEqual(summary["probe_ids"], ["first", "second"])

    def test_rejects_missing_and_duplicate_trajectory_points(self) -> None:
        value = config(probe("wanted", 3, 9))
        with self.assertRaisesRegex(ValueError, "does not contain semantic probe point"):
            MODULE.build_probe_bundle(
                [trajectory_prompt(3, 8)],
                value,
                config_sha256="c",
                trajectory_sha256="t",
            )
        with self.assertRaisesRegex(ValueError, "trajectory contains duplicate point"):
            MODULE.build_probe_bundle(
                [trajectory_prompt(3, 9), trajectory_prompt(3, 9, prompt_id="again")],
                value,
                config_sha256="c",
                trajectory_sha256="t",
            )

    def test_summary_and_scored_token_union_are_deterministic(self) -> None:
        value = config(
            probe(
                "later",
                2,
                5,
                positive=[token("z")],
                negative=[token("a")],
            ),
            probe(
                "earlier",
                1,
                3,
                positive=[token("c"), token("a")],
                negative=[token("b")],
            ),
        )
        trajectory = [trajectory_prompt(1, 3), trajectory_prompt(2, 5)]
        original_config = copy.deepcopy(value)
        original_trajectory = copy.deepcopy(trajectory)
        first_bundle, first_summary = MODULE.build_probe_bundle(
            trajectory,
            value,
            config_sha256="1" * 64,
            trajectory_sha256="2" * 64,
        )
        second_bundle, second_summary = MODULE.build_probe_bundle(
            copy.deepcopy(trajectory),
            copy.deepcopy(value),
            config_sha256="1" * 64,
            trajectory_sha256="2" * 64,
        )
        self.assertEqual(first_bundle, second_bundle)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_summary["probe_ids"], ["later", "earlier"])
        self.assertEqual(
            first_summary["scored_token_ids"],
            sorted({ord("a"), ord("b"), ord("c"), ord("z")}),
        )
        self.assertEqual(value, original_config)
        self.assertEqual(trajectory, original_trajectory)
        self.assertEqual(
            json.dumps(first_summary, sort_keys=True, separators=(",", ":")),
            json.dumps(second_summary, sort_keys=True, separators=(",", ":")),
        )

    @unittest.skipUnless(
        HAS_PINNED_INTEGRATION,
        "pinned tokenizer, semantic config, and trajectory bundle are required",
    )
    def test_real_pinned_config_tokens_and_points_validate(self) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
        real_config = MODULE.require_mapping(
            json.loads(MODULE.DEFAULT_CONFIG.read_text(encoding="utf-8")),
            "semantic config",
        )
        probes, scored_token_ids = MODULE.validate_config(
            real_config, tokenizer=tokenizer
        )
        self.assertEqual(
            [item["id"] for item in probes],
            [
                "correct_identifier",
                "failure_confirmed",
                "fix_succeeded",
                "pytest_unavailable",
                "focused_test_passed",
            ],
        )
        self.assertEqual(len(scored_token_ids), 28)
        self.assertEqual(scored_token_ids[0], 981)
        self.assertEqual(scored_token_ids[-1], 82170)

        trajectory = json.loads(MODULE.DEFAULT_TRAJECTORY.read_text(encoding="utf-8"))
        bundle, summary = MODULE.build_probe_bundle(
            trajectory,
            real_config,
            config_sha256=MODULE.sha256_file(MODULE.DEFAULT_CONFIG),
            trajectory_sha256=MODULE.sha256_file(MODULE.DEFAULT_TRAJECTORY),
        )
        self.assertEqual(len(bundle), 5)
        self.assertEqual(summary["scored_token_ids"], list(scored_token_ids))
        self.assertEqual(
            [MODULE.trajectory_point(item) for item in bundle],
            [(3, 32), (4, 0), (6, 0), (8, 0), (9, 0)],
        )


if __name__ == "__main__":
    unittest.main()
