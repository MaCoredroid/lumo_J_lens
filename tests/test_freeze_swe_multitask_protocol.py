#!/usr/bin/env python3
"""Tests for deterministic SWE multitask protocol freezing."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_swe_multitask_protocol",
    ROOT / "scripts" / "freeze_swe_multitask_protocol.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExactTokenizer:
    def __init__(self, exact_texts: list[str]) -> None:
        self.text_to_id = {
            text: 10_000 + index for index, text in enumerate(exact_texts)
        }
        self.id_to_text = {value: key for key, value in self.text_to_id.items()}

    def __len__(self) -> int:
        return MODULE.TOKENIZER_VOCABULARY_SIZE

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("protocol tokenization must disable special tokens")
        if text in self.text_to_id:
            return [self.text_to_id[text]]
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int], **_: object) -> str:
        if len(token_ids) == 1 and token_ids[0] in self.id_to_text:
            return self.id_to_text[token_ids[0]]
        return bytes(token_ids).decode("utf-8")


def concept(kind: str, target: str, *, path: str, contrast: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": path,
        "kind": kind,
        "target": target,
        "sources": [{"artifact": "patch", "file_index": 0, "derivation": "test"}],
    }
    if contrast is not None:
        value["contrast"] = contrast
    return value


def task(index: int, *, base: bool = False) -> dict[str, object]:
    if base:
        return {
            "repo": "sympy/sympy",
            "instance_id": MODULE.TEMPLATE_BASE_INSTANCE_ID,
            "base_commit": "b" * 40,
            "version": "1.0",
            "problem_statement": "Base issue.",
            "patch_sha256": "1" * 64,
            "test_patch_sha256": "2" * 64,
            "source_provenance": {"dataset_row_index": 999},
            "concepts": [],
        }
    return {
        "repo": f"owner{index}/project{index}",
        "instance_id": f"owner{index}__project{index}-{index}",
        "base_commit": f"{index:x}" * 40,
        "version": f"{index}.0",
        "problem_statement": "Synthetic issue.\r\nKeep these line endings." if index == 0 else "Synthetic issue.",
        "patch_sha256": f"{(index + 3) % 16:x}" * 64,
        "test_patch_sha256": f"{(index + 4) % 16:x}" * 64,
        "source_provenance": {"dataset_row_index": index},
        "concepts": [
            concept("symbol", f"needle{index}", path=f"pkg/module{index}.py"),
            concept("file_stem", f"module{index}", path=f"pkg/module{index}.py"),
        ],
    }


def manifest() -> dict[str, object]:
    tasks = [task(99, base=True)] + [task(index) for index in range(10)]
    return {
        "schema_version": 1,
        "kind": "swe_verified_initial_probe_candidates",
        "source": {"mode": "test"},
        "extraction": {"parser": "test"},
        "task_count": len(tasks),
        "concept_count": sum(len(value["concepts"]) for value in tasks),
        "tasks": tasks,
    }


def tokenizer_for(value: dict[str, object]) -> ExactTokenizer:
    texts: set[str] = set()
    for selected_task in value["tasks"]:
        for raw_concept in selected_task["concepts"]:
            target = raw_concept["target"]
            texts.update([target, f" {target}"])
    return ExactTokenizer(sorted(texts))


def template_fixture(value: dict[str, object], tokenizer: ExactTokenizer) -> tuple[bytes, dict[str, object]]:
    base = next(
        item for item in value["tasks"] if item["instance_id"] == MODULE.TEMPLATE_BASE_INSTANCE_ID
    )
    agents = MODULE.render_agents_md(base)
    prompt = (
        "GENERIC PREFIX\n"
        + MODULE.AGENTS_START_MARKER
        + "\n"
        + agents.rstrip("\n")
        + "\n"
        + MODULE.AGENTS_END_MARKER
        + "\npaths/"
        + MODULE.TEMPLATE_BASE_INSTANCE_ID
        + "/projects/"
        + MODULE.TEMPLATE_BASE_INSTANCE_ID.replace("_", "-")
    )
    report = {
        "experiments": [
            {
                "id": "synthetic-template",
                "prompt": prompt,
                "prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=False),
            }
        ]
    }
    raw = (json.dumps(report, sort_keys=True) + "\n").encode("utf-8")
    extracted = MODULE.extract_template(
        raw,
        expected_report_sha256=hashlib.sha256(raw).hexdigest(),
        expected_prompt_sha256=MODULE.sha256_text(prompt),
    )
    return raw, extracted


def freeze(value: dict[str, object]) -> dict[str, object]:
    tokenizer = tokenizer_for(value)
    _, template = template_fixture(value, tokenizer)
    return MODULE.freeze_protocol(
        value,
        candidate_manifest_sha256="a" * 64,
        candidate_manifest_path="candidate.json",
        template=template,
        template_report_path="template.json",
        tokenizer=tokenizer,
        model_snapshot_path="snapshot",
    )


class FreezeSweMultitaskProtocolTest(unittest.TestCase):
    def test_freeze_is_order_independent_and_preserves_task_bytes(self) -> None:
        value = manifest()
        first = freeze(value)
        reversed_value = copy.deepcopy(value)
        reversed_value["tasks"].reverse()
        second = freeze(reversed_value)
        self.assertEqual(first, second)
        self.assertEqual(first["coverage"]["selected_task_count"], 10)
        self.assertEqual(first["coverage"]["selected_repo_count"], 10)
        self.assertTrue(first["lens_outputs_used_for_selection"] is False)
        selected = {item["instance_id"]: item for item in first["tasks"]}
        self.assertIn("\r\n", selected["owner0__project0-0"]["problem_statement"])
        self.assertTrue(
            all(item["projected_prompt_token_count"] <= MODULE.MAXIMUM_PROMPT_TOKENS for item in first["tasks"])
        )

    def test_targets_are_unique_and_foils_are_cross_task_same_family(self) -> None:
        value = manifest()
        value["tasks"][2]["concepts"][0]["target"] = "needle0"
        protocol = freeze(value)
        targets = [
            concept_value["target"].casefold()
            for selected_task in protocol["tasks"]
            for concept_value in selected_task["concepts"]
        ]
        self.assertEqual(len(targets), len(set(targets)))
        self.assertEqual(targets.count("needle0"), 1)
        for selected_task in protocol["tasks"]:
            for concept_value in selected_task["concepts"]:
                for foil in concept_value["foils"]:
                    self.assertNotEqual(foil["task_instance_id"], selected_task["instance_id"])
                    self.assertEqual(foil["family"], concept_value["family"])

    def test_leakage_stopword_short_and_token_filters(self) -> None:
        tokenizer = ExactTokenizer(["hidden", " hidden", "if", " if", "ab", " ab", "models", " models"])
        cases = (
            (concept("symbol", "hidden", path="pkg/x.py"), "Issue mentions hidden.", "", "target_visible_in_task_agents"),
            (concept("symbol", "hidden", path="pkg/x.py"), "Issue.", "generic hidden tail", "target_visible_in_template_remainder"),
            (concept("symbol", "models", path="pkg/x.py"), "Issue.", "", "generic_coding_stopword"),
            (concept("symbol", "if", path="pkg/x.py"), "Issue.", "", "python_keyword"),
            (concept("symbol", "ab", path="pkg/x.py"), "Issue.", "", "target_canonical_too_short"),
            (concept("symbol", "not-tokenized", path="pkg/x.py"), "Issue.", "", "not_python_identifier"),
        )
        for candidate, visible, remainder, expected in cases:
            with self.subTest(expected=expected):
                selected, reason = MODULE.concept_eligibility(
                    candidate,
                    task_visible_text=visible,
                    template_remainder=remainder,
                    tokenizer=tokenizer,
                )
                self.assertIsNone(selected)
                self.assertEqual(reason, expected)

    def test_template_hash_markers_and_projected_identifiers_are_bound(self) -> None:
        value = manifest()
        tokenizer = tokenizer_for(value)
        raw, template = template_fixture(value, tokenizer)
        with self.assertRaisesRegex(ValueError, "template report SHA-256 mismatch"):
            MODULE.extract_template(
                raw,
                expected_report_sha256="0" * 64,
                expected_prompt_sha256=template["prompt_sha256"],
            )
        base = value["tasks"][0]
        target_value = value["tasks"][1]
        projected = MODULE.project_prompt(template, base, target_value)
        self.assertNotIn(MODULE.TEMPLATE_BASE_INSTANCE_ID, projected)
        self.assertNotIn(MODULE.TEMPLATE_BASE_INSTANCE_ID.replace("_", "-"), projected)
        self.assertIn(target_value["instance_id"], projected)
        self.assertIn(target_value["problem_statement"], projected)

    def test_manifest_rejects_count_drift_and_unsafe_paths(self) -> None:
        value = manifest()
        value["task_count"] += 1
        with self.assertRaisesRegex(ValueError, "task_count mismatch"):
            MODULE.validate_candidate_manifest(value)
        value = manifest()
        value["tasks"][1]["concepts"][0]["path"] = "../outside.py"
        with self.assertRaisesRegex(ValueError, "unsafe/non-Python"):
            MODULE.validate_candidate_manifest(value)

    def test_home_cache_path_is_portable(self) -> None:
        cached = Path.home() / ".cache" / "huggingface" / "snapshot"
        self.assertEqual(
            MODULE.portable_path(cached), "$HOME/.cache/huggingface/snapshot"
        )


if __name__ == "__main__":
    unittest.main()
