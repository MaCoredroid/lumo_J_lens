#!/usr/bin/env python3
"""Focused tests for frozen SWE task-start prompt materialization."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "materialize_swe_multitask_initial_probes",
    ROOT / "scripts" / "materialize_swe_multitask_initial_probes.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    vocab_size = 10_000

    def __init__(self) -> None:
        self.forms = {" needle": 9001, " rival": 9002, " locator": 9003}
        self.reverse = {value: key for key, value in self.forms.items()}

    def __len__(self) -> int:
        return 10_000

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("special tokens must stay disabled")
        if text in self.forms:
            return [self.forms[text]]
        return [100 + ord(character) for character in text]

    def decode(self, token_ids: list[int], **_: object) -> str:
        return self.reverse.get(token_ids[0], "not-a-pinned-form")


def task(instance_id: str, problem: str) -> dict[str, object]:
    return {
        "repo": instance_id.split("__", 1)[0] + "/repo",
        "instance_id": instance_id,
        "base_commit": "a" * 40,
        "version": "1.0",
        "problem_statement": problem,
        "patch_sha256": "b" * 64,
        "test_patch_sha256": "c" * 64,
        "source_provenance": {"dataset_row_index": 0},
        "concepts": [],
    }


def compact_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def fixture(*, target_problem: str = "Correct the broken behavior.") -> dict[str, object]:
    tokenizer = FakeTokenizer()
    source = task("base__repo-1", "First line\r\nSecond line")
    target = task("other__repo-2", target_problem)
    source_concept = {
        "kind": "symbol",
        "path": "pkg/source.py",
        "target": "source_symbol",
        "sources": [{"artifact": "patch", "file_index": 0}],
    }
    selected_candidate = {
        "kind": "symbol",
        "path": "pkg/engine.py",
        "target": "needle",
        "sources": [{"artifact": "patch", "file_index": 0, "hunk_index": 1}],
    }
    source["concepts"] = [source_concept]
    target["concepts"] = [selected_candidate]
    candidates = {
        "schema_version": 1,
        "kind": "swe_verified_initial_probe_candidates",
        "source": {"mode": "synthetic", "sha256": "d" * 64},
        "extraction": {"parser": "test"},
        "task_count": 2,
        "concept_count": 2,
        "tasks": [source, target],
    }
    candidate_bytes = json.dumps(candidates, sort_keys=True).encode()
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()

    start_marker = "--- Context from: AGENTS.md ---"
    end_marker = "--- End of Context from: AGENTS.md ---"
    source_block = "\n" + MODULE.render_agents_md(source).rstrip("\n") + "\n"
    prefix = "PREFIX base__repo-1 base--repo-1\n"
    suffix = "\nSUFFIX"
    source_prompt = prefix + start_marker + source_block + end_marker + suffix
    source_ids = tokenizer.encode(source_prompt, add_special_tokens=False)
    report = {
        "experiments": [
            {
                "id": "swe-base-request-01",
                "prompt": source_prompt,
                "prompt_token_ids": source_ids,
            }
        ]
    }
    report_bytes = json.dumps(report, sort_keys=True).encode()
    report_sha = hashlib.sha256(report_bytes).hexdigest()

    selected = copy.deepcopy(target)
    selected["selection_index"] = 0
    selected["concepts"] = [
        {
            **copy.deepcopy(selected_candidate),
            "id": "other-repo-2-symbol-needle",
            "family": "hunk_symbol",
            "forms": [{"kind": "leading_space", "text": " needle", "token_id": 9001}],
            "form_exclusions": [],
            "foils": [
                {
                    "task_instance_id": "foil__repo-3",
                    "concept_id": "foil-symbol-rival",
                    "family": "hunk_symbol",
                    "target": "rival",
                    "forms": [
                        {"kind": "leading_space", "text": " rival", "token_id": 9002}
                    ],
                }
            ],
            "foil_status": "available",
        }
    ]
    selected["score_token_ids"] = [9001, 9002]
    target_block = "\n" + MODULE.render_agents_md(selected).rstrip("\n") + "\n"
    projected = prefix + start_marker + target_block + end_marker + suffix
    projected = projected.replace("base__repo-1", "other__repo-2").replace(
        "base--repo-1", "other--repo-2"
    )
    selected["projected_prompt_sha256"] = MODULE.sha256_text(projected)
    selected["projected_prompt_token_count"] = len(
        tokenizer.encode(projected, add_special_tokens=False)
    )

    protocol = {
        "schema_version": 1,
        "kind": "swe_verified_initial_probe_protocol",
        "status": "exploratory_development_pilot",
        "lens_outputs_used_for_selection": False,
        "pins": {
            "candidate_manifest": {
                "sha256": candidate_sha,
                **{key: copy.deepcopy(candidates[key]) for key in (
                    "schema_version", "kind", "source", "extraction", "task_count", "concept_count"
                )},
            },
            "template": {
                "report_path": "synthetic.json",
                "report_sha256": report_sha,
                "experiment_index": 0,
                "experiment_id": "swe-base-request-01",
                "rendered_prompt_sha256": MODULE.sha256_text(source_prompt),
                "prompt_token_ids_sha256": compact_sha(source_ids),
                "agents_start_marker": start_marker,
                "agents_end_marker": end_marker,
                "agents_block_sha256": MODULE.sha256_text(source_block),
                "remainder_sha256": MODULE.sha256_text(prefix + suffix),
                "base_instance_id": "base__repo-1",
                "base_hyphenated_project_slug": "base--repo-1",
            },
            "model": {
                "repo_id": MODULE.MODEL_REPO,
                "revision": MODULE.MODEL_REVISION,
                "tokenizer_json_sha256": MODULE.TOKENIZER_JSON_SHA256,
                "tokenizer_vocabulary_size": 10_000,
                "logit_vocabulary_size": 10_000,
            },
        },
        "metric_contract": {"middle_band_layers": list(range(16, 48))},
        "scored_vocabulary": {"token_ids": [9001, 9002]},
        "tasks": [selected],
    }
    protocol_bytes = json.dumps(protocol, sort_keys=True).encode()
    return {
        "tokenizer": tokenizer,
        "source": source,
        "source_prompt": source_prompt,
        "projected": projected,
        "protocol": protocol,
        "protocol_sha": hashlib.sha256(protocol_bytes).hexdigest(),
        "candidates": candidates,
        "candidate_sha": candidate_sha,
        "report": report,
        "report_sha": report_sha,
    }


def build(value: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    with mock.patch.object(MODULE, "_validate_model_pin"):
        return MODULE.build_probe_bundle(
            value["protocol"],
            value["candidates"],
            value["report"],
            protocol_sha256=value["protocol_sha"],
            candidate_sha256=value["candidate_sha"],
            report_sha256=value["report_sha"],
            tokenizer=value["tokenizer"],
            snapshot=Path("/synthetic") / MODULE.MODEL_REVISION,
        )


class MaterializeInitialProbesTest(unittest.TestCase):
    def test_exact_base_block_crlf_substitution_and_metadata(self) -> None:
        value = fixture()
        self.assertIn("First line\r\nSecond line", value["source_prompt"])
        bundle, summary = build(value)
        prompt = bundle[0]
        self.assertEqual(prompt["text"], value["projected"])
        self.assertNotIn("base__repo-1", prompt["text"])
        self.assertNotIn("base--repo-1", prompt["text"])
        self.assertIn("other__repo-2", prompt["text"])
        self.assertIn("other--repo-2", prompt["text"])
        self.assertEqual(prompt["token_ids"], value["tokenizer"].encode(prompt["text"], add_special_tokens=False))
        self.assertEqual(prompt["score_token_ids"], [9001, 9002])
        self.assertEqual(prompt["metadata"]["checkpoint"]["id"], "C0")
        self.assertEqual(prompt["metadata"]["concepts"][0]["evidence"][0]["hunk_index"], 1)
        self.assertEqual(summary["prompt_count"], 1)

    def test_rejects_base_block_byte_drift_including_crlf(self) -> None:
        value = fixture()
        prompt = value["report"]["experiments"][0]["prompt"].replace("\r\n", "\n")
        value["report"]["experiments"][0]["prompt"] = prompt
        value["report"]["experiments"][0]["prompt_token_ids"] = value["tokenizer"].encode(
            prompt, add_special_tokens=False
        )
        pin = value["protocol"]["pins"]["template"]
        pin["rendered_prompt_sha256"] = MODULE.sha256_text(prompt)
        pin["prompt_token_ids_sha256"] = compact_sha(
            value["report"]["experiments"][0]["prompt_token_ids"]
        )
        block = prompt.split(pin["agents_start_marker"], 1)[1].split(pin["agents_end_marker"], 1)[0]
        pin["agents_block_sha256"] = MODULE.sha256_text(block)
        value["report_sha"] = pin["report_sha256"] = hashlib.sha256(
            json.dumps(value["report"], sort_keys=True).encode()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "certified AGENTS byte contract"):
            build(value)

    def test_rejects_target_and_foil_leakage(self) -> None:
        for problem, expected in [
            ("Please repair NEEDLE now.", "target leakage"),
            ("The Rival path is broken.", "foil leakage"),
        ]:
            value = fixture(target_problem=problem)
            with self.subTest(problem=problem):
                with self.assertRaisesRegex(ValueError, expected):
                    build(value)

    def test_rejects_hash_and_score_vocabulary_mismatches(self) -> None:
        bad_hash = fixture()
        bad_hash["candidate_sha"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "candidate manifest hash"):
            build(bad_hash)

        bad_vocab = fixture()
        bad_vocab["protocol"]["tasks"][0]["score_token_ids"] = [9001]
        with self.assertRaisesRegex(ValueError, "score vocabulary"):
            build(bad_vocab)

        bad_global = fixture()
        bad_global["protocol"]["scored_vocabulary"]["token_ids"] = [9001]
        with self.assertRaisesRegex(ValueError, "global score vocabulary"):
            build(bad_global)


if __name__ == "__main__":
    unittest.main()
