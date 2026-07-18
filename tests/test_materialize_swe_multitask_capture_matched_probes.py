#!/usr/bin/env python3
"""Focused tests for capture-matched task-start SWE probe materialization."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "materialize_swe_multitask_capture_matched_probes",
    ROOT / "scripts" / "materialize_swe_multitask_capture_matched_probes.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    forms = {" secret": 9001, " decoy": 9002}
    reverse = {token_id: text for text, token_id in forms.items()}
    generation_suffix = "<|im_start|>assistant\n<think>\n"

    def apply_chat_template(self, messages: object, **_: object) -> str:
        values = list(messages)
        rendered = "PROMPT\n"
        for index, message in enumerate(values):
            if index >= 2 and message["role"] == "assistant":
                rendered += self.generation_suffix
            rendered += json.dumps(message, sort_keys=True, ensure_ascii=False) + "\n"
        return rendered + self.generation_suffix

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("special tokens must be disabled")
        if text in self.forms:
            return [self.forms[text]]
        token_ids = [100 + ord(character) for character in text]
        marker = "TOKEN_ONLY"
        if marker in text:
            insertion = text.index(marker) + len(marker)
            token_ids.insert(insertion, 9001)
        return token_ids

    def decode(self, token_ids: list[int], **_: object) -> str:
        return self.reverse.get(token_ids[0], "not-a-form")


def protocol() -> dict[str, object]:
    concept = {
        "id": "task-00-concept-00",
        "kind": "symbol",
        "family": "hunk_symbol",
        "path": "src/module.py",
        "target": "secret",
        "sources": [{"artifact": "patch", "file_index": 0, "hunk_index": 0}],
        "forms": [{"kind": "leading_space", "text": " secret", "token_id": 9001}],
        "foils": [
            {
                "task_instance_id": "foil__repo-2",
                "concept_id": "task-01-concept-00",
                "family": "hunk_symbol",
                "kind": "symbol",
                "target": "decoy",
                "forms": [
                    {"kind": "leading_space", "text": " decoy", "token_id": 9002}
                ],
            }
        ],
    }
    return {
        "schema_version": 1,
        "kind": "swe_verified_initial_probe_protocol",
        "status": "exploratory_development_pilot",
        "lens_outputs_used_for_selection": False,
        "pins": {"candidate_manifest": {}, "template": {}, "model": {}},
        "metric_contract": {"middle_band_layers": list(range(16, 48))},
        "tasks": [
            {
                "selection_index": 0,
                "repo": "owner/repo",
                "instance_id": "owner__repo-1",
                "base_commit": "a" * 40,
                "version": "1.0",
                "problem_statement": "Repair the behavior without naming its implementation.",
                "patch_sha256": "b" * 64,
                "test_patch_sha256": "c" * 64,
                "source_provenance": {"dataset_row_index": 0},
                "projected_prompt_sha256": "d" * 64,
                "projected_prompt_token_count": 100,
                "concepts": [concept],
                "score_token_ids": [9001, 9002],
            }
        ],
    }


def request_pair(initial_text: str) -> tuple[dict[str, object], dict[str, object]]:
    tools = [
        {
            "type": "function",
            "function": {"name": "run_shell_command", "description": "Run shell"},
        }
    ]
    initial_messages = [
        {"role": "system", "content": "Use the repository tool."},
        {"role": "user", "content": initial_text},
    ]
    first = {
        "model": "qwen3.6-27b-nvfp4",
        "messages": copy.deepcopy(initial_messages),
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "seed": 880001234,
        "tools": copy.deepcopy(tools),
    }
    command = "rg implementation src"
    call_id = "call-1"
    assistant = {
        "role": "assistant",
        "content": "I will inspect the repository.",
        "reasoning_content": "I need a repository observation.",
        "reasoning": "I need a repository observation.",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "run_shell_command",
                    "arguments": json.dumps(
                        {"command": command, "description": "Search repository"},
                        separators=(",", ":"),
                    ),
                },
            }
        ],
    }
    result = (
        f"Command: {command}\nDirectory: (root)\nOutput: src/module.py:class Implementation:"
        "\nError: (none)\nExit Code: 0\nSignal: 0\nProcess Group PGID: 123"
    )
    tool_result = {
        "role": "tool",
        "tool_call_id": call_id,
        "content": [{"type": "text", "text": result}],
    }
    second = {
        **copy.deepcopy(first),
        "seed": 880001235,
        "messages": [*copy.deepcopy(initial_messages), assistant, tool_result],
    }
    return first, second


def fixture(
    initial_text: str = "Task owner__repo-1: repair the behavior.",
) -> dict[str, object]:
    tokenizer = FakeTokenizer()
    frozen_protocol = protocol()
    task = frozen_protocol["tasks"][0]
    first, second = request_pair(initial_text)
    first_text, first_ids, _, _ = MODULE.C1.render_request(
        tokenizer, request=first, template="test-template"
    )
    second_text, second_ids, normalized_count, second_messages = MODULE.C1.render_request(
        tokenizer, request=second, template="test-template"
    )
    source_concept = task["concepts"][0]
    retained = {
        "id": source_concept["id"],
        "family": source_concept["family"],
        "target": source_concept["target"],
        "path": source_concept["path"],
        "evidence": copy.deepcopy(source_concept["sources"]),
        "visibility": "oracle_hidden",
        "forms": copy.deepcopy(source_concept["forms"]),
        "foils": copy.deepcopy(source_concept["foils"]),
    }
    protocol_sha = "f" * 64
    capture_sha = "3" * 64
    usage_sha = "4" * 64
    c1_prompt = {
        "id": "swe-c1-02-owner__repo-1",
        "text": second_text,
        "token_ids": second_ids,
        "score_token_ids": [9001, 9002],
        "metadata": {
            "kind": "swe_verified_multitask_initial_probe",
            "protocol_sha256": protocol_sha,
            "lens_outputs_used_for_selection": False,
            "task": {
                "instance_id": task["instance_id"],
                "repo": task["repo"],
                "base_commit": task["base_commit"],
                "problem_statement_sha256": MODULE.C1.sha256_text(task["problem_statement"]),
                "patch_sha256": task["patch_sha256"],
                "test_patch_sha256": task["test_patch_sha256"],
            },
            "checkpoint": copy.deepcopy(MODULE.C1.C1_CHECKPOINT),
            "middle_band_layers": list(range(16, 48)),
            "concepts": [retained],
            "observation_audit": {
                "capture_manifest_sha256": capture_sha,
                "usage_manifest_sha256": usage_sha,
                "first_request_index": 1,
                "second_request_index": 2,
                "first_request_sha256": "1" * 64,
                "second_request_sha256": "2" * 64,
                "normalized_messages_sha256": MODULE.C1.sha256_json(second_messages),
                "normalized_string_tool_call_arguments": normalized_count,
            },
        },
    }
    return {
        "protocol": frozen_protocol,
        "protocol_sha": protocol_sha,
        "c1_prompts": [c1_prompt],
        "c1_sha": "5" * 64,
        "requests": [first, second],
        "sources": [
            {
                "sha256": "1" * 64,
                "capture_instance_id": task["instance_id"],
                "capture_request_index": 1,
            },
            {
                "sha256": "2" * 64,
                "capture_instance_id": task["instance_id"],
                "capture_request_index": 2,
            },
        ],
        "usage": [
            {"idx": 1, "usage": {"prompt_tokens": len(first_ids)}},
            {"idx": 2, "usage": {"prompt_tokens": len(second_ids)}},
        ],
        "tokenizer": tokenizer,
        "template": "test-template",
        "capture_sha": capture_sha,
        "usage_sha": usage_sha,
        "first_text": first_text,
        "first_ids": first_ids,
        "second_text": second_text,
        "second_ids": second_ids,
    }


def build(value: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    return MODULE.build_c0m_bundle(
        value["protocol"],
        protocol_sha256=value["protocol_sha"],
        c1_prompts=value["c1_prompts"],
        c1_prompt_bundle_sha256=value["c1_sha"],
        requests=value["requests"],
        request_sources=value["sources"],
        usage_records=value["usage"],
        tokenizer=value["tokenizer"],
        template=value["template"],
        capture_manifest_sha256=value["capture_sha"],
        usage_manifest_sha256=value["usage_sha"],
    )


class MaterializeCaptureMatchedProbesTest(unittest.TestCase):
    def test_materializes_exact_task_start_prefix_with_c1_vocabulary(self) -> None:
        value = fixture()
        prompts, summary = build(value)
        self.assertEqual(len(prompts), 1)
        prompt = prompts[0]
        c1_prompt = value["c1_prompts"][0]
        self.assertEqual(prompt["text"], value["first_text"])
        self.assertEqual(prompt["token_ids"], value["first_ids"])
        self.assertEqual(c1_prompt["token_ids"][: len(prompt["token_ids"])], prompt["token_ids"])
        self.assertEqual(prompt["score_token_ids"], c1_prompt["score_token_ids"])
        self.assertEqual(prompt["metadata"]["concepts"], c1_prompt["metadata"]["concepts"])
        self.assertEqual(prompt["metadata"]["checkpoint"], MODULE.C0M_CHECKPOINT)
        match = prompt["metadata"]["capture_match"]
        self.assertTrue(match["exact_message_prefix"])
        self.assertTrue(match["exact_rendered_text_prefix"])
        self.assertTrue(match["exact_token_id_prefix"])
        visibility = prompt["metadata"]["visibility_audit"]
        self.assertTrue(visibility["all_retained_forms_hidden"])
        self.assertTrue(all(record["hidden"] for record in visibility["records"]))
        self.assertEqual(summary["selected_prompt_count"], 1)
        self.assertEqual(summary["retained_primary_concept_count"], 1)
        self.assertEqual(summary["retained_foil_count"], 1)

    def test_rejects_surface_and_token_only_exposure_in_full_prompt(self) -> None:
        for text in (
            "Task owner__repo-1: inspect secret_implementation.",
            "Task owner__repo-1: inspect TOKEN_ONLY.",
        ):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "full rendered prompt exposes target"):
                    build(fixture(text))

    def test_rejects_usage_and_capture_hash_drift(self) -> None:
        bad_usage = fixture()
        bad_usage["usage"][0]["usage"]["prompt_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "pinned usage token count"):
            build(bad_usage)

        bad_hash = fixture()
        bad_hash["sources"][0]["sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "first_request_sha256"):
            build(bad_hash)

    def test_rejects_c1_concept_or_prompt_prefix_drift(self) -> None:
        bad_concept = fixture()
        bad_concept["c1_prompts"][0]["metadata"]["concepts"][0]["foils"][0][
            "target"
        ] = "different"
        with self.assertRaisesRegex(ValueError, "C1 retained foil"):
            build(bad_concept)

        bad_prompt = fixture()
        bad_prompt["c1_prompts"][0]["token_ids"] = bad_prompt["second_ids"][:-1]
        with self.assertRaisesRegex(ValueError, "paired C1 token IDs"):
            build(bad_prompt)

    def test_rejects_nonmatching_message_prefix(self) -> None:
        value = fixture()
        value["requests"][1]["messages"][0]["content"] = "history drift"
        with self.assertRaisesRegex(ValueError, "exactly preserve"):
            build(value)

    def test_real_c1_and_capture_pins_are_available(self) -> None:
        self.assertEqual(
            MODULE.C1.sha256_file(MODULE.DEFAULT_C1_PROMPTS),
            MODULE.EXPECTED_C1_PROMPT_BUNDLE_SHA256,
        )
        c1_prompts = json.loads(MODULE.DEFAULT_C1_PROMPTS.read_bytes())
        self.assertEqual(len(c1_prompts), 8)
        requests, sources, usage, capture_sha, usage_sha = MODULE.C1.load_merged_capture(
            ROOT
        )
        self.assertEqual((len(requests), len(sources), len(usage)), (20, 20, 20))
        self.assertEqual(
            capture_sha,
            "62fd83a16262845475057f950d2ac32816ad93fc9bd3802b6cdbb4f66319f895",
        )
        self.assertEqual(
            usage_sha,
            "eb738f3be2f5537db78ac5fe154aa30d681bc948bf1960da65955ecd3493a967",
        )


if __name__ == "__main__":
    unittest.main()
