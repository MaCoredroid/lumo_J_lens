#!/usr/bin/env python3
"""Focused tests for deterministic C1 SWE probe materialization."""

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
    "materialize_swe_multitask_c1_probes",
    ROOT / "scripts" / "materialize_swe_multitask_c1_probes.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
HAS_MERGED_CAPTURE = all(
    (ROOT / spec["stored_proxy_dir"] / f"chat_{spec['second_index']:04d}.json").exists()
    for spec in MODULE.CAPTURE_SPECS
)


class FakeTokenizer:
    forms = {" secret": 9001, " decoy": 9002}
    reverse = {token_id: text for text, token_id in forms.items()}

    def apply_chat_template(self, messages: object, **_: object) -> str:
        return (
            "PROMPT\n"
            + json.dumps(messages, sort_keys=True, ensure_ascii=False)
            + "\n<|im_start|>assistant\n<think>\n"
        )

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("special tokens must be disabled")
        if text in self.forms:
            return [self.forms[text]]
        if "TOKEN_ONLY" in text:
            return [9001]
        return [100 + ord(character) for character in text]

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


def shell_result(command: str, output: str, *, exit_code: int = 0) -> str:
    return (
        f"Command: {command}\nDirectory: (root)\nOutput: {output}\n"
        f"Error: (none)\nExit Code: {exit_code}\nSignal: 0\nProcess Group PGID: 123"
    )


def request_pair(
    *,
    initial_text: str = "Task owner__repo-1: repair the behavior.",
    assistant_text: str = "I will inspect the repository.",
    command: str = 'grep -r "implementation" src --include="*.py"',
    output: str = "src/module.py:class Implementation:",
    exit_code: int = 0,
) -> tuple[dict[str, object], dict[str, object]]:
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
    call_id = "call-1"
    assistant = {
        "role": "assistant",
        "content": assistant_text,
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
    tool_result = {
        "role": "tool",
        "tool_call_id": call_id,
        "content": [
            {
                "type": "text",
                "text": shell_result(command, output, exit_code=exit_code),
            }
        ],
    }
    second = {
        **copy.deepcopy(first),
        "seed": 880001235,
        "messages": [*copy.deepcopy(initial_messages), assistant, tool_result],
    }
    return first, second


def build(
    *,
    initial_text: str = "Task owner__repo-1: repair the behavior.",
    assistant_text: str = "I will inspect the repository.",
    output: str = "src/module.py:class Implementation:",
    exit_code: int = 0,
    mutate_pair: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    tokenizer = FakeTokenizer()
    first, second = request_pair(
        initial_text=initial_text,
        assistant_text=assistant_text,
        output=output,
        exit_code=exit_code,
    )
    if mutate_pair:
        second["messages"][0]["content"] = "history drift"
    rendered, token_ids, _, _ = MODULE.render_request(
        tokenizer, request=second, template="test-template"
    )
    usage = [
        {"idx": 1, "usage": {"prompt_tokens": 0}},
        {"idx": 2, "usage": {"prompt_tokens": len(token_ids)}},
    ]
    return MODULE.build_c1_bundle(
        protocol(),
        protocol_sha256="f" * 64,
        requests=[first, second],
        request_sources=[
            {"path": "chat_0001.json", "sha256": "1" * 64},
            {"path": "chat_0002.json", "sha256": "2" * 64},
        ],
        usage_records=usage,
        tokenizer=tokenizer,
        template="test-template",
        capture_manifest_sha256="3" * 64,
        usage_manifest_sha256="4" * 64,
    )


class MaterializeC1ProbesTest(unittest.TestCase):
    def test_selects_exact_second_request_after_successful_search(self) -> None:
        prompts, summary = build()
        self.assertEqual(len(prompts), 1)
        prompt = prompts[0]
        self.assertEqual(prompt["metadata"]["checkpoint"], MODULE.C1_CHECKPOINT)
        self.assertEqual(prompt["metadata"]["concepts"][0]["visibility"], "oracle_hidden")
        self.assertEqual(prompt["score_token_ids"], [9001, 9002])
        self.assertEqual(prompt["token_ids"], FakeTokenizer().encode(prompt["text"], add_special_tokens=False))
        self.assertEqual(summary["selected_prompt_count"], 1)
        self.assertTrue(summary["pair_audits"][0]["selected"])
        observation = prompt["metadata"]["observation_audit"]["observations"][0]
        self.assertTrue(observation["successful_repository_observation"])

    def test_audits_assistant_target_as_explicit_control_and_excludes_it(self) -> None:
        prompts, summary = build(assistant_text="I should change the secret implementation.")
        self.assertEqual(prompts, [])
        self.assertEqual(summary["excluded_control_count"], 1)
        target = next(
            record
            for record in summary["excluded_visibility_records"]
            if record["subject"] == "target"
        )
        self.assertEqual(target["analysis_role"], "explicit_control_excluded")
        self.assertEqual(target["visible_channels"], ["assistant_text"])

    def test_drops_visible_foil_but_keeps_hidden_primary(self) -> None:
        prompts, summary = build(output="src/module.py uses decoy here")
        self.assertEqual(len(prompts), 1)
        concept = prompts[0]["metadata"]["concepts"][0]
        self.assertEqual(concept["foils"], [])
        self.assertEqual(prompts[0]["score_token_ids"], [9001])
        self.assertEqual(summary["excluded_control_count"], 1)
        audit = prompts[0]["metadata"]["observation_audit"]
        self.assertEqual(audit["excluded_controls"][0]["visible_channels"], ["tool_outputs"])

    def test_compound_and_scored_token_exposure_are_audited(self) -> None:
        compound_prompts, compound_summary = build(
            assistant_text="Inspect secret_implementation next."
        )
        self.assertEqual(compound_prompts, [])
        compound = next(
            record
            for record in compound_summary["excluded_visibility_records"]
            if record["subject"] == "target"
        )
        self.assertEqual(
            compound["channel_evidence"]["assistant_text"][
                "case_sensitive_compound_identifier_hits"
            ],
            ["secret_implementation"],
        )

        token_prompts, token_summary = build(output="TOKEN_ONLY")
        self.assertEqual(token_prompts, [])
        token_record = next(
            record
            for record in token_summary["excluded_visibility_records"]
            if record["subject"] == "target"
        )
        self.assertEqual(
            token_record["channel_evidence"]["tool_outputs"][
                "scored_form_token_id_hits"
            ],
            [9001],
        )

    def test_scored_token_in_initial_rendered_prompt_is_excluded(self) -> None:
        prompts, summary = build(
            initial_text="Task owner__repo-1: repair TOKEN_ONLY behavior."
        )
        self.assertEqual(prompts, [])
        target = next(
            record
            for record in summary["excluded_visibility_records"]
            if record["subject"] == "target"
        )
        self.assertEqual(target["analysis_role"], "preexisting_visible_excluded")
        self.assertTrue(target["visible_in_rendered_prompt"])

    def test_rejects_false_success_and_non_preserved_pair(self) -> None:
        prompts, summary = build(output="grep: src: No such file or directory")
        self.assertEqual(prompts, [])
        self.assertEqual(
            summary["pair_audits"][0]["exclusion_reason"],
            "no_genuinely_successful_repository_read_or_search",
        )
        with self.assertRaisesRegex(ValueError, "exactly preserve"):
            build(mutate_pair=True)

    def test_shell_envelope_requires_exact_command_and_nonzero_is_not_success(self) -> None:
        prompts, summary = build(exit_code=1)
        self.assertEqual(prompts, [])
        self.assertFalse(
            summary["pair_audits"][0]["observations"][0][
                "successful_repository_observation"
            ]
        )
        with self.assertRaisesRegex(ValueError, "reproduce tool-call arguments"):
            MODULE.audit_observations(
                request_pair()[1]["messages"][2],
                [
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "content": [
                            {"type": "text", "text": shell_result("different command", "ok")}
                        ],
                    }
                ],
            )

    @unittest.skipUnless(HAS_MERGED_CAPTURE, "pinned v2/v7 merged capture is required")
    def test_real_merged_capture_hashes_and_task_order(self) -> None:
        requests, sources, usage, capture_hash, usage_hash = MODULE.load_merged_capture(
            ROOT
        )
        self.assertEqual(len(requests), 20)
        self.assertEqual(len(sources), 20)
        self.assertEqual(len(usage), 20)
        self.assertEqual(
            capture_hash,
            "62fd83a16262845475057f950d2ac32816ad93fc9bd3802b6cdbb4f66319f895",
        )
        self.assertEqual(
            usage_hash,
            "eb738f3be2f5537db78ac5fe154aa30d681bc948bf1960da65955ecd3493a967",
        )
        expected_instances = [
            spec["instance_id"] for spec in MODULE.CAPTURE_SPECS for _ in range(2)
        ]
        self.assertEqual(
            [source["capture_instance_id"] for source in sources],
            expected_instances,
        )
        self.assertEqual([record["idx"] for record in usage], list(range(1, 21)))


if __name__ == "__main__":
    unittest.main()
