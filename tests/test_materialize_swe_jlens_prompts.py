#!/usr/bin/env python3
"""Tests for exact certified-SWE Jacobian-Lens prompt materialization."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_swe_jlens_prompts",
    ROOT / "scripts" / "materialize_swe_jlens_prompts.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ByteTokenizer:
    def __init__(self, rendered: list[str] | None = None) -> None:
        self.rendered = list(rendered or [])

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("materializer must disable implicit special tokens")
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int], **_: object) -> str:
        return bytes(token_ids).decode("utf-8")

    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        self.last_messages = messages
        self.last_kwargs = kwargs
        return self.rendered.pop(0)


class ContextChangingTokenizer(ByteTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if text == "a":
            return [1]
        if text == "ab":
            return [2]
        return super().encode(text, add_special_tokens=add_special_tokens)


class MaterializeSwePromptsTest(unittest.TestCase):
    def test_normalizes_string_tool_arguments_without_mutating_source(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "run_shell_command",
                            "arguments": '{"command":"pwd","timeout":100}',
                        },
                    }
                ],
            }
        ]
        original = copy.deepcopy(messages)
        normalized, count = MODULE.normalize_tool_call_arguments(messages)
        self.assertEqual(count, 1)
        self.assertEqual(messages, original)
        self.assertEqual(
            normalized[0]["tool_calls"][0]["function"]["arguments"],
            {"command": "pwd", "timeout": 100},
        )

    def test_rejects_tool_arguments_that_do_not_normalize_to_object(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "tool", "arguments": "[1, 2]"}}
                ],
            }
        ]
        with self.assertRaisesRegex(ValueError, "must normalize to an object"):
            MODULE.normalize_tool_call_arguments(messages)

    def test_load_usage_requires_exact_certified_counts(self) -> None:
        records = [
            {
                "idx": index,
                "usage": {"prompt_tokens": token_count},
                "finish_reason": "stop",
            }
            for index, token_count in enumerate(MODULE.EXPECTED_PROMPT_TOKENS, 1)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "usage.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            self.assertEqual(MODULE.load_usage(path), records)
            records[4]["usage"]["prompt_tokens"] += 1
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "prompt token count"):
                MODULE.load_usage(path)

    def test_extracts_exactly_one_thinking_and_completion_per_request(self) -> None:
        trace: list[dict[str, object]] = []
        for index, token_count in enumerate(MODULE.EXPECTED_PROMPT_TOKENS, 1):
            trace.extend(
                [
                    {
                        "type": "assistant",
                        "uuid": f"thinking-{index}",
                        "message": {
                            "id": f"thinking-message-{index}",
                            "content": [
                                {"type": "thinking", "thinking": f"thought {index}"}
                            ],
                            "usage": {"input_tokens": 0},
                        },
                    },
                    {
                        "type": "assistant",
                        "uuid": f"completion-{index}",
                        "message": {
                            "id": f"completion-message-{index}",
                            "content": [{"type": "tool_use"}],
                            "usage": {"input_tokens": token_count},
                            "stop_reason": "tool_use",
                        },
                    },
                ]
            )
        thinking, completions = MODULE.extract_trace_records(trace)
        self.assertEqual(len(thinking), 9)
        self.assertEqual(len(completions), 9)
        self.assertEqual(thinking[2]["text"], "thought 3")
        self.assertEqual(completions[2]["usage"]["input_tokens"], 12743)

    def test_boundary_target_is_derived_from_full_prompt_tokenization(self) -> None:
        tokenizer = ByteTokenizer()
        prompt = "prompt\n"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        result = MODULE.boundary_continuation_ids(
            tokenizer,
            prompt,
            prompt_ids,
            "next",
        )
        self.assertEqual(result, list(b"next"))
        with self.assertRaisesRegex(ValueError, "changed the verified prompt"):
            MODULE.boundary_continuation_ids(
                ContextChangingTokenizer(),
                "a",
                [1],
                "b",
            )

    def test_stage_prompts_keep_sampled_target_only_in_metadata(self) -> None:
        suffix = "<|im_start|>assistant\n<think>\n"
        rendered = [f"stage-{index}-{suffix}" for index in range(1, 10)]
        token_counts = tuple(len(value.encode("utf-8")) for value in rendered)
        tokenizer = ByteTokenizer(rendered)
        requests = [
            {
                "messages": [{"role": "user", "content": f"request {index}"}],
                "tools": [],
                "seed": 880001233 + index,
            }
            for index in range(1, 10)
        ]
        usage = [
            {"idx": index, "usage": {"prompt_tokens": token_count}}
            for index, token_count in enumerate(token_counts, 1)
        ]
        thinking = [
            {
                "entry_index": index * 2,
                "entry_uuid": f"thinking-{index}",
                "text": f"thought {index}",
            }
            for index in range(1, 10)
        ]
        completions = [
            {
                "entry_index": index * 2 + 1,
                "entry_uuid": f"completion-{index}",
                "usage": {"input_tokens": token_count},
            }
            for index, token_count in enumerate(token_counts, 1)
        ]
        with mock.patch.object(MODULE, "EXPECTED_PROMPT_TOKENS", token_counts):
            prompts = MODULE.build_stage_prompts(
                tokenizer,
                requests=requests,
                request_sources=[{"path": f"chat-{index}"} for index in range(9)],
                usage_records=usage,
                thinking_records=thinking,
                completion_records=completions,
                template="template",
                provenance_id="a" * 64,
            )
        self.assertEqual(len(prompts), 9)
        self.assertTrue(all("target_token_id" not in prompt for prompt in prompts))
        self.assertEqual(
            prompts[0]["metadata"]["sampled_next"]["first_token_id"], ord("t")
        )
        self.assertEqual(
            prompts[0]["metadata"]["sampled_next"]["original_next_text"],
            "thought 1",
        )
        self.assertTrue(tokenizer.last_kwargs["add_generation_prompt"])
        self.assertTrue(tokenizer.last_kwargs["enable_thinking"])

    def test_candidate_probes_emit_every_teacher_forced_sequence_step(self) -> None:
        tokenizer = ByteTokenizer()
        stage_prompts: list[dict[str, object]] = []
        for index in range(1, 4):
            text = f"stage-{index}<think>\n"
            thought = "irrelevant"
            if index == 3:
                thought = (
                    "I can see the bug. It checks `cotm`, but the variable is actually "
                    "`cothm` and should use the defined value."
                )
            stage_prompts.append(
                {
                    "id": f"stage-{index}",
                    "text": text,
                    "token_ids": tokenizer.encode(text, add_special_tokens=False),
                    "metadata": {"sampled_next": {"original_next_text": thought}},
                }
            )
        probes = MODULE.build_candidate_probes(
            tokenizer,
            stage_prompts=stage_prompts,
            provenance_id="b" * 64,
        )
        self.assertEqual(len(probes), len("cothm") + len("cotm"))
        groups: dict[str, list[dict[str, object]]] = {}
        for probe in probes:
            self.assertIn("target_token_id", probe)
            candidate = probe["metadata"]["candidate"]["identifier"]
            groups.setdefault(candidate, []).append(probe)
        self.assertEqual(set(groups), {"cothm", "cotm"})
        for candidate, candidate_probes in groups.items():
            self.assertEqual(len(candidate_probes), len(candidate))
            for step_index, probe in enumerate(candidate_probes):
                step = probe["metadata"]["step"]
                self.assertEqual(step["index"], step_index)
                self.assertEqual(step["count"], len(candidate))
                self.assertEqual(
                    step["teacher_forced_candidate_prefix"], candidate[:step_index]
                )
                self.assertEqual(probe["target_token_id"], ord(candidate[step_index]))


if __name__ == "__main__":
    unittest.main()
