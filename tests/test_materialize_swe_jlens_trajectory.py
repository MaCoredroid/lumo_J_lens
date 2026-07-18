#!/usr/bin/env python3
"""Tests for dense teacher-forced SWE trajectory materialization."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_swe_jlens_trajectory",
    ROOT / "scripts" / "materialize_swe_jlens_trajectory.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ByteTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("implicit special tokens must remain disabled")
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int], **_: object) -> str:
        return bytes(token_ids).decode("utf-8")


class MergingTokenizer(ByteTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if text == "P":
            return [80]
        if text == "Pa":
            return [80, 1]
        if text == "Pab":
            return [80, 2]
        return super().encode(text, add_special_tokens=add_special_tokens)


def trace_turn(
    request_index: int,
    *,
    thinking: str = "think\n",
    text: str = "\n\ntext\n\n",
    tools: tuple[dict[str, object], ...] = (),
) -> MODULE.TraceTurn:
    return MODULE.TraceTurn(
        request_index=request_index,
        entry_indices=(request_index * 3 - 2, request_index * 3 - 1),
        thinking_entry_index=request_index * 3 - 2,
        thinking_entry_uuid=f"thinking-{request_index}",
        completion_entry_index=request_index * 3 - 1,
        completion_entry_uuid=f"completion-{request_index}",
        thinking_text=thinking,
        visible_text=text,
        tool_uses=tools,
        usage={"input_tokens": 10, "output_tokens": 20},
        stop_reason="tool_use" if tools else None,
    )


def canonical_turn(
    request_index: int,
    completion: bytes = b"abcdefghij",
) -> MODULE.CanonicalTurn:
    trace = trace_turn(request_index)
    return MODULE.CanonicalTurn(
        request_index=request_index,
        stage_name=f"stage-{request_index}",
        prompt_text="P",
        prompt_token_ids=(80,),
        completion_text=completion.decode("ascii"),
        completion_token_ids=tuple(completion),
        trace=trace,
        source_hashes={"trace_sha256": "a" * 64},
        events={0: ("prompt_boundary",), 3: ("semantic",), len(completion) - 1: ("pre_eos",)},
        boundary_offsets={
            "thinking_end": 2,
            "think_close_end": 3,
            "visible_text_end": 5,
            "im_end_start": len(completion) - 1,
        },
        tool_spans=((5, len(completion) - 1),),
    )


class MaterializeSweTrajectoryTest(unittest.TestCase):
    def test_removes_exactly_template_added_newline(self) -> None:
        value = "generated whitespace\n<|im_end|>\n"
        self.assertEqual(
            MODULE.remove_template_trailing_newline(value),
            "generated whitespace\n<|im_end|>",
        )
        for invalid in ("<|im_end|>", "<|im_end|>\n\n", "text\n"):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "must end"):
                    MODULE.remove_template_trailing_newline(invalid)

    def test_request_nine_synthesis_removes_one_validated_thinking_newline(self) -> None:
        turn = trace_turn(9, thinking="final thought\n", text="final answer")
        message = MODULE.synthesize_final_assistant(turn)
        self.assertEqual(message["reasoning_content"], "final thought")
        self.assertEqual(message["reasoning"], "final thought")
        self.assertEqual(message["content"], "final answer")
        with self.assertRaisesRegex(ValueError, "tool calls"):
            MODULE.synthesize_final_assistant(
                trace_turn(
                    9,
                    tools=(
                        {"id": "tool-1", "name": "shell", "input": {}},
                    ),
                )
            )

    def test_extract_trace_turns_groups_split_blocks_and_two_tools(self) -> None:
        trace: list[dict[str, object]] = []
        prompt_counts = MODULE.SOURCE.EXPECTED_PROMPT_TOKENS
        completion_counts = MODULE.EXPECTED_COMPLETION_TOKENS
        for request_index in range(1, 10):
            trace.append(
                {
                    "type": "assistant",
                    "uuid": f"thinking-{request_index}",
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": f"thought {request_index}\n"}
                        ],
                        "usage": {"input_tokens": 0},
                    },
                }
            )
            trace.append(
                {
                    "type": "assistant",
                    "uuid": f"text-{request_index}",
                    "message": {
                        "content": [{"type": "text", "text": "\n\n"}],
                        "usage": {"input_tokens": 0},
                    },
                }
            )
            tool_count = 2 if request_index == 5 else (0 if request_index == 9 else 1)
            content = [
                {
                    "type": "tool_use",
                    "id": f"tool-{request_index}-{tool_index}",
                    "name": "run_shell_command",
                    "input": {"command": "true"},
                }
                for tool_index in range(tool_count)
            ]
            trace.append(
                {
                    "type": "assistant",
                    "uuid": f"completion-{request_index}",
                    "message": {
                        "content": content or [{"type": "text", "text": "done"}],
                        "usage": {
                            "input_tokens": prompt_counts[request_index - 1],
                            "output_tokens": completion_counts[request_index - 1],
                        },
                        "stop_reason": "tool_use" if content else None,
                    },
                }
            )
        # Request 9's terminal text belongs in its usage-bearing record, so remove
        # the otherwise duplicate earlier text block.
        del trace[-2]
        turns = MODULE.extract_trace_turns(trace)
        self.assertEqual(len(turns), 9)
        self.assertEqual(len(turns[4].tool_uses), 2)
        self.assertEqual(turns[8].visible_text, "done")
        self.assertEqual(turns[2].thinking_entry_uuid, "thinking-3")

    def test_appended_assistant_is_bound_to_trace_and_tool_results(self) -> None:
        tool = {
            "id": "tool-1",
            "name": "run_shell_command",
            "input": {"command": "pwd"},
        }
        turn = trace_turn(1, tools=(tool,))
        request = {"messages": [{"role": "user", "content": "task"}]}
        assistant = {
            "role": "assistant",
            "content": turn.visible_text,
            "reasoning_content": "think",
            "reasoning": "think",
            "tool_calls": [
                {
                    "id": "tool-1",
                    "type": "function",
                    "function": {
                        "name": "run_shell_command",
                        "arguments": '{"command":"pwd"}',
                    },
                }
            ],
        }
        next_request = {
            "messages": [
                *request["messages"],
                assistant,
                {"role": "tool", "tool_call_id": "tool-1", "content": "result"},
            ]
        }
        self.assertEqual(
            MODULE.appended_assistant_message(request, next_request, turn), assistant
        )
        broken = copy.deepcopy(next_request)
        broken["messages"][1]["reasoning"] = "different"
        with self.assertRaisesRegex(ValueError, "reasoning disagrees"):
            MODULE.appended_assistant_message(request, broken, turn)

    def test_character_events_fail_closed_or_snap_to_next_token_boundary(self) -> None:
        tokenizer = MergingTokenizer()
        with self.assertRaisesRegex(ValueError, "not a stable token boundary"):
            MODULE.char_boundary_token_offset(
                tokenizer,
                prompt_text="P",
                prompt_token_ids=[80],
                completion_text="ab",
                combined_token_ids=[80, 2],
                char_offset=1,
            )
        self.assertEqual(
            MODULE.char_boundary_token_offset(
                tokenizer,
                prompt_text="P",
                prompt_token_ids=[80],
                completion_text="ab",
                combined_token_ids=[80, 2],
                char_offset=1,
                snap_forward=True,
            ),
            1,
        )

    def test_selection_is_sorted_deduplicated_and_request_three_is_exhaustive(self) -> None:
        ordinary = MODULE.selected_offsets(canonical_turn(1), stride=4)
        self.assertEqual(list(ordinary), [0, 3, 4, 8, 9])
        self.assertIn("event", ordinary[3])
        self.assertIn("last_target", ordinary[9])
        exhaustive = MODULE.selected_offsets(canonical_turn(3), stride=8)
        self.assertEqual(list(exhaustive), [0, 1, 3, 8, 9])
        self.assertIn("request_3_reasoning_every_token", exhaustive[0])
        self.assertIn("request_3_reasoning_every_token", exhaustive[1])
        self.assertNotIn("request_3_reasoning_every_token", exhaustive[3])

    def test_prompt_bundle_uses_exact_prefix_and_next_target(self) -> None:
        tokenizer = ByteTokenizer()
        turn = canonical_turn(1)
        prompts = MODULE.build_prompt_bundle(
            tokenizer,
            [turn],
            provenance_id="b" * 64,
            stride=4,
            max_model_len=32,
        )
        offsets = [
            prompt["metadata"]["trajectory"]["completion_token_offset"]
            for prompt in prompts
        ]
        self.assertEqual(offsets, [0, 3, 4, 8, 9])
        offset_three = prompts[1]
        self.assertEqual(offset_three["token_ids"], [80, *b"abc"])
        self.assertEqual(offset_three["target_token_id"], ord("d"))
        self.assertEqual(
            offset_three["metadata"]["trajectory"]["target_region"],
            "visible_text",
        )
        self.assertEqual(offset_three["metadata"]["trajectory"]["offset"], 3)
        self.assertEqual(
            offset_three["metadata"]["trajectory"]["region"], "visible_text"
        )
        self.assertEqual(
            offset_three["metadata"]["trajectory"]["target_token_id"], ord("d")
        )
        self.assertEqual(
            offset_three["metadata"]["source_hashes"]["trace_sha256"], "a" * 64
        )
        with self.assertRaisesRegex(ValueError, "model slots"):
            MODULE.build_prompt_bundle(
                tokenizer,
                [turn],
                provenance_id="b" * 64,
                stride=4,
                max_model_len=4,
            )

    @unittest.skipUnless(
        (ROOT / MODULE.SOURCE.RUN_RELATIVE_PATH).exists()
        and (
            Path.home()
            / ".cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/"
            "snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404"
        ).exists(),
        "owner-only certified run and pinned tokenizer snapshot are required",
    )
    def test_owner_artifacts_reconstruct_all_pinned_completion_hashes(self) -> None:
        from transformers import AutoTokenizer

        snapshot = (
            Path.home()
            / ".cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/"
            "snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404"
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        turns, source = MODULE.load_certified_turns(
            root=ROOT,
            run_dir=ROOT / MODULE.SOURCE.RUN_RELATIVE_PATH,
            template_path=ROOT / MODULE.SOURCE.TEMPLATE_RELATIVE_PATH,
            snapshot=snapshot,
            tokenizer=tokenizer,
        )
        self.assertEqual(source["provenance_id"], MODULE.EXPECTED_PROVENANCE_ID)
        self.assertEqual(
            [len(turn.completion_token_ids) for turn in turns],
            list(MODULE.EXPECTED_COMPLETION_TOKENS),
        )
        self.assertEqual(
            [turn.source_hashes["completion_text_sha256"] for turn in turns],
            list(MODULE.EXPECTED_COMPLETION_TEXT_SHA256),
        )
        self.assertEqual(
            sum(len(MODULE.selected_offsets(turn)) for turn in turns), 293
        )
        self.assertEqual(
            turns[2].boundary_offsets["correct_identifier_prediction_boundary"],
            32,
        )


if __name__ == "__main__":
    unittest.main()
