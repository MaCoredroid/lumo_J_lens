#!/usr/bin/env python3
"""Materialize dense teacher-forced prompts from the certified Qwen Code trace."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import materialize_swe_jlens_prompts as SOURCE  # noqa: E402


DEFAULT_OUTPUT = Path(
    ".cache/swe_jlens_trajectory/swe_jlens_trajectory_prompts.json"
)
DEFAULT_STRIDE = 8
DEFAULT_MAX_MODEL_LEN = 16384
EXPECTED_PROVENANCE_ID = (
    "72da18d1cead29ce7c4fe2627608040599c5f43c9e2b589855f89efa39afe038"
)
EXPECTED_COMPLETION_TOKENS = (172, 119, 195, 150, 175, 215, 143, 117, 153)
EXPECTED_COMPLETION_TEXT_SHA256 = (
    "e208f572ed4507893b5c97cac7cabedcd39ffda84019e45a0bee3aec822b54a9",
    "2518c39df625fd64da0aa3007e9c752521c04b92dba531624101a7d7520a6da1",
    "60f5a346b8204ed4041c156ffaf31ea2f95f3ee06056891585ed823751f031f6",
    "271588583eaa1365f4930907d11ce082b970ba01431781ff0ccbd988bde3f8f0",
    "afeaa2c1c056b87471fead3bc0935bdf47a280eb590c941ab3aebb0bc29b0427",
    "33aad60377bfeb78674e11d15833b2e9d2226725d1565b04ae07e0aa0dd54518",
    "bb6b2f1e2ad8928bf766e9e518de8c6a1d62a23920565f0b8cd69b53e6c3dd54",
    "5e7a99d9f656e3b96f32fef406a64d4e2cdd4e2ba58a87dd928f279186630b54",
    "7ee77b5bd438521b6b920d9b12bbd0d375c29c049f09488bb0dee4e3708c6aab",
)
EXPECTED_COMPLETION_TOKEN_IDS_SHA256 = (
    "7ad12bd715de8d15fa07682f2d48cc3fc750f259fecdaed49d6b01b5d29b6300",
    "00e8009e11cb5d648e833c4af7b001f2a07668711e583519f1f0e53ff5b593d0",
    "d4f04855a874c88a238f47a433e53bbb77c5eb400525f581ec2547898311e03b",
    "bd9a67319221f44d410c02d8efba1b7cea68d73e611bd01ce0106c37df6a6247",
    "6fe46b13a8401b3871c2a115369a043b50fa44a5b1de9ceaddb0864b8181cc24",
    "6082e56b373ed8999f453ff516ac0d9aedf8b32e0b0d8450a4ab23c6d6783e2a",
    "7850129d64b34013014e559bc3244d2a3cd3b204fc6be96b5b1b144caedf2e5d",
    "9aaf35431c9ba257d04421f8fa5b4bcd5a6c94e4b0c9d8df2395371543239646",
    "b2def4912e5ccb2f007ba0e27da26e32a5936e7213f45240964a5ed55a07d80d",
)

# Each literal must occur exactly once in that request's thinking block. The
# event is the state after the literal, immediately before its next token.
SEMANTIC_EVENTS: dict[int, tuple[tuple[str, str], ...]] = {
    1: (("diagnosis_named", "NameError: name 'cotm' is not defined"),),
    2: (("source_location_reaffirmed", "line 590"),),
    3: (
        ("bug_recognized", "I can see the bug clearly."),
        ("correct_identifier_prediction_boundary", "but the variable is actually `"),
        ("correct_identifier_named", "cothm` (defined on line 589)"),
        ("reproduction_planned", "Let me also reproduce the issue first to confirm"),
    ),
    4: (
        ("failure_confirmed", "Bug confirmed."),
        ("patch_target_named", "`cothm` on line 590."),
    ),
    5: (("fix_verification_planned", "Let me verify the fix"),),
    6: (
        ("fix_working", "The fix works."),
        (
            "original_reproduction_passed",
            "original reproduction case now works without error.",
        ),
    ),
    7: (("broader_values_passed", "All values work without error now."),),
    8: (("pytest_unavailable", "pytest is not installed."),),
    9: (
        ("focused_test_passed", "The test passed"),
        ("task_resolved", "The bug report is now resolved."),
    ),
}


@dataclass(frozen=True)
class TraceTurn:
    request_index: int
    entry_indices: tuple[int, ...]
    thinking_entry_index: int
    thinking_entry_uuid: str
    completion_entry_index: int
    completion_entry_uuid: str
    thinking_text: str
    visible_text: str
    tool_uses: tuple[dict[str, Any], ...]
    usage: dict[str, Any]
    stop_reason: str | None


@dataclass(frozen=True)
class CanonicalTurn:
    request_index: int
    stage_name: str
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    completion_text: str
    completion_token_ids: tuple[int, ...]
    trace: TraceTurn
    source_hashes: dict[str, str]
    events: dict[int, tuple[str, ...]]
    boundary_offsets: dict[str, int]
    tool_spans: tuple[tuple[int, int], ...]


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def extract_trace_turns(value: Any) -> list[TraceTurn]:
    """Group the split thinking/text/tool records into nine accepted turns."""

    if not isinstance(value, list):
        raise ValueError("qwen_trace.json must contain a JSON list")
    pending: list[tuple[int, dict[str, Any]]] = []
    turns: list[TraceTurn] = []
    for entry_index, raw_entry in enumerate(value):
        if not isinstance(raw_entry, dict) or raw_entry.get("type") != "assistant":
            continue
        entry = raw_entry
        message = _require_object(entry.get("message"), f"trace entry {entry_index} message")
        content = message.get("content")
        if not isinstance(content, list):
            raise ValueError(f"trace entry {entry_index} content must be a list")
        pending.append((entry_index, entry))
        usage = message.get("usage")
        if not isinstance(usage, dict) or usage.get("input_tokens", 0) <= 0:
            continue

        thinking: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        text_items: list[dict[str, Any]] = []
        tools: list[dict[str, Any]] = []
        for pending_index, pending_entry in pending:
            pending_message = _require_object(
                pending_entry.get("message"), f"trace entry {pending_index} message"
            )
            for raw_item in pending_message.get("content", []):
                item = _require_object(raw_item, f"trace entry {pending_index} content item")
                item_type = item.get("type")
                if item_type == "thinking":
                    thinking.append((pending_index, pending_entry, item))
                elif item_type == "text":
                    text_items.append(item)
                elif item_type == "tool_use":
                    tools.append(copy.deepcopy(item))
                else:
                    raise ValueError(
                        f"trace entry {pending_index} has unsupported content type "
                        f"{item_type!r}"
                    )
        if len(thinking) != 1 or len(text_items) != 1:
            raise ValueError(
                f"trace turn {len(turns) + 1} must contain one thinking and one text block"
            )
        thought_index, thought_entry, thought_item = thinking[0]
        thinking_text = _require_string(
            thought_item.get("thinking"), f"trace turn {len(turns) + 1} thinking"
        )
        if not thinking_text or not thinking_text.endswith("\n"):
            raise ValueError("certified thinking text must end in exactly one newline")
        if thinking_text.endswith("\n\n"):
            raise ValueError("certified thinking text has an ambiguous trailing newline")
        visible_text = _require_string(
            text_items[0].get("text"), f"trace turn {len(turns) + 1} text"
        )
        for tool_index, tool in enumerate(tools, 1):
            if not isinstance(tool.get("id"), str) or not tool["id"]:
                raise ValueError(f"trace tool {tool_index} has no id")
            if not isinstance(tool.get("name"), str) or not tool["name"]:
                raise ValueError(f"trace tool {tool_index} has no name")
            if not isinstance(tool.get("input"), dict):
                raise ValueError(f"trace tool {tool_index} input must be an object")
        turns.append(
            TraceTurn(
                request_index=len(turns) + 1,
                entry_indices=tuple(index for index, _ in pending),
                thinking_entry_index=thought_index,
                thinking_entry_uuid=_require_string(
                    thought_entry.get("uuid"), "thinking entry UUID"
                ),
                completion_entry_index=entry_index,
                completion_entry_uuid=_require_string(
                    entry.get("uuid"), "completion entry UUID"
                ),
                thinking_text=thinking_text,
                visible_text=visible_text,
                tool_uses=tuple(tools),
                usage=copy.deepcopy(usage),
                stop_reason=message.get("stop_reason"),
            )
        )
        pending = []
    if pending:
        raise ValueError("qwen trace ends with an incomplete assistant turn")
    if len(turns) != 9:
        raise ValueError(f"certified trace must contain exactly 9 turns, found {len(turns)}")
    return turns


def synthesize_final_assistant(turn: TraceTurn) -> dict[str, Any]:
    """Build request 9's absent history message without normalizing its content."""

    if turn.request_index != 9:
        raise ValueError("only request 9 may synthesize a terminal assistant message")
    if turn.tool_uses:
        raise ValueError("terminal request 9 unexpectedly contains tool calls")
    reasoning = turn.thinking_text[:-1]  # Remove the one validated trace newline.
    return {
        "role": "assistant",
        "content": turn.visible_text,
        "reasoning_content": reasoning,
        "reasoning": reasoning,
    }


def _assistant_tool_records(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ValueError("assistant tool_calls must be a list")
    records: list[dict[str, Any]] = []
    for index, raw_call in enumerate(calls, 1):
        call = _require_object(raw_call, f"assistant tool call {index}")
        function = call.get("function", call)
        function = _require_object(function, f"assistant tool call {index} function")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"assistant tool call {index} has invalid JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError(f"assistant tool call {index} arguments must be an object")
        records.append(
            {
                "id": call.get("id"),
                "name": function.get("name"),
                "input": arguments,
            }
        )
    return records


def appended_assistant_message(
    request: Mapping[str, Any],
    next_request: Mapping[str, Any],
    turn: TraceTurn,
) -> dict[str, Any]:
    current_messages = request.get("messages")
    next_messages = next_request.get("messages")
    if not isinstance(current_messages, list) or not isinstance(next_messages, list):
        raise ValueError("requests must contain message lists")
    if next_messages[: len(current_messages)] != current_messages:
        raise ValueError(f"request {turn.request_index + 1} does not preserve history")
    appended = next_messages[len(current_messages) :]
    expected_added = 1 + len(turn.tool_uses)
    if len(appended) != expected_added:
        raise ValueError(
            f"request {turn.request_index + 1} appended {len(appended)} messages; "
            f"expected {expected_added}"
        )
    assistant = _require_object(appended[0], "appended assistant message")
    if assistant.get("role") != "assistant":
        raise ValueError("next request does not append an assistant message first")
    reasoning = turn.thinking_text[:-1]
    if assistant.get("reasoning_content") != reasoning or assistant.get("reasoning") != reasoning:
        raise ValueError("appended assistant reasoning disagrees with qwen trace")
    if assistant.get("content") != turn.visible_text:
        raise ValueError("appended assistant text disagrees with qwen trace")
    observed_tools = _assistant_tool_records(assistant)
    expected_tools = [
        {"id": tool["id"], "name": tool["name"], "input": tool["input"]}
        for tool in turn.tool_uses
    ]
    if observed_tools != expected_tools:
        raise ValueError("appended assistant tool calls disagree with qwen trace")
    result_ids = []
    for index, tool_result in enumerate(appended[1:], 1):
        result = _require_object(tool_result, f"appended tool result {index}")
        if result.get("role") != "tool":
            raise ValueError("assistant message must be followed only by tool results")
        result_ids.append(result.get("tool_call_id"))
    if result_ids != [tool["id"] for tool in turn.tool_uses]:
        raise ValueError("tool-result IDs disagree with assistant tool calls")
    return copy.deepcopy(assistant)


def remove_template_trailing_newline(rendered_suffix: str) -> str:
    """Remove only Jinja's history newline, preserving generated whitespace."""

    terminal = "<|im_end|>\n"
    if not rendered_suffix.endswith(terminal):
        raise ValueError("canonical assistant rendering must end with '<|im_end|>\\n'")
    completion = rendered_suffix[:-1]
    if not completion.endswith("<|im_end|>"):
        raise AssertionError("one-character newline removal corrupted the terminal token")
    return completion


def char_boundary_token_offset(
    tokenizer: Any,
    *,
    prompt_text: str,
    prompt_token_ids: Sequence[int],
    completion_text: str,
    combined_token_ids: Sequence[int],
    char_offset: int,
    snap_forward: bool = False,
) -> int:
    if not 0 <= char_offset <= len(completion_text):
        raise ValueError("completion character offset is out of range")
    for candidate in range(char_offset, len(completion_text) + 1):
        prefix_ids = SOURCE.encode_exact(
            tokenizer, prompt_text + completion_text[:candidate]
        )
        if list(combined_token_ids[: len(prefix_ids)]) == prefix_ids:
            result = len(prefix_ids) - len(prompt_token_ids)
            if result < 0:
                raise ValueError("completion boundary precedes the certified prompt")
            return result
        if not snap_forward:
            break
    qualifier = " at or after" if snap_forward else ""
    raise ValueError(
        f"completion character offset{qualifier} {char_offset} is not a stable "
        "token boundary"
    )


def build_event_boundaries(
    tokenizer: Any,
    *,
    prompt_text: str,
    prompt_token_ids: Sequence[int],
    completion_text: str,
    combined_token_ids: Sequence[int],
    turn: TraceTurn,
    semantic_events: Mapping[int, Sequence[tuple[str, str]]] = SEMANTIC_EVENTS,
) -> tuple[dict[int, tuple[str, ...]], dict[str, int], tuple[tuple[int, int], ...]]:
    prefix = turn.thinking_text + "</think>" + turn.visible_text
    if not completion_text.startswith(prefix):
        raise ValueError("canonical completion disagrees with trace thinking/text blocks")
    if not completion_text.endswith("<|im_end|>"):
        raise ValueError("canonical completion has no terminal token")

    char_events: list[tuple[str, int, bool]] = [
        ("prompt_boundary", 0, False),
        ("thinking_end", len(turn.thinking_text), False),
        ("think_close_end", len(turn.thinking_text) + len("</think>"), False),
        ("visible_text_end", len(prefix), False),
    ]
    tool_char_spans: list[tuple[int, int]] = []
    cursor = len(prefix)
    for tool_number in range(1, len(turn.tool_uses) + 1):
        start = completion_text.find("<tool_call>", cursor)
        if start < 0:
            raise ValueError(f"canonical completion omits tool call {tool_number}")
        end_marker = completion_text.find("</tool_call>", start)
        if end_marker < 0:
            raise ValueError(f"canonical completion does not close tool call {tool_number}")
        end = end_marker + len("</tool_call>")
        tool_char_spans.append((start, end))
        char_events.extend(
            (
                (f"tool_{tool_number}_start", start, False),
                (f"tool_{tool_number}_end", end, False),
            )
        )
        cursor = end
    if completion_text.find("<tool_call>", cursor) >= 0:
        raise ValueError("canonical completion contains unbound tool-call markup")
    im_end_start = len(completion_text) - len("<|im_end|>")
    char_events.extend(
        (("im_end_start", im_end_start, False), ("pre_eos", im_end_start, False))
    )

    for label, literal in semantic_events.get(turn.request_index, ()):
        if turn.thinking_text.count(literal) != 1:
            raise ValueError(
                f"semantic event {label!r} must occur once in request "
                f"{turn.request_index} thinking"
            )
        char_events.append(
            (label, turn.thinking_text.index(literal) + len(literal), True)
        )

    token_events: dict[int, list[str]] = {}
    boundary_offsets: dict[str, int] = {}
    for label, char_offset, snap_forward in char_events:
        token_offset = char_boundary_token_offset(
            tokenizer,
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            completion_text=completion_text,
            combined_token_ids=combined_token_ids,
            char_offset=char_offset,
            snap_forward=snap_forward,
        )
        boundary_offsets[label] = token_offset
        token_events.setdefault(token_offset, []).append(label)
    tool_token_spans = tuple(
        (
            char_boundary_token_offset(
                tokenizer,
                prompt_text=prompt_text,
                prompt_token_ids=prompt_token_ids,
                completion_text=completion_text,
                combined_token_ids=combined_token_ids,
                char_offset=start,
            ),
            char_boundary_token_offset(
                tokenizer,
                prompt_text=prompt_text,
                prompt_token_ids=prompt_token_ids,
                completion_text=completion_text,
                combined_token_ids=combined_token_ids,
                char_offset=end,
            ),
        )
        for start, end in tool_char_spans
    )
    return (
        {offset: tuple(labels) for offset, labels in sorted(token_events.items())},
        boundary_offsets,
        tool_token_spans,
    )


def reconstruct_canonical_turn(
    tokenizer: Any,
    *,
    request_index: int,
    request: Mapping[str, Any],
    next_request: Mapping[str, Any] | None,
    usage_record: Mapping[str, Any],
    trace_turn: TraceTurn,
    template: str,
    source_hashes: Mapping[str, str],
    enforce_certified_hashes: bool = True,
    semantic_events: Mapping[int, Sequence[tuple[str, str]]] = SEMANTIC_EVENTS,
) -> CanonicalTurn:
    if trace_turn.request_index != request_index:
        raise ValueError("request and trace indices disagree")
    prompt_text, prompt_ids, _, _ = SOURCE.render_request(
        tokenizer, request=request, template=template
    )
    if enforce_certified_hashes and len(prompt_ids) != SOURCE.EXPECTED_PROMPT_TOKENS[
        request_index - 1
    ]:
        raise ValueError(f"request {request_index} prompt token count changed")

    if request_index < 9:
        if next_request is None:
            raise ValueError(f"request {request_index} requires its following request")
        assistant = appended_assistant_message(request, next_request, trace_turn)
    else:
        if next_request is not None:
            raise ValueError("terminal request 9 must not have a following request")
        assistant = synthesize_final_assistant(trace_turn)

    normalized_messages, _ = SOURCE.normalize_tool_call_arguments(
        [*request["messages"], assistant]
    )
    rendered_history = tokenizer.apply_chat_template(
        normalized_messages,
        tools=request["tools"],
        chat_template=template,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    if not isinstance(rendered_history, str) or not rendered_history.startswith(prompt_text):
        raise ValueError("canonical assistant history does not preserve its rendered prompt")
    completion_text = remove_template_trailing_newline(
        rendered_history[len(prompt_text) :]
    )
    combined_ids = SOURCE.encode_exact(tokenizer, prompt_text + completion_text)
    if combined_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("canonical completion changed the certified prompt token prefix")
    completion_ids = combined_ids[len(prompt_ids) :]
    if not completion_ids:
        raise ValueError("canonical completion contains no tokens")

    usage = _require_object(usage_record.get("usage"), "usage record usage")
    expected_completion_tokens = usage.get("completion_tokens")
    if expected_completion_tokens != len(completion_ids):
        raise ValueError(
            f"request {request_index} reconstructed {len(completion_ids)} completion "
            f"tokens; usage recorded {expected_completion_tokens}"
        )
    if usage.get("prompt_tokens") != len(prompt_ids):
        raise ValueError(f"request {request_index} usage prompt count changed")
    if usage.get("total_tokens") != len(prompt_ids) + len(completion_ids):
        raise ValueError(f"request {request_index} usage total is inconsistent")
    if trace_turn.usage.get("input_tokens") != len(prompt_ids):
        raise ValueError(f"request {request_index} trace input count changed")
    if trace_turn.usage.get("output_tokens") != len(completion_ids):
        raise ValueError(f"request {request_index} trace output count changed")
    expected_finish = "stop" if request_index == 9 else "tool_calls"
    if usage_record.get("finish_reason") != expected_finish:
        raise ValueError(f"request {request_index} finish reason changed")

    text_sha256 = SOURCE.sha256_bytes(completion_text.encode("utf-8"))
    token_ids_sha256 = SOURCE.sha256_json(completion_ids)
    if enforce_certified_hashes:
        expected_count = EXPECTED_COMPLETION_TOKENS[request_index - 1]
        if len(completion_ids) != expected_count:
            raise ValueError(f"request {request_index} completion token count changed")
        if text_sha256 != EXPECTED_COMPLETION_TEXT_SHA256[request_index - 1]:
            raise ValueError(f"request {request_index} canonical completion text changed")
        if token_ids_sha256 != EXPECTED_COMPLETION_TOKEN_IDS_SHA256[request_index - 1]:
            raise ValueError(f"request {request_index} canonical completion tokens changed")
        if completion_ids[-1] != SOURCE.EXPECTED_EOS_TOKEN_ID:
            raise ValueError(f"request {request_index} no longer ends in <|im_end|>")

    events, boundary_offsets, tool_spans = build_event_boundaries(
        tokenizer,
        prompt_text=prompt_text,
        prompt_token_ids=prompt_ids,
        completion_text=completion_text,
        combined_token_ids=combined_ids,
        turn=trace_turn,
        semantic_events=semantic_events,
    )
    hashes = dict(source_hashes)
    hashes.update(
        {
            "rendered_prompt_sha256": SOURCE.sha256_bytes(prompt_text.encode("utf-8")),
            "prompt_token_ids_sha256": SOURCE.sha256_json(prompt_ids),
            "completion_text_sha256": text_sha256,
            "completion_token_ids_sha256": token_ids_sha256,
        }
    )
    return CanonicalTurn(
        request_index=request_index,
        stage_name=SOURCE.STAGE_NAMES[request_index - 1],
        prompt_text=prompt_text,
        prompt_token_ids=tuple(prompt_ids),
        completion_text=completion_text,
        completion_token_ids=tuple(completion_ids),
        trace=trace_turn,
        source_hashes=hashes,
        events=events,
        boundary_offsets=boundary_offsets,
        tool_spans=tool_spans,
    )


def selected_offsets(
    turn: CanonicalTurn, *, stride: int = DEFAULT_STRIDE
) -> dict[int, tuple[str, ...]]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    count = len(turn.completion_token_ids)
    reasons: dict[int, set[str]] = {}

    def add(offset: int, reason: str) -> None:
        if not 0 <= offset < count:
            raise ValueError(f"selected completion offset {offset} has no target token")
        reasons.setdefault(offset, set()).add(reason)

    for offset in range(0, count, stride):
        add(offset, f"stride_{stride}")
    for offset in turn.events:
        add(offset, "event")
    add(count - 1, "last_target")
    if turn.request_index == 3:
        for offset in range(turn.boundary_offsets["thinking_end"]):
            add(offset, "request_3_reasoning_every_token")
    return {
        offset: tuple(sorted(offset_reasons))
        for offset, offset_reasons in sorted(reasons.items())
    }


def target_region(turn: CanonicalTurn, offset: int) -> str:
    boundaries = turn.boundary_offsets
    if offset < boundaries["thinking_end"]:
        return "reasoning"
    if offset < boundaries["think_close_end"]:
        return "think_close"
    if offset < boundaries["visible_text_end"]:
        return "visible_text"
    if offset >= boundaries["im_end_start"]:
        return "terminal"
    for tool_number, (start, end) in enumerate(turn.tool_spans, 1):
        if start <= offset < end:
            return f"tool_call_{tool_number}"
        if tool_number > 1 and turn.tool_spans[tool_number - 2][1] <= offset < start:
            return "inter_tool_separator"
    return "assistant_separator"


def build_prompt_bundle(
    tokenizer: Any,
    turns: Sequence[CanonicalTurn],
    *,
    provenance_id: str,
    stride: int = DEFAULT_STRIDE,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
) -> list[dict[str, Any]]:
    if max_model_len <= 1:
        raise ValueError("max model length must exceed one token")
    prompts: list[dict[str, Any]] = []
    for turn in turns:
        offsets = selected_offsets(turn, stride=stride)
        for offset, selection_reasons in offsets.items():
            prefix_ids = [
                *turn.prompt_token_ids,
                *turn.completion_token_ids[:offset],
            ]
            if len(prefix_ids) + 1 > max_model_len:
                raise ValueError(
                    f"request {turn.request_index} offset {offset} needs "
                    f"{len(prefix_ids) + 1} model slots; max is {max_model_len}"
                )
            target_token_id = turn.completion_token_ids[offset]
            region = target_region(turn, offset)
            target_text = tokenizer.decode(
                [target_token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            prompts.append(
                {
                    "id": (
                        f"swe-sympy-13480-request-{turn.request_index:02d}-"
                        f"offset-{offset:04d}"
                    ),
                    "token_ids": prefix_ids,
                    "target_token_id": target_token_id,
                    "metadata": {
                        "kind": "certified_swe_teacher_forced_trajectory",
                        "provenance_id": provenance_id,
                        "request_index": turn.request_index,
                        "stage_name": turn.stage_name,
                        "stage": {
                            "request_index": turn.request_index,
                            "name": turn.stage_name,
                        },
                        "trajectory": {
                            "offset": offset,
                            "region": region,
                            "target_token_id": target_token_id,
                            "completion_token_offset": offset,
                            "completion_token_count": len(turn.completion_token_ids),
                            "absolute_target_token_position": len(prefix_ids),
                            "state_token_position": len(prefix_ids) - 1,
                            "target_region": region,
                            "target_token_text": target_text,
                            "events": list(turn.events.get(offset, ())),
                            "selection_reasons": list(selection_reasons),
                        },
                        "trace": {
                            "assistant_entry_indices": list(turn.trace.entry_indices),
                            "thinking_entry_index": turn.trace.thinking_entry_index,
                            "thinking_entry_uuid": turn.trace.thinking_entry_uuid,
                            "completion_entry_index": turn.trace.completion_entry_index,
                            "completion_entry_uuid": turn.trace.completion_entry_uuid,
                        },
                        "source_hashes": copy.deepcopy(turn.source_hashes),
                    },
                }
            )
    prompts.sort(
        key=lambda prompt: (
            prompt["metadata"]["request_index"],
            prompt["metadata"]["trajectory"]["completion_token_offset"],
        )
    )
    if len({prompt["id"] for prompt in prompts}) != len(prompts):
        raise AssertionError("trajectory prompt IDs are not unique")
    return prompts


def load_certified_turns(
    *,
    root: Path,
    run_dir: Path,
    template_path: Path,
    snapshot: Path,
    tokenizer: Any,
) -> tuple[list[CanonicalTurn], dict[str, Any]]:
    proxy_dir = run_dir / "proxy_dumps"
    usage_path = proxy_dir / "usage.jsonl"
    trace_path = run_dir / SOURCE.TRACE_RELATIVE_PATH
    SOURCE.require_file_hash(usage_path, SOURCE.EXPECTED_USAGE_SHA256, label="usage.jsonl")
    SOURCE.require_file_hash(
        trace_path, SOURCE.EXPECTED_TRACE_SHA256, label="qwen_trace.json"
    )
    SOURCE.require_file_hash(
        template_path, SOURCE.EXPECTED_TEMPLATE_SHA256, label=template_path.name
    )

    requests: list[dict[str, Any]] = []
    request_sources: list[dict[str, Any]] = []
    for index, expected_hash in enumerate(SOURCE.EXPECTED_CHAT_SHA256, 1):
        path = proxy_dir / f"chat_{index:04d}.json"
        SOURCE.require_file_hash(path, expected_hash, label=path.name)
        request = SOURCE.validate_request(
            json.loads(path.read_text(encoding="utf-8")), request_index=index
        )
        requests.append(request)
        request_sources.append(
            SOURCE.file_record(path, display_path=str(path.relative_to(root)))
        )
    usage_records = SOURCE.load_usage(usage_path)
    trace_turns = extract_trace_turns(json.loads(trace_path.read_text(encoding="utf-8")))
    template = template_path.read_text(encoding="utf-8")
    tokenizer_source = SOURCE.validate_tokenizer(tokenizer, snapshot)
    usage_source = SOURCE.file_record(
        usage_path, display_path=str(usage_path.relative_to(root))
    )
    trace_source = SOURCE.file_record(
        trace_path, display_path=str(trace_path.relative_to(root))
    )
    template_source = SOURCE.file_record(
        template_path, display_path=str(template_path.relative_to(root))
    )
    contract = SOURCE.source_contract(
        request_sources=request_sources,
        usage_source=usage_source,
        trace_source=trace_source,
        template_source=template_source,
        tokenizer_source=tokenizer_source,
    )
    provenance_id = SOURCE.sha256_json(contract)
    if provenance_id != EXPECTED_PROVENANCE_ID:
        raise ValueError(
            f"prompt source provenance changed: expected {EXPECTED_PROVENANCE_ID}, "
            f"got {provenance_id}"
        )

    common_hashes = {
        "usage_sha256": usage_source["sha256"],
        "trace_sha256": trace_source["sha256"],
        "chat_template_sha256": template_source["sha256"],
        "tokenizer_json_sha256": tokenizer_source["files"]["tokenizer.json"]["sha256"],
    }
    turns = [
        reconstruct_canonical_turn(
            tokenizer,
            request_index=index,
            request=requests[index - 1],
            next_request=requests[index] if index < 9 else None,
            usage_record=usage_records[index - 1],
            trace_turn=trace_turns[index - 1],
            template=template,
            source_hashes={
                **common_hashes,
                "request_sha256": request_sources[index - 1]["sha256"],
            },
        )
        for index in range(1, 10)
    ]
    source_summary = {
        "provenance_id": provenance_id,
        "source_contract_sha256": provenance_id,
        "episode": contract["episode"],
        "model": {
            "repo": SOURCE.MODEL_REPO,
            "revision": SOURCE.MODEL_REVISION,
        },
        "hashes": {
            **common_hashes,
            "requests": [record["sha256"] for record in request_sources],
        },
    }
    return turns, source_summary


def build_summary(
    turns: Sequence[CanonicalTurn],
    prompts: Sequence[Mapping[str, Any]],
    *,
    source_summary: Mapping[str, Any],
    stride: int,
    max_model_len: int,
) -> dict[str, Any]:
    by_request = []
    for turn in turns:
        selected = [
            prompt
            for prompt in prompts
            if prompt["metadata"]["request_index"] == turn.request_index
        ]
        by_request.append(
            {
                "request_index": turn.request_index,
                "stage_name": turn.stage_name,
                "prompt_token_count": len(turn.prompt_token_ids),
                "completion_token_count": len(turn.completion_token_ids),
                "selected_prompt_count": len(selected),
                "completion_text_sha256": turn.source_hashes[
                    "completion_text_sha256"
                ],
                "completion_token_ids_sha256": turn.source_hashes[
                    "completion_token_ids_sha256"
                ],
                "event_offsets": {
                    label: offset
                    for label, offset in sorted(turn.boundary_offsets.items())
                },
            }
        )
    return {
        "schema_version": 1,
        "kind": "certified_swe_teacher_forced_trajectory_summary",
        "source": copy.deepcopy(dict(source_summary)),
        "selection": {
            "stride": stride,
            "request_3_reasoning_every_token": True,
            "include_all_events": True,
            "include_last_target": True,
            "max_model_len": max_model_len,
        },
        "task_request_count": len(turns),
        "original_completion_token_count": sum(
            len(turn.completion_token_ids) for turn in turns
        ),
        "trajectory_prompt_count": len(prompts),
        "maximum_prompt_token_count": max(len(prompt["token_ids"]) for prompt in prompts),
        "requests": by_request,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="permit the pinned Hugging Face download when not already cached",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")
    if args.max_model_len <= 1:
        raise SystemExit("--max-model-len must exceed one")
    root = args.root.resolve(strict=True)
    run_dir = (args.run_dir or (root / SOURCE.RUN_RELATIVE_PATH)).resolve(strict=True)
    template_path = (
        args.template or (root / SOURCE.TEMPLATE_RELATIVE_PATH)
    ).resolve(strict=True)
    output = (args.output or (root / DEFAULT_OUTPUT)).resolve()
    summary_output = (
        args.summary_output
        or output.with_name(f"{output.stem}.summary.json")
    ).resolve()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if args.model_snapshot is None:
        snapshot = Path(
            snapshot_download(
                SOURCE.MODEL_REPO,
                revision=SOURCE.MODEL_REVISION,
                local_files_only=not args.allow_download,
            )
        ).resolve(strict=True)
    else:
        snapshot = args.model_snapshot.resolve(strict=True)
    if snapshot.name != SOURCE.MODEL_REVISION:
        raise ValueError(
            f"model snapshot must end in pinned revision {SOURCE.MODEL_REVISION}"
        )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    turns, source_summary = load_certified_turns(
        root=root,
        run_dir=run_dir,
        template_path=template_path,
        snapshot=snapshot,
        tokenizer=tokenizer,
    )
    prompts = build_prompt_bundle(
        tokenizer,
        turns,
        provenance_id=source_summary["provenance_id"],
        stride=args.stride,
        max_model_len=args.max_model_len,
    )
    summary = build_summary(
        turns,
        prompts,
        source_summary=source_summary,
        stride=args.stride,
        max_model_len=args.max_model_len,
    )
    if not args.dry_run:
        SOURCE.atomic_write_json(output, prompts)
        summary["output"] = SOURCE.file_record(output, display_path=str(output))
        SOURCE.atomic_write_json(summary_output, summary)
        print(
            f"wrote {output} ({len(prompts)} prompts, "
            f"sha256={summary['output']['sha256']})"
        )
        print(
            f"wrote {summary_output} "
            f"(sha256={SOURCE.sha256_file(summary_output)})"
        )
    print(json.dumps(summary, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
