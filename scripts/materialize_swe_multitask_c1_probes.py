#!/usr/bin/env python3
"""Materialize leakage-audited post-observation (C1) SWE probe prompts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import materialize_swe_jlens_prompts as RENDER
import materialize_swe_multitask_initial_probes as C0


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/swe_multitask_initial_protocol.json"
DEFAULT_TEMPLATE = ROOT / "configs/qwen3-openai-codex.jinja"
DEFAULT_OUTPUT = ROOT / ".cache/swe_multitask_c1/prompts.json"
EXPECTED_PROTOCOL_SHA256 = (
    "9a7ac3d1206f83c1c964245f8f3ed2824cbfcd471b17fe81ae87b77c47045cb5"
)
MAX_PROMPT_TOKENS = 16_383
C1_CHECKPOINT = {
    "id": "C1",
    "name": "post_first_repository_observation",
    "visibility_boundary": (
        "after_successful_repository_read_or_search_before_second_assistant_token"
    ),
}
READ_SEARCH_COMMAND_RE = re.compile(
    r"(?:^|[;&|()]\s*|\bxargs\s+)(?:find|grep|rg|cat|head|tail|sed|ls)\b"
)
MUTATING_COMMAND_RE = re.compile(
    r"(?:^|[;&|()]\s*)(?:rm|mv|cp|touch|mkdir|rmdir|tee|truncate|patch)\b"
    r"|\bsed\s+-[^\n;|]*i\b|\bgit\s+(?:checkout|reset|clean|apply)\b"
)
OUTPUT_DIAGNOSTIC_RE = re.compile(
    r"(?:^|\n)(?:find|grep|cat|ls|head|tail|sed):[^\n]*(?:No such file|not found|cannot access)",
    re.IGNORECASE,
)

# Each pair occupies its original global request indices, but the first three
# corrected observations come from v7 and the remaining audit pairs from v2.
# File hashes bind the reviewed copies tracked under validation/ while portable
# source paths preserve the original campaign provenance in derived artifacts.
CAPTURE_USAGE_SHA256 = {
    "runs/swe_multitask_c1_host_v7_20260718/proxy_dumps": (
        "aae33e4a762cef130cec1ccf1f7c1bd18980e5e068e4cb6a7a504ee342c6490a"
    ),
    "runs/swe_multitask_c1_host_v2_20260718/proxy_dumps": (
        "724c1538c2afd9ac0b84f0422472c78c15784f294d554bf62ffca8ac669dd734"
    ),
}
CAPTURE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "instance_id": "pydata__xarray-6938",
        "proxy_dir": "runs/swe_multitask_c1_host_v7_20260718/proxy_dumps",
        "stored_proxy_dir": "validation/swe-multitask-c1-capture-v7",
        "first_index": 1,
        "first_sha256": "b28bf4a646b4c62b53f3442fb03fbab225d76447f0ff7035e70f90c772b387ea",
        "second_index": 2,
        "second_sha256": "f82d14243af61826c2930c4757e93759acee0397fa3fb3ea4a40e798e47088cd",
    },
    {
        "instance_id": "scikit-learn__scikit-learn-9288",
        "proxy_dir": "runs/swe_multitask_c1_host_v7_20260718/proxy_dumps",
        "stored_proxy_dir": "validation/swe-multitask-c1-capture-v7",
        "first_index": 3,
        "first_sha256": "0ce62df8829f7991f7fd814121efec4441dbb50ba69881b291f3a29f65a28e1f",
        "second_index": 4,
        "second_sha256": "5505068fd9ae4d130b6acbb7d071bc2fb7989aa51796db333ca4484105e5db33",
    },
    {
        "instance_id": "django__django-13297",
        "proxy_dir": "runs/swe_multitask_c1_host_v7_20260718/proxy_dumps",
        "stored_proxy_dir": "validation/swe-multitask-c1-capture-v7",
        "first_index": 5,
        "first_sha256": "fa7cebf1d33386520317c170586e1b6d68ec0006e649e5a36b6c189db811a238",
        "second_index": 6,
        "second_sha256": "5c7f84d176701c50a7eb6b61adc58f8ac2d674d7118136411ed9f427a0e11c55",
    },
    {
        "instance_id": "sympy__sympy-21847",
        "proxy_dir": "runs/swe_multitask_c1_host_v2_20260718/proxy_dumps",
        "stored_proxy_dir": "validation/swe-multitask-c1-capture-v2",
        "first_index": 7,
        "first_sha256": "0b35bfeada7901a6237ade668031e34830d706ce04c4350e4005f576f45f56da",
        "second_index": 8,
        "second_sha256": "3c6859d724663ac6ddede626dfca2542d215abbf1a7c54c03829049a90e737f4",
    },
    {
        "instance_id": "pylint-dev__pylint-4551",
        "proxy_dir": "runs/swe_multitask_c1_host_v2_20260718/proxy_dumps",
        "stored_proxy_dir": "validation/swe-multitask-c1-capture-v2",
        "first_index": 9,
        "first_sha256": "068fa942dd5a6b21e2d1fcc4982c4e088d3cddfab5862c27aa7db3f1c9c84a2c",
        "second_index": 10,
        "second_sha256": "2bb2e39ae5ed5874ff954375f16847384b1a1315eded3989b0f2047e11c4d5bc",
    },
    {
        "instance_id": "sphinx-doc__sphinx-8638",
        "proxy_dir": "runs/swe_multitask_c1_host_v2_20260718/proxy_dumps",
        "stored_proxy_dir": "validation/swe-multitask-c1-capture-v2",
        "first_index": 11,
        "first_sha256": "1b9cd91eeb8496d2bad89b915a02035c6bf536f0d327694b5e6d9e89673b5943",
        "second_index": 12,
        "second_sha256": "7868f77ff0858545adbe09fd4cc41490b8ef738a8738d67a4f7de90890242681",
    },
    {
        "instance_id": "mwaskom__seaborn-3187",
        "proxy_dir": "runs/swe_multitask_c1_host_v2_20260718/proxy_dumps",
        "stored_proxy_dir": "validation/swe-multitask-c1-capture-v2",
        "first_index": 13,
        "first_sha256": "7f61636cf0472c62ca1fbf3388ad991daa2102b446a7b3396d9100ba311be9e7",
        "second_index": 14,
        "second_sha256": "7c7785914d509a04eb868a89c275edf5a460339546865c128a26b3872728aa59",
    },
    {
        "instance_id": "matplotlib__matplotlib-25287",
        "proxy_dir": "runs/swe_multitask_c1_host_v2_20260718/proxy_dumps",
        "stored_proxy_dir": "validation/swe-multitask-c1-capture-v2",
        "first_index": 15,
        "first_sha256": "2ab766c1000c4a5ac1db1604d2246cfc254b35a55add89a43f95e66a0c7c66f2",
        "second_index": 16,
        "second_sha256": "43cfff42e7b5ee3e6f581db06d1c3663f819575378b24c3f887d019ea4019e61",
    },
    {
        "instance_id": "astropy__astropy-14598",
        "proxy_dir": "runs/swe_multitask_c1_host_v2_20260718/proxy_dumps",
        "stored_proxy_dir": "validation/swe-multitask-c1-capture-v2",
        "first_index": 17,
        "first_sha256": "0db66dde1f66b739786cf703afc6610f2f57e40b1c4a026969aa53a41e8bbefc",
        "second_index": 18,
        "second_sha256": "7a688d105d7b226fb54450f559379e010c508a148861582b9a865693263b412e",
    },
    {
        "instance_id": "pytest-dev__pytest-7571",
        "proxy_dir": "runs/swe_multitask_c1_host_v2_20260718/proxy_dumps",
        "stored_proxy_dir": "validation/swe-multitask-c1-capture-v2",
        "first_index": 19,
        "first_sha256": "e9554a7d2d3fa8ae47785430f70772469719398f60e0ce2197cccbd1a97429de",
        "second_index": 20,
        "second_sha256": "90cdc60c664f09513c0468279538b725ea9d4c0aa3cbb68dfd68ea15fc643008",
    },
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def flatten_text_content(value: Any, label: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError(f"{label} must be text or a content-block array")
    pieces: list[str] = []
    for index, block in enumerate(value):
        item = require_mapping(block, f"{label} block {index}")
        if item.get("type") != "text" or not isinstance(item.get("text"), str):
            raise ValueError(f"{label} block {index} is not a text block")
        pieces.append(item["text"])
    return "\n".join(pieces)


def flatten_string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [piece for item in value for piece in flatten_string_values(item)]
    if isinstance(value, dict):
        return [
            piece
            for key in sorted(value)
            for piece in flatten_string_values(value[key])
        ]
    return []


def surface_present(text: str, surface: str) -> bool:
    if not isinstance(surface, str) or not surface:
        raise ValueError("visibility surfaces must be nonempty strings")
    return (
        re.search(rf"(?<!\w){re.escape(surface)}(?!\w)", text, re.IGNORECASE)
        is not None
    )


def compound_identifier_hits(text: str, surface: str) -> list[str]:
    def contains_component(identifier: str) -> bool:
        if identifier == surface:
            return False
        if surface.startswith("_"):
            return surface in identifier
        return f"_{surface}" in identifier or f"{surface}_" in identifier

    return sorted(
        {
            identifier
            for identifier in re.findall(r"\w+", text, flags=re.UNICODE)
            if contains_component(identifier)
        }
    )


def exposure_evidence(
    text: str, surface: str, forms: Sequence[Mapping[str, Any]], tokenizer: Any
) -> dict[str, Any]:
    canonical_hit = surface_present(text, surface)
    compound_identifiers = compound_identifier_hits(text, surface)
    scored_ids = {
        int(form["token_id"])
        for form in forms
        if isinstance(form, dict)
        and isinstance(form.get("token_id"), int)
        and not isinstance(form.get("token_id"), bool)
    }
    observed_ids = set(tokenizer.encode(text, add_special_tokens=False)) if text else set()
    token_hits = sorted(scored_ids & observed_ids)
    return {
        "canonical_identifier_boundary_hit": canonical_hit,
        "case_sensitive_compound_identifier_hits": compound_identifiers,
        "scored_form_token_id_hits": token_hits,
        "exposed": canonical_hit or bool(compound_identifiers) or bool(token_hits),
    }


def validate_capture_request(request: Mapping[str, Any], *, request_index: int) -> None:
    if request.get("model") != RENDER.SERVED_MODEL or request.get("stream") is not True:
        raise ValueError(f"capture request {request_index} changed model or stream mode")
    template_kwargs = require_mapping(
        request.get("chat_template_kwargs"), f"request {request_index} template kwargs"
    )
    if set(template_kwargs) != {"enable_thinking"} or not isinstance(
        template_kwargs["enable_thinking"], bool
    ):
        raise ValueError(f"capture request {request_index} has unpinned template kwargs")
    tools = require_list(request.get("tools"), f"request {request_index} tools")
    if len(tools) != 1:
        raise ValueError(f"capture request {request_index} must declare one tool")
    tool = require_mapping(tools[0], f"request {request_index} tool")
    function = require_mapping(tool.get("function"), f"request {request_index} tool function")
    if tool.get("type") != "function" or function.get("name") != "run_shell_command":
        raise ValueError(f"capture request {request_index} declares an unexpected tool")


def render_request(
    tokenizer: Any, *, request: Mapping[str, Any], template: str
) -> tuple[str, list[int], int, list[dict[str, Any]]]:
    normalized_messages, normalized_count = RENDER.normalize_tool_call_arguments(
        request["messages"]
    )
    enable_thinking = request["chat_template_kwargs"]["enable_thinking"]
    rendered = tokenizer.apply_chat_template(
        normalized_messages,
        tools=request["tools"],
        chat_template=template,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template did not return text")
    suffix = (
        "<|im_start|>assistant\n<think>\n"
        if enable_thinking
        else "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    if not rendered.endswith(suffix):
        raise ValueError("rendered C1 prompt does not end at its generation boundary")
    return rendered, RENDER.encode_exact(tokenizer, rendered), normalized_count, normalized_messages


def assistant_channel_text(assistant: Mapping[str, Any]) -> str:
    reasoning = assistant.get("reasoning_content")
    duplicate_reasoning = assistant.get("reasoning")
    if reasoning is None and duplicate_reasoning is None:
        reasoning = ""
    elif not isinstance(reasoning, str) or duplicate_reasoning != reasoning:
        raise ValueError("assistant reasoning fields are absent or inconsistent")
    content = assistant.get("content")
    if not isinstance(content, str):
        raise ValueError("assistant visible content must be text")
    return reasoning + "\n" + content


def parse_tool_result(text: str) -> dict[str, Any]:
    match = re.fullmatch(
        r"Command: (?P<command>.*?)\nDirectory: (?P<directory>.*?)\nOutput: "
        r"(?P<output>.*?)\nError: (?P<error>.*?)\nExit Code: (?P<exit>-?\d+)\n"
        r"Signal: (?P<signal>-?\d+)\nProcess Group PGID: (?P<pgid>-?\d+)",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("tool result does not match the pinned Qwen shell envelope")
    return {
        "command": match.group("command"),
        "directory": match.group("directory"),
        "output": match.group("output"),
        "error": match.group("error"),
        "exit_code": int(match.group("exit")),
        "signal": int(match.group("signal")),
    }


def is_read_search_command(command: str) -> bool:
    return READ_SEARCH_COMMAND_RE.search(command) is not None and MUTATING_COMMAND_RE.search(
        command
    ) is None


def successful_repository_observation(command: str, result: Mapping[str, Any]) -> bool:
    output = str(result["output"])
    return (
        is_read_search_command(command)
        and result["exit_code"] == 0
        and result["signal"] == 0
        and result["error"] == "(none)"
        and bool(output.strip())
        and output.strip() != "(empty)"
        and OUTPUT_DIAGNOSTIC_RE.search(output) is None
    )


def audit_observations(
    assistant: Mapping[str, Any], tool_messages: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    calls = require_list(assistant.get("tool_calls"), "assistant tool calls")
    if not calls or len(calls) != len(tool_messages):
        raise ValueError("assistant tool calls and tool results must be nonempty and paired")
    audits: list[dict[str, Any]] = []
    argument_texts: list[str] = []
    output_texts: list[str] = []
    for call_index, (raw_call, raw_message) in enumerate(
        zip(calls, tool_messages, strict=True)
    ):
        call = require_mapping(raw_call, f"tool call {call_index}")
        message = require_mapping(raw_message, f"tool result {call_index}")
        function = require_mapping(call.get("function"), f"tool call {call_index} function")
        if call.get("type") != "function" or not isinstance(function.get("name"), str):
            raise ValueError("C1 capture contains a malformed tool call")
        if message.get("role") != "tool" or message.get("tool_call_id") != call.get("id"):
            raise ValueError("tool result order/ID does not match the assistant calls")
        raw_arguments = function.get("arguments")
        if not isinstance(raw_arguments, str):
            raise ValueError("captured tool-call arguments must be raw JSON strings")
        try:
            arguments = require_mapping(
                json.loads(raw_arguments), f"tool call {call_index} arguments"
            )
        except json.JSONDecodeError as exc:
            raise ValueError("captured tool-call arguments are invalid JSON") from exc
        result_text = flatten_text_content(message.get("content"), "tool result content")
        argument_texts.extend(flatten_string_values(arguments))
        output_texts.append(result_text)
        if function["name"] == "run_shell_command":
            command = arguments.get("command")
            if not isinstance(command, str) or not command:
                raise ValueError("captured shell call has no command")
            result = parse_tool_result(result_text)
            if result["command"] != command:
                raise ValueError("tool result command does not reproduce tool-call arguments")
            qualifying = successful_repository_observation(command, result)
            command_sha256 = sha256_text(command)
            exit_code: int | None = result["exit_code"]
            signal: int | None = result["signal"]
            output_bytes = len(str(result["output"]).encode("utf-8"))
            read_search = is_read_search_command(command)
            classification = (
                "successful_repository_read_or_search"
                if qualifying
                else "nonqualifying_shell_observation"
            )
        else:
            qualifying = False
            command_sha256 = None
            exit_code = None
            signal = None
            output_bytes = len(result_text.encode("utf-8"))
            read_search = False
            classification = "non_shell_or_permission_denied_observation"
        audits.append(
            {
                "call_index": call_index,
                "tool_call_id": call.get("id"),
                "tool_name": function["name"],
                "arguments_sha256": sha256_text(raw_arguments),
                "command_sha256": command_sha256,
                "result_sha256": sha256_text(result_text),
                "exit_code": exit_code,
                "signal": signal,
                "output_bytes": output_bytes,
                "read_search_command": read_search,
                "successful_repository_observation": qualifying,
                "classification": classification,
            }
        )
    channels = {
        "assistant_text": assistant_channel_text(assistant),
        "tool_call_args": "\n".join(argument_texts),
        "tool_outputs": "\n".join(output_texts),
    }
    return audits, channels


def concept_visibility(
    task: Mapping[str, Any],
    channels: Mapping[str, str],
    rendered_prompt: str,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_concept in require_list(task.get("concepts"), "task concepts"):
        concept = require_mapping(raw_concept, "task concept")
        subjects = [
            {
                "subject": "target",
                "task_instance_id": task["instance_id"],
                "concept_id": concept["id"],
                "family": concept["family"],
                "target": concept["target"],
                "forms": copy.deepcopy(concept["forms"]),
            }
        ]
        for raw_foil in require_list(concept.get("foils"), "concept foils"):
            foil = require_mapping(raw_foil, "concept foil")
            subjects.append(
                {
                    "subject": "foil",
                    "task_instance_id": foil["task_instance_id"],
                    "concept_id": foil["concept_id"],
                    "family": foil["family"],
                    "target": foil["target"],
                    "forms": copy.deepcopy(foil["forms"]),
                }
            )
        for subject in subjects:
            channel_evidence = {
                name: exposure_evidence(
                    text,
                    str(subject["target"]),
                    require_list(subject["forms"], "visibility forms"),
                    tokenizer,
                )
                for name, text in channels.items()
            }
            visibility = {
                name: evidence["exposed"]
                for name, evidence in channel_evidence.items()
            }
            new_visible = any(visibility.values())
            rendered_visible = exposure_evidence(
                rendered_prompt,
                str(subject["target"]),
                require_list(subject["forms"], "visibility forms"),
                tokenizer,
            )
            rendered_visible = bool(rendered_visible["exposed"])
            if new_visible and not rendered_visible:
                raise ValueError("visible observation surface is absent from rendered C1 prompt")
            role = (
                "explicit_control_excluded"
                if new_visible
                else (
                    "preexisting_visible_excluded"
                    if rendered_visible
                    else (
                        "primary_hidden"
                        if subject["subject"] == "target"
                        else "matched_hidden_foil"
                    )
                )
            )
            records.append(
                {
                    **subject,
                    "channels": visibility,
                    "channel_evidence": channel_evidence,
                    "visible_channels": [
                        name for name, visible in visibility.items() if visible
                    ],
                    "visible_in_rendered_prompt": rendered_visible,
                    "analysis_role": role,
                }
            )
    return records


def retained_concepts(
    task: Mapping[str, Any], visibility: Sequence[Mapping[str, Any]], tokenizer: Any
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    by_identity = {
        (record["subject"], record["task_instance_id"], record["concept_id"]): record
        for record in visibility
    }
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    retained_token_ids: set[int] = set()
    for raw_concept in require_list(task.get("concepts"), "task concepts"):
        concept = require_mapping(raw_concept, "task concept")
        target_record = by_identity[("target", task["instance_id"], concept["id"])]
        if target_record["analysis_role"] != "primary_hidden":
            excluded.append(copy.deepcopy(dict(target_record)))
            for raw_foil in concept.get("foils", []):
                foil = require_mapping(raw_foil, "concept foil")
                excluded.append(
                    copy.deepcopy(
                        dict(
                            by_identity[
                                ("foil", foil["task_instance_id"], foil["concept_id"])
                            ]
                        )
                    )
                )
            continue

        forms = copy.deepcopy(require_list(concept.get("forms"), "target forms"))
        for form_index, raw_form in enumerate(forms):
            form = require_mapping(raw_form, f"target form {form_index}")
            _, token_id = C0._validate_form(form, tokenizer, "C1 target")
            retained_token_ids.add(token_id)
        foils: list[dict[str, Any]] = []
        for raw_foil in require_list(concept.get("foils"), "concept foils"):
            foil = require_mapping(raw_foil, "concept foil")
            record = by_identity[("foil", foil["task_instance_id"], foil["concept_id"])]
            if record["analysis_role"] != "matched_hidden_foil":
                excluded.append(copy.deepcopy(dict(record)))
                continue
            foil_copy = copy.deepcopy(dict(foil))
            for form_index, raw_form in enumerate(foil_copy["forms"]):
                form = require_mapping(raw_form, f"foil form {form_index}")
                _, token_id = C0._validate_form(form, tokenizer, "C1 foil")
                retained_token_ids.add(token_id)
            foils.append(foil_copy)
        retained.append(
            {
                "id": concept["id"],
                "family": concept["family"],
                "target": concept["target"],
                "path": concept["path"],
                "evidence": copy.deepcopy(concept["sources"]),
                "visibility": "oracle_hidden",
                "forms": forms,
                "foils": foils,
            }
        )
    frozen_ids = require_list(task.get("score_token_ids"), "task score token IDs")
    score_ids = [token_id for token_id in frozen_ids if token_id in retained_token_ids]
    if set(score_ids) != retained_token_ids:
        raise ValueError("retained C1 vocabulary is not a subset of the frozen C0 task vocabulary")
    return retained, score_ids, excluded


def validate_pair(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    first_index: int,
    task: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    second_index = first_index + 1
    validate_capture_request(first, request_index=first_index)
    validate_capture_request(second, request_index=second_index)
    first_messages = require_list(first.get("messages"), "first request messages")
    second_messages = require_list(second.get("messages"), "second request messages")
    if len(first_messages) != 2 or second_messages[:2] != first_messages:
        raise ValueError("second request does not exactly preserve the two-message task start")
    if first.get("tools") != second.get("tools") or first.get("model") != second.get("model"):
        raise ValueError("request pair changes the model or tool schema")
    initial_json = json.dumps(first_messages, ensure_ascii=False, sort_keys=True)
    instance_id = str(task["instance_id"])
    if initial_json.count(instance_id) < 1:
        raise ValueError(f"request pair does not bind frozen task {instance_id}")
    assistant = require_mapping(second_messages[2], "first assistant response")
    if assistant.get("role") != "assistant":
        raise ValueError("second request does not append an assistant response")
    tool_messages = [require_mapping(value, "tool result") for value in second_messages[3:]]
    return assistant, tool_messages


def build_c1_bundle(
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    requests: Sequence[Mapping[str, Any]],
    request_sources: Sequence[Mapping[str, Any]],
    usage_records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    template: str,
    capture_manifest_sha256: str,
    usage_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    C0._validate_protocol_header(protocol)
    tasks = require_list(protocol.get("tasks"), "protocol tasks")
    if len(requests) != 2 * len(tasks) or len(request_sources) != len(requests):
        raise ValueError("capture must contain exactly two requests per frozen C0 task")
    if len(usage_records) != len(requests):
        raise ValueError("usage ledger and capture request counts differ")

    prompts: list[dict[str, Any]] = []
    pair_audits: list[dict[str, Any]] = []
    all_excluded: list[dict[str, Any]] = []
    for task_index, raw_task in enumerate(tasks):
        task = require_mapping(raw_task, f"protocol task {task_index}")
        if task.get("selection_index") != task_index:
            raise ValueError("protocol task selection order changed")
        first_index = task_index * 2 + 1
        second_index = first_index + 1
        first = requests[first_index - 1]
        second = requests[second_index - 1]
        for source in request_sources[first_index - 1 : second_index]:
            captured_instance = source.get("capture_instance_id")
            if captured_instance is not None and captured_instance != task["instance_id"]:
                raise ValueError("merged capture source is bound to the wrong frozen task")
        assistant, tool_messages = validate_pair(
            first, second, first_index=first_index, task=task
        )
        observations, channels = audit_observations(assistant, tool_messages)
        rendered, token_ids, normalized_count, normalized_messages = render_request(
            tokenizer, request=second, template=template
        )
        usage = require_mapping(usage_records[second_index - 1], "second-request usage")
        usage_values = require_mapping(usage.get("usage"), "second-request usage values")
        if usage.get("idx") != second_index or usage_values.get("prompt_tokens") != len(
            token_ids
        ):
            raise ValueError("rendered second request disagrees with the pinned usage ledger")
        if len(token_ids) > MAX_PROMPT_TOKENS:
            raise ValueError(f"C1 request {second_index} exceeds the model context limit")

        visibility = concept_visibility(task, channels, rendered, tokenizer)
        concepts, score_ids, excluded = retained_concepts(task, visibility, tokenizer)
        all_excluded.extend(
            {
                **copy.deepcopy(record),
                "receiver_task_instance_id": task["instance_id"],
                "request_index": second_index,
            }
            for record in excluded
        )
        observation_success = any(
            record["successful_repository_observation"] for record in observations
        )
        selected = observation_success and bool(concepts)
        exclusion_reason = None
        if not observation_success:
            exclusion_reason = "no_genuinely_successful_repository_read_or_search"
        elif not concepts:
            exclusion_reason = "no_hidden_primary_target_after_visibility_audit"
        pair_audits.append(
            {
                "task_selection_index": task_index,
                "instance_id": task["instance_id"],
                "first_request_index": first_index,
                "second_request_index": second_index,
                "first_request_sha256": request_sources[first_index - 1]["sha256"],
                "second_request_sha256": request_sources[second_index - 1]["sha256"],
                "rendered_prompt_sha256": sha256_text(rendered),
                "prompt_token_count": len(token_ids),
                "observations": observations,
                "concept_visibility": visibility,
                "selected": selected,
                "exclusion_reason": exclusion_reason,
            }
        )
        if not selected:
            continue

        channel_hashes = {name: sha256_text(text) for name, text in channels.items()}
        prompts.append(
            {
                "id": f"swe-c1-{second_index:02d}-{task['instance_id']}",
                "text": rendered,
                "token_ids": token_ids,
                "score_token_ids": score_ids,
                "metadata": {
                    "kind": "swe_verified_multitask_initial_probe",
                    "protocol_sha256": protocol_sha256,
                    "lens_outputs_used_for_selection": False,
                    "task": {
                        "instance_id": task["instance_id"],
                        "repo": task["repo"],
                        "base_commit": task["base_commit"],
                        "problem_statement_sha256": sha256_text(task["problem_statement"]),
                        "patch_sha256": task["patch_sha256"],
                        "test_patch_sha256": task["test_patch_sha256"],
                    },
                    "checkpoint": copy.deepcopy(C1_CHECKPOINT),
                    "middle_band_layers": list(C0.MIDDLE_BAND_LAYERS),
                    "concepts": concepts,
                    "observation_audit": {
                        "capture_manifest_sha256": capture_manifest_sha256,
                        "usage_manifest_sha256": usage_manifest_sha256,
                        "first_request_index": first_index,
                        "second_request_index": second_index,
                        "first_request_sha256": request_sources[first_index - 1]["sha256"],
                        "second_request_sha256": request_sources[second_index - 1]["sha256"],
                        "normalized_messages_sha256": sha256_json(normalized_messages),
                        "normalized_string_tool_call_arguments": normalized_count,
                        "channel_sha256": channel_hashes,
                        "observations": copy.deepcopy(observations),
                        "concept_visibility": copy.deepcopy(visibility),
                        "excluded_controls": copy.deepcopy(excluded),
                    },
                },
            }
        )

    summary = {
        "schema_version": 1,
        "kind": "swe_verified_multitask_c1_materialization",
        "checkpoint": copy.deepcopy(C1_CHECKPOINT),
        "protocol_sha256": protocol_sha256,
        "capture_manifest_sha256": capture_manifest_sha256,
        "usage_manifest_sha256": usage_manifest_sha256,
        "capture_sources": copy.deepcopy(list(request_sources)),
        "capture_policy": {
            "included": (
                "v7 request pairs 1-6 for xarray/scikit-learn/django; "
                "v2 request pairs 7-20 for the remaining frozen tasks"
            ),
            "excluded_campaigns": [
                {
                    "campaigns": ["v3", "v4", "v5", "v6"],
                    "reason": (
                        "failed repository diagnostics, denied non-shell tools, "
                        "or incomplete request pairs"
                    ),
                }
            ],
            "task_level_exclusions_recorded_in": "pair_audits",
        },
        "chat_template_sha256": sha256_text(template),
        "tokenizer_json_sha256": C0.TOKENIZER_JSON_SHA256,
        "lens_outputs_used_for_selection": False,
        "request_count": len(requests),
        "pair_count": len(pair_audits),
        "selected_prompt_count": len(prompts),
        "selected_task_count": len(prompts),
        "retained_primary_concept_count": sum(
            len(prompt["metadata"]["concepts"]) for prompt in prompts
        ),
        "retained_foil_count": sum(
            len(concept["foils"])
            for prompt in prompts
            for concept in prompt["metadata"]["concepts"]
        ),
        "excluded_control_count": sum(
            record["analysis_role"] == "explicit_control_excluded"
            for record in all_excluded
        ),
        "excluded_visibility_records": all_excluded,
        "pair_audits": pair_audits,
        "prompts": [
            {
                "id": prompt["id"],
                "instance_id": prompt["metadata"]["task"]["instance_id"],
                "prompt_sha256": sha256_text(prompt["text"]),
                "token_ids_sha256": sha256_json(prompt["token_ids"]),
                "prompt_token_count": len(prompt["token_ids"]),
                "score_token_ids_sha256": sha256_json(prompt["score_token_ids"]),
                "score_token_count": len(prompt["score_token_ids"]),
                "concept_count": len(prompt["metadata"]["concepts"]),
            }
            for prompt in prompts
        ],
    }
    return prompts, summary


def load_merged_capture(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str, str]:
    requests: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    usage_records: list[dict[str, Any]] = []
    usage_cache: dict[str, list[dict[str, Any]]] = {}
    usage_hashes: dict[str, str] = {}
    for task_index, spec in enumerate(CAPTURE_SPECS):
        expected_first = task_index * 2 + 1
        expected_second = expected_first + 1
        if (
            spec["first_index"] != expected_first
            or spec["second_index"] != expected_second
        ):
            raise ValueError("merged capture request indices are not canonical")
        proxy_dir = (root / spec["stored_proxy_dir"]).resolve(strict=True)
        usage_path = proxy_dir / "usage.jsonl"
        portable_usage_path = f"{spec['proxy_dir']}/usage.jsonl"
        actual_usage_sha256 = sha256_file(usage_path)
        expected_usage_sha256 = CAPTURE_USAGE_SHA256.get(str(spec["proxy_dir"]))
        if actual_usage_sha256 != expected_usage_sha256:
            raise ValueError(f"usage ledger hash mismatch for {spec['instance_id']}")
        usage_hashes[portable_usage_path] = actual_usage_sha256
        if portable_usage_path not in usage_cache:
            usage_cache[portable_usage_path] = [
                dict(require_mapping(json.loads(line), f"{portable_usage_path} line"))
                for line in usage_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        ledger = usage_cache[portable_usage_path]

        for ordinal in ("first", "second"):
            request_index = int(spec[f"{ordinal}_index"])
            request_path = proxy_dir / f"chat_{request_index:04d}.json"
            actual_sha256 = sha256_file(request_path)
            if actual_sha256 != spec[f"{ordinal}_sha256"]:
                raise ValueError(
                    f"request hash mismatch for {spec['instance_id']} {ordinal} request"
                )
            requests.append(
                dict(
                    require_mapping(
                        json.loads(request_path.read_bytes()), request_path.name
                    )
                )
            )
            sources.append(
                {
                    "path": f"{spec['proxy_dir']}/{request_path.name}",
                    "bytes": request_path.stat().st_size,
                    "sha256": actual_sha256,
                    "capture_instance_id": spec["instance_id"],
                    "capture_request_index": request_index,
                }
            )
            try:
                usage_record = ledger[request_index - 1]
            except IndexError as exc:
                raise ValueError(
                    f"usage ledger lacks request {request_index} for {spec['instance_id']}"
                ) from exc
            if usage_record.get("idx") != request_index:
                raise ValueError("usage ledger request indices are not canonical")
            usage_records.append(copy.deepcopy(usage_record))

    canonical_specs = [
        {key: value for key, value in spec.items() if key != "stored_proxy_dir"}
        for spec in CAPTURE_SPECS
    ]
    manifest_sha256 = sha256_json(
        {"request_pairs": canonical_specs, "usage_ledgers": CAPTURE_USAGE_SHA256}
    )
    usage_manifest_sha256 = sha256_json(usage_hashes)
    return (
        requests,
        sources,
        usage_records,
        manifest_sha256,
        usage_manifest_sha256,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--capture-root", type=Path, default=ROOT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    protocol_path = args.protocol.expanduser().resolve(strict=True)
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha256 = sha256_bytes(protocol_bytes)
    if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("frozen C0 protocol SHA-256 mismatch")
    protocol = require_mapping(json.loads(protocol_bytes), "C0 protocol")
    capture_root = args.capture_root.expanduser().resolve(strict=True)
    (
        requests,
        sources,
        usage,
        capture_sha256,
        usage_manifest_sha256,
    ) = load_merged_capture(capture_root)
    template_path = args.template.expanduser().resolve(strict=True)
    if sha256_file(template_path) != RENDER.EXPECTED_TEMPLATE_SHA256:
        raise ValueError("pinned Qwen chat template SHA-256 mismatch")
    template = template_path.read_text(encoding="utf-8")
    snapshot = Path(
        args.model_snapshot
        or snapshot_download(
            RENDER.MODEL_REPO, revision=RENDER.MODEL_REVISION, local_files_only=True
        )
    ).expanduser().resolve(strict=True)
    if snapshot.name != RENDER.MODEL_REVISION:
        raise ValueError("model snapshot is not the pinned Qwen revision")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    RENDER.validate_tokenizer(tokenizer, snapshot)
    _, _, model_pin, _ = C0._validate_protocol_header(protocol)
    C0._validate_model_pin(model_pin, tokenizer, snapshot)
    prompts, summary = build_c1_bundle(
        protocol,
        protocol_sha256=protocol_sha256,
        requests=requests,
        request_sources=sources,
        usage_records=usage,
        tokenizer=tokenizer,
        template=template,
        capture_manifest_sha256=capture_sha256,
        usage_manifest_sha256=usage_manifest_sha256,
    )
    output = args.output.expanduser().resolve()
    summary_path = (
        args.summary or output.with_name(f"{output.stem}_summary.json")
    ).expanduser().resolve()
    atomic_write_json(output, prompts)
    summary["prompt_bundle_sha256"] = sha256_file(output)
    atomic_write_json(summary_path, summary)
    print(f"wrote {output} ({len(prompts)} C1 prompts, sha256={summary['prompt_bundle_sha256']})")
    print(f"wrote {summary_path} (sha256={sha256_file(summary_path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
