#!/usr/bin/env python3
"""Materialize exact Jacobian-Lens prompts from the certified SWE episode."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
SERVED_MODEL = "qwen3.6-27b-nvfp4"
RUN_RELATIVE_PATH = Path("runs/publication_certified_v2_20260715")
TRACE_RELATIVE_PATH = Path(
    "generation/verified/per_task/sympy__sympy-13480/qwen_trace.json"
)
TEMPLATE_RELATIVE_PATH = Path("configs/qwen3-openai-codex.jinja")
DEFAULT_OUTPUT_DIR = Path(".cache/swe_jlens_prompts")

EXPECTED_PROMPT_TOKENS = (
    11861,
    12148,
    12743,
    13629,
    13883,
    14522,
    15073,
    15327,
    15678,
)
EXPECTED_CHAT_SHA256 = (
    "559e3357e4bf869ce5d842abf8872a779a5db1c9336db44344e8ad53559978fc",
    "68eec2079c2e7698cb888ae9f89a67c35e163a47c0eb3e325ac13070785f87c0",
    "ac6114406541945afd712e6bf6d51206291bdbe4159ca04f01b064514c015044",
    "40fa434ffd4c89feb49199aaa37b02cd649df80ec0cbf5cbe07ada4218e21d79",
    "ec7ba71d34ea26e84870a94096fbea7c3a913b047cb0b24cf0c25a631a38758b",
    "f37c38768c8d8d501c51d493f47881248189870fbedc0426ab972b1843e0c92f",
    "2432af56b74ff60662a0654bf15ad90da3fbea159b68891f2a7e28a32d8361fb",
    "db38f1f0e22e328c1209ba3066421d0876e7411257f7464efeec5e445e191f79",
    "91b4fa6f7220131c6a2d930c2c9526093d38a936fc9fbdcf0bbb9d81d6e5df50",
)
EXPECTED_USAGE_SHA256 = (
    "60757ba7d953c5378392bf2a4d3fe8812ef739645a7977a9eeff311f0d1313e9"
)
EXPECTED_TRACE_SHA256 = (
    "3e39701da94a3f590e62efc9e67aa22220e155b385cceb27cbecc4d9a56d632a"
)
EXPECTED_TEMPLATE_SHA256 = (
    "c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da"
)
EXPECTED_TOKENIZER_FILES = {
    "tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "tokenizer_config.json": (
        "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
    ),
    "vocab.json": (
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
    ),
}
EXPECTED_MODEL_METADATA_FILES = {
    "config.json": (
        "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338"
    ),
    "model.safetensors.index.json": (
        "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2"
    ),
}
EXPECTED_TOKENIZER_CLASS = "Qwen2Tokenizer"
EXPECTED_TOKENIZER_LENGTH = 248077
EXPECTED_VOCAB_SIZE = 248044
EXPECTED_BOS_TOKEN_ID = None
EXPECTED_EOS_TOKEN_ID = 248046

STAGE_NAMES = (
    "initial_task_analysis",
    "source_location_known",
    "buggy_source_inspected_def_use_correction",
    "failure_reproduced",
    "patch_applied",
    "smoke_fix_verified",
    "broader_values_verified",
    "pytest_unavailable",
    "focused_sympy_test_passed",
)
PROBE_STAGE_INDEX = 3
PROBE_SLOT_MARKER = "but the variable is actually `"
PROBE_CANDIDATES = (
    ("correct_def_use", "cothm"),
    ("buggy_undefined_name", "cotm"),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def file_record(path: Path, *, display_path: str) -> dict[str, Any]:
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_file_hash(path: Path, expected: str, *, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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


def normalize_tool_call_arguments(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Parse OpenAI JSON-string arguments without mutating the source request."""
    normalized = copy.deepcopy(list(messages))
    normalized_count = 0
    for message_index, message in enumerate(normalized):
        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            continue
        if not isinstance(tool_calls, list):
            raise ValueError(f"message {message_index} tool_calls is not a list")
        for call_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                raise ValueError(
                    f"message {message_index} tool call {call_index} is not an object"
                )
            function = tool_call.get("function", tool_call)
            if not isinstance(function, dict):
                raise ValueError(
                    f"message {message_index} tool call {call_index} has no function object"
                )
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"message {message_index} tool call {call_index} has invalid "
                        "JSON-string arguments"
                    ) from exc
                normalized_count += 1
            if arguments is not None and not isinstance(arguments, dict):
                raise ValueError(
                    f"message {message_index} tool call {call_index} arguments "
                    "must normalize to an object"
                )
            function["arguments"] = arguments
    return normalized, normalized_count


def load_usage(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"usage ledger contains a blank line at {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"usage row {line_number} is not an object")
        records.append(value)
    if len(records) != len(EXPECTED_PROMPT_TOKENS):
        raise ValueError(f"expected 9 usage rows, found {len(records)}")
    for index, (record, expected_tokens) in enumerate(
        zip(records, EXPECTED_PROMPT_TOKENS, strict=True), 1
    ):
        if record.get("idx") != index:
            raise ValueError(f"usage row {index} has an unexpected request index")
        usage = record.get("usage")
        if not isinstance(usage, dict) or usage.get("prompt_tokens") != expected_tokens:
            raise ValueError(f"usage row {index} has an unexpected prompt token count")
    return records


def extract_trace_records(
    value: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(value, list):
        raise ValueError("qwen_trace.json must contain a JSON list")
    thinking: list[dict[str, Any]] = []
    completions: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(value):
        if not isinstance(entry, dict) or entry.get("type") != "assistant":
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            raise ValueError(f"assistant trace entry {entry_index} has no message")
        content = message.get("content")
        if not isinstance(content, list):
            raise ValueError(f"assistant trace entry {entry_index} has invalid content")
        for item in content:
            if isinstance(item, dict) and item.get("type") == "thinking":
                text = item.get("thinking")
                if not isinstance(text, str) or not text:
                    raise ValueError(
                        f"assistant trace entry {entry_index} has invalid thinking text"
                    )
                thinking.append(
                    {
                        "entry_index": entry_index,
                        "entry_uuid": entry.get("uuid"),
                        "message_id": message.get("id"),
                        "text": text,
                    }
                )
        usage = message.get("usage")
        if isinstance(usage, dict) and usage.get("input_tokens", 0) > 0:
            completions.append(
                {
                    "entry_index": entry_index,
                    "entry_uuid": entry.get("uuid"),
                    "message_id": message.get("id"),
                    "usage": usage,
                    "stop_reason": message.get("stop_reason"),
                    "content_types": [
                        item.get("type") for item in content if isinstance(item, dict)
                    ],
                }
            )
    if len(thinking) != 9 or len(completions) != 9:
        raise ValueError(
            "certified trace must contain exactly 9 thinking blocks and 9 completions"
        )
    for index, (thought, completion, expected_tokens) in enumerate(
        zip(thinking, completions, EXPECTED_PROMPT_TOKENS, strict=True), 1
    ):
        if thought["entry_index"] >= completion["entry_index"]:
            raise ValueError(f"trace thinking/completion order is invalid at request {index}")
        if completion["usage"].get("input_tokens") != expected_tokens:
            raise ValueError(
                f"trace completion {index} has an unexpected input token count"
            )
    return thinking, completions


def encode_exact(tokenizer: Any, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(token_ids, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in token_ids
    ):
        raise TypeError("tokenizer.encode returned invalid token IDs")
    return token_ids


def boundary_continuation_ids(
    tokenizer: Any,
    prompt_text: str,
    prompt_token_ids: Sequence[int],
    continuation: str,
) -> list[int]:
    combined = encode_exact(tokenizer, prompt_text + continuation)
    prefix = list(prompt_token_ids)
    if combined[: len(prefix)] != prefix:
        raise ValueError("continuation changed the verified prompt token prefix")
    result = combined[len(prefix) :]
    if not result:
        raise ValueError("continuation produced no target tokens")
    return result


def decoded_tokens(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    return [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]


def validate_request(request: Any, *, request_index: int) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError(f"chat_{request_index:04d}.json is not an object")
    expected_seed = 880001233 + request_index
    expected_keys = {
        "model": SERVED_MODEL,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "seed": expected_seed,
    }
    for key, expected in expected_keys.items():
        if request.get(key) != expected:
            raise ValueError(
                f"chat_{request_index:04d}.json has unexpected {key}: "
                f"{request.get(key)!r}"
            )
    messages = request.get("messages")
    tools = request.get("tools")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"chat_{request_index:04d}.json has invalid messages")
    if not isinstance(tools, list) or len(tools) != 1:
        raise ValueError(f"chat_{request_index:04d}.json must declare one tool")
    function = tools[0].get("function") if isinstance(tools[0], dict) else None
    if not isinstance(function, dict) or function.get("name") != "run_shell_command":
        raise ValueError(f"chat_{request_index:04d}.json has an unexpected tool")
    return request


def render_request(
    tokenizer: Any,
    *,
    request: Mapping[str, Any],
    template: str,
) -> tuple[str, list[int], int, list[dict[str, Any]]]:
    normalized_messages, normalized_count = normalize_tool_call_arguments(
        request["messages"]
    )
    rendered = tokenizer.apply_chat_template(
        normalized_messages,
        tools=request["tools"],
        chat_template=template,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template did not return text")
    expected_suffix = "<|im_start|>assistant\n<think>\n"
    if not rendered.endswith(expected_suffix):
        raise ValueError("rendered prompt does not end at the thinking generation boundary")
    return rendered, encode_exact(tokenizer, rendered), normalized_count, normalized_messages


def build_stage_prompts(
    tokenizer: Any,
    *,
    requests: Sequence[Mapping[str, Any]],
    request_sources: Sequence[Mapping[str, Any]],
    usage_records: Sequence[Mapping[str, Any]],
    thinking_records: Sequence[Mapping[str, Any]],
    completion_records: Sequence[Mapping[str, Any]],
    template: str,
    provenance_id: str,
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for index, (
        request,
        request_source,
        usage_record,
        thinking_record,
        completion_record,
        expected_tokens,
        stage_name,
    ) in enumerate(
        zip(
            requests,
            request_sources,
            usage_records,
            thinking_records,
            completion_records,
            EXPECTED_PROMPT_TOKENS,
            STAGE_NAMES,
            strict=True,
        ),
        1,
    ):
        rendered, token_ids, normalized_count, normalized_messages = render_request(
            tokenizer,
            request=request,
            template=template,
        )
        if len(token_ids) != expected_tokens:
            raise ValueError(
                f"request {index} rendered to {len(token_ids)} tokens; "
                f"expected {expected_tokens}"
            )
        if usage_record["usage"]["prompt_tokens"] != len(token_ids):
            raise ValueError(f"request {index} disagrees with usage.jsonl")
        if completion_record["usage"]["input_tokens"] != len(token_ids):
            raise ValueError(f"request {index} disagrees with qwen_trace.json")

        thinking_text = thinking_record["text"]
        continuation_ids = boundary_continuation_ids(
            tokenizer,
            rendered,
            token_ids,
            thinking_text,
        )
        stage_id = f"swe-sympy-13480-request-{index:02d}"
        prompts.append(
            {
                "id": stage_id,
                "text": rendered,
                "token_ids": token_ids,
                "metadata": {
                    "kind": "certified_swe_stage",
                    "provenance_id": provenance_id,
                    "stage": {
                        "request_index": index,
                        "name": stage_name,
                        "seed": request["seed"],
                        "message_count": len(request["messages"]),
                        "normalized_string_tool_call_arguments": normalized_count,
                    },
                    "source_request": dict(request_source),
                    "normalized_messages_sha256": sha256_json(normalized_messages),
                    "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
                    "token_ids_sha256": sha256_json(token_ids),
                    "prompt_token_count": len(token_ids),
                    "usage": copy.deepcopy(usage_record),
                    "sampled_next": {
                        "source": "qwen_trace.json assistant thinking block",
                        "original_next_text": thinking_text,
                        "original_next_text_sha256": sha256_bytes(
                            thinking_text.encode("utf-8")
                        ),
                        "boundary_tokenization": (
                            "tokenize(rendered_prompt + original_next_text), then "
                            "remove the verified rendered_prompt token prefix"
                        ),
                        "sequence_token_ids": continuation_ids,
                        "sequence_token_count": len(continuation_ids),
                        "first_token_id": continuation_ids[0],
                        "first_token_text": decoded_tokens(
                            tokenizer, continuation_ids[:1]
                        )[0],
                        "trace_thinking_entry_index": thinking_record["entry_index"],
                        "trace_thinking_entry_uuid": thinking_record["entry_uuid"],
                        "trace_completion_entry_index": completion_record["entry_index"],
                        "trace_completion_entry_uuid": completion_record["entry_uuid"],
                    },
                },
            }
        )
    return prompts


def candidate_sequence(
    tokenizer: Any,
    *,
    context_text: str,
    context_token_ids: Sequence[int],
    candidate: str,
) -> tuple[list[int], list[str]]:
    candidate_ids = boundary_continuation_ids(
        tokenizer,
        context_text,
        context_token_ids,
        candidate,
    )
    pieces = decoded_tokens(tokenizer, candidate_ids)
    if "".join(pieces) != candidate:
        raise ValueError(
            f"candidate {candidate!r} does not round-trip through per-token decoding"
        )
    return candidate_ids, pieces


def build_candidate_probes(
    tokenizer: Any,
    *,
    stage_prompts: Sequence[Mapping[str, Any]],
    provenance_id: str,
) -> list[dict[str, Any]]:
    stage = stage_prompts[PROBE_STAGE_INDEX - 1]
    sampled_next = stage["metadata"]["sampled_next"]
    thinking_text = sampled_next["original_next_text"]
    if thinking_text.count(PROBE_SLOT_MARKER) != 1:
        raise ValueError("turn-3 thinking does not contain the unique semantic slot marker")
    marker_end = thinking_text.index(PROBE_SLOT_MARKER) + len(PROBE_SLOT_MARKER)
    reasoning_prefix = thinking_text[:marker_end]
    if thinking_text[marker_end : marker_end + len("cothm")] != "cothm":
        raise ValueError("the certified semantic slot is not followed by cothm")

    base_text = stage["text"]
    base_token_ids = stage["token_ids"]
    context_text = base_text + reasoning_prefix
    context_token_ids = encode_exact(tokenizer, context_text)
    if context_token_ids[: len(base_token_ids)] != base_token_ids:
        raise ValueError("teacher-forced reasoning changed the turn-3 prompt prefix")
    context_record = {
        "base_stage_id": stage["id"],
        "semantic_slot": "observed def-use correction after source inspection",
        "slot_marker": PROBE_SLOT_MARKER,
        "retained_reasoning_prefix": reasoning_prefix,
        "retained_reasoning_prefix_sha256": sha256_bytes(
            reasoning_prefix.encode("utf-8")
        ),
        "text_sha256": sha256_bytes(context_text.encode("utf-8")),
        "token_ids_sha256": sha256_json(context_token_ids),
        "token_count": len(context_token_ids),
    }

    probes: list[dict[str, Any]] = []
    for label, candidate in PROBE_CANDIDATES:
        candidate_ids, pieces = candidate_sequence(
            tokenizer,
            context_text=context_text,
            context_token_ids=context_token_ids,
            candidate=candidate,
        )
        full_candidate_ids = encode_exact(tokenizer, context_text + candidate)
        if full_candidate_ids != context_token_ids + candidate_ids:
            raise ValueError(f"candidate {candidate!r} does not preserve its context prefix")
        sequence_sha256 = sha256_json(candidate_ids)
        for step_index, target_token_id in enumerate(candidate_ids):
            teacher_forced_prefix = "".join(pieces[:step_index])
            step_text = context_text + teacher_forced_prefix
            step_token_ids = encode_exact(tokenizer, step_text)
            expected_step_ids = context_token_ids + candidate_ids[:step_index]
            if step_token_ids != expected_step_ids:
                raise ValueError(
                    f"candidate {candidate!r} step {step_index} changed token boundaries"
                )
            probes.append(
                {
                    "id": (
                        f"swe-turn-03-def-use-{candidate}-step-{step_index + 1:02d}"
                    ),
                    "text": step_text,
                    "token_ids": step_token_ids,
                    "target_token_id": target_token_id,
                    "metadata": {
                        "kind": "certified_swe_teacher_forced_candidate_step",
                        "provenance_id": provenance_id,
                        "probe_context": context_record,
                        "candidate": {
                            "label": label,
                            "identifier": candidate,
                            "sequence_token_ids": candidate_ids,
                            "sequence_token_texts": pieces,
                            "sequence_token_count": len(candidate_ids),
                            "sequence_token_ids_sha256": sequence_sha256,
                        },
                        "step": {
                            "index": step_index,
                            "number": step_index + 1,
                            "count": len(candidate_ids),
                            "teacher_forced_candidate_prefix": teacher_forced_prefix,
                            "prompt_token_count": len(step_token_ids),
                            "prompt_token_ids_sha256": sha256_json(step_token_ids),
                            "target_token_text": pieces[step_index],
                            "remaining_sequence_token_ids": candidate_ids[step_index:],
                        },
                    },
                }
            )
    return probes


def validate_tokenizer(tokenizer: Any, snapshot: Path) -> dict[str, Any]:
    if type(tokenizer).__name__ != EXPECTED_TOKENIZER_CLASS:
        raise ValueError(f"unexpected tokenizer class: {type(tokenizer).__name__}")
    actual_identity = {
        "length": len(tokenizer),
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "truncation_side": tokenizer.truncation_side,
    }
    expected_identity = {
        "length": EXPECTED_TOKENIZER_LENGTH,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "bos_token_id": EXPECTED_BOS_TOKEN_ID,
        "eos_token_id": EXPECTED_EOS_TOKEN_ID,
        "truncation_side": "right",
    }
    if actual_identity != expected_identity:
        raise ValueError(f"unexpected tokenizer identity: {actual_identity}")
    files: dict[str, Any] = {}
    for filename, expected_sha256 in EXPECTED_TOKENIZER_FILES.items():
        path = snapshot / filename
        require_file_hash(path, expected_sha256, label=f"tokenizer file {filename}")
        files[filename] = file_record(path, display_path=filename)
    model_metadata_files: dict[str, Any] = {}
    for filename, expected_sha256 in EXPECTED_MODEL_METADATA_FILES.items():
        path = snapshot / filename
        require_file_hash(path, expected_sha256, label=f"model file {filename}")
        model_metadata_files[filename] = file_record(path, display_path=filename)
    return {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "class": EXPECTED_TOKENIZER_CLASS,
        **actual_identity,
        "files": files,
        "model_metadata_files": model_metadata_files,
    }


def source_contract(
    *,
    request_sources: Sequence[Mapping[str, Any]],
    usage_source: Mapping[str, Any],
    trace_source: Mapping[str, Any],
    template_source: Mapping[str, Any],
    tokenizer_source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "certified_swe_jlens_prompt_source_contract",
        "episode": "publication_certified_v2_20260715/sympy__sympy-13480",
        "requests": list(request_sources),
        "usage": dict(usage_source),
        "trace": dict(trace_source),
        "chat_template": dict(template_source),
        "tokenizer": dict(tokenizer_source),
        "renderer": {
            "tool_call_argument_normalization": (
                "JSON-decode every string function.arguments and require an object"
            ),
            "add_generation_prompt": True,
            "enable_thinking": True,
            "tokenize_rendered_text_with_add_special_tokens": False,
            "expected_prompt_token_counts": list(EXPECTED_PROMPT_TOKENS),
        },
        "trace_binding": {
            "required_thinking_blocks": 9,
            "sampled_next_boundary_method": (
                "tokenize rendered_prompt + exact thinking text and remove the "
                "verified rendered_prompt token prefix"
            ),
        },
        "candidate_probe": {
            "stage_index": PROBE_STAGE_INDEX,
            "slot_marker": PROBE_SLOT_MARKER,
            "candidates": [candidate for _, candidate in PROBE_CANDIDATES],
            "method": (
                "teacher-force the certified turn-3 reasoning to the semantic slot; "
                "tokenize full context-plus-candidate strings and emit one runner "
                "prompt per candidate BPE step"
            ),
        },
        "token_id_digest_encoding": "canonical compact sorted-key JSON",
    }


def materialize(
    *,
    root: Path,
    run_dir: Path,
    template_path: Path,
    snapshot: Path,
    tokenizer: Any,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    proxy_dir = run_dir / "proxy_dumps"
    usage_path = proxy_dir / "usage.jsonl"
    trace_path = run_dir / TRACE_RELATIVE_PATH

    require_file_hash(usage_path, EXPECTED_USAGE_SHA256, label="usage.jsonl")
    require_file_hash(trace_path, EXPECTED_TRACE_SHA256, label="qwen_trace.json")
    require_file_hash(
        template_path,
        EXPECTED_TEMPLATE_SHA256,
        label="qwen3-openai-codex.jinja",
    )

    requests: list[dict[str, Any]] = []
    request_sources: list[dict[str, Any]] = []
    for index, expected_sha256 in enumerate(EXPECTED_CHAT_SHA256, 1):
        path = proxy_dir / f"chat_{index:04d}.json"
        require_file_hash(path, expected_sha256, label=path.name)
        request = validate_request(
            json.loads(path.read_text(encoding="utf-8")),
            request_index=index,
        )
        requests.append(request)
        request_sources.append(
            file_record(
                path,
                display_path=str(path.relative_to(root)),
            )
        )

    usage_records = load_usage(usage_path)
    trace_value = json.loads(trace_path.read_text(encoding="utf-8"))
    thinking_records, completion_records = extract_trace_records(trace_value)
    template = template_path.read_text(encoding="utf-8")
    tokenizer_source = validate_tokenizer(tokenizer, snapshot)

    usage_source = file_record(
        usage_path,
        display_path=str(usage_path.relative_to(root)),
    )
    trace_source = file_record(
        trace_path,
        display_path=str(trace_path.relative_to(root)),
    )
    template_source = file_record(
        template_path,
        display_path=str(template_path.relative_to(root)),
    )
    contract = source_contract(
        request_sources=request_sources,
        usage_source=usage_source,
        trace_source=trace_source,
        template_source=template_source,
        tokenizer_source=tokenizer_source,
    )
    provenance_id = sha256_json(contract)
    stage_prompts = build_stage_prompts(
        tokenizer,
        requests=requests,
        request_sources=request_sources,
        usage_records=usage_records,
        thinking_records=thinking_records,
        completion_records=completion_records,
        template=template,
        provenance_id=provenance_id,
    )
    candidate_prompts = build_candidate_probes(
        tokenizer,
        stage_prompts=stage_prompts,
        provenance_id=provenance_id,
    )

    stages_path = output_dir / "swe_jlens_stage_prompts.json"
    candidates_path = output_dir / "swe_jlens_candidate_prompts.json"
    manifest_path = output_dir / "swe_jlens_prompt_provenance.json"
    atomic_write_json(stages_path, stage_prompts)
    atomic_write_json(candidates_path, candidate_prompts)
    outputs = {
        "stage_prompts": {
            **file_record(stages_path, display_path=stages_path.name),
            "entry_count": len(stage_prompts),
            "target_token_id_overrides": 0,
        },
        "candidate_prompts": {
            **file_record(candidates_path, display_path=candidates_path.name),
            "entry_count": len(candidate_prompts),
            "candidate_count": len(PROBE_CANDIDATES),
            "target_token_id_overrides": len(candidate_prompts),
        },
    }
    manifest = {
        "schema_version": 1,
        "kind": "certified_swe_jlens_prompt_bundle",
        "provenance_id": provenance_id,
        "source_contract": contract,
        "source_contract_sha256": provenance_id,
        "outputs": outputs,
    }
    atomic_write_json(manifest_path, manifest)
    return stages_path, candidates_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="permit a pinned Hugging Face download when the snapshot is not cached",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve(strict=True)
    run_dir = (args.run_dir or (root / RUN_RELATIVE_PATH)).resolve(strict=True)
    template_path = (
        args.template or (root / TEMPLATE_RELATIVE_PATH)
    ).resolve(strict=True)
    output_dir = (args.output_dir or (root / DEFAULT_OUTPUT_DIR)).resolve()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if args.model_snapshot is None:
        snapshot = Path(
            snapshot_download(
                MODEL_REPO,
                revision=MODEL_REVISION,
                local_files_only=not args.allow_download,
            )
        ).resolve(strict=True)
    else:
        snapshot = args.model_snapshot.resolve(strict=True)
    if snapshot.name != MODEL_REVISION:
        raise ValueError(
            f"model snapshot must end in the pinned revision {MODEL_REVISION}"
        )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    paths = materialize(
        root=root,
        run_dir=run_dir,
        template_path=template_path,
        snapshot=snapshot,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )
    for path in paths:
        print(
            f"wrote {path} (bytes={path.stat().st_size}, sha256={sha256_file(path)})"
        )


if __name__ == "__main__":
    main()
