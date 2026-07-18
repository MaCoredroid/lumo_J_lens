#!/usr/bin/env python3
"""Materialize label-independent behavioral probes from full Qwen SWE runs.

The materializer deliberately has no dataset input. Future targets come only
from the generated patch, captured mutation completions, and the agent's
terminal result summary. Benchmark gold patches and test patches are therefore
outside this program's input surface.
"""

from __future__ import annotations

import argparse
import io
import copy
import json
import keyword
import math
import os
from pathlib import Path
import re
import tempfile
import tokenize
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

import materialize_swe_multitask_c1_probes as C1
import materialize_swe_multitask_initial_probes as C0


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "configs/swe_behavioral_campaign.json"
DEFAULT_ACTION_PROTOCOL = ROOT / "configs/swe_stage_action_probes.json"
DEFAULT_TEMPLATE = ROOT / "configs/qwen3-openai-codex.jinja"
DEFAULT_COHORT_MANIFEST = ROOT / "configs/swe_behavioral_n20_cohort.json"
DEFAULT_OUTPUT = ROOT / ".cache/swe_behavioral/prompts.json"
CAMPAIGN_KIND = "swe_verified_behavioral_trajectory_campaign"
PROMPT_KIND = "swe_verified_behavioral_probe"
SUMMARY_KIND = "swe_verified_behavioral_probe_materialization"
MAX_CHECKPOINTS = 8
SEED_BASE = 880_001_233
ACTION_IDS = ("inspect", "edit", "validate", "finalize")
OUTCOME_IDS = ("success", "failure")
EXPECTED_SAMPLING = {
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "stream": True,
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {"enable_thinking": True},
}
REQUESTED_MAX_TOKENS = 8192
CONTEXT_FIT_MARGIN = 64
CONTEXT_FIT_MIN_ROOM = 256

IDENTIFIER_RE = re.compile(r"(?u)(?:[^\W\d]|_)\w*")
ASCII_SEGMENT_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)
SHELL_RESULT_RE = re.compile(
    r"Command: (?P<command>.*?)\nDirectory: (?P<directory>.*?)\nOutput: "
    r"(?P<output>.*?)\nError: (?P<error>.*?)\nExit Code: (?P<exit>-?\d+)\n"
    r"Signal: (?P<signal>-?\d+)\nProcess Group PGID: (?P<pgid>-?\d+)",
    flags=re.DOTALL,
)
GENERIC_FAILURE_RE = re.compile(
    r"(?i)(?:\b(?:error|failed|failure|exception)\b|traceback \(most recent call last\))"
)
READ_SEARCH_RE = re.compile(
    r"(?i)(?:^|[;&|]\s*|\n\s*)(?:"
    r"cat|head|tail|less|more|rg|grep|find|ls|tree|pwd|git\s+(?:diff|status|log|show)|"
    r"sed\s+-n|python(?:3|[0-9.]*)?\s+-c|wc|file"
    r")\b"
)
REDIRECTION_TARGET_RE = re.compile(r">{1,2}\s*['\"]?([^\s'\";&|]+)")
NON_REPOSITORY_PREFIXES = (
    "/dev/",
    "/proc/",
    "/sys/",
    "/tmp/",
    "/var/tmp/",
    "/run/",
)
VALIDATION_POSITIVE_RES = (
    re.compile(r"(?ims)\bRan\s+[1-9][0-9]*\s+tests?\b.*?^\s*OK\s*$"),
    re.compile(r"(?i)\b[1-9][0-9]*\s+passed\b"),
    re.compile(r"(?i)\ball tests passed\b"),
)
VALIDATION_NEGATIVE_RES = (
    re.compile(r"(?im)^\s*(?:FAILED|FAILURES?|ERRORS?)\b"),
    re.compile(r"(?i)\b[1-9][0-9]*\s+(?:failed|failures?|errors?)\b"),
    re.compile(r"(?i)\b0\s+(?:tests?\s+)?passed\b"),
    re.compile(r"(?im)^\s*Ran\s+0\s+tests?\b"),
    re.compile(r"(?i)\b(?:no tests ran|no tests? (?:were )?found|collected 0 items?)\b"),
    re.compile(r"(?i)\bTraceback \(most recent call last\)"),
    re.compile(r"(?i)\b(?:ModuleNotFoundError|ImportError|command not found|No module named)\b"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be a nonempty string")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    rendered = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _relative_file(root: Path, path: Path, label: str) -> str:
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(resolved_root), f"{label} escapes the run root")
    return resolved.relative_to(resolved_root).as_posix()


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [piece for item in value for piece in _flatten_strings(item)]
    if isinstance(value, dict):
        return [piece for key in sorted(value) for piece in _flatten_strings(value[key])]
    return []


def _flatten_text(value: Any, label: str) -> str:
    def visit(item: Any) -> list[str]:
        if isinstance(item, str):
            return [item]
        if isinstance(item, list):
            return [piece for child in item for piece in visit(child)]
        if isinstance(item, dict):
            require(item.get("type") == "text", f"{label} contains an unsupported content block")
            text = item.get("text")
            require(isinstance(text, str), f"{label} text block has no string text")
            return [text]
        raise MaterializationError(f"{label} contains an unsupported content value")

    pieces = visit(value)
    require(bool(pieces), f"{label} contains no text")
    return "\n".join(pieces)


def validate_campaign(campaign: Mapping[str, Any]) -> list[str]:
    require(campaign.get("schema_version") == 1, "campaign schema mismatch")
    require(campaign.get("kind") == CAMPAIGN_KIND, "campaign kind mismatch")
    selection = mapping(campaign.get("selection"), "campaign selection")
    require(selection.get("lens_outputs_used") is False, "campaign selection used lens output")
    generation = mapping(campaign.get("generation"), "campaign generation")
    for field in (
        "model_repo_id",
        "model_revision",
        "served_model",
        "qwen_code_version",
    ):
        nonempty_string(generation.get(field), f"campaign generation {field}")
    instance_ids = [
        nonempty_string(value, f"campaign instance {index}")
        for index, value in enumerate(sequence(campaign.get("instance_ids"), "instances"))
    ]
    require(bool(instance_ids), "campaign must select at least one instance")
    require(len(instance_ids) == len(set(instance_ids)), "campaign instance IDs repeat")
    return instance_ids


def _validate_class_records(
    value: Any,
    *,
    expected_ids: Sequence[str],
    tokenizer: Any,
    seen_ids: set[int],
    label: str,
) -> list[dict[str, Any]]:
    records = [mapping(item, label) for item in sequence(value, label)]
    require([record.get("id") for record in records] == list(expected_ids), f"{label} order changed")
    normalized: list[dict[str, Any]] = []
    class_sizes: set[int] = set()
    for record in records:
        class_id = str(record["id"])
        tokens: list[dict[str, Any]] = []
        for raw_token in sequence(record.get("tokens"), f"{label}.{class_id}.tokens"):
            token = mapping(raw_token, "class token")
            text = nonempty_string(token.get("text"), "class token text")
            token_id = token.get("token_id")
            require(
                isinstance(token_id, int)
                and not isinstance(token_id, bool)
                and token_id >= 0
                and token_id not in seen_ids,
                "class token ID is invalid or overlaps another class",
            )
            require(
                tokenizer.encode(text, add_special_tokens=False) == [token_id],
                f"class form {text!r} is not its pinned single token",
            )
            decoded = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            require(decoded == text, f"class token {token_id} no longer decodes exactly")
            seen_ids.add(token_id)
            tokens.append({"text": text, "token_id": token_id})
        require(bool(tokens), f"{label}.{class_id} is empty")
        class_sizes.add(len(tokens))
        normalized.append({"id": class_id, "tokens": tokens})
    require(len(class_sizes) == 1, f"{label} vocabulary sizes differ")
    return normalized


def _compile_patterns(value: Any, label: str) -> list[re.Pattern[str]]:
    result: list[re.Pattern[str]] = []
    for index, raw in enumerate(sequence(value, label)):
        text = nonempty_string(raw, f"{label}[{index}]")
        try:
            result.append(re.compile(text))
        except re.error as error:
            raise ValueError(f"invalid {label}[{index}]: {error}") from error
    require(bool(result), f"{label} is empty")
    return result


def validate_action_protocol(
    protocol: Mapping[str, Any], *, tokenizer: Any, campaign: Mapping[str, Any]
) -> dict[str, Any]:
    require(protocol.get("schema_version") == 1, "action protocol schema mismatch")
    require(
        protocol.get("kind") == "swe_verified_stage_action_probe_protocol",
        "action protocol kind mismatch",
    )
    require(protocol.get("lens_outputs_used_for_labels") is False, "action labels used lens output")
    pins = mapping(protocol.get("pins"), "action protocol pins")
    model_pin = mapping(pins.get("model"), "action model pin")
    tokenizer_pin = mapping(pins.get("tokenizer"), "action tokenizer pin")
    generation = mapping(campaign.get("generation"), "campaign generation")
    require(
        model_pin.get("repo_id") == generation.get("model_repo_id")
        and model_pin.get("revision") == generation.get("model_revision"),
        "campaign/action model pins differ",
    )
    require(
        isinstance(tokenizer_pin.get("vocabulary_size"), int)
        and len(tokenizer) == tokenizer_pin["vocabulary_size"],
        "tokenizer vocabulary size differs from the action protocol",
    )
    seen_ids: set[int] = set()
    actions = _validate_class_records(
        protocol.get("action_classes"),
        expected_ids=ACTION_IDS,
        tokenizer=tokenizer,
        seen_ids=seen_ids,
        label="action classes",
    )
    outcomes = _validate_class_records(
        protocol.get("outcome_classes"),
        expected_ids=OUTCOME_IDS,
        tokenizer=tokenizer,
        seen_ids=seen_ids,
        label="outcome classes",
    )
    classifier = mapping(protocol.get("next_completion_classifier"), "action classifier")
    require(
        classifier.get("action_precedence")
        == [
            "terminal_no_tool_response",
            "mutating_source_command",
            "test_command",
            "read_or_search_command",
            "validation_intent_assistant_text",
        ],
        "action-class precedence changed",
    )
    require(
        classifier.get("unclassified_policy") == "missing_not_imputed",
        "unclassified action policy changed",
    )
    return {
        "action_classes": actions,
        "outcome_classes": outcomes,
        "action_token_ids": [token["token_id"] for row in actions for token in row["tokens"]],
        "outcome_token_ids": [token["token_id"] for row in outcomes for token in row["tokens"]],
        "fixed_token_ids": set(seen_ids),
        "diagnosis_regexes": _compile_patterns(
            classifier.get("diagnosis_assistant_text_regexes"), "diagnosis regexes"
        ),
        "mutation_regexes": _compile_patterns(
            classifier.get("mutating_command_regexes"), "mutation regexes"
        ),
        "test_regexes": _compile_patterns(classifier.get("test_command_regexes"), "test regexes"),
        "validation_intent_regexes": _compile_patterns(
            classifier.get("validation_assistant_text_regexes"),
            "validation-intent regexes",
        ),
        "inspection_tools": set(
            nonempty_string(value, "inspection tool name")
            for value in sequence(
                classifier.get("generic_inspection_tool_names"), "inspection tools"
            )
        ),
        "mutation_tools": set(
            nonempty_string(value, "mutation tool name")
            for value in sequence(
                classifier.get("generic_mutation_tool_names"), "mutation tools"
            )
        ),
    }


def uniform_request_indices(count: int, *, limit: int = MAX_CHECKPOINTS) -> list[int]:
    require(isinstance(count, int) and not isinstance(count, bool) and count >= 0, "invalid request count")
    require(isinstance(limit, int) and not isinstance(limit, bool) and limit >= 1, "invalid checkpoint limit")
    if count <= limit:
        return list(range(1, count + 1))
    indices = [1 + math.floor(index * (count - 1) / (limit - 1)) for index in range(limit)]
    require(indices[0] == 1 and indices[-1] == count, "uniform endpoints changed")
    require(len(indices) == len(set(indices)), "uniform checkpoint selection duplicated an index")
    return indices


def _terminal_trace_serving_evidence(task: Mapping[str, Any]) -> dict[str, Any]:
    path = task.get("qwen_trace_path")
    if not isinstance(path, Path):
        return {
            "status": "unknown",
            "derivation": "terminal_qwen_trace_absent",
            "trace_sha256": None,
        }
    value = json.loads(path.read_bytes())
    require(isinstance(value, list), "qwen trace is not a JSON array")
    assistants = [
        mapping(row, "qwen trace row")
        for row in value
        if isinstance(row, dict) and row.get("type") == "assistant"
    ]
    if not assistants:
        return {
            "status": "unknown",
            "derivation": "terminal_qwen_trace_has_no_assistant_record",
            "trace_sha256": sha256_file(path),
        }
    message = mapping(assistants[-1].get("message"), "terminal qwen trace assistant message")
    usage = mapping(message.get("usage"), "terminal qwen trace usage")
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    require(
        isinstance(input_tokens, int)
        and not isinstance(input_tokens, bool)
        and input_tokens >= 0
        and isinstance(output_tokens, int)
        and not isinstance(output_tokens, bool)
        and output_tokens >= 0,
        "terminal qwen trace usage is invalid",
    )
    content = "\n".join(_flatten_strings(message.get("content")))
    if input_tokens > 0 or output_tokens > 0:
        status = "served"
        derivation = "terminal_qwen_trace_nonzero_usage"
    elif re.search(r"(?i)\[API Error:\s*400\b", content):
        status = "unserved"
        derivation = "terminal_qwen_trace_zero_usage_api_error_400"
    else:
        status = "unknown"
        derivation = "terminal_qwen_trace_zero_usage_without_supported_status"
    return {
        "status": status,
        "derivation": derivation,
        "trace_sha256": sha256_file(path),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "assistant_content_sha256": sha256_text(content),
    }


def select_probeable_requests(
    task: Mapping[str, Any], *, max_prompt_tokens: int, limit: int = MAX_CHECKPOINTS
) -> dict[str, Any]:
    captures = [mapping(value, "task capture") for value in sequence(task.get("captures"), "task captures")]
    require(max_prompt_tokens >= 1, "maximum replay prompt tokens is invalid")
    terminal_trace = _terminal_trace_serving_evidence(task)
    candidates: list[int] = []
    excluded: list[dict[str, Any]] = []
    serving_evidence: dict[int, dict[str, Any]] = {}
    for local_index, capture in enumerate(captures, 1):
        usage = mapping(capture.get("usage"), "capture usage")
        if usage.get("telemetry_status") == "available":
            service = {
                "status": "served",
                "derivation": "bound_proxy_usage_record",
            }
        elif local_index < len(captures):
            service = {
                "status": "served",
                "derivation": "exact_following_raw_extension_materializes_completion",
            }
        else:
            service = copy.deepcopy(terminal_trace)
        serving_evidence[local_index] = service
        token_count = len(sequence(capture.get("token_ids"), "capture token IDs"))
        reasons: list[str] = []
        if service.get("status") != "served":
            reasons.append(str(service.get("derivation")))
        if token_count > max_prompt_tokens:
            reasons.append("canonical_prompt_exceeds_one_token_replay_ceiling")
        if reasons:
            excluded.append(
                {
                    "task_request_index": local_index,
                    "global_request_index": capture["global_index"],
                    "prompt_token_count": token_count,
                    "usage_telemetry_status": usage.get("telemetry_status"),
                    "serving_evidence": service,
                    "reasons": reasons,
                }
            )
        else:
            candidates.append(local_index)
    candidate_ordinals = uniform_request_indices(len(candidates), limit=limit)
    selected = [candidates[ordinal - 1] for ordinal in candidate_ordinals]
    return {
        "algorithm": "uniform_probeable_request_indices_v1",
        "max_prompt_tokens": max_prompt_tokens,
        "probeable_request_indices": candidates,
        "excluded_request_indices": [row["task_request_index"] for row in excluded],
        "excluded_requests": excluded,
        "selected_request_indices": selected,
        "selected_candidate_ordinals": candidate_ordinals,
        "serving_evidence": serving_evidence,
        "terminal_trace_serving_evidence": terminal_trace,
    }


def _read_usage(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    require(all(line.strip() for line in lines), "usage ledger contains blank rows")
    return [dict(mapping(json.loads(line), f"usage row {index}")) for index, line in enumerate(lines, 1)]


def _load_task_metadata(run_root: Path, instance_ids: Sequence[str]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for selection_index, instance_id in enumerate(instance_ids):
        task_root = run_root / "generation/verified/per_task" / instance_id
        metadata_path = task_root / "runner_metadata.json"
        require(metadata_path.is_file(), f"runner metadata is missing for {instance_id}")
        metadata = mapping(json.loads(metadata_path.read_bytes()), f"{instance_id} runner metadata")
        require(metadata.get("instance_id") == instance_id, "runner metadata task order mismatch")
        qwen_value = metadata.get("qwen")
        if isinstance(qwen_value, dict):
            num_turns = qwen_value.get("num_turns")
            require(
                isinstance(num_turns, int) and not isinstance(num_turns, bool) and num_turns >= 0,
                f"{instance_id} qwen.num_turns is invalid",
            )
        else:
            num_turns = 0
        patch_path = task_root / "patch.diff"
        qwen_trace_path = task_root / "qwen_trace.json"
        tasks.append(
            {
                "selection_index": selection_index,
                "instance_id": instance_id,
                "task_root": task_root,
                "metadata_path": metadata_path,
                "metadata": dict(metadata),
                "request_count": num_turns,
                "patch_path": patch_path if patch_path.is_file() else None,
                "qwen_trace_path": qwen_trace_path if qwen_trace_path.is_file() else None,
            }
        )
    return tasks


def _validate_request_sampling(
    request: Mapping[str, Any],
    *,
    global_index: int,
    served_model: str,
    context_limit: int,
) -> None:
    require(request.get("model") == served_model, "capture model differs from campaign")
    for field, expected in EXPECTED_SAMPLING.items():
        require(request.get(field) == expected, f"capture sampling field {field} changed")
    serialized_prompt_chars = sum(
        len(json.dumps(request[key], ensure_ascii=False))
        for key in ("messages", "tools")
        if request.get(key) is not None
    )
    estimated_prompt_tokens = serialized_prompt_chars // 4
    room = context_limit - estimated_prompt_tokens
    if room >= CONTEXT_FIT_MIN_ROOM and room - CONTEXT_FIT_MARGIN >= 1:
        expected_max_tokens = min(
            REQUESTED_MAX_TOKENS,
            room - CONTEXT_FIT_MARGIN,
        )
    else:
        expected_max_tokens = REQUESTED_MAX_TOKENS
    require(
        request.get("max_tokens") == expected_max_tokens,
        "capture max_tokens differs from the frozen proxy context-fit policy",
    )
    require(request.get("seed") == SEED_BASE + global_index, "capture seed sequence changed")
    tools = [mapping(item, "capture tool") for item in sequence(request.get("tools"), "capture tools")]
    require(len(tools) == 1, "capture must expose exactly one model tool")
    function = mapping(tools[0].get("function"), "capture tool function")
    require(
        tools[0].get("type") == "function" and function.get("name") == "run_shell_command",
        "capture tool boundary changed",
    )


def map_global_captures(
    *,
    run_root: Path,
    campaign: Mapping[str, Any],
    task_records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    template: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proxy_dir = run_root / "proxy_dumps"
    usage_path = proxy_dir / "usage.jsonl"
    require(proxy_dir.is_dir() and usage_path.is_file(), "global proxy captures are incomplete")
    request_paths = sorted(proxy_dir.glob("chat_*.json"))
    usage_records = _read_usage(usage_path)
    expected_total = sum(int(task["request_count"]) for task in task_records)
    require(len(request_paths) == expected_total, "global chat count differs from frozen task turn counts")
    for index, path in enumerate(request_paths, 1):
        require(path.name == f"chat_{index:04d}.json", "global chat files are not contiguous")
    usage_by_index: dict[int, dict[str, Any]] = {}
    for row in usage_records:
        index = row.get("idx")
        require(
            isinstance(index, int)
            and not isinstance(index, bool)
            and 1 <= index <= expected_total,
            "usage index is outside the frozen global request range",
        )
        require(index not in usage_by_index, "usage index repeats")
        usage_by_index[index] = row
    missing_usage_indices = sorted(set(range(1, expected_total + 1)) - set(usage_by_index))
    generation = mapping(campaign.get("generation"), "campaign generation")
    served_model = nonempty_string(generation.get("served_model"), "served model")
    context_limit = generation.get("max_model_len")
    require(
        isinstance(context_limit, int)
        and not isinstance(context_limit, bool)
        and context_limit >= CONTEXT_FIT_MIN_ROOM,
        "campaign max_model_len is invalid",
    )
    usage_sha256 = sha256_file(usage_path)
    mapped_tasks: list[dict[str, Any]] = []
    global_offset = 0
    campaign_tools: Any = None
    for raw_task in task_records:
        task = dict(raw_task)
        request_count = int(task["request_count"])
        records: list[dict[str, Any]] = []
        previous_messages: list[Any] | None = None
        previous_rendered = ""
        previous_token_ids: list[int] = []
        for local_index in range(1, request_count + 1):
            global_index = global_offset + local_index
            path = request_paths[global_index - 1]
            request = mapping(json.loads(path.read_bytes()), f"request {global_index}")
            observed_usage = usage_by_index.get(global_index)
            usage = (
                dict(observed_usage)
                if observed_usage is not None
                else {
                    "idx": global_index,
                    "telemetry_status": "missing",
                    "usage": None,
                    "finish_reason": None,
                }
            )
            usage.setdefault("telemetry_status", "available")
            _validate_request_sampling(
                request,
                global_index=global_index,
                served_model=served_model,
                context_limit=context_limit,
            )
            if campaign_tools is None:
                campaign_tools = copy.deepcopy(request.get("tools"))
            require(request.get("tools") == campaign_tools, "campaign tool schema drifted")
            messages = sequence(request.get("messages"), f"request {global_index} messages")
            if local_index == 1:
                require(
                    [mapping(message, "initial message").get("role") for message in messages]
                    == ["system", "user"],
                    f"{task['instance_id']} does not start at an exact task boundary",
                )
                require(
                    str(task["instance_id"])
                    in json.dumps(messages, sort_keys=True, ensure_ascii=False),
                    f"task-start capture does not contain {task['instance_id']}",
                )
            else:
                assert previous_messages is not None
                require(
                    messages[: len(previous_messages)] == previous_messages,
                    f"{task['instance_id']} raw request prefix drifted",
                )
                require(len(messages) > len(previous_messages), "request appended no completion")
            rendered, token_ids, normalized_count, normalized_messages = C1.render_request(
                tokenizer, request=request, template=template
            )
            if local_index > 1:
                require(rendered.startswith(previous_rendered), "canonical rendered prefix drifted")
                require(
                    token_ids[: len(previous_token_ids)] == previous_token_ids,
                    "canonical token prefix drifted",
                )
            require(usage.get("idx") == global_index, "usage/global request index mismatch")
            if observed_usage is not None:
                usage_values = mapping(usage.get("usage"), "usage token counts")
                require(
                    usage_values.get("prompt_tokens") == len(token_ids),
                    "usage prompt token count differs from canonical rendering",
                )
                completion_tokens = usage_values.get("completion_tokens")
                total_tokens = usage_values.get("total_tokens")
                require(
                    isinstance(completion_tokens, int)
                    and completion_tokens >= 0
                    and total_tokens == len(token_ids) + completion_tokens,
                    "usage completion/total token counts are inconsistent",
                )
                finish_reason = usage.get("finish_reason")
                require(
                    finish_reason in {"tool_calls", "stop", "length"},
                    "unsupported finish reason",
                )
                if local_index < request_count:
                    require(finish_reason == "tool_calls", "nonterminal capture did not call a tool")
            records.append(
                {
                    "local_index": local_index,
                    "global_index": global_index,
                    "path": _relative_file(run_root, path, "raw request"),
                    "sha256": sha256_file(path),
                    "request": dict(request),
                    "messages": copy.deepcopy(messages),
                    "rendered": rendered,
                    "token_ids": list(token_ids),
                    "normalized_messages_sha256": sha256_json(normalized_messages),
                    "normalized_string_tool_call_arguments": normalized_count,
                    "usage": dict(usage),
                }
            )
            previous_messages = copy.deepcopy(messages)
            previous_rendered = rendered
            previous_token_ids = list(token_ids)
        task["captures"] = records
        task["global_request_start"] = global_offset + 1 if request_count else None
        task["global_request_end"] = global_offset + request_count if request_count else None
        mapped_tasks.append(task)
        global_offset += request_count
    require(global_offset == expected_total, "campaign mapping did not consume every capture")
    return mapped_tasks, {
        "usage_path": _relative_file(run_root, usage_path, "usage ledger"),
        "usage_sha256": usage_sha256,
        "global_request_count": expected_total,
        "usage_record_count": len(usage_records),
        "missing_usage_indices": missing_usage_indices,
        "mapping_algorithm": "campaign_order_cumulative_runner_num_turns_v1",
        "exact_raw_request_coverage": True,
        "exact_usage_index_coverage": not missing_usage_indices,
        "exact_global_coverage": not missing_usage_indices,
    }


def _assistant_text(assistant: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    reasoning = assistant.get("reasoning_content")
    duplicate = assistant.get("reasoning")
    if isinstance(reasoning, str):
        require(duplicate in {None, reasoning}, "assistant reasoning fields disagree")
        pieces.append(reasoning)
    elif isinstance(duplicate, str):
        pieces.append(duplicate)
    content = assistant.get("content")
    if isinstance(content, str):
        pieces.append(content)
    elif content is not None:
        pieces.extend(_flatten_strings(content))
    return "\n".join(piece for piece in pieces if piece)


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str]:
    text = nonempty_string(raw, "tool arguments")
    try:
        value = mapping(json.loads(text), "tool arguments")
    except json.JSONDecodeError as error:
        raise ValueError("tool arguments are not valid JSON") from error
    return dict(value), text


def _shell_result(text: str) -> dict[str, Any] | None:
    match = SHELL_RESULT_RE.fullmatch(text)
    if match is None:
        return None
    values = match.groupdict()
    return {
        "command": values["command"],
        "directory": values["directory"],
        "output": values["output"],
        "error": values["error"],
        "exit_code": int(values["exit"]),
        "signal": int(values["signal"]),
    }


def _repository_target(value: str) -> bool:
    target = value.strip().strip("'\"").rstrip(",)")
    if not target or target == "-" or target.startswith(NON_REPOSITORY_PREFIXES):
        return False
    if target.startswith(("$", "~")):
        return False
    path = Path(target)
    if path.is_absolute():
        return target.startswith(("/testbed/", "/workspace/"))
    return ".." not in path.parts


def _patch_application_mutates(command: str) -> bool:
    for segment in re.split(r"[;&|\n]+", command):
        if re.search(r"(?i)\bapply_patch\b", segment):
            return True
        if re.search(r"(?i)\bgit\s+apply\b", segment):
            if re.search(r"(?i)\B--(?:check|stat|numstat|summary)\b", segment):
                continue
            return True
        if re.search(r"(?i)\bpatch\b[^\n;|&]*\s-p\d+\b", segment):
            if re.search(r"(?i)\B--dry-run\b", segment):
                continue
            return True
    return False


def is_source_mutation(
    tool_name: str,
    arguments: Mapping[str, Any],
    argument_text: str,
    *,
    mutation_tools: set[str],
    mutation_regexes: Sequence[re.Pattern[str]],
) -> tuple[bool, list[str]]:
    rule_hits = [
        pattern.pattern for pattern in mutation_regexes if pattern.search(argument_text)
    ]
    if tool_name in mutation_tools:
        target = next(
            (
                arguments.get(field)
                for field in ("path", "file_path", "filename")
                if isinstance(arguments.get(field), str)
            ),
            None,
        )
        if isinstance(target, str) and _repository_target(target):
            return True, [*rule_hits, f"tool_name:{tool_name}"]
        if tool_name == "apply_patch":
            return True, [*rule_hits, f"tool_name:{tool_name}"]
    if not rule_hits:
        return False, []
    if _patch_application_mutates(argument_text):
        return True, rule_hits
    if re.search(r"(?im)\bsed\b[^\n;|]*\s-i(?:[.A-Za-z0-9_-]*)?\b", argument_text):
        return True, rule_hits
    targets = [match.group(1) for match in REDIRECTION_TARGET_RE.finditer(argument_text)]
    targets.extend(
        match.group(1)
        for pattern in (
            re.compile(r"\bopen\s*\(\s*['\"]([^'\"]+)['\"]"),
            re.compile(r"\bPath\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
            re.compile(r"\btee(?:\s+-[A-Za-z]+)*\s+['\"]?([^\s'\";&|]+)"),
        )
        for match in pattern.finditer(argument_text)
    )
    return any(_repository_target(target) for target in targets), rule_hits


def _tool_success(tool_name: str, result_text: str, command: str | None) -> tuple[bool, dict[str, Any]]:
    parsed = _shell_result(result_text) if tool_name == "run_shell_command" else None
    if parsed is not None:
        require(command is not None and parsed["command"] == command, "shell command/result mismatch")
        success = (
            parsed["exit_code"] == 0
            and parsed["signal"] == 0
            and parsed["error"] == "(none)"
        )
        return success, {
            "envelope": "qwen_shell_v1",
            "exit_code": parsed["exit_code"],
            "signal": parsed["signal"],
            "output": parsed["output"],
        }
    success = GENERIC_FAILURE_RE.search(result_text) is None
    return success, {
        "envelope": "generic_text",
        "generic_failure_pattern_hit": not success,
        "output": result_text,
    }


def _validation_success(tool_audits: Sequence[Mapping[str, Any]]) -> tuple[bool, dict[str, Any]]:
    outputs = "\n".join(str(audit.get("output", "")) for audit in tool_audits)
    positive = sorted({pattern.pattern for pattern in VALIDATION_POSITIVE_RES if pattern.search(outputs)})
    negative = sorted({pattern.pattern for pattern in VALIDATION_NEGATIVE_RES if pattern.search(outputs)})
    success = bool(tool_audits) and all(bool(audit["successful"]) for audit in tool_audits)
    return success and bool(positive) and not negative, {
        "positive_output_regex_hits": positive,
        "negative_output_regex_hits": negative,
    }


def derive_completion(
    task: Mapping[str, Any], local_index: int, *, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    captures = sequence(task.get("captures"), "task captures")
    require(1 <= local_index <= len(captures), "completion index is outside the task")
    current = mapping(captures[local_index - 1], "current capture")
    usage = mapping(current.get("usage"), "current usage")
    observed_finish_reason = usage.get("finish_reason")
    finish_reason = observed_finish_reason
    finish_reason_derivation = "proxy_usage"
    if finish_reason is None and local_index < len(captures):
        finish_reason = "tool_calls"
        finish_reason_derivation = "exact_following_raw_extension"
    base = {
        "completion_index": local_index,
        "finish_reason": observed_finish_reason,
        "effective_finish_reason": finish_reason,
        "finish_reason_derivation": finish_reason_derivation,
        "source_request_global_index": current["global_index"],
        "source_request_sha256": current["sha256"],
    }
    if finish_reason is None:
        return {
            **base,
            "status": "missing_completion_telemetry",
            "assistant_text": "",
            "argument_text": "",
            "action": {
                "status": "missing",
                "class_id": None,
                "derivation": "no_usage_and_no_following_capture",
            },
            "tool_execution": {
                "status": "missing",
                "class_id": None,
                "derivation": "no_usage_and_no_following_capture",
            },
            "validation": {
                "status": "missing",
                "class_id": None,
                "derivation": "no_usage_and_no_following_capture",
            },
            "next_request_global_index": None,
            "next_request_sha256": None,
            "extension_sha256": None,
            "tool_audits": [],
        }
    if finish_reason == "stop":
        return {
            **base,
            "status": "terminal",
            "assistant_text": "",
            "argument_text": "",
            "action": {"status": "available", "class_id": "finalize", "derivation": "terminal_stop"},
            "tool_execution": {
                "status": "not_applicable",
                "class_id": None,
                "derivation": "terminal_completion_has_no_tool_execution",
            },
            "validation": {
                "status": "not_applicable",
                "class_id": None,
                "derivation": "terminal_completion_is_not_validation",
            },
            "next_request_global_index": None,
            "next_request_sha256": None,
            "extension_sha256": None,
            "tool_audits": [],
        }
    if finish_reason == "length":
        return {
            **base,
            "status": "truncated",
            "assistant_text": "",
            "argument_text": "",
            "action": {"status": "missing", "class_id": None, "derivation": "length_no_imputation"},
            "tool_execution": {
                "status": "not_applicable",
                "class_id": None,
                "derivation": "truncated_before_observed_tool_execution",
            },
            "validation": {
                "status": "not_applicable",
                "class_id": None,
                "derivation": "truncated_completion",
            },
            "next_request_global_index": None,
            "next_request_sha256": None,
            "extension_sha256": None,
            "tool_audits": [],
        }
    require(finish_reason == "tool_calls", "unexpected completion finish reason")
    if local_index == len(captures):
        return {
            **base,
            "status": "unobserved_after_task_end",
            "assistant_text": "",
            "argument_text": "",
            "action": {"status": "missing", "class_id": None, "derivation": "no_following_capture"},
            "tool_execution": {
                "status": "missing",
                "class_id": None,
                "derivation": "no_following_capture",
            },
            "validation": {
                "status": "not_applicable",
                "class_id": None,
                "derivation": "no_following_capture",
            },
            "next_request_global_index": None,
            "next_request_sha256": None,
            "extension_sha256": None,
            "tool_audits": [],
        }
    following = mapping(captures[local_index], "following capture")
    current_messages = sequence(current.get("messages"), "current messages")
    following_messages = sequence(following.get("messages"), "following messages")
    require(
        following_messages[: len(current_messages)] == current_messages,
        "following request is not an exact raw extension",
    )
    extension = following_messages[len(current_messages) :]
    require(bool(extension), "following request materializes no completion")
    assistant = mapping(extension[0], "completion assistant message")
    require(assistant.get("role") == "assistant", "completion does not start with assistant")
    tool_messages = [mapping(item, "completion tool result") for item in extension[1:]]
    require(all(message.get("role") == "tool" for message in tool_messages), "completion extension has non-tool tail")
    calls = [mapping(item, "completion tool call") for item in sequence(assistant.get("tool_calls"), "tool calls")]
    require(bool(calls), "tool-calling completion declares no tool calls")
    results_by_id: dict[str, Mapping[str, Any]] = {}
    for message in tool_messages:
        call_id = nonempty_string(message.get("tool_call_id"), "tool result ID")
        require(call_id not in results_by_id, "tool result ID repeats")
        results_by_id[call_id] = message
    require({str(call.get("id")) for call in calls} == set(results_by_id), "tool call/result IDs differ")
    assistant_text = _assistant_text(assistant)
    diagnosis_hits = sorted(
        {
            pattern.pattern
            for pattern in protocol["diagnosis_regexes"]
            if pattern.search(assistant_text)
        }
    )
    validation_intent_hits = sorted(
        {
            pattern.pattern
            for pattern in protocol["validation_intent_regexes"]
            if pattern.search(assistant_text)
        }
    )
    argument_texts: list[str] = []
    tool_audits: list[dict[str, Any]] = []
    any_mutation = False
    any_validation = False
    any_inspection = False
    for call_index, call in enumerate(calls):
        require(call.get("type") == "function", "tool call type changed")
        function = mapping(call.get("function"), "tool function")
        tool_name = nonempty_string(function.get("name"), "tool name")
        arguments, raw_arguments = _parse_arguments(function.get("arguments"))
        flattened_arguments = "\n".join(_flatten_strings(arguments))
        argument_texts.append(flattened_arguments)
        command_value = arguments.get("command") if tool_name == "run_shell_command" else None
        command = command_value if isinstance(command_value, str) else None
        mutation, mutation_hits = is_source_mutation(
            tool_name,
            arguments,
            flattened_arguments,
            mutation_tools=protocol["mutation_tools"],
            mutation_regexes=protocol["mutation_regexes"],
        )
        test_hits = sorted(
            {
                pattern.pattern
                for pattern in protocol["test_regexes"]
                if pattern.search(flattened_arguments)
            }
        )
        validation = bool(test_hits)
        inspection = (
            tool_name in protocol["inspection_tools"]
            or (command is not None and READ_SEARCH_RE.search(command) is not None)
        )
        result_text = _flatten_text(results_by_id[str(call["id"])].get("content"), "tool result")
        successful, outcome = _tool_success(tool_name, result_text, command)
        any_mutation = any_mutation or mutation
        any_validation = any_validation or validation
        any_inspection = any_inspection or inspection
        tool_audits.append(
            {
                "call_index": call_index,
                "tool_call_id": call["id"],
                "tool_name": tool_name,
                "arguments_sha256": sha256_text(raw_arguments),
                "result_sha256": sha256_text(result_text),
                "source_mutation": mutation,
                "mutation_rule_hits": sorted(set(mutation_hits)),
                "validation_command": validation,
                "test_rule_hits": test_hits,
                "inspection": inspection,
                "successful": successful,
                **outcome,
            }
        )
    if any_mutation:
        action_id = "edit"
        action_derivation = "source_mutation_command"
    elif any_validation:
        action_id = "validate"
        action_derivation = "validation_command"
    elif any_inspection:
        action_id = "inspect"
        action_derivation = "inspection_command"
    elif validation_intent_hits:
        action_id = "validate"
        action_derivation = "validation_intent_assistant_text"
    else:
        action_id = None
        action_derivation = "unclassified_missing_not_imputed"
    all_tools_successful = bool(tool_audits) and all(audit["successful"] for audit in tool_audits)
    if any_validation and not any_mutation:
        validation_success, validation_evidence = _validation_success(tool_audits)
        validation_label = {
            "status": "available",
            "class_id": "success" if validation_success else "failure",
            "derivation": "test_envelope_and_positive_output_v1",
            **validation_evidence,
        }
    else:
        validation_label = {
            "status": "not_applicable",
            "class_id": None,
            "derivation": (
                "mutation_precedence_over_validation" if any_mutation and any_validation else "not_a_validation_completion"
            ),
        }
    return {
        **base,
        "status": "materialized_in_following_request",
        "assistant_text": assistant_text,
        "argument_text": "\n".join(argument_texts),
        "action": {
            "status": "available" if action_id else "missing",
            "class_id": action_id,
            "derivation": action_derivation,
        },
        "tool_execution": {
            "status": "available",
            "class_id": "success" if all_tools_successful else "failure",
            "derivation": "all_materialized_tool_results_have_successful_envelopes",
        },
        "validation": validation_label,
        "next_request_global_index": following["global_index"],
        "next_request_sha256": following["sha256"],
        "extension_sha256": sha256_json(extension),
        "assistant_text_sha256": sha256_text(assistant_text),
        "argument_text_sha256": sha256_text("\n".join(argument_texts)),
        "diagnosis_expressed": bool(diagnosis_hits),
        "diagnosis_regex_hits": diagnosis_hits,
        "validation_intent_regex_hits": validation_intent_hits,
        "tool_audits": tool_audits,
    }


def _normalized_identifier(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def identifier_kind(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    if normalized.startswith("__") and normalized.endswith("__"):
        return "dunder_identifier"
    if normalized.startswith("_"):
        return "private_identifier"
    letters = "".join(character for character in normalized if character.isalpha())
    if letters and letters.isupper():
        return "constant_identifier"
    if normalized[:1].isupper() or any(character.isupper() for character in normalized[1:]):
        return "camel_identifier"
    return "identifier"


def _identifier_occurrences(text: str) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    for match in IDENTIFIER_RE.finditer(text):
        identifier = match.group(0)
        normalized = _normalized_identifier(identifier)
        if keyword.iskeyword(normalized):
            continue
        occurrences.append(
            {
                "identifier": identifier,
                "normalized": normalized,
                "kind": identifier_kind(identifier),
                "span": [match.start(), match.end()],
            }
        )
    return occurrences


def _python_code_identifier_occurrences(line: str) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(f"{line}\n").readline)
        for token in tokens:
            if token.type != tokenize.NAME or keyword.iskeyword(token.string):
                continue
            occurrences.append(
                {
                    "identifier": token.string,
                    "normalized": _normalized_identifier(token.string),
                    "kind": identifier_kind(token.string),
                    "span": [token.start[1], token.end[1]],
                }
            )
    except (IndentationError, tokenize.TokenError):
        # Tokens yielded before an incomplete continuation remain valid evidence.
        pass
    return occurrences


def parse_generated_patch(patch_text: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"added": [], "removed": [], "context": []}
    current_path: str | None = None
    for line_number, line in enumerate(patch_text.splitlines(), 1):
        match = re.fullmatch(r"diff --git a/(.+) b/(.+)", line)
        if match is not None:
            require(match.group(1) == match.group(2), "renamed files are unsupported by target derivation")
            current_path = match.group(2)
            continue
        if current_path is None or line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("+"):
            category = "added"
        elif line.startswith("-"):
            category = "removed"
        elif line.startswith(" "):
            category = "context"
        else:
            continue
        if Path(current_path).suffix.casefold() not in {".py", ".pyi", ".pyx"}:
            continue
        body = line[1:]
        for occurrence in _python_code_identifier_occurrences(body):
            result[category].append(
                {
                    **occurrence,
                    "path": current_path,
                    "patch_line_number": line_number,
                    "line_sha256": sha256_text(body),
                    "category": category,
                }
            )
    return result


def _token_forms(value: str, tokenizer: Any, *, forbidden_ids: set[int]) -> list[dict[str, Any]]:
    forms: list[dict[str, Any]] = []
    seen: set[int] = set()
    for kind, text in (("bare", value), ("leading_space", f" {value}")):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        token_id = int(token_ids[0])
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != text or token_id in forbidden_ids or token_id in seen:
            continue
        seen.add(token_id)
        forms.append({"kind": kind, "text": text, "token_id": token_id})
    return forms


def _first_occurrence(values: Iterable[Mapping[str, Any]], normalized: str) -> Mapping[str, Any] | None:
    return next((value for value in values if value.get("normalized") == normalized), None)


def derive_task_targets(
    *,
    task: Mapping[str, Any],
    completions: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    fixed_token_ids: set[int],
    run_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    patch_path_value = task.get("patch_path")
    metadata = mapping(task.get("metadata"), "runner metadata")
    qwen = metadata.get("qwen") if isinstance(metadata.get("qwen"), dict) else {}
    terminal_summary = qwen.get("result_tail") if isinstance(qwen.get("result_tail"), str) else ""
    if not isinstance(patch_path_value, Path):
        return [], {
            "status": "no_generated_patch",
            "added_identifier_count": 0,
            "mutation_identifier_count": 0,
            "terminal_identifier_count": len(_identifier_occurrences(terminal_summary)),
            "intersection_count": 0,
        }
    patch_path = patch_path_value
    patch_text = patch_path.read_text(encoding="utf-8")
    patch_records = parse_generated_patch(patch_text)
    mutation_occurrences: list[dict[str, Any]] = []
    for completion in completions:
        action = mapping(completion.get("action"), "completion action")
        if action.get("class_id") != "edit":
            continue
        for channel in ("assistant_text", "argument_text"):
            if channel == "assistant_text" and not completion.get("diagnosis_expressed"):
                continue
            text = str(completion.get(channel, ""))
            for occurrence in _identifier_occurrences(text):
                mutation_occurrences.append(
                    {
                        **occurrence,
                        "completion_index": completion["completion_index"],
                        "source_request_global_index": completion["source_request_global_index"],
                        "next_request_global_index": completion.get("next_request_global_index"),
                        "channel": channel,
                        "channel_text_sha256": sha256_text(text),
                    }
                )
    terminal_occurrences = [
        {**occurrence, "field": "/qwen/result_tail", "text_sha256": sha256_text(terminal_summary)}
        for occurrence in _identifier_occurrences(terminal_summary)
    ]
    added_keys = {value["normalized"] for value in patch_records["added"]}
    mutation_keys = {value["normalized"] for value in mutation_occurrences}
    terminal_keys = {value["normalized"] for value in terminal_occurrences}
    intersection = sorted(added_keys & mutation_keys & terminal_keys)
    targets: list[dict[str, Any]] = []
    rejected = {"no_single_token_form": 0, "no_same_kind_foil": 0}
    for normalized in intersection:
        added = _first_occurrence(patch_records["added"], normalized)
        mutation = _first_occurrence(mutation_occurrences, normalized)
        terminal = _first_occurrence(terminal_occurrences, normalized)
        assert added is not None and mutation is not None and terminal is not None
        target_text = str(added["identifier"])
        forms = _token_forms(target_text, tokenizer, forbidden_ids=fixed_token_ids)
        if not forms:
            rejected["no_single_token_form"] += 1
            continue
        target_kind = str(added["kind"])
        foil_candidates: dict[str, Mapping[str, Any]] = {}
        for category_rank, category in enumerate(("removed", "context")):
            for occurrence in patch_records[category]:
                foil_key = str(occurrence["normalized"])
                if foil_key == normalized or occurrence["kind"] != target_kind:
                    continue
                existing = foil_candidates.get(foil_key)
                rank = (
                    occurrence["path"] != added["path"],
                    category_rank,
                    occurrence["path"],
                    occurrence["patch_line_number"],
                    foil_key,
                )
                if existing is None or rank < existing["_rank"]:
                    foil_candidates[foil_key] = {**occurrence, "_rank": rank}
        foils: list[dict[str, Any]] = []
        used_token_ids = set(fixed_token_ids) | {form["token_id"] for form in forms}
        for foil_key, raw_foil in sorted(
            foil_candidates.items(), key=lambda item: item[1]["_rank"]
        ):
            foil_text = str(raw_foil["identifier"])
            foil_forms = _token_forms(foil_text, tokenizer, forbidden_ids=used_token_ids)
            if not foil_forms:
                continue
            used_token_ids.update(form["token_id"] for form in foil_forms)
            foils.append(
                {
                    "id": f"foil-{len(foils):02d}-{foil_key}",
                    "task_instance_id": task["instance_id"],
                    "kind": target_kind,
                    "target": foil_text,
                    "forms": foil_forms,
                    "aliases": [foil_text],
                    "source": {
                        "type": f"generated_patch_{raw_foil['category']}_identifier",
                        "path": raw_foil["path"],
                        "patch_line_number": raw_foil["patch_line_number"],
                        "line_sha256": raw_foil["line_sha256"],
                    },
                }
            )
            if len(foils) == 4:
                break
        if not foils:
            rejected["no_same_kind_foil"] += 1
            continue
        target_id = f"target-{len(targets):02d}-{normalized}"
        target = {
            "id": target_id,
            "kind": target_kind,
            "target": target_text,
            "forms": forms,
            "aliases": [target_text],
            "future_support": {
                "contract": "intersection_of_agent_generated_patch_mutation_completion_and_terminal_summary_v1",
                "benchmark_gold_used": False,
                "lens_output_used": False,
                "generated_patch": {
                    "path": _relative_file(run_root, patch_path, "generated patch"),
                    "sha256": sha256_file(patch_path),
                    "source_path": added["path"],
                    "patch_line_number": added["patch_line_number"],
                    "line_sha256": added["line_sha256"],
                    "span": added["span"],
                },
                "mutation_completion": {
                    key: mutation[key]
                    for key in (
                        "completion_index",
                        "source_request_global_index",
                        "next_request_global_index",
                        "channel",
                        "channel_text_sha256",
                        "span",
                    )
                },
                "terminal_summary": {
                    "runner_metadata_path": _relative_file(
                        run_root, task["metadata_path"], "runner metadata"
                    ),
                    "runner_metadata_sha256": sha256_file(task["metadata_path"]),
                    "field": terminal["field"],
                    "text_sha256": terminal["text_sha256"],
                    "span": terminal["span"],
                },
            },
            "foils": foils,
        }
        validate_target_contract(target, instance_id=str(task["instance_id"]))
        targets.append(target)
    return targets, {
        "status": "derived",
        "generated_patch_path": _relative_file(run_root, patch_path, "generated patch"),
        "generated_patch_sha256": sha256_file(patch_path),
        "terminal_summary_sha256": sha256_text(terminal_summary),
        "added_identifier_count": len(patch_records["added"]),
        "mutation_identifier_count": len(mutation_occurrences),
        "terminal_identifier_count": len(terminal_occurrences),
        "intersection_count": len(intersection),
        "retained_target_count": len(targets),
        "rejections": rejected,
    }


def validate_target_contract(target: Mapping[str, Any], *, instance_id: str) -> None:
    target_id = nonempty_string(target.get("id"), "target ID")
    target_kind = nonempty_string(target.get("kind"), f"{target_id} kind")
    nonempty_string(target.get("target"), f"{target_id} text")
    require(bool(sequence(target.get("forms"), f"{target_id} forms")), "target has no forms")
    support = mapping(target.get("future_support"), f"{target_id} support")
    require(
        support.get("benchmark_gold_used") is False
        and support.get("lens_output_used") is False,
        "future target used forbidden evidence",
    )
    raw_foils = sequence(target.get("foils"), f"{target_id} foils")
    require(bool(raw_foils), "target has no same-task same-kind foil")
    for raw_foil in raw_foils:
        foil = mapping(raw_foil, "target foil")
        require(foil.get("task_instance_id") == instance_id, "cross-task behavioral foil")
        require(foil.get("kind") == target_kind, "behavioral foil kind differs from target")
        require(bool(sequence(foil.get("forms"), "foil forms")), "behavioral foil has no forms")
        source = mapping(foil.get("source"), "foil source")
        require(
            source.get("type") in {
                "generated_patch_removed_identifier",
                "generated_patch_context_identifier",
            },
            "behavioral foil is not from same-task removed/context code",
        )


def identifier_segments(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).strip("_")
    pieces: list[str] = []
    for component in normalized.split("_"):
        ascii_segments = ASCII_SEGMENT_RE.findall(component)
        if ascii_segments and "".join(ascii_segments).casefold() == component.casefold():
            pieces.extend(segment.casefold() for segment in ascii_segments)
        elif component:
            pieces.append(component.casefold())
    return tuple(pieces)


def _identifier_exposure(text: str, aliases: Sequence[str]) -> list[dict[str, Any]]:
    requested = [(alias, identifier_segments(alias)) for alias in aliases if identifier_segments(alias)]
    counts: dict[tuple[str, str, str], int] = {}
    for occurrence in _identifier_occurrences(text):
        identifier = str(occurrence["identifier"])
        identifier_normalized = _normalized_identifier(identifier).strip("_")
        segments = identifier_segments(identifier)
        for alias, alias_segments in requested:
            alias_normalized = _normalized_identifier(alias).strip("_")
            width = len(alias_segments)
            if identifier_normalized == alias_normalized:
                kind = "nfkc_casefold_full_identifier"
            elif any(
                segments[offset : offset + width] == alias_segments
                for offset in range(len(segments) - width + 1)
            ):
                kind = "nfkc_casefold_identifier_segment"
            else:
                continue
            key = (alias, identifier, kind)
            counts[key] = counts.get(key, 0) + 1
    return [
        {"alias": alias, "identifier": identifier, "match_kind": kind, "occurrences": count}
        for (alias, identifier, kind), count in sorted(counts.items())
    ]


def request_channels(request: Mapping[str, Any]) -> dict[str, str]:
    channels = {
        "system": [],
        "user": [],
        "assistant_text": [],
        "tool_arguments": [],
        "tool_results": [],
        "other_request_fields": [],
    }
    messages = sequence(request.get("messages"), "visibility messages")
    for raw_message in messages:
        message = mapping(raw_message, "visibility message")
        role = message.get("role")
        if role in {"system", "user"}:
            channels[str(role)].extend(_flatten_strings(message.get("content")))
        elif role == "assistant":
            channels["assistant_text"].append(_assistant_text(message))
            for raw_call in message.get("tool_calls") or []:
                call = mapping(raw_call, "visibility tool call")
                function = mapping(call.get("function"), "visibility tool function")
                channels["tool_arguments"].append(str(function.get("arguments", "")))
        elif role == "tool":
            channels["tool_results"].extend(_flatten_strings(message.get("content")))
        else:
            channels["other_request_fields"].extend(_flatten_strings(message))
    request_without_messages = {key: value for key, value in request.items() if key != "messages"}
    channels["other_request_fields"].extend(_flatten_strings(request_without_messages))
    return {key: "\n".join(piece for piece in values if piece) for key, values in channels.items()}


def exposure_evidence(
    *, text: str, aliases: Sequence[str], forms: Sequence[Mapping[str, Any]], tokenizer: Any
) -> dict[str, Any]:
    identifier_hits = _identifier_exposure(text, aliases)
    form_ids = {int(form["token_id"]) for form in forms}
    observed_ids = set(tokenizer.encode(text, add_special_tokens=False)) if text else set()
    token_hits = sorted(form_ids & observed_ids)
    return {
        "normalization": "NFKC_then_casefold_with_snake_camel_identifier_segments_v1",
        "aliases": list(aliases),
        "identifier_hits": identifier_hits,
        "scored_form_token_id_hits": token_hits,
        "exposed": bool(identifier_hits or token_hits),
    }


def audit_target_visibility(
    *,
    targets: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
    rendered: str,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    channels = request_channels(request)
    result: list[dict[str, Any]] = []
    for target in targets:
        validate_target_contract(target, instance_id=str(target.get("task_instance_id", ""))) if "task_instance_id" in target else None
        target_channel = {
            name: exposure_evidence(
                text=text,
                aliases=[str(value) for value in sequence(target.get("aliases"), "target aliases")],
                forms=sequence(target.get("forms"), "target forms"),
                tokenizer=tokenizer,
            )
            for name, text in channels.items()
        }
        target_rendered = exposure_evidence(
            text=rendered,
            aliases=[str(value) for value in sequence(target.get("aliases"), "target aliases")],
            forms=sequence(target.get("forms"), "target forms"),
            tokenizer=tokenizer,
        )
        target_exposed = target_rendered["exposed"] or any(
            evidence["exposed"] for evidence in target_channel.values()
        )
        foil_rows: list[dict[str, Any]] = []
        retained: list[str] = []
        excluded: list[dict[str, Any]] = []
        for raw_foil in sequence(target.get("foils"), "target foils"):
            foil = mapping(raw_foil, "target foil")
            aliases = [str(value) for value in sequence(foil.get("aliases"), "foil aliases")]
            forms = sequence(foil.get("forms"), "foil forms")
            channel_evidence = {
                name: exposure_evidence(text=text, aliases=aliases, forms=forms, tokenizer=tokenizer)
                for name, text in channels.items()
            }
            rendered_evidence = exposure_evidence(
                text=rendered, aliases=aliases, forms=forms, tokenizer=tokenizer
            )
            exposed = rendered_evidence["exposed"] or any(
                evidence["exposed"] for evidence in channel_evidence.values()
            )
            foil_id = str(foil["id"])
            if exposed:
                excluded.append({"foil_id": foil_id, "reason": "prompt_exposure"})
            else:
                retained.append(foil_id)
            foil_rows.append(
                {
                    "foil_id": foil_id,
                    "exposed": exposed,
                    "channel_evidence": channel_evidence,
                    "rendered_evidence": rendered_evidence,
                }
            )
        if target_exposed:
            status = "target_exposed"
        elif not retained:
            status = "insufficient_hidden_foils"
        else:
            status = "eligible"
        result.append(
            {
                "target_id": target["id"],
                "target_exposed": target_exposed,
                "retained_hidden_foil_ids": retained,
                "excluded_foils": excluded,
                "status": status,
                "target_channel_evidence": target_channel,
                "target_rendered_evidence": target_rendered,
                "foil_evidence": foil_rows,
            }
        )
    return result


def bind_official_outcomes(
    *,
    run_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    instance_ids: Sequence[str],
    campaign_sha256: str,
    required: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = run_root / "official_score/official_outcomes.json"
    if not path.exists():
        require(not required, "required official outcome aggregate is missing")
        labels = {
            instance_id: {
                "status": "missing",
                "class_id": None,
                "verdict": None,
                "derivation": "official_outcome_aggregate_absent",
                "official_outcomes_path": None,
                "official_outcomes_sha256": None,
                "outcome_record_sha256": None,
            }
            for instance_id in instance_ids
        }
        return labels, {
            "status": "missing",
            "required": required,
            "path": None,
            "sha256": None,
            "outcome_count": 0,
            "generation_skip_evaluations_used": False,
        }
    require(path.is_file() and not path.is_symlink(), "official outcome aggregate is not a regular file")
    value = mapping(json.loads(path.read_bytes()), "official outcome aggregate")
    require(
        value.get("schema_version") == 1
        and value.get("kind") == "swe_verified_behavioral_official_outcomes",
        "official outcome aggregate schema mismatch",
    )
    require(value.get("instance_ids") == list(instance_ids), "official outcome campaign coverage/order mismatch")
    inputs = mapping(value.get("inputs"), "official outcome inputs")
    hashes = mapping(inputs.get("hashes"), "official outcome input hashes")
    require(
        hashes.get("campaign_config_sha256") == campaign_sha256,
        "official outcome campaign hash mismatch",
    )
    rows = [mapping(row, "official outcome row") for row in sequence(value.get("outcomes"), "official outcomes")]
    require(
        [row.get("instance_id") for row in rows] == list(instance_ids),
        "official outcome row coverage/order mismatch",
    )
    task_by_id = {str(task["instance_id"]): task for task in tasks}
    category_counts = {name: 0 for name in ("resolved", "unresolved", "error", "empty")}
    artifact_sha256 = sha256_file(path)
    relative_path = _relative_file(run_root, path, "official outcome aggregate")
    labels: dict[str, dict[str, Any]] = {}
    for row in rows:
        instance_id = nonempty_string(row.get("instance_id"), "official outcome instance ID")
        verdict = row.get("outcome")
        require(verdict in category_counts, f"unsupported official outcome for {instance_id}")
        category_counts[str(verdict)] += 1
        task = task_by_id[instance_id]
        patch_path = task.get("patch_path")
        patch_bytes = patch_path.read_bytes() if isinstance(patch_path, Path) else b""
        require(
            row.get("patch_bytes") == len(patch_bytes)
            and row.get("patch_sha256") == sha256_bytes(patch_bytes),
            f"official outcome patch binding mismatch for {instance_id}",
        )
        if verdict == "resolved":
            status = "available"
            class_id = "success"
            derivation = "bound_official_swe_bench_aggregate"
        elif verdict == "unresolved":
            status = "available"
            class_id = "failure"
            derivation = "bound_official_swe_bench_aggregate"
        else:
            status = "missing"
            class_id = None
            derivation = "official_nonbinary_infrastructure_or_empty_outcome"
        labels[instance_id] = {
            "status": status,
            "class_id": class_id,
            "verdict": verdict,
            "derivation": derivation,
            "official_outcomes_path": relative_path,
            "official_outcomes_sha256": artifact_sha256,
            "outcome_record_sha256": sha256_json(row),
        }
    require(value.get("counts") == category_counts, "official outcome category counts mismatch")
    return labels, {
        "status": "available",
        "required": required,
        "path": relative_path,
        "sha256": artifact_sha256,
        "outcome_count": len(rows),
        "run_id": value.get("run_id"),
        "evidence_id": value.get("evidence_id"),
        "campaign_config_sha256": campaign_sha256,
        "generation_skip_evaluations_used": False,
    }


def _dynamic_score_ids(targets: Sequence[Mapping[str, Any]]) -> list[int]:
    result: list[int] = []
    for target in targets:
        for forms in [target["forms"], *(foil["forms"] for foil in target["foils"])]:
            for form in forms:
                token_id = int(form["token_id"])
                if token_id not in result:
                    result.append(token_id)
    return result


def _prompt_record_payload_sha256(prompt: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(prompt))
    provenance = mapping(mapping(payload.get("metadata"), "prompt metadata").get("provenance"), "prompt provenance")
    provenance.pop("prompt_record_payload_sha256", None)
    return sha256_json(payload)


def build_behavioral_bundle(
    *,
    run_root: Path,
    campaign: Mapping[str, Any],
    campaign_sha256: str,
    action_protocol: Mapping[str, Any],
    action_protocol_sha256: str,
    tokenizer: Any,
    template: str,
    template_sha256: str,
    require_official_outcomes: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    instance_ids = validate_campaign(campaign)
    max_model_len = mapping(campaign.get("generation"), "campaign generation").get(
        "max_model_len"
    )
    require(
        isinstance(max_model_len, int)
        and not isinstance(max_model_len, bool)
        and max_model_len >= 2,
        "campaign max_model_len cannot support replay",
    )
    max_prompt_tokens = max_model_len - 1
    protocol = validate_action_protocol(action_protocol, tokenizer=tokenizer, campaign=campaign)
    task_records = _load_task_metadata(run_root, instance_ids)
    mapped_tasks, global_binding = map_global_captures(
        run_root=run_root,
        campaign=campaign,
        task_records=task_records,
        tokenizer=tokenizer,
        template=template,
    )
    official_labels, official_binding = bind_official_outcomes(
        run_root=run_root,
        tasks=mapped_tasks,
        instance_ids=instance_ids,
        campaign_sha256=campaign_sha256,
        required=require_official_outcomes,
    )
    prompts: list[dict[str, Any]] = []
    task_audits: list[dict[str, Any]] = []
    for task in mapped_tasks:
        completions = [
            derive_completion(task, index, protocol=protocol)
            for index in range(1, int(task["request_count"]) + 1)
        ]
        targets, target_derivation = derive_task_targets(
            task=task,
            completions=completions,
            tokenizer=tokenizer,
            fixed_token_ids=protocol["fixed_token_ids"],
            run_root=run_root,
        )
        for target in targets:
            target["task_instance_id"] = task["instance_id"]
            validate_target_contract(target, instance_id=str(task["instance_id"]))
        probeability = select_probeable_requests(
            task, max_prompt_tokens=max_prompt_tokens
        )
        selected = probeability["selected_request_indices"]
        selected_candidate_ordinals = {
            local_index: ordinal
            for local_index, ordinal in zip(
                selected,
                probeability["selected_candidate_ordinals"],
                strict=True,
            )
        }
        official_outcome = official_labels[str(task["instance_id"])]
        dynamic_ids = _dynamic_score_ids(targets)
        score_ids = list(
            dict.fromkeys(
                dynamic_ids + protocol["action_token_ids"] + protocol["outcome_token_ids"]
            )
        )
        require(len(score_ids) == len(set(score_ids)), "scored token IDs are not unique")
        for checkpoint_ordinal, local_index in enumerate(selected):
            capture = mapping(task["captures"][local_index - 1], "selected capture")
            require(
                len(capture["token_ids"]) <= max_prompt_tokens,
                "selected prompt exceeds the one-token replay ceiling",
            )
            completion = completions[local_index - 1]
            eligibility = audit_target_visibility(
                targets=targets,
                request=mapping(capture["request"], "selected request"),
                rendered=str(capture["rendered"]),
                tokenizer=tokenizer,
            )
            labels = {
                "action": copy.deepcopy(completion["action"]),
                "tool_execution": copy.deepcopy(completion["tool_execution"]),
                "validation": copy.deepcopy(completion["validation"]),
                "official_outcome": copy.deepcopy(official_outcome),
                "terminal": {
                    "finish_reason": completion["finish_reason"],
                    "is_terminal": completion["status"] == "terminal",
                    "is_terminal_completion": completion["status"] == "terminal",
                    "is_episode_endpoint": local_index == int(task["request_count"]),
                    "is_probeable_endpoint": local_index
                    == probeability["probeable_request_indices"][-1],
                },
            }
            metadata_path = task["metadata_path"]
            patch_path = task.get("patch_path")
            prompt = {
                "id": (
                    f"swe-behavioral-{task['selection_index']:02d}-"
                    f"q{checkpoint_ordinal:02d}-r{local_index:04d}-{task['instance_id']}"
                ),
                "text": capture["rendered"],
                "token_ids": capture["token_ids"],
                "score_token_ids": score_ids,
                "metadata": {
                    "schema_version": 1,
                    "kind": PROMPT_KIND,
                    "campaign_sha256": campaign_sha256,
                    "action_protocol_sha256": action_protocol_sha256,
                    "chat_template_sha256": template_sha256,
                    "selection": {
                        "cohort": "uniform_probeable_request_index",
                        "algorithm": "uniform_probeable_request_indices_v1",
                        "max_checkpoints": MAX_CHECKPOINTS,
                        "max_prompt_tokens": max_prompt_tokens,
                        "task_request_index": local_index,
                        "global_request_index": capture["global_index"],
                        "checkpoint_ordinal": checkpoint_ordinal,
                        "checkpoint_count": len(selected),
                        "candidate_ordinal": selected_candidate_ordinals[local_index],
                        "candidate_ordinal_base": 1,
                        "candidate_count": len(
                            probeability["probeable_request_indices"]
                        ),
                        "probeable_request_indices": copy.deepcopy(
                            probeability["probeable_request_indices"]
                        ),
                        "excluded_request_indices": copy.deepcopy(
                            probeability["excluded_request_indices"]
                        ),
                        "primary_for_action_evaluation": True,
                        "independent_of_next_action_label": True,
                    },
                    "task": {
                        "selection_index": task["selection_index"],
                        "instance_id": task["instance_id"],
                        "repo": task["metadata"].get("repo"),
                        "base_commit": task["metadata"].get("base_commit"),
                        "request_count": task["request_count"],
                        "probeable_request_count": len(
                            probeability["probeable_request_indices"]
                        ),
                        "probeable_request_indices": copy.deepcopy(
                            probeability["probeable_request_indices"]
                        ),
                        "excluded_request_indices": copy.deepcopy(
                            probeability["excluded_request_indices"]
                        ),
                    },
                    "labels": labels,
                    "targets": copy.deepcopy(targets),
                    "target_eligibility": eligibility,
                    "analysis_role": (
                        "future_agent_hidden_available"
                        if any(row["status"] == "eligible" for row in eligibility)
                        else "action_outcome_or_exposure_control"
                    ),
                    "provenance": {
                        "mapping_algorithm": global_binding["mapping_algorithm"],
                        "raw_request_path": capture["path"],
                        "raw_request_sha256": capture["sha256"],
                        "usage_path": global_binding["usage_path"],
                        "usage_sha256": global_binding["usage_sha256"],
                        "usage_record_sha256": sha256_json(capture["usage"]),
                        "official_outcomes": {
                            "status": official_binding["status"],
                            "path": official_outcome["official_outcomes_path"],
                            "sha256": official_outcome["official_outcomes_sha256"],
                            "outcome_record_sha256": official_outcome[
                                "outcome_record_sha256"
                            ],
                        },
                        "runner_metadata_path": _relative_file(run_root, metadata_path, "runner metadata"),
                        "runner_metadata_sha256": sha256_file(metadata_path),
                        "qwen_trace_path": (
                            _relative_file(
                                run_root, task["qwen_trace_path"], "qwen trace"
                            )
                            if isinstance(task.get("qwen_trace_path"), Path)
                            else None
                        ),
                        "qwen_trace_sha256": (
                            sha256_file(task["qwen_trace_path"])
                            if isinstance(task.get("qwen_trace_path"), Path)
                            else None
                        ),
                        "generated_patch_path": (
                            _relative_file(run_root, patch_path, "generated patch")
                            if isinstance(patch_path, Path)
                            else None
                        ),
                        "generated_patch_sha256": (
                            sha256_file(patch_path) if isinstance(patch_path, Path) else None
                        ),
                        "rendered_prompt_sha256": sha256_text(capture["rendered"]),
                        "token_ids_sha256": sha256_json(capture["token_ids"]),
                        "prompt_token_count": len(capture["token_ids"]),
                        "normalized_messages_sha256": capture["normalized_messages_sha256"],
                        "normalized_string_tool_call_arguments": capture[
                            "normalized_string_tool_call_arguments"
                        ],
                        "next_completion": {
                            key: copy.deepcopy(value)
                            for key, value in completion.items()
                            if key not in {"assistant_text", "argument_text", "action", "tool_execution", "validation"}
                        },
                    },
                },
            }
            prompt["metadata"]["provenance"]["prompt_record_payload_sha256"] = (
                _prompt_record_payload_sha256(prompt)
            )
            prompts.append(prompt)
        task_audits.append(
            {
                "selection_index": task["selection_index"],
                "instance_id": task["instance_id"],
                "runner_status": task["metadata"].get("status", "completed"),
                "request_count": task["request_count"],
                "global_request_start": task["global_request_start"],
                "global_request_end": task["global_request_end"],
                "selected_request_indices": selected,
                "probeable_request_indices": probeability[
                    "probeable_request_indices"
                ],
                "excluded_request_indices": probeability[
                    "excluded_request_indices"
                ],
                "excluded_requests": probeability["excluded_requests"],
                "max_prompt_tokens": max_prompt_tokens,
                "terminal_trace_serving_evidence": probeability[
                    "terminal_trace_serving_evidence"
                ],
                "selected_checkpoint_count": len(selected),
                "official_outcome": official_outcome,
                "target_derivation": target_derivation,
                "target_ids": [target["id"] for target in targets],
                "action_class_counts_all_requests": {
                    action_id: sum(
                        completion["action"].get("class_id") == action_id
                        for completion in completions
                    )
                    for action_id in ACTION_IDS
                },
                "missing_action_count_all_requests": sum(
                    completion["action"].get("status") == "missing"
                    for completion in completions
                ),
            }
        )
    prompt_ids = [prompt["id"] for prompt in prompts]
    require(len(prompt_ids) == len(set(prompt_ids)), "behavioral prompt IDs repeat")
    summary = {
        "schema_version": 1,
        "kind": SUMMARY_KIND,
        "campaign_sha256": campaign_sha256,
        "action_protocol_sha256": action_protocol_sha256,
        "chat_template_sha256": template_sha256,
        "selection_contract": {
            "cohort": "uniform_probeable_request_index",
            "algorithm": "uniform_probeable_request_indices_v1",
            "max_checkpoints_per_task": MAX_CHECKPOINTS,
            "max_prompt_tokens": max_prompt_tokens,
            "probeability": (
                "served_by_proxy_usage_or_exact_following_extension_or_terminal_trace_"
                "and_canonical_prompt_with_one_token_replay_room"
            ),
            "includes_first_and_last_probeable_request_when_nonempty": True,
            "primary_for_action_evaluation": True,
            "independent_of_next_action_label": True,
        },
        "target_contract": {
            "source": "agent_artifacts_only",
            "intersection": [
                "generated_patch_added_identifiers",
                "captured_mutation_completion_diagnosis_or_arguments",
                "terminal_result_summary",
            ],
            "benchmark_gold_patch_read": False,
            "benchmark_test_patch_read": False,
            "lens_output_read": False,
            "foil_source": "same_task_same_kind_removed_or_context_patch_identifier",
            "visibility": "per_checkpoint_nfkc_casefold_camel_snake_and_token_id",
        },
        "global_capture_binding": global_binding,
        "official_outcome_binding": official_binding,
        "task_count": len(mapped_tasks),
        "task_with_request_count": sum(bool(task["request_count"]) for task in mapped_tasks),
        "prompt_count": len(prompts),
        "dynamic_target_count": sum(len(task["target_ids"]) for task in task_audits),
        "eligible_target_checkpoint_count": sum(
            row["status"] == "eligible"
            for prompt in prompts
            for row in prompt["metadata"]["target_eligibility"]
        ),
        "available_action_label_count": sum(
            prompt["metadata"]["labels"]["action"]["status"] == "available"
            for prompt in prompts
        ),
        "missing_action_label_count": sum(
            prompt["metadata"]["labels"]["action"]["status"] == "missing"
            for prompt in prompts
        ),
        "action_class_counts": {
            action_id: sum(
                prompt["metadata"]["labels"]["action"]["class_id"] == action_id
                for prompt in prompts
            )
            for action_id in ACTION_IDS
        },
        "tool_execution_class_counts": {
            outcome_id: sum(
                prompt["metadata"]["labels"]["tool_execution"]["class_id"] == outcome_id
                for prompt in prompts
            )
            for outcome_id in OUTCOME_IDS
        },
        "validation_class_counts": {
            outcome_id: sum(
                prompt["metadata"]["labels"]["validation"]["class_id"] == outcome_id
                for prompt in prompts
            )
            for outcome_id in OUTCOME_IDS
        },
        "task_audits": task_audits,
        "prompts": [
            {
                "id": prompt["id"],
                "instance_id": prompt["metadata"]["task"]["instance_id"],
                "task_request_index": prompt["metadata"]["selection"]["task_request_index"],
                "global_request_index": prompt["metadata"]["selection"]["global_request_index"],
                "action_class": prompt["metadata"]["labels"]["action"]["class_id"],
                "prompt_sha256": sha256_text(prompt["text"]),
                "token_ids_sha256": sha256_json(prompt["token_ids"]),
                "score_token_ids_sha256": sha256_json(prompt["score_token_ids"]),
                "prompt_record_payload_sha256": prompt["metadata"]["provenance"][
                    "prompt_record_payload_sha256"
                ],
            }
            for prompt in prompts
        ],
    }
    return prompts, summary


def _offset_global_index(container: dict[str, Any], field: str, offset: int) -> None:
    value = container.get(field)
    source_field = f"source_{field}"
    require(source_field not in container, f"{source_field} already exists before combination")
    container[source_field] = value
    if value is not None:
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 1,
            f"{field} is invalid before combination",
        )
        container[field] = value + offset


def combine_behavioral_bundles(
    sources: Sequence[Mapping[str, Any]], *, cohort_manifest_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(len(sources) >= 2, "combination requires at least two source cohorts")
    combined: list[dict[str, Any]] = []
    combined_task_audits: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    campaign_hashes: list[str] = []
    protocol_hash: str | None = None
    template_hash: str | None = None
    global_offset = 0
    task_offset = 0
    dynamic_target_count = 0
    eligible_target_checkpoint_count = 0
    seen_prompt_ids: set[str] = set()
    seen_instance_ids: set[str] = set()
    for cohort_index, raw_source in enumerate(sources):
        source = mapping(raw_source, f"source cohort {cohort_index}")
        cohort_id = nonempty_string(source.get("id"), "cohort ID")
        require(re.fullmatch(r"[a-z][a-z0-9_-]*", cohort_id) is not None, "unsafe cohort ID")
        require(cohort_id not in {row["id"] for row in cohort_rows}, "cohort ID repeats")
        campaign_sha256 = nonempty_string(source.get("campaign_sha256"), "source campaign hash")
        require(re.fullmatch(r"[0-9a-f]{64}", campaign_sha256) is not None, "invalid source campaign hash")
        require(campaign_sha256 not in campaign_hashes, "source campaign hash repeats")
        campaign_hashes.append(campaign_sha256)
        instance_ids = [
            nonempty_string(value, "source instance ID")
            for value in sequence(source.get("instance_ids"), "source instance IDs")
        ]
        require(bool(instance_ids) and len(instance_ids) == len(set(instance_ids)), "source task IDs repeat")
        require(not (set(instance_ids) & seen_instance_ids), "task ID overlaps source cohorts")
        seen_instance_ids.update(instance_ids)
        source_prompts = [mapping(value, "source prompt") for value in sequence(source.get("prompts"), "source prompts")]
        source_summary = mapping(source.get("summary"), "source summary")
        require(source_summary.get("campaign_sha256") == campaign_sha256, "source summary campaign hash mismatch")
        require(source_summary.get("task_count") == len(instance_ids), "source summary task count mismatch")
        require(source_summary.get("prompt_count") == len(source_prompts), "source summary prompt count mismatch")
        source_dynamic_count = source_summary.get("dynamic_target_count")
        source_eligible_count = source_summary.get("eligible_target_checkpoint_count")
        require(
            isinstance(source_dynamic_count, int)
            and not isinstance(source_dynamic_count, bool)
            and source_dynamic_count >= 0
            and isinstance(source_eligible_count, int)
            and not isinstance(source_eligible_count, bool)
            and source_eligible_count >= 0,
            "source target summary counts are invalid",
        )
        dynamic_target_count += source_dynamic_count
        eligible_target_checkpoint_count += source_eligible_count
        task_audits = [mapping(value, "source task audit") for value in sequence(source_summary.get("task_audits"), "source task audits")]
        require(
            [audit.get("instance_id") for audit in task_audits] == instance_ids,
            "source summary task order/coverage mismatch",
        )
        source_global_count = mapping(
            source_summary.get("global_capture_binding"), "source global binding"
        ).get("global_request_count")
        require(
            isinstance(source_global_count, int)
            and not isinstance(source_global_count, bool)
            and source_global_count >= 0,
            "source global request count is invalid",
        )
        current_protocol_hash = nonempty_string(
            source_summary.get("action_protocol_sha256"), "source action protocol hash"
        )
        current_template_hash = nonempty_string(
            source_summary.get("chat_template_sha256"), "source template hash"
        )
        if protocol_hash is None:
            protocol_hash = current_protocol_hash
            template_hash = current_template_hash
        require(
            current_protocol_hash == protocol_hash and current_template_hash == template_hash,
            "source cohorts use different action protocol or template",
        )
        source_summary_sha256 = sha256_json(source_summary)
        source_run_id = f"run-{source_summary_sha256[:20]}"
        run_label = nonempty_string(source.get("run_label"), "source run label")
        cohort_metadata = {
            "index": cohort_index,
            "id": cohort_id,
            "campaign_sha256": campaign_sha256,
            "source_run_id": source_run_id,
            "source_run_label": run_label,
            "source_summary_sha256": source_summary_sha256,
            "source_task_count": len(instance_ids),
            "source_task_instance_ids": instance_ids,
            "source_global_request_count": source_global_count,
            "source_prompt_count": len(source_prompts),
            "global_request_offset": global_offset,
            "task_offset": task_offset,
            "cohort_manifest_sha256": cohort_manifest_sha256,
        }
        for raw_prompt in source_prompts:
            source_prompt = copy.deepcopy(dict(raw_prompt))
            require(
                _prompt_record_payload_sha256(source_prompt)
                == mapping(
                    mapping(source_prompt.get("metadata"), "source prompt metadata").get(
                        "provenance"
                    ),
                    "source prompt provenance",
                ).get("prompt_record_payload_sha256"),
                "source prompt payload hash mismatch",
            )
            metadata = mapping(source_prompt.get("metadata"), "source prompt metadata")
            require(metadata.get("campaign_sha256") == campaign_sha256, "source prompt campaign hash mismatch")
            source_prompt_id = nonempty_string(source_prompt.get("id"), "source prompt ID")
            selection = dict(mapping(metadata.get("selection"), "source prompt selection"))
            _offset_global_index(selection, "global_request_index", global_offset)
            task = dict(mapping(metadata.get("task"), "source prompt task"))
            source_task_index = task.get("selection_index")
            require(
                isinstance(source_task_index, int)
                and not isinstance(source_task_index, bool)
                and 0 <= source_task_index < len(instance_ids),
                "source task selection index is invalid",
            )
            task["source_selection_index"] = source_task_index
            task["selection_index"] = source_task_index + task_offset
            require(task.get("instance_id") == instance_ids[source_task_index], "source prompt task binding mismatch")
            provenance = dict(mapping(metadata.get("provenance"), "source prompt provenance"))
            next_completion = dict(mapping(provenance.get("next_completion"), "source next completion"))
            for field in ("source_request_global_index", "next_request_global_index"):
                if field in next_completion:
                    original_field = f"source_campaign_{field}"
                    require(original_field not in next_completion, "combined next-completion field already exists")
                    next_completion[original_field] = next_completion[field]
                    if next_completion[field] is not None:
                        next_completion[field] += global_offset
            provenance["next_completion"] = next_completion
            provenance["combination"] = {
                "source_prompt_id": source_prompt_id,
                "source_prompt_record_payload_sha256": provenance[
                    "prompt_record_payload_sha256"
                ],
                "source_campaign_global_request_index": selection[
                    "source_global_request_index"
                ],
                "combined_global_request_index": selection["global_request_index"],
                "cohort_manifest_sha256": cohort_manifest_sha256,
            }
            targets = copy.deepcopy(sequence(metadata.get("targets"), "source targets"))
            for target in targets:
                support = mapping(mapping(target, "combined target").get("future_support"), "target support")
                mutation = dict(mapping(support.get("mutation_completion"), "target mutation support"))
                for field in ("source_request_global_index", "next_request_global_index"):
                    if field in mutation:
                        source_field = f"source_campaign_{field}"
                        mutation[source_field] = mutation[field]
                        if mutation[field] is not None:
                            mutation[field] += global_offset
                support["mutation_completion"] = mutation
            metadata["selection"] = selection
            metadata["task"] = task
            metadata["targets"] = targets
            metadata["provenance"] = provenance
            metadata["cohort"] = copy.deepcopy(cohort_metadata)
            source_prompt["metadata"] = metadata
            source_prompt["id"] = f"swe-behavioral-{cohort_id}-{campaign_sha256[:8]}-{source_prompt_id}"
            require(source_prompt["id"] not in seen_prompt_ids, "combined prompt ID repeats")
            seen_prompt_ids.add(source_prompt["id"])
            provenance["prompt_record_payload_sha256"] = _prompt_record_payload_sha256(
                source_prompt
            )
            combined.append(source_prompt)
        for raw_audit in task_audits:
            audit = copy.deepcopy(dict(raw_audit))
            audit["source_selection_index"] = audit["selection_index"]
            audit["selection_index"] += task_offset
            for field in ("global_request_start", "global_request_end"):
                audit[f"source_{field}"] = audit.get(field)
                if audit.get(field) is not None:
                    audit[field] += global_offset
            audit["cohort_id"] = cohort_id
            audit["campaign_sha256"] = campaign_sha256
            combined_task_audits.append(audit)
        cohort_rows.append(copy.deepcopy(cohort_metadata))
        global_offset += source_global_count
        task_offset += len(instance_ids)
    require(protocol_hash is not None and template_hash is not None, "combined hashes are missing")
    summary = {
        "schema_version": 1,
        "kind": "swe_verified_behavioral_probe_combination",
        "cohort_manifest_sha256": cohort_manifest_sha256,
        "source_campaign_sha256s": campaign_hashes,
        "campaign_sha256s": campaign_hashes,
        "action_protocol_sha256": protocol_hash,
        "chat_template_sha256": template_hash,
        "cohorts": cohort_rows,
        "cohort_count": len(cohort_rows),
        "task_count": task_offset,
        "global_request_count": global_offset,
        "prompt_count": len(combined),
        "dynamic_target_count": dynamic_target_count,
        "eligible_target_checkpoint_count": eligible_target_checkpoint_count,
        "task_audits": combined_task_audits,
        "prompts": [
            {
                "id": prompt["id"],
                "cohort_id": mapping(prompt["metadata"], "combined metadata")["cohort"]["id"],
                "instance_id": mapping(prompt["metadata"], "combined metadata")["task"]["instance_id"],
                "global_request_index": mapping(prompt["metadata"], "combined metadata")["selection"]["global_request_index"],
                "prompt_record_payload_sha256": mapping(prompt["metadata"], "combined metadata")["provenance"]["prompt_record_payload_sha256"],
            }
            for prompt in combined
        ],
    }
    return combined, summary


def validate_cohort_manifest(
    manifest: Mapping[str, Any],
    *,
    cohort_pairs: Sequence[tuple[Path, Path]],
    action_protocol_sha256: str,
    template_sha256: str,
) -> list[dict[str, Any]]:
    require(
        manifest.get("schema_version") == 1
        and manifest.get("kind") == "swe_verified_behavioral_n20_cohort",
        "cohort manifest schema mismatch",
    )
    require(
        manifest.get("lens_outputs_used_for_selection") is False
        and manifest.get("official_outcomes_used_for_selection") is False,
        "cohort selection used forbidden post-generation evidence",
    )
    for field, expected_path, expected_hash in (
        ("action_protocol", DEFAULT_ACTION_PROTOCOL, action_protocol_sha256),
        ("chat_template", DEFAULT_TEMPLATE, template_sha256),
    ):
        binding = mapping(manifest.get(field), f"cohort {field}")
        require(
            binding.get("path") == expected_path.relative_to(ROOT).as_posix()
            and binding.get("sha256") == expected_hash,
            f"cohort {field} binding mismatch",
        )
    rows = [mapping(row, "cohort manifest row") for row in sequence(manifest.get("cohorts"), "cohorts")]
    require(len(rows) == len(cohort_pairs) and len(rows) >= 2, "cohort argument count/order mismatch")
    specs: list[dict[str, Any]] = []
    combined_ids: list[str] = []
    for index, (row, pair) in enumerate(zip(rows, cohort_pairs, strict=True)):
        campaign_path, run_root = pair
        logical = nonempty_string(row.get("campaign_path"), "cohort campaign path")
        relative = Path(logical)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe cohort campaign path")
        expected_campaign_path = (ROOT / relative).resolve(strict=True)
        require(campaign_path == expected_campaign_path, "cohort campaign path/order mismatch")
        campaign_sha256 = sha256_file(campaign_path)
        require(row.get("campaign_sha256") == campaign_sha256, "cohort campaign SHA-256 mismatch")
        run_label = nonempty_string(row.get("run_label"), "cohort run label")
        require(run_root.name == run_label, "cohort run label/root mismatch")
        campaign = mapping(json.loads(campaign_path.read_bytes()), "cohort campaign")
        instance_ids = validate_campaign(campaign)
        require(row.get("instance_ids") == instance_ids, "cohort task order/coverage mismatch")
        require(not (set(instance_ids) & set(combined_ids)), "cohort task IDs overlap")
        combined_ids.extend(instance_ids)
        specs.append(
            {
                "index": index,
                "id": nonempty_string(row.get("id"), "cohort ID"),
                "campaign_path": campaign_path,
                "campaign": campaign,
                "campaign_sha256": campaign_sha256,
                "run_root": run_root,
                "run_label": run_label,
                "instance_ids": instance_ids,
            }
        )
    require(manifest.get("instance_ids") == combined_ids, "combined cohort task order changed")
    return specs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument(
        "--cohort",
        nargs=2,
        action="append",
        type=Path,
        metavar=("CAMPAIGN", "RUN_ROOT"),
        help="repeat in frozen manifest order to materialize a combined cohort",
    )
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT_MANIFEST)
    parser.add_argument("--action-protocol", type=Path, default=DEFAULT_ACTION_PROTOCOL)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--require-official-outcomes", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    if args.cohort:
        if args.run_root is not None or args.campaign is not None:
            parser.error("--cohort cannot be combined with --run-root/--campaign")
        if len(args.cohort) < 2:
            parser.error("combined mode requires at least two --cohort pairs")
    elif args.run_root is None:
        parser.error("single-cohort mode requires --run-root")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    protocol_path = args.action_protocol.expanduser().resolve(strict=True)
    template_path = args.template.expanduser().resolve(strict=True)
    protocol_bytes = protocol_path.read_bytes()
    template_bytes = template_path.read_bytes()
    protocol = mapping(json.loads(protocol_bytes), "action protocol")
    if args.cohort:
        cohort_pairs = [
            (
                campaign_path.expanduser().resolve(strict=True),
                run_root.expanduser().resolve(strict=True),
            )
            for campaign_path, run_root in args.cohort
        ]
        cohort_manifest_path = args.cohort_manifest.expanduser().resolve(strict=True)
        cohort_manifest_bytes = cohort_manifest_path.read_bytes()
        specs = validate_cohort_manifest(
            mapping(json.loads(cohort_manifest_bytes), "cohort manifest"),
            cohort_pairs=cohort_pairs,
            action_protocol_sha256=sha256_bytes(protocol_bytes),
            template_sha256=sha256_bytes(template_bytes),
        )
    else:
        campaign_path = (args.campaign or DEFAULT_CAMPAIGN).expanduser().resolve(strict=True)
        campaign_bytes = campaign_path.read_bytes()
        campaign = mapping(json.loads(campaign_bytes), "campaign")
        specs = [
            {
                "id": "single",
                "campaign_path": campaign_path,
                "campaign": campaign,
                "campaign_sha256": sha256_bytes(campaign_bytes),
                "run_root": args.run_root.expanduser().resolve(strict=True),
                "run_label": args.run_root.expanduser().resolve(strict=True).name,
                "instance_ids": validate_campaign(campaign),
            }
        ]
        cohort_manifest_bytes = None
    generation = mapping(specs[0]["campaign"].get("generation"), "campaign generation")
    for spec in specs[1:]:
        require(
            mapping(spec["campaign"].get("generation"), "campaign generation").get(
                "model_repo_id"
            )
            == generation.get("model_repo_id")
            and mapping(spec["campaign"].get("generation"), "campaign generation").get(
                "model_revision"
            )
            == generation.get("model_revision"),
            "cohort model pins differ",
        )
    snapshot = Path(
        args.model_snapshot
        or snapshot_download(
            generation["model_repo_id"],
            revision=generation["model_revision"],
            local_files_only=True,
        )
    ).expanduser().resolve(strict=True)
    tokenizer_pin = mapping(mapping(protocol.get("pins"), "protocol pins").get("tokenizer"), "tokenizer pin")
    require(
        sha256_file(snapshot / "tokenizer.json") == tokenizer_pin.get("json_sha256"),
        "tokenizer.json SHA-256 differs from action protocol",
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    source_bundles: list[dict[str, Any]] = []
    for spec in specs:
        source_prompts, source_summary = build_behavioral_bundle(
            run_root=spec["run_root"],
            campaign=spec["campaign"],
            campaign_sha256=spec["campaign_sha256"],
            action_protocol=protocol,
            action_protocol_sha256=sha256_bytes(protocol_bytes),
            tokenizer=tokenizer,
            template=template_bytes.decode("utf-8"),
            template_sha256=sha256_bytes(template_bytes),
            require_official_outcomes=args.require_official_outcomes,
        )
        source_bundles.append(
            {
                **spec,
                "prompts": source_prompts,
                "summary": source_summary,
            }
        )
    if len(source_bundles) == 1:
        prompts = source_bundles[0]["prompts"]
        summary = source_bundles[0]["summary"]
    else:
        assert cohort_manifest_bytes is not None
        prompts, summary = combine_behavioral_bundles(
            source_bundles,
            cohort_manifest_sha256=sha256_bytes(cohort_manifest_bytes),
        )
    output = args.output.expanduser().resolve()
    summary_path = (
        args.summary or output.with_name(f"{output.stem}_summary.json")
    ).expanduser().resolve()
    atomic_write_json(output, prompts)
    summary["prompt_bundle_sha256"] = sha256_file(output)
    atomic_write_json(summary_path, summary)
    print(f"wrote {output} ({len(prompts)} prompts, sha256={summary['prompt_bundle_sha256']})")
    print(f"wrote {summary_path} (sha256={sha256_file(summary_path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
