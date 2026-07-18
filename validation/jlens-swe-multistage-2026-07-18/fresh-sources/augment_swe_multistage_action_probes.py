#!/usr/bin/env python3
"""Add leakage-independent next-action and outcome controls to SWE probes."""

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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/swe_stage_action_probes.json"
DEFAULT_LIFECYCLE_PROTOCOL = ROOT / "configs/swe_multistage_protocol.json"
DEFAULT_INPUT = ROOT / ".cache/swe_multistage/prompts.json"
DEFAULT_OUTPUT = ROOT / ".cache/swe_multistage/action_prompts.json"
MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
TOKENIZER_JSON_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
TOKENIZER_VOCABULARY_SIZE = 248_077
PROMPT_KIND = "swe_verified_multistage_probe"
ACTION_METADATA_KIND = "swe_verified_stage_action_probe_binding"
PROTOCOL_KIND = "swe_verified_stage_action_probe_protocol"
LIFECYCLE_PROTOCOL_KIND = "swe_verified_multistage_probe_protocol"
ACTION_IDS = ("inspect", "edit", "validate", "finalize")
OUTCOME_IDS = ("success", "failure")
FIXED_LAYER_BAND = tuple(range(24, 48))
READ_SEARCH_COMMAND_RE = re.compile(
    r"(?:^|[;&|()]\s*|\bxargs\s+)(?:find|grep|rg|cat|head|tail|sed|ls)\b"
)
READ_MUTATION_RE = re.compile(
    r"(?:^|[;&|()]\s*)(?:rm|mv|cp|touch|mkdir|rmdir|tee|truncate|patch)\b"
    r"|\bsed\s+-[^\n;|]*i\b|\bgit\s+(?:checkout|reset|clean|apply)\b"
)
GENERIC_TOOL_FAILURE_RE = re.compile(
    r"\b(?:permission (?:was )?declined|permission denied|failed|error|exception)\b",
    re.IGNORECASE,
)
NON_REPOSITORY_TARGET_PREFIXES = (
    "/dev/",
    "/tmp/",
    "/var/tmp/",
    "$HOME/",
    "${HOME}/",
    "~/",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return value


def nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be a nonempty string")
    return value


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
        ).encode("ascii")
    )


def materialized_json_sha256(value: Any) -> str:
    return sha256_bytes(
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
            "ascii"
        )
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


def _flatten_text(value: Any, label: str) -> str:
    if isinstance(value, str):
        return value
    result: list[str] = []
    for index, raw_part in enumerate(sequence(value, label)):
        part = mapping(raw_part, f"{label}[{index}]")
        require(part.get("type") == "text", f"{label}[{index}] is not text")
        result.append(nonempty_string(part.get("text"), f"{label}[{index}].text"))
    return "".join(result)


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for part in value for item in _flatten_strings(part)]
    if isinstance(value, dict):
        return [item for part in value.values() for item in _flatten_strings(part)]
    return []


def _assistant_text(assistant: Mapping[str, Any]) -> str:
    reasoning = assistant.get("reasoning_content")
    duplicate = assistant.get("reasoning")
    if reasoning is None and duplicate is None:
        reasoning = ""
    else:
        require(
            isinstance(reasoning, str) and duplicate == reasoning,
            "assistant reasoning fields are inconsistent",
        )
    content = assistant.get("content")
    require(isinstance(content, str), "assistant content must be text")
    return f"{reasoning}\n{content}"


def _parse_shell_result(text: str) -> dict[str, Any]:
    match = re.fullmatch(
        r"Command: (?P<command>.*?)\nDirectory: (?P<directory>.*?)\nOutput: "
        r"(?P<output>.*?)\nError: (?P<error>.*?)\nExit Code: (?P<exit>-?\d+)\n"
        r"Signal: (?P<signal>-?\d+)\nProcess Group PGID: (?P<pgid>-?\d+)",
        text,
        flags=re.DOTALL,
    )
    require(match is not None, "shell result does not match the captured Qwen envelope")
    assert match is not None
    return {
        "command": match.group("command"),
        "output": match.group("output"),
        "error": match.group("error"),
        "exit_code": int(match.group("exit")),
        "signal": int(match.group("signal")),
    }


def _compile_regexes(values: Any, label: str) -> list[re.Pattern[str]]:
    result: list[re.Pattern[str]] = []
    for index, raw_value in enumerate(sequence(values, label)):
        value = nonempty_string(raw_value, f"{label}[{index}]")
        try:
            result.append(re.compile(value))
        except re.error as error:
            raise ValueError(f"{label}[{index}] is invalid: {error}") from error
    require(bool(result), f"{label} must not be empty")
    return result


def validate_lifecycle_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    require(protocol.get("schema_version") == 1, "lifecycle protocol schema mismatch")
    require(
        protocol.get("kind") == LIFECYCLE_PROTOCOL_KIND,
        "lifecycle protocol kind mismatch",
    )
    pins = mapping(protocol.get("pins"), "lifecycle protocol pins")
    model = mapping(pins.get("model"), "lifecycle model pin")
    require(
        model.get("repo_id") == MODEL_REPO
        and model.get("revision") == MODEL_REVISION
        and model.get("tokenizer_json_sha256") == TOKENIZER_JSON_SHA256,
        "lifecycle model/tokenizer pin mismatch",
    )
    events = mapping(protocol.get("event_contract"), "lifecycle event contract")
    validation = mapping(
        events.get("successful_validation"), "successful-validation contract"
    )
    require(
        validation.get("requires_successful_tool_result") is True,
        "validation must require a successful tool result",
    )
    return {
        "validation_positive_regexes": _compile_regexes(
            validation.get("positive_output_regexes"),
            "validation positive-output regexes",
        ),
        "validation_negative_regexes": _compile_regexes(
            validation.get("negative_output_regexes"),
            "validation negative-output regexes",
        ),
    }


def _validate_classes(
    value: Any,
    *,
    label: str,
    expected_ids: Sequence[str],
    tokenizer: Any,
    global_ids: set[int],
) -> list[dict[str, Any]]:
    classes = sequence(value, label)
    require(
        [mapping(item, label).get("id") for item in classes] == list(expected_ids),
        f"{label} IDs/order changed",
    )
    result: list[dict[str, Any]] = []
    sizes: set[int] = set()
    for raw_class in classes:
        class_record = mapping(raw_class, label)
        class_id = str(class_record["id"])
        raw_tokens = sequence(class_record.get("tokens"), f"{label}.{class_id}.tokens")
        require(bool(raw_tokens), f"{label}.{class_id}.tokens must not be empty")
        tokens: list[dict[str, Any]] = []
        for index, raw_token in enumerate(raw_tokens):
            token = mapping(raw_token, f"{label}.{class_id}.tokens[{index}]")
            text = nonempty_string(token.get("text"), "class token text")
            token_id = token.get("token_id")
            require(
                text.startswith(" ") and not text.startswith("  "),
                f"{label}.{class_id} token {text!r} is not a leading-space form",
            )
            require(
                isinstance(token_id, int)
                and not isinstance(token_id, bool)
                and 0 <= token_id < TOKENIZER_VOCABULARY_SIZE,
                f"{label}.{class_id} token ID is invalid",
            )
            require(token_id not in global_ids, "action/outcome token IDs must be disjoint")
            encoded = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            require(encoded == [token_id], f"{text!r} is not the pinned single token")
            require(decoded == text, f"token {token_id} does not decode exactly as {text!r}")
            global_ids.add(token_id)
            tokens.append({"text": text, "token_id": token_id})
        sizes.add(len(tokens))
        result.append({"id": class_id, "tokens": tokens})
    require(len(sizes) == 1, f"{label} classes must have equal vocabulary size")
    return result


def validate_protocol(
    protocol: Mapping[str, Any],
    tokenizer: Any,
    *,
    lifecycle_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    lifecycle = validate_lifecycle_protocol(lifecycle_protocol)
    require(protocol.get("schema_version") == 1, "action protocol schema mismatch")
    require(protocol.get("kind") == PROTOCOL_KIND, "action protocol kind mismatch")
    require(
        protocol.get("lens_outputs_used_for_labels") is False,
        "action labels must not use lens outputs",
    )
    pins = mapping(protocol.get("pins"), "protocol pins")
    model = mapping(pins.get("model"), "model pin")
    tokenizer_pin = mapping(pins.get("tokenizer"), "tokenizer pin")
    require(
        model.get("repo_id") == MODEL_REPO and model.get("revision") == MODEL_REVISION,
        "model pin mismatch",
    )
    require(
        tokenizer_pin.get("json_sha256") == TOKENIZER_JSON_SHA256
        and tokenizer_pin.get("vocabulary_size") == TOKENIZER_VOCABULARY_SIZE,
        "tokenizer pin mismatch",
    )
    require(len(tokenizer) == TOKENIZER_VOCABULARY_SIZE, "tokenizer vocabulary size mismatch")
    global_ids: set[int] = set()
    actions = _validate_classes(
        protocol.get("action_classes"),
        label="action classes",
        expected_ids=ACTION_IDS,
        tokenizer=tokenizer,
        global_ids=global_ids,
    )
    outcomes = _validate_classes(
        protocol.get("outcome_classes"),
        label="outcome classes",
        expected_ids=OUTCOME_IDS,
        tokenizer=tokenizer,
        global_ids=global_ids,
    )
    band = mapping(protocol.get("fixed_layer_band"), "fixed layer band")
    require(
        band.get("start") == 24
        and band.get("end") == 47
        and band.get("end_inclusive") is True
        and band.get("layers") == list(FIXED_LAYER_BAND),
        "fixed layer band must be inclusive layers 24 through 47",
    )
    require(
        protocol.get("class_score_reduction") == "logmeanexp_over_class_tokens",
        "class score reduction changed",
    )
    classifier = mapping(
        protocol.get("next_completion_classifier"), "next completion classifier"
    )
    require(
        classifier.get("action_precedence")
        == [
            "terminal_no_tool_response",
            "mutating_source_command",
            "test_command",
            "read_or_search_command",
            "validation_intent_assistant_text",
        ],
        "next-action precedence changed",
    )
    require(
        classifier.get("unclassified_policy") == "missing_not_imputed",
        "unclassified-transition policy changed",
    )
    reasoning_state = mapping(
        protocol.get("reasoning_state_contract"), "reasoning-state contract"
    )
    require(
        reasoning_state
        == {
            "representation": "independent_binary_multilabel_metadata",
            "labels": ["diagnosis_expressed"],
            "lens_scored_vocabulary": False,
            "included_in_action_accuracy": False,
        },
        "reasoning-state contract changed",
    )
    outcome = mapping(protocol.get("outcome_control_contract"), "outcome contract")
    verdict_mapping = mapping(outcome.get("official_verdict_mapping"), "verdict mapping")
    require(
        verdict_mapping
        == {
            "error": "failure",
            "incomplete": "failure",
            "resolved": "success",
            "unresolved": "failure",
        },
        "official verdict mapping changed",
    )
    return {
        "action_classes": actions,
        "outcome_classes": outcomes,
        "action_token_ids": [
            token["token_id"] for record in actions for token in record["tokens"]
        ],
        "outcome_token_ids": [
            token["token_id"] for record in outcomes for token in record["tokens"]
        ],
        "diagnosis_regexes": _compile_regexes(
            classifier.get("diagnosis_assistant_text_regexes"), "diagnosis regexes"
        ),
        "mutation_regexes": _compile_regexes(
            classifier.get("mutating_command_regexes"), "mutation regexes"
        ),
        "test_regexes": _compile_regexes(
            classifier.get("test_command_regexes"), "test regexes"
        ),
        "validation_intent_regexes": _compile_regexes(
            classifier.get("validation_assistant_text_regexes"),
            "validation-intent regexes",
        ),
        "inspection_tools": set(
            sequence(classifier.get("generic_inspection_tool_names"), "inspection tools")
        ),
        "mutation_tools": set(
            sequence(classifier.get("generic_mutation_tool_names"), "mutation tools")
        ),
        "verdict_mapping": dict(verdict_mapping),
        "reasoning_state_contract": dict(reasoning_state),
        **lifecycle,
    }


def _safe_artifact_path(artifact_root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe artifact path")
    path = (artifact_root / relative).resolve(strict=True)
    require(path.is_relative_to(artifact_root), "artifact path escapes its root")
    return path


def _request_messages(request: Mapping[str, Any], label: str) -> list[Mapping[str, Any]]:
    return [mapping(item, label) for item in sequence(request.get("messages"), label)]


def _tool_success(
    name: str,
    result_text: str,
    command: str | None,
    *,
    test_command: bool,
    validation_positive_regexes: Sequence[re.Pattern[str]],
    validation_negative_regexes: Sequence[re.Pattern[str]],
) -> tuple[bool, dict[str, Any]]:
    if name == "run_shell_command":
        require(command is not None, "shell tool call has no command")
        result = _parse_shell_result(result_text)
        require(result["command"] == command, "shell result command mismatch")
        envelope_success = (
            result["exit_code"] == 0
            and result["signal"] == 0
            and result["error"] == "(none)"
        )
        validation_text = str(result["output"])
        evidence: dict[str, Any] = {
            "exit_code": result["exit_code"],
            "signal": result["signal"],
        }
    else:
        envelope_success = GENERIC_TOOL_FAILURE_RE.search(result_text) is None
        validation_text = result_text
        evidence = {"generic_failure_pattern_hit": not envelope_success}
    positive_hits = sorted(
        {
            pattern.pattern
            for pattern in validation_positive_regexes
            if test_command and pattern.search(validation_text)
        }
    )
    negative_hits = sorted(
        {
            pattern.pattern
            for pattern in validation_negative_regexes
            if test_command and pattern.search(validation_text)
        }
    )
    evidence.update(
        {
            "validation_positive_output_regex_hits": positive_hits,
            "validation_negative_output_regex_hits": negative_hits,
            "validation_evidence_required": test_command,
        }
    )
    if test_command:
        return envelope_success and bool(positive_hits) and not negative_hits, evidence
    return envelope_success, evidence


def _is_repository_mutation_target(
    raw_target: str, *, source_paths: Sequence[str]
) -> bool:
    target = raw_target.strip().strip("'\"").rstrip(",)")
    if not target or target == "-" or target.startswith(NON_REPOSITORY_TARGET_PREFIXES):
        return False
    if target.startswith("$") or target.startswith("~"):
        return False
    path = Path(target)
    if path.is_absolute():
        return any(
            target == source_path or target.endswith(f"/{source_path}")
            for source_path in source_paths
        )
    return ".." not in path.parts


def _source_mutation(
    command_or_arguments: str,
    *,
    source_paths: Sequence[str],
    mutation_regexes: Sequence[re.Pattern[str]],
) -> tuple[bool, list[str]]:
    hits = [
        pattern.pattern
        for pattern in mutation_regexes
        if pattern.search(command_or_arguments)
    ]
    if not hits:
        return False, hits
    target_patterns = (
        re.compile(r">{1,2}\s*['\"]?([^\s'\";&|]+)"),
        re.compile(r"\bopen\s*\(\s*['\"]([^'\"]+)['\"]"),
        re.compile(r"\bPath\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
        re.compile(r"\btee(?:\s+-[A-Za-z]+)*\s+['\"]?([^\s'\";&|]+)"),
    )
    targets = [
        match.group(1)
        for pattern in target_patterns
        for match in pattern.finditer(command_or_arguments)
    ]
    if any(
        _is_repository_mutation_target(target, source_paths=source_paths)
        for target in targets
    ):
        return True, hits
    if re.search(
        r"(?im)\bsed\b[^\n;|]*\s-i(?:[.A-Za-z0-9_-]*)?\b",
        command_or_arguments,
    ) and any(path in command_or_arguments for path in source_paths):
        return True, hits
    if re.search(
        r"(?im)\b(?:git\s+apply|patch\s+-p\d+|apply_patch)\b",
        command_or_arguments,
    ):
        return True, hits
    return False, hits


def _is_read_search(command: str) -> bool:
    return READ_SEARCH_COMMAND_RE.search(command) is not None and READ_MUTATION_RE.search(
        command
    ) is None


def derive_next_completion(
    prompt: Mapping[str, Any],
    *,
    artifact_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = mapping(prompt.get("metadata"), "prompt metadata")
    provenance = mapping(metadata.get("provenance"), "prompt provenance")
    task = mapping(metadata.get("task"), "prompt task")
    request_index = provenance.get("raw_request_index")
    require(
        isinstance(request_index, int)
        and not isinstance(request_index, bool)
        and request_index >= 1,
        "raw request index is invalid",
    )
    raw_request_path = nonempty_string(
        provenance.get("raw_request_path"), "raw request path"
    )
    current_path = _safe_artifact_path(artifact_root, raw_request_path)
    require(
        sha256_file(current_path) == provenance.get("raw_request_sha256"),
        "current raw request hash mismatch",
    )
    current_request = mapping(json.loads(current_path.read_bytes()), "current request")
    current_messages = _request_messages(current_request, "current request messages")
    usage = mapping(provenance.get("usage"), "bound usage record")
    require(usage.get("idx") == request_index, "bound usage index mismatch")
    finish_reason = usage.get("finish_reason")
    require(finish_reason in {"tool_calls", "stop", "length"}, "unsupported finish reason")
    declared_next = mapping(
        provenance.get("next_completion_transition"),
        "lifecycle next-completion transition",
    )
    require(
        declared_next.get("contract")
        == "completion_N_is_materialized_by_chat_N_plus_1_or_bound_by_terminal_usage"
        and declared_next.get("completion_index") == request_index
        and declared_next.get("used_only_for_declared_stage_selection_or_action_analysis")
        is True,
        "lifecycle next-completion binding mismatch",
    )
    official = mapping(provenance.get("official_verdict"), "official verdict")
    verdict = official.get("verdict")
    official_outcome = protocol["verdict_mapping"].get(verdict)
    require(official_outcome in OUTCOME_IDS, "official verdict cannot be mapped")
    base = {
        "completion_index": request_index,
        "current_request_path": raw_request_path,
        "current_request_sha256": provenance["raw_request_sha256"],
        "finish_reason": finish_reason,
        "lifecycle_next_completion_binding_sha256": sha256_json(declared_next),
        "lens_outputs_used_for_label": False,
        "official_task_outcome_class": official_outcome,
        "official_verdict": verdict,
    }
    if finish_reason == "stop":
        require(
            declared_next.get("terminal_response") is True
            and declared_next.get("materialized_in_request_index") is None
            and declared_next.get("materialized_in_raw_request_sha256") is None
            and declared_next.get("usage_finish_reason") == "stop"
            and declared_next.get("usage_record_sha256")
            == provenance.get("usage_record_sha256")
            and "finalize" in sequence(
                declared_next.get("event_labels"), "terminal event labels"
            ),
            "terminal lifecycle next-completion binding mismatch",
        )
        return {
            **base,
            "label_status": "available",
            "expected_action_class": "finalize",
            "transition_outcome_class": official_outcome,
            "derivation": "terminal_no_tool_response_from_bound_usage",
            "next_request_path": None,
            "next_request_sha256": None,
            "diagnosis_expressed": False,
            "diagnosis_regex_hits": [],
            "tool_audits": [],
        }
    if finish_reason == "length":
        require(
            declared_next.get("terminal_response") is True,
            "length termination is not terminal in lifecycle metadata",
        )
        return {
            **base,
            "label_status": "missing",
            "expected_action_class": None,
            "transition_outcome_class": "failure",
            "derivation": "length_termination_has_no_imputed_action",
            "next_request_path": None,
            "next_request_sha256": None,
            "diagnosis_expressed": False,
            "diagnosis_regex_hits": [],
            "tool_audits": [],
        }

    filename_match = re.fullmatch(r"chat_(\d{4})\.json", current_path.name)
    require(filename_match is not None, "raw request filename is not a numbered chat capture")
    require(int(filename_match.group(1)) == request_index, "request filename/index mismatch")
    next_path = current_path.with_name(f"chat_{request_index + 1:04d}.json")
    require(next_path.is_file(), "tool-calling completion has no following captured request")
    next_sha256 = sha256_file(next_path)
    require(
        declared_next.get("terminal_response") is False
        and declared_next.get("materialized_in_request_index") == request_index + 1
        and declared_next.get("materialized_in_raw_request_sha256") == next_sha256,
        "captured request disagrees with lifecycle next-completion binding",
    )
    declared_transition = mapping(
        declared_next.get("transition"), "declared next-completion transition"
    )
    require(
        declared_next.get("transition_sha256") == sha256_json(declared_transition),
        "declared next-completion transition hash mismatch",
    )
    next_request = mapping(json.loads(next_path.read_bytes()), "next request")
    next_messages = _request_messages(next_request, "next request messages")
    require(
        next_messages[: len(current_messages)] == current_messages,
        "next request is not an exact raw-message extension",
    )
    extension = next_messages[len(current_messages) :]
    require(bool(extension), "next request appends no completion")
    assistant = mapping(extension[0], "appended assistant message")
    require(assistant.get("role") == "assistant", "extension does not begin with assistant")
    tool_messages = [mapping(item, "appended tool message") for item in extension[1:]]
    require(
        all(message.get("role") == "tool" for message in tool_messages),
        "extension contains a non-tool message after the assistant",
    )
    calls = sequence(assistant.get("tool_calls"), "assistant tool calls")
    require(bool(calls) and len(calls) == len(tool_messages), "tool calls/results mismatch")
    assistant_text = _assistant_text(assistant)
    diagnosis_hits = [
        pattern.pattern
        for pattern in protocol["diagnosis_regexes"]
        if pattern.search(assistant_text)
    ]
    validation_intent_hits = [
        pattern.pattern
        for pattern in protocol["validation_intent_regexes"]
        if pattern.search(assistant_text)
    ]
    mutation_hits: list[str] = []
    test_hits: list[str] = []
    inspect_hits: list[str] = []
    tool_audits: list[dict[str, Any]] = []
    source_paths = sorted(
        {
            nonempty_string(mapping(concept, "concept").get("path"), "concept path")
            for concept in sequence(metadata.get("concepts"), "prompt concepts")
        }
    )
    successes: list[bool] = []
    for call_index, (raw_call, tool_message) in enumerate(
        zip(calls, tool_messages, strict=True)
    ):
        call = mapping(raw_call, f"tool call {call_index}")
        function = mapping(call.get("function"), f"tool call {call_index} function")
        require(call.get("type") == "function", "tool-call type changed")
        name = nonempty_string(function.get("name"), "tool name")
        require(tool_message.get("tool_call_id") == call.get("id"), "tool result ID mismatch")
        raw_arguments = nonempty_string(function.get("arguments"), "raw tool arguments")
        try:
            arguments = mapping(json.loads(raw_arguments), "tool arguments")
        except json.JSONDecodeError as error:
            raise ValueError("tool-call arguments are invalid JSON") from error
        argument_text = "\n".join(_flatten_strings(arguments))
        result_text = _flatten_text(tool_message.get("content"), "tool result")
        command = arguments.get("command") if name == "run_shell_command" else None
        if command is not None:
            command = nonempty_string(command, "shell command")
        source_mutation, matched_mutations = _source_mutation(
            argument_text,
            source_paths=source_paths,
            mutation_regexes=protocol["mutation_regexes"],
        )
        if name in protocol["mutation_tools"]:
            generic_path = next(
                (
                    arguments.get(field)
                    for field in ("path", "file_path", "filename")
                    if isinstance(arguments.get(field), str)
                ),
                None,
            )
            if generic_path is not None and _is_repository_mutation_target(
                generic_path, source_paths=source_paths
            ):
                source_mutation = True
                matched_mutations.append(f"tool_name:{name}")
            elif name == "apply_patch" and any(
                path in argument_text for path in source_paths
            ):
                source_mutation = True
                matched_mutations.append(f"tool_name:{name}")
        matched_tests = [
            pattern.pattern for pattern in protocol["test_regexes"] if pattern.search(argument_text)
        ]
        inspection = name in protocol["inspection_tools"] or (
            command is not None and not source_mutation and _is_read_search(command)
        )
        if source_mutation:
            mutation_hits.extend(matched_mutations)
        if matched_tests:
            test_hits.extend(matched_tests)
        if inspection:
            inspect_hits.append(f"tool_name:{name}")
        success, outcome_evidence = _tool_success(
            name,
            result_text,
            command,
            test_command=bool(matched_tests),
            validation_positive_regexes=protocol["validation_positive_regexes"],
            validation_negative_regexes=protocol["validation_negative_regexes"],
        )
        successes.append(success)
        tool_audits.append(
            {
                "call_index": call_index,
                "tool_call_id": call.get("id"),
                "tool_name": name,
                "arguments_sha256": sha256_text(raw_arguments),
                "result_sha256": sha256_text(result_text),
                "source_mutation": source_mutation,
                "test_command": bool(matched_tests),
                "read_or_search": inspection,
                "successful": success,
                **outcome_evidence,
            }
        )
    if mutation_hits:
        expected_action = "edit"
        derivation = "mutating_source_command"
    elif test_hits:
        expected_action = "validate"
        derivation = "test_command"
    elif inspect_hits:
        expected_action = "inspect"
        derivation = "read_or_search_command"
    elif validation_intent_hits:
        expected_action = "validate"
        derivation = "validation_intent_assistant_text"
    else:
        expected_action = None
        derivation = "unclassified_missing_not_imputed"
    return {
        **base,
        "label_status": "available" if expected_action else "missing",
        "expected_action_class": expected_action,
        "transition_outcome_class": "success" if all(successes) else "failure",
        "derivation": derivation,
        "next_request_path": str(next_path.relative_to(artifact_root)),
        "next_request_sha256": next_sha256,
        "assistant_text_sha256": sha256_text(assistant_text),
        "diagnosis_expressed": bool(diagnosis_hits),
        "diagnosis_regex_hits": sorted(set(diagnosis_hits)),
        "validation_intent_regex_hits": sorted(set(validation_intent_hits)),
        "mutation_rule_hits": sorted(set(mutation_hits)),
        "test_rule_hits": sorted(set(test_hits)),
        "inspection_rule_hits": sorted(set(inspect_hits)),
        "tool_audits": tool_audits,
    }


def build_action_bundle(
    prompts_value: Any,
    *,
    source_bundle_sha256: str,
    action_protocol: Mapping[str, Any],
    action_protocol_sha256: str,
    lifecycle_protocol: Mapping[str, Any],
    lifecycle_protocol_sha256: str,
    tokenizer: Any,
    artifact_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = validate_protocol(
        action_protocol,
        tokenizer,
        lifecycle_protocol=lifecycle_protocol,
    )
    prompts = sequence(prompts_value, "multistage prompt bundle")
    require(bool(prompts), "multistage prompt bundle must not be empty")
    require(
        materialized_json_sha256(prompts) == source_bundle_sha256,
        "source prompt bundle hash does not match canonical materialization",
    )
    result: list[dict[str, Any]] = []
    prompt_ids: set[str] = set()
    for index, raw_prompt in enumerate(prompts):
        prompt = mapping(raw_prompt, f"prompt[{index}]")
        prompt_id = nonempty_string(prompt.get("id"), f"prompt[{index}].id")
        require(prompt_id not in prompt_ids, f"duplicate prompt ID: {prompt_id}")
        prompt_ids.add(prompt_id)
        text = nonempty_string(prompt.get("text"), f"prompt {prompt_id}.text")
        token_ids = sequence(prompt.get("token_ids"), f"prompt {prompt_id}.token_ids")
        require(
            bool(token_ids)
            and all(isinstance(value, int) and not isinstance(value, bool) for value in token_ids),
            f"prompt {prompt_id} has invalid token IDs",
        )
        original_score_ids = sequence(
            prompt.get("score_token_ids"), f"prompt {prompt_id}.score_token_ids"
        )
        require(
            bool(original_score_ids)
            and len(original_score_ids) == len(set(original_score_ids))
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in original_score_ids
            ),
            f"prompt {prompt_id} has invalid scored token IDs",
        )
        metadata = mapping(prompt.get("metadata"), f"prompt {prompt_id}.metadata")
        require(metadata.get("kind") == PROMPT_KIND, f"prompt {prompt_id} kind mismatch")
        require("stage_action_probe" not in metadata, "input prompt is already action-augmented")
        provenance = mapping(metadata.get("provenance"), "prompt provenance")
        require(
            provenance.get("rendered_prompt_sha256") == sha256_text(text)
            and provenance.get("token_ids_sha256") == sha256_json(token_ids)
            and provenance.get("prompt_token_count") == len(token_ids),
            f"prompt {prompt_id} provenance does not bind exact prompt text/token IDs",
        )
        next_completion = derive_next_completion(
            prompt, artifact_root=artifact_root, protocol=protocol
        )
        augmented = copy.deepcopy(dict(prompt))
        action_ids = protocol["action_token_ids"] + protocol["outcome_token_ids"]
        augmented["score_token_ids"] = list(dict.fromkeys(original_score_ids + action_ids))
        augmented_metadata = copy.deepcopy(dict(metadata))
        augmented_metadata["stage_action_probe"] = {
            "schema_version": 1,
            "kind": ACTION_METADATA_KIND,
            "action_protocol_sha256": action_protocol_sha256,
            "lifecycle_protocol_sha256": lifecycle_protocol_sha256,
            "source_prompt_bundle_sha256": source_bundle_sha256,
            "source_prompt_record_sha256": sha256_json(prompt),
            "source_score_token_ids": list(original_score_ids),
            "source_score_token_ids_sha256": sha256_json(original_score_ids),
            "augmented_score_token_ids_sha256": sha256_json(
                augmented["score_token_ids"]
            ),
            "exact_prompt_text_preserved": True,
            "exact_prompt_token_ids_preserved": True,
            "fixed_layer_band": list(FIXED_LAYER_BAND),
            "reasoning_state_contract": copy.deepcopy(
                protocol["reasoning_state_contract"]
            ),
            "scored_vocabulary": {
                "action_classes": copy.deepcopy(protocol["action_classes"]),
                "outcome_classes": copy.deepcopy(protocol["outcome_classes"]),
            },
            "next_completion": next_completion,
        }
        augmented["metadata"] = augmented_metadata
        require(augmented["text"] == text, "augmentation changed exact prompt text")
        require(augmented["token_ids"] == token_ids, "augmentation changed exact prompt token IDs")
        result.append(augmented)
    summary = {
        "schema_version": 1,
        "kind": "swe_verified_stage_action_probe_materialization",
        "source_prompt_bundle_sha256": source_bundle_sha256,
        "action_protocol_sha256": action_protocol_sha256,
        "lifecycle_protocol_sha256": lifecycle_protocol_sha256,
        "lens_outputs_used_for_labels": False,
        "fixed_layer_band": list(FIXED_LAYER_BAND),
        "prompt_count": len(result),
        "task_count": len(
            {prompt["metadata"]["task"]["instance_id"] for prompt in result}
        ),
        "available_action_label_count": sum(
            prompt["metadata"]["stage_action_probe"]["next_completion"]["label_status"]
            == "available"
            for prompt in result
        ),
        "missing_action_label_count": sum(
            prompt["metadata"]["stage_action_probe"]["next_completion"]["label_status"]
            == "missing"
            for prompt in result
        ),
        "diagnosis_expressed_count": sum(
            prompt["metadata"]["stage_action_probe"]["next_completion"][
                "diagnosis_expressed"
            ]
            for prompt in result
        ),
        "action_class_counts": {
            class_id: sum(
                prompt["metadata"]["stage_action_probe"]["next_completion"][
                    "expected_action_class"
                ]
                == class_id
                for prompt in result
            )
            for class_id in ACTION_IDS
        },
        "prompts": [
            {
                "id": prompt["id"],
                "instance_id": prompt["metadata"]["task"]["instance_id"],
                "stage_id": prompt["metadata"]["stage"]["id"],
                "analysis_role": prompt["metadata"]["analysis_role"],
                "expected_action_class": prompt["metadata"]["stage_action_probe"][
                    "next_completion"
                ]["expected_action_class"],
                "diagnosis_expressed": prompt["metadata"]["stage_action_probe"][
                    "next_completion"
                ]["diagnosis_expressed"],
                "transition_outcome_class": prompt["metadata"]["stage_action_probe"][
                    "next_completion"
                ]["transition_outcome_class"],
                "official_task_outcome_class": prompt["metadata"]["stage_action_probe"][
                    "next_completion"
                ]["official_task_outcome_class"],
                "prompt_sha256": sha256_text(prompt["text"]),
                "token_ids_sha256": sha256_json(prompt["token_ids"]),
                "score_token_ids_sha256": sha256_json(prompt["score_token_ids"]),
            }
            for prompt in result
        ],
    }
    return result, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--action-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--lifecycle-protocol", type=Path, default=DEFAULT_LIFECYCLE_PROTOCOL
    )
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    input_path = args.input.expanduser().resolve(strict=True)
    protocol_path = args.action_protocol.expanduser().resolve(strict=True)
    lifecycle_protocol_path = args.lifecycle_protocol.expanduser().resolve(strict=True)
    artifact_root = args.artifact_root.expanduser().resolve(strict=True)
    snapshot = Path(
        args.model_snapshot
        or snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)
    ).expanduser().resolve(strict=True)
    require(snapshot.name == MODEL_REVISION, "model snapshot revision mismatch")
    require(
        sha256_file(snapshot / "tokenizer.json") == TOKENIZER_JSON_SHA256,
        "tokenizer.json hash mismatch",
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    input_bytes = input_path.read_bytes()
    protocol_bytes = protocol_path.read_bytes()
    lifecycle_protocol_bytes = lifecycle_protocol_path.read_bytes()
    prompts = json.loads(input_bytes)
    action_protocol = mapping(json.loads(protocol_bytes), "action protocol")
    lifecycle_protocol = mapping(
        json.loads(lifecycle_protocol_bytes), "lifecycle protocol"
    )
    bundle, summary = build_action_bundle(
        prompts,
        source_bundle_sha256=sha256_bytes(input_bytes),
        action_protocol=action_protocol,
        action_protocol_sha256=sha256_bytes(protocol_bytes),
        lifecycle_protocol=lifecycle_protocol,
        lifecycle_protocol_sha256=sha256_bytes(lifecycle_protocol_bytes),
        tokenizer=tokenizer,
        artifact_root=artifact_root,
    )
    output = args.output.expanduser().resolve()
    summary_path = (
        args.summary or output.with_name(f"{output.stem}_summary.json")
    ).expanduser().resolve()
    atomic_write_json(output, bundle)
    summary["prompt_bundle_sha256"] = sha256_file(output)
    atomic_write_json(summary_path, summary)
    print(f"wrote {output} ({len(bundle)} prompts, sha256={summary['prompt_bundle_sha256']})")
    print(f"wrote {summary_path} (sha256={sha256_file(summary_path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
