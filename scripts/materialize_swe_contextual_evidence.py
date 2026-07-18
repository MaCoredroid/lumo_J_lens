#!/usr/bin/env python3
"""Materialize paired SWE prompts for frozen contextual-evidence readouts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from typing import Any, Mapping, Sequence

import materialize_swe_jlens_prompts as RENDER
import materialize_swe_multitask_c1_probes as C1
import materialize_swe_multitask_initial_probes as C0


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "configs/qwen3-openai-codex.jinja"
PROTOCOL_KIND = "swe_contextual_evidence_update_protocol"
ANALYSIS_VERSION = "paired-evidence-update-development-v1"
OUTPUT_KIND = "swe_contextual_evidence_prompt"
MANIFEST_KIND = "swe_contextual_evidence_materialization"
STATES = ("before", "after")
COHORTS = ("development", "replication")
STRATA = ("novel_inference", "evidence_reweighting")
CONTROL_MATCH_STATUSES = (
    "matched_exposed_target_and_foils",
    "matched_newly_exposed_target_and_foils",
    "matched_novel_unexposed_target_and_foils",
    "matched_static_exposed_target_and_foils",
    "descriptive_exposure_frequency_mismatch",
    "descriptive_novel_target_with_exposed_foils",
)
TASK_CARD_FIELDS = ("why", "where", "evidence", "next", "claim_scope")
MAXIMUM_CONTEXT_LIMIT = 65_535


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


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def json_document_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json_document_bytes(value))
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


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def require_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def require_sha256(value: Any, label: str) -> str:
    digest = require_string(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def file_record(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    return {
        "path": display_path or str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalized_identifier(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def identifier_occurrences(text: str, aliases: Sequence[str]) -> dict[str, Any]:
    """Count aliases as complete Unicode identifier/path segments after NFKC+casefold."""
    normalized_text = normalized_identifier(text)
    normalized_aliases: dict[str, list[str]] = {}
    for index, alias in enumerate(aliases):
        value = require_string(alias, f"alias {index}")
        normalized = normalized_identifier(value)
        normalized_aliases.setdefault(normalized, []).append(value)
    per_alias: list[dict[str, Any]] = []
    total = 0
    for normalized, source_aliases in normalized_aliases.items():
        pattern = re.compile(rf"(?<!\w){re.escape(normalized)}(?!\w)", re.UNICODE)
        count = sum(1 for _ in pattern.finditer(normalized_text))
        total += count
        per_alias.append(
            {
                "source_aliases": source_aliases,
                "normalized_alias": normalized,
                "occurrences": count,
            }
        )
    return {
        "normalization": "NFKC_then_casefold",
        "matching": "exact_alias_with_unicode_identifier_boundaries",
        "identifier_occurrences": total,
        "present": total > 0,
        "per_alias": per_alias,
    }


def case_sensitive_identifier_occurrences(
    text: str, aliases: Sequence[str]
) -> dict[str, Any]:
    """Count exact aliases with the ASCII identifier boundary used by the audit."""
    seen: set[str] = set()
    per_alias: list[dict[str, Any]] = []
    total = 0
    for index, alias in enumerate(aliases):
        value = require_string(alias, f"alias {index}")
        if value in seen:
            raise ValueError("concept aliases are duplicated")
        seen.add(value)
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])")
        count = sum(1 for _ in pattern.finditer(text))
        total += count
        per_alias.append({"alias": value, "occurrences": count})
    return {
        "normalization": "case_sensitive_identifier_boundary_v1",
        "matching": "exact_alias_with_ascii_identifier_boundaries",
        "identifier_occurrences": total,
        "present": total > 0,
        "per_alias": per_alias,
    }


def validate_protocol_header(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    require_equal(protocol.get("schema_version"), 1, "protocol schema version")
    require_equal(protocol.get("kind"), PROTOCOL_KIND, "protocol kind")
    require_equal(
        protocol.get("analysis_version"), ANALYSIS_VERSION, "protocol analysis version"
    )
    require_equal(
        protocol.get("lens_outputs_used_for_boundary_or_labels"),
        False,
        "lens-output boundary/label flag",
    )
    pins = require_mapping(protocol.get("pins"), "protocol pins")
    for name in ("model", "tokenizer", "chat_template", "sources", "lenses"):
        require_mapping(pins.get(name), f"protocol {name} pin")
    sources = require_mapping(pins["sources"], "protocol source pins")
    for cohort in COHORTS:
        pin = require_mapping(sources.get(cohort), f"protocol {cohort} source pin")
        require_string(pin.get("run_root"), f"{cohort} run root")
        require_string(pin.get("campaign_path"), f"{cohort} campaign path")
        require_sha256(pin.get("campaign_sha256"), f"{cohort} campaign hash")
    layer_band = require_mapping(protocol.get("fixed_layer_band"), "fixed layer band")
    require_equal(layer_band.get("layers"), list(range(24, 48)), "fixed layer list")
    prompt_context = require_mapping(protocol.get("prompt_context"), "prompt context")
    require_equal(
        prompt_context.get("maximum_prompt_tokens"),
        MAXIMUM_CONTEXT_LIMIT,
        "maximum prompt tokens",
    )
    for name in (
        "numerical_certification",
        "score_reduction",
        "controls",
        "decision",
    ):
        require_mapping(protocol.get(name), f"protocol {name}")
    numerical = require_mapping(protocol["numerical_certification"], "numerical certification")
    require_mapping(numerical.get("primary_stable"), "primary stable certification")
    require_mapping(numerical.get("legacy_strict"), "legacy strict certification")
    return pins


def validate_form(
    form: Mapping[str, Any], tokenizer: Any, label: str, *, logit_vocabulary_size: int
) -> dict[str, Any]:
    kind = form.get("kind")
    if kind not in ("bare", "leading_space"):
        raise ValueError(f"{label} has an invalid form kind")
    text = require_string(form.get("text"), f"{label} text")
    if (kind == "leading_space") != text.startswith(" "):
        raise ValueError(f"{label} text does not match its form kind")
    token_id = require_integer(form.get("token_id"), f"{label} token ID")
    if token_id >= logit_vocabulary_size:
        raise ValueError(f"{label} token ID exceeds the pinned logit vocabulary")
    encoded = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if encoded != [token_id] or decoded != text:
        raise ValueError(
            f"{label} token pin changed for {text!r}: encoded={encoded}, decoded={decoded!r}"
        )
    return {"kind": kind, "text": text, "token_id": token_id}


def validate_expected_exposure(value: Any, label: str) -> dict[str, Any]:
    exposure = require_mapping(value, label)
    if set(exposure) != set(STATES):
        raise ValueError(f"{label} must declare exactly before and after")
    result: dict[str, Any] = {}
    for state in STATES:
        item = require_mapping(exposure.get(state), f"{label} {state}")
        if set(item) != {"present", "identifier_occurrences"}:
            raise ValueError(
                f"{label} {state} must contain present and identifier_occurrences"
            )
        present = item.get("present")
        if not isinstance(present, bool):
            raise ValueError(f"{label} {state} present must be boolean")
        count = require_integer(
            item.get("identifier_occurrences"), f"{label} {state} occurrence count"
        )
        if present != (count > 0):
            raise ValueError(f"{label} {state} exposure fields are inconsistent")
        result[state] = {"present": present, "identifier_occurrences": count}
    return result


def validate_concept(
    raw: Any,
    tokenizer: Any,
    label: str,
    *,
    expected_future: bool,
    logit_vocabulary_size: int,
) -> dict[str, Any]:
    concept = require_mapping(raw, label)
    concept_id = require_string(concept.get("id"), f"{label} ID")
    concept_label = require_string(concept.get("label"), f"{label} label")
    aliases = [
        require_string(alias, f"{label} alias {index}")
        for index, alias in enumerate(require_list(concept.get("aliases"), f"{label} aliases"))
    ]
    if not aliases:
        raise ValueError(f"{label} must declare at least one alias")
    require_equal(
        concept.get("exposure_normalization"),
        "case_sensitive_identifier_boundary_v1",
        f"{label} exposure normalization",
    )
    case_sensitive_identifier_occurrences("", aliases)
    # This validates normalized uniqueness without requiring a text occurrence.
    identifier_occurrences("", aliases)
    forms = [
        validate_form(
            require_mapping(form, f"{label} form {index}"),
            tokenizer,
            f"{label} form {index}",
            logit_vocabulary_size=logit_vocabulary_size,
        )
        for index, form in enumerate(require_list(concept.get("forms"), f"{label} forms"))
    ]
    if not forms:
        raise ValueError(f"{label} must declare at least one scored form")
    if len({form["token_id"] for form in forms}) != len(forms):
        raise ValueError(f"{label} form token IDs are duplicated")
    future_present = concept.get("future_present")
    require_equal(future_present, expected_future, f"{label} future-presence label")
    return {
        "id": concept_id,
        "label": concept_label,
        "aliases": aliases,
        "exposure_normalization": "case_sensitive_identifier_boundary_v1",
        "forms": forms,
        "expected_exposure": validate_expected_exposure(
            concept.get("expected_exposure"), f"{label} expected exposure"
        ),
        "future_present": future_present,
    }


def validate_task_card(value: Any, label: str) -> dict[str, str]:
    card = require_mapping(value, label)
    if set(card) != set(TASK_CARD_FIELDS):
        raise ValueError(f"{label} must contain exactly {', '.join(TASK_CARD_FIELDS)}")
    return {
        field: require_string(card.get(field), f"{label} {field}")
        for field in TASK_CARD_FIELDS
    }


def validate_task(
    raw: Any, tokenizer: Any, *, logit_vocabulary_size: int
) -> dict[str, Any]:
    task = require_mapping(raw, "protocol task")
    task_id = require_string(task.get("id"), "task ID")
    instance_id = require_string(task.get("instance_id"), f"task {task_id} instance ID")
    repo = require_string(task.get("repo"), f"task {task_id} repository")
    expected_instance_prefix = repo.replace("/", "__") + "-"
    if not instance_id.startswith(expected_instance_prefix):
        raise ValueError(f"task {task_id} instance ID does not match repository")
    cohort = task.get("cohort")
    if cohort not in COHORTS:
        raise ValueError(f"task {task_id} has an invalid cohort")
    global_index = require_integer(
        task.get("after_global_request_index"), f"task {task_id} global index", minimum=2
    )
    task_index = require_integer(
        task.get("after_task_request_index"), f"task {task_id} task index", minimum=2
    )
    raw_sha = require_mapping(task.get("raw_sha256"), f"task {task_id} raw hashes")
    if set(raw_sha) != {"before", "after", "label"}:
        raise ValueError(f"task {task_id} must pin before, after, and label hashes")
    raw_hashes = {
        state: require_sha256(raw_sha.get(state), f"task {task_id} {state} hash")
        for state in ("before", "after", "label")
    }
    stratum = task.get("stratum")
    if stratum not in STRATA:
        raise ValueError(f"task {task_id} has an invalid stratum")
    primary_control_eligible = task.get("primary_control_eligible")
    if not isinstance(primary_control_eligible, bool):
        raise ValueError(f"task {task_id} primary control eligibility must be boolean")
    control_match_status = task.get("control_match_status")
    if control_match_status not in CONTROL_MATCH_STATUSES:
        raise ValueError(f"task {task_id} has an invalid control match status")
    if primary_control_eligible != str(control_match_status).startswith("matched_"):
        raise ValueError(f"task {task_id} control eligibility and match status disagree")
    target = validate_concept(
        task.get("target"),
        tokenizer,
        f"task {task_id} target",
        expected_future=True,
        logit_vocabulary_size=logit_vocabulary_size,
    )
    raw_foils = require_list(task.get("foils"), f"task {task_id} foils")
    if len(raw_foils) != 3:
        raise ValueError(f"task {task_id} must declare exactly three foils")
    foils = [
        validate_concept(
            foil,
            tokenizer,
            f"task {task_id} foil {index}",
            expected_future=False,
            logit_vocabulary_size=logit_vocabulary_size,
        )
        for index, foil in enumerate(raw_foils)
    ]
    concept_ids = [target["id"], *(foil["id"] for foil in foils)]
    if len(concept_ids) != len(set(concept_ids)):
        raise ValueError(f"task {task_id} concept IDs are duplicated")
    return {
        "id": task_id,
        "instance_id": instance_id,
        "repo": repo,
        "cohort": cohort,
        "after_global_request_index": global_index,
        "after_task_request_index": task_index,
        "raw_sha256": raw_hashes,
        "stratum": stratum,
        "primary_control_eligible": primary_control_eligible,
        "control_match_status": control_match_status,
        "target": target,
        "foils": foils,
        "task_card": validate_task_card(task.get("task_card"), f"task {task_id} card"),
    }


def assistant_reasoning_and_visible_text(assistant: Mapping[str, Any]) -> str:
    reasoning = assistant.get("reasoning_content")
    duplicate = assistant.get("reasoning")
    if reasoning is None and duplicate is None:
        reasoning = ""
    elif not isinstance(reasoning, str) or duplicate != reasoning:
        raise ValueError("future assistant reasoning fields are absent or inconsistent")
    visible = C1.flatten_text_content(assistant.get("content"), "future assistant content")
    return reasoning + "\n" + visible


def validate_request_extension(
    earlier: Mapping[str, Any], later: Mapping[str, Any], label: str
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    earlier_messages = require_list(earlier.get("messages"), f"{label} earlier messages")
    later_messages = require_list(later.get("messages"), f"{label} later messages")
    if (
        len(later_messages) <= len(earlier_messages)
        or later_messages[: len(earlier_messages)] != earlier_messages
    ):
        raise ValueError(f"{label} does not exactly preserve and extend its message prefix")
    for field in ("model", "tools", "chat_template_kwargs"):
        require_equal(later.get(field), earlier.get(field), f"{label} {field}")
    extension = [
        require_mapping(message, f"{label} extension message {index}")
        for index, message in enumerate(later_messages[len(earlier_messages) :])
    ]
    if extension[0].get("role") != "assistant":
        raise ValueError(f"{label} extension does not start with an assistant completion")
    if any(message.get("role") != "tool" for message in extension[1:]):
        raise ValueError(f"{label} extension contains more than one assistant completion")
    return extension[0], extension


def form_exposure(
    token_ids: Sequence[int], forms: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for form in forms:
        token_id = int(form["token_id"])
        indices = [index for index, observed in enumerate(token_ids) if observed == token_id]
        records.append(
            {
                "kind": form["kind"],
                "text": form["text"],
                "token_id": token_id,
                "token_occurrences": len(indices),
                "last_token_distance": (len(token_ids) - 1 - indices[-1]) if indices else None,
            }
        )
    return records


def concept_exposure(
    raw_messages_text: str,
    rendered: str,
    token_ids: Sequence[int],
    concept: Mapping[str, Any],
) -> dict[str, Any]:
    identifiers = case_sensitive_identifier_occurrences(
        raw_messages_text, concept["aliases"]
    )
    return {
        **identifiers,
        "source": "newline_joined_recursive_string_values_from_raw_request_messages",
        "raw_messages_sha256": sha256_text(raw_messages_text),
        "supplemental_rendered": {
            "case_sensitive": case_sensitive_identifier_occurrences(
                rendered, concept["aliases"]
            ),
            "nfkc_casefold": identifier_occurrences(rendered, concept["aliases"]),
        },
        "forms": form_exposure(token_ids, concept["forms"]),
    }


def validate_declared_exposure(
    actual: Mapping[str, Any], concept: Mapping[str, Any], state: str, label: str
) -> None:
    expected = concept["expected_exposure"][state]
    require_equal(actual.get("present"), expected["present"], f"{label} {state} presence")
    require_equal(
        actual.get("identifier_occurrences"),
        expected["identifier_occurrences"],
        f"{label} {state} identifier occurrences",
    )


def validate_source_record(value: Any, label: str, expected_sha256: str) -> dict[str, Any]:
    source = require_mapping(value, label)
    path = require_string(source.get("path"), f"{label} path")
    byte_count = require_integer(source.get("bytes"), f"{label} bytes")
    digest = require_sha256(source.get("sha256"), f"{label} hash")
    require_equal(digest, expected_sha256, f"{label} frozen hash")
    return {"path": path, "bytes": byte_count, "sha256": digest}


def validate_triplet(
    task: Mapping[str, Any], value: Any
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    triplet = require_mapping(value, f"task {task['id']} input")
    requests_value = require_mapping(triplet.get("requests"), f"task {task['id']} requests")
    sources_value = require_mapping(triplet.get("sources"), f"task {task['id']} sources")
    names = ("before", "after", "label")
    if set(requests_value) != set(names) or set(sources_value) != set(names):
        raise ValueError(f"task {task['id']} input must contain before, after, and label")
    requests = {
        name: require_mapping(requests_value[name], f"task {task['id']} {name} request")
        for name in names
    }
    sources = {
        name: validate_source_record(
            sources_value[name], f"task {task['id']} {name} source", task["raw_sha256"][name]
        )
        for name in names
    }
    global_index = task["after_global_request_index"]
    expected_names = {
        "before": f"chat_{global_index - 1:04d}.json",
        "after": f"chat_{global_index:04d}.json",
        "label": f"chat_{global_index + 1:04d}.json",
    }
    for name, expected in expected_names.items():
        if Path(sources[name]["path"]).name != expected:
            raise ValueError(f"task {task['id']} {name} source index mismatch")
        offset = {"before": -1, "after": 0, "label": 1}[name]
        C1.validate_capture_request(
            requests[name], request_index=global_index + offset
        )

    before_assistant, before_extension = validate_request_extension(
        requests["before"], requests["after"], f"task {task['id']} before-to-after"
    )
    label_assistant, label_extension = validate_request_extension(
        requests["after"], requests["label"], f"task {task['id']} after-to-label"
    )
    del before_assistant
    after_messages = require_list(requests["after"].get("messages"), "after messages")
    observed_task_index = 1 + sum(
        require_mapping(message, "after message").get("role") == "assistant"
        for message in after_messages
    )
    require_equal(
        observed_task_index,
        task["after_task_request_index"],
        f"task {task['id']} task-local request index",
    )
    initial_messages = after_messages[:2]
    initial_text = json.dumps(initial_messages, sort_keys=True, ensure_ascii=False)
    if identifier_occurrences(initial_text, [task["instance_id"]])["identifier_occurrences"] < 1:
        raise ValueError(f"task {task['id']} request chain is not bound to its instance ID")

    future_text = assistant_reasoning_and_visible_text(label_assistant)
    target_future = case_sensitive_identifier_occurrences(
        future_text, task["target"]["aliases"]
    )
    if not target_future["present"]:
        raise ValueError(f"task {task['id']} future assistant text does not contain target")
    foil_future: list[dict[str, Any]] = []
    for foil in task["foils"]:
        evidence = case_sensitive_identifier_occurrences(future_text, foil["aliases"])
        if evidence["present"]:
            raise ValueError(f"task {task['id']} future assistant text contains foil {foil['id']}")
        foil_future.append({"id": foil["id"], **evidence})

    boundary = {
        "before_to_after_exact_prefix_extension": True,
        "after_to_label_exact_prefix_extension": True,
        "before_message_count": len(requests["before"]["messages"]),
        "after_message_count": len(requests["after"]["messages"]),
        "label_message_count": len(requests["label"]["messages"]),
        "before_extension_message_count": len(before_extension),
        "label_extension_message_count": len(label_extension),
        "observed_after_task_request_index": observed_task_index,
    }
    future_audit = {
        "assistant_message_sha256": sha256_json(label_assistant),
        "extension_sha256": sha256_json(label_extension),
        "reasoning_and_visible_text_sha256": sha256_text(future_text),
        "target": {"id": task["target"]["id"], **target_future},
        "foils": foil_future,
        "target_present": True,
        "all_foils_absent": True,
        "future_text_retained": False,
    }
    return requests, sources, {"boundary": boundary, "future": future_audit}


def build_evidence_bundle(
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    task_inputs: Mapping[str, Any],
    tokenizer: Any,
    template: str,
    protocol_source: Mapping[str, Any] | None = None,
    environment_sources: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pins = validate_protocol_header(protocol)
    require_sha256(protocol_sha256, "protocol hash")
    tokenizer_pin = require_mapping(pins["tokenizer"], "tokenizer pin")
    logit_vocabulary_size = require_integer(
        tokenizer_pin.get("logit_vocabulary_size"), "logit vocabulary size", minimum=1
    )
    tasks = [
        validate_task(raw, tokenizer, logit_vocabulary_size=logit_vocabulary_size)
        for raw in require_list(protocol.get("tasks"), "protocol tasks")
    ]
    if not tasks:
        raise ValueError("protocol contains no tasks")
    task_ids = [task["id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("protocol task IDs are duplicated")
    if set(task_inputs) != set(task_ids):
        raise ValueError("task inputs do not exactly match frozen protocol tasks")

    token_text_by_id: dict[int, str] = {}
    for task in tasks:
        for concept in (task["target"], *task["foils"]):
            for form in concept["forms"]:
                token_id = form["token_id"]
                previous = token_text_by_id.setdefault(token_id, form["text"])
                if previous != form["text"]:
                    raise ValueError("one scored token ID is pinned to multiple token texts")
    score_token_ids = sorted(token_text_by_id)

    prompts: list[dict[str, Any]] = []
    task_manifest: list[dict[str, Any]] = []
    maximum_tokens = protocol["prompt_context"]["maximum_prompt_tokens"]
    for task_ordinal, task in enumerate(tasks):
        requests, sources, audits = validate_triplet(task, task_inputs[task["id"]])
        rendered_by_state: dict[str, str] = {}
        token_ids_by_state: dict[str, list[int]] = {}
        normalized_by_state: dict[str, dict[str, Any]] = {}
        exposures_by_state: dict[str, dict[str, Any]] = {}
        prompt_manifest: list[dict[str, Any]] = []
        for state in STATES:
            raw_messages_text = "\n".join(
                C1.flatten_string_values(requests[state]["messages"])
            )
            rendered, token_ids, normalized_count, normalized_messages = C1.render_request(
                tokenizer, request=requests[state], template=template
            )
            if len(token_ids) > maximum_tokens:
                raise ValueError(
                    f"task {task['id']} {state} prompt has {len(token_ids)} "
                    f"tokens, exceeding {maximum_tokens}"
                )
            rendered_by_state[state] = rendered
            token_ids_by_state[state] = token_ids
            normalized_by_state[state] = {
                "messages_sha256": sha256_json(normalized_messages),
                "string_tool_call_argument_count": normalized_count,
            }
            target_exposure = concept_exposure(
                raw_messages_text, rendered, token_ids, task["target"]
            )
            validate_declared_exposure(
                target_exposure, task["target"], state, f"task {task['id']} target"
            )
            foil_exposure: list[dict[str, Any]] = []
            for foil in task["foils"]:
                evidence = concept_exposure(raw_messages_text, rendered, token_ids, foil)
                validate_declared_exposure(
                    evidence, foil, state, f"task {task['id']} foil {foil['id']}"
                )
                foil_exposure.append({"id": foil["id"], **evidence})
            exposures_by_state[state] = {
                "target": {"id": task["target"]["id"], **target_exposure},
                "foils": foil_exposure,
            }

        concepts = {
            "target": copy.deepcopy(task["target"]),
            "foils": copy.deepcopy(task["foils"]),
        }
        task_metadata = {
            key: copy.deepcopy(task[key])
            for key in (
                "id",
                "instance_id",
                "repo",
                "cohort",
                "after_global_request_index",
                "after_task_request_index",
                "stratum",
                "primary_control_eligible",
                "control_match_status",
            )
        }
        for state in STATES:
            text = rendered_by_state[state]
            token_ids = token_ids_by_state[state]
            prompt_hash = sha256_text(text)
            token_hash = sha256_json(token_ids)
            prompt_id = f"swe-contextual-evidence-{task['id']}-{state}"
            record = {
                "id": prompt_id,
                "text": text,
                "token_ids": token_ids,
                "score_token_ids": list(score_token_ids),
                "metadata": {
                    "kind": OUTPUT_KIND,
                    "analysis_version": ANALYSIS_VERSION,
                    "protocol_sha256": protocol_sha256,
                    "lens_outputs_used_for_boundary_or_labels": False,
                    "state": state,
                    "task": copy.deepcopy(task_metadata),
                    "raw_sha256": copy.deepcopy(task["raw_sha256"]),
                    "prompt": {
                        "sha256": prompt_hash,
                        "token_ids_sha256": token_hash,
                        "token_count": len(token_ids),
                        "normalized_messages_sha256": normalized_by_state[state][
                            "messages_sha256"
                        ],
                        "normalized_string_tool_call_arguments": normalized_by_state[state][
                            "string_tool_call_argument_count"
                        ],
                    },
                    "concepts": copy.deepcopy(concepts),
                    "exposure": copy.deepcopy(exposures_by_state[state]),
                    "task_card": copy.deepcopy(task["task_card"]),
                    "fixed_layer_band": copy.deepcopy(protocol["fixed_layer_band"]),
                    "score_reduction": copy.deepcopy(protocol["score_reduction"]),
                },
            }
            prompts.append(record)
            prompt_manifest.append(
                {
                    "id": prompt_id,
                    "state": state,
                    "prompt_sha256": prompt_hash,
                    "token_ids_sha256": token_hash,
                    "prompt_token_count": len(token_ids),
                    "score_token_ids_sha256": sha256_json(score_token_ids),
                    "score_token_count": len(score_token_ids),
                    "exposure": copy.deepcopy(exposures_by_state[state]),
                }
            )
        task_manifest.append(
            {
                "task_ordinal": task_ordinal,
                "task": copy.deepcopy(task_metadata),
                "raw_sources": copy.deepcopy(sources),
                "boundary_audit": audits["boundary"],
                "future_label_audit": audits["future"],
                "concepts": copy.deepcopy(concepts),
                "task_card": copy.deepcopy(task["task_card"]),
                "prompts": prompt_manifest,
            }
        )

    prompt_bytes = json_document_bytes(prompts)
    manifest = {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "analysis_version": ANALYSIS_VERSION,
        "protocol": {
            "sha256": protocol_sha256,
            **(copy.deepcopy(dict(protocol_source)) if protocol_source is not None else {}),
        },
        "lens_outputs_used_for_boundary_or_labels": False,
        "fixed_layer_band": copy.deepcopy(protocol["fixed_layer_band"]),
        "score_reduction": copy.deepcopy(protocol["score_reduction"]),
        "environment_sources": copy.deepcopy([dict(value) for value in environment_sources]),
        "score_vocabulary": {
            "token_ids": score_token_ids,
            "token_text": {
                str(token_id): token_text_by_id[token_id]
                for token_id in score_token_ids
            },
            "token_ids_sha256": sha256_json(score_token_ids),
            "token_count": len(score_token_ids),
            "scope": "union_of_all_declared_target_and_foil_forms_on_every_prompt",
        },
        "task_count": len(tasks),
        "prompt_count": len(prompts),
        "tasks": task_manifest,
        "prompt_bundle": {
            "sha256": sha256_bytes(prompt_bytes),
            "bytes": len(prompt_bytes),
            "count": len(prompts),
            "serialization": "indented_sorted_key_ascii_json_with_trailing_newline",
        },
    }
    return prompts, manifest


def validate_environment(
    protocol: Mapping[str, Any],
    *,
    tokenizer: Any,
    snapshot: Path,
    template_path: Path,
) -> list[dict[str, Any]]:
    pins = validate_protocol_header(protocol)
    model = require_mapping(pins["model"], "model pin")
    tokenizer_pin = require_mapping(pins["tokenizer"], "tokenizer pin")
    require_equal(model.get("repo_id"), RENDER.MODEL_REPO, "model repository")
    require_equal(model.get("revision"), RENDER.MODEL_REVISION, "model revision")
    require_equal(tokenizer_pin.get("repo_id"), RENDER.MODEL_REPO, "tokenizer repository")
    require_equal(tokenizer_pin.get("revision"), RENDER.MODEL_REVISION, "tokenizer revision")
    require_equal(len(tokenizer), tokenizer_pin.get("vocabulary_size"), "tokenizer size")
    require_equal(
        tokenizer_pin.get("logit_vocabulary_size"),
        C0.LOGIT_VOCABULARY_SIZE,
        "logit vocabulary size",
    )
    files = (
        (snapshot / "config.json", model.get("config_sha256"), "model config"),
        (
            snapshot / "model.safetensors.index.json",
            model.get("index_sha256"),
            "model index",
        ),
        (snapshot / "tokenizer.json", tokenizer_pin.get("json_sha256"), "tokenizer JSON"),
    )
    records: list[dict[str, Any]] = []
    for path, expected, label in files:
        digest = require_sha256(expected, f"{label} pin")
        require_equal(sha256_file(path), digest, f"{label} hash")
        records.append(file_record(path, display_path=path.name))
    template_pin = require_mapping(pins["chat_template"], "chat template pin")
    require_equal(
        sha256_file(template_path),
        require_sha256(template_pin.get("sha256"), "chat template hash pin"),
        "chat template hash",
    )
    records.append(file_record(template_path, display_path=str(template_pin.get("path"))))
    return records


def load_campaign_sources(
    protocol: Mapping[str, Any], roots: Mapping[str, Path]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    pins = require_mapping(protocol["pins"], "protocol pins")
    source_pins = require_mapping(pins["sources"], "source pins")
    campaigns: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for cohort in COHORTS:
        source = require_mapping(source_pins[cohort], f"{cohort} source pin")
        campaign_path = Path(require_string(source.get("campaign_path"), "campaign path"))
        if not campaign_path.is_absolute():
            campaign_path = ROOT / campaign_path
        campaign_path = campaign_path.expanduser().resolve(strict=True)
        expected = require_sha256(source.get("campaign_sha256"), "campaign hash")
        require_equal(sha256_file(campaign_path), expected, f"{cohort} campaign hash")
        campaign = require_mapping(json.loads(campaign_path.read_bytes()), f"{cohort} campaign")
        generation = require_mapping(campaign.get("generation"), f"{cohort} generation")
        require_equal(generation.get("model_repo_id"), RENDER.MODEL_REPO, "campaign model")
        require_equal(generation.get("model_revision"), RENDER.MODEL_REVISION, "campaign revision")
        require_equal(generation.get("served_model"), RENDER.SERVED_MODEL, "campaign served model")
        require_equal(generation.get("max_model_len"), 65_536, "campaign model length")
        root = roots[cohort].expanduser().resolve(strict=True)
        if not (root / "proxy_dumps").is_dir():
            raise ValueError(f"{cohort} root has no proxy_dumps directory")
        campaigns[cohort] = dict(campaign)
        records.append(file_record(campaign_path, display_path=str(source["campaign_path"])))
    return campaigns, records


def load_task_inputs(
    protocol: Mapping[str, Any],
    roots: Mapping[str, Path],
    campaigns: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_task in require_list(protocol.get("tasks"), "protocol tasks"):
        task = require_mapping(raw_task, "protocol task")
        task_id = require_string(task.get("id"), "task ID")
        cohort = require_string(task.get("cohort"), f"task {task_id} cohort")
        if cohort not in roots:
            raise ValueError(f"task {task_id} has no source root")
        instance_id = require_string(task.get("instance_id"), f"task {task_id} instance ID")
        campaign_instances = require_list(
            campaigns[cohort].get("instance_ids"), "campaign instances"
        )
        if instance_id not in campaign_instances:
            raise ValueError(f"task {task_id} is absent from its pinned campaign")
        global_index = require_integer(
            task.get("after_global_request_index"), f"task {task_id} global index", minimum=2
        )
        requests: dict[str, Any] = {}
        sources: dict[str, Any] = {}
        for name, index in (
            ("before", global_index - 1),
            ("after", global_index),
            ("label", global_index + 1),
        ):
            path = roots[cohort] / "proxy_dumps" / f"chat_{index:04d}.json"
            path = path.resolve(strict=True)
            requests[name] = require_mapping(
                json.loads(path.read_bytes()), f"task {task_id} {name}"
            )
            sources[name] = file_record(
                path,
                display_path=(
                    f"{protocol['pins']['sources'][cohort]['run_root']}"
                    f"/proxy_dumps/{path.name}"
                ),
            )
        result[task_id] = {"requests": requests, "sources": sources}
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-prompts", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--development-root", type=Path)
    parser.add_argument("--replication-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    protocol_path = args.protocol.expanduser().resolve(strict=True)
    protocol_bytes = protocol_path.read_bytes()
    protocol = require_mapping(json.loads(protocol_bytes), "protocol")
    validate_protocol_header(protocol)
    source_pins = protocol["pins"]["sources"]
    roots = {
        "development": Path(
            args.development_root or ROOT / source_pins["development"]["run_root"]
        ),
        "replication": Path(
            args.replication_root or ROOT / source_pins["replication"]["run_root"]
        ),
    }
    template_path = args.template.expanduser().resolve(strict=True)
    template = template_path.read_text(encoding="utf-8")
    snapshot = Path(
        args.tokenizer_path
        or snapshot_download(
            RENDER.MODEL_REPO,
            revision=RENDER.MODEL_REVISION,
            local_files_only=True,
        )
    ).expanduser().resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    RENDER.validate_tokenizer(tokenizer, snapshot)
    environment = validate_environment(
        protocol, tokenizer=tokenizer, snapshot=snapshot, template_path=template_path
    )
    campaigns, campaign_sources = load_campaign_sources(protocol, roots)
    environment.extend(campaign_sources)
    inputs = load_task_inputs(protocol, roots, campaigns)
    protocol_source = file_record(protocol_path, display_path=str(args.protocol))
    prompts, manifest = build_evidence_bundle(
        protocol,
        protocol_sha256=sha256_bytes(protocol_bytes),
        task_inputs=inputs,
        tokenizer=tokenizer,
        template=template,
        protocol_source=protocol_source,
        environment_sources=environment,
    )
    output_prompts = args.output_prompts.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve()
    manifest["prompt_bundle"]["path"] = str(args.output_prompts)
    atomic_write_json(output_prompts, prompts)
    require_equal(
        sha256_file(output_prompts), manifest["prompt_bundle"]["sha256"], "written prompt bundle"
    )
    atomic_write_json(output_manifest, manifest)
    print(
        f"wrote {output_prompts} ({len(prompts)} prompts, "
        f"sha256={manifest['prompt_bundle']['sha256']})"
    )
    print(f"wrote {output_manifest} (sha256={sha256_file(output_manifest)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
