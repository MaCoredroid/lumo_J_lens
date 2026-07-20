#!/usr/bin/env python3
"""Build blinded annotation packets for the prospective V4 chain target.

The exporter reads only the pinned development prompt bundle and the
label-free alignment index.  It never opens a report, an activation artifact,
an official outcome, or an existing observable-event sidecar.  Completion
text is recovered from the exact rendered-prefix extension from prompt t to
prompt t+1 and must reproduce the already-pinned assistant-text SHA-256.

Two packet passes are supported:

* ``completion_chain`` exposes only completion-t assistant prose.
* ``prefix_novelty`` exposes a locked hypothesis span and a tool-redacted
  prefix.  It consumes completion-pass records only to obtain that span; no
  other annotation slot is copied into the packet.

Any rendered-prefix discontinuity fails closed.  In particular, this module
does not invent a normalization for an optional thinking boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_epistemic_chain_annotation.json"
CODEBOOK_PATH = ROOT / "configs" / "swe_task_state_v4_epistemic_chain_codebook.json"
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_FILE_SHA256 = "4c7e985a349f36e8700922dd40652bb8c94e9939aa102fdf931288808f64166d"
CONFIG_CANONICAL_SHA256 = "0b64d71e212a5b92f8330c5eebf85a1ae67b09ec8c95cfcdf5a4f43ecbce79c2"
CODEBOOK_FILE_SHA256 = "0575f3ffeccf10fe737877ce47374f21ece96197fcb64f0d61bf514b5e9e75f9"
CODEBOOK_CANONICAL_SHA256 = "418f054f5feb2b2bac82fb38c23aea9fca8d44ad9390ba4900bf4ffc328c91e2"
ANNOTATOR_PROMPT_OR_MODEL_IDENTITY_SHA256 = (
    "9f6d63af6ff655edc1dfd1315b1a31c6fe7dc23d254122981c4a7b54d1f083e6"
)
SCHEMA_VERSION = 1
FORBIDDEN_PATH_FRAGMENTS = ("reserved", "validation")

COMPLETION_PACKET_KIND = (
    "swe_task_state_v4_epistemic_chain_completion_annotation_packet"
)
PREFIX_PACKET_KIND = (
    "swe_task_state_v4_epistemic_chain_prefix_novelty_annotation_packet"
)
PACKET_MANIFEST_KIND = "swe_task_state_v4_epistemic_chain_packet_manifest"
PASSES = ("completion_chain", "prefix_novelty")

GENERATION_BOUNDARY = "<|im_start|>assistant\n<think>\n"
THINK_CLOSE = "</think>"
MESSAGE_END = "<|im_end|>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"

SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


class AnnotationPacketError(ValueError):
    """Raised when an annotation boundary cannot be authenticated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnnotationPacketError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray)),
        f"{label} must be an array",
    )
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnnotationPacketError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_strict(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnnotationPacketError(f"cannot load strict JSON {path}: {error}") from error


def _lexical_path_forbidden(path: Path) -> bool:
    normalized = Path(os.path.normpath(os.path.abspath(os.fspath(path))))
    return any(
        fragment in component.lower()
        for component in normalized.parts
        for fragment in FORBIDDEN_PATH_FRAGMENTS
    )


def lexical_path_preflight(paths: Iterable[Path | None]) -> None:
    """Reject forbidden path text before a stat, resolve, hash, or read."""

    for raw in paths:
        if raw is not None and _lexical_path_forbidden(Path(raw)):
            raise AnnotationPacketError(
                f"forbidden path rejected before filesystem access: {raw}"
            )


def canonical_path_preflight(
    *, input_paths: Iterable[Path | None], output_paths: Iterable[Path | None]
) -> None:
    """Reject forbidden canonical inputs and existing output-parent symlinks."""

    for raw, is_input in [
        *((path, True) for path in input_paths),
        *((path, False) for path in output_paths),
    ]:
        if raw is None:
            continue
        path = Path(raw)
        try:
            resolved = (
                path.resolve(strict=True)
                if is_input
                else path.parent.resolve(strict=True) / path.name
            )
        except OSError as error:
            raise AnnotationPacketError(f"cannot resolve annotation path {path}: {error}") from error
        if _lexical_path_forbidden(resolved):
            raise AnnotationPacketError(
                f"forbidden canonical path rejected before file read or write: {path}"
            )


def validate_config(value: Any) -> dict[str, Any]:
    """Validate the entire frozen config by its canonical JSON identity."""

    config = dict(_mapping(value, "epistemic-chain annotation config"))
    observed = sha256_bytes(canonical_json_bytes(config))
    _require(
        observed == CONFIG_CANONICAL_SHA256,
        "epistemic-chain annotation config differs from the frozen contract",
    )
    _require(
        config.get("schema_version") == 1
        and config.get("id") == "visible-novel-epistemic-action-chain-v1"
        and config.get("target_name")
        == "visible_novel_epistemic_action_chain_v1",
        "epistemic-chain annotation config identity changed",
    )
    materialization_scope = _mapping(
        config.get("packet_materialization_scope"), "packet materialization scope"
    )
    _require(
        materialization_scope
        == {
            "artifact_status": (
                "packet_materialization_passed_target_pass_annotation_not_run"
            ),
            "artifact_kind": "blinded_annotation_packets_only",
            "development_data_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "annotation_not_run_for_target_pass": True,
            "target_pass_annotation_records_in_artifact": False,
        },
        "packet materialization scope changed",
    )
    codebook = _mapping(
        config.get("annotation_codebook_contract"), "annotation codebook contract"
    )
    _require(
        codebook
        == {
            "path": "configs/swe_task_state_v4_epistemic_chain_codebook.json",
            "sha256": CODEBOOK_FILE_SHA256,
            "size_bytes": 10264,
            "schema_version": 1,
            "kind": "swe_task_state_v4_epistemic_chain_annotation_codebook",
            "annotator_prompt_or_model_identity_sha256": (
                ANNOTATOR_PROMPT_OR_MODEL_IDENTITY_SHA256
            ),
        },
        "annotation codebook binding changed",
    )
    claim_scope = _mapping(config.get("claim_scope"), "claim scope")
    _require(
        claim_scope.get("target_definition_frozen") is True
        and claim_scope.get("target_is_future_trace_visible_proposition_chain")
        is True
        and claim_scope.get("target_is_not_private_chain_of_thought") is True
        and claim_scope.get(
            "semantic_sentence_or_chain_decoding_established_before_gate_and_evaluation"
        )
        is False
        and claim_scope.get("cot_or_cot_alike_decoding_established") is False
        and claim_scope.get("private_chain_of_thought_decoding_established")
        is False
        and claim_scope.get("affect_or_emotion_decoding_established") is False,
        "epistemic-chain claim scope changed",
    )
    source = _mapping(config.get("source_contract"), "source contract")
    prompt = _mapping(
        source.get("development_prompt_bundle"), "development prompt binding"
    )
    alignment = _mapping(
        source.get("label_free_alignment_index"), "alignment binding"
    )
    _require(
        prompt
        == {
            "path": ".cache/swe_state_interpreter_v3_development/prompts.json",
            "sha256": "17f664b3029220458ff62b8e80a90ec5e796f0372217f02b727d6589df38d3d0",
            "size_bytes": 803705234,
            "row_count": 1708,
        }
        and alignment
        == {
            "path": ".cache/swe_task_state_v4_raw_capture/n60-final/alignment-index-v4.json",
            "sha256": "c0d9e4bd6ad8f6962cf58776441ec4ef042eebbbae8a9959b4149e605c9ae38e",
            "size_bytes": 567392,
            "stable_row_count": 1606,
        }
        and source.get("assistant_text_semantics")
        == "reasoning_content_or_reasoning_once_then_content"
        and source.get("assistant_tool_arguments_excluded") is True
        and source.get("tool_results_excluded") is True
        and source.get("materialized_completion_required") is True,
        "epistemic-chain source contract changed",
    )
    exclusions = source.get("frozen_unknown_exclusions")
    _require(
        exclusions
        == [
            {
                "source_id_sha256": "a496d5afee65b937424edd85f1e95db436ba1692a81ac46c644f8c9f5fc0c436",
                "current_rendered_prefix_sha256": "15b3f25f29d8226f473ebe89f4b2d077da5c775c796f1372db157b27eb010b8a",
                "following_source_id_sha256": "c336deb3c4cef156e8c45ef1f811aebba61a42706dd6a8c95e787c53613e1f8b",
                "following_rendered_prefix_sha256": "b304e3f45953f81d1f0cee1ca974cd07b53d8d8ee585811665abe411f89bba20",
                "following_raw_request_sha256": "8cf47fb7175e3fc380aecbd51657e6dbc4c6e492ba49ce0fc1a4f55d9fe6e45d",
                "materialized_completion_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "current_prefix_char_count": 78648,
                "following_prefix_char_count": 79907,
                "common_prefix_char_count": 78642,
                "dropped_current_suffix_sha256": "a15be88304e589c6852fe518f0ec05f7b2ecf92fcf826a210f3ed76f71d66ada",
                "following_extension_sha256": "6d050648a92765ee06bef0d09991591fb671fcf903a5bc0883a5b165a0e50541",
                "unknown_reason": "frozen_nonprefix_empty_assistant_rendering",
            }
        ],
        "frozen rendered-prefix unknown exclusion changed",
    )
    return config


def validate_codebook(value: Any) -> dict[str, Any]:
    """Validate the exact frozen, synthetic pre-annotation codebook."""

    codebook = dict(_mapping(value, "epistemic-chain annotation codebook"))
    _require(
        sha256_bytes(canonical_json_bytes(codebook)) == CODEBOOK_CANONICAL_SHA256,
        "epistemic-chain annotation codebook differs from the frozen contract",
    )
    _require(
        set(codebook)
        == {
            "schema_version",
            "kind",
            "id",
            "status",
            "scope",
            "annotator_prompt_contract",
            "chain_rule",
            "span_boundaries",
            "positive_examples",
            "negative_examples",
            "novelty_rule",
            "novelty_examples",
            "unknown_reasons",
            "unknown_record_rule",
            "earliest_chain_example",
        }
        and codebook.get("schema_version") == 1
        and codebook.get("kind")
        == "swe_task_state_v4_epistemic_chain_annotation_codebook"
        and codebook.get("id")
        == "visible-novel-epistemic-action-chain-codebook-v1"
        and codebook.get("status")
        == "frozen_pre_annotation_synthetic_examples_only",
        "epistemic-chain annotation codebook identity changed",
    )
    scope = _mapping(codebook.get("scope"), "codebook scope")
    _require(
        scope.get("examples_are_synthetic") is True
        and scope.get("development_annotation_only") is True
        and scope.get("reserved_validation_closed") is True
        and scope.get("private_chain_of_thought_ground_truth_claimed") is False
        and scope.get("affect_or_emotion_targeted") is False,
        "codebook scope changed",
    )
    prompt = _mapping(
        codebook.get("annotator_prompt_contract"), "annotator prompt contract"
    )
    payload = prompt.get("payload")
    _require(
        isinstance(payload, str)
        and bool(payload)
        and prompt.get("prompt_or_model_identity_sha256")
        == ANNOTATOR_PROMPT_OR_MODEL_IDENTITY_SHA256
        and sha256_text(payload) == ANNOTATOR_PROMPT_OR_MODEL_IDENTITY_SHA256,
        "annotator prompt or model identity digest changed",
    )
    chain_rule = _mapping(codebook.get("chain_rule"), "codebook chain rule")
    _require(
        chain_rule.get("order") == "E_before_H_before_A"
        and chain_rule.get("maximum_intervening_prose_clauses_between_H_and_A")
        == 2
        and chain_rule.get("earliest_qualifying_chain_only") is True
        and chain_rule.get("implicit_relation_is_not_positive") is True
        and chain_rule.get("tool_arguments_and_tool_results_are_not_annotator_visible_evidence")
        is True,
        "codebook chain rule changed",
    )
    spans = _mapping(codebook.get("span_boundaries"), "codebook span boundaries")
    _require(
        spans.get("coordinate_system")
        == "unicode_code_point_index_into_exact_materialized_assistant_text"
        and spans.get("start") == "inclusive"
        and spans.get("end") == "exclusive"
        and spans.get("span_must_be_nonempty") is True
        and spans.get("assistant_tool_call_blocks_excluded") is True
        and spans.get("tool_result_blocks_excluded") is True,
        "codebook span boundary contract changed",
    )
    positives = _sequence(codebook.get("positive_examples"), "positive examples")
    _require(
        {
            str(_mapping(item, "positive example").get("belief_edge"))
            for item in positives
        }
        == {"supports", "refutes", "narrows"}
        and all(
            _mapping(item, "positive example").get("has_chain") is True
            for item in positives
        ),
        "codebook positive edge examples changed",
    )
    for raw_example in positives:
        example = _mapping(raw_example, "positive example")
        text_value = example.get("assistant_text")
        span_texts = [
            example.get("evidence_span_text"),
            example.get("hypothesis_span_text"),
            example.get("action_span_text"),
        ]
        _require(
            isinstance(text_value, str)
            and all(isinstance(item, str) and bool(item) for item in span_texts)
            and all(text_value.count(str(item)) == 1 for item in span_texts),
            "codebook positive example span text is not exact and unique",
        )
        starts = [text_value.index(str(item)) for item in span_texts]
        _require(
            starts == sorted(starts) and len(set(starts)) == 3,
            "codebook positive example spans violate E-before-H-before-A",
        )
    negatives = _sequence(codebook.get("negative_examples"), "negative examples")
    _require(
        {
            str(_mapping(item, "negative example").get("id")) for item in negatives
        }
        == {
            "synthetic-negative-missing-E",
            "synthetic-negative-missing-H",
            "synthetic-negative-missing-A",
            "synthetic-negative-wrong-order",
            "synthetic-negative-H-A-distance",
            "synthetic-negative-implicit-relation",
            "synthetic-negative-tool-arguments-only",
        }
        and all(
            _mapping(item, "negative example").get("has_chain") is False
            for item in negatives
        ),
        "codebook negative examples changed",
    )
    novelty = _sequence(codebook.get("novelty_examples"), "novelty examples")
    novelty_by_id = {
        str(_mapping(item, "novelty example").get("id")): _mapping(
            item, "novelty example"
        ).get("novelty_status")
        for item in novelty
    }
    _require(
        novelty_by_id
        == {
            "synthetic-novelty-novel": "novel",
            "synthetic-novelty-prefix-exposed-exact": "prefix_exposed",
            "synthetic-novelty-prefix-exposed-entailed": "prefix_exposed",
            "synthetic-novelty-ambiguous": "ambiguous",
        },
        "codebook novelty examples changed",
    )
    unknowns = _sequence(codebook.get("unknown_reasons"), "unknown reasons")
    _require(
        [
            _mapping(item, "unknown reason").get("value") for item in unknowns
        ]
        == [
            "frozen_nonprefix_empty_assistant_rendering",
            "completion_semantics_ambiguous",
            "span_or_slot_adjudication_unresolved",
        ]
        and _mapping(
            codebook.get("unknown_record_rule"), "unknown record rule"
        ).get("unknown_is_never_coerced_to_no_chain")
        is True
        and _mapping(
            codebook.get("earliest_chain_example"), "earliest chain example"
        ).get("later_chain_ignored")
        is True,
        "codebook unknown or earliest-chain rule changed",
    )
    earliest = _mapping(
        codebook.get("earliest_chain_example"), "earliest chain example"
    )
    earliest_text = earliest.get("assistant_text")
    earliest_spans = [
        earliest.get("selected_evidence_span_text"),
        earliest.get("selected_hypothesis_span_text"),
        earliest.get("selected_action_span_text"),
    ]
    _require(
        isinstance(earliest_text, str)
        and all(isinstance(item, str) and bool(item) for item in earliest_spans)
        and all(earliest_text.count(str(item)) == 1 for item in earliest_spans)
        and [earliest_text.index(str(item)) for item in earliest_spans]
        == sorted(earliest_text.index(str(item)) for item in earliest_spans),
        "codebook earliest-chain example span selection changed",
    )
    return codebook


def packet_materialization_disclosure(
    config: Mapping[str, Any],
    *,
    annotation_pass: str,
    locked_chain_records_present: bool,
) -> dict[str, Any]:
    """Describe packet production without implying annotation or decoding."""

    config = validate_config(config)
    _require(annotation_pass in PASSES, "packet disclosure pass invalid")
    _require(
        isinstance(locked_chain_records_present, bool),
        "locked-record presence flag must be boolean",
    )
    _require(
        (annotation_pass == "completion_chain" and not locked_chain_records_present)
        or (annotation_pass == "prefix_novelty" and locked_chain_records_present),
        "locked-record presence differs from annotation-pass semantics",
    )
    frozen = config["packet_materialization_scope"]
    return {
        "status": frozen["artifact_status"],
        "scope": {
            "artifact_kind": frozen["artifact_kind"],
            "development_data_only": frozen["development_data_only"],
            "reserved_validation_closed": frozen["reserved_validation_closed"],
            "reserved_validation_accessed": frozen["reserved_validation_accessed"],
        },
        "annotation_execution": {
            "status": "not_run_for_target_pass",
            "annotation_not_run": frozen["annotation_not_run_for_target_pass"],
            "target_annotation_pass": annotation_pass,
            "producer_executes_annotation": False,
            "target_pass_annotation_records_in_artifact": frozen[
                "target_pass_annotation_records_in_artifact"
            ],
            "upstream_locked_chain_records_input_present": (
                locked_chain_records_present
            ),
        },
        "claim_scope": dict(config["claim_scope"]),
    }


def _authenticate_bound_file(record: Mapping[str, Any], label: str) -> Path:
    path_text = record.get("path")
    _require(isinstance(path_text, str) and bool(path_text), f"{label} path invalid")
    path = ROOT / path_text
    lexical_path_preflight((path,))
    canonical_path_preflight(input_paths=(path,), output_paths=())
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    _require(
        stat.st_size == record.get("size_bytes"), f"{label} size changed"
    )
    _require(
        sha256_file(resolved) == record.get("sha256"), f"{label} SHA-256 changed"
    )
    return resolved


def validate_alignment_index(
    value: Any, *, expected_row_count: int, expected_stable_count: int
) -> list[dict[str, Any]]:
    index = _mapping(value, "label-free alignment index")
    _require(
        set(index)
        == {
            "schema_version",
            "kind",
            "status",
            "scope",
            "config",
            "implementation",
            "sources",
            "eligibility_source",
            "row_count",
            "stable_row_count",
            "feature_use",
            "rows",
        },
        "alignment index schema changed",
    )
    _require(
        index.get("schema_version") == 1
        and index.get("kind") == "swe_task_state_v4_label_free_alignment_index"
        and index.get("status") == "passed"
        and index.get("scope") == "grouping_order_and_stability_only_no_labels"
        and index.get("row_count") == expected_row_count
        and index.get("stable_row_count") == expected_stable_count,
        "alignment index identity or counts changed",
    )
    raw_rows = _sequence(index.get("rows"), "alignment rows")
    _require(len(raw_rows) == expected_row_count, "alignment row count changed")
    keys = {
        "global_index",
        "source_id_sha256",
        "task_id_sha256",
        "repository",
        "request_index",
        "stable_feature_eligible",
    }
    seen_sources: set[str] = set()
    last_request_by_task: dict[tuple[str, str], int] = {}
    stable_count = 0
    rows: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_rows):
        row = dict(_mapping(raw, f"alignment row {position}"))
        _require(set(row) == keys, f"alignment row {position} fields changed")
        source = row.get("source_id_sha256")
        task = row.get("task_id_sha256")
        repository = row.get("repository")
        request = row.get("request_index")
        eligible = row.get("stable_feature_eligible")
        _require(
            row.get("global_index") == position
            and _is_sha256(source)
            and _is_sha256(task)
            and isinstance(repository, str)
            and bool(repository)
            and isinstance(request, int)
            and not isinstance(request, bool)
            and request >= 1
            and isinstance(eligible, bool)
            and source not in seen_sources,
            f"alignment row {position} invalid",
        )
        task_key = (repository, str(task))
        previous = last_request_by_task.get(task_key)
        _require(
            previous is None or request == previous + 1,
            f"alignment request order is not consecutive at row {position}",
        )
        last_request_by_task[task_key] = request
        seen_sources.add(str(source))
        stable_count += int(eligible)
        rows.append(row)
    _require(stable_count == expected_stable_count, "alignment stable count changed")
    return rows


def _validate_prompt_row(
    raw: Any, *, position: int, alignment: Mapping[str, Any]
) -> dict[str, Any]:
    row = _mapping(raw, f"prompt row {position}")
    _require(
        set(row) == {"id", "metadata", "score_token_ids", "text", "token_ids"},
        f"prompt row {position} schema changed",
    )
    prompt_id = row.get("id")
    text = row.get("text")
    token_ids = row.get("token_ids")
    metadata = _mapping(row.get("metadata"), f"prompt metadata {position}")
    task = _mapping(metadata.get("task"), f"prompt task {position}")
    selection = _mapping(metadata.get("selection"), f"prompt selection {position}")
    provenance = _mapping(metadata.get("provenance"), f"prompt provenance {position}")
    next_completion = _mapping(
        provenance.get("next_completion"), f"next completion {position}"
    )
    _require(
        isinstance(prompt_id, str)
        and bool(prompt_id)
        and isinstance(text, str)
        and isinstance(token_ids, list)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in token_ids),
        f"prompt row {position} core fields invalid",
    )
    source_hash = sha256_text(prompt_id)
    instance_id = task.get("instance_id")
    repository = task.get("repo")
    request_index = selection.get("task_request_index")
    global_request_index = selection.get("global_request_index")
    _require(
        alignment.get("global_index") == position
        and alignment.get("source_id_sha256") == source_hash
        and isinstance(instance_id, str)
        and bool(instance_id)
        and alignment.get("task_id_sha256") == sha256_text(instance_id)
        and repository == alignment.get("repository")
        and request_index == alignment.get("request_index")
        and global_request_index == position + 1,
        f"prompt row {position} differs from label-free alignment",
    )
    rendered_sha = provenance.get("rendered_prompt_sha256")
    token_sha = provenance.get("token_ids_sha256")
    _require(
        _is_sha256(rendered_sha)
        and sha256_text(text) == rendered_sha
        and _is_sha256(token_sha)
        and sha256_bytes(canonical_json_bytes(token_ids)) == token_sha,
        f"prompt row {position} rendered or token identity changed",
    )
    status = next_completion.get("status")
    _require(
        status
        in {
            "materialized_in_following_request",
            "terminal",
            "truncated",
            "unobserved_after_task_end",
        },
        f"prompt row {position} next-completion status invalid",
    )
    assistant_sha = next_completion.get("assistant_text_sha256")
    if status == "materialized_in_following_request":
        _require(
            _is_sha256(assistant_sha)
            and isinstance(next_completion.get("next_request_global_index"), int)
            and _is_sha256(next_completion.get("next_request_sha256")),
            f"prompt row {position} materialized-completion binding invalid",
        )
    return {
        "position": position,
        "prompt_id": prompt_id,
        "source_id_sha256": source_hash,
        "task_id": instance_id,
        "repository": repository,
        "request_index": request_index,
        "global_request_index": global_request_index,
        "stable_feature_eligible": bool(alignment["stable_feature_eligible"]),
        "text": text,
        "text_sha256": rendered_sha,
        "raw_request_sha256": provenance.get("raw_request_sha256"),
        "next_completion": dict(next_completion),
    }


def _validate_tool_call_suffix(value: str) -> list[dict[str, Any]]:
    """Validate a pure sequence of rendered tool-call blocks."""

    removed: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(value):
        while cursor < len(value) and value[cursor] in "\r\n\t ":
            cursor += 1
        if cursor == len(value):
            break
        _require(
            value.startswith(TOOL_CALL_OPEN, cursor),
            "assistant rendered tail contains prose after the first tool-call block",
        )
        start = cursor
        body_start = cursor + len(TOOL_CALL_OPEN)
        nested = value.find(TOOL_CALL_OPEN, body_start)
        close = value.find(TOOL_CALL_CLOSE, body_start)
        _require(
            close >= 0 and (nested < 0 or close < nested),
            "assistant rendered tool-call block is unbalanced or nested",
        )
        cursor = close + len(TOOL_CALL_CLOSE)
        removed.append(
            {
                "kind": "assistant_tool_call",
                "relative_char_start": start,
                "relative_char_end": cursor,
                "sha256": sha256_text(value[start:cursor]),
            }
        )
    _require(bool(removed), "assistant tool-call marker yielded no complete block")
    return removed


def reconstruct_materialized_completion(
    current: Mapping[str, Any], following: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate assistant text from an exact rendered t -> t+1 extension."""

    current_text = current.get("text")
    following_text = following.get("text")
    _require(
        isinstance(current_text, str) and isinstance(following_text, str),
        "rendered prompt texts are invalid",
    )
    next_completion = _mapping(
        current.get("next_completion"), "current next-completion binding"
    )
    _require(
        next_completion.get("status") == "materialized_in_following_request",
        "completion reconstruction requires a materialized current row",
    )
    same_task = current.get("task_id") == following.get("task_id")
    immediate_request = following.get("request_index") == current.get("request_index") + 1
    exact_next_global = (
        following.get("global_request_index")
        == next_completion.get("next_request_global_index")
    )
    exact_next_request = (
        following.get("raw_request_sha256") == next_completion.get("next_request_sha256")
    )
    _require(
        same_task and immediate_request and exact_next_global and exact_next_request,
        "materialized completion does not bind the exact immediate following request",
    )
    if not following_text.startswith(current_text):
        raise AnnotationPacketError(
            "exact t->t+1 rendered prefix extension failed for "
            f"{current.get('source_id_sha256')}; config correction required "
            "(no thinking-boundary normalization is authorized)"
        )
    _require(
        current_text.endswith(GENERATION_BOUNDARY),
        "current prompt does not end at the frozen thinking generation boundary",
    )
    extension = following_text[len(current_text) :]
    message_end = extension.find(MESSAGE_END)
    _require(message_end >= 0, "rendered extension has no assistant-message terminator")
    assistant_rendered = extension[:message_end]
    _require(
        assistant_rendered.count(THINK_CLOSE) == 1,
        "assistant rendered completion must contain exactly one thinking close",
    )
    thinking_close = assistant_rendered.index(THINK_CLOSE)
    rendered_reasoning = assistant_rendered[:thinking_close]
    _require(
        rendered_reasoning.endswith("\n"),
        "assistant reasoning rendering lacks the exact template separator",
    )
    reasoning = rendered_reasoning[:-1]
    after_thinking = assistant_rendered[thinking_close + len(THINK_CLOSE) :]
    tool_start = after_thinking.find(TOOL_CALL_OPEN)
    if tool_start < 0:
        content = after_thinking
        tool_suffix = ""
        excluded_calls: list[dict[str, Any]] = []
    else:
        content = after_thinking[:tool_start]
        tool_suffix = after_thinking[tool_start:]
        excluded_calls = _validate_tool_call_suffix(tool_suffix)
    assistant_text = (reasoning + "\n" if reasoning else "") + content
    _require(
        TOOL_CALL_OPEN not in assistant_text
        and TOOL_CALL_CLOSE not in assistant_text
        and TOOL_RESPONSE_OPEN not in assistant_text
        and TOOL_RESPONSE_CLOSE not in assistant_text,
        "annotator completion text contains a rendered tool block",
    )
    expected_sha = next_completion.get("assistant_text_sha256")
    observed_sha = sha256_text(assistant_text)
    _require(
        observed_sha == expected_sha,
        "reconstructed assistant text does not match its pinned SHA-256; "
        "config correction required",
    )
    prefix_end = len(current_text)
    assistant_end = prefix_end + message_end
    reasoning_start = prefix_end
    reasoning_end = prefix_end + thinking_close
    content_start = prefix_end + thinking_close + len(THINK_CLOSE)
    content_end = content_start + len(content)
    excluded_ranges = []
    if tool_suffix:
        suffix_start = content_end
        for item in excluded_calls:
            excluded_ranges.append(
                {
                    "kind": item["kind"],
                    "rendered_char_start": suffix_start
                    + int(item["relative_char_start"]),
                    "rendered_char_end": suffix_start
                    + int(item["relative_char_end"]),
                    "sha256": item["sha256"],
                }
            )
    return {
        "assistant_text": assistant_text,
        "assistant_text_sha256": observed_sha,
        "assistant_text_char_start": 0,
        "assistant_text_char_end": len(assistant_text),
        "current_prefix_sha256": current.get("text_sha256"),
        "current_prefix_char_start": 0,
        "current_prefix_char_end": prefix_end,
        "following_prefix_sha256": following.get("text_sha256"),
        "following_prefix_char_start": 0,
        "following_prefix_char_end": len(following_text),
        "rendered_extension_sha256": sha256_text(extension),
        "rendered_extension_char_start": prefix_end,
        "rendered_extension_char_end": len(following_text),
        "assistant_envelope_rendered_char_start": prefix_end,
        "assistant_envelope_rendered_char_end": assistant_end,
        "reasoning_rendered_char_start": reasoning_start,
        "reasoning_rendered_char_end": reasoning_end,
        "content_rendered_char_start": content_start,
        "content_rendered_char_end": content_end,
        "excluded_rendered_ranges": excluded_ranges,
    }


def validate_frozen_unknown_exclusion(
    current: Mapping[str, Any],
    following: Mapping[str, Any],
    exclusion: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the sole frozen non-prefix row without normalizing it."""

    current_text = current.get("text")
    following_text = following.get("text")
    next_completion = _mapping(
        current.get("next_completion"), "frozen unknown next-completion binding"
    )
    _require(
        isinstance(current_text, str)
        and isinstance(following_text, str)
        and current.get("source_id_sha256") == exclusion.get("source_id_sha256")
        and current.get("text_sha256")
        == exclusion.get("current_rendered_prefix_sha256")
        and following.get("source_id_sha256")
        == exclusion.get("following_source_id_sha256")
        and following.get("text_sha256")
        == exclusion.get("following_rendered_prefix_sha256")
        and following.get("raw_request_sha256")
        == exclusion.get("following_raw_request_sha256")
        and next_completion.get("next_request_sha256")
        == exclusion.get("following_raw_request_sha256")
        and next_completion.get("assistant_text_sha256")
        == exclusion.get("materialized_completion_sha256")
        and next_completion.get("status") == "materialized_in_following_request"
        and current.get("task_id") == following.get("task_id")
        and following.get("request_index") == current.get("request_index") + 1
        and following.get("global_request_index")
        == next_completion.get("next_request_global_index")
        and len(current_text) == exclusion.get("current_prefix_char_count")
        and len(following_text) == exclusion.get("following_prefix_char_count"),
        "frozen rendered-prefix unknown exclusion binding changed",
    )
    common = 0
    for left, right in zip(current_text, following_text):
        if left != right:
            break
        common += 1
    dropped = current_text[common:]
    following_extension = following_text[common:]
    _require(
        common == exclusion.get("common_prefix_char_count")
        and sha256_text(dropped) == exclusion.get("dropped_current_suffix_sha256")
        and sha256_text(following_extension)
        == exclusion.get("following_extension_sha256")
        and current_text[:common].endswith("<t")
        and dropped == "hink>\n"
        and following_extension.startswith("ool_call>\n")
        and not following_text.startswith(current_text)
        and next_completion.get("assistant_text_sha256") == sha256_text(""),
        "frozen rendered-prefix unknown exclusion geometry changed",
    )
    return {
        "source_id_sha256": exclusion["source_id_sha256"],
        "materialized_completion_sha256": exclusion[
            "materialized_completion_sha256"
        ],
        "annotation_status": "unknown",
        "unknown_reason": exclusion["unknown_reason"],
    }


def strip_rendered_tool_blocks(value: str) -> dict[str, Any]:
    """Remove tool-call and tool-response blocks from an annotator prefix."""

    _require(isinstance(value, str), "prefix annotator source must be text")
    tag_pairs = {
        TOOL_CALL_OPEN: (TOOL_CALL_CLOSE, "tool_call"),
        TOOL_RESPONSE_OPEN: (TOOL_RESPONSE_CLOSE, "tool_response"),
    }
    cursor = 0
    pieces: list[str] = []
    removed: list[dict[str, Any]] = []
    while cursor < len(value):
        candidates = [
            (value.find(open_tag, cursor), open_tag)
            for open_tag in tag_pairs
            if value.find(open_tag, cursor) >= 0
        ]
        if not candidates:
            pieces.append(value[cursor:])
            break
        start, open_tag = min(candidates)
        close_tag, kind = tag_pairs[open_tag]
        pieces.append(value[cursor:start])
        body_start = start + len(open_tag)
        nested_candidates = [
            value.find(candidate_open, body_start) for candidate_open in tag_pairs
        ]
        nested_candidates = [position for position in nested_candidates if position >= 0]
        close = value.find(close_tag, body_start)
        _require(
            close >= 0 and (not nested_candidates or close < min(nested_candidates)),
            "rendered prefix contains an unbalanced or nested tool block",
        )
        end = close + len(close_tag)
        removed.append(
            {
                "kind": kind,
                "source_char_start": start,
                "source_char_end": end,
                "sha256": sha256_text(value[start:end]),
            }
        )
        cursor = end
    text = "".join(pieces)
    _require(
        all(marker not in text for marker in (*tag_pairs, *(pair[0] for pair in tag_pairs.values()))),
        "tool-block marker survived prefix sanitization",
    )
    return {
        "text": text,
        "sha256": sha256_text(text),
        "char_start": 0,
        "char_end": len(text),
        "removed_ranges": removed,
    }


def blind_shard_assignment(
    source_id_sha256: str, *, annotation_pass: str, shard_count: int
) -> dict[str, int]:
    _require(_is_sha256(source_id_sha256), "source identity SHA-256 invalid")
    _require(annotation_pass in PASSES, "annotation pass invalid")
    _require(
        isinstance(shard_count, int)
        and not isinstance(shard_count, bool)
        and 2 <= shard_count <= 128,
        "blind shard count must be in [2, 128]",
    )

    def draw(lane: str) -> int:
        payload = (
            f"{CONFIG_CANONICAL_SHA256}\0{annotation_pass}\0{lane}\0"
            f"{source_id_sha256}"
        )
        return int(sha256_text(payload)[:16], 16) % shard_count

    lane_a = draw("independent-a")
    lane_b = draw("independent-b")
    if lane_b == lane_a:
        offset = 1 + int(
            sha256_text(f"distinct\0{annotation_pass}\0{source_id_sha256}")[:16], 16
        ) % (shard_count - 1)
        lane_b = (lane_a + offset) % shard_count
    return {"independent_a": lane_a, "independent_b": lane_b}


def _packet_id(source_id_sha256: str, annotation_pass: str) -> str:
    return sha256_text(
        f"{CONFIG_CANONICAL_SHA256}\0packet\0{annotation_pass}\0{source_id_sha256}"
    )


def make_completion_packet(
    *,
    current: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    shard_count: int,
) -> dict[str, Any]:
    source = current.get("source_id_sha256")
    _require(_is_sha256(source), "completion packet source identity invalid")
    text = reconstruction.get("assistant_text")
    _require(isinstance(text, str), "completion packet assistant text invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": COMPLETION_PACKET_KIND,
        "annotation_pass": "completion_chain",
        "packet_id_sha256": _packet_id(str(source), "completion_chain"),
        "source_id_sha256": source,
        "blind_shards": blind_shard_assignment(
            str(source), annotation_pass="completion_chain", shard_count=shard_count
        ),
        "materialized_assistant_text": {
            "text": text,
            "sha256": reconstruction["assistant_text_sha256"],
            "char_start": 0,
            "char_end": len(text),
        },
        "authenticated_boundaries": {
            key: value
            for key, value in reconstruction.items()
            if key != "assistant_text"
        },
        "annotator_visibility": {
            "complete_prefix_text_present": False,
            "assistant_tool_arguments_present": False,
            "tool_results_present": False,
            "repository_or_task_identity_present": False,
            "model_features_present": False,
        },
    }


def _span(
    value: Any,
    *,
    label: str,
    completion_text: str | None,
) -> dict[str, Any]:
    span = dict(_mapping(value, label))
    _require(
        set(span) == {"start", "end", "text_sha256"},
        f"{label} fields invalid",
    )
    start = span.get("start")
    end = span.get("end")
    digest = span.get("text_sha256")
    _require(
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start < end
        and _is_sha256(digest),
        f"{label} bounds or hash invalid",
    )
    if completion_text is not None:
        _require(end <= len(completion_text), f"{label} exceeds completion text")
        _require(
            sha256_text(completion_text[start:end]) == digest,
            f"{label} exact text hash differs from completion text",
        )
    return span


def validate_annotation_record(
    value: Any,
    *,
    config: Mapping[str, Any],
    stage: str = "final",
    completion_text: str | None = None,
) -> dict[str, Any]:
    """Validate one completion-stage or final annotation record."""

    validate_config(config)
    _require(stage in {"completion", "final"}, "annotation record stage invalid")
    record = dict(_mapping(value, "annotation record"))
    required = config["annotation_record"]["required_fields"]
    _require(
        set(record) == set(required),
        "annotation record fields differ from the frozen required registry",
    )
    _require(
        _is_sha256(record.get("source_id_sha256"))
        and _is_sha256(record.get("materialized_completion_sha256"))
        and _is_sha256(record.get("annotator_id_sha256"))
        and _is_sha256(
            record.get("annotator_prompt_or_model_identity_sha256")
        )
        and _is_sha256(record.get("codebook_sha256")),
        "annotation record identity hash invalid",
    )
    codebook_contract = config["annotation_codebook_contract"]
    _require(
        record.get("codebook_sha256") == codebook_contract["sha256"]
        and record.get("annotator_prompt_or_model_identity_sha256")
        == codebook_contract["annotator_prompt_or_model_identity_sha256"],
        "annotation record codebook or annotator identity digest differs from frozen binding",
    )
    if completion_text is not None:
        _require(
            sha256_text(completion_text)
            == record["materialized_completion_sha256"],
            "annotation record completion hash differs from authenticated text",
        )
    status = record.get("annotation_status")
    unknown_reason = record.get("unknown_reason")
    _require(
        status in config["annotation_record"]["annotation_statuses"],
        "annotation status is outside the frozen registry",
    )
    has_chain = record.get("has_chain")
    spans = ("evidence_span", "hypothesis_span", "action_span")
    slots = (
        "evidence_kind",
        "belief_edge",
        "hypothesis_domain",
        "action_intent",
    )
    if status == "unknown":
        _require(
            unknown_reason in config["annotation_record"]["unknown_reasons"]
            and has_chain is None
            and all(record.get(name) is None for name in (*spans, *slots))
            and record.get("novelty_status") is None
            and record.get("exact_signature") is None,
            "unknown annotation requires a frozen reason and null chain fields",
        )
        return record
    _require(
        status == "available" and unknown_reason is None,
        "available annotation must have a null unknown reason",
    )
    _require(isinstance(has_chain, bool), "has_chain must be boolean")
    if not has_chain:
        _require(
            all(record.get(name) is None for name in (*spans, *slots))
            and record.get("novelty_status") is None
            and record.get("exact_signature") is None,
            "no-chain spans, slots, novelty, and signature must all be null",
        )
        return record

    evidence = _span(
        record.get("evidence_span"),
        label="evidence span",
        completion_text=completion_text,
    )
    hypothesis = _span(
        record.get("hypothesis_span"),
        label="hypothesis span",
        completion_text=completion_text,
    )
    action = _span(
        record.get("action_span"),
        label="action span",
        completion_text=completion_text,
    )
    _require(
        evidence["start"] < hypothesis["start"] < action["start"]
        and evidence["end"] <= hypothesis["start"]
        and hypothesis["end"] <= action["start"],
        "positive annotation spans violate E-before-H-before-A",
    )
    ontology = config["ontology"]
    _require(
        record.get("evidence_kind") in ontology["evidence_kind"]
        and record.get("belief_edge") in config["chain_definition"]["belief_edges"]
        and record.get("hypothesis_domain") in ontology["hypothesis_domain"]
        and record.get("action_intent") in ontology["action_intent"],
        "positive annotation slot is outside the frozen ontology",
    )
    signature = ">".join(
        [
            record["evidence_kind"],
            record["belief_edge"],
            record["hypothesis_domain"],
            "motivates",
            record["action_intent"],
        ]
    )
    _require(
        record.get("exact_signature") == signature,
        "annotation exact signature disagrees with its slots",
    )
    novelty = record.get("novelty_status")
    if stage == "completion":
        _require(
            novelty is None,
            "completion-pass record must not contain a prefix-novelty decision",
        )
    else:
        _require(
            novelty in config["prefix_novelty"]["statuses"],
            "final positive record lacks a frozen prefix-novelty status",
        )
    return record


def make_prefix_novelty_packet(
    *,
    current: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    locked_record: Mapping[str, Any],
    config: Mapping[str, Any],
    shard_count: int,
) -> dict[str, Any] | None:
    text = reconstruction.get("assistant_text")
    _require(isinstance(text, str), "prefix packet completion text invalid")
    record = validate_annotation_record(
        locked_record,
        config=config,
        stage="completion",
        completion_text=text,
    )
    if not record["has_chain"]:
        return None
    hypothesis = record["hypothesis_span"]
    hypothesis_text = text[hypothesis["start"] : hypothesis["end"]]
    prefix = current.get("text")
    _require(isinstance(prefix, str), "prefix packet source text invalid")
    sanitized = strip_rendered_tool_blocks(prefix)
    source = str(current["source_id_sha256"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PREFIX_PACKET_KIND,
        "annotation_pass": "prefix_novelty",
        "packet_id_sha256": _packet_id(source, "prefix_novelty"),
        "source_id_sha256": source,
        "blind_shards": blind_shard_assignment(
            source, annotation_pass="prefix_novelty", shard_count=shard_count
        ),
        "locked_hypothesis": {
            "text": hypothesis_text,
            "sha256": hypothesis["text_sha256"],
            "completion_char_start": hypothesis["start"],
            "completion_char_end": hypothesis["end"],
            "materialized_completion_sha256": reconstruction[
                "assistant_text_sha256"
            ],
        },
        "authenticated_prefix": {
            "source_sha256": current["text_sha256"],
            "source_char_start": 0,
            "source_char_end": len(prefix),
            "annotator_text": sanitized["text"],
            "annotator_text_sha256": sanitized["sha256"],
            "annotator_char_start": 0,
            "annotator_char_end": sanitized["char_end"],
            "removed_ranges": sanitized["removed_ranges"],
        },
        "annotator_visibility": {
            "completion_chain_slots_present": False,
            "assistant_tool_arguments_present": False,
            "tool_results_present": False,
            "repository_or_task_identity_present": False,
            "model_features_present": False,
        },
    }


def _iter_prompt_rows(path: Path) -> Iterator[Any]:
    try:
        import ijson
    except ImportError as error:
        raise AnnotationPacketError(
            "ijson is required for the authenticated prompt stream"
        ) from error
    try:
        with path.open("rb") as handle:
            yield from ijson.items(handle, "item")
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise AnnotationPacketError(f"prompt stream failed: {error}") from error


def load_annotation_records_jsonl(
    path: Path,
    *,
    expected_sha256: str,
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    _require(_is_sha256(expected_sha256), "locked record SHA-256 invalid")
    _require(sha256_file(path) == expected_sha256, "locked record file hash changed")
    records: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                _require(bool(line.strip()), f"blank locked-record line {line_number}")
                try:
                    raw = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
                except json.JSONDecodeError as error:
                    raise AnnotationPacketError(
                        f"invalid locked-record JSON at line {line_number}: {error}"
                    ) from error
                record = validate_annotation_record(
                    raw, config=config, stage="completion"
                )
                source = record["source_id_sha256"]
                _require(source not in records, "duplicate locked annotation source")
                records[source] = record
    except (OSError, UnicodeDecodeError) as error:
        raise AnnotationPacketError(f"cannot read locked annotation records: {error}") from error
    return records


def _write_packet_line(handle: TextIO, packet: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(
            packet,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def export_packets(
    *,
    config: Mapping[str, Any],
    prompt_path: Path,
    alignment_rows: Sequence[Mapping[str, Any]],
    packet_handle: TextIO,
    annotation_pass: str,
    shard_count: int,
    locked_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stream authenticated rows and write packets to an already-temporary file."""

    config = validate_config(config)
    _require(annotation_pass in PASSES, "packet export pass invalid")
    if annotation_pass == "completion_chain":
        _require(locked_records is None, "completion pass cannot consume annotations")
    else:
        _require(locked_records is not None, "prefix pass requires locked chain records")
    counts = {
        "all_rows": 0,
        "stable_rows": 0,
        "stable_materialized_rows": 0,
        "packets": 0,
        "available_no_chain_rows_omitted_from_prefix_pass": 0,
        "annotation_unknown_rows_omitted_from_prefix_pass": 0,
        "frozen_unknown_exclusions": 0,
    }
    exclusions = {
        item["source_id_sha256"]: item
        for item in config["source_contract"]["frozen_unknown_exclusions"]
    }
    observed_exclusions: set[str] = set()
    materialized_sources: set[str] = set()
    previous: dict[str, Any] | None = None
    for position, raw in enumerate(_iter_prompt_rows(prompt_path)):
        _require(position < len(alignment_rows), "prompt stream exceeds alignment")
        current = _validate_prompt_row(
            raw, position=position, alignment=alignment_rows[position]
        )
        counts["all_rows"] += 1
        counts["stable_rows"] += int(current["stable_feature_eligible"])
        if previous is not None:
            next_binding = previous["next_completion"]
            if (
                previous["stable_feature_eligible"]
                and next_binding["status"] == "materialized_in_following_request"
            ):
                counts["stable_materialized_rows"] += 1
                source = previous["source_id_sha256"]
                materialized_sources.add(source)
                exclusion = exclusions.get(source)
                if exclusion is not None:
                    frozen = validate_frozen_unknown_exclusion(
                        previous, current, exclusion
                    )
                    observed_exclusions.add(source)
                    counts["frozen_unknown_exclusions"] += 1
                    packet = None
                    if annotation_pass == "prefix_novelty":
                        assert locked_records is not None
                        _require(
                            source in locked_records,
                            "locked records omit the frozen unknown source",
                        )
                        record = validate_annotation_record(
                            locked_records[source], config=config, stage="completion"
                        )
                        _require(
                            record["annotation_status"] == "unknown"
                            and record["unknown_reason"] == frozen["unknown_reason"]
                            and record["materialized_completion_sha256"]
                            == frozen["materialized_completion_sha256"],
                            "locked record differs from the frozen unknown exclusion",
                        )
                        counts[
                            "annotation_unknown_rows_omitted_from_prefix_pass"
                        ] += 1
                else:
                    reconstruction = reconstruct_materialized_completion(
                        previous, current
                    )
                    if annotation_pass == "completion_chain":
                        packet = make_completion_packet(
                            current=previous,
                            reconstruction=reconstruction,
                            shard_count=shard_count,
                        )
                    else:
                        assert locked_records is not None
                        _require(
                            source in locked_records,
                            "locked records omit a stable materialized source",
                        )
                        record = locked_records[source]
                        packet = make_prefix_novelty_packet(
                            current=previous,
                            reconstruction=reconstruction,
                            locked_record=record,
                            config=config,
                            shard_count=shard_count,
                        )
                        if packet is None:
                            if record["annotation_status"] == "unknown":
                                counts[
                                    "annotation_unknown_rows_omitted_from_prefix_pass"
                                ] += 1
                            else:
                                counts[
                                    "available_no_chain_rows_omitted_from_prefix_pass"
                                ] += 1
                if packet is not None:
                    _write_packet_line(packet_handle, packet)
                    counts["packets"] += 1
        previous = current
    _require(
        counts["all_rows"] == len(alignment_rows),
        "prompt stream row count differs from alignment",
    )
    if previous is not None:
        _require(
            not (
                previous["stable_feature_eligible"]
                and previous["next_completion"]["status"]
                == "materialized_in_following_request"
            ),
            "final stable materialized row has no following prompt",
        )
    _require(
        counts["stable_rows"]
        == config["source_contract"]["label_free_alignment_index"][
            "stable_row_count"
        ],
        "streamed stable row count changed",
    )
    _require(
        observed_exclusions == set(exclusions),
        "frozen unknown exclusions were not encountered exactly once",
    )
    if locked_records is not None:
        _require(
            set(locked_records) == materialized_sources,
            "locked records do not exactly cover stable materialized rows",
        )
    return counts


def _temporary_peer(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def _require_output_pair(packet_path: Path, manifest_path: Path) -> None:
    _require(
        packet_path != manifest_path
        and packet_path.parent == manifest_path.parent
        and packet_path.suffix == ".jsonl"
        and manifest_path.suffix == ".json",
        "packet and manifest outputs must be distinct .jsonl/.json peers",
    )
    _require(
        not packet_path.exists()
        and not packet_path.is_symlink()
        and not manifest_path.exists()
        and not manifest_path.is_symlink(),
        "refusing to overwrite annotation packet output",
    )


def _write_manifest(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--pass", dest="annotation_pass", choices=PASSES, required=True)
    parser.add_argument("--output-packets", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--locked-chain-records", type=Path)
    parser.add_argument("--expected-locked-chain-records-sha256")
    args = parser.parse_args(argv)

    # Intentionally first: no stat/hash/resolve may precede this lexical gate.
    lexical_path_preflight(
        (
            args.config,
            args.output_packets,
            args.output_manifest,
            args.locked_chain_records,
        )
    )
    canonical_path_preflight(
        input_paths=(args.config, args.locked_chain_records),
        output_paths=(args.output_packets, args.output_manifest),
    )
    config_path = args.config.resolve(strict=True)
    _require(
        config_path == CONFIG_PATH
        and sha256_file(config_path) == CONFIG_FILE_SHA256,
        "only the exact frozen epistemic-chain annotation config is allowed",
    )
    config = validate_config(load_json_strict(config_path))
    source = config["source_contract"]
    codebook_contract = config["annotation_codebook_contract"]
    prompt_unresolved = ROOT / source["development_prompt_bundle"]["path"]
    alignment_unresolved = ROOT / source["label_free_alignment_index"]["path"]
    codebook_unresolved = ROOT / codebook_contract["path"]
    lexical_path_preflight(
        (prompt_unresolved, alignment_unresolved, codebook_unresolved)
    )
    canonical_path_preflight(
        input_paths=(prompt_unresolved, alignment_unresolved, codebook_unresolved),
        output_paths=(),
    )
    prompt_path = _authenticate_bound_file(
        source["development_prompt_bundle"], "development prompt bundle"
    )
    alignment_path = _authenticate_bound_file(
        source["label_free_alignment_index"], "label-free alignment index"
    )
    codebook_path = _authenticate_bound_file(
        codebook_contract, "epistemic-chain annotation codebook"
    )
    codebook = validate_codebook(load_json_strict(codebook_path))
    _require(
        codebook["schema_version"] == codebook_contract["schema_version"]
        and codebook["kind"] == codebook_contract["kind"]
        and codebook["annotator_prompt_contract"][
            "prompt_or_model_identity_sha256"
        ]
        == codebook_contract["annotator_prompt_or_model_identity_sha256"],
        "authenticated annotation codebook schema or prompt binding changed",
    )
    alignment_rows = validate_alignment_index(
        load_json_strict(alignment_path),
        expected_row_count=source["development_prompt_bundle"]["row_count"],
        expected_stable_count=source["label_free_alignment_index"][
            "stable_row_count"
        ],
    )
    if args.annotation_pass == "completion_chain":
        _require(
            args.locked_chain_records is None
            and args.expected_locked_chain_records_sha256 is None,
            "completion pass cannot accept locked chain records",
        )
        locked_records = None
        locked_binding = None
    else:
        _require(
            args.locked_chain_records is not None
            and _is_sha256(args.expected_locked_chain_records_sha256),
            "prefix pass requires an explicitly hashed locked chain-record file",
        )
        assert args.locked_chain_records is not None
        locked_path = args.locked_chain_records.resolve(strict=True)
        locked_records = load_annotation_records_jsonl(
            locked_path,
            expected_sha256=str(args.expected_locked_chain_records_sha256),
            config=config,
        )
        locked_binding = {
            "path": os.fspath(locked_path),
            "sha256": args.expected_locked_chain_records_sha256,
        }
    packet_path = args.output_packets.resolve(strict=False)
    manifest_path = args.output_manifest.resolve(strict=False)
    _require_output_pair(packet_path, manifest_path)
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_packet = _temporary_peer(packet_path)
    temporary_manifest = _temporary_peer(manifest_path)
    _require(
        not temporary_packet.exists()
        and not temporary_packet.is_symlink()
        and not temporary_manifest.exists()
        and not temporary_manifest.is_symlink(),
        "annotation temporary output already exists",
    )
    authenticated = (
        config_path,
        SCRIPT_PATH,
        prompt_path,
        alignment_path,
        codebook_path,
    )
    pre_hashes = {os.fspath(path): sha256_file(path) for path in authenticated}
    try:
        with temporary_packet.open("x", encoding="utf-8") as handle:
            counts = export_packets(
                config=config,
                prompt_path=prompt_path,
                alignment_rows=alignment_rows,
                packet_handle=handle,
                annotation_pass=args.annotation_pass,
                shard_count=args.shard_count,
                locked_records=locked_records,
            )
            handle.flush()
            os.fsync(handle.fileno())
        for path_text, expected in pre_hashes.items():
            _require(
                sha256_file(Path(path_text)) == expected,
                f"authenticated annotation input changed during export: {path_text}",
            )
        packet_sha = sha256_file(temporary_packet)
        disclosure = packet_materialization_disclosure(
            config,
            annotation_pass=args.annotation_pass,
            locked_chain_records_present=locked_binding is not None,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": PACKET_MANIFEST_KIND,
            "status": disclosure["status"],
            "annotation_pass": args.annotation_pass,
            "scope": disclosure["scope"],
            "annotation_execution": disclosure["annotation_execution"],
            "claim_scope": disclosure["claim_scope"],
            "config": {
                "path": str(CONFIG_PATH.relative_to(ROOT)),
                "sha256": CONFIG_FILE_SHA256,
            },
            "implementation": {
                "path": str(SCRIPT_PATH.relative_to(ROOT)),
                "sha256": sha256_file(SCRIPT_PATH),
            },
            "inputs": {
                "development_prompt_bundle": {
                    "path": str(prompt_path),
                    "sha256": source["development_prompt_bundle"]["sha256"],
                },
                "label_free_alignment_index": {
                    "path": str(alignment_path),
                    "sha256": source["label_free_alignment_index"]["sha256"],
                },
                "epistemic_chain_annotation_codebook": {
                    "path": str(codebook_path),
                    "sha256": codebook_contract["sha256"],
                    "size_bytes": codebook_contract["size_bytes"],
                    "schema_version": codebook_contract["schema_version"],
                    "kind": codebook_contract["kind"],
                    "annotator_prompt_or_model_identity_sha256": (
                        codebook_contract[
                            "annotator_prompt_or_model_identity_sha256"
                        ]
                    ),
                },
                "locked_chain_records": locked_binding,
            },
            "packets": {
                "path": packet_path.name,
                "sha256": packet_sha,
                "format": "strict_json_lines",
                "count": counts["packets"],
            },
            "counts": counts,
            "frozen_unknown_exclusions": source["frozen_unknown_exclusions"],
            "blinding": {
                "shard_count": args.shard_count,
                "two_independent_distinct_shards": True,
                "repository_task_and_request_indices_omitted": True,
            },
            "boundary": {
                "exact_rendered_prefix_extension_required_for_all_nonexcluded_rows": True,
                "frozen_nonprefix_rows_normalized": False,
                "assistant_text_sha256_required": True,
                "assistant_tool_arguments_excluded": True,
                "tool_results_excluded": True,
                "model_features_excluded": True,
                "free_form_private_reasoning_claimed": False,
            },
        }
        _write_manifest(temporary_manifest, manifest)
        os.replace(temporary_packet, packet_path)
        try:
            os.replace(temporary_manifest, manifest_path)
        except BaseException:
            if packet_path.exists() and not packet_path.is_symlink():
                packet_path.unlink()
            raise
    finally:
        for temporary in (temporary_packet, temporary_manifest):
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
    print(
        f"wrote {counts['packets']} {args.annotation_pass} packets to {packet_path}"
    )
    return 0


__all__ = [
    "AnnotationPacketError",
    "ANNOTATOR_PROMPT_OR_MODEL_IDENTITY_SHA256",
    "CODEBOOK_FILE_SHA256",
    "CODEBOOK_PATH",
    "CONFIG_PATH",
    "blind_shard_assignment",
    "canonical_json_bytes",
    "export_packets",
    "lexical_path_preflight",
    "load_annotation_records_jsonl",
    "make_completion_packet",
    "make_prefix_novelty_packet",
    "packet_materialization_disclosure",
    "reconstruct_materialized_completion",
    "sha256_file",
    "sha256_text",
    "strip_rendered_tool_blocks",
    "validate_alignment_index",
    "validate_annotation_record",
    "validate_codebook",
    "validate_config",
    "validate_frozen_unknown_exclusion",
]


if __name__ == "__main__":
    raise SystemExit(main())
