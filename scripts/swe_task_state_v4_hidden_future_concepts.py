#!/usr/bin/env python3
"""Audit supplied hidden-future identifier surface forms with frozen readouts.

The candidate vocabulary is materialized independently from completed agent
artifacts and is retrospective.  At each eligible final-prompt boundary this
program verifies that the target and every retained foil surface form are
absent from the complete model-visible prefix, then ranks the neutral supplied
candidate set using only frozen ordinary-logit or public-Jacobian token scores.
The target identity is introduced only after ranking, in the evaluation seam.
Any sentence framing or proposition relation is supplied, never decoded.

This is intentionally a bounded-memory, development-only audit.  It never
opens reserved validation, benchmark gold, completions as features, private
reasoning, or emotion labels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence
import unicodedata

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/swe_task_state_v4_hidden_future_concepts.json"
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_SHA256 = "978d231975b4a939c7dc8a9a4a90ca84ca963eb007a95f167f365c146fefa302"
SCHEMA_VERSION = 1
REPORT_KIND = "swe_task_state_v4_hidden_future_identifier_surface_ranking_report"
FORBIDDEN_PATH_FRAGMENTS = ("reserved", "validation")
METHODS = ("ordinary_logit", "public_jacobian")
LAYERS = tuple(range(24, 48))
OPEN_VOCABULARY_TOP_K = (10, 100, 1000, 10000)
VOCABULARY_SIZE = 248320
CANDIDATE_METRICS = (
    "top1_accuracy",
    "mean_reciprocal_rank",
    "negative_log_likelihood",
    "multiclass_brier",
)
OPEN_VOCABULARY_METRICS = (
    "rank_utility",
    *(f"top_k_layer_fraction@{value}" for value in OPEN_VOCABULARY_TOP_K),
)
ALL_METRICS = (*CANDIDATE_METRICS, *OPEN_VOCABULARY_METRICS)

MANDATORY_LIMITATIONS = (
    "The independently materialized target identifier surface form and every retained foil identifier surface form are absent from the complete JSONL-visible prefix at the scored boundary, but the fixed candidate set is derived retrospectively from the completed future agent trajectory.",
    "This is a fixed-vocabulary multiple-choice identifier-surface ranking audit, not concept decoding, proposition decoding, open-ended sentence generation, COT or COT-like decoding, private chain-of-thought recovery, hidden reasoning, intent, or understanding.",
    "Any proposition relation or semantic framing is supplied by the frozen future-support contract and fixed renderer; the readout only ranks supplied identifier token forms and decodes neither a proposition nor a semantic concept chain.",
    "No emotion, felt confidence, doubt, or stress label is fitted or inferred.",
    "Public-J is a deterministic transformation of the same residual state; comparison with ordinary logit measures representation, not access to additional hidden information.",
    "Development-only point estimates and unadjusted clustered intervals are not untouched confirmation or operational reliability evidence.",
)

EXPECTED_INPUTS = {
    "development_prompts": {
        "path": ".cache/swe_state_interpreter_v3_development/prompts.json",
        "sha256": "17f664b3029220458ff62b8e80a90ec5e796f0372217f02b727d6589df38d3d0",
        "size_bytes": 803705234,
    },
    "development_public_report": {
        "path": ".cache/swe_state_interpreter_v3_development/replay/public-report.json",
        "sha256": "7c943132163749f69bd35e4fa2e52bcfee2318fe349fa77603324a37ffaabe46",
        "size_bytes": 1241239850,
    },
    "label_free_alignment_index": {
        "path": ".cache/swe_task_state_v4_raw_capture/n60-final/alignment-index-v4.json",
        "sha256": "c0d9e4bd6ad8f6962cf58776441ec4ef042eebbbae8a9959b4149e605c9ae38e",
        "size_bytes": 567392,
    },
    "v3_protocol": {
        "path": "configs/swe_task_state_interpreter_v3.json",
        "sha256": "9d8b0a7d5c45dc192365429af27c6193de752cc160458eff8e21807d37662b1d",
        "size_bytes": 26800,
    },
    "v3_action_protocol": {
        "path": "configs/swe_task_state_v3_action_probes.json",
        "sha256": "0ebd258a2b46beb2a9be3d42cab24680803a2f971cb21e96acecb78e19cd81bf",
        "size_bytes": 6364,
    },
}

EXPECTED_CODE_DEPENDENCIES = [
    {
        "role": "frozen_v3_streaming_analyzer",
        "path": "scripts/analyze_swe_task_state_v3.py",
        "sha256": "53c7d41688f6c5ab21f7ad029d343af06e9b13c777fd2e5517ff8d5254ad9e6c",
        "size_bytes": 215115,
    },
    {
        "role": "v4_streaming_extractor",
        "path": "scripts/swe_task_state_v4_extract.py",
        "sha256": "43d22531de0cb04d94bf65e5a6ac8897b340c9c34fb2213ff12bfaa662efed10",
        "size_bytes": 16528,
    },
    {
        "role": "v4_visible_baseline_authentication_helpers",
        "path": "scripts/swe_task_state_v4_visible_baselines.py",
        "sha256": "eedf004886ab570fc6ec4fe4f70634fa0f185c8356f74bb19cbb23bf70b20600",
        "size_bytes": 37782,
    },
    {
        "role": "future_target_materializer",
        "path": "scripts/materialize_swe_behavioral_probes.py",
        "sha256": "c63fac2907b887d973920c8fc71adf219affa1d6373a0aeb8ac2fffd59940a4e",
        "size_bytes": 121628,
    },
]

EXPECTED_SOURCE_CONTRACT = {
    "all_boundary_count": 1708,
    "stable_boundary_count": 1606,
    "future_hidden_boundary_count_before_stability_filter": 61,
    "eligible_target_boundary_instances_before_stability_filter": 77,
    "analysis_role": "future_agent_hidden_available",
    "target_kind": "identifier",
    "future_support_contract": "intersection_of_agent_generated_patch_mutation_completion_and_terminal_summary_v1",
    "benchmark_gold_used": False,
    "lens_output_used_to_derive_targets": False,
    "target_and_retained_foils_absent_from_complete_visible_prefix": True,
    "retained_foil_scope": "same_task_same_identifier_kind_generated_patch_removed_or_context_identifier",
    "completion_at_or_after_current_boundary_as_feature_forbidden": True,
}

EXPECTED_SCORING = {
    "methods_in_order": list(METHODS),
    "layers_in_order": list(LAYERS),
    "position": "final_visible_prefix_token_immediately_before_ensuing_completion",
    "token_form_reduction": "logmeanexp_of_raw_vocabulary_scores_within_layer",
    "layer_reduction": "arithmetic_mean_across_all_24_fixed_layers",
    "candidate_probability": "softmax_over_target_and_all_retained_hidden_foil_identifier_surface_scores",
    "candidate_tie_break": "lexical_candidate_id",
    "vocabulary_size": VOCABULARY_SIZE,
    "open_vocabulary_rank_utility": "mean_over_layers_of_log(vocabulary_size/best_target_form_rank)/log(vocabulary_size)",
    "open_vocabulary_top_k": list(OPEN_VOCABULARY_TOP_K),
}

EXPECTED_ESTIMAND = {
    "weighting": "equal_repository_then_equal_task_then_equal_target_then_equal_boundary",
    "candidate_metrics": list(CANDIDATE_METRICS),
    "open_vocabulary_metrics": ["rank_utility", "top_k_layer_fraction"],
    "paired_reference": "ordinary_logit",
    "paired_candidate": "public_jacobian",
}

EXPECTED_UNCERTAINTY = {
    "status": "development_descriptive_no_refit_needed_for_fixed_readouts",
    "algorithm": "paired_repository_then_task_cluster_bootstrap_keep_complete_target_trajectories",
    "draw_count": 5000,
    "seed": 20260720,
    "confidence_level": 0.95,
    "multiplicity_adjusted": False,
}

EXPECTED_SUPPORT_GATE = {
    "minimum_tasks": 10,
    "minimum_repositories": 6,
    "origin": "preexisting_future_identifier_control_support_gate",
    "passing_support_does_not_establish_semantic_recovery": True,
}

EXPECTED_RENDERER = {
    "identifier_ranking_template": "At request {request_index}, the {method_label} fixed-vocabulary readout ranks `{candidate_text}` highest among the supplied future-derived identifier surface forms.",
    "grouping": "repository_then_task_then_future_target_id_in_increasing_request_order",
    "true_target_text_never_selects_or_changes_rendered_ranking": True,
    "candidate_set_is_future_derived_and_not_available_to_an_online_open_ended_decoder": True,
    "proposition_relation_supplied_not_decoded": True,
    "rendered_sentence_is_fixed_wrapper_not_decoded_language": True,
    "open_ended_sentence_generation": False,
}

EXPECTED_FORBIDDEN_INPUTS = [
    "official_outcome",
    "benchmark_gold_patch_or_test",
    "completion_text_as_feature",
    "completion_at_or_after_current_boundary_as_feature",
    "target_identity_as_numeric_feature",
    "target_or_foil_source_kind_as_numeric_feature",
    "repository_or_task_identity_as_numeric_feature",
    "private_reasoning",
    "emotion_label",
]

EXPECTED_CLAIM_SCOPE = {
    "target_and_retained_foil_identifier_surface_forms_absent_from_visible_prefix": True,
    "fixed_future_derived_identifier_candidate_ranking_evaluated": True,
    "proposition_relation_supplied_not_decoded": True,
    "open_ended_concept_generation_established": False,
    "sentence_level_semantic_reconstruction_established": False,
    "semantic_concept_chain_decoded": False,
    "cot_or_cot_like_decoded": False,
    "private_chain_of_thought_reconstructed": False,
    "hidden_reasoning_or_understanding_established": False,
    "emotion_confidence_doubt_or_stress_decoded": False,
    "causal_interpretation_established": False,
    "operational_reliability_established": False,
}


class HiddenFutureConceptError(ValueError):
    """Raised whenever a frozen contract or leakage boundary fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HiddenFutureConceptError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)),
        f"{label} must be an array",
    )
    return value


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be nonempty text")
    return str(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return int(value)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return str(value)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HiddenFutureConceptError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HiddenFutureConceptError(f"cannot read JSON: {path}: {error}") from error


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _lexical_path_has_forbidden_fragment(path: Path) -> bool:
    normalized = os.path.normpath(os.path.abspath(os.fspath(path))).lower()
    return any(
        fragment in component
        for fragment in FORBIDDEN_PATH_FRAGMENTS
        for component in Path(normalized).parts
    )


def lexical_path_preflight(paths: Iterable[Path | None]) -> None:
    """Reject closed-tree text before any stat, hash, resolve, or read."""

    for path in paths:
        if path is not None and _lexical_path_has_forbidden_fragment(Path(path)):
            raise HiddenFutureConceptError(
                f"forbidden path rejected before filesystem access: {path}"
            )


def canonical_path_preflight(
    *, input_paths: Iterable[Path | None], output_paths: Iterable[Path | None]
) -> None:
    """Reject canonical forbidden parents before file contents are touched."""

    for path, strict in [
        *((path, True) for path in input_paths),
        *((path, False) for path in output_paths),
    ]:
        if path is None:
            continue
        try:
            resolved = Path(path).resolve(strict=strict)
        except OSError as error:
            raise HiddenFutureConceptError(f"cannot resolve path: {path}: {error}") from error
        if any(
            fragment in component.lower()
            for fragment in FORBIDDEN_PATH_FRAGMENTS
            for component in resolved.parts
        ):
            raise HiddenFutureConceptError(
                f"forbidden canonical path rejected before file read: {path}"
            )


def _bound_path(record: Mapping[str, Any]) -> Path:
    raw = Path(str(record["path"]))
    return raw if raw.is_absolute() else ROOT / raw


def _validate_bound_record(value: Any, *, label: str) -> dict[str, Any]:
    record = dict(_mapping(value, label))
    _require(
        set(record) == {"path", "sha256", "size_bytes"},
        f"{label} binding schema changed",
    )
    _text(record["path"], f"{label} path")
    _sha256(record["sha256"], f"{label} SHA-256")
    _integer(record["size_bytes"], f"{label} size", minimum=1)
    _require(
        not _lexical_path_has_forbidden_fragment(Path(record["path"])),
        f"{label} path is forbidden",
    )
    return record


def validate_config(value: Any) -> dict[str, Any]:
    config = dict(_mapping(value, "hidden-future config"))
    expected_keys = {
        "schema_version",
        "id",
        "status",
        "inputs",
        "code_dependencies",
        "source_contract",
        "scoring",
        "estimand",
        "uncertainty",
        "support_gate",
        "renderer",
        "forbidden_inputs",
        "claim_scope",
        "mandatory_limitations",
    }
    _require(set(config) == expected_keys, "hidden-future config schema changed")
    _require(
        config["schema_version"] == 1
        and config["id"]
        == "swe-task-state-v4-hidden-future-identifier-surface-ranking"
        and config["status"] == "development_only_reserved_validation_closed",
        "hidden-future config identity changed",
    )
    inputs = dict(_mapping(config["inputs"], "hidden-future inputs"))
    _require(inputs == EXPECTED_INPUTS, "hidden-future input bindings changed")
    for name, record in inputs.items():
        _validate_bound_record(record, label=f"input {name}")
    dependencies = list(_sequence(config["code_dependencies"], "code dependencies"))
    _require(
        dependencies == EXPECTED_CODE_DEPENDENCIES,
        "hidden-future code dependency bindings changed",
    )
    for position, record in enumerate(dependencies):
        value = dict(_mapping(record, f"code dependency {position}"))
        _require(set(value) == {"role", "path", "sha256", "size_bytes"},
                 f"code dependency {position} schema changed")
        _text(value["role"], f"code dependency {position} role")
        _validate_bound_record(
            {key: value[key] for key in ("path", "sha256", "size_bytes")},
            label=f"code dependency {position}",
        )
    exact_sections = (
        ("source_contract", EXPECTED_SOURCE_CONTRACT),
        ("scoring", EXPECTED_SCORING),
        ("estimand", EXPECTED_ESTIMAND),
        ("uncertainty", EXPECTED_UNCERTAINTY),
        ("support_gate", EXPECTED_SUPPORT_GATE),
        ("renderer", EXPECTED_RENDERER),
    )
    for name, expected in exact_sections:
        _require(config[name] == expected, f"hidden-future {name} changed")
    _require(
        config["forbidden_inputs"] == EXPECTED_FORBIDDEN_INPUTS,
        "hidden-future forbidden-input registry changed",
    )
    _require(
        config["claim_scope"] == EXPECTED_CLAIM_SCOPE,
        "hidden-future claim scope changed",
    )
    _require(
        config["mandatory_limitations"] == list(MANDATORY_LIMITATIONS),
        "hidden-future mandatory limitations changed",
    )
    return config


def authenticate_records(
    records: Sequence[Mapping[str, Any]], *, label: str
) -> list[Path]:
    """Authenticate an already-validated registry after a global path preflight."""

    paths = [_bound_path(record) for record in records]
    lexical_path_preflight(paths)
    canonical_path_preflight(input_paths=paths, output_paths=[])
    resolved: list[Path] = []
    for position, (path, record) in enumerate(zip(paths, records, strict=True)):
        _require(
            path.is_file() and not path.is_symlink(),
            f"{label} {position} is not a regular nonsymlink file",
        )
        observed_size = path.stat().st_size
        observed_hash = sha256_file(path)
        _require(
            observed_size == record["size_bytes"]
            and observed_hash == record["sha256"],
            f"{label} {position} byte binding changed",
        )
        resolved.append(path.resolve(strict=True))
    return resolved


def validate_alignment_index(
    value: Any, *, expected_total_count: int, expected_stable_count: int
) -> list[dict[str, Any]]:
    index = dict(_mapping(value, "alignment index"))
    expected_top_keys = {
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
    }
    _require(set(index) == expected_top_keys, "alignment index schema changed")
    _require(
        index["schema_version"] == 1
        and index["kind"] == "swe_task_state_v4_label_free_alignment_index"
        and index["status"] == "passed"
        and index["scope"] == "grouping_order_and_stability_only_no_labels"
        and index["row_count"] == expected_total_count
        and index["stable_row_count"] == expected_stable_count
        and index["feature_use"]
        == {
            "allowed": [
                "task-local ordering for causal temporal transforms",
                "repository and task grouping for held-out splits and weights",
                "stable eligibility filtering",
            ],
            "forbidden": [
                "hashing or one-hot encoding IDs as model features",
                "repository or request index as semantic model features",
            ],
        },
        "alignment index identity or feature-use contract changed",
    )
    rows = list(_sequence(index["rows"], "alignment rows"))
    _require(len(rows) == expected_total_count, "alignment row count changed")
    row_keys = {
        "global_index",
        "source_id_sha256",
        "task_id_sha256",
        "repository",
        "request_index",
        "stable_feature_eligible",
    }
    seen_sources: set[str] = set()
    seen_requests: set[tuple[str, str, int]] = set()
    previous_by_task: dict[tuple[str, str], int] = {}
    requests_by_task: dict[tuple[str, str], list[int]] = defaultdict(list)
    repository_by_task: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    for position, raw in enumerate(rows):
        row = dict(_mapping(raw, f"alignment row {position}"))
        _require(set(row) == row_keys, f"alignment row {position} schema changed")
        _require(row["global_index"] == position, f"alignment row {position} index changed")
        source_id = _sha256(row["source_id_sha256"], f"alignment row {position} source")
        task_id = _sha256(row["task_id_sha256"], f"alignment row {position} task")
        repository = _text(row["repository"], f"alignment row {position} repository")
        request_index = _integer(
            row["request_index"], f"alignment row {position} request index", minimum=1
        )
        stable = row["stable_feature_eligible"]
        _require(isinstance(stable, bool), f"alignment row {position} stability changed")
        _require(source_id not in seen_sources, "alignment source identities repeat")
        _require(
            task_id not in repository_by_task or repository_by_task[task_id] == repository,
            "one alignment task maps to multiple repositories",
        )
        repository_by_task[task_id] = repository
        task_key = (repository, task_id)
        request_key = (*task_key, request_index)
        previous = previous_by_task.get(task_key)
        _require(request_key not in seen_requests, "alignment task requests repeat")
        _require(
            previous is None or request_index > previous,
            "alignment task request order changed",
        )
        seen_sources.add(source_id)
        seen_requests.add(request_key)
        previous_by_task[task_key] = request_index
        requests_by_task[task_key].append(request_index)
        normalized.append(row)
    for requests in requests_by_task.values():
        _require(
            requests == list(range(1, len(requests) + 1)),
            "alignment task requests are not complete and consecutive",
        )
    _require(
        sum(bool(row["stable_feature_eligible"]) for row in normalized)
        == expected_stable_count,
        "alignment stable count changed",
    )
    return normalized


IDENTIFIER_RE = re.compile(r"(?u)(?:[^\W\d]|_)\w*")
ASCII_SEGMENT_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)


def _normalized_identifier(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def identifier_segments(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).strip("_")
    pieces: list[str] = []
    for component in normalized.split("_"):
        segments = ASCII_SEGMENT_RE.findall(component)
        if segments and "".join(segments).casefold() == component.casefold():
            pieces.extend(segment.casefold() for segment in segments)
        elif component:
            pieces.append(component.casefold())
    return tuple(pieces)


def identifier_exposure(text: str, aliases: Sequence[str]) -> list[dict[str, Any]]:
    """Recompute the materializer's NFKC/casefold identifier exposure seam."""

    requested = [
        (alias, identifier_segments(alias))
        for alias in aliases
        if identifier_segments(alias)
    ]
    counts: dict[tuple[str, str, str], int] = {}
    for match in IDENTIFIER_RE.finditer(text):
        identifier = match.group(0)
        identifier_normalized = _normalized_identifier(identifier).strip("_")
        segments = identifier_segments(identifier)
        for alias, alias_segments in requested:
            alias_normalized = _normalized_identifier(alias).strip("_")
            width = len(alias_segments)
            if identifier_normalized == alias_normalized:
                match_kind = "nfkc_casefold_full_identifier"
            elif width and any(
                segments[offset : offset + width] == alias_segments
                for offset in range(len(segments) - width + 1)
            ):
                match_kind = "nfkc_casefold_identifier_segment"
            else:
                continue
            key = (alias, identifier, match_kind)
            counts[key] = counts.get(key, 0) + 1
    return [
        {
            "alias": alias,
            "identifier": identifier,
            "match_kind": match_kind,
            "occurrences": count,
        }
        for (alias, identifier, match_kind), count in sorted(counts.items())
    ]


def _safe_relative_provenance_path(value: Any, label: str) -> str:
    text = _text(value, label)
    path = Path(text)
    _require(
        not path.is_absolute() and ".." not in path.parts,
        f"{label} is not a safe relative provenance path",
    )
    return text


def _validate_span(value: Any, label: str) -> list[int]:
    span = list(_sequence(value, label))
    _require(len(span) == 2, f"{label} must contain two offsets")
    start = _integer(span[0], f"{label} start")
    end = _integer(span[1], f"{label} end")
    _require(end > start, f"{label} is empty or reversed")
    return [start, end]


def validate_future_support(value: Any) -> dict[str, Any]:
    support = dict(_mapping(value, "future support"))
    _require(
        set(support)
        == {
            "contract",
            "benchmark_gold_used",
            "lens_output_used",
            "generated_patch",
            "mutation_completion",
            "terminal_summary",
        },
        "future-support schema changed",
    )
    _require(
        support["contract"] == EXPECTED_SOURCE_CONTRACT["future_support_contract"]
        and support["benchmark_gold_used"] is False
        and support["lens_output_used"] is False,
        "future target used forbidden evidence or changed support contract",
    )
    patch = dict(_mapping(support["generated_patch"], "generated-patch support"))
    _require(
        set(patch)
        == {"path", "sha256", "source_path", "patch_line_number", "line_sha256", "span"},
        "generated-patch support schema changed",
    )
    _safe_relative_provenance_path(patch["path"], "generated patch path")
    _safe_relative_provenance_path(patch["source_path"], "generated source path")
    _sha256(patch["sha256"], "generated patch SHA-256")
    _sha256(patch["line_sha256"], "generated patch line SHA-256")
    _integer(patch["patch_line_number"], "generated patch line", minimum=1)
    _validate_span(patch["span"], "generated patch span")
    mutation = dict(
        _mapping(support["mutation_completion"], "mutation-completion support")
    )
    _require(
        set(mutation)
        == {
            "completion_index",
            "source_request_global_index",
            "next_request_global_index",
            "source_campaign_source_request_global_index",
            "source_campaign_next_request_global_index",
            "channel",
            "channel_text_sha256",
            "span",
        },
        "mutation-completion support schema changed",
    )
    _integer(mutation["completion_index"], "mutation completion index", minimum=1)
    source_global = _integer(
        mutation["source_request_global_index"], "mutation source request", minimum=1
    )
    next_global = _integer(
        mutation["next_request_global_index"], "mutation next request", minimum=1
    )
    campaign_source = _integer(
        mutation["source_campaign_source_request_global_index"],
        "mutation campaign source request",
        minimum=1,
    )
    campaign_next = _integer(
        mutation["source_campaign_next_request_global_index"],
        "mutation campaign next request",
        minimum=1,
    )
    _require(
        next_global == source_global + 1 and campaign_next == campaign_source + 1,
        "mutation completion does not bind the immediately following request",
    )
    _require(
        mutation["channel"] in {"assistant_text", "argument_text"},
        "mutation completion channel changed",
    )
    _sha256(mutation["channel_text_sha256"], "mutation completion text SHA-256")
    _validate_span(mutation["span"], "mutation completion span")
    terminal = dict(_mapping(support["terminal_summary"], "terminal-summary support"))
    _require(
        set(terminal)
        == {
            "runner_metadata_path",
            "runner_metadata_sha256",
            "field",
            "text_sha256",
            "span",
        },
        "terminal-summary support schema changed",
    )
    _safe_relative_provenance_path(
        terminal["runner_metadata_path"], "runner metadata path"
    )
    _sha256(terminal["runner_metadata_sha256"], "runner metadata SHA-256")
    _require(terminal["field"] == "/qwen/result_tail", "terminal summary field changed")
    _sha256(terminal["text_sha256"], "terminal summary text SHA-256")
    _validate_span(terminal["span"], "terminal summary span")
    return support


def _validate_forms(value: Any, *, candidate_text: str, label: str) -> list[dict[str, Any]]:
    raw_forms = list(_sequence(value, f"{label} forms"))
    _require(bool(raw_forms), f"{label} has no single-token forms")
    forms: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_kinds: set[str] = set()
    for position, raw in enumerate(raw_forms):
        form = dict(_mapping(raw, f"{label} form {position}"))
        _require(
            set(form) == {"kind", "text", "token_id"},
            f"{label} form schema changed",
        )
        kind = form["kind"]
        _require(kind in {"bare", "leading_space"}, f"{label} form kind changed")
        expected_text = candidate_text if kind == "bare" else f" {candidate_text}"
        _require(form["text"] == expected_text, f"{label} form text changed")
        token_id = _integer(form["token_id"], f"{label} token ID")
        _require(token_id < VOCABULARY_SIZE, f"{label} token ID exceeds vocabulary")
        _require(
            token_id not in seen_ids and kind not in seen_kinds,
            f"{label} forms repeat",
        )
        seen_ids.add(token_id)
        seen_kinds.add(str(kind))
        forms.append({"kind": str(kind), "text": expected_text, "token_id": token_id})
    return forms


def _validate_aliases(value: Any, *, candidate_text: str, label: str) -> list[str]:
    aliases = [_text(item, f"{label} alias") for item in _sequence(value, f"{label} aliases")]
    _require(
        aliases == [candidate_text],
        f"{label} alias contract changed",
    )
    return aliases


def _validate_embedded_absence_evidence(
    value: Any, *, expected_aliases: Sequence[str], label: str
) -> None:
    evidence = dict(_mapping(value, label))
    _require(
        set(evidence)
        == {
            "normalization",
            "aliases",
            "identifier_hits",
            "scored_form_token_id_hits",
            "exposed",
        }
        and evidence["normalization"]
        == "NFKC_then_casefold_with_snake_camel_identifier_segments_v1"
        and evidence["aliases"] == list(expected_aliases)
        and evidence["identifier_hits"] == []
        and evidence["scored_form_token_id_hits"] == []
        and evidence["exposed"] is False,
        f"{label} does not prove prefix absence",
    )


def _validate_complete_prefix_absence(
    *,
    prefix_text: str,
    prefix_token_ids: Sequence[int],
    aliases: Sequence[str],
    forms: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    _require(
        identifier_exposure(prefix_text, aliases) == [],
        f"{label} appears in the complete visible prefix",
    )
    prefix_ids = set(prefix_token_ids)
    form_ids = {int(form["token_id"]) for form in forms}
    _require(
        prefix_ids.isdisjoint(form_ids),
        f"{label} scored token form appears in the complete visible prefix",
    )


def _neutral_candidate(
    *, candidate_text: str, kind: str, forms: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    neutral_content = {
        "candidate_text": candidate_text,
        "kind": kind,
        "forms": [dict(form) for form in forms],
    }
    return {
        "candidate_id": f"candidate-{sha256_json(neutral_content)}",
        "candidate_text": candidate_text,
        "forms": [dict(form) for form in forms],
    }


def build_hidden_candidate_set(
    *,
    target: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    task_instance_id: str,
    prefix_text: str,
    prefix_token_ids: Sequence[int],
) -> tuple[list[dict[str, Any]], str]:
    """Validate one eligible target and strip every target/foil source marker.

    The returned candidate records contain exactly a neutral ID, text for the
    fixed renderer, and token forms.  The true neutral ID is returned through a
    separate value used only by the evaluation seam.
    """

    target_value = dict(_mapping(target, "future target"))
    _require(
        set(target_value)
        == {
            "id",
            "kind",
            "target",
            "forms",
            "aliases",
            "future_support",
            "foils",
            "task_instance_id",
        },
        "future target schema changed",
    )
    target_id = _text(target_value["id"], "future target ID")
    _require(
        target_value["task_instance_id"] == task_instance_id,
        "future target crosses task boundary",
    )
    kind = _text(target_value["kind"], "future target kind")
    _require(kind == EXPECTED_SOURCE_CONTRACT["target_kind"], "future target kind changed")
    target_text = _text(target_value["target"], "future target text")
    target_forms = _validate_forms(
        target_value["forms"], candidate_text=target_text, label="future target"
    )
    target_aliases = _validate_aliases(
        target_value["aliases"], candidate_text=target_text, label="future target"
    )
    validate_future_support(target_value["future_support"])
    raw_foils = list(_sequence(target_value["foils"], "future target foils"))
    _require(bool(raw_foils), "future target has no same-kind foils")
    foils_by_id: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(raw_foils):
        foil = dict(_mapping(raw, f"future foil {position}"))
        _require(
            set(foil)
            == {"id", "task_instance_id", "kind", "target", "forms", "aliases", "source"},
            f"future foil {position} schema changed",
        )
        foil_id = _text(foil["id"], f"future foil {position} ID")
        _require(foil_id not in foils_by_id, "future foil IDs repeat")
        _require(
            foil["task_instance_id"] == task_instance_id and foil["kind"] == kind,
            "future foil crosses task or identifier kind",
        )
        foil_text = _text(foil["target"], f"future foil {position} text")
        foil["_validated_forms"] = _validate_forms(
            foil["forms"], candidate_text=foil_text, label=f"future foil {position}"
        )
        foil["_validated_aliases"] = _validate_aliases(
            foil["aliases"], candidate_text=foil_text, label=f"future foil {position}"
        )
        source = dict(_mapping(foil["source"], f"future foil {position} source"))
        _require(
            set(source) == {"type", "path", "patch_line_number", "line_sha256"}
            and source["type"]
            in {
                "generated_patch_removed_identifier",
                "generated_patch_context_identifier",
            },
            "future foil source contract changed",
        )
        _safe_relative_provenance_path(source["path"], f"future foil {position} path")
        _integer(source["patch_line_number"], f"future foil {position} line", minimum=1)
        _sha256(source["line_sha256"], f"future foil {position} line SHA-256")
        foils_by_id[foil_id] = foil

    eligible = dict(_mapping(eligibility, "target eligibility"))
    _require(
        set(eligible)
        == {
            "target_id",
            "target_exposed",
            "retained_hidden_foil_ids",
            "excluded_foils",
            "status",
            "target_channel_evidence",
            "target_rendered_evidence",
            "foil_evidence",
        }
        and eligible["target_id"] == target_id
        and eligible["target_exposed"] is False
        and eligible["status"] == "eligible",
        "target is not an eligible hidden future identifier surface form",
    )
    retained_ids = [
        _text(item, "retained foil ID")
        for item in _sequence(
            eligible["retained_hidden_foil_ids"], "retained hidden foil IDs"
        )
    ]
    _require(
        bool(retained_ids)
        and len(retained_ids) == len(set(retained_ids))
        and all(identifier in foils_by_id for identifier in retained_ids),
        "retained hidden foil registry changed",
    )
    _validate_embedded_absence_evidence(
        eligible["target_rendered_evidence"],
        expected_aliases=target_aliases,
        label="target rendered evidence",
    )
    for channel, evidence in _mapping(
        eligible["target_channel_evidence"], "target channel evidence"
    ).items():
        _text(channel, "target evidence channel")
        _validate_embedded_absence_evidence(
            evidence,
            expected_aliases=target_aliases,
            label=f"target channel evidence {channel}",
        )
    foil_evidence_rows = list(_sequence(eligible["foil_evidence"], "foil evidence"))
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for position, raw in enumerate(foil_evidence_rows):
        evidence = dict(_mapping(raw, f"foil evidence {position}"))
        _require(
            set(evidence)
            == {"foil_id", "exposed", "channel_evidence", "rendered_evidence"},
            f"foil evidence {position} schema changed",
        )
        foil_id = _text(evidence["foil_id"], f"foil evidence {position} ID")
        _require(foil_id in foils_by_id and foil_id not in evidence_by_id,
                 "foil evidence registry changed")
        evidence_by_id[foil_id] = evidence
    _require(set(evidence_by_id) == set(foils_by_id), "foil evidence coverage changed")
    excluded_ids: set[str] = set()
    for position, raw in enumerate(_sequence(eligible["excluded_foils"], "excluded foils")):
        excluded = dict(_mapping(raw, f"excluded foil {position}"))
        _require(
            set(excluded) == {"foil_id", "reason"}
            and excluded["reason"] == "prompt_exposure"
            and excluded["foil_id"] in foils_by_id,
            "excluded foil contract changed",
        )
        excluded_ids.add(str(excluded["foil_id"]))
    _require(
        not (set(retained_ids) & excluded_ids)
        and set(retained_ids) | excluded_ids == set(foils_by_id),
        "retained and excluded foil partition changed",
    )

    prefix_ids = [
        _integer(item, "visible prefix token ID") for item in prefix_token_ids
    ]
    _validate_complete_prefix_absence(
        prefix_text=prefix_text,
        prefix_token_ids=prefix_ids,
        aliases=target_aliases,
        forms=target_forms,
        label="future target",
    )
    target_candidate = _neutral_candidate(
        candidate_text=target_text, kind=kind, forms=target_forms
    )
    candidates = [target_candidate]
    all_form_ids = {int(form["token_id"]) for form in target_forms}
    all_texts = {_normalized_identifier(target_text)}
    for foil_id in retained_ids:
        foil = foils_by_id[foil_id]
        evidence = evidence_by_id[foil_id]
        _require(evidence["exposed"] is False, "retained foil is marked exposed")
        foil_aliases = list(foil["_validated_aliases"])
        _validate_embedded_absence_evidence(
            evidence["rendered_evidence"],
            expected_aliases=foil_aliases,
            label=f"retained foil {foil_id} rendered evidence",
        )
        for channel, channel_evidence in _mapping(
            evidence["channel_evidence"], f"retained foil {foil_id} channel evidence"
        ).items():
            _text(channel, "foil evidence channel")
            _validate_embedded_absence_evidence(
                channel_evidence,
                expected_aliases=foil_aliases,
                label=f"retained foil {foil_id} channel evidence {channel}",
            )
        foil_forms = list(foil["_validated_forms"])
        _validate_complete_prefix_absence(
            prefix_text=prefix_text,
            prefix_token_ids=prefix_ids,
            aliases=foil_aliases,
            forms=foil_forms,
            label=f"retained foil {foil_id}",
        )
        form_ids = {int(form["token_id"]) for form in foil_forms}
        normalized_text = _normalized_identifier(str(foil["target"]))
        _require(
            all_form_ids.isdisjoint(form_ids) and normalized_text not in all_texts,
            "candidate texts or token forms collide",
        )
        all_form_ids.update(form_ids)
        all_texts.add(normalized_text)
        candidates.append(
            _neutral_candidate(
                candidate_text=str(foil["target"]), kind=kind, forms=foil_forms
            )
        )
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    _require(len(candidate_ids) == len(set(candidate_ids)), "neutral candidate IDs collide")
    _require(
        all(set(candidate) == {"candidate_id", "candidate_text", "forms"} for candidate in candidates),
        "source-kind field leaked into neutral candidate records",
    )
    candidates.sort(key=lambda candidate: str(candidate["candidate_id"]))
    return candidates, str(target_candidate["candidate_id"])


def logmeanexp(values: Sequence[float]) -> float:
    _require(bool(values), "cannot reduce an empty score group")
    finite_values = [_finite(value, "token score") for value in values]
    maximum = max(finite_values)
    result = maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in finite_values)
        / len(finite_values)
    )
    _require(math.isfinite(result), "logmeanexp produced a nonfinite value")
    return result


def _validated_readout_tokens(
    value: Any, *, label: str
) -> dict[int, dict[str, Any]]:
    readout = dict(_mapping(value, label))
    required = {
        "scored_tokens",
        "scores",
        "token_ids",
        "tokens",
        "target_logprob",
        "target_rank",
        "target_score",
        "target_token",
        "target_token_id",
        "final_distribution_fidelity",
    }
    _require(set(readout) == required, f"{label} schema changed")
    scored = list(_sequence(readout["scored_tokens"], f"{label} scored tokens"))
    token_ids = list(_sequence(readout["token_ids"], f"{label} token IDs"))
    tokens = list(_sequence(readout["tokens"], f"{label} tokens"))
    scores = list(_sequence(readout["scores"], f"{label} scores"))
    _require(
        bool(token_ids) and len(token_ids) == len(tokens) == len(scores),
        f"{label} top-vocabulary arrays differ",
    )
    top_ids: set[int] = set()
    for position, (raw_id, raw_token, raw_score) in enumerate(
        zip(token_ids, tokens, scores, strict=True)
    ):
        top_id = _integer(raw_id, f"{label} top token ID")
        _require(
            top_id < VOCABULARY_SIZE and top_id not in top_ids,
            f"{label} top token IDs are invalid or repeated",
        )
        top_ids.add(top_id)
        _text(raw_token, f"{label} top token text")
        _finite(raw_score, f"{label} top token score")
    by_token: dict[int, dict[str, Any]] = {}
    for position, raw in enumerate(scored):
        item = dict(_mapping(raw, f"{label} scored token {position}"))
        _require(
            set(item) == {"token_id", "token", "score", "logprob", "rank"},
            f"{label} scored token schema changed",
        )
        token_id = _integer(item["token_id"], f"{label} token ID")
        _require(token_id < VOCABULARY_SIZE, f"{label} token ID exceeds vocabulary")
        token = _text(item["token"], f"{label} token text")
        score = _finite(item["score"], f"{label} token score")
        logprob = _finite(item["logprob"], f"{label} token log probability")
        rank = _integer(item["rank"], f"{label} token rank", minimum=1)
        _require(rank <= VOCABULARY_SIZE, f"{label} token rank exceeds vocabulary")
        _require(token_id not in by_token, f"{label} token IDs repeat")
        by_token[token_id] = {
            "token_id": token_id,
            "token": token,
            "score": score,
            "logprob": logprob,
            "rank": rank,
        }
    return by_token


def score_candidate_set(
    *,
    experiment: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    method: str,
    expected_token_position: int,
    layers: Sequence[int] = LAYERS,
) -> dict[str, Any]:
    """Score neutral candidates without accepting any target or source-kind input."""

    _require(method in METHODS, f"unknown hidden-future scoring method: {method}")
    layer_order = [
        _integer(layer, "configured layer", minimum=0) for layer in layers
    ]
    _require(layer_order == list(LAYERS), "hidden-future layer order changed")
    candidate_rows = [dict(_mapping(value, "neutral candidate")) for value in candidates]
    _require(len(candidate_rows) >= 2, "candidate audit requires target plus a foil")
    seen_candidates: set[str] = set()
    seen_form_ids: set[int] = set()
    normalized_candidates: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidate_rows):
        _require(
            set(candidate) == {"candidate_id", "candidate_text", "forms"},
            "candidate scorer received a source-kind or target field",
        )
        candidate_id = _text(candidate["candidate_id"], f"candidate {position} ID")
        candidate_text = _text(
            candidate["candidate_text"], f"candidate {position} text"
        )
        _require(candidate_id not in seen_candidates, "neutral candidate IDs repeat")
        seen_candidates.add(candidate_id)
        forms = _validate_forms(
            candidate["forms"], candidate_text=candidate_text, label=f"candidate {position}"
        )
        form_ids = {int(form["token_id"]) for form in forms}
        _require(
            seen_form_ids.isdisjoint(form_ids),
            "candidate scorer received colliding token forms",
        )
        seen_form_ids.update(form_ids)
        normalized_candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_text": candidate_text,
                "forms": forms,
            }
        )

    layer_values = list(_sequence(experiment.get("layers"), "report layers"))
    by_layer: dict[int, Mapping[str, Any]] = {}
    for raw in layer_values:
        layer = _mapping(raw, "report layer")
        layer_id = _integer(layer.get("layer"), "report layer ID")
        _require(layer_id not in by_layer, "report contains duplicate layers")
        by_layer[layer_id] = layer
    readout_key = "logit_lens" if method == "ordinary_logit" else "jacobian_lens"
    per_candidate_scores: dict[str, list[float]] = {
        str(candidate["candidate_id"]): [] for candidate in normalized_candidates
    }
    per_candidate_best_ranks: dict[str, list[int]] = {
        str(candidate["candidate_id"]): [] for candidate in normalized_candidates
    }
    for layer_id in layer_order:
        layer = _mapping(by_layer.get(layer_id), f"report layer {layer_id}")
        positions = list(_sequence(layer.get("positions"), f"layer {layer_id} positions"))
        _require(len(positions) == 1, "readout must contain one capture position")
        position = _mapping(positions[0], f"layer {layer_id} position")
        _require(
            _integer(position.get("token_position"), "captured token position")
            == expected_token_position,
            "readout was not captured at the final visible prefix token",
        )
        tokens = _validated_readout_tokens(
            position.get(readout_key), label=f"{method} layer {layer_id}"
        )
        for candidate in normalized_candidates:
            candidate_id = str(candidate["candidate_id"])
            form_records: list[Mapping[str, Any]] = []
            for form in candidate["forms"]:
                token_id = int(form["token_id"])
                _require(
                    token_id in tokens,
                    f"{method} layer {layer_id} lacks a declared candidate token",
                )
                token_record = tokens[token_id]
                _require(
                    token_record["token"] == form["text"],
                    f"{method} layer {layer_id} candidate token text changed",
                )
                form_records.append(token_record)
            per_candidate_scores[candidate_id].append(
                logmeanexp([float(record["score"]) for record in form_records])
            )
            per_candidate_best_ranks[candidate_id].append(
                min(int(record["rank"]) for record in form_records)
            )

    reduced_scores = {
        candidate_id: math.fsum(values) / len(values)
        for candidate_id, values in per_candidate_scores.items()
    }
    _require(
        all(math.isfinite(value) for value in reduced_scores.values()),
        "candidate layer reduction is nonfinite",
    )
    ranking = sorted(reduced_scores, key=lambda key: (-reduced_scores[key], key))
    maximum = max(reduced_scores.values())
    normalizer = maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in reduced_scores.values())
    )
    by_id = {str(candidate["candidate_id"]): candidate for candidate in normalized_candidates}
    scored_candidates: list[dict[str, Any]] = []
    for candidate_id in sorted(by_id):
        candidate = by_id[candidate_id]
        best_ranks = per_candidate_best_ranks[candidate_id]
        log_probability = reduced_scores[candidate_id] - normalizer
        probability = math.exp(log_probability)
        rank_utility = math.fsum(
            math.log(VOCABULARY_SIZE / rank) / math.log(VOCABULARY_SIZE)
            for rank in best_ranks
        ) / len(best_ranks)
        scored_candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_text": candidate["candidate_text"],
                "candidate_score": reduced_scores[candidate_id],
                "candidate_probability": probability,
                "candidate_log_probability": log_probability,
                "candidate_rank": ranking.index(candidate_id) + 1,
                "layer_scores": per_candidate_scores[candidate_id],
                "best_form_rank_by_layer": best_ranks,
                "open_vocabulary_rank_utility": rank_utility,
                "open_vocabulary_top_k_layer_fraction": {
                    str(k): sum(rank <= k for rank in best_ranks) / len(best_ranks)
                    for k in OPEN_VOCABULARY_TOP_K
                },
            }
        )
    _require(
        abs(math.fsum(row["candidate_probability"] for row in scored_candidates) - 1.0)
        <= 1e-12,
        "candidate softmax probabilities do not sum to one",
    )
    return {
        "method": method,
        "layers": layer_order,
        "candidate_order": ranking,
        "predicted_candidate_id": ranking[0],
        "candidates": scored_candidates,
        "target_identity_used_for_prediction": False,
        "candidate_source_kind_used_for_prediction": False,
    }


def evaluate_scored_candidate_set(
    scored: Mapping[str, Any], *, true_candidate_id: str
) -> dict[str, Any]:
    """Introduce true identity only after a complete neutral prediction exists."""

    candidates = [
        dict(_mapping(value, "scored candidate"))
        for value in _sequence(scored.get("candidates"), "scored candidates")
    ]
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    _require(
        len(by_id) == len(candidates) and true_candidate_id in by_id,
        "evaluation target is absent from the scored candidate set",
    )
    target = by_id[true_candidate_id]
    probability = _finite(target["candidate_probability"], "true candidate probability")
    _require(0.0 <= probability <= 1.0, "true candidate probability is invalid")
    log_probability = _finite(
        target["candidate_log_probability"], "true candidate log probability"
    )
    rank = _integer(target["candidate_rank"], "true candidate rank", minimum=1)
    brier = math.fsum(
        (
            _finite(candidate["candidate_probability"], "candidate probability")
            - float(candidate["candidate_id"] == true_candidate_id)
        )
        ** 2
        for candidate in candidates
    )
    metrics = {
        "top1_accuracy": float(scored["predicted_candidate_id"] == true_candidate_id),
        "mean_reciprocal_rank": 1.0 / rank,
        "negative_log_likelihood": -log_probability,
        "multiclass_brier": brier,
        "rank_utility": _finite(
            target["open_vocabulary_rank_utility"], "target rank utility"
        ),
    }
    top_k = _mapping(
        target["open_vocabulary_top_k_layer_fraction"], "target top-k fractions"
    )
    for k in OPEN_VOCABULARY_TOP_K:
        metrics[f"top_k_layer_fraction@{k}"] = _finite(
            top_k[str(k)], f"target top-{k} layer fraction"
        )
    _require(set(metrics) == set(ALL_METRICS), "evaluation metric registry changed")
    return {
        "true_candidate_id": true_candidate_id,
        "true_candidate_rank": rank,
        "predicted_candidate_id": scored["predicted_candidate_id"],
        "prediction_correct": scored["predicted_candidate_id"] == true_candidate_id,
        "metrics": metrics,
        "target_identity_used_only_after_prediction": True,
    }


def render_identifier_ranking(
    *, request_index: int, method_label: str, candidate_text: str, template: str
) -> str:
    """Wrap one ranked identifier; no target or proposition argument is accepted."""

    request = _integer(request_index, "identifier-ranking request index", minimum=1)
    method = _text(method_label, "identifier-ranking method label")
    _require(method in METHODS, "identifier-ranking method label changed")
    candidate = _text(candidate_text, "ranked identifier text")
    _require(
        template == EXPECTED_RENDERER["identifier_ranking_template"],
        "identifier-ranking template changed",
    )
    return template.format(
        request_index=request, method_label=method, candidate_text=candidate
    )


def validate_prompt_report_binding(
    *,
    prompt: Mapping[str, Any],
    experiment: Mapping[str, Any],
    alignment_row: Mapping[str, Any],
    global_index: int,
) -> dict[str, Any]:
    """Bind one visible prefix and readout row to the label-free alignment."""

    _require(
        set(prompt) == {"id", "text", "token_ids", "score_token_ids", "metadata"},
        f"prompt row {global_index} schema changed",
    )
    prompt_id = _text(prompt["id"], f"prompt row {global_index} ID")
    prefix_text = _text(prompt["text"], f"prompt row {global_index} visible prefix")
    token_ids = [
        _integer(item, f"prompt row {global_index} token ID")
        for item in _sequence(prompt["token_ids"], f"prompt row {global_index} tokens")
    ]
    _require(bool(token_ids), f"prompt row {global_index} has no visible tokens")
    score_token_ids = [
        _integer(item, f"prompt row {global_index} score token ID")
        for item in _sequence(
            prompt["score_token_ids"], f"prompt row {global_index} score tokens"
        )
    ]
    _require(
        len(score_token_ids) == len(set(score_token_ids))
        and all(token_id < VOCABULARY_SIZE for token_id in score_token_ids),
        f"prompt row {global_index} scored vocabulary changed",
    )
    metadata = dict(_mapping(prompt["metadata"], f"prompt row {global_index} metadata"))
    task = _mapping(metadata.get("task"), f"prompt row {global_index} task")
    selection = _mapping(
        metadata.get("selection"), f"prompt row {global_index} selection"
    )
    task_instance_id = _text(task.get("instance_id"), "prompt task instance ID")
    repository = _text(task.get("repo"), "prompt task repository")
    request_index = _integer(
        selection.get("task_request_index"), "prompt task request index", minimum=1
    )
    _require(
        alignment_row.get("global_index") == global_index
        and alignment_row.get("source_id_sha256") == sha256_text(prompt_id)
        and alignment_row.get("task_id_sha256") == sha256_text(task_instance_id)
        and alignment_row.get("repository") == repository
        and alignment_row.get("request_index") == request_index,
        f"prompt row {global_index} differs from label-free alignment",
    )
    _require(
        experiment.get("id") == prompt_id
        and experiment.get("prompt") == prefix_text
        and experiment.get("prompt_token_ids") == token_ids
        and experiment.get("metadata") == metadata,
        f"prompt/report payload binding changed at row {global_index}",
    )
    expected_position = len(token_ids) - 1
    _require(
        experiment.get("capture_positions_resolved") == [expected_position],
        f"row {global_index} was not captured only at the final prefix token",
    )
    scored_vocabulary = _mapping(
        experiment.get("scored_vocabulary"), f"row {global_index} scored vocabulary"
    )
    _require(
        set(scored_vocabulary) == {"token_ids", "tokens"}
        and scored_vocabulary["token_ids"] == score_token_ids,
        f"row {global_index} report vocabulary differs from prompt contract",
    )
    return {
        "prompt_id": prompt_id,
        "prefix_text": prefix_text,
        "prefix_token_ids": token_ids,
        "score_token_ids": score_token_ids,
        "metadata": metadata,
        "task_instance_id": task_instance_id,
        "task_id_sha256": str(alignment_row["task_id_sha256"]),
        "repository": repository,
        "request_index": request_index,
        "expected_token_position": expected_position,
    }


def hierarchical_row_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Equalize repository, then task, then target, then retained boundary."""

    _require(bool(rows), "hierarchical weights require scored rows")
    groups: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    seen_boundaries: set[tuple[str, str, str, int]] = set()
    for position, row in enumerate(rows):
        repository = _text(row.get("repository"), "weight repository")
        task_id = _text(row.get("task_id_sha256"), "weight task")
        target_id = _text(row.get("future_target_id"), "weight target")
        global_index = _integer(row.get("global_index"), "weight global index")
        identity = (repository, task_id, target_id, global_index)
        _require(identity not in seen_boundaries, "scored target boundaries repeat")
        seen_boundaries.add(identity)
        groups[repository][task_id][target_id].append(position)
    weights = np.zeros(len(rows), dtype=np.float64)
    repository_mass = 1.0 / len(groups)
    for tasks in groups.values():
        task_mass = repository_mass / len(tasks)
        for targets in tasks.values():
            target_mass = task_mass / len(targets)
            for indices in targets.values():
                boundary_mass = target_mass / len(indices)
                weights[np.asarray(indices, dtype=np.int64)] = boundary_mass
    _require(
        bool(np.all(np.isfinite(weights)))
        and bool(np.all(weights > 0.0))
        and abs(float(weights.sum()) - 1.0) <= 1e-12,
        "hierarchical weights are invalid",
    )
    return weights


def _row_metric(row: Mapping[str, Any], method: str, metric: str) -> float:
    methods = _mapping(row.get("methods"), "row methods")
    method_row = _mapping(methods.get(method), f"row method {method}")
    evaluation = _mapping(method_row.get("evaluation"), f"{method} evaluation")
    metrics = _mapping(evaluation.get("metrics"), f"{method} metrics")
    return _finite(metrics.get(metric), f"{method} {metric}")


def weighted_method_metrics(
    rows: Sequence[Mapping[str, Any]], weights: Sequence[float]
) -> dict[str, dict[str, float]]:
    weight_array = np.asarray(weights, dtype=np.float64)
    _require(
        weight_array.shape == (len(rows),)
        and bool(np.all(np.isfinite(weight_array)))
        and bool(np.all(weight_array > 0.0))
        and abs(float(weight_array.sum()) - 1.0) <= 1e-12,
        "metric weights are invalid",
    )
    result: dict[str, dict[str, float]] = {}
    for method in METHODS:
        result[method] = {}
        for metric in ALL_METRICS:
            values = np.asarray(
                [_row_metric(row, method, metric) for row in rows], dtype=np.float64
            )
            _require(bool(np.all(np.isfinite(values))), f"{method} {metric} is nonfinite")
            result[method][metric] = float(np.sum(weight_array * values, dtype=np.float64))
    return result


def _task_metric_aggregates(
    rows: Sequence[Mapping[str, Any]], *, method: str, metric: str
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        repository = str(row["repository"])
        task = str(row["task_id_sha256"])
        target = str(row["future_target_id"])
        values[repository][task][target].append(_row_metric(row, method, metric))
    result: dict[str, dict[str, float]] = {}
    for repository, tasks in values.items():
        result[repository] = {}
        for task, targets in tasks.items():
            target_means = [math.fsum(items) / len(items) for items in targets.values()]
            result[repository][task] = math.fsum(target_means) / len(target_means)
    return result


def paired_repository_task_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    point_metrics: Mapping[str, Mapping[str, float]],
    draw_count: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Paired repo->task cluster bootstrap retaining each target trajectory."""

    draws = _integer(draw_count, "bootstrap draw count", minimum=1)
    rng_seed = _integer(seed, "bootstrap seed")
    confidence = _finite(confidence_level, "bootstrap confidence level")
    _require(0.0 < confidence < 1.0, "bootstrap confidence level is invalid")
    repositories = sorted({str(row["repository"]) for row in rows})
    _require(bool(repositories), "bootstrap requires repositories")
    alpha = (1.0 - confidence) / 2.0
    results: dict[str, Any] = {}
    for metric in ALL_METRICS:
        reference = _task_metric_aggregates(
            rows, method="ordinary_logit", metric=metric
        )
        candidate = _task_metric_aggregates(
            rows, method="public_jacobian", metric=metric
        )
        _require(reference.keys() == candidate.keys(), "paired repository support changed")
        task_deltas: dict[str, dict[str, float]] = {}
        for repository in repositories:
            _require(
                reference[repository].keys() == candidate[repository].keys(),
                "paired task support changed",
            )
            task_deltas[repository] = {
                task: candidate[repository][task] - reference[repository][task]
                for task in reference[repository]
            }
        rng = np.random.Generator(np.random.PCG64(rng_seed))
        bootstrap_values = np.empty(draws, dtype=np.float64)
        for draw in range(draws):
            sampled_repositories = rng.integers(0, len(repositories), size=len(repositories))
            repository_values: list[float] = []
            for repository_index in sampled_repositories:
                repository = repositories[int(repository_index)]
                tasks = sorted(task_deltas[repository])
                sampled_tasks = rng.integers(0, len(tasks), size=len(tasks))
                repository_values.append(
                    math.fsum(
                        task_deltas[repository][tasks[int(task_index)]]
                        for task_index in sampled_tasks
                    )
                    / len(tasks)
                )
            bootstrap_values[draw] = math.fsum(repository_values) / len(repository_values)
        _require(bool(np.all(np.isfinite(bootstrap_values))), "bootstrap became nonfinite")
        point_delta = (
            float(point_metrics["public_jacobian"][metric])
            - float(point_metrics["ordinary_logit"][metric])
        )
        interval = np.quantile(
            bootstrap_values,
            [alpha, 1.0 - alpha],
            method="inverted_cdf",
        )
        favorable_is_positive = metric not in {
            "negative_log_likelihood",
            "multiclass_brier",
        }
        favorable = (
            bootstrap_values > 0.0
            if favorable_is_positive
            else bootstrap_values < 0.0
        )
        results[metric] = {
            "point_delta_public_jacobian_minus_ordinary_logit": point_delta,
            "bootstrap_mean": float(np.mean(bootstrap_values, dtype=np.float64)),
            "interval": [float(interval[0]), float(interval[1])],
            "interval_contains_zero": bool(interval[0] <= 0.0 <= interval[1]),
            "favorable_direction": "positive" if favorable_is_positive else "negative",
            "fraction_of_draws_favorable": float(np.mean(favorable, dtype=np.float64)),
        }
    return {
        "algorithm": EXPECTED_UNCERTAINTY["algorithm"],
        "draw_count": draws,
        "seed": rng_seed,
        "confidence_level": confidence,
        "interval": "equal_tailed_inverted_cdf",
        "paired_resampling": True,
        "complete_target_trajectories_retained": True,
        "model_refit_needed": False,
        "multiplicity_adjusted": False,
        "results": results,
    }


def build_identifier_ranking_chains(
    rows: Sequence[Mapping[str, Any]], *, template: str
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["repository"]),
            str(row["task_id_sha256"]),
            str(row["future_target_id"]),
        )
        groups[key].append(row)
    chains: list[dict[str, Any]] = []
    for (repository, task_id, target_id), target_rows in sorted(groups.items()):
        ordered = sorted(
            target_rows,
            key=lambda row: (int(row["request_index"]), int(row["global_index"])),
        )
        _require(
            [int(row["request_index"]) for row in ordered]
            == sorted({int(row["request_index"]) for row in ordered}),
            "identifier-ranking target boundaries repeat or are out of order",
        )
        methods: dict[str, list[dict[str, Any]]] = {}
        for method in METHODS:
            method_rows: list[dict[str, Any]] = []
            for row in ordered:
                method_value = _mapping(
                    _mapping(row["methods"], "row methods")[method],
                    f"identifier-ranking method {method}",
                )
                scored = _mapping(method_value["scored"], f"{method} scored candidates")
                predicted_id = str(scored["predicted_candidate_id"])
                predicted = next(
                    (
                        candidate
                        for candidate in _sequence(scored["candidates"], "scored candidates")
                        if candidate["candidate_id"] == predicted_id
                    ),
                    None,
                )
                _require(predicted is not None, "predicted candidate record is missing")
                rendered_ranking = render_identifier_ranking(
                    request_index=int(row["request_index"]),
                    method_label=method,
                    candidate_text=str(predicted["candidate_text"]),
                    template=template,
                )
                method_rows.append(
                    {
                        "global_index": int(row["global_index"]),
                        "request_index": int(row["request_index"]),
                        "predicted_candidate_id": predicted_id,
                        "predicted_candidate_text": str(predicted["candidate_text"]),
                        "rendered_identifier_ranking": rendered_ranking,
                        "true_target_text_used_to_select_or_render": False,
                        "proposition_relation_supplied_not_decoded": True,
                    }
                )
            methods[method] = method_rows
        chains.append(
            {
                "repository": repository,
                "task_id_sha256": task_id,
                "future_target_id": target_id,
                "boundary_count": len(ordered),
                "methods": methods,
            }
        )
    return chains


def build_replay_validation_protocol(
    v3_protocol_value: Any, action_protocol_value: Any
) -> dict[str, Any]:
    """Derive only public-report provenance and numerical-stability checks.

    Deliberately do not call V3's broad protocol validator: that path owns a
    historical closed-tree selection proof which is outside this audit.
    """

    protocol = dict(_mapping(v3_protocol_value, "V3 protocol"))
    actions = dict(_mapping(action_protocol_value, "V3 action protocol"))
    _require(
        (protocol.get("schema_version"), protocol.get("id"))
        == (1, "swe-task-state-interpreter-v3"),
        "V3 protocol identity changed",
    )
    _require(
        (actions.get("schema_version"), actions.get("kind"))
        == (1, "swe_verified_stage_action_probe_protocol"),
        "V3 action protocol identity changed",
    )
    pins = _mapping(protocol.get("pins"), "V3 pins")
    feature = _mapping(protocol.get("feature_contract"), "V3 feature contract")
    target = _mapping(protocol.get("target_contract"), "V3 target contract")
    eligibility = _mapping(
        protocol.get("eligibility_contract"), "V3 eligibility contract"
    )
    _require(
        pins.get("v3_action_protocol_sha256")
        == EXPECTED_INPUTS["v3_action_protocol"]["sha256"],
        "V3 action-protocol pin changed",
    )
    _require(
        feature.get("source_layers") == list(LAYERS)
        and target.get("source_action_classes_in_order")
        == ["inspect", "edit", "validate", "finalize"],
        "V3 source layer or action order changed",
    )
    action_records = list(_sequence(actions.get("action_classes"), "V3 action classes"))
    _require(
        [record.get("id") if isinstance(record, Mapping) else None for record in action_records]
        == ["inspect", "edit", "validate", "finalize"],
        "V3 action registry order changed",
    )
    stability = _mapping(eligibility.get("numerical_stability"), "V3 stability")
    _require(
        (
            stability.get("final_logits_rms_error_maximum_inclusive"),
            stability.get("final_logits_max_abs_error_maximum_inclusive"),
        )
        == (0.02, 0.125),
        "V3 numerical-stability thresholds changed",
    )
    model = _mapping(pins.get("model"), "V3 model pins")
    lens = _mapping(pins.get("public_lens"), "V3 lens pins")
    runtime = _mapping(pins.get("replay_runtime"), "V3 runtime pins")
    model_keys = ("repo_id", "revision", "config_sha256", "index_sha256")
    lens_keys = ("repo_id", "revision", "sha256", "n_prompts")
    runtime_keys = (
        "enforce_eager",
        "mtp_enabled",
        "max_model_len",
        "max_num_batched_tokens",
        "mamba_block_size",
        "kv_cache_dtype",
        "kv_offloading_size",
        "kv_offloading_backend",
        "stream_final_only",
    )
    try:
        report_pins = {
            "model": {key: model[key] for key in model_keys},
            "public_lens": {key: lens[key] for key in lens_keys},
            "runtime": {key: runtime[key] for key in runtime_keys},
        }
    except KeyError as error:
        raise HiddenFutureConceptError(f"V3 replay pin is missing: {error}") from error
    return {
        "report_helper_protocol": {"report_pins": report_pins},
        "eligibility": {"stable_rms": 0.02, "stable_max": 0.125},
    }


def load_pinned_v3_analyzer(path: Path, *, expected_sha256: str) -> Any:
    """Load the frozen streaming parser only after its exact bytes are checked."""

    _sha256(expected_sha256, "V3 analyzer SHA-256")
    _require(
        path.is_file() and not path.is_symlink() and sha256_file(path) == expected_sha256,
        "frozen V3 analyzer changed before import",
    )
    module_name = "scripts.analyze_swe_task_state_v3"
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_path = Path(str(getattr(existing, "__file__", ""))).resolve(strict=True)
        _require(
            existing_path == path.resolve(strict=True)
            and sha256_file(existing_path) == expected_sha256,
            "loaded V3 analyzer resolves to unexpected bytes",
        )
        return existing
    specification = importlib.util.spec_from_file_location(module_name, path)
    _require(
        specification is not None and specification.loader is not None,
        "cannot construct frozen V3 analyzer import",
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
        _require(
            sha256_file(path) == expected_sha256,
            "frozen V3 analyzer changed during import",
        )
    except BaseException:
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise
    return module


def _validate_target_registry_basics(
    *, target: Mapping[str, Any], task_instance_id: str
) -> None:
    value = dict(_mapping(target, "future target"))
    _require(
        set(value)
        == {
            "id",
            "kind",
            "target",
            "forms",
            "aliases",
            "future_support",
            "foils",
            "task_instance_id",
        },
        "future target schema changed",
    )
    _text(value["id"], "future target ID")
    _require(
        value["task_instance_id"] == task_instance_id
        and value["kind"] == EXPECTED_SOURCE_CONTRACT["target_kind"],
        "future target task or kind changed",
    )
    target_text = _text(value["target"], "future target text")
    _validate_forms(value["forms"], candidate_text=target_text, label="future target")
    _validate_aliases(value["aliases"], candidate_text=target_text, label="future target")
    validate_future_support(value["future_support"])


def evaluate_streaming_sources(
    *,
    prompts_path: Path,
    report_path: Path,
    alignment_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    replay_protocol: Mapping[str, Any],
    v3: Any,
) -> dict[str, Any]:
    """Consume prompt/report rows in lockstep while retaining only scored audits."""

    report_metadata: dict[str, Any] = {}
    prompt_iterator = iter(v3._stream_json_array_objects(prompts_path, label="prompt bundle"))
    experiment_iterator = iter(
        v3._stream_report_experiments(report_path, report_metadata)
    )
    scored_rows: list[dict[str, Any]] = []
    processed_count = 0
    recomputed_stable_count = 0
    future_hidden_boundary_count = 0
    eligible_target_boundary_count = 0
    stable_future_hidden_boundary_count = 0
    stable_eligible_target_boundary_count = 0
    sentinel = object()
    try:
        while True:
            prompt = next(prompt_iterator, sentinel)
            experiment = next(experiment_iterator, sentinel)
            if prompt is sentinel and experiment is sentinel:
                break
            _require(prompt is not sentinel, "public report contains trailing rows")
            _require(experiment is not sentinel, "prompt bundle contains trailing rows")
            _require(
                processed_count < len(alignment_rows),
                "prompt/report streams exceed alignment coverage",
            )
            prompt_value = _mapping(prompt, f"prompt row {processed_count}")
            experiment_value = _mapping(
                experiment, f"report experiment {processed_count}"
            )
            alignment = alignment_rows[processed_count]
            binding = validate_prompt_report_binding(
                prompt=prompt_value,
                experiment=experiment_value,
                alignment_row=alignment,
                global_index=processed_count,
            )
            stable, stability_reasons = v3.HISTORICAL_V1._numerically_stable(
                experiment_value, replay_protocol["eligibility"]
            )
            _require(
                bool(stable) == bool(alignment["stable_feature_eligible"]),
                f"recomputed numerical stability differs at row {processed_count}: {stability_reasons}",
            )
            recomputed_stable_count += int(bool(stable))
            metadata = binding["metadata"]
            analysis_role = metadata.get("analysis_role")
            _require(
                analysis_role
                in {"future_agent_hidden_available", "action_outcome_or_exposure_control"},
                f"prompt row {processed_count} analysis role changed",
            )
            targets = [
                dict(_mapping(value, "prompt future target"))
                for value in _sequence(metadata.get("targets"), "prompt future targets")
            ]
            target_ids = [_text(target.get("id"), "prompt future target ID") for target in targets]
            _require(len(target_ids) == len(set(target_ids)), "prompt future target IDs repeat")
            targets_by_id = dict(zip(target_ids, targets, strict=True))
            for target in targets:
                _validate_target_registry_basics(
                    target=target, task_instance_id=str(binding["task_instance_id"])
                )
            eligibility_rows = [
                dict(_mapping(value, "target eligibility"))
                for value in _sequence(
                    metadata.get("target_eligibility"), "target eligibility rows"
                )
            ]
            eligibility_ids = [
                _text(value.get("target_id"), "eligibility target ID")
                for value in eligibility_rows
            ]
            _require(
                len(eligibility_ids) == len(set(eligibility_ids))
                and set(eligibility_ids) == set(target_ids),
                "target eligibility coverage changed",
            )
            eligible_rows = [
                value for value in eligibility_rows if value.get("status") == "eligible"
            ]
            _require(
                (analysis_role == "future_agent_hidden_available")
                == bool(eligible_rows),
                "future-hidden analysis role disagrees with eligible targets",
            )
            if eligible_rows:
                future_hidden_boundary_count += 1
                eligible_target_boundary_count += len(eligible_rows)
                if stable:
                    stable_future_hidden_boundary_count += 1
                    stable_eligible_target_boundary_count += len(eligible_rows)
            for eligibility in eligible_rows:
                target_id = str(eligibility["target_id"])
                candidates, true_candidate_id = build_hidden_candidate_set(
                    target=targets_by_id[target_id],
                    eligibility=eligibility,
                    task_instance_id=str(binding["task_instance_id"]),
                    prefix_text=str(binding["prefix_text"]),
                    prefix_token_ids=binding["prefix_token_ids"],
                )
                candidate_token_ids = {
                    int(form["token_id"])
                    for candidate in candidates
                    for form in candidate["forms"]
                }
                _require(
                    candidate_token_ids.issubset(set(binding["score_token_ids"])),
                    f"row {processed_count} report vocabulary omits a hidden candidate form",
                )
                if not stable:
                    continue
                method_results: dict[str, Any] = {}
                for method in METHODS:
                    scored = score_candidate_set(
                        experiment=experiment_value,
                        candidates=candidates,
                        method=method,
                        expected_token_position=int(binding["expected_token_position"]),
                    )
                    evaluation = evaluate_scored_candidate_set(
                        scored, true_candidate_id=true_candidate_id
                    )
                    method_results[method] = {
                        "scored": scored,
                        "evaluation": evaluation,
                    }
                scored_rows.append(
                    {
                        "global_index": processed_count,
                        "source_id_sha256": str(alignment["source_id_sha256"]),
                        "repository": str(binding["repository"]),
                        "task_id_sha256": str(binding["task_id_sha256"]),
                        "request_index": int(binding["request_index"]),
                        "future_target_id": target_id,
                        "candidate_count": len(candidates),
                        "true_candidate_id": true_candidate_id,
                        "methods": method_results,
                        "feature_boundary": {
                            "complete_visible_prefix_ended_before_completion": True,
                            "target_and_retained_foils_absent": True,
                            "target_identity_used_only_for_evaluation": True,
                            "candidate_source_kind_used_for_prediction": False,
                            "completion_text_used_as_feature": False,
                        },
                    }
                )
            processed_count += 1
    except HiddenFutureConceptError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        raise HiddenFutureConceptError(
            f"cannot evaluate streamed hidden-future rows: {error}"
        ) from error
    finally:
        for iterator in (prompt_iterator, experiment_iterator):
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    source_contract = config["source_contract"]
    _require(
        processed_count == len(alignment_rows) == source_contract["all_boundary_count"],
        "streamed boundary count changed",
    )
    _require(
        recomputed_stable_count == source_contract["stable_boundary_count"],
        "streamed stable-boundary count changed",
    )
    _require(
        future_hidden_boundary_count
        == source_contract["future_hidden_boundary_count_before_stability_filter"],
        "future-hidden boundary support changed",
    )
    _require(
        eligible_target_boundary_count
        == source_contract["eligible_target_boundary_instances_before_stability_filter"],
        "eligible target-boundary support changed",
    )
    _require(bool(scored_rows), "no stable hidden-future target boundaries remain")
    try:
        v3.HISTORICAL_V1._validate_report_provenance(
            report_metadata, protocol=replay_protocol["report_helper_protocol"]
        )
    except (ValueError, TypeError, KeyError) as error:
        raise HiddenFutureConceptError(
            f"public report provenance changed: {error}"
        ) from error
    return {
        "rows": scored_rows,
        "coverage": {
            "all_boundary_count": processed_count,
            "stable_boundary_count": recomputed_stable_count,
            "future_hidden_boundary_count_before_stability_filter": future_hidden_boundary_count,
            "eligible_target_boundary_instances_before_stability_filter": eligible_target_boundary_count,
            "stable_future_hidden_boundary_count": stable_future_hidden_boundary_count,
            "stable_eligible_target_boundary_instance_count": stable_eligible_target_boundary_count,
            "scored_target_boundary_instance_count": len(scored_rows),
        },
        "report_provenance_sha256": sha256_json(report_metadata),
    }


def verify_authenticated_paths_unchanged(
    paths: Sequence[Path], records: Sequence[Mapping[str, Any]], *, label: str
) -> None:
    _require(len(paths) == len(records), f"{label} path registry changed")
    for position, (path, record) in enumerate(zip(paths, records, strict=True)):
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == record["size_bytes"]
            and sha256_file(path) == record["sha256"],
            f"{label} {position} changed during evaluation",
        )


def _write_json_no_clobber(path: Path, value: Any) -> None:
    """Publish canonical JSON with an atomic no-clobber hard-link commit."""

    _require(not path.exists() and not path.is_symlink(), "refusing to overwrite output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _require(
        not temporary.exists() and not temporary.is_symlink(),
        "temporary output path already exists",
    )
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise HiddenFutureConceptError("refusing to overwrite output") from error
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def make_report(
    *,
    config: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    input_paths: Mapping[str, Path],
    point_metrics: Mapping[str, Mapping[str, float]],
    bootstrap: Mapping[str, Any],
    weights: np.ndarray,
) -> dict[str, Any]:
    rows = list(_sequence(evaluation["rows"], "scored rows"))
    repositories = sorted({str(row["repository"]) for row in rows})
    tasks = sorted({str(row["task_id_sha256"]) for row in rows})
    targets = sorted(
        {(str(row["task_id_sha256"]), str(row["future_target_id"])) for row in rows}
    )
    support = {
        "repository_count": len(repositories),
        "task_count": len(tasks),
        "future_target_count": len(targets),
        "scored_target_boundary_instance_count": len(rows),
        "minimum_repositories": config["support_gate"]["minimum_repositories"],
        "minimum_tasks": config["support_gate"]["minimum_tasks"],
    }
    support["passed"] = (
        support["repository_count"] >= support["minimum_repositories"]
        and support["task_count"] >= support["minimum_tasks"]
    )
    support["passing_support_does_not_establish_semantic_recovery"] = True
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "status": "passed" if support["passed"] else "failed_support_gate",
        "scope": "development_only_reserved_validation_closed",
        "config": {
            "path": str(CONFIG_PATH.relative_to(ROOT)),
            "sha256": sha256_file(CONFIG_PATH),
            "size_bytes": CONFIG_PATH.stat().st_size,
        },
        "implementation": {
            "path": str(SCRIPT_PATH.relative_to(ROOT)),
            "sha256": sha256_file(SCRIPT_PATH),
            "size_bytes": SCRIPT_PATH.stat().st_size,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ijson": "3.5.0",
            "bootstrap_bit_generator": "PCG64",
        },
        "inputs": {
            name: {
                "path": os.fspath(input_paths[name]),
                "sha256": config["inputs"][name]["sha256"],
                "size_bytes": config["inputs"][name]["size_bytes"],
            }
            for name in EXPECTED_INPUTS
        },
        "code_dependencies": [dict(record) for record in config["code_dependencies"]],
        "authentication": {
            "all_input_and_code_dependency_hashes_matched_before_scoring": True,
            "all_input_and_code_dependency_hashes_reverified_after_scoring": True,
            "prompt_report_payloads_bound_in_exact_order": True,
            "alignment_identity_and_stability_bound": True,
            "target_and_retained_foil_prefix_absence_recomputed": True,
            "future_support_contract_validated": True,
            "reserved_validation_access_authorized": False,
        },
        "coverage": dict(evaluation["coverage"]),
        "report_provenance_sha256": evaluation["report_provenance_sha256"],
        "support_gate": support,
        "scoring_contract": dict(config["scoring"]),
        "estimand": {
            "weighting": config["estimand"]["weighting"],
            "row_weight_sha256": hashlib.sha256(
                np.ascontiguousarray(weights.astype("<f8", copy=False)).tobytes(order="C")
            ).hexdigest(),
            "row_weight_sum": float(weights.sum()),
            "methods": dict(point_metrics),
        },
        "paired_bootstrap": dict(bootstrap),
        "identifier_ranking_chains": build_identifier_ranking_chains(
            rows, template=config["renderer"]["identifier_ranking_template"]
        ),
        "scored_rows": rows,
        "feature_and_label_separation": {
            "neutral_candidate_fields": ["candidate_id", "candidate_text", "forms"],
            "true_target_identity_used_only_by_evaluation": True,
            "target_or_foil_source_kind_used_by_prediction": False,
            "repository_or_task_identity_used_by_prediction": False,
            "completion_at_or_after_boundary_used_as_feature": False,
            "official_outcome_used": False,
            "proposition_relation_supplied_not_decoded": True,
            "semantic_concept_chain_decoded": False,
            "cot_or_cot_like_decoded": False,
        },
        "claim_scope": dict(config["claim_scope"]),
        "mandatory_limitations": list(config["mandatory_limitations"]),
        "forbidden_inputs_excluded": list(config["forbidden_inputs"]),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    # Lexical rejection is deliberately the first post-parse operation.
    lexical_path_preflight([args.config, args.output])
    canonical_path_preflight(input_paths=[args.config], output_paths=[args.output])
    config_path = args.config.resolve(strict=True)
    _require(config_path == CONFIG_PATH, "hidden-future config path changed")
    _require(
        sha256_file(config_path) == CONFIG_SHA256,
        "hidden-future config SHA-256 changed",
    )
    config = validate_config(load_json_strict(config_path))
    output_path = args.output.resolve(strict=False)
    _require(
        not output_path.exists() and not output_path.is_symlink(),
        "refusing to overwrite output",
    )

    input_records = [config["inputs"][name] for name in EXPECTED_INPUTS]
    dependency_records = [
        {key: record[key] for key in ("path", "sha256", "size_bytes")}
        for record in config["code_dependencies"]
    ]
    input_paths_list = authenticate_records(input_records, label="hidden-future input")
    dependency_paths = authenticate_records(
        dependency_records, label="hidden-future code dependency"
    )
    input_paths = dict(zip(EXPECTED_INPUTS, input_paths_list, strict=True))
    alignment_rows = validate_alignment_index(
        load_json_strict(input_paths["label_free_alignment_index"]),
        expected_total_count=config["source_contract"]["all_boundary_count"],
        expected_stable_count=config["source_contract"]["stable_boundary_count"],
    )
    replay_protocol = build_replay_validation_protocol(
        load_json_strict(input_paths["v3_protocol"]),
        load_json_strict(input_paths["v3_action_protocol"]),
    )
    v3_dependency = next(
        record
        for record in config["code_dependencies"]
        if record["role"] == "frozen_v3_streaming_analyzer"
    )
    v3_path = dependency_paths[
        [record["role"] for record in config["code_dependencies"]].index(
            "frozen_v3_streaming_analyzer"
        )
    ]
    v3 = load_pinned_v3_analyzer(v3_path, expected_sha256=v3_dependency["sha256"])
    evaluation = evaluate_streaming_sources(
        prompts_path=input_paths["development_prompts"],
        report_path=input_paths["development_public_report"],
        alignment_rows=alignment_rows,
        config=config,
        replay_protocol=replay_protocol,
        v3=v3,
    )
    rows = list(_sequence(evaluation["rows"], "scored rows"))
    weights = hierarchical_row_weights(rows)
    point_metrics = weighted_method_metrics(rows, weights)
    bootstrap = paired_repository_task_bootstrap(
        rows,
        point_metrics=point_metrics,
        draw_count=config["uncertainty"]["draw_count"],
        seed=config["uncertainty"]["seed"],
        confidence_level=config["uncertainty"]["confidence_level"],
    )
    verify_authenticated_paths_unchanged(
        input_paths_list, input_records, label="hidden-future input"
    )
    verify_authenticated_paths_unchanged(
        dependency_paths, dependency_records, label="hidden-future code dependency"
    )
    _require(
        sha256_file(config_path) == CONFIG_SHA256,
        "hidden-future config changed during evaluation",
    )
    report = make_report(
        config=config,
        evaluation=evaluation,
        input_paths=input_paths,
        point_metrics=point_metrics,
        bootstrap=bootstrap,
        weights=weights,
    )
    _write_json_no_clobber(output_path, report)
    print(
        f"wrote {len(rows)} stable hidden-future target-boundary audits to {output_path}"
    )
    return 0 if report["status"] == "passed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


__all__ = [
    "ALL_METRICS",
    "CONFIG_PATH",
    "CONFIG_SHA256",
    "EXPECTED_CLAIM_SCOPE",
    "HiddenFutureConceptError",
    "LAYERS",
    "MANDATORY_LIMITATIONS",
    "METHODS",
    "OPEN_VOCABULARY_TOP_K",
    "authenticate_records",
    "build_hidden_candidate_set",
    "build_parser",
    "build_identifier_ranking_chains",
    "build_replay_validation_protocol",
    "canonical_path_preflight",
    "evaluate_scored_candidate_set",
    "hierarchical_row_weights",
    "identifier_exposure",
    "lexical_path_preflight",
    "load_json_strict",
    "logmeanexp",
    "paired_repository_task_bootstrap",
    "render_identifier_ranking",
    "run",
    "score_candidate_set",
    "sha256_file",
    "sha256_text",
    "validate_alignment_index",
    "validate_config",
    "validate_future_support",
    "validate_prompt_report_binding",
    "weighted_method_metrics",
]


if __name__ == "__main__":
    raise SystemExit(main())
