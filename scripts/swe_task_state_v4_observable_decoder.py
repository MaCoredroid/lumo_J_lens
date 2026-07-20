#!/usr/bin/env python3
"""Repository-held-out decoders for frozen, visible SWE process events.

The numeric feature bundle is label-free.  This module authenticates and
validates that bundle before it opens the physically separate observable-label
sidecar.  Repository and identity fields are used only for alignment,
weighting, and outer folds; they are never admitted to a model matrix.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_observable_decoder.json"
SCRIPT_PATH = Path(__file__).resolve()
SCHEMA_VERSION = 1
FEATURE_BUNDLE_KIND = (
    "swe_task_state_v4_label_free_observable_decoder_feature_bundle"
)
FEATURE_BUNDLE_SCHEMA_VERSION = 2
FEATURE_BUNDLE_PRODUCER_PATH = (
    ROOT / "scripts" / "swe_task_state_v4_observable_feature_bundle.py"
)
REPORT_KIND = "swe_task_state_v4_repository_held_out_observable_decoder_report"
FROZEN_FORBIDDEN_PATH_FRAGMENTS = ("reserved", "validation")

BASE_BLOCK_WIDTHS = {
    "history_only": 14,
    "sequence_j": 136,
    "sequence_logit": 136,
    "sequence_logit_j": 256,
    "raw_activation_current": 192,
    "public_j_activation_current": 192,
    "raw_activation_sequence": 578,
    "public_j_activation_sequence": 578,
    "raw_public_j_activation_sequence": 1154,
}
METADATA_ARRAYS = (
    "global_index",
    "source_id_sha256",
    "task_id_sha256",
    "repository",
    "request_index",
)
VISIBLE_BASELINE_BLOCKS = (
    "history_only",
    "sequence_j",
    "sequence_logit",
    "sequence_logit_j",
)
ACTIVATION_BLOCKS = (
    "raw_activation_current",
    "public_j_activation_current",
    "raw_activation_sequence",
    "public_j_activation_sequence",
    "raw_public_j_activation_sequence",
)
SOURCE_FEATURE_BLOCKS = {
    "visible_baselines": [
        "history_only",
        "sequence_logit",
        "sequence_j",
        "sequence_logit_j",
    ],
    "activation_features": [
        "public_j_activation_current",
        "public_j_activation_sequence",
        "raw_activation_current",
        "raw_activation_sequence",
        "raw_public_j_activation_sequence",
    ],
}
SOURCE_ARRAY_KEYS = {
    role: ["global_index", *blocks]
    for role, blocks in SOURCE_FEATURE_BLOCKS.items()
}
UPSTREAM_DESCRIPTIVE_DTYPES = {
    "little-endian-int64": "<i8",
    "little-endian-float64": "<f8",
}
PRIMARY_TARGET = "observable_rationale_language_marker"
PRIMARY_HORIZON = (
    "feature_h_t_is_immediately_before_ensuing_visible_completion_t_and_"
    "target_uses_only_completion_t"
)
MANDATORY_LIMITATIONS = (
    "The primary label is a frozen surface-language marker in the ensuing visible assistant completion and only an observable process-trace proxy; it is not a diagnosis, private chain-of-thought, hidden thought, understanding, intent, or causal explanation.",
    "No emotion label is fitted or inferred, and these results provide no evidence of emotion decoding.",
    "Public-J states are deterministic functions of raw residual states under the pinned public map; their comparison measures representation and inductive bias, not access to additional hidden information.",
    "The ordinary and public-J comparison matrices are existing action-word-probe sequence baselines; the sparse retrospective concept ontology is not an all-row comparator.",
    "Paired held-out point-metric differences alone do not establish incremental value or statistical reliability.",
    "The nested augmented variants and comparisons were added after initial out-of-fold results were observed and were not preregistered; no full-refit or model-or-feature-selection uncertainty is included, and no untouched confirmation set was evaluated.",
    "Passed status records execution and contract-check completion only; it is not a scientific validity, reliability, confirmation, or transport gate.",
)

STATUS_SCOPE = (
    "execution_and_contract_checks_passed_only_not_a_scientific_validity_gate"
)
ANALYSIS_CHRONOLOGY = {
    "nested_augmented_variants_added_after_initial_oof_results_were_observed": True,
    "nested_comparisons_preregistered": False,
    "post_hoc_exploratory_only": True,
    "full_refit_uncertainty_included": False,
    "model_or_feature_selection_uncertainty_included": False,
    "untouched_confirmation_set_evaluated": False,
}


class DecoderError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DecoderError(f"duplicate JSON key: {key!r}")
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


def _display_path(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _stable_file_state(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _artifact_byte_record(path: Path, *, label: str) -> dict[str, Any]:
    """Hash one stable regular-file generation and return its exact byte record."""

    _require(not path.is_symlink(), f"{label} must not be a symlink")
    resolved = path.resolve(strict=True)
    _require(resolved.is_file(), f"{label} is not a regular file")
    before = _stable_file_state(resolved)
    digest = sha256_file(resolved)
    after = _stable_file_state(resolved)
    _require(before == after, f"{label} changed while its byte record was captured")
    return {
        "path": _display_path(resolved),
        "sha256": digest,
        "size_bytes": after[2],
    }


def _read_startup_bytes(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    """Read the exact startup bytes used by the run and bind their file generation."""

    _require(not path.is_symlink(), f"{label} must not be a symlink")
    resolved = path.resolve(strict=True)
    _require(resolved.is_file(), f"{label} is not a regular file")
    with resolved.open("rb") as handle:
        before_stat = os.fstat(handle.fileno())
        payload = handle.read()
        after_stat = os.fstat(handle.fileno())
    path_state = _stable_file_state(resolved)
    descriptor_before = (
        before_stat.st_dev,
        before_stat.st_ino,
        before_stat.st_size,
        before_stat.st_mtime_ns,
    )
    descriptor_after = (
        after_stat.st_dev,
        after_stat.st_ino,
        after_stat.st_size,
        after_stat.st_mtime_ns,
    )
    _require(
        descriptor_before == descriptor_after == path_state
        and len(payload) == before_stat.st_size,
        f"{label} changed while startup bytes were captured",
    )
    return payload, {
        "path": _display_path(resolved),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _assert_byte_records_unchanged(
    bindings: Mapping[str, tuple[Path, Mapping[str, Any]]],
) -> None:
    """Rehash every authenticated file generation immediately before commit."""

    _require(bool(bindings), "precommit byte-binding registry is empty")
    for label, (path, expected_value) in bindings.items():
        expected = dict(expected_value)
        _require(
            set(expected) == {"path", "sha256", "size_bytes"}
            and _is_sha256(expected["sha256"])
            and isinstance(expected["size_bytes"], int)
            and not isinstance(expected["size_bytes"], bool)
            and expected["size_bytes"] >= 0,
            f"{label} startup byte record is invalid",
        )
        observed = _artifact_byte_record(path, label=label)
        _require(observed == expected, f"{label} changed before report commit")


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return sha256_bytes(header + b"\0" + array.tobytes(order="C"))


def source_logical_array_sha256(name: str, value: np.ndarray) -> str:
    """Match the logical-array digest emitted by the two source producers."""

    array = np.ascontiguousarray(value)
    header = canonical_json_bytes(
        {"name": name, "shape": list(array.shape), "dtype": array.dtype.str}
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DecoderError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _lexical_path_has_forbidden_fragment(path: Path) -> bool:
    normalized = os.path.normpath(os.path.abspath(os.fspath(path))).lower()
    return any(
        fragment in component
        for fragment in FROZEN_FORBIDDEN_PATH_FRAGMENTS
        for component in Path(normalized).parts
    )


def frozen_lexical_path_preflight(paths: Iterable[Path | None]) -> None:
    """Reject forbidden path text before any filesystem operation."""

    for path in paths:
        if path is not None and _lexical_path_has_forbidden_fragment(Path(path)):
            raise DecoderError(
                f"forbidden path rejected before filesystem access: {path}"
            )


def frozen_canonical_path_preflight(
    *, input_paths: Iterable[Path | None], output_paths: Iterable[Path | None]
) -> None:
    """Resolve metadata, then reject forbidden canonical parents before reads."""

    for path, strict in [
        *((path, True) for path in input_paths),
        *((path, False) for path in output_paths),
    ]:
        if path is None:
            continue
        try:
            resolved = Path(path).resolve(strict=strict)
        except OSError as error:
            raise DecoderError(f"cannot resolve decoder path: {path}: {error}") from error
        lowered = [part.lower() for part in resolved.parts]
        if any(
            fragment in component
            for fragment in FROZEN_FORBIDDEN_PATH_FRAGMENTS
            for component in lowered
        ):
            raise DecoderError(
                f"forbidden canonical path rejected before file read or hash: {path}"
            )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)


def _load_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DecoderError(f"cannot load strict {label} JSON bytes: {error}") from error


def _variant_identity() -> list[dict[str, Any]]:
    return [
        {
            "name": "history_only",
            "role": "visible_history_baseline",
            "components": ["history_only"],
            "width": 14,
        },
        {
            "name": "sequence_logit",
            "role": "ordinary_action_word_probe_sequence_baseline",
            "components": ["sequence_logit"],
            "width": 136,
        },
        {
            "name": "sequence_j",
            "role": "public_j_action_word_probe_sequence_baseline",
            "components": ["sequence_j"],
            "width": 136,
        },
        {
            "name": "sequence_logit_j",
            "role": "combined_action_word_probe_sequence_baseline",
            "components": ["sequence_logit_j"],
            "width": 256,
        },
        {
            "name": "raw_activation_current",
            "role": "activation_candidate",
            "components": ["raw_activation_current"],
            "width": 192,
        },
        {
            "name": "public_j_activation_current",
            "role": "activation_candidate",
            "components": ["public_j_activation_current"],
            "width": 192,
        },
        {
            "name": "sequence_logit_j_plus_raw_activation_current",
            "role": "visible_baseline_plus_activation_candidate",
            "components": ["sequence_logit_j", "raw_activation_current"],
            "width": 448,
        },
        {
            "name": "sequence_logit_j_plus_public_j_activation_current",
            "role": "visible_baseline_plus_activation_candidate",
            "components": ["sequence_logit_j", "public_j_activation_current"],
            "width": 448,
        },
        {
            "name": "raw_activation_sequence",
            "role": "activation_candidate",
            "components": ["raw_activation_sequence"],
            "width": 578,
        },
        {
            "name": "public_j_activation_sequence",
            "role": "activation_candidate",
            "components": ["public_j_activation_sequence"],
            "width": 578,
        },
        {
            "name": "raw_public_j_activation_sequence",
            "role": "activation_candidate",
            "components": ["raw_public_j_activation_sequence"],
            "width": 1154,
        },
        {
            "name": "sequence_logit_plus_raw_public_j_activation_sequence",
            "role": "visible_baseline_plus_activation_candidate",
            "components": [
                "sequence_logit",
                "raw_public_j_activation_sequence",
            ],
            "width": 1290,
        },
    ]


def validate_config(value: Any) -> dict[str, Any]:
    config = dict(_mapping(value, "observable-decoder config"))
    _require(
        set(config)
        == {
            "schema_version",
            "id",
            "status",
            "status_scope",
            "path_guard",
            "artifacts",
            "target_policy",
            "split",
            "weighting",
            "model",
            "variants",
            "paired_comparisons",
            "analysis_chronology",
            "metrics",
            "claim_scope",
            "mandatory_limitations",
        },
        "observable-decoder config schema changed",
    )
    _require(
        config["schema_version"] == 1
        and config["id"]
        == "swe-task-state-v4-repository-held-out-observable-event-decoder"
        and config["status"] == "development_only_reserved_validation_closed"
        and config["status_scope"] == STATUS_SCOPE,
        "observable-decoder config identity changed",
    )
    _require(
        config["path_guard"]
        == {"forbidden_path_fragments": ["reserved", "validation"]},
        "observable-decoder path guard changed",
    )
    artifacts = _mapping(config["artifacts"], "artifact contract")
    _require(
        artifacts.get("alignment_index_kind")
        == "swe_task_state_v4_label_free_alignment_index"
        and artifacts.get("label_sidecar_kind")
        == "swe_task_state_v4_observable_event_label_sidecar"
        and artifacts.get("feature_bundle_kind") == FEATURE_BUNDLE_KIND
        and artifacts.get("feature_bundle_scope")
        == "numeric_features_and_grouping_only_no_labels_or_outcomes"
        and artifacts.get("feature_bundle_format") == "npz_allow_pickle_false"
        and artifacts.get("row_count") == 1606
        and artifacts.get("required_source_roles")
        == ["visible_baselines", "activation_features"]
        and artifacts.get("required_source_kinds")
        == {
            "visible_baselines": (
                "swe_task_state_v4_visible_word_probe_baseline_features"
            ),
            "activation_features": (
                "swe_task_state_v4_label_free_activation_feature_campaign"
            ),
        }
        and artifacts.get("source_path_base")
        == "repository_root_unless_absolute"
        and artifacts.get("metadata_arrays") == list(METADATA_ARRAYS)
        and artifacts.get("base_feature_blocks") == BASE_BLOCK_WIDTHS,
        "observable-decoder artifact contract changed",
    )
    _require(
        artifacts.get("visible_baseline_semantics")
        == {
            "history_only": "14 frozen causal-history features",
            "sequence_logit": "ordinary action-word-probe sequence baseline: 96 current action-class token-score features transformed into compact current/delta/prior-EMA blocks, plus 14 causal-history and 2 task-gap features",
            "sequence_j": "public-J action-word-probe sequence baseline: 96 current action-class token-score features transformed into compact current/delta/prior-EMA blocks, plus 14 causal-history and 2 task-gap features",
            "sequence_logit_j": "combined ordinary/public-J action-word-probe sequence baseline with both compact temporal blocks, plus 14 causal-history and 2 task-gap features",
        },
        "visible action-word-probe baseline semantics changed",
    )
    forbidden = artifacts.get("forbidden_array_fields")
    _require(
        isinstance(forbidden, list)
        and len(forbidden) == len(set(forbidden))
        and {
            "label",
            "target",
            "outcome",
            "completion_text",
            "emotion",
        }.issubset(forbidden),
        "forbidden feature registry changed",
    )
    _require(
        artifacts.get("grouping_arrays_never_numeric_features")
        == [
            "repository",
            "request_index",
            "source_id_sha256",
            "task_id_sha256",
            "global_index",
        ],
        "grouping-only feature registry changed",
    )
    target = _mapping(config["target_policy"], "target policy")
    _require(
        target.get("primary_target") == PRIMARY_TARGET
        and target.get("primary_horizon") == PRIMARY_HORIZON
        and target.get("primary_semantics")
        == "frozen_visible_completion_surface_language_regex_marker_observable_process_trace_proxy"
        and target.get("unknown_status_excluded_never_coerced_to_negative") is True
        and target.get("private_or_hidden_semantics_forbidden") is True
        and PRIMARY_TARGET in target.get("allowed_targets", []),
        "observable target or horizon changed",
    )
    _require(
        config["split"]
        == {
            "algorithm": "leave_one_repository_out",
            "repository_never_used_as_numeric_feature": True,
            "heldout_labels_never_used_for_fit_or_standardization": True,
            "minimum_repositories": 2,
            "repository_order": "sorted_utf8",
        },
        "outer split contract changed",
    )
    _require(
        config["weighting"]
        == {
            "estimand": "equal_repository_then_equal_task_within_repository_then_equal_available_row_within_task",
            "training_weights_recomputed_on_outer_training_rows_only": True,
            "fit_weight_mass": "rescale_unit_mass_hierarchical_weights_to_training_row_count",
            "class_weights": None,
            "same_evaluation_rows_and_weights_for_every_variant": True,
        },
        "observable-decoder weighting contract changed",
    )
    model = _mapping(config["model"], "model contract")
    _require(
        model
        == {
            "family": "weighted_l2_logistic_regression",
            "implementation": "sklearn.linear_model.LogisticRegression",
            "C": 0.01,
            "solver": "lbfgs",
            "l1_ratio": 0.0,
            "fit_intercept": True,
            "max_iter": 4000,
            "tol": 1e-10,
            "random_state": 0,
            "class_weight": None,
            "standardization": "training_hierarchical_weighted_mean_and_variance_only",
            "zero_variance_scale": 1.0,
            "probability_floor": 1e-12,
            "single_class_training_fold": "explicit_constant_observed_training_class_predictor",
            "partially_supported_training_fold": "fit_supported_classes_and_floor_absent_class_probabilities",
        },
        "fixed-C logistic model contract changed",
    )
    _require(config["variants"] == _variant_identity(), "variant registry changed")
    comparisons = _mapping(config["paired_comparisons"], "comparison contract")
    nested = comparisons.get("nested_comparisons")
    _require(
        comparisons.get("candidate_roles")
        == ["activation_candidate", "visible_baseline_plus_activation_candidate"]
        and comparisons.get("reference_baselines")
        == ["history_only", "sequence_logit", "sequence_logit_j"]
        and nested
        == [
            {
                "candidate": "sequence_logit_j_plus_raw_activation_current",
                "reference": "sequence_logit_j",
            },
            {
                "candidate": "sequence_logit_j_plus_public_j_activation_current",
                "reference": "sequence_logit_j",
            },
        ]
        and comparisons.get(
            "identical_oof_rows_labels_weights_and_folds_required"
        )
        is True
        and comparisons.get(
            "point_deltas_only_do_not_establish_incremental_value"
        )
        is True,
        "paired comparison contract changed",
    )
    variants_by_name = {spec["name"]: spec for spec in config["variants"]}
    for comparison in nested:
        candidate = variants_by_name[comparison["candidate"]]
        reference = variants_by_name[comparison["reference"]]
        _require(
            set(reference["components"]) < set(candidate["components"]),
            "nested comparison reference is not a strict component subset",
        )
    _require(
        config["analysis_chronology"] == ANALYSIS_CHRONOLOGY,
        "observable-decoder analysis chronology changed",
    )
    claims = _mapping(config["claim_scope"], "claim scope")
    _require(
        set(claims)
        == {
            "primary_target_is_visible_surface_language_marker_only",
            "private_chain_of_thought_reconstructed",
            "cot_or_cot_like_decoding_established",
            "semantic_sentence_or_chain_decoding_established",
            "hidden_thought_understanding_or_intent_established",
            "emotion_decoding_established",
            "causal_interpretation_established",
            "retrospective_sparse_concept_ontology_used_as_all_row_baseline",
            "point_metric_superiority_establishes_incremental_value",
        }
        and claims.get("primary_target_is_visible_surface_language_marker_only") is True
        and all(
            claims.get(key) is False
            for key in (
                "private_chain_of_thought_reconstructed",
                "cot_or_cot_like_decoding_established",
                "semantic_sentence_or_chain_decoding_established",
                "hidden_thought_understanding_or_intent_established",
                "emotion_decoding_established",
                "causal_interpretation_established",
                "retrospective_sparse_concept_ontology_used_as_all_row_baseline",
                "point_metric_superiority_establishes_incremental_value",
            )
        ),
        "observable-decoder claim scope changed",
    )
    _require(
        config["mandatory_limitations"] == list(MANDATORY_LIMITATIONS),
        "mandatory limitation text changed",
    )
    return config


def validate_alignment_index(value: Any) -> list[dict[str, Any]]:
    index = _mapping(value, "alignment index")
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
        index["schema_version"] == 1
        and index["kind"] == "swe_task_state_v4_label_free_alignment_index"
        and index["status"] == "passed"
        and index["scope"] == "grouping_order_and_stability_only_no_labels"
        and index["row_count"] == 1708
        and index["stable_row_count"] == 1606,
        "alignment index identity or counts changed",
    )
    rows = index["rows"]
    _require(isinstance(rows, list) and len(rows) == 1708, "alignment rows changed")
    keys = {
        "global_index",
        "source_id_sha256",
        "task_id_sha256",
        "repository",
        "request_index",
        "stable_feature_eligible",
    }
    seen_sources: set[str] = set()
    seen_requests: set[tuple[str, str, int]] = set()
    repository_by_task: dict[str, str] = {}
    previous_request_by_task: dict[tuple[str, str], int] = {}
    request_indices_by_task: dict[tuple[str, str], list[int]] = {}
    stable_count = 0
    for position, raw in enumerate(rows):
        row = _mapping(raw, f"alignment row {position}")
        source = row.get("source_id_sha256")
        task = row.get("task_id_sha256")
        repository = row.get("repository")
        request = row.get("request_index")
        _require(set(row) == keys, f"alignment row {position} fields changed")
        _require(
            row.get("global_index") == position
            and _is_sha256(source)
            and _is_sha256(task)
            and isinstance(repository, str)
            and bool(repository)
            and isinstance(request, int)
            and not isinstance(request, bool)
            and request >= 1
            and isinstance(row.get("stable_feature_eligible"), bool),
            f"alignment row {position} is invalid",
        )
        task_text = str(task)
        repository_text = str(repository)
        if (
            task_text in repository_by_task
            and repository_by_task[task_text] != repository_text
        ):
            raise DecoderError("one alignment task maps to multiple repositories")
        repository_by_task[task_text] = repository_text
        task_key = (repository_text, task_text)
        identity = (*task_key, int(request))
        previous_request = previous_request_by_task.get(task_key)
        _require(
            str(source) not in seen_sources
            and identity not in seen_requests
            and (previous_request is None or int(request) > previous_request),
            "alignment identities or task request order changed",
        )
        seen_sources.add(str(source))
        seen_requests.add(identity)
        previous_request_by_task[task_key] = int(request)
        request_indices_by_task.setdefault(task_key, []).append(int(request))
        stable_count += int(row["stable_feature_eligible"])
    for request_indices in request_indices_by_task.values():
        _require(
            request_indices == list(range(1, len(request_indices) + 1)),
            "alignment task requests are not complete and consecutive",
        )
    _require(stable_count == 1606, "alignment stable count changed")
    return [dict(row) for row in rows]


def validate_label_sidecar(
    value: Any,
    *,
    alignment_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sidecar = _mapping(value, "observable-label sidecar")
    _require(
        set(sidecar)
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
            "support",
            "temporal_contract",
            "target_contracts",
            "positive_control_contracts",
            "forbidden_inputs",
            "claim_scope",
            "rows",
        },
        "observable-label sidecar schema changed",
    )
    artifacts = config["artifacts"]
    _require(
        sidecar["schema_version"] == 1
        and sidecar["kind"] == artifacts["label_sidecar_kind"]
        and sidecar["status"] == "passed"
        and sidecar["scope"] == "labels_and_grouping_only_never_feature_input"
        and sidecar["row_count"] == 1708
        and sidecar["stable_row_count"] == 1606,
        "observable-label sidecar identity or counts changed",
    )
    temporal = _mapping(sidecar["temporal_contract"], "temporal contract")
    _require(
        temporal.get("index_symbol") == "t"
        and temporal.get("feature_cutoff")
        == "h_t is the final captured prompt token immediately before ensuing assistant completion t"
        and _mapping(
            temporal.get("target_completion_offsets"),
            "target completion offsets",
        ).get(PRIMARY_TARGET)
        == ["t"]
        and temporal.get("completion_t_or_later_as_feature_forbidden") is True,
        "primary observable marker horizon changed",
    )
    contracts = _mapping(sidecar["target_contracts"], "target contracts")
    _require(
        set(config["target_policy"]["allowed_targets"]) <= set(contracts),
        "observable-label sidecar omits an allowed target",
    )
    primary_contract = _mapping(
        contracts[PRIMARY_TARGET], "primary observable-marker contract"
    )
    _require(
        primary_contract
        == {
            "classes": ["no", "yes"],
            "definition": "ensuing materialized assistant completion t matches at least one frozen broad surface-language regex historically stored as diagnosis_expressed; patterns include diagnostic and action-plan wording",
            "cot_like_scope": "observable process-trace proxy only, not a diagnosis, reasoning, understanding, or hidden/private chain-of-thought",
        },
        "primary observable-marker semantics changed",
    )
    _require(
        {
            "completion_text_as_feature",
            "diagnosis_regex_hits_as_feature",
            "observable_rationale_language_label_as_feature",
            "completion_t_or_later_as_feature",
            "private_reasoning",
            "emotion_label",
        }.issubset(set(sidecar["forbidden_inputs"]))
        and all(value is False for value in sidecar["claim_scope"].values()),
        "observable-label sidecar feature boundary or claim scope changed",
    )
    rows = sidecar["rows"]
    _require(
        isinstance(rows, list) and len(rows) == len(alignment_rows) == 1708,
        "observable-label sidecar rows changed",
    )
    grouping_keys = (
        "global_index",
        "source_id_sha256",
        "task_id_sha256",
        "repository",
        "request_index",
        "stable_feature_eligible",
    )
    expected_row_keys = {*grouping_keys, "targets", "positive_controls"}
    result = []
    for position, (raw, alignment) in enumerate(zip(rows, alignment_rows)):
        row = _mapping(raw, f"observable-label row {position}")
        _require(
            set(row) == expected_row_keys,
            f"observable-label row {position} fields changed",
        )
        _require(
            all(row[key] == alignment[key] for key in grouping_keys),
            f"observable-label row {position} is not exactly aligned",
        )
        targets = _mapping(row["targets"], f"observable targets {position}")
        _require(
            set(targets) == set(contracts),
            f"observable target registry changed at row {position}",
        )
        for target_name, contract_value in contracts.items():
            contract = _mapping(contract_value, f"target contract {target_name}")
            classes = contract.get("classes")
            _require(
                isinstance(classes, list)
                and len(classes) >= 2
                and len(classes) == len(set(classes))
                and all(isinstance(item, str) and item for item in classes),
                f"target classes are invalid: {target_name}",
            )
            record = _mapping(targets[target_name], f"target {target_name} row {position}")
            _require(
                set(record) == {"status", "value", "reason"},
                f"target record fields changed: {target_name}",
            )
            if record["status"] == "available":
                _require(
                    record["value"] in classes and record["reason"] is None,
                    f"available target value is invalid: {target_name}",
                )
            else:
                _require(
                    record["status"] == "unknown"
                    and record["value"] is None
                    and isinstance(record["reason"], str)
                    and bool(record["reason"]),
                    f"unknown target record is invalid: {target_name}",
                )
        result.append(dict(row))
    return result


def sanitize_visible_baseline_rows(
    extracted_rows: Sequence[Mapping[str, Any]],
    *,
    alignment_rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Project authenticated V4 extraction rows into label-free matrices.

    Only ``row_id`` and the four frozen numeric feature variants are read.
    Label, action, auxiliary-diagnostic, completion, and outcome fields that
    may coexist in the extractor row are deliberately ignored and cannot be
    emitted by this API.
    """

    stable = [row for row in alignment_rows if row["stable_feature_eligible"]]
    _require(len(stable) == 1606, "baseline seam requires 1606 stable rows")
    _require(
        isinstance(extracted_rows, Sequence)
        and not isinstance(extracted_rows, (str, bytes, bytearray))
        and len(extracted_rows) == len(stable),
        "visible baseline extraction row count changed",
    )
    by_source: dict[str, Mapping[str, Any]] = {}
    for position, raw in enumerate(extracted_rows):
        row = _mapping(raw, f"visible baseline source row {position}")
        row_id = row.get("row_id")
        _require(
            isinstance(row_id, str) and bool(row_id),
            f"visible baseline row {position} lacks row_id",
        )
        source_hash = sha256_text(row_id)
        _require(source_hash not in by_source, "duplicate visible baseline row identity")
        by_source[source_hash] = row
    _require(
        set(by_source) == {str(row["source_id_sha256"]) for row in stable},
        "visible baseline identities differ from stable alignment",
    )
    outputs: dict[str, list[np.ndarray]] = {
        name: [] for name in VISIBLE_BASELINE_BLOCKS
    }
    for alignment in stable:
        row = by_source[str(alignment["source_id_sha256"])]
        features = _mapping(row.get("features"), "visible baseline features")
        _require(
            set(features) == set(VISIBLE_BASELINE_BLOCKS),
            "visible baseline feature registry changed",
        )
        for name in VISIBLE_BASELINE_BLOCKS:
            matrix_row = np.asarray(features[name], dtype=np.float64)
            _require(
                matrix_row.shape == (BASE_BLOCK_WIDTHS[name],)
                and bool(np.all(np.isfinite(matrix_row))),
                f"visible baseline block is invalid: {name}",
            )
            outputs[name].append(matrix_row)
    result = {
        name: np.asarray(rows, dtype=np.float64) for name, rows in outputs.items()
    }
    result["global_index"] = np.asarray(
        [row["global_index"] for row in stable], dtype=np.int64
    )
    return result


def assemble_label_free_feature_arrays(
    *,
    alignment_rows: Sequence[Mapping[str, Any]],
    activation_features: Mapping[str, np.ndarray],
    visible_baselines: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Join two label-free numeric seams; no label argument is accepted."""

    stable = [row for row in alignment_rows if row["stable_feature_eligible"]]
    _require(len(stable) == 1606, "feature assembly requires 1606 stable rows")
    _require(
        set(activation_features) == {*ACTIVATION_BLOCKS, "global_index"},
        "activation feature artifact keys changed",
    )
    _require(
        set(visible_baselines) == {*VISIBLE_BASELINE_BLOCKS, "global_index"},
        "visible baseline artifact keys changed",
    )
    global_index = np.asarray(activation_features["global_index"])
    visible_global_index = np.asarray(visible_baselines["global_index"])
    expected_global = np.asarray(
        [row["global_index"] for row in stable], dtype=np.int64
    )
    _require(
        global_index.dtype == np.int64
        and visible_global_index.dtype == np.int64
        and np.array_equal(global_index, expected_global)
        and np.array_equal(visible_global_index, expected_global),
        "source retained global indices differ from stable alignment",
    )
    arrays: dict[str, np.ndarray] = {
        "global_index": expected_global,
        "source_id_sha256": np.asarray(
            [row["source_id_sha256"] for row in stable], dtype="<U64"
        ),
        "task_id_sha256": np.asarray(
            [row["task_id_sha256"] for row in stable], dtype="<U64"
        ),
        "repository": np.asarray([row["repository"] for row in stable], dtype=str),
        "request_index": np.asarray(
            [row["request_index"] for row in stable], dtype=np.int64
        ),
    }
    for source in (visible_baselines, activation_features):
        for name, width in BASE_BLOCK_WIDTHS.items():
            if name not in source:
                continue
            values = np.asarray(source[name])
            _require(
                values.dtype == np.float64
                and values.shape == (1606, width)
                and bool(np.all(np.isfinite(values))),
                f"feature block geometry, dtype, or finiteness changed: {name}",
            )
            arrays[name] = np.ascontiguousarray(values)
    _require(
        set(arrays) == {*METADATA_ARRAYS, *BASE_BLOCK_WIDTHS},
        "assembled feature array registry changed",
    )
    return arrays


def validate_feature_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    alignment_rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    expected_keys = {*METADATA_ARRAYS, *BASE_BLOCK_WIDTHS}
    _require(
        set(arrays) == expected_keys,
        "feature bundle array allowlist changed or contains a forbidden field",
    )
    stable = [row for row in alignment_rows if row["stable_feature_eligible"]]
    _require(len(stable) == 1606, "feature bundle stable alignment changed")
    result = {key: np.asarray(value) for key, value in arrays.items()}
    expected_metadata = {
        "global_index": np.asarray(
            [row["global_index"] for row in stable], dtype=np.int64
        ),
        "source_id_sha256": np.asarray(
            [row["source_id_sha256"] for row in stable], dtype="<U64"
        ),
        "task_id_sha256": np.asarray(
            [row["task_id_sha256"] for row in stable], dtype="<U64"
        ),
        "repository": np.asarray([row["repository"] for row in stable], dtype=str),
        "request_index": np.asarray(
            [row["request_index"] for row in stable], dtype=np.int64
        ),
    }
    for name, expected in expected_metadata.items():
        observed = result[name]
        _require(
            observed.ndim == 1
            and observed.shape == (1606,)
            and observed.dtype.kind != "O"
            and (
                observed.dtype == np.int64
                if name in {"global_index", "request_index"}
                else observed.dtype == np.dtype("<U64")
                if name in {"source_id_sha256", "task_id_sha256"}
                else observed.dtype.kind == "U"
            )
            and np.array_equal(observed, expected),
            f"feature bundle metadata does not exactly match alignment: {name}",
        )
    for name, width in BASE_BLOCK_WIDTHS.items():
        matrix = result[name]
        _require(
            matrix.dtype == np.float64
            and matrix.shape == (1606, width)
            and bool(np.all(np.isfinite(matrix))),
            f"feature bundle numeric block is invalid: {name}",
        )
        result[name] = np.ascontiguousarray(matrix)
    return result


def build_variant_matrices(
    arrays: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Build X solely from the frozen numeric block registry."""

    matrices: dict[str, np.ndarray] = {}
    for spec in config["variants"]:
        components = spec["components"]
        _require(
            all(component in BASE_BLOCK_WIDTHS for component in components),
            f"variant contains a nonnumeric or forbidden component: {spec['name']}",
        )
        matrix = np.concatenate(
            [np.asarray(arrays[component], dtype=np.float64) for component in components],
            axis=1,
        )
        _require(
            matrix.shape == (1606, spec["width"])
            and bool(np.all(np.isfinite(matrix))),
            f"variant matrix changed: {spec['name']}",
        )
        matrices[spec["name"]] = matrix
    return matrices


def _validate_artifact_record(value: Any, label: str) -> dict[str, Any]:
    record = dict(_mapping(value, label))
    _require(
        set(record) == {"path", "sha256", "size_bytes"}
        and isinstance(record["path"], str)
        and bool(record["path"])
        and _is_sha256(record["sha256"])
        and isinstance(record["size_bytes"], int)
        and not isinstance(record["size_bytes"], bool)
        and record["size_bytes"] > 0
        and not _lexical_path_has_forbidden_fragment(Path(record["path"])),
        f"{label} is invalid or forbidden",
    )
    return record


def _expected_feature_array_record(name: str) -> tuple[list[int], str]:
    if name in {"global_index", "request_index"}:
        return [1606], "<i8"
    if name in {"source_id_sha256", "task_id_sha256"}:
        return [1606], "<U64"
    if name == "repository":
        return [1606], "<U25"
    _require(name in BASE_BLOCK_WIDTHS, f"unknown feature array: {name}")
    return [1606, BASE_BLOCK_WIDTHS[name]], "<f8"


def _validate_array_records(
    value: Any, *, names: Sequence[str], label: str
) -> dict[str, Any]:
    records = dict(_mapping(value, label))
    _require(set(records) == set(names), f"{label} registry changed")
    normalized: dict[str, Any] = {}
    for name in names:
        record = dict(_mapping(records[name], f"{label} {name}"))
        shape, dtype = _expected_feature_array_record(name)
        _require(
            set(record) == {"shape", "dtype", "logical_sha256"}
            and record["shape"] == shape
            and record["dtype"] == dtype
            and _is_sha256(record["logical_sha256"]),
            f"{label} contract changed: {name}",
        )
        normalized[name] = record
    return normalized


def validate_feature_manifest(
    value: Any,
    *,
    config: Mapping[str, Any],
    expected_alignment_sha256: str,
    expected_decoder_contract: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate only the source-authenticated V2 feature-bundle schema."""

    manifest = dict(_mapping(value, "feature bundle manifest"))
    _require(
        set(manifest)
        == {
            "schema_version",
            "kind",
            "status",
            "scope",
            "producer",
            "decoder_contract",
            "data",
            "alignment_index",
            "sources",
            "source_feature_blocks",
            "construction",
            "row_count",
            "metadata_arrays",
            "base_feature_blocks",
            "forbidden_array_fields_absent",
            "grouping_arrays_never_numeric_features",
            "claim_scope",
        },
        "feature bundle manifest schema changed",
    )
    artifacts = config["artifacts"]
    _require(
        manifest["schema_version"] == FEATURE_BUNDLE_SCHEMA_VERSION
        and manifest["kind"] == FEATURE_BUNDLE_KIND
        and manifest["status"] == "passed"
        and manifest["scope"] == artifacts["feature_bundle_scope"]
        and manifest["row_count"] == 1606
        and manifest["metadata_arrays"] == list(METADATA_ARRAYS)
        and manifest["base_feature_blocks"] == BASE_BLOCK_WIDTHS
        and manifest["forbidden_array_fields_absent"]
        == artifacts["forbidden_array_fields"]
        and manifest["grouping_arrays_never_numeric_features"]
        == artifacts["grouping_arrays_never_numeric_features"],
        "feature bundle manifest identity changed",
    )
    producer = _validate_artifact_record(manifest["producer"], "bundle producer")
    _require(
        FEATURE_BUNDLE_PRODUCER_PATH.is_file()
        and not FEATURE_BUNDLE_PRODUCER_PATH.is_symlink()
        and producer
        == {
            "path": str(FEATURE_BUNDLE_PRODUCER_PATH.relative_to(ROOT)),
            "sha256": sha256_file(FEATURE_BUNDLE_PRODUCER_PATH),
            "size_bytes": FEATURE_BUNDLE_PRODUCER_PATH.stat().st_size,
        },
        "feature bundle producer binding changed",
    )
    decoder_contract = dict(
        _mapping(manifest["decoder_contract"], "decoder contract")
    )
    _require(
        set(decoder_contract) == {"config", "implementation"},
        "decoder contract schema changed",
    )
    config_record = _validate_artifact_record(
        decoder_contract["config"], "decoder config binding"
    )
    implementation_record = _validate_artifact_record(
        decoder_contract["implementation"], "decoder implementation binding"
    )
    if expected_decoder_contract is None:
        expected_decoder_contract = {
            "config": _artifact_byte_record(CONFIG_PATH, label="decoder config"),
            "implementation": _artifact_byte_record(
                SCRIPT_PATH, label="decoder implementation"
            ),
        }
    _require(
        set(expected_decoder_contract) == {"config", "implementation"},
        "expected decoder contract schema changed",
    )
    _require(
        config_record == expected_decoder_contract["config"]
        and implementation_record == expected_decoder_contract["implementation"],
        "feature bundle decoder binding changed",
    )
    expected_bundle_arrays = sorted([*METADATA_ARRAYS, *BASE_BLOCK_WIDTHS])
    data = dict(_mapping(manifest["data"], "feature data binding"))
    _require(
        set(data)
        == {
            "path",
            "sha256",
            "size_bytes",
            "format",
            "array_keys",
            "arrays",
            "reload_verified",
        }
        and isinstance(data["path"], str)
        and bool(data["path"])
        and not Path(data["path"]).is_absolute()
        and Path(data["path"]).name == data["path"]
        and _is_sha256(data["sha256"])
        and isinstance(data["size_bytes"], int)
        and data["size_bytes"] > 0
        and data["format"] == "npz_allow_pickle_false"
        and data["array_keys"] == expected_bundle_arrays
        and data["reload_verified"] is True,
        "feature data binding changed",
    )
    _validate_array_records(
        data["arrays"], names=expected_bundle_arrays, label="bundle arrays"
    )
    alignment = _validate_artifact_record(
        manifest["alignment_index"], "alignment binding"
    )
    _require(
        alignment["sha256"] == expected_alignment_sha256,
        "feature bundle alignment binding changed",
    )
    _require(
        manifest["source_feature_blocks"] == SOURCE_FEATURE_BLOCKS,
        "source feature-block registry changed",
    )
    sources = manifest["sources"]
    _require(
        isinstance(sources, list)
        and [source.get("role") for source in sources if isinstance(source, Mapping)]
        == artifacts["required_source_roles"],
        "feature source roles or order changed",
    )
    for position, source_value in enumerate(sources):
        source = dict(_mapping(source_value, f"feature source {position}"))
        role = artifacts["required_source_roles"][position]
        expected_keys = SOURCE_ARRAY_KEYS[role]
        _require(
            set(source)
            == {
                "role",
                "kind",
                "manifest",
                "data",
                "selected_arrays",
                "alignment_index_sha256",
            }
            and source["role"] == role
            and source["kind"] == artifacts["required_source_kinds"][role]
            and source["selected_arrays"] == expected_keys
            and source["alignment_index_sha256"] == expected_alignment_sha256,
            f"feature source contract changed: {role}",
        )
        _validate_artifact_record(source["manifest"], f"{role} manifest")
        source_data = dict(_mapping(source["data"], f"{role} data"))
        _require(
            set(source_data)
            == {
                "path",
                "sha256",
                "size_bytes",
                "format",
                "array_keys",
                "arrays",
            }
            and isinstance(source_data["path"], str)
            and bool(source_data["path"])
            and not _lexical_path_has_forbidden_fragment(Path(source_data["path"]))
            and _is_sha256(source_data["sha256"])
            and isinstance(source_data["size_bytes"], int)
            and source_data["size_bytes"] > 0
            and source_data["format"] == "npz_allow_pickle_false"
            and source_data["array_keys"] == expected_keys,
            f"feature source data contract changed: {role}",
        )
        _validate_array_records(
            source_data["arrays"], names=expected_keys, label=f"{role} arrays"
        )
    _require(
        manifest["construction"]
        == {
            "algorithm": "authenticated_source_npz_exact_join_v1",
            "join_key": "global_index",
            "stable_alignment_only": True,
            "caller_supplied_feature_arrays_accepted": False,
            "labels_targets_outcomes_or_completion_text_opened": False,
            "source_manifest_hashes_verified": True,
            "source_data_hashes_verified": True,
            "source_logical_hashes_verified": True,
            "source_global_indices_equal_alignment": True,
            "source_arrays_equal_bundle_arrays": True,
            "bundle_reload_verified": True,
            "pre_and_post_source_bindings_equal": True,
        },
        "feature construction proof changed",
    )
    _require(
        manifest["claim_scope"]
        == {
            "labels_or_outcomes_present": False,
            "semantic_grouping_fields_used_as_features": False,
            "private_chain_of_thought_reconstructed": False,
            "emotion_decoding_established": False,
        },
        "feature bundle claim scope changed",
    )
    return manifest


def _legacy_make_feature_manifest(
    *,
    data_path: Path,
    data_sha256: str,
    alignment_index_path: Path,
    alignment_index_sha256: str,
    source_bindings: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    config = validate_config(config)
    _require(_is_sha256(data_sha256), "feature data SHA-256 is invalid")
    _require(_is_sha256(alignment_index_sha256), "alignment SHA-256 is invalid")
    manifest = {
        "schema_version": 1,
        "kind": FEATURE_BUNDLE_KIND,
        "status": "passed",
        "scope": config["artifacts"]["feature_bundle_scope"],
        "config": {
            "path": str(CONFIG_PATH.relative_to(ROOT)),
            "sha256": sha256_file(CONFIG_PATH),
        },
        "implementation": {
            "path": str(SCRIPT_PATH.relative_to(ROOT)),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "data": {
            "path": os.fspath(data_path),
            "sha256": data_sha256,
            "format": "npz_allow_pickle_false",
            "array_keys": sorted([*METADATA_ARRAYS, *BASE_BLOCK_WIDTHS]),
        },
        "alignment_index": {
            "path": os.fspath(alignment_index_path),
            "sha256": alignment_index_sha256,
        },
        "sources": [dict(binding) for binding in source_bindings],
        "row_count": 1606,
        "metadata_arrays": list(METADATA_ARRAYS),
        "base_feature_blocks": dict(BASE_BLOCK_WIDTHS),
        "forbidden_array_fields_absent": list(
            config["artifacts"]["forbidden_array_fields"]
        ),
        "grouping_arrays_never_numeric_features": list(
            config["artifacts"]["grouping_arrays_never_numeric_features"]
        ),
        "claim_scope": {
            "labels_or_outcomes_present": False,
            "semantic_grouping_fields_used_as_features": False,
            "private_chain_of_thought_reconstructed": False,
            "emotion_decoding_established": False,
        },
    }
    # Validate structure without opening the referenced data.
    return validate_feature_manifest(
        manifest,
        config=config,
        expected_alignment_sha256=alignment_index_sha256,
    )


def _authenticate_feature_sources(
    source_bindings: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> list[Path]:
    """Re-hash both label-free source artifacts without parsing either one."""

    expected_roles = config["artifacts"]["required_source_roles"]
    _require(
        isinstance(source_bindings, Sequence)
        and not isinstance(source_bindings, (str, bytes, bytearray))
        and len(source_bindings) == len(expected_roles),
        "feature source binding count changed",
    )
    paths: list[Path] = []
    for position, raw in enumerate(source_bindings):
        binding = _mapping(raw, f"feature source binding {position}")
        _require(
            set(binding) == {"role", "kind", "path", "sha256"}
            and binding.get("role") == expected_roles[position]
            and binding.get("kind")
            == config["artifacts"]["required_source_kinds"][
                expected_roles[position]
            ]
            and isinstance(binding.get("path"), str)
            and bool(binding.get("path"))
            and _is_sha256(binding.get("sha256")),
            f"feature source binding {position} is invalid",
        )
        raw_path = Path(str(binding["path"]))
        paths.append(raw_path if raw_path.is_absolute() else ROOT / raw_path)
    frozen_lexical_path_preflight(paths)
    frozen_canonical_path_preflight(input_paths=paths, output_paths=[])
    resolved = [path.resolve(strict=True) for path in paths]
    for path, binding in zip(resolved, source_bindings):
        _require(
            sha256_file(path) == binding["sha256"],
            f"feature source hash changed: {binding['role']}",
        )
    return resolved


def _legacy_write_feature_bundle(
    *,
    arrays: Mapping[str, np.ndarray],
    alignment_rows: Sequence[Mapping[str, Any]],
    data_path: Path,
    manifest_path: Path,
    alignment_index_path: Path,
    alignment_index_sha256: str,
    source_bindings: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically write a no-clobber, label-free NPZ and bound manifest."""

    config = validate_config(config)
    frozen_lexical_path_preflight([data_path, manifest_path, alignment_index_path])
    frozen_canonical_path_preflight(
        input_paths=[alignment_index_path], output_paths=[data_path, manifest_path]
    )
    _require(
        not data_path.exists()
        and not data_path.is_symlink()
        and not manifest_path.exists()
        and not manifest_path.is_symlink(),
        "refusing to overwrite feature bundle output",
    )
    _require(
        _is_sha256(alignment_index_sha256)
        and sha256_file(alignment_index_path.resolve(strict=True))
        == alignment_index_sha256,
        "alignment index hash changed before feature bundle write",
    )
    _authenticate_feature_sources(source_bindings, config=config)
    validated = validate_feature_arrays(arrays, alignment_rows=alignment_rows)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_data = data_path.with_name(f".{data_path.name}.tmp-{os.getpid()}")
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.tmp-{os.getpid()}"
    )
    try:
        with temporary_data.open("wb") as handle:
            np.savez_compressed(handle, **validated)
        os.replace(temporary_data, data_path)
        manifest = _legacy_make_feature_manifest(
            data_path=Path(os.path.relpath(data_path, manifest_path.parent)),
            data_sha256=sha256_file(data_path),
            alignment_index_path=alignment_index_path,
            alignment_index_sha256=alignment_index_sha256,
            source_bindings=source_bindings,
            config=config,
        )
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        for temporary in (temporary_data, temporary_manifest):
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
    return manifest


def _resolve_and_authenticate_artifact(
    record_value: Any, *, label: str, base: Path = ROOT
) -> Path:
    record = _validate_artifact_record(record_value, label)
    raw = Path(record["path"])
    path = raw if raw.is_absolute() else base / raw
    frozen_lexical_path_preflight([path])
    frozen_canonical_path_preflight(input_paths=[path], output_paths=[])
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    resolved = path.resolve(strict=True)
    _require(
        resolved.is_file()
        and resolved.stat().st_size == record["size_bytes"]
        and sha256_file(resolved) == record["sha256"],
        f"{label} byte binding changed",
    )
    return resolved


def _load_bound_npz(
    *,
    path: Path,
    expected_sha256: str,
    expected_names: Sequence[str],
    records_value: Any,
    label: str,
) -> dict[str, np.ndarray]:
    records = _validate_array_records(
        records_value, names=expected_names, label=f"{label} arrays"
    )
    _require(sha256_file(path) == expected_sha256, f"{label} hash changed")
    try:
        with np.load(path, allow_pickle=False) as archive:
            _require(
                archive.files == list(expected_names),
                f"{label} NPZ key order differs from manifest",
            )
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise DecoderError(f"cannot load authenticated {label} NPZ: {error}") from error
    _require(sha256_file(path) == expected_sha256, f"{label} changed during reload")
    for name, array in arrays.items():
        shape, dtype = _expected_feature_array_record(name)
        _require(
            list(array.shape) == shape
            and array.dtype.str == dtype
            and array.dtype.kind != "O"
            and (array.dtype.kind != "f" or bool(np.all(np.isfinite(array))))
            and source_logical_array_sha256(name, array)
            == records[name]["logical_sha256"],
            f"{label} array contract or logical hash changed: {name}",
        )
        arrays[name] = np.ascontiguousarray(array)
    return arrays


def _canonicalize_upstream_array_records(
    value: Any, *, names: Sequence[str], label: str
) -> dict[str, Any]:
    """Normalize source-manifest dtype spelling without relaxing its schema."""

    records = dict(_mapping(value, label))
    _require(set(records) == set(names), f"{label} registry changed")
    normalized: dict[str, Any] = {}
    for name in names:
        record = dict(_mapping(records[name], f"{label} {name}"))
        expected_shape, expected_dtype = _expected_feature_array_record(name)
        _require(
            set(record) == {"shape", "dtype", "logical_sha256"}
            and record["shape"] == expected_shape
            and isinstance(record["dtype"], str)
            and UPSTREAM_DESCRIPTIVE_DTYPES.get(record["dtype"])
            == expected_dtype
            and _is_sha256(record["logical_sha256"]),
            f"{label} contract changed: {name}",
        )
        normalized[name] = {
            "shape": list(record["shape"]),
            "dtype": expected_dtype,
            "logical_sha256": record["logical_sha256"],
        }
    return normalized


def _validate_upstream_source_manifest(
    *,
    source: Mapping[str, Any],
    manifest_path: Path,
    data_path: Path,
    expected_alignment_sha256: str,
) -> None:
    value = _mapping(load_json(manifest_path), f"{source['role']} source manifest")
    role = str(source["role"])
    _require(
        value.get("schema_version") == 1
        and value.get("kind") == source["kind"]
        and value.get("status") == "passed"
        and value.get("forbidden_path_guard_passed") is True
        and value.get("reserved_validation_access_authorized") is False,
        f"{role} upstream manifest identity changed",
    )
    output = _mapping(value.get("output"), f"{role} upstream output")
    upstream_arrays = _canonicalize_upstream_array_records(
        output.get("arrays"),
        names=list(source["selected_arrays"]),
        label=f"{role} upstream output arrays",
    )
    _require(
        output.get("sha256") == source["data"]["sha256"]
        and output.get("size_bytes") == source["data"]["size_bytes"]
        and output.get("keys") == source["selected_arrays"]
        and upstream_arrays == source["data"]["arrays"],
        f"{role} upstream output differs from bundle source binding",
    )
    upstream_raw = Path(str(output.get("path")))
    _require(not upstream_raw.is_absolute(), f"{role} upstream data path is absolute")
    upstream_path = (manifest_path.parent / upstream_raw).resolve(strict=True)
    _require(upstream_path == data_path, f"{role} source data path binding changed")
    inputs = _mapping(value.get("inputs"), f"{role} upstream inputs")
    alignment = (
        _mapping(inputs.get("label_free_alignment_index"), "visible alignment")
        if role == "visible_baselines"
        else _mapping(inputs.get("alignment_index"), "activation alignment")
    )
    _require(
        alignment.get("sha256") == expected_alignment_sha256,
        f"{role} upstream alignment binding changed",
    )


def load_authenticated_feature_bundle(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_alignment_sha256: str,
    alignment_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    expected_decoder_contract: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
    Path,
    dict[str, tuple[Path, dict[str, Any]]],
]:
    """Load V2 bundle and independently reauthenticate both source NPZs."""

    config = validate_config(config)
    _require(_is_sha256(expected_manifest_sha256), "feature manifest SHA-256 invalid")
    _require(_is_sha256(expected_alignment_sha256), "alignment SHA-256 invalid")
    _require(
        sha256_file(manifest_path) == expected_manifest_sha256,
        "feature manifest hash changed",
    )
    manifest = validate_feature_manifest(
        load_json(manifest_path),
        config=config,
        expected_alignment_sha256=expected_alignment_sha256,
        expected_decoder_contract=expected_decoder_contract,
    )
    alignment_path = _resolve_and_authenticate_artifact(
        manifest["alignment_index"], label="bundle alignment index"
    )
    _require(
        sha256_file(alignment_path) == expected_alignment_sha256,
        "bundle alignment index differs from caller binding",
    )
    data_record = manifest["data"]
    raw_data_path = Path(data_record["path"])
    data_path = manifest_path.parent / raw_data_path
    frozen_lexical_path_preflight([data_path])
    frozen_canonical_path_preflight(input_paths=[data_path], output_paths=[])
    _require(
        not data_path.is_symlink()
        and data_path.resolve(strict=True).parent == manifest_path.parent.resolve(strict=True)
        and data_path.stat().st_size == data_record["size_bytes"],
        "feature data path or size binding changed",
    )
    bundle = _load_bound_npz(
        path=data_path,
        expected_sha256=data_record["sha256"],
        expected_names=data_record["array_keys"],
        records_value=data_record["arrays"],
        label="feature bundle",
    )
    bundle = validate_feature_arrays(bundle, alignment_rows=alignment_rows)

    authenticated_paths = [
        manifest_path.resolve(strict=True),
        alignment_path,
        data_path.resolve(strict=True),
    ]
    labeled_paths: dict[str, Path] = {
        "feature_manifest": manifest_path.resolve(strict=True),
        "alignment_index": alignment_path,
        "feature_data": data_path.resolve(strict=True),
    }
    for source_value in manifest["sources"]:
        source = dict(_mapping(source_value, "feature source binding"))
        role = str(source["role"])
        source_manifest_path = _resolve_and_authenticate_artifact(
            source["manifest"], label=f"{role} manifest"
        )
        source_data_path = _resolve_and_authenticate_artifact(
            {
                key: source["data"][key]
                for key in ("path", "sha256", "size_bytes")
            },
            label=f"{role} data",
        )
        _validate_upstream_source_manifest(
            source=source,
            manifest_path=source_manifest_path,
            data_path=source_data_path,
            expected_alignment_sha256=expected_alignment_sha256,
        )
        source_arrays = _load_bound_npz(
            path=source_data_path,
            expected_sha256=source["data"]["sha256"],
            expected_names=source["selected_arrays"],
            records_value=source["data"]["arrays"],
            label=f"{role} source",
        )
        _require(
            np.array_equal(source_arrays["global_index"], bundle["global_index"]),
            f"{role} global_index differs from bundle",
        )
        for name in SOURCE_FEATURE_BLOCKS[role]:
            _require(
                np.array_equal(source_arrays[name], bundle[name], equal_nan=False),
                f"{role} block differs from bundle: {name}",
            )
        authenticated_paths.extend([source_manifest_path, source_data_path])
        labeled_paths[f"{role}_source_manifest"] = source_manifest_path
        labeled_paths[f"{role}_source_data"] = source_data_path
    _require(
        sha256_file(manifest_path) == expected_manifest_sha256,
        "feature manifest changed during source reauthentication",
    )
    byte_bindings = {
        label: (path, _artifact_byte_record(path, label=label))
        for label, path in labeled_paths.items()
    }
    _require(
        byte_bindings["feature_manifest"][1]["sha256"]
        == expected_manifest_sha256
        and byte_bindings["alignment_index"][1]["sha256"]
        == expected_alignment_sha256
        and byte_bindings["feature_data"][1]["sha256"]
        == manifest["data"]["sha256"],
        "feature byte-binding ledger differs from authenticated manifest",
    )
    for source in manifest["sources"]:
        role = str(source["role"])
        _require(
            byte_bindings[f"{role}_source_manifest"][1]
            == source["manifest"]
            and byte_bindings[f"{role}_source_data"][1]["sha256"]
            == source["data"]["sha256"]
            and byte_bindings[f"{role}_source_data"][1]["size_bytes"]
            == source["data"]["size_bytes"],
            f"{role} byte-binding ledger differs from authenticated manifest",
        )
    return bundle, manifest, data_path.resolve(strict=True), byte_bindings


def hierarchical_row_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Equal repository, then task, then row mass on exactly these rows."""

    _require(bool(rows), "hierarchical weights require at least one row")
    repo_tasks: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(rows):
        repository = row.get("repository")
        task = row.get("task_id_sha256")
        _require(
            isinstance(repository, str)
            and bool(repository)
            and isinstance(task, str)
            and bool(task),
            "weighting row grouping is invalid",
        )
        repo_tasks[repository][task].append(index)
    weights = np.zeros(len(rows), dtype=np.float64)
    repo_mass = 1.0 / len(repo_tasks)
    for tasks in repo_tasks.values():
        task_mass = repo_mass / len(tasks)
        for indices in tasks.values():
            row_mass = task_mass / len(indices)
            weights[np.asarray(indices, dtype=np.int64)] = row_mass
    _require(
        bool(np.all(weights > 0.0))
        and bool(np.all(np.isfinite(weights)))
        and abs(float(weights.sum()) - 1.0) <= 1e-12,
        "hierarchical row weights are invalid",
    )
    return weights


def fit_weighted_standardizer(
    X_train: np.ndarray, unit_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit weighted mean/variance on an outer-training matrix only."""

    X = np.asarray(X_train, dtype=np.float64)
    weights = np.asarray(unit_weights, dtype=np.float64)
    _require(
        X.ndim == 2
        and len(X) == len(weights)
        and len(X) > 0
        and bool(np.all(np.isfinite(X)))
        and bool(np.all(weights > 0.0))
        and abs(float(weights.sum()) - 1.0) <= 1e-12,
        "standardizer training inputs are invalid",
    )
    mean = weights @ X
    centered = X - mean
    variance = weights @ (centered * centered)
    variance = np.maximum(variance, 0.0)
    scale = np.sqrt(variance)
    constant = scale <= np.finfo(np.float64).eps
    scale[constant] = 1.0
    _require(
        bool(np.all(np.isfinite(mean)))
        and bool(np.all(np.isfinite(scale)))
        and bool(np.all(scale > 0.0)),
        "standardizer parameters are invalid",
    )
    return mean, scale, constant


def _floored_probabilities(probabilities: np.ndarray, floor: float) -> np.ndarray:
    observed = np.asarray(probabilities, dtype=np.float64)
    _require(
        observed.ndim == 2 and bool(np.all(np.isfinite(observed))),
        "probability matrix is invalid",
    )
    clipped = np.maximum(observed, floor)
    clipped /= clipped.sum(axis=1, keepdims=True)
    _require(
        bool(np.all(clipped > 0.0))
        and bool(np.allclose(clipped.sum(axis=1), 1.0, atol=1e-12)),
        "floored probability matrix is invalid",
    )
    return clipped


def _fit_predict_fold(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    training_rows: Sequence[Mapping[str, Any]],
    X_test: np.ndarray,
    class_count: int,
    model_contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one fold without any held-out label argument."""

    _require(
        X_train.ndim == X_test.ndim == 2
        and X_train.shape[1] == X_test.shape[1]
        and len(X_train) == len(y_train) == len(training_rows)
        and class_count >= 2,
        "fold matrices or labels are misaligned",
    )
    weights = hierarchical_row_weights(training_rows)
    mean, scale, constant = fit_weighted_standardizer(X_train, weights)
    standardized_train = (X_train - mean) / scale
    standardized_test = (X_test - mean) / scale
    supported = np.unique(y_train)
    _require(
        len(supported) >= 1
        and int(supported.min()) >= 0
        and int(supported.max()) < class_count,
        "training labels are invalid",
    )
    floor = float(model_contract["probability_floor"])
    full = np.zeros((len(X_test), class_count), dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "training_row_count": len(X_train),
        "supported_training_class_indices": supported.astype(int).tolist(),
        "unsupported_training_class_indices": sorted(
            set(range(class_count)) - set(supported.astype(int).tolist())
        ),
        "standardizer_fitted_on_outer_training_only": True,
        "constant_feature_count": int(constant.sum()),
        "standardizer_mean_sha256": sha256_array(mean.astype("<f8")),
        "standardizer_scale_sha256": sha256_array(scale.astype("<f8")),
        "heldout_labels_used_for_fit_or_standardization": False,
    }
    if len(supported) == 1:
        full[:, int(supported[0])] = 1.0
        diagnostics.update(
            {
                "fit_status": "single_class_train_constant",
                "model_fitted": False,
                "n_iter": 0,
            }
        )
        return _floored_probabilities(full, floor), diagnostics

    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:
        raise DecoderError(
            "scikit-learn is required for the fixed-C observable decoder"
        ) from error
    model = LogisticRegression(
        C=float(model_contract["C"]),
        solver=str(model_contract["solver"]),
        l1_ratio=float(model_contract["l1_ratio"]),
        fit_intercept=bool(model_contract["fit_intercept"]),
        max_iter=int(model_contract["max_iter"]),
        tol=float(model_contract["tol"]),
        random_state=int(model_contract["random_state"]),
        class_weight=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(
            standardized_train,
            y_train,
            sample_weight=weights * len(weights),
        )
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    _require(not convergence, "fixed-C logistic fit did not converge")
    supported_model = np.asarray(model.classes_, dtype=np.int64)
    _require(
        np.array_equal(supported_model, supported),
        "fitted logistic class order changed",
    )
    full[:, supported_model] = model.predict_proba(standardized_test)
    fit_status = (
        "fit_all_contract_classes"
        if len(supported) == class_count
        else "fit_partial_training_class_support"
    )
    diagnostics.update(
        {
            "fit_status": fit_status,
            "model_fitted": True,
            "n_iter": int(np.max(model.n_iter_)),
            "coefficient_sha256": sha256_array(
                np.asarray(model.coef_, dtype="<f8")
            ),
            "intercept_sha256": sha256_array(
                np.asarray(model.intercept_, dtype="<f8")
            ),
        }
    )
    return _floored_probabilities(full, floor), diagnostics


def _metric(value: float | None, status: str, **details: Any) -> dict[str, Any]:
    if value is not None:
        _require(math.isfinite(value), "metric value must be finite")
    return {"value": value, "status": status, **details}


def weighted_metrics(
    *,
    y: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    classes: Sequence[str],
    positive_class: str = "yes",
) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    _require(
        labels.shape == w.shape == (len(p),)
        and p.shape == (len(labels), len(classes))
        and bool(np.all(p > 0.0))
        and bool(np.allclose(p.sum(axis=1), 1.0, atol=1e-12))
        and bool(np.all(w > 0.0))
        and abs(float(w.sum()) - 1.0) <= 1e-12,
        "metric inputs are invalid",
    )
    true_p = p[np.arange(len(labels)), labels]
    nll = float(np.sum(w * -np.log(true_p)))
    one_hot = np.eye(len(classes), dtype=np.float64)[labels]
    brier = float(np.sum(w * np.sum((p - one_hot) ** 2, axis=1)))
    predicted = np.argmax(p, axis=1)
    accuracy = float(np.sum(w * (predicted == labels)))
    recalls = []
    supported_names = []
    for class_index, name in enumerate(classes):
        mask = labels == class_index
        if np.any(mask):
            recalls.append(float(np.sum(w[mask] * (predicted[mask] == class_index)) / np.sum(w[mask])))
            supported_names.append(name)
    balanced = float(np.mean(recalls))

    auc_status = "undefined_single_class_oof"
    auroc: float | None = None
    auprc: float | None = None
    auc_classes: list[str] = []
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as error:
        raise DecoderError("scikit-learn metrics are required") from error
    if len(classes) == 2 and positive_class in classes:
        positive_index = list(classes).index(positive_class)
        binary = (labels == positive_index).astype(np.int64)
        if len(np.unique(binary)) == 2:
            auroc = float(roc_auc_score(binary, p[:, positive_index], sample_weight=w))
            auprc = float(
                average_precision_score(binary, p[:, positive_index], sample_weight=w)
            )
            auc_status = "available_binary_positive_class"
            auc_classes = [positive_class]
    else:
        roc_values = []
        pr_values = []
        for class_index, name in enumerate(classes):
            binary = (labels == class_index).astype(np.int64)
            if len(np.unique(binary)) < 2:
                continue
            roc_values.append(
                float(roc_auc_score(binary, p[:, class_index], sample_weight=w))
            )
            pr_values.append(
                float(
                    average_precision_score(
                        binary, p[:, class_index], sample_weight=w
                    )
                )
            )
            auc_classes.append(name)
        if roc_values:
            auroc = float(np.mean(roc_values))
            auprc = float(np.mean(pr_values))
            auc_status = (
                "available_macro_ovr"
                if len(roc_values) == len(classes)
                else "partial_macro_ovr_missing_oof_class"
            )
    return {
        "weighted_nll": _metric(nll, "available"),
        "weighted_multiclass_brier": _metric(brier, "available"),
        "weighted_accuracy": _metric(accuracy, "available"),
        "weighted_balanced_accuracy": _metric(
            balanced,
            "available_all_classes"
            if len(recalls) == len(classes)
            else "partial_missing_oof_class",
            supported_classes=supported_names,
        ),
        "weighted_auprc": _metric(
            auprc, auc_status, evaluated_classes=auc_classes
        ),
        "weighted_auroc": _metric(
            auroc, auc_status, evaluated_classes=auc_classes
        ),
    }


def repository_heldout_oof(
    *,
    X_by_variant: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    classes: Sequence[str],
    variant_specs: Sequence[Mapping[str, Any]],
    model_contract: Mapping[str, Any],
    positive_class: str = "yes",
) -> dict[str, Any]:
    """Run fixed-C LORO on one already aligned, available-label row set."""

    _require(
        len(rows) == len(labels) > 0
        and len(classes) >= 2
        and len(classes) == len(set(classes)),
        "repository-held-out target inputs are invalid",
    )
    class_to_index = {name: index for index, name in enumerate(classes)}
    _require(
        all(label in class_to_index for label in labels),
        "target label is outside the frozen target contract",
    )
    y = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    repositories = np.asarray([str(row["repository"]) for row in rows], dtype=str)
    repo_order = sorted(set(repositories.tolist()))
    _require(len(repo_order) >= 2, "LORO requires at least two repositories")
    names = [str(spec["name"]) for spec in variant_specs]
    _require(
        len(names) == len(set(names)) and set(X_by_variant) == set(names),
        "variant matrices and specifications differ",
    )
    matrices = {}
    for spec in variant_specs:
        name = str(spec["name"])
        matrix = np.asarray(X_by_variant[name], dtype=np.float64)
        _require(
            matrix.shape == (len(rows), int(spec["width"]))
            and bool(np.all(np.isfinite(matrix))),
            f"OOF variant matrix is invalid: {name}",
        )
        matrices[name] = matrix
    evaluation_weights = hierarchical_row_weights(rows)
    variant_results: dict[str, Any] = {}
    probabilities_by_variant: dict[str, np.ndarray] = {}
    common_fold_indices = []
    for heldout_repository in repo_order:
        test_indices = np.flatnonzero(repositories == heldout_repository)
        train_indices = np.flatnonzero(repositories != heldout_repository)
        _require(
            len(test_indices) > 0 and len(train_indices) > 0,
            "outer repository split is empty",
        )
        common_fold_indices.append((heldout_repository, train_indices, test_indices))
    for spec in variant_specs:
        name = str(spec["name"])
        oof = np.full((len(rows), len(classes)), np.nan, dtype=np.float64)
        folds = []
        for heldout_repository, train_indices, test_indices in common_fold_indices:
            training_rows = [rows[int(index)] for index in train_indices]
            heldout_support = Counter(labels[int(index)] for index in test_indices)
            heldout_supported_classes = [
                class_name
                for class_name in classes
                if heldout_support.get(class_name, 0) > 0
            ]
            predicted, diagnostics = _fit_predict_fold(
                X_train=matrices[name][train_indices],
                y_train=y[train_indices],
                training_rows=training_rows,
                X_test=matrices[name][test_indices],
                class_count=len(classes),
                model_contract=model_contract,
            )
            oof[test_indices] = predicted
            heldout_rows = [rows[int(index)] for index in test_indices]
            heldout_weights = hierarchical_row_weights(heldout_rows)
            heldout_metrics = weighted_metrics(
                y=y[test_indices],
                probabilities=predicted,
                weights=heldout_weights,
                classes=classes,
                positive_class=positive_class,
            )
            folds.append(
                {
                    "heldout_repository": heldout_repository,
                    "heldout_row_count": len(test_indices),
                    "training_repositories": sorted(
                        set(repositories[train_indices].tolist())
                    ),
                    "heldout_class_support": {
                        class_name: int(heldout_support.get(class_name, 0))
                        for class_name in classes
                    },
                    "heldout_supported_classes": heldout_supported_classes,
                    "heldout_all_contract_classes_present": (
                        len(heldout_supported_classes) == len(classes)
                    ),
                    "heldout_balanced_accuracy_and_auc_eligible": (
                        len(heldout_supported_classes) == len(classes)
                    ),
                    "heldout_weighting": "equal_task_then_equal_available_row_within_task",
                    "heldout_evaluation_weight_sha256": sha256_array(
                        heldout_weights.astype("<f8")
                    ),
                    "heldout_metrics": heldout_metrics,
                    "per_fold_balanced_or_auc_metrics_reported": True,
                    "heldout_repository_absent_from_training": heldout_repository
                    not in set(repositories[train_indices].tolist()),
                    **diagnostics,
                }
            )
        _require(
            bool(np.all(np.isfinite(oof)))
            and bool(np.all(oof > 0.0))
            and bool(np.allclose(oof.sum(axis=1), 1.0, atol=1e-12)),
            f"OOF predictions are incomplete: {name}",
        )
        metrics = weighted_metrics(
            y=y,
            probabilities=oof,
            weights=evaluation_weights,
            classes=classes,
            positive_class=positive_class,
        )
        probabilities_by_variant[name] = oof
        variant_results[name] = {
            "role": spec["role"],
            "components": list(spec["components"]),
            "width": int(spec["width"]),
            "folds": folds,
            "metrics": metrics,
            "oof_probability_sha256": sha256_array(oof.astype("<f8")),
            "oof_probabilities": oof.tolist(),
        }
    identity_payload = [
        {
            "global_index": row.get("global_index"),
            "source_id_sha256": row.get("source_id_sha256"),
            "task_id_sha256": row.get("task_id_sha256"),
            "repository": row.get("repository"),
            "request_index": row.get("request_index"),
        }
        for row in rows
    ]
    return {
        "row_count": len(rows),
        "classes": list(classes),
        "repositories_in_order": repo_order,
        "same_outer_folds_for_every_variant": True,
        "same_oof_rows_labels_and_weights_for_every_variant": True,
        "pooled_auroc_and_auprc_rank_probabilities_from_distinct_fold_specific_models": True,
        "row_identity_sha256": sha256_bytes(canonical_json_bytes(identity_payload)),
        "label_sha256": sha256_bytes(canonical_json_bytes(list(labels))),
        "evaluation_weight_sha256": sha256_array(
            evaluation_weights.astype("<f8")
        ),
        "evaluation_weights": evaluation_weights.tolist(),
        "label_indices": y.tolist(),
        "variants": variant_results,
        "_probabilities": probabilities_by_variant,
    }


def paired_oof_comparison(
    *,
    candidate_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    candidate_metrics: Mapping[str, Any],
    reference_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    reference = np.asarray(reference_probabilities, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    _require(
        candidate.shape == reference.shape
        and candidate.shape[0] == len(labels) == len(w),
        "paired OOF arrays are not aligned",
    )
    row = np.arange(len(labels))
    log_loss_delta = -np.log(candidate[row, labels]) + np.log(
        reference[row, labels]
    )
    one_hot = np.eye(candidate.shape[1])[labels]
    brier_delta = np.sum((candidate - one_hot) ** 2, axis=1) - np.sum(
        (reference - one_hot) ** 2, axis=1
    )
    candidate_correct = (np.argmax(candidate, axis=1) == labels).astype(np.float64)
    reference_correct = (np.argmax(reference, axis=1) == labels).astype(np.float64)
    accuracy_delta = candidate_correct - reference_correct

    def metric_delta(name: str) -> float | None:
        left = candidate_metrics[name]["value"]
        right = reference_metrics[name]["value"]
        return None if left is None or right is None else float(left - right)

    return {
        "identical_rows_labels_weights_and_folds": True,
        "paired_row_count": len(labels),
        "candidate_minus_reference": {
            "weighted_nll": metric_delta("weighted_nll"),
            "weighted_multiclass_brier": metric_delta(
                "weighted_multiclass_brier"
            ),
            "weighted_accuracy": metric_delta("weighted_accuracy"),
            "weighted_balanced_accuracy": metric_delta(
                "weighted_balanced_accuracy"
            ),
            "weighted_auprc": metric_delta("weighted_auprc"),
            "weighted_auroc": metric_delta("weighted_auroc"),
        },
        "paired_weighted_mean_row_deltas": {
            "negative_log_likelihood": float(np.sum(w * log_loss_delta)),
            "multiclass_brier": float(np.sum(w * brier_delta)),
            "correctness": float(np.sum(w * accuracy_delta)),
        },
        "paired_row_nll_delta_sha256": sha256_array(
            log_loss_delta.astype("<f8")
        ),
        "point_deltas_establish_incremental_value": False,
    }


def paired_oof_comparison_by_repository(
    *,
    candidate_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    classes: Sequence[str],
    positive_class: str,
) -> dict[str, Any]:
    """Report paired deltas inside each held-out repository fold."""

    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    reference = np.asarray(reference_probabilities, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    _require(
        candidate.shape == reference.shape
        and candidate.shape[0] == len(rows) == len(labels),
        "repository paired comparison inputs are not aligned",
    )
    repositories = np.asarray([str(row["repository"]) for row in rows], dtype=str)
    result: dict[str, Any] = {}
    for repository in sorted(set(repositories.tolist())):
        indices = np.flatnonzero(repositories == repository)
        heldout_rows = [rows[int(index)] for index in indices]
        weights = hierarchical_row_weights(heldout_rows)
        candidate_metrics = weighted_metrics(
            y=labels[indices],
            probabilities=candidate[indices],
            weights=weights,
            classes=classes,
            positive_class=positive_class,
        )
        reference_metrics = weighted_metrics(
            y=labels[indices],
            probabilities=reference[indices],
            weights=weights,
            classes=classes,
            positive_class=positive_class,
        )
        result[repository] = {
            "row_count": len(indices),
            "weighting": "equal_task_then_equal_available_row_within_task",
            "candidate_metrics": candidate_metrics,
            "reference_metrics": reference_metrics,
            "comparison": paired_oof_comparison(
                candidate_probabilities=candidate[indices],
                reference_probabilities=reference[indices],
                y=labels[indices],
                weights=weights,
                candidate_metrics=candidate_metrics,
                reference_metrics=reference_metrics,
            ),
        }
    return result


def evaluate_observable_target(
    *,
    config: Mapping[str, Any],
    alignment_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    target_contracts: Mapping[str, Any],
    feature_arrays: Mapping[str, np.ndarray],
    target_name: str,
) -> dict[str, Any]:
    config = validate_config(config)
    _require(
        target_name in config["target_policy"]["allowed_targets"],
        "requested observable target is not frozen",
    )
    arrays = validate_feature_arrays(feature_arrays, alignment_rows=alignment_rows)
    variants_all = build_variant_matrices(arrays, config)
    contract = _mapping(target_contracts.get(target_name), "requested target contract")
    classes = contract.get("classes")
    _require(
        isinstance(classes, list) and len(classes) >= 2,
        "requested target classes are invalid",
    )
    stable_position_by_global = {
        int(global_index): position
        for position, global_index in enumerate(arrays["global_index"])
    }
    selected_positions = []
    rows = []
    labels = []
    unknown_count = 0
    for label_row in label_rows:
        if not label_row["stable_feature_eligible"]:
            continue
        global_index = int(label_row["global_index"])
        _require(
            global_index in stable_position_by_global,
            "stable label row is absent from feature bundle",
        )
        record = label_row["targets"][target_name]
        if record["status"] == "unknown":
            unknown_count += 1
            continue
        position = stable_position_by_global[global_index]
        _require(
            arrays["source_id_sha256"][position]
            == label_row["source_id_sha256"],
            "selected target row identity differs from feature bundle",
        )
        selected_positions.append(position)
        rows.append(
            {
                "global_index": global_index,
                "source_id_sha256": str(label_row["source_id_sha256"]),
                "task_id_sha256": str(label_row["task_id_sha256"]),
                "repository": str(label_row["repository"]),
                "request_index": int(label_row["request_index"]),
            }
        )
        labels.append(str(record["value"]))
    _require(
        len(rows) + unknown_count == 1606,
        "available and unknown target rows do not partition stable rows",
    )
    positions = np.asarray(selected_positions, dtype=np.int64)
    X = {name: matrix[positions] for name, matrix in variants_all.items()}
    core = repository_heldout_oof(
        X_by_variant=X,
        rows=rows,
        labels=labels,
        classes=classes,
        variant_specs=config["variants"],
        model_contract=config["model"],
        positive_class=config["metrics"]["binary_positive_class"],
    )
    probabilities = core.pop("_probabilities")
    y = np.asarray(core["label_indices"], dtype=np.int64)
    weights = np.asarray(core["evaluation_weights"], dtype=np.float64)
    comparisons: dict[str, Any] = {}
    candidate_roles = set(config["paired_comparisons"]["candidate_roles"])
    nested_pairs = {
        (item["candidate"], item["reference"])
        for item in config["paired_comparisons"]["nested_comparisons"]
    }
    for spec in config["variants"]:
        if spec["role"] not in candidate_roles:
            continue
        candidate_name = spec["name"]
        for reference_name in config["paired_comparisons"]["reference_baselines"]:
            key = f"{candidate_name}__vs__{reference_name}"
            global_comparison = paired_oof_comparison(
                candidate_probabilities=probabilities[candidate_name],
                reference_probabilities=probabilities[reference_name],
                y=y,
                weights=weights,
                candidate_metrics=core["variants"][candidate_name]["metrics"],
                reference_metrics=core["variants"][reference_name]["metrics"],
            )
            per_repository = paired_oof_comparison_by_repository(
                candidate_probabilities=probabilities[candidate_name],
                reference_probabilities=probabilities[reference_name],
                rows=rows,
                y=y,
                classes=classes,
                positive_class=config["metrics"]["binary_positive_class"],
            )
            additive_keys = (
                "negative_log_likelihood",
                "multiclass_brier",
                "correctness",
            )
            repository_means = {
                metric: float(
                    np.mean(
                        [
                            value["comparison"]["paired_weighted_mean_row_deltas"][
                                metric
                            ]
                            for value in per_repository.values()
                        ]
                    )
                )
                for metric in additive_keys
            }
            for metric in additive_keys:
                _require(
                    abs(
                        repository_means[metric]
                        - global_comparison["paired_weighted_mean_row_deltas"][
                            metric
                        ]
                    )
                    <= 1e-12,
                    "equal-repository paired delta does not equal repository mean",
                )
            comparisons[key] = {
                "candidate": candidate_name,
                "reference": reference_name,
                "strictly_nested_component_comparison": (
                    candidate_name,
                    reference_name,
                )
                in nested_pairs,
                **global_comparison,
                "per_repository": per_repository,
                "equal_repository_mean_of_additive_deltas": repository_means,
            }
    support = Counter(labels)
    is_primary = target_name == PRIMARY_TARGET
    return {
        "target": target_name,
        "target_contract": dict(contract),
        "target_is_primary_visible_surface_language_marker": is_primary,
        "horizon": PRIMARY_HORIZON if is_primary else "as_defined_by_frozen_target_contract",
        "stable_row_count": 1606,
        "available_target_row_count": len(rows),
        "unknown_target_row_count_excluded_never_coerced_to_negative": unknown_count,
        "available_target_support": {
            class_name: int(support.get(class_name, 0)) for class_name in classes
        },
        "outer_evaluation": core,
        "paired_comparisons": comparisons,
        "claims": {
            "visible_surface_language_marker_observable_process_trace_proxy_only": is_primary,
            "private_chain_of_thought_reconstructed": False,
            "cot_or_cot_like_decoding_established": False,
            "semantic_sentence_or_chain_decoding_established": False,
            "hidden_thought_understanding_or_intent_established": False,
            "emotion_decoding_established": False,
            "causal_interpretation_established": False,
            "retrospective_sparse_concept_ontology_used_as_all_row_baseline": False,
            "incremental_value_established_by_point_metrics": False,
        },
        "mandatory_limitations": list(MANDATORY_LIMITATIONS),
    }


def _write_json_no_clobber(
    path: Path,
    value: Any,
    *,
    before_publish: Callable[[], None] | None = None,
) -> None:
    """Serialize durably, run the final verifier, then hard-link without clobber."""

    _require(
        not path.exists() and not path.is_symlink(),
        f"refusing to overwrite decoder report: {path}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(
        path.parent.is_dir() and not path.parent.is_symlink(),
        "decoder report parent is not a regular directory",
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if before_publish is not None:
            before_publish()
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise DecoderError(f"refusing to overwrite decoder report: {path}") from error
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _authenticated_json(path: Path, expected_sha256: str, label: str) -> Any:
    return _authenticated_json_with_record(path, expected_sha256, label)[0]


def _authenticated_json_with_record(
    path: Path, expected_sha256: str, label: str
) -> tuple[Any, dict[str, Any]]:
    _require(_is_sha256(expected_sha256), f"{label} expected SHA-256 is invalid")
    record = _artifact_byte_record(path, label=label)
    _require(record["sha256"] == expected_sha256, f"{label} hash changed")
    value = load_json(path)
    _require(
        _artifact_byte_record(path, label=label) == record,
        f"{label} changed during JSON read",
    )
    return value, record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--expected-feature-manifest-sha256", required=True)
    parser.add_argument("--alignment-index", type=Path, required=True)
    parser.add_argument("--expected-alignment-index-sha256", required=True)
    parser.add_argument("--label-sidecar", type=Path, required=True)
    parser.add_argument("--expected-label-sidecar-sha256", required=True)
    parser.add_argument("--target", default=PRIMARY_TARGET)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    # This lexical guard is intentionally the first post-parse operation.
    frozen_lexical_path_preflight(
        [
            args.config,
            args.feature_manifest,
            args.alignment_index,
            args.label_sidecar,
            args.output,
        ]
    )
    # Resolve parent symlinks before any file is hashed or read.
    frozen_canonical_path_preflight(
        input_paths=[
            args.config,
            args.feature_manifest,
            args.alignment_index,
            args.label_sidecar,
        ],
        output_paths=[args.output],
    )
    config_path = args.config.resolve(strict=True)
    _require(config_path == CONFIG_PATH, "decoder config path changed")
    config_bytes, config_record = _read_startup_bytes(
        config_path, label="decoder config"
    )
    _script_bytes, implementation_record = _read_startup_bytes(
        SCRIPT_PATH, label="decoder implementation"
    )
    startup_decoder_contract = {
        "config": config_record,
        "implementation": implementation_record,
    }
    config = validate_config(
        _load_json_bytes(config_bytes, label="observable-decoder config")
    )
    alignment_path = args.alignment_index.resolve(strict=True)
    label_path = args.label_sidecar.resolve(strict=True)
    feature_manifest_path = args.feature_manifest.resolve(strict=True)
    alignment_value, alignment_record = _authenticated_json_with_record(
        alignment_path,
        args.expected_alignment_index_sha256,
        "alignment index",
    )
    alignment_rows = validate_alignment_index(alignment_value)
    (
        arrays,
        feature_manifest,
        feature_data_path,
        feature_byte_bindings,
    ) = load_authenticated_feature_bundle(
        manifest_path=feature_manifest_path,
        expected_manifest_sha256=args.expected_feature_manifest_sha256,
        expected_alignment_sha256=args.expected_alignment_index_sha256,
        alignment_rows=alignment_rows,
        config=config,
        expected_decoder_contract=startup_decoder_contract,
    )
    bundle_alignment_path, bundle_alignment_record = feature_byte_bindings[
        "alignment_index"
    ]
    _require(
        bundle_alignment_path == alignment_path
        and bundle_alignment_record == alignment_record,
        "caller and bundle alignment byte bindings differ",
    )
    # The numeric feature artifact is completely authenticated and validated
    # before the physically separate label sidecar is opened.
    label_value, label_record = _authenticated_json_with_record(
        label_path,
        args.expected_label_sidecar_sha256,
        "observable-label sidecar",
    )
    label_rows = validate_label_sidecar(
        label_value, alignment_rows=alignment_rows, config=config
    )
    result = evaluate_observable_target(
        config=config,
        alignment_rows=alignment_rows,
        label_rows=label_rows,
        target_contracts=label_value["target_contracts"],
        feature_arrays=arrays,
        target_name=args.target,
    )
    try:
        import sklearn
    except ImportError as error:
        raise DecoderError("scikit-learn runtime identity is unavailable") from error
    precommit_bindings: dict[str, tuple[Path, Mapping[str, Any]]] = {
        "decoder_config": (config_path, config_record),
        "decoder_implementation": (SCRIPT_PATH, implementation_record),
        **feature_byte_bindings,
        "observable_label_sidecar": (label_path, label_record),
    }
    _require(
        set(precommit_bindings)
        == {
            "decoder_config",
            "decoder_implementation",
            "feature_manifest",
            "feature_data",
            "visible_baselines_source_manifest",
            "visible_baselines_source_data",
            "activation_features_source_manifest",
            "activation_features_source_data",
            "alignment_index",
            "observable_label_sidecar",
        },
        "precommit byte-binding inventory changed",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "status": "passed",
        "status_scope": STATUS_SCOPE,
        "status_interpretation": {
            "execution_completed": True,
            "contract_checks_passed": True,
            "scientific_validity_reliability_confirmation_or_transport_gate_passed": False,
        },
        "scope": "development_only_reserved_validation_closed",
        "config": config_record,
        "implementation": implementation_record,
        "execution_identity": {
            "config_and_implementation_exact_bytes_captured_at_start_of_run": True,
            "startup_byte_records": startup_decoder_contract,
            "same_startup_bytes_required_immediately_before_atomic_publish": True,
        },
        "runtime": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "inputs": {
            "feature_manifest": feature_byte_bindings["feature_manifest"][1],
            "feature_data": feature_byte_bindings["feature_data"][1],
            "feature_sources": feature_manifest["sources"],
            "alignment_index": alignment_record,
            "label_sidecar": label_record,
            "all_hashes_matched_before_model_fit": True,
            "feature_source_hashes_reverified_before_model_fit": True,
            "feature_and_label_artifacts_physically_separate": True,
            "reserved_validation_access_authorized": False,
        },
        "precommit_reauthentication": {
            "required_after_evaluation_and_after_report_serialization": True,
            "all_authenticated_files_rehashed_immediately_before_atomic_publish": True,
            "authenticated_file_labels": sorted(precommit_bindings),
            "byte_records": {
                label: dict(record)
                for label, (_path, record) in precommit_bindings.items()
            },
            "publication": "same_directory_fsynced_temporary_then_atomic_hard_link_no_clobber",
        },
        "analysis_chronology": dict(config["analysis_chronology"]),
        "claim_scope": dict(config["claim_scope"]),
        "mandatory_limitations": list(config["mandatory_limitations"]),
        "evaluation": result,
    }
    output_path = Path(os.path.abspath(os.fspath(args.output)))
    _require(not output_path.is_symlink(), "decoder report output must not be a symlink")
    _write_json_no_clobber(
        output_path,
        report,
        before_publish=lambda: _assert_byte_records_unchanged(precommit_bindings),
    )
    print(
        f"wrote {result['available_target_row_count']} held-out observable-event rows to {args.output}"
    )
    return 0


__all__ = [
    "ACTIVATION_BLOCKS",
    "BASE_BLOCK_WIDTHS",
    "CONFIG_PATH",
    "DecoderError",
    "MANDATORY_LIMITATIONS",
    "METADATA_ARRAYS",
    "PRIMARY_HORIZON",
    "PRIMARY_TARGET",
    "VISIBLE_BASELINE_BLOCKS",
    "assemble_label_free_feature_arrays",
    "build_variant_matrices",
    "evaluate_observable_target",
    "fit_weighted_standardizer",
    "frozen_canonical_path_preflight",
    "frozen_lexical_path_preflight",
    "hierarchical_row_weights",
    "load_authenticated_feature_bundle",
    "paired_oof_comparison",
    "paired_oof_comparison_by_repository",
    "repository_heldout_oof",
    "sanitize_visible_baseline_rows",
    "validate_alignment_index",
    "validate_config",
    "validate_feature_arrays",
    "validate_feature_manifest",
    "validate_label_sidecar",
    "weighted_metrics",
]


if __name__ == "__main__":
    raise SystemExit(main())
