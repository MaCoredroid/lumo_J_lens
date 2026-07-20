#!/usr/bin/env python3
"""Construct causal sequence features from label-independent projections."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_activation_features.json"
PROJECTION_MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_activation_projection.py"

_spec = importlib.util.spec_from_file_location(
    "swe_task_state_v4_activation_projection_for_features", PROJECTION_MODULE_PATH
)
_projection = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_projection)


class FeatureError(ValueError):
    pass


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict) or set(config) != {
        "schema_version",
        "id",
        "status",
        "projection",
        "temporal",
        "variants",
        "campaign",
        "feature_boundary",
        "claim_scope",
    }:
        raise FeatureError("activation-feature config schema changed")
    if config["schema_version"] != 1 or config["id"] != (
        "swe-task-state-v4-causal-activation-sequence-features"
    ) or config["status"] != "development_only_reserved_validation_closed":
        raise FeatureError("activation-feature config identity changed")
    if config["projection"] != {
        "primary_seed_index": 0,
        "sensitivity_seed_indices": [1, 2, 3],
        "sources": ["raw_residual", "public_j_state"],
        "band_count": 3,
        "stored_width_per_band": 256,
        "folded_width_per_band": 64,
        "current_width_per_source": 192,
    }:
        raise FeatureError("activation projection feature contract changed")
    if config["temporal"] != {
        "features_constructed_before_state_update": True,
        "state_isolated_by_task": True,
        "unstable_rows_do_not_update_state": True,
        "previous_delta": True,
        "prior_ema_deviation": True,
        "ema_alpha": 0.5,
        "gap_feature": "log1p(current_request_index - previous_stable_request_index)",
        "no_previous_flag": True,
    }:
        raise FeatureError("activation temporal feature contract changed")
    if config["variants"] != {
        "raw_activation_current": 192,
        "public_j_activation_current": 192,
        "raw_activation_sequence": 578,
        "public_j_activation_sequence": 578,
        "raw_public_j_activation_sequence": 1154,
    }:
        raise FeatureError("activation feature variant widths changed")
    if config["campaign"] != {
        "projection_chunk_count": 4,
        "projection_chunk_boundary_counts": [517, 370, 414, 407],
        "total_boundary_count": 1708,
        "stable_feature_count": 1606,
        "required_alignment": {
            "kind": "swe_task_state_v4_label_free_alignment_index",
            "schema_version": 1,
            "scope": "grouping_order_and_stability_only_no_labels",
            "config_sha256": (
                "a28a8e1a43a91fff79a5df1b8319467eb4c497224c5294aa50471a415fca664d"
            ),
            "implementation_sha256": (
                "182b87fd1c38fcd1cdffd264922ff474b8b73211025dd6ef2c0d780d1cf3e7ee"
            ),
            "source_sha256s": [
                "8910cc66b6ce7949af53809d60d6a6e4b5c447bfa2e6907aa0cbe01f55459350",
                "e0fbb86323eb7263d334936d05c43bddb4d6bb3a2c6b7d0fb3c1bcc040a91f1e",
                "3210772eac7a1106feb2886e84dc326ff1fc65d766af4a7c293b3d1b9094c5b0",
                "72d968ee8273455a5b0dcb5e665cb24a907197d1e82ce23bc35a5dfc5dea093a",
            ],
            "eligibility_source_sha256": (
                "e1436a9c7fe763a4fab6ae505af17395a4290faefd0734b230d22127bff0cf8c"
            ),
        },
        "required_projection_kind": (
            "swe_task_state_v4_label_independent_activation_projection"
        ),
        "required_projection_schema_version": 1,
        "required_projection_array_keys": [
            "band_statistics",
            "boundary_index",
            "sketches",
            "source_id_sha256",
            "token_ids_sha256",
        ],
        "output_tensor_keys": [
            "global_index",
            "public_j_activation_current",
            "public_j_activation_sequence",
            "raw_activation_current",
            "raw_activation_sequence",
            "raw_public_j_activation_sequence",
        ],
        "output_dtypes": {
            "global_index": "little-endian-int64",
            "public_j_activation_current": "little-endian-float64",
            "public_j_activation_sequence": "little-endian-float64",
            "raw_activation_current": "little-endian-float64",
            "raw_activation_sequence": "little-endian-float64",
            "raw_public_j_activation_sequence": "little-endian-float64",
        },
        "authentication": {
            "require_projection_manifest_sha256": True,
            "require_projection_data_sha256": True,
            "require_projection_manifest_status_passed": True,
            "require_projection_reload_verified": True,
            "require_alignment_index_sha256": True,
            "require_exact_chunk_order": True,
            "require_exact_source_id_order_and_coverage": True,
            "require_pre_and_post_hash_equality": True,
            "forbidden_path_fragments": ["reserved", "validation"],
            "no_clobber": True,
        },
    }:
        raise FeatureError("activation feature campaign contract changed")
    boundary = config["feature_boundary"]
    if boundary != {
        "allowed_numeric_inputs": [
            "primary_seed_raw_residual_sketches",
            "primary_seed_public_j_state_sketches",
        ],
        "grouping_or_authentication_only": [
            "source_id_sha256",
            "task_id_sha256",
            "repository",
            "request_index",
            "stable_feature_eligible",
            "projection_boundary_index",
            "artifact_paths_and_hashes",
        ],
        "labels_or_outcomes_accepted": False,
        "semantic_ids_as_features_forbidden": True,
        "repository_as_feature_forbidden": True,
        "request_index_as_semantic_feature_forbidden": True,
    }:
        raise FeatureError("activation feature boundary changed")
    claims = config["claim_scope"]
    if set(claims) != {
        "private_chain_of_thought_reconstructed",
        "cot_like_observable_event_decoding_established",
        "emotion_decoding_established",
        "causal_interpretation_established",
        "incremental_value_over_visible_baselines_established",
    } or any(value is not False for value in claims.values()):
        raise FeatureError("activation features cannot establish claims")
    return config


def validate_alignment_index(index: Any) -> list[dict[str, Any]]:
    if not isinstance(index, dict) or set(index) != {
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
    }:
        raise FeatureError("alignment index schema changed")
    if (
        index["schema_version"] != 1
        or index["kind"] != "swe_task_state_v4_label_free_alignment_index"
        or index["status"] != "passed"
        or index["scope"] != "grouping_order_and_stability_only_no_labels"
        or index["row_count"] != 1708
        or index["stable_row_count"] != 1606
    ):
        raise FeatureError("alignment index identity or counts changed")
    rows = index["rows"]
    if not isinstance(rows, list) or len(rows) != index["row_count"]:
        raise FeatureError("alignment index rows changed")
    expected_keys = {
        "global_index",
        "source_id_sha256",
        "task_id_sha256",
        "repository",
        "request_index",
        "stable_feature_eligible",
    }
    seen_source_ids: set[str] = set()
    seen_task_requests: set[tuple[str, str, int]] = set()
    previous_request_by_task: dict[tuple[str, str], int] = {}
    request_indices_by_task: dict[tuple[str, str], list[int]] = {}
    repository_by_task_id: dict[str, str] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise FeatureError(f"alignment row {position} allowlist changed")
        source_id = row["source_id_sha256"]
        task_id = row["task_id_sha256"]
        repository = row["repository"]
        request_index = row["request_index"]
        if (
            row["global_index"] != position
            or not isinstance(source_id, str)
            or len(source_id) != 64
            or any(character not in "0123456789abcdef" for character in source_id)
            or not isinstance(task_id, str)
            or len(task_id) != 64
            or any(character not in "0123456789abcdef" for character in task_id)
            or isinstance(request_index, bool)
            or not isinstance(request_index, int)
            or request_index < 1
            or not isinstance(repository, str)
            or not repository
            or not isinstance(row["stable_feature_eligible"], bool)
        ):
            raise FeatureError(f"alignment row {position} is invalid")
        if task_id in repository_by_task_id and repository_by_task_id[task_id] != repository:
            raise FeatureError("one alignment task maps to multiple repositories")
        repository_by_task_id[task_id] = repository
        task_key = (repository, task_id)
        if source_id in seen_source_ids or (*task_key, request_index) in seen_task_requests:
            raise FeatureError("alignment index contains duplicate identities")
        previous_request = previous_request_by_task.get(task_key)
        if previous_request is not None and request_index <= previous_request:
            raise FeatureError("alignment task request order changed")
        seen_source_ids.add(source_id)
        seen_task_requests.add((*task_key, request_index))
        previous_request_by_task[task_key] = request_index
        request_indices_by_task.setdefault(task_key, []).append(request_index)
    for request_indices in request_indices_by_task.values():
        if request_indices != list(range(1, len(request_indices) + 1)):
            raise FeatureError("alignment task requests are not complete and consecutive")
    if sum(row["stable_feature_eligible"] for row in rows) != 1606:
        raise FeatureError("alignment stable count changed")
    return rows


def concatenate_projection_arrays(
    chunks: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not isinstance(chunks, Sequence) or not chunks:
        raise FeatureError("projection chunks must be a nonempty sequence")
    expected_keys = {
        "sketches",
        "band_statistics",
        "source_id_sha256",
        "token_ids_sha256",
        "boundary_index",
    }
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in expected_keys}
    for chunk_index, chunk in enumerate(chunks):
        if set(chunk) != expected_keys:
            raise FeatureError(f"projection chunk {chunk_index} keys changed")
        count = len(np.asarray(chunk["source_id_sha256"]))
        if count < 1 or any(len(np.asarray(chunk[key])) != count for key in expected_keys):
            raise FeatureError(f"projection chunk {chunk_index} row counts differ")
        if not np.array_equal(
            np.asarray(chunk["boundary_index"]), np.arange(count, dtype=np.int64)
        ):
            raise FeatureError(f"projection chunk {chunk_index} boundary order changed")
        for key in expected_keys:
            arrays[key].append(np.asarray(chunk[key]))
    return {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}


def primary_current_matrices(
    projections: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    sketches = np.asarray(projections.get("sketches"))
    if sketches.shape != (1708, 4, 2, 3, 256) or sketches.dtype != np.float32:
        raise FeatureError("assembled primary sketch geometry changed")
    raw = _projection.extract_primary_current_features(
        projections, seed_index=0, source="raw_residual"
    )
    public_j = _projection.extract_primary_current_features(
        projections, seed_index=0, source="public_j_state"
    )
    if raw.shape != (1708, 192) or public_j.shape != (1708, 192):
        raise FeatureError("primary current matrix geometry changed")
    return raw.astype(np.float64), public_j.astype(np.float64)


class _TaskState:
    def __init__(self) -> None:
        self.previous_index: int | None = None
        self.previous: dict[str, np.ndarray] = {}
        self.ema: dict[str, np.ndarray] = {}
        self.seen_indices: set[int] = set()


def build_causal_activation_features(
    *,
    alignment_rows: Sequence[Mapping[str, Any]],
    raw_current: np.ndarray,
    public_j_current: np.ndarray,
    ema_alpha: float = 0.5,
) -> dict[str, np.ndarray]:
    """Build X before update; no label object is accepted by this API."""

    if len(alignment_rows) != 1708:
        raise FeatureError("alignment row count changed")
    if raw_current.shape != (1708, 192) or public_j_current.shape != (1708, 192):
        raise FeatureError("activation current matrices must be 1708x192")
    if not (np.all(np.isfinite(raw_current)) and np.all(np.isfinite(public_j_current))):
        raise FeatureError("activation current matrices are non-finite")
    if ema_alpha != 0.5:
        raise FeatureError("EMA alpha changed")
    states: dict[tuple[str, str], _TaskState] = {}
    outputs: dict[str, list[np.ndarray]] = {
        "raw_activation_current": [],
        "public_j_activation_current": [],
        "raw_activation_sequence": [],
        "public_j_activation_sequence": [],
        "raw_public_j_activation_sequence": [],
    }
    retained_indices = []
    for global_index, row in enumerate(alignment_rows):
        task_key = (str(row["repository"]), str(row["task_id_sha256"]))
        request_index = int(row["request_index"])
        state = states.setdefault(task_key, _TaskState())
        if request_index in state.seen_indices:
            raise FeatureError("duplicate request index in task feature stream")
        if state.previous_index is not None and request_index <= state.previous_index:
            raise FeatureError("task request indices are not strictly increasing")
        # Unstable rows are omitted and deliberately do not update task state.
        if not row["stable_feature_eligible"]:
            continue
        raw = np.asarray(raw_current[global_index], dtype=np.float64)
        public_j = np.asarray(public_j_current[global_index], dtype=np.float64)
        triplets = {}
        for source, current in (("raw", raw), ("public_j", public_j)):
            previous = state.previous.get(source)
            prior_ema = state.ema.get(source)
            delta = np.zeros_like(current) if previous is None else current - previous
            deviation = np.zeros_like(current) if prior_ema is None else current - prior_ema
            triplets[source] = np.concatenate((current, delta, deviation))
        no_previous = state.previous_index is None
        gap = (
            0.0
            if no_previous
            else math.log1p(request_index - int(state.previous_index))
        )
        shared = np.asarray([gap, float(no_previous)], dtype=np.float64)
        outputs["raw_activation_current"].append(raw.copy())
        outputs["public_j_activation_current"].append(public_j.copy())
        outputs["raw_activation_sequence"].append(
            np.concatenate((triplets["raw"], shared))
        )
        outputs["public_j_activation_sequence"].append(
            np.concatenate((triplets["public_j"], shared))
        )
        outputs["raw_public_j_activation_sequence"].append(
            np.concatenate((triplets["raw"], triplets["public_j"], shared))
        )
        retained_indices.append(global_index)

        for source, current in (("raw", raw), ("public_j", public_j)):
            prior_ema = state.ema.get(source)
            state.previous[source] = current.copy()
            state.ema[source] = (
                current.copy()
                if prior_ema is None
                else ema_alpha * current + (1.0 - ema_alpha) * prior_ema
            )
        state.previous_index = request_index
        state.seen_indices.add(request_index)

    result = {
        name: np.asarray(values, dtype=np.float64) for name, values in outputs.items()
    }
    result["global_index"] = np.asarray(retained_indices, dtype=np.int64)
    expected_widths = {
        "raw_activation_current": 192,
        "public_j_activation_current": 192,
        "raw_activation_sequence": 578,
        "public_j_activation_sequence": 578,
        "raw_public_j_activation_sequence": 1154,
    }
    for name, width in expected_widths.items():
        if result[name].shape != (1606, width) or not np.all(np.isfinite(result[name])):
            raise FeatureError(f"{name} output geometry or finiteness changed")
    if result["global_index"].shape != (1606,):
        raise FeatureError("retained global index geometry changed")
    return result


__all__ = [
    "FeatureError",
    "build_causal_activation_features",
    "concatenate_projection_arrays",
    "primary_current_matrices",
    "validate_alignment_index",
    "validate_config",
]
