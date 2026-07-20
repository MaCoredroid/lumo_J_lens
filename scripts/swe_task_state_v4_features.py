#!/usr/bin/env python3
"""Causal sequence features for a V4 SWE task-state interpreter.

The builder consumes *already filtered, numerically stable* rows.  It never
examines action labels or status metadata: every supplied row updates the
per-task sensor state, including rows whose action is unknown.  Features are
formed before that update, so a row can depend only on earlier stable rows for
the same task.

The forty-value sensor summaries deliberately call V3's frozen
``compact_layer_shape`` helper.  Deltas and EMA deviations are computed in the
original 96-wide layer-by-action space and compacted only afterward.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V3_ANALYZER_PATH = ROOT / "scripts/analyze_swe_task_state_v3.py"
V3_ANALYZER_SHA256 = (
    "53c7d41688f6c5ab21f7ad029d343af06e9b13c777fd2e5517ff8d5254ad9e6c"
)


def _authenticate_v3_source() -> None:
    if (
        not V3_ANALYZER_PATH.is_file()
        or V3_ANALYZER_PATH.is_symlink()
        or hashlib.sha256(V3_ANALYZER_PATH.read_bytes()).hexdigest()
        != V3_ANALYZER_SHA256
    ):
        raise RuntimeError("frozen V3 analyzer byte identity changed")


_authenticate_v3_source()

try:
    from scripts import analyze_swe_task_state_v3 as _V3
except ModuleNotFoundError as error:  # Support direct execution from scripts/.
    if error.name != "scripts":
        raise
    import analyze_swe_task_state_v3 as _V3  # type: ignore[no-redef]

if Path(_V3.__file__).resolve() != V3_ANALYZER_PATH:
    raise RuntimeError("frozen V3 analyzer resolved from an unexpected path")


HISTORY_WIDTH = 14
SENSOR_WIDTH = 96
COMPACT_WIDTH = 40
EMA_ALPHA = 0.5

VARIANTS = (
    "history_only",
    "sequence_j",
    "sequence_logit",
    "sequence_logit_j",
)
VARIANT_WIDTHS = {
    "history_only": 14,
    "sequence_j": 136,
    "sequence_logit": 136,
    "sequence_logit_j": 256,
}

_JACOBIAN = "public_jacobian"
_LOGIT = "ordinary_logit"
_SENSOR_ORDER = (_LOGIT, _JACOBIAN)
_SHARED_FEATURE_NAMES = ("log1p_request_gap", "no_previous_stable_row")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _task_id(row: Mapping[str, Any], position: int) -> str:
    value = row.get("task_id")
    _require(
        isinstance(value, str) and bool(value),
        f"stable row {position} task_id must be nonempty text",
    )
    return value


def _request_index(row: Mapping[str, Any], position: int) -> int:
    value = row.get("task_request_index")
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 1,
        f"stable row {position} task_request_index must be an integer >= 1",
    )
    return value


def _finite_vector(
    row: Mapping[str, Any], key: str, width: int, *, position: int
) -> np.ndarray:
    raw = row.get(key)
    try:
        object_values = np.asarray(raw, dtype=object)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"stable row {position} {key} must be numeric") from error
    _require(
        object_values.shape == (width,),
        f"stable row {position} {key} width must be {width}",
    )
    _require(
        all(
            isinstance(item, Real)
            and not isinstance(item, (bool, np.bool_))
            for item in object_values.flat
        ),
        f"stable row {position} {key} must contain numeric, non-boolean values",
    )
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"stable row {position} {key} must be numeric") from error
    _require(
        bool(np.all(np.isfinite(result))),
        f"stable row {position} {key} must contain only finite values",
    )
    return result


def _compact(values: np.ndarray, *, label: str) -> np.ndarray:
    """Call the frozen V3 compact representation and validate its contract."""

    result = np.asarray(_V3.compact_layer_shape(values), dtype=np.float64)
    _require(
        result.shape == (COMPACT_WIDTH,) and bool(np.all(np.isfinite(result))),
        f"{label} V3 compact layer shape is invalid",
    )
    return result


def _sensor_feature_names(sensor: str) -> list[str]:
    return [
        *_V3.compact_feature_names(f"current_{sensor}"),
        *_V3.compact_feature_names(f"delta_from_previous_stable_{sensor}"),
        *_V3.compact_feature_names(f"deviation_from_prior_ema_{sensor}"),
    ]


def feature_names(variant: str) -> list[str]:
    """Return the exact feature order for one V4 variant."""

    _require(variant in VARIANTS, f"unknown V4 feature variant: {variant}")
    names = list(_V3.HISTORY_FEATURE_NAMES)
    if variant in {"sequence_logit", "sequence_logit_j"}:
        names.extend(_sensor_feature_names(_LOGIT))
    if variant in {"sequence_j", "sequence_logit_j"}:
        names.extend(_sensor_feature_names(_JACOBIAN))
    if variant != "history_only":
        names.extend(_SHARED_FEATURE_NAMES)
    _require(
        len(names) == VARIANT_WIDTHS[variant],
        f"{variant} feature-name width changed",
    )
    return names


class _TaskState:
    def __init__(self) -> None:
        self.previous_index: int | None = None
        self.previous: dict[str, np.ndarray] = {}
        self.ema: dict[str, np.ndarray] = {}
        self.observed_indices: set[int] = set()


def _sensor_triplet(
    current: np.ndarray,
    *,
    sensor: str,
    state: _TaskState,
) -> np.ndarray:
    previous = state.previous.get(sensor)
    prior_ema = state.ema.get(sensor)
    delta = np.zeros_like(current) if previous is None else current - previous
    deviation = np.zeros_like(current) if prior_ema is None else current - prior_ema
    result = np.concatenate(
        [
            _compact(current, label=f"current {sensor}"),
            _compact(delta, label=f"{sensor} delta"),
            _compact(deviation, label=f"{sensor} prior-EMA deviation"),
        ]
    )
    _require(
        result.shape == (3 * COMPACT_WIDTH,)
        and bool(np.all(np.isfinite(result))),
        f"{sensor} sequence feature block is invalid",
    )
    return result


def _update_sensor_state(
    current_by_sensor: Mapping[str, np.ndarray], state: _TaskState
) -> None:
    """Update every sensor after all features for the current row exist."""

    for sensor in _SENSOR_ORDER:
        current = current_by_sensor[sensor]
        prior_ema = state.ema.get(sensor)
        state.previous[sensor] = current.copy()
        state.ema[sensor] = (
            current.copy()
            if prior_ema is None
            else EMA_ALPHA * current + (1.0 - EMA_ALPHA) * prior_ema
        )


def _variants_for_row(
    history: np.ndarray,
    sensor_blocks: Mapping[str, np.ndarray],
    shared: np.ndarray,
    *,
    position: int,
) -> dict[str, list[float]]:
    # Match V3's hybrid convention: ordinary logits precede public Jacobians.
    blocks = {
        "history_only": history,
        "sequence_j": np.concatenate([history, sensor_blocks[_JACOBIAN], shared]),
        "sequence_logit": np.concatenate([history, sensor_blocks[_LOGIT], shared]),
        "sequence_logit_j": np.concatenate(
            [
                history,
                sensor_blocks[_LOGIT],
                sensor_blocks[_JACOBIAN],
                shared,
            ]
        ),
    }
    result: dict[str, list[float]] = {}
    for variant in VARIANTS:
        values = np.asarray(blocks[variant], dtype=np.float64)
        _require(
            values.shape == (VARIANT_WIDTHS[variant],)
            and bool(np.all(np.isfinite(values))),
            f"stable row {position} {variant} feature vector is invalid",
        )
        result[variant] = values.tolist()
    _require(
        tuple(result) == VARIANTS,
        "V4 feature variant order changed",
    )
    return result


def build_feature_rows(
    stable_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach causal V4 features to stable rows in their supplied order.

    The caller owns numerical-stability filtering.  Rows for different tasks
    may be interleaved, but indices for each individual task must be unique and
    strictly increasing in the supplied stable-row stream.  Returned rows are
    shallow copies and preserve arbitrary label/status metadata; those fields
    never enter feature construction or state updates.
    """

    _require(
        isinstance(stable_rows, Sequence)
        and not isinstance(stable_rows, (str, bytes, bytearray)),
        "stable_rows must be a sequence",
    )
    states: dict[str, _TaskState] = {}
    result: list[dict[str, Any]] = []
    for position, source_row in enumerate(stable_rows):
        _require(
            isinstance(source_row, Mapping),
            f"stable row {position} must be a mapping",
        )
        task_id = _task_id(source_row, position)
        request_index = _request_index(source_row, position)
        state = states.setdefault(task_id, _TaskState())
        _require(
            request_index not in state.observed_indices,
            f"duplicate stable request index {request_index} in {task_id}",
        )
        _require(
            state.previous_index is None or request_index > state.previous_index,
            f"stable request indices in {task_id} must be strictly increasing",
        )

        history = _finite_vector(
            source_row, "history", HISTORY_WIDTH, position=position
        )
        current_by_sensor = {
            sensor: _finite_vector(
                source_row, sensor, SENSOR_WIDTH, position=position
            )
            for sensor in _SENSOR_ORDER
        }

        no_previous = state.previous_index is None
        gap = (
            0.0
            if no_previous
            else math.log1p(request_index - int(state.previous_index))
        )
        shared = np.asarray([gap, float(no_previous)], dtype=np.float64)
        sensor_blocks = {
            sensor: _sensor_triplet(
                current_by_sensor[sensor], sensor=sensor, state=state
            )
            for sensor in _SENSOR_ORDER
        }
        features = _variants_for_row(
            history, sensor_blocks, shared, position=position
        )

        # Labels and status metadata are intentionally neither read nor used to
        # decide whether this update occurs.
        _update_sensor_state(current_by_sensor, state)
        state.previous_index = request_index
        state.observed_indices.add(request_index)

        result.append({**dict(source_row), "features": features})

    _require(len(result) == len(stable_rows), "stable row output count changed")
    return result


def build_sequence_features(
    stable_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, list[float]]]:
    """Return only aligned per-row feature mappings."""

    return [row["features"] for row in build_feature_rows(stable_rows)]


__all__ = [
    "COMPACT_WIDTH",
    "EMA_ALPHA",
    "HISTORY_WIDTH",
    "SENSOR_WIDTH",
    "VARIANTS",
    "VARIANT_WIDTHS",
    "V3_ANALYZER_PATH",
    "V3_ANALYZER_SHA256",
    "_authenticate_v3_source",
    "build_feature_rows",
    "build_sequence_features",
    "feature_names",
]
