#!/usr/bin/env python3
"""Pure forecast transformations for the V4 SWE state interpreter.

The caller supplies probabilities, frozen three-class labels, and already
computed positive hierarchical row weights.  This module only selects a
temperature from a finite declared grid by weighted multiclass NLL and applies
the resulting label-free settings.  It also exposes the frozen label-free
geometric probability pool used before calibration.  It performs no file I/O
and deliberately does not implement decision offsets.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


CLASSES = ("inspect", "edit", "check_or_finish")
CLASS_INDEX = {class_id: index for index, class_id in enumerate(CLASSES)}
SCHEMA_VERSION = 1
CALIBRATOR_IDENTITY = "swe-task-state-v4-scalar-temperature-v1"
SELECTION_OBJECTIVE = "caller-weighted-multiclass-nll"
PROBABILITY_TOLERANCE = 1e-12

_SETTINGS_KEYS = frozenset(
    {
        "schema_version",
        "calibrator_identity",
        "classes_in_order",
        "selection_objective",
        "temperature_grid",
        "temperature",
    }
)
_MIN_POSITIVE = np.nextafter(np.float64(0.0), np.float64(1.0))
_MIN_LOG = math.log(float(_MIN_POSITIVE))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_real(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_)),
        f"{label} must be a real numeric value, not a boolean or string",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _numeric_array(value: Any, label: str) -> np.ndarray:
    """Inspect values before conversion so strings and booleans cannot coerce."""

    try:
        object_values = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a rectangular numeric array") from error
    _require(
        all(
            isinstance(item, (int, float, np.integer, np.floating))
            and not isinstance(item, (bool, np.bool_))
            for item in object_values.flat
        ),
        f"{label} must contain real numeric values, not booleans or strings",
    )
    try:
        return np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a rectangular numeric array") from error


def validate_probability_matrix(
    value: Any,
    label: str = "probabilities",
    *,
    expected_rows: int | None = None,
) -> np.ndarray:
    """Return a float64 copy of a strictly positive three-class matrix."""

    result = _numeric_array(value, label)
    _require(
        result.ndim == 2
        and result.shape[0] > 0
        and result.shape[1] == len(CLASSES),
        f"{label} must have shape (nonzero_rows, {len(CLASSES)})",
    )
    if expected_rows is not None:
        _require(
            isinstance(expected_rows, int) and not isinstance(expected_rows, bool),
            "expected row count must be an integer",
        )
        _require(
            result.shape[0] == expected_rows,
            f"{label} must have the expected row count",
        )
    _require(
        np.all(np.isfinite(result)) and np.all(result > 0.0),
        f"{label} must contain only finite, strictly positive probabilities",
    )
    row_sums = result.sum(axis=1, dtype=np.float64)
    _require(
        np.all(
            np.isclose(
                row_sums,
                1.0,
                rtol=PROBABILITY_TOLERANCE,
                atol=PROBABILITY_TOLERANCE,
            )
        ),
        f"{label} rows must sum to one",
    )
    return result


def validate_weights(value: Any, row_count: int) -> np.ndarray:
    """Return a normalized copy of caller-supplied positive row weights."""

    _require(
        isinstance(row_count, int) and not isinstance(row_count, bool) and row_count > 0,
        "row count must be a positive integer",
    )
    result = _numeric_array(value, "weights")
    _require(
        result.shape == (row_count,),
        "weights must have exactly one value per probability row",
    )
    _require(
        np.all(np.isfinite(result)) and np.all(result > 0.0),
        "weights must be finite and strictly positive",
    )
    total = float(result.sum(dtype=np.float64))
    _require(math.isfinite(total) and total > 0.0, "weight mass must be finite and positive")
    result /= total
    _require(
        np.all(np.isfinite(result))
        and np.all(result > 0.0)
        and math.isclose(
            float(result.sum(dtype=np.float64)),
            1.0,
            rel_tol=PROBABILITY_TOLERANCE,
            abs_tol=PROBABILITY_TOLERANCE,
        ),
        "weight normalization failed",
    )
    return result


def _validated_labels(value: Sequence[str], row_count: int) -> np.ndarray:
    _require(
        not isinstance(value, (str, bytes, bytearray, Mapping)),
        "labels must be a sequence with one frozen class per row",
    )
    try:
        raw = list(value)
    except TypeError as error:
        raise ValueError("labels must be a finite sequence") from error
    labels = np.asarray(raw, dtype=object)
    _require(labels.shape == (row_count,), "labels must have one value per row")
    _require(
        all(isinstance(item, str) and item in CLASS_INDEX for item in raw),
        "labels must use only the frozen three classes",
    )
    _require(
        set(raw) == set(CLASSES),
        "calibration labels must contain all three frozen classes",
    )
    return labels


def canonicalize_temperature_grid(value: Sequence[float]) -> tuple[float, ...]:
    """Validate and sort a finite, unique, strictly positive temperature grid."""

    _require(
        not isinstance(
            value,
            (str, bytes, bytearray, Mapping, set, frozenset),
        ),
        "temperature grid must be an ordered finite sequence",
    )
    try:
        raw = list(value)
    except TypeError as error:
        raise ValueError("temperature grid must be an ordered finite sequence") from error
    _require(bool(raw), "temperature grid must not be empty")
    temperatures = tuple(
        _finite_real(item, "temperature grid value") for item in raw
    )
    _require(
        all(item > 0.0 for item in temperatures),
        "temperature grid values must be strictly positive",
    )
    _require(
        len(temperatures) == len(set(temperatures)),
        "temperature grid values must be unique",
    )
    return tuple(sorted(temperatures))


def temperature_scale_probabilities(
    probabilities: Any,
    temperature: float,
) -> np.ndarray:
    """Apply scalar temperature scaling and return a new probability matrix."""

    values = validate_probability_matrix(probabilities)
    selected_temperature = _finite_real(temperature, "temperature")
    _require(selected_temperature > 0.0, "temperature must be strictly positive")
    if selected_temperature == 1.0:
        return values

    # Center before division so even the smallest positive temperature retains
    # a finite zero for each row's largest class.  Clamp only the low tail to
    # preserve strictly positive float64 probabilities under exponentiation.
    centered_log_probabilities = np.log(values)
    centered_log_probabilities -= centered_log_probabilities.max(
        axis=1, keepdims=True
    )
    with np.errstate(over="ignore", divide="ignore", under="ignore"):
        scaled = centered_log_probabilities / selected_temperature
    scaled = np.maximum(scaled, _MIN_LOG)
    result = np.exp(scaled)
    result /= result.sum(axis=1, keepdims=True, dtype=np.float64)
    return validate_probability_matrix(result, "temperature-scaled probabilities")


def geometric_pool_probabilities(
    first: Any,
    second: Any,
    *,
    alpha: float,
) -> np.ndarray:
    """Return the normalized logarithmic opinion pool of two forecasts.

    The result is proportional to the exponential of the alpha-weighted
    elementwise log probabilities. Endpoints return exact validated copies of
    their corresponding inputs. This transformation accepts no labels and
    never mutates caller inputs.
    """

    first_values = validate_probability_matrix(first, "first probabilities")
    second_values = validate_probability_matrix(
        second,
        "second probabilities",
        expected_rows=len(first_values),
    )
    blend = _finite_real(alpha, "geometric-pool alpha")
    _require(0.0 <= blend <= 1.0, "geometric-pool alpha must lie in [0, 1]")
    if blend == 0.0:
        return first_values
    if blend == 1.0:
        return second_values

    pooled_log = (1.0 - blend) * np.log(first_values) + blend * np.log(
        second_values
    )
    pooled_log -= pooled_log.max(axis=1, keepdims=True)
    result = np.exp(pooled_log)
    result /= result.sum(axis=1, keepdims=True, dtype=np.float64)
    return validate_probability_matrix(result, "geometrically pooled probabilities")


def weighted_multiclass_nll(
    probabilities: Any,
    labels: Sequence[str],
    weights: Any,
) -> float:
    """Compute NLL using caller-supplied positive hierarchical row weights."""

    values = validate_probability_matrix(probabilities)
    validated_labels = _validated_labels(labels, len(values))
    normalized_weights = validate_weights(weights, len(values))
    label_indices = np.asarray(
        [CLASS_INDEX[str(label)] for label in validated_labels], dtype=np.int64
    )
    true_probabilities = values[np.arange(len(values)), label_indices]
    result = -float(np.dot(normalized_weights, np.log(true_probabilities)))
    _require(math.isfinite(result), "weighted multiclass NLL must be finite")
    return result


def select_temperature_settings(
    probabilities: Any,
    labels: Sequence[str],
    weights: Any,
    *,
    temperature_grid: Sequence[float],
) -> dict[str, Any]:
    """Select the frozen-grid temperature minimizing weighted multiclass NLL.

    Exact NLL ties prefer the candidate numerically nearest one, then the lower
    temperature.  Grid canonicalization makes selection independent of the
    order supplied by the caller.
    """

    values = validate_probability_matrix(probabilities)
    validated_labels = _validated_labels(labels, len(values))
    normalized_weights = validate_weights(weights, len(values))
    grid = canonicalize_temperature_grid(temperature_grid)
    label_indices = np.asarray(
        [CLASS_INDEX[str(label)] for label in validated_labels], dtype=np.int64
    )

    candidates: list[tuple[float, float, float]] = []
    for temperature in grid:
        calibrated = temperature_scale_probabilities(values, temperature)
        true_probabilities = calibrated[np.arange(len(values)), label_indices]
        nll = -float(np.dot(normalized_weights, np.log(true_probabilities)))
        _require(math.isfinite(nll), "candidate weighted multiclass NLL must be finite")
        candidates.append((nll, abs(temperature - 1.0), temperature))
    selected_temperature = min(candidates)[2]

    return {
        "schema_version": SCHEMA_VERSION,
        "calibrator_identity": CALIBRATOR_IDENTITY,
        "classes_in_order": list(CLASSES),
        "selection_objective": SELECTION_OBJECTIVE,
        "temperature_grid": list(grid),
        "temperature": selected_temperature,
    }


def _validate_settings(
    settings: Any,
    frozen_grid: tuple[float, ...],
) -> float:
    _require(isinstance(settings, Mapping), "calibration settings must be a mapping")
    _require(
        set(settings) == _SETTINGS_KEYS,
        "calibration settings keys do not match the frozen schema",
    )
    schema_version = settings["schema_version"]
    _require(
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version == SCHEMA_VERSION,
        "calibration settings have the wrong schema identity",
    )
    _require(
        settings["calibrator_identity"] == CALIBRATOR_IDENTITY
        and isinstance(settings["calibrator_identity"], str),
        "calibration settings have the wrong calibrator identity",
    )
    _require(
        settings["classes_in_order"] == list(CLASSES)
        and isinstance(settings["classes_in_order"], list),
        "calibration settings have the wrong class identity or order",
    )
    _require(
        settings["selection_objective"] == SELECTION_OBJECTIVE
        and isinstance(settings["selection_objective"], str),
        "calibration settings have the wrong selection objective identity",
    )

    stored_grid_value = settings["temperature_grid"]
    _require(
        isinstance(stored_grid_value, list),
        "stored temperature grid must be a canonical list",
    )
    stored_grid = canonicalize_temperature_grid(stored_grid_value)
    _require(
        stored_grid_value == list(stored_grid),
        "stored temperature grid is not canonical",
    )
    _require(
        stored_grid == frozen_grid,
        "stored temperature grid does not match the frozen grid identity",
    )

    temperature = _finite_real(settings["temperature"], "selected temperature")
    _require(temperature > 0.0, "selected temperature must be strictly positive")
    _require(
        temperature in frozen_grid,
        "selected temperature is off the frozen grid",
    )
    return temperature


def apply_temperature_settings(
    probabilities: Any,
    settings: Mapping[str, Any],
    *,
    temperature_grid: Sequence[float],
) -> np.ndarray:
    """Apply authenticated frozen-grid settings without accepting any labels."""

    values = validate_probability_matrix(probabilities)
    frozen_grid = canonicalize_temperature_grid(temperature_grid)
    temperature = _validate_settings(settings, frozen_grid)
    return temperature_scale_probabilities(values, temperature)


# Short aliases keep analyzer call sites readable without creating another
# implementation path.
select_temperature = select_temperature_settings
apply_temperature_calibration = apply_temperature_settings
