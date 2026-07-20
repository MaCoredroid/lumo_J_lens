#!/usr/bin/env python3
"""Pure decision and abstention helpers for a V4 SWE task-state readout.

This module deliberately performs no file I/O, model fitting, fold creation, or
label discovery.  A nested evaluator supplies two already-cross-fitted
probability matrices, labels, and deterministic row weights.  The helpers then
select and apply only the finite settings declared by that evaluator:

* a convex blend of the ordinary-logit and direct logit-plus-J branches;
* additive edit/check-or-finish offsets in log-probability space, selected
  under conjunctive class-recall and balanced-accuracy floors; and
* one confidence threshold for each *predicted* class.

Decision application and threshold application never accept labels.  This
keeps the deployment path mechanically separate from the inner-fold labels
used to select settings.

The normalized offset scores are decision probabilities, not a claim of final
posterior calibration.  Their NLL is only the declared tertiary selection
tie-break after recall and balanced-accuracy floors, accuracy, and balanced
accuracy.  A nested
analyzer may temperature-calibrate a separate confidence matrix and pass that
matrix to the abstention helpers below.
"""

from __future__ import annotations

from itertools import product
import math
from typing import Any, Mapping, Sequence

import numpy as np


CLASSES = ("inspect", "edit", "check_or_finish")
CLASS_INDEX = {class_id: index for index, class_id in enumerate(CLASSES)}
SCHEMA_VERSION = 1
PROBABILITY_TOLERANCE = 1e-12


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_)),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _numeric_array(value: Any, label: str) -> np.ndarray:
    """Reject boolean/string/complex coercions before making a float64 copy."""

    try:
        object_values = np.asarray(value, dtype=object)
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a rectangular numeric array") from error
    valid = all(
        isinstance(item, (int, float, np.integer, np.floating))
        and not isinstance(item, (bool, np.bool_))
        for item in object_values.flat
    )
    _require(valid, f"{label} must contain numeric, non-boolean values")
    return raw


def validate_probability_matrix(
    value: Any,
    label: str = "probabilities",
    *,
    expected_rows: int | None = None,
) -> np.ndarray:
    """Return a float64 copy of one strictly positive three-class matrix."""

    _numeric_array(value, label)
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a numeric matrix") from error
    _require(
        result.ndim == 2
        and result.shape[0] > 0
        and result.shape[1] == len(CLASSES),
        f"{label} must have shape (nonzero_rows, {len(CLASSES)})",
    )
    if expected_rows is not None:
        _require(
            result.shape[0] == expected_rows,
            f"{label} row count differs from the paired matrix",
        )
    _require(
        np.all(np.isfinite(result)) and np.all(result > 0.0),
        f"{label} must contain only finite, strictly positive values",
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
    """Validate and deterministically normalize one positive row-weight vector."""

    _numeric_array(value, "weights")
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("weights must be a numeric vector") from error
    _require(
        result.shape == (row_count,)
        and np.all(np.isfinite(result))
        and np.all(result > 0.0),
        "weights must be finite, strictly positive, and have one value per row",
    )
    total = float(result.sum(dtype=np.float64))
    _require(math.isfinite(total) and total > 0.0, "weight mass must be positive")
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
    labels = np.asarray(list(value), dtype=object)
    _require(labels.shape == (row_count,), "labels must have one value per row")
    _require(
        all(isinstance(item, str) and item in CLASS_INDEX for item in labels),
        "labels must use only the frozen three classes",
    )
    _require(
        set(labels.tolist()) == set(CLASSES),
        "selection rows must contain positive support for every class",
    )
    return labels


def _validated_grid(
    value: Sequence[float],
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, ...]:
    try:
        raw = list(value)
    except TypeError as error:
        raise ValueError(f"{label} must be a finite sequence") from error
    _require(bool(raw), f"{label} must not be empty")
    values = tuple(_finite(item, f"{label} value") for item in raw)
    _require(len(values) == len(set(values)), f"{label} values must be unique")
    if minimum is not None:
        _require(all(item >= minimum for item in values), f"{label} is below {minimum}")
    if maximum is not None:
        _require(all(item <= maximum for item in values), f"{label} exceeds {maximum}")
    return tuple(sorted(values))


def _validated_recall_floors(value: Mapping[str, float]) -> dict[str, float]:
    floors = _mapping(value, "recall floors")
    _require(set(floors) == set(CLASSES), "recall floors must name every class exactly")
    result = {
        class_id: _finite(floors[class_id], f"{class_id} recall floor")
        for class_id in CLASSES
    }
    _require(
        all(0.0 <= item <= 1.0 for item in result.values()),
        "recall floors must lie in [0, 1]",
    )
    return result


def _labels_as_indices(labels: np.ndarray) -> np.ndarray:
    return np.asarray([CLASS_INDEX[str(label)] for label in labels], dtype=np.int64)


def fuse_probabilities(
    logit_probabilities: Any,
    logit_j_probabilities: Any,
    alpha: float,
) -> np.ndarray:
    """Convexly blend two direct probability matrices.

    ``alpha=0`` is the ordinary-logit branch and ``alpha=1`` is the direct
    logit-plus-J branch.  Intermediate values are probability-space mixtures,
    not a reconstruction from unavailable estimator logits.
    """

    ordinary = validate_probability_matrix(logit_probabilities, "logit probabilities")
    direct = validate_probability_matrix(
        logit_j_probabilities,
        "direct logit-plus-J probabilities",
        expected_rows=len(ordinary),
    )
    blend = _finite(alpha, "fusion alpha")
    _require(0.0 <= blend <= 1.0, "fusion alpha must lie in [0, 1]")
    if blend == 0.0:
        result = ordinary
    elif blend == 1.0:
        result = direct
    else:
        result = (1.0 - blend) * ordinary + blend * direct
    return validate_probability_matrix(result, "fused probabilities")


def apply_class_logit_offsets(
    probabilities: Any,
    *,
    edit_offset: float,
    check_or_finish_offset: float,
) -> np.ndarray:
    """Apply inspect-fixed additive offsets and return normalized probabilities."""

    values = validate_probability_matrix(probabilities)
    offsets = np.asarray(
        [
            0.0,
            _finite(edit_offset, "edit logit offset"),
            _finite(check_or_finish_offset, "check-or-finish logit offset"),
        ],
        dtype=np.float64,
    )
    shifted = np.log(values) + offsets
    shifted -= shifted.max(axis=1, keepdims=True)
    result = np.exp(shifted)
    result /= result.sum(axis=1, keepdims=True, dtype=np.float64)
    return validate_probability_matrix(result, "offset decision probabilities")


def _decision_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    label_indices = _labels_as_indices(labels)
    predicted_indices = np.argmax(probabilities, axis=1)
    correct = predicted_indices == label_indices
    recalls: dict[str, float] = {}
    for class_id, class_index in CLASS_INDEX.items():
        mask = label_indices == class_index
        class_mass = float(weights[mask].sum(dtype=np.float64))
        _require(class_mass > 0.0, f"selection rows lack class {class_id}")
        recalls[class_id] = float(
            np.sum(weights[mask] * correct[mask], dtype=np.float64) / class_mass
        )
    return {
        "accuracy": float(np.sum(weights * correct, dtype=np.float64)),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "multiclass_negative_log_likelihood": float(
            -np.sum(
                weights
                * np.log(probabilities[np.arange(len(labels)), label_indices]),
                dtype=np.float64,
            )
        ),
        "per_class_recall": recalls,
    }


def _decision_complexity_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    edit = float(candidate["class_logit_offsets"]["edit"])
    check = float(candidate["class_logit_offsets"]["check_or_finish"])
    return (
        abs(edit) + abs(check),
        abs(edit),
        abs(check),
        float(candidate["alpha"]),
        edit,
        check,
    )


def _passing_decision_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    return (
        -float(metrics["accuracy"]),
        -float(metrics["balanced_accuracy"]),
        float(metrics["multiclass_negative_log_likelihood"]),
        *_decision_complexity_key(candidate),
    )


def _fallback_decision_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    return (
        -float(candidate["decision_floor_count"]),
        float(candidate["decision_shortfall"]),
        -float(metrics["accuracy"]),
        -float(metrics["balanced_accuracy"]),
        float(metrics["multiclass_negative_log_likelihood"]),
        *_decision_complexity_key(candidate),
    )


def select_decision_settings(
    logit_probabilities: Any,
    logit_j_probabilities: Any,
    labels: Sequence[str],
    weights: Any,
    *,
    alpha_grid: Sequence[float],
    edit_offset_grid: Sequence[float],
    check_or_finish_offset_grid: Sequence[float],
    recall_floors: Mapping[str, float],
    balanced_accuracy_minimum: float,
    include_candidates: bool = True,
) -> dict[str, Any]:
    """Select fusion and class offsets from inner-fold predictions only.

    Candidates meeting every weighted recall floor and the weighted balanced-
    accuracy floor are ranked by accuracy, balanced accuracy, and then NLL.
    Remaining ties prefer smaller offsets and less reliance on the added J
    branch.  If no candidate meets every floor, the returned setting is
    explicitly marked as a fallback; it first maximizes the number of floors
    met, then minimizes total shortfall before applying the same metric and
    complexity ordering.  Applying a fallback is allowed so an outer fold can
    still emit diagnostic predictions, but ``selected_under_floors=False`` is
    explicit evidence that the requested inner operating point was unavailable.
    """

    ordinary = validate_probability_matrix(logit_probabilities, "logit probabilities")
    direct = validate_probability_matrix(
        logit_j_probabilities,
        "direct logit-plus-J probabilities",
        expected_rows=len(ordinary),
    )
    label_values = _validated_labels(labels, len(ordinary))
    normalized_weights = validate_weights(weights, len(ordinary))
    alphas = _validated_grid(alpha_grid, "alpha grid", minimum=0.0, maximum=1.0)
    edit_offsets = _validated_grid(edit_offset_grid, "edit offset grid")
    check_offsets = _validated_grid(
        check_or_finish_offset_grid, "check-or-finish offset grid"
    )
    floors = _validated_recall_floors(recall_floors)
    balanced_accuracy_floor = _finite(
        balanced_accuracy_minimum, "balanced-accuracy minimum"
    )
    _require(
        0.0 <= balanced_accuracy_floor <= 1.0,
        "balanced-accuracy minimum must lie in [0, 1]",
    )

    candidates: list[dict[str, Any]] = []
    fused_by_alpha = {
        alpha: fuse_probabilities(ordinary, direct, alpha) for alpha in alphas
    }
    for alpha, edit_offset, check_offset in product(
        alphas, edit_offsets, check_offsets
    ):
        decision_probabilities = apply_class_logit_offsets(
            fused_by_alpha[alpha],
            edit_offset=edit_offset,
            check_or_finish_offset=check_offset,
        )
        metrics = _decision_metrics(
            label_values, decision_probabilities, normalized_weights
        )
        shortfalls = {
            class_id: max(
                0.0,
                floors[class_id] - float(metrics["per_class_recall"][class_id]),
            )
            for class_id in CLASSES
        }
        floor_count = sum(shortfall == 0.0 for shortfall in shortfalls.values())
        balanced_accuracy_shortfall = max(
            0.0,
            balanced_accuracy_floor - float(metrics["balanced_accuracy"]),
        )
        meets_balanced_accuracy_floor = balanced_accuracy_shortfall == 0.0
        candidates.append(
            {
                "alpha": alpha,
                "class_logit_offsets": {
                    "inspect": 0.0,
                    "edit": edit_offset,
                    "check_or_finish": check_offset,
                },
                "metrics": metrics,
                "recall_shortfall_by_class": shortfalls,
                "recall_shortfall": float(sum(shortfalls.values())),
                "recall_floor_count": int(floor_count),
                "meets_recall_floors": floor_count == len(CLASSES),
                "balanced_accuracy_shortfall": balanced_accuracy_shortfall,
                "meets_balanced_accuracy_floor": meets_balanced_accuracy_floor,
                "decision_shortfall": float(
                    sum(shortfalls.values()) + balanced_accuracy_shortfall
                ),
                "decision_floor_count": int(
                    floor_count + int(meets_balanced_accuracy_floor)
                ),
                "meets_decision_floors": (
                    floor_count == len(CLASSES)
                    and meets_balanced_accuracy_floor
                ),
            }
        )

    passing = [
        candidate for candidate in candidates if candidate["meets_decision_floors"]
    ]
    if passing:
        selected = min(passing, key=_passing_decision_key)
        fallback_used = False
    else:
        selected = min(candidates, key=_fallback_decision_key)
        fallback_used = True

    return {
        "schema_version": SCHEMA_VERSION,
        "classes_in_order": list(CLASSES),
        "alpha": float(selected["alpha"]),
        "class_logit_offsets": dict(selected["class_logit_offsets"]),
        "selected_under_floors": not fallback_used,
        "fallback_used": fallback_used,
        "recall_floors": floors,
        "balanced_accuracy_minimum": balanced_accuracy_floor,
        "selected_metrics": dict(selected["metrics"]),
        "selected_recall_shortfall_by_class": dict(
            selected["recall_shortfall_by_class"]
        ),
        "selected_balanced_accuracy_shortfall": float(
            selected["balanced_accuracy_shortfall"]
        ),
        "selected_decision_shortfall": float(selected["decision_shortfall"]),
        "selection_rule": (
            "all_weighted_recall_and_balanced_accuracy_floors_then_accuracy_"
            "balanced_accuracy_nll_then_minimum_offset_and_alpha_complexity"
        ),
        "fallback_rule": (
            "maximum_decision_floor_count_then_minimum_total_decision_shortfall_"
            "then_accuracy_balanced_accuracy_nll_then_minimum_complexity"
        ),
        "grids": {
            "alpha": list(alphas),
            "edit_offset": list(edit_offsets),
            "check_or_finish_offset": list(check_offsets),
        },
        "candidate_count": len(candidates),
        **({"candidates": candidates} if include_candidates else {}),
    }


def _validated_decision_settings(
    settings: Any,
    *,
    alpha_grid: Sequence[float],
    edit_offset_grid: Sequence[float],
    check_or_finish_offset_grid: Sequence[float],
) -> tuple[float, float, float]:
    value = _mapping(settings, "decision settings")
    _require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("classes_in_order") == list(CLASSES),
        "decision settings identity changed",
    )
    alphas = _validated_grid(alpha_grid, "alpha grid", minimum=0.0, maximum=1.0)
    edit_offsets = _validated_grid(edit_offset_grid, "edit offset grid")
    check_offsets = _validated_grid(
        check_or_finish_offset_grid, "check-or-finish offset grid"
    )
    alpha = _finite(value.get("alpha"), "selected fusion alpha")
    offsets = _mapping(value.get("class_logit_offsets"), "selected class offsets")
    _require(set(offsets) == set(CLASSES), "selected class offsets changed")
    inspect_offset = _finite(offsets.get("inspect"), "selected inspect offset")
    edit_offset = _finite(offsets.get("edit"), "selected edit offset")
    check_offset = _finite(
        offsets.get("check_or_finish"), "selected check-or-finish offset"
    )
    _require(inspect_offset == 0.0, "inspect offset must remain exactly zero")
    _require(alpha in alphas, "selected fusion alpha is off the frozen grid")
    _require(edit_offset in edit_offsets, "selected edit offset is off the frozen grid")
    _require(
        check_offset in check_offsets,
        "selected check-or-finish offset is off the frozen grid",
    )
    return alpha, edit_offset, check_offset


def apply_decision_settings(
    logit_probabilities: Any,
    logit_j_probabilities: Any,
    settings: Any,
    *,
    alpha_grid: Sequence[float],
    edit_offset_grid: Sequence[float],
    check_or_finish_offset_grid: Sequence[float],
) -> np.ndarray:
    """Apply one grid-authenticated setting without consulting labels."""

    alpha, edit_offset, check_offset = _validated_decision_settings(
        settings,
        alpha_grid=alpha_grid,
        edit_offset_grid=edit_offset_grid,
        check_or_finish_offset_grid=check_or_finish_offset_grid,
    )
    fused = fuse_probabilities(logit_probabilities, logit_j_probabilities, alpha)
    return apply_class_logit_offsets(
        fused,
        edit_offset=edit_offset,
        check_or_finish_offset=check_offset,
    )


def _threshold_metrics(
    labels: np.ndarray,
    predicted_indices: np.ndarray,
    accepted: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    label_indices = _labels_as_indices(labels)
    correct = predicted_indices == label_indices
    coverage = float(weights[accepted].sum(dtype=np.float64))
    accepted_accuracy = (
        float(
            np.sum(weights[accepted] * correct[accepted], dtype=np.float64)
            / coverage
        )
        if coverage > 0.0
        else None
    )
    return {
        "coverage": coverage,
        "accepted_accuracy": accepted_accuracy,
        "accepted_row_count": int(accepted.sum()),
        "accepted_rows_per_true_class": {
            class_id: int(np.sum(accepted & (label_indices == class_index)))
            for class_id, class_index in CLASS_INDEX.items()
        },
    }


def confidence_for_decisions(
    decision_probabilities: Any,
    confidence_probabilities: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return decision argmax indices and their corresponding confidence values.

    A separate confidence matrix may have a different argmax.  Confidence is
    nevertheless read at the class chosen by the decision matrix; otherwise a
    row could be accepted based on confidence assigned to a different action.
    """

    decisions = validate_probability_matrix(
        decision_probabilities, "decision probabilities"
    )
    confidence_matrix = (
        decisions.copy()
        if confidence_probabilities is None
        else validate_probability_matrix(
            confidence_probabilities,
            "confidence probabilities",
            expected_rows=len(decisions),
        )
    )
    predicted_indices = np.argmax(decisions, axis=1)
    confidence = confidence_matrix[np.arange(len(decisions)), predicted_indices]
    _require(
        np.all(np.isfinite(confidence)) and np.all(confidence > 0.0),
        "decision confidence is invalid",
    )
    return predicted_indices, confidence


def _threshold_vector_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    thresholds = candidate["thresholds"]
    values = tuple(float(thresholds[class_id]) for class_id in CLASSES)
    return (sum(values), *values)


def _passing_threshold_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        -float(candidate["coverage"]),
        -float(candidate["accepted_accuracy"]),
        *_threshold_vector_key(candidate),
    )


def _fallback_threshold_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        -float(candidate["accepted_accuracy"]),
        -float(candidate["coverage"]),
        *_threshold_vector_key(candidate),
    )


def select_class_confidence_thresholds(
    decision_probabilities: Any,
    labels: Sequence[str],
    weights: Any,
    *,
    threshold_grid: Sequence[float],
    accepted_accuracy_minimum: float,
    coverage_minimum: float,
    minimum_accepted_rows_per_true_class: int,
    confidence_probabilities: Any | None = None,
    include_candidates: bool = True,
) -> dict[str, Any]:
    """Exhaustively select one confidence threshold per predicted class.

    A fallback remains applicable for complete outer-fold diagnostics, but is
    returned with ``selected_under_floors=False`` so the parent evaluator must
    fail promotion closed.
    """

    decisions = validate_probability_matrix(
        decision_probabilities, "decision probabilities"
    )
    label_values = _validated_labels(labels, len(decisions))
    normalized_weights = validate_weights(weights, len(decisions))
    thresholds = _validated_grid(
        threshold_grid, "confidence threshold grid", minimum=0.0, maximum=1.0
    )
    accuracy_floor = _finite(
        accepted_accuracy_minimum, "accepted-accuracy minimum"
    )
    coverage_floor = _finite(coverage_minimum, "coverage minimum")
    _require(
        0.0 <= accuracy_floor <= 1.0 and 0.0 <= coverage_floor <= 1.0,
        "accepted-accuracy and coverage minima must lie in [0, 1]",
    )
    _require(
        isinstance(minimum_accepted_rows_per_true_class, (int, np.integer))
        and not isinstance(minimum_accepted_rows_per_true_class, (bool, np.bool_))
        and int(minimum_accepted_rows_per_true_class) >= 1,
        "minimum accepted rows per true class must be a positive integer",
    )
    minimum_rows = int(minimum_accepted_rows_per_true_class)
    predicted_indices, confidence = confidence_for_decisions(
        decisions, confidence_probabilities
    )

    candidates: list[dict[str, Any]] = []
    for threshold_values in product(thresholds, repeat=len(CLASSES)):
        threshold_array = np.asarray(threshold_values, dtype=np.float64)
        accepted = confidence >= threshold_array[predicted_indices]
        metrics = _threshold_metrics(
            label_values, predicted_indices, accepted, normalized_weights
        )
        accepted_accuracy = metrics["accepted_accuracy"]
        meets = (
            accepted_accuracy is not None
            and float(accepted_accuracy) >= accuracy_floor
            and float(metrics["coverage"]) >= coverage_floor
            and all(
                count >= minimum_rows
                for count in metrics["accepted_rows_per_true_class"].values()
            )
        )
        candidates.append(
            {
                "thresholds": {
                    class_id: float(threshold_values[class_index])
                    for class_id, class_index in CLASS_INDEX.items()
                },
                **metrics,
                "meets_floors": bool(meets),
            }
        )

    passing = [candidate for candidate in candidates if candidate["meets_floors"]]
    if passing:
        selected = min(passing, key=_passing_threshold_key)
        fallback_used = False
    else:
        coverage_pool = [
            candidate
            for candidate in candidates
            if candidate["accepted_accuracy"] is not None
            and float(candidate["coverage"]) >= coverage_floor
        ]
        pool = coverage_pool or [
            candidate
            for candidate in candidates
            if candidate["accepted_accuracy"] is not None
        ]
        _require(bool(pool), "threshold grid produced no accepted predictions")
        selected = min(pool, key=_fallback_threshold_key)
        fallback_used = True

    return {
        "schema_version": SCHEMA_VERSION,
        "classes_in_order": list(CLASSES),
        "thresholds": dict(selected["thresholds"]),
        "selected_under_floors": not fallback_used,
        "fallback_used": fallback_used,
        "accepted_accuracy_minimum": accuracy_floor,
        "coverage_minimum": coverage_floor,
        "minimum_accepted_rows_per_true_class": minimum_rows,
        "selected_metrics": {
            key: selected[key]
            for key in (
                "coverage",
                "accepted_accuracy",
                "accepted_row_count",
                "accepted_rows_per_true_class",
                "meets_floors",
            )
        },
        "confidence_source": (
            "decision_probabilities"
            if confidence_probabilities is None
            else "separate_confidence_probabilities"
        ),
        "selection_rule": (
            "maximum_weighted_coverage_meeting_accepted_accuracy_coverage_and_"
            "true_class_count_floors_then_accuracy_then_lowest_threshold_vector"
        ),
        "fallback_rule": (
            "maximum_accepted_accuracy_subject_to_coverage_then_coverage_then_"
            "lowest_threshold_vector"
        ),
        "threshold_grid": list(thresholds),
        "candidate_count": len(candidates),
        **({"candidates": candidates} if include_candidates else {}),
    }


def _validated_threshold_settings(
    settings: Any, *, threshold_grid: Sequence[float]
) -> tuple[np.ndarray, str]:
    value = _mapping(settings, "confidence-threshold settings")
    _require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("classes_in_order") == list(CLASSES),
        "confidence-threshold settings identity changed",
    )
    grid = _validated_grid(
        threshold_grid, "confidence threshold grid", minimum=0.0, maximum=1.0
    )
    selected = _mapping(value.get("thresholds"), "selected confidence thresholds")
    _require(set(selected) == set(CLASSES), "selected confidence thresholds changed")
    values = np.asarray(
        [
            _finite(selected[class_id], f"selected {class_id} confidence threshold")
            for class_id in CLASSES
        ],
        dtype=np.float64,
    )
    _require(
        all(float(item) in grid for item in values),
        "selected confidence threshold is off the frozen grid",
    )
    confidence_source = value.get("confidence_source")
    _require(
        confidence_source
        in {
            "decision_probabilities",
            "separate_confidence_probabilities",
        },
        "confidence-threshold settings have an invalid confidence source",
    )
    return values, str(confidence_source)


def apply_class_confidence_thresholds(
    decision_probabilities: Any,
    settings: Any,
    *,
    threshold_grid: Sequence[float],
    confidence_probabilities: Any | None = None,
) -> np.ndarray:
    """Apply predicted-class thresholds without consulting labels or weights."""

    thresholds, expected_confidence_source = _validated_threshold_settings(
        settings, threshold_grid=threshold_grid
    )
    observed_confidence_source = (
        "decision_probabilities"
        if confidence_probabilities is None
        else "separate_confidence_probabilities"
    )
    _require(
        observed_confidence_source == expected_confidence_source,
        "confidence matrix source differs from the selected threshold settings",
    )
    predicted_indices, confidence = confidence_for_decisions(
        decision_probabilities, confidence_probabilities
    )
    return np.asarray(
        confidence >= thresholds[predicted_indices], dtype=bool
    )
