#!/usr/bin/env python3
"""Pure known-row metrics for the V4 two-output task-state procedure.

The evaluator deliberately keeps two probability roles separate:

* forecast probabilities ``q`` own NLL, Brier score, and ordinary top-label
  calibration; and
* decision probabilities ``d`` contribute only their deterministic argmax to
  operational accuracy, recall, balanced accuracy, and selective accuracy.

Decision-confidence calibration is the bridge between the two outputs: its
confidence is ``q[row, argmax(d[row])]`` and its correctness is the correctness
of that decision.  That same binary event owns decision-confidence log loss
and Brier score.  Inputs are copied after strict validation and are never
mutated.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


CLASSES = ("inspect", "edit", "check_or_finish")
CLASS_INDEX = {class_id: index for index, class_id in enumerate(CLASSES)}
SCHEMA_VERSION = 1
PROBABILITY_TOLERANCE = 1e-12


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object_array(value: Any, label: str) -> np.ndarray:
    try:
        return np.array(value, dtype=object, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a rectangular array") from error


def _numeric_nonboolean(item: Any) -> bool:
    return isinstance(item, (int, float, np.integer, np.floating)) and not isinstance(
        item, (bool, np.bool_)
    )


def _probability_matrix(
    value: Any,
    label: str,
    *,
    expected_rows: int | None = None,
) -> np.ndarray:
    raw = _object_array(value, label)
    _require(
        raw.ndim == 2 and raw.shape[0] > 0 and raw.shape[1] == len(CLASSES),
        f"{label} must have shape (nonzero_rows, {len(CLASSES)})",
    )
    if expected_rows is not None:
        _require(
            raw.shape[0] == expected_rows,
            f"{label} must have the same row count as forecast probabilities q",
        )
    _require(
        all(_numeric_nonboolean(item) for item in raw.flat),
        f"{label} must contain numeric, non-boolean values",
    )
    try:
        result = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must contain float64-compatible values") from error
    _require(
        np.all(np.isfinite(result)),
        f"{label} must contain only finite values",
    )
    _require(
        np.all(result > 0.0),
        f"{label} must contain strictly positive values",
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


def _labels(
    value: Any,
    row_count: int,
    *,
    require_all_classes: bool,
) -> np.ndarray:
    labels = _object_array(value, "labels")
    _require(labels.shape == (row_count,), "labels must have one value per row")
    _require(
        all(isinstance(item, str) and item in CLASS_INDEX for item in labels),
        "labels must use only the frozen three class strings",
    )
    _require(
        isinstance(require_all_classes, bool),
        "require_all_classes must be a boolean",
    )
    if require_all_classes:
        _require(
            set(labels.tolist()) == set(CLASSES),
            "known rows must contain positive label support for every class",
        )
    return labels


def _weights(value: Any, row_count: int) -> tuple[np.ndarray, float]:
    raw = _object_array(value, "weights")
    _require(raw.shape == (row_count,), "weights must have one value per row")
    _require(
        all(_numeric_nonboolean(item) for item in raw.flat),
        "weights must contain numeric, non-boolean values",
    )
    try:
        result = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("weights must contain float64-compatible values") from error
    _require(
        np.all(np.isfinite(result)) and np.all(result > 0.0),
        "weights must be finite and strictly positive",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        input_total = float(result.sum(dtype=np.float64))
    _require(
        math.isfinite(input_total) and input_total > 0.0,
        "total caller weight must be finite and positive",
    )
    result /= input_total
    _require(
        np.all(np.isfinite(result)) and np.all(result > 0.0),
        "normalized weights must remain finite and strictly positive",
    )
    _require(
        math.isclose(
            float(result.sum(dtype=np.float64)),
            1.0,
            rel_tol=PROBABILITY_TOLERANCE,
            abs_tol=PROBABILITY_TOLERANCE,
        ),
        "normalized weights must sum to one",
    )
    return result, input_total


def _accepted_mask(value: Any, row_count: int) -> np.ndarray:
    raw = _object_array(value, "accepted mask")
    _require(
        raw.shape == (row_count,),
        "accepted mask must have one value per row",
    )
    _require(
        all(isinstance(item, (bool, np.bool_)) for item in raw.flat),
        "accepted mask must contain only boolean values",
    )
    return np.array(raw, dtype=np.bool_, copy=True)


def _ece_bin_count(value: Any) -> int:
    _require(
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_)),
        "ece_bins must be a positive integer",
    )
    result = int(value)
    _require(result > 0, "ece_bins must be a positive integer")
    return result


def _weighted_ece(
    confidence: np.ndarray,
    correct: np.ndarray,
    weights: np.ndarray,
    bin_count: int,
) -> float:
    """Return equal-width ECE without allocating one object per empty bin."""

    # int(confidence * bin_count) implements lower-inclusive bins.  Clamping
    # places an exactly-one confidence in the inclusive final bin.
    bin_ids = [
        min(int(float(item) * bin_count), bin_count - 1) for item in confidence
    ]
    ece = 0.0
    for bin_id in sorted(set(bin_ids)):
        mask = np.asarray([item == bin_id for item in bin_ids], dtype=np.bool_)
        mass = float(weights[mask].sum(dtype=np.float64))
        if mass == 0.0:
            continue
        bin_accuracy = float(
            np.sum(weights[mask] * correct[mask], dtype=np.float64) / mass
        )
        bin_confidence = float(
            np.sum(weights[mask] * confidence[mask], dtype=np.float64) / mass
        )
        ece += mass * abs(bin_accuracy - bin_confidence)
    return float(ece)


def _metric_sources() -> dict[str, Any]:
    operational = {
        "prediction": "argmax(decision_probabilities_d)",
        "probability_magnitudes_used": False,
        "label_scope": "known_rows",
        "weighting": "normalized_positive_caller_weights",
    }
    return {
        "accuracy": dict(operational),
        "balanced_accuracy": dict(operational),
        "per_class_recall": dict(operational),
        "recall_inspect": dict(operational),
        "recall_edit": dict(operational),
        "recall_check_or_finish": dict(operational),
        "multiclass_negative_log_likelihood": {
            "probabilities": "forecast_probabilities_q",
            "target": "true_label_probability",
            "weighting": "normalized_positive_caller_weights",
        },
        "multiclass_brier": {
            "probabilities": "forecast_probabilities_q",
            "target": "three_class_one_hot_label",
            "convention": "unhalved_sum_over_classes",
            "weighting": "normalized_positive_caller_weights",
        },
        "top_label_ece": {
            "confidence": "max(forecast_probabilities_q)",
            "correctness": "argmax(forecast_probabilities_q) == true_label",
            "binning": "equal_width_lower_inclusive_upper_exclusive_except_final",
            "weighting": "normalized_positive_caller_weights",
        },
        "decision_confidence_ece": {
            "confidence": "forecast_probabilities_q[row, argmax(decision_probabilities_d[row])]",
            "correctness": "argmax(decision_probabilities_d) == true_label",
            "binning": "equal_width_lower_inclusive_upper_exclusive_except_final",
            "weighting": "normalized_positive_caller_weights",
        },
        "decision_confidence_binary_log_loss": {
            "confidence": "forecast_probabilities_q[row, argmax(decision_probabilities_d[row])]",
            "target": "argmax(decision_probabilities_d) == true_label",
            "loss": "-z*log(c) - (1-z)*log(1-c)",
            "weighting": "normalized_positive_caller_weights",
        },
        "decision_confidence_binary_brier": {
            "confidence": "forecast_probabilities_q[row, argmax(decision_probabilities_d[row])]",
            "target": "argmax(decision_probabilities_d) == true_label",
            "loss": "(c-z)^2",
            "weighting": "normalized_positive_caller_weights",
        },
        "selected_coverage": {
            "selection": "accepted_boolean_mask",
            "weighting": "normalized_positive_caller_weights",
        },
        "selected_accepted_accuracy": {
            "selection": "accepted_boolean_mask",
            "correctness": "argmax(decision_probabilities_d) == true_label",
            "weighting": "normalized_positive_caller_weights_conditioned_on_acceptance",
        },
        "per_true_class_accepted_coverage": {
            "selection": "accepted_boolean_mask",
            "conditioning": "true_label_class",
            "weighting": "normalized_positive_caller_weights_within_true_class",
        },
    }


def known_row_metrics(
    forecast_probabilities_q: Any,
    decision_probabilities_d: Any,
    labels: Any,
    weights: Any,
    accepted: Any,
    *,
    ece_bins: int,
    require_all_classes: bool = True,
) -> dict[str, Any]:
    """Compute weighted V4 metrics under explicit ``q``/``d`` roles.

    Complete development evidence uses the strict default and must contain all
    three classes.  Per-repository diagnostics may set
    ``require_all_classes=False``; only unsupported recalls, balanced accuracy,
    and true-class acceptance coverage then become null.  Metrics whose
    denominators remain defined are still emitted.
    """

    q = _probability_matrix(
        forecast_probabilities_q,
        "forecast probabilities q",
    )
    d = _probability_matrix(
        decision_probabilities_d,
        "decision probabilities d",
        expected_rows=len(q),
    )
    known_labels = _labels(
        labels,
        len(q),
        require_all_classes=require_all_classes,
    )
    metric_weights, input_weight_sum = _weights(weights, len(q))
    accepted_mask = _accepted_mask(accepted, len(q))
    bin_count = _ece_bin_count(ece_bins)

    y = np.asarray(
        [CLASS_INDEX[str(label)] for label in known_labels], dtype=np.int64
    )
    row_indices = np.arange(len(q), dtype=np.int64)

    forecast_predictions = np.argmax(q, axis=1)
    forecast_correct = forecast_predictions == y
    forecast_confidence = q[row_indices, forecast_predictions]

    decision_predictions = np.argmax(d, axis=1)
    decision_correct = decision_predictions == y
    decision_confidence = q[row_indices, decision_predictions]
    decision_target = decision_correct.astype(np.float64)
    decision_outcome_probability = np.where(
        decision_correct,
        decision_confidence,
        1.0 - decision_confidence,
    )
    _require(
        np.all(np.isfinite(decision_outcome_probability))
        and np.all(decision_outcome_probability > 0.0),
        "decision-confidence binary outcome probabilities must be finite and positive",
    )

    per_class_recall: dict[str, float | None] = {}
    per_true_class_accepted_coverage: dict[str, float | None] = {}
    for class_index, class_id in enumerate(CLASSES):
        true_class = y == class_index
        class_mass = float(metric_weights[true_class].sum(dtype=np.float64))
        if class_mass > 0.0:
            per_class_recall[class_id] = float(
                np.sum(
                    metric_weights[true_class] * decision_correct[true_class],
                    dtype=np.float64,
                )
                / class_mass
            )
            per_true_class_accepted_coverage[class_id] = float(
                metric_weights[true_class & accepted_mask].sum(dtype=np.float64)
                / class_mass
            )
        else:
            per_class_recall[class_id] = None
            per_true_class_accepted_coverage[class_id] = None

    balanced_accuracy = (
        float(
            math.fsum(
                float(per_class_recall[class_id]) for class_id in CLASSES
            )
            / len(CLASSES)
        )
        if all(per_class_recall[class_id] is not None for class_id in CLASSES)
        else None
    )

    true_forecast_probability = q[row_indices, y]
    one_hot = np.eye(len(CLASSES), dtype=np.float64)[y]
    selected_coverage = float(
        metric_weights[accepted_mask].sum(dtype=np.float64)
    )
    selected_accepted_accuracy = (
        float(
            np.sum(
                metric_weights[accepted_mask] * decision_correct[accepted_mask],
                dtype=np.float64,
            )
            / selected_coverage
        )
        if selected_coverage > 0.0
        else None
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "classes_in_order": list(CLASSES),
        "row_count": len(q),
        "accuracy": float(
            np.sum(metric_weights * decision_correct, dtype=np.float64)
        ),
        "balanced_accuracy": balanced_accuracy,
        "per_class_recall": per_class_recall,
        "recall_inspect": per_class_recall["inspect"],
        "recall_edit": per_class_recall["edit"],
        "recall_check_or_finish": per_class_recall["check_or_finish"],
        "multiclass_negative_log_likelihood": float(
            -np.sum(
                metric_weights * np.log(true_forecast_probability),
                dtype=np.float64,
            )
        ),
        "multiclass_brier": float(
            np.sum(
                metric_weights * np.sum((q - one_hot) ** 2, axis=1),
                dtype=np.float64,
            )
        ),
        "top_label_ece": _weighted_ece(
            forecast_confidence,
            forecast_correct,
            metric_weights,
            bin_count,
        ),
        "decision_confidence_ece": _weighted_ece(
            decision_confidence,
            decision_correct,
            metric_weights,
            bin_count,
        ),
        "decision_confidence_binary_log_loss": float(
            -np.sum(
                metric_weights * np.log(decision_outcome_probability),
                dtype=np.float64,
            )
        ),
        "decision_confidence_binary_brier": float(
            np.sum(
                metric_weights * (decision_confidence - decision_target) ** 2,
                dtype=np.float64,
            )
        ),
        "selected_coverage": selected_coverage,
        "selected_accepted_accuracy": selected_accepted_accuracy,
        "per_true_class_accepted_coverage": per_true_class_accepted_coverage,
        "ece_bin_count": bin_count,
        "ece_binning": "equal_width_lower_inclusive_upper_exclusive_except_final",
        "weighting": {
            "source": "positive_caller_weights",
            "normalization": "divide_by_total_before_all_metrics",
            "input_weight_sum": input_weight_sum,
            "normalized_weight_sum": float(
                metric_weights.sum(dtype=np.float64)
            ),
        },
        "metric_sources": _metric_sources(),
    }
