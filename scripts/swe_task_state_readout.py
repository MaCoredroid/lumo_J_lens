#!/usr/bin/env python3
"""Focused multinomial readout solver for SWE task-state experiments."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


CLASS_IDS = ("inspect", "edit", "validate", "finalize")
GRADIENT_TOLERANCE = 1e-5
LBFGS_HISTORY_SIZE = 10
LINE_SEARCH_MAXIMUM_STEPS = 60
ARMIJO_CONSTANT = 1e-4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _normalized_class_ids(class_ids: Sequence[str]) -> tuple[str, ...]:
    result = tuple(class_ids)
    require(
        len(result) >= 2
        and all(isinstance(class_id, str) and class_id for class_id in result)
        and len(result) == len(set(result)),
        "multinomial class IDs are invalid",
    )
    return result


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(x.ndim == 2 and x.shape[0] > 0, "cannot scale an empty feature matrix")
    mean = np.mean(x, axis=0, dtype=np.float64)
    scale = np.std(x, axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)
    transformed = (x - mean) / scale
    require(np.all(np.isfinite(transformed)), "feature scaling produced nonfinite values")
    return transformed, mean, scale


def _weighted_multinomial_objective(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    *,
    class_count: int,
    c_value: float,
) -> tuple[float, np.ndarray]:
    feature_count = x.shape[1]
    weights = theta[: class_count * feature_count].reshape(class_count, feature_count)
    intercept = theta[class_count * feature_count :]
    logits = x @ weights.T + intercept
    maximum = np.max(logits, axis=1, keepdims=True)
    shifted = logits - maximum
    exponentials = np.exp(shifted)
    normalizers = np.sum(exponentials, axis=1, keepdims=True)
    probabilities = exponentials / normalizers
    log_normalizers = maximum[:, 0] + np.log(normalizers[:, 0])
    losses = log_normalizers - logits[np.arange(len(y)), y]
    objective = float(np.dot(sample_weights, losses)) + float(
        0.5 * np.sum(weights * weights) / c_value
    )
    error = probabilities
    error[np.arange(len(y)), y] -= 1.0
    error *= sample_weights[:, None]
    gradient_weights = error.T @ x + weights / c_value
    gradient_intercept = np.sum(error, axis=0)
    gradient = np.concatenate([gradient_weights.ravel(), gradient_intercept])
    require(
        math.isfinite(objective) and np.all(np.isfinite(gradient)),
        "multinomial objective produced nonfinite values",
    )
    return objective, gradient


def _update_lbfgs_history(
    history: list[tuple[np.ndarray, np.ndarray, float]],
    step_vector: np.ndarray,
    gradient_delta: np.ndarray,
) -> None:
    curvature = float(np.dot(step_vector, gradient_delta))
    curvature_floor = 1e-12 * max(
        1.0,
        float(np.linalg.norm(step_vector)) * float(np.linalg.norm(gradient_delta)),
    )
    if curvature > curvature_floor:
        history.append((step_vector, gradient_delta, 1.0 / curvature))
        if len(history) > LBFGS_HISTORY_SIZE:
            history.pop(0)
    else:
        # A rejected pair invalidates the old local inverse-Hessian approximation.
        history.clear()


def fit_multinomial_lbfgs(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c_value: float,
    maximum_iterations: int,
    class_ids: Sequence[str] = CLASS_IDS,
) -> dict[str, Any]:
    """Fit a class-balanced L2 multinomial readout at any positive feature width."""
    normalized_class_ids = _normalized_class_ids(class_ids)
    class_count = len(normalized_class_ids)
    require(
        x.ndim == 2
        and y.ndim == 1
        and len(x) == len(y)
        and x.shape[1] > 0
        and len(x) > 0
        and np.all(np.isfinite(x)),
        "multinomial fit matrix is invalid",
    )
    require(
        np.issubdtype(y.dtype, np.integer)
        and np.all(y >= 0)
        and np.all(y < class_count),
        "multinomial labels are invalid",
    )
    require(c_value > 0.0 and maximum_iterations > 0, "multinomial fit contract is invalid")
    counts = np.bincount(y, minlength=class_count).astype(np.float64)
    require(np.all(counts > 0.0), "multinomial training split lacks a declared class")
    sample_weights = len(y) / (class_count * counts[y])
    scaled, mean, scale = _standardize_fit(x)
    parameter_count = class_count * x.shape[1] + class_count
    weight_parameter_count = class_count * x.shape[1]
    curvature_reference = float(len(y)) / class_count
    weight_coordinate_scale = 1.0 / math.sqrt(1.0 / c_value + curvature_reference)
    intercept_coordinate_scale = 1.0 / math.sqrt(curvature_reference)

    def evaluate(coordinate: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        original = coordinate.copy()
        original[:weight_parameter_count] *= weight_coordinate_scale
        original[weight_parameter_count:] *= intercept_coordinate_scale
        value, original_gradient = _weighted_multinomial_objective(
            original,
            scaled,
            y,
            sample_weights,
            class_count=class_count,
            c_value=c_value,
        )
        coordinate_gradient = original_gradient.copy()
        coordinate_gradient[:weight_parameter_count] *= weight_coordinate_scale
        coordinate_gradient[weight_parameter_count:] *= intercept_coordinate_scale
        return value, coordinate_gradient, original_gradient

    theta = np.zeros(parameter_count, dtype=np.float64)
    objective, gradient, original_gradient = evaluate(theta)
    history: list[tuple[np.ndarray, np.ndarray, float]] = []
    converged = float(np.max(np.abs(original_gradient))) <= GRADIENT_TOLERANCE
    iterations = 0
    line_search_evaluations = 0
    failure_reason: str | None = None
    for iteration in range(1, maximum_iterations + 1):
        if converged:
            break
        iterations = iteration
        direction_work = gradient.copy()
        alphas: list[float] = []
        for step_vector, gradient_delta, inverse_curvature in reversed(history):
            alpha = inverse_curvature * float(np.dot(step_vector, direction_work))
            alphas.append(alpha)
            direction_work -= alpha * gradient_delta
        if history:
            last_step, last_delta, _ = history[-1]
            denominator = float(np.dot(last_delta, last_delta))
            gamma = (
                float(np.dot(last_step, last_delta)) / denominator
                if denominator > 0.0
                else 1.0
            )
            gamma = max(1e-12, min(1e12, gamma))
        else:
            gamma = 1.0
        direction_work *= gamma
        for (step_vector, gradient_delta, inverse_curvature), alpha in zip(
            history, reversed(alphas), strict=True
        ):
            beta = inverse_curvature * float(np.dot(gradient_delta, direction_work))
            direction_work += step_vector * (alpha - beta)
        direction = -direction_work
        directional_derivative = float(np.dot(gradient, direction))
        if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
            history.clear()
            direction = -gradient
            directional_derivative = -float(np.dot(gradient, gradient))

        step_size = min(1.0, 1.0 / max(1.0, float(np.max(np.abs(direction)))))
        accepted = False
        candidate_theta = theta
        candidate_objective = objective
        candidate_gradient = gradient
        candidate_original_gradient = original_gradient
        for _ in range(LINE_SEARCH_MAXIMUM_STEPS):
            candidate_theta = theta + step_size * direction
            candidate_objective, candidate_gradient, candidate_original_gradient = evaluate(
                candidate_theta
            )
            line_search_evaluations += 1
            if candidate_objective <= (
                objective + ARMIJO_CONSTANT * step_size * directional_derivative
            ):
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            failure_reason = "armijo_line_search_failed"
            break
        step_vector = candidate_theta - theta
        gradient_delta = candidate_gradient - gradient
        _update_lbfgs_history(history, step_vector, gradient_delta)
        theta = candidate_theta
        objective = candidate_objective
        gradient = candidate_gradient
        original_gradient = candidate_original_gradient
        converged = float(np.max(np.abs(original_gradient))) <= GRADIENT_TOLERANCE
    if not converged and failure_reason is None:
        failure_reason = "maximum_iterations_reached"
    feature_count = x.shape[1]
    weights = (
        theta[: class_count * feature_count].reshape(class_count, feature_count)
        * weight_coordinate_scale
    )
    intercept = theta[class_count * feature_count :] * intercept_coordinate_scale
    model_payload = {
        "mean": [float(item) for item in mean],
        "scale": [float(item) for item in scale],
        "weights": [[float(item) for item in row] for row in weights],
        "intercept": [float(item) for item in intercept],
    }
    return {
        "converged": converged,
        "failure_reason": failure_reason,
        "iterations": iterations,
        "line_search_evaluations": line_search_evaluations,
        "objective": objective,
        "gradient_infinity_norm": float(np.max(np.abs(original_gradient))),
        "solver_coordinate_gradient_infinity_norm": float(np.max(np.abs(gradient))),
        "solver_weight_coordinate_scale": weight_coordinate_scale,
        "solver_intercept_coordinate_scale": intercept_coordinate_scale,
        "c_value": c_value,
        "class_support": {
            class_id: int(counts[index])
            for index, class_id in enumerate(normalized_class_ids)
        },
        "class_weights": {
            class_id: float(len(y) / (class_count * counts[index]))
            for index, class_id in enumerate(normalized_class_ids)
        },
        "parameter_sha256": sha256_json(model_payload),
        "parameter_l2_norm": float(np.linalg.norm(weights)),
        "scaler_zero_variance_feature_count": int(np.sum(np.std(x, axis=0) == 0.0)),
        "_mean": mean,
        "_scale": scale,
        "_weights": weights,
        "_intercept": intercept,
    }


def predict_multinomial(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    require(x.ndim == 2 and x.shape[1] == len(model["_mean"]), "prediction matrix shape changed")
    scaled = (x - model["_mean"]) / model["_scale"]
    logits = scaled @ model["_weights"].T + model["_intercept"]
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    require(np.all(np.isfinite(probabilities)), "multinomial prediction is nonfinite")
    return probabilities


def public_model_record(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: model[key]
        for key in (
            "converged",
            "failure_reason",
            "iterations",
            "line_search_evaluations",
            "objective",
            "gradient_infinity_norm",
            "solver_coordinate_gradient_infinity_norm",
            "solver_weight_coordinate_scale",
            "solver_intercept_coordinate_scale",
            "c_value",
            "class_support",
            "class_weights",
            "parameter_sha256",
            "parameter_l2_norm",
            "scaler_zero_variance_feature_count",
        )
    }
