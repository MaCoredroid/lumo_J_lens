#!/usr/bin/env python3
"""Leakage-safe nested evaluator for the V4 SWE task-state procedure.

This module owns model fitting, repository cross-fitting, fixed forecast
construction, shared action-rule selection, and point metrics.  Extraction,
CLI/provenance checks, gates, and the full-refit bootstrap live outside this
core.  The implementation imports the frozen V3 analyzer only after
authenticating its exact bytes and reuses only its generic weighting and
ExtraTrees primitives.
"""

from __future__ import annotations

import hashlib
import math
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
    from scripts import analyze_swe_task_state_v3 as V3
    from scripts import swe_task_state_v4_calibration as CALIBRATION
    from scripts import swe_task_state_v4_decision as DECISION
    from scripts import swe_task_state_v4_features as FEATURES
    from scripts import swe_task_state_v4_metrics as METRICS
except ModuleNotFoundError as error:  # Support direct execution from scripts/.
    if error.name != "scripts":
        raise
    import analyze_swe_task_state_v3 as V3  # type: ignore[no-redef]
    import swe_task_state_v4_calibration as CALIBRATION  # type: ignore[no-redef]
    import swe_task_state_v4_decision as DECISION  # type: ignore[no-redef]
    import swe_task_state_v4_features as FEATURES  # type: ignore[no-redef]
    import swe_task_state_v4_metrics as METRICS  # type: ignore[no-redef]


if Path(V3.__file__).resolve() != V3_ANALYZER_PATH:
    raise RuntimeError("frozen V3 analyzer resolved from an unexpected path")


CLASSES = ("inspect", "edit", "check_or_finish")
BASE_VARIANTS = (
    "history_only",
    "sequence_j",
    "sequence_logit",
    "sequence_logit_j",
)
PRIMARY_PROCEDURE = "j_forecast_geometric_pool_logit_policy"
REFERENCE_PROCEDURE = "sequence_logit"
PROCEDURES = (*BASE_VARIANTS, PRIMARY_PROCEDURE)
SHARED_ACTION_PROCEDURES = (REFERENCE_PROCEDURE, PRIMARY_PROCEDURE)
FORECAST_POOL_J_WEIGHT = 0.20
ZERO_ALPHA_GRID = (0.0,)
SCHEMA_VERSION = 1

# V4 chose the frozen V3 generic readout family.  These are protocol identity,
# not caller-tunable hyperparameters: four matched variants times five seeds is
# the exact ordered 20-estimator schedule.
MODEL_SEEDS_IN_ORDER = (271828, 314159, 161803, 141421, 173205)
MODEL_PARAMETERS = {
    "bootstrap": False,
    "ccp_alpha": 0.0,
    "class_weight": None,
    "criterion": "gini",
    "max_depth": None,
    "max_features": 0.5,
    "max_leaf_nodes": None,
    "max_samples": None,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 5,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "monotonic_cst": None,
    "n_estimators": 100,
    "n_jobs": 1,
    "oob_score": False,
    "verbose": 0,
    "warm_start": False,
}
MODEL_FIT_EXECUTION = {
    "parallel_unit": "one_variant_seed_estimator",
    "backend": "sklearn_joblib_loky_processes",
    "worker_count": 20,
    "estimator_fit_n_jobs": 1,
    "persisted_estimator_n_jobs": 1,
    "submission_order": "variant_then_seed",
    "result_collection_order": "variant_then_seed",
    "deterministic_ordered_collection": True,
}
MODEL_PREDICTION_EXECUTION = {
    "estimator_n_jobs": 1,
    "tree_probability_reduction_order": "serial_estimator_order",
    "repeated_prediction_must_be_bitwise_identical": True,
    "parallel_prediction_forbidden": True,
    "reason": (
        "floating_point_tree_probability_accumulation_order_must_not_depend_"
        "on_worker_lock_scheduling"
    ),
}
POINT_ESTIMAND = (
    "equal_repository_then_equal_known_task_within_repository_then_"
    "equal_known_row_within_task"
)


if not (
    CLASSES
    == tuple(V3.CLASSES)
    == tuple(DECISION.CLASSES)
    == tuple(CALIBRATION.CLASSES)
    == tuple(METRICS.CLASSES)
):
    raise RuntimeError("V4 class identity differs across authenticated modules")
if BASE_VARIANTS != tuple(FEATURES.VARIANTS):
    raise RuntimeError("V4 base-variant identity differs from the feature module")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    _require(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray, Mapping)),
        f"{label} must be a sequence",
    )
    return list(value)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_)),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_)),
        f"{label} must be an integer",
    )
    result = int(value)
    _require(result >= minimum, f"{label} is below {minimum}")
    return result


def _float_grid(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[float]:
    values = [_finite(item, f"{label} value") for item in _sequence(value, label)]
    _require(bool(values), f"{label} must not be empty")
    _require(len(values) == len(set(values)), f"{label} values must be unique")
    values = sorted(values)
    if minimum is not None:
        _require(all(item >= minimum for item in values), f"{label} is too small")
    if maximum is not None:
        _require(all(item <= maximum for item in values), f"{label} is too large")
    return values


def validate_contract(value: Any) -> dict[str, Any]:
    """Validate the evaluator-relevant, pre-normalized V4 protocol sections."""

    protocol = _mapping(value, "V4 evaluator contract")
    model = _mapping(protocol.get("model"), "model contract")
    weighting = _mapping(protocol.get("weighting"), "weighting contract")
    nested = _mapping(protocol.get("nested"), "nested contract")
    forecast_pool = _mapping(
        protocol.get("forecast_pool"), "forecast-pool contract"
    )
    decision = _mapping(protocol.get("decision"), "decision contract")
    calibration = _mapping(protocol.get("calibration"), "calibration contract")
    abstention = _mapping(protocol.get("abstention"), "abstention contract")
    metrics = _mapping(protocol.get("metrics"), "metrics contract")

    seeds = [
        _integer(item, "model seed", minimum=0)
        for item in _sequence(model.get("seeds"), "model seeds")
    ]
    _require(bool(seeds) and len(seeds) == len(set(seeds)), "model seeds changed")
    probability_floor = _finite(model.get("probability_floor"), "probability floor")
    _require(
        0.0 < probability_floor < 1.0 / len(CLASSES),
        "probability floor is invalid",
    )
    parameters = dict(_mapping(model.get("parameters"), "model parameters"))
    fit_execution = dict(
        _mapping(model.get("fit_execution"), "model fit execution")
    )
    prediction_execution = dict(
        _mapping(model.get("prediction_execution"), "model prediction execution")
    )
    _require(
        model.get("family")
        == "fixed_multiseed_ExtraTreesClassifier_probability_ensemble"
        and model.get("probability_reduction")
        == (
            "serial_tree_order_within_each_seed_then_arithmetic_mean_across_"
            "seed_estimators_then_floor_and_renormalize"
        )
        and model.get("same_hyperparameters_and_seed_order_for_all_variants")
        is True,
        "generic V3 model-family semantics changed",
    )
    _require(
        seeds == list(MODEL_SEEDS_IN_ORDER)
        and probability_floor == 1e-6
        and parameters == MODEL_PARAMETERS
        and fit_execution == MODEL_FIT_EXECUTION
        and prediction_execution == MODEL_PREDICTION_EXECUTION,
        "V4 model parameters, seed order, or worker count differ from the frozen ordered 20-estimator schedule",
    )

    _require(
        weighting.get("point_estimand") == POINT_ESTIMAND
        and weighting.get("training")
        == (
            "restrict_and_renormalize_point_or_bayesian_base_weights_to_the_"
            "current_training_split_then_apply_split_local_exact_three_class_"
            "rebalance"
        )
        and weighting.get("calibration_threshold_and_evaluation")
        == (
            "restrict_and_renormalize_point_or_bayesian_base_weights_without_"
            "class_rebalance"
        )
        and weighting.get(
            "point_weights_are_computed_once_on_all_known_development_rows_then_restricted_per_split"
        )
        is True
        and weighting.get(
            "bayesian_draw_weights_are_computed_once_on_all_known_development_rows_then_restricted_per_split"
        )
        is True
        and weighting.get("same_row_weights_across_all_matched_variants") is True
        and weighting.get("unknown_current_actions_excluded_from_fit_and_metric_weights")
        is True
        and weighting.get("prevalidated_training_weight_fit_transport")
        == (
            "validate_positive_finite_float64_near_unit_mass_and_near_one_third_"
            "class_mass_then_preserve_every_float64_bit_without_second_"
            "normalization"
        )
        and _finite(
            weighting.get("prevalidated_training_weight_unit_mass_absolute_tolerance"),
            "training-weight unit-mass tolerance",
        )
        == 1e-12
        and weighting.get("prevalidated_crossfit_base_weight_transport")
        == (
            "validate_positive_finite_float64_and_near_unit_mass_then_preserve_"
            "every_float64_bit_before_inner_split_restriction"
        )
        and _finite(
            weighting.get("prevalidated_crossfit_base_weight_unit_mass_absolute_tolerance"),
            "crossfit base-weight unit-mass tolerance",
        )
        == 1e-12,
        "V4 hierarchical base-weight transport semantics changed",
    )
    _require(
        nested.get("outer_algorithm") == "leave_one_repository_out"
        and nested.get("inner_algorithm")
        == "leave_one_repository_out_within_outer_training"
        and nested.get(
            "outer_heldout_labels_used_for_fit_calibration_or_threshold_selection"
        )
        is False
        and nested.get(
            "same_outer_and_inner_folds_weights_seed_order_and_hyperparameters_across_variants"
        )
        is True,
        "V4 nested LORO semantics changed",
    )

    recall_floors_value = _mapping(decision.get("recall_floors"), "recall floors")
    _require(
        set(recall_floors_value) == set(CLASSES),
        "recall floors must name every class exactly",
    )
    recall_floors = {
        class_id: _finite(recall_floors_value[class_id], f"{class_id} recall floor")
        for class_id in CLASSES
    }
    _require(
        all(0.0 <= item <= 1.0 for item in recall_floors.values()),
        "recall floors must lie in [0, 1]",
    )
    balanced_accuracy_minimum = _finite(
        decision.get("balanced_accuracy_minimum"),
        "balanced-accuracy minimum",
    )
    _require(
        0.0 <= balanced_accuracy_minimum <= 1.0,
        "balanced-accuracy minimum must lie in [0, 1]",
    )

    temperatures = list(
        CALIBRATION.canonicalize_temperature_grid(
            _sequence(calibration.get("temperature_grid"), "temperature grid")
        )
    )
    threshold_grid = _float_grid(
        abstention.get("threshold_grid"),
        "threshold grid",
        minimum=0.0,
        maximum=1.0,
    )
    accepted_accuracy_minimum = _finite(
        abstention.get("accepted_accuracy_minimum"),
        "accepted-accuracy minimum",
    )
    coverage_minimum = _finite(
        abstention.get("coverage_minimum"), "coverage minimum"
    )
    _require(
        0.0 <= accepted_accuracy_minimum <= 1.0
        and 0.0 <= coverage_minimum <= 1.0,
        "abstention minima must lie in [0, 1]",
    )

    normalized = {
        "model": {
            "seeds": seeds,
            "probability_floor": probability_floor,
            "parameters": parameters,
            "fit_execution": fit_execution,
            "prediction_execution": prediction_execution,
            "family": model["family"],
            "probability_reduction": model["probability_reduction"],
            "same_hyperparameters_and_seed_order_for_all_variants": True,
        },
        "weighting": dict(weighting),
        "nested": {
            "outer_algorithm": "leave_one_repository_out",
            "inner_algorithm": "leave_one_repository_out_within_outer_training",
            "outer_heldout_labels_used_for_fit_calibration_or_threshold_selection": False,
            "minimum_inner_repositories": _integer(
                nested.get("minimum_inner_repositories"),
                "minimum inner repositories",
                minimum=2,
            ),
            "same_outer_and_inner_folds_weights_seed_order_and_hyperparameters_across_variants": True,
        },
        "forecast_pool": {
            "candidate": forecast_pool.get("candidate"),
            "reference": forecast_pool.get("reference"),
            "candidate_logit_weight": _finite(
                forecast_pool.get("candidate_logit_weight"),
                "candidate logit pool weight",
            ),
            "candidate_logit_j_weight": _finite(
                forecast_pool.get("candidate_logit_j_weight"),
                "candidate logit-plus-J pool weight",
            ),
            "selection": forecast_pool.get("selection"),
            "normalization": forecast_pool.get("normalization"),
            "shared_action_source": forecast_pool.get("shared_action_source"),
        },
        "decision": {
            "edit_offset_grid": _float_grid(
                decision.get("edit_offset_grid"), "edit-offset grid"
            ),
            "check_or_finish_offset_grid": _float_grid(
                decision.get("check_or_finish_offset_grid"),
                "check-or-finish-offset grid",
            ),
            "recall_floors": recall_floors,
            "balanced_accuracy_minimum": balanced_accuracy_minimum,
        },
        "calibration": {"temperature_grid": temperatures},
        "abstention": {
            "threshold_grid": threshold_grid,
            "accepted_accuracy_minimum": accepted_accuracy_minimum,
            "coverage_minimum": coverage_minimum,
            "minimum_accepted_rows_per_true_class": _integer(
                abstention.get("minimum_accepted_rows_per_true_class"),
                "minimum accepted rows per true class",
                minimum=1,
            ),
        },
        "metrics": {
            "ece_bins": _integer(metrics.get("ece_bins"), "ECE bins", minimum=1)
        },
    }
    _require(
        normalized["forecast_pool"]
        == {
            "candidate": "fixed_geometric_log_opinion_pool",
            "reference": "sequence_logit_raw_probability",
            "candidate_logit_weight": 0.80,
            "candidate_logit_j_weight": FORECAST_POOL_J_WEIGHT,
            "selection": "none_fixed_before_fresh_development_selection",
            "normalization": "log_space_then_exact_row_normalization",
            "shared_action_source": REFERENCE_PROCEDURE,
        },
        "V4 fixed geometric forecast-pool semantics changed",
    )
    _require(
        0.0 in normalized["abstention"]["threshold_grid"],
        "threshold grid must contain zero so B2 fallback always emits predictions",
    )
    _require(
        len(seeds) * len(BASE_VARIANTS) == 20,
        "ordered base-variant/seed schedule must contain exactly 20 estimators",
    )
    _require(
        recall_floors
        == {"inspect": 0.75, "edit": 0.65, "check_or_finish": 0.75}
        and balanced_accuracy_minimum == 0.75,
        "B2 action-rule floors changed",
    )
    _require(
        accepted_accuracy_minimum == 0.86
        and coverage_minimum == 0.70
        and normalized["abstention"]["minimum_accepted_rows_per_true_class"]
        == 2
        and normalized["nested"]["minimum_inner_repositories"] == 5
        and normalized["metrics"]["ece_bins"] == 10,
        "B2 nested, abstention, or metric constants changed",
    )
    return normalized


def known_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("metric_evaluable") is True]


def _validate_row_roles(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject denominator-changing truthiness and inconsistent unknown labels."""

    for index, row in enumerate(rows):
        evaluable = row.get("metric_evaluable")
        _require(
            type(evaluable) is bool,
            f"row {index} metric_evaluable must be boolean",
        )
        label = row.get("label")
        if bool(evaluable):
            _require(
                row.get("source_action_label_status") == "available"
                and row.get("source_action_class_id") in V3.COLLAPSE
                and V3.COLLAPSE[row["source_action_class_id"]] == label
                and row.get("label_status") == "available"
                and isinstance(label, str)
                and label in CLASSES,
                f"known row {index} has an invalid target label",
            )
        else:
            _require(
                row.get("source_action_class_id") is None
                and row.get("label_status") == "unknown_current_action"
                and label is None,
                f"unknown row {index} has inconsistent label state or a metric target",
            )


def labels_for(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    labels = np.asarray([row.get("label") for row in rows], dtype=object)
    _require(
        labels.shape == (len(rows),)
        and all(isinstance(item, str) and item in CLASSES for item in labels),
        "known rows contain an invalid target label",
    )
    return labels


def matrix_for(rows: Sequence[Mapping[str, Any]], variant: str) -> np.ndarray:
    _require(variant in BASE_VARIANTS, f"unknown V4 base variant: {variant}")
    try:
        values = np.asarray(
            [
                _mapping(row.get("features"), "row features")[variant]
                for row in rows
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{variant} feature matrix must be numeric") from error
    _require(
        values.shape == (len(rows), FEATURES.VARIANT_WIDTHS[variant])
        and np.all(np.isfinite(values)),
        f"{variant} feature matrix is invalid",
    )
    return values


def _row_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values = [row.get("row_id") for row in rows]
    _require(
        all(isinstance(item, str) and bool(item) for item in values)
        and len(values) == len(set(values)),
        "row IDs must be nonempty and unique",
    )
    return [str(item) for item in values]


def ordered_row_identity_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Bind ordered metadata, labels, and every V4 feature float64 bit."""

    records: list[dict[str, Any]] = []
    for row in rows:
        features = _mapping(row.get("features"), "row features")
        feature_hashes: dict[str, str] = {}
        for variant in BASE_VARIANTS:
            try:
                values = np.asarray(features.get(variant), dtype="<f8")
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"{variant} row identity features are invalid"
                ) from error
            _require(
                values.shape == (FEATURES.VARIANT_WIDTHS[variant],)
                and np.all(np.isfinite(values)),
                f"{variant} row identity features are invalid",
            )
            feature_hashes[variant] = V3.sha256_bytes(values.tobytes(order="C"))
        records.append(
            {
                "row_id": str(row["row_id"]),
                "task_id": str(row["task_id"]),
                "repo": str(row["repo"]),
                "cohort_id": row.get("cohort_id"),
                "task_request_index": row.get("task_request_index"),
                "checkpoint_ordinal": row.get("checkpoint_ordinal"),
                "source_action_label_status": row.get(
                    "source_action_label_status"
                ),
                "source_action_class_id": row.get("source_action_class_id"),
                "label_status": row.get("label_status"),
                "label": row.get("label"),
                "metric_evaluable": row.get("metric_evaluable"),
                "feature_float64_sha256": feature_hashes,
            }
        )
    return V3.canonical_json_sha256(records)


def _forecast_sources(procedure: str) -> tuple[str, ...]:
    _require(procedure in PROCEDURES, f"unknown V4 procedure: {procedure}")
    if procedure == PRIMARY_PROCEDURE:
        return (REFERENCE_PROCEDURE, "sequence_logit_j")
    return (procedure,)


def _decision_source(procedure: str) -> str:
    _require(procedure in PROCEDURES, f"unknown V4 procedure: {procedure}")
    return REFERENCE_PROCEDURE if procedure == PRIMARY_PROCEDURE else procedure


def _base_probability(
    raw_probabilities: Mapping[str, Any], variant: str, label: str
) -> np.ndarray:
    _require(variant in raw_probabilities, f"{label} lacks {variant}")
    return DECISION.validate_probability_matrix(
        raw_probabilities[variant], f"{label} {variant} raw probabilities"
    )


def _forecast_raw_probability(
    raw_probabilities: Mapping[str, Any], procedure: str
) -> np.ndarray:
    sources = _forecast_sources(procedure)
    first = _base_probability(raw_probabilities, sources[0], procedure)
    if len(sources) == 1:
        return first
    second = _base_probability(raw_probabilities, sources[1], procedure)
    _require(len(second) == len(first), "forecast-pool row count changed")
    return CALIBRATION.geometric_pool_probabilities(
        first,
        second,
        alpha=FORECAST_POOL_J_WEIGHT,
    )


def _fit_all_base_variants(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    base_weights: Sequence[float] | np.ndarray | None = None,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Fit the declared variant/seed schedule in deterministic V3 order."""

    rows = list(rows)
    _require(bool(rows) and len(known_rows(rows)) == len(rows), "fit requires known rows")
    ExtraTreesClassifier, Parallel, delayed = V3._ml_dependencies()
    weights, shared_weight_diagnostics = V3.training_weights(
        rows, base_weights
    )
    matrices: dict[str, np.ndarray] = {}
    contexts: dict[str, tuple[Any, ...]] = {}
    for variant in BASE_VARIANTS:
        matrices[variant] = matrix_for(rows, variant)
        contexts[variant] = V3._prepare_ensemble_fit(
            matrices[variant], rows, protocol=contract, weights=weights
        )
        _require(
            contexts[variant][2]["weight_float64_sha256"]
            == shared_weight_diagnostics["weight_float64_sha256"],
            "matched base variants did not use identical training weights",
        )

    execution = _mapping(
        _mapping(contract.get("model"), "model contract").get("fit_execution"),
        "fit execution",
    )
    worker_count = _integer(
        execution.get("worker_count"), "fit worker count", minimum=1
    )
    fit_n_jobs = _integer(
        execution.get("estimator_fit_n_jobs"),
        "fit estimator n_jobs",
        minimum=1,
    )
    persisted_n_jobs = _integer(
        execution.get("persisted_estimator_n_jobs"),
        "persisted estimator n_jobs",
        minimum=1,
    )
    _require(
        execution.get("parallel_unit") == "one_variant_seed_estimator"
        and execution.get("backend") == "sklearn_joblib_loky_processes"
        and execution.get("submission_order") == "variant_then_seed"
        and execution.get("result_collection_order") == "variant_then_seed"
        and execution.get("deterministic_ordered_collection") is True,
        "V4 model fit execution contract changed",
    )
    parameters = _mapping(
        _mapping(contract.get("model"), "model contract").get("parameters"),
        "model parameters",
    )
    _require(
        fit_n_jobs == 1
        and persisted_n_jobs == _integer(
            parameters.get("n_jobs"), "model parameter n_jobs", minimum=1
        ),
        "fit or persisted estimator n_jobs changed",
    )
    specifications = [
        (variant, seed)
        for variant in BASE_VARIANTS
        for seed in contexts[variant][4]
    ]
    _require(
        specifications
        == [
            (variant, seed)
            for variant in BASE_VARIANTS
            for seed in MODEL_SEEDS_IN_ORDER
        ]
        and worker_count == len(specifications) == 20,
        "worker count or ordering differs from the exact 20-estimator schedule",
    )
    fitted = Parallel(
        n_jobs=worker_count,
        backend="loky",
        pre_dispatch=worker_count,
    )(
        delayed(V3._fit_seed_estimator)(
            ExtraTreesClassifier,
            matrices[variant],
            contexts[variant][0],
            contexts[variant][1],
            parameters=contexts[variant][3],
            seed=seed,
            fit_n_jobs=fit_n_jobs,
        )
        for variant, seed in specifications
    )
    _require(
        len(fitted) == len(specifications),
        "ordered estimator result count changed",
    )
    models: dict[str, list[Any]] = {variant: [] for variant in BASE_VARIANTS}
    for (variant, _), model in zip(specifications, fitted, strict=True):
        models[variant].append(model)
    diagnostics = {
        variant: V3._ensemble_fit_diagnostics(
            models[variant],
            contexts[variant][2],
            contexts[variant][4],
            len(rows),
        )
        for variant in BASE_VARIANTS
    }
    return models, {
        "shared": shared_weight_diagnostics,
        "variants": diagnostics,
        "same_weights_and_seed_order_across_base_variants": True,
        "ordered_specifications": [
            {"variant": variant, "seed": seed}
            for variant, seed in specifications
        ],
    }


def _predict_all_base_variants(
    models: Mapping[str, Sequence[Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    floor = _finite(
        _mapping(contract.get("model"), "model contract").get(
            "probability_floor"
        ),
        "probability floor",
    )
    result: dict[str, np.ndarray] = {}
    for variant in BASE_VARIANTS:
        _require(variant in models, f"fitted models lack {variant}")
        result[variant] = V3.aligned_ensemble_probabilities(
            list(models[variant]),
            matrix_for(rows, variant),
            probability_floor=floor,
        )
    return result


def _fit_predict_all_base_variants(
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    training_base_weights: Sequence[float] | np.ndarray | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Single patch point for focused leakage/fold tests."""

    models, diagnostics = _fit_all_base_variants(
        training_rows,
        contract=contract,
        base_weights=training_base_weights,
    )
    return (
        _predict_all_base_variants(models, evaluation_rows, contract=contract),
        diagnostics,
    )


def crossfit_raw_probabilities(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    base_weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Repository-crossfit every base variant with matched folds and weights."""

    contract = validate_contract(contract)
    rows = list(rows)
    _validate_row_roles(rows)
    _require(
        bool(rows) and len(known_rows(rows)) == len(rows),
        "inner crossfit requires known rows",
    )
    _row_ids(rows)
    repositories = np.asarray([str(row.get("repo")) for row in rows], dtype=object)
    _require(
        all(isinstance(row.get("repo"), str) and bool(row.get("repo")) for row in rows),
        "crossfit rows require repository IDs",
    )
    master_weights = (
        V3.hierarchical_equal_weights(rows)
        if base_weights is None
        else V3._validated_unit_weight_vector(
            rows,
            base_weights,
            absolute_tolerance=_finite(
                _mapping(contract.get("weighting"), "weighting contract").get(
                    "prevalidated_crossfit_base_weight_unit_mass_absolute_tolerance"
                ),
                "crossfit unit-mass tolerance",
            ),
            label="V4 crossfit base weights",
        )
    )
    repositories_in_order = sorted(set(repositories.tolist()))
    _require(
        len(repositories_in_order)
        >= _integer(
            _mapping(contract.get("nested"), "nested contract").get(
                "minimum_inner_repositories"
            ),
            "minimum inner repositories",
            minimum=2,
        ),
        "too few repositories for V4 inner crossfit",
    )
    probabilities = {
        variant: np.full((len(rows), len(CLASSES)), np.nan, dtype=np.float64)
        for variant in BASE_VARIANTS
    }
    covered = np.zeros(len(rows), dtype=np.bool_)
    folds: list[dict[str, Any]] = []
    for heldout_repository in repositories_in_order:
        training_indices = np.flatnonzero(repositories != heldout_repository)
        evaluation_indices = np.flatnonzero(repositories == heldout_repository)
        training_rows = [rows[int(index)] for index in training_indices]
        evaluation_rows = [rows[int(index)] for index in evaluation_indices]
        _require(
            set(labels_for(training_rows).tolist()) == set(CLASSES),
            f"inner training split for {heldout_repository} lacks a class",
        )
        training_base_weights = V3.restrict_base_weights(
            rows, master_weights, training_indices
        )
        fold_probabilities, diagnostics = _fit_predict_all_base_variants(
            training_rows,
            evaluation_rows,
            contract=contract,
            training_base_weights=training_base_weights,
        )
        for variant in BASE_VARIANTS:
            values = DECISION.validate_probability_matrix(
                fold_probabilities[variant],
                f"{variant} inner heldout probabilities",
                expected_rows=len(evaluation_rows),
            )
            probabilities[variant][evaluation_indices] = values
        covered[evaluation_indices] = True
        folds.append(
            {
                "heldout_repository": heldout_repository,
                "training_repositories": sorted(
                    set(repositories[training_indices].tolist())
                ),
                "training_row_ids_sha256": V3.canonical_json_sha256(
                    _row_ids(training_rows)
                ),
                "evaluation_row_ids_sha256": V3.canonical_json_sha256(
                    _row_ids(evaluation_rows)
                ),
                "training_rows": len(training_rows),
                "evaluation_rows": len(evaluation_rows),
                "heldout_labels_used_for_fit_or_selection": False,
                "seed_order": list(MODEL_SEEDS_IN_ORDER),
                "shared_training_weight_sha256": diagnostics["shared"][
                    "weight_float64_sha256"
                ],
                "training_base_weight_sha256": V3.sha256_bytes(
                    np.asarray(training_base_weights, dtype="<f8").tobytes(
                        order="C"
                    )
                ),
            }
        )
    _require(np.all(covered), "inner crossfit did not cover every row")
    for variant in BASE_VARIANTS:
        probabilities[variant] = DECISION.validate_probability_matrix(
            probabilities[variant], f"{variant} inner OOF probabilities"
        )
    return {
        "probabilities": probabilities,
        "folds": folds,
        "repositories_in_order": repositories_in_order,
        "all_rows_covered_once": True,
    }


def _validate_action_settings_identity(
    settings: Any,
    *,
    contract: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    value = _mapping(settings, label)
    decision_contract = _mapping(contract.get("decision"), "decision contract")
    expected_grids = {
        "alpha": list(ZERO_ALPHA_GRID),
        "edit_offset": list(decision_contract["edit_offset_grid"]),
        "check_or_finish_offset": list(
            decision_contract["check_or_finish_offset_grid"]
        ),
    }
    _require(
        value.get("schema_version") == DECISION.SCHEMA_VERSION
        and value.get("classes_in_order") == list(CLASSES)
        and value.get("grids") == expected_grids
        and value.get("recall_floors") == decision_contract["recall_floors"]
        and value.get("balanced_accuracy_minimum")
        == decision_contract["balanced_accuracy_minimum"]
        and value.get("candidate_count")
        == math.prod(len(grid) for grid in expected_grids.values())
        and value.get("alpha") == 0.0
        and isinstance(value.get("fallback_used"), bool)
        and isinstance(value.get("selected_under_floors"), bool)
        and value.get("selected_under_floors")
        is (not value.get("fallback_used")),
        f"{label} contract identity changed",
    )
    return value


def select_shared_action_settings(
    raw_probabilities: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    weights: Sequence[float] | np.ndarray,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the one ordinary-logit action rule shared by both study arms."""

    contract = validate_contract(contract)
    rows = list(rows)
    _validate_row_roles(rows)
    _require(
        bool(rows) and len(known_rows(rows)) == len(rows),
        "shared action selection requires known rows",
    )
    raw_action = _base_probability(
        raw_probabilities, REFERENCE_PROCEDURE, "shared action selection"
    )
    decision_contract = _mapping(contract.get("decision"), "decision contract")
    selected = DECISION.select_decision_settings(
        raw_action,
        raw_action,
        labels_for(rows).tolist(),
        weights,
        alpha_grid=ZERO_ALPHA_GRID,
        edit_offset_grid=decision_contract["edit_offset_grid"],
        check_or_finish_offset_grid=decision_contract[
            "check_or_finish_offset_grid"
        ],
        recall_floors=decision_contract["recall_floors"],
        balanced_accuracy_minimum=decision_contract[
            "balanced_accuracy_minimum"
        ],
        include_candidates=False,
    )
    _validate_action_settings_identity(
        selected, contract=contract, label="shared action settings"
    )
    return selected


def select_procedure_settings(
    raw_probabilities: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    weights: Sequence[float] | np.ndarray,
    procedure: str,
    *,
    contract: Mapping[str, Any],
    shared_action_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one forecast/calibration/threshold procedure on inner OOF rows."""

    contract = validate_contract(contract)
    rows = list(rows)
    _validate_row_roles(rows)
    _require(
        bool(rows) and len(known_rows(rows)) == len(rows),
        "procedure selection requires known rows",
    )
    label_values = labels_for(rows).tolist()
    decision_contract = _mapping(contract.get("decision"), "decision contract")
    calibration_contract = _mapping(
        contract.get("calibration"), "calibration contract"
    )
    abstention_contract = _mapping(
        contract.get("abstention"), "abstention contract"
    )
    forecast_raw_p = _forecast_raw_probability(raw_probabilities, procedure)
    decision_raw_r = _base_probability(
        raw_probabilities,
        _decision_source(procedure),
        f"{procedure} decision source",
    )
    _require(
        len(forecast_raw_p) == len(decision_raw_r) == len(rows),
        "procedure-selection row count changed",
    )

    if procedure in SHARED_ACTION_PROCEDURES:
        _require(
            shared_action_settings is not None,
            f"{procedure} requires the once-selected shared action settings",
        )
        decision_settings = _validate_action_settings_identity(
            shared_action_settings,
            contract=contract,
            label=f"{procedure} shared action settings",
        )
        shared_action_settings_sha256: str | None = V3.canonical_json_sha256(
            decision_settings
        )
        decision_role = "shared_sequence_logit_action_policy"
    else:
        _require(
            shared_action_settings is None,
            f"{procedure} cannot receive shared action settings",
        )
        decision_settings = DECISION.select_decision_settings(
            decision_raw_r,
            decision_raw_r,
            label_values,
            weights,
            alpha_grid=ZERO_ALPHA_GRID,
            edit_offset_grid=decision_contract["edit_offset_grid"],
            check_or_finish_offset_grid=decision_contract[
                "check_or_finish_offset_grid"
            ],
            recall_floors=decision_contract["recall_floors"],
            balanced_accuracy_minimum=decision_contract[
                "balanced_accuracy_minimum"
            ],
            include_candidates=False,
        )
        _validate_action_settings_identity(
            decision_settings,
            contract=contract,
            label=f"{procedure} action settings",
        )
        shared_action_settings_sha256 = None
        decision_role = "procedure_specific_action_diagnostic"
    calibration_settings = CALIBRATION.select_temperature_settings(
        forecast_raw_p,
        label_values,
        weights,
        temperature_grid=calibration_contract["temperature_grid"],
    )
    forecast_q = CALIBRATION.apply_temperature_settings(
        forecast_raw_p,
        calibration_settings,
        temperature_grid=calibration_contract["temperature_grid"],
    )
    decision_d = DECISION.apply_decision_settings(
        decision_raw_r,
        decision_raw_r,
        decision_settings,
        alpha_grid=ZERO_ALPHA_GRID,
        edit_offset_grid=decision_contract["edit_offset_grid"],
        check_or_finish_offset_grid=decision_contract[
            "check_or_finish_offset_grid"
        ],
    )
    threshold_settings = DECISION.select_class_confidence_thresholds(
        decision_d,
        label_values,
        weights,
        threshold_grid=abstention_contract["threshold_grid"],
        accepted_accuracy_minimum=abstention_contract[
            "accepted_accuracy_minimum"
        ],
        coverage_minimum=abstention_contract["coverage_minimum"],
        minimum_accepted_rows_per_true_class=abstention_contract[
            "minimum_accepted_rows_per_true_class"
        ],
        confidence_probabilities=forecast_q,
        include_candidates=False,
    )
    accepted = DECISION.apply_class_confidence_thresholds(
        decision_d,
        threshold_settings,
        threshold_grid=abstention_contract["threshold_grid"],
        confidence_probabilities=forecast_q,
    )
    metrics = METRICS.known_row_metrics(
        forecast_q,
        decision_d,
        label_values,
        weights,
        accepted,
        ece_bins=_mapping(contract.get("metrics"), "metrics contract")[
            "ece_bins"
        ],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "procedure": procedure,
        "forecast_raw_sources_in_order": list(_forecast_sources(procedure)),
        "decision_raw_source": _decision_source(procedure),
        "decision_role": decision_role,
        "shared_action_settings_sha256": shared_action_settings_sha256,
        "selection_order": [
            "once_selected_shared_action_offsets_where_applicable",
            "fixed_forecast_raw_probability_construction",
            "independent_scalar_temperature_forecast_q",
            "shared_or_diagnostic_raw_offset_decision_d",
            "predicted_class_thresholds_using_q_at_argmax_d",
        ],
        "decision": decision_settings,
        "calibration": calibration_settings,
        "abstention": threshold_settings,
        "inner_selection_metrics": metrics,
    }


def _validate_procedure_settings_identity(
    settings: Any,
    procedure: str,
    *,
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = _mapping(settings, f"{procedure} settings")
    _require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("procedure") == procedure
        and value.get("forecast_raw_sources_in_order")
        == list(_forecast_sources(procedure))
        and value.get("decision_raw_source") == _decision_source(procedure)
        and value.get("selection_order")
        == [
            "once_selected_shared_action_offsets_where_applicable",
            "fixed_forecast_raw_probability_construction",
            "independent_scalar_temperature_forecast_q",
            "shared_or_diagnostic_raw_offset_decision_d",
            "predicted_class_thresholds_using_q_at_argmax_d",
        ],
        f"{procedure} settings identity changed",
    )
    decision_settings = _validate_action_settings_identity(
        value.get("decision"),
        contract=contract,
        label=f"{procedure} decision settings",
    )
    _mapping(value.get("calibration"), f"{procedure} calibration settings")
    abstention_settings = _mapping(
        value.get("abstention"), f"{procedure} abstention settings"
    )
    abstention_contract = _mapping(
        contract.get("abstention"), "abstention contract"
    )
    shared = procedure in SHARED_ACTION_PROCEDURES
    expected_shared_hash = (
        V3.canonical_json_sha256(decision_settings) if shared else None
    )
    _require(
        value.get("decision_role")
        == (
            "shared_sequence_logit_action_policy"
            if shared
            else "procedure_specific_action_diagnostic"
        )
        and value.get("shared_action_settings_sha256") == expected_shared_hash,
        f"{procedure} shared-action identity changed",
    )
    _require(
        abstention_settings.get("threshold_grid")
        == abstention_contract["threshold_grid"]
        and abstention_settings.get("accepted_accuracy_minimum")
        == abstention_contract["accepted_accuracy_minimum"]
        and abstention_settings.get("coverage_minimum")
        == abstention_contract["coverage_minimum"]
        and abstention_settings.get("minimum_accepted_rows_per_true_class")
        == abstention_contract["minimum_accepted_rows_per_true_class"]
        and abstention_settings.get("candidate_count")
        == len(abstention_contract["threshold_grid"]) ** len(CLASSES)
        and abstention_settings.get("confidence_source")
        == "separate_confidence_probabilities"
        and isinstance(abstention_settings.get("fallback_used"), bool)
        and isinstance(abstention_settings.get("selected_under_floors"), bool)
        and abstention_settings.get("selected_under_floors")
        is (not abstention_settings.get("fallback_used")),
        f"{procedure} abstention-setting contract identity changed",
    )
    return value


def apply_procedure_settings(
    raw_probabilities: Mapping[str, Any],
    settings: Any,
    procedure: str,
    *,
    contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Apply one inner-selected two-head procedure without labels or weights."""

    contract = validate_contract(contract)
    value = _validate_procedure_settings_identity(
        settings, procedure, contract=contract
    )
    decision_contract = _mapping(contract.get("decision"), "decision contract")
    calibration_contract = _mapping(
        contract.get("calibration"), "calibration contract"
    )
    abstention_contract = _mapping(
        contract.get("abstention"), "abstention contract"
    )
    forecast_raw_p = _forecast_raw_probability(raw_probabilities, procedure)
    decision_raw_r = _base_probability(
        raw_probabilities,
        _decision_source(procedure),
        f"{procedure} decision application",
    )
    _require(
        len(forecast_raw_p) == len(decision_raw_r),
        "forecast and decision row counts differ",
    )
    decision_settings = value["decision"]
    forecast_q = CALIBRATION.apply_temperature_settings(
        forecast_raw_p,
        value["calibration"],
        temperature_grid=calibration_contract["temperature_grid"],
    )
    decision_d = DECISION.apply_decision_settings(
        decision_raw_r,
        decision_raw_r,
        decision_settings,
        alpha_grid=ZERO_ALPHA_GRID,
        edit_offset_grid=decision_contract["edit_offset_grid"],
        check_or_finish_offset_grid=decision_contract[
            "check_or_finish_offset_grid"
        ],
    )
    accepted = DECISION.apply_class_confidence_thresholds(
        decision_d,
        value["abstention"],
        threshold_grid=abstention_contract["threshold_grid"],
        confidence_probabilities=forecast_q,
    )
    return {
        "forecast_raw_probability_p": forecast_raw_p,
        "decision_raw_probability_r": decision_raw_r,
        "forecast_probability_q": forecast_q,
        "decision_probability_d": decision_d,
        "accepted": accepted,
    }


def _per_repository_metrics(
    rows: Sequence[Mapping[str, Any]],
    forecast_q: np.ndarray,
    decision_d: np.ndarray,
    accepted: np.ndarray,
    weights: Sequence[float] | np.ndarray,
    *,
    ece_bins: int,
) -> dict[str, Any]:
    rows = list(rows)
    master_weights = V3._normalized_positive_weights(
        rows, weights, "V4 per-repository metric weights"
    )
    result: dict[str, Any] = {}
    for repository in sorted({str(row["repo"]) for row in rows}):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if str(row["repo"]) == repository
            ],
            dtype=np.int64,
        )
        repository_rows = [rows[int(index)] for index in indices]
        repository_labels = labels_for(repository_rows)
        class_support = {
            class_id: int(np.sum(repository_labels == class_id))
            for class_id in CLASSES
        }
        repository_weights = V3.restrict_base_weights(
            rows, master_weights, indices
        )
        repository_metrics = METRICS.known_row_metrics(
            forecast_q[indices],
            decision_d[indices],
            repository_labels.tolist(),
            repository_weights,
            accepted[indices],
            ece_bins=ece_bins,
            require_all_classes=False,
        )
        repository_metrics["status"] = (
            "complete_three_class_support"
            if all(class_support.values())
            else "partial_class_support"
        )
        repository_metrics["class_support"] = class_support
        result[repository] = repository_metrics
    return result


def _prediction_record(
    row: Mapping[str, Any],
    forecast_raw_p: np.ndarray,
    decision_raw_r: np.ndarray,
    forecast_q: np.ndarray,
    decision_d: np.ndarray,
    accepted: bool,
) -> dict[str, Any]:
    decision_index = int(np.argmax(decision_d))
    forecast_index = int(np.argmax(forecast_q))
    return {
        **{
            key: row.get(key)
            for key in (
                "row_id",
                "task_id",
                "repo",
                "cohort_id",
                "task_request_index",
                "checkpoint_ordinal",
                "source_action_label_status",
                "source_action_class_id",
                "label_status",
                "label",
                "metric_evaluable",
                "auxiliary_diagnostics",
            )
        },
        "forecast_raw_probabilities_p": {
            class_id: float(forecast_raw_p[index])
            for index, class_id in enumerate(CLASSES)
        },
        "decision_raw_probabilities_r": {
            class_id: float(decision_raw_r[index])
            for index, class_id in enumerate(CLASSES)
        },
        "forecast_probabilities_q": {
            class_id: float(forecast_q[index])
            for index, class_id in enumerate(CLASSES)
        },
        "decision_probabilities_d": {
            class_id: float(decision_d[index])
            for index, class_id in enumerate(CLASSES)
        },
        "forecast_top_class": CLASSES[forecast_index],
        "forecast_top_confidence": float(forecast_q[forecast_index]),
        "predicted_class": CLASSES[decision_index],
        "decision_confidence_from_q": float(forecast_q[decision_index]),
        "accepted": bool(accepted),
    }


def nested_leave_one_repository_out(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    base_weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Run full nested LORO and predict known plus unknown stable rows once."""

    contract = validate_contract(contract)
    rows = list(rows)
    _require(bool(rows), "nested V4 evaluation has no rows")
    _row_ids(rows)
    _validate_row_roles(rows)
    for row in rows:
        _require(
            isinstance(row.get("repo"), str)
            and bool(row.get("repo"))
            and isinstance(row.get("task_id"), str)
            and bool(row.get("task_id")),
            "nested rows require nonempty repository and task IDs",
        )
    known_indices_in_all = [
        index
        for index, row in enumerate(rows)
        if row.get("metric_evaluable") is True
    ]
    known = [rows[index] for index in known_indices_in_all]
    _require(bool(known), "nested V4 evaluation has no known rows")
    _require(set(labels_for(known).tolist()) == set(CLASSES), "known rows lack a class")
    known_base_weights = (
        V3.hierarchical_equal_weights(known)
        if base_weights is None
        else V3._normalized_positive_weights(
            known, base_weights, "V4 nested known-row base weights"
        )
    )
    base_weight_source = (
        "hierarchical_equal_point_weights"
        if base_weights is None
        else "hierarchical_bayesian_bootstrap_draw_weights"
    )
    all_prediction_weights = V3.hierarchical_equal_weights(rows)
    known_mask_all = np.asarray(
        [row.get("metric_evaluable") is True for row in rows], dtype=np.bool_
    )
    hierarchical_known_action_fraction = float(
        all_prediction_weights[known_mask_all].sum(dtype=np.float64)
    )
    repositories = np.asarray([str(row["repo"]) for row in rows], dtype=object)
    repositories_in_order = sorted(set(repositories.tolist()))
    _require(len(repositories_in_order) >= 3, "nested V4 LORO needs repositories")
    _require(
        sorted({str(row["repo"]) for row in known}) == repositories_in_order,
        "every prediction repository must contain a known row",
    )

    forecast_raw_p = {
        procedure: np.full((len(rows), len(CLASSES)), np.nan, dtype=np.float64)
        for procedure in PROCEDURES
    }
    decision_raw_r = {
        procedure: np.full((len(rows), len(CLASSES)), np.nan, dtype=np.float64)
        for procedure in PROCEDURES
    }
    forecast_q = {
        procedure: np.full((len(rows), len(CLASSES)), np.nan, dtype=np.float64)
        for procedure in PROCEDURES
    }
    decision_d = {
        procedure: np.full((len(rows), len(CLASSES)), np.nan, dtype=np.float64)
        for procedure in PROCEDURES
    }
    accepted = {
        procedure: np.zeros(len(rows), dtype=np.bool_)
        for procedure in PROCEDURES
    }
    covered = np.zeros(len(rows), dtype=np.bool_)
    fold_records: list[dict[str, Any]] = []

    for heldout_repository in repositories_in_order:
        outer_training_known_indices = [
            index
            for index, row in enumerate(known)
            if str(row["repo"]) != heldout_repository
        ]
        outer_training_rows = [known[index] for index in outer_training_known_indices]
        outer_training_weights = V3.restrict_base_weights(
            known, known_base_weights, outer_training_known_indices
        )
        evaluation_indices = np.flatnonzero(repositories == heldout_repository)
        evaluation_rows = [rows[int(index)] for index in evaluation_indices]
        _require(
            set(labels_for(outer_training_rows).tolist()) == set(CLASSES),
            f"outer training split for {heldout_repository} lacks a class",
        )

        inner = crossfit_raw_probabilities(
            outer_training_rows,
            contract=contract,
            base_weights=outer_training_weights,
        )
        shared_action_settings = select_shared_action_settings(
            inner["probabilities"],
            outer_training_rows,
            outer_training_weights,
            contract=contract,
        )
        settings = {
            procedure: select_procedure_settings(
                inner["probabilities"],
                outer_training_rows,
                outer_training_weights,
                procedure,
                contract=contract,
                shared_action_settings=(
                    shared_action_settings
                    if procedure in SHARED_ACTION_PROCEDURES
                    else None
                ),
            )
            for procedure in PROCEDURES
        }
        raw_heldout, fit_diagnostics = _fit_predict_all_base_variants(
            outer_training_rows,
            evaluation_rows,
            contract=contract,
            training_base_weights=outer_training_weights,
        )
        for procedure in PROCEDURES:
            outputs = apply_procedure_settings(
                raw_heldout,
                settings[procedure],
                procedure,
                contract=contract,
            )
            forecast_raw_p[procedure][evaluation_indices] = outputs[
                "forecast_raw_probability_p"
            ]
            decision_raw_r[procedure][evaluation_indices] = outputs[
                "decision_raw_probability_r"
            ]
            forecast_q[procedure][evaluation_indices] = outputs[
                "forecast_probability_q"
            ]
            decision_d[procedure][evaluation_indices] = outputs[
                "decision_probability_d"
            ]
            accepted[procedure][evaluation_indices] = outputs["accepted"]
        _require(
            settings[PRIMARY_PROCEDURE]["decision"]
            == settings[REFERENCE_PROCEDURE]["decision"]
            == shared_action_settings
            and settings[PRIMARY_PROCEDURE]["shared_action_settings_sha256"]
            == settings[REFERENCE_PROCEDURE]["shared_action_settings_sha256"]
            == V3.canonical_json_sha256(shared_action_settings),
            "candidate and reference did not retain one shared action artifact",
        )
        _require(
            np.array_equal(
                decision_raw_r[PRIMARY_PROCEDURE][evaluation_indices],
                decision_raw_r[REFERENCE_PROCEDURE][evaluation_indices],
            )
            and np.array_equal(
                decision_d[PRIMARY_PROCEDURE][evaluation_indices],
                decision_d[REFERENCE_PROCEDURE][evaluation_indices],
            ),
            "candidate and reference action outputs differ in an outer fold",
        )
        covered[evaluation_indices] = True

        selection_ids = _row_ids(outer_training_rows)
        heldout_ids = _row_ids(evaluation_rows)
        _require(
            set(selection_ids).isdisjoint(heldout_ids),
            "outer heldout rows leaked into V4 selection",
        )
        fold_records.append(
            {
                "heldout_repository": heldout_repository,
                "heldout_prediction_rows": len(evaluation_rows),
                "heldout_known_rows": sum(
                    row.get("metric_evaluable") is True for row in evaluation_rows
                ),
                "outer_training_known_rows": len(outer_training_rows),
                "outer_training_repositories": sorted(
                    {str(row["repo"]) for row in outer_training_rows}
                ),
                "inner_fold_count": len(inner["folds"]),
                "inner_folds": inner["folds"],
                "inner_selection_row_ids_sha256": V3.canonical_json_sha256(
                    selection_ids
                ),
                "heldout_row_ids_sha256": V3.canonical_json_sha256(heldout_ids),
                "inner_and_heldout_row_ids_disjoint": True,
                "heldout_labels_used_for_fit_or_selection": False,
                "settings": settings,
                "shared_action_settings": shared_action_settings,
                "shared_action_settings_sha256": V3.canonical_json_sha256(
                    shared_action_settings
                ),
                "candidate_reference_decision_raw_r_exactly_equal": True,
                "candidate_reference_decision_d_exactly_equal": True,
                "shared_outer_training_weight_sha256": fit_diagnostics["shared"][
                    "weight_float64_sha256"
                ],
                "outer_training_base_weight_sha256": V3.sha256_bytes(
                    np.asarray(outer_training_weights, dtype="<f8").tobytes(
                        order="C"
                    )
                ),
                "same_folds_weights_seeds_and_model_across_base_variants": True,
            }
        )

    _require(np.all(covered), "outer V4 LORO did not predict every stable row")
    known_indices_array = np.asarray(known_indices_in_all, dtype=np.int64)
    ece_bins = _integer(
        _mapping(contract.get("metrics"), "metrics contract").get("ece_bins"),
        "ECE bins",
        minimum=1,
    )
    results: dict[str, Any] = {}
    for procedure in PROCEDURES:
        forecast_raw_p[procedure] = DECISION.validate_probability_matrix(
            forecast_raw_p[procedure], f"{procedure} outer forecast raw p"
        )
        decision_raw_r[procedure] = DECISION.validate_probability_matrix(
            decision_raw_r[procedure], f"{procedure} outer decision raw r"
        )
        forecast_q[procedure] = DECISION.validate_probability_matrix(
            forecast_q[procedure], f"{procedure} outer forecast q"
        )
        decision_d[procedure] = DECISION.validate_probability_matrix(
            decision_d[procedure], f"{procedure} outer decision d"
        )
        known_q = forecast_q[procedure][known_indices_array]
        known_d = decision_d[procedure][known_indices_array]
        known_accepted = accepted[procedure][known_indices_array]
        metrics = METRICS.known_row_metrics(
            known_q,
            known_d,
            labels_for(known).tolist(),
            known_base_weights,
            known_accepted,
            ece_bins=ece_bins,
        )
        metrics["known_action_fraction"] = hierarchical_known_action_fraction
        metrics["selected_coverage_denominator"] = (
            "known_current_action_metric_rows_only_not_all_stable_emissions"
        )
        predictions = [
            _prediction_record(row, p, r, q, d, bool(is_accepted))
            for row, p, r, q, d, is_accepted in zip(
                rows,
                forecast_raw_p[procedure],
                decision_raw_r[procedure],
                forecast_q[procedure],
                decision_d[procedure],
                accepted[procedure],
                strict=True,
            )
        ]
        results[procedure] = {
            "inference_prediction_count": len(rows),
            "known_action_metric_row_count": len(known),
            "unknown_action_prediction_count": len(rows) - len(known),
            "accepted_inference_prediction_count": int(
                accepted[procedure].sum()
            ),
            "inference_acceptance_fraction": float(
                accepted[procedure].mean()
            ),
            "inference_acceptance_fraction_denominator": (
                "all_stable_feature_complete_predictions"
            ),
            "metrics": metrics,
            "per_repository_metrics": _per_repository_metrics(
                known,
                known_q,
                known_d,
                known_accepted,
                known_base_weights,
                ece_bins=ece_bins,
            ),
            "predictions": predictions,
            "forecast_raw_p_float64_sha256": V3.sha256_bytes(
                np.asarray(forecast_raw_p[procedure], dtype="<f8").tobytes(
                    order="C"
                )
            ),
            "decision_raw_r_float64_sha256": V3.sha256_bytes(
                np.asarray(decision_raw_r[procedure], dtype="<f8").tobytes(
                    order="C"
                )
            ),
            "forecast_q_float64_sha256": V3.sha256_bytes(
                np.asarray(forecast_q[procedure], dtype="<f8").tobytes(order="C")
            ),
            "decision_d_float64_sha256": V3.sha256_bytes(
                np.asarray(decision_d[procedure], dtype="<f8").tobytes(order="C")
            ),
            "acceptance_sha256": V3.sha256_bytes(
                np.asarray(accepted[procedure], dtype=np.uint8).tobytes(order="C")
            ),
        }

    _require(
        np.array_equal(
            decision_raw_r[PRIMARY_PROCEDURE],
            decision_raw_r[REFERENCE_PROCEDURE],
        )
        and np.array_equal(
            decision_d[PRIMARY_PROCEDURE], decision_d[REFERENCE_PROCEDURE]
        ),
        "candidate and reference action outputs differ across development",
    )
    shared_decision_raw_r_sha256 = results[PRIMARY_PROCEDURE][
        "decision_raw_r_float64_sha256"
    ]
    shared_decision_d_sha256 = results[PRIMARY_PROCEDURE][
        "decision_d_float64_sha256"
    ]
    _require(
        shared_decision_raw_r_sha256
        == results[REFERENCE_PROCEDURE]["decision_raw_r_float64_sha256"]
        and shared_decision_d_sha256
        == results[REFERENCE_PROCEDURE]["decision_d_float64_sha256"],
        "candidate and reference action hashes differ",
    )

    full_crossfit = crossfit_raw_probabilities(
        known, contract=contract, base_weights=known_base_weights
    )
    full_shared_action_settings = select_shared_action_settings(
        full_crossfit["probabilities"],
        known,
        known_base_weights,
        contract=contract,
    )
    full_settings = {
        procedure: select_procedure_settings(
            full_crossfit["probabilities"],
            known,
            known_base_weights,
            procedure,
            contract=contract,
            shared_action_settings=(
                full_shared_action_settings
                if procedure in SHARED_ACTION_PROCEDURES
                else None
            ),
        )
        for procedure in PROCEDURES
    }
    _require(
        full_settings[PRIMARY_PROCEDURE]["decision"]
        == full_settings[REFERENCE_PROCEDURE]["decision"]
        == full_shared_action_settings,
        "full-development candidate/reference action artifact differs",
    )
    primary_full_settings = full_settings[PRIMARY_PROCEDURE]
    full_action_rule_passed = (
        primary_full_settings["decision"].get("selected_under_floors") is True
        and primary_full_settings["decision"].get("fallback_used") is False
    )
    full_candidate_abstention_passed = (
        primary_full_settings["abstention"].get("selected_under_floors") is True
        and primary_full_settings["abstention"].get("fallback_used") is False
    )
    reference_full_settings = full_settings[REFERENCE_PROCEDURE]
    full_reference_abstention_passed = (
        reference_full_settings["abstention"].get("selected_under_floors") is True
        and reference_full_settings["abstention"].get("fallback_used") is False
    )
    both_full_abstention_passed = (
        full_candidate_abstention_passed and full_reference_abstention_passed
    )
    full_development_promotion = {
        "primary_procedure": PRIMARY_PROCEDURE,
        "candidate_reference_shared_action_identity_passed": True,
        "action_rule_selected_under_floors": full_action_rule_passed,
        "candidate_abstention_selected_under_floors": (
            full_candidate_abstention_passed
        ),
        "reference_abstention_selected_under_floors": (
            full_reference_abstention_passed
        ),
        "both_abstention_branches_selected_under_floors": (
            both_full_abstention_passed
        ),
        "fallback_blocks_promotion": not (
            full_action_rule_passed and both_full_abstention_passed
        ),
        "eligible_on_full_development_selection": (
            full_action_rule_passed and both_full_abstention_passed
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm": (
            "nested_leave_one_repository_out_v4_fixed_geometric_j_forecast_"
            "shared_logit_action"
        ),
        "base_variants_in_order": list(BASE_VARIANTS),
        "ordered_base_variant_seed_schedule": [
            {"variant": variant, "seed": seed}
            for variant in BASE_VARIANTS
            for seed in MODEL_SEEDS_IN_ORDER
        ],
        "procedures_in_order": list(PROCEDURES),
        "primary_procedure": PRIMARY_PROCEDURE,
        "reference_procedure": REFERENCE_PROCEDURE,
        "forecast_pool_j_weight": FORECAST_POOL_J_WEIGHT,
        "candidate_reference_shared_action_policy": True,
        "candidate_reference_decision_raw_r_exactly_equal": True,
        "candidate_reference_decision_d_exactly_equal": True,
        "shared_decision_raw_r_float64_sha256": shared_decision_raw_r_sha256,
        "shared_decision_d_float64_sha256": shared_decision_d_sha256,
        "evaluator_contract_sha256": V3.canonical_json_sha256(contract),
        "ordered_row_identity_sha256": ordered_row_identity_sha256(rows),
        "known_ordered_row_identity_sha256": ordered_row_identity_sha256(known),
        "known_base_weight_float64_sha256": V3.sha256_bytes(
            np.asarray(known_base_weights, dtype="<f8").tobytes(order="C")
        ),
        "base_weight_source": base_weight_source,
        "point_estimand": POINT_ESTIMAND,
        "repositories_in_order": repositories_in_order,
        "outer_fold_count": len(fold_records),
        "all_stable_feature_complete_rows_predicted_once": True,
        "unknown_current_actions_received_predictions": True,
        "outer_heldout_labels_used_for_fit_or_selection": False,
        "folds": fold_records,
        "results": results,
        "full_development_crossfit_folds": full_crossfit["folds"],
        "full_development_shared_action_settings": full_shared_action_settings,
        "full_development_shared_action_settings_sha256": (
            V3.canonical_json_sha256(full_shared_action_settings)
        ),
        "full_development_settings": full_settings,
        "full_development_promotion": full_development_promotion,
        "full_development_oof_raw_probabilities": {
            variant: {
                "known_row_ids_sha256": V3.canonical_json_sha256(
                    _row_ids(known)
                ),
                "raw_probabilities": full_crossfit["probabilities"][variant].tolist(),
                "float64_sha256": V3.sha256_bytes(
                    np.asarray(
                        full_crossfit["probabilities"][variant], dtype="<f8"
                    ).tobytes(order="C")
                ),
            }
            for variant in BASE_VARIANTS
        },
    }


__all__ = [
    "BASE_VARIANTS",
    "CLASSES",
    "FORECAST_POOL_J_WEIGHT",
    "MODEL_FIT_EXECUTION",
    "MODEL_PARAMETERS",
    "MODEL_PREDICTION_EXECUTION",
    "MODEL_SEEDS_IN_ORDER",
    "POINT_ESTIMAND",
    "PRIMARY_PROCEDURE",
    "PROCEDURES",
    "REFERENCE_PROCEDURE",
    "SHARED_ACTION_PROCEDURES",
    "V3_ANALYZER_PATH",
    "V3_ANALYZER_SHA256",
    "_authenticate_v3_source",
    "apply_procedure_settings",
    "crossfit_raw_probabilities",
    "known_rows",
    "labels_for",
    "matrix_for",
    "nested_leave_one_repository_out",
    "ordered_row_identity_sha256",
    "select_shared_action_settings",
    "select_procedure_settings",
    "validate_contract",
]
