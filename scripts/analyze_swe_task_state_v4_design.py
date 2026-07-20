#!/usr/bin/env python3
"""Run the explicitly nonconfirmatory V4 screen on frozen V3 N60 evidence.

This command can only consume the exact authenticated V3 development bundle.
It cannot run a bootstrap, fit a deployable model, open validation, or write
outside the dedicated V4 design cache.  Its sole purpose is to debug and
choose a V4 protocol before a fresh disjoint development cohort is selected.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts import swe_task_state_v4_evaluator as EVALUATOR
    from scripts import swe_task_state_v4_extract as EXTRACT
except ModuleNotFoundError as error:  # Support direct execution from scripts/.
    if error.name != "scripts":
        raise
    import swe_task_state_v4_evaluator as EVALUATOR  # type: ignore[no-redef]
    import swe_task_state_v4_extract as EXTRACT  # type: ignore[no-redef]


V3 = EVALUATOR.V3
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN_CONFIG = (
    ROOT / "configs/swe_task_state_interpreter_v4_design_screen.json"
)
DEFAULT_V3_PROTOCOL = ROOT / "configs/swe_task_state_interpreter_v3.json"
DEFAULT_ACTION_PROTOCOL = ROOT / "configs/swe_task_state_v3_action_probes.json"
DEFAULT_COHORT = ROOT / "configs/swe_task_state_v3_development_cohort.json"
DEFAULT_PROMPTS = ROOT / ".cache/swe_state_interpreter_v3_development/prompts.json"
DEFAULT_PROMPTS_SUMMARY = (
    ROOT / ".cache/swe_state_interpreter_v3_development/prompts-summary.json"
)
DEFAULT_PUBLIC_REPORT = (
    ROOT / ".cache/swe_state_interpreter_v3_development/replay/public-report.json"
)
DEFAULT_REPLAY_RECEIPT = (
    ROOT / ".cache/swe_state_interpreter_v3_development/replay/merge-manifest.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / ".cache/swe_state_interpreter_v4_design"
PRIOR_DESIGN_RESULT = (
    DEFAULT_OUTPUT_ROOT / "v3-n60-sequence-b2-design.json"
)
V3_STOP_DECISION = ROOT / "validation/swe-task-state-v3-development-decision.json"

SCHEMA_VERSION = 1
DESIGN_ID = "swe-task-state-interpreter-v4-v3-n60-design-screen-v2"
DESIGN_CONFIG_SHA256 = (
    "cca7687e9c061e7f600469d4db89312e9b55739034bf09fd7a81be2a1f86d04d"
)
DESIGN_CONFIG_CANONICAL_SHA256 = (
    "6378cc92f78f6e662dc9493756a240ec1304ccbc9735ce17ef5932290e81624d"
)

EXPECTED_PRIOR_DESIGN_RESULT = {
    "path": (
        ".cache/swe_state_interpreter_v4_design/v3-n60-sequence-b2-design.json"
    ),
    "sha256": (
        "8ec24b65b71b831e5206af353857766a58fbc0a343acee6f6ab7071cccb3489e"
    ),
    "status": "completed_design_screen_not_confirmatory",
    "confirmatory_interpretation_forbidden": True,
    "all_design_support_point_and_selection_checks_passed": False,
    "point_failures": [
        {
            "metric": "balanced_accuracy",
            "observed": 0.7491604610268023,
            "operator": ">=",
            "threshold": 0.75,
        },
        {
            "metric": "selected_accepted_accuracy",
            "observed": 0.8421313478421713,
            "operator": ">=",
            "threshold": 0.85,
        },
    ],
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _read_json(path: Path, label: str) -> Any:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {path}: {error}") from error


def _exact_numeric_mapping(
    value: Any,
    expected: Mapping[str, float | int],
    label: str,
) -> dict[str, float | int]:
    observed = _mapping(value, label)
    _require(set(observed) == set(expected), f"{label} keys changed")
    result: dict[str, float | int] = {}
    for key, expected_value in expected.items():
        item = observed.get(key)
        _require(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            and type(item) is type(expected_value)
            and item == expected_value,
            f"{label} value changed: {key}",
        )
        result[key] = item
    return result


def validate_design_config(value: Any) -> dict[str, Any]:
    config = _mapping(value, "V4 design-screen config")
    _require(
        set(config)
        == {
            "schema_version",
            "id",
            "status",
            "decision_scope",
            "prior_design_result",
            "design_data",
            "target_contract",
            "feature_contract",
            "model_source",
            "evaluator_contract",
            "procedures",
            "point_screen_thresholds",
            "support_screen_thresholds",
            "output_contract",
            "reserved_validation_policy",
        },
        "V4 design-screen top-level fields changed",
    )
    _require(
        config.get("schema_version") == SCHEMA_VERSION
        and config.get("id")
        == DESIGN_ID
        and config.get("status") == "design_only_not_confirmatory_not_frozen"
        and config.get("decision_scope")
        == "authenticated_v3_n60_development_evidence_only_reserved_validation_closed"
        and config.get("reserved_validation_policy")
        == "forbidden_and_not_opened_by_any_design_screen_result",
        "V4 design-screen identity or closed-validation status changed",
    )
    prior_design_result = _mapping(
        config.get("prior_design_result"), "prior design-result binding"
    )
    _require(
        dict(prior_design_result) == EXPECTED_PRIOR_DESIGN_RESULT,
        "prior failed design-result binding changed",
    )
    design_data = _mapping(config.get("design_data"), "design-data binding")
    expected_design_data = {
        "role": "architecture_and_protocol_design_only",
        "independent_v4_development_evidence": False,
        "fresh_disjoint_nonreserved_confirmation_required": True,
        "v3_development_stop_decision_path": (
            "validation/swe-task-state-v3-development-decision.json"
        ),
        "v3_development_stop_decision_sha256": (
            "2eccd406f7963aef83b764b0bd4d94d93d18e6c16c7fc7ecc24d7686e872711d"
        ),
        "v3_analyzer_sha256": EVALUATOR.V3_ANALYZER_SHA256,
        "v3_protocol_sha256": (
            "9d8b0a7d5c45dc192365429af27c6193de752cc160458eff8e21807d37662b1d"
        ),
        "v3_prompt_bundle_sha256": (
            "17f664b3029220458ff62b8e80a90ec5e796f0372217f02b727d6589df38d3d0"
        ),
        "v3_prompt_summary_sha256": (
            "b70b3751e9cb5a896bc264d2214b425cc7a690ff8026c6c972d3a21cbd22d44d"
        ),
        "v3_public_report_sha256": (
            "7c943132163749f69bd35e4fa2e52bcfee2318fe349fa77603324a37ffaabe46"
        ),
        "v3_replay_merge_manifest_sha256": (
            "302437210d582081e4c343cadb30afecf9e7e0bfb18a8d7b12fdb10c3d782e6f"
        ),
        "v3_materialization_receipt_sha256": (
            "3f7d6dbecb3157badc7e0db09444c8b77bd6f51c8941604b3a5bf81bf30e7812"
        ),
    }
    _require(
        dict(design_data) == expected_design_data,
        "V3 design-data binding changed",
    )
    target = _mapping(config.get("target_contract"), "target contract")
    _require(
        dict(target)
        == {
            "classes_in_order": list(EVALUATOR.CLASSES),
            "temporal_alignment": (
                "ensuing_same_request_completion_observable_action_from_final_"
                "prompt_boundary"
            ),
            "unknown_action": (
                "prediction_emitted_but_excluded_from_fit_selection_and_metrics"
            ),
            "later_request_actions_used_for_target_or_features": False,
        },
        "V4 design target changed",
    )
    features = _mapping(config.get("feature_contract"), "feature contract")
    _require(
        dict(features)
        == {
            "variants_in_order": list(EVALUATOR.BASE_VARIANTS),
            "variant_widths": EVALUATOR.FEATURES.VARIANT_WIDTHS,
            "raw_sensor_width": EVALUATOR.FEATURES.SENSOR_WIDTH,
            "compact_width": EVALUATOR.FEATURES.COMPACT_WIDTH,
            "ema_alpha": EVALUATOR.FEATURES.EMA_ALPHA,
            "difference_order": (
                "subtract_raw_96_wide_vectors_then_compact_each_result"
            ),
            "unknown_stable_rows_update_sensor_state": True,
            "unstable_rows_update_sensor_state": False,
        },
        "V4 design feature identity changed",
    )
    _require(
        dict(_mapping(config.get("model_source"), "model source"))
        == {
            "algorithm": "inherit_exact_normalized_v3_multiseed_extratrees_contract",
            "same_hyperparameters_weights_folds_and_seed_order_for_all_base_variants": True,
        },
        "V4 model-source semantics changed",
    )
    evaluator_sections = _mapping(
        config.get("evaluator_contract"), "evaluator contract"
    )
    _require(
        set(evaluator_sections)
        == {"forecast_pool", "decision", "calibration", "abstention", "metrics"},
        "V4 evaluator section identity changed",
    )
    _require(
        dict(
            _mapping(
                evaluator_sections.get("forecast_pool"),
                "forecast-pool contract",
            )
        )
        == {
            "candidate": "fixed_geometric_log_opinion_pool",
            "reference": "sequence_logit_raw_probability",
            "candidate_logit_weight": 0.80,
            "candidate_logit_j_weight": 0.20,
            "selection": "none_fixed_before_fresh_development_selection",
            "normalization": "log_space_then_exact_row_normalization",
            "shared_action_source": EVALUATOR.REFERENCE_PROCEDURE,
        },
        "V4 fixed forecast-pool semantics changed",
    )
    _require(
        dict(_mapping(config.get("procedures"), "procedures"))
        == {
            "primary": EVALUATOR.PRIMARY_PROCEDURE,
            "reference": EVALUATOR.REFERENCE_PROCEDURE,
            "proper_scores_use": "temperature_calibrated_forecast_q",
            "decision_metrics_use": "argmax_raw_offset_decision_d",
            "decision_confidence": "forecast_q_at_argmax_decision_d",
        },
        "V4 procedure semantics changed",
    )
    point_thresholds = _exact_numeric_mapping(
        config.get("point_screen_thresholds"),
        {
            "known_action_fraction_minimum": 0.95,
            "balanced_accuracy_minimum": 0.75,
            "recall_inspect_minimum": 0.75,
            "recall_edit_minimum": 0.65,
            "recall_check_or_finish_minimum": 0.75,
            "selected_accepted_accuracy_minimum": 0.85,
            "selected_coverage_minimum": 0.7,
            "multiclass_negative_log_likelihood_maximum": 0.6,
        },
        "point-screen thresholds",
    )
    support_thresholds = _exact_numeric_mapping(
        config.get("support_screen_thresholds"),
        {
            "stable_prediction_rows_minimum": 1500,
            "known_action_rows_minimum": 1425,
            "prediction_tasks_minimum": 55,
            "prediction_repositories_minimum": 9,
            "known_action_tasks_minimum": 55,
            "known_action_repositories_minimum": 9,
            "hierarchical_known_action_fraction_minimum": 0.95,
            "known_inspect_tasks_minimum": 10,
            "known_edit_tasks_minimum": 10,
            "known_check_or_finish_tasks_minimum": 10,
            "known_inspect_repositories_minimum": 6,
            "known_edit_repositories_minimum": 6,
            "known_check_or_finish_repositories_minimum": 6,
            "numerical_stability_fraction_minimum": 0.9,
            "stable_feature_complete_prediction_fraction_minimum": 0.9,
        },
        "support-screen thresholds",
    )
    _require(
        dict(_mapping(config.get("output_contract"), "output contract"))
        == {
            "dedicated_root": ".cache/swe_state_interpreter_v4_design",
            "new_outputs_no_clobber": True,
            "validation_or_reserved_output_paths_forbidden": True,
        },
        "V4 output contract changed",
    )
    _require(
        V3.canonical_json_sha256(config) == DESIGN_CONFIG_CANONICAL_SHA256,
        "V4 design-screen canonical content changed",
    )
    return {
        "value": dict(config),
        "prior_design_result": dict(prior_design_result),
        "design_data": dict(design_data),
        "evaluator_sections": dict(evaluator_sections),
        "point_screen_thresholds": point_thresholds,
        "support_screen_thresholds": support_thresholds,
    }


def build_evaluator_contract(
    design: Mapping[str, Any], v3_protocol: Mapping[str, Any]
) -> dict[str, Any]:
    sections = _mapping(design.get("evaluator_sections"), "evaluator sections")
    v3_model = dict(_mapping(v3_protocol.get("model"), "normalized V3 model"))
    # V3's normalizer retains the concrete execution schedule but intentionally
    # omits the descriptive generic-family fields from its returned value.  V4
    # binds both: the exact frozen V3 schedule and an explicit statement of the
    # only generic primitives that are being reused.
    v4_model = {
        **v3_model,
        "family": "fixed_multiseed_ExtraTreesClassifier_probability_ensemble",
        "probability_reduction": (
            "serial_tree_order_within_each_seed_then_arithmetic_mean_across_"
            "seed_estimators_then_floor_and_renormalize"
        ),
        "same_hyperparameters_and_seed_order_for_all_variants": True,
    }
    v3_nested = dict(_mapping(v3_protocol.get("nested"), "normalized V3 nested"))
    v4_nested = {
        **v3_nested,
        "outer_algorithm": "leave_one_repository_out",
        "inner_algorithm": "leave_one_repository_out_within_outer_training",
        "outer_heldout_labels_used_for_fit_calibration_or_threshold_selection": False,
        "same_outer_and_inner_folds_weights_seed_order_and_hyperparameters_across_variants": True,
    }
    return EVALUATOR.validate_contract(
        {
            "model": v4_model,
            "weighting": v3_protocol["weighting"],
            "nested": v4_nested,
            "forecast_pool": sections["forecast_pool"],
            "decision": sections["decision"],
            "calibration": sections["calibration"],
            "abstention": sections["abstention"],
            "metrics": sections["metrics"],
        }
    )


def _point_screen(
    nested: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    results = _mapping(nested.get("results"), "nested results")
    primary_result = _mapping(
        results.get(EVALUATOR.PRIMARY_PROCEDURE), "primary result"
    )
    reference_result = _mapping(
        results.get(EVALUATOR.REFERENCE_PROCEDURE), "reference result"
    )
    primary = _mapping(primary_result.get("metrics"), "primary metrics")
    reference = _mapping(reference_result.get("metrics"), "reference metrics")
    comparisons = (
        ("known_action_fraction", "known_action_fraction_minimum", ">="),
        ("balanced_accuracy", "balanced_accuracy_minimum", ">="),
        ("recall_inspect", "recall_inspect_minimum", ">="),
        ("recall_edit", "recall_edit_minimum", ">="),
        ("recall_check_or_finish", "recall_check_or_finish_minimum", ">="),
        (
            "selected_accepted_accuracy",
            "selected_accepted_accuracy_minimum",
            ">=",
        ),
        ("selected_coverage", "selected_coverage_minimum", ">="),
        (
            "multiclass_negative_log_likelihood",
            "multiclass_negative_log_likelihood_maximum",
            "<=",
        ),
    )
    checks: list[dict[str, Any]] = []
    for metric, threshold_id, operator in comparisons:
        observed = primary.get(metric)
        threshold = thresholds.get(threshold_id)
        _require(
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and math.isfinite(float(observed))
            and isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and math.isfinite(float(threshold)),
            f"point screen lacks numeric {metric}",
        )
        passed = (
            float(observed) >= float(threshold)
            if operator == ">="
            else float(observed) <= float(threshold)
        )
        checks.append(
            {
                "metric": metric,
                "observed": float(observed),
                "operator": operator,
                "threshold": float(threshold),
                "passed": bool(passed),
            }
        )
    full_settings = _mapping(
        nested.get("full_development_settings"), "full settings"
    )
    full = _mapping(
        full_settings.get(EVALUATOR.PRIMARY_PROCEDURE),
        "full primary settings",
    )
    full_reference = _mapping(
        full_settings.get(EVALUATOR.REFERENCE_PROCEDURE),
        "full reference settings",
    )
    decision_selected = _mapping(
        full.get("decision"), "full primary decision"
    ).get("selected_under_floors")
    threshold_selected = _mapping(
        full.get("abstention"), "full primary abstention"
    ).get("selected_under_floors")
    reference_threshold_selected = _mapping(
        full_reference.get("abstention"), "full reference abstention"
    ).get("selected_under_floors")
    _require(
        type(decision_selected) is bool
        and type(threshold_selected) is bool
        and type(reference_threshold_selected) is bool,
        "full candidate/reference floor-selection flags must be booleans",
    )
    primary_raw_r_hash = primary_result.get("decision_raw_r_float64_sha256")
    reference_raw_r_hash = reference_result.get("decision_raw_r_float64_sha256")
    primary_d_hash = primary_result.get("decision_d_float64_sha256")
    reference_d_hash = reference_result.get("decision_d_float64_sha256")
    nested_raw_r_identity = (
        nested.get("candidate_reference_decision_raw_r_exactly_equal") is True
        and isinstance(primary_raw_r_hash, str)
        and bool(primary_raw_r_hash)
        and primary_raw_r_hash
        == reference_raw_r_hash
        == nested.get("shared_decision_raw_r_float64_sha256")
    )
    nested_d_identity = (
        nested.get("candidate_reference_decision_d_exactly_equal") is True
        and isinstance(primary_d_hash, str)
        and bool(primary_d_hash)
        and primary_d_hash
        == reference_d_hash
        == nested.get("shared_decision_d_float64_sha256")
    )

    folds = nested.get("folds")
    _require(isinstance(folds, list) and bool(folds), "nested folds are missing")
    fold_raw_r_identity_checks: list[bool] = []
    fold_d_identity_checks: list[bool] = []
    fold_settings_identity_checks: list[bool] = []
    for fold_index, fold_value in enumerate(folds):
        fold = _mapping(fold_value, f"outer fold {fold_index}")
        shared = _mapping(
            fold.get("shared_action_settings"),
            f"outer fold {fold_index} shared action settings",
        )
        settings = _mapping(
            fold.get("settings"), f"outer fold {fold_index} settings"
        )
        primary_settings = _mapping(
            settings.get(EVALUATOR.PRIMARY_PROCEDURE),
            f"outer fold {fold_index} primary settings",
        )
        reference_settings = _mapping(
            settings.get(EVALUATOR.REFERENCE_PROCEDURE),
            f"outer fold {fold_index} reference settings",
        )
        shared_hash = V3.canonical_json_sha256(shared)
        fold_raw_r_identity_checks.append(
            fold.get("candidate_reference_decision_raw_r_exactly_equal") is True
        )
        fold_d_identity_checks.append(
            fold.get("candidate_reference_decision_d_exactly_equal") is True
        )
        fold_settings_identity_checks.append(
            fold.get("shared_action_settings_sha256") == shared_hash
            and primary_settings.get("decision")
            == reference_settings.get("decision")
            == shared
            and primary_settings.get("shared_action_settings_sha256")
            == reference_settings.get("shared_action_settings_sha256")
            == shared_hash
        )

    full_shared = _mapping(
        nested.get("full_development_shared_action_settings"),
        "full-development shared action settings",
    )
    full_shared_hash = V3.canonical_json_sha256(full_shared)
    full_identity = (
        nested.get("full_development_shared_action_settings_sha256")
        == full_shared_hash
        and full.get("decision") == full_reference.get("decision") == full_shared
        and full.get("shared_action_settings_sha256")
        == full_reference.get("shared_action_settings_sha256")
        == full_shared_hash
    )
    full_promotion = _mapping(
        nested.get("full_development_promotion"),
        "full-development promotion",
    )
    shared_action_metrics = (
        "accuracy",
        "balanced_accuracy",
        "recall_inspect",
        "recall_edit",
        "recall_check_or_finish",
    )
    for metric in (
        *shared_action_metrics,
        "multiclass_negative_log_likelihood",
        "multiclass_brier",
    ):
        for role, metric_map in (("primary", primary), ("reference", reference)):
            observed = metric_map.get(metric)
            _require(
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and math.isfinite(float(observed)),
                f"{role} point metrics lack numeric {metric}",
            )
    action_metric_identity = all(
        float(primary[metric]) == float(reference[metric])
        for metric in shared_action_metrics
    )
    selection_checks = {
        "full_primary_decision_selected_under_floors": decision_selected is True,
        "full_candidate_threshold_selected_under_floors": threshold_selected is True,
        "full_reference_threshold_selected_under_floors": (
            reference_threshold_selected is True
        ),
        "nested_candidate_reference_shared_action_policy": (
            nested.get("candidate_reference_shared_action_policy") is True
        ),
        "nested_candidate_reference_decision_raw_r_exact_identity": (
            nested_raw_r_identity
        ),
        "nested_candidate_reference_decision_d_exact_identity": nested_d_identity,
        "every_outer_fold_decision_raw_r_exact_identity": all(
            fold_raw_r_identity_checks
        ),
        "every_outer_fold_decision_d_exact_identity": all(
            fold_d_identity_checks
        ),
        "every_outer_fold_shared_action_settings_exact_identity": all(
            fold_settings_identity_checks
        ),
        "full_development_shared_action_settings_exact_identity": full_identity,
        "full_development_promotion_shared_action_identity": (
            full_promotion.get(
                "candidate_reference_shared_action_identity_passed"
            )
            is True
        ),
        "full_development_promotion_floor_flags_exact_identity": (
            full_promotion.get("action_rule_selected_under_floors")
            is decision_selected
            and full_promotion.get(
                "candidate_abstention_selected_under_floors"
            )
            is threshold_selected
            and full_promotion.get(
                "reference_abstention_selected_under_floors"
            )
            is reference_threshold_selected
            and full_promotion.get(
                "both_abstention_branches_selected_under_floors"
            )
            is (threshold_selected and reference_threshold_selected)
            and full_promotion.get("fallback_blocks_promotion")
            is not (
                decision_selected
                and threshold_selected
                and reference_threshold_selected
            )
            and full_promotion.get("eligible_on_full_development_selection")
            is (
                decision_selected
                and threshold_selected
                and reference_threshold_selected
            )
        ),
        "candidate_reference_shared_action_point_metrics_exact_identity": (
            action_metric_identity
        ),
    }
    paired_metrics = (
        "accuracy",
        "balanced_accuracy",
        "multiclass_negative_log_likelihood",
        "multiclass_brier",
    )
    paired = {
        metric: float(primary[metric]) - float(reference[metric])
        for metric in paired_metrics
    }
    return {
        "role": "design_screen_only_not_a_reliability_gate_result",
        "absolute_point_checks": checks,
        "full_development_selection_checks": selection_checks,
        "candidate_minus_reference_point_differences": paired,
        "all_point_and_full_selection_checks_passed": (
            all(item["passed"] for item in checks)
            and all(selection_checks.values())
        ),
        "bootstrap_intervals_available": False,
        "confirmatory_interpretation_forbidden": True,
    }


def _support_screen(
    support: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    comparisons = (
        ("stable_prediction_rows", "stable_prediction_rows_minimum"),
        ("known_action_rows", "known_action_rows_minimum"),
        ("prediction_tasks", "prediction_tasks_minimum"),
        ("prediction_repositories", "prediction_repositories_minimum"),
        ("known_action_tasks", "known_action_tasks_minimum"),
        ("known_action_repositories", "known_action_repositories_minimum"),
        (
            "hierarchical_known_action_fraction",
            "hierarchical_known_action_fraction_minimum",
        ),
        ("known_inspect_tasks", "known_inspect_tasks_minimum"),
        ("known_edit_tasks", "known_edit_tasks_minimum"),
        (
            "known_check_or_finish_tasks",
            "known_check_or_finish_tasks_minimum",
        ),
        (
            "known_inspect_repositories",
            "known_inspect_repositories_minimum",
        ),
        ("known_edit_repositories", "known_edit_repositories_minimum"),
        (
            "known_check_or_finish_repositories",
            "known_check_or_finish_repositories_minimum",
        ),
        (
            "numerical_stability_fraction",
            "numerical_stability_fraction_minimum",
        ),
        (
            "stable_feature_complete_prediction_fraction",
            "stable_feature_complete_prediction_fraction_minimum",
        ),
    )
    checks: list[dict[str, Any]] = []
    for metric, threshold_id in comparisons:
        observed = support.get(metric)
        threshold = thresholds.get(threshold_id)
        _require(
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and math.isfinite(float(observed))
            and isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and math.isfinite(float(threshold)),
            f"support screen lacks numeric {metric}",
        )
        checks.append(
            {
                "metric": metric,
                "observed": observed,
                "operator": ">=",
                "threshold": threshold,
                "passed": bool(observed >= threshold),
            }
        )
    return {
        "role": "design_support_screen_only_not_confirmatory",
        "checks": checks,
        "all_support_checks_passed": all(item["passed"] for item in checks),
        "confirmatory_interpretation_forbidden": True,
    }


def _validate_prior_design_result(
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    configured_path = (ROOT / str(binding.get("path"))).resolve(strict=True)
    _require(
        configured_path == PRIOR_DESIGN_RESULT.resolve(strict=True)
        and PRIOR_DESIGN_RESULT.is_file()
        and not PRIOR_DESIGN_RESULT.is_symlink(),
        "prior design result must use the exact canonical regular-file path",
    )
    _require(
        V3.sha256_file(PRIOR_DESIGN_RESULT) == binding.get("sha256"),
        "prior failed design-result bytes changed",
    )
    prior = _mapping(
        _read_json(PRIOR_DESIGN_RESULT, "prior failed design result"),
        "prior failed design result",
    )
    prior_point_screen = _mapping(
        prior.get("point_screen"), "prior failed point screen"
    )
    absolute_checks = prior_point_screen.get("absolute_point_checks")
    _require(
        isinstance(absolute_checks, list),
        "prior failed point checks are missing",
    )
    observed_failures = [
        {
            "metric": item.get("metric"),
            "observed": item.get("observed"),
            "operator": item.get("operator"),
            "threshold": item.get("threshold"),
        }
        for item in absolute_checks
        if isinstance(item, Mapping) and item.get("passed") is False
    ]
    _require(
        prior.get("status") == binding.get("status")
        and prior_point_screen.get("confirmatory_interpretation_forbidden")
        == binding.get("confirmatory_interpretation_forbidden")
        and prior.get("all_design_support_point_and_selection_checks_passed")
        == binding.get("all_design_support_point_and_selection_checks_passed")
        and observed_failures == binding.get("point_failures"),
        "prior failed design-result status or point failures changed",
    )
    return {
        **dict(binding),
        "canonical_path": str(configured_path),
        "exact_bytes_and_failure_status_verified": True,
    }


def prepare_inputs(args: argparse.Namespace) -> dict[str, Any]:
    design_path = args.design_config.resolve(strict=True)
    _require(
        design_path == DEFAULT_DESIGN_CONFIG.resolve(strict=True)
        and V3.sha256_file(args.design_config) == DESIGN_CONFIG_SHA256,
        "design run requires the exact canonical V4 design-screen config",
    )
    design = validate_design_config(
        _read_json(args.design_config, "V4 design-screen config")
    )
    prior_design_result_binding = _validate_prior_design_result(
        design["prior_design_result"]
    )
    _require(
        V3.sha256_file(V3_STOP_DECISION)
        == design["design_data"]["v3_development_stop_decision_sha256"],
        "V3 stop-decision bytes changed",
    )
    _require(
        V3.sha256_file(EVALUATOR.V3_ANALYZER_PATH)
        == design["design_data"]["v3_analyzer_sha256"],
        "V3 analyzer bytes changed",
    )
    _require(
        args.v3_protocol.resolve(strict=True) == DEFAULT_V3_PROTOCOL.resolve(strict=True)
        and V3.sha256_file(args.v3_protocol)
        == design["design_data"]["v3_protocol_sha256"],
        "V3 protocol path or bytes changed",
    )
    action_value = _read_json(args.action_protocol, "V3 action protocol")
    v3_protocol = V3.validate_protocol(
        _read_json(args.v3_protocol, "V3 protocol"),
        action_protocol_value=action_value,
    )
    evaluator_contract = build_evaluator_contract(design, v3_protocol)

    development_binding = V3.validate_development_bundle(
        cohort_path=args.development_cohort,
        prompts_path=args.prompts,
        summary_path=args.prompts_summary,
    )
    replay_binding = V3.validate_replay_merge_receipt(
        receipt_path=args.replay_merge_receipt,
        public_report_path=args.public_report,
        prompts_path=args.prompts,
        summary_path=args.prompts_summary,
    )
    observed_hashes = {
        "prompts": V3.sha256_file(args.prompts),
        "prompts_summary": V3.sha256_file(args.prompts_summary),
        "public_report": V3.sha256_file(args.public_report),
        "replay_merge_receipt": V3.sha256_file(args.replay_merge_receipt),
    }
    expected_hashes = {
        "prompts": design["design_data"]["v3_prompt_bundle_sha256"],
        "prompts_summary": design["design_data"]["v3_prompt_summary_sha256"],
        "public_report": design["design_data"]["v3_public_report_sha256"],
        "replay_merge_receipt": design["design_data"][
            "v3_replay_merge_manifest_sha256"
        ],
    }
    _require(observed_hashes == expected_hashes, "V3 design evidence bytes changed")
    _require(
        replay_binding.get("materialization_receipt_sha256")
        == design["design_data"]["v3_materialization_receipt_sha256"],
        "V3 materialization receipt binding changed",
    )
    extracted = EXTRACT.extract_stable_rows_streaming(
        args.prompts, args.public_report, protocol=v3_protocol
    )
    _require(
        {
            "prompts": V3.sha256_file(args.prompts),
            "prompts_summary": V3.sha256_file(args.prompts_summary),
            "public_report": V3.sha256_file(args.public_report),
            "replay_merge_receipt": V3.sha256_file(args.replay_merge_receipt),
        }
        == observed_hashes,
        "V3 design evidence changed while it was being consumed",
    )
    return {
        "design": design,
        "v3_protocol": v3_protocol,
        "evaluator_contract": evaluator_contract,
        "rows": extracted["rows"],
        "eligibility": extracted["eligibility"],
        "development_binding": development_binding,
        "replay_binding": replay_binding,
        "prior_design_result_binding": prior_design_result_binding,
        "hashes": {
            **observed_hashes,
            "design_config": V3.sha256_file(args.design_config),
            "prior_design_result": V3.sha256_file(PRIOR_DESIGN_RESULT),
            "v3_protocol": V3.sha256_file(args.v3_protocol),
            "action_protocol": V3.sha256_file(args.action_protocol),
            "v3_analyzer": V3.sha256_file(EVALUATOR.V3_ANALYZER_PATH),
            "v4_design_analyzer": V3.sha256_file(Path(__file__).resolve()),
            "v4_extractor": V3.sha256_file(Path(EXTRACT.__file__).resolve()),
            "v4_features": V3.sha256_file(
                Path(EVALUATOR.FEATURES.__file__).resolve()
            ),
            "v4_evaluator": V3.sha256_file(Path(EVALUATOR.__file__).resolve()),
            "v4_decision": V3.sha256_file(
                Path(EVALUATOR.DECISION.__file__).resolve()
            ),
            "v4_calibration": V3.sha256_file(
                Path(EVALUATOR.CALIBRATION.__file__).resolve()
            ),
            "v4_metrics": V3.sha256_file(
                Path(EVALUATOR.METRICS.__file__).resolve()
            ),
            "v3_stop_decision": V3.sha256_file(V3_STOP_DECISION),
        },
    }


def analyze_command(args: argparse.Namespace) -> int:
    initial_path_validation = validate_cli_paths(args)
    prepared = prepare_inputs(args)
    rows = prepared["rows"]
    nested = EVALUATOR.nested_leave_one_repository_out(
        rows, contract=prepared["evaluator_contract"]
    )
    support = V3.support_summary(rows, prepared["eligibility"])
    support_screen = _support_screen(
        support, prepared["design"]["support_screen_thresholds"]
    )
    point_screen = _point_screen(
        nested, prepared["design"]["point_screen_thresholds"]
    )
    all_design_checks_passed = bool(
        support_screen["all_support_checks_passed"]
        and point_screen["all_point_and_full_selection_checks_passed"]
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "id": DESIGN_ID,
        "status": "completed_design_screen_not_confirmatory",
        "scope": "authenticated_v3_n60_design_only_reserved_validation_closed",
        "path_validation": initial_path_validation,
        "inputs": prepared["hashes"],
        "prior_design_result_binding": prepared[
            "prior_design_result_binding"
        ],
        "development_data_binding": prepared["development_binding"],
        "replay_merge_binding": prepared["replay_binding"],
        "design_contract": prepared["design"]["value"],
        "eligibility": prepared["eligibility"],
        "support": support,
        "support_screen": support_screen,
        "nested_design_evaluation": nested,
        "point_screen": point_screen,
        "all_design_support_point_and_selection_checks_passed": (
            all_design_checks_passed
        ),
        "model_refit_bootstrap": {
            "status": "forbidden_for_v3_design_screen",
            "samples": 0,
            "reason": "V3 N60 is design-only and cannot become V4 confirmatory evidence",
        },
        "operational_reliability_claim": False,
        "independent_v4_development_result": False,
        "reserved_validation_accessed": False,
        "reserved_validation_allowed": False,
        "fresh_disjoint_nonreserved_development_confirmation_required": True,
    }
    _require(
        validate_cli_paths(args) == initial_path_validation,
        "design CLI paths changed before output creation",
    )
    V3.atomic_write_json_no_clobber(args.output, output)
    primary_metrics = nested["results"][EVALUATOR.PRIMARY_PROCEDURE]["metrics"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "stable_predictions": len(rows),
                "known_metric_rows": len(EVALUATOR.known_rows(rows)),
                "primary_balanced_accuracy": primary_metrics["balanced_accuracy"],
                "primary_edit_recall": primary_metrics["recall_edit"],
                "primary_accepted_accuracy": primary_metrics[
                    "selected_accepted_accuracy"
                ],
                "point_screen_passed": point_screen[
                    "all_point_and_full_selection_checks_passed"
                ],
                "support_screen_passed": support_screen[
                    "all_support_checks_passed"
                ],
                "all_design_checks_passed": all_design_checks_passed,
                "confirmatory_interpretation_forbidden": True,
                "reserved_validation_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_no_symlink_components(path: Path, label: str) -> None:
    cursor = path.expanduser().absolute()
    while True:
        if cursor.exists() or cursor.is_symlink():
            _require(not cursor.is_symlink(), f"{label} traverses a symlink: {cursor}")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent


def validate_cli_paths(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [
        args.design_config,
        args.v3_protocol,
        args.action_protocol,
        args.development_cohort,
        args.prompts,
        args.prompts_summary,
        args.public_report,
        args.replay_merge_receipt,
    ]
    canonical_inputs = [
        DEFAULT_DESIGN_CONFIG,
        DEFAULT_V3_PROTOCOL,
        DEFAULT_ACTION_PROTOCOL,
        DEFAULT_COHORT,
        DEFAULT_PROMPTS,
        DEFAULT_PROMPTS_SUMMARY,
        DEFAULT_PUBLIC_REPORT,
        DEFAULT_REPLAY_RECEIPT,
    ]
    for path in inputs:
        _require(path.is_file() and not path.is_symlink(), f"unsafe input: {path}")
        _require_no_symlink_components(path, "input path")
    resolved_inputs = [path.resolve(strict=True) for path in inputs]
    resolved_canonical_inputs = [path.resolve(strict=True) for path in canonical_inputs]
    _require(
        resolved_inputs == resolved_canonical_inputs,
        "design screen accepts only the exact canonical V3 input paths",
    )
    _require(len(resolved_inputs) == len(set(resolved_inputs)), "input paths alias")
    input_inodes = [(path.stat().st_dev, path.stat().st_ino) for path in inputs]
    _require(len(input_inodes) == len(set(input_inodes)), "input hard links alias")

    _require_no_symlink_components(args.output.parent, "output parent")
    output = args.output.expanduser().resolve(strict=False)
    root = DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    _require(
        _is_relative_to(output, root),
        "design output must remain beneath the dedicated V4 design root",
    )
    relative_parts = output.relative_to(root).parts
    _require(
        bool(relative_parts)
        and output.suffix == ".json"
        and not any(
            "reserved" in part.lower() or "validation" in part.lower()
            for part in relative_parts
        )
        and not _is_relative_to(output, (ROOT / "validation").resolve()),
        "reserved and validation output paths are forbidden",
    )
    _require(
        output not in set(resolved_inputs)
        and not args.output.exists()
        and not args.output.is_symlink(),
        "design output aliases an input or already exists",
    )
    args.output = output
    return {
        "dedicated_output_root": str(root),
        "resolved_output": str(output),
        "canonical_input_paths_required": True,
        "canonical_paths_distinct": True,
        "new_output_no_clobber": True,
        "reserved_and_validation_outputs_forbidden": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-config", type=Path, default=DEFAULT_DESIGN_CONFIG)
    parser.add_argument("--v3-protocol", type=Path, default=DEFAULT_V3_PROTOCOL)
    parser.add_argument("--action-protocol", type=Path, default=DEFAULT_ACTION_PROTOCOL)
    parser.add_argument("--development-cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompts-summary", type=Path, default=DEFAULT_PROMPTS_SUMMARY)
    parser.add_argument("--public-report", type=Path, default=DEFAULT_PUBLIC_REPORT)
    parser.add_argument(
        "--replay-merge-receipt", type=Path, default=DEFAULT_REPLAY_RECEIPT
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=analyze_command)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
