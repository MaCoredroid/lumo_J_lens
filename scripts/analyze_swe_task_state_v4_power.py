#!/usr/bin/env python3
"""Run the nonconfirmatory V4 paired proper-score power screen.

The only data source is one exact, contract-closed V3-N60 design artifact.
Task identifiers are used solely as opaque grouping keys.  Task text, prompts,
completions, patches, trajectories, and reserved validation are outside this
program's input contract.  All results condition on frozen out-of-fold
predictions and therefore are not full model-refit power.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from scripts import swe_task_state_v4_calibration as CALIBRATION
except ModuleNotFoundError as error:  # Support direct execution from scripts/.
    if error.name != "scripts":
        raise
    import swe_task_state_v4_calibration as CALIBRATION  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/swe_task_state_v4_power_screen.json"
DEFAULT_SOURCE_ARTIFACT = (
    ROOT
    / ".cache/swe_state_interpreter_v4_design"
    / "v3-n60-geometric-a020-shared-action-contract-closed.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / ".cache/swe_state_interpreter_v4_design"
DEFAULT_OUTPUT = (
    DEFAULT_OUTPUT_ROOT
    / "v3-n60-geometric-a020-paired-proper-power-screen.json"
)

SCHEMA_VERSION = 1
SCREEN_ID = "swe-task-state-v4-paired-proper-power-screen-v1"
OUTPUT_ID = "swe-task-state-v4-paired-proper-power-result-v1"
POWER_CONFIG_SHA256 = (
    "d696ca7edd58be17abd29cc4908322d4741a54fee2d90c89a7e05dcf61fa6574"
)
SOURCE_ARTIFACT_SHA256 = (
    "e1436a9c7fe763a4fab6ae505af17395a4290faefd0734b230d22127bff0cf8c"
)
CLASSES = ("inspect", "edit", "check_or_finish")
CLASS_INDEX = {class_id: index for index, class_id in enumerate(CLASSES)}
PRIMARY_PROCEDURE = "j_forecast_geometric_pool_logit_policy"
REFERENCE_PROCEDURE = "sequence_logit"
FLASK_REPOSITORY = "pallets/flask"
DJANGO_REPOSITORY = "django/django"
PROBABILITY_TOLERANCE = 1e-12

EXPECTED_SOURCE_REPOSITORIES = (
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "psf/requests",
    "pydata/xarray",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
)
EXPECTED_ALLOCATION = OrderedDict(
    (
        ("astropy/astropy", 7),
        ("django/django", 177),
        ("matplotlib/matplotlib", 16),
        ("pallets/flask", 1),
        ("pydata/xarray", 5),
        ("pytest-dev/pytest", 2),
        ("scikit-learn/scikit-learn", 16),
        ("sphinx-doc/sphinx", 23),
        ("sympy/sympy", 39),
    )
)
SCENARIO_ORDER = (
    "django_domain_proxy",
    "exchangeable_repository_task_proxy",
    "neutral_zero_effect",
    "empirical_marginal_p90_adverse",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_)),
        f"{label} must be an integer",
    )
    result = int(value)
    _require(result >= minimum, f"{label} is below {minimum}")
    return result


def _strict_float(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    _require(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_)),
        f"{label} must be numeric and non-boolean",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    if minimum is not None:
        _require(result >= minimum, f"{label} is below {minimum}")
    if maximum is not None:
        _require(result <= maximum, f"{label} exceeds {maximum}")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float64_sha256(values: Any) -> str:
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _read_json(path: Path, label: str) -> Any:
    _require(
        path.is_file() and not path.is_symlink(),
        f"{label} must be a regular non-symlink file",
    )
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {path}: {error}") from error


def _probability_row(value: Any, label: str) -> np.ndarray:
    observed = _mapping(value, label)
    _require(
        set(observed) == set(CLASSES),
        f"{label} must name every frozen class exactly",
    )
    raw = [observed[class_id] for class_id in CLASSES]
    _require(
        all(
            isinstance(item, (int, float, np.integer, np.floating))
            and not isinstance(item, (bool, np.bool_))
            for item in raw
        ),
        f"{label} must contain numeric non-boolean values",
    )
    result = np.asarray(raw, dtype=np.float64)
    _require(
        np.all(np.isfinite(result)) and np.all(result > 0.0),
        f"{label} must contain finite strictly positive probabilities",
    )
    _require(
        math.isclose(
            float(result.sum(dtype=np.float64)),
            1.0,
            rel_tol=PROBABILITY_TOLERANCE,
            abs_tol=PROBABILITY_TOLERANCE,
        ),
        f"{label} must sum to one",
    )
    return result


def validate_power_config(value: Any) -> dict[str, Any]:
    config = _mapping(value, "V4 power config")
    _require(
        set(config)
        == {
            "schema_version",
            "id",
            "status",
            "source_artifact",
            "future_allocation_in_order",
            "loss_contract",
            "resampling_contract",
            "flask_scenarios_in_order",
            "decision_rule",
            "temperature_reselection_sensitivity",
            "output_contract",
            "data_policy",
        },
        "V4 power config top-level fields changed",
    )
    _require(
        config.get("schema_version") == SCHEMA_VERSION
        and config.get("id") == SCREEN_ID
        and config.get("status")
        == "design_only_nonconfirmatory_fixed_prediction_power",
        "V4 power config identity changed",
    )

    source = _mapping(config.get("source_artifact"), "source-artifact binding")
    _require(
        source.get("canonical_path")
        == (
            ".cache/swe_state_interpreter_v4_design/"
            "v3-n60-geometric-a020-shared-action-contract-closed.json"
        )
        and source.get("sha256") == SOURCE_ARTIFACT_SHA256
        and source.get("required_status")
        == "completed_design_screen_not_confirmatory"
        and source.get("required_scope")
        == "authenticated_v3_n60_design_only_reserved_validation_closed"
        and source.get("required_reserved_validation_accessed") is False
        and source.get("required_reserved_validation_allowed") is False
        and source.get("required_operational_reliability_claim") is False
        and source.get("required_independent_v4_development_result") is False
        and source.get("required_fresh_disjoint_confirmation") is True
        and source.get("candidate_procedure") == PRIMARY_PROCEDURE
        and source.get("reference_procedure") == REFERENCE_PROCEDURE
        and source.get("classes_in_order") == list(CLASSES)
        and source.get("known_metric_rows") == 1570
        and source.get("prediction_rows") == 1606,
        "source-artifact identity or closed-validation binding changed",
    )
    for key in (
        "known_ordered_row_identity_sha256",
        "ordered_row_identity_sha256",
        "evaluator_contract_sha256",
        "candidate_forecast_q_float64_sha256",
        "reference_forecast_q_float64_sha256",
        "shared_decision_d_float64_sha256",
        "shared_decision_raw_r_float64_sha256",
        "candidate_forecast_raw_p_float64_sha256",
        "reference_forecast_raw_p_float64_sha256",
    ):
        item = source.get(key)
        _require(
            isinstance(item, str)
            and len(item) == 64
            and set(item) <= set("0123456789abcdef"),
            f"source-artifact hash binding is invalid: {key}",
        )

    allocation = _mapping(
        config.get("future_allocation_in_order"), "future allocation"
    )
    _require(
        list(allocation) == list(EXPECTED_ALLOCATION)
        and dict(allocation) == dict(EXPECTED_ALLOCATION)
        and sum(
            _strict_int(value, f"future allocation {repo}", minimum=1)
            for repo, value in allocation.items()
        )
        == 286,
        "exact future allocation changed",
    )

    loss = _mapping(config.get("loss_contract"), "loss contract")
    _require(
        dict(loss)
        == {
            "difference_direction": "candidate_minus_reference",
            "nll_row_difference": (
                "-log(q_candidate_true)+log(q_reference_true)"
            ),
            "brier_row_difference": (
                "sum_k((q_candidate_k-one_hot_k)^2)-"
                "sum_k((q_reference_k-one_hot_k)^2)"
            ),
            "task_profile": "equal_mean_of_known_rows_within_whole_task",
            "point_estimand": (
                "equal_repository_then_equal_known_task_within_repository_"
                "then_equal_known_row_within_task"
            ),
            "paired_rows_required": True,
        },
        "paired loss contract changed",
    )

    resampling = _mapping(
        config.get("resampling_contract"), "resampling contract"
    )
    _require(
        resampling.get("numpy_version") == "2.5.1"
        and resampling.get("bit_generator") == "PCG64"
        and resampling.get("seed") == 20260727
        and resampling.get("outer_cohorts") == 10000
        and resampling.get("outer_batch_size") == 10
        and resampling.get("outer_task_resampling")
        == "whole_task_profiles_with_replacement_within_same_repository"
        and resampling.get("inner_bootstrap_draws") == 1000
        and resampling.get("every_sampled_task_retained_in_every_inner_draw")
        is True
        and resampling.get("candidate_reference_use_same_draws") is True
        and resampling.get("upper_quantile") == 0.975
        and resampling.get("quantile_method") == "linear",
        "fixed resampling identity changed",
    )
    for key, scope in (
        ("repository_weight_distribution", "repositories"),
        ("task_weight_distribution", "tasks"),
    ):
        distribution = _mapping(resampling.get(key), key)
        expected_normalization = (
            "divide_by_sum_across_repositories"
            if scope == "repositories"
            else "divide_by_sum_within_repository"
        )
        _require(
            dict(distribution)
            == {
                "family": "Gamma",
                "shape": 1.0,
                "scale": 1.0,
                "normalization": expected_normalization,
            },
            f"{key} changed",
        )

    scenarios = _mapping(
        config.get("flask_scenarios_in_order"), "Flask scenarios"
    )
    _require(
        list(scenarios) == list(SCENARIO_ORDER),
        "Flask scenario order changed",
    )
    _require(
        dict(_mapping(scenarios["django_domain_proxy"], "Django proxy"))
        == {
            "kind": "sample_one_whole_task_profile",
            "source_repository": DJANGO_REPOSITORY,
        }
        and dict(
            _mapping(
                scenarios["exchangeable_repository_task_proxy"],
                "exchangeable proxy",
            )
        )
        == {
            "kind": "sample_repository_uniformly_then_one_task_uniformly",
            "repository_source": "all_ten_frozen_source_repositories",
        }
        and dict(_mapping(scenarios["neutral_zero_effect"], "neutral proxy"))
        == {
            "kind": "fixed_paired_task_profile",
            "nll_difference": 0.0,
            "brier_difference": 0.0,
        }
        and dict(
            _mapping(
                scenarios["empirical_marginal_p90_adverse"],
                "p90 adverse proxy",
            )
        )
        == {
            "kind": "fixed_empirical_marginal_quantile_profile",
            "quantile": 0.9,
            "source": "all_sixty_frozen_task_profiles",
            "quantile_method": "linear",
        },
        "Flask scenario semantics changed",
    )

    decision = _mapping(config.get("decision_rule"), "decision rule")
    _require(
        dict(decision)
        == {
            "nll_gate": "bootstrap_upper_strictly_less_than_zero",
            "brier_gate": "bootstrap_upper_strictly_less_than_zero",
            "cohort_joint_pass": "nll_gate_and_brier_gate",
            "minimum_acceptable_joint_power": 0.8,
            "go_requires_every_declared_flask_scenario_at_or_above_minimum": True,
            "pass_label": "GO",
            "fail_label": "NO_GO",
            "fresh_cohort_selection_or_generation_authorized_by_power_screen": False,
        },
        "power decision rule changed",
    )

    sensitivity = _mapping(
        config.get("temperature_reselection_sensitivity"),
        "temperature sensitivity",
    )
    _require(
        sensitivity.get("enabled") is True
        and sensitivity.get("interpretation")
        == "fixed_oof_raw_probability_sensitivity_not_full_model_refit_power"
        and sensitivity.get("flask_scenario") == "django_domain_proxy"
        and sensitivity.get("seed") == 20260723
        and sensitivity.get("outer_cohorts") == 10000
        and sensitivity.get("outer_batch_size") == 5
        and sensitivity.get("inner_bootstrap_draws") == 1000
        and sensitivity.get("temperature_grid")
        == [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
        and sensitivity.get("fixed_temperature") == 1.0
        and sensitivity.get("selection_objective")
        == "independent_branch_weighted_nll_minimum"
        and sensitivity.get("reference_raw_source") == "sequence_logit"
        and sensitivity.get("candidate_secondary_raw_source")
        == "sequence_logit_j"
        and sensitivity.get("candidate_geometric_secondary_weight") == 0.2
        and sensitivity.get("same_hierarchical_draws_for_both_branches")
        is True
        and sensitivity.get("models_refit") is False,
        "temperature sensitivity contract changed",
    )

    output = _mapping(config.get("output_contract"), "output contract")
    _require(
        dict(output)
        == {
            "dedicated_root": ".cache/swe_state_interpreter_v4_design",
            "default_filename": (
                "v3-n60-geometric-a020-paired-proper-power-screen.json"
            ),
            "new_output_no_clobber": True,
            "forbidden_casefolded_path_components": [
                "reserved",
                "validation",
            ],
        },
        "output contract changed",
    )
    policy = _mapping(config.get("data_policy"), "data policy")
    _require(
        policy.get("input_scope")
        == "exact_contract_closed_nonconfirmatory_design_artifact_only"
        and policy.get("prediction_fields_consumed")
        == [
            "row_id",
            "task_id",
            "repo",
            "label",
            "metric_evaluable",
            "forecast_probabilities_q",
            "decision_probabilities_d",
        ]
        and policy.get(
            "task_text_prompt_completion_patch_or_trajectory_fields_consumed"
        )
        is False
        and policy.get("reserved_validation_accessed") is False
        and policy.get("reserved_validation_allowed") is False
        and policy.get("confirmatory_interpretation_forbidden") is True
        and policy.get("full_model_refit_power_estimated") is False,
        "closed data policy changed",
    )
    return json.loads(json.dumps(config))


def _validate_prediction_pair(
    candidate: Sequence[Any],
    reference: Sequence[Any],
    *,
    source_binding: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    _require(
        len(candidate) == len(reference) == int(source_binding["prediction_rows"]),
        "candidate/reference prediction row count changed",
    )
    candidate_rows: list[Mapping[str, Any]] = []
    reference_rows: list[Mapping[str, Any]] = []
    candidate_q: list[np.ndarray] = []
    reference_q: list[np.ndarray] = []
    candidate_d: list[np.ndarray] = []
    reference_d: list[np.ndarray] = []
    known_count = 0
    for index, (candidate_value, reference_value) in enumerate(
        zip(candidate, reference, strict=True)
    ):
        left = _mapping(candidate_value, f"candidate prediction {index}")
        right = _mapping(reference_value, f"reference prediction {index}")
        for key in (
            "row_id",
            "task_id",
            "repo",
            "label",
            "metric_evaluable",
        ):
            _require(
                left.get(key) == right.get(key),
                f"paired prediction identity differs at row {index}: {key}",
            )
        for key in ("row_id", "task_id", "repo"):
            _require(
                isinstance(left.get(key), str) and bool(left.get(key)),
                f"prediction row {index} lacks opaque identity field {key}",
            )
        evaluable = left.get("metric_evaluable")
        _require(
            isinstance(evaluable, bool),
            f"prediction row {index} metric_evaluable is not boolean",
        )
        if evaluable:
            _require(
                left.get("label") in CLASS_INDEX,
                f"known prediction row {index} lacks a frozen label",
            )
            known_count += 1
        candidate_q.append(
            _probability_row(
                left.get("forecast_probabilities_q"),
                f"candidate q row {index}",
            )
        )
        reference_q.append(
            _probability_row(
                right.get("forecast_probabilities_q"),
                f"reference q row {index}",
            )
        )
        candidate_d.append(
            _probability_row(
                left.get("decision_probabilities_d"),
                f"candidate d row {index}",
            )
        )
        reference_d.append(
            _probability_row(
                right.get("decision_probabilities_d"),
                f"reference d row {index}",
            )
        )
        candidate_rows.append(left)
        reference_rows.append(right)

    _require(
        known_count == int(source_binding["known_metric_rows"]),
        "known metric row count changed",
    )
    candidate_q_array = np.asarray(candidate_q, dtype=np.float64)
    reference_q_array = np.asarray(reference_q, dtype=np.float64)
    candidate_d_array = np.asarray(candidate_d, dtype=np.float64)
    reference_d_array = np.asarray(reference_d, dtype=np.float64)
    _require(
        _float64_sha256(candidate_q_array)
        == source_binding["candidate_forecast_q_float64_sha256"]
        and _float64_sha256(reference_q_array)
        == source_binding["reference_forecast_q_float64_sha256"],
        "candidate/reference q bytes differ from the frozen hashes",
    )
    _require(
        np.array_equal(candidate_d_array, reference_d_array),
        "candidate/reference decision d differs",
    )
    _require(
        _float64_sha256(candidate_d_array)
        == source_binding["shared_decision_d_float64_sha256"]
        and _float64_sha256(reference_d_array)
        == source_binding["shared_decision_d_float64_sha256"],
        "shared decision d bytes differ from the frozen hash",
    )
    return candidate_rows, reference_rows


def validate_source_artifact(
    value: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = _mapping(value, "contract-closed source artifact")
    source = _mapping(config["source_artifact"], "source binding")
    _require(
        artifact.get("schema_version") == 1
        and artifact.get("id")
        == "swe-task-state-interpreter-v4-v3-n60-design-screen-v2"
        and artifact.get("status") == source["required_status"]
        and artifact.get("scope") == source["required_scope"]
        and artifact.get("reserved_validation_accessed") is False
        and artifact.get("reserved_validation_allowed") is False
        and artifact.get("operational_reliability_claim") is False
        and artifact.get("independent_v4_development_result") is False
        and artifact.get(
            "fresh_disjoint_nonreserved_development_confirmation_required"
        )
        is True,
        "source artifact is not the required closed nonconfirmatory result",
    )
    nested = _mapping(
        artifact.get("nested_design_evaluation"), "nested design evaluation"
    )
    _require(
        nested.get("schema_version") == 1
        and nested.get("algorithm")
        == (
            "nested_leave_one_repository_out_v4_fixed_geometric_j_forecast_"
            "shared_logit_action"
        )
        and nested.get("primary_procedure") == PRIMARY_PROCEDURE
        and nested.get("reference_procedure") == REFERENCE_PROCEDURE
        and nested.get("candidate_reference_shared_action_policy") is True
        and nested.get("candidate_reference_decision_d_exactly_equal") is True
        and nested.get("candidate_reference_decision_raw_r_exactly_equal")
        is True
        and nested.get("outer_heldout_labels_used_for_fit_or_selection") is False
        and nested.get("forecast_pool_j_weight") == 0.2
        and nested.get("known_ordered_row_identity_sha256")
        == source["known_ordered_row_identity_sha256"]
        and nested.get("ordered_row_identity_sha256")
        == source["ordered_row_identity_sha256"]
        and nested.get("evaluator_contract_sha256")
        == source["evaluator_contract_sha256"]
        and nested.get("shared_decision_d_float64_sha256")
        == source["shared_decision_d_float64_sha256"]
        and nested.get("shared_decision_raw_r_float64_sha256")
        == source["shared_decision_raw_r_float64_sha256"]
        and nested.get("repositories_in_order")
        == list(EXPECTED_SOURCE_REPOSITORIES),
        "nested candidate/reference identity changed",
    )
    results = _mapping(nested.get("results"), "nested results")
    candidate_result = _mapping(
        results.get(PRIMARY_PROCEDURE), "candidate result"
    )
    reference_result = _mapping(
        results.get(REFERENCE_PROCEDURE), "reference result"
    )
    _require(
        candidate_result.get("forecast_q_float64_sha256")
        == source["candidate_forecast_q_float64_sha256"]
        and reference_result.get("forecast_q_float64_sha256")
        == source["reference_forecast_q_float64_sha256"]
        and candidate_result.get("forecast_raw_p_float64_sha256")
        == source["candidate_forecast_raw_p_float64_sha256"]
        and reference_result.get("forecast_raw_p_float64_sha256")
        == source["reference_forecast_raw_p_float64_sha256"]
        and candidate_result.get("decision_d_float64_sha256")
        == source["shared_decision_d_float64_sha256"]
        and reference_result.get("decision_d_float64_sha256")
        == source["shared_decision_d_float64_sha256"]
        and candidate_result.get("decision_raw_r_float64_sha256")
        == source["shared_decision_raw_r_float64_sha256"]
        and reference_result.get("decision_raw_r_float64_sha256")
        == source["shared_decision_raw_r_float64_sha256"],
        "stored candidate/reference q, p, or d hash binding changed",
    )
    candidate_rows, reference_rows = _validate_prediction_pair(
        candidate_result.get("predictions"),
        reference_result.get("predictions"),
        source_binding=source,
    )
    return {
        "artifact": artifact,
        "nested": nested,
        "candidate_result": candidate_result,
        "reference_result": reference_result,
        "candidate_rows": candidate_rows,
        "reference_rows": reference_rows,
    }


def build_task_loss_profiles(
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _require(
        len(candidate_rows) == len(reference_rows),
        "paired rows differ before task aggregation",
    )
    grouped: OrderedDict[
        tuple[str, str], list[tuple[float, float]]
    ] = OrderedDict()
    known_rows = 0
    for index, (candidate, reference) in enumerate(
        zip(candidate_rows, reference_rows, strict=True)
    ):
        if candidate.get("metric_evaluable") is not True:
            continue
        _require(
            reference.get("metric_evaluable") is True,
            f"reference known-row status differs at row {index}",
        )
        label = str(candidate["label"])
        label_index = CLASS_INDEX[label]
        candidate_q = _probability_row(
            candidate["forecast_probabilities_q"],
            f"candidate known q row {index}",
        )
        reference_q = _probability_row(
            reference["forecast_probabilities_q"],
            f"reference known q row {index}",
        )
        one_hot = np.eye(len(CLASSES), dtype=np.float64)[label_index]
        nll_difference = -math.log(float(candidate_q[label_index])) + math.log(
            float(reference_q[label_index])
        )
        brier_difference = float(
            np.sum((candidate_q - one_hot) ** 2, dtype=np.float64)
            - np.sum((reference_q - one_hot) ** 2, dtype=np.float64)
        )
        key = (str(candidate["repo"]), str(candidate["task_id"]))
        grouped.setdefault(key, []).append((nll_difference, brier_difference))
        known_rows += 1
    _require(bool(grouped), "no known paired task profiles were constructed")

    by_repository: OrderedDict[str, list[np.ndarray]] = OrderedDict()
    for (repository, _opaque_task_id), row_losses in grouped.items():
        profile = np.mean(np.asarray(row_losses, dtype=np.float64), axis=0)
        by_repository.setdefault(repository, []).append(profile)
    profiles = {
        repository: np.asarray(items, dtype=np.float64)
        for repository, items in by_repository.items()
    }
    _require(
        set(profiles) == set(EXPECTED_SOURCE_REPOSITORIES)
        and sum(len(items) for items in profiles.values()) == 60
        and known_rows == 1570,
        "frozen source task/repository support changed",
    )
    repository_effects = {
        repository: np.mean(profiles[repository], axis=0)
        for repository in EXPECTED_SOURCE_REPOSITORIES
    }
    point = np.mean(
        np.asarray(list(repository_effects.values()), dtype=np.float64),
        axis=0,
    )
    all_task_profiles = np.concatenate(
        [profiles[repository] for repository in EXPECTED_SOURCE_REPOSITORIES],
        axis=0,
    )
    p90 = np.quantile(
        all_task_profiles,
        0.9,
        axis=0,
        method="linear",
    )
    summary = {
        "known_row_count": known_rows,
        "task_profile_count": len(all_task_profiles),
        "repository_count": len(profiles),
        "task_profile_count_by_repository": {
            repository: len(profiles[repository])
            for repository in EXPECTED_SOURCE_REPOSITORIES
        },
        "repository_point_differences": {
            repository: {
                "multiclass_negative_log_likelihood": float(
                    repository_effects[repository][0]
                ),
                "multiclass_brier": float(repository_effects[repository][1]),
            }
            for repository in EXPECTED_SOURCE_REPOSITORIES
        },
        "repository_improving_sign_count": {
            "multiclass_negative_log_likelihood": int(
                sum(value[0] < 0.0 for value in repository_effects.values())
            ),
            "multiclass_brier": int(
                sum(value[1] < 0.0 for value in repository_effects.values())
            ),
        },
        "paired_point_differences": {
            "multiclass_negative_log_likelihood": float(point[0]),
            "multiclass_brier": float(point[1]),
        },
        "empirical_marginal_p90_adverse_profile": {
            "multiclass_negative_log_likelihood": float(p90[0]),
            "multiclass_brier": float(p90[1]),
        },
    }
    return profiles, summary


def _gamma_task_weighted_draws(
    sampled_profiles: np.ndarray,
    *,
    rng: np.random.Generator,
    inner_draws: int,
) -> np.ndarray:
    _require(
        sampled_profiles.ndim >= 3
        and sampled_profiles.shape[0] > 0
        and sampled_profiles.shape[1] > 0,
        "sampled task profiles have invalid shape",
    )
    outer_batch, task_count = sampled_profiles.shape[:2]
    gamma = rng.gamma(
        shape=1.0,
        scale=1.0,
        size=(outer_batch, inner_draws, task_count),
    )
    _require(
        np.all(np.isfinite(gamma)) and np.all(gamma > 0.0),
        "task Gamma weights are invalid",
    )
    gamma /= gamma.sum(axis=2, keepdims=True, dtype=np.float64)
    flattened = sampled_profiles.reshape(outer_batch, task_count, -1)
    weighted = np.einsum(
        "sbn,snf->sbf",
        gamma,
        flattened,
        optimize=True,
    )
    return weighted.reshape(
        outer_batch,
        inner_draws,
        *sampled_profiles.shape[2:],
    )


def _power_summary(
    upper: np.ndarray,
    inner_joint_favorable_fraction: np.ndarray,
) -> dict[str, Any]:
    _require(
        upper.ndim == 2 and upper.shape[1] == 2,
        "bootstrap upper array has invalid shape",
    )
    nll_pass = upper[:, 0] < 0.0
    brier_pass = upper[:, 1] < 0.0
    joint_pass = nll_pass & brier_pass
    count = len(upper)

    def gate(values: np.ndarray) -> dict[str, Any]:
        passes = int(values.sum())
        power = passes / count
        standard_error = math.sqrt(power * (1.0 - power) / count)
        return {
            "passing_cohort_count": passes,
            "cohort_count": count,
            "power": power,
            "monte_carlo_standard_error": standard_error,
            "monte_carlo_normal_95_interval": [
                max(0.0, power - 1.96 * standard_error),
                min(1.0, power + 1.96 * standard_error),
            ],
        }

    return {
        "nll_gate": gate(nll_pass),
        "brier_gate": gate(brier_pass),
        "joint_gate": gate(joint_pass),
        "bootstrap_upper_distribution": {
            "median": {
                "multiclass_negative_log_likelihood": float(
                    np.quantile(upper[:, 0], 0.5, method="linear")
                ),
                "multiclass_brier": float(
                    np.quantile(upper[:, 1], 0.5, method="linear")
                ),
            },
            "p90": {
                "multiclass_negative_log_likelihood": float(
                    np.quantile(upper[:, 0], 0.9, method="linear")
                ),
                "multiclass_brier": float(
                    np.quantile(upper[:, 1], 0.9, method="linear")
                ),
            },
        },
        "mean_fraction_of_inner_draws_with_both_favorable_signs": float(
            np.mean(inner_joint_favorable_fraction)
        ),
    }


def simulate_paired_proper_power(
    profiles: Mapping[str, np.ndarray],
    allocation: Mapping[str, int],
    *,
    seed: int,
    outer_cohorts: int,
    inner_draws: int,
    batch_size: int,
    upper_quantile: float,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    seed = _strict_int(seed, "power seed")
    outer_cohorts = _strict_int(
        outer_cohorts, "outer cohort count", minimum=1
    )
    inner_draws = _strict_int(inner_draws, "inner draw count", minimum=1)
    batch_size = _strict_int(batch_size, "outer batch size", minimum=1)
    upper_quantile = _strict_float(
        upper_quantile,
        "upper quantile",
        minimum=0.0,
        maximum=1.0,
    )
    _require(
        list(allocation) == list(EXPECTED_ALLOCATION)
        and all(int(value) > 0 for value in allocation.values()),
        "simulation requires the exact ordered future allocation",
    )
    _require(
        set(profiles) == set(EXPECTED_SOURCE_REPOSITORIES),
        "simulation source repository profiles changed",
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    upper_records: dict[str, list[np.ndarray]] = {
        scenario: [] for scenario in SCENARIO_ORDER
    }
    inner_sign_records: dict[str, list[np.ndarray]] = {
        scenario: [] for scenario in SCENARIO_ORDER
    }
    all_profiles = np.concatenate(
        [profiles[repository] for repository in EXPECTED_SOURCE_REPOSITORIES],
        axis=0,
    )
    p90 = np.quantile(
        all_profiles,
        0.9,
        axis=0,
        method="linear",
    )
    exchangeable_repositories = list(EXPECTED_SOURCE_REPOSITORIES)

    for start in range(0, outer_cohorts, batch_size):
        current_batch = min(batch_size, outer_cohorts - start)
        repository_draws: dict[str, np.ndarray] = {}
        for repository, future_count_value in allocation.items():
            if repository == FLASK_REPOSITORY:
                continue
            future_count = _strict_int(
                future_count_value,
                f"future count for {repository}",
                minimum=1,
            )
            source_profiles = np.asarray(profiles[repository], dtype=np.float64)
            sampled = source_profiles[
                rng.integers(
                    0,
                    len(source_profiles),
                    size=(current_batch, future_count),
                )
            ]
            repository_draws[repository] = _gamma_task_weighted_draws(
                sampled,
                rng=rng,
                inner_draws=inner_draws,
            )

        django_profiles = np.asarray(
            profiles[DJANGO_REPOSITORY], dtype=np.float64
        )
        flask_values: dict[str, np.ndarray] = {
            "django_domain_proxy": django_profiles[
                rng.integers(
                    0,
                    len(django_profiles),
                    size=current_batch,
                )
            ]
        }
        exchangeable = np.empty((current_batch, 2), dtype=np.float64)
        repository_picks = rng.integers(
            0,
            len(exchangeable_repositories),
            size=current_batch,
        )
        for batch_index, repository_index in enumerate(repository_picks):
            source_profiles = np.asarray(
                profiles[exchangeable_repositories[int(repository_index)]],
                dtype=np.float64,
            )
            exchangeable[batch_index] = source_profiles[
                int(rng.integers(0, len(source_profiles)))
            ]
        flask_values["exchangeable_repository_task_proxy"] = exchangeable
        flask_values["neutral_zero_effect"] = np.zeros(
            (current_batch, 2), dtype=np.float64
        )
        flask_values["empirical_marginal_p90_adverse"] = np.repeat(
            p90[None, :], current_batch, axis=0
        )

        repository_gamma = rng.gamma(
            shape=1.0,
            scale=1.0,
            size=(current_batch, inner_draws, len(allocation)),
        )
        _require(
            np.all(np.isfinite(repository_gamma))
            and np.all(repository_gamma > 0.0),
            "repository Gamma weights are invalid",
        )
        repository_gamma /= repository_gamma.sum(
            axis=2, keepdims=True, dtype=np.float64
        )
        for scenario in SCENARIO_ORDER:
            stacked = np.empty(
                (current_batch, inner_draws, len(allocation), 2),
                dtype=np.float64,
            )
            for repository_index, repository in enumerate(allocation):
                if repository == FLASK_REPOSITORY:
                    stacked[:, :, repository_index, :] = flask_values[
                        scenario
                    ][:, None, :]
                else:
                    stacked[:, :, repository_index, :] = repository_draws[
                        repository
                    ]
            paired_draws = np.sum(
                repository_gamma[:, :, :, None] * stacked,
                axis=2,
                dtype=np.float64,
            )
            upper = np.quantile(
                paired_draws,
                upper_quantile,
                axis=1,
                method="linear",
            )
            upper_records[scenario].append(upper)
            inner_sign_records[scenario].append(
                np.mean(np.all(paired_draws < 0.0, axis=2), axis=1)
            )
        completed = start + current_batch
        if progress is not None and (
            completed == outer_cohorts or completed % 1000 == 0
        ):
            progress(f"paired proper-score cohorts complete: {completed}")

    return {
        "seed": seed,
        "outer_cohorts": outer_cohorts,
        "inner_bootstrap_draws_per_cohort": inner_draws,
        "outer_batch_size": batch_size,
        "upper_quantile": upper_quantile,
        "quantile_method": "linear",
        "flask_empirical_marginal_p90_adverse_profile": {
            "multiclass_negative_log_likelihood": float(p90[0]),
            "multiclass_brier": float(p90[1]),
        },
        "scenarios": {
            scenario: _power_summary(
                np.concatenate(upper_records[scenario], axis=0),
                np.concatenate(inner_sign_records[scenario], axis=0),
            )
            for scenario in SCENARIO_ORDER
        },
    }


def _validate_raw_probability_block(
    nested: Mapping[str, Any],
    *,
    source_name: str,
    expected_hash: str,
    expected_row_ids_hash: str,
    expected_rows: int,
) -> np.ndarray:
    all_raw = _mapping(
        nested.get("full_development_oof_raw_probabilities"),
        "full-development raw probabilities",
    )
    block = _mapping(all_raw.get(source_name), f"{source_name} raw block")
    _require(
        block.get("float64_sha256") == expected_hash
        and block.get("known_row_ids_sha256") == expected_row_ids_hash,
        f"{source_name} raw-probability binding changed",
    )
    values = CALIBRATION.validate_probability_matrix(
        block.get("raw_probabilities"),
        f"{source_name} raw probabilities",
        expected_rows=expected_rows,
    )
    _require(
        _float64_sha256(values) == expected_hash,
        f"{source_name} raw-probability bytes changed",
    )
    return values


def build_temperature_task_profiles(
    validated_artifact: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sensitivity = _mapping(
        config["temperature_reselection_sensitivity"],
        "temperature sensitivity",
    )
    candidate_rows = validated_artifact["candidate_rows"]
    known_rows = [
        row for row in candidate_rows if row.get("metric_evaluable") is True
    ]
    nested = validated_artifact["nested"]
    expected_row_ids_hash = str(sensitivity["known_row_ids_sha256"])
    reference_raw = _validate_raw_probability_block(
        nested,
        source_name=str(sensitivity["reference_raw_source"]),
        expected_hash=str(sensitivity["reference_raw_float64_sha256"]),
        expected_row_ids_hash=expected_row_ids_hash,
        expected_rows=len(known_rows),
    )
    secondary_raw = _validate_raw_probability_block(
        nested,
        source_name=str(sensitivity["candidate_secondary_raw_source"]),
        expected_hash=str(
            sensitivity["candidate_secondary_raw_float64_sha256"]
        ),
        expected_row_ids_hash=expected_row_ids_hash,
        expected_rows=len(known_rows),
    )
    candidate_raw = CALIBRATION.geometric_pool_probabilities(
        reference_raw,
        secondary_raw,
        alpha=float(sensitivity["candidate_geometric_secondary_weight"]),
    )
    temperatures = tuple(float(item) for item in sensitivity["temperature_grid"])
    labels = np.asarray(
        [CLASS_INDEX[str(row["label"])] for row in known_rows],
        dtype=np.int64,
    )
    one_hot = np.eye(len(CLASSES), dtype=np.float64)[labels]
    row_losses = np.empty(
        (len(known_rows), 2, len(temperatures), 2),
        dtype=np.float64,
    )
    for branch_index, raw in enumerate((reference_raw, candidate_raw)):
        for temperature_index, temperature in enumerate(temperatures):
            q = CALIBRATION.temperature_scale_probabilities(raw, temperature)
            row_losses[:, branch_index, temperature_index, 0] = -np.log(
                q[np.arange(len(labels)), labels]
            )
            row_losses[:, branch_index, temperature_index, 1] = np.sum(
                (q - one_hot) ** 2,
                axis=1,
            )

    grouped: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    for index, row in enumerate(known_rows):
        key = (str(row["repo"]), str(row["task_id"]))
        grouped.setdefault(key, []).append(index)
    by_repository: OrderedDict[str, list[np.ndarray]] = OrderedDict()
    for (repository, _opaque_task_id), indices in grouped.items():
        by_repository.setdefault(repository, []).append(
            np.mean(row_losses[np.asarray(indices, dtype=np.int64)], axis=0)
        )
    profiles = {
        repository: np.asarray(items, dtype=np.float64)
        for repository, items in by_repository.items()
    }
    _require(
        set(profiles) == set(EXPECTED_SOURCE_REPOSITORIES)
        and sum(len(items) for items in profiles.values()) == 60,
        "temperature task-profile support changed",
    )
    return profiles, {
        "reference_raw_float64_sha256": _float64_sha256(reference_raw),
        "candidate_secondary_raw_float64_sha256": _float64_sha256(
            secondary_raw
        ),
        "candidate_geometric_raw_float64_sha256": _float64_sha256(
            candidate_raw
        ),
        "known_row_count": len(known_rows),
        "task_profile_count": 60,
        "temperatures_in_order": list(temperatures),
    }


def _temperature_indices(
    nll_by_temperature: np.ndarray,
    temperatures: Sequence[float],
) -> np.ndarray:
    minimum = np.min(nll_by_temperature, axis=2)
    preference = sorted(
        range(len(temperatures)),
        key=lambda index: (
            abs(float(temperatures[index]) - 1.0),
            float(temperatures[index]),
        ),
    )
    selected = np.full(minimum.shape, -1, dtype=np.int64)
    for index in preference:
        mask = (selected < 0) & (
            nll_by_temperature[:, :, index] == minimum
        )
        selected[mask] = index
    _require(np.all(selected >= 0), "temperature selection failed")
    return selected


def simulate_temperature_reselection(
    profiles: Mapping[str, np.ndarray],
    allocation: Mapping[str, int],
    *,
    temperatures: Sequence[float],
    fixed_temperature: float,
    seed: int,
    outer_cohorts: int,
    inner_draws: int,
    batch_size: int,
    upper_quantile: float,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    temperatures = tuple(float(item) for item in temperatures)
    _require(
        fixed_temperature in temperatures,
        "fixed temperature is absent from the frozen grid",
    )
    fixed_index = temperatures.index(float(fixed_temperature))
    rng = np.random.Generator(np.random.PCG64(seed))
    modes = ("fixed_temperature", "independently_reselected_temperature")
    upper_records = {mode: [] for mode in modes}
    sign_records = {mode: [] for mode in modes}
    reference_temperature_counts = np.zeros(len(temperatures), dtype=np.int64)
    candidate_temperature_counts = np.zeros(len(temperatures), dtype=np.int64)

    for start in range(0, outer_cohorts, batch_size):
        current_batch = min(batch_size, outer_cohorts - start)
        repository_draws: dict[str, np.ndarray] = {}
        for repository, future_count in allocation.items():
            if repository == FLASK_REPOSITORY:
                continue
            source_profiles = np.asarray(profiles[repository], dtype=np.float64)
            sampled = source_profiles[
                rng.integers(
                    0,
                    len(source_profiles),
                    size=(current_batch, int(future_count)),
                )
            ]
            repository_draws[repository] = _gamma_task_weighted_draws(
                sampled,
                rng=rng,
                inner_draws=inner_draws,
            )
        django_profiles = np.asarray(
            profiles[DJANGO_REPOSITORY], dtype=np.float64
        )
        flask = django_profiles[
            rng.integers(0, len(django_profiles), size=current_batch)
        ]
        repository_gamma = rng.gamma(
            1.0,
            1.0,
            size=(current_batch, inner_draws, len(allocation)),
        )
        repository_gamma /= repository_gamma.sum(
            axis=2, keepdims=True, dtype=np.float64
        )
        stacked = np.empty(
            (
                current_batch,
                inner_draws,
                len(allocation),
                2,
                len(temperatures),
                2,
            ),
            dtype=np.float64,
        )
        for repository_index, repository in enumerate(allocation):
            if repository == FLASK_REPOSITORY:
                stacked[:, :, repository_index] = flask[:, None]
            else:
                stacked[:, :, repository_index] = repository_draws[repository]
        aggregate = np.sum(
            repository_gamma[:, :, :, None, None, None] * stacked,
            axis=2,
            dtype=np.float64,
        )
        outer_index = np.arange(current_batch)[:, None]
        draw_index = np.arange(inner_draws)[None, :]

        fixed_nll = (
            aggregate[:, :, 1, fixed_index, 0]
            - aggregate[:, :, 0, fixed_index, 0]
        )
        fixed_brier = (
            aggregate[:, :, 1, fixed_index, 1]
            - aggregate[:, :, 0, fixed_index, 1]
        )
        fixed_draws = np.stack((fixed_nll, fixed_brier), axis=2)

        reference_selected = _temperature_indices(
            aggregate[:, :, 0, :, 0],
            temperatures,
        )
        candidate_selected = _temperature_indices(
            aggregate[:, :, 1, :, 0],
            temperatures,
        )
        selected_nll = (
            aggregate[
                outer_index,
                draw_index,
                1,
                candidate_selected,
                0,
            ]
            - aggregate[
                outer_index,
                draw_index,
                0,
                reference_selected,
                0,
            ]
        )
        selected_brier = (
            aggregate[
                outer_index,
                draw_index,
                1,
                candidate_selected,
                1,
            ]
            - aggregate[
                outer_index,
                draw_index,
                0,
                reference_selected,
                1,
            ]
        )
        selected_draws = np.stack((selected_nll, selected_brier), axis=2)
        reference_temperature_counts += np.bincount(
            reference_selected.ravel(),
            minlength=len(temperatures),
        )
        candidate_temperature_counts += np.bincount(
            candidate_selected.ravel(),
            minlength=len(temperatures),
        )
        for mode, paired_draws in (
            ("fixed_temperature", fixed_draws),
            ("independently_reselected_temperature", selected_draws),
        ):
            upper_records[mode].append(
                np.quantile(
                    paired_draws,
                    upper_quantile,
                    axis=1,
                    method="linear",
                )
            )
            sign_records[mode].append(
                np.mean(np.all(paired_draws < 0.0, axis=2), axis=1)
            )
        completed = start + current_batch
        if progress is not None and (
            completed == outer_cohorts or completed % 1000 == 0
        ):
            progress(f"temperature sensitivity cohorts complete: {completed}")

    total_selections = outer_cohorts * inner_draws
    return {
        "seed": seed,
        "outer_cohorts": outer_cohorts,
        "inner_bootstrap_draws_per_cohort": inner_draws,
        "outer_batch_size": batch_size,
        "flask_scenario": "django_domain_proxy",
        "fixed_temperature": fixed_temperature,
        "temperatures_in_order": list(temperatures),
        "modes": {
            mode: _power_summary(
                np.concatenate(upper_records[mode], axis=0),
                np.concatenate(sign_records[mode], axis=0),
            )
            for mode in modes
        },
        "selected_temperature_fractions": {
            "reference": {
                str(temperature): int(reference_temperature_counts[index])
                / total_selections
                for index, temperature in enumerate(temperatures)
            },
            "candidate": {
                str(temperature): int(candidate_temperature_counts[index])
                / total_selections
                for index, temperature in enumerate(temperatures)
            },
        },
        "models_refit": False,
        "interpretation": (
            "fixed_oof_raw_probability_sensitivity_not_full_model_refit_power"
        ),
    }


def build_power_report(
    config: Mapping[str, Any],
    validated_artifact: Mapping[str, Any],
    *,
    config_sha256: str,
    artifact_sha256: str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    profiles, source_summary = build_task_loss_profiles(
        validated_artifact["candidate_rows"],
        validated_artifact["reference_rows"],
    )
    resampling = config["resampling_contract"]
    paired_power = simulate_paired_proper_power(
        profiles,
        config["future_allocation_in_order"],
        seed=int(resampling["seed"]),
        outer_cohorts=int(resampling["outer_cohorts"]),
        inner_draws=int(resampling["inner_bootstrap_draws"]),
        batch_size=int(resampling["outer_batch_size"]),
        upper_quantile=float(resampling["upper_quantile"]),
        progress=progress,
    )
    sensitivity_config = config["temperature_reselection_sensitivity"]
    temperature_profiles, temperature_binding = build_temperature_task_profiles(
        validated_artifact,
        config,
    )
    temperature_sensitivity = simulate_temperature_reselection(
        temperature_profiles,
        config["future_allocation_in_order"],
        temperatures=sensitivity_config["temperature_grid"],
        fixed_temperature=float(sensitivity_config["fixed_temperature"]),
        seed=int(sensitivity_config["seed"]),
        outer_cohorts=int(sensitivity_config["outer_cohorts"]),
        inner_draws=int(sensitivity_config["inner_bootstrap_draws"]),
        batch_size=int(sensitivity_config["outer_batch_size"]),
        upper_quantile=float(resampling["upper_quantile"]),
        progress=progress,
    )

    threshold = float(config["decision_rule"]["minimum_acceptable_joint_power"])
    scenario_powers = {
        scenario: float(
            paired_power["scenarios"][scenario]["joint_gate"]["power"]
        )
        for scenario in SCENARIO_ORDER
    }
    all_scenarios_pass = all(
        power >= threshold for power in scenario_powers.values()
    )
    decision = (
        config["decision_rule"]["pass_label"]
        if all_scenarios_pass
        else config["decision_rule"]["fail_label"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "id": OUTPUT_ID,
        "status": "completed_design_power_screen_not_confirmatory",
        "scope": "fixed_prediction_paired_proper_score_power_only",
        "decision": decision,
        "decision_rule": dict(config["decision_rule"]),
        "all_declared_flask_scenarios_meet_joint_power_threshold": (
            all_scenarios_pass
        ),
        "minimum_observed_scenario_joint_power": min(
            scenario_powers.values()
        ),
        "scenario_joint_powers": scenario_powers,
        "source_binding": {
            "config_path": str(DEFAULT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "artifact_path": str(DEFAULT_SOURCE_ARTIFACT.relative_to(ROOT)),
            "artifact_sha256": artifact_sha256,
            "candidate_procedure": PRIMARY_PROCEDURE,
            "reference_procedure": REFERENCE_PROCEDURE,
            "candidate_forecast_q_float64_sha256": config["source_artifact"][
                "candidate_forecast_q_float64_sha256"
            ],
            "reference_forecast_q_float64_sha256": config["source_artifact"][
                "reference_forecast_q_float64_sha256"
            ],
            "shared_decision_d_float64_sha256": config["source_artifact"][
                "shared_decision_d_float64_sha256"
            ],
            "candidate_reference_decision_d_exactly_equal": True,
        },
        "future_allocation_in_order": dict(
            config["future_allocation_in_order"]
        ),
        "future_task_count": sum(
            int(value)
            for value in config["future_allocation_in_order"].values()
        ),
        "source_task_loss_profiles": source_summary,
        "paired_proper_score_power": paired_power,
        "temperature_reselection_binding": temperature_binding,
        "temperature_reselection_sensitivity": temperature_sensitivity,
        "fixed_predictions_conditioned_on": True,
        "models_refit_inside_power_screen": False,
        "full_model_refit_power": {
            "status": "not_estimated",
            "reason": (
                "source artifact contains one frozen set of out-of-fold "
                "predictions; no model is refit in this design power screen"
            ),
        },
        "independent_v4_development_result": False,
        "operational_reliability_claim": False,
        "fresh_cohort_selection_or_generation_authorized": False,
        "confirmatory_interpretation_forbidden": True,
        "reserved_validation_accessed": False,
        "reserved_validation_allowed": False,
        "task_text_prompt_completion_patch_or_trajectory_fields_consumed": False,
        "data_policy": dict(config["data_policy"]),
        "runtime": {
            "numpy_version": np.__version__,
            "bit_generator": "PCG64",
            "quantile_method": "linear",
        },
    }


def _forbidden_component(path: Path, forbidden: set[str]) -> bool:
    return any(
        part.casefold() in forbidden
        or Path(part).stem.casefold() in forbidden
        for part in path.parts
    )


def validate_cli_paths(
    *,
    config_path: Path,
    source_artifact_path: Path,
    output_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Path]:
    _require(
        config_path.is_file() and not config_path.is_symlink(),
        "power config must be a regular non-symlink file",
    )
    _require(
        source_artifact_path.is_file()
        and not source_artifact_path.is_symlink(),
        "source artifact must be a regular non-symlink file",
    )
    _require(
        config_path.resolve(strict=True) == DEFAULT_CONFIG.resolve(strict=True),
        "power config must use the canonical path",
    )
    _require(
        source_artifact_path.resolve(strict=True)
        == DEFAULT_SOURCE_ARTIFACT.resolve(strict=True),
        "source artifact must use the canonical path",
    )
    _require(
        output_root.is_dir() and not output_root.is_symlink(),
        "dedicated V4 design output root is unavailable",
    )
    resolved_root = output_root.resolve(strict=True)
    resolved_output = output_path.resolve(strict=False)
    _require(
        resolved_output.parent == resolved_root
        and resolved_output.suffix == ".json",
        "output must be one JSON file directly under the dedicated V4 design root",
    )
    _require(
        not output_path.exists() and not output_path.is_symlink(),
        "power output already exists; no-clobber is mandatory",
    )
    forbidden = {"reserved", "validation"}
    _require(
        not _forbidden_component(config_path.resolve(strict=True), forbidden)
        and not _forbidden_component(
            source_artifact_path.resolve(strict=True), forbidden
        )
        and not _forbidden_component(resolved_output, forbidden),
        "reserved or validation path components are forbidden",
    )
    return {
        "config": config_path.resolve(strict=True),
        "source_artifact": source_artifact_path.resolve(strict=True),
        "output": resolved_output,
    }


def _write_json_no_clobber(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--source-artifact",
        type=Path,
        default=DEFAULT_SOURCE_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = validate_cli_paths(
        config_path=args.config,
        source_artifact_path=args.source_artifact,
        output_path=args.output,
    )
    _require(
        np.__version__ == "2.5.1",
        "power screen requires NumPy 2.5.1",
    )
    config_sha256 = _sha256_file(paths["config"])
    _require(
        config_sha256 == POWER_CONFIG_SHA256,
        "power config bytes differ from the frozen hash",
    )
    artifact_sha256 = _sha256_file(paths["source_artifact"])
    _require(
        artifact_sha256 == SOURCE_ARTIFACT_SHA256,
        "source artifact bytes differ from the frozen hash",
    )
    config = validate_power_config(
        _read_json(paths["config"], "V4 power config")
    )
    validated_artifact = validate_source_artifact(
        _read_json(paths["source_artifact"], "contract-closed source artifact"),
        config,
    )
    report = build_power_report(
        config,
        validated_artifact,
        config_sha256=config_sha256,
        artifact_sha256=artifact_sha256,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    _write_json_no_clobber(paths["output"], report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "output": str(paths["output"]),
                "reserved_validation_accessed": False,
                "full_model_refit_power": "not_estimated",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
