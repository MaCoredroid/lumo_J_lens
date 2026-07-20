#!/usr/bin/env python3
"""Run the V4 nonconfirmatory transport and source-supply power screen.

The screen consumes one exact contract-closed design artifact and one exact
aggregate-only source-feasibility artifact.  It does not select task identities,
read task payloads, generate tasks, or access reserved validation data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy
from scipy.stats import beta as beta_distribution


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Direct ``python scripts/...py`` execution otherwise exposes only the
    # scripts directory, while the hash-bound power module uses its package
    # name.  Add the exact repository root before any bound import occurs.
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "configs/swe_task_state_v4_transport_power_screen.json"
DEFAULT_DESIGN_ARTIFACT = (
    ROOT
    / ".cache/swe_state_interpreter_v4_design"
    / "v3-n60-geometric-a020-shared-action-contract-closed.json"
)
DEFAULT_SOURCE_FEASIBILITY_ARTIFACT = (
    ROOT
    / ".cache/swe_state_interpreter_v4_design"
    / "swe-task-state-v4-source-feasibility.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / ".cache/swe_state_interpreter_v4_design"
DEFAULT_OUTPUT = (
    DEFAULT_OUTPUT_ROOT / "swe-task-state-v4-transport-power-screen.json"
)
SOURCE_FEASIBILITY_CONFIG = (
    ROOT / "configs/swe_task_state_v4_source_feasibility.json"
)
SOURCE_FEASIBILITY_ANALYZER = (
    ROOT / "scripts/analyze_swe_task_state_v4_source_feasibility.py"
)
PAIRED_POWER_CONFIG = ROOT / "configs/swe_task_state_v4_power_screen.json"
PAIRED_POWER_ANALYZER = ROOT / "scripts/analyze_swe_task_state_v4_power.py"

SCHEMA_VERSION = 1
CONFIG_ID = "swe-task-state-v4-transport-power-screen-v1"
OUTPUT_ID = "swe-task-state-v4-transport-power-result-v1"
TRANSPORT_CONFIG_SHA256 = (
    "5fb51e564f927f60474708bc47454fffada7d87e1fd9bd9b5907c19287191525"
)
DESIGN_ARTIFACT_SHA256 = (
    "e1436a9c7fe763a4fab6ae505af17395a4290faefd0734b230d22127bff0cf8c"
)
SOURCE_FEASIBILITY_ARTIFACT_SHA256 = (
    "e7b2f6a0fecbc7c4ff10e41463ff0c83898a5a59b070856ac820279b4652be89"
)
SOURCE_FEASIBILITY_CONFIG_SHA256 = (
    "165f8c5d179c24855e8edfec30d22c078706b3703969f655af9a062de725afb8"
)
SOURCE_FEASIBILITY_ANALYZER_SHA256 = (
    "c9235bf6a96888338ecd7b3bc692c1c5b26c1c63b192486e9807240bf541a1c2"
)
PAIRED_POWER_CONFIG_SHA256 = (
    "d696ca7edd58be17abd29cc4908322d4741a54fee2d90c89a7e05dcf61fa6574"
)
PAIRED_POWER_ANALYZER_SHA256 = (
    "0a21926c03578490b4423fa07d54ff7120babaee1e007689ad7b161d3c2191d7"
)

REPOSITORIES = (
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "mwaskom/seaborn",
    "pallets/flask",
    "psf/requests",
    "pydata/xarray",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
)
KNOWN_REPOSITORIES = (
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
UNSEEN_REPOSITORIES = ("mwaskom/seaborn", "pallets/flask")
DJANGO_REPOSITORY = "django/django"

OFFICIAL_SUPPLY = OrderedDict(
    (
        ("astropy/astropy", 73),
        ("django/django", 619),
        ("matplotlib/matplotlib", 150),
        ("mwaskom/seaborn", 20),
        ("pallets/flask", 10),
        ("psf/requests", 36),
        ("pydata/xarray", 88),
        ("pylint-dev/pylint", 47),
        ("pytest-dev/pytest", 100),
        ("scikit-learn/scikit-learn", 197),
        ("sphinx-doc/sphinx", 143),
        ("sympy/sympy", 311),
    )
)
PILOT_ALLOCATION = OrderedDict(
    (("mwaskom/seaborn", 10), ("pallets/flask", 5))
)
CONFIRMATION_ALLOCATION = OrderedDict(
    (
        ("astropy/astropy", 20),
        ("django/django", 20),
        ("matplotlib/matplotlib", 20),
        ("mwaskom/seaborn", 20),
        ("pallets/flask", 10),
        ("psf/requests", 20),
        ("pydata/xarray", 20),
        ("pylint-dev/pylint", 20),
        ("pytest-dev/pytest", 20),
        ("scikit-learn/scikit-learn", 20),
        ("sphinx-doc/sphinx", 20),
        ("sympy/sympy", 20),
    )
)
QUALIFIED_SHORTFALL = OrderedDict(
    (("mwaskom/seaborn", 10), ("pallets/flask", 5))
)

PRIMARY_SCENARIOS = (
    "django_domain_proxy",
    "exchangeable_repository_task_proxy",
    "neutral_zero_effect",
    "empirical_marginal_p90_adverse",
)
SENSITIVITY_SCENARIOS = (
    "django_domain_proxy_independent_unseen_repositories",
    "exchangeable_repository_task_proxy_independent_unseen_repositories",
)
CHILD_STREAMS = (
    "known_task_sampling_and_gamma",
    "repository_gamma",
    "django_proxy_perfect_coupling",
    "exchangeable_proxy_perfect_coupling",
    "django_proxy_independent_sensitivity",
    "exchangeable_proxy_independent_sensitivity",
)
EXPECTED_DECISION_LABELS = (
    "NO_GO_RETAINED_TRANSPORT_ENVELOPE",
    "NO_GO_QUALIFIED_SOURCE_SHORTFALL",
)
FORBIDDEN_PATH_TOKENS = {"reserved", "validation"}
METRIC_KEYS = (
    "multiclass_negative_log_likelihood",
    "multiclass_brier",
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


def _require_exact_file_hash(path: Path, expected: str, label: str) -> str:
    _require(
        path.is_file() and not path.is_symlink(),
        f"{label} must be a regular non-symlink file",
    )
    observed = _sha256_file(path)
    _require(observed == expected, f"{label} bytes differ from frozen hash")
    return observed


def _same_numbers(
    observed: Mapping[str, Any],
    expected: Mapping[str, float],
    label: str,
    *,
    tolerance: float = 1e-15,
) -> None:
    _require(set(observed) == set(expected), f"{label} metric keys changed")
    for key, expected_value in expected.items():
        value = _strict_float(observed[key], f"{label} {key}")
        _require(
            math.isclose(value, expected_value, rel_tol=0.0, abs_tol=tolerance),
            f"{label} {key} changed",
        )


def validate_transport_config(value: Any) -> dict[str, Any]:
    config = _mapping(value, "transport power config")
    expected_top_level = {
        "schema_version",
        "id",
        "status",
        "bindings",
        "repository_order",
        "known_source_repositories",
        "unseen_transport_repositories",
        "official_source_supply",
        "two_stage_allocation",
        "loss_contract",
        "transport_scenarios",
        "resampling_contract",
        "monte_carlo_decision",
        "structural_p90_contract",
        "decision_rule",
        "output_contract",
        "data_policy",
    }
    _require(
        set(config) == expected_top_level,
        "transport power config top-level fields changed",
    )
    _require(
        config.get("schema_version") == SCHEMA_VERSION
        and config.get("id") == CONFIG_ID
        and config.get("status")
        == "design_only_nonconfirmatory_transport_power",
        "transport power config identity changed",
    )

    bindings = _mapping(config.get("bindings"), "transport bindings")
    _require(
        set(bindings)
        == {
            "contract_closed_design_artifact",
            "source_feasibility_artifact",
            "paired_power_implementation",
        },
        "transport binding fields changed",
    )
    design = _mapping(
        bindings["contract_closed_design_artifact"], "design binding"
    )
    _require(
        dict(design)
        == {
            "canonical_path": (
                ".cache/swe_state_interpreter_v4_design/"
                "v3-n60-geometric-a020-shared-action-contract-closed.json"
            ),
            "sha256": DESIGN_ARTIFACT_SHA256,
            "required_status": "completed_design_screen_not_confirmatory",
            "required_reserved_validation_accessed": False,
            "required_reserved_validation_allowed": False,
        },
        "design binding changed",
    )
    source = _mapping(
        bindings["source_feasibility_artifact"],
        "source-feasibility binding",
    )
    _require(
        dict(source)
        == {
            "canonical_path": (
                ".cache/swe_state_interpreter_v4_design/"
                "swe-task-state-v4-source-feasibility.json"
            ),
            "sha256": SOURCE_FEASIBILITY_ARTIFACT_SHA256,
            "required_id": "swe-task-state-v4-source-feasibility-result-v1",
            "required_status": (
                "completed_design_source_feasibility_not_selection"
            ),
            "config_path": "configs/swe_task_state_v4_source_feasibility.json",
            "config_sha256": SOURCE_FEASIBILITY_CONFIG_SHA256,
            "analyzer_path": (
                "scripts/analyze_swe_task_state_v4_source_feasibility.py"
            ),
            "analyzer_sha256": SOURCE_FEASIBILITY_ANALYZER_SHA256,
        },
        "source-feasibility binding changed",
    )
    implementation = _mapping(
        bindings["paired_power_implementation"],
        "paired-power implementation binding",
    )
    _require(
        dict(implementation)
        == {
            "analyzer_path": "scripts/analyze_swe_task_state_v4_power.py",
            "analyzer_sha256": PAIRED_POWER_ANALYZER_SHA256,
            "config_path": "configs/swe_task_state_v4_power_screen.json",
            "config_sha256": PAIRED_POWER_CONFIG_SHA256,
        },
        "paired-power implementation binding changed",
    )

    _require(
        config.get("repository_order") == list(REPOSITORIES)
        and config.get("known_source_repositories")
        == list(KNOWN_REPOSITORIES)
        and config.get("unseen_transport_repositories")
        == list(UNSEEN_REPOSITORIES),
        "repository identity or order changed",
    )
    _require(
        config.get("official_source_supply") == dict(OFFICIAL_SUPPLY),
        "official source supply changed",
    )
    allocation = _mapping(
        config.get("two_stage_allocation"), "two-stage allocation"
    )
    _require(
        allocation.get("transport_pilot") == dict(PILOT_ALLOCATION)
        and allocation.get("fresh_confirmation")
        == dict(CONFIRMATION_ALLOCATION)
        and allocation.get("pilot_task_count") == 15
        and allocation.get("confirmation_cap_per_repository") == 20
        and allocation.get("confirmation_allocation_rule")
        == "min_cap_and_official_complement_availability_before_pilot"
        and allocation.get("confirmation_task_count") == 230
        and allocation.get("combined_task_count") == 245
        and allocation.get("pilot_outcomes_may_not_enter_confirmation") is True
        and allocation.get(
            "confirmation_must_remain_untouched_until_pilot_gate_and_power_rescreen"
        )
        is True
        and allocation.get("qualified_shortfall")
        == dict(QUALIFIED_SHORTFALL),
        "two-stage allocation or separation contract changed",
    )
    loss = _mapping(config.get("loss_contract"), "loss contract")
    _require(
        dict(loss)
        == {
            "difference_direction": "candidate_minus_reference",
            "task_profile": "equal_mean_of_known_rows_within_whole_task",
            "point_estimand": (
                "equal_repository_then_equal_task_then_equal_known_row"
            ),
            "nll_gate": "bootstrap_upper_strictly_less_than_zero",
            "brier_gate": "bootstrap_upper_strictly_less_than_zero",
            "joint_gate": "nll_gate_and_brier_gate",
        },
        "loss contract changed",
    )

    scenarios = _mapping(
        config.get("transport_scenarios"), "transport scenarios"
    )
    _require(
        scenarios.get("primary_order") == list(PRIMARY_SCENARIOS)
        and scenarios.get("independent_sensitivity_order")
        == list(SENSITIVITY_SCENARIOS)
        and scenarios.get("primary_unseen_repository_coupling")
        == (
            "one_repository_level_latent_profile_per_outer_cohort_"
            "perfectly_shared_by_flask_and_seaborn"
        )
        and scenarios.get("independent_sensitivity_coupling")
        == (
            "one_independent_repository_level_latent_profile_for_each_"
            "unseen_repository_per_outer_cohort"
        )
        and scenarios.get(
            "novel_task_counts_do_not_reduce_repository_level_latent_transport_uncertainty"
        )
        is True,
        "transport scenario identity or coupling changed",
    )
    _require(
        scenarios.get("django_domain_proxy")
        == "one_whole_django_task_profile_per_outer_cohort"
        and scenarios.get("exchangeable_repository_task_proxy")
        == (
            "one_known_repository_uniform_then_one_whole_task_uniform_"
            "per_outer_cohort"
        )
        and scenarios.get("neutral_zero_effect")
        == {"nll_difference": 0.0, "brier_difference": 0.0}
        and scenarios.get("empirical_marginal_p90_adverse")
        == {
            "quantile": 0.9,
            "method": "linear",
            "same_fixed_profile_for_both_unseen_repositories": True,
        },
        "transport scenario definitions changed",
    )

    resampling = _mapping(
        config.get("resampling_contract"), "resampling contract"
    )
    _require(
        resampling.get("numpy_version") == "2.5.1"
        and resampling.get("scipy_version") == "1.18.0"
        and resampling.get("bit_generator") == "PCG64"
        and resampling.get("seed_sequence_entropy") == 20260729
        and resampling.get("child_streams_in_order") == list(CHILD_STREAMS)
        and resampling.get("child_stream_derivation")
        == "numpy_SeedSequence_spawn_once_six_children_in_order"
        and resampling.get("outer_cohorts") == 10000
        and resampling.get("outer_batch_size") == 10
        and resampling.get("inner_bootstrap_draws") == 1000
        and resampling.get("upper_quantile") == 0.975
        and resampling.get("quantile_method") == "linear"
        and resampling.get(
            "same_known_task_and_repository_weight_draws_across_scenarios"
        )
        is True,
        "resampling identity changed",
    )
    _require(
        resampling.get("task_weight_distribution")
        == "Gamma_shape_1_scale_1_normalized_within_repository"
        and resampling.get("repository_weight_distribution")
        == "Gamma_shape_1_scale_1_normalized_across_twelve_repositories",
        "Gamma weighting contract changed",
    )
    monte_carlo = _mapping(
        config.get("monte_carlo_decision"), "Monte Carlo decision"
    )
    _require(
        dict(monte_carlo)
        == {
            "minimum_acceptable_joint_power": 0.8,
            "familywise_alpha": 0.05,
            "declared_primary_scenario_count": 4,
            "one_sided_alpha_per_scenario": 0.0125,
            "lower_bound": "exact_clopper_pearson_beta_ppf",
            "scipy_beta_parameters": (
                "beta_ppf(alpha,passes,cohorts-passes+1)"
            ),
            "go_requires_every_primary_scenario_lower_bound_at_or_above_threshold": True,
        },
        "Monte Carlo decision changed",
    )

    structural = _mapping(
        config.get("structural_p90_contract"), "structural p90 contract"
    )
    _same_numbers(
        _mapping(
            structural.get("known_ten_repository_point"),
            "known ten-repository point",
        ),
        {
            METRIC_KEYS[0]: -0.006178540156555733,
            METRIC_KEYS[1]: -0.0021992920266340173,
        },
        "known ten-repository point",
    )
    _same_numbers(
        _mapping(
            structural.get("empirical_marginal_p90_profile"),
            "p90 profile",
        ),
        {
            METRIC_KEYS[0]: 0.0022797267620328285,
            METRIC_KEYS[1]: 0.003122176862007263,
        },
        "p90 profile",
    )
    _same_numbers(
        _mapping(
            structural.get("infinite_task_equal_twelve_repository_point"),
            "infinite-task point",
        ),
        {
            METRIC_KEYS[0]: -0.00476882900345764,
            METRIC_KEYS[1]: -0.0013123805451938037,
        },
        "infinite-task point",
    )
    _require(
        structural.get("recovered_repository_gamma_upper_0_975")
        == {
            METRIC_KEYS[0]: -0.00194661,
            METRIC_KEYS[1]: 0.00005254,
        }
        and structural.get("strict_brier_gate_passes") is False
        and structural.get("more_within_repository_tasks_can_repair") is False,
        "structural p90 fail-closed result changed",
    )

    decision = _mapping(config.get("decision_rule"), "decision rule")
    _require(
        dict(decision)
        == {
            "overall_decision": "NO_GO",
            "fail_closed_labels_in_order": list(EXPECTED_DECISION_LABELS),
            "fresh_cohort_selection_or_generation_authorized": False,
            "pilot_authorized": False,
            "confirmation_authorized": False,
        },
        "fail-closed decision rule changed",
    )
    output = _mapping(config.get("output_contract"), "output contract")
    _require(
        output.get("dedicated_root")
        == ".cache/swe_state_interpreter_v4_design"
        and output.get("default_filename")
        == "swe-task-state-v4-transport-power-screen.json"
        and output.get("direct_child_json_only") is True
        and output.get("new_output_no_clobber") is True
        and output.get("aggregate_only_no_raw_task_ids") is True
        and output.get("forbidden_casefolded_path_tokens")
        == ["reserved", "validation"],
        "output contract changed",
    )
    policy = _mapping(config.get("data_policy"), "data policy")
    _require(
        dict(policy)
        == {
            "input_scope": (
                "exact_contract_closed_design_and_aggregate_source_"
                "feasibility_only"
            ),
            "task_text_prompt_completion_patch_or_trajectory_fields_consumed": False,
            "raw_task_ids_emitted": False,
            "selection_performed": False,
            "selection_authorized": False,
            "generation_performed": False,
            "generation_authorized": False,
            "reserved_membership_accessed": False,
            "reserved_validation_accessed": False,
            "reserved_validation_allowed": False,
            "confirmatory_interpretation": False,
            "operational_reliability_claim": False,
            "independent_v4_development_result": False,
        },
        "data policy changed",
    )
    return dict(config)


def validate_source_feasibility_artifact(
    value: Any, config: Mapping[str, Any]
) -> dict[str, Any]:
    artifact = _mapping(value, "source-feasibility artifact")
    binding = _mapping(
        _mapping(config["bindings"], "bindings")[
            "source_feasibility_artifact"
        ],
        "source-feasibility binding",
    )
    _require(
        artifact.get("schema_version") == 1
        and artifact.get("id") == binding["required_id"]
        and artifact.get("status") == binding["required_status"]
        and artifact.get("source_feasibility_passed") is True,
        "source-feasibility artifact identity changed",
    )
    complement = _mapping(artifact.get("complement"), "source complement")
    _require(
        artifact.get("candidate_universe")
        == "full_test_identity_set_minus_verified_identity_set"
        and complement.get("instance_count") == 1794
        and complement.get("instance_ids_set_sha256")
        == "953b83337651cfa8e68f812f30e3ba1394a8a08e1f66980680832a1d6bd02861"
        and complement.get("repository_counts") == dict(OFFICIAL_SUPPLY),
        "source complement identity or counts changed",
    )
    source_binding = _mapping(
        artifact.get("source_binding"), "artifact source binding"
    )
    _require(
        source_binding.get("config_path") == binding["config_path"]
        and source_binding.get("config_sha256") == binding["config_sha256"]
        and source_binding.get("analyzer_path") == binding["analyzer_path"]
        and source_binding.get("analyzer_sha256")
        == binding["analyzer_sha256"],
        "source-feasibility config/analyzer binding changed",
    )
    for key in (
        "fresh_cohort_selection_authorized",
        "fresh_cohort_selection_performed",
        "generation_authorized",
        "generation_performed",
        "independent_v4_development_result",
        "operational_reliability_claim",
        "power_analysis_performed",
        "raw_instance_ids_emitted",
        "reserved_membership_accessed",
        "reserved_validation_accessed",
        "reserved_validation_allowed",
        "reserved_validation_data_accessed",
        "task_payload_fields_read",
        "confirmatory_interpretation",
    ):
        _require(artifact.get(key) is False, f"source policy flag changed: {key}")
    policy = _mapping(artifact.get("data_policy"), "source data policy")
    for key in (
        "selection_authorized",
        "selection_performed",
        "generation_authorized",
        "generation_performed",
        "reserved_membership_accessed",
        "reserved_validation_data_accessed",
        "reserved_validation_allowed",
        "task_payload_fields_read",
        "confirmatory_interpretation",
        "operational_reliability_claim",
        "independent_v4_development_result",
        "power_analysis_performed",
    ):
        _require(policy.get(key) is False, f"source data-policy flag changed: {key}")
    return dict(artifact)


def verify_bound_files() -> dict[str, str]:
    return {
        "transport_config_sha256": _require_exact_file_hash(
            DEFAULT_CONFIG, TRANSPORT_CONFIG_SHA256, "transport config"
        ),
        "design_artifact_sha256": _require_exact_file_hash(
            DEFAULT_DESIGN_ARTIFACT,
            DESIGN_ARTIFACT_SHA256,
            "contract-closed design artifact",
        ),
        "source_feasibility_artifact_sha256": _require_exact_file_hash(
            DEFAULT_SOURCE_FEASIBILITY_ARTIFACT,
            SOURCE_FEASIBILITY_ARTIFACT_SHA256,
            "source-feasibility artifact",
        ),
        "source_feasibility_config_sha256": _require_exact_file_hash(
            SOURCE_FEASIBILITY_CONFIG,
            SOURCE_FEASIBILITY_CONFIG_SHA256,
            "source-feasibility config",
        ),
        "source_feasibility_analyzer_sha256": _require_exact_file_hash(
            SOURCE_FEASIBILITY_ANALYZER,
            SOURCE_FEASIBILITY_ANALYZER_SHA256,
            "source-feasibility analyzer",
        ),
        "paired_power_config_sha256": _require_exact_file_hash(
            PAIRED_POWER_CONFIG,
            PAIRED_POWER_CONFIG_SHA256,
            "paired-power config",
        ),
        "paired_power_analyzer_sha256": _require_exact_file_hash(
            PAIRED_POWER_ANALYZER,
            PAIRED_POWER_ANALYZER_SHA256,
            "paired-power analyzer",
        ),
    }


def load_bound_power_module() -> ModuleType:
    _require_exact_file_hash(
        PAIRED_POWER_ANALYZER,
        PAIRED_POWER_ANALYZER_SHA256,
        "paired-power analyzer",
    )
    _require_exact_file_hash(
        PAIRED_POWER_CONFIG,
        PAIRED_POWER_CONFIG_SHA256,
        "paired-power config",
    )
    module = importlib.import_module("scripts.analyze_swe_task_state_v4_power")
    _require(
        module.POWER_CONFIG_SHA256 == PAIRED_POWER_CONFIG_SHA256
        and module.SOURCE_ARTIFACT_SHA256 == DESIGN_ARTIFACT_SHA256,
        "imported paired-power implementation binding changed",
    )
    return module


def load_frozen_task_profiles() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _require_exact_file_hash(
        DEFAULT_DESIGN_ARTIFACT,
        DESIGN_ARTIFACT_SHA256,
        "contract-closed design artifact",
    )
    power = load_bound_power_module()
    power_config = power.validate_power_config(
        _read_json(PAIRED_POWER_CONFIG, "paired-power config")
    )
    validated = power.validate_source_artifact(
        _read_json(DEFAULT_DESIGN_ARTIFACT, "contract-closed design artifact"),
        power_config,
    )
    profiles, summary = power.build_task_loss_profiles(
        validated["candidate_rows"], validated["reference_rows"]
    )
    return profiles, summary


def derive_supply_feasibility(
    official_supply: Mapping[str, int],
    pilot: Mapping[str, int],
    confirmation: Mapping[str, int],
) -> dict[str, Any]:
    _require(
        list(official_supply) == list(REPOSITORIES),
        "official supply repository order changed",
    )
    _require(
        list(pilot) == list(PILOT_ALLOCATION),
        "pilot repository order changed",
    )
    _require(
        list(confirmation) == list(REPOSITORIES),
        "confirmation repository order changed",
    )
    combined: OrderedDict[str, int] = OrderedDict()
    shortfall: OrderedDict[str, int] = OrderedDict()
    remaining_after_pilot: OrderedDict[str, int] = OrderedDict()
    feasible_confirmation: OrderedDict[str, int] = OrderedDict()
    for repository in REPOSITORIES:
        available = _strict_int(
            official_supply[repository], f"supply {repository}"
        )
        pilot_count = _strict_int(
            pilot.get(repository, 0), f"pilot {repository}"
        )
        confirmation_count = _strict_int(
            confirmation[repository], f"confirmation {repository}"
        )
        demand = pilot_count + confirmation_count
        combined[repository] = demand
        if demand > available:
            shortfall[repository] = demand - available
        remaining = max(0, available - pilot_count)
        remaining_after_pilot[repository] = remaining
        feasible_confirmation[repository] = min(confirmation_count, remaining)
    return {
        "official_supply": dict(official_supply),
        "pilot_allocation": dict(pilot),
        "fresh_confirmation_allocation": dict(confirmation),
        "combined_demand": dict(combined),
        "remaining_after_pilot": dict(remaining_after_pilot),
        "official_only_feasible_confirmation": dict(feasible_confirmation),
        "pilot_task_count": sum(pilot.values()),
        "fresh_confirmation_task_count": sum(confirmation.values()),
        "combined_task_count": sum(combined.values()),
        "official_only_feasible_confirmation_task_count": sum(
            feasible_confirmation.values()
        ),
        "qualified_shortfall": dict(shortfall),
        "qualified_source_supply_passes": not shortfall,
    }


def _validate_profiles(profiles: Mapping[str, np.ndarray]) -> None:
    _require(
        list(profiles) == list(KNOWN_REPOSITORIES),
        "known task-profile repository order changed",
    )
    for repository, values in profiles.items():
        array = np.asarray(values, dtype=np.float64)
        _require(
            array.ndim == 2
            and array.shape[0] > 0
            and array.shape[1] == 2
            and np.all(np.isfinite(array)),
            f"task profiles are invalid for {repository}",
        )


def _validate_confirmation_allocation(
    allocation: Mapping[str, int],
) -> OrderedDict[str, int]:
    _require(
        list(allocation) == list(REPOSITORIES),
        "confirmation allocation repository order changed",
    )
    result: OrderedDict[str, int] = OrderedDict()
    for repository, value in allocation.items():
        result[repository] = _strict_int(
            value, f"confirmation allocation {repository}", minimum=1
        )
    return result


def _spawn_generators(
    seed: int,
) -> tuple[dict[str, np.random.Generator], dict[str, Any]]:
    entropy = _strict_int(seed, "SeedSequence entropy")
    seed_sequence = np.random.SeedSequence(entropy)
    children = seed_sequence.spawn(len(CHILD_STREAMS))
    generators = {
        name: np.random.Generator(np.random.PCG64(child))
        for name, child in zip(CHILD_STREAMS, children, strict=True)
    }
    manifest = {
        "entropy": entropy,
        "pool_size": seed_sequence.pool_size,
        "child_streams_in_order": [
            {"name": name, "spawn_key": list(child.spawn_key)}
            for name, child in zip(CHILD_STREAMS, children, strict=True)
        ],
        "spawn_called_once_with_child_count": len(CHILD_STREAMS),
    }
    return generators, manifest


def _gamma_task_weighted_draws(
    sampled_profiles: np.ndarray,
    *,
    rng: np.random.Generator,
    inner_draws: int,
) -> np.ndarray:
    _require(
        sampled_profiles.ndim == 3
        and sampled_profiles.shape[0] > 0
        and sampled_profiles.shape[1] > 0
        and sampled_profiles.shape[2] == 2,
        "sampled task profiles have invalid shape",
    )
    outer_batch, task_count, _metrics = sampled_profiles.shape
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
    return np.einsum(
        "bin,bnf->bif", gamma, sampled_profiles, optimize=True
    )


def _sample_exchangeable_profiles(
    profiles: Mapping[str, np.ndarray],
    *,
    rng: np.random.Generator,
    outer_batch: int,
) -> np.ndarray:
    result = np.empty((outer_batch, 2), dtype=np.float64)
    repository_picks = rng.integers(
        0, len(KNOWN_REPOSITORIES), size=outer_batch
    )
    for index, repository_index in enumerate(repository_picks):
        source = np.asarray(
            profiles[KNOWN_REPOSITORIES[int(repository_index)]],
            dtype=np.float64,
        )
        result[index] = source[int(rng.integers(0, len(source)))]
    return result


def clopper_pearson_lower(
    passes: int, cohorts: int, one_sided_alpha: float
) -> float:
    count = _strict_int(cohorts, "cohort count", minimum=1)
    successes = _strict_int(passes, "pass count")
    _require(successes <= count, "pass count exceeds cohort count")
    alpha = _strict_float(
        one_sided_alpha,
        "one-sided alpha",
        minimum=np.finfo(np.float64).tiny,
        maximum=1.0 - np.finfo(np.float64).eps,
    )
    if successes == 0:
        return 0.0
    result = float(
        beta_distribution.ppf(alpha, successes, count - successes + 1)
    )
    _require(
        math.isfinite(result) and 0.0 <= result <= 1.0,
        "Clopper-Pearson lower bound is invalid",
    )
    return result


def _gate_summary(
    upper: np.ndarray, *, one_sided_alpha: float
) -> dict[str, Any]:
    _require(
        upper.ndim == 2 and upper.shape[0] > 0 and upper.shape[1] == 2,
        "bootstrap upper array has invalid shape",
    )
    nll_pass = upper[:, 0] < 0.0
    brier_pass = upper[:, 1] < 0.0
    joint_pass = nll_pass & brier_pass
    count = len(upper)

    def summarize(values: np.ndarray) -> dict[str, Any]:
        passes = int(values.sum())
        power = passes / count
        return {
            "passing_cohort_count": passes,
            "cohort_count": count,
            "power": power,
            "monte_carlo_standard_error": math.sqrt(
                power * (1.0 - power) / count
            ),
            "exact_one_sided_clopper_pearson_lower": clopper_pearson_lower(
                passes, count, one_sided_alpha
            ),
            "one_sided_alpha": one_sided_alpha,
        }

    return {
        "nll_gate": summarize(nll_pass),
        "brier_gate": summarize(brier_pass),
        "joint_gate": summarize(joint_pass),
        "strict_zero_gate": True,
    }


def simulate_transport_power(
    profiles: Mapping[str, np.ndarray],
    allocation: Mapping[str, int],
    *,
    seed: int,
    outer_cohorts: int,
    inner_draws: int,
    batch_size: int,
    upper_quantile: float,
    one_sided_alpha: float,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    _validate_profiles(profiles)
    ordered_allocation = _validate_confirmation_allocation(allocation)
    outer_count = _strict_int(
        outer_cohorts, "outer cohort count", minimum=1
    )
    inner_count = _strict_int(inner_draws, "inner draw count", minimum=1)
    batch_count = _strict_int(batch_size, "outer batch size", minimum=1)
    upper_q = _strict_float(
        upper_quantile,
        "upper quantile",
        minimum=0.0,
        maximum=1.0,
    )
    alpha = _strict_float(
        one_sided_alpha,
        "one-sided alpha",
        minimum=np.finfo(np.float64).tiny,
        maximum=1.0 - np.finfo(np.float64).eps,
    )
    generators, stream_manifest = _spawn_generators(seed)
    known_rng = generators[CHILD_STREAMS[0]]
    repository_rng = generators[CHILD_STREAMS[1]]
    django_coupled_rng = generators[CHILD_STREAMS[2]]
    exchangeable_coupled_rng = generators[CHILD_STREAMS[3]]
    django_independent_rng = generators[CHILD_STREAMS[4]]
    exchangeable_independent_rng = generators[CHILD_STREAMS[5]]

    all_task_profiles = np.concatenate(
        [
            np.asarray(profiles[repository], dtype=np.float64)
            for repository in KNOWN_REPOSITORIES
        ],
        axis=0,
    )
    p90 = np.quantile(
        all_task_profiles, 0.9, axis=0, method="linear"
    )
    upper_records: dict[str, list[np.ndarray]] = {
        scenario: []
        for scenario in (*PRIMARY_SCENARIOS, *SENSITIVITY_SCENARIOS)
    }

    for start in range(0, outer_count, batch_count):
        current_batch = min(batch_count, outer_count - start)
        known_draws: dict[str, np.ndarray] = {}
        for repository in REPOSITORIES:
            if repository in UNSEEN_REPOSITORIES:
                continue
            future_count = ordered_allocation[repository]
            source_profiles = np.asarray(
                profiles[repository], dtype=np.float64
            )
            sampled = source_profiles[
                known_rng.integers(
                    0,
                    len(source_profiles),
                    size=(current_batch, future_count),
                )
            ]
            known_draws[repository] = _gamma_task_weighted_draws(
                sampled, rng=known_rng, inner_draws=inner_count
            )

        repository_gamma = repository_rng.gamma(
            shape=1.0,
            scale=1.0,
            size=(current_batch, inner_count, len(REPOSITORIES)),
        )
        _require(
            np.all(np.isfinite(repository_gamma))
            and np.all(repository_gamma > 0.0),
            "repository Gamma weights are invalid",
        )
        repository_gamma /= repository_gamma.sum(
            axis=2, keepdims=True, dtype=np.float64
        )

        django_profiles = np.asarray(
            profiles[DJANGO_REPOSITORY], dtype=np.float64
        )
        coupled_django = django_profiles[
            django_coupled_rng.integers(
                0, len(django_profiles), size=current_batch
            )
        ]
        coupled_exchangeable = _sample_exchangeable_profiles(
            profiles,
            rng=exchangeable_coupled_rng,
            outer_batch=current_batch,
        )
        independent_django = {
            repository: django_profiles[
                django_independent_rng.integers(
                    0, len(django_profiles), size=current_batch
                )
            ]
            for repository in UNSEEN_REPOSITORIES
        }
        independent_exchangeable = {
            repository: _sample_exchangeable_profiles(
                profiles,
                rng=exchangeable_independent_rng,
                outer_batch=current_batch,
            )
            for repository in UNSEEN_REPOSITORIES
        }

        def evaluate_scenario(
            scenario: str,
            unseen_values: Mapping[str, np.ndarray],
        ) -> None:
            stacked = np.empty(
                (current_batch, inner_count, len(REPOSITORIES), 2),
                dtype=np.float64,
            )
            for repository_index, repository in enumerate(REPOSITORIES):
                if repository in UNSEEN_REPOSITORIES:
                    values = np.asarray(
                        unseen_values[repository], dtype=np.float64
                    )
                    _require(
                        values.shape == (current_batch, 2),
                        f"unseen scenario profile shape changed: {scenario}",
                    )
                    stacked[:, :, repository_index, :] = values[:, None, :]
                else:
                    stacked[:, :, repository_index, :] = known_draws[
                        repository
                    ]
            paired_draws = np.sum(
                repository_gamma[:, :, :, None] * stacked,
                axis=2,
                dtype=np.float64,
            )
            upper_records[scenario].append(
                np.quantile(
                    paired_draws,
                    upper_q,
                    axis=1,
                    method="linear",
                )
            )

        shared_django = {
            repository: coupled_django
            for repository in UNSEEN_REPOSITORIES
        }
        shared_exchangeable = {
            repository: coupled_exchangeable
            for repository in UNSEEN_REPOSITORIES
        }
        neutral = {
            repository: np.zeros((current_batch, 2), dtype=np.float64)
            for repository in UNSEEN_REPOSITORIES
        }
        adverse = {
            repository: np.broadcast_to(p90, (current_batch, 2))
            for repository in UNSEEN_REPOSITORIES
        }
        evaluate_scenario(PRIMARY_SCENARIOS[0], shared_django)
        evaluate_scenario(PRIMARY_SCENARIOS[1], shared_exchangeable)
        evaluate_scenario(PRIMARY_SCENARIOS[2], neutral)
        evaluate_scenario(PRIMARY_SCENARIOS[3], adverse)
        evaluate_scenario(SENSITIVITY_SCENARIOS[0], independent_django)
        evaluate_scenario(
            SENSITIVITY_SCENARIOS[1], independent_exchangeable
        )

        completed = start + current_batch
        if progress is not None and (
            completed == outer_count or completed % 1000 == 0
        ):
            progress(f"transport power cohorts complete: {completed}")

    return {
        "seed_sequence": stream_manifest,
        "outer_cohorts": outer_count,
        "inner_bootstrap_draws_per_cohort": inner_count,
        "outer_batch_size": batch_count,
        "upper_quantile": upper_q,
        "quantile_method": "linear",
        "repository_order": list(REPOSITORIES),
        "confirmation_allocation": dict(ordered_allocation),
        "confirmation_task_count": sum(ordered_allocation.values()),
        "primary_unseen_repository_coupling": (
            "perfectly_shared_repository_level_latent_profile"
        ),
        "sensitivity_unseen_repository_coupling": (
            "independent_repository_level_latent_profiles"
        ),
        "empirical_marginal_p90_adverse_profile": {
            METRIC_KEYS[0]: float(p90[0]),
            METRIC_KEYS[1]: float(p90[1]),
        },
        "primary_scenarios": {
            scenario: _gate_summary(
                np.concatenate(upper_records[scenario], axis=0),
                one_sided_alpha=alpha,
            )
            for scenario in PRIMARY_SCENARIOS
        },
        "independent_unseen_repository_sensitivities": {
            scenario: _gate_summary(
                np.concatenate(upper_records[scenario], axis=0),
                one_sided_alpha=alpha,
            )
            for scenario in SENSITIVITY_SCENARIOS
        },
    }


def derive_structural_p90(
    profile_summary: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    structural = _mapping(
        config["structural_p90_contract"], "structural p90 contract"
    )
    known = _mapping(
        profile_summary.get("paired_point_differences"),
        "observed known-repository point",
    )
    p90 = _mapping(
        profile_summary.get("empirical_marginal_p90_adverse_profile"),
        "observed p90 profile",
    )
    expected_known = _mapping(
        structural["known_ten_repository_point"],
        "expected known-repository point",
    )
    expected_p90 = _mapping(
        structural["empirical_marginal_p90_profile"], "expected p90 profile"
    )
    _same_numbers(known, expected_known, "known-repository point")
    _same_numbers(p90, expected_p90, "empirical p90 profile")
    infinite = {
        key: (10.0 * float(known[key]) + 2.0 * float(p90[key])) / 12.0
        for key in METRIC_KEYS
    }
    expected_infinite = _mapping(
        structural["infinite_task_equal_twelve_repository_point"],
        "expected infinite-task point",
    )
    _same_numbers(infinite, expected_infinite, "infinite-task point")
    recovered_upper = dict(
        _mapping(
            structural["recovered_repository_gamma_upper_0_975"],
            "recovered repository-Gamma upper",
        )
    )
    strict_nll_pass = float(recovered_upper[METRIC_KEYS[0]]) < 0.0
    strict_brier_pass = float(recovered_upper[METRIC_KEYS[1]]) < 0.0
    _require(
        strict_nll_pass
        and not strict_brier_pass
        and structural["strict_brier_gate_passes"] is False
        and structural["more_within_repository_tasks_can_repair"] is False,
        "structural p90 fail-closed conclusion changed",
    )
    return {
        "known_ten_repository_point": dict(known),
        "empirical_marginal_p90_profile": dict(p90),
        "arithmetic": (
            "(10*known_equal_repository_point+2*p90_profile)/12"
        ),
        "infinite_task_equal_twelve_repository_point": infinite,
        "recovered_repository_gamma_upper_0_975": recovered_upper,
        "strict_nll_gate_passes": strict_nll_pass,
        "strict_brier_gate_passes": strict_brier_pass,
        "joint_gate_passes": strict_nll_pass and strict_brier_pass,
        "more_within_repository_tasks_can_repair": False,
    }


def build_transport_report(
    config: Mapping[str, Any],
    source_artifact: Mapping[str, Any],
    profiles: Mapping[str, np.ndarray],
    profile_summary: Mapping[str, Any],
    *,
    binding_hashes: Mapping[str, str],
    analyzer_sha256: str,
    outer_cohorts: int | None = None,
    inner_draws: int | None = None,
    batch_size: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    validated_config = validate_transport_config(config)
    validate_source_feasibility_artifact(source_artifact, validated_config)
    supply = derive_supply_feasibility(
        OFFICIAL_SUPPLY, PILOT_ALLOCATION, CONFIRMATION_ALLOCATION
    )
    _require(
        supply["qualified_shortfall"] == dict(QUALIFIED_SHORTFALL)
        and supply["qualified_source_supply_passes"] is False,
        "qualified source-supply fail-closed result changed",
    )
    structural = derive_structural_p90(profile_summary, validated_config)
    resampling = validated_config["resampling_contract"]
    actual_outer = (
        int(resampling["outer_cohorts"])
        if outer_cohorts is None
        else _strict_int(outer_cohorts, "outer override", minimum=1)
    )
    actual_inner = (
        int(resampling["inner_bootstrap_draws"])
        if inner_draws is None
        else _strict_int(inner_draws, "inner override", minimum=1)
    )
    actual_batch = (
        int(resampling["outer_batch_size"])
        if batch_size is None
        else _strict_int(batch_size, "batch override", minimum=1)
    )
    production_contract_executed = (
        actual_outer == int(resampling["outer_cohorts"])
        and actual_inner == int(resampling["inner_bootstrap_draws"])
        and actual_batch == int(resampling["outer_batch_size"])
    )
    simulation = simulate_transport_power(
        profiles,
        CONFIRMATION_ALLOCATION,
        seed=int(resampling["seed_sequence_entropy"]),
        outer_cohorts=actual_outer,
        inner_draws=actual_inner,
        batch_size=actual_batch,
        upper_quantile=float(resampling["upper_quantile"]),
        one_sided_alpha=float(
            validated_config["monte_carlo_decision"][
                "one_sided_alpha_per_scenario"
            ]
        ),
        progress=progress,
    )
    threshold = float(
        validated_config["monte_carlo_decision"][
            "minimum_acceptable_joint_power"
        ]
    )
    primary_lower_bounds = {
        scenario: simulation["primary_scenarios"][scenario]["joint_gate"][
            "exact_one_sided_clopper_pearson_lower"
        ]
        for scenario in PRIMARY_SCENARIOS
    }
    all_primary_pass = all(
        value >= threshold for value in primary_lower_bounds.values()
    )
    retained_envelope_pass = (
        all_primary_pass and structural["joint_gate_passes"]
    )
    labels: list[str] = []
    if not retained_envelope_pass:
        labels.append(EXPECTED_DECISION_LABELS[0])
    if not supply["qualified_source_supply_passes"]:
        labels.append(EXPECTED_DECISION_LABELS[1])
    _require(
        labels == list(EXPECTED_DECISION_LABELS),
        "fail-closed decision labels changed",
    )

    aggregate_profile_summary = {
        "known_row_count": profile_summary["known_row_count"],
        "task_profile_count": profile_summary["task_profile_count"],
        "repository_count": profile_summary["repository_count"],
        "task_profile_count_by_repository": dict(
            profile_summary["task_profile_count_by_repository"]
        ),
        "repository_improving_sign_count": dict(
            profile_summary["repository_improving_sign_count"]
        ),
        "paired_point_differences": dict(
            profile_summary["paired_point_differences"]
        ),
        "empirical_marginal_p90_adverse_profile": dict(
            profile_summary["empirical_marginal_p90_adverse_profile"]
        ),
    }
    policy = dict(validated_config["data_policy"])
    return {
        "schema_version": SCHEMA_VERSION,
        "id": OUTPUT_ID,
        "status": "completed_design_transport_power_not_confirmatory",
        "scope": "fixed_prediction_transport_and_source_supply_power_only",
        "decision": "NO_GO",
        "fail_closed_decision_labels": labels,
        "all_primary_scenario_lower_bounds_meet_threshold": all_primary_pass,
        "retained_transport_envelope_passes": retained_envelope_pass,
        "qualified_source_supply_passes": False,
        "minimum_acceptable_joint_power": threshold,
        "primary_scenario_joint_power_lower_bounds": primary_lower_bounds,
        "source_binding": {
            "transport_config_path": str(DEFAULT_CONFIG.relative_to(ROOT)),
            "transport_config_sha256": binding_hashes[
                "transport_config_sha256"
            ],
            "transport_analyzer_path": str(Path(__file__).relative_to(ROOT)),
            "transport_analyzer_sha256": analyzer_sha256,
            "design_artifact_path": str(
                DEFAULT_DESIGN_ARTIFACT.relative_to(ROOT)
            ),
            "design_artifact_sha256": binding_hashes[
                "design_artifact_sha256"
            ],
            "source_feasibility_artifact_path": str(
                DEFAULT_SOURCE_FEASIBILITY_ARTIFACT.relative_to(ROOT)
            ),
            "source_feasibility_artifact_sha256": binding_hashes[
                "source_feasibility_artifact_sha256"
            ],
            "source_feasibility_config_sha256": binding_hashes[
                "source_feasibility_config_sha256"
            ],
            "source_feasibility_analyzer_sha256": binding_hashes[
                "source_feasibility_analyzer_sha256"
            ],
            "paired_power_config_sha256": binding_hashes[
                "paired_power_config_sha256"
            ],
            "paired_power_analyzer_sha256": binding_hashes[
                "paired_power_analyzer_sha256"
            ],
            "source_complement_instance_count": source_artifact["complement"][
                "instance_count"
            ],
            "source_complement_identity_set_sha256": source_artifact[
                "complement"
            ]["instance_ids_set_sha256"],
        },
        "aggregate_task_profile_summary": aggregate_profile_summary,
        "two_stage_source_supply": supply,
        "structural_p90_analysis": structural,
        "transport_power": simulation,
        "production_resampling_contract_executed": (
            production_contract_executed
        ),
        "pilot_authorized": False,
        "confirmation_authorized": False,
        "fresh_cohort_selection_or_generation_authorized": False,
        "selection_performed": False,
        "selection_authorized": False,
        "generation_performed": False,
        "generation_authorized": False,
        "reserved_membership_accessed": False,
        "reserved_validation_accessed": False,
        "reserved_validation_allowed": False,
        "task_text_prompt_completion_patch_or_trajectory_fields_consumed": False,
        "raw_task_ids_emitted": False,
        "confirmatory_interpretation": False,
        "operational_reliability_claim": False,
        "independent_v4_development_result": False,
        "data_policy": policy,
        "runtime": {
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "bit_generator": "PCG64",
            "quantile_method": "linear",
        },
    }


def _forbidden_component(path: Path, forbidden: set[str]) -> bool:
    return any(
        token in part.casefold()
        for part in path.parts
        for token in forbidden
    )


def validate_cli_paths(
    *,
    config_path: Path,
    design_artifact_path: Path,
    source_feasibility_artifact_path: Path,
    output_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Path]:
    inputs = {
        "config": (config_path, DEFAULT_CONFIG),
        "design_artifact": (design_artifact_path, DEFAULT_DESIGN_ARTIFACT),
        "source_feasibility_artifact": (
            source_feasibility_artifact_path,
            DEFAULT_SOURCE_FEASIBILITY_ARTIFACT,
        ),
    }
    resolved: dict[str, Path] = {}
    for label, (observed, canonical) in inputs.items():
        _require(
            observed.is_file() and not observed.is_symlink(),
            f"{label} must be a regular non-symlink file",
        )
        observed_resolved = observed.resolve(strict=True)
        _require(
            observed_resolved == canonical.resolve(strict=True),
            f"{label} must use the canonical path",
        )
        resolved[label] = observed_resolved
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
        "transport power output already exists; no-clobber is mandatory",
    )
    checked_paths = [*resolved.values(), resolved_output]
    _require(
        not any(
            _forbidden_component(path, FORBIDDEN_PATH_TOKENS)
            for path in checked_paths
        ),
        "reserved or validation path components are forbidden",
    )
    resolved["output"] = resolved_output
    return resolved


def _write_json_no_clobber(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
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
        "--design-artifact", type=Path, default=DEFAULT_DESIGN_ARTIFACT
    )
    parser.add_argument(
        "--source-feasibility-artifact",
        type=Path,
        default=DEFAULT_SOURCE_FEASIBILITY_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = validate_cli_paths(
        config_path=args.config,
        design_artifact_path=args.design_artifact,
        source_feasibility_artifact_path=args.source_feasibility_artifact,
        output_path=args.output,
    )
    _require(np.__version__ == "2.5.1", "transport screen requires NumPy 2.5.1")
    _require(
        scipy.__version__ == "1.18.0",
        "transport screen requires SciPy 1.18.0",
    )

    # Every frozen byte dependency is checked before any JSON is parsed or the
    # bound paired-power implementation is imported and used.
    binding_hashes = verify_bound_files()
    config = validate_transport_config(
        _read_json(paths["config"], "transport power config")
    )
    source_artifact = validate_source_feasibility_artifact(
        _read_json(
            paths["source_feasibility_artifact"],
            "source-feasibility artifact",
        ),
        config,
    )
    profiles, profile_summary = load_frozen_task_profiles()
    analyzer_sha256 = _sha256_file(Path(__file__).resolve())
    report = build_transport_report(
        config,
        source_artifact,
        profiles,
        profile_summary,
        binding_hashes=binding_hashes,
        analyzer_sha256=analyzer_sha256,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    _write_json_no_clobber(paths["output"], report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "fail_closed_decision_labels": report[
                    "fail_closed_decision_labels"
                ],
                "output": str(paths["output"]),
                "reserved_validation_accessed": False,
                "selection_authorized": False,
                "generation_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
