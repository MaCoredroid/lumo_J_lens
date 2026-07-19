#!/usr/bin/env python3
"""Evaluate a repository-held-out, selectively abstaining SWE task-state readout.

The analyzer consumes one prompt bundle and the matching public Jacobian-lens
report.  That report contains both public-J and ordinary-logit layer readouts,
which keeps every comparison paired.  Model selection, feature scaling,
temperature calibration, and confidence-threshold selection are all performed
without using labels from the outer held-out repository.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/swe_task_state_interpreter_protocol.json"
DEFAULT_BEHAVIORAL_PROTOCOL = ROOT / "configs/swe_behavioral_readout_protocol.json"
TASK_STATE_READOUT_PATH = ROOT / "scripts/swe_task_state_readout.py"
VARIANTS = (
    "progress_only",
    "lexical_progress",
    "history_context",
    "ordinary_logit",
    "logit_context",
    "public_jacobian",
    "jacobian_context",
    "hybrid",
)
HISTORY_VARIANTS = (
    "history_context",
    "logit_context",
    "jacobian_context",
    "hybrid",
)


def _load_task_state_readout_module():
    name = "swe_task_state_readout"
    if name in sys.modules:
        return sys.modules[name]
    path = TASK_STATE_READOUT_PATH
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


READOUT = _load_task_state_readout_module()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be nonempty text")
    return value


def finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_sha(value: Any, label: str) -> str:
    result = nonempty_string(value, label)
    require(
        len(result) == 64 and all(character in "0123456789abcdef" for character in result),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return result


def _validate_optional_pin(value: Any, observed: str, label: str) -> None:
    if value is None:
        return
    require(_validate_sha(value, label) == observed, f"{label} differs from the supplied input")


def validate_protocol(
    value: Any,
    *,
    behavioral_protocol_value: Any,
    behavioral_protocol_sha256: str,
    prompt_sha256: str | None = None,
    report_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize the frozen interpreter protocol."""
    protocol = mapping(value, "task-state protocol")
    require(
        protocol.get("schema_version") == 1
        and protocol.get("id") == "swe-task-state-interpreter-v1"
        and protocol.get("decision_scope") == "development_screen_only",
        "task-state protocol identity changed",
    )
    behavioral = mapping(behavioral_protocol_value, "behavioral protocol")
    action_classes = sequence(behavioral.get("action_classes"), "behavioral action classes")
    class_ids = [nonempty_string(item.get("id"), "action class ID") for item in action_classes]
    require(len(class_ids) >= 2 and len(class_ids) == len(set(class_ids)), "action classes are invalid")
    token_ids_by_class: dict[str, list[int]] = {}
    token_texts_by_class: dict[str, list[str]] = {}
    for class_value in action_classes:
        class_id = str(class_value["id"])
        tokens = sequence(class_value.get("tokens"), f"{class_id} tokens")
        token_ids = [integer(item.get("token_id"), f"{class_id} token ID") for item in tokens]
        token_texts = [nonempty_string(item.get("text"), f"{class_id} token text") for item in tokens]
        require(bool(token_ids) and len(token_ids) == len(set(token_ids)), f"{class_id} tokens are invalid")
        require(len(token_texts) == len(set(token_texts)), f"{class_id} token texts are invalid")
        token_ids_by_class[class_id] = token_ids
        token_texts_by_class[class_id] = token_texts
    behavioral_layers = [
        integer(item, "behavioral layer")
        for item in sequence(
            mapping(behavioral.get("fixed_layer_band"), "behavioral layer band").get("layers"),
            "behavioral layers",
        )
    ]

    pins = mapping(protocol.get("input_pins"), "input pins")
    require(
        _validate_sha(pins.get("behavioral_protocol_sha256"), "behavioral protocol pin")
        == behavioral_protocol_sha256,
        "behavioral protocol pin differs from the supplied protocol",
    )
    if prompt_sha256 is not None:
        _validate_optional_pin(pins.get("prompt_bundle_sha256"), prompt_sha256, "prompt pin")
    if report_sha256 is not None:
        _validate_optional_pin(pins.get("public_report_sha256"), report_sha256, "report pin")
    model_pin = mapping(pins.get("model"), "model pin")
    lens_pin = mapping(pins.get("public_lens"), "public lens pin")
    runtime_pin = mapping(pins.get("replay_runtime"), "replay runtime pin")
    require(
        model_pin
        == {
            "repo_id": "nvidia/Qwen3.6-27B-NVFP4",
            "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
            "config_sha256": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
            "index_sha256": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
        }
        and lens_pin
        == {
            "repo_id": "neuronpedia/jacobian-lens",
            "revision": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
            "sha256": "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1",
            "n_prompts": 1000,
        }
        and runtime_pin
        == {
            "enforce_eager": True,
            "mtp_enabled": False,
            "max_model_len": 65536,
            "max_num_batched_tokens": 4096,
            "mamba_block_size": 1024,
            "kv_cache_dtype": "fp8_e4m3",
            "kv_offloading_size": 8.0,
            "kv_offloading_backend": "native",
            "stream_final_only": True,
        },
        "task-state model, lens, or replay pin changed",
    )

    scope = mapping(protocol.get("scope"), "scope")
    require(scope.get("class_ids_in_order") == class_ids, "task-state class order changed")
    require(scope.get("variants") == list(VARIANTS), "task-state variants changed")

    features = mapping(protocol.get("feature_contract"), "feature contract")
    layers = [integer(item, "feature layer") for item in sequence(features.get("layers"), "feature layers")]
    require(
        layers == behavioral_layers
        and features.get("feature_order") == "layer-major_then_class-order"
        and features.get("within_class_reduction")
        == "logmeanexp_over_declared_token_logits"
        and features.get("progress_features")
        == ["task_request_index", "log1p_task_request_index"]
        and features.get("lexical_features")
        == [
            "log1p_exact_token_occurrence_count_per_class",
            "normalized_token_recency_per_class",
            "log1p_exact_string_occurrence_count_per_class",
            "normalized_string_recency_per_class",
        ]
        and features.get("history_features")
        == [
            "log1p_cumulative_prior_action_count_per_class",
            "previous_action_one_hot_per_class",
            "log1p_cumulative_unknown_prior_action_count",
            "previous_action_unknown",
            "has_edited",
            "has_validated",
            "turns_since_edit_or_minus_one",
            "turns_since_validate_or_minus_one",
        ]
        and features.get("history_requires_complete_consecutive_probe_bundle") is True
        and features.get("future_trajectory_fields_forbidden") is True,
        "task-state feature contract changed",
    )
    variant_blocks = mapping(features.get("variant_blocks"), "variant blocks")
    require(
        variant_blocks
        == {
            "progress_only": ["progress"],
            "lexical_progress": ["lexical", "progress"],
            "history_context": ["history", "lexical", "progress"],
            "ordinary_logit": ["ordinary_logit"],
            "logit_context": ["ordinary_logit", "history", "lexical", "progress"],
            "public_jacobian": ["public_jacobian"],
            "jacobian_context": [
                "public_jacobian",
                "history",
                "lexical",
                "progress",
            ],
            "hybrid": [
                "public_jacobian",
                "ordinary_logit",
                "history",
                "lexical",
                "progress",
            ],
        },
        "task-state feature blocks changed",
    )

    eligibility = mapping(protocol.get("eligibility"), "eligibility contract")
    stable = mapping(eligibility.get("numerical_stability"), "numerical stability")
    require(
        eligibility.get("action_label_status") == "available"
        and isinstance(eligibility.get("require_primary_selection"), bool)
        and eligibility.get("require_finite_action_scores") is True
        and stable.get("final_layer_top1_matches_greedy") is True
        and stable.get("final_norm_within_tolerance") is True
        and stable.get("final_logits_top_k_prefix_token_ids_match") is True,
        "task-state eligibility contract changed",
    )
    stable_rms = finite(stable.get("final_logits_rms_error_maximum_inclusive"), "stable RMS maximum")
    stable_max = finite(stable.get("final_logits_max_abs_error_maximum_inclusive"), "stable max-abs maximum")
    require(stable_rms >= 0.0 and stable_max >= 0.0, "stability thresholds must be nonnegative")

    model = mapping(protocol.get("model_contract"), "model contract")
    c_grid = [finite(item, "regularization C") for item in sequence(model.get("regularization_C_grid"), "C grid")]
    require(
        model.get("family") == "multinomial_logistic_regression"
        and model.get("penalty") == "L2"
        and model.get("solver") == "lbfgs"
        and model.get("class_weight") == "balanced_from_current_training_split_only"
        and model.get("fit_intercept") is True
        and model.get("scaler_fit") == "current_training_split_only"
        and bool(c_grid)
        and all(item > 0.0 for item in c_grid),
        "task-state model contract changed",
    )
    maximum_iterations = integer(model.get("maximum_iterations"), "maximum iterations", minimum=1)

    outer = mapping(protocol.get("outer_evaluation"), "outer evaluation")
    inner = mapping(protocol.get("inner_model_selection"), "inner model selection")
    require(
        outer.get("algorithm") == "leave_one_repository_out"
        and outer.get("heldout_labels_used_for_fit_scaling_calibration_or_threshold_selection")
        is False
        and inner.get("algorithm") == "leave_one_repository_out_within_outer_training"
        and inner.get("selection_metric") == "balanced_accuracy"
        and inner.get("tie_break") == "smallest_C"
        and inner.get("complete_inner_prediction_coverage_required") is True,
        "task-state grouped evaluation contract changed",
    )
    minimum_inner = integer(inner.get("minimum_valid_inner_folds"), "minimum inner folds", minimum=2)

    calibration = mapping(protocol.get("calibration"), "calibration contract")
    temperatures = [finite(item, "temperature") for item in sequence(calibration.get("temperature_grid"), "temperature grid")]
    require(
        calibration.get("algorithm") == "inner_repository_crossfit_temperature_grid"
        and calibration.get("selection_metric") == "multiclass_negative_log_likelihood"
        and calibration.get("tie_break") == "closest_to_one_then_smallest"
        and calibration.get("outer_heldout_labels_used") is False
        and bool(temperatures)
        and all(item > 0.0 for item in temperatures),
        "task-state calibration contract changed",
    )

    abstention = mapping(protocol.get("abstention"), "abstention contract")
    thresholds = [finite(item, "confidence threshold") for item in sequence(abstention.get("confidence_threshold_grid"), "confidence threshold grid")]
    selection_contract = mapping(abstention.get("selection"), "threshold selection")
    require(
        abstention.get("confidence") == "maximum_temperature_scaled_class_probability"
        and abstention.get("accept_when") == "confidence_greater_than_or_equal_to_threshold"
        and abstention.get("sweep_role") == "descriptive_fixed_grid"
        and selection_contract.get("algorithm") == "maximum_coverage_meeting_floors"
        and selection_contract.get("tie_break") == "lowest_threshold"
        and selection_contract.get("fallback")
        == "maximum_balanced_accepted_recall_then_accuracy_then_coverage"
        and selection_contract.get("floor_weighting") == "task_equal_primary"
        and selection_contract.get("outer_heldout_labels_used") is False
        and bool(thresholds)
        and thresholds == sorted(set(thresholds))
        and all(0.0 <= item <= 1.0 for item in thresholds),
        "task-state abstention contract changed",
    )
    threshold_floors = {
        "accepted_accuracy_minimum": finite(selection_contract.get("accepted_accuracy_minimum"), "accepted accuracy floor"),
        "balanced_accepted_recall_minimum": finite(selection_contract.get("balanced_accepted_recall_minimum"), "balanced accepted recall floor"),
        "coverage_minimum": finite(selection_contract.get("coverage_minimum"), "coverage floor"),
        "minimum_accepted_rows_per_class": integer(selection_contract.get("minimum_accepted_rows_per_class"), "minimum accepted rows per class", minimum=1),
    }
    require(
        all(0.0 <= threshold_floors[key] <= 1.0 for key in (
            "accepted_accuracy_minimum",
            "balanced_accepted_recall_minimum",
            "coverage_minimum",
        )),
        "threshold-selection floors must be probabilities",
    )

    metrics = mapping(protocol.get("metrics"), "metric contract")
    ece_bins = integer(metrics.get("ece_equal_width_bin_count"), "ECE bin count", minimum=2)
    require(
        metrics.get("probability_metrics")
        == ["multiclass_negative_log_likelihood", "multiclass_brier", "top_label_ece"]
        and metrics.get("selective_metrics")
        == [
            "coverage",
            "accepted_accuracy",
            "balanced_accepted_recall",
            "per_class_accepted_coverage",
        ],
        "task-state metric contract changed",
    )

    bootstrap = mapping(protocol.get("bootstrap"), "bootstrap contract")
    bootstrap_samples = integer(bootstrap.get("samples"), "bootstrap samples", minimum=1)
    bootstrap_seed = integer(bootstrap.get("seed"), "bootstrap seed")
    confidence_level = finite(bootstrap.get("confidence_level"), "bootstrap confidence level")
    minimum_valid_fraction = finite(bootstrap.get("minimum_valid_fraction"), "minimum bootstrap valid fraction")
    require(
        bootstrap.get("algorithm") == "hierarchical_repository_then_task_percentile_v1"
        and bootstrap.get("row_resampling_forbidden") is True
        and bootstrap.get("models_refit_inside_bootstrap") is False
        and bootstrap.get("interval_interpretation")
        == "conditional_on_frozen_out_of_repository_predictions_excludes_fit_and_selection_uncertainty"
        and bootstrap.get("operational_reliability_proof") is False
        and 0.0 < confidence_level < 1.0
        and 0.0 < minimum_valid_fraction <= 1.0,
        "task-state bootstrap contract changed",
    )

    gates = mapping(protocol.get("reliability_gates"), "reliability gates")
    support_gates = mapping(gates.get("support"), "support gates")
    normalized_support = {
        "minimum_rows": integer(support_gates.get("minimum_rows"), "minimum rows", minimum=1),
        "minimum_tasks": integer(support_gates.get("minimum_tasks"), "minimum tasks", minimum=1),
        "minimum_repositories": integer(support_gates.get("minimum_repositories"), "minimum repositories", minimum=2),
    }
    absolute_gates = _validate_metric_gates(
        sequence(gates.get("absolute"), "absolute reliability gates"), paired=False
    )
    paired_gates = _validate_metric_gates(
        sequence(gates.get("paired"), "paired reliability gates"), paired=True
    )
    require(
        isinstance(gates.get("require_all_outer_folds_available"), bool)
        and isinstance(gates.get("require_all_calibration_targets_met"), bool),
        "reliability boolean gates are invalid",
    )
    calibration_target_variants = [
        nonempty_string(item, "calibration target variant")
        for item in sequence(
            gates.get("calibration_target_variants"),
            "calibration target variants",
        )
    ]
    require(
        bool(calibration_target_variants)
        and len(calibration_target_variants) == len(set(calibration_target_variants))
        and all(item in VARIANTS for item in calibration_target_variants),
        "calibration target variants are invalid",
    )

    return {
        "value": protocol,
        "class_ids": class_ids,
        "token_ids_by_class": token_ids_by_class,
        "token_texts_by_class": token_texts_by_class,
        "report_pins": {
            "model": dict(model_pin),
            "public_lens": dict(lens_pin),
            "runtime": dict(runtime_pin),
        },
        "layers": layers,
        "variant_blocks": dict(variant_blocks),
        "eligibility": {
            "require_primary": bool(eligibility["require_primary_selection"]),
            "stable_rms": stable_rms,
            "stable_max": stable_max,
        },
        "model": {"c_grid": c_grid, "maximum_iterations": maximum_iterations},
        "evaluation": {"minimum_inner": minimum_inner},
        "calibration": {"temperatures": temperatures},
        "abstention": {"thresholds": thresholds, **threshold_floors},
        "metrics": {"ece_bins": ece_bins},
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "minimum_valid_fraction": minimum_valid_fraction,
            "interval_interpretation": bootstrap["interval_interpretation"],
            "operational_reliability_proof": False,
        },
        "gates": {
            "support": normalized_support,
            "absolute": absolute_gates,
            "paired": paired_gates,
            "require_all_outer_folds_available": gates["require_all_outer_folds_available"],
            "require_all_calibration_targets_met": gates["require_all_calibration_targets_met"],
            "calibration_target_variants": calibration_target_variants,
        },
    }


def _validate_metric_gates(values: Sequence[Any], *, paired: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    allowed_metrics = {
        "task_macro_overall_accuracy",
        "task_macro_balanced_accuracy",
        "task_macro_selected_coverage",
        "task_macro_selective_accuracy",
        "task_macro_selective_correct_coverage",
        "overall_accuracy",
        "balanced_accuracy",
        "multiclass_negative_log_likelihood",
        "multiclass_brier",
        "top_label_ece",
        "selected_coverage",
        "selected_accepted_accuracy",
        "selected_balanced_accepted_recall",
        "selected_minimum_per_class_accepted_coverage",
    }
    for index, raw in enumerate(values):
        gate = mapping(raw, f"reliability gate {index}")
        identifier = nonempty_string(gate.get("id"), "reliability gate ID")
        require(identifier not in identifiers, "duplicate reliability gate ID")
        identifiers.add(identifier)
        metric = nonempty_string(gate.get("metric"), "reliability metric")
        require(metric in allowed_metrics, f"unsupported reliability metric: {metric}")
        bound = nonempty_string(gate.get("bound"), "reliability bound")
        operator = nonempty_string(gate.get("operator"), "reliability operator")
        require(
            bound in {"point", "bootstrap_lower", "bootstrap_upper"}
            and operator
            in {
                "minimum_inclusive",
                "minimum_exclusive",
                "maximum_inclusive",
                "maximum_exclusive",
            },
            "reliability bound or operator is invalid",
        )
        normalized = {
            "id": identifier,
            "metric": metric,
            "bound": bound,
            "operator": operator,
            "value": finite(gate.get("value"), "reliability threshold"),
        }
        if paired:
            candidate = nonempty_string(gate.get("candidate"), "candidate variant")
            reference = nonempty_string(gate.get("reference"), "reference variant")
            require(candidate in VARIANTS and reference in VARIANTS and candidate != reference, "paired gate variants are invalid")
            normalized.update({"candidate": candidate, "reference": reference})
        else:
            variant = nonempty_string(gate.get("variant"), "gate variant")
            require(variant in VARIANTS, "absolute gate variant is invalid")
            normalized["variant"] = variant
        result.append(normalized)
    return result


def logmeanexp(values: Sequence[float]) -> float:
    require(bool(values), "cannot reduce an empty score group")
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values) / len(values))


def _numerically_stable(experiment: Mapping[str, Any], protocol: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if experiment.get("final_layer_top1_matches_greedy") is not True:
        reasons.append("final_layer_top1_mismatch")
    norm = mapping(experiment.get("final_norm_reconstruction"), "final norm reconstruction")
    if norm.get("within_tolerance") is not True:
        reasons.append("final_norm_outside_tolerance")
    logits = mapping(experiment.get("final_logits_reconstruction"), "final logits reconstruction")
    if logits.get("top_k_prefix_token_ids_match") is not True:
        reasons.append("final_logits_top_k_prefix_mismatch")
    if finite(logits.get("rms_error"), "final-logit RMS error") > float(protocol["stable_rms"]):
        reasons.append("final_logits_rms_error")
    if finite(logits.get("max_abs_error"), "final-logit max-abs error") > float(protocol["stable_max"]):
        reasons.append("final_logits_max_abs_error")
    return not reasons, reasons


def _layer_class_features(
    experiment: Mapping[str, Any],
    *,
    layers: Sequence[int],
    class_ids: Sequence[str],
    token_ids_by_class: Mapping[str, Sequence[int]],
    method: str,
    expected_token_position: int,
) -> list[float]:
    layer_values = sequence(experiment.get("layers"), "report layers")
    by_layer = {integer(item.get("layer"), "report layer"): item for item in layer_values}
    require(len(by_layer) == len(layer_values), "report contains duplicate layers")
    result: list[float] = []
    readout_key = "logit_lens" if method == "ordinary_logit" else "jacobian_lens"
    for layer_id in layers:
        layer = mapping(by_layer.get(layer_id), f"layer {layer_id}")
        positions = sequence(layer.get("positions"), f"layer {layer_id} positions")
        require(len(positions) == 1, "task-state reports must contain exactly one capture position")
        position = mapping(positions[0], "layer position")
        require(
            integer(position.get("token_position"), "captured token position")
            == expected_token_position,
            "task-state readout was not captured at the final prompt token",
        )
        readout = mapping(position.get(readout_key), readout_key)
        scored = sequence(readout.get("scored_tokens"), f"{readout_key} scored tokens")
        scores_by_token: dict[int, float] = {}
        for item in scored:
            token_id = integer(item.get("token_id"), "scored token ID")
            require(token_id not in scores_by_token, "scored token IDs are duplicated")
            scores_by_token[token_id] = finite(item.get("score"), "scored token logit")
        for class_id in class_ids:
            token_ids = token_ids_by_class[class_id]
            require(
                all(token_id in scores_by_token for token_id in token_ids),
                f"{method}/layer {layer_id}/{class_id} lacks declared action tokens",
            )
            result.append(logmeanexp([scores_by_token[token_id] for token_id in token_ids]))
    require(
        len(result) == len(layers) * len(class_ids) and all(math.isfinite(item) for item in result),
        f"{method} feature vector is invalid",
    )
    return result


def _recency_fraction(last_index: int, length: int) -> float:
    if last_index < 0 or length <= 0:
        return 1.0
    return (length - 1 - last_index) / max(1, length)


def _lexical_features(
    prompt: Mapping[str, Any],
    *,
    class_ids: Sequence[str],
    token_ids_by_class: Mapping[str, Sequence[int]],
    token_texts_by_class: Mapping[str, Sequence[str]],
) -> list[float]:
    token_ids = [integer(item, "prompt token ID") for item in sequence(prompt.get("token_ids"), "prompt token IDs")]
    text = nonempty_string(prompt.get("text"), "rendered prompt")
    token_positions: dict[int, list[int]] = {}
    for position, token_id in enumerate(token_ids):
        token_positions.setdefault(token_id, []).append(position)
    result: list[float] = []
    for class_id in class_ids:
        positions = [
            position
            for token_id in token_ids_by_class[class_id]
            for position in token_positions.get(token_id, [])
        ]
        token_count = len(positions)
        token_recency = _recency_fraction(max(positions, default=-1), len(token_ids))
        texts = token_texts_by_class[class_id]
        string_count = sum(text.count(value) for value in texts)
        string_last = max(
            (
                start + len(value) - 1 if (start := text.rfind(value)) >= 0 else -1
                for value in texts
            ),
            default=-1,
        )
        string_recency = _recency_fraction(string_last, len(text))
        result.extend(
            [
                math.log1p(token_count),
                token_recency,
                math.log1p(string_count),
                string_recency,
            ]
        )
    return result


def _causal_history_features(
    prompts: Sequence[Mapping[str, Any]], class_ids: Sequence[str]
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Derive history only when the bundle contains every declared prior request."""
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for prompt in prompts:
        metadata = mapping(prompt.get("metadata"), "prompt metadata")
        task_id = nonempty_string(mapping(metadata.get("task"), "prompt task").get("instance_id"), "task ID")
        by_task.setdefault(task_id, []).append(prompt)
    result: dict[str, list[float]] = {}
    task_records: dict[str, Any] = {}
    for task_id, task_prompts in sorted(by_task.items()):
        ordered = sorted(
            task_prompts,
            key=lambda item: integer(
                mapping(mapping(item.get("metadata"), "prompt metadata").get("selection"), "selection").get("task_request_index"),
                "task request index",
                minimum=1,
            ),
        )
        task_metadata = mapping(mapping(ordered[0].get("metadata"), "prompt metadata").get("task"), "task")
        declared = [
            integer(item, "probeable request index", minimum=1)
            for item in sequence(task_metadata.get("probeable_request_indices"), "probeable request indices")
        ]
        observed = [
            integer(
                mapping(mapping(item.get("metadata"), "prompt metadata").get("selection"), "selection").get("task_request_index"),
                "task request index",
                minimum=1,
            )
            for item in ordered
        ]
        declared_is_consecutive = declared == list(range(1, len(declared) + 1))
        complete = (
            declared_is_consecutive
            and observed == declared
            and len(observed) == len(set(observed))
        )
        task_records[task_id] = {
            "complete_consecutive_probe_bundle": complete,
            "declared_probeable_indices_are_consecutive_from_one": declared_is_consecutive,
            "observed_request_indices": observed,
            "declared_probeable_request_indices": declared,
        }
        if not complete:
            continue
        counts = {class_id: 0 for class_id in class_ids}
        previous: str | None = None
        previous_unknown = False
        unknown_count = 0
        last_seen: dict[str, int | None] = {class_id: None for class_id in class_ids}
        for prompt, request_index in zip(ordered, observed, strict=True):
            previous_one_hot = [float(previous == class_id) for class_id in class_ids]
            history = [math.log1p(counts[class_id]) for class_id in class_ids]
            history.extend(previous_one_hot)
            history.extend([math.log1p(unknown_count), float(previous_unknown)])
            history.extend(
                [
                    float(counts.get("edit", 0) > 0),
                    float(counts.get("validate", 0) > 0),
                    float(request_index - int(last_seen["edit"])) if "edit" in last_seen and last_seen["edit"] is not None else -1.0,
                    float(request_index - int(last_seen["validate"])) if "validate" in last_seen and last_seen["validate"] is not None else -1.0,
                ]
            )
            result[str(prompt["id"])] = history
            metadata = mapping(prompt.get("metadata"), "prompt metadata")
            action = mapping(mapping(metadata.get("labels"), "labels").get("action"), "action label")
            action_id = action.get("class_id")
            if action.get("status") != "available" or action_id not in class_ids:
                unknown_count += 1
                previous = None
                previous_unknown = True
                continue
            counts[str(action_id)] += 1
            previous = str(action_id)
            previous_unknown = False
            last_seen[str(action_id)] = request_index
    return result, {
        "task_count": len(by_task),
        "complete_history_task_count": sum(
            bool(item["complete_consecutive_probe_bundle"]) for item in task_records.values()
        ),
        "tasks": task_records,
    }


def _validate_report_provenance(
    report: Mapping[str, Any], *, protocol: Mapping[str, Any]
) -> None:
    pins = protocol["report_pins"]
    require(
        report.get("schema_version") == 3
        and report.get("score_encoding") == "unrounded-float32",
        "public report schema or score encoding changed",
    )
    assertions = mapping(report.get("assertions"), "report assertions")
    require(
        assertions.get("lens_hash_matches") is True
        and assertions.get("lens_metadata_matches") is True
        and assertions.get("model_architecture_matches") is True,
        "public report did not verify its model or lens artifact",
    )
    model = mapping(report.get("model"), "report model")
    require(
        all(model.get(key) == value for key, value in pins["model"].items()),
        "public report model pin differs",
    )
    lens = mapping(report.get("lens"), "report lens")
    require(
        all(lens.get(key) == value for key, value in pins["public_lens"].items()),
        "public report lens pin differs",
    )
    runtime = mapping(report.get("runtime"), "report runtime")
    require(
        all(runtime.get(key) == value for key, value in pins["runtime"].items()),
        "public report replay runtime differs",
    )


def extract_rows(
    prompt_bundle_value: Any,
    report_value: Any,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    prompts = sequence(prompt_bundle_value, "prompt bundle")
    report = mapping(report_value, "public report")
    _validate_report_provenance(report, protocol=protocol)
    experiments = sequence(report.get("experiments"), "report experiments")
    require(len(prompts) == len(experiments), "prompt/report row counts differ")
    prompt_ids = [nonempty_string(item.get("id"), "prompt ID") for item in prompts]
    experiment_ids = [nonempty_string(item.get("id"), "experiment ID") for item in experiments]
    require(len(prompt_ids) == len(set(prompt_ids)), "prompt IDs are duplicated")
    require(prompt_ids == experiment_ids, "prompt/report IDs or order differ")

    history_by_id, history_coverage = _causal_history_features(prompts, protocol["class_ids"])
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    exclusion_counts: dict[str, int] = {}
    for prompt, experiment in zip(prompts, experiments, strict=True):
        prompt_id = str(prompt["id"])
        require(
            experiment.get("prompt") == prompt.get("text")
            and experiment.get("prompt_token_ids") == prompt.get("token_ids")
            and experiment.get("metadata") == prompt.get("metadata"),
            f"{prompt_id} report payload is not bound to the supplied prompt",
        )
        prompt_token_ids = sequence(prompt.get("token_ids"), f"{prompt_id} prompt token IDs")
        require(bool(prompt_token_ids), f"{prompt_id} prompt token IDs are empty")
        expected_token_position = len(prompt_token_ids) - 1
        require(
            experiment.get("capture_positions_resolved") == [expected_token_position],
            f"{prompt_id} was not captured only at the final prompt token",
        )
        scored_vocabulary = mapping(
            experiment.get("scored_vocabulary"), f"{prompt_id} scored vocabulary"
        )
        require(
            scored_vocabulary.get("token_ids") == prompt.get("score_token_ids"),
            f"{prompt_id} scored vocabulary differs from the prompt contract",
        )
        metadata = mapping(prompt.get("metadata"), f"{prompt_id} metadata")
        labels = mapping(metadata.get("labels"), f"{prompt_id} labels")
        action = mapping(labels.get("action"), f"{prompt_id} action label")
        reason: str | None = None
        details: list[str] = []
        if action.get("status") != "available" or action.get("class_id") not in protocol["class_ids"]:
            reason = "action_label_unavailable"
        selection = mapping(metadata.get("selection"), f"{prompt_id} selection")
        if reason is None and protocol["eligibility"]["require_primary"] and selection.get("primary_for_action_evaluation") is not True:
            reason = "not_primary_selection"
        stable, stable_reasons = _numerically_stable(experiment, protocol["eligibility"])
        if reason is None and not stable:
            reason = "numerically_unstable"
            details = stable_reasons
        if reason is not None:
            exclusions.append({"row_id": prompt_id, "reason": reason, "details": details})
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
            continue

        task = mapping(metadata.get("task"), f"{prompt_id} task")
        task_id = nonempty_string(task.get("instance_id"), "task instance ID")
        repository = nonempty_string(task.get("repo"), "task repository")
        task_request_index = integer(selection.get("task_request_index"), "task request index", minimum=1)
        cohort_value = metadata.get("cohort")
        cohort_id = "unspecified"
        if isinstance(cohort_value, dict) and isinstance(cohort_value.get("id"), str):
            cohort_id = str(cohort_value["id"])
        progress = [float(task_request_index), math.log1p(task_request_index)]
        lexical = _lexical_features(
            prompt,
            class_ids=protocol["class_ids"],
            token_ids_by_class=protocol["token_ids_by_class"],
            token_texts_by_class=protocol["token_texts_by_class"],
        )
        ordinary = _layer_class_features(
            experiment,
            layers=protocol["layers"],
            class_ids=protocol["class_ids"],
            token_ids_by_class=protocol["token_ids_by_class"],
            method="ordinary_logit",
            expected_token_position=expected_token_position,
        )
        public = _layer_class_features(
            experiment,
            layers=protocol["layers"],
            class_ids=protocol["class_ids"],
            token_ids_by_class=protocol["token_ids_by_class"],
            method="public_jacobian",
            expected_token_position=expected_token_position,
        )
        history = history_by_id.get(prompt_id)
        blocks = {
            "progress": progress,
            "lexical": lexical,
            "history": history,
            "ordinary_logit": ordinary,
            "public_jacobian": public,
        }
        features: dict[str, list[float] | None] = {}
        for variant in VARIANTS:
            block_names = protocol["variant_blocks"][variant]
            if any(blocks[block] is None for block in block_names):
                features[variant] = None
            else:
                features[variant] = [
                    value
                    for block in block_names
                    for value in (blocks[block] or [])
                ]
        rows.append(
            {
                "row_id": prompt_id,
                "task_id": task_id,
                "repo": repository,
                "cohort_id": cohort_id,
                "label": str(action["class_id"]),
                "task_request_index": task_request_index,
                "checkpoint_ordinal": selection.get("checkpoint_ordinal"),
                "causal_history_available": history is not None,
                "features": features,
            }
        )
    return {
        "rows": rows,
        "eligibility": {
            "prompt_count": len(prompts),
            "eligible_row_count": len(rows),
            "excluded_row_count": len(exclusions),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "exclusions": exclusions,
            "default_track": "numerically_stable_labeled_rows",
            "causal_history": history_coverage,
        },
    }


def _class_support(rows: Sequence[Mapping[str, Any]], class_ids: Sequence[str]) -> dict[str, int]:
    return {class_id: sum(str(row["label"]) == class_id for row in rows) for class_id in class_ids}


def _all_classes(rows: Sequence[Mapping[str, Any]], class_ids: Sequence[str]) -> bool:
    return all(value > 0 for value in _class_support(rows, class_ids).values())


def _matrix(
    rows: Sequence[Mapping[str, Any]], variant: str, class_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([mapping(row["features"], "row features")[variant] for row in rows], dtype=np.float64)
    y = np.asarray([class_ids.index(str(row["label"])) for row in rows], dtype=np.int64)
    require(x.ndim == 2 and len(x) == len(rows) and x.shape[1] > 0 and np.all(np.isfinite(x)), "feature matrix is invalid")
    return x, y


def _prediction_records(
    rows: Sequence[Mapping[str, Any]], probabilities: np.ndarray, class_ids: Sequence[str]
) -> list[dict[str, Any]]:
    require(probabilities.shape == (len(rows), len(class_ids)), "prediction shape is invalid")
    result: list[dict[str, Any]] = []
    for row, probability in zip(rows, probabilities, strict=True):
        values = [float(item) for item in probability]
        predicted_index = int(np.argmax(probability))
        record = {
            key: row[key]
            for key in (
                "row_id",
                "task_id",
                "repo",
                "cohort_id",
                "label",
                "task_request_index",
                "checkpoint_ordinal",
            )
            if key in row
        }
        result.append(
            {
                **record,
                "class_ids": list(class_ids),
                "probabilities": values,
                "prediction": class_ids[predicted_index],
                "confidence": max(values),
            }
        )
    return result


def _apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    require(temperature > 0.0, "temperature must be positive")
    logits = np.log(np.clip(probabilities, 1e-300, 1.0)) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    result = np.exp(logits)
    result /= np.sum(result, axis=1, keepdims=True)
    require(np.all(np.isfinite(result)), "temperature scaling produced nonfinite probabilities")
    return result


def _probability_metrics(
    predictions: Sequence[Mapping[str, Any]], class_ids: Sequence[str], *, ece_bins: int
) -> dict[str, Any]:
    if not predictions:
        return {
            "row_count": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "per_class_recall": {class_id: None for class_id in class_ids},
            "multiclass_negative_log_likelihood": None,
            "multiclass_brier": None,
            "top_label_ece": None,
        }
    y = np.asarray([class_ids.index(str(row["label"])) for row in predictions], dtype=np.int64)
    probabilities = np.asarray([row["probabilities"] for row in predictions], dtype=np.float64)
    predicted = np.argmax(probabilities, axis=1)
    correct = predicted == y
    recalls: dict[str, float | None] = {}
    for class_index, class_id in enumerate(class_ids):
        mask = y == class_index
        recalls[class_id] = float(np.mean(correct[mask])) if np.any(mask) else None
    balanced = None if any(value is None for value in recalls.values()) else float(math.fsum(float(value) for value in recalls.values()) / len(class_ids))
    true_probabilities = np.clip(probabilities[np.arange(len(y)), y], 1e-300, 1.0)
    one_hot = np.eye(len(class_ids), dtype=np.float64)[y]
    confidence = np.max(probabilities, axis=1)
    ece = 0.0
    for index in range(ece_bins):
        lower = index / ece_bins
        upper = (index + 1) / ece_bins
        mask = (confidence >= lower) & (confidence < upper if index + 1 < ece_bins else confidence <= upper)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
    return {
        "row_count": len(predictions),
        "accuracy": float(np.mean(correct)),
        "balanced_accuracy": balanced,
        "per_class_recall": recalls,
        "multiclass_negative_log_likelihood": float(-np.mean(np.log(true_probabilities))),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "top_label_ece": ece,
    }


def _selective_metrics(
    predictions: Sequence[Mapping[str, Any]],
    class_ids: Sequence[str],
    *,
    ece_bins: int,
    threshold: float | None = None,
) -> dict[str, Any]:
    if threshold is None:
        accepted = [bool(row.get("accepted")) for row in predictions]
    else:
        accepted = [finite(row.get("confidence"), "prediction confidence") >= threshold for row in predictions]
    accepted_rows = [row for row, keep in zip(predictions, accepted, strict=True) if keep]
    per_class: dict[str, Any] = {}
    conditional: list[float] = []
    selective_correct_coverage: list[float] = []
    for class_id in class_ids:
        class_rows = [row for row in predictions if row["label"] == class_id]
        class_accepted = [row for row, keep in zip(predictions, accepted, strict=True) if keep and row["label"] == class_id]
        correct = sum(row["prediction"] == class_id for row in class_accepted)
        conditional_accuracy = correct / len(class_accepted) if class_accepted else None
        accepted_coverage = len(class_accepted) / len(class_rows) if class_rows else None
        correct_coverage = correct / len(class_rows) if class_rows else None
        if conditional_accuracy is not None:
            conditional.append(conditional_accuracy)
        if correct_coverage is not None:
            selective_correct_coverage.append(correct_coverage)
        per_class[class_id] = {
            "support": len(class_rows),
            "accepted_count": len(class_accepted),
            "accepted_coverage": accepted_coverage,
            "accepted_correct_count": correct,
            "accepted_recall": correct_coverage,
            "conditional_accuracy": conditional_accuracy,
        }
    probability = _probability_metrics(accepted_rows, class_ids, ece_bins=ece_bins)
    return {
        "threshold": threshold,
        "row_count": len(predictions),
        "accepted_row_count": len(accepted_rows),
        "coverage": len(accepted_rows) / len(predictions) if predictions else 0.0,
        "accepted_accuracy": probability["accuracy"],
        "balanced_accepted_recall": (
            math.fsum(selective_correct_coverage) / len(class_ids)
            if len(selective_correct_coverage) == len(class_ids)
            else None
        ),
        "balanced_accepted_conditional_accuracy": (
            math.fsum(conditional) / len(class_ids)
            if len(conditional) == len(class_ids)
            else None
        ),
        "balanced_selective_correct_coverage": (
            math.fsum(selective_correct_coverage) / len(class_ids)
            if len(selective_correct_coverage) == len(class_ids)
            else None
        ),
        "per_class": per_class,
        "accepted_probability_metrics": probability,
    }


def _task_equal_metrics(
    predictions: Sequence[Mapping[str, Any]],
    class_ids: Sequence[str],
    *,
    ece_bins: int,
    threshold: float | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(str(row["task_id"]), []).append(row)
    per_task: dict[str, Any] = {}
    accuracy_values: list[float] = []
    balanced_values: list[float] = []
    coverage_values: list[float] = []
    selective_accuracy_values: list[float] = []
    selective_correct_coverage_values: list[float] = []
    nll_values: list[float] = []
    brier_values: list[float] = []
    ece_values: list[float] = []
    for task_id, rows in sorted(grouped.items()):
        full = _probability_metrics(rows, class_ids, ece_bins=ece_bins)
        selective = _selective_metrics(
            rows, class_ids, ece_bins=ece_bins, threshold=threshold
        )
        present_recalls = [
            float(value) for value in full["per_class_recall"].values() if value is not None
        ]
        present_balanced = (
            math.fsum(present_recalls) / len(present_recalls) if present_recalls else None
        )
        accuracy_values.append(float(full["accuracy"]))
        if present_balanced is not None:
            balanced_values.append(present_balanced)
        coverage_values.append(float(selective["coverage"]))
        if selective["accepted_accuracy"] is not None:
            selective_accuracy_values.append(float(selective["accepted_accuracy"]))
        accepted_correct = sum(
            (
                bool(row.get("accepted"))
                if threshold is None
                else float(row["confidence"]) >= threshold
            )
            and row["prediction"] == row["label"]
            for row in rows
        )
        selective_correct_coverage_values.append(accepted_correct / len(rows))
        nll_values.append(float(full["multiclass_negative_log_likelihood"]))
        brier_values.append(float(full["multiclass_brier"]))
        ece_values.append(float(full["top_label_ece"]))
        per_task[task_id] = {
            "row_count": len(rows),
            "accuracy": full["accuracy"],
            "balanced_accuracy_over_present_classes": present_balanced,
            "accepted_coverage": selective["coverage"],
            "selective_accuracy": selective["accepted_accuracy"],
            "selective_correct_coverage": accepted_correct / len(rows),
        }
    mean = lambda values: math.fsum(values) / len(values) if values else None
    return {
        "task_count": len(grouped),
        "task_macro_overall_accuracy": mean(accuracy_values),
        "task_macro_balanced_accuracy": mean(balanced_values),
        "task_macro_balanced_accuracy_valid_task_count": len(balanced_values),
        "task_macro_selected_coverage": mean(coverage_values),
        "task_macro_selective_accuracy": mean(selective_accuracy_values),
        "task_macro_selective_accuracy_valid_task_count": len(selective_accuracy_values),
        "task_macro_selective_correct_coverage": mean(selective_correct_coverage_values),
        "task_macro_multiclass_negative_log_likelihood": mean(nll_values),
        "task_macro_multiclass_brier": mean(brier_values),
        "task_macro_top_label_ece": mean(ece_values),
        "per_task": per_task,
    }


def _threshold_sweep(
    predictions: Sequence[Mapping[str, Any]],
    class_ids: Sequence[str],
    thresholds: Sequence[float],
    *,
    ece_bins: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = _selective_metrics(
            predictions, class_ids, ece_bins=ece_bins, threshold=threshold
        )
        metrics["task_equal"] = _task_equal_metrics(
            predictions, class_ids, ece_bins=ece_bins, threshold=threshold
        )
        result.append(metrics)
    return result


def _select_temperature(
    predictions: Sequence[Mapping[str, Any]],
    class_ids: Sequence[str],
    temperatures: Sequence[float],
    *,
    ece_bins: int,
) -> dict[str, Any]:
    raw = np.asarray([row["probabilities"] for row in predictions], dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    for temperature in temperatures:
        scaled = _prediction_records(
            predictions, _apply_temperature(raw, temperature), class_ids
        )
        metrics = _probability_metrics(scaled, class_ids, ece_bins=ece_bins)
        candidates.append({"temperature": temperature, **metrics})
    selected = min(
        candidates,
        key=lambda item: (
            float(item["multiclass_negative_log_likelihood"]),
            abs(float(item["temperature"]) - 1.0),
            float(item["temperature"]),
        ),
    )
    return {"selected_temperature": selected["temperature"], "candidates": candidates}


def _select_threshold(
    predictions: Sequence[Mapping[str, Any]],
    class_ids: Sequence[str],
    thresholds: Sequence[float],
    *,
    contract: Mapping[str, Any],
    ece_bins: int,
) -> dict[str, Any]:
    candidates = _threshold_sweep(predictions, class_ids, thresholds, ece_bins=ece_bins)
    passing: list[dict[str, Any]] = []
    minimum_per_class = int(contract["minimum_accepted_rows_per_class"])
    for candidate in candidates:
        candidate["selection_floors_met"] = bool(
            candidate["accepted_accuracy"] is not None
            and candidate["balanced_accepted_recall"] is not None
            and candidate["task_equal"]["task_macro_selective_accuracy"] is not None
            and float(candidate["task_equal"]["task_macro_selective_accuracy"])
            >= float(contract["accepted_accuracy_minimum"])
            and float(candidate["balanced_accepted_recall"])
            >= float(contract["balanced_accepted_recall_minimum"])
            and float(candidate["task_equal"]["task_macro_selected_coverage"])
            >= float(contract["coverage_minimum"])
            and all(
                int(item["accepted_count"]) >= minimum_per_class
                for item in candidate["per_class"].values()
            )
        )
        if candidate["selection_floors_met"]:
            passing.append(candidate)
    if passing:
        selected = max(
            passing,
            key=lambda item: (
                float(item["task_equal"]["task_macro_selected_coverage"]),
                -float(item["threshold"]),
            ),
        )
        target_met = True
    else:
        def fallback_key(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
            balanced = item["balanced_accepted_recall"]
            accuracy = item["accepted_accuracy"]
            return (
                float(balanced) if balanced is not None else -1.0,
                (
                    float(item["task_equal"]["task_macro_selective_accuracy"])
                    if item["task_equal"]["task_macro_selective_accuracy"] is not None
                    else (float(accuracy) if accuracy is not None else -1.0)
                ),
                float(item["task_equal"]["task_macro_selected_coverage"]),
                -float(item["threshold"]),
            )

        selected = max(candidates, key=fallback_key)
        target_met = False
    return {
        "selected_threshold": selected["threshold"],
        "selection_floors_met": target_met,
        "selected_calibration_metrics": selected,
        "candidates": candidates,
    }


def _public_model_record(model: Mapping[str, Any]) -> dict[str, Any]:
    return READOUT.public_model_record(model)


def nested_repository_evaluation(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    class_ids: Sequence[str],
    c_grid: Sequence[float],
    maximum_iterations: int,
    minimum_valid_inner_folds: int,
    temperatures: Sequence[float],
    thresholds: Sequence[float],
    threshold_contract: Mapping[str, Any],
    ece_bins: int,
) -> dict[str, Any]:
    require(variant in VARIANTS and bool(rows), "variant evaluation is invalid")
    row_ids = [str(row["row_id"]) for row in rows]
    require(len(row_ids) == len(set(row_ids)), "rows contain duplicate IDs")
    task_repositories: dict[str, set[str]] = {}
    for row in rows:
        task_repositories.setdefault(str(row["task_id"]), set()).add(str(row["repo"]))
    require(all(len(value) == 1 for value in task_repositories.values()), "a task crosses repository folds")
    repositories = sorted({str(row["repo"]) for row in rows})
    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for outer_repository in repositories:
        outer_train = [row for row in rows if row["repo"] != outer_repository]
        outer_evaluation = [row for row in rows if row["repo"] == outer_repository]
        fold: dict[str, Any] = {
            "heldout_repository": outer_repository,
            "training_row_count": len(outer_train),
            "evaluation_row_count": len(outer_evaluation),
            "training_class_support": _class_support(outer_train, class_ids),
            "heldout_labels_used_for_fit_scaling_calibration_or_threshold_selection": False,
            "candidate_models": [],
        }
        if not outer_evaluation or not _all_classes(outer_train, class_ids):
            fold["status"] = "insufficient_outer_training_support"
            folds.append(fold)
            continue
        candidate_internal: dict[float, list[dict[str, Any]]] = {}
        inner_repositories = sorted({str(row["repo"]) for row in outer_train})
        for c_value in c_grid:
            inner_predictions: list[dict[str, Any]] = []
            fit_records: list[dict[str, Any]] = []
            skipped: list[str] = []
            for inner_repository in inner_repositories:
                inner_train = [row for row in outer_train if row["repo"] != inner_repository]
                inner_evaluation = [row for row in outer_train if row["repo"] == inner_repository]
                if not inner_evaluation or not _all_classes(inner_train, class_ids):
                    skipped.append(inner_repository)
                    continue
                x_train, y_train = _matrix(inner_train, variant, class_ids)
                model = READOUT.fit_multinomial_lbfgs(
                    x_train,
                    y_train,
                    c_value=float(c_value),
                    maximum_iterations=maximum_iterations,
                    class_ids=class_ids,
                )
                fit_records.append({"heldout_repository": inner_repository, **_public_model_record(model)})
                if not model["converged"]:
                    continue
                x_evaluation, _ = _matrix(inner_evaluation, variant, class_ids)
                inner_predictions.extend(
                    _prediction_records(
                        inner_evaluation,
                        READOUT.predict_multinomial(model, x_evaluation),
                        class_ids,
                    )
                )
            predicted_ids = {str(row["row_id"]) for row in inner_predictions}
            expected_ids = {str(row["row_id"]) for row in outer_train}
            valid_fits = [item for item in fit_records if item["converged"]]
            candidate: dict[str, Any] = {
                "c_value": float(c_value),
                "valid_inner_fold_count": len(valid_fits),
                "inner_fold_count": len(inner_repositories),
                "skipped_inner_repositories": skipped,
                "complete_inner_prediction_coverage": predicted_ids == expected_ids,
                "all_attempted_fits_converged": len(valid_fits) == len(fit_records),
            }
            if (
                len(valid_fits) < minimum_valid_inner_folds
                or len(valid_fits) != len(inner_repositories)
                or predicted_ids != expected_ids
            ):
                candidate.update({"status": "incomplete_inner_crossfit", "balanced_accuracy": None})
            else:
                metrics = _probability_metrics(inner_predictions, class_ids, ece_bins=ece_bins)
                candidate.update(
                    {
                        "status": "available",
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "inner_validation_row_count": len(inner_predictions),
                    }
                )
                candidate_internal[float(c_value)] = inner_predictions
            fold["candidate_models"].append(candidate)
        available = [item for item in fold["candidate_models"] if item["status"] == "available"]
        if len(available) != len(c_grid):
            fold["status"] = "incomplete_regularization_grid"
            folds.append(fold)
            continue
        selected = min(available, key=lambda item: (-float(item["balanced_accuracy"]), float(item["c_value"])))
        selected_c = float(selected["c_value"])
        inner_predictions = candidate_internal[selected_c]
        temperature_selection = _select_temperature(
            inner_predictions, class_ids, temperatures, ece_bins=ece_bins
        )
        temperature = float(temperature_selection["selected_temperature"])
        inner_raw = np.asarray([row["probabilities"] for row in inner_predictions], dtype=np.float64)
        inner_calibrated = _prediction_records(
            inner_predictions, _apply_temperature(inner_raw, temperature), class_ids
        )
        threshold_selection = _select_threshold(
            inner_calibrated,
            class_ids,
            thresholds,
            contract=threshold_contract,
            ece_bins=ece_bins,
        )
        threshold = float(threshold_selection["selected_threshold"])
        x_train, y_train = _matrix(outer_train, variant, class_ids)
        final_model = READOUT.fit_multinomial_lbfgs(
            x_train,
            y_train,
            c_value=selected_c,
            maximum_iterations=maximum_iterations,
            class_ids=class_ids,
        )
        fold.update(
            {
                "selected_c": selected_c,
                "selected_inner_balanced_accuracy": selected["balanced_accuracy"],
                "temperature_calibration": temperature_selection,
                "threshold_selection": threshold_selection,
                "outer_model": _public_model_record(final_model),
            }
        )
        if not final_model["converged"]:
            fold["status"] = "outer_model_nonconverged"
            folds.append(fold)
            continue
        x_evaluation, _ = _matrix(outer_evaluation, variant, class_ids)
        raw_probabilities = READOUT.predict_multinomial(final_model, x_evaluation)
        calibrated_probabilities = _apply_temperature(raw_probabilities, temperature)
        evaluation_predictions = _prediction_records(outer_evaluation, calibrated_probabilities, class_ids)
        for prediction in evaluation_predictions:
            prediction["temperature"] = temperature
            prediction["acceptance_threshold"] = threshold
            prediction["accepted"] = float(prediction["confidence"]) >= threshold
        predictions.extend(evaluation_predictions)
        fold["status"] = "available"
        folds.append(fold)
    order = {row_id: index for index, row_id in enumerate(row_ids)}
    predictions.sort(key=lambda item: order[str(item["row_id"])])
    complete = {str(row["row_id"]) for row in predictions} == set(row_ids)
    status = "available" if complete and all(fold["status"] == "available" for fold in folds) else "incomplete_nested_repository_crossfit"
    return {
        "status": status,
        "algorithm": "nested_repository_heldout_calibrated_selective_multinomial_v1",
        "variant": variant,
        "row_count": len(rows),
        "repository_count": len(repositories),
        "complete_prediction_coverage": complete,
        "folds": folds,
        "predictions": predictions,
        "metrics": {
            "primary_task_equal": _task_equal_metrics(
                predictions, class_ids, ece_bins=ece_bins
            ),
            "row_equal_role": "secondary",
            "full": _probability_metrics(predictions, class_ids, ece_bins=ece_bins),
            "selected_abstention": _selective_metrics(predictions, class_ids, ece_bins=ece_bins),
            "fixed_threshold_sweep": _threshold_sweep(
                predictions, class_ids, thresholds, ece_bins=ece_bins
            ),
        },
    }


def _flatten_metrics(
    predictions: Sequence[Mapping[str, Any]], class_ids: Sequence[str], *, ece_bins: int
) -> dict[str, float | None]:
    full = _probability_metrics(predictions, class_ids, ece_bins=ece_bins)
    selected = _selective_metrics(predictions, class_ids, ece_bins=ece_bins)
    task_equal = _task_equal_metrics(predictions, class_ids, ece_bins=ece_bins)
    coverages = [item["accepted_coverage"] for item in selected["per_class"].values()]
    minimum_coverage = min(float(item) for item in coverages) if coverages and all(item is not None for item in coverages) else None
    return {
        "task_macro_overall_accuracy": task_equal["task_macro_overall_accuracy"],
        "task_macro_balanced_accuracy": task_equal["task_macro_balanced_accuracy"],
        "task_macro_selected_coverage": task_equal["task_macro_selected_coverage"],
        "task_macro_selective_accuracy": task_equal["task_macro_selective_accuracy"],
        "task_macro_selective_correct_coverage": task_equal[
            "task_macro_selective_correct_coverage"
        ],
        "overall_accuracy": full["accuracy"],
        "balanced_accuracy": full["balanced_accuracy"],
        "multiclass_negative_log_likelihood": full["multiclass_negative_log_likelihood"],
        "multiclass_brier": full["multiclass_brier"],
        "top_label_ece": full["top_label_ece"],
        "selected_coverage": selected["coverage"],
        "selected_accepted_accuracy": selected["accepted_accuracy"],
        "selected_balanced_accepted_recall": selected["balanced_accepted_recall"],
        "selected_minimum_per_class_accepted_coverage": minimum_coverage,
    }


def _hierarchical_sample_indices(
    rows: Sequence[Mapping[str, Any]], rng: np.random.Generator
) -> list[tuple[int, str, str]]:
    repositories: dict[str, dict[str, list[int]]] = {}
    for index, row in enumerate(rows):
        repositories.setdefault(str(row["repo"]), {}).setdefault(str(row["task_id"]), []).append(index)
    repository_ids = sorted(repositories)
    sampled: list[tuple[int, str, str]] = []
    for repository_draw, repository_index in enumerate(
        rng.integers(0, len(repository_ids), size=len(repository_ids))
    ):
        repository = repositories[repository_ids[int(repository_index)]]
        task_ids = sorted(repository)
        bootstrap_repository = f"bootstrap-repository-{repository_draw}"
        for task_draw, task_index in enumerate(
            rng.integers(0, len(task_ids), size=len(task_ids))
        ):
            bootstrap_task = f"{bootstrap_repository}-task-{task_draw}"
            sampled.extend(
                (index, bootstrap_repository, bootstrap_task)
                for index in repository[task_ids[int(task_index)]]
            )
    return sampled


def hierarchical_bootstrap(
    predictions_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    class_ids: Sequence[str],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
    minimum_valid_fraction: float,
    ece_bins: int,
) -> dict[str, Any]:
    available_variants = [variant for variant in VARIANTS if variant in predictions_by_variant]
    require(bool(available_variants), "no variants are available for bootstrap")
    reference = list(predictions_by_variant[available_variants[0]])
    reference_ids = [str(row["row_id"]) for row in reference]
    for variant in available_variants:
        require(
            [str(row["row_id"]) for row in predictions_by_variant[variant]] == reference_ids,
            "bootstrap variants are not exactly paired",
        )
    observed = {
        variant: _flatten_metrics(predictions_by_variant[variant], class_ids, ece_bins=ece_bins)
        for variant in available_variants
    }
    comparison_pairs = (
        ("hybrid", "logit_context"),
        ("hybrid", "jacobian_context"),
        ("hybrid", "history_context"),
        ("hybrid", "progress_only"),
        ("jacobian_context", "lexical_progress"),
        ("logit_context", "lexical_progress"),
        ("jacobian_context", "logit_context"),
        ("public_jacobian", "ordinary_logit"),
    )
    comparison_pairs = tuple(
        pair for pair in comparison_pairs if pair[0] in observed and pair[1] in observed
    )
    variant_draws = {
        variant: {metric: [] for metric in observed[variant]} for variant in available_variants
    }
    paired_draws = {
        f"{candidate}_minus_{reference_name}": {metric: [] for metric in observed[candidate]}
        for candidate, reference_name in comparison_pairs
    }
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        sampled_indices = _hierarchical_sample_indices(reference, rng)
        sampled_metrics: dict[str, dict[str, float | None]] = {}
        for variant in available_variants:
            variant_predictions = predictions_by_variant[variant]
            sampled_rows = [
                {
                    **variant_predictions[index],
                    "repo": bootstrap_repository,
                    "task_id": bootstrap_task,
                }
                for index, bootstrap_repository, bootstrap_task in sampled_indices
            ]
            metrics = _flatten_metrics(sampled_rows, class_ids, ece_bins=ece_bins)
            sampled_metrics[variant] = metrics
            for metric, value in metrics.items():
                if value is not None and math.isfinite(float(value)):
                    variant_draws[variant][metric].append(float(value))
        for candidate, reference_name in comparison_pairs:
            name = f"{candidate}_minus_{reference_name}"
            for metric in paired_draws[name]:
                left = sampled_metrics[candidate][metric]
                right = sampled_metrics[reference_name][metric]
                if left is not None and right is not None:
                    paired_draws[name][metric].append(float(left) - float(right))

    alpha = (1.0 - confidence_level) / 2.0

    def summarize(draws: Sequence[float], point: float | None) -> dict[str, Any]:
        valid_fraction = len(draws) / samples
        return {
            "status": "available" if point is not None and valid_fraction >= minimum_valid_fraction else "insufficient_valid_bootstrap_draws",
            "point": point,
            "lower": float(np.quantile(draws, alpha)) if draws and valid_fraction >= minimum_valid_fraction else None,
            "upper": float(np.quantile(draws, 1.0 - alpha)) if draws and valid_fraction >= minimum_valid_fraction else None,
            "valid_sample_count": len(draws),
            "valid_sample_fraction": valid_fraction,
        }

    variants = {
        variant: {
            metric: summarize(draws, observed[variant][metric])
            for metric, draws in variant_draws[variant].items()
        }
        for variant in available_variants
    }
    paired: dict[str, Any] = {}
    for candidate, reference_name in comparison_pairs:
        name = f"{candidate}_minus_{reference_name}"
        paired[name] = {
            "candidate": candidate,
            "reference": reference_name,
            "metrics": {
                metric: summarize(
                    draws,
                    (
                        float(observed[candidate][metric]) - float(observed[reference_name][metric])
                        if observed[candidate][metric] is not None and observed[reference_name][metric] is not None
                        else None
                    ),
                )
                for metric, draws in paired_draws[name].items()
            },
        }
    return {
        "algorithm": "hierarchical_repository_then_task_percentile_v1",
        "samples": samples,
        "seed": seed,
        "confidence_level": confidence_level,
        "minimum_valid_fraction": minimum_valid_fraction,
        "models_refit_inside_bootstrap": False,
        "variants": variants,
        "paired_deltas": paired,
    }


def evaluate_reliability_gates(
    rows: Sequence[Mapping[str, Any]],
    evaluations: Mapping[str, Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    gates = protocol["gates"]
    support = {
        "row_count": len(rows),
        "task_count": len({str(row["task_id"]) for row in rows}),
        "repository_count": len({str(row["repo"]) for row in rows}),
        "class_support": _class_support(rows, protocol["class_ids"]),
    }
    checks: list[dict[str, Any]] = []
    for metric, minimum in (
        ("row_count", gates["support"]["minimum_rows"]),
        ("task_count", gates["support"]["minimum_tasks"]),
        ("repository_count", gates["support"]["minimum_repositories"]),
    ):
        checks.append({"id": f"support_{metric}", "observed": support[metric], "operator": "minimum_inclusive", "threshold": minimum, "passed": support[metric] >= minimum})
    if gates["require_all_outer_folds_available"]:
        passed = all(evaluations[variant]["status"] == "available" for variant in VARIANTS)
        checks.append({"id": "all_outer_folds_available", "passed": passed})
    if gates["require_all_calibration_targets_met"]:
        passed = all(
            fold["threshold_selection"]["selection_floors_met"]
            for variant in gates["calibration_target_variants"]
            for fold in evaluations[variant]["folds"]
            if fold["status"] == "available"
        ) and all(
            evaluations[variant]["status"] == "available"
            for variant in gates["calibration_target_variants"]
        )
        checks.append({"id": "all_calibration_targets_met", "passed": passed})

    flattened = {
        variant: _flatten_metrics(
            evaluations[variant]["predictions"], protocol["class_ids"], ece_bins=protocol["metrics"]["ece_bins"]
        )
        for variant in VARIANTS
    }
    for gate in gates["absolute"]:
        if gate["bound"] == "point":
            observed = flattened[gate["variant"]][gate["metric"]]
        else:
            key = "lower" if gate["bound"] == "bootstrap_lower" else "upper"
            variant_bootstrap = bootstrap["variants"].get(gate["variant"])
            observed = (
                variant_bootstrap[gate["metric"]][key]
                if variant_bootstrap is not None
                else None
            )
        checks.append(_gate_result(gate, observed))
    for gate in gates["paired"]:
        comparison = bootstrap["paired_deltas"].get(f"{gate['candidate']}_minus_{gate['reference']}")
        if comparison is None:
            observed = None
        else:
            metric = comparison["metrics"][gate["metric"]]
            observed = metric["point"] if gate["bound"] == "point" else metric["lower" if gate["bound"] == "bootstrap_lower" else "upper"]
        checks.append(_gate_result(gate, observed))
    return {
        "support": support,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "thresholds_are_protocol_supplied": True,
    }


def _gate_result(gate: Mapping[str, Any], observed: float | None) -> dict[str, Any]:
    threshold = float(gate["value"])
    operator = str(gate["operator"])
    passed = False
    if observed is not None:
        value = float(observed)
        if operator == "minimum_inclusive":
            passed = value >= threshold
        elif operator == "minimum_exclusive":
            passed = value > threshold
        elif operator == "maximum_inclusive":
            passed = value <= threshold
        else:
            passed = value < threshold
    result = {
        "id": gate["id"],
        "metric": gate["metric"],
        "bound": gate["bound"],
        "operator": operator,
        "threshold": threshold,
        "observed": observed,
        "passed": passed,
    }
    for key in ("variant", "candidate", "reference"):
        if key in gate:
            result[key] = gate[key]
    return result


def build_analysis(
    *,
    prompt_bundle_value: Any,
    report_value: Any,
    protocol_value: Any,
    behavioral_protocol_value: Any,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    protocol = validate_protocol(
        protocol_value,
        behavioral_protocol_value=behavioral_protocol_value,
        behavioral_protocol_sha256=input_hashes["behavioral_protocol"],
        prompt_sha256=input_hashes["prompts"],
        report_sha256=input_hashes["public_report"],
    )
    extracted = extract_rows(prompt_bundle_value, report_value, protocol=protocol)
    rows = extracted["rows"]
    require(bool(rows), "no numerically stable labeled task-state rows remain")
    complete_history = all(row["features"]["history_context"] is not None for row in rows)
    evaluations: dict[str, Any] = {}
    for variant in VARIANTS:
        if variant in HISTORY_VARIANTS and not complete_history:
            evaluations[variant] = {
                "status": "unavailable_incomplete_causal_history",
                "variant": variant,
                "row_count": 0,
                "repository_count": 0,
                "complete_prediction_coverage": False,
                "folds": [],
                "predictions": [],
                "metrics": None,
            }
            continue
        evaluations[variant] = nested_repository_evaluation(
            rows,
            variant=variant,
            class_ids=protocol["class_ids"],
            c_grid=protocol["model"]["c_grid"],
            maximum_iterations=protocol["model"]["maximum_iterations"],
            minimum_valid_inner_folds=protocol["evaluation"]["minimum_inner"],
            temperatures=protocol["calibration"]["temperatures"],
            thresholds=protocol["abstention"]["thresholds"],
            threshold_contract=protocol["abstention"],
            ece_bins=protocol["metrics"]["ece_bins"],
        )
    available_predictions = {
        variant: evaluations[variant]["predictions"]
        for variant in VARIANTS
        if evaluations[variant]["status"] == "available"
    }
    bootstrap = hierarchical_bootstrap(
        available_predictions,
        protocol["class_ids"],
        samples=protocol["bootstrap"]["samples"],
        seed=protocol["bootstrap"]["seed"],
        confidence_level=protocol["bootstrap"]["confidence_level"],
        minimum_valid_fraction=protocol["bootstrap"]["minimum_valid_fraction"],
        ece_bins=protocol["metrics"]["ece_bins"],
    )
    bootstrap["interval_interpretation"] = protocol["bootstrap"][
        "interval_interpretation"
    ]
    bootstrap["operational_reliability_proof"] = protocol["bootstrap"][
        "operational_reliability_proof"
    ]
    reliability = evaluate_reliability_gates(
        rows, evaluations, bootstrap, protocol=protocol
    )
    reliability["decision_scope"] = "development_screen_only"
    reliability["operational_reliability_claim"] = False
    return {
        "schema_version": 1,
        "kind": "swe_task_state_selective_interpreter_analysis",
        "analysis_version": "repository-heldout-calibrated-selective-v1",
        "inputs": dict(input_hashes),
        "protocol": {
            "id": protocol_value["id"],
            "schema_version": protocol_value["schema_version"],
            "decision_scope": protocol_value["decision_scope"],
            "thresholds_are_external_protocol_values": True,
        },
        "eligibility": extracted["eligibility"],
        "feature_dimensions": {
            variant: (
                len(rows[0]["features"][variant])
                if rows[0]["features"][variant] is not None
                else None
            )
            for variant in VARIANTS
        },
        "class_ids": protocol["class_ids"],
        "evaluations": evaluations,
        "clustered_bootstrap": bootstrap,
        "reliability": reliability,
        "interpretation_scope": (
            "observable next-action class only; this is not a decoder of hidden chain-of-thought"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--behavioral-protocol", type=Path, default=DEFAULT_BEHAVIORAL_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.prompts, args.public_report, args.protocol, args.behavioral_protocol):
        require(path.is_file(), f"missing input: {path}")
    input_hashes = {
        "prompts": sha256_file(args.prompts),
        "public_report": sha256_file(args.public_report),
        "protocol": sha256_file(args.protocol),
        "behavioral_protocol": sha256_file(args.behavioral_protocol),
        "analyzer_implementation": sha256_file(Path(__file__)),
        "task_state_readout_implementation": sha256_file(TASK_STATE_READOUT_PATH),
    }
    prompt_bundle = json.loads(args.prompts.read_bytes())
    report = json.loads(args.public_report.read_bytes())
    protocol = json.loads(args.protocol.read_bytes())
    behavioral_protocol = json.loads(args.behavioral_protocol.read_bytes())
    analysis = build_analysis(
        prompt_bundle_value=prompt_bundle,
        report_value=report,
        protocol_value=protocol,
        behavioral_protocol_value=behavioral_protocol,
        input_hashes=input_hashes,
    )
    del prompt_bundle, report
    gc.collect()
    atomic_write_json(args.output, analysis)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "eligible_rows": analysis["eligibility"]["eligible_row_count"],
                "development_gates_passed": analysis["reliability"]["passed"],
                "operational_reliability_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
