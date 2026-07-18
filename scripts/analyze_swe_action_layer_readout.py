#!/usr/bin/env python3
"""Fit the predeclared nested repository-held-out SWE action readout.

The strict track consumes raw fixed-layer features already validated and emitted by
the frozen behavioral analysis.  The sensitivity track additionally revalidates
the raw reports because numerically stable rows outside the strict certification
gate are intentionally absent from that analysis.
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
DEFAULT_PROTOCOL = ROOT / "configs/swe_action_layer_readout_protocol.json"
DEFAULT_BEHAVIORAL_PROTOCOL = ROOT / "configs/swe_behavioral_readout_protocol.json"
DEFAULT_TRANSPORT_PROTOCOL = ROOT / "configs/swe_next_token_transport_protocol.json"
LAYERS = tuple(range(24, 48))
CLASS_IDS = ("inspect", "edit", "validate", "finalize")
METHODS = (
    "ordinary_logit",
    "public_jacobian",
    "nf4_jacobian",
    "native_jacobian",
)
GRADIENT_TOLERANCE = 1e-5
LBFGS_HISTORY_SIZE = 10
LINE_SEARCH_MAXIMUM_STEPS = 60
ARMIJO_CONSTANT = 1e-4


def _load_behavioral_module():
    name = "analyze_swe_behavioral_probes"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "scripts/analyze_swe_behavioral_probes.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BEHAVIORAL = _load_behavioral_module()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
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


def nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be nonempty text")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    )


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
    digest = nonempty_string(value, label)
    require(
        len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return digest


def validate_protocol(
    value: Any,
    *,
    behavioral_protocol_sha256: str,
    transport_protocol_sha256: str,
) -> dict[str, Any]:
    protocol = mapping(value, "action-layer protocol")
    require(
        protocol.get("schema_version") == 1
        and protocol.get("id") == "swe-n20-action-layer-readout-v1"
        and protocol.get("status") == "frozen_before_n20_action_score_inspection",
        "action-layer protocol identity changed",
    )
    pins = mapping(protocol.get("input_pins"), "action-layer input pins")
    require(
        pins.get("behavioral_protocol_sha256") == behavioral_protocol_sha256
        and pins.get("transport_protocol_sha256") == transport_protocol_sha256,
        "action-layer upstream protocol pin mismatch",
    )
    cohort_sha = _validate_sha(pins.get("cohort_manifest_sha256"), "cohort manifest pin")
    prompt_sha = _validate_sha(pins.get("prompt_bundle_sha256"), "prompt bundle pin")

    scope = mapping(protocol.get("scope"), "action-layer scope")
    require(
        scope.get("checkpoint_count") == 160
        and scope.get("action_labeled_checkpoint_count") == 154
        and scope.get("task_count") == 20
        and scope.get("repository_count") == 11
        and scope.get("class_ids_in_order") == list(CLASS_IDS)
        and scope.get("methods") == list(METHODS)
        and scope.get("cohort_subgroups_are_descriptive_only") is True,
        "action-layer scope changed",
    )
    disclosure = mapping(protocol.get("design_disclosure"), "design disclosure")
    known_counts = mapping(disclosure.get("known_class_counts"), "known class counts")
    require(
        disclosure.get("action_labels_and_class_counts_inspected") is True
        and disclosure.get("public_nf4_or_native_action_scores_predictions_or_metrics_inspected")
        is False
        and known_counts
        == {"inspect": 101, "edit": 12, "validate": 26, "finalize": 15, "missing": 6},
        "action-layer design disclosure changed",
    )

    features = mapping(protocol.get("feature_contract"), "feature contract")
    require(
        features.get("layers") == list(LAYERS)
        and features.get("feature_order") == "layer-major then class order"
        and features.get("feature_count") == len(LAYERS) * len(CLASS_IDS)
        and features.get("best_layer_or_feature_selection") == "forbidden"
        and features.get("fit_scaler_on_heldout_rows") is False
        and features.get("missing_feature_imputation") is False,
        "action-layer feature contract changed",
    )
    model = mapping(protocol.get("model_contract"), "model contract")
    c_grid = [finite(item, "regularization C") for item in sequence(model.get("regularization_C_grid"), "C grid")]
    require(
        model.get("family") == "multinomial logistic regression"
        and model.get("penalty") == "L2"
        and model.get("solver") == "lbfgs"
        and model.get("class_weight") == "balanced from the current training split only"
        and model.get("fit_intercept") is True
        and integer(model.get("maximum_iterations"), "maximum iterations", minimum=1) == 4000
        and model.get("convergence_required") is True
        and c_grid == [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
        and all(item > 0.0 for item in c_grid),
        "action-layer model contract changed",
    )
    outer = mapping(protocol.get("outer_evaluation"), "outer evaluation")
    inner = mapping(protocol.get("inner_model_selection"), "inner model selection")
    require(
        outer.get("algorithm") == "leave_one_repository_out"
        and outer.get("all_checkpoints_from_heldout_repository_excluded_from_training") is True
        and outer.get("heldout_labels_used_for_fit_or_hyperparameter_selection") is False
        and outer.get("complete_predictions_for_all_eligible_rows_required") is True
        and inner.get("algorithm")
        == "leave_one_repository_out_within_outer_training_repositories"
        and inner.get("selection_metric")
        == "balanced_accuracy_over_concatenated_inner-heldout_predictions"
        and integer(inner.get("minimum_valid_inner_folds"), "minimum inner folds", minimum=1)
        == 5
        and inner.get("every_class_required_in_concatenated_inner_validation") is True
        and inner.get("every_class_required_in_each_inner_training_split") is True
        and inner.get("tie_break") == "smallest_C"
        and inner.get("outer_heldout_repository_excluded") is True,
        "nested repository evaluation contract changed",
    )
    tracks = mapping(protocol.get("numerical_tracks"), "numerical tracks")
    strict = mapping(tracks.get("strict_primary"), "strict track")
    sensitivity = mapping(
        tracks.get("paired_stable_reconstruction_sensitivity"), "sensitivity track"
    )
    require(
        strict.get("minimum_rows") == 128
        and strict.get("minimum_tasks") == 20
        and strict.get("minimum_repositories") == 11
        and strict.get("all_classes_required") is True
        and sensitivity.get("minimum_rows") == 128
        and sensitivity.get("minimum_tasks") == 20
        and sensitivity.get("minimum_repositories") == 11
        and sensitivity.get("minimum_rows_per_task") == 6
        and sensitivity.get("all_classes_required") is True
        and sensitivity.get("must_be_labeled_sensitivity") is True
        and sensitivity.get("primary_decision_override_forbidden") is True,
        "action-layer numerical support gates changed",
    )
    paired = mapping(protocol.get("paired_inference"), "paired inference")
    require(
        paired.get("algorithm")
        == "paired_hierarchical_repository_then_task_percentile_v1"
        and integer(paired.get("samples"), "bootstrap samples", minimum=1) == 5000
        and finite(paired.get("confidence_level"), "confidence level") == 0.95
        and integer(paired.get("seed"), "bootstrap seed") == 61987
        and paired.get("same_draw_for_candidate_and_reference") is True
        and paired.get("resample_repositories_then_tasks_within_repository") is True
        and paired.get("row_resampling_forbidden") is True
        and paired.get("models_refit_inside_bootstrap") is False,
        "action-layer paired inference contract changed",
    )
    comparisons = sequence(protocol.get("primary_comparisons"), "primary comparisons")
    require(
        comparisons
        == [
            "learned_public_jacobian_minus_frozen_public_jacobian_readout",
            "learned_public_jacobian_minus_learned_ordinary_logit",
            "learned_public_jacobian_minus_learned_native_jacobian",
            "learned_public_jacobian_minus_learned_nf4_jacobian",
        ],
        "action-layer primary comparisons changed",
    )
    metrics = mapping(protocol.get("metrics"), "action-layer metrics")
    require(
        metrics.get("primary") == "balanced_accuracy"
        and metrics.get("secondary")
        == [
            "micro_accuracy",
            "per_class_recall",
            "multiclass_negative_log_likelihood",
            "multiclass_brier",
            "confusion_matrix",
        ]
        and metrics.get("baselines")
        == [
            "frozen intercept-plus-temperature crossfit readout",
            "training-fold majority class",
        ],
        "action-layer metric or baseline contract changed",
    )
    material = mapping(protocol.get("material_effect_rules"), "material effect rules")
    require(
        material
        == {
            "readout_refinement_signal": {
                "learned_public_minus_frozen_public_balanced_accuracy_minimum_inclusive": 0.1,
                "confidence_interval_lower_minimum_exclusive": 0.0,
            },
            "native_fit_capacity_signal": {
                "readout_refinement_signal_required": True,
                "learned_public_minus_learned_native_balanced_accuracy_minimum_inclusive": 0.1,
                "confidence_interval_lower_minimum_exclusive": 0.0,
            },
            "no_native_refit_signal": {
                "learned_public_minus_learned_native_confidence_interval_upper_maximum_exclusive": 0.1
            },
        },
        "action-layer material effect rules changed",
    )
    return {
        "value": protocol,
        "pins": {"cohort": cohort_sha, "prompts": prompt_sha},
        "class_ids": list(CLASS_IDS),
        "methods": list(METHODS),
        "c_grid": c_grid,
        "maximum_iterations": int(model["maximum_iterations"]),
        "minimum_valid_inner_folds": int(inner["minimum_valid_inner_folds"]),
        "track_contracts": {"strict_primary": strict, "sensitivity": sensitivity},
        "paired": paired,
        "comparisons": comparisons,
    }


def flatten_raw_fixed_band(value: Any) -> tuple[list[float], list[float]]:
    band = mapping(value, "raw fixed band")
    require(band.get("layers") == list(LAYERS), "raw fixed-band layer list changed")
    per_layer = sequence(band.get("per_layer"), "raw per-layer scores")
    require(len(per_layer) == len(LAYERS), "raw per-layer score count changed")
    features: list[float] = []
    aggregate = {class_id: [] for class_id in CLASS_IDS}
    for expected_layer, raw_layer in zip(LAYERS, per_layer, strict=True):
        layer = mapping(raw_layer, "raw layer")
        require(layer.get("layer") == expected_layer, "raw fixed layers are out of order")
        scores = mapping(layer.get("class_scores"), "raw layer class scores")
        require(set(scores) == set(CLASS_IDS), "raw layer class-score keys changed")
        for class_id in CLASS_IDS:
            score = finite(scores[class_id], f"{class_id} layer score")
            features.append(score)
            aggregate[class_id].append(score)
    band_scores = mapping(band.get("band_mean_class_scores"), "band-mean class scores")
    require(set(band_scores) == set(CLASS_IDS), "band-mean class-score keys changed")
    means: list[float] = []
    for class_id in CLASS_IDS:
        observed = finite(band_scores[class_id], f"{class_id} band mean")
        expected = math.fsum(aggregate[class_id]) / len(LAYERS)
        require(
            math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12),
            f"{class_id} band mean is inconsistent with per-layer scores",
        )
        means.append(observed)
    require(len(features) == 96, "action-layer feature vector is not 96-dimensional")
    return features, means


def _row_base(value: Mapping[str, Any]) -> dict[str, Any]:
    row_id = nonempty_string(value.get("prompt_id", value.get("row_id")), "row ID")
    label = nonempty_string(value.get("expected_action", value.get("label")), "action label")
    require(label in CLASS_IDS, f"undeclared action label: {label}")
    return {
        "row_id": row_id,
        "task_id": nonempty_string(value.get("task_id"), "task ID"),
        "repo": nonempty_string(value.get("repo"), "repository"),
        "cohort_id": nonempty_string(value.get("cohort_id"), "cohort ID"),
        "label": label,
        "task_request_index": integer(
            value.get("task_request_index"), "task request index", minimum=1
        ),
    }


def validate_behavioral_analysis(
    value: Any,
    *,
    input_hashes: Mapping[str, str],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    analysis = mapping(value, "behavioral analysis")
    require(
        analysis.get("schema_version") == 1
        and analysis.get("kind") == "swe_verified_behavioral_task_held_out_analysis"
        and analysis.get("analysis_version") == "task-held-out-paired-decision-v2",
        "behavioral analysis identity changed",
    )
    analysis_inputs = mapping(analysis.get("inputs"), "behavioral analysis inputs")
    for key in ("prompts", "public_report", "nf4_report", "native_report", "protocol"):
        require(
            analysis_inputs.get(key) == input_hashes[key],
            f"behavioral analysis {key} hash does not bind the supplied input",
        )
    analysis_protocol = mapping(analysis.get("protocol"), "behavioral protocol record")
    require(
        analysis_protocol.get("sha256") == input_hashes["protocol"],
        "behavioral protocol digest changed in analysis",
    )
    campaign = mapping(analysis.get("campaign"), "behavioral campaign")
    require(
        campaign.get("combined_cohort_manifest_sha256") == protocol["pins"]["cohort"]
        and campaign.get("task_count") == 20
        and campaign.get("selected_task_count") == 20
        and campaign.get("repository_count") == 11
        and campaign.get("prompt_count") == 160
        and campaign.get("primary_probeable_uniform_prompt_count") == 160,
        "behavioral analysis campaign scope changed",
    )
    track = mapping(mapping(analysis.get("tracks"), "behavioral tracks").get("next_action"), "next-action track")
    detailed = sequence(track.get("rows"), "next-action rows")
    require(len(detailed) == 154, "behavioral action-labeled row count changed")
    known_counts = mapping(
        mapping(protocol["value"].get("design_disclosure"), "design disclosure").get(
            "known_class_counts"
        ),
        "known counts",
    )
    observed_counts = {class_id: 0 for class_id in CLASS_IDS}
    strict_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    detailed_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_row in detailed:
        row = mapping(raw_row, "next-action row")
        base = _row_base(row)
        require(base["row_id"] not in detailed_by_id, f"duplicate row: {base['row_id']}")
        detailed_by_id[base["row_id"]] = row
        observed_counts[base["label"]] += 1
        certified = row.get("numerically_certified_across_all_reports")
        require(isinstance(certified, bool), "joint certification flag must be boolean")
        methods = mapping(row.get("methods"), "next-action methods")
        require(
            set(methods) == (set(METHODS) if certified else set()),
            "strict behavioral method presence differs from joint certification",
        )
        if not certified:
            continue
        for method in METHODS:
            method_value = mapping(methods[method], f"{method} behavioral row")
            feature, band_scores = flatten_raw_fixed_band(method_value.get("raw_fixed_band"))
            strict_rows[method].append(
                {**base, "feature": feature, "band_scores": band_scores}
            )
    require(
        observed_counts == {class_id: int(known_counts[class_id]) for class_id in CLASS_IDS},
        "behavioral action class counts differ from the disclosed counts",
    )

    crossfit = mapping(track.get("task_held_out_crossfit"), "frozen crossfit")
    frozen = mapping(crossfit.get("public_jacobian"), "frozen public readout")
    frozen_predictions = []
    for raw_prediction in sequence(frozen.get("predictions"), "frozen public predictions"):
        prediction = mapping(raw_prediction, "frozen public prediction")
        base = {
            key: nonempty_string(prediction.get(key), f"frozen {key}")
            for key in ("row_id", "task_id", "repo", "label")
        }
        require(base["label"] in CLASS_IDS, "frozen prediction label is undeclared")
        probabilities = [finite(item, "frozen probability") for item in sequence(prediction.get("probabilities"), "frozen probabilities")]
        require(
            prediction.get("class_ids") == list(CLASS_IDS)
            and len(probabilities) == len(CLASS_IDS)
            and all(item >= 0.0 for item in probabilities)
            and math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-9),
            "frozen public probabilities are invalid",
        )
        frozen_predictions.append(
            {
                **base,
                "cohort_id": nonempty_string(
                    prediction.get("cohort_id", "unspecified"), "frozen cohort"
                ),
                "class_ids": list(CLASS_IDS),
                "probabilities": probabilities,
                "prediction": nonempty_string(prediction.get("prediction"), "frozen prediction"),
            }
        )
    return {
        "analysis": analysis,
        "strict_rows": strict_rows,
        "frozen_public_strict": frozen_predictions,
        "detailed_by_id": detailed_by_id,
        "frozen_crossfit_status": frozen.get("status"),
    }


def _stable_reconstruction_eligible(
    report_row: Mapping[str, Any], transport_protocol: Mapping[str, Any]
) -> bool:
    checks = mapping(
        mapping(
            transport_protocol.get("paired_stable_reconstruction_sensitivity"),
            "transport sensitivity",
        ).get("checks"),
        "transport sensitivity checks",
    )
    baseline = mapping(report_row.get("baseline_binding"), "report baseline binding")
    norm = mapping(baseline.get("final_norm_reconstruction"), "final norm reconstruction")
    logits = mapping(
        baseline.get("final_logits_reconstruction"), "final logits reconstruction"
    )
    return bool(
        baseline.get("final_layer_top1_matches_greedy")
        is checks.get("captured_and_reconstructed_greedy_top1_match")
        and logits.get("top_k_prefix_token_ids_match")
        is checks.get("captured_and_reconstructed_top5_prefix_match")
        and norm.get("within_tolerance") is checks.get("final_norm_within_existing_strict_tolerances")
        and finite(logits.get("rms_error"), "final-logit RMS error")
        <= finite(
            checks.get("full_final_logits_rms_error_maximum_inclusive"),
            "sensitivity RMS threshold",
        )
        and finite(logits.get("max_abs_error"), "final-logit maximum error")
        <= finite(
            checks.get("full_final_logits_max_abs_error_maximum_inclusive"),
            "sensitivity maximum threshold",
        )
    )


def _load_and_validate_report(
    path: Path,
    *,
    label: str,
    prompt_contract: Mapping[str, Any],
    behavioral_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    raw = json.loads(path.read_bytes())
    result = BEHAVIORAL.validate_report(
        mapping(raw, f"{label} raw report"),
        label=label,
        prompt_contract=prompt_contract,
        protocol=behavioral_protocol,
    )
    del raw
    gc.collect()
    return result


def reconstruct_sensitivity_rows(
    *,
    prompt_path: Path,
    report_paths: Mapping[str, Path],
    behavioral_protocol_value: Mapping[str, Any],
    behavioral_protocol_sha256: str,
    transport_protocol_value: Mapping[str, Any],
    strict_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    behavioral_protocol = BEHAVIORAL.validate_protocol(
        behavioral_protocol_value, protocol_sha256=behavioral_protocol_sha256
    )
    prompt_raw = json.loads(prompt_path.read_bytes())
    prompt_contract = BEHAVIORAL.validate_prompt_bundle(
        prompt_raw, protocol=behavioral_protocol
    )
    del prompt_raw
    gc.collect()
    validated_list = [
        _load_and_validate_report(
            report_paths[label],
            label=label,
            prompt_contract=prompt_contract,
            behavioral_protocol=behavioral_protocol,
        )
        for label in ("public", "nf4", "native")
    ]
    pairing = BEHAVIORAL.validate_report_pairing(validated_list)
    reports = {item["label"]: item for item in validated_list}
    result: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    strict_from_raw: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    eligibility: list[dict[str, Any]] = []
    for row_index, prompt in enumerate(prompt_contract["prompts"]):
        if not prompt["primary"] or prompt["action_status"] != "available":
            continue
        base = {
            "row_id": str(prompt["id"]),
            "task_id": str(prompt["task_id"]),
            "repo": str(prompt["repo"]),
            "cohort_id": str(prompt["cohort_id"]),
            "label": str(prompt["action_label"]),
            "task_request_index": int(prompt["task_request_index"]),
            "checkpoint_ordinal": int(prompt["checkpoint_ordinal"]),
        }
        strict = all(
            reports[label]["rows"][row_index]["numerically_certified"]
            for label in ("public", "nf4", "native")
        )
        stable = all(
            _stable_reconstruction_eligible(
                reports[label]["rows"][row_index], transport_protocol_value
            )
            for label in ("public", "nf4", "native")
        )
        eligibility.append({**base, "strict": strict, "sensitivity": stable})
        if not (strict or stable):
            continue
        for method in METHODS:
            band = BEHAVIORAL._group_band_scores(
                BEHAVIORAL._method_evidence(reports, row_index, method),
                behavioral_protocol["action_classes"],
            )
            feature, band_scores = flatten_raw_fixed_band(band)
            record = {**base, "feature": feature, "band_scores": band_scores}
            if strict:
                strict_from_raw[method].append(record)
            if stable:
                result[method].append(record)

    for method in METHODS:
        expected = {str(row["row_id"]): row for row in strict_rows[method]}
        observed = {str(row["row_id"]): row for row in strict_from_raw[method]}
        require(set(expected) == set(observed), f"{method} strict row IDs differ in raw reports")
        for row_id in expected:
            common_fields = (
                "row_id",
                "task_id",
                "repo",
                "cohort_id",
                "label",
                "task_request_index",
                "feature",
                "band_scores",
            )
            require(
                all(expected[row_id][field] == observed[row_id][field] for field in common_fields),
                f"{method}/{row_id} strict features differ from behavioral analysis",
            )
    return {
        "rows": result,
        "eligibility": eligibility,
        "pairing": pairing,
        "behavioral_protocol": behavioral_protocol,
        "prompt_contract": prompt_contract,
    }


def _class_support(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        class_id: sum(str(row["label"]) == class_id for row in rows)
        for class_id in CLASS_IDS
    }


def _all_classes(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(count > 0 for count in _class_support(rows).values())


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


def fit_multinomial_lbfgs(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c_value: float,
    maximum_iterations: int,
) -> dict[str, Any]:
    require(
        x.ndim == 2
        and len(x) == len(y)
        and x.shape[1] == 96
        and len(x) > 0
        and np.all(np.isfinite(x)),
        "multinomial fit matrix is invalid",
    )
    require(c_value > 0.0 and maximum_iterations > 0, "multinomial fit contract is invalid")
    counts = np.bincount(y, minlength=len(CLASS_IDS)).astype(np.float64)
    require(np.all(counts > 0.0), "multinomial training split lacks a declared class")
    sample_weights = len(y) / (len(CLASS_IDS) * counts[y])
    scaled, mean, scale = _standardize_fit(x)
    parameter_count = len(CLASS_IDS) * x.shape[1] + len(CLASS_IDS)
    weight_parameter_count = len(CLASS_IDS) * x.shape[1]
    curvature_reference = float(len(y)) / len(CLASS_IDS)
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
            class_count=len(CLASS_IDS),
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
        for _ in range(LINE_SEARCH_MAXIMUM_STEPS):
            candidate_theta = theta + step_size * direction
            candidate_objective, candidate_gradient, candidate_original_gradient = evaluate(
                candidate_theta
            )
            line_search_evaluations += 1
            if candidate_objective <= objective + ARMIJO_CONSTANT * step_size * directional_derivative:
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            failure_reason = "armijo_line_search_failed"
            break
        step_vector = candidate_theta - theta
        gradient_delta = candidate_gradient - gradient
        curvature = float(np.dot(step_vector, gradient_delta))
        curvature_floor = 1e-12 * max(
            1.0,
            float(np.linalg.norm(step_vector)) * float(np.linalg.norm(gradient_delta)),
        )
        if curvature > curvature_floor:
            history.append((step_vector, gradient_delta, 1.0 / curvature))
            if len(history) > LBFGS_HISTORY_SIZE:
                history.pop(0)
        theta = candidate_theta
        objective = candidate_objective
        gradient = candidate_gradient
        original_gradient = candidate_original_gradient
        converged = (
            float(np.max(np.abs(original_gradient))) <= GRADIENT_TOLERANCE
        )
    if not converged and failure_reason is None:
        failure_reason = "maximum_iterations_reached"
    feature_count = x.shape[1]
    weights = (
        theta[: len(CLASS_IDS) * feature_count]
        .reshape(len(CLASS_IDS), feature_count)
        * weight_coordinate_scale
    )
    intercept = (
        theta[len(CLASS_IDS) * feature_count :] * intercept_coordinate_scale
    )
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
        "solver_coordinate_gradient_infinity_norm": float(
            np.max(np.abs(gradient))
        ),
        "solver_weight_coordinate_scale": weight_coordinate_scale,
        "solver_intercept_coordinate_scale": intercept_coordinate_scale,
        "c_value": c_value,
        "class_support": {class_id: int(counts[index]) for index, class_id in enumerate(CLASS_IDS)},
        "class_weights": {
            class_id: float(len(y) / (len(CLASS_IDS) * counts[index]))
            for index, class_id in enumerate(CLASS_IDS)
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
    scaled = (x - model["_mean"]) / model["_scale"]
    logits = scaled @ model["_weights"].T + model["_intercept"]
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    require(np.all(np.isfinite(probabilities)), "multinomial prediction is nonfinite")
    return probabilities


def _matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row["feature"] for row in rows], dtype=np.float64)
    y = np.asarray([CLASS_IDS.index(str(row["label"])) for row in rows], dtype=np.int64)
    require(x.shape == (len(rows), 96), "action-layer feature matrix shape changed")
    return x, y


def _prediction_records(
    rows: Sequence[Mapping[str, Any]], probabilities: np.ndarray
) -> list[dict[str, Any]]:
    require(len(rows) == len(probabilities), "prediction row count changed")
    result: list[dict[str, Any]] = []
    for row, probability in zip(rows, probabilities, strict=True):
        values = [float(item) for item in probability]
        prediction_index = max(range(len(CLASS_IDS)), key=lambda index: values[index])
        result.append(
            {
                **{key: str(row[key]) for key in ("row_id", "task_id", "repo", "cohort_id", "label")},
                "task_request_index": int(row["task_request_index"]),
                "checkpoint_ordinal": int(row["checkpoint_ordinal"]),
                "class_ids": list(CLASS_IDS),
                "probabilities": values,
                "prediction": CLASS_IDS[prediction_index],
            }
        )
    return result


def _majority_predictions(
    train_rows: Sequence[Mapping[str, Any]], evaluation_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    support = _class_support(train_rows)
    total = sum(support.values())
    require(total > 0 and all(value > 0 for value in support.values()), "majority training support is incomplete")
    probabilities = np.asarray([support[class_id] / total for class_id in CLASS_IDS])
    return _prediction_records(
        evaluation_rows, np.repeat(probabilities[None, :], len(evaluation_rows), axis=0)
    )


def checkpoint_ordinal_prior_baseline(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the explicitly posthoc temporal prior without using held-out labels."""
    repositories = sorted({str(row["repo"]) for row in rows})
    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for heldout_repository in repositories:
        train_rows = [row for row in rows if row["repo"] != heldout_repository]
        evaluation_rows = [row for row in rows if row["repo"] == heldout_repository]
        global_support = _class_support(train_rows)
        global_total = sum(global_support.values())
        require(global_total > 0, "ordinal prior outer training fold is empty")
        global_probabilities = np.asarray(
            [global_support[class_id] / global_total for class_id in CLASS_IDS],
            dtype=np.float64,
        )
        ordinal_support: dict[int, dict[str, int]] = {}
        for ordinal in sorted({int(row["checkpoint_ordinal"]) for row in train_rows}):
            ordinal_support[ordinal] = _class_support(
                [row for row in train_rows if int(row["checkpoint_ordinal"]) == ordinal]
            )
        fallback_count = 0
        for row in evaluation_rows:
            ordinal = int(row["checkpoint_ordinal"])
            support = ordinal_support.get(ordinal)
            total = sum(support.values()) if support is not None else 0
            if total == 0:
                probabilities = global_probabilities
                fallback_count += 1
            else:
                probabilities = np.asarray(
                    [support[class_id] / total for class_id in CLASS_IDS],
                    dtype=np.float64,
                )
            predictions.extend(_prediction_records([row], probabilities[None, :]))
        folds.append(
            {
                "heldout_repository": heldout_repository,
                "training_row_count": len(train_rows),
                "evaluation_row_count": len(evaluation_rows),
                "heldout_labels_used_to_estimate_priors": False,
                "global_training_support": global_support,
                "ordinal_training_support": {
                    str(ordinal): support
                    for ordinal, support in sorted(ordinal_support.items())
                },
                "global_prior_fallback_evaluation_row_count": fallback_count,
            }
        )
    order = {str(row["row_id"]): index for index, row in enumerate(rows)}
    predictions.sort(key=lambda item: order[str(item["row_id"])])
    require(
        {str(row["row_id"]) for row in predictions} == set(order),
        "ordinal prior lacks complete outer-held-out prediction coverage",
    )
    return {
        "status": "POSTHOC_DESCRIPTIVE_ONLY",
        "reason_added": (
            "action-label checkpoint-ordinal distribution inspected after the frozen "
            "primary comparison protocol"
        ),
        "definition": (
            "empirical action-class distribution at the same checkpoint ordinal in "
            "the outer training repositories; empirical global outer-training prior "
            "when that ordinal is absent"
        ),
        "laplace_smoothing": 0.0,
        "used_for_feature_or_hyperparameter_selection": False,
        "used_for_paired_primary_comparisons": False,
        "used_for_material_effect_signals_or_classification": False,
        "folds": folds,
        "predictions": predictions,
        "metrics": BEHAVIORAL.classification_metrics(predictions, list(CLASS_IDS)),
    }


def _public_model_record(model: Mapping[str, Any]) -> dict[str, Any]:
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


def nested_leave_one_repository_out(
    rows: Sequence[Mapping[str, Any]],
    *,
    c_grid: Sequence[float],
    maximum_iterations: int,
    minimum_valid_inner_folds: int,
) -> dict[str, Any]:
    require(bool(rows), "cannot evaluate an empty action-layer track")
    row_ids = [str(row["row_id"]) for row in rows]
    require(len(row_ids) == len(set(row_ids)), "action-layer rows contain duplicate IDs")
    task_repositories: dict[str, set[str]] = {}
    for row in rows:
        task_repositories.setdefault(str(row["task_id"]), set()).add(str(row["repo"]))
    require(
        all(len(repositories) == 1 for repositories in task_repositories.values()),
        "a task crosses repository folds",
    )
    repositories = sorted({str(row["repo"]) for row in rows})
    predictions: list[dict[str, Any]] = []
    majority_predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for outer_repository in repositories:
        outer_train = [row for row in rows if row["repo"] != outer_repository]
        outer_evaluation = [row for row in rows if row["repo"] == outer_repository]
        fold: dict[str, Any] = {
            "heldout_repository": outer_repository,
            "evaluation_row_count": len(outer_evaluation),
            "evaluation_task_ids": sorted({str(row["task_id"]) for row in outer_evaluation}),
            "training_repository_count": len({str(row["repo"]) for row in outer_train}),
            "training_support": _class_support(outer_train),
            "outer_heldout_labels_used_for_model_selection": False,
            "candidate_models": [],
        }
        if not outer_evaluation or not _all_classes(outer_train):
            fold["status"] = "insufficient_outer_training_class_support"
            folds.append(fold)
            continue
        candidate_results: list[dict[str, Any]] = []
        inner_repositories = sorted({str(row["repo"]) for row in outer_train})
        for c_value in c_grid:
            inner_predictions: list[dict[str, Any]] = []
            fit_records: list[dict[str, Any]] = []
            skipped: list[str] = []
            for inner_repository in inner_repositories:
                inner_train = [
                    row for row in outer_train if row["repo"] != inner_repository
                ]
                inner_evaluation = [
                    row for row in outer_train if row["repo"] == inner_repository
                ]
                if not inner_evaluation or not _all_classes(inner_train):
                    skipped.append(inner_repository)
                    continue
                x_train, y_train = _matrix(inner_train)
                model = fit_multinomial_lbfgs(
                    x_train,
                    y_train,
                    c_value=float(c_value),
                    maximum_iterations=maximum_iterations,
                )
                fit_records.append(
                    {
                        "heldout_repository": inner_repository,
                        **_public_model_record(model),
                    }
                )
                if not model["converged"]:
                    continue
                x_evaluation, _ = _matrix(inner_evaluation)
                inner_predictions.extend(
                    _prediction_records(
                        inner_evaluation, predict_multinomial(model, x_evaluation)
                    )
                )
            valid_fits = [item for item in fit_records if item["converged"]]
            candidate: dict[str, Any] = {
                "c_value": float(c_value),
                "valid_inner_fold_count": len(valid_fits),
                "skipped_inner_repositories_missing_training_class": skipped,
                "all_attempted_fits_converged": len(valid_fits) == len(fit_records),
                "fit_iteration_maximum": max(
                    (int(item["iterations"]) for item in fit_records), default=0
                ),
                "fit_gradient_infinity_norm_maximum": max(
                    (float(item["gradient_infinity_norm"]) for item in fit_records),
                    default=None,
                ),
            }
            if (
                len(valid_fits) < minimum_valid_inner_folds
                or len(valid_fits) != len(fit_records)
                or not _all_classes(inner_predictions)
            ):
                candidate.update(
                    {
                        "status": "insufficient_inner_support_or_convergence",
                        "balanced_accuracy": None,
                    }
                )
            else:
                metrics = BEHAVIORAL.classification_metrics(
                    inner_predictions, list(CLASS_IDS)
                )
                candidate.update(
                    {
                        "status": "available",
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "inner_validation_row_count": len(inner_predictions),
                        "inner_validation_payload_sha256": sha256_json(
                            inner_predictions
                        ),
                    }
                )
            candidate_results.append(candidate)
        fold["candidate_models"] = candidate_results
        available = [item for item in candidate_results if item["status"] == "available"]
        if len(available) != len(c_grid):
            fold["status"] = "incomplete_regularization_grid"
            folds.append(fold)
            continue
        selected = min(
            available,
            key=lambda item: (-float(item["balanced_accuracy"]), float(item["c_value"])),
        )
        x_train, y_train = _matrix(outer_train)
        final_model = fit_multinomial_lbfgs(
            x_train,
            y_train,
            c_value=float(selected["c_value"]),
            maximum_iterations=maximum_iterations,
        )
        fold["selected_c"] = selected["c_value"]
        fold["selected_inner_balanced_accuracy"] = selected["balanced_accuracy"]
        fold["outer_model"] = _public_model_record(final_model)
        fold["training_payload_sha256"] = sha256_json(
            [
                {
                    "row_id": row["row_id"],
                    "task_id": row["task_id"],
                    "repo": row["repo"],
                    "label": row["label"],
                    "feature": row["feature"],
                }
                for row in sorted(outer_train, key=lambda item: str(item["row_id"]))
            ]
        )
        if not final_model["converged"]:
            fold["status"] = "outer_model_nonconverged"
            folds.append(fold)
            continue
        x_evaluation, _ = _matrix(outer_evaluation)
        predictions.extend(
            _prediction_records(
                outer_evaluation, predict_multinomial(final_model, x_evaluation)
            )
        )
        majority_predictions.extend(_majority_predictions(outer_train, outer_evaluation))
        fold["status"] = "available"
        folds.append(fold)
    order = {row_id: index for index, row_id in enumerate(row_ids)}
    predictions.sort(key=lambda item: order[item["row_id"]])
    majority_predictions.sort(key=lambda item: order[item["row_id"]])
    predicted_ids = {str(row["row_id"]) for row in predictions}
    complete = predicted_ids == set(row_ids)
    status = (
        "available"
        if complete and all(fold["status"] == "available" for fold in folds)
        else "incomplete_nested_repository_crossfit"
    )
    return {
        "status": status,
        "algorithm": "nested_leave_one_repository_out_l2_multinomial_logistic_v1",
        "row_count": len(rows),
        "repository_count": len(repositories),
        "complete_prediction_coverage": complete,
        "successful_fold_count": sum(fold["status"] == "available" for fold in folds),
        "fold_count": len(folds),
        "folds": folds,
        "predictions": predictions,
        "metrics": BEHAVIORAL.classification_metrics(predictions, list(CLASS_IDS)),
        "majority_baseline": {
            "definition": "empirical class prior over the complete outer training fold",
            "predictions": majority_predictions,
            "metrics": BEHAVIORAL.classification_metrics(
                majority_predictions, list(CLASS_IDS)
            ),
        },
    }


def _coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_counts: dict[str, int] = {}
    for row in rows:
        task_counts[str(row["task_id"])] = task_counts.get(str(row["task_id"]), 0) + 1
    return {
        "row_count": len(rows),
        "task_count": len(task_counts),
        "repository_count": len({str(row["repo"]) for row in rows}),
        "class_support": _class_support(rows),
        "minimum_rows_per_task": min(task_counts.values(), default=0),
        "task_row_counts": dict(sorted(task_counts.items())),
    }


def _support(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    method_results: Mapping[str, Mapping[str, Any]],
    sensitivity: bool,
) -> dict[str, Any]:
    coverage = _coverage(rows)
    checks = {
        "minimum_rows": coverage["row_count"] >= int(contract["minimum_rows"]),
        "minimum_tasks": coverage["task_count"] >= int(contract["minimum_tasks"]),
        "minimum_repositories": coverage["repository_count"]
        >= int(contract["minimum_repositories"]),
        "all_classes": all(value > 0 for value in coverage["class_support"].values()),
        "all_methods_complete_nested_crossfit": all(
            result["status"] == "available" for result in method_results.values()
        ),
    }
    if sensitivity:
        checks["minimum_rows_per_task"] = coverage["minimum_rows_per_task"] >= int(
            contract["minimum_rows_per_task"]
        )
    return {**coverage, "checks": checks, "complete": all(checks.values())}


def _normalize_frozen_predictions(
    predictions: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    row_ids = [str(row["row_id"]) for row in rows]
    prediction_ids = [str(item["row_id"]) for item in predictions]
    require(
        len(prediction_ids) == len(set(prediction_ids)),
        "frozen public predictions contain duplicate row IDs",
    )
    require(
        set(prediction_ids) == set(row_ids) and len(prediction_ids) == len(row_ids),
        "frozen public prediction coverage differs from eligible rows",
    )
    by_id = {str(item["row_id"]): item for item in predictions}
    result = []
    for row in rows:
        row_id = str(row["row_id"])
        prediction = dict(by_id[row_id])
        for field in ("task_id", "repo", "label", "cohort_id"):
            require(
                str(prediction[field]) == str(row[field]),
                f"frozen public prediction differs in {field}: {row_id}",
            )
        prediction["task_request_index"] = int(row["task_request_index"])
        prediction["checkpoint_ordinal"] = int(row["checkpoint_ordinal"])
        result.append(prediction)
    return result


def _descriptive_slices(
    predictions_by_method: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method, predictions in predictions_by_method.items():
        slices: dict[str, Any] = {}
        group_specs = {
            "checkpoint_ordinal": lambda row: str(int(row["checkpoint_ordinal"])),
            "cohort": lambda row: str(row["cohort_id"]),
            "cohort_by_checkpoint_ordinal": lambda row: (
                f"{row['cohort_id']}::{int(row['checkpoint_ordinal'])}"
            ),
        }
        for slice_name, key_function in group_specs.items():
            grouped: dict[str, list[Mapping[str, Any]]] = {}
            for row in predictions:
                grouped.setdefault(key_function(row), []).append(row)
            slices[slice_name] = {
                key: {
                    "row_count": len(rows),
                    "task_count": len({str(row["task_id"]) for row in rows}),
                    "repository_count": len({str(row["repo"]) for row in rows}),
                    "class_support": _class_support(rows),
                    "metrics": BEHAVIORAL.classification_metrics(
                        rows, list(CLASS_IDS)
                    ),
                }
                for key, rows in sorted(grouped.items())
            }
        result[method] = slices
    return {
        "status": "descriptive_only_noninferential",
        "used_for_feature_or_C_selection": False,
        "used_for_material_effect_signals": False,
        "cohort_subgroups_are_independent_replication": False,
        "methods": result,
    }


def analyze_track(
    rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    frozen_public_predictions: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    track_name: str,
    bootstrap_samples: int,
    minimum_valid_bootstrap_fraction: float,
) -> dict[str, Any]:
    reference_ids = [str(row["row_id"]) for row in rows_by_method["public_jacobian"]]
    for method in METHODS:
        require(
            [str(row["row_id"]) for row in rows_by_method[method]] == reference_ids,
            f"{track_name}/{method} row coverage or order differs",
        )
    method_results = {
        method: nested_leave_one_repository_out(
            rows_by_method[method],
            c_grid=protocol["c_grid"],
            maximum_iterations=protocol["maximum_iterations"],
            minimum_valid_inner_folds=protocol["minimum_valid_inner_folds"],
        )
        for method in METHODS
    }
    rows = rows_by_method["public_jacobian"]
    frozen = _normalize_frozen_predictions(frozen_public_predictions, rows)
    comparison_specs = [
        (
            "learned_public_jacobian_minus_frozen_public_jacobian_readout",
            method_results["public_jacobian"]["predictions"],
            frozen,
            "learned_public_jacobian",
            "frozen_public_jacobian_readout",
        ),
        (
            "learned_public_jacobian_minus_learned_ordinary_logit",
            method_results["public_jacobian"]["predictions"],
            method_results["ordinary_logit"]["predictions"],
            "learned_public_jacobian",
            "learned_ordinary_logit",
        ),
        (
            "learned_public_jacobian_minus_learned_native_jacobian",
            method_results["public_jacobian"]["predictions"],
            method_results["native_jacobian"]["predictions"],
            "learned_public_jacobian",
            "learned_native_jacobian",
        ),
        (
            "learned_public_jacobian_minus_learned_nf4_jacobian",
            method_results["public_jacobian"]["predictions"],
            method_results["nf4_jacobian"]["predictions"],
            "learned_public_jacobian",
            "learned_nf4_jacobian",
        ),
    ]
    comparisons: dict[str, Any] = {}
    for index, (name, candidate, reference, candidate_name, reference_name) in enumerate(
        comparison_specs
    ):
        comparisons[name] = {
            "candidate": candidate_name,
            "reference": reference_name,
            **BEHAVIORAL.bootstrap_paired_classification(
                candidate,
                reference,
                list(CLASS_IDS),
                samples=bootstrap_samples,
                seed=int(protocol["paired"]["seed"]) + index,
                confidence_level=float(protocol["paired"]["confidence_level"]),
                minimum_valid_fraction=minimum_valid_bootstrap_fraction,
            ),
        }
    track_contract = protocol["track_contracts"][
        "sensitivity" if track_name == "sensitivity" else "strict_primary"
    ]
    support = _support(
        rows,
        contract=track_contract,
        method_results=method_results,
        sensitivity=track_name == "sensitivity",
    )
    rules = mapping(protocol["value"].get("material_effect_rules"), "material rules")
    refinement_rule = mapping(rules.get("readout_refinement_signal"), "refinement rule")
    native_rule = mapping(rules.get("native_fit_capacity_signal"), "native rule")
    no_refit_rule = mapping(rules.get("no_native_refit_signal"), "no-refit rule")

    def comparison_signal(name: str, minimum: float) -> bool:
        comparison = comparisons[name]
        interval = comparison.get("intervals", {}).get("balanced_accuracy_gain")
        point = comparison.get("observed_benefit_deltas", {}).get(
            "balanced_accuracy_gain"
        )
        return bool(
            comparison.get("status") == "available"
            and interval is not None
            and point is not None
            and float(point) >= minimum
            and float(interval["lower"]) > 0.0
        )

    refinement = support["complete"] and comparison_signal(
        "learned_public_jacobian_minus_frozen_public_jacobian_readout",
        float(
            refinement_rule[
                "learned_public_minus_frozen_public_balanced_accuracy_minimum_inclusive"
            ]
        ),
    )
    native_capacity = refinement and comparison_signal(
        "learned_public_jacobian_minus_learned_native_jacobian",
        float(
            native_rule[
                "learned_public_minus_learned_native_balanced_accuracy_minimum_inclusive"
            ]
        ),
    )
    native_comparison = comparisons[
        "learned_public_jacobian_minus_learned_native_jacobian"
    ]
    native_interval = native_comparison.get("intervals", {}).get(
        "balanced_accuracy_gain"
    )
    no_native_refit = bool(
        support["complete"]
        and native_comparison.get("status") == "available"
        and native_interval is not None
        and float(native_interval["upper"])
        < float(
            no_refit_rule[
                "learned_public_minus_learned_native_confidence_interval_upper_maximum_exclusive"
            ]
        )
    )
    ordinal_prior = checkpoint_ordinal_prior_baseline(rows)
    predictions_for_slices = {
        method: method_results[method]["predictions"] for method in METHODS
    }
    predictions_for_slices["frozen_public_jacobian_readout"] = frozen
    predictions_for_slices["training_fold_majority"] = method_results[
        "ordinary_logit"
    ]["majority_baseline"]["predictions"]
    predictions_for_slices["POSTHOC_checkpoint_ordinal_prior"] = ordinal_prior[
        "predictions"
    ]
    return {
        "role": (
            "primary_supplemental_readout_refinement_track"
            if track_name == "strict_primary"
            else "post_public_numeric_diagnostic_sensitivity_only"
        ),
        "primary_decision_override_forbidden": track_name == "sensitivity",
        "support": support,
        "methods": method_results,
        "frozen_public_readout": {
            "prediction_count": len(frozen),
            "complete_prediction_coverage": len(frozen) == len(rows),
            "metrics": BEHAVIORAL.classification_metrics(frozen, list(CLASS_IDS)),
            "predictions": frozen,
        },
        "paired_comparisons": comparisons,
        "posthoc_descriptive_checkpoint_ordinal_prior": ordinal_prior,
        "descriptive_checkpoint_and_cohort_slices": _descriptive_slices(
            predictions_for_slices
        ),
        "signals": {
            "readout_refinement_signal": refinement,
            "native_fit_capacity_signal": native_capacity,
            "no_native_refit_signal": no_native_refit,
            "actionable": support["complete"],
        },
        "bootstrap_is_conditional_on_frozen_nested_crossfit_predictions": True,
        "models_refit_inside_bootstrap": False,
    }


def build_analysis(
    *,
    behavioral_analysis_value: Any,
    behavioral_protocol_value: Any,
    transport_protocol_value: Any,
    action_protocol_value: Any,
    input_hashes: Mapping[str, str],
    prompt_path: Path,
    report_paths: Mapping[str, Path],
    bootstrap_samples: int,
) -> dict[str, Any]:
    action_protocol = validate_protocol(
        action_protocol_value,
        behavioral_protocol_sha256=input_hashes["behavioral_protocol"],
        transport_protocol_sha256=input_hashes["transport_protocol"],
    )
    require(
        input_hashes["prompts"] == action_protocol["pins"]["prompts"],
        "supplied prompt bundle differs from the action-layer protocol pin",
    )
    transport = mapping(transport_protocol_value, "transport protocol")
    require(
        transport.get("schema_version") == 2
        and transport.get("id") == "swe-n20-greedy-next-token-transport-v2"
        and mapping(transport.get("scope"), "transport scope").get(
            "prompt_bundle_sha256"
        )
        == action_protocol["pins"]["prompts"],
        "transport sensitivity protocol identity or prompt pin changed",
    )
    behavior = validate_behavioral_analysis(
        behavioral_analysis_value,
        input_hashes={
            "prompts": input_hashes["prompts"],
            "public_report": input_hashes["public_report"],
            "nf4_report": input_hashes["nf4_report"],
            "native_report": input_hashes["native_report"],
            "protocol": input_hashes["behavioral_protocol"],
        },
        protocol=action_protocol,
    )
    sensitivity = reconstruct_sensitivity_rows(
        prompt_path=prompt_path,
        report_paths=report_paths,
        behavioral_protocol_value=mapping(
            behavioral_protocol_value, "behavioral protocol"
        ),
        behavioral_protocol_sha256=input_hashes["behavioral_protocol"],
        transport_protocol_value=transport,
        strict_rows=behavior["strict_rows"],
    )
    ordinal_by_id = {
        str(row["row_id"]): {
            "checkpoint_ordinal": int(row["checkpoint_ordinal"]),
            "task_request_index": int(row["task_request_index"]),
        }
        for row in sensitivity["eligibility"]
    }
    strict_rows = {
        method: [
            {**row, **ordinal_by_id[str(row["row_id"])]}
            for row in behavior["strict_rows"][method]
        ]
        for method in METHODS
    }
    behavioral_protocol = sensitivity["behavioral_protocol"]
    minimum_valid_fraction = float(
        mapping(behavioral_protocol.get("bootstrap"), "behavioral bootstrap")[
            "minimum_valid_fraction"
        ]
    )
    strict_track = analyze_track(
        strict_rows,
        frozen_public_predictions=behavior["frozen_public_strict"],
        protocol=action_protocol,
        track_name="strict_primary",
        bootstrap_samples=bootstrap_samples,
        minimum_valid_bootstrap_fraction=minimum_valid_fraction,
    )
    sensitivity_public_band_rows = [
        {
            **{
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
            },
            "scores": row["band_scores"],
        }
        for row in sensitivity["rows"]["public_jacobian"]
    ]
    all_repositories = sorted(
        {str(row["repo"]) for row in sensitivity_public_band_rows}
    )
    frozen_sensitivity = BEHAVIORAL.crossfit_track(
        sensitivity_public_band_rows,
        class_ids=list(CLASS_IDS),
        all_repositories=all_repositories,
        contract=behavioral_protocol["crossfit"],
    )
    sensitivity_track = analyze_track(
        sensitivity["rows"],
        frozen_public_predictions=frozen_sensitivity["predictions"],
        protocol=action_protocol,
        track_name="sensitivity",
        bootstrap_samples=bootstrap_samples,
        minimum_valid_bootstrap_fraction=minimum_valid_fraction,
    )
    strict_signals = strict_track["signals"]
    if not strict_track["support"]["complete"]:
        classification = "insufficient_strict_support"
        next_step = "treat sensitivity as descriptive and collect strict numerical coverage"
    elif strict_signals["native_fit_capacity_signal"]:
        classification = "native_fit_capacity_signal"
        next_step = "investigate a larger native fit without overriding the frozen semantic decision"
    elif strict_signals["readout_refinement_signal"] and strict_signals[
        "no_native_refit_signal"
    ]:
        classification = "readout_refinement_signal_no_native_refit_signal"
        next_step = "retain lenses and use the learned fixed-band action readout on more tasks"
    elif strict_signals["readout_refinement_signal"]:
        classification = "readout_refinement_signal_native_comparison_inconclusive"
        next_step = "expand the unchanged held-out action evaluation before considering refit"
    elif strict_signals["no_native_refit_signal"]:
        classification = "no_readout_refinement_signal_no_native_refit_signal"
        next_step = "refine semantic labels or targets rather than refitting the native lens"
    else:
        classification = "no_predeclared_signal"
        next_step = "collect additional fixed action controls before changing lens or readout"
    ledger = []
    strict_ids = {
        str(row["row_id"]) for row in strict_rows["public_jacobian"]
    }
    sensitivity_ids = {
        str(row["row_id"])
        for row in sensitivity["rows"]["public_jacobian"]
    }
    for row in sensitivity["eligibility"]:
        ledger.append(
            {
                **row,
                "strict_feature_retained": row["row_id"] in strict_ids,
                "sensitivity_feature_retained": row["row_id"] in sensitivity_ids,
            }
        )
    return {
        "schema_version": 1,
        "kind": "swe_n20_action_layer_nested_readout_analysis",
        "status": "complete",
        "classification": classification,
        "next_step": next_step,
        "inputs": dict(input_hashes),
        "protocol": {
            "sha256": input_hashes["action_protocol"],
            "id": action_protocol_value["id"],
            "feature_order": "layer-major then class order",
            "layers": list(LAYERS),
            "class_ids": list(CLASS_IDS),
            "feature_count": 96,
            "model": {
                **dict(action_protocol_value["model_contract"]),
                "objective_normalization": (
                    "sum class-balanced negative log likelihood plus "
                    "L2(weights)/(2*C); intercept is unpenalized"
                ),
                "gradient_infinity_tolerance": GRADIENT_TOLERANCE,
                "lbfgs_history_size": LBFGS_HISTORY_SIZE,
                "line_search": "Armijo backtracking",
                "solver_preconditioning": (
                    "diagonal curvature-reference scaling of weight and intercept "
                    "search coordinates; require the unchanged gradient tolerance in "
                    "original W/intercept coordinates"
                ),
            },
            "bootstrap": {
                **dict(action_protocol_value["paired_inference"]),
                "samples": bootstrap_samples,
                "minimum_valid_fraction_inherited_from_frozen_behavioral_protocol": minimum_valid_fraction,
            },
        },
        "raw_report_revalidation": {
            "required_for_sensitivity_rows_omitted_by_strict_behavioral_analysis": True,
            "strict_features_exactly_equal_to_behavioral_analysis": True,
            "pairing": sensitivity["pairing"],
        },
        "eligibility_ledger": ledger,
        "tracks": {
            "strict_primary": strict_track,
            "paired_stable_reconstruction_sensitivity": sensitivity_track,
        },
        "decision_audit": {
            "strict_track_controls_classification": True,
            "sensitivity_track_can_override_primary": False,
            "cohort_subgroups_are_independent_replication": False,
            "best_layer_or_feature_selection_performed": False,
            "outer_heldout_repository_used_for_scaling_class_weights_or_C_selection": False,
            "bootstrap_refit_models": False,
            "interpretation": (
                "behavioral vocabulary-feature readout, not hidden prose, chain of thought, "
                "or a replacement Jacobian-lens fit"
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavioral-analysis", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument("--nf4-report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--behavioral-protocol", type=Path, default=DEFAULT_BEHAVIORAL_PROTOCOL
    )
    parser.add_argument(
        "--transport-protocol", type=Path, default=DEFAULT_TRANSPORT_PROTOCOL
    )
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {
        "behavioral_analysis": args.behavioral_analysis.expanduser().resolve(strict=True),
        "prompts": args.prompts.expanduser().resolve(strict=True),
        "public_report": args.public_report.expanduser().resolve(strict=True),
        "nf4_report": args.nf4_report.expanduser().resolve(strict=True),
        "native_report": args.native_report.expanduser().resolve(strict=True),
        "action_protocol": args.protocol.expanduser().resolve(strict=True),
        "behavioral_protocol": args.behavioral_protocol.expanduser().resolve(strict=True),
        "transport_protocol": args.transport_protocol.expanduser().resolve(strict=True),
    }
    input_hashes = {key: sha256_file(path) for key, path in paths.items()}
    action_protocol_value = mapping(
        json.loads(paths["action_protocol"].read_bytes()), "action protocol"
    )
    configured_samples = integer(
        mapping(action_protocol_value.get("paired_inference"), "paired inference").get(
            "samples"
        ),
        "configured bootstrap samples",
        minimum=1,
    )
    bootstrap_samples = (
        configured_samples if args.bootstrap_samples is None else args.bootstrap_samples
    )
    require(bootstrap_samples >= 0, "bootstrap samples must be nonnegative")
    analysis = build_analysis(
        behavioral_analysis_value=json.loads(paths["behavioral_analysis"].read_bytes()),
        behavioral_protocol_value=json.loads(paths["behavioral_protocol"].read_bytes()),
        transport_protocol_value=json.loads(paths["transport_protocol"].read_bytes()),
        action_protocol_value=action_protocol_value,
        input_hashes=input_hashes,
        prompt_path=paths["prompts"],
        report_paths={
            "public": paths["public_report"],
            "nf4": paths["nf4_report"],
            "native": paths["native_report"],
        },
        bootstrap_samples=bootstrap_samples,
    )
    output = args.output.expanduser().resolve()
    atomic_write_json(output, analysis)
    print(
        f"wrote {output} (sha256={sha256_file(output)}, "
        f"classification={analysis['classification']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
