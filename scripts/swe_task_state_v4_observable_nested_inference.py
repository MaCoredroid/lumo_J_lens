#!/usr/bin/env python3
"""Exploratory repository inference for frozen V4 nested observable decoders.

The analysis authenticates a saved development-only decoder report and the
permitted V4 alignment/observable-label sidecar, reconstructs the exact OOF row
set, and independently recomputes paired NLL/Brier deltas from saved
probabilities.  It does not refit a model and makes no private-COT or emotion
claim.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scripts import swe_task_state_v4_observable_decoder as DECODER
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    import swe_task_state_v4_observable_decoder as DECODER  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_observable_nested_inference.json"
SCHEMA_VERSION = 1
KIND = "swe_task_state_v4_observable_nested_fixed_oof_repository_inference_report"
TARGET = "observable_rationale_language_marker"
REFERENCE = "sequence_logit_j"
METRICS = ("negative_log_likelihood", "multiclass_brier")
COMPARISONS = (
    {
        "candidate": "sequence_logit_j_plus_raw_activation_current",
        "reference": REFERENCE,
    },
    {
        "candidate": "sequence_logit_j_plus_public_j_activation_current",
        "reference": REFERENCE,
    },
)
EXPECTED_INPUTS = {
    "decoder_report": {
        "path": ".cache/swe_task_state_v4_raw_capture/n60-final/observable-rationale-marker-v2f.json",
        "sha256": "5b7a9f5f152548f7d63d45ee41b2a1b9b338525e085af23d6bdcc3b9fc15c11c",
    },
    "alignment_index": {
        "path": ".cache/swe_task_state_v4_raw_capture/n60-final/alignment-index-v4.json",
        "sha256": "c0d9e4bd6ad8f6962cf58776441ec4ef042eebbbae8a9959b4149e605c9ae38e",
    },
    "observable_label_sidecar": {
        "path": ".cache/swe_task_state_v4_raw_capture/n60-final/observable-events-v4.json",
        "sha256": "3e54d3510fea892059369b2f7cbac93381504fcc22a2732bb8fca5252a42cf18",
    },
}
FAMILY_ORDER = tuple(
    f"{comparison['candidate']}__vs__{REFERENCE}::{metric}"
    for comparison in COMPARISONS
    for metric in METRICS
)


class InferenceError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InferenceError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return sha256_bytes(header + b"\0" + array.tobytes(order="C"))


def load_json(path: Path) -> Any:
    try:
        return DECODER.load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, DECODER.DecoderError) as error:
        raise InferenceError(f"cannot load strict JSON: {path}: {error}") from error


def validate_config(value: Any) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "nested-inference config must be an object")
    config = dict(value)
    _require(
        set(config)
        == {
            "schema_version",
            "id",
            "status",
            "path_guard",
            "inputs",
            "target",
            "reference",
            "comparisons",
            "metrics",
            "repository_inference",
            "multiple_testing",
            "interpretation",
            "claim_scope",
            "mandatory_limitations",
        },
        "nested-inference config schema changed",
    )
    _require(
        config["schema_version"] == 1
        and config["id"]
        == "swe-task-state-v4-observable-nested-fixed-oof-repository-inference"
        and config["status"] == "development_only_reserved_validation_closed"
        and config["path_guard"]
        == {"forbidden_path_fragments": ["reserved", "validation"]}
        and config["inputs"] == EXPECTED_INPUTS
        and config["target"] == TARGET
        and config["reference"] == REFERENCE
        and config["comparisons"] == list(COMPARISONS)
        and config["metrics"] == list(METRICS),
        "nested-inference identity, inputs, or frozen comparison family changed",
    )
    repository = config["repository_inference"]
    _require(
        repository
        == {
            "repository_count": 10,
            "within_repository_weighting": "equal_task_then_equal_available_row_within_task",
            "estimand": "unweighted_mean_of_ten_heldout_repository_paired_deltas",
            "student_t_interval": {
                "confidence_level": 0.95,
                "degrees_of_freedom": 9,
                "assumption": "repository_deltas_are_iid",
            },
            "exact_sign_flip": {
                "sign_values": [-1, 1],
                "assignment_count": 1024,
                "statistic": "absolute_unweighted_mean_repository_delta",
                "alternative": "two_sided",
                "observed_assignment_included": True,
                "plus_one_correction": False,
            },
        },
        "repository-inference contract changed",
    )
    _require(
        config["multiple_testing"]
        == {
            "method": "holm_step_down",
            "family_size": 4,
            "family_order": list(FAMILY_ORDER),
        },
        "multiple-testing family changed",
    )
    interpretation = config["interpretation"]
    _require(
        interpretation
        == {
            "scope": "exploratory_development_fixed_oof_repository_cluster_inference_only",
            "configured_after_point_results_were_observed": True,
            "nested_variants_added_after_initial_oof_results_were_observed": True,
            "comparisons_preregistered": False,
            "post_hoc_exploratory_only": True,
            "fixed_oof_predictions": True,
            "model_refit_uncertainty_included": False,
            "model_or_feature_selection_uncertainty_included": False,
            "repository_iid_assumption_required_for_student_t_interval": True,
            "repository_sign_symmetry_required_for_exact_sign_flip_test": True,
            "untouched_confirmation_set_evaluated": False,
            "transport_or_confirmation_claim_allowed": False,
        },
        "nested-inference interpretation changed",
    )
    claims = config["claim_scope"]
    _require(
        isinstance(claims, Mapping)
        and set(claims)
        == {
            "private_chain_of_thought_reconstructed",
            "cot_or_cot_like_decoding_established",
            "semantic_sentence_or_chain_decoding_established",
            "hidden_thought_understanding_or_intent_established",
            "emotion_decoding_established",
            "causal_interpretation_established",
            "incremental_value_confirmed",
            "operational_reliability_established",
        }
        and all(value is False for value in claims.values()),
        "nested-inference claim boundary changed",
    )
    limitations = config["mandatory_limitations"]
    _require(
        isinstance(limitations, list)
        and len(limitations) == 4
        and all(isinstance(item, str) and bool(item) for item in limitations),
        "mandatory limitations changed",
    )
    return config


def _resolve_authenticated_input(record: Mapping[str, str], label: str) -> Path:
    _require(
        set(record) == {"path", "sha256"}
        and isinstance(record["path"], str)
        and bool(record["path"])
        and _is_sha256(record["sha256"]),
        f"{label} binding is invalid",
    )
    raw = Path(record["path"])
    path = raw if raw.is_absolute() else ROOT / raw
    DECODER.frozen_lexical_path_preflight([path])
    DECODER.frozen_canonical_path_preflight(input_paths=[path], output_paths=[])
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    resolved = path.resolve(strict=True)
    _require(
        resolved.is_file() and sha256_file(resolved) == record["sha256"],
        f"{label} hash changed",
    )
    return resolved


def available_rows_and_labels(
    alignment_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    _require(
        len(alignment_rows) == len(label_rows) == 1708,
        "alignment and label-sidecar row counts differ",
    )
    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    grouping = (
        "global_index",
        "source_id_sha256",
        "task_id_sha256",
        "repository",
        "request_index",
    )
    for alignment, label_row in zip(alignment_rows, label_rows, strict=True):
        _require(
            all(label_row[key] == alignment[key] for key in grouping),
            "alignment and label-sidecar grouping differ",
        )
        if not alignment["stable_feature_eligible"]:
            continue
        record = label_row["targets"][target]
        if record["status"] == "unknown":
            continue
        rows.append({key: alignment[key] for key in grouping})
        labels.append(str(record["value"]))
    _require(
        len(rows) == len(labels) == 1549
        and set(labels) == {"no", "yes"},
        "available primary-target row set changed",
    )
    return rows, labels


def hierarchical_row_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Independently implement equal repo, then task, then row weighting."""

    _require(bool(rows), "hierarchical weights require rows")
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for position, row in enumerate(rows):
        repository = row.get("repository")
        task = row.get("task_id_sha256")
        _require(
            isinstance(repository, str)
            and bool(repository)
            and isinstance(task, str)
            and bool(task),
            "weighting grouping is invalid",
        )
        groups[repository][task].append(position)
    weights = np.zeros(len(rows), dtype=np.float64)
    for tasks in groups.values():
        for positions in tasks.values():
            weights[np.asarray(positions, dtype=np.int64)] = (
                1.0 / len(groups) / len(tasks) / len(positions)
            )
    _require(
        bool(np.all(weights > 0.0))
        and abs(float(weights.sum()) - 1.0) <= 1e-12,
        "hierarchical weights are invalid",
    )
    return weights


def validate_decoder_report(
    value: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], Mapping[str, Any]]:
    _require(isinstance(value, Mapping), "decoder report must be an object")
    report = dict(value)
    _require(
        report.get("schema_version") == 1
        and report.get("kind") == DECODER.REPORT_KIND
        and report.get("status") == "passed"
        and report.get("status_scope") == DECODER.STATUS_SCOPE
        and report.get("scope") == "development_only_reserved_validation_closed",
        "decoder report identity changed",
    )
    _require(
        report.get("config")
        == {
            "path": str(DECODER.CONFIG_PATH.relative_to(ROOT)),
            "sha256": sha256_file(DECODER.CONFIG_PATH),
            "size_bytes": DECODER.CONFIG_PATH.stat().st_size,
        }
        and report.get("implementation")
        == {
            "path": str(DECODER.SCRIPT_PATH.relative_to(ROOT)),
            "sha256": sha256_file(DECODER.SCRIPT_PATH),
            "size_bytes": DECODER.SCRIPT_PATH.stat().st_size,
        },
        "decoder report producer binding changed",
    )
    status_interpretation = report.get("status_interpretation")
    execution_identity = report.get("execution_identity")
    precommit = report.get("precommit_reauthentication")
    report_claims = report.get("claim_scope")
    _require(
        report.get("analysis_chronology") == DECODER.ANALYSIS_CHRONOLOGY
        and isinstance(status_interpretation, Mapping)
        and status_interpretation.get(
            "scientific_validity_reliability_confirmation_or_transport_gate_passed"
        )
        is False
        and isinstance(execution_identity, Mapping)
        and execution_identity.get(
            "config_and_implementation_exact_bytes_captured_at_start_of_run"
        )
        is True
        and execution_identity.get(
            "same_startup_bytes_required_immediately_before_atomic_publish"
        )
        is True
        and isinstance(precommit, Mapping)
        and precommit.get(
            "all_authenticated_files_rehashed_immediately_before_atomic_publish"
        )
        is True
        and isinstance(report_claims, Mapping)
        and report_claims.get("cot_or_cot_like_decoding_established") is False
        and report_claims.get("semantic_sentence_or_chain_decoding_established")
        is False,
        "decoder chronology, status, claim, or precommit authentication changed",
    )
    inputs = report.get("inputs")
    _require(
        isinstance(inputs, Mapping)
        and inputs.get("alignment_index", {}).get("sha256")
        == config["inputs"]["alignment_index"]["sha256"]
        and inputs.get("label_sidecar", {}).get("sha256")
        == config["inputs"]["observable_label_sidecar"]["sha256"]
        and inputs.get("all_hashes_matched_before_model_fit") is True
        and inputs.get("feature_source_hashes_reverified_before_model_fit") is True
        and inputs.get("feature_and_label_artifacts_physically_separate") is True
        and inputs.get("reserved_validation_access_authorized") is False,
        "decoder report input authentication changed",
    )
    evaluation = report.get("evaluation")
    _require(
        isinstance(evaluation, Mapping)
        and evaluation.get("target") == TARGET
        and evaluation.get("available_target_row_count") == 1549
        and evaluation.get("unknown_target_row_count_excluded_never_coerced_to_negative")
        == 57,
        "decoder target or availability changed",
    )
    outer = evaluation.get("outer_evaluation")
    _require(isinstance(outer, Mapping), "outer evaluation is missing")
    _require(
        outer.get("row_count") == 1549
        and outer.get("classes") == ["no", "yes"]
        and len(outer.get("repositories_in_order", [])) == 10
        and outer.get("same_outer_folds_for_every_variant") is True
        and outer.get("same_oof_rows_labels_and_weights_for_every_variant") is True
        and outer.get("row_identity_sha256")
        == sha256_bytes(canonical_json_bytes(list(rows)))
        and outer.get("label_sha256")
        == sha256_bytes(canonical_json_bytes(list(labels))),
        "decoder OOF row, label, or fold identity changed",
    )
    y = np.asarray(outer.get("label_indices"), dtype=np.int64)
    expected_y = np.asarray([0 if label == "no" else 1 for label in labels], dtype=np.int64)
    weights = np.asarray(outer.get("evaluation_weights"), dtype=np.float64)
    expected_weights = hierarchical_row_weights(rows)
    _require(
        np.array_equal(y, expected_y)
        and np.array_equal(weights, expected_weights)
        and sha256_array(weights.astype("<f8"))
        == outer.get("evaluation_weight_sha256"),
        "decoder OOF labels or weights changed",
    )
    required_names = {REFERENCE, *(item["candidate"] for item in COMPARISONS)}
    variants = outer.get("variants")
    _require(
        isinstance(variants, Mapping) and required_names <= set(variants),
        "required nested variants are absent",
    )
    probabilities: dict[str, np.ndarray] = {}
    for name in required_names:
        variant = variants[name]
        matrix = np.asarray(variant.get("oof_probabilities"), dtype=np.float64)
        _require(
            matrix.shape == (1549, 2)
            and bool(np.all(np.isfinite(matrix)))
            and bool(np.all(matrix > 0.0))
            and bool(np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12))
            and sha256_array(matrix.astype("<f8"))
            == variant.get("oof_probability_sha256"),
            f"saved OOF probability matrix changed: {name}",
        )
        probabilities[name] = matrix
    paired = evaluation.get("paired_comparisons")
    _require(isinstance(paired, Mapping), "decoder paired comparisons are absent")
    for comparison in COMPARISONS:
        key = f"{comparison['candidate']}__vs__{REFERENCE}"
        stored = paired.get(key)
        _require(
            isinstance(stored, Mapping)
            and stored.get("candidate") == comparison["candidate"]
            and stored.get("reference") == REFERENCE
            and stored.get("strictly_nested_component_comparison") is True,
            f"decoder nested comparison changed: {key}",
        )
    return y, weights, probabilities, evaluation


def loss_vectors(probabilities: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    _require(
        matrix.shape == (len(labels), 2)
        and bool(np.all(matrix > 0.0))
        and bool(np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12)),
        "loss inputs are invalid",
    )
    positions = np.arange(len(labels))
    one_hot = np.eye(2, dtype=np.float64)[labels]
    return {
        "negative_log_likelihood": -np.log(matrix[positions, labels]),
        "multiclass_brier": np.sum((matrix - one_hot) ** 2, axis=1),
    }


def student_t_interval(
    values: Sequence[float], *, confidence_level: float
) -> dict[str, Any]:
    observed = np.asarray(values, dtype=np.float64)
    _require(
        observed.ndim == 1
        and len(observed) >= 2
        and bool(np.all(np.isfinite(observed)))
        and 0.0 < confidence_level < 1.0,
        "Student-t interval inputs are invalid",
    )
    try:
        from scipy.stats import t as student_t
    except ImportError as error:
        raise InferenceError("scipy is required for Student-t quantiles") from error
    mean = float(observed.mean())
    sd = float(observed.std(ddof=1))
    standard_error = sd / math.sqrt(len(observed))
    critical = float(student_t.ppf(0.5 + confidence_level / 2.0, len(observed) - 1))
    return {
        "confidence_level": confidence_level,
        "degrees_of_freedom": len(observed) - 1,
        "critical_value": critical,
        "repository_delta_standard_deviation": sd,
        "standard_error": standard_error,
        "interval": [
            mean - critical * standard_error,
            mean + critical * standard_error,
        ],
        "assumption": "repository_deltas_are_iid",
    }


def exact_sign_flip(values: Sequence[float]) -> dict[str, Any]:
    observed = np.asarray(values, dtype=np.float64)
    _require(
        observed.ndim == 1
        and len(observed) >= 1
        and bool(np.all(np.isfinite(observed))),
        "sign-flip inputs are invalid",
    )
    point = float(observed.mean())
    assignment_count = 2 ** len(observed)
    extreme = 0
    for signs in product((-1.0, 1.0), repeat=len(observed)):
        statistic = abs(float(np.mean(np.asarray(signs) * observed)))
        extreme += int(statistic >= abs(point))
    return {
        "alternative": "two_sided",
        "statistic": "absolute_unweighted_mean_repository_delta",
        "assignment_count": assignment_count,
        "extreme_assignment_count": extreme,
        "p_value": extreme / assignment_count,
        "observed_assignment_included": True,
        "plus_one_correction": False,
        "assumption": "repository_delta_sign_symmetry",
    }


def holm_adjust(
    p_values: Mapping[str, float], *, family_order: Sequence[str]
) -> dict[str, dict[str, Any]]:
    _require(
        set(p_values) == set(family_order)
        and len(family_order) == len(set(family_order)) >= 1,
        "Holm family and p-values differ",
    )
    for value in p_values.values():
        _require(0.0 <= value <= 1.0 and math.isfinite(value), "Holm p-value is invalid")
    order_index = {name: index for index, name in enumerate(family_order)}
    ranked = sorted(p_values, key=lambda name: (p_values[name], order_index[name]))
    result: dict[str, dict[str, Any]] = {}
    running = 0.0
    family_size = len(ranked)
    for zero_rank, name in enumerate(ranked):
        raw_adjusted = min(1.0, (family_size - zero_rank) * p_values[name])
        running = max(running, raw_adjusted)
        result[name] = {
            "family_rank": zero_rank + 1,
            "raw_p_value": p_values[name],
            "adjusted_p_value": running,
        }
    return result


def recompute_inference(
    *,
    rows: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    decoder_evaluation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    repository_array = np.asarray([str(row["repository"]) for row in rows], dtype=str)
    repositories = sorted(set(repository_array.tolist()))
    _require(len(repositories) == 10, "repository count changed")
    comparison_results: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for comparison in config["comparisons"]:
        candidate_name = comparison["candidate"]
        reference_name = comparison["reference"]
        key = f"{candidate_name}__vs__{reference_name}"
        candidate_loss = loss_vectors(probabilities[candidate_name], y)
        reference_loss = loss_vectors(probabilities[reference_name], y)
        per_repository: dict[str, Any] = {}
        delta_by_metric: dict[str, list[float]] = {metric: [] for metric in METRICS}
        for repository in repositories:
            indices = np.flatnonzero(repository_array == repository)
            repository_rows = [rows[int(index)] for index in indices]
            weights = hierarchical_row_weights(repository_rows)
            support = Counter(y[indices].tolist())
            metrics: dict[str, Any] = {}
            for metric in METRICS:
                reference_value = float(weights @ reference_loss[metric][indices])
                candidate_value = float(weights @ candidate_loss[metric][indices])
                delta = candidate_value - reference_value
                delta_by_metric[metric].append(delta)
                metrics[metric] = {
                    "reference": reference_value,
                    "candidate": candidate_value,
                    "candidate_minus_reference": delta,
                }
                stored = decoder_evaluation["paired_comparisons"][key][
                    "per_repository"
                ][repository]["comparison"]["paired_weighted_mean_row_deltas"][metric]
                _require(
                    abs(delta - stored) <= 1e-14,
                    f"independent repository delta differs from decoder report: {key}/{repository}/{metric}",
                )
            per_repository[repository] = {
                "row_count": len(indices),
                "task_count": len({str(row["task_id_sha256"]) for row in repository_rows}),
                "class_support": {"no": int(support[0]), "yes": int(support[1])},
                "weighting": "equal_task_then_equal_available_row_within_task",
                "metrics": metrics,
            }
        metric_results: dict[str, Any] = {}
        for metric in METRICS:
            values = np.asarray(delta_by_metric[metric], dtype=np.float64)
            point = float(values.mean())
            stored_point = decoder_evaluation["paired_comparisons"][key][
                "paired_weighted_mean_row_deltas"
            ][metric]
            _require(
                abs(point - stored_point) <= 1e-14,
                f"independent equal-repository point delta differs from decoder report: {key}/{metric}",
            )
            sign_flip = exact_sign_flip(values)
            family_name = f"{key}::{metric}"
            p_values[family_name] = float(sign_flip["p_value"])
            metric_results[metric] = {
                "point_delta_candidate_minus_reference": point,
                "repository_deltas_in_order": values.tolist(),
                "favorable_negative_repository_count": int(np.sum(values < 0.0)),
                "unfavorable_positive_repository_count": int(np.sum(values > 0.0)),
                "zero_repository_count": int(np.sum(values == 0.0)),
                "student_t": student_t_interval(
                    values,
                    confidence_level=config["repository_inference"][
                        "student_t_interval"
                    ]["confidence_level"],
                ),
                "exact_sign_flip": sign_flip,
            }
        comparison_results[key] = {
            "candidate": candidate_name,
            "reference": reference_name,
            "strictly_nested_component_comparison": True,
            "repositories_in_order": repositories,
            "metrics": metric_results,
            "per_repository": per_repository,
            "independently_recomputed_from_saved_oof_probabilities": True,
            "decoder_stored_points_matched": True,
        }
    holm = holm_adjust(p_values, family_order=config["multiple_testing"]["family_order"])
    for key, comparison in comparison_results.items():
        for metric, result in comparison["metrics"].items():
            result["holm"] = holm[f"{key}::{metric}"]
    return {
        "repository_count": len(repositories),
        "repositories_in_order": repositories,
        "comparison_count": len(comparison_results),
        "metric_count_per_comparison": len(METRICS),
        "multiple_testing": {
            **config["multiple_testing"],
            "results": holm,
        },
        "comparisons": comparison_results,
    }


def _write_no_clobber(path: Path, value: Any) -> None:
    _require(not path.exists() and not path.is_symlink(), "nested-inference output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _require(not temporary.exists() and not temporary.is_symlink(), "temporary output exists")
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise InferenceError("nested-inference output exists") from error
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    # Caller-controlled path text is rejected before any filesystem operation.
    DECODER.frozen_lexical_path_preflight([args.config, args.output])
    DECODER.frozen_canonical_path_preflight(
        input_paths=[args.config], output_paths=[args.output]
    )
    config_path = Path(args.config).resolve(strict=True)
    _require(config_path == CONFIG_PATH, "nested-inference config path changed")
    config = validate_config(load_json(config_path))
    input_paths = {
        name: ROOT / record["path"] for name, record in config["inputs"].items()
    }
    DECODER.frozen_lexical_path_preflight(input_paths.values())
    DECODER.frozen_canonical_path_preflight(
        input_paths=[config_path, *input_paths.values()], output_paths=[args.output]
    )
    resolved = {
        name: _resolve_authenticated_input(config["inputs"][name], name)
        for name in config["inputs"]
    }
    decoder_config = DECODER.validate_config(load_json(DECODER.CONFIG_PATH))
    authenticated_paths = [
        config_path,
        SCRIPT_PATH,
        DECODER.CONFIG_PATH,
        DECODER.SCRIPT_PATH,
        *resolved.values(),
    ]
    pre_hashes = {path: sha256_file(path) for path in authenticated_paths}
    alignment_value = load_json(resolved["alignment_index"])
    alignment_rows = DECODER.validate_alignment_index(alignment_value)
    sidecar_value = load_json(resolved["observable_label_sidecar"])
    label_rows = DECODER.validate_label_sidecar(
        sidecar_value,
        alignment_rows=alignment_rows,
        config=decoder_config,
    )
    rows, labels = available_rows_and_labels(
        alignment_rows,
        label_rows,
        target=config["target"],
    )
    decoder_report = load_json(resolved["decoder_report"])
    y, _weights, probabilities, decoder_evaluation = validate_decoder_report(
        decoder_report,
        rows=rows,
        labels=labels,
        config=config,
    )
    inference = recompute_inference(
        rows=rows,
        y=y,
        probabilities=probabilities,
        decoder_evaluation=decoder_evaluation,
        config=config,
    )
    for path, digest in pre_hashes.items():
        _require(
            sha256_file(path) == digest,
            f"authenticated input changed during nested inference: {path}",
        )
    try:
        import scipy
    except ImportError as error:
        raise InferenceError("scipy runtime identity is unavailable") from error
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "passed",
        "scope": config["interpretation"]["scope"],
        "config": {
            "path": str(CONFIG_PATH.relative_to(ROOT)),
            "sha256": sha256_file(CONFIG_PATH),
        },
        "implementation": {
            "path": str(SCRIPT_PATH.relative_to(ROOT)),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "runtime": {"numpy": np.__version__, "scipy": scipy.__version__},
        "inputs": {
            name: {
                "path": str(path),
                "sha256": config["inputs"][name]["sha256"],
                "size_bytes": path.stat().st_size,
            }
            for name, path in resolved.items()
        },
        "authentication": {
            "all_input_hashes_matched": True,
            "decoder_report_row_identity_recomputed": True,
            "decoder_report_label_identity_recomputed": True,
            "decoder_report_hierarchical_weights_recomputed": True,
            "saved_oof_probability_hashes_recomputed": True,
            "stored_decoder_points_and_repository_deltas_matched": True,
            "reserved_validation_access_authorized": False,
        },
        "target": config["target"],
        "row_count": len(rows),
        "available_target_support": dict(Counter(labels)),
        "inference": inference,
        "interpretation": config["interpretation"],
        "claim_scope": config["claim_scope"],
        "mandatory_limitations": config["mandatory_limitations"],
    }
    _write_no_clobber(Path(args.output).resolve(strict=False), report)
    print(
        f"wrote {inference['comparison_count']} frozen nested comparisons "
        f"across {inference['repository_count']} repositories to {args.output}"
    )
    return 0


__all__ = [
    "COMPARISONS",
    "CONFIG_PATH",
    "EXPECTED_INPUTS",
    "FAMILY_ORDER",
    "InferenceError",
    "KIND",
    "METRICS",
    "REFERENCE",
    "TARGET",
    "available_rows_and_labels",
    "exact_sign_flip",
    "hierarchical_row_weights",
    "holm_adjust",
    "loss_vectors",
    "recompute_inference",
    "run",
    "sha256_array",
    "sha256_file",
    "student_t_interval",
    "validate_config",
    "validate_decoder_report",
]


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
