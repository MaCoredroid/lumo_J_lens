#!/usr/bin/env python3
"""Bootstrap paired fixed-OOF observable-decoder deltas by repo/task/row."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
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
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_observable_bootstrap.json"
SCRIPT_PATH = Path(__file__).resolve()
KIND = "swe_task_state_v4_fixed_oof_hierarchical_bootstrap_report"
METRICS = ("negative_log_likelihood", "multiclass_brier", "correctness")


class BootstrapError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BootstrapError(message)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=DECODER._reject_duplicate_keys,
    )


def validate_config(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "bootstrap config must be an object")
    config = dict(value)
    _require(
        set(config)
        == {
            "schema_version",
            "id",
            "status",
            "inputs",
            "target",
            "reference",
            "candidates",
            "additional_comparisons",
            "bootstrap",
            "interpretation",
            "claim_scope",
        },
        "bootstrap config schema changed",
    )
    _require(
        config["schema_version"] == 1
        and config["id"] == "swe-task-state-v4-fixed-oof-hierarchical-bootstrap"
        and config["status"] == "development_only_reserved_validation_closed"
        and config["target"] == DECODER.PRIMARY_TARGET
        and config["reference"] == "sequence_logit",
        "bootstrap identity or primary comparison changed",
    )
    expected_inputs = {
        "decoder_report": (
            ".cache/swe_task_state_v4_raw_capture/n60-final/observable-rationale-marker-v1.json",
            "f731281e071bee8323156d8a6f966c6b92e40623db29455d31f795edb3efbb6b",
        ),
        "alignment_index": (
            ".cache/swe_task_state_v4_raw_capture/n60-final/alignment-index-v4.json",
            "c0d9e4bd6ad8f6962cf58776441ec4ef042eebbbae8a9959b4149e605c9ae38e",
        ),
        "observable_label_sidecar": (
            ".cache/swe_task_state_v4_raw_capture/n60-final/observable-events-v4.json",
            "3e54d3510fea892059369b2f7cbac93381504fcc22a2732bb8fca5252a42cf18",
        ),
    }
    _require(set(config["inputs"]) == set(expected_inputs), "bootstrap inputs changed")
    for name, (path, digest) in expected_inputs.items():
        _require(
            config["inputs"][name] == {"path": path, "sha256": digest},
            f"bootstrap input binding changed: {name}",
        )
    expected_candidates = [
        "raw_activation_current",
        "public_j_activation_current",
        "raw_activation_sequence",
        "public_j_activation_sequence",
        "raw_public_j_activation_sequence",
        "sequence_logit_plus_raw_public_j_activation_sequence",
    ]
    _require(config["candidates"] == expected_candidates, "bootstrap candidates changed")
    _require(
        config["additional_comparisons"]
        == [
            {
                "candidate": "public_j_activation_current",
                "reference": "raw_activation_current",
            }
        ],
        "additional bootstrap comparison changed",
    )
    _require(
        config["bootstrap"]
        == {
            "draw_count": 5000,
            "rng": "numpy.random.Generator(PCG64)",
            "seed": 20260719,
            "hierarchy": [
                "repository_with_replacement",
                "task_within_repository_with_replacement",
                "available_row_within_task_with_replacement",
            ],
            "estimand": "equal_repository_then_equal_task_within_repository_then_equal_available_row_within_task",
            "interval": "equal_tailed_percentile",
            "lower_quantile": 0.025,
            "upper_quantile": 0.975,
            "fixed_oof_predictions_no_model_refit": True,
            "paired_resampling_for_candidate_and_reference": True,
        },
        "bootstrap sampling contract changed",
    )
    _require(
        config["interpretation"]
        == {
            "configured_after_point_results_were_observed": True,
            "multiplicity_adjusted": False,
            "refit_uncertainty_included": False,
            "transport_or_confirmation_claim_allowed": False,
        }
        and set(config["claim_scope"])
        == {
            "private_chain_of_thought_reconstructed",
            "hidden_thought_understanding_or_intent_established",
            "emotion_decoding_established",
            "causal_interpretation_established",
            "incremental_value_statistically_established",
            "operational_reliability_established",
        }
        and all(item is False for item in config["claim_scope"].values()),
        "bootstrap interpretation or claim boundary changed",
    )
    return config


def available_rows_and_labels(
    alignment_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    for alignment, label_row in zip(alignment_rows, label_rows, strict=True):
        if not alignment["stable_feature_eligible"]:
            continue
        record = label_row["targets"][target]
        if record["status"] == "unknown":
            continue
        rows.append(
            {
                "global_index": alignment["global_index"],
                "source_id_sha256": alignment["source_id_sha256"],
                "task_id_sha256": alignment["task_id_sha256"],
                "repository": alignment["repository"],
                "request_index": alignment["request_index"],
            }
        )
        labels.append(str(record["value"]))
    _require(len(rows) == len(labels) == 1549, "available primary row count changed")
    return rows, labels


def validate_report(
    report: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    _require(isinstance(report, dict), "decoder report must be an object")
    _require(
        report.get("schema_version") == 1
        and report.get("kind") == DECODER.REPORT_KIND
        and report.get("status") == "passed",
        "decoder report identity changed",
    )
    evaluation = report.get("evaluation")
    _require(
        isinstance(evaluation, dict)
        and evaluation.get("target") == DECODER.PRIMARY_TARGET
        and evaluation.get("available_target_row_count") == 1549,
        "decoder report target changed",
    )
    outer = evaluation["outer_evaluation"]
    _require(
        outer["row_count"] == 1549
        and outer["classes"] == ["no", "yes"]
        and outer["row_identity_sha256"]
        == DECODER.sha256_bytes(DECODER.canonical_json_bytes(list(rows)))
        and outer["label_sha256"]
        == DECODER.sha256_bytes(DECODER.canonical_json_bytes(list(labels))),
        "decoder OOF row or label identity changed",
    )
    y = np.asarray(outer["label_indices"], dtype=np.int64)
    weights = np.asarray(outer["evaluation_weights"], dtype=np.float64)
    expected_y = np.asarray([0 if label == "no" else 1 for label in labels], dtype=np.int64)
    expected_weights = DECODER.hierarchical_row_weights(rows)
    _require(
        np.array_equal(y, expected_y)
        and np.array_equal(weights, expected_weights)
        and DECODER.sha256_array(weights.astype("<f8"))
        == outer["evaluation_weight_sha256"],
        "decoder labels or evaluation weights changed",
    )
    probabilities: dict[str, np.ndarray] = {}
    for name, result in outer["variants"].items():
        matrix = np.asarray(result["oof_probabilities"], dtype=np.float64)
        _require(
            matrix.shape == (1549, 2)
            and np.all(matrix > 0.0)
            and np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12)
            and DECODER.sha256_array(matrix.astype("<f8"))
            == result["oof_probability_sha256"],
            f"decoder OOF probability matrix changed: {name}",
        )
        probabilities[name] = matrix
    return y, weights, probabilities


def paired_delta_matrix(
    probabilities: Mapping[str, np.ndarray],
    y: np.ndarray,
    comparisons: Sequence[Mapping[str, str]],
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    row = np.arange(len(y))
    one_hot = np.eye(2, dtype=np.float64)[y]
    for comparison in comparisons:
        candidate_name = comparison["candidate"]
        reference_name = comparison["reference"]
        candidate = probabilities[candidate_name]
        reference = probabilities[reference_name]
        values = (
            -np.log(candidate[row, y]) + np.log(reference[row, y]),
            np.sum((candidate - one_hot) ** 2, axis=1)
            - np.sum((reference - one_hot) ** 2, axis=1),
            (np.argmax(candidate, axis=1) == y).astype(np.float64)
            - (np.argmax(reference, axis=1) == y).astype(np.float64),
        )
        for metric, delta in zip(METRICS, values, strict=True):
            columns.append(np.asarray(delta, dtype=np.float64))
            names.append(f"{candidate_name}__vs__{reference_name}::{metric}")
    matrix = np.column_stack(columns)
    _require(np.all(np.isfinite(matrix)), "paired delta matrix is non-finite")
    return matrix, names


def hierarchical_bootstrap(
    delta: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    draw_count: int,
    seed: int,
) -> np.ndarray:
    """Paired repo/task/row bootstrap of an equal-mass hierarchical estimand."""

    values = np.asarray(delta, dtype=np.float64)
    _require(
        values.ndim == 2
        and len(values) == len(rows)
        and draw_count >= 1
        and np.all(np.isfinite(values)),
        "bootstrap inputs are invalid",
    )
    grouped: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    temporary: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for position, row in enumerate(rows):
        temporary[str(row["repository"])][str(row["task_id_sha256"])].append(position)
    for repository, tasks in temporary.items():
        grouped[repository] = {
            task: np.asarray(indices, dtype=np.int64) for task, indices in tasks.items()
        }
    repositories = sorted(grouped)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty((draw_count, values.shape[1]), dtype=np.float64)
    for draw_index in range(draw_count):
        repo_means = []
        for repo_position in rng.integers(0, len(repositories), size=len(repositories)):
            tasks = grouped[repositories[int(repo_position)]]
            task_names = sorted(tasks)
            task_means = []
            for task_position in rng.integers(0, len(task_names), size=len(task_names)):
                indices = tasks[task_names[int(task_position)]]
                sampled = indices[rng.integers(0, len(indices), size=len(indices))]
                task_means.append(values[sampled].mean(axis=0))
            repo_means.append(np.asarray(task_means).mean(axis=0))
        draws[draw_index] = np.asarray(repo_means).mean(axis=0)
    _require(np.all(np.isfinite(draws)), "bootstrap draws are non-finite")
    return draws


def summarize(
    *,
    names: Sequence[str],
    point: np.ndarray,
    draws: np.ndarray,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, name in enumerate(names):
        metric = name.split("::", 1)[1]
        favorable = draws[:, index] < 0.0 if metric != "correctness" else draws[:, index] > 0.0
        low, high = np.quantile(draws[:, index], [lower, upper])
        result[name] = {
            "point_delta_candidate_minus_reference": float(point[index]),
            "bootstrap_mean": float(draws[:, index].mean()),
            "interval": [float(low), float(high)],
            "interval_contains_zero": bool(low <= 0.0 <= high),
            "favorable_direction": "negative" if metric != "correctness" else "positive",
            "fraction_of_draws_favorable": float(np.mean(favorable)),
        }
    return result


def _write_no_clobber(path: Path, value: Any) -> None:
    _require(not path.exists() and not path.is_symlink(), "bootstrap output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    DECODER.frozen_lexical_path_preflight([args.config, args.output])
    config_path = args.config.resolve(strict=True)
    config = validate_config(load_json(config_path))
    input_paths = [ROOT / config["inputs"][name]["path"] for name in config["inputs"]]
    DECODER.frozen_lexical_path_preflight(input_paths)
    DECODER.frozen_canonical_path_preflight(
        input_paths=[config_path, *input_paths], output_paths=[args.output]
    )
    values = {}
    for name, path in zip(config["inputs"], input_paths, strict=True):
        expected = config["inputs"][name]["sha256"]
        _require(sha256_file(path) == expected, f"bootstrap input hash changed: {name}")
        values[name] = load_json(path)
    alignment_rows = DECODER.validate_alignment_index(values["alignment_index"])
    decoder_config = DECODER.validate_config(DECODER.load_json(DECODER.CONFIG_PATH))
    label_rows = DECODER.validate_label_sidecar(
        values["observable_label_sidecar"],
        alignment_rows=alignment_rows,
        config=decoder_config,
    )
    rows, labels = available_rows_and_labels(
        alignment_rows,
        label_rows,
        target=config["target"],
    )
    y, weights, probabilities = validate_report(
        values["decoder_report"], rows=rows, labels=labels
    )
    comparisons = [
        {"candidate": candidate, "reference": config["reference"]}
        for candidate in config["candidates"]
    ] + list(config["additional_comparisons"])
    delta, names = paired_delta_matrix(probabilities, y, comparisons)
    point = weights @ delta
    bootstrap = config["bootstrap"]
    draws = hierarchical_bootstrap(
        delta,
        rows,
        draw_count=bootstrap["draw_count"],
        seed=bootstrap["seed"],
    )
    report = {
        "schema_version": 1,
        "kind": KIND,
        "status": "passed",
        "scope": "retrospective_development_fixed_oof_uncertainty_only",
        "config": {
            "path": str(CONFIG_PATH.relative_to(ROOT)),
            "sha256": sha256_file(CONFIG_PATH),
        },
        "implementation": {
            "path": str(SCRIPT_PATH.relative_to(ROOT)),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "inputs": config["inputs"],
        "target": config["target"],
        "row_count": len(rows),
        "comparison_count": len(comparisons),
        "metric_count_per_comparison": len(METRICS),
        "bootstrap": bootstrap,
        "results": summarize(
            names=names,
            point=point,
            draws=draws,
            lower=bootstrap["lower_quantile"],
            upper=bootstrap["upper_quantile"],
        ),
        "interpretation": config["interpretation"],
        "claim_scope": config["claim_scope"],
        "mandatory_limitation": (
            "Intervals use fixed out-of-fold predictions and a bootstrap configured "
            "after point results were observed; they omit model-refit uncertainty, "
            "are not multiplicity-adjusted, and establish neither private COT, "
            "emotion, transport, incremental value, nor operational reliability."
        ),
        "reserved_validation_access_authorized": False,
    }
    _write_no_clobber(args.output.resolve(strict=False), report)
    print(f"wrote {bootstrap['draw_count']} fixed-OOF bootstrap draws to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
