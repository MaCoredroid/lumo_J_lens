#!/usr/bin/env python3
"""P2 faithfulness metric: surface-vs-internal divergence (Qwen-only).

`source_disagreement` = normalized base-2 Jensen-Shannon divergence between the
per-boundary sequence-logit (surface / ordinary-logit) and sequence-J (internal
public-J) action distributions. Both are read from the materialized out-of-fold
probabilities in `observable-action-phase-v1.json` (1570 N60-cohort rows, same
rows/labels/folds for every variant), so no decoder run is needed.

The evaluation asks whether HIGH divergence flags lens error, operationalized as
the pooled `sequence_logit_j` forecast mispredicting the agent's own next action
(a free, self-supervised label). A label-permutation null guards against
over-claiming. Read-only over materialized artifacts; no model, no capture, and
the divergence never re-enters the predictor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTION_PHASE_ARTIFACT = (
    ROOT
    / ".cache/swe_task_state_v4_raw_capture/n60-final/observable-action-phase-v1.json"
)
CLASSES = ("inspect", "edit", "check")
_SURFACE = "sequence_logit"
_INTERNAL = "sequence_j"
_POOLED = "sequence_logit_j"


@dataclass(frozen=True)
class SourceProbabilities:
    classes: tuple[str, ...]
    surface: np.ndarray  # (N, 3) sequence-logit
    internal: np.ndarray  # (N, 3) sequence-J
    pooled: np.ndarray  # (N, 3) sequence_logit_j forecast
    labels: np.ndarray  # (N,) int class index
    weights: np.ndarray  # (N,) float


def _distribution_matrix(rows: Any, label: str) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(CLASSES):
        raise ValueError(f"{label} must be (N, {len(CLASSES)})")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError(f"{label} has invalid probabilities")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{label} rows must sum to one")
    return matrix


def load_action_phase_sources(
    path: Path = ACTION_PHASE_ARTIFACT,
) -> SourceProbabilities:
    outer = json.loads(Path(path).read_text())["evaluation"]["outer_evaluation"]
    if tuple(outer["classes"]) != CLASSES:
        raise ValueError(f"unexpected class order {outer['classes']}")
    if not outer.get("same_oof_rows_labels_and_weights_for_every_variant"):
        raise ValueError("variants do not share identical OOF rows/labels")
    variants = outer["variants"]
    surface = _distribution_matrix(variants[_SURFACE]["oof_probabilities"], _SURFACE)
    internal = _distribution_matrix(variants[_INTERNAL]["oof_probabilities"], _INTERNAL)
    pooled = _distribution_matrix(variants[_POOLED]["oof_probabilities"], _POOLED)
    labels = np.asarray(outer["label_indices"], dtype=np.int64)
    weights = np.asarray(outer["evaluation_weights"], dtype=np.float64)
    n = surface.shape[0]
    if not (
        internal.shape[0] == pooled.shape[0] == labels.shape[0] == weights.shape[0] == n
    ):
        raise ValueError("row counts differ across sources/labels/weights")
    return SourceProbabilities(CLASSES, surface, internal, pooled, labels, weights)


def _entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0.0, -p * np.log2(p), 0.0)
    return terms.sum(axis=-1)


def normalized_jsd(surface: np.ndarray, internal: np.ndarray) -> np.ndarray:
    """Row-wise base-2 JSD in [0, 1] (0 = identical, 1 = disjoint support)."""
    mixture = 0.5 * (surface + internal)
    jsd = _entropy_bits(mixture) - 0.5 * (
        _entropy_bits(surface) + _entropy_bits(internal)
    )
    return np.clip(jsd, 0.0, 1.0)


def per_row_divergence(sources: SourceProbabilities) -> np.ndarray:
    return normalized_jsd(sources.surface, sources.internal)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        return float("nan")
    return float(np.dot(values, weights) / total)


def _rank_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """P(score[positive] > score[negative]); 0.5 = no separation."""
    pos = scores[positive]
    neg = scores[~positive]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # average ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    rank_sum_pos = ranks[positive].sum()
    n_pos, n_neg = pos.size, neg.size
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def evaluate_divergence_flags_error(
    sources: SourceProbabilities,
    *,
    permutations: int = 2000,
    seed: int = 20260720,
) -> dict[str, Any]:
    divergence = per_row_divergence(sources)
    predicted = sources.pooled.argmax(axis=1)
    error = predicted != sources.labels
    w = sources.weights

    mean_error = _weighted_mean(divergence[error], w[error])
    mean_correct = _weighted_mean(divergence[~error], w[~error])
    effect = mean_error - mean_correct
    auc = _rank_auc(divergence, error)

    rng = np.random.default_rng(seed)
    null_ge = 0
    for _ in range(permutations):
        shuffled = rng.permutation(error)
        m_e = _weighted_mean(divergence[shuffled], w[shuffled])
        m_c = _weighted_mean(divergence[~shuffled], w[~shuffled])
        if (m_e - m_c) >= effect:
            null_ge += 1
    p_value = (null_ge + 1) / (permutations + 1)

    return {
        "n_rows": int(divergence.size),
        "n_error": int(error.sum()),
        "n_correct": int((~error).sum()),
        "divergence_overall_weighted_mean": _weighted_mean(divergence, w),
        "divergence_mean_on_error": mean_error,
        "divergence_mean_on_correct": mean_correct,
        "error_minus_correct_effect": effect,
        "error_detection_auc": auc,
        "permutation_p_value": p_value,
        "permutations": permutations,
        "error_definition": "pooled sequence_logit_j argmax != agent next-action label",
    }


def main(argv: Any = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ACTION_PHASE_ARTIFACT)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args(argv)
    sources = load_action_phase_sources(args.artifact)
    result = evaluate_divergence_flags_error(sources, permutations=args.permutations)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
