#!/usr/bin/env python3
"""Score the 14 general concept families from a captured VJP-lens readout (P7c).

Reuses the concept-chain's scoring math (mean form/layer log-probability per family)
but with the TASK-INDEPENDENT general concept-form vocab, so it works on any cohort
task's readout. For each captured boundary it computes a family score = mean of the
family's form-token log-probabilities over all captured layers, ranks the families,
and (retrospectively, across the run's boundaries) baseline-centers to a top-1 —
matching the faithfulness module's background-centering. Read-only; no model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERAL_VOCAB = ROOT / "configs/swe_task_state_v4_general_concept_forms.json"
_LENS = "jacobian_lens"


def family_token_ids(vocab_path: Path = GENERAL_VOCAB) -> dict[str, list[int]]:
    doc = json.loads(Path(vocab_path).read_text())
    return {
        family: [form["token_id"] for form in forms]
        for family, forms in doc["families"].items()
    }


def _layer_logprobs(experiment: dict[str, Any], lens: str = _LENS) -> list[dict[int, float]]:
    """Per layer: token_id -> log-probability, from scored_tokens."""
    out = []
    for layer in experiment.get("layers", []):
        pos = layer["positions"][0]
        block = pos.get(lens, {})
        out.append({e["token_id"]: e["logprob"] for e in block.get("scored_tokens", [])})
    return out


def score_experiment(
    experiment: dict[str, Any], family_ids: dict[str, list[int]], *, lens: str = _LENS
) -> dict[str, float]:
    """family -> mean form/layer log-probability (concept-chain convention)."""
    per_layer = _layer_logprobs(experiment, lens)
    scores: dict[str, float] = {}
    for family, ids in family_ids.items():
        values = [
            layer[tid] for layer in per_layer for tid in ids if tid in layer
        ]
        if values:
            scores[family] = math.fsum(values) / len(values)
    return scores


def _top1(scores: dict[str, float]) -> str | None:
    return max(scores, key=scores.get) if scores else None


def score_report(
    report_path: Path, *, vocab_path: Path = GENERAL_VOCAB, lens: str = _LENS
) -> list[dict[str, Any]]:
    report = json.loads(Path(report_path).read_text())
    family_ids = family_token_ids(vocab_path)
    rows = []
    for i, exp in enumerate(report.get("experiments", [])):
        scores = score_experiment(exp, family_ids, lens=lens)
        rows.append(
            {"index": i, "id": exp.get("id"), "family_scores": scores, "top1": _top1(scores)}
        )
    _attach_centered_top1(rows)
    return rows


def _attach_centered_top1(rows: list[dict[str, Any]]) -> None:
    """Subtract each family's cross-boundary mean, then re-rank (retrospective)."""
    families = sorted({f for r in rows for f in r["family_scores"]})
    n = len(rows)
    baselines = {
        f: (
            math.fsum(r["family_scores"][f] for r in rows if f in r["family_scores"])
            / max(1, sum(1 for r in rows if f in r["family_scores"]))
        )
        for f in families
    }
    for i, r in enumerate(rows):
        centered = {
            f: r["family_scores"][f]
            - (
                # leave-one-out baseline to avoid trivial self-inclusion
                (
                    math.fsum(
                        rows[j]["family_scores"][f]
                        for j in range(n)
                        if j != i and f in rows[j]["family_scores"]
                    )
                    / max(1, sum(1 for j in range(n) if j != i and f in rows[j]["family_scores"]))
                )
                if n > 1
                else baselines[f]
            )
            for f in r["family_scores"]
        }
        r["top1_centered"] = _top1(centered)


def main(argv: Any = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--lens", default=_LENS)
    args = parser.parse_args(argv)
    rows = score_report(args.report, lens=args.lens)
    for r in rows[:20]:
        print(f"  {str(r['id'])[:36]:36s} raw={r['top1']:22s} centered={r.get('top1_centered')}")
    print(f"{len(rows)} boundaries scored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
