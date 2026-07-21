#!/usr/bin/env python3
"""Lever 1: per-layer faithfulness sweep on an existing cohort readout (P7c).

The v3 cohort number (0.31 centered) averages the family form/layer log-probs over ALL
captured layers (16-47). If the concept signal is layer-localized, that averaging dilutes
it. This re-scores the SAME readout one layer at a time (per-layer LOO baseline-centering,
same faithfulness definition), to find the depth(s) where the internal state best tracks the
CoT's tagged concept. No capture, no GPU -- pure re-analysis of artifacts/cohort-perturn-readout-v3.json.
"""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "artifacts/cohort-perturn-readout-v3.json"
DEFAULT_OUT = ROOT / "artifacts/cohort-faithfulness-perlayer-v3.json"


def _scorer():
    spec = importlib.util.spec_from_file_location(
        "general_concept_scorer", ROOT / "scripts/swe_task_state_v4_general_concept_scorer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scores_at_layers(per_layer_ids: list[dict[int, float]], layer_idxs, family_ids) -> dict[str, float]:
    """family -> mean form log-prob over one or more captured layer indices."""
    import math

    scores = {}
    for fam, ids in family_ids.items():
        vals = [
            per_layer_ids[li][tid]
            for li in layer_idxs
            for tid in ids
            if li < len(per_layer_ids) and tid in per_layer_ids[li]
        ]
        if vals:
            scores[fam] = math.fsum(vals) / len(vals)
    return scores


def _scores_at_layer(per_layer_ids: list[dict[int, float]], layer_idx: int, family_ids) -> dict[str, float]:
    return _scores_at_layers(per_layer_ids, [layer_idx], family_ids)


def _faith_at(prepped, layer_idxs, family_ids, scorer) -> float:
    rows = [{"family_scores": _scores_at_layers(pl, layer_idxs, family_ids)} for _, pl in prepped]
    for r in rows:
        r["top1"] = scorer._top1(r["family_scores"])
    scorer._attach_centered_top1(rows)
    n = len(prepped)
    return sum(rows[i]["top1_centered"] == prepped[i][0] for i in range(n)) / n if n else 0.0


def compute(report_path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    scorer = _scorer()
    report = json.loads(Path(report_path).read_text())
    experiments = report.get("experiments", [])
    family_ids = scorer.family_token_ids()

    # precompute per-experiment: (tag, [ {tid:logprob} per captured layer ]) and the layer labels
    prepped = []
    layer_labels: list[int] = []
    for e in experiments:
        tag = (e.get("metadata") or {}).get("tag")
        if not tag or tag == "none":
            continue
        per_layer = scorer._layer_logprobs(e)
        if not layer_labels:
            layer_labels = [ly.get("layer") for ly in e.get("layers", [])]
        prepped.append((tag, per_layer))

    n_layers = len(layer_labels)
    n = len(prepped)
    per_layer_faith = []
    for li in range(n_layers):
        rows = [
            {"family_scores": _scores_at_layer(pl, li, family_ids), "top1": None}
            for _, pl in prepped
        ]
        for r in rows:
            r["top1"] = scorer._top1(r["family_scores"])
        scorer._attach_centered_top1(rows)
        hits_c = sum(rows[i]["top1_centered"] == prepped[i][0] for i in range(n))
        hits_r = sum(rows[i]["top1"] == prepped[i][0] for i in range(n))
        per_layer_faith.append(
            {
                "layer": layer_labels[li],
                "faithfulness_centered": round(hits_c / n, 4),
                "faithfulness_raw": round(hits_r / n, 4),
            }
        )

    best = max(per_layer_faith, key=lambda d: d["faithfulness_centered"])
    best_li = layer_labels.index(best["layer"])

    # contiguous late-layer bands (averaging denoises a single noisy layer)
    def band(lo: int, hi: int):
        idxs = [i for i, ll in enumerate(layer_labels) if lo <= ll <= hi]
        return {"band": f"{lo}-{hi}", "n_layers": len(idxs),
                "faithfulness_centered": round(_faith_at(prepped, idxs, family_ids, scorer), 4)}

    bands = [band(lo, 47) for lo in (44, 42, 40, 38, 36, 32)]
    bands += [band(39, 43), band(40, 44), band(38, 45)]
    best_band = max(bands, key=lambda d: d["faithfulness_centered"])

    # per-family recall at the best layer (centered)
    rows = [{"family_scores": _scores_at_layer(pl, best_li, family_ids)} for _, pl in prepped]
    for r in rows:
        r["top1"] = scorer._top1(r["family_scores"])
    scorer._attach_centered_top1(rows)
    by_tag = defaultdict(list)
    for i, (tag, _) in enumerate(prepped):
        by_tag[tag].append(rows[i]["top1_centered"] == tag)
    best_layer_recall = {
        fam: {"n": len(hits), "recall_centered": round(sum(hits) / len(hits), 3)}
        for fam, hits in sorted(by_tag.items(), key=lambda kv: -len(kv[1]))
    }

    return {
        "kind": "swe_task_state_v4_cohort_faithfulness_perlayer_v3",
        "report": str(report_path),
        "n_turns": n,
        "n_layers": n_layers,
        "all_layer_avg_centered": 0.307,  # v3 baseline for reference
        "best_layer": best["layer"],
        "best_layer_centered": best["faithfulness_centered"],
        "best_layer_raw": best["faithfulness_raw"],
        "best_band": best_band,
        "bands": bands,
        "per_layer": per_layer_faith,
        "best_layer_per_family_recall": best_layer_recall,
    }


def main(argv: Any = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    result = compute(args.report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"n turns: {result['n_turns']} | all-layer-avg centered: {result['all_layer_avg_centered']}")
    print(f"BEST LAYER: {result['best_layer']} -> centered {result['best_layer_centered']} (raw {result['best_layer_raw']})")
    print(f"BEST BAND: {result['best_band']['band']} -> centered {result['best_band']['faithfulness_centered']}")
    print("bands:", ", ".join(f"{b['band']}={b['faithfulness_centered']}" for b in result["bands"]))
    print("per-layer centered faithfulness:")
    for d in result["per_layer"]:
        bar = "#" * round(d["faithfulness_centered"] * 60)
        print(f"  L{d['layer']:2d} {d['faithfulness_centered']:.3f} {bar}")
    print(f"\nper-family recall at best layer L{result['best_layer']}:")
    for fam, s in result["best_layer_per_family_recall"].items():
        print(f"  {fam:24s} n={s['n']:3d} recall={s['recall_centered']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
