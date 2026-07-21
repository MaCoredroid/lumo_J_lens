#!/usr/bin/env python3
"""Diagnose weak-family failure mode: lens confusion matrix at the peak layer (P7c).

For each tagged concept family, what does the lens's baseline-centered top-1 actually say?
If a weak family (repair, source_edit, ...) is mostly misread as a SEMANTICALLY ADJACENT
family (edit<->substitution<->repair are all "change code"), the fix is to merge/sharpen the
boundary; if it scatters randomly, the fix is better per-concept token forms. Reads one readout
at a chosen layer + vocab. CPU-only.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READOUT = ROOT / "artifacts/cohort-perturn-readout-v4.json"
VOCAB_V2 = ROOT / "configs/swe_task_state_v4_general_concept_forms_v2.json"
DEFAULT_OUT = ROOT / "artifacts/cohort-faithfulness-confusion-L44.json"
PEAK = 44


def _scorer():
    spec = importlib.util.spec_from_file_location(
        "general_concept_scorer", ROOT / "scripts/swe_task_state_v4_general_concept_scorer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute(readout: Path = DEFAULT_READOUT, *, vocab: Path = VOCAB_V2, layer: int = PEAK) -> dict[str, Any]:
    import math

    scorer = _scorer()
    family_ids = scorer.family_token_ids(vocab)
    report = json.loads(Path(readout).read_text())
    exps = report["experiments"]
    labels = [ly.get("layer") for ly in exps[0]["layers"]]
    li = labels.index(layer)

    tags, rows = [], []
    for e in exps:
        tag = (e.get("metadata") or {}).get("tag")
        if not tag or tag == "none":
            continue
        layer_ids = scorer._layer_logprobs(e)[li]
        scores = {}
        for fam, ids in family_ids.items():
            vals = [layer_ids[tid] for tid in ids if tid in layer_ids]
            if vals:
                scores[fam] = math.fsum(vals) / len(vals)
        rows.append({"family_scores": scores})
        tags.append(tag)
    scorer._attach_centered_top1(rows)

    confusion: dict[str, Counter] = defaultdict(Counter)
    for tag, r in zip(tags, rows, strict=True):
        confusion[tag][r["top1_centered"]] += 1

    out = {}
    for tag in sorted(confusion, key=lambda t: -sum(confusion[t].values())):
        c = confusion[tag]
        n = sum(c.values())
        top3 = c.most_common(4)
        out[tag] = {
            "n": n,
            "recall": round(c.get(tag, 0) / n, 3),
            "predicted_as": [{"family": f, "frac": round(k / n, 3)} for f, k in top3],
        }
    return {"kind": "faithfulness_confusion", "layer": layer, "vocab": str(vocab), "by_true_family": out}


def main() -> int:
    result = compute()
    Path(DEFAULT_OUT).write_text(json.dumps(result, indent=2) + "\n")
    print(f"=== confusion @ L{result['layer']} (true family -> where the lens puts it) ===")
    for tag, s in result["by_true_family"].items():
        pred = "  ".join(f"{p['family']}={p['frac']}" for p in s["predicted_as"])
        print(f"  {tag:22s} n={s['n']:3d} recall={s['recall']:.3f} | {pred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
