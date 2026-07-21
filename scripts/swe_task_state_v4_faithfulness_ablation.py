#!/usr/bin/env python3
"""Levers 2+3 ablation: bounded-vs-full context x v1-vs-v2 vocab, all at the L41 peak (P7c).

v1's 63 concept tokens are a strict subset of v2's 126, and the v4 (bounded, 535-turn) readout
scored the full v2 set, so a single pair of readouts (v3 full/293, v4 bounded/535) yields a
clean factorial at the Lever-1 peak layer:

  * context  : full (v3) vs bounded (v4)   -- does bounding the context degrade faithfulness?
  * coverage : 293 shared turns vs all 535 -- do the late turns (dropped in v3) behave differently?
  * vocab    : v1 (63) vs v2 (126 forms)   -- does the richer vocab lift faithfulness?

Centering is always done over exactly the evaluated turn-set. Reads the two readouts one at a
time (memory). CPU-only.
"""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V3_READOUT = ROOT / "artifacts/cohort-perturn-readout-v3.json"
V4_READOUT = ROOT / "artifacts/cohort-perturn-readout-v4.json"
VOCAB_V1 = ROOT / "configs/swe_task_state_v4_general_concept_forms.json"
VOCAB_V2 = ROOT / "configs/swe_task_state_v4_general_concept_forms_v2.json"
OUT = ROOT / "artifacts/cohort-faithfulness-ablation-L41.json"
PEAK_LAYER = 41


def _scorer():
    spec = importlib.util.spec_from_file_location(
        "general_concept_scorer", ROOT / "scripts/swe_task_state_v4_general_concept_scorer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prep(experiments, scorer):
    """[(id, task, turn, tag, [per-layer {tid:logprob}])] for tagged turns; + layer labels."""
    out, labels = [], []
    for e in experiments:
        md = e.get("metadata") or {}
        tag = md.get("tag")
        if not tag or tag == "none":
            continue
        if not labels:
            labels = [ly.get("layer") for ly in e.get("layers", [])]
        out.append((e.get("id"), md.get("task"), md.get("turn"), tag, scorer._layer_logprobs(e)))
    return out, labels


def _faith(prepped, labels, family_ids, layer, scorer, id_filter=None):
    import math

    li = labels.index(layer)
    sel = [p for p in prepped if id_filter is None or p[0] in id_filter]
    rows = []
    for _id, _task, _turn, tag, per_layer in sel:
        layer_ids = per_layer[li]
        scores = {}
        for fam, ids in family_ids.items():
            vals = [layer_ids[tid] for tid in ids if tid in layer_ids]
            if vals:
                scores[fam] = math.fsum(vals) / len(vals)
        rows.append({"family_scores": scores, "tag": tag})
    scorer._attach_centered_top1(rows)
    n = len(rows)
    hits = sum(rows[i]["top1_centered"] == sel[i][3] for i in range(n))
    # per-family recall
    by = defaultdict(list)
    for i in range(n):
        by[sel[i][3]].append(rows[i]["top1_centered"] == sel[i][3])
    recall = {f: round(sum(h) / len(h), 3) for f, h in sorted(by.items(), key=lambda kv: -len(kv[1]))}
    fam_n = {f: len(h) for f, h in by.items()}
    return {"n": n, "faithfulness_centered": round(hits / n, 4) if n else 0.0, "recall": recall, "fam_n": fam_n}


def compute() -> dict[str, Any]:
    scorer = _scorer()
    v1 = scorer.family_token_ids(VOCAB_V1)
    v2 = scorer.family_token_ids(VOCAB_V2)

    # v3 (full context, 293 turns) at L41, v1 vocab
    v3 = json.loads(V3_READOUT.read_text())
    v3_prep, v3_labels = _prep(v3["experiments"], scorer)
    shared_ids = {p[0] for p in v3_prep}
    r_v3_full_v1 = _faith(v3_prep, v3_labels, v1, PEAK_LAYER, scorer)
    del v3, v3_prep

    # v4 (bounded context, 535 turns) at L41, both vocabs, both coverages
    v4 = json.loads(V4_READOUT.read_text())
    v4_prep, v4_labels = _prep(v4["experiments"], scorer)
    r_v4_bounded_v1_all = _faith(v4_prep, v4_labels, v1, PEAK_LAYER, scorer)
    r_v4_bounded_v2_all = _faith(v4_prep, v4_labels, v2, PEAK_LAYER, scorer)
    r_v4_bounded_v1_shared = _faith(v4_prep, v4_labels, v1, PEAK_LAYER, scorer, id_filter=shared_ids)
    r_v4_bounded_v2_shared = _faith(v4_prep, v4_labels, v2, PEAK_LAYER, scorer, id_filter=shared_ids)

    return {
        "kind": "swe_task_state_v4_faithfulness_ablation_L41",
        "peak_layer": PEAK_LAYER,
        "cells": {
            "v3_full_v1_293": r_v3_full_v1,
            "v4_bounded_v1_shared293": r_v4_bounded_v1_shared,
            "v4_bounded_v2_shared293": r_v4_bounded_v2_shared,
            "v4_bounded_v1_all535": r_v4_bounded_v1_all,
            "v4_bounded_v2_all535": r_v4_bounded_v2_all,
        },
    }


def main() -> int:
    result = compute()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"=== faithfulness ablation @ L{result['peak_layer']} (centered) ===")
    for name, cell in result["cells"].items():
        print(f"  {name:28s} n={cell['n']:3d}  centered={cell['faithfulness_centered']}")
    print("\ninterpretation deltas:")
    c = result["cells"]
    print(f"  bounded vs full (v1, shared293): {c['v4_bounded_v1_shared293']['faithfulness_centered']} vs {c['v3_full_v1_293']['faithfulness_centered']}")
    print(f"  v2 vs v1 vocab (bounded, 535):   {c['v4_bounded_v2_all535']['faithfulness_centered']} vs {c['v4_bounded_v1_all535']['faithfulness_centered']}")
    print(f"  535 vs shared293 (v2, bounded):  {c['v4_bounded_v2_all535']['faithfulness_centered']} vs {c['v4_bounded_v2_shared293']['faithfulness_centered']}")
    print("\nv4 bounded v2 535 per-family recall:")
    for fam, rec in c["v4_bounded_v2_all535"]["recall"].items():
        print(f"  {fam:24s} n={c['v4_bounded_v2_all535']['fam_n'][fam]:3d} recall={rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
