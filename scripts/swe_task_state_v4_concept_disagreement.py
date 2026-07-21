#!/usr/bin/env python3
"""The payoff: a meaningful lens-vs-CoT disagreement detector on the trustworthy 3-concept lens.

At the granularity the residual actually supports -- locate / modify_code / assess, each with
held-out recall 0.68-0.83 -- a disagreement between the lens's concept and the CoT's concept is
a real signal, not noise. This computes, per turn, the lens's baseline-centered top-1 super-concept
vs the CoT's tagged super-concept, reports the agreement rate, and ranks the DISAGREEMENTS by a
confidence margin (how much more the residual encodes the lens's concept than the CoT's). High-margin
disagreements are candidate surface-internal divergences -- turns where the internal state encodes a
different mode of reasoning than the model said it was in -- for review. A triage tool, not a verdict
(the lens still errs ~20-30%). CPU-only.
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V5_READOUT = ROOT / "artifacts/cohort-perturn-readout-v5.json"
V2 = ROOT / "configs/swe_task_state_v4_general_concept_forms_v2.json"
TAGS = ROOT / ".cache/swe_jlens_cohort/cohort_cot_tags.json"
OUT = ROOT / "artifacts/cohort-concept-disagreements-L44.json"
PEAK = 44

# the trustworthy granularity (held-out per-family recall 0.68-0.83)
CONCEPTS = {
    "locate": ["source_localization", "located_source", "defined_identifier"],
    "modify_code": ["source_edit", "repair", "substitution_operation"],
    "assess": [
        "verification", "focused_validation", "failure_confirmation",
        "broad_success", "test_success", "task_resolution",
        "runtime_name_failure", "dependency_unavailable",
    ],
}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SCORER = _load("general_concept_scorer", "scripts/swe_task_state_v4_general_concept_scorer.py")
_CT = _load("cohort_traces", "scripts/swe_task_state_v4_cohort_traces.py")


def _super_of(fine: str) -> str | None:
    for s, members in CONCEPTS.items():
        if fine in members:
            return s
    return None


def compute(readout: Path = V5_READOUT, *, layer: int = PEAK, n_examples: int = 12) -> dict[str, Any]:
    fine = _SCORER.family_token_ids(V2)
    super_ids = {s: sorted({t for m in members if m in fine for t in fine[m]}) for s, members in CONCEPTS.items()}
    report = json.loads(Path(readout).read_text())
    exps = report["experiments"]
    labels = [ly.get("layer") for ly in exps[0]["layers"]]
    li = labels.index(layer)

    meta, rows = [], []
    for e in exps:
        md = e.get("metadata") or {}
        s_tag = _super_of(md.get("tag")) if md.get("tag") and md.get("tag") != "none" else None
        if not s_tag:
            continue
        layer_ids = {s["token_id"]: s["logprob"] for s in e["layers"][li]["positions"][0]["jacobian_lens"]["scored_tokens"]}
        scores = {}
        for s, ids in super_ids.items():
            vals = [layer_ids[t] for t in ids if t in layer_ids]
            if vals:
                scores[s] = math.fsum(vals) / len(vals)
        rows.append(scores)
        meta.append({"id": e.get("id"), "task": md.get("task"), "turn": md.get("turn"), "cot": s_tag})

    # leave-one-out cross-boundary centering (same as the scorer)
    n = len(rows)
    centered = []
    for i, r in enumerate(rows):
        cen = {}
        for f in r:
            others = [rows[j][f] for j in range(n) if j != i and f in rows[j]]
            base = math.fsum(others) / len(others) if others else 0.0
            cen[f] = r[f] - base
        centered.append(cen)

    records = []
    agree = 0
    for i, cen in enumerate(centered):
        lens = max(cen, key=cen.get)
        cot = meta[i]["cot"]
        hit = lens == cot
        agree += hit
        margin = round(cen[lens] - cen.get(cot, min(cen.values())), 4)
        records.append({**meta[i], "lens": lens, "agree": hit, "margin": margin})

    disagreements = sorted([r for r in records if not r["agree"]], key=lambda r: -r["margin"])

    # attach thinking snippets to the top disagreements
    survey = {v["task"]: v for v in _CT.survey()["usable"]}
    think_cache: dict[str, list[str]] = {}
    examples = []
    for r in disagreements[:n_examples]:
        task = r["task"]
        if task not in think_cache and task in survey:
            think_cache[task] = _CT.task_thinking_blocks(Path(survey[task]["trace"]))
        blocks = think_cache.get(task, [])
        snip = blocks[r["turn"] - 1][:280].replace("\n", " ") if r["turn"] - 1 < len(blocks) else ""
        examples.append({**r, "thinking_snippet": snip})

    return {
        "kind": "concept_disagreement_L44",
        "layer": layer,
        "concepts": list(CONCEPTS),
        "n_turns": n,
        "agreement_rate": round(agree / n, 4) if n else 0.0,
        "n_disagreements": len(disagreements),
        "disagreement_directions": dict(Counter(f"{r['cot']}->{r['lens']}" for r in disagreements).most_common()),
        "top_disagreement_examples": examples,
    }


def main() -> int:
    r = compute()
    OUT.write_text(json.dumps(r, indent=2) + "\n")
    print(f"3-concept lens @ L{r['layer']}: agreement {r['agreement_rate']} over {r['n_turns']} turns "
          f"({r['n_disagreements']} disagreements)")
    print("disagreement directions (cot -> lens):")
    for d, k in r["disagreement_directions"].items():
        print(f"  {d:28s} {k}")
    print("\ntop confident disagreements (candidate divergences):")
    for e in r["top_disagreement_examples"]:
        print(f"  {e['task']} t{e['turn']:02d} cot={e['cot']:11s} lens={e['lens']:11s} margin={e['margin']}")
        print(f"      \"{e['thinking_snippet'][:140]}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
