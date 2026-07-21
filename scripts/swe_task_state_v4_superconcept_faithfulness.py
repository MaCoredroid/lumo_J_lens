#!/usr/bin/env python3
"""Super-concept faithfulness: is the lens trustworthy at a coarser granularity? (P7c)

The L44 confusion matrix shows the weak fine families fail by ADJACENT-family confusion inside
two blobs: a code-modification blob {source_edit, repair, substitution} and a check/success blob
{verification, focused_validation, failure_confirmation, broad_success, test_success,
task_resolution}. This scores the residual at the SUPER-concept level (union of each group's
forms), to test whether the residual cleanly encodes the coarse concept even when it can't split
the fine ones -- i.e. the ceiling a per-concept probe would have to beat. Reads one readout at a
layer + vocab. CPU-only.
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READOUT = ROOT / "artifacts/cohort-perturn-readout-v4.json"
VOCAB_V2 = ROOT / "configs/swe_task_state_v4_general_concept_forms_v2.json"
DEFAULT_OUT = ROOT / "artifacts/cohort-superconcept-faithfulness-L44.json"
PEAK = 44

# super-concept grouping, from the L44 confusion structure
SUPER = {
    "locate": ["source_localization", "located_source", "defined_identifier"],
    "modify_code": ["source_edit", "repair", "substitution_operation"],
    "verify_check": ["verification", "focused_validation", "failure_confirmation"],
    "success_resolve": ["broad_success", "test_success", "task_resolution"],
    "fault": ["runtime_name_failure", "dependency_unavailable"],
}


def _scorer():
    spec = importlib.util.spec_from_file_location(
        "general_concept_scorer", ROOT / "scripts/swe_task_state_v4_general_concept_scorer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _super_of(fine: str) -> str | None:
    for s, members in SUPER.items():
        if fine in members:
            return s
    return None


def compute(readout: Path = DEFAULT_READOUT, *, vocab: Path = VOCAB_V2, layer: int = PEAK) -> dict[str, Any]:
    scorer = _scorer()
    fine_ids = scorer.family_token_ids(vocab)
    # super-family token ids = union of member fine-family forms
    super_ids = {
        s: sorted({tid for m in members if m in fine_ids for tid in fine_ids[m]})
        for s, members in SUPER.items()
    }

    report = json.loads(Path(readout).read_text())
    exps = report["experiments"]
    labels = [ly.get("layer") for ly in exps[0]["layers"]]
    li = labels.index(layer)

    super_tags, rows = [], []
    for e in exps:
        tag = (e.get("metadata") or {}).get("tag")
        s_tag = _super_of(tag) if tag and tag != "none" else None
        if not s_tag:
            continue
        layer_ids = scorer._layer_logprobs(e)[li]
        scores = {}
        for s, ids in super_ids.items():
            vals = [layer_ids[tid] for tid in ids if tid in layer_ids]
            if vals:
                scores[s] = math.fsum(vals) / len(vals)
        rows.append({"family_scores": scores})
        super_tags.append(s_tag)
    scorer._attach_centered_top1(rows)

    n = len(rows)
    hits = sum(rows[i]["top1_centered"] == super_tags[i] for i in range(n))
    by = defaultdict(list)
    conf = defaultdict(lambda: defaultdict(int))
    for i in range(n):
        by[super_tags[i]].append(rows[i]["top1_centered"] == super_tags[i])
        conf[super_tags[i]][rows[i]["top1_centered"]] += 1
    per_super = {
        s: {
            "n": len(h),
            "recall": round(sum(h) / len(h), 3),
            "confused_with": sorted(
                ({k: round(v / len(h), 3) for k, v in conf[s].items() if k != s}).items(),
                key=lambda kv: -kv[1],
            )[:2],
        }
        for s, h in sorted(by.items(), key=lambda kv: -len(kv[1]))
    }
    n_super = len(SUPER)
    return {
        "kind": "superconcept_faithfulness",
        "layer": layer,
        "grouping": SUPER,
        "n_turns": n,
        "faithfulness_centered": round(hits / n, 4) if n else 0.0,
        "baseline_uniform": round(1.0 / n_super, 4),
        "per_super": per_super,
    }


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, default=DEFAULT_READOUT)
    p.add_argument("--vocab", type=Path, default=VOCAB_V2)
    p.add_argument("--layer", type=int, default=PEAK)
    a = p.parse_args()
    r = compute(a.report, vocab=a.vocab, layer=a.layer)
    Path(DEFAULT_OUT).write_text(json.dumps(r, indent=2) + "\n")
    print(f"=== super-concept faithfulness @ L{r['layer']} (5 groups) ===")
    print(f"n={r['n_turns']}  centered={r['faithfulness_centered']}  (uniform {r['baseline_uniform']})")
    for s, d in r["per_super"].items():
        cw = "  ".join(f"{k}={v}" for k, v in d["confused_with"])
        print(f"  {s:16s} n={d['n']:3d} recall={d['recall']:.3f} | conf: {cw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
