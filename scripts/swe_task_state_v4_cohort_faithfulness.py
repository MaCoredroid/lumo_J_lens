#!/usr/bin/env python3
"""Cohort faithfulness: internal concept readout vs the CoT's tagged concept (P7c).

Joins a per-turn VJP-lens readout (scored over the general concept-form vocab) to the
per-turn CoT tags, and asks: at each turn's end-of-thinking boundary, does the internal
top-1 concept family match the concept the model's own thinking expressed (the tag)?

Faithfulness = fraction of comparable turns where internal top-1 == tag. Reported both
raw and baseline-centered (the scorer's leave-one-out cross-boundary centering, which
removes each family's frequency bias), with per-family recall, per-task breakdown, and
the majority-class baseline to beat. Turns tagged `none` (untaggable) are excluded.
Read-only; no model, no GPU.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "artifacts/cohort-perturn-readout-v3.json"
DEFAULT_OUT = ROOT / "artifacts/cohort-faithfulness-v3.json"


def _scorer():
    spec = importlib.util.spec_from_file_location(
        "general_concept_scorer", ROOT / "scripts/swe_task_state_v4_general_concept_scorer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute(report_path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    scorer = _scorer()
    report = json.loads(Path(report_path).read_text())  # load once (report can be ~600 MB)
    experiments = report.get("experiments", [])
    meta_by_id = {e.get("id"): (e.get("metadata") or {}) for e in experiments}
    family_ids = scorer.family_token_ids()
    rows = []
    for i, e in enumerate(experiments):
        scores = scorer.score_experiment(e, family_ids)
        rows.append({"index": i, "id": e.get("id"), "family_scores": scores, "top1": scorer._top1(scores)})
    scorer._attach_centered_top1(rows)

    records = []
    for r in rows:
        m = meta_by_id.get(r["id"], {})
        tag = m.get("tag")
        if not tag or tag == "none":
            continue
        records.append(
            {
                "id": r["id"],
                "task": m.get("task"),
                "turn": m.get("turn"),
                "tag": tag,
                "top1_raw": r["top1"],
                "top1_centered": r.get("top1_centered"),
                "hit_raw": r["top1"] == tag,
                "hit_centered": r.get("top1_centered") == tag,
            }
        )

    n = len(records)
    faith_raw = sum(r["hit_raw"] for r in records) / n if n else 0.0
    faith_centered = sum(r["hit_centered"] for r in records) / n if n else 0.0

    # per-family recall (centered): of turns tagged X, how many read X internally
    per_family = {}
    by_tag = defaultdict(list)
    for r in records:
        by_tag[r["tag"]].append(r)
    for fam, rs in sorted(by_tag.items(), key=lambda kv: -len(kv[1])):
        per_family[fam] = {
            "n": len(rs),
            "recall_centered": round(sum(x["hit_centered"] for x in rs) / len(rs), 3),
            "recall_raw": round(sum(x["hit_raw"] for x in rs) / len(rs), 3),
        }

    # per-task faithfulness
    per_task = {}
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    for task, rs in sorted(by_task.items()):
        per_task[task] = {
            "n": len(rs),
            "faithfulness_centered": round(sum(x["hit_centered"] for x in rs) / len(rs), 3),
        }

    # baselines to beat
    tag_counts = Counter(r["tag"] for r in records)
    majority_tag, majority_n = (tag_counts.most_common(1)[0] if tag_counts else ("", 0))
    n_families = len({r["tag"] for r in records})

    return {
        "kind": "swe_task_state_v4_cohort_faithfulness_v3",
        "report": str(report_path),
        "n_turns_compared": n,
        "faithfulness_centered": round(faith_centered, 4),
        "faithfulness_raw": round(faith_raw, 4),
        "baseline_majority_class": round(majority_n / n, 4) if n else 0.0,
        "baseline_majority_tag": majority_tag,
        "baseline_uniform": round(1.0 / n_families, 4) if n_families else 0.0,
        "n_distinct_tags": n_families,
        "per_family": per_family,
        "per_task": per_task,
        "records": records,
    }


def main(argv: Any = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    result = compute(args.report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # keep the artifact compact: drop per-record detail into a sibling if large
    args.out.write_text(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2) + "\n")
    (args.out.parent / (args.out.stem + "-records.json")).write_text(json.dumps(result["records"]))

    print(f"n turns compared: {result['n_turns_compared']}")
    print(f"faithfulness (centered): {result['faithfulness_centered']}  raw: {result['faithfulness_raw']}")
    print(f"baselines -> majority-class({result['baseline_majority_tag']}): "
          f"{result['baseline_majority_class']}  uniform: {result['baseline_uniform']}")
    print("per-family recall (centered):")
    for fam, s in result["per_family"].items():
        print(f"  {fam:24s} n={s['n']:3d} recall={s['recall_centered']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
