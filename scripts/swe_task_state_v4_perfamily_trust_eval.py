#!/usr/bin/env python3
"""Honest per-family trust eval: train-derived vocab, held-out test tasks (P7c).

The goal is a lens that is TRUSTWORTHY per family, so a lens-vs-CoT disagreement is meaningful.
This measures trust without circularity: derive the data-driven concept forms on 10 TRAIN tasks,
then evaluate per-family recall on the 10 held-out TEST tasks' turns (v5 pooled readout, L44),
against the v2 vocab and against the super-concept grouping. If a family's recall is high AND
holds out-of-sample, a disagreement there is meaningful; if not, that family is over-fractioned.
CPU-only.
"""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V5_READOUT = ROOT / "artifacts/cohort-perturn-readout-v5.json"
V2 = ROOT / "configs/swe_task_state_v4_general_concept_forms_v2.json"
TAGS = ROOT / ".cache/swe_jlens_cohort/cohort_cot_tags.json"
OUT = ROOT / "artifacts/cohort-perfamily-trust-eval-L44.json"
PEAK = 44


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SCORER = _load("general_concept_scorer", "scripts/swe_task_state_v4_general_concept_scorer.py")
_DD = _load("datadriven", "scripts/swe_task_state_v4_datadriven_concept_forms.py")
_SUPER = _load("superconcept", "scripts/swe_task_state_v4_superconcept_faithfulness.py")


def _split() -> tuple[set[str], set[str]]:
    tasks = sorted(t["task"] for t in json.loads(TAGS.read_text())["tasks"])
    train = {t for i, t in enumerate(tasks) if i % 2 == 0}
    test = {t for i, t in enumerate(tasks) if i % 2 == 1}
    return train, test


def _train_vocab(tokenizer, train: set[str], pool: set[int]) -> dict[str, list[int]]:
    """data-driven forms derived on TRAIN tasks, filled from v2, restricted to the captured pool."""
    dd = _DD.derive(tokenizer, _DD._family_corpora(train_tasks=train))
    fam = {f: [x["token_id"] for x in forms if x["token_id"] in pool] for f, forms in dd["families"].items()}
    v2 = _SCORER.family_token_ids(V2)
    for f, ids in v2.items():
        if not fam.get(f):  # fill families the train split couldn't derive
            fam[f] = [t for t in ids if t in pool]
    return {f: ids for f, ids in fam.items() if ids}


def _recall(experiments, family_ids, layer, test: set[str], *, super_map=False) -> dict[str, Any]:
    import math

    labels = [ly.get("layer") for ly in experiments[0]["layers"]]
    li = labels.index(layer)
    tags, rows = [], []
    for e in experiments:
        md = e.get("metadata") or {}
        tag, task = md.get("tag"), md.get("task")
        if not tag or tag == "none" or task not in test:
            continue
        if super_map:
            tag = _SUPER._super_of(tag)
            if not tag:
                continue
        layer_ids = _SCORER._layer_logprobs(e)[li]
        scores = {}
        for fam, ids in family_ids.items():
            vals = [layer_ids[tid] for tid in ids if tid in layer_ids]
            if vals:
                scores[fam] = math.fsum(vals) / len(vals)
        rows.append({"family_scores": scores})
        tags.append(tag)
    _SCORER._attach_centered_top1(rows)
    n = len(rows)
    hits = sum(rows[i]["top1_centered"] == tags[i] for i in range(n))
    by = defaultdict(list)
    for i in range(n):
        by[tags[i]].append(rows[i]["top1_centered"] == tags[i])
    per = {f: {"n": len(h), "recall": round(sum(h) / len(h), 3)} for f, h in sorted(by.items(), key=lambda kv: -len(kv[1]))}
    return {"n": n, "faithfulness": round(hits / n, 4) if n else 0.0, "per_family": per}


def compute() -> dict[str, Any]:
    from transformers import AutoTokenizer

    snap = next(Path.home().glob(".cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/*"))
    tokenizer = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=False)

    train, test = _split()
    report = json.loads(V5_READOUT.read_text())
    exps = report["experiments"]
    pool = {s["token_id"] for s in exps[0]["layers"][0]["positions"][0]["jacobian_lens"]["scored_tokens"]}

    v2_ids = {f: [t for t in ids if t in pool] for f, ids in _SCORER.family_token_ids(V2).items()}
    dd_ids = _train_vocab(tokenizer, train, pool)
    super_ids = {
        s: sorted({t for m in members if m in v2_ids for t in v2_ids[m]})
        for s, members in _SUPER.SUPER.items()
    }

    return {
        "kind": "perfamily_trust_eval",
        "layer": PEAK,
        "n_train_tasks": len(train),
        "n_test_tasks": len(test),
        "test_v2_fine": _recall(exps, v2_ids, PEAK, test),
        "test_datadriven_fine": _recall(exps, dd_ids, PEAK, test),
        "test_super_v2": _recall(exps, super_ids, PEAK, test, super_map=True),
    }


def main() -> int:
    r = compute()
    OUT.write_text(json.dumps(r, indent=2) + "\n")
    print(f"=== held-out test tasks (n_test={r['n_test_tasks']}) @ L{r['layer']} ===")
    for key in ("test_v2_fine", "test_datadriven_fine", "test_super_v2"):
        c = r[key]
        print(f"\n{key}: n={c['n']} faithfulness={c['faithfulness']}")
        for fam, s in c["per_family"].items():
            print(f"  {fam:22s} n={s['n']:3d} recall={s['recall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
