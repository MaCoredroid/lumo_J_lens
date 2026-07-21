#!/usr/bin/env python3
"""Learned per-concept probe: logistic regression over the lens's token-logprob features (P7c).

The hand-form + argmax-of-means readout lets the substitution magnet dominate the code-modification
blob, so the weak fine families (repair, source_edit) score ~0.1-0.18. A PROPER per-concept probe
can learn to down-weight the magnet tokens and up-weight family-specific ones. This trains a
multinomial logistic regression on per-turn scored-token logprobs (from the v5 pooled readout, late
layers) to predict the fine family, with an honest 10/10 train/test task split, and reports held-out
per-family recall vs the argmax-readout baseline. It is the rigorous test of whether the fine
distinctions are DECODABLE from the residual (via the lens) at all. Uses sklearn.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V5_READOUT = ROOT / "artifacts/cohort-perturn-readout-v5.json"
TAGS = ROOT / ".cache/swe_jlens_cohort/cohort_cot_tags.json"
OUT = ROOT / "artifacts/cohort-concept-linear-probe.json"
USE_LAYERS = [40, 41, 42, 43, 44, 45, 46, 47]  # late-layer band where concept signal lives


def _split() -> tuple[set[str], set[str]]:
    tasks = sorted(t["task"] for t in json.loads(TAGS.read_text())["tasks"])
    return {t for i, t in enumerate(tasks) if i % 2 == 0}, {t for i, t in enumerate(tasks) if i % 2 == 1}


def _features(readout: Path, layers: list[int]):
    import numpy as np

    d = json.loads(Path(readout).read_text())
    exps = d["experiments"]
    all_layers = [ly["layer"] for ly in exps[0]["layers"]]
    lidx = [all_layers.index(ln) for ln in layers]
    pool = sorted(s["token_id"] for s in exps[0]["layers"][0]["positions"][0]["jacobian_lens"]["scored_tokens"])
    col = {t: j for j, t in enumerate(pool)}

    X, y, tasks, meta = [], [], [], []
    for e in exps:
        md = e.get("metadata") or {}
        tag = md.get("tag")
        if not tag or tag == "none":
            continue
        # mean-pool the token logprob over the late-layer band (denoises + keeps features at 219)
        acc = np.zeros(len(pool), dtype=np.float64)
        for li in lidx:
            for s in e["layers"][li]["positions"][0]["jacobian_lens"]["scored_tokens"]:
                acc[col[s["token_id"]]] += s["logprob"]
        X.append((acc / len(lidx)).astype(np.float32))
        y.append(tag)
        tasks.append(md.get("task"))
        meta.append({"id": e.get("id"), "task": md.get("task"), "turn": md.get("turn")})
    return np.array(X), np.array(y), np.array(tasks), meta


def compute(readout: Path = V5_READOUT, *, layers: list[int] = None, C: float = 0.05, folds: int = 5) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    layers = layers or USE_LAYERS
    X, y, tasks, meta = _features(readout, layers)

    # grouped k-fold by task: every turn is predicted while ITS task is held out -> honest, and
    # every family accumulates test support across folds (stable small-family estimates).
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=4000))
    cv = GroupKFold(n_splits=folds)
    proba = cross_val_predict(pipe, X, y, groups=tasks, cv=cv, method="predict_proba")
    classes = sorted(set(y))
    ci = {c: j for j, c in enumerate(classes)}
    pred = np.array([classes[j] for j in proba.argmax(1)])
    conf_val = proba.max(1)

    by = defaultdict(lambda: [0, 0])           # family -> [hits, support]
    confusion = defaultdict(lambda: defaultdict(int))
    for p, t in zip(pred, y, strict=True):
        by[t][1] += 1
        confusion[t][p] += 1
        if p == t:
            by[t][0] += 1
    per_family = {}
    for f, (h, s) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        top_err = sorted(((k, v) for k, v in confusion[f].items() if k != f), key=lambda kv: -kv[1])[:2]
        per_family[f] = {"n": s, "recall": round(h / s, 3), "confused_with": dict(top_err)}

    # per-turn held-out records + high-confidence disagreements (candidate divergences)
    records = []
    for i, m in enumerate(meta):
        tag = y[i]
        records.append({
            **m, "tag": tag, "pred": pred[i], "agree": bool(pred[i] == tag),
            "confidence": round(float(conf_val[i]), 3),
            "p_tag": round(float(proba[i][ci[tag]]), 3),
        })
    return {
        "kind": "concept_linear_probe_groupkfold",
        "layers": layers,
        "C": C,
        "folds": folds,
        "n_turns": int(len(y)),
        "n_features": int(X.shape[1]),
        "held_out_accuracy": round(float((pred == y).mean()), 4),
        "held_out_macro_recall": round(float(np.mean([h / s for h, s in by.values()])), 4),
        "per_family": per_family,
        "records": records,
    }


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--C", type=float, default=0.05)
    p.add_argument("--layers", type=int, nargs="+", default=None)
    a = p.parse_args()
    r = compute(layers=a.layers, C=a.C)
    records = r.pop("records")
    OUT.write_text(json.dumps(r, indent=2) + "\n")
    (OUT.parent / (OUT.stem + "-records.json")).write_text(json.dumps(records))
    print(f"linear probe (band-mean {r['layers']}, C={r['C']}, {r['n_features']} feats, "
          f"{r['folds']}-fold GroupKFold over {r['n_turns']} turns)")
    print(f"held-out accuracy={r['held_out_accuracy']}  macro-recall={r['held_out_macro_recall']}")
    print("held-out per-family recall (vs argmax-readout baseline):")
    base = {"source_localization": 0.443, "verification": 0.586, "source_edit": 0.176, "repair": 0.1,
            "failure_confirmation": 0.2, "dependency_unavailable": 0.0, "broad_success": 0.125,
            "task_resolution": 0.0, "test_success": 0.0}
    for fam, s in r["per_family"].items():
        b = base.get(fam)
        delta = f" (argmax {b:.2f} -> {'%+.2f' % (s['recall'] - b)})" if b is not None else ""
        print(f"  {fam:24s} n={s['n']:3d} recall={s['recall']}{delta}")
    dis = sorted((x for x in records if not x["agree"]), key=lambda x: -x["confidence"])
    print(f"\ntop confident held-out disagreements (candidate divergences) of {len(dis)}:")
    for x in dis[:8]:
        print(f"  {x['task']} t{x['turn']:02d} tag={x['tag']:20s} probe={x['pred']:20s} conf={x['confidence']} p_tag={x['p_tag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
