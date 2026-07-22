#!/usr/bin/env python3
"""Analyze the J-space swap flip rates: per-trial detail + bootstrap CI on Δ_J vs Δ_⊥."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "artifacts/jspace-swap-experiment.json"
TRIALS = ROOT / "artifacts/jspace-swap-experiment-trials.json"


def main() -> int:
    d = json.loads(RESULT.read_text())
    t = json.loads(TRIALS.read_text())
    print(f"n_pairs={d['n_pairs']} band={d['band']} clean_items={d['clean_items']}")
    print("per-trial (strict flip to C'):  dJ_nm | dperp_nm | full")
    for r in t:
        rr = r["res"]
        print(f"  {r['p']:24s} {r['C']!r:9s} -> {r['pprime']:24s} {r['Cp']!r:9s} | "
              f"dJ={int(rr['dJ_nm']['strict'])} dperp={int(rr['dperp_nm']['strict'])} "
              f"full={int(rr['full']['strict'])}  tok(dJ)={rr['dJ_nm']['tok']!r}")

    for metric in ("strict", "loose"):
        dj = [r["res"]["dJ_nm"][metric] for r in t]
        dp = [r["res"]["dperp_nm"][metric] for r in t]
        n = len(t)
        random.seed(0)
        diffs = []
        for _ in range(20000):
            idx = [random.randrange(n) for _ in range(n)]
            diffs.append(sum(dj[i] for i in idx) / n - sum(dp[i] for i in idx) / n)
        diffs.sort()
        print(f"\n[{metric}] dJ_nm={sum(dj)/n:.3f}  dperp_nm={sum(dp)/n:.3f}  "
              f"diff={sum(dj)/n - sum(dp)/n:.3f}  95%CI(diff)=[{diffs[500]:.3f}, {diffs[19500]:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
