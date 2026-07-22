#!/usr/bin/env python3
"""J-space vs non-J-space causal swap on two-hop trials (the reproduction). Loads model once.

For each two-hop minimal pair (P->C, P'->C', same length, differing only in the cue that sets the
intermediate concept), we patch at the DIFFERING (subject) positions across a middle-layer BAND
(final-position patching is overridden by downstream re-derivation from the explicit cue). At each
(layer L, differing position p) the empirical concept-swap delta is Δ = h_{P'}[L,p] - h_P[L,p];
we split it into Δ_J = P_J Δ (row(J[L]), what the lens reads) and Δ_perp (the non-J remainder,
most of the variance), norm-match them, inject each across the band, and measure how often P's
greedy answer flips. Strict flip = answer==C'; loose flip = answer!=C. Controls: full Δ, raw
components, norm-matched random directions in row(J)/ker(J), no-op. Gates to pairs the model
solves (both hops, single-token, distinct). Run via run_jspace_swap_experiment.sh (CUDA env)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MULTIHOP = ROOT / ".cache/research/anthropics-jacobian-lens/data/evaluations/lens-eval-multihop.json"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_P = _load("run_jlens_patch", "scripts/run_jlens_patch.py")
_JD = _load("jspace_decompose", "scripts/jspace_decompose.py")


def _clean_items(llm, tok) -> list[dict]:
    items = json.loads(MULTIHOP.read_text())["items"]
    clean = []
    for it in items:
        ids = tok.encode(it["prompt"], add_special_tokens=True)
        got = _P.greedy_token(llm, ids, it["prompt"])
        exp = {tok.encode(it["target"], add_special_tokens=False)[0],
               tok.encode(" " + it["target"], add_special_tokens=False)[0]}
        if got in exp:
            clean.append({**it, "ids": ids, "answer_tid": got, "answer": tok.decode([got])})
    return clean


def _minimal_pairs(clean: list[dict], *, max_diff: int) -> list[tuple[dict, dict, list[int]]]:
    pairs = []
    for a in clean:
        for b in clean:
            if a["name"] == b["name"] or len(a["ids"]) != len(b["ids"]) or a["answer_tid"] == b["answer_tid"]:
                continue
            diff = [i for i, (x, y) in enumerate(zip(a["ids"], b["ids"], strict=True)) if x != y]
            if 1 <= len(diff) <= max_diff:
                pairs.append((a, b, diff))
    return pairs


def run(llm, *, band: list[int], max_diff: int, max_pairs: int) -> dict[str, Any]:
    import torch

    tok = llm.get_tokenizer()
    clean = _clean_items(llm, tok)
    pairs = _minimal_pairs(clean, max_diff=max_diff)[:max_pairs]

    projectors = {L: _JD.row_space_projector(_JD.load_J(L, device="cuda"))[0] for L in band}

    conditions = ["full", "dJ_raw", "dperp_raw", "dJ_nm", "dperp_nm", "rand_row_nm", "rand_ker_nm", "noop"]
    agg = {c: {"strict": 0, "loose": 0, "n": 0} for c in conditions}
    per_trial = []

    for a, b, diff in pairs:
        tid_C, tid_Cp = a["answer_tid"], b["answer_tid"]
        h_p = _P.capture_all_layers(llm, a["ids"], a["prompt"])
        h_pp = _P.capture_all_layers(llm, b["ids"], b["prompt"])
        # build per-condition band inject dicts: {layer: [(pos, delta), ...]}
        inj = {c: defaultdict(list) for c in conditions}
        for L in band:
            P_J = projectors[L]
            for p in diff:
                delta = (h_pp[L][p] - h_p[L][p]).to("cuda", dtype=torch.float32)
                dJ, dperp = _JD.decompose(delta, P_J)
                dJ_nm, dperp_nm = _JD.norm_match(dJ, dperp)
                rr, rk = _JD.random_in_subspaces(P_J, torch.linalg.norm(dJ_nm), seed_vec=torch.flip(delta, dims=[0]))
                for c, dv in [("full", delta), ("dJ_raw", dJ), ("dperp_raw", dperp), ("dJ_nm", dJ_nm),
                              ("dperp_nm", dperp_nm), ("rand_row_nm", rr), ("rand_ker_nm", rk),
                              ("noop", torch.zeros_like(delta))]:
                    inj[c][L].append((p, dv.to(torch.bfloat16)))
        row = {"p": a["name"], "pprime": b["name"], "C": a["answer"], "Cp": b["answer"], "n_diff": len(diff), "res": {}}
        for c in conditions:
            tid = _P.greedy_token(llm, a["ids"], a["prompt"], inject=dict(inj[c]))
            strict, loose = tid == tid_Cp, tid != tid_C
            agg[c]["strict"] += int(strict)
            agg[c]["loose"] += int(loose)
            agg[c]["n"] += 1
            row["res"][c] = {"tok": tok.decode([tid]), "strict": strict, "loose": loose}
        per_trial.append(row)

    flip = {c: {"strict_rate": round(v["strict"] / v["n"], 3) if v["n"] else 0.0,
                "loose_rate": round(v["loose"] / v["n"], 3) if v["n"] else 0.0, "n": v["n"]}
            for c, v in agg.items()}
    return {"kind": "jspace_swap_experiment", "band": [band[0], band[-1]], "n_pairs": len(pairs),
            "clean_items": len(clean), "flip": flip, "per_trial": per_trial}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--band", type=int, nargs=2, default=[16, 40])
    ap.add_argument("--max-diff", type=int, default=3)
    ap.add_argument("--max-pairs", type=int, default=40)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/jspace-swap-experiment.json")
    args = ap.parse_args()

    llm = _P.load_llm()
    band = list(range(args.band[0], args.band[1] + 1))
    result = run(llm, band=band, max_diff=args.max_diff, max_pairs=args.max_pairs)
    trials = result.pop("per_trial")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    (args.out.parent / (args.out.stem + "-trials.json")).write_text(json.dumps(trials))
    print(f"clean_items={result['clean_items']} n_pairs={result['n_pairs']} band={result['band']}")
    print("condition          strict(->Cprime)  loose(!=C)   n")
    for c in ["full", "dJ_nm", "dperp_nm", "dJ_raw", "dperp_raw", "rand_row_nm", "rand_ker_nm", "noop"]:
        f = result["flip"][c]
        print(f"  {c:16s} {f['strict_rate']:<16.3f} {f['loose_rate']:<11.3f} {f['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
