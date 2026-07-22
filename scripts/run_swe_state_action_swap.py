#!/usr/bin/env python3
"""State->action causal swap on the SWE agent: does the J-space concept component drive the action?

The agentic analog of the two-hop reproduction. Build the locate->modify concept-difference
direction Delta[L] = mean(residual over modify turns) - mean(residual over locate turns) at the
end-of-thinking boundary (from tagged cohort turns). On HELD-OUT locate turns, inject Delta's
J-space vs non-J-space component (norm-matched, scaled to the residual norm) across band L16-40
during prefill, generate the continuation, and classify the emitted action (locate-type command:
grep/find/ls/cat/read; modify-type: sed -i/edit/write/replace). If the small J-space component
shifts the agent's action from locate toward modify while the norm-matched non-J component does
not, the concept causally mediates behavior via the workspace channel -- upgrading our
correlational faithfulness to a causal claim. Loads model once. Run via the .sh (CUDA env)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BOUNDED = ROOT / ".cache/swe_jlens_cohort/cohort-perturn-prompts-bounded.json"

LOCATE = {"source_localization", "located_source", "defined_identifier"}
MODIFY = {"source_edit", "repair", "substitution_operation"}
LOC_KW = [" grep", " find ", " ls", " cat ", " head", " tail", "read_file", "grep_search", " rg ", " less ", " awk", "search"]
MOD_KW = ["sed -i", " edit", "write_file", " replace", " insert", "apply_patch", " patch", ">>", "rewrite", " tee "]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_P = _load("run_jlens_patch", "scripts/run_jlens_patch.py")
_JD = _load("jspace_decompose", "scripts/jspace_decompose.py")


def classify(text: str) -> str:
    loc = sum(text.count(k) for k in LOC_KW)
    mod = sum(text.count(k) for k in MOD_KW)
    if mod > loc:
        return "modify"
    if loc > mod:
        return "locate"
    return "none"


def _split(prompts):
    by_task = defaultdict(list)
    for p in prompts:
        tag = p["metadata"]["tag"]
        grp = "locate" if tag in LOCATE else ("modify" if tag in MODIFY else None)
        if grp:
            by_task[p["metadata"]["task"]].append({**p, "grp": grp})
    tasks = sorted(by_task)
    train_tasks = set(tasks[::2])
    train = [p for t in tasks if t in train_tasks for p in by_task[t]]
    test = [p for t in tasks if t not in train_tasks for p in by_task[t] if p["grp"] == "locate"]
    return train, test


def run(llm, *, band: list[int], n_train: int, n_test: int, alpha: float) -> dict[str, Any]:
    import torch

    tok = llm.get_tokenizer()
    prompts = json.loads(BOUNDED.read_text())
    train, test = _split(prompts)
    loc_tr = [p for p in train if p["grp"] == "locate"][:n_train]
    mod_tr = [p for p in train if p["grp"] == "modify"][:n_train]
    test = test[:n_test]

    # capture end-of-thinking residuals -> mean per group per band layer
    def mean_resid(turns):
        acc = {L: None for L in band}
        for p in turns:
            ids = p["token_ids"]
            h = _P.capture_all_layers(llm, ids, tok.decode(ids))
            for L in band:
                v = h[L][-1].float()
                acc[L] = v if acc[L] is None else acc[L] + v
        return {L: acc[L] / len(turns) for L in band}

    mloc, mmod = mean_resid(loc_tr), mean_resid(mod_tr)
    delta = {L: (mmod[L] - mloc[L]) for L in band}                 # locate -> modify concept direction

    projectors = {}
    for L in band:
        J_L = _JD.load_J(L, device="cuda")
        projectors[L], _, _ = _JD.row_space_projector(J_L)
        projectors[L] = projectors[L].cpu()
        del J_L
        torch.cuda.empty_cache()

    # per-layer J/non-J split of the concept direction, each scaled to alpha * ||residual[L]|| (norm-matched)
    variants = {c: {} for c in ["full", "dJ", "dperp", "rand_row", "rand_ker", "noop"]}
    for L in band:
        P_J = projectors[L]
        dJ, dperp = _JD.decompose(delta[L], P_J)
        scale = alpha * float(torch.linalg.norm(mloc[L]))         # target injection norm
        def unit(x):
            n = torch.linalg.norm(x)
            return x * (scale / n) if n > 1e-6 else x
        rr, rk = _JD.random_in_subspaces(P_J, torch.tensor(scale), seed_vec=torch.flip(delta[L], dims=[0]))
        variants["full"][L] = unit(delta[L])
        variants["dJ"][L] = unit(dJ)
        variants["dperp"][L] = unit(dperp)
        variants["rand_row"][L] = rr
        variants["rand_ker"][L] = rk
        variants["noop"][L] = torch.zeros_like(delta[L])

    conds = list(variants)
    dist = {c: Counter_like() for c in ["baseline"] + conds}
    per_trial = []
    for p in test:
        ids, text = p["token_ids"], tok.decode(p["token_ids"])
        base = classify(_P.generate_text(llm, ids, text, max_tokens=110))
        dist["baseline"].add(base)
        row = {"task": p["metadata"]["task"], "turn": p["metadata"]["turn"], "baseline": base, "res": {}}
        for c in conds:
            inj = {L: [(-1, variants[c][L].to(torch.bfloat16))] for L in band}
            act = classify(_P.generate_text(llm, ids, text, max_tokens=110, inject=inj))
            dist[c].add(act)
            row["res"][c] = act
        per_trial.append(row)

    def modify_rate(c):
        d = dist[c]
        return round(d.c["modify"] / max(1, d.total()), 3)

    return {
        "kind": "swe_state_action_swap", "band": [band[0], band[-1]], "alpha": alpha,
        "n_train_locate": len(loc_tr), "n_train_modify": len(mod_tr), "n_test": len(test),
        "action_dist": {c: dist[c].c for c in dist},
        "modify_rate": {c: modify_rate(c) for c in dist},
        "per_trial": per_trial,
    }


class Counter_like:
    def __init__(self):
        self.c = {"locate": 0, "modify": 0, "none": 0}
    def add(self, k):
        self.c[k] = self.c.get(k, 0) + 1
    def total(self):
        return sum(self.c.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--band", type=int, nargs=2, default=[16, 40])
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/swe-state-action-swap.json")
    args = ap.parse_args()

    llm = _P.load_llm(max_model_len=32768, gpu_mem=0.82)
    band = list(range(args.band[0], args.band[1] + 1))
    result = run(llm, band=band, n_train=args.n_train, n_test=args.n_test, alpha=args.alpha)
    trials = result.pop("per_trial")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    (args.out.parent / (args.out.stem + "-trials.json")).write_text(json.dumps(trials))
    print(f"n_train_loc={result['n_train_locate']} n_train_mod={result['n_train_modify']} "
          f"n_test={result['n_test']} alpha={result['alpha']} band={result['band']}")
    print("condition       action_dist{locate,modify,none}   modify_rate")
    for c in ["baseline", "full", "dJ", "dperp", "rand_row", "rand_ker", "noop"]:
        print(f"  {c:12s} {result['action_dist'][c]}   {result['modify_rate'][c]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
