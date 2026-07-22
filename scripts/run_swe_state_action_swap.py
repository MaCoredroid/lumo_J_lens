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
    """Classify by the EARLIEST action keyword (the first thing the agent moves to do)."""
    best_pos, best = len(text) + 1, "none"
    for kw in LOC_KW:
        i = text.find(kw)
        if 0 <= i < best_pos:
            best_pos, best = i, "locate"
    for kw in MOD_KW:
        i = text.find(kw)
        if 0 <= i < best_pos:
            best_pos, best = i, "modify"
    return best


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


def run(llm, *, band: list[int], n_train: int, n_test: int, alphas: list[float], force_cmd: bool = False) -> dict[str, Any]:
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

    # base J/non-J directions, NORM-MATCHED at the concept-difference scale (like the factual case).
    # Injection = alpha * base; sweep alpha to find the window (too small -> recovers, too big -> garbles).
    bdir = {c: {} for c in ["full", "dJ", "dperp", "rand_row", "rand_ker"]}
    diag = []
    for L in band:
        P_J = projectors[L]
        dJ, dperp = _JD.decompose(delta[L], P_J)
        dJ_nm, dperp_nm = _JD.norm_match(dJ, dperp)                # both at sqrt(||dJ||*||dperp||)
        g = float(torch.linalg.norm(dJ_nm))
        rr, rk = _JD.random_in_subspaces(P_J, torch.tensor(g), seed_vec=torch.flip(delta[L], dims=[0]))
        bdir["full"][L] = delta[L]
        bdir["dJ"][L] = dJ_nm
        bdir["dperp"][L] = dperp_nm
        bdir["rand_row"][L] = rr
        bdir["rand_ker"][L] = rk
        if L in (16, 28, 40):
            diag.append({"layer": L, "norm_delta": round(float(torch.linalg.norm(delta[L])), 2),
                         "norm_dJ": round(float(torch.linalg.norm(dJ)), 2),
                         "norm_dperp": round(float(torch.linalg.norm(dperp)), 2),
                         "norm_residual": round(float(torch.linalg.norm(mloc[L])), 2)})

    conds = ["full", "dJ", "dperp", "rand_row", "rand_ker", "noop"]
    dist = {k: Counter_like() for k in ["baseline"] + [f"{c}@{a}" for a in alphas for c in conds]}
    per_trial = []
    # force the agent to the command-choice point so the NEXT tokens are the action itself
    # (tightest test: inject the concept right where the command is chosen, read it immediately).
    preamble = tok.encode("</think>\n\n<tool_call>\n<function=run_shell_command>\n<parameter=command>\n", add_special_tokens=False)
    gen_tokens = 90 if not force_cmd else 18

    for p in test:
        ids0 = p["token_ids"]
        ids = ids0 + preamble if force_cmd else ids0
        text = tok.decode(ids)
        base = classify(_P.generate_text(llm, ids, text, max_tokens=gen_tokens))
        dist["baseline"].add(base)
        row = {"task": p["metadata"]["task"], "turn": p["metadata"]["turn"], "baseline": base, "res": {}}
        for a in alphas:
            for c in conds:
                injd = {} if c == "noop" else {L: [(-1, (a * bdir[c][L]).to(torch.bfloat16))] for L in band}
                act = classify(_P.generate_text(llm, ids, text, max_tokens=gen_tokens, inject=injd))
                dist[f"{c}@{a}"].add(act)
                row["res"][f"{c}@{a}"] = act
        per_trial.append(row)

    def rates(k):
        d = dist[k].c
        n = max(1, sum(d.values()))
        return {"dist": d, "modify": round(d["modify"] / n, 3), "locate": round(d["locate"] / n, 3),
                "valid": round((d["locate"] + d["modify"]) / n, 3)}

    return {
        "kind": "swe_state_action_swap", "band": [band[0], band[-1]], "alphas": alphas, "force_cmd": force_cmd,
        "n_train_locate": len(loc_tr), "n_train_modify": len(mod_tr), "n_test": len(test),
        "norm_diag": diag, "summary": {k: rates(k) for k in dist}, "per_trial": per_trial,
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
    ap.add_argument("--alphas", type=float, nargs="+", default=[3.0, 8.0, 20.0])
    ap.add_argument("--force-cmd", action="store_true", help="inject at the command-choice point; read the immediate command")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/swe-state-action-swap.json")
    args = ap.parse_args()

    llm = _P.load_llm(max_model_len=32768, gpu_mem=0.82)
    band = list(range(args.band[0], args.band[1] + 1))
    result = run(llm, band=band, n_train=args.n_train, n_test=args.n_test, alphas=args.alphas, force_cmd=args.force_cmd)
    trials = result.pop("per_trial")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    (args.out.parent / (args.out.stem + "-trials.json")).write_text(json.dumps(trials))
    s = result["summary"]
    print(f"n_test={result['n_test']} band={result['band']} force_cmd={result['force_cmd']} alphas={result['alphas']}")
    print("norm diag:", result["norm_diag"])
    print(f"  baseline: modify={s['baseline']['modify']} locate={s['baseline']['locate']} valid={s['baseline']['valid']}  {s['baseline']['dist']}")
    print(f"{'key':18s} {'modify':>7s} {'locate':>7s} {'valid':>7s}")
    for a in result["alphas"]:
        for c in ["full", "dJ", "dperp", "rand_row", "rand_ker", "noop"]:
            k = f"{c}@{a}"
            print(f"  {k:18s} {s[k]['modify']:>7.3f} {s[k]['locate']:>7.3f} {s[k]['valid']:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
