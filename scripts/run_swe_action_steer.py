#!/usr/bin/env python3
"""Decisive test: does the J-space concept component STEER the SWE agent's action? (logit-level)

Fixes the two weak links of the earlier null: (1) a TRAINED steering vector (contrastive
difference-of-means [CAA] and a logistic residual direction), not just a raw mean difference;
(2) a SENSITIVE logit-level metric -- the action-propensity margin logprob(grep)-logprob(sed) at
the command-choice point -- instead of coarse free-generation flip counting (which garbled).

Build the locate->modify steering vector from tagged turns' end-of-thinking residuals, split into
J-space vs non-J-space (norm-matched), inject across band L16-40 at held-out locate turns forced to
the command position, and measure how much each component shifts the action margin toward modify.
Random-in-row/ker controls separate direction-specific steering from generic disruption. If Delta_J
shifts the action margin toward modify more than norm-matched Delta_perp (and more than random),
the concept causally steers behavior; else the null holds. Loads model once. Memory-safe (chunked)."""

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
CMD_PREAMBLE = "</think>\n\n<tool_call>\n<function=run_shell_command>\n<parameter=command>\n"
LOC_CMD = {"grep", "find", "cat", "ls", "head", "rg", "wc", "less", "tail", "read"}
MOD_CMD = {"sed", "edit", "write", "tee", "patch", "insert", "replace", "cat>"}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_P = _load("run_jlens_patch", "scripts/run_jlens_patch.py")
_JD = _load("jspace_decompose", "scripts/jspace_decompose.py")


def _cmd_margin(lp: dict, tok) -> float:
    """max logprob over locate commands minus max over modify commands (high=locate, low=modify)."""
    loc, mod = -25.0, -25.0
    for tid, v in lp.items():
        w = tok.decode([tid]).strip().lower()
        if w in LOC_CMD:
            loc = max(loc, v)
        if w in MOD_CMD:
            mod = max(mod, v)
    return loc - mod


def _split(prompts):
    by_task = defaultdict(list)
    for p in prompts:
        t = p["metadata"]["tag"]
        g = "locate" if t in LOCATE else ("modify" if t in MODIFY else None)
        if g:
            by_task[p["metadata"]["task"]].append({**p, "grp": g})
    tasks = sorted(by_task)
    tr = {tasks[i] for i in range(0, len(tasks), 2)}
    train = [p for t in tasks if t in tr for p in by_task[t]]
    test = [p for t in tasks if t not in tr for p in by_task[t] if p["grp"] == "locate"]
    return train, test


def run(llm, *, band, n_train, n_test, alphas) -> dict[str, Any]:
    import numpy as np
    import torch

    tok = llm.get_tokenizer()
    prompts = json.loads(BOUNDED.read_text())
    train, test = _split(prompts)
    loc_tr = [p for p in train if p["grp"] == "locate"][:n_train]
    mod_tr = [p for p in train if p["grp"] == "modify"][:n_train]
    test = test[:n_test]

    # capture end-of-thinking residuals per band layer, per group
    def resids(turns):
        out = {L: [] for L in band}
        for p in turns:
            h = _P.capture_all_layers(llm, p["token_ids"], tok.decode(p["token_ids"]))
            for L in band:
                out[L].append(h[L][-1].float().numpy())
        return {L: np.stack(out[L]) for L in band}

    Rloc, Rmod = resids(loc_tr), resids(mod_tr)

    projectors = {}
    for L in band:
        J_L = _JD.load_J(L, device="cuda")
        projectors[L], _, _ = _JD.row_space_projector(J_L)
        projectors[L] = projectors[L].cpu()
        del J_L
        torch.cuda.empty_cache()

    # two trained steering vectors per layer: CAA (contrastive mean-diff) and shrinkage-LDA
    # (covariance-aware Fisher direction; robust for small n, pure numpy).
    bdir = {c: {} for c in ["caa_full", "caa_dJ", "caa_dperp", "lda_dJ", "lda_dperp", "rand_row", "rand_ker"]}
    for L in band:
        P_J = projectors[L]
        mu = Rmod[L].mean(0) - Rloc[L].mean(0)
        caa = torch.tensor(mu, dtype=torch.float32)
        Xc = np.concatenate([Rloc[L] - Rloc[L].mean(0), Rmod[L] - Rmod[L].mean(0)]).astype(np.float64)
        n = Xc.shape[0]
        Sig = (Xc.T @ Xc) / max(1, n - 2)                              # pooled within-class covariance
        lam = 0.5 * np.trace(Sig) / Sig.shape[0]                       # shrinkage toward isotropic
        w = np.linalg.solve(Sig + lam * np.eye(Sig.shape[0]), mu.astype(np.float64))
        lda = torch.tensor(w.astype("float32"))
        lda = lda / torch.linalg.norm(lda) * torch.linalg.norm(caa)   # scale to CAA norm

        cJ, cP = _JD.norm_match(*_JD.decompose(caa, P_J))
        lJ, lP = _JD.norm_match(*_JD.decompose(lda, P_J))
        rr, rk = _JD.random_in_subspaces(P_J, torch.linalg.norm(cJ), seed_vec=torch.flip(caa, dims=[0]))
        bdir["caa_full"][L] = caa
        bdir["caa_dJ"][L], bdir["caa_dperp"][L] = cJ, cP
        bdir["lda_dJ"][L], bdir["lda_dperp"][L] = lJ, lP
        bdir["rand_row"][L], bdir["rand_ker"][L] = rr, rk

    conds = list(bdir) + ["noop"]
    shifts = {f"{c}@{a}": [] for a in alphas for c in conds}
    per_trial = []
    for p in test:
        ids = p["token_ids"] + tok.encode(CMD_PREAMBLE, add_special_tokens=False)
        text = tok.decode(ids)
        base_lp, _ = _P.next_logprobs(llm, ids, text, top_k=20)
        base = _cmd_margin(base_lp, tok)
        row = {"task": p["metadata"]["task"], "turn": p["metadata"]["turn"], "base_margin": round(base, 2), "res": {}}
        for a in alphas:
            for c in conds:
                injd = {} if c == "noop" else {L: [(-1, (a * bdir[c][L]).to(torch.bfloat16))] for L in band}
                lp, _ = _P.next_logprobs(llm, ids, text, top_k=20, inject=injd)
                sh = _cmd_margin(lp, tok) - base       # negative = shifted toward modify
                shifts[f"{c}@{a}"].append(sh)
                row["res"][f"{c}@{a}"] = round(sh, 2)
        per_trial.append(row)

    def summ(k):
        v = shifts[k]
        return {"mean_shift": round(sum(v) / len(v), 3), "n": len(v)}

    return {
        "kind": "swe_action_steer", "band": [band[0], band[-1]], "alphas": alphas,
        "n_train": n_train, "n_test": len(test),
        "base_margin_mean": round(sum(r["base_margin"] for r in per_trial) / max(1, len(per_trial)), 3),
        "shift": {k: summ(k) for k in shifts}, "per_trial": per_trial,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--band", type=int, nargs=2, default=[16, 40])
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--alphas", type=float, nargs="+", default=[2.0, 6.0])
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/swe-action-steer.json")
    args = ap.parse_args()

    llm = _P.load_llm(max_model_len=32768, gpu_mem=0.82)
    band = list(range(args.band[0], args.band[1] + 1))
    r = run(llm, band=band, n_train=args.n_train, n_test=args.n_test, alphas=args.alphas)
    trials = r.pop("per_trial")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=2) + "\n")
    (args.out.parent / (args.out.stem + "-trials.json")).write_text(json.dumps(trials))
    print(f"n_test={r['n_test']} band={r['band']} alphas={r['alphas']} base_margin={r['base_margin_mean']}")
    print("(mean_shift < 0 = steered toward modify; compare dJ vs dperp vs random)")
    for a in r["alphas"]:
        print(f"--- alpha {a} ---")
        for c in ["caa_full", "caa_dJ", "caa_dperp", "lda_dJ", "lda_dperp", "rand_row", "rand_ker", "noop"]:
            print(f"  {c:12s} mean_shift={r['shift'][f'{c}@{a}']['mean_shift']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
