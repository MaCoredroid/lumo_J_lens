#!/usr/bin/env python3
"""Characterize the LDA-django steering hint: real task-tuned steering, or overfitting?

The decisive-steering test found one positive: the shrinkage-LDA J-space direction shifted the
action margin strongly toward `sed` -- but only on django test turns, not astropy. Two prongs
settle whether that is real steering or an overfit artifact of fitting a 5120-dim covariance from
~60 samples:

(1) LEAVE-DJANGO-OUT: train the LDA direction on train tasks EXCLUDING django, test on django. If
    the effect persists -> a genuine, generalizing modify direction; if it vanishes -> django-family
    memorization.
(2) OUTPUT VALIDITY: generate the actual command under LDA-J on django turns. A valid `sed -i`/`edit`
    -> real action steering; garbage -> a margin artifact, null airtight.

Builds CAA_all, LDA_all, LDA_nodjango; on django AND astropy held-out locate turns, generates the
command at the effect scale and classifies modify/locate/garbage. Loads model once; memory-safe."""

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


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_P = _load("run_jlens_patch", "scripts/run_jlens_patch.py")
_JD = _load("jspace_decompose", "scripts/jspace_decompose.py")


def classify_cmd(text: str) -> str:
    """Classify the generated command: modify (edits) vs locate (reads) vs garbage/neutral."""
    t = text.lower()
    mod = any(k in t for k in ("sed -i", "edit", "write_file", " > ", ">>", "tee ", "apply_patch", "insert", "cat >"))
    loc = any(k in t for k in ("grep", "find ", " ls", "cat ", "head", "sed -n", "less ", "rg ", " read"))
    if mod and not loc:
        return "modify"
    if loc and not mod:
        return "locate"
    if mod and loc:
        return "both"
    return "garbage/neutral"


def _lda(Rloc, Rmod):
    import numpy as np
    import torch
    mu = Rmod.mean(0) - Rloc.mean(0)
    Xc = np.concatenate([Rloc - Rloc.mean(0), Rmod - Rmod.mean(0)]).astype(np.float64)
    Sig = (Xc.T @ Xc) / max(1, Xc.shape[0] - 2)
    lam = 0.5 * np.trace(Sig) / Sig.shape[0]
    w = np.linalg.solve(Sig + lam * np.eye(Sig.shape[0]), mu.astype(np.float64))
    caa = torch.tensor(mu, dtype=torch.float32)
    lda = torch.tensor(w.astype("float32"))
    lda = lda / torch.linalg.norm(lda) * torch.linalg.norm(caa)
    return caa, lda


def run(llm, *, band, n_train, alpha) -> dict[str, Any]:
    import numpy as np
    import torch

    tok = llm.get_tokenizer()
    prompts = json.loads(BOUNDED.read_text())
    by_task = defaultdict(list)
    for p in prompts:
        t = p["metadata"]["tag"]
        g = "locate" if t in LOCATE else ("modify" if t in MODIFY else None)
        if g:
            by_task[p["metadata"]["task"]].append({**p, "grp": g})
    tasks = sorted(by_task)
    train_tasks = {tasks[i] for i in range(0, len(tasks), 2)}
    train = [p for t in tasks if t in train_tasks for p in by_task[t]]
    loc_tr = [p for p in train if p["grp"] == "locate"][:n_train]
    mod_tr = [p for p in train if p["grp"] == "modify"][:n_train]
    dj_test = [p for t in tasks if t not in train_tasks and t.startswith("django") for p in by_task[t] if p["grp"] == "locate"][:6]
    as_test = [p for t in tasks if t not in train_tasks and t.startswith("astropy") for p in by_task[t] if p["grp"] == "locate"][:6]

    def capture(turns):
        acc = {L: [] for L in band}
        djflag = []
        for p in turns:
            h = _P.capture_all_layers(llm, p["token_ids"], tok.decode(p["token_ids"]))
            for L in band:
                acc[L].append(h[L][-1].float().numpy())
            djflag.append(p["metadata"]["task"].startswith("django"))
        return {L: np.stack(acc[L]) for L in band}, np.array(djflag)

    Rloc, djloc = capture(loc_tr)
    Rmod, djmod = capture(mod_tr)

    projectors = {}
    for L in band:
        J_L = _JD.load_J(L, device="cuda")
        projectors[L], _, _ = _JD.row_space_projector(J_L)
        projectors[L] = projectors[L].cpu()
        del J_L
        torch.cuda.empty_cache()

    # per-layer J-space directions for: CAA(all), LDA(all), LDA(no-django)
    dirs = {"caa_all_dJ": {}, "lda_all_dJ": {}, "lda_nodj_dJ": {}}
    for L in band:
        P_J = projectors[L]
        caa_all, lda_all = _lda(Rloc[L], Rmod[L])
        _, lda_nodj = _lda(Rloc[L][~djloc], Rmod[L][~djmod])
        # J-space part, norm-matched (against its own perp) so magnitudes are comparable
        dirs["caa_all_dJ"][L] = _JD.norm_match(*_JD.decompose(caa_all, P_J))[0]
        dirs["lda_all_dJ"][L] = _JD.norm_match(*_JD.decompose(lda_all, P_J))[0]
        dirs["lda_nodj_dJ"][L] = _JD.norm_match(*_JD.decompose(lda_nodj, P_J))[0]

    conds = ["baseline", "caa_all_dJ", "lda_all_dJ", "lda_nodj_dJ"]
    out = {"kind": "lda_characterize", "alpha": alpha, "n_train": len(loc_tr),
           "n_django_train_loc": int(djloc.sum()), "n_django_train_mod": int(djmod.sum()),
           "by_group": {}, "trials": []}
    for label, test in (("django", dj_test), ("astropy", as_test)):
        dist = {c: Counter_like() for c in conds}
        for p in test:
            ids = p["token_ids"] + tok.encode(CMD_PREAMBLE, add_special_tokens=False)
            text = tok.decode(ids)
            row = {"group": label, "task": p["metadata"]["task"], "turn": p["metadata"]["turn"], "cmd": {}}
            for c in conds:
                injd = None if c == "baseline" else {L: [(-1, (alpha * dirs[c][L]).to(torch.bfloat16))] for L in band}
                gen = _P.generate_text(llm, ids, text, max_tokens=20, inject=injd)
                cls = classify_cmd(gen)
                dist[c].add(cls)
                row["cmd"][c] = {"class": cls, "gen": gen[:70].replace("\n", " ")}
            out["trials"].append(row)
        out["by_group"][label] = {c: dist[c].c for c in conds}
    return out


class Counter_like:
    def __init__(self):
        self.c = {"locate": 0, "modify": 0, "both": 0, "garbage/neutral": 0}

    def add(self, k):
        self.c[k] = self.c.get(k, 0) + 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--band", type=int, nargs=2, default=[16, 40])
    ap.add_argument("--n-train", type=int, default=36)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/swe-lda-characterize.json")
    args = ap.parse_args()

    llm = _P.load_llm(max_model_len=32768, gpu_mem=0.82)
    band = list(range(args.band[0], args.band[1] + 1))
    r = run(llm, band=band, n_train=args.n_train, alpha=args.alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"alpha={r['alpha']} n_train={r['n_train']} (django in train: loc={r['n_django_train_loc']} mod={r['n_django_train_mod']})")
    for grp in ("django", "astropy"):
        print(f"\n=== {grp} test turns: command class distribution ===")
        for c in ["baseline", "caa_all_dJ", "lda_all_dJ", "lda_nodj_dJ"]:
            print(f"  {c:14s} {r['by_group'][grp][c]}")
    print("\n--- sample generated commands (django) ---")
    for row in r["trials"]:
        if row["group"] == "django":
            print(f"  {row['task'][:16]}-{row['turn']}: base={row['cmd']['baseline']['gen']!r}")
            print(f"       lda_all={row['cmd']['lda_all_dJ']['gen']!r}  lda_nodj={row['cmd']['lda_nodj_dJ']['gen']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
