#!/usr/bin/env python3
"""Diagnose the two-hop data + patching method for the J-space swap (loads model once).

(1) Greedy-answer all 93 upstream two-hop items -> which does the model solve (both hops, single
token)? (2) On clean same-template pairs, test whether injecting the full concept delta flips the
answer under SINGLE-layer vs multi-layer BAND patching at the final position -- to see whether
downstream re-derivation from the explicit input overrides a single-layer patch.
"""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MULTIHOP = ROOT / ".cache/research/anthropics-jacobian-lens/data/evaluations/lens-eval-multihop.json"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_P = _load("run_jlens_patch", "scripts/run_jlens_patch.py")


def main() -> int:
    import torch

    llm = _P.load_llm()
    tok = llm.get_tokenizer()
    items = json.loads(MULTIHOP.read_text())["items"]

    clean = []
    for it in items:
        ids = tok.encode(it["prompt"], add_special_tokens=True)
        got = _P.greedy_token(llm, ids, it["prompt"])
        exp = {tok.encode(it["target"], add_special_tokens=False)[0],
               tok.encode(" " + it["target"], add_special_tokens=False)[0]}
        ok = got in exp
        rec = {"name": it["name"], "target": it["target"], "answer": tok.decode([got]), "ok": ok,
               "ids": ids, "answer_tid": got, "prompt": it["prompt"]}
        clean.append(rec)
    n_ok = sum(r["ok"] for r in clean)
    print(f"=== {n_ok}/{len(clean)} items solved correctly (both hops, single token) ===")
    for r in clean:
        if r["ok"]:
            print(f"  OK  {r['name']:26s} target={r['target']!r:12s} answer={r['answer']!r}")
    # families among the correct ones
    fam = defaultdict(list)
    for r in clean:
        if not r["ok"]:
            continue
        n = r["name"]
        key = "atomic_symbol" if (n.startswith("atomic-") and n.endswith("-symbol")) else \
              "nhop_element" if n.endswith("-element") else \
              "nhop_planet" if n.endswith("-planet") else "other"
        fam[key].append(r)
    print("\ncorrect-item families:", {k: len(v) for k, v in fam.items()})

    # method test: pick one clean same-template pair, test single-layer vs band full-delta flip
    pair = None
    for key, rs in fam.items():
        if key != "other" and len(rs) >= 2:
            distinct = [(a, b) for a in rs for b in rs if a["answer_tid"] != b["answer_tid"]]
            if distinct:
                pair = distinct[0]
                break
    if pair is None:
        print("\nNo clean same-template pair with distinct answers -> need better data.")
        return 0
    a, b = pair
    print(f"\n=== method test on pair P={a['name']}({a['answer']!r}) <- P'={b['name']}({b['answer']!r}) ===")
    # subject position = where the two prompts' tokens differ (the cue token); require same length
    if len(a["ids"]) != len(b["ids"]):
        print("pair token lengths differ; skipping subject-position test")
        return 0
    diff = [i for i, (x, y) in enumerate(zip(a["ids"], b["ids"], strict=True)) if x != y]
    subj_pos = diff[0] if diff else len(a["ids"]) - 1
    print(f"subject position (differing token) = {subj_pos} of {len(a['ids'])} "
          f"(P={tok.decode([a['ids'][subj_pos]])!r} P'={tok.decode([b['ids'][subj_pos]])!r})")
    tid_Cp = b["answer_tid"]

    def sweep(pos, label):
        h_p = _P.capture_all_layers(llm, a["ids"], a["prompt"], pos=pos)
        h_pp = _P.capture_all_layers(llm, b["ids"], b["prompt"], pos=pos)
        print(f"--- patching at {label} (pos {pos}) ---")
        for L in [8, 16, 24, 32, 40]:
            d = (h_pp[L][0] - h_p[L][0]).to("cuda", dtype=torch.bfloat16)
            t = _P.greedy_token(llm, a["ids"], a["prompt"], inject={L: d}, pos=pos)
            print(f"  single L{L:2d} -> {tok.decode([t])!r} flip={t == tid_Cp}")
        for lo, hi in [(16, 40), (8, 44), (24, 40)]:
            inj = {L: (h_pp[L][0] - h_p[L][0]).to("cuda", dtype=torch.bfloat16) for L in range(lo, hi + 1)}
            t = _P.greedy_token(llm, a["ids"], a["prompt"], inject=inj, pos=pos)
            print(f"  band L[{lo}-{hi}] -> {tok.decode([t])!r} flip={t == tid_Cp}")

    sweep(subj_pos, "SUBJECT position")
    sweep(-1, "FINAL position")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
