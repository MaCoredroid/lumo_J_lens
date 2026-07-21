#!/usr/bin/env python3
"""Lever toward per-family trust: DATA-DRIVEN discriminative concept forms (v3). (P7c)

Instead of hand-guessing surface forms, derive per-family single-token forms from the actual
thinking texts of the turns tagged that family: tokens that are over-represented in a family's
CoT vs every other family (log-odds ratio with add-1 smoothing + a min-count floor), filtered
to clean content-word tokens and made globally unique (each token assigned to its argmax family).
This is the general analog of the single-task probe that reached 0.60 -- forms matched to what
Qwen actually writes when reasoning in each concept. Emits a broad candidate pool (top-K/family)
to capture once, so per-family selection + a train/test split can be done post-hoc from the
readout. Uses the tokenizer only.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAGS = ROOT / ".cache/swe_jlens_cohort/cohort_cot_tags.json"
OUTPUT = ROOT / "configs/swe_task_state_v4_datadriven_concept_forms_v3.json"
TOP_K = 16          # candidate forms per family in the pool
MIN_COUNT = 4       # a token must appear >= this many times in a family to qualify

_STOP = set(
    "the a an is are was were be been being to of in on at for and or but if then so this that "
    "these those it its as by with from into out up down we i you he she they them us our your "
    "will would can could should may might must do does did done has have had not no yes just now "
    "here there what which who whom whose when where why how all any both each few more most other "
    "some such only own same than too very s t re ve ll d m o also let me my need see look also "
    "will need going want know think first next now use using used one two get got make made".split()
)
_WORD = re.compile(r"^[A-Za-z][A-Za-z]{2,}$")  # clean alphabetic content word, >=3 chars


def _tokenizer():
    from transformers import AutoTokenizer

    snap = next(Path.home().glob(".cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/*"))
    return AutoTokenizer.from_pretrained(str(snap), trust_remote_code=False)


def _ct():
    spec = importlib.util.spec_from_file_location("cohort_traces", ROOT / "scripts/swe_task_state_v4_cohort_traces.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _family_corpora(train_tasks: set[str] | None = None) -> dict[str, list[tuple[str, str]]]:
    """family -> list of (thinking text, task) per tagged turn, optionally restricted to train tasks."""
    ct = _ct()
    tags = json.loads(TAGS.read_text())["tasks"]
    survey = {v["task"]: v for v in ct.survey()["usable"]}
    corpora: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for t in tags:
        task = t["task"]
        if train_tasks is not None and task not in train_tasks:
            continue
        if task not in survey:
            continue
        thinks = ct.task_thinking_blocks(Path(survey[task]["trace"]))
        for turn in t["turns"]:
            i = turn["turn"] - 1
            if turn["tag"] and turn["tag"] != "none" and i < len(thinks):
                corpora[turn["tag"]].append((thinks[i], task))
    return corpora


def derive(tokenizer, corpora: dict[str, list[tuple[str, str]]], *, top_k: int = TOP_K, min_tasks: int = 2) -> dict[str, Any]:
    # per-family token doc-counts (once per turn) + the set of distinct tasks each token appears in
    fam_counts: dict[str, Counter] = {}
    fam_tasks: dict[str, dict[int, set]] = {}
    for fam, docs in corpora.items():
        c: Counter = Counter()
        tset: dict[int, set] = defaultdict(set)
        for text, task in docs:
            ids = set(tokenizer.encode(text, add_special_tokens=False))
            for tid in ids:
                dec = tokenizer.decode([tid]).strip()
                if _WORD.match(dec) and dec.lower() not in _STOP:
                    c[tid] += 1
                    tset[tid].add(task)
        fam_counts[fam] = c
        fam_tasks[fam] = tset

    total = Counter()
    for c in fam_counts.values():
        total.update(c)
    n_docs = {fam: len(docs) for fam, docs in corpora.items()}
    grand = sum(n_docs.values())

    # log-odds (add-1) of doc-frequency: token distinctive to a family vs the rest
    scored: dict[str, list[tuple[int, float]]] = {}
    for fam, c in fam_counts.items():
        nf = max(1, n_docs[fam])
        rows = []
        for tid, k in c.items():
            if k < MIN_COUNT or len(fam_tasks[fam][tid]) < min_tasks:
                continue  # require the token to recur across >= min_tasks tasks (kill task-identity leakage)
            rest_k = total[tid] - k
            rest_n = max(1, grand - nf)
            lo = math.log((k + 1) / (nf - k + 1)) - math.log((rest_k + 1) / (rest_n - rest_k + 1))
            rows.append((tid, lo))
        rows.sort(key=lambda r: -r[1])
        scored[fam] = rows

    # global uniqueness: assign each token to its argmax family
    best_fam: dict[int, tuple[str, float]] = {}
    for fam, rows in scored.items():
        for tid, lo in rows:
            if tid not in best_fam or lo > best_fam[tid][1]:
                best_fam[tid] = (fam, lo)

    families: dict[str, Any] = {}
    for fam, rows in scored.items():
        forms = []
        for tid, lo in rows:
            if best_fam[tid][0] != fam:
                continue
            forms.append({"form": tokenizer.decode([tid]).strip(), "token_id": tid, "log_odds": round(lo, 3)})
            if len(forms) >= top_k:
                break
        if forms:
            families[fam] = forms
    return {
        "kind": "swe_task_state_v4_datadriven_concept_forms_v3",
        "top_k": top_k,
        "min_count": MIN_COUNT,
        "n_families": len(families),
        "families": families,
    }


def main() -> int:
    tokenizer = _tokenizer()
    result = derive(tokenizer, _family_corpora())
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    n = sum(len(f) for f in result["families"].values())
    print(f"data-driven v3: {result['n_families']} families, {n} forms")
    for fam, forms in result["families"].items():
        print(f"  {fam:24s} ({len(forms):2d}) {[f['form'] for f in forms[:12]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
