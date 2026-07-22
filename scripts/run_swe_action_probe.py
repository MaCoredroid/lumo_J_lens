#!/usr/bin/env python3
"""Diagnose the measurable action space: what tool/command tokens the model considers at the
decision point, for locate vs modify turns. Prints top next-token logprobs at the <function= and
<parameter=command> positions, so we can pick a clean, well-populated behavioral contrast + the
exact tokens to measure for the decisive steering test. Loads model once."""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOUNDED = ROOT / ".cache/swe_jlens_cohort/cohort-perturn-prompts-bounded.json"
LOCATE = {"source_localization", "located_source", "defined_identifier"}
MODIFY = {"source_edit", "repair", "substitution_operation"}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_P = _load("run_jlens_patch", "scripts/run_jlens_patch.py")

FUNC = "</think>\n\n<tool_call>\n<function="
CMD = "</think>\n\n<tool_call>\n<function=run_shell_command>\n<parameter=command>\n"


def main() -> int:
    llm = _P.load_llm(max_model_len=32768, gpu_mem=0.82)
    tok = llm.get_tokenizer()
    prompts = json.loads(BOUNDED.read_text())
    by = defaultdict(list)
    for p in prompts:
        t = p["metadata"]["tag"]
        g = "locate" if t in LOCATE else ("modify" if t in MODIFY else None)
        if g:
            by[g].append(p)

    def show(p, suffix, label):
        ids = p["token_ids"] + tok.encode(suffix, add_special_tokens=False)
        lp, g = _P.next_logprobs(llm, ids, tok.decode(ids), top_k=14)
        top = sorted(lp.items(), key=lambda kv: -kv[1])[:12]
        toks = "  ".join(f"{tok.decode([tid])!r}:{v:.1f}" for tid, v in top)
        print(f"    [{label}] greedy={tok.decode([g])!r} | {toks}")

    for grp in ("locate", "modify"):
        print(f"\n===== {grp} turns =====")
        for p in by[grp][:4]:
            print(f"  {p['metadata']['task']} turn {p['metadata']['turn']}")
            show(p, FUNC, "function")
            show(p, CMD, "command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
