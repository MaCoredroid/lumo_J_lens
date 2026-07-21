#!/usr/bin/env python3
"""Tag every cohort CoT turn with the validated Qwen concept tagger (P7b).

Runs the auto-tagger over all usable cohort tasks (swe_task_state_v4_cohort_traces
survey), producing the per-turn CoT-concept tags — the "CoT side" of cohort-scale
faithfulness. Resumable: skips tasks already in the output. A few concurrent
requests keep the server (max_num_seqs=2) busy without overrunning it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".cache/swe_jlens_cohort/cohort_cot_tags.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ct = _load("swe_task_state_v4_cohort_traces")
tagger = _load("swe_task_state_v4_cot_concept_tagger")


def tag_task(task_row: dict[str, Any], *, workers: int = 2) -> dict[str, Any]:
    blocks = ct.task_thinking_blocks(Path(task_row["trace"]))

    def _one(item: tuple[int, str]) -> dict[str, Any]:
        i, text = item
        try:
            tag = tagger.tag_cot_text(text)
        except Exception as error:  # noqa: BLE001 - record, don't abort the cohort
            tag = f"error:{type(error).__name__}"
        return {"turn": i + 1, "tag": tag}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        turns = sorted(pool.map(_one, enumerate(blocks)), key=lambda r: r["turn"])
    return {"task": task_row["task"], "n_turns": len(turns), "turns": turns}


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, Any] = {}
    if args.output.exists():
        done = {r["task"]: r for r in json.loads(args.output.read_text())["tasks"]}

    usable = ct.survey()["usable"]
    if args.limit:
        usable = usable[: args.limit]
    for row in usable:
        if row["task"] in done:
            continue
        result = tag_task(row)
        done[row["task"]] = result
        payload = {"kind": "cohort_cot_tags_v1", "tasks": sorted(done.values(), key=lambda r: r["task"])}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        tags = [t["tag"] for t in result["turns"]]
        print(f"  {result['task']:32s} {result['n_turns']:3d} turns  e.g. {tags[:4]}", flush=True)

    total = sum(r["n_turns"] for r in done.values())
    print(f"DONE: {len(done)} tasks, {total} turns tagged -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
