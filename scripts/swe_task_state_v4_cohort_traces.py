#!/usr/bin/env python3
"""Robust reader over the heterogeneous cohort qwen_trace.json corpus.

The per-task agent traces under runs/*/generation/verified/per_task/<task>/ come
in mixed formats (JSON list vs JSONL) across ~17 runs, and some files are empty or
partial. This module reads any of them, extracts the per-turn CoT `thinking` text,
and lists the tasks with usable traces — the real input set for cohort-scale
faithfulness. Read-only; no model, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def read_trace_entries(path: Path) -> list[dict[str, Any]]:
    """Parse a trace as a JSON list or JSONL; return its entries ([] if unusable)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return [e for e in value if isinstance(e, dict)]
        if isinstance(value, dict):
            return [value]
    except json.JSONDecodeError:
        pass
    entries: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def _thinking_from_content(content: Any) -> Iterator[str]:
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "thinking" and part.get("thinking", "").strip():
                    yield part["thinking"].strip()
                # some formats carry reasoning under 'reasoning'/'reasoning_content'
                for key in ("reasoning", "reasoning_content"):
                    val = part.get(key)
                    if isinstance(val, str) and val.strip():
                        yield val.strip()


def task_thinking_blocks(path: Path) -> list[str]:
    """Per-turn CoT thinking text from a trace, in order."""
    blocks: list[str] = []
    for entry in read_trace_entries(path):
        msg = entry.get("message", entry)
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant" or entry.get("type") == "assistant":
            for text in _thinking_from_content(msg.get("content")):
                blocks.append(text)
        for key in ("reasoning", "reasoning_content"):
            val = msg.get(key)
            if isinstance(val, str) and val.strip():
                blocks.append(val.strip())
    return blocks


def _task_name(trace: Path) -> str:
    # runs/<run>/generation/verified/per_task/<task>/qwen_trace.json
    return trace.parent.name


def list_traces() -> list[Path]:
    return sorted(RUNS.glob("*/generation/verified/per_task/*/qwen_trace.json"))


def survey() -> dict[str, Any]:
    """Which tasks have a usable trace (non-empty, parseable, >=1 thinking block)."""
    by_task: dict[str, dict[str, Any]] = {}
    for trace in list_traces():
        task = _task_name(trace)
        n = len(task_thinking_blocks(trace))
        prev = by_task.get(task)
        # keep the richest trace per task across runs
        if prev is None or n > prev["n_turns"]:
            by_task[task] = {"task": task, "trace": str(trace), "n_turns": n}
    usable = {t: v for t, v in by_task.items() if v["n_turns"] > 0}
    return {
        "n_trace_files": len(list_traces()),
        "n_distinct_tasks": len(by_task),
        "n_usable_tasks": len(usable),
        "total_usable_turns": sum(v["n_turns"] for v in usable.values()),
        "usable": sorted(usable.values(), key=lambda v: v["task"]),
    }


def main() -> int:
    s = survey()
    print(
        f"trace files={s['n_trace_files']} distinct tasks={s['n_distinct_tasks']} "
        f"usable tasks={s['n_usable_tasks']} usable turns={s['total_usable_turns']}"
    )
    for v in s["usable"][:12]:
        print(f"  {v['task']:32s} turns={v['n_turns']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
