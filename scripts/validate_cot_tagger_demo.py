#!/usr/bin/env python3
"""Validate the Qwen CoT concept tagger against the hand-curated demo labels.

Gate before cohort-scale: the automated tagger must roughly reproduce the
hand-curated SEMANTIC_EVENTS concept assignments on the one demo task
(swe-sympy-13480). Extracts the 9 per-turn `thinking` blocks from the raw
qwen_trace.json, tags each via the served Qwen model, and reports agreement vs
the hand-curated per-turn acceptable-concept sets. Requires the Qwen server (9952).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE = (
    ROOT
    / "runs/publication_certified_v2_20260715/generation/verified/per_task"
    / "sympy__sympy-13480/qwen_trace.json"
)

_spec = importlib.util.spec_from_file_location(
    "cot_concept_tagger", ROOT / "scripts/swe_task_state_v4_cot_concept_tagger.py"
)
tagger = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tagger
_spec.loader.exec_module(tagger)

# Per-turn acceptable concepts from the hand-curated SEMANTIC_EVENTS -> family map.
EXPECTED: dict[int, set[str]] = {
    1: {"source_localization"},
    2: {"located_source", "source_localization"},
    3: {"defined_identifier", "runtime_name_failure", "failure_confirmation"},
    4: {"failure_confirmation", "source_edit", "repair"},
    5: {"repair", "source_edit", "verification"},
    6: {"broad_success", "repair", "verification"},
    7: {"broad_success", "verification"},
    8: {"dependency_unavailable"},
    9: {"test_success", "task_resolution", "focused_validation"},
}


def demo_thinking_blocks(trace_path: Path = TRACE) -> list[str]:
    trace = json.loads(Path(trace_path).read_text())
    blocks: list[str] = []
    for entry in trace:
        if not (isinstance(entry, dict) and entry.get("type") == "assistant"):
            continue
        for part in entry.get("message", {}).get("content", []):
            if isinstance(part, dict) and part.get("type") == "thinking":
                text = (part.get("thinking") or "").strip()
                if text:
                    blocks.append(text)
    return blocks


def validate(**kwargs: object) -> dict[str, object]:
    blocks = demo_thinking_blocks()
    rows = []
    hits = 0
    for i, text in enumerate(blocks[:9], start=1):
        tag = tagger.tag_cot_text(text, **kwargs)  # type: ignore[arg-type]
        acceptable = EXPECTED.get(i, set())
        hit = tag in acceptable
        hits += hit
        rows.append(
            {"turn": i, "tag": tag, "acceptable": sorted(acceptable), "hit": hit}
        )
    n = len(rows)
    return {
        "n_turns": n,
        "hits": hits,
        "agreement": (hits / n) if n else None,
        "rows": rows,
    }


def main() -> int:
    result = validate()
    for r in result["rows"]:
        print(f"  turn {r['turn']}: {r['tag']:22s} {'HIT' if r['hit'] else 'miss'}")
    print(f"tagger-vs-hand agreement: {result['hits']}/{result['n_turns']} = {result['agreement']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
