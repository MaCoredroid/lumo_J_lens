#!/usr/bin/env python3
"""Read-only reader for the FREE reasoning context in a Qwen J-lens trajectory.

The certified teacher-forced trajectory
(``.cache/swe_jlens_trajectory/swe_jlens_trajectory_prompts.json``) already
carries, per lens boundary, the Qwen agent's own observable reasoning structure:
its stage, the completion region (`reasoning` / `think_close` / `visible_text` /
`tool_call_*` / `terminal`), and self-labeled semantic events
(`diagnosis_named`, `bug_recognized`, `patch_target_named`, `fix_working`,
`task_resolved`, ...). This module surfaces that timeline for the Qwen-only lens
report — it is EVALUATION/OBSERVATION-ONLY and must never feed the predictor's
features. It performs no capture, no model call, and no tokenizer decode.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY = (
    ROOT / ".cache/swe_jlens_trajectory/swe_jlens_trajectory_prompts.json"
)

# Events that mark completion structure rather than reasoning content.
_STRUCTURAL_SUFFIXES = ("_start", "_end", "_boundary")
_STRUCTURAL_EXACT = frozenset({"prompt_boundary", "pre_eos"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def is_semantic_event(event: str) -> bool:
    """A reasoning-content event, not a structural completion marker."""
    if event in _STRUCTURAL_EXACT:
        return False
    return not event.endswith(_STRUCTURAL_SUFFIXES)


def _boundary_record(entry: Mapping[str, Any]) -> dict[str, Any]:
    meta = entry["metadata"]
    traj = meta["trajectory"]
    events = list(traj.get("events", []))
    return {
        "id": entry["id"],
        "turn": int(meta["request_index"]),
        "stage": meta["stage_name"],
        "region": traj.get("region"),
        "offset": int(traj.get("offset", traj.get("completion_token_offset", 0))),
        "events": events,
        "semantic_events": [e for e in events if is_semantic_event(e)],
        "target_token_text": traj.get("target_token_text"),
    }


def load_trajectory_boundaries(
    path: Path = DEFAULT_TRAJECTORY,
) -> list[dict[str, Any]]:
    """Per-boundary free reasoning records, in file order."""
    data = json.loads(Path(path).read_text())
    _require(isinstance(data, list) and bool(data), "trajectory must be a non-empty list")
    records = [_boundary_record(entry) for entry in data]
    _require(
        all(r["region"] is not None for r in records),
        "every boundary must carry a region",
    )
    return records


def summarize_turns(
    boundaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse boundaries into one ordered record per agent turn."""
    by_turn: dict[int, list[Mapping[str, Any]]] = {}
    for b in boundaries:
        by_turn.setdefault(b["turn"], []).append(b)
    turns = []
    for turn in sorted(by_turn):
        rows = by_turn[turn]
        semantic: list[str] = []
        for b in rows:  # preserve first-seen order along the completion
            for e in b["semantic_events"]:
                if e not in semantic:
                    semantic.append(e)
        regions = Counter(b["region"] for b in rows)
        turns.append(
            {
                "turn": turn,
                "stage": rows[0]["stage"],
                "n_boundaries": len(rows),
                "regions": dict(regions),
                "semantic_events": semantic,
                "has_reasoning": regions.get("reasoning", 0) > 0,
                "has_tool_call": any(str(r).startswith("tool_call") for r in regions),
            }
        )
    return turns


def read_free_reasoning_context(
    path: Path = DEFAULT_TRAJECTORY,
) -> dict[str, Any]:
    """Full free reasoning context: per-boundary records + per-turn timeline."""
    boundaries = load_trajectory_boundaries(path)
    turns = summarize_turns(boundaries)
    return {
        "trajectory_path": str(path),
        "n_boundaries": len(boundaries),
        "n_turns": len(turns),
        "boundaries": boundaries,
        "turns": turns,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    args = parser.parse_args(argv)
    ctx = read_free_reasoning_context(args.trajectory)
    print(f"boundaries={ctx['n_boundaries']} turns={ctx['n_turns']}")
    for t in ctx["turns"]:
        chain = " -> ".join(t["semantic_events"]) or "(none)"
        print(f"  turn {t['turn']:>2} [{t['stage']}] events: {chain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
