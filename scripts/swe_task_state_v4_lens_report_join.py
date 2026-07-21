#!/usr/bin/env python3
"""Join reasoning_trace latent rows with the free observed reasoning timeline.

Each `reasoning_trace` output row carries the latent Qwen-only indices
(diffuse-uncertainty entropy, ambivalence, source-disagreement) keyed by
`task_id` + `task_request_index`. This module attaches, ALONGSIDE those indices,
the same turn's observed stage and self-labeled semantic events from the
trajectory CoT reader (`swe_task_state_v4_trajectory_cot_reader`).

Pure data join, no model/tokenizer/sibling-import. EVALUATION/OBSERVATION-ONLY:
the attached observed reasoning is a report annotation and must never re-enter the
predictor's features.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


OBSERVED_KEY = "observed_reasoning"


def _turn_context_index(
    turns: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    for turn in turns:
        key = int(turn["turn"])
        if key in index:
            raise ValueError(f"duplicate turn {key} in timeline")
        index[key] = {
            "stage": turn["stage"],
            "semantic_events": list(turn.get("semantic_events", [])),
            "n_boundaries": turn.get("n_boundaries"),
        }
    return index


def _row_turn(row: Mapping[str, Any]) -> int | None:
    """The agent turn a trace row belongs to.

    `build_reasoning_trace` output nests it at ``boundary.request_index``; plain
    prediction dicts carry ``task_request_index`` at top level.
    """
    boundary = row.get("boundary")
    if isinstance(boundary, Mapping) and isinstance(
        boundary.get("request_index"), int
    ):
        return boundary["request_index"]
    turn = row.get("task_request_index")
    return turn if isinstance(turn, int) else None


def attach_reasoning_context(
    trace_rows: Sequence[Mapping[str, Any]],
    turns: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return trace rows each augmented with its turn's observed reasoning.

    A row whose turn has no matching timeline entry is annotated
    `status: unavailable_no_matching_turn` rather than dropped, so coverage stays
    explicit. Input rows are not mutated.
    """
    index = _turn_context_index(turns)
    merged: list[dict[str, Any]] = []
    for row in trace_rows:
        turn = _row_turn(row)
        context = index.get(turn) if turn is not None else None
        out = dict(row)
        if context is None:
            out[OBSERVED_KEY] = {
                "status": "unavailable_no_matching_turn",
                "task_request_index": turn,
            }
        else:
            out[OBSERVED_KEY] = {"status": "available", **context}
        merged.append(out)
    return merged


def coverage(merged_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How many rows got an observed-reasoning context attached."""
    available = sum(
        1
        for row in merged_rows
        if row.get(OBSERVED_KEY, {}).get("status") == "available"
    )
    return {
        "rows": len(merged_rows),
        "with_observed_reasoning": available,
        "unavailable": len(merged_rows) - available,
    }
