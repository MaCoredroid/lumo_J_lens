#!/usr/bin/env python3
"""Materialize every probeable SWE request through the historical CLI."""

from __future__ import annotations

from contextlib import contextmanager
import sys
from typing import Any, Iterator, Mapping, Sequence

import materialize_swe_behavioral_probes as historical


ALL_PROBEABLE_FLAG = "--all-probeable"


def _strip_required_all_probeable(argv: Sequence[str]) -> list[str]:
    delegated = list(argv)
    matches = [
        index
        for index, value in enumerate(delegated)
        if value == ALL_PROBEABLE_FLAG
    ]
    if len(matches) != 1:
        raise SystemExit(
            "dense materialization requires exactly one --all-probeable flag"
        )
    delegated.pop(matches[0])
    return delegated


@contextmanager
def _dense_materializer_patch() -> Iterator[None]:
    original_selector = historical.select_probeable_requests
    original_max_checkpoints = historical.MAX_CHECKPOINTS

    def select_all_probeable_requests(
        task: Mapping[str, Any], *, max_prompt_tokens: int, **_: Any
    ) -> dict[str, Any]:
        captures = task.get("captures")
        capture_count = len(captures) if isinstance(captures, list) else 0
        return original_selector(
            task,
            max_prompt_tokens=max_prompt_tokens,
            limit=max(1, capture_count),
        )

    historical.select_probeable_requests = select_all_probeable_requests
    historical.MAX_CHECKPOINTS = None
    try:
        yield
    finally:
        historical.select_probeable_requests = original_selector
        historical.MAX_CHECKPOINTS = original_max_checkpoints


def main(argv: Sequence[str] | None = None) -> int:
    delegated = _strip_required_all_probeable(sys.argv[1:] if argv is None else argv)
    with _dense_materializer_patch():
        return historical.main(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
