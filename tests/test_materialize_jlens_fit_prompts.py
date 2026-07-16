#!/usr/bin/env python3
"""Tests for deterministic Jacobian Lens corpus materialization."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_jlens_fit_prompts",
    ROOT / "scripts" / "materialize_jlens_fit_prompts.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        max_length = int(kwargs["max_length"])
        return {"input_ids": list(range(len(text.split())))[:max_length]}


class PromptMaterializationTest(unittest.TestCase):
    def test_selects_first_rows_matching_both_gates(self) -> None:
        rows = [
            {"text": "too short"},
            {"text": "one two three four"},
            {"text": "one two three four five six"},
            {"text": "zero one two three four"},
        ]
        selected = MODULE.select_rows(
            rows,
            FakeTokenizer(),
            count=2,
            sequence_length=5,
            min_chars=10,
        )
        self.assertEqual([item["row_index"] for item in selected], [2, 3])
        self.assertEqual([item["token_ids"] for item in selected], [list(range(5))] * 2)
        self.assertEqual(
            selected[0]["text_sha256"], MODULE.sha256_text(selected[0]["text"])
        )

    def test_rejects_an_exhausted_dataset(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected 1"):
            MODULE.select_rows(
                [{"text": "short"}],
                FakeTokenizer(),
                count=1,
                sequence_length=5,
                min_chars=100,
            )


if __name__ == "__main__":
    unittest.main()
