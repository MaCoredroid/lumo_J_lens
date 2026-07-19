#!/usr/bin/env python3
"""Focused tests for the all-probeable materializer adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from typing import Mapping
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "materialize_swe_dense_behavioral_probes",
    ROOT / "scripts/materialize_swe_dense_behavioral_probes.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DenseMaterializerTests(unittest.TestCase):
    def test_requires_and_strips_exactly_one_all_probeable_flag(self) -> None:
        self.assertEqual(
            MODULE._strip_required_all_probeable(
                ["--run-root", "run", "--all-probeable", "--output", "out"]
            ),
            ["--run-root", "run", "--output", "out"],
        )
        for argv in (
            ["--run-root", "run"],
            ["--all-probeable", "--all-probeable"],
        ):
            with self.subTest(argv=argv), self.assertRaisesRegex(
                SystemExit, "requires exactly one --all-probeable"
            ):
                MODULE._strip_required_all_probeable(argv)

    def test_fake_build_selects_all_and_restores_historical_defaults(self) -> None:
        fake = types.SimpleNamespace(MAX_CHECKPOINTS=8)

        def select_probeable_requests(
            task: Mapping[str, list[object]],
            *,
            max_prompt_tokens: int,
            limit: int = 8,
        ) -> dict[str, object]:
            del max_prompt_tokens
            indices = list(range(1, len(task["captures"]) + 1))
            return {"selected_request_indices": indices[:limit]}

        fake.select_probeable_requests = select_probeable_requests

        def build() -> tuple[dict[str, object], dict[str, object]]:
            selected = fake.select_probeable_requests(
                {"captures": [{} for _ in range(11)]}, max_prompt_tokens=100
            )["selected_request_indices"]
            return (
                {
                    "selected_request_indices": selected,
                    "max_checkpoints": fake.MAX_CHECKPOINTS,
                },
                {"max_checkpoints_per_task": fake.MAX_CHECKPOINTS},
            )

        default_prompt, default_summary = build()
        self.assertEqual(default_prompt["selected_request_indices"], list(range(1, 9)))
        self.assertEqual(default_prompt["max_checkpoints"], 8)
        self.assertEqual(default_summary["max_checkpoints_per_task"], 8)

        with mock.patch.object(MODULE, "historical", fake):
            with MODULE._dense_materializer_patch():
                dense_prompt, dense_summary = build()
                self.assertEqual(
                    dense_prompt["selected_request_indices"], list(range(1, 12))
                )
                self.assertIsNone(dense_prompt["max_checkpoints"])
                self.assertIsNone(dense_summary["max_checkpoints_per_task"])

        restored_prompt, restored_summary = build()
        self.assertEqual(restored_prompt, default_prompt)
        self.assertEqual(restored_summary, default_summary)
        self.assertEqual(fake.MAX_CHECKPOINTS, 8)
        self.assertIs(fake.select_probeable_requests, select_probeable_requests)

    def test_main_delegates_without_dense_flag_and_restores_on_failure(self) -> None:
        calls: list[list[str]] = []
        fake = types.SimpleNamespace(MAX_CHECKPOINTS=8)

        def select_probeable_requests(
            task: Mapping[str, list[object]],
            *,
            max_prompt_tokens: int,
            limit: int = 8,
        ) -> dict[str, object]:
            del task, max_prompt_tokens, limit
            return {}

        def delegated_main(argv: list[str]) -> int:
            calls.append(argv)
            self.assertIsNone(fake.MAX_CHECKPOINTS)
            raise RuntimeError("fixture failure")

        fake.select_probeable_requests = select_probeable_requests
        fake.main = delegated_main
        with mock.patch.object(MODULE, "historical", fake):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                MODULE.main(
                    ["--all-probeable", "--run-root", "run", "--output", "out"]
                )

        self.assertEqual(calls, [["--run-root", "run", "--output", "out"]])
        self.assertEqual(fake.MAX_CHECKPOINTS, 8)
        self.assertIs(fake.select_probeable_requests, select_probeable_requests)


if __name__ == "__main__":
    unittest.main()
