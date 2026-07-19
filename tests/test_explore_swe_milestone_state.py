#!/usr/bin/env python3
"""Focused tests for fail-closed future milestone labels."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "explore_swe_milestone_state",
    ROOT / "scripts/explore_swe_milestone_state.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def prompt(
    request_index: int,
    action: str | None,
    *,
    declared: list[int],
    task_id: str = "owner__repo-1",
) -> dict:
    return {
        "id": f"{task_id}-r{request_index}",
        "metadata": {
            "task": {
                "instance_id": task_id,
                "repo": "owner/repo",
                "probeable_request_indices": declared,
            },
            "selection": {"task_request_index": request_index},
            "labels": {
                "action": {
                    "status": "available" if action is not None else "missing",
                    "class_id": action,
                }
            },
        },
    }


class MilestoneLabelsTest(unittest.TestCase):
    def test_skips_inspections_until_known_milestone(self) -> None:
        declared = [1, 2, 3]
        prompts = [
            prompt(1, "inspect", declared=declared),
            prompt(2, "inspect", declared=declared),
            prompt(3, "edit", declared=declared),
        ]

        labels, summary = MODULE.milestone_labels(prompts)

        self.assertEqual(labels[prompts[0]["id"]]["label"], "edit")
        self.assertEqual(labels[prompts[0]["id"]]["horizon_requests"], 2)
        self.assertEqual(labels[prompts[0]["id"]]["target_request_index"], 3)
        self.assertEqual(labels[prompts[2]["id"]]["horizon_requests"], 0)
        self.assertTrue(summary["owner__repo-1"]["complete_consecutive_bundle"])

    def test_unknown_before_candidate_censors_fail_closed(self) -> None:
        declared = [1, 2, 3]
        prompts = [
            prompt(1, "inspect", declared=declared),
            prompt(2, None, declared=declared),
            prompt(3, "validate", declared=declared),
        ]

        labels, _ = MODULE.milestone_labels(prompts)

        for row in prompts[:2]:
            self.assertEqual(labels[row["id"]]["status"], "censored")
            self.assertEqual(
                labels[row["id"]]["reason"],
                "unknown_before_next_milestone",
            )
        self.assertEqual(labels[prompts[2]["id"]]["label"], "validate")

    def test_no_future_milestone_censors_trailing_inspections(self) -> None:
        declared = [1, 2, 3]
        prompts = [
            prompt(1, "edit", declared=declared),
            prompt(2, "inspect", declared=declared),
            prompt(3, "inspect", declared=declared),
        ]

        labels, _ = MODULE.milestone_labels(prompts)

        self.assertEqual(labels[prompts[0]["id"]]["label"], "edit")
        for row in prompts[1:]:
            self.assertEqual(labels[row["id"]]["status"], "censored")
            self.assertEqual(
                labels[row["id"]]["reason"],
                "no_observed_future_milestone",
            )

    def test_nonconsecutive_bundle_censors_every_checkpoint(self) -> None:
        declared = [1, 3]
        prompts = [
            prompt(1, "inspect", declared=declared),
            prompt(3, "finalize", declared=declared),
        ]

        labels, summary = MODULE.milestone_labels(prompts)

        self.assertFalse(summary["owner__repo-1"]["complete_consecutive_bundle"])
        for row in prompts:
            self.assertEqual(labels[row["id"]]["status"], "censored")
            self.assertEqual(
                labels[row["id"]]["reason"],
                "incomplete_nonconsecutive_bundle",
            )


class HierarchicalBootstrapTest(unittest.TestCase):
    def test_duplicate_cluster_draws_receive_distinct_bootstrap_ids(self) -> None:
        paired = []
        for repo in ("a/repo", "b/repo"):
            for task in ("task-1", "task-2"):
                row = {
                    "row_id": f"{repo}-{task}",
                    "repo": repo,
                    "task_id": task,
                    "label": "edit",
                    "prediction": "edit",
                }
                paired.append((dict(row), dict(row)))

        class DuplicateFirstClusterRng:
            @staticmethod
            def integers(_low: int, _high: int, size: int) -> np.ndarray:
                return np.zeros(size, dtype=np.int64)

        sampled = MODULE._hierarchical_sample_pairs(
            paired, DuplicateFirstClusterRng()
        )
        left_rows = [left for left, _ in sampled]

        self.assertEqual(len(left_rows), 4)
        self.assertEqual(len({row["repo"] for row in left_rows}), 2)
        self.assertEqual(len({row["task_id"] for row in left_rows}), 4)
        self.assertEqual({row["row_id"] for row in left_rows}, {"a/repo-task-1"})


if __name__ == "__main__":
    unittest.main()
