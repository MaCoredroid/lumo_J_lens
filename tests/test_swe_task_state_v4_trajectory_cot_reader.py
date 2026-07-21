from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/swe_task_state_v4_trajectory_cot_reader.py"

spec = importlib.util.spec_from_file_location("trajectory_cot_reader", MODULE_PATH)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class SemanticEventClassifierTests(unittest.TestCase):
    def test_structural_markers_are_not_semantic(self):
        for e in (
            "prompt_boundary",
            "pre_eos",
            "thinking_end",
            "think_close_end",
            "visible_text_end",
            "im_end_start",
            "tool_1_start",
            "tool_1_end",
            "tool_2_start",
            "correct_identifier_prediction_boundary",
        ):
            self.assertFalse(reader.is_semantic_event(e), e)

    def test_reasoning_events_are_semantic(self):
        for e in (
            "diagnosis_named",
            "bug_recognized",
            "correct_identifier_named",
            "patch_target_named",
            "fix_working",
            "original_reproduction_passed",
            "pytest_unavailable",
            "task_resolved",
        ):
            self.assertTrue(reader.is_semantic_event(e), e)


class TrajectoryReaderIntegrationTests(unittest.TestCase):
    def setUp(self):
        if not reader.DEFAULT_TRAJECTORY.exists():
            self.skipTest("captured trajectory not present")
        self.ctx = reader.read_free_reasoning_context()

    def test_shape_matches_captured_trajectory(self):
        self.assertEqual(self.ctx["n_turns"], 9)
        self.assertEqual(self.ctx["n_boundaries"], 293)
        self.assertEqual(len(self.ctx["boundaries"]), 293)

    def test_every_boundary_has_a_valid_region(self):
        valid = {
            "reasoning",
            "think_close",
            "visible_text",
            "tool_call_1",
            "tool_call_2",
            "terminal",
            "inter_tool_separator",
        }
        for b in self.ctx["boundaries"]:
            self.assertIn(b["region"], valid, b["id"])

    def test_semantic_event_timeline_is_present_and_ordered(self):
        turns = {t["turn"]: t for t in self.ctx["turns"]}
        # first turn diagnoses, last turn resolves
        self.assertIn("diagnosis_named", turns[1]["semantic_events"])
        self.assertIn("task_resolved", turns[9]["semantic_events"])
        # the key epistemic milestones each appear exactly once across the run
        all_semantic = [
            e for t in self.ctx["turns"] for e in t["semantic_events"]
        ]
        for milestone in (
            "bug_recognized",
            "patch_target_named",
            "fix_working",
            "task_resolved",
        ):
            self.assertEqual(all_semantic.count(milestone), 1, milestone)

    def test_working_turns_have_reasoning_and_a_tool_call(self):
        turns = {t["turn"]: t for t in self.ctx["turns"]}
        for turn in range(1, 9):  # turn 9 is terminal (task_resolved, no tool)
            self.assertTrue(turns[turn]["has_reasoning"], turn)
            self.assertTrue(turns[turn]["has_tool_call"], turn)


if __name__ == "__main__":
    unittest.main()
