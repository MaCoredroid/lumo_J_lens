from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from scripts import swe_task_state_v4_reasoning_trace as trace


ROOT = Path(__file__).resolve().parents[1]

_join_spec = importlib.util.spec_from_file_location(
    "lens_report_join", ROOT / "scripts/swe_task_state_v4_lens_report_join.py"
)
join = importlib.util.module_from_spec(_join_spec)
_join_spec.loader.exec_module(join)

_reader_spec = importlib.util.spec_from_file_location(
    "trajectory_cot_reader", ROOT / "scripts/swe_task_state_v4_trajectory_cot_reader.py"
)
reader = importlib.util.module_from_spec(_reader_spec)
_reader_spec.loader.exec_module(reader)


def _uniform() -> dict[str, float]:
    return {c: 1.0 / len(trace.CLASSES) for c in trace.CLASSES}


def _prediction(task_id: str, request_index: int) -> dict[str, object]:
    q = _uniform()
    top = max(trace.CLASSES, key=lambda c: q[c])
    return {
        "row_id": f"{task_id}-r{request_index}",
        "task_id": task_id,
        "task_request_index": request_index,
        "forecast_probabilities_q": q,
        "decision_probabilities_d": dict(q),
        "forecast_top_class": top,
        "forecast_top_confidence": q[top],
        "predicted_class": top,
        "decision_confidence_from_q": q[top],
        "accepted": True,
    }


class JoinLogicTests(unittest.TestCase):
    def test_attaches_turn_context_and_leaves_input_unmutated(self):
        rows = [{"task_id": "t", "task_request_index": 1, "row_id": "a"}]
        turns = [{"turn": 1, "stage": "s", "semantic_events": ["e1"], "n_boundaries": 3}]
        merged = join.attach_reasoning_context(rows, turns)
        self.assertNotIn(join.OBSERVED_KEY, rows[0])  # input untouched
        obs = merged[0][join.OBSERVED_KEY]
        self.assertEqual(obs["status"], "available")
        self.assertEqual(obs["stage"], "s")
        self.assertEqual(obs["semantic_events"], ["e1"])

    def test_missing_turn_is_marked_unavailable_not_dropped(self):
        rows = [{"task_id": "t", "task_request_index": 99, "row_id": "a"}]
        merged = join.attach_reasoning_context(rows, [{"turn": 1, "stage": "s"}])
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            merged[0][join.OBSERVED_KEY]["status"], "unavailable_no_matching_turn"
        )

    def test_duplicate_turn_rejected(self):
        with self.assertRaises(ValueError):
            join.attach_reasoning_context(
                [], [{"turn": 1, "stage": "a"}, {"turn": 1, "stage": "b"}]
            )


class JoinAgainstRealTrajectoryTests(unittest.TestCase):
    def setUp(self):
        if not reader.DEFAULT_TRAJECTORY.exists():
            self.skipTest("captured trajectory not present")
        self.turns = reader.read_free_reasoning_context()["turns"]
        # real reasoning_trace rows for turns 1..9 of the captured task
        preds = [_prediction("swe-sympy-13480", i) for i in range(1, 10)]
        self.rows = trace.build_reasoning_trace(preds)

    def test_every_trace_row_gets_its_turns_epistemic_events(self):
        merged = join.attach_reasoning_context(self.rows, self.turns)
        cov = join.coverage(merged)
        self.assertEqual(cov["with_observed_reasoning"], cov["rows"])
        by_turn = {join._row_turn(m): m[join.OBSERVED_KEY] for m in merged}
        self.assertEqual(by_turn[1]["stage"], "initial_task_analysis")
        self.assertIn("diagnosis_named", by_turn[1]["semantic_events"])
        self.assertIn("task_resolved", by_turn[9]["semantic_events"])
        # latent indices still present alongside the observed context
        self.assertIn("proxies", merged[0])
        self.assertIn("phase_forecast", merged[0])


if __name__ == "__main__":
    unittest.main()
