from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "cot_concept_faithfulness",
    ROOT / "scripts/swe_task_state_v4_cot_concept_faithfulness.py",
)
faith = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = faith
_spec.loader.exec_module(faith)

# The three families that fail closed as unscorable — the mapping must avoid them.
_UNSCORABLE = {"typographical_error", "repair_success", "repair_summary"}


class MappingTests(unittest.TestCase):
    def test_headline_mapping_targets_only_scorable_families(self):
        for event, concept in faith.EVENT_TO_CONCEPT.items():
            self.assertNotIn(concept, _UNSCORABLE, f"{event}->{concept} is unscorable")

    def test_nearest_boundary_picks_same_turn_closest_offset(self):
        bs = [
            {"request_index": 3, "offset": 0},
            {"request_index": 3, "offset": 40},
            {"request_index": 4, "offset": 5},
        ]
        got = faith._nearest_boundary(bs, 3, 32)
        self.assertEqual(got["offset"], 40)
        self.assertIsNone(faith._nearest_boundary(bs, 9, 0))


class RealTaskFaithfulnessTests(unittest.TestCase):
    def setUp(self):
        if not faith.CONCEPT_CHAIN_ARTIFACT.exists():
            self.skipTest("concept-chain artifact not present")
        if not faith.cot.DEFAULT_TRAJECTORY.exists():
            self.skipTest("trajectory not present")
        self.result = faith.score_faithfulness()

    def test_result_shape_and_bounded_rates(self):
        r = self.result
        self.assertEqual(r["reliability_status"], "descriptive_single_task_uncalibrated")
        self.assertGreater(r["n_mapped_events_aligned"], 0)
        for key in (
            "faithfulness_top1_agreement_all",
            "faithfulness_topk_agreement_all",
            "free_event_vs_human_label_agreement",
        ):
            self.assertGreaterEqual(r[key], 0.0)
            self.assertLessEqual(r[key], 1.0)

    def test_enrichments_present(self):
        r = self.result
        # native_j agreement reported alongside public_j
        self.assertIsNotNone(r["faithfulness_top1_agreement_native_j_all"])
        # uncertain (lower-confidence) mapping reported separately
        self.assertIn("uncertain_mapping", r)
        self.assertIn("n_mapped_events_aligned", r["uncertain_mapping"])
        # baseline-centered agreement is reported alongside raw
        self.assertIsNotNone(r["faithfulness_top1_agreement_baseline_centered_all"])
        # the focused_validation degeneracy is quantified, and centering removes it
        bias = r["focused_validation_bias"]
        self.assertEqual(bias["n_boundaries"], 10)
        self.assertGreater(bias["public_j_top1_is_focused_validation_raw"], 0.4)
        self.assertLess(bias["public_j_top1_is_focused_validation_centered"], 0.2)

    def test_known_boundaries(self):
        by_event = {row["event"]: row for row in self.result["per_event"]}
        # task_resolved: internal top-1 encodes task_resolution -> a genuine match
        self.assertTrue(by_event["task_resolved"]["match_top1"])
        # the free CoT events broadly agree with the human positive labels
        self.assertGreater(self.result["free_event_vs_human_label_agreement"], 0.5)


if __name__ == "__main__":
    unittest.main()
