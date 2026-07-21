from __future__ import annotations

import unittest

from scripts import swe_task_state_v4_qwen_lens_report as report
from scripts import swe_task_state_v4_source_divergence as divergence


class QwenLensReportTests(unittest.TestCase):
    def setUp(self):
        if not divergence.ACTION_PHASE_ARTIFACT.exists():
            self.skipTest("action-phase artifact not present")
        self.report = report.build_report(permutations=200)

    def test_scope_is_qwen_only(self):
        scope = self.report["scope"]
        self.assertEqual(scope["subject_model_family"], "qwen")
        self.assertFalse(scope["gpt_oss_or_mistral_used"])
        self.assertFalse(scope["external_annotators_used"])
        self.assertFalse(scope["human_labels_used"])

    def test_action_gauge_present_and_reasonable(self):
        gauge = self.report["action_gauge"]["per_source"]
        acc = gauge["sequence_logit_j"]["weighted_accuracy"]
        self.assertIsNotNone(acc)
        self.assertGreater(acc, 0.6)  # gauge is a real signal, well above chance
        # activation sources beat the history-only baseline on ranking
        self.assertGreater(
            gauge["sequence_logit_j"]["weighted_auprc"],
            gauge["history_only"]["weighted_auprc"],
        )

    def test_reliability_flag_present_and_scoped_as_not_faithfulness(self):
        flag = self.report["lens_reliability_flag"]
        self.assertIn("source_disagreement", flag["metric"])
        self.assertIn("NOT a CoT-faithfulness", flag["interpretation"])
        self.assertGreaterEqual(flag["permutation_p_value"], 0.0)
        self.assertLessEqual(flag["permutation_p_value"], 1.0)
        self.assertGreaterEqual(flag["error_detection_auc"], 0.0)
        self.assertLessEqual(flag["error_detection_auc"], 1.0)
        self.assertEqual(flag["n_error"] + flag["n_correct"], flag["n_rows"])

    def test_free_cot_timeline_and_limitations(self):
        self.assertTrue(self.report["limitations"])
        demo = self.report["free_cot_timeline_demo"]
        if "epistemic_chain" in demo:
            all_events = [e for t in demo["epistemic_chain"] for e in t["events"]]
            self.assertIn("task_resolved", all_events)


if __name__ == "__main__":
    unittest.main()
