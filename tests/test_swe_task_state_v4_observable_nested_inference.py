from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "scripts" / "swe_task_state_v4_observable_nested_inference.py"
)
spec = importlib.util.spec_from_file_location(
    "swe_task_state_v4_observable_nested_inference", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class ObservableNestedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = module.validate_config(module.load_json(module.CONFIG_PATH))
        decoder_config = module.DECODER.validate_config(
            module.load_json(module.DECODER.CONFIG_PATH)
        )
        alignment = module.DECODER.validate_alignment_index(
            module.load_json(
                ROOT / cls.config["inputs"]["alignment_index"]["path"]
            )
        )
        labels = module.DECODER.validate_label_sidecar(
            module.load_json(
                ROOT / cls.config["inputs"]["observable_label_sidecar"]["path"]
            ),
            alignment_rows=alignment,
            config=decoder_config,
        )
        cls.rows, cls.labels = module.available_rows_and_labels(
            alignment,
            labels,
            target=cls.config["target"],
        )
        cls.decoder_report = module.load_json(
            ROOT / cls.config["inputs"]["decoder_report"]["path"]
        )

    def test_config_pins_current_v2f_and_keeps_claims_false(self) -> None:
        binding = self.config["inputs"]["decoder_report"]
        self.assertTrue(binding["path"].endswith("observable-rationale-marker-v2f.json"))
        self.assertEqual(
            binding["sha256"],
            "5b7a9f5f152548f7d63d45ee41b2a1b9b338525e085af23d6bdcc3b9fc15c11c",
        )
        self.assertTrue(
            self.config["interpretation"][
                "configured_after_point_results_were_observed"
            ]
        )
        self.assertFalse(
            self.config["interpretation"]["model_refit_uncertainty_included"]
        )
        self.assertTrue(
            self.config["interpretation"][
                "nested_variants_added_after_initial_oof_results_were_observed"
            ]
        )
        self.assertFalse(
            self.config["interpretation"]["comparisons_preregistered"]
        )
        self.assertFalse(
            self.config["interpretation"]["untouched_confirmation_set_evaluated"]
        )
        self.assertTrue(all(value is False for value in self.config["claim_scope"].values()))
        self.assertFalse(
            self.config["claim_scope"]["cot_or_cot_like_decoding_established"]
        )
        self.assertFalse(
            self.config["claim_scope"][
                "semantic_sentence_or_chain_decoding_established"
            ]
        )
        limitations = " ".join(self.config["mandatory_limitations"]).lower()
        self.assertIn("private chain-of-thought", limitations)
        self.assertIn("emotion", limitations)

    def test_hierarchical_weights_equalize_repository_task_and_row(self) -> None:
        rows = [
            {"repository": "a", "task_id_sha256": "a1"},
            {"repository": "a", "task_id_sha256": "a1"},
            {"repository": "a", "task_id_sha256": "a2"},
            {"repository": "b", "task_id_sha256": "b1"},
            {"repository": "b", "task_id_sha256": "b1"},
            {"repository": "b", "task_id_sha256": "b1"},
        ]
        weights = module.hierarchical_row_weights(rows)
        np.testing.assert_allclose(
            weights,
            [0.125, 0.125, 0.25, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
        )
        self.assertAlmostEqual(float(weights[:3].sum()), 0.5)
        self.assertAlmostEqual(float(weights[3:].sum()), 0.5)

    def test_loss_vectors_have_candidate_minus_reference_favorable_sign(self) -> None:
        labels = np.asarray([0, 1], dtype=np.int64)
        candidate = module.loss_vectors(
            np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64), labels
        )
        reference = module.loss_vectors(
            np.asarray([[0.6, 0.4], [0.4, 0.6]], dtype=np.float64), labels
        )
        for metric in module.METRICS:
            self.assertTrue(np.all(candidate[metric] - reference[metric] < 0.0))

    def test_student_t_interval_is_centered_and_uses_n_minus_one_df(self) -> None:
        values = [-0.1, -0.2, -0.3, -0.4]
        result = module.student_t_interval(values, confidence_level=0.95)
        lower, upper = result["interval"]
        self.assertEqual(result["degrees_of_freedom"], 3)
        self.assertAlmostEqual((lower + upper) / 2.0, float(np.mean(values)))
        self.assertAlmostEqual(float(np.mean(values)) - lower, upper - float(np.mean(values)))

    def test_exact_sign_flip_enumerates_all_assignments_without_plus_one(self) -> None:
        result = module.exact_sign_flip([-1.0, -2.0, -4.0, -8.0])
        self.assertEqual(result["assignment_count"], 16)
        self.assertEqual(result["extreme_assignment_count"], 2)
        self.assertEqual(result["p_value"], 0.125)
        self.assertFalse(result["plus_one_correction"])

    def test_holm_step_down_is_monotone_and_tie_stable(self) -> None:
        family = ["a", "b", "c", "d"]
        result = module.holm_adjust(
            {"a": 0.01, "b": 0.01, "c": 0.03, "d": 0.2},
            family_order=family,
        )
        self.assertEqual(result["a"]["family_rank"], 1)
        self.assertEqual(result["b"]["family_rank"], 2)
        self.assertAlmostEqual(result["a"]["adjusted_p_value"], 0.04)
        self.assertAlmostEqual(result["b"]["adjusted_p_value"], 0.04)
        self.assertAlmostEqual(result["c"]["adjusted_p_value"], 0.06)
        self.assertAlmostEqual(result["d"]["adjusted_p_value"], 0.2)

    def test_current_v2f_report_authenticates_and_recomputes_exact_results(self) -> None:
        y, _weights, probabilities, evaluation = module.validate_decoder_report(
            self.decoder_report,
            rows=self.rows,
            labels=self.labels,
            config=self.config,
        )
        result = module.recompute_inference(
            rows=self.rows,
            y=y,
            probabilities=probabilities,
            decoder_evaluation=evaluation,
            config=self.config,
        )
        self.assertEqual(result["repository_count"], 10)
        self.assertEqual(result["comparison_count"], 2)
        for comparison in result["comparisons"].values():
            for metric in comparison["metrics"].values():
                self.assertEqual(metric["favorable_negative_repository_count"], 10)
                self.assertLess(metric["student_t"]["interval"][1], 0.0)
                self.assertEqual(metric["exact_sign_flip"]["p_value"], 2 / 1024)
                self.assertEqual(metric["holm"]["adjusted_p_value"], 8 / 1024)

    def test_probability_tamper_is_rejected(self) -> None:
        changed = copy.deepcopy(self.decoder_report)
        variant = changed["evaluation"]["outer_evaluation"]["variants"][
            module.COMPARISONS[0]["candidate"]
        ]
        variant["oof_probabilities"][0][0] += 1e-6
        variant["oof_probabilities"][0][1] -= 1e-6
        with self.assertRaisesRegex(module.InferenceError, "probability matrix changed"):
            module.validate_decoder_report(
                changed,
                rows=self.rows,
                labels=self.labels,
                config=self.config,
            )

    def test_no_clobber_writer_preserves_first_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            module._write_no_clobber(output, {"first": True})
            with self.assertRaisesRegex(module.InferenceError, "output exists"):
                module._write_no_clobber(output, {"second": True})
            self.assertEqual(module.load_json(output), {"first": True})


if __name__ == "__main__":
    unittest.main()
