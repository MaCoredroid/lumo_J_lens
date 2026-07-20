#!/usr/bin/env python3

from __future__ import annotations

import math
import unittest

import numpy as np

from scripts import swe_task_state_v4_metrics as metrics


class V4KnownRowMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.q = np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ],
            dtype=np.float64,
        )
        self.labels = np.asarray(metrics.CLASSES, dtype=object)
        self.weights = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        self.accepted = np.asarray([True, False, True], dtype=np.bool_)

    def evaluate(
        self,
        *,
        q: object | None = None,
        d: object | None = None,
        labels: object | None = None,
        weights: object | None = None,
        accepted: object | None = None,
        ece_bins: object = 2,
    ) -> dict[str, object]:
        return metrics.known_row_metrics(
            self.q if q is None else q,
            self.q if d is None else d,
            self.labels if labels is None else labels,
            self.weights if weights is None else weights,
            self.accepted if accepted is None else accepted,
            ece_bins=ece_bins,  # type: ignore[arg-type]
        )

    def test_exact_hand_calculations_and_metric_sources(self) -> None:
        result = self.evaluate()

        expected_nll = -(math.log(0.8) + math.log(0.7) + math.log(0.7)) / 3.0
        expected_brier = (0.06 + 0.14 + 0.14) / 3.0
        expected_ece = 1.0 - (0.8 + 0.7 + 0.7) / 3.0
        expected_decision_binary_brier = (0.2**2 + 0.3**2 + 0.3**2) / 3.0

        self.assertEqual(result["row_count"], 3)
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["balanced_accuracy"], 1.0)
        self.assertEqual(
            result["per_class_recall"],
            {class_id: 1.0 for class_id in metrics.CLASSES},
        )
        self.assertAlmostEqual(
            result["multiclass_negative_log_likelihood"], expected_nll
        )
        self.assertAlmostEqual(result["multiclass_brier"], expected_brier)
        self.assertAlmostEqual(result["top_label_ece"], expected_ece)
        self.assertAlmostEqual(result["decision_confidence_ece"], expected_ece)
        self.assertAlmostEqual(
            result["decision_confidence_binary_log_loss"], expected_nll
        )
        self.assertAlmostEqual(
            result["decision_confidence_binary_brier"],
            expected_decision_binary_brier,
        )
        self.assertAlmostEqual(result["selected_coverage"], 2.0 / 3.0)
        self.assertAlmostEqual(result["selected_accepted_accuracy"], 1.0)
        self.assertEqual(
            result["per_true_class_accepted_coverage"],
            {"inspect": 1.0, "edit": 0.0, "check_or_finish": 1.0},
        )
        sources = result["metric_sources"]
        self.assertEqual(
            sources["accuracy"]["prediction"],
            "argmax(decision_probabilities_d)",
        )
        self.assertEqual(
            sources["multiclass_negative_log_likelihood"]["probabilities"],
            "forecast_probabilities_q",
        )
        self.assertEqual(
            sources["decision_confidence_ece"]["confidence"],
            "forecast_probabilities_q[row, argmax(decision_probabilities_d[row])]",
        )
        self.assertEqual(
            sources["decision_confidence_binary_log_loss"]["target"],
            "argmax(decision_probabilities_d) == true_label",
        )
        self.assertEqual(
            sources["decision_confidence_binary_brier"]["confidence"],
            "forecast_probabilities_q[row, argmax(decision_probabilities_d[row])]",
        )

    def test_changing_d_changes_operational_recalls_not_q_proper_scores(self) -> None:
        good = self.evaluate()
        all_inspect_d = np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.6, 0.3, 0.1],
                [0.7, 0.1, 0.2],
            ],
            dtype=np.float64,
        )
        changed = self.evaluate(d=all_inspect_d)

        for metric_name in (
            "multiclass_negative_log_likelihood",
            "multiclass_brier",
            "top_label_ece",
        ):
            self.assertEqual(changed[metric_name], good[metric_name])
        self.assertEqual(
            changed["per_class_recall"],
            {"inspect": 1.0, "edit": 0.0, "check_or_finish": 0.0},
        )
        self.assertAlmostEqual(changed["balanced_accuracy"], 1.0 / 3.0)
        self.assertNotEqual(
            changed["decision_confidence_ece"], good["decision_confidence_ece"]
        )
        self.assertAlmostEqual(
            changed["decision_confidence_binary_log_loss"],
            -(math.log(0.8) + math.log(0.8) + math.log(0.9)) / 3.0,
        )
        self.assertAlmostEqual(
            changed["decision_confidence_binary_brier"],
            (0.2**2 + 0.2**2 + 0.1**2) / 3.0,
        )
        self.assertNotEqual(
            changed["decision_confidence_binary_log_loss"],
            good["decision_confidence_binary_log_loss"],
        )
        self.assertNotEqual(
            changed["decision_confidence_binary_brier"],
            good["decision_confidence_binary_brier"],
        )

    def test_changing_q_changes_scores_and_confidence_not_decisions(self) -> None:
        original = self.evaluate()
        wrong_forecast_q = np.asarray(
            [
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.7, 0.1, 0.2],
            ],
            dtype=np.float64,
        )
        changed = self.evaluate(q=wrong_forecast_q, d=self.q)

        for metric_name in (
            "accuracy",
            "balanced_accuracy",
            "per_class_recall",
            "recall_inspect",
            "recall_edit",
            "recall_check_or_finish",
            "selected_accepted_accuracy",
        ):
            self.assertEqual(changed[metric_name], original[metric_name])
        for metric_name in (
            "multiclass_negative_log_likelihood",
            "multiclass_brier",
            "top_label_ece",
            "decision_confidence_ece",
            "decision_confidence_binary_log_loss",
            "decision_confidence_binary_brier",
        ):
            self.assertNotEqual(changed[metric_name], original[metric_name])
        self.assertAlmostEqual(
            changed["decision_confidence_binary_log_loss"], -math.log(0.2)
        )
        self.assertAlmostEqual(changed["decision_confidence_binary_brier"], 0.8**2)

    def test_caller_weighting_applies_to_operational_and_selective_metrics(self) -> None:
        labels = [
            "inspect",
            "edit",
            "check_or_finish",
            "inspect",
            "edit",
            "check_or_finish",
        ]
        q = np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )
        d = np.asarray(
            [
                [0.8, 0.1, 0.1],  # inspect: correct
                [0.8, 0.1, 0.1],  # edit: wrong
                [0.1, 0.1, 0.8],  # check: correct
                [0.1, 0.8, 0.1],  # inspect: wrong
                [0.1, 0.8, 0.1],  # edit: correct
                [0.8, 0.1, 0.1],  # check: wrong
            ]
        )
        result = metrics.known_row_metrics(
            q,
            d,
            labels,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [True, True, False, False, True, True],
            ece_bins=5,
        )

        self.assertAlmostEqual(result["accuracy"], 9.0 / 21.0)
        self.assertAlmostEqual(result["recall_inspect"], 1.0 / 5.0)
        self.assertAlmostEqual(result["recall_edit"], 5.0 / 7.0)
        self.assertAlmostEqual(result["recall_check_or_finish"], 1.0 / 3.0)
        self.assertAlmostEqual(
            result["balanced_accuracy"],
            (1.0 / 5.0 + 5.0 / 7.0 + 1.0 / 3.0) / 3.0,
        )
        self.assertAlmostEqual(result["selected_coverage"], 14.0 / 21.0)
        self.assertAlmostEqual(result["selected_accepted_accuracy"], 6.0 / 14.0)
        expected_coverage = result["per_true_class_accepted_coverage"]
        self.assertAlmostEqual(expected_coverage["inspect"], 1.0 / 5.0)
        self.assertAlmostEqual(expected_coverage["edit"], 1.0)
        self.assertAlmostEqual(expected_coverage["check_or_finish"], 2.0 / 3.0)
        self.assertEqual(result["weighting"]["input_weight_sum"], 21.0)
        self.assertAlmostEqual(result["weighting"]["normalized_weight_sum"], 1.0)

    def test_empty_selection_has_zero_coverage_and_no_conditional_accuracy(self) -> None:
        result = self.evaluate(accepted=[False, False, False])
        self.assertEqual(result["selected_coverage"], 0.0)
        self.assertIsNone(result["selected_accepted_accuracy"])
        self.assertEqual(
            result["per_true_class_accepted_coverage"],
            {class_id: 0.0 for class_id in metrics.CLASSES},
        )

    def test_partial_class_support_nulls_only_undefined_metrics(self) -> None:
        result = metrics.known_row_metrics(
            self.q[:2],
            self.q[:2],
            ["inspect", "edit"],
            [1.0, 2.0],
            [True, False],
            ece_bins=2,
            require_all_classes=False,
        )

        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(
            result["multiclass_negative_log_likelihood"],
            -(math.log(0.8) / 3.0 + 2.0 * math.log(0.7) / 3.0),
        )
        self.assertAlmostEqual(result["recall_inspect"], 1.0)
        self.assertAlmostEqual(result["recall_edit"], 1.0)
        self.assertIsNone(result["recall_check_or_finish"])
        self.assertIsNone(result["balanced_accuracy"])
        self.assertEqual(
            result["per_true_class_accepted_coverage"],
            {"inspect": 1.0, "edit": 0.0, "check_or_finish": None},
        )
        self.assertAlmostEqual(result["selected_coverage"], 1.0 / 3.0)
        self.assertAlmostEqual(result["selected_accepted_accuracy"], 1.0)

        with self.assertRaisesRegex(ValueError, "require_all_classes"):
            metrics.known_row_metrics(
                self.q[:2],
                self.q[:2],
                ["inspect", "edit"],
                [1.0, 2.0],
                [True, False],
                ece_bins=2,
                require_all_classes=1,  # type: ignore[arg-type]
            )

    def test_inputs_are_never_mutated(self) -> None:
        q = self.q.copy()
        d = self.q.copy()
        labels = self.labels.copy()
        weights = self.weights.copy()
        accepted = self.accepted.copy()
        originals = [value.copy() for value in (q, d, labels, weights, accepted)]

        metrics.known_row_metrics(
            q, d, labels, weights, accepted, ece_bins=np.int64(4)
        )

        for value, original in zip(
            (q, d, labels, weights, accepted), originals, strict=True
        ):
            np.testing.assert_array_equal(value, original)

    def test_strict_validation_rejects_bad_probabilities(self) -> None:
        invalid_q_values = {
            "zero": [[0.8, 0.2, 0.0]] * 3,
            "nan": [[np.nan, 0.5, 0.5]] * 3,
            "bad sum": [[0.2, 0.2, 0.2]] * 3,
            "boolean": [[True, False, False]] * 3,
            "string": [["0.8", "0.1", "0.1"]] * 3,
            "complex": [[0.8 + 0.0j, 0.1, 0.1]] * 3,
            "one dimensional": [0.8, 0.1, 0.1],
            "wrong class count": [[0.5, 0.5]] * 3,
            "empty": np.empty((0, 3)),
        }
        for case, value in invalid_q_values.items():
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.evaluate(q=value)

        with self.assertRaisesRegex(ValueError, "same row count"):
            self.evaluate(d=np.asarray([self.q[0], self.q[1]]))
        bad_d = self.q.copy()
        bad_d[0, 0] = 0.0
        bad_d[0, 1] = 0.9
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            self.evaluate(d=bad_d)

    def test_strict_validation_rejects_labels_weights_mask_and_bins(self) -> None:
        invalid_calls = [
            {"labels": ["inspect", "edit", "edit"]},
            {"labels": ["inspect", "edit", "unknown"]},
            {"labels": [["inspect"], ["edit"], ["check_or_finish"]]},
            {"weights": [1.0, 0.0, 1.0]},
            {"weights": [1.0, np.inf, 1.0]},
            {"weights": [True, 1.0, 1.0]},
            {"weights": ["1.0", "1.0", "1.0"]},
            {"weights": [1.0 + 0.0j, 1.0, 1.0]},
            {"weights": [[1.0], [1.0], [1.0]]},
            {"accepted": [1, 0, 1]},
            {"accepted": [True, False, "yes"]},
            {"accepted": [[True], [False], [True]]},
            {"ece_bins": 0},
            {"ece_bins": -1},
            {"ece_bins": 2.0},
            {"ece_bins": True},
            {"ece_bins": "2"},
        ]
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.evaluate(**arguments)


if __name__ == "__main__":
    unittest.main()
