#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest

import numpy as np

from scripts import swe_task_state_v4_decision as decision


def class_probability(class_id: str, confidence: float = 0.8) -> list[float]:
    remainder = (1.0 - confidence) / 2.0
    values = [remainder, remainder, remainder]
    values[decision.CLASS_INDEX[class_id]] = confidence
    return values


class V4DecisionTests(unittest.TestCase):
    def test_validators_reject_invalid_probabilities_weights_and_grids(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            decision.validate_probability_matrix([[1.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "finite"):
            decision.validate_probability_matrix([[np.nan, 0.5, 0.5]])
        with self.assertRaisesRegex(ValueError, "sum to one"):
            decision.validate_probability_matrix([[0.2, 0.2, 0.2]])
        with self.assertRaisesRegex(ValueError, "non-boolean"):
            decision.validate_probability_matrix([[True, False, False]])
        with self.assertRaisesRegex(ValueError, "non-boolean"):
            decision.validate_weights([True], 1)
        with self.assertRaisesRegex(ValueError, "non-boolean"):
            decision.validate_weights(["1.0"], 1)
        with self.assertRaisesRegex(ValueError, "non-boolean"):
            decision.validate_probability_matrix([[0.2 + 0.1j, 0.3, 0.5]])
        with self.assertRaisesRegex(ValueError, "shape"):
            decision.validate_probability_matrix([[0.5, 0.5]])

        probabilities = np.asarray(
            [class_probability(class_id) for class_id in decision.CLASSES]
        )
        labels = list(decision.CLASSES)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            decision.select_decision_settings(
                probabilities,
                probabilities,
                labels,
                [1.0, 0.0, 1.0],
                alpha_grid=[0.0],
                edit_offset_grid=[0.0],
                check_or_finish_offset_grid=[0.0],
                recall_floors={class_id: 0.5 for class_id in decision.CLASSES},
                balanced_accuracy_minimum=0.5,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            decision.select_decision_settings(
                probabilities,
                probabilities,
                labels,
                [1.0, np.inf, 1.0],
                alpha_grid=[0.0],
                edit_offset_grid=[0.0],
                check_or_finish_offset_grid=[0.0],
                recall_floors={class_id: 0.5 for class_id in decision.CLASSES},
                balanced_accuracy_minimum=0.5,
            )
        with self.assertRaisesRegex(ValueError, "alpha grid exceeds"):
            decision.select_decision_settings(
                probabilities,
                probabilities,
                labels,
                [1.0, 1.0, 1.0],
                alpha_grid=[1.1],
                edit_offset_grid=[0.0],
                check_or_finish_offset_grid=[0.0],
                recall_floors={class_id: 0.5 for class_id in decision.CLASSES},
                balanced_accuracy_minimum=0.5,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            decision.select_decision_settings(
                probabilities,
                probabilities,
                labels,
                [1.0, 1.0, 1.0],
                alpha_grid=[0.0, 0.0],
                edit_offset_grid=[0.0],
                check_or_finish_offset_grid=[0.0],
                recall_floors={class_id: 0.5 for class_id in decision.CLASSES},
                balanced_accuracy_minimum=0.5,
            )
        with self.assertRaisesRegex(ValueError, "balanced-accuracy minimum"):
            decision.select_decision_settings(
                probabilities,
                probabilities,
                labels,
                [1.0, 1.0, 1.0],
                alpha_grid=[0.0],
                edit_offset_grid=[0.0],
                check_or_finish_offset_grid=[0.0],
                recall_floors={class_id: 0.0 for class_id in decision.CLASSES},
                balanced_accuracy_minimum=1.1,
            )

    def test_convex_fusion_and_recall_floor_selection(self) -> None:
        ordinary = np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.55, 0.4, 0.05],
                [0.1, 0.1, 0.8],
            ],
            dtype=np.float64,
        )
        direct = np.asarray(
            [
                [0.7, 0.2, 0.1],
                [0.50, 0.45, 0.05],
                [0.1, 0.2, 0.7],
            ],
            dtype=np.float64,
        )
        midpoint = decision.fuse_probabilities(ordinary, direct, 0.5)
        np.testing.assert_allclose(midpoint, (ordinary + direct) / 2.0)
        np.testing.assert_array_equal(
            decision.fuse_probabilities(ordinary, direct, 0.0), ordinary
        )
        np.testing.assert_array_equal(
            decision.fuse_probabilities(ordinary, direct, 1.0), direct
        )
        near_ordinary = ordinary.copy()
        near_direct = direct.copy()
        near_ordinary[0, 0] += 5e-13
        near_direct[0, 1] -= 5e-13
        expected_near = 0.75 * near_ordinary + 0.25 * near_direct
        np.testing.assert_array_equal(
            decision.fuse_probabilities(near_ordinary, near_direct, 0.25),
            expected_near,
        )

        selected = decision.select_decision_settings(
            ordinary,
            direct,
            ["inspect", "edit", "check_or_finish"],
            [1.0, 2.0, 1.0],
            alpha_grid=[1.0, 0.0, 0.5],
            edit_offset_grid=[0.0, 0.5],
            check_or_finish_offset_grid=[0.0],
            recall_floors={class_id: 1.0 for class_id in decision.CLASSES},
            balanced_accuracy_minimum=1.0,
        )
        self.assertTrue(selected["selected_under_floors"])
        self.assertFalse(selected["fallback_used"])
        self.assertEqual(selected["class_logit_offsets"]["edit"], 0.5)
        applied = decision.apply_decision_settings(
            ordinary,
            direct,
            selected,
            alpha_grid=[0.0, 0.5, 1.0],
            edit_offset_grid=[0.0, 0.5],
            check_or_finish_offset_grid=[0.0],
        )
        self.assertEqual(
            [decision.CLASSES[index] for index in np.argmax(applied, axis=1)],
            ["inspect", "edit", "check_or_finish"],
        )

    def test_offsets_move_argmax_and_application_rejects_off_grid(self) -> None:
        ordinary = np.asarray(
            [
                [0.55, 0.40, 0.05],
                [0.10, 0.80, 0.10],
                [0.10, 0.10, 0.80],
            ]
        )
        direct = ordinary.copy()
        moved = decision.apply_class_logit_offsets(
            ordinary, edit_offset=1.0, check_or_finish_offset=0.0
        )
        self.assertEqual(int(np.argmax(ordinary[0])), decision.CLASS_INDEX["inspect"])
        self.assertEqual(int(np.argmax(moved[0])), decision.CLASS_INDEX["edit"])

        settings = {
            "schema_version": decision.SCHEMA_VERSION,
            "classes_in_order": list(decision.CLASSES),
            "alpha": 0.0,
            "class_logit_offsets": {
                "inspect": 0.0,
                "edit": 1.0,
                "check_or_finish": 0.0,
            },
        }
        applied = decision.apply_decision_settings(
            ordinary,
            direct,
            settings,
            alpha_grid=[0.0, 1.0],
            edit_offset_grid=[0.0, 1.0],
            check_or_finish_offset_grid=[0.0],
        )
        np.testing.assert_array_equal(applied, moved)

        tampered = copy.deepcopy(settings)
        tampered["alpha"] = 0.5
        with self.assertRaisesRegex(ValueError, "off the frozen grid"):
            decision.apply_decision_settings(
                ordinary,
                direct,
                tampered,
                alpha_grid=[0.0, 1.0],
                edit_offset_grid=[0.0, 1.0],
                check_or_finish_offset_grid=[0.0],
            )
        inspect_tampered = copy.deepcopy(settings)
        inspect_tampered["class_logit_offsets"]["inspect"] = 0.1
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            decision.apply_decision_settings(
                ordinary,
                direct,
                inspect_tampered,
                alpha_grid=[0.0, 1.0],
                edit_offset_grid=[0.0, 1.0],
                check_or_finish_offset_grid=[0.0],
            )
        check_tampered = copy.deepcopy(settings)
        check_tampered["class_logit_offsets"]["check_or_finish"] = 0.5
        with self.assertRaisesRegex(ValueError, "off the frozen grid"):
            decision.apply_decision_settings(
                ordinary,
                direct,
                check_tampered,
                alpha_grid=[0.0, 1.0],
                edit_offset_grid=[0.0, 1.0],
                check_or_finish_offset_grid=[0.0],
            )
        edit_tampered = copy.deepcopy(settings)
        edit_tampered["class_logit_offsets"]["edit"] = 0.5
        with self.assertRaisesRegex(ValueError, "off the frozen grid"):
            decision.apply_decision_settings(
                ordinary,
                direct,
                edit_tampered,
                alpha_grid=[0.0, 1.0],
                edit_offset_grid=[0.0, 1.0],
                check_or_finish_offset_grid=[0.0],
            )

    def test_selection_is_deterministic_and_application_is_label_free(self) -> None:
        labels = ["inspect", "edit", "check_or_finish"] * 2
        uniform = np.full((len(labels), 3), 1.0 / 3.0, dtype=np.float64)
        weights = np.arange(1, len(labels) + 1, dtype=np.float64)
        uniform_before = uniform.copy()
        weights_before = weights.copy()
        kwargs = {
            "edit_offset_grid": [0.0],
            "check_or_finish_offset_grid": [0.0],
            "recall_floors": {class_id: 0.0 for class_id in decision.CLASSES},
            "balanced_accuracy_minimum": 0.0,
            "include_candidates": False,
        }
        first = decision.select_decision_settings(
            uniform,
            uniform,
            labels,
            weights,
            alpha_grid=[1.0, 0.5, 0.0],
            **kwargs,
        )
        second = decision.select_decision_settings(
            uniform,
            uniform,
            list(reversed(labels)),
            weights,
            alpha_grid=[0.0, 1.0, 0.5],
            **kwargs,
        )
        self.assertEqual(first["alpha"], 0.0)
        self.assertEqual(second["alpha"], 0.0)
        np.testing.assert_array_equal(uniform, uniform_before)
        np.testing.assert_array_equal(weights, weights_before)

        applied_before = decision.apply_decision_settings(
            uniform,
            uniform,
            first,
            alpha_grid=[0.0, 0.5, 1.0],
            edit_offset_grid=[0.0],
            check_or_finish_offset_grid=[0.0],
        )
        labels[:] = ["check_or_finish"] * len(labels)
        applied_after = decision.apply_decision_settings(
            uniform,
            uniform,
            first,
            alpha_grid=[1.0, 0.0, 0.5],
            edit_offset_grid=[0.0],
            check_or_finish_offset_grid=[0.0],
        )
        np.testing.assert_array_equal(applied_before, applied_after)

    def test_decision_fallback_is_explicit_and_fail_closed(self) -> None:
        labels = ["inspect", "edit", "check_or_finish"]
        always_inspect = np.asarray([[0.8, 0.1, 0.1]] * len(labels))
        selected = decision.select_decision_settings(
            always_inspect,
            always_inspect,
            labels,
            [1.0, 1.0, 1.0],
            alpha_grid=[0.0, 1.0],
            edit_offset_grid=[0.0],
            check_or_finish_offset_grid=[0.0],
            recall_floors={class_id: 0.8 for class_id in decision.CLASSES},
            balanced_accuracy_minimum=0.8,
        )
        self.assertTrue(selected["fallback_used"])
        self.assertFalse(selected["selected_under_floors"])
        self.assertLess(
            sum(selected["selected_recall_shortfall_by_class"].values()), 2.0
        )

    def test_balanced_accuracy_floor_is_conjunctive_with_recall_floors(self) -> None:
        labels = ["inspect", "edit", "check_or_finish"]
        always_inspect = np.asarray([[0.8, 0.1, 0.1]] * len(labels))
        common = {
            "alpha_grid": [0.0],
            "edit_offset_grid": [0.0],
            "check_or_finish_offset_grid": [0.0],
            "recall_floors": {class_id: 0.0 for class_id in decision.CLASSES},
            "include_candidates": False,
        }
        blocked = decision.select_decision_settings(
            always_inspect,
            always_inspect,
            labels,
            [1.0, 1.0, 1.0],
            balanced_accuracy_minimum=0.5,
            **common,
        )
        self.assertTrue(blocked["fallback_used"])
        self.assertEqual(blocked["selected_recall_shortfall_by_class"], {
            class_id: 0.0 for class_id in decision.CLASSES
        })
        self.assertGreater(blocked["selected_balanced_accuracy_shortfall"], 0.0)

        allowed = decision.select_decision_settings(
            always_inspect,
            always_inspect,
            labels,
            [1.0, 1.0, 1.0],
            balanced_accuracy_minimum=0.3,
            **common,
        )
        self.assertFalse(allowed["fallback_used"])

    def test_class_specific_thresholds_outperform_every_global_threshold(self) -> None:
        labels: list[str] = []
        predicted: list[str] = []
        separate_confidence: list[list[float]] = []

        # Six safe inspect decisions can be accepted at 0.5.
        for _ in range(6):
            labels.append("inspect")
            predicted.append("inspect")
            separate_confidence.append(class_probability("inspect", 0.6))
        # Each consequential class has two high-confidence correct rows.
        for class_id in ("edit", "check_or_finish"):
            for _ in range(2):
                labels.append(class_id)
                predicted.append(class_id)
                separate_confidence.append(class_probability(class_id, 0.9))
        # Low-confidence consequential errors should be rejected.
        for predicted_class, true_class in (
            ("edit", "inspect"),
            ("edit", "check_or_finish"),
            ("check_or_finish", "inspect"),
            ("check_or_finish", "edit"),
        ):
            labels.append(true_class)
            predicted.append(predicted_class)
            separate_confidence.append(class_probability(predicted_class, 0.6))

        # The decision matrix intentionally has the same 0.6 confidence on every row.
        decisions = np.asarray(
            [class_probability(class_id, 0.6) for class_id in predicted]
        )
        confidence = np.asarray(separate_confidence)
        weights = np.ones(len(labels), dtype=np.float64)
        selected = decision.select_class_confidence_thresholds(
            decisions,
            labels,
            weights,
            threshold_grid=[0.5, 0.8],
            accepted_accuracy_minimum=0.85,
            coverage_minimum=0.70,
            minimum_accepted_rows_per_true_class=1,
            confidence_probabilities=confidence,
        )
        self.assertTrue(selected["selected_under_floors"])
        self.assertFalse(selected["fallback_used"])
        self.assertEqual(
            selected["thresholds"],
            {"inspect": 0.5, "edit": 0.8, "check_or_finish": 0.8},
        )
        self.assertEqual(
            selected["confidence_source"], "separate_confidence_probabilities"
        )
        self.assertGreaterEqual(
            selected["selected_metrics"]["accepted_accuracy"], 0.85
        )
        self.assertGreaterEqual(selected["selected_metrics"]["coverage"], 0.70)
        reordered = decision.select_class_confidence_thresholds(
            decisions,
            labels,
            weights,
            threshold_grid=[0.8, 0.5],
            accepted_accuracy_minimum=0.85,
            coverage_minimum=0.70,
            minimum_accepted_rows_per_true_class=1,
            confidence_probabilities=confidence,
            include_candidates=False,
        )
        selected_without_candidates = {
            key: value for key, value in selected.items() if key != "candidates"
        }
        self.assertEqual(reordered, selected_without_candidates)

        # Neither global candidate reaches both floors with the decision matrix.
        global_results = []
        for threshold in (0.5, 0.8):
            settings = {
                "schema_version": decision.SCHEMA_VERSION,
                "classes_in_order": list(decision.CLASSES),
                "thresholds": {class_id: threshold for class_id in decision.CLASSES},
                "confidence_source": "separate_confidence_probabilities",
            }
            accepted = decision.apply_class_confidence_thresholds(
                decisions,
                settings,
                threshold_grid=[0.5, 0.8],
                confidence_probabilities=confidence,
            )
            correct = np.asarray(
                [left == right for left, right in zip(labels, predicted, strict=True)]
            )
            coverage = float(accepted.mean())
            accuracy = float(correct[accepted].mean()) if accepted.any() else 0.0
            global_results.append((accuracy, coverage))
        self.assertTrue(
            all(accuracy < 0.85 or coverage < 0.70 for accuracy, coverage in global_results)
        )

        accepted = decision.apply_class_confidence_thresholds(
            decisions,
            selected,
            threshold_grid=[0.8, 0.5],
            confidence_probabilities=confidence,
        )
        self.assertEqual(int(accepted.sum()), 10)
        missing_confidence = copy.deepcopy(selected)
        with self.assertRaisesRegex(ValueError, "source differs"):
            decision.apply_class_confidence_thresholds(
                decisions,
                missing_confidence,
                threshold_grid=[0.5, 0.8],
            )
        tampered = copy.deepcopy(selected)
        tampered["thresholds"]["edit"] = 0.7
        with self.assertRaisesRegex(ValueError, "off the frozen grid"):
            decision.apply_class_confidence_thresholds(
                decisions,
                tampered,
                threshold_grid=[0.5, 0.8],
                confidence_probabilities=confidence,
            )

    def test_separate_confidence_is_read_at_the_decision_class(self) -> None:
        decisions = np.asarray([[0.6, 0.3, 0.1]], dtype=np.float64)
        confidence = np.asarray([[0.3, 0.6, 0.1]], dtype=np.float64)
        predicted_indices, values = decision.confidence_for_decisions(
            decisions, confidence
        )
        self.assertEqual(predicted_indices.tolist(), [decision.CLASS_INDEX["inspect"]])
        self.assertEqual(values.tolist(), [0.3])

    def test_threshold_fallback_prefers_accuracy_subject_to_coverage(self) -> None:
        labels = [
            "inspect",
            "edit",
            "check_or_finish",
            "edit",
            "check_or_finish",
            "inspect",
        ]
        predicted = [
            "inspect",
            "edit",
            "check_or_finish",
            "inspect",
            "inspect",
            "edit",
        ]
        probabilities = np.asarray(
            [class_probability(class_id, 0.6) for class_id in predicted]
        )
        selected = decision.select_class_confidence_thresholds(
            probabilities,
            labels,
            np.ones(len(labels)),
            threshold_grid=[0.0, 0.9],
            accepted_accuracy_minimum=0.9,
            coverage_minimum=0.5,
            minimum_accepted_rows_per_true_class=1,
        )
        self.assertTrue(selected["fallback_used"])
        self.assertFalse(selected["selected_under_floors"])
        self.assertGreaterEqual(selected["selected_metrics"]["coverage"], 0.5)
        self.assertLess(selected["selected_metrics"]["accepted_accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
