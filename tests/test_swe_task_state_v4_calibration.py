#!/usr/bin/env python3

from __future__ import annotations

import copy
import inspect
import unittest

import numpy as np

from scripts import swe_task_state_v4_calibration as calibration


class V4CalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.probabilities = np.asarray(
            [
                [0.60, 0.25, 0.15],
                [0.15, 0.70, 0.15],
                [0.20, 0.20, 0.60],
                [0.65, 0.20, 0.15],
                [0.15, 0.65, 0.20],
                [0.15, 0.20, 0.65],
            ],
            dtype=np.float64,
        )
        self.labels = ["inspect", "edit", "check_or_finish"] * 2
        self.weights = np.asarray([3.0, 2.0, 1.0, 3.0, 2.0, 1.0])

    def test_selects_weighted_nll_minimum_and_applies_exactly(self) -> None:
        probabilities_before = self.probabilities.copy()
        weights_before = self.weights.copy()
        grid = [2.0, 0.5, 1.0]
        grid_before = list(grid)

        settings = calibration.select_temperature_settings(
            self.probabilities,
            self.labels,
            self.weights,
            temperature_grid=grid,
        )

        candidate_nll = {
            temperature: calibration.weighted_multiclass_nll(
                calibration.temperature_scale_probabilities(
                    self.probabilities, temperature
                ),
                self.labels,
                self.weights,
            )
            for temperature in grid
        }
        self.assertEqual(settings["temperature"], min(candidate_nll, key=candidate_nll.get))
        self.assertEqual(settings["temperature"], 0.5)
        self.assertEqual(settings["temperature_grid"], [0.5, 1.0, 2.0])

        expected = calibration.temperature_scale_probabilities(
            self.probabilities, settings["temperature"]
        )
        actual = calibration.apply_temperature_settings(
            self.probabilities,
            settings,
            temperature_grid=reversed(grid),
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(self.probabilities, probabilities_before)
        np.testing.assert_array_equal(self.weights, weights_before)
        self.assertEqual(grid, grid_before)

    def test_selection_uses_supplied_positive_weights(self) -> None:
        probabilities = np.asarray(
            [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.90, 0.05, 0.05],
            ]
        )
        labels = list(calibration.CLASSES)
        grid = [0.5, 1.0, 2.0]
        light_error = calibration.select_temperature_settings(
            probabilities,
            labels,
            [100.0, 100.0, 1.0],
            temperature_grid=grid,
        )
        heavy_error = calibration.select_temperature_settings(
            probabilities,
            labels,
            [1.0, 1.0, 20.0],
            temperature_grid=grid,
        )
        self.assertEqual(light_error["temperature"], 0.5)
        self.assertEqual(heavy_error["temperature"], 2.0)

    def test_geometric_pool_formula_endpoints_and_no_mutation(self) -> None:
        signature = inspect.signature(calibration.geometric_pool_probabilities)
        self.assertEqual(tuple(signature.parameters), ("first", "second", "alpha"))
        self.assertEqual(
            signature.parameters["alpha"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )

        first = np.asarray(
            [
                [0.70, 0.20, 0.10],
                [0.15, 0.65, 0.20],
            ],
            dtype=np.float64,
        )
        second = np.asarray(
            [
                [0.20, 0.30, 0.50],
                [0.45, 0.25, 0.30],
            ],
            dtype=np.float64,
        )
        first_before = first.copy()
        second_before = second.copy()

        unnormalized = np.exp(0.8 * np.log(first) + 0.2 * np.log(second))
        expected = unnormalized / unnormalized.sum(axis=1, keepdims=True)
        actual = calibration.geometric_pool_probabilities(
            first,
            second,
            alpha=0.2,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)
        np.testing.assert_array_equal(
            calibration.geometric_pool_probabilities(first, second, alpha=0.0),
            first,
        )
        np.testing.assert_array_equal(
            calibration.geometric_pool_probabilities(first, second, alpha=1.0),
            second,
        )
        self.assertIsNot(
            calibration.geometric_pool_probabilities(first, second, alpha=0.0),
            first,
        )
        self.assertIsNot(
            calibration.geometric_pool_probabilities(first, second, alpha=1.0),
            second,
        )
        np.testing.assert_array_equal(first, first_before)
        np.testing.assert_array_equal(second, second_before)

    def test_geometric_pool_strict_validation(self) -> None:
        valid = np.asarray(
            [
                [0.60, 0.25, 0.15],
                [0.15, 0.70, 0.15],
            ],
            dtype=np.float64,
        )
        invalid_matrices = (
            [[0.6, 0.4, 0.0], [0.15, 0.70, 0.15]],
            [[0.6, 0.25, 0.15]],
            [[True, False, False], [0.15, 0.70, 0.15]],
            [["0.6", "0.25", "0.15"], [0.15, 0.70, 0.15]],
        )
        for value in invalid_matrices:
            with self.subTest(probabilities=value), self.assertRaises(ValueError):
                calibration.geometric_pool_probabilities(valid, value, alpha=0.2)

        for alpha in (-0.1, 1.1, np.nan, np.inf, True, "0.2"):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                calibration.geometric_pool_probabilities(
                    valid,
                    valid,
                    alpha=alpha,  # type: ignore[arg-type]
                )

    def test_grid_order_invariance_and_tie_breaks(self) -> None:
        uniform = np.full((6, 3), 1.0 / 3.0)
        first = calibration.select_temperature_settings(
            uniform,
            self.labels,
            self.weights,
            temperature_grid=[2.0, 0.5, 1.0],
        )
        second = calibration.select_temperature_settings(
            uniform,
            self.labels,
            self.weights,
            temperature_grid=[1.0, 2.0, 0.5],
        )
        self.assertEqual(first, second)
        self.assertEqual(first["temperature"], 1.0)

        lower_tie = calibration.select_temperature_settings(
            uniform,
            self.labels,
            self.weights,
            temperature_grid=[1.25, 0.75],
        )
        self.assertEqual(lower_tie["temperature"], 0.75)

    def test_application_is_label_free_and_does_not_mutate_settings(self) -> None:
        settings = calibration.select_temperature_settings(
            self.probabilities,
            self.labels,
            self.weights,
            temperature_grid=[0.5, 1.0, 2.0],
        )
        settings_before = copy.deepcopy(settings)
        signature = inspect.signature(calibration.apply_temperature_settings)
        self.assertNotIn("labels", signature.parameters)
        self.assertNotIn("weights", signature.parameters)

        first = calibration.apply_temperature_settings(
            self.probabilities,
            settings,
            temperature_grid=[0.5, 1.0, 2.0],
        )
        self.labels[:] = ["check_or_finish"] * len(self.labels)
        second = calibration.apply_temperature_settings(
            self.probabilities,
            settings,
            temperature_grid=[2.0, 1.0, 0.5],
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(settings, settings_before)

    def test_rejects_invalid_probabilities_labels_weights_and_grid(self) -> None:
        invalid_probabilities = (
            [[1.0, 0.0, 0.0]],
            [[np.nan, 0.5, 0.5]],
            [[0.2, 0.2, 0.2]],
            [[True, False, False]],
            [["0.2", "0.3", "0.5"]],
        )
        for value in invalid_probabilities:
            with self.subTest(probabilities=value):
                with self.assertRaises(ValueError):
                    calibration.validate_probability_matrix(value)

        for value in ([1.0, 0.0, 1.0], [1.0, np.inf, 1.0], [True, 1.0, 1.0], ["1", 1, 1]):
            with self.subTest(weights=value):
                with self.assertRaises(ValueError):
                    calibration.select_temperature_settings(
                        self.probabilities[:3],
                        list(calibration.CLASSES),
                        value,
                        temperature_grid=[1.0],
                    )

        for labels in (
            ["inspect", "edit", "edit"] * 2,
            ["inspect", "edit", "unknown"] * 2,
            ["inspect", "edit"],
        ):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    calibration.select_temperature_settings(
                        self.probabilities,
                        labels,
                        self.weights,
                        temperature_grid=[1.0],
                    )

        invalid_grids = (
            [],
            [0.0, 1.0],
            [-1.0, 1.0],
            [np.inf, 1.0],
            [True, 1.0],
            ["0.5", 1.0],
            [0.5, 0.5],
            {0.5, 1.0},
        )
        for grid in invalid_grids:
            with self.subTest(grid=grid):
                with self.assertRaises(ValueError):
                    calibration.canonicalize_temperature_grid(grid)

    def test_application_rejects_tampering_off_grid_and_wrong_identity(self) -> None:
        grid = [0.5, 1.0, 2.0]
        settings = calibration.select_temperature_settings(
            self.probabilities,
            self.labels,
            self.weights,
            temperature_grid=grid,
        )

        mutations = []
        wrong_identity = copy.deepcopy(settings)
        wrong_identity["calibrator_identity"] = "some-other-calibrator"
        mutations.append(wrong_identity)
        wrong_schema = copy.deepcopy(settings)
        wrong_schema["schema_version"] = True
        mutations.append(wrong_schema)
        wrong_classes = copy.deepcopy(settings)
        wrong_classes["classes_in_order"] = list(reversed(calibration.CLASSES))
        mutations.append(wrong_classes)
        wrong_objective = copy.deepcopy(settings)
        wrong_objective["selection_objective"] = "accuracy"
        mutations.append(wrong_objective)
        reordered_stored_grid = copy.deepcopy(settings)
        reordered_stored_grid["temperature_grid"] = [2.0, 1.0, 0.5]
        mutations.append(reordered_stored_grid)
        off_grid = copy.deepcopy(settings)
        off_grid["temperature"] = 0.75
        mutations.append(off_grid)
        extra_field = copy.deepcopy(settings)
        extra_field["selected_nll"] = 0.1
        mutations.append(extra_field)

        for tampered in mutations:
            with self.subTest(settings=tampered):
                with self.assertRaises(ValueError):
                    calibration.apply_temperature_settings(
                        self.probabilities,
                        tampered,
                        temperature_grid=grid,
                    )

        with self.assertRaisesRegex(ValueError, "frozen grid identity"):
            calibration.apply_temperature_settings(
                self.probabilities,
                settings,
                temperature_grid=[0.5, 1.0],
            )


if __name__ == "__main__":
    unittest.main()
