#!/usr/bin/env python3
"""Focused tests for the task-state multinomial readout solver."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "swe_task_state_readout",
    ROOT / "scripts/swe_task_state_readout.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


FROZEN_SPEC = importlib.util.spec_from_file_location(
    "frozen_action_readout_for_equivalence",
    ROOT / "scripts/analyze_swe_action_layer_readout.py",
)
assert FROZEN_SPEC and FROZEN_SPEC.loader
FROZEN = importlib.util.module_from_spec(FROZEN_SPEC)
sys.modules[FROZEN_SPEC.name] = FROZEN
FROZEN_SPEC.loader.exec_module(FROZEN)


class SweTaskStateReadoutTest(unittest.TestCase):
    def test_four_class_96_feature_fit_matches_frozen_solver(self) -> None:
        rng = np.random.default_rng(9281)
        labels = np.repeat(np.arange(4, dtype=np.int64), 8)
        features = rng.normal(0.0, 0.08, size=(len(labels), 96))
        for row_index, class_index in enumerate(labels):
            features[row_index, class_index::4] += 2.0

        frozen = FROZEN.fit_multinomial_lbfgs(
            features,
            labels,
            c_value=0.1,
            maximum_iterations=4000,
        )
        extracted = MODULE.fit_multinomial_lbfgs(
            features,
            labels,
            c_value=0.1,
            maximum_iterations=4000,
        )

        self.assertEqual(extracted["parameter_sha256"], frozen["parameter_sha256"])
        self.assertEqual(extracted["iterations"], frozen["iterations"])
        np.testing.assert_array_equal(
            MODULE.predict_multinomial(extracted, features),
            FROZEN.predict_multinomial(frozen, features),
        )

    def test_arbitrary_feature_width_and_class_count_converge(self) -> None:
        rng = np.random.default_rng(1928)
        class_ids = ("edit", "validate", "finalize")
        labels = np.repeat(np.arange(len(class_ids), dtype=np.int64), 12)
        features = rng.normal(0.0, 0.05, size=(len(labels), 7))
        for row_index, class_index in enumerate(labels):
            features[row_index, class_index] += 2.0

        model = MODULE.fit_multinomial_lbfgs(
            features,
            labels,
            c_value=0.1,
            maximum_iterations=4000,
            class_ids=class_ids,
        )
        probabilities = MODULE.predict_multinomial(model, features)

        self.assertTrue(model["converged"], model)
        self.assertLessEqual(
            model["gradient_infinity_norm"], MODULE.GRADIENT_TOLERANCE
        )
        self.assertEqual(probabilities.shape, (len(labels), len(class_ids)))
        self.assertGreaterEqual(
            float(np.mean(np.argmax(probabilities, axis=1) == labels)),
            0.95,
        )
        self.assertEqual(set(model["class_support"]), set(class_ids))

    def test_objective_gradient_at_arbitrary_width_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(77)
        features = rng.normal(size=(8, 5))
        labels = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        sample_weights = np.ones(len(labels), dtype=np.float64)
        theta = rng.normal(0.0, 0.02, size=4 * features.shape[1] + 4)
        _, gradient = MODULE._weighted_multinomial_objective(
            theta,
            features,
            labels,
            sample_weights,
            class_count=4,
            c_value=0.3,
        )
        epsilon = 1e-6
        for index in (0, 7, 19, 20, 23):
            left = theta.copy()
            right = theta.copy()
            left[index] -= epsilon
            right[index] += epsilon
            left_value, _ = MODULE._weighted_multinomial_objective(
                left,
                features,
                labels,
                sample_weights,
                class_count=4,
                c_value=0.3,
            )
            right_value, _ = MODULE._weighted_multinomial_objective(
                right,
                features,
                labels,
                sample_weights,
                class_count=4,
                c_value=0.3,
            )
            self.assertAlmostEqual(
                gradient[index],
                (right_value - left_value) / (2 * epsilon),
                places=6,
            )

    def test_rejected_curvature_clears_stale_history(self) -> None:
        stale_step = np.asarray([1.0, 0.0], dtype=np.float64)
        stale_delta = np.asarray([1.0, 0.0], dtype=np.float64)
        history = [(stale_step, stale_delta, 1.0)]
        tiny_step = np.asarray([1e-8, 0.0], dtype=np.float64)
        tiny_delta = np.asarray([1e-8, 0.0], dtype=np.float64)

        MODULE._update_lbfgs_history(history, tiny_step, tiny_delta)

        self.assertEqual(history, [])


if __name__ == "__main__":
    unittest.main()
