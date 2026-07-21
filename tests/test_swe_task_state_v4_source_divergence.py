from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

from scripts import swe_task_state_v4_reasoning_trace as trace


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "source_divergence", ROOT / "scripts/swe_task_state_v4_source_divergence.py"
)
div = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = div  # dataclass field resolution needs this
_spec.loader.exec_module(div)


class JsdTests(unittest.TestCase):
    def test_identical_distributions_have_zero_divergence(self):
        p = np.array([[0.7, 0.2, 0.1], [0.2, 0.2, 0.6]])
        self.assertTrue(np.allclose(div.normalized_jsd(p, p), 0.0, atol=1e-12))

    def test_divergence_is_in_unit_interval_and_positive_when_different(self):
        p = np.array([[0.9, 0.05, 0.05]])
        q = np.array([[0.05, 0.9, 0.05]])
        value = div.normalized_jsd(p, q)[0]
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_matches_reasoning_trace_jsd(self):
        p = [0.6, 0.3, 0.1]
        q = [0.2, 0.3, 0.5]
        mine = float(div.normalized_jsd(np.array([p]), np.array([q]))[0])
        keys = trace.CLASSES
        theirs = trace.normalized_js_divergence(
            dict(zip(keys, p)), dict(zip(keys, q))
        )
        self.assertTrue(math.isclose(mine, theirs, rel_tol=1e-9, abs_tol=1e-9))


class RankAucTests(unittest.TestCase):
    def test_perfect_and_reversed_separation(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        positive = np.array([False, False, True, True])
        self.assertAlmostEqual(div._rank_auc(scores, positive), 1.0)
        self.assertAlmostEqual(div._rank_auc(scores, ~positive), 0.0)


class EvaluateSyntheticTests(unittest.TestCase):
    def _sources(self):
        n = 200
        surface = np.tile([0.8, 0.1, 0.1], (n, 1))
        internal = surface.copy()
        pooled = surface.copy()
        labels = np.zeros(n, dtype=np.int64)
        # second half: internal disagrees (high JSD) and pooled mispredicts
        internal[n // 2 :] = [0.1, 0.8, 0.1]
        pooled[n // 2 :] = [0.1, 0.8, 0.1]  # argmax 1 != label 0 -> error
        weights = np.full(n, 1.0 / n)
        return div.SourceProbabilities(
            div.CLASSES, surface, internal, pooled, labels, weights
        )

    def test_divergence_flags_error_with_significant_effect(self):
        result = div.evaluate_divergence_flags_error(self._sources(), permutations=500)
        self.assertEqual(result["n_rows"], 200)
        self.assertEqual(result["n_error"], 100)
        self.assertGreater(result["error_minus_correct_effect"], 0.0)
        self.assertGreater(result["error_detection_auc"], 0.9)
        self.assertLess(result["permutation_p_value"], 0.01)


class RealCohortTests(unittest.TestCase):
    def setUp(self):
        if not div.ACTION_PHASE_ARTIFACT.exists():
            self.skipTest("action-phase artifact not present")
        self.sources = div.load_action_phase_sources()

    def test_loads_full_cohort(self):
        self.assertEqual(self.sources.surface.shape, (1570, 3))
        self.assertEqual(self.sources.internal.shape, (1570, 3))
        self.assertEqual(self.sources.labels.shape, (1570,))

    def test_divergence_and_eval_are_well_formed(self):
        d = div.per_row_divergence(self.sources)
        self.assertEqual(d.shape, (1570,))
        self.assertTrue(np.all(d >= 0.0) and np.all(d <= 1.0))
        result = div.evaluate_divergence_flags_error(self.sources, permutations=500)
        self.assertEqual(result["n_error"] + result["n_correct"], 1570)
        self.assertGreaterEqual(result["permutation_p_value"], 0.0)
        self.assertLessEqual(result["permutation_p_value"], 1.0)
        self.assertGreaterEqual(result["error_detection_auc"], 0.0)
        self.assertLessEqual(result["error_detection_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
