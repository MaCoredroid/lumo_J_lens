from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_observable_bootstrap.py"
spec = importlib.util.spec_from_file_location(
    "swe_task_state_v4_observable_bootstrap", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class ObservableBootstrapTests(unittest.TestCase):
    def test_config_is_exact_and_claims_false(self):
        config = module.validate_config(module.load_json(module.CONFIG_PATH))
        self.assertEqual(config["bootstrap"]["draw_count"], 5000)
        self.assertTrue(config["interpretation"]["configured_after_point_results_were_observed"])
        self.assertFalse(config["interpretation"]["multiplicity_adjusted"])
        self.assertTrue(all(value is False for value in config["claim_scope"].values()))

    def test_paired_delta_signs(self):
        probabilities = {
            "candidate": np.asarray([[0.9, 0.1], [0.1, 0.9]]),
            "reference": np.asarray([[0.6, 0.4], [0.4, 0.6]]),
        }
        delta, names = module.paired_delta_matrix(
            probabilities,
            np.asarray([0, 1]),
            [{"candidate": "candidate", "reference": "reference"}],
        )
        self.assertEqual(
            names,
            [
                "candidate__vs__reference::negative_log_likelihood",
                "candidate__vs__reference::multiclass_brier",
                "candidate__vs__reference::correctness",
            ],
        )
        self.assertTrue(np.all(delta[:, :2] < 0.0))
        self.assertTrue(np.all(delta[:, 2] == 0.0))

    def test_hierarchical_bootstrap_is_deterministic_and_paired(self):
        rows = []
        for repository in ("a", "b"):
            for task in ("1", "2"):
                for row in range(3):
                    rows.append(
                        {
                            "repository": repository,
                            "task_id_sha256": f"{repository}{task}",
                            "request_index": row + 1,
                        }
                    )
        delta = np.column_stack(
            [np.arange(len(rows), dtype=np.float64), -np.arange(len(rows), dtype=np.float64)]
        )
        first = module.hierarchical_bootstrap(delta, rows, draw_count=50, seed=7)
        second = module.hierarchical_bootstrap(delta, rows, draw_count=50, seed=7)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first[:, 0], -first[:, 1])

    def test_summary_marks_zero_crossing_and_direction(self):
        draws = np.asarray([[-1.0, 1.0], [-0.5, 2.0], [-0.1, -1.0]])
        summary = module.summarize(
            names=["a::negative_log_likelihood", "b::correctness"],
            point=np.asarray([-0.4, 0.3]),
            draws=draws,
            lower=0.0,
            upper=1.0,
        )
        self.assertFalse(summary["a::negative_log_likelihood"]["interval_contains_zero"])
        self.assertTrue(summary["b::correctness"]["interval_contains_zero"])
        self.assertEqual(summary["b::correctness"]["favorable_direction"], "positive")


if __name__ == "__main__":
    unittest.main()
