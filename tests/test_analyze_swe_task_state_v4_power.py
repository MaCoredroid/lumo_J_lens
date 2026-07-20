#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import analyze_swe_task_state_v4_power as power


class V4PairedProperPowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_config = json.loads(power.DEFAULT_CONFIG.read_text())
        cls.raw_artifact = json.loads(power.DEFAULT_SOURCE_ARTIFACT.read_text())

    def test_frozen_config_and_actual_artifact_bind_exactly(self) -> None:
        self.assertEqual(
            power._sha256_file(power.DEFAULT_CONFIG),
            power.POWER_CONFIG_SHA256,
        )
        self.assertEqual(
            power._sha256_file(power.DEFAULT_SOURCE_ARTIFACT),
            power.SOURCE_ARTIFACT_SHA256,
        )
        config = power.validate_power_config(self.raw_config)
        validated = power.validate_source_artifact(self.raw_artifact, config)
        profiles, summary = power.build_task_loss_profiles(
            validated["candidate_rows"],
            validated["reference_rows"],
        )

        self.assertEqual(sum(map(len, profiles.values())), 60)
        self.assertEqual(summary["known_row_count"], 1570)
        self.assertEqual(summary["repository_count"], 10)
        self.assertEqual(
            summary["repository_improving_sign_count"],
            {
                "multiclass_negative_log_likelihood": 10,
                "multiclass_brier": 10,
            },
        )
        self.assertAlmostEqual(
            summary["paired_point_differences"][
                "multiclass_negative_log_likelihood"
            ],
            -0.006178540156555723,
            places=15,
        )
        self.assertAlmostEqual(
            summary["paired_point_differences"]["multiclass_brier"],
            -0.0021992920266340477,
            places=15,
        )
        self.assertAlmostEqual(
            summary["empirical_marginal_p90_adverse_profile"][
                "multiclass_negative_log_likelihood"
            ],
            0.0022797267620328285,
            places=15,
        )

    def test_config_rejects_power_downgrades_and_policy_changes(self) -> None:
        tampered = copy.deepcopy(self.raw_config)
        tampered["resampling_contract"]["outer_cohorts"] = 100
        with self.assertRaisesRegex(ValueError, "resampling identity"):
            power.validate_power_config(tampered)

        tampered = copy.deepcopy(self.raw_config)
        tampered["future_allocation_in_order"]["django/django"] = 176
        with self.assertRaisesRegex(ValueError, "future allocation"):
            power.validate_power_config(tampered)

        tampered = copy.deepcopy(self.raw_config)
        tampered["decision_rule"]["minimum_acceptable_joint_power"] = 0.5
        with self.assertRaisesRegex(ValueError, "decision rule"):
            power.validate_power_config(tampered)

        tampered = copy.deepcopy(self.raw_config)
        tampered["data_policy"]["reserved_validation_allowed"] = True
        with self.assertRaisesRegex(ValueError, "data policy"):
            power.validate_power_config(tampered)

    def test_artifact_rejects_q_or_shared_d_tampering(self) -> None:
        config = power.validate_power_config(self.raw_config)
        nested = self.raw_artifact["nested_design_evaluation"]
        candidate = nested["results"][power.PRIMARY_PROCEDURE]["predictions"][0]

        q = candidate["forecast_probabilities_q"]
        original_q = dict(q)
        q_epsilon = min(float(q["inspect"]), float(q["edit"])) * 0.1
        q["inspect"] += q_epsilon
        q["edit"] -= q_epsilon
        try:
            with self.assertRaisesRegex(ValueError, "q bytes"):
                power.validate_source_artifact(self.raw_artifact, config)
        finally:
            q.clear()
            q.update(original_q)

        d = candidate["decision_probabilities_d"]
        original_d = dict(d)
        d_epsilon = min(float(d["inspect"]), float(d["edit"])) * 0.1
        d["inspect"] += d_epsilon
        d["edit"] -= d_epsilon
        try:
            with self.assertRaisesRegex(ValueError, "decision d differs"):
                power.validate_source_artifact(self.raw_artifact, config)
        finally:
            d.clear()
            d.update(original_d)

    @staticmethod
    def _uniform_profiles() -> dict[str, np.ndarray]:
        return {
            repository: np.asarray(
                [
                    [-0.01, -0.005],
                    [-0.01, -0.005],
                ],
                dtype=np.float64,
            )
            for repository in power.EXPECTED_SOURCE_REPOSITORIES
        }

    def test_small_paired_simulation_is_exactly_reproducible(self) -> None:
        profiles = self._uniform_profiles()
        originals = {
            key: value.copy() for key, value in profiles.items()
        }
        arguments = {
            "seed": 1234,
            "outer_cohorts": 20,
            "inner_draws": 40,
            "batch_size": 5,
            "upper_quantile": 0.975,
        }
        first = power.simulate_paired_proper_power(
            profiles,
            power.EXPECTED_ALLOCATION,
            **arguments,
        )
        second = power.simulate_paired_proper_power(
            profiles,
            power.EXPECTED_ALLOCATION,
            **arguments,
        )

        self.assertEqual(first, second)
        self.assertEqual(list(first["scenarios"]), list(power.SCENARIO_ORDER))
        for scenario in power.SCENARIO_ORDER:
            result = first["scenarios"][scenario]
            self.assertEqual(result["joint_gate"]["power"], 1.0)
            self.assertEqual(result["nll_gate"]["power"], 1.0)
            self.assertEqual(result["brier_gate"]["power"], 1.0)
        for key in profiles:
            np.testing.assert_array_equal(profiles[key], originals[key])

    def test_power_summary_uses_both_strict_upper_gates(self) -> None:
        upper = np.asarray(
            [
                [-0.1, -0.2],
                [-0.1, 0.0],
                [0.0, -0.2],
                [0.1, 0.2],
            ]
        )
        result = power._power_summary(
            upper,
            np.asarray([1.0, 0.8, 0.7, 0.0]),
        )
        self.assertEqual(result["nll_gate"]["power"], 0.5)
        self.assertEqual(result["brier_gate"]["power"], 0.5)
        self.assertEqual(result["joint_gate"]["power"], 0.25)
        self.assertAlmostEqual(
            result["mean_fraction_of_inner_draws_with_both_favorable_signs"],
            0.625,
        )

    def test_temperature_tie_rule_and_small_sensitivity(self) -> None:
        nll = np.asarray([[[0.4, 0.4, 0.4]]], dtype=np.float64)
        selected = power._temperature_indices(nll, [0.5, 1.0, 1.5])
        self.assertEqual(int(selected[0, 0]), 1)

        task = np.asarray(
            [
                [
                    [[0.5, 0.30], [0.4, 0.25]],
                    [[0.49, 0.29], [0.39, 0.24]],
                ],
                [
                    [[0.5, 0.30], [0.4, 0.25]],
                    [[0.49, 0.29], [0.39, 0.24]],
                ],
            ],
            dtype=np.float64,
        )
        profiles = {
            repository: task.copy()
            for repository in power.EXPECTED_SOURCE_REPOSITORIES
        }
        result = power.simulate_temperature_reselection(
            profiles,
            power.EXPECTED_ALLOCATION,
            temperatures=[0.5, 1.0],
            fixed_temperature=1.0,
            seed=987,
            outer_cohorts=10,
            inner_draws=20,
            batch_size=5,
            upper_quantile=0.975,
        )
        self.assertEqual(
            result["modes"]["fixed_temperature"]["joint_gate"]["power"],
            1.0,
        )
        self.assertEqual(
            result["modes"]["independently_reselected_temperature"][
                "joint_gate"
            ]["power"],
            1.0,
        )
        self.assertFalse(result["models_refit"])

    def test_paths_and_no_clobber_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            safe = root / "power.json"
            validated = power.validate_cli_paths(
                config_path=power.DEFAULT_CONFIG,
                source_artifact_path=power.DEFAULT_SOURCE_ARTIFACT,
                output_path=safe,
                output_root=root,
            )
            self.assertEqual(validated["output"], safe)

            power._write_json_no_clobber(safe, {"ok": True})
            with self.assertRaises(FileExistsError):
                power._write_json_no_clobber(safe, {"ok": False})
            with self.assertRaisesRegex(ValueError, "already exists"):
                power.validate_cli_paths(
                    config_path=power.DEFAULT_CONFIG,
                    source_artifact_path=power.DEFAULT_SOURCE_ARTIFACT,
                    output_path=safe,
                    output_root=root,
                )
            with self.assertRaisesRegex(ValueError, "directly under"):
                power.validate_cli_paths(
                    config_path=power.DEFAULT_CONFIG,
                    source_artifact_path=power.DEFAULT_SOURCE_ARTIFACT,
                    output_path=root.parent / "outside.json",
                    output_root=root,
                )
            with self.assertRaisesRegex(ValueError, "reserved or validation"):
                power.validate_cli_paths(
                    config_path=power.DEFAULT_CONFIG,
                    source_artifact_path=power.DEFAULT_SOURCE_ARTIFACT,
                    output_path=root / "validation.json",
                    output_root=root,
                )


if __name__ == "__main__":
    unittest.main()
