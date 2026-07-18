#!/usr/bin/env python3
"""Focused tests for the frozen N20 learned action-layer readout."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module(
    "analyze_swe_action_layer_readout",
    ROOT / "scripts/analyze_swe_action_layer_readout.py",
)


def feature_row(repository_index: int, class_index: int) -> dict[str, object]:
    feature = np.zeros(96, dtype=np.float64)
    for layer_index in range(24):
        feature[layer_index * 4 + class_index] = 2.5
        feature[layer_index * 4 + ((class_index + repository_index + 1) % 4)] += 0.05
    return {
        "row_id": f"repo-{repository_index}-class-{class_index}",
        "task_id": f"repo-{repository_index}__task",
        "repo": f"owner-{repository_index}/repo",
        "cohort_id": "development" if repository_index < 4 else "replication",
        "label": MODULE.CLASS_IDS[class_index],
        "task_request_index": class_index + 1,
        "checkpoint_ordinal": class_index,
        "feature": feature.tolist(),
        "band_scores": [float(class_index == item) for item in range(4)],
    }


def prediction(row: dict[str, object], predicted_index: int) -> dict[str, object]:
    probabilities = [0.05] * 4
    probabilities[predicted_index] = 0.85
    return {
        **{
            key: row[key]
            for key in (
                "row_id",
                "task_id",
                "repo",
                "cohort_id",
                "label",
                "task_request_index",
                "checkpoint_ordinal",
            )
        },
        "class_ids": list(MODULE.CLASS_IDS),
        "probabilities": probabilities,
        "prediction": MODULE.CLASS_IDS[predicted_index],
    }


class ActionLayerReadoutTests(unittest.TestCase):
    def test_frozen_protocol_and_upstream_hashes_validate(self):
        action_path = ROOT / "configs/swe_action_layer_readout_protocol.json"
        behavioral_path = ROOT / "configs/swe_behavioral_readout_protocol.json"
        transport_path = ROOT / "configs/swe_next_token_transport_protocol.json"
        result = MODULE.validate_protocol(
            json.loads(action_path.read_bytes()),
            behavioral_protocol_sha256=MODULE.sha256_file(behavioral_path),
            transport_protocol_sha256=MODULE.sha256_file(transport_path),
        )
        self.assertEqual(result["class_ids"], list(MODULE.CLASS_IDS))
        self.assertEqual(result["c_grid"], [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0])
        self.assertEqual(result["pins"]["prompts"], "a84abaf67e991409bb0e192c8a4c84d3251e4aaf4595bc1087244c391c2b698d")

    def test_feature_flattening_is_layer_major_then_class_order(self):
        per_layer = []
        for layer in MODULE.LAYERS:
            per_layer.append(
                {
                    "layer": layer,
                    "class_scores": {
                        class_id: float(layer * 10 + class_index)
                        for class_index, class_id in enumerate(MODULE.CLASS_IDS)
                    },
                }
            )
        raw = {
            "layers": list(MODULE.LAYERS),
            "per_layer": per_layer,
            "band_mean_class_scores": {
                class_id: sum(
                    float(layer * 10 + class_index) for layer in MODULE.LAYERS
                )
                / len(MODULE.LAYERS)
                for class_index, class_id in enumerate(MODULE.CLASS_IDS)
            },
        }
        feature, means = MODULE.flatten_raw_fixed_band(raw)
        self.assertEqual(feature[:8], [240.0, 241.0, 242.0, 243.0, 250.0, 251.0, 252.0, 253.0])
        self.assertEqual(len(feature), 96)
        self.assertEqual(means, [355.0, 356.0, 357.0, 358.0])
        broken = json.loads(json.dumps(raw))
        broken["per_layer"][0]["layer"] = 25
        with self.assertRaisesRegex(ValueError, "out of order"):
            MODULE.flatten_raw_fixed_band(broken)

    def test_balanced_multinomial_fit_converges_and_predicts(self):
        rng = np.random.default_rng(9281)
        supports = [24, 5, 10, 7]
        labels = np.concatenate(
            [np.full(count, class_index) for class_index, count in enumerate(supports)]
        )
        features = rng.normal(0.0, 0.08, size=(len(labels), 96))
        for row_index, class_index in enumerate(labels):
            features[row_index, class_index::4] += 2.0
        model = MODULE.fit_multinomial_lbfgs(
            features, labels, c_value=0.1, maximum_iterations=4000
        )
        probabilities = MODULE.predict_multinomial(model, features)
        self.assertTrue(model["converged"], model)
        self.assertLessEqual(
            model["gradient_infinity_norm"], MODULE.GRADIENT_TOLERANCE
        )
        self.assertGreaterEqual(float(np.mean(np.argmax(probabilities, axis=1) == labels)), 0.95)
        for class_index, class_id in enumerate(MODULE.CLASS_IDS):
            self.assertAlmostEqual(
                model["class_weights"][class_id],
                len(labels) / (4 * supports[class_index]),
            )
        self.assertEqual(probabilities.shape, (len(labels), 4))

    def test_multinomial_objective_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(77)
        features = rng.normal(size=(8, 96))
        labels = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        sample_weights = np.ones(len(labels), dtype=np.float64)
        theta = rng.normal(0.0, 0.02, size=4 * 96 + 4)
        objective, gradient = MODULE._weighted_multinomial_objective(
            theta,
            features,
            labels,
            sample_weights,
            class_count=4,
            c_value=0.3,
        )
        self.assertTrue(np.isfinite(objective))
        epsilon = 1e-6
        for index in (0, 47, 383, 384, 387):
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
                gradient[index], (right_value - left_value) / (2 * epsilon), places=6
            )

    def test_nested_loro_is_complete_and_repository_isolated(self):
        rows = [
            feature_row(repository_index, class_index)
            for repository_index in range(8)
            for class_index in range(4)
        ]
        result = MODULE.nested_leave_one_repository_out(
            rows,
            c_grid=[0.001, 0.1],
            maximum_iterations=1000,
            minimum_valid_inner_folds=5,
        )
        self.assertEqual(result["status"], "available")
        self.assertTrue(result["complete_prediction_coverage"])
        self.assertEqual(len(result["predictions"]), len(rows))
        self.assertEqual(result["fold_count"], 8)
        self.assertEqual(result["successful_fold_count"], 8)
        self.assertEqual(result["metrics"]["balanced_accuracy"], 1.0)
        for fold in result["folds"]:
            self.assertTrue(fold["outer_heldout_labels_used_for_model_selection"] is False)
            self.assertEqual(len(fold["candidate_models"]), 2)

        changed_rows = copy.deepcopy(rows)
        heldout_repository = "owner-0/repo"
        for row in changed_rows:
            if row["repo"] == heldout_repository:
                class_index = MODULE.CLASS_IDS.index(str(row["label"]))
                row["label"] = MODULE.CLASS_IDS[(class_index + 1) % len(MODULE.CLASS_IDS)]
        changed = MODULE.nested_leave_one_repository_out(
            changed_rows,
            c_grid=[0.001, 0.1],
            maximum_iterations=1000,
            minimum_valid_inner_folds=5,
        )
        original_fold = next(
            fold for fold in result["folds"] if fold["heldout_repository"] == heldout_repository
        )
        changed_fold = next(
            fold for fold in changed["folds"] if fold["heldout_repository"] == heldout_repository
        )
        self.assertEqual(original_fold, changed_fold)
        original_probabilities = {
            row["row_id"]: row["probabilities"]
            for row in result["predictions"]
            if row["repo"] == heldout_repository
        }
        changed_probabilities = {
            row["row_id"]: row["probabilities"]
            for row in changed["predictions"]
            if row["repo"] == heldout_repository
        }
        self.assertEqual(original_probabilities, changed_probabilities)

    def test_frozen_baseline_requires_exact_row_coverage(self):
        rows = [feature_row(repository, class_index) for repository in range(2) for class_index in range(4)]
        predictions = [
            prediction(row, MODULE.CLASS_IDS.index(str(row["label"]))) for row in rows
        ]
        normalized = MODULE._normalize_frozen_predictions(predictions, rows)
        self.assertEqual(
            [row["row_id"] for row in normalized], [row["row_id"] for row in rows]
        )
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            MODULE._normalize_frozen_predictions(predictions[:-1], rows)
        with self.assertRaisesRegex(ValueError, "duplicate row IDs"):
            MODULE._normalize_frozen_predictions(predictions + [predictions[0]], rows)

    def test_sensitivity_gate_uses_the_exact_transport_checks(self):
        transport = json.loads(
            (ROOT / "configs/swe_next_token_transport_protocol.json").read_bytes()
        )
        baseline = {
            "final_layer_top1_matches_greedy": True,
            "final_norm_reconstruction": {"within_tolerance": True},
            "final_logits_reconstruction": {
                "top_k_prefix_token_ids_match": True,
                "rms_error": 0.019,
                "max_abs_error": 0.125,
            },
        }
        self.assertTrue(
            MODULE._stable_reconstruction_eligible(
                {"baseline_binding": baseline}, transport
            )
        )
        baseline["final_logits_reconstruction"]["rms_error"] = 0.0200001
        self.assertFalse(
            MODULE._stable_reconstruction_eligible(
                {"baseline_binding": baseline}, transport
            )
        )

    def test_descriptive_checkpoint_and_cohort_slices_are_noninferential(self):
        rows = [feature_row(repository, class_index) for repository in range(8) for class_index in range(4)]
        predictions = [
            prediction(row, MODULE.CLASS_IDS.index(str(row["label"]))) for row in rows
        ]
        result = MODULE._descriptive_slices({"public_jacobian": predictions})
        self.assertEqual(result["status"], "descriptive_only_noninferential")
        self.assertFalse(result["used_for_feature_or_C_selection"])
        self.assertFalse(result["used_for_material_effect_signals"])
        slices = result["methods"]["public_jacobian"]
        self.assertEqual(set(slices["checkpoint_ordinal"]), {"0", "1", "2", "3"})
        self.assertEqual(set(slices["cohort"]), {"development", "replication"})
        self.assertEqual(
            slices["cohort_by_checkpoint_ordinal"]["development::0"]["row_count"],
            4,
        )

    def test_posthoc_ordinal_prior_is_outer_heldout_and_non_decision(self):
        rows = []
        for repository in range(8):
            for ordinal in range(4):
                row = feature_row(repository, ordinal)
                row["label"] = MODULE.CLASS_IDS[ordinal]
                rows.append(row)
        result = MODULE.checkpoint_ordinal_prior_baseline(rows)
        self.assertEqual(result["status"], "POSTHOC_DESCRIPTIVE_ONLY")
        self.assertFalse(result["used_for_paired_primary_comparisons"])
        self.assertFalse(result["used_for_material_effect_signals_or_classification"])
        self.assertEqual(result["metrics"]["balanced_accuracy"], 1.0)
        self.assertEqual(len(result["predictions"]), len(rows))
        for fold in result["folds"]:
            self.assertFalse(fold["heldout_labels_used_to_estimate_priors"])
            self.assertEqual(fold["global_prior_fallback_evaluation_row_count"], 0)
            self.assertEqual(
                sum(fold["global_training_support"].values()),
                fold["training_row_count"],
            )


if __name__ == "__main__":
    unittest.main()
