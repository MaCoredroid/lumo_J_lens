#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import inspect
import unittest
from unittest import mock

import numpy as np

from scripts import swe_task_state_v4_evaluator as evaluator


def contract() -> dict[str, object]:
    seeds = list(evaluator.MODEL_SEEDS_IN_ORDER)
    return {
        "model": {
            "family": "fixed_multiseed_ExtraTreesClassifier_probability_ensemble",
            "seeds": seeds,
            "probability_floor": 1e-6,
            "probability_reduction": (
                "serial_tree_order_within_each_seed_then_arithmetic_mean_across_"
                "seed_estimators_then_floor_and_renormalize"
            ),
            "parameters": dict(evaluator.MODEL_PARAMETERS),
            "fit_execution": dict(evaluator.MODEL_FIT_EXECUTION),
            "prediction_execution": dict(evaluator.MODEL_PREDICTION_EXECUTION),
            "same_hyperparameters_and_seed_order_for_all_variants": True,
        },
        "weighting": {
            "point_estimand": evaluator.POINT_ESTIMAND,
            "training": (
                "restrict_and_renormalize_point_or_bayesian_base_weights_to_the_"
                "current_training_split_then_apply_split_local_exact_three_class_"
                "rebalance"
            ),
            "calibration_threshold_and_evaluation": (
                "restrict_and_renormalize_point_or_bayesian_base_weights_without_"
                "class_rebalance"
            ),
            "point_weights_are_computed_once_on_all_known_development_rows_then_restricted_per_split": True,
            "bayesian_draw_weights_are_computed_once_on_all_known_development_rows_then_restricted_per_split": True,
            "same_row_weights_across_all_matched_variants": True,
            "prevalidated_training_weight_fit_transport": (
                "validate_positive_finite_float64_near_unit_mass_and_near_one_third_"
                "class_mass_then_preserve_every_float64_bit_without_second_"
                "normalization"
            ),
            "prevalidated_crossfit_base_weight_unit_mass_absolute_tolerance": 1e-12,
            "prevalidated_training_weight_unit_mass_absolute_tolerance": 1e-12,
            "prevalidated_crossfit_base_weight_transport": (
                "validate_positive_finite_float64_and_near_unit_mass_then_preserve_"
                "every_float64_bit_before_inner_split_restriction"
            ),
            "unknown_current_actions_excluded_from_fit_and_metric_weights": True,
        },
        "nested": {
            "outer_algorithm": "leave_one_repository_out",
            "inner_algorithm": "leave_one_repository_out_within_outer_training",
            "outer_heldout_labels_used_for_fit_calibration_or_threshold_selection": False,
            "minimum_inner_repositories": 5,
            "same_outer_and_inner_folds_weights_seed_order_and_hyperparameters_across_variants": True,
        },
        "forecast_pool": {
            "candidate": "fixed_geometric_log_opinion_pool",
            "reference": "sequence_logit_raw_probability",
            "candidate_logit_weight": 0.80,
            "candidate_logit_j_weight": 0.20,
            "selection": "none_fixed_before_fresh_development_selection",
            "normalization": "log_space_then_exact_row_normalization",
            "shared_action_source": evaluator.REFERENCE_PROCEDURE,
        },
        "decision": {
            "edit_offset_grid": [0.0],
            "check_or_finish_offset_grid": [0.0],
            "recall_floors": {
                "inspect": 0.75,
                "edit": 0.65,
                "check_or_finish": 0.75,
            },
            "balanced_accuracy_minimum": 0.75,
        },
        "calibration": {"temperature_grid": [1.0, 0.5, 2.0]},
        "abstention": {
            "threshold_grid": [0.8, 0.0],
            "accepted_accuracy_minimum": 0.86,
            "coverage_minimum": 0.70,
            "minimum_accepted_rows_per_true_class": 2,
        },
        "metrics": {"ece_bins": 10},
    }


def feature_mapping(signal: int) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for variant in evaluator.BASE_VARIANTS:
        values = np.zeros(evaluator.FEATURES.VARIANT_WIDTHS[variant], dtype=np.float64)
        values[0] = float(signal)
        result[variant] = values.tolist()
    return result


def synthetic_rows(*, repositories: int = 6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repository_index in range(repositories):
        repo = f"owner/repo-{repository_index}"
        for class_index, class_id in enumerate(evaluator.CLASSES):
            for repeat in range(2):
                row_id = f"r{repository_index}-{class_id}-{repeat}"
                rows.append(
                    {
                        "row_id": row_id,
                        "task_id": f"task-{repository_index}-{class_id}-{repeat}",
                        "repo": repo,
                        "cohort_id": "synthetic",
                        "task_request_index": 1,
                        "checkpoint_ordinal": 1,
                        "source_action_label_status": "available",
                        "source_action_class_id": (
                            "validate" if class_id == "check_or_finish" else class_id
                        ),
                        "label_status": "available",
                        "label": class_id,
                        "metric_evaluable": True,
                        "auxiliary_diagnostics": {},
                        "signal_index": class_index,
                        "features": feature_mapping(class_index),
                    }
                )
        rows.append(
            {
                "row_id": f"r{repository_index}-unknown",
                "task_id": f"task-{repository_index}-unknown",
                "repo": repo,
                "cohort_id": "synthetic",
                "task_request_index": 1,
                "checkpoint_ordinal": 1,
                "source_action_label_status": "missing",
                "source_action_class_id": None,
                "label_status": "unknown_current_action",
                "label": None,
                "metric_evaluable": False,
                "auxiliary_diagnostics": {},
                "signal_index": repository_index % len(evaluator.CLASSES),
                "features": feature_mapping(repository_index % len(evaluator.CLASSES)),
            }
        )
    return rows


def probability_for(signal: int, confidence: float) -> np.ndarray:
    remainder = (1.0 - confidence) / 2.0
    result = np.full(len(evaluator.CLASSES), remainder, dtype=np.float64)
    result[signal] = confidence
    return result


class FakeMatchedFitter:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []
        self.training_weight_calls: list[np.ndarray] = []

    def __call__(
        self,
        training_rows,
        evaluation_rows,
        *,
        contract,
        training_base_weights,
    ):
        del contract
        training_ids = [str(row["row_id"]) for row in training_rows]
        evaluation_ids = [str(row["row_id"]) for row in evaluation_rows]
        if set(training_ids) & set(evaluation_ids):
            raise AssertionError("synthetic fitter observed leakage")
        self.calls.append((training_ids, evaluation_ids))
        self.training_weight_calls.append(
            np.array(training_base_weights, dtype=np.float64, copy=True)
        )
        confidences = {
            "history_only": 0.55,
            "sequence_j": 0.65,
            "sequence_logit": 0.70,
            "sequence_logit_j": 0.82,
        }
        probabilities = {
            variant: np.asarray(
                [
                    probability_for(int(row["signal_index"]), confidence)
                    for row in evaluation_rows
                ],
                dtype=np.float64,
            )
            for variant, confidence in confidences.items()
        }
        weight_hash = evaluator.V3.sha256_bytes(
            np.asarray(training_base_weights, dtype="<f8").tobytes(order="C")
        )
        return probabilities, {
            "shared": {"weight_float64_sha256": weight_hash}
        }


class V4EvaluatorTests(unittest.TestCase):
    def test_contract_is_canonical_and_fail_closed(self) -> None:
        normalized = evaluator.validate_contract(contract())
        self.assertEqual(normalized["model"]["seeds"], list(evaluator.MODEL_SEEDS_IN_ORDER))
        self.assertEqual(normalized["model"]["parameters"], evaluator.MODEL_PARAMETERS)
        self.assertEqual(normalized["model"]["fit_execution"]["worker_count"], 20)
        self.assertEqual(
            normalized["forecast_pool"],
            {
                "candidate": "fixed_geometric_log_opinion_pool",
                "reference": "sequence_logit_raw_probability",
                "candidate_logit_weight": 0.80,
                "candidate_logit_j_weight": 0.20,
                "selection": "none_fixed_before_fresh_development_selection",
                "normalization": "log_space_then_exact_row_normalization",
                "shared_action_source": evaluator.REFERENCE_PROCEDURE,
            },
        )
        self.assertEqual(normalized["calibration"]["temperature_grid"], [0.5, 1.0, 2.0])
        self.assertEqual(normalized["abstention"]["threshold_grid"], [0.0, 0.8])
        self.assertEqual(normalized["abstention"]["accepted_accuracy_minimum"], 0.86)

        wrong_pool_weight = contract()
        wrong_pool_weight["forecast_pool"]["candidate_logit_j_weight"] = 0.25  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "forecast-pool semantics"):
            evaluator.validate_contract(wrong_pool_weight)

        wrong_inner_accuracy = contract()
        wrong_inner_accuracy["abstention"]["accepted_accuracy_minimum"] = 0.85  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "abstention, or metric constants"):
            evaluator.validate_contract(wrong_inner_accuracy)

        wrong_workers = contract()
        wrong_workers["model"]["fit_execution"]["worker_count"] = 7  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "worker count"):
            evaluator.validate_contract(wrong_workers)

        wrong_seeds = contract()
        wrong_seeds["model"]["seeds"] = [11, 13]  # type: ignore[index]
        wrong_seeds["model"]["fit_execution"]["worker_count"] = 8  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "20-estimator schedule"):
            evaluator.validate_contract(wrong_seeds)

        wrong_model = contract()
        wrong_model["model"]["parameters"]["min_samples_leaf"] = 1  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "model parameters"):
            evaluator.validate_contract(wrong_model)

        no_total_fallback = contract()
        no_total_fallback["abstention"]["threshold_grid"] = [1.0]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "must contain zero"):
            evaluator.validate_contract(no_total_fallback)

    def test_fixed_j_forecast_pool_and_shared_logit_action_stay_separate(self) -> None:
        rows = synthetic_rows(repositories=1)[:-1]
        labels = [str(row["label"]) for row in rows]
        raw_logit = np.asarray(
            [probability_for(evaluator.CLASSES.index(label), 0.70) for label in labels]
        )
        raw_hybrid = np.asarray(
            [probability_for(evaluator.CLASSES.index(label), 0.82) for label in labels]
        )
        raw = {
            variant: raw_logit.copy() for variant in evaluator.BASE_VARIANTS
        }
        raw["sequence_logit_j"] = raw_hybrid
        normalized = evaluator.validate_contract(contract())
        weights = np.ones(len(rows), dtype=np.float64)
        shared_action = evaluator.select_shared_action_settings(
            raw,
            rows,
            weights,
            contract=normalized,
        )
        primary_settings = evaluator.select_procedure_settings(
            raw,
            rows,
            weights,
            evaluator.PRIMARY_PROCEDURE,
            contract=normalized,
            shared_action_settings=shared_action,
        )
        reference_settings = evaluator.select_procedure_settings(
            raw,
            rows,
            weights,
            evaluator.REFERENCE_PROCEDURE,
            contract=normalized,
            shared_action_settings=shared_action,
        )
        shared_hash = evaluator.V3.canonical_json_sha256(shared_action)
        self.assertEqual(shared_action["alpha"], 0.0)
        self.assertEqual(primary_settings["decision"], shared_action)
        self.assertEqual(reference_settings["decision"], shared_action)
        self.assertEqual(
            primary_settings["shared_action_settings_sha256"], shared_hash
        )
        self.assertEqual(
            reference_settings["shared_action_settings_sha256"], shared_hash
        )
        self.assertEqual(primary_settings["calibration"]["temperature"], 0.5)

        primary_outputs = evaluator.apply_procedure_settings(
            raw,
            primary_settings,
            evaluator.PRIMARY_PROCEDURE,
            contract=normalized,
        )
        reference_outputs = evaluator.apply_procedure_settings(
            raw,
            reference_settings,
            evaluator.REFERENCE_PROCEDURE,
            contract=normalized,
        )
        expected_pool = evaluator.CALIBRATION.geometric_pool_probabilities(
            raw_logit,
            raw_hybrid,
            alpha=evaluator.FORECAST_POOL_J_WEIGHT,
        )
        np.testing.assert_array_equal(
            primary_outputs["forecast_raw_probability_p"], expected_pool
        )
        np.testing.assert_array_equal(
            primary_outputs["decision_raw_probability_r"], raw_logit
        )
        np.testing.assert_array_equal(
            reference_outputs["forecast_raw_probability_p"], raw_logit
        )
        np.testing.assert_array_equal(
            primary_outputs["decision_raw_probability_r"],
            reference_outputs["decision_raw_probability_r"],
        )
        np.testing.assert_array_equal(
            primary_outputs["decision_probability_d"],
            reference_outputs["decision_probability_d"],
        )
        self.assertFalse(
            np.array_equal(
                primary_outputs["forecast_raw_probability_p"],
                primary_outputs["forecast_probability_q"],
            )
        )
        # With zero offsets, the shared action d is exactly ordinary-logit r;
        # J enters only the candidate's fixed forecast pool and calibrated q.
        np.testing.assert_allclose(
            primary_outputs["decision_probability_d"], raw_logit
        )

        mutated_raw = {variant: values.copy() for variant, values in raw.items()}
        mutated_raw["sequence_logit_j"] = np.asarray(
            [probability_for(evaluator.CLASSES.index(label), 0.55) for label in labels]
        )
        mutated_outputs = evaluator.apply_procedure_settings(
            mutated_raw,
            primary_settings,
            evaluator.PRIMARY_PROCEDURE,
            contract=normalized,
        )
        self.assertFalse(
            np.array_equal(
                mutated_outputs["forecast_probability_q"],
                primary_outputs["forecast_probability_q"],
            )
        )
        np.testing.assert_array_equal(
            mutated_outputs["decision_raw_probability_r"],
            primary_outputs["decision_raw_probability_r"],
        )
        np.testing.assert_array_equal(
            mutated_outputs["decision_probability_d"],
            primary_outputs["decision_probability_d"],
        )

        tampered = copy.deepcopy(primary_settings)
        tampered["decision"]["recall_floors"]["edit"] = 0.64
        with self.assertRaisesRegex(ValueError, "decision settings contract identity"):
            evaluator.apply_procedure_settings(
                raw,
                tampered,
                evaluator.PRIMARY_PROCEDURE,
                contract=normalized,
            )
        tampered_hash = copy.deepcopy(primary_settings)
        tampered_hash["shared_action_settings_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "shared-action identity"):
            evaluator.apply_procedure_settings(
                raw,
                tampered_hash,
                evaluator.PRIMARY_PROCEDURE,
                contract=normalized,
            )
        signature = inspect.signature(evaluator.apply_procedure_settings)
        self.assertNotIn("labels", signature.parameters)
        self.assertNotIn("weights", signature.parameters)

    def test_nested_loro_predicts_unknown_rows_and_never_trains_on_heldout_repo(self) -> None:
        rows = synthetic_rows()
        normalized = evaluator.validate_contract(contract())
        fake = FakeMatchedFitter()
        with mock.patch.object(
            evaluator, "_fit_predict_all_base_variants", side_effect=fake
        ):
            result = evaluator.nested_leave_one_repository_out(
                rows, contract=normalized
            )

        self.assertEqual(result["outer_fold_count"], 6)
        self.assertEqual(
            result["ordered_base_variant_seed_schedule"],
            [
                {"variant": variant, "seed": seed}
                for variant in evaluator.BASE_VARIANTS
                for seed in evaluator.MODEL_SEEDS_IN_ORDER
            ],
        )
        self.assertEqual(
            result["evaluator_contract_sha256"],
            evaluator.V3.canonical_json_sha256(normalized),
        )
        self.assertTrue(result["unknown_current_actions_received_predictions"])
        self.assertFalse(result["outer_heldout_labels_used_for_fit_or_selection"])
        self.assertTrue(result["candidate_reference_shared_action_policy"])
        self.assertTrue(
            result["candidate_reference_decision_raw_r_exactly_equal"]
        )
        self.assertTrue(result["candidate_reference_decision_d_exactly_equal"])
        for fold in result["folds"]:
            self.assertTrue(fold["inner_and_heldout_row_ids_disjoint"])
            self.assertFalse(fold["heldout_labels_used_for_fit_or_selection"])
            self.assertTrue(
                fold["candidate_reference_decision_raw_r_exactly_equal"]
            )
            self.assertTrue(fold["candidate_reference_decision_d_exactly_equal"])
            self.assertEqual(
                fold["settings"][evaluator.PRIMARY_PROCEDURE]["decision"],
                fold["settings"][evaluator.REFERENCE_PROCEDURE]["decision"],
            )
            self.assertEqual(
                fold["shared_action_settings"],
                fold["settings"][evaluator.PRIMARY_PROCEDURE]["decision"],
            )
            self.assertEqual(
                fold["shared_action_settings_sha256"],
                evaluator.V3.canonical_json_sha256(
                    fold["shared_action_settings"]
                ),
            )
            self.assertEqual(
                fold["settings"][evaluator.PRIMARY_PROCEDURE][
                    "shared_action_settings_sha256"
                ],
                fold["settings"][evaluator.REFERENCE_PROCEDURE][
                    "shared_action_settings_sha256"
                ],
            )
            self.assertTrue(
                all(
                    inner["seed_order"] == list(evaluator.MODEL_SEEDS_IN_ORDER)
                    for inner in fold["inner_folds"]
                )
            )
        self.assertGreater(len(fake.calls), result["outer_fold_count"])
        for training_ids, evaluation_ids in fake.calls:
            self.assertTrue(set(training_ids).isdisjoint(evaluation_ids))

        primary = result["results"][evaluator.PRIMARY_PROCEDURE]
        self.assertEqual(primary["inference_prediction_count"], len(rows))
        self.assertEqual(primary["unknown_action_prediction_count"], 6)
        self.assertAlmostEqual(primary["metrics"]["accuracy"], 1.0)
        self.assertAlmostEqual(primary["metrics"]["balanced_accuracy"], 1.0)
        self.assertIn("forecast_raw_p_float64_sha256", primary)
        self.assertIn("decision_raw_r_float64_sha256", primary)
        recorded_forecast_p = np.asarray(
            [
                [
                    row["forecast_raw_probabilities_p"][class_id]
                    for class_id in evaluator.CLASSES
                ]
                for row in primary["predictions"]
            ],
            dtype="<f8",
        )
        recorded_decision_r = np.asarray(
            [
                [
                    row["decision_raw_probabilities_r"][class_id]
                    for class_id in evaluator.CLASSES
                ]
                for row in primary["predictions"]
            ],
            dtype="<f8",
        )
        self.assertEqual(
            primary["forecast_raw_p_float64_sha256"],
            evaluator.V3.sha256_bytes(recorded_forecast_p.tobytes(order="C")),
        )
        self.assertEqual(
            primary["decision_raw_r_float64_sha256"],
            evaluator.V3.sha256_bytes(recorded_decision_r.tobytes(order="C")),
        )
        reference = result["results"][evaluator.REFERENCE_PROCEDURE]
        recorded_reference_r = np.asarray(
            [
                [
                    row["decision_raw_probabilities_r"][class_id]
                    for class_id in evaluator.CLASSES
                ]
                for row in reference["predictions"]
            ],
            dtype="<f8",
        )
        np.testing.assert_array_equal(recorded_decision_r, recorded_reference_r)
        self.assertEqual(
            primary["decision_raw_r_float64_sha256"],
            reference["decision_raw_r_float64_sha256"],
        )
        self.assertEqual(
            primary["decision_d_float64_sha256"],
            reference["decision_d_float64_sha256"],
        )
        self.assertEqual(
            result["shared_decision_raw_r_float64_sha256"],
            primary["decision_raw_r_float64_sha256"],
        )
        self.assertEqual(
            result["shared_decision_d_float64_sha256"],
            primary["decision_d_float64_sha256"],
        )
        self.assertEqual(
            primary["inference_acceptance_fraction_denominator"],
            "all_stable_feature_complete_predictions",
        )
        self.assertEqual(
            primary["metrics"]["selected_coverage_denominator"],
            "known_current_action_metric_rows_only_not_all_stable_emissions",
        )
        unknown_predictions = [
            row for row in primary["predictions"] if not row["metric_evaluable"]
        ]
        self.assertEqual(len(unknown_predictions), 6)
        self.assertTrue(
            all(row["predicted_class"] in evaluator.CLASSES for row in unknown_predictions)
        )
        self.assertTrue(
            all("forecast_raw_probabilities_p" in row for row in unknown_predictions)
        )
        self.assertTrue(
            all("decision_raw_probabilities_r" in row for row in unknown_predictions)
        )
        self.assertTrue(
            result["full_development_promotion"][
                "eligible_on_full_development_selection"
            ]
        )
        self.assertFalse(
            result["full_development_promotion"]["fallback_blocks_promotion"]
        )
        self.assertTrue(
            result["full_development_promotion"][
                "candidate_abstention_selected_under_floors"
            ]
        )
        self.assertTrue(
            result["full_development_promotion"][
                "reference_abstention_selected_under_floors"
            ]
        )
        self.assertTrue(
            result["full_development_promotion"][
                "both_abstention_branches_selected_under_floors"
            ]
        )
        full_primary = result["full_development_settings"][
            evaluator.PRIMARY_PROCEDURE
        ]
        self.assertTrue(full_primary["decision"]["selected_under_floors"])
        self.assertTrue(full_primary["abstention"]["selected_under_floors"])
        self.assertEqual(full_primary["decision"]["alpha"], 0.0)
        full_reference = result["full_development_settings"][
            evaluator.REFERENCE_PROCEDURE
        ]
        self.assertEqual(full_primary["decision"], full_reference["decision"])
        self.assertEqual(
            full_primary["decision"],
            result["full_development_shared_action_settings"],
        )
        self.assertEqual(
            full_primary["shared_action_settings_sha256"],
            full_reference["shared_action_settings_sha256"],
        )
        self.assertEqual(
            result["full_development_shared_action_settings_sha256"],
            evaluator.V3.canonical_json_sha256(
                result["full_development_shared_action_settings"]
            ),
        )
        self.assertTrue(
            result["full_development_promotion"][
                "candidate_reference_shared_action_identity_passed"
            ]
        )
        known_id_hash = evaluator.V3.canonical_json_sha256(
            [str(row["row_id"]) for row in evaluator.known_rows(rows)]
        )
        for record in result["full_development_oof_raw_probabilities"].values():
            self.assertEqual(record["known_row_ids_sha256"], known_id_hash)

    def test_per_repository_metrics_preserve_partial_support_diagnostics(self) -> None:
        rows = [
            {
                "row_id": "inspect-row",
                "task_id": "inspect-task",
                "repo": "owner/partial",
                "label": "inspect",
            },
            {
                "row_id": "edit-row",
                "task_id": "edit-task",
                "repo": "owner/partial",
                "label": "edit",
            },
        ]
        probabilities = np.asarray(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=np.float64
        )
        result = evaluator._per_repository_metrics(
            rows,
            probabilities,
            probabilities,
            np.asarray([True, False], dtype=np.bool_),
            [1.0, 1.0],
            ece_bins=5,
        )["owner/partial"]

        self.assertEqual(result["status"], "partial_class_support")
        self.assertEqual(
            result["class_support"],
            {"inspect": 1, "edit": 1, "check_or_finish": 0},
        )
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertIsNotNone(result["multiclass_negative_log_likelihood"])
        self.assertIsNotNone(result["top_label_ece"])
        self.assertIsNone(result["recall_check_or_finish"])
        self.assertIsNone(result["balanced_accuracy"])
        self.assertIsNone(
            result["per_true_class_accepted_coverage"]["check_or_finish"]
        )

    def test_crossfit_transports_restricted_master_weights_without_index_drift(self) -> None:
        rows = evaluator.known_rows(synthetic_rows())
        raw_weights = np.arange(1, len(rows) + 1, dtype=np.float64)
        master_weights = raw_weights / raw_weights.sum(dtype=np.float64)
        normalized = evaluator.validate_contract(contract())
        fake = FakeMatchedFitter()
        with mock.patch.object(
            evaluator, "_fit_predict_all_base_variants", side_effect=fake
        ):
            evaluator.crossfit_raw_probabilities(
                rows,
                contract=normalized,
                base_weights=master_weights,
            )

        index_by_id = {str(row["row_id"]): index for index, row in enumerate(rows)}
        self.assertEqual(len(fake.calls), 6)
        for (training_ids, _), observed in zip(
            fake.calls, fake.training_weight_calls, strict=True
        ):
            indices = [index_by_id[row_id] for row_id in training_ids]
            expected = evaluator.V3.restrict_base_weights(
                rows, master_weights, indices
            )
            np.testing.assert_array_equal(observed, expected)

    def test_unknown_row_roles_are_strict_and_full_fallback_blocks_promotion(self) -> None:
        normalized = evaluator.validate_contract(contract())
        invalid_truthiness = synthetic_rows()
        invalid_truthiness[-1]["metric_evaluable"] = 0
        with self.assertRaisesRegex(ValueError, "metric_evaluable must be boolean"):
            evaluator.nested_leave_one_repository_out(
                invalid_truthiness, contract=normalized
            )

        invalid_numpy_boolean = synthetic_rows()
        invalid_numpy_boolean[0]["metric_evaluable"] = np.bool_(True)
        with self.assertRaisesRegex(ValueError, "metric_evaluable must be boolean"):
            evaluator.nested_leave_one_repository_out(
                invalid_numpy_boolean, contract=normalized
            )

        invalid_unknown_label = synthetic_rows()
        invalid_unknown_label[-1]["label"] = "inspect"
        with self.assertRaisesRegex(ValueError, "inconsistent label state"):
            evaluator.nested_leave_one_repository_out(
                invalid_unknown_label, contract=normalized
            )

        invalid_known_status = synthetic_rows()
        invalid_known_status[0]["label_status"] = "unknown_current_action"
        with self.assertRaisesRegex(ValueError, "known row"):
            evaluator.nested_leave_one_repository_out(
                invalid_known_status, contract=normalized
            )

        class AllInspectFitter(FakeMatchedFitter):
            def __call__(
                self,
                training_rows,
                evaluation_rows,
                *,
                contract,
                training_base_weights,
            ):
                result, diagnostics = super().__call__(
                    training_rows,
                    evaluation_rows,
                    contract=contract,
                    training_base_weights=training_base_weights,
                )
                return {
                    variant: np.asarray(
                        [probability_for(0, 0.70) for _ in evaluation_rows]
                    )
                    for variant in evaluator.BASE_VARIANTS
                }, diagnostics

        with mock.patch.object(
            evaluator,
            "_fit_predict_all_base_variants",
            side_effect=AllInspectFitter(),
        ):
            result = evaluator.nested_leave_one_repository_out(
                synthetic_rows(), contract=normalized
            )
        promotion = result["full_development_promotion"]
        self.assertFalse(promotion["action_rule_selected_under_floors"])
        self.assertTrue(promotion["fallback_blocks_promotion"])
        self.assertFalse(promotion["eligible_on_full_development_selection"])

    def test_own_outer_label_cannot_change_that_rows_prediction(self) -> None:
        rows = synthetic_rows()
        normalized = evaluator.validate_contract(contract())
        target_id = "r0-inspect-0"

        def run(input_rows):
            fake = FakeMatchedFitter()
            with mock.patch.object(
                evaluator, "_fit_predict_all_base_variants", side_effect=fake
            ):
                return evaluator.nested_leave_one_repository_out(
                    input_rows, contract=normalized
                )

        before = run(rows)
        mutated = copy.deepcopy(rows)
        for row in mutated:
            if row["row_id"] == target_id:
                row["label"] = "edit"
                row["source_action_class_id"] = "edit"
        after = run(mutated)

        def prediction(result):
            return next(
                row
                for row in result["results"][evaluator.PRIMARY_PROCEDURE][
                    "predictions"
                ]
                if row["row_id"] == target_id
            )

        left = prediction(before)
        right = prediction(after)
        self.assertEqual(left["forecast_probabilities_q"], right["forecast_probabilities_q"])
        self.assertEqual(left["decision_probabilities_d"], right["decision_probabilities_d"])
        self.assertEqual(left["accepted"], right["accepted"])

    def test_v3_dependency_pin_and_matrix_validation(self) -> None:
        self.assertEqual(
            hashlib.sha256(evaluator.V3_ANALYZER_PATH.read_bytes()).hexdigest(),
            evaluator.V3_ANALYZER_SHA256,
        )
        evaluator._authenticate_v3_source()
        with mock.patch.object(
            type(evaluator.V3_ANALYZER_PATH), "read_bytes", return_value=b"changed"
        ):
            with self.assertRaisesRegex(RuntimeError, "byte identity changed"):
                evaluator._authenticate_v3_source()

        rows = synthetic_rows(repositories=1)
        matrix = evaluator.matrix_for(rows, "sequence_logit_j")
        self.assertEqual(
            matrix.shape,
            (len(rows), evaluator.FEATURES.VARIANT_WIDTHS["sequence_logit_j"]),
        )
        broken = copy.deepcopy(rows)
        broken[0]["features"]["sequence_logit_j"] = [0.0]  # type: ignore[index]
        with self.assertRaisesRegex(
            ValueError, "feature matrix (?:is invalid|must be numeric)"
        ):
            evaluator.matrix_for(broken, "sequence_logit_j")


if __name__ == "__main__":
    unittest.main()
