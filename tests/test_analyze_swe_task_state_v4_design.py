#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import analyze_swe_task_state_v4_design as design


def nested_fixture() -> dict[str, object]:
    shared_action_settings = {
        "alpha": 0.0,
        "edit_offset": 0.125,
        "check_or_finish_offset": -0.125,
        "selected_under_floors": True,
        "fallback_used": False,
    }
    shared_action_settings_sha256 = design.V3.canonical_json_sha256(
        shared_action_settings
    )
    shared_raw_r_sha256 = "1" * 64
    shared_d_sha256 = "2" * 64
    primary_metrics = {
        "known_action_fraction": 0.96,
        "balanced_accuracy": 0.76,
        "recall_inspect": 0.77,
        "recall_edit": 0.68,
        "recall_check_or_finish": 0.83,
        "selected_accepted_accuracy": 0.86,
        "selected_coverage": 0.72,
        "multiclass_negative_log_likelihood": 0.52,
        "multiclass_brier": 0.30,
        "accuracy": 0.78,
    }
    reference_metrics = {
        **primary_metrics,
        "multiclass_negative_log_likelihood": 0.54,
        "multiclass_brier": 0.315,
    }
    return {
        "candidate_reference_shared_action_policy": True,
        "candidate_reference_decision_raw_r_exactly_equal": True,
        "candidate_reference_decision_d_exactly_equal": True,
        "shared_decision_raw_r_float64_sha256": shared_raw_r_sha256,
        "shared_decision_d_float64_sha256": shared_d_sha256,
        "folds": [
            {
                "candidate_reference_decision_raw_r_exactly_equal": True,
                "candidate_reference_decision_d_exactly_equal": True,
                "shared_action_settings": shared_action_settings,
                "shared_action_settings_sha256": shared_action_settings_sha256,
                "settings": {
                    design.EVALUATOR.PRIMARY_PROCEDURE: {
                        "decision": shared_action_settings,
                        "shared_action_settings_sha256": (
                            shared_action_settings_sha256
                        ),
                    },
                    design.EVALUATOR.REFERENCE_PROCEDURE: {
                        "decision": shared_action_settings,
                        "shared_action_settings_sha256": (
                            shared_action_settings_sha256
                        ),
                    },
                },
            }
        ],
        "results": {
            design.EVALUATOR.PRIMARY_PROCEDURE: {
                "metrics": primary_metrics,
                "decision_raw_r_float64_sha256": shared_raw_r_sha256,
                "decision_d_float64_sha256": shared_d_sha256,
            },
            design.EVALUATOR.REFERENCE_PROCEDURE: {
                "metrics": reference_metrics,
                "decision_raw_r_float64_sha256": shared_raw_r_sha256,
                "decision_d_float64_sha256": shared_d_sha256,
            },
        },
        "full_development_shared_action_settings": shared_action_settings,
        "full_development_shared_action_settings_sha256": (
            shared_action_settings_sha256
        ),
        "full_development_settings": {
            design.EVALUATOR.PRIMARY_PROCEDURE: {
                "decision": shared_action_settings,
                "shared_action_settings_sha256": shared_action_settings_sha256,
                "abstention": {"selected_under_floors": True},
            },
            design.EVALUATOR.REFERENCE_PROCEDURE: {
                "decision": shared_action_settings,
                "shared_action_settings_sha256": shared_action_settings_sha256,
                "abstention": {"selected_under_floors": True},
            },
        },
        "full_development_promotion": {
            "candidate_reference_shared_action_identity_passed": True,
            "action_rule_selected_under_floors": True,
            "candidate_abstention_selected_under_floors": True,
            "reference_abstention_selected_under_floors": True,
            "both_abstention_branches_selected_under_floors": True,
            "fallback_blocks_promotion": False,
            "eligible_on_full_development_selection": True,
        },
    }


def support_fixture() -> dict[str, object]:
    return {
        "stable_prediction_rows": 1600,
        "known_action_rows": 1560,
        "prediction_tasks": 60,
        "prediction_repositories": 10,
        "known_action_tasks": 60,
        "known_action_repositories": 10,
        "hierarchical_known_action_fraction": 0.96,
        "known_inspect_tasks": 55,
        "known_edit_tasks": 40,
        "known_check_or_finish_tasks": 45,
        "known_inspect_repositories": 10,
        "known_edit_repositories": 9,
        "known_check_or_finish_repositories": 9,
        "numerical_stability_fraction": 0.94,
        "stable_feature_complete_prediction_fraction": 0.99,
    }


class V4DesignAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_config = json.loads(design.DEFAULT_DESIGN_CONFIG.read_text())

    def test_canonical_config_and_evaluator_contract_validate(self) -> None:
        normalized = design.validate_design_config(self.raw_config)
        v3_protocol = design.V3.validate_protocol(
            json.loads(design.DEFAULT_V3_PROTOCOL.read_text()),
            action_protocol_value=json.loads(design.DEFAULT_ACTION_PROTOCOL.read_text()),
        )
        evaluator_contract = design.build_evaluator_contract(
            normalized, v3_protocol
        )
        self.assertNotIn("alpha_grid", evaluator_contract["decision"])
        self.assertEqual(
            evaluator_contract["forecast_pool"],
            {
                "candidate": "fixed_geometric_log_opinion_pool",
                "reference": "sequence_logit_raw_probability",
                "candidate_logit_weight": 0.80,
                "candidate_logit_j_weight": 0.20,
                "selection": "none_fixed_before_fresh_development_selection",
                "normalization": "log_space_then_exact_row_normalization",
                "shared_action_source": design.EVALUATOR.REFERENCE_PROCEDURE,
            },
        )
        self.assertEqual(
            evaluator_contract["abstention"]["accepted_accuracy_minimum"],
            0.86,
        )
        self.assertEqual(len(evaluator_contract["model"]["seeds"]), 5)
        self.assertEqual(
            evaluator_contract["model"]["fit_execution"]["worker_count"], 20
        )

        tampered = copy.deepcopy(self.raw_config)
        tampered["reserved_validation_policy"] = "allowed"
        with self.assertRaisesRegex(ValueError, "closed-validation"):
            design.validate_design_config(tampered)

        tampered = copy.deepcopy(self.raw_config)
        tampered["point_screen_thresholds"]["balanced_accuracy_minimum"] = 0.0
        with self.assertRaisesRegex(ValueError, "point-screen thresholds"):
            design.validate_design_config(tampered)

        tampered = copy.deepcopy(self.raw_config)
        tampered["procedures"]["primary"] = "sequence_logit"
        with self.assertRaisesRegex(ValueError, "procedure semantics"):
            design.validate_design_config(tampered)

        tampered = copy.deepcopy(self.raw_config)
        tampered["forecast_pool"] = tampered["evaluator_contract"].pop(
            "forecast_pool"
        )
        with self.assertRaisesRegex(ValueError, "top-level fields"):
            design.validate_design_config(tampered)

        tampered = copy.deepcopy(self.raw_config)
        tampered["evaluator_contract"]["forecast_pool"][
            "candidate_logit_j_weight"
        ] = 0.25
        with self.assertRaisesRegex(ValueError, "forecast-pool semantics"):
            design.validate_design_config(tampered)

        tampered = copy.deepcopy(self.raw_config)
        tampered["prior_design_result"]["point_failures"] = []
        with self.assertRaisesRegex(ValueError, "prior failed design-result"):
            design.validate_design_config(tampered)

    def test_point_screen_is_explicitly_nonconfirmatory(self) -> None:
        normalized = design.validate_design_config(self.raw_config)
        screen = design._point_screen(
            nested_fixture(), normalized["point_screen_thresholds"]
        )
        self.assertTrue(screen["all_point_and_full_selection_checks_passed"])
        self.assertFalse(screen["bootstrap_intervals_available"])
        self.assertTrue(screen["confirmatory_interpretation_forbidden"])
        self.assertAlmostEqual(
            screen["candidate_minus_reference_point_differences"][
                "multiclass_negative_log_likelihood"
            ],
            -0.02,
        )
        self.assertTrue(
            all(screen["full_development_selection_checks"].values())
        )

        failed = nested_fixture()
        failed["results"][design.EVALUATOR.PRIMARY_PROCEDURE]["metrics"][
            "recall_edit"
        ] = 0.60
        screen = design._point_screen(
            failed, normalized["point_screen_thresholds"]
        )
        self.assertFalse(screen["all_point_and_full_selection_checks_passed"])

        failed_identity = nested_fixture()
        failed_identity["candidate_reference_decision_d_exactly_equal"] = False
        screen = design._point_screen(
            failed_identity, normalized["point_screen_thresholds"]
        )
        self.assertFalse(screen["all_point_and_full_selection_checks_passed"])

        failed_fold_identity = nested_fixture()
        failed_fold_identity["folds"][0][
            "candidate_reference_decision_raw_r_exactly_equal"
        ] = False
        screen = design._point_screen(
            failed_fold_identity, normalized["point_screen_thresholds"]
        )
        self.assertFalse(screen["all_point_and_full_selection_checks_passed"])

        failed_action_metric_identity = nested_fixture()
        failed_action_metric_identity["results"][
            design.EVALUATOR.REFERENCE_PROCEDURE
        ]["metrics"]["accuracy"] = 0.79
        screen = design._point_screen(
            failed_action_metric_identity, normalized["point_screen_thresholds"]
        )
        self.assertFalse(screen["all_point_and_full_selection_checks_passed"])

        failed_reference_threshold = nested_fixture()
        failed_reference_threshold["full_development_settings"][
            design.EVALUATOR.REFERENCE_PROCEDURE
        ]["abstention"]["selected_under_floors"] = False
        failed_reference_threshold["full_development_promotion"][
            "reference_abstention_selected_under_floors"
        ] = False
        failed_reference_threshold["full_development_promotion"][
            "both_abstention_branches_selected_under_floors"
        ] = False
        failed_reference_threshold["full_development_promotion"][
            "fallback_blocks_promotion"
        ] = True
        failed_reference_threshold["full_development_promotion"][
            "eligible_on_full_development_selection"
        ] = False
        screen = design._point_screen(
            failed_reference_threshold, normalized["point_screen_thresholds"]
        )
        self.assertFalse(screen["all_point_and_full_selection_checks_passed"])

        support = design._support_screen(
            support_fixture(), normalized["support_screen_thresholds"]
        )
        self.assertTrue(support["all_support_checks_passed"])
        failed_support = support_fixture()
        failed_support["prediction_repositories"] = 8
        support = design._support_screen(
            failed_support, normalized["support_screen_thresholds"]
        )
        self.assertFalse(support["all_support_checks_passed"])

    def test_cli_paths_reject_outside_or_existing_output(self) -> None:
        common = {
            "design_config": design.DEFAULT_DESIGN_CONFIG,
            "v3_protocol": design.DEFAULT_V3_PROTOCOL,
            "action_protocol": design.DEFAULT_ACTION_PROTOCOL,
            "development_cohort": design.DEFAULT_COHORT,
            "prompts": design.DEFAULT_PROMPTS,
            "prompts_summary": design.DEFAULT_PROMPTS_SUMMARY,
            "public_report": design.DEFAULT_PUBLIC_REPORT,
            "replay_merge_receipt": design.DEFAULT_REPLAY_RECEIPT,
        }
        safe = argparse.Namespace(
            **common,
            output=design.DEFAULT_OUTPUT_ROOT / "unit-test-does-not-exist.json",
        )
        design.validate_cli_paths(safe)
        outside = argparse.Namespace(
            **common,
            output=Path(tempfile.gettempdir()) / "v4-design-output.json",
        )
        with self.assertRaisesRegex(ValueError, "dedicated V4 design root"):
            design.validate_cli_paths(outside)
        for forbidden_name in ("validation", "reserved_validation"):
            forbidden = argparse.Namespace(
                **common,
                output=(
                    design.DEFAULT_OUTPUT_ROOT
                    / forbidden_name
                    / "v4-design-output.json"
                ),
            )
            with self.subTest(forbidden_name=forbidden_name), self.assertRaisesRegex(
                ValueError, "reserved and validation"
            ):
                design.validate_cli_paths(forbidden)
        existing = argparse.Namespace(**common, output=design.DEFAULT_DESIGN_CONFIG)
        with self.assertRaises(ValueError):
            design.validate_cli_paths(existing)

        with tempfile.TemporaryDirectory() as temporary_directory:
            copied_action = Path(temporary_directory) / "action-protocol.json"
            copied_action.write_bytes(design.DEFAULT_ACTION_PROTOCOL.read_bytes())
            noncanonical = argparse.Namespace(
                **{**common, "action_protocol": copied_action},
                output=design.DEFAULT_OUTPUT_ROOT / "copied-input.json",
            )
            with self.assertRaisesRegex(ValueError, "canonical V3 input paths"):
                design.validate_cli_paths(noncanonical)

        with mock.patch.object(design, "prepare_inputs") as prepare:
            with self.assertRaisesRegex(ValueError, "dedicated V4 design root"):
                design.analyze_command(outside)
            prepare.assert_not_called()

    def test_analyze_output_can_never_claim_reliability_or_open_reserved(self) -> None:
        rows = [
            {
                "row_id": "row-1",
                "task_id": "task-1",
                "repo": "repo/repo",
                "metric_evaluable": True,
                "label": "inspect",
            }
        ]
        prepared = {
            "rows": rows,
            "eligibility": {},
            "evaluator_contract": {},
            "design": {
                "value": self.raw_config,
                **{
                    key: design.validate_design_config(self.raw_config)[key]
                    for key in (
                        "point_screen_thresholds",
                        "support_screen_thresholds",
                    )
                },
            },
            "hashes": {},
            "prior_design_result_binding": {
                **design.EXPECTED_PRIOR_DESIGN_RESULT,
                "exact_bytes_and_failure_status_verified": True,
            },
            "development_binding": {},
            "replay_binding": {},
        }
        captured: dict[str, object] = {}

        def capture(_path, value):
            captured.update(value)

        args = argparse.Namespace(
            design_config=design.DEFAULT_DESIGN_CONFIG,
            v3_protocol=design.DEFAULT_V3_PROTOCOL,
            action_protocol=design.DEFAULT_ACTION_PROTOCOL,
            development_cohort=design.DEFAULT_COHORT,
            prompts=design.DEFAULT_PROMPTS,
            prompts_summary=design.DEFAULT_PROMPTS_SUMMARY,
            public_report=design.DEFAULT_PUBLIC_REPORT,
            replay_merge_receipt=design.DEFAULT_REPLAY_RECEIPT,
            output=design.DEFAULT_OUTPUT_ROOT / "unit-handler-does-not-exist.json",
        )
        with (
            mock.patch.object(design, "prepare_inputs", return_value=prepared),
            mock.patch.object(
                design.EVALUATOR,
                "nested_leave_one_repository_out",
                return_value=nested_fixture(),
            ),
            mock.patch.object(
                design.V3, "support_summary", return_value=support_fixture()
            ),
            mock.patch.object(design.V3, "atomic_write_json_no_clobber", side_effect=capture),
            mock.patch("builtins.print"),
        ):
            self.assertEqual(design.analyze_command(args), 0)

        self.assertFalse(captured["operational_reliability_claim"])
        self.assertFalse(captured["independent_v4_development_result"])
        self.assertFalse(captured["reserved_validation_accessed"])
        self.assertFalse(captured["reserved_validation_allowed"])
        self.assertEqual(captured["id"], design.DESIGN_ID)
        self.assertTrue(
            captured["prior_design_result_binding"][
                "exact_bytes_and_failure_status_verified"
            ]
        )
        self.assertTrue(
            captured["fresh_disjoint_nonreserved_development_confirmation_required"]
        )


if __name__ == "__main__":
    unittest.main()
