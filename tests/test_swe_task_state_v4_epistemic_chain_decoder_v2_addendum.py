from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ADDENDUM_PATH = (
    ROOT
    / "configs"
    / "swe_task_state_v4_epistemic_chain_decoder_v2_addendum.json"
)
ADDENDUM_SHA256 = (
    "d0ce668fd966f7185db8e8ce263d42b8857f42ff01bba9571961403f12bc7f85"
)


class AddendumContractError(RuntimeError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise AddendumContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise AddendumContractError(f"top-level JSON value is not an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class EpistemicChainDecoderV2AddendumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addendum = _load_json(ADDENDUM_PATH)
        lineage = cls.addendum["frozen_lineage"]
        assert isinstance(lineage, dict)
        cls.lineage = lineage
        cls.base = _load_json(
            ROOT / str(lineage["base_decoder_config"]["path"])
        )
        cls.codebook = _load_json(
            ROOT / str(lineage["v2_annotation_codebook"]["path"])
        )

    def test_addendum_and_all_four_lineage_inputs_are_exactly_hash_bound(self) -> None:
        self.assertEqual(_sha256_file(ADDENDUM_PATH), ADDENDUM_SHA256)
        expected = {
            "base_decoder_config": (
                "configs/swe_task_state_v4_epistemic_chain_decoder.json",
                "2c9a4c16a36ef96c9db32e231aa337947a099f21bb791b2411c261f631d087a1",
            ),
            "base_decoder_script": (
                "scripts/swe_task_state_v4_epistemic_chain_decoder.py",
                "970fa58e1ffb6d65c3d6a025cbaf8437aa0b93ee18305f946a3b6af7bb80b619",
            ),
            "v2_annotation_codebook": (
                "configs/swe_task_state_v4_epistemic_chain_codebook_v2.json",
                "2105a50c7bc13a064ca75c4a69aad631869fbbb2c17c94970eeaa92722aff85c",
            ),
            "v1_pilot_failure_report": (
                ".cache/swe_task_state_v4_raw_capture/n60-final/epistemic-chain-quality-audit-v5-final-report.json",
                "d7488092850a19af43f5efa7f7d2c4e38abec02fba45e0b8548c902907d80be6",
            ),
        }
        for name, (relative_path, expected_sha256) in expected.items():
            with self.subTest(name=name):
                binding = self.lineage[name]
                self.assertEqual(binding["path"], relative_path)
                self.assertEqual(binding["sha256"], expected_sha256)
                self.assertNotIn("reserved", relative_path.casefold())
                self.assertNotIn("validation", relative_path.casefold())
                self.assertEqual(_sha256_file(ROOT / relative_path), expected_sha256)

    def test_scope_is_prospective_closed_and_contains_no_fit_or_result_claim(self) -> None:
        self.assertEqual(self.addendum["schema_version"], 1)
        self.assertEqual(
            self.addendum["kind"],
            "swe_task_state_v4_epistemic_chain_decoder_v2_addendum",
        )
        scope = self.addendum["scope"]
        self.assertTrue(scope["development_data_only"])
        self.assertTrue(scope["reserved_validation_closed"])
        self.assertFalse(scope["reserved_validation_accessed"])
        for field in (
            "v2_control_outcomes_available_when_frozen",
            "v2_real_pilot_outcomes_available_when_frozen",
            "v2_target_annotations_available_when_frozen",
            "v2_target_results_observed_when_frozen",
            "model_fit_performed_under_this_addendum_when_frozen",
            "target_annotation_run_authorized_when_frozen",
        ):
            self.assertFalse(scope[field], field)

    def test_base_method_and_claim_sections_are_exact_inherited_floors(self) -> None:
        composition = self.addendum["composition_contract"]
        self.assertEqual(
            composition["operator"],
            "logical_conjunction_of_authenticated_base_decoder_and_this_addendum",
        )
        self.assertTrue(composition["base_config_and_script_bytes_must_remain_exact"])
        self.assertFalse(
            composition["base_decoder_source_or_config_mutation_authorized"]
        )
        self.assertTrue(composition["base_requirements_are_floors_and_may_not_be_weakened"])

        frozen_sections = composition["unchanged_base_sections_canonical_sha256"]
        required_sections = {
            "annotation_evidence_contract",
            "target_projection",
            "support_and_agreement_gate",
            "full_prefix_semantic_baseline",
            "frozen_proposition_embedding_model",
            "proposition_content_lane",
            "causal_prior_chain_control",
            "numeric_feature_blocks",
            "variants",
            "nested_comparisons",
            "split_and_weighting",
            "model",
            "nested_calibration_and_abstention",
            "metrics",
            "full_refit_uncertainty",
            "controls_and_sensitivities",
            "renderer",
            "claim_scope",
            "mandatory_limitations",
        }
        self.assertTrue(required_sections.issubset(frozen_sections))
        for section, expected_sha256 in frozen_sections.items():
            with self.subTest(section=section):
                self.assertIn(section, self.base)
                self.assertEqual(
                    _canonical_sha256(self.base[section]), expected_sha256
                )

        successor = composition["v2_annotation_successor_rule"]
        self.assertTrue(successor["new_v2_target_rows_must_use_only_the_bound_v2_codebook"])
        self.assertTrue(successor["mixing_v1_and_v2_labels_in_one_fit_or_report_forbidden"])
        self.assertTrue(
            successor[
                "all_base_annotation_independence_adjudication_support_and_agreement_requirements_remain_floors"
            ]
        )

    def test_semantic_target_is_marker_free_and_marker_flags_are_never_gold(self) -> None:
        target = self.addendum["semantic_target_contract"]
        self.assertTrue(target["semantic_marker_free_target"])
        self.assertFalse(target["named_relation_or_action_link_word_required"])
        self.assertTrue(target["marker_absence_does_not_make_a_semantic_positive_negative_or_unknown"])
        self.assertTrue(target["marker_presence_does_not_make_a_semantic_negative_positive"])
        self.assertTrue(target["lexical_marker_decoys_must_remain_negative"])

        rule = self.codebook["chain_rule"]
        self.assertTrue(rule["semantic_relation_not_lexical_marker"])
        self.assertFalse(rule["named_relation_word_required"])
        self.assertFalse(rule["therefore_or_because_marker_required"])
        diagnostics = self.addendum["lexical_marker_diagnostics"]
        self.assertTrue(diagnostics["diagnostic_only_never_semantic_target_or_gold"])
        self.assertTrue(
            diagnostics[
                "never_a_numeric_feature_training_label_calibration_label_or_selection_criterion"
            ]
        )
        self.assertTrue(
            diagnostics[
                "no_chain_semantic_unknown_interface_unknown_and_unresolved_rows_are_not_applicable_never_marker_absent"
            ]
        )
        for marker in ("relation_marker_present", "action_marker_present"):
            self.assertTrue(
                self.codebook["marker_observations"][marker][
                    "not_part_of_chain_label"
                ]
            )

    def test_quote_first_status_projection_is_total_and_fail_closed(self) -> None:
        contract = self.addendum["quote_first_annotation_contract"]
        self.assertTrue(contract["model_emits_literal_E_H_A_quotes_and_never_numeric_offsets"])
        self.assertTrue(
            contract[
                "materialize_offsets_only_for_exactly_one_nonoverlapping_E_before_H_before_A_tuple"
            ]
        )
        self.assertEqual(
            set(contract["materialization_statuses"]),
            {
                "resolved_chain",
                "not_applicable_no_chain",
                "deterministic_no_chain_empty_visible_prose",
                "not_applicable_semantic_unknown",
                "interface_unknown",
            },
        )
        self.assertEqual(
            set(contract["interface_unknown_reasons"]),
            {
                "invalid_structured_output",
                "invalid_semantic_proposal_interface",
                "missing_exact_quote",
                "no_valid_ordered_quote_tuple",
                "ambiguous_ordered_quote_tuple",
            },
        )
        self.assertTrue(contract["any_unknown_never_coerced_to_no_chain"])
        self.assertFalse(self.codebook["quote_interface"]["model_outputs_numeric_offsets"])
        self.assertEqual(
            self.codebook["quote_interface"]["zero_or_multiple_valid_ordered_tuples"],
            "explicit_interface_unknown",
        )

    def test_three_family_control_gate_has_exact_role_specific_counts(self) -> None:
        gate = self.addendum["family_diverse_readiness_gate"]
        self.assertTrue(gate["target_annotation_run_forbidden_until_gate_passes"])
        self.assertTrue(gate["any_decoder_model_fit_forbidden_until_gate_passes"])
        self.assertEqual(
            gate["decision_roles"],
            ["independent_a", "independent_b", "adjudicator"],
        )
        diversity = gate["model_family_identity"]
        self.assertTrue(diversity["all_three_base_model_lineages_must_be_pairwise_distinct"])
        self.assertTrue(
            diversity[
                "different_checkpoint_size_revision_quantization_or_alias_within_one_base_lineage_is_not_family_diversity"
            ]
        )
        self.assertEqual(
            gate["sealed_control_counts"],
            {"completion": 32, "novelty": 8, "adjudication": 12},
        )
        self.assertEqual(
            gate["role_control_requirements"],
            {
                "independent_a": {"completion": 32, "novelty": 8},
                "independent_b": {"completion": 32, "novelty": 8},
                "adjudicator": {"adjudication": 12},
            },
        )
        controls = gate["sealed_control_protocol"]
        self.assertTrue(controls["both_primaries_must_independently_pass_every_assigned_control"])
        self.assertTrue(
            controls[
                "adjudicator_must_pass_every_assigned_control_including_candidate_1_candidate_2_and_neither_correct_cases"
            ]
        )
        self.assertEqual(controls["unambiguous_category_accuracy"], 1.0)
        self.assertEqual(controls["lexical_marker_pair_invariance"], 1.0)
        self.assertEqual(controls["invalid_structured_output_rate"], 0.0)

    def test_real_pilot_must_be_non_degenerate_positive_and_low_unknown(self) -> None:
        gate = self.addendum["family_diverse_readiness_gate"]
        pilot = gate["real_development_pilot"]
        self.assertEqual(pilot["row_count_minimum_inclusive"], 32)
        self.assertEqual(pilot["row_count_maximum_inclusive"], 64)
        self.assertTrue(
            pilot["rows_are_real_authenticated_development_packets_not_teaching_or_control_rows"]
        )
        self.assertTrue(
            pilot["completion_has_chain_cohen_kappa_must_be_defined_finite_and_non_degenerate"]
        )
        self.assertTrue(
            pilot["novelty_cohen_kappa_must_be_defined_finite_and_non_degenerate"]
        )
        self.assertGreaterEqual(
            pilot["minimum_completion_has_chain_cohen_kappa"],
            self.base["support_and_agreement_gate"]["minimum_completion_has_chain_kappa"],
        )
        self.assertGreaterEqual(
            pilot["minimum_novelty_cohen_kappa"],
            self.base["support_and_agreement_gate"]["minimum_novelty_kappa"],
        )
        for field in (
            "minimum_available_chain_positive_rows_per_primary",
            "minimum_adjudicated_available_chain_positive_rows",
            "minimum_adjudicated_novel_positive_rows",
            "minimum_adjudicated_prefix_exposed_positive_rows",
        ):
            self.assertGreater(pilot[field], 0, field)
        for field in (
            "maximum_each_primary_completion_unknown_rate",
            "maximum_adjudicated_final_completion_unknown_rate",
            "maximum_each_novelty_lane_unknown_rate",
        ):
            self.assertGreaterEqual(pilot[field], 0.0, field)
            self.assertLessEqual(pilot[field], 0.1, field)
        self.assertTrue(
            pilot["unknowns_never_removed_from_rate_denominators_or_coerced_to_a_class"]
        )
        required_pass_components = set(gate["pass_is_conjunction_of"])
        self.assertIn(
            "independent_a_completion_and_novelty_controls_passed",
            required_pass_components,
        )
        self.assertIn(
            "independent_b_completion_and_novelty_controls_passed",
            required_pass_components,
        )
        self.assertIn("adjudicator_controls_passed", required_pass_components)

    def test_marker_strata_repeat_exact_base_nested_pairs_without_refitting_changes(self) -> None:
        comparisons = self.addendum["marker_stratified_nested_comparisons"]
        self.assertEqual(
            comparisons["candidate_reference_pairs"],
            self.base["nested_comparisons"],
        )
        self.assertEqual(
            comparisons["model_fits_predictions_splits_weights_and_hyperparameters"],
            "unchanged_base_out_of_fold_pipeline",
        )
        expected_strata = {
            "relation_marker_absent",
            "relation_marker_present",
            "action_marker_absent",
            "action_marker_present",
            "joint_relation_absent_action_absent",
            "joint_relation_absent_action_present",
            "joint_relation_present_action_absent",
            "joint_relation_present_action_present",
        }
        strata = comparisons["exact_strata"]
        self.assertEqual({item["id"] for item in strata}, expected_strata)
        self.assertEqual(len(strata), len(expected_strata))
        self.assertTrue(
            comparisons[
                "same_exact_rows_folds_weights_targets_predictions_and_paired_refit_draws_within_each_candidate_reference_stratum"
            ]
        )
        self.assertTrue(
            comparisons[
                "insufficient_or_unknown_rows_may_not_be_dropped_pooled_reassigned_or_called_marker_absent"
            ]
        )

        family = comparisons["stratified_inferential_family"]
        expected_family_size = (
            family["candidate_count"]
            * family["required_metric_count"]
            * family["stratum_count"]
        )
        self.assertEqual(expected_family_size, 48)
        self.assertEqual(family["family_size"], expected_family_size)
        tail = (1.0 - family["interval_level"]) / (2 * expected_family_size)
        self.assertTrue(math.isclose(family["adjusted_lower_quantile"], tail))
        self.assertTrue(
            math.isclose(family["adjusted_upper_quantile"], 1.0 - tail)
        )
        self.assertTrue(
            comparisons[
                "marker_present_performance_alone_never_supports_a_marker_free_semantic_claim"
            ]
        )
        claim_requirements = comparisons["marker_free_semantic_claim_requires"]
        self.assertTrue(
            any("joint_absent_stratum" in item for item in claim_requirements)
        )

    def test_proposition_content_lane_is_bitwise_semantically_unchanged(self) -> None:
        inheritance = self.addendum["content_proposition_lane_inheritance"]
        self.assertEqual(inheritance["mode"], "exact_unchanged_inheritance")
        observed = _canonical_sha256(self.base["proposition_content_lane"])
        self.assertEqual(observed, inheritance["canonical_sha256"])
        self.assertEqual(
            observed,
            self.addendum["composition_contract"][
                "unchanged_base_sections_canonical_sha256"
            ]["proposition_content_lane"],
        )
        self.assertTrue(
            inheritance[
                "target_regressor_retrieval_abstention_metrics_quality_floors_and_claim_gate_unchanged"
            ]
        )
        self.assertTrue(
            inheritance["lexical_marker_flags_never_enter_proposition_embeddings_or_features"]
        )

    def test_any_codebook_prompt_or_model_change_resets_readiness(self) -> None:
        invalidation = self.addendum["gate_invalidation_contract"]
        for field in (
            "any_codebook_change_resets_controls_and_real_pilot_gate",
            "any_prompt_or_schema_change_resets_controls_and_real_pilot_gate",
            "any_role_model_family_checkpoint_revision_quantization_or_dtype_change_resets_controls_and_real_pilot_gate",
            "any_generation_or_output_extraction_change_resets_controls_and_real_pilot_gate",
            "stale_or_partial_receipts_may_not_authorize_target_run",
        ):
            self.assertTrue(invalidation[field], field)
        fingerprint = set(invalidation["fingerprint_must_bind"])
        self.assertIn("v2_codebook_file_sha256", fingerprint)
        self.assertIn("all_system_user_and_adjudication_prompt_bytes", fingerprint)
        self.assertIn(
            "each_role_base_model_lineage_repo_revision_snapshot_quantization_and_dtype",
            fingerprint,
        )

    def test_claims_remain_visible_semantic_only_not_private_affect_or_operational(self) -> None:
        claims = self.addendum["claim_boundary"]
        self.assertTrue(claims["visible_future_semantic_cot_like_chain_is_the_only_new_target"])
        for field in (
            "private_or_hidden_chain_of_thought_recovery_claim_forbidden",
            "literal_heldout_sentence_reconstruction_claim_forbidden",
            "emotion_affect_confidence_doubt_stress_or_subjective_state_claim_forbidden",
            "causal_understanding_intent_or_conscious_experience_claim_forbidden",
            "operational_reliability_claim_forbidden_without_separate_untouched_confirmation",
            "development_controls_or_pilot_success_is_not_decoder_success",
            "visible_lexical_marker_detection_is_not_semantic_chain_decoding",
        ):
            self.assertTrue(claims[field], field)


if __name__ == "__main__":
    unittest.main()
