import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "swe_task_state_v4_cot_claim_ladder_v1.json"
STAGE_A = ROOT / "configs" / "swe_task_state_v4_counterfactual_stage_a_runtime.json"


class CotClaimLadderV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_cot_is_first_class_and_separate_from_affect(self):
        goal = self.config["goal_contract"]
        self.assertIs(goal["cot_or_cot_like_reasoning_is_a_first_class_target"], True)
        self.assertIs(goal["cot_or_cot_like_reasoning_is_not_redefined_as_emotion_or_affect"], True)
        self.assertIs(goal["private_or_verbatim_cot_is_in_scope_as_an_aspirational_target"], True)
        self.assertIs(
            self.config["scope"]["emotion_affect_confidence_doubt_and_stress_remain_a_separate_goal_lane"],
            True,
        )

    def test_claims_form_strict_four_level_ladder(self):
        ladder = self.config["claims_ladder"]
        self.assertEqual([row["level"] for row in ladder], [0, 1, 2, 3])
        by_level = {row["level"]: row for row in ladder}
        self.assertIn("same_model_produces_the_completion_and_the_measured_internal_states", by_level[2]["required_evidence"])
        self.assertIn(
            "online_activation_capture_in_the_exact_autoregressive_generation_run_or_an_explicit_replay_only_claim_with_authenticated_exact_token_prefix_runtime_cache_position_state_and_logit_parity",
            by_level[2]["required_evidence"],
        )
        self.assertIn("capture_at_three_or_more_predeclared_causally_ordered_boundaries", by_level[2]["required_evidence"])
        self.assertIn("private_scratchpad_or_reasoning_channel_recorded_physically_separately_and_hash_bound", by_level[3]["required_evidence"])
        self.assertIn("private_cot_of_codex_or_any_other_uninstrumented_agent", by_level[3]["does_not_establish"])

    def test_level_two_preserves_temporal_leakage_causal_and_replication_gates(self):
        level_two = self.config["claims_ladder"][2]["required_evidence"]
        required = {
            "each_boundary_feature_excludes_all_future_visible_tokens_tool_results_and_labels",
            "target_packet_selection_annotation_split_and_label_locks_frozen_blind_to_activations_lens_scores_outcomes_and_decoder_performance",
            "separate_E_H_and_A_proposition_targets_not_only_coarse_slots_or_words",
            "predeclared_temporal_order_or_anticipation_lead_time_gate_for_E_then_H_then_A",
            "strict_incremental_improvement_over_the_same_boundarys_complete_visible_prefix_baseline",
            "task_and_repository_heldout_generalization",
            "marker_free_and_prefix_absent_target_strata",
            "matched_norm_random_direction_orthogonal_direction_layer_and_time_shuffle_controls",
            "predeclared_activation_intervention_changes_the_later_semantic_target_in_the_predicted_direction",
            "paired_intervention_and_sham_runs_have_identical_prefix_generated_token_history_through_intervention_sampler_seed_runtime_and_cache_with_the_activation_delta_as_the_sole_difference",
            "replication_on_a_second_untouched_same_model_generation_cohort",
        }
        self.assertTrue(required.issubset(set(level_two)))

    def test_private_cot_preserves_lock_leakage_open_set_and_scope_gates(self):
        level_three = self.config["claims_ladder"][3]
        required = {
            "private_ground_truth_withheld_until_predictions_and_abstentions_are_locked",
            "private_ground_truth_absent_from_all_visible_prefixes_features_prompts_and_candidate_pools",
            "training_only_open_set_retrieval_or_generation_not_future_supplied_closed_set_only",
            "exact_copy_leakage_canaries_and_randomized_private_target_controls",
            "untouched_confirmation_cohort",
            "claim_limited_to_the_exact_named_model_runtime_task_distribution_and_reasoning_channel",
        }
        self.assertTrue(required.issubset(set(level_three["required_evidence"])))
        self.assertIn(
            "private_cot_of_codex_or_any_other_uninstrumented_agent",
            level_three["does_not_establish"],
        )

    def test_current_teacher_forced_replay_boundary_is_explicit(self):
        boundary = self.config["current_evidence_boundary"]
        self.assertEqual(boundary["current_raw_capture_states"], "teacher_forced_replay_states_at_the_exact_visible_prefix_tail")
        self.assertIs(boundary["current_replay_states_are_original_codex_agent_internal_states"], False)
        self.assertIs(boundary["current_replay_states_can_establish_original_agent_private_cot"], False)
        self.assertIs(boundary["failed_v2_semantic_control_is_cot_evidence"], False)

    def test_stage_a_binding_and_single_boundary_limit(self):
        stage = self.config["same_model_stage_a_boundary"]
        self.assertEqual(
            hashlib.sha256(STAGE_A.read_bytes()).hexdigest(),
            stage["prospective_stage_a_runtime_config_sha256"],
        )
        self.assertIs(stage["single_prompt_tail_capture_is_a_temporal_trajectory"], False)
        self.assertIs(stage["three_or_more_causal_boundaries_remain_required_for_level_2"], True)
        self.assertIs(stage["generation_or_capture_executed_when_this_contract_was_frozen"], False)

    def test_every_scientific_claim_starts_false(self):
        self.assertTrue(self.config["claim_state"])
        self.assertTrue(all(value is False for value in self.config["claim_state"].values()))
        self.assertEqual(
            self.config["failure_actions"]["affect_lane_result"],
            "never_substitute_affect_evidence_for_cot_or_cot_like_evidence",
        )


if __name__ == "__main__":
    unittest.main()
