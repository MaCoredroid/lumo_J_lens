from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CODEBOOK_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_codebook_v2.json"
)


class SemanticChainCodebookV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.codebook = json.loads(CODEBOOK_PATH.read_text(encoding="utf-8"))

    def test_identity_scope_and_lineage_are_explicit(self) -> None:
        self.assertEqual(self.codebook["schema_version"], 2)
        self.assertEqual(
            self.codebook["id"],
            "visible-semantic-epistemic-action-chain-codebook-v2",
        )
        self.assertTrue(self.codebook["lineage"]["changes_are_pilot_informed"])
        self.assertTrue(
            self.codebook["lineage"][
                "v1_outputs_must_not_be_reinterpreted_or_relabelled"
            ]
        )
        self.assertTrue(self.codebook["scope"]["reserved_validation_closed"])
        self.assertFalse(self.codebook["scope"]["reserved_validation_accessed"])
        self.assertFalse(
            self.codebook["scope"]["private_chain_of_thought_ground_truth_claimed"]
        )

    def test_prompt_payload_hash_is_frozen_and_has_no_placeholder(self) -> None:
        prompt = self.codebook["annotator_prompt_contract"]["payload"]
        observed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self.assertEqual(
            observed,
            self.codebook["annotator_prompt_contract"]["prompt_contract_sha256"],
        )
        self.assertNotIn("TO_BE_FROZEN", CODEBOOK_PATH.read_text(encoding="utf-8"))

    def test_target_is_semantic_and_marker_presence_is_separate(self) -> None:
        rule = self.codebook["chain_rule"]
        self.assertTrue(rule["semantic_relation_not_lexical_marker"])
        self.assertFalse(rule["named_relation_word_required"])
        self.assertFalse(rule["therefore_or_because_marker_required"])
        observations = self.codebook["marker_observations"]
        self.assertTrue(observations["relation_marker_present"]["not_part_of_chain_label"])
        self.assertTrue(observations["action_marker_present"]["not_part_of_chain_label"])
        self.assertTrue(
            any(
                not item["relation_marker_present"]
                for item in self.codebook["positive_teaching_examples"]
            )
        )

    def test_every_positive_teaching_quote_is_exact_unique_and_ordered(self) -> None:
        allowed_evidence = set(self.codebook["ontology"]["evidence_kind"])
        allowed_edges = set(self.codebook["ontology"]["belief_edge"])
        allowed_domains = set(self.codebook["ontology"]["hypothesis_domain"])
        allowed_actions = set(self.codebook["ontology"]["action_intent"])
        for item in self.codebook["positive_teaching_examples"]:
            text = item["assistant_text"]
            starts = []
            ends = []
            for field in ("evidence_quote", "hypothesis_quote", "action_quote"):
                quote = item[field]
                self.assertTrue(quote)
                self.assertEqual(text.count(quote), 1, item["id"])
                start = text.index(quote)
                starts.append(start)
                ends.append(start + len(quote))
            self.assertLessEqual(ends[0], starts[1], item["id"])
            self.assertLessEqual(ends[1], starts[2], item["id"])
            self.assertIn(item["evidence_kind"], allowed_evidence)
            self.assertIn(item["belief_edge"], allowed_edges)
            self.assertIn(item["hypothesis_domain"], allowed_domains)
            self.assertIn(item["action_intent"], allowed_actions)

    def test_quote_resolution_and_control_gates_are_fail_closed(self) -> None:
        interface = self.codebook["quote_interface"]
        self.assertFalse(interface["model_outputs_numeric_offsets"])
        self.assertTrue(
            interface[
                "single_occurrence_per_quote_not_required_when_one_ordered_tuple_is_unique"
            ]
        )
        self.assertEqual(
            interface["zero_or_multiple_valid_ordered_tuples"],
            "explicit_interface_unknown",
        )
        self.assertTrue(interface["fuzzy_or_normalized_matching_forbidden"])
        controls = self.codebook["acceptance_controls_contract"]
        self.assertTrue(
            controls["teaching_examples_in_this_codebook_are_forbidden_as_acceptance_controls"]
        )
        self.assertGreaterEqual(controls["sealed_completion_controls_minimum"], 32)
        self.assertGreaterEqual(controls["sealed_novelty_controls_minimum"], 8)
        self.assertGreaterEqual(controls["sealed_adjudication_controls_minimum"], 12)
        for value in controls["per_independent_annotator_gates"].values():
            self.assertIn(value, (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
