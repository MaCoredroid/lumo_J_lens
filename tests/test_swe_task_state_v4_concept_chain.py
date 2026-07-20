from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import swe_task_state_v4_concept_chain as chain


def score_map(
    top: str = "repair",
    *,
    top_score: float = -1.0,
    second: str | None = None,
    second_score: float = -2.0,
) -> dict[str, float]:
    values = {
        concept_id: -100.0 - position
        for position, concept_id in enumerate(chain.SCORABLE_CONCEPTS)
    }
    values[top] = top_score
    if second is not None:
        values[second] = second_score
    return values


def boundary(
    boundary_id: str = "b1",
    request_index: int = 1,
    offset: int = 0,
    *,
    public_top: str = "repair",
    native_top: str | None = None,
    ordinary_top: str = "repair",
    public_strict: bool = True,
    native_strict: bool = True,
) -> dict[str, object]:
    return {
        "boundary_id": boundary_id,
        "request_index": request_index,
        "offset": offset,
        "numerical_fidelity": {
            "public_strict_adapter_pass": public_strict,
            "native_strict_adapter_pass": native_strict,
        },
        "source_concept_scores": {
            "public_j": score_map(public_top),
            "native_j": score_map(native_top or public_top),
            "ordinary_logit": score_map(ordinary_top),
        },
    }


def labels(
    rows: list[dict[str, object]],
    positives: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, object]]:
    chosen = {} if positives is None else positives
    return {
        str(row["boundary_id"]): {
            "boundary_id": row["boundary_id"],
            "positive_concept_ids": chosen.get(
                str(row["boundary_id"]), ["repair"]
            ),
            "leakage_class": "tool_outcome_explicit",
        }
        for row in rows
    }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CommonOntologyConceptChainTests(unittest.TestCase):
    def test_protocol_freezes_retrospective_and_no_cot_claims(self) -> None:
        protocol = chain.load_protocol()
        self.assertEqual(protocol["schema_version"], 2)
        self.assertEqual(protocol["protocol_id"], chain.PROTOCOL_ID)
        claim = protocol["claim_scope"]
        self.assertFalse(claim["private_chain_of_thought_reconstructed"])
        self.assertFalse(claim["subjective_emotion_inferred"])
        self.assertFalse(claim["raw_activation_decoder_used"])
        self.assertFalse(claim["concepts_absent_from_jsonl_visible_prefix_established"])
        self.assertFalse(claim["human_positive_labels_used_for_inference"])
        self.assertTrue(claim["human_positive_labels_used_for_evaluation_only"])
        self.assertTrue(claim["ontology_selected_from_completed_visible_trace"])
        self.assertTrue(claim["coordinates_selected_from_completed_visible_trace"])
        self.assertEqual(protocol["scorable_concept_order"], list(chain.SCORABLE_CONCEPTS))
        self.assertEqual(protocol["excluded_concepts"], chain.EXCLUDED_CONCEPTS)
        self.assertFalse(protocol["selection"]["calibrated"])

        changed = copy.deepcopy(protocol)
        changed["claim_scope"]["private_chain_of_thought_reconstructed"] = True
        with self.assertRaisesRegex(ValueError, "claim_scope changed"):
            chain.validate_protocol(changed)

        changed = copy.deepcopy(protocol)
        changed["scoring"]["minimum_unique_forms_per_scorable_concept"] = 1
        with self.assertRaisesRegex(ValueError, "scoring changed"):
            chain.validate_protocol(changed)

        changed = copy.deepcopy(protocol)
        changed["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "top-level fields changed"):
            chain.validate_protocol(changed)

        with tempfile.TemporaryDirectory() as temporary:
            alias = Path(temporary) / "protocol.json"
            alias.symlink_to(chain.DEFAULT_PROTOCOL)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                chain.load_protocol(alias)

    def test_every_boundary_requires_the_exact_common_candidate_grid(self) -> None:
        row = boundary()
        del row["source_concept_scores"]["public_j"]["source_localization"]
        with self.assertRaisesRegex(ValueError, "concept grid changed"):
            chain.build_concept_chain([row], trajectory_id="grid")

        row = boundary()
        row["source_concept_scores"]["native_j"]["unknown"] = -4.0
        with self.assertRaisesRegex(ValueError, "concept grid changed"):
            chain.build_concept_chain([row], trajectory_id="grid")

        row = boundary()
        row["source_concept_scores"]["ordinary_logit"]["repair"] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            chain.build_concept_chain([row], trajectory_id="grid")

    def test_selection_requires_strict_fidelity_and_paired_top1_agreement(self) -> None:
        rows = [
            boundary("agree", 1, public_top="repair", ordinary_top="verification"),
            boundary(
                "disagree",
                2,
                public_top="repair",
                native_top="verification",
                ordinary_top="repair",
            ),
            boundary(
                "fidelity",
                3,
                public_top="repair",
                native_top="repair",
                public_strict=False,
            ),
        ]
        output = chain.build_concept_chain(rows, trajectory_id="selection")
        first, second, third = output["boundaries"]
        self.assertEqual(first["selection"]["selected_concept_id"], "repair")
        self.assertEqual(
            first["selection"]["status"],
            "source_agreement_candidate_uncalibrated",
        )
        self.assertIsNone(first["selection"]["concept_probability"])
        self.assertIsNone(first["selection"]["confidence"])
        self.assertEqual(
            second["selection"]["abstention_reason"],
            "public_native_j_top1_disagreement",
        )
        self.assertEqual(
            third["selection"]["abstention_reason"],
            "paired_strict_numerical_fidelity_failed",
        )
        self.assertEqual(output["concept_chain"]["selected_boundary_count"], 1)
        self.assertFalse(output["concept_chain"]["operationally_reliable"])
        for source in chain.SCORE_SOURCES:
            ranking = first["candidate_rankings"][source]
            self.assertEqual(len(ranking["full_ranking"]), len(chain.SCORABLE_CONCEPTS))
            self.assertFalse(ranking["score_is_probability"])
            self.assertFalse(ranking["score_is_confidence"])

    def test_exact_top_score_ties_abstain_instead_of_using_ontology_order(self) -> None:
        row = boundary()
        tied = {concept_id: -7.0 for concept_id in chain.SCORABLE_CONCEPTS}
        row["source_concept_scores"]["public_j"] = dict(tied)
        row["source_concept_scores"]["native_j"] = dict(tied)
        output = chain.build_concept_chain([row], trajectory_id="ties")
        selected = output["boundaries"][0]["selection"]
        self.assertIsNone(selected["selected_concept_id"])
        self.assertEqual(selected["abstention_reason"], "non_unique_source_top1")
        self.assertFalse(selected["public_top1_unique"])
        self.assertFalse(selected["native_top1_unique"])
        self.assertEqual(
            output["boundaries"][0]["online_summary"]["template_inputs"][
                "public_top_two_margin"
            ],
            0.0,
        )
        self.assertIn(
            "non-unique top-one score",
            output["boundaries"][0]["online_summary"]["text"],
        )
        self.assertEqual(output["concept_chain"]["selected_boundary_count"], 0)

    def test_ordinary_logit_never_changes_selection(self) -> None:
        row = boundary(public_top="repair", native_top="repair", ordinary_top="repair")
        changed = copy.deepcopy(row)
        changed["source_concept_scores"]["ordinary_logit"] = score_map(
            "task_resolution"
        )
        before = chain.build_concept_chain([row], trajectory_id="ordinary")
        after = chain.build_concept_chain([changed], trajectory_id="ordinary")
        self.assertEqual(
            before["boundaries"][0]["selection"],
            after["boundaries"][0]["selection"],
        )
        self.assertNotEqual(
            before["boundaries"][0]["online_summary"]["text"],
            after["boundaries"][0]["online_summary"]["text"],
        )
        self.assertFalse(
            after["boundaries"][0]["selection"]["ordinary_logit_used_for_selection"]
        )

    def test_renderer_uses_conservative_concept_family_language_and_exact_scores(self) -> None:
        row = boundary(
            public_top="defined_identifier",
            native_top="defined_identifier",
            ordinary_top="verification",
        )
        output = chain.build_concept_chain([row], trajectory_id="render")
        summary = output["boundaries"][0]["online_summary"]
        expected = (
            "At request 1, offset 0, public and native J common-ontology rankings "
            "both led with defined-identifier concept family (scores -1.000/-1.000; "
            "top-two margins 99.000/99.000). The ordinary-logit ranking led with "
            "verification concept family (-1.000). This is only an uncalibrated "
            "source-agreement candidate; evidence is synthetic or unverified test input."
        )
        self.assertEqual(summary["text"], expected)
        self.assertEqual(summary["text_sha256"], sha256_text(expected))
        self.assertEqual(
            chain.render_boundary_sentence(summary["template_inputs"]), expected
        )
        self.assertNotIn("the defined identifier", expected.lower())
        self.assertFalse(summary["evaluation_labels_used"])

        tampered = copy.deepcopy(summary["template_inputs"])
        tampered["public_top"]["display_phrase"] = "the model understood the variable"
        with self.assertRaisesRegex(ValueError, "phrase changed"):
            chain.render_boundary_sentence(tampered)

    def test_label_permutation_cannot_create_or_change_the_rendered_chain(self) -> None:
        rows = [
            boundary("b1", 1, public_top="repair", native_top="repair"),
            boundary("b2", 2, public_top="verification", native_top="verification"),
        ]
        correct = labels(
            rows,
            {"b1": ["repair"], "b2": ["verification"]},
        )
        permuted = labels(
            rows,
            {"b1": ["task_resolution"], "b2": ["source_edit"]},
        )
        before = chain.build_concept_chain(
            rows,
            trajectory_id="permutation",
            evaluation_labels_by_boundary=correct,
        )
        after = chain.build_concept_chain(
            rows,
            trajectory_id="permutation",
            evaluation_labels_by_boundary=permuted,
        )
        for field in ("boundaries", "relations", "concept_chain", "rendering"):
            self.assertEqual(before[field], after[field])
        self.assertEqual(
            before["evaluation"]["aggregate"]["selected_accuracy"], 1.0
        )
        self.assertEqual(
            after["evaluation"]["aggregate"]["selected_accuracy"], 0.0
        )
        self.assertFalse(before["evaluation"]["labels_used_for_inference_or_rendering"])

    def test_forbidden_trace_text_future_labels_and_outcomes_are_inert(self) -> None:
        baseline = boundary()
        augmented = copy.deepcopy(baseline)
        for position, field in enumerate(chain.FORBIDDEN_FIELDS):
            augmented[field] = f"forbidden trace prose {position}"
            augmented["source_concept_scores"]["public_j"][field] = -999.0
        # Unknown score-map keys fail closed rather than influencing output.
        with self.assertRaisesRegex(ValueError, "concept grid changed"):
            chain.build_concept_chain([augmented], trajectory_id="forbidden")

        inert = copy.deepcopy(baseline)
        for field in chain.FORBIDDEN_FIELDS:
            inert[field] = "ignored boundary metadata"
        self.assertEqual(
            chain.build_concept_chain([baseline], trajectory_id="forbidden"),
            chain.build_concept_chain([inert], trajectory_id="forbidden"),
        )

    def test_future_score_mutation_cannot_change_earlier_online_nodes(self) -> None:
        rows = [
            boundary("b1", 1, public_top="repair"),
            boundary("b2", 2, public_top="verification"),
            boundary("b3", 3, public_top="task_resolution"),
        ]
        changed = copy.deepcopy(rows)
        changed[2]["source_concept_scores"]["public_j"] = score_map("source_edit")
        changed[2]["source_concept_scores"]["native_j"] = score_map("source_edit")
        before = chain.build_concept_chain(rows, trajectory_id="future")
        after = chain.build_concept_chain(changed, trajectory_id="future")
        self.assertEqual(before["boundaries"][:2], after["boundaries"][:2])

    def test_sparse_chain_sentence_preserves_abstention_gaps_scores_and_baseline(self) -> None:
        rows = [
            boundary("b1", 1, public_top="repair", ordinary_top="repair"),
            boundary(
                "b2",
                2,
                public_top="repair",
                native_top="verification",
            ),
            boundary("b3", 100, public_top="task_resolution", ordinary_top="verification"),
        ]
        output = chain.build_concept_chain(rows, trajectory_id="gaps")
        sentence = output["rendering"]["chain_sentence"]
        self.assertIn("request 1, offset 0", sentence)
        self.assertIn("after 1 abstaining registered boundary", sentence)
        self.assertIn("request 100, offset 0", sentence)
        self.assertIn("J scores", sentence)
        self.assertIn("ordinary top: verification concept family", sentence)
        self.assertIn("synthetic or unverified test input", sentence)
        self.assertEqual(
            output["relations"][0]["intervening_abstention_count"], 1
        )
        self.assertEqual(output["relations"][0]["request_delta"], 99)
        self.assertFalse(output["relations"][0]["causal_claim"])
        self.assertFalse(output["relations"][0]["backdated"])

    def test_order_duplicates_boolean_and_nonfinite_values_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "strict request/offset order"):
            chain.build_concept_chain(
                [boundary("b2", 2), boundary("b1", 1)],
                trajectory_id="bad",
            )
        with self.assertRaisesRegex(ValueError, "duplicate score coordinate"):
            chain.build_concept_chain(
                [boundary("b1", 1), boundary("b2", 1)],
                trajectory_id="bad",
            )
        row = boundary()
        row["numerical_fidelity"]["public_strict_adapter_pass"] = 1
        with self.assertRaisesRegex(ValueError, "fidelity changed"):
            chain.build_concept_chain([row], trajectory_id="bad")
        row = boundary()
        row["source_concept_scores"]["public_j"]["repair"] = math.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            chain.build_concept_chain([row], trajectory_id="bad")

    def test_input_is_not_mutated_and_json_is_finite_and_deterministic(self) -> None:
        rows = [boundary()]
        label_rows = labels(rows)
        rows_before = copy.deepcopy(rows)
        labels_before = copy.deepcopy(label_rows)
        first = chain.build_concept_chain(
            rows,
            trajectory_id="stable",
            evaluation_labels_by_boundary=label_rows,
        )
        second = chain.build_concept_chain(
            rows,
            trajectory_id="stable",
            evaluation_labels_by_boundary=label_rows,
        )
        self.assertEqual(first, second)
        self.assertEqual(rows, rows_before)
        self.assertEqual(label_rows, labels_before)
        payload = json.dumps(first, sort_keys=True, allow_nan=False)
        self.assertIsInstance(json.loads(payload), dict)
        self.assertEqual(
            first["rendering"]["canonical_text_sha256"],
            sha256_text(first["rendering"]["canonical_text"]),
        )

    def test_evaluation_is_explicitly_unavailable_without_separate_labels(self) -> None:
        output = chain.build_concept_chain([boundary()], trajectory_id="no-labels")
        self.assertEqual(
            output["evaluation"]["status"],
            "unavailable_no_separate_evaluation_labels",
        )
        self.assertIsNone(output["evaluation"]["aggregate"])
        self.assertFalse(
            output["provenance"]["human_positive_labels_used_for_evaluation_only"]
        )

    def test_authenticated_end_to_end_falsifies_a_reliable_chain_claim(self) -> None:
        output = chain.build_concept_chain_from_intermediate_artifacts()
        result = output["concept_chain"]
        self.assertEqual(result["registered_boundary_count"], 10)
        self.assertEqual(result["strict_fidelity_boundary_count"], 5)
        self.assertEqual(result["selected_boundary_count"], 3)
        self.assertEqual(result["abstained_boundary_count"], 7)
        self.assertFalse(result["operationally_reliable"])
        self.assertFalse(result["concepts_absent_from_jsonl_visible_prefix_established"])
        self.assertEqual(
            [group["concept_id"] for group in result["groups"]],
            ["focused_validation", "defined_identifier", "focused_validation"],
        )
        self.assertEqual(
            [group["intervening_abstention_count"] for group in result["groups"]],
            [0, 1, 5],
        )
        aggregate = output["evaluation"]["aggregate"]
        self.assertEqual(aggregate["source_agreement_selected_count"], 3)
        self.assertEqual(aggregate["source_agreement_correct_count"], 2)
        self.assertAlmostEqual(aggregate["selected_accuracy"], 2.0 / 3.0)
        self.assertEqual(aggregate["coverage_over_all_boundaries"], 0.3)
        self.assertEqual(
            aggregate["public_j_top1_accuracy_on_strict_boundaries"], 0.6
        )
        self.assertEqual(
            aggregate["native_j_top1_accuracy_on_strict_boundaries"], 0.6
        )
        self.assertEqual(
            aggregate["ordinary_logit_top1_accuracy_on_strict_boundaries"], 0.6
        )
        self.assertEqual(aggregate["non_prefix_explicit_boundary_count"], 0)
        self.assertFalse(aggregate["incremental_j_value_established"])
        self.assertEqual(
            aggregate["reliability_status"],
            "not_established_single_task_uncalibrated",
        )
        # Request 1 is a source-agreement false positive.
        first_eval = output["evaluation"]["boundary_rows"][0]
        self.assertEqual(first_eval["selected_concept_id"], "focused_validation")
        self.assertFalse(first_eval["selected_matches_registered_positive"])
        # Request 8 has a plausible source agreement but fails numerical fidelity.
        request_eight = output["boundaries"][7]
        self.assertTrue(request_eight["selection"]["public_native_top1_agree"])
        self.assertIsNone(request_eight["selection"]["selected_concept_id"])
        self.assertEqual(
            request_eight["selection"]["abstention_reason"],
            "paired_strict_numerical_fidelity_failed",
        )
        self.assertTrue(output["provenance"]["input_authenticated"])
        self.assertFalse(output["provenance"]["human_positive_labels_used_for_inference"])
        self.assertTrue(
            output["provenance"]["human_positive_labels_used_for_evaluation_only"]
        )
        self.assertTrue(output["provenance"]["ontology_selected_from_completed_visible_trace"])
        self.assertFalse(output["provenance"]["operational_reliability_claim"])
        self.assertFalse(output["provenance"]["reserved_validation_opened"])
        self.assertEqual(len(output["ontology"]["scorable_concepts"]), 14)
        self.assertEqual(output["ontology"]["excluded_concepts"], chain.EXCLUDED_CONCEPTS)
        self.assertIsNone(
            output["unavailable_claims"]["hidden_content_absent_from_jsonl"]["value"]
        )
        self.assertIn(
            "provide no evidence of concepts absent from the JSONL-visible prefix",
            output["rendering"]["retrospective_selection_disclaimer"],
        )

    def test_authenticated_paths_symlinks_and_hash_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alias = root / "analysis.json"
            alias.symlink_to(chain.ANALYSIS_PATH)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                chain.build_concept_chain_from_intermediate_artifacts(alias)

            copied = root / "copy.json"
            copied.write_bytes(chain.PUBLIC_REPORT_PATH.read_bytes())
            with self.assertRaisesRegex(ValueError, "path differs"):
                chain.build_concept_chain_from_intermediate_artifacts(
                    public_report_path=copied
                )

            drifted = root / "drifted.json"
            drifted.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(chain, "ANALYSIS_PATH", drifted):
                with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                    chain.build_concept_chain_from_intermediate_artifacts(drifted)

    def test_disclaimers_and_unavailable_claims_are_mandatory(self) -> None:
        output = chain.build_concept_chain([boundary()], trajectory_id="scope")
        canonical = output["rendering"]["canonical_text"]
        self.assertIn("not private chain-of-thought", canonical)
        self.assertIn("or emotion", canonical)
        self.assertIn("teacher-forced replay", canonical)
        self.assertIn("retrospectively specified", canonical)
        self.assertNotIn("because the model", canonical.lower())
        self.assertIsNone(output["unavailable_claims"]["private_chain_of_thought"]["value"])
        self.assertIsNone(output["unavailable_claims"]["emotion_or_stress"]["value"])
        self.assertIsNone(
            output["unavailable_claims"]["raw_activation_decoded_concepts"]["value"]
        )


if __name__ == "__main__":
    unittest.main()
