from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import swe_task_state_v4_reasoning_trace as trace


def probabilities(top: str, confidence: float = 0.8) -> dict[str, float]:
    remainder = (1.0 - confidence) / 2.0
    result = {class_id: remainder for class_id in trace.CLASSES}
    result[top] = confidence
    return result


def prediction(
    row_id: str,
    task_id: str,
    request_index: int,
    *,
    q: dict[str, float] | None = None,
    d: dict[str, float] | None = None,
    accepted: bool = True,
) -> dict[str, object]:
    forecast = probabilities("inspect") if q is None else q
    decision = dict(forecast) if d is None else d
    forecast_top = max(trace.CLASSES, key=lambda class_id: forecast[class_id])
    decision_top = max(trace.CLASSES, key=lambda class_id: decision[class_id])
    return {
        "row_id": row_id,
        "task_id": task_id,
        "task_request_index": request_index,
        "forecast_probabilities_q": forecast,
        "decision_probabilities_d": decision,
        "forecast_top_class": forecast_top,
        "forecast_top_confidence": forecast[forecast_top],
        "predicted_class": decision_top,
        "decision_confidence_from_q": forecast[decision_top],
        "accepted": accepted,
    }


def diagnostics(
    *,
    logit: dict[str, float] | None = None,
    public_j: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "sequence_logit_probabilities": (
            probabilities("inspect", 0.7) if logit is None else logit
        ),
        "sequence_j_probabilities": (
            probabilities("inspect", 0.7) if public_j is None else public_j
        ),
    }


class ReasoningTraceTests(unittest.TestCase):
    def test_canonical_protocol_and_claim_boundary(self) -> None:
        protocol = trace.load_protocol()
        self.assertEqual(protocol["protocol_id"], trace.PROTOCOL_ID)
        self.assertEqual(protocol["action_classes_in_order"], list(trace.CLASSES))
        self.assertFalse(protocol["claim_scope"]["private_chain_of_thought_reconstructed"])
        self.assertFalse(protocol["claim_scope"]["subjective_emotion_inferred"])
        self.assertFalse(protocol["evidence_status"]["fresh_transport_confirmation"])
        self.assertEqual(
            protocol["evidence_status"]["row_level_reliability_intervals"],
            "unavailable_no_fresh_transport_confirmatory_bootstrap",
        )

        changed = copy.deepcopy(protocol)
        changed["claim_scope"]["private_chain_of_thought_reconstructed"] = True
        with self.assertRaisesRegex(ValueError, "claim scope changed"):
            trace.validate_protocol(changed)

        changed = copy.deepcopy(protocol)
        changed["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "top-level fields changed"):
            trace.validate_protocol(changed)

        changed = copy.deepcopy(protocol)
        changed["thresholds"]["uncertainty_band_edges"][0] += 1e-15
        with self.assertRaisesRegex(ValueError, "thresholds changed"):
            trace.validate_protocol(changed)

        with tempfile.TemporaryDirectory() as temporary:
            alias = Path(temporary) / "protocol.json"
            alias.symlink_to(trace.DEFAULT_PROTOCOL)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                trace.load_protocol(alias)

    def test_confidence_doubt_entropy_and_ambivalence_are_exact(self) -> None:
        q = {"inspect": 0.7, "edit": 0.2, "check_or_finish": 0.1}
        record = trace.build_reasoning_trace(
            [prediction("r1", "task-a", 1, q=q)]
        )[0]
        phase = record["phase_forecast"]
        proxies = record["proxies"]
        self.assertEqual(phase["selected_phase"], "information_gathering")
        self.assertAlmostEqual(phase["decision_confidence"], 0.7)
        self.assertAlmostEqual(proxies["forecast_doubt"]["value"], 0.3)
        expected_entropy = -sum(value * math.log(value) for value in q.values()) / math.log(3)
        self.assertAlmostEqual(proxies["diffuse_uncertainty"]["value"], expected_entropy)
        self.assertAlmostEqual(
            proxies["ambivalence"]["value"], 1.0 - (0.7 - 0.2) / (0.7 + 0.2)
        )
        self.assertEqual(
            proxies["forecast_volatility"]["status"], "unavailable_first_boundary"
        )
        self.assertIsNone(proxies["forecast_volatility"]["value"])
        self.assertEqual(
            record["phase_forecast"]["confidence_status"], "synthetic_unvalidated"
        )
        self.assertFalse(record["provenance"]["input_authenticated"])

    def test_confidence_is_q_at_decision_argmax_not_forecast_top(self) -> None:
        q = {"inspect": 0.60, "edit": 0.25, "check_or_finish": 0.15}
        d = {"inspect": 0.20, "edit": 0.70, "check_or_finish": 0.10}
        record = trace.build_reasoning_trace(
            [prediction("r1", "task-a", 1, q=q, d=d)]
        )[0]
        self.assertEqual(record["phase_forecast"]["forecast_top_phase"], "information_gathering")
        self.assertEqual(record["phase_forecast"]["candidate_phase"], "implementation")
        self.assertAlmostEqual(record["phase_forecast"]["decision_confidence"], 0.25)
        self.assertAlmostEqual(record["proxies"]["forecast_doubt"]["value"], 0.75)
        self.assertTrue(record["proxies"]["forecast_decision_class_conflict"]["value"])

    def test_abstention_nulls_selected_phase_but_retains_probabilities(self) -> None:
        record = trace.build_reasoning_trace(
            [prediction("r1", "task-a", 1, accepted=False)]
        )[0]
        self.assertIsNone(record["phase_forecast"]["selected_phase"])
        self.assertEqual(record["phase_forecast"]["candidate_phase"], "information_gathering")
        self.assertEqual(record["reasoning_trace"]["events"], ["uncertain_transition"])
        self.assertTrue(record["abstention"]["abstained"])
        self.assertTrue(record["abstention"]["probability_vector_retained"])
        self.assertIn("abstained", record["reasoning_trace"]["rationale"])

    def test_high_level_phase_transition_events(self) -> None:
        rows = [
            prediction("r1", "task-a", 1, q=probabilities("inspect")),
            prediction("r2", "task-a", 2, q=probabilities("edit")),
            prediction("r3", "task-a", 3, q=probabilities("check_or_finish")),
            prediction("r4", "task-a", 4, q=probabilities("inspect")),
            prediction("r5", "task-a", 5, q=probabilities("inspect")),
        ]
        output = trace.build_reasoning_trace(rows)
        self.assertEqual(
            [row["reasoning_trace"]["events"][0] for row in output],
            [
                "phase_observation",
                "inspection_to_implementation",
                "implementation_to_verification_or_completion",
                "reconsideration_or_rework_like",
                "phase_continuation",
            ],
        )

    def test_interleaved_tasks_have_isolated_temporal_state(self) -> None:
        rows = [
            prediction("a1", "task-a", 1, q=probabilities("inspect")),
            prediction("b1", "task-b", 4, q=probabilities("edit")),
            prediction("a2", "task-a", 2, q=probabilities("edit")),
            prediction("b2", "task-b", 5, q=probabilities("check_or_finish")),
        ]
        output = trace.build_reasoning_trace(rows)
        self.assertEqual(output[0]["reasoning_trace"]["events"], ["phase_observation"])
        self.assertEqual(output[1]["reasoning_trace"]["events"], ["phase_observation"])
        self.assertEqual(
            output[2]["reasoning_trace"]["events"], ["inspection_to_implementation"]
        )
        self.assertEqual(
            output[3]["reasoning_trace"]["events"],
            ["implementation_to_verification_or_completion"],
        )

        bad = [
            prediction("a1", "task-a", 3),
            prediction("a2", "task-a", 2),
        ]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            trace.build_reasoning_trace(bad)

        gap = trace.build_reasoning_trace(
            [
                prediction("g1", "task-gap", 1, q=probabilities("check_or_finish")),
                prediction("g2", "task-gap", 10, q=probabilities("inspect")),
            ]
        )
        self.assertEqual(gap[1]["reasoning_trace"]["events"], ["phase_observation"])
        self.assertIsNone(
            gap[1]["reasoning_trace"]["template_inputs"]["previous_accepted_phase"]
        )

    def test_label_text_outcome_and_private_reasoning_fields_are_inert(self) -> None:
        baseline = prediction("r1", "task-a", 1)
        augmented = copy.deepcopy(baseline)
        for index, field in enumerate(trace.FORBIDDEN_INFERENCE_FIELDS):
            augmented[field] = {"arbitrary_future_value": index}
        augmented["source_action_class_id"] = "finalize"
        augmented["label"] = "check_or_finish"
        augmented["reasoning"] = "secret future prose"
        augmented["official_outcome"] = "resolved"
        self.assertEqual(
            trace.build_reasoning_trace([baseline]),
            trace.build_reasoning_trace([augmented]),
        )

    def test_future_mutation_cannot_change_earlier_trace_rows(self) -> None:
        original = [
            prediction("r1", "task-a", 1, q=probabilities("inspect")),
            prediction("r2", "task-a", 2, q=probabilities("edit")),
            prediction("r3", "task-a", 3, q=probabilities("check_or_finish")),
        ]
        changed = copy.deepcopy(original)
        changed[2] = prediction("r3", "task-a", 3, q=probabilities("inspect", 0.95))
        before = trace.build_reasoning_trace(original)
        after = trace.build_reasoning_trace(changed)
        self.assertEqual(before[:2], after[:2])

    def test_source_disagreement_is_synthetic_and_load_stays_partial(self) -> None:
        rows = [
            prediction("r1", "task-a", 1, q=probabilities("inspect", 0.7)),
            prediction("r2", "task-a", 2, q=probabilities("edit", 0.7)),
        ]
        diagnostic_rows = {
            "r1": diagnostics(),
            "r2": diagnostics(
                logit=probabilities("inspect", 0.9),
                public_j=probabilities("edit", 0.9),
            ),
        }
        output = trace.build_reasoning_trace(
            rows,
            diagnostics_by_row=diagnostic_rows,
        )
        first_load = output[0]["proxies"]["activation_load_like"]
        self.assertEqual(first_load["available_component_count"], 2)
        self.assertEqual(first_load["status"], "partial_diagnostic_not_comparable")
        self.assertIsNone(first_load["percentile"])
        self.assertEqual(
            output[0]["proxies"]["ordinary_activation_innovation"]["status"],
            "unavailable_first_boundary",
        )

        second = output[1]["proxies"]
        self.assertGreater(second["source_disagreement"]["value"], 0.0)
        self.assertEqual(
            second["source_disagreement"]["status"],
            "available_synthetic_test_only",
        )
        self.assertEqual(second["activation_load_like"]["available_component_count"], 3)
        self.assertEqual(
            second["activation_load_like"]["status"],
            "partial_diagnostic_not_comparable",
        )
        self.assertIsNone(second["activation_load_like"]["percentile"])

    def test_recovery_like_remains_null_without_authenticated_fitted_episode_rule(self) -> None:
        rows = [
            prediction("r1", "task-a", 1, q=probabilities("inspect", 0.6)),
            prediction("r2", "task-a", 100, q=probabilities("edit", 0.96)),
        ]
        output = trace.build_reasoning_trace(rows)
        for record in output:
            self.assertNotIn("recovery_like", record["reasoning_trace"]["events"])
            self.assertIsNone(record["proxies"]["recovery_like_event"]["value"])
            self.assertEqual(
                record["proxies"]["recovery_like_event"]["status"],
                "unavailable_unfitted_authenticated_load_episode_rule",
            )

    def test_unfitted_behavioral_probabilities_and_emotion_are_explicitly_unavailable(self) -> None:
        record = trace.build_reasoning_trace([prediction("r1", "task-a", 1)])[0]
        for name in (
            "hesitation_like_probability",
            "trajectory_continuation_probability",
            "recovery_within_2_probability",
        ):
            self.assertIsNone(record["proxies"][name]["value"])
            self.assertTrue(record["proxies"][name]["status"].startswith("unavailable_unfitted_"))
        self.assertEqual(record["emotion_inference"]["status"], "unsupported")
        self.assertIsNone(record["emotion_inference"]["value"])
        self.assertFalse(record["claim_scope"]["private_chain_of_thought_reconstructed"])
        self.assertEqual(
            record["reliability"]["status"],
            "unavailable_no_fresh_transport_confirmatory_bootstrap",
        )
        self.assertIsNone(record["reliability"]["decision_confidence_interval"])

    def test_rationale_is_exactly_reproducible_and_contains_no_intent_claim(self) -> None:
        record = trace.build_reasoning_trace([prediction("r1", "task-a", 1)])[0]
        reasoning = record["reasoning_trace"]
        self.assertEqual(reasoning["rationale"], trace.render_rationale(reasoning["template_inputs"]))
        lowered = reasoning["rationale"].lower()
        for forbidden in ("i think", "i feel", "the model thinks", "the model feels", "intends to"):
            self.assertNotIn(forbidden, lowered)
        self.assertIn("not private chain-of-thought or emotion", lowered)
        self.assertIn("synthetic or unverified", lowered)

        changed = copy.deepcopy(reasoning["template_inputs"])
        changed["uncertainty_band"] = (
            "high" if changed["uncertainty_band"] != "high" else "low"
        )
        with self.assertRaisesRegex(ValueError, "not entailed"):
            trace.render_rationale(changed)

        changed = copy.deepcopy(reasoning["template_inputs"])
        changed["events"] = ["inspection_to_implementation"]
        with self.assertRaisesRegex(ValueError, "not entailed"):
            trace.render_rationale(changed)

    def test_diagnostics_reject_non_allowlisted_or_unpaired_inputs(self) -> None:
        row = prediction("r1", "task-a", 1)
        with self.assertRaisesRegex(ValueError, "non-allowlisted"):
            trace.build_reasoning_trace(
                [row], diagnostics_by_row={"r1": {"task_text": "future leak"}}
            )
        incomplete = trace.build_reasoning_trace(
            [row],
            diagnostics_by_row={
                "r1": {"sequence_logit_probabilities": probabilities("inspect")}
            },
        )[0]
        self.assertEqual(
            incomplete["proxies"]["source_disagreement"]["status"],
            "unavailable_incomplete_source_probability_pair",
        )
        with self.assertRaisesRegex(ValueError, "unknown prediction row IDs"):
            trace.build_reasoning_trace([row], diagnostics_by_row={"other": {}})

    def test_invalid_probabilities_and_evaluator_summary_drift_fail_closed(self) -> None:
        bad = prediction("r1", "task-a", 1)
        bad["forecast_probabilities_q"] = {
            "inspect": 0.5,
            "edit": 0.5,
            "check_or_finish": 0.5,
        }
        with self.assertRaisesRegex(ValueError, "sum to one"):
            trace.build_reasoning_trace([bad])

        inconsistent = prediction("r1", "task-a", 1)
        inconsistent["predicted_class"] = "edit"
        with self.assertRaisesRegex(ValueError, "predicted_class is inconsistent"):
            trace.build_reasoning_trace([inconsistent])

        boolean = prediction("r1", "task-a", 1)
        boolean["forecast_probabilities_q"] = {
            "inspect": True,
            "edit": False,
            "check_or_finish": False,
        }
        with self.assertRaisesRegex(ValueError, "non-boolean"):
            trace.build_reasoning_trace([boolean])

        missing_consistency = prediction("r1", "task-a", 1)
        del missing_consistency["decision_confidence_from_q"]
        with self.assertRaisesRegex(ValueError, "requires evaluator consistency field"):
            trace.build_reasoning_trace([missing_consistency])

        loose_only = prediction("r1", "task-a", 1)
        loose_only["forecast_probabilities_q"] = {
            "inspect": 0.8,
            "edit": 0.1,
            "check_or_finish": 0.1000000005,
        }
        loose_only["forecast_top_confidence"] = 0.8
        loose_only["decision_confidence_from_q"] = 0.8
        with self.assertRaisesRegex(ValueError, "sum to one"):
            trace.build_reasoning_trace([loose_only])

    def test_output_is_finite_json_and_task_identity_is_hashed(self) -> None:
        row = prediction("r1", "sensitive-task-identity", 1)
        output = trace.build_reasoning_trace([row])
        payload = json.dumps(output, allow_nan=False, sort_keys=True)
        self.assertNotIn("sensitive-task-identity", payload)
        self.assertNotIn('"row_id": "r1"', payload)
        self.assertEqual(len(output[0]["boundary"]["row_id_sha256"]), 64)
        self.assertEqual(len(output[0]["boundary"]["task_id_sha256"]), 64)

    @unittest.skipUnless(
        trace.DESIGN_ARTIFACT_PATH.is_file(), "pinned design artifact is unavailable"
    )
    def test_exact_authenticated_design_artifact_end_to_end(self) -> None:
        output = trace.build_reasoning_trace_from_v4_design_artifact()
        self.assertEqual(len(output), 1606)
        self.assertEqual(
            len({record["boundary"]["task_id_sha256"] for record in output}), 60
        )
        self.assertEqual(sum(record["phase_forecast"]["accepted"] for record in output), 1202)
        self.assertTrue(all(record["provenance"]["input_authenticated"] for record in output))
        self.assertTrue(
            all(
                record["phase_forecast"]["confidence_status"]
                == "development_calibrated_transport_unconfirmed"
                for record in output
            )
        )
        self.assertTrue(
            all(
                "design-only and transport-unconfirmed"
                in record["reasoning_trace"]["rationale"]
                for record in output
            )
        )
        self.assertTrue(
            all(
                record["proxies"]["activation_load_like"]["percentile"] is None
                for record in output
            )
        )
        self.assertTrue(
            all(
                record["proxies"]["source_disagreement"]["status"]
                == "available_authenticated_design_diagnostic"
                for record in output
            )
        )
        self.assertTrue(
            all(
                record["provenance"]["source_binding"]["artifact_sha256"]
                == trace.DESIGN_ARTIFACT_SHA256
                for record in output
            )
        )
        json.dumps(output, allow_nan=False)

        with mock.patch.object(Path, "read_bytes", return_value=b"{}"):
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                trace.build_reasoning_trace_from_v4_design_artifact()


if __name__ == "__main__":
    unittest.main()
