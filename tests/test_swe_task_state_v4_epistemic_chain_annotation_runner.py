from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_annotation_runner.py"
)
spec = importlib.util.spec_from_file_location(
    "swe_task_state_v4_epistemic_chain_annotation_runner", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


CHAIN_TEXT = (
    "The focused test failed. "
    "This supports the source-logic conclusion that the branch is active. "
    "Therefore, I will inspect the branch."
)


def packet(source_digit: str, text: str) -> dict[str, object]:
    source = source_digit * 64
    return {
        "schema_version": 1,
        "kind": module.COMPLETION_PACKET_KIND,
        "annotation_pass": "completion_chain",
        "packet_id_sha256": module.sha256_text("packet\0" + source),
        "source_id_sha256": source,
        "blind_shards": {"independent_a": 2, "independent_b": 5},
        "materialized_assistant_text": {
            "char_start": 0,
            "char_end": len(text),
            "sha256": module.sha256_text(text),
            "text": text,
        },
        "authenticated_boundaries": {
            "ignored_by_model": "REPOSITORY_SECRET_CANARY"
        },
        "annotator_visibility": {
            "assistant_tool_arguments_present": False,
            "complete_prefix_text_present": False,
            "model_features_present": False,
            "repository_or_task_identity_present": False,
            "tool_results_present": False,
        },
    }


def prefix_packet(source_digit: str, locked_record: dict[str, object]) -> dict[str, object]:
    source = source_digit * 64
    hypothesis_text = (
        "This supports the source-logic conclusion that the branch is active."
    )
    prefix = "The focused test is ready to run."
    return {
        "schema_version": 1,
        "kind": module.PREFIX_PACKET_KIND,
        "annotation_pass": "prefix_novelty",
        "packet_id_sha256": module.sha256_text("prefix-packet\0" + source),
        "source_id_sha256": source,
        "blind_shards": {"independent_a": 1, "independent_b": 6},
        "locked_hypothesis": {
            "text": hypothesis_text,
            "sha256": module.sha256_text(hypothesis_text),
            "completion_char_start": locked_record["hypothesis_span"]["start"],
            "completion_char_end": locked_record["hypothesis_span"]["end"],
            "materialized_completion_sha256": locked_record[
                "materialized_completion_sha256"
            ],
        },
        "authenticated_prefix": {
            "source_sha256": module.sha256_text(prefix),
            "source_char_start": 0,
            "source_char_end": len(prefix),
            "annotator_text": prefix,
            "annotator_text_sha256": module.sha256_text(prefix),
            "annotator_char_start": 0,
            "annotator_char_end": len(prefix),
            "removed_ranges": [],
        },
        "annotator_visibility": {
            "completion_chain_slots_present": False,
            "assistant_tool_arguments_present": False,
            "tool_results_present": False,
            "repository_or_task_identity_present": False,
            "model_features_present": False,
        },
    }


def no_chain_proposal() -> dict[str, object]:
    return {
        "decision": "no_chain",
        "unknown_reason": "",
        "evidence_start": 0,
        "evidence_end": 0,
        "evidence_text": "",
        "hypothesis_start": 0,
        "hypothesis_end": 0,
        "hypothesis_text": "",
        "action_start": 0,
        "action_end": 0,
        "action_text": "",
        "evidence_kind": "none",
        "belief_edge": "none",
        "hypothesis_domain": "none",
        "action_intent": "none",
    }


def chain_proposal(text: str = CHAIN_TEXT) -> dict[str, object]:
    evidence = "The focused test failed."
    hypothesis = (
        "This supports the source-logic conclusion that the branch is active."
    )
    action = "Therefore, I will inspect the branch."

    def bounds(piece: str) -> tuple[int, int]:
        start = text.index(piece)
        return start, start + len(piece)

    e_start, e_end = bounds(evidence)
    h_start, h_end = bounds(hypothesis)
    a_start, a_end = bounds(action)
    return {
        "decision": "chain",
        "unknown_reason": "",
        "evidence_start": e_start,
        "evidence_end": e_end,
        "evidence_text": evidence,
        "hypothesis_start": h_start,
        "hypothesis_end": h_end,
        "hypothesis_text": hypothesis,
        "action_start": a_start,
        "action_end": a_end,
        "action_text": action,
        "evidence_kind": "tool_or_test",
        "belief_edge": "supports",
        "hypothesis_domain": "source_logic",
        "action_intent": "inspect",
    }


class FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return "\n".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )


def mock_factory(*, model_path, model_spec, generation_config):
    del model_path, generation_config
    repo = model_spec["repo_id"]

    def generate(prompts, schema, seed):
        del schema, seed
        results = []
        for prompt_text in prompts:
            if repo == "Qwen/Qwen3.5-9B":
                proposal = (
                    chain_proposal()
                    if "The focused test failed." in prompt_text
                    else no_chain_proposal()
                )
            elif repo == "Qwen/Qwen3.5-4B":
                proposal = no_chain_proposal()
            else:
                proposal = chain_proposal()
            text = module.canonical_json_text(proposal)
            results.append(
                module.GenerationResult(
                    text=text,
                    prompt_token_count=len(prompt_text.split()),
                    output_token_count=len(text.split()),
                    finish_reason="stop",
                )
            )
        return results

    return generate, {"load_seconds": 0.01, "mock": True}, FakeTokenizer()


class EpistemicChainAnnotationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = module.validate_config(module.load_json_strict(module.CONFIG_PATH))
        cls.annotation_config, cls.codebook = module.authenticate_inputs(cls.config)

    def test_runner_config_is_frozen_and_keeps_claims_narrow(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["claim_scope"]["semantic_decoding_established"] = True
        with self.assertRaisesRegex(
            module.AnnotationRunnerError, "differs from frozen contract"
        ):
            module.validate_config(changed)
        self.assertTrue(self.config["claim_scope"]["annotation_is_not_lens_decoding"])
        self.assertTrue(self.config["claim_scope"]["target_is_not_private_chain_of_thought"])
        self.assertFalse(self.config["claim_scope"]["semantic_decoding_established"])
        primary_roles = ("independent_a", "independent_b", "adjudicator")
        self.assertEqual(
            len(
                {
                    (
                        self.config["roles"][role]["repo_id"],
                        self.config["roles"][role]["revision"],
                    )
                    for role in primary_roles
                }
            ),
            3,
        )
        self.assertEqual(
            self.config["roles"]["quality_audit"]["revision"],
            self.config["roles"]["adjudicator"]["revision"],
        )

    def test_prompt_is_built_from_strict_text_allowlist(self) -> None:
        item = packet("a", CHAIN_TEXT)
        messages = module.build_messages(
            packet=item,
            codebook=self.codebook,
            annotation_pass="completion_chain",
        )
        serialized = module.canonical_json_text(messages)
        self.assertIn(CHAIN_TEXT, serialized)
        self.assertNotIn("REPOSITORY_SECRET_CANARY", serialized)
        self.assertNotIn(str(item["source_id_sha256"]), serialized)
        self.assertNotIn(str(item["packet_id_sha256"]), serialized)
        self.assertIn("every start/end=0", messages[0]["content"])
        self.assertIn("not a cautious substitute for no_chain", messages[0]["content"])

    def test_independent_prompt_never_accepts_other_lane_annotation(self) -> None:
        item = packet("b", CHAIN_TEXT)
        independent = module.build_messages(
            packet=item,
            codebook=self.codebook,
            annotation_pass="completion_chain",
        )
        adjudication = module.build_messages(
            packet=item,
            codebook=self.codebook,
            annotation_pass="completion_chain",
            candidate_records=(
                self._record(item, chain_proposal()),
                self._record(item, no_chain_proposal()),
            ),
        )
        self.assertNotIn("candidate_annotations", independent[1]["content"])
        payload = json.loads(adjudication[1]["content"])
        self.assertEqual(len(payload["candidate_annotations"]), 2)
        for candidate in payload["candidate_annotations"]:
            self.assertNotIn("annotator_id_sha256", candidate)
            self.assertNotIn("source_id_sha256", candidate)

    def test_packet_validation_fails_closed_on_visibility_or_extra_outcome(self) -> None:
        visible = packet("c", CHAIN_TEXT)
        visible["annotator_visibility"]["model_features_present"] = True
        with self.assertRaisesRegex(module.AnnotationRunnerError, "blinding"):
            module.validate_packet(visible, annotation_pass="completion_chain")
        extra = packet("c", CHAIN_TEXT)
        extra["official_outcome"] = "resolved"
        with self.assertRaisesRegex(module.AnnotationRunnerError, "fields changed"):
            module.validate_packet(extra, annotation_pass="completion_chain")

    def test_exact_span_conversion_and_mismatch_rejection(self) -> None:
        item = packet("d", CHAIN_TEXT)
        record = self._record(item, chain_proposal())
        self.assertTrue(record["has_chain"])
        self.assertEqual(
            record["exact_signature"],
            "tool_or_test>supports>source_logic>motivates>inspect",
        )
        bad = chain_proposal()
        bad["hypothesis_start"] += 1
        with self.assertRaisesRegex(module.AnnotationRunnerError, "exactly match"):
            self._record(item, bad)

        inconsistent = no_chain_proposal()
        inconsistent["evidence_text"] = "ignored model text"
        with self.assertRaisesRegex(module.AnnotationRunnerError, "empty sentinels"):
            self._record(item, inconsistent)

    def test_more_than_64_rows_requires_explicit_full_run(self) -> None:
        rows = [packet(f"{index % 10}", f"row {index}") for index in range(65)]
        # Packet ids collide when source digits repeat; selection itself has no
        # identity dependency and still enforces the operational interlock.
        with self.assertRaisesRegex(module.AnnotationRunnerError, "allow-full-run"):
            module.select_packets(rows, offset=0, limit=None, allow_full_run=False)
        self.assertEqual(
            len(module.select_packets(rows, offset=0, limit=None, allow_full_run=True)),
            65,
        )

    def test_explicit_packet_selection_is_exact_and_ordered(self) -> None:
        rows = [packet("6", "first"), packet("7", "second"), packet("8", "third")]
        wanted = [rows[2]["packet_id_sha256"], rows[0]["packet_id_sha256"]]
        selected = module.select_packets(
            rows,
            offset=0,
            limit=None,
            allow_full_run=False,
            packet_ids=wanted,
        )
        self.assertEqual([item["packet_id_sha256"] for item in selected], wanted)
        with self.assertRaisesRegex(module.AnnotationRunnerError, "cannot be combined"):
            module.select_packets(
                rows,
                offset=1,
                limit=None,
                allow_full_run=False,
                packet_ids=wanted,
            )

    def test_agreement_metrics_are_pass_specific(self) -> None:
        positive = self._record(packet("9", CHAIN_TEXT), chain_proposal())
        completion = module.independent_agreement_metrics(
            [(positive, copy.deepcopy(positive))],
            annotation_pass="completion_chain",
        )
        self.assertIsNone(completion["has_chain_cohen_kappa"])
        self.assertEqual(
            completion["has_chain_kappa_undefined_reason"],
            "degenerate_single_category_marginals",
        )
        self.assertEqual(completion["exact_graph_agreement"], 1.0)
        left = {"annotation_status": "available", "novelty_status": "novel"}
        right = {"annotation_status": "available", "novelty_status": "prefix_exposed"}
        novelty = module.independent_agreement_metrics(
            [(left, right)], annotation_pass="prefix_novelty"
        )
        self.assertEqual(novelty["novelty_exact_agreement"], 0.0)
        self.assertEqual(novelty["novelty_cohen_kappa"], 0.0)

    def test_prefix_novelty_pass_preserves_locked_chain_and_only_adds_novelty(self) -> None:
        completion_packet = packet("e", CHAIN_TEXT)
        locked = self._record(completion_packet, chain_proposal())
        novelty_packet = prefix_packet("e", locked)
        messages = module.build_messages(
            packet=novelty_packet,
            codebook=self.codebook,
            annotation_pass="prefix_novelty",
        )
        payload = json.loads(messages[1]["content"])
        self.assertEqual(
            set(payload), {"visible_prefix", "locked_hypothesis"}
        )
        final = module.proposal_to_record(
            proposal={"decision": "novel", "unknown_reason": ""},
            packet=novelty_packet,
            annotation_pass="prefix_novelty",
            annotator_id_sha256="f" * 64,
            annotation_config=self.annotation_config,
            locked_completion_record=locked,
        )
        self.assertEqual(final["novelty_status"], "novel")
        self.assertEqual(final["exact_signature"], locked["exact_signature"])

    def test_mock_two_independent_lanes_adjudication_and_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                packet("1", "I will inspect the file."),
                packet("2", CHAIN_TEXT),
            ]
            packet_path = root / "packets.jsonl"
            packet_path.write_text(
                "".join(module.canonical_json_text(item) + "\n" for item in rows),
                encoding="utf-8",
            )
            packet_manifest_path = root / "packets.json"
            packet_manifest = {
                "schema_version": 1,
                "kind": "swe_task_state_v4_epistemic_chain_packet_manifest",
                "annotation_pass": "completion_chain",
                "scope": {
                    "development_data_only": True,
                    "reserved_validation_closed": True,
                    "reserved_validation_accessed": False,
                },
                "packets": {
                    "path": packet_path.name,
                    "sha256": module.sha256_file(packet_path),
                    "count": len(rows),
                },
            }
            packet_manifest_path.write_bytes(
                module.canonical_json_bytes(packet_manifest) + b"\n"
            )
            lane_a_path = root / "lane-a.json"
            lane_b_path = root / "lane-b.json"
            adjudicator_path = root / "adjudicator.json"
            final_path = root / "final.json"
            module.run_lane(
                config=self.config,
                packet_manifest_path=packet_manifest_path,
                expected_packet_manifest_sha256=module.sha256_file(packet_manifest_path),
                role="independent_a",
                output_manifest_path=lane_a_path,
                offset=0,
                limit=2,
                allow_full_run=False,
                generator_factory=mock_factory,
                verify_model_contents=False,
            )
            module.run_lane(
                config=self.config,
                packet_manifest_path=packet_manifest_path,
                expected_packet_manifest_sha256=module.sha256_file(packet_manifest_path),
                role="independent_b",
                output_manifest_path=lane_b_path,
                offset=0,
                limit=2,
                allow_full_run=False,
                generator_factory=mock_factory,
                verify_model_contents=False,
            )
            adjudicator_manifest = module.run_lane(
                config=self.config,
                packet_manifest_path=packet_manifest_path,
                expected_packet_manifest_sha256=module.sha256_file(packet_manifest_path),
                role="adjudicator",
                output_manifest_path=adjudicator_path,
                offset=0,
                limit=2,
                allow_full_run=False,
                lane_a_manifest_path=lane_a_path,
                lane_b_manifest_path=lane_b_path,
                generator_factory=mock_factory,
                verify_model_contents=False,
            )
            self.assertEqual(
                adjudicator_manifest["counts"]["semantic_disagreements_selected"],
                1,
            )
            final = module.finalize_lanes(
                config=self.config,
                packet_manifest_path=packet_manifest_path,
                lane_a_manifest_path=lane_a_path,
                lane_b_manifest_path=lane_b_path,
                adjudicator_manifest_path=adjudicator_path,
                output_manifest_path=final_path,
            )
            self.assertEqual(
                final["counts"],
                {
                    "records": 2,
                    "exact_semantic_agreements": 1,
                    "semantic_disagreements": 1,
                    "third_model_adjudications": 1,
                    "unresolved_explicit_unknowns": 0,
                },
            )
            self.assertEqual(final["agreement_rate"], 0.5)
            metrics = final["independent_agreement_metrics"]
            self.assertEqual(metrics["paired_rows"], 2)
            self.assertEqual(metrics["has_chain_exact_agreement"], 0.5)
            self.assertEqual(metrics["has_chain_cohen_kappa"], 0.0)
            self.assertIsNone(metrics["exact_graph_agreement"])
            output_records = [
                json.loads(line)
                for line in (root / "final-records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertFalse(output_records[0]["has_chain"])
            self.assertTrue(output_records[1]["has_chain"])
            audit_records = [
                json.loads(line)
                for line in (root / "final-audit.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                audit_records[1]["adjudication_reason"],
                "independent_semantic_disagreement_resolved_by_third_model",
            )
            hashes = audit_records[1]["source_record_hashes"]
            self.assertTrue(
                module._is_sha256(hashes["independent_a_annotation_record_sha256"])
            )
            self.assertTrue(
                module._is_sha256(hashes["adjudicator_lane_record_sha256"])
            )

    def _record(self, item, proposal):
        return module.proposal_to_record(
            proposal=proposal,
            packet=item,
            annotation_pass="completion_chain",
            annotator_id_sha256="d" * 64,
            annotation_config=self.annotation_config,
        )


if __name__ == "__main__":
    unittest.main()
