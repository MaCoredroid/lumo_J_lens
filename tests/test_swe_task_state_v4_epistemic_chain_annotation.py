from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_annotation.py"

spec = importlib.util.spec_from_file_location(
    "swe_task_state_v4_epistemic_chain_annotation", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def synthetic_pair() -> tuple[dict[str, object], dict[str, object], str]:
    prefix = "system and visible history\n" + module.GENERATION_BOUNDARY
    reasoning = "The inspected branch returns the wrong shape."
    content = "\nThis supports a data-shape diagnosis. I will edit the branch.\n"
    tool_secret = "DO_NOT_EXPOSE_ARGUMENTS"
    result_secret = "DO_NOT_EXPOSE_RESULT"
    tool_block = (
        module.TOOL_CALL_OPEN
        + "\n<function=run_shell_command>\n<parameter=command>\n"
        + tool_secret
        + "\n</parameter>\n</function>\n"
        + module.TOOL_CALL_CLOSE
    )
    assistant_rendered = reasoning + "\n" + module.THINK_CLOSE + content + tool_block
    following_text = (
        prefix
        + assistant_rendered
        + module.MESSAGE_END
        + "\n"
        + module.TOOL_RESPONSE_OPEN
        + result_secret
        + module.TOOL_RESPONSE_CLOSE
        + "\n"
        + module.GENERATION_BOUNDARY
    )
    assistant_text = reasoning + "\n" + content
    next_request_sha = "b" * 64
    current = {
        "source_id_sha256": "a" * 64,
        "task_id": "task-1",
        "request_index": 1,
        "global_request_index": 1,
        "text": prefix,
        "text_sha256": module.sha256_text(prefix),
        "next_completion": {
            "status": "materialized_in_following_request",
            "assistant_text_sha256": module.sha256_text(assistant_text),
            "next_request_global_index": 2,
            "next_request_sha256": next_request_sha,
        },
    }
    following = {
        "source_id_sha256": "c" * 64,
        "task_id": "task-1",
        "request_index": 2,
        "global_request_index": 2,
        "text": following_text,
        "text_sha256": module.sha256_text(following_text),
        "raw_request_sha256": next_request_sha,
        "next_completion": {"status": "terminal_no_tool_response"},
    }
    return current, following, assistant_text


def positive_record(
    *, source: str, completion: str, stage: str = "completion"
) -> dict[str, object]:
    evidence_text = "The inspected branch returns the wrong shape."
    hypothesis_text = "This supports a data-shape diagnosis."
    action_text = "I will edit the branch."

    def span(piece: str) -> dict[str, object]:
        start = completion.index(piece)
        return {
            "start": start,
            "end": start + len(piece),
            "text_sha256": module.sha256_text(piece),
        }

    return {
        "source_id_sha256": source,
        "materialized_completion_sha256": module.sha256_text(completion),
        "annotation_status": "available",
        "unknown_reason": None,
        "has_chain": True,
        "evidence_span": span(evidence_text),
        "hypothesis_span": span(hypothesis_text),
        "action_span": span(action_text),
        "evidence_kind": "code",
        "belief_edge": "supports",
        "hypothesis_domain": "data_type_shape",
        "action_intent": "edit",
        "novelty_status": None if stage == "completion" else "novel",
        "exact_signature": "code>supports>data_type_shape>motivates>edit",
        "annotator_id_sha256": "d" * 64,
        "annotator_prompt_or_model_identity_sha256": (
            module.ANNOTATOR_PROMPT_OR_MODEL_IDENTITY_SHA256
        ),
        "codebook_sha256": module.CODEBOOK_FILE_SHA256,
    }


class NullWriter:
    def write(self, value: str) -> int:
        return len(value)


class EpistemicChainAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = module.validate_config(module.load_json_strict(module.CONFIG_PATH))
        cls.codebook = module.validate_codebook(
            module.load_json_strict(module.CODEBOOK_PATH)
        )

    def test_config_is_exactly_frozen(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["ontology"]["action_intent"].append("finish")
        with self.assertRaisesRegex(
            module.AnnotationPacketError, "differs from the frozen contract"
        ):
            module.validate_config(changed)

    def test_packet_materialization_disclosure_does_not_claim_annotation_or_decoding(
        self,
    ) -> None:
        disclosure = module.packet_materialization_disclosure(
            self.config,
            annotation_pass="completion_chain",
            locked_chain_records_present=False,
        )
        self.assertEqual(
            disclosure["status"],
            "packet_materialization_passed_target_pass_annotation_not_run",
        )
        self.assertEqual(
            disclosure["scope"],
            {
                "artifact_kind": "blinded_annotation_packets_only",
                "development_data_only": True,
                "reserved_validation_closed": True,
                "reserved_validation_accessed": False,
            },
        )
        self.assertEqual(
            disclosure["annotation_execution"],
            {
                "status": "not_run_for_target_pass",
                "annotation_not_run": True,
                "target_annotation_pass": "completion_chain",
                "producer_executes_annotation": False,
                "target_pass_annotation_records_in_artifact": False,
                "upstream_locked_chain_records_input_present": False,
            },
        )
        claims = disclosure["claim_scope"]
        self.assertTrue(claims["target_definition_frozen"])
        self.assertTrue(claims["target_is_future_trace_visible_proposition_chain"])
        for name in (
            "semantic_sentence_or_chain_decoding_established_before_gate_and_evaluation",
            "cot_or_cot_alike_decoding_established",
            "private_chain_of_thought_decoding_established",
            "affect_or_emotion_decoding_established",
        ):
            self.assertIs(claims[name], False)
        prefix = module.packet_materialization_disclosure(
            self.config,
            annotation_pass="prefix_novelty",
            locked_chain_records_present=True,
        )
        self.assertEqual(
            prefix["annotation_execution"]["target_annotation_pass"],
            "prefix_novelty",
        )
        self.assertTrue(
            prefix["annotation_execution"][
                "upstream_locked_chain_records_input_present"
            ]
        )
        with self.assertRaisesRegex(
            module.AnnotationPacketError, "annotation-pass semantics"
        ):
            module.packet_materialization_disclosure(
                self.config,
                annotation_pass="completion_chain",
                locked_chain_records_present=True,
            )

    def test_codebook_is_exactly_frozen_and_covers_required_examples(self) -> None:
        binding = self.config["annotation_codebook_contract"]
        self.assertEqual(module.sha256_file(module.CODEBOOK_PATH), binding["sha256"])
        self.assertEqual(module.CODEBOOK_PATH.stat().st_size, binding["size_bytes"])
        self.assertEqual(self.codebook["schema_version"], binding["schema_version"])
        self.assertEqual(self.codebook["kind"], binding["kind"])
        self.assertEqual(
            self.codebook["annotator_prompt_contract"][
                "prompt_or_model_identity_sha256"
            ],
            binding["annotator_prompt_or_model_identity_sha256"],
        )
        self.assertEqual(
            {item["belief_edge"] for item in self.codebook["positive_examples"]},
            {"supports", "refutes", "narrows"},
        )
        self.assertEqual(len(self.codebook["negative_examples"]), 7)
        self.assertEqual(len(self.codebook["novelty_examples"]), 4)
        changed = copy.deepcopy(self.codebook)
        changed["chain_rule"][
            "maximum_intervening_prose_clauses_between_H_and_A"
        ] = 3
        with self.assertRaisesRegex(
            module.AnnotationPacketError, "differs from the frozen contract"
        ):
            module.validate_codebook(changed)

    def test_annotation_record_rejects_wrong_bound_contract_digests(self) -> None:
        current, _following, completion = synthetic_pair()
        record = positive_record(
            source=str(current["source_id_sha256"]), completion=completion
        )
        wrong_codebook = copy.deepcopy(record)
        wrong_codebook["codebook_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            module.AnnotationPacketError, "differs from frozen binding"
        ):
            module.validate_annotation_record(
                wrong_codebook,
                config=self.config,
                stage="completion",
                completion_text=completion,
            )
        wrong_identity = copy.deepcopy(record)
        wrong_identity["annotator_prompt_or_model_identity_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            module.AnnotationPacketError, "differs from frozen binding"
        ):
            module.validate_annotation_record(
                wrong_identity,
                config=self.config,
                stage="completion",
                completion_text=completion,
            )

    def test_exact_rendered_extension_reconstructs_only_assistant_prose(self) -> None:
        current, following, expected = synthetic_pair()
        reconstructed = module.reconstruct_materialized_completion(current, following)
        self.assertEqual(reconstructed["assistant_text"], expected)
        self.assertEqual(
            reconstructed["assistant_text_sha256"], module.sha256_text(expected)
        )
        packet = module.make_completion_packet(
            current=current,
            reconstruction=reconstructed,
            shard_count=8,
        )
        serialized = json.dumps(packet)
        self.assertNotIn("DO_NOT_EXPOSE_ARGUMENTS", serialized)
        self.assertNotIn("DO_NOT_EXPOSE_RESULT", serialized)
        self.assertNotIn("repository", packet)
        self.assertNotIn("task_id", packet)
        self.assertFalse(
            packet["annotator_visibility"]["assistant_tool_arguments_present"]
        )
        self.assertFalse(packet["annotator_visibility"]["tool_results_present"])

    def test_missing_exact_thinking_prefix_fails_closed(self) -> None:
        current, following, _expected = synthetic_pair()
        prefix = str(current["text"])
        following["text"] = (
            prefix[: -len("<think>\n")]
            + module.TOOL_CALL_OPEN
            + "\n...\n"
            + module.TOOL_CALL_CLOSE
            + module.MESSAGE_END
        )
        current["next_completion"]["assistant_text_sha256"] = module.sha256_text("")
        with self.assertRaisesRegex(
            module.AnnotationPacketError, "config correction required"
        ):
            module.reconstruct_materialized_completion(current, following)

    def test_annotation_record_validates_spans_slots_and_stage(self) -> None:
        current, _following, completion = synthetic_pair()
        record = positive_record(
            source=str(current["source_id_sha256"]), completion=completion
        )
        observed = module.validate_annotation_record(
            record,
            config=self.config,
            stage="completion",
            completion_text=completion,
        )
        self.assertEqual(observed["belief_edge"], "supports")

        final = copy.deepcopy(record)
        final["novelty_status"] = "novel"
        module.validate_annotation_record(
            final,
            config=self.config,
            stage="final",
            completion_text=completion,
        )

        bad = copy.deepcopy(record)
        bad["action_intent"] = "inspect"
        with self.assertRaisesRegex(module.AnnotationPacketError, "signature"):
            module.validate_annotation_record(
                bad,
                config=self.config,
                stage="completion",
                completion_text=completion,
            )

    def test_no_chain_and_explicit_unknown_are_strict(self) -> None:
        current, _following, completion = synthetic_pair()
        record = positive_record(
            source=str(current["source_id_sha256"]), completion=completion
        )
        record["has_chain"] = False
        for field in (
            "evidence_span",
            "hypothesis_span",
            "action_span",
            "evidence_kind",
            "belief_edge",
            "hypothesis_domain",
            "action_intent",
            "novelty_status",
            "exact_signature",
        ):
            record[field] = None
        module.validate_annotation_record(
            record,
            config=self.config,
            stage="final",
            completion_text=completion,
        )
        record["annotation_status"] = "unknown"
        record["unknown_reason"] = "completion_semantics_ambiguous"
        record["has_chain"] = None
        module.validate_annotation_record(
            record,
            config=self.config,
            stage="final",
            completion_text=completion,
        )
        record["unknown_reason"] = None
        with self.assertRaisesRegex(module.AnnotationPacketError, "frozen reason"):
            module.validate_annotation_record(
                record,
                config=self.config,
                stage="final",
                completion_text=completion,
            )

    def test_prefix_packet_is_separate_and_removes_tool_blocks(self) -> None:
        current, following, completion = synthetic_pair()
        current["text"] = (
            "visible prose before\n"
            + module.TOOL_CALL_OPEN
            + "SECRET_CALL"
            + module.TOOL_CALL_CLOSE
            + "\nvisible middle\n"
            + module.TOOL_RESPONSE_OPEN
            + "SECRET_RESULT"
            + module.TOOL_RESPONSE_CLOSE
            + "\nvisible prose after\n"
            + module.GENERATION_BOUNDARY
        )
        current["text_sha256"] = module.sha256_text(str(current["text"]))
        reconstruction = {
            "assistant_text": completion,
            "assistant_text_sha256": module.sha256_text(completion),
        }
        record = positive_record(
            source=str(current["source_id_sha256"]), completion=completion
        )
        packet = module.make_prefix_novelty_packet(
            current=current,
            reconstruction=reconstruction,
            locked_record=record,
            config=self.config,
            shard_count=8,
        )
        assert packet is not None
        serialized = json.dumps(packet)
        self.assertNotIn("SECRET_CALL", serialized)
        self.assertNotIn("SECRET_RESULT", serialized)
        self.assertNotIn("belief_edge", serialized)
        self.assertNotIn("evidence_kind", serialized)
        self.assertIn("visible prose before", packet["authenticated_prefix"]["annotator_text"])
        self.assertEqual(len(packet["authenticated_prefix"]["removed_ranges"]), 2)

    def test_blind_shards_are_deterministic_distinct_and_pass_specific(self) -> None:
        source = "a" * 64
        first = module.blind_shard_assignment(
            source, annotation_pass="completion_chain", shard_count=8
        )
        second = module.blind_shard_assignment(
            source, annotation_pass="completion_chain", shard_count=8
        )
        novelty = module.blind_shard_assignment(
            source, annotation_pass="prefix_novelty", shard_count=8
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first["independent_a"], first["independent_b"])
        self.assertNotEqual(first, novelty)

    def test_paths_reject_forbidden_text_before_access(self) -> None:
        with self.assertRaisesRegex(
            module.AnnotationPacketError, "before filesystem access"
        ):
            module.lexical_path_preflight(
                [Path("/tmp/reserved-epistemic-chain/packets.jsonl")]
            )

    def test_locked_record_loader_rejects_duplicates(self) -> None:
        current, _following, completion = synthetic_pair()
        record = positive_record(
            source=str(current["source_id_sha256"]), completion=completion
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "locked.jsonl"
            line = json.dumps(record, sort_keys=True) + "\n"
            path.write_text(line + line)
            with self.assertRaisesRegex(
                module.AnnotationPacketError, "duplicate locked annotation source"
            ):
                module.load_annotation_records_jsonl(
                    path,
                    expected_sha256=module.sha256_file(path),
                    config=self.config,
                )

    def test_pinned_bundle_freezes_only_the_exact_nonprefix_row_as_unknown(self) -> None:
        source = self.config["source_contract"]
        alignment = module.validate_alignment_index(
            module.load_json_strict(
                ROOT / source["label_free_alignment_index"]["path"]
            ),
            expected_row_count=source["development_prompt_bundle"]["row_count"],
            expected_stable_count=source["label_free_alignment_index"][
                "stable_row_count"
            ],
        )
        counts = module.export_packets(
            config=self.config,
            prompt_path=ROOT / source["development_prompt_bundle"]["path"],
            alignment_rows=alignment,
            packet_handle=NullWriter(),
            annotation_pass="completion_chain",
            shard_count=8,
        )
        self.assertEqual(counts["stable_materialized_rows"], 1549)
        self.assertEqual(counts["frozen_unknown_exclusions"], 1)
        self.assertEqual(counts["packets"], 1548)


if __name__ == "__main__":
    unittest.main()
