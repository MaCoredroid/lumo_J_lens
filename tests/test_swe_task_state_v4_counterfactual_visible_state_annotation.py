from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "scripts"
    / "swe_task_state_v4_counterfactual_visible_state_annotation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "swe_task_state_v4_counterfactual_visible_state_annotation", MODULE_PATH
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class CounterfactualVisibleStateAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = module.validate_config(module.load_json(module.CONFIG_PATH))
        cls.prompt_ids = {
            module.sha256_text(f"synthetic-stage-a-prompt-{index}")
            for index in range(240)
        }

    def completion_bundle(
        self,
        *,
        overrides: dict[str, tuple[str, str | None]] | None = None,
    ) -> dict:
        overrides = overrides or {}
        records = []
        for prompt_id in sorted(self.prompt_ids):
            status, text = overrides.get(
                prompt_id,
                (
                    "complete",
                    "The visible completion reports no special state and stops.",
                ),
            )
            records.append(
                {
                    "prompt_id": prompt_id,
                    "completion_status": status,
                    "completion_text": text,
                }
            )
        contract = self.config["completion_bundle_contract"]
        return {
            "schema_version": 1,
            "kind": contract["kind"],
            "status": contract["status"],
            "generation_prompt_bundle_sha256": self.config[
                "frozen_selector_scaffold"
            ]["stage_a_generation_prompts"]["sha256"],
            "surface_policy": contract["surface_policy"],
            "records": records,
            "stage_b_records_present": False,
            "reserved_validation_access_authorized": False,
        }

    def absent_response(self) -> dict:
        absent_values = {
            "evidence_assessment": "no_visible_assessment",
            "epistemic_commitment": "no_visible_epistemic_commitment",
            "action_class": "no_visible_action",
            "action_commitment": "no_visible_action_commitment",
            "recheck_behavior": "absent",
            "seek_information_behavior": "absent",
            "preserve_alternatives_behavior": "absent",
            "explicit_doubt_language": "absent",
            "explicit_affect_language": "absent",
            "pressure_rush_acknowledgment": "absent",
        }
        return {
            "schema_version": 1,
            "labels": {
                target: {"value": absent_values[target], "quote": None}
                for target in self.config["quote_first_interface"]["target_order"]
            },
        }

    def write_input_materialization(
        self, directory: Path, bundle: dict
    ) -> tuple[Path, dict]:
        completion_path = directory / "stage-a-completions.json"
        module._write_json(completion_path, bundle)
        input_dir = directory / "annotation-inputs"
        with mock.patch.object(
            module, "validate_prelock_bindings", return_value=set(self.prompt_ids)
        ):
            manifest = module.materialize_inputs(completion_path, input_dir)
        return input_dir, manifest

    def test_config_is_hash_frozen_and_claims_remain_strictly_limited(self) -> None:
        self.assertEqual(module.sha256_file(module.CONFIG_PATH), module.CONFIG_SHA256)
        self.assertTrue(module._is_sha256(module.CONFIG_SHA256))
        claims = self.config["claim_scope"]
        self.assertTrue(claims["annotation_contract_and_cpu_materializer_implemented"])
        for key in (
            "stage_a_completions_generated",
            "stage_a_model_annotation_run",
            "stage_a_annotations_locked",
            "condition_key_join_or_pair_analysis_run",
            "activation_capture_or_decoder_fit_run",
            "visible_behavioral_or_language_effect_established",
            "incremental_activation_readout_established",
            "private_chain_of_thought_reconstructed",
            "subjective_confidence_or_doubt_inferred",
            "experienced_stress_inferred",
            "experienced_emotion_inferred",
            "COT_or_COT_like_target_pooled_with_affect_or_state_targets",
            "outer_or_reserved_validation_generalization_established",
        ):
            self.assertFalse(claims[key])

    def test_frozen_scaffold_hashes_validate_without_condition_or_split_access(self) -> None:
        called: list[str] = []
        original = module._validate_bound_file

        def recording_validator(record: dict, label: str) -> Path:
            called.append(label)
            return original(record, label)

        with mock.patch.object(
            module, "_validate_bound_file", side_effect=recording_validator
        ):
            observed = module.validate_prelock_bindings(self.config)
        self.assertEqual(len(observed), 240)
        self.assertEqual(
            called,
            [
                "config",
                "implementation",
                "tests",
                "materialization_manifest",
                "stage_a_generation_prompts",
            ],
        )
        self.assertNotIn("split_manifest_identity_only", called)
        self.assertNotIn("stage_a_condition_key_post_lock_only", called)

    def test_targets_cover_state_language_and_keep_cot_separate(self) -> None:
        expected = {
            "evidence_assessment",
            "epistemic_commitment",
            "action_class",
            "action_commitment",
            "recheck_behavior",
            "seek_information_behavior",
            "preserve_alternatives_behavior",
            "explicit_doubt_language",
            "explicit_affect_language",
            "pressure_rush_acknowledgment",
        }
        self.assertEqual(
            set(self.config["quote_first_interface"]["target_order"]), expected
        )
        semantics = self.config["target_semantics"]
        self.assertTrue(
            semantics["labels_are_observable_completion_language_or_behavior_only"]
        )
        self.assertTrue(
            semantics["labels_are_not_direct_measurements_of_hidden_or_subjective_state"]
        )
        pairing = self.config["post_lock_pairing_contract"]
        self.assertIn(
            "COT_or_COT_like_semantic_chain_targets",
            pairing["visible_labels_remain_separate_from"],
        )
        self.assertIn(
            "subjective_confidence_or_doubt",
            pairing["visible_labels_remain_separate_from"],
        )
        self.assertIn(
            "experienced_stress_or_emotion",
            pairing["visible_labels_remain_separate_from"],
        )

    def test_materializer_packet_is_exact_condition_blind_and_unicode_preserving(self) -> None:
        prompt_id = sorted(self.prompt_ids)[0]
        visible = "I’m uncertain — re-run Δ.\nThen preserve option β."
        bundle = self.completion_bundle(overrides={prompt_id: ("complete", visible)})
        packets, identity = module.build_annotation_inputs(
            bundle,
            expected_prompt_ids=set(self.prompt_ids),
            config=self.config,
        )
        packet = next(row for row in packets if row["visible_completion_text"] == visible)
        self.assertEqual(
            set(packet), {"annotation_id", "visible_completion_text"}
        )
        self.assertEqual(packet["visible_completion_text"], visible)
        serialized = module.canonical_json_bytes(packet).decode("ascii")
        for forbidden in (
            "condition_id",
            "evidence_level",
            "pressure_level",
            "selector_code",
            "activation_features",
            "task_id",
            "repository",
            "source_id",
            "prompt_id",
        ):
            self.assertNotIn(forbidden, serialized)
        identity_row = next(
            row for row in identity["records"] if row["annotation_id"] == packet["annotation_id"]
        )
        self.assertEqual(identity_row["prompt_id"], prompt_id)
        self.assertTrue(identity["model_annotators_must_not_read_this_file"])
        self.assertFalse(
            identity[
                "selector_condition_activation_outcome_task_and_repository_fields_present"
            ]
        )

    def test_completion_schema_rejects_identity_or_condition_leakage(self) -> None:
        bundle = self.completion_bundle()
        bundle["records"][0]["pressure_level"] = "neutral"
        with self.assertRaisesRegex(module.VisibleStateAnnotationError, "keys changed"):
            module.build_annotation_inputs(
                bundle,
                expected_prompt_ids=set(self.prompt_ids),
                config=self.config,
            )
        bundle = self.completion_bundle()
        bundle["task_id"] = "leak"
        with self.assertRaisesRegex(module.VisibleStateAnnotationError, "keys changed"):
            module.build_annotation_inputs(
                bundle,
                expected_prompt_ids=set(self.prompt_ids),
                config=self.config,
            )

    def test_completion_schema_rejects_missing_duplicate_and_invalid_status_rows(self) -> None:
        bundle = self.completion_bundle()
        bundle["records"][-1] = dict(bundle["records"][0])
        with self.assertRaisesRegex(
            module.VisibleStateAnnotationError, "unknown or duplicate"
        ):
            module.build_annotation_inputs(
                bundle,
                expected_prompt_ids=set(self.prompt_ids),
                config=self.config,
            )
        bundle = self.completion_bundle()
        bundle["records"][0]["completion_status"] = "generation_error"
        with self.assertRaisesRegex(module.VisibleStateAnnotationError, "must be null"):
            module.build_annotation_inputs(
                bundle,
                expected_prompt_ids=set(self.prompt_ids),
                config=self.config,
            )

    def test_exact_quote_resolution_uses_unicode_code_point_offsets(self) -> None:
        text = "Préface Δ.\nI’m not fully certain; I will re-run the suite."
        raw = {
            "value": "present",
            "quote": "I’m not fully certain",
        }
        label = module._normalize_label(
            target="explicit_doubt_language",
            raw=raw,
            visible_text=text,
            config=self.config,
        )
        self.assertEqual(label["value"], "present")
        self.assertEqual(label["resolution_status"], "resolved_exact_unique_quote")
        self.assertEqual(
            text[label["quote_start"] : label["quote_end"]], raw["quote"]
        )
        self.assertEqual(label["quote_start"], text.index(raw["quote"]))

    def test_quote_resolver_fails_closed_without_normalization_or_guessing(self) -> None:
        cases = (
            (
                {"value": "present", "quote": "uncertain"},
                "uncertain, still uncertain",
                "quote_nonunique",
            ),
            (
                {"value": "present", "quote": "not certain"},
                "not  certain",
                "quote_not_found_exactly",
            ),
            (
                {"value": "present", "quote": ""},
                "uncertain",
                "required_quote_missing_or_empty",
            ),
            (
                {"value": "absent", "quote": "none"},
                "none",
                "quote_forbidden_for_value",
            ),
            (
                {"value": "invented", "quote": None},
                "anything",
                "invalid_label_value",
            ),
        )
        for raw, text, expected_status in cases:
            with self.subTest(expected_status=expected_status):
                label = module._normalize_label(
                    target="explicit_doubt_language",
                    raw=raw,
                    visible_text=text,
                    config=self.config,
                )
                self.assertEqual(label["value"], "unknown")
                self.assertIsNone(label["quote"])
                self.assertEqual(label["resolution_status"], expected_status)

    def test_overlapping_exact_occurrences_are_ambiguous(self) -> None:
        self.assertEqual(module._literal_occurrences("aaa", "aa"), [0, 1])
        label = module._normalize_label(
            target="recheck_behavior",
            raw={"value": "present", "quote": "aa"},
            visible_text="aaa",
            config=self.config,
        )
        self.assertEqual(label["value"], "unknown")
        self.assertEqual(label["resolution_status"], "quote_nonunique")

    def test_malformed_transport_and_missing_response_fail_closed(self) -> None:
        annotation_id = module.sha256_text("annotation")
        completion_sha = module.sha256_text("visible")
        for raw, reason in (
            (None, "missing_model_response"),
            (
                {
                    "annotation_id": annotation_id,
                    "transport_status": "timeout",
                    "response": None,
                },
                "transport_timeout",
            ),
            (
                {
                    "annotation_id": annotation_id,
                    "transport_status": "ok",
                    "response": {"schema_version": 1, "labels": {}},
                },
                "invalid_response_schema",
            ),
        ):
            with self.subTest(reason=reason):
                normalized = module.normalize_annotation_record(
                    annotation_id=annotation_id,
                    completion_text_sha256=completion_sha,
                    visible_text="visible",
                    raw_record=raw,
                    config=self.config,
                )
                self.assertEqual(normalized["interface_status"], "all_unknown")
                self.assertEqual(
                    {item["resolution_status"] for item in normalized["labels"].values()},
                    {reason},
                )

    def test_unavailable_completions_have_no_packet_and_become_unknown(self) -> None:
        first, second = sorted(self.prompt_ids)[:2]
        bundle = self.completion_bundle(
            overrides={first: ("empty", ""), second: ("generation_error", None)}
        )
        packets, identity = module.build_annotation_inputs(
            bundle,
            expected_prompt_ids=set(self.prompt_ids),
            config=self.config,
        )
        self.assertEqual(len(packets), 238)
        unavailable = [
            row for row in identity["records"] if not row["model_packet_present"]
        ]
        self.assertEqual(len(unavailable), 2)
        raw_bundle = {
            "schema_version": 1,
            "kind": self.config["raw_annotation_bundle_contract"]["kind"],
            "status": self.config["raw_annotation_bundle_contract"]["status"],
            "model_inputs_sha256": module.sha256_text("packets"),
            "annotator_instructions_sha256": module.sha256_text("instructions"),
            "annotator_lineage_sha256": module.sha256_text("annotator"),
            "records": [
                {
                    "annotation_id": packet["annotation_id"],
                    "transport_status": "timeout",
                    "response": None,
                }
                for packet in packets
            ],
            "reserved_validation_access_authorized": False,
        }
        locked, _ = module.build_locked_annotations(
            packets=packets,
            identity_key=identity,
            raw_bundle=raw_bundle,
            model_inputs_sha256=module.sha256_text("packets"),
            annotator_instructions_sha256=module.sha256_text("instructions"),
            config=self.config,
        )
        unavailable_ids = {row["annotation_id"] for row in unavailable}
        unavailable_locked = [
            row for row in locked if row["annotation_id"] in unavailable_ids
        ]
        self.assertEqual(len(unavailable_locked), 2)
        for row in unavailable_locked:
            self.assertEqual(row["interface_status"], "all_unknown")
            self.assertEqual(
                {label["resolution_status"] for label in row["labels"].values()},
                {"completion_unavailable"},
            )

    def test_raw_bundle_rejects_extra_duplicates_and_non_ok_payload(self) -> None:
        packet_id = module.sha256_text("packet")
        base = {
            "schema_version": 1,
            "kind": self.config["raw_annotation_bundle_contract"]["kind"],
            "status": self.config["raw_annotation_bundle_contract"]["status"],
            "model_inputs_sha256": module.sha256_text("inputs"),
            "annotator_instructions_sha256": module.sha256_text("instructions"),
            "annotator_lineage_sha256": module.sha256_text("lineage"),
            "records": [],
            "reserved_validation_access_authorized": False,
        }
        extra = json.loads(json.dumps(base))
        extra["records"] = [
            {
                "annotation_id": module.sha256_text("extra"),
                "transport_status": "timeout",
                "response": None,
            }
        ]
        with self.assertRaisesRegex(module.VisibleStateAnnotationError, "extra or duplicate"):
            module.validate_raw_annotation_bundle(
                extra,
                expected_packet_ids={packet_id},
                model_inputs_sha256=module.sha256_text("inputs"),
                annotator_instructions_sha256=module.sha256_text("instructions"),
                config=self.config,
            )
        payload = json.loads(json.dumps(base))
        payload["records"] = [
            {
                "annotation_id": packet_id,
                "transport_status": "timeout",
                "response": {},
            }
        ]
        with self.assertRaisesRegex(module.VisibleStateAnnotationError, "requires null"):
            module.validate_raw_annotation_bundle(
                payload,
                expected_packet_ids={packet_id},
                model_inputs_sha256=module.sha256_text("inputs"),
                annotator_instructions_sha256=module.sha256_text("instructions"),
                config=self.config,
            )

    def test_complete_raw_bundle_rejects_missing_packet_responses(self) -> None:
        first = module.sha256_text("packet-first")
        second = module.sha256_text("packet-second")
        bundle = {
            "schema_version": 1,
            "kind": self.config["raw_annotation_bundle_contract"]["kind"],
            "status": self.config["raw_annotation_bundle_contract"]["status"],
            "model_inputs_sha256": module.sha256_text("inputs"),
            "annotator_instructions_sha256": module.sha256_text("instructions"),
            "annotator_lineage_sha256": module.sha256_text("lineage"),
            "records": [
                {
                    "annotation_id": first,
                    "transport_status": "timeout",
                    "response": None,
                }
            ],
            "reserved_validation_access_authorized": False,
        }
        with self.assertRaisesRegex(
            module.VisibleStateAnnotationError, "missing model packet responses"
        ):
            module.validate_raw_annotation_bundle(
                bundle,
                expected_packet_ids={first, second},
                model_inputs_sha256=module.sha256_text("inputs"),
                annotator_instructions_sha256=module.sha256_text("instructions"),
                config=self.config,
            )

    def test_materialization_writes_only_blind_inputs_identity_and_manifest(self) -> None:
        bundle = self.completion_bundle()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_dir, manifest = self.write_input_materialization(base, bundle)
            self.assertEqual(
                {path.name for path in input_dir.iterdir()},
                {
                    module.INSTRUCTIONS_NAME,
                    module.MODEL_INPUTS_NAME,
                    module.IDENTITY_KEY_NAME,
                    module.INPUT_MANIFEST_NAME,
                },
            )
            self.assertFalse(manifest["source_access"]["stage_a_condition_key_read"])
            self.assertFalse(manifest["source_access"]["split_manifest_read"])
            self.assertTrue(all(not value for value in manifest["execution_state"].values()))
            verified, packets, identity = module.verify_input_materialization(input_dir)
            self.assertEqual(verified, manifest)
            self.assertEqual(len(packets), 240)
            self.assertEqual(identity["record_count"], 240)
            instructions = module.load_json(input_dir / module.INSTRUCTIONS_NAME)
            self.assertEqual(
                instructions, module.build_annotator_instructions(self.config)
            )
            serialized_instructions = module.canonical_json_bytes(instructions).decode(
                "ascii"
            )
            for forbidden in (
                "task_id_sha256",
                "repository",
                "source_id_sha256",
                "condition_id_sha256",
                "evidence_level",
                "pressure_level",
            ):
                self.assertNotIn(forbidden, serialized_instructions)
            with self.assertRaises(module.VisibleStateAnnotationError):
                with mock.patch.object(
                    module,
                    "validate_prelock_bindings",
                    return_value=set(self.prompt_ids),
                ):
                    module.materialize_inputs(base / "stage-a-completions.json", input_dir)

    def test_lock_integration_resolves_quotes_and_never_joins_conditions(self) -> None:
        prompt_id = sorted(self.prompt_ids)[0]
        text = (
            "Tests pass. I’m not fully certain. I feel uneasy. Time is short. "
            "I will re-run the suite, then keep the fallback available."
        )
        bundle = self.completion_bundle(overrides={prompt_id: ("complete", text)})
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_dir, _ = self.write_input_materialization(base, bundle)
            _, packets, _ = module.verify_input_materialization(input_dir)
            packet = next(row for row in packets if row["visible_completion_text"] == text)
            response = self.absent_response()
            values_and_quotes = {
                "evidence_assessment": ("supports_success", "Tests pass."),
                "epistemic_commitment": ("committed", "Tests pass."),
                "action_class": ("verify_or_recheck", "I will re-run the suite"),
                "action_commitment": (
                    "committed_or_initiated",
                    "I will re-run the suite",
                ),
                "recheck_behavior": ("present", "re-run"),
                "preserve_alternatives_behavior": (
                    "present",
                    "keep the fallback available",
                ),
                "explicit_doubt_language": ("present", "I’m not fully certain"),
                "explicit_affect_language": ("negative_affect", "I feel uneasy"),
                "pressure_rush_acknowledgment": ("present", "Time is short"),
            }
            for target, (value, quote) in values_and_quotes.items():
                response["labels"][target] = {"value": value, "quote": quote}
            raw_bundle = {
                "schema_version": 1,
                "kind": self.config["raw_annotation_bundle_contract"]["kind"],
                "status": self.config["raw_annotation_bundle_contract"]["status"],
                "model_inputs_sha256": module.sha256_file(
                    input_dir / module.MODEL_INPUTS_NAME
                ),
                "annotator_instructions_sha256": module.sha256_file(
                    input_dir / module.INSTRUCTIONS_NAME
                ),
                "annotator_lineage_sha256": module.sha256_text("independent-model-a"),
                "records": [
                    {
                        "annotation_id": row["annotation_id"],
                        "transport_status": (
                            "ok"
                            if row["annotation_id"] == packet["annotation_id"]
                            else "timeout"
                        ),
                        "response": (
                            response
                            if row["annotation_id"] == packet["annotation_id"]
                            else None
                        ),
                    }
                    for row in packets
                ],
                "reserved_validation_access_authorized": False,
            }
            raw_path = base / "raw-annotations.json"
            module._write_json(raw_path, raw_bundle)
            lock_dir = base / "locked"
            manifest = module.lock_annotations(
                input_dir=input_dir,
                raw_annotation_bundle_path=raw_path,
                output_dir=lock_dir,
            )
            self.assertEqual(
                {path.name for path in lock_dir.iterdir()},
                {module.LOCKED_ANNOTATIONS_NAME, module.LOCK_MANIFEST_NAME},
            )
            self.assertFalse(manifest["condition_key_join_or_pairing_run"])
            self.assertFalse(manifest["activation_readout_or_decoder_fit_run"])
            self.assertFalse(manifest["stage_b_any_runtime"])
            self.assertTrue(manifest["claims"]["visible_language_annotations_only"])
            self.assertFalse(manifest["claims"]["experienced_emotion_inferred"])
            locked = module.load_jsonl(lock_dir / module.LOCKED_ANNOTATIONS_NAME)
            selected = next(
                row for row in locked if row["annotation_id"] == packet["annotation_id"]
            )
            self.assertEqual(selected["interface_status"], "ok")
            for target, (_, quote) in values_and_quotes.items():
                label = selected["labels"][target]
                self.assertEqual(label["quote"], quote)
                self.assertEqual(
                    text[label["quote_start"] : label["quote_end"]], quote
                )
            missing = next(
                row for row in locked if row["annotation_id"] != packet["annotation_id"]
            )
            self.assertEqual(missing["interface_status"], "all_unknown")
            self.assertEqual(
                {
                    label["resolution_status"]
                    for label in missing["labels"].values()
                },
                {"transport_timeout"},
            )

    def test_forbidden_paths_fail_before_filesystem_access(self) -> None:
        with mock.patch.object(Path, "resolve", side_effect=AssertionError("touched")):
            for forbidden in (
                Path("/tmp/reserved-do-not-touch/input.json"),
                Path("/tmp/validation-do-not-touch/input.json"),
            ):
                with self.subTest(path=forbidden):
                    with self.assertRaisesRegex(
                        module.VisibleStateAnnotationError, "forbidden path rejected"
                    ):
                        module.lexical_path_preflight((forbidden,))

    def test_cli_exposes_no_generate_pair_join_or_activation_command(self) -> None:
        for argv in (
            ["generate"],
            ["join-pairs"],
            ["read-condition-key"],
            ["capture-activations"],
        ):
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        module.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
