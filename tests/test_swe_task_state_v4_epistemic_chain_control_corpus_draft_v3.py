from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
INPUT_MODULE_PATH = (
    ROOT
    / "scripts"
    / "swe_task_state_v4_epistemic_chain_control_inputs_draft_v3.py"
)
KEY_MODULE_PATH = (
    ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_control_key_draft_v3.py"
)
INPUT_CONFIG_PATH = (
    ROOT
    / "configs"
    / "swe_task_state_v4_epistemic_chain_control_inputs_draft_v3.json"
)
KEY_CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_control_key_draft_v3.json"
)
V2_CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_controls_v2.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


inputs = load_module("control_inputs_draft_v3_for_test", INPUT_MODULE_PATH)
key = load_module("control_key_draft_v3_for_test", KEY_MODULE_PATH)


def recursive_keys(value):
    if isinstance(value, dict):
        for name, item in value.items():
            yield str(name)
            yield from recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from recursive_keys(item)


def rehash_projection(projection):
    projection["messages_sha256"] = inputs.sha256_bytes(
        inputs.canonical_json_bytes(projection["messages"])
    )
    projection["response_schema_sha256"] = inputs.sha256_bytes(
        inputs.canonical_json_bytes(projection["response_schema"])
    )
    projection["request_projection_sha256"] = inputs.sha256_bytes(
        inputs.canonical_json_bytes(
            {
                "messages": projection["messages"],
                "response_schema": projection["response_schema"],
            }
        )
    )


def coherently_rehash_input_manifest(manifest):
    """Simulate an attacker who updates every generation-side self-hash."""

    records = manifest["completion_records"] + manifest["novelty_records"]
    for record in manifest["completion_records"]:
        for projection in record["model_input_projections"].values():
            rehash_projection(projection)
    for record in manifest["novelty_records"]:
        rehash_projection(record["model_input_projection"])
    for record in records:
        record["packet_sha256"] = inputs.sha256_bytes(
            inputs.canonical_json_bytes(record["packet"])
        )
        unsigned = dict(record)
        unsigned.pop("record_sha256", None)
        record["record_sha256"] = inputs.sha256_bytes(
            inputs.canonical_json_bytes(unsigned)
        )
    manifest["catalog_manifest"] = inputs.catalog.build_catalog_manifest(
        packets=[row["packet"] for row in manifest["completion_records"]],
        catalogs=[row["catalog"] for row in manifest["completion_records"]],
    )
    manifest["ordered_packet_ids_sha256"] = inputs.sha256_bytes(
        inputs.canonical_json_bytes(
            [row["packet"]["packet_id_sha256"] for row in records]
        )
    )
    manifest["ordered_record_sha256s_sha256"] = inputs.sha256_bytes(
        inputs.canonical_json_bytes([row["record_sha256"] for row in records])
    )
    return inputs.sha256_bytes(inputs.canonical_json_bytes(manifest))


class V3ControlCorpusDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.input_config, cls.input_config_sha = inputs.load_input_config()
        cls.input_envelope = inputs.build_input_manifest_draft()
        cls.input_manifest = cls.input_envelope["manifest"]
        cls.external_input_hash = inputs.sha256_bytes(
            inputs.canonical_json_bytes(cls.input_manifest)
        )
        cls.key_envelope = key.materialize_control_key_draft(
            input_manifest=cls.input_manifest,
            independently_supplied_manifest_sha256=cls.external_input_hash,
        )
        cls.key_manifest = cls.key_envelope["manifest"]

    def test_draft_only_scope_and_physical_one_way_separation(self) -> None:
        self.assertEqual(
            self.input_config["status"], "prospective_draft_not_sealed_not_run"
        )
        self.assertEqual(
            self.key_manifest["status"], "prospective_draft_not_sealed_not_run"
        )
        for scope in (self.input_config["scope"], self.key_manifest["scope"]):
            self.assertTrue(scope["reserved_validation_closed"])
            self.assertFalse(scope["reserved_validation_accessed"])
            self.assertTrue(
                scope["visible_semantic_evidence_hypothesis_action_only"]
            )
            self.assertFalse(scope["cot_recovery_claimed"])
            self.assertFalse(scope["cot_like_recovery_claimed"])
            self.assertFalse(scope["private_cot_recovery_claimed"])
            self.assertFalse(
                scope["affect_emotion_confidence_doubt_or_stress_claimed"]
            )
            self.assertFalse(scope["model_run_performed"])

        source = INPUT_MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn(KEY_MODULE_PATH.name, source)
        self.assertNotIn(KEY_CONFIG_PATH.name, source)
        self.assertNotIn("import swe_task_state_v4_epistemic_chain_control_key", source)
        self.assertNotIn("adjudication", source.casefold())
        generation_forbidden = ("expected", "expectation", "answer", "gold", "category", "diagnostic")
        for field in recursive_keys(self.input_config):
            self.assertFalse(
                any(token in field.casefold() for token in generation_forbidden),
                field,
            )
        for field in recursive_keys(self.input_manifest):
            self.assertFalse(
                any(token in field.casefold() for token in generation_forbidden),
                field,
            )

    def test_bindings_authenticate_exact_draft_sources(self) -> None:
        input_bindings = self.input_config["bindings"]
        self.assertEqual(
            input_bindings["input_builder"]["sha256"],
            inputs.sha256_file(INPUT_MODULE_PATH),
        )
        raw_key = json.loads(KEY_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            raw_key["bindings"]["key_materializer"]["sha256"],
            key.sha256_file(KEY_MODULE_PATH),
        )
        for name in (
            "candidate_catalog_config",
            "candidate_catalog_builder",
            "runner_v3",
            "codebook_v2",
        ):
            self.assertEqual(input_bindings[name], raw_key["bindings"][name])

    def test_fresh_bytes_identity_domains_and_packet_ids_do_not_reuse_v2(self) -> None:
        v2 = json.loads(V2_CONFIG_PATH.read_text(encoding="utf-8"))
        old_completion = {row["assistant_text"] for row in v2["completion_controls"]}
        old_prefixes = {row["visible_prefix"] for row in v2["novelty_controls"]}
        old_hypotheses = {
            row["locked_hypothesis"] for row in v2["novelty_controls"]
        }
        new_completion = {
            row["assistant_text"] for row in self.input_config["completion_inputs"]
        }
        new_prefixes = {
            row["visible_prefix"] for row in self.input_config["novelty_inputs"]
        }
        new_hypotheses = {
            row["locked_hypothesis"] for row in self.input_config["novelty_inputs"]
        }
        self.assertFalse((new_completion - {""}) & (old_completion - {""}))
        self.assertFalse(new_prefixes & old_prefixes)
        self.assertFalse(new_hypotheses & old_hypotheses)

        v2_bytes = V2_CONFIG_PATH.read_bytes()
        for name, value in self.input_config["identity"].items():
            if isinstance(value, str):
                self.assertNotIn(value.encode("utf-8"), v2_bytes, name)
        old_packet_ids = {
            hashlib.sha256(
                f"quote-first-sealed-controls-v2\0completion\0C{index:02d}\00".encode(
                    "utf-8"
                )
            ).hexdigest()
            for index in range(1, 33)
        } | {
            hashlib.sha256(
                f"quote-first-sealed-controls-v2\0novelty\0V{index:02d}\00".encode(
                    "utf-8"
                )
            ).hexdigest()
            for index in range(1, 9)
        }
        new_packet_ids = {
            row["packet"]["packet_id_sha256"]
            for row in self.input_manifest["completion_records"]
            + self.input_manifest["novelty_records"]
        }
        self.assertEqual(len(new_packet_ids), 40)
        self.assertFalse(new_packet_ids & old_packet_ids)

    def test_input_manifest_catalogs_requests_and_empty_host_bypass(self) -> None:
        self.assertEqual(self.input_manifest["counts"], {"completion": 32, "novelty": 8, "total": 40})
        completions = {
            row["control_id"]: row
            for row in self.input_manifest["completion_records"]
        }
        for control_id, row in completions.items():
            self.assertEqual(row["catalog"]["catalog_status"], "available")
            self.assertTrue(row["catalog"]["catalog_usable"])
            self.assertEqual(
                row["catalog_sha256"],
                inputs.catalog.catalog_result_sha256(row["catalog"]),
            )
            if control_id == "C32":
                self.assertEqual(row["catalog"]["unit_count"], 0)
                self.assertEqual(row["model_input_projections"], {})
                self.assertEqual(row["host_action"], "bypass_exact_empty_visible_text")
            else:
                self.assertIn("decision", row["model_input_projections"])
                self.assertEqual(row["host_action"], "invoke_completion_decision")
        self.assertEqual(
            len(self.input_manifest["novelty_records"]), 8
        )
        self.assertTrue(
            all("model_input_projection" in row for row in self.input_manifest["novelty_records"])
        )

    def test_semicolon_fence_earliest_and_repeated_occurrence_catalogs(self) -> None:
        completions = {
            row["control_id"]: row
            for row in self.input_manifest["completion_records"]
        }
        units = lambda cid: [
            item["text"]
            for item in completions[cid]["catalog"]["candidate_unit_bundle"]["units"]
        ]
        self.assertIn(";", units("C02")[1])
        self.assertEqual(len(units("C02")), 3)
        self.assertEqual(len(units("C10")), 3)
        self.assertTrue(units("C10")[0].startswith("The diagnostic emitted:\n```"))
        self.assertIn("dtype=μ8", units("C10")[0])
        self.assertEqual(len(units("C16")), 6)
        self.assertEqual(units("C17")[0], units("C17")[3])
        self.assertEqual(units("C18")[0], units("C18")[1])

    def test_external_hash_is_required_before_key_config_is_opened(self) -> None:
        signature = inspect.signature(key.materialize_control_key_draft)
        self.assertEqual(
            tuple(signature.parameters),
            ("input_manifest", "independently_supplied_manifest_sha256"),
        )
        self.assertTrue(
            all(
                item.kind is inspect.Parameter.KEYWORD_ONLY
                for item in signature.parameters.values()
            )
        )
        called = False

        def trap():
            nonlocal called
            called = True
            raise AssertionError("key config opened before external auth")

        wrong = "0" * 64
        if wrong == self.external_input_hash:
            wrong = "1" * 64
        with mock.patch.object(key, "_load_key_config_after_external_auth", trap):
            with self.assertRaises(key.ControlKeyDraftError):
                key.materialize_control_key_draft(
                    input_manifest=self.input_manifest,
                    independently_supplied_manifest_sha256=wrong,
                )
        self.assertFalse(called)

        tampered = copy.deepcopy(self.input_manifest)
        tampered["completion_records"][0]["packet"]["materialized_assistant_text"][
            "text"
        ] += " altered"
        with self.assertRaises(key.ControlKeyDraftError):
            key.materialize_control_key_draft(
                input_manifest=tampered,
                independently_supplied_manifest_sha256=self.external_input_hash,
            )

    def test_key_side_validation_is_generation_independent_and_exactly_bound(self) -> None:
        source = KEY_MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn(
            "import swe_task_state_v4_epistemic_chain_control_inputs_draft_v3",
            source,
        )
        self.assertEqual(
            self.input_manifest["implementation_bindings"],
            key.EXPECTED_INPUT_IMPLEMENTATION_BINDINGS,
        )
        self.assertEqual(
            self.input_manifest["implementation_bindings"]["input_builder"][
                "sha256"
            ],
            inputs.sha256_file(INPUT_MODULE_PATH),
        )
        with mock.patch.object(
            inputs,
            "build_input_manifest_draft",
            side_effect=AssertionError("generation module must not be consulted"),
        ):
            rebuilt = key.materialize_control_key_draft(
                input_manifest=self.input_manifest,
                independently_supplied_manifest_sha256=self.external_input_hash,
            )
        self.assertEqual(rebuilt, self.key_envelope)

    def test_recursive_forbidden_fields_reject_coherently_rehashed_both_lanes(self) -> None:
        forbidden = (
            "gold",
            "expected",
            "category",
            "diagnostic",
            "answer",
            "key",
            "expectation",
        )
        for lane, records_name in (
            ("completion", "completion_records"),
            ("novelty", "novelty_records"),
        ):
            for field in forbidden:
                with self.subTest(lane=lane, field=field):
                    attack = copy.deepcopy(self.input_manifest)
                    attack[records_name][0][field] = {"attacker_value": True}
                    external_hash = coherently_rehash_input_manifest(attack)
                    with self.assertRaises(key.ControlKeyDraftError):
                        key.materialize_control_key_draft(
                            input_manifest=attack,
                            independently_supplied_manifest_sha256=external_hash,
                        )

        nested_attacks = []
        attack = copy.deepcopy(self.input_manifest)
        attack["completion_records"][0]["packet"]["authenticated_boundaries"][
            "expected_category"
        ] = "chain"
        nested_attacks.append(("completion_packet_expected", attack))
        attack = copy.deepcopy(self.input_manifest)
        attack["completion_records"][0]["model_input_projections"]["decision"][
            "messages"
        ][0]["answer_key"] = "chain"
        nested_attacks.append(("completion_message_answer_key", attack))
        attack = copy.deepcopy(self.input_manifest)
        attack["novelty_records"][0]["packet"]["locked_hypothesis"][
            "diagnostic"
        ] = "novel"
        nested_attacks.append(("novelty_packet_diagnostic", attack))
        attack = copy.deepcopy(self.input_manifest)
        attack["novelty_records"][0]["model_input_projection"][
            "response_schema"
        ]["gold_category"] = "novel"
        nested_attacks.append(("novelty_schema_gold_category", attack))
        for label, attack in nested_attacks:
            with self.subTest(label=label):
                external_hash = coherently_rehash_input_manifest(attack)
                with self.assertRaises(key.ControlKeyDraftError):
                    key.materialize_control_key_draft(
                        input_manifest=attack,
                        independently_supplied_manifest_sha256=external_hash,
                    )

    def test_exact_nested_shapes_and_production_request_rebuild_fail_closed(self) -> None:
        attacks = []

        attack = copy.deepcopy(self.input_manifest)
        attack["completion_records"][0]["metadata"] = "coherently hashed"
        attacks.append(("completion_record_shape", attack))

        attack = copy.deepcopy(self.input_manifest)
        attack["completion_records"][0]["packet"]["authenticated_boundaries"][
            "metadata"
        ] = "coherently hashed"
        attacks.append(("completion_packet_nested_shape", attack))

        attack = copy.deepcopy(self.input_manifest)
        attack["completion_records"][0]["model_input_projections"]["decision"][
            "metadata"
        ] = "coherently hashed"
        attacks.append(("completion_projection_shape", attack))

        attack = copy.deepcopy(self.input_manifest)
        attack["completion_records"][0]["model_input_projections"]["decision"][
            "messages"
        ][1]["content"] += " "
        attacks.append(("completion_message_rewrite", attack))

        attack = copy.deepcopy(self.input_manifest)
        attack["completion_records"][0]["model_input_projections"]["decision"][
            "response_schema"
        ]["properties"]["decision"]["enum"].append("other")
        attacks.append(("completion_schema_rewrite", attack))

        attack = copy.deepcopy(self.input_manifest)
        attack["novelty_records"][0]["metadata"] = "coherently hashed"
        attacks.append(("novelty_record_shape", attack))

        attack = copy.deepcopy(self.input_manifest)
        attack["novelty_records"][0]["packet"]["locked_hypothesis"][
            "metadata"
        ] = "coherently hashed"
        attacks.append(("novelty_packet_nested_shape", attack))

        attack = copy.deepcopy(self.input_manifest)
        attack["novelty_records"][0]["model_input_projection"][
            "metadata"
        ] = "coherently hashed"
        attacks.append(("novelty_projection_shape", attack))

        attack = copy.deepcopy(self.input_manifest)
        attack["novelty_records"][0]["model_input_projection"]["messages"][1][
            "content"
        ] += " "
        attacks.append(("novelty_message_rewrite", attack))

        attack = copy.deepcopy(self.input_manifest)
        attack["novelty_records"][0]["model_input_projection"][
            "response_schema"
        ]["properties"]["decision"]["enum"].append("other")
        attacks.append(("novelty_schema_rewrite", attack))

        for label, attack in attacks:
            with self.subTest(label=label):
                external_hash = coherently_rehash_input_manifest(attack)
                with self.assertRaises(key.ControlKeyDraftError):
                    key.materialize_control_key_draft(
                        input_manifest=attack,
                        independently_supplied_manifest_sha256=external_hash,
                    )

        attack = copy.deepcopy(self.input_manifest)
        attack["completion_records"] = tuple(attack["completion_records"])
        external_hash = inputs.sha256_bytes(inputs.canonical_json_bytes(attack))
        with self.assertRaises(key.ControlKeyDraftError):
            key.materialize_control_key_draft(
                input_manifest=attack,
                independently_supplied_manifest_sha256=external_hash,
            )

        attack = copy.deepcopy(self.input_manifest)
        attack["novelty_records"][0]["packet"]["authenticated_prefix"][
            "source_char_start"
        ] = False
        external_hash = coherently_rehash_input_manifest(attack)
        with self.assertRaises(key.ControlKeyDraftError):
            key.materialize_control_key_draft(
                input_manifest=attack,
                independently_supplied_manifest_sha256=external_hash,
            )

    def test_nested_implementation_binding_tamper_and_cotamper_rejected(self) -> None:
        attacks = []

        attack = copy.deepcopy(self.input_manifest)
        attack["implementation_bindings"]["runner_v3"]["sha256"] = "0" * 64
        attacks.append(("nested_hash_tamper", attack))

        attack = copy.deepcopy(self.input_manifest)
        replacement = attack["implementation_bindings"][
            "candidate_catalog_builder"
        ]
        attack["implementation_bindings"]["runner_v3"] = copy.deepcopy(
            replacement
        )
        attacks.append(("path_and_hash_cotamper", attack))

        attack = copy.deepcopy(self.input_manifest)
        attack["implementation_bindings"]["runner_v3"]["metadata"] = True
        attacks.append(("nested_binding_shape", attack))

        attack = copy.deepcopy(self.input_manifest)
        attack["implementation_bindings"]["runner_v3"] = {
            "path": "scripts/answer_key.py",
            "sha256": "f" * 64,
        }
        attacks.append(("forbidden_binding_path", attack))

        for label, attack in attacks:
            with self.subTest(label=label):
                external_hash = coherently_rehash_input_manifest(attack)
                with self.assertRaises(key.ControlKeyDraftError):
                    key.materialize_control_key_draft(
                        input_manifest=attack,
                        independently_supplied_manifest_sha256=external_hash,
                    )

    def test_external_catalog_manifest_authentication_rebuilds_exact_inputs(self) -> None:
        completions = self.input_manifest["completion_records"]
        envelope = self.input_manifest["catalog_manifest"]
        authenticated = inputs.catalog.authenticate_catalog_manifest(
            value=envelope,
            expected_manifest_sha256=envelope["manifest_sha256"],
            packets=[row["packet"] for row in completions],
            catalogs=[row["catalog"] for row in completions],
        )
        self.assertEqual(authenticated, envelope["manifest"])
        wrong = "f" * 64
        if wrong == envelope["manifest_sha256"]:
            wrong = "e" * 64
        with self.assertRaises(inputs.catalog.CandidateCatalogError):
            inputs.catalog.authenticate_catalog_manifest(
                value=envelope,
                expected_manifest_sha256=wrong,
                packets=[row["packet"] for row in completions],
                catalogs=[row["catalog"] for row in completions],
            )

    def test_completion_gold_schedule_exact_ids_ontology_and_markers(self) -> None:
        rows = self.key_manifest["completion_rows"]
        categories = [row["gold_category"] for row in rows]
        self.assertEqual(categories[:18], ["chain"] * 18)
        self.assertEqual(categories[18:30], ["no_chain"] * 12)
        self.assertEqual(categories[30:], ["unknown", "no_chain"])
        tuples = [row["gold_exact_unit_id_tuple"] for row in rows if row["gold_category"] == "chain"]
        self.assertEqual(len(tuples), 18)
        self.assertTrue(
            all(
                set(item)
                == {"evidence_unit_id", "hypothesis_unit_id", "action_unit_id"}
                and len(set(item.values())) == 3
                and all(inputs.runner.UNIT_ID_RE.fullmatch(value) for value in item.values())
                for item in tuples
            )
        )
        codebook = inputs._load_codebook(self.input_config)
        ontology = codebook["ontology"]
        for row in rows[:18]:
            annotation = row["gold_materialized_result"]["annotation_record"]
            self.assertIn(annotation["evidence_kind"], ontology["evidence_kind"])
            self.assertIn(annotation["belief_edge"], ontology["belief_edge"])
            self.assertIn(annotation["hypothesis_domain"], ontology["hypothesis_domain"])
            self.assertIn(annotation["action_intent"], ontology["action_intent"])
            materialized = inputs.runner.derive_marker_observations(
                evidence_text=annotation["evidence_span"]["text"],
                hypothesis_text=annotation["hypothesis_span"]["text"],
                action_text=annotation["action_span"]["text"],
            )
            self.assertEqual(
                materialized,
                {
                    "relation_marker_present": annotation["relation_marker_present"],
                    "action_marker_present": annotation["action_marker_present"],
                },
            )

    def test_earliest_and_first_occurrence_ids_are_selected_without_ambiguity_metric(self) -> None:
        inputs_by_id = {
            row["control_id"]: row
            for row in self.input_manifest["completion_records"]
        }
        keys_by_id = {
            row["control_id"]: row for row in self.key_manifest["completion_rows"]
        }
        for control_id, expected_ordinals in {
            "C16": (0, 1, 2),
            "C17": (0, 1, 2),
            "C18": (0, 2, 3),
        }.items():
            units = inputs_by_id[control_id]["catalog"]["candidate_unit_bundle"]["units"]
            tuple_value = keys_by_id[control_id]["gold_exact_unit_id_tuple"]
            self.assertEqual(
                tuple(tuple_value.values()),
                tuple(units[index]["unit_id"] for index in expected_ordinals),
            )
        c18 = keys_by_id["C18"]
        self.assertNotIn("ambiguity", json.dumps(c18, sort_keys=True).casefold())
        self.assertEqual(c18["gold_materialized_result"]["materialization_status"], "resolved_authenticated_unit_chain")

    def test_c29_only_generic_action_defect_and_c32_exact_empty_bypass(self) -> None:
        inputs_by_id = {
            row["control_id"]: row
            for row in self.input_manifest["completion_records"]
        }
        keys_by_id = {
            row["control_id"]: row for row in self.key_manifest["completion_rows"]
        }
        c29_units = [
            item["text"]
            for item in inputs_by_id["C29"]["catalog"]["candidate_unit_bundle"]["units"]
        ]
        self.assertEqual(len(c29_units), 3)
        self.assertRegex(c29_units[0], r"18% more memory")
        self.assertRegex(c29_units[1], r"confirms that allocation pressure increased")
        self.assertEqual(c29_units[2], "Therefore I will keep going.")
        self.assertEqual(keys_by_id["C29"]["gold_category"], "no_chain")
        self.assertEqual(
            keys_by_id["C29"]["diagnostic"],
            "generic_non_specific_action_only_defect",
        )
        self.assertEqual(
            inputs_by_id["C32"]["packet"]["materialized_assistant_text"]["text"],
            "",
        )
        self.assertFalse(keys_by_id["C32"]["model_invocation_required"])
        self.assertTrue(
            all(
                row["model_invocation_required"]
                for control_id, row in keys_by_id.items()
                if control_id != "C32"
            )
        )
        self.assertEqual(keys_by_id["C32"]["gold_category"], "no_chain")

    def test_novelty_schedule_is_two_four_two_and_deterministic(self) -> None:
        rows = self.key_manifest["novelty_rows"]
        self.assertEqual(
            [row["gold_category"] for row in rows],
            [
                "novel",
                "novel",
                "prefix_exposed",
                "prefix_exposed",
                "prefix_exposed",
                "prefix_exposed",
                "ambiguous",
                "ambiguous",
            ],
        )
        rebuilt = inputs.build_input_manifest_draft()
        self.assertEqual(rebuilt, self.input_envelope)
        rebuilt_key = key.materialize_control_key_draft(
            input_manifest=rebuilt["manifest"],
            independently_supplied_manifest_sha256=rebuilt["manifest_sha256"],
        )
        self.assertEqual(rebuilt_key, self.key_envelope)


if __name__ == "__main__":
    unittest.main()
