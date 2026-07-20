from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = ROOT / "scripts"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_inputs_draft_v3 as inputs
import swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_key_draft_v3 as key


fixture = key.fixture
runner = key.runner
INPUT_CONFIG_PATH = inputs.CONFIG_PATH
KEY_CONFIG_PATH = key.CONFIG_PATH
FIXTURE_CONFIG_PATH = fixture.DEFAULT_CONFIG_PATH
FIXTURE_CONFIG_SHA256 = "558c2d4dc5285a2ed74ec33c502ae8989645d831d7b868604349ef5689b04bd4"
KEY_CONFIG_SHA256 = "d676af22f287e882400d3f356b49b921af3464b6b353477d6a04dedfb71e09cf"
INPUT_MANIFEST_SHA256 = "81338adf399e0835fc5030f79228f019dc3505893cc550a5957a1ac1346aef9e"


class DraftTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        digest = runner.sha256_bytes(runner.canonical_json_bytes(messages))
        return [1, int(digest[:8], 16), len(messages), 2]


def generation_context():
    model_identity = {
        "base_model_lineage": "fixture-corpus-draft-cpu-no-execution",
        "repo_id": runner.v2.MISTRAL_LOCAL_REPO_ID,
        "revision": "7" * 40,
        "snapshot_tree_sha256": "8" * 64,
        "quantization": "none",
        "dtype": "float32",
    }
    tokenizer_identity = {
        "repo_id": runner.v2.MISTRAL_LOCAL_REPO_ID,
        "revision": "7" * 40,
        "snapshot_tree_sha256": "8" * 64,
        "tokenizer_mode": "mistral",
        "tokenizer_class": "FixtureCorpusDraftTokenizer",
        "vocab_identity_sha256": "9" * 64,
    }
    return runner.authenticate_native_generation_context(
        tokenizer=DraftTokenizer(),
        model_identity=model_identity,
        expected_model_identity_sha256=runner.sha256_bytes(
            runner.canonical_json_bytes(model_identity)
        ),
        tokenizer_identity=tokenizer_identity,
        expected_tokenizer_identity_sha256=runner.sha256_bytes(
            runner.canonical_json_bytes(tokenizer_identity)
        ),
        chat_template_kwargs={},
    )


class AdjudicationFixtureCorpusDraftV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.input_envelope = inputs.build_input_manifest_draft()
        cls.input_hash = cls.input_envelope["manifest_sha256"]
        assert cls.input_hash == INPUT_MANIFEST_SHA256
        cls.key_config = json.loads(KEY_CONFIG_PATH.read_text(encoding="utf-8"))
        cls.key_config_sha256 = KEY_CONFIG_SHA256
        assert hashlib.sha256(KEY_CONFIG_PATH.read_bytes()).hexdigest() == KEY_CONFIG_SHA256
        cls.fixture_config = json.loads(
            FIXTURE_CONFIG_PATH.read_text(encoding="utf-8")
        )
        cls.context = generation_context()
        # This one obvious, fixed test value is never retried or searched.  Any
        # observed balance remains non-gating because chronology is unproven.
        cls.suite_nonce = runner.sha256_text(
            "fixed-external-test-suite-nonce-no-retry"
        )
        cls.receipt = runner.sha256_text("fixed-external-test-receipt")
        cls.contracts, cls.contract_hashes, cls.verdict_seeds, cls.repair_seeds = (
            cls._contracts_for_nonce(cls.suite_nonce, cls.receipt)
        )

    @classmethod
    def _contracts_for_nonce(cls, suite_nonce: str, receipt: str):
        contracts = {}
        hashes = {}
        verdict_seeds = {}
        repair_seeds = {}
        domain = cls.key_config["nonce_contract"]["case_nonce_derivation_domain"]
        for index in range(1, 13):
            case_id = f"F{index:02d}"
            verdict_seeds[case_id] = 1000 + index
            repair_seeds[case_id] = 2000 + index
            case_nonce = key.derive_case_nonce_sha256(
                suite_nonce_sha256=suite_nonce,
                case_id=case_id,
                input_manifest_sha256=cls.input_hash,
                domain=domain,
            )
            contract = fixture.build_fixture_generation_contract(
                generation_context=cls.context,
                runtime_identity={
                    "runtime_kind": "cpu_draft_no_execution",
                    "runtime_package_lock_sha256": runner.sha256_text(
                        "fixture-corpus-draft-runtime-lock"
                    ),
                    "runtime_build_sha256": runner.sha256_text(
                        "fixture-corpus-draft-runtime-build"
                    ),
                },
                native_adapter_identity={
                    "adapter_kind": "fixture-corpus-draft-no-callback",
                    "adapter_source_sha256": runner.sha256_text(
                        "fixture-corpus-draft-adapter-source"
                    ),
                    "adapter_config_sha256": runner.sha256_text(
                        "fixture-corpus-draft-adapter-config"
                    ),
                },
                verdict_seed=verdict_seeds[case_id],
                repair_seed=repair_seeds[case_id],
                fixture_nonce_sha256=case_nonce,
                outer_nonce_precommit_receipt_sha256=receipt,
            )
            contracts[case_id] = contract
            hashes[case_id] = runner.sha256_bytes(
                runner.canonical_json_bytes(contract)
            )
        return contracts, hashes, verdict_seeds, repair_seeds

    def materialize_kwargs(self):
        return {
            "input_manifest_envelope": self.input_envelope,
            "expected_input_manifest_sha256": self.input_hash,
            "expected_key_config_sha256": self.key_config_sha256,
            "externally_precommitted_suite_nonce_sha256": self.suite_nonce,
            "outer_nonce_precommit_receipt_sha256": self.receipt,
            "fixture_config": self.fixture_config,
            "expected_fixture_config_sha256": FIXTURE_CONFIG_SHA256,
            "generation_context": self.context,
            "generation_contracts": self.contracts,
            "expected_generation_contract_sha256s": self.contract_hashes,
            "verdict_seeds": self.verdict_seeds,
            "repair_seeds": self.repair_seeds,
        }

    def test_input_manifest_is_exact_answer_blind_and_nonce_placeholder_only(self):
        envelope = self.input_envelope
        manifest = envelope["manifest"]
        self.assertEqual(
            envelope["manifest_sha256"],
            inputs.sha256_bytes(inputs.canonical_json_bytes(manifest)),
        )
        self.assertEqual(manifest["counts"], {"completion": 9, "novelty": 3, "total": 12})
        self.assertEqual(
            [record["case_id"] for record in manifest["records"]],
            [f"F{index:02d}" for index in range(1, 13)],
        )
        inputs.assert_no_answer_fields(manifest)
        key.assert_input_semantically_blind(manifest)
        provenance = manifest["nonce_provenance"]
        self.assertEqual(provenance["status"], "external_precommit_pending")
        self.assertFalse(provenance["actual_fixture_nonce_present"])
        self.assertFalse(provenance["precommit_chronology_claimed"])
        self.assertFalse(provenance["semantic_access_chronology_claimed"])
        for record in manifest["records"]:
            self.assertEqual(len(record["authored_candidates"]), 2)
            self.assertNotEqual(
                record["authored_candidates"][0]["projection_sha256"],
                record["authored_candidates"][1]["projection_sha256"],
            )
            for candidate in record["authored_candidates"]:
                self.assertFalse(
                    candidate["control_fixture_provenance"]["model_generated"]
                )
                self.assertFalse(
                    candidate["control_fixture_provenance"][
                        "native_model_record_lineage_present"
                    ]
                )

    def test_catalogs_cover_unicode_fence_and_occurrence_specific_finite_ids(self):
        records = {
            item["case_id"]: item for item in self.input_envelope["manifest"]["records"]
        }
        self.assertIn("μ8", records["F04"]["packet"]["materialized_assistant_text"]["text"])
        self.assertIn("🚀", records["F04"]["packet"]["materialized_assistant_text"]["text"])
        self.assertIn("```", records["F05"]["packet"]["materialized_assistant_text"]["text"])
        self.assertIn("é", records["F10"]["packet"]["materialized_assistant_text"]["text"])
        for case_id in ("F02", "F10"):
            record = records[case_id]
            self.assertEqual(record["candidate_catalog"]["unit_count"], 4)
            units = record["candidate_catalog"]["candidate_unit_bundle"]["units"]
            self.assertNotEqual(units[0]["unit_id"], units[1]["unit_id"])
            self.assertEqual(units[0]["text"], units[1]["text"])
            for unit in units:
                self.assertRegex(unit["unit_id"], runner.UNIT_ID_RE)
        self.assertEqual(records["F05"]["candidate_catalog"]["unit_count"], 3)

    def test_key_freezes_routes_repairs_and_aggregate_intent_not_case_slots(self):
        validated, observed_hash = key.authenticate_key_config_after_input_auth(
            expected_key_config_sha256=self.key_config_sha256
        )
        self.assertEqual(observed_hash, self.key_config_sha256)
        self.assertEqual(validated["counts"]["direct"], 8)
        self.assertEqual(validated["counts"]["neither"], 4)
        self.assertEqual(validated["counts"]["repair_decision"], 4)
        self.assertEqual(validated["counts"]["completion_chain_detail_repair"], 3)
        self.assertEqual(
            (
                validated["counts"]["intended_candidate_1"],
                validated["counts"]["intended_candidate_2"],
                validated["counts"]["intended_neither"],
            ),
            (4, 4, 4),
        )
        for row in validated["cases"]:
            self.assertNotIn("realized_presentation_class", row)
            self.assertNotIn("candidate_1", row.values())
            self.assertNotIn("candidate_2", row.values())

    def test_fixed_nonretried_nonce_reports_balance_without_execution_or_sealing(self):
        self.assertNotIn("generate_native", inspect.signature(key.materialize_key_draft).parameters)
        result = key.materialize_key_draft(**self.materialize_kwargs())
        manifest = result["manifest"]
        self.assertEqual(
            manifest["status"],
            "prospective_draft_balance_observed_chronology_unverified_not_seal_ready",
        )
        self.assertEqual(
            manifest["presentation_balance"]["observed_counts"],
            {"candidate_1": 4, "candidate_2": 4, "neither": 4},
        )
        self.assertTrue(manifest["presentation_balance"]["balance_satisfied"])
        self.assertFalse(manifest["nonce_retry_permitted"])
        self.assertFalse(manifest["seal_ready"])
        self.assertFalse(manifest["actual_model_execution_claimed"])
        self.assertFalse(manifest["generation_callback_invoked"])
        self.assertFalse(manifest["outer_nonce_precommit_receipt_contents_authenticated"])
        self.assertFalse(
            manifest["single_use_nonce_or_receipt_enforced_by_key_materializer"]
        )
        self.assertFalse(
            manifest["repeated_materialization_prevention_enforced_by_key_materializer"]
        )
        self.assertFalse(
            manifest["authoritative_packet_regeneration_performed_by_key_materializer"]
        )
        self.assertEqual(
            sum(row["route_requirement"] == "direct" for row in manifest["cases"]),
            8,
        )
        self.assertEqual(
            sum(row["repair_decision"] is not None for row in manifest["cases"]),
            4,
        )
        self.assertEqual(
            sum(row["completion_chain_detail_repair_required"] for row in manifest["cases"]),
            3,
        )
        for row in manifest["cases"]:
            lock = row["fixture_lock"]
            self.assertFalse(lock["inner_route_claims_actual_model_execution"])
            self.assertFalse(lock["precommit_chronology_verified_by_fixture_route"])
            self.assertFalse(lock["expectation_access_chronology_verified_by_fixture_route"])
            self.assertEqual(
                lock["outer_nonce_precommit_receipt_sha256"], self.receipt
            )
            frozen_seed = next(
                item
                for item in manifest["generation_seed_schedule"]["cases"]
                if item["case_id"] == row["case_id"]
            )
            self.assertEqual(row["verdict_seed"], frozen_seed["verdict_seed"])
            self.assertEqual(row["repair_seed"], frozen_seed["repair_seed"])
            self.assertEqual(
                lock["generation_contract"]["seeds"],
                {
                    "verdict_seed": row["verdict_seed"],
                    "repair_seed": row["repair_seed"],
                },
            )

    def test_balance_validator_accepts_only_exact_4_4_4_and_never_allows_retry(self):
        balanced = key.validate_presentation_balance(
            ["candidate_1"] * 4 + ["candidate_2"] * 4 + ["neither"] * 4
        )
        self.assertTrue(balanced["balance_satisfied"])
        self.assertFalse(balanced["nonce_retry_permitted"])
        imbalanced = key.validate_presentation_balance(
            ["candidate_1"] * 3 + ["candidate_2"] * 5 + ["neither"] * 4
        )
        self.assertFalse(imbalanced["balance_satisfied"])
        self.assertEqual(
            imbalanced["imbalance_disposition"],
            "invalidate_entire_suite_and_start_new_predeclared_round",
        )

    def test_external_hash_authentication_happens_before_key_config_read(self):
        wrong_hash = "0" * 64
        with mock.patch.object(
            key, "_read_key_config_source", side_effect=AssertionError("key read too early")
        ) as loader:
            with self.assertRaisesRegex(
                key.FixtureCorpusKeyDraftError, "independently supplied"
            ):
                key.materialize_key_draft(
                    **{
                        **self.materialize_kwargs(),
                        "expected_input_manifest_sha256": wrong_hash,
                    }
                )
        loader.assert_not_called()

    def test_recursive_answer_field_leakage_is_rejected_before_key_read(self):
        for field in (
            "presentation-class",
            "route",
            "routingHint",
            "slot-id",
            "winnerPosition",
            "preferred_position",
        ):
            with self.subTest(field=field):
                malicious = copy.deepcopy(self.input_envelope)
                malicious["manifest"]["records"][0]["packet"]["nested"] = {
                    field: "candidate_1"
                }
                malicious_hash = key.sha256_bytes(
                    key.canonical_json_bytes(malicious["manifest"])
                )
                malicious["manifest_sha256"] = malicious_hash
                with mock.patch.object(
                    key,
                    "_read_key_config_source",
                    side_effect=AssertionError("key read too early"),
                ) as loader:
                    with self.assertRaisesRegex(
                        key.FixtureCorpusKeyDraftError, "forbidden semantic field"
                    ):
                        key.materialize_key_draft(
                            **{
                                **self.materialize_kwargs(),
                                "input_manifest_envelope": malicious,
                                "expected_input_manifest_sha256": malicious_hash,
                            }
                        )
                loader.assert_not_called()
        misplaced = copy.deepcopy(self.input_envelope)
        misplaced["manifest"]["completion_chain_slots_present"] = False
        misplaced_hash = key.sha256_bytes(
            key.canonical_json_bytes(misplaced["manifest"])
        )
        misplaced["manifest_sha256"] = misplaced_hash
        with self.assertRaisesRegex(
            key.FixtureCorpusKeyDraftError, "not allowed"
        ):
            key.authenticate_input_manifest_before_key_read(
                value=misplaced,
                expected_input_manifest_sha256=misplaced_hash,
            )

    def test_external_exact_key_hash_precedes_semantics_nonce_and_contracts(self):
        with mock.patch.object(
            key,
            "_validate_authenticated_manifest",
            side_effect=AssertionError("semantic materialization started"),
        ) as semantic:
            with self.assertRaisesRegex(
                key.FixtureCorpusKeyDraftError, "independently supplied exact-byte"
            ):
                key.materialize_key_draft(
                    **{
                        **self.materialize_kwargs(),
                        "expected_key_config_sha256": "0" * 64,
                    }
                )
        semantic.assert_not_called()

    def test_coherent_key_bound_input_winner_route_cotamper_hits_stale_external_key_hash(self):
        attacked_input = copy.deepcopy(self.input_envelope)
        attacked_input["manifest"]["records"][6]["packet"][
            "authenticated_prefix"
        ]["annotator_text"] += " benign-cotamper"
        attacked_input_hash = key.sha256_bytes(
            key.canonical_json_bytes(attacked_input["manifest"])
        )
        attacked_input["manifest_sha256"] = attacked_input_hash
        attacked_key = copy.deepcopy(self.key_config)
        attacked_key["external_input_authentication"][
            "bound_input_manifest_sha256"
        ] = attacked_input_hash
        attacked_key["cases"][0]["route_requirement"] = "neither"
        attacked_key["cases"][0]["winner"] = "candidate_2"
        attacked_raw = json.dumps(
            attacked_key, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.assertNotEqual(hashlib.sha256(attacked_raw).hexdigest(), self.key_config_sha256)
        with mock.patch.object(
            key, "_read_key_config_source", return_value=(attacked_key, attacked_raw)
        ):
            with self.assertRaisesRegex(
                key.FixtureCorpusKeyDraftError, "independently supplied exact-byte"
            ):
                key.materialize_key_draft(
                    **{
                        **self.materialize_kwargs(),
                        "input_manifest_envelope": attacked_input,
                        "expected_input_manifest_sha256": attacked_input_hash,
                    }
                )

    def test_seed_schedule_is_external_key_bound_and_caller_is_not_authoritative(self):
        caller_seeds = copy.deepcopy(self.verdict_seeds)
        caller_seeds["F01"] += 1
        with self.assertRaisesRegex(
            key.FixtureCorpusKeyDraftError, "frozen schedule"
        ):
            key.materialize_key_draft(
                **{
                    **self.materialize_kwargs(),
                    "verdict_seeds": caller_seeds,
                }
            )
        attacked_key = copy.deepcopy(self.key_config)
        attacked_key["generation_seed_schedule"]["cases"][0]["verdict_seed"] += 9
        attacked_raw = json.dumps(attacked_key, sort_keys=True).encode("utf-8")
        with mock.patch.object(
            key, "_read_key_config_source", return_value=(attacked_key, attacked_raw)
        ):
            with self.assertRaisesRegex(
                key.FixtureCorpusKeyDraftError, "independently supplied exact-byte"
            ):
                key.authenticate_key_config_after_input_auth(
                    expected_key_config_sha256=self.key_config_sha256
                )

    def test_nonce_provenance_shape_rejects_coherent_extra_field(self):
        attacked = copy.deepcopy(self.input_envelope)
        attacked["manifest"]["nonce_provenance"]["extra_protocol_flag"] = False
        attacked_hash = key.sha256_bytes(
            key.canonical_json_bytes(attacked["manifest"])
        )
        attacked["manifest_sha256"] = attacked_hash
        authenticated = key.authenticate_input_manifest_before_key_read(
            value=attacked, expected_input_manifest_sha256=attacked_hash
        )
        config = copy.deepcopy(self.key_config)
        config["external_input_authentication"][
            "bound_input_manifest_sha256"
        ] = attacked_hash
        with self.assertRaisesRegex(
            key.FixtureCorpusKeyDraftError, "nonce placeholder provenance"
        ):
            key._validate_authenticated_manifest(
                manifest=authenticated,
                manifest_sha256=attacked_hash,
                config=config,
            )

    def test_coherent_manifest_rehash_and_record_order_attacks_hit_key_bound_hash(self):
        attacks = []
        content = copy.deepcopy(self.input_envelope)
        content["manifest"]["records"][6]["packet"]["authenticated_prefix"][
            "annotator_text"
        ] += "x"
        attacks.append(content)
        ordering = copy.deepcopy(self.input_envelope)
        ordering["manifest"]["records"][0], ordering["manifest"]["records"][1] = (
            ordering["manifest"]["records"][1],
            ordering["manifest"]["records"][0],
        )
        ordering["manifest"]["ordered_case_ids_sha256"] = key.sha256_bytes(
            key.canonical_json_bytes(
                [row["case_id"] for row in ordering["manifest"]["records"]]
            )
        )
        ordering["manifest"]["ordered_record_sha256s_sha256"] = key.sha256_bytes(
            key.canonical_json_bytes(
                [row["record_sha256"] for row in ordering["manifest"]["records"]]
            )
        )
        attacks.append(ordering)
        binding = copy.deepcopy(self.input_envelope)
        binding["manifest"]["implementation_bindings"]["fixture_protocol"]["sha256"] = "a" * 64
        attacks.append(binding)
        for attack in attacks:
            with self.subTest(kind=len(attacks)):
                attack_hash = key.sha256_bytes(
                    key.canonical_json_bytes(attack["manifest"])
                )
                attack["manifest_sha256"] = attack_hash
                with self.assertRaisesRegex(
                    key.FixtureCorpusKeyDraftError, "key-bound manifest hash"
                ):
                    key.materialize_key_draft(
                        **{
                            **self.materialize_kwargs(),
                            "input_manifest_envelope": attack,
                            "expected_input_manifest_sha256": attack_hash,
                        }
                    )

    def test_nonce_contract_and_fixture_config_attacks_fail_closed(self):
        alternate_nonce = runner.sha256_text("different-precommitted-suite-nonce")
        with self.assertRaisesRegex(
            fixture.FixtureRouteError, "nonce precommit reference"
        ):
            key.materialize_key_draft(
                **{
                    **self.materialize_kwargs(),
                    "externally_precommitted_suite_nonce_sha256": alternate_nonce,
                }
            )
        tampered_config = copy.deepcopy(self.fixture_config)
        tampered_config["scope"] = "tampered"
        with self.assertRaisesRegex(fixture.FixtureRouteError, "source file"):
            key.materialize_key_draft(
                **{
                    **self.materialize_kwargs(),
                    "fixture_config": tampered_config,
                }
            )

    def test_key_module_has_no_input_import_and_does_not_hash_input_builder(self):
        source = Path(key.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any("fixture_corpus_inputs" in name for name in imported))
        original = key.sha256_file
        forbidden_name = Path(inputs.__file__).name

        def guarded(path):
            if Path(path).name == forbidden_name:
                raise AssertionError("semantic side read input builder")
            return original(path)

        with mock.patch.object(key, "sha256_file", side_effect=guarded):
            result = key.materialize_key_draft(**self.materialize_kwargs())
        self.assertFalse(result["manifest"]["seal_ready"])

    def test_source_and_config_hashes_are_bound_and_no_reserved_path_is_referenced(self):
        input_config = json.loads(INPUT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            input_config["bindings"]["input_builder"]["sha256"],
            hashlib.sha256(Path(inputs.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            self.key_config["bindings"]["key_materializer"]["sha256"],
            hashlib.sha256(Path(key.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            self.key_config["external_input_authentication"][
                "bound_input_manifest_sha256"
            ],
            self.input_hash,
        )
        for path in (
            INPUT_CONFIG_PATH,
            KEY_CONFIG_PATH,
            Path(inputs.__file__),
            Path(key.__file__),
        ):
            self.assertNotIn("validation/", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
