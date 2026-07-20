from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "scripts"
    / "swe_task_state_v4_epistemic_chain_adjudication_fixture_v3.py"
)
CONFIG_PATH = (
    ROOT
    / "configs"
    / "swe_task_state_v4_epistemic_chain_adjudication_fixture_v3.json"
)
CODEBOOK_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_codebook_v2.json"
)
spec = importlib.util.spec_from_file_location(
    "swe_task_state_v4_epistemic_chain_adjudication_fixture_v3", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
production = module.production


def completion_packet(text: str, *, salt: str = "main") -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": production.v2.legacy.COMPLETION_PACKET_KIND,
        "annotation_pass": "completion_chain",
        "packet_id_sha256": production.sha256_text(
            "fixture-completion-packet\0" + salt + "\0" + text
        ),
        "source_id_sha256": production.sha256_text(
            "fixture-completion-source\0" + salt + "\0" + text
        ),
        "blind_shards": {"independent_a": 2, "independent_b": 5},
        "materialized_assistant_text": {
            "char_start": 0,
            "char_end": len(text),
            "sha256": production.sha256_text(text),
            "text": text,
        },
        "authenticated_boundaries": {
            "not_model_visible": "FIXTURE_BOUNDARY_CANARY"
        },
        "annotator_visibility": {
            "assistant_tool_arguments_present": False,
            "complete_prefix_text_present": False,
            "model_features_present": False,
            "repository_or_task_identity_present": False,
            "tool_results_present": False,
        },
    }


def novelty_packet(prefix: str, hypothesis: str) -> dict[str, object]:
    source = production.sha256_text(
        "fixture-novelty-source\0" + prefix + "\0" + hypothesis
    )
    return {
        "schema_version": 1,
        "kind": production.v2.legacy.PREFIX_PACKET_KIND,
        "annotation_pass": "prefix_novelty",
        "packet_id_sha256": production.sha256_text(
            "fixture-novelty-packet\0" + source
        ),
        "source_id_sha256": source,
        "blind_shards": {"independent_a": 1, "independent_b": 6},
        "locked_hypothesis": {
            "text": hypothesis,
            "sha256": production.sha256_text(hypothesis),
            "completion_char_start": 4,
            "completion_char_end": 4 + len(hypothesis),
            "materialized_completion_sha256": "c" * 64,
        },
        "authenticated_prefix": {
            "source_sha256": production.sha256_text(prefix),
            "source_char_start": 0,
            "source_char_end": len(prefix),
            "annotator_text": prefix,
            "annotator_text_sha256": production.sha256_text(prefix),
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


def spans(text: str, parts: list[str]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = 0
    for part in parts:
        start = text.index(part, cursor)
        end = start + len(part)
        result.append((start, end))
        cursor = end
    return result


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        digest = production.sha256_bytes(production.canonical_json_bytes(messages))
        return [1, int(digest[:8], 16), len(messages), 2]


def generation_context(*, revision_character: str = "7"):
    model_identity = {
        "base_model_lineage": "fixture-cpu-fake",
        "repo_id": production.v2.MISTRAL_LOCAL_REPO_ID,
        "revision": revision_character * 40,
        "snapshot_tree_sha256": "8" * 64,
        "quantization": "none",
        "dtype": "float32",
    }
    tokenizer_identity = {
        "repo_id": production.v2.MISTRAL_LOCAL_REPO_ID,
        "revision": revision_character * 40,
        "snapshot_tree_sha256": "8" * 64,
        "tokenizer_mode": "mistral",
        "tokenizer_class": "FixtureFakeTokenizer",
        "vocab_identity_sha256": "9" * 64,
    }
    tokenizer = FakeTokenizer()
    return production.authenticate_native_generation_context(
        tokenizer=tokenizer,
        model_identity=model_identity,
        expected_model_identity_sha256=production.sha256_bytes(
            production.canonical_json_bytes(model_identity)
        ),
        tokenizer_identity=tokenizer_identity,
        expected_tokenizer_identity_sha256=production.sha256_bytes(
            production.canonical_json_bytes(tokenizer_identity)
        ),
        chat_template_kwargs={},
    )


def fixture_generation_contract(
    context,
    *,
    nonce: str,
    verdict_seed: int,
    repair_seed: int,
):
    contract = module.build_fixture_generation_contract(
        generation_context=context,
        runtime_identity={
            "runtime_kind": "cpu_mock_native_protocol",
            "runtime_package_lock_sha256": production.sha256_text(
                "fixture-runtime-package-lock"
            ),
            "runtime_build_sha256": production.sha256_text(
                "fixture-runtime-build"
            ),
        },
        native_adapter_identity={
            "adapter_kind": "fixture-scripted-native-adapter",
            "adapter_source_sha256": production.sha256_text(
                "fixture-native-adapter-source"
            ),
            "adapter_config_sha256": production.sha256_text(
                "fixture-native-adapter-config"
            ),
        },
        verdict_seed=verdict_seed,
        repair_seed=repair_seed,
        fixture_nonce_sha256=nonce,
        outer_nonce_precommit_receipt_sha256=production.sha256_text(
            "outer-precommit-receipt-reference\0" + nonce
        ),
    )
    return contract, production.sha256_bytes(
        production.canonical_json_bytes(contract)
    )


def native_result(request, text: str):
    prompt_ids = list(request.body["submitted_prompt_token_ids"])
    return production.build_native_generation_result(
        request=request,
        text=text,
        submitted_prompt_token_ids=prompt_ids,
        engine_prompt_token_ids=prompt_ids,
        output_token_ids=[101, len(text), int(production.sha256_text(text)[:8], 16)],
        finish_reason="stop",
    )


class ScriptedNative:
    def __init__(self, responses):
        self.responses = dict(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        response = self.responses[request.body["stage"]]
        if callable(response):
            response = response(request)
        return native_result(request, response)


class AdjudicationFixtureV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.codebook = json.loads(CODEBOOK_PATH.read_text(encoding="utf-8"))
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.config_sha256 = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
        cls.nonce = production.sha256_text("fixture-only-nonce-A")
        cls.text = (
            "The focused test fails. "
            "The serializer truncates e\u0301. "
            "I will inspect length 🚀."
        )
        cls.packet = completion_packet(cls.text)
        cls.parts = [
            "The focused test fails.",
            "The serializer truncates e\u0301.",
            "I will inspect length 🚀.",
        ]
        cls.bundle = production.build_candidate_unit_bundle(
            packet=cls.packet, spans=spans(cls.text, cls.parts)
        )
        cls.bundle_sha256 = production.candidate_unit_bundle_sha256(cls.bundle)
        cls.units = production.authenticate_candidate_unit_bundle(
            value=cls.bundle,
            packet=cls.packet,
            expected_bundle_sha256=cls.bundle_sha256,
        )
        cls.chain = {
            "decision": "chain",
            "evidence_unit_id": cls.units.units[0].unit_id,
            "hypothesis_unit_id": cls.units.units[1].unit_id,
            "action_unit_id": cls.units.units[2].unit_id,
            "evidence_kind": "tool_or_test",
            "belief_edge": "supports",
            "hypothesis_domain": "source_logic",
            "action_intent": "inspect",
        }
        cls.completion_candidates = tuple(
            module.build_authored_fixture_candidate(
                packet=cls.packet,
                codebook=cls.codebook,
                annotation_pass="completion_chain",
                projection=projection,
                candidate_unit_bundle=cls.bundle,
                expected_candidate_unit_bundle_sha256=cls.bundle_sha256,
            )
            for projection in (cls.chain, {"decision": "no_chain"})
        )
        cls.novelty_packet = novelty_packet(
            "The prefix discusses a parser failure.",
            "The failure comes from stale decoder state.",
        )
        cls.novelty_candidates = tuple(
            module.build_authored_fixture_candidate(
                packet=cls.novelty_packet,
                codebook=cls.codebook,
                annotation_pass="prefix_novelty",
                projection={"decision": decision},
            )
            for decision in ("novel", "prefix_exposed")
        )

    def completion_lock(
        self,
        *,
        candidates=None,
        nonce=None,
        context=None,
        generation_contract=None,
        generation_contract_sha256=None,
        verdict_seed=11,
        repair_seed=17,
    ):
        candidates = candidates or self.completion_candidates
        nonce = nonce or self.nonce
        context = context or generation_context()
        if generation_contract is None:
            generation_contract, generation_contract_sha256 = (
                fixture_generation_contract(
                    context,
                    nonce=nonce,
                    verdict_seed=verdict_seed,
                    repair_seed=repair_seed,
                )
            )
        lock = module.build_fixture_adjudication_lock(
            packet=self.packet,
            codebook=self.codebook,
            annotation_pass="completion_chain",
            fixture_candidates=candidates,
            fixture_nonce_sha256=nonce,
            fixture_config=self.config,
            expected_fixture_config_sha256=self.config_sha256,
            generation_contract=generation_contract,
            expected_generation_contract_sha256=(
                generation_contract_sha256
            ),
            generation_context=context,
            expected_verdict_seed=verdict_seed,
            expected_repair_seed=repair_seed,
            candidate_unit_bundle=self.bundle,
            expected_candidate_unit_bundle_sha256=self.bundle_sha256,
        )
        return lock, module.fixture_adjudication_lock_sha256(lock)

    def novelty_lock(
        self,
        *,
        candidates=None,
        nonce=None,
        context=None,
        generation_contract=None,
        generation_contract_sha256=None,
        verdict_seed=23,
        repair_seed=29,
    ):
        candidates = candidates or self.novelty_candidates
        nonce = nonce or self.nonce
        context = context or generation_context()
        if generation_contract is None:
            generation_contract, generation_contract_sha256 = (
                fixture_generation_contract(
                    context,
                    nonce=nonce,
                    verdict_seed=verdict_seed,
                    repair_seed=repair_seed,
                )
            )
        lock = module.build_fixture_adjudication_lock(
            packet=self.novelty_packet,
            codebook=self.codebook,
            annotation_pass="prefix_novelty",
            fixture_candidates=candidates,
            fixture_nonce_sha256=nonce,
            fixture_config=self.config,
            expected_fixture_config_sha256=self.config_sha256,
            generation_contract=generation_contract,
            expected_generation_contract_sha256=(
                generation_contract_sha256
            ),
            generation_context=context,
            expected_verdict_seed=verdict_seed,
            expected_repair_seed=repair_seed,
        )
        return lock, module.fixture_adjudication_lock_sha256(lock)

    def completion_run_kwargs(self, native, context=None):
        context = context or generation_context()
        contract, contract_sha = fixture_generation_contract(
            context,
            nonce=self.nonce,
            verdict_seed=11,
            repair_seed=17,
        )
        lock, lock_sha = self.completion_lock(
            context=context,
            generation_contract=contract,
            generation_contract_sha256=contract_sha,
        )
        return {
            "packet": self.packet,
            "codebook": self.codebook,
            "annotation_pass": "completion_chain",
            "fixture_candidates": self.completion_candidates,
            "fixture_nonce_sha256": self.nonce,
            "fixture_config": self.config,
            "expected_fixture_config_sha256": self.config_sha256,
            "generation_contract": contract,
            "expected_generation_contract_sha256": contract_sha,
            "generation_context": context,
            "expected_verdict_seed": 11,
            "expected_repair_seed": 17,
            "fixture_lock": lock,
            "expected_fixture_lock_sha256": lock_sha,
            "generate_native": native,
            "candidate_unit_bundle": self.bundle,
            "expected_candidate_unit_bundle_sha256": self.bundle_sha256,
        }

    def novelty_run_kwargs(self, native, context=None):
        context = context or generation_context()
        contract, contract_sha = fixture_generation_contract(
            context,
            nonce=self.nonce,
            verdict_seed=23,
            repair_seed=29,
        )
        lock, lock_sha = self.novelty_lock(
            context=context,
            generation_contract=contract,
            generation_contract_sha256=contract_sha,
        )
        return {
            "packet": self.novelty_packet,
            "codebook": self.codebook,
            "annotation_pass": "prefix_novelty",
            "fixture_candidates": self.novelty_candidates,
            "fixture_nonce_sha256": self.nonce,
            "fixture_config": self.config,
            "expected_fixture_config_sha256": self.config_sha256,
            "generation_contract": contract,
            "expected_generation_contract_sha256": contract_sha,
            "generation_context": context,
            "expected_verdict_seed": 23,
            "expected_repair_seed": 29,
            "fixture_lock": lock,
            "expected_fixture_lock_sha256": lock_sha,
            "generate_native": native,
        }

    def validate_completion_record(self, record, kwargs):
        return module.validate_fixture_adjudication_record(
            record=record,
            packet=kwargs["packet"],
            codebook=kwargs["codebook"],
            annotation_pass=kwargs["annotation_pass"],
            fixture_candidates=kwargs["fixture_candidates"],
            fixture_nonce_sha256=kwargs["fixture_nonce_sha256"],
            fixture_config=kwargs["fixture_config"],
            expected_fixture_config_sha256=kwargs[
                "expected_fixture_config_sha256"
            ],
            generation_contract=kwargs["generation_contract"],
            expected_generation_contract_sha256=kwargs[
                "expected_generation_contract_sha256"
            ],
            generation_context=kwargs["generation_context"],
            expected_verdict_seed=kwargs["expected_verdict_seed"],
            expected_repair_seed=kwargs["expected_repair_seed"],
            fixture_lock=kwargs["fixture_lock"],
            expected_fixture_lock_sha256=kwargs[
                "expected_fixture_lock_sha256"
            ],
            candidate_unit_bundle=kwargs["candidate_unit_bundle"],
            expected_candidate_unit_bundle_sha256=kwargs[
                "expected_candidate_unit_bundle_sha256"
            ],
        )

    def test_config_candidate_provenance_and_claim_scope_are_explicit(self):
        authenticated = module.authenticate_fixture_config(
            value=self.config,
            expected_config_sha256=self.config_sha256,
        )
        self.assertEqual(
            authenticated.production_runner_sha256,
            hashlib.sha256(Path(production.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            authenticated.codebook_source_sha256,
            hashlib.sha256(CODEBOOK_PATH.read_bytes()).hexdigest(),
        )
        for candidate in self.completion_candidates + self.novelty_candidates:
            self.assertEqual(
                candidate["control_fixture_provenance"], module.FIXTURE_PROVENANCE
            )
            self.assertFalse(candidate["control_fixture_provenance"]["model_generated"])
            self.assertFalse(
                candidate["control_fixture_provenance"][
                    "native_model_record_lineage_present"
                ]
            )
            self.assertNotIn("generation", candidate)
            self.assertNotEqual(candidate["kind"], production.RECORD_KIND)
        self.assertTrue(module.CLAIM_SCOPE["visible_semantic_cot_like_structure_targeted"])
        self.assertFalse(
            module.CLAIM_SCOPE["private_chain_of_thought_recovery_established"]
        )
        self.assertFalse(
            module.CLAIM_SCOPE["affect_or_emotion_recovery_established"]
        )

    def test_generation_contract_and_lock_bind_context_seeds_tokens_and_scope(self):
        context = generation_context()
        contract, contract_sha = fixture_generation_contract(
            context,
            nonce=self.nonce,
            verdict_seed=11,
            repair_seed=17,
        )
        authenticated = module.authenticate_fixture_generation_contract(
            value=contract,
            expected_generation_contract_sha256=contract_sha,
            generation_context=context,
            expected_verdict_seed=11,
            expected_repair_seed=17,
            expected_fixture_nonce_sha256=self.nonce,
        )
        lock, _ = self.completion_lock(
            context=context,
            generation_contract=contract,
            generation_contract_sha256=contract_sha,
        )
        self.assertEqual(lock["generation_contract"], contract)
        self.assertEqual(lock["generation_contract_sha256"], contract_sha)
        self.assertEqual(
            lock["runtime_identity_sha256"],
            authenticated.runtime_identity_sha256,
        )
        self.assertEqual(
            lock["native_adapter_identity_sha256"],
            authenticated.native_adapter_identity_sha256,
        )
        self.assertEqual(lock["inner_execution_mode"], module.INNER_EXECUTION_MODE)
        self.assertFalse(lock["inner_route_claims_actual_model_execution"])
        self.assertFalse(lock["precommit_chronology_verified_by_fixture_route"])
        self.assertFalse(
            lock["expectation_access_chronology_verified_by_fixture_route"]
        )
        self.assertEqual(
            lock["chronology_responsibility"],
            module.OUTER_EXECUTOR_RESPONSIBILITY,
        )
        self.assertEqual(
            lock["nonce_precommit_sha256"],
            module.nonce_precommit_sha256(
                fixture_nonce_sha256=self.nonce
            ),
        )
        for binding in lock["model_input_bindings"].values():
            if binding is None:
                continue
            self.assertTrue(binding["submitted_prompt_token_ids"])
            self.assertEqual(
                binding["submitted_prompt_token_ids_sha256"],
                production.sha256_bytes(
                    production.canonical_json_bytes(
                        binding["submitted_prompt_token_ids"]
                    )
                ),
            )
            self.assertEqual(
                binding["submitted_prompt_token_count"],
                len(binding["submitted_prompt_token_ids"]),
            )

    def test_inner_route_cannot_claim_actual_model_execution(self):
        parameters = inspect.signature(module.run_fixture_adjudication).parameters
        self.assertNotIn("execution_mode", parameters)
        context = generation_context()
        contract, _ = fixture_generation_contract(
            context,
            nonce=self.nonce,
            verdict_seed=11,
            repair_seed=17,
        )
        forged = copy.deepcopy(contract)
        forged["inner_route_claims_actual_model_execution"] = True
        forged_sha = production.sha256_bytes(
            production.canonical_json_bytes(forged)
        )
        with self.assertRaisesRegex(
            module.FixtureRouteError, "identity or claim scope"
        ):
            module.authenticate_fixture_generation_contract(
                value=forged,
                expected_generation_contract_sha256=forged_sha,
                generation_context=context,
                expected_verdict_seed=11,
                expected_repair_seed=17,
                expected_fixture_nonce_sha256=self.nonce,
            )

    def test_nonce_blinding_is_deterministic_symmetric_and_binds_original_order(self):
        first, first_sha = self.completion_lock()
        repeat, repeat_sha = self.completion_lock()
        self.assertEqual(first, repeat)
        self.assertEqual(first_sha, repeat_sha)
        swapped, _ = self.completion_lock(
            candidates=(self.completion_candidates[1], self.completion_candidates[0])
        )
        self.assertEqual(
            first["blinded_projection_sha256s"],
            swapped["blinded_projection_sha256s"],
        )
        self.assertNotEqual(
            first["authored_projection_sha256s_in_original_order"],
            swapped["authored_projection_sha256s_in_original_order"],
        )
        self.assertNotEqual(
            first["original_order_binding_sha256"],
            swapped["original_order_binding_sha256"],
        )
        alternate, _ = self.completion_lock(
            nonce=production.sha256_text("fixture-only-nonce-B")
        )
        self.assertNotEqual(
            first["fixture_nonce_sha256"], alternate["fixture_nonce_sha256"]
        )
        self.assertNotEqual(first, alternate)

    def test_model_input_is_exact_production_shape_unlabeled_and_nonce_free(self):
        context = generation_context()
        contract, contract_sha = fixture_generation_contract(
            context,
            nonce=self.nonce,
            verdict_seed=11,
            repair_seed=17,
        )
        prepared = module._prepare_fixture(
            packet=self.packet,
            codebook=self.codebook,
            annotation_pass="completion_chain",
            fixture_candidates=self.completion_candidates,
            fixture_nonce_sha256=self.nonce,
            fixture_config=self.config,
            expected_fixture_config_sha256=self.config_sha256,
            generation_contract=contract,
            expected_generation_contract_sha256=contract_sha,
            generation_context=context,
            expected_verdict_seed=11,
            expected_repair_seed=17,
            candidate_unit_bundle=self.bundle,
            expected_candidate_unit_bundle_sha256=self.bundle_sha256,
        )
        direct = production.build_adjudication_messages(
            packet=prepared.packet,
            codebook=prepared.codebook,
            annotation_pass="completion_chain",
            blinded_candidates=prepared.blinded_candidates,
            authenticated_units=prepared.authenticated_units,
        )
        self.assertEqual(list(prepared.verdict_messages), direct)
        payload = module.assert_unlabeled_model_input(
            messages=direct,
            forbidden_field_fragments=prepared.config.forbidden_field_fragments,
        )
        self.assertEqual(
            set(payload),
            {"assistant_text", "candidate_units", "candidate_annotations"},
        )
        for item in payload["candidate_annotations"]:
            self.assertEqual(set(item), {"candidate_id", "annotation"})
        serialized = production.canonical_json_text(direct)
        self.assertNotIn(self.nonce, serialized)
        self.assertNotIn("authored_bounded_control_fixture_projection", serialized)
        self.assertNotIn("expectation", direct[1]["content"].casefold())

    def test_direct_completion_verdict_uses_native_parity_and_materializes(self):
        context = generation_context()
        native = ScriptedNative(
            {"control_fixture_adjudication_verdict": '{"verdict":"candidate_2"}'}
        )
        kwargs = self.completion_run_kwargs(native, context=context)
        record = module.run_fixture_adjudication(**kwargs)
        self.assertEqual(record["adjudication_verdict"], "candidate_2")
        self.assertFalse(record["repair_invoked"])
        self.assertIsNone(record["repair_generation"])
        self.assertEqual(
            record["raw_semantic_proposal"],
            self.completion_candidates[
                record["blinded_original_indexes"][1]
            ]["projection"],
        )
        self.assertFalse(record["fixture_provenance"]["candidate_model_generated"])
        self.assertFalse(record["fixture_provenance"]["control_gate_eligible"])
        request = native.requests[0]
        binding = record["fixture_bindings"]["model_input_bindings"][
            "adjudication_verdict"
        ]
        self.assertEqual(request.body["messages_sha256"], binding["messages_sha256"])
        self.assertEqual(
            request.body["response_schema_sha256"],
            production.sha256_bytes(
                production.canonical_json_bytes(
                    production.adjudication_response_schema()
                )
            ),
        )
        self.assertTrue(request.body["per_packet_schema"])
        self.assertFalse(request.body["free_string_response_fields"])
        self.assertFalse(request.body["string_round_trip_used"])
        self.assertEqual(
            request.body["submitted_prompt_token_ids"],
            record["verdict_generation"]["result"]["engine_prompt_token_ids"],
        )
        self.assertEqual(
            request.body["submitted_prompt_token_ids"],
            binding["submitted_prompt_token_ids"],
        )
        self.validate_completion_record(record, kwargs)

    def test_neither_completion_decision_repair_is_bounded(self):
        native = ScriptedNative(
            {
                "control_fixture_adjudication_verdict": '{"verdict":"neither"}',
                "control_fixture_completion_repair_decision": (
                    '{"decision":"no_chain"}'
                ),
            }
        )
        kwargs = self.completion_run_kwargs(native)
        record = module.run_fixture_adjudication(**kwargs)
        self.assertTrue(record["repair_invoked"])
        self.assertEqual(record["raw_semantic_proposal"], {"decision": "no_chain"})
        self.assertFalse(record["annotation_record"]["has_chain"])
        self.assertIsNone(record["repair_generation"]["chain_detail"])
        self.assertEqual(len(native.requests), 2)
        repair_schema = native.requests[1].body["response_schema"]
        self.assertEqual(set(repair_schema["properties"]), {"decision"})
        self.validate_completion_record(record, kwargs)

    def test_neither_completion_chain_detail_uses_ids_and_materializes(self):
        detail = {
            key: value for key, value in self.chain.items() if key != "decision"
        }
        native = ScriptedNative(
            {
                "control_fixture_adjudication_verdict": '{"verdict":"neither"}',
                "control_fixture_completion_repair_decision": (
                    '{"decision":"chain"}'
                ),
                "control_fixture_completion_repair_chain_detail": (
                    production.canonical_json_text(detail)
                ),
            }
        )
        kwargs = self.completion_run_kwargs(native)
        record = module.run_fixture_adjudication(**kwargs)
        self.assertTrue(record["annotation_record"]["has_chain"])
        self.assertEqual(record["raw_semantic_proposal"], self.chain)
        self.assertIsNotNone(record["repair_generation"]["chain_detail"])
        self.assertEqual(len(native.requests), 3)
        detail_request = native.requests[2]
        self.assertEqual(
            set(detail_request.body["response_schema"]["properties"]),
            set(production.CHAIN_DETAIL_FIELDS),
        )
        self.assertEqual(
            detail_request.body["lineage_bindings"][
                "parent_repair_decision_result_sha256"
            ],
            record["repair_generation"]["decision"]["result_sha256"],
        )
        self.validate_completion_record(record, kwargs)

    def test_novelty_supports_direct_and_neither_repair_routes(self):
        with self.subTest("direct"):
            native = ScriptedNative(
                {
                    "control_fixture_adjudication_verdict": (
                        '{"verdict":"candidate_1"}'
                    )
                }
            )
            kwargs = self.novelty_run_kwargs(native)
            record = module.run_fixture_adjudication(**kwargs)
            self.assertIn(
                record["annotation_record"]["novelty_status"],
                {"novel", "prefix_exposed"},
            )
            self.assertNotIn("candidate_unit_bundle_sha256", record)
            self.assertIsNone(
                record["fixture_bindings"]["candidate_unit_bundle_sha256"]
            )
        with self.subTest("neither"):
            native = ScriptedNative(
                {
                    "control_fixture_adjudication_verdict": (
                        '{"verdict":"neither"}'
                    ),
                    "control_fixture_novelty_repair_decision": (
                        '{"decision":"ambiguous"}'
                    ),
                }
            )
            kwargs = self.novelty_run_kwargs(native)
            record = module.run_fixture_adjudication(**kwargs)
            self.assertEqual(record["raw_semantic_proposal"], {"decision": "ambiguous"})
            self.assertEqual(record["annotation_record"]["novelty_status"], "ambiguous")
            self.assertEqual(len(native.requests), 2)
            module.validate_fixture_adjudication_record(
                record=record,
                packet=kwargs["packet"],
                codebook=kwargs["codebook"],
                annotation_pass=kwargs["annotation_pass"],
                fixture_candidates=kwargs["fixture_candidates"],
                fixture_nonce_sha256=kwargs["fixture_nonce_sha256"],
                fixture_config=kwargs["fixture_config"],
                expected_fixture_config_sha256=kwargs[
                    "expected_fixture_config_sha256"
                ],
                generation_contract=kwargs["generation_contract"],
                expected_generation_contract_sha256=kwargs[
                    "expected_generation_contract_sha256"
                ],
                generation_context=kwargs["generation_context"],
                expected_verdict_seed=kwargs["expected_verdict_seed"],
                expected_repair_seed=kwargs["expected_repair_seed"],
                fixture_lock=kwargs["fixture_lock"],
                expected_fixture_lock_sha256=kwargs[
                    "expected_fixture_lock_sha256"
                ],
            )

    def test_expectation_objects_are_denied_by_generation_api(self):
        parameters = inspect.signature(module.run_fixture_adjudication).parameters
        self.assertFalse(
            {"expectation", "expectations", "expected", "gold"} & set(parameters)
        )
        self.assertFalse(module.generation_api_accepts_expectation_object())
        native = ScriptedNative(
            {"control_fixture_adjudication_verdict": '{"verdict":"candidate_1"}'}
        )
        kwargs = self.completion_run_kwargs(native)
        separate_expectation = {
            "expected_verdict_class": "candidate_1",
            "fixture_lock_sha256": kwargs["expected_fixture_lock_sha256"],
        }
        with self.assertRaises(TypeError):
            module.run_fixture_adjudication(
                **kwargs, expectation=separate_expectation
            )
        self.assertEqual(native.requests, [])

    def test_forbidden_label_field_names_fail_closed(self):
        base = [
            {"role": "system", "content": "fixture"},
            {
                "role": "user",
                "content": production.canonical_json_text(
                    {"candidate_annotations": []}
                ),
            },
        ]
        fragments = self.config["model_input"][
            "forbidden_field_name_fragments"
        ]
        for field in (
            "gold",
            "is_wrong",
            "expectedVerdict",
            "winner_id",
            "position-class",
            "positionClass",
        ):
            with self.subTest(field=field):
                malicious = copy.deepcopy(base)
                malicious[1]["content"] = production.canonical_json_text(
                    {"candidate_annotations": [], field: "candidate_1"}
                )
                with self.assertRaisesRegex(
                    module.FixtureRouteError, "forbidden labeled"
                ):
                    module.assert_unlabeled_model_input(
                        messages=malicious,
                        forbidden_field_fragments=fragments,
                    )

    def test_candidate_extra_fields_hash_tamper_and_cross_packet_ids_fail(self):
        with self.subTest("extra gold field"):
            bad = copy.deepcopy(self.completion_candidates[0])
            bad["gold"] = True
            with self.assertRaisesRegex(module.FixtureRouteError, "fields invalid"):
                self.completion_lock(
                    candidates=(bad, self.completion_candidates[1])
                )
        with self.subTest("projection hash"):
            bad = copy.deepcopy(self.completion_candidates[0])
            bad["projection_sha256"] = "0" * 64
            with self.assertRaisesRegex(module.FixtureRouteError, "projection hash"):
                self.completion_lock(
                    candidates=(bad, self.completion_candidates[1])
                )
        with self.subTest("cross packet candidate"):
            other_text = (
                "Another test fails. Another hypothesis follows. I inspect it."
            )
            other_packet = completion_packet(other_text, salt="other")
            other_parts = [
                "Another test fails.",
                "Another hypothesis follows.",
                "I inspect it.",
            ]
            other_bundle = production.build_candidate_unit_bundle(
                packet=other_packet, spans=spans(other_text, other_parts)
            )
            other_sha = production.candidate_unit_bundle_sha256(other_bundle)
            other_units = production.authenticate_candidate_unit_bundle(
                value=other_bundle,
                packet=other_packet,
                expected_bundle_sha256=other_sha,
            )
            other_projection = copy.deepcopy(self.chain)
            other_projection["evidence_unit_id"] = other_units.units[0].unit_id
            other_projection["hypothesis_unit_id"] = other_units.units[1].unit_id
            other_projection["action_unit_id"] = other_units.units[2].unit_id
            cross = copy.deepcopy(self.completion_candidates[0])
            cross["projection"] = other_projection
            cross["projection_sha256"] = production.sha256_bytes(
                production.canonical_json_bytes(other_projection)
            )
            with self.assertRaisesRegex(
                production.BoundedIdRunnerError, "unknown candidate unit ID"
            ):
                self.completion_lock(
                    candidates=(cross, self.completion_candidates[1])
                )

    def test_bundle_config_lock_and_order_tamper_fail_before_generation(self):
        native = ScriptedNative(
            {"control_fixture_adjudication_verdict": '{"verdict":"candidate_1"}'}
        )
        with self.subTest("bundle"):
            kwargs = self.completion_run_kwargs(native)
            bad_bundle = copy.deepcopy(self.bundle)
            bad_bundle["units"][0]["text"] += "x"
            kwargs["candidate_unit_bundle"] = bad_bundle
            with self.assertRaisesRegex(
                production.BoundedIdRunnerError, "out-of-band authenticated hash"
            ):
                module.run_fixture_adjudication(**kwargs)
        with self.subTest("config object"):
            kwargs = self.completion_run_kwargs(native)
            bad_config = copy.deepcopy(self.config)
            bad_config["model_input"][
                "expectation_object_accepted_by_generation_api"
            ] = True
            kwargs["fixture_config"] = bad_config
            with self.assertRaisesRegex(
                module.FixtureRouteError, "source file"
            ):
                module.run_fixture_adjudication(**kwargs)
        for field in (
            "blinded_original_indexes",
            "candidate_order_sha256",
            "model_input_bindings",
        ):
            with self.subTest(lock_field=field):
                kwargs = self.completion_run_kwargs(native)
                tampered = copy.deepcopy(kwargs["fixture_lock"])
                if field == "blinded_original_indexes":
                    tampered[field] = list(reversed(tampered[field]))
                elif field == "candidate_order_sha256":
                    tampered[field] = "0" * 64
                else:
                    tampered[field]["adjudication_verdict"][
                        "messages_sha256"
                    ] = "0" * 64
                kwargs["fixture_lock"] = tampered
                kwargs["expected_fixture_lock_sha256"] = (
                    module.fixture_adjudication_lock_sha256(tampered)
                )
                with self.assertRaisesRegex(
                    module.FixtureRouteError, "content or ordering"
                ):
                    module.run_fixture_adjudication(**kwargs)
        self.assertEqual(native.requests, [])

    def test_response_schema_smuggling_via_callback_is_detected(self):
        class MutatingNative:
            def __init__(self):
                self.called = False

            def __call__(self, request):
                self.called = True
                result = native_result(request, '{"verdict":"candidate_1"}')
                request.body["response_schema"]["properties"]["gold"] = {
                    "type": "string",
                    "enum": ["yes"],
                }
                return result

        native = MutatingNative()
        kwargs = self.completion_run_kwargs(native)
        with self.assertRaisesRegex(
            production.BoundedIdRunnerError,
            "property name invalid|response-schema hash invalid",
        ):
            module.run_fixture_adjudication(**kwargs)
        self.assertTrue(native.called)

    def test_coherent_prompt_id_request_and_result_cotamper_is_detected(self):
        class PromptIdCotamperingNative:
            def __init__(self):
                self.called = False

            def __call__(self, request):
                self.called = True
                changed = list(request.body["submitted_prompt_token_ids"]) + [31337]
                request.body["submitted_prompt_token_ids"] = changed
                request.body["submitted_prompt_token_ids_sha256"] = (
                    production.sha256_bytes(
                        production.canonical_json_bytes(changed)
                    )
                )
                object.__setattr__(
                    request,
                    "request_sha256",
                    production.sha256_bytes(
                        production.canonical_json_bytes(request.body)
                    ),
                )
                text = '{"verdict":"candidate_1"}'
                return production.build_native_generation_result(
                    request=request,
                    text=text,
                    submitted_prompt_token_ids=changed,
                    engine_prompt_token_ids=changed,
                    output_token_ids=[101, 102],
                    finish_reason="stop",
                )

        native = PromptIdCotamperingNative()
        kwargs = self.completion_run_kwargs(native)
        with self.assertRaisesRegex(
            module.FixtureRouteError,
            "executed native request differs from exact reconstructed request",
        ):
            module.run_fixture_adjudication(**kwargs)
        self.assertTrue(native.called)

    def test_seed_context_and_nonce_tamper_fail_before_generation(self):
        native = ScriptedNative(
            {"control_fixture_adjudication_verdict": '{"verdict":"candidate_1"}'}
        )
        with self.subTest("external expected seed"):
            kwargs = self.completion_run_kwargs(native)
            kwargs["expected_verdict_seed"] = 12
            with self.assertRaisesRegex(module.FixtureRouteError, "external expected seeds"):
                module.run_fixture_adjudication(**kwargs)
        with self.subTest("authenticated context"):
            kwargs = self.completion_run_kwargs(native)
            kwargs["generation_context"] = generation_context(
                revision_character="6"
            )
            with self.assertRaisesRegex(
                module.FixtureRouteError, "context differs"
            ):
                module.run_fixture_adjudication(**kwargs)
        with self.subTest("nonce and precommit"):
            kwargs = self.completion_run_kwargs(native)
            kwargs["fixture_nonce_sha256"] = production.sha256_text(
                "tampered-fixture-nonce"
            )
            with self.assertRaisesRegex(
                module.FixtureRouteError, "nonce precommit reference"
            ):
                module.run_fixture_adjudication(**kwargs)
        self.assertEqual(native.requests, [])

    def test_record_provenance_order_and_native_lineage_tamper_fail_closed(self):
        native = ScriptedNative(
            {"control_fixture_adjudication_verdict": '{"verdict":"candidate_1"}'}
        )
        kwargs = self.completion_run_kwargs(native)
        record = module.run_fixture_adjudication(**kwargs)
        with self.subTest("candidate provenance"):
            tampered = copy.deepcopy(record)
            tampered["fixture_provenance"]["candidate_model_generated"] = True
            with self.assertRaisesRegex(module.FixtureRouteError, "provenance"):
                self.validate_completion_record(tampered, kwargs)
        with self.subTest("fake adjudicator execution claim"):
            tampered = copy.deepcopy(record)
            tampered["fixture_provenance"][
                "adjudicator_model_execution_claimed"
            ] = True
            with self.assertRaisesRegex(module.FixtureRouteError, "provenance"):
                self.validate_completion_record(tampered, kwargs)
        with self.subTest("blinded order"):
            tampered = copy.deepcopy(record)
            tampered["blinded_original_indexes"] = list(
                reversed(tampered["blinded_original_indexes"])
            )
            with self.assertRaisesRegex(
                module.FixtureRouteError, "candidate, order"
            ):
                self.validate_completion_record(tampered, kwargs)
        with self.subTest("native lineage"):
            tampered = copy.deepcopy(record)
            tampered["verdict_generation"]["request"]["lineage_bindings"][
                "candidate_order_sha256"
            ] = "0" * 64
            with self.assertRaises(
                (module.FixtureRouteError, production.BoundedIdRunnerError)
            ):
                self.validate_completion_record(tampered, kwargs)
        with self.subTest("expectation field"):
            tampered = copy.deepcopy(record)
            tampered["expectation"] = {"winner": "candidate_1"}
            with self.assertRaisesRegex(module.FixtureRouteError, "fields invalid"):
                self.validate_completion_record(tampered, kwargs)

    def test_invalid_model_fields_fail_to_interface_unknown_not_free_text(self):
        native = ScriptedNative(
            {
                "control_fixture_adjudication_verdict": (
                    '{"verdict":"candidate_1","explanation":"gold"}'
                )
            }
        )
        kwargs = self.completion_run_kwargs(native)
        record = module.run_fixture_adjudication(**kwargs)
        self.assertIsNone(record["adjudication_verdict"])
        self.assertEqual(record["semantic_validation_status"], "invalid")
        self.assertEqual(
            record["interface_unknown_reason"],
            "invalid_adjudication_verdict_interface",
        )
        self.assertFalse(record["repair_invoked"])
        self.validate_completion_record(record, kwargs)


if __name__ == "__main__":
    unittest.main()
