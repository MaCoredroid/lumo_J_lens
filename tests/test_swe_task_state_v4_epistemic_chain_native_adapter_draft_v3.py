from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_epistemic_chain_native_adapter_draft_v3.json"
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_native_adapter_draft_v3.py"
V2_CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_epistemic_chain_annotation_runner_v2.json"
V3_RUNNER_PATH = ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_annotation_runner_v3.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


adapter = load_module("native_adapter_draft_v3", MODULE_PATH)
runner = adapter.runner_v3


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        digest = runner.sha256_bytes(runner.canonical_json_bytes(messages))
        return [1, int(digest[:8], 16), len(messages), 2]


class Untouchable:
    def __getattribute__(self, name):
        raise AssertionError(f"unauthorized path touched {name}")

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("unauthorized callback invoked")


class NativeAdapterDraftV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_sha = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
        cls.config = adapter.load_draft_config(
            path=CONFIG_PATH, expected_draft_config_sha256=cls.config_sha
        )
        cls.runtime_identity = {
            "engine": "vllm_offline_structured_outputs",
            "package": "vllm",
            "package_version": "future-reviewed-version",
            "engine_build_sha256": "a" * 64,
            "python_version": "future-reviewed-python",
            "cuda_runtime_version": "future-reviewed-cuda",
            "platform": "future-reviewed-platform",
        }
        cls.runtime_hash = adapter.sha256_value(cls.runtime_identity)
        cls.adapter_identity = {
            "kind": "reviewed_production_native_adapter_v3",
            "source_sha256": "b" * 64,
            "entrypoint": "execute_native_token_request",
            "tokens_prompt_api": "vllm.inputs.TokensPrompt",
            "runner_sha256": cls.config["frozen_historical_inputs"]["v3_bounded_id_runner"]["sha256"],
        }
        cls.adapter_hash = adapter.sha256_value(cls.adapter_identity)

    def make_request(self, role: str):
        role_config = self.config["historical_v2_roles"][role]
        model_identity = {
            field: role_config[field] for field in adapter.MODEL_IDENTITY_FIELDS
        }
        tokenizer_identity = {
            "repo_id": role_config["repo_id"],
            "revision": role_config["revision"],
            "snapshot_tree_sha256": role_config["snapshot_tree_sha256"],
            "tokenizer_mode": role_config["vllm_engine_kwargs"].get("tokenizer_mode", "auto"),
            "tokenizer_class": f"Frozen{role}Tokenizer",
            "vocab_identity_sha256": "c" * 64,
        }
        context = runner.authenticate_native_generation_context(
            tokenizer=FakeTokenizer(),
            model_identity=model_identity,
            expected_model_identity_sha256=runner.sha256_bytes(runner.canonical_json_bytes(model_identity)),
            tokenizer_identity=tokenizer_identity,
            expected_tokenizer_identity_sha256=runner.sha256_bytes(runner.canonical_json_bytes(tokenizer_identity)),
            chat_template_kwargs=role_config["chat_template_kwargs"],
        )
        messages = [
            {"role": "system", "content": "finite enum decision"},
            {"role": "user", "content": "choose the candidate"},
        ]
        schema = runner.adjudication_response_schema()
        values = {
            "seed": 41003,
            "stage": "source_only_contract_test",
            "annotation_pass": "completion_chain",
            "packet_id_sha256": "d" * 64,
            "source_id_sha256": "e" * 64,
            "lineage_bindings": {"candidate_manifest_sha256": "f" * 64},
        }
        request = runner.build_native_generation_request(
            context=context,
            messages=messages,
            schema=schema,
            **values,
        )
        reconstruction = adapter.NativeRequestReconstruction(
            generation_context=context,
            messages=messages,
            schema=schema,
            **values,
        )
        return request, reconstruction

    def plan_kwargs(self, role, request, reconstruction):
        return {
            "config_path": CONFIG_PATH,
            "expected_draft_config_sha256": self.config_sha,
            "role": role,
            "request": request,
            "expected_native_request_sha256": request.request_sha256,
            "request_reconstruction": reconstruction,
            "runtime_identity": self.runtime_identity,
            "expected_runtime_identity_sha256": self.runtime_hash,
            "adapter_identity": self.adapter_identity,
            "expected_adapter_identity_sha256": self.adapter_hash,
        }

    def make_plan(self, role="independent_a"):
        request, reconstruction = self.make_request(role)
        plan = adapter.derive_source_only_invocation_plan(
            **self.plan_kwargs(role, request, reconstruction)
        )
        return request, reconstruction, plan

    def validation_kwargs(self, role, request, reconstruction, plan):
        return {
            **self.plan_kwargs(role, request, reconstruction),
            "expected_plan_sha256": plan.plan_sha256,
        }

    def make_observation(self, request, plan):
        prompt_ids = list(request.body["submitted_prompt_token_ids"])
        output_ids = [101, 202, 303]
        text = '{"verdict":"candidate_1"}'
        if plan.body["output_extraction"]["mode"] == "openai_harmony_final_channel":
            extraction = {
                "mode": "openai_harmony_final_channel",
                "parser_package": "openai-harmony",
                "strict": True,
                "final_channel_count": 1,
                "analysis_or_tool_channels_excluded_from_text": True,
                "final_text_sha256": adapter.sha256_bytes(text.encode("utf-8")),
                "output_token_ids_sha256": adapter.sha256_value(output_ids),
            }
        else:
            extraction = {
                "mode": "direct_structured_text",
                "engine_text_used_directly": True,
                "text_sha256": adapter.sha256_bytes(text.encode("utf-8")),
            }
        return {
            "schema_version": 1,
            "kind": adapter.DRAFT_OBSERVATION_KIND,
            "plan_sha256": plan.plan_sha256,
            "native_request_sha256": request.request_sha256,
            "response_schema_sha256": request.body["response_schema_sha256"],
            "seed": request.body["seed"],
            "sampling_params": copy.deepcopy(plan.body["sampling_params"]),
            "sampling_params_sha256": plan.body["sampling_params_sha256"],
            "structured_outputs_config": copy.deepcopy(plan.body["engine"]["structured_outputs_config"]),
            "submitted_prompt_token_ids": prompt_ids,
            "submitted_prompt_token_ids_sha256": adapter.sha256_value(prompt_ids),
            "engine_prompt_token_ids": list(prompt_ids),
            "engine_prompt_token_ids_sha256": adapter.sha256_value(prompt_ids),
            "prompt_truncated": False,
            "output_token_ids": output_ids,
            "output_token_ids_sha256": adapter.sha256_value(output_ids),
            "finish_reason": "stop",
            "text": text,
            "text_sha256": adapter.sha256_bytes(text.encode("utf-8")),
            "output_extraction": extraction,
            "output_text_token_decode_parity_verified": False,
            "authenticated_tokenizer_decode_reconstruction_sha256": None,
            "actual_model_execution": False,
            "gate_eligible": False,
            "production_receipt": False,
        }

    def observation_kwargs(self, role, request, reconstruction, plan, observation):
        return {
            **self.validation_kwargs(role, request, reconstruction, plan),
            "plan": plan,
            "expected_observation_sha256": adapter.sha256_value(observation),
        }

    def expected_record_hash(self, request, plan, observation):
        body = {
            "schema_version": 1,
            "kind": adapter.SCRIPTED_NON_GATE_RECORD_KIND,
            "execution_mode": "scripted_cpu_non_gate",
            "plan_sha256": plan.plan_sha256,
            "native_request_sha256": request.request_sha256,
            "observation": copy.deepcopy(observation),
            "observation_sha256": adapter.sha256_value(observation),
            "claims": {
                "actual_model_execution": False,
                "gate_eligible": False,
                "production_receipt": False,
                "sealed_control_evidence": False,
                "annotation_interface_readiness_established": False,
                "output_text_token_decode_parity_established": False,
            },
        }
        return adapter.sha256_value(body)

    def test_config_authenticates_bytes_and_full_bound_v2_objects(self):
        v2 = json.loads(V2_CONFIG_PATH.read_text(encoding="utf-8"))
        authenticated = adapter.authenticate_draft_config(
            path=CONFIG_PATH, expected_draft_config_sha256=self.config_sha
        )
        self.assertEqual(authenticated.value["historical_v2_roles"], v2["roles"])
        self.assertEqual(authenticated.value["historical_v2_generation"], v2["generation"])
        self.assertEqual(authenticated.v2_config_sha256, hashlib.sha256(V2_CONFIG_PATH.read_bytes()).hexdigest())
        self.assertEqual(authenticated.v3_runner_sha256, hashlib.sha256(V3_RUNNER_PATH.read_bytes()).hexdigest())

    def test_config_external_hash_and_full_v2_equality_reject_co_tamper(self):
        original = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for section, mutate, pattern in (
            ("role", lambda value: value["historical_v2_roles"]["independent_a"].__setitem__("repo_id", "attacker/recomputed"), "roles differ"),
            ("generation", lambda value: value["historical_v2_generation"].__setitem__("max_num_seqs", 99), "generation differs"),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            with self.subTest(section=section), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "draft.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                changed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(adapter.NativeAdapterDraftError, "external authenticated hash"):
                    adapter.authenticate_draft_config(path=path, expected_draft_config_sha256=self.config_sha)
                with self.assertRaisesRegex(adapter.NativeAdapterDraftError, pattern):
                    adapter.authenticate_draft_config(path=path, expected_draft_config_sha256=changed_hash)

    def test_current_config_is_unauthorized_unfrozen_claims_false_and_parity_unproven(self):
        self.assertTrue(all(value is False for value in self.config["authorization"].values()))
        implementation = self.config["implementation"]
        self.assertFalse(implementation["production_native_adapter_present"])
        self.assertIsNone(implementation["production_native_adapter_sha256"])
        self.assertFalse(implementation["production_runtime_present"])
        self.assertIsNone(implementation["production_runtime_sha256"])
        self.assertFalse(implementation["production_receipt_emitter_present"])
        self.assertTrue(all(value is False for value in self.config["claim_scope"].values()))
        self.assertFalse(self.config["future_exact_token_contract"]["source_only_output_text_token_decode_parity_verified"])
        self.assertTrue(self.config["future_exact_token_contract"]["output_text_matches_authenticated_token_decode_required_for_real"])
        self.assertFalse(adapter.PRODUCTION_CODE_FREEZE_COMPLETE)

    def test_reported_preflight_remains_non_authenticating_and_non_authorizing(self):
        preflight = self.config["observed_read_only_preflight"]
        self.assertEqual(preflight["packages"]["vllm"], "0.23.0")
        self.assertIsNone(preflight["evidence_receipt_sha256"])
        self.assertFalse(preflight["authenticated_runtime_identity"])
        self.assertFalse(preflight["authorizes_execution"])
        self.assertFalse(preflight["used_by_source_only_plan_identity"])

    def test_source_has_no_runtime_backend_tokenizer_cli_or_output_writer(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"^\s*(?:from|import)\s+vllm\b", source, re.MULTILINE))
        self.assertIsNone(re.search(r"^\s*(?:from|import)\s+transformers\b", source, re.MULTILINE))
        self.assertNotIn("AutoTokenizer", source)
        self.assertNotIn("LLM(", source)
        self.assertNotIn("write_text(", source)
        self.assertFalse(hasattr(adapter, "main"))

    def test_production_gate_receipt_stubs_reject_before_any_access_even_if_flags_mutate(self):
        adapter.PRODUCTION_CODE_FREEZE_COMPLETE = True
        try:
            for function in (
                adapter.execute_production_native_adapter,
                adapter.execute_gate_eligible_native_adapter,
                adapter.emit_production_gate_receipt,
            ):
                with self.subTest(function=function.__name__), mock.patch(
                    "builtins.__import__", side_effect=AssertionError("dynamic import")
                ), mock.patch.object(Path, "open", side_effect=AssertionError("path access")):
                    with self.assertRaisesRegex(adapter.NativeAdapterDraftError, "absent|unauthorized"):
                        function(config=Untouchable(), backend=Untouchable(), output=Untouchable(), callback=Untouchable())
        finally:
            adapter.PRODUCTION_CODE_FREEZE_COMPLETE = False

    def test_bad_config_hash_rejects_before_request_context_plan_or_identity_access(self):
        with self.assertRaisesRegex(adapter.NativeAdapterDraftError, "draft config differs"):
            adapter.derive_source_only_invocation_plan(
                config_path=CONFIG_PATH,
                expected_draft_config_sha256="0" * 64,
                role="independent_a",
                request=Untouchable(),
                expected_native_request_sha256="1" * 64,
                request_reconstruction=Untouchable(),
                runtime_identity=Untouchable(),
                expected_runtime_identity_sha256="2" * 64,
                adapter_identity=Untouchable(),
                expected_adapter_identity_sha256="3" * 64,
            )

    def test_plan_uses_exact_reconstructed_runner_request_and_direct_token_ids(self):
        request, reconstruction, plan = self.make_plan()
        body = adapter.validate_source_only_invocation_plan(
            plan, **self.validation_kwargs("independent_a", request, reconstruction, plan)
        )
        self.assertEqual(body["draft_config_sha256"], self.config_sha)
        self.assertTrue(body["native_request_exactly_reconstructed"])
        self.assertEqual(body["token_prompt"]["api"], "vllm.inputs.TokensPrompt")
        self.assertEqual(body["token_prompt"]["prompt_token_ids"], request.body["submitted_prompt_token_ids"])
        self.assertTrue(all(value is False for value in body["input_transformations"].values()))
        self.assertEqual(body["sampling_params"]["structured_outputs"]["json"], request.body["response_schema"])
        self.assertEqual(body["sampling_params"]["seed"], request.body["seed"])
        self.assertIsNone(body["sampling_params"]["truncate_prompt_tokens"])
        self.assertFalse(body["output_contract"]["output_text_token_decode_parity_verified"])
        self.assertTrue(body["output_contract"]["future_real_adapter_requires_authenticated_tokenizer_reconstruction"])

    def test_request_ids_messages_lineage_and_self_hash_co_tamper_fail(self):
        request, reconstruction = self.make_request("independent_a")
        body = copy.deepcopy(request.body)
        body["submitted_prompt_token_ids"].append(999)
        body["submitted_prompt_token_ids_sha256"] = runner.sha256_bytes(
            runner.canonical_json_bytes(body["submitted_prompt_token_ids"])
        )
        body["messages_sha256"] = "1" * 64
        body["lineage_bindings"] = {"candidate_manifest_sha256": "2" * 64}
        forged = runner.NativeGenerationRequest(
            body=body,
            request_sha256=runner.sha256_bytes(runner.canonical_json_bytes(body)),
        )
        kwargs = self.plan_kwargs("independent_a", forged, reconstruction)
        kwargs["expected_native_request_sha256"] = request.request_sha256
        with self.assertRaisesRegex(adapter.NativeAdapterDraftError, "external authenticated hash"):
            adapter.derive_source_only_invocation_plan(**kwargs)
        kwargs["expected_native_request_sha256"] = forged.request_sha256
        with self.assertRaisesRegex(adapter.NativeAdapterDraftError, "independently reconstructed"):
            adapter.derive_source_only_invocation_plan(**kwargs)

    def test_sha_named_identity_fields_and_external_identity_hashes_fail_closed(self):
        request, reconstruction = self.make_request("independent_a")
        changed = copy.deepcopy(self.runtime_identity)
        changed["platform"] = "co-tampered-platform"
        kwargs = self.plan_kwargs("independent_a", request, reconstruction)
        kwargs["runtime_identity"] = changed
        with self.assertRaisesRegex(adapter.NativeAdapterDraftError, "external authenticated hash"):
            adapter.derive_source_only_invocation_plan(**kwargs)
        changed = copy.deepcopy(self.runtime_identity)
        changed["engine_build_sha256"] = "A" * 64
        kwargs["runtime_identity"] = changed
        kwargs["expected_runtime_identity_sha256"] = adapter.sha256_value(changed)
        with self.assertRaisesRegex(adapter.NativeAdapterDraftError, "must be SHA-256"):
            adapter.derive_source_only_invocation_plan(**kwargs)
        bad_adapter = copy.deepcopy(self.adapter_identity)
        bad_adapter["source_sha256"] = "B" * 64
        kwargs = self.plan_kwargs("independent_a", request, reconstruction)
        kwargs["adapter_identity"] = bad_adapter
        kwargs["expected_adapter_identity_sha256"] = adapter.sha256_value(bad_adapter)
        with self.assertRaisesRegex(adapter.NativeAdapterDraftError, "must be SHA-256"):
            adapter.derive_source_only_invocation_plan(**kwargs)

    def test_external_plan_hash_rejects_plan_and_internal_hash_co_tamper(self):
        request, reconstruction, plan = self.make_plan()
        body = copy.deepcopy(plan.body)
        body["sampling_params"]["seed"] += 1
        body["sampling_params_sha256"] = adapter.sha256_value(body["sampling_params"])
        tampered = adapter.DraftInvocationPlan(body=body, plan_sha256=adapter.sha256_value(body))
        with self.assertRaisesRegex(adapter.NativeAdapterDraftError, "external authenticated hash"):
            adapter.validate_source_only_invocation_plan(
                tampered,
                **self.validation_kwargs("independent_a", request, reconstruction, plan),
            )

    def test_valid_observation_and_scripted_cpu_record_are_externally_bound_non_gate(self):
        request, reconstruction, plan = self.make_plan()
        observation = self.make_observation(request, plan)
        obs_kwargs = self.observation_kwargs("independent_a", request, reconstruction, plan, observation)
        validated = adapter.validate_draft_engine_observation(observation, **obs_kwargs)
        self.assertFalse(validated["output_text_token_decode_parity_verified"])
        self.assertIsNone(validated["authenticated_tokenizer_decode_reconstruction_sha256"])
        expected_record = self.expected_record_hash(request, plan, observation)
        record = adapter.execute_scripted_cpu_non_gate(
            **obs_kwargs,
            expected_record_sha256=expected_record,
            scripted_observation=lambda *_args: observation,
        )
        self.assertEqual(record.record_sha256, expected_record)
        self.assertTrue(all(value is False for value in record.body["claims"].values()))
        adapter.validate_scripted_cpu_non_gate_record(
            record,
            **obs_kwargs,
            expected_record_sha256=expected_record,
        )

    def test_prompt_truncation_seed_schema_sampling_finish_and_output_id_tamper_fail(self):
        request, reconstruction, plan = self.make_plan()
        base = self.make_observation(request, plan)
        cases = []
        prompt = copy.deepcopy(base)
        prompt["engine_prompt_token_ids"].append(999)
        prompt["engine_prompt_token_ids_sha256"] = adapter.sha256_value(prompt["engine_prompt_token_ids"])
        cases.append((prompt, "engine prompt token IDs"))
        truncated = copy.deepcopy(base); truncated["prompt_truncated"] = True
        cases.append((truncated, "truncation"))
        seed = copy.deepcopy(base); seed["seed"] += 1
        cases.append((seed, "seed"))
        schema = copy.deepcopy(base)
        schema["sampling_params"]["structured_outputs"]["json"] = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        schema["sampling_params_sha256"] = adapter.sha256_value(schema["sampling_params"])
        cases.append((schema, "sampling"))
        finish = copy.deepcopy(base); finish["finish_reason"] = "length"
        cases.append((finish, "finish"))
        output = copy.deepcopy(base); output["output_token_ids"].append(404)
        cases.append((output, "output token IDs"))
        for changed, pattern in cases:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(adapter.NativeAdapterDraftError, pattern):
                adapter.validate_draft_engine_observation(
                    changed,
                    **self.observation_kwargs("independent_a", request, reconstruction, plan, changed),
                )

    def test_gpt_harmony_strictness_and_coherent_output_self_tamper_fail_external_hash(self):
        request, reconstruction, plan = self.make_plan("independent_b")
        original = self.make_observation(request, plan)
        original_hash = adapter.sha256_value(original)
        adapter.validate_draft_engine_observation(
            original,
            **self.observation_kwargs("independent_b", request, reconstruction, plan, original),
        )
        strict = copy.deepcopy(original)
        strict["output_extraction"]["strict"] = False
        with self.assertRaisesRegex(adapter.NativeAdapterDraftError, "Harmony"):
            adapter.validate_draft_engine_observation(
                strict,
                **self.observation_kwargs("independent_b", request, reconstruction, plan, strict),
            )
        coherent = copy.deepcopy(original)
        coherent["output_token_ids"] = [707, 808]
        coherent["output_token_ids_sha256"] = adapter.sha256_value(coherent["output_token_ids"])
        coherent["text"] = '{"verdict":"candidate_2"}'
        coherent["text_sha256"] = adapter.sha256_bytes(coherent["text"].encode("utf-8"))
        coherent["output_extraction"]["final_text_sha256"] = coherent["text_sha256"]
        coherent["output_extraction"]["output_token_ids_sha256"] = coherent["output_token_ids_sha256"]
        kwargs = self.observation_kwargs("independent_b", request, reconstruction, plan, coherent)
        kwargs["expected_observation_sha256"] = original_hash
        with self.assertRaisesRegex(adapter.NativeAdapterDraftError, "external authenticated hash"):
            adapter.validate_draft_engine_observation(coherent, **kwargs)

    def test_fake_callback_cannot_self_label_real_or_return_runner_result(self):
        request, reconstruction, plan = self.make_plan()
        false_claim = self.make_observation(request, plan)
        false_claim["actual_model_execution"] = True
        obs_kwargs = self.observation_kwargs("independent_a", request, reconstruction, plan, false_claim)
        with self.assertRaisesRegex(adapter.NativeAdapterDraftError, "real or gate"):
            adapter.execute_scripted_cpu_non_gate(
                **obs_kwargs,
                expected_record_sha256="9" * 64,
                scripted_observation=lambda *_args: false_claim,
            )
        valid = self.make_observation(request, plan)
        runner_result = runner.build_native_generation_result(
            request=request,
            text=valid["text"],
            submitted_prompt_token_ids=valid["submitted_prompt_token_ids"],
            engine_prompt_token_ids=valid["engine_prompt_token_ids"],
            output_token_ids=valid["output_token_ids"],
            finish_reason="stop",
        )
        with self.assertRaisesRegex(adapter.NativeAdapterDraftError, "must be an object"):
            adapter.execute_scripted_cpu_non_gate(
                **self.observation_kwargs("independent_a", request, reconstruction, plan, valid),
                expected_record_sha256="9" * 64,
                scripted_observation=lambda *_args: runner_result,
            )

    def test_cpu_record_false_claim_and_whole_record_co_tamper_fail(self):
        request, reconstruction, plan = self.make_plan()
        observation = self.make_observation(request, plan)
        obs_kwargs = self.observation_kwargs("independent_a", request, reconstruction, plan, observation)
        expected_record = self.expected_record_hash(request, plan, observation)
        record = adapter.execute_scripted_cpu_non_gate(
            **obs_kwargs,
            expected_record_sha256=expected_record,
            scripted_observation=lambda *_args: observation,
        )
        body = copy.deepcopy(record.body)
        body["claims"]["actual_model_execution"] = True
        tampered = adapter.ScriptedCpuNonGateRecord(body=body, record_sha256=adapter.sha256_value(body))
        with self.assertRaisesRegex(adapter.NativeAdapterDraftError, "external authenticated hash"):
            adapter.validate_scripted_cpu_non_gate_record(
                tampered,
                **obs_kwargs,
                expected_record_sha256=expected_record,
            )

    def test_display_claim_global_mutation_or_rebinding_cannot_authorize_cpu_record(self):
        with self.assertRaises(TypeError):
            adapter.FALSE_EXECUTION_CLAIMS["actual_model_execution"] = True

        original_display = adapter.FALSE_EXECUTION_CLAIMS
        try:
            adapter.FALSE_EXECUTION_CLAIMS = {
                "actual_model_execution": True,
                "gate_eligible": True,
                "production_receipt": True,
                "sealed_control_evidence": True,
                "annotation_interface_readiness_established": True,
                "output_text_token_decode_parity_established": True,
            }
            request, reconstruction, plan = self.make_plan()
            observation = self.make_observation(request, plan)
            obs_kwargs = self.observation_kwargs(
                "independent_a", request, reconstruction, plan, observation
            )
            expected_record = self.expected_record_hash(request, plan, observation)
            record = adapter.execute_scripted_cpu_non_gate(
                **obs_kwargs,
                expected_record_sha256=expected_record,
                scripted_observation=lambda *_args: observation,
            )
            self.assertTrue(all(value is False for value in record.body["claims"].values()))

            malicious_body = copy.deepcopy(record.body)
            malicious_body["claims"] = dict(adapter.FALSE_EXECUTION_CLAIMS)
            malicious = adapter.ScriptedCpuNonGateRecord(
                body=malicious_body,
                record_sha256=adapter.sha256_value(malicious_body),
            )
            with self.assertRaisesRegex(
                adapter.NativeAdapterDraftError, "false claims"
            ):
                adapter.validate_scripted_cpu_non_gate_record(
                    malicious,
                    **obs_kwargs,
                    expected_record_sha256=malicious.record_sha256,
                )
        finally:
            adapter.FALSE_EXECUTION_CLAIMS = original_display


if __name__ == "__main__":
    unittest.main()
