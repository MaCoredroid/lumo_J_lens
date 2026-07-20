from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3.json"
)
MODULE_PATH = (
    ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3.py"
)
V2_CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_annotation_runner_v2.json"
)
DRAFT_CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_native_adapter_draft_v3.json"
)
DRAFT_SOURCE_PATH = (
    ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_native_adapter_draft_v3.py"
)
RUNNER_PATH = (
    ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_annotation_runner_v3.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


adapter = load_module("native_vllm_adapter_v3", MODULE_PATH)
runner = adapter.runner_v3


class FakeTokenizer:
    def __init__(self, decoded: str = '{"verdict":"candidate_1"}'):
        self.decoded = decoded
        self.decode_calls: list[tuple[list[int], dict[str, object]]] = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        digest = runner.sha256_bytes(runner.canonical_json_bytes(messages))
        return [11, int(digest[:8], 16), len(messages), 12]

    def decode(self, token_ids, **kwargs):
        self.decode_calls.append((list(token_ids), dict(kwargs)))
        return self.decoded


class Untouchable:
    def __getattribute__(self, name):
        raise AssertionError(f"unauthorized object touched: {name}")

    def __iter__(self):
        raise AssertionError("unauthorized iterable touched")


class NativeVllmAdapterV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_sha = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
        cls.source_sha = hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()
        cls.config = adapter.load_adapter_config(
            path=CONFIG_PATH, expected_config_sha256=cls.config_sha
        )

    def make_context(self, role: str = "independent_a", tokenizer=None):
        role_spec = self.config["roles"][role]
        model_identity = {
            field: copy.deepcopy(role_spec[field])
            for field in adapter.MODEL_IDENTITY_FIELDS
        }
        tokenizer_identity = {
            "repo_id": role_spec["repo_id"],
            "revision": role_spec["revision"],
            "snapshot_tree_sha256": role_spec["snapshot_tree_sha256"],
            "tokenizer_mode": role_spec["tokenizer_mode"],
            "tokenizer_class": role_spec["tokenizer_class"],
            "vocab_identity_sha256": role_spec["vocab_identity_sha256"],
        }
        return runner.authenticate_native_generation_context(
            tokenizer=tokenizer or FakeTokenizer(),
            model_identity=model_identity,
            expected_model_identity_sha256=adapter.sha256_value(model_identity),
            tokenizer_identity=tokenizer_identity,
            expected_tokenizer_identity_sha256=adapter.sha256_value(tokenizer_identity),
            chat_template_kwargs=role_spec["chat_template_kwargs"],
        )

    def make_request_and_spec(
        self, role: str = "independent_a", seed: int | None = None
    ):
        role_spec = self.config["roles"][role]
        request_seed = role_spec["seed"] if seed is None else seed
        messages = [
            {"role": "system", "content": "Return one schema object."},
            {"role": "user", "content": "Choose x."},
        ]
        schema = {
            "type": "object",
            "properties": {"verdict": {"type": "string", "enum": ["candidate_1"]}},
            "required": ["verdict"],
            "additionalProperties": False,
        }
        kwargs = {
            "context": self.make_context(role),
            "messages": messages,
            "schema": schema,
            "seed": request_seed,
            "stage": "completion_decision",
            "annotation_pass": "completion_chain",
            "packet_id_sha256": "a" * 64,
            "source_id_sha256": "b" * 64,
            "lineage_bindings": {"candidate_catalog_sha256": "c" * 64},
        }
        request = runner.build_native_generation_request(**kwargs)
        spec = adapter.NativeRequestSpec(
            messages=messages,
            schema=schema,
            seed=request_seed,
            stage="completion_decision",
            annotation_pass="completion_chain",
            packet_id_sha256="a" * 64,
            source_id_sha256="b" * 64,
            lineage_bindings={"candidate_catalog_sha256": "c" * 64},
            expected_native_request_sha256=request.request_sha256,
        )
        return request, spec

    def make_launch(self, role: str, request_batch_sha256: str):
        role_spec = self.config["roles"][role]
        model_identity = {
            field: copy.deepcopy(role_spec[field])
            for field in adapter.MODEL_IDENTITY_FIELDS
        }
        tokenizer_identity = {
            "repo_id": role_spec["repo_id"],
            "revision": role_spec["revision"],
            "snapshot_tree_sha256": role_spec["snapshot_tree_sha256"],
            "tokenizer_mode": role_spec["tokenizer_mode"],
            "tokenizer_class": role_spec["tokenizer_class"],
            "vocab_identity_sha256": role_spec["vocab_identity_sha256"],
        }
        return {
            "schema_version": 1,
            "kind": adapter.LAUNCH_KIND,
            "execution_authorized": True,
            "model_access_authorized": True,
            "gpu_access_authorized": True,
            "output_authorized": True,
            "production_receipt_authorized": True,
            "gate_eligible_execution_authorized": True,
            "adapter_config_sha256": self.config_sha,
            "adapter_source_sha256": self.source_sha,
            "runner_sha256": hashlib.sha256(RUNNER_PATH.read_bytes()).hexdigest(),
            "draft_config_sha256": hashlib.sha256(
                DRAFT_CONFIG_PATH.read_bytes()
            ).hexdigest(),
            "draft_source_sha256": hashlib.sha256(
                DRAFT_SOURCE_PATH.read_bytes()
            ).hexdigest(),
            "v2_config_sha256": hashlib.sha256(V2_CONFIG_PATH.read_bytes()).hexdigest(),
            "role": role,
            "model_identity": model_identity,
            "model_identity_sha256": adapter.sha256_value(model_identity),
            "snapshot_inventory_sha256": "d" * 64,
            "tokenizer_identity": tokenizer_identity,
            "tokenizer_identity_sha256": adapter.sha256_value(tokenizer_identity),
            "package_bundle_sha256": "e" * 64,
            "runtime_identity_sha256": "f" * 64,
            "environment_identity_sha256": "1" * 64,
            "gpu_identity_sha256": "2" * 64,
            "request_batch_sha256": request_batch_sha256,
            "authorization_nonce_sha256": "3" * 64,
        }

    def test_config_authenticates_exact_source_runner_draft_v2_and_full_roster(self):
        authenticated = adapter.authenticate_adapter_config(
            path=CONFIG_PATH, expected_config_sha256=self.config_sha
        )
        self.assertEqual(authenticated.source_sha256, self.source_sha)
        self.assertEqual(
            authenticated.source_sha256,
            self.config["implementation"]["source_sha256"],
        )
        self.assertEqual(
            authenticated.runner_sha256,
            "c35af19c9f3f1e38208ba7f23467c386ebff2e5fb6572582f9a620f51f394aeb",
        )
        self.assertEqual(
            authenticated.draft_config_sha256,
            "156244aa7508306b0791f913ce52461cdde79526e50a03873801a6e38d6d5eeb",
        )
        self.assertEqual(
            authenticated.draft_source_sha256,
            "50868a75d5c1dcd1181db4e655f61d7614007a15d60ad6fc7715fb11fbfc8d97",
        )
        v2 = json.loads(V2_CONFIG_PATH.read_text(encoding="utf-8"))
        projection = {
            role: {key: value[key] for key in adapter.V2_ROLE_FIELDS}
            for role, value in self.config["roles"].items()
        }
        self.assertEqual(projection, v2["roles"])
        self.assertEqual(
            {
                key: self.config["generation"][key]
                for key in adapter.V2_GENERATION_FIELDS
            },
            v2["generation"],
        )

    def test_bad_external_config_hash_rejects_before_any_other_input_access(self):
        with self.assertRaisesRegex(
            adapter.NativeVllmAdapterError, "external authenticated hash"
        ):
            adapter.execute_production_native_batch(
                expected_config_sha256="0" * 64,
                launch_binding_path=Untouchable(),
                expected_launch_binding_sha256="1" * 64,
                role=Untouchable(),
                request_specs=Untouchable(),
            )

    def test_adapter_config_must_remain_all_false_and_cannot_embed_launch_authority(
        self,
    ):
        authorization = self.config["authorization"]
        flag_names = [name for name in authorization if name.endswith("authorized")]
        self.assertTrue(all(authorization[name] is False for name in flag_names))
        self.assertIsNone(authorization["authorized_launch_binding_path"])
        self.assertIsNone(authorization["authorized_launch_binding_sha256"])
        changed = copy.deepcopy(self.config)
        for name in flag_names:
            changed["authorization"][name] = True
        changed["authorization"]["authorized_launch_binding_path"] = (
            "configs/attacker.json"
        )
        changed["authorization"]["authorized_launch_binding_sha256"] = "4" * 64
        with self.assertRaisesRegex(
            adapter.NativeVllmAdapterError, "must remain false-state"
        ):
            adapter.validate_adapter_config(changed)

    def test_config_external_hash_and_source_hash_reject_co_tamper(self):
        changed = copy.deepcopy(self.config)
        changed["roles"]["independent_a"]["repo_id"] = "attacker/model"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            changed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                adapter.NativeVllmAdapterError, "external authenticated hash"
            ):
                adapter.authenticate_adapter_config(
                    path=path, expected_config_sha256=self.config_sha
                )
            with self.assertRaises(adapter.NativeVllmAdapterError):
                adapter.authenticate_adapter_config(
                    path=path, expected_config_sha256=changed_hash
                )

    def test_exact_three_family_tokenizer_snapshot_and_extraction_roster(self):
        expected = {
            "independent_a": (
                "nvidia/Qwen3.6-27B-NVFP4",
                "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
                17,
                21941623844,
                "transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer",
                248077,
                "420ab4b96193b4325156f56cd7c3876a8a0d46f515fe4f711a7a6bf5553bf8fa",
                "direct_structured_text",
            ),
            "independent_b": (
                "openai/gpt-oss-20b",
                "6cee5e81ee83917806bbde320786a8fb61efebee",
                16,
                13789278132,
                "transformers.tokenization_utils_tokenizers.TokenizersBackend",
                200019,
                "c473d7dea3722ad7fe8f6abf41274a326ee1ad25d1aba9034a36a9c663145d0e",
                "openai_harmony_final_channel",
            ),
            "adjudicator": (
                "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",
                "72add9a622385f641297a6feef5837a06300eecb",
                17,
                15057754757,
                "transformers.tokenization_mistral_common.MistralCommonBackend",
                130044,
                "02f7bdb5b17ed8abcfb48ea188eac98a07d766c9e215c675baf357fbdee44786",
                "direct_structured_text",
            ),
        }
        for role, values in expected.items():
            spec = self.config["roles"][role]
            observed = (
                spec["repo_id"],
                spec["revision"],
                spec["snapshot_file_count"],
                spec["snapshot_size_bytes"],
                spec["tokenizer_class"],
                spec["vocab_size"],
                spec["vocab_identity_sha256"],
                spec["output_extraction"],
            )
            self.assertEqual(observed, values)

    def test_snapshot_inventory_binds_paths_file_bytes_count_size_and_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            snapshot.mkdir()
            (snapshot / "a.json").write_bytes(b"alpha")
            subdir = snapshot / "sub"
            subdir.mkdir()
            (subdir / "b.bin").write_bytes(b"beta")
            inventory = adapter.snapshot_inventory(snapshot)
            entries = [
                {
                    "path": "a.json",
                    "size_bytes": 5,
                    "sha256": hashlib.sha256(b"alpha").hexdigest(),
                },
                {
                    "path": "sub/b.bin",
                    "size_bytes": 4,
                    "sha256": hashlib.sha256(b"beta").hexdigest(),
                },
            ]
            self.assertEqual(inventory["files"], entries)
            self.assertEqual(inventory["file_count"], 2)
            self.assertEqual(inventory["size_bytes"], 9)
            self.assertEqual(inventory["tree_sha256"], adapter.sha256_value(entries))
            before = inventory["inventory_sha256"]
            (subdir / "b.bin").write_bytes(b"changed")
            self.assertNotEqual(
                adapter.snapshot_inventory(snapshot)["inventory_sha256"], before
            )

    def test_runtime_package_and_environment_contracts_are_exact_and_hashed(self):
        self.assertEqual(
            self.config["package_contract"]["runtime_import_verification"],
            {
                "root_specs_verified_before_import": True,
                "security_module_files_verified_after_import": True,
                "module_file_must_be_under_distribution_root": True,
                "record_sha256_and_size_required": True,
            },
        )
        runtime = adapter.runtime_identity(self.config)
        self.assertEqual(runtime["python_version"], "3.12.13")
        self.assertEqual(
            runtime["runtime_identity_sha256"],
            adapter.sha256_value(
                {
                    key: value
                    for key, value in runtime.items()
                    if key != "runtime_identity_sha256"
                }
            ),
        )
        env = self.config["environment_contract"]["exact_values"]
        patched = dict(env)
        patched.pop("CUDA_HOME", None)
        with mock.patch.dict(os.environ, patched, clear=True):
            identity = adapter.environment_identity(self.config)
        body = {
            key: value
            for key, value in identity.items()
            if key != "environment_identity_sha256"
        }
        self.assertEqual(
            identity["environment_identity_sha256"], adapter.sha256_value(body)
        )
        with mock.patch.dict(
            os.environ, {**patched, "HF_HUB_OFFLINE": "0"}, clear=True
        ):
            with self.assertRaisesRegex(
                adapter.NativeVllmAdapterError, "environment values"
            ):
                adapter.environment_identity(self.config)

    def test_imported_runtime_version_check_does_not_require_missing_xgrammar_dunder(
        self,
    ):
        packages = adapter.package_identity_bundle(self.config)

        def module_stub(
            distribution_name: str,
            module_name: str,
            relative_path: str,
            version: str | None = None,
        ):
            distribution = importlib.metadata.distribution(distribution_name)
            values = {
                "__name__": module_name,
                "__file__": str(distribution.locate_file(relative_path)),
            }
            if version is not None:
                values["__version__"] = version
            return SimpleNamespace(**values)

        vllm_module = module_stub("vllm", "vllm", "vllm/__init__.py", "0.23.0")
        torch_module = module_stub(
            "torch", "torch", "torch/__init__.py", "2.11.0+cu130"
        )
        transformers_module = module_stub(
            "transformers",
            "transformers",
            "transformers/__init__.py",
            "5.12.1",
        )
        xgrammar_module = module_stub("xgrammar", "xgrammar", "xgrammar/__init__.py")
        harmony_module = module_stub(
            "openai-harmony",
            "openai_harmony",
            "openai_harmony/__init__.py",
        )
        adapter._validate_runtime_import_specs(packages)
        adapter._validate_imported_runtime_versions(
            config=self.config,
            package_identity=packages,
            vllm_module=vllm_module,
            torch_module=torch_module,
            transformers_module=transformers_module,
            xgrammar_module=xgrammar_module,
            openai_harmony_module=harmony_module,
        )
        with tempfile.TemporaryDirectory() as directory:
            shadow = Path(directory) / "xgrammar.py"
            shadow.write_text("__version__ = '0.2.2'\n", encoding="utf-8")
            with self.assertRaisesRegex(
                adapter.NativeVllmAdapterError, "outside.*distribution root"
            ):
                adapter._verify_imported_distribution_module(
                    module=SimpleNamespace(__name__="xgrammar", __file__=str(shadow)),
                    expected_module_name="xgrammar",
                    distribution_name="xgrammar",
                    package_identity=packages,
                )

    def test_request_batch_and_non_gate_plan_bind_exact_native_ids_schema_seed_and_no_transform(
        self,
    ):
        request, spec = self.make_request_and_spec()
        batch = adapter.request_batch_descriptor(
            role="independent_a", request_specs=[spec], config=self.config
        )
        self.assertEqual(batch["request_count"], 1)
        self.assertEqual(
            batch["requests"][0]["expected_native_request_sha256"],
            request.request_sha256,
        )
        plan = adapter.build_non_gate_exact_token_plan(
            request=request, config=self.config
        )
        self.assertEqual(
            plan.body["tokens_prompt"]["prompt_token_ids"],
            request.body["submitted_prompt_token_ids"],
        )
        self.assertEqual(
            plan.body["sampling_params"]["seed"],
            self.config["roles"]["independent_a"]["seed"],
        )
        self.assertEqual(
            plan.body["sampling_params"]["structured_outputs"]["json"],
            request.body["response_schema"],
        )
        self.assertTrue(
            all(value is False for value in plan.body["input_transformations"].values())
        )
        self.assertTrue(all(value is False for value in plan.body["claims"].values()))

    def test_authenticated_batch_cap_is_distinct_from_scheduler_concurrency(self):
        _request, spec = self.make_request_and_spec()
        scheduler_limit = int(self.config["generation"]["max_num_seqs"])
        batch = adapter.request_batch_descriptor(
            role="independent_a",
            request_specs=[spec] * (scheduler_limit + 1),
            config=self.config,
        )
        self.assertEqual(batch["request_count"], scheduler_limit + 1)
        with self.assertRaisesRegex(
            adapter.NativeVllmAdapterError, "authenticated batch cap"
        ):
            adapter.request_batch_descriptor(
                role="independent_a",
                request_specs=[spec]
                * (adapter.MAX_AUTHENTICATED_BATCH_REQUESTS + 1),
                config=self.config,
            )

    def test_per_case_adjudication_seed_is_request_and_launch_bound_not_role_seed(self):
        request, spec = self.make_request_and_spec("adjudicator", seed=1001)
        self.assertNotEqual(spec.seed, self.config["roles"]["adjudicator"]["seed"])
        batch = adapter.request_batch_descriptor(
            role="adjudicator", request_specs=[spec], config=self.config
        )
        self.assertEqual(batch["requests"][0]["seed"], 1001)
        plan = adapter.build_non_gate_exact_token_plan(
            request=request, config=self.config
        )
        self.assertEqual(plan.body["sampling_params"]["seed"], 1001)
        self.assertEqual(
            self.config["generation"]["per_request_sampling_seed_source"],
            "native_generation_request.seed_bound_in_launch_request_batch",
        )

    def test_non_gate_claims_resist_export_mutation_rebind_and_coherent_rehash(self):
        request, _spec = self.make_request_and_spec()
        plan = adapter.build_non_gate_exact_token_plan(
            request=request, config=self.config
        )
        record = adapter.build_non_gate_test_record(
            plan=plan,
            engine_prompt_token_ids=request.body["submitted_prompt_token_ids"],
            output_token_ids=[91, 92, 93],
            finish_reason="stop",
        )
        adapter.validate_non_gate_test_record(
            record,
            plan=plan,
            expected_record_sha256=record.record_sha256,
        )
        self.assertIsInstance(adapter.FALSE_NON_GATE_CLAIMS, MappingProxyType)
        with self.assertRaises(TypeError):
            adapter.FALSE_NON_GATE_CLAIMS["actual_model_execution"] = True

        original_export = adapter.FALSE_NON_GATE_CLAIMS
        adapter.FALSE_NON_GATE_CLAIMS = {key: True for key in original_export}
        try:
            rebuilt = adapter.build_non_gate_test_record(
                plan=plan,
                engine_prompt_token_ids=request.body["submitted_prompt_token_ids"],
                output_token_ids=[91, 92, 93],
                finish_reason="stop",
            )
            self.assertTrue(
                all(value is False for value in rebuilt.body["claims"].values())
            )
            malicious_body = copy.deepcopy(record.body)
            malicious_body["claims"] = {
                "actual_model_execution": True,
                "gate_eligible": True,
                "production_receipt": True,
                "sealed_control_evidence": True,
                "output_token_text_parity_observed_on_real_model": True,
            }
            malicious = adapter.NonGateTestRecord(
                body=malicious_body,
                record_sha256=adapter.sha256_value(malicious_body),
            )
            with self.assertRaisesRegex(adapter.NativeVllmAdapterError, "claims"):
                adapter.validate_non_gate_test_record(
                    malicious,
                    plan=plan,
                    expected_record_sha256=malicious.record_sha256,
                )
        finally:
            adapter.FALSE_NON_GATE_CLAIMS = original_export

    def test_launch_requires_external_hash_and_binds_every_static_identity(self):
        _request, spec = self.make_request_and_spec()
        batch = adapter.request_batch_descriptor(
            role="independent_a", request_specs=[spec], config=self.config
        )
        launch = self.make_launch("independent_a", batch["request_batch_sha256"])
        authenticated_config = adapter.authenticate_adapter_config(
            path=CONFIG_PATH, expected_config_sha256=self.config_sha
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launch.json"
            path.write_text(json.dumps(launch, sort_keys=True), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            authenticated = adapter.authenticate_launch_binding(
                authenticated_config=authenticated_config,
                launch_binding_path=path,
                expected_launch_binding_sha256=digest,
                role="independent_a",
                request_batch_sha256=batch["request_batch_sha256"],
            )
            self.assertEqual(authenticated.file_sha256, digest)
            with self.assertRaisesRegex(
                adapter.NativeVllmAdapterError, "external authenticated hash"
            ):
                adapter.authenticate_launch_binding(
                    authenticated_config=authenticated_config,
                    launch_binding_path=path,
                    expected_launch_binding_sha256="0" * 64,
                    role="independent_a",
                    request_batch_sha256=batch["request_batch_sha256"],
                )
            changed = copy.deepcopy(launch)
            changed["request_batch_sha256"] = "9" * 64
            path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
            changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                adapter.NativeVllmAdapterError, "request hashes"
            ):
                adapter.authenticate_launch_binding(
                    authenticated_config=authenticated_config,
                    launch_binding_path=path,
                    expected_launch_binding_sha256=changed_digest,
                    role="independent_a",
                    request_batch_sha256=batch["request_batch_sha256"],
                )

    def test_malformed_launch_rejects_before_any_runtime_import_or_gpu_access(self):
        _request, spec = self.make_request_and_spec()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launch.json"
            path.write_text("{}", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            real_import = __import__

            def guarded_import(name, *args, **kwargs):
                if name.split(".")[0] in {
                    "vllm",
                    "torch",
                    "transformers",
                    "xgrammar",
                    "openai_harmony",
                }:
                    raise AssertionError(f"runtime import touched: {name}")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=guarded_import):
                with self.assertRaisesRegex(
                    adapter.NativeVllmAdapterError, "launch binding fields"
                ):
                    adapter.execute_production_native_batch(
                        expected_config_sha256=self.config_sha,
                        launch_binding_path=path,
                        expected_launch_binding_sha256=digest,
                        role="independent_a",
                        request_specs=[spec],
                    )

    def test_production_signature_has_no_injected_backend_and_static_token_path_is_exact(
        self,
    ):
        signature = inspect.signature(adapter.execute_production_native_batch)
        self.assertEqual(
            list(signature.parameters),
            [
                "expected_config_sha256",
                "launch_binding_path",
                "expected_launch_binding_sha256",
                "role",
                "request_specs",
            ],
        )
        forbidden = {
            "backend",
            "callback",
            "engine",
            "llm",
            "tokenizer",
            "session",
            "output",
        }
        self.assertFalse(forbidden & set(signature.parameters))

        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        production = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_execute_authorized_production"
        )
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(production)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports |= {
            node.module.split(".")[0]
            for node in ast.walk(production)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertTrue({"torch", "transformers", "vllm", "xgrammar"} <= imports)
        tokens_prompt_calls = [
            node
            for node in ast.walk(production)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TokensPrompt"
        ]
        self.assertEqual(len(tokens_prompt_calls), 1)
        call = tokens_prompt_calls[0]
        self.assertEqual(call.args, [])
        self.assertEqual([item.arg for item in call.keywords], ["prompt_token_ids"])
        sampling_calls = [
            node
            for node in ast.walk(production)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SamplingParams"
        ]
        self.assertEqual(len(sampling_calls), 1)
        self.assertEqual(
            {item.arg for item in sampling_calls[0].keywords},
            {"temperature", "top_p", "seed", "max_tokens", "structured_outputs"},
        )
        token_block = source[
            source.index("token_prompts = [") : source.index("outputs = llm.generate(")
        ]
        self.assertNotIn(".decode(", token_block)
        self.assertNotIn("apply_chat_template", token_block)
        self.assertNotIn("encode(", token_block)
        self.assertNotIn("truncate_prompt_tokens", token_block)

    def test_auto_tokenizer_is_local_only_no_remote_code_and_engine_kwargs_exact(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "AutoTokenizer"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertIsInstance(keywords["local_files_only"], ast.Constant)
        self.assertIs(keywords["local_files_only"].value, True)
        self.assertIs(keywords["trust_remote_code"].value, False)
        for role in self.config["roles"]:
            kwargs = adapter.build_vllm_engine_kwargs(
                config=self.config, role=role, snapshot_path=ROOT
            )
            spec = self.config["roles"][role]
            self.assertEqual(kwargs["seed"], spec["seed"])
            self.assertEqual(kwargs["quantization"], spec["quantization"])
            self.assertEqual(
                kwargs["structured_outputs_config"],
                {"backend": "xgrammar", "disable_any_whitespace": True},
            )
            for key, value in spec["vllm_engine_kwargs"].items():
                self.assertEqual(kwargs[key], value)

    def test_output_token_decode_parity_is_exact_and_candidate_tamper_fails(self):
        tokenizer = FakeTokenizer('{"verdict":"candidate_1"}')
        text, provenance = adapter.decode_output_ids_with_exact_parity(
            tokenizer=tokenizer,
            output_token_ids=[91, 92, 93],
            candidate_text='{"verdict":"candidate_1"}',
        )
        self.assertEqual(text, '{"verdict":"candidate_1"}')
        self.assertTrue(provenance["candidate_text_exact_parity"])
        self.assertEqual(
            tokenizer.decode_calls,
            [
                (
                    [91, 92, 93],
                    {
                        "skip_special_tokens": True,
                        "clean_up_tokenization_spaces": False,
                    },
                )
            ],
        )
        with self.assertRaisesRegex(adapter.NativeVllmAdapterError, "differs"):
            adapter.decode_output_ids_with_exact_parity(
                tokenizer=tokenizer,
                output_token_ids=[91, 92, 93],
                candidate_text="tampered",
            )
        with self.assertRaisesRegex(adapter.NativeVllmAdapterError, "candidate text"):
            adapter.decode_output_ids_with_exact_parity(
                tokenizer=tokenizer,
                output_token_ids=[91, 92, 93],
                candidate_text=None,
            )

    def test_actual_sampling_object_binds_schema_whitespace_seed_and_no_truncation(
        self,
    ):
        request, _spec = self.make_request_and_spec()
        body = request.body
        structured = SimpleNamespace(
            json=copy.deepcopy(body["response_schema"]),
            disable_any_whitespace=True,
        )
        params = SimpleNamespace(
            temperature=0.0,
            top_p=1.0,
            seed=body["seed"],
            max_tokens=768,
            truncate_prompt_tokens=None,
            structured_outputs=structured,
        )
        descriptor = adapter.validate_sampling_params_against_native_request(
            params=params,
            request_body=body,
            generation=self.config["generation"],
        )
        self.assertIsNone(descriptor["truncate_prompt_tokens"])
        self.assertEqual(
            descriptor["structured_outputs"]["json"], body["response_schema"]
        )
        self.assertEqual(
            adapter._role_max_output_tokens(
                role_spec=self.config["roles"]["independent_b"],
                generation=self.config["generation"],
            ),
            2048,
        )
        self.assertEqual(
            adapter._role_max_output_tokens(
                role_spec=self.config["roles"]["independent_a"],
                generation=self.config["generation"],
            ),
            768,
        )
        harmony_params = copy.deepcopy(params)
        harmony_params.max_tokens = 2048
        harmony_descriptor = adapter.validate_sampling_params_against_native_request(
            params=harmony_params,
            request_body=body,
            generation=self.config["generation"],
            max_output_tokens=2048,
        )
        self.assertEqual(harmony_descriptor["max_tokens"], 2048)
        attacks = []
        wrong_schema = copy.deepcopy(params)
        wrong_schema.structured_outputs.json = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        attacks.append(wrong_schema)
        whitespace = copy.deepcopy(params)
        whitespace.structured_outputs.disable_any_whitespace = False
        attacks.append(whitespace)
        truncated = copy.deepcopy(params)
        truncated.truncate_prompt_tokens = 128
        attacks.append(truncated)
        seed = copy.deepcopy(params)
        seed.seed += 1
        attacks.append(seed)
        float_seed = copy.deepcopy(params)
        float_seed.seed = float(params.seed)
        attacks.append(float_seed)
        float_max_tokens = copy.deepcopy(params)
        float_max_tokens.max_tokens = float(params.max_tokens)
        attacks.append(float_max_tokens)
        boolean_temperature = copy.deepcopy(params)
        boolean_temperature.temperature = False
        attacks.append(boolean_temperature)
        for changed in attacks:
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(
                    adapter.NativeVllmAdapterError, "sampling params"
                ),
            ):
                adapter.validate_sampling_params_against_native_request(
                    params=changed,
                    request_body=body,
                    generation=self.config["generation"],
                )

    def test_preflight_and_runtime_receipts_have_exact_shapes_and_gate_binding(self):
        request, spec = self.make_request_and_spec()
        batch = adapter.request_batch_descriptor(
            role="independent_a", request_specs=[spec], config=self.config
        )
        runtime_body = {
            "implementation": self.config["package_contract"]["interpreter"][
                "implementation"
            ],
            "python_version": self.config["package_contract"]["interpreter"]["version"],
            "executable": str(
                Path(
                    self.config["package_contract"]["interpreter"]["executable"]
                ).resolve(strict=True)
            ),
            "platform": self.config["package_contract"]["platform"],
        }
        runtime_identity = {
            **runtime_body,
            "runtime_identity_sha256": adapter.sha256_value(runtime_body),
        }
        environment_body = {
            "exact_values": copy.deepcopy(
                self.config["environment_contract"]["exact_values"]
            ),
            "absent_values": {
                name: None
                for name in self.config["environment_contract"]["must_be_absent"]
            },
        }
        environment_identity = {
            **environment_body,
            "environment_identity_sha256": adapter.sha256_value(environment_body),
        }
        distributions = {}
        for name, version in self.config["package_contract"]["distributions"].items():
            identity_body = {
                "requested_name": name,
                "distribution_name": name,
                "version": version,
                "metadata_sha256": adapter.sha256_bytes(
                    f"{name}-metadata".encode("utf-8")
                ),
                "record_sha256": adapter.sha256_bytes(f"{name}-record".encode("utf-8")),
            }
            distributions[name] = {
                **identity_body,
                "identity_sha256": adapter.sha256_value(identity_body),
            }
        package_body = {
            "algorithm": self.config["package_contract"][
                "distribution_identity_algorithm"
            ],
            "distributions": distributions,
        }
        package_identity = {
            **package_body,
            "package_bundle_sha256": adapter.sha256_value(package_body),
        }
        snapshot_files = [
            {
                "path": "config.json",
                "size_bytes": 2,
                "sha256": adapter.sha256_bytes(b"{}"),
            }
        ]
        snapshot_body = {
            "snapshot_path": "/authenticated/test-snapshot",
            "files": snapshot_files,
            "tree_sha256": adapter.sha256_value(snapshot_files),
            "file_count": len(snapshot_files),
            "size_bytes": sum(item["size_bytes"] for item in snapshot_files),
        }
        snapshot = {
            **snapshot_body,
            "inventory_sha256": adapter.sha256_value(snapshot_body),
        }
        gpu_body = {
            "torch_cuda_version": "13.0",
            "cudnn_version": 91000,
            "visible_device_count": self.config["gpu_contract"][
                "expected_visible_device_count"
            ],
            "device_index": self.config["gpu_contract"]["expected_device_index"],
            "device_name": self.config["gpu_contract"]["expected_device_name"],
            "compute_capability": [12, 0],
            "total_memory_bytes": 34_000_000_000,
        }
        gpu_identity = {
            **gpu_body,
            "gpu_identity_sha256": adapter.sha256_value(gpu_body),
        }
        launch_value = self.make_launch("independent_a", batch["request_batch_sha256"])
        launch_value.update(
            {
                "runtime_identity_sha256": runtime_identity["runtime_identity_sha256"],
                "environment_identity_sha256": environment_identity[
                    "environment_identity_sha256"
                ],
                "package_bundle_sha256": package_identity["package_bundle_sha256"],
                "snapshot_inventory_sha256": snapshot["inventory_sha256"],
                "gpu_identity_sha256": gpu_identity["gpu_identity_sha256"],
            }
        )
        authenticated_config = adapter.authenticate_adapter_config(
            path=CONFIG_PATH, expected_config_sha256=self.config_sha
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launch.json"
            path.write_text(json.dumps(launch_value, sort_keys=True), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            launch = adapter.authenticate_launch_binding(
                authenticated_config=authenticated_config,
                launch_binding_path=path,
                expected_launch_binding_sha256=digest,
                role="independent_a",
                request_batch_sha256=batch["request_batch_sha256"],
            )
            authority = adapter._production_authority(
                authenticated_config=authenticated_config,
                launch=launch,
                request_batch=batch,
            )
            tokenizer_identity = copy.deepcopy(launch_value["tokenizer_identity"])
            preflight = adapter._build_preflight_receipt(
                authority=authority,
                role="independent_a",
                runtime=runtime_identity,
                environment=environment_identity,
                packages=package_identity,
                snapshot=snapshot,
                tokenizer_identity=tokenizer_identity,
                gpu_identity=gpu_identity,
            )
            adapter.validate_preflight_receipt(
                preflight,
                authority=authority,
                expected_receipt_sha256=preflight.receipt_sha256,
            )
            changed_preflight_body = copy.deepcopy(preflight.body)
            changed_preflight_body["attacker"] = True
            changed_preflight = adapter.PreflightReceipt(
                body=changed_preflight_body,
                receipt_sha256=adapter.sha256_value(changed_preflight_body),
            )
            with self.assertRaisesRegex(
                adapter.NativeVllmAdapterError, "identity or claim"
            ):
                adapter.validate_preflight_receipt(
                    changed_preflight,
                    authority=authority,
                    expected_receipt_sha256=changed_preflight.receipt_sha256,
                )

            def change_runtime(value):
                identity = value["runtime_identity"]
                identity["platform"] = "attacker-platform"
                identity["runtime_identity_sha256"] = adapter.sha256_value(
                    {
                        key: item
                        for key, item in identity.items()
                        if key != "runtime_identity_sha256"
                    }
                )

            def change_environment(value):
                identity = value["environment_identity"]
                identity["exact_values"]["HF_HUB_OFFLINE"] = "0"
                identity["environment_identity_sha256"] = adapter.sha256_value(
                    {
                        key: item
                        for key, item in identity.items()
                        if key != "environment_identity_sha256"
                    }
                )

            def change_package(value):
                package = value["package_identity"]
                identity = package["distributions"]["vllm"]
                identity["distribution_name"] = "attacker-vllm"
                identity["identity_sha256"] = adapter.sha256_value(
                    {
                        key: item
                        for key, item in identity.items()
                        if key != "identity_sha256"
                    }
                )
                package["package_bundle_sha256"] = adapter.sha256_value(
                    {
                        key: item
                        for key, item in package.items()
                        if key != "package_bundle_sha256"
                    }
                )

            def change_snapshot(value):
                inventory = value["snapshot_inventory"]
                inventory["files"][0]["size_bytes"] += 1
                inventory["tree_sha256"] = adapter.sha256_value(inventory["files"])
                inventory["size_bytes"] = sum(
                    item["size_bytes"] for item in inventory["files"]
                )
                inventory["inventory_sha256"] = adapter.sha256_value(
                    {
                        key: item
                        for key, item in inventory.items()
                        if key != "inventory_sha256"
                    }
                )

            def change_tokenizer(value):
                value["tokenizer_identity"]["tokenizer_class"] = "attacker.Tokenizer"
                value["tokenizer_identity_sha256"] = adapter.sha256_value(
                    value["tokenizer_identity"]
                )

            def change_gpu(value):
                identity = value["gpu_identity"]
                identity["device_name"] = "attacker GPU"
                identity["gpu_identity_sha256"] = adapter.sha256_value(
                    {
                        key: item
                        for key, item in identity.items()
                        if key != "gpu_identity_sha256"
                    }
                )

            for label, mutate in (
                ("runtime", change_runtime),
                ("environment", change_environment),
                ("package", change_package),
                ("snapshot", change_snapshot),
                ("tokenizer", change_tokenizer),
                ("gpu", change_gpu),
            ):
                changed = copy.deepcopy(preflight.body)
                mutate(changed)
                forged = adapter.PreflightReceipt(
                    body=changed, receipt_sha256=adapter.sha256_value(changed)
                )
                with (
                    self.subTest(preflight_identity=label),
                    self.assertRaises(adapter.NativeVllmAdapterError),
                ):
                    adapter.validate_preflight_receipt(
                        forged,
                        authority=authority,
                        expected_receipt_sha256=forged.receipt_sha256,
                    )

            sampling = {
                "temperature": 0,
                "top_p": 1.0,
                "seed": self.config["roles"]["independent_a"]["seed"],
                "max_tokens": 768,
                "structured_outputs": {
                    "json": copy.deepcopy(request.body["response_schema"]),
                    "disable_any_whitespace": True,
                },
                "truncate_prompt_tokens": None,
            }
            record = {
                "position": 0,
                "native_request_sha256": request.request_sha256,
                "native_result_sha256": "4" * 64,
                "response_schema_sha256": request.body["response_schema_sha256"],
                "sampling_params": sampling,
                "sampling_params_sha256": adapter.sha256_value(sampling),
                "submitted_prompt_token_ids_sha256": request.body[
                    "submitted_prompt_token_ids_sha256"
                ],
                "engine_prompt_token_ids_sha256": request.body[
                    "submitted_prompt_token_ids_sha256"
                ],
                "engine_prompt_matches_submitted": True,
                "output_token_ids_sha256": "5" * 64,
                "output_text_sha256": "6" * 64,
                "candidate_text_token_decode_parity": True,
                "output_extraction": {
                    "mode": "direct_structured_text",
                    "schema_text_source": "authenticated_output_token_decode",
                    "output_token_decode_parity": {
                        "decode_source": (
                            "authenticated_tokenizer.decode(output_token_ids)"
                        ),
                        "skip_special_tokens": True,
                        "clean_up_tokenization_spaces": False,
                        "output_token_ids_sha256": "5" * 64,
                        "decoded_text_sha256": "6" * 64,
                        "candidate_text_sha256": "6" * 64,
                        "candidate_text_exact_parity": True,
                    },
                    "analysis_content_excluded": True,
                },
                "finish_reason": "stop",
            }
            runtime_receipt = adapter._build_runtime_receipt(
                authority=authority,
                role="independent_a",
                preflight=preflight,
                engine_kwargs={"owned": True},
                request_records=[record],
                snapshot_before=snapshot,
                snapshot_after=copy.deepcopy(snapshot),
            )
            adapter.validate_runtime_receipt(
                runtime_receipt,
                authority=authority,
                expected_receipt_sha256=runtime_receipt.receipt_sha256,
            )
            for label, mutate in (
                ("gate", lambda value: value.__setitem__("gate_eligible", False)),
                (
                    "record_field",
                    lambda value: value["request_records"][0].__setitem__(
                        "attacker", True
                    ),
                ),
                (
                    "decode_parity",
                    lambda value: value["request_records"][0]["output_extraction"][
                        "output_token_decode_parity"
                    ].__setitem__("candidate_text_exact_parity", False),
                ),
                (
                    "analysis_exclusion",
                    lambda value: value["request_records"][0][
                        "output_extraction"
                    ].__setitem__("analysis_content_excluded", False),
                ),
            ):
                changed = copy.deepcopy(runtime_receipt.body)
                mutate(changed)
                changed["request_records_sha256"] = adapter.sha256_value(
                    changed["request_records"]
                )
                forged = adapter.RuntimeReceipt(
                    body=changed, receipt_sha256=adapter.sha256_value(changed)
                )
                with (
                    self.subTest(label=label),
                    self.assertRaises(adapter.NativeVllmAdapterError),
                ):
                    adapter.validate_runtime_receipt(
                        forged,
                        authority=authority,
                        expected_receipt_sha256=forged.receipt_sha256,
                    )

    @unittest.skipUnless(
        importlib.util.find_spec("openai_harmony") is not None,
        "openai-harmony is installed only in the vLLM runtime",
    )
    def test_harmony_parser_returns_one_final_excludes_analysis_and_rejects_two_finals(
        self,
    ):
        import openai_harmony as harmony

        encoding = harmony.load_harmony_encoding(
            harmony.HarmonyEncodingName.HARMONY_GPT_OSS
        )
        analysis_text = "analysis content must be excluded"
        final_text = '{"verdict":"candidate_1"}'
        analysis_ids = encoding.render(
            harmony.Message.from_role_and_content(
                harmony.Role.ASSISTANT, analysis_text
            ).with_channel("analysis")
        )
        final_ids = encoding.render(
            harmony.Message.from_role_and_content(
                harmony.Role.ASSISTANT, final_text
            ).with_channel("final")
        )
        extracted, provenance = adapter.extract_openai_harmony_final_channel(
            [*analysis_ids, *final_ids]
        )
        self.assertEqual(extracted, final_text)
        self.assertEqual(provenance["analysis_message_count"], 1)
        self.assertIsNone(provenance["final_recipient"])
        self.assertTrue(provenance["analysis_content_excluded"])
        self.assertNotIn(analysis_text, json.dumps(provenance, sort_keys=True))

        constrained_ids = encoding.encode(
            "<|channel|>final<|constrain|>JSON<|message|>" + final_text + "<|return|>",
            allowed_special=encoding.special_tokens_set,
        )
        constrained_text, constrained_provenance = (
            adapter.extract_openai_harmony_final_channel(constrained_ids)
        )
        self.assertEqual(constrained_text, final_text)
        self.assertEqual(constrained_provenance["final_recipient"], "<|constrain|>JSON")

        choice_constrained_ids = encoding.encode(
            "<|channel|>final<|constrain|>chain<|message|>"
            + final_text
            + "<|return|>",
            allowed_special=encoding.special_tokens_set,
        )
        choice_text, choice_provenance = (
            adapter.extract_openai_harmony_final_channel(choice_constrained_ids)
        )
        self.assertEqual(choice_text, final_text)
        self.assertEqual(choice_provenance["final_recipient"], "<|constrain|>chain")

        routed_ids = encoding.encode(
            "<|channel|>final to=unsafe_tool<|message|>" + final_text + "<|return|>",
            allowed_special=encoding.special_tokens_set,
        )
        with self.assertRaisesRegex(
            adapter.NativeVllmAdapterError, "valid constraint sentinel"
        ):
            adapter.extract_openai_harmony_final_channel(routed_ids)
        with self.assertRaisesRegex(
            adapter.NativeVllmAdapterError, "exactly one final"
        ):
            adapter.extract_openai_harmony_final_channel([*final_ids, *final_ids])

    def test_harmony_receipt_provenance_binds_final_analysis_and_token_decode(self):
        record = {
            "output_token_ids_sha256": "5" * 64,
            "output_text_sha256": "6" * 64,
        }
        extraction = {
            "mode": "openai_harmony_final_channel",
            "schema_text_source": (
                "strict_harmony_final_channel_from_output_token_ids"
            ),
            "output_token_decode_parity": {
                "decode_source": "authenticated_tokenizer.decode(output_token_ids)",
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
                "output_token_ids_sha256": "5" * 64,
                "decoded_text_sha256": "7" * 64,
                "candidate_text_sha256": "7" * 64,
                "candidate_text_exact_parity": True,
            },
            "harmony": {
                "mode": "openai_harmony_final_channel",
                "parser_package": "openai-harmony",
                "parser_version": self.config["package_contract"]["distributions"][
                    "openai-harmony"
                ],
                "parser_strict": True,
                "assistant_message_count": 2,
                "analysis_message_count": 1,
                "analysis_content_excluded": True,
                "unknown_channel_count": 0,
                "final_message_count": 1,
                "final_recipient": "<|constrain|>JSON",
                "final_text_sha256": "6" * 64,
                "completion_token_ids_sha256": "5" * 64,
            },
            "analysis_content_excluded": True,
        }
        adapter._validate_output_extraction_receipt(
            value=extraction,
            record=record,
            role_spec=self.config["roles"]["independent_b"],
            package_versions=self.config["package_contract"]["distributions"],
        )
        choice_constrained = copy.deepcopy(extraction)
        choice_constrained["harmony"]["final_recipient"] = "<|constrain|>chain"
        adapter._validate_output_extraction_receipt(
            value=choice_constrained,
            record=record,
            role_spec=self.config["roles"]["independent_b"],
            package_versions=self.config["package_contract"]["distributions"],
        )
        for label, mutate in (
            (
                "unknown_channel",
                lambda value: value["harmony"].__setitem__("unknown_channel_count", 1),
            ),
            (
                "analysis_not_excluded",
                lambda value: value["harmony"].__setitem__(
                    "analysis_content_excluded", False
                ),
            ),
            (
                "wrong_final_hash",
                lambda value: value["harmony"].__setitem__(
                    "final_text_sha256", "8" * 64
                ),
            ),
            (
                "unsafe_final_recipient",
                lambda value: value["harmony"].__setitem__(
                    "final_recipient", "unsafe_tool"
                ),
            ),
            (
                "candidate_not_parity",
                lambda value: value["output_token_decode_parity"].__setitem__(
                    "candidate_text_exact_parity", False
                ),
            ),
        ):
            changed = copy.deepcopy(extraction)
            mutate(changed)
            with (
                self.subTest(harmony_receipt=label),
                self.assertRaises(adapter.NativeVllmAdapterError),
            ):
                adapter._validate_output_extraction_receipt(
                    value=changed,
                    record=record,
                    role_spec=self.config["roles"]["independent_b"],
                    package_versions=self.config["package_contract"]["distributions"],
                )

    def test_source_has_no_artifact_writer_cli_or_injected_production_helper(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("write_text(", source)
        self.assertNotIn("write_bytes(", source)
        self.assertNotIn("argparse", source)
        self.assertFalse(hasattr(adapter, "main"))
        self.assertNotRegex(source, r"open\([^\n]*['\"](?:w|a|x)[bt]?['\"]")


if __name__ == "__main__":
    unittest.main()
