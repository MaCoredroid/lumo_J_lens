from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "scripts"
    / "swe_task_state_v4_epistemic_chain_control_executor_v2.py"
)
CONTROLS_PATH = (
    ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_controls_v2.py"
)
RUNTIME_CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_annotation_runner_v2.json"
)
MODEL_INPUT_MANIFEST_PATH = (
    ROOT / "artifacts" / "epistemic-chain-controls-v2-model-inputs-manifest.json"
)
FROZEN_EXECUTION_CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_control_executor_v2.json"
)
EXPECTATION_MANIFEST_PATH = (
    ROOT / "artifacts" / "epistemic-chain-controls-v2-expectations-manifest.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


module = load_module("semantic_control_executor_v2", MODULE_PATH)
controls = load_module("semantic_controls_v2_for_executor_test", CONTROLS_PATH)


class FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return "\n".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )


def mock_model_resolver(model_spec, *, verify_contents):
    if verify_contents is not True:
        raise AssertionError("model contents must be authenticated")
    identity = {
        "repo_id": model_spec["repo_id"],
        "revision": model_spec["revision"],
        "snapshot_tree_sha256": model_spec["snapshot_tree_sha256"],
        "quantization": model_spec.get("quantization"),
        "dtype": model_spec["dtype"],
    }
    return ROOT, {
        "snapshot_path": str(ROOT),
        "file_count": model_spec["snapshot_file_count"],
        "size_bytes": model_spec["snapshot_size_bytes"],
        "files": [
            {
                "path": "mock-model-file",
                "size_bytes": 1,
                "sha256": "f" * 64,
            }
        ],
        "tree_sha256": model_spec["snapshot_tree_sha256"],
        "model_identity": identity,
        "model_identity_sha256": module.sha256_bytes(
            module.canonical_json_bytes(identity)
        ),
    }


def mock_generator_factory(*, model_path, model_spec, generation_config):
    if model_path != ROOT:
        raise AssertionError("mock resolver path changed")
    if generation_config["temperature"] != 0:
        raise AssertionError("generation contract changed")

    def generate(prompts, schema, seed):
        if seed != model_spec["seed"]:
            raise AssertionError("role seed changed")
        is_novelty = set(schema["properties"]) == {"decision", "unknown_reason"}
        proposal = (
            {"decision": "novel", "unknown_reason": ""}
            if is_novelty
            else module.quote_runner.deterministic_no_chain_proposal()
        )
        raw = module.canonical_json_text(proposal)
        return [
            module.quote_runner.GenerationResult(
                text=raw,
                prompt_token_count=len(prompt),
                output_token_count=7,
                finish_reason="stop",
                output_extraction={
                    "output_extraction": model_spec["output_extraction"],
                    "analysis_content_retained": False,
                    "final_message_count": 1,
                    "final_text_sha256": module.sha256_text(raw),
                    **(
                        {
                            "parser_package": "openai-harmony",
                            "parser_strict": True,
                        }
                        if model_spec["output_extraction"]
                        == "openai_harmony_final_channel"
                        else {}
                    ),
                },
            )
            for prompt in prompts
        ]

    return generate, {
        "mock": True,
        "load_seconds": 0.0,
        "output_extraction": model_spec["output_extraction"],
        "chat_template_kwargs": dict(model_spec["chat_template_kwargs"]),
        "chat_template_kwargs_sha256": module.sha256_bytes(
            module.canonical_json_bytes(model_spec["chat_template_kwargs"])
        ),
        "vllm_use_flashinfer_sampler": generation_config[
            "vllm_use_flashinfer_sampler"
        ],
        "vllm_enable_v1_multiprocessing": generation_config[
            "vllm_enable_v1_multiprocessing"
        ],
        "vllm_disabled_kernels": ",".join(
            generation_config["vllm_disabled_kernels"]
        ),
        "cuda_home": None,
        "prompt_token_accounting": "vllm_request_output.prompt_token_ids",
        "external_tokenizer_preflight_count_used": False,
        "prompt_truncation_requested": False,
        "engine_authoritative_context_reservation_checked": True,
        "structured_outputs_config": dict(
            module.quote_runner.STRUCTURED_OUTPUTS_ENGINE_CONFIG
        ),
        "structured_outputs_config_sha256": module.sha256_bytes(
            module.canonical_json_bytes(
                module.quote_runner.STRUCTURED_OUTPUTS_ENGINE_CONFIG
            )
        ),
        "language_model_only": True,
        "role_vllm_engine_kwargs": dict(model_spec["vllm_engine_kwargs"]),
        "llm_kwargs_sha256": "e" * 64,
        **(
            {
                "harmony_parser_package": "openai-harmony",
                "harmony_parser_strict": True,
            }
            if model_spec["output_extraction"] == "openai_harmony_final_channel"
            else {}
        ),
    }, FakeTokenizer()


class SemanticControlExecutorV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        cache = ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            dir=cache, prefix="semantic-control-executor-v2-test-"
        )
        self.work = Path(self.temporary.name)
        frozen = module.freeze_execution_config(
            runtime_config_path=RUNTIME_CONFIG_PATH,
            model_input_manifest_path=MODEL_INPUT_MANIFEST_PATH,
        )
        self.execution_config = self.work / "execution-config.json"
        self.execution_config.write_bytes(
            module.canonical_json_bytes(frozen) + b"\n"
        )
        self.execution_config_sha = module.sha256_file(self.execution_config)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_all_52_inputs_adapt_to_exact_runner_schemas(self) -> None:
        _manifest, rows, manifest_path, _record_path = module.load_sealed_model_inputs(
            MODEL_INPUT_MANIFEST_PATH,
            expected_sha256=module.sha256_file(MODEL_INPUT_MANIFEST_PATH),
        )
        adapted = [
            module.adapt_control_input_to_runner_packet(
                row,
                model_input_manifest_sha256=module.sha256_file(manifest_path),
            )
            for row in rows
        ]
        self.assertEqual(len(adapted), 52)
        self.assertEqual(
            sum(item["annotation_pass"] == "completion_chain" for item in adapted),
            44,
        )
        self.assertEqual(
            sum(item["annotation_pass"] == "prefix_novelty" for item in adapted),
            8,
        )
        for source, packet in zip(rows, adapted, strict=True):
            self.assertEqual(source["packet_id_sha256"], packet["packet_id_sha256"])
            annotation_pass = (
                "prefix_novelty" if source["pass"] == "novelty" else "completion_chain"
            )
            module.quote_runner.legacy.validate_packet(
                packet, annotation_pass=annotation_pass
            )

    def test_production_execution_config_is_exact_freeze_helper_output(self) -> None:
        frozen = json.loads(
            FROZEN_EXECUTION_CONFIG_PATH.read_text(encoding="utf-8")
        )
        expected = module.freeze_execution_config(
            runtime_config_path=RUNTIME_CONFIG_PATH,
            model_input_manifest_path=MODEL_INPUT_MANIFEST_PATH,
        )
        self.assertEqual(frozen, expected)
        self.assertEqual(
            frozen["bindings"]["runtime_config"]["sha256"],
            "a22604d78938917972dc23c517c942fb9022e867e17ce0b5a060189dd0dd1b4d",
        )
        self.assertEqual(
            frozen["bindings"]["implementations"]["control_executor_v2"][
                "sha256"
            ],
            module.sha256_file(MODULE_PATH),
        )
        module._authenticate_execution_context(
            execution_config_path=FROZEN_EXECUTION_CONFIG_PATH,
            expected_execution_config_sha256=module.sha256_file(
                FROZEN_EXECUTION_CONFIG_PATH
            ),
        )

    def test_mocked_roles_are_answer_isolated_resumable_and_dual_scorable(self) -> None:
        paths = {
            role: self.work / f"{role}-manifest.json"
            for role in module.ROLES
        }
        original_path_open = Path.open

        def forbid_expectation_open(path, *args, **kwargs):
            if "expectation" in str(path).casefold():
                raise AssertionError(f"generation opened expectation artifact: {path}")
            return original_path_open(path, *args, **kwargs)

        # This guard covers every generation and combine file open.  The runtime
        # config may bind an expectation path as forbidden, but it is never
        # dereferenced by the executor.
        with mock.patch.object(Path, "open", forbid_expectation_open):
            for role in module.ROLES:
                result = module.run_role(
                    execution_config_path=self.execution_config,
                    expected_execution_config_sha256=self.execution_config_sha,
                    role=role,
                    output_manifest_path=paths[role],
                    generator_factory=mock_generator_factory,
                    model_resolver=mock_model_resolver,
                    _allow_mocked_cpu_test=True,
                )
                self.assertEqual(
                    result["record_count"], module.ROLE_RECORD_COUNTS[role]
                )

            resumed = module.run_role(
                execution_config_path=self.execution_config,
                expected_execution_config_sha256=self.execution_config_sha,
                role="independent_a",
                output_manifest_path=paths["independent_a"],
                resume=True,
                expected_resume_manifest_sha256=module.sha256_file(
                    paths["independent_a"]
                ),
                generator_factory=mock.Mock(
                    side_effect=AssertionError("complete resume initialized a model")
                ),
                model_resolver=mock.Mock(
                    side_effect=AssertionError("complete resume resolved a model")
                ),
                _allow_mocked_cpu_test=True,
            )
            self.assertEqual(resumed["status"], "authenticated_complete_role_resume")

            with self.assertRaisesRegex(
                module.ControlExecutorError, "role-manifest hash changed"
            ):
                module.run_role(
                    execution_config_path=self.execution_config,
                    expected_execution_config_sha256=self.execution_config_sha,
                    role="independent_a",
                    output_manifest_path=paths["independent_a"],
                    resume=True,
                    expected_resume_manifest_sha256="0" * 64,
                    generator_factory=mock_generator_factory,
                    model_resolver=mock_model_resolver,
                    _allow_mocked_cpu_test=True,
                )

            with self.assertRaisesRegex(
                module.ControlExecutorError, "role-manifest hash changed"
            ):
                module.combine_primary_outputs(
                    execution_config_path=self.execution_config,
                    expected_execution_config_sha256=self.execution_config_sha,
                    independent_a_manifest_path=paths["independent_a"],
                    expected_independent_a_manifest_sha256="0" * 64,
                    independent_b_manifest_path=paths["independent_b"],
                    expected_independent_b_manifest_sha256=module.sha256_file(
                        paths["independent_b"]
                    ),
                    adjudicator_manifest_path=paths["adjudicator"],
                    expected_adjudicator_manifest_sha256=module.sha256_file(
                        paths["adjudicator"]
                    ),
                    output_a_records_path=self.work / "must-not-combine-a.jsonl",
                    output_a_manifest_path=self.work / "must-not-combine-a.json",
                    output_b_records_path=self.work / "must-not-combine-b.jsonl",
                    output_b_manifest_path=self.work / "must-not-combine-b.json",
                )

            combined = module.combine_primary_outputs(
                execution_config_path=self.execution_config,
                expected_execution_config_sha256=self.execution_config_sha,
                independent_a_manifest_path=paths["independent_a"],
                expected_independent_a_manifest_sha256=module.sha256_file(
                    paths["independent_a"]
                ),
                independent_b_manifest_path=paths["independent_b"],
                expected_independent_b_manifest_sha256=module.sha256_file(
                    paths["independent_b"]
                ),
                adjudicator_manifest_path=paths["adjudicator"],
                expected_adjudicator_manifest_sha256=module.sha256_file(
                    paths["adjudicator"]
                ),
                output_a_records_path=self.work / "combined-a.jsonl",
                output_a_manifest_path=self.work / "combined-a-manifest.json",
                output_b_records_path=self.work / "combined-b.jsonl",
                output_b_manifest_path=self.work / "combined-b-manifest.json",
            )

            dual = module.lock_both_primary_outputs(
                execution_config_path=self.execution_config,
                expected_execution_config_sha256=self.execution_config_sha,
                combined_a_manifest_path=Path(
                    combined["independent_a"]["manifest_path"]
                ),
                expected_combined_a_manifest_sha256=combined["independent_a"][
                    "manifest_sha256"
                ],
                combined_b_manifest_path=Path(
                    combined["independent_b"]["manifest_path"]
                ),
                expected_combined_b_manifest_sha256=combined["independent_b"][
                    "manifest_sha256"
                ],
                output_a_lock_manifest_path=self.work / "combined-a-lock.json",
                output_b_lock_manifest_path=self.work / "combined-b-lock.json",
                output_dual_lock_receipt_path=self.work / "dual-lock-receipt.json",
            )

        manifests = {
            role: json.loads(paths[role].read_text(encoding="utf-8"))
            for role in module.ROLES
        }
        identities = {
            manifests[role]["model"]["model_identity_sha256"]
            for role in module.ROLES
        }
        self.assertEqual(len(identities), 3)
        for role, manifest in manifests.items():
            self.assertEqual(manifest["role"], role)
            self.assertTrue(
                manifest["scope"][
                    "model_visible_payload_byte_equal_to_sealed_payload"
                ]
            )
            records = [
                json.loads(line)
                for line in (self.work / f"{role}-manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(records), module.ROLE_RECORD_COUNTS[role])
            for record in records:
                self.assertEqual(
                    record["input_payload_sha256"],
                    record["model_visible_payload_sha256"],
                )
                self.assertEqual(
                    record["result"]["provenance"]["model_identity_sha256"],
                    manifest["model"]["model_identity_sha256"],
                )

        output_a = Path(combined["independent_a"]["records_path"])
        output_b = Path(combined["independent_b"]["records_path"])
        rows_a = [json.loads(line) for line in output_a.read_text().splitlines()]
        rows_b = [json.loads(line) for line in output_b.read_text().splitlines()]
        self.assertEqual(len(rows_a), 52)
        self.assertEqual(len(rows_b), 52)
        self.assertEqual(
            [row["packet_id_sha256"] for row in rows_a],
            [row["packet_id_sha256"] for row in rows_b],
        )
        for left, right in zip(rows_a, rows_b, strict=True):
            if left["pass"] == "adjudication":
                self.assertEqual(left["result"], right["result"])
                self.assertEqual(left["result"]["role"], "adjudicator")
            else:
                self.assertEqual(left["result"]["role"], "independent_a")
                self.assertEqual(right["result"]["role"], "independent_b")

        # The score command can open expectations only through the already
        # authenticated dual-lock receipt produced under the denial guard.
        final = module.score_both_and_finalize(
            execution_config_path=self.execution_config,
            expected_execution_config_sha256=self.execution_config_sha,
            dual_lock_receipt_path=Path(dual["receipt_path"]),
            expected_dual_lock_receipt_sha256=dual["receipt_sha256"],
            expectation_manifest_path=EXPECTATION_MANIFEST_PATH,
            expected_expectation_manifest_sha256=module.sha256_file(
                EXPECTATION_MANIFEST_PATH
            ),
            expected_decoder_addendum_sha256=module.sha256_file(
                module.DECODER_ADDENDUM_PATH
            ),
            output_a_report_path=self.work / "report-a.json",
            output_b_report_path=self.work / "report-b.json",
            output_final_receipt_path=self.work / "final-gate-receipt.json",
        )
        report_a = json.loads((self.work / "report-a.json").read_text())
        report_b = json.loads((self.work / "report-b.json").read_text())
        self.assertEqual(report_a["kind"], controls.REPORT_KIND)
        self.assertEqual(report_b["kind"], controls.REPORT_KIND)
        self.assertEqual(report_a["counts"]["completion"], 32)
        self.assertEqual(report_b["counts"]["adjudication"], 12)
        self.assertEqual(
            final["status"], "mocked_cpu_test_protocol_only_no_readiness_claim"
        )

        # Even if an attacker rewrites both records and manifest and supplies
        # the new manifest hash, strict persisted provenance rejects the pair.
        original_manifest = manifests["independent_a"]
        original_records = [
            json.loads(line)
            for line in (self.work / "independent_a-manifest.jsonl")
            .read_text()
            .splitlines()
        ]
        original_records[0]["result"]["provenance"][
            "execution_config_sha256"
        ] = "0" * 64
        altered_records_path = self.work / "altered-role.jsonl"
        altered_records_path.write_text(
            "".join(
                module.canonical_json_text(row) + "\n" for row in original_records
            ),
            encoding="utf-8",
        )
        altered_manifest = json.loads(json.dumps(original_manifest))
        altered_manifest["records"]["path"] = altered_records_path.name
        altered_manifest["records"]["sha256"] = module.sha256_file(
            altered_records_path
        )
        altered_manifest_path = self.work / "altered-role.json"
        altered_manifest_path.write_bytes(
            module.canonical_json_bytes(altered_manifest) + b"\n"
        )
        with self.assertRaisesRegex(
            module.ControlExecutorError, "provenance or blinding"
        ):
            module.run_role(
                execution_config_path=self.execution_config,
                expected_execution_config_sha256=self.execution_config_sha,
                role="independent_a",
                output_manifest_path=altered_manifest_path,
                resume=True,
                expected_resume_manifest_sha256=module.sha256_file(
                    altered_manifest_path
                ),
                generator_factory=mock_generator_factory,
                model_resolver=mock_model_resolver,
                _allow_mocked_cpu_test=True,
            )

    def test_tampered_runtime_binding_fails_before_model_resolution(self) -> None:
        value = json.loads(self.execution_config.read_text(encoding="utf-8"))
        value["bindings"]["runtime_config"]["sha256"] = "0" * 64
        tampered = self.work / "tampered-execution-config.json"
        tampered.write_bytes(module.canonical_json_bytes(value) + b"\n")
        resolver = mock.Mock(side_effect=AssertionError("model resolved after auth failure"))
        with self.assertRaisesRegex(module.ControlExecutorError, "runtime config hash"):
            module.run_role(
                execution_config_path=tampered,
                expected_execution_config_sha256=module.sha256_file(tampered),
                role="independent_a",
                output_manifest_path=self.work / "must-not-exist.json",
                generator_factory=mock_generator_factory,
                model_resolver=resolver,
                _allow_mocked_cpu_test=True,
            )
        resolver.assert_not_called()

    def test_perfect_mock_reports_are_structurally_non_gate(self) -> None:
        perfect = {"status": "passed", "gates": {"all": True}}
        mock_binding = {
            "path": str(self.work / "mock-role.json"),
            "sha256": "a" * 64,
            "model_identity_sha256": "b" * 64,
            "execution_backend": "injected_test_non_gate",
            "gate_eligible": False,
        }
        roles = {
            "independent_a": dict(mock_binding),
            "independent_b": dict(mock_binding),
            "adjudicator": dict(mock_binding),
        }
        self.assertFalse(
            module._final_gate_passed(
                report_a=perfect, report_b=perfect, decision_roles=roles
            )
        )

    def test_output_guard_checks_canonical_reserved_parent(self) -> None:
        logical_parent = self.work / "alias"
        logical_parent.mkdir()
        logical_output = logical_parent / "output.json"
        original_resolve = Path.resolve

        def redirected_resolve(path, *args, **kwargs):
            if path == logical_parent:
                # No reserved directory is created or opened; this simulates a
                # symlink-parent canonicalization result for the guard.
                return self.work / "validation" / "sink"
            return original_resolve(path, *args, **kwargs)

        with mock.patch.object(Path, "resolve", redirected_resolve):
            with self.assertRaisesRegex(
                module.ControlExecutorError, "resolves through a reserved path"
            ):
                module._guard_output(logical_output, label="canonical guard test")


if __name__ == "__main__":
    unittest.main()
