from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PRODUCER = load_module(
    "stage_a_same_run_producer_v2",
    ROOT / "scripts" / "swe_task_state_v4_counterfactual_stage_a_same_run_producer_v2.py",
)
VERIFY = load_module(
    "stage_a_same_run_verifier_v2",
    ROOT / "scripts" / "swe_task_state_v4_counterfactual_stage_a_same_run_verifier_v2.py",
)


class MockBackend:
    backend_kind = "injected_mock_same_run_backend"

    def __init__(self, *, mismatch_prompt: bool = False, mismatch_first: bool = False, stop_reason=None):
        self.implementation_binding = {"path": "tests/mock.py", "sha256": "a" * 64, "size_bytes": 1}
        self.model_instance_id = "b" * 64
        self.execution_provenance = {"mode": "injected_test_non_gate"}
        self.calls = 0
        self.mismatch_prompt = mismatch_prompt
        self.mismatch_first = mismatch_first
        self.stop_reason = stop_reason

    def run_same_autoregressive_request(self, *, request, sampling, capture):
        self.calls += 1
        hidden = 4
        layer_count = 4
        vocab = 8
        all_layer = np.arange(layer_count * hidden, dtype=np.float32).reshape(layer_count, hidden)
        raw = all_layer[[1, 2]].copy()
        public = raw + np.float32(0.25)
        final = np.arange(hidden, dtype=np.float32)
        logits = np.arange(vocab, dtype=np.float32)
        first = 6 if self.mismatch_first else 7
        submitted = list(request.token_ids)
        engine = submitted + [1] if self.mismatch_prompt else list(submitted)
        base = f"{request.index:064x}"[-64:]
        return {
            "request_id": base,
            "model_instance_id": self.model_instance_id,
            "prefill_forward_id": ("1" + base[1:]),
            "kv_cache_sequence_id": ("2" + base[1:]),
            "submitted_prompt_token_ids": submitted,
            "engine_returned_prompt_token_ids": engine,
            "prompt_tail_position_id": len(submitted) - 1,
            "prompt_tail_cache_write_index": len(submitted) - 1,
            "first_decode_cache_length": len(submitted),
            "capture_completed_monotonic_ns": 100 + request.index,
            "first_decode_started_monotonic_ns": 101 + request.index,
            "same_backend_call": True,
            "capture_was_online_in_prefill": True,
            "generation_consumed_same_kv_cache": True,
            "replay_between_capture_and_first_decode": False,
            "all_layer_residual": all_layer,
            "raw_residual": raw,
            "public_j_state": public,
            "prompt_tail_final_state": final,
            "prompt_tail_logits": logits,
            "generated_token_ids": [first] + [0] * 255,
            "completion_text": "mock completion",
            "finish_reason": "length",
            "stop_reason": self.stop_reason,
            "engine_sampling": dict(sampling),
            "final_norm_reconstruction_within_tolerance": True,
            "final_logits_reconstruction_within_tolerance": True,
        }


class UntouchableProductionBackend:
    """Any backend access proves authorization was checked too late."""

    calls = 0

    def __getattribute__(self, name):
        if name == "calls":
            return object.__getattribute__(self, name)
        raise AssertionError(f"unauthorized producer touched backend attribute {name}")


class SameRunV2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = VERIFY.load_and_validate_config()
        cls.bindings = VERIFY.verify_frozen_bindings(cls.config)
        cls.capture_rows, cls.generation_rows = VERIFY.load_production_prompt_rows(
            cls.config, cls.bindings
        )

    def tiny_config(self):
        return {
            "implementation": {"producer": PRODUCER.file_binding(PRODUCER.SCRIPT_PATH, display_root=ROOT)},
            "generation_prompt_count": 2,
            "model": {"vocabulary_size": 8, "hidden_size": 4, "layer_count": 4},
            "runtime": {"max_model_len": 512},
            "capture": {"layers": [1, 2]},
            "sampling": {
                "max_new_tokens": 256,
                "min_new_tokens": 256,
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "min_p": 0,
                "seed": 0,
                "ignore_eos": True,
                "stop": [],
                "stop_token_ids": [],
                "detokenize": True,
                "skip_special_tokens": True,
                "spaces_between_special_tokens": True,
                "repetition_penalty": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "truncate_prompt_tokens": None,
            },
        }

    @staticmethod
    def tiny_rows():
        return [
            {"id": "3" * 64, "token_ids": [1, 2, 3]},
            {"id": "4" * 64, "token_ids": [4, 5]},
        ]

    def test_production_contract_is_prospective_cpu_only_and_claims_false(self):
        self.assertFalse(self.config["authorization"]["real_gpu_producer_authorized_by_this_config"])
        self.assertFalse(self.config["authorization"]["real_backend_implementation_present"])
        self.assertFalse(self.config["authorization"]["real_traced_launcher_present"])
        self.assertFalse(self.config["authorization"]["raw_strace_normalizer_present"])
        self.assertFalse(self.config["authorization"]["raw_vs_normalized_trace_equivalence_established"])
        self.assertFalse(self.config["authorization"]["filesystem_alias_closure_established"])
        self.assertFalse(self.config["authorization"]["real_final_verification_receipt_emission_authorized"])
        self.assertIsNone(self.config["implementation"]["production_backend"])
        self.assertIsNone(self.config["implementation"]["real_traced_launcher"])
        self.assertIsNone(self.config["implementation"]["raw_strace_normalizer"])
        self.assertEqual(self.config["generation_prompt_count"], 240)
        self.assertEqual(self.config["sampling"]["max_new_tokens"], 256)
        self.assertEqual(self.config["sampling"]["min_new_tokens"], 256)
        self.assertTrue(self.config["sampling"]["ignore_eos"])
        self.assertFalse(self.config["runtime"]["enable_prefix_caching"])
        self.assertTrue(
            all(
                value is False
                for key, value in self.config["claim_scope"].items()
                if key != "cpu_contract_and_mock_tests_implemented"
            )
        )
        self.assertFalse(self.config["boundary_limits"]["one_boundary_can_establish_temporal_cot_trajectory"])
        self.assertFalse(self.config["boundary_limits"]["one_boundary_can_establish_private_or_verbatim_cot"])
        self.assertFalse(self.config["blocking_state"]["current_verifier_can_emit_a_real_final_verification_receipt"])

    def test_exact_240_generation_rows_are_opaque_and_fit_256_tokens(self):
        self.assertEqual((len(self.capture_rows), len(self.generation_rows)), (500, 240))
        self.assertEqual(min(len(row["token_ids"]) for row in self.generation_rows), 13397)
        self.assertEqual(max(len(row["token_ids"]) for row in self.generation_rows), 47413)
        self.assertLessEqual(47413 + 256, 49152)
        self.assertTrue(all(set(row) == {"id", "token_ids"} for row in self.generation_rows))

    def test_producer_exposes_one_indivisible_backend_call_and_no_gpu_cli(self):
        source = PRODUCER.SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertEqual(source.count("backend.run_same_autoregressive_request("), 1)
        self.assertNotIn("from vllm import", source)
        self.assertNotIn("import vllm", source)
        self.assertNotIn("LLM(", source)
        self.assertFalse(hasattr(PRODUCER, "main"))

    def test_mock_producer_writes_same_run_shards_but_is_irrevocably_non_gate(self):
        backend = MockBackend()
        with tempfile.TemporaryDirectory(prefix="jlens-same-run-mock-") as directory:
            root = Path(directory) / "out"
            result = PRODUCER.produce_same_run_bundle(
                config=self.tiny_config(),
                runtime_config_sha256="5" * 64,
                prompt_rows=self.tiny_rows(),
                backend=backend,
                output_root=root,
                preflight_receipt={"path": "p", "sha256": "6" * 64, "size_bytes": 1},
                capture_lock_receipt={"path": "c", "sha256": "7" * 64, "size_bytes": 1},
                backend_mode="injected_test_non_gate",
            )
            self.assertEqual(backend.calls, 2)
            self.assertEqual(result["status"], "injected_test_non_gate")
            self.assertFalse(result["gate_eligible"])
            self.assertEqual(result["aggregate"]["exact_generated_tokens_per_record"], 256)
            first = json.loads((root / "records" / "same-run-000000.json").read_text())
            self.assertTrue(first["prompt"]["exact_equal"])
            self.assertTrue(first["same_run"]["generation_consumed_same_kv_cache"])
            self.assertFalse(first["same_run"]["replay_between_capture_and_first_decode"])
            self.assertFalse(first["boundary_limits"]["one_prompt_tail_boundary_is_a_temporal_cot_trajectory"])

    def test_mock_prompt_echo_mismatch_fails_before_shard_write(self):
        with tempfile.TemporaryDirectory(prefix="jlens-same-run-bad-echo-") as directory:
            with self.assertRaisesRegex(PRODUCER.SameRunProducerError, "engine-returned"):
                PRODUCER.produce_same_run_bundle(
                    config=self.tiny_config(), runtime_config_sha256="5" * 64,
                    prompt_rows=self.tiny_rows(), backend=MockBackend(mismatch_prompt=True),
                    output_root=Path(directory) / "out", preflight_receipt={},
                    capture_lock_receipt={}, backend_mode="injected_test_non_gate",
                )

    def test_mock_first_token_or_stop_reason_mismatch_fails(self):
        for backend, pattern in (
            (MockBackend(mismatch_first=True), "first generated token"),
            (MockBackend(stop_reason="eos"), "exact 256-token length"),
        ):
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory(prefix="jlens-same-run-bad-") as directory:
                with self.assertRaisesRegex(PRODUCER.SameRunProducerError, pattern):
                    PRODUCER.produce_same_run_bundle(
                        config=self.tiny_config(), runtime_config_sha256="5" * 64,
                        prompt_rows=self.tiny_rows(), backend=backend,
                        output_root=Path(directory) / "out", preflight_receipt={},
                        capture_lock_receipt={}, backend_mode="injected_test_non_gate",
                    )

    def reference_report(self, prompt):
        generated = 7
        position = len(prompt["token_ids"]) - 1
        checkpoint = {
            "policy": "ModelOptCheckpoint(strict_pinned=True)",
            "validated_before_model_load": True,
            "validated_after_evaluation": True,
            "shards": {
                name: {"bytes": item["size_bytes"], "sha256": item["sha256"]}
                for name, item in self.config["model"]["checkpoint_shards"].items()
            },
        }
        experiment = {
            "id": prompt["id"],
            "prompt_token_ids": list(prompt["token_ids"]),
            "positions_requested": [-1],
            "positions_resolved": [position],
            "capture_positions_resolved": [position],
            "final_validation_position": position,
            "generated_token_id": generated,
            "final_layer_top1_matches_greedy": True,
            "final_model_readout": [{"target_token_id": generated, "token_ids": [generated]}],
            "captured_final_model_readout": [{"target_token_id": generated, "token_ids": [generated]}],
            "final_norm_reconstruction": {"within_tolerance": True},
            "final_logits_reconstruction": {"within_tolerance": True, "top_k_prefix_token_ids_match": True},
            "residual_capture_manifest": {
                "algorithm": "SHA-256 over length-prefixed canonical layer/shape/dtype/token-position/byte-count headers and logical row-major FP32 bytes",
                "logical_bytes": 64 * 5120 * 4,
                "sha256": "8" * 64,
                "tensor_count": 64,
                "token_positions": [position],
            },
        }
        return {
            "schema_version": 3,
            "score_encoding": "unrounded-float32",
            "status": "passed",
            "started_at": "2026-07-20T00:00:00+00:00",
            "completed_at": "2026-07-20T01:00:00+00:00",
            "elapsed_seconds": 3600,
            "host": {},
            "model": {
                "repo_id": self.config["model"]["repo_id"],
                "revision": self.config["model"]["revision"],
                "config_sha256": self.config["model"]["config_sha256"],
                "index_sha256": self.config["model"]["model_index_sha256"],
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "model_info": {"hidden_size": 5120, "layer_count": 64, "language_model_class": "Qwen3_5ForCausalLM"},
                "checkpoint_integrity": checkpoint,
            },
            "lens": {
                "repo_id": self.config["public_j_lens"]["repo_id"],
                "revision": self.config["public_j_lens"]["revision"],
                "filename": self.config["public_j_lens"]["filename"],
                "sha256": self.config["public_j_lens"]["sha256"],
            },
            "runtime": dict(
                self.config["reference_report_authentication"]["reference_runtime_exact"]
            ),
            "scored_vocabulary": {},
            "assertions": {
                "lens_hash_matches": True,
                "lens_metadata_matches": True,
                "model_architecture_matches": True,
                "all_final_layer_top1_match_greedy": True,
                "all_final_adapter_reconstructions_within_tolerance": True,
            },
            "experiments": [experiment],
        }

    def test_reference_report_is_content_authenticated_and_failed_or_mismatched_report_fails(self):
        prompt = {"id": "9" * 64, "token_ids": [1, 2, 3]}
        config = copy.deepcopy(self.config)
        config["reference_report_authentication"]["expected_experiment_count"] = 1
        report = self.reference_report(prompt)
        receipt = VERIFY.authenticate_reference_report_content(report, capture_rows=[prompt], config=config)
        self.assertTrue(receipt["content_authenticated_not_hash_only"])
        self.assertTrue(receipt["all_prompt_ids_and_token_ids_exact"])
        failed = copy.deepcopy(report)
        failed["status"] = "failed"
        with self.assertRaisesRegex(VERIFY.SameRunVerificationError, "schema/status"):
            VERIFY.authenticate_reference_report_content(failed, capture_rows=[prompt], config=config)
        mismatched = copy.deepcopy(report)
        mismatched["experiments"][0]["prompt_token_ids"][-1] = 4
        with self.assertRaisesRegex(VERIFY.SameRunVerificationError, "prompt/tail"):
            VERIFY.authenticate_reference_report_content(mismatched, capture_rows=[prompt], config=config)

    def test_trace_requires_allowlist_and_receipts_before_model_read(self):
        with tempfile.TemporaryDirectory(prefix="jlens-same-run-trace-") as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            model = root / "model"
            model.mkdir()
            model_file = model / "config.json"
            model_file.write_text("{}", encoding="utf-8")
            lens = root / "lens.pt"
            lens.write_bytes(b"lens")
            preflight = root / "preflight.json"
            preflight.write_text("{}", encoding="utf-8")
            capture = root / "capture.json"
            capture.write_text("{}", encoding="utf-8")
            raw = root / "raw.tar"
            raw.write_bytes(b"raw")
            trace = root / "trace.jsonl"
            header = {
                "schema_version": 2,
                "kind": "swe_task_state_v4_stage_a_same_run_filesystem_trace_header_v2",
                "tracer": self.config["filesystem_trace"]["tracer"],
                "producer_pid": 10,
                "capture_started_before_producer_exec": True,
                "all_descendants_traced": True,
                "lost_event_count": 0,
                "sanitized_environment": self.config["environment"]["exact_values"],
            }
            event_paths = [preflight, capture, model_file, lens]
            events = [
                {"sequence": index, "pid": 10, "operation": "read", "path": str(path), "result": "success"}
                for index, path in enumerate(event_paths)
            ]
            trace.write_text("".join(json.dumps(row) + "\n" for row in [header, *events]), encoding="utf-8")
            result = VERIFY.verify_filesystem_trace(
                config=self.config, trace_path=trace, raw_trace_archive_path=raw,
                output_root=output, exact_input_paths=[preflight, capture],
                model_snapshot_path=model, lens_path=lens,
                preflight_path=preflight, capture_lock_path=capture,
            )
            self.assertTrue(result["all_successful_reads_allowlisted"])
            self.assertTrue(result["receipts_read_before_model_or_lens"])
            bad_events = list(events)
            bad_events.append({"sequence": 4, "pid": 10, "operation": "read", "path": str(root / "stage-a-condition-key.json"), "result": "enoent"})
            bad = root / "bad-trace.jsonl"
            bad.write_text("".join(json.dumps(row) + "\n" for row in [header, *bad_events]), encoding="utf-8")
            with self.assertRaisesRegex(VERIFY.SameRunVerificationError, "forbidden trace path"):
                VERIFY.verify_filesystem_trace(
                    config=self.config, trace_path=bad, raw_trace_archive_path=raw,
                    output_root=output, exact_input_paths=[preflight, capture],
                    model_snapshot_path=model, lens_path=lens,
                    preflight_path=preflight, capture_lock_path=capture,
                )

    def test_mock_bundle_cannot_pass_production_verifier_even_if_perfect(self):
        result = {
            "schema_version": 2,
            "kind": "swe_task_state_v4_counterfactual_stage_a_same_run_producer_result_v2",
            "status": "injected_test_non_gate",
            "runtime_config_sha256": VERIFY.CONFIG_SHA256,
            "producer": self.config["implementation"]["producer"],
            "backend": {"mode": "injected_test_non_gate", "kind": "injected_mock_same_run_backend"},
            "execution_provenance": {"mode": "injected_test_non_gate"},
            "pre_generation_receipts": {}, "records": [], "completion_bundle": {},
            "aggregate": {}, "claims": PRODUCER.FALSE_CLAIMS, "boundary_limits": {},
            "gate_eligible": False, "reserved_validation_access_authorized": False,
        }
        with tempfile.TemporaryDirectory(prefix="jlens-same-run-nongate-") as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(VERIFY.SameRunVerificationError, "not authorized"):
                VERIFY.verify_producer_bundle(
                    config=self.config, generation_rows=self.generation_rows,
                    producer_result_path=path, output_root=Path(directory),
                    model_snapshot_path=Path(directory), preflight_path=path,
                    capture_lock_path=path,
                )

    def test_fake_production_backend_is_rejected_before_backend_or_output_access(self):
        backend = UntouchableProductionBackend()
        with tempfile.TemporaryDirectory(prefix="jlens-same-run-unauthorized-") as directory:
            output = Path(directory) / "must-not-exist"
            with self.assertRaisesRegex(
                PRODUCER.SameRunProducerError,
                "not authorized by this frozen config",
            ):
                PRODUCER.produce_same_run_bundle(
                    config=self.config,
                    runtime_config_sha256=VERIFY.CONFIG_SHA256,
                    prompt_rows=self.generation_rows,
                    backend=backend,
                    output_root=output,
                    preflight_receipt={},
                    capture_lock_receipt={},
                    backend_mode="real_local_pinned_vllm_online_capture",
                )
            self.assertFalse(output.exists())
            self.assertEqual(backend.calls, 0)

    def test_true_authorization_flags_without_exact_backend_launcher_and_normalizer_bindings_fail(self):
        mutated = copy.deepcopy(self.config)
        for key in (
            "real_gpu_producer_authorized_by_this_config",
            "real_backend_implementation_present",
            "real_traced_launcher_present",
            "raw_strace_normalizer_present",
            "raw_vs_normalized_trace_equivalence_established",
            "filesystem_alias_closure_established",
            "real_final_verification_receipt_emission_authorized",
        ):
            mutated["authorization"][key] = True
        with tempfile.TemporaryDirectory(prefix="jlens-same-run-unbound-") as directory:
            output = Path(directory) / "must-not-exist"
            with self.assertRaisesRegex(PRODUCER.SameRunProducerError, "new versioned producer code freeze"):
                PRODUCER.produce_same_run_bundle(
                    config=mutated,
                    runtime_config_sha256=VERIFY.CONFIG_SHA256,
                    prompt_rows=self.generation_rows,
                    backend=UntouchableProductionBackend(),
                    output_root=output,
                    preflight_receipt={},
                    capture_lock_receipt={},
                    backend_mode="real_local_pinned_vllm_online_capture",
                )
            self.assertFalse(output.exists())

    def test_even_complete_fake_bindings_require_new_producer_and_verifier_code_freezes(self):
        mutated = copy.deepcopy(self.config)
        for key in (
            "real_gpu_producer_authorized_by_this_config",
            "real_backend_implementation_present",
            "real_traced_launcher_present",
            "raw_strace_normalizer_present",
            "raw_vs_normalized_trace_equivalence_established",
            "filesystem_alias_closure_established",
            "real_final_verification_receipt_emission_authorized",
        ):
            mutated["authorization"][key] = True
        binding = {
            "path": "/tmp/jlens-v2-complete-looking-binding-must-not-be-read.py",
            "sha256": "e" * 64,
            "size_bytes": 123,
        }
        mutated["implementation"]["production_backend"] = binding
        mutated["implementation"]["real_traced_launcher"] = binding
        mutated["implementation"]["raw_strace_normalizer"] = binding
        with mock.patch.object(
            PRODUCER,
            "_verify_exact_file_binding",
            side_effect=AssertionError("producer binding/path access occurred"),
        ):
            with self.assertRaisesRegex(PRODUCER.SameRunProducerError, "new versioned producer code freeze"):
                PRODUCER._require_real_production_authorization(mutated)
        with mock.patch.object(
            VERIFY,
            "_verify_binding",
            side_effect=AssertionError("verifier binding/path access occurred"),
        ):
            with self.assertRaisesRegex(VERIFY.SameRunVerificationError, "new versioned verifier code freeze"):
                VERIFY._require_real_verification_authorization(mutated)

    def test_verifier_rejects_current_config_before_producer_artifact_access(self):
        fake_production_result = {
            "schema_version": 2,
            "kind": "swe_task_state_v4_counterfactual_stage_a_same_run_producer_result_v2",
            "status": "producer_outputs_complete_pending_external_trace_and_cpu_verification",
            "runtime_config_sha256": VERIFY.CONFIG_SHA256,
            "gate_eligible": True,
            "backend": {
                "mode": "real_local_pinned_vllm_online_capture",
                "kind": "real_local_pinned_vllm_online_capture",
                "implementation": {"path": "missing.py", "sha256": "f" * 64, "size_bytes": 1},
            },
        }
        with tempfile.TemporaryDirectory(prefix="jlens-v2-fake-result-") as directory:
            root = Path(directory)
            result_path = root / "production-shaped-result.json"
            result_path.write_text(json.dumps(fake_production_result), encoding="utf-8")
            with mock.patch.object(VERIFY, "load_json", side_effect=AssertionError("producer artifact read")):
                with self.assertRaisesRegex(VERIFY.SameRunVerificationError, "not authorized"):
                    VERIFY.verify_producer_bundle(
                        config=self.config,
                        generation_rows=self.generation_rows,
                        producer_result_path=result_path,
                        output_root=root / "output-must-not-be-read",
                        model_snapshot_path=root / "model-must-not-be-read",
                        preflight_path=root / "preflight-must-not-be-read",
                        capture_lock_path=root / "lock-must-not-be-read",
                    )

    def test_run_verify_rejects_before_frozen_or_output_artifact_access(self):
        args = argparse.Namespace(
            config=VERIFY.CONFIG_PATH,
            preflight=Path("/tmp/jlens-v2-preflight-no-read"),
            capture_lock=Path("/tmp/jlens-v2-lock-no-read"),
            producer_result=Path("/tmp/jlens-v2-result-no-read"),
            producer_output_root=Path("/tmp/jlens-v2-output-no-read"),
            normalized_trace=Path("/tmp/jlens-v2-trace-no-read"),
            raw_trace_archive=Path("/tmp/jlens-v2-raw-no-read"),
            model_snapshot=Path("/tmp/jlens-v2-model-no-read"),
            lens=Path("/tmp/jlens-v2-lens-no-read"),
            output=Path("/tmp/jlens-v2-final-no-write"),
        )
        with mock.patch.object(
            VERIFY,
            "verify_frozen_bindings",
            side_effect=AssertionError("post-authorization artifact access"),
        ):
            with self.assertRaisesRegex(VERIFY.SameRunVerificationError, "not authorized"):
                VERIFY.run_verify(args)
        self.assertFalse(args.output.exists())

    def test_schemas_are_strict_and_bind_exact_256_and_one_boundary_false(self):
        shard = json.loads((ROOT / "configs" / "swe_task_state_v4_counterfactual_stage_a_same_run_shard_schema_v2.json").read_text())
        completion = json.loads((ROOT / "configs" / "swe_task_state_v4_counterfactual_stage_a_same_run_completion_bundle_schema_v2.json").read_text())
        self.assertFalse(shard["additionalProperties"])
        self.assertEqual(shard["properties"]["generation"]["properties"]["token_ids"]["minItems"], 256)
        self.assertEqual(shard["properties"]["generation"]["properties"]["token_ids"]["maxItems"], 256)
        self.assertFalse(shard["properties"]["boundary_limits"]["properties"]["one_prompt_tail_boundary_is_a_temporal_cot_trajectory"]["const"])
        self.assertEqual(completion["properties"]["records"]["minItems"], 240)
        self.assertEqual(completion["properties"]["records"]["maxItems"], 240)

    def test_forbidden_paths_fail_before_resolve_and_config_mutation_fails_closed(self):
        with mock.patch.object(Path, "resolve", side_effect=AssertionError("resolve called")):
            with self.assertRaises(VERIFY.SameRunVerificationError):
                VERIFY.lexical_path_preflight((Path("/tmp/validation/data.json"),))
        mutated = copy.deepcopy(self.config)
        mutated["claim_scope"]["cot_or_cot_like_decoding_established"] = True
        with self.assertRaisesRegex(VERIFY.SameRunVerificationError, "config object"):
            VERIFY.validate_config(mutated)

    def test_atomic_receipts_refuse_clobber(self):
        with tempfile.TemporaryDirectory(prefix="jlens-same-run-lock-") as directory:
            path = Path(directory) / "receipt.json"
            VERIFY._atomic_json(path, {"x": 1})
            with self.assertRaisesRegex(VERIFY.SameRunVerificationError, "overwrite"):
                VERIFY._atomic_json(path, {"x": 2})


if __name__ == "__main__":
    unittest.main()
