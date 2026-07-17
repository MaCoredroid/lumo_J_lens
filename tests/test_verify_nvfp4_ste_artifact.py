#!/usr/bin/env python3
"""Focused tests for native NVFP4/FP8-STE artifact verification."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_nvfp4_ste_artifact",
    ROOT / "scripts" / "verify_nvfp4_ste_artifact.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@unittest.skipIf(torch is None, "torch is required for checkpoint verification")
class VerifyNativeArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.d_model = 4
        self.layers = (0, 2)
        self.n_prompts = 2
        self.io_rows = 2
        self.prompt_manifest_sha256 = "a" * 64
        source = self.root / "fit.py"
        source.write_text("frozen source\n", encoding="utf-8")
        self.source_files = [
            {
                "path": "fit.py",
                "bytes": source.stat().st_size,
                "sha256": MODULE.sha256_file(source),
            }
        ]
        self.source_files_sha256 = MODULE.canonical_sha256(self.source_files)
        self.tensors = {
            0: torch.arange(16, dtype=torch.float32).reshape(4, 4),
            2: torch.arange(16, dtype=torch.float32).reshape(4, 4) + 2,
        }
        self.checkpoint = self.root / "lens.pt"
        self.metadata_path = self.root / "metadata.json"
        self.state_path = self.root / "state.json"
        self.metadata, self.contract_sha256, self.prompt_entries_sha256 = (
            self.build_metadata(self.tensors)
        )
        self.write_checkpoint(self.tensors)
        self.write_metadata(self.metadata)
        self.state = self.build_state()
        self.write_state(self.state)
        self.state_sha256 = MODULE.sha256_file(self.state_path)

    @staticmethod
    def tensor_sha256(tensor) -> str:
        return hashlib.sha256(
            tensor.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()

    def prompt_entries(self):
        entries = []
        for index in range(self.n_prompts):
            token_ids = list(range(index, index + MODULE.TOKEN_COUNT))
            entries.append(
                {
                    "manifest_index": index,
                    "token_count": MODULE.TOKEN_COUNT,
                    "token_ids": token_ids,
                    "text_sha256": f"{index + 1:064x}",
                    "token_ids_sha256": f"{index + 11:064x}",
                }
            )
        return entries

    @staticmethod
    def file_record(path: str, seed: int) -> dict[str, object]:
        return {"path": path, "bytes": seed + 1, "sha256": f"{seed + 1:064x}"}

    def capture_binding(self, index: int, entry: dict[str, object]):
        model_identity = {
            "policy": MODULE.MODEL_IDENTITY_POLICY,
            "repo_id": MODULE.MODEL_REPO,
            "revision": MODULE.MODEL_REVISION,
            "metadata_sha256": MODULE.MODEL_METADATA_SHA256,
            "resolved_path": (
                "/cache/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/"
                + MODULE.MODEL_REVISION
            ),
            "strict_pinned_validation": True,
            "validator": "ModelOptCheckpoint(strict_pinned=True)",
        }
        observer_scope = {
            "mode": "compiled-observer",
            "compiled_observer": True,
            "target_profile": "all",
            "target_layers": list(MODULE.CAPTURE_LAYERS),
            "compiled_observer_patch": {
                "capacity": MODULE.TOKEN_COUNT,
                "custom_op": MODULE.OBSERVER_CUSTOM_OP,
                "patched_linear_classes": list(MODULE.OBSERVER_LINEAR_CLASSES),
                "post_output_only": True,
                "target_layers": list(MODULE.CAPTURE_LAYERS),
            },
            "install": {
                "allocated_tensor_count": MODULE.EXPECTED_CAPTURE_TENSORS,
                "capture_capacity": MODULE.TOKEN_COUNT,
                "full_attention_layers": list(MODULE.FULL_ATTENTION_LAYERS),
                "gdn_layers": list(MODULE.GDN_LAYERS),
                "linear_boundary_count": MODULE.EXPECTED_LINEAR_BOUNDARIES,
                "target_layers": list(MODULE.CAPTURE_LAYERS),
            },
        }
        generation_record = {
            "finish_reason": "length",
            "generated_text": "x",
            "generated_token_id": 1,
            "prompt_token_ids": entry["token_ids"],
            "top_logprobs": [],
        }
        generation_sha256 = MODULE.canonical_sha256(generation_record)
        proof_claim = {
            "claim": {
                "scope": (
                    "exact compiled main-model prefill for the pinned prompt and runtime shape"
                ),
                "mtp": "off",
                "observer_graph_modified": True,
                "observer_modification_discharged": True,
                "discharge_basis": [
                    "full generation-record equality with an isolated uninstrumented compiled baseline",
                    "bit-exact parity for all 688 shared internal tensors",
                    "shape/dtype/content-hash parity for all 785 replay parameters",
                ],
                "not_claimed": [],
            },
            "generation_record_parity": {
                "exact": True,
                "record": generation_record,
                "record_sha256": generation_sha256,
                "records_compared": [
                    "baseline.authoritative_compiled_generation",
                    "baseline.instrumented_generation",
                    "observer.instrumented_generation",
                ],
            },
            "shared_internal_tensor_parity": {
                "all_shared_bit_exact": True,
                "baseline_only_tensor_count": 0,
                "baseline_tensor_count": MODULE.EXPECTED_SHARED_TENSORS,
                "shared_tensor_count": MODULE.EXPECTED_SHARED_TENSORS,
                "observer_only_tensor_count": MODULE.EXPECTED_OBSERVER_ONLY_TENSORS,
                "observer_tensor_count": MODULE.EXPECTED_OBSERVER_TENSORS,
                "shared_logical_bytes": MODULE.EXPECTED_SHARED_LOGICAL_BYTES,
                "mismatches": [],
                "shared_names_sha256": "1" * 64,
                "shared_tensor_manifest_sha256": "2" * 64,
            },
            "replay_parameter_parity": {
                "all_content_hashes_equal": True,
                "all_dtypes_equal": True,
                "all_names_equal": True,
                "all_shapes_equal": True,
                "json_provenance_equal": True,
                "parameter_count": MODULE.EXPECTED_REPLAY_PARAMETERS,
                "logical_bytes": MODULE.EXPECTED_PARAMETER_LOGICAL_BYTES,
                "mismatches": [],
                "parameter_manifest_sha256": "3" * 64,
                "parameter_names_sha256": "4" * 64,
            },
            "observer_capture_completeness": {
                "compile_visible_names_equal_observer_only_names": True,
                "compile_visible_observer_tensor_count": (
                    MODULE.EXPECTED_OBSERVER_ONLY_TENSORS
                ),
                "required_missing": [],
                "truncated": [],
            },
        }
        return {
            "schema_version": 1,
            "prompt_index": index,
            "capture_contract_sha256": f"{index + 40:064x}",
            "capture_state_path": f"/captures/prompt-{index:02d}-capture-state.json",
            "baseline_json": self.file_record(f"/captures/{index}-baseline.json", 50),
            "baseline_tensors_deleted_record": self.file_record(
                f"/captures/{index}-baseline.pt", 51
            ),
            "observer_json": self.file_record(f"/captures/{index}-observer.json", 52),
            "observer_tensors": self.file_record(f"/captures/{index}-observer.pt", 53),
            "proof": self.file_record(f"/captures/{index}-proof.json", 54),
            "observer_scope": observer_scope,
            "observer_scope_sha256": MODULE.canonical_sha256(observer_scope),
            "proof_claim": proof_claim,
            "proof_claim_sha256": MODULE.canonical_sha256(proof_claim),
            "model_identity": model_identity,
            "model_identity_sha256": MODULE.canonical_sha256(model_identity),
            "baseline_generation_sha256": generation_sha256,
        }

    def progress_chunks(self, index: int) -> list[dict[str, object]]:
        result = []
        for chunk_index, start in enumerate(range(0, self.d_model, 2)):
            result.append(
                {
                    "start": start,
                    "stop": min(start + 2, self.d_model),
                    "sha256": f"{index * 10 + chunk_index + 60:064x}",
                    "elapsed_seconds": 1.0,
                    "cuda": {
                        "allocated_bytes": 1,
                        "reserved_bytes": 2,
                        "peak_allocated_bytes": 3,
                        "peak_reserved_bytes": 4,
                    },
                }
            )
        return result

    def build_metadata(self, tensors):
        entries = self.prompt_entries()
        entries_sha256 = MODULE.canonical_sha256(entries)
        layout = MODULE._expected_layout(
            d_model=self.d_model,
            source_layers=self.layers,
            n_prompts=self.n_prompts,
            io_rows=self.io_rows,
        )
        contract = {
            "model": {
                "id": MODULE.MODEL_REPO,
                "revision": MODULE.MODEL_REVISION,
                "checkpoint_integrity_before_each_prompt_commit": True,
                "checkpoint_files": {
                    "metadata_sha256": MODULE.MODEL_METADATA_SHA256,
                    "shards": MODULE.MODEL_SHARDS,
                },
            },
            "estimator": {
                "name": MODULE.FIT_ESTIMATOR_LABEL,
                "hidden_size": self.d_model,
                "source_layers": list(self.layers),
                "target_layer": 3,
                "decoder_layers": 4,
                "prompt_count": self.n_prompts,
                "token_count": MODULE.TOKEN_COUNT,
                "skip_first": MODULE.SKIP_FIRST,
                "mean_over_source_positions": True,
                "is_grads_batched": True,
                "cotangent_batch": 2,
                "checkpoint_interval": 2,
            },
            "storage": layout,
            "surrogate_backward": {
                "activation_ste_policy": "identity",
                "literal_rounding_derivative": False,
                "clipped_ste_supported": False,
            },
            "prompts": {
                "entries": entries,
                "entries_sha256": entries_sha256,
                "manifest": {"sha256": self.prompt_manifest_sha256},
            },
            "source_files": self.source_files,
            "source_files_sha256": self.source_files_sha256,
        }
        contract_sha256 = MODULE.canonical_sha256(contract)
        committed = []
        progress_prompts = {}
        sum_records = [
            {
                "layer": layer,
                "filename": f"layer-{layer:02d}.f32",
                "shape": [self.d_model, self.d_model],
                "dtype": "little-endian-float32",
                "size": self.d_model * self.d_model * 4,
                "sha256": f"{layer + 100:064x}",
            }
            for layer in self.layers
        ]
        self.sum_records = sum_records
        final_sum_sha256 = MODULE.canonical_sha256(sum_records)
        for index, entry in enumerate(entries):
            capture = self.capture_binding(index, entry)
            prompt = {
                "manifest_prompt": entry,
                "capture": capture,
                "fit": {
                    "source_layers": list(self.layers),
                    "target_layer": 3,
                    "skip_first": MODULE.SKIP_FIRST,
                    "cotangent_batch": 2,
                    "checkpoint_interval": 2,
                    "ste_policy": "identity",
                },
            }
            chunks = self.progress_chunks(index)
            authoritative_chunks = [
                {key: chunk[key] for key in ("start", "stop", "sha256")}
                for chunk in chunks
            ]
            commit = {
                "prompt_index": index,
                "prompt": prompt,
                "prompt_sha256": MODULE.canonical_sha256(prompt),
                "chunk_count": 2,
                "chunk_manifest_sha256": MODULE.canonical_sha256(
                    authoritative_chunks
                ),
                "sum_generation": index + 1,
                "sum_aggregate_sha256": (
                    final_sum_sha256 if index == len(entries) - 1 else f"{index + 31:064x}"
                ),
                "committed_at": f"2026-07-17T00:00:0{index}+00:00",
            }
            committed.append(commit)
            progress_prompts[str(index)] = {
                "capture_binding": capture,
                "capture_invocations": [
                    {
                        "argv": ["python", "capture.py", "--prompt-index", str(index)],
                        "resume_used": False,
                        "completed_at": f"2026-07-17T00:00:0{index}+00:00",
                        "stdout_sha256": "5" * 64,
                        "stderr_sha256": "6" * 64,
                        "capture_state": self.file_record(
                            f"/captures/{index}-state.json", 55
                        ),
                    }
                ],
                "chunks": chunks,
                "commit": commit,
            }
        progress = {
            "schema_version": 1,
            "contract_sha256": contract_sha256,
            "prompts": progress_prompts,
            "max_cuda_peak_allocated_bytes": 123,
            "max_cuda_peak_reserved_bytes": 456,
        }
        payload = {
            "schema_version": 1,
            "fit_type": MODULE.FIT_TYPE,
            "contract": contract,
            "contract_sha256": contract_sha256,
            "progress": progress,
            "progress_sha256": MODULE.canonical_sha256(progress),
            "committed_prompts": committed,
            "committed_prompts_sha256": MODULE.canonical_sha256(committed),
            "disclosure": {
                "forward": "exact deployed compiled NVFP4/FP8 observer capture",
                "backward": "packed W4 and live FP8 declared surrogate VJPs",
                "literal_derivative_of_quantized_rounding": False,
            },
        }
        layer_records = [
            {
                "layer": layer,
                "filename": f"layer-{layer:02d}.f32",
                "shape": [self.d_model, self.d_model],
                "dtype": "little-endian-float32",
                "size": self.d_model * self.d_model * 4,
                "sha256": self.tensor_sha256(tensors[layer]),
            }
            for layer in self.layers
        ]
        metadata = {
            "schema_version": 1,
            "artifact_type": "lumo-jlens-dense-fp32-means",
            "created_at": "2026-07-17T00:00:00+00:00",
            "run_id": "test-run",
            "contract_sha256": contract_sha256,
            "layout": layout,
            "n_prompts": self.n_prompts,
            "averaging": "arithmetic mean of cumulative little-endian FP32 sums",
            "layers": layer_records,
            "layer_aggregate_sha256": MODULE.canonical_sha256(layer_records),
            "committed_prompts_sha256": MODULE.canonical_sha256(committed),
            "metadata": payload,
            "metadata_sha256": MODULE.canonical_sha256(payload),
        }
        return metadata, contract_sha256, entries_sha256

    def build_state(self):
        metadata_sha256 = MODULE.sha256_file(self.metadata_path)
        committed = self.metadata["metadata"]["committed_prompts"]
        return {
            "schema_version": 1,
            "status": "completed",
            "run_id": self.metadata["run_id"],
            "contract": self.metadata["metadata"]["contract"],
            "contract_sha256": self.contract_sha256,
            "layout": self.metadata["layout"],
            "prompt_count": self.n_prompts,
            "n_done": self.n_prompts,
            "next_prompt": self.n_prompts,
            "sum_generation": self.n_prompts,
            "sum_integrity": {
                "generation": self.n_prompts,
                "layers": self.sum_records,
                "aggregate_sha256": MODULE.canonical_sha256(self.sum_records),
            },
            "committed_prompts": committed,
            "current": None,
            "final_artifact": {
                "directory": "final-mean",
                "metadata_path": "final-mean/metadata.json",
                "metadata_sha256": metadata_sha256,
                "layer_aggregate_sha256": self.metadata["layer_aggregate_sha256"],
                "metadata_payload_sha256": self.metadata["metadata_sha256"],
                "n_prompts": self.n_prompts,
            },
            "started_at": "2026-07-17T00:00:00+00:00",
            "updated_at": "2026-07-17T00:10:00+00:00",
        }

    def write_checkpoint(self, tensors) -> None:
        torch.save(
            {
                "J": tensors,
                "d_model": self.d_model,
                "n_prompts": self.n_prompts,
                "source_layers": list(self.layers),
            },
            self.checkpoint,
        )

    def write_metadata(self, metadata) -> None:
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_state(self, state) -> None:
        self.state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def rewrite_metadata_and_state(self, metadata) -> None:
        payload = metadata["metadata"]
        committed = payload["committed_prompts"]
        for index, commit in enumerate(committed):
            commit["prompt_sha256"] = MODULE.canonical_sha256(commit["prompt"])
            progress = payload["progress"]["prompts"][str(index)]
            progress["commit"] = copy.deepcopy(commit)
            progress["capture_binding"] = copy.deepcopy(commit["prompt"]["capture"])
        committed_sha256 = MODULE.canonical_sha256(committed)
        payload["committed_prompts_sha256"] = committed_sha256
        payload["progress_sha256"] = MODULE.canonical_sha256(payload["progress"])
        metadata["committed_prompts_sha256"] = committed_sha256
        metadata["metadata_sha256"] = MODULE.canonical_sha256(payload)
        self.metadata = metadata
        self.write_metadata(metadata)
        self.state = self.build_state()
        self.write_state(self.state)
        self.state_sha256 = MODULE.sha256_file(self.state_path)

    def verify(self, *, expected_sha256=None, expected_state_sha256=None):
        return MODULE.verify_nvfp4_ste_artifact(
            self.checkpoint,
            expected_sha256=expected_sha256
            or MODULE.sha256_file(self.checkpoint),
            provenance_path=self.metadata_path,
            state_path=self.state_path,
            expected_state_sha256=expected_state_sha256 or self.state_sha256,
            d_model=self.d_model,
            source_layers=self.layers,
            target_layer=3,
            expected_n_prompts=self.n_prompts,
            io_rows=self.io_rows,
            expected_contract_sha256=self.contract_sha256,
            expected_source_files_sha256=self.source_files_sha256,
            expected_prompt_manifest_sha256=self.prompt_manifest_sha256,
            expected_prompt_entries_sha256=self.prompt_entries_sha256,
            source_root=self.root,
        )

    def test_valid_export_is_bound_to_final_metadata_and_sources(self) -> None:
        result = self.verify()
        self.assertEqual(result["kind"], "native_nvfp4_ste_fit")
        self.assertEqual(result["fit_model"], MODULE.MODEL_REPO)
        self.assertEqual(result["source_layers"], list(self.layers))
        self.assertTrue(result["finite_checked"])
        json.dumps(result, allow_nan=False)
        self.assertNotIn("fd_path", result)

    def test_tensor_must_match_final_mean_layer_hash(self) -> None:
        altered = copy.deepcopy(self.tensors)
        altered[0] = altered[0].clone()
        altered[0][0, 0] += 1
        self.write_checkpoint(altered)
        with self.assertRaisesRegex(ValueError, "differs from final-mean"):
            self.verify()

    def test_metadata_payload_hash_tampering_is_rejected(self) -> None:
        altered = copy.deepcopy(self.metadata)
        altered["metadata"]["disclosure"][
            "literal_derivative_of_quantized_rounding"
        ] = True
        self.write_metadata(altered)
        with self.assertRaisesRegex(ValueError, "provenance payload hash"):
            self.verify()

    def test_frozen_source_drift_is_rejected(self) -> None:
        (self.root / "fit.py").write_text("changed source\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "size mismatch|SHA-256 mismatch"):
            self.verify()

    def test_checkpoint_and_metadata_symlinks_are_rejected(self) -> None:
        checkpoint_link = self.root / "lens-link.pt"
        checkpoint_link.symlink_to(self.checkpoint)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            MODULE.verify_nvfp4_ste_artifact(
                checkpoint_link,
                expected_sha256=MODULE.sha256_file(self.checkpoint),
                provenance_path=self.metadata_path,
                state_path=self.state_path,
                expected_state_sha256=self.state_sha256,
            )

        metadata_link = self.root / "metadata-link.json"
        metadata_link.symlink_to(self.metadata_path)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            MODULE.verify_nvfp4_ste_artifact(
                self.checkpoint,
                expected_sha256=MODULE.sha256_file(self.checkpoint),
                provenance_path=metadata_link,
                state_path=self.state_path,
                expected_state_sha256=self.state_sha256,
            )

        state_link = self.root / "state-link.json"
        state_link.symlink_to(self.state_path)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            MODULE.verify_nvfp4_ste_artifact(
                self.checkpoint,
                expected_sha256=MODULE.sha256_file(self.checkpoint),
                provenance_path=self.metadata_path,
                state_path=state_link,
                expected_state_sha256=self.state_sha256,
            )

    def test_incomplete_fit_state_is_rejected_even_when_expected_hash_matches(self) -> None:
        state = copy.deepcopy(self.state)
        state["status"] = "running"
        self.write_state(state)
        self.state_sha256 = MODULE.sha256_file(self.state_path)
        with self.assertRaisesRegex(ValueError, "not completed"):
            self.verify()

    def test_expected_fit_state_hash_is_mandatory(self) -> None:
        with self.assertRaisesRegex(ValueError, "fit-state SHA-256 mismatch"):
            self.verify(expected_state_sha256="f" * 64)

    def test_final_artifact_state_binding_is_rejected_when_rewritten(self) -> None:
        state = copy.deepcopy(self.state)
        state["final_artifact"]["metadata_sha256"] = "f" * 64
        self.write_state(state)
        self.state_sha256 = MODULE.sha256_file(self.state_path)
        with self.assertRaisesRegex(ValueError, "final fit-state binding"):
            self.verify()

    def test_failed_capture_proof_is_rejected_when_self_hashes_are_rewritten(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        capture = metadata["metadata"]["committed_prompts"][0]["prompt"][
            "capture"
        ]
        capture["proof_claim"]["shared_internal_tensor_parity"][
            "all_shared_bit_exact"
        ] = False
        capture["proof_claim_sha256"] = MODULE.canonical_sha256(
            capture["proof_claim"]
        )
        self.rewrite_metadata_and_state(metadata)
        with self.assertRaisesRegex(ValueError, "shared tensor proof"):
            self.verify()

    def test_noncontiguous_chunks_are_rejected_when_self_hashes_are_rewritten(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        metadata["metadata"]["progress"]["prompts"]["0"]["chunks"][1][
            "start"
        ] = 1
        self.rewrite_metadata_and_state(metadata)
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            self.verify()

    def test_fit_config_drift_is_rejected_when_self_hashes_are_rewritten(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        metadata["metadata"]["committed_prompts"][0]["prompt"]["fit"][
            "ste_policy"
        ] = "clipped"
        self.rewrite_metadata_and_state(metadata)
        with self.assertRaisesRegex(ValueError, "fit config"):
            self.verify()

    def test_held_descriptor_survives_path_replacement(self) -> None:
        original_sha256 = MODULE.sha256_file(self.checkpoint)
        replacement = self.root / "replacement.pt"
        altered = copy.deepcopy(self.tensors)
        altered[0] = altered[0] + 100
        torch.save(
            {
                "J": altered,
                "d_model": self.d_model,
                "n_prompts": self.n_prompts,
                "source_layers": list(self.layers),
            },
            replacement,
        )
        with MODULE.open_verified_nvfp4_ste_artifact(
            self.checkpoint,
            expected_sha256=original_sha256,
            provenance_path=self.metadata_path,
            state_path=self.state_path,
            expected_state_sha256=self.state_sha256,
            d_model=self.d_model,
            source_layers=self.layers,
            target_layer=3,
            expected_n_prompts=self.n_prompts,
            io_rows=self.io_rows,
            expected_contract_sha256=self.contract_sha256,
            expected_source_files_sha256=self.source_files_sha256,
            expected_prompt_manifest_sha256=self.prompt_manifest_sha256,
            expected_prompt_entries_sha256=self.prompt_entries_sha256,
            source_root=self.root,
        ) as artifact:
            replacement.replace(self.checkpoint)
            held = torch.load(
                artifact.fd_path, map_location="cpu", weights_only=True, mmap=True
            )
            self.assertTrue(torch.equal(held["J"][0], self.tensors[0]))
            self.assertEqual(artifact.record["sha256"], original_sha256)
            with self.assertRaisesRegex(
                ValueError, "changed while it was hashed|content changed"
            ):
                artifact.require_unchanged()

    def test_in_place_mutation_is_rejected_with_restored_mtime(self) -> None:
        original_sha256 = MODULE.sha256_file(self.checkpoint)
        before = self.checkpoint.stat()
        with MODULE.open_verified_nvfp4_ste_artifact(
            self.checkpoint,
            expected_sha256=original_sha256,
            provenance_path=self.metadata_path,
            state_path=self.state_path,
            expected_state_sha256=self.state_sha256,
            d_model=self.d_model,
            source_layers=self.layers,
            target_layer=3,
            expected_n_prompts=self.n_prompts,
            io_rows=self.io_rows,
            expected_contract_sha256=self.contract_sha256,
            expected_source_files_sha256=self.source_files_sha256,
            expected_prompt_manifest_sha256=self.prompt_manifest_sha256,
            expected_prompt_entries_sha256=self.prompt_entries_sha256,
            source_root=self.root,
        ) as artifact:
            with self.checkpoint.open("r+b") as handle:
                original = handle.read(1)
                self.assertEqual(len(original), 1)
                handle.seek(0)
                handle.write(bytes([original[0] ^ 1]))
                handle.flush()
                os.fsync(handle.fileno())
            os.utime(
                self.checkpoint,
                ns=(before.st_atime_ns, before.st_mtime_ns),
            )
            with self.assertRaisesRegex(
                ValueError, "changed while it was hashed|content changed"
            ):
                artifact.require_unchanged()

    def test_nonfinite_export_is_rejected_even_when_hashes_are_rewritten(self) -> None:
        altered = copy.deepcopy(self.tensors)
        altered[2] = altered[2].clone()
        altered[2][0, 0] = float("nan")
        metadata, contract_sha256, entries_sha256 = self.build_metadata(altered)
        self.metadata = metadata
        self.contract_sha256 = contract_sha256
        self.prompt_entries_sha256 = entries_sha256
        self.write_metadata(metadata)
        self.write_checkpoint(altered)
        self.state = self.build_state()
        self.write_state(self.state)
        self.state_sha256 = MODULE.sha256_file(self.state_path)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
