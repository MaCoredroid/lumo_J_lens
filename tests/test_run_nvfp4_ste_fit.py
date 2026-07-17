from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_nvfp4_ste_fit", ROOT / "scripts" / "run_nvfp4_ste_fit.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeFactory:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.validated: list[int] = []

    def validate_layer(self, layer: int) -> None:
        self.validated.append(layer)


class FakeCheckpoint:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.integrity_calls = 0

    def validate_pinned_integrity(self) -> dict[str, int]:
        self.integrity_calls += 1
        if self.error is not None:
            raise self.error
        return {"integrity_call": self.integrity_calls}


class FakeRuntime:
    def __init__(
        self,
        root: Path,
        prompts: list[dict],
        spec: MODULE.FitRunSpec,
        *,
        fail_compute_call: int | None = None,
        mutate_observer_compute_call: int | None = None,
        checkpoint_error: Exception | None = None,
    ) -> None:
        self.root = root
        self.prompts = prompts
        self.spec = spec
        self.fail_compute_call = fail_compute_call
        self.mutate_observer_compute_call = mutate_observer_compute_call
        self.checkpoint = FakeCheckpoint(checkpoint_error)
        self.compute_calls: list[tuple[int, int, int]] = []
        self.capture_calls: list[int] = []
        self.release_count = 0
        self.clock = 0.0
        self.observer_paths: list[Path] = []
        root.mkdir(parents=True, exist_ok=True)
        for index in range(spec.prompt_count):
            path = root / f"observer-{index}.pt"
            path.write_bytes(f"observer-{index}".encode("ascii"))
            self.observer_paths.append(path)

    def capture(self, request: MODULE.CaptureRequest) -> MODULE.CaptureResult:
        self.capture_calls.append(request.prompt_index)
        path = self.observer_paths[request.prompt_index]
        return MODULE.CaptureResult(
            binding={
                "schema_version": 1,
                "prompt_index": request.prompt_index,
                "observer_tensors": MODULE._file_record(path),
                "proof_claim_sha256": "a" * 64,
            },
            invocation={
                "prompt_index": request.prompt_index,
                "invocation_number": len(self.capture_calls),
            },
        )

    def load_observer(self, path: Path) -> dict:
        index = self.observer_paths.index(path)
        tensors = {
            f"h{layer}_post_block": np.zeros(
                (self.spec.token_count, self.spec.hidden_size), dtype=np.float32
            )
            for layer in range(self.spec.decoder_layers)
        }
        return {
            "schema_version": 1,
            "mode": "compiled-observer",
            "model_revision": MODULE.MODEL_REVISION,
            "model_identity": {
                "policy": MODULE.MODEL_IDENTITY_POLICY,
                "repo_id": MODULE.MODEL_ID,
                "revision": MODULE.MODEL_REVISION,
                "resolved_path": str(
                    self.root.parent
                    / "models--nvidia--Qwen3.6-27B-NVFP4"
                    / "snapshots"
                    / MODULE.MODEL_REVISION
                ),
                "metadata_sha256": MODULE._capture_module().PINNED_METADATA_SHA256,
                "strict_pinned_validation": True,
                "validator": "ModelOptCheckpoint(strict_pinned=True)",
            },
            "prompt": self.prompts[index]["text"],
            "prompt_token_ids": self.prompts[index]["token_ids"],
            "target_profile": "all",
            "target_layers": list(self.spec.capture_layers),
            "prompt_index": index,
            "tensors": tensors,
        }

    def make_factory(
        self,
        payload: dict,
        _checkpoint: object,
        _ste_policy: str,
        _checkpoint_interval: int,
    ) -> FakeFactory:
        return FakeFactory(payload)

    def compute_rows(
        self,
        factory: FakeFactory,
        valid_positions: tuple[int, ...],
        row_start: int,
        row_stop: int,
        target_layer: int,
        device: str,
    ) -> dict[int, np.ndarray]:
        index = factory.payload["prompt_index"]
        self.compute_calls.append((index, row_start, row_stop))
        if (
            self.fail_compute_call is not None
            and len(self.compute_calls) == self.fail_compute_call
        ):
            raise RuntimeError("simulated replay crash")
        assert valid_positions == tuple(
            range(self.spec.skip_first, self.spec.token_count - 1)
        )
        assert target_layer == self.spec.target_layer
        assert device == "fake:0"
        rows: dict[int, np.ndarray] = {}
        for layer in self.spec.source_layers:
            value = np.empty(
                (row_stop - row_start, self.spec.hidden_size), dtype=np.float32
            )
            for offset, row in enumerate(range(row_start, row_stop)):
                value[offset] = index * 10 + layer * 100 + row
            rows[layer] = value
        if (
            self.mutate_observer_compute_call is not None
            and len(self.compute_calls) == self.mutate_observer_compute_call
        ):
            self.observer_paths[index].write_bytes(b"mutated observer")
        return rows

    def monotonic(self) -> float:
        self.clock += 0.25
        return self.clock

    def dependencies(self) -> MODULE.RunnerDependencies:
        return MODULE.RunnerDependencies(
            capture_prompt=self.capture,
            load_observer=self.load_observer,
            make_checkpoint=lambda _snapshot: self.checkpoint,
            make_factory=self.make_factory,
            compute_rows=self.compute_rows,
            runtime_provenance=lambda device: {
                "test": "cpu-only-injected-runtime",
                "requested_device": device,
            },
            begin_chunk=lambda _device: None,
            finish_chunk=lambda _device: {
                "peak_allocated_bytes": 123,
                "peak_reserved_bytes": 456,
            },
            release_prompt=self.release_prompt,
            monotonic=self.monotonic,
        )

    def release_prompt(self) -> None:
        self.release_count += 1


class RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.spec = MODULE.FitRunSpec(
            hidden_size=3,
            source_layers=(0, 1),
            target_layer=2,
            decoder_layers=3,
            prompt_count=2,
            token_count=5,
            skip_first=1,
            checkpoint_interval=2,
            io_rows=2,
            manifest_sha256="0" * 64,
        )
        self.prompts = [
            {
                "manifest_index": index,
                "text": f"prompt {index}",
                "token_ids": [index, 1, 2, 3, 4],
            }
            for index in range(self.spec.prompt_count)
        ]
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"prompts": self.prompts}))
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()

    def args(
        self,
        work: Path,
        *,
        resume: bool = False,
        cotangent_batch: int = 2,
        ste_policy: str = "identity",
    ) -> Namespace:
        return Namespace(
            work_dir=work,
            capture_dir=work / "captures",
            snapshot=self.snapshot,
            prompt_manifest=self.manifest,
            python=self.root / "python",
            capture_orchestrator=self.root / "capture.py",
            runtime_capture_script=self.root / "runtime.py",
            proof_script=self.root / "proof.py",
            ste_policy=ste_policy,
            cotangent_batch=cotangent_batch,
            device="fake:0",
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            max_num_batched_tokens=2048,
            max_num_seqs=1,
            hash_chunk_mib=1,
            resume=resume,
            plan_only=False,
        )

    def test_production_contract_constants_and_default_batch(self) -> None:
        spec = MODULE.PRODUCTION_SPEC
        self.assertEqual(spec.hidden_size, 5120)
        self.assertEqual(spec.source_layers, tuple(range(63)))
        self.assertEqual(spec.target_layer, 63)
        self.assertEqual(spec.prompt_count, 10)
        self.assertEqual(spec.token_count, 128)
        self.assertEqual(spec.skip_first, 16)
        self.assertEqual(spec.checkpoint_interval, 16)
        args = MODULE._parse_args(["--snapshot", str(self.snapshot)])
        self.assertEqual(args.cotangent_batch, 256)
        self.assertEqual(args.ste_policy, "identity")
        self.assertEqual(args.python, ROOT / ".venv-vllm" / "bin" / "python")
        request = MODULE._capture_request(args, self.prompts[0], 0, self.spec)
        self.assertEqual(request.output_dir, args.capture_dir / "prompt-00")

        runtime = FakeRuntime(self.root / "clipped-runtime", self.prompts, self.spec)
        with self.assertRaisesRegex(
            MODULE.FitOrchestrationError, "identity STE only"
        ):
            MODULE.execute(
                self.args(
                    self.root / "clipped-fit",
                    ste_policy="clipped",
                ),
                dependencies=runtime.dependencies(),
                spec=self.spec,
                prompts=self.prompts,
            )

    def test_pinned_manifest_loads_exact_frozen_ids(self) -> None:
        prompts = MODULE.load_pinned_prompts(MODULE.PINNED_MANIFEST)
        self.assertEqual(len(prompts), 10)
        self.assertTrue(all(prompt["token_count"] == 128 for prompt in prompts))
        self.assertTrue(all(len(prompt["token_ids"]) == 128 for prompt in prompts))
        self.assertEqual(
            prompts[0]["manifest_sha256"], MODULE.PINNED_MANIFEST_SHA256
        )

    def test_capture_command_deletes_baseline_and_resumes_explicitly(self) -> None:
        request = MODULE._capture_request(
            self.args(self.root / "work"), self.prompts[0], 0, self.spec
        )
        fresh = MODULE.capture_command(request, resume=False)
        resumed = MODULE.capture_command(request, resume=True)
        self.assertIn("--delete-baseline-pt-after-proof", fresh)
        self.assertNotIn("--model-path", fresh)
        self.assertNotIn("--resume", fresh)
        self.assertEqual(resumed[-1], "--resume")
        layer_flag = resumed.index("--capture-capacity")
        self.assertEqual(resumed[layer_flag + 1], "128")

    def test_observer_payload_and_row_results_fail_closed(self) -> None:
        runtime = FakeRuntime(self.root, self.prompts, self.spec)
        payload = runtime.load_observer(runtime.observer_paths[0])
        MODULE.validate_observer_payload(payload, self.prompts[0], spec=self.spec)

        changed = dict(payload)
        changed["prompt_token_ids"] = [9] * self.spec.token_count
        with self.assertRaisesRegex(MODULE.FitOrchestrationError, "token IDs"):
            MODULE.validate_observer_payload(changed, self.prompts[0], spec=self.spec)

        changed = dict(payload)
        changed["model_identity"] = {
            **payload["model_identity"],
            "revision": "alternate",
        }
        with self.assertRaisesRegex(MODULE.FitOrchestrationError, "model identity"):
            MODULE.validate_observer_payload(changed, self.prompts[0], spec=self.spec)

        rows = {
            layer: np.ones((2, self.spec.hidden_size), dtype=np.float64)
            for layer in self.spec.source_layers
        }
        normalized = MODULE.normalize_rows(
            rows, row_start=0, row_stop=2, spec=self.spec
        )
        self.assertTrue(all(value.dtype == MODULE.F32_LE for value in normalized.values()))
        with self.assertRaisesRegex(MODULE.FitOrchestrationError, "source layers"):
            MODULE.normalize_rows(
                {0: rows[0]}, row_start=0, row_stop=2, spec=self.spec
            )

    def test_capture_acceptance_binds_all_layer_proof_and_artifacts(self) -> None:
        args = self.args(self.root / "acceptance-work")
        request = MODULE._capture_request(args, self.prompts[0], 0, self.spec)
        capture_dir = self.root / "acceptance-capture"
        capture_dir.mkdir()
        paths = {
            name: capture_dir / filename
            for name, filename in {
                "state": "state.json",
                "lock": "lock",
                "baseline_json": "baseline.json",
                "baseline_tensors": "deleted-baseline.pt",
                "baseline_log": "baseline.log",
                "observer_json": "observer.json",
                "observer_tensors": "observer.pt",
                "observer_log": "observer.log",
                "proof": "proof.json",
                "proof_log": "proof.log",
            }.items()
        }
        for key in ("state", "baseline_json", "observer_json", "observer_tensors", "proof"):
            paths[key].write_bytes(key.encode("ascii"))

        model_identity = {
            "policy": MODULE.MODEL_IDENTITY_POLICY,
            "repo_id": MODULE.MODEL_ID,
            "revision": MODULE.MODEL_REVISION,
        }
        expected_layers = list(self.spec.capture_layers)
        patch = {
            "custom_op": MODULE.OBSERVER_CUSTOM_OP,
            "post_output_only": True,
            "target_layers": expected_layers,
        }
        install = {
            "target_layers": expected_layers,
            "capture_capacity": MODULE.CAPTURE_CAPACITY,
            "linear_boundary_count": MODULE.EXPECTED_LINEAR_BOUNDARIES,
        }
        baseline = {
            "model": {"identity": model_identity},
            "authoritative_compiled_generation": {"token": 7},
        }
        observer = {
            "model": {"identity": model_identity},
            "runtime": {
                "mode": "compiled-observer",
                "compiled_observer": True,
                "target_profile": "all",
                "target_layers": expected_layers,
                "compiled_observer_patch": patch,
            },
            "capture": {
                "install": install,
                "replay_parameter_provenance": {
                    f"parameter-{index}": {}
                    for index in range(MODULE.EXPECTED_REPLAY_PARAMETERS)
                },
            },
        }
        state = {
            "status": "complete",
            "contract_sha256": "c" * 64,
            "stages": {
                name: {"status": "complete"}
                for name in ("baseline", "observer", "proof")
            },
            "retention": {"baseline_tensors": "deleted_after_proof"},
            "artifacts": {
                key: MODULE._file_record(paths[key])
                for key in ("baseline_json", "observer_json", "observer_tensors", "proof")
            },
        }
        state["artifacts"]["baseline_tensors"] = {
            "path": str(paths["baseline_tensors"].resolve()),
            "bytes": 99,
            "sha256": "b" * 64,
        }
        proof = {
            "claim": {"mtp": "off"},
            "generation_record_parity": {"exact": True},
            "shared_internal_tensor_parity": {"all_shared_bit_exact": True},
            "replay_parameter_parity": {"all_content_hashes_equal": True},
            "observer_capture_completeness": {
                "required_missing": [],
                "truncated": [],
            },
        }
        capture = mock.Mock()
        capture._artifact_paths.return_value = paths
        capture._load_json.return_value = state
        capture._build_contract.return_value = (
            {"proof_script": {"sha256": "v" * 64}},
            {},
        )
        capture._validate_capture_json.side_effect = [baseline, observer]
        capture._validate_proof.return_value = proof
        with mock.patch.object(MODULE, "_capture_module", return_value=capture):
            result = MODULE.validate_capture_artifacts(
                request,
                command=["capture"],
                resume_used=False,
                stdout="ok",
                stderr="",
            )
        self.assertEqual(result.binding["model_identity"], model_identity)
        self.assertEqual(
            result.binding["observer_scope"]["install"]["linear_boundary_count"],
            MODULE.EXPECTED_LINEAR_BOUNDARIES,
        )
        self.assertEqual(
            result.binding["observer_tensors"],
            MODULE._file_record(paths["observer_tensors"]),
        )

    def test_interrupted_rows_resume_and_finalization_is_idempotent(self) -> None:
        work = self.root / "fit"
        first = FakeRuntime(
            self.root / "first-runtime",
            self.prompts,
            self.spec,
            fail_compute_call=2,
        )
        with self.assertRaisesRegex(RuntimeError, "simulated replay crash"):
            MODULE.execute(
                self.args(work),
                dependencies=first.dependencies(),
                spec=self.spec,
                prompts=self.prompts,
            )
        state = MODULE.read_json(work / "state.json")
        self.assertEqual(state["current"]["next_row"], 2)
        self.assertEqual(first.compute_calls, [(0, 0, 2), (0, 2, 3)])

        resumed = FakeRuntime(
            self.root / "resumed-runtime", self.prompts, self.spec
        )
        # Preserve the exact observer artifact binding established before failure.
        resumed.observer_paths = first.observer_paths
        result = MODULE.execute(
            self.args(work, resume=True),
            dependencies=resumed.dependencies(),
            spec=self.spec,
            prompts=self.prompts,
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            resumed.compute_calls,
            [(0, 2, 3), (1, 0, 2), (1, 2, 3)],
        )
        self.assertEqual(resumed.release_count, 2)

        for layer in self.spec.source_layers:
            path = work / "final-mean" / f"layer-{layer:02d}.f32"
            matrix = np.memmap(
                path,
                mode="r",
                dtype=MODULE.F32_LE,
                shape=(self.spec.hidden_size, self.spec.hidden_size),
            )
            expected = np.repeat(
                (
                    np.arange(self.spec.hidden_size, dtype=np.float32)
                    + 5
                    + layer * 100
                )[:, None],
                self.spec.hidden_size,
                axis=1,
            )
            np.testing.assert_array_equal(matrix, expected)
            del matrix

        metadata_path = work / "final-mean" / "metadata.json"
        first_metadata_hash = MODULE.sha256_file(metadata_path)
        second = MODULE.execute(
            self.args(work, resume=True),
            dependencies=resumed.dependencies(),
            spec=self.spec,
            prompts=self.prompts,
        )
        self.assertEqual(second["status"], "completed")
        self.assertEqual(MODULE.sha256_file(metadata_path), first_metadata_hash)

    def test_observer_is_rehashed_immediately_before_prompt_commit(self) -> None:
        work = self.root / "observer-mutation-fit"
        runtime = FakeRuntime(
            self.root / "observer-mutation-runtime",
            self.prompts,
            self.spec,
            mutate_observer_compute_call=2,
        )
        with self.assertRaisesRegex(
            MODULE.FitOrchestrationError,
            "observer PT changed before prompt commit",
        ):
            MODULE.execute(
                self.args(work),
                dependencies=runtime.dependencies(),
                spec=self.spec,
                prompts=self.prompts,
            )
        state = MODULE.read_json(work / "state.json")
        self.assertEqual(state["n_done"], 0)
        self.assertIsNone(state["current"])
        self.assertFalse((work / "current").exists())
        self.assertEqual(runtime.checkpoint.integrity_calls, 1)

        recovered = FakeRuntime(
            self.root / "observer-mutation-runtime",
            self.prompts,
            self.spec,
        )
        result = MODULE.execute(
            self.args(work, resume=True),
            dependencies=recovered.dependencies(),
            spec=self.spec,
            prompts=self.prompts,
        )
        self.assertEqual(result["status"], "completed")
        journal = MODULE.read_json(work / "run-progress.json")
        attempt = journal["prompts"]["0"]["invalidated_chunk_attempts"][0]
        self.assertIsInstance(attempt["capture_binding"], dict)
        self.assertEqual(len(attempt["capture_invocations"]), 1)

    def test_checkpoint_integrity_failure_prevents_prompt_commit(self) -> None:
        work = self.root / "checkpoint-failure-fit"
        runtime = FakeRuntime(
            self.root / "checkpoint-failure-runtime",
            self.prompts,
            self.spec,
            checkpoint_error=RuntimeError("simulated checkpoint integrity failure"),
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint integrity failure"):
            MODULE.execute(
                self.args(work),
                dependencies=runtime.dependencies(),
                spec=self.spec,
                prompts=self.prompts,
            )
        state = MODULE.read_json(work / "state.json")
        self.assertEqual(state["n_done"], 0)
        self.assertIsNone(state["current"])
        self.assertFalse((work / "current").exists())
        self.assertEqual(runtime.checkpoint.integrity_calls, 1)
        with self.assertRaisesRegex(
            MODULE.FitOrchestrationError,
            "lacks validate_pinned_integrity",
        ):
            MODULE._validate_checkpoint_before_commit(object(), required=True)

        recovered = FakeRuntime(
            self.root / "checkpoint-failure-runtime",
            self.prompts,
            self.spec,
        )
        result = MODULE.execute(
            self.args(work, resume=True),
            dependencies=recovered.dependencies(),
            spec=self.spec,
            prompts=self.prompts,
        )
        self.assertEqual(result["status"], "completed")
        journal = MODULE.read_json(work / "run-progress.json")
        prompt_zero = journal["prompts"]["0"]
        self.assertEqual(len(prompt_zero["invalidated_chunk_attempts"]), 1)
        self.assertEqual(
            [(chunk["start"], chunk["stop"]) for chunk in prompt_zero["chunks"]],
            [(0, 2), (2, 3)],
        )

    def test_resume_rejects_contract_and_capture_binding_changes(self) -> None:
        contract_work = self.root / "contract-fit"
        runtime_root = self.root / "contract-runtime"
        runtime = FakeRuntime(
            runtime_root,
            self.prompts,
            self.spec,
            fail_compute_call=2,
        )
        with self.assertRaisesRegex(RuntimeError, "simulated replay crash"):
            MODULE.execute(
                self.args(contract_work),
                dependencies=runtime.dependencies(),
                spec=self.spec,
                prompts=self.prompts,
            )
        with self.assertRaisesRegex(RuntimeError, "resume contract mismatch"):
            MODULE.execute(
                self.args(contract_work, resume=True, cotangent_batch=1),
                dependencies=runtime.dependencies(),
                spec=self.spec,
                prompts=self.prompts,
            )

        runtime.observer_paths[0].write_bytes(b"changed observer")
        with self.assertRaisesRegex(
            MODULE.FitOrchestrationError, "capture binding changed"
        ):
            MODULE.execute(
                self.args(contract_work, resume=True),
                dependencies=runtime.dependencies(),
                spec=self.spec,
                prompts=self.prompts,
            )

    def test_source_rehash_and_capture_quarantine_fail_closed(self) -> None:
        source_root = self.root / "source-root"
        source = source_root / "scripts" / "bound.py"
        source.parent.mkdir(parents=True)
        source.write_text("value = 1\n", encoding="ascii")
        record = {
            "path": "scripts/bound.py",
            "bytes": source.stat().st_size,
            "sha256": MODULE.sha256_file(source),
        }
        with mock.patch.object(MODULE, "ROOT", source_root):
            MODULE._validate_bound_source_files([record])
            source.write_text("value = 2\n", encoding="ascii")
            with self.assertRaisesRegex(
                MODULE.FitOrchestrationError, "bound source changed"
            ):
                MODULE._validate_bound_source_files([record])

        capture = self.root / "captures" / "prompt-00"
        capture.mkdir(parents=True)
        (capture / "artifact.pt").write_bytes(b"bad")
        quarantine = MODULE._quarantine_capture_directory(capture, 0)
        self.assertIsNotNone(quarantine)
        self.assertFalse(capture.exists())
        self.assertEqual((quarantine / "artifact.pt").read_bytes(), b"bad")


if __name__ == "__main__":
    unittest.main()
