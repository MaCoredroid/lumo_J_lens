#!/usr/bin/env python3
"""Run the production Qwen3.6-27B NVFP4/FP8-STE Jacobian Lens fit.

Each pinned prompt is captured in an isolated compiled vLLM subprocess and
accepted only after exact baseline/observer pair proof.  The observer payload
is replayed block-by-block with packed ModelOpt NVFP4 and live post-load FP8
VJPs.  Dense output rows are committed through ``FitStateStore`` so an
interrupted 10-prompt fit resumes from its last integrity-hashed row chunk.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

try:
    from scripts.nvfp4_fit_state import (
        F32_LE,
        FitLayout,
        FitStateStore,
        atomic_write_json,
        canonical_json_value,
        canonical_sha256,
        read_json,
        sha256_file,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from nvfp4_fit_state import (
        F32_LE,
        FitLayout,
        FitStateStore,
        atomic_write_json,
        canonical_json_value,
        canonical_sha256,
        read_json,
        sha256_file,
    )


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
MODEL_IDENTITY_POLICY = "pinned-nvidia-modelopt-snapshot-v1"
PINNED_MANIFEST = ROOT / "configs" / "jlens_nf4_fit_prompts.json"
PINNED_MANIFEST_SHA256 = (
    "2c36f17dee7287c096f7d1fdb7f8d7ecb8372c6cf8d13d7af8cfaed820439d3b"
)
PROGRESS_SCHEMA_VERSION = 1
CAPTURE_CAPACITY = 128
EXPECTED_LINEAR_BOUNDARIES = 304
EXPECTED_REPLAY_PARAMETERS = 785
OBSERVER_CUSTOM_OP = "jlens_nvfp4::capture_output"
PRODUCTION_STE_POLICY = "identity"


class FitOrchestrationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FitOrchestrationError(message)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _absolute_without_symlink_resolution(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (Path.cwd() / expanded).absolute()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _validate_bound_source_files(records: Sequence[Mapping[str, Any]]) -> None:
    """Rehash every source bound into the frozen run contract."""

    for record in records:
        relative = record.get("path")
        _require(isinstance(relative, str), "source record path is not a string")
        relative_path = Path(relative)
        _require(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"unsafe source record path: {relative}",
        )
        path = (ROOT / relative_path).resolve()
        _require(path.is_relative_to(ROOT), f"source record escapes repository: {relative}")
        current = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        _require(current == record, f"bound source changed during fit: {relative}")


def _stable_artifact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(Path(record["path"]).resolve()),
        "bytes": int(record["bytes"]),
        "sha256": str(record["sha256"]),
    }


@dataclass(frozen=True)
class FitRunSpec:
    hidden_size: int = 5120
    source_layers: tuple[int, ...] = tuple(range(63))
    target_layer: int = 63
    decoder_layers: int = 64
    prompt_count: int = 10
    token_count: int = 128
    skip_first: int = 16
    checkpoint_interval: int = 16
    io_rows: int = 64
    manifest_sha256: str = PINNED_MANIFEST_SHA256

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.prompt_count <= 0 or self.token_count <= 0:
            raise ValueError("fit dimensions must be positive")
        if self.source_layers != tuple(range(self.target_layer)):
            raise ValueError("source layers must be exactly 0..target_layer-1")
        if self.decoder_layers != self.target_layer + 1:
            raise ValueError("target layer must be the final decoder layer")
        if not 0 <= self.skip_first < self.token_count - 1:
            raise ValueError("skip_first leaves no estimator positions")
        if self.checkpoint_interval <= 0 or self.io_rows <= 0:
            raise ValueError("checkpoint and I/O intervals must be positive")

    @property
    def capture_layers(self) -> tuple[int, ...]:
        return tuple(range(self.decoder_layers))

    def fit_layout(self) -> FitLayout:
        return FitLayout(
            hidden_size=self.hidden_size,
            source_layers=self.source_layers,
            prompt_count=self.prompt_count,
            io_rows=self.io_rows,
        )

    def record(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "source_layers": list(self.source_layers),
            "target_layer": self.target_layer,
            "decoder_layers": self.decoder_layers,
            "prompt_count": self.prompt_count,
            "token_count": self.token_count,
            "skip_first": self.skip_first,
            "checkpoint_interval": self.checkpoint_interval,
            "io_rows": self.io_rows,
        }


PRODUCTION_SPEC = FitRunSpec()


@dataclass(frozen=True)
class CaptureRequest:
    prompt: Mapping[str, Any]
    prompt_index: int
    output_dir: Path
    python: Path
    capture_orchestrator: Path
    runtime_capture_script: Path
    proof_script: Path
    prompt_manifest: Path
    gpu_memory_utilization: float
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    hash_chunk_mib: int
    capture_layers: tuple[int, ...]


@dataclass(frozen=True)
class CaptureResult:
    binding: Mapping[str, Any]
    invocation: Mapping[str, Any]


class CaptureRunner(Protocol):
    def __call__(self, request: CaptureRequest) -> CaptureResult: ...


@dataclass(frozen=True)
class RunnerDependencies:
    capture_prompt: CaptureRunner
    load_observer: Callable[[Path], Mapping[str, Any]]
    make_checkpoint: Callable[[Path], Any]
    make_factory: Callable[[Mapping[str, Any], Any, str, int], Any]
    compute_rows: Callable[
        [Any, tuple[int, ...], int, int, int, str], Mapping[int, Any]
    ]
    runtime_provenance: Callable[[str], Mapping[str, Any]]
    begin_chunk: Callable[[str], None]
    finish_chunk: Callable[[str], Mapping[str, Any]]
    release_prompt: Callable[[], None]
    monotonic: Callable[[], float] = time.monotonic


def _capture_module() -> Any:
    try:
        from scripts import capture_nvfp4_fit_prompt as capture
    except ImportError:
        import capture_nvfp4_fit_prompt as capture
    return capture


def load_pinned_prompts(
    manifest_path: Path,
    *,
    spec: FitRunSpec = PRODUCTION_SPEC,
) -> list[dict[str, Any]]:
    """Load every frozen prompt and bind its exact token IDs and manifest hash."""

    resolved = manifest_path.expanduser().resolve()
    raw = resolved.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    _require(
        actual_hash == spec.manifest_sha256,
        f"prompt manifest SHA-256 mismatch: {actual_hash}",
    )
    value = json.loads(raw)
    prompts = value.get("prompts") if isinstance(value, dict) else None
    _require(
        isinstance(prompts, list) and len(prompts) == spec.prompt_count,
        f"prompt manifest must contain exactly {spec.prompt_count} prompts",
    )
    frozen: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        _require(isinstance(prompt, dict), f"prompt {index} is not an object")
        text = prompt.get("text")
        token_ids = prompt.get("token_ids")
        _require(isinstance(text, str) and bool(text), f"prompt {index} has no text")
        _require(
            isinstance(token_ids, list)
            and len(token_ids) == spec.token_count
            and all(isinstance(token, int) and token >= 0 for token in token_ids),
            f"prompt {index} frozen token IDs are invalid",
        )
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        _require(
            prompt.get("text_sha256") == text_hash,
            f"prompt {index} text SHA-256 mismatch",
        )
        _require(
            prompt.get("token_count") == spec.token_count,
            f"prompt {index} token count mismatch",
        )
        frozen.append(
            {
                "manifest_path": str(resolved),
                "manifest_sha256": actual_hash,
                "manifest_prompt_count": spec.prompt_count,
                "manifest_index": index,
                "row_index": prompt.get("row_index"),
                "text": text,
                "text_sha256": text_hash,
                "token_count": spec.token_count,
                "token_ids": list(token_ids),
                "token_ids_sha256": canonical_sha256(token_ids),
            }
        )
    return frozen


def capture_command(request: CaptureRequest, *, resume: bool) -> list[str]:
    python = _absolute_without_symlink_resolution(request.python)
    command = [
        str(python),
        str(request.capture_orchestrator.resolve()),
        "--prompt-index",
        str(request.prompt_index),
        "--prompt-manifest",
        str(request.prompt_manifest.resolve()),
        "--output-dir",
        str(request.output_dir.resolve()),
        "--python",
        str(python),
        "--capture-script",
        str(request.runtime_capture_script.resolve()),
        "--proof-script",
        str(request.proof_script.resolve()),
        "--gpu-memory-utilization",
        str(request.gpu_memory_utilization),
        "--max-model-len",
        str(request.max_model_len),
        "--max-num-batched-tokens",
        str(request.max_num_batched_tokens),
        "--max-num-seqs",
        str(request.max_num_seqs),
        "--capture-capacity",
        str(CAPTURE_CAPACITY),
        "--hash-chunk-mib",
        str(request.hash_chunk_mib),
        "--delete-baseline-pt-after-proof",
    ]
    if resume:
        command.append("--resume")
    return command


def _capture_artifacts_exist(paths: Mapping[str, Path]) -> bool:
    return any(
        path.exists()
        for name, path in paths.items()
        if name != "lock"
    )


def _matching_artifact(
    state: Mapping[str, Any],
    key: str,
    path: Path,
) -> dict[str, Any]:
    record = state.get("artifacts", {}).get(key)
    _require(isinstance(record, dict), f"capture state has no {key} hash")
    current = _file_record(path)
    expected = _stable_artifact_record(record)
    _require(current == expected, f"capture artifact hash mismatch: {key}")
    return current


def validate_capture_artifacts(
    request: CaptureRequest,
    *,
    command: Sequence[str],
    resume_used: bool,
    stdout: str,
    stderr: str,
) -> CaptureResult:
    """Fail closed unless capture state, observer scope, and pair proof agree."""

    capture = _capture_module()
    paths = capture._artifact_paths(request.output_dir, request.prompt_index)
    state = capture._load_json(paths["state"])
    capture_args = argparse.Namespace(
        python=request.python,
        capture_script=request.runtime_capture_script,
        proof_script=request.proof_script,
        capture_capacity=CAPTURE_CAPACITY,
        gpu_memory_utilization=request.gpu_memory_utilization,
        max_model_len=request.max_model_len,
        max_num_batched_tokens=request.max_num_batched_tokens,
        max_num_seqs=request.max_num_seqs,
        hash_chunk_mib=request.hash_chunk_mib,
    )
    expected_contract, _commands = capture._build_contract(
        capture_args, dict(request.prompt), paths
    )
    capture._validate_state_contract(state, expected_contract)
    _require(state.get("status") == "complete", "capture state is not complete")
    for stage in ("baseline", "observer", "proof"):
        _require(
            state.get("stages", {}).get(stage, {}).get("status") == "complete",
            f"capture {stage} stage is not complete",
        )
    _require(
        state.get("retention", {}).get("baseline_tensors")
        == "deleted_after_proof",
        "capture did not record proved baseline PT deletion",
    )
    _require(
        not paths["baseline_tensors"].exists(),
        "baseline PT remains after requested proved deletion",
    )

    baseline = capture._validate_capture_json(
        paths["baseline_json"],
        mode="compiled",
        prompt=dict(request.prompt),
        tensor_path=paths["baseline_tensors"],
        capture_capacity=CAPTURE_CAPACITY,
    )
    observer = capture._validate_capture_json(
        paths["observer_json"],
        mode="compiled-observer",
        prompt=dict(request.prompt),
        tensor_path=paths["observer_tensors"],
        capture_capacity=CAPTURE_CAPACITY,
    )
    runtime = observer.get("runtime", {})
    patch = runtime.get("compiled_observer_patch", {})
    install = observer.get("capture", {}).get("install", {})
    expected_layers = list(request.capture_layers)
    _require(runtime.get("compiled_observer") is True, "observer mode is not compiled")
    _require(runtime.get("target_profile") == "all", "observer scope is not all")
    _require(runtime.get("target_layers") == expected_layers, "observer layers mismatch")
    _require(
        patch.get("custom_op") == OBSERVER_CUSTOM_OP
        and patch.get("post_output_only") is True
        and patch.get("target_layers") == expected_layers,
        "compiled observer patch scope mismatch",
    )
    _require(
        install.get("target_layers") == expected_layers
        and install.get("capture_capacity") == CAPTURE_CAPACITY
        and install.get("linear_boundary_count") == EXPECTED_LINEAR_BOUNDARIES,
        "compiled observer install scope mismatch",
    )
    _require(
        len(observer.get("capture", {}).get("replay_parameter_provenance", {}))
        == EXPECTED_REPLAY_PARAMETERS,
        "observer replay parameter inventory mismatch",
    )

    current_records = {
        "baseline_json": _matching_artifact(state, "baseline_json", paths["baseline_json"]),
        "observer_json": _matching_artifact(state, "observer_json", paths["observer_json"]),
        "observer_tensors": _matching_artifact(
            state, "observer_tensors", paths["observer_tensors"]
        ),
        "proof": _matching_artifact(state, "proof", paths["proof"]),
    }
    baseline_tensor_record = state.get("artifacts", {}).get("baseline_tensors")
    _require(
        isinstance(baseline_tensor_record, dict),
        "capture state lost the deleted baseline PT hash",
    )
    proof_inputs = {
        key: state["artifacts"][key]
        for key in (
            "baseline_json",
            "baseline_tensors",
            "observer_json",
            "observer_tensors",
        )
    }
    proof = capture._validate_proof(
        paths["proof"],
        prompt=dict(request.prompt),
        artifacts=proof_inputs,
        proof_script_sha256=expected_contract["proof_script"]["sha256"],
    )
    proof_claim = {
        "claim": proof.get("claim"),
        "generation_record_parity": proof.get("generation_record_parity"),
        "shared_internal_tensor_parity": proof.get("shared_internal_tensor_parity"),
        "replay_parameter_parity": proof.get("replay_parameter_parity"),
        "observer_capture_completeness": proof.get(
            "observer_capture_completeness"
        ),
    }
    observer_scope = {
        "mode": runtime.get("mode"),
        "compiled_observer": runtime.get("compiled_observer"),
        "target_profile": runtime.get("target_profile"),
        "target_layers": runtime.get("target_layers"),
        "compiled_observer_patch": patch,
        "install": install,
    }
    model_identity = observer.get("model", {}).get("identity")
    _require(
        model_identity == baseline.get("model", {}).get("identity"),
        "baseline and observer model identities differ",
    )
    binding = {
        "schema_version": 1,
        "prompt_index": request.prompt_index,
        "capture_contract_sha256": state["contract_sha256"],
        "capture_state_path": str(paths["state"].resolve()),
        "baseline_json": current_records["baseline_json"],
        "baseline_tensors_deleted_record": _stable_artifact_record(
            baseline_tensor_record
        ),
        "observer_json": current_records["observer_json"],
        "observer_tensors": current_records["observer_tensors"],
        "proof": current_records["proof"],
        "observer_scope": observer_scope,
        "observer_scope_sha256": canonical_sha256(observer_scope),
        "proof_claim": proof_claim,
        "proof_claim_sha256": canonical_sha256(proof_claim),
        "model_identity": model_identity,
        "model_identity_sha256": canonical_sha256(model_identity),
        "baseline_generation_sha256": canonical_sha256(
            baseline.get("authoritative_compiled_generation")
        ),
    }
    invocation = {
        "argv": list(command),
        "resume_used": resume_used,
        "completed_at": _utc_now(),
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "capture_state": _file_record(paths["state"]),
    }
    return CaptureResult(binding=binding, invocation=invocation)


def default_capture_prompt(request: CaptureRequest) -> CaptureResult:
    capture = _capture_module()
    paths = capture._artifact_paths(request.output_dir, request.prompt_index)
    resume = _capture_artifacts_exist(paths)
    command = capture_command(request, resume=resume)
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return validate_capture_artifacts(
        request,
        command=command,
        resume_used=resume,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def validate_observer_payload(
    payload: Mapping[str, Any],
    prompt: Mapping[str, Any],
    *,
    spec: FitRunSpec,
) -> None:
    _require(payload.get("schema_version") == 1, "observer payload schema mismatch")
    _require(payload.get("mode") == "compiled-observer", "observer payload mode mismatch")
    _require(
        payload.get("model_revision") == MODEL_REVISION,
        "observer payload model revision mismatch",
    )
    capture = _capture_module()
    model_identity = payload.get("model_identity")
    _require(isinstance(model_identity, Mapping), "observer model identity is missing")
    _require(
        model_identity.get("policy") == MODEL_IDENTITY_POLICY
        and model_identity.get("repo_id") == MODEL_ID
        and model_identity.get("revision") == MODEL_REVISION
        and model_identity.get("metadata_sha256")
        == capture.PINNED_METADATA_SHA256
        and model_identity.get("strict_pinned_validation") is True
        and model_identity.get("validator")
        == "ModelOptCheckpoint(strict_pinned=True)",
        "observer payload lacks strict pinned model identity",
    )
    resolved_model_path = model_identity.get("resolved_path")
    _require(
        isinstance(resolved_model_path, str),
        "observer payload has no resolved model path",
    )
    snapshot = Path(resolved_model_path)
    _require(
        snapshot.name == MODEL_REVISION
        and snapshot.parent.name == "snapshots"
        and snapshot.parent.parent.name
        == "models--nvidia--Qwen3.6-27B-NVFP4",
        "observer payload model path is not the pinned snapshot layout",
    )
    _require(
        payload.get("target_profile") == "all"
        and payload.get("target_layers") == list(spec.capture_layers),
        "observer payload target scope mismatch",
    )
    _require(
        payload.get("prompt_token_ids") == prompt.get("token_ids"),
        "observer payload frozen token IDs mismatch",
    )
    _require(payload.get("prompt") == prompt.get("text"), "observer payload text mismatch")
    tensors = payload.get("tensors")
    _require(isinstance(tensors, Mapping), "observer payload tensors are missing")
    for layer in range(spec.target_layer + 1):
        name = f"h{layer}_post_block"
        _require(name in tensors, f"observer payload is missing {name}")
        value = tensors[name]
        shape = tuple(getattr(value, "shape", ()))
        _require(
            shape == (spec.token_count, spec.hidden_size),
            f"observer payload {name} shape mismatch: {shape}",
        )


def normalize_rows(
    rows: Mapping[int, Any],
    *,
    row_start: int,
    row_stop: int,
    spec: FitRunSpec,
) -> dict[int, np.ndarray]:
    _require(set(rows) == set(spec.source_layers), "row result source layers mismatch")
    expected = (row_stop - row_start, spec.hidden_size)
    normalized: dict[int, np.ndarray] = {}
    for layer in spec.source_layers:
        value = rows[layer]
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        array = np.asarray(value, dtype=F32_LE)
        _require(array.shape == expected, f"layer {layer} row shape mismatch")
        _require(bool(np.isfinite(array).all()), f"layer {layer} rows are non-finite")
        normalized[layer] = array
    return normalized


def _default_load_observer(path: Path) -> Mapping[str, Any]:
    import torch

    value = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(value, Mapping):
        raise FitOrchestrationError("observer PT root is not a mapping")
    return value


def _default_make_checkpoint(snapshot: Path) -> Any:
    try:
        from modelopt_checkpoint import ModelOptCheckpoint
    except ImportError:
        from scripts.modelopt_checkpoint import ModelOptCheckpoint
    return ModelOptCheckpoint(snapshot)


def _default_make_factory(
    payload: Mapping[str, Any],
    checkpoint: Any,
    ste_policy: str,
    checkpoint_interval: int,
) -> Any:
    try:
        from nvfp4_block_replay import (
            CapturedQwenBlockReplayFactory,
            LiveFp8Backend,
            PackedNvFp4W4Backend,
        )
    except ImportError:
        from scripts.nvfp4_block_replay import (
            CapturedQwenBlockReplayFactory,
            LiveFp8Backend,
            PackedNvFp4W4Backend,
        )
    factory = CapturedQwenBlockReplayFactory(
        payload,
        PackedNvFp4W4Backend(checkpoint),
        fp8_backend=LiveFp8Backend(),
        ste_policy=ste_policy,
        checkpoint_interval=checkpoint_interval,
        require_exact_input=True,
    )
    return factory


def _default_compute_rows(
    factory: Any,
    valid_positions: tuple[int, ...],
    row_start: int,
    row_stop: int,
    target_layer: int,
    device: str,
) -> Mapping[int, Any]:
    import torch

    positions = torch.tensor(valid_positions, dtype=torch.long)
    result = factory.reverse_replay_rows(
        positions,
        row_start,
        row_stop,
        first_block=1,
        target_layer=target_layer,
        device=device,
    )
    return result.rows


def _package_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "triton",
        "vllm",
        "transformers",
        "safetensors",
        "numpy",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _default_runtime_provenance(device: str) -> Mapping[str, Any]:
    import torch

    target = torch.device(device)
    _require(target.type == "cuda", "production replay device must be CUDA")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    properties = torch.cuda.get_device_properties(target)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": _package_versions(),
        "cuda": {
            "device": str(target),
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "torch_cuda": torch.version.cuda,
        },
    }


def _default_begin_chunk(device: str) -> None:
    import torch

    target = torch.device(device)
    torch.cuda.synchronize(target)
    torch.cuda.reset_peak_memory_stats(target)


def _default_finish_chunk(device: str) -> Mapping[str, Any]:
    import torch

    target = torch.device(device)
    torch.cuda.synchronize(target)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(target),
        "reserved_bytes": torch.cuda.memory_reserved(target),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(target),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(target),
    }


def _default_release_prompt() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def default_dependencies() -> RunnerDependencies:
    # Import the complete replay graph before hashing its sources. Loaded
    # production code cannot drift while isolated captures are running.
    import fit_jlens_nvfp4_ste  # noqa: F401
    import nvfp4_block_replay  # noqa: F401

    return RunnerDependencies(
        capture_prompt=default_capture_prompt,
        load_observer=_default_load_observer,
        make_checkpoint=_default_make_checkpoint,
        make_factory=_default_make_factory,
        compute_rows=_default_compute_rows,
        runtime_provenance=_default_runtime_provenance,
        begin_chunk=_default_begin_chunk,
        finish_chunk=_default_finish_chunk,
        release_prompt=_default_release_prompt,
    )


def _source_records() -> list[dict[str, Any]]:
    files = (
        "scripts/run_nvfp4_ste_fit.py",
        "scripts/nvfp4_fit_state.py",
        "scripts/capture_nvfp4_fit_prompt.py",
        "scripts/check_nvfp4_runtime_capture.py",
        "scripts/prove_nvfp4_capture_pair.py",
        "scripts/modelopt_checkpoint.py",
        "scripts/nvfp4_block_replay.py",
        "scripts/fit_jlens_nvfp4_ste.py",
        "scripts/nvfp4_packed_vjp.py",
        "scripts/fp8_live_vjp.py",
        "scripts/nvfp4_gdn.py",
        "scripts/nvfp4_attention.py",
        "scripts/nvfp4_ste.py",
    )
    return [
        {
            "path": relative,
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": sha256_file(ROOT / relative),
        }
        for relative in files
    ]


def build_run_contract(
    args: argparse.Namespace,
    prompts: Sequence[Mapping[str, Any]],
    runtime_provenance: Mapping[str, Any],
    *,
    spec: FitRunSpec,
) -> dict[str, Any]:
    try:
        from scripts import modelopt_checkpoint as checkpoint_module
    except ImportError:
        import modelopt_checkpoint as checkpoint_module

    source_records = _source_records()
    checkpoint_files = {
        "metadata_sha256": checkpoint_module.PINNED_METADATA_SHA256,
        "shards": {
            name: {"bytes": size, "blob_sha256": digest}
            for name, (size, digest) in checkpoint_module.PINNED_SHARDS.items()
        },
    }
    return {
        "schema_version": 1,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot": str(args.snapshot.resolve()),
            "checkpoint_files": checkpoint_files,
            "checkpoint_integrity_before_each_prompt_commit": True,
        },
        "prompts": {
            "manifest": _file_record(args.prompt_manifest),
            "entries": list(prompts),
            "entries_sha256": canonical_sha256(prompts),
        },
        "estimator": {
            **spec.record(),
            "name": "anthropic-future-summed-vjp",
            "mean_over_source_positions": True,
            "is_grads_batched": True,
            "cotangent_batch": args.cotangent_batch,
        },
        "capture": {
            "scope": "exact compiled main-model observer, all decoder layers",
            "model_repo": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_identity_policy": MODEL_IDENTITY_POLICY,
            "model_path_override_allowed": False,
            "capture_capacity": CAPTURE_CAPACITY,
            "observer_custom_op": OBSERVER_CUSTOM_OP,
            "pair_proof_required": True,
            "observer_rehash_before_each_prompt_commit": True,
            "prompt_scoped_directories": True,
            "rejected_capture_quarantine": True,
            "mtp_enabled": False,
            "delete_baseline_pt_after_proof": True,
            "output_dir": str(args.capture_dir.resolve()),
            "python": str(_absolute_without_symlink_resolution(args.python)),
            "python_resolved_target": str(args.python.resolve()),
            "capture_orchestrator": str(args.capture_orchestrator.resolve()),
            "runtime_capture_script": str(args.runtime_capture_script.resolve()),
            "proof_script": str(args.proof_script.resolve()),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "hash_chunk_mib": args.hash_chunk_mib,
        },
        "surrogate_backward": {
            "w4_backend": "PackedNvFp4W4Backend raw ModelOpt E2M1/FP8 scales",
            "fp8_backend": "LiveFp8Backend post-load E4M3/scalar scale",
            "activation_ste_policy": args.ste_policy,
            "clipped_ste_supported": False,
            "clipped_ste_exclusion": (
                "compiled captures omit exact pre-linear inputs needed for masks"
            ),
            "literal_rounding_derivative": False,
            "checkpoint_interval": spec.checkpoint_interval,
        },
        "storage": spec.fit_layout().record(),
        "device": args.device,
        "runtime_provenance": canonical_json_value(runtime_provenance),
        "source_files": source_records,
        "source_files_sha256": canonical_sha256(source_records),
        "source_integrity_before_each_prompt_commit": True,
    }


class ProgressJournal:
    """Non-authoritative atomic timing journal reconciled to fit-state chunks."""

    def __init__(self, path: Path, state: dict[str, Any]) -> None:
        self.path = path
        self.state = state

    @classmethod
    def open(
        cls,
        path: Path,
        contract_sha256: str,
        *,
        resume: bool,
        fit_state: Mapping[str, Any],
    ) -> "ProgressJournal":
        if path.exists():
            value = read_json(path)
            _require(isinstance(value, dict), "progress journal is not an object")
            _require(
                value.get("schema_version") == PROGRESS_SCHEMA_VERSION
                and value.get("contract_sha256") == contract_sha256,
                "progress journal contract mismatch",
            )
            journal = cls(path, value)
        else:
            _require(
                not resume
                or (
                    fit_state.get("n_done") == 0
                    and fit_state.get("current") is None
                ),
                "resume state has no progress journal",
            )
            now = _utc_now()
            journal = cls(
                path,
                {
                    "schema_version": PROGRESS_SCHEMA_VERSION,
                    "contract_sha256": contract_sha256,
                    "created_at": now,
                    "updated_at": now,
                    "status": "running",
                    "prompts": {},
                    "max_cuda_peak_allocated_bytes": 0,
                    "max_cuda_peak_reserved_bytes": 0,
                },
            )
            journal.save()
        journal.reconcile(fit_state)
        return journal

    def save(self) -> None:
        self.state["updated_at"] = _utc_now()
        atomic_write_json(self.path, self.state)

    def record_capture(
        self,
        index: int,
        result: CaptureResult,
    ) -> None:
        record = self.state["prompts"].setdefault(
            str(index), {"chunks": [], "capture_invocations": []}
        )
        previous = record.get("capture_binding")
        binding = canonical_json_value(result.binding)
        _require(
            previous is None or previous == binding,
            f"prompt {index} capture binding changed",
        )
        record["capture_binding"] = binding
        record["capture_invocations"].append(canonical_json_value(result.invocation))
        self.save()

    def record_chunk(
        self,
        index: int,
        chunk: Mapping[str, Any],
        *,
        elapsed_seconds: float,
        cuda: Mapping[str, Any],
    ) -> None:
        record = self.state["prompts"].setdefault(
            str(index), {"chunks": [], "capture_invocations": []}
        )
        key = (chunk["start"], chunk["stop"])
        existing = {
            (entry["start"], entry["stop"]): entry for entry in record["chunks"]
        }
        _require(key not in existing, f"prompt {index} chunk telemetry duplicated")
        entry = {
            "start": chunk["start"],
            "stop": chunk["stop"],
            "sha256": chunk["sha256"],
            "elapsed_seconds": elapsed_seconds,
            "cuda": canonical_json_value(cuda),
        }
        record["chunks"].append(entry)
        peak_allocated = int(cuda.get("peak_allocated_bytes", 0))
        peak_reserved = int(cuda.get("peak_reserved_bytes", 0))
        self.state["max_cuda_peak_allocated_bytes"] = max(
            self.state["max_cuda_peak_allocated_bytes"], peak_allocated
        )
        self.state["max_cuda_peak_reserved_bytes"] = max(
            self.state["max_cuda_peak_reserved_bytes"], peak_reserved
        )
        self.save()

    def record_commit(self, index: int, commit: Mapping[str, Any]) -> None:
        record = self.state["prompts"].setdefault(
            str(index), {"chunks": [], "capture_invocations": []}
        )
        record["commit"] = canonical_json_value(commit)
        self.save()

    def invalidate_uncommitted_chunks(
        self,
        index: int,
        reason: str,
        *,
        invalidate_capture: bool = False,
    ) -> None:
        """Archive telemetry for rows that the authoritative fit state discarded."""

        record = self.state["prompts"].setdefault(
            str(index), {"chunks": [], "capture_invocations": []}
        )
        _require("commit" not in record, f"cannot invalidate committed prompt {index}")
        chunks = record.get("chunks")
        _require(isinstance(chunks, list), f"prompt {index} chunks are not a list")
        if chunks or invalidate_capture:
            attempts = record.setdefault("invalidated_chunk_attempts", [])
            _require(
                isinstance(attempts, list),
                f"prompt {index} invalidated attempts are not a list",
            )
            attempt = {
                "reason": reason,
                "invalidated_at": _utc_now(),
                "chunks": canonical_json_value(chunks),
            }
            if invalidate_capture:
                attempt["capture_binding"] = record.pop("capture_binding", None)
                attempt["capture_invocations"] = canonical_json_value(
                    record.get("capture_invocations", [])
                )
                record["capture_invocations"] = []
            attempts.append(attempt)
            record["chunks"] = []
            self.save()

    def reconcile(self, fit_state: Mapping[str, Any]) -> None:
        changed = False
        for commit in fit_state.get("committed_prompts", []):
            index = int(commit["prompt_index"])
            record = self.state["prompts"].setdefault(
                str(index), {"chunks": [], "capture_invocations": []}
            )
            if "commit" not in record:
                record["commit"] = canonical_json_value(commit)
                record["commit_telemetry_recovered"] = True
                changed = True
        current = fit_state.get("current")
        current_index = (
            int(current["prompt_index"]) if isinstance(current, dict) else None
        )
        next_prompt = int(fit_state.get("next_prompt", 0))
        for index_text, record in self.state["prompts"].items():
            index = int(index_text)
            if (
                index >= next_prompt
                and index != current_index
                and "commit" not in record
                and record.get("chunks")
            ):
                chunks = record["chunks"]
                attempts = record.setdefault("invalidated_chunk_attempts", [])
                attempts.append(
                    {
                        "reason": "fit state has no matching current prompt",
                        "invalidated_at": _utc_now(),
                        "chunks": canonical_json_value(chunks),
                    }
                )
                record["chunks"] = []
                changed = True
        if isinstance(current, dict):
            index = int(current["prompt_index"])
            record = self.state["prompts"].setdefault(
                str(index), {"chunks": [], "capture_invocations": []}
            )
            existing = {
                (entry["start"], entry["stop"]): entry
                for entry in record["chunks"]
            }
            for chunk in current.get("chunks", []):
                key = (chunk["start"], chunk["stop"])
                if key not in existing:
                    record["chunks"].append(
                        {
                            **chunk,
                            "elapsed_seconds": None,
                            "cuda": None,
                            "telemetry_missing_due_to_crash": True,
                        }
                    )
                    changed = True
        if changed:
            self.save()

    def complete(self) -> None:
        self.state["status"] = "completed"
        self.state["completed_at"] = _utc_now()
        self.save()

    def provenance_record(self) -> dict[str, Any]:
        """Return final metadata that is invariant to completion bookkeeping."""

        return canonical_json_value(
            {
                "schema_version": self.state["schema_version"],
                "contract_sha256": self.state["contract_sha256"],
                "prompts": self.state["prompts"],
                "max_cuda_peak_allocated_bytes": self.state[
                    "max_cuda_peak_allocated_bytes"
                ],
                "max_cuda_peak_reserved_bytes": self.state[
                    "max_cuda_peak_reserved_bytes"
                ],
            }
        )


def _prompt_fit_record(
    prompt: Mapping[str, Any],
    binding: Mapping[str, Any],
    args: argparse.Namespace,
    spec: FitRunSpec,
) -> dict[str, Any]:
    return {
        "manifest_prompt": canonical_json_value(prompt),
        "capture": canonical_json_value(binding),
        "fit": {
            "source_layers": list(spec.source_layers),
            "target_layer": spec.target_layer,
            "skip_first": spec.skip_first,
            "cotangent_batch": args.cotangent_batch,
            "checkpoint_interval": spec.checkpoint_interval,
            "ste_policy": args.ste_policy,
        },
    }


def _capture_request(
    args: argparse.Namespace,
    prompt: Mapping[str, Any],
    index: int,
    spec: FitRunSpec,
) -> CaptureRequest:
    return CaptureRequest(
        prompt=prompt,
        prompt_index=index,
        output_dir=args.capture_dir / f"prompt-{index:02d}",
        python=args.python,
        capture_orchestrator=args.capture_orchestrator,
        runtime_capture_script=args.runtime_capture_script,
        proof_script=args.proof_script,
        prompt_manifest=args.prompt_manifest,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        hash_chunk_mib=args.hash_chunk_mib,
        capture_layers=spec.capture_layers,
    )


def _quarantine_capture_directory(directory: Path, index: int) -> Path | None:
    """Atomically move a rejected per-prompt capture out of the active path."""

    if not directory.exists():
        return None
    mode = directory.lstat().st_mode
    _require(stat.S_ISDIR(mode), f"capture path is not a real directory: {directory}")
    expected_name = f"prompt-{index:02d}"
    _require(directory.name == expected_name, "capture directory is not prompt-scoped")
    quarantine = directory.with_name(
        f"{expected_name}.rejected.{uuid.uuid4().hex}"
    )
    os.replace(directory, quarantine)
    descriptor = os.open(quarantine.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return quarantine


def _validate_factory(factory: Any, spec: FitRunSpec) -> None:
    if hasattr(factory, "validate_layer"):
        for layer in range(1, spec.target_layer + 1):
            factory.validate_layer(layer)


def _validate_checkpoint_before_commit(checkpoint: Any, *, required: bool) -> Any:
    callback = getattr(checkpoint, "validate_pinned_integrity", None)
    _require(
        callable(callback) or not required,
        "production checkpoint lacks validate_pinned_integrity",
    )
    return callback() if callable(callback) else None


def execute(
    args: argparse.Namespace,
    *,
    dependencies: RunnerDependencies | None = None,
    spec: FitRunSpec = PRODUCTION_SPEC,
    prompts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run or exactly resume a fit; injectable dependencies keep tests CPU-only."""

    production_dependencies = dependencies is None
    deps = default_dependencies() if production_dependencies else dependencies
    if prompts is None:
        prompts = load_pinned_prompts(args.prompt_manifest, spec=spec)
    prompts = canonical_json_value(prompts)
    _require(len(prompts) == spec.prompt_count, "prompt count differs from fit spec")
    _require(args.cotangent_batch > 0, "cotangent batch must be positive")
    _require(
        args.ste_policy == PRODUCTION_STE_POLICY,
        "production captures support identity STE only; clipped STE requires "
        "exact pre-linear inputs that are not retained",
    )
    _require(0 < args.gpu_memory_utilization < 1, "invalid GPU memory utilization")
    _require(args.hash_chunk_mib > 0, "hash chunk size must be positive")

    runtime = deps.runtime_provenance(args.device)
    contract = build_run_contract(args, prompts, runtime, spec=spec)
    contract_hash = canonical_sha256(contract)
    if args.plan_only:
        request = _capture_request(args, prompts[0], 0, spec)
        return {
            "schema_version": 1,
            "status": "planned",
            "contract": contract,
            "contract_sha256": contract_hash,
            "first_capture_command": capture_command(request, resume=False),
        }

    checkpoint = deps.make_checkpoint(args.snapshot)
    if production_dependencies:
        _require(
            callable(getattr(checkpoint, "validate_pinned_integrity", None)),
            "production checkpoint lacks validate_pinned_integrity",
        )
    layout = spec.fit_layout()
    store = (
        FitStateStore.resume(args.work_dir, contract, layout=layout)
        if args.resume
        else FitStateStore.create(args.work_dir, contract, layout=layout)
    )
    with store:
        progress = ProgressJournal.open(
            args.work_dir / "run-progress.json",
            contract_hash,
            resume=args.resume,
            fit_state=store.state,
        )
        valid_positions = tuple(range(spec.skip_first, spec.token_count - 1))
        while store.state["next_prompt"] < spec.prompt_count:
            current = store.state.get("current")
            index = (
                int(current["prompt_index"])
                if isinstance(current, dict)
                else int(store.state["next_prompt"])
            )
            prompt = prompts[index]
            capture_result = deps.capture_prompt(
                _capture_request(args, prompt, index, spec)
            )
            progress.record_capture(index, capture_result)
            prompt_record = _prompt_fit_record(
                prompt, capture_result.binding, args, spec
            )
            if current is None:
                store.begin_prompt(prompt_record)
            else:
                _require(
                    current.get("prompt") == prompt_record,
                    f"current prompt {index} capture/fit binding changed",
                )

            observer_binding = canonical_json_value(
                capture_result.binding["observer_tensors"]
            )
            _require(
                isinstance(observer_binding, dict),
                "observer artifact binding is not an object",
            )
            observer_path = Path(observer_binding["path"])
            _require(
                _file_record(observer_path) == observer_binding,
                "observer PT changed before load",
            )
            payload = deps.load_observer(observer_path)
            validate_observer_payload(payload, prompt, spec=spec)
            factory = deps.make_factory(
                payload,
                checkpoint,
                args.ste_policy,
                spec.checkpoint_interval,
            )
            _validate_factory(factory, spec)

            row_start = int(store.state["current"]["next_row"])
            while row_start < spec.hidden_size:
                row_stop = min(row_start + args.cotangent_batch, spec.hidden_size)
                deps.begin_chunk(args.device)
                started = deps.monotonic()
                rows = deps.compute_rows(
                    factory,
                    valid_positions,
                    row_start,
                    row_stop,
                    spec.target_layer,
                    args.device,
                )
                elapsed = deps.monotonic() - started
                cuda = deps.finish_chunk(args.device)
                normalized = normalize_rows(
                    rows,
                    row_start=row_start,
                    row_stop=row_stop,
                    spec=spec,
                )
                chunk = store.write_current_chunk(normalized, start=row_start)
                progress.record_chunk(
                    index,
                    chunk,
                    elapsed_seconds=elapsed,
                    cuda=cuda,
                )
                del rows, normalized
                row_start = row_stop

            try:
                _validate_bound_source_files(contract["source_files"])
                _validate_checkpoint_before_commit(
                    checkpoint,
                    required=production_dependencies,
                )
            except Exception as error:
                try:
                    progress.invalidate_uncommitted_chunks(
                        index,
                        f"{type(error).__name__}: {error}",
                    )
                finally:
                    store.discard_current_prompt()
                raise

            try:
                _require(
                    _file_record(observer_path) == observer_binding,
                    "observer PT changed before prompt commit",
                )
            except Exception as error:
                try:
                    progress.invalidate_uncommitted_chunks(
                        index,
                        f"{type(error).__name__}: {error}",
                        invalidate_capture=True,
                    )
                finally:
                    store.discard_current_prompt()
                if production_dependencies:
                    _quarantine_capture_directory(
                        _capture_request(args, prompt, index, spec).output_dir,
                        index,
                    )
                raise
            commit = store.commit_current_prompt()
            progress.record_commit(index, commit)
            del factory, payload
            deps.release_prompt()

        progress_provenance = progress.provenance_record()
        final_provenance = {
            "schema_version": 1,
            "fit_type": "Qwen3.6-27B NVFP4 exact-forward surrogate-backward J-lens",
            "contract": contract,
            "contract_sha256": contract_hash,
            "progress": progress_provenance,
            "progress_sha256": canonical_sha256(progress_provenance),
            "committed_prompts": store.state["committed_prompts"],
            "committed_prompts_sha256": canonical_sha256(
                store.state["committed_prompts"]
            ),
            "disclosure": {
                "forward": "exact deployed compiled NVFP4/FP8 observer capture",
                "backward": "packed W4 and live FP8 declared surrogate VJPs",
                "literal_derivative_of_quantized_rounding": False,
            },
        }
        artifact = store.finalize_means(final_provenance)
        progress.complete()
        return {
            "schema_version": 1,
            "status": "completed",
            "work_dir": str(args.work_dir.resolve()),
            "capture_dir": str(args.capture_dir.resolve()),
            "contract_sha256": contract_hash,
            "n_prompts": store.state["n_done"],
            "final_artifact": store.state["final_artifact"],
            "final_metadata_sha256": sha256_file(
                args.work_dir / "final-mean" / "metadata.json"
            ),
            "layer_aggregate_sha256": artifact["layer_aggregate_sha256"],
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_python = ROOT / ".venv-vllm" / "bin" / "python"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / ".cache" / "nvfp4_ste_fit",
    )
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--prompt-manifest", type=Path, default=PINNED_MANIFEST)
    parser.add_argument("--python", type=Path, default=default_python)
    parser.add_argument(
        "--capture-orchestrator",
        type=Path,
        default=ROOT / "scripts" / "capture_nvfp4_fit_prompt.py",
    )
    parser.add_argument(
        "--runtime-capture-script",
        type=Path,
        default=ROOT / "scripts" / "check_nvfp4_runtime_capture.py",
    )
    parser.add_argument(
        "--proof-script",
        type=Path,
        default=ROOT / "scripts" / "prove_nvfp4_capture_pair.py",
    )
    parser.add_argument(
        "--ste-policy",
        choices=(PRODUCTION_STE_POLICY,),
        default=PRODUCTION_STE_POLICY,
    )
    parser.add_argument("--cotangent-batch", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--hash-chunk-mib", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.snapshot is None:
        try:
            from scripts.modelopt_checkpoint import default_pinned_snapshot
        except ImportError:
            from modelopt_checkpoint import default_pinned_snapshot
        args.snapshot = default_pinned_snapshot()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.capture_dir = (
        args.capture_dir.expanduser().resolve()
        if args.capture_dir is not None
        else args.work_dir / "captures"
    )
    args.python = _absolute_without_symlink_resolution(args.python)
    for name in (
        "snapshot",
        "prompt_manifest",
        "capture_orchestrator",
        "runtime_capture_script",
        "proof_script",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def main() -> None:
    args = _parse_args()
    result = execute(args)
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
