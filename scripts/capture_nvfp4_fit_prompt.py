#!/usr/bin/env python3
"""Capture and prove one pinned NVFP4 fit prompt with crash-safe resume state."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
PINNED_MANIFEST = ROOT / "configs" / "jlens_nf4_fit_prompts.json"
PINNED_MANIFEST_SHA256 = (
    "2c36f17dee7287c096f7d1fdb7f8d7ecb8372c6cf8d13d7af8cfaed820439d3b"
)
PINNED_PROMPT_COUNT = 10
PINNED_TOKEN_COUNT = 128
MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
MODEL_IDENTITY_POLICY = "pinned-nvidia-modelopt-snapshot-v1"
PINNED_METADATA_SHA256 = {
    "config.json": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
    "hf_quant_config.json": "fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1",
    "model.safetensors.index.json": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
}
PINNED_SHARDS = {
    "model-00001-of-00003.safetensors": (
        9_965_652_512,
        "b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d",
    ),
    "model-00002-of-00003.safetensors": (
        9_985_757_032,
        "06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d",
    ),
    "model-00003-of-00003.safetensors": (
        1_970_287_640,
        "e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845",
    ),
}
EXPECTED_TARGET_LAYERS = list(range(64))
EXPECTED_REPLAY_PARAMETERS = 785
EXPECTED_BASELINE_TENSORS = 688
EXPECTED_OBSERVER_TENSORS = 1120
STATE_SCHEMA_VERSION = 1
ENVIRONMENT_OVERRIDES = {
    "VLLM_DISABLE_COMPILE_CACHE": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
}


class OrchestrationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OrchestrationError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _file_record(path: Path, *, chunk_bytes: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
            size += len(chunk)
    stat = path.stat()
    _require(size == stat.st_size, f"file changed while hashing: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "mtime_ns": stat.st_mtime_ns,
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON file is not an object: {path}")
    return value


def _load_pinned_prompt(path: Path, index: int) -> dict[str, Any]:
    resolved = path.resolve()
    _require(resolved.is_file(), f"prompt manifest does not exist: {resolved}")
    manifest_bytes = resolved.read_bytes()
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    _require(
        manifest_sha256 == PINNED_MANIFEST_SHA256,
        "prompt manifest does not match the pinned 10-prompt SHA-256",
    )
    manifest = json.loads(manifest_bytes)
    _require(
        isinstance(manifest, dict) and manifest.get("schema_version") == 1,
        "prompt manifest must use schema version 1",
    )
    prompts = manifest.get("prompts")
    _require(
        isinstance(prompts, list) and len(prompts) == PINNED_PROMPT_COUNT,
        f"prompt manifest must contain exactly {PINNED_PROMPT_COUNT} entries",
    )
    _require(0 <= index < len(prompts), f"prompt index {index} is out of range")
    for entry_index, entry in enumerate(prompts):
        _require(isinstance(entry, dict), f"prompt {entry_index} is not an object")
        text = entry.get("text")
        token_ids = entry.get("token_ids")
        _require(isinstance(text, str), f"prompt {entry_index} has no text")
        _require(
            isinstance(token_ids, list)
            and len(token_ids) == PINNED_TOKEN_COUNT
            and all(isinstance(token, int) and token >= 0 for token in token_ids),
            f"prompt {entry_index} does not contain 128 frozen token IDs",
        )
        _require(
            entry.get("token_count") == len(token_ids),
            f"prompt {entry_index} token count is inconsistent",
        )
        _require(
            entry.get("text_sha256")
            == _sha256_bytes(text.encode("utf-8")),
            f"prompt {entry_index} text hash is inconsistent",
        )
    entry = prompts[index]
    return {
        "manifest_path": str(resolved),
        "manifest_sha256": manifest_sha256,
        "manifest_prompt_count": len(prompts),
        "manifest_index": index,
        "row_index": entry.get("row_index"),
        "text": entry["text"],
        "text_sha256": entry["text_sha256"],
        "token_count": entry["token_count"],
        "token_ids": entry["token_ids"],
        "token_ids_sha256": _canonical_sha256(entry["token_ids"]),
    }


def _pinned_snapshot_path() -> Path:
    hf_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ).expanduser()
    return (
        hf_home
        / "hub"
        / "models--nvidia--Qwen3.6-27B-NVFP4"
        / "snapshots"
        / MODEL_REVISION
    ).resolve()


def _validate_pinned_snapshot(snapshot: Path) -> dict[str, Any]:
    resolved = snapshot.expanduser().resolve()
    expected = _pinned_snapshot_path()
    _require(resolved == expected, f"model snapshot is not the exact pin: {resolved}")
    _require(resolved.is_dir(), f"pinned model snapshot does not exist: {resolved}")
    actual_metadata: dict[str, str] = {}
    for filename, expected_sha256 in PINNED_METADATA_SHA256.items():
        path = resolved / filename
        _require(path.is_file(), f"required pinned model metadata is missing: {path}")
        actual_sha256 = _sha256_bytes(path.read_bytes())
        _require(
            actual_sha256 == expected_sha256,
            f"pinned model metadata hash mismatch for {filename}",
        )
        actual_metadata[filename] = actual_sha256
    for filename, (expected_bytes, expected_blob) in PINNED_SHARDS.items():
        path = resolved / filename
        _require(path.is_file(), f"required pinned model shard is missing: {path}")
        _require(
            path.stat().st_size == expected_bytes,
            f"pinned model shard size mismatch for {filename}",
        )
        if path.is_symlink():
            _require(
                Path(os.readlink(path)).name == expected_blob,
                f"pinned model shard blob mismatch for {filename}",
            )
    return {
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "resolved_path": str(resolved),
        "identity_policy": MODEL_IDENTITY_POLICY,
        "metadata_sha256": actual_metadata,
        "local_pinned_validation": True,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_python = ROOT / ".venv-vllm" / "bin" / "python"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--prompt-manifest", type=Path, default=PINNED_MANIFEST)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / ".cache" / "nvfp4_fit"
    )
    parser.add_argument("--python", type=Path, default=default_python)
    parser.add_argument(
        "--capture-script",
        type=Path,
        default=ROOT / "scripts" / "check_nvfp4_runtime_capture.py",
    )
    parser.add_argument(
        "--proof-script",
        type=Path,
        default=ROOT / "scripts" / "prove_nvfp4_capture_pair.py",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--capture-capacity", type=int, default=128)
    parser.add_argument("--hash-chunk-mib", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delete-baseline-pt-after-proof", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate inputs and print the frozen commands without writing state",
    )
    return parser.parse_args(argv)


def _artifact_paths(output_dir: Path, index: int) -> dict[str, Path]:
    prefix = output_dir.resolve() / f"prompt-{index:02d}"
    return {
        "state": prefix.with_name(f"{prefix.name}-capture-state.json"),
        "lock": prefix.with_name(f"{prefix.name}-capture.lock"),
        "baseline_json": prefix.with_name(f"{prefix.name}-compiled.json"),
        "baseline_tensors": prefix.with_name(f"{prefix.name}-compiled.pt"),
        "baseline_log": prefix.with_name(f"{prefix.name}-compiled.log"),
        "observer_json": prefix.with_name(f"{prefix.name}-compiled-observer.json"),
        "observer_tensors": prefix.with_name(f"{prefix.name}-compiled-observer.pt"),
        "observer_log": prefix.with_name(f"{prefix.name}-compiled-observer.log"),
        "proof": prefix.with_name(f"{prefix.name}-capture-proof.json"),
        "proof_log": prefix.with_name(f"{prefix.name}-capture-proof.log"),
    }


def _absolute_without_symlink_resolution(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (Path.cwd() / expanded).absolute()


def _capture_command(
    args: argparse.Namespace,
    prompt: dict[str, Any],
    paths: dict[str, Path],
    mode: str,
    model_path: Path,
) -> list[str]:
    label = "baseline" if mode == "compiled" else "observer"
    command = [
        str(_absolute_without_symlink_resolution(args.python)),
        str(args.capture_script.resolve()),
        "--mode",
        mode,
        "--prompt-manifest",
        prompt["manifest_path"],
        "--prompt-index",
        str(prompt["manifest_index"]),
        "--output",
        str(paths[f"{label}_json"]),
        "--tensor-output",
        str(paths[f"{label}_tensors"]),
        "--target-layers",
        "all",
        "--model-path",
        str(model_path),
        "--capture-capacity",
        str(args.capture_capacity),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]
    return command


def _proof_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    return [
        str(_absolute_without_symlink_resolution(args.python)),
        str(args.proof_script.resolve()),
        "--baseline-json",
        str(paths["baseline_json"]),
        "--baseline-tensors",
        str(paths["baseline_tensors"]),
        "--observer-json",
        str(paths["observer_json"]),
        "--observer-tensors",
        str(paths["observer_tensors"]),
        "--output",
        str(paths["proof"]),
        "--hash-chunk-mib",
        str(args.hash_chunk_mib),
    ]


def _command_record(argv: list[str], *, environment: dict[str, str]) -> dict[str, Any]:
    record = {"argv": argv, "environment_overrides": environment}
    return {**record, "sha256": _canonical_sha256(record)}


def _build_contract(
    args: argparse.Namespace,
    prompt: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    capture_script = _file_record(args.capture_script.resolve(), chunk_bytes=1024 * 1024)
    proof_script = _file_record(args.proof_script.resolve(), chunk_bytes=1024 * 1024)
    capture_script.pop("mtime_ns")
    proof_script.pop("mtime_ns")
    python_path = _absolute_without_symlink_resolution(args.python)
    _require(python_path.is_file(), f"Python interpreter does not exist: {python_path}")
    model_identity = _validate_pinned_snapshot(_pinned_snapshot_path())
    model_path = Path(model_identity["resolved_path"])
    commands = {
        "baseline": _command_record(
            _capture_command(args, prompt, paths, "compiled", model_path),
            environment=ENVIRONMENT_OVERRIDES,
        ),
        "observer": _command_record(
            _capture_command(args, prompt, paths, "compiled-observer", model_path),
            environment=ENVIRONMENT_OVERRIDES,
        ),
        "proof": _command_record(
            _proof_command(args, paths), environment={}
        ),
    }
    contract = {
        "prompt": prompt,
        "model": model_identity,
        "target_profile": "all",
        "target_layers": EXPECTED_TARGET_LAYERS,
        "runtime": {
            "capture_capacity": args.capture_capacity,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "mtp_enabled": False,
            "language_model_only": True,
        },
        "python": str(python_path),
        "capture_script": capture_script,
        "proof_script": proof_script,
        "paths": {
            name: str(path.resolve())
            for name, path in paths.items()
            if name not in {"state", "lock"}
        },
        "commands": commands,
    }
    return contract, commands


def _new_state(
    contract: dict[str, Any], commands: dict[str, Any], *, adopted: bool
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": "initialized",
        "created_at": now,
        "updated_at": now,
        "contract": contract,
        "contract_sha256": _canonical_sha256(contract),
        "commands": commands,
        "artifacts": {},
        "stages": {},
        "history": [],
        "adopted_existing_artifacts": adopted,
        "retention": {
            "baseline_tensors": "present",
            "delete_after_proof_requested": False,
        },
    }


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    _atomic_write_json(path, state)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OrchestrationError(f"capture is already locked: {path}") from exc
        yield


def _existing_stage_files(paths: dict[str, Path]) -> list[Path]:
    return [
        path
        for name, path in paths.items()
        if name not in {"state", "lock"} and path.exists()
    ]


def _validate_state_contract(state: dict[str, Any], contract: dict[str, Any]) -> None:
    _require(
        state.get("schema_version") == STATE_SCHEMA_VERSION,
        "unsupported capture state schema",
    )
    _require(
        state.get("contract_sha256") == _canonical_sha256(contract)
        and state.get("contract") == contract,
        "resume contract mismatch",
    )
    _require(
        state.get("commands") == contract.get("commands"),
        "resume command record mismatch",
    )


def _validate_capture_json(
    path: Path,
    *,
    mode: str,
    prompt: dict[str, Any],
    tensor_path: Path,
    capture_capacity: int,
) -> dict[str, Any]:
    metadata = _load_json(path)
    runtime = metadata.get("runtime", {})
    _require(runtime.get("mode") == mode, f"{path.name} mode mismatch")
    _require(runtime.get("target_profile") == "all", f"{path.name} is not all-layer")
    _require(
        runtime.get("target_layers") == EXPECTED_TARGET_LAYERS,
        f"{path.name} target layers mismatch",
    )
    _require(runtime.get("mtp_enabled") is False, f"{path.name} has MTP enabled")
    _require(
        runtime.get("language_model_only") is True,
        f"{path.name} is not main-model-only",
    )
    model = metadata.get("model", {})
    _require(
        model.get("repo_id") == MODEL_REPO
        and model.get("revision") == MODEL_REVISION,
        f"{path.name} model repository/revision mismatch",
    )
    identity = model.get("identity", {})
    _require(
        identity.get("policy") == MODEL_IDENTITY_POLICY
        and identity.get("repo_id") == MODEL_REPO
        and identity.get("revision") == MODEL_REVISION
        and identity.get("metadata_sha256") == PINNED_METADATA_SHA256
        and identity.get("strict_pinned_validation") is True
        and identity.get("validator")
        == "ModelOptCheckpoint(strict_pinned=True)",
        f"{path.name} lacks strict pinned model identity",
    )
    resolved_model_path = identity.get("resolved_path")
    _require(
        isinstance(resolved_model_path, str)
        and resolved_model_path == model.get("resolved_path"),
        f"{path.name} model path provenance mismatch",
    )
    snapshot_path = Path(resolved_model_path)
    _require(
        snapshot_path.resolve() == _pinned_snapshot_path(),
        f"{path.name} resolved model path is not the exact pinned snapshot",
    )
    prompt_record = metadata.get("prompt", {})
    provenance = prompt_record.get("provenance")
    _require(isinstance(provenance, dict), f"{path.name} has no prompt provenance")
    for key in (
        "manifest_sha256",
        "manifest_index",
        "row_index",
        "text_sha256",
        "token_count",
        "token_ids",
    ):
        _require(
            provenance.get(key) == prompt.get(key),
            f"{path.name} prompt provenance mismatch for {key}",
        )
    _require(
        Path(provenance["manifest_path"]).resolve()
        == Path(prompt["manifest_path"]).resolve(),
        f"{path.name} prompt manifest path mismatch",
    )
    _require(prompt_record.get("text") == prompt["text"], f"{path.name} text mismatch")
    _require(
        prompt_record.get("token_ids") == prompt["token_ids"],
        f"{path.name} token IDs mismatch",
    )
    capture = metadata.get("capture", {})
    _require(
        Path(capture.get("tensor_output", "")).resolve() == tensor_path.resolve(),
        f"{path.name} tensor-output path mismatch",
    )
    _require(
        capture.get("install", {}).get("capture_capacity") == capture_capacity,
        f"{path.name} capture capacity mismatch",
    )
    _require(
        len(capture.get("replay_parameter_provenance", {}))
        == EXPECTED_REPLAY_PARAMETERS,
        f"{path.name} replay parameter count mismatch",
    )
    expected_summaries = (
        EXPECTED_BASELINE_TENSORS
        if mode == "compiled"
        else EXPECTED_OBSERVER_TENSORS
    )
    _require(
        len(capture.get("tensor_summaries", {})) == expected_summaries,
        f"{path.name} tensor summary count mismatch",
    )
    if mode == "compiled":
        _require(
            isinstance(metadata.get("authoritative_compiled_generation"), dict),
            f"{path.name} has no authoritative generation",
        )
    else:
        _require(metadata.get("status") == "captured", f"{path.name} is not captured")
        _require(capture.get("missing_required") == [], f"{path.name} has required missing tensors")
        _require(capture.get("truncated") == [], f"{path.name} has truncated tensors")
    return metadata


def _validate_proof(
    path: Path,
    *,
    prompt: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    proof_script_sha256: str,
) -> dict[str, Any]:
    proof = _load_json(path)
    _require(proof.get("status") == "passed", "capture proof did not pass")
    _require(
        proof.get("claim", {}).get("mtp") == "off",
        "capture proof does not bind MTP off",
    )
    _require(
        proof.get("claim", {}).get("observer_graph_modified") is True
        and proof.get("claim", {}).get("observer_modification_discharged") is True,
        "capture proof does not discharge observer modification",
    )
    _require(
        proof.get("generation_record_parity", {}).get("exact") is True,
        "capture proof generation parity is not exact",
    )
    _require(
        proof.get("shared_internal_tensor_parity", {}).get("shared_tensor_count")
        == EXPECTED_BASELINE_TENSORS
        and proof.get("shared_internal_tensor_parity", {}).get(
            "all_shared_bit_exact"
        )
        is True,
        "capture proof internal parity mismatch",
    )
    _require(
        proof.get("replay_parameter_parity", {}).get("parameter_count")
        == EXPECTED_REPLAY_PARAMETERS
        and proof.get("replay_parameter_parity", {}).get(
            "all_content_hashes_equal"
        )
        is True,
        "capture proof replay parity mismatch",
    )
    completeness = proof.get("observer_capture_completeness", {})
    _require(
        completeness.get("required_missing") == []
        and completeness.get("truncated") == [],
        "capture proof observer completeness mismatch",
    )
    proof_prompt = proof.get("configuration", {}).get("prompt", {})
    proof_provenance = proof_prompt.get("provenance", {})
    _require(
        proof_provenance.get("manifest_sha256") == prompt["manifest_sha256"]
        and proof_provenance.get("manifest_index") == prompt["manifest_index"]
        and proof_provenance.get("token_ids") == prompt["token_ids"],
        "capture proof prompt identity mismatch",
    )
    for key in (
        "baseline_json",
        "baseline_tensors",
        "observer_json",
        "observer_tensors",
    ):
        proof_record = proof.get("artifacts", {}).get(key, {})
        expected = artifacts[key]
        _require(
            proof_record.get("sha256") == expected["sha256"]
            and proof_record.get("bytes") == expected["bytes"],
            f"capture proof does not bind {key}",
        )
        _require(
            Path(proof_record.get("path", "")).resolve()
            == Path(expected["path"]).resolve(),
            f"capture proof path mismatch for {key}",
        )
    _require(
        proof.get("verifier", {}).get("sha256") == proof_script_sha256,
        "capture proof verifier source hash mismatch",
    )
    return proof


def _record_artifact(
    state: dict[str, Any], key: str, path: Path, *, chunk_bytes: int
) -> dict[str, Any]:
    current = _file_record(path, chunk_bytes=chunk_bytes)
    previous = state["artifacts"].get(key)
    if previous is not None:
        _require(
            previous.get("sha256") == current["sha256"]
            and previous.get("bytes") == current["bytes"],
            f"recorded artifact changed: {key}",
        )
    state["artifacts"][key] = current
    return current


def _stage_presence(paths: dict[str, Path], names: tuple[str, ...]) -> tuple[bool, ...]:
    return tuple(paths[name].exists() for name in names)


def _require_complete_or_absent(
    paths: dict[str, Path], names: tuple[str, ...], label: str
) -> bool:
    presence = _stage_presence(paths, names)
    if any(presence) and not all(presence):
        existing = [name for name, present in zip(names, presence) if present]
        missing = [name for name, present in zip(names, presence) if not present]
        raise OrchestrationError(
            f"partial {label} artifacts fail closed; existing={existing}, missing={missing}"
        )
    return all(presence)


def _mark_stage(
    state: dict[str, Any], stage: str, status: str, **extra: Any
) -> None:
    record = state["stages"].setdefault(stage, {})
    record.update({"status": status, "updated_at": _utc_now(), **extra})
    state["status"] = f"{stage}_{status}"
    state["history"].append(
        {"stage": stage, "status": status, "at": _utc_now(), **extra}
    )


def _run_command(
    *,
    state: dict[str, Any],
    state_path: Path,
    stage: str,
    command: dict[str, Any],
    log_path: Path,
) -> None:
    _require(not log_path.exists(), f"refusing to overwrite stage log: {log_path}")
    _mark_stage(state, stage, "running", command_sha256=command["sha256"])
    _save_state(state_path, state)
    environment = os.environ.copy()
    environment.update(command["environment_overrides"])
    with log_path.open("xb") as log:
        completed = subprocess.run(
            command["argv"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode != 0:
        _mark_stage(state, stage, "failed", returncode=completed.returncode)
        _save_state(state_path, state)
        raise subprocess.CalledProcessError(completed.returncode, command["argv"])


def _verify_baseline(
    state: dict[str, Any],
    paths: dict[str, Path],
    prompt: dict[str, Any],
    *,
    capture_capacity: int,
    chunk_bytes: int,
    allow_deleted: bool,
) -> bool:
    core_present = _stage_presence(paths, ("baseline_json", "baseline_tensors"))
    deleted = state.get("retention", {}).get("baseline_tensors") == "deleted_after_proof"
    deletion_pending = state.get("retention", {}).get(
        "delete_after_proof_requested"
    ) is True
    if paths["baseline_json"].exists() and not paths["baseline_tensors"].exists():
        _require(
            allow_deleted and (deleted or deletion_pending) and paths["proof"].exists(),
            "partial baseline artifacts: PT is absent without a proved deletion record",
        )
        _record_artifact(state, "baseline_json", paths["baseline_json"], chunk_bytes=chunk_bytes)
        _require("baseline_tensors" in state["artifacts"], "deleted baseline PT has no recorded hash")
        return True
    if any(core_present) and not all(core_present):
        raise OrchestrationError("partial baseline JSON/PT artifacts fail closed")
    if not all(core_present):
        _require(not paths["baseline_log"].exists(), "orphan baseline log fails closed")
        return False
    _validate_capture_json(
        paths["baseline_json"],
        mode="compiled",
        prompt=prompt,
        tensor_path=paths["baseline_tensors"],
        capture_capacity=capture_capacity,
    )
    _record_artifact(state, "baseline_json", paths["baseline_json"], chunk_bytes=chunk_bytes)
    _record_artifact(state, "baseline_tensors", paths["baseline_tensors"], chunk_bytes=chunk_bytes)
    if paths["baseline_log"].exists():
        _record_artifact(state, "baseline_log", paths["baseline_log"], chunk_bytes=chunk_bytes)
    return True


def _verify_observer(
    state: dict[str, Any],
    paths: dict[str, Path],
    prompt: dict[str, Any],
    *,
    capture_capacity: int,
    chunk_bytes: int,
) -> bool:
    complete = _require_complete_or_absent(
        paths, ("observer_json", "observer_tensors"), "observer"
    )
    if not complete:
        _require(not paths["observer_log"].exists(), "orphan observer log fails closed")
        return False
    _validate_capture_json(
        paths["observer_json"],
        mode="compiled-observer",
        prompt=prompt,
        tensor_path=paths["observer_tensors"],
        capture_capacity=capture_capacity,
    )
    _record_artifact(state, "observer_json", paths["observer_json"], chunk_bytes=chunk_bytes)
    _record_artifact(state, "observer_tensors", paths["observer_tensors"], chunk_bytes=chunk_bytes)
    if paths["observer_log"].exists():
        _record_artifact(state, "observer_log", paths["observer_log"], chunk_bytes=chunk_bytes)
    return True


def _verify_existing_proof(
    state: dict[str, Any],
    paths: dict[str, Path],
    prompt: dict[str, Any],
    *,
    proof_script_sha256: str,
    chunk_bytes: int,
) -> bool:
    if not paths["proof"].exists():
        _require(not paths["proof_log"].exists(), "orphan proof log fails closed")
        return False
    required = {
        key: state["artifacts"].get(key)
        for key in (
            "baseline_json",
            "baseline_tensors",
            "observer_json",
            "observer_tensors",
        )
    }
    _require(all(required.values()), "proof inputs do not all have recorded hashes")
    _validate_proof(
        paths["proof"],
        prompt=prompt,
        artifacts=required,
        proof_script_sha256=proof_script_sha256,
    )
    _record_artifact(state, "proof", paths["proof"], chunk_bytes=chunk_bytes)
    if paths["proof_log"].exists():
        _record_artifact(state, "proof_log", paths["proof_log"], chunk_bytes=chunk_bytes)
    return True


def _delete_baseline_after_proof(
    state: dict[str, Any], state_path: Path, paths: dict[str, Path]
) -> None:
    retention = state["retention"]
    if retention.get("baseline_tensors") == "deleted_after_proof":
        _require(not paths["baseline_tensors"].exists(), "deleted baseline PT reappeared")
        return
    if (
        retention.get("delete_after_proof_requested") is True
        and not paths["baseline_tensors"].exists()
    ):
        retention["baseline_tensors"] = "deleted_after_proof"
        retention["deleted_at"] = _utc_now()
        _save_state(state_path, state)
        return
    _require("proof" in state["artifacts"], "baseline PT deletion requires a proof")
    _require(paths["baseline_tensors"].is_file(), "baseline PT is already absent")
    retention["delete_after_proof_requested"] = True
    retention["baseline_tensor_record"] = state["artifacts"]["baseline_tensors"]
    retention["deletion_requested_at"] = _utc_now()
    _save_state(state_path, state)
    paths["baseline_tensors"].unlink()
    directory_fd = os.open(paths["baseline_tensors"].parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    retention["baseline_tensors"] = "deleted_after_proof"
    retention["deleted_at"] = _utc_now()
    _save_state(state_path, state)


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    _require(args.hash_chunk_mib > 0, "hash chunk size must be positive")
    _require(args.capture_capacity == PINNED_TOKEN_COUNT, "capture capacity must be exactly 128")
    _require(0 < args.gpu_memory_utilization < 1, "GPU memory utilization is invalid")
    prompt = _load_pinned_prompt(args.prompt_manifest, args.prompt_index)
    paths = _artifact_paths(args.output_dir, args.prompt_index)
    contract, commands = _build_contract(args, prompt, paths)
    if args.plan_only:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "status": "planned",
            "contract_sha256": _canonical_sha256(contract),
            "prompt": prompt,
            "commands": commands,
            "paths": {name: str(path) for name, path in paths.items()},
            "environment_overrides": ENVIRONMENT_OVERRIDES,
        }

    paths["state"].parent.mkdir(parents=True, exist_ok=True)
    chunk_bytes = args.hash_chunk_mib * 1024 * 1024
    with _exclusive_lock(paths["lock"]):
        existing = _existing_stage_files(paths)
        state_exists = paths["state"].exists()
        if not args.resume:
            if state_exists or existing:
                raise FileExistsError(
                    "capture state/artifacts already exist; use --resume for verification"
                )
            state = _new_state(contract, commands, adopted=False)
            _save_state(paths["state"], state)
        else:
            _require(
                state_exists or bool(existing),
                "--resume requires existing state or artifacts",
            )
            if state_exists:
                state = _load_json(paths["state"])
                _validate_state_contract(state, contract)
            else:
                state = _new_state(contract, commands, adopted=True)

        baseline_complete = _verify_baseline(
            state,
            paths,
            prompt,
            capture_capacity=args.capture_capacity,
            chunk_bytes=chunk_bytes,
            allow_deleted=args.resume,
        )
        if not baseline_complete:
            _run_command(
                state=state,
                state_path=paths["state"],
                stage="baseline",
                command=commands["baseline"],
                log_path=paths["baseline_log"],
            )
            _require(
                _verify_baseline(
                    state,
                    paths,
                    prompt,
                    capture_capacity=args.capture_capacity,
                    chunk_bytes=chunk_bytes,
                    allow_deleted=False,
                ),
                "baseline command did not produce a complete artifact pair",
            )
            _mark_stage(state, "baseline", "complete")
            _save_state(paths["state"], state)
        elif state.get("stages", {}).get("baseline", {}).get("status") != "complete":
            _mark_stage(state, "baseline", "complete", resumed_or_adopted=True)
            _save_state(paths["state"], state)

        observer_complete = _verify_observer(
            state,
            paths,
            prompt,
            capture_capacity=args.capture_capacity,
            chunk_bytes=chunk_bytes,
        )
        if not observer_complete:
            _run_command(
                state=state,
                state_path=paths["state"],
                stage="observer",
                command=commands["observer"],
                log_path=paths["observer_log"],
            )
            _require(
                _verify_observer(
                    state,
                    paths,
                    prompt,
                    capture_capacity=args.capture_capacity,
                    chunk_bytes=chunk_bytes,
                ),
                "observer command did not produce a complete artifact pair",
            )
            _mark_stage(state, "observer", "complete")
            _save_state(paths["state"], state)
        elif state.get("stages", {}).get("observer", {}).get("status") != "complete":
            _mark_stage(state, "observer", "complete", resumed_or_adopted=True)
            _save_state(paths["state"], state)

        proof_complete = _verify_existing_proof(
            state,
            paths,
            prompt,
            proof_script_sha256=contract["proof_script"]["sha256"],
            chunk_bytes=chunk_bytes,
        )
        if not proof_complete:
            _require(
                paths["baseline_tensors"].is_file(),
                "cannot prove pair after baseline PT deletion",
            )
            _run_command(
                state=state,
                state_path=paths["state"],
                stage="proof",
                command=commands["proof"],
                log_path=paths["proof_log"],
            )
            _require(
                _verify_existing_proof(
                    state,
                    paths,
                    prompt,
                    proof_script_sha256=contract["proof_script"]["sha256"],
                    chunk_bytes=chunk_bytes,
                ),
                "proof command did not produce a passing proof",
            )
            _mark_stage(state, "proof", "complete")
            _save_state(paths["state"], state)
        elif state.get("stages", {}).get("proof", {}).get("status") != "complete":
            _mark_stage(state, "proof", "complete", resumed_or_adopted=True)
            _save_state(paths["state"], state)

        if (
            args.delete_baseline_pt_after_proof
            or state["retention"].get("delete_after_proof_requested") is True
        ):
            _delete_baseline_after_proof(state, paths["state"], paths)
        state["status"] = "complete"
        state["completed_at"] = state.get("completed_at", _utc_now())
        _save_state(paths["state"], state)
        return state


def main() -> None:
    args = _parse_args()
    result = _execute(args)
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
