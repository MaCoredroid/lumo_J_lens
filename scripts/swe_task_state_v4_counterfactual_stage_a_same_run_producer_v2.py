#!/usr/bin/env python3
"""Frozen Stage-A V2 same-request producer core.

The module intentionally does not contain a vLLM backend and has no executable
GPU command.  A separately reviewed backend must implement exactly one
``run_same_autoregressive_request`` call per prompt.  That call must return the
prompt-tail activations captured during the prefill which created the cache
used by the same request's 256 decode steps.  This core validates that result,
writes one tensor shard and one JSON receipt per request, and never accepts a
separate capture/replay result.

CPU tests use an injected mock backend.  Mock output is always marked
``injected_test_non_gate`` and cannot satisfy the production verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# This source freeze contains no real backend/launcher/normalizer.  A caller
# cannot turn production on by mutating a config object; a new versioned code
# freeze must deliberately change this sentinel after adding those sources.
REAL_PRODUCTION_CODE_FREEZE_COMPLETE = False

FALSE_CLAIMS = {
    "private_chain_of_thought_reconstructed": False,
    "cot_or_cot_like_decoding_established": False,
    "latent_cot_like_concept_trajectory_established": False,
    "subjective_confidence_inferred": False,
    "subjective_doubt_inferred": False,
    "experienced_stress_inferred": False,
    "experienced_emotion_inferred": False,
    "causal_affect_or_state_effect_established": False,
    "incremental_activation_readout_established": False,
    "outer_or_reserved_validation_generalization_established": False,
}


class SameRunProducerError(ValueError):
    """Raised before incomplete or ambiguous same-run evidence is written."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SameRunProducerError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    return sha256_bytes(canonical_json_bytes(list(token_ids)))


def file_binding(path: Path, *, display_root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved.is_file(), f"bound path is not a file: {path}")
    display = str(resolved)
    if display_root is not None:
        try:
            display = str(resolved.relative_to(display_root.resolve()))
        except ValueError:
            pass
    return {
        "path": display,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verify_exact_file_binding(value: Any, *, label: str) -> tuple[dict[str, Any], Path]:
    _require(isinstance(value, Mapping), f"{label} binding is absent")
    binding = dict(value)
    _require(set(binding) == {"path", "sha256", "size_bytes"}, f"{label} binding fields changed")
    _require(
        isinstance(binding["path"], str)
        and bool(binding["path"])
        and isinstance(binding["sha256"], str)
        and SHA256_RE.fullmatch(binding["sha256"]) is not None
        and isinstance(binding["size_bytes"], int)
        and not isinstance(binding["size_bytes"], bool)
        and binding["size_bytes"] > 0,
        f"{label} binding is invalid",
    )
    raw = Path(binding["path"])
    path = raw if raw.is_absolute() else ROOT / raw
    forbidden = (
        "reserved", "validation", "split-manifest", "condition-key", "stage-b",
        "semantic-answer", "expectation", "task-outcome", "visible-state-annotation",
    )
    normalized = Path(os.path.normpath(os.path.abspath(os.fspath(path))))
    _require(
        not any(fragment in part.lower() for part in normalized.parts for fragment in forbidden),
        f"{label} path is forbidden",
    )
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SameRunProducerError(f"{label} is unavailable: {error}") from error
    _require(resolved.is_file(), f"{label} must be a regular file")
    _require(
        resolved.stat().st_size == binding["size_bytes"]
        and sha256_file(resolved) == binding["sha256"],
        f"{label} bytes changed",
    )
    return binding, resolved


def _require_real_production_authorization(
    config: Mapping[str, Any], *, backend: SameAutoregressiveBackend | None = None
) -> dict[str, Any]:
    """Fail before output/backend use unless a future exact freeze authorizes it."""

    authorization = config.get("authorization")
    _require(isinstance(authorization, Mapping), "real producer authorization is absent")
    required_true = (
        "real_gpu_producer_authorized_by_this_config",
        "real_backend_implementation_present",
        "real_traced_launcher_present",
        "raw_strace_normalizer_present",
        "raw_vs_normalized_trace_equivalence_established",
        "filesystem_alias_closure_established",
        "real_final_verification_receipt_emission_authorized",
    )
    _require(
        all(authorization.get(key) is True for key in required_true),
        "real same-run producer is not authorized by this frozen config",
    )
    # This immutable source-level sentinel precedes implementation lookup or
    # path handling.  Mutating config flags/bindings cannot induce arbitrary
    # filesystem reads from the current prospective producer.
    _require(
        REAL_PRODUCTION_CODE_FREEZE_COMPLETE is True,
        "real production requires a new versioned producer code freeze",
    )
    implementation = config.get("implementation")
    _require(isinstance(implementation, Mapping), "implementation contract is absent")
    backend_binding, _path = _verify_exact_file_binding(
        implementation.get("production_backend"), label="production backend source"
    )
    _verify_exact_file_binding(
        implementation.get("real_traced_launcher"), label="real traced launcher source"
    )
    _verify_exact_file_binding(
        implementation.get("raw_strace_normalizer"), label="raw strace normalizer source"
    )
    if backend is not None:
        _require(
            dict(backend.implementation_binding) == backend_binding,
            "backend implementation does not equal the frozen source binding",
        )
    return backend_binding


def _verify_pre_generation_receipt(
    value: Any,
    *,
    label: str,
    expected_kind: str,
    expected_status: str,
    runtime_config_sha256: str,
) -> dict[str, Any]:
    binding, path = _verify_exact_file_binding(value, label=label)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SameRunProducerError(f"cannot parse {label}: {error}") from error
    _require(isinstance(receipt, dict), f"{label} content is invalid")
    _require(
        receipt.get("kind") == expected_kind
        and receipt.get("status") == expected_status,
        f"{label} identity changed",
    )
    observed_config_sha = (
        receipt.get("runtime_config_sha256")
        if "runtime_config_sha256" in receipt
        else receipt.get("runtime_config", {}).get("sha256")
    )
    _require(
        observed_config_sha == runtime_config_sha256,
        f"{label} does not bind the exact runtime config",
    )
    return binding


def _atomic_json(path: Path, value: Any) -> None:
    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _require(
        not temporary.exists() and not temporary.is_symlink(),
        f"temporary output exists: {temporary}",
    )
    try:
        temporary.write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _logical_array_sha256(
    array: Any,
    *,
    name: str,
    expected_shape: Sequence[int],
    layer_ids: Sequence[int] | None = None,
) -> str:
    import numpy as np

    _require(isinstance(array, np.ndarray), f"{name} is not an ndarray")
    _require(list(array.shape) == list(expected_shape), f"{name} shape changed")
    _require(array.dtype == np.dtype("float32"), f"{name} dtype changed")
    _require(bool(np.isfinite(array).all()), f"{name} contains non-finite values")
    little = np.asarray(array, dtype="<f4", order="C")
    header: dict[str, Any] = {
        "name": name,
        "shape": list(expected_shape),
        "dtype": "little-endian-float32",
    }
    if layer_ids is not None:
        header["layer_ids"] = list(layer_ids)
    rendered = canonical_json_bytes(header)
    digest = hashlib.sha256()
    digest.update(len(rendered).to_bytes(8, "big"))
    digest.update(rendered)
    digest.update(little.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class SameRunRequest:
    index: int
    prompt_id: str
    token_ids: tuple[int, ...]


class SameAutoregressiveBackend(Protocol):
    """One-call backend boundary; capture and decode cannot be separate calls."""

    backend_kind: str
    implementation_binding: Mapping[str, Any]
    model_instance_id: str
    execution_provenance: Mapping[str, Any]

    def run_same_autoregressive_request(
        self,
        *,
        request: SameRunRequest,
        sampling: Mapping[str, Any],
        capture: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return one prompt-prefill capture and its same-cache decode result."""


def _validate_identifier(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 identifier",
    )
    return value


def _validate_token_ids(
    value: Any, *, label: str, vocabulary_size: int, exact_count: int | None = None
) -> list[int]:
    _require(isinstance(value, list), f"{label} must be a list")
    if exact_count is not None:
        _require(len(value) == exact_count, f"{label} count changed")
    _require(
        all(
            isinstance(item, int)
            and not isinstance(item, bool)
            and 0 <= item < vocabulary_size
            for item in value
        ),
        f"{label} contains invalid token IDs",
    )
    return list(value)


def _validate_backend_result(
    raw: Mapping[str, Any],
    *,
    request: SameRunRequest,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one indivisible online-prefill-plus-decode backend result."""

    import numpy as np

    expected_keys = {
        "request_id",
        "model_instance_id",
        "prefill_forward_id",
        "kv_cache_sequence_id",
        "submitted_prompt_token_ids",
        "engine_returned_prompt_token_ids",
        "prompt_tail_position_id",
        "prompt_tail_cache_write_index",
        "first_decode_cache_length",
        "capture_completed_monotonic_ns",
        "first_decode_started_monotonic_ns",
        "same_backend_call",
        "capture_was_online_in_prefill",
        "generation_consumed_same_kv_cache",
        "replay_between_capture_and_first_decode",
        "all_layer_residual",
        "raw_residual",
        "public_j_state",
        "prompt_tail_final_state",
        "prompt_tail_logits",
        "generated_token_ids",
        "completion_text",
        "finish_reason",
        "stop_reason",
        "engine_sampling",
        "final_norm_reconstruction_within_tolerance",
        "final_logits_reconstruction_within_tolerance",
    }
    _require(set(raw) == expected_keys, "backend result fields changed")
    runtime = config["runtime"]
    capture = config["capture"]
    sampling = config["sampling"]
    model = config["model"]
    submitted = _validate_token_ids(
        raw["submitted_prompt_token_ids"],
        label="submitted prompt token IDs",
        vocabulary_size=model["vocabulary_size"],
    )
    engine = _validate_token_ids(
        raw["engine_returned_prompt_token_ids"],
        label="engine-returned prompt token IDs",
        vocabulary_size=model["vocabulary_size"],
    )
    _require(
        submitted == list(request.token_ids) and engine == submitted,
        "submitted, frozen, and engine-returned prompt token IDs differ",
    )
    tail = len(submitted) - 1
    _require(
        raw["prompt_tail_position_id"] == tail
        and raw["prompt_tail_cache_write_index"] == tail
        and raw["first_decode_cache_length"] == len(submitted),
        "prompt-tail position or cache continuation changed",
    )
    _require(
        raw["same_backend_call"] is True
        and raw["capture_was_online_in_prefill"] is True
        and raw["generation_consumed_same_kv_cache"] is True
        and raw["replay_between_capture_and_first_decode"] is False,
        "capture and generation are not one online autoregressive request",
    )
    capture_ns = raw["capture_completed_monotonic_ns"]
    decode_ns = raw["first_decode_started_monotonic_ns"]
    _require(
        isinstance(capture_ns, int)
        and not isinstance(capture_ns, bool)
        and isinstance(decode_ns, int)
        and not isinstance(decode_ns, bool)
        and 0 <= capture_ns <= decode_ns,
        "capture/decode monotonic chronology changed",
    )
    for name in (
        "request_id",
        "model_instance_id",
        "prefill_forward_id",
        "kv_cache_sequence_id",
    ):
        _validate_identifier(raw[name], name)
    selected_layers = capture["layers"]
    hidden = model["hidden_size"]
    vocab = model["vocabulary_size"]
    tensors = {
        "all_layer_residual": raw["all_layer_residual"],
        "raw_residual": raw["raw_residual"],
        "public_j_state": raw["public_j_state"],
        "prompt_tail_final_state": raw["prompt_tail_final_state"],
        "prompt_tail_logits": raw["prompt_tail_logits"],
    }
    shapes = {
        "all_layer_residual": [model["layer_count"], hidden],
        "raw_residual": [len(selected_layers), hidden],
        "public_j_state": [len(selected_layers), hidden],
        "prompt_tail_final_state": [hidden],
        "prompt_tail_logits": [vocab],
    }
    layer_ids = {
        "all_layer_residual": list(range(model["layer_count"])),
        "raw_residual": selected_layers,
        "public_j_state": selected_layers,
        "prompt_tail_final_state": None,
        "prompt_tail_logits": None,
    }
    hashes = {
        name: _logical_array_sha256(
            value,
            name=name,
            expected_shape=shapes[name],
            layer_ids=layer_ids[name],
        )
        for name, value in tensors.items()
    }
    selected = np.asarray(
        tensors["all_layer_residual"][selected_layers, :], dtype=np.float32
    )
    _require(
        np.array_equal(selected, tensors["raw_residual"]),
        "selected raw residual is not the exact slice of all-layer state",
    )
    generated = _validate_token_ids(
        raw["generated_token_ids"],
        label="generated token IDs",
        vocabulary_size=vocab,
        exact_count=sampling["max_new_tokens"],
    )
    _require(
        sampling["min_new_tokens"] == sampling["max_new_tokens"] == 256
        and sampling["ignore_eos"] is True
        and raw["finish_reason"] == "length"
        and raw["stop_reason"] is None,
        "generation did not end only at the exact 256-token length boundary",
    )
    _require(raw["engine_sampling"] == sampling, "engine sampling record changed")
    logits = tensors["prompt_tail_logits"]
    prompt_tail_argmax = int(np.argmax(logits))
    _require(
        generated[0] == prompt_tail_argmax,
        "first generated token differs from captured prompt-tail logits argmax",
    )
    _require(
        raw["final_norm_reconstruction_within_tolerance"] is True
        and raw["final_logits_reconstruction_within_tolerance"] is True,
        "prompt-tail forward reconstruction canary failed",
    )
    _require(isinstance(raw["completion_text"], str), "completion text is invalid")
    _require(
        len(submitted) + len(generated) <= runtime["max_model_len"],
        "same-run request exceeds frozen context",
    )
    normalized = {
        key: raw[key]
        for key in expected_keys
        if key not in tensors
    }
    normalized["submitted_prompt_token_ids"] = submitted
    normalized["engine_returned_prompt_token_ids"] = engine
    normalized["generated_token_ids"] = generated
    normalized["prompt_tail_logits_argmax_token_id"] = prompt_tail_argmax
    normalized["tensor_logical_sha256"] = hashes
    return normalized, tensors


def _validate_execution_provenance(
    value: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    backend_mode: str,
    model_instance_id: str,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "backend execution provenance missing")
    observed = dict(value)
    if backend_mode == "injected_test_non_gate":
        _require(observed == {"mode": "injected_test_non_gate"}, "mock provenance changed")
        return observed
    _exact = {
        "model", "tokenizer", "runtime", "sanitized_environment",
        "model_instance_id", "model_loaded_once", "engine_seed",
        "started_at", "completed_at",
    }
    _require(set(observed) == _exact, "production execution provenance fields changed")
    model = config["model"]
    _require(
        observed["model"] == {
            "repo_id": model["repo_id"],
            "revision": model["revision"],
            "snapshot_tree_sha256_before": model["snapshot_tree_sha256"],
            "snapshot_tree_sha256_after": model["snapshot_tree_sha256"],
            "snapshot_file_count": model["snapshot_file_count"],
            "snapshot_size_bytes": model["snapshot_size_bytes"],
            "checkpoint_rehashed_before_model_load": True,
            "checkpoint_rehashed_after_engine_shutdown": True,
        },
        "production model execution provenance changed",
    )
    tokenizer = config["tokenizer"]
    _require(
        observed["tokenizer"] == {
            "tokenizer_json_sha256": tokenizer["tokenizer_json_sha256"],
            "tokenizer_config_sha256": tokenizer["tokenizer_config_sha256"],
            "generation_config_sha256": tokenizer["generation_config_sha256"],
            "same_snapshot_as_model": True,
        },
        "production tokenizer execution provenance changed",
    )
    _require(observed["runtime"] == config["runtime"], "production runtime/cache provenance changed")
    _require(
        observed["sanitized_environment"] == config["environment"]["exact_values"],
        "production sanitized environment provenance changed",
    )
    _require(
        observed["model_instance_id"] == model_instance_id
        and observed["model_loaded_once"] is True
        and observed["engine_seed"] == 0
        and isinstance(observed["started_at"], str)
        and bool(observed["started_at"])
        and isinstance(observed["completed_at"], str)
        and bool(observed["completed_at"]),
        "production model instance/load/seed chronology changed",
    )
    return observed


def _write_tensor_shard(
    path: Path,
    *,
    tensors: Mapping[str, Any],
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    import numpy as np
    from safetensors.numpy import save_file

    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _require(not temporary.exists(), f"temporary shard exists: {temporary}")
    try:
        save_file(
            {
                name: np.asarray(value, dtype="<f4", order="C")
                for name, value in tensors.items()
            },
            temporary,
            metadata=dict(metadata),
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    binding = file_binding(path, display_root=path.parent.parent)
    binding["tensor_keys"] = sorted(tensors)
    return binding


def produce_same_run_bundle(
    *,
    config: Mapping[str, Any],
    runtime_config_sha256: str,
    prompt_rows: Sequence[Mapping[str, Any]],
    backend: SameAutoregressiveBackend,
    output_root: Path,
    preflight_receipt: Mapping[str, Any],
    capture_lock_receipt: Mapping[str, Any],
    backend_mode: str,
) -> dict[str, Any]:
    """Run the indivisible backend boundary once for every ordered prompt row.

    This function is deliberately dependency-injected.  Production verification
    accepts only ``real_local_pinned_vllm_online_capture``; the CPU test mode is
    permanently non-gate even if all mock values are internally perfect.
    """

    _require(SHA256_RE.fullmatch(runtime_config_sha256) is not None, "config hash invalid")
    _require(
        backend_mode in {"real_local_pinned_vllm_online_capture", "injected_test_non_gate"},
        "backend mode is invalid",
    )
    if backend_mode == "real_local_pinned_vllm_online_capture":
        # Authorization is checked from the frozen config before touching the
        # backend object, receipts, output path, prompt rows, or model runtime.
        # The current V2 config deliberately fails this gate.
        _require_real_production_authorization(config)
        _require(
            backend.backend_kind == backend_mode,
            "production backend identity changed",
        )
        _require_real_production_authorization(config, backend=backend)
        preflight_receipt = _verify_pre_generation_receipt(
            preflight_receipt,
            label="immutable preflight receipt",
            expected_kind=config["pre_generation_receipts"]["preflight_kind"],
            expected_status="passed_cpu_preflight_no_model_or_gpu_runtime",
            runtime_config_sha256=runtime_config_sha256,
        )
        capture_lock_receipt = _verify_pre_generation_receipt(
            capture_lock_receipt,
            label="immutable pre-generation capture lock",
            expected_kind=config["pre_generation_receipts"]["capture_lock_kind"],
            expected_status="passed_pre_generation_capture_and_reference_content_lock",
            runtime_config_sha256=runtime_config_sha256,
        )
    else:
        _require(
            backend.backend_kind == "injected_mock_same_run_backend",
            "test mode requires the explicit mock backend identity",
        )
    execution_provenance = _validate_execution_provenance(
        backend.execution_provenance,
        config=config,
        backend_mode=backend_mode,
        model_instance_id=backend.model_instance_id,
    )
    _require(not output_root.exists() and not output_root.is_symlink(), "output root exists")
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "shards").mkdir()
    (output_root / "records").mkdir()
    producer_binding = file_binding(SCRIPT_PATH, display_root=ROOT)
    expected_binding = config["implementation"]["producer"]
    _require(producer_binding == expected_binding, "producer source binding changed")
    _require(
        len(prompt_rows) == config["generation_prompt_count"],
        "generation prompt count changed",
    )
    records: list[dict[str, Any]] = []
    completion_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_prompt in enumerate(prompt_rows):
        _require(set(raw_prompt) == {"id", "token_ids"}, f"prompt row {index} fields changed")
        prompt_id = _validate_identifier(raw_prompt["id"], f"prompt row {index} ID")
        _require(prompt_id not in seen, f"prompt row {index} ID duplicated")
        seen.add(prompt_id)
        token_ids = _validate_token_ids(
            raw_prompt["token_ids"],
            label=f"prompt row {index}",
            vocabulary_size=config["model"]["vocabulary_size"],
        )
        request = SameRunRequest(index=index, prompt_id=prompt_id, token_ids=tuple(token_ids))
        # This is the only backend call.  There is no capture call and no replay call.
        raw_result = backend.run_same_autoregressive_request(
            request=request,
            sampling=config["sampling"],
            capture=config["capture"],
        )
        normalized, tensors = _validate_backend_result(raw_result, request=request, config=config)
        _require(
            normalized["model_instance_id"] == backend.model_instance_id,
            "backend model instance changed within the producer",
        )
        shard_relative = Path("shards") / f"same-run-{index:06d}.safetensors"
        shard_binding = _write_tensor_shard(
            output_root / shard_relative,
            tensors=tensors,
            metadata={
                "schema": "swe-task-state-v4-stage-a-same-run-v2",
                "prompt_id": prompt_id,
                "request_id": normalized["request_id"],
                "runtime_config_sha256": runtime_config_sha256,
            },
        )
        shard_binding["path"] = shard_relative.as_posix()
        record = {
            "schema_version": 2,
            "kind": "swe_task_state_v4_counterfactual_stage_a_same_run_shard_v2",
            "status": (
                "same_autoregressive_request_complete"
                if backend_mode == "real_local_pinned_vllm_online_capture"
                else "injected_test_non_gate"
            ),
            "runtime_config_sha256": runtime_config_sha256,
            "producer": producer_binding,
            "backend": {
                "mode": backend_mode,
                "kind": backend.backend_kind,
                "implementation": dict(backend.implementation_binding),
                "model_instance_id": backend.model_instance_id,
            },
            "index": index,
            "prompt_id": prompt_id,
            "prompt": {
                "submitted_token_ids": normalized["submitted_prompt_token_ids"],
                "submitted_token_ids_sha256": token_ids_sha256(
                    normalized["submitted_prompt_token_ids"]
                ),
                "engine_returned_token_ids": normalized[
                    "engine_returned_prompt_token_ids"
                ],
                "engine_returned_token_ids_sha256": token_ids_sha256(
                    normalized["engine_returned_prompt_token_ids"]
                ),
                "token_count": len(token_ids),
                "exact_equal": True,
            },
            "same_run": {
                key: normalized[key]
                for key in (
                    "request_id",
                    "model_instance_id",
                    "prefill_forward_id",
                    "kv_cache_sequence_id",
                    "prompt_tail_position_id",
                    "prompt_tail_cache_write_index",
                    "first_decode_cache_length",
                    "capture_completed_monotonic_ns",
                    "first_decode_started_monotonic_ns",
                    "same_backend_call",
                    "capture_was_online_in_prefill",
                    "generation_consumed_same_kv_cache",
                    "replay_between_capture_and_first_decode",
                )
            },
            "state": {
                "shard": shard_binding,
                "tensor_logical_sha256": normalized["tensor_logical_sha256"],
                "selected_layers": list(config["capture"]["layers"]),
                "prompt_tail_only": True,
            },
            "forward_canary": {
                "prompt_tail_logits_argmax_token_id": normalized[
                    "prompt_tail_logits_argmax_token_id"
                ],
                "first_generated_token_id": normalized["generated_token_ids"][0],
                "first_generated_token_matches_prompt_tail_argmax": True,
                "final_norm_reconstruction_within_tolerance": normalized[
                    "final_norm_reconstruction_within_tolerance"
                ],
                "final_logits_reconstruction_within_tolerance": normalized[
                    "final_logits_reconstruction_within_tolerance"
                ],
            },
            "generation": {
                "token_ids": normalized["generated_token_ids"],
                "token_ids_sha256": token_ids_sha256(normalized["generated_token_ids"]),
                "token_count": len(normalized["generated_token_ids"]),
                "completion_text": normalized["completion_text"],
                "completion_text_sha256": sha256_bytes(
                    normalized["completion_text"].encode("utf-8")
                ),
                "finish_reason": normalized["finish_reason"],
                "stop_reason": normalized["stop_reason"],
                "sampling": normalized["engine_sampling"],
            },
            "claims": FALSE_CLAIMS,
            "boundary_limits": {
                "captured_causal_boundary_count": 1,
                "one_prompt_tail_boundary_is_a_temporal_cot_trajectory": False,
                "private_or_verbatim_cot_ground_truth_present": False,
                "this_record_can_establish_private_cot": False,
            },
            "reserved_validation_access_authorized": False,
        }
        record_path = output_root / "records" / f"same-run-{index:06d}.json"
        _atomic_json(record_path, record)
        record_binding = file_binding(record_path, display_root=output_root)
        records.append(
            {
                "index": index,
                "prompt_id": prompt_id,
                "request_id": normalized["request_id"],
                "record": record_binding,
                "tensor_shard": shard_binding,
            }
        )
        completion_records.append(
            {
                "prompt_id": prompt_id,
                "completion_status": "forced_exact_256_token_window",
                "completion_text": normalized["completion_text"],
            }
        )
    _require(len({row["request_id"] for row in records}) == len(records), "request IDs duplicated")
    completion_bundle = {
        "schema_version": 2,
        "kind": "swe_task_state_v4_counterfactual_stage_a_same_run_completion_bundle_v2",
        "status": (
            "stage_a_same_run_generation_complete_condition_blind"
            if backend_mode == "real_local_pinned_vllm_online_capture"
            else "injected_test_non_gate"
        ),
        "runtime_config_sha256": runtime_config_sha256,
        "surface_policy": "exact_256_token_same_run_completion_only_no_prompt_or_condition_join",
        "records": completion_records,
        "stage_b_records_present": False,
        "condition_key_joined": False,
        "reserved_validation_access_authorized": False,
    }
    completion_path = output_root / "stage-a-same-run-completions-v2.json"
    _atomic_json(completion_path, completion_bundle)
    return {
        "schema_version": 2,
        "kind": "swe_task_state_v4_counterfactual_stage_a_same_run_producer_result_v2",
        "status": (
            "producer_outputs_complete_pending_external_trace_and_cpu_verification"
            if backend_mode == "real_local_pinned_vllm_online_capture"
            else "injected_test_non_gate"
        ),
        "runtime_config_sha256": runtime_config_sha256,
        "producer": producer_binding,
        "backend": {
            "mode": backend_mode,
            "kind": backend.backend_kind,
            "implementation": dict(backend.implementation_binding),
            "model_instance_id": backend.model_instance_id,
        },
        "execution_provenance": execution_provenance,
        "pre_generation_receipts": {
            "preflight": dict(preflight_receipt),
            "capture_verification_lock": dict(capture_lock_receipt),
            "both_received_before_first_backend_call": True,
        },
        "records": records,
        "completion_bundle": file_binding(completion_path, display_root=output_root),
        "aggregate": {
            "record_count": len(records),
            "backend_call_count": len(records),
            "unique_request_count": len(records),
            "single_model_instance": True,
            "exact_generated_tokens_per_record": 256,
            "first_token_parity_all": True,
        },
        "claims": FALSE_CLAIMS,
        "boundary_limits": {
            "captured_causal_boundary_count_per_generation": 1,
            "one_prompt_tail_boundary_is_a_temporal_cot_trajectory": False,
            "private_or_verbatim_cot_ground_truth_present": False,
            "this_bundle_can_establish_private_cot": False,
        },
        "gate_eligible": backend_mode == "real_local_pinned_vllm_online_capture",
        "reserved_validation_access_authorized": False,
    }


__all__ = [
    "FALSE_CLAIMS",
    "SameAutoregressiveBackend",
    "SameRunProducerError",
    "SameRunRequest",
    "canonical_json_bytes",
    "file_binding",
    "produce_same_run_bundle",
    "sha256_bytes",
    "sha256_file",
    "token_ids_sha256",
]
