#!/usr/bin/env python3
"""Verify a completed native NVFP4/FP8-STE Jacobian Lens export.

The exported checkpoint intentionally uses the four-key upstream Jacobian Lens
format, so its tensor header alone cannot distinguish it from the public lens.
This verifier binds every exported tensor to the authoritative final-mean
metadata, the frozen production contract, and the source files used by the fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
MODEL_METADATA_SHA256 = {
    "config.json": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
    "hf_quant_config.json": "fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1",
    "model.safetensors.index.json": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
}
MODEL_SHARDS = {
    "model-00001-of-00003.safetensors": {
        "bytes": 9_965_652_512,
        "blob_sha256": "b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d",
    },
    "model-00002-of-00003.safetensors": {
        "bytes": 9_985_757_032,
        "blob_sha256": "06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d",
    },
    "model-00003-of-00003.safetensors": {
        "bytes": 1_970_287_640,
        "blob_sha256": "e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845",
    },
}
PROMPT_MANIFEST_SHA256 = (
    "2c36f17dee7287c096f7d1fdb7f8d7ecb8372c6cf8d13d7af8cfaed820439d3b"
)
PROMPT_ENTRIES_SHA256 = (
    "ca06a4904a378964cecec933a08df0451efaaf7151a6fa938e5489d6e55784a7"
)
PRODUCTION_CONTRACT_SHA256 = (
    "7944ea163b548edc3372fa67242fbbcfbe0a5abbe95c04ce4a378107ebe03dd0"
)
PRODUCTION_SOURCE_FILES_SHA256 = (
    "dbe1f28bbd829fa30cb48b4c593419de205c440d195c12bed398c0036ed16400"
)

D_MODEL = 5120
SOURCE_LAYERS = tuple(range(63))
TARGET_LAYER = 63
N_PROMPTS = 10
TOKEN_COUNT = 128
SKIP_FIRST = 16
CHECKPOINT_KEYS = {"J", "d_model", "n_prompts", "source_layers"}
FINAL_METADATA_KEYS = {
    "schema_version",
    "artifact_type",
    "created_at",
    "run_id",
    "contract_sha256",
    "layout",
    "n_prompts",
    "averaging",
    "layers",
    "layer_aggregate_sha256",
    "committed_prompts_sha256",
    "metadata",
    "metadata_sha256",
}
LAYER_RECORD_KEYS = {"layer", "filename", "shape", "dtype", "size", "sha256"}
FIT_TYPE = "Qwen3.6-27B NVFP4 exact-forward surrogate-backward J-lens"
FIT_QUANTIZATION_LABEL = "nvidia-modelopt-nvfp4-fp8-exact-forward-identity-ste"
FIT_ESTIMATOR_LABEL = "anthropic-future-summed-vjp"
HASH_CHUNK_BYTES = 16 * 1024 * 1024
MODEL_IDENTITY_POLICY = "pinned-nvidia-modelopt-snapshot-v1"
CAPTURE_LAYERS = tuple(range(64))
FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))
GDN_LAYERS = tuple(layer for layer in CAPTURE_LAYERS if layer not in FULL_ATTENTION_LAYERS)
OBSERVER_CUSTOM_OP = "jlens_nvfp4::capture_output"
OBSERVER_LINEAR_CLASSES = (
    "LinearBase",
    "ReplicatedLinear",
    "ColumnParallelLinear",
    "RowParallelLinear",
)
EXPECTED_LINEAR_BOUNDARIES = 304
EXPECTED_REPLAY_PARAMETERS = 785
EXPECTED_SHARED_TENSORS = 688
EXPECTED_OBSERVER_ONLY_TENSORS = 432
EXPECTED_OBSERVER_TENSORS = 1120
EXPECTED_CAPTURE_TENSORS = 1568
EXPECTED_SHARED_LOGICAL_BYTES = 892_731_392
EXPECTED_PARAMETER_LOGICAL_BYTES = 7_266_685_440
COMMIT_KEYS = {
    "prompt_index",
    "prompt",
    "prompt_sha256",
    "chunk_count",
    "chunk_manifest_sha256",
    "sum_generation",
    "sum_aggregate_sha256",
    "committed_at",
}
FIT_STATE_KEYS = {
    "schema_version",
    "status",
    "run_id",
    "contract",
    "contract_sha256",
    "layout",
    "prompt_count",
    "n_done",
    "next_prompt",
    "sum_generation",
    "sum_integrity",
    "committed_prompts",
    "current",
    "final_artifact",
    "started_at",
    "updated_at",
}
FINAL_STATE_RECORD_KEYS = {
    "directory",
    "metadata_path",
    "metadata_sha256",
    "layer_aggregate_sha256",
    "metadata_payload_sha256",
    "n_prompts",
}


@dataclass(frozen=True)
class _OpenedRegularFile:
    path: Path
    fd: int
    opened_stat: os.stat_result


class HeldRegularFile:
    """A pinned regular-file descriptor with a post-consumer integrity check."""

    def __init__(
        self,
        *,
        descriptor: int,
        path: Path,
        opened_stat: os.stat_result,
        expected_sha256: str,
        label: str,
    ) -> None:
        self._descriptor = descriptor
        self.path = path
        self._opened_stat = opened_stat
        self._expected_sha256 = expected_sha256
        self._label = label

    @property
    def fd_path(self) -> str:
        if self._descriptor < 0:
            raise RuntimeError(f"held {self._label} descriptor is closed")
        return f"/proc/self/fd/{self._descriptor}"

    def require_unchanged(self) -> None:
        if self._descriptor < 0:
            raise RuntimeError(f"held {self._label} descriptor is closed")
        opened = _OpenedRegularFile(
            path=self.path,
            fd=self._descriptor,
            opened_stat=self._opened_stat,
        )
        digest, size = _hash_fd(opened, label=self._label)
        _require(
            digest == self._expected_sha256 and size == self._opened_stat.st_size,
            f"{self._label} content changed after verification",
        )

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def __enter__(self) -> "HeldRegularFile":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class VerifiedNvfp4SteArtifact(HeldRegularFile):
    """JSON metadata plus a held descriptor for the exact verified lens inode."""

    def __init__(
        self,
        *,
        descriptor: int,
        path: Path,
        opened_stat: os.stat_result,
        record: dict[str, Any],
    ) -> None:
        super().__init__(
            descriptor=descriptor,
            path=path,
            opened_stat=opened_stat,
            expected_sha256=record["sha256"],
            label="native lens checkpoint",
        )
        self.record = record

    def __enter__(self) -> "VerifiedNvfp4SteArtifact":
        return self


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalize_sha256(value: Any, *, label: str) -> str:
    _require(_is_sha256(value), f"{label} is not a lowercase SHA-256")
    return value


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_regular(path: Path, *, label: str) -> _OpenedRegularFile:
    path = _absolute_path(path)
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} does not exist: {path}") from None
    _require(
        stat.S_ISREG(before.st_mode),
        f"{label} must be a regular non-symlink file",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"could not securely open {label}: {path}") from error
    try:
        opened = os.fstat(descriptor)
        _require(stat.S_ISREG(opened.st_mode), f"{label} must be a regular file")
        _require(
            (before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino),
            f"{label} changed before it was opened",
        )
        return _OpenedRegularFile(path=path, fd=descriptor, opened_stat=opened)
    except BaseException:
        os.close(descriptor)
        raise


def open_held_regular_file(
    path: Path, *, label: str, expected_sha256: str
) -> HeldRegularFile:
    """Open one expected regular-file inode for verification and consumption."""

    expected_sha256 = _normalize_sha256(expected_sha256, label=f"expected {label} hash")
    opened = _open_regular(path, label=label)
    return HeldRegularFile(
        descriptor=opened.fd,
        path=opened.path,
        opened_stat=opened.opened_stat,
        expected_sha256=expected_sha256,
        label=label,
    )


def _read_fd(opened: _OpenedRegularFile, *, label: str) -> tuple[bytes, str, int]:
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    total = 0
    offset = 0
    while True:
        chunk = os.pread(opened.fd, HASH_CHUNK_BYTES, offset)
        if not chunk:
            break
        chunks.append(chunk)
        digest.update(chunk)
        total += len(chunk)
        offset += len(chunk)
    after = os.fstat(opened.fd)
    _require(
        _stat_identity(after) == _stat_identity(opened.opened_stat)
        and total == opened.opened_stat.st_size,
        f"{label} changed while it was read",
    )
    return b"".join(chunks), digest.hexdigest(), total


def _hash_fd(opened: _OpenedRegularFile, *, label: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    offset = 0
    while True:
        chunk = os.pread(opened.fd, HASH_CHUNK_BYTES, offset)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
        offset += len(chunk)
    after = os.fstat(opened.fd)
    _require(
        _stat_identity(after) == _stat_identity(opened.opened_stat)
        and total == opened.opened_stat.st_size,
        f"{label} changed while it was hashed",
    )
    return digest.hexdigest(), total


def sha256_file(path: Path) -> str:
    opened = _open_regular(path, label="artifact input")
    try:
        digest, _size = _hash_fd(opened, label="artifact input")
        return digest
    finally:
        os.close(opened.fd)


def _read_json_file(
    opened: _OpenedRegularFile, *, label: str
) -> tuple[dict[str, Any], str, int]:
    raw, digest, size = _read_fd(opened, label=label)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value, digest, size


def _parse_timestamp(value: Any, *, label: str) -> None:
    _require(isinstance(value, str) and value, f"{label} must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} is not an ISO-8601 timestamp") from error
    _require(parsed.tzinfo is not None, f"{label} must include a timezone")


def _expected_layout(
    *, d_model: int, source_layers: Sequence[int], n_prompts: int, io_rows: int
) -> dict[str, Any]:
    matrix_bytes = d_model * d_model * 4
    return {
        "hidden_size": d_model,
        "source_layers": list(source_layers),
        "source_layer_count": len(source_layers),
        "prompt_count": n_prompts,
        "io_rows": io_rows,
        "matrix_dtype": "little-endian-float32",
        "matrix_shape": [d_model, d_model],
        "matrix_bytes": matrix_bytes,
        "matrix_set_bytes": matrix_bytes * len(source_layers),
    }


def _verify_contract(
    contract: Mapping[str, Any],
    *,
    expected_contract_sha256: str,
    expected_source_files_sha256: str,
    expected_prompt_manifest_sha256: str,
    expected_prompt_entries_sha256: str,
    d_model: int,
    source_layers: Sequence[int],
    target_layer: int,
    n_prompts: int,
    io_rows: int,
    source_root: Path | None,
) -> None:
    _require(
        canonical_sha256(contract) == expected_contract_sha256,
        "native fit contract SHA-256 mismatch",
    )
    model = _require_mapping(contract.get("model"), label="native fit model")
    _require(model.get("id") == MODEL_REPO, "native fit model repository mismatch")
    _require(model.get("revision") == MODEL_REVISION, "native fit model revision mismatch")
    checkpoint = _require_mapping(
        model.get("checkpoint_files"), label="native fit checkpoint files"
    )
    _require(
        checkpoint.get("metadata_sha256") == MODEL_METADATA_SHA256,
        "native fit model metadata hashes mismatch",
    )
    _require(checkpoint.get("shards") == MODEL_SHARDS, "native fit shard hashes mismatch")
    _require(
        model.get("checkpoint_integrity_before_each_prompt_commit") is True,
        "native fit did not require checkpoint integrity before prompt commit",
    )

    estimator = _require_mapping(
        contract.get("estimator"), label="native fit estimator"
    )
    expected_estimator = {
        "name": FIT_ESTIMATOR_LABEL,
        "hidden_size": d_model,
        "source_layers": list(source_layers),
        "target_layer": target_layer,
        "decoder_layers": target_layer + 1,
        "prompt_count": n_prompts,
        "token_count": TOKEN_COUNT,
        "skip_first": SKIP_FIRST,
        "mean_over_source_positions": True,
        "is_grads_batched": True,
    }
    for key, expected in expected_estimator.items():
        _require(estimator.get(key) == expected, f"native fit estimator {key} mismatch")
    cotangent_batch = estimator.get("cotangent_batch")
    _require(
        isinstance(cotangent_batch, int)
        and not isinstance(cotangent_batch, bool)
        and cotangent_batch > 0
        and d_model % cotangent_batch == 0,
        "native fit cotangent batch is invalid",
    )

    _require(
        contract.get("storage")
        == _expected_layout(
            d_model=d_model,
            source_layers=source_layers,
            n_prompts=n_prompts,
            io_rows=io_rows,
        ),
        "native fit storage contract mismatch",
    )
    backward = _require_mapping(
        contract.get("surrogate_backward"), label="native surrogate backward"
    )
    _require(
        backward.get("activation_ste_policy") == "identity"
        and backward.get("literal_rounding_derivative") is False
        and backward.get("clipped_ste_supported") is False,
        "native surrogate derivative disclosure mismatch",
    )

    prompts = _require_mapping(contract.get("prompts"), label="native fit prompts")
    prompt_entries = prompts.get("entries")
    _require(
        isinstance(prompt_entries, list) and len(prompt_entries) == n_prompts,
        "native fit prompt count mismatch",
    )
    _require(
        prompts.get("entries_sha256") == canonical_sha256(prompt_entries)
        == expected_prompt_entries_sha256,
        "native fit prompt entries hash mismatch",
    )
    manifest = _require_mapping(
        prompts.get("manifest"), label="native fit prompt manifest"
    )
    _require(
        manifest.get("sha256") == expected_prompt_manifest_sha256,
        "native fit prompt manifest hash mismatch",
    )
    for index, prompt in enumerate(prompt_entries):
        _require_mapping(prompt, label=f"native fit prompt {index}")
        _require(prompt.get("manifest_index") == index, "native prompt order mismatch")
        _require(prompt.get("token_count") == TOKEN_COUNT, "native prompt token count mismatch")
        token_ids = prompt.get("token_ids")
        _require(
            isinstance(token_ids, list)
            and len(token_ids) == TOKEN_COUNT
            and all(isinstance(token, int) and not isinstance(token, bool) for token in token_ids),
            "native prompt token IDs mismatch",
        )
        _normalize_sha256(prompt.get("text_sha256"), label="native prompt text hash")
        _normalize_sha256(
            prompt.get("token_ids_sha256"), label="native prompt token-ID hash"
        )

    source_files = contract.get("source_files")
    _require(isinstance(source_files, list) and source_files, "native source files missing")
    _require(
        canonical_sha256(source_files) == expected_source_files_sha256
        and contract.get("source_files_sha256") == expected_source_files_sha256,
        "native source-file manifest hash mismatch",
    )
    seen: set[str] = set()
    for record in source_files:
        record = _require_mapping(record, label="native source file record")
        relative = record.get("path")
        _require(isinstance(relative, str) and relative, "native source path missing")
        pure = PurePosixPath(relative)
        _require(
            not pure.is_absolute() and ".." not in pure.parts and relative not in seen,
            "native source path is unsafe or duplicated",
        )
        seen.add(relative)
        expected_sha256 = _normalize_sha256(
            record.get("sha256"), label=f"native source {relative} hash"
        )
        size = record.get("bytes")
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            f"native source {relative} size is invalid",
        )
        if source_root is not None:
            path = source_root / relative
            opened = _open_regular(path, label=f"native source {relative}")
            try:
                actual_sha256, actual_size = _hash_fd(
                    opened, label=f"native source {relative}"
                )
                _require(
                    actual_size == size,
                    f"native source {relative} size mismatch",
                )
                _require(
                    actual_sha256 == expected_sha256,
                    f"native source {relative} SHA-256 mismatch",
                )
            finally:
                os.close(opened.fd)


def _verify_sha_file_record(value: Any, *, label: str) -> Mapping[str, Any]:
    record = _require_mapping(value, label=label)
    _require(set(record) == {"path", "bytes", "sha256"}, f"{label} fields mismatch")
    _require(
        isinstance(record.get("path"), str) and record["path"],
        f"{label} path is invalid",
    )
    _require(
        isinstance(record.get("bytes"), int)
        and not isinstance(record["bytes"], bool)
        and record["bytes"] > 0,
        f"{label} size is invalid",
    )
    _normalize_sha256(record.get("sha256"), label=f"{label} hash")
    return record


def _verify_model_identity(value: Any) -> Mapping[str, Any]:
    identity = _require_mapping(value, label="native capture model identity")
    _require(
        identity.get("policy") == MODEL_IDENTITY_POLICY
        and identity.get("repo_id") == MODEL_REPO
        and identity.get("revision") == MODEL_REVISION
        and identity.get("metadata_sha256") == MODEL_METADATA_SHA256
        and identity.get("strict_pinned_validation") is True
        and identity.get("validator") == "ModelOptCheckpoint(strict_pinned=True)",
        "native capture model identity mismatch",
    )
    resolved = identity.get("resolved_path")
    _require(isinstance(resolved, str), "native capture model path is missing")
    model_path = PurePosixPath(resolved)
    _require(
        len(model_path.parts) >= 3
        and model_path.name == MODEL_REVISION
        and model_path.parent.name == "snapshots"
        and model_path.parent.parent.name == "models--nvidia--Qwen3.6-27B-NVFP4",
        "native capture model path is not the pinned snapshot",
    )
    return identity


def _verify_observer_scope(value: Any) -> Mapping[str, Any]:
    scope = _require_mapping(value, label="native observer scope")
    _require(
        scope.get("mode") == "compiled-observer"
        and scope.get("compiled_observer") is True
        and scope.get("target_profile") == "all"
        and scope.get("target_layers") == list(CAPTURE_LAYERS),
        "native observer scope mismatch",
    )
    patch = _require_mapping(
        scope.get("compiled_observer_patch"), label="native observer patch"
    )
    _require(
        patch.get("capacity") == TOKEN_COUNT
        and patch.get("custom_op") == OBSERVER_CUSTOM_OP
        and patch.get("patched_linear_classes") == list(OBSERVER_LINEAR_CLASSES)
        and patch.get("post_output_only") is True
        and patch.get("target_layers") == list(CAPTURE_LAYERS),
        "native compiled observer patch mismatch",
    )
    install = _require_mapping(scope.get("install"), label="native observer install")
    _require(
        install.get("allocated_tensor_count") == EXPECTED_CAPTURE_TENSORS
        and install.get("capture_capacity") == TOKEN_COUNT
        and install.get("full_attention_layers") == list(FULL_ATTENTION_LAYERS)
        and install.get("gdn_layers") == list(GDN_LAYERS)
        and install.get("linear_boundary_count") == EXPECTED_LINEAR_BOUNDARIES
        and install.get("target_layers") == list(CAPTURE_LAYERS),
        "native observer install mismatch",
    )
    return scope


def _verify_proof_claim(
    value: Any, *, prompt_entry: Mapping[str, Any], baseline_generation_sha256: str
) -> Mapping[str, Any]:
    proof = _require_mapping(value, label="native capture proof claim")
    _require(
        set(proof)
        == {
            "claim",
            "generation_record_parity",
            "shared_internal_tensor_parity",
            "replay_parameter_parity",
            "observer_capture_completeness",
        },
        "native proof claim fields mismatch",
    )
    claim = _require_mapping(proof.get("claim"), label="native proof claim")
    _require(
        claim.get("scope")
        == "exact compiled main-model prefill for the pinned prompt and runtime shape"
        and claim.get("mtp") == "off"
        and claim.get("observer_graph_modified") is True
        and claim.get("observer_modification_discharged") is True,
        "native exact-forward proof claim mismatch",
    )
    discharge = claim.get("discharge_basis")
    _require(
        discharge
        == [
            "full generation-record equality with an isolated uninstrumented compiled baseline",
            "bit-exact parity for all 688 shared internal tensors",
            "shape/dtype/content-hash parity for all 785 replay parameters",
        ],
        "native proof discharge basis mismatch",
    )

    generation = _require_mapping(
        proof.get("generation_record_parity"), label="native generation parity"
    )
    record = _require_mapping(
        generation.get("record"), label="native generation parity record"
    )
    record_sha256 = _normalize_sha256(
        generation.get("record_sha256"), label="native generation record hash"
    )
    _require(
        generation.get("exact") is True
        and generation.get("records_compared")
        == [
            "baseline.authoritative_compiled_generation",
            "baseline.instrumented_generation",
            "observer.instrumented_generation",
        ]
        and record_sha256 == canonical_sha256(record) == baseline_generation_sha256
        and record.get("prompt_token_ids") == prompt_entry.get("token_ids"),
        "native generation parity mismatch",
    )

    shared = _require_mapping(
        proof.get("shared_internal_tensor_parity"), label="native shared tensor parity"
    )
    _require(
        shared.get("all_shared_bit_exact") is True
        and shared.get("baseline_only_tensor_count") == 0
        and shared.get("baseline_tensor_count") == EXPECTED_SHARED_TENSORS
        and shared.get("shared_tensor_count") == EXPECTED_SHARED_TENSORS
        and shared.get("observer_only_tensor_count") == EXPECTED_OBSERVER_ONLY_TENSORS
        and shared.get("observer_tensor_count") == EXPECTED_OBSERVER_TENSORS
        and shared.get("shared_logical_bytes") == EXPECTED_SHARED_LOGICAL_BYTES
        and shared.get("mismatches") == [],
        "native shared tensor proof mismatch",
    )
    for key in (
        "shared_names_sha256",
        "shared_tensor_manifest_sha256",
    ):
        _normalize_sha256(shared.get(key), label=f"native shared proof {key}")

    replay = _require_mapping(
        proof.get("replay_parameter_parity"), label="native replay parameter parity"
    )
    _require(
        replay.get("all_content_hashes_equal") is True
        and replay.get("all_dtypes_equal") is True
        and replay.get("all_names_equal") is True
        and replay.get("all_shapes_equal") is True
        and replay.get("json_provenance_equal") is True
        and replay.get("parameter_count") == EXPECTED_REPLAY_PARAMETERS
        and replay.get("logical_bytes") == EXPECTED_PARAMETER_LOGICAL_BYTES
        and replay.get("mismatches") == [],
        "native replay parameter proof mismatch",
    )
    for key in ("parameter_manifest_sha256", "parameter_names_sha256"):
        _normalize_sha256(replay.get(key), label=f"native replay proof {key}")

    completeness = _require_mapping(
        proof.get("observer_capture_completeness"),
        label="native observer completeness",
    )
    _require(
        completeness.get("compile_visible_names_equal_observer_only_names") is True
        and completeness.get("compile_visible_observer_tensor_count")
        == EXPECTED_OBSERVER_ONLY_TENSORS
        and completeness.get("required_missing") == []
        and completeness.get("truncated") == [],
        "native observer completeness proof mismatch",
    )
    return proof


def _verify_capture_binding(
    value: Any, *, prompt_index: int, prompt_entry: Mapping[str, Any]
) -> Mapping[str, Any]:
    binding = _require_mapping(value, label=f"native capture binding {prompt_index}")
    expected_keys = {
        "schema_version",
        "prompt_index",
        "capture_contract_sha256",
        "capture_state_path",
        "baseline_json",
        "baseline_tensors_deleted_record",
        "observer_json",
        "observer_tensors",
        "proof",
        "observer_scope",
        "observer_scope_sha256",
        "proof_claim",
        "proof_claim_sha256",
        "model_identity",
        "model_identity_sha256",
        "baseline_generation_sha256",
    }
    _require(set(binding) == expected_keys, "native capture binding fields mismatch")
    _require(
        binding.get("schema_version") == 1
        and binding.get("prompt_index") == prompt_index,
        "native capture binding prompt mismatch",
    )
    _normalize_sha256(
        binding.get("capture_contract_sha256"), label="native capture contract hash"
    )
    state_path = binding.get("capture_state_path")
    _require(
        isinstance(state_path, str)
        and PurePosixPath(state_path).name
        == f"prompt-{prompt_index:02d}-capture-state.json",
        "native capture state path mismatch",
    )
    for key in (
        "baseline_json",
        "baseline_tensors_deleted_record",
        "observer_json",
        "observer_tensors",
        "proof",
    ):
        _verify_sha_file_record(binding.get(key), label=f"native capture {key}")

    scope = _verify_observer_scope(binding.get("observer_scope"))
    _require(
        binding.get("observer_scope_sha256") == canonical_sha256(scope),
        "native observer scope hash mismatch",
    )
    identity = _verify_model_identity(binding.get("model_identity"))
    _require(
        binding.get("model_identity_sha256") == canonical_sha256(identity),
        "native model identity hash mismatch",
    )
    baseline_hash = _normalize_sha256(
        binding.get("baseline_generation_sha256"),
        label="native baseline generation hash",
    )
    proof = _verify_proof_claim(
        binding.get("proof_claim"),
        prompt_entry=prompt_entry,
        baseline_generation_sha256=baseline_hash,
    )
    _require(
        binding.get("proof_claim_sha256") == canonical_sha256(proof),
        "native proof claim hash mismatch",
    )
    return binding


def _verify_capture_invocation(value: Any, *, prompt_index: int) -> None:
    invocation = _require_mapping(value, label="native capture invocation")
    _require(
        set(invocation)
        == {
            "argv",
            "resume_used",
            "completed_at",
            "stdout_sha256",
            "stderr_sha256",
            "capture_state",
        },
        "native capture invocation fields mismatch",
    )
    argv = invocation.get("argv")
    _require(
        isinstance(argv, list)
        and argv
        and all(isinstance(argument, str) for argument in argv)
        and "--prompt-index" in argv
        and str(prompt_index) in argv,
        "native capture invocation argv mismatch",
    )
    _require(
        isinstance(invocation.get("resume_used"), bool),
        "native capture resume flag is invalid",
    )
    _parse_timestamp(invocation.get("completed_at"), label="native capture completion")
    _normalize_sha256(invocation.get("stdout_sha256"), label="native capture stdout hash")
    _normalize_sha256(invocation.get("stderr_sha256"), label="native capture stderr hash")
    _verify_sha_file_record(
        invocation.get("capture_state"), label="native capture state record"
    )


def _verify_progress_chunks(
    progress_record: Mapping[str, Any],
    *,
    prompt_index: int,
    commit: Mapping[str, Any],
    d_model: int,
    cotangent_batch: int,
) -> None:
    chunks = progress_record.get("chunks")
    expected_count = math.ceil(d_model / cotangent_batch)
    _require(
        isinstance(chunks, list) and len(chunks) == expected_count,
        f"native prompt {prompt_index} progress chunks are incomplete",
    )
    authoritative: list[dict[str, Any]] = []
    for chunk_index, chunk_value in enumerate(chunks):
        chunk = _require_mapping(
            chunk_value, label=f"native prompt {prompt_index} chunk {chunk_index}"
        )
        start = chunk_index * cotangent_batch
        stop = min(start + cotangent_batch, d_model)
        _require(
            chunk.get("start") == start and chunk.get("stop") == stop,
            f"native prompt {prompt_index} chunks are not contiguous",
        )
        digest = _normalize_sha256(
            chunk.get("sha256"), label=f"native prompt {prompt_index} chunk hash"
        )
        authoritative.append({"start": start, "stop": stop, "sha256": digest})
        recovered = chunk.get("telemetry_missing_due_to_crash") is True
        if recovered:
            _require(
                chunk.get("elapsed_seconds") is None and chunk.get("cuda") is None,
                "native recovered chunk telemetry is inconsistent",
            )
        else:
            elapsed = chunk.get("elapsed_seconds")
            _require(
                isinstance(elapsed, (int, float))
                and not isinstance(elapsed, bool)
                and math.isfinite(float(elapsed))
                and elapsed >= 0,
                "native chunk elapsed time is invalid",
            )
            cuda = _require_mapping(chunk.get("cuda"), label="native chunk CUDA telemetry")
            for key in (
                "allocated_bytes",
                "reserved_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            ):
                amount = cuda.get(key)
                _require(
                    isinstance(amount, int)
                    and not isinstance(amount, bool)
                    and amount >= 0,
                    f"native chunk CUDA {key} is invalid",
                )
    _require(
        commit.get("chunk_count") == expected_count
        and commit.get("chunk_manifest_sha256") == canonical_sha256(authoritative),
        f"native prompt {prompt_index} chunk manifest mismatch",
    )


def _verify_final_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_contract_sha256: str,
    expected_source_files_sha256: str,
    expected_prompt_manifest_sha256: str,
    expected_prompt_entries_sha256: str,
    d_model: int,
    source_layers: Sequence[int],
    target_layer: int,
    n_prompts: int,
    io_rows: int,
    source_root: Path | None,
) -> list[Mapping[str, Any]]:
    _require(set(metadata) == FINAL_METADATA_KEYS, "native final metadata fields mismatch")
    _require(metadata.get("schema_version") == 1, "native final metadata schema mismatch")
    _require(
        metadata.get("artifact_type") == "lumo-jlens-dense-fp32-means",
        "native final artifact type mismatch",
    )
    _parse_timestamp(metadata.get("created_at"), label="native final creation")
    _require(
        isinstance(metadata.get("run_id"), str) and metadata["run_id"],
        "native final run ID is invalid",
    )
    _require(
        metadata.get("contract_sha256") == expected_contract_sha256,
        "native final contract hash mismatch",
    )
    _require(metadata.get("n_prompts") == n_prompts, "native final prompt count mismatch")
    layout = _expected_layout(
        d_model=d_model,
        source_layers=source_layers,
        n_prompts=n_prompts,
        io_rows=io_rows,
    )
    _require(metadata.get("layout") == layout, "native final layout mismatch")
    _require(
        metadata.get("averaging")
        == "arithmetic mean of cumulative little-endian FP32 sums",
        "native final averaging rule mismatch",
    )

    layer_records = metadata.get("layers")
    _require(
        isinstance(layer_records, list) and len(layer_records) == len(source_layers),
        "native final layer manifest is incomplete",
    )
    for layer, record in zip(source_layers, layer_records, strict=True):
        record = _require_mapping(record, label=f"native final layer {layer}")
        _require(set(record) == LAYER_RECORD_KEYS, f"native layer {layer} fields mismatch")
        _require(
            record.get("layer") == layer
            and record.get("filename") == f"layer-{layer:02d}.f32"
            and record.get("shape") == [d_model, d_model]
            and record.get("dtype") == "little-endian-float32"
            and record.get("size") == d_model * d_model * 4,
            f"native layer {layer} geometry mismatch",
        )
        _normalize_sha256(record.get("sha256"), label=f"native layer {layer} hash")
    _require(
        metadata.get("layer_aggregate_sha256") == canonical_sha256(layer_records),
        "native final layer aggregate hash mismatch",
    )

    payload = _require_mapping(metadata.get("metadata"), label="native fit provenance")
    _require(
        metadata.get("metadata_sha256") == canonical_sha256(payload),
        "native fit provenance payload hash mismatch",
    )
    _require(payload.get("schema_version") == 1, "native fit provenance schema mismatch")
    _require(payload.get("fit_type") == FIT_TYPE, "native fit type mismatch")
    contract = _require_mapping(payload.get("contract"), label="native fit contract")
    _require(
        payload.get("contract_sha256") == expected_contract_sha256,
        "native provenance contract hash mismatch",
    )
    _verify_contract(
        contract,
        expected_contract_sha256=expected_contract_sha256,
        expected_source_files_sha256=expected_source_files_sha256,
        expected_prompt_manifest_sha256=expected_prompt_manifest_sha256,
        expected_prompt_entries_sha256=expected_prompt_entries_sha256,
        d_model=d_model,
        source_layers=source_layers,
        target_layer=target_layer,
        n_prompts=n_prompts,
        io_rows=io_rows,
        source_root=source_root,
    )

    progress = _require_mapping(payload.get("progress"), label="native fit progress")
    _require(
        payload.get("progress_sha256") == canonical_sha256(progress),
        "native fit progress hash mismatch",
    )
    _require(
        progress.get("schema_version") == 1
        and progress.get("contract_sha256") == expected_contract_sha256,
        "native fit progress contract mismatch",
    )
    prompt_progress = progress.get("prompts")
    _require(
        isinstance(prompt_progress, dict)
        and set(prompt_progress) == {str(index) for index in range(n_prompts)},
        "native fit progress prompt set mismatch",
    )
    for key in ("max_cuda_peak_allocated_bytes", "max_cuda_peak_reserved_bytes"):
        value = progress.get(key)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"native fit progress {key} is invalid",
        )

    committed = payload.get("committed_prompts")
    _require(
        isinstance(committed, list) and len(committed) == n_prompts,
        "native committed prompt set is incomplete",
    )
    committed_sha256 = canonical_sha256(committed)
    _require(
        payload.get("committed_prompts_sha256") == committed_sha256
        and metadata.get("committed_prompts_sha256") == committed_sha256,
        "native committed prompt hash mismatch",
    )
    prompt_entries = contract["prompts"]["entries"]
    estimator = contract["estimator"]
    expected_fit = {
        "source_layers": list(source_layers),
        "target_layer": target_layer,
        "skip_first": SKIP_FIRST,
        "cotangent_batch": estimator["cotangent_batch"],
        "checkpoint_interval": estimator["checkpoint_interval"],
        "ste_policy": "identity",
    }
    for index, commit in enumerate(committed):
        commit = _require_mapping(commit, label=f"native prompt commit {index}")
        _require(set(commit) == COMMIT_KEYS, "native prompt commit fields mismatch")
        _require(
            commit.get("prompt_index") == index
            and commit.get("sum_generation") == index + 1,
            "native prompt commit order mismatch",
        )
        _parse_timestamp(
            commit.get("committed_at"), label=f"native prompt {index} commit"
        )
        prompt = _require_mapping(commit.get("prompt"), label="native committed prompt")
        _require(
            set(prompt) == {"manifest_prompt", "capture", "fit"},
            "native committed prompt fields mismatch",
        )
        _require(
            commit.get("prompt_sha256") == canonical_sha256(prompt),
            "native committed prompt payload hash mismatch",
        )
        _require(
            prompt.get("manifest_prompt") == prompt_entries[index],
            "native committed prompt differs from contract",
        )
        _require(prompt.get("fit") == expected_fit, "native committed fit config mismatch")
        capture = _verify_capture_binding(
            prompt.get("capture"),
            prompt_index=index,
            prompt_entry=prompt_entries[index],
        )
        progress_record = _require_mapping(
            prompt_progress[str(index)], label=f"native prompt progress {index}"
        )
        _require(
            progress_record.get("commit") == commit,
            "native progress/commit record mismatch",
        )
        _require(
            progress_record.get("capture_binding") == capture,
            "native progress/capture binding mismatch",
        )
        invocations = progress_record.get("capture_invocations")
        _require(
            isinstance(invocations, list) and invocations,
            "native capture invocation history is missing",
        )
        for invocation in invocations:
            _verify_capture_invocation(invocation, prompt_index=index)
        _verify_progress_chunks(
            progress_record,
            prompt_index=index,
            commit=commit,
            d_model=d_model,
            cotangent_batch=estimator["cotangent_batch"],
        )
        _normalize_sha256(
            commit.get("sum_aggregate_sha256"), label="native sum aggregate hash"
        )

    _require(
        payload.get("disclosure")
        == {
            "forward": "exact deployed compiled NVFP4/FP8 observer capture",
            "backward": "packed W4 and live FP8 declared surrogate VJPs",
            "literal_derivative_of_quantized_rounding": False,
        },
        "native fit disclosure mismatch",
    )
    return layer_records


def _verify_fit_state(
    state: Mapping[str, Any],
    *,
    state_file_sha256: str,
    expected_state_sha256: str,
    metadata: Mapping[str, Any],
    metadata_file_sha256: str,
    expected_contract_sha256: str,
    d_model: int,
    source_layers: Sequence[int],
    n_prompts: int,
    io_rows: int,
) -> None:
    _require(
        state_file_sha256 == expected_state_sha256,
        "native fit-state SHA-256 mismatch",
    )
    _require(set(state) == FIT_STATE_KEYS, "native fit-state fields mismatch")
    _require(
        state.get("schema_version") == 1 and state.get("status") == "completed",
        "native fit state is not completed",
    )
    _require(state.get("current") is None, "native fit state still has a current prompt")
    _require(
        state.get("run_id") == metadata.get("run_id"),
        "native fit-state run ID mismatch",
    )
    _parse_timestamp(state.get("started_at"), label="native fit start")
    _parse_timestamp(state.get("updated_at"), label="native fit update")
    expected_layout = _expected_layout(
        d_model=d_model,
        source_layers=source_layers,
        n_prompts=n_prompts,
        io_rows=io_rows,
    )
    payload = _require_mapping(metadata.get("metadata"), label="native fit provenance")
    _require(
        state.get("contract_sha256") == expected_contract_sha256
        and state.get("contract") == payload.get("contract")
        and state.get("layout") == expected_layout,
        "native fit-state contract/layout mismatch",
    )
    _require(
        state.get("prompt_count") == n_prompts
        and state.get("n_done") == n_prompts
        and state.get("next_prompt") == n_prompts
        and state.get("sum_generation") == n_prompts,
        "native fit-state completion counts mismatch",
    )
    _require(
        state.get("committed_prompts") == payload.get("committed_prompts"),
        "native fit-state committed prompts mismatch",
    )

    sum_integrity = _require_mapping(
        state.get("sum_integrity"), label="native fit-state sum integrity"
    )
    _require(
        set(sum_integrity) == {"generation", "layers", "aggregate_sha256"}
        and sum_integrity.get("generation") == n_prompts,
        "native fit-state sum generation mismatch",
    )
    sum_layers = sum_integrity.get("layers")
    _require(
        isinstance(sum_layers, list) and len(sum_layers) == len(source_layers),
        "native fit-state sum layer manifest is incomplete",
    )
    for layer, record_value in zip(source_layers, sum_layers, strict=True):
        record = _require_mapping(record_value, label=f"native sum layer {layer}")
        _require(
            set(record) == LAYER_RECORD_KEYS
            and record.get("layer") == layer
            and record.get("filename") == f"layer-{layer:02d}.f32"
            and record.get("shape") == [d_model, d_model]
            and record.get("dtype") == "little-endian-float32"
            and record.get("size") == d_model * d_model * 4,
            f"native sum layer {layer} geometry mismatch",
        )
        _normalize_sha256(record.get("sha256"), label=f"native sum layer {layer} hash")
    sum_aggregate = canonical_sha256(sum_layers)
    _require(
        sum_integrity.get("aggregate_sha256") == sum_aggregate
        and payload["committed_prompts"][-1].get("sum_aggregate_sha256")
        == sum_aggregate,
        "native final committed-sum binding mismatch",
    )

    final = _require_mapping(
        state.get("final_artifact"), label="native final fit-state record"
    )
    _require(set(final) == FINAL_STATE_RECORD_KEYS, "native final-state fields mismatch")
    _require(
        final.get("directory") == "final-mean"
        and final.get("metadata_path") == "final-mean/metadata.json"
        and final.get("metadata_sha256") == metadata_file_sha256
        and final.get("layer_aggregate_sha256")
        == metadata.get("layer_aggregate_sha256")
        and final.get("metadata_payload_sha256") == metadata.get("metadata_sha256")
        and final.get("n_prompts") == n_prompts,
        "native final fit-state binding mismatch",
    )


def _tensor_sha256(tensor: Any, *, check_finite: bool, layer: int) -> str:
    import torch

    digest = hashlib.sha256()
    for start in range(0, tensor.shape[0], 64):
        chunk = tensor[start : min(start + 64, tensor.shape[0])].detach().cpu().contiguous()
        if check_finite and not bool(torch.isfinite(chunk).all()):
            raise ValueError(f"native lens layer {layer} contains non-finite values")
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _verify_checkpoint(
    path: Path,
    *,
    layer_records: Sequence[Mapping[str, Any]],
    d_model: int,
    source_layers: Sequence[int],
    target_layer: int,
    n_prompts: int,
    check_finite: bool,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    _require(
        isinstance(checkpoint, dict) and set(checkpoint) == CHECKPOINT_KEYS,
        "unexpected native lens checkpoint keys",
    )
    _require(checkpoint["d_model"] == d_model, "native lens d_model mismatch")
    _require(checkpoint["n_prompts"] == n_prompts, "native lens prompt count mismatch")
    _require(
        checkpoint["source_layers"] == list(source_layers),
        "native lens source layers mismatch",
    )
    jacobians = checkpoint["J"]
    _require(
        isinstance(jacobians, dict) and set(jacobians) == set(source_layers),
        "native lens Jacobian keys mismatch",
    )
    for layer, record in zip(source_layers, layer_records, strict=True):
        tensor = jacobians[layer]
        _require(torch.is_tensor(tensor), f"native lens layer {layer} is not a tensor")
        _require(
            tuple(tensor.shape) == (d_model, d_model),
            f"native lens layer {layer} shape mismatch",
        )
        _require(tensor.dtype == torch.float32, f"native lens layer {layer} dtype mismatch")
        _require(
            tensor.device.type == "cpu",
            f"native lens layer {layer} device mismatch",
        )
        _require(
            _tensor_sha256(tensor, check_finite=check_finite, layer=layer)
            == record["sha256"],
            f"native lens layer {layer} differs from final-mean metadata",
        )
    del checkpoint
    return {
        "checkpoint_keys": sorted(CHECKPOINT_KEYS),
        "d_model": d_model,
        "n_prompts": n_prompts,
        "source_layers": list(source_layers),
        "target_layer": target_layer,
        "tensor_dtype": "torch.float32",
        "tensor_shape": [d_model, d_model],
        "finite_checked": check_finite,
    }


def open_verified_nvfp4_ste_artifact(
    path: Path,
    *,
    expected_sha256: str,
    provenance_path: Path,
    state_path: Path,
    expected_state_sha256: str,
    check_finite: bool = True,
    d_model: int = D_MODEL,
    source_layers: Sequence[int] = SOURCE_LAYERS,
    target_layer: int = TARGET_LAYER,
    expected_n_prompts: int = N_PROMPTS,
    io_rows: int = 64,
    expected_contract_sha256: str = PRODUCTION_CONTRACT_SHA256,
    expected_source_files_sha256: str = PRODUCTION_SOURCE_FILES_SHA256,
    expected_prompt_manifest_sha256: str = PROMPT_MANIFEST_SHA256,
    expected_prompt_entries_sha256: str = PROMPT_ENTRIES_SHA256,
    source_root: Path | None = ROOT,
) -> VerifiedNvfp4SteArtifact:
    """Verify one exact completed run and retain its lens inode for consumers."""

    expected_sha256 = _normalize_sha256(
        expected_sha256, label="expected native lens hash"
    )
    expected_state_sha256 = _normalize_sha256(
        expected_state_sha256, label="expected native fit-state hash"
    )
    lens = _open_regular(path, label="native lens checkpoint")
    provenance: _OpenedRegularFile | None = None
    state_file: _OpenedRegularFile | None = None
    try:
        provenance = _open_regular(
            provenance_path, label="native final metadata"
        )
        state_file = _open_regular(state_path, label="native fit state")
        artifact_sha256, artifact_size = _hash_fd(
            lens, label="native lens checkpoint"
        )
        _require(
            artifact_sha256 == expected_sha256,
            f"native lens SHA-256 mismatch: expected {expected_sha256}, got {artifact_sha256}",
        )
        metadata, metadata_file_sha256, metadata_file_size = _read_json_file(
            provenance, label="native final metadata"
        )
        state, state_file_sha256, state_file_size = _read_json_file(
            state_file, label="native fit state"
        )
        layer_records = _verify_final_metadata(
            metadata,
            expected_contract_sha256=expected_contract_sha256,
            expected_source_files_sha256=expected_source_files_sha256,
            expected_prompt_manifest_sha256=expected_prompt_manifest_sha256,
            expected_prompt_entries_sha256=expected_prompt_entries_sha256,
            d_model=d_model,
            source_layers=source_layers,
            target_layer=target_layer,
            n_prompts=expected_n_prompts,
            io_rows=io_rows,
            source_root=source_root,
        )
        _verify_fit_state(
            state,
            state_file_sha256=state_file_sha256,
            expected_state_sha256=expected_state_sha256,
            metadata=metadata,
            metadata_file_sha256=metadata_file_sha256,
            expected_contract_sha256=expected_contract_sha256,
            d_model=d_model,
            source_layers=source_layers,
            n_prompts=expected_n_prompts,
            io_rows=io_rows,
        )
        checkpoint = _verify_checkpoint(
            Path(f"/proc/self/fd/{lens.fd}"),
            layer_records=layer_records,
            d_model=d_model,
            source_layers=source_layers,
            target_layer=target_layer,
            n_prompts=expected_n_prompts,
            check_finite=check_finite,
        )
        final_lens_sha256, final_lens_size = _hash_fd(
            lens, label="native lens checkpoint"
        )
        _require(
            (final_lens_sha256, final_lens_size)
            == (artifact_sha256, artifact_size),
            "native lens changed during verification",
        )
        record = {
            "kind": "native_nvfp4_ste_fit",
            "verification_scope": "exact pinned production run; not a generic portable fit",
            "path": str(lens.path),
            "size_bytes": artifact_size,
            "sha256": artifact_sha256,
            "provenance_path": str(provenance.path),
            "provenance_size_bytes": metadata_file_size,
            "provenance_sha256": metadata_file_sha256,
            "state_path": str(state_file.path),
            "state_size_bytes": state_file_size,
            "state_sha256": state_file_sha256,
            "run_id": metadata["run_id"],
            "contract_sha256": expected_contract_sha256,
            "layer_aggregate_sha256": metadata["layer_aggregate_sha256"],
            "committed_prompts_sha256": metadata["committed_prompts_sha256"],
            "fit_model": MODEL_REPO,
            "fit_model_revision": MODEL_REVISION,
            "fit_quantization": FIT_QUANTIZATION_LABEL,
            "fit_estimator": FIT_ESTIMATOR_LABEL,
            "surrogate_backward": (
                "identity STE; not the literal derivative of quantized rounding"
            ),
            **checkpoint,
        }
        return VerifiedNvfp4SteArtifact(
            descriptor=lens.fd,
            path=lens.path,
            opened_stat=lens.opened_stat,
            record=record,
        )
    except BaseException:
        os.close(lens.fd)
        raise
    finally:
        if provenance is not None:
            os.close(provenance.fd)
        if state_file is not None:
            os.close(state_file.fd)


def verify_nvfp4_ste_artifact(
    path: Path,
    *,
    expected_sha256: str,
    provenance_path: Path,
    state_path: Path,
    expected_state_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return JSON-safe exact-run verification metadata and close the lens fd."""

    with open_verified_nvfp4_ste_artifact(
        path,
        expected_sha256=expected_sha256,
        provenance_path=provenance_path,
        state_path=state_path,
        expected_state_sha256=expected_state_sha256,
        **kwargs,
    ) as artifact:
        return dict(artifact.record)


__all__ = [
    "FIT_ESTIMATOR_LABEL",
    "FIT_QUANTIZATION_LABEL",
    "HeldRegularFile",
    "MODEL_REPO",
    "MODEL_REVISION",
    "VerifiedNvfp4SteArtifact",
    "open_verified_nvfp4_ste_artifact",
    "open_held_regular_file",
    "verify_nvfp4_ste_artifact",
]
