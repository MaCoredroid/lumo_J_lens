#!/usr/bin/env python3
"""Two-phase authenticated native smoke runner for the V3 model adapter.

``freeze`` performs no model generation and no GPU query.  It authenticates the
checked-in adapter, exact runtime/environment/package metadata, one local model
snapshot and tokenizer, then writes one finite-schema request specification and
one separately hashable launch authorization.

``run`` accepts the launch and request files only with caller-supplied exact
SHA-256 values.  It delegates the real generation exclusively to the production
adapter and writes one immutable smoke receipt.  A successful receipt proves an
exact-token native batch ran; it is not a sealed control result, COT evidence,
or affect/emotion/confidence/doubt/stress evidence.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3 as adapter  # noqa: E402


SCHEMA_VERSION = 1
INTERFACE_VERSION = 3
FREEZE_KIND = "swe_task_state_v4_epistemic_chain_native_smoke_freeze_v3"
RECEIPT_KIND = "swe_task_state_v4_epistemic_chain_native_smoke_receipt_v3"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROLES = ("independent_a", "independent_b", "adjudicator")

# This is an expected launch identity, not a pre-authorization live GPU query.
# The production adapter independently queries and exactly compares every field
# after authenticating the launch file.  Values are already recorded by the
# repository's public runtime provenance for this host.
EXPECTED_GPU_IDENTITY = {
    "torch_cuda_version": "13.0",
    "cudnn_version": 91900,
    "visible_device_count": 1,
    "device_index": 0,
    "device_name": "NVIDIA GeForce RTX 5090",
    "compute_capability": [12, 0],
    "total_memory_bytes": 33635434496,
}


class NativeSmokeError(RuntimeError):
    """Raised when the smoke harness cannot preserve its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NativeSmokeError(message)


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


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: str, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _strict_json_bytes(raw: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise NativeSmokeError(f"cannot parse {label}: {error}") from error


def _read_exact_json(path: Path, expected_sha256: str, label: str) -> Any:
    expected = _sha256(expected_sha256, f"expected {label} hash")
    candidate = Path(path)
    _require(not candidate.is_symlink(), f"{label} must not be a symlink")
    resolved = candidate.resolve(strict=True)
    _require(resolved.is_file(), f"{label} must be a regular file")
    raw = resolved.read_bytes()
    _require(sha256_bytes(raw) == expected, f"{label} differs from external hash")
    return _strict_json_bytes(raw, label)


def _assert_no_symlink_components(path: Path) -> None:
    candidate = Path(path).absolute()
    for component in (candidate, *candidate.parents):
        if component.exists():
            _require(
                not component.is_symlink(),
                f"output path has symlink component: {component}",
            )


def _exclusive_write_json(path: Path, value: Any) -> str:
    candidate = Path(path).absolute()
    _assert_no_symlink_components(candidate)
    _require(candidate.parent.is_dir(), f"output parent missing: {candidate.parent}")
    raw = canonical_json_bytes(value)
    try:
        descriptor = os.open(
            candidate,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except OSError as error:
        raise NativeSmokeError(
            f"cannot exclusively create {candidate}: {error}"
        ) from error
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "smoke artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_bytes(raw)


def _model_identity(role_spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: copy.deepcopy(role_spec[name]) for name in adapter.MODEL_IDENTITY_FIELDS
    }


def _tokenizer_identity(role_spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "repo_id": role_spec["repo_id"],
        "revision": role_spec["revision"],
        "snapshot_tree_sha256": role_spec["snapshot_tree_sha256"],
        "tokenizer_mode": role_spec["tokenizer_mode"],
        "tokenizer_class": role_spec["tokenizer_class"],
        "vocab_identity_sha256": role_spec["vocab_identity_sha256"],
    }


def expected_gpu_identity() -> dict[str, Any]:
    body = copy.deepcopy(EXPECTED_GPU_IDENTITY)
    return {**body, "gpu_identity_sha256": sha256_value(body)}


def smoke_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return exactly one JSON object satisfying the supplied schema. "
                "Do not add prose."
            ),
        },
        {
            "role": "user",
            "content": "Native exact-token smoke check. Set verdict to ok.",
        },
    ]


def smoke_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["ok"]},
        },
        "required": ["verdict"],
        "additionalProperties": False,
    }


def build_launch_authorization(
    *,
    authenticated_config: adapter.AuthenticatedAdapterConfig,
    role: str,
    request_batch_sha256: str,
    runtime_identity: Mapping[str, Any],
    environment_identity: Mapping[str, Any],
    package_identity: Mapping[str, Any],
    snapshot_inventory: Mapping[str, Any],
    authorization_nonce_sha256: str,
) -> dict[str, Any]:
    """Build the exact external true-state object consumed by the adapter."""

    _require(role in ROLES, "smoke role invalid")
    config = authenticated_config.value
    role_spec = config["roles"][role]
    model_identity = _model_identity(role_spec)
    tokenizer_identity = _tokenizer_identity(role_spec)
    gpu = expected_gpu_identity()
    launch = {
        "schema_version": SCHEMA_VERSION,
        "kind": adapter.LAUNCH_KIND,
        "execution_authorized": True,
        "model_access_authorized": True,
        "gpu_access_authorized": True,
        "output_authorized": True,
        "production_receipt_authorized": True,
        "gate_eligible_execution_authorized": True,
        "adapter_config_sha256": authenticated_config.config_sha256,
        "adapter_source_sha256": authenticated_config.source_sha256,
        "runner_sha256": authenticated_config.runner_sha256,
        "draft_config_sha256": authenticated_config.draft_config_sha256,
        "draft_source_sha256": authenticated_config.draft_source_sha256,
        "v2_config_sha256": authenticated_config.v2_config_sha256,
        "role": role,
        "model_identity": model_identity,
        "model_identity_sha256": sha256_value(model_identity),
        "snapshot_inventory_sha256": snapshot_inventory["inventory_sha256"],
        "tokenizer_identity": tokenizer_identity,
        "tokenizer_identity_sha256": sha256_value(tokenizer_identity),
        "package_bundle_sha256": package_identity["package_bundle_sha256"],
        "runtime_identity_sha256": runtime_identity["runtime_identity_sha256"],
        "environment_identity_sha256": environment_identity[
            "environment_identity_sha256"
        ],
        "gpu_identity_sha256": gpu["gpu_identity_sha256"],
        "request_batch_sha256": _sha256(request_batch_sha256, "request batch hash"),
        "authorization_nonce_sha256": _sha256(
            authorization_nonce_sha256, "authorization nonce hash"
        ),
    }
    _require(set(launch) == adapter.LAUNCH_FIELDS, "launch field set changed")
    return launch


def _load_verified_tokenizer_context(
    *,
    authenticated_config: adapter.AuthenticatedAdapterConfig,
    role: str,
    snapshot_path: Path,
) -> adapter.runner_v3.AuthenticatedNativeGenerationContext:
    # Local tokenizer metadata is needed to precommit the exact native request
    # hash.  No model weights are loaded and no CUDA module is imported here.
    from transformers import AutoTokenizer

    config = authenticated_config.value
    role_spec = config["roles"][role]
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path), local_files_only=True, trust_remote_code=False
    )
    qualified_class = f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    _require(
        qualified_class == role_spec["tokenizer_class"],
        "smoke tokenizer class differs from frozen identity",
    )
    vocab = tokenizer.get_vocab()
    _require(
        isinstance(vocab, Mapping)
        and len(vocab) == role_spec["vocab_size"]
        and sha256_value(vocab) == role_spec["vocab_identity_sha256"],
        "smoke tokenizer vocabulary differs from frozen identity",
    )
    model_identity = _model_identity(role_spec)
    tokenizer_identity = _tokenizer_identity(role_spec)
    return adapter.runner_v3.authenticate_native_generation_context(
        tokenizer=tokenizer,
        model_identity=model_identity,
        expected_model_identity_sha256=sha256_value(model_identity),
        tokenizer_identity=tokenizer_identity,
        expected_tokenizer_identity_sha256=sha256_value(tokenizer_identity),
        chat_template_kwargs=copy.deepcopy(role_spec["chat_template_kwargs"]),
    )


def freeze_smoke(
    *,
    role: str,
    expected_adapter_config_sha256: str,
    expected_adapter_source_sha256: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Freeze one exact request and launch file without model/GPU execution."""

    _require(role in ROLES, "smoke role invalid")
    authenticated = adapter.authenticate_adapter_config(
        path=adapter.CONFIG_PATH,
        expected_config_sha256=_sha256(
            expected_adapter_config_sha256, "adapter config hash"
        ),
    )
    _require(
        authenticated.source_sha256
        == _sha256(expected_adapter_source_sha256, "adapter source hash"),
        "adapter source differs from external hash",
    )
    config = authenticated.value
    runtime = adapter.runtime_identity(config)
    environment = adapter.environment_identity(config)
    packages = adapter.package_identity_bundle(config)
    adapter._validate_runtime_import_specs(packages)
    snapshot_path, snapshot = adapter.resolve_and_verify_role_snapshot(
        config=config, role=role
    )
    context = _load_verified_tokenizer_context(
        authenticated_config=authenticated, role=role, snapshot_path=snapshot_path
    )
    role_spec = config["roles"][role]
    packet_hash = sha256_value({"domain": "v3-native-smoke-packet", "role": role})
    source_hash = sha256_value({"domain": "v3-native-smoke-source", "role": role})
    lineage = {
        "smoke_harness_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    request = adapter.runner_v3.build_native_generation_request(
        context=context,
        messages=smoke_messages(),
        schema=smoke_schema(),
        seed=int(role_spec["seed"]),
        stage="native_exact_token_smoke",
        annotation_pass="completion_chain",
        packet_id_sha256=packet_hash,
        source_id_sha256=source_hash,
        lineage_bindings=lineage,
    )
    request_spec = adapter.NativeRequestSpec(
        messages=smoke_messages(),
        schema=smoke_schema(),
        seed=int(role_spec["seed"]),
        stage="native_exact_token_smoke",
        annotation_pass="completion_chain",
        packet_id_sha256=packet_hash,
        source_id_sha256=source_hash,
        lineage_bindings=lineage,
        expected_native_request_sha256=request.request_sha256,
    )
    request_batch = adapter.request_batch_descriptor(
        role=role, request_specs=[request_spec], config=config
    )
    authorization_nonce_sha256 = sha256_bytes(secrets.token_bytes(32))
    launch = build_launch_authorization(
        authenticated_config=authenticated,
        role=role,
        request_batch_sha256=request_batch["request_batch_sha256"],
        runtime_identity=runtime,
        environment_identity=environment,
        package_identity=packages,
        snapshot_inventory=snapshot,
        authorization_nonce_sha256=authorization_nonce_sha256,
    )

    output = Path(output_directory).absolute()
    _assert_no_symlink_components(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    request_specs = [asdict(request_spec)]
    request_specs_path = output / "request-specs.json"
    launch_path = output / "launch-authorization.json"
    request_specs_file_sha256 = _exclusive_write_json(request_specs_path, request_specs)
    launch_file_sha256 = _exclusive_write_json(launch_path, launch)
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": FREEZE_KIND,
        "status": "native_smoke_frozen_not_run",
        "role": role,
        "adapter_config_sha256": authenticated.config_sha256,
        "adapter_source_sha256": authenticated.source_sha256,
        "smoke_harness_source_sha256": lineage["smoke_harness_source_sha256"],
        "request_specs_path": str(request_specs_path),
        "request_specs_file_sha256": request_specs_file_sha256,
        "expected_native_request_sha256": request.request_sha256,
        "request_batch_sha256": request_batch["request_batch_sha256"],
        "launch_authorization_path": str(launch_path),
        "launch_authorization_file_sha256": launch_file_sha256,
        "snapshot_inventory_sha256": snapshot["inventory_sha256"],
        "package_bundle_sha256": packages["package_bundle_sha256"],
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "environment_identity_sha256": environment["environment_identity_sha256"],
        "expected_gpu_identity_sha256": expected_gpu_identity()["gpu_identity_sha256"],
        "claims": {
            "model_or_gpu_execution_performed": False,
            "sealed_control_evidence_established": False,
            "private_or_verbatim_cot_recovery_established": False,
            "latent_cot_like_trajectory_recovery_established": False,
            "affect_emotion_confidence_doubt_or_stress_recovery_established": False,
        },
    }
    freeze_path = output / "freeze-manifest.json"
    freeze_file_sha256 = _exclusive_write_json(freeze_path, freeze)
    return {
        "freeze_manifest_path": str(freeze_path),
        "freeze_manifest_file_sha256": freeze_file_sha256,
        "launch_authorization_path": str(launch_path),
        "launch_authorization_file_sha256": launch_file_sha256,
        "request_specs_path": str(request_specs_path),
        "request_specs_file_sha256": request_specs_file_sha256,
    }


def _load_request_specs(value: Any) -> list[adapter.NativeRequestSpec]:
    _require(isinstance(value, list) and bool(value), "request specs must be an array")
    specs: list[adapter.NativeRequestSpec] = []
    for position, item in enumerate(value):
        _require(
            isinstance(item, Mapping), f"request spec {position} must be an object"
        )
        try:
            specs.append(adapter.NativeRequestSpec(**copy.deepcopy(dict(item))))
        except TypeError as error:
            raise NativeSmokeError(
                f"request spec {position} fields invalid: {error}"
            ) from error
    return specs


def run_smoke(
    *,
    role: str,
    expected_adapter_config_sha256: str,
    expected_adapter_source_sha256: str,
    launch_path: Path,
    expected_launch_file_sha256: str,
    request_specs_path: Path,
    expected_request_specs_file_sha256: str,
    receipt_path: Path,
) -> dict[str, Any]:
    """Run one authenticated production batch and persist a non-sealed receipt."""

    _require(role in ROLES, "smoke role invalid")
    config_hash = _sha256(expected_adapter_config_sha256, "adapter config hash")
    source_hash = _sha256(expected_adapter_source_sha256, "adapter source hash")
    _require(
        sha256_file(adapter.CONFIG_PATH) == config_hash
        and sha256_file(adapter.SOURCE_PATH) == source_hash,
        "adapter config/source differs from external hash",
    )
    launch = _read_exact_json(
        launch_path, expected_launch_file_sha256, "launch authorization"
    )
    _require(
        isinstance(launch, Mapping) and launch.get("role") == role,
        "launch role invalid",
    )
    raw_specs = _read_exact_json(
        request_specs_path,
        expected_request_specs_file_sha256,
        "request specs",
    )
    specs = _load_request_specs(raw_specs)
    batch = adapter.execute_production_native_batch(
        expected_config_sha256=config_hash,
        launch_binding_path=launch_path,
        expected_launch_binding_sha256=_sha256(
            expected_launch_file_sha256, "launch file hash"
        ),
        role=role,
        request_specs=specs,
    )
    _require(
        len(batch.requests) == len(batch.results) == len(specs) == 1,
        "smoke batch cardinality invalid",
    )
    request = batch.requests[0]
    result = batch.results[0]
    request_body = adapter.runner_v3.validate_native_generation_request(request)
    result_body = adapter.runner_v3.validate_native_generation_result(
        request=request, result=result
    )
    parsed = _strict_json_bytes(result_body["text"].encode("utf-8"), "smoke output")
    _require(
        parsed == {"verdict": "ok"}, "smoke model output differs from finite schema"
    )
    receipt_body = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": RECEIPT_KIND,
        "status": "native_exact_token_smoke_complete_not_sealed_control_evidence",
        "role": role,
        "adapter_config_sha256": config_hash,
        "adapter_source_sha256": source_hash,
        "smoke_harness_source_sha256": sha256_file(Path(__file__).resolve()),
        "launch_authorization_file_sha256": _sha256(
            expected_launch_file_sha256, "launch file hash"
        ),
        "request_specs_file_sha256": _sha256(
            expected_request_specs_file_sha256, "request specs file hash"
        ),
        "native_request": request_body,
        "native_request_sha256": request.request_sha256,
        "native_result": result_body,
        "native_result_sha256": result.result_sha256,
        "preflight_receipt": copy.deepcopy(dict(batch.preflight_receipt.body)),
        "preflight_receipt_sha256": batch.preflight_receipt.receipt_sha256,
        "runtime_receipt": copy.deepcopy(dict(batch.runtime_receipt.body)),
        "runtime_receipt_sha256": batch.runtime_receipt.receipt_sha256,
        "claims": {
            "actual_model_execution_established_for_this_smoke": True,
            "exact_prompt_token_identity_established_for_this_smoke": True,
            "output_token_decode_parity_established_for_this_smoke": True,
            "sealed_control_evidence_established": False,
            "private_or_verbatim_cot_recovery_established": False,
            "latent_cot_like_trajectory_recovery_established": False,
            "affect_emotion_confidence_doubt_or_stress_recovery_established": False,
            "reserved_validation_accessed": False,
        },
    }
    envelope = {
        "receipt": receipt_body,
        "receipt_sha256": sha256_value(receipt_body),
    }
    receipt_file_sha256 = _exclusive_write_json(receipt_path, envelope)
    return {
        "receipt_path": str(Path(receipt_path).absolute()),
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_sha256": envelope["receipt_sha256"],
        "role": role,
        "native_result_sha256": result.result_sha256,
        "output": parsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--role", choices=ROLES, required=True)
    freeze.add_argument("--adapter-config-sha256", required=True)
    freeze.add_argument("--adapter-source-sha256", required=True)
    freeze.add_argument("--output-directory", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--role", choices=ROLES, required=True)
    run.add_argument("--adapter-config-sha256", required=True)
    run.add_argument("--adapter-source-sha256", required=True)
    run.add_argument("--launch", type=Path, required=True)
    run.add_argument("--launch-sha256", required=True)
    run.add_argument("--request-specs", type=Path, required=True)
    run.add_argument("--request-specs-sha256", required=True)
    run.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        result = freeze_smoke(
            role=args.role,
            expected_adapter_config_sha256=args.adapter_config_sha256,
            expected_adapter_source_sha256=args.adapter_source_sha256,
            output_directory=args.output_directory,
        )
    else:
        result = run_smoke(
            role=args.role,
            expected_adapter_config_sha256=args.adapter_config_sha256,
            expected_adapter_source_sha256=args.adapter_source_sha256,
            launch_path=args.launch,
            expected_launch_file_sha256=args.launch_sha256,
            request_specs_path=args.request_specs,
            expected_request_specs_file_sha256=args.request_specs_sha256,
            receipt_path=args.receipt,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
