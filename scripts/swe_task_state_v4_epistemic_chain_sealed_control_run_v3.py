#!/usr/bin/env python3
"""Authenticated staged runner for the V3 sealed development controls.

This controller fills the orchestration gap between the frozen bounded-ID
annotation runner, the production native adapter, and the outer executor.  It
uses immutable stage artifacts and caller-supplied whole-file SHA-256 values.
The initial primary round contains all nonempty completion decisions and all
novelty decisions; a second round contains only chain-detail requests selected
by authenticated first-round outputs.  One role therefore shares at most two
model loads while vLLM schedules prompts using its separately frozen
``max_num_seqs`` concurrency.

Only development semantic controls are in scope.  This script does not open
the control scoring key, reserved validation, or any affect/emotion source and
makes no private or verbatim chain-of-thought claim.
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
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import swe_task_state_v4_epistemic_chain_annotation_runner_v3 as runner  # noqa: E402
import swe_task_state_v4_epistemic_chain_native_smoke_v3 as smoke  # noqa: E402
import swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3 as adapter  # noqa: E402
import swe_task_state_v4_epistemic_chain_sealed_control_executor_v3 as executor  # noqa: E402


SCHEMA_VERSION = 1
INTERFACE_VERSION = 3
SUITE_INIT_KIND = "swe_task_state_v4_epistemic_chain_suite_init_v3"
BATCH_FREEZE_KIND = "swe_task_state_v4_epistemic_chain_batch_freeze_v3"
BATCH_ARTIFACT_KIND = "swe_task_state_v4_epistemic_chain_native_batch_artifact_v3"
CANDIDATE_MANIFEST_KIND = "swe_task_state_v4_epistemic_chain_candidate_manifest_v3"
CANDIDATE_LOCK_MANIFEST_KIND = (
    "swe_task_state_v4_epistemic_chain_candidate_lock_manifest_v3"
)
HOST_BYPASS_LOCK_KIND = (
    "swe_task_state_v4_epistemic_chain_deterministic_host_bypass_lock_v3"
)
PRIMARY_RECEIPT_KIND = "swe_task_state_v4_epistemic_chain_primary_receipt_v3"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PRIMARY_ROLES = ("independent_a", "independent_b")
PRIMARY_ROUNDS = ("initial", "detail")


class SealedControlRunError(RuntimeError):
    """Raised when a staged run cannot preserve its authenticated contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SealedControlRunError(message)


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


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return copy.deepcopy(dict(value))


def _sequence(value: Any, label: str) -> list[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be an array",
    )
    return copy.deepcopy(list(value))


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
        raise SealedControlRunError(f"cannot parse {label}: {error}") from error


def _assert_no_symlink_components(path: Path) -> None:
    candidate = Path(path).absolute()
    for component in (candidate, *candidate.parents):
        if component.exists():
            _require(
                not component.is_symlink(),
                f"path has symlink component: {component}",
            )


def _read_exact(path: Path, expected_sha256: str, label: str) -> bytes:
    expected = _sha256(expected_sha256, f"expected {label} hash")
    candidate = Path(path)
    _assert_no_symlink_components(candidate)
    resolved = candidate.resolve(strict=True)
    _require(resolved.is_file(), f"{label} is not a regular file")
    raw = resolved.read_bytes()
    _require(sha256_bytes(raw) == expected, f"{label} differs from external hash")
    return raw


def _read_exact_json(path: Path, expected_sha256: str, label: str) -> Any:
    return _strict_json_bytes(_read_exact(path, expected_sha256, label), label)


def _exclusive_write(path: Path, raw: bytes, label: str) -> str:
    candidate = Path(path).absolute()
    _assert_no_symlink_components(candidate)
    _require(candidate.parent.is_dir(), f"{label} parent directory is missing")
    try:
        descriptor = os.open(
            candidate,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except OSError as error:
        raise SealedControlRunError(f"cannot create {label}: {error}") from error
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, f"{label} write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_bytes(raw)


def _exclusive_write_json(path: Path, value: Any, label: str) -> str:
    return _exclusive_write(path, canonical_json_bytes(value), label)


def _load_codebook() -> dict[str, Any]:
    path = ROOT / "configs" / "swe_task_state_v4_epistemic_chain_codebook_v2.json"
    return _mapping(_strict_json_bytes(path.read_bytes(), "V2 codebook"), "V2 codebook")


def _authenticated_executor(
    *, expected_config_sha256: str, expected_source_sha256: str
) -> executor.AuthenticatedExecutorConfig:
    return executor.authenticate_executor_config(
        expected_config_sha256=_sha256(
            expected_config_sha256, "executor config hash"
        ),
        expected_source_sha256=_sha256(
            expected_source_sha256, "executor source hash"
        ),
    )


def _authenticated_adapter(
    *, expected_config_sha256: str, expected_source_sha256: str
) -> adapter.AuthenticatedAdapterConfig:
    authenticated = adapter.authenticate_adapter_config(
        path=adapter.CONFIG_PATH,
        expected_config_sha256=_sha256(
            expected_config_sha256, "adapter config hash"
        ),
    )
    _require(
        authenticated.source_sha256
        == _sha256(expected_source_sha256, "adapter source hash"),
        "adapter source differs from external hash",
    )
    return authenticated


def _load_context(
    *, authenticated_adapter: adapter.AuthenticatedAdapterConfig, role: str
) -> tuple[runner.AuthenticatedNativeGenerationContext, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = authenticated_adapter.value
    runtime = adapter.runtime_identity(config)
    environment = adapter.environment_identity(config)
    packages = adapter.package_identity_bundle(config)
    adapter._validate_runtime_import_specs(packages)
    snapshot_path, snapshot = adapter.resolve_and_verify_role_snapshot(
        config=config, role=role
    )
    context = smoke._load_verified_tokenizer_context(
        authenticated_config=authenticated_adapter,
        role=role,
        snapshot_path=snapshot_path,
    )
    return context, runtime, environment, packages, snapshot


def _request_spec(
    *,
    context: runner.AuthenticatedNativeGenerationContext,
    messages: Sequence[Mapping[str, str]],
    schema: Mapping[str, Any],
    seed: int,
    stage: str,
    annotation_pass: str,
    packet_id_sha256: str,
    source_id_sha256: str,
    lineage_bindings: Mapping[str, Any],
) -> adapter.NativeRequestSpec:
    request = runner.build_native_generation_request(
        context=context,
        messages=messages,
        schema=schema,
        seed=seed,
        stage=stage,
        annotation_pass=annotation_pass,
        packet_id_sha256=packet_id_sha256,
        source_id_sha256=source_id_sha256,
        lineage_bindings=lineage_bindings,
    )
    return adapter.NativeRequestSpec(
        messages=copy.deepcopy(list(messages)),
        schema=copy.deepcopy(dict(schema)),
        seed=seed,
        stage=stage,
        annotation_pass=annotation_pass,
        packet_id_sha256=packet_id_sha256,
        source_id_sha256=source_id_sha256,
        lineage_bindings=copy.deepcopy(dict(lineage_bindings)),
        expected_native_request_sha256=request.request_sha256,
    )


def _control_records_by_id(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _sequence(manifest["completion_records"], "completion records") + _sequence(
        manifest["novelty_records"], "novelty records"
    )
    result = {str(row["control_id"]): _mapping(row, "control record") for row in rows}
    _require(set(result) == set(executor.CONTROL_IDS), "control coverage changed")
    return result


def build_primary_initial_specs(
    *,
    role: str,
    control_manifest: Mapping[str, Any],
    context: runner.AuthenticatedNativeGenerationContext,
    seed: int,
) -> list[adapter.NativeRequestSpec]:
    """Build the exact 39-request first round for one primary lane."""

    _require(role in PRIMARY_ROLES, "primary role invalid")
    codebook = _load_codebook()
    records = _control_records_by_id(control_manifest)
    specs: list[adapter.NativeRequestSpec] = []
    for control_id in executor.CONTROL_IDS:
        record = records[control_id]
        packet = _mapping(record["packet"], f"{control_id} packet")
        if control_id == "C32":
            _require(
                packet["materialized_assistant_text"]["text"] == "",
                "C32 host bypass text changed",
            )
            continue
        if control_id.startswith("C"):
            catalog = _mapping(record["catalog"], f"{control_id} catalog")
            bundle = _mapping(
                catalog["candidate_unit_bundle"], f"{control_id} candidate bundle"
            )
            bundle_hash = str(catalog["candidate_unit_bundle_sha256"])
            units = runner.authenticate_candidate_unit_bundle(
                value=bundle,
                packet=packet,
                expected_bundle_sha256=bundle_hash,
            )
            messages = runner.build_independent_messages(
                packet=packet,
                codebook=codebook,
                annotation_pass="completion_chain",
                authenticated_units=units,
                response_route="decision",
            )
            schema = runner.completion_decision_response_schema(units)
            stage = "independent_completion_decision"
            annotation_pass = "completion_chain"
            lineage = {"candidate_unit_bundle_sha256": bundle_hash}
        else:
            messages = runner.build_independent_messages(
                packet=packet,
                codebook=codebook,
                annotation_pass="prefix_novelty",
            )
            schema = runner.novelty_decision_response_schema(codebook)
            stage = "independent_novelty_decision"
            annotation_pass = "prefix_novelty"
            lineage = {}
        specs.append(
            _request_spec(
                context=context,
                messages=messages,
                schema=schema,
                seed=seed,
                stage=stage,
                annotation_pass=annotation_pass,
                packet_id_sha256=str(packet["packet_id_sha256"]),
                source_id_sha256=str(packet["source_id_sha256"]),
                lineage_bindings=lineage,
            )
        )
    _require(len(specs) == 39, "primary initial request count changed")
    return specs


def _restore_schema_property_order(value: Any, *, label: str) -> dict[str, Any]:
    schema = copy.deepcopy(_mapping(value, label))
    properties = _mapping(schema.get("properties"), f"{label} properties")
    required = _sequence(schema.get("required"), f"{label} required")
    _require(
        all(isinstance(name, str) for name in required)
        and len(required) == len(set(required))
        and set(required) == set(properties),
        f"{label} required/properties mismatch",
    )
    schema["properties"] = {
        name: copy.deepcopy(properties[name]) for name in required
    }
    return schema


def _native_request(value: Any) -> runner.NativeGenerationRequest:
    item = _mapping(value, "stored native request")
    body = copy.deepcopy(_mapping(item["body"], "stored request body"))
    body["response_schema"] = _restore_schema_property_order(
        body.get("response_schema"), label="stored request response schema"
    )
    request = runner.NativeGenerationRequest(
        body=body,
        request_sha256=_sha256(item["request_sha256"], "stored request hash"),
    )
    runner.validate_native_generation_request(request)
    return request


def _native_result(
    value: Any, *, request: runner.NativeGenerationRequest
) -> runner.NativeGenerationResult:
    item = _mapping(value, "stored native result")
    result = runner.NativeGenerationResult(
        body=_mapping(item["body"], "stored result body"),
        result_sha256=_sha256(item["result_sha256"], "stored result hash"),
    )
    runner.validate_native_generation_result(request=request, result=result)
    return result


def validate_batch_artifact(
    value: Any, *, expected_role: str, expected_round: str
) -> dict[str, Any]:
    envelope = _mapping(value, "native batch artifact envelope")
    _require(
        set(envelope) == {"artifact", "artifact_sha256"},
        "native batch artifact envelope fields invalid",
    )
    artifact = _mapping(envelope["artifact"], "native batch artifact")
    _require(
        envelope["artifact_sha256"] == sha256_value(artifact),
        "native batch artifact self hash invalid",
    )
    _require(
        set(artifact)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "status",
            "suite_id",
            "role",
            "round",
            "freeze_manifest_file_sha256",
            "adapter_launch_file_sha256",
            "outer_launch_file_sha256",
            "request_count",
            "requests",
            "results",
            "preflight_receipt",
            "runtime_receipt",
            "completed_monotonic_ns",
            "claims",
        }
        and artifact["schema_version"] == SCHEMA_VERSION
        and artifact["interface_version"] == INTERFACE_VERSION
        and artifact["kind"] == BATCH_ARTIFACT_KIND
        and artifact["status"] == "authenticated_native_batch_complete"
        and artifact["role"] == expected_role
        and artifact["round"] == expected_round
        and artifact["claims"]
        == {
            "actual_model_execution": True,
            "gate_eligible": True,
            "reserved_validation_accessed": False,
            "private_or_verbatim_cot_recovery_established": False,
            "affect_emotion_confidence_doubt_or_stress_established": False,
        },
        "native batch artifact identity invalid",
    )
    requests = _sequence(artifact["requests"], "stored requests")
    results = _sequence(artifact["results"], "stored results")
    _require(
        artifact["request_count"] == len(requests) == len(results) > 0,
        "native batch artifact counts invalid",
    )
    for request_value, result_value in zip(requests, results, strict=True):
        request = _native_request(request_value)
        _native_result(result_value, request=request)
    for name in (
        "freeze_manifest_file_sha256",
        "adapter_launch_file_sha256",
        "outer_launch_file_sha256",
    ):
        _sha256(artifact[name], f"batch artifact {name}")
    for name in ("preflight_receipt", "runtime_receipt"):
        receipt = _mapping(artifact[name], name)
        _require(
            set(receipt) == {"body", "receipt_sha256"}
            and receipt["receipt_sha256"] == sha256_value(receipt["body"]),
            f"{name} hash invalid",
        )
    return artifact


def build_primary_detail_specs(
    *,
    role: str,
    control_manifest: Mapping[str, Any],
    context: runner.AuthenticatedNativeGenerationContext,
    seed: int,
    initial_batch: Mapping[str, Any],
) -> list[adapter.NativeRequestSpec]:
    """Build detail requests only for authenticated ``chain`` decisions."""

    _require(role in PRIMARY_ROLES, "primary role invalid")
    codebook = _load_codebook()
    records = _control_records_by_id(control_manifest)
    initial = validate_batch_artifact(
        initial_batch, expected_role=role, expected_round="initial"
    )
    by_packet: dict[str, tuple[runner.NativeGenerationRequest, runner.NativeGenerationResult]] = {}
    for request_value, result_value in zip(
        initial["requests"], initial["results"], strict=True
    ):
        request = _native_request(request_value)
        result = _native_result(result_value, request=request)
        packet_id = str(request.body["packet_id_sha256"])
        _require(packet_id not in by_packet, "duplicate first-round packet")
        by_packet[packet_id] = (request, result)
    specs: list[adapter.NativeRequestSpec] = []
    for control_id in executor.CONTROL_IDS:
        if not control_id.startswith("C") or control_id == "C32":
            continue
        record = records[control_id]
        packet = _mapping(record["packet"], f"{control_id} packet")
        packet_id = str(packet["packet_id_sha256"])
        _require(packet_id in by_packet, f"missing initial result for {control_id}")
        parent_request, parent_result = by_packet[packet_id]
        try:
            decision = runner.validate_completion_decision(
                runner._parse_json_object(
                    str(parent_result.body["text"]),
                    label=f"{control_id} completion decision",
                )
            )
        except runner.BoundedIdRunnerError:
            continue
        if decision != "chain":
            continue
        catalog = _mapping(record["catalog"], f"{control_id} catalog")
        bundle = _mapping(
            catalog["candidate_unit_bundle"], f"{control_id} candidate bundle"
        )
        bundle_hash = str(catalog["candidate_unit_bundle_sha256"])
        units = runner.authenticate_candidate_unit_bundle(
            value=bundle,
            packet=packet,
            expected_bundle_sha256=bundle_hash,
        )
        messages = runner.build_independent_messages(
            packet=packet,
            codebook=codebook,
            annotation_pass="completion_chain",
            authenticated_units=units,
            response_route="chain_detail",
        )
        schema = runner.completion_chain_detail_response_schema(codebook, units)
        specs.append(
            _request_spec(
                context=context,
                messages=messages,
                schema=schema,
                seed=seed,
                stage="independent_completion_chain_detail",
                annotation_pass="completion_chain",
                packet_id_sha256=packet_id,
                source_id_sha256=str(packet["source_id_sha256"]),
                lineage_bindings={
                    "candidate_unit_bundle_sha256": bundle_hash,
                    "parent_decision_request_sha256": parent_request.request_sha256,
                    "parent_decision_result_sha256": parent_result.result_sha256,
                },
            )
        )
    return specs


def _append_trace(
    *,
    journal_path: Path,
    journal_sha256: str,
    stage: str,
    event_type: str,
    artifact_class: str,
    artifact_id: str,
    path: str | None = None,
    expected_sha256: str | None = None,
    observed_sha256: str | None = None,
) -> str:
    appended = executor.append_read_trace_event(
        journal_path=journal_path,
        expected_journal_file_sha256=journal_sha256,
        stage=stage,
        event_type=event_type,
        artifact_class=artifact_class,
        artifact_id=artifact_id,
        path=path,
        expected_sha256=expected_sha256,
        observed_sha256=observed_sha256,
    )
    return str(appended["journal_file_sha256"])


def init_suite(
    *,
    suite_id: str,
    generation_root: Path,
    key_root: Path,
    expected_executor_config_sha256: str,
    expected_executor_source_sha256: str,
    expected_controller_source_sha256: str,
) -> dict[str, Any]:
    """Create authoritative inputs, live trace, and one precommitted nonce."""

    _require(
        sha256_file(Path(__file__).resolve())
        == _sha256(expected_controller_source_sha256, "controller source hash"),
        "controller source differs from external hash",
    )
    authenticated = _authenticated_executor(
        expected_config_sha256=expected_executor_config_sha256,
        expected_source_sha256=expected_executor_source_sha256,
    )
    generation, keys = executor.assert_disjoint_roots(
        generation_root=generation_root, key_root=key_root
    )
    _require(isinstance(suite_id, str) and len(suite_id) >= 16, "suite ID invalid")
    manifests = executor.regenerate_authoritative_inputs(authenticated)
    control_path = generation / "control-input.json"
    fixture_path = generation / "fixture-input.json"
    control_hash = _exclusive_write_json(
        control_path, manifests["control"]["manifest"], "control input"
    )
    fixture_hash = _exclusive_write_json(
        fixture_path, manifests["fixture"]["manifest"], "fixture input"
    )
    trace_path = generation / "read-trace.jsonl"
    journal_hash = executor.create_read_trace_journal(trace_path)
    for artifact_class, artifact_id, path, digest in (
        (
            "protocol_config",
            "executor_config",
            executor.CONFIG_PATH.resolve(),
            authenticated.config_sha256,
        ),
        ("control_input", "control_manifest_regenerated", control_path, control_hash),
        ("fixture_input", "fixture_manifest_regenerated", fixture_path, fixture_hash),
    ):
        journal_hash = _append_trace(
            journal_path=trace_path,
            journal_sha256=journal_hash,
            stage="freeze_helper",
            event_type="read",
            artifact_class=artifact_class,
            artifact_id=artifact_id,
            path=str(path.resolve(strict=True)),
            expected_sha256=digest,
            observed_sha256=digest,
        )
    freeze_completed = time.monotonic_ns()
    freeze_receipt = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": executor.FREEZE_RECEIPT_KIND,
        "status": "authoritative_inputs_frozen_before_model_execution",
        "suite_id": suite_id,
        "executor_config_sha256": authenticated.config_sha256,
        "executor_source_sha256": authenticated.source_sha256,
        "controller_source_sha256": expected_controller_source_sha256,
        "control_input_manifest_sha256": control_hash,
        "fixture_input_manifest_sha256": fixture_hash,
        "completed_monotonic_ns": freeze_completed,
        "claims": {
            "model_or_gpu_execution_performed": False,
            "fixture_key_read": False,
            "control_scoring_key_read": False,
            "reserved_validation_accessed": False,
        },
    }
    freeze_receipt_hash = sha256_value(freeze_receipt)
    freeze_receipt_path = generation / "freeze-receipt.json"
    freeze_receipt_file_hash = _exclusive_write_json(
        freeze_receipt_path, freeze_receipt, "freeze receipt"
    )
    journal_hash = _append_trace(
        journal_path=trace_path,
        journal_sha256=journal_hash,
        stage="freeze_helper",
        event_type="transition",
        artifact_class="stage",
        artifact_id="freeze_complete",
    )
    nonce = executor.precommit_suite_nonce(
        suite_id=suite_id,
        generation_root=generation,
        key_root=keys,
        nonce_secret_path=generation / "suite.nonce",
        single_use_marker_path=generation / "suite-nonce-precommit.marker.json",
        receipt_path=generation / "nonce-precommit-receipt.json",
        freeze_receipt_sha256=freeze_receipt_hash,
    )
    journal_hash = _append_trace(
        journal_path=trace_path,
        journal_sha256=journal_hash,
        stage="precommit_nonce",
        event_type="transition",
        artifact_class="stage",
        artifact_id="nonce_precommitted",
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": SUITE_INIT_KIND,
        "status": "suite_initialized_nonce_precommitted_no_model_execution",
        "suite_id": suite_id,
        "generation_root": str(generation),
        "key_root": str(keys),
        "executor_config_sha256": authenticated.config_sha256,
        "executor_source_sha256": authenticated.source_sha256,
        "controller_source_sha256": expected_controller_source_sha256,
        "control_input_path": str(control_path),
        "control_input_sha256": control_hash,
        "fixture_input_path": str(fixture_path),
        "fixture_input_sha256": fixture_hash,
        "freeze_receipt_path": str(freeze_receipt_path),
        "freeze_receipt_file_sha256": freeze_receipt_file_hash,
        "freeze_receipt_sha256": freeze_receipt_hash,
        "freeze_completed_monotonic_ns": freeze_completed,
        "nonce_secret_path": str(generation / "suite.nonce"),
        "nonce_secret_file_sha256": nonce["receipt"]["nonce_secret_file_sha256"],
        "suite_nonce_sha256": nonce["receipt"]["suite_nonce_sha256"],
        "nonce_precommit_receipt_path": str(
            generation / "nonce-precommit-receipt.json"
        ),
        "nonce_precommit_receipt_file_sha256": nonce["receipt_file_sha256"],
        "nonce_precommit_receipt_sha256": nonce["receipt_canonical_sha256"],
        "nonce_completed_monotonic_ns": nonce["receipt"]["completed_monotonic_ns"],
        "trace_journal_path": str(trace_path),
        "trace_journal_file_sha256": journal_hash,
        "claims": {
            "model_or_gpu_execution_performed": False,
            "fixture_generation_key_read": False,
            "control_scoring_key_read": False,
            "reserved_validation_accessed": False,
            "sealed_control_results_established": False,
        },
    }
    receipt_path = generation / "suite-init-receipt.json"
    receipt_file_hash = _exclusive_write_json(receipt_path, receipt, "suite init receipt")
    return {
        "suite_init_receipt_path": str(receipt_path),
        "suite_init_receipt_file_sha256": receipt_file_hash,
        "trace_journal_file_sha256": journal_hash,
        "control_input_sha256": control_hash,
        "fixture_input_sha256": fixture_hash,
        "suite_nonce_sha256": nonce["receipt"]["suite_nonce_sha256"],
    }


def _validate_suite_init(value: Any) -> dict[str, Any]:
    receipt = _mapping(value, "suite init receipt")
    _require(
        receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("interface_version") == INTERFACE_VERSION
        and receipt.get("kind") == SUITE_INIT_KIND
        and receipt.get("status")
        == "suite_initialized_nonce_precommitted_no_model_execution"
        and receipt.get("claims")
        == {
            "model_or_gpu_execution_performed": False,
            "fixture_generation_key_read": False,
            "control_scoring_key_read": False,
            "reserved_validation_accessed": False,
            "sealed_control_results_established": False,
        },
        "suite init receipt invalid",
    )
    for name in (
        "executor_config_sha256",
        "executor_source_sha256",
        "controller_source_sha256",
        "control_input_sha256",
        "fixture_input_sha256",
        "freeze_receipt_file_sha256",
        "freeze_receipt_sha256",
        "nonce_secret_file_sha256",
        "suite_nonce_sha256",
        "nonce_precommit_receipt_file_sha256",
        "nonce_precommit_receipt_sha256",
        "trace_journal_file_sha256",
    ):
        _sha256(receipt[name], f"suite init {name}")
    _require(
        sha256_file(Path(__file__).resolve())
        == receipt["controller_source_sha256"],
        "controller source differs from suite precommit",
    )
    return receipt


def _build_adapter_and_outer_launches(
    *,
    output_directory: Path,
    suite: Mapping[str, Any],
    role: str,
    round_name: str,
    request_specs: Sequence[adapter.NativeRequestSpec],
    authenticated_executor: executor.AuthenticatedExecutorConfig,
    authenticated_adapter: adapter.AuthenticatedAdapterConfig,
    runtime: Mapping[str, Any],
    environment: Mapping[str, Any],
    packages: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    prior_lock_sha256: str | None,
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any]]:
    descriptor = adapter.request_batch_descriptor(
        role=role,
        request_specs=request_specs,
        config=authenticated_adapter.value,
    )
    adapter_launch = smoke.build_launch_authorization(
        authenticated_config=authenticated_adapter,
        role=role,
        request_batch_sha256=descriptor["request_batch_sha256"],
        runtime_identity=runtime,
        environment_identity=environment,
        package_identity=packages,
        snapshot_inventory=snapshot,
        authorization_nonce_sha256=sha256_bytes(secrets.token_bytes(32)),
    )
    adapter_launch_path = output_directory / "adapter-launch.json"
    adapter_launch_file_hash = _exclusive_write_json(
        adapter_launch_path, adapter_launch, "adapter launch"
    )
    outer_nonce = sha256_bytes(secrets.token_bytes(32))
    stage = "run_adjudicator" if role == "adjudicator" else "run_primary"
    single_use_id = executor.sha256_value(
        {
            "domain": "v3-outer-launch-single-use-citrine",
            "suite_id": suite["suite_id"],
            "stage": stage,
            "role": role,
            "request_batch_sha256": descriptor["request_batch_sha256"],
            "authorization_nonce_sha256": outer_nonce,
        }
    )
    outer_launch = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": executor.OUTER_LAUNCH_KIND,
        "suite_id": suite["suite_id"],
        "generation_root": suite["generation_root"],
        "key_root": suite["key_root"],
        "stage": stage,
        "role": role,
        "execution_authorized": True,
        "model_access_authorized": True,
        "gpu_access_authorized": True,
        "artifact_output_authorized": True,
        "gate_evidence_authorized": True,
        "executor_config_sha256": authenticated_executor.config_sha256,
        "executor_source_sha256": authenticated_executor.source_sha256,
        "adapter_config_sha256": authenticated_adapter.config_sha256,
        "adapter_source_sha256": authenticated_adapter.source_sha256,
        "adapter_launch_binding_sha256": adapter_launch_file_hash,
        "request_batch_sha256": descriptor["request_batch_sha256"],
        "suite_nonce_sha256": suite["suite_nonce_sha256"],
        "nonce_precommit_receipt_sha256": suite[
            "nonce_precommit_receipt_sha256"
        ],
        "prior_lock_sha256": prior_lock_sha256,
        "authorization_nonce_sha256": outer_nonce,
        "single_use_authorization_id": single_use_id,
        "retry_permitted": False,
        "reserved_validation_accessed": False,
    }
    _require(set(outer_launch) == executor.OUTER_LAUNCH_FIELDS, "outer launch fields changed")
    outer_launch_path = output_directory / "outer-launch.json"
    outer_launch_file_hash = _exclusive_write_json(
        outer_launch_path, outer_launch, "outer launch"
    )
    paths = {
        "adapter_launch_path": str(adapter_launch_path),
        "outer_launch_path": str(outer_launch_path),
        "outer_consumption_marker_path": str(
            Path(suite["generation_root"])
            / f"outer-launch-{single_use_id}.consumed.json"
        ),
    }
    return (
        adapter_launch,
        adapter_launch_file_hash,
        outer_launch,
        outer_launch_file_hash,
        paths,
    )


def freeze_primary_round(
    *,
    role: str,
    round_name: str,
    suite_init_path: Path,
    expected_suite_init_file_sha256: str,
    expected_executor_config_sha256: str,
    expected_executor_source_sha256: str,
    expected_adapter_config_sha256: str,
    expected_adapter_source_sha256: str,
    expected_controller_source_sha256: str,
    output_directory: Path,
    initial_batch_path: Path | None = None,
    expected_initial_batch_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze one primary request round without model or GPU execution."""

    _require(role in PRIMARY_ROLES and round_name in PRIMARY_ROUNDS, "primary round invalid")
    _require(
        sha256_file(Path(__file__).resolve())
        == _sha256(expected_controller_source_sha256, "controller source hash"),
        "controller source differs from external hash",
    )
    suite = _validate_suite_init(
        _read_exact_json(
            suite_init_path, expected_suite_init_file_sha256, "suite init receipt"
        )
    )
    authenticated_executor = _authenticated_executor(
        expected_config_sha256=expected_executor_config_sha256,
        expected_source_sha256=expected_executor_source_sha256,
    )
    authenticated_adapter = _authenticated_adapter(
        expected_config_sha256=expected_adapter_config_sha256,
        expected_source_sha256=expected_adapter_source_sha256,
    )
    _require(
        suite["executor_config_sha256"] == authenticated_executor.config_sha256
        and suite["executor_source_sha256"] == authenticated_executor.source_sha256
        and suite["controller_source_sha256"] == expected_controller_source_sha256,
        "suite source/config binding changed",
    )
    control_manifest = _mapping(
        _read_exact_json(
            Path(suite["control_input_path"]),
            suite["control_input_sha256"],
            "control input",
        ),
        "control input",
    )
    context, runtime, environment, packages, snapshot = _load_context(
        authenticated_adapter=authenticated_adapter, role=role
    )
    seed = int(authenticated_adapter.value["roles"][role]["seed"])
    initial_batch_file_hash: str | None = None
    if round_name == "initial":
        _require(
            initial_batch_path is None and expected_initial_batch_file_sha256 is None,
            "initial round must not receive an initial batch",
        )
        specs = build_primary_initial_specs(
            role=role,
            control_manifest=control_manifest,
            context=context,
            seed=seed,
        )
    else:
        _require(
            initial_batch_path is not None
            and expected_initial_batch_file_sha256 is not None,
            "detail round requires the initial batch and external hash",
        )
        initial_batch_file_hash = _sha256(
            expected_initial_batch_file_sha256, "initial batch file hash"
        )
        initial_batch = _read_exact_json(
            initial_batch_path, initial_batch_file_hash, "initial batch artifact"
        )
        specs = build_primary_detail_specs(
            role=role,
            control_manifest=control_manifest,
            context=context,
            seed=seed,
            initial_batch=initial_batch,
        )
    output = Path(output_directory).absolute()
    _assert_no_symlink_components(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    if not specs:
        freeze = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "kind": BATCH_FREEZE_KIND,
            "status": "empty_detail_round_no_model_execution_required",
            "suite_id": suite["suite_id"],
            "role": role,
            "round": round_name,
            "suite_init_file_sha256": expected_suite_init_file_sha256,
            "controller_source_sha256": expected_controller_source_sha256,
            "adapter_config_sha256": authenticated_adapter.config_sha256,
            "adapter_source_sha256": authenticated_adapter.source_sha256,
            "initial_batch_file_sha256": initial_batch_file_hash,
            "request_count": 0,
            "request_specs_path": None,
            "request_specs_file_sha256": None,
            "adapter_launch_path": None,
            "adapter_launch_file_sha256": None,
            "outer_launch_path": None,
            "outer_launch_file_sha256": None,
            "outer_consumption_marker_path": None,
            "request_batch_sha256": None,
            "claims": {
                "model_or_gpu_execution_performed": False,
                "reserved_validation_accessed": False,
            },
        }
    else:
        request_specs_path = output / "request-specs.json"
        request_specs_file_hash = _exclusive_write_json(
            request_specs_path,
            [asdict(item) for item in specs],
            "request specs",
        )
        descriptor = adapter.request_batch_descriptor(
            role=role,
            request_specs=specs,
            config=authenticated_adapter.value,
        )
        (
            _adapter_launch,
            adapter_launch_file_hash,
            _outer_launch,
            outer_launch_file_hash,
            paths,
        ) = _build_adapter_and_outer_launches(
            output_directory=output,
            suite=suite,
            role=role,
            round_name=round_name,
            request_specs=specs,
            authenticated_executor=authenticated_executor,
            authenticated_adapter=authenticated_adapter,
            runtime=runtime,
            environment=environment,
            packages=packages,
            snapshot=snapshot,
            prior_lock_sha256=None,
        )
        freeze = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "kind": BATCH_FREEZE_KIND,
            "status": "authenticated_primary_batch_frozen_not_run",
            "suite_id": suite["suite_id"],
            "role": role,
            "round": round_name,
            "suite_init_file_sha256": expected_suite_init_file_sha256,
            "controller_source_sha256": expected_controller_source_sha256,
            "adapter_config_sha256": authenticated_adapter.config_sha256,
            "adapter_source_sha256": authenticated_adapter.source_sha256,
            "initial_batch_file_sha256": initial_batch_file_hash,
            "request_count": len(specs),
            "request_specs_path": str(request_specs_path),
            "request_specs_file_sha256": request_specs_file_hash,
            "adapter_launch_path": paths["adapter_launch_path"],
            "adapter_launch_file_sha256": adapter_launch_file_hash,
            "outer_launch_path": paths["outer_launch_path"],
            "outer_launch_file_sha256": outer_launch_file_hash,
            "outer_consumption_marker_path": paths[
                "outer_consumption_marker_path"
            ],
            "request_batch_sha256": descriptor["request_batch_sha256"],
            "claims": {
                "model_or_gpu_execution_performed": False,
                "reserved_validation_accessed": False,
            },
        }
    freeze_path = output / "freeze-manifest.json"
    freeze_file_hash = _exclusive_write_json(
        freeze_path, freeze, "batch freeze manifest"
    )
    return {
        "freeze_manifest_path": str(freeze_path),
        "freeze_manifest_file_sha256": freeze_file_hash,
        "request_count": freeze["request_count"],
        "status": freeze["status"],
    }


def _load_request_specs(value: Any) -> list[adapter.NativeRequestSpec]:
    specs: list[adapter.NativeRequestSpec] = []
    for position, item in enumerate(_sequence(value, "request specs")):
        loaded = copy.deepcopy(_mapping(item, f"request spec {position}"))
        # Canonical JSON sorts object keys, so a schema loaded from a frozen
        # request file no longer has the construction-time property order.
        # The bounded runner's validator expects `required` and `properties`
        # to iterate in the same order. Restore that representational detail
        # without changing the schema's canonical JSON or authenticated hash.
        loaded["schema"] = _restore_schema_property_order(
            loaded.get("schema"), label=f"request spec {position} schema"
        )
        specs.append(adapter.NativeRequestSpec(**loaded))
    return specs


def run_primary_round(
    *,
    freeze_manifest_path: Path,
    expected_freeze_manifest_file_sha256: str,
    suite_init_path: Path,
    expected_suite_init_file_sha256: str,
    expected_executor_config_sha256: str,
    expected_executor_source_sha256: str,
    expected_adapter_config_sha256: str,
    expected_adapter_source_sha256: str,
    expected_trace_journal_file_sha256: str | None,
) -> dict[str, Any]:
    """Execute one externally frozen primary batch and persist all evidence."""

    freeze = _mapping(
        _read_exact_json(
            freeze_manifest_path,
            expected_freeze_manifest_file_sha256,
            "batch freeze manifest",
        ),
        "batch freeze manifest",
    )
    _require(
        freeze.get("kind") == BATCH_FREEZE_KIND
        and freeze.get("role") in PRIMARY_ROLES
        and freeze.get("round") in PRIMARY_ROUNDS,
        "primary freeze identity invalid",
    )
    suite = _validate_suite_init(
        _read_exact_json(
            suite_init_path, expected_suite_init_file_sha256, "suite init receipt"
        )
    )
    _require(
        freeze["suite_id"] == suite["suite_id"]
        and freeze["suite_init_file_sha256"]
        == expected_suite_init_file_sha256,
        "freeze belongs to a different suite",
    )
    if freeze["request_count"] == 0:
        _require(
            freeze["status"] == "empty_detail_round_no_model_execution_required"
            and freeze["round"] == "detail",
            "empty native batch is not an allowed detail no-op",
        )
        return {
            "status": freeze["status"],
            "request_count": 0,
            "batch_artifact_path": None,
            "batch_artifact_file_sha256": None,
            "trace_journal_file_sha256": expected_trace_journal_file_sha256,
        }
    authenticated_executor = _authenticated_executor(
        expected_config_sha256=expected_executor_config_sha256,
        expected_source_sha256=expected_executor_source_sha256,
    )
    _authenticated_adapter(
        expected_config_sha256=expected_adapter_config_sha256,
        expected_source_sha256=expected_adapter_source_sha256,
    )
    specs = _load_request_specs(
        _read_exact_json(
            Path(freeze["request_specs_path"]),
            freeze["request_specs_file_sha256"],
            "request specs",
        )
    )
    _require(len(specs) == freeze["request_count"], "frozen request count changed")
    journal_hash = expected_trace_journal_file_sha256
    if freeze["round"] == "initial":
        _require(journal_hash is not None, "initial round requires trace journal hash")
        journal_hash = executor.read_exact_and_append_trace(
            journal_path=Path(suite["trace_journal_path"]),
            expected_journal_file_sha256=_sha256(
                journal_hash, "trace journal hash"
            ),
            artifact_path=Path(suite["control_input_path"]),
            expected_artifact_sha256=suite["control_input_sha256"],
            stage="run_primary",
            artifact_class="control_input",
            artifact_id=(
                "control_manifest_regenerated_for_a"
                if freeze["role"] == "independent_a"
                else "control_manifest_regenerated_for_b"
            ),
        )["journal_file_sha256"]
    batch = executor.delegate_native_adapter_batch(
        authenticated_executor_config=authenticated_executor,
        outer_launch_authorization_path=Path(freeze["outer_launch_path"]),
        expected_outer_launch_authorization_sha256=freeze[
            "outer_launch_file_sha256"
        ],
        outer_authorization_consumption_marker_path=Path(
            freeze["outer_consumption_marker_path"]
        ),
        expected_adapter_config_sha256=expected_adapter_config_sha256,
        expected_adapter_source_sha256=expected_adapter_source_sha256,
        launch_binding_path=Path(freeze["adapter_launch_path"]),
        expected_launch_binding_sha256=freeze["adapter_launch_file_sha256"],
        role=freeze["role"],
        request_specs=specs,
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": BATCH_ARTIFACT_KIND,
        "status": "authenticated_native_batch_complete",
        "suite_id": suite["suite_id"],
        "role": freeze["role"],
        "round": freeze["round"],
        "freeze_manifest_file_sha256": expected_freeze_manifest_file_sha256,
        "adapter_launch_file_sha256": freeze["adapter_launch_file_sha256"],
        "outer_launch_file_sha256": freeze["outer_launch_file_sha256"],
        "request_count": len(batch.requests),
        "requests": [asdict(item) for item in batch.requests],
        "results": [asdict(item) for item in batch.results],
        "preflight_receipt": asdict(batch.preflight_receipt),
        "runtime_receipt": asdict(batch.runtime_receipt),
        "completed_monotonic_ns": time.monotonic_ns(),
        "claims": {
            "actual_model_execution": True,
            "gate_eligible": True,
            "reserved_validation_accessed": False,
            "private_or_verbatim_cot_recovery_established": False,
            "affect_emotion_confidence_doubt_or_stress_established": False,
        },
    }
    envelope = {"artifact": artifact, "artifact_sha256": sha256_value(artifact)}
    artifact_path = Path(freeze_manifest_path).parent / "batch-artifact.json"
    artifact_file_hash = _exclusive_write_json(
        artifact_path, envelope, "native batch artifact"
    )
    validate_batch_artifact(
        envelope,
        expected_role=str(freeze["role"]),
        expected_round=str(freeze["round"]),
    )
    return {
        "status": artifact["status"],
        "request_count": artifact["request_count"],
        "batch_artifact_path": str(artifact_path),
        "batch_artifact_file_sha256": artifact_file_hash,
        "batch_artifact_sha256": envelope["artifact_sha256"],
        "trace_journal_file_sha256": journal_hash,
    }


def _result_callback_from_batches(
    batches: Sequence[Mapping[str, Any]],
) -> tuple[Any, dict[str, dict[str, Any]]]:
    by_request: dict[str, runner.NativeGenerationResult] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for batch in batches:
        for request_value, result_value in zip(
            batch["requests"], batch["results"], strict=True
        ):
            request = _native_request(request_value)
            result = _native_result(result_value, request=request)
            _require(
                request.request_sha256 not in by_request,
                "replayed request across primary rounds",
            )
            by_request[request.request_sha256] = result
            evidence[request.request_sha256] = {
                "outer_launch_authorization_sha256": batch[
                    "outer_launch_file_sha256"
                ],
                "launch_binding_sha256": batch["adapter_launch_file_sha256"],
                "preflight_receipt_sha256": batch["preflight_receipt"][
                    "receipt_sha256"
                ],
                "runtime_receipt_sha256": batch["runtime_receipt"][
                    "receipt_sha256"
                ],
                "native_result_sha256": result.result_sha256,
            }

    def generate(request: runner.NativeGenerationRequest) -> runner.NativeGenerationResult:
        runner.validate_native_generation_request(request)
        _require(
            request.request_sha256 in by_request,
            "materialization requested an unexecuted native request",
        )
        return by_request[request.request_sha256]

    return generate, evidence


def _public_proposal(record: Mapping[str, Any]) -> dict[str, Any]:
    proposal = record.get("raw_semantic_proposal")
    if isinstance(proposal, Mapping):
        return copy.deepcopy(dict(proposal))
    if record["annotation_pass"] == "completion_chain":
        return {
            "decision": "unknown",
            "unknown_reason": runner.COMPLETION_UNKNOWN_REASON,
        }
    return {
        "decision": "unknown",
        "unknown_reason": runner.NOVELTY_UNKNOWN_REASON,
    }


def _native_evidence_for_record(
    *,
    role: str,
    record: Mapping[str, Any],
    adapter_config_sha256: str,
    adapter_source_sha256: str,
    evidence_by_request: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    generation = _mapping(record["generation"], "record generation")
    if generation["model_invoked"] is False:
        return None
    stages = [generation["decision"]]
    if generation.get("chain_detail") is not None:
        stages.append(generation["chain_detail"])
    request_hashes = [str(stage["request_sha256"]) for stage in stages]
    result_hashes = [str(stage["result_sha256"]) for stage in stages]
    rows = [evidence_by_request[item] for item in request_hashes]

    def unique(name: str) -> list[str]:
        values: list[str] = []
        for row in rows:
            value = str(row[name])
            if value not in values:
                values.append(value)
        return values

    return {
        "adapter_role": role,
        "adapter_config_sha256": adapter_config_sha256,
        "adapter_source_sha256": adapter_source_sha256,
        "outer_launch_authorization_sha256s": unique(
            "outer_launch_authorization_sha256"
        ),
        "launch_binding_sha256s": unique("launch_binding_sha256"),
        "preflight_receipt_sha256s": unique("preflight_receipt_sha256"),
        "runtime_receipt_sha256s": unique("runtime_receipt_sha256"),
        "native_request_sha256s": request_hashes,
        "native_result_sha256s": result_hashes,
        "actual_model_execution": True,
        "model_loaded": True,
        "generation_performed": True,
        "gate_eligible": True,
    }


def finalize_primary(
    *,
    role: str,
    suite_init_path: Path,
    expected_suite_init_file_sha256: str,
    initial_batch_path: Path,
    expected_initial_batch_file_sha256: str,
    detail_freeze_manifest_path: Path,
    expected_detail_freeze_manifest_file_sha256: str,
    detail_batch_path: Path | None,
    expected_detail_batch_file_sha256: str | None,
    expected_adapter_config_sha256: str,
    expected_adapter_source_sha256: str,
    expected_trace_journal_file_sha256: str,
) -> dict[str, Any]:
    """Materialize one primary lane, lock records, and close its trace stage."""

    _require(role in PRIMARY_ROLES, "primary role invalid")
    suite = _validate_suite_init(
        _read_exact_json(
            suite_init_path, expected_suite_init_file_sha256, "suite init receipt"
        )
    )
    authenticated_adapter = _authenticated_adapter(
        expected_config_sha256=expected_adapter_config_sha256,
        expected_source_sha256=expected_adapter_source_sha256,
    )
    control_manifest = _mapping(
        _read_exact_json(
            Path(suite["control_input_path"]),
            suite["control_input_sha256"],
            "control input",
        ),
        "control input",
    )
    initial_envelope = _read_exact_json(
        initial_batch_path,
        expected_initial_batch_file_sha256,
        "initial batch artifact",
    )
    initial = validate_batch_artifact(
        initial_envelope, expected_role=role, expected_round="initial"
    )
    detail_freeze = _mapping(
        _read_exact_json(
            detail_freeze_manifest_path,
            expected_detail_freeze_manifest_file_sha256,
            "detail freeze manifest",
        ),
        "detail freeze manifest",
    )
    _require(
        detail_freeze.get("role") == role
        and detail_freeze.get("round") == "detail",
        "detail freeze role/round invalid",
    )
    batches = [initial]
    if detail_freeze["request_count"]:
        _require(
            detail_batch_path is not None and expected_detail_batch_file_sha256 is not None,
            "nonempty detail freeze requires a batch artifact",
        )
        detail_envelope = _read_exact_json(
            detail_batch_path,
            expected_detail_batch_file_sha256,
            "detail batch artifact",
        )
        batches.append(
            validate_batch_artifact(
                detail_envelope, expected_role=role, expected_round="detail"
            )
        )
    else:
        _require(
            detail_batch_path is None and expected_detail_batch_file_sha256 is None,
            "empty detail freeze cannot receive a batch artifact",
        )
    context, _runtime, _environment, _packages, _snapshot = _load_context(
        authenticated_adapter=authenticated_adapter, role=role
    )
    codebook = _load_codebook()
    completion_records = _sequence(
        control_manifest["completion_records"], "completion records"
    )
    novelty_records = _sequence(control_manifest["novelty_records"], "novelty records")
    completion_packets = [row["packet"] for row in completion_records]
    bundle_by_packet = {
        str(row["packet"]["packet_id_sha256"]): row["catalog"][
            "candidate_unit_bundle"
        ]
        for row in completion_records
    }
    bundle_hash_by_packet = {
        str(row["packet"]["packet_id_sha256"]): str(
            row["catalog"]["candidate_unit_bundle_sha256"]
        )
        for row in completion_records
    }
    generate, evidence_by_request = _result_callback_from_batches(batches)
    seed = int(authenticated_adapter.value["roles"][role]["seed"])
    records = runner.annotate_completion_packets(
        packets=completion_packets,
        codebook=codebook,
        role=role,
        generate_native=generate,
        generation_context=context,
        seed=seed,
        candidate_unit_bundles_by_packet=bundle_by_packet,
        expected_candidate_unit_bundle_sha256_by_packet=bundle_hash_by_packet,
    ) + runner.annotate_novelty_packets(
        packets=[row["packet"] for row in novelty_records],
        codebook=codebook,
        role=role,
        generate_native=generate,
        generation_context=context,
        seed=seed,
    )
    _require(len(records) == 40, "materialized primary record count changed")
    by_packet = {str(item["packet_id_sha256"]): item for item in records}
    ordered_records = [
        by_packet[
            str(_control_records_by_id(control_manifest)[control_id]["packet"]["packet_id_sha256"])
        ]
        for control_id in executor.CONTROL_IDS
    ]
    candidate_manifest = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": CANDIDATE_MANIFEST_KIND,
        "suite_id": suite["suite_id"],
        "role": role,
        "control_input_sha256": suite["control_input_sha256"],
        "record_count": 40,
        "ordered_records": ordered_records,
        "ordered_record_sha256s": [sha256_value(item) for item in ordered_records],
    }
    candidate_manifest_sha = sha256_value(candidate_manifest)
    locks = []
    for control_id, record in zip(executor.CONTROL_IDS, ordered_records, strict=True):
        if control_id == "C32":
            lock = {
                "schema_version": SCHEMA_VERSION,
                "interface_version": INTERFACE_VERSION,
                "kind": HOST_BYPASS_LOCK_KIND,
                "role": role,
                "control_id": "C32",
                "record_sha256": sha256_value(record),
                "manifest_sha256": candidate_manifest_sha,
                "execution_path": "deterministic_host_bypass",
                "native_model_record": False,
            }
        else:
            lock = runner.build_candidate_manifest_lock(
                record=record, manifest_sha256=candidate_manifest_sha
            )
        locks.append(lock)
    lock_hashes = [sha256_value(item) for item in locks]
    lock_manifest = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": CANDIDATE_LOCK_MANIFEST_KIND,
        "suite_id": suite["suite_id"],
        "role": role,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "lock_count": 40,
        "ordered_locks": locks,
        "ordered_lock_sha256s": lock_hashes,
    }
    candidate_manifest_path = Path(suite["generation_root"]) / f"{role}-candidate-manifest.json"
    lock_manifest_path = Path(suite["generation_root"]) / f"{role}-candidate-locks.json"
    candidate_manifest_file_sha = _exclusive_write_json(
        candidate_manifest_path, candidate_manifest, f"{role} candidate manifest"
    )
    lock_manifest_file_sha = _exclusive_write_json(
        lock_manifest_path, lock_manifest, f"{role} candidate lock manifest"
    )
    input_by_id = _control_records_by_id(control_manifest)
    rows: list[dict[str, Any]] = []
    for ordinal, (control_id, record) in enumerate(
        zip(executor.CONTROL_IDS, ordered_records, strict=True), start=1
    ):
        proposal = _public_proposal(record)
        bypass = control_id == "C32"
        rows.append(
            {
                "control_id": control_id,
                "ordinal": ordinal,
                "execution_path": "deterministic_host_bypass" if bypass else "native_model",
                "packet_sha256": input_by_id[control_id]["packet_sha256"],
                "result": proposal,
                "result_sha256": sha256_value(proposal),
                "native_evidence": _native_evidence_for_record(
                    role=role,
                    record=record,
                    adapter_config_sha256=authenticated_adapter.config_sha256,
                    adapter_source_sha256=authenticated_adapter.source_sha256,
                    evidence_by_request=evidence_by_request,
                ),
            }
        )
    completed = time.monotonic_ns()
    receipt_body = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": PRIMARY_RECEIPT_KIND,
        "status": "primary_role_complete_and_locked",
        "suite_id": suite["suite_id"],
        "role": role,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "candidate_manifest_file_sha256": candidate_manifest_file_sha,
        "candidate_lock_manifest_sha256": sha256_value(lock_manifest),
        "candidate_lock_manifest_file_sha256": lock_manifest_file_sha,
        "model_execution_count": 39,
        "host_bypass_count": 1,
        "result_count": 40,
        "ordered_results": rows,
        "completed_monotonic_ns": completed,
        "claims": {
            "actual_model_execution_authenticated": True,
            "reserved_validation_accessed": False,
            "control_scoring_key_read": False,
            "private_or_verbatim_cot_recovery_established": False,
            "affect_emotion_confidence_doubt_or_stress_established": False,
        },
    }
    receipt_sha = sha256_value(receipt_body)
    role_manifest = {
        "role": role,
        "model_execution_count": 39,
        "host_bypass_count": 1,
        "result_count": 40,
        "ordered_results": rows,
        "role_receipt_sha256": receipt_sha,
    }
    receipt = {
        "receipt": receipt_body,
        "receipt_sha256": receipt_sha,
        "role_manifest": role_manifest,
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_file_sha256": candidate_manifest_file_sha,
        "candidate_lock_manifest_path": str(lock_manifest_path),
        "candidate_lock_manifest_file_sha256": lock_manifest_file_sha,
    }
    receipt_path = Path(suite["generation_root"]) / f"{role}-primary-receipt.json"
    receipt_file_sha = _exclusive_write_json(
        receipt_path, receipt, f"{role} primary receipt"
    )
    journal_hash = _append_trace(
        journal_path=Path(suite["trace_journal_path"]),
        journal_sha256=expected_trace_journal_file_sha256,
        stage="run_primary",
        event_type="write",
        artifact_class="primary_output",
        artifact_id=f"primary_{role}_receipt",
        path=str(receipt_path.resolve(strict=True)),
        observed_sha256=receipt_file_sha,
    )
    journal_hash = _append_trace(
        journal_path=Path(suite["trace_journal_path"]),
        journal_sha256=journal_hash,
        stage="run_primary",
        event_type="transition",
        artifact_class="stage",
        artifact_id=f"primary_{role}_complete",
    )
    return {
        "primary_receipt_path": str(receipt_path),
        "primary_receipt_file_sha256": receipt_file_sha,
        "primary_receipt_sha256": receipt_sha,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "candidate_lock_manifest_file_sha256": lock_manifest_file_sha,
        "trace_journal_file_sha256": journal_hash,
        "chain_decision_count": sum(
            row["result"].get("decision") == "chain" for row in rows
        ),
        "unknown_count": sum(
            row["result"].get("decision") == "unknown" for row in rows
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init-suite")
    init.add_argument("--suite-id", required=True)
    init.add_argument("--generation-root", type=Path, required=True)
    init.add_argument("--key-root", type=Path, required=True)
    init.add_argument("--executor-config-sha256", required=True)
    init.add_argument("--executor-source-sha256", required=True)
    init.add_argument("--controller-source-sha256", required=True)

    freeze = subparsers.add_parser("freeze-primary")
    freeze.add_argument("--role", choices=PRIMARY_ROLES, required=True)
    freeze.add_argument("--round", choices=PRIMARY_ROUNDS, required=True)
    freeze.add_argument("--suite-init", type=Path, required=True)
    freeze.add_argument("--suite-init-sha256", required=True)
    freeze.add_argument("--executor-config-sha256", required=True)
    freeze.add_argument("--executor-source-sha256", required=True)
    freeze.add_argument("--adapter-config-sha256", required=True)
    freeze.add_argument("--adapter-source-sha256", required=True)
    freeze.add_argument("--controller-source-sha256", required=True)
    freeze.add_argument("--output-directory", type=Path, required=True)
    freeze.add_argument("--initial-batch", type=Path)
    freeze.add_argument("--initial-batch-sha256")

    run = subparsers.add_parser("run-primary")
    run.add_argument("--freeze-manifest", type=Path, required=True)
    run.add_argument("--freeze-manifest-sha256", required=True)
    run.add_argument("--suite-init", type=Path, required=True)
    run.add_argument("--suite-init-sha256", required=True)
    run.add_argument("--executor-config-sha256", required=True)
    run.add_argument("--executor-source-sha256", required=True)
    run.add_argument("--adapter-config-sha256", required=True)
    run.add_argument("--adapter-source-sha256", required=True)
    run.add_argument("--trace-journal-sha256")

    finalize = subparsers.add_parser("finalize-primary")
    finalize.add_argument("--role", choices=PRIMARY_ROLES, required=True)
    finalize.add_argument("--suite-init", type=Path, required=True)
    finalize.add_argument("--suite-init-sha256", required=True)
    finalize.add_argument("--initial-batch", type=Path, required=True)
    finalize.add_argument("--initial-batch-sha256", required=True)
    finalize.add_argument("--detail-freeze-manifest", type=Path, required=True)
    finalize.add_argument("--detail-freeze-manifest-sha256", required=True)
    finalize.add_argument("--detail-batch", type=Path)
    finalize.add_argument("--detail-batch-sha256")
    finalize.add_argument("--adapter-config-sha256", required=True)
    finalize.add_argument("--adapter-source-sha256", required=True)
    finalize.add_argument("--trace-journal-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init-suite":
        result = init_suite(
            suite_id=args.suite_id,
            generation_root=args.generation_root,
            key_root=args.key_root,
            expected_executor_config_sha256=args.executor_config_sha256,
            expected_executor_source_sha256=args.executor_source_sha256,
            expected_controller_source_sha256=args.controller_source_sha256,
        )
    elif args.command == "freeze-primary":
        result = freeze_primary_round(
            role=args.role,
            round_name=args.round,
            suite_init_path=args.suite_init,
            expected_suite_init_file_sha256=args.suite_init_sha256,
            expected_executor_config_sha256=args.executor_config_sha256,
            expected_executor_source_sha256=args.executor_source_sha256,
            expected_adapter_config_sha256=args.adapter_config_sha256,
            expected_adapter_source_sha256=args.adapter_source_sha256,
            expected_controller_source_sha256=args.controller_source_sha256,
            output_directory=args.output_directory,
            initial_batch_path=args.initial_batch,
            expected_initial_batch_file_sha256=args.initial_batch_sha256,
        )
    elif args.command == "run-primary":
        result = run_primary_round(
            freeze_manifest_path=args.freeze_manifest,
            expected_freeze_manifest_file_sha256=args.freeze_manifest_sha256,
            suite_init_path=args.suite_init,
            expected_suite_init_file_sha256=args.suite_init_sha256,
            expected_executor_config_sha256=args.executor_config_sha256,
            expected_executor_source_sha256=args.executor_source_sha256,
            expected_adapter_config_sha256=args.adapter_config_sha256,
            expected_adapter_source_sha256=args.adapter_source_sha256,
            expected_trace_journal_file_sha256=args.trace_journal_sha256,
        )
    else:
        result = finalize_primary(
            role=args.role,
            suite_init_path=args.suite_init,
            expected_suite_init_file_sha256=args.suite_init_sha256,
            initial_batch_path=args.initial_batch,
            expected_initial_batch_file_sha256=args.initial_batch_sha256,
            detail_freeze_manifest_path=args.detail_freeze_manifest,
            expected_detail_freeze_manifest_file_sha256=(
                args.detail_freeze_manifest_sha256
            ),
            detail_batch_path=args.detail_batch,
            expected_detail_batch_file_sha256=args.detail_batch_sha256,
            expected_adapter_config_sha256=args.adapter_config_sha256,
            expected_adapter_source_sha256=args.adapter_source_sha256,
            expected_trace_journal_file_sha256=args.trace_journal_sha256,
        )
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
