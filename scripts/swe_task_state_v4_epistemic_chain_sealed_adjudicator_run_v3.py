#!/usr/bin/env python3
"""Dual-lock, adjudicate, fixture-control, and final-lock V3 control runs.

This module is the adjudicator-side companion to
``sealed_control_run_v3``.  It authenticates and locks both primary lanes
before consuming the single suite nonce or reading the fixture-generation key.
It then builds one mixed adjudicator verdict batch for 39 semantic controls and
12 authored fixtures.  Later rounds contain only ``neither`` repairs and only
the completion repairs that selected ``chain``.

The control scoring key is never read here.  Reserved validation and affect,
emotion, confidence, doubt, and stress remain out of scope.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_key_draft_v3 as fixture_key  # noqa: E402
import swe_task_state_v4_epistemic_chain_adjudication_fixture_v3 as fixture  # noqa: E402
import swe_task_state_v4_epistemic_chain_annotation_runner_v3 as runner  # noqa: E402
import swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3 as adapter  # noqa: E402
import swe_task_state_v4_epistemic_chain_sealed_control_executor_v3 as executor  # noqa: E402
import swe_task_state_v4_epistemic_chain_sealed_control_run_v3 as control_run  # noqa: E402


SCHEMA_VERSION = 1
INTERFACE_VERSION = 3
DUAL_LOCK_ENVELOPE_KIND = "swe_task_state_v4_epistemic_chain_dual_primary_lock_envelope_v3"
ADJUDICATOR_PREPARATION_KIND = "swe_task_state_v4_epistemic_chain_adjudicator_preparation_v3"
ADJUDICATOR_FREEZE_KIND = "swe_task_state_v4_epistemic_chain_adjudicator_batch_freeze_v3"
ADJUDICATOR_RECEIPT_KIND = "swe_task_state_v4_epistemic_chain_adjudicator_receipt_v3"
FINAL_LOCK_ENVELOPE_KIND = "swe_task_state_v4_epistemic_chain_final_lock_envelope_v3"
ROUNDS = ("verdict", "repair", "detail")


class SealedAdjudicatorRunError(RuntimeError):
    """Raised when the adjudicator chronology or evidence fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SealedAdjudicatorRunError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return copy.deepcopy(dict(value))


def _sequence(value: Any, label: str) -> list[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be an array",
    )
    return copy.deepcopy(list(value))


def _load_json_file(path: Path) -> dict[str, Any]:
    return _mapping(
        control_run._strict_json_bytes(Path(path).read_bytes(), str(path)), str(path)
    )


def _validate_primary_receipt(value: Any, *, expected_role: str) -> dict[str, Any]:
    receipt = _mapping(value, f"{expected_role} primary receipt")
    _require(
        set(receipt)
        == {
            "receipt",
            "receipt_sha256",
            "role_manifest",
            "candidate_manifest_path",
            "candidate_manifest_file_sha256",
            "candidate_lock_manifest_path",
            "candidate_lock_manifest_file_sha256",
        },
        f"{expected_role} primary receipt fields invalid",
    )
    body = _mapping(receipt["receipt"], f"{expected_role} receipt body")
    _require(
        body.get("kind") == control_run.PRIMARY_RECEIPT_KIND
        and body.get("status") == "primary_role_complete_and_locked"
        and body.get("role") == expected_role
        and body.get("model_execution_count") == 39
        and body.get("host_bypass_count") == 1
        and body.get("result_count") == 40
        and receipt["receipt_sha256"] == control_run.sha256_value(body),
        f"{expected_role} primary receipt identity invalid",
    )
    role_manifest = _mapping(receipt["role_manifest"], f"{expected_role} role manifest")
    _require(
        role_manifest
        == {
            "role": expected_role,
            "model_execution_count": 39,
            "host_bypass_count": 1,
            "result_count": 40,
            "ordered_results": body["ordered_results"],
            "role_receipt_sha256": receipt["receipt_sha256"],
        },
        f"{expected_role} role manifest differs from receipt",
    )
    candidate_manifest = control_run._read_exact_json(
        Path(receipt["candidate_manifest_path"]),
        receipt["candidate_manifest_file_sha256"],
        f"{expected_role} candidate manifest",
    )
    candidate_locks = control_run._read_exact_json(
        Path(receipt["candidate_lock_manifest_path"]),
        receipt["candidate_lock_manifest_file_sha256"],
        f"{expected_role} candidate locks",
    )
    candidate_manifest = _mapping(candidate_manifest, "candidate manifest")
    candidate_locks = _mapping(candidate_locks, "candidate lock manifest")
    _require(
        candidate_manifest.get("kind") == control_run.CANDIDATE_MANIFEST_KIND
        and candidate_manifest.get("role") == expected_role
        and candidate_manifest.get("record_count") == 40
        and control_run.sha256_value(candidate_manifest)
        == body["candidate_manifest_sha256"]
        and candidate_locks.get("kind")
        == control_run.CANDIDATE_LOCK_MANIFEST_KIND
        and candidate_locks.get("role") == expected_role
        and candidate_locks.get("lock_count") == 40
        and candidate_locks.get("candidate_manifest_sha256")
        == body["candidate_manifest_sha256"],
        f"{expected_role} candidate records or locks invalid",
    )
    records = _sequence(candidate_manifest["ordered_records"], "candidate records")
    normalized_records: list[dict[str, Any]] = []
    for position, record_value in enumerate(records):
        record = _mapping(record_value, f"candidate record {position}")
        generation_value = record.get("generation")
        if isinstance(generation_value, Mapping):
            generation = _mapping(
                generation_value, f"candidate record {position} generation"
            )
            for stage_name in ("decision", "chain_detail"):
                stage_value = generation.get(stage_name)
                if not isinstance(stage_value, Mapping):
                    continue
                stage = _mapping(
                    stage_value,
                    f"candidate record {position} {stage_name} provenance",
                )
                request = _mapping(
                    stage.get("request"),
                    f"candidate record {position} {stage_name} request",
                )
                request["response_schema"] = (
                    control_run._restore_schema_property_order(
                        request.get("response_schema"),
                        label=(
                            f"candidate record {position} {stage_name} "
                            "response schema"
                        ),
                    )
                )
                stage["request"] = request
                generation[stage_name] = stage
            record["generation"] = generation
        normalized_records.append(record)
    records = normalized_records
    candidate_manifest["ordered_records"] = records
    locks = _sequence(candidate_locks["ordered_locks"], "candidate locks")
    lock_hashes = _sequence(
        candidate_locks["ordered_lock_sha256s"], "candidate lock hashes"
    )
    _require(len(records) == len(locks) == len(lock_hashes) == 40, "candidate coverage invalid")
    for control_id, record, lock, lock_hash in zip(
        executor.CONTROL_IDS, records, locks, lock_hashes, strict=True
    ):
        if control_id == "C32":
            _require(
                lock
                == {
                    "schema_version": SCHEMA_VERSION,
                    "interface_version": INTERFACE_VERSION,
                    "kind": control_run.HOST_BYPASS_LOCK_KIND,
                    "role": expected_role,
                    "control_id": "C32",
                    "record_sha256": control_run.sha256_value(record),
                    "manifest_sha256": body["candidate_manifest_sha256"],
                    "execution_path": "deterministic_host_bypass",
                    "native_model_record": False,
                },
                "C32 host-bypass lock invalid",
            )
        _require(
            control_run.sha256_value(lock) == lock_hash
            and lock["record_sha256"] == control_run.sha256_value(record)
            and lock["manifest_sha256"] == body["candidate_manifest_sha256"],
            "candidate lock binding invalid",
        )
    receipt["authenticated_candidate_manifest"] = candidate_manifest
    receipt["authenticated_candidate_locks"] = candidate_locks
    return receipt


def lock_primaries(
    *,
    suite_init_path: Path,
    expected_suite_init_file_sha256: str,
    primary_a_path: Path,
    expected_primary_a_file_sha256: str,
    primary_b_path: Path,
    expected_primary_b_file_sha256: str,
    expected_trace_journal_file_sha256: str,
    expected_adjudicator_source_sha256: str,
) -> dict[str, Any]:
    """Create the immutable dual-primary lock before any fixture key access."""

    _require(
        control_run.sha256_file(Path(__file__).resolve())
        == control_run._sha256(
            expected_adjudicator_source_sha256, "adjudicator source hash"
        ),
        "adjudicator source differs from external hash",
    )
    suite = control_run._validate_suite_init(
        control_run._read_exact_json(
            suite_init_path, expected_suite_init_file_sha256, "suite init receipt"
        )
    )
    primary_a = _validate_primary_receipt(
        control_run._read_exact_json(
            primary_a_path, expected_primary_a_file_sha256, "primary A receipt"
        ),
        expected_role="independent_a",
    )
    primary_b = _validate_primary_receipt(
        control_run._read_exact_json(
            primary_b_path, expected_primary_b_file_sha256, "primary B receipt"
        ),
        expected_role="independent_b",
    )
    _require(
        primary_a["receipt"]["suite_id"] == suite["suite_id"]
        == primary_b["receipt"]["suite_id"]
        and primary_a["receipt_sha256"] != primary_b["receipt_sha256"],
        "primary receipts are not distinct members of this suite",
    )
    completed = time.monotonic_ns()
    body = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": executor.DUAL_PRIMARY_LOCK_KIND,
        "status": "both_primary_roles_locked_fixture_key_unread",
        "suite_id": suite["suite_id"],
        "adjudicator_source_sha256": expected_adjudicator_source_sha256,
        "primary_independent_a_sha256": primary_a["receipt_sha256"],
        "primary_independent_a_file_sha256": expected_primary_a_file_sha256,
        "primary_independent_b_sha256": primary_b["receipt_sha256"],
        "primary_independent_b_file_sha256": expected_primary_b_file_sha256,
        "completed_monotonic_ns": completed,
        "claims": {
            "fixture_generation_key_read": False,
            "control_scoring_key_read": False,
            "reserved_validation_accessed": False,
        },
    }
    body_hash = control_run.sha256_value(body)
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": DUAL_LOCK_ENVELOPE_KIND,
        "lock": body,
        "lock_sha256": body_hash,
        "primary_a_path": str(Path(primary_a_path).resolve(strict=True)),
        "primary_a_file_sha256": expected_primary_a_file_sha256,
        "primary_b_path": str(Path(primary_b_path).resolve(strict=True)),
        "primary_b_file_sha256": expected_primary_b_file_sha256,
    }
    output_path = Path(suite["generation_root"]) / "dual-primary-lock.json"
    output_file_hash = control_run._exclusive_write_json(
        output_path, envelope, "dual primary lock"
    )
    journal_hash = control_run._append_trace(
        journal_path=Path(suite["trace_journal_path"]),
        journal_sha256=expected_trace_journal_file_sha256,
        stage="lock_primaries",
        event_type="write",
        artifact_class="dual_primary_lock",
        artifact_id="dual_primary_lock",
        path=str(output_path.resolve(strict=True)),
        observed_sha256=output_file_hash,
    )
    journal_hash = control_run._append_trace(
        journal_path=Path(suite["trace_journal_path"]),
        journal_sha256=journal_hash,
        stage="lock_primaries",
        event_type="transition",
        artifact_class="stage",
        artifact_id="dual_primary_lock_complete",
    )
    return {
        "dual_primary_lock_path": str(output_path),
        "dual_primary_lock_file_sha256": output_file_hash,
        "dual_primary_lock_sha256": body_hash,
        "completed_monotonic_ns": completed,
        "trace_journal_file_sha256": journal_hash,
    }


def _validate_dual_lock(value: Any, *, suite_id: str) -> dict[str, Any]:
    envelope = _mapping(value, "dual primary lock")
    _require(
        envelope.get("kind") == DUAL_LOCK_ENVELOPE_KIND,
        "dual primary lock envelope invalid",
    )
    body = _mapping(envelope["lock"], "dual primary lock body")
    _require(
        body.get("kind") == executor.DUAL_PRIMARY_LOCK_KIND
        and body.get("status") == "both_primary_roles_locked_fixture_key_unread"
        and body.get("suite_id") == suite_id
        and envelope["lock_sha256"] == control_run.sha256_value(body),
        "dual primary lock identity invalid",
    )
    return envelope


def _load_primary_material(
    dual_lock: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_a = _validate_primary_receipt(
        control_run._read_exact_json(
            Path(dual_lock["primary_a_path"]),
            dual_lock["primary_a_file_sha256"],
            "primary A receipt",
        ),
        expected_role="independent_a",
    )
    primary_b = _validate_primary_receipt(
        control_run._read_exact_json(
            Path(dual_lock["primary_b_path"]),
            dual_lock["primary_b_file_sha256"],
            "primary B receipt",
        ),
        expected_role="independent_b",
    )
    return primary_a, primary_b


def _control_preparation(
    *,
    control_id: str,
    input_record: Mapping[str, Any],
    primary_a: Mapping[str, Any],
    primary_b: Mapping[str, Any],
    codebook: Mapping[str, Any],
) -> dict[str, Any]:
    ordinal = executor.CONTROL_IDS.index(control_id)
    packet = _mapping(input_record["packet"], f"{control_id} packet")
    annotation_pass = "completion_chain" if control_id.startswith("C") else "prefix_novelty"
    units = None
    if annotation_pass == "completion_chain":
        catalog = _mapping(input_record["catalog"], f"{control_id} catalog")
        units = runner.authenticate_candidate_unit_bundle(
            value=catalog["candidate_unit_bundle"],
            packet=packet,
            expected_bundle_sha256=catalog["candidate_unit_bundle_sha256"],
        )
    records = (
        primary_a["authenticated_candidate_manifest"]["ordered_records"][ordinal],
        primary_b["authenticated_candidate_manifest"]["ordered_records"][ordinal],
    )
    locks = {
        "independent_a": primary_a["authenticated_candidate_locks"]["ordered_locks"][ordinal],
        "independent_b": primary_b["authenticated_candidate_locks"]["ordered_locks"][ordinal],
    }
    lock_hashes = {
        "independent_a": primary_a["authenticated_candidate_locks"]["ordered_lock_sha256s"][ordinal],
        "independent_b": primary_b["authenticated_candidate_locks"]["ordered_lock_sha256s"][ordinal],
    }
    blinded = runner.authenticate_and_blind_candidate_records(
        packet=packet,
        codebook=codebook,
        annotation_pass=annotation_pass,
        candidate_records=records,
        candidate_manifest_locks_by_role=locks,
        expected_candidate_manifest_lock_sha256_by_role=lock_hashes,
        authenticated_units=units,
    )
    _require(blinded.record_provenance is not None, "candidate provenance missing")
    lineage = {
        "candidate_order_sha256": blinded.order_sha256,
        "candidate_record_sha256s": [item["record_sha256"] for item in blinded.record_provenance],
        "candidate_manifest_sha256s": [item["manifest_sha256"] for item in blinded.record_provenance],
        "candidate_manifest_lock_sha256s": [item["manifest_lock_sha256"] for item in blinded.record_provenance],
        "candidate_parent_decision_request_sha256s": [item["decision_request_sha256"] for item in blinded.record_provenance],
        "candidate_parent_decision_result_sha256s": [item["decision_result_sha256"] for item in blinded.record_provenance],
        "candidate_parent_chain_detail_request_sha256s": [item["chain_detail_request_sha256"] for item in blinded.record_provenance],
        "candidate_parent_chain_detail_result_sha256s": [item["chain_detail_result_sha256"] for item in blinded.record_provenance],
    }
    return {
        "packet": packet,
        "annotation_pass": annotation_pass,
        "authenticated_units": units,
        "candidate_records": records,
        "candidate_locks": locks,
        "candidate_lock_hashes": lock_hashes,
        "blinded": blinded,
        "lineage": lineage,
    }


def _fixture_preparation(
    *,
    input_record: Mapping[str, Any],
    key_row: Mapping[str, Any],
    generation_contract: Mapping[str, Any],
    generation_context: runner.AuthenticatedNativeGenerationContext,
    fixture_config: Mapping[str, Any],
    codebook: Mapping[str, Any],
) -> fixture.PreparedFixture:
    annotation_pass = str(input_record["annotation_pass"])
    catalog = input_record.get("candidate_catalog")
    bundle = None if annotation_pass == "prefix_novelty" else catalog["candidate_unit_bundle"]
    bundle_hash = None if annotation_pass == "prefix_novelty" else input_record["candidate_unit_bundle_sha256"]
    return fixture._authenticate_fixture_lock(
        fixture_lock=key_row["fixture_lock"],
        expected_fixture_lock_sha256=key_row["fixture_lock_sha256"],
        packet=input_record["packet"],
        codebook=codebook,
        annotation_pass=annotation_pass,
        fixture_candidates=tuple(input_record["authored_candidates"]),
        fixture_nonce_sha256=key_row["fixture_nonce_sha256"],
        fixture_config=fixture_config,
        expected_fixture_config_sha256=control_run.sha256_file(fixture.DEFAULT_CONFIG_PATH),
        generation_contract=generation_contract,
        expected_generation_contract_sha256=key_row["generation_contract_sha256"],
        generation_context=generation_context,
        expected_verdict_seed=key_row["verdict_seed"],
        expected_repair_seed=key_row["repair_seed"],
        candidate_unit_bundle=bundle,
        expected_candidate_unit_bundle_sha256=bundle_hash,
    )


def _build_generation_contracts(
    *,
    fixture_input: Mapping[str, Any],
    suite: Mapping[str, Any],
    context: runner.AuthenticatedNativeGenerationContext,
    runtime: Mapping[str, Any],
    packages: Mapping[str, Any],
    authenticated_adapter: adapter.AuthenticatedAdapterConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    runtime_identity = {
        "runtime_kind": "vllm_native_exact_token_v3",
        "runtime_package_lock_sha256": packages["package_bundle_sha256"],
        "runtime_build_sha256": runtime["runtime_identity_sha256"],
    }
    adapter_identity = {
        "adapter_kind": adapter.CONFIG_KIND,
        "adapter_source_sha256": authenticated_adapter.source_sha256,
        "adapter_config_sha256": authenticated_adapter.config_sha256,
    }
    schedule = {row["case_id"]: row for row in executor.validate_executor_config(
        _load_json_file(executor.CONFIG_PATH)
    )["fixture_seed_schedule"]}
    contracts: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for case_id in executor.FIXTURE_IDS:
        nonce = executor.derive_fixture_case_nonce_sha256(
            suite_nonce_sha256=suite["suite_nonce_sha256"],
            case_id=case_id,
            fixture_input_manifest_sha256=suite["fixture_input_sha256"],
        )
        row = schedule[case_id]
        contract = fixture.build_fixture_generation_contract(
            generation_context=context,
            runtime_identity=runtime_identity,
            native_adapter_identity=adapter_identity,
            verdict_seed=row["verdict_seed"],
            repair_seed=row["repair_seed"],
            fixture_nonce_sha256=nonce,
            outer_nonce_precommit_receipt_sha256=suite["nonce_precommit_receipt_sha256"],
        )
        contracts[case_id] = contract
        hashes[case_id] = control_run.sha256_value(contract)
    return contracts, hashes


def prepare_and_freeze_verdict(
    *,
    suite_init_path: Path,
    expected_suite_init_file_sha256: str,
    dual_primary_lock_path: Path,
    expected_dual_primary_lock_file_sha256: str,
    expected_executor_config_sha256: str,
    expected_executor_source_sha256: str,
    expected_adapter_config_sha256: str,
    expected_adapter_source_sha256: str,
    expected_adjudicator_source_sha256: str,
    expected_trace_journal_file_sha256: str,
    output_directory: Path,
    fixture_key_output_path: Path,
) -> dict[str, Any]:
    """Consume the nonce, read/materialize the fixture key, and freeze 51 verdicts."""

    _require(
        control_run.sha256_file(Path(__file__).resolve())
        == expected_adjudicator_source_sha256,
        "adjudicator source differs from external hash",
    )
    suite = control_run._validate_suite_init(
        control_run._read_exact_json(
            suite_init_path, expected_suite_init_file_sha256, "suite init receipt"
        )
    )
    dual_lock = _validate_dual_lock(
        control_run._read_exact_json(
            dual_primary_lock_path,
            expected_dual_primary_lock_file_sha256,
            "dual primary lock",
        ),
        suite_id=suite["suite_id"],
    )
    authenticated_executor = control_run._authenticated_executor(
        expected_config_sha256=expected_executor_config_sha256,
        expected_source_sha256=expected_executor_source_sha256,
    )
    authenticated_adapter = control_run._authenticated_adapter(
        expected_config_sha256=expected_adapter_config_sha256,
        expected_source_sha256=expected_adapter_source_sha256,
    )
    primary_a, primary_b = _load_primary_material(dual_lock)
    control_input = _mapping(
        control_run._read_exact_json(
            Path(suite["control_input_path"]), suite["control_input_sha256"], "control input"
        ),
        "control input",
    )
    fixture_input = _mapping(
        control_run._read_exact_json(
            Path(suite["fixture_input_path"]), suite["fixture_input_sha256"], "fixture input"
        ),
        "fixture input",
    )
    journal_hash = expected_trace_journal_file_sha256
    for artifact_path, digest, artifact_class, artifact_id in (
        (Path(suite["control_input_path"]), suite["control_input_sha256"], "control_input", "control_manifest_regenerated_for_j"),
        (Path(suite["fixture_input_path"]), suite["fixture_input_sha256"], "fixture_input", "fixture_manifest_regenerated_for_j"),
        (Path(fixture_key.__file__).resolve(), control_run.sha256_file(Path(fixture_key.__file__).resolve()), "fixture_key_materializer", "fixture_key_materializer_authenticated"),
        (fixture_key.CONFIG_PATH.resolve(), control_run.sha256_file(fixture_key.CONFIG_PATH), "fixture_generation_key", "fixture_key_opened_once"),
    ):
        read = executor.read_exact_and_append_trace(
            journal_path=Path(suite["trace_journal_path"]),
            expected_journal_file_sha256=journal_hash,
            artifact_path=artifact_path,
            expected_artifact_sha256=digest,
            stage="run_adjudicator",
            artifact_class=artifact_class,
            artifact_id=artifact_id,
        )
        journal_hash = read["journal_file_sha256"]
    nonce_consumption = executor.consume_suite_nonce_once(
        suite_id=suite["suite_id"],
        generation_root=Path(suite["generation_root"]),
        key_root=Path(suite["key_root"]),
        nonce_secret_path=Path(suite["nonce_secret_path"]),
        expected_nonce_secret_file_sha256=suite["nonce_secret_file_sha256"],
        expected_suite_nonce_sha256=suite["suite_nonce_sha256"],
        consumption_marker_path=Path(suite["generation_root"]) / "suite-nonce-consumed.json",
        dual_primary_lock_sha256=dual_lock["lock_sha256"],
    )
    journal_hash = control_run._append_trace(
        journal_path=Path(suite["trace_journal_path"]),
        journal_sha256=journal_hash,
        stage="run_adjudicator",
        event_type="consume",
        artifact_class="nonce_secret",
        artifact_id="suite_nonce_consumed_once",
        path=str((Path(suite["generation_root"]) / "suite-nonce-consumed.json").resolve(strict=True)),
        observed_sha256=nonce_consumption["marker_file_sha256"],
    )
    context, runtime, environment, packages, snapshot = control_run._load_context(
        authenticated_adapter=authenticated_adapter, role="adjudicator"
    )
    codebook = control_run._load_codebook()
    fixture_config = _load_json_file(fixture.DEFAULT_CONFIG_PATH)
    contracts, contract_hashes = _build_generation_contracts(
        fixture_input=fixture_input,
        suite=suite,
        context=context,
        runtime=runtime,
        packages=packages,
        authenticated_adapter=authenticated_adapter,
    )
    schedule = {row["case_id"]: row for row in authenticated_executor.value["fixture_seed_schedule"]}
    fixture_key_envelope = fixture_key.materialize_key_draft(
        input_manifest_envelope={"manifest": fixture_input, "manifest_sha256": suite["fixture_input_sha256"]},
        expected_input_manifest_sha256=suite["fixture_input_sha256"],
        expected_key_config_sha256=authenticated_executor.value["key_contracts"]["fixture_generation_key"]["config_sha256"],
        externally_precommitted_suite_nonce_sha256=suite["suite_nonce_sha256"],
        outer_nonce_precommit_receipt_sha256=suite["nonce_precommit_receipt_sha256"],
        fixture_config=fixture_config,
        expected_fixture_config_sha256=control_run.sha256_file(fixture.DEFAULT_CONFIG_PATH),
        generation_context=context,
        generation_contracts=contracts,
        expected_generation_contract_sha256s=contract_hashes,
        verdict_seeds={case_id: row["verdict_seed"] for case_id, row in schedule.items()},
        repair_seeds={case_id: row["repair_seed"] for case_id, row in schedule.items()},
    )
    fixture_key_file_sha = control_run._exclusive_write_json(
        fixture_key_output_path, fixture_key_envelope, "materialized fixture key"
    )
    output = Path(output_directory).absolute()
    control_run._assert_no_symlink_components(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    contracts_path = output / "fixture-generation-contracts.json"
    contracts_file_sha = control_run._exclusive_write_json(
        contracts_path, contracts, "fixture generation contracts"
    )
    control_by_id = control_run._control_records_by_id(control_input)
    specs: list[adapter.NativeRequestSpec] = []
    for control_id in executor.MODEL_CONTROL_IDS:
        prepared = _control_preparation(
            control_id=control_id,
            input_record=control_by_id[control_id],
            primary_a=primary_a,
            primary_b=primary_b,
            codebook=codebook,
        )
        messages = runner.build_adjudication_messages(
            packet=prepared["packet"],
            codebook=codebook,
            annotation_pass=prepared["annotation_pass"],
            blinded_candidates=prepared["blinded"],
            authenticated_units=prepared["authenticated_units"],
        )
        specs.append(control_run._request_spec(
            context=context,
            messages=messages,
            schema=runner.adjudication_response_schema(),
            seed=int(authenticated_adapter.value["roles"]["adjudicator"]["seed"]),
            stage="adjudication_verdict",
            annotation_pass=prepared["annotation_pass"],
            packet_id_sha256=prepared["packet"]["packet_id_sha256"],
            source_id_sha256=prepared["packet"]["source_id_sha256"],
            lineage_bindings=prepared["lineage"],
        ))
    fixture_input_by_id = {row["case_id"]: row for row in fixture_input["records"]}
    fixture_key_by_id = {row["case_id"]: row for row in fixture_key_envelope["manifest"]["cases"]}
    for case_id in executor.FIXTURE_IDS:
        prepared = _fixture_preparation(
            input_record=fixture_input_by_id[case_id],
            key_row=fixture_key_by_id[case_id],
            generation_contract=contracts[case_id],
            generation_context=context,
            fixture_config=fixture_config,
            codebook=codebook,
        )
        lineage = fixture._stage_lineage(
            prepared=prepared,
            fixture_lock_sha256=fixture_key_by_id[case_id]["fixture_lock_sha256"],
            model_input_stage="adjudication_verdict",
        )
        specs.append(control_run._request_spec(
            context=context,
            messages=prepared.verdict_messages,
            schema=prepared.verdict_schema,
            seed=prepared.generation_contract.verdict_seed,
            stage="control_fixture_adjudication_verdict",
            annotation_pass=str(prepared.packet["annotation_pass"]),
            packet_id_sha256=str(prepared.packet["packet_id_sha256"]),
            source_id_sha256=str(prepared.packet["source_id_sha256"]),
            lineage_bindings=lineage,
        ))
    _require(len(specs) == 51, "adjudicator verdict request count changed")
    request_specs_path = output / "request-specs.json"
    request_specs_file_sha = control_run._exclusive_write_json(
        request_specs_path, [asdict(item) for item in specs], "adjudicator request specs"
    )
    (_adapter_launch, adapter_launch_sha, _outer_launch, outer_launch_sha, paths) = (
        control_run._build_adapter_and_outer_launches(
            output_directory=output,
            suite=suite,
            role="adjudicator",
            round_name="verdict",
            request_specs=specs,
            authenticated_executor=authenticated_executor,
            authenticated_adapter=authenticated_adapter,
            runtime=runtime,
            environment=environment,
            packages=packages,
            snapshot=snapshot,
            prior_lock_sha256=dual_lock["lock_sha256"],
        )
    )
    preparation = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": ADJUDICATOR_PREPARATION_KIND,
        "status": "fixture_key_materialized_nonce_consumed_verdict_frozen_not_run",
        "suite_id": suite["suite_id"],
        "adjudicator_source_sha256": expected_adjudicator_source_sha256,
        "dual_primary_lock_path": str(Path(dual_primary_lock_path).resolve(strict=True)),
        "dual_primary_lock_file_sha256": expected_dual_primary_lock_file_sha256,
        "dual_primary_lock_sha256": dual_lock["lock_sha256"],
        "fixture_key_path": str(Path(fixture_key_output_path).resolve(strict=True)),
        "fixture_key_file_sha256": fixture_key_file_sha,
        "fixture_key_manifest_sha256": fixture_key_envelope["manifest_sha256"],
        "fixture_generation_contracts_path": str(contracts_path),
        "fixture_generation_contracts_file_sha256": contracts_file_sha,
        "nonce_consumption_receipt_sha256": nonce_consumption["marker_file_sha256"],
        "primary_a_path": dual_lock["primary_a_path"],
        "primary_a_file_sha256": dual_lock["primary_a_file_sha256"],
        "primary_b_path": dual_lock["primary_b_path"],
        "primary_b_file_sha256": dual_lock["primary_b_file_sha256"],
        "claims": {
            "fixture_generation_key_read_once": True,
            "suite_nonce_consumed_once": True,
            "control_scoring_key_read": False,
            "reserved_validation_accessed": False,
        },
    }
    preparation_path = output / "preparation.json"
    preparation_file_sha = control_run._exclusive_write_json(
        preparation_path, preparation, "adjudicator preparation"
    )
    descriptor = adapter.request_batch_descriptor(
        role="adjudicator", request_specs=specs, config=authenticated_adapter.value
    )
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": ADJUDICATOR_FREEZE_KIND,
        "status": "authenticated_adjudicator_batch_frozen_not_run",
        "suite_id": suite["suite_id"],
        "round": "verdict",
        "suite_init_file_sha256": expected_suite_init_file_sha256,
        "preparation_path": str(preparation_path),
        "preparation_file_sha256": preparation_file_sha,
        "request_count": 51,
        "request_specs_path": str(request_specs_path),
        "request_specs_file_sha256": request_specs_file_sha,
        "adapter_launch_path": paths["adapter_launch_path"],
        "adapter_launch_file_sha256": adapter_launch_sha,
        "outer_launch_path": paths["outer_launch_path"],
        "outer_launch_file_sha256": outer_launch_sha,
        "outer_consumption_marker_path": paths["outer_consumption_marker_path"],
        "request_batch_sha256": descriptor["request_batch_sha256"],
        "claims": {"model_or_gpu_execution_performed": False, "control_scoring_key_read": False},
    }
    freeze_path = output / "freeze-manifest.json"
    freeze_file_sha = control_run._exclusive_write_json(
        freeze_path, freeze, "adjudicator freeze manifest"
    )
    return {
        "freeze_manifest_path": str(freeze_path),
        "freeze_manifest_file_sha256": freeze_file_sha,
        "preparation_file_sha256": preparation_file_sha,
        "request_count": 51,
        "fixture_key_file_sha256": fixture_key_file_sha,
        "trace_journal_file_sha256": journal_hash,
    }


def _validate_preparation(value: Any, *, suite_id: str) -> dict[str, Any]:
    preparation = _mapping(value, "adjudicator preparation")
    _require(
        preparation.get("schema_version") == SCHEMA_VERSION
        and preparation.get("interface_version") == INTERFACE_VERSION
        and preparation.get("kind") == ADJUDICATOR_PREPARATION_KIND
        and preparation.get("status")
        == "fixture_key_materialized_nonce_consumed_verdict_frozen_not_run"
        and preparation.get("suite_id") == suite_id
        and preparation.get("claims")
        == {
            "fixture_generation_key_read_once": True,
            "suite_nonce_consumed_once": True,
            "control_scoring_key_read": False,
            "reserved_validation_accessed": False,
        },
        "adjudicator preparation invalid",
    )
    for name in (
        "adjudicator_source_sha256",
        "dual_primary_lock_file_sha256",
        "dual_primary_lock_sha256",
        "fixture_key_file_sha256",
        "fixture_key_manifest_sha256",
        "fixture_generation_contracts_file_sha256",
        "nonce_consumption_receipt_sha256",
        "primary_a_file_sha256",
        "primary_b_file_sha256",
    ):
        control_run._sha256(preparation[name], f"preparation {name}")
    return preparation


def _validate_adjudicator_freeze(value: Any, *, expected_round: str) -> dict[str, Any]:
    freeze = _mapping(value, "adjudicator freeze manifest")
    _require(
        freeze.get("schema_version") == SCHEMA_VERSION
        and freeze.get("interface_version") == INTERFACE_VERSION
        and freeze.get("kind") == ADJUDICATOR_FREEZE_KIND
        and freeze.get("round") == expected_round
        and freeze.get("status")
        in {
            "authenticated_adjudicator_batch_frozen_not_run",
            "empty_adjudicator_round_no_model_execution_required",
        }
        and freeze.get("claims")
        == {
            "model_or_gpu_execution_performed": False,
            "control_scoring_key_read": False,
        },
        "adjudicator freeze identity invalid",
    )
    return freeze


def run_adjudicator_round(
    *,
    freeze_manifest_path: Path,
    expected_freeze_manifest_file_sha256: str,
    suite_init_path: Path,
    expected_suite_init_file_sha256: str,
    expected_executor_config_sha256: str,
    expected_executor_source_sha256: str,
    expected_adapter_config_sha256: str,
    expected_adapter_source_sha256: str,
) -> dict[str, Any]:
    """Execute one frozen adjudicator round and persist full native evidence."""

    freeze = _mapping(
        control_run._read_exact_json(
            freeze_manifest_path,
            expected_freeze_manifest_file_sha256,
            "adjudicator freeze manifest",
        ),
        "adjudicator freeze manifest",
    )
    _require(freeze.get("round") in ROUNDS, "adjudicator round invalid")
    freeze = _validate_adjudicator_freeze(
        freeze, expected_round=str(freeze["round"])
    )
    suite = control_run._validate_suite_init(
        control_run._read_exact_json(
            suite_init_path, expected_suite_init_file_sha256, "suite init receipt"
        )
    )
    _require(
        freeze["suite_id"] == suite["suite_id"]
        and freeze["suite_init_file_sha256"] == expected_suite_init_file_sha256,
        "adjudicator freeze belongs to a different suite",
    )
    if freeze["request_count"] == 0:
        return {
            "status": freeze["status"],
            "request_count": 0,
            "batch_artifact_path": None,
            "batch_artifact_file_sha256": None,
        }
    authenticated_executor = control_run._authenticated_executor(
        expected_config_sha256=expected_executor_config_sha256,
        expected_source_sha256=expected_executor_source_sha256,
    )
    control_run._authenticated_adapter(
        expected_config_sha256=expected_adapter_config_sha256,
        expected_source_sha256=expected_adapter_source_sha256,
    )
    specs = control_run._load_request_specs(
        control_run._read_exact_json(
            Path(freeze["request_specs_path"]),
            freeze["request_specs_file_sha256"],
            "adjudicator request specs",
        )
    )
    _require(len(specs) == freeze["request_count"], "adjudicator request count changed")
    batch = executor.delegate_native_adapter_batch(
        authenticated_executor_config=authenticated_executor,
        outer_launch_authorization_path=Path(freeze["outer_launch_path"]),
        expected_outer_launch_authorization_sha256=freeze["outer_launch_file_sha256"],
        outer_authorization_consumption_marker_path=Path(
            freeze["outer_consumption_marker_path"]
        ),
        expected_adapter_config_sha256=expected_adapter_config_sha256,
        expected_adapter_source_sha256=expected_adapter_source_sha256,
        launch_binding_path=Path(freeze["adapter_launch_path"]),
        expected_launch_binding_sha256=freeze["adapter_launch_file_sha256"],
        role="adjudicator",
        request_specs=specs,
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": control_run.BATCH_ARTIFACT_KIND,
        "status": "authenticated_native_batch_complete",
        "suite_id": suite["suite_id"],
        "role": "adjudicator",
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
    envelope = {"artifact": artifact, "artifact_sha256": control_run.sha256_value(artifact)}
    control_run.validate_batch_artifact(
        envelope, expected_role="adjudicator", expected_round=str(freeze["round"])
    )
    output_path = Path(freeze_manifest_path).parent / "batch-artifact.json"
    output_file_sha = control_run._exclusive_write_json(
        output_path, envelope, "adjudicator batch artifact"
    )
    return {
        "status": artifact["status"],
        "request_count": artifact["request_count"],
        "batch_artifact_path": str(output_path),
        "batch_artifact_file_sha256": output_file_sha,
        "batch_artifact_sha256": envelope["artifact_sha256"],
    }


def _load_adjudication_material(
    *,
    suite: Mapping[str, Any],
    preparation: Mapping[str, Any],
    authenticated_adapter: adapter.AuthenticatedAdapterConfig,
) -> dict[str, Any]:
    dual_lock = _validate_dual_lock(
        control_run._read_exact_json(
            Path(preparation["dual_primary_lock_path"]),
            preparation["dual_primary_lock_file_sha256"],
            "dual primary lock",
        ),
        suite_id=suite["suite_id"],
    )
    primary_a, primary_b = _load_primary_material(dual_lock)
    control_input = _mapping(
        control_run._read_exact_json(
            Path(suite["control_input_path"]), suite["control_input_sha256"], "control input"
        ),
        "control input",
    )
    fixture_input = _mapping(
        control_run._read_exact_json(
            Path(suite["fixture_input_path"]), suite["fixture_input_sha256"], "fixture input"
        ),
        "fixture input",
    )
    fixture_key_envelope = _mapping(
        control_run._read_exact_json(
            Path(preparation["fixture_key_path"]),
            preparation["fixture_key_file_sha256"],
            "materialized fixture key",
        ),
        "materialized fixture key",
    )
    _require(
        fixture_key_envelope.get("manifest_sha256")
        == preparation["fixture_key_manifest_sha256"],
        "fixture key manifest hash changed",
    )
    contracts = _mapping(
        control_run._read_exact_json(
            Path(preparation["fixture_generation_contracts_path"]),
            preparation["fixture_generation_contracts_file_sha256"],
            "fixture generation contracts",
        ),
        "fixture generation contracts",
    )
    context, runtime, environment, packages, snapshot = control_run._load_context(
        authenticated_adapter=authenticated_adapter, role="adjudicator"
    )
    return {
        "dual_lock": dual_lock,
        "primary_a": primary_a,
        "primary_b": primary_b,
        "control_input": control_input,
        "fixture_input": fixture_input,
        "fixture_key": fixture_key_envelope,
        "contracts": contracts,
        "context": context,
        "runtime": runtime,
        "environment": environment,
        "packages": packages,
        "snapshot": snapshot,
        "codebook": control_run._load_codebook(),
        "fixture_config": _load_json_file(fixture.DEFAULT_CONFIG_PATH),
    }


def _batch_by_packet(
    envelope: Mapping[str, Any], *, expected_round: str
) -> tuple[dict[str, tuple[runner.NativeGenerationRequest, runner.NativeGenerationResult]], dict[str, Any]]:
    artifact = control_run.validate_batch_artifact(
        envelope, expected_role="adjudicator", expected_round=expected_round
    )
    by_packet: dict[str, tuple[runner.NativeGenerationRequest, runner.NativeGenerationResult]] = {}
    for request_value, result_value in zip(artifact["requests"], artifact["results"], strict=True):
        request = control_run._native_request(request_value)
        result = control_run._native_result(result_value, request=request)
        packet_id = str(request.body["packet_id_sha256"])
        _require(packet_id not in by_packet, f"duplicate {expected_round} packet")
        by_packet[packet_id] = (request, result)
    return by_packet, artifact


def _parse_verdict(result: runner.NativeGenerationResult) -> str | None:
    try:
        return runner.validate_adjudication_verdict(
            runner._parse_json_object(str(result.body["text"]), label="adjudication verdict")
        )
    except runner.BoundedIdRunnerError:
        return None


def _parse_completion_decision(result: runner.NativeGenerationResult) -> str | None:
    try:
        return runner.validate_completion_decision(
            runner._parse_json_object(str(result.body["text"]), label="completion repair")
        )
    except runner.BoundedIdRunnerError:
        return None


def freeze_followup_round(
    *,
    round_name: str,
    suite_init_path: Path,
    expected_suite_init_file_sha256: str,
    preparation_path: Path,
    expected_preparation_file_sha256: str,
    verdict_batch_path: Path,
    expected_verdict_batch_file_sha256: str,
    expected_executor_config_sha256: str,
    expected_executor_source_sha256: str,
    expected_adapter_config_sha256: str,
    expected_adapter_source_sha256: str,
    expected_adjudicator_source_sha256: str,
    output_directory: Path,
    repair_batch_path: Path | None = None,
    expected_repair_batch_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze only verdict-selected repairs or chain-selected repair details."""

    _require(round_name in {"repair", "detail"}, "follow-up round invalid")
    _require(
        control_run.sha256_file(Path(__file__).resolve())
        == expected_adjudicator_source_sha256,
        "adjudicator source differs from external hash",
    )
    suite = control_run._validate_suite_init(
        control_run._read_exact_json(
            suite_init_path, expected_suite_init_file_sha256, "suite init receipt"
        )
    )
    preparation = _validate_preparation(
        control_run._read_exact_json(
            preparation_path, expected_preparation_file_sha256, "adjudicator preparation"
        ),
        suite_id=suite["suite_id"],
    )
    authenticated_executor = control_run._authenticated_executor(
        expected_config_sha256=expected_executor_config_sha256,
        expected_source_sha256=expected_executor_source_sha256,
    )
    authenticated_adapter = control_run._authenticated_adapter(
        expected_config_sha256=expected_adapter_config_sha256,
        expected_source_sha256=expected_adapter_source_sha256,
    )
    material = _load_adjudication_material(
        suite=suite,
        preparation=preparation,
        authenticated_adapter=authenticated_adapter,
    )
    verdict_envelope = _mapping(
        control_run._read_exact_json(
            verdict_batch_path, expected_verdict_batch_file_sha256, "verdict batch"
        ),
        "verdict batch",
    )
    verdict_by_packet, _verdict_artifact = _batch_by_packet(
        verdict_envelope, expected_round="verdict"
    )
    repair_by_packet: dict[str, tuple[runner.NativeGenerationRequest, runner.NativeGenerationResult]] = {}
    if round_name == "detail":
        _require(
            (repair_batch_path is None) == (expected_repair_batch_file_sha256 is None),
            "detail repair batch path/hash must be supplied together",
        )
        if repair_batch_path is not None:
            repair_envelope = _mapping(
                control_run._read_exact_json(
                    repair_batch_path,
                    expected_repair_batch_file_sha256,
                    "repair batch",
                ),
                "repair batch",
            )
            repair_by_packet, _repair_artifact = _batch_by_packet(
                repair_envelope, expected_round="repair"
            )
    else:
        _require(
            repair_batch_path is None and expected_repair_batch_file_sha256 is None,
            "repair round must not receive a repair batch",
        )
    codebook = material["codebook"]
    context = material["context"]
    control_by_id = control_run._control_records_by_id(material["control_input"])
    specs: list[adapter.NativeRequestSpec] = []
    for control_id in executor.MODEL_CONTROL_IDS:
        prepared = _control_preparation(
            control_id=control_id,
            input_record=control_by_id[control_id],
            primary_a=material["primary_a"],
            primary_b=material["primary_b"],
            codebook=codebook,
        )
        packet_id = str(prepared["packet"]["packet_id_sha256"])
        verdict_request, verdict_result = verdict_by_packet[packet_id]
        if _parse_verdict(verdict_result) != "neither":
            continue
        if round_name == "repair":
            messages = runner.build_neither_repair_messages(
                packet=prepared["packet"],
                codebook=codebook,
                annotation_pass=prepared["annotation_pass"],
                blinded_candidates=prepared["blinded"],
                authenticated_units=prepared["authenticated_units"],
                response_route="decision",
            )
            schema = runner.neither_repair_response_schema(
                codebook=codebook,
                annotation_pass=prepared["annotation_pass"],
                authenticated_units=prepared["authenticated_units"],
                response_route="decision",
            )
            stage = (
                "adjudication_completion_repair_decision"
                if prepared["annotation_pass"] == "completion_chain"
                else "adjudication_novelty_repair_decision"
            )
            lineage = {
                **prepared["lineage"],
                "parent_verdict_request_sha256": verdict_request.request_sha256,
                "parent_verdict_result_sha256": verdict_result.result_sha256,
            }
        else:
            if prepared["annotation_pass"] != "completion_chain" or packet_id not in repair_by_packet:
                continue
            repair_request, repair_result = repair_by_packet[packet_id]
            if _parse_completion_decision(repair_result) != "chain":
                continue
            messages = runner.build_neither_repair_messages(
                packet=prepared["packet"],
                codebook=codebook,
                annotation_pass="completion_chain",
                blinded_candidates=prepared["blinded"],
                authenticated_units=prepared["authenticated_units"],
                response_route="chain_detail",
            )
            schema = runner.neither_repair_response_schema(
                codebook=codebook,
                annotation_pass="completion_chain",
                authenticated_units=prepared["authenticated_units"],
                response_route="chain_detail",
            )
            stage = "adjudication_completion_repair_chain_detail"
            lineage = {
                **prepared["lineage"],
                "parent_verdict_request_sha256": verdict_request.request_sha256,
                "parent_verdict_result_sha256": verdict_result.result_sha256,
                "parent_repair_decision_request_sha256": repair_request.request_sha256,
                "parent_repair_decision_result_sha256": repair_result.result_sha256,
            }
        specs.append(control_run._request_spec(
            context=context,
            messages=messages,
            schema=schema,
            seed=int(authenticated_adapter.value["roles"]["adjudicator"]["seed"]),
            stage=stage,
            annotation_pass=prepared["annotation_pass"],
            packet_id_sha256=packet_id,
            source_id_sha256=str(prepared["packet"]["source_id_sha256"]),
            lineage_bindings=lineage,
        ))
    fixture_input_by_id = {row["case_id"]: row for row in material["fixture_input"]["records"]}
    fixture_key_by_id = {row["case_id"]: row for row in material["fixture_key"]["manifest"]["cases"]}
    for case_id in executor.FIXTURE_IDS:
        key_row = fixture_key_by_id[case_id]
        prepared = _fixture_preparation(
            input_record=fixture_input_by_id[case_id],
            key_row=key_row,
            generation_contract=material["contracts"][case_id],
            generation_context=context,
            fixture_config=material["fixture_config"],
            codebook=codebook,
        )
        packet_id = str(prepared.packet["packet_id_sha256"])
        verdict_request, verdict_result = verdict_by_packet[packet_id]
        if _parse_verdict(verdict_result) != "neither":
            continue
        if round_name == "repair":
            messages = prepared.repair_decision_messages
            schema = prepared.repair_decision_schema
            stage = (
                "control_fixture_completion_repair_decision"
                if prepared.packet["annotation_pass"] == "completion_chain"
                else "control_fixture_novelty_repair_decision"
            )
            lineage = {
                **fixture._stage_lineage(
                    prepared=prepared,
                    fixture_lock_sha256=key_row["fixture_lock_sha256"],
                    model_input_stage="neither_repair_decision",
                ),
                "parent_verdict_request_sha256": verdict_request.request_sha256,
                "parent_verdict_result_sha256": verdict_result.result_sha256,
            }
        else:
            if prepared.packet["annotation_pass"] != "completion_chain" or packet_id not in repair_by_packet:
                continue
            repair_request, repair_result = repair_by_packet[packet_id]
            if _parse_completion_decision(repair_result) != "chain":
                continue
            _require(
                prepared.repair_detail_messages is not None
                and prepared.repair_detail_schema is not None,
                "fixture chain repair detail unavailable",
            )
            messages = prepared.repair_detail_messages
            schema = prepared.repair_detail_schema
            stage = "control_fixture_completion_repair_chain_detail"
            lineage = {
                **fixture._stage_lineage(
                    prepared=prepared,
                    fixture_lock_sha256=key_row["fixture_lock_sha256"],
                    model_input_stage="neither_repair_chain_detail",
                ),
                "parent_verdict_request_sha256": verdict_request.request_sha256,
                "parent_verdict_result_sha256": verdict_result.result_sha256,
                "parent_repair_decision_request_sha256": repair_request.request_sha256,
                "parent_repair_decision_result_sha256": repair_result.result_sha256,
            }
        specs.append(control_run._request_spec(
            context=context,
            messages=messages,
            schema=schema,
            seed=prepared.generation_contract.repair_seed,
            stage=stage,
            annotation_pass=str(prepared.packet["annotation_pass"]),
            packet_id_sha256=packet_id,
            source_id_sha256=str(prepared.packet["source_id_sha256"]),
            lineage_bindings=lineage,
        ))
    output = Path(output_directory).absolute()
    control_run._assert_no_symlink_components(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    request_specs_path: str | None = None
    request_specs_file_sha: str | None = None
    adapter_launch_path: str | None = None
    adapter_launch_sha: str | None = None
    outer_launch_path: str | None = None
    outer_launch_sha: str | None = None
    outer_marker_path: str | None = None
    request_batch_sha: str | None = None
    if specs:
        specs_path = output / "request-specs.json"
        request_specs_file_sha = control_run._exclusive_write_json(
            specs_path, [asdict(item) for item in specs], "adjudicator follow-up specs"
        )
        request_specs_path = str(specs_path)
        descriptor = adapter.request_batch_descriptor(
            role="adjudicator", request_specs=specs, config=authenticated_adapter.value
        )
        request_batch_sha = descriptor["request_batch_sha256"]
        (_adapter_launch, adapter_launch_sha, _outer_launch, outer_launch_sha, paths) = (
            control_run._build_adapter_and_outer_launches(
                output_directory=output,
                suite=suite,
                role="adjudicator",
                round_name=round_name,
                request_specs=specs,
                authenticated_executor=authenticated_executor,
                authenticated_adapter=authenticated_adapter,
                runtime=material["runtime"],
                environment=material["environment"],
                packages=material["packages"],
                snapshot=material["snapshot"],
                prior_lock_sha256=preparation["dual_primary_lock_sha256"],
            )
        )
        adapter_launch_path = paths["adapter_launch_path"]
        outer_launch_path = paths["outer_launch_path"]
        outer_marker_path = paths["outer_consumption_marker_path"]
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": ADJUDICATOR_FREEZE_KIND,
        "status": (
            "authenticated_adjudicator_batch_frozen_not_run"
            if specs
            else "empty_adjudicator_round_no_model_execution_required"
        ),
        "suite_id": suite["suite_id"],
        "round": round_name,
        "suite_init_file_sha256": expected_suite_init_file_sha256,
        "preparation_path": str(Path(preparation_path).resolve(strict=True)),
        "preparation_file_sha256": expected_preparation_file_sha256,
        "verdict_batch_file_sha256": expected_verdict_batch_file_sha256,
        "repair_batch_file_sha256": expected_repair_batch_file_sha256,
        "request_count": len(specs),
        "request_specs_path": request_specs_path,
        "request_specs_file_sha256": request_specs_file_sha,
        "adapter_launch_path": adapter_launch_path,
        "adapter_launch_file_sha256": adapter_launch_sha,
        "outer_launch_path": outer_launch_path,
        "outer_launch_file_sha256": outer_launch_sha,
        "outer_consumption_marker_path": outer_marker_path,
        "request_batch_sha256": request_batch_sha,
        "claims": {
            "model_or_gpu_execution_performed": False,
            "control_scoring_key_read": False,
        },
    }
    freeze_path = output / "freeze-manifest.json"
    freeze_file_sha = control_run._exclusive_write_json(
        freeze_path, freeze, "adjudicator follow-up freeze"
    )
    return {
        "freeze_manifest_path": str(freeze_path),
        "freeze_manifest_file_sha256": freeze_file_sha,
        "request_count": len(specs),
        "status": freeze["status"],
    }


def _load_optional_round(
    *,
    freeze_path: Path,
    expected_freeze_file_sha256: str,
    expected_round: str,
    batch_path: Path | None,
    expected_batch_file_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    freeze = _validate_adjudicator_freeze(
        control_run._read_exact_json(
            freeze_path, expected_freeze_file_sha256, f"{expected_round} freeze"
        ),
        expected_round=expected_round,
    )
    if freeze["request_count"] == 0:
        _require(
            batch_path is None and expected_batch_file_sha256 is None,
            f"empty {expected_round} round received a batch",
        )
        return freeze, None
    _require(
        batch_path is not None and expected_batch_file_sha256 is not None,
        f"nonempty {expected_round} round lacks its batch",
    )
    envelope = _mapping(
        control_run._read_exact_json(
            batch_path, expected_batch_file_sha256, f"{expected_round} batch"
        ),
        f"{expected_round} batch",
    )
    artifact = control_run.validate_batch_artifact(
        envelope, expected_role="adjudicator", expected_round=expected_round
    )
    _require(
        artifact["request_count"] == freeze["request_count"],
        f"{expected_round} freeze/batch count differs",
    )
    return freeze, artifact


def _native_evidence_from_stages(
    *,
    record: Mapping[str, Any],
    adapter_config_sha256: str,
    adapter_source_sha256: str,
    evidence_by_request: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    stages = [record["verdict_generation"]]
    repair = record.get("repair_generation")
    if repair is not None:
        stages.append(repair["decision"])
        if repair.get("chain_detail") is not None:
            stages.append(repair["chain_detail"])
    request_hashes = [str(stage["request_sha256"]) for stage in stages]
    result_hashes = [str(stage["result_sha256"]) for stage in stages]
    evidence_rows = [evidence_by_request[item] for item in request_hashes]

    def unique(name: str) -> list[str]:
        values: list[str] = []
        for row in evidence_rows:
            value = str(row[name])
            if value not in values:
                values.append(value)
        return values

    return {
        "adapter_role": "adjudicator",
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


def _public_result(record: Mapping[str, Any]) -> dict[str, Any]:
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


def finalize_adjudicator(
    *,
    suite_init_path: Path,
    expected_suite_init_file_sha256: str,
    preparation_path: Path,
    expected_preparation_file_sha256: str,
    verdict_batch_path: Path,
    expected_verdict_batch_file_sha256: str,
    repair_freeze_path: Path,
    expected_repair_freeze_file_sha256: str,
    repair_batch_path: Path | None,
    expected_repair_batch_file_sha256: str | None,
    detail_freeze_path: Path,
    expected_detail_freeze_file_sha256: str,
    detail_batch_path: Path | None,
    expected_detail_batch_file_sha256: str | None,
    expected_adapter_config_sha256: str,
    expected_adapter_source_sha256: str,
    expected_trace_journal_file_sha256: str,
) -> dict[str, Any]:
    """Materialize controls and fixtures from all authenticated model rounds."""

    suite = control_run._validate_suite_init(
        control_run._read_exact_json(
            suite_init_path, expected_suite_init_file_sha256, "suite init receipt"
        )
    )
    preparation = _validate_preparation(
        control_run._read_exact_json(
            preparation_path, expected_preparation_file_sha256, "adjudicator preparation"
        ),
        suite_id=suite["suite_id"],
    )
    authenticated_adapter = control_run._authenticated_adapter(
        expected_config_sha256=expected_adapter_config_sha256,
        expected_source_sha256=expected_adapter_source_sha256,
    )
    material = _load_adjudication_material(
        suite=suite,
        preparation=preparation,
        authenticated_adapter=authenticated_adapter,
    )
    verdict_envelope = _mapping(
        control_run._read_exact_json(
            verdict_batch_path,
            expected_verdict_batch_file_sha256,
            "verdict batch",
        ),
        "verdict batch",
    )
    _verdict_by_packet, verdict_artifact = _batch_by_packet(
        verdict_envelope, expected_round="verdict"
    )
    _repair_freeze, repair_artifact = _load_optional_round(
        freeze_path=repair_freeze_path,
        expected_freeze_file_sha256=expected_repair_freeze_file_sha256,
        expected_round="repair",
        batch_path=repair_batch_path,
        expected_batch_file_sha256=expected_repair_batch_file_sha256,
    )
    _detail_freeze, detail_artifact = _load_optional_round(
        freeze_path=detail_freeze_path,
        expected_freeze_file_sha256=expected_detail_freeze_file_sha256,
        expected_round="detail",
        batch_path=detail_batch_path,
        expected_batch_file_sha256=expected_detail_batch_file_sha256,
    )
    batches = [verdict_artifact]
    if repair_artifact is not None:
        batches.append(repair_artifact)
    if detail_artifact is not None:
        batches.append(detail_artifact)
    generate, evidence_by_request = control_run._result_callback_from_batches(batches)
    codebook = material["codebook"]
    context = material["context"]
    control_by_id = control_run._control_records_by_id(material["control_input"])
    adjudicated_records: dict[str, dict[str, Any]] = {}
    for control_id in executor.MODEL_CONTROL_IDS:
        input_record = control_by_id[control_id]
        prepared = _control_preparation(
            control_id=control_id,
            input_record=input_record,
            primary_a=material["primary_a"],
            primary_b=material["primary_b"],
            codebook=codebook,
        )
        catalog = input_record.get("catalog")
        record = runner.adjudicate_packet(
            packet=prepared["packet"],
            codebook=codebook,
            annotation_pass=prepared["annotation_pass"],
            candidate_records=prepared["candidate_records"],
            candidate_manifest_locks_by_role=prepared["candidate_locks"],
            expected_candidate_manifest_lock_sha256_by_role=prepared[
                "candidate_lock_hashes"
            ],
            generate_native=generate,
            generation_context=context,
            verdict_seed=int(
                authenticated_adapter.value["roles"]["adjudicator"]["seed"]
            ),
            repair_seed=int(
                authenticated_adapter.value["roles"]["adjudicator"]["seed"]
            ),
            candidate_unit_bundle=(
                None
                if prepared["annotation_pass"] == "prefix_novelty"
                else catalog["candidate_unit_bundle"]
            ),
            expected_candidate_unit_bundle_sha256=(
                None
                if prepared["annotation_pass"] == "prefix_novelty"
                else catalog["candidate_unit_bundle_sha256"]
            ),
        )
        adjudicated_records[control_id] = record
    fixture_input_by_id = {
        row["case_id"]: row for row in material["fixture_input"]["records"]
    }
    fixture_key_by_id = {
        row["case_id"]: row for row in material["fixture_key"]["manifest"]["cases"]
    }
    fixture_records: dict[str, dict[str, Any]] = {}
    for case_id in executor.FIXTURE_IDS:
        input_record = fixture_input_by_id[case_id]
        key_row = fixture_key_by_id[case_id]
        annotation_pass = str(input_record["annotation_pass"])
        catalog = input_record.get("candidate_catalog")
        fixture_records[case_id] = fixture.run_fixture_adjudication(
            packet=input_record["packet"],
            codebook=codebook,
            annotation_pass=annotation_pass,
            fixture_candidates=tuple(input_record["authored_candidates"]),
            fixture_nonce_sha256=key_row["fixture_nonce_sha256"],
            fixture_config=material["fixture_config"],
            expected_fixture_config_sha256=control_run.sha256_file(
                fixture.DEFAULT_CONFIG_PATH
            ),
            generation_contract=material["contracts"][case_id],
            expected_generation_contract_sha256=key_row[
                "generation_contract_sha256"
            ],
            generation_context=context,
            expected_verdict_seed=key_row["verdict_seed"],
            expected_repair_seed=key_row["repair_seed"],
            fixture_lock=key_row["fixture_lock"],
            expected_fixture_lock_sha256=key_row["fixture_lock_sha256"],
            generate_native=generate,
            candidate_unit_bundle=(
                None
                if annotation_pass == "prefix_novelty"
                else catalog["candidate_unit_bundle"]
            ),
            expected_candidate_unit_bundle_sha256=(
                None
                if annotation_pass == "prefix_novelty"
                else input_record["candidate_unit_bundle_sha256"]
            ),
        )
    rows: list[dict[str, Any]] = []
    for ordinal, control_id in enumerate(executor.CONTROL_IDS, start=1):
        input_record = control_by_id[control_id]
        if control_id == "C32":
            result = {"decision": "no_chain"}
            native_evidence = None
            execution_path = "deterministic_host_bypass"
        else:
            record = adjudicated_records[control_id]
            result = _public_result(record)
            native_evidence = _native_evidence_from_stages(
                record=record,
                adapter_config_sha256=authenticated_adapter.config_sha256,
                adapter_source_sha256=authenticated_adapter.source_sha256,
                evidence_by_request=evidence_by_request,
            )
            execution_path = "native_model"
        rows.append(
            {
                "control_id": control_id,
                "ordinal": ordinal,
                "execution_path": execution_path,
                "packet_sha256": input_record["packet_sha256"],
                "result": result,
                "result_sha256": control_run.sha256_value(result),
                "native_evidence": native_evidence,
            }
        )
    fixture_rows: list[dict[str, Any]] = []
    for ordinal, case_id in enumerate(executor.FIXTURE_IDS, start=1):
        key_row = fixture_key_by_id[case_id]
        record = fixture_records[case_id]
        result = _public_result(record)
        fixture_rows.append(
            {
                "case_id": case_id,
                "ordinal": ordinal,
                "annotation_pass": str(record["annotation_pass"]),
                "result": result,
                "result_sha256": control_run.sha256_value(result),
                "fixture_lock_sha256": key_row["fixture_lock_sha256"],
                "generation_contract_sha256": key_row[
                    "generation_contract_sha256"
                ],
                "fixture_nonce_sha256": key_row["fixture_nonce_sha256"],
                "verdict_seed": key_row["verdict_seed"],
                "repair_seed": key_row["repair_seed"],
                "native_evidence": _native_evidence_from_stages(
                    record=record,
                    adapter_config_sha256=authenticated_adapter.config_sha256,
                    adapter_source_sha256=authenticated_adapter.source_sha256,
                    evidence_by_request=evidence_by_request,
                ),
            }
        )
    completed = time.monotonic_ns()
    body = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": ADJUDICATOR_RECEIPT_KIND,
        "status": "adjudicator_controls_and_fixtures_complete",
        "suite_id": suite["suite_id"],
        "adapter_config_sha256": authenticated_adapter.config_sha256,
        "adapter_source_sha256": authenticated_adapter.source_sha256,
        "preparation_file_sha256": expected_preparation_file_sha256,
        "model_execution_count": 39,
        "host_bypass_count": 1,
        "result_count": 40,
        "ordered_results": rows,
        "fixture_result_count": 12,
        "fixture_results": fixture_rows,
        "completed_monotonic_ns": completed,
        "claims": {
            "actual_model_execution_authenticated": True,
            "fixture_controls_executed": True,
            "control_scoring_key_read": False,
            "reserved_validation_accessed": False,
            "private_or_verbatim_cot_recovery_established": False,
            "affect_emotion_confidence_doubt_or_stress_established": False,
        },
    }
    body_sha = control_run.sha256_value(body)
    role_manifest = {
        "role": "adjudicator",
        "model_execution_count": 39,
        "host_bypass_count": 1,
        "result_count": 40,
        "ordered_results": rows,
        "role_receipt_sha256": body_sha,
    }
    envelope = {
        "receipt": body,
        "receipt_sha256": body_sha,
        "role_manifest": role_manifest,
        "fixture_results": fixture_rows,
    }
    output_path = Path(suite["generation_root"]) / "adjudicator-receipt.json"
    output_file_sha = control_run._exclusive_write_json(
        output_path, envelope, "adjudicator receipt"
    )
    journal_sha = control_run._append_trace(
        journal_path=Path(suite["trace_journal_path"]),
        journal_sha256=expected_trace_journal_file_sha256,
        stage="run_adjudicator",
        event_type="write",
        artifact_class="adjudicator_output",
        artifact_id="adjudicator_receipt",
        path=str(output_path.resolve(strict=True)),
        observed_sha256=output_file_sha,
    )
    journal_sha = control_run._append_trace(
        journal_path=Path(suite["trace_journal_path"]),
        journal_sha256=journal_sha,
        stage="run_adjudicator",
        event_type="transition",
        artifact_class="stage",
        artifact_id="adjudicator_complete",
    )
    return {
        "adjudicator_receipt_path": str(output_path),
        "adjudicator_receipt_file_sha256": output_file_sha,
        "adjudicator_receipt_sha256": body_sha,
        "completed_monotonic_ns": completed,
        "trace_journal_file_sha256": journal_sha,
        "repair_request_count": 0 if repair_artifact is None else repair_artifact["request_count"],
        "detail_request_count": 0 if detail_artifact is None else detail_artifact["request_count"],
    }


def _validate_adjudicator_receipt(value: Any, *, suite_id: str) -> dict[str, Any]:
    envelope = _mapping(value, "adjudicator receipt")
    _require(
        set(envelope) == {"receipt", "receipt_sha256", "role_manifest", "fixture_results"},
        "adjudicator receipt fields invalid",
    )
    body = _mapping(envelope["receipt"], "adjudicator receipt body")
    _require(
        body.get("kind") == ADJUDICATOR_RECEIPT_KIND
        and body.get("status") == "adjudicator_controls_and_fixtures_complete"
        and body.get("suite_id") == suite_id
        and body.get("result_count") == 40
        and body.get("fixture_result_count") == 12
        and envelope["receipt_sha256"] == control_run.sha256_value(body)
        and envelope["fixture_results"] == body["fixture_results"]
        and envelope["role_manifest"]["ordered_results"] == body["ordered_results"]
        and envelope["role_manifest"]["role_receipt_sha256"]
        == envelope["receipt_sha256"],
        "adjudicator receipt identity invalid",
    )
    return envelope


def lock_all(
    *,
    suite_init_path: Path,
    expected_suite_init_file_sha256: str,
    dual_primary_lock_path: Path,
    expected_dual_primary_lock_file_sha256: str,
    adjudicator_receipt_path: Path,
    expected_adjudicator_receipt_file_sha256: str,
    expected_executor_config_sha256: str,
    expected_executor_source_sha256: str,
    expected_adapter_config_sha256: str,
    expected_adapter_source_sha256: str,
    expected_trace_journal_file_sha256: str,
) -> dict[str, Any]:
    """Final-lock A/B/J, close the trace, and emit the public generation bundle."""

    suite = control_run._validate_suite_init(
        control_run._read_exact_json(
            suite_init_path, expected_suite_init_file_sha256, "suite init receipt"
        )
    )
    authenticated_executor = control_run._authenticated_executor(
        expected_config_sha256=expected_executor_config_sha256,
        expected_source_sha256=expected_executor_source_sha256,
    )
    authenticated_adapter = control_run._authenticated_adapter(
        expected_config_sha256=expected_adapter_config_sha256,
        expected_source_sha256=expected_adapter_source_sha256,
    )
    dual_lock = _validate_dual_lock(
        control_run._read_exact_json(
            dual_primary_lock_path,
            expected_dual_primary_lock_file_sha256,
            "dual primary lock",
        ),
        suite_id=suite["suite_id"],
    )
    primary_a, primary_b = _load_primary_material(dual_lock)
    adjudicator = _validate_adjudicator_receipt(
        control_run._read_exact_json(
            adjudicator_receipt_path,
            expected_adjudicator_receipt_file_sha256,
            "adjudicator receipt",
        ),
        suite_id=suite["suite_id"],
    )
    completed = time.monotonic_ns()
    final_body = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": executor.FINAL_LOCK_KIND,
        "status": "all_generation_outputs_locked_scoring_keys_unread",
        "suite_id": suite["suite_id"],
        "primary_independent_a_sha256": primary_a["receipt_sha256"],
        "primary_independent_b_sha256": primary_b["receipt_sha256"],
        "dual_primary_lock_sha256": dual_lock["lock_sha256"],
        "adjudicator_sha256": adjudicator["receipt_sha256"],
        "completed_monotonic_ns": completed,
        "claims": {
            "all_generation_outputs_locked": True,
            "control_scoring_key_read": False,
            "reserved_validation_accessed": False,
        },
    }
    final_sha = control_run.sha256_value(final_body)
    final_envelope = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": FINAL_LOCK_ENVELOPE_KIND,
        "lock": final_body,
        "lock_sha256": final_sha,
    }
    final_path = Path(suite["generation_root"]) / "final-generation-lock.json"
    final_file_sha = control_run._exclusive_write_json(
        final_path, final_envelope, "final generation lock"
    )
    journal_sha = control_run._append_trace(
        journal_path=Path(suite["trace_journal_path"]),
        journal_sha256=expected_trace_journal_file_sha256,
        stage="lock_all",
        event_type="write",
        artifact_class="final_lock",
        artifact_id="final_generation_lock",
        path=str(final_path.resolve(strict=True)),
        observed_sha256=final_file_sha,
    )
    journal_sha = control_run._append_trace(
        journal_path=Path(suite["trace_journal_path"]),
        journal_sha256=journal_sha,
        stage="lock_all",
        event_type="transition",
        artifact_class="stage",
        artifact_id="final_lock_complete",
    )
    journal_sha = control_run._append_trace(
        journal_path=Path(suite["trace_journal_path"]),
        journal_sha256=journal_sha,
        stage="lock_all",
        event_type="transition",
        artifact_class="stage",
        artifact_id="trace_closed",
    )
    trace_path = Path(suite["generation_root"]) / "read-trace-envelope.json"
    trace_result = executor.close_read_trace_journal(
        journal_path=Path(suite["trace_journal_path"]),
        expected_journal_file_sha256=journal_sha,
        suite_id=suite["suite_id"],
        output_envelope_path=trace_path,
    )
    trace_envelope = trace_result["envelope"]
    schedule = copy.deepcopy(authenticated_executor.value["fixture_seed_schedule"])
    chronology = {
        "freeze_helper": {
            "stage": "freeze_helper",
            "receipt_sha256": suite["freeze_receipt_sha256"],
            "completed_monotonic_ns": suite["freeze_completed_monotonic_ns"],
        },
        "precommit_nonce": {
            "stage": "precommit_nonce",
            "receipt_sha256": suite["nonce_precommit_receipt_sha256"],
            "completed_monotonic_ns": suite["nonce_completed_monotonic_ns"],
        },
        "primary_roles": {
            "independent_a": {
                "stage": "run_primary",
                "receipt_sha256": primary_a["receipt_sha256"],
                "completed_monotonic_ns": primary_a["receipt"]["completed_monotonic_ns"],
            },
            "independent_b": {
                "stage": "run_primary",
                "receipt_sha256": primary_b["receipt_sha256"],
                "completed_monotonic_ns": primary_b["receipt"]["completed_monotonic_ns"],
            },
        },
        "lock_primaries": {
            "stage": "lock_primaries",
            "receipt_sha256": dual_lock["lock_sha256"],
            "completed_monotonic_ns": dual_lock["lock"]["completed_monotonic_ns"],
        },
        "adjudicator": {
            "stage": "run_adjudicator",
            "receipt_sha256": adjudicator["receipt_sha256"],
            "completed_monotonic_ns": adjudicator["receipt"]["completed_monotonic_ns"],
        },
        "lock_all": {
            "stage": "lock_all",
            "receipt_sha256": final_sha,
            "completed_monotonic_ns": completed,
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": executor.PUBLIC_BUNDLE_KIND,
        "status": "all_generation_outputs_locked_scoring_keys_unread",
        "scope": copy.deepcopy(executor.EXPECTED_SCOPE),
        "suite_id": suite["suite_id"],
        "executor_identity": {
            "config_path": str(executor.CONFIG_PATH.relative_to(ROOT)),
            "config_sha256": authenticated_executor.config_sha256,
            "source_path": str(executor.SOURCE_PATH.relative_to(ROOT)),
            "source_sha256": authenticated_executor.source_sha256,
            "adapter_config_sha256": authenticated_adapter.config_sha256,
            "adapter_source_sha256": authenticated_adapter.source_sha256,
        },
        "suite_nonce": {
            "suite_nonce_sha256": suite["suite_nonce_sha256"],
            "precommit_receipt_sha256": suite["nonce_precommit_receipt_sha256"],
            "single_use_consumption_receipt_sha256": adjudicator["receipt"][
                "preparation_file_sha256"
            ],
            "retry_permitted": False,
            "raw_nonce_public": False,
        },
        "filesystem_roots": {
            "generation_root": suite["generation_root"],
            "key_root": suite["key_root"],
        },
        "inputs": {
            "control": {
                "path": suite["control_input_path"],
                "sha256": suite["control_input_sha256"],
            },
            "fixture": {
                "path": suite["fixture_input_path"],
                "sha256": suite["fixture_input_sha256"],
            },
        },
        "key_commitments": {
            "fixture_generation_key": {
                "sha256": authenticated_executor.value["key_contracts"][
                    "fixture_generation_key"
                ]["config_sha256"],
                "read_by_executor": True,
                "first_read_stage": "run_adjudicator",
                "read_count": 1,
            },
            "control_scoring_key": {
                "sha256": authenticated_executor.value["key_contracts"][
                    "control_scoring_key"
                ]["config_sha256"],
                "read_by_executor": False,
                "scorer_reads_last": True,
            },
        },
        "chronology": chronology,
        "roles": {
            "independent_a": primary_a["role_manifest"],
            "independent_b": primary_b["role_manifest"],
            "adjudicator": adjudicator["role_manifest"],
        },
        "fixture_results": adjudicator["fixture_results"],
        "fixture_seed_schedule": schedule,
        "fixture_seed_schedule_sha256": control_run.sha256_value(schedule),
        "locks": {
            "primary_independent_a_sha256": primary_a["receipt_sha256"],
            "primary_independent_b_sha256": primary_b["receipt_sha256"],
            "dual_primary_lock_sha256": dual_lock["lock_sha256"],
            "adjudicator_sha256": adjudicator["receipt_sha256"],
            "final_lock_sha256": final_sha,
        },
        "read_trace": {
            "trace_envelope_sha256": trace_result["envelope_sha256"],
            "trace_sha256": trace_envelope["trace_sha256"],
            "head_sha256": trace_envelope["trace"]["head_sha256"],
            "event_count": trace_envelope["trace"]["event_count"],
        },
        "claims": copy.deepcopy(executor.PUBLIC_CLAIMS),
    }
    # Bind the exact nonce-consumption marker file hash recorded by the trace.
    consume_events = [
        event
        for event in trace_envelope["trace"]["events"]
        if event["event_type"] == "consume"
        and event["artifact_class"] == "nonce_secret"
    ]
    _require(len(consume_events) == 1, "nonce consumption trace event missing")
    manifest["suite_nonce"]["single_use_consumption_receipt_sha256"] = consume_events[0][
        "observed_sha256"
    ]
    bundle = {"manifest": manifest, "manifest_sha256": control_run.sha256_value(manifest)}
    bundle_external_sha = control_run.sha256_value(bundle)
    authenticated_trace = executor.validate_read_trace_envelope(
        trace_envelope, independently_supplied_sha256=trace_result["envelope_sha256"]
    )
    executor.validate_public_generation_bundle(
        bundle,
        independently_supplied_sha256=bundle_external_sha,
        authenticated_trace=authenticated_trace,
    )
    bundle_path = Path(suite["generation_root"]) / "public-generation-bundle.json"
    bundle_file_sha = control_run._exclusive_write_json(
        bundle_path, bundle, "public generation bundle"
    )
    return {
        "final_lock_path": str(final_path),
        "final_lock_file_sha256": final_file_sha,
        "final_lock_sha256": final_sha,
        "read_trace_path": str(trace_path),
        "read_trace_file_sha256": trace_result["envelope_file_sha256"],
        "read_trace_envelope_sha256": trace_result["envelope_sha256"],
        "public_bundle_path": str(bundle_path),
        "public_bundle_file_sha256": bundle_file_sha,
        "public_bundle_envelope_sha256": bundle_external_sha,
        "event_count": trace_envelope["trace"]["event_count"],
        "claims": {
            "sealed_development_generation_bundle_complete": True,
            "scoring_performed": False,
            "control_scoring_key_read": False,
            "reserved_validation_accessed": False,
            "private_or_verbatim_cot_recovery_established": False,
            "affect_emotion_confidence_doubt_or_stress_established": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock = subparsers.add_parser("lock-primaries")
    lock.add_argument("--suite-init", type=Path, required=True)
    lock.add_argument("--suite-init-sha256", required=True)
    lock.add_argument("--primary-a", type=Path, required=True)
    lock.add_argument("--primary-a-sha256", required=True)
    lock.add_argument("--primary-b", type=Path, required=True)
    lock.add_argument("--primary-b-sha256", required=True)
    lock.add_argument("--trace-journal-sha256", required=True)
    lock.add_argument("--adjudicator-source-sha256", required=True)

    freeze = subparsers.add_parser("freeze-verdict")
    freeze.add_argument("--suite-init", type=Path, required=True)
    freeze.add_argument("--suite-init-sha256", required=True)
    freeze.add_argument("--dual-primary-lock", type=Path, required=True)
    freeze.add_argument("--dual-primary-lock-sha256", required=True)
    freeze.add_argument("--executor-config-sha256", required=True)
    freeze.add_argument("--executor-source-sha256", required=True)
    freeze.add_argument("--adapter-config-sha256", required=True)
    freeze.add_argument("--adapter-source-sha256", required=True)
    freeze.add_argument("--adjudicator-source-sha256", required=True)
    freeze.add_argument("--trace-journal-sha256", required=True)
    freeze.add_argument("--output-directory", type=Path, required=True)
    freeze.add_argument("--fixture-key-output", type=Path, required=True)

    run = subparsers.add_parser("run-round")
    run.add_argument("--freeze-manifest", type=Path, required=True)
    run.add_argument("--freeze-manifest-sha256", required=True)
    run.add_argument("--suite-init", type=Path, required=True)
    run.add_argument("--suite-init-sha256", required=True)
    run.add_argument("--executor-config-sha256", required=True)
    run.add_argument("--executor-source-sha256", required=True)
    run.add_argument("--adapter-config-sha256", required=True)
    run.add_argument("--adapter-source-sha256", required=True)

    followup = subparsers.add_parser("freeze-followup")
    followup.add_argument("--round", choices=("repair", "detail"), required=True)
    followup.add_argument("--suite-init", type=Path, required=True)
    followup.add_argument("--suite-init-sha256", required=True)
    followup.add_argument("--preparation", type=Path, required=True)
    followup.add_argument("--preparation-sha256", required=True)
    followup.add_argument("--verdict-batch", type=Path, required=True)
    followup.add_argument("--verdict-batch-sha256", required=True)
    followup.add_argument("--repair-batch", type=Path)
    followup.add_argument("--repair-batch-sha256")
    followup.add_argument("--executor-config-sha256", required=True)
    followup.add_argument("--executor-source-sha256", required=True)
    followup.add_argument("--adapter-config-sha256", required=True)
    followup.add_argument("--adapter-source-sha256", required=True)
    followup.add_argument("--adjudicator-source-sha256", required=True)
    followup.add_argument("--output-directory", type=Path, required=True)

    finalize = subparsers.add_parser("finalize-adjudicator")
    finalize.add_argument("--suite-init", type=Path, required=True)
    finalize.add_argument("--suite-init-sha256", required=True)
    finalize.add_argument("--preparation", type=Path, required=True)
    finalize.add_argument("--preparation-sha256", required=True)
    finalize.add_argument("--verdict-batch", type=Path, required=True)
    finalize.add_argument("--verdict-batch-sha256", required=True)
    finalize.add_argument("--repair-freeze", type=Path, required=True)
    finalize.add_argument("--repair-freeze-sha256", required=True)
    finalize.add_argument("--repair-batch", type=Path)
    finalize.add_argument("--repair-batch-sha256")
    finalize.add_argument("--detail-freeze", type=Path, required=True)
    finalize.add_argument("--detail-freeze-sha256", required=True)
    finalize.add_argument("--detail-batch", type=Path)
    finalize.add_argument("--detail-batch-sha256")
    finalize.add_argument("--adapter-config-sha256", required=True)
    finalize.add_argument("--adapter-source-sha256", required=True)
    finalize.add_argument("--trace-journal-sha256", required=True)

    final_lock = subparsers.add_parser("lock-all")
    final_lock.add_argument("--suite-init", type=Path, required=True)
    final_lock.add_argument("--suite-init-sha256", required=True)
    final_lock.add_argument("--dual-primary-lock", type=Path, required=True)
    final_lock.add_argument("--dual-primary-lock-sha256", required=True)
    final_lock.add_argument("--adjudicator-receipt", type=Path, required=True)
    final_lock.add_argument("--adjudicator-receipt-sha256", required=True)
    final_lock.add_argument("--executor-config-sha256", required=True)
    final_lock.add_argument("--executor-source-sha256", required=True)
    final_lock.add_argument("--adapter-config-sha256", required=True)
    final_lock.add_argument("--adapter-source-sha256", required=True)
    final_lock.add_argument("--trace-journal-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "lock-primaries":
        result = lock_primaries(
            suite_init_path=args.suite_init,
            expected_suite_init_file_sha256=args.suite_init_sha256,
            primary_a_path=args.primary_a,
            expected_primary_a_file_sha256=args.primary_a_sha256,
            primary_b_path=args.primary_b,
            expected_primary_b_file_sha256=args.primary_b_sha256,
            expected_trace_journal_file_sha256=args.trace_journal_sha256,
            expected_adjudicator_source_sha256=args.adjudicator_source_sha256,
        )
    elif args.command == "freeze-verdict":
        result = prepare_and_freeze_verdict(
            suite_init_path=args.suite_init,
            expected_suite_init_file_sha256=args.suite_init_sha256,
            dual_primary_lock_path=args.dual_primary_lock,
            expected_dual_primary_lock_file_sha256=args.dual_primary_lock_sha256,
            expected_executor_config_sha256=args.executor_config_sha256,
            expected_executor_source_sha256=args.executor_source_sha256,
            expected_adapter_config_sha256=args.adapter_config_sha256,
            expected_adapter_source_sha256=args.adapter_source_sha256,
            expected_adjudicator_source_sha256=args.adjudicator_source_sha256,
            expected_trace_journal_file_sha256=args.trace_journal_sha256,
            output_directory=args.output_directory,
            fixture_key_output_path=args.fixture_key_output,
        )
    elif args.command == "run-round":
        result = run_adjudicator_round(
            freeze_manifest_path=args.freeze_manifest,
            expected_freeze_manifest_file_sha256=args.freeze_manifest_sha256,
            suite_init_path=args.suite_init,
            expected_suite_init_file_sha256=args.suite_init_sha256,
            expected_executor_config_sha256=args.executor_config_sha256,
            expected_executor_source_sha256=args.executor_source_sha256,
            expected_adapter_config_sha256=args.adapter_config_sha256,
            expected_adapter_source_sha256=args.adapter_source_sha256,
        )
    elif args.command == "freeze-followup":
        result = freeze_followup_round(
            round_name=args.round,
            suite_init_path=args.suite_init,
            expected_suite_init_file_sha256=args.suite_init_sha256,
            preparation_path=args.preparation,
            expected_preparation_file_sha256=args.preparation_sha256,
            verdict_batch_path=args.verdict_batch,
            expected_verdict_batch_file_sha256=args.verdict_batch_sha256,
            repair_batch_path=args.repair_batch,
            expected_repair_batch_file_sha256=args.repair_batch_sha256,
            expected_executor_config_sha256=args.executor_config_sha256,
            expected_executor_source_sha256=args.executor_source_sha256,
            expected_adapter_config_sha256=args.adapter_config_sha256,
            expected_adapter_source_sha256=args.adapter_source_sha256,
            expected_adjudicator_source_sha256=args.adjudicator_source_sha256,
            output_directory=args.output_directory,
        )
    elif args.command == "finalize-adjudicator":
        result = finalize_adjudicator(
            suite_init_path=args.suite_init,
            expected_suite_init_file_sha256=args.suite_init_sha256,
            preparation_path=args.preparation,
            expected_preparation_file_sha256=args.preparation_sha256,
            verdict_batch_path=args.verdict_batch,
            expected_verdict_batch_file_sha256=args.verdict_batch_sha256,
            repair_freeze_path=args.repair_freeze,
            expected_repair_freeze_file_sha256=args.repair_freeze_sha256,
            repair_batch_path=args.repair_batch,
            expected_repair_batch_file_sha256=args.repair_batch_sha256,
            detail_freeze_path=args.detail_freeze,
            expected_detail_freeze_file_sha256=args.detail_freeze_sha256,
            detail_batch_path=args.detail_batch,
            expected_detail_batch_file_sha256=args.detail_batch_sha256,
            expected_adapter_config_sha256=args.adapter_config_sha256,
            expected_adapter_source_sha256=args.adapter_source_sha256,
            expected_trace_journal_file_sha256=args.trace_journal_sha256,
        )
    else:
        result = lock_all(
            suite_init_path=args.suite_init,
            expected_suite_init_file_sha256=args.suite_init_sha256,
            dual_primary_lock_path=args.dual_primary_lock,
            expected_dual_primary_lock_file_sha256=args.dual_primary_lock_sha256,
            adjudicator_receipt_path=args.adjudicator_receipt,
            expected_adjudicator_receipt_file_sha256=(
                args.adjudicator_receipt_sha256
            ),
            expected_executor_config_sha256=args.executor_config_sha256,
            expected_executor_source_sha256=args.executor_source_sha256,
            expected_adapter_config_sha256=args.adapter_config_sha256,
            expected_adapter_source_sha256=args.adapter_source_sha256,
            expected_trace_journal_file_sha256=args.trace_journal_sha256,
        )
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
