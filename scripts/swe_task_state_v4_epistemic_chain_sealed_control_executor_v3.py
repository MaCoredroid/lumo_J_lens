#!/usr/bin/env python3
"""Fail-closed outer protocol for prospective V3 development controls.

This module defines the generation-side chronology, locking, and public-bundle
contract.  It never imports or reads the control scoring key.  The one fixture
generation key may be opened only after both primary lanes have been locked.
The separately implemented scorer must authenticate the closed public bundle,
trace, and locks before it opens either scoring input.

The checked-in executor and native-adapter configurations are deliberately
unauthorized.  Consequently the checked-in files alone cannot load a model,
access a GPU, write gate evidence, or claim that a sealed run occurred.  A
future run requires separately frozen exact config/source/launch hashes and a
true-state external launch authorization.  Native launches
are delegated exclusively to
``native_vllm_adapter_v3.execute_production_native_batch``; this module accepts
no injected backend, tokenizer, engine, generation callback, or model output.

Only development controls are in scope.  Reserved validation remains closed.
The target is visible semantic COT-like E->H->A structure, not private or
verbatim chain-of-thought and not affect/emotion/confidence/doubt/stress.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

CONFIG_PATH = (
    ROOT
    / "configs"
    / "swe_task_state_v4_epistemic_chain_sealed_control_executor_v3.json"
)
SOURCE_PATH = Path(__file__).resolve()
SCHEMA_VERSION = 1
INTERFACE_VERSION = 3
CONFIG_KIND = "swe_task_state_v4_epistemic_chain_sealed_control_executor_v3"
READ_TRACE_KIND = "swe_task_state_v4_epistemic_chain_read_trace_v3"
PUBLIC_BUNDLE_KIND = (
    "swe_task_state_v4_epistemic_chain_public_generation_bundle_v3"
)
FREEZE_RECEIPT_KIND = (
    "swe_task_state_v4_epistemic_chain_outer_freeze_receipt_v3"
)
NONCE_RECEIPT_KIND = (
    "swe_task_state_v4_epistemic_chain_suite_nonce_precommit_receipt_v3"
)
ROLE_RECEIPT_KIND = (
    "swe_task_state_v4_epistemic_chain_outer_role_receipt_v3"
)
DUAL_PRIMARY_LOCK_KIND = (
    "swe_task_state_v4_epistemic_chain_dual_primary_lock_v3"
)
FINAL_LOCK_KIND = "swe_task_state_v4_epistemic_chain_final_generation_lock_v3"
OUTER_LAUNCH_KIND = (
    "swe_task_state_v4_epistemic_chain_outer_launch_authorization_v3"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ZERO_SHA256 = "0" * 64

PRIMARY_ROLES = ("independent_a", "independent_b")
ALL_ROLES = (*PRIMARY_ROLES, "adjudicator")
CONTROL_IDS = tuple(
    [f"C{index:02d}" for index in range(1, 33)]
    + [f"V{index:02d}" for index in range(1, 9)]
)
MODEL_CONTROL_IDS = tuple(item for item in CONTROL_IDS if item != "C32")
FIXTURE_IDS = tuple(f"F{index:02d}" for index in range(1, 13))
FIXTURE_PASSES = {
    "F01": "completion_chain",
    "F02": "completion_chain",
    "F03": "completion_chain",
    "F04": "completion_chain",
    "F05": "completion_chain",
    "F06": "completion_chain",
    "F07": "prefix_novelty",
    "F08": "prefix_novelty",
    "F09": "completion_chain",
    "F10": "completion_chain",
    "F11": "completion_chain",
    "F12": "prefix_novelty",
}
STAGE_RANKS = {
    "freeze_helper": 0,
    "precommit_nonce": 1,
    "run_primary": 2,
    "lock_primaries": 3,
    "run_adjudicator": 4,
    "lock_all": 5,
}
TRACE_EVENT_TYPES = frozenset({"read", "write", "transition", "consume"})
TRACE_ARTIFACT_CLASSES = frozenset(
    {
        "protocol_config",
        "adapter_config",
        "adapter_source",
        "control_input",
        "fixture_input",
        "fixture_generation_key",
        "fixture_key_materializer",
        "nonce_secret",
        "nonce_receipt",
        "primary_output",
        "dual_primary_lock",
        "adjudicator_output",
        "final_lock",
        "public_bundle",
        "read_trace",
        "stage",
    }
)
FORBIDDEN_EXECUTOR_ARTIFACT_CLASSES = frozenset(
    {"control_scoring_key", "control_key", "scoring_key", "expectation_key"}
)

EXPECTED_SCOPE = {
    "development_controls_only": True,
    "reserved_validation_closed": True,
    "reserved_validation_accessed": False,
    "visible_semantic_cot_like_structure_targeted": True,
    "private_or_verbatim_chain_of_thought_recovery_claimed": False,
    "affect_emotion_confidence_doubt_or_stress_targeted": False,
    "model_or_gpu_execution_performed_by_this_configuration": False,
    "sealed_control_run_claimed": False,
}
EXPECTED_CHECKED_IN_AUTHORIZATION = {
    "execution_authorized": False,
    "model_access_authorized": False,
    "gpu_access_authorized": False,
    "artifact_output_authorized": False,
    "gate_evidence_authorized": False,
    "future_enablement_requires_external_exact_hashes": True,
    "future_enablement_requires_separately_frozen_outer_launch_authorization": True,
    "future_enablement_requires_separately_frozen_adapter_launch_bindings": True,
}
PUBLIC_CLAIMS = {
    "development_controls_only": True,
    "reserved_validation_accessed": False,
    "visible_semantic_cot_like_control_evidence_only": True,
    "private_or_verbatim_chain_of_thought_recovery_established": False,
    "affect_emotion_confidence_doubt_or_stress_established": False,
    "actual_model_execution_authenticated": True,
    "all_generation_outputs_locked": True,
    "scoring_performed": False,
    "control_scoring_key_read_by_generation_executor": False,
    "gate_passed": False,
}


class SealedControlExecutorError(RuntimeError):
    """Raised whenever a V3 outer-protocol invariant fails closed."""


class ExecutionUnauthorized(SealedControlExecutorError):
    """Raised before a checked-in configuration can touch a model or GPU."""


def _assert_static_contract() -> None:
    """Reject runtime rebinding/mutation of security-sensitive display constants."""

    _require(
        CONTROL_IDS
        == tuple(
            [f"C{index:02d}" for index in range(1, 33)]
            + [f"V{index:02d}" for index in range(1, 9)]
        )
        and MODEL_CONTROL_IDS == tuple(item for item in CONTROL_IDS if item != "C32")
        and FIXTURE_IDS == tuple(f"F{index:02d}" for index in range(1, 13))
        and FIXTURE_PASSES
        == {
            "F01": "completion_chain",
            "F02": "completion_chain",
            "F03": "completion_chain",
            "F04": "completion_chain",
            "F05": "completion_chain",
            "F06": "completion_chain",
            "F07": "prefix_novelty",
            "F08": "prefix_novelty",
            "F09": "completion_chain",
            "F10": "completion_chain",
            "F11": "completion_chain",
            "F12": "prefix_novelty",
        }
        and PRIMARY_ROLES == ("independent_a", "independent_b")
        and ALL_ROLES == ("independent_a", "independent_b", "adjudicator")
        and STAGE_RANKS
        == {
            "freeze_helper": 0,
            "precommit_nonce": 1,
            "run_primary": 2,
            "lock_primaries": 3,
            "run_adjudicator": 4,
            "lock_all": 5,
        }
        and EXPECTED_SCOPE
        == {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "visible_semantic_cot_like_structure_targeted": True,
            "private_or_verbatim_chain_of_thought_recovery_claimed": False,
            "affect_emotion_confidence_doubt_or_stress_targeted": False,
            "model_or_gpu_execution_performed_by_this_configuration": False,
            "sealed_control_run_claimed": False,
        }
        and EXPECTED_CHECKED_IN_AUTHORIZATION
        == {
            "execution_authorized": False,
            "model_access_authorized": False,
            "gpu_access_authorized": False,
            "artifact_output_authorized": False,
            "gate_evidence_authorized": False,
            "future_enablement_requires_external_exact_hashes": True,
            "future_enablement_requires_separately_frozen_outer_launch_authorization": True,
            "future_enablement_requires_separately_frozen_adapter_launch_bindings": True,
        }
        and PUBLIC_CLAIMS
        == {
            "development_controls_only": True,
            "reserved_validation_accessed": False,
            "visible_semantic_cot_like_control_evidence_only": True,
            "private_or_verbatim_chain_of_thought_recovery_established": False,
            "affect_emotion_confidence_doubt_or_stress_established": False,
            "actual_model_execution_authenticated": True,
            "all_generation_outputs_locked": True,
            "scoring_performed": False,
            "control_scoring_key_read_by_generation_executor": False,
            "gate_passed": False,
        }
        and NATIVE_EVIDENCE_FIELDS
        == {
            "adapter_role",
            "adapter_config_sha256",
            "adapter_source_sha256",
            "outer_launch_authorization_sha256s",
            "launch_binding_sha256s",
            "preflight_receipt_sha256s",
            "runtime_receipt_sha256s",
            "native_request_sha256s",
            "native_result_sha256s",
            "actual_model_execution",
            "model_loaded",
            "generation_performed",
            "gate_eligible",
        }
        and OUTER_LAUNCH_FIELDS
        == {
            "schema_version",
            "interface_version",
            "kind",
            "suite_id",
            "generation_root",
            "key_root",
            "stage",
            "role",
            "execution_authorized",
            "model_access_authorized",
            "gpu_access_authorized",
            "artifact_output_authorized",
            "gate_evidence_authorized",
            "executor_config_sha256",
            "executor_source_sha256",
            "adapter_config_sha256",
            "adapter_source_sha256",
            "adapter_launch_binding_sha256",
            "request_batch_sha256",
            "suite_nonce_sha256",
            "nonce_precommit_receipt_sha256",
            "prior_lock_sha256",
            "authorization_nonce_sha256",
            "single_use_authorization_id",
            "retry_permitted",
            "reserved_validation_accessed",
        },
        "security-sensitive module constants were mutated or rebound",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SealedControlExecutorError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be an array",
    )
    return list(value)


def _exact_int(value: Any, *, minimum: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), "integer invalid")
    if minimum is not None:
        _require(value >= minimum, "integer below minimum")
    return int(value)


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be lowercase SHA-256",
    )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SealedControlExecutorError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise SealedControlExecutorError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _strict_json_bytes(value: bytes, label: str) -> Any:
    try:
        return json.loads(
            value.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealedControlExecutorError(f"{label} is not strict JSON: {error}") from error


def _assert_no_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    existing: list[Path] = []
    cursor = absolute
    while True:
        if cursor.exists() or cursor.is_symlink():
            existing.append(cursor)
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    for component in existing:
        try:
            info = component.lstat()
        except OSError as error:
            raise SealedControlExecutorError(
                f"cannot inspect {label} path component: {error}"
            ) from error
        _require(
            not stat.S_ISLNK(info.st_mode),
            f"{label} path contains symlink component {component}",
        )


def _read_regular_file(path: Path, label: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as error:
        raise SealedControlExecutorError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(info.st_mode), f"{label} must not be a symlink")
    _require(stat.S_ISREG(info.st_mode), f"{label} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as error:
        raise SealedControlExecutorError(f"cannot read {label}: {error}") from error


def _read_exact_file(path: Path, expected_sha256: str, label: str) -> bytes:
    expected = _sha256(expected_sha256, f"expected {label} hash")
    value = _read_regular_file(path, label)
    _require(sha256_bytes(value) == expected, f"{label} differs from external hash")
    return value


def _resolve_repo_path(relative_path: str, label: str) -> Path:
    _require(
        isinstance(relative_path, str)
        and bool(relative_path)
        and not Path(relative_path).is_absolute(),
        f"{label} path invalid",
    )
    try:
        root = ROOT.resolve(strict=True)
        path = (ROOT / relative_path).resolve(strict=True)
        path.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise SealedControlExecutorError(f"{label} escapes repository") from error
    _require(path.is_file(), f"{label} is not a file")
    return path


def _write_new_file(path: Path, value: bytes, label: str) -> str:
    """Create one artifact exactly once, without overwrite or symlink following."""

    absolute = path.absolute()
    _require(absolute.parent.is_dir(), f"{label} parent directory is missing")
    _require(not absolute.parent.is_symlink(), f"{label} parent is a symlink")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise SealedControlExecutorError(f"{label} already exists") from error
    except OSError as error:
        raise SealedControlExecutorError(f"cannot create {label}: {error}") from error
    return sha256_bytes(value)


def _write_new_json(path: Path, value: Any, label: str) -> str:
    return _write_new_file(path, canonical_json_bytes(value) + b"\n", label)


def _validate_binding(value: Any, label: str) -> dict[str, str]:
    binding = _mapping(value, label)
    _require(
        set(binding) == {"path", "sha256"}
        and isinstance(binding["path"], str)
        and bool(binding["path"]),
        f"{label} binding invalid",
    )
    _sha256(binding["sha256"], f"{label} hash")
    return {"path": binding["path"], "sha256": binding["sha256"]}


def _validate_seed_schedule(value: Any) -> list[dict[str, int | str]]:
    rows = _sequence(value, "fixture seed schedule")
    _require(len(rows) == 12, "fixture seed schedule must contain 12 rows")
    result: list[dict[str, int | str]] = []
    for ordinal, (raw, case_id) in enumerate(zip(rows, FIXTURE_IDS), start=1):
        row = _mapping(raw, f"fixture seed row {ordinal}")
        _require(
            set(row) == {"case_id", "verdict_seed", "repair_seed"}
            and row["case_id"] == case_id
            and _exact_int(row["verdict_seed"], minimum=1) == 1000 + ordinal
            and _exact_int(row["repair_seed"], minimum=1) == 2000 + ordinal,
            f"fixture seed row {case_id} changed",
        )
        result.append(copy.deepcopy(row))
    return result


@dataclass(frozen=True)
class AuthenticatedExecutorConfig:
    value: Mapping[str, Any]
    config_sha256: str
    source_sha256: str


def validate_executor_config(value: Any) -> dict[str, Any]:
    _assert_static_contract()
    config = copy.deepcopy(_mapping(value, "executor config"))
    _require(
        set(config)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "status",
            "scope",
            "authorization",
            "adapter_interface",
            "input_contracts",
            "key_contracts",
            "stages",
            "role_contracts",
            "host_bypass_contract",
            "fixture_seed_schedule",
            "nonce_contract",
            "filesystem_contract",
            "public_bundle_contract",
        },
        "executor config fields invalid",
    )
    _require(
        config["schema_version"] == SCHEMA_VERSION
        and config["interface_version"] == INTERFACE_VERSION
        and config["kind"] == CONFIG_KIND
        and config["status"]
        == "prospective_development_control_protocol_execution_unauthorized_not_run"
        and config["scope"] == EXPECTED_SCOPE
        and config["authorization"] == EXPECTED_CHECKED_IN_AUTHORIZATION,
        "executor config identity, scope, status, or authorization invalid",
    )

    adapter = _mapping(config["adapter_interface"], "adapter interface")
    _require(
        set(adapter)
        == {
            "config_path",
            "source_path",
            "entrypoint",
            "config_sha256_supplied_externally_per_launch",
            "source_sha256_supplied_externally_per_launch",
            "launch_binding_path_and_sha256_supplied_externally_per_batch",
            "caller_supplied_backend_forbidden",
            "non_gate_test_records_forbidden_as_generation_evidence",
        }
        and adapter["config_path"]
        == "configs/swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3.json"
        and adapter["source_path"]
        == "scripts/swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3.py"
        and adapter["entrypoint"] == "execute_production_native_batch"
        and all(
            adapter[name] is True
            for name in (
                "config_sha256_supplied_externally_per_launch",
                "source_sha256_supplied_externally_per_launch",
                "launch_binding_path_and_sha256_supplied_externally_per_batch",
                "caller_supplied_backend_forbidden",
                "non_gate_test_records_forbidden_as_generation_evidence",
            )
        ),
        "adapter interface changed",
    )

    inputs = _mapping(config["input_contracts"], "input contracts")
    _require(set(inputs) == {"control", "fixture"}, "input contracts changed")
    expected_inputs = {
        "control": {
            "builder_path": "scripts/swe_task_state_v4_epistemic_chain_control_inputs_draft_v3.py",
            "builder_sha256": "770457bd760fd81f50966203d3284e7dd697d661c0cab2fe472bb6b99aa92553",
            "config_path": "configs/swe_task_state_v4_epistemic_chain_control_inputs_draft_v3.json",
            "config_sha256": "97811c475349370c395f340b9a18d798f974f702e797ebf4606d06c21ebff4e4",
            "expected_manifest_sha256": "cb2080a895cb219c8995e3944a0c86b5a0239e96fbde24086244a79de3049567",
            "authoritative_regeneration_required_at_every_generation_stage": True,
        },
        "fixture": {
            "builder_path": "scripts/swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_inputs_draft_v3.py",
            "builder_sha256": "2984da9e9af8c358370a7719d7dacd5957063bebe39e9322904d37a6f6bcb5c8",
            "config_path": "configs/swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_inputs_draft_v3.json",
            "config_sha256": "54edd4d68c08e0faea7f2d04d59f7cd32fdbed8c46865abe63e33165a0598ee5",
            "expected_manifest_sha256": "81338adf399e0835fc5030f79228f019dc3505893cc550a5957a1ac1346aef9e",
            "authoritative_regeneration_required_at_every_generation_stage": True,
        },
    }
    _require(inputs == expected_inputs, "input contracts differ from frozen values")
    for contract in inputs.values():
        for path_name, hash_name in (
            ("builder_path", "builder_sha256"),
            ("config_path", "config_sha256"),
        ):
            path = _resolve_repo_path(contract[path_name], path_name)
            _require(
                sha256_file(path) == contract[hash_name],
                f"{path_name} source changed",
            )

    keys = _mapping(config["key_contracts"], "key contracts")
    _require(
        set(keys) == {"fixture_generation_key", "control_scoring_key"},
        "key contracts changed",
    )
    expected_fixture_key = {
        "config_path": "configs/swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_key_draft_v3.json",
        "config_sha256": "d676af22f287e882400d3f356b49b921af3464b6b353477d6a04dedfb71e09cf",
        "materializer_path": "scripts/swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_key_draft_v3.py",
        "materializer_sha256": "9febe2a281acd2f20223db5dec5dd842365752df581ed522b2a628bc6e5f2172",
        "earliest_read_stage": "run_adjudicator",
        "dual_primary_lock_required_before_read": True,
        "suite_nonce_precommit_required_before_read": True,
        "single_read_and_single_materialization_required": True,
    }
    expected_scoring_key = {
        "config_path": "configs/swe_task_state_v4_epistemic_chain_control_key_draft_v3.json",
        "config_sha256": "806da55baf4f39f18f7835d061f8729e2a55b8968a80abcb719960c699ac8250",
        "materializer_path": "scripts/swe_task_state_v4_epistemic_chain_control_key_draft_v3.py",
        "materializer_sha256": "9ffe5e56ec2f20a78877d7018726a235e48a708a97dba0aaae023b10b868de27",
        "generation_executor_read_forbidden": True,
        "separate_scorer_reads_after_final_lock": True,
    }
    _require(
        keys["fixture_generation_key"] == expected_fixture_key
        and keys["control_scoring_key"] == expected_scoring_key,
        "key contracts differ from frozen values",
    )
    # Do not resolve or open either key config *or key-side materializer* here.
    # The fixture materializer is authenticated and traced only after the dual
    # primary lock.  The scoring materializer belongs exclusively to the later
    # separate scorer.  Frozen literal path/hash commitments above are the only
    # pre-lock key-side information this generation config may carry.

    stages = _sequence(config["stages"], "stages")
    _require(
        stages == [
            {"name": name, "rank": rank} for name, rank in STAGE_RANKS.items()
        ],
        "stage order changed",
    )
    roles = _mapping(config["role_contracts"], "role contracts")
    _require(set(roles) == set(ALL_ROLES), "role contracts changed")
    for role in ALL_ROLES:
        contract = _mapping(roles[role], f"role contract {role}")
        expected_fields = {
            "adapter_role",
            "model_lineage",
            "ordered_control_results",
            "native_model_results",
            "deterministic_host_bypasses",
        }
        if role == "adjudicator":
            expected_fields |= {"ordered_fixture_results", "native_fixture_results"}
        _require(
            set(contract) == expected_fields
            and contract["adapter_role"] == role
            and contract["ordered_control_results"] == 40
            and contract["native_model_results"] == 39
            and contract["deterministic_host_bypasses"] == 1,
            f"role contract {role} invalid",
        )
        if role == "adjudicator":
            _require(
                contract["ordered_fixture_results"] == 12
                and contract["native_fixture_results"] == 12,
                "adjudicator fixture counts changed",
            )

    bypass = _mapping(config["host_bypass_contract"], "host bypass contract")
    _require(
        bypass
        == {
            "sole_control_id": "C32",
            "exact_visible_text_sha256": hashlib.sha256(b"").hexdigest(),
            "execution_path": "deterministic_host_bypass",
            "result": {"decision": "no_chain"},
            "native_evidence": None,
        },
        "C32 host bypass contract changed",
    )
    _validate_seed_schedule(config["fixture_seed_schedule"])
    nonce = _mapping(config["nonce_contract"], "nonce contract")
    _require(
        nonce
        == {
            "one_random_32_byte_suite_nonce": True,
            "public_nonce_is_sha256_only": True,
            "case_nonce_derivation_domain": "v3-fixture-corpus-case-nonce-citrine",
            "precommit_before_any_fixture_key_read": True,
            "single_use_consumption_marker_required": True,
            "retry_or_replacement_forbidden": True,
            "all_case_nonces_derived_from_suite_nonce": True,
        },
        "nonce contract changed",
    )
    filesystem = _mapping(config["filesystem_contract"], "filesystem contract")
    _require(
        set(filesystem)
        == {
            "generation_root_and_key_root_must_be_disjoint",
            "symlinks_for_protocol_artifacts_forbidden",
            "outputs_created_exclusively",
            "read_trace_hash_chained_and_closed_by_final_lock",
            "external_hash_required_for_every_cross_command_input",
            "primary_locks_are_distinct",
            "dual_primary_lock_required",
            "final_all_outputs_lock_required",
        }
        and all(item is True for item in filesystem.values()),
        "filesystem contract changed",
    )
    public = _mapping(config["public_bundle_contract"], "public bundle contract")
    _require(
        public
        == {
            "envelope_fields": ["manifest", "manifest_sha256"],
            "separate_read_trace_envelope_fields": ["trace", "trace_sha256"],
            "scoring_keys_absent": True,
            "scorer_authenticates_public_chronology_before_reading_keys": True,
        },
        "public bundle contract changed",
    )
    return config


def authenticate_executor_config(
    *, expected_config_sha256: str, expected_source_sha256: str
) -> AuthenticatedExecutorConfig:
    """Authenticate exact checked-in source/config before any protocol action."""

    expected_config = _sha256(expected_config_sha256, "executor config hash")
    expected_source = _sha256(expected_source_sha256, "executor source hash")
    config_bytes = _read_exact_file(CONFIG_PATH, expected_config, "executor config")
    source_bytes = _read_exact_file(SOURCE_PATH, expected_source, "executor source")
    config = validate_executor_config(_strict_json_bytes(config_bytes, "executor config"))
    _require(
        canonical_json_bytes(config)
        == canonical_json_bytes(_strict_json_bytes(config_bytes, "executor config")),
        "executor config changed during authentication",
    )
    return AuthenticatedExecutorConfig(
        value=copy.deepcopy(config),
        config_sha256=sha256_bytes(config_bytes),
        source_sha256=sha256_bytes(source_bytes),
    )


def _validate_input_envelope(
    value: Any, *, expected_manifest_sha256: str, fixture: bool
) -> dict[str, Any]:
    envelope = copy.deepcopy(_mapping(value, "input manifest envelope"))
    _require(
        set(envelope) == {"manifest", "manifest_sha256"},
        "input manifest envelope fields invalid",
    )
    manifest = _mapping(envelope["manifest"], "input manifest")
    observed = sha256_value(manifest)
    _require(
        envelope["manifest_sha256"] == observed
        and observed == _sha256(expected_manifest_sha256, "input manifest hash"),
        "input manifest differs from frozen exact hash",
    )
    if fixture:
        records = _sequence(manifest.get("records"), "fixture input records")
        _require(
            manifest.get("counts") == {"completion": 9, "novelty": 3, "total": 12}
            and [item.get("case_id") for item in records] == list(FIXTURE_IDS)
            and [item.get("annotation_pass") for item in records]
            == [FIXTURE_PASSES[item] for item in FIXTURE_IDS],
            "fixture input identity/order/count changed",
        )
    else:
        completions = _sequence(
            manifest.get("completion_records"), "completion input records"
        )
        novelties = _sequence(manifest.get("novelty_records"), "novelty input records")
        _require(
            manifest.get("counts") == {"completion": 32, "novelty": 8, "total": 40}
            and [item.get("control_id") for item in completions]
            == list(CONTROL_IDS[:32])
            and [item.get("control_id") for item in novelties]
            == list(CONTROL_IDS[32:]),
            "control input identity/order/count changed",
        )
        c32 = completions[31]
        _require(
            c32.get("host_action") == "bypass_exact_empty_visible_text"
            and c32.get("model_input_projections") == {}
            and c32.get("packet", {})
            .get("materialized_assistant_text", {})
            .get("text")
            == ""
            and all(
                item.get("host_action") == "invoke_completion_decision"
                for item in completions[:31]
            ),
            "sole C32 empty host bypass changed",
        )
    return copy.deepcopy(manifest)


def regenerate_authoritative_inputs(
    authenticated_config: AuthenticatedExecutorConfig,
) -> dict[str, dict[str, Any]]:
    """Rebuild both answer-blind input manifests from frozen authoring modules."""

    _require(
        isinstance(authenticated_config, AuthenticatedExecutorConfig),
        "executor config is not authenticated",
    )
    config = authenticated_config.value
    modules = {
        "control": "swe_task_state_v4_epistemic_chain_control_inputs_draft_v3",
        "fixture": (
            "swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_inputs_draft_v3"
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, module_name in modules.items():
        contract = config["input_contracts"][name]
        _require(
            sha256_file(_resolve_repo_path(contract["builder_path"], name))
            == contract["builder_sha256"]
            and sha256_file(_resolve_repo_path(contract["config_path"], name))
            == contract["config_sha256"],
            f"{name} authoring source/config changed before regeneration",
        )
        module = importlib.import_module(module_name)
        envelope = module.build_input_manifest_draft()
        _validate_input_envelope(
            envelope,
            expected_manifest_sha256=contract["expected_manifest_sha256"],
            fixture=name == "fixture",
        )
        result[name] = copy.deepcopy(envelope)
    return result


def _event_body(value: Mapping[str, Any]) -> dict[str, Any]:
    return {name: copy.deepcopy(item) for name, item in value.items() if name != "event_sha256"}


def _validate_trace_event(
    value: Any,
    *,
    expected_ordinal: int,
    previous_hash: str,
    previous_monotonic_ns: int | None,
    previous_stage_rank: int | None,
) -> dict[str, Any]:
    event = copy.deepcopy(_mapping(value, f"trace event {expected_ordinal}"))
    _require(
        set(event)
        == {
            "ordinal",
            "stage",
            "stage_rank",
            "event_type",
            "artifact_class",
            "artifact_id",
            "path",
            "expected_sha256",
            "observed_sha256",
            "monotonic_ns",
            "previous_event_sha256",
            "event_sha256",
        },
        f"trace event {expected_ordinal} fields invalid",
    )
    stage = event["stage"]
    stage_rank = event["stage_rank"]
    _require(
        event["ordinal"] == expected_ordinal
        and stage in STAGE_RANKS
        and stage_rank == STAGE_RANKS[stage]
        and event["event_type"] in TRACE_EVENT_TYPES
        and event["artifact_class"] in TRACE_ARTIFACT_CLASSES
        and event["artifact_class"] not in FORBIDDEN_EXECUTOR_ARTIFACT_CLASSES
        and isinstance(event["artifact_id"], str)
        and bool(event["artifact_id"]),
        f"trace event {expected_ordinal} identity invalid",
    )
    monotonic_ns = _exact_int(event["monotonic_ns"], minimum=1)
    if previous_monotonic_ns is not None:
        _require(
            monotonic_ns > previous_monotonic_ns,
            "trace monotonic chronology did not strictly advance",
        )
    if previous_stage_rank is not None:
        _require(stage_rank >= previous_stage_rank, "trace stage chronology regressed")
    _require(
        event["previous_event_sha256"] == previous_hash,
        f"trace event {expected_ordinal} previous hash invalid",
    )
    event_type = event["event_type"]
    if event_type == "read":
        _require(
            isinstance(event["path"], str)
            and Path(event["path"]).is_absolute()
            and _sha256(event["expected_sha256"], "expected read hash")
            == _sha256(event["observed_sha256"], "observed read hash"),
            f"trace read event {expected_ordinal} invalid",
        )
    elif event_type in {"write", "consume"}:
        _require(
            isinstance(event["path"], str)
            and Path(event["path"]).is_absolute()
            and event["expected_sha256"] is None
            and SHA256_RE.fullmatch(str(event["observed_sha256"])) is not None,
            f"trace {event_type} event {expected_ordinal} invalid",
        )
    else:
        _require(
            event["path"] is None
            and event["expected_sha256"] is None
            and event["observed_sha256"] is None
            and event["artifact_class"] == "stage",
            f"trace transition event {expected_ordinal} invalid",
        )
    observed_event_hash = sha256_value(_event_body(event))
    _require(
        event["event_sha256"] == observed_event_hash,
        f"trace event {expected_ordinal} hash invalid",
    )
    return event


def validate_read_trace_envelope(
    value: Any, *, independently_supplied_sha256: str
) -> dict[str, Any]:
    """Authenticate a closed, append-only read trace from an external hash."""

    _assert_static_contract()
    envelope = copy.deepcopy(_mapping(value, "read trace envelope"))
    _require(
        set(envelope) == {"trace", "trace_sha256"},
        "read trace envelope fields invalid",
    )
    external = _sha256(independently_supplied_sha256, "external trace envelope hash")
    _require(
        sha256_value(envelope) == external,
        "read trace envelope differs from independently supplied hash",
    )
    trace = copy.deepcopy(_mapping(envelope["trace"], "read trace"))
    _require(
        set(trace)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "suite_id",
            "events",
            "event_count",
            "head_sha256",
            "closed",
        }
        and trace["schema_version"] == SCHEMA_VERSION
        and trace["interface_version"] == INTERFACE_VERSION
        and trace["kind"] == READ_TRACE_KIND
        and isinstance(trace["suite_id"], str)
        and len(trace["suite_id"]) >= 16
        and trace["closed"] is True,
        "read trace identity invalid",
    )
    _require(
        envelope["trace_sha256"] == sha256_value(trace),
        "read trace self hash invalid",
    )
    events = _sequence(trace["events"], "trace events")
    _require(
        len(events) == _exact_int(trace["event_count"], minimum=1),
        "read trace event count invalid",
    )
    previous_hash = ZERO_SHA256
    previous_time: int | None = None
    previous_rank: int | None = None
    validated: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(events):
        event = _validate_trace_event(
            raw,
            expected_ordinal=ordinal,
            previous_hash=previous_hash,
            previous_monotonic_ns=previous_time,
            previous_stage_rank=previous_rank,
        )
        validated.append(event)
        previous_hash = event["event_sha256"]
        previous_time = int(event["monotonic_ns"])
        previous_rank = int(event["stage_rank"])
    _require(trace["head_sha256"] == previous_hash, "read trace head hash invalid")
    _require(
        validated[-1]["event_type"] == "transition"
        and validated[-1]["stage"] == "lock_all"
        and validated[-1]["artifact_id"] == "trace_closed",
        "read trace is not closed by lock_all",
    )

    transitions = [
        event["artifact_id"]
        for event in validated
        if event["event_type"] == "transition"
    ]
    required_once = (
        "freeze_complete",
        "nonce_precommitted",
        "primary_independent_a_complete",
        "primary_independent_b_complete",
        "dual_primary_lock_complete",
        "adjudicator_complete",
        "final_lock_complete",
        "trace_closed",
    )
    _require(
        all(transitions.count(name) == 1 for name in required_once),
        "required stage transition missing, repeated, or replayed",
    )
    transition_positions = {name: transitions.index(name) for name in required_once}
    _require(
        transition_positions["freeze_complete"]
        < transition_positions["nonce_precommitted"]
        < min(
            transition_positions["primary_independent_a_complete"],
            transition_positions["primary_independent_b_complete"],
        )
        and max(
            transition_positions["primary_independent_a_complete"],
            transition_positions["primary_independent_b_complete"],
        )
        < transition_positions["dual_primary_lock_complete"]
        < transition_positions["adjudicator_complete"]
        < transition_positions["final_lock_complete"]
        < transition_positions["trace_closed"],
        "stage transition order invalid",
    )
    fixture_key_reads = [
        event
        for event in validated
        if event["artifact_class"] == "fixture_generation_key"
        and event["event_type"] == "read"
    ]
    fixture_materializer_reads = [
        event
        for event in validated
        if event["artifact_class"] == "fixture_key_materializer"
        and event["event_type"] == "read"
    ]
    _require(
        len(fixture_key_reads) == 1
        and len(fixture_materializer_reads) == 1
        and fixture_key_reads[0]["stage"] == "run_adjudicator"
        and fixture_materializer_reads[0]["stage"] == "run_adjudicator",
        "fixture key/config materializer must each be read exactly once by adjudicator",
    )
    dual_time = next(
        event["monotonic_ns"]
        for event in validated
        if event["artifact_id"] == "dual_primary_lock_complete"
    )
    nonce_time = next(
        event["monotonic_ns"]
        for event in validated
        if event["artifact_id"] == "nonce_precommitted"
    )
    _require(
        all(
            event["monotonic_ns"] > dual_time
            and event["monotonic_ns"] > nonce_time
            for event in fixture_key_reads + fixture_materializer_reads
        ),
        "fixture key was read before nonce precommit or dual primary lock",
    )
    consumptions = [
        event
        for event in validated
        if event["event_type"] == "consume"
        and event["artifact_class"] == "nonce_secret"
    ]
    _require(
        len(consumptions) == 1
        and consumptions[0]["stage"] == "run_adjudicator"
        and consumptions[0]["monotonic_ns"] > dual_time,
        "suite nonce was not consumed exactly once after dual lock",
    )
    return trace


def make_trace_event(
    *,
    ordinal: int,
    stage: str,
    event_type: str,
    artifact_class: str,
    artifact_id: str,
    path: str | None,
    expected_sha256: str | None,
    observed_sha256: str | None,
    monotonic_ns: int,
    previous_event_sha256: str,
) -> dict[str, Any]:
    """Construct one canonical event; useful to the future CLI and CPU tests."""

    _assert_static_contract()
    event = {
        "ordinal": ordinal,
        "stage": stage,
        "stage_rank": STAGE_RANKS.get(stage),
        "event_type": event_type,
        "artifact_class": artifact_class,
        "artifact_id": artifact_id,
        "path": path,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "monotonic_ns": monotonic_ns,
        "previous_event_sha256": previous_event_sha256,
    }
    event["event_sha256"] = sha256_value(event)
    _validate_trace_event(
        event,
        expected_ordinal=ordinal,
        previous_hash=previous_event_sha256,
        previous_monotonic_ns=None,
        previous_stage_rank=None,
    )
    return event


def build_read_trace_envelope(
    *, suite_id: str, events: Sequence[Mapping[str, Any]], closed: bool
) -> dict[str, Any]:
    """Build an envelope; only a lock_all-closed trace can validate for scoring."""

    copied = copy.deepcopy(list(events))
    head = copied[-1]["event_sha256"] if copied else ZERO_SHA256
    trace = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": READ_TRACE_KIND,
        "suite_id": suite_id,
        "events": copied,
        "event_count": len(copied),
        "head_sha256": head,
        "closed": closed,
    }
    return {"trace": trace, "trace_sha256": sha256_value(trace)}


def create_read_trace_journal(path: Path) -> str:
    """Create one empty append-only JSONL journal with exclusive creation."""

    _assert_static_contract()
    return _write_new_file(path, b"", "read trace journal")


def _parse_trace_journal(value: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_hash = ZERO_SHA256
    previous_time: int | None = None
    previous_rank: int | None = None
    for ordinal, line in enumerate(value.splitlines()):
        _require(bool(line), "read trace journal contains blank line")
        event = _validate_trace_event(
            _strict_json_bytes(line, f"trace journal line {ordinal}"),
            expected_ordinal=ordinal,
            previous_hash=previous_hash,
            previous_monotonic_ns=previous_time,
            previous_stage_rank=previous_rank,
        )
        events.append(event)
        previous_hash = event["event_sha256"]
        previous_time = int(event["monotonic_ns"])
        previous_rank = int(event["stage_rank"])
    return events


def append_read_trace_event(
    *,
    journal_path: Path,
    expected_journal_file_sha256: str,
    stage: str,
    event_type: str,
    artifact_class: str,
    artifact_id: str,
    path: str | None,
    expected_sha256: str | None,
    observed_sha256: str | None,
    monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Append exactly one event after authenticating the complete prior journal."""

    _assert_static_contract()
    prior = _read_exact_file(
        journal_path,
        expected_journal_file_sha256,
        "read trace journal",
    )
    events = _parse_trace_journal(prior)
    _require(
        not events or events[-1]["artifact_id"] != "trace_closed",
        "read trace is already closed",
    )
    now = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
    if events:
        _require(
            now > events[-1]["monotonic_ns"],
            "appended trace event does not advance monotonic time",
        )
    event = make_trace_event(
        ordinal=len(events),
        stage=stage,
        event_type=event_type,
        artifact_class=artifact_class,
        artifact_id=artifact_id,
        path=path,
        expected_sha256=expected_sha256,
        observed_sha256=observed_sha256,
        monotonic_ns=now,
        previous_event_sha256=(
            events[-1]["event_sha256"] if events else ZERO_SHA256
        ),
    )
    line = canonical_json_bytes(event) + b"\n"
    flags = os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(journal_path.absolute(), flags)
        with os.fdopen(descriptor, "ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise SealedControlExecutorError(
            f"cannot append read trace journal: {error}"
        ) from error
    return {
        "event": event,
        "journal_file_sha256": sha256_file(journal_path),
        "event_count": len(events) + 1,
        "head_sha256": event["event_sha256"],
    }


def read_exact_and_append_trace(
    *,
    journal_path: Path,
    expected_journal_file_sha256: str,
    artifact_path: Path,
    expected_artifact_sha256: str,
    stage: str,
    artifact_class: str,
    artifact_id: str,
    monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Read exact bytes and atomically bind that successful read to the next event."""

    artifact_hash = _sha256(expected_artifact_sha256, "expected artifact hash")
    artifact = _read_exact_file(artifact_path, artifact_hash, artifact_id)
    append = append_read_trace_event(
        journal_path=journal_path,
        expected_journal_file_sha256=expected_journal_file_sha256,
        stage=stage,
        event_type="read",
        artifact_class=artifact_class,
        artifact_id=artifact_id,
        path=str(artifact_path.resolve(strict=True)),
        expected_sha256=artifact_hash,
        observed_sha256=sha256_bytes(artifact),
        monotonic_ns=monotonic_ns,
    )
    return {"artifact_bytes": artifact, **append}


def close_read_trace_journal(
    *,
    journal_path: Path,
    expected_journal_file_sha256: str,
    suite_id: str,
    output_envelope_path: Path,
) -> dict[str, Any]:
    """Create the immutable public trace envelope after the final close event."""

    _assert_static_contract()
    journal = _read_exact_file(
        journal_path,
        expected_journal_file_sha256,
        "read trace journal",
    )
    events = _parse_trace_journal(journal)
    envelope = build_read_trace_envelope(
        suite_id=suite_id, events=events, closed=True
    )
    external_hash = sha256_value(envelope)
    validate_read_trace_envelope(
        envelope, independently_supplied_sha256=external_hash
    )
    file_hash = _write_new_json(
        output_envelope_path, envelope, "closed read trace envelope"
    )
    return {
        "envelope": envelope,
        "envelope_sha256": external_hash,
        "envelope_file_sha256": file_hash,
        "journal_file_sha256": sha256_bytes(journal),
    }


NATIVE_EVIDENCE_FIELDS = {
    "adapter_role",
    "adapter_config_sha256",
    "adapter_source_sha256",
    "outer_launch_authorization_sha256s",
    "launch_binding_sha256s",
    "preflight_receipt_sha256s",
    "runtime_receipt_sha256s",
    "native_request_sha256s",
    "native_result_sha256s",
    "actual_model_execution",
    "model_loaded",
    "generation_performed",
    "gate_eligible",
}


def _validate_public_proposal(
    value: Any, *, annotation_pass: str, label: str
) -> dict[str, Any]:
    proposal = copy.deepcopy(_mapping(value, label))
    decision = proposal.get("decision")
    if annotation_pass == "completion_chain":
        _require(
            decision in {"chain", "no_chain", "unknown"},
            f"{label} completion decision outside frozen schema",
        )
        if decision == "chain":
            _require(
                set(proposal)
                == {
                    "decision",
                    "evidence_unit_id",
                    "hypothesis_unit_id",
                    "action_unit_id",
                    "evidence_kind",
                    "belief_edge",
                    "hypothesis_domain",
                    "action_intent",
                },
                f"{label} completion chain fields invalid",
            )
            unit_ids = [
                proposal["evidence_unit_id"],
                proposal["hypothesis_unit_id"],
                proposal["action_unit_id"],
            ]
            _require(
                all(isinstance(item, str) and bool(item) for item in unit_ids)
                and len(set(unit_ids)) == 3
                and proposal["evidence_kind"]
                in {"code", "tool_or_test", "spec_contract", "environment"}
                and proposal["belief_edge"] in {"supports", "refutes", "narrows"}
                and proposal["hypothesis_domain"]
                in {
                    "source_logic",
                    "interface_contract",
                    "data_type_shape",
                    "environment_dependency",
                    "tooling_path",
                    "test_fixture",
                    "other",
                }
                and proposal["action_intent"] in {"inspect", "edit", "validate"},
                f"{label} completion chain IDs or ontology invalid",
            )
        elif decision == "unknown":
            _require(
                proposal
                == {
                    "decision": "unknown",
                    "unknown_reason": "completion_semantics_ambiguous",
                },
                f"{label} completion unknown fields invalid",
            )
        else:
            _require(
                proposal == {"decision": "no_chain"},
                f"{label} completion no-chain fields invalid",
            )
    else:
        _require(annotation_pass == "prefix_novelty", f"{label} pass invalid")
        _require(
            decision in {"novel", "prefix_exposed", "ambiguous", "unknown"},
            f"{label} novelty decision outside frozen schema",
        )
        if decision == "unknown":
            _require(
                proposal
                == {
                    "decision": "unknown",
                    "unknown_reason": "novelty_semantics_ambiguous",
                },
                f"{label} novelty unknown fields invalid",
            )
        else:
            _require(set(proposal) == {"decision"}, f"{label} novelty fields invalid")
    return proposal


def _validate_chain_unit_ids_against_catalog(
    *, proposal: Mapping[str, Any], catalog_value: Any, label: str
) -> None:
    if proposal.get("decision") != "chain":
        return
    catalog_body = _mapping(catalog_value, f"{label} candidate catalog")
    bundle = _mapping(
        catalog_body.get("candidate_unit_bundle"), f"{label} candidate unit bundle"
    )
    units = _sequence(bundle.get("units"), f"{label} candidate units")
    positions = {
        str(_mapping(unit, f"{label} unit").get("unit_id")): position
        for position, unit in enumerate(units)
    }
    ids = [
        str(proposal["evidence_unit_id"]),
        str(proposal["hypothesis_unit_id"]),
        str(proposal["action_unit_id"]),
    ]
    _require(
        all(item in positions for item in ids)
        and positions[ids[0]] < positions[ids[1]] < positions[ids[2]],
        f"{label} chain IDs are not an authenticated E<H<A catalog tuple",
    )


def _validate_native_evidence(value: Any, *, role: str, label: str) -> dict[str, Any]:
    evidence = copy.deepcopy(_mapping(value, label))
    _require(set(evidence) == NATIVE_EVIDENCE_FIELDS, f"{label} fields invalid")
    _require(
        evidence["adapter_role"] == role
        and evidence["actual_model_execution"] is True
        and evidence["model_loaded"] is True
        and evidence["generation_performed"] is True
        and evidence["gate_eligible"] is True,
        f"{label} execution claims invalid",
    )
    _sha256(evidence["adapter_config_sha256"], f"{label} adapter config")
    _sha256(evidence["adapter_source_sha256"], f"{label} adapter source")
    list_fields = (
        "outer_launch_authorization_sha256s",
        "launch_binding_sha256s",
        "preflight_receipt_sha256s",
        "runtime_receipt_sha256s",
        "native_request_sha256s",
        "native_result_sha256s",
    )
    lists: dict[str, list[Any]] = {}
    for name in list_fields:
        items = _sequence(evidence[name], f"{label} {name}")
        _require(bool(items), f"{label} {name} is empty")
        for item in items:
            _sha256(item, f"{label} {name}")
        _require(len(set(items)) == len(items), f"{label} {name} contains replay")
        lists[name] = items
    _require(
        len(lists["outer_launch_authorization_sha256s"])
        == len(lists["preflight_receipt_sha256s"])
        == len(lists["runtime_receipt_sha256s"])
        == len(lists["launch_binding_sha256s"]),
        f"{label} batch receipt counts differ",
    )
    _require(
        len(lists["native_request_sha256s"])
        == len(lists["native_result_sha256s"]),
        f"{label} native request/result counts differ",
    )
    return evidence


def _validate_control_row(
    value: Any, *, expected_id: str, ordinal: int, role: str
) -> dict[str, Any]:
    row = copy.deepcopy(_mapping(value, f"{role} control row {expected_id}"))
    _require(
        set(row)
        == {
            "control_id",
            "ordinal",
            "execution_path",
            "packet_sha256",
            "result",
            "result_sha256",
            "native_evidence",
        }
        and row["control_id"] == expected_id
        and row["ordinal"] == ordinal,
        f"{role} control row {expected_id} identity invalid",
    )
    _sha256(row["packet_sha256"], f"{role} {expected_id} packet")
    result = _validate_public_proposal(
        row["result"],
        annotation_pass=(
            "completion_chain" if expected_id.startswith("C") else "prefix_novelty"
        ),
        label=f"{role} {expected_id} result",
    )
    _require(
        row["result_sha256"] == sha256_value(result),
        f"{role} {expected_id} result hash invalid",
    )
    if expected_id == "C32":
        _require(
            row["execution_path"] == "deterministic_host_bypass"
            and result == {"decision": "no_chain"}
            and row["native_evidence"] is None,
            f"{role} C32 deterministic host bypass invalid",
        )
    else:
        _require(
            row["execution_path"] == "native_model",
            f"{role} {expected_id} must use native_model",
        )
        _validate_native_evidence(
            row["native_evidence"], role=role, label=f"{role} {expected_id} evidence"
        )
    return row


def _validate_role_manifest(value: Any, *, role: str) -> dict[str, Any]:
    manifest = copy.deepcopy(_mapping(value, f"role manifest {role}"))
    _require(
        set(manifest)
        == {
            "role",
            "model_execution_count",
            "host_bypass_count",
            "result_count",
            "ordered_results",
            "role_receipt_sha256",
        }
        and manifest["role"] == role
        and manifest["model_execution_count"] == 39
        and manifest["host_bypass_count"] == 1
        and manifest["result_count"] == 40,
        f"role manifest {role} identity or counts invalid",
    )
    _sha256(manifest["role_receipt_sha256"], f"{role} receipt")
    rows = _sequence(manifest["ordered_results"], f"{role} results")
    _require(len(rows) == 40, f"{role} result count changed")
    validated = [
        _validate_control_row(raw, expected_id=control_id, ordinal=ordinal, role=role)
        for ordinal, (raw, control_id) in enumerate(zip(rows, CONTROL_IDS), start=1)
    ]
    _require(
        sum(item["execution_path"] == "native_model" for item in validated) == 39
        and sum(
            item["execution_path"] == "deterministic_host_bypass"
            for item in validated
        )
        == 1,
        f"role {role} execution path counts invalid",
    )
    native_request_hashes = [
        digest
        for item in validated
        if item["native_evidence"] is not None
        for digest in item["native_evidence"]["native_request_sha256s"]
    ]
    native_result_hashes = [
        digest
        for item in validated
        if item["native_evidence"] is not None
        for digest in item["native_evidence"]["native_result_sha256s"]
    ]
    _require(
        len(native_request_hashes) == len(set(native_request_hashes))
        and len(native_result_hashes) == len(set(native_result_hashes)),
        f"role {role} contains replayed native request/result evidence",
    )
    return manifest


def derive_fixture_case_nonce_sha256(
    *, suite_nonce_sha256: str, case_id: str, fixture_input_manifest_sha256: str
) -> str:
    _assert_static_contract()
    _sha256(suite_nonce_sha256, "suite nonce hash")
    _require(case_id in FIXTURE_IDS, "fixture case ID invalid")
    _sha256(fixture_input_manifest_sha256, "fixture input manifest hash")
    return sha256_value(
        {
            "domain": "v3-fixture-corpus-case-nonce-citrine",
            "suite_nonce_sha256": suite_nonce_sha256,
            "case_id": case_id,
            "input_manifest_sha256": fixture_input_manifest_sha256,
        }
    )


def _validate_fixture_results(
    value: Any,
    *,
    suite_nonce_sha256: str,
    fixture_input_manifest_sha256: str,
    seed_schedule: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = _sequence(value, "fixture results")
    _require(len(rows) == 12, "fixture result count changed")
    schedule = {row["case_id"]: row for row in seed_schedule}
    validated: list[dict[str, Any]] = []
    native_result_hashes: list[str] = []
    for ordinal, (raw, case_id) in enumerate(zip(rows, FIXTURE_IDS), start=1):
        row = copy.deepcopy(_mapping(raw, f"fixture result {case_id}"))
        _require(
            set(row)
            == {
                "case_id",
                "ordinal",
                "annotation_pass",
                "result",
                "result_sha256",
                "fixture_lock_sha256",
                "generation_contract_sha256",
                "fixture_nonce_sha256",
                "verdict_seed",
                "repair_seed",
                "native_evidence",
            }
            and row["case_id"] == case_id
            and row["ordinal"] == ordinal
            and row["annotation_pass"] == FIXTURE_PASSES[case_id]
            and row["verdict_seed"] == schedule[case_id]["verdict_seed"]
            and row["repair_seed"] == schedule[case_id]["repair_seed"],
            f"fixture result {case_id} identity/pass/seed invalid",
        )
        result = _validate_public_proposal(
            row["result"],
            annotation_pass=FIXTURE_PASSES[case_id],
            label=f"fixture {case_id} result",
        )
        _require(
            row["result_sha256"] == sha256_value(result),
            f"fixture {case_id} result hash invalid",
        )
        for name in (
            "fixture_lock_sha256",
            "generation_contract_sha256",
            "fixture_nonce_sha256",
        ):
            _sha256(row[name], f"fixture {case_id} {name}")
        _require(
            row["fixture_nonce_sha256"]
            == derive_fixture_case_nonce_sha256(
                suite_nonce_sha256=suite_nonce_sha256,
                case_id=case_id,
                fixture_input_manifest_sha256=fixture_input_manifest_sha256,
            ),
            f"fixture {case_id} nonce derivation invalid",
        )
        evidence = _validate_native_evidence(
            row["native_evidence"],
            role="adjudicator",
            label=f"fixture {case_id} evidence",
        )
        native_result_hashes.extend(evidence["native_result_sha256s"])
        validated.append(row)
    _require(
        len(native_result_hashes) == len(set(native_result_hashes)),
        "fixture native result replay detected",
    )
    return validated


def _validate_chronology(value: Any) -> dict[str, Any]:
    chronology = copy.deepcopy(_mapping(value, "bundle chronology"))
    _require(
        set(chronology)
        == {
            "freeze_helper",
            "precommit_nonce",
            "primary_roles",
            "lock_primaries",
            "adjudicator",
            "lock_all",
        },
        "bundle chronology fields invalid",
    )

    def stage(raw: Any, *, expected_name: str) -> dict[str, Any]:
        item = _mapping(raw, f"chronology {expected_name}")
        _require(
            set(item) == {"stage", "receipt_sha256", "completed_monotonic_ns"}
            and item["stage"] == expected_name,
            f"chronology stage {expected_name} invalid",
        )
        _sha256(item["receipt_sha256"], f"{expected_name} receipt")
        _exact_int(item["completed_monotonic_ns"], minimum=1)
        return item

    freeze = stage(chronology["freeze_helper"], expected_name="freeze_helper")
    nonce = stage(chronology["precommit_nonce"], expected_name="precommit_nonce")
    primary = _mapping(chronology["primary_roles"], "primary chronology")
    _require(set(primary) == set(PRIMARY_ROLES), "primary chronology roles changed")
    primary_a = stage(primary["independent_a"], expected_name="run_primary")
    primary_b = stage(primary["independent_b"], expected_name="run_primary")
    dual = stage(chronology["lock_primaries"], expected_name="lock_primaries")
    adjudicator = stage(chronology["adjudicator"], expected_name="run_adjudicator")
    final = stage(chronology["lock_all"], expected_name="lock_all")
    _require(
        freeze["completed_monotonic_ns"]
        < nonce["completed_monotonic_ns"]
        < min(
            primary_a["completed_monotonic_ns"],
            primary_b["completed_monotonic_ns"],
        )
        and max(
            primary_a["completed_monotonic_ns"],
            primary_b["completed_monotonic_ns"],
        )
        < dual["completed_monotonic_ns"]
        < adjudicator["completed_monotonic_ns"]
        < final["completed_monotonic_ns"],
        "bundle monotonic chronology invalid",
    )
    return chronology


def validate_public_generation_bundle(
    value: Any,
    *,
    independently_supplied_sha256: str,
    authenticated_trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate all public generation evidence before any scorer key read.

    ``independently_supplied_sha256`` authenticates the whole envelope, not its
    self-reported manifest hash.  ``authenticated_trace`` must be the trace body
    returned by :func:`validate_read_trace_envelope`.
    """

    _assert_static_contract()
    envelope = copy.deepcopy(_mapping(value, "public generation bundle"))
    _require(
        set(envelope) == {"manifest", "manifest_sha256"},
        "public bundle envelope fields invalid",
    )
    external = _sha256(independently_supplied_sha256, "external bundle hash")
    _require(
        sha256_value(envelope) == external,
        "public bundle differs from independently supplied whole-envelope hash",
    )
    manifest = copy.deepcopy(_mapping(envelope["manifest"], "public bundle manifest"))
    _require(
        envelope["manifest_sha256"] == sha256_value(manifest),
        "public bundle self hash invalid",
    )
    _require(
        set(manifest)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "status",
            "scope",
            "suite_id",
            "executor_identity",
            "suite_nonce",
            "filesystem_roots",
            "inputs",
            "key_commitments",
            "chronology",
            "roles",
            "fixture_results",
            "fixture_seed_schedule",
            "fixture_seed_schedule_sha256",
            "locks",
            "read_trace",
            "claims",
        }
        and manifest["schema_version"] == SCHEMA_VERSION
        and manifest["interface_version"] == INTERFACE_VERSION
        and manifest["kind"] == PUBLIC_BUNDLE_KIND
        and manifest["status"]
        == "all_generation_outputs_locked_scoring_keys_unread"
        and manifest["scope"] == EXPECTED_SCOPE
        and manifest["claims"] == PUBLIC_CLAIMS
        and isinstance(manifest["suite_id"], str)
        and len(manifest["suite_id"]) >= 16,
        "public bundle identity, scope, status, or claims invalid",
    )

    trace = _mapping(authenticated_trace, "authenticated trace")
    _require(
        trace.get("kind") == READ_TRACE_KIND
        and trace.get("closed") is True
        and trace.get("suite_id") == manifest["suite_id"],
        "authenticated trace is not the bundle suite's closed trace",
    )
    trace_binding = _mapping(manifest["read_trace"], "bundle trace binding")
    reconstructed_trace_envelope = {
        "trace": copy.deepcopy(trace),
        "trace_sha256": sha256_value(trace),
    }
    _require(
        set(trace_binding)
        == {"trace_envelope_sha256", "trace_sha256", "head_sha256", "event_count"}
        and trace_binding["trace_envelope_sha256"]
        == sha256_value(reconstructed_trace_envelope)
        and trace_binding["trace_sha256"] == sha256_value(trace)
        and trace_binding["head_sha256"] == trace["head_sha256"]
        and trace_binding["event_count"] == trace["event_count"],
        "bundle read-trace binding invalid",
    )
    _sha256(trace_binding["trace_envelope_sha256"], "trace envelope binding")

    identity = _mapping(manifest["executor_identity"], "executor identity")
    _require(
        set(identity)
        == {
            "config_path",
            "config_sha256",
            "source_path",
            "source_sha256",
            "adapter_config_sha256",
            "adapter_source_sha256",
        }
        and identity["config_path"] == str(CONFIG_PATH.relative_to(ROOT))
        and identity["source_path"] == str(SOURCE_PATH.relative_to(ROOT))
        and identity["config_sha256"] == sha256_file(CONFIG_PATH)
        and identity["source_sha256"] == sha256_file(SOURCE_PATH),
        "executor identity paths or fields invalid",
    )
    for name in (
        "config_sha256",
        "source_sha256",
        "adapter_config_sha256",
        "adapter_source_sha256",
    ):
        _sha256(identity[name], f"executor identity {name}")

    suite_nonce = _mapping(manifest["suite_nonce"], "suite nonce")
    _require(
        set(suite_nonce)
        == {
            "suite_nonce_sha256",
            "precommit_receipt_sha256",
            "single_use_consumption_receipt_sha256",
            "retry_permitted",
            "raw_nonce_public",
        }
        and suite_nonce["retry_permitted"] is False
        and suite_nonce["raw_nonce_public"] is False,
        "suite nonce contract invalid",
    )
    for name in (
        "suite_nonce_sha256",
        "precommit_receipt_sha256",
        "single_use_consumption_receipt_sha256",
    ):
        _sha256(suite_nonce[name], f"suite nonce {name}")

    filesystem_roots = _mapping(manifest["filesystem_roots"], "filesystem roots")
    _require(
        set(filesystem_roots) == {"generation_root", "key_root"},
        "filesystem root fields invalid",
    )
    generation_root, key_root = assert_disjoint_roots(
        generation_root=Path(str(filesystem_roots["generation_root"])),
        key_root=Path(str(filesystem_roots["key_root"])),
    )

    inputs = _mapping(manifest["inputs"], "bundle inputs")
    _require(set(inputs) == {"control", "fixture"}, "bundle input names invalid")
    control_input = _validate_binding(inputs["control"], "control input")
    fixture_input = _validate_binding(inputs["fixture"], "fixture input")
    _require(
        control_input["sha256"]
        == "cb2080a895cb219c8995e3944a0c86b5a0239e96fbde24086244a79de3049567"
        and fixture_input["sha256"]
        == "81338adf399e0835fc5030f79228f019dc3505893cc550a5957a1ac1346aef9e",
        "bundle input hashes differ from frozen authoritative manifests",
    )
    authenticated_input_manifests: dict[str, dict[str, Any]] = {}
    for name, binding in (("control", control_input), ("fixture", fixture_input)):
        _assert_no_symlink_components(Path(binding["path"]), f"{name} input")
        try:
            input_path = Path(binding["path"]).resolve(strict=True)
            input_path.relative_to(generation_root)
        except (OSError, RuntimeError, ValueError) as error:
            raise SealedControlExecutorError(
                f"{name} input is not physically inside generation root"
            ) from error
        input_bytes = _read_exact_file(
            input_path, binding["sha256"], f"{name} input manifest"
        )
        input_body = _strict_json_bytes(input_bytes, f"{name} input manifest")
        authenticated_input_manifests[name] = _validate_input_envelope(
            {"manifest": input_body, "manifest_sha256": binding["sha256"]},
            expected_manifest_sha256=binding["sha256"],
            fixture=name == "fixture",
        )

    key_commitments = _mapping(manifest["key_commitments"], "key commitments")
    _require(
        set(key_commitments) == {"fixture_generation_key", "control_scoring_key"},
        "key commitment names invalid",
    )
    fixture_key = _mapping(
        key_commitments["fixture_generation_key"], "fixture key commitment"
    )
    scoring_key = _mapping(
        key_commitments["control_scoring_key"], "scoring key commitment"
    )
    _require(
        set(fixture_key)
        == {"sha256", "read_by_executor", "first_read_stage", "read_count"}
        and fixture_key["sha256"]
        == "d676af22f287e882400d3f356b49b921af3464b6b353477d6a04dedfb71e09cf"
        and fixture_key["read_by_executor"] is True
        and fixture_key["first_read_stage"] == "run_adjudicator"
        and fixture_key["read_count"] == 1
        and set(scoring_key)
        == {"sha256", "read_by_executor", "scorer_reads_last"}
        and scoring_key["sha256"]
        == "806da55baf4f39f18f7835d061f8729e2a55b8968a80abcb719960c699ac8250"
        and scoring_key["read_by_executor"] is False
        and scoring_key["scorer_reads_last"] is True,
        "key commitment/read contract invalid",
    )
    trace_events = _sequence(trace["events"], "authenticated trace events")
    fixture_key_events = [
        event
        for event in trace_events
        if event["artifact_class"] == "fixture_generation_key"
        and event["event_type"] == "read"
    ]
    _require(
        len(fixture_key_events) == 1
        and fixture_key_events[0]["expected_sha256"] == fixture_key["sha256"]
        and fixture_key_events[0]["observed_sha256"] == fixture_key["sha256"],
        "fixture key trace event differs from key commitment",
    )
    try:
        fixture_key_trace_path = Path(fixture_key_events[0]["path"])
        _assert_no_symlink_components(fixture_key_trace_path, "fixture key trace")
        fixture_key_trace_path.resolve(strict=True).relative_to(key_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise SealedControlExecutorError(
            "fixture key trace path is not physically inside key root"
        ) from error
    materializer_events = [
        event
        for event in trace_events
        if event["artifact_class"] == "fixture_key_materializer"
        and event["event_type"] == "read"
    ]
    _require(
        len(materializer_events) == 1
        and materializer_events[0]["expected_sha256"]
        == "9febe2a281acd2f20223db5dec5dd842365752df581ed522b2a628bc6e5f2172"
        and materializer_events[0]["observed_sha256"]
        == "9febe2a281acd2f20223db5dec5dd842365752df581ed522b2a628bc6e5f2172",
        "fixture key materializer trace binding invalid",
    )
    for name, binding in (("control", control_input), ("fixture", fixture_input)):
        observed_events = [
            event
            for event in trace_events
            if event["artifact_class"] == f"{name}_input"
            and event["event_type"] == "read"
        ]
        required_stages = (
            {"freeze_helper", "run_primary", "run_adjudicator"}
            if name == "control"
            else {"freeze_helper", "run_adjudicator"}
        )
        required_artifact_ids = (
            {
                "control_manifest_regenerated",
                "control_manifest_regenerated_for_a",
                "control_manifest_regenerated_for_b",
                "control_manifest_regenerated_for_j",
            }
            if name == "control"
            else {
                "fixture_manifest_regenerated",
                "fixture_manifest_regenerated_for_j",
            }
        )
        _require(
            {event["stage"] for event in observed_events} == required_stages
            and len(observed_events) == len(required_artifact_ids)
            and {event["artifact_id"] for event in observed_events}
            == required_artifact_ids
            and all(
                event["path"] == str(Path(binding["path"]).resolve(strict=True))
                and event["expected_sha256"] == binding["sha256"]
                and event["observed_sha256"] == binding["sha256"]
                for event in observed_events
            ),
            f"{name} input was not authoritatively re-read at every required stage",
        )

    chronology = _validate_chronology(manifest["chronology"])
    roles = _mapping(manifest["roles"], "bundle roles")
    _require(set(roles) == set(ALL_ROLES), "bundle role set changed")
    validated_roles = {
        role: _validate_role_manifest(roles[role], role=role) for role in ALL_ROLES
    }
    control_records = {
        str(record["control_id"]): record
        for record in (
            authenticated_input_manifests["control"]["completion_records"]
            + authenticated_input_manifests["control"]["novelty_records"]
        )
    }
    for position, control_id in enumerate(CONTROL_IDS):
        packet_hashes = {
            validated_roles[role]["ordered_results"][position]["packet_sha256"]
            for role in ALL_ROLES
        }
        _require(
            packet_hashes == {control_records[control_id]["packet_sha256"]},
            f"packet identity differs from authoritative input for {control_id}",
        )
        if control_id.startswith("C"):
            for role in ALL_ROLES:
                _validate_chain_unit_ids_against_catalog(
                    proposal=validated_roles[role]["ordered_results"][position]["result"],
                    catalog_value=control_records[control_id]["catalog"],
                    label=f"{role} {control_id}",
                )
    for role in ALL_ROLES:
        adapter_hashes = {
            row["native_evidence"]["adapter_config_sha256"]
            for row in validated_roles[role]["ordered_results"]
            if row["native_evidence"] is not None
        }
        source_hashes = {
            row["native_evidence"]["adapter_source_sha256"]
            for row in validated_roles[role]["ordered_results"]
            if row["native_evidence"] is not None
        }
        _require(
            adapter_hashes == {identity["adapter_config_sha256"]}
            and source_hashes == {identity["adapter_source_sha256"]},
            f"{role} adapter identity differs from bundle",
        )

    schedule = _validate_seed_schedule(manifest["fixture_seed_schedule"])
    _require(
        manifest["fixture_seed_schedule_sha256"] == sha256_value(schedule),
        "fixture seed schedule hash invalid",
    )
    validated_fixture_rows = _validate_fixture_results(
        manifest["fixture_results"],
        suite_nonce_sha256=suite_nonce["suite_nonce_sha256"],
        fixture_input_manifest_sha256=fixture_input["sha256"],
        seed_schedule=schedule,
    )
    fixture_input_records = {
        str(record["case_id"]): record
        for record in authenticated_input_manifests["fixture"]["records"]
    }
    for row in validated_fixture_rows:
        case_id = str(row["case_id"])
        if row["annotation_pass"] == "completion_chain":
            _validate_chain_unit_ids_against_catalog(
                proposal=row["result"],
                catalog_value=fixture_input_records[case_id]["candidate_catalog"],
                label=f"fixture {case_id}",
            )
    all_native_request_hashes = [
        digest
        for role in ALL_ROLES
        for row in validated_roles[role]["ordered_results"]
        if row["native_evidence"] is not None
        for digest in row["native_evidence"]["native_request_sha256s"]
    ] + [
        digest
        for row in validated_fixture_rows
        for digest in row["native_evidence"]["native_request_sha256s"]
    ]
    all_native_result_hashes = [
        digest
        for role in ALL_ROLES
        for row in validated_roles[role]["ordered_results"]
        if row["native_evidence"] is not None
        for digest in row["native_evidence"]["native_result_sha256s"]
    ] + [
        digest
        for row in validated_fixture_rows
        for digest in row["native_evidence"]["native_result_sha256s"]
    ]
    _require(
        len(all_native_request_hashes) == len(set(all_native_request_hashes))
        and len(all_native_result_hashes) == len(set(all_native_result_hashes)),
        "native request/result replay detected across finalized bundle",
    )

    locks = _mapping(manifest["locks"], "bundle locks")
    _require(
        set(locks)
        == {
            "primary_independent_a_sha256",
            "primary_independent_b_sha256",
            "dual_primary_lock_sha256",
            "adjudicator_sha256",
            "final_lock_sha256",
        },
        "bundle lock fields invalid",
    )
    for name, value_hash in locks.items():
        _sha256(value_hash, f"bundle lock {name}")
    _require(
        locks["primary_independent_a_sha256"]
        == validated_roles["independent_a"]["role_receipt_sha256"]
        and locks["primary_independent_b_sha256"]
        == validated_roles["independent_b"]["role_receipt_sha256"]
        and locks["adjudicator_sha256"]
        == validated_roles["adjudicator"]["role_receipt_sha256"]
        and locks["primary_independent_a_sha256"]
        != locks["primary_independent_b_sha256"]
        and locks["dual_primary_lock_sha256"]
        == chronology["lock_primaries"]["receipt_sha256"]
        and locks["final_lock_sha256"]
        == chronology["lock_all"]["receipt_sha256"]
        and suite_nonce["precommit_receipt_sha256"]
        == chronology["precommit_nonce"]["receipt_sha256"]
        and locks["primary_independent_a_sha256"]
        == chronology["primary_roles"]["independent_a"]["receipt_sha256"]
        and locks["primary_independent_b_sha256"]
        == chronology["primary_roles"]["independent_b"]["receipt_sha256"]
        and locks["adjudicator_sha256"]
        == chronology["adjudicator"]["receipt_sha256"],
        "bundle lock/chronology/role bindings invalid",
    )
    return manifest


def assert_disjoint_roots(*, generation_root: Path, key_root: Path) -> tuple[Path, Path]:
    """Resolve and prove physical separation of generation and key roots."""

    _assert_static_contract()
    _assert_no_symlink_components(generation_root, "generation root")
    _assert_no_symlink_components(key_root, "key root")
    try:
        generation = generation_root.resolve(strict=True)
        keys = key_root.resolve(strict=True)
    except OSError as error:
        raise SealedControlExecutorError(f"protocol root missing: {error}") from error
    _require(generation.is_dir() and keys.is_dir(), "protocol roots must be directories")
    _require(not generation.is_symlink() and not keys.is_symlink(), "protocol roots cannot be symlinks")
    _require(
        generation != keys
        and generation not in keys.parents
        and keys not in generation.parents,
        "generation and key roots must be disjoint and non-nested",
    )
    return generation, keys


def precommit_suite_nonce(
    *,
    suite_id: str,
    generation_root: Path,
    key_root: Path,
    nonce_secret_path: Path,
    single_use_marker_path: Path,
    receipt_path: Path,
    freeze_receipt_sha256: str,
    completed_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Generate one suite nonce and precommit it exactly once.

    This helper is intentionally independent of model authorization.  It writes
    only a private nonce, a single-use marker, and a public hash receipt.  It
    accepts no caller-supplied nonce and has no retry path.
    """

    _assert_static_contract()
    generation, _ = assert_disjoint_roots(
        generation_root=generation_root, key_root=key_root
    )
    _require(isinstance(suite_id, str) and len(suite_id) >= 16, "suite ID invalid")
    for path, label in (
        (nonce_secret_path, "nonce secret"),
        (single_use_marker_path, "single-use marker"),
        (receipt_path, "nonce receipt"),
    ):
        _require(path.absolute().parent == generation, f"{label} must be directly in generation root")
        _require(not path.absolute().exists(), f"{label} already exists; nonce retry forbidden")
    freeze_hash = _sha256(freeze_receipt_sha256, "freeze receipt")
    raw_nonce = secrets.token_bytes(32)
    nonce_sha256 = sha256_bytes(raw_nonce)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "kind": "swe_task_state_v4_epistemic_chain_suite_nonce_single_use_marker_v3",
        "suite_id": suite_id,
        "suite_nonce_sha256": nonce_sha256,
        "retry_permitted": False,
    }
    marker_file_sha256 = _write_new_json(
        single_use_marker_path, marker, "suite nonce single-use marker"
    )
    secret_file_sha256 = _write_new_file(nonce_secret_path, raw_nonce, "suite nonce secret")
    when = time.monotonic_ns() if completed_monotonic_ns is None else completed_monotonic_ns
    _exact_int(when, minimum=1)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": NONCE_RECEIPT_KIND,
        "status": "suite_nonce_precommitted_before_fixture_key_access",
        "suite_id": suite_id,
        "freeze_receipt_sha256": freeze_hash,
        "suite_nonce_sha256": nonce_sha256,
        "nonce_secret_file_sha256": secret_file_sha256,
        "single_use_marker_file_sha256": marker_file_sha256,
        "raw_nonce_public": False,
        "retry_permitted": False,
        "completed_monotonic_ns": when,
        "claims": {
            "model_or_gpu_execution_performed": False,
            "fixture_key_read": False,
            "scoring_key_read": False,
            "sealed_run_claimed": False,
        },
    }
    receipt_file_sha256 = _write_new_json(receipt_path, receipt, "nonce receipt")
    return {
        "receipt": receipt,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_canonical_sha256": sha256_value(receipt),
    }


def consume_suite_nonce_once(
    *,
    suite_id: str,
    generation_root: Path,
    key_root: Path,
    nonce_secret_path: Path,
    expected_nonce_secret_file_sha256: str,
    expected_suite_nonce_sha256: str,
    consumption_marker_path: Path,
    dual_primary_lock_sha256: str,
) -> dict[str, Any]:
    """Consume the precommitted nonce once, after an external dual-lock root."""

    _assert_static_contract()
    generation, _ = assert_disjoint_roots(
        generation_root=generation_root, key_root=key_root
    )
    _require(
        nonce_secret_path.absolute().parent == generation
        and consumption_marker_path.absolute().parent == generation,
        "nonce paths must be directly in generation root",
    )
    _require(
        not consumption_marker_path.absolute().exists(),
        "suite nonce consumption marker already exists; replay forbidden",
    )
    secret = _read_exact_file(
        nonce_secret_path,
        expected_nonce_secret_file_sha256,
        "suite nonce secret",
    )
    _require(len(secret) == 32, "suite nonce must be exactly 32 bytes")
    nonce_hash = sha256_bytes(secret)
    _require(
        nonce_hash == _sha256(expected_suite_nonce_sha256, "suite nonce hash"),
        "suite nonce differs from precommit",
    )
    marker = {
        "schema_version": SCHEMA_VERSION,
        "kind": "swe_task_state_v4_epistemic_chain_suite_nonce_consumption_v3",
        "suite_id": suite_id,
        "suite_nonce_sha256": nonce_hash,
        "dual_primary_lock_sha256": _sha256(
            dual_primary_lock_sha256, "dual primary lock"
        ),
        "consumed_once": True,
        "retry_permitted": False,
        "completed_monotonic_ns": time.monotonic_ns(),
    }
    marker_file_sha256 = _write_new_json(
        consumption_marker_path, marker, "suite nonce consumption marker"
    )
    return {
        "marker": marker,
        "marker_file_sha256": marker_file_sha256,
        "suite_nonce_sha256": nonce_hash,
    }


def delegate_native_adapter_batch(
    *,
    authenticated_executor_config: AuthenticatedExecutorConfig,
    outer_launch_authorization_path: Path,
    expected_outer_launch_authorization_sha256: str,
    outer_authorization_consumption_marker_path: Path,
    expected_adapter_config_sha256: str,
    expected_adapter_source_sha256: str,
    launch_binding_path: Path,
    expected_launch_binding_sha256: str,
    role: str,
    request_specs: Sequence[Any],
) -> Any:
    """Delegate one real batch only through the module-owned adapter entrypoint.

    The checked-in executor remains false-state and contributes no authority.
    A separately frozen outer launch file is hash-authenticated first and must
    carry every true outer execution flag.  A distinct adapter launch binding
    is authenticated independently; the production adapter then performs the
    full request-batch/runtime/model/GPU authentication.  Neither authority is
    inferred from a callback or a caller-supplied execution claim.
    """

    _assert_static_contract()
    _require(
        isinstance(authenticated_executor_config, AuthenticatedExecutorConfig),
        "executor config is not authenticated",
    )
    adapter_contract = authenticated_executor_config.value["adapter_interface"]
    adapter_config_path = _resolve_repo_path(
        adapter_contract["config_path"], "native adapter config"
    )
    adapter_source_path = _resolve_repo_path(
        adapter_contract["source_path"], "native adapter source"
    )
    _require(
        sha256_file(adapter_config_path)
        == _sha256(expected_adapter_config_sha256, "adapter config hash")
        and sha256_file(adapter_source_path)
        == _sha256(expected_adapter_source_sha256, "adapter source hash"),
        "native adapter config/source differs from external hashes",
    )
    _require(role in ALL_ROLES, "adapter role invalid")
    outer_launch = _precheck_outer_launch_authorization(
        path=outer_launch_authorization_path,
        expected_sha256=expected_outer_launch_authorization_sha256,
        authenticated_executor_config=authenticated_executor_config,
        adapter_config_sha256=expected_adapter_config_sha256,
        adapter_source_sha256=expected_adapter_source_sha256,
        adapter_launch_binding_sha256=expected_launch_binding_sha256,
        role=role,
    )
    _precheck_external_launch_binding(
        path=launch_binding_path,
        expected_sha256=expected_launch_binding_sha256,
        role=role,
    )
    module = importlib.import_module(
        "swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3"
    )
    adapter_config = module.load_adapter_config(
        path=adapter_config_path,
        expected_config_sha256=expected_adapter_config_sha256,
    )
    request_batch = module.request_batch_descriptor(
        role=role,
        request_specs=request_specs,
        config=adapter_config,
    )
    _require(
        request_batch["request_batch_sha256"]
        == outer_launch["request_batch_sha256"],
        "outer launch authorization differs from exact adapter request batch",
    )
    generation_root = Path(str(outer_launch["generation_root"]))
    expected_marker_path = generation_root / (
        "outer-launch-"
        + str(outer_launch["single_use_authorization_id"])
        + ".consumed.json"
    )
    _require(
        outer_authorization_consumption_marker_path.absolute()
        == expected_marker_path,
        "outer launch consumption marker path differs from single-use authorization ID",
    )
    consumption_marker = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": "swe_task_state_v4_epistemic_chain_outer_launch_consumption_v3",
        "suite_id": outer_launch["suite_id"],
        "role": role,
        "stage": outer_launch["stage"],
        "outer_launch_authorization_sha256": _sha256(
            expected_outer_launch_authorization_sha256,
            "outer launch authorization hash",
        ),
        "request_batch_sha256": request_batch["request_batch_sha256"],
        "single_use_authorization_id": outer_launch["single_use_authorization_id"],
        "consumed_once": True,
        "retry_permitted": False,
        "completed_monotonic_ns": time.monotonic_ns(),
    }
    _write_new_json(
        outer_authorization_consumption_marker_path,
        consumption_marker,
        "outer launch authorization consumption marker",
    )
    entrypoint = getattr(module, adapter_contract["entrypoint"], None)
    _require(callable(entrypoint), "native adapter production entrypoint missing")
    return entrypoint(
        expected_config_sha256=expected_adapter_config_sha256,
        launch_binding_path=launch_binding_path,
        expected_launch_binding_sha256=_sha256(
            expected_launch_binding_sha256, "launch binding hash"
        ),
        role=role,
        request_specs=request_specs,
    )


OUTER_LAUNCH_FIELDS = {
    "schema_version",
    "interface_version",
    "kind",
    "suite_id",
    "generation_root",
    "key_root",
    "stage",
    "role",
    "execution_authorized",
    "model_access_authorized",
    "gpu_access_authorized",
    "artifact_output_authorized",
    "gate_evidence_authorized",
    "executor_config_sha256",
    "executor_source_sha256",
    "adapter_config_sha256",
    "adapter_source_sha256",
    "adapter_launch_binding_sha256",
    "request_batch_sha256",
    "suite_nonce_sha256",
    "nonce_precommit_receipt_sha256",
    "prior_lock_sha256",
    "authorization_nonce_sha256",
    "single_use_authorization_id",
    "retry_permitted",
    "reserved_validation_accessed",
}


def _precheck_outer_launch_authorization(
    *,
    path: Path,
    expected_sha256: str,
    authenticated_executor_config: AuthenticatedExecutorConfig,
    adapter_config_sha256: str,
    adapter_source_sha256: str,
    adapter_launch_binding_sha256: str,
    role: str,
) -> dict[str, Any]:
    """Authenticate independent outer authority before request-spec access."""

    _require(
        isinstance(authenticated_executor_config, AuthenticatedExecutorConfig),
        "executor config is not authenticated",
    )
    value = _strict_json_bytes(
        _read_exact_file(
            path,
            _sha256(expected_sha256, "external outer launch authorization hash"),
            "external outer launch authorization",
        ),
        "external outer launch authorization",
    )
    launch = _mapping(value, "external outer launch authorization")
    stage = "run_adjudicator" if role == "adjudicator" else "run_primary"
    _require(
        set(launch) == OUTER_LAUNCH_FIELDS
        and launch["schema_version"] == SCHEMA_VERSION
        and launch["interface_version"] == INTERFACE_VERSION
        and launch["kind"] == OUTER_LAUNCH_KIND
        and isinstance(launch["suite_id"], str)
        and len(launch["suite_id"]) >= 16
        and launch["stage"] == stage
        and launch["role"] == role,
        "outer launch authorization identity/stage/role invalid",
    )
    generation_root, key_root = assert_disjoint_roots(
        generation_root=Path(str(launch["generation_root"])),
        key_root=Path(str(launch["key_root"])),
    )
    _require(
        launch["generation_root"] == str(generation_root)
        and launch["key_root"] == str(key_root),
        "outer launch filesystem roots are not canonical",
    )
    for name in (
        "execution_authorized",
        "model_access_authorized",
        "gpu_access_authorized",
        "artifact_output_authorized",
        "gate_evidence_authorized",
    ):
        _require(launch[name] is True, f"outer launch does not authorize {name}")
    expected_bindings = {
        "executor_config_sha256": authenticated_executor_config.config_sha256,
        "executor_source_sha256": authenticated_executor_config.source_sha256,
        "adapter_config_sha256": _sha256(
            adapter_config_sha256, "outer launch adapter config"
        ),
        "adapter_source_sha256": _sha256(
            adapter_source_sha256, "outer launch adapter source"
        ),
        "adapter_launch_binding_sha256": _sha256(
            adapter_launch_binding_sha256, "outer launch adapter binding"
        ),
    }
    _require(
        all(launch[name] == digest for name, digest in expected_bindings.items()),
        "outer launch source/config/adapter binding changed",
    )
    for name in (
        "request_batch_sha256",
        "suite_nonce_sha256",
        "nonce_precommit_receipt_sha256",
        "authorization_nonce_sha256",
        "single_use_authorization_id",
    ):
        _sha256(launch[name], f"outer launch {name}")
    _require(
        launch["single_use_authorization_id"]
        == sha256_value(
            {
                "domain": "v3-outer-launch-single-use-citrine",
                "suite_id": launch["suite_id"],
                "stage": launch["stage"],
                "role": launch["role"],
                "request_batch_sha256": launch["request_batch_sha256"],
                "authorization_nonce_sha256": launch[
                    "authorization_nonce_sha256"
                ],
            }
        ),
        "outer launch single-use authorization ID derivation invalid",
    )
    if role == "adjudicator":
        _sha256(launch["prior_lock_sha256"], "outer launch dual primary lock")
    else:
        _require(
            launch["prior_lock_sha256"] is None,
            "primary outer launch must precede any primary lock",
        )
    _require(
        launch["retry_permitted"] is False
        and launch["reserved_validation_accessed"] is False,
        "outer launch retry/reserved-validation contract invalid",
    )
    return copy.deepcopy(launch)


def _precheck_external_launch_binding(
    *, path: Path, expected_sha256: str, role: str
) -> dict[str, Any]:
    """Reject absent/malformed external authority before request-spec access."""

    value = _strict_json_bytes(
        _read_exact_file(
            path,
            _sha256(expected_sha256, "external launch binding hash"),
            "external launch binding",
        ),
        "external launch binding",
    )
    launch = _mapping(value, "external launch binding")
    required_flags = {
        "execution_authorized",
        "model_access_authorized",
        "gpu_access_authorized",
        "output_authorized",
        "production_receipt_authorized",
        "gate_eligible_execution_authorized",
    }
    _require(
        launch.get("schema_version") == SCHEMA_VERSION
        and launch.get("kind")
        == "swe_task_state_v4_epistemic_chain_native_vllm_launch_authorization_v3"
        and launch.get("role") == role
        and all(launch.get(name) is True for name in required_flags),
        "external launch binding identity/role/authorization invalid",
    )
    # The adapter owns the exact field set and all static/request/runtime
    # identity comparisons.  This is only a fail-first authorization shell.
    return copy.deepcopy(launch)


def freeze_helper(
    *, expected_config_sha256: str, expected_source_sha256: str
) -> dict[str, Any]:
    """Build a CPU-only in-memory freeze draft; never claim or write a run."""

    _assert_static_contract()
    authenticated = authenticate_executor_config(
        expected_config_sha256=expected_config_sha256,
        expected_source_sha256=expected_source_sha256,
    )
    manifests = regenerate_authoritative_inputs(authenticated)
    body = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": FREEZE_RECEIPT_KIND,
        "status": "in_memory_freeze_draft_execution_unauthorized_not_run",
        "executor_config_sha256": authenticated.config_sha256,
        "executor_source_sha256": authenticated.source_sha256,
        "control_input_manifest_sha256": manifests["control"]["manifest_sha256"],
        "fixture_input_manifest_sha256": manifests["fixture"]["manifest_sha256"],
        "authoritative_packet_regeneration_performed": True,
        "authorization": copy.deepcopy(authenticated.value["authorization"]),
        "claims": {
            "model_or_gpu_execution_performed": False,
            "persistent_artifact_written": False,
            "fixture_key_read": False,
            "control_scoring_key_read": False,
            "sealed_run_claimed": False,
        },
    }
    return {"receipt": body, "receipt_sha256": sha256_value(body)}


def _load_cli_json(path: Path, expected_sha256: str, label: str) -> Any:
    return _strict_json_bytes(_read_exact_file(path, expected_sha256, label), label)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze-helper", help="regenerate inputs and emit an in-memory unauthorized draft"
    )
    freeze.add_argument("--executor-config-sha256", required=True)
    freeze.add_argument("--executor-source-sha256", required=True)

    nonce = subparsers.add_parser(
        "precommit-nonce", help="create one private suite nonce and public receipt"
    )
    nonce.add_argument("--suite-id", required=True)
    nonce.add_argument("--generation-root", required=True, type=Path)
    nonce.add_argument("--key-root", required=True, type=Path)
    nonce.add_argument("--nonce-secret", required=True, type=Path)
    nonce.add_argument("--single-use-marker", required=True, type=Path)
    nonce.add_argument("--receipt", required=True, type=Path)
    nonce.add_argument("--freeze-receipt-sha256", required=True)

    run_primary = subparsers.add_parser(
        "run-primary", help="delegate an authenticated native primary batch"
    )
    run_primary.add_argument("--role", choices=PRIMARY_ROLES, required=True)
    run_primary.add_argument("--executor-config-sha256", required=True)
    run_primary.add_argument("--executor-source-sha256", required=True)
    run_primary.add_argument("--adapter-config-sha256", required=True)
    run_primary.add_argument("--adapter-source-sha256", required=True)
    run_primary.add_argument("--outer-launch-authorization", required=True, type=Path)
    run_primary.add_argument("--outer-launch-authorization-sha256", required=True)
    run_primary.add_argument(
        "--outer-authorization-consumption-marker", required=True, type=Path
    )
    run_primary.add_argument("--launch-binding", required=True, type=Path)
    run_primary.add_argument("--launch-binding-sha256", required=True)
    run_primary.add_argument("--request-specs", required=True, type=Path)
    run_primary.add_argument("--request-specs-sha256", required=True)

    subparsers.add_parser(
        "lock-primaries",
        help="reserved for authorized run: dual-lock two exact primary receipts",
    )
    subparsers.add_parser(
        "run-adjudicator",
        help="reserved for authorized run: consume nonce, open fixture key, launch J",
    )
    subparsers.add_parser(
        "lock-all",
        help="reserved for authorized run: final-lock A/B/J and close public trace",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze-helper":
        result = freeze_helper(
            expected_config_sha256=args.executor_config_sha256,
            expected_source_sha256=args.executor_source_sha256,
        )
    elif args.command == "precommit-nonce":
        result = precommit_suite_nonce(
            suite_id=args.suite_id,
            generation_root=args.generation_root,
            key_root=args.key_root,
            nonce_secret_path=args.nonce_secret,
            single_use_marker_path=args.single_use_marker,
            receipt_path=args.receipt,
            freeze_receipt_sha256=args.freeze_receipt_sha256,
        )
    elif args.command == "run-primary":
        authenticated = authenticate_executor_config(
            expected_config_sha256=args.executor_config_sha256,
            expected_source_sha256=args.executor_source_sha256,
        )
        # Authenticate both external true-state shells before reading specs.
        # The exact outer batch binding is rechecked after typed construction.
        _precheck_outer_launch_authorization(
            path=args.outer_launch_authorization,
            expected_sha256=args.outer_launch_authorization_sha256,
            authenticated_executor_config=authenticated,
            adapter_config_sha256=args.adapter_config_sha256,
            adapter_source_sha256=args.adapter_source_sha256,
            adapter_launch_binding_sha256=args.launch_binding_sha256,
            role=args.role,
        )
        _precheck_external_launch_binding(
            path=args.launch_binding,
            expected_sha256=args.launch_binding_sha256,
            role=args.role,
        )
        raw_specs = _load_cli_json(
            args.request_specs, args.request_specs_sha256, "request specs"
        )
        adapter = importlib.import_module(
            "swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3"
        )
        specs = [adapter.NativeRequestSpec(**item) for item in _sequence(raw_specs, "specs")]
        batch = delegate_native_adapter_batch(
            authenticated_executor_config=authenticated,
            outer_launch_authorization_path=args.outer_launch_authorization,
            expected_outer_launch_authorization_sha256=(
                args.outer_launch_authorization_sha256
            ),
            outer_authorization_consumption_marker_path=(
                args.outer_authorization_consumption_marker
            ),
            expected_adapter_config_sha256=args.adapter_config_sha256,
            expected_adapter_source_sha256=args.adapter_source_sha256,
            launch_binding_path=args.launch_binding,
            expected_launch_binding_sha256=args.launch_binding_sha256,
            role=args.role,
            request_specs=specs,
        )
        result = {
            "status": "native_batch_returned_in_memory_not_persisted",
            "request_count": len(batch.requests),
            "result_count": len(batch.results),
            "preflight_receipt_sha256": batch.preflight_receipt.receipt_sha256,
            "runtime_receipt_sha256": batch.runtime_receipt.receipt_sha256,
        }
    else:
        raise ExecutionUnauthorized(
            f"{args.command} unavailable while checked-in execution authorization is false"
        )
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


__all__ = [
    "ALL_ROLES",
    "CONFIG_PATH",
    "CONTROL_IDS",
    "ExecutionUnauthorized",
    "FIXTURE_IDS",
    "INTERFACE_VERSION",
    "MODEL_CONTROL_IDS",
    "OUTER_LAUNCH_FIELDS",
    "OUTER_LAUNCH_KIND",
    "PRIMARY_ROLES",
    "PUBLIC_BUNDLE_KIND",
    "READ_TRACE_KIND",
    "SCHEMA_VERSION",
    "SealedControlExecutorError",
    "assert_disjoint_roots",
    "append_read_trace_event",
    "authenticate_executor_config",
    "build_read_trace_envelope",
    "canonical_json_bytes",
    "close_read_trace_journal",
    "consume_suite_nonce_once",
    "create_read_trace_journal",
    "delegate_native_adapter_batch",
    "derive_fixture_case_nonce_sha256",
    "freeze_helper",
    "make_trace_event",
    "precommit_suite_nonce",
    "read_exact_and_append_trace",
    "regenerate_authoritative_inputs",
    "sha256_bytes",
    "sha256_file",
    "sha256_value",
    "validate_executor_config",
    "validate_public_generation_bundle",
    "validate_read_trace_envelope",
]


if __name__ == "__main__":
    raise SystemExit(main())
