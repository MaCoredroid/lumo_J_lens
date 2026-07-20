#!/usr/bin/env python3
"""Execute the sealed quote-first V2 semantic controls without answer access.

This module is an additive control-only bridge between the sealed 52-row model
input artifact and the quote-first V2 runner.  It deliberately has no argument
for an expectation artifact and never imports or opens one during generation.

Each invocation runs exactly one model role:

* ``independent_a`` and ``independent_b`` each receive only the 32 completion
  and 8 prefix-novelty controls;
* ``adjudicator`` receives only the 12 supplied blind candidate controls; and
* no target annotation packet kind is accepted anywhere in this module.

The role outputs retain the complete quote-runner record, including raw final
model output, generation details, and authenticated model identity.  A later
deterministic combine step emits two exact 52-row files (A+adjudicator and
B+adjudicator) in the schema consumed by the existing control lock/scorer.
Answers may be joined only after those files have been independently locked by
``swe_task_state_v4_epistemic_chain_controls_v2.py``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import swe_task_state_v4_epistemic_chain_annotation_runner_v2 as quote_runner  # noqa: E402


SCHEMA_VERSION = 1
EXECUTOR_KIND = "swe_task_state_v4_epistemic_chain_control_executor_v2"
FROZEN_STATUS = "prospectively_frozen_for_sealed_control_generation"
ROLE_RECORD_KIND = "swe_task_state_v4_epistemic_chain_control_role_record_v2"
ROLE_MANIFEST_KIND = "swe_task_state_v4_epistemic_chain_control_role_manifest_v2"
COMBINED_MANIFEST_KIND = (
    "swe_task_state_v4_epistemic_chain_control_combined_manifest_v2"
)
DUAL_LOCK_RECEIPT_KIND = (
    "swe_task_state_v4_epistemic_chain_control_dual_lock_receipt_v2"
)
FINAL_GATE_RECEIPT_KIND = (
    "swe_task_state_v4_epistemic_chain_control_final_gate_receipt_v2"
)
MODEL_INPUT_KIND = "swe_task_state_v4_epistemic_chain_control_model_input_v2"
MODEL_INPUT_MANIFEST_KIND = (
    "swe_task_state_v4_epistemic_chain_control_model_input_manifest_v2"
)
SCORER_OUTPUT_KIND = "swe_task_state_v4_epistemic_chain_control_output_v2"
CONTROL_BUILDER_PATH = (
    ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_controls_v2.py"
)
DECODER_ADDENDUM_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_decoder_v2_addendum.json"
)
RUNNER_PATH = Path(quote_runner.__file__).resolve()
EXECUTOR_PATH = Path(__file__).resolve()
PRIMARY_ROLES = ("independent_a", "independent_b")
ROLES = (*PRIMARY_ROLES, "adjudicator")
CONTROL_PASSES = ("completion", "novelty", "adjudication")
EXPECTED_COUNTS = {
    "completion": 32,
    "novelty": 8,
    "adjudication": 12,
    "total": 52,
}
ROLE_PASS_CONTRACT = {
    "independent_a": ("completion", "novelty"),
    "independent_b": ("completion", "novelty"),
    "adjudicator": ("adjudication",),
}
ROLE_RECORD_COUNTS = {
    "independent_a": 40,
    "independent_b": 40,
    "adjudicator": 12,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ControlExecutorError(RuntimeError):
    """Raised when an execution, isolation, or immutable-output check fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlExecutorError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be an array",
    )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ControlExecutorError(f"duplicate JSON key: {key}")
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


def canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _guard_existing_repo_file(path: Path, *, label: str) -> Path:
    logical = Path(path).absolute()
    lowered_parts = {part.casefold() for part in logical.parts}
    _require(
        "validation" not in lowered_parts and ".git" not in lowered_parts,
        f"{label} may not use a reserved path",
    )
    _require(not logical.is_symlink(), f"{label} may not be a logical symlink")
    try:
        resolved = logical.resolve(strict=True)
        resolved.relative_to(ROOT.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as error:
        raise ControlExecutorError(
            f"{label} must resolve to a regular file inside the repository"
        ) from error
    _require(resolved.is_file(), f"{label} is not a regular file")
    resolved_parts = {part.casefold() for part in resolved.parts}
    _require(
        "validation" not in resolved_parts and ".git" not in resolved_parts,
        f"{label} resolves through a reserved path",
    )
    return resolved


def _guard_generation_input(path: Path, *, label: str) -> Path:
    resolved = _guard_existing_repo_file(path, label=label)
    lowered = resolved.name.casefold()
    _require(
        not any(token in lowered for token in ("expectation", "answer", "gold")),
        f"{label} may not be an answer-bearing generation input",
    )
    return resolved


def _guard_output(path: Path, *, label: str, allow_existing: bool = False) -> Path:
    logical = Path(path).absolute()
    lowered_parts = {part.casefold() for part in logical.parts}
    _require(
        "validation" not in lowered_parts and ".git" not in lowered_parts,
        f"{label} may not use a reserved path",
    )
    try:
        parent = logical.parent.resolve(strict=True)
        parent.relative_to(ROOT.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as error:
        raise ControlExecutorError(
            f"{label} parent must exist inside the repository"
        ) from error
    resolved = parent / logical.name
    resolved_parts = {part.casefold() for part in resolved.parts}
    _require(
        "validation" not in resolved_parts and ".git" not in resolved_parts,
        f"{label} resolves through a reserved path",
    )
    if not allow_existing:
        _require(not resolved.exists(), f"refusing to overwrite {label}: {resolved}")
    return resolved


def load_json_strict(path: Path, *, generation_input: bool, label: str) -> Any:
    resolved = (
        _guard_generation_input(path, label=label)
        if generation_input
        else _guard_existing_repo_file(path, label=label)
    )
    try:
        return json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlExecutorError(f"cannot load strict JSON {resolved}: {error}") from error


def _load_jsonl(path: Path, *, generation_input: bool, label: str) -> list[dict[str, Any]]:
    resolved = (
        _guard_generation_input(path, label=label)
        if generation_input
        else _guard_existing_repo_file(path, label=label)
    )
    rows: list[dict[str, Any]] = []
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                _require(bool(line.strip()), f"blank {label} line {line_number}")
                value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
                rows.append(dict(_mapping(value, f"{label} line {line_number}")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlExecutorError(f"cannot load {label}: {error}") from error
    return rows


def _write_jsonl_atomic(
    path: Path, rows: Iterable[Mapping[str, Any]], *, label: str
) -> tuple[int, str]:
    resolved = _guard_output(path, label=label)
    temporary = Path(str(resolved) + ".tmp")
    _require(not temporary.exists(), f"temporary output already exists: {temporary}")
    count = 0
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json_text(row) + "\n")
                count += 1
        os.replace(temporary, resolved)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return count, sha256_file(resolved)


def _write_json_atomic(path: Path, value: Mapping[str, Any], *, label: str) -> str:
    resolved = _guard_output(path, label=label)
    temporary = Path(str(resolved) + ".tmp")
    _require(not temporary.exists(), f"temporary output already exists: {temporary}")
    try:
        temporary.write_bytes(canonical_json_bytes(value) + b"\n")
        os.replace(temporary, resolved)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return sha256_file(resolved)


def _repo_relative(path: Path) -> str:
    return str(path.resolve(strict=True).relative_to(ROOT.resolve(strict=True)))


def _canonical_binding(path: Path) -> dict[str, Any]:
    resolved = _guard_generation_input(path, label="bound JSON")
    value = load_json_strict(resolved, generation_input=True, label="bound JSON")
    return {
        "path": _repo_relative(resolved),
        "sha256": sha256_file(resolved),
        "canonical_sha256": sha256_bytes(canonical_json_bytes(value)),
    }


def _file_binding(path: Path, *, generation_input: bool, label: str) -> dict[str, Any]:
    resolved = (
        _guard_generation_input(path, label=label)
        if generation_input
        else _guard_existing_repo_file(path, label=label)
    )
    return {"path": _repo_relative(resolved), "sha256": sha256_file(resolved)}


def _resolve_binding_path(binding: Mapping[str, Any], *, label: str) -> Path:
    raw = Path(str(binding.get("path")))
    path = raw if raw.is_absolute() else ROOT / raw
    return _guard_generation_input(path, label=label)


def _binding_matches_file(
    binding: Mapping[str, Any],
    *,
    label: str,
    require_canonical: bool,
) -> tuple[Path, Any | None]:
    expected_fields = {"path", "sha256"}
    if require_canonical:
        expected_fields.add("canonical_sha256")
    _require(set(binding) == expected_fields, f"{label} binding fields changed")
    _require(
        SHA256_RE.fullmatch(str(binding.get("sha256"))) is not None,
        f"{label} SHA-256 invalid",
    )
    path = _resolve_binding_path(binding, label=label)
    _require(sha256_file(path) == binding["sha256"], f"{label} hash changed")
    value: Any | None = None
    if require_canonical:
        _require(
            SHA256_RE.fullmatch(str(binding.get("canonical_sha256"))) is not None,
            f"{label} canonical SHA-256 invalid",
        )
        value = load_json_strict(path, generation_input=True, label=label)
        _require(
            sha256_bytes(canonical_json_bytes(value))
            == binding["canonical_sha256"],
            f"{label} canonical hash changed",
        )
    return path, value


def _model_manifest_contains_answer_reference(value: Any) -> bool:
    forbidden = ("expectation", "expected", "gold", "label", "reason")
    if isinstance(value, Mapping):
        return any(
            any(token in str(key).casefold() for token in forbidden)
            or _model_manifest_contains_answer_reference(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_model_manifest_contains_answer_reference(item) for item in value)
    if isinstance(value, str):
        return any(token in value.casefold() for token in forbidden)
    return False


def _validate_model_input_row(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    _require(
        set(row) == {"schema_version", "kind", "pass", "packet_id_sha256", "payload"}
        and row.get("schema_version") == SCHEMA_VERSION
        and row.get("kind") == MODEL_INPUT_KIND
        and row.get("pass") in CONTROL_PASSES
        and isinstance(row.get("packet_id_sha256"), str)
        and SHA256_RE.fullmatch(str(row["packet_id_sha256"])) is not None,
        "sealed control model-input identity or fields changed",
    )
    payload = dict(_mapping(row.get("payload"), "sealed control payload"))
    pass_name = str(row["pass"])
    if pass_name == "completion":
        _require(
            set(payload) == {"assistant_text"}
            and isinstance(payload.get("assistant_text"), str),
            "completion control payload schema changed",
        )
    elif pass_name == "novelty":
        _require(
            set(payload) == {"visible_prefix", "locked_hypothesis"}
            and isinstance(payload.get("visible_prefix"), str)
            and isinstance(payload.get("locked_hypothesis"), str)
            and bool(payload["locked_hypothesis"]),
            "novelty control payload schema changed",
        )
    else:
        _require(
            set(payload) == {"assistant_text", "candidate_annotations"}
            and isinstance(payload.get("assistant_text"), str),
            "adjudication control payload schema changed",
        )
        candidates = list(
            _sequence(payload.get("candidate_annotations"), "blind candidates")
        )
        _require(len(candidates) == 2, "adjudication must supply exactly two candidates")
        validated = [quote_runner.validate_model_proposal(item) for item in candidates]
        _require(
            quote_runner.blind_candidate_order(
                packet_id_sha256=str(row["packet_id_sha256"]),
                left=validated[0],
                right=validated[1],
            )
            == validated,
            "sealed adjudication candidate order is not runner-authentic",
        )
        payload["candidate_annotations"] = validated
    row["payload"] = payload
    return row


def load_sealed_model_inputs(
    manifest_path: Path, *, expected_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path]:
    """Load only the answer-free 52-row model artifact and authenticate bytes."""

    _require(
        SHA256_RE.fullmatch(expected_sha256) is not None,
        "sealed model-input manifest expected SHA-256 invalid",
    )
    resolved_manifest = _guard_generation_input(
        manifest_path, label="sealed control model-input manifest"
    )
    _require(
        sha256_file(resolved_manifest) == expected_sha256,
        "sealed control model-input manifest hash changed",
    )
    manifest = dict(
        _mapping(
            load_json_strict(
                resolved_manifest,
                generation_input=True,
                label="sealed control model-input manifest",
            ),
            "sealed control model-input manifest",
        )
    )
    _require(
        set(manifest)
        == {
            "schema_version",
            "kind",
            "status",
            "scope",
            "passes",
            "counts",
            "records",
            "codebook",
        }
        and manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("kind") == MODEL_INPUT_MANIFEST_KIND
        and manifest.get("status") == "model_inputs_sealed",
        "sealed control model-input manifest identity or fields changed",
    )
    _require(
        not _model_manifest_contains_answer_reference(manifest),
        "sealed model-input manifest contains an answer reference",
    )
    scope = _mapping(manifest.get("scope"), "sealed model-input scope")
    _require(
        dict(scope)
        == {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
        },
        "sealed model-input scope changed",
    )
    _require(
        manifest.get("passes") == list(CONTROL_PASSES)
        and manifest.get("counts") == EXPECTED_COUNTS,
        "sealed model-input pass order or counts changed",
    )
    binding = _mapping(manifest.get("records"), "sealed model-input record binding")
    _require(
        set(binding) == {"path", "sha256", "count"}
        and binding.get("count") == 52
        and SHA256_RE.fullmatch(str(binding.get("sha256"))) is not None,
        "sealed model-input record binding invalid",
    )
    raw_record_path = resolved_manifest.parent / str(binding["path"])
    record_path = _guard_generation_input(
        raw_record_path, label="sealed control model-input JSONL"
    )
    _require(
        sha256_file(record_path) == binding["sha256"],
        "sealed model-input JSONL hash changed",
    )
    rows = [
        _validate_model_input_row(item)
        for item in _load_jsonl(
            record_path,
            generation_input=True,
            label="sealed control model-input JSONL",
        )
    ]
    _require(len(rows) == 52, "sealed model-input row count changed")
    ids = [str(row["packet_id_sha256"]) for row in rows]
    _require(len(set(ids)) == 52, "sealed model-input packet IDs are duplicated")
    observed = {
        name: sum(str(row["pass"]) == name for row in rows)
        for name in CONTROL_PASSES
    }
    _require(
        observed
        == {
            "completion": 32,
            "novelty": 8,
            "adjudication": 12,
        },
        "sealed model-input pass coverage changed",
    )
    return manifest, rows, resolved_manifest, record_path


def _codebook_binding(path: Path, codebook: Mapping[str, Any]) -> dict[str, Any]:
    resolved = _guard_generation_input(path, label="V2 codebook")
    return {
        "path": _repo_relative(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "canonical_sha256": sha256_bytes(canonical_json_bytes(codebook)),
    }


def _validate_runtime_control_contract(
    *,
    runtime_config: Mapping[str, Any],
    runtime_path: Path,
    model_manifest_path: Path,
    model_manifest_sha256: str,
) -> None:
    """Cross-bind the runner roster to this exact answer-free control seal.

    The expectation binding is inspected only as inert config metadata to prove
    that the roster marks it forbidden.  Its referenced path is never resolved,
    stat'ed, hashed, or opened here.
    """

    scope = _mapping(runtime_config.get("scope"), "runtime control scope")
    _require(
        scope.get("development_data_only") is True
        and scope.get("reserved_validation_closed") is True
        and scope.get("reserved_validation_accessed") is False
        and scope.get("target_annotation_authorized") is False
        and scope.get("controls_are_interface_readiness_only") is True
        and scope.get("affect_or_emotion_targeted") is False,
        "runtime config is not frozen for control-only interface readiness",
    )
    inputs = _mapping(runtime_config.get("inputs"), "runtime control inputs")
    sealed = _mapping(
        inputs.get("sealed_control_model_inputs"),
        "runtime sealed model-input binding",
    )
    sealed_path = Path(str(sealed.get("path")))
    if not sealed_path.is_absolute():
        sealed_path = ROOT / sealed_path
    sealed_path = _guard_generation_input(
        sealed_path, label="runtime sealed model-input manifest"
    )
    _require(
        set(sealed) == {"path", "sha256"}
        and sealed_path == model_manifest_path.resolve(strict=True)
        and sealed.get("sha256") == model_manifest_sha256,
        "runtime roster and executor bind different sealed model inputs",
    )
    expectation = _mapping(
        inputs.get("sealed_control_expectations"),
        "runtime forbidden expectation metadata",
    )
    _require(
        set(expectation) == {"path", "sha256", "forbidden_during_generation"}
        and isinstance(expectation.get("path"), str)
        and bool(str(expectation["path"]))
        and SHA256_RE.fullmatch(str(expectation.get("sha256"))) is not None
        and expectation.get("forbidden_during_generation") is True,
        "runtime roster does not mark sealed expectations forbidden during generation",
    )
    implementation = _mapping(
        _mapping(runtime_config.get("implementation"), "runtime implementation").get(
            "quote_runner"
        ),
        "runtime quote-runner binding",
    )
    runner_path = Path(str(implementation.get("path")))
    if not runner_path.is_absolute():
        runner_path = ROOT / runner_path
    runner_path = _guard_generation_input(runner_path, label="runtime quote runner")
    _require(
        set(implementation) == {"path", "sha256"}
        and runner_path == RUNNER_PATH.resolve(strict=True)
        and implementation.get("sha256") == sha256_file(RUNNER_PATH),
        "runtime roster quote-runner binding changed",
    )
    gate = _mapping(runtime_config.get("readiness_gate"), "runtime readiness gate")
    _require(
        gate.get("status_when_frozen") == "not_run"
        and gate.get(
            "target_packet_selection_or_generation_forbidden_until_controls_pass"
        )
        is True
        and gate.get("passing_controls_establishes_annotation_interface_readiness_only")
        is True
        and gate.get("any_bound_byte_or_model_identity_change_resets_gate") is True,
        "runtime readiness gate no longer forbids target generation",
    )
    del runtime_path  # retained to make the authenticated source explicit at call sites


def freeze_execution_config(
    *, runtime_config_path: Path, model_input_manifest_path: Path
) -> dict[str, Any]:
    """Return a prospective exact execution contract; never write or generate."""

    runtime_path = _guard_generation_input(
        runtime_config_path, label="V2 runtime config"
    )
    runtime_value = quote_runner.load_v2_config(runtime_path)
    codebook, codebook_path = quote_runner.authenticate_v2_codebook(
        runtime_value, config_path=runtime_path
    )
    runtime_binding = {
        "path": _repo_relative(runtime_path),
        "sha256": sha256_file(runtime_path),
        "canonical_sha256": sha256_bytes(canonical_json_bytes(runtime_value)),
    }
    model_manifest, _rows, manifest_path, record_path = load_sealed_model_inputs(
        model_input_manifest_path,
        expected_sha256=sha256_file(
            _guard_generation_input(
                model_input_manifest_path,
                label="sealed control model-input manifest",
            )
        ),
    )
    _validate_runtime_control_contract(
        runtime_config=runtime_value,
        runtime_path=runtime_path,
        model_manifest_path=manifest_path,
        model_manifest_sha256=sha256_file(manifest_path),
    )
    active_codebook_binding = _codebook_binding(codebook_path, codebook)
    _require(
        dict(_mapping(model_manifest.get("codebook"), "sealed model codebook"))
        == active_codebook_binding,
        "runtime and sealed model inputs bind different codebooks",
    )
    model_binding = _mapping(model_manifest["records"], "model records")
    config = {
        "schema_version": SCHEMA_VERSION,
        "kind": EXECUTOR_KIND,
        "status": FROZEN_STATUS,
        "scope": {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "target_annotations_present": False,
            "private_chain_of_thought_ground_truth_claimed": False,
            "affect_emotion_confidence_doubt_or_stress_targeted": False,
        },
        "bindings": {
            "runtime_config": runtime_binding,
            "codebook": active_codebook_binding,
            "sealed_model_inputs": {
                "path": _repo_relative(manifest_path),
                "sha256": sha256_file(manifest_path),
                "records": {
                    "path": _repo_relative(record_path),
                    "sha256": model_binding["sha256"],
                    "count": model_binding["count"],
                },
            },
            "decoder_v2_addendum": _canonical_binding(DECODER_ADDENDUM_PATH),
            "implementations": {
                "quote_runner_v2": _file_binding(
                    RUNNER_PATH, generation_input=True, label="quote runner V2"
                ),
                "control_builder_v2": _file_binding(
                    CONTROL_BUILDER_PATH,
                    generation_input=True,
                    label="control builder V2",
                ),
                "control_executor_v2": _file_binding(
                    EXECUTOR_PATH,
                    generation_input=True,
                    label="control executor V2",
                ),
            },
        },
        "execution_contract": {
            "only_sealed_control_manifest_kind": MODEL_INPUT_MANIFEST_KIND,
            "role_record_counts": dict(ROLE_RECORD_COUNTS),
            "primary_input_passes": ["completion", "novelty"],
            "adjudicator_input_passes": ["adjudication"],
            "separate_process_per_role": True,
            "expectations_inaccessible_during_generation": True,
            "resume_requires_authenticated_complete_role_manifest": True,
            "target_packets_forbidden": True,
            "combined_output_records_per_primary": 52,
        },
    }
    return validate_execution_config(config)


def validate_execution_config(value: Any) -> dict[str, Any]:
    config = dict(_mapping(value, "control execution config"))
    _require(
        set(config)
        == {
            "schema_version",
            "kind",
            "status",
            "scope",
            "bindings",
            "execution_contract",
        }
        and config.get("schema_version") == SCHEMA_VERSION
        and config.get("kind") == EXECUTOR_KIND
        and config.get("status") == FROZEN_STATUS,
        "control execution config is not prospectively frozen and runnable",
    )
    _require(
        dict(_mapping(config.get("scope"), "execution scope"))
        == {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "target_annotations_present": False,
            "private_chain_of_thought_ground_truth_claimed": False,
            "affect_emotion_confidence_doubt_or_stress_targeted": False,
        },
        "control execution scope changed",
    )
    contract = dict(_mapping(config.get("execution_contract"), "execution contract"))
    _require(
        contract
        == {
            "only_sealed_control_manifest_kind": MODEL_INPUT_MANIFEST_KIND,
            "role_record_counts": dict(ROLE_RECORD_COUNTS),
            "primary_input_passes": ["completion", "novelty"],
            "adjudicator_input_passes": ["adjudication"],
            "separate_process_per_role": True,
            "expectations_inaccessible_during_generation": True,
            "resume_requires_authenticated_complete_role_manifest": True,
            "target_packets_forbidden": True,
            "combined_output_records_per_primary": 52,
        },
        "control execution contract changed",
    )
    bindings = _mapping(config.get("bindings"), "execution bindings")
    _require(
        set(bindings)
        == {
            "runtime_config",
            "codebook",
            "sealed_model_inputs",
            "decoder_v2_addendum",
            "implementations",
        },
        "execution binding fields changed",
    )
    runtime = _mapping(bindings.get("runtime_config"), "runtime config binding")
    _require(
        set(runtime) == {"path", "sha256", "canonical_sha256"}
        and SHA256_RE.fullmatch(str(runtime.get("sha256"))) is not None
        and SHA256_RE.fullmatch(str(runtime.get("canonical_sha256"))) is not None,
        "runtime config binding invalid",
    )
    codebook = _mapping(bindings.get("codebook"), "codebook binding")
    _require(
        set(codebook) == {"path", "size_bytes", "sha256", "canonical_sha256"}
        and isinstance(codebook.get("size_bytes"), int)
        and not isinstance(codebook.get("size_bytes"), bool)
        and int(codebook["size_bytes"]) > 0
        and SHA256_RE.fullmatch(str(codebook.get("sha256"))) is not None
        and SHA256_RE.fullmatch(str(codebook.get("canonical_sha256"))) is not None,
        "codebook binding invalid",
    )
    model = _mapping(bindings.get("sealed_model_inputs"), "sealed model binding")
    records = _mapping(model.get("records"), "sealed model record binding")
    _require(
        set(model) == {"path", "sha256", "records"}
        and SHA256_RE.fullmatch(str(model.get("sha256"))) is not None
        and set(records) == {"path", "sha256", "count"}
        and SHA256_RE.fullmatch(str(records.get("sha256"))) is not None
        and records.get("count") == 52,
        "sealed model-input binding invalid",
    )
    decoder = _mapping(
        bindings.get("decoder_v2_addendum"), "decoder V2 addendum binding"
    )
    _require(
        set(decoder) == {"path", "sha256", "canonical_sha256"}
        and SHA256_RE.fullmatch(str(decoder.get("sha256"))) is not None
        and SHA256_RE.fullmatch(str(decoder.get("canonical_sha256"))) is not None,
        "decoder V2 addendum binding invalid",
    )
    implementations = _mapping(bindings.get("implementations"), "implementations")
    _require(
        set(implementations)
        == {"quote_runner_v2", "control_builder_v2", "control_executor_v2"},
        "implementation binding set changed",
    )
    for name, binding in implementations.items():
        item = _mapping(binding, f"{name} implementation binding")
        _require(
            set(item) == {"path", "sha256"}
            and SHA256_RE.fullmatch(str(item.get("sha256"))) is not None,
            f"{name} implementation binding invalid",
        )
    return config


def _authenticate_execution_context(
    *, execution_config_path: Path, expected_execution_config_sha256: str
) -> dict[str, Any]:
    _require(
        SHA256_RE.fullmatch(expected_execution_config_sha256) is not None,
        "expected execution-config SHA-256 is required",
    )
    config_path = _guard_generation_input(
        execution_config_path, label="control execution config"
    )
    _require(
        sha256_file(config_path) == expected_execution_config_sha256,
        "control execution config hash changed",
    )
    config = validate_execution_config(
        load_json_strict(
            config_path, generation_input=True, label="control execution config"
        )
    )
    bindings = _mapping(config["bindings"], "execution bindings")
    implementations = _mapping(bindings["implementations"], "implementations")
    expected_implementation_paths = {
        "quote_runner_v2": RUNNER_PATH,
        "control_builder_v2": CONTROL_BUILDER_PATH,
        "control_executor_v2": EXECUTOR_PATH,
    }
    resolved_implementations: dict[str, dict[str, Any]] = {}
    for name, expected_path in expected_implementation_paths.items():
        binding = _mapping(implementations[name], f"{name} binding")
        resolved, _unused = _binding_matches_file(
            binding, label=name, require_canonical=False
        )
        _require(
            resolved == expected_path.resolve(strict=True),
            f"{name} binding points to a different implementation",
        )
        resolved_implementations[name] = dict(binding)
    decoder_binding = _mapping(
        bindings["decoder_v2_addendum"], "decoder V2 addendum binding"
    )
    decoder_path, decoder_value = _binding_matches_file(
        decoder_binding, label="decoder V2 addendum", require_canonical=True
    )
    _require(
        decoder_path == DECODER_ADDENDUM_PATH.resolve(strict=True)
        and _mapping(decoder_value, "decoder V2 addendum").get("kind")
        == "swe_task_state_v4_epistemic_chain_decoder_v2_addendum",
        "decoder V2 addendum identity changed",
    )

    runtime_binding = _mapping(bindings["runtime_config"], "runtime binding")
    runtime_path, runtime_disk = _binding_matches_file(
        runtime_binding, label="V2 runtime config", require_canonical=True
    )
    runtime_config = quote_runner.load_v2_config(runtime_path)
    _require(
        canonical_json_bytes(runtime_config) == canonical_json_bytes(runtime_disk),
        "validated runtime config differs from authenticated bytes",
    )
    _require(
        all(
            _mapping(runtime_config["roles"][role], f"{role} runtime role").get(
                "execution_mode"
            )
            == "local_model"
            for role in ROLES
        ),
        "sealed real-model controls require three local model roles",
    )

    runtime_codebook, runtime_codebook_path = quote_runner.authenticate_v2_codebook(
        runtime_config, config_path=runtime_path
    )
    codebook_binding = dict(_mapping(bindings["codebook"], "codebook binding"))
    expected_codebook = _codebook_binding(runtime_codebook_path, runtime_codebook)
    _require(
        codebook_binding == expected_codebook,
        "execution config and runtime config bind different codebook bytes",
    )

    model_binding = _mapping(bindings["sealed_model_inputs"], "sealed model binding")
    model_manifest_path = _resolve_binding_path(
        model_binding, label="sealed model-input manifest"
    )
    manifest, rows, model_manifest_path, model_record_path = load_sealed_model_inputs(
        model_manifest_path, expected_sha256=str(model_binding["sha256"])
    )
    _validate_runtime_control_contract(
        runtime_config=runtime_config,
        runtime_path=runtime_path,
        model_manifest_path=model_manifest_path,
        model_manifest_sha256=str(model_binding["sha256"]),
    )
    record_binding = _mapping(model_binding["records"], "sealed record binding")
    _require(
        _repo_relative(model_record_path) == record_binding["path"]
        and sha256_file(model_record_path) == record_binding["sha256"]
        and len(rows) == record_binding["count"],
        "execution config sealed model-input JSONL binding changed",
    )
    _require(
        dict(_mapping(manifest["codebook"], "model-input codebook"))
        == codebook_binding,
        "sealed model inputs bind different codebook bytes",
    )
    # This authenticates the exact config/codebook pair without consulting the
    # config's separately declared expectation path.
    quote_runner.authenticate_passed_contracts(
        config=runtime_config,
        config_path=runtime_path,
        codebook=runtime_codebook,
        codebook_path=runtime_codebook_path,
    )
    return {
        "execution_config": config,
        "execution_config_path": config_path,
        "execution_config_sha256": expected_execution_config_sha256,
        "runtime_config": runtime_config,
        "runtime_config_path": runtime_path,
        "codebook": runtime_codebook,
        "codebook_path": runtime_codebook_path,
        "model_input_manifest": manifest,
        "model_input_manifest_path": model_manifest_path,
        "model_input_record_path": model_record_path,
        "model_input_rows": rows,
        "implementations": resolved_implementations,
        "decoder_addendum_path": decoder_path,
        "decoder_addendum": decoder_value,
    }


def _blind_shards(packet_id_sha256: str) -> dict[str, int]:
    digest = bytes.fromhex(packet_id_sha256)
    left = digest[0] % 8
    right = digest[1] % 7
    if right >= left:
        right += 1
    return {"independent_a": left, "independent_b": right}


def adapt_control_input_to_runner_packet(
    row: Mapping[str, Any], *, model_input_manifest_sha256: str
) -> dict[str, Any]:
    """Adapt one sealed control to the exact legacy-authenticated runner schema."""

    validated = _validate_model_input_row(row)
    pass_name = str(validated["pass"])
    _require(pass_name != "adjudication" or "candidate_annotations" in validated["payload"],
             "adjudication control candidates missing")
    packet_id = str(validated["packet_id_sha256"])
    payload = dict(validated["payload"])
    payload_sha = sha256_bytes(canonical_json_bytes(payload))
    source_id = sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "sealed-epistemic-chain-control-v2-source",
                "packet_id_sha256": packet_id,
                "pass": pass_name,
                "payload_sha256": payload_sha,
                "model_input_manifest_sha256": model_input_manifest_sha256,
            }
        )
    )
    common = {
        "schema_version": SCHEMA_VERSION,
        "packet_id_sha256": packet_id,
        "source_id_sha256": source_id,
        "blind_shards": _blind_shards(packet_id),
    }
    if pass_name in {"completion", "adjudication"}:
        text = str(payload["assistant_text"])
        packet = {
            **common,
            "kind": quote_runner.legacy.COMPLETION_PACKET_KIND,
            "annotation_pass": "completion_chain",
            "materialized_assistant_text": {
                "char_start": 0,
                "char_end": len(text),
                "sha256": sha256_text(text),
                "text": text,
            },
            "authenticated_boundaries": {
                "sealed_control_model_input_manifest_sha256": model_input_manifest_sha256,
                "sealed_control_payload_sha256": payload_sha,
            },
            "annotator_visibility": {
                "assistant_tool_arguments_present": False,
                "complete_prefix_text_present": False,
                "model_features_present": False,
                "repository_or_task_identity_present": False,
                "tool_results_present": False,
            },
        }
        return quote_runner.legacy.validate_packet(
            packet, annotation_pass="completion_chain"
        )
    prefix = str(payload["visible_prefix"])
    hypothesis = str(payload["locked_hypothesis"])
    completion_sha = sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "sealed-novelty-control-locked-completion",
                "packet_id_sha256": packet_id,
                "locked_hypothesis": hypothesis,
            }
        )
    )
    packet = {
        **common,
        "kind": quote_runner.legacy.PREFIX_PACKET_KIND,
        "annotation_pass": "prefix_novelty",
        "locked_hypothesis": {
            "text": hypothesis,
            "sha256": sha256_text(hypothesis),
            "completion_char_start": 0,
            "completion_char_end": len(hypothesis),
            "materialized_completion_sha256": completion_sha,
        },
        "authenticated_prefix": {
            "source_sha256": sha256_text(prefix),
            "source_char_start": 0,
            "source_char_end": len(prefix),
            "annotator_text": prefix,
            "annotator_text_sha256": sha256_text(prefix),
            "annotator_char_start": 0,
            "annotator_char_end": len(prefix),
            "removed_ranges": [],
        },
        "annotator_visibility": {
            "completion_chain_slots_present": False,
            "assistant_tool_arguments_present": False,
            "tool_results_present": False,
            "repository_or_task_identity_present": False,
            "model_features_present": False,
        },
    }
    return quote_runner.legacy.validate_packet(packet, annotation_pass="prefix_novelty")


def _model_identity(role_spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "repo_id": role_spec["repo_id"],
        "revision": role_spec["revision"],
        "snapshot_tree_sha256": role_spec["snapshot_tree_sha256"],
        "quantization": role_spec.get("quantization"),
        "dtype": role_spec["dtype"],
    }


def _validate_model_inventory(
    *,
    inventory: Mapping[str, Any],
    role_spec: Mapping[str, Any],
    production_backend: bool,
) -> dict[str, Any]:
    value = copy.deepcopy(dict(inventory))
    identity = _model_identity(role_spec)
    identity_sha = sha256_bytes(canonical_json_bytes(identity))
    _require(
        value.get("model_identity") == identity
        and value.get("model_identity_sha256") == identity_sha
        and value.get("tree_sha256") == role_spec["snapshot_tree_sha256"]
        and value.get("file_count") == role_spec["snapshot_file_count"]
        and value.get("size_bytes") == role_spec["snapshot_size_bytes"]
        and isinstance(value.get("snapshot_path"), str)
        and bool(value["snapshot_path"])
        and value.get("verification_deferred") is not True,
        "resolved model inventory does not authenticate the frozen role identity",
    )
    if production_backend:
        expected_snapshot = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / ("models--" + str(role_spec["repo_id"]).replace("/", "--"))
            / "snapshots"
            / str(role_spec["revision"])
        ).resolve(strict=True)
        _require(
            Path(str(value["snapshot_path"])).resolve(strict=True)
            == expected_snapshot,
            "production model inventory snapshot path differs from frozen repo/revision",
        )
    return value


def _role_source_rows(context: Mapping[str, Any], role: str) -> list[dict[str, Any]]:
    _require(role in ROLES, "control role invalid")
    allowed = set(ROLE_PASS_CONTRACT[role])
    rows = [
        dict(row)
        for row in _sequence(context["model_input_rows"], "model input rows")
        if str(_mapping(row, "model input row")["pass"]) in allowed
    ]
    _require(
        len(rows) == ROLE_RECORD_COUNTS[role]
        and {str(row["pass"]) for row in rows} == allowed,
        f"{role} sealed input coverage changed",
    )
    return rows


def _role_record_path(manifest_path: Path) -> Path:
    return Path(manifest_path).absolute().with_suffix(".jsonl")


def _runner_annotation_pass(control_pass: str) -> str:
    return "prefix_novelty" if control_pass == "novelty" else "completion_chain"


def _expected_result_provenance(
    *,
    role: str,
    role_spec: Mapping[str, Any],
    model_identity_sha256: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "role": role,
        "base_model_lineage": quote_runner._lineage_id(role_spec, role),
        "model_identity_sha256": model_identity_sha256,
        "runner_config_sha256": sha256_file(context["runtime_config_path"]),
        "codebook_sha256": sha256_file(context["codebook_path"]),
        "sealed_model_input_manifest_sha256": sha256_file(
            context["model_input_manifest_path"]
        ),
        "generation_sha256": sha256_bytes(
            canonical_json_bytes(context["runtime_config"]["generation"])
        ),
        "seed": role_spec["seed"],
    }
    return {
        "annotator_identity": identity,
        "annotator_identity_sha256": sha256_bytes(canonical_json_bytes(identity)),
        "model_identity_sha256": model_identity_sha256,
        "execution_config_sha256": context["execution_config_sha256"],
        "runtime_config_canonical_sha256": sha256_bytes(
            canonical_json_bytes(context["runtime_config"])
        ),
        "codebook_canonical_sha256": sha256_bytes(
            canonical_json_bytes(context["codebook"])
        ),
    }


def _result_blinding_contract() -> dict[str, bool]:
    return {
        "sealed_control_payload_allowlist_only": True,
        "expectations_opened_or_model_visible": False,
        "repository_or_task_identity_exposed": False,
        "tool_arguments_or_results_exposed": False,
        "activations_or_lens_predictions_exposed": False,
        "outcomes_exposed": False,
        "candidate_lane_identity_exposed": False,
    }


def _result_with_provenance(
    *,
    result: Mapping[str, Any],
    role: str,
    role_spec: Mapping[str, Any],
    model_inventory: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(dict(result))
    value["provenance"] = _expected_result_provenance(
        role=role,
        role_spec=role_spec,
        model_identity_sha256=str(model_inventory["model_identity_sha256"]),
        context=context,
    )
    value["blinding"] = _result_blinding_contract()
    return value


def _validate_result_record(
    *,
    result: Mapping[str, Any],
    role: str,
    control_pass: str,
    packet_id: str,
    source_id: str,
    model_identity_sha256: str,
    expected_prompt_messages_sha256: str,
    expected_candidate_order_sha256: str | None,
    role_spec: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    _require(
        result.get("schema_version") == SCHEMA_VERSION
        and result.get("interface_version") == quote_runner.INTERFACE_VERSION
        and result.get("kind") == quote_runner.RECORD_KIND
        and result.get("role") == role
        and result.get("annotation_pass") == _runner_annotation_pass(control_pass)
        and result.get("packet_id_sha256") == packet_id
        and result.get("source_id_sha256") == source_id,
        "runner control result identity changed",
    )
    generation = _mapping(result.get("generation"), "runner generation record")
    _require(
        set(generation)
        == {
            "model_invoked",
            "reason",
            "raw_output",
            "prompt_messages_sha256",
            "rendered_prompt_sha256",
            "candidate_order_sha256",
            "prompt_token_count",
            "output_token_count",
            "finish_reason",
            "output_extraction",
        },
        "runner generation fields changed",
    )
    if generation.get("model_invoked") is True:
        prompt_count = generation.get("prompt_token_count")
        output_count = generation.get("output_token_count")
        extraction = _mapping(
            generation.get("output_extraction"), "runner output extraction"
        )
        _require(
            isinstance(generation.get("raw_output"), str)
            and generation["raw_output"] == result.get("raw_model_output")
            and result.get("raw_model_output_sha256")
            == sha256_text(str(generation["raw_output"]))
            and generation.get("reason") is None
            and generation.get("prompt_messages_sha256")
            == expected_prompt_messages_sha256
            and SHA256_RE.fullmatch(
                str(generation.get("rendered_prompt_sha256"))
            )
            is not None
            and generation.get("candidate_order_sha256")
            == expected_candidate_order_sha256
            and isinstance(prompt_count, int)
            and not isinstance(prompt_count, bool)
            and prompt_count > 0
            and isinstance(output_count, int)
            and not isinstance(output_count, bool)
            and 0 < output_count <= context["runtime_config"]["generation"][
                "max_output_tokens"
            ]
            and prompt_count
            + context["runtime_config"]["generation"]["max_output_tokens"]
            <= context["runtime_config"]["generation"]["max_model_len"]
            and isinstance(generation.get("finish_reason"), str)
            and bool(generation["finish_reason"])
            and extraction.get("output_extraction")
            == role_spec["output_extraction"]
            and extraction.get("analysis_content_retained") is False
            and extraction.get("final_message_count") == 1
            and extraction.get("final_text_sha256")
            == sha256_text(str(generation["raw_output"])),
            "raw final model output was not preserved exactly",
        )
        if role_spec["output_extraction"] == "openai_harmony_final_channel":
            _require(
                extraction.get("parser_package") == "openai-harmony"
                and extraction.get("parser_strict") is True,
                "GPT-OSS Harmony extraction provenance changed",
            )
    else:
        _require(
            role in PRIMARY_ROLES
            and control_pass == "completion"
            and generation.get("reason")
            == "empty_visible_prose_deterministic_no_chain"
            and generation.get("raw_output") is None
            and generation.get("prompt_messages_sha256") is None
            and generation.get("rendered_prompt_sha256") is None
            and generation.get("candidate_order_sha256") is None
            and generation.get("prompt_token_count") == 0
            and generation.get("output_token_count") == 0
            and generation.get("finish_reason") is None
            and generation.get("output_extraction") is None,
            "only the sealed empty completion may bypass model generation",
        )
    provenance = _mapping(result.get("provenance"), "control result provenance")
    _require(
        dict(provenance)
        == _expected_result_provenance(
            role=role,
            role_spec=role_spec,
            model_identity_sha256=model_identity_sha256,
            context=context,
        )
        and result.get("blinding") == _result_blinding_contract(),
        "control result provenance or blinding contract changed",
    )


def _runner_messages_for_control(
    *,
    source: Mapping[str, Any],
    packet: Mapping[str, Any],
    codebook: Mapping[str, Any],
    role: str,
) -> list[dict[str, str]]:
    candidates = None
    if role == "adjudicator":
        supplied = list(
            _sequence(
                _mapping(source["payload"], "adjudication payload")[
                    "candidate_annotations"
                ],
                "adjudication candidates",
            )
        )
        candidates = (supplied[0], supplied[1])
    return quote_runner.build_messages(
        packet=packet,
        codebook=codebook,
        annotation_pass=_runner_annotation_pass(str(source["pass"])),
        candidate_records=candidates,
    )


def _assert_model_visible_payload_matches_seal(
    *,
    source: Mapping[str, Any],
    packet: Mapping[str, Any],
    codebook: Mapping[str, Any],
    role: str,
) -> str:
    """Prove the runner user payload is byte-equal to the sealed payload."""

    messages = _runner_messages_for_control(
        source=source, packet=packet, codebook=codebook, role=role
    )
    _require(len(messages) == 2, "runner prompt message count changed")
    visible = json.loads(
        str(messages[1]["content"]), object_pairs_hook=_reject_duplicate_keys
    )
    sealed_payload = dict(_mapping(source["payload"], "sealed payload"))
    _require(
        canonical_json_bytes(visible) == canonical_json_bytes(sealed_payload),
        "runner model-visible payload differs from the exact sealed payload",
    )
    return sha256_bytes(canonical_json_bytes(visible))


def _expected_prompt_bindings(
    *,
    source: Mapping[str, Any],
    packet: Mapping[str, Any],
    codebook: Mapping[str, Any],
    role: str,
) -> tuple[str, str | None]:
    messages = _runner_messages_for_control(
        source=source, packet=packet, codebook=codebook, role=role
    )
    candidate_sha: str | None = None
    if role == "adjudicator":
        visible = json.loads(
            str(messages[1]["content"]), object_pairs_hook=_reject_duplicate_keys
        )
        candidate_sha = sha256_bytes(
            canonical_json_bytes(visible["candidate_annotations"])
        )
    return sha256_bytes(canonical_json_bytes(messages)), candidate_sha


def _role_fingerprints(
    *,
    context: Mapping[str, Any],
    role: str,
    role_spec: Mapping[str, Any],
    model_identity_sha256: str,
) -> dict[str, Any]:
    passes = ROLE_PASS_CONTRACT[role]
    schema_hashes: dict[str, str] = {}
    prompt_hashes: dict[str, str] = {}
    for control_pass in passes:
        annotation_pass = _runner_annotation_pass(control_pass)
        schema_hashes[control_pass] = sha256_bytes(
            canonical_json_bytes(
                quote_runner.response_schema(
                    context["codebook"], annotation_pass=annotation_pass
                )
            )
        )
        prompt_hashes[control_pass] = sha256_text(
            quote_runner._system_prompt(
                codebook=context["codebook"],
                annotation_pass=annotation_pass,
                adjudication=role == "adjudicator",
            )
        )
    return {
        "model_identity_sha256": model_identity_sha256,
        "chat_template_kwargs_sha256": sha256_bytes(
            canonical_json_bytes(role_spec["chat_template_kwargs"])
        ),
        "output_extraction": role_spec["output_extraction"],
        "generation_sha256": sha256_bytes(
            canonical_json_bytes(context["runtime_config"]["generation"])
        ),
        "response_schema_sha256_by_pass": schema_hashes,
        "prompt_template_sha256_by_pass": prompt_hashes,
        "seed": role_spec["seed"],
    }


def _validate_persisted_runtime(
    *, runtime: Mapping[str, Any], role_spec: Mapping[str, Any], context: Mapping[str, Any]
) -> None:
    generation = _mapping(context["runtime_config"]["generation"], "generation")
    _require(
        runtime.get("output_extraction") == role_spec["output_extraction"]
        and runtime.get("chat_template_kwargs") == role_spec["chat_template_kwargs"]
        and runtime.get("chat_template_kwargs_sha256")
        == sha256_bytes(canonical_json_bytes(role_spec["chat_template_kwargs"]))
        and runtime.get("vllm_use_flashinfer_sampler")
        == generation["vllm_use_flashinfer_sampler"]
        and runtime.get("vllm_enable_v1_multiprocessing")
        == generation["vllm_enable_v1_multiprocessing"]
        and runtime.get("vllm_disabled_kernels")
        == ",".join(generation["vllm_disabled_kernels"])
        and runtime.get("cuda_home") is None
        and runtime.get("prompt_token_accounting")
        == "vllm_request_output.prompt_token_ids"
        and runtime.get("external_tokenizer_preflight_count_used") is False
        and runtime.get("prompt_truncation_requested") is False
        and runtime.get("engine_authoritative_context_reservation_checked") is True
        and runtime.get("structured_outputs_config")
        == quote_runner.STRUCTURED_OUTPUTS_ENGINE_CONFIG
        and runtime.get("structured_outputs_config_sha256")
        == sha256_bytes(
            canonical_json_bytes(quote_runner.STRUCTURED_OUTPUTS_ENGINE_CONFIG)
        )
        and runtime.get("language_model_only") is True
        and runtime.get("role_vllm_engine_kwargs")
        == role_spec["vllm_engine_kwargs"]
        and SHA256_RE.fullmatch(str(runtime.get("llm_kwargs_sha256"))) is not None
        and isinstance(runtime.get("load_seconds"), (int, float))
        and not isinstance(runtime.get("load_seconds"), bool)
        and runtime["load_seconds"] >= 0
        and isinstance(runtime.get("control_generation_seconds"), (int, float))
        and not isinstance(runtime.get("control_generation_seconds"), bool)
        and runtime["control_generation_seconds"] >= 0,
        "persisted role runtime differs from the frozen generation contract",
    )
    if role_spec["output_extraction"] == "openai_harmony_final_channel":
        _require(
            runtime.get("harmony_parser_package") == "openai-harmony"
            and runtime.get("harmony_parser_strict") is True,
            "persisted GPT-OSS Harmony runtime contract changed",
        )


def _manifest_input_bindings(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "execution_config": {
            "path": str(Path(context["execution_config_path"]).resolve(strict=True)),
            "sha256": context["execution_config_sha256"],
        },
        "runtime_config": {
            "path": str(Path(context["runtime_config_path"]).resolve(strict=True)),
            "sha256": sha256_file(context["runtime_config_path"]),
            "canonical_sha256": sha256_bytes(
                canonical_json_bytes(context["runtime_config"])
            ),
        },
        "codebook": {
            "path": str(Path(context["codebook_path"]).resolve(strict=True)),
            "sha256": sha256_file(context["codebook_path"]),
            "canonical_sha256": sha256_bytes(
                canonical_json_bytes(context["codebook"])
            ),
        },
        "sealed_model_inputs": {
            "path": str(Path(context["model_input_manifest_path"]).resolve(strict=True)),
            "sha256": sha256_file(context["model_input_manifest_path"]),
            "records_path": str(
                Path(context["model_input_record_path"]).resolve(strict=True)
            ),
            "records_sha256": sha256_file(context["model_input_record_path"]),
        },
        "decoder_v2_addendum": {
            "path": str(Path(context["decoder_addendum_path"]).resolve(strict=True)),
            "sha256": sha256_file(context["decoder_addendum_path"]),
            "canonical_sha256": sha256_bytes(
                canonical_json_bytes(context["decoder_addendum"])
            ),
        },
        "implementations": copy.deepcopy(dict(context["implementations"])),
    }


def _validate_role_manifest(
    *,
    context: Mapping[str, Any],
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_role: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    _require(expected_role in ROLES, "expected role invalid")
    _require(
        SHA256_RE.fullmatch(expected_manifest_sha256) is not None,
        f"expected {expected_role} role-manifest SHA-256 is required",
    )
    resolved_manifest = _guard_existing_repo_file(
        manifest_path, label=f"{expected_role} complete role manifest"
    )
    _require(
        sha256_file(resolved_manifest) == expected_manifest_sha256,
        f"{expected_role} complete role-manifest hash changed",
    )
    manifest = dict(
        _mapping(
            load_json_strict(
                resolved_manifest,
                generation_input=False,
                label=f"{expected_role} complete role manifest",
            ),
            "complete role manifest",
        )
    )
    _require(
        set(manifest)
        == {
            "schema_version",
            "kind",
            "status",
            "role",
            "scope",
            "passes",
            "counts",
            "records",
            "inputs",
            "model",
            "runtime",
            "fingerprints",
            "execution_backend",
            "gate_eligible",
        }
        and manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("kind") == ROLE_MANIFEST_KIND
        and manifest.get("status") == "sealed_control_role_generation_complete"
        and manifest.get("role") == expected_role
        and manifest.get("execution_backend")
        in {"real_local_model_default_runtime", "injected_test_non_gate"}
        and manifest.get("gate_eligible")
        is (manifest.get("execution_backend") == "real_local_model_default_runtime"),
        "complete role manifest identity or fields changed",
    )
    _require(
        dict(_mapping(manifest.get("scope"), "role scope"))
        == {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "target_annotations_present": False,
            "expectations_opened_during_generation": False,
            "model_visible_payload_byte_equal_to_sealed_payload": True,
        },
        "role output scope changed",
    )
    expected_source_rows = _role_source_rows(context, expected_role)
    expected_passes = list(ROLE_PASS_CONTRACT[expected_role])
    _require(
        manifest.get("passes") == expected_passes
        and manifest.get("counts")
        == {
            "input_records": ROLE_RECORD_COUNTS[expected_role],
            "output_records": ROLE_RECORD_COUNTS[expected_role],
            "model_invocations": (
                ROLE_RECORD_COUNTS[expected_role]
                - (
                    1
                    if expected_role in PRIMARY_ROLES
                    and any(
                        row["pass"] == "completion"
                        and row["payload"]["assistant_text"] == ""
                        for row in expected_source_rows
                    )
                    else 0
                )
            ),
        },
        "complete role pass or count contract changed",
    )
    _require(
        manifest.get("inputs") == _manifest_input_bindings(context),
        "complete role inputs differ from authenticated execution context",
    )
    role_spec = _mapping(
        context["runtime_config"]["roles"][expected_role], "runtime role spec"
    )
    model = _validate_model_inventory(
        inventory=_mapping(manifest.get("model"), "role model inventory"),
        role_spec=role_spec,
        production_backend=bool(manifest["gate_eligible"]),
    )
    persisted_runtime = _mapping(manifest.get("runtime"), "role runtime")
    _validate_persisted_runtime(
        runtime=persisted_runtime, role_spec=role_spec, context=context
    )
    _require(
        manifest.get("fingerprints")
        == _role_fingerprints(
            context=context,
            role=expected_role,
            role_spec=role_spec,
            model_identity_sha256=str(model["model_identity_sha256"]),
        ),
        "role prompt, schema, generation, or model fingerprints changed",
    )
    binding = _mapping(manifest.get("records"), "role record binding")
    _require(
        set(binding) == {"path", "sha256", "count"}
        and binding.get("count") == ROLE_RECORD_COUNTS[expected_role]
        and SHA256_RE.fullmatch(str(binding.get("sha256"))) is not None,
        "complete role record binding invalid",
    )
    record_path = _guard_existing_repo_file(
        resolved_manifest.parent / str(binding["path"]),
        label=f"{expected_role} complete role records",
    )
    _require(
        sha256_file(record_path) == binding["sha256"],
        "complete role record hash changed",
    )
    records = _load_jsonl(
        record_path,
        generation_input=False,
        label=f"{expected_role} complete role records",
    )
    _require(
        len(records) == ROLE_RECORD_COUNTS[expected_role],
        "complete role record count changed",
    )
    expected_pairs = [
        (str(row["packet_id_sha256"]), str(row["pass"]))
        for row in expected_source_rows
    ]
    observed_pairs: list[tuple[str, str]] = []
    for record, source in zip(records, expected_source_rows, strict=True):
        _require(
            set(record)
            == {
                "schema_version",
                "kind",
                "role",
                "pass",
                "packet_id_sha256",
                "input_payload_sha256",
                "model_visible_payload_sha256",
                "runner_packet_sha256",
                "result",
            }
            and record.get("schema_version") == SCHEMA_VERSION
            and record.get("kind") == ROLE_RECORD_KIND
            and record.get("role") == expected_role
            and record.get("packet_id_sha256") == source["packet_id_sha256"]
            and record.get("pass") == source["pass"]
            and record.get("input_payload_sha256")
            == sha256_bytes(canonical_json_bytes(source["payload"])),
            "complete role record fields, order, or input binding changed",
        )
        adapted = adapt_control_input_to_runner_packet(
            source,
            model_input_manifest_sha256=sha256_file(
                context["model_input_manifest_path"]
            ),
        )
        _require(
            record.get("runner_packet_sha256")
            == sha256_bytes(canonical_json_bytes(adapted)),
            "complete role adapted packet binding changed",
        )
        _require(
            record.get("model_visible_payload_sha256")
            == _assert_model_visible_payload_matches_seal(
                source=source,
                packet=adapted,
                codebook=context["codebook"],
                role=expected_role,
            ),
            "complete role model-visible payload binding changed",
        )
        result = _mapping(record.get("result"), "role result")
        prompt_sha, candidate_sha = _expected_prompt_bindings(
            source=source,
            packet=adapted,
            codebook=context["codebook"],
            role=expected_role,
        )
        _validate_result_record(
            result=result,
            role=expected_role,
            control_pass=str(source["pass"]),
            packet_id=str(source["packet_id_sha256"]),
            source_id=str(adapted["source_id_sha256"]),
            model_identity_sha256=str(model["model_identity_sha256"]),
            expected_prompt_messages_sha256=prompt_sha,
            expected_candidate_order_sha256=candidate_sha,
            role_spec=role_spec,
            context=context,
        )
        observed_pairs.append((str(record["packet_id_sha256"]), str(record["pass"])))
    _require(observed_pairs == expected_pairs, "complete role coverage or order changed")
    return manifest, records, resolved_manifest


GeneratorFactory = Callable[..., tuple[quote_runner.GenerateBatch, Mapping[str, Any], Any]]
ModelResolver = Callable[..., tuple[Path, Mapping[str, Any]]]


def run_role(
    *,
    execution_config_path: Path,
    expected_execution_config_sha256: str,
    role: str,
    output_manifest_path: Path,
    resume: bool = False,
    expected_resume_manifest_sha256: str | None = None,
    generator_factory: GeneratorFactory = quote_runner.make_vllm_generator_v2,
    model_resolver: ModelResolver = quote_runner.legacy.resolve_model_snapshot,
    _allow_mocked_cpu_test: bool = False,
) -> dict[str, Any]:
    """Run exactly one isolated control role or authenticate a complete resume."""

    _require(role in ROLES, "control execution role invalid")
    injected = (
        generator_factory is not quote_runner.make_vllm_generator_v2
        or model_resolver is not quote_runner.legacy.resolve_model_snapshot
    )
    _require(
        not injected or _allow_mocked_cpu_test is True,
        "custom model runtime injection is forbidden outside explicit CPU tests",
    )
    execution_backend = (
        "injected_test_non_gate"
        if injected
        else "real_local_model_default_runtime"
    )
    gate_eligible = not injected
    context = _authenticate_execution_context(
        execution_config_path=execution_config_path,
        expected_execution_config_sha256=expected_execution_config_sha256,
    )
    output_manifest = Path(output_manifest_path).absolute()
    output_records = _role_record_path(output_manifest)
    existing_manifest = output_manifest.exists()
    existing_records = output_records.exists()
    if existing_manifest or existing_records:
        _require(
            resume
            and existing_manifest
            and existing_records
            and isinstance(expected_resume_manifest_sha256, str),
            "existing role output may resume only from a complete manifest/record pair",
        )
        manifest, records, resolved = _validate_role_manifest(
            context=context,
            manifest_path=output_manifest,
            expected_manifest_sha256=str(expected_resume_manifest_sha256),
            expected_role=role,
        )
        return {
            "status": "authenticated_complete_role_resume",
            "role": role,
            "manifest_path": str(resolved),
            "manifest_sha256": sha256_file(resolved),
            "record_count": len(records),
            "manifest": manifest,
        }
    _require(
        not resume and expected_resume_manifest_sha256 is None,
        "resume requested but no complete role output exists",
    )
    _guard_output(output_manifest, label=f"{role} role manifest")
    _guard_output(output_records, label=f"{role} role records")

    runtime_config = _mapping(context["runtime_config"], "runtime config")
    role_spec = _mapping(runtime_config["roles"][role], f"{role} model spec")
    try:
        model_path, raw_inventory = model_resolver(role_spec, verify_contents=True)
    except quote_runner.legacy.AnnotationRunnerError as error:
        raise ControlExecutorError(str(error)) from error
    inventory = _validate_model_inventory(
        inventory=_mapping(raw_inventory, "resolved model inventory"),
        role_spec=role_spec,
        production_backend=gate_eligible,
    )
    generate, runtime, tokenizer = generator_factory(
        model_path=Path(model_path),
        model_spec=role_spec,
        generation_config=_mapping(runtime_config["generation"], "generation config"),
    )
    source_rows = _role_source_rows(context, role)
    manifest_sha = sha256_file(context["model_input_manifest_path"])
    adapted_by_id = {
        str(row["packet_id_sha256"]): adapt_control_input_to_runner_packet(
            row, model_input_manifest_sha256=manifest_sha
        )
        for row in source_rows
    }
    visible_payload_sha_by_id = {
        str(row["packet_id_sha256"]): _assert_model_visible_payload_matches_seal(
            source=row,
            packet=adapted_by_id[str(row["packet_id_sha256"])],
            codebook=context["codebook"],
            role=role,
        )
        for row in source_rows
    }
    started = time.perf_counter()
    runner_records: list[dict[str, Any]] = []
    if role in PRIMARY_ROLES:
        completion_rows = [row for row in source_rows if row["pass"] == "completion"]
        novelty_rows = [row for row in source_rows if row["pass"] == "novelty"]
        runner_records.extend(
            quote_runner.annotate_completion_packets(
                packets=[adapted_by_id[str(row["packet_id_sha256"])] for row in completion_rows],
                codebook=context["codebook"],
                role=role,
                generate=generate,
                tokenizer=tokenizer,
                seed=int(role_spec["seed"]),
                chat_template_kwargs=role_spec["chat_template_kwargs"],
            )
        )
        runner_records.extend(
            quote_runner.annotate_novelty_packets(
                packets=[adapted_by_id[str(row["packet_id_sha256"])] for row in novelty_rows],
                codebook=context["codebook"],
                role=role,
                generate=generate,
                tokenizer=tokenizer,
                seed=int(role_spec["seed"]),
                chat_template_kwargs=role_spec["chat_template_kwargs"],
            )
        )
    else:
        candidates = {
            str(row["packet_id_sha256"]): (
                row["payload"]["candidate_annotations"][0],
                row["payload"]["candidate_annotations"][1],
            )
            for row in source_rows
        }
        runner_records.extend(
            quote_runner.annotate_completion_packets(
                packets=[adapted_by_id[str(row["packet_id_sha256"])] for row in source_rows],
                codebook=context["codebook"],
                role=role,
                generate=generate,
                tokenizer=tokenizer,
                seed=int(role_spec["seed"]),
                chat_template_kwargs=role_spec["chat_template_kwargs"],
                candidate_records_by_packet=candidates,
            )
        )
    generation_seconds = time.perf_counter() - started
    by_id = {str(item["packet_id_sha256"]): item for item in runner_records}
    _require(
        len(by_id) == len(runner_records) == ROLE_RECORD_COUNTS[role],
        f"{role} runner output coverage changed or duplicated",
    )
    role_records: list[dict[str, Any]] = []
    for source in source_rows:
        packet_id = str(source["packet_id_sha256"])
        _require(packet_id in by_id, f"{role} runner output omitted a sealed control")
        result = _result_with_provenance(
            result=by_id[packet_id],
            role=role,
            role_spec=role_spec,
            model_inventory=inventory,
            context=context,
        )
        prompt_sha, candidate_sha = _expected_prompt_bindings(
            source=source,
            packet=adapted_by_id[packet_id],
            codebook=context["codebook"],
            role=role,
        )
        _validate_result_record(
            result=result,
            role=role,
            control_pass=str(source["pass"]),
            packet_id=packet_id,
            source_id=str(adapted_by_id[packet_id]["source_id_sha256"]),
            model_identity_sha256=str(inventory["model_identity_sha256"]),
            expected_prompt_messages_sha256=prompt_sha,
            expected_candidate_order_sha256=candidate_sha,
            role_spec=role_spec,
            context=context,
        )
        role_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": ROLE_RECORD_KIND,
                "role": role,
                "pass": source["pass"],
                "packet_id_sha256": packet_id,
                "input_payload_sha256": sha256_bytes(
                    canonical_json_bytes(source["payload"])
                ),
                "model_visible_payload_sha256": visible_payload_sha_by_id[packet_id],
                "runner_packet_sha256": sha256_bytes(
                    canonical_json_bytes(adapted_by_id[packet_id])
                ),
                "result": result,
            }
        )
    count, record_sha = _write_jsonl_atomic(
        output_records, role_records, label=f"{role} role records"
    )
    model_invocations = sum(
        bool(_mapping(item["result"]["generation"], "generation")["model_invoked"])
        for item in role_records
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": ROLE_MANIFEST_KIND,
        "status": "sealed_control_role_generation_complete",
        "role": role,
        "scope": {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "target_annotations_present": False,
            "expectations_opened_during_generation": False,
            "model_visible_payload_byte_equal_to_sealed_payload": True,
        },
        "passes": list(ROLE_PASS_CONTRACT[role]),
        "counts": {
            "input_records": len(source_rows),
            "output_records": count,
            "model_invocations": model_invocations,
        },
        "records": {
            "path": output_records.name,
            "sha256": record_sha,
            "count": count,
        },
        "inputs": _manifest_input_bindings(context),
        "model": inventory,
        "runtime": {
            **copy.deepcopy(dict(_mapping(runtime, "model runtime"))),
            "control_generation_seconds": generation_seconds,
        },
        "fingerprints": _role_fingerprints(
            context=context,
            role=role,
            role_spec=role_spec,
            model_identity_sha256=str(inventory["model_identity_sha256"]),
        ),
        "execution_backend": execution_backend,
        "gate_eligible": gate_eligible,
    }
    manifest_sha = _write_json_atomic(
        output_manifest, manifest, label=f"{role} role manifest"
    )
    return {
        "status": "sealed_control_role_generation_complete",
        "role": role,
        "manifest_path": str(output_manifest),
        "manifest_sha256": manifest_sha,
        "record_count": count,
        "model_identity_sha256": inventory["model_identity_sha256"],
    }


def _combined_rows(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    primary_records: Sequence[Mapping[str, Any]],
    adjudicator_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    primary_by_id = {
        str(item["packet_id_sha256"]): item for item in primary_records
    }
    adjudicator_by_id = {
        str(item["packet_id_sha256"]): item for item in adjudicator_records
    }
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        packet_id = str(source["packet_id_sha256"])
        pass_name = str(source["pass"])
        role_record = (
            adjudicator_by_id.get(packet_id)
            if pass_name == "adjudication"
            else primary_by_id.get(packet_id)
        )
        _require(role_record is not None, "combined output source role coverage incomplete")
        _require(
            role_record.get("pass") == pass_name,
            "combined output source pass changed",
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": SCORER_OUTPUT_KIND,
                "pass": pass_name,
                "packet_id_sha256": packet_id,
                "result": copy.deepcopy(dict(_mapping(role_record["result"], "role result"))),
            }
        )
    _require(len(rows) == 52, "combined output must contain exactly 52 records")
    return rows


def _combined_manifest(
    *,
    context: Mapping[str, Any],
    primary_role: str,
    output_record_path: Path,
    output_record_sha256: str,
    primary_manifest_path: Path,
    adjudicator_manifest_path: Path,
    primary_manifest: Mapping[str, Any],
    adjudicator_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": COMBINED_MANIFEST_KIND,
        "status": "deterministic_52_record_output_ready_for_pre_answer_lock",
        "primary_role": primary_role,
        "scope": {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "target_annotations_present": False,
            "expectations_joined": False,
        },
        "counts": dict(EXPECTED_COUNTS),
        "records": {
            "path": output_record_path.name,
            "sha256": output_record_sha256,
            "count": 52,
        },
        "inputs": _manifest_input_bindings(context),
        "role_manifests": {
            "primary": {
                "path": str(primary_manifest_path.resolve(strict=True)),
                "sha256": sha256_file(primary_manifest_path),
                "model_identity_sha256": primary_manifest["model"][
                    "model_identity_sha256"
                ],
                "execution_backend": primary_manifest["execution_backend"],
                "gate_eligible": primary_manifest["gate_eligible"],
            },
            "adjudicator": {
                "path": str(adjudicator_manifest_path.resolve(strict=True)),
                "sha256": sha256_file(adjudicator_manifest_path),
                "model_identity_sha256": adjudicator_manifest["model"][
                    "model_identity_sha256"
                ],
                "execution_backend": adjudicator_manifest["execution_backend"],
                "gate_eligible": adjudicator_manifest["gate_eligible"],
            },
        },
    }


def combine_primary_outputs(
    *,
    execution_config_path: Path,
    expected_execution_config_sha256: str,
    independent_a_manifest_path: Path,
    expected_independent_a_manifest_sha256: str,
    independent_b_manifest_path: Path,
    expected_independent_b_manifest_sha256: str,
    adjudicator_manifest_path: Path,
    expected_adjudicator_manifest_sha256: str,
    output_a_records_path: Path,
    output_a_manifest_path: Path,
    output_b_records_path: Path,
    output_b_manifest_path: Path,
) -> dict[str, Any]:
    """Build deterministic A+J and B+J scorer inputs without opening answers."""

    context = _authenticate_execution_context(
        execution_config_path=execution_config_path,
        expected_execution_config_sha256=expected_execution_config_sha256,
    )
    manifest_a, records_a, resolved_a = _validate_role_manifest(
        context=context,
        manifest_path=independent_a_manifest_path,
        expected_manifest_sha256=expected_independent_a_manifest_sha256,
        expected_role="independent_a",
    )
    manifest_b, records_b, resolved_b = _validate_role_manifest(
        context=context,
        manifest_path=independent_b_manifest_path,
        expected_manifest_sha256=expected_independent_b_manifest_sha256,
        expected_role="independent_b",
    )
    manifest_j, records_j, resolved_j = _validate_role_manifest(
        context=context,
        manifest_path=adjudicator_manifest_path,
        expected_manifest_sha256=expected_adjudicator_manifest_sha256,
        expected_role="adjudicator",
    )
    output_paths = [
        Path(output_a_records_path).absolute(),
        Path(output_a_manifest_path).absolute(),
        Path(output_b_records_path).absolute(),
        Path(output_b_manifest_path).absolute(),
    ]
    _require(len(set(output_paths)) == 4, "combined output paths must be distinct")
    for path in output_paths:
        _guard_output(path, label="combined control output")
    source_rows = list(context["model_input_rows"])
    combined_a = _combined_rows(
        source_rows=source_rows,
        primary_records=records_a,
        adjudicator_records=records_j,
    )
    combined_b = _combined_rows(
        source_rows=source_rows,
        primary_records=records_b,
        adjudicator_records=records_j,
    )
    count_a, sha_a = _write_jsonl_atomic(
        output_paths[0], combined_a, label="combined independent A records"
    )
    count_b, sha_b = _write_jsonl_atomic(
        output_paths[2], combined_b, label="combined independent B records"
    )
    _require(count_a == count_b == 52, "combined control output count changed")
    combined_manifest_a = _combined_manifest(
        context=context,
        primary_role="independent_a",
        output_record_path=output_paths[0],
        output_record_sha256=sha_a,
        primary_manifest_path=resolved_a,
        adjudicator_manifest_path=resolved_j,
        primary_manifest=manifest_a,
        adjudicator_manifest=manifest_j,
    )
    combined_manifest_b = _combined_manifest(
        context=context,
        primary_role="independent_b",
        output_record_path=output_paths[2],
        output_record_sha256=sha_b,
        primary_manifest_path=resolved_b,
        adjudicator_manifest_path=resolved_j,
        primary_manifest=manifest_b,
        adjudicator_manifest=manifest_j,
    )
    manifest_sha_a = _write_json_atomic(
        output_paths[1], combined_manifest_a, label="combined independent A manifest"
    )
    manifest_sha_b = _write_json_atomic(
        output_paths[3], combined_manifest_b, label="combined independent B manifest"
    )
    return {
        "status": "two_deterministic_outputs_ready_for_independent_locking",
        "independent_a": {
            "records_path": str(output_paths[0]),
            "records_sha256": sha_a,
            "manifest_path": str(output_paths[1]),
            "manifest_sha256": manifest_sha_a,
        },
        "independent_b": {
            "records_path": str(output_paths[2]),
            "records_sha256": sha_b,
            "manifest_path": str(output_paths[3]),
            "manifest_sha256": manifest_sha_b,
        },
    }


def _load_controls_module() -> Any:
    """Import the lock/scorer only in post-generation protocol commands."""

    import swe_task_state_v4_epistemic_chain_controls_v2 as controls

    _require(
        Path(controls.__file__).resolve(strict=True)
        == CONTROL_BUILDER_PATH.resolve(strict=True),
        "post-generation control builder resolves to different code",
    )
    return controls


def _validate_combined_manifest(
    *,
    context: Mapping[str, Any],
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_primary_role: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path]:
    _require(
        expected_primary_role in PRIMARY_ROLES
        and SHA256_RE.fullmatch(expected_manifest_sha256) is not None,
        "combined manifest role or expected SHA-256 invalid",
    )
    resolved = _guard_existing_repo_file(
        manifest_path, label=f"{expected_primary_role} combined manifest"
    )
    _require(
        sha256_file(resolved) == expected_manifest_sha256,
        f"{expected_primary_role} combined-manifest hash changed",
    )
    manifest = dict(
        _mapping(
            load_json_strict(
                resolved,
                generation_input=False,
                label=f"{expected_primary_role} combined manifest",
            ),
            "combined manifest",
        )
    )
    _require(
        set(manifest)
        == {
            "schema_version",
            "kind",
            "status",
            "primary_role",
            "scope",
            "counts",
            "records",
            "inputs",
            "role_manifests",
        }
        and manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("kind") == COMBINED_MANIFEST_KIND
        and manifest.get("status")
        == "deterministic_52_record_output_ready_for_pre_answer_lock"
        and manifest.get("primary_role") == expected_primary_role
        and manifest.get("counts") == EXPECTED_COUNTS
        and manifest.get("inputs") == _manifest_input_bindings(context),
        "combined manifest identity, counts, or inputs changed",
    )
    _require(
        manifest.get("scope")
        == {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "target_annotations_present": False,
            "expectations_joined": False,
        },
        "combined manifest scope changed",
    )
    roles = _mapping(manifest.get("role_manifests"), "combined role manifests")
    _require(set(roles) == {"primary", "adjudicator"}, "combined role set changed")
    primary_binding = _mapping(roles["primary"], "combined primary role binding")
    adjudicator_binding = _mapping(
        roles["adjudicator"], "combined adjudicator role binding"
    )
    for label, binding in (
        ("primary", primary_binding),
        ("adjudicator", adjudicator_binding),
    ):
        _require(
            set(binding)
            == {
                "path",
                "sha256",
                "model_identity_sha256",
                "execution_backend",
                "gate_eligible",
            }
            and SHA256_RE.fullmatch(str(binding.get("sha256"))) is not None
            and SHA256_RE.fullmatch(str(binding.get("model_identity_sha256")))
            is not None
            and binding.get("execution_backend")
            in {"real_local_model_default_runtime", "injected_test_non_gate"}
            and binding.get("gate_eligible")
            is (
                binding.get("execution_backend")
                == "real_local_model_default_runtime"
            ),
            f"combined {label} role binding invalid",
        )
    primary_manifest, primary_records, primary_path = _validate_role_manifest(
        context=context,
        manifest_path=Path(str(primary_binding["path"])),
        expected_manifest_sha256=str(primary_binding["sha256"]),
        expected_role=expected_primary_role,
    )
    adjudicator_manifest, adjudicator_records, adjudicator_path = (
        _validate_role_manifest(
            context=context,
            manifest_path=Path(str(adjudicator_binding["path"])),
            expected_manifest_sha256=str(adjudicator_binding["sha256"]),
            expected_role="adjudicator",
        )
    )
    _require(
        primary_binding["model_identity_sha256"]
        == primary_manifest["model"]["model_identity_sha256"]
        and primary_binding["execution_backend"]
        == primary_manifest["execution_backend"]
        and primary_binding["gate_eligible"] == primary_manifest["gate_eligible"]
        and adjudicator_binding["model_identity_sha256"]
        == adjudicator_manifest["model"]["model_identity_sha256"]
        and adjudicator_binding["execution_backend"]
        == adjudicator_manifest["execution_backend"]
        and adjudicator_binding["gate_eligible"]
        == adjudicator_manifest["gate_eligible"],
        "combined role model identity changed",
    )
    record_binding = _mapping(manifest.get("records"), "combined records")
    _require(
        set(record_binding) == {"path", "sha256", "count"}
        and record_binding.get("count") == 52
        and SHA256_RE.fullmatch(str(record_binding.get("sha256"))) is not None,
        "combined record binding invalid",
    )
    record_path = _guard_existing_repo_file(
        resolved.parent / str(record_binding["path"]), label="combined records"
    )
    _require(
        sha256_file(record_path) == record_binding["sha256"],
        "combined record hash changed",
    )
    observed = _load_jsonl(
        record_path, generation_input=False, label="combined control records"
    )
    expected = _combined_rows(
        source_rows=context["model_input_rows"],
        primary_records=primary_records,
        adjudicator_records=adjudicator_records,
    )
    _require(
        canonical_json_bytes(observed) == canonical_json_bytes(expected),
        "combined records differ from exact authenticated role records",
    )
    del primary_path, adjudicator_path
    return manifest, observed, resolved, record_path


def lock_both_primary_outputs(
    *,
    execution_config_path: Path,
    expected_execution_config_sha256: str,
    combined_a_manifest_path: Path,
    expected_combined_a_manifest_sha256: str,
    combined_b_manifest_path: Path,
    expected_combined_b_manifest_sha256: str,
    output_a_lock_manifest_path: Path,
    output_b_lock_manifest_path: Path,
    output_dual_lock_receipt_path: Path,
) -> dict[str, Any]:
    """Authenticate and lock both 52-row outputs before any answer join."""

    context = _authenticate_execution_context(
        execution_config_path=execution_config_path,
        expected_execution_config_sha256=expected_execution_config_sha256,
    )
    combined_a, _rows_a, resolved_a, records_a = _validate_combined_manifest(
        context=context,
        manifest_path=combined_a_manifest_path,
        expected_manifest_sha256=expected_combined_a_manifest_sha256,
        expected_primary_role="independent_a",
    )
    combined_b, _rows_b, resolved_b, records_b = _validate_combined_manifest(
        context=context,
        manifest_path=combined_b_manifest_path,
        expected_manifest_sha256=expected_combined_b_manifest_sha256,
        expected_primary_role="independent_b",
    )
    _require(
        combined_a["role_manifests"]["adjudicator"]
        == combined_b["role_manifests"]["adjudicator"],
        "two combined outputs do not bind the same adjudicator role",
    )
    lock_a = Path(output_a_lock_manifest_path).absolute()
    lock_b = Path(output_b_lock_manifest_path).absolute()
    receipt_path = Path(output_dual_lock_receipt_path).absolute()
    _require(len({lock_a, lock_b, receipt_path}) == 3, "dual-lock outputs must differ")
    for path in (lock_a, lock_b, receipt_path):
        _guard_output(path, label="dual-lock output")
    controls = _load_controls_module()
    controls.lock_outputs(
        model_input_manifest_path=context["model_input_manifest_path"],
        output_records_path=records_a,
        output_manifest_path=lock_a,
    )
    controls.lock_outputs(
        model_input_manifest_path=context["model_input_manifest_path"],
        output_records_path=records_b,
        output_manifest_path=lock_b,
    )
    controls._load_locked_outputs(lock_a)
    controls._load_locked_outputs(lock_b)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": DUAL_LOCK_RECEIPT_KIND,
        "status": "both_primary_outputs_locked_before_any_expectation_join",
        "scope": {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "target_annotations_present": False,
            "expectations_opened": False,
        },
        "inputs": _manifest_input_bindings(context),
        "combined_manifests": {
            "independent_a": {
                "path": str(resolved_a),
                "sha256": sha256_file(resolved_a),
            },
            "independent_b": {
                "path": str(resolved_b),
                "sha256": sha256_file(resolved_b),
            },
        },
        "locked_outputs": {
            "independent_a": {
                "path": str(lock_a),
                "sha256": sha256_file(lock_a),
            },
            "independent_b": {
                "path": str(lock_b),
                "sha256": sha256_file(lock_b),
            },
        },
        "decision_roles": {
            "independent_a": combined_a["role_manifests"]["primary"],
            "independent_b": combined_b["role_manifests"]["primary"],
            "adjudicator": combined_a["role_manifests"]["adjudicator"],
        },
    }
    receipt_sha = _write_json_atomic(
        receipt_path, receipt, label="dual-lock receipt"
    )
    return {
        "status": receipt["status"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "lock_a_sha256": sha256_file(lock_a),
        "lock_b_sha256": sha256_file(lock_b),
    }


def _validate_dual_lock_receipt(
    *,
    context: Mapping[str, Any],
    receipt_path: Path,
    expected_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path], Path]:
    _require(
        SHA256_RE.fullmatch(expected_receipt_sha256) is not None,
        "expected dual-lock receipt SHA-256 is required",
    )
    resolved = _guard_existing_repo_file(receipt_path, label="dual-lock receipt")
    _require(
        sha256_file(resolved) == expected_receipt_sha256,
        "dual-lock receipt hash changed",
    )
    receipt = dict(
        _mapping(
            load_json_strict(
                resolved, generation_input=False, label="dual-lock receipt"
            ),
            "dual-lock receipt",
        )
    )
    _require(
        set(receipt)
        == {
            "schema_version",
            "kind",
            "status",
            "scope",
            "inputs",
            "combined_manifests",
            "locked_outputs",
            "decision_roles",
        }
        and receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("kind") == DUAL_LOCK_RECEIPT_KIND
        and receipt.get("status")
        == "both_primary_outputs_locked_before_any_expectation_join"
        and receipt.get("inputs") == _manifest_input_bindings(context)
        and receipt.get("scope")
        == {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "target_annotations_present": False,
            "expectations_opened": False,
        },
        "dual-lock receipt identity, scope, or inputs changed",
    )
    combined = _mapping(receipt["combined_manifests"], "dual combined manifests")
    _require(set(combined) == set(PRIMARY_ROLES), "dual combined role set changed")
    validated_combined: dict[str, dict[str, Any]] = {}
    combined_record_paths: dict[str, Path] = {}
    for role in PRIMARY_ROLES:
        binding = _mapping(combined[role], f"{role} combined binding")
        _require(
            set(binding) == {"path", "sha256"}
            and SHA256_RE.fullmatch(str(binding.get("sha256"))) is not None,
            f"{role} combined receipt binding invalid",
        )
        manifest, _rows, _manifest_path, record_path = _validate_combined_manifest(
            context=context,
            manifest_path=Path(str(binding["path"])),
            expected_manifest_sha256=str(binding["sha256"]),
            expected_primary_role=role,
        )
        validated_combined[role] = manifest
        combined_record_paths[role] = record_path
    _require(
        receipt.get("decision_roles")
        == {
            "independent_a": validated_combined["independent_a"][
                "role_manifests"
            ]["primary"],
            "independent_b": validated_combined["independent_b"][
                "role_manifests"
            ]["primary"],
            "adjudicator": validated_combined["independent_a"][
                "role_manifests"
            ]["adjudicator"],
        }
        and validated_combined["independent_a"]["role_manifests"]["adjudicator"]
        == validated_combined["independent_b"]["role_manifests"]["adjudicator"],
        "dual-lock decision-role bindings changed",
    )
    locks = _mapping(receipt["locked_outputs"], "dual locked outputs")
    _require(set(locks) == set(PRIMARY_ROLES), "dual locked-output set changed")
    controls = _load_controls_module()
    resolved_locks: dict[str, Path] = {}
    for role in PRIMARY_ROLES:
        binding = _mapping(locks[role], f"{role} locked output")
        _require(
            set(binding) == {"path", "sha256"}
            and SHA256_RE.fullmatch(str(binding.get("sha256"))) is not None,
            f"{role} lock receipt binding invalid",
        )
        lock_path = _guard_existing_repo_file(
            Path(str(binding["path"])), label=f"{role} locked output"
        )
        _require(
            sha256_file(lock_path) == binding["sha256"],
            f"{role} locked-output hash changed",
        )
        lock_value = _mapping(
            load_json_strict(
                lock_path, generation_input=False, label=f"{role} locked output"
            ),
            f"{role} locked output",
        )
        lock_records = _mapping(lock_value.get("records"), f"{role} locked records")
        lock_model = _mapping(
            lock_value.get("model_inputs"), f"{role} locked model inputs"
        )
        _require(
            Path(str(lock_records.get("path"))).resolve(strict=True)
            == combined_record_paths[role].resolve(strict=True)
            and lock_records.get("sha256")
            == sha256_file(combined_record_paths[role])
            and Path(str(lock_model.get("path"))).resolve(strict=True)
            == Path(context["model_input_manifest_path"]).resolve(strict=True)
            and lock_model.get("sha256")
            == sha256_file(context["model_input_manifest_path"]),
            f"{role} lock does not bind its authenticated combined output",
        )
        controls._load_locked_outputs(lock_path)
        resolved_locks[role] = lock_path
    return receipt, resolved_locks, resolved


def score_both_and_finalize(
    *,
    execution_config_path: Path,
    expected_execution_config_sha256: str,
    dual_lock_receipt_path: Path,
    expected_dual_lock_receipt_sha256: str,
    expectation_manifest_path: Path,
    expected_expectation_manifest_sha256: str,
    expected_decoder_addendum_sha256: str,
    output_a_report_path: Path,
    output_b_report_path: Path,
    output_final_receipt_path: Path,
) -> dict[str, Any]:
    """Open answers only after authenticating both locks, then score both lanes."""

    context = _authenticate_execution_context(
        execution_config_path=execution_config_path,
        expected_execution_config_sha256=expected_execution_config_sha256,
    )
    dual, locks, resolved_dual = _validate_dual_lock_receipt(
        context=context,
        receipt_path=dual_lock_receipt_path,
        expected_receipt_sha256=expected_dual_lock_receipt_sha256,
    )
    _require(
        SHA256_RE.fullmatch(expected_decoder_addendum_sha256) is not None
        and sha256_file(context["decoder_addendum_path"])
        == expected_decoder_addendum_sha256,
        "decoder V2 addendum expected hash changed",
    )
    _require(
        SHA256_RE.fullmatch(expected_expectation_manifest_sha256) is not None,
        "expected expectation-manifest SHA-256 is required",
    )
    declared_expectation = _mapping(
        context["runtime_config"]["inputs"]["sealed_control_expectations"],
        "runtime expectation binding",
    )
    declared_path = Path(str(declared_expectation["path"]))
    if not declared_path.is_absolute():
        declared_path = ROOT / declared_path
    supplied_logical = Path(expectation_manifest_path).absolute()
    _require(
        supplied_logical == declared_path.absolute()
        and expected_expectation_manifest_sha256 == declared_expectation["sha256"],
        "post-lock expectation input differs from the prospectively frozen binding",
    )
    # This is the first expectation-artifact filesystem access in the protocol.
    expectation_path = _guard_existing_repo_file(
        supplied_logical, label="post-dual-lock expectation manifest"
    )
    _require(
        sha256_file(expectation_path) == expected_expectation_manifest_sha256,
        "post-dual-lock expectation manifest hash changed",
    )
    report_a_path = Path(output_a_report_path).absolute()
    report_b_path = Path(output_b_report_path).absolute()
    final_path = Path(output_final_receipt_path).absolute()
    _require(
        len({report_a_path, report_b_path, final_path}) == 3,
        "final score output paths must be distinct",
    )
    for path in (report_a_path, report_b_path, final_path):
        _guard_output(path, label="final control gate output")
    controls = _load_controls_module()
    report_a = controls.score_locked_outputs(
        locked_output_manifest_path=locks["independent_a"],
        expectation_manifest_path=expectation_path,
    )
    report_b = controls.score_locked_outputs(
        locked_output_manifest_path=locks["independent_b"],
        expectation_manifest_path=expectation_path,
    )
    report_a_sha = _write_json_atomic(
        report_a_path, report_a, label="independent A control report"
    )
    report_b_sha = _write_json_atomic(
        report_b_path, report_b, label="independent B control report"
    )
    decision_roles = _mapping(dual["decision_roles"], "decision roles")
    real_runtime = _all_decision_roles_gate_eligible(decision_roles)
    passed = _final_gate_passed(
        report_a=report_a, report_b=report_b, decision_roles=decision_roles
    )
    final = {
        "schema_version": SCHEMA_VERSION,
        "kind": FINAL_GATE_RECEIPT_KIND,
        "status": (
            "passed_visible_semantic_annotation_interface_readiness_controls"
            if passed
            else (
                "mocked_cpu_test_protocol_only_no_readiness_claim"
                if not real_runtime
                else "failed_visible_semantic_annotation_interface_readiness_controls"
            )
        ),
        "scope": {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "target_annotations_present": False,
            "private_chain_of_thought_recovery_established": False,
            "latent_sentence_or_concept_chain_recovery_established": False,
            "emotion_affect_confidence_doubt_or_stress_recovery_established": False,
            "passing_establishes_annotation_interface_readiness_only": True,
            "all_decision_roles_bound_real_model_runtime": real_runtime,
        },
        "inputs": _manifest_input_bindings(context),
        "dual_lock_receipt": {
            "path": str(resolved_dual),
            "sha256": sha256_file(resolved_dual),
        },
        "decision_roles": dual["decision_roles"],
        "expectation_manifest": {
            "path": str(expectation_path),
            "sha256": sha256_file(expectation_path),
        },
        "decoder_v2_addendum": {
            "path": str(Path(context["decoder_addendum_path"]).resolve(strict=True)),
            "sha256": sha256_file(context["decoder_addendum_path"]),
            "canonical_sha256": sha256_bytes(
                canonical_json_bytes(context["decoder_addendum"])
            ),
        },
        "locked_outputs": copy.deepcopy(dict(dual["locked_outputs"])),
        "reports": {
            "independent_a": {
                "path": str(report_a_path),
                "sha256": report_a_sha,
                "status": report_a["status"],
                "gates": report_a["gates"],
            },
            "independent_b": {
                "path": str(report_b_path),
                "sha256": report_b_sha,
                "status": report_b["status"],
                "gates": report_b["gates"],
            },
        },
    }
    final_sha = _write_json_atomic(final_path, final, label="final control gate receipt")
    return {
        "status": final["status"],
        "final_receipt_path": str(final_path),
        "final_receipt_sha256": final_sha,
        "report_a_sha256": report_a_sha,
        "report_b_sha256": report_b_sha,
    }


def _all_decision_roles_gate_eligible(
    decision_roles: Mapping[str, Any],
) -> bool:
    _require(
        set(decision_roles) == {"independent_a", "independent_b", "adjudicator"},
        "final decision-role set changed",
    )
    return all(
        _mapping(binding, f"{role} decision role").get("execution_backend")
        == "real_local_model_default_runtime"
        and _mapping(binding, f"{role} decision role").get("gate_eligible") is True
        for role, binding in decision_roles.items()
    )


def _final_gate_passed(
    *,
    report_a: Mapping[str, Any],
    report_b: Mapping[str, Any],
    decision_roles: Mapping[str, Any],
) -> bool:
    return (
        report_a.get("status") == report_b.get("status") == "passed"
        and _all_decision_roles_gate_eligible(decision_roles)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze-helper",
        help="print a reviewable exact execution config; never generate",
    )
    freeze.add_argument("--runtime-config", required=True, type=Path)
    freeze.add_argument("--model-input-manifest", required=True, type=Path)

    run = subparsers.add_parser(
        "run-role", help="run exactly one isolated real-model control role"
    )
    run.add_argument("--execution-config", required=True, type=Path)
    run.add_argument("--execution-config-sha256", required=True)
    run.add_argument("--role", required=True, choices=ROLES)
    run.add_argument("--output-manifest", required=True, type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--resume-manifest-sha256")

    combine = subparsers.add_parser(
        "combine", help="build A+J and B+J 52-row pre-lock scorer inputs"
    )
    combine.add_argument("--execution-config", required=True, type=Path)
    combine.add_argument("--execution-config-sha256", required=True)
    combine.add_argument("--independent-a-manifest", required=True, type=Path)
    combine.add_argument("--independent-a-manifest-sha256", required=True)
    combine.add_argument("--independent-b-manifest", required=True, type=Path)
    combine.add_argument("--independent-b-manifest-sha256", required=True)
    combine.add_argument("--adjudicator-manifest", required=True, type=Path)
    combine.add_argument("--adjudicator-manifest-sha256", required=True)
    combine.add_argument("--output-a-records", required=True, type=Path)
    combine.add_argument("--output-a-manifest", required=True, type=Path)
    combine.add_argument("--output-b-records", required=True, type=Path)
    combine.add_argument("--output-b-manifest", required=True, type=Path)

    lock_both = subparsers.add_parser(
        "lock-both",
        help="lock both primary 52-row outputs before any expectation join",
    )
    lock_both.add_argument("--execution-config", required=True, type=Path)
    lock_both.add_argument("--execution-config-sha256", required=True)
    lock_both.add_argument("--combined-a-manifest", required=True, type=Path)
    lock_both.add_argument("--combined-a-manifest-sha256", required=True)
    lock_both.add_argument("--combined-b-manifest", required=True, type=Path)
    lock_both.add_argument("--combined-b-manifest-sha256", required=True)
    lock_both.add_argument("--output-a-lock-manifest", required=True, type=Path)
    lock_both.add_argument("--output-b-lock-manifest", required=True, type=Path)
    lock_both.add_argument("--output-dual-lock-receipt", required=True, type=Path)

    score_both = subparsers.add_parser(
        "score-both",
        help="after authenticating the dual lock, score both and freeze a gate receipt",
    )
    score_both.add_argument("--execution-config", required=True, type=Path)
    score_both.add_argument("--execution-config-sha256", required=True)
    score_both.add_argument("--dual-lock-receipt", required=True, type=Path)
    score_both.add_argument("--dual-lock-receipt-sha256", required=True)
    score_both.add_argument("--expectation-manifest", required=True, type=Path)
    score_both.add_argument("--expectation-manifest-sha256", required=True)
    score_both.add_argument("--decoder-addendum-sha256", required=True)
    score_both.add_argument("--output-a-report", required=True, type=Path)
    score_both.add_argument("--output-b-report", required=True, type=Path)
    score_both.add_argument("--output-final-receipt", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze-helper":
        result = freeze_execution_config(
            runtime_config_path=args.runtime_config,
            model_input_manifest_path=args.model_input_manifest,
        )
    elif args.command == "run-role":
        result = run_role(
            execution_config_path=args.execution_config,
            expected_execution_config_sha256=args.execution_config_sha256,
            role=args.role,
            output_manifest_path=args.output_manifest,
            resume=args.resume,
            expected_resume_manifest_sha256=args.resume_manifest_sha256,
        )
    elif args.command == "combine":
        result = combine_primary_outputs(
            execution_config_path=args.execution_config,
            expected_execution_config_sha256=args.execution_config_sha256,
            independent_a_manifest_path=args.independent_a_manifest,
            expected_independent_a_manifest_sha256=args.independent_a_manifest_sha256,
            independent_b_manifest_path=args.independent_b_manifest,
            expected_independent_b_manifest_sha256=args.independent_b_manifest_sha256,
            adjudicator_manifest_path=args.adjudicator_manifest,
            expected_adjudicator_manifest_sha256=args.adjudicator_manifest_sha256,
            output_a_records_path=args.output_a_records,
            output_a_manifest_path=args.output_a_manifest,
            output_b_records_path=args.output_b_records,
            output_b_manifest_path=args.output_b_manifest,
        )
    elif args.command == "lock-both":
        result = lock_both_primary_outputs(
            execution_config_path=args.execution_config,
            expected_execution_config_sha256=args.execution_config_sha256,
            combined_a_manifest_path=args.combined_a_manifest,
            expected_combined_a_manifest_sha256=args.combined_a_manifest_sha256,
            combined_b_manifest_path=args.combined_b_manifest,
            expected_combined_b_manifest_sha256=args.combined_b_manifest_sha256,
            output_a_lock_manifest_path=args.output_a_lock_manifest,
            output_b_lock_manifest_path=args.output_b_lock_manifest,
            output_dual_lock_receipt_path=args.output_dual_lock_receipt,
        )
    else:
        result = score_both_and_finalize(
            execution_config_path=args.execution_config,
            expected_execution_config_sha256=args.execution_config_sha256,
            dual_lock_receipt_path=args.dual_lock_receipt,
            expected_dual_lock_receipt_sha256=args.dual_lock_receipt_sha256,
            expectation_manifest_path=args.expectation_manifest,
            expected_expectation_manifest_sha256=args.expectation_manifest_sha256,
            expected_decoder_addendum_sha256=args.decoder_addendum_sha256,
            output_a_report_path=args.output_a_report,
            output_b_report_path=args.output_b_report,
            output_final_receipt_path=args.output_final_receipt,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMBINED_MANIFEST_KIND",
    "ControlExecutorError",
    "EXECUTOR_KIND",
    "FROZEN_STATUS",
    "ROLE_MANIFEST_KIND",
    "ROLE_RECORD_KIND",
    "adapt_control_input_to_runner_packet",
    "combine_primary_outputs",
    "freeze_execution_config",
    "lock_both_primary_outputs",
    "load_sealed_model_inputs",
    "run_role",
    "score_both_and_finalize",
    "validate_execution_config",
]
