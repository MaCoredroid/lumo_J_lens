#!/usr/bin/env python3
"""Build prospective V3 visible-semantic control inputs in memory.

This generation-side draft accepts only authored visible text, authenticates
runner packets and deterministic candidate catalogs, and renders prospective
model requests.  It has no filesystem write path and no dependency on any
scoring/key module.  Returned values are drafts, never sealed artifacts.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import swe_task_state_v4_epistemic_chain_candidate_catalog_v3 as catalog  # noqa: E402


runner = catalog.runner
SCHEMA_VERSION = 1
INTERFACE_VERSION = 3
CONFIG_PATH = (
    ROOT
    / "configs"
    / "swe_task_state_v4_epistemic_chain_control_inputs_draft_v3.json"
)
CONFIG_KIND = "swe_task_state_v4_epistemic_chain_control_inputs_draft_v3"
MANIFEST_KIND = (
    "swe_task_state_v4_epistemic_chain_control_input_manifest_draft_v3"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMPLETION_ID_RE = re.compile(r"^C(?:0[1-9]|[12][0-9]|3[0-2])$")
NOVELTY_ID_RE = re.compile(r"^V0[1-8]$")
CLAIM_SCOPE = {
    "development_controls_only": True,
    "reserved_validation_closed": True,
    "reserved_validation_accessed": False,
    "visible_semantic_evidence_hypothesis_action_only": True,
    "cot_recovery_claimed": False,
    "cot_like_recovery_claimed": False,
    "private_cot_recovery_claimed": False,
    "affect_emotion_confidence_doubt_or_stress_claimed": False,
    "model_run_performed": False,
}


class ControlInputDraftError(RuntimeError):
    """Raised when prospective input authorship or authentication fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlInputDraftError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{name} must be an array",
    )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ControlInputDraftError(f"duplicate JSON key: {key}")
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


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_file(binding: Mapping[str, Any], *, name: str) -> Path:
    _require(
        set(binding) == {"path", "sha256"}
        and isinstance(binding.get("path"), str)
        and SHA256_RE.fullmatch(str(binding.get("sha256"))) is not None,
        f"{name} binding invalid",
    )
    logical = ROOT / str(binding["path"])
    try:
        path = logical.resolve(strict=True)
        path.relative_to(ROOT.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as error:
        raise ControlInputDraftError(
            f"{name} binding must resolve inside repository"
        ) from error
    _require(path.is_file(), f"{name} binding is not a file")
    _require(sha256_file(path) == binding["sha256"], f"{name} hash changed")
    return path


def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlInputDraftError(f"cannot load strict JSON {path}: {error}") from error


def validate_input_config(value: Any) -> dict[str, Any]:
    config = dict(_mapping(value, "input draft config"))
    _require(
        set(config)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "status",
            "scope",
            "identity",
            "counts",
            "bindings",
            "completion_inputs",
            "novelty_inputs",
        },
        "input draft config fields invalid",
    )
    _require(
        config["schema_version"] == SCHEMA_VERSION
        and config["interface_version"] == INTERFACE_VERSION
        and config["kind"] == CONFIG_KIND
        and config["status"] == "prospective_draft_not_sealed_not_run",
        "input draft identity invalid",
    )
    _require(dict(_mapping(config["scope"], "input scope")) == CLAIM_SCOPE,
             "input draft claim scope invalid")

    identity = dict(_mapping(config["identity"], "input identity"))
    _require(
        set(identity)
        == {
            "suite_id",
            "packet_id_domain",
            "packet_salt",
            "source_id_domain",
            "source_salt",
            "shard_seed",
            "completion_nonce_base",
            "novelty_nonce_base",
        },
        "input identity fields invalid",
    )
    for key in (
        "suite_id",
        "packet_id_domain",
        "packet_salt",
        "source_id_domain",
        "source_salt",
        "shard_seed",
    ):
        _require(isinstance(identity[key], str) and len(identity[key]) >= 16,
                 f"input identity {key} invalid")
    for key in ("completion_nonce_base", "novelty_nonce_base"):
        _require(
            isinstance(identity[key], int)
            and not isinstance(identity[key], bool)
            and identity[key] > 10_000,
            f"input identity {key} invalid",
        )
    _require(
        identity["completion_nonce_base"] != identity["novelty_nonce_base"],
        "input nonce bases must differ",
    )
    _require(
        dict(_mapping(config["counts"], "input counts"))
        == {"completion": 32, "novelty": 8, "total": 40},
        "input counts changed",
    )
    bindings = dict(_mapping(config["bindings"], "input bindings"))
    _require(
        set(bindings)
        == {
            "input_builder",
            "candidate_catalog_config",
            "candidate_catalog_builder",
            "runner_v3",
            "codebook_v2",
        },
        "input binding names invalid",
    )
    for name, raw in bindings.items():
        path = _repo_file(_mapping(raw, name), name=name)
        expected_names = {
            "input_builder": Path(__file__).name,
            "candidate_catalog_config": catalog.CONFIG_PATH.name,
            "candidate_catalog_builder": Path(catalog.__file__).name,
            "runner_v3": Path(runner.__file__).name,
            "codebook_v2": "swe_task_state_v4_epistemic_chain_codebook_v2.json",
        }
        _require(path.name == expected_names[name], f"{name} binds wrong file")

    completion_rows = [
        dict(_mapping(item, "completion input"))
        for item in _sequence(config["completion_inputs"], "completion inputs")
    ]
    novelty_rows = [
        dict(_mapping(item, "novelty input"))
        for item in _sequence(config["novelty_inputs"], "novelty inputs")
    ]
    _require(len(completion_rows) == 32 and len(novelty_rows) == 8,
             "input row counts changed")
    completion_ids: list[str] = []
    completion_texts: list[str] = []
    for index, row in enumerate(completion_rows, start=1):
        _require(set(row) == {"id", "assistant_text"},
                 "completion input fields invalid")
        control_id = row.get("id")
        text = row.get("assistant_text")
        _require(
            isinstance(control_id, str)
            and COMPLETION_ID_RE.fullmatch(control_id) is not None
            and control_id == f"C{index:02d}",
            "completion input IDs changed",
        )
        _require(isinstance(text, str), f"{control_id} text must be Unicode")
        _require((control_id == "C32") == (text == ""),
                 "only C32 may be exactly empty")
        completion_ids.append(control_id)
        completion_texts.append(text)
    _require(len(set(completion_ids)) == 32, "completion IDs duplicate")
    _require(len(set(completion_texts)) == 32, "completion texts duplicate")

    novelty_ids: list[str] = []
    novelty_pairs: list[tuple[str, str]] = []
    for index, row in enumerate(novelty_rows, start=1):
        _require(set(row) == {"id", "visible_prefix", "locked_hypothesis"},
                 "novelty input fields invalid")
        control_id = row.get("id")
        prefix = row.get("visible_prefix")
        hypothesis = row.get("locked_hypothesis")
        _require(
            isinstance(control_id, str)
            and NOVELTY_ID_RE.fullmatch(control_id) is not None
            and control_id == f"V{index:02d}",
            "novelty input IDs changed",
        )
        _require(
            isinstance(prefix, str)
            and bool(prefix)
            and isinstance(hypothesis, str)
            and bool(hypothesis),
            f"{control_id} novelty text invalid",
        )
        novelty_ids.append(control_id)
        novelty_pairs.append((prefix, hypothesis))
    _require(len(set(novelty_ids)) == 8, "novelty IDs duplicate")
    _require(len(set(novelty_pairs)) == 8, "novelty pairs duplicate")
    return config


def load_input_config() -> tuple[dict[str, Any], str]:
    value = validate_input_config(_load_json(CONFIG_PATH))
    return value, sha256_file(CONFIG_PATH)


def _blind_shards(*, packet_id: str, seed: str) -> dict[str, int]:
    values = []
    for lane in ("independent_a", "independent_b"):
        digest = sha256_bytes(
            canonical_json_bytes(
                {
                    "domain": "visible-semantic-v3-draft-blind-shard",
                    "seed": seed,
                    "lane": lane,
                    "packet_id_sha256": packet_id,
                }
            )
        )
        values.append(int(digest[:16], 16) % 8)
    if values[0] == values[1]:
        values[1] = (values[1] + 1) % 8
    return {"independent_a": values[0], "independent_b": values[1]}


def _packet_id(
    *, config: Mapping[str, Any], pass_name: str, control_id: str, nonce: int,
    payload_sha256: str
) -> str:
    identity = _mapping(config["identity"], "input identity")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": identity["packet_id_domain"],
                "salt": identity["packet_salt"],
                "pass": pass_name,
                "control_id": control_id,
                "nonce": nonce,
                "payload_sha256": payload_sha256,
            }
        )
    )


def _source_id(
    *, config: Mapping[str, Any], packet_id: str, pass_name: str,
    payload_sha256: str, config_sha256: str
) -> str:
    identity = _mapping(config["identity"], "input identity")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": identity["source_id_domain"],
                "salt": identity["source_salt"],
                "pass": pass_name,
                "packet_id_sha256": packet_id,
                "payload_sha256": payload_sha256,
                "input_config_sha256": config_sha256,
            }
        )
    )


def _completion_packet(
    *, config: Mapping[str, Any], config_sha256: str, row: Mapping[str, Any],
    ordinal: int
) -> dict[str, Any]:
    text = str(row["assistant_text"])
    payload_sha = sha256_bytes(canonical_json_bytes({"assistant_text": text}))
    nonce = int(config["identity"]["completion_nonce_base"]) + ordinal
    packet_id = _packet_id(
        config=config,
        pass_name="completion",
        control_id=str(row["id"]),
        nonce=nonce,
        payload_sha256=payload_sha,
    )
    source_id = _source_id(
        config=config,
        packet_id=packet_id,
        pass_name="completion",
        payload_sha256=payload_sha,
        config_sha256=config_sha256,
    )
    packet = {
        "schema_version": SCHEMA_VERSION,
        "kind": runner.v2.legacy.COMPLETION_PACKET_KIND,
        "annotation_pass": "completion_chain",
        "packet_id_sha256": packet_id,
        "source_id_sha256": source_id,
        "blind_shards": _blind_shards(
            packet_id=packet_id, seed=str(config["identity"]["shard_seed"])
        ),
        "materialized_assistant_text": {
            "char_start": 0,
            "char_end": len(text),
            "sha256": sha256_text(text),
            "text": text,
        },
        "authenticated_boundaries": {
            "draft_input_config_sha256": config_sha256,
            "draft_payload_sha256": payload_sha,
            "draft_nonce": nonce,
        },
        "annotator_visibility": {
            "assistant_tool_arguments_present": False,
            "complete_prefix_text_present": False,
            "model_features_present": False,
            "repository_or_task_identity_present": False,
            "tool_results_present": False,
        },
    }
    return runner.v2.legacy.validate_packet(
        packet, annotation_pass="completion_chain"
    )


def _novelty_packet(
    *, config: Mapping[str, Any], config_sha256: str, row: Mapping[str, Any],
    ordinal: int
) -> dict[str, Any]:
    prefix = str(row["visible_prefix"])
    hypothesis = str(row["locked_hypothesis"])
    payload = {"visible_prefix": prefix, "locked_hypothesis": hypothesis}
    payload_sha = sha256_bytes(canonical_json_bytes(payload))
    nonce = int(config["identity"]["novelty_nonce_base"]) + ordinal
    packet_id = _packet_id(
        config=config,
        pass_name="novelty",
        control_id=str(row["id"]),
        nonce=nonce,
        payload_sha256=payload_sha,
    )
    source_id = _source_id(
        config=config,
        packet_id=packet_id,
        pass_name="novelty",
        payload_sha256=payload_sha,
        config_sha256=config_sha256,
    )
    completion_sha = sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "visible-semantic-v3-draft-locked-completion-cobalt",
                "packet_id_sha256": packet_id,
                "nonce": nonce,
                "locked_hypothesis_sha256": sha256_text(hypothesis),
            }
        )
    )
    packet = {
        "schema_version": SCHEMA_VERSION,
        "kind": runner.v2.legacy.PREFIX_PACKET_KIND,
        "annotation_pass": "prefix_novelty",
        "packet_id_sha256": packet_id,
        "source_id_sha256": source_id,
        "blind_shards": _blind_shards(
            packet_id=packet_id, seed=str(config["identity"]["shard_seed"])
        ),
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
    return runner.v2.legacy.validate_packet(packet, annotation_pass="prefix_novelty")


def _load_codebook(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = _mapping(config["bindings"]["codebook_v2"], "codebook binding")
    path = _repo_file(binding, name="codebook_v2")
    try:
        return runner.v2.validate_v2_codebook(_load_json(path))
    except runner.v2.QuoteFirstRunnerError as error:
        raise ControlInputDraftError(str(error)) from error


def _request_projection(
    *, messages: Sequence[Mapping[str, Any]], response_schema: Mapping[str, Any]
) -> dict[str, Any]:
    value = {
        "messages": copy.deepcopy(list(messages)),
        "response_schema": copy.deepcopy(dict(response_schema)),
    }
    return {
        **value,
        "messages_sha256": sha256_bytes(canonical_json_bytes(value["messages"])),
        "response_schema_sha256": sha256_bytes(
            canonical_json_bytes(value["response_schema"])
        ),
        "request_projection_sha256": sha256_bytes(canonical_json_bytes(value)),
    }


def _completion_record(
    *, config: Mapping[str, Any], config_sha256: str, codebook: Mapping[str, Any],
    row: Mapping[str, Any], ordinal: int
) -> dict[str, Any]:
    packet = _completion_packet(
        config=config, config_sha256=config_sha256, row=row, ordinal=ordinal
    )
    catalog_result = catalog.build_candidate_catalog(packet=packet)
    _require(
        catalog_result.get("catalog_status") == "available"
        and catalog_result.get("catalog_usable") is True,
        f"{row['id']} candidate catalog unavailable",
    )
    bundle = _mapping(catalog_result["candidate_unit_bundle"], "candidate bundle")
    try:
        authenticated = runner.authenticate_candidate_unit_bundle(
            value=bundle,
            packet=packet,
            expected_bundle_sha256=str(
                catalog_result["candidate_unit_bundle_sha256"]
            ),
        )
    except runner.BoundedIdRunnerError as error:
        raise ControlInputDraftError(str(error)) from error
    text = str(row["assistant_text"])
    projections: dict[str, Any] = {}
    host_action = "invoke_completion_decision"
    if text == "":
        host_action = "bypass_exact_empty_visible_text"
    else:
        decision_messages = runner.build_independent_messages(
            packet=packet,
            codebook=codebook,
            annotation_pass="completion_chain",
            authenticated_units=authenticated,
            response_route="decision",
        )
        projections["decision"] = _request_projection(
            messages=decision_messages,
            response_schema=runner.completion_decision_response_schema(
                authenticated
            ),
        )
        if len(authenticated.units) >= 3:
            detail_messages = runner.build_independent_messages(
                packet=packet,
                codebook=codebook,
                annotation_pass="completion_chain",
                authenticated_units=authenticated,
                response_route="chain_detail",
            )
            projections["chain_detail_if_opened"] = _request_projection(
                messages=detail_messages,
                response_schema=runner.completion_chain_detail_response_schema(
                    codebook, authenticated
                ),
            )
    record = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": "swe_task_state_v4_epistemic_chain_completion_input_draft_v3",
        "control_id": row["id"],
        "pass": "completion",
        "packet": packet,
        "packet_sha256": sha256_bytes(canonical_json_bytes(packet)),
        "catalog": catalog_result,
        "catalog_sha256": catalog.catalog_result_sha256(catalog_result),
        "host_action": host_action,
        "model_input_projections": projections,
    }
    record["record_sha256"] = sha256_bytes(canonical_json_bytes(record))
    return record


def _novelty_record(
    *, config: Mapping[str, Any], config_sha256: str, codebook: Mapping[str, Any],
    row: Mapping[str, Any], ordinal: int
) -> dict[str, Any]:
    packet = _novelty_packet(
        config=config, config_sha256=config_sha256, row=row, ordinal=ordinal
    )
    messages = runner.build_independent_messages(
        packet=packet,
        codebook=codebook,
        annotation_pass="prefix_novelty",
        response_route="decision",
    )
    projection = _request_projection(
        messages=messages,
        response_schema=runner.novelty_decision_response_schema(codebook),
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": "swe_task_state_v4_epistemic_chain_novelty_input_draft_v3",
        "control_id": row["id"],
        "pass": "novelty",
        "packet": packet,
        "packet_sha256": sha256_bytes(canonical_json_bytes(packet)),
        "model_input_projection": projection,
    }
    record["record_sha256"] = sha256_bytes(canonical_json_bytes(record))
    return record


def build_input_manifest_draft() -> dict[str, Any]:
    """Return an authenticated in-memory draft; never write or seal it."""

    config, config_sha256 = load_input_config()
    codebook = _load_codebook(config)
    completions = [
        _completion_record(
            config=config,
            config_sha256=config_sha256,
            codebook=codebook,
            row=_mapping(row, "completion input"),
            ordinal=index,
        )
        for index, row in enumerate(config["completion_inputs"], start=1)
    ]
    novelties = [
        _novelty_record(
            config=config,
            config_sha256=config_sha256,
            codebook=codebook,
            row=_mapping(row, "novelty input"),
            ordinal=index,
        )
        for index, row in enumerate(config["novelty_inputs"], start=1)
    ]
    packets = [record["packet"] for record in completions]
    catalogs = [record["catalog"] for record in completions]
    catalog_manifest = catalog.build_catalog_manifest(
        packets=packets, catalogs=catalogs
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": MANIFEST_KIND,
        "status": "prospective_draft_not_sealed_not_run",
        "scope": copy.deepcopy(CLAIM_SCOPE),
        "input_config_sha256": config_sha256,
        "implementation_bindings": copy.deepcopy(config["bindings"]),
        "counts": {"completion": 32, "novelty": 8, "total": 40},
        "completion_records": completions,
        "novelty_records": novelties,
        "catalog_manifest": catalog_manifest,
        "ordered_packet_ids_sha256": sha256_bytes(
            canonical_json_bytes(
                [record["packet"]["packet_id_sha256"] for record in completions + novelties]
            )
        ),
        "ordered_record_sha256s_sha256": sha256_bytes(
            canonical_json_bytes(
                [record["record_sha256"] for record in completions + novelties]
            )
        ),
    }
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
    }


__all__ = [
    "CONFIG_PATH",
    "ControlInputDraftError",
    "build_input_manifest_draft",
    "canonical_json_bytes",
    "load_input_config",
    "sha256_bytes",
    "sha256_file",
    "validate_input_config",
]
