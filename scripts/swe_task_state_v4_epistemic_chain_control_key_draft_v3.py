#!/usr/bin/env python3
"""Materialize the physically separate V3 control key after external input auth.

The caller must provide the exact in-memory input manifest and an independently
transported SHA-256.  Authentication happens before this module reads its key
config or constructs a semantic row.  This module neither imports nor reads the
generation-side input authoring module/config and has no artifact write path.
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
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_control_key_draft_v3.json"
)
CONFIG_KIND = "swe_task_state_v4_epistemic_chain_control_key_draft_v3"
INPUT_MANIFEST_KIND = (
    "swe_task_state_v4_epistemic_chain_control_input_manifest_draft_v3"
)
KEY_MANIFEST_KIND = (
    "swe_task_state_v4_epistemic_chain_control_key_manifest_draft_v3"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_GENERATION_FIELD_TOKENS = (
    "gold",
    "expected",
    "expectation",
    "answer",
    "category",
    "diagnostic",
    "key",
)
EXPECTED_INPUT_IMPLEMENTATION_BINDINGS = {
    "input_builder": {
        "path": "scripts/swe_task_state_v4_epistemic_chain_control_inputs_draft_v3.py",
        "sha256": "770457bd760fd81f50966203d3284e7dd697d661c0cab2fe472bb6b99aa92553",
    },
    "candidate_catalog_config": {
        "path": "configs/swe_task_state_v4_epistemic_chain_candidate_catalog_v3.json",
        "sha256": "c9f2598985d2f3eab27e923bd2aa35049ed2f478a311181354349a4cc88ba15e",
    },
    "candidate_catalog_builder": {
        "path": "scripts/swe_task_state_v4_epistemic_chain_candidate_catalog_v3.py",
        "sha256": "0d38bdcb0ec3aae1197375aaf34ceeb16f470df9dd855dc00c698b82e1ee2176",
    },
    "runner_v3": {
        "path": "scripts/swe_task_state_v4_epistemic_chain_annotation_runner_v3.py",
        "sha256": "c35af19c9f3f1e38208ba7f23467c386ebff2e5fb6572582f9a620f51f394aeb",
    },
    "codebook_v2": {
        "path": "configs/swe_task_state_v4_epistemic_chain_codebook_v2.json",
        "sha256": "2105a50c7bc13a064ca75c4a69aad631869fbbb2c17c94970eeaa92722aff85c",
    },
}
EXPECTED_KEY_IMPLEMENTATION_PATHS = {
    "key_materializer": (
        "scripts/swe_task_state_v4_epistemic_chain_control_key_draft_v3.py"
    ),
    **{
        name: str(binding["path"])
        for name, binding in EXPECTED_INPUT_IMPLEMENTATION_BINDINGS.items()
        if name != "input_builder"
    },
}
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


class ControlKeyDraftError(RuntimeError):
    """Raised when external authentication or key consistency fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlKeyDraftError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    _require(
        isinstance(value, list),
        f"{name} must be an array",
    )
    return value


def _exact_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ControlKeyDraftError(f"duplicate JSON key: {key}")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlKeyDraftError(f"cannot load strict JSON {path}: {error}") from error


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
        raise ControlKeyDraftError(
            f"{name} binding must resolve inside repository"
        ) from error
    _require(path.is_file(), f"{name} binding is not a file")
    _require(sha256_file(path) == binding["sha256"], f"{name} hash changed")
    return path


def _assert_generation_manifest_semantic_blind(
    value: Any, *, path: tuple[str, ...] = ()
) -> None:
    """Reject hidden key/label channels at every object depth.

    Visible text values are deliberately not scanned: words such as
    ``diagnostic`` may legitimately occur in authored prose.  Object field
    names and implementation binding paths are protocol, not prose, so they
    must remain expectation-blind.
    """

    if isinstance(value, Mapping):
        for raw_name, item in value.items():
            _require(isinstance(raw_name, str), "input manifest field names must be strings")
            folded = raw_name.casefold()
            _require(
                not any(token in folded for token in FORBIDDEN_GENERATION_FIELD_TOKENS),
                f"forbidden semantic/key field at {'.'.join(path + (raw_name,))}",
            )
            _assert_generation_manifest_semantic_blind(
                item, path=path + (raw_name,)
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_generation_manifest_semantic_blind(
                item, path=path + (str(index),)
            )


def _validate_input_implementation_bindings(value: Any) -> dict[str, Any]:
    raw_bindings = _mapping(value, "input implementation bindings")
    _require(
        set(raw_bindings) == set(EXPECTED_INPUT_IMPLEMENTATION_BINDINGS),
        "input implementation binding names changed",
    )
    bindings: dict[str, Any] = {}
    for name in EXPECTED_INPUT_IMPLEMENTATION_BINDINGS:
        binding = dict(
            _mapping(raw_bindings[name], f"implementation binding {name}")
        )
        _require(
            set(binding) == {"path", "sha256"}
            and isinstance(binding["path"], str)
            and isinstance(binding["sha256"], str)
            and SHA256_RE.fullmatch(binding["sha256"]) is not None,
            f"input implementation {name} binding shape invalid",
        )
        binding_path = str(binding["path"])
        _require(
            not any(
                token in binding_path.casefold()
                for token in FORBIDDEN_GENERATION_FIELD_TOKENS
            ),
            f"forbidden semantic/key implementation path for {name}",
        )
        bindings[name] = binding
    _require(
        bindings == EXPECTED_INPUT_IMPLEMENTATION_BINDINGS,
        "input implementation bindings differ from the prospectively fixed contract",
    )
    for name, binding in bindings.items():
        # The generation module is deliberately not read or imported by the
        # physically separate key side.  Its hash is compared to the frozen
        # literal above.  All shared production dependencies are independently
        # resolved and hashed here.
        if name != "input_builder":
            _repo_file(binding, name=f"input implementation {name}")
    return bindings


def _load_generation_codebook(bindings: Mapping[str, Any]) -> dict[str, Any]:
    path = _repo_file(
        _mapping(bindings["codebook_v2"], "input codebook binding"),
        name="input codebook_v2",
    )
    try:
        return runner.v2.validate_v2_codebook(_load_json(path))
    except runner.v2.QuoteFirstRunnerError as error:
        raise ControlKeyDraftError(str(error)) from error


def _request_projection(
    *, messages: Sequence[Mapping[str, Any]], response_schema: Mapping[str, Any]
) -> dict[str, Any]:
    message_rows = copy.deepcopy(list(messages))
    schema = copy.deepcopy(dict(response_schema))
    value = {"messages": message_rows, "response_schema": schema}
    return {
        **value,
        "messages_sha256": sha256_bytes(canonical_json_bytes(message_rows)),
        "response_schema_sha256": sha256_bytes(canonical_json_bytes(schema)),
        "request_projection_sha256": sha256_bytes(canonical_json_bytes(value)),
    }


def _validate_completion_packet(
    *, packet: Any, control_id: str, input_config_sha256: str
) -> dict[str, Any]:
    try:
        validated = runner.v2.legacy.validate_packet(
            packet, annotation_pass="completion_chain"
        )
    except runner.v2.legacy.AnnotationRunnerError as error:
        raise ControlKeyDraftError(str(error)) from error
    _require(
        validated == packet
        and _exact_int(validated.get("schema_version"), SCHEMA_VERSION),
        f"{control_id} completion packet changed during validation",
    )
    text_record = dict(
        _mapping(validated["materialized_assistant_text"], "assistant text")
    )
    boundaries = dict(
        _mapping(validated["authenticated_boundaries"], "authenticated boundaries")
    )
    _require(
        set(text_record) == {"char_start", "char_end", "sha256", "text"},
        f"{control_id} assistant text record fields invalid",
    )
    _require(
        set(boundaries)
        == {"draft_input_config_sha256", "draft_payload_sha256", "draft_nonce"},
        f"{control_id} authenticated boundary fields invalid",
    )
    text = text_record["text"]
    nonce = boundaries["draft_nonce"]
    _require(
        _exact_int(text_record["char_start"], 0)
        and _exact_int(text_record["char_end"], len(text))
        and isinstance(text_record["sha256"], str)
        and SHA256_RE.fullmatch(text_record["sha256"]) is not None
        and boundaries["draft_input_config_sha256"] == input_config_sha256
        and boundaries["draft_payload_sha256"]
        == sha256_bytes(canonical_json_bytes({"assistant_text": text}))
        and isinstance(nonce, int)
        and not isinstance(nonce, bool)
        and nonce > 10_000,
        f"{control_id} authenticated boundaries invalid",
    )
    _require(
        (control_id == "C32") == (text == ""),
        "only C32 may carry exactly empty completion text",
    )
    return validated


def _validate_novelty_packet(*, packet: Any, control_id: str) -> dict[str, Any]:
    try:
        validated = runner.v2.legacy.validate_packet(
            packet, annotation_pass="prefix_novelty"
        )
    except runner.v2.legacy.AnnotationRunnerError as error:
        raise ControlKeyDraftError(str(error)) from error
    _require(
        validated == packet
        and _exact_int(validated.get("schema_version"), SCHEMA_VERSION),
        f"{control_id} novelty packet changed during validation",
    )
    hypothesis = dict(_mapping(validated["locked_hypothesis"], "locked hypothesis"))
    prefix = dict(_mapping(validated["authenticated_prefix"], "authenticated prefix"))
    _require(
        set(hypothesis)
        == {
            "text",
            "sha256",
            "completion_char_start",
            "completion_char_end",
            "materialized_completion_sha256",
        }
        and isinstance(hypothesis["text"], str)
        and bool(hypothesis["text"])
        and hypothesis["sha256"] == runner.sha256_text(hypothesis["text"])
        and _exact_int(hypothesis["completion_char_start"], 0)
        and _exact_int(
            hypothesis["completion_char_end"], len(hypothesis["text"])
        )
        and isinstance(hypothesis["materialized_completion_sha256"], str)
        and SHA256_RE.fullmatch(hypothesis["materialized_completion_sha256"])
        is not None,
        f"{control_id} locked hypothesis shape invalid",
    )
    _require(
        set(prefix)
        == {
            "source_sha256",
            "source_char_start",
            "source_char_end",
            "annotator_text",
            "annotator_text_sha256",
            "annotator_char_start",
            "annotator_char_end",
            "removed_ranges",
        }
        and isinstance(prefix["annotator_text"], str)
        and bool(prefix["annotator_text"])
        and prefix["source_sha256"] == runner.sha256_text(prefix["annotator_text"])
        and prefix["annotator_text_sha256"]
        == runner.sha256_text(prefix["annotator_text"])
        and _exact_int(prefix["source_char_start"], 0)
        and _exact_int(prefix["source_char_end"], len(prefix["annotator_text"]))
        and _exact_int(prefix["annotator_char_start"], 0)
        and _exact_int(
            prefix["annotator_char_end"], len(prefix["annotator_text"])
        )
        and prefix["removed_ranges"] == [],
        f"{control_id} authenticated prefix shape invalid",
    )
    return validated


def _validated_authenticated_units(
    *, record: Mapping[str, Any], rebuilt_catalog: Mapping[str, Any]
):
    try:
        return runner.authenticate_candidate_unit_bundle(
            value=_mapping(
                rebuilt_catalog["candidate_unit_bundle"], "candidate unit bundle"
            ),
            packet=_mapping(record["packet"], "completion packet"),
            expected_bundle_sha256=str(
                rebuilt_catalog["candidate_unit_bundle_sha256"]
            ),
        )
    except runner.BoundedIdRunnerError as error:
        raise ControlKeyDraftError(str(error)) from error


def _validate_key_config(value: Any) -> dict[str, Any]:
    config = dict(_mapping(value, "control key draft config"))
    _require(
        set(config)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "status",
            "scope",
            "external_input_authentication",
            "counts",
            "bindings",
            "completion_gold",
            "novelty_gold",
        },
        "control key draft config fields invalid",
    )
    _require(
        config["schema_version"] == SCHEMA_VERSION
        and config["interface_version"] == INTERFACE_VERSION
        and config["kind"] == CONFIG_KIND
        and config["status"] == "prospective_draft_not_sealed_not_run",
        "control key draft identity invalid",
    )
    _require(dict(_mapping(config["scope"], "key scope")) == CLAIM_SCOPE,
             "control key claim scope invalid")
    _require(
        dict(_mapping(config["external_input_authentication"], "external auth"))
        == {
            "independently_supplied_manifest_sha256_required": True,
            "self_reported_manifest_sha256_is_not_a_trust_root": True,
            "input_module_import_forbidden": True,
            "input_config_read_forbidden": True,
            "materialization_before_external_hash_authentication_forbidden": True,
        },
        "external input authentication contract invalid",
    )
    _require(
        dict(_mapping(config["counts"], "key counts"))
        == {
            "completion": 32,
            "completion_chain": 18,
            "completion_no_chain": 13,
            "completion_unknown": 1,
            "novelty": 8,
            "novelty_novel": 2,
            "novelty_prefix_exposed": 4,
            "novelty_ambiguous": 2,
            "total": 40,
        },
        "control key counts changed",
    )
    bindings = dict(_mapping(config["bindings"], "key bindings"))
    _require(
        set(bindings)
        == {
            "key_materializer",
            "candidate_catalog_config",
            "candidate_catalog_builder",
            "runner_v3",
            "codebook_v2",
        },
        "control key binding names invalid",
    )
    expected_names = {
        "key_materializer": Path(__file__).name,
        "candidate_catalog_config": catalog.CONFIG_PATH.name,
        "candidate_catalog_builder": Path(catalog.__file__).name,
        "runner_v3": Path(runner.__file__).name,
        "codebook_v2": "swe_task_state_v4_epistemic_chain_codebook_v2.json",
    }
    for name, raw in bindings.items():
        binding = _mapping(raw, name)
        _require(
            binding.get("path") == EXPECTED_KEY_IMPLEMENTATION_PATHS[name],
            f"{name} binding path changed",
        )
        path = _repo_file(binding, name=name)
        _require(path.name == expected_names[name], f"{name} binds wrong file")

    completion_rows = [
        dict(_mapping(item, "completion gold"))
        for item in _sequence(config["completion_gold"], "completion gold")
    ]
    novelty_rows = [
        dict(_mapping(item, "novelty gold"))
        for item in _sequence(config["novelty_gold"], "novelty gold")
    ]
    _require(len(completion_rows) == 32 and len(novelty_rows) == 8,
             "control key row counts changed")
    _require(
        [row.get("id") for row in completion_rows]
        == [f"C{index:02d}" for index in range(1, 33)],
        "completion gold IDs changed",
    )
    _require(
        [row.get("id") for row in novelty_rows]
        == [f"V{index:02d}" for index in range(1, 9)],
        "novelty gold IDs changed",
    )
    completion_categories = [row.get("gold_category") for row in completion_rows]
    _require(
        completion_categories.count("chain") == 18
        and completion_categories.count("no_chain") == 13
        and completion_categories.count("unknown") == 1
        and completion_categories[:18] == ["chain"] * 18
        and completion_categories[18:30] == ["no_chain"] * 12
        and completion_categories[30] == "unknown"
        and completion_categories[31] == "no_chain",
        "completion category schedule changed",
    )
    novelty_categories = [row.get("gold_category") for row in novelty_rows]
    _require(
        novelty_categories == [
            "novel",
            "novel",
            "prefix_exposed",
            "prefix_exposed",
            "prefix_exposed",
            "prefix_exposed",
            "ambiguous",
            "ambiguous",
        ],
        "novelty category schedule changed",
    )
    for row in completion_rows:
        category = row["gold_category"]
        common = {"id", "gold_category", "diagnostic"}
        if category == "chain":
            _require(
                set(row)
                == common | {"gold_unit_ordinals", "gold_ontology", "gold_markers"},
                f"{row['id']} chain gold fields invalid",
            )
            ordinals = dict(_mapping(row["gold_unit_ordinals"], "unit ordinals"))
            _require(
                set(ordinals) == {"evidence", "hypothesis", "action"}
                and all(
                    isinstance(item, int) and not isinstance(item, bool) and item >= 0
                    for item in ordinals.values()
                )
                and ordinals["evidence"] < ordinals["hypothesis"] < ordinals["action"],
                f"{row['id']} unit ordinals invalid",
            )
            ontology = dict(_mapping(row["gold_ontology"], "gold ontology"))
            _require(
                set(ontology)
                == {"evidence_kind", "belief_edge", "hypothesis_domain", "action_intent"}
                and all(isinstance(item, str) for item in ontology.values()),
                f"{row['id']} ontology fields invalid",
            )
            markers = dict(_mapping(row["gold_markers"], "gold markers"))
            _require(
                set(markers)
                == {"relation_marker_present", "action_marker_present"}
                and all(isinstance(item, bool) for item in markers.values()),
                f"{row['id']} marker fields invalid",
            )
        elif row["id"] == "C32":
            _require(
                set(row) == common | {"host_bypass_required"}
                and row["host_bypass_required"] is True,
                "C32 host bypass key invalid",
            )
        else:
            _require(set(row) == common, f"{row['id']} non-chain gold fields invalid")
        _require(isinstance(row.get("diagnostic"), str) and bool(row["diagnostic"]),
                 f"{row['id']} diagnostic invalid")
    for row in novelty_rows:
        _require(
            set(row) == {"id", "gold_category", "diagnostic"}
            and isinstance(row["diagnostic"], str)
            and bool(row["diagnostic"]),
            f"{row['id']} novelty gold fields invalid",
        )
    return config


def _load_key_config_after_external_auth() -> tuple[dict[str, Any], str]:
    config = _validate_key_config(_load_json(CONFIG_PATH))
    return config, sha256_file(CONFIG_PATH)


def _validate_record_hash(record: Mapping[str, Any], *, control_id: str) -> str:
    reported = record.get("record_sha256")
    unsigned = dict(record)
    unsigned.pop("record_sha256", None)
    _require(
        isinstance(reported, str)
        and SHA256_RE.fullmatch(reported) is not None
        and sha256_bytes(canonical_json_bytes(unsigned)) == reported,
        f"{control_id} input record hash invalid",
    )
    return reported


def _validate_completion_record(
    *, raw_record: Any, expected_control_id: str, input_config_sha256: str,
    codebook: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str, str, str, int]:
    record = dict(_mapping(raw_record, "completion input record"))
    _require(
        set(record)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "control_id",
            "pass",
            "packet",
            "packet_sha256",
            "catalog",
            "catalog_sha256",
            "host_action",
            "model_input_projections",
            "record_sha256",
        },
        f"{expected_control_id} completion record fields invalid",
    )
    _require(
        _exact_int(record["schema_version"], SCHEMA_VERSION)
        and _exact_int(record["interface_version"], INTERFACE_VERSION)
        and record["kind"]
        == "swe_task_state_v4_epistemic_chain_completion_input_draft_v3"
        and record["control_id"] == expected_control_id
        and record["pass"] == "completion",
        f"{expected_control_id} completion record identity invalid",
    )
    packet = _validate_completion_packet(
        packet=record["packet"],
        control_id=expected_control_id,
        input_config_sha256=input_config_sha256,
    )
    _require(
        record["packet_sha256"] == sha256_bytes(canonical_json_bytes(packet)),
        f"{expected_control_id} packet hash invalid",
    )
    try:
        rebuilt_catalog = catalog.build_candidate_catalog(packet=packet)
    except catalog.CandidateCatalogError as error:
        raise ControlKeyDraftError(str(error)) from error
    _require(
        rebuilt_catalog.get("catalog_status") == "available"
        and rebuilt_catalog.get("catalog_usable") is True,
        f"{expected_control_id} candidate catalog unavailable",
    )
    _require(
        record["catalog"] == rebuilt_catalog,
        f"{expected_control_id} catalog differs from independent rebuild",
    )
    _require(
        record["catalog_sha256"] == catalog.catalog_result_sha256(rebuilt_catalog),
        f"{expected_control_id} catalog result hash invalid",
    )
    authenticated = _validated_authenticated_units(
        record=record, rebuilt_catalog=rebuilt_catalog
    )
    text = str(packet["materialized_assistant_text"]["text"])
    if text == "":
        expected_host_action = "bypass_exact_empty_visible_text"
        expected_projections: dict[str, Any] = {}
        _require(
            len(authenticated.units) == 0,
            "C32 empty host bypass unexpectedly has candidate units",
        )
    else:
        expected_host_action = "invoke_completion_decision"
        try:
            decision_messages = runner.build_independent_messages(
                packet=packet,
                codebook=codebook,
                annotation_pass="completion_chain",
                authenticated_units=authenticated,
                response_route="decision",
            )
            expected_projections = {
                "decision": _request_projection(
                    messages=decision_messages,
                    response_schema=runner.completion_decision_response_schema(
                        authenticated
                    ),
                )
            }
            if len(authenticated.units) >= 3:
                detail_messages = runner.build_independent_messages(
                    packet=packet,
                    codebook=codebook,
                    annotation_pass="completion_chain",
                    authenticated_units=authenticated,
                    response_route="chain_detail",
                )
                expected_projections["chain_detail_if_opened"] = _request_projection(
                    messages=detail_messages,
                    response_schema=runner.completion_chain_detail_response_schema(
                        codebook, authenticated
                    ),
                )
        except runner.BoundedIdRunnerError as error:
            raise ControlKeyDraftError(str(error)) from error
    _require(
        record["host_action"] == expected_host_action,
        f"{expected_control_id} host action invalid",
    )
    _require(
        record["model_input_projections"] == expected_projections,
        f"{expected_control_id} production completion request projection changed",
    )
    record_hash = _validate_record_hash(record, control_id=expected_control_id)
    return (
        record,
        rebuilt_catalog,
        str(packet["packet_id_sha256"]),
        record_hash,
        text,
        int(packet["authenticated_boundaries"]["draft_nonce"]),
    )


def _validate_novelty_record(
    *, raw_record: Any, expected_control_id: str, codebook: Mapping[str, Any]
) -> tuple[dict[str, Any], str, str, tuple[str, str]]:
    record = dict(_mapping(raw_record, "novelty input record"))
    _require(
        set(record)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "control_id",
            "pass",
            "packet",
            "packet_sha256",
            "model_input_projection",
            "record_sha256",
        },
        f"{expected_control_id} novelty record fields invalid",
    )
    _require(
        _exact_int(record["schema_version"], SCHEMA_VERSION)
        and _exact_int(record["interface_version"], INTERFACE_VERSION)
        and record["kind"]
        == "swe_task_state_v4_epistemic_chain_novelty_input_draft_v3"
        and record["control_id"] == expected_control_id
        and record["pass"] == "novelty",
        f"{expected_control_id} novelty record identity invalid",
    )
    packet = _validate_novelty_packet(
        packet=record["packet"], control_id=expected_control_id
    )
    _require(
        record["packet_sha256"] == sha256_bytes(canonical_json_bytes(packet)),
        f"{expected_control_id} packet hash invalid",
    )
    try:
        messages = runner.build_independent_messages(
            packet=packet,
            codebook=codebook,
            annotation_pass="prefix_novelty",
            response_route="decision",
        )
        expected_projection = _request_projection(
            messages=messages,
            response_schema=runner.novelty_decision_response_schema(codebook),
        )
    except runner.BoundedIdRunnerError as error:
        raise ControlKeyDraftError(str(error)) from error
    _require(
        record["model_input_projection"] == expected_projection,
        f"{expected_control_id} production novelty request projection changed",
    )
    record_hash = _validate_record_hash(record, control_id=expected_control_id)
    prefix = str(packet["authenticated_prefix"]["annotator_text"])
    hypothesis = str(packet["locked_hypothesis"]["text"])
    return (
        record,
        str(packet["packet_id_sha256"]),
        record_hash,
        (prefix, hypothesis),
    )


def _authenticate_input_manifest_first(
    *, input_manifest: Mapping[str, Any], independently_supplied_manifest_sha256: str
) -> dict[str, Any]:
    _require(
        isinstance(independently_supplied_manifest_sha256, str)
        and SHA256_RE.fullmatch(independently_supplied_manifest_sha256) is not None,
        "independently supplied input manifest hash invalid",
    )
    manifest = copy.deepcopy(dict(_mapping(input_manifest, "input manifest")))
    try:
        observed = sha256_bytes(canonical_json_bytes(manifest))
    except (TypeError, ValueError) as error:
        raise ControlKeyDraftError("input manifest is not canonical JSON") from error
    _require(
        observed == independently_supplied_manifest_sha256,
        "input manifest differs from independently supplied frozen hash",
    )
    _assert_generation_manifest_semantic_blind(manifest)
    _require(
        set(manifest)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "status",
            "scope",
            "input_config_sha256",
            "implementation_bindings",
            "counts",
            "completion_records",
            "novelty_records",
            "catalog_manifest",
            "ordered_packet_ids_sha256",
            "ordered_record_sha256s_sha256",
        },
        "input manifest fields invalid",
    )
    _require(
        _exact_int(manifest["schema_version"], SCHEMA_VERSION)
        and _exact_int(manifest["interface_version"], INTERFACE_VERSION)
        and manifest["kind"] == INPUT_MANIFEST_KIND
        and manifest["status"] == "prospective_draft_not_sealed_not_run"
        and dict(_mapping(manifest["scope"], "input manifest scope")) == CLAIM_SCOPE
        and manifest["counts"] == {"completion": 32, "novelty": 8, "total": 40}
        and isinstance(manifest["input_config_sha256"], str)
        and SHA256_RE.fullmatch(manifest["input_config_sha256"]) is not None,
        "input manifest identity or scope invalid",
    )
    bindings = _validate_input_implementation_bindings(
        manifest["implementation_bindings"]
    )
    codebook = _load_generation_codebook(bindings)
    completion_items = list(
        _sequence(manifest["completion_records"], "completion records")
    )
    novelty_items = list(_sequence(manifest["novelty_records"], "novelty records"))
    _require(
        len(completion_items) == 32 and len(novelty_items) == 8,
        "input manifest record counts changed",
    )
    completion_records: list[dict[str, Any]] = []
    novelty_records: list[dict[str, Any]] = []
    catalogs: list[dict[str, Any]] = []
    packet_ids: list[str] = []
    source_ids: list[str] = []
    record_hashes: list[str] = []
    completion_texts: list[str] = []
    completion_nonces: list[int] = []
    novelty_pairs: list[tuple[str, str]] = []
    for index, raw_record in enumerate(completion_items, start=1):
        (
            record,
            rebuilt_catalog,
            packet_id,
            record_hash,
            text,
            nonce,
        ) = _validate_completion_record(
            raw_record=raw_record,
            expected_control_id=f"C{index:02d}",
            input_config_sha256=str(manifest["input_config_sha256"]),
            codebook=codebook,
        )
        completion_records.append(record)
        catalogs.append(rebuilt_catalog)
        packet_ids.append(packet_id)
        source_ids.append(str(record["packet"]["source_id_sha256"]))
        record_hashes.append(record_hash)
        completion_texts.append(text)
        completion_nonces.append(nonce)
    for index, raw_record in enumerate(novelty_items, start=1):
        record, packet_id, record_hash, pair = _validate_novelty_record(
            raw_record=raw_record,
            expected_control_id=f"V{index:02d}",
            codebook=codebook,
        )
        novelty_records.append(record)
        packet_ids.append(packet_id)
        source_ids.append(str(record["packet"]["source_id_sha256"]))
        record_hashes.append(record_hash)
        novelty_pairs.append(pair)
    _require(
        len(packet_ids) == len(set(packet_ids)) == 40,
        "input packet IDs duplicate",
    )
    _require(
        len(source_ids) == len(set(source_ids)) == 40,
        "input source IDs duplicate",
    )
    _require(
        len(set(completion_texts)) == 32,
        "completion texts duplicate",
    )
    _require(
        all(
            right == left + 1
            for left, right in zip(completion_nonces, completion_nonces[1:])
        ),
        "completion draft nonces are not one fixed consecutive schedule",
    )
    _require(len(set(novelty_pairs)) == 8, "novelty text pairs duplicate")
    _require(
        manifest["ordered_packet_ids_sha256"]
        == sha256_bytes(canonical_json_bytes(packet_ids))
        and manifest["ordered_record_sha256s_sha256"]
        == sha256_bytes(canonical_json_bytes(record_hashes)),
        "input ordering hashes invalid",
    )
    packets = [record["packet"] for record in completion_records]
    try:
        rebuilt_catalog_envelope = catalog.build_catalog_manifest(
            packets=packets, catalogs=catalogs
        )
        _require(
            manifest["catalog_manifest"] == rebuilt_catalog_envelope,
            "catalog manifest differs from independent rebuild",
        )
        catalog.authenticate_catalog_manifest(
            value=rebuilt_catalog_envelope,
            expected_manifest_sha256=str(
                rebuilt_catalog_envelope["manifest_sha256"]
            ),
            packets=packets,
            catalogs=catalogs,
        )
    except catalog.CandidateCatalogError as error:
        raise ControlKeyDraftError(str(error)) from error
    return manifest


def _load_codebook(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _repo_file(
        _mapping(config["bindings"]["codebook_v2"], "codebook binding"),
        name="codebook_v2",
    )
    try:
        return runner.v2.validate_v2_codebook(_load_json(path))
    except runner.v2.QuoteFirstRunnerError as error:
        raise ControlKeyDraftError(str(error)) from error


def _authenticated_units(record: Mapping[str, Any]):
    catalog_result = _mapping(record["catalog"], "catalog result")
    try:
        return runner.authenticate_candidate_unit_bundle(
            value=_mapping(
                catalog_result["candidate_unit_bundle"], "candidate unit bundle"
            ),
            packet=_mapping(record["packet"], "completion packet"),
            expected_bundle_sha256=str(
                catalog_result["candidate_unit_bundle_sha256"]
            ),
        )
    except runner.BoundedIdRunnerError as error:
        raise ControlKeyDraftError(str(error)) from error


def _completion_key_row(
    *, gold: Mapping[str, Any], input_record: Mapping[str, Any],
    codebook: Mapping[str, Any]
) -> dict[str, Any]:
    category = str(gold["gold_category"])
    authenticated = _authenticated_units(input_record)
    exact_tuple: dict[str, str] | None = None
    if category == "chain":
        ordinals = _mapping(gold["gold_unit_ordinals"], "gold unit ordinals")
        units = authenticated.units
        try:
            evidence = units[int(ordinals["evidence"])]
            hypothesis = units[int(ordinals["hypothesis"])]
            action = units[int(ordinals["action"])]
        except IndexError as error:
            raise ControlKeyDraftError(
                f"{gold['id']} gold ordinal outside authenticated catalog"
            ) from error
        _require(
            evidence.assistant_char_start < hypothesis.assistant_char_start
            < action.assistant_char_start
            and evidence.assistant_char_end <= hypothesis.assistant_char_start
            and hypothesis.assistant_char_end <= action.assistant_char_start,
            f"{gold['id']} gold units are not exact E-before-H-before-A",
        )
        ontology = dict(_mapping(gold["gold_ontology"], "gold ontology"))
        proposal = {
            "decision": "chain",
            "evidence_unit_id": evidence.unit_id,
            "hypothesis_unit_id": hypothesis.unit_id,
            "action_unit_id": action.unit_id,
            **ontology,
        }
        exact_tuple = {
            "evidence_unit_id": evidence.unit_id,
            "hypothesis_unit_id": hypothesis.unit_id,
            "action_unit_id": action.unit_id,
        }
    elif category == "no_chain":
        proposal = {"decision": "no_chain"}
    else:
        _require(category == "unknown", f"{gold['id']} completion category invalid")
        proposal = {
            "decision": "unknown",
            "unknown_reason": runner.COMPLETION_UNKNOWN_REASON,
        }
    result = runner.materialize_completion_proposal(
        proposal=proposal,
        codebook=codebook,
        authenticated_units=authenticated,
        decision_source="prospective_control_key_draft",
    )
    _require(result["semantic_validation_status"] == "valid",
             f"{gold['id']} gold proposal is invalid")
    if category == "chain":
        observed_markers = {
            "relation_marker_present": result["annotation_record"][
                "relation_marker_present"
            ],
            "action_marker_present": result["annotation_record"][
                "action_marker_present"
            ],
        }
        _require(
            observed_markers == gold["gold_markers"],
            f"{gold['id']} host-derived marker gold changed",
        )
        annotation = _mapping(result["annotation_record"], "chain annotation")
        ontology = _mapping(gold["gold_ontology"], "gold ontology")
        _require(
            all(annotation[key] == ontology[key] for key in ontology),
            f"{gold['id']} materialized ontology changed",
        )
    host_bypass = gold.get("host_bypass_required") is True
    if host_bypass:
        _require(
            input_record.get("host_action") == "bypass_exact_empty_visible_text"
            and input_record.get("model_input_projections") == {}
            and len(authenticated.units) == 0
            and input_record["packet"]["materialized_assistant_text"]["text"] == "",
            "C32 exact empty host bypass changed",
        )
    row = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": "swe_task_state_v4_epistemic_chain_completion_key_row_draft_v3",
        "control_id": gold["id"],
        "pass": "completion",
        "packet_id_sha256": input_record["packet"]["packet_id_sha256"],
        "input_record_sha256": input_record["record_sha256"],
        "gold_category": category,
        "diagnostic": gold["diagnostic"],
        "gold_proposal": proposal,
        "gold_exact_unit_id_tuple": exact_tuple,
        "gold_materialized_result": result,
        "model_invocation_required": not host_bypass,
    }
    return row


def _novelty_key_row(
    *, gold: Mapping[str, Any], input_record: Mapping[str, Any]
) -> dict[str, Any]:
    category = str(gold["gold_category"])
    proposal = {"decision": category}
    result = runner.materialize_novelty_proposal(
        proposal=proposal, decision_source="prospective_control_key_draft"
    )
    _require(
        result["semantic_validation_status"] == "valid"
        and result["annotation_record"]["novelty_status"] == category,
        f"{gold['id']} novelty gold invalid",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": "swe_task_state_v4_epistemic_chain_novelty_key_row_draft_v3",
        "control_id": gold["id"],
        "pass": "novelty",
        "packet_id_sha256": input_record["packet"]["packet_id_sha256"],
        "input_record_sha256": input_record["record_sha256"],
        "gold_category": category,
        "diagnostic": gold["diagnostic"],
        "gold_proposal": proposal,
        "gold_materialized_result": result,
        "model_invocation_required": True,
    }


def materialize_control_key_draft(
    *, input_manifest: Mapping[str, Any],
    independently_supplied_manifest_sha256: str
) -> dict[str, Any]:
    """Authenticate the frozen input hash first, then build the separate key."""

    authenticated_manifest = _authenticate_input_manifest_first(
        input_manifest=input_manifest,
        independently_supplied_manifest_sha256=independently_supplied_manifest_sha256,
    )
    # The physical key is intentionally opened only after the preceding trust
    # check succeeds.  No generation-side module or config is imported/read.
    config, key_config_sha256 = _load_key_config_after_external_auth()
    shared_binding_names = {
        "candidate_catalog_config",
        "candidate_catalog_builder",
        "runner_v3",
        "codebook_v2",
    }
    _require(
        {
            name: authenticated_manifest["implementation_bindings"][name]
            for name in shared_binding_names
        }
        == {name: config["bindings"][name] for name in shared_binding_names},
        "input and key production implementation bindings differ",
    )
    codebook = _load_codebook(config)
    completion_inputs = {
        str(row["control_id"]): row
        for row in authenticated_manifest["completion_records"]
    }
    novelty_inputs = {
        str(row["control_id"]): row
        for row in authenticated_manifest["novelty_records"]
    }
    completion_rows = [
        _completion_key_row(
            gold=_mapping(gold, "completion gold"),
            input_record=_mapping(completion_inputs[str(gold["id"])], "completion input"),
            codebook=codebook,
        )
        for gold in config["completion_gold"]
    ]
    novelty_rows = [
        _novelty_key_row(
            gold=_mapping(gold, "novelty gold"),
            input_record=_mapping(novelty_inputs[str(gold["id"])], "novelty input"),
        )
        for gold in config["novelty_gold"]
    ]
    _require(
        sum(row["gold_exact_unit_id_tuple"] is not None for row in completion_rows)
        == 18,
        "exact occurrence-specific E/H/A tuple count changed",
    )
    key_manifest = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": KEY_MANIFEST_KIND,
        "status": "prospective_draft_not_sealed_not_run",
        "scope": copy.deepcopy(CLAIM_SCOPE),
        "externally_authenticated_input_manifest_sha256": (
            independently_supplied_manifest_sha256
        ),
        "key_config_sha256": key_config_sha256,
        "counts": copy.deepcopy(config["counts"]),
        "completion_rows": completion_rows,
        "novelty_rows": novelty_rows,
        "ordered_key_rows_sha256": sha256_bytes(
            canonical_json_bytes(completion_rows + novelty_rows)
        ),
    }
    return {
        "manifest": key_manifest,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(key_manifest)),
    }


__all__ = [
    "CONFIG_PATH",
    "ControlKeyDraftError",
    "canonical_json_bytes",
    "materialize_control_key_draft",
    "sha256_bytes",
    "sha256_file",
]
