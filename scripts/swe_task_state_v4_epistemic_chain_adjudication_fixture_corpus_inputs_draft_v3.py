#!/usr/bin/env python3
"""Build the answer-blind side of a prospective V3 adjudication fixture corpus.

The returned envelope contains only authored visible packets, exact candidate
unit catalogs, and two unlabeled authored projections per case.  It contains no
semantic answer, preferred projection, presentation slot, adjudication route,
or scoring object.  It never writes an artifact, creates a fixture nonce, builds
a generation lock, calls a generation callback, or claims precommit chronology.

The nonce receipt stored here is deliberately a placeholder reference.  A
future sealed outer executor must generate and precommit real nonces before it
may read the physically separate semantic-key configuration.
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

import swe_task_state_v4_epistemic_chain_adjudication_fixture_v3 as fixture  # noqa: E402
import swe_task_state_v4_epistemic_chain_candidate_catalog_v3 as catalog  # noqa: E402


runner = fixture.production
SCHEMA_VERSION = 1
INTERFACE_VERSION = 3
CONFIG_PATH = (
    ROOT
    / "configs"
    / "swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_inputs_draft_v3.json"
)
CONFIG_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_inputs_draft_v3"
)
MANIFEST_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_input_manifest_draft_v3"
)
COMPLETION_RECORD_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_fixture_completion_input_draft_v3"
)
NOVELTY_RECORD_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_fixture_novelty_input_draft_v3"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CASE_ID_RE = re.compile(r"^F(?:0[1-9]|1[0-2])$")
FORBIDDEN_ANSWER_FIELD_FRAGMENTS = (
    "gold",
    "expected",
    "expectation",
    "winner",
    "label",
    "correct",
    "preferred",
    "route",
    "routing",
    "slot",
    "position",
    "presentation_class",
    "route_requirement",
)
ALLOWED_PROTOCOL_FIELD_NAMES = frozenset({"completion_chain_slots_present"})
CLAIM_SCOPE = {
    "development_controls_only": True,
    "reserved_validation_closed": True,
    "reserved_validation_accessed": False,
    "visible_semantic_cot_like_structure_targeted": True,
    "private_chain_of_thought_recovery_established": False,
    "verbatim_private_chain_of_thought_recovery_established": False,
    "affect_emotion_confidence_doubt_or_stress_targeted": False,
    "actual_model_execution_claimed": False,
}


class FixtureCorpusInputDraftError(RuntimeError):
    """Raised when draft input authorship or dependency binding fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureCorpusInputDraftError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    _require(
        isinstance(value, list),
        f"{name} must be an array",
    )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureCorpusInputDraftError(f"duplicate JSON key: {key}")
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


def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixtureCorpusInputDraftError(
            f"cannot load strict JSON {path}: {error}"
        ) from error


def _repo_file(binding: Mapping[str, Any], *, name: str) -> Path:
    _require(
        set(binding) == {"path", "sha256"}
        and isinstance(binding.get("path"), str)
        and SHA256_RE.fullmatch(str(binding.get("sha256"))) is not None,
        f"{name} binding invalid",
    )
    logical = ROOT / str(binding["path"])
    try:
        resolved = logical.resolve(strict=True)
        resolved.relative_to(ROOT.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as error:
        raise FixtureCorpusInputDraftError(
            f"{name} binding must resolve inside repository"
        ) from error
    _require(resolved.is_file(), f"{name} binding is not a file")
    _require(sha256_file(resolved) == binding["sha256"], f"{name} hash changed")
    return resolved


def assert_no_answer_fields(value: Any, *, path: tuple[str, ...] = ()) -> None:
    """Reject answer-like object field names recursively, but not visible prose."""

    if isinstance(value, Mapping):
        for raw_name, item in value.items():
            _require(isinstance(raw_name, str), "input field names must be strings")
            if raw_name in ALLOWED_PROTOCOL_FIELD_NAMES:
                _require(
                    path[-1:] == ("annotator_visibility",) and item is False,
                    f"protocol field {raw_name!r} is not allowed at {'.'.join(path)}",
                )
                assert_no_answer_fields(item, path=path + (raw_name,))
                continue
            folded = re.sub(r"[^a-z0-9]+", "_", raw_name.casefold()).strip("_")
            collapsed = folded.replace("_", "")
            for token in FORBIDDEN_ANSWER_FIELD_FRAGMENTS:
                normalized = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
                _require(
                    normalized not in folded
                    and normalized.replace("_", "") not in collapsed,
                    f"forbidden answer field at {'.'.join(path + (raw_name,))}",
                )
            assert_no_answer_fields(item, path=path + (raw_name,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_answer_fields(item, path=path + (str(index),))


def _validate_candidate_spec(value: Any, *, annotation_pass: str) -> dict[str, Any]:
    spec = copy.deepcopy(dict(_mapping(value, "authored candidate spec")))
    decision = spec.get("decision")
    if annotation_pass == "prefix_novelty":
        _require(
            set(spec) == {"decision"}
            and decision in {"novel", "prefix_exposed", "ambiguous"},
            "novelty candidate spec invalid",
        )
        return spec
    _require(annotation_pass == "completion_chain", "annotation pass invalid")
    if decision == "no_chain":
        _require(set(spec) == {"decision"}, "no_chain candidate spec invalid")
        return spec
    if decision == "unknown":
        _require(
            set(spec) == {"decision", "unknown_reason"}
            and spec["unknown_reason"] == runner.COMPLETION_UNKNOWN_REASON,
            "unknown candidate spec invalid",
        )
        return spec
    _require(decision == "chain", "completion candidate decision invalid")
    _require(
        set(spec) == {"decision", "unit_ordinals", "ontology"},
        "chain candidate spec fields invalid",
    )
    ordinals = dict(_mapping(spec["unit_ordinals"], "chain unit ordinals"))
    _require(
        set(ordinals) == {"evidence", "hypothesis", "action"}
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in ordinals.values()
        )
        and ordinals["evidence"] < ordinals["hypothesis"] < ordinals["action"],
        "chain candidate ordinals invalid",
    )
    ontology = dict(_mapping(spec["ontology"], "chain ontology"))
    _require(
        set(ontology)
        == {"evidence_kind", "belief_edge", "hypothesis_domain", "action_intent"}
        and all(isinstance(item, str) and bool(item) for item in ontology.values()),
        "chain candidate ontology invalid",
    )
    return spec


def validate_input_config(value: Any) -> dict[str, Any]:
    config = copy.deepcopy(dict(_mapping(value, "fixture corpus input config")))
    _require(
        set(config)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "status",
            "scope",
            "counts",
            "identity",
            "nonce_provenance",
            "bindings",
            "cases",
        },
        "fixture corpus input config fields invalid",
    )
    _require(
        config["schema_version"] == SCHEMA_VERSION
        and config["interface_version"] == INTERFACE_VERSION
        and config["kind"] == CONFIG_KIND
        and config["status"] == "prospective_draft_not_sealed_not_run"
        and config["scope"] == CLAIM_SCOPE,
        "fixture corpus input config identity or scope invalid",
    )
    _require(
        config["counts"] == {"completion": 9, "novelty": 3, "total": 12},
        "fixture corpus input counts changed",
    )
    identity = dict(_mapping(config["identity"], "fixture input identity"))
    _require(
        set(identity)
        == {
            "suite_id",
            "packet_id_domain",
            "packet_salt",
            "source_id_domain",
            "source_salt",
            "shard_seed",
            "packet_serial_base",
        }
        and all(
            isinstance(identity[field], str) and len(identity[field]) >= 16
            for field in (
                "suite_id",
                "packet_id_domain",
                "packet_salt",
                "source_id_domain",
                "source_salt",
                "shard_seed",
            )
        )
        and isinstance(identity["packet_serial_base"], int)
        and not isinstance(identity["packet_serial_base"], bool)
        and identity["packet_serial_base"] > 10_000,
        "fixture input identity invalid",
    )
    provenance = dict(_mapping(config["nonce_provenance"], "nonce provenance"))
    _require(
        set(provenance)
        == {
            "status",
            "placeholder_receipt_reference",
            "placeholder_receipt_reference_sha256",
            "actual_fixture_nonce_present",
            "precommit_chronology_claimed",
            "semantic_access_chronology_claimed",
            "chronology_responsibility",
        }
        and provenance["status"] == "external_precommit_pending"
        and isinstance(provenance["placeholder_receipt_reference"], str)
        and bool(provenance["placeholder_receipt_reference"])
        and provenance["placeholder_receipt_reference_sha256"]
        == sha256_text(provenance["placeholder_receipt_reference"])
        and provenance["actual_fixture_nonce_present"] is False
        and provenance["precommit_chronology_claimed"] is False
        and provenance["semantic_access_chronology_claimed"] is False
        and provenance["chronology_responsibility"]
        == fixture.OUTER_EXECUTOR_RESPONSIBILITY,
        "fixture nonce placeholder or chronology scope invalid",
    )
    bindings = dict(_mapping(config["bindings"], "fixture input bindings"))
    expected_names = {
        "input_builder": Path(__file__).name,
        "fixture_config": fixture.DEFAULT_CONFIG_PATH.name,
        "fixture_protocol": Path(fixture.__file__).name,
        "candidate_catalog_config": catalog.CONFIG_PATH.name,
        "candidate_catalog_builder": Path(catalog.__file__).name,
        "runner_v3": Path(runner.__file__).name,
        "codebook_v2": "swe_task_state_v4_epistemic_chain_codebook_v2.json",
    }
    _require(set(bindings) == set(expected_names), "fixture input bindings changed")
    for name, expected_name in expected_names.items():
        path = _repo_file(_mapping(bindings[name], name), name=name)
        _require(path.name == expected_name, f"{name} binds wrong file")

    cases = [dict(_mapping(item, "fixture input case")) for item in _sequence(config["cases"], "fixture input cases")]
    _require(len(cases) == 12, "fixture input case count changed")
    completion_count = 0
    novelty_count = 0
    for index, case in enumerate(cases, start=1):
        case_id = case.get("case_id")
        annotation_pass = case.get("annotation_pass")
        _require(
            isinstance(case_id, str)
            and CASE_ID_RE.fullmatch(case_id) is not None
            and case_id == f"F{index:02d}",
            "fixture case IDs or ordering changed",
        )
        if annotation_pass == "completion_chain":
            completion_count += 1
            _require(
                set(case)
                == {"case_id", "annotation_pass", "assistant_text", "authored_candidates"}
                and isinstance(case["assistant_text"], str)
                and bool(case["assistant_text"]),
                f"{case_id} completion input shape invalid",
            )
        else:
            novelty_count += 1
            _require(
                annotation_pass == "prefix_novelty"
                and set(case)
                == {
                    "case_id",
                    "annotation_pass",
                    "visible_prefix",
                    "locked_hypothesis",
                    "authored_candidates",
                }
                and isinstance(case["visible_prefix"], str)
                and bool(case["visible_prefix"])
                and isinstance(case["locked_hypothesis"], str)
                and bool(case["locked_hypothesis"]),
                f"{case_id} novelty input shape invalid",
            )
        candidates = [
            _validate_candidate_spec(item, annotation_pass=str(annotation_pass))
            for item in _sequence(case["authored_candidates"], "authored candidates")
        ]
        _require(
            len(candidates) == 2
            and canonical_json_bytes(candidates[0]) != canonical_json_bytes(candidates[1]),
            f"{case_id} must contain two distinct authored candidates",
        )
    _require(
        (completion_count, novelty_count) == (9, 3),
        "fixture input pass counts changed",
    )
    assert_no_answer_fields(config)
    return config


def load_input_config() -> tuple[dict[str, Any], str]:
    config = validate_input_config(_load_json(CONFIG_PATH))
    return config, sha256_file(CONFIG_PATH)


def _blind_shards(*, packet_id: str, seed: str) -> dict[str, int]:
    values = []
    for lane in ("independent_a", "independent_b"):
        digest = sha256_bytes(
            canonical_json_bytes(
                {
                    "domain": "fixture-corpus-input-shard-v3",
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


def _packet_ids(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    case_id: str,
    annotation_pass: str,
    serial: int,
    payload_sha256: str,
) -> tuple[str, str]:
    identity = _mapping(config["identity"], "input identity")
    packet_id = sha256_bytes(
        canonical_json_bytes(
            {
                "domain": identity["packet_id_domain"],
                "salt": identity["packet_salt"],
                "case_id": case_id,
                "annotation_pass": annotation_pass,
                "packet_serial": serial,
                "payload_sha256": payload_sha256,
            }
        )
    )
    source_id = sha256_bytes(
        canonical_json_bytes(
            {
                "domain": identity["source_id_domain"],
                "salt": identity["source_salt"],
                "packet_id_sha256": packet_id,
                "payload_sha256": payload_sha256,
                "input_config_sha256": config_sha256,
            }
        )
    )
    return packet_id, source_id


def _completion_packet(
    *, config: Mapping[str, Any], config_sha256: str, case: Mapping[str, Any], ordinal: int
) -> dict[str, Any]:
    text = str(case["assistant_text"])
    payload_hash = sha256_bytes(canonical_json_bytes({"assistant_text": text}))
    serial = int(config["identity"]["packet_serial_base"]) + ordinal
    packet_id, source_id = _packet_ids(
        config=config,
        config_sha256=config_sha256,
        case_id=str(case["case_id"]),
        annotation_pass="completion_chain",
        serial=serial,
        payload_sha256=payload_hash,
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
            "draft_payload_sha256": payload_hash,
            "packet_serial": serial,
        },
        "annotator_visibility": {
            "assistant_tool_arguments_present": False,
            "complete_prefix_text_present": False,
            "model_features_present": False,
            "repository_or_task_identity_present": False,
            "tool_results_present": False,
        },
    }
    return runner.v2.legacy.validate_packet(packet, annotation_pass="completion_chain")


def _novelty_packet(
    *, config: Mapping[str, Any], config_sha256: str, case: Mapping[str, Any], ordinal: int
) -> dict[str, Any]:
    prefix = str(case["visible_prefix"])
    hypothesis = str(case["locked_hypothesis"])
    payload = {"visible_prefix": prefix, "locked_hypothesis": hypothesis}
    payload_hash = sha256_bytes(canonical_json_bytes(payload))
    serial = int(config["identity"]["packet_serial_base"]) + ordinal
    packet_id, source_id = _packet_ids(
        config=config,
        config_sha256=config_sha256,
        case_id=str(case["case_id"]),
        annotation_pass="prefix_novelty",
        serial=serial,
        payload_sha256=payload_hash,
    )
    completion_hash = sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "fixture-corpus-locked-completion-v3",
                "packet_id_sha256": packet_id,
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
            "materialized_completion_sha256": completion_hash,
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
    path = _repo_file(_mapping(config["bindings"]["codebook_v2"], "codebook binding"), name="codebook_v2")
    try:
        return runner.v2.validate_v2_codebook(_load_json(path))
    except runner.v2.QuoteFirstRunnerError as error:
        raise FixtureCorpusInputDraftError(str(error)) from error


def _completion_projection(
    *, spec: Mapping[str, Any], authenticated_units: runner.AuthenticatedCandidateUnits
) -> dict[str, Any]:
    decision = str(spec["decision"])
    if decision == "no_chain":
        return {"decision": "no_chain"}
    if decision == "unknown":
        return {
            "decision": "unknown",
            "unknown_reason": runner.COMPLETION_UNKNOWN_REASON,
        }
    ordinals = _mapping(spec["unit_ordinals"], "chain unit ordinals")
    units = authenticated_units.units
    indexes = [int(ordinals[name]) for name in ("evidence", "hypothesis", "action")]
    _require(
        all(index < len(units) for index in indexes),
        "chain candidate ordinal exceeds authenticated catalog",
    )
    ontology = _mapping(spec["ontology"], "chain ontology")
    return {
        "decision": "chain",
        "evidence_unit_id": units[indexes[0]].unit_id,
        "hypothesis_unit_id": units[indexes[1]].unit_id,
        "action_unit_id": units[indexes[2]].unit_id,
        "evidence_kind": ontology["evidence_kind"],
        "belief_edge": ontology["belief_edge"],
        "hypothesis_domain": ontology["hypothesis_domain"],
        "action_intent": ontology["action_intent"],
    }


def _build_record(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    codebook: Mapping[str, Any],
    case: Mapping[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    annotation_pass = str(case["annotation_pass"])
    catalog_result: dict[str, Any] | None = None
    authenticated_units: runner.AuthenticatedCandidateUnits | None = None
    if annotation_pass == "completion_chain":
        packet = _completion_packet(
            config=config, config_sha256=config_sha256, case=case, ordinal=ordinal
        )
        catalog_result = catalog.build_candidate_catalog(packet=packet)
        _require(
            catalog_result.get("catalog_status") == "available"
            and catalog_result.get("catalog_usable") is True,
            f"{case['case_id']} candidate catalog unavailable",
        )
        bundle = _mapping(catalog_result["candidate_unit_bundle"], "candidate bundle")
        authenticated_units = runner.authenticate_candidate_unit_bundle(
            value=bundle,
            packet=packet,
            expected_bundle_sha256=str(catalog_result["candidate_unit_bundle_sha256"]),
        )
    else:
        packet = _novelty_packet(
            config=config, config_sha256=config_sha256, case=case, ordinal=ordinal
        )
        bundle = None

    authored = []
    for raw_spec in case["authored_candidates"]:
        spec = _mapping(raw_spec, "authored candidate spec")
        if annotation_pass == "completion_chain":
            _require(authenticated_units is not None and bundle is not None, "completion units missing")
            projection = _completion_projection(
                spec=spec, authenticated_units=authenticated_units
            )
            candidate = fixture.build_authored_fixture_candidate(
                packet=packet,
                codebook=codebook,
                annotation_pass=annotation_pass,
                projection=projection,
                candidate_unit_bundle=bundle,
                expected_candidate_unit_bundle_sha256=str(
                    catalog_result["candidate_unit_bundle_sha256"]
                ),
            )
        else:
            projection = {"decision": str(spec["decision"])}
            candidate = fixture.build_authored_fixture_candidate(
                packet=packet,
                codebook=codebook,
                annotation_pass=annotation_pass,
                projection=projection,
            )
        authored.append(candidate)
    _require(
        len(authored) == 2
        and authored[0]["projection_sha256"] != authored[1]["projection_sha256"],
        f"{case['case_id']} authored candidate projections are not distinct",
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": (
            COMPLETION_RECORD_KIND
            if annotation_pass == "completion_chain"
            else NOVELTY_RECORD_KIND
        ),
        "case_id": case["case_id"],
        "annotation_pass": annotation_pass,
        "packet": packet,
        "packet_sha256": sha256_bytes(canonical_json_bytes(packet)),
        "candidate_catalog": catalog_result,
        "candidate_catalog_sha256": (
            None
            if catalog_result is None
            else catalog.catalog_result_sha256(catalog_result)
        ),
        "candidate_unit_bundle_sha256": (
            None
            if catalog_result is None
            else catalog_result["candidate_unit_bundle_sha256"]
        ),
        "authored_candidates": authored,
        "authored_candidate_sha256s": [
            sha256_bytes(canonical_json_bytes(item)) for item in authored
        ],
    }
    record["record_sha256"] = sha256_bytes(canonical_json_bytes(record))
    assert_no_answer_fields(record)
    return record


def build_input_manifest_draft() -> dict[str, Any]:
    """Return the deterministic answer-blind draft in memory; never seal it."""

    config, config_sha256 = load_input_config()
    codebook = _load_codebook(config)
    records = [
        _build_record(
            config=config,
            config_sha256=config_sha256,
            codebook=codebook,
            case=_mapping(case, "fixture case"),
            ordinal=index,
        )
        for index, case in enumerate(config["cases"], start=1)
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": MANIFEST_KIND,
        "status": "prospective_draft_not_sealed_not_run",
        "scope": copy.deepcopy(CLAIM_SCOPE),
        "input_config_sha256": config_sha256,
        "implementation_bindings": copy.deepcopy(config["bindings"]),
        "nonce_provenance": copy.deepcopy(config["nonce_provenance"]),
        "counts": copy.deepcopy(config["counts"]),
        "records": records,
        "ordered_case_ids_sha256": sha256_bytes(
            canonical_json_bytes([record["case_id"] for record in records])
        ),
        "ordered_record_sha256s_sha256": sha256_bytes(
            canonical_json_bytes([record["record_sha256"] for record in records])
        ),
    }
    assert_no_answer_fields(manifest)
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
    }


__all__ = [
    "CONFIG_PATH",
    "FixtureCorpusInputDraftError",
    "MANIFEST_KIND",
    "assert_no_answer_fields",
    "build_input_manifest_draft",
    "canonical_json_bytes",
    "load_input_config",
    "sha256_bytes",
    "sha256_file",
    "validate_input_config",
]
