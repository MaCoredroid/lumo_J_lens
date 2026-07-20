#!/usr/bin/env python3
"""Materialize the separate semantic side of the V3 adjudication fixture draft.

The whole answer-blind input manifest is authenticated against an independently
supplied SHA-256 *before* this module reads its semantic-key config.  This module
does not import or read the input builder or its config.  After authentication,
it rebuilds catalogs and authored candidates, derives one case nonce from a
single externally supplied suite nonce, and invokes the hardened fixture lock
API.  It never calls a model-generation callback.

Candidate positions are not authored in the key.  They are derived from the
authenticated input plus the externally supplied nonce.  The draft checks for
an aggregate 4/4/4 candidate_1/candidate_2/neither presentation.  A miss makes
the entire suite unusable and explicitly forbids nonce retry after semantic-key
access.  Even a balanced result remains not seal-ready here because this inner
module cannot prove nonce-precommit or semantic-access chronology.
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
    / "swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_key_draft_v3.json"
)
CONFIG_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_key_draft_v3"
)
INPUT_MANIFEST_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_input_manifest_draft_v3"
)
KEY_MANIFEST_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_key_manifest_draft_v3"
)
COMPLETION_RECORD_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_fixture_completion_input_draft_v3"
)
NOVELTY_RECORD_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_fixture_novelty_input_draft_v3"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CASE_ID_RE = re.compile(r"^F(?:0[1-9]|1[0-2])$")
FORBIDDEN_INPUT_FIELD_FRAGMENTS = (
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
ALLOWED_INPUT_PROTOCOL_FIELD_NAMES = frozenset({"completion_chain_slots_present"})
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
TARGET_PRESENTATION_COUNTS = {
    "candidate_1": 4,
    "candidate_2": 4,
    "neither": 4,
}


class FixtureCorpusKeyDraftError(RuntimeError):
    """Raised when separation, binding, or semantic consistency fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureCorpusKeyDraftError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    _require(isinstance(value, list), f"{name} must be an array")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureCorpusKeyDraftError(f"duplicate JSON key: {key}")
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
        raise FixtureCorpusKeyDraftError(
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
        raise FixtureCorpusKeyDraftError(
            f"{name} binding must resolve inside repository"
        ) from error
    _require(resolved.is_file(), f"{name} binding is not a file")
    _require(sha256_file(resolved) == binding["sha256"], f"{name} hash changed")
    return resolved


def assert_input_semantically_blind(
    value: Any, *, path: tuple[str, ...] = ()
) -> None:
    """Reject answer channels in object names at every input-manifest depth."""

    if isinstance(value, Mapping):
        for raw_name, item in value.items():
            _require(isinstance(raw_name, str), "input field names must be strings")
            if raw_name in ALLOWED_INPUT_PROTOCOL_FIELD_NAMES:
                _require(
                    path[-1:] == ("annotator_visibility",) and item is False,
                    f"protocol field {raw_name!r} is not allowed at {'.'.join(path)}",
                )
                assert_input_semantically_blind(item, path=path + (raw_name,))
                continue
            folded = re.sub(r"[^a-z0-9]+", "_", raw_name.casefold()).strip("_")
            collapsed = folded.replace("_", "")
            for token in FORBIDDEN_INPUT_FIELD_FRAGMENTS:
                normalized = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
                _require(
                    normalized not in folded
                    and normalized.replace("_", "") not in collapsed,
                    f"forbidden semantic field at {'.'.join(path + (raw_name,))}",
                )
            assert_input_semantically_blind(item, path=path + (raw_name,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_input_semantically_blind(item, path=path + (str(index),))


def authenticate_input_manifest_before_key_read(
    *, value: Any, expected_input_manifest_sha256: str
) -> dict[str, Any]:
    """Authenticate only from supplied bytes/hash; do not touch the key config."""

    envelope = copy.deepcopy(dict(_mapping(value, "input manifest envelope")))
    _require(
        set(envelope) == {"manifest", "manifest_sha256"},
        "input manifest envelope fields invalid",
    )
    _require(
        isinstance(expected_input_manifest_sha256, str)
        and SHA256_RE.fullmatch(expected_input_manifest_sha256) is not None,
        "external input manifest hash invalid",
    )
    manifest = copy.deepcopy(dict(_mapping(envelope["manifest"], "input manifest")))
    observed = sha256_bytes(canonical_json_bytes(manifest))
    _require(
        isinstance(envelope["manifest_sha256"], str)
        and SHA256_RE.fullmatch(envelope["manifest_sha256"]) is not None
        and envelope["manifest_sha256"] == observed,
        "self-reported input manifest hash invalid",
    )
    _require(
        observed == expected_input_manifest_sha256,
        "input manifest differs from independently supplied whole-manifest hash",
    )
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("interface_version") == INTERFACE_VERSION
        and manifest.get("kind") == INPUT_MANIFEST_KIND
        and manifest.get("status") == "prospective_draft_not_sealed_not_run",
        "input manifest preliminary identity invalid",
    )
    assert_input_semantically_blind(manifest)
    return manifest


def _validate_projection_spec(value: Any, *, annotation_pass: str) -> dict[str, Any]:
    spec = copy.deepcopy(dict(_mapping(value, "semantic projection spec")))
    decision = spec.get("decision")
    if annotation_pass == "prefix_novelty":
        _require(
            set(spec) == {"decision"}
            and decision in {"novel", "prefix_exposed", "ambiguous"},
            "novelty semantic projection invalid",
        )
        return spec
    _require(annotation_pass == "completion_chain", "annotation pass invalid")
    if decision == "no_chain":
        _require(set(spec) == {"decision"}, "no_chain semantic projection invalid")
        return spec
    _require(
        decision == "chain"
        and set(spec) == {"decision", "unit_ordinals", "ontology"},
        "completion semantic projection fields invalid",
    )
    ordinals = dict(_mapping(spec["unit_ordinals"], "semantic unit ordinals"))
    _require(
        set(ordinals) == {"evidence", "hypothesis", "action"}
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in ordinals.values()
        )
        and ordinals["evidence"] < ordinals["hypothesis"] < ordinals["action"],
        "semantic unit ordinals invalid",
    )
    ontology = dict(_mapping(spec["ontology"], "semantic ontology"))
    _require(
        set(ontology)
        == {"evidence_kind", "belief_edge", "hypothesis_domain", "action_intent"}
        and all(isinstance(item, str) and bool(item) for item in ontology.values()),
        "semantic ontology invalid",
    )
    return spec


def _validate_key_config(value: Any) -> dict[str, Any]:
    config = copy.deepcopy(dict(_mapping(value, "fixture corpus key config")))
    _require(
        set(config)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "status",
            "scope",
            "external_input_authentication",
            "nonce_contract",
            "generation_seed_schedule",
            "packet_regeneration_contract",
            "counts",
            "bindings",
            "cases",
        },
        "fixture corpus key config fields invalid",
    )
    _require(
        config["schema_version"] == SCHEMA_VERSION
        and config["interface_version"] == INTERFACE_VERSION
        and config["kind"] == CONFIG_KIND
        and config["status"] == "prospective_draft_not_sealed_not_run"
        and config["scope"] == CLAIM_SCOPE,
        "fixture corpus key identity or scope invalid",
    )
    authentication = dict(
        _mapping(config["external_input_authentication"], "external input authentication")
    )
    _require(
        set(authentication)
        == {
            "whole_manifest_sha256_required_before_key_config_read",
            "self_reported_manifest_sha256_is_not_a_trust_root",
            "input_module_import_forbidden",
            "input_module_read_forbidden",
            "input_config_read_forbidden",
            "bound_input_manifest_sha256",
        }
        and authentication["whole_manifest_sha256_required_before_key_config_read"] is True
        and authentication["self_reported_manifest_sha256_is_not_a_trust_root"] is True
        and authentication["input_module_import_forbidden"] is True
        and authentication["input_module_read_forbidden"] is True
        and authentication["input_config_read_forbidden"] is True
        and SHA256_RE.fullmatch(str(authentication["bound_input_manifest_sha256"]))
        is not None,
        "external input authentication contract invalid",
    )
    nonce = dict(_mapping(config["nonce_contract"], "nonce contract"))
    _require(
        set(nonce)
        == {
            "single_externally_precommitted_suite_nonce_required",
            "per_case_nonce_input_accepted",
            "case_nonce_derivation_domain",
            "nonce_retry_after_semantic_key_access_forbidden",
            "imbalance_invalidates_entire_suite",
            "fixture_route_verifies_precommit_chronology",
            "future_sealed_outer_executor_must_establish_chronology",
            "receipt_contents_authenticated_here",
            "receipt_contents_and_issuance_authentication_responsibility",
            "single_use_nonce_or_receipt_enforced_here",
            "single_use_nonce_or_receipt_responsibility",
            "repeated_materialization_prevention_enforced_here",
            "repeated_materialization_prevention_responsibility",
        }
        and nonce["single_externally_precommitted_suite_nonce_required"] is True
        and nonce["per_case_nonce_input_accepted"] is False
        and isinstance(nonce["case_nonce_derivation_domain"], str)
        and bool(nonce["case_nonce_derivation_domain"])
        and nonce["nonce_retry_after_semantic_key_access_forbidden"] is True
        and nonce["imbalance_invalidates_entire_suite"] is True
        and nonce["fixture_route_verifies_precommit_chronology"] is False
        and nonce["future_sealed_outer_executor_must_establish_chronology"] is True
        and nonce["receipt_contents_authenticated_here"] is False
        and nonce["receipt_contents_and_issuance_authentication_responsibility"]
        == fixture.OUTER_EXECUTOR_RESPONSIBILITY
        and nonce["single_use_nonce_or_receipt_enforced_here"] is False
        and nonce["single_use_nonce_or_receipt_responsibility"]
        == fixture.OUTER_EXECUTOR_RESPONSIBILITY
        and nonce["repeated_materialization_prevention_enforced_here"] is False
        and nonce["repeated_materialization_prevention_responsibility"]
        == fixture.OUTER_EXECUTOR_RESPONSIBILITY,
        "nonce contract invalid",
    )
    seed_schedule = dict(
        _mapping(config["generation_seed_schedule"], "generation seed schedule")
    )
    _require(
        set(seed_schedule) == {"kind", "derivation", "cases"}
        and seed_schedule["kind"]
        == "exact_per_case_verdict_and_repair_seed_schedule_v3"
        and seed_schedule["derivation"]
        == "frozen_explicit_values_not_caller_authoritative",
        "generation seed schedule identity invalid",
    )
    seed_rows = [
        dict(_mapping(item, "generation seed row"))
        for item in _sequence(seed_schedule["cases"], "generation seed rows")
    ]
    _require(len(seed_rows) == 12, "generation seed schedule count changed")
    for index, row in enumerate(seed_rows, start=1):
        _require(
            set(row) == {"case_id", "verdict_seed", "repair_seed"}
            and row["case_id"] == f"F{index:02d}"
            and isinstance(row["verdict_seed"], int)
            and not isinstance(row["verdict_seed"], bool)
            and row["verdict_seed"] == 1000 + index
            and isinstance(row["repair_seed"], int)
            and not isinstance(row["repair_seed"], bool)
            and row["repair_seed"] == 2000 + index,
            "generation seed schedule values changed",
        )
    packet_contract = dict(
        _mapping(config["packet_regeneration_contract"], "packet regeneration contract")
    )
    _require(
        packet_contract
        == {
            "key_materializer_rebuilds_packet_ids_from_raw_identity": False,
            "exact_external_input_manifest_and_record_hashes_bound": True,
            "production_packet_and_catalog_validation_performed": True,
            "authoritative_packet_regeneration_responsibility": (
                fixture.OUTER_EXECUTOR_RESPONSIBILITY
            ),
        },
        "packet regeneration responsibility invalid",
    )
    _require(
        config["counts"]
        == {
            "completion": 9,
            "novelty": 3,
            "total": 12,
            "direct": 8,
            "neither": 4,
            "repair_decision": 4,
            "completion_chain_detail_repair": 3,
            "intended_candidate_1": 4,
            "intended_candidate_2": 4,
            "intended_neither": 4,
        },
        "fixture corpus key counts changed",
    )
    bindings = dict(_mapping(config["bindings"], "key bindings"))
    expected_names = {
        "key_materializer": Path(__file__).name,
        "input_builder_declaration": (
            "swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_inputs_draft_v3.py"
        ),
        "fixture_config": fixture.DEFAULT_CONFIG_PATH.name,
        "fixture_protocol": Path(fixture.__file__).name,
        "candidate_catalog_config": catalog.CONFIG_PATH.name,
        "candidate_catalog_builder": Path(catalog.__file__).name,
        "runner_v3": Path(runner.__file__).name,
        "codebook_v2": "swe_task_state_v4_epistemic_chain_codebook_v2.json",
    }
    _require(set(bindings) == set(expected_names), "key binding names changed")
    for name, expected_name in expected_names.items():
        binding = dict(_mapping(bindings[name], f"key binding {name}"))
        _require(
            set(binding) == {"path", "sha256"}
            and isinstance(binding["path"], str)
            and Path(binding["path"]).name == expected_name
            and SHA256_RE.fullmatch(str(binding["sha256"])) is not None,
            f"key binding {name} invalid",
        )
        # The declaration is compared to authenticated manifest metadata only.
        # Reading the input builder source from the semantic side is forbidden.
        if name != "input_builder_declaration":
            _repo_file(binding, name=name)

    cases = [dict(_mapping(item, "semantic key case")) for item in _sequence(config["cases"], "semantic key cases")]
    _require(len(cases) == 12, "semantic key case count changed")
    route_counts = {"direct": 0, "neither": 0}
    completion = 0
    novelty = 0
    repair_count = 0
    chain_detail_count = 0
    for index, case in enumerate(cases, start=1):
        _require(
            set(case)
            == {
                "case_id",
                "annotation_pass",
                "route_requirement",
                "semantic_projection",
                "repair_decision",
                "diagnostic",
            }
            and case["case_id"] == f"F{index:02d}"
            and CASE_ID_RE.fullmatch(str(case["case_id"])) is not None
            and case["annotation_pass"] in runner.PASSES
            and case["route_requirement"] in route_counts
            and isinstance(case["diagnostic"], str)
            and bool(case["diagnostic"]),
            "semantic key case shape or order invalid",
        )
        projection = _validate_projection_spec(
            case["semantic_projection"], annotation_pass=str(case["annotation_pass"])
        )
        route_counts[str(case["route_requirement"])] += 1
        if case["annotation_pass"] == "completion_chain":
            completion += 1
        else:
            novelty += 1
        if case["route_requirement"] == "direct":
            _require(case["repair_decision"] is None, "direct case must not repair")
        else:
            repair_count += 1
            _require(
                case["repair_decision"] == projection["decision"],
                "neither repair decision differs from semantic projection",
            )
            if case["annotation_pass"] == "completion_chain":
                _require(projection["decision"] == "chain", "completion repair must open chain detail")
                chain_detail_count += 1
    _require(
        (completion, novelty) == (9, 3)
        and route_counts == {"direct": 8, "neither": 4}
        and repair_count == 4
        and chain_detail_count == 3,
        "semantic key route coverage changed",
    )
    return config


def _read_key_config_source() -> tuple[Any, bytes]:
    """Read the fixed source once so its exact bytes and JSON cannot diverge."""

    try:
        raw = CONFIG_PATH.read_bytes()
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixtureCorpusKeyDraftError(
            f"cannot load exact key config source: {error}"
        ) from error
    return value, raw


def authenticate_key_config_after_input_auth(
    *, expected_key_config_sha256: str
) -> tuple[dict[str, Any], str]:
    """Authenticate exact fixed-source bytes against an independent hash."""

    _require(
        isinstance(expected_key_config_sha256, str)
        and SHA256_RE.fullmatch(expected_key_config_sha256) is not None,
        "external key config hash invalid",
    )
    value, raw = _read_key_config_source()
    observed = sha256_bytes(raw)
    _require(
        observed == expected_key_config_sha256,
        "key config differs from independently supplied exact-byte hash",
    )
    return _validate_key_config(value), observed


def _load_codebook(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _repo_file(
        _mapping(config["bindings"]["codebook_v2"], "codebook binding"),
        name="codebook_v2",
    )
    try:
        return runner.v2.validate_v2_codebook(_load_json(path))
    except runner.v2.QuoteFirstRunnerError as error:
        raise FixtureCorpusKeyDraftError(str(error)) from error


def _record_without_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(record))
    value.pop("record_sha256", None)
    return value


def _validate_authenticated_manifest(
    *, manifest: Mapping[str, Any], manifest_sha256: str, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
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
            "nonce_provenance",
            "counts",
            "records",
            "ordered_case_ids_sha256",
            "ordered_record_sha256s_sha256",
        }
        and manifest["scope"] == CLAIM_SCOPE
        and SHA256_RE.fullmatch(str(manifest["input_config_sha256"])) is not None
        and manifest["counts"] == {"completion": 9, "novelty": 3, "total": 12},
        "authenticated input manifest exact shape invalid",
    )
    bound_hash = config["external_input_authentication"]["bound_input_manifest_sha256"]
    _require(
        manifest_sha256 == bound_hash,
        "co-tampered input manifest differs from key-bound manifest hash",
    )
    input_bindings = dict(
        _mapping(manifest["implementation_bindings"], "input implementation bindings")
    )
    _require(
        set(input_bindings)
        == {
            "input_builder",
            "fixture_config",
            "fixture_protocol",
            "candidate_catalog_config",
            "candidate_catalog_builder",
            "runner_v3",
            "codebook_v2",
        },
        "input implementation binding names invalid",
    )
    _require(
        input_bindings["input_builder"]
        == config["bindings"]["input_builder_declaration"],
        "input builder declaration differs from authenticated manifest",
    )
    for name in (
        "fixture_config",
        "fixture_protocol",
        "candidate_catalog_config",
        "candidate_catalog_builder",
        "runner_v3",
        "codebook_v2",
    ):
        _require(
            input_bindings[name] == config["bindings"][name],
            f"input/core binding {name} differs from semantic key",
        )
    provenance = dict(_mapping(manifest["nonce_provenance"], "input nonce provenance"))
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
        and provenance.get("status") == "external_precommit_pending"
        and provenance.get("actual_fixture_nonce_present") is False
        and provenance.get("precommit_chronology_claimed") is False
        and provenance.get("semantic_access_chronology_claimed") is False
        and provenance.get("chronology_responsibility")
        == fixture.OUTER_EXECUTOR_RESPONSIBILITY
        and isinstance(provenance.get("placeholder_receipt_reference"), str)
        and provenance.get("placeholder_receipt_reference_sha256")
        == sha256_text(str(provenance.get("placeholder_receipt_reference"))),
        "input nonce placeholder provenance invalid",
    )
    records = [dict(_mapping(item, "input fixture record")) for item in _sequence(manifest["records"], "input fixture records")]
    _require(len(records) == 12, "input fixture record count changed")
    _require(
        manifest["ordered_case_ids_sha256"]
        == sha256_bytes(canonical_json_bytes([item.get("case_id") for item in records]))
        and manifest["ordered_record_sha256s_sha256"]
        == sha256_bytes(canonical_json_bytes([item.get("record_sha256") for item in records])),
        "input fixture order binding invalid",
    )
    return records


def _authenticate_record(
    *, record: Mapping[str, Any], expected_case_id: str, codebook: Mapping[str, Any]
) -> tuple[dict[str, Any], runner.AuthenticatedCandidateUnits | None]:
    value = copy.deepcopy(dict(record))
    _require(
        set(value)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "case_id",
            "annotation_pass",
            "packet",
            "packet_sha256",
            "candidate_catalog",
            "candidate_catalog_sha256",
            "candidate_unit_bundle_sha256",
            "authored_candidates",
            "authored_candidate_sha256s",
            "record_sha256",
        }
        and value["schema_version"] == SCHEMA_VERSION
        and value["interface_version"] == INTERFACE_VERSION
        and value["case_id"] == expected_case_id
        and value["annotation_pass"] in runner.PASSES,
        f"{expected_case_id} input record exact shape invalid",
    )
    annotation_pass = str(value["annotation_pass"])
    expected_kind = (
        COMPLETION_RECORD_KIND
        if annotation_pass == "completion_chain"
        else NOVELTY_RECORD_KIND
    )
    _require(value["kind"] == expected_kind, f"{expected_case_id} record kind invalid")
    packet = fixture._validated_packet(
        _mapping(value["packet"], "input packet"), annotation_pass=annotation_pass
    )
    _require(
        value["packet_sha256"] == sha256_bytes(canonical_json_bytes(packet))
        and value["record_sha256"]
        == sha256_bytes(canonical_json_bytes(_record_without_hash(value))),
        f"{expected_case_id} packet or record hash invalid",
    )
    authenticated_units: runner.AuthenticatedCandidateUnits | None = None
    bundle: Mapping[str, Any] | None = None
    if annotation_pass == "completion_chain":
        supplied_catalog = dict(_mapping(value["candidate_catalog"], "candidate catalog"))
        rebuilt_catalog = catalog.build_candidate_catalog(packet=packet)
        _require(
            supplied_catalog == rebuilt_catalog
            and value["candidate_catalog_sha256"]
            == catalog.catalog_result_sha256(rebuilt_catalog)
            and value["candidate_unit_bundle_sha256"]
            == rebuilt_catalog["candidate_unit_bundle_sha256"],
            f"{expected_case_id} catalog differs from authenticated rebuild",
        )
        bundle = _mapping(rebuilt_catalog["candidate_unit_bundle"], "candidate unit bundle")
        authenticated_units = runner.authenticate_candidate_unit_bundle(
            value=bundle,
            packet=packet,
            expected_bundle_sha256=str(value["candidate_unit_bundle_sha256"]),
        )
    else:
        _require(
            value["candidate_catalog"] is None
            and value["candidate_catalog_sha256"] is None
            and value["candidate_unit_bundle_sha256"] is None,
            f"{expected_case_id} novelty record contains completion catalog",
        )
    candidates = [dict(_mapping(item, "authored fixture candidate")) for item in _sequence(value["authored_candidates"], "authored fixture candidates")]
    hashes = list(_sequence(value["authored_candidate_sha256s"], "authored candidate hashes"))
    _require(len(candidates) == len(hashes) == 2, f"{expected_case_id} candidate count invalid")
    rebuilt_candidates = []
    for candidate in candidates:
        if annotation_pass == "completion_chain":
            _require(bundle is not None, "completion bundle missing")
            rebuilt = fixture.build_authored_fixture_candidate(
                packet=packet,
                codebook=codebook,
                annotation_pass=annotation_pass,
                projection=_mapping(candidate.get("projection"), "candidate projection"),
                candidate_unit_bundle=bundle,
                expected_candidate_unit_bundle_sha256=str(value["candidate_unit_bundle_sha256"]),
            )
        else:
            rebuilt = fixture.build_authored_fixture_candidate(
                packet=packet,
                codebook=codebook,
                annotation_pass=annotation_pass,
                projection=_mapping(candidate.get("projection"), "candidate projection"),
            )
        rebuilt_candidates.append(rebuilt)
    _require(
        candidates == rebuilt_candidates
        and hashes == [sha256_bytes(canonical_json_bytes(item)) for item in candidates]
        and candidates[0]["projection_sha256"] != candidates[1]["projection_sha256"],
        f"{expected_case_id} authored candidate binding invalid",
    )
    return value, authenticated_units


def _semantic_projection(
    *,
    spec: Mapping[str, Any],
    annotation_pass: str,
    codebook: Mapping[str, Any],
    authenticated_units: runner.AuthenticatedCandidateUnits | None,
) -> dict[str, Any]:
    if annotation_pass == "prefix_novelty":
        return runner.validate_novelty_proposal({"decision": str(spec["decision"])})
    _require(authenticated_units is not None, "completion semantic units missing")
    if spec["decision"] == "no_chain":
        return runner.validate_completion_proposal(
            {"decision": "no_chain"},
            codebook=codebook,
            authenticated_units=authenticated_units,
        )
    ordinals = _mapping(spec["unit_ordinals"], "semantic unit ordinals")
    indexes = [int(ordinals[name]) for name in ("evidence", "hypothesis", "action")]
    _require(
        all(index < len(authenticated_units.units) for index in indexes),
        "semantic unit ordinal exceeds authenticated catalog",
    )
    ontology = _mapping(spec["ontology"], "semantic ontology")
    return runner.validate_completion_proposal(
        {
            "decision": "chain",
            "evidence_unit_id": authenticated_units.units[indexes[0]].unit_id,
            "hypothesis_unit_id": authenticated_units.units[indexes[1]].unit_id,
            "action_unit_id": authenticated_units.units[indexes[2]].unit_id,
            "evidence_kind": ontology["evidence_kind"],
            "belief_edge": ontology["belief_edge"],
            "hypothesis_domain": ontology["hypothesis_domain"],
            "action_intent": ontology["action_intent"],
        },
        codebook=codebook,
        authenticated_units=authenticated_units,
    )


def derive_case_nonce_sha256(
    *, suite_nonce_sha256: str, case_id: str, input_manifest_sha256: str, domain: str
) -> str:
    _require(
        SHA256_RE.fullmatch(suite_nonce_sha256) is not None
        and CASE_ID_RE.fullmatch(case_id) is not None
        and SHA256_RE.fullmatch(input_manifest_sha256) is not None
        and isinstance(domain, str)
        and bool(domain),
        "case nonce derivation input invalid",
    )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": domain,
                "suite_nonce_sha256": suite_nonce_sha256,
                "case_id": case_id,
                "input_manifest_sha256": input_manifest_sha256,
            }
        )
    )


def validate_presentation_balance(classes: Sequence[str]) -> dict[str, Any]:
    values = list(classes)
    _require(
        len(values) == 12
        and all(item in TARGET_PRESENTATION_COUNTS for item in values),
        "presentation classes invalid",
    )
    observed = {name: values.count(name) for name in TARGET_PRESENTATION_COUNTS}
    return {
        "intended_counts": copy.deepcopy(TARGET_PRESENTATION_COUNTS),
        "observed_counts": observed,
        "balance_satisfied": observed == TARGET_PRESENTATION_COUNTS,
        "nonce_retry_permitted": False,
        "imbalance_disposition": "invalidate_entire_suite_and_start_new_predeclared_round",
    }


def materialize_key_draft(
    *,
    input_manifest_envelope: Mapping[str, Any],
    expected_input_manifest_sha256: str,
    expected_key_config_sha256: str,
    externally_precommitted_suite_nonce_sha256: str,
    outer_nonce_precommit_receipt_sha256: str,
    fixture_config: Mapping[str, Any],
    expected_fixture_config_sha256: str,
    generation_context: runner.AuthenticatedNativeGenerationContext,
    generation_contracts: Mapping[str, Mapping[str, Any]],
    expected_generation_contract_sha256s: Mapping[str, str],
    verdict_seeds: Mapping[str, int],
    repair_seeds: Mapping[str, int],
) -> dict[str, Any]:
    """Build a non-executed key/lock draft after external input authentication."""

    # Input authentication must be the first operation.  Only after it passes
    # may exact fixed-source key-config bytes be read and authenticated against
    # a second independently supplied trust root.
    manifest = authenticate_input_manifest_before_key_read(
        value=input_manifest_envelope,
        expected_input_manifest_sha256=expected_input_manifest_sha256,
    )
    config, key_config_sha256 = authenticate_key_config_after_input_auth(
        expected_key_config_sha256=expected_key_config_sha256
    )
    records = _validate_authenticated_manifest(
        manifest=manifest,
        manifest_sha256=expected_input_manifest_sha256,
        config=config,
    )
    _require(
        SHA256_RE.fullmatch(externally_precommitted_suite_nonce_sha256) is not None
        and SHA256_RE.fullmatch(outer_nonce_precommit_receipt_sha256) is not None,
        "external suite nonce or receipt reference invalid",
    )
    fixture.authenticate_fixture_config(
        value=fixture_config,
        expected_config_sha256=expected_fixture_config_sha256,
    )
    _require(
        expected_fixture_config_sha256 == config["bindings"]["fixture_config"]["sha256"],
        "fixture config differs from semantic-key binding",
    )
    _require(
        isinstance(generation_context, runner.AuthenticatedNativeGenerationContext),
        "generation context is not authenticated",
    )
    case_ids = [f"F{index:02d}" for index in range(1, 13)]
    supplied_maps = {
        "generation contracts": generation_contracts,
        "generation contract hashes": expected_generation_contract_sha256s,
        "verdict seeds": verdict_seeds,
        "repair seeds": repair_seeds,
    }
    for name, supplied in supplied_maps.items():
        _require(
            isinstance(supplied, Mapping) and set(supplied) == set(case_ids),
            f"{name} case coverage invalid",
        )
    frozen_seed_rows = {
        str(row["case_id"]): {
            "verdict_seed": int(row["verdict_seed"]),
            "repair_seed": int(row["repair_seed"]),
        }
        for row in config["generation_seed_schedule"]["cases"]
    }
    frozen_verdict_seeds = {
        case_id: row["verdict_seed"] for case_id, row in frozen_seed_rows.items()
    }
    frozen_repair_seeds = {
        case_id: row["repair_seed"] for case_id, row in frozen_seed_rows.items()
    }
    _require(
        dict(verdict_seeds) == frozen_verdict_seeds
        and dict(repair_seeds) == frozen_repair_seeds,
        "caller-supplied seeds differ from externally authenticated frozen schedule",
    )
    seed_schedule_sha256 = sha256_bytes(
        canonical_json_bytes(config["generation_seed_schedule"])
    )
    codebook = _load_codebook(config)
    key_rows = [dict(_mapping(item, "semantic key row")) for item in config["cases"]]
    output_rows = []
    realized_classes = []
    derivation_domain = str(config["nonce_contract"]["case_nonce_derivation_domain"])
    for record, key_row, case_id in zip(records, key_rows, case_ids, strict=True):
        _require(
            key_row["case_id"] == case_id
            and record["case_id"] == case_id
            and key_row["annotation_pass"] == record["annotation_pass"],
            "input/key case identity or order mismatch",
        )
        authenticated_record, authenticated_units = _authenticate_record(
            record=record, expected_case_id=case_id, codebook=codebook
        )
        annotation_pass = str(authenticated_record["annotation_pass"])
        projection_spec = _validate_projection_spec(
            key_row["semantic_projection"], annotation_pass=annotation_pass
        )
        semantic_projection = _semantic_projection(
            spec=projection_spec,
            annotation_pass=annotation_pass,
            codebook=codebook,
            authenticated_units=authenticated_units,
        )
        semantic_projection_sha256 = sha256_bytes(
            canonical_json_bytes(semantic_projection)
        )
        original_projection_hashes = [
            str(item["projection_sha256"])
            for item in authenticated_record["authored_candidates"]
        ]
        route_requirement = str(key_row["route_requirement"])
        if route_requirement == "direct":
            _require(
                original_projection_hashes.count(semantic_projection_sha256) == 1,
                f"{case_id} direct semantic projection not present exactly once",
            )
        else:
            _require(
                semantic_projection_sha256 not in original_projection_hashes,
                f"{case_id} neither semantic projection appears among candidates",
            )
        case_nonce = derive_case_nonce_sha256(
            suite_nonce_sha256=externally_precommitted_suite_nonce_sha256,
            case_id=case_id,
            input_manifest_sha256=expected_input_manifest_sha256,
            domain=derivation_domain,
        )
        contract = _mapping(generation_contracts[case_id], "generation contract")
        contract_hash = str(expected_generation_contract_sha256s[case_id])
        verdict_seed = verdict_seeds[case_id]
        repair_seed = repair_seeds[case_id]
        _require(
            isinstance(verdict_seed, int)
            and not isinstance(verdict_seed, bool)
            and isinstance(repair_seed, int)
            and not isinstance(repair_seed, bool),
            f"{case_id} generation seeds invalid",
        )
        bundle = (
            None
            if annotation_pass == "prefix_novelty"
            else authenticated_record["candidate_catalog"]["candidate_unit_bundle"]
        )
        bundle_hash = (
            None
            if annotation_pass == "prefix_novelty"
            else str(authenticated_record["candidate_unit_bundle_sha256"])
        )
        lock = fixture.build_fixture_adjudication_lock(
            packet=authenticated_record["packet"],
            codebook=codebook,
            annotation_pass=annotation_pass,
            fixture_candidates=tuple(authenticated_record["authored_candidates"]),
            fixture_nonce_sha256=case_nonce,
            fixture_config=fixture_config,
            expected_fixture_config_sha256=expected_fixture_config_sha256,
            generation_contract=contract,
            expected_generation_contract_sha256=contract_hash,
            generation_context=generation_context,
            expected_verdict_seed=int(verdict_seed),
            expected_repair_seed=int(repair_seed),
            candidate_unit_bundle=bundle,
            expected_candidate_unit_bundle_sha256=bundle_hash,
        )
        _require(
            lock["outer_nonce_precommit_receipt_sha256"]
            == outer_nonce_precommit_receipt_sha256
            and lock["inner_route_claims_actual_model_execution"] is False
            and lock["precommit_chronology_verified_by_fixture_route"] is False
            and lock["expectation_access_chronology_verified_by_fixture_route"] is False,
            f"{case_id} receipt binding or chronology scope invalid",
        )
        if route_requirement == "neither":
            realized = "neither"
        else:
            blinded_hashes = list(lock["blinded_projection_sha256s"])
            _require(
                blinded_hashes.count(semantic_projection_sha256) == 1,
                f"{case_id} semantic projection lost during blinding",
            )
            realized = (
                "candidate_1"
                if blinded_hashes[0] == semantic_projection_sha256
                else "candidate_2"
            )
        realized_classes.append(realized)
        output_rows.append(
            {
                "case_id": case_id,
                "annotation_pass": annotation_pass,
                "route_requirement": route_requirement,
                "semantic_projection": semantic_projection,
                "semantic_projection_sha256": semantic_projection_sha256,
                "repair_decision": key_row["repair_decision"],
                "completion_chain_detail_repair_required": (
                    route_requirement == "neither"
                    and annotation_pass == "completion_chain"
                    and semantic_projection["decision"] == "chain"
                ),
                "diagnostic": key_row["diagnostic"],
                "realized_presentation_class": realized,
                "fixture_nonce_sha256": case_nonce,
                "verdict_seed": int(verdict_seed),
                "repair_seed": int(repair_seed),
                "generation_seed_schedule_sha256": seed_schedule_sha256,
                "generation_contract_sha256": contract_hash,
                "fixture_lock": lock,
                "fixture_lock_sha256": fixture.fixture_adjudication_lock_sha256(lock),
            }
        )
    balance = validate_presentation_balance(realized_classes)
    status = (
        "prospective_draft_balance_observed_chronology_unverified_not_seal_ready"
        if balance["balance_satisfied"]
        else "prospective_draft_unusable_presentation_imbalance_no_retry"
    )
    key_manifest = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": KEY_MANIFEST_KIND,
        "status": status,
        "scope": copy.deepcopy(CLAIM_SCOPE),
        "input_manifest_sha256": expected_input_manifest_sha256,
        "key_config_sha256": key_config_sha256,
        "externally_precommitted_suite_nonce_sha256": (
            externally_precommitted_suite_nonce_sha256
        ),
        "outer_nonce_precommit_receipt_sha256": (
            outer_nonce_precommit_receipt_sha256
        ),
        "outer_nonce_precommit_receipt_contents_authenticated": False,
        "receipt_contents_and_issuance_authentication_responsibility": (
            fixture.OUTER_EXECUTOR_RESPONSIBILITY
        ),
        "single_use_nonce_or_receipt_enforced_by_key_materializer": False,
        "single_use_nonce_or_receipt_responsibility": (
            fixture.OUTER_EXECUTOR_RESPONSIBILITY
        ),
        "precommit_chronology_verified_by_key_materializer": False,
        "semantic_access_chronology_verified_by_key_materializer": False,
        "repeated_materialization_prevention_enforced_by_key_materializer": False,
        "repeated_materialization_prevention_responsibility": (
            fixture.OUTER_EXECUTOR_RESPONSIBILITY
        ),
        "authoritative_packet_regeneration_performed_by_key_materializer": False,
        "authoritative_packet_regeneration_responsibility": (
            fixture.OUTER_EXECUTOR_RESPONSIBILITY
        ),
        "actual_model_execution_claimed": False,
        "generation_callback_invoked": False,
        "seal_ready": False,
        "nonce_retry_permitted": False,
        "presentation_balance": balance,
        "generation_seed_schedule": copy.deepcopy(
            config["generation_seed_schedule"]
        ),
        "generation_seed_schedule_sha256": seed_schedule_sha256,
        "counts": copy.deepcopy(config["counts"]),
        "cases": output_rows,
        "ordered_fixture_lock_sha256s_sha256": sha256_bytes(
            canonical_json_bytes([row["fixture_lock_sha256"] for row in output_rows])
        ),
    }
    return {
        "manifest": key_manifest,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(key_manifest)),
    }


__all__ = [
    "CONFIG_PATH",
    "FixtureCorpusKeyDraftError",
    "TARGET_PRESENTATION_COUNTS",
    "assert_input_semantically_blind",
    "authenticate_input_manifest_before_key_read",
    "authenticate_key_config_after_input_auth",
    "canonical_json_bytes",
    "derive_case_nonce_sha256",
    "materialize_key_draft",
    "sha256_bytes",
    "sha256_file",
    "validate_presentation_balance",
]
