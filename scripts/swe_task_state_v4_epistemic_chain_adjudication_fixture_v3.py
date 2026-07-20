#!/usr/bin/env python3
"""Controls-only V3 adjudication route for authored bounded fixtures.

This additive module exists solely to exercise the audited V3 semantic
adjudication interface with authored control projections.  Fixture candidates
are explicitly *not* independent model records and this module never fabricates
their native-generation lineage.  Only the adjudicator/repair requests use the
production native-token request/result path.

The exact model input is built by the production V3 runner.  It contains only
the packet-visible data, opaque unit IDs with exact visible text, and two
unlabeled finite projections.  Gold/wrong/expected/winner/position-class data
and all expectation objects remain physically outside this API.

This inner route records only a native-protocol-adapter execution and always
keeps the actual-model-execution claim false.  A caller-authenticated generation
contract and fixture lock bind identities, runtime/adapter hashes, seeds, nonce
precommit references, and exact native prompt IDs.  The route rebuilds every
production request before execution and during record validation.  It does not
prove precommit or expectation-access chronology; a future sealed outer
executor must establish those facts and any actual model-run claim.

The scientific scope is visible semantic COT-like structure.  No private or
verbatim chain-of-thought recovery, ground-truth reasoning access, affect,
emotion, confidence, doubt, or stress claim is made here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "swe_task_state_v4_epistemic_chain_adjudication_fixture_v3.json"
)
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import swe_task_state_v4_epistemic_chain_annotation_runner_v3 as production  # noqa: E402


SCHEMA_VERSION = 1
INTERFACE_VERSION = 3
CONFIG_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_control_fixture_config_v3"
)
FIXTURE_CANDIDATE_KIND = (
    "swe_task_state_v4_epistemic_chain_authored_adjudication_fixture_candidate_v3"
)
FIXTURE_LOCK_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_control_fixture_lock_v3"
)
FIXTURE_RECORD_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_control_fixture_record_v3"
)
GENERATION_CONTRACT_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_fixture_generation_contract_v3"
)
FIXTURE_ROLE = "control_fixture_adjudicator"
INNER_EXECUTION_MODE = "native_protocol_adapter_non_gate"
OUTER_EXECUTOR_RESPONSIBILITY = "future_sealed_outer_executor"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FIXTURE_PROVENANCE = {
    "origin": "authored_bounded_control_fixture_projection",
    "intended_use": "sealed_semantic_adjudication_control_fixture_only",
    "model_generated": False,
    "native_model_record_lineage_present": False,
    "production_candidate_record_kind_claimed": False,
}
CLAIM_SCOPE = copy.deepcopy(production.CLAIM_SCOPE)
REQUIRED_FORBIDDEN_FIELD_FRAGMENTS = frozenset(
    {
        "gold",
        "wrong",
        "expected",
        "winner",
        "position_class",
        "positionclass",
        "correct",
        "incorrect",
        "preferred",
        "preference",
        "label",
        "target",
        "expectation",
    }
)
MATERIALIZATION_FIELDS = (
    "raw_semantic_decision",
    "raw_semantic_proposal",
    "decision_source",
    "semantic_validation_status",
    "semantic_validation_error",
    "materialization_status",
    "interface_unknown_reason",
    "annotation_record",
)


class FixtureRouteError(RuntimeError):
    """Raised when fixture provenance, blinding, or a lock fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureRouteError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be an array",
    )
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return production.canonical_json_bytes(value)


def canonical_json_text(value: Any) -> str:
    return production.canonical_json_text(value)


def sha256_bytes(value: bytes) -> str:
    return production.sha256_bytes(value)


def sha256_text(value: str) -> str:
    return production.sha256_text(value)


def _file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def fixture_module_sha256() -> str:
    """Hash the exact controls-only implementation used for a fixture lock."""

    return _file_sha256(Path(__file__).resolve())


def production_runner_sha256() -> str:
    return _file_sha256(Path(production.__file__).resolve())


@dataclass(frozen=True)
class AuthenticatedFixtureConfig:
    body: Mapping[str, Any]
    config_sha256: str
    production_runner_sha256: str
    codebook_sha256: str
    codebook_source_sha256: str
    forbidden_field_fragments: tuple[str, ...]
    blinding_domain: str


def authenticate_fixture_config(
    *, value: Any, expected_config_sha256: str
) -> AuthenticatedFixtureConfig:
    """Authenticate the frozen fixture contract and its production dependencies."""

    config = copy.deepcopy(dict(_mapping(value, "fixture config")))
    expected_fields = {
        "schema_version",
        "interface_version",
        "kind",
        "scope",
        "production_runner",
        "production_codebook",
        "fixture_candidate_provenance",
        "blinding",
        "model_input",
        "generation",
        "outer_executor_responsibility",
        "claim_scope",
    }
    _require(set(config) == expected_fields, "fixture config fields invalid")
    _require(DEFAULT_CONFIG_PATH.is_file(), "fixture config source file missing")
    try:
        source_config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FixtureRouteError(f"fixture config source JSON invalid: {error}") from error
    _require(
        source_config == config,
        "fixture config object differs from the controls-module source file",
    )
    observed_config_hash = _file_sha256(DEFAULT_CONFIG_PATH)
    _require(
        SHA256_RE.fullmatch(expected_config_sha256) is not None
        and observed_config_hash == expected_config_sha256,
        "fixture config differs from out-of-band authenticated hash",
    )
    _require(
        config["schema_version"] == SCHEMA_VERSION
        and config["interface_version"] == INTERFACE_VERSION
        and config["kind"] == CONFIG_KIND
        and config["scope"] == "controls_module_only_cpu_testable_fixture_route",
        "fixture config identity invalid",
    )

    runner_config = dict(_mapping(config["production_runner"], "production runner"))
    _require(
        set(runner_config) == {"kind", "relative_path", "sha256"}
        and runner_config["kind"] == production.RUNNER_KIND
        and runner_config["relative_path"]
        == "scripts/swe_task_state_v4_epistemic_chain_annotation_runner_v3.py"
        and SHA256_RE.fullmatch(str(runner_config["sha256"])) is not None,
        "production runner config invalid",
    )
    configured_runner_path = (
        REPOSITORY_ROOT / str(runner_config["relative_path"])
    ).resolve()
    _require(
        configured_runner_path == Path(production.__file__).resolve()
        and _file_sha256(configured_runner_path) == runner_config["sha256"],
        "production runner source differs from fixture config",
    )

    codebook_config = dict(
        _mapping(config["production_codebook"], "production codebook")
    )
    _require(
        set(codebook_config)
        == {"relative_path", "file_sha256", "canonical_json_sha256"}
        and codebook_config["relative_path"]
        == "configs/swe_task_state_v4_epistemic_chain_codebook_v2.json"
        and SHA256_RE.fullmatch(str(codebook_config["file_sha256"])) is not None
        and SHA256_RE.fullmatch(
            str(codebook_config["canonical_json_sha256"])
        )
        is not None,
        "production codebook config invalid",
    )
    codebook_path = (
        REPOSITORY_ROOT / str(codebook_config["relative_path"])
    ).resolve()
    try:
        source_codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FixtureRouteError(f"production codebook JSON invalid: {error}") from error
    _require(
        _file_sha256(codebook_path) == codebook_config["file_sha256"]
        and sha256_bytes(canonical_json_bytes(source_codebook))
        == codebook_config["canonical_json_sha256"],
        "production codebook file differs from fixture config",
    )
    _require(
        config["fixture_candidate_provenance"] == FIXTURE_PROVENANCE,
        "fixture candidate provenance contract invalid",
    )

    blinding = dict(_mapping(config["blinding"], "blinding config"))
    _require(
        set(blinding)
        == {
            "domain",
            "nonce_format",
            "nonce_model_visible",
            "canonical_projection_sort_before_nonce_selector",
            "original_order_bound",
        }
        and isinstance(blinding["domain"], str)
        and bool(blinding["domain"])
        and blinding["nonce_format"] == "lowercase_sha256_hex"
        and blinding["nonce_model_visible"] is False
        and blinding["canonical_projection_sort_before_nonce_selector"] is True
        and blinding["original_order_bound"] is True,
        "fixture blinding contract invalid",
    )

    model_input = dict(_mapping(config["model_input"], "model-input config"))
    _require(
        set(model_input)
        == {
            "builder",
            "unlabeled",
            "forbidden_field_name_fragments",
            "expectation_object_accepted_by_generation_api",
            "expectation_object_in_fixture_lock_or_record",
        }
        and model_input["builder"]
        == "production_runner.build_adjudication_messages"
        and model_input["unlabeled"] is True
        and model_input["expectation_object_accepted_by_generation_api"] is False
        and model_input["expectation_object_in_fixture_lock_or_record"] is False,
        "fixture model-input contract invalid",
    )
    forbidden = list(
        _sequence(
            model_input["forbidden_field_name_fragments"],
            "forbidden model-input field fragments",
        )
    )
    _require(
        all(isinstance(item, str) and bool(item) for item in forbidden)
        and len(forbidden) == len(set(forbidden))
        and set(forbidden) == REQUIRED_FORBIDDEN_FIELD_FRAGMENTS,
        "forbidden model-input field coverage invalid",
    )

    generation = dict(_mapping(config["generation"], "generation config"))
    _require(
        generation
        == {
            "native_execution": "production_runner.execute_native_generation",
            "native_request_reconstruction": (
                "production_runner.build_native_generation_request"
            ),
            "verdict_schema": "production_runner.adjudication_response_schema",
            "repair_schema": "production_runner.neither_repair_response_schema",
            "materialization": (
                "production_runner.materialize_completion_proposal_or_"
                "materialize_novelty_proposal"
            ),
            "cpu_mock_execution_is_non_gate": True,
            "fixture_candidates_are_never_native_candidate_records": True,
            "inner_execution_mode": INNER_EXECUTION_MODE,
            "inner_route_claims_actual_model_execution": False,
            "actual_model_execution_claim_responsibility": (
                OUTER_EXECUTOR_RESPONSIBILITY
            ),
            "externally_authenticated_generation_contract_required": True,
            "exact_prompt_token_ids_bound_before_generation": True,
            "exact_native_request_reconstruction_required": True,
        },
        "fixture generation contract invalid",
    )
    outer = dict(
        _mapping(
            config["outer_executor_responsibility"],
            "outer-executor responsibility",
        )
    )
    _require(
        outer
        == {
            "nonce_precommit_receipt_reference_required": True,
            "fixture_route_verifies_precommit_chronology": False,
            "fixture_route_verifies_expectation_access_chronology": False,
            "future_sealed_outer_executor_must_establish_chronology": True,
        },
        "outer-executor chronology responsibility invalid",
    )
    _require(config["claim_scope"] == CLAIM_SCOPE, "fixture claim scope invalid")
    return AuthenticatedFixtureConfig(
        body=config,
        config_sha256=observed_config_hash,
        production_runner_sha256=str(runner_config["sha256"]),
        codebook_sha256=str(codebook_config["canonical_json_sha256"]),
        codebook_source_sha256=str(codebook_config["file_sha256"]),
        forbidden_field_fragments=tuple(forbidden),
        blinding_domain=str(blinding["domain"]),
    )


def nonce_precommit_sha256(*, fixture_nonce_sha256: str) -> str:
    _require(
        isinstance(fixture_nonce_sha256, str)
        and SHA256_RE.fullmatch(fixture_nonce_sha256) is not None,
        "fixture-only nonce must be lowercase SHA-256 form",
    )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "adjudication-fixture-nonce-precommit-v3",
                "fixture_nonce_sha256": fixture_nonce_sha256,
            }
        )
    )


def _validated_runtime_identity(value: Any) -> dict[str, Any]:
    identity = copy.deepcopy(dict(_mapping(value, "runtime identity")))
    _require(
        set(identity)
        == {"runtime_kind", "runtime_package_lock_sha256", "runtime_build_sha256"}
        and isinstance(identity["runtime_kind"], str)
        and bool(identity["runtime_kind"])
        and SHA256_RE.fullmatch(str(identity["runtime_package_lock_sha256"]))
        is not None
        and SHA256_RE.fullmatch(str(identity["runtime_build_sha256"])) is not None,
        "runtime identity invalid",
    )
    return identity


def _validated_native_adapter_identity(value: Any) -> dict[str, Any]:
    identity = copy.deepcopy(dict(_mapping(value, "native adapter identity")))
    _require(
        set(identity)
        == {"adapter_kind", "adapter_source_sha256", "adapter_config_sha256"}
        and isinstance(identity["adapter_kind"], str)
        and bool(identity["adapter_kind"])
        and SHA256_RE.fullmatch(str(identity["adapter_source_sha256"])) is not None
        and SHA256_RE.fullmatch(str(identity["adapter_config_sha256"])) is not None,
        "native adapter identity invalid",
    )
    return identity


def build_fixture_generation_contract(
    *,
    generation_context: production.AuthenticatedNativeGenerationContext,
    runtime_identity: Mapping[str, Any],
    native_adapter_identity: Mapping[str, Any],
    verdict_seed: int,
    repair_seed: int,
    fixture_nonce_sha256: str,
    outer_nonce_precommit_receipt_sha256: str,
) -> dict[str, Any]:
    """Build a serializable context contract without claiming run chronology."""

    _require(
        isinstance(
            generation_context, production.AuthenticatedNativeGenerationContext
        ),
        "generation context is not authenticated",
    )
    _require(
        isinstance(verdict_seed, int)
        and not isinstance(verdict_seed, bool)
        and isinstance(repair_seed, int)
        and not isinstance(repair_seed, bool),
        "fixture generation seeds invalid",
    )
    _require(
        SHA256_RE.fullmatch(outer_nonce_precommit_receipt_sha256) is not None,
        "outer nonce-precommit receipt reference must be SHA-256",
    )
    runtime = _validated_runtime_identity(runtime_identity)
    adapter = _validated_native_adapter_identity(native_adapter_identity)
    kwargs = copy.deepcopy(dict(generation_context.chat_template_kwargs))
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": GENERATION_CONTRACT_KIND,
        "execution_mode": INNER_EXECUTION_MODE,
        "inner_route_claims_actual_model_execution": False,
        "actual_model_execution_claim_responsibility": (
            OUTER_EXECUTOR_RESPONSIBILITY
        ),
        "model_identity": copy.deepcopy(dict(generation_context.model_identity)),
        "model_identity_sha256": generation_context.model_identity_sha256,
        "tokenizer_identity": copy.deepcopy(
            dict(generation_context.tokenizer_identity)
        ),
        "tokenizer_identity_sha256": (
            generation_context.tokenizer_identity_sha256
        ),
        "chat_template_kwargs": kwargs,
        "chat_template_kwargs_sha256": sha256_bytes(canonical_json_bytes(kwargs)),
        "runtime_identity": runtime,
        "runtime_identity_sha256": sha256_bytes(canonical_json_bytes(runtime)),
        "native_adapter_identity": adapter,
        "native_adapter_identity_sha256": sha256_bytes(
            canonical_json_bytes(adapter)
        ),
        "seeds": {
            "verdict_seed": verdict_seed,
            "repair_seed": repair_seed,
        },
        "nonce_provenance": {
            "fixture_nonce_sha256": fixture_nonce_sha256,
            "nonce_precommit_sha256": nonce_precommit_sha256(
                fixture_nonce_sha256=fixture_nonce_sha256
            ),
            "outer_nonce_precommit_receipt_sha256": (
                outer_nonce_precommit_receipt_sha256
            ),
            "precommit_chronology_verified_by_fixture_route": False,
            "expectation_access_chronology_verified_by_fixture_route": False,
            "chronology_responsibility": OUTER_EXECUTOR_RESPONSIBILITY,
        },
        "claim_scope": copy.deepcopy(CLAIM_SCOPE),
    }


@dataclass(frozen=True)
class AuthenticatedFixtureGenerationContract:
    body: Mapping[str, Any]
    contract_sha256: str
    verdict_seed: int
    repair_seed: int
    runtime_identity_sha256: str
    native_adapter_identity_sha256: str
    nonce_precommit_sha256: str
    outer_nonce_precommit_receipt_sha256: str


def authenticate_fixture_generation_contract(
    *,
    value: Any,
    expected_generation_contract_sha256: str,
    generation_context: production.AuthenticatedNativeGenerationContext,
    expected_verdict_seed: int,
    expected_repair_seed: int,
    expected_fixture_nonce_sha256: str,
) -> AuthenticatedFixtureGenerationContract:
    """Authenticate context, seeds, and nonce references against an outer hash."""

    contract = copy.deepcopy(dict(_mapping(value, "fixture generation contract")))
    expected_fields = {
        "schema_version",
        "interface_version",
        "kind",
        "execution_mode",
        "inner_route_claims_actual_model_execution",
        "actual_model_execution_claim_responsibility",
        "model_identity",
        "model_identity_sha256",
        "tokenizer_identity",
        "tokenizer_identity_sha256",
        "chat_template_kwargs",
        "chat_template_kwargs_sha256",
        "runtime_identity",
        "runtime_identity_sha256",
        "native_adapter_identity",
        "native_adapter_identity_sha256",
        "seeds",
        "nonce_provenance",
        "claim_scope",
    }
    _require(
        set(contract) == expected_fields,
        "fixture generation contract fields invalid",
    )
    observed_hash = sha256_bytes(canonical_json_bytes(contract))
    _require(
        SHA256_RE.fullmatch(expected_generation_contract_sha256) is not None
        and observed_hash == expected_generation_contract_sha256,
        "fixture generation contract differs from external authenticated hash",
    )
    _require(
        isinstance(
            generation_context, production.AuthenticatedNativeGenerationContext
        )
        and contract["schema_version"] == SCHEMA_VERSION
        and contract["interface_version"] == INTERFACE_VERSION
        and contract["kind"] == GENERATION_CONTRACT_KIND
        and contract["execution_mode"] == INNER_EXECUTION_MODE
        and contract["inner_route_claims_actual_model_execution"] is False
        and contract["actual_model_execution_claim_responsibility"]
        == OUTER_EXECUTOR_RESPONSIBILITY
        and contract["claim_scope"] == CLAIM_SCOPE,
        "fixture generation contract identity or claim scope invalid",
    )
    model = dict(_mapping(contract["model_identity"], "contract model identity"))
    tokenizer = dict(
        _mapping(contract["tokenizer_identity"], "contract tokenizer identity")
    )
    kwargs = dict(
        _mapping(contract["chat_template_kwargs"], "contract chat-template kwargs")
    )
    _require(
        model == generation_context.model_identity
        and contract["model_identity_sha256"]
        == generation_context.model_identity_sha256
        == sha256_bytes(canonical_json_bytes(model))
        and tokenizer == generation_context.tokenizer_identity
        and contract["tokenizer_identity_sha256"]
        == generation_context.tokenizer_identity_sha256
        == sha256_bytes(canonical_json_bytes(tokenizer))
        and kwargs == generation_context.chat_template_kwargs
        and contract["chat_template_kwargs_sha256"]
        == sha256_bytes(canonical_json_bytes(kwargs)),
        "fixture generation context differs from authenticated contract",
    )
    runtime = _validated_runtime_identity(contract["runtime_identity"])
    adapter = _validated_native_adapter_identity(
        contract["native_adapter_identity"]
    )
    _require(
        contract["runtime_identity_sha256"]
        == sha256_bytes(canonical_json_bytes(runtime))
        and contract["native_adapter_identity_sha256"]
        == sha256_bytes(canonical_json_bytes(adapter)),
        "runtime or native-adapter identity hash invalid",
    )
    seeds = dict(_mapping(contract["seeds"], "contract seeds"))
    _require(
        set(seeds) == {"verdict_seed", "repair_seed"}
        and isinstance(expected_verdict_seed, int)
        and not isinstance(expected_verdict_seed, bool)
        and isinstance(expected_repair_seed, int)
        and not isinstance(expected_repair_seed, bool)
        and seeds["verdict_seed"] == expected_verdict_seed
        and seeds["repair_seed"] == expected_repair_seed,
        "fixture generation seeds differ from external expected seeds",
    )
    nonce = dict(_mapping(contract["nonce_provenance"], "nonce provenance"))
    expected_nonce_fields = {
        "fixture_nonce_sha256",
        "nonce_precommit_sha256",
        "outer_nonce_precommit_receipt_sha256",
        "precommit_chronology_verified_by_fixture_route",
        "expectation_access_chronology_verified_by_fixture_route",
        "chronology_responsibility",
    }
    _require(
        set(nonce) == expected_nonce_fields
        and nonce["fixture_nonce_sha256"] == expected_fixture_nonce_sha256
        and nonce["nonce_precommit_sha256"]
        == nonce_precommit_sha256(
            fixture_nonce_sha256=expected_fixture_nonce_sha256
        )
        and SHA256_RE.fullmatch(
            str(nonce["outer_nonce_precommit_receipt_sha256"])
        )
        is not None
        and nonce["precommit_chronology_verified_by_fixture_route"] is False
        and nonce["expectation_access_chronology_verified_by_fixture_route"]
        is False
        and nonce["chronology_responsibility"] == OUTER_EXECUTOR_RESPONSIBILITY,
        "nonce precommit reference or chronology scope invalid",
    )
    return AuthenticatedFixtureGenerationContract(
        body=contract,
        contract_sha256=observed_hash,
        verdict_seed=int(seeds["verdict_seed"]),
        repair_seed=int(seeds["repair_seed"]),
        runtime_identity_sha256=str(contract["runtime_identity_sha256"]),
        native_adapter_identity_sha256=str(
            contract["native_adapter_identity_sha256"]
        ),
        nonce_precommit_sha256=str(nonce["nonce_precommit_sha256"]),
        outer_nonce_precommit_receipt_sha256=str(
            nonce["outer_nonce_precommit_receipt_sha256"]
        ),
    )


def _validated_packet(
    packet: Mapping[str, Any], *, annotation_pass: str
) -> dict[str, Any]:
    # Deliberately use the audited V3 packet adapter so fixture semantics cannot
    # drift from production semantics.
    return production._validated_packet(packet, annotation_pass=annotation_pass)


def _validated_codebook(value: Mapping[str, Any]) -> dict[str, Any]:
    return production._validated_codebook(value)


def _authenticate_units(
    *,
    packet: Mapping[str, Any],
    annotation_pass: str,
    candidate_unit_bundle: Mapping[str, Any] | None,
    expected_candidate_unit_bundle_sha256: str | None,
) -> production.AuthenticatedCandidateUnits | None:
    if annotation_pass == "completion_chain":
        _require(
            candidate_unit_bundle is not None
            and expected_candidate_unit_bundle_sha256 is not None,
            "completion fixture requires an authenticated candidate-unit bundle",
        )
        return production.authenticate_candidate_unit_bundle(
            value=candidate_unit_bundle,
            packet=packet,
            expected_bundle_sha256=expected_candidate_unit_bundle_sha256,
        )
    _require(
        annotation_pass == "prefix_novelty"
        and candidate_unit_bundle is None
        and expected_candidate_unit_bundle_sha256 is None,
        "novelty fixture must not receive completion candidate units",
    )
    return None


def build_authored_fixture_candidate(
    *,
    packet: Mapping[str, Any],
    codebook: Mapping[str, Any],
    annotation_pass: str,
    projection: Mapping[str, Any],
    candidate_unit_bundle: Mapping[str, Any] | None = None,
    expected_candidate_unit_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one explicit authored projection; never attach model lineage."""

    _require(annotation_pass in production.PASSES, "annotation pass invalid")
    validated_packet = _validated_packet(packet, annotation_pass=annotation_pass)
    validated_codebook = _validated_codebook(codebook)
    authenticated_units = _authenticate_units(
        packet=validated_packet,
        annotation_pass=annotation_pass,
        candidate_unit_bundle=candidate_unit_bundle,
        expected_candidate_unit_bundle_sha256=(
            expected_candidate_unit_bundle_sha256
        ),
    )
    if annotation_pass == "completion_chain":
        _require(authenticated_units is not None, "completion units missing")
        validated_projection = production.validate_completion_proposal(
            projection,
            codebook=validated_codebook,
            authenticated_units=authenticated_units,
        )
    else:
        validated_projection = production.validate_novelty_proposal(projection)
    projection_hash = sha256_bytes(canonical_json_bytes(validated_projection))
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": FIXTURE_CANDIDATE_KIND,
        "annotation_pass": annotation_pass,
        "packet_id_sha256": validated_packet["packet_id_sha256"],
        "source_id_sha256": validated_packet["source_id_sha256"],
        "candidate_unit_bundle_sha256": (
            None if authenticated_units is None else authenticated_units.bundle_sha256
        ),
        "control_fixture_provenance": copy.deepcopy(FIXTURE_PROVENANCE),
        "claim_scope": copy.deepcopy(CLAIM_SCOPE),
        "projection": copy.deepcopy(validated_projection),
        "projection_sha256": projection_hash,
    }


@dataclass(frozen=True)
class AuthenticatedFixtureCandidate:
    projection: Mapping[str, Any]
    candidate_sha256: str
    projection_sha256: str


def _authenticate_fixture_candidate(
    *,
    value: Any,
    packet: Mapping[str, Any],
    codebook: Mapping[str, Any],
    annotation_pass: str,
    authenticated_units: production.AuthenticatedCandidateUnits | None,
    config: AuthenticatedFixtureConfig,
) -> AuthenticatedFixtureCandidate:
    candidate = copy.deepcopy(dict(_mapping(value, "authored fixture candidate")))
    expected_fields = {
        "schema_version",
        "interface_version",
        "kind",
        "annotation_pass",
        "packet_id_sha256",
        "source_id_sha256",
        "candidate_unit_bundle_sha256",
        "control_fixture_provenance",
        "claim_scope",
        "projection",
        "projection_sha256",
    }
    _require(set(candidate) == expected_fields, "fixture candidate fields invalid")
    _require(
        candidate["schema_version"] == SCHEMA_VERSION
        and candidate["interface_version"] == INTERFACE_VERSION
        and candidate["kind"] == FIXTURE_CANDIDATE_KIND
        and candidate["annotation_pass"] == annotation_pass
        and candidate["packet_id_sha256"] == packet["packet_id_sha256"]
        and candidate["source_id_sha256"] == packet["source_id_sha256"]
        and candidate["control_fixture_provenance"]
        == config.body["fixture_candidate_provenance"]
        and candidate["claim_scope"] == CLAIM_SCOPE,
        "fixture candidate identity, packet, or provenance invalid",
    )
    if annotation_pass == "completion_chain":
        _require(
            authenticated_units is not None
            and candidate["candidate_unit_bundle_sha256"]
            == authenticated_units.bundle_sha256,
            "fixture candidate unit-bundle binding invalid",
        )
        projection = production.validate_completion_proposal(
            candidate["projection"],
            codebook=codebook,
            authenticated_units=authenticated_units,
        )
    else:
        _require(
            authenticated_units is None
            and candidate["candidate_unit_bundle_sha256"] is None,
            "novelty fixture candidate bundle binding invalid",
        )
        projection = production.validate_novelty_proposal(candidate["projection"])
    projection_hash = sha256_bytes(canonical_json_bytes(projection))
    _require(
        candidate["projection_sha256"] == projection_hash,
        "fixture candidate projection hash invalid",
    )
    return AuthenticatedFixtureCandidate(
        projection=copy.deepcopy(projection),
        candidate_sha256=sha256_bytes(canonical_json_bytes(candidate)),
        projection_sha256=projection_hash,
    )


def _normalized_field_name(value: str) -> tuple[str, str]:
    separated = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return separated, separated.replace("_", "")


def assert_unlabeled_model_input(
    *,
    messages: Sequence[Mapping[str, str]],
    forbidden_field_fragments: Sequence[str],
) -> dict[str, Any]:
    """Reject label/expectation leakage in model-visible object field names."""

    message_list = list(messages)
    _require(
        len(message_list) == 2
        and message_list[0].get("role") == "system"
        and message_list[1].get("role") == "user"
        and set(message_list[0]) == {"role", "content"}
        and set(message_list[1]) == {"role", "content"},
        "fixture model input must be the exact two-message production shape",
    )
    try:
        payload = json.loads(str(message_list[1]["content"]))
    except json.JSONDecodeError as error:
        raise FixtureRouteError(f"fixture user payload is invalid JSON: {error}") from error
    _require(isinstance(payload, dict), "fixture user payload must be an object")
    fragments = list(forbidden_field_fragments)
    _require(
        set(fragments) >= REQUIRED_FORBIDDEN_FIELD_FRAGMENTS,
        "unlabeled audit lacks required forbidden-field coverage",
    )
    normalized_fragments = [_normalized_field_name(item) for item in fragments]

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                _require(isinstance(key, str), f"non-string model-input key at {path}")
                separated, collapsed = _normalized_field_name(key)
                for forbidden_separated, forbidden_collapsed in normalized_fragments:
                    _require(
                        forbidden_separated not in separated
                        and forbidden_collapsed not in collapsed,
                        f"forbidden labeled model-input field {key!r} at {path}",
                    )
                visit(nested, f"{path}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")

    visit(payload, "$user")
    serialized = canonical_json_text(payload)
    _require(
        str(message_list[1]["content"]) == serialized,
        "fixture user payload is not canonical production JSON",
    )
    return payload


@dataclass(frozen=True)
class PreparedFixture:
    config: AuthenticatedFixtureConfig
    generation_contract: AuthenticatedFixtureGenerationContract
    generation_context: production.AuthenticatedNativeGenerationContext
    packet: Mapping[str, Any]
    packet_sha256: str
    codebook: Mapping[str, Any]
    codebook_sha256: str
    codebook_source_sha256: str
    authenticated_units: production.AuthenticatedCandidateUnits | None
    candidates_original: tuple[AuthenticatedFixtureCandidate, AuthenticatedFixtureCandidate]
    blinded_candidates: production.BlindedCandidates
    candidate_hashes_original: tuple[str, str]
    projection_hashes_original: tuple[str, str]
    candidate_hashes_blinded: tuple[str, str]
    projection_hashes_blinded: tuple[str, str]
    original_order_binding_sha256: str
    fixture_nonce_sha256: str
    verdict_messages: tuple[Mapping[str, str], ...]
    verdict_schema: Mapping[str, Any]
    repair_decision_messages: tuple[Mapping[str, str], ...]
    repair_decision_schema: Mapping[str, Any]
    repair_detail_messages: tuple[Mapping[str, str], ...] | None
    repair_detail_schema: Mapping[str, Any] | None
    model_input_bindings: Mapping[str, Any]


def _model_input_binding(
    *,
    messages: Sequence[Mapping[str, str]],
    schema: Mapping[str, Any],
    payload: Any,
    generation_context: production.AuthenticatedNativeGenerationContext,
) -> dict[str, Any]:
    validated_schema = production.validate_executable_response_schema(schema)
    rendered = production.render_messages_to_native_token_ids(
        generation_context.tokenizer,
        messages,
        chat_template_kwargs=generation_context.chat_template_kwargs,
        tokenizer_identity=generation_context.tokenizer_identity,
    )
    token_ids = list(rendered["token_ids"])
    return {
        "messages_sha256": sha256_bytes(canonical_json_bytes(list(messages))),
        "user_payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
        "response_schema_sha256": sha256_bytes(
            canonical_json_bytes(validated_schema)
        ),
        "submitted_prompt_token_ids": token_ids,
        "submitted_prompt_token_ids_sha256": sha256_bytes(
            canonical_json_bytes(token_ids)
        ),
        "submitted_prompt_token_count": len(token_ids),
    }


def _prepare_fixture(
    *,
    packet: Mapping[str, Any],
    codebook: Mapping[str, Any],
    annotation_pass: str,
    fixture_candidates: tuple[Mapping[str, Any], Mapping[str, Any]],
    fixture_nonce_sha256: str,
    fixture_config: Mapping[str, Any],
    expected_fixture_config_sha256: str,
    generation_contract: Mapping[str, Any],
    expected_generation_contract_sha256: str,
    generation_context: production.AuthenticatedNativeGenerationContext,
    expected_verdict_seed: int,
    expected_repair_seed: int,
    candidate_unit_bundle: Mapping[str, Any] | None,
    expected_candidate_unit_bundle_sha256: str | None,
) -> PreparedFixture:
    _require(annotation_pass in production.PASSES, "annotation pass invalid")
    _require(
        isinstance(fixture_candidates, tuple) and len(fixture_candidates) == 2,
        "fixture adjudication requires exactly two authored candidates",
    )
    _require(
        isinstance(fixture_nonce_sha256, str)
        and SHA256_RE.fullmatch(fixture_nonce_sha256) is not None,
        "fixture-only nonce must be lowercase SHA-256 form",
    )
    config = authenticate_fixture_config(
        value=fixture_config,
        expected_config_sha256=expected_fixture_config_sha256,
    )
    authenticated_generation_contract = (
        authenticate_fixture_generation_contract(
            value=generation_contract,
            expected_generation_contract_sha256=(
                expected_generation_contract_sha256
            ),
            generation_context=generation_context,
            expected_verdict_seed=expected_verdict_seed,
            expected_repair_seed=expected_repair_seed,
            expected_fixture_nonce_sha256=fixture_nonce_sha256,
        )
    )
    validated_packet = _validated_packet(packet, annotation_pass=annotation_pass)
    validated_codebook = _validated_codebook(codebook)
    codebook_hash = sha256_bytes(canonical_json_bytes(validated_codebook))
    _require(
        codebook_hash == config.codebook_sha256,
        "runtime codebook differs from authenticated fixture config",
    )
    authenticated_units = _authenticate_units(
        packet=validated_packet,
        annotation_pass=annotation_pass,
        candidate_unit_bundle=candidate_unit_bundle,
        expected_candidate_unit_bundle_sha256=(
            expected_candidate_unit_bundle_sha256
        ),
    )
    candidates = tuple(
        _authenticate_fixture_candidate(
            value=item,
            packet=validated_packet,
            codebook=validated_codebook,
            annotation_pass=annotation_pass,
            authenticated_units=authenticated_units,
            config=config,
        )
        for item in fixture_candidates
    )
    _require(
        candidates[0].projection_sha256 != candidates[1].projection_sha256,
        "fixture candidates must contain two distinct semantic projections",
    )

    indexed = list(enumerate(candidates))
    indexed.sort(
        key=lambda pair: (
            canonical_json_bytes(pair[1].projection),
            pair[1].candidate_sha256,
        )
    )
    selector = int(
        sha256_bytes(
            canonical_json_bytes(
                {
                    "domain": config.blinding_domain,
                    "annotation_pass": annotation_pass,
                    "packet_id_sha256": validated_packet["packet_id_sha256"],
                    "fixture_nonce_sha256": fixture_nonce_sha256,
                    "sorted_candidate_sha256s": sorted(
                        item.candidate_sha256 for item in candidates
                    ),
                    "sorted_projection_sha256s": sorted(
                        item.projection_sha256 for item in candidates
                    ),
                }
            )
        )[:16],
        16,
    ) % 2
    if selector:
        indexed.reverse()
    ordered = [item[1] for item in indexed]
    visible = [
        {
            "candidate_id": f"candidate_{index + 1}",
            "annotation": copy.deepcopy(candidate.projection),
        }
        for index, candidate in enumerate(ordered)
    ]
    blinded = production.BlindedCandidates(
        proposals=(
            copy.deepcopy(ordered[0].projection),
            copy.deepcopy(ordered[1].projection),
        ),
        original_indexes=(indexed[0][0], indexed[1][0]),
        order_sha256=sha256_bytes(canonical_json_bytes(visible)),
        record_provenance=None,
    )
    original_candidate_hashes = (
        candidates[0].candidate_sha256,
        candidates[1].candidate_sha256,
    )
    original_projection_hashes = (
        candidates[0].projection_sha256,
        candidates[1].projection_sha256,
    )
    original_binding = sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "authored-fixture-original-order-binding-v3",
                "candidate_sha256s": list(original_candidate_hashes),
                "projection_sha256s": list(original_projection_hashes),
            }
        )
    )

    verdict_messages = production.build_adjudication_messages(
        packet=validated_packet,
        codebook=validated_codebook,
        annotation_pass=annotation_pass,
        blinded_candidates=blinded,
        authenticated_units=authenticated_units,
    )
    verdict_payload = assert_unlabeled_model_input(
        messages=verdict_messages,
        forbidden_field_fragments=config.forbidden_field_fragments,
    )
    verdict_schema = production.adjudication_response_schema()

    repair_decision_messages = production.build_neither_repair_messages(
        packet=validated_packet,
        codebook=validated_codebook,
        annotation_pass=annotation_pass,
        blinded_candidates=blinded,
        authenticated_units=authenticated_units,
        response_route="decision",
    )
    repair_decision_payload = assert_unlabeled_model_input(
        messages=repair_decision_messages,
        forbidden_field_fragments=config.forbidden_field_fragments,
    )
    repair_decision_schema = production.neither_repair_response_schema(
        codebook=validated_codebook,
        annotation_pass=annotation_pass,
        authenticated_units=authenticated_units,
        response_route="decision",
    )

    repair_detail_messages: list[dict[str, str]] | None = None
    repair_detail_schema: dict[str, Any] | None = None
    repair_detail_binding: dict[str, Any] | None = None
    if (
        annotation_pass == "completion_chain"
        and authenticated_units is not None
        and len(authenticated_units.units) >= 3
    ):
        repair_detail_messages = production.build_neither_repair_messages(
            packet=validated_packet,
            codebook=validated_codebook,
            annotation_pass=annotation_pass,
            blinded_candidates=blinded,
            authenticated_units=authenticated_units,
            response_route="chain_detail",
        )
        repair_detail_payload = assert_unlabeled_model_input(
            messages=repair_detail_messages,
            forbidden_field_fragments=config.forbidden_field_fragments,
        )
        repair_detail_schema = production.neither_repair_response_schema(
            codebook=validated_codebook,
            annotation_pass=annotation_pass,
            authenticated_units=authenticated_units,
            response_route="chain_detail",
        )
        repair_detail_binding = _model_input_binding(
            messages=repair_detail_messages,
            schema=repair_detail_schema,
            payload=repair_detail_payload,
            generation_context=generation_context,
        )
    model_input_bindings = {
        "adjudication_verdict": _model_input_binding(
            messages=verdict_messages,
            schema=verdict_schema,
            payload=verdict_payload,
            generation_context=generation_context,
        ),
        "neither_repair_decision": _model_input_binding(
            messages=repair_decision_messages,
            schema=repair_decision_schema,
            payload=repair_decision_payload,
            generation_context=generation_context,
        ),
        "neither_repair_chain_detail": repair_detail_binding,
    }
    return PreparedFixture(
        config=config,
        generation_contract=authenticated_generation_contract,
        generation_context=generation_context,
        packet=copy.deepcopy(validated_packet),
        packet_sha256=sha256_bytes(canonical_json_bytes(validated_packet)),
        codebook=copy.deepcopy(validated_codebook),
        codebook_sha256=codebook_hash,
        codebook_source_sha256=config.codebook_source_sha256,
        authenticated_units=authenticated_units,
        candidates_original=(candidates[0], candidates[1]),
        blinded_candidates=blinded,
        candidate_hashes_original=original_candidate_hashes,
        projection_hashes_original=original_projection_hashes,
        candidate_hashes_blinded=(
            ordered[0].candidate_sha256,
            ordered[1].candidate_sha256,
        ),
        projection_hashes_blinded=(
            ordered[0].projection_sha256,
            ordered[1].projection_sha256,
        ),
        original_order_binding_sha256=original_binding,
        fixture_nonce_sha256=fixture_nonce_sha256,
        verdict_messages=tuple(copy.deepcopy(verdict_messages)),
        verdict_schema=copy.deepcopy(verdict_schema),
        repair_decision_messages=tuple(copy.deepcopy(repair_decision_messages)),
        repair_decision_schema=copy.deepcopy(repair_decision_schema),
        repair_detail_messages=(
            None
            if repair_detail_messages is None
            else tuple(copy.deepcopy(repair_detail_messages))
        ),
        repair_detail_schema=copy.deepcopy(repair_detail_schema),
        model_input_bindings=copy.deepcopy(model_input_bindings),
    )


def _lock_from_prepared(prepared: PreparedFixture) -> dict[str, Any]:
    bundle_hash = (
        None
        if prepared.authenticated_units is None
        else prepared.authenticated_units.bundle_sha256
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": FIXTURE_LOCK_KIND,
        "scope": "controls_module_only_authored_candidates",
        "annotation_pass": prepared.packet["annotation_pass"],
        "packet_id_sha256": prepared.packet["packet_id_sha256"],
        "packet_sha256": prepared.packet_sha256,
        "source_id_sha256": prepared.packet["source_id_sha256"],
        "candidate_unit_bundle_sha256": bundle_hash,
        "codebook_sha256": prepared.codebook_sha256,
        "codebook_source_sha256": prepared.codebook_source_sha256,
        "production_runner_sha256": prepared.config.production_runner_sha256,
        "fixture_runner_sha256": fixture_module_sha256(),
        "fixture_config_sha256": prepared.config.config_sha256,
        "generation_contract_sha256": (
            prepared.generation_contract.contract_sha256
        ),
        "runtime_identity_sha256": (
            prepared.generation_contract.runtime_identity_sha256
        ),
        "native_adapter_identity_sha256": (
            prepared.generation_contract.native_adapter_identity_sha256
        ),
        "fixture_nonce_sha256": prepared.fixture_nonce_sha256,
        "nonce_precommit_sha256": (
            prepared.generation_contract.nonce_precommit_sha256
        ),
        "outer_nonce_precommit_receipt_sha256": (
            prepared.generation_contract.outer_nonce_precommit_receipt_sha256
        ),
        "generation_contract": copy.deepcopy(
            dict(prepared.generation_contract.body)
        ),
        "inner_execution_mode": INNER_EXECUTION_MODE,
        "inner_route_claims_actual_model_execution": False,
        "precommit_chronology_verified_by_fixture_route": False,
        "expectation_access_chronology_verified_by_fixture_route": False,
        "chronology_responsibility": OUTER_EXECUTOR_RESPONSIBILITY,
        "authored_candidate_fixture_sha256s_in_original_order": list(
            prepared.candidate_hashes_original
        ),
        "authored_projection_sha256s_in_original_order": list(
            prepared.projection_hashes_original
        ),
        "original_order_binding_sha256": prepared.original_order_binding_sha256,
        "blinded_original_indexes": list(
            prepared.blinded_candidates.original_indexes
        ),
        "blinded_candidate_fixture_sha256s": list(
            prepared.candidate_hashes_blinded
        ),
        "blinded_projection_sha256s": list(prepared.projection_hashes_blinded),
        "candidate_order_sha256": prepared.blinded_candidates.order_sha256,
        "model_input_bindings": copy.deepcopy(prepared.model_input_bindings),
        "fixture_candidate_provenance": copy.deepcopy(FIXTURE_PROVENANCE),
        "candidate_model_generated": False,
        "candidate_native_model_record_lineage_present": False,
        "expectation_object_bound": False,
        "claim_scope": copy.deepcopy(CLAIM_SCOPE),
    }


def build_fixture_adjudication_lock(
    *,
    packet: Mapping[str, Any],
    codebook: Mapping[str, Any],
    annotation_pass: str,
    fixture_candidates: tuple[Mapping[str, Any], Mapping[str, Any]],
    fixture_nonce_sha256: str,
    fixture_config: Mapping[str, Any],
    expected_fixture_config_sha256: str,
    generation_contract: Mapping[str, Any],
    expected_generation_contract_sha256: str,
    generation_context: production.AuthenticatedNativeGenerationContext,
    expected_verdict_seed: int,
    expected_repair_seed: int,
    candidate_unit_bundle: Mapping[str, Any] | None = None,
    expected_candidate_unit_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the complete pre-generation fixture lock body."""

    prepared = _prepare_fixture(
        packet=packet,
        codebook=codebook,
        annotation_pass=annotation_pass,
        fixture_candidates=fixture_candidates,
        fixture_nonce_sha256=fixture_nonce_sha256,
        fixture_config=fixture_config,
        expected_fixture_config_sha256=expected_fixture_config_sha256,
        generation_contract=generation_contract,
        expected_generation_contract_sha256=(
            expected_generation_contract_sha256
        ),
        generation_context=generation_context,
        expected_verdict_seed=expected_verdict_seed,
        expected_repair_seed=expected_repair_seed,
        candidate_unit_bundle=candidate_unit_bundle,
        expected_candidate_unit_bundle_sha256=(
            expected_candidate_unit_bundle_sha256
        ),
    )
    return _lock_from_prepared(prepared)


def fixture_adjudication_lock_sha256(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _authenticate_fixture_lock(
    *,
    fixture_lock: Mapping[str, Any],
    expected_fixture_lock_sha256: str,
    packet: Mapping[str, Any],
    codebook: Mapping[str, Any],
    annotation_pass: str,
    fixture_candidates: tuple[Mapping[str, Any], Mapping[str, Any]],
    fixture_nonce_sha256: str,
    fixture_config: Mapping[str, Any],
    expected_fixture_config_sha256: str,
    generation_contract: Mapping[str, Any],
    expected_generation_contract_sha256: str,
    generation_context: production.AuthenticatedNativeGenerationContext,
    expected_verdict_seed: int,
    expected_repair_seed: int,
    candidate_unit_bundle: Mapping[str, Any] | None,
    expected_candidate_unit_bundle_sha256: str | None,
) -> PreparedFixture:
    prepared = _prepare_fixture(
        packet=packet,
        codebook=codebook,
        annotation_pass=annotation_pass,
        fixture_candidates=fixture_candidates,
        fixture_nonce_sha256=fixture_nonce_sha256,
        fixture_config=fixture_config,
        expected_fixture_config_sha256=expected_fixture_config_sha256,
        generation_contract=generation_contract,
        expected_generation_contract_sha256=(
            expected_generation_contract_sha256
        ),
        generation_context=generation_context,
        expected_verdict_seed=expected_verdict_seed,
        expected_repair_seed=expected_repair_seed,
        candidate_unit_bundle=candidate_unit_bundle,
        expected_candidate_unit_bundle_sha256=(
            expected_candidate_unit_bundle_sha256
        ),
    )
    supplied = copy.deepcopy(dict(_mapping(fixture_lock, "fixture lock")))
    expected = _lock_from_prepared(prepared)
    observed_hash = fixture_adjudication_lock_sha256(supplied)
    _require(
        SHA256_RE.fullmatch(expected_fixture_lock_sha256) is not None
        and observed_hash == expected_fixture_lock_sha256,
        "fixture lock differs from out-of-band authenticated hash",
    )
    _require(supplied == expected, "fixture lock content or ordering bindings invalid")
    return prepared


def _stage_lineage(
    *,
    prepared: PreparedFixture,
    fixture_lock_sha256: str,
    model_input_stage: str,
) -> dict[str, Any]:
    binding = prepared.model_input_bindings[model_input_stage]
    _require(binding is not None, f"model-input stage {model_input_stage} unavailable")
    return {
        "fixture_lock_sha256": fixture_lock_sha256,
        "packet_sha256": prepared.packet_sha256,
        "candidate_unit_bundle_sha256": (
            None
            if prepared.authenticated_units is None
            else prepared.authenticated_units.bundle_sha256
        ),
        "codebook_sha256": prepared.codebook_sha256,
        "codebook_source_sha256": prepared.codebook_source_sha256,
        "production_runner_sha256": prepared.config.production_runner_sha256,
        "fixture_runner_sha256": fixture_module_sha256(),
        "fixture_config_sha256": prepared.config.config_sha256,
        "generation_contract_sha256": (
            prepared.generation_contract.contract_sha256
        ),
        "runtime_identity_sha256": (
            prepared.generation_contract.runtime_identity_sha256
        ),
        "native_adapter_identity_sha256": (
            prepared.generation_contract.native_adapter_identity_sha256
        ),
        "fixture_nonce_sha256": prepared.fixture_nonce_sha256,
        "nonce_precommit_sha256": (
            prepared.generation_contract.nonce_precommit_sha256
        ),
        "outer_nonce_precommit_receipt_sha256": (
            prepared.generation_contract.outer_nonce_precommit_receipt_sha256
        ),
        "original_order_binding_sha256": prepared.original_order_binding_sha256,
        "candidate_order_sha256": prepared.blinded_candidates.order_sha256,
        "authored_candidate_fixture_sha256s_in_original_order": list(
            prepared.candidate_hashes_original
        ),
        "blinded_candidate_fixture_sha256s": list(
            prepared.candidate_hashes_blinded
        ),
        "model_input_messages_sha256": binding["messages_sha256"],
        "model_input_user_payload_sha256": binding["user_payload_sha256"],
    }


def generation_api_accepts_expectation_object() -> bool:
    """Machine-checkable denial: no fixture generation API has such a parameter."""

    forbidden = {"expectation", "expectations", "expected", "gold"}
    parameters = set(inspect.signature(run_fixture_adjudication).parameters)
    return bool(parameters & forbidden)


def _parse_model_object(text: str, *, label: str) -> dict[str, Any]:
    return production._parse_json_object(text, label=label)


def _build_and_execute_exact_native(
    *,
    prepared: PreparedFixture,
    generate_native: production.GenerateNative,
    messages: Sequence[Mapping[str, str]],
    schema: Mapping[str, Any],
    seed: int,
    stage: str,
    model_input_stage: str,
    lineage_bindings: Mapping[str, Any],
) -> tuple[
    production.NativeGenerationRequest,
    production.NativeGenerationResult,
]:
    """Rebuild and compare the exact production request around execution."""

    context = prepared.generation_context
    annotation_pass = str(prepared.packet["annotation_pass"])
    packet_id = str(prepared.packet["packet_id_sha256"])
    source_id = str(prepared.packet["source_id_sha256"])
    expected_request = production.build_native_generation_request(
        context=context,
        messages=messages,
        schema=schema,
        seed=seed,
        stage=stage,
        annotation_pass=annotation_pass,
        packet_id_sha256=packet_id,
        source_id_sha256=source_id,
        lineage_bindings=lineage_bindings,
    )
    expected_body = production.validate_native_generation_request(
        expected_request, context=context
    )
    binding = prepared.model_input_bindings[model_input_stage]
    _require(binding is not None, f"model-input stage {model_input_stage} unavailable")
    _require(
        expected_body["messages_sha256"] == binding["messages_sha256"]
        and expected_body["response_schema_sha256"]
        == binding["response_schema_sha256"]
        and expected_body["submitted_prompt_token_ids"]
        == binding["submitted_prompt_token_ids"]
        and expected_body["submitted_prompt_token_ids_sha256"]
        == binding["submitted_prompt_token_ids_sha256"],
        "reconstructed native request differs from locked model-input binding",
    )
    request, result = production.execute_native_generation(
        context=context,
        generate_native=generate_native,
        messages=messages,
        schema=schema,
        seed=seed,
        stage=stage,
        annotation_pass=annotation_pass,
        packet_id_sha256=packet_id,
        source_id_sha256=source_id,
        lineage_bindings=lineage_bindings,
    )
    actual_body = production.validate_native_generation_request(
        request, context=context
    )
    production.validate_native_generation_result(request=request, result=result)
    _require(
        request.request_sha256 == expected_request.request_sha256
        and actual_body == expected_body,
        "executed native request differs from exact reconstructed request",
    )
    return request, result


def _execute_fixture(
    *,
    prepared: PreparedFixture,
    expected_fixture_lock_sha256: str,
    generate_native: production.GenerateNative,
) -> dict[str, Any]:
    verdict_lineage = _stage_lineage(
        prepared=prepared,
        fixture_lock_sha256=expected_fixture_lock_sha256,
        model_input_stage="adjudication_verdict",
    )
    verdict_request, verdict_result = _build_and_execute_exact_native(
        prepared=prepared,
        generate_native=generate_native,
        messages=prepared.verdict_messages,
        schema=prepared.verdict_schema,
        seed=prepared.generation_contract.verdict_seed,
        stage="control_fixture_adjudication_verdict",
        model_input_stage="adjudication_verdict",
        lineage_bindings=verdict_lineage,
    )
    raw_verdict = str(verdict_result.body["text"])
    verdict: str | None
    verdict_error: str | None = None
    try:
        verdict = production.validate_adjudication_verdict(
            _parse_model_object(raw_verdict, label="fixture adjudication verdict")
        )
    except production.BoundedIdRunnerError as error:
        verdict = None
        verdict_error = str(error)

    annotation_pass = str(prepared.packet["annotation_pass"])
    raw_repair: str | None = None
    raw_repair_detail: str | None = None
    repair_generation: dict[str, Any] | None = None
    if verdict is None:
        materialized = production._invalid_adjudication_materialization(
            annotation_pass=annotation_pass,
            error=str(verdict_error),
        )
    elif verdict in {"candidate_1", "candidate_2"}:
        selected = prepared.blinded_candidates.proposals[
            0 if verdict == "candidate_1" else 1
        ]
        if annotation_pass == "completion_chain":
            _require(prepared.authenticated_units is not None, "completion units lost")
            materialized = production.materialize_completion_proposal(
                proposal=selected,
                codebook=prepared.codebook,
                authenticated_units=prepared.authenticated_units,
                decision_source=f"control_fixture_adjudicator_{verdict}",
            )
        else:
            materialized = production.materialize_novelty_proposal(
                proposal=selected,
                decision_source=f"control_fixture_adjudicator_{verdict}",
            )
    else:
        repair_lineage = {
            **_stage_lineage(
                prepared=prepared,
                fixture_lock_sha256=expected_fixture_lock_sha256,
                model_input_stage="neither_repair_decision",
            ),
            "parent_verdict_request_sha256": verdict_request.request_sha256,
            "parent_verdict_result_sha256": verdict_result.result_sha256,
        }
        repair_request, repair_result = _build_and_execute_exact_native(
            prepared=prepared,
            generate_native=generate_native,
            messages=prepared.repair_decision_messages,
            schema=prepared.repair_decision_schema,
            seed=prepared.generation_contract.repair_seed,
            stage=(
                "control_fixture_completion_repair_decision"
                if annotation_pass == "completion_chain"
                else "control_fixture_novelty_repair_decision"
            ),
            model_input_stage="neither_repair_decision",
            lineage_bindings=repair_lineage,
        )
        raw_repair = str(repair_result.body["text"])
        repair_generation = {
            "decision": production.native_stage_provenance(
                repair_request, repair_result
            ),
            "chain_detail": None,
        }
        if annotation_pass == "prefix_novelty":
            try:
                repair_decision = production.validate_novelty_decision(
                    _parse_model_object(raw_repair, label="fixture novelty repair")
                )
            except production.BoundedIdRunnerError as error:
                materialized = production.materialize_novelty_proposal(
                    proposal=None,
                    decision_source="control_fixture_adjudicator_neither_repair",
                )
                materialized["semantic_validation_error"] = str(error)
            else:
                materialized = production.materialize_novelty_proposal(
                    proposal=production.assemble_novelty_proposal(
                        decision=repair_decision
                    ),
                    decision_source="control_fixture_adjudicator_neither_repair",
                )
        else:
            _require(prepared.authenticated_units is not None, "completion units lost")
            repair_parse_error: str | None = None
            try:
                repair_decision = production.validate_completion_decision(
                    _parse_model_object(raw_repair, label="fixture completion repair")
                )
            except production.BoundedIdRunnerError as error:
                repair_decision = None
                repair_parse_error = str(error)
            if repair_decision is None:
                materialized = production.materialize_completion_proposal(
                    proposal=None,
                    codebook=prepared.codebook,
                    authenticated_units=prepared.authenticated_units,
                    decision_source="control_fixture_adjudicator_neither_repair",
                )
                materialized["semantic_validation_error"] = repair_parse_error
            elif repair_decision != "chain":
                repaired = production.assemble_completion_proposal(
                    decision=repair_decision,
                    codebook=prepared.codebook,
                    authenticated_units=prepared.authenticated_units,
                )
                materialized = production.materialize_completion_proposal(
                    proposal=repaired,
                    codebook=prepared.codebook,
                    authenticated_units=prepared.authenticated_units,
                    decision_source="control_fixture_adjudicator_neither_repair",
                )
            elif (
                prepared.repair_detail_messages is None
                or prepared.repair_detail_schema is None
            ):
                materialized = production.materialize_completion_proposal(
                    proposal=None,
                    codebook=prepared.codebook,
                    authenticated_units=prepared.authenticated_units,
                    decision_source="control_fixture_adjudicator_neither_repair",
                )
                materialized["semantic_validation_error"] = (
                    "chain repair unavailable for fewer than three candidate units"
                )
            else:
                detail_lineage = {
                    **_stage_lineage(
                        prepared=prepared,
                        fixture_lock_sha256=expected_fixture_lock_sha256,
                        model_input_stage="neither_repair_chain_detail",
                    ),
                    "parent_verdict_request_sha256": verdict_request.request_sha256,
                    "parent_verdict_result_sha256": verdict_result.result_sha256,
                    "parent_repair_decision_request_sha256": (
                        repair_request.request_sha256
                    ),
                    "parent_repair_decision_result_sha256": (
                        repair_result.result_sha256
                    ),
                }
                detail_request, detail_result = _build_and_execute_exact_native(
                    prepared=prepared,
                    generate_native=generate_native,
                    messages=prepared.repair_detail_messages,
                    schema=prepared.repair_detail_schema,
                    seed=prepared.generation_contract.repair_seed,
                    stage="control_fixture_completion_repair_chain_detail",
                    model_input_stage="neither_repair_chain_detail",
                    lineage_bindings=detail_lineage,
                )
                raw_repair_detail = str(detail_result.body["text"])
                repair_generation["chain_detail"] = (
                    production.native_stage_provenance(
                        detail_request, detail_result
                    )
                )
                try:
                    repaired = production.assemble_completion_proposal(
                        decision="chain",
                        chain_detail=_parse_model_object(
                            raw_repair_detail,
                            label="fixture completion repair chain detail",
                        ),
                        codebook=prepared.codebook,
                        authenticated_units=prepared.authenticated_units,
                    )
                except production.BoundedIdRunnerError as error:
                    materialized = production.materialize_completion_proposal(
                        proposal=None,
                        codebook=prepared.codebook,
                        authenticated_units=prepared.authenticated_units,
                        decision_source="control_fixture_adjudicator_neither_repair",
                    )
                    materialized["semantic_validation_error"] = str(error)
                else:
                    materialized = production.materialize_completion_proposal(
                        proposal=repaired,
                        codebook=prepared.codebook,
                        authenticated_units=prepared.authenticated_units,
                        decision_source="control_fixture_adjudicator_neither_repair",
                    )
    if annotation_pass == "completion_chain":
        _require(prepared.authenticated_units is not None, "completion units lost")
        materialized.setdefault(
            "candidate_unit_bundle_sha256",
            prepared.authenticated_units.bundle_sha256,
        )
    return {
        "raw_adjudication_output": raw_verdict,
        "raw_adjudication_output_sha256": sha256_text(raw_verdict),
        "adjudication_verdict": verdict,
        "adjudication_verdict_error": verdict_error,
        "repair_invoked": verdict == "neither",
        "raw_repair_output": raw_repair,
        "raw_repair_output_sha256": (
            None if raw_repair is None else sha256_text(raw_repair)
        ),
        "raw_repair_detail_output": raw_repair_detail,
        "raw_repair_detail_output_sha256": (
            None if raw_repair_detail is None else sha256_text(raw_repair_detail)
        ),
        "verdict_generation": production.native_stage_provenance(
            verdict_request, verdict_result
        ),
        "repair_generation": repair_generation,
        **materialized,
    }


def run_fixture_adjudication(
    *,
    packet: Mapping[str, Any],
    codebook: Mapping[str, Any],
    annotation_pass: str,
    fixture_candidates: tuple[Mapping[str, Any], Mapping[str, Any]],
    fixture_nonce_sha256: str,
    fixture_config: Mapping[str, Any],
    expected_fixture_config_sha256: str,
    generation_contract: Mapping[str, Any],
    expected_generation_contract_sha256: str,
    generation_context: production.AuthenticatedNativeGenerationContext,
    expected_verdict_seed: int,
    expected_repair_seed: int,
    fixture_lock: Mapping[str, Any],
    expected_fixture_lock_sha256: str,
    generate_native: production.GenerateNative,
    candidate_unit_bundle: Mapping[str, Any] | None = None,
    expected_candidate_unit_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the fixture route; intentionally accepts no expectation object."""

    _require(
        callable(generate_native)
        and isinstance(
            generation_context, production.AuthenticatedNativeGenerationContext
        ),
        "fixture native generation adapter or context invalid",
    )
    _require(
        not generation_api_accepts_expectation_object(),
        "fixture generation API unexpectedly accepts expectation data",
    )
    prepared = _authenticate_fixture_lock(
        fixture_lock=fixture_lock,
        expected_fixture_lock_sha256=expected_fixture_lock_sha256,
        packet=packet,
        codebook=codebook,
        annotation_pass=annotation_pass,
        fixture_candidates=fixture_candidates,
        fixture_nonce_sha256=fixture_nonce_sha256,
        fixture_config=fixture_config,
        expected_fixture_config_sha256=expected_fixture_config_sha256,
        generation_contract=generation_contract,
        expected_generation_contract_sha256=(
            expected_generation_contract_sha256
        ),
        generation_context=generation_context,
        expected_verdict_seed=expected_verdict_seed,
        expected_repair_seed=expected_repair_seed,
        candidate_unit_bundle=candidate_unit_bundle,
        expected_candidate_unit_bundle_sha256=(
            expected_candidate_unit_bundle_sha256
        ),
    )
    outcome = _execute_fixture(
        prepared=prepared,
        expected_fixture_lock_sha256=expected_fixture_lock_sha256,
        generate_native=generate_native,
    )
    lock_copy = copy.deepcopy(dict(fixture_lock))
    record = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": FIXTURE_RECORD_KIND,
        "annotation_pass": annotation_pass,
        "role": FIXTURE_ROLE,
        "packet_id_sha256": prepared.packet["packet_id_sha256"],
        "source_id_sha256": prepared.packet["source_id_sha256"],
        "claim_scope": copy.deepcopy(CLAIM_SCOPE),
        "fixture_provenance": {
            "candidate_origin": "authored_control_fixture",
            "candidate_model_generated": False,
            "candidate_native_model_record_lineage_present": False,
            "candidate_production_record_lineage_claimed": False,
            "adjudicator_execution_mode": INNER_EXECUTION_MODE,
            "adjudicator_model_execution_claimed": False,
            "actual_model_execution_claim_responsibility": (
                OUTER_EXECUTOR_RESPONSIBILITY
            ),
            "precommit_chronology_verified_by_fixture_route": False,
            "expectation_access_chronology_verified_by_fixture_route": False,
            "control_gate_eligible": False,
            "expectation_object_consumed": False,
        },
        "fixture_lock": lock_copy,
        "fixture_lock_sha256": expected_fixture_lock_sha256,
        "fixture_bindings": {
            "packet_sha256": prepared.packet_sha256,
            "candidate_unit_bundle_sha256": (
                None
                if prepared.authenticated_units is None
                else prepared.authenticated_units.bundle_sha256
            ),
            "codebook_sha256": prepared.codebook_sha256,
            "codebook_source_sha256": prepared.codebook_source_sha256,
            "production_runner_sha256": (
                prepared.config.production_runner_sha256
            ),
            "fixture_runner_sha256": fixture_module_sha256(),
            "fixture_config_sha256": prepared.config.config_sha256,
            "generation_contract_sha256": (
                prepared.generation_contract.contract_sha256
            ),
            "runtime_identity_sha256": (
                prepared.generation_contract.runtime_identity_sha256
            ),
            "native_adapter_identity_sha256": (
                prepared.generation_contract.native_adapter_identity_sha256
            ),
            "nonce_precommit_sha256": (
                prepared.generation_contract.nonce_precommit_sha256
            ),
            "outer_nonce_precommit_receipt_sha256": (
                prepared.generation_contract.outer_nonce_precommit_receipt_sha256
            ),
            "original_order_binding_sha256": (
                prepared.original_order_binding_sha256
            ),
            "candidate_order_sha256": (
                prepared.blinded_candidates.order_sha256
            ),
            "model_input_bindings": copy.deepcopy(
                prepared.model_input_bindings
            ),
        },
        "authored_candidate_fixture_sha256s_in_original_order": list(
            prepared.candidate_hashes_original
        ),
        "blinded_original_indexes": list(
            prepared.blinded_candidates.original_indexes
        ),
        "blinded_candidate_fixture_sha256s": list(
            prepared.candidate_hashes_blinded
        ),
        **outcome,
    }
    # Exact record validation is intentionally a separate public operation; the
    # producer invokes it before returning so tamper checks share one contract.
    validate_fixture_adjudication_record(
        record=record,
        packet=packet,
        codebook=codebook,
        annotation_pass=annotation_pass,
        fixture_candidates=fixture_candidates,
        fixture_nonce_sha256=fixture_nonce_sha256,
        fixture_config=fixture_config,
        expected_fixture_config_sha256=expected_fixture_config_sha256,
        generation_contract=generation_contract,
        expected_generation_contract_sha256=(
            expected_generation_contract_sha256
        ),
        generation_context=generation_context,
        expected_verdict_seed=expected_verdict_seed,
        expected_repair_seed=expected_repair_seed,
        fixture_lock=fixture_lock,
        expected_fixture_lock_sha256=expected_fixture_lock_sha256,
        candidate_unit_bundle=candidate_unit_bundle,
        expected_candidate_unit_bundle_sha256=(
            expected_candidate_unit_bundle_sha256
        ),
    )
    return record


def _authenticate_record_stage(
    *,
    value: Any,
    prepared: PreparedFixture,
    fixture_lock_sha256: str,
    model_input_stage: str,
    expected_runtime_stage: str,
    extra_lineage: Mapping[str, Any] | None = None,
) -> tuple[
    production.NativeGenerationRequest,
    production.NativeGenerationResult,
]:
    request, result = production.authenticate_native_stage_provenance(value)
    request_body = production.validate_native_generation_request(
        request, context=prepared.generation_context
    )
    production.validate_native_generation_result(request=request, result=result)
    binding = prepared.model_input_bindings[model_input_stage]
    _require(binding is not None, f"record stage {model_input_stage} unavailable")
    expected_lineage = _stage_lineage(
        prepared=prepared,
        fixture_lock_sha256=fixture_lock_sha256,
        model_input_stage=model_input_stage,
    )
    if extra_lineage:
        expected_lineage.update(copy.deepcopy(dict(extra_lineage)))
    if model_input_stage == "adjudication_verdict":
        expected_messages = prepared.verdict_messages
        expected_schema = prepared.verdict_schema
        expected_seed = prepared.generation_contract.verdict_seed
    elif model_input_stage == "neither_repair_decision":
        expected_messages = prepared.repair_decision_messages
        expected_schema = prepared.repair_decision_schema
        expected_seed = prepared.generation_contract.repair_seed
    else:
        _require(
            model_input_stage == "neither_repair_chain_detail"
            and prepared.repair_detail_messages is not None
            and prepared.repair_detail_schema is not None,
            "record native model-input stage invalid",
        )
        expected_messages = prepared.repair_detail_messages
        expected_schema = prepared.repair_detail_schema
        expected_seed = prepared.generation_contract.repair_seed
    reconstructed = production.build_native_generation_request(
        context=prepared.generation_context,
        messages=expected_messages,
        schema=expected_schema,
        seed=expected_seed,
        stage=expected_runtime_stage,
        annotation_pass=str(prepared.packet["annotation_pass"]),
        packet_id_sha256=str(prepared.packet["packet_id_sha256"]),
        source_id_sha256=str(prepared.packet["source_id_sha256"]),
        lineage_bindings=expected_lineage,
    )
    reconstructed_body = production.validate_native_generation_request(
        reconstructed, context=prepared.generation_context
    )
    _require(
        request.request_sha256 == reconstructed.request_sha256
        and request_body == reconstructed_body
        and request_body["stage"] == expected_runtime_stage
        and request_body["messages_sha256"] == binding["messages_sha256"]
        and request_body["response_schema_sha256"]
        == binding["response_schema_sha256"]
        and request_body["submitted_prompt_token_ids"]
        == binding["submitted_prompt_token_ids"]
        and request_body["submitted_prompt_token_ids_sha256"]
        == binding["submitted_prompt_token_ids_sha256"]
        and request_body["lineage_bindings"] == expected_lineage,
        f"record native stage {expected_runtime_stage} binding invalid",
    )
    return request, result


def _expected_materialization_from_record(
    *,
    record: Mapping[str, Any],
    prepared: PreparedFixture,
    fixture_lock_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate all native stages and recompute the semantic outcome."""

    verdict_request, verdict_result = _authenticate_record_stage(
        value=record["verdict_generation"],
        prepared=prepared,
        fixture_lock_sha256=fixture_lock_sha256,
        model_input_stage="adjudication_verdict",
        expected_runtime_stage="control_fixture_adjudication_verdict",
    )
    raw_verdict = str(verdict_result.body["text"])
    verdict: str | None
    verdict_error: str | None = None
    try:
        verdict = production.validate_adjudication_verdict(
            _parse_model_object(raw_verdict, label="fixture adjudication verdict")
        )
    except production.BoundedIdRunnerError as error:
        verdict = None
        verdict_error = str(error)

    annotation_pass = str(prepared.packet["annotation_pass"])
    raw_repair: str | None = None
    raw_repair_detail: str | None = None
    repair_generation = record["repair_generation"]
    if verdict is None:
        _require(
            repair_generation is None,
            "invalid verdict record must not contain repair generation",
        )
        materialized = production._invalid_adjudication_materialization(
            annotation_pass=annotation_pass,
            error=str(verdict_error),
        )
    elif verdict in {"candidate_1", "candidate_2"}:
        _require(
            repair_generation is None,
            "direct fixture verdict must not contain repair generation",
        )
        selected = prepared.blinded_candidates.proposals[
            0 if verdict == "candidate_1" else 1
        ]
        if annotation_pass == "completion_chain":
            _require(prepared.authenticated_units is not None, "completion units lost")
            materialized = production.materialize_completion_proposal(
                proposal=selected,
                codebook=prepared.codebook,
                authenticated_units=prepared.authenticated_units,
                decision_source=f"control_fixture_adjudicator_{verdict}",
            )
        else:
            materialized = production.materialize_novelty_proposal(
                proposal=selected,
                decision_source=f"control_fixture_adjudicator_{verdict}",
            )
    else:
        repair = dict(_mapping(repair_generation, "fixture repair generation"))
        _require(
            set(repair) == {"decision", "chain_detail"},
            "fixture repair generation fields invalid",
        )
        repair_request, repair_result = _authenticate_record_stage(
            value=repair["decision"],
            prepared=prepared,
            fixture_lock_sha256=fixture_lock_sha256,
            model_input_stage="neither_repair_decision",
            expected_runtime_stage=(
                "control_fixture_completion_repair_decision"
                if annotation_pass == "completion_chain"
                else "control_fixture_novelty_repair_decision"
            ),
            extra_lineage={
                "parent_verdict_request_sha256": verdict_request.request_sha256,
                "parent_verdict_result_sha256": verdict_result.result_sha256,
            },
        )
        raw_repair = str(repair_result.body["text"])
        if annotation_pass == "prefix_novelty":
            _require(
                repair["chain_detail"] is None,
                "novelty fixture repair cannot contain chain detail",
            )
            try:
                novelty_decision = production.validate_novelty_decision(
                    _parse_model_object(raw_repair, label="fixture novelty repair")
                )
            except production.BoundedIdRunnerError as error:
                materialized = production.materialize_novelty_proposal(
                    proposal=None,
                    decision_source="control_fixture_adjudicator_neither_repair",
                )
                materialized["semantic_validation_error"] = str(error)
            else:
                materialized = production.materialize_novelty_proposal(
                    proposal=production.assemble_novelty_proposal(
                        decision=novelty_decision
                    ),
                    decision_source="control_fixture_adjudicator_neither_repair",
                )
        else:
            _require(prepared.authenticated_units is not None, "completion units lost")
            repair_parse_error: str | None = None
            try:
                completion_decision = production.validate_completion_decision(
                    _parse_model_object(raw_repair, label="fixture completion repair")
                )
            except production.BoundedIdRunnerError as error:
                completion_decision = None
                repair_parse_error = str(error)
            if completion_decision is None:
                _require(
                    repair["chain_detail"] is None,
                    "invalid completion repair cannot contain chain detail",
                )
                materialized = production.materialize_completion_proposal(
                    proposal=None,
                    codebook=prepared.codebook,
                    authenticated_units=prepared.authenticated_units,
                    decision_source="control_fixture_adjudicator_neither_repair",
                )
                materialized["semantic_validation_error"] = repair_parse_error
            elif completion_decision != "chain":
                _require(
                    repair["chain_detail"] is None,
                    "non-chain completion repair cannot contain chain detail",
                )
                repaired = production.assemble_completion_proposal(
                    decision=completion_decision,
                    codebook=prepared.codebook,
                    authenticated_units=prepared.authenticated_units,
                )
                materialized = production.materialize_completion_proposal(
                    proposal=repaired,
                    codebook=prepared.codebook,
                    authenticated_units=prepared.authenticated_units,
                    decision_source="control_fixture_adjudicator_neither_repair",
                )
            elif (
                prepared.repair_detail_messages is None
                or prepared.repair_detail_schema is None
            ):
                _require(
                    repair["chain_detail"] is None,
                    "unavailable chain-detail route contains native provenance",
                )
                materialized = production.materialize_completion_proposal(
                    proposal=None,
                    codebook=prepared.codebook,
                    authenticated_units=prepared.authenticated_units,
                    decision_source="control_fixture_adjudicator_neither_repair",
                )
                materialized["semantic_validation_error"] = (
                    "chain repair unavailable for fewer than three candidate units"
                )
            else:
                detail_request, detail_result = _authenticate_record_stage(
                    value=repair["chain_detail"],
                    prepared=prepared,
                    fixture_lock_sha256=fixture_lock_sha256,
                    model_input_stage="neither_repair_chain_detail",
                    expected_runtime_stage=(
                        "control_fixture_completion_repair_chain_detail"
                    ),
                    extra_lineage={
                        "parent_verdict_request_sha256": (
                            verdict_request.request_sha256
                        ),
                        "parent_verdict_result_sha256": (
                            verdict_result.result_sha256
                        ),
                        "parent_repair_decision_request_sha256": (
                            repair_request.request_sha256
                        ),
                        "parent_repair_decision_result_sha256": (
                            repair_result.result_sha256
                        ),
                    },
                )
                raw_repair_detail = str(detail_result.body["text"])
                try:
                    repaired = production.assemble_completion_proposal(
                        decision="chain",
                        chain_detail=_parse_model_object(
                            raw_repair_detail,
                            label="fixture completion repair chain detail",
                        ),
                        codebook=prepared.codebook,
                        authenticated_units=prepared.authenticated_units,
                    )
                except production.BoundedIdRunnerError as error:
                    materialized = production.materialize_completion_proposal(
                        proposal=None,
                        codebook=prepared.codebook,
                        authenticated_units=prepared.authenticated_units,
                        decision_source="control_fixture_adjudicator_neither_repair",
                    )
                    materialized["semantic_validation_error"] = str(error)
                else:
                    materialized = production.materialize_completion_proposal(
                        proposal=repaired,
                        codebook=prepared.codebook,
                        authenticated_units=prepared.authenticated_units,
                        decision_source="control_fixture_adjudicator_neither_repair",
                    )

    if annotation_pass == "completion_chain":
        _require(prepared.authenticated_units is not None, "completion units lost")
        materialized.setdefault(
            "candidate_unit_bundle_sha256",
            prepared.authenticated_units.bundle_sha256,
        )
    stage_fields = {
        "raw_adjudication_output": raw_verdict,
        "raw_adjudication_output_sha256": sha256_text(raw_verdict),
        "adjudication_verdict": verdict,
        "adjudication_verdict_error": verdict_error,
        "repair_invoked": verdict == "neither",
        "raw_repair_output": raw_repair,
        "raw_repair_output_sha256": (
            None if raw_repair is None else sha256_text(raw_repair)
        ),
        "raw_repair_detail_output": raw_repair_detail,
        "raw_repair_detail_output_sha256": (
            None if raw_repair_detail is None else sha256_text(raw_repair_detail)
        ),
    }
    return stage_fields, materialized


def validate_fixture_adjudication_record(
    *,
    record: Mapping[str, Any],
    packet: Mapping[str, Any],
    codebook: Mapping[str, Any],
    annotation_pass: str,
    fixture_candidates: tuple[Mapping[str, Any], Mapping[str, Any]],
    fixture_nonce_sha256: str,
    fixture_config: Mapping[str, Any],
    expected_fixture_config_sha256: str,
    generation_contract: Mapping[str, Any],
    expected_generation_contract_sha256: str,
    generation_context: production.AuthenticatedNativeGenerationContext,
    expected_verdict_seed: int,
    expected_repair_seed: int,
    fixture_lock: Mapping[str, Any],
    expected_fixture_lock_sha256: str,
    candidate_unit_bundle: Mapping[str, Any] | None = None,
    expected_candidate_unit_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    """Reauthenticate a fixture record and all exact native-stage lineage."""

    prepared = _authenticate_fixture_lock(
        fixture_lock=fixture_lock,
        expected_fixture_lock_sha256=expected_fixture_lock_sha256,
        packet=packet,
        codebook=codebook,
        annotation_pass=annotation_pass,
        fixture_candidates=fixture_candidates,
        fixture_nonce_sha256=fixture_nonce_sha256,
        fixture_config=fixture_config,
        expected_fixture_config_sha256=expected_fixture_config_sha256,
        generation_contract=generation_contract,
        expected_generation_contract_sha256=(
            expected_generation_contract_sha256
        ),
        generation_context=generation_context,
        expected_verdict_seed=expected_verdict_seed,
        expected_repair_seed=expected_repair_seed,
        candidate_unit_bundle=candidate_unit_bundle,
        expected_candidate_unit_bundle_sha256=(
            expected_candidate_unit_bundle_sha256
        ),
    )
    value = copy.deepcopy(dict(_mapping(record, "fixture adjudication record")))
    expected_fields = {
        "schema_version",
        "interface_version",
        "kind",
        "annotation_pass",
        "role",
        "packet_id_sha256",
        "source_id_sha256",
        "claim_scope",
        "fixture_provenance",
        "fixture_lock",
        "fixture_lock_sha256",
        "fixture_bindings",
        "authored_candidate_fixture_sha256s_in_original_order",
        "blinded_original_indexes",
        "blinded_candidate_fixture_sha256s",
        "raw_adjudication_output",
        "raw_adjudication_output_sha256",
        "adjudication_verdict",
        "adjudication_verdict_error",
        "repair_invoked",
        "raw_repair_output",
        "raw_repair_output_sha256",
        "raw_repair_detail_output",
        "raw_repair_detail_output_sha256",
        "verdict_generation",
        "repair_generation",
        *MATERIALIZATION_FIELDS,
    }
    if annotation_pass == "completion_chain":
        expected_fields.add("candidate_unit_bundle_sha256")
    _require(set(value) == expected_fields, "fixture adjudication record fields invalid")
    _require(
        value["schema_version"] == SCHEMA_VERSION
        and value["interface_version"] == INTERFACE_VERSION
        and value["kind"] == FIXTURE_RECORD_KIND
        and value["annotation_pass"] == annotation_pass
        and value["role"] == FIXTURE_ROLE
        and value["packet_id_sha256"] == prepared.packet["packet_id_sha256"]
        and value["source_id_sha256"] == prepared.packet["source_id_sha256"]
        and value["claim_scope"] == CLAIM_SCOPE,
        "fixture record identity, packet, or claim scope invalid",
    )
    provenance = dict(_mapping(value["fixture_provenance"], "fixture provenance"))
    expected_provenance_fields = {
        "candidate_origin",
        "candidate_model_generated",
        "candidate_native_model_record_lineage_present",
        "candidate_production_record_lineage_claimed",
        "adjudicator_execution_mode",
        "adjudicator_model_execution_claimed",
        "actual_model_execution_claim_responsibility",
        "precommit_chronology_verified_by_fixture_route",
        "expectation_access_chronology_verified_by_fixture_route",
        "control_gate_eligible",
        "expectation_object_consumed",
    }
    _require(
        set(provenance) == expected_provenance_fields
        and provenance["candidate_origin"] == "authored_control_fixture"
        and provenance["candidate_model_generated"] is False
        and provenance["candidate_native_model_record_lineage_present"] is False
        and provenance["candidate_production_record_lineage_claimed"] is False
        and provenance["adjudicator_execution_mode"] == INNER_EXECUTION_MODE
        and provenance["adjudicator_model_execution_claimed"] is False
        and provenance["actual_model_execution_claim_responsibility"]
        == OUTER_EXECUTOR_RESPONSIBILITY
        and provenance["precommit_chronology_verified_by_fixture_route"]
        is False
        and provenance[
            "expectation_access_chronology_verified_by_fixture_route"
        ]
        is False
        and provenance["control_gate_eligible"] is False
        and provenance["expectation_object_consumed"] is False,
        "fixture record provenance invalid",
    )
    _require(
        value["fixture_lock"] == fixture_lock
        and value["fixture_lock_sha256"] == expected_fixture_lock_sha256,
        "fixture record lock binding invalid",
    )
    expected_bindings = {
        "packet_sha256": prepared.packet_sha256,
        "candidate_unit_bundle_sha256": (
            None
            if prepared.authenticated_units is None
            else prepared.authenticated_units.bundle_sha256
        ),
        "codebook_sha256": prepared.codebook_sha256,
        "codebook_source_sha256": prepared.codebook_source_sha256,
        "production_runner_sha256": prepared.config.production_runner_sha256,
        "fixture_runner_sha256": fixture_module_sha256(),
        "fixture_config_sha256": prepared.config.config_sha256,
        "generation_contract_sha256": (
            prepared.generation_contract.contract_sha256
        ),
        "runtime_identity_sha256": (
            prepared.generation_contract.runtime_identity_sha256
        ),
        "native_adapter_identity_sha256": (
            prepared.generation_contract.native_adapter_identity_sha256
        ),
        "nonce_precommit_sha256": (
            prepared.generation_contract.nonce_precommit_sha256
        ),
        "outer_nonce_precommit_receipt_sha256": (
            prepared.generation_contract.outer_nonce_precommit_receipt_sha256
        ),
        "original_order_binding_sha256": prepared.original_order_binding_sha256,
        "candidate_order_sha256": prepared.blinded_candidates.order_sha256,
        "model_input_bindings": copy.deepcopy(prepared.model_input_bindings),
    }
    _require(
        value["fixture_bindings"] == expected_bindings
        and value["authored_candidate_fixture_sha256s_in_original_order"]
        == list(prepared.candidate_hashes_original)
        and value["blinded_original_indexes"]
        == list(prepared.blinded_candidates.original_indexes)
        and value["blinded_candidate_fixture_sha256s"]
        == list(prepared.candidate_hashes_blinded),
        "fixture record candidate, order, or model-input binding invalid",
    )
    stage_fields, expected_materialized = _expected_materialization_from_record(
        record=value,
        prepared=prepared,
        fixture_lock_sha256=expected_fixture_lock_sha256,
    )
    for field, expected in stage_fields.items():
        _require(value[field] == expected, f"fixture record {field} invalid")
    for field in MATERIALIZATION_FIELDS:
        _require(
            value[field] == expected_materialized[field],
            f"fixture record materialized {field} invalid",
        )
    if annotation_pass == "completion_chain":
        _require(
            value["candidate_unit_bundle_sha256"]
            == expected_materialized["candidate_unit_bundle_sha256"]
            == prepared.authenticated_units.bundle_sha256,
            "fixture record materialization bundle hash invalid",
        )
    return value
