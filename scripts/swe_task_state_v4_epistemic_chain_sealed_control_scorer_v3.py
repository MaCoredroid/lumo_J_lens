#!/usr/bin/env python3
"""Key-read-last scorer for the prospective V3 development controls.

The sole public entry point authenticates this scorer, the separately frozen
executor implementation, the executor read trace, and the finalized public
bundle before it opens either semantic key.  The main control key and fixture
key are distinct files and are the final two input reads.  This version can
only emit a development score receipt with every execution, gate, and COT-like
claim false; a real sealed run requires a new reviewed version.

No model/runtime package is imported here.  The executor source is compiled
from already authenticated bytes, so validating its interface cannot introduce
an unauthenticated second read of that source.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import types
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
INTERFACE_VERSION = 3
CONFIG_KIND = "swe_task_state_v4_epistemic_chain_sealed_control_scorer_v3"
RECEIPT_KIND = "swe_task_state_v4_epistemic_chain_control_score_receipt_v3"
MAIN_KEY_KIND = "swe_task_state_v4_epistemic_chain_control_key_manifest_draft_v3"
FIXTURE_KEY_KIND = "swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_key_manifest_draft_v3"
COMPLETION_KEY_ROW_KIND = (
    "swe_task_state_v4_epistemic_chain_completion_key_row_draft_v3"
)
NOVELTY_KEY_ROW_KIND = "swe_task_state_v4_epistemic_chain_novelty_key_row_draft_v3"
FIXTURE_LOCK_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_control_fixture_lock_v3"
)
FIXTURE_GENERATION_CONTRACT_KIND = (
    "swe_task_state_v4_epistemic_chain_adjudication_fixture_generation_contract_v3"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SealedControlScorerError(RuntimeError):
    """Raised when the scorer cannot preserve its key-read-last contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SealedControlScorerError(message)


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


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be lowercase SHA-256",
    )
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be an array",
    )
    return list(value)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        _require(name not in result, f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise SealedControlScorerError(f"non-finite JSON value: {value}")


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SealedControlScorerError(f"cannot parse {label}: {error}") from error
    return _mapping(parsed, label)


def _control_ids() -> tuple[str, ...]:
    return tuple(f"C{index:02d}" for index in range(1, 33)) + tuple(
        f"V{index:02d}" for index in range(1, 9)
    )


def _fixture_ids() -> tuple[str, ...]:
    return tuple(f"F{index:02d}" for index in range(1, 13))


def _role_ids() -> tuple[str, ...]:
    return ("independent_a", "independent_b", "adjudicator")


def _false_claims() -> dict[str, bool]:
    # Always allocate a fresh literal.  No mutable module-level object can be
    # changed to promote a CPU/development receipt into execution evidence.
    return {
        "actual_model_execution_established": False,
        "gate_eligible": False,
        "sealed_control_run_established": False,
        "annotation_interface_readiness_established": False,
        "private_chain_of_thought_recovery_established": False,
        "latent_cot_like_trajectory_recovery_established": False,
        "emotion_affect_confidence_doubt_or_stress_recovery_established": False,
    }


@dataclass
class _ReadLedger:
    events: list[dict[str, Any]] = field(default_factory=list)
    public_evidence_verified: bool = False
    key_reads: list[str] = field(default_factory=list)

    def mark_public_verified(self) -> None:
        _require(not self.public_evidence_verified, "public evidence verified twice")
        _require(not self.key_reads, "key read preceded public verification")
        self.public_evidence_verified = True

    def before_read(self, label: str, *, key_read: bool) -> None:
        if key_read:
            _require(
                self.public_evidence_verified,
                "semantic key access attempted before public evidence verification",
            )
            expected = (
                "main_control_key" if not self.key_reads else "fixture_adjudication_key"
            )
            _require(
                len(self.key_reads) < 2 and label == expected,
                "semantic keys must be the final two input reads in fixed order",
            )
            self.key_reads.append(label)
        else:
            _require(
                not self.key_reads,
                "non-key input read attempted after semantic key access began",
            )

    def after_read(
        self, *, label: str, path: Path, observed_sha256: str, key_read: bool
    ) -> None:
        self.events.append(
            {
                "ordinal": len(self.events),
                "input_class": "semantic_key" if key_read else "public_or_code",
                "label": label,
                "resolved_path_sha256": sha256_bytes(str(path).encode("utf-8")),
                "observed_sha256": observed_sha256,
            }
        )

    def final_projection(self) -> dict[str, Any]:
        _require(
            self.public_evidence_verified
            and self.key_reads == ["main_control_key", "fixture_adjudication_key"],
            "scorer input chronology incomplete",
        )
        _require(
            [item["input_class"] for item in self.events[-2:]]
            == ["semantic_key", "semantic_key"],
            "semantic keys were not the final two input reads",
        )
        return {
            "input_read_count": len(self.events),
            "ordered_input_read_sha256": sha256_value(self.events),
            "public_evidence_verified_before_key_reads": True,
            "key_read_order": ["main_control_key", "fixture_adjudication_key"],
            "keys_are_final_two_input_reads": True,
        }


def _read_authenticated_bytes(
    *,
    path: Path,
    expected_sha256: str,
    label: str,
    ledger: _ReadLedger,
    key_read: bool = False,
) -> tuple[bytes, Path]:
    expected = _sha256(expected_sha256, f"expected {label} hash")
    ledger.before_read(label, key_read=key_read)
    candidate = Path(path)
    _require(not candidate.is_symlink(), f"{label} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        _require(resolved.is_file(), f"{label} is not a regular file")
        raw = resolved.read_bytes()
    except OSError as error:
        raise SealedControlScorerError(f"cannot read {label}: {error}") from error
    observed = sha256_bytes(raw)
    _require(observed == expected, f"{label} differs from external exact-byte hash")
    ledger.after_read(
        label=label,
        path=resolved,
        observed_sha256=observed,
        key_read=key_read,
    )
    return raw, resolved


def _validate_config(value: Any) -> dict[str, Any]:
    config = _mapping(value, "scorer config")
    _require(
        set(config)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "status",
            "scope",
            "authorization",
            "bindings",
            "key_read_contract",
            "suite_contract",
            "proposal_contract",
            "receipt_contract",
            "blockers",
        },
        "scorer config fields changed",
    )
    _require(
        config["schema_version"] == SCHEMA_VERSION
        and config["interface_version"] == INTERFACE_VERSION
        and config["kind"] == CONFIG_KIND
        and config["status"] == "prospective_development_scorer_unauthorized_not_run",
        "scorer config identity invalid",
    )
    scope = _mapping(config["scope"], "scorer scope")
    _require(
        scope
        == {
            "development_controls_only": True,
            "model_or_gpu_execution_performed": False,
            "sealed_control_run_claimed": False,
            "private_or_verbatim_chain_of_thought_recovery_claimed": False,
            "latent_cot_like_trajectory_recovery_claimed": False,
            "affect_emotion_confidence_doubt_or_stress_recovery_claimed": False,
        },
        "scorer scope changed",
    )
    authorization = _mapping(config["authorization"], "scorer authorization")
    _require(
        set(authorization)
        == {
            "model_access_authorized",
            "gpu_access_authorized",
            "generation_authorized",
            "gate_evidence_authorized",
            "sealed_run_receipt_authorized",
        }
        and all(item is False for item in authorization.values()),
        "scorer configuration authorizes execution or gate evidence",
    )
    bindings = _mapping(config["bindings"], "scorer bindings")
    _require(
        set(bindings) == {"scorer_source", "executor_config", "executor_source"},
        "scorer binding names changed",
    )
    for name, expected_path in (
        (
            "scorer_source",
            "scripts/swe_task_state_v4_epistemic_chain_sealed_control_scorer_v3.py",
        ),
        (
            "executor_config",
            "configs/swe_task_state_v4_epistemic_chain_sealed_control_executor_v3.json",
        ),
        (
            "executor_source",
            "scripts/swe_task_state_v4_epistemic_chain_sealed_control_executor_v3.py",
        ),
    ):
        binding = _mapping(bindings[name], f"binding {name}")
        _require(
            set(binding) == {"path", "sha256"}
            and binding["path"] == expected_path
            and SHA256_RE.fullmatch(str(binding["sha256"])) is not None,
            f"binding {name} invalid",
        )
    key_contract = _mapping(config["key_read_contract"], "key-read contract")
    _require(
        key_contract
        == {
            "public_bundle_and_trace_authenticated_first": True,
            "all_public_locks_authenticated_before_key_read": True,
            "main_control_key_read_ordinal_from_end": 2,
            "fixture_adjudication_key_read_ordinal_from_end": 1,
            "distinct_resolved_paths_required": True,
            "distinct_device_inode_required": True,
            "symlinks_forbidden": True,
            "no_non_key_input_read_after_first_key": True,
            "external_exact_byte_and_manifest_hashes_required": True,
        },
        "key-read contract weakened",
    )
    suite = _mapping(config["suite_contract"], "suite contract")
    _require(
        suite
        == {
            "roles": list(_role_ids()),
            "ordered_control_ids": list(_control_ids()),
            "ordered_fixture_ids": list(_fixture_ids()),
            "results_per_role": 40,
            "native_results_per_role": 39,
            "sole_host_bypass_control_id": "C32",
            "fixture_results": 12,
            "single_suite_nonce": True,
            "nonce_retry_forbidden": True,
            "frozen_fixture_seed_schedule_required": True,
        },
        "suite contract changed",
    )
    proposal = _mapping(config["proposal_contract"], "proposal contract")
    _require(
        proposal
        == {
            "completion_decisions": ["chain", "no_chain", "unknown"],
            "novelty_decisions": [
                "novel",
                "prefix_exposed",
                "ambiguous",
                "unknown",
            ],
            "completion_unknown_reason": "completion_semantics_ambiguous",
            "novelty_unknown_reason": "novelty_semantics_ambiguous",
            "evidence_kind": [
                "code",
                "tool_or_test",
                "spec_contract",
                "environment",
            ],
            "belief_edge": ["supports", "refutes", "narrows"],
            "hypothesis_domain": [
                "source_logic",
                "interface_contract",
                "data_type_shape",
                "environment_dependency",
                "tooling_path",
                "test_fixture",
                "other",
            ],
            "action_intent": ["inspect", "edit", "validate"],
        },
        "proposal contract changed",
    )
    receipt = _mapping(config["receipt_contract"], "receipt contract")
    _require(
        receipt
        == {
            "exclusive_create_required": True,
            "self_hash_envelope_required": True,
            "all_claims_false": True,
            "caller_authoritative_labels_forbidden_in_public_bundle": True,
            "real_sealed_run_requires_new_reviewed_version": True,
        },
        "receipt contract weakened",
    )
    blockers = _sequence(config["blockers"], "scorer blockers")
    _require(
        len(blockers) >= 3
        and all(isinstance(item, str) and bool(item) for item in blockers),
        "scorer blockers absent",
    )
    return copy.deepcopy(config)


def _authenticate_config_and_code(
    *,
    config_path: Path,
    expected_config_sha256: str,
    expected_scorer_source_sha256: str,
    expected_executor_config_sha256: str,
    expected_executor_source_sha256: str,
    ledger: _ReadLedger,
) -> tuple[dict[str, Any], bytes, Path, dict[str, Any], bytes, Path]:
    config_raw, resolved_config = _read_authenticated_bytes(
        path=config_path,
        expected_sha256=expected_config_sha256,
        label="scorer_config",
        ledger=ledger,
    )
    config = _validate_config(_strict_json_bytes(config_raw, "scorer config"))
    repository_root = resolved_config.parents[1]
    bindings = config["bindings"]

    scorer_source_raw, _scorer_source_path = _read_authenticated_bytes(
        path=repository_root / bindings["scorer_source"]["path"],
        expected_sha256=expected_scorer_source_sha256,
        label="scorer_source",
        ledger=ledger,
    )
    _require(
        bindings["scorer_source"]["sha256"] == expected_scorer_source_sha256,
        "scorer source differs from config binding",
    )
    executor_config_raw, _executor_config_path = _read_authenticated_bytes(
        path=repository_root / bindings["executor_config"]["path"],
        expected_sha256=expected_executor_config_sha256,
        label="executor_config",
        ledger=ledger,
    )
    executor_config = _strict_json_bytes(executor_config_raw, "executor config")
    _require(
        bindings["executor_config"]["sha256"] == expected_executor_config_sha256,
        "executor config differs from scorer binding",
    )
    executor_source_raw, executor_source_path = _read_authenticated_bytes(
        path=repository_root / bindings["executor_source"]["path"],
        expected_sha256=expected_executor_source_sha256,
        label="executor_source",
        ledger=ledger,
    )
    _require(
        bindings["executor_source"]["sha256"] == expected_executor_source_sha256,
        "executor source differs from scorer binding",
    )
    _require(
        sha256_bytes(scorer_source_raw) == expected_scorer_source_sha256,
        "scorer source authentication changed during startup",
    )
    return (
        config,
        scorer_source_raw,
        resolved_config,
        executor_config,
        executor_source_raw,
        executor_source_path,
    )


def _load_authenticated_executor_module(
    *, source: bytes, source_path: Path
) -> types.ModuleType:
    """Compile authenticated bytes directly; never ask a loader to reread them."""

    module_name = (
        "_authenticated_sealed_control_executor_v3_" + sha256_bytes(source)[:16]
    )
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(source, str(source_path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    for name in ("validate_read_trace_envelope", "validate_public_generation_bundle"):
        _require(
            callable(getattr(module, name, None)), f"executor export {name} absent"
        )
    return module


def _validate_hash_envelope(
    value: Any,
    *,
    body_field: str,
    hash_field: str,
    expected_body_sha256: str,
    label: str,
) -> dict[str, Any]:
    envelope = _mapping(value, f"{label} envelope")
    _require(
        set(envelope) == {body_field, hash_field},
        f"{label} envelope fields invalid",
    )
    body = _mapping(envelope[body_field], f"{label} body")
    observed = sha256_value(body)
    _require(
        envelope[hash_field]
        == observed
        == _sha256(expected_body_sha256, f"expected {label} body hash"),
        f"{label} differs from external authenticated body hash",
    )
    return body


def _assert_no_caller_labels(value: Any, *, path: tuple[str, ...] = ()) -> None:
    forbidden_exact = {
        "gold",
        "goldlabel",
        "expectedlabel",
        "target",
        "targetlabel",
        "winner",
        "correct",
        "incorrect",
        "answerkey",
        "score",
        "accuracy",
    }
    if isinstance(value, Mapping):
        for raw_name, item in value.items():
            _require(isinstance(raw_name, str), "public bundle field name invalid")
            normalized = re.sub(r"[^a-z0-9]", "", raw_name.lower())
            _require(
                normalized not in forbidden_exact
                and not normalized.startswith("gold")
                and not normalized.endswith("label"),
                "caller-authoritative label field in public bundle: "
                + ".".join(path + (raw_name,)),
            )
            _assert_no_caller_labels(item, path=path + (raw_name,))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_no_caller_labels(item, path=path + (str(index),))


def _validate_proposal(
    value: Any, *, annotation_pass: str, proposal_contract: Mapping[str, Any]
) -> dict[str, Any]:
    proposal = _mapping(value, "finite semantic projection")
    decision = proposal.get("decision")
    if annotation_pass == "completion_chain":
        _require(
            decision in proposal_contract["completion_decisions"],
            "completion decision outside frozen schema",
        )
        if decision == "chain":
            fields = {
                "decision",
                "evidence_unit_id",
                "hypothesis_unit_id",
                "action_unit_id",
                "evidence_kind",
                "belief_edge",
                "hypothesis_domain",
                "action_intent",
            }
            _require(set(proposal) == fields, "completion chain fields invalid")
            ids = [
                proposal["evidence_unit_id"],
                proposal["hypothesis_unit_id"],
                proposal["action_unit_id"],
            ]
            _require(
                all(isinstance(item, str) and bool(item) for item in ids)
                and len(set(ids)) == 3,
                "completion chain unit IDs invalid",
            )
            for field_name in (
                "evidence_kind",
                "belief_edge",
                "hypothesis_domain",
                "action_intent",
            ):
                _require(
                    proposal[field_name] in proposal_contract[field_name],
                    f"completion {field_name} outside frozen ontology",
                )
        elif decision == "unknown":
            _require(
                proposal
                == {
                    "decision": "unknown",
                    "unknown_reason": proposal_contract["completion_unknown_reason"],
                },
                "completion unknown projection invalid",
            )
        else:
            _require(
                proposal == {"decision": "no_chain"},
                "completion no-chain projection invalid",
            )
    else:
        _require(annotation_pass == "prefix_novelty", "annotation pass invalid")
        _require(
            decision in proposal_contract["novelty_decisions"],
            "novelty decision outside frozen schema",
        )
        if decision == "unknown":
            _require(
                proposal
                == {
                    "decision": "unknown",
                    "unknown_reason": proposal_contract["novelty_unknown_reason"],
                },
                "novelty unknown projection invalid",
            )
        else:
            _require(set(proposal) == {"decision"}, "novelty fields invalid")
    return copy.deepcopy(proposal)


def _validate_materialized_span(
    value: Any, *, expected_unit_id: str, label: str
) -> dict[str, Any]:
    span = _mapping(value, label)
    _require(
        set(span)
        == {
            "unit_id",
            "assistant_char_start",
            "assistant_char_end",
            "source_char_start",
            "source_char_end",
            "text",
            "text_sha256",
        }
        and span["unit_id"] == expected_unit_id
        and all(
            isinstance(span[name], int) and not isinstance(span[name], bool)
            for name in (
                "assistant_char_start",
                "assistant_char_end",
                "source_char_start",
                "source_char_end",
            )
        )
        and 0 <= span["assistant_char_start"] < span["assistant_char_end"]
        and 0 <= span["source_char_start"] < span["source_char_end"]
        and span["assistant_char_end"] - span["assistant_char_start"]
        == span["source_char_end"] - span["source_char_start"]
        and isinstance(span["text"], str)
        and bool(span["text"])
        and len(span["text"])
        == span["assistant_char_end"] - span["assistant_char_start"]
        and span["text_sha256"] == sha256_bytes(span["text"].encode("utf-8")),
        f"{label} materialized unit span invalid",
    )
    return copy.deepcopy(span)


def _validate_main_key_materialization(
    value: Any,
    *,
    proposal: Mapping[str, Any],
    annotation_pass: str,
    label: str,
) -> dict[str, Any]:
    materialized = _mapping(value, f"{label} materialization")
    common_fields = {
        "raw_semantic_decision",
        "raw_semantic_proposal",
        "decision_source",
        "semantic_validation_status",
        "semantic_validation_error",
        "materialization_status",
        "interface_unknown_reason",
        "annotation_record",
    }
    expected_fields = (
        common_fields | {"candidate_unit_bundle_sha256"}
        if annotation_pass == "completion_chain"
        else common_fields
    )
    _require(
        set(materialized) == expected_fields
        and materialized["raw_semantic_decision"] == proposal["decision"]
        and materialized["raw_semantic_proposal"] == proposal
        and materialized["decision_source"] == "prospective_control_key_draft"
        and materialized["semantic_validation_status"] == "valid"
        and materialized["semantic_validation_error"] is None
        and materialized["interface_unknown_reason"] is None,
        f"{label} materialized projection fields invalid",
    )
    annotation = _mapping(
        materialized["annotation_record"], f"{label} annotation record"
    )
    if annotation_pass == "completion_chain":
        _sha256(
            materialized["candidate_unit_bundle_sha256"],
            f"{label} candidate-unit bundle hash",
        )
        _require(
            set(annotation)
            == {
                "annotation_status",
                "unknown_reason",
                "has_chain",
                "evidence_span",
                "hypothesis_span",
                "action_span",
                "evidence_kind",
                "belief_edge",
                "hypothesis_domain",
                "action_intent",
                "relation_marker_present",
                "action_marker_present",
                "exact_signature",
            },
            f"{label} completion annotation fields invalid",
        )
        if proposal["decision"] == "chain":
            _require(
                materialized["materialization_status"]
                == "resolved_authenticated_unit_chain"
                and annotation["annotation_status"] == "available"
                and annotation["unknown_reason"] is None
                and annotation["has_chain"] is True
                and all(
                    isinstance(annotation[name], bool)
                    for name in (
                        "relation_marker_present",
                        "action_marker_present",
                    )
                )
                and all(
                    annotation[name] == proposal[name]
                    for name in (
                        "evidence_kind",
                        "belief_edge",
                        "hypothesis_domain",
                        "action_intent",
                    )
                )
                and annotation["exact_signature"]
                == ">".join(
                    [
                        str(proposal["evidence_kind"]),
                        str(proposal["belief_edge"]),
                        str(proposal["hypothesis_domain"]),
                        "motivates",
                        str(proposal["action_intent"]),
                    ]
                ),
                f"{label} completion chain materialization invalid",
            )
            for span_name, unit_name in (
                ("evidence_span", "evidence_unit_id"),
                ("hypothesis_span", "hypothesis_unit_id"),
                ("action_span", "action_unit_id"),
            ):
                _validate_materialized_span(
                    annotation[span_name],
                    expected_unit_id=str(proposal[unit_name]),
                    label=f"{label} {span_name}",
                )
        else:
            unknown = proposal["decision"] == "unknown"
            _require(
                materialized["materialization_status"]
                == (
                    "not_applicable_semantic_unknown"
                    if unknown
                    else "not_applicable_no_chain"
                )
                and annotation["annotation_status"]
                == ("semantic_unknown" if unknown else "available")
                and annotation["unknown_reason"]
                == ("completion_semantics_ambiguous" if unknown else None)
                and annotation["has_chain"] is (None if unknown else False)
                and all(
                    annotation[name] is None
                    for name in (
                        "evidence_span",
                        "hypothesis_span",
                        "action_span",
                        "evidence_kind",
                        "belief_edge",
                        "hypothesis_domain",
                        "action_intent",
                        "relation_marker_present",
                        "action_marker_present",
                        "exact_signature",
                    )
                ),
                f"{label} empty completion materialization invalid",
            )
    else:
        _require(
            annotation_pass == "prefix_novelty"
            and set(annotation)
            == {"annotation_status", "unknown_reason", "novelty_status"},
            f"{label} novelty annotation fields invalid",
        )
        unknown = proposal["decision"] == "unknown"
        _require(
            materialized["materialization_status"]
            == (
                "not_applicable_semantic_unknown"
                if unknown
                else "not_applicable_novelty_classification"
            )
            and annotation["annotation_status"]
            == ("semantic_unknown" if unknown else "available")
            and annotation["unknown_reason"]
            == ("novelty_semantics_ambiguous" if unknown else None)
            and annotation["novelty_status"]
            == (None if unknown else proposal["decision"]),
            f"{label} novelty materialization invalid",
        )
    return copy.deepcopy(materialized)


def _validate_native_evidence_projection(
    value: Any, *, role: str, label: str
) -> dict[str, Any]:
    evidence = _mapping(value, label)
    _require(
        set(evidence)
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
        and evidence["adapter_role"] == role
        and evidence["actual_model_execution"] is True
        and evidence["model_loaded"] is True
        and evidence["generation_performed"] is True
        and evidence["gate_eligible"] is True,
        f"{label} native evidence identity or execution facts invalid",
    )
    for name in ("adapter_config_sha256", "adapter_source_sha256"):
        _sha256(evidence[name], f"{label} {name}")
    lists: dict[str, list[Any]] = {}
    for name in (
        "outer_launch_authorization_sha256s",
        "launch_binding_sha256s",
        "preflight_receipt_sha256s",
        "runtime_receipt_sha256s",
        "native_request_sha256s",
        "native_result_sha256s",
    ):
        items = _sequence(evidence[name], f"{label} {name}")
        _require(bool(items), f"{label} {name} empty")
        for item in items:
            _sha256(item, f"{label} {name} item")
        _require(len(items) == len(set(items)), f"{label} {name} replay detected")
        lists[name] = items
    _require(
        len(lists["outer_launch_authorization_sha256s"])
        == len(lists["launch_binding_sha256s"])
        == len(lists["preflight_receipt_sha256s"])
        == len(lists["runtime_receipt_sha256s"])
        and len(lists["native_request_sha256s"]) == len(lists["native_result_sha256s"]),
        f"{label} native receipt/request/result cardinality invalid",
    )
    return copy.deepcopy(evidence)


def _validate_public_trace_independently(trace: Any) -> dict[str, Any]:
    value = _mapping(trace, "executor read trace")
    _require(
        set(value)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "suite_id",
            "events",
            "event_count",
            "head_sha256",
            "closed",
        },
        "executor read trace fields invalid",
    )
    events = [
        _mapping(item, "executor trace event")
        for item in _sequence(value["events"], "executor trace events")
    ]
    _require(
        value["schema_version"] == SCHEMA_VERSION
        and value["interface_version"] == INTERFACE_VERSION
        and value["kind"] == "swe_task_state_v4_epistemic_chain_read_trace_v3"
        and isinstance(value["suite_id"], str)
        and len(value["suite_id"]) >= 16
        and value["event_count"] == len(events)
        and value["closed"] is True
        and bool(events),
        "executor read trace identity or closure invalid",
    )
    event_fields = {
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
    }
    previous = "0" * 64
    previous_rank = -1
    previous_time = -1
    stage_ranks = {
        "freeze_helper": 0,
        "precommit_nonce": 1,
        "run_primary": 2,
        "lock_primaries": 3,
        "run_adjudicator": 4,
        "lock_all": 5,
    }
    event_types = {"read", "write", "transition", "consume"}
    artifact_classes = {
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
    for ordinal, event in enumerate(events):
        _require(set(event) == event_fields, "executor trace event fields invalid")
        _require(
            event["ordinal"] == ordinal
            and event["stage"] in stage_ranks
            and isinstance(event["stage_rank"], int)
            and not isinstance(event["stage_rank"], bool)
            and event["stage_rank"] == stage_ranks[event["stage"]]
            and event["stage_rank"] >= previous_rank
            and event["event_type"] in event_types
            and event["artifact_class"] in artifact_classes
            and isinstance(event["artifact_id"], str)
            and bool(event["artifact_id"])
            and isinstance(event["monotonic_ns"], int)
            and not isinstance(event["monotonic_ns"], bool)
            and event["monotonic_ns"] > previous_time,
            "executor trace order or monotonic chronology invalid",
        )
        _require(
            event["previous_event_sha256"] == previous,
            "executor trace hash-chain predecessor invalid",
        )
        if event["event_type"] == "read":
            _require(
                isinstance(event["path"], str)
                and Path(event["path"]).is_absolute()
                and _sha256(
                    event["expected_sha256"], "executor trace expected read hash"
                )
                == _sha256(
                    event["observed_sha256"], "executor trace observed read hash"
                ),
                "executor trace read event invalid",
            )
        elif event["event_type"] in {"write", "consume"}:
            _require(
                isinstance(event["path"], str)
                and Path(event["path"]).is_absolute()
                and event["expected_sha256"] is None,
                "executor trace write/consume event invalid",
            )
            _sha256(event["observed_sha256"], "executor trace write/consume hash")
        else:
            _require(
                event["path"] is None
                and event["expected_sha256"] is None
                and event["observed_sha256"] is None
                and event["artifact_class"] == "stage",
                "executor trace transition event invalid",
            )
        payload = {name: event[name] for name in event_fields - {"event_sha256"}}
        observed = sha256_value(payload)
        _require(
            event["event_sha256"] == observed,
            "executor trace event hash invalid",
        )
        text = " ".join(
            str(event[name]).lower()
            for name in ("event_type", "artifact_class", "artifact_id", "path")
        )
        _require("retry" not in text, "retry event appears in frozen suite trace")
        # The fixture-generation key is legitimately consumed once after the
        # dual-primary lock.  Only the separate scoring key must remain unread.
        _require(
            not (
                "control_scoring_key" in text
                or "control_key" in text
                or "scoring_key" in text
                or "expectation_key" in text
            ),
            "control scoring key access appears in executor trace",
        )
        previous = observed
        previous_rank = int(event["stage_rank"])
        previous_time = int(event["monotonic_ns"])
    _require(value["head_sha256"] == previous, "executor trace head invalid")
    _require(
        events[-1]["event_type"] == "transition"
        and events[-1]["stage"] == "lock_all"
        and events[-1]["artifact_id"] == "trace_closed",
        "executor trace did not close at final lock",
    )
    transitions = [
        event["artifact_id"] for event in events if event["event_type"] == "transition"
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
        "executor trace required transition missing, repeated, or replayed",
    )
    positions = {name: transitions.index(name) for name in required_once}
    _require(
        positions["freeze_complete"]
        < positions["nonce_precommitted"]
        < min(
            positions["primary_independent_a_complete"],
            positions["primary_independent_b_complete"],
        )
        and max(
            positions["primary_independent_a_complete"],
            positions["primary_independent_b_complete"],
        )
        < positions["dual_primary_lock_complete"]
        < positions["adjudicator_complete"]
        < positions["final_lock_complete"]
        < positions["trace_closed"],
        "executor trace stage-transition order invalid",
    )
    dual_time = next(
        event["monotonic_ns"]
        for event in events
        if event["artifact_id"] == "dual_primary_lock_complete"
    )
    fixture_key_reads = [
        event
        for event in events
        if event["artifact_class"] == "fixture_generation_key"
        and event["event_type"] == "read"
    ]
    fixture_materializer_reads = [
        event
        for event in events
        if event["artifact_class"] == "fixture_key_materializer"
        and event["event_type"] == "read"
    ]
    nonce_consumptions = [
        event
        for event in events
        if event["artifact_class"] == "nonce_secret"
        and event["event_type"] == "consume"
    ]
    _require(
        len(fixture_key_reads) == 1
        and len(fixture_materializer_reads) == 1
        and len(nonce_consumptions) == 1
        and all(
            event["stage"] == "run_adjudicator" and event["monotonic_ns"] > dual_time
            for event in fixture_key_reads
            + fixture_materializer_reads
            + nonce_consumptions
        ),
        "executor fixture key/materializer/nonce chronology invalid",
    )
    return copy.deepcopy(value)


def _validate_public_bundle_independently(
    manifest: Any,
    *,
    trace: Mapping[str, Any],
    executor_config: Mapping[str, Any],
    proposal_contract: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(manifest, "public generation manifest")
    _require(
        set(value)
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
        },
        "public generation manifest fields invalid",
    )
    _assert_no_caller_labels(value)
    _require(
        value["schema_version"] == SCHEMA_VERSION
        and value["interface_version"] == INTERFACE_VERSION
        and value["kind"]
        == "swe_task_state_v4_epistemic_chain_public_generation_bundle_v3"
        and value["status"] == "all_generation_outputs_locked_scoring_keys_unread"
        and value["suite_id"] == trace["suite_id"],
        "public bundle identity differs from read trace",
    )
    _mapping(value["scope"], "public bundle scope")
    _mapping(value["executor_identity"], "public executor identity")
    public_claims = _mapping(value["claims"], "public bundle claims")
    _require(
        public_claims
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
        },
        "public bundle execution/lock/scoring claim contract invalid",
    )
    filesystem_roots = _mapping(value["filesystem_roots"], "public filesystem roots")
    _require(
        set(filesystem_roots) == {"generation_root", "key_root"}
        and all(
            isinstance(filesystem_roots[name], str) and bool(filesystem_roots[name])
            for name in ("generation_root", "key_root")
        )
        and filesystem_roots["generation_root"] != filesystem_roots["key_root"],
        "public filesystem-root contract invalid",
    )
    inputs = _mapping(value["inputs"], "public inputs")
    _require(set(inputs) == {"control", "fixture"}, "public input names invalid")
    for input_name in ("control", "fixture"):
        item = _mapping(inputs[input_name], f"public {input_name} input")
        _require(
            set(item) == {"path", "sha256"}
            and isinstance(item["path"], str)
            and bool(item["path"]),
            f"public {input_name} input fields invalid",
        )
        _sha256(item["sha256"], f"public {input_name} input hash")
    commitments = _mapping(value["key_commitments"], "public key commitments")
    _require(
        set(commitments) == {"fixture_generation_key", "control_scoring_key"},
        "public key commitment names invalid",
    )
    fixture_commitment = _mapping(
        commitments["fixture_generation_key"], "fixture-generation key commitment"
    )
    scoring_commitment = _mapping(
        commitments["control_scoring_key"], "control-scoring key commitment"
    )
    _require(
        set(fixture_commitment)
        == {"sha256", "read_by_executor", "first_read_stage", "read_count"}
        and fixture_commitment["read_by_executor"] is True
        and fixture_commitment["first_read_stage"] == "run_adjudicator"
        and fixture_commitment["read_count"] == 1
        and set(scoring_commitment)
        == {"sha256", "read_by_executor", "scorer_reads_last"}
        and scoring_commitment["read_by_executor"] is False
        and scoring_commitment["scorer_reads_last"] is True,
        "public key access commitment invalid",
    )
    _sha256(fixture_commitment["sha256"], "fixture-generation key commitment hash")
    _sha256(scoring_commitment["sha256"], "control-scoring key commitment hash")
    key_contracts = _mapping(
        executor_config.get("key_contracts"), "executor key contracts"
    )
    _require(
        set(key_contracts) == {"fixture_generation_key", "control_scoring_key"},
        "executor key contract names invalid",
    )
    fixture_key_contract = _mapping(
        key_contracts["fixture_generation_key"],
        "executor fixture-generation key contract",
    )
    scoring_key_contract = _mapping(
        key_contracts["control_scoring_key"],
        "executor control-scoring key contract",
    )
    _require(
        fixture_commitment["sha256"] == fixture_key_contract.get("config_sha256")
        and scoring_commitment["sha256"] == scoring_key_contract.get("config_sha256"),
        "public key commitments differ from frozen executor key contracts",
    )
    fixture_materializer_hash = _sha256(
        fixture_key_contract.get("materializer_sha256"),
        "executor fixture-key materializer hash",
    )
    suite_nonce = _mapping(value["suite_nonce"], "public suite nonce")
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
        "public suite-nonce fields invalid",
    )
    for name in (
        "suite_nonce_sha256",
        "precommit_receipt_sha256",
        "single_use_consumption_receipt_sha256",
    ):
        _sha256(suite_nonce[name], f"public suite nonce {name}")
    trace_events = _sequence(trace["events"], "authenticated trace events")
    fixture_key_trace_events = [
        event
        for event in trace_events
        if event["artifact_class"] == "fixture_generation_key"
        and event["event_type"] == "read"
    ]
    fixture_materializer_trace_events = [
        event
        for event in trace_events
        if event["artifact_class"] == "fixture_key_materializer"
        and event["event_type"] == "read"
    ]
    nonce_consumption_events = [
        event
        for event in trace_events
        if event["artifact_class"] == "nonce_secret"
        and event["event_type"] == "consume"
    ]
    _require(
        len(fixture_key_trace_events) == 1
        and fixture_key_trace_events[0]["expected_sha256"]
        == fixture_commitment["sha256"]
        and fixture_key_trace_events[0]["observed_sha256"]
        == fixture_commitment["sha256"]
        and len(fixture_materializer_trace_events) == 1
        and fixture_materializer_trace_events[0]["expected_sha256"]
        == fixture_materializer_hash
        and fixture_materializer_trace_events[0]["observed_sha256"]
        == fixture_materializer_hash
        and len(nonce_consumption_events) == 1
        and nonce_consumption_events[0]["observed_sha256"]
        == suite_nonce["single_use_consumption_receipt_sha256"],
        "executor trace differs from key/materializer/nonce commitments",
    )
    config_roles = _mapping(
        executor_config.get("role_contracts"), "executor role contracts"
    )
    roles = _mapping(value["roles"], "public role results")
    _require(
        set(roles) == set(_role_ids()) == set(config_roles),
        "public role coverage invalid",
    )
    expected_ids = list(_control_ids())
    all_native_request_hashes: list[str] = []
    all_native_result_hashes: list[str] = []
    packets_by_control: dict[str, set[str]] = {
        control_id: set() for control_id in _control_ids()
    }
    for role_name in _role_ids():
        role = _mapping(roles[role_name], f"public role {role_name}")
        _require(
            set(role)
            == {
                "role",
                "model_execution_count",
                "host_bypass_count",
                "result_count",
                "ordered_results",
                "role_receipt_sha256",
            },
            f"public role {role_name} fields invalid",
        )
        rows = [
            _mapping(item, f"{role_name} result")
            for item in _sequence(
                role["ordered_results"], f"{role_name} ordered results"
            )
        ]
        _require(
            role["role"] == role_name
            and role["model_execution_count"] == 39
            and role["host_bypass_count"] == 1
            and role["result_count"] == 40
            and len(rows) == 40
            and _sha256(role["role_receipt_sha256"], f"{role_name} role receipt hash"),
            f"public role {role_name} counts invalid",
        )
        bypass_ids: list[str] = []
        for ordinal, (control_id, row) in enumerate(
            zip(expected_ids, rows, strict=True), start=1
        ):
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
                and row["control_id"] == control_id
                and row["ordinal"] == ordinal
                and SHA256_RE.fullmatch(str(row["packet_sha256"])) is not None,
                f"{role_name} control order or row shape invalid",
            )
            packets_by_control[control_id].add(str(row["packet_sha256"]))
            annotation_pass = (
                "completion_chain" if control_id.startswith("C") else "prefix_novelty"
            )
            projection = _validate_proposal(
                row["result"],
                annotation_pass=annotation_pass,
                proposal_contract=proposal_contract,
            )
            _require(
                row["result_sha256"] == sha256_value(projection),
                f"{role_name} {control_id} result hash invalid",
            )
            if row["execution_path"] == "deterministic_host_bypass":
                bypass_ids.append(control_id)
                _require(
                    control_id == "C32"
                    and projection == {"decision": "no_chain"}
                    and row["native_evidence"] is None,
                    f"{role_name} has an invalid deterministic bypass",
                )
            else:
                _require(
                    row["execution_path"] == "native_model"
                    and isinstance(row["native_evidence"], Mapping)
                    and bool(row["native_evidence"]),
                    f"{role_name} {control_id} lacks native evidence",
                )
                evidence = _validate_native_evidence_projection(
                    row["native_evidence"],
                    role=role_name,
                    label=f"{role_name} {control_id}",
                )
                all_native_request_hashes.extend(evidence["native_request_sha256s"])
                all_native_result_hashes.extend(evidence["native_result_sha256s"])
        _require(bypass_ids == ["C32"], f"{role_name} sole bypass proof failed")
    _require(
        all(len(values) == 1 for values in packets_by_control.values()),
        "packet identity differs across A/B/J for a control",
    )

    schedule = [
        _mapping(item, "fixture seed row")
        for item in _sequence(value["fixture_seed_schedule"], "fixture seed schedule")
    ]
    expected_schedule = executor_config.get("fixture_seed_schedule")
    _require(
        schedule == expected_schedule
        and [row.get("case_id") for row in schedule] == list(_fixture_ids())
        and value["fixture_seed_schedule_sha256"] == sha256_value(schedule),
        "fixture seed schedule differs from frozen executor config",
    )
    fixture_rows = [
        _mapping(item, "public fixture result")
        for item in _sequence(value["fixture_results"], "public fixture results")
    ]
    _require(
        len(fixture_rows) == 12
        and [row.get("case_id") for row in fixture_rows] == list(_fixture_ids())
        and [row.get("ordinal") for row in fixture_rows] == list(range(1, 13)),
        "public fixture result count or order invalid",
    )
    fixture_passes = {
        **{f"F{index:02d}": "completion_chain" for index in range(1, 7)},
        "F07": "prefix_novelty",
        "F08": "prefix_novelty",
        "F09": "completion_chain",
        "F10": "completion_chain",
        "F11": "completion_chain",
        "F12": "prefix_novelty",
    }
    fixture_input = _mapping(inputs.get("fixture"), "public fixture input")
    fixture_input_hash = _sha256(
        fixture_input.get("sha256"), "public fixture input hash"
    )
    suite_nonce_hash = _sha256(
        suite_nonce.get("suite_nonce_sha256"), "public suite nonce hash"
    )
    schedule_by_id = {str(row["case_id"]): row for row in schedule}
    for ordinal, (case_id, row) in enumerate(
        zip(_fixture_ids(), fixture_rows, strict=True), start=1
    ):
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
            and row["annotation_pass"] == fixture_passes[case_id]
            and row["verdict_seed"] == schedule_by_id[case_id]["verdict_seed"]
            and row["repair_seed"] == schedule_by_id[case_id]["repair_seed"],
            f"public fixture {case_id} shape/pass/order/seed invalid",
        )
        projection = _validate_proposal(
            row["result"],
            annotation_pass=fixture_passes[case_id],
            proposal_contract=proposal_contract,
        )
        _require(
            row["result_sha256"] == sha256_value(projection),
            f"public fixture {case_id} result hash invalid",
        )
        for name in (
            "fixture_lock_sha256",
            "generation_contract_sha256",
            "fixture_nonce_sha256",
        ):
            _sha256(row[name], f"public fixture {case_id} {name}")
        expected_case_nonce = sha256_value(
            {
                "domain": "v3-fixture-corpus-case-nonce-citrine",
                "suite_nonce_sha256": suite_nonce_hash,
                "case_id": case_id,
                "input_manifest_sha256": fixture_input_hash,
            }
        )
        _require(
            row["fixture_nonce_sha256"] == expected_case_nonce,
            f"public fixture {case_id} nonce derivation invalid",
        )
        evidence = _validate_native_evidence_projection(
            row["native_evidence"], role="adjudicator", label=f"fixture {case_id}"
        )
        all_native_request_hashes.extend(evidence["native_request_sha256s"])
        all_native_result_hashes.extend(evidence["native_result_sha256s"])
    _require(
        len(all_native_request_hashes) == len(set(all_native_request_hashes))
        and len(all_native_result_hashes) == len(set(all_native_result_hashes)),
        "native request/result replay detected across finalized bundle",
    )
    chronology = _mapping(value["chronology"], "public bundle chronology")
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
        "public bundle chronology fields invalid",
    )
    stage_rows = {
        "freeze_helper": (chronology["freeze_helper"], "freeze_helper"),
        "precommit_nonce": (chronology["precommit_nonce"], "precommit_nonce"),
        "lock_primaries": (chronology["lock_primaries"], "lock_primaries"),
        "adjudicator": (chronology["adjudicator"], "run_adjudicator"),
        "lock_all": (chronology["lock_all"], "lock_all"),
    }
    for name, (raw_stage, expected_stage) in stage_rows.items():
        stage = _mapping(raw_stage, f"public chronology {name}")
        _require(
            set(stage) == {"stage", "receipt_sha256", "completed_monotonic_ns"}
            and stage["stage"] == expected_stage,
            f"public chronology {name} fields invalid",
        )
        _sha256(stage["receipt_sha256"], f"public chronology {name} receipt")
    primary_stages = _mapping(
        chronology["primary_roles"], "public primary-role chronology"
    )
    _require(
        set(primary_stages) == {"independent_a", "independent_b"},
        "public primary-role chronology coverage invalid",
    )
    for role_name in ("independent_a", "independent_b"):
        stage = _mapping(primary_stages[role_name], f"public chronology {role_name}")
        _require(
            set(stage) == {"stage", "receipt_sha256", "completed_monotonic_ns"}
            and stage["stage"] == "run_primary",
            f"public chronology {role_name} fields invalid",
        )
        _sha256(stage["receipt_sha256"], f"public chronology {role_name} receipt")
    ordered_times = [
        chronology["freeze_helper"]["completed_monotonic_ns"],
        chronology["precommit_nonce"]["completed_monotonic_ns"],
        min(
            chronology["primary_roles"]["independent_a"]["completed_monotonic_ns"],
            chronology["primary_roles"]["independent_b"]["completed_monotonic_ns"],
        ),
        max(
            chronology["primary_roles"]["independent_a"]["completed_monotonic_ns"],
            chronology["primary_roles"]["independent_b"]["completed_monotonic_ns"],
        ),
        chronology["lock_primaries"]["completed_monotonic_ns"],
        chronology["adjudicator"]["completed_monotonic_ns"],
        chronology["lock_all"]["completed_monotonic_ns"],
    ]
    _require(
        all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in ordered_times
        )
        and all(left < right for left, right in zip(ordered_times, ordered_times[1:])),
        "public bundle chronology order invalid",
    )
    locks = _mapping(value["locks"], "public bundle locks")
    _require(
        set(locks)
        == {
            "primary_independent_a_sha256",
            "primary_independent_b_sha256",
            "dual_primary_lock_sha256",
            "adjudicator_sha256",
            "final_lock_sha256",
        }
        and locks["primary_independent_a_sha256"]
        == roles["independent_a"]["role_receipt_sha256"]
        and locks["primary_independent_b_sha256"]
        == roles["independent_b"]["role_receipt_sha256"]
        and locks["adjudicator_sha256"] == roles["adjudicator"]["role_receipt_sha256"]
        and locks["primary_independent_a_sha256"]
        != locks["primary_independent_b_sha256"]
        and locks["dual_primary_lock_sha256"]
        == chronology["lock_primaries"]["receipt_sha256"]
        and locks["final_lock_sha256"] == chronology["lock_all"]["receipt_sha256"],
        "public bundle lock/role/chronology binding invalid",
    )
    for name, lock_hash in locks.items():
        _sha256(lock_hash, f"public bundle {name}")
    _require(
        len(set(locks.values())) == len(locks),
        "public bundle lock hashes are not independently distinct",
    )
    trace_binding = _mapping(value["read_trace"], "public read-trace binding")
    _require(
        set(trace_binding)
        == {"trace_envelope_sha256", "trace_sha256", "head_sha256", "event_count"}
        and trace_binding.get("trace_sha256") == sha256_value(trace)
        and trace_binding.get("head_sha256") == trace["head_sha256"]
        and trace_binding.get("event_count") == trace["event_count"],
        "public bundle read-trace binding invalid",
    )
    for name in ("trace_envelope_sha256", "trace_sha256", "head_sha256"):
        _sha256(trace_binding[name], f"public read-trace {name}")
    return copy.deepcopy(value)


def _validate_main_key(
    envelope: Any,
    *,
    expected_manifest_sha256: str,
    proposal_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _validate_hash_envelope(
        envelope,
        body_field="manifest",
        hash_field="manifest_sha256",
        expected_body_sha256=expected_manifest_sha256,
        label="main control key",
    )
    _require(
        set(manifest)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "status",
            "scope",
            "externally_authenticated_input_manifest_sha256",
            "key_config_sha256",
            "counts",
            "completion_rows",
            "novelty_rows",
            "ordered_key_rows_sha256",
        }
        and manifest["schema_version"] == SCHEMA_VERSION
        and manifest["interface_version"] == INTERFACE_VERSION
        and manifest["kind"] == MAIN_KEY_KIND
        and manifest["status"] == "prospective_draft_not_sealed_not_run",
        "main control key identity or fields invalid",
    )
    _mapping(manifest["scope"], "main control key scope")
    _sha256(
        manifest["externally_authenticated_input_manifest_sha256"],
        "main control key input-manifest hash",
    )
    _sha256(manifest["key_config_sha256"], "main control key config hash")
    completion = [
        _mapping(item, "completion key row")
        for item in _sequence(manifest.get("completion_rows"), "completion key rows")
    ]
    novelty = [
        _mapping(item, "novelty key row")
        for item in _sequence(manifest.get("novelty_rows"), "novelty key rows")
    ]
    _require(
        [item.get("control_id") for item in completion] == list(_control_ids()[:32])
        and [item.get("control_id") for item in novelty] == list(_control_ids()[32:]),
        "main control key order invalid",
    )
    counts = _mapping(manifest.get("counts"), "main key counts")
    _require(
        counts
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
        }
        and manifest.get("ordered_key_rows_sha256")
        == sha256_value(completion + novelty),
        "main control key counts or ordered-row binding invalid",
    )
    rows: dict[str, dict[str, Any]] = {}
    for annotation_pass, items in (
        ("completion_chain", completion),
        ("prefix_novelty", novelty),
    ):
        for row in items:
            control_id = str(row.get("control_id"))
            if annotation_pass == "completion_chain":
                expected_fields = {
                    "schema_version",
                    "interface_version",
                    "kind",
                    "control_id",
                    "pass",
                    "packet_id_sha256",
                    "input_record_sha256",
                    "gold_category",
                    "diagnostic",
                    "gold_proposal",
                    "gold_exact_unit_id_tuple",
                    "gold_materialized_result",
                    "model_invocation_required",
                }
                expected_kind = COMPLETION_KEY_ROW_KIND
                expected_pass = "completion"
            else:
                expected_fields = {
                    "schema_version",
                    "interface_version",
                    "kind",
                    "control_id",
                    "pass",
                    "packet_id_sha256",
                    "input_record_sha256",
                    "gold_category",
                    "diagnostic",
                    "gold_proposal",
                    "gold_materialized_result",
                    "model_invocation_required",
                }
                expected_kind = NOVELTY_KEY_ROW_KIND
                expected_pass = "novelty"
            _require(
                set(row) == expected_fields
                and row["schema_version"] == SCHEMA_VERSION
                and row["interface_version"] == INTERFACE_VERSION
                and row["kind"] == expected_kind
                and row["pass"] == expected_pass
                and control_id not in rows
                and row.get("model_invocation_required") is (control_id != "C32")
                and isinstance(row["diagnostic"], str)
                and bool(row["diagnostic"]),
                f"main key {control_id} row or invocation contract invalid",
            )
            _sha256(row["packet_id_sha256"], f"main key {control_id} packet hash")
            _sha256(row["input_record_sha256"], f"main key {control_id} input hash")
            proposal = _validate_proposal(
                row.get("gold_proposal"),
                annotation_pass=annotation_pass,
                proposal_contract=proposal_contract,
            )
            category = str(row.get("gold_category"))
            _require(
                category == proposal["decision"],
                f"main key {control_id} category/proposal mismatch",
            )
            if annotation_pass == "completion_chain":
                exact_tuple = row["gold_exact_unit_id_tuple"]
                if proposal["decision"] == "chain":
                    _require(
                        exact_tuple
                        == {
                            "evidence_unit_id": proposal["evidence_unit_id"],
                            "hypothesis_unit_id": proposal["hypothesis_unit_id"],
                            "action_unit_id": proposal["action_unit_id"],
                        },
                        f"main key {control_id} exact tuple invalid",
                    )
                else:
                    _require(
                        exact_tuple is None,
                        f"main key {control_id} unexpected exact tuple",
                    )
            _validate_main_key_materialization(
                row["gold_materialized_result"],
                proposal=proposal,
                annotation_pass=annotation_pass,
                label=f"main key {control_id}",
            )
            if control_id == "C32":
                _require(
                    proposal == {"decision": "no_chain"}
                    and row.get("model_invocation_required") is False,
                    "C32 key contract invalid",
                )
            row_copy = copy.deepcopy(row)
            row_copy["_proposal"] = proposal
            rows[control_id] = row_copy
    _require(len(rows) == 40, "main control key coverage invalid")
    return manifest, rows


def _fixture_nonce_precommit_hash(fixture_nonce_sha256: str) -> str:
    return sha256_value(
        {
            "domain": "adjudication-fixture-nonce-precommit-v3",
            "fixture_nonce_sha256": fixture_nonce_sha256,
        }
    )


def _validate_fixture_generation_contract(
    value: Any,
    *,
    expected_sha256: str,
    fixture_nonce_sha256: str,
    outer_nonce_receipt_sha256: str,
    verdict_seed: int,
    repair_seed: int,
    label: str,
) -> dict[str, Any]:
    contract = _mapping(value, f"{label} generation contract")
    _require(
        set(contract)
        == {
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
        and contract["schema_version"] == SCHEMA_VERSION
        and contract["interface_version"] == INTERFACE_VERSION
        and contract["kind"] == FIXTURE_GENERATION_CONTRACT_KIND
        and contract["execution_mode"] == "native_protocol_adapter_non_gate"
        and contract["inner_route_claims_actual_model_execution"] is False
        and contract["actual_model_execution_claim_responsibility"]
        == "future_sealed_outer_executor"
        and sha256_value(contract) == expected_sha256,
        f"{label} generation contract identity or hash invalid",
    )
    for body_name, hash_name in (
        ("model_identity", "model_identity_sha256"),
        ("tokenizer_identity", "tokenizer_identity_sha256"),
        ("chat_template_kwargs", "chat_template_kwargs_sha256"),
        ("runtime_identity", "runtime_identity_sha256"),
        ("native_adapter_identity", "native_adapter_identity_sha256"),
    ):
        body = _mapping(contract[body_name], f"{label} {body_name}")
        _require(
            contract[hash_name] == sha256_value(body),
            f"{label} {body_name} hash invalid",
        )
    seeds = _mapping(contract["seeds"], f"{label} generation seeds")
    _require(
        seeds == {"verdict_seed": verdict_seed, "repair_seed": repair_seed},
        f"{label} generation seeds invalid",
    )
    nonce = _mapping(contract["nonce_provenance"], f"{label} nonce provenance")
    _require(
        set(nonce)
        == {
            "fixture_nonce_sha256",
            "nonce_precommit_sha256",
            "outer_nonce_precommit_receipt_sha256",
            "precommit_chronology_verified_by_fixture_route",
            "expectation_access_chronology_verified_by_fixture_route",
            "chronology_responsibility",
        }
        and nonce["fixture_nonce_sha256"] == fixture_nonce_sha256
        and nonce["nonce_precommit_sha256"]
        == _fixture_nonce_precommit_hash(fixture_nonce_sha256)
        and nonce["outer_nonce_precommit_receipt_sha256"] == outer_nonce_receipt_sha256
        and nonce["precommit_chronology_verified_by_fixture_route"] is False
        and nonce["expectation_access_chronology_verified_by_fixture_route"] is False
        and nonce["chronology_responsibility"] == "future_sealed_outer_executor",
        f"{label} nonce-provenance contract invalid",
    )
    _mapping(contract["claim_scope"], f"{label} generation claim scope")
    return copy.deepcopy(contract)


def _validate_fixture_lock(
    value: Any,
    *,
    annotation_pass: str,
    fixture_nonce_sha256: str,
    outer_nonce_receipt_sha256: str,
    generation_contract_sha256: str,
    verdict_seed: int,
    repair_seed: int,
    label: str,
) -> dict[str, Any]:
    lock = _mapping(value, f"{label} lock")
    _require(
        set(lock)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "scope",
            "annotation_pass",
            "packet_id_sha256",
            "packet_sha256",
            "source_id_sha256",
            "candidate_unit_bundle_sha256",
            "codebook_sha256",
            "codebook_source_sha256",
            "production_runner_sha256",
            "fixture_runner_sha256",
            "fixture_config_sha256",
            "generation_contract_sha256",
            "runtime_identity_sha256",
            "native_adapter_identity_sha256",
            "fixture_nonce_sha256",
            "nonce_precommit_sha256",
            "outer_nonce_precommit_receipt_sha256",
            "generation_contract",
            "inner_execution_mode",
            "inner_route_claims_actual_model_execution",
            "precommit_chronology_verified_by_fixture_route",
            "expectation_access_chronology_verified_by_fixture_route",
            "chronology_responsibility",
            "authored_candidate_fixture_sha256s_in_original_order",
            "authored_projection_sha256s_in_original_order",
            "original_order_binding_sha256",
            "blinded_original_indexes",
            "blinded_candidate_fixture_sha256s",
            "blinded_projection_sha256s",
            "candidate_order_sha256",
            "model_input_bindings",
            "fixture_candidate_provenance",
            "candidate_model_generated",
            "candidate_native_model_record_lineage_present",
            "expectation_object_bound",
            "claim_scope",
        }
        and lock["schema_version"] == SCHEMA_VERSION
        and lock["interface_version"] == INTERFACE_VERSION
        and lock["kind"] == FIXTURE_LOCK_KIND
        and lock["scope"] == "controls_module_only_authored_candidates"
        and lock["annotation_pass"] == annotation_pass
        and lock["generation_contract_sha256"] == generation_contract_sha256
        and lock["fixture_nonce_sha256"] == fixture_nonce_sha256
        and lock["nonce_precommit_sha256"]
        == _fixture_nonce_precommit_hash(fixture_nonce_sha256)
        and lock["outer_nonce_precommit_receipt_sha256"] == outer_nonce_receipt_sha256
        and lock["inner_execution_mode"] == "native_protocol_adapter_non_gate"
        and lock["inner_route_claims_actual_model_execution"] is False
        and lock["precommit_chronology_verified_by_fixture_route"] is False
        and lock["expectation_access_chronology_verified_by_fixture_route"] is False
        and lock["chronology_responsibility"] == "future_sealed_outer_executor"
        and lock["candidate_model_generated"] is False
        and lock["candidate_native_model_record_lineage_present"] is False
        and lock["expectation_object_bound"] is False,
        f"{label} lock identity or chronology scope invalid",
    )
    for name in (
        "packet_id_sha256",
        "packet_sha256",
        "source_id_sha256",
        "codebook_sha256",
        "codebook_source_sha256",
        "production_runner_sha256",
        "fixture_runner_sha256",
        "fixture_config_sha256",
        "generation_contract_sha256",
        "runtime_identity_sha256",
        "native_adapter_identity_sha256",
        "fixture_nonce_sha256",
        "nonce_precommit_sha256",
        "outer_nonce_precommit_receipt_sha256",
        "original_order_binding_sha256",
        "candidate_order_sha256",
    ):
        _sha256(lock[name], f"{label} lock {name}")
    if lock["candidate_unit_bundle_sha256"] is not None:
        _sha256(
            lock["candidate_unit_bundle_sha256"],
            f"{label} candidate-unit bundle hash",
        )
    contract = _validate_fixture_generation_contract(
        lock["generation_contract"],
        expected_sha256=generation_contract_sha256,
        fixture_nonce_sha256=fixture_nonce_sha256,
        outer_nonce_receipt_sha256=outer_nonce_receipt_sha256,
        verdict_seed=verdict_seed,
        repair_seed=repair_seed,
        label=label,
    )
    _require(
        lock["runtime_identity_sha256"] == contract["runtime_identity_sha256"]
        and lock["native_adapter_identity_sha256"]
        == contract["native_adapter_identity_sha256"]
        and lock["claim_scope"] == contract["claim_scope"],
        f"{label} lock differs from generation contract",
    )
    list_names = (
        "authored_candidate_fixture_sha256s_in_original_order",
        "authored_projection_sha256s_in_original_order",
        "blinded_candidate_fixture_sha256s",
        "blinded_projection_sha256s",
    )
    lists: dict[str, list[Any]] = {}
    for name in list_names:
        items = _sequence(lock[name], f"{label} lock {name}")
        _require(len(items) == 2, f"{label} lock {name} count invalid")
        for item in items:
            _sha256(item, f"{label} lock {name} item")
        _require(len(set(items)) == 2, f"{label} lock {name} duplicates")
        lists[name] = items
    blinded_indexes = _sequence(
        lock["blinded_original_indexes"], f"{label} blinded original indexes"
    )
    _require(
        len(blinded_indexes) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in blinded_indexes
        )
        and sorted(blinded_indexes) == [0, 1]
        and sorted(lists["blinded_candidate_fixture_sha256s"])
        == sorted(lists["authored_candidate_fixture_sha256s_in_original_order"])
        and sorted(lists["blinded_projection_sha256s"])
        == sorted(lists["authored_projection_sha256s_in_original_order"]),
        f"{label} lock blinding permutation invalid",
    )
    _mapping(lock["model_input_bindings"], f"{label} model-input bindings")
    provenance = _mapping(
        lock["fixture_candidate_provenance"], f"{label} fixture provenance"
    )
    _require(
        provenance
        == {
            "origin": "authored_bounded_control_fixture_projection",
            "intended_use": "sealed_semantic_adjudication_control_fixture_only",
            "model_generated": False,
            "native_model_record_lineage_present": False,
            "production_candidate_record_kind_claimed": False,
        },
        f"{label} fixture provenance invalid",
    )
    _mapping(lock["claim_scope"], f"{label} lock claim scope")
    return copy.deepcopy(lock)


def _validate_fixture_key(
    envelope: Any,
    *,
    expected_manifest_sha256: str,
    proposal_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _validate_hash_envelope(
        envelope,
        body_field="manifest",
        hash_field="manifest_sha256",
        expected_body_sha256=expected_manifest_sha256,
        label="fixture adjudication key",
    )
    _require(
        set(manifest)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "status",
            "scope",
            "input_manifest_sha256",
            "key_config_sha256",
            "externally_precommitted_suite_nonce_sha256",
            "outer_nonce_precommit_receipt_sha256",
            "outer_nonce_precommit_receipt_contents_authenticated",
            "receipt_contents_and_issuance_authentication_responsibility",
            "single_use_nonce_or_receipt_enforced_by_key_materializer",
            "single_use_nonce_or_receipt_responsibility",
            "precommit_chronology_verified_by_key_materializer",
            "semantic_access_chronology_verified_by_key_materializer",
            "repeated_materialization_prevention_enforced_by_key_materializer",
            "repeated_materialization_prevention_responsibility",
            "authoritative_packet_regeneration_performed_by_key_materializer",
            "authoritative_packet_regeneration_responsibility",
            "actual_model_execution_claimed",
            "generation_callback_invoked",
            "seal_ready",
            "nonce_retry_permitted",
            "presentation_balance",
            "generation_seed_schedule",
            "generation_seed_schedule_sha256",
            "counts",
            "cases",
            "ordered_fixture_lock_sha256s_sha256",
        }
        and manifest["schema_version"] == SCHEMA_VERSION
        and manifest["interface_version"] == INTERFACE_VERSION
        and manifest["kind"] == FIXTURE_KEY_KIND
        and manifest["status"]
        == "prospective_draft_balance_observed_chronology_unverified_not_seal_ready"
        and manifest["outer_nonce_precommit_receipt_contents_authenticated"] is False
        and manifest["receipt_contents_and_issuance_authentication_responsibility"]
        == "future_sealed_outer_executor"
        and manifest["single_use_nonce_or_receipt_enforced_by_key_materializer"]
        is False
        and manifest["single_use_nonce_or_receipt_responsibility"]
        == "future_sealed_outer_executor"
        and manifest["precommit_chronology_verified_by_key_materializer"] is False
        and manifest["semantic_access_chronology_verified_by_key_materializer"] is False
        and manifest["repeated_materialization_prevention_enforced_by_key_materializer"]
        is False
        and manifest["repeated_materialization_prevention_responsibility"]
        == "future_sealed_outer_executor"
        and manifest["authoritative_packet_regeneration_performed_by_key_materializer"]
        is False
        and manifest["authoritative_packet_regeneration_responsibility"]
        == "future_sealed_outer_executor"
        and manifest["actual_model_execution_claimed"] is False
        and manifest["generation_callback_invoked"] is False
        and manifest["seal_ready"] is False
        and manifest["nonce_retry_permitted"] is False,
        "fixture key identity, fields, or responsibility contract invalid",
    )
    _mapping(manifest["scope"], "fixture key scope")
    for name in (
        "input_manifest_sha256",
        "key_config_sha256",
        "externally_precommitted_suite_nonce_sha256",
        "outer_nonce_precommit_receipt_sha256",
        "generation_seed_schedule_sha256",
        "ordered_fixture_lock_sha256s_sha256",
    ):
        _sha256(manifest[name], f"fixture key {name}")
    cases = [
        _mapping(item, "fixture key case")
        for item in _sequence(manifest.get("cases"), "fixture key cases")
    ]
    _require(
        [item.get("case_id") for item in cases] == list(_fixture_ids()),
        "fixture key order invalid",
    )
    counts = _mapping(manifest.get("counts"), "fixture key counts")
    _require(
        counts
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
        "fixture key counts invalid",
    )
    schedule = _mapping(
        manifest["generation_seed_schedule"], "fixture key generation schedule"
    )
    _require(
        set(schedule) == {"kind", "derivation", "cases"}
        and schedule["kind"] == "exact_per_case_verdict_and_repair_seed_schedule_v3"
        and schedule["derivation"] == "frozen_explicit_values_not_caller_authoritative"
        and manifest["generation_seed_schedule_sha256"] == sha256_value(schedule),
        "fixture key generation schedule invalid",
    )
    schedule_rows = [
        _mapping(item, "fixture key seed row")
        for item in _sequence(schedule["cases"], "fixture key seed rows")
    ]
    _require(
        [row.get("case_id") for row in schedule_rows] == list(_fixture_ids())
        and all(
            set(row) == {"case_id", "verdict_seed", "repair_seed"}
            and isinstance(row["verdict_seed"], int)
            and not isinstance(row["verdict_seed"], bool)
            and isinstance(row["repair_seed"], int)
            and not isinstance(row["repair_seed"], bool)
            for row in schedule_rows
        ),
        "fixture key seed schedule rows invalid",
    )
    schedule_by_id = {str(row["case_id"]): row for row in schedule_rows}
    rows: dict[str, dict[str, Any]] = {}
    for row in cases:
        case_id = str(row["case_id"])
        annotation_pass = str(row.get("annotation_pass"))
        _require(
            set(row)
            == {
                "case_id",
                "annotation_pass",
                "route_requirement",
                "semantic_projection",
                "semantic_projection_sha256",
                "repair_decision",
                "completion_chain_detail_repair_required",
                "diagnostic",
                "realized_presentation_class",
                "fixture_nonce_sha256",
                "verdict_seed",
                "repair_seed",
                "generation_seed_schedule_sha256",
                "generation_contract_sha256",
                "fixture_lock",
                "fixture_lock_sha256",
            }
            and isinstance(row["diagnostic"], str)
            and bool(row["diagnostic"])
            and row["generation_seed_schedule_sha256"]
            == manifest["generation_seed_schedule_sha256"]
            and row["verdict_seed"] == schedule_by_id[case_id]["verdict_seed"]
            and row["repair_seed"] == schedule_by_id[case_id]["repair_seed"],
            f"fixture key {case_id} row shape or frozen seed invalid",
        )
        for name in (
            "semantic_projection_sha256",
            "fixture_nonce_sha256",
            "generation_seed_schedule_sha256",
            "generation_contract_sha256",
            "fixture_lock_sha256",
        ):
            _sha256(row[name], f"fixture key {case_id} {name}")
        projection = _validate_proposal(
            row.get("semantic_projection"),
            annotation_pass=annotation_pass,
            proposal_contract=proposal_contract,
        )
        projection_hash = sha256_value(projection)
        _require(
            row.get("semantic_projection_sha256") == projection_hash,
            f"fixture key {case_id} semantic projection hash invalid",
        )
        lock = _validate_fixture_lock(
            row["fixture_lock"],
            annotation_pass=annotation_pass,
            fixture_nonce_sha256=row["fixture_nonce_sha256"],
            outer_nonce_receipt_sha256=manifest["outer_nonce_precommit_receipt_sha256"],
            generation_contract_sha256=row["generation_contract_sha256"],
            verdict_seed=row["verdict_seed"],
            repair_seed=row["repair_seed"],
            label=f"fixture key {case_id}",
        )
        lock_hash = sha256_value(lock)
        _require(
            row.get("fixture_lock_sha256") == lock_hash,
            f"fixture key {case_id} lock hash invalid",
        )
        route = row.get("route_requirement")
        realized = row.get("realized_presentation_class")
        blinded = _sequence(
            lock.get("blinded_projection_sha256s"),
            f"fixture key {case_id} blinded projections",
        )
        _require(
            len(blinded) == 2, f"fixture key {case_id} blinded projection count invalid"
        )
        if route == "direct":
            _require(
                realized in {"candidate_1", "candidate_2"}
                and blinded[0 if realized == "candidate_1" else 1] == projection_hash
                and blinded.count(projection_hash) == 1,
                f"fixture key {case_id} direct presentation invalid",
            )
        else:
            _require(
                route == "neither"
                and realized == "neither"
                and projection_hash not in blinded,
                f"fixture key {case_id} neither presentation invalid",
            )
        expected_detail_repair = (
            route == "neither"
            and annotation_pass == "completion_chain"
            and projection["decision"] == "chain"
        )
        _require(
            row["repair_decision"]
            == (None if route == "direct" else projection["decision"])
            and row["completion_chain_detail_repair_required"]
            is expected_detail_repair,
            f"fixture key {case_id} repair contract invalid",
        )
        for seed_name in ("verdict_seed", "repair_seed"):
            _require(
                isinstance(row.get(seed_name), int)
                and not isinstance(row.get(seed_name), bool),
                f"fixture key {case_id} {seed_name} invalid",
            )
        row_copy = copy.deepcopy(row)
        row_copy["_proposal"] = projection
        rows[case_id] = row_copy
    _require(len(rows) == 12, "fixture key coverage invalid")
    route_counts = {
        name: sum(row["route_requirement"] == name for row in rows.values())
        for name in ("direct", "neither")
    }
    realized_counts = {
        name: sum(row["realized_presentation_class"] == name for row in rows.values())
        for name in ("candidate_1", "candidate_2", "neither")
    }
    balance = _mapping(manifest["presentation_balance"], "fixture presentation balance")
    _require(
        route_counts == {"direct": 8, "neither": 4}
        and realized_counts == {"candidate_1": 4, "candidate_2": 4, "neither": 4}
        and balance
        == {
            "intended_counts": {
                "candidate_1": 4,
                "candidate_2": 4,
                "neither": 4,
            },
            "observed_counts": realized_counts,
            "balance_satisfied": True,
            "nonce_retry_permitted": False,
            "imbalance_disposition": (
                "invalidate_entire_suite_and_start_new_predeclared_round"
            ),
        },
        "fixture presentation balance differs from the exact realized corpus",
    )
    return manifest, rows


def _score_roles(
    *,
    public_manifest: Mapping[str, Any],
    main_rows: Mapping[str, Mapping[str, Any]],
    proposal_contract: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for role_name in _role_ids():
        role = _mapping(public_manifest["roles"][role_name], f"role {role_name}")
        rows = _sequence(role["ordered_results"], f"role {role_name} results")
        exact = 0
        completion_decision = 0
        chain_tuple = 0
        chain_ontology = 0
        novelty = 0
        chain_total = 0
        for control_id, raw_row in zip(_control_ids(), rows, strict=True):
            row = _mapping(raw_row, f"role {role_name} {control_id}")
            annotation_pass = (
                "completion_chain" if control_id.startswith("C") else "prefix_novelty"
            )
            observed = _validate_proposal(
                row["result"],
                annotation_pass=annotation_pass,
                proposal_contract=proposal_contract,
            )
            expected = dict(main_rows[control_id]["_proposal"])
            exact += int(observed == expected)
            if annotation_pass == "prefix_novelty":
                novelty += int(observed.get("decision") == expected.get("decision"))
            else:
                completion_decision += int(
                    observed.get("decision") == expected.get("decision")
                )
                if expected["decision"] == "chain":
                    chain_total += 1
                    chain_tuple += int(
                        all(
                            observed.get(name) == expected[name]
                            for name in (
                                "evidence_unit_id",
                                "hypothesis_unit_id",
                                "action_unit_id",
                            )
                        )
                    )
                    chain_ontology += int(
                        all(
                            observed.get(name) == expected[name]
                            for name in (
                                "evidence_kind",
                                "belief_edge",
                                "hypothesis_domain",
                                "action_intent",
                            )
                        )
                    )
        output[role_name] = {
            "exact_projection": {"correct": exact, "total": 40},
            "completion_decision": {"correct": completion_decision, "total": 32},
            "completion_chain_exact_unit_tuple": {
                "correct": chain_tuple,
                "total": chain_total,
            },
            "completion_chain_ontology": {
                "correct": chain_ontology,
                "total": chain_total,
            },
            "novelty_decision": {"correct": novelty, "total": 8},
            "c32_deterministic_host_bypass_correct": (
                rows[31]["result"] == {"decision": "no_chain"}
                and rows[31]["execution_path"] == "deterministic_host_bypass"
                and rows[31]["native_evidence"] is None
            ),
        }
    return output


def _fixture_public_row_fields(
    row: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    case_id = str(row.get("case_id"))
    annotation_pass = str(row.get("annotation_pass"))
    result = _mapping(row.get("result"), f"public fixture {case_id} result")
    return case_id, annotation_pass, result


def _score_fixture(
    *,
    public_manifest: Mapping[str, Any],
    fixture_manifest: Mapping[str, Any],
    fixture_rows: Mapping[str, Mapping[str, Any]],
    proposal_contract: Mapping[str, Any],
) -> dict[str, Any]:
    public_rows = [
        _mapping(item, "public fixture result")
        for item in _sequence(
            public_manifest["fixture_results"], "public fixture results"
        )
    ]
    _require(
        [row.get("case_id") for row in public_rows] == list(_fixture_ids()),
        "public fixture order invalid",
    )
    exact = 0
    direct_correct = 0
    neither_repair_correct = 0
    for public_row, case_id in zip(public_rows, _fixture_ids(), strict=True):
        observed_case_id, annotation_pass, raw_result = _fixture_public_row_fields(
            public_row
        )
        _require(observed_case_id == case_id, "public fixture case identity invalid")
        observed = _validate_proposal(
            raw_result,
            annotation_pass=annotation_pass,
            proposal_contract=proposal_contract,
        )
        _require(
            public_row.get("result_sha256") == sha256_value(observed),
            f"public fixture {case_id} result hash invalid",
        )
        key_row = fixture_rows[case_id]
        expected = dict(key_row["_proposal"])
        exact += int(observed == expected)
        _require(
            public_row.get("fixture_lock_sha256") == key_row["fixture_lock_sha256"]
            and public_row.get("generation_contract_sha256")
            == key_row["generation_contract_sha256"]
            and public_row.get("fixture_nonce_sha256")
            == key_row["fixture_nonce_sha256"]
            and public_row.get("verdict_seed") == key_row["verdict_seed"]
            and public_row.get("repair_seed") == key_row["repair_seed"],
            f"public fixture {case_id} lock, nonce, contract, or seed differs from key",
        )
        route = key_row["route_requirement"]
        if route == "direct":
            direct_correct += int(observed == expected)
        else:
            neither_repair_correct += int(observed == expected)
    balance = _mapping(
        fixture_manifest.get("presentation_balance"), "fixture presentation balance"
    )
    _require(
        balance.get("balance_satisfied") is True
        and balance.get("nonce_retry_permitted") is False
        and fixture_manifest.get("nonce_retry_permitted") is False,
        "fixture nonce balance/retry contract invalid",
    )
    return {
        "exact_final_projection": {"correct": exact, "total": 12},
        "direct_final_projection": {"correct": direct_correct, "total": 8},
        "neither_repair_final_projection": {
            "correct": neither_repair_correct,
            "total": 4,
        },
        "exact_result_count": 12,
    }


def _cross_bind_keys(
    *,
    public_manifest: Mapping[str, Any],
    main_manifest: Mapping[str, Any],
    fixture_manifest: Mapping[str, Any],
) -> None:
    commitments = _mapping(public_manifest["key_commitments"], "public key commitments")
    scoring_commitment = _mapping(
        commitments.get("control_scoring_key"), "control scoring key commitment"
    )
    fixture_commitment = _mapping(
        commitments.get("fixture_generation_key"), "fixture generation key commitment"
    )
    _require(
        scoring_commitment.get("sha256") == main_manifest.get("key_config_sha256")
        and fixture_commitment.get("sha256")
        == fixture_manifest.get("key_config_sha256"),
        "materialized keys differ from pre-generation key commitments",
    )
    public_inputs = _mapping(public_manifest["inputs"], "public input bindings")
    control_input = _mapping(public_inputs.get("control"), "public control input")
    fixture_input = _mapping(public_inputs.get("fixture"), "public fixture input")
    _require(
        main_manifest.get("externally_authenticated_input_manifest_sha256")
        == control_input.get("sha256")
        and fixture_manifest.get("input_manifest_sha256")
        == fixture_input.get("sha256"),
        "materialized keys differ from authenticated public input manifests",
    )
    nonce = _mapping(public_manifest["suite_nonce"], "public suite nonce")
    _require(
        fixture_manifest.get("externally_precommitted_suite_nonce_sha256")
        == nonce.get("suite_nonce_sha256")
        and fixture_manifest.get("outer_nonce_precommit_receipt_sha256")
        == nonce.get("precommit_receipt_sha256"),
        "fixture key differs from sole precommitted suite nonce",
    )
    _require(
        fixture_manifest.get("generation_seed_schedule")
        == {
            "kind": "exact_per_case_verdict_and_repair_seed_schedule_v3",
            "derivation": "frozen_explicit_values_not_caller_authoritative",
            "cases": public_manifest["fixture_seed_schedule"],
        },
        "fixture key seed schedule differs from public frozen schedule",
    )
    public_lock_hashes = [
        str(row["fixture_lock_sha256"])
        for row in _sequence(
            public_manifest["fixture_results"], "public fixture results"
        )
    ]
    fixture_cases = _sequence(fixture_manifest.get("cases"), "fixture key cases")
    key_lock_hashes = [str(row["fixture_lock_sha256"]) for row in fixture_cases]
    _require(
        public_lock_hashes == key_lock_hashes
        and fixture_manifest.get("ordered_fixture_lock_sha256s_sha256")
        == sha256_value(key_lock_hashes),
        "public fixture locks differ from ordered semantic-key locks",
    )


def _receipt_body(
    *,
    config_hash: str,
    scorer_source_hash: str,
    executor_config_hash: str,
    executor_source_hash: str,
    trace_file_hash: str,
    trace_envelope_hash: str,
    trace_body_hash: str,
    bundle_file_hash: str,
    bundle_envelope_hash: str,
    bundle_manifest_hash: str,
    main_key_file_hash: str,
    main_key_manifest_hash: str,
    fixture_key_file_hash: str,
    fixture_key_manifest_hash: str,
    public_manifest: Mapping[str, Any],
    role_scores: Mapping[str, Any],
    fixture_score: Mapping[str, Any],
    read_chronology: Mapping[str, Any],
) -> dict[str, Any]:
    claims = _false_claims()
    _require(all(item is False for item in claims.values()), "receipt claims not false")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": RECEIPT_KIND,
        "status": "development_controls_scored_not_sealed_not_gate_evidence",
        "suite_id": public_manifest["suite_id"],
        "authenticated_inputs": {
            "scorer_config_sha256": config_hash,
            "scorer_source_sha256": scorer_source_hash,
            "executor_config_sha256": executor_config_hash,
            "executor_source_sha256": executor_source_hash,
            "executor_read_trace_file_sha256": trace_file_hash,
            "executor_read_trace_envelope_sha256": trace_envelope_hash,
            "executor_read_trace_body_sha256": trace_body_hash,
            "public_bundle_file_sha256": bundle_file_hash,
            "public_bundle_envelope_sha256": bundle_envelope_hash,
            "public_bundle_manifest_sha256": bundle_manifest_hash,
            "main_control_key_file_sha256": main_key_file_hash,
            "main_control_key_manifest_sha256": main_key_manifest_hash,
            "fixture_key_file_sha256": fixture_key_file_hash,
            "fixture_key_manifest_sha256": fixture_key_manifest_hash,
        },
        "suite_invariants": {
            "roles": list(_role_ids()),
            "results_per_role": 40,
            "native_results_per_role": 39,
            "sole_model_bypass": "C32",
            "fixture_result_count": 12,
            "one_suite_nonce": True,
            "nonce_retry_observed": False,
            "caller_authoritative_labels_in_public_bundle": False,
        },
        "read_chronology": copy.deepcopy(dict(read_chronology)),
        "role_scores": copy.deepcopy(dict(role_scores)),
        "fixture_score": copy.deepcopy(dict(fixture_score)),
        "claims": claims,
    }


def _exclusive_write_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    _require(
        not target.exists() and not target.is_symlink(),
        "score receipt path already exists",
    )
    raw = canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except OSError as error:
        raise SealedControlScorerError(
            f"cannot exclusively create score receipt: {error}"
        ) from error
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "score receipt write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def score_finalized_executor_bundle(
    *,
    config_path: Path,
    expected_config_sha256: str,
    expected_scorer_source_sha256: str,
    expected_executor_config_sha256: str,
    expected_executor_source_sha256: str,
    read_trace_path: Path,
    expected_read_trace_file_sha256: str,
    expected_read_trace_envelope_sha256: str,
    public_bundle_path: Path,
    expected_public_bundle_file_sha256: str,
    expected_public_bundle_envelope_sha256: str,
    main_control_key_path: Path,
    expected_main_control_key_file_sha256: str,
    expected_main_control_key_manifest_sha256: str,
    fixture_adjudication_key_path: Path,
    expected_fixture_adjudication_key_file_sha256: str,
    expected_fixture_adjudication_key_manifest_sha256: str,
    receipt_path: Path,
) -> dict[str, Any]:
    """Authenticate public evidence, read the two keys last, score, emit once."""

    # Validate every externally transported hash before touching any path.
    for label, value in (
        ("scorer config", expected_config_sha256),
        ("scorer source", expected_scorer_source_sha256),
        ("executor config", expected_executor_config_sha256),
        ("executor source", expected_executor_source_sha256),
        ("read trace file", expected_read_trace_file_sha256),
        ("read trace envelope", expected_read_trace_envelope_sha256),
        ("public bundle file", expected_public_bundle_file_sha256),
        ("public bundle envelope", expected_public_bundle_envelope_sha256),
        ("main key file", expected_main_control_key_file_sha256),
        ("main key manifest", expected_main_control_key_manifest_sha256),
        ("fixture key file", expected_fixture_adjudication_key_file_sha256),
        ("fixture key manifest", expected_fixture_adjudication_key_manifest_sha256),
    ):
        _sha256(value, f"expected {label} hash")
    output = Path(receipt_path)
    _require(
        not output.exists() and not output.is_symlink(),
        "score receipt path already exists",
    )

    ledger = _ReadLedger()
    (
        config,
        _scorer_source_raw,
        _resolved_config,
        executor_config,
        executor_source_raw,
        executor_source_path,
    ) = _authenticate_config_and_code(
        config_path=config_path,
        expected_config_sha256=expected_config_sha256,
        expected_scorer_source_sha256=expected_scorer_source_sha256,
        expected_executor_config_sha256=expected_executor_config_sha256,
        expected_executor_source_sha256=expected_executor_source_sha256,
        ledger=ledger,
    )
    executor_module = _load_authenticated_executor_module(
        source=executor_source_raw, source_path=executor_source_path
    )

    trace_raw, resolved_trace = _read_authenticated_bytes(
        path=read_trace_path,
        expected_sha256=expected_read_trace_file_sha256,
        label="executor_read_trace",
        ledger=ledger,
    )
    trace_envelope = _strict_json_bytes(trace_raw, "executor read trace")
    _require(
        sha256_value(trace_envelope) == expected_read_trace_envelope_sha256,
        "executor read trace differs from external canonical-envelope hash",
    )
    trace = _validate_hash_envelope(
        trace_envelope,
        body_field="trace",
        hash_field="trace_sha256",
        expected_body_sha256=str(trace_envelope.get("trace_sha256")),
        label="executor read trace",
    )
    try:
        executor_module.validate_read_trace_envelope(
            trace_envelope,
            independently_supplied_sha256=expected_read_trace_envelope_sha256,
        )
    except Exception as error:
        raise SealedControlScorerError(
            f"executor rejected read trace: {error}"
        ) from error
    trace = _validate_public_trace_independently(trace)

    bundle_raw, resolved_bundle = _read_authenticated_bytes(
        path=public_bundle_path,
        expected_sha256=expected_public_bundle_file_sha256,
        label="public_executor_bundle",
        ledger=ledger,
    )
    bundle_envelope = _strict_json_bytes(bundle_raw, "public executor bundle")
    _require(
        sha256_value(bundle_envelope) == expected_public_bundle_envelope_sha256,
        "public executor bundle differs from external canonical-envelope hash",
    )
    public_manifest = _validate_hash_envelope(
        bundle_envelope,
        body_field="manifest",
        hash_field="manifest_sha256",
        expected_body_sha256=str(bundle_envelope.get("manifest_sha256")),
        label="public executor bundle",
    )
    try:
        executor_module.validate_public_generation_bundle(
            bundle_envelope,
            independently_supplied_sha256=expected_public_bundle_envelope_sha256,
            authenticated_trace=trace,
        )
    except Exception as error:
        raise SealedControlScorerError(
            f"executor rejected public bundle: {error}"
        ) from error
    public_manifest = _validate_public_bundle_independently(
        public_manifest,
        trace=trace,
        executor_config=executor_config,
        proposal_contract=config["proposal_contract"],
    )
    _require(
        public_manifest["read_trace"].get("trace_envelope_sha256")
        == expected_read_trace_envelope_sha256,
        "public bundle trace-envelope binding differs from external hash",
    )
    _require(
        resolved_trace != resolved_bundle,
        "executor trace and public bundle must be distinct inputs",
    )
    ledger.mark_public_verified()

    # No semantic-key path is dereferenced before every public gate above.
    main_candidate = Path(main_control_key_path)
    fixture_candidate = Path(fixture_adjudication_key_path)
    _require(
        not main_candidate.is_symlink() and not fixture_candidate.is_symlink(),
        "semantic key paths must not be symlinks",
    )
    try:
        main_resolved = main_candidate.resolve(strict=True)
        fixture_resolved = fixture_candidate.resolve(strict=True)
        main_stat = main_resolved.stat()
        fixture_stat = fixture_resolved.stat()
    except OSError as error:
        raise SealedControlScorerError(
            f"cannot authenticate semantic key paths: {error}"
        ) from error
    _require(
        main_resolved != fixture_resolved
        and (main_stat.st_dev, main_stat.st_ino)
        != (fixture_stat.st_dev, fixture_stat.st_ino),
        "main and fixture semantic keys are not physically distinct files",
    )
    _require(
        output.resolve(strict=False)
        not in {main_resolved, fixture_resolved, resolved_trace, resolved_bundle},
        "score receipt path aliases an input",
    )

    main_key_raw, _ = _read_authenticated_bytes(
        path=main_resolved,
        expected_sha256=expected_main_control_key_file_sha256,
        label="main_control_key",
        ledger=ledger,
        key_read=True,
    )
    main_key_envelope = _strict_json_bytes(main_key_raw, "main control key")
    main_manifest, main_rows = _validate_main_key(
        main_key_envelope,
        expected_manifest_sha256=expected_main_control_key_manifest_sha256,
        proposal_contract=config["proposal_contract"],
    )

    fixture_key_raw, _ = _read_authenticated_bytes(
        path=fixture_resolved,
        expected_sha256=expected_fixture_adjudication_key_file_sha256,
        label="fixture_adjudication_key",
        ledger=ledger,
        key_read=True,
    )
    fixture_key_envelope = _strict_json_bytes(
        fixture_key_raw, "fixture adjudication key"
    )
    fixture_manifest, fixture_rows = _validate_fixture_key(
        fixture_key_envelope,
        expected_manifest_sha256=expected_fixture_adjudication_key_manifest_sha256,
        proposal_contract=config["proposal_contract"],
    )
    _cross_bind_keys(
        public_manifest=public_manifest,
        main_manifest=main_manifest,
        fixture_manifest=fixture_manifest,
    )
    role_scores = _score_roles(
        public_manifest=public_manifest,
        main_rows=main_rows,
        proposal_contract=config["proposal_contract"],
    )
    fixture_score = _score_fixture(
        public_manifest=public_manifest,
        fixture_manifest=fixture_manifest,
        fixture_rows=fixture_rows,
        proposal_contract=config["proposal_contract"],
    )
    chronology = ledger.final_projection()
    body = _receipt_body(
        config_hash=expected_config_sha256,
        scorer_source_hash=expected_scorer_source_sha256,
        executor_config_hash=expected_executor_config_sha256,
        executor_source_hash=expected_executor_source_sha256,
        trace_file_hash=expected_read_trace_file_sha256,
        trace_envelope_hash=expected_read_trace_envelope_sha256,
        trace_body_hash=str(trace_envelope["trace_sha256"]),
        bundle_file_hash=expected_public_bundle_file_sha256,
        bundle_envelope_hash=expected_public_bundle_envelope_sha256,
        bundle_manifest_hash=str(bundle_envelope["manifest_sha256"]),
        main_key_file_hash=expected_main_control_key_file_sha256,
        main_key_manifest_hash=expected_main_control_key_manifest_sha256,
        fixture_key_file_hash=expected_fixture_adjudication_key_file_sha256,
        fixture_key_manifest_hash=expected_fixture_adjudication_key_manifest_sha256,
        public_manifest=public_manifest,
        role_scores=role_scores,
        fixture_score=fixture_score,
        read_chronology=chronology,
    )
    envelope = {"receipt": body, "receipt_sha256": sha256_value(body)}
    validate_score_receipt_envelope(
        envelope,
        independently_supplied_receipt_sha256=envelope["receipt_sha256"],
    )
    _exclusive_write_json(output, envelope)
    return copy.deepcopy(envelope)


def validate_score_receipt_envelope(
    value: Any, *, independently_supplied_receipt_sha256: str
) -> dict[str, Any]:
    receipt = _validate_hash_envelope(
        value,
        body_field="receipt",
        hash_field="receipt_sha256",
        expected_body_sha256=independently_supplied_receipt_sha256,
        label="score receipt",
    )
    _require(
        set(receipt)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "status",
            "suite_id",
            "authenticated_inputs",
            "suite_invariants",
            "read_chronology",
            "role_scores",
            "fixture_score",
            "claims",
        }
        and receipt["schema_version"] == SCHEMA_VERSION
        and receipt["interface_version"] == INTERFACE_VERSION
        and receipt["kind"] == RECEIPT_KIND
        and receipt["status"]
        == "development_controls_scored_not_sealed_not_gate_evidence",
        "score receipt identity invalid",
    )
    _require(
        isinstance(receipt["suite_id"], str) and len(receipt["suite_id"]) >= 16,
        "score receipt suite identity invalid",
    )
    authenticated_inputs = _mapping(
        receipt["authenticated_inputs"], "receipt authenticated inputs"
    )
    _require(
        set(authenticated_inputs)
        == {
            "scorer_config_sha256",
            "scorer_source_sha256",
            "executor_config_sha256",
            "executor_source_sha256",
            "executor_read_trace_file_sha256",
            "executor_read_trace_envelope_sha256",
            "executor_read_trace_body_sha256",
            "public_bundle_file_sha256",
            "public_bundle_envelope_sha256",
            "public_bundle_manifest_sha256",
            "main_control_key_file_sha256",
            "main_control_key_manifest_sha256",
            "fixture_key_file_sha256",
            "fixture_key_manifest_sha256",
        },
        "score receipt authenticated-input fields invalid",
    )
    for name, item in authenticated_inputs.items():
        _sha256(item, f"receipt authenticated input {name}")
    _require(
        receipt["claims"] == _false_claims()
        and all(item is False for item in receipt["claims"].values()),
        "score receipt contains an execution, gate, or recovery claim",
    )
    chronology = _mapping(receipt["read_chronology"], "receipt read chronology")
    _require(
        set(chronology)
        == {
            "input_read_count",
            "ordered_input_read_sha256",
            "public_evidence_verified_before_key_reads",
            "key_read_order",
            "keys_are_final_two_input_reads",
        }
        and chronology.get("input_read_count") == 8
        and SHA256_RE.fullmatch(str(chronology.get("ordered_input_read_sha256")))
        is not None
        and chronology.get("public_evidence_verified_before_key_reads") is True
        and chronology.get("key_read_order")
        == ["main_control_key", "fixture_adjudication_key"]
        and chronology.get("keys_are_final_two_input_reads") is True,
        "score receipt key-read chronology invalid",
    )
    invariants = _mapping(receipt["suite_invariants"], "receipt suite invariants")
    _require(
        invariants
        == {
            "roles": list(_role_ids()),
            "results_per_role": 40,
            "native_results_per_role": 39,
            "sole_model_bypass": "C32",
            "fixture_result_count": 12,
            "one_suite_nonce": True,
            "nonce_retry_observed": False,
            "caller_authoritative_labels_in_public_bundle": False,
        },
        "score receipt suite invariants invalid",
    )
    role_scores = _mapping(receipt["role_scores"], "receipt role scores")
    _require(
        set(role_scores) == set(_role_ids()),
        "score receipt role-score coverage invalid",
    )
    expected_metric_totals = {
        "exact_projection": 40,
        "completion_decision": 32,
        "completion_chain_exact_unit_tuple": 18,
        "completion_chain_ontology": 18,
        "novelty_decision": 8,
    }
    for role_name in _role_ids():
        role = _mapping(role_scores[role_name], f"receipt role score {role_name}")
        _require(
            set(role)
            == set(expected_metric_totals) | {"c32_deterministic_host_bypass_correct"}
            and role["c32_deterministic_host_bypass_correct"] is True,
            f"receipt role score {role_name} fields invalid",
        )
        for metric_name, total in expected_metric_totals.items():
            metric = _mapping(role[metric_name], f"receipt {role_name} {metric_name}")
            _require(
                set(metric) == {"correct", "total"}
                and metric["total"] == total
                and isinstance(metric["correct"], int)
                and not isinstance(metric["correct"], bool)
                and 0 <= metric["correct"] <= total,
                f"receipt {role_name} {metric_name} count invalid",
            )
    fixture_score = _mapping(receipt["fixture_score"], "receipt fixture score")
    _require(
        set(fixture_score)
        == {
            "exact_final_projection",
            "direct_final_projection",
            "neither_repair_final_projection",
            "exact_result_count",
        }
        and fixture_score["exact_result_count"] == 12,
        "receipt fixture score fields invalid",
    )
    for metric_name, total in (
        ("exact_final_projection", 12),
        ("direct_final_projection", 8),
        ("neither_repair_final_projection", 4),
    ):
        metric = _mapping(fixture_score[metric_name], f"receipt fixture {metric_name}")
        _require(
            set(metric) == {"correct", "total"}
            and metric["total"] == total
            and isinstance(metric["correct"], int)
            and not isinstance(metric["correct"], bool)
            and 0 <= metric["correct"] <= total,
            f"receipt fixture {metric_name} count invalid",
        )
    return copy.deepcopy(receipt)


__all__ = [
    "CONFIG_KIND",
    "FIXTURE_KEY_KIND",
    "MAIN_KEY_KIND",
    "RECEIPT_KIND",
    "SealedControlScorerError",
    "canonical_json_bytes",
    "score_finalized_executor_bundle",
    "sha256_bytes",
    "sha256_value",
    "validate_score_receipt_envelope",
]
