from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_sealed_control_scorer_v3.py"
)
CONFIG = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_sealed_control_scorer_v3.json"
)
EXECUTOR_SOURCE = (
    ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_sealed_control_executor_v3.py"
)
EXECUTOR_CONFIG = (
    ROOT
    / "configs"
    / "swe_task_state_v4_epistemic_chain_sealed_control_executor_v3.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scorer = load_module("sealed_control_scorer_v3_test", SOURCE)


def h(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def raw_json(value) -> bytes:
    return scorer.canonical_json_bytes(value) + b"\n"


def completion_proposal(control_id: str) -> dict:
    number = int(control_id[1:])
    if number <= 18:
        return {
            "decision": "chain",
            "evidence_unit_id": f"{control_id}-evidence",
            "hypothesis_unit_id": f"{control_id}-hypothesis",
            "action_unit_id": f"{control_id}-action",
            "evidence_kind": "tool_or_test",
            "belief_edge": "supports",
            "hypothesis_domain": "source_logic",
            "action_intent": "inspect",
        }
    if control_id == "C31":
        return {
            "decision": "unknown",
            "unknown_reason": "completion_semantics_ambiguous",
        }
    return {"decision": "no_chain"}


def novelty_proposal(control_id: str) -> dict:
    number = int(control_id[1:])
    if number <= 2:
        decision = "novel"
    elif number <= 6:
        decision = "prefix_exposed"
    else:
        decision = "ambiguous"
    return {"decision": decision}


def control_proposal(control_id: str) -> dict:
    return (
        completion_proposal(control_id)
        if control_id.startswith("C")
        else novelty_proposal(control_id)
    )


def fixture_pass(case_id: str) -> str:
    return "prefix_novelty" if case_id in {"F07", "F08", "F12"} else "completion_chain"


def fixture_proposal(case_id: str) -> dict:
    if fixture_pass(case_id) == "prefix_novelty":
        return {"decision": "ambiguous" if case_id == "F12" else "novel"}
    if case_id in {"F09", "F10", "F11"}:
        return {
            "decision": "chain",
            "evidence_unit_id": f"{case_id}-evidence",
            "hypothesis_unit_id": f"{case_id}-hypothesis",
            "action_unit_id": f"{case_id}-action",
            "evidence_kind": "tool_or_test",
            "belief_edge": "supports",
            "hypothesis_domain": "source_logic",
            "action_intent": "inspect",
        }
    return {"decision": "no_chain"}


def native_evidence(role: str, row_id: str) -> dict:
    return {
        "adapter_role": role,
        "adapter_config_sha256": h("adapter-config"),
        "adapter_source_sha256": h("adapter-source"),
        "outer_launch_authorization_sha256s": [h(f"{role}-{row_id}-outer-launch")],
        "launch_binding_sha256s": [h(f"{role}-{row_id}-launch")],
        "preflight_receipt_sha256s": [h(f"{role}-{row_id}-preflight")],
        "runtime_receipt_sha256s": [h(f"{role}-{row_id}-runtime")],
        "native_request_sha256s": [h(f"{role}-{row_id}-request")],
        "native_result_sha256s": [h(f"{role}-{row_id}-result")],
        "actual_model_execution": True,
        "model_loaded": True,
        "generation_performed": True,
        "gate_eligible": True,
    }


def trace_event(
    *,
    ordinal: int,
    stage: str,
    rank: int,
    artifact_id: str,
    previous: str,
    event_type: str = "transition",
    artifact_class: str = "stage",
    path: str | None = None,
    expected_sha256: str | None = None,
    observed_sha256: str | None = None,
) -> dict:
    event = {
        "ordinal": ordinal,
        "stage": stage,
        "stage_rank": rank,
        "event_type": event_type,
        "artifact_class": artifact_class,
        "artifact_id": artifact_id,
        "path": path,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "monotonic_ns": 1000 + ordinal * 10,
        "previous_event_sha256": previous,
    }
    event["event_sha256"] = scorer.sha256_value(event)
    return event


def make_trace(
    suite_id: str,
    *,
    fixture_key_config_hash: str,
    fixture_key_materializer_hash: str,
) -> dict:
    rows = []
    previous = "0" * 64
    for stage, rank, event_id, event_type, artifact_class, path, event_hash in (
        ("freeze_helper", 0, "freeze_complete", "transition", "stage", None, None),
        ("precommit_nonce", 1, "nonce_precommitted", "transition", "stage", None, None),
        (
            "run_primary",
            2,
            "primary_independent_a_complete",
            "transition",
            "stage",
            None,
            None,
        ),
        (
            "run_primary",
            2,
            "primary_independent_b_complete",
            "transition",
            "stage",
            None,
            None,
        ),
        (
            "lock_primaries",
            3,
            "dual_primary_lock_complete",
            "transition",
            "stage",
            None,
            None,
        ),
        (
            "run_adjudicator",
            4,
            "fixture-generation-key-read",
            "read",
            "fixture_generation_key",
            "/synthetic/fixture-key-config.json",
            fixture_key_config_hash,
        ),
        (
            "run_adjudicator",
            4,
            "fixture-key-materializer-read",
            "read",
            "fixture_key_materializer",
            "/synthetic/fixture-key-materializer.py",
            fixture_key_materializer_hash,
        ),
        (
            "run_adjudicator",
            4,
            "suite-nonce-consumed",
            "consume",
            "nonce_secret",
            "/synthetic/suite-nonce",
            h("nonce-consumed"),
        ),
        (
            "run_adjudicator",
            4,
            "adjudicator_complete",
            "transition",
            "stage",
            None,
            None,
        ),
        ("lock_all", 5, "final_lock_complete", "transition", "stage", None, None),
        ("lock_all", 5, "trace_closed", "transition", "stage", None, None),
    ):
        row = trace_event(
            ordinal=len(rows),
            stage=stage,
            rank=rank,
            artifact_id=event_id,
            previous=previous,
            event_type=event_type,
            artifact_class=artifact_class,
            path=path,
            expected_sha256=event_hash if event_type == "read" else None,
            observed_sha256=event_hash,
        )
        rows.append(row)
        previous = row["event_sha256"]
    trace = {
        "schema_version": 1,
        "interface_version": 3,
        "kind": "swe_task_state_v4_epistemic_chain_read_trace_v3",
        "suite_id": suite_id,
        "events": rows,
        "event_count": len(rows),
        "head_sha256": previous,
        "closed": True,
    }
    return {"trace": trace, "trace_sha256": scorer.sha256_value(trace)}


def role_manifest(role: str) -> dict:
    rows = []
    for ordinal, control_id in enumerate(scorer._control_ids(), start=1):
        bypass = control_id == "C32"
        result = control_proposal(control_id)
        rows.append(
            {
                "control_id": control_id,
                "ordinal": ordinal,
                "execution_path": (
                    "deterministic_host_bypass" if bypass else "native_model"
                ),
                "packet_sha256": h(f"packet-{control_id}"),
                "result": result,
                "result_sha256": scorer.sha256_value(result),
                "native_evidence": (
                    None if bypass else native_evidence(role, control_id)
                ),
            }
        )
    return {
        "role": role,
        "model_execution_count": 39,
        "host_bypass_count": 1,
        "result_count": 40,
        "ordered_results": rows,
        "role_receipt_sha256": h(f"{role}-receipt"),
    }


def fixture_schedule() -> list[dict]:
    return [
        {
            "case_id": case_id,
            "verdict_seed": 1000 + ordinal,
            "repair_seed": 2000 + ordinal,
        }
        for ordinal, case_id in enumerate(scorer._fixture_ids(), start=1)
    ]


def fixture_case_nonce(*, suite_nonce: str, case_id: str, input_hash: str) -> str:
    return scorer.sha256_value(
        {
            "domain": "v3-fixture-corpus-case-nonce-citrine",
            "suite_nonce_sha256": suite_nonce,
            "case_id": case_id,
            "input_manifest_sha256": input_hash,
        }
    )


def fixture_generation_contract(
    *,
    case_id: str,
    fixture_nonce: str,
    nonce_receipt: str,
    verdict_seed: int,
    repair_seed: int,
) -> dict:
    model = {"identity": f"model-{case_id}"}
    tokenizer = {"identity": f"tokenizer-{case_id}"}
    chat_kwargs = {"add_generation_prompt": True}
    runtime = {"identity": f"runtime-{case_id}"}
    adapter = {"identity": f"adapter-{case_id}"}
    return {
        "schema_version": 1,
        "interface_version": 3,
        "kind": scorer.FIXTURE_GENERATION_CONTRACT_KIND,
        "execution_mode": "native_protocol_adapter_non_gate",
        "inner_route_claims_actual_model_execution": False,
        "actual_model_execution_claim_responsibility": "future_sealed_outer_executor",
        "model_identity": model,
        "model_identity_sha256": scorer.sha256_value(model),
        "tokenizer_identity": tokenizer,
        "tokenizer_identity_sha256": scorer.sha256_value(tokenizer),
        "chat_template_kwargs": chat_kwargs,
        "chat_template_kwargs_sha256": scorer.sha256_value(chat_kwargs),
        "runtime_identity": runtime,
        "runtime_identity_sha256": scorer.sha256_value(runtime),
        "native_adapter_identity": adapter,
        "native_adapter_identity_sha256": scorer.sha256_value(adapter),
        "seeds": {"verdict_seed": verdict_seed, "repair_seed": repair_seed},
        "nonce_provenance": {
            "fixture_nonce_sha256": fixture_nonce,
            "nonce_precommit_sha256": scorer._fixture_nonce_precommit_hash(
                fixture_nonce
            ),
            "outer_nonce_precommit_receipt_sha256": nonce_receipt,
            "precommit_chronology_verified_by_fixture_route": False,
            "expectation_access_chronology_verified_by_fixture_route": False,
            "chronology_responsibility": "future_sealed_outer_executor",
        },
        "claim_scope": {},
    }


def fixture_lock(
    *,
    case_id: str,
    annotation_pass: str,
    fixture_nonce: str,
    nonce_receipt: str,
    verdict_seed: int,
    repair_seed: int,
    blinded_projection_hashes: list[str],
) -> tuple[dict, str]:
    contract = fixture_generation_contract(
        case_id=case_id,
        fixture_nonce=fixture_nonce,
        nonce_receipt=nonce_receipt,
        verdict_seed=verdict_seed,
        repair_seed=repair_seed,
    )
    contract_hash = scorer.sha256_value(contract)
    authored_candidates = [h(f"{case_id}-candidate-1"), h(f"{case_id}-candidate-2")]
    lock = {
        "schema_version": 1,
        "interface_version": 3,
        "kind": scorer.FIXTURE_LOCK_KIND,
        "scope": "controls_module_only_authored_candidates",
        "annotation_pass": annotation_pass,
        "packet_id_sha256": h(f"{case_id}-packet-id"),
        "packet_sha256": h(f"{case_id}-packet"),
        "source_id_sha256": h(f"{case_id}-source"),
        "candidate_unit_bundle_sha256": h(f"{case_id}-unit-bundle"),
        "codebook_sha256": h("codebook"),
        "codebook_source_sha256": h("codebook-source"),
        "production_runner_sha256": h("production-runner"),
        "fixture_runner_sha256": h("fixture-runner"),
        "fixture_config_sha256": h("fixture-config"),
        "generation_contract_sha256": contract_hash,
        "runtime_identity_sha256": contract["runtime_identity_sha256"],
        "native_adapter_identity_sha256": contract["native_adapter_identity_sha256"],
        "fixture_nonce_sha256": fixture_nonce,
        "nonce_precommit_sha256": scorer._fixture_nonce_precommit_hash(fixture_nonce),
        "outer_nonce_precommit_receipt_sha256": nonce_receipt,
        "generation_contract": contract,
        "inner_execution_mode": "native_protocol_adapter_non_gate",
        "inner_route_claims_actual_model_execution": False,
        "precommit_chronology_verified_by_fixture_route": False,
        "expectation_access_chronology_verified_by_fixture_route": False,
        "chronology_responsibility": "future_sealed_outer_executor",
        "authored_candidate_fixture_sha256s_in_original_order": authored_candidates,
        "authored_projection_sha256s_in_original_order": blinded_projection_hashes,
        "original_order_binding_sha256": h(f"{case_id}-original-order"),
        "blinded_original_indexes": [0, 1],
        "blinded_candidate_fixture_sha256s": authored_candidates,
        "blinded_projection_sha256s": blinded_projection_hashes,
        "candidate_order_sha256": h(f"{case_id}-candidate-order"),
        "model_input_bindings": {},
        "fixture_candidate_provenance": {
            "origin": "authored_bounded_control_fixture_projection",
            "intended_use": "sealed_semantic_adjudication_control_fixture_only",
            "model_generated": False,
            "native_model_record_lineage_present": False,
            "production_candidate_record_kind_claimed": False,
        },
        "candidate_model_generated": False,
        "candidate_native_model_record_lineage_present": False,
        "expectation_object_bound": False,
        "claim_scope": {},
    }
    return lock, contract_hash


def make_fixture_key(
    *,
    fixture_input_hash: str,
    fixture_key_config_hash: str,
    suite_nonce: str,
    nonce_receipt: str,
) -> dict:
    schedule = {
        "kind": "exact_per_case_verdict_and_repair_seed_schedule_v3",
        "derivation": "frozen_explicit_values_not_caller_authoritative",
        "cases": fixture_schedule(),
    }
    schedule_hash = scorer.sha256_value(schedule)
    cases = []
    for ordinal, case_id in enumerate(scorer._fixture_ids(), start=1):
        proposal = fixture_proposal(case_id)
        projection_hash = scorer.sha256_value(proposal)
        direct = ordinal <= 8
        realized = "candidate_1" if ordinal % 2 else "candidate_2"
        if direct:
            blinded = [h(f"{case_id}-wrong"), h(f"{case_id}-other")]
            blinded[0 if realized == "candidate_1" else 1] = projection_hash
            route = "direct"
        else:
            blinded = [h(f"{case_id}-wrong-1"), h(f"{case_id}-wrong-2")]
            route = "neither"
            realized = "neither"
        case_nonce = fixture_case_nonce(
            suite_nonce=suite_nonce,
            case_id=case_id,
            input_hash=fixture_input_hash,
        )
        lock, contract_hash = fixture_lock(
            case_id=case_id,
            annotation_pass=fixture_pass(case_id),
            fixture_nonce=case_nonce,
            nonce_receipt=nonce_receipt,
            verdict_seed=1000 + ordinal,
            repair_seed=2000 + ordinal,
            blinded_projection_hashes=blinded,
        )
        cases.append(
            {
                "case_id": case_id,
                "annotation_pass": fixture_pass(case_id),
                "route_requirement": route,
                "semantic_projection": proposal,
                "semantic_projection_sha256": projection_hash,
                "repair_decision": None if direct else proposal["decision"],
                "completion_chain_detail_repair_required": (
                    not direct
                    and fixture_pass(case_id) == "completion_chain"
                    and proposal["decision"] == "chain"
                ),
                "diagnostic": f"synthetic-{case_id}",
                "realized_presentation_class": realized,
                "fixture_nonce_sha256": case_nonce,
                "verdict_seed": 1000 + ordinal,
                "repair_seed": 2000 + ordinal,
                "generation_seed_schedule_sha256": schedule_hash,
                "generation_contract_sha256": contract_hash,
                "fixture_lock": lock,
                "fixture_lock_sha256": scorer.sha256_value(lock),
            }
        )
    manifest = {
        "schema_version": 1,
        "interface_version": 3,
        "kind": scorer.FIXTURE_KEY_KIND,
        "status": "prospective_draft_balance_observed_chronology_unverified_not_seal_ready",
        "scope": {},
        "input_manifest_sha256": fixture_input_hash,
        "key_config_sha256": fixture_key_config_hash,
        "externally_precommitted_suite_nonce_sha256": suite_nonce,
        "outer_nonce_precommit_receipt_sha256": nonce_receipt,
        "outer_nonce_precommit_receipt_contents_authenticated": False,
        "receipt_contents_and_issuance_authentication_responsibility": "future_sealed_outer_executor",
        "single_use_nonce_or_receipt_enforced_by_key_materializer": False,
        "single_use_nonce_or_receipt_responsibility": "future_sealed_outer_executor",
        "precommit_chronology_verified_by_key_materializer": False,
        "semantic_access_chronology_verified_by_key_materializer": False,
        "repeated_materialization_prevention_enforced_by_key_materializer": False,
        "repeated_materialization_prevention_responsibility": "future_sealed_outer_executor",
        "authoritative_packet_regeneration_performed_by_key_materializer": False,
        "authoritative_packet_regeneration_responsibility": "future_sealed_outer_executor",
        "actual_model_execution_claimed": False,
        "generation_callback_invoked": False,
        "seal_ready": False,
        "nonce_retry_permitted": False,
        "presentation_balance": {
            "intended_counts": {
                "candidate_1": 4,
                "candidate_2": 4,
                "neither": 4,
            },
            "observed_counts": {
                "candidate_1": 4,
                "candidate_2": 4,
                "neither": 4,
            },
            "balance_satisfied": True,
            "nonce_retry_permitted": False,
            "imbalance_disposition": "invalidate_entire_suite_and_start_new_predeclared_round",
        },
        "generation_seed_schedule": schedule,
        "generation_seed_schedule_sha256": schedule_hash,
        "counts": {
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
        "cases": cases,
        "ordered_fixture_lock_sha256s_sha256": scorer.sha256_value(
            [item["fixture_lock_sha256"] for item in cases]
        ),
    }
    return {"manifest": manifest, "manifest_sha256": scorer.sha256_value(manifest)}


def materialized_span(unit_id: str, ordinal: int) -> dict:
    text = f"{unit_id}-text"
    assistant_start = ordinal * 100
    source_start = 1000 + assistant_start
    return {
        "unit_id": unit_id,
        "assistant_char_start": assistant_start,
        "assistant_char_end": assistant_start + len(text),
        "source_char_start": source_start,
        "source_char_end": source_start + len(text),
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def main_key_materialization(control_id: str, proposal: dict) -> dict:
    if control_id.startswith("C"):
        if proposal["decision"] == "chain":
            annotation = {
                "annotation_status": "available",
                "unknown_reason": None,
                "has_chain": True,
                "evidence_span": materialized_span(proposal["evidence_unit_id"], 1),
                "hypothesis_span": materialized_span(proposal["hypothesis_unit_id"], 2),
                "action_span": materialized_span(proposal["action_unit_id"], 3),
                "evidence_kind": proposal["evidence_kind"],
                "belief_edge": proposal["belief_edge"],
                "hypothesis_domain": proposal["hypothesis_domain"],
                "action_intent": proposal["action_intent"],
                "relation_marker_present": True,
                "action_marker_present": True,
                "exact_signature": ">".join(
                    [
                        proposal["evidence_kind"],
                        proposal["belief_edge"],
                        proposal["hypothesis_domain"],
                        "motivates",
                        proposal["action_intent"],
                    ]
                ),
            }
            status = "resolved_authenticated_unit_chain"
        else:
            unknown = proposal["decision"] == "unknown"
            annotation = {
                "annotation_status": "semantic_unknown" if unknown else "available",
                "unknown_reason": (
                    "completion_semantics_ambiguous" if unknown else None
                ),
                "has_chain": None if unknown else False,
                "evidence_span": None,
                "hypothesis_span": None,
                "action_span": None,
                "evidence_kind": None,
                "belief_edge": None,
                "hypothesis_domain": None,
                "action_intent": None,
                "relation_marker_present": None,
                "action_marker_present": None,
                "exact_signature": None,
            }
            status = (
                "not_applicable_semantic_unknown"
                if unknown
                else "not_applicable_no_chain"
            )
        return {
            "raw_semantic_decision": proposal["decision"],
            "raw_semantic_proposal": copy.deepcopy(proposal),
            "decision_source": "prospective_control_key_draft",
            "semantic_validation_status": "valid",
            "semantic_validation_error": None,
            "materialization_status": status,
            "interface_unknown_reason": None,
            "candidate_unit_bundle_sha256": h(f"{control_id}-unit-bundle"),
            "annotation_record": annotation,
        }
    unknown = proposal["decision"] == "unknown"
    return {
        "raw_semantic_decision": proposal["decision"],
        "raw_semantic_proposal": copy.deepcopy(proposal),
        "decision_source": "prospective_control_key_draft",
        "semantic_validation_status": "valid",
        "semantic_validation_error": None,
        "materialization_status": (
            "not_applicable_semantic_unknown"
            if unknown
            else "not_applicable_novelty_classification"
        ),
        "interface_unknown_reason": None,
        "annotation_record": {
            "annotation_status": "semantic_unknown" if unknown else "available",
            "unknown_reason": "novelty_semantics_ambiguous" if unknown else None,
            "novelty_status": None if unknown else proposal["decision"],
        },
    }


def make_main_key(*, control_input_hash: str, main_key_config_hash: str) -> dict:
    completion = []
    novelty = []
    for control_id in scorer._control_ids():
        proposal = control_proposal(control_id)
        completion_pass = control_id.startswith("C")
        row = {
            "schema_version": 1,
            "interface_version": 3,
            "kind": (
                scorer.COMPLETION_KEY_ROW_KIND
                if completion_pass
                else scorer.NOVELTY_KEY_ROW_KIND
            ),
            "control_id": control_id,
            "pass": "completion" if completion_pass else "novelty",
            "packet_id_sha256": h(f"{control_id}-packet-id"),
            "input_record_sha256": h(f"{control_id}-input-record"),
            "gold_category": proposal["decision"],
            "diagnostic": f"synthetic-{control_id}",
            "gold_proposal": proposal,
            "gold_materialized_result": main_key_materialization(control_id, proposal),
            "model_invocation_required": control_id != "C32",
        }
        if completion_pass:
            row["gold_exact_unit_id_tuple"] = (
                {
                    "evidence_unit_id": proposal["evidence_unit_id"],
                    "hypothesis_unit_id": proposal["hypothesis_unit_id"],
                    "action_unit_id": proposal["action_unit_id"],
                }
                if proposal["decision"] == "chain"
                else None
            )
            completion.append(row)
        else:
            novelty.append(row)
    manifest = {
        "schema_version": 1,
        "interface_version": 3,
        "kind": scorer.MAIN_KEY_KIND,
        "status": "prospective_draft_not_sealed_not_run",
        "scope": {},
        "externally_authenticated_input_manifest_sha256": control_input_hash,
        "key_config_sha256": main_key_config_hash,
        "counts": {
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
        "completion_rows": completion,
        "novelty_rows": novelty,
        "ordered_key_rows_sha256": scorer.sha256_value(completion + novelty),
    }
    return {"manifest": manifest, "manifest_sha256": scorer.sha256_value(manifest)}


def make_public_bundle(
    *,
    trace_envelope: dict,
    generation_root: Path,
    key_root: Path,
    control_input_hash: str,
    fixture_input_hash: str,
    main_key_config_hash: str,
    fixture_key_config_hash: str,
) -> dict:
    suite_id = trace_envelope["trace"]["suite_id"]
    nonce = h("suite-nonce")
    nonce_receipt = h("nonce-receipt")
    schedule = fixture_schedule()
    roles = {role: role_manifest(role) for role in scorer._role_ids()}
    fixture_rows = []
    fixture_key = make_fixture_key(
        fixture_input_hash=fixture_input_hash,
        fixture_key_config_hash=fixture_key_config_hash,
        suite_nonce=nonce,
        nonce_receipt=nonce_receipt,
    )
    by_case = {row["case_id"]: row for row in fixture_key["manifest"]["cases"]}
    for ordinal, case_id in enumerate(scorer._fixture_ids(), start=1):
        key_row = by_case[case_id]
        result = fixture_proposal(case_id)
        fixture_rows.append(
            {
                "case_id": case_id,
                "ordinal": ordinal,
                "annotation_pass": fixture_pass(case_id),
                "result": result,
                "result_sha256": scorer.sha256_value(result),
                "fixture_lock_sha256": key_row["fixture_lock_sha256"],
                "generation_contract_sha256": key_row["generation_contract_sha256"],
                "fixture_nonce_sha256": key_row["fixture_nonce_sha256"],
                "verdict_seed": 1000 + ordinal,
                "repair_seed": 2000 + ordinal,
                "native_evidence": native_evidence("adjudicator", f"fixture-{case_id}"),
            }
        )
    chronology = {
        "freeze_helper": {
            "stage": "freeze_helper",
            "receipt_sha256": h("freeze"),
            "completed_monotonic_ns": 100,
        },
        "precommit_nonce": {
            "stage": "precommit_nonce",
            "receipt_sha256": nonce_receipt,
            "completed_monotonic_ns": 200,
        },
        "primary_roles": {
            "independent_a": {
                "stage": "run_primary",
                "receipt_sha256": roles["independent_a"]["role_receipt_sha256"],
                "completed_monotonic_ns": 300,
            },
            "independent_b": {
                "stage": "run_primary",
                "receipt_sha256": roles["independent_b"]["role_receipt_sha256"],
                "completed_monotonic_ns": 310,
            },
        },
        "lock_primaries": {
            "stage": "lock_primaries",
            "receipt_sha256": h("dual-lock"),
            "completed_monotonic_ns": 400,
        },
        "adjudicator": {
            "stage": "run_adjudicator",
            "receipt_sha256": roles["adjudicator"]["role_receipt_sha256"],
            "completed_monotonic_ns": 500,
        },
        "lock_all": {
            "stage": "lock_all",
            "receipt_sha256": h("final-lock"),
            "completed_monotonic_ns": 600,
        },
    }
    manifest = {
        "schema_version": 1,
        "interface_version": 3,
        "kind": "swe_task_state_v4_epistemic_chain_public_generation_bundle_v3",
        "status": "all_generation_outputs_locked_scoring_keys_unread",
        "scope": {},
        "suite_id": suite_id,
        "executor_identity": {},
        "suite_nonce": {
            "suite_nonce_sha256": nonce,
            "precommit_receipt_sha256": nonce_receipt,
            "single_use_consumption_receipt_sha256": h("nonce-consumed"),
            "retry_permitted": False,
            "raw_nonce_public": False,
        },
        "filesystem_roots": {
            "generation_root": str(generation_root),
            "key_root": str(key_root),
        },
        "inputs": {
            "control": {"path": "synthetic-control", "sha256": control_input_hash},
            "fixture": {"path": "synthetic-fixture", "sha256": fixture_input_hash},
        },
        "key_commitments": {
            "fixture_generation_key": {
                "sha256": fixture_key_config_hash,
                "read_by_executor": True,
                "first_read_stage": "run_adjudicator",
                "read_count": 1,
            },
            "control_scoring_key": {
                "sha256": main_key_config_hash,
                "read_by_executor": False,
                "scorer_reads_last": True,
            },
        },
        "chronology": chronology,
        "roles": roles,
        "fixture_results": fixture_rows,
        "fixture_seed_schedule": schedule,
        "fixture_seed_schedule_sha256": scorer.sha256_value(schedule),
        "locks": {
            "primary_independent_a_sha256": roles["independent_a"][
                "role_receipt_sha256"
            ],
            "primary_independent_b_sha256": roles["independent_b"][
                "role_receipt_sha256"
            ],
            "dual_primary_lock_sha256": h("dual-lock"),
            "adjudicator_sha256": roles["adjudicator"]["role_receipt_sha256"],
            "final_lock_sha256": h("final-lock"),
        },
        "read_trace": {
            "trace_envelope_sha256": scorer.sha256_value(trace_envelope),
            "trace_sha256": trace_envelope["trace_sha256"],
            "head_sha256": trace_envelope["trace"]["head_sha256"],
            "event_count": trace_envelope["trace"]["event_count"],
        },
        "claims": {
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
    }
    bundle = {"manifest": manifest, "manifest_sha256": scorer.sha256_value(manifest)}
    return bundle


def rehash_bundle(bundle: dict) -> dict:
    result = copy.deepcopy(bundle)
    result["manifest_sha256"] = scorer.sha256_value(result["manifest"])
    return result


def rehash_trace(trace_envelope: dict) -> dict:
    result = copy.deepcopy(trace_envelope)
    previous = "0" * 64
    for ordinal, event in enumerate(result["trace"]["events"]):
        event["ordinal"] = ordinal
        event["previous_event_sha256"] = previous
        event["event_sha256"] = scorer.sha256_value(
            {name: item for name, item in event.items() if name != "event_sha256"}
        )
        previous = event["event_sha256"]
    result["trace"]["event_count"] = len(result["trace"]["events"])
    result["trace"]["head_sha256"] = previous
    result["trace_sha256"] = scorer.sha256_value(result["trace"])
    return result


class FakeExecutor:
    @staticmethod
    def validate_read_trace_envelope(value, *, independently_supplied_sha256):
        if scorer.sha256_value(value) != independently_supplied_sha256:
            raise ValueError("trace external hash")
        return copy.deepcopy(value["trace"])

    @staticmethod
    def validate_public_generation_bundle(
        value, *, independently_supplied_sha256, authenticated_trace
    ):
        if scorer.sha256_value(value) != independently_supplied_sha256:
            raise ValueError("bundle external hash")
        if value["manifest"]["suite_id"] != authenticated_trace["suite_id"]:
            raise ValueError("suite mismatch")
        return copy.deepcopy(value["manifest"])


class ScorerFixture:
    def __init__(self, directory: Path):
        self.root = directory
        self.generation = directory / "generation"
        self.keys = directory / "keys"
        self.generation.mkdir()
        self.keys.mkdir()
        self.control_input_hash = h("control-input")
        self.fixture_input_hash = h("fixture-input")
        executor_config = json.loads(EXECUTOR_CONFIG.read_text(encoding="utf-8"))
        key_contracts = executor_config["key_contracts"]
        self.main_key_config_hash = key_contracts["control_scoring_key"][
            "config_sha256"
        ]
        self.fixture_key_config_hash = key_contracts["fixture_generation_key"][
            "config_sha256"
        ]
        self.trace = make_trace(
            "development-scorer-suite-citrine-v3",
            fixture_key_config_hash=self.fixture_key_config_hash,
            fixture_key_materializer_hash=key_contracts["fixture_generation_key"][
                "materializer_sha256"
            ],
        )
        self.main_key = make_main_key(
            control_input_hash=self.control_input_hash,
            main_key_config_hash=self.main_key_config_hash,
        )
        self.fixture_key = make_fixture_key(
            fixture_input_hash=self.fixture_input_hash,
            fixture_key_config_hash=self.fixture_key_config_hash,
            suite_nonce=h("suite-nonce"),
            nonce_receipt=h("nonce-receipt"),
        )
        self.bundle = make_public_bundle(
            trace_envelope=self.trace,
            generation_root=self.generation,
            key_root=self.keys,
            control_input_hash=self.control_input_hash,
            fixture_input_hash=self.fixture_input_hash,
            main_key_config_hash=self.main_key_config_hash,
            fixture_key_config_hash=self.fixture_key_config_hash,
        )
        self.trace_path = self.generation / "trace.json"
        self.bundle_path = self.generation / "bundle.json"
        self.main_key_path = self.keys / "main-key.json"
        self.fixture_key_path = self.keys / "fixture-key.json"
        self.receipt_path = directory / "score-receipt.json"
        self.write_all()

    def write_all(self):
        self.trace_path.write_bytes(raw_json(self.trace))
        self.bundle_path.write_bytes(raw_json(self.bundle))
        self.main_key_path.write_bytes(raw_json(self.main_key))
        self.fixture_key_path.write_bytes(raw_json(self.fixture_key))

    def kwargs(self) -> dict:
        return {
            "config_path": CONFIG,
            "expected_config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "expected_scorer_source_sha256": hashlib.sha256(
                SOURCE.read_bytes()
            ).hexdigest(),
            "expected_executor_config_sha256": hashlib.sha256(
                EXECUTOR_CONFIG.read_bytes()
            ).hexdigest(),
            "expected_executor_source_sha256": hashlib.sha256(
                EXECUTOR_SOURCE.read_bytes()
            ).hexdigest(),
            "read_trace_path": self.trace_path,
            "expected_read_trace_file_sha256": hashlib.sha256(
                self.trace_path.read_bytes()
            ).hexdigest(),
            "expected_read_trace_envelope_sha256": scorer.sha256_value(self.trace),
            "public_bundle_path": self.bundle_path,
            "expected_public_bundle_file_sha256": hashlib.sha256(
                self.bundle_path.read_bytes()
            ).hexdigest(),
            "expected_public_bundle_envelope_sha256": scorer.sha256_value(self.bundle),
            "main_control_key_path": self.main_key_path,
            "expected_main_control_key_file_sha256": hashlib.sha256(
                self.main_key_path.read_bytes()
            ).hexdigest(),
            "expected_main_control_key_manifest_sha256": self.main_key[
                "manifest_sha256"
            ],
            "fixture_adjudication_key_path": self.fixture_key_path,
            "expected_fixture_adjudication_key_file_sha256": hashlib.sha256(
                self.fixture_key_path.read_bytes()
            ).hexdigest(),
            "expected_fixture_adjudication_key_manifest_sha256": self.fixture_key[
                "manifest_sha256"
            ],
            "receipt_path": self.receipt_path,
        }


class SealedControlScorerV3Tests(unittest.TestCase):
    def call(self, fixture: ScorerFixture, *, prepared_kwargs=None, **changes):
        kwargs = fixture.kwargs() if prepared_kwargs is None else dict(prepared_kwargs)
        kwargs.update(changes)
        with mock.patch.object(
            scorer, "_load_authenticated_executor_module", return_value=FakeExecutor()
        ):
            return scorer.score_finalized_executor_bundle(**kwargs)

    def test_valid_score_reads_keys_last_and_emits_exclusive_false_claim_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScorerFixture(Path(directory))
            observed_reads = []
            original = Path.read_bytes

            def tracked(path):
                observed_reads.append(path.resolve(strict=False))
                return original(path)

            with mock.patch.object(Path, "read_bytes", tracked):
                envelope = self.call(fixture)
            receipt = scorer.validate_score_receipt_envelope(
                envelope,
                independently_supplied_receipt_sha256=envelope["receipt_sha256"],
            )
            self.assertTrue(fixture.receipt_path.is_file())
            self.assertEqual(
                observed_reads[-2:],
                [fixture.main_key_path.resolve(), fixture.fixture_key_path.resolve()],
            )
            self.assertEqual(
                receipt["read_chronology"]["key_read_order"],
                ["main_control_key", "fixture_adjudication_key"],
            )
            self.assertTrue(all(value is False for value in receipt["claims"].values()))
            for role in scorer._role_ids():
                self.assertEqual(
                    receipt["role_scores"][role]["exact_projection"],
                    {"correct": 40, "total": 40},
                )
            self.assertEqual(
                receipt["fixture_score"]["exact_final_projection"],
                {"correct": 12, "total": 12},
            )

    def test_bad_public_hash_rejects_before_either_key_read(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScorerFixture(Path(directory))
            prepared_kwargs = fixture.kwargs()
            reads = []
            original = Path.read_bytes

            def tracked(path):
                resolved = path.resolve(strict=False)
                if resolved in {
                    fixture.main_key_path.resolve(),
                    fixture.fixture_key_path.resolve(),
                }:
                    reads.append(resolved)
                return original(path)

            with (
                mock.patch.object(Path, "read_bytes", tracked),
                self.assertRaisesRegex(
                    scorer.SealedControlScorerError, "canonical-envelope hash"
                ),
            ):
                self.call(
                    fixture,
                    prepared_kwargs=prepared_kwargs,
                    expected_public_bundle_envelope_sha256="0" * 64,
                )
            self.assertEqual(reads, [])

    def test_bad_scorer_source_hash_rejects_before_either_key_read(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScorerFixture(Path(directory))
            prepared_kwargs = fixture.kwargs()
            reads = []
            original = Path.read_bytes

            def tracked(path):
                resolved = path.resolve(strict=False)
                if resolved in {
                    fixture.main_key_path.resolve(),
                    fixture.fixture_key_path.resolve(),
                }:
                    reads.append(resolved)
                return original(path)

            with (
                mock.patch.object(Path, "read_bytes", tracked),
                self.assertRaisesRegex(
                    scorer.SealedControlScorerError, "scorer_source differs"
                ),
            ):
                self.call(
                    fixture,
                    prepared_kwargs=prepared_kwargs,
                    expected_scorer_source_sha256="0" * 64,
                )
            self.assertEqual(reads, [])

    def test_public_seed_retry_label_and_native_evidence_mutations_fail_before_keys(
        self,
    ):
        def mutate_seed(bundle):
            bundle["manifest"]["fixture_seed_schedule"][0]["verdict_seed"] += 1
            bundle["manifest"]["fixture_seed_schedule_sha256"] = scorer.sha256_value(
                bundle["manifest"]["fixture_seed_schedule"]
            )

        def mutate_retry(bundle):
            bundle["manifest"]["suite_nonce"]["retry_permitted"] = True

        def mutate_label(bundle):
            bundle["manifest"]["scope"]["expected_label"] = "caller-controlled"

        def mutate_native_evidence(bundle):
            del bundle["manifest"]["roles"]["independent_a"]["ordered_results"][0][
                "native_evidence"
            ]["outer_launch_authorization_sha256s"]

        for mutate, pattern in (
            (mutate_seed, "seed schedule"),
            (mutate_retry, "suite-nonce"),
            (mutate_label, "caller-authoritative label"),
            (mutate_native_evidence, "native evidence"),
        ):
            with (
                self.subTest(pattern=pattern),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = ScorerFixture(Path(directory))
                mutate(fixture.bundle)
                fixture.bundle = rehash_bundle(fixture.bundle)
                fixture.bundle_path.write_bytes(raw_json(fixture.bundle))
                prepared_kwargs = fixture.kwargs()
                key_reads = []
                original = Path.read_bytes

                def tracked(path):
                    resolved = path.resolve(strict=False)
                    if resolved in {
                        fixture.main_key_path.resolve(),
                        fixture.fixture_key_path.resolve(),
                    }:
                        key_reads.append(resolved)
                    return original(path)

                with (
                    mock.patch.object(Path, "read_bytes", tracked),
                    self.assertRaisesRegex(scorer.SealedControlScorerError, pattern),
                ):
                    self.call(fixture, prepared_kwargs=prepared_kwargs)
                self.assertEqual(key_reads, [])

    def test_rehashed_trace_key_commitment_substitution_fails_before_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScorerFixture(Path(directory))
            key_event = next(
                event
                for event in fixture.trace["trace"]["events"]
                if event["artifact_class"] == "fixture_generation_key"
            )
            key_event["expected_sha256"] = h("substituted-key")
            key_event["observed_sha256"] = h("substituted-key")
            fixture.trace = rehash_trace(fixture.trace)
            trace_binding = fixture.bundle["manifest"]["read_trace"]
            trace_binding["trace_envelope_sha256"] = scorer.sha256_value(fixture.trace)
            trace_binding["trace_sha256"] = fixture.trace["trace_sha256"]
            trace_binding["head_sha256"] = fixture.trace["trace"]["head_sha256"]
            trace_binding["event_count"] = fixture.trace["trace"]["event_count"]
            fixture.bundle = rehash_bundle(fixture.bundle)
            fixture.write_all()
            prepared_kwargs = fixture.kwargs()
            key_reads = []
            original = Path.read_bytes

            def tracked(path):
                resolved = path.resolve(strict=False)
                if resolved in {
                    fixture.main_key_path.resolve(),
                    fixture.fixture_key_path.resolve(),
                }:
                    key_reads.append(resolved)
                return original(path)

            with (
                mock.patch.object(Path, "read_bytes", tracked),
                self.assertRaisesRegex(
                    scorer.SealedControlScorerError,
                    "trace differs from key/materializer/nonce commitments",
                ),
            ):
                self.call(fixture, prepared_kwargs=prepared_kwargs)
            self.assertEqual(key_reads, [])

    def test_coherent_output_rehash_fails_independent_bundle_root_before_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScorerFixture(Path(directory))
            external = scorer.sha256_value(fixture.bundle)
            row = fixture.bundle["manifest"]["roles"]["independent_a"][
                "ordered_results"
            ][0]
            row["result"] = {"decision": "no_chain"}
            row["result_sha256"] = scorer.sha256_value(row["result"])
            fixture.bundle = rehash_bundle(fixture.bundle)
            fixture.bundle_path.write_bytes(raw_json(fixture.bundle))
            with self.assertRaisesRegex(
                scorer.SealedControlScorerError, "canonical-envelope hash"
            ):
                self.call(
                    fixture,
                    expected_public_bundle_file_sha256=hashlib.sha256(
                        fixture.bundle_path.read_bytes()
                    ).hexdigest(),
                    expected_public_bundle_envelope_sha256=external,
                )

    def test_control_order_replay_and_extra_bypass_reject_before_keys(self):
        mutators = []

        def reorder(bundle):
            rows = bundle["manifest"]["roles"]["independent_a"]["ordered_results"]
            rows[0], rows[1] = rows[1], rows[0]

        mutators.append((reorder, "order"))

        def replay(bundle):
            rows = bundle["manifest"]["roles"]["independent_a"]["ordered_results"]
            rows[1]["native_evidence"]["native_request_sha256s"] = copy.deepcopy(
                rows[0]["native_evidence"]["native_request_sha256s"]
            )

        mutators.append((replay, "replay"))

        def bypass(bundle):
            role = bundle["manifest"]["roles"]["independent_a"]
            row = role["ordered_results"][30]
            row["execution_path"] = "deterministic_host_bypass"
            row["native_evidence"] = None
            role["model_execution_count"] = 38
            role["host_bypass_count"] = 2

        mutators.append((bypass, "counts"))

        for mutate, pattern in mutators:
            with (
                self.subTest(pattern=pattern),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = ScorerFixture(Path(directory))
                mutate(fixture.bundle)
                fixture.bundle = rehash_bundle(fixture.bundle)
                fixture.bundle_path.write_bytes(raw_json(fixture.bundle))
                with self.assertRaisesRegex(scorer.SealedControlScorerError, pattern):
                    self.call(
                        fixture,
                        expected_public_bundle_file_sha256=hashlib.sha256(
                            fixture.bundle_path.read_bytes()
                        ).hexdigest(),
                        expected_public_bundle_envelope_sha256=scorer.sha256_value(
                            fixture.bundle
                        ),
                    )

    def test_fixture_lock_co_tamper_with_new_key_hash_fails_public_lock_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScorerFixture(Path(directory))
            case = fixture.fixture_key["manifest"]["cases"][0]
            case["fixture_lock"]["blinded_projection_sha256s"][0] = h("tampered")
            case["fixture_lock_sha256"] = scorer.sha256_value(case["fixture_lock"])
            fixture.fixture_key["manifest"]["ordered_fixture_lock_sha256s_sha256"] = (
                scorer.sha256_value(
                    [
                        row["fixture_lock_sha256"]
                        for row in fixture.fixture_key["manifest"]["cases"]
                    ]
                )
            )
            fixture.fixture_key["manifest_sha256"] = scorer.sha256_value(
                fixture.fixture_key["manifest"]
            )
            fixture.fixture_key_path.write_bytes(raw_json(fixture.fixture_key))
            with self.assertRaisesRegex(
                scorer.SealedControlScorerError,
                "public fixture locks|lock, nonce, contract|direct presentation|blinding permutation",
            ):
                self.call(
                    fixture,
                    expected_fixture_adjudication_key_file_sha256=hashlib.sha256(
                        fixture.fixture_key_path.read_bytes()
                    ).hexdigest(),
                    expected_fixture_adjudication_key_manifest_sha256=fixture.fixture_key[
                        "manifest_sha256"
                    ],
                )

    def test_fake_execution_or_gate_claim_rejects_with_coherent_bundle_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScorerFixture(Path(directory))
            fixture.bundle["manifest"]["claims"]["gate_passed"] = True
            fixture.bundle = rehash_bundle(fixture.bundle)
            fixture.bundle_path.write_bytes(raw_json(fixture.bundle))
            with self.assertRaisesRegex(
                scorer.SealedControlScorerError, "claim contract"
            ):
                self.call(
                    fixture,
                    expected_public_bundle_file_sha256=hashlib.sha256(
                        fixture.bundle_path.read_bytes()
                    ).hexdigest(),
                    expected_public_bundle_envelope_sha256=scorer.sha256_value(
                        fixture.bundle
                    ),
                )

    def test_receipt_claim_co_tamper_rejects_even_with_new_self_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScorerFixture(Path(directory))
            envelope = self.call(fixture)
            changed = copy.deepcopy(envelope)
            changed["receipt"]["claims"]["gate_eligible"] = True
            changed["receipt_sha256"] = scorer.sha256_value(changed["receipt"])
            with self.assertRaisesRegex(scorer.SealedControlScorerError, "claim"):
                scorer.validate_score_receipt_envelope(
                    changed,
                    independently_supplied_receipt_sha256=changed["receipt_sha256"],
                )
            self.assertFalse(hasattr(scorer, "FALSE_EXECUTION_CLAIMS"))

    def test_existing_receipt_rejects_before_any_key_read(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScorerFixture(Path(directory))
            prepared_kwargs = fixture.kwargs()
            fixture.receipt_path.write_bytes(b"occupied")
            with (
                mock.patch.object(
                    Path,
                    "read_bytes",
                    side_effect=AssertionError("input touched after occupied output"),
                ),
                self.assertRaisesRegex(
                    scorer.SealedControlScorerError, "already exists"
                ),
            ):
                self.call(fixture, prepared_kwargs=prepared_kwargs)

    def test_fixture_key_not_read_if_main_key_hash_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScorerFixture(Path(directory))
            prepared_kwargs = fixture.kwargs()
            reads = []
            original = Path.read_bytes

            def tracked(path):
                resolved = path.resolve(strict=False)
                if resolved in {
                    fixture.main_key_path.resolve(),
                    fixture.fixture_key_path.resolve(),
                }:
                    reads.append(resolved)
                return original(path)

            with (
                mock.patch.object(Path, "read_bytes", tracked),
                self.assertRaisesRegex(
                    scorer.SealedControlScorerError, "main_control_key differs"
                ),
            ):
                self.call(
                    fixture,
                    prepared_kwargs=prepared_kwargs,
                    expected_main_control_key_file_sha256="0" * 64,
                )
            self.assertEqual(reads, [fixture.main_key_path.resolve()])

    def test_main_key_schema_co_tamper_rejects_before_fixture_key_read(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScorerFixture(Path(directory))
            fixture.main_key["manifest"]["completion_rows"][0]["unexpected"] = True
            fixture.main_key["manifest"]["ordered_key_rows_sha256"] = (
                scorer.sha256_value(
                    fixture.main_key["manifest"]["completion_rows"]
                    + fixture.main_key["manifest"]["novelty_rows"]
                )
            )
            fixture.main_key["manifest_sha256"] = scorer.sha256_value(
                fixture.main_key["manifest"]
            )
            fixture.main_key_path.write_bytes(raw_json(fixture.main_key))
            prepared_kwargs = fixture.kwargs()
            key_reads = []
            original = Path.read_bytes

            def tracked(path):
                resolved = path.resolve(strict=False)
                if resolved in {
                    fixture.main_key_path.resolve(),
                    fixture.fixture_key_path.resolve(),
                }:
                    key_reads.append(resolved)
                return original(path)

            with (
                mock.patch.object(Path, "read_bytes", tracked),
                self.assertRaisesRegex(
                    scorer.SealedControlScorerError, "row or invocation contract"
                ),
            ):
                self.call(fixture, prepared_kwargs=prepared_kwargs)
            self.assertEqual(key_reads, [fixture.main_key_path.resolve()])

    def test_same_inode_keys_reject_before_semantic_content_read(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScorerFixture(Path(directory))
            fixture.fixture_key_path.unlink()
            os.link(fixture.main_key_path, fixture.fixture_key_path)
            with self.assertRaisesRegex(
                scorer.SealedControlScorerError, "physically distinct"
            ):
                self.call(
                    fixture,
                    expected_fixture_adjudication_key_file_sha256=hashlib.sha256(
                        fixture.fixture_key_path.read_bytes()
                    ).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
