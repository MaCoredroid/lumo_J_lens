#!/usr/bin/env python3
"""Common-ontology, fail-closed COT-like candidate-chain sidecar.

The sidecar ranks the same frozen lexical concept ontology at every permitted
teacher-forced boundary. Human positive labels are structurally separate and
used only for retrospective evaluation. A semantic candidate is emitted only
when both public and native J-lens top-one rankings agree and both reports pass
strict numerical fidelity at that boundary.

This is not private chain-of-thought recovery. It is an uncalibrated,
vocabulary-derived candidate decoder over one retrospectively annotated task.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/swe_task_state_v4_concept_chain.json"
ANALYSIS_RELATIVE_PATH = ".cache/swe_jlens_intermediate/analysis.json"
ANALYSIS_PATH = ROOT / ANALYSIS_RELATIVE_PATH
ANALYSIS_SHA256 = "29f7cb2f1ffe7948f7836c49db46020864fd4e2876ec060ca03637edd5e034db"
PROBE_CONFIG_RELATIVE_PATH = "configs/swe_intermediate_concept_probes.json"
PROBE_CONFIG_PATH = ROOT / PROBE_CONFIG_RELATIVE_PATH
PROBE_CONFIG_SHA256 = "2cae42b3b3f559209a81ae80d55800ff215be0786a28865b5f95d0a16fdba1cc"
PUBLIC_REPORT_RELATIVE_PATH = ".cache/swe_jlens_intermediate/public-report.json"
PUBLIC_REPORT_PATH = ROOT / PUBLIC_REPORT_RELATIVE_PATH
PUBLIC_REPORT_SHA256 = "16a8c781db6c3dea9dd6602a1e6a113d1a7b29b5ada1d1668168a0ff0d9290b7"
NATIVE_REPORT_RELATIVE_PATH = ".cache/swe_jlens_intermediate/native-report.json"
NATIVE_REPORT_PATH = ROOT / NATIVE_REPORT_RELATIVE_PATH
NATIVE_REPORT_SHA256 = "a307b236c259bc58703ba1449ecfe404f6387376a1db10d3f675b0c1c21b5068"
PROMPT_BUNDLE_SHA256 = "fc3293d64323cb25ea4ae1626e5a3508c983bec18a25ff2241ddf74d3a86e9fc"
TRAJECTORY_BUNDLE_SHA256 = "3b206326f138a0dbde2cdc363fcc2abe1e629715e6400182ad1cdf297cfbe0b4"
OUTPUT_PATH = ROOT / ".cache/swe_task_state_v4_concept_chain/common-ontology-chain.json"

SCHEMA_VERSION = 2
PROTOCOL_ID = "swe-task-state-v4-common-ontology-chain-v2"
PROTOCOL_STATUS = "exploratory_common_ontology_decoder_reliability_unestablished"
OUTPUT_KIND = "swe_task_state_v4_common_ontology_candidate_chain"
ANALYSIS_KIND = "exploratory_swe_intermediate_concept_analysis"
LAYERS = tuple(range(16, 48))
SCORE_SOURCES = ("public_j", "native_j", "ordinary_logit")
REGISTERED_CONCEPTS = (
    "source_localization",
    "substitution_operation",
    "located_source",
    "defined_identifier",
    "typographical_error",
    "runtime_name_failure",
    "failure_confirmation",
    "source_edit",
    "repair",
    "repair_success",
    "verification",
    "broad_success",
    "dependency_unavailable",
    "focused_validation",
    "test_success",
    "task_resolution",
    "repair_summary",
)
SCORABLE_CONCEPTS = (
    "source_localization",
    "substitution_operation",
    "located_source",
    "defined_identifier",
    "runtime_name_failure",
    "failure_confirmation",
    "source_edit",
    "repair",
    "verification",
    "broad_success",
    "dependency_unavailable",
    "focused_validation",
    "test_success",
    "task_resolution",
)
EXCLUDED_CONCEPTS = {
    "typographical_error": "only_one_globally_unique_registered_token_form",
    "repair_success": "no_globally_unique_registered_token_form",
    "repair_summary": "no_globally_unique_registered_token_form",
}
CONCEPT_FAMILY_LEXICON = {
    concept_id: f"{concept_id.replace('_', '-') } concept family"
    for concept_id in REGISTERED_CONCEPTS
}
LEAKAGE_CLASSES = (
    "task_explicit",
    "tool_outcome_explicit",
    "tool_outcome_implicit_success",
    "teacher_forced_explicit_positive_control",
)
INFERENCE_INPUT_ALLOWLIST = (
    "boundary_id",
    "request_index",
    "offset",
    "numerical_fidelity",
    "source_concept_scores",
)
EVALUATION_LABEL_ALLOWLIST = (
    "boundary_id",
    "positive_concept_ids",
    "leakage_class",
)
FORBIDDEN_FIELDS = (
    "state",
    "rationale",
    "event_family",
    "best_token",
    "best_token_id",
    "forms",
    "task_text",
    "reasoning",
    "reasoning_content",
    "completion",
    "current_tool_result",
    "accepted_target_token",
    "accepted_target_token_id",
    "later_actions",
    "official_outcome",
    "positive_concept_ids",
    "leakage_class",
)
CLAIM_SCOPE = {
    "kind": "uncalibrated_common_ontology_candidate_chain",
    "private_chain_of_thought_reconstructed": False,
    "subjective_emotion_inferred": False,
    "task_specific_intent_inferred": False,
    "causal_explanation_recovered": False,
    "raw_activation_decoder_used": False,
    "concepts_absent_from_jsonl_visible_prefix_established": False,
    "future_metadata_used_at_inference": False,
    "human_positive_labels_used_for_inference": False,
    "human_positive_labels_used_for_evaluation_only": True,
    "ontology_selected_from_completed_visible_trace": True,
    "coordinates_selected_from_completed_visible_trace": True,
    "task_or_completion_text_used_by_renderer": False,
    "retrospective_summary_may_relabel_prior_nodes": False,
    "grounding": "common_ontology_vocabulary_readout_not_activation_decoder",
    "rendering": "deterministic_closed_concept_family_templates",
}
CAUSAL_CUTOFF = "exact_teacher_forced_prefix_before_the_accepted_target_token"
COORDINATE_KIND = "teacher_forced_replay_request_and_completion_offset"
SCORING = {
    "registered_concept_count": 17,
    "scorable_concept_count": 14,
    "shared_token_policy": "exclude_every_token_id_registered_to_more_than_one_concept",
    "minimum_unique_forms_per_scorable_concept": 2,
    "layer_ids": list(LAYERS),
    "concept_score": "arithmetic_mean_logprob_over_every_unique_form_and_every_fixed_layer",
    "higher_score_is_better": True,
    "source_tie_break": "frozen_scorable_concept_order",
    "score_is_probability": False,
    "score_is_confidence": False,
    "top_k_diagnostics": 3,
}
SELECTION = {
    "numerical_fidelity_required": "public_and_native_strict_adapter_pass",
    "candidate_rule": "strictly_unique_public_j_and_native_j_top1_concept_agreement",
    "probability_threshold": None,
    "calibrated": False,
    "claims_gate_preregistered": False,
    "operational_chain_available_without_evaluation": False,
    "ordinary_logit_used_for_selection": False,
}
RELATION_TYPES = ("later_source_agreement_candidate",)
RENDERER = {
    "renderer_id": "common-ontology-candidate-chain-renderer-v2",
    "boundary_template_id": "paired-source-top1-candidate-or-abstention-v2",
    "chain_template_id": "sparse-candidates-with-explicit-gaps-v2",
    "score_decimal_places": 3,
    "evidence_status": "exploratory_one_task_uncalibrated",
    "mandatory_disclaimer": (
        "This sparse sequence is an uncalibrated common-ontology vocabulary-readout "
        "candidate chain, not private chain-of-thought, hidden content, intent, "
        "causal explanation, or emotion."
    ),
    "teacher_forcing_disclaimer": (
        "These states come from exact-prefix teacher-forced replay, not captured "
        "generation-time hidden states."
    ),
    "retrospective_selection_disclaimer": (
        "The ontology, positive evaluation labels, and ten coordinates were "
        "retrospectively specified from the completed visible trace; they provide "
        "no evidence of concepts absent from the JSONL-visible prefix."
    ),
}
EVIDENCE_STATUS = {
    "task_count": 1,
    "repository_count": 1,
    "registered_boundary_count": 10,
    "strict_fidelity_boundary_count": 5,
    "non_prefix_explicit_boundary_count": 0,
    "fresh_transport_confirmation": False,
    "operational_reliability_claim": False,
    "activation_decoder_available": False,
    "concept_probability_calibration_available": False,
    "reserved_validation_opened": False,
}
LIMITATIONS = (
    "The ontology, positive labels, and coordinates were retrospectively derived from one completed visible SWE trace.",
    "Every evaluation boundary is task/tool explicit, implicit-success, or a teacher-forced lexical positive control.",
    "Only five of ten paired boundaries pass strict adapter fidelity.",
    "The fourteen scorable concept families share no scoring token IDs, but they remain lexical associations rather than propositions.",
    "Source top-one agreement is an abstention heuristic, not calibrated confidence.",
    "The current artifact cannot establish concepts absent from JSONL, reliable COT-like decoding, or incremental J value.",
    "A hash-authenticated raw-residual/public-J-state artifact and held-out observable-event decoder are required for a beyond-word-probes claim.",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray)),
        f"{label} must be a sequence",
    )
    return value


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be nonempty text")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return int(value)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, Real) and not isinstance(value, bool),
        f"{label} must be numeric and non-boolean",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _detached(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def validate_protocol(value: Any) -> dict[str, Any]:
    protocol = _mapping(value, "concept-chain protocol")
    _require(
        set(protocol)
        == {
            "schema_version",
            "protocol_id",
            "status",
            "claim_scope",
            "causal_cutoff",
            "coordinate_kind",
            "inference_input_allowlist",
            "evaluation_label_allowlist",
            "forbidden_inference_or_render_fields",
            "score_sources",
            "scoring",
            "selection",
            "relation_types",
            "registered_concept_order",
            "scorable_concept_order",
            "excluded_concepts",
            "concept_family_lexicon",
            "leakage_classes",
            "renderer",
            "authenticated_inputs",
            "evidence_status",
            "limitations",
        },
        "concept-chain protocol top-level fields changed",
    )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": PROTOCOL_STATUS,
        "claim_scope": CLAIM_SCOPE,
        "causal_cutoff": CAUSAL_CUTOFF,
        "coordinate_kind": COORDINATE_KIND,
        "inference_input_allowlist": list(INFERENCE_INPUT_ALLOWLIST),
        "evaluation_label_allowlist": list(EVALUATION_LABEL_ALLOWLIST),
        "forbidden_inference_or_render_fields": list(FORBIDDEN_FIELDS),
        "score_sources": list(SCORE_SOURCES),
        "scoring": SCORING,
        "selection": SELECTION,
        "relation_types": list(RELATION_TYPES),
        "registered_concept_order": list(REGISTERED_CONCEPTS),
        "scorable_concept_order": list(SCORABLE_CONCEPTS),
        "excluded_concepts": EXCLUDED_CONCEPTS,
        "concept_family_lexicon": CONCEPT_FAMILY_LEXICON,
        "leakage_classes": list(LEAKAGE_CLASSES),
        "renderer": RENDERER,
        "evidence_status": EVIDENCE_STATUS,
        "limitations": list(LIMITATIONS),
    }
    for field, field_expected in expected.items():
        _require(protocol.get(field) == field_expected, f"concept-chain protocol {field} changed")
    authenticated = _mapping(protocol.get("authenticated_inputs"), "authenticated inputs")
    _require(
        dict(authenticated)
        == {
            "analysis_path": ANALYSIS_RELATIVE_PATH,
            "analysis_sha256": ANALYSIS_SHA256,
            "analysis_kind": ANALYSIS_KIND,
            "probe_config_path": PROBE_CONFIG_RELATIVE_PATH,
            "probe_config_sha256": PROBE_CONFIG_SHA256,
            "public_report_path": PUBLIC_REPORT_RELATIVE_PATH,
            "public_report_sha256": PUBLIC_REPORT_SHA256,
            "native_report_path": NATIVE_REPORT_RELATIVE_PATH,
            "native_report_sha256": NATIVE_REPORT_SHA256,
            "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
            "trajectory_bundle_sha256": TRAJECTORY_BUNDLE_SHA256,
        },
        "concept-chain authenticated inputs changed",
    )
    return _detached(protocol)


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    expanded = path.expanduser()
    _require(
        expanded.is_file() and not expanded.is_symlink(),
        "protocol must be a regular non-symlink file",
    )
    try:
        value = json.loads(expanded.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not read concept-chain protocol") from error
    return validate_protocol(value)


def _score_map(value: Any, label: str) -> dict[str, float]:
    source = _mapping(value, label)
    _require(set(source) == set(SCORABLE_CONCEPTS), f"{label} concept grid changed")
    return {
        concept_id: _finite(source[concept_id], f"{label}.{concept_id}")
        for concept_id in SCORABLE_CONCEPTS
    }


def _normalize_boundaries(value: Any) -> list[dict[str, Any]]:
    rows = list(_sequence(value, "score boundaries"))
    _require(bool(rows), "score boundaries must not be empty")
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_coordinates: set[tuple[int, int]] = set()
    previous_coordinate: tuple[int, int] | None = None
    for position, raw in enumerate(rows):
        row = _mapping(raw, f"score boundary {position}")
        boundary_id = _text(row.get("boundary_id"), f"score boundary {position} ID")
        _require(boundary_id not in seen_ids, f"duplicate score boundary ID: {boundary_id}")
        seen_ids.add(boundary_id)
        request_index = _integer(
            row.get("request_index"), f"score boundary {position} request index", minimum=1
        )
        offset = _integer(row.get("offset"), f"score boundary {position} offset", minimum=0)
        coordinate = (request_index, offset)
        _require(coordinate not in seen_coordinates, f"duplicate score coordinate: {coordinate}")
        seen_coordinates.add(coordinate)
        _require(
            previous_coordinate is None or coordinate > previous_coordinate,
            "score boundaries must be in strict request/offset order",
        )
        previous_coordinate = coordinate
        fidelity = _mapping(
            row.get("numerical_fidelity"), f"score boundary {position} numerical fidelity"
        )
        _require(
            set(fidelity) == {"public_strict_adapter_pass", "native_strict_adapter_pass"}
            and all(isinstance(fidelity[key], bool) for key in fidelity),
            f"score boundary {position} numerical fidelity changed",
        )
        raw_scores = _mapping(
            row.get("source_concept_scores"), f"score boundary {position} source scores"
        )
        _require(set(raw_scores) == set(SCORE_SOURCES), "score source grid changed")
        result.append(
            {
                "boundary_id": boundary_id,
                "request_index": request_index,
                "offset": offset,
                "numerical_fidelity": dict(fidelity),
                "source_concept_scores": {
                    source_id: _score_map(
                        raw_scores[source_id],
                        f"score boundary {position}.{source_id}",
                    )
                    for source_id in SCORE_SOURCES
                },
            }
        )
    return result


def _ranking(scores: Mapping[str, float]) -> list[dict[str, Any]]:
    order = {concept_id: position for position, concept_id in enumerate(SCORABLE_CONCEPTS)}
    ranked_ids = sorted(
        SCORABLE_CONCEPTS,
        key=lambda concept_id: (-scores[concept_id], order[concept_id]),
    )
    return [
        {
            "rank": rank,
            "concept_id": concept_id,
            "display_phrase": CONCEPT_FAMILY_LEXICON[concept_id],
            "mean_logprob_score": scores[concept_id],
        }
        for rank, concept_id in enumerate(ranked_ids, 1)
    ]


def _top_template(row: Mapping[str, Any], label: str) -> dict[str, Any]:
    concept_id = _text(row.get("concept_id"), f"{label} concept ID")
    _require(concept_id in SCORABLE_CONCEPTS, f"{label} concept is not scorable")
    phrase = _text(row.get("display_phrase"), f"{label} phrase")
    _require(phrase == CONCEPT_FAMILY_LEXICON[concept_id], f"{label} phrase changed")
    return {
        "concept_id": concept_id,
        "display_phrase": phrase,
        "score": _finite(row.get("score"), f"{label} score"),
    }


def render_boundary_sentence(template_inputs: Mapping[str, Any]) -> str:
    values = _mapping(template_inputs, "boundary template inputs")
    _require(
        set(values)
        == {
            "request_index",
            "offset",
            "strict_fidelity",
            "public_top",
            "native_top",
            "ordinary_top",
            "public_top_two_margin",
            "native_top_two_margin",
            "selected_concept_id",
            "evidence_status",
        },
        "boundary template input fields changed",
    )
    request_index = _integer(values["request_index"], "template request index", minimum=1)
    offset = _integer(values["offset"], "template offset", minimum=0)
    _require(isinstance(values["strict_fidelity"], bool), "template fidelity must be boolean")
    public = _top_template(_mapping(values["public_top"], "public top"), "public top")
    native = _top_template(_mapping(values["native_top"], "native top"), "native top")
    ordinary = _top_template(_mapping(values["ordinary_top"], "ordinary top"), "ordinary top")
    public_margin = _finite(values["public_top_two_margin"], "public top-two margin")
    native_margin = _finite(values["native_top_two_margin"], "native top-two margin")
    _require(public_margin >= 0.0 and native_margin >= 0.0, "top-two margins must be nonnegative")
    evidence_status = _text(values["evidence_status"], "template evidence status")
    _require(
        evidence_status in {RENDERER["evidence_status"], "synthetic_or_unverified_test_input"},
        "template evidence status changed",
    )
    evidence = (
        "exploratory one-task, uncalibrated"
        if evidence_status == RENDERER["evidence_status"]
        else "synthetic or unverified test input"
    )
    coordinate = f"request {request_index}, offset {offset}"
    ordinary_clause = (
        f"The ordinary-logit ranking led with {ordinary['display_phrase']} "
        f"({ordinary['score']:.3f})."
    )
    selected = values["selected_concept_id"]
    _require(
        selected is None or selected in SCORABLE_CONCEPTS,
        "template selected concept is invalid",
    )
    if not values["strict_fidelity"]:
        _require(selected is None, "numerically ineligible boundary cannot select a concept")
        return (
            f"At {coordinate}, semantic decoding abstained because paired strict "
            "numerical fidelity did not pass. Public/native J diagnostics led with "
            f"{public['display_phrase']} ({public['score']:.3f}) and "
            f"{native['display_phrase']} ({native['score']:.3f}); no concept sentence "
            f"is licensed. {ordinary_clause} Evidence is {evidence}."
        )
    if public_margin <= 0.0 or native_margin <= 0.0:
        _require(selected is None, "non-unique source top one cannot select a concept")
        return (
            f"At {coordinate}, no semantic candidate is selected because at least "
            "one J source has a non-unique top-one score. Public/native leading "
            f"diagnostics are {public['display_phrase']} ({public['score']:.3f}; "
            f"margin {public_margin:.3f}) and {native['display_phrase']} "
            f"({native['score']:.3f}; margin {native_margin:.3f}). "
            f"{ordinary_clause} Evidence is {evidence}."
        )
    if public["concept_id"] != native["concept_id"]:
        _require(selected is None, "source disagreement cannot select a concept")
        return (
            f"At {coordinate}, no semantic candidate is selected: public J led with "
            f"{public['display_phrase']} ({public['score']:.3f}) while native J led "
            f"with {native['display_phrase']} ({native['score']:.3f}); source top-one "
            f"rankings disagreed. {ordinary_clause} Evidence is {evidence}."
        )
    _require(selected == public["concept_id"], "selected concept is not source-agreed top one")
    return (
        f"At {coordinate}, public and native J common-ontology rankings both led with "
        f"{public['display_phrase']} (scores {public['score']:.3f}/"
        f"{native['score']:.3f}; top-two margins {public_margin:.3f}/"
        f"{native_margin:.3f}). {ordinary_clause} This is only an uncalibrated "
        f"source-agreement candidate; evidence is {evidence}."
    )


def render_chain_sentence(template_inputs: Mapping[str, Any]) -> str:
    values = _mapping(template_inputs, "chain template inputs")
    _require(
        set(values) == {"groups", "evidence_status"},
        "chain template input fields changed",
    )
    _require(
        values["evidence_status"] in {
            RENDERER["evidence_status"],
            "synthetic_or_unverified_test_input",
        },
        "chain evidence status changed",
    )
    evidence = (
        "exploratory one-task, uncalibrated"
        if values["evidence_status"] == RENDERER["evidence_status"]
        else "synthetic or unverified test input"
    )
    groups = list(_sequence(values["groups"], "chain groups"))
    if not groups:
        return f"No source-agreement candidate sequence is emitted ({evidence})."
    clauses: list[str] = []
    previous_coordinate: tuple[int, int] | None = None
    for position, raw in enumerate(groups):
        group = _mapping(raw, f"chain group {position}")
        _require(
            set(group)
            == {
                "request_index",
                "offset",
                "concept_id",
                "display_phrase",
                "public_j_score",
                "native_j_score",
                "ordinary_top_concept_id",
                "ordinary_top_display_phrase",
                "intervening_abstention_count",
            },
            f"chain group {position} fields changed",
        )
        request_index = _integer(
            group["request_index"], f"chain group {position} request", minimum=1
        )
        offset = _integer(group["offset"], f"chain group {position} offset", minimum=0)
        coordinate = (request_index, offset)
        _require(
            previous_coordinate is None or coordinate > previous_coordinate,
            "chain groups are not strictly ordered",
        )
        previous_coordinate = coordinate
        concept_id = _text(group["concept_id"], f"chain group {position} concept")
        ordinary_id = _text(
            group["ordinary_top_concept_id"], f"chain group {position} ordinary concept"
        )
        _require(
            concept_id in SCORABLE_CONCEPTS and ordinary_id in SCORABLE_CONCEPTS,
            "chain group contains an unscorable concept",
        )
        phrase = _text(group["display_phrase"], f"chain group {position} phrase")
        ordinary_phrase = _text(
            group["ordinary_top_display_phrase"],
            f"chain group {position} ordinary phrase",
        )
        _require(phrase == CONCEPT_FAMILY_LEXICON[concept_id], "chain group phrase changed")
        _require(
            ordinary_phrase == CONCEPT_FAMILY_LEXICON[ordinary_id],
            "chain ordinary phrase changed",
        )
        public_score = _finite(group["public_j_score"], "chain public J score")
        native_score = _finite(group["native_j_score"], "chain native J score")
        gap = _integer(
            group["intervening_abstention_count"],
            f"chain group {position} abstention gap",
            minimum=0,
        )
        prefix = ""
        if position > 0:
            noun = "boundary" if gap == 1 else "boundaries"
            prefix = f"after {gap} abstaining registered {noun}, "
        baseline = "same concept family" if ordinary_id == concept_id else ordinary_phrase
        clauses.append(
            f"{prefix}request {request_index}, offset {offset} — {phrase} "
            f"(J scores {public_score:.3f}/{native_score:.3f}; ordinary top: {baseline})"
        )
    return (
        f"Sparse source-agreement candidate sequence ({evidence}): "
        + "; ".join(clauses)
        + "."
    )


def _normalize_evaluation_labels(
    value: Any,
    *,
    boundary_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    source = _mapping(value, "evaluation labels by boundary")
    _require(
        set(source) == set(boundary_ids),
        "evaluation label boundary grid differs from inference boundary grid",
    )
    result: dict[str, dict[str, Any]] = {}
    for boundary_id in boundary_ids:
        row = _mapping(source[boundary_id], f"evaluation label {boundary_id}")
        _require(
            set(row) == {"boundary_id", "positive_concept_ids", "leakage_class"},
            f"evaluation label {boundary_id} fields changed",
        )
        _require(row["boundary_id"] == boundary_id, "evaluation boundary identity changed")
        positives = [
            _text(item, f"evaluation label {boundary_id} positive concept")
            for item in _sequence(
                row["positive_concept_ids"],
                f"evaluation label {boundary_id} positives",
            )
        ]
        _require(bool(positives), f"evaluation label {boundary_id} needs positives")
        _require(
            len(positives) == len(set(positives))
            and all(concept_id in REGISTERED_CONCEPTS for concept_id in positives),
            f"evaluation label {boundary_id} positives are invalid",
        )
        leakage = _text(row["leakage_class"], f"evaluation label {boundary_id} leakage")
        _require(leakage in LEAKAGE_CLASSES, "evaluation leakage class is unknown")
        result[boundary_id] = {
            "boundary_id": boundary_id,
            "positive_concept_ids": positives,
            "leakage_class": leakage,
        }
    return result


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _evaluate(
    inference_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not labels:
        return {
            "status": "unavailable_no_separate_evaluation_labels",
            "labels_used_for_inference_or_rendering": False,
            "boundary_rows": [],
            "aggregate": None,
        }
    rows: list[dict[str, Any]] = []
    strict_count = 0
    selected_count = 0
    selected_correct_count = 0
    public_correct = 0
    native_correct = 0
    ordinary_correct = 0
    non_prefix_count = 0
    for inference in inference_rows:
        boundary_id = str(inference["boundary_id"])
        label = labels[boundary_id]
        positives = set(label["positive_concept_ids"])
        strict = bool(inference["strict_fidelity"])
        selected = inference["selected_concept_id"]
        if strict:
            strict_count += 1
            public_correct += int(inference["public_top_concept_id"] in positives)
            native_correct += int(inference["native_top_concept_id"] in positives)
            ordinary_correct += int(inference["ordinary_top_concept_id"] in positives)
        if selected is not None:
            selected_count += 1
            selected_correct_count += int(selected in positives)
        # Every frozen class is visible-prefix explicit/implicit or a lexical control.
        hidden_from_visible_prefix = label["leakage_class"] not in LEAKAGE_CLASSES
        non_prefix_count += int(hidden_from_visible_prefix)
        rows.append(
            {
                "boundary_id_sha256": inference["boundary_id_sha256"],
                "request_index": inference["request_index"],
                "offset": inference["offset"],
                "strict_fidelity": strict,
                "selected_concept_id": selected,
                "selected_matches_registered_positive": (
                    None if selected is None else selected in positives
                ),
                "public_j_top1_matches_registered_positive": (
                    None
                    if not strict
                    else inference["public_top_concept_id"] in positives
                ),
                "native_j_top1_matches_registered_positive": (
                    None
                    if not strict
                    else inference["native_top_concept_id"] in positives
                ),
                "ordinary_logit_top1_matches_registered_positive": (
                    None
                    if not strict
                    else inference["ordinary_top_concept_id"] in positives
                ),
                "positive_concept_ids": list(label["positive_concept_ids"]),
                "leakage_class": label["leakage_class"],
                "concept_absent_from_jsonl_visible_prefix": hidden_from_visible_prefix,
            }
        )
    boundary_count = len(rows)
    return {
        "status": "descriptive_one_task_retrospective_labels_only",
        "labels_used_for_inference_or_rendering": False,
        "boundary_rows": rows,
        "aggregate": {
            "boundary_count": boundary_count,
            "strict_fidelity_boundary_count": strict_count,
            "source_agreement_selected_count": selected_count,
            "source_agreement_correct_count": selected_correct_count,
            "coverage_over_all_boundaries": _safe_fraction(selected_count, boundary_count),
            "coverage_over_strict_fidelity_boundaries": _safe_fraction(
                selected_count, strict_count
            ),
            "selected_accuracy": _safe_fraction(
                selected_correct_count, selected_count
            ),
            "public_j_top1_accuracy_on_strict_boundaries": _safe_fraction(
                public_correct, strict_count
            ),
            "native_j_top1_accuracy_on_strict_boundaries": _safe_fraction(
                native_correct, strict_count
            ),
            "ordinary_logit_top1_accuracy_on_strict_boundaries": _safe_fraction(
                ordinary_correct, strict_count
            ),
            "non_prefix_explicit_boundary_count": non_prefix_count,
            "reliability_status": "not_established_single_task_uncalibrated",
            "incremental_j_value_established": False,
            "confidence_intervals": None,
        },
    }


def _build_concept_chain(
    score_boundaries: Any,
    *,
    trajectory_id: str,
    evaluation_labels_by_boundary: Any,
    protocol: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    contract = validate_protocol(protocol)
    trajectory = _text(trajectory_id, "trajectory ID")
    boundaries = _normalize_boundaries(score_boundaries)
    labels = _normalize_evaluation_labels(
        evaluation_labels_by_boundary,
        boundary_ids=[row["boundary_id"] for row in boundaries],
    )
    binding = dict(_mapping(source_binding, "source binding"))
    authenticated = binding.get("kind") == "authenticated_common_ontology_reports"
    expected_binding = {
        "kind": "authenticated_common_ontology_reports",
        "input_authenticated": True,
        "analysis_sha256": ANALYSIS_SHA256,
        "probe_config_sha256": PROBE_CONFIG_SHA256,
        "public_report_sha256": PUBLIC_REPORT_SHA256,
        "native_report_sha256": NATIVE_REPORT_SHA256,
        "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
        "trajectory_bundle_sha256": TRAJECTORY_BUNDLE_SHA256,
        "human_labels_structurally_separate": True,
    }
    if authenticated:
        _require(binding == expected_binding, "authenticated source binding is invalid")
        evidence_status = RENDERER["evidence_status"]
    else:
        _require(
            binding
            == {
                "kind": "synthetic_or_unverified_test_input",
                "input_authenticated": False,
            },
            "synthetic source binding is invalid",
        )
        evidence_status = "synthetic_or_unverified_test_input"

    trajectory_hash = _sha256_text(trajectory)
    output_boundaries: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    previous_selected: dict[str, Any] | None = None
    abstentions_since_selected = 0

    for boundary in boundaries:
        boundary_hash = _sha256_text(
            f"{trajectory_hash}:{boundary['boundary_id']}:{boundary['request_index']}:{boundary['offset']}"
        )
        rankings = {
            source_id: _ranking(boundary["source_concept_scores"][source_id])
            for source_id in SCORE_SOURCES
        }
        public_top = rankings["public_j"][0]
        native_top = rankings["native_j"][0]
        ordinary_top = rankings["ordinary_logit"][0]
        strict_fidelity = (
            boundary["numerical_fidelity"]["public_strict_adapter_pass"]
            and boundary["numerical_fidelity"]["native_strict_adapter_pass"]
        )
        sources_agree = public_top["concept_id"] == native_top["concept_id"]
        public_margin = (
            public_top["mean_logprob_score"]
            - rankings["public_j"][1]["mean_logprob_score"]
        )
        native_margin = (
            native_top["mean_logprob_score"]
            - rankings["native_j"][1]["mean_logprob_score"]
        )
        public_top_unique = public_margin > 0.0
        native_top_unique = native_margin > 0.0
        selected = (
            public_top["concept_id"]
            if strict_fidelity
            and sources_agree
            and public_top_unique
            and native_top_unique
            else None
        )
        template_inputs = {
            "request_index": boundary["request_index"],
            "offset": boundary["offset"],
            "strict_fidelity": strict_fidelity,
            "public_top": {
                "concept_id": public_top["concept_id"],
                "display_phrase": public_top["display_phrase"],
                "score": public_top["mean_logprob_score"],
            },
            "native_top": {
                "concept_id": native_top["concept_id"],
                "display_phrase": native_top["display_phrase"],
                "score": native_top["mean_logprob_score"],
            },
            "ordinary_top": {
                "concept_id": ordinary_top["concept_id"],
                "display_phrase": ordinary_top["display_phrase"],
                "score": ordinary_top["mean_logprob_score"],
            },
            "public_top_two_margin": public_margin,
            "native_top_two_margin": native_margin,
            "selected_concept_id": selected,
            "evidence_status": evidence_status,
        }
        sentence = render_boundary_sentence(template_inputs)
        if selected is None:
            abstentions_since_selected += 1
            abstention_reason = (
                "paired_strict_numerical_fidelity_failed"
                if not strict_fidelity
                else (
                    "non_unique_source_top1"
                    if not public_top_unique or not native_top_unique
                    else "public_native_j_top1_disagreement"
                )
            )
        else:
            group = {
                "request_index": boundary["request_index"],
                "offset": boundary["offset"],
                "concept_id": selected,
                "display_phrase": CONCEPT_FAMILY_LEXICON[selected],
                "public_j_score": public_top["mean_logprob_score"],
                "native_j_score": native_top["mean_logprob_score"],
                "ordinary_top_concept_id": ordinary_top["concept_id"],
                "ordinary_top_display_phrase": ordinary_top["display_phrase"],
                "intervening_abstention_count": (
                    0 if previous_selected is None else abstentions_since_selected
                ),
            }
            groups.append(group)
            abstention_reason = None
            if previous_selected is not None:
                relation_id = _canonical_sha256(
                    {
                        "relation": "later_source_agreement_candidate",
                        "from": previous_selected["boundary_id_sha256"],
                        "to": boundary_hash,
                    }
                )
                relations.append(
                    {
                        "relation_id": relation_id,
                        "relation": "later_source_agreement_candidate",
                        "from_boundary_id_sha256": previous_selected[
                            "boundary_id_sha256"
                        ],
                        "to_boundary_id_sha256": boundary_hash,
                        "from_concept_id": previous_selected["concept_id"],
                        "to_concept_id": selected,
                        "request_delta": boundary["request_index"]
                        - previous_selected["request_index"],
                        "offset_delta_within_same_request": (
                            boundary["offset"] - previous_selected["offset"]
                            if boundary["request_index"]
                            == previous_selected["request_index"]
                            else None
                        ),
                        "intervening_abstention_count": abstentions_since_selected,
                        "causal_claim": False,
                        "backdated": False,
                    }
                )
            previous_selected = {
                "boundary_id_sha256": boundary_hash,
                "concept_id": selected,
                "request_index": boundary["request_index"],
                "offset": boundary["offset"],
            }
            abstentions_since_selected = 0

        output_boundaries.append(
            {
                "boundary": {
                    "boundary_id_sha256": boundary_hash,
                    "trajectory_id_sha256": trajectory_hash,
                    "request_index": boundary["request_index"],
                    "offset": boundary["offset"],
                    "coordinate_kind": COORDINATE_KIND,
                    "causal_cutoff": CAUSAL_CUTOFF,
                },
                "numerical_fidelity": {
                    **boundary["numerical_fidelity"],
                    "paired_strict_adapter_pass": strict_fidelity,
                },
                "candidate_rankings": {
                    source_id: {
                        "top_k": rankings[source_id][: SCORING["top_k_diagnostics"]],
                        "full_ranking": rankings[source_id],
                        "score_kind": SCORING["concept_score"],
                        "score_is_probability": False,
                        "score_is_confidence": False,
                    }
                    for source_id in SCORE_SOURCES
                },
                "selection": {
                    "selected_concept_id": selected,
                    "selected_display_phrase": (
                        None if selected is None else CONCEPT_FAMILY_LEXICON[selected]
                    ),
                    "status": (
                        "source_agreement_candidate_uncalibrated"
                        if selected is not None
                        else "abstained"
                    ),
                    "abstention_reason": abstention_reason,
                    "calibrated": False,
                    "concept_probability": None,
                    "confidence": None,
                    "public_native_top1_agree": sources_agree,
                    "public_top1_unique": public_top_unique,
                    "native_top1_unique": native_top_unique,
                    "ordinary_logit_used_for_selection": False,
                },
                "online_summary": {
                    "template_id": RENDERER["boundary_template_id"],
                    "text": sentence,
                    "text_sha256": _sha256_text(sentence),
                    "template_inputs": template_inputs,
                    "evaluation_labels_used": False,
                    "input_paths": [
                        "boundary.request_index",
                        "boundary.offset",
                        "numerical_fidelity.paired_strict_adapter_pass",
                        "candidate_rankings.public_j.top_k",
                        "candidate_rankings.native_j.top_k",
                        "candidate_rankings.ordinary_logit.top_k",
                        "provenance.evidence_level",
                    ],
                },
            }
        )
        inference_rows.append(
            {
                "boundary_id": boundary["boundary_id"],
                "boundary_id_sha256": boundary_hash,
                "request_index": boundary["request_index"],
                "offset": boundary["offset"],
                "strict_fidelity": strict_fidelity,
                "selected_concept_id": selected,
                "public_top_concept_id": public_top["concept_id"],
                "native_top_concept_id": native_top["concept_id"],
                "ordinary_top_concept_id": ordinary_top["concept_id"],
            }
        )

    chain_template_inputs = {
        "groups": groups,
        "evidence_status": evidence_status,
    }
    chain_sentence = render_chain_sentence(chain_template_inputs)
    evaluation = _evaluate(inference_rows, labels)
    canonical_sentences = [
        *(row["online_summary"]["text"] for row in output_boundaries),
        chain_sentence,
        RENDERER["teacher_forcing_disclaimer"],
        RENDERER["retrospective_selection_disclaimer"],
        RENDERER["mandatory_disclaimer"],
    ]
    canonical_text = " ".join(canonical_sentences)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": OUTPUT_KIND,
        "claim_scope": dict(CLAIM_SCOPE),
        "trajectory": {
            "trajectory_id_sha256": trajectory_hash,
            "coordinate_kind": COORDINATE_KIND,
            "teacher_forced_replay": True,
            "generation_time_hidden_states": False,
        },
        "ontology": {
            "registered_concepts": list(REGISTERED_CONCEPTS),
            "scorable_concepts": list(SCORABLE_CONCEPTS),
            "excluded_concepts": dict(EXCLUDED_CONCEPTS),
            "common_candidate_grid_at_every_boundary": True,
            "shared_scoring_token_ids_between_scorable_concepts": False,
            "scoring": dict(SCORING),
            "selected_from_completed_visible_trace": True,
        },
        "boundaries": output_boundaries,
        "relations": relations,
        "concept_chain": {
            "kind": "sparse_uncalibrated_source_agreement_candidate_sequence",
            "groups": groups,
            "registered_boundary_count": len(output_boundaries),
            "strict_fidelity_boundary_count": sum(
                row["strict_fidelity"] for row in inference_rows
            ),
            "selected_boundary_count": len(groups),
            "abstained_boundary_count": len(output_boundaries) - len(groups),
            "operationally_reliable": False,
            "concepts_absent_from_jsonl_visible_prefix_established": False,
            "causal_relations_inferred": False,
        },
        "rendering": {
            "renderer_id": RENDERER["renderer_id"],
            "chain_template_id": RENDERER["chain_template_id"],
            "chain_sentence": chain_sentence,
            "chain_sentence_sha256": _sha256_text(chain_sentence),
            "chain_template_inputs": chain_template_inputs,
            "teacher_forcing_disclaimer": RENDERER["teacher_forcing_disclaimer"],
            "retrospective_selection_disclaimer": RENDERER[
                "retrospective_selection_disclaimer"
            ],
            "mandatory_disclaimer": RENDERER["mandatory_disclaimer"],
            "canonical_text": canonical_text,
            "canonical_text_sha256": _sha256_text(canonical_text),
        },
        "evaluation": evaluation,
        "unavailable_claims": {
            "private_chain_of_thought": {
                "value": None,
                "status": "unsupported_not_observable",
            },
            "emotion_or_stress": {
                "value": None,
                "status": "unsupported_no_validated_subjective_target",
            },
            "hidden_content_absent_from_jsonl": {
                "value": None,
                "status": "unsupported_no_non_prefix_explicit_evaluation_stratum",
            },
            "raw_activation_decoded_concepts": {
                "value": None,
                "status": "unavailable_no_hash_authenticated_raw_residual_or_j_state_artifact",
            },
            "arbitrary_phrase_or_proposition_decoder": {
                "value": None,
                "status": "unavailable_no_compositional_decoder",
            },
        },
        "provenance": {
            "protocol_id": PROTOCOL_ID,
            "protocol_canonical_sha256": _canonical_sha256(contract),
            "source_binding": binding,
            "input_authenticated": authenticated,
            "evidence_level": evidence_status,
            "grounding": "common_ontology_vocabulary_readout",
            "human_positive_labels_used_for_inference": False,
            "human_positive_labels_used_for_evaluation_only": bool(labels),
            "ontology_selected_from_completed_visible_trace": True,
            "coordinates_selected_from_completed_visible_trace": True,
            "fresh_transport_confirmation": False,
            "operational_reliability_claim": False,
            "reserved_validation_opened": False,
        },
        "limitations": list(LIMITATIONS),
    }


def build_concept_chain(
    score_boundaries: Any,
    *,
    trajectory_id: str,
    evaluation_labels_by_boundary: Any = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a synthetic/unverified chain with labels structurally separated."""

    return _build_concept_chain(
        score_boundaries,
        trajectory_id=trajectory_id,
        evaluation_labels_by_boundary=evaluation_labels_by_boundary,
        protocol=protocol if protocol is not None else load_protocol(),
        source_binding={
            "kind": "synthetic_or_unverified_test_input",
            "input_authenticated": False,
        },
    )


def _read_authenticated_json(
    path: Path,
    *,
    expected_path: Path,
    expected_sha256: str,
    label: str,
) -> Mapping[str, Any]:
    expanded = path.expanduser()
    _require(
        expanded.is_file() and not expanded.is_symlink(),
        f"{label} must be a regular non-symlink file",
    )
    resolved = expanded.resolve(strict=True)
    _require(
        resolved == expected_path.resolve(strict=True),
        f"{label} path differs from the authenticated contract",
    )
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read {label}") from error
    _require(_sha256_bytes(payload) == expected_sha256, f"{label} SHA-256 changed")
    try:
        return _mapping(json.loads(payload), label)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _validate_analysis(value: Mapping[str, Any]) -> None:
    _require(value.get("schema_version") == 1, "analysis schema changed")
    _require(value.get("kind") == ANALYSIS_KIND, "analysis kind changed")
    source = _mapping(value.get("source_bindings"), "analysis source bindings")
    _require(
        dict(source)
        == {
            "config_sha256": PROBE_CONFIG_SHA256,
            "trajectory_bundle_sha256": TRAJECTORY_BUNDLE_SHA256,
            "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
        },
        "analysis source bindings changed",
    )
    inputs = _mapping(value.get("inputs"), "analysis inputs")
    for key, expected_sha256 in (
        ("config", PROBE_CONFIG_SHA256),
        ("public_report", PUBLIC_REPORT_SHA256),
        ("native_report", NATIVE_REPORT_SHA256),
        ("prompts", PROMPT_BUNDLE_SHA256),
    ):
        row = _mapping(inputs.get(key), f"analysis input {key}")
        _require(row.get("sha256") == expected_sha256, f"analysis input {key} hash changed")
    pairing = _mapping(value.get("pairing"), "analysis pairing")
    _require(
        pairing.get("exact_prompt_residual_and_logit_pairing") is True
        and pairing.get("item_count") == 10,
        "analysis pairing changed",
    )
    evaluation = _mapping(value.get("evaluation"), "analysis evaluation")
    _require(
        evaluation.get("middle_band_layers") == list(LAYERS)
        and evaluation.get("accepted_target_token_scored") is False
        and evaluation.get("claims_gate_preregistered") is False,
        "analysis evaluation scope changed",
    )
    for source_name in ("public", "native"):
        source_result = _mapping(value.get(source_name), f"analysis {source_name}")
        eligibility = _mapping(
            source_result.get("numerical_eligibility"),
            f"analysis {source_name} numerical eligibility",
        )
        counts = _mapping(eligibility.get("counts"), "analysis eligibility counts")
        _require(
            eligibility.get("experiment_count") == 10
            and eligibility.get("strict_report_status") == "failed"
            and counts.get("strict_adapter_pass") == 5,
            f"analysis {source_name} numerical eligibility changed",
        )


def _registry_and_labels(
    probe_config: Mapping[str, Any],
) -> tuple[
    dict[str, tuple[int, ...]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    str,
    set[int],
]:
    _require(probe_config.get("schema_version") == 1, "probe config schema changed")
    _require(
        probe_config.get("kind") == "swe_verified_intermediate_concept_eval",
        "probe config kind changed",
    )
    adaptation = _mapping(probe_config.get("adaptation"), "probe adaptation")
    _require(
        adaptation.get("lens_outputs_used_for_selection") is False,
        "probe concept selection used lens outputs",
    )
    source = _mapping(probe_config.get("source"), "probe source")
    _require(
        source.get("trajectory_bundle_sha256") == TRAJECTORY_BUNDLE_SHA256,
        "probe trajectory binding changed",
    )
    middle_band = _mapping(probe_config.get("middle_band"), "probe middle band")
    _require(middle_band.get("layers") == list(LAYERS), "probe layer grid changed")
    task = _mapping(probe_config.get("task"), "probe task")
    trajectory_id = _text(task.get("instance_id"), "probe task instance ID")
    items = list(_sequence(probe_config.get("items"), "probe items"))
    _require(len(items) == 10, "probe boundary count changed")

    forms_by_concept: dict[str, tuple[int, ...]] = {}
    boundaries: list[dict[str, Any]] = []
    labels: dict[str, dict[str, Any]] = {}
    previous_coordinate: tuple[int, int] | None = None
    for position, raw_item in enumerate(items):
        item = _mapping(raw_item, f"probe item {position}")
        boundary_id = _text(item.get("id"), f"probe item {position} ID")
        request_index = _integer(
            item.get("request_index"), f"probe item {position} request", minimum=1
        )
        offset = _integer(item.get("offset"), f"probe item {position} offset", minimum=0)
        coordinate = (request_index, offset)
        _require(
            previous_coordinate is None or coordinate > previous_coordinate,
            "probe items are not strictly ordered",
        )
        previous_coordinate = coordinate
        leakage = _text(item.get("leakage_class"), f"probe item {position} leakage")
        _require(leakage in LEAKAGE_CLASSES, "probe leakage class changed")
        positives: list[str] = []
        for concept_position, raw_concept in enumerate(
            _sequence(item.get("intermediates"), f"probe item {position} concepts")
        ):
            concept = _mapping(
                raw_concept, f"probe item {position} concept {concept_position}"
            )
            concept_id = _text(
                concept.get("key"), f"probe item {position} concept {concept_position} ID"
            )
            _require(concept_id in REGISTERED_CONCEPTS, "probe concept is unregistered")
            _require(concept_id not in forms_by_concept, f"duplicate probe concept {concept_id}")
            token_ids: list[int] = []
            for form_position, raw_form in enumerate(
                _sequence(concept.get("forms"), f"probe concept {concept_id} forms")
            ):
                form = _mapping(raw_form, f"probe concept {concept_id} form {form_position}")
                token_id = _integer(
                    form.get("token_id"),
                    f"probe concept {concept_id} form token ID",
                    minimum=0,
                )
                token_ids.append(token_id)
            _require(
                bool(token_ids) and len(token_ids) == len(set(token_ids)),
                f"probe concept {concept_id} token forms are invalid",
            )
            forms_by_concept[concept_id] = tuple(token_ids)
            positives.append(concept_id)
        boundaries.append(
            {
                "boundary_id": boundary_id,
                "request_index": request_index,
                "offset": offset,
            }
        )
        labels[boundary_id] = {
            "boundary_id": boundary_id,
            "positive_concept_ids": positives,
            "leakage_class": leakage,
        }
    _require(
        tuple(forms_by_concept) == REGISTERED_CONCEPTS,
        "probe registered concept order changed",
    )
    membership = Counter(
        token_id for token_ids in forms_by_concept.values() for token_id in token_ids
    )
    unique_forms = {
        concept_id: tuple(
            token_id for token_id in forms_by_concept[concept_id] if membership[token_id] == 1
        )
        for concept_id in REGISTERED_CONCEPTS
    }
    observed_scorable = tuple(
        concept_id
        for concept_id in REGISTERED_CONCEPTS
        if len(unique_forms[concept_id]) >= 2
    )
    _require(observed_scorable == SCORABLE_CONCEPTS, "scorable concept registry changed")
    observed_excluded = {
        concept_id: (
            "only_one_globally_unique_registered_token_form"
            if len(unique_forms[concept_id]) == 1
            else "no_globally_unique_registered_token_form"
        )
        for concept_id in REGISTERED_CONCEPTS
        if len(unique_forms[concept_id]) < 2
    }
    _require(observed_excluded == EXCLUDED_CONCEPTS, "excluded concept registry changed")
    all_token_ids = set(membership)
    _require(len(all_token_ids) == 58, "global scored token vocabulary changed")
    return unique_forms, boundaries, labels, trajectory_id, all_token_ids


def _strict_adapter_pass(experiment: Mapping[str, Any], label: str) -> bool:
    top1 = experiment.get("final_layer_top1_matches_greedy")
    final_norm = _mapping(
        experiment.get("final_norm_reconstruction"), f"{label} final norm"
    )
    final_logits = _mapping(
        experiment.get("final_logits_reconstruction"), f"{label} final logits"
    )
    norm_within = final_norm.get("within_tolerance")
    logits_within = final_logits.get("within_tolerance")
    _require(
        all(isinstance(value, bool) for value in (top1, norm_within, logits_within)),
        f"{label} numerical fidelity flags changed",
    )
    return bool(top1 and norm_within and logits_within)


def _logprob_map(value: Any, *, token_ids: set[int], label: str) -> dict[int, float]:
    readout = _mapping(value, label)
    rows = list(_sequence(readout.get("scored_tokens"), f"{label} scored tokens"))
    result: dict[int, float] = {}
    for position, raw in enumerate(rows):
        row = _mapping(raw, f"{label} scored token {position}")
        token_id = _integer(row.get("token_id"), f"{label} token ID", minimum=0)
        _require(token_id not in result, f"{label} duplicate token ID")
        result[token_id] = _finite(row.get("logprob"), f"{label} token logprob")
    _require(set(result) == token_ids, f"{label} scored token grid changed")
    return result


def _report_scores(
    report: Mapping[str, Any],
    *,
    report_label: str,
    boundaries: Sequence[Mapping[str, Any]],
    unique_forms: Mapping[str, Sequence[int]],
    all_token_ids: set[int],
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[bool]]:
    _require(report.get("schema_version") == 3, f"{report_label} report schema changed")
    _require(
        report.get("score_encoding") == "unrounded-float32",
        f"{report_label} score encoding changed",
    )
    _require(report.get("status") == "failed", f"{report_label} report status changed")
    vocabulary = _mapping(report.get("scored_vocabulary"), f"{report_label} vocabulary")
    _require(
        set(vocabulary.get("token_ids", [])) == all_token_ids,
        f"{report_label} global vocabulary changed",
    )
    experiments = list(_sequence(report.get("experiments"), f"{report_label} experiments"))
    _require(len(experiments) == len(boundaries), f"{report_label} experiment count changed")
    j_scores: list[dict[str, float]] = []
    logit_scores: list[dict[str, float]] = []
    strict_flags: list[bool] = []
    for position, (raw_experiment, boundary) in enumerate(
        zip(experiments, boundaries, strict=True)
    ):
        experiment = _mapping(raw_experiment, f"{report_label} experiment {position}")
        expected_id = f"swe-intermediate-{boundary['boundary_id']}"
        _require(experiment.get("id") == expected_id, f"{report_label} experiment ID changed")
        layers = list(_sequence(experiment.get("layers"), f"{report_label} layers"))
        layer_ids = [
            _mapping(layer, f"{report_label} layer").get("layer") for layer in layers
        ]
        _require(layer_ids == list(LAYERS), f"{report_label} layer grid changed")
        accumulated = {
            "jacobian_lens": {concept_id: [] for concept_id in SCORABLE_CONCEPTS},
            "logit_lens": {concept_id: [] for concept_id in SCORABLE_CONCEPTS},
        }
        for layer_position, raw_layer in enumerate(layers):
            layer = _mapping(raw_layer, f"{report_label} layer {layer_position}")
            positions = list(
                _sequence(layer.get("positions"), f"{report_label} layer positions")
            )
            _require(len(positions) == 1, f"{report_label} layer position count changed")
            readout = _mapping(positions[0], f"{report_label} position")
            for method in ("jacobian_lens", "logit_lens"):
                token_logprobs = _logprob_map(
                    readout.get(method),
                    token_ids=all_token_ids,
                    label=f"{report_label} {method} layer {layer_ids[layer_position]}",
                )
                for concept_id in SCORABLE_CONCEPTS:
                    accumulated[method][concept_id].extend(
                        token_logprobs[token_id]
                        for token_id in unique_forms[concept_id]
                    )
        j_scores.append(
            {
                concept_id: math.fsum(accumulated["jacobian_lens"][concept_id])
                / len(accumulated["jacobian_lens"][concept_id])
                for concept_id in SCORABLE_CONCEPTS
            }
        )
        logit_scores.append(
            {
                concept_id: math.fsum(accumulated["logit_lens"][concept_id])
                / len(accumulated["logit_lens"][concept_id])
                for concept_id in SCORABLE_CONCEPTS
            }
        )
        strict_flags.append(
            _strict_adapter_pass(experiment, f"{report_label} experiment {position}")
        )
    return j_scores, logit_scores, strict_flags


def _extract_authenticated_inputs(
    analysis: Mapping[str, Any],
    probe_config: Mapping[str, Any],
    public_report: Mapping[str, Any],
    native_report: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    _validate_analysis(analysis)
    unique_forms, boundaries, labels, trajectory_id, all_token_ids = _registry_and_labels(
        probe_config
    )
    public_j, public_logit, public_strict = _report_scores(
        public_report,
        report_label="public",
        boundaries=boundaries,
        unique_forms=unique_forms,
        all_token_ids=all_token_ids,
    )
    native_j, native_logit, native_strict = _report_scores(
        native_report,
        report_label="native",
        boundaries=boundaries,
        unique_forms=unique_forms,
        all_token_ids=all_token_ids,
    )
    _require(public_strict == native_strict, "paired strict-fidelity flags differ")
    _require(sum(public_strict) == 5, "paired strict-fidelity count changed")
    score_boundaries: list[dict[str, Any]] = []
    for position, boundary in enumerate(boundaries):
        for concept_id in SCORABLE_CONCEPTS:
            _require(
                math.isclose(
                    public_logit[position][concept_id],
                    native_logit[position][concept_id],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"paired ordinary-logit score differs for {concept_id}",
            )
        score_boundaries.append(
            {
                **boundary,
                "numerical_fidelity": {
                    "public_strict_adapter_pass": public_strict[position],
                    "native_strict_adapter_pass": native_strict[position],
                },
                "source_concept_scores": {
                    "public_j": public_j[position],
                    "native_j": native_j[position],
                    "ordinary_logit": public_logit[position],
                },
            }
        )
    return score_boundaries, labels, trajectory_id


def build_concept_chain_from_intermediate_artifacts(
    analysis_path: Path = ANALYSIS_PATH,
    *,
    probe_config_path: Path = PROBE_CONFIG_PATH,
    public_report_path: Path = PUBLIC_REPORT_PATH,
    native_report_path: Path = NATIVE_REPORT_PATH,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = validate_protocol(protocol if protocol is not None else load_protocol())
    analysis = _read_authenticated_json(
        analysis_path,
        expected_path=ANALYSIS_PATH,
        expected_sha256=ANALYSIS_SHA256,
        label="intermediate analysis artifact",
    )
    probe_config = _read_authenticated_json(
        probe_config_path,
        expected_path=PROBE_CONFIG_PATH,
        expected_sha256=PROBE_CONFIG_SHA256,
        label="intermediate probe config",
    )
    public_report = _read_authenticated_json(
        public_report_path,
        expected_path=PUBLIC_REPORT_PATH,
        expected_sha256=PUBLIC_REPORT_SHA256,
        label="public intermediate report",
    )
    native_report = _read_authenticated_json(
        native_report_path,
        expected_path=NATIVE_REPORT_PATH,
        expected_sha256=NATIVE_REPORT_SHA256,
        label="native intermediate report",
    )
    score_boundaries, labels, trajectory_id = _extract_authenticated_inputs(
        analysis, probe_config, public_report, native_report
    )
    return _build_concept_chain(
        score_boundaries,
        trajectory_id=trajectory_id,
        evaluation_labels_by_boundary=labels,
        protocol=contract,
        source_binding={
            "kind": "authenticated_common_ontology_reports",
            "input_authenticated": True,
            "analysis_sha256": ANALYSIS_SHA256,
            "probe_config_sha256": PROBE_CONFIG_SHA256,
            "public_report_sha256": PUBLIC_REPORT_SHA256,
            "native_report_sha256": NATIVE_REPORT_SHA256,
            "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
            "trajectory_bundle_sha256": TRAJECTORY_BUNDLE_SHA256,
            "human_labels_structurally_separate": True,
        },
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS_PATH)
    parser.add_argument("--probe-config", type=Path, default=PROBE_CONFIG_PATH)
    parser.add_argument("--public-report", type=Path, default=PUBLIC_REPORT_PATH)
    parser.add_argument("--native-report", type=Path, default=NATIVE_REPORT_PATH)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    result = build_concept_chain_from_intermediate_artifacts(
        args.analysis,
        probe_config_path=args.probe_config,
        public_report_path=args.public_report,
        native_report_path=args.native_report,
        protocol=load_protocol(args.protocol),
    )
    output_path = args.output.expanduser().resolve()
    _atomic_write_json(output_path, result)
    chain = result["concept_chain"]
    evaluation = result["evaluation"]["aggregate"]
    print(
        f"wrote {chain['registered_boundary_count']} boundaries / "
        f"{chain['selected_boundary_count']} source-agreement candidates to {output_path}; "
        f"selected accuracy={evaluation['selected_accuracy']:.3f}, "
        f"coverage={evaluation['coverage_over_all_boundaries']:.3f}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ANALYSIS_PATH",
    "CLAIM_SCOPE",
    "CONCEPT_FAMILY_LEXICON",
    "DEFAULT_PROTOCOL",
    "EXCLUDED_CONCEPTS",
    "FORBIDDEN_FIELDS",
    "NATIVE_REPORT_PATH",
    "PROBE_CONFIG_PATH",
    "PROTOCOL_ID",
    "PROTOCOL_STATUS",
    "PUBLIC_REPORT_PATH",
    "REGISTERED_CONCEPTS",
    "RENDERER",
    "SCORABLE_CONCEPTS",
    "SCORING",
    "build_concept_chain",
    "build_concept_chain_from_intermediate_artifacts",
    "load_protocol",
    "render_boundary_sentence",
    "render_chain_sentence",
    "validate_protocol",
]
