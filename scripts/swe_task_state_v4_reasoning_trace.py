#!/usr/bin/env python3
"""Causal, deterministic COT-like trace sidecar for the V4 SWE interpreter.

The sidecar summarizes observable model outputs at final-prompt boundaries.  It
does not reconstruct private chain of thought, infer subjective emotion, or
read task text, current/future action labels, completion text, tool results, or
task outcomes.  Its three phase names are semantic aliases for V4's calibrated
next-action classes; transition events and diagnostic indices are explicitly
proxy measurements.

The public entry point deliberately accepts V4 prediction records as they are.
Those records also contain label-time metadata, but this module reads only the
allowlisted identity, probability, and abstention fields.  Optional diagnostics
must arrive in a separate, narrowly validated mapping.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/swe_task_state_v4_reasoning_trace.json"
DESIGN_ARTIFACT_RELATIVE_PATH = (
    ".cache/swe_state_interpreter_v4_design/"
    "v3-n60-geometric-a020-shared-action-contract-closed.json"
)
DESIGN_ARTIFACT_PATH = ROOT / DESIGN_ARTIFACT_RELATIVE_PATH
DESIGN_ARTIFACT_SHA256 = (
    "e1436a9c7fe763a4fab6ae505af17395a4290faefd0734b230d22127bff0cf8c"
)
DESIGN_ARTIFACT_ID = "swe-task-state-interpreter-v4-v3-n60-design-screen-v2"
DESIGN_PRIMARY_PROCEDURE = "j_forecast_geometric_pool_logit_policy"
DESIGN_SOURCE_PROCEDURES = {
    "sequence_logit_probabilities": "sequence_logit",
    "sequence_j_probabilities": "sequence_j",
}

SCHEMA_VERSION = 1
PROTOCOL_ID = "swe-task-state-v4-reasoning-trace-v1"
PROTOCOL_STATUS = "design_only_transport_unconfirmed"
CLASSES = ("inspect", "edit", "check_or_finish")
PHASE_BY_CLASS = {
    "inspect": "information_gathering",
    "edit": "implementation",
    "check_or_finish": "verification_or_completion",
}
TRACE_EVENTS = (
    "phase_observation",
    "phase_continuation",
    "inspection_to_implementation",
    "implementation_to_verification_or_completion",
    "reconsideration_or_rework_like",
    "phase_transition",
    "uncertain_transition",
    "recovery_like",
)
PREDICTION_INPUT_ALLOWLIST = (
    "row_id",
    "task_id",
    "task_request_index",
    "forecast_probabilities_q",
    "decision_probabilities_d",
    "accepted",
)
PREDICTION_CONSISTENCY_FIELDS = (
    "forecast_top_class",
    "forecast_top_confidence",
    "predicted_class",
    "decision_confidence_from_q",
)
DIAGNOSTIC_ALLOWLIST = (
    "sequence_logit_probabilities",
    "sequence_j_probabilities",
)
FORBIDDEN_INFERENCE_FIELDS = (
    "source_action_class_id",
    "label",
    "label_status",
    "metric_evaluable",
    "auxiliary_diagnostics",
    "reasoning",
    "reasoning_content",
    "current_completion",
    "current_tool_result",
    "current_usage",
    "finish_reason",
    "official_outcome",
    "later_actions",
    "task_text",
)
CLAIM_SCOPE = {
    "kind": "observable_activation_and_trajectory_proxy",
    "private_chain_of_thought_reconstructed": False,
    "subjective_emotion_inferred": False,
    "task_specific_intent_inferred": False,
    "task_text_used_as_readout_feature": False,
    "current_or_future_action_label_used": False,
    "task_outcome_used": False,
    "rationale_generation": "deterministic_template_from_emitted_numeric_fields",
}
UNFITTED_HEADS = {
    "hesitation_like_probability": {
        "target": "at_least_two_consecutive_future_noncommitment_boundaries",
        "status": "unavailable_unfitted_causal_head",
    },
    "trajectory_continuation_probability": {
        "target": (
            "at_least_two_further_known_boundaries_before_explicit_finalize_with_censoring"
        ),
        "status": "unavailable_unfitted_censoring_aware_head",
    },
    "recovery_within_2_probability": {
        "target": "return_to_accepted_low_load_action_regime_within_two_boundaries",
        "status": "unavailable_unfitted_causal_head",
    },
}
PROBABILITY_TOLERANCE = 1e-12
LOAD_COMPONENTS = (
    "diffuse_uncertainty",
    "forecast_volatility",
    "source_disagreement",
    "ordinary_activation_innovation",
    "public_jacobian_activation_innovation",
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


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, Real) and not isinstance(value, bool),
        f"{label} must be numeric and non-boolean",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _unit_interval(value: Any, label: str) -> float:
    result = _finite(value, label)
    _require(0.0 <= result <= 1.0, f"{label} must lie in [0, 1]")
    return result


def _nonempty_text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be nonempty text")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _probabilities(value: Any, label: str) -> dict[str, float]:
    source = _mapping(value, label)
    _require(set(source) == set(CLASSES), f"{label} must name the three V4 classes")
    result = {
        class_id: _finite(source[class_id], f"{label}.{class_id}")
        for class_id in CLASSES
    }
    _require(
        all(probability > 0.0 for probability in result.values()),
        f"{label} probabilities must be strictly positive",
    )
    total = math.fsum(result.values())
    _require(
        math.isclose(total, 1.0, rel_tol=PROBABILITY_TOLERANCE, abs_tol=PROBABILITY_TOLERANCE),
        f"{label} probabilities must sum to one",
    )
    return result


def build_reasoning_trace(
    predictions: Sequence[Mapping[str, Any]],
    *,
    diagnostics_by_row: Mapping[str, Mapping[str, Any]] | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a synthetic/unverified trace for tests and schema exploration.

    This entry point intentionally cannot emit authenticated or calibrated
    provenance.  Use :func:`build_reasoning_trace_from_v4_design_artifact` for
    the one exact hash-bound V4 design artifact currently supported.
    """

    return _build_reasoning_trace(
        predictions,
        diagnostics_by_row=diagnostics_by_row,
        protocol=protocol,
        source_binding={
            "kind": "synthetic_or_unverified_test_input",
            "input_authenticated": False,
        },
    )


def _artifact_sha256(value: Any, label: str) -> str:
    result = _nonempty_text(value, label)
    _require(
        len(result) == 64
        and all(character in "0123456789abcdef" for character in result),
        f"{label} must be lowercase SHA-256 text",
    )
    return result


def build_reasoning_trace_from_v4_design_artifact(
    path: Path = DESIGN_ARTIFACT_PATH,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Render the exact pinned V4 design artifact with authenticated provenance."""

    contract = validate_protocol(protocol if protocol is not None else load_protocol())
    expanded = path.expanduser()
    _require(
        expanded.is_file() and not expanded.is_symlink(),
        "V4 design artifact must be a regular non-symlink file",
    )
    resolved = expanded.resolve(strict=True)
    _require(
        resolved == DESIGN_ARTIFACT_PATH.resolve(strict=True),
        "V4 design artifact path differs from the authenticated contract",
    )
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise ValueError("could not read V4 design artifact") from error
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    _require(
        observed_sha256 == DESIGN_ARTIFACT_SHA256,
        "V4 design artifact SHA-256 changed",
    )
    try:
        artifact = _mapping(json.loads(payload), "V4 design artifact")
    except json.JSONDecodeError as error:
        raise ValueError("V4 design artifact is not valid JSON") from error
    _require(
        artifact.get("id") == DESIGN_ARTIFACT_ID
        and artifact.get("reserved_validation_accessed") is False
        and artifact.get("reserved_validation_allowed") is False
        and artifact.get("operational_reliability_claim") is False
        and artifact.get("independent_v4_development_result") is False
        and artifact.get("fresh_disjoint_nonreserved_development_confirmation_required")
        is True,
        "V4 design artifact scope or identity changed",
    )

    nested = _mapping(
        artifact.get("nested_design_evaluation"), "nested design evaluation"
    )
    results = _mapping(nested.get("results"), "nested design results")
    required_procedures = {
        DESIGN_PRIMARY_PROCEDURE,
        *DESIGN_SOURCE_PROCEDURES.values(),
    }
    _require(
        required_procedures.issubset(results),
        "V4 design artifact omits a bound trace procedure",
    )
    primary_result = _mapping(
        results[DESIGN_PRIMARY_PROCEDURE], "primary trace procedure result"
    )
    primary_predictions = list(
        _sequence(primary_result.get("predictions"), "primary predictions")
    )
    _require(bool(primary_predictions), "primary predictions must not be empty")

    prediction_fields = {
        "row_id",
        "task_id",
        "repo",
        "cohort_id",
        "task_request_index",
        "checkpoint_ordinal",
        "source_action_label_status",
        "source_action_class_id",
        "label_status",
        "label",
        "metric_evaluable",
        "auxiliary_diagnostics",
        "forecast_raw_probabilities_p",
        "decision_raw_probabilities_r",
        "forecast_probabilities_q",
        "decision_probabilities_d",
        "forecast_top_class",
        "forecast_top_confidence",
        "predicted_class",
        "decision_confidence_from_q",
        "accepted",
    }
    for position, raw in enumerate(primary_predictions):
        prediction_row = _mapping(raw, f"authenticated prediction {position}")
        _require(
            prediction_fields.issubset(prediction_row),
            f"authenticated prediction {position} schema is incomplete",
        )

    diagnostics_by_row: dict[str, dict[str, Any]] = {
        str(_mapping(row, "primary prediction")["row_id"]): {}
        for row in primary_predictions
    }
    primary_identity = [
        (
            str(_mapping(row, "primary prediction")["row_id"]),
            str(row["task_id"]),
            row["task_request_index"],
        )
        for row in primary_predictions
    ]
    for diagnostic_field, procedure in DESIGN_SOURCE_PROCEDURES.items():
        procedure_result = _mapping(results[procedure], f"{procedure} result")
        procedure_predictions = list(
            _sequence(
                procedure_result.get("predictions"), f"{procedure} predictions"
            )
        )
        procedure_identity = [
            (
                str(_mapping(row, f"{procedure} prediction")["row_id"]),
                str(row["task_id"]),
                row["task_request_index"],
            )
            for row in procedure_predictions
        ]
        _require(
            procedure_identity == primary_identity,
            f"{procedure} prediction identities differ from the primary branch",
        )
        for raw in procedure_predictions:
            diagnostic_prediction = _mapping(raw, f"{procedure} prediction")
            row_id = str(diagnostic_prediction["row_id"])
            diagnostics_by_row[row_id][diagnostic_field] = _probabilities(
                diagnostic_prediction.get("forecast_raw_probabilities_p"),
                f"{procedure} raw forecast probabilities",
            )

    source_binding = {
        "kind": "authenticated_v4_design_artifact",
        "input_authenticated": True,
        "artifact_id": DESIGN_ARTIFACT_ID,
        "artifact_sha256": observed_sha256,
        "primary_procedure": DESIGN_PRIMARY_PROCEDURE,
        "source_diagnostic_procedures": dict(DESIGN_SOURCE_PROCEDURES),
        "evaluator_contract_sha256": _artifact_sha256(
            nested.get("evaluator_contract_sha256"), "evaluator contract SHA-256"
        ),
        "shared_action_settings_sha256": _artifact_sha256(
            nested.get("full_development_shared_action_settings_sha256"),
            "shared action settings SHA-256",
        ),
        "forecast_q_float64_sha256": _artifact_sha256(
            primary_result.get("forecast_q_float64_sha256"),
            "primary forecast-q SHA-256",
        ),
        "decision_d_float64_sha256": _artifact_sha256(
            primary_result.get("decision_d_float64_sha256"),
            "primary decision-d SHA-256",
        ),
        "acceptance_sha256": _artifact_sha256(
            primary_result.get("acceptance_sha256"), "primary acceptance SHA-256"
        ),
    }
    return _build_reasoning_trace(
        primary_predictions,
        diagnostics_by_row=diagnostics_by_row,
        protocol=contract,
        source_binding=source_binding,
    )


def _argmax_class(probabilities: Mapping[str, float]) -> str:
    # Tuple order provides deterministic tie breaking.
    return max(CLASSES, key=lambda class_id: probabilities[class_id])


def normalized_entropy(probabilities: Mapping[str, float]) -> float:
    """Return three-class Shannon entropy normalized to [0, 1]."""

    values = _probabilities(probabilities, "entropy probabilities")
    result = -math.fsum(value * math.log(value) for value in values.values()) / math.log(
        len(CLASSES)
    )
    return min(1.0, max(0.0, result))


def ambivalence_index(probabilities: Mapping[str, float]) -> float:
    """Return top-two closeness; one means an exact top-two tie."""

    values = sorted(
        _probabilities(probabilities, "ambivalence probabilities").values(),
        reverse=True,
    )
    result = 1.0 - (values[0] - values[1]) / (values[0] + values[1])
    return min(1.0, max(0.0, result))


def normalized_js_divergence(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    """Return Jensen-Shannon divergence normalized by its log(2) maximum."""

    first = _probabilities(left, "left divergence probabilities")
    second = _probabilities(right, "right divergence probabilities")
    midpoint = {
        class_id: 0.5 * (first[class_id] + second[class_id])
        for class_id in CLASSES
    }

    def kl(source: Mapping[str, float]) -> float:
        return math.fsum(
            source[class_id] * math.log(source[class_id] / midpoint[class_id])
            for class_id in CLASSES
        )

    result = 0.5 * (kl(first) + kl(second)) / math.log(2.0)
    return min(1.0, max(0.0, result))


def _expected_protocol_shape() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": PROTOCOL_STATUS,
        "action_classes_in_order": list(CLASSES),
        "phase_mapping": dict(PHASE_BY_CLASS),
        "trace_events": list(TRACE_EVENTS),
        "prediction_input_allowlist": list(PREDICTION_INPUT_ALLOWLIST),
        "prediction_consistency_fields": list(PREDICTION_CONSISTENCY_FIELDS),
        "optional_diagnostic_allowlist": list(DIAGNOSTIC_ALLOWLIST),
        "forbidden_inference_fields": list(FORBIDDEN_INFERENCE_FIELDS),
    }


def validate_protocol(value: Any) -> dict[str, Any]:
    """Validate the trace contract and return a detached JSON-compatible copy."""

    protocol = _mapping(value, "reasoning-trace protocol")
    _require(
        set(protocol)
        == {
            "schema_version",
            "protocol_id",
            "status",
            "claim_scope",
            "causal_cutoff",
            "action_classes_in_order",
            "phase_mapping",
            "trace_events",
            "prediction_input_allowlist",
            "prediction_consistency_fields",
            "optional_diagnostic_allowlist",
            "forbidden_inference_fields",
            "proxy_formulas",
            "thresholds",
            "authenticated_design_artifact",
            "unfitted_behavioral_heads",
            "evidence_status",
        },
        "reasoning-trace protocol top-level fields changed",
    )
    for key, expected in _expected_protocol_shape().items():
        _require(protocol.get(key) == expected, f"reasoning-trace protocol {key} changed")

    claim = _mapping(protocol.get("claim_scope"), "claim scope")
    _require(dict(claim) == CLAIM_SCOPE, "reasoning-trace claim scope changed")
    _require(
        protocol.get("causal_cutoff")
        == "final_prompt_boundary_before_current_action",
        "reasoning-trace causal cutoff changed",
    )

    formulas = _mapping(protocol.get("proxy_formulas"), "proxy formulas")
    _require(
        formulas
        == {
            "decision_confidence": "q[argmax(d)]",
            "forecast_doubt": "1-decision_confidence",
            "diffuse_uncertainty": "entropy(q)/log(3)",
            "ambivalence": "1-(q_top1-q_top2)/(q_top1+q_top2)",
            "forecast_volatility": "JSD(q_t,q_previous)/log(2)",
            "source_disagreement": "JSD(p_sequence_logit,p_sequence_j)/log(2)",
            "activation_load_raw": (
                "mean(diffuse_uncertainty,forecast_volatility,source_disagreement,"
                "ordinary_activation_innovation,public_jacobian_activation_innovation)"
            ),
            "activation_load_percentile": "training_ECDF(activation_load_raw)",
        },
        "reasoning-trace proxy formulas changed",
    )

    thresholds = _mapping(protocol.get("thresholds"), "trace thresholds")
    _require(
        dict(thresholds)
        == {
            "uncertainty_band_edges": [1.0 / 3.0, 2.0 / 3.0],
            "trend_minimum_absolute_delta": 0.05,
            "high_friction_percentile_inclusive": 0.75,
            "recovery_low_percentile_inclusive": 0.4,
            "recovery_confidence_rise_minimum": 0.15,
        },
        "reasoning-trace thresholds changed",
    )

    heads = _mapping(protocol.get("unfitted_behavioral_heads"), "behavioral heads")
    _require(
        dict(heads) == UNFITTED_HEADS,
        "reasoning-trace unfitted-head contract changed",
    )
    artifact = _mapping(
        protocol.get("authenticated_design_artifact"),
        "authenticated design artifact",
    )
    _require(
        dict(artifact)
        == {
            "path": DESIGN_ARTIFACT_RELATIVE_PATH,
            "sha256": DESIGN_ARTIFACT_SHA256,
            "artifact_id": DESIGN_ARTIFACT_ID,
            "primary_procedure": DESIGN_PRIMARY_PROCEDURE,
            "source_diagnostic_procedures": dict(DESIGN_SOURCE_PROCEDURES),
            "accepted_and_calibration_identity": (
                "authenticated_by_exact_whole_artifact_sha256"
            ),
            "activation_innovation_and_training_ecdf": (
                "unavailable_until_a_separate_hash_authenticated_causal_training_"
                "artifact_exists"
            ),
        },
        "reasoning-trace authenticated-artifact contract changed",
    )
    evidence = _mapping(protocol.get("evidence_status"), "evidence status")
    _require(
        dict(evidence)
        == {
            "forecast_calibration": "development_design_only",
            "fresh_transport_confirmation": False,
            "operational_reliability_claim": False,
            "row_level_reliability_intervals": (
                "unavailable_no_fresh_transport_confirmatory_bootstrap"
            ),
            "reserved_validation_opened": False,
        },
        "reasoning-trace evidence status changed",
    )
    return json.loads(json.dumps(protocol, allow_nan=False))


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    expanded = path.expanduser()
    _require(
        expanded.is_file() and not expanded.is_symlink(),
        "protocol must be a regular non-symlink file",
    )
    resolved = expanded.resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not read reasoning-trace protocol") from error
    return validate_protocol(value)


def _prediction(row: Mapping[str, Any], position: int) -> dict[str, Any]:
    row_id = _nonempty_text(row.get("row_id"), f"prediction {position} row_id")
    task_id = _nonempty_text(row.get("task_id"), f"prediction {position} task_id")
    request_index = row.get("task_request_index")
    _require(
        isinstance(request_index, int)
        and not isinstance(request_index, bool)
        and request_index >= 1,
        f"prediction {position} task_request_index must be an integer >= 1",
    )
    q = _probabilities(
        row.get("forecast_probabilities_q"),
        f"prediction {position} forecast probabilities q",
    )
    d = _probabilities(
        row.get("decision_probabilities_d"),
        f"prediction {position} decision probabilities d",
    )
    accepted = row.get("accepted")
    _require(isinstance(accepted, bool), f"prediction {position} accepted must be boolean")
    forecast_top = _argmax_class(q)
    decision_top = _argmax_class(d)
    confidence = q[decision_top]

    consistency = {
        "forecast_top_class": forecast_top,
        "forecast_top_confidence": q[forecast_top],
        "predicted_class": decision_top,
        "decision_confidence_from_q": confidence,
    }
    for field, expected in consistency.items():
        _require(
            field in row,
            f"prediction {position} requires evaluator consistency field {field}",
        )
        observed = row[field]
        if isinstance(expected, str):
            _require(observed == expected, f"prediction {position} {field} is inconsistent")
        else:
            numeric = _finite(observed, f"prediction {position} {field}")
            _require(
                math.isclose(
                    numeric,
                    expected,
                    rel_tol=PROBABILITY_TOLERANCE,
                    abs_tol=PROBABILITY_TOLERANCE,
                ),
                f"prediction {position} {field} is inconsistent",
            )
    return {
        "row_id": row_id,
        "task_id": task_id,
        "task_request_index": int(request_index),
        "q": q,
        "d": d,
        "accepted": accepted,
        "forecast_top": forecast_top,
        "decision_top": decision_top,
        "confidence": confidence,
    }


def _diagnostic_row(
    row_id: str, diagnostics_by_row: Mapping[str, Any]
) -> dict[str, Any]:
    if row_id not in diagnostics_by_row:
        return {}
    raw = _mapping(diagnostics_by_row[row_id], f"diagnostics for {row_id}")
    unknown = sorted(set(raw) - set(DIAGNOSTIC_ALLOWLIST))
    _require(not unknown, f"diagnostics for {row_id} contain non-allowlisted fields: {unknown}")
    return dict(raw)


def _source_disagreement(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    has_logit = "sequence_logit_probabilities" in diagnostics
    has_j = "sequence_j_probabilities" in diagnostics
    if not has_logit and not has_j:
        return {
            "value": None,
            "kind": "normalized_jensen_shannon_divergence",
            "status": "unavailable_missing_source_probability_pair",
        }
    if has_logit != has_j:
        return {
            "value": None,
            "kind": "normalized_jensen_shannon_divergence",
            "status": "unavailable_incomplete_source_probability_pair",
        }
    logit = _probabilities(
        diagnostics["sequence_logit_probabilities"],
        "sequence-logit diagnostic probabilities",
    )
    public_j = _probabilities(
        diagnostics["sequence_j_probabilities"],
        "sequence-J diagnostic probabilities",
    )
    return {
        "value": normalized_js_divergence(logit, public_j),
        "kind": "normalized_jensen_shannon_divergence",
        "status": "available_descriptive",
        "sequence_logit_top_class": _argmax_class(logit),
        "sequence_j_top_class": _argmax_class(public_j),
    }


def _innovation(*, first_boundary: bool) -> dict[str, Any]:
    if first_boundary:
        # V4 initializes the raw delta/EMA deviation to zero.  That convention
        # must never be reinterpreted as evidence of genuinely low innovation.
        return {
            "value": None,
            "kind": "robust_scale_bounded_activation_innovation",
            "status": "unavailable_first_boundary",
        }
    return {
        "value": None,
        "kind": "robust_scale_bounded_activation_innovation",
        "status": "unavailable_no_hash_authenticated_causal_training_scale",
    }


def _activation_load(
    *,
    uncertainty: float,
    volatility: float | None,
    disagreement: float | None,
    ordinary_innovation: float | None,
    public_j_innovation: float | None,
) -> dict[str, Any]:
    components = {
        "diffuse_uncertainty": uncertainty,
        "forecast_volatility": volatility,
        "source_disagreement": disagreement,
        "ordinary_activation_innovation": ordinary_innovation,
        "public_jacobian_activation_innovation": public_j_innovation,
    }
    available = [value for value in components.values() if value is not None]
    _require(bool(available), "activation-load calculation has no components")
    raw_index = math.fsum(available) / len(available)
    complete = len(available) == len(LOAD_COMPONENTS)
    if complete:
        status = "raw_index_only_no_training_reference"
    else:
        status = "partial_diagnostic_not_comparable"
    return {
        "raw_index": raw_index,
        "percentile": None,
        "kind": "activation_load_like_not_emotion_probability",
        "status": status,
        "available_component_count": len(available),
        "required_component_count": len(LOAD_COMPONENTS),
        "components": components,
        "training_reference_count": 0,
        "emotion_semantics": False,
    }


def _band(value: float, edges: Sequence[float]) -> str:
    if value < edges[0]:
        return "low"
    if value < edges[1]:
        return "moderate"
    return "high"


def _trend(current: float, previous: float | None, minimum_delta: float) -> str:
    if previous is None:
        return "unavailable_first_boundary"
    delta = current - previous
    if delta >= minimum_delta:
        return "rising"
    if delta <= -minimum_delta:
        return "falling"
    return "stable"


def _primary_event(
    *, accepted: bool, current_phase: str, previous_phase: str | None
) -> str:
    if not accepted:
        return "uncertain_transition"
    if previous_phase is None:
        return "phase_observation"
    if current_phase == previous_phase:
        return "phase_continuation"
    if previous_phase == "information_gathering" and current_phase == "implementation":
        return "inspection_to_implementation"
    if (
        previous_phase == "implementation"
        and current_phase == "verification_or_completion"
    ):
        return "implementation_to_verification_or_completion"
    if (
        previous_phase == "verification_or_completion"
        and current_phase in {"information_gathering", "implementation"}
    ):
        return "reconsideration_or_rework_like"
    return "phase_transition"


def _display_phase(phase: str) -> str:
    return phase.replace("_", " ").replace("verification or completion", "verification or completion")


def render_rationale(template_inputs: Mapping[str, Any]) -> str:
    """Render the exact non-generative rationale template from emitted fields."""

    values = _mapping(template_inputs, "rationale template inputs")
    accepted = values.get("accepted")
    _require(isinstance(accepted, bool), "rationale accepted must be boolean")
    candidate_phase = _nonempty_text(values.get("candidate_phase"), "candidate phase")
    selected_phase = values.get("selected_phase")
    if accepted:
        _require(selected_phase == candidate_phase, "accepted rationale phase mismatch")
    else:
        _require(selected_phase is None, "abstained rationale must not select a phase")
    confidence = _unit_interval(values.get("decision_confidence"), "decision confidence")
    uncertainty = _unit_interval(values.get("diffuse_uncertainty"), "diffuse uncertainty")
    uncertainty_band = _nonempty_text(values.get("uncertainty_band"), "uncertainty band")
    uncertainty_trend = _nonempty_text(values.get("uncertainty_trend"), "uncertainty trend")
    previous_uncertainty = values.get("previous_diffuse_uncertainty")
    if previous_uncertainty is not None:
        previous_uncertainty = _unit_interval(
            previous_uncertainty, "previous diffuse uncertainty"
        )
    fixed_edges = (1.0 / 3.0, 2.0 / 3.0)
    _require(
        uncertainty_band == _band(uncertainty, fixed_edges),
        "rationale uncertainty band is not entailed by its value",
    )
    _require(
        uncertainty_trend == _trend(uncertainty, previous_uncertainty, 0.05),
        "rationale uncertainty trend is not entailed by its values",
    )
    previous_phase = values.get("previous_accepted_phase")
    _require(
        previous_phase is None or previous_phase in set(PHASE_BY_CLASS.values()),
        "previous accepted phase is invalid",
    )
    events = list(_sequence(values.get("events"), "rationale events"))
    _require(
        bool(events) and all(event in TRACE_EVENTS for event in events),
        "rationale events are invalid",
    )
    _require(
        events[0]
        == _primary_event(
            accepted=accepted,
            current_phase=candidate_phase,
            previous_phase=previous_phase,
        )
        and events[1:] in ([], ["recovery_like"]),
        "rationale events are not entailed by phase inputs",
    )
    evidence_status = _nonempty_text(values.get("evidence_status"), "evidence status")
    _require(
        evidence_status
        in {"development_design_only", "synthetic_or_unverified_test_input"},
        "rationale evidence status is invalid",
    )

    phase_text = _display_phase(candidate_phase)
    if accepted:
        clauses = [
            f"{phase_text.capitalize()} is the selected next phase "
            f"({confidence:.3f} provisional forecast confidence for the selected action)."
        ]
    else:
        clauses = [
            "The phase readout abstained; the decision candidate was "
            f"{phase_text} with {confidence:.3f} provisional forecast confidence."
        ]
    if uncertainty_trend == "unavailable_first_boundary":
        clauses.append(
            f"Normalized forecast uncertainty is {uncertainty_band} ({uncertainty:.3f}); "
            "no prior boundary is available for a trend comparison."
        )
    else:
        clauses.append(
            f"Normalized forecast uncertainty is {uncertainty_band} ({uncertainty:.3f}) "
            f"and {uncertainty_trend}."
        )

    disagreement = values.get("source_disagreement")
    if disagreement is not None:
        disagreement_value = _unit_interval(disagreement, "source disagreement")
        disagreement_band = _nonempty_text(
            values.get("source_disagreement_band"), "source disagreement band"
        )
        _require(
            disagreement_band == _band(disagreement_value, fixed_edges),
            "rationale disagreement band is not entailed by its value",
        )
        clauses.append(
            "Logit-only and J-only forecast disagreement is "
            f"{disagreement_band} ({disagreement_value:.3f})."
        )

    load_percentile = values.get("activation_load_percentile")
    load_raw = _unit_interval(values.get("activation_load_raw"), "activation-load raw index")
    load_status = _nonempty_text(values.get("activation_load_status"), "activation-load status")
    load_count = values.get("activation_load_component_count")
    _require(
        isinstance(load_count, int) and not isinstance(load_count, bool),
        "activation-load component count must be an integer",
    )
    if load_percentile is not None:
        percentile = _unit_interval(load_percentile, "activation-load percentile")
        _require(
            load_status == "available_training_reference_percentile"
            and load_count == len(LOAD_COMPONENTS),
            "rationale load percentile is inconsistent with load status",
        )
        clauses.append(
            "Activation-load-like evidence is at the "
            f"{percentile:.3f} training-reference percentile."
        )
    elif load_status == "raw_index_only_no_training_reference":
        _require(
            load_count == len(LOAD_COMPONENTS),
            "raw-only rationale must have every load component",
        )
        clauses.append(
            f"Activation-load-like evidence has raw diagnostic index {load_raw:.3f}; "
            "no training-reference percentile is available."
        )
    else:
        _require(
            load_status == "partial_diagnostic_not_comparable"
            and 0 < load_count < len(LOAD_COMPONENTS),
            "partial-load rationale status is inconsistent",
        )
        clauses.append(
            f"Only {load_count}/{len(LOAD_COMPONENTS)} activation-load components are "
            "available, so no comparable load percentile is emitted."
        )

    if previous_phase is not None:
        clauses.append(f"The prior accepted phase was {_display_phase(previous_phase)}.")
    if "recovery_like" in events:
        clauses.append(
            "The preregistered recovery-like proxy rule fired; this remains a "
            "descriptive task-state proxy."
        )
    if evidence_status == "development_design_only":
        clauses.append("Evidence is design-only and transport-unconfirmed.")
    else:
        clauses.append(
            "Input provenance is synthetic or unverified; this rationale is for "
            "schema testing only."
        )
    clauses.append(
        "This is an observable task-state proxy, not private chain-of-thought or emotion."
    )
    return " ".join(clauses)


def _unfitted_heads(protocol: Mapping[str, Any]) -> dict[str, Any]:
    heads = _mapping(protocol["unfitted_behavioral_heads"], "behavioral heads")
    return {
        name: {
            "value": None,
            "target": str(_mapping(heads[name], name)["target"]),
            "status": str(heads[name]["status"]),
        }
        for name in (
            "hesitation_like_probability",
            "trajectory_continuation_probability",
            "recovery_within_2_probability",
        )
    }


def _build_reasoning_trace(
    predictions: Sequence[Mapping[str, Any]],
    *,
    diagnostics_by_row: Mapping[str, Mapping[str, Any]] | None = None,
    protocol: Mapping[str, Any] | None = None,
    source_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build a deterministic, causal reasoning-state sidecar.

    Prediction records may contain arbitrary extra metadata.  Only the six
    fields in ``PREDICTION_INPUT_ALLOWLIST`` affect inference; four required
    evaluator summary fields are read solely to fail on internal inconsistency.
    Diagnostic inputs are separate and reject every non-allowlisted key.
    """

    rows = _sequence(predictions, "predictions")
    _require(bool(rows), "predictions must not be empty")
    contract = validate_protocol(protocol if protocol is not None else load_protocol())
    diagnostics = (
        {}
        if diagnostics_by_row is None
        else dict(_mapping(diagnostics_by_row, "diagnostics by row"))
    )
    for diagnostic_row_id, diagnostic_value in diagnostics.items():
        _nonempty_text(diagnostic_row_id, "diagnostic row ID")
        _mapping(diagnostic_value, f"diagnostics for {diagnostic_row_id}")
    binding = dict(_mapping(source_binding, "source binding"))
    authenticated = binding.get("kind") == "authenticated_v4_design_artifact"
    if authenticated:
        _require(
            binding.get("artifact_sha256") == DESIGN_ARTIFACT_SHA256
            and binding.get("artifact_id") == DESIGN_ARTIFACT_ID
            and binding.get("primary_procedure") == DESIGN_PRIMARY_PROCEDURE
            and binding.get("source_diagnostic_procedures")
            == DESIGN_SOURCE_PROCEDURES
            and binding.get("input_authenticated") is True,
            "authenticated source binding is invalid",
        )
        evidence_level = "development_design_only"
        confidence_status = "development_calibrated_transport_unconfirmed"
        diagnostic_status = "available_authenticated_design_diagnostic"
    else:
        _require(
            binding
            == {
                "kind": "synthetic_or_unverified_test_input",
                "input_authenticated": False,
            },
            "unverified source binding is invalid",
        )
        evidence_level = "synthetic_or_unverified_test_input"
        confidence_status = "synthetic_unvalidated"
        diagnostic_status = "available_synthetic_test_only"
    thresholds = _mapping(contract["thresholds"], "trace thresholds")
    edges = [float(item) for item in thresholds["uncertainty_band_edges"]]
    trend_threshold = float(thresholds["trend_minimum_absolute_delta"])
    protocol_sha256 = _canonical_sha256(contract)

    seen_rows: set[str] = set()
    state_by_task: dict[str, dict[str, Any]] = {}
    result: list[dict[str, Any]] = []
    for position, source in enumerate(rows):
        row = _mapping(source, f"prediction {position}")
        current = _prediction(row, position)
        row_id = current["row_id"]
        _require(row_id not in seen_rows, f"duplicate prediction row_id: {row_id}")
        seen_rows.add(row_id)
        task_id = current["task_id"]
        request_index = current["task_request_index"]
        state = state_by_task.setdefault(
            task_id,
            {
                "previous_request_index": None,
                "previous_q": None,
                "previous_uncertainty": None,
                "previous_accepted_phase": None,
            },
        )
        previous_index = state["previous_request_index"]
        _require(
            previous_index is None or request_index > previous_index,
            f"prediction request indices for {task_id} must be strictly increasing",
        )

        row_diagnostics = _diagnostic_row(row_id, diagnostics)
        first_boundary = previous_index is None
        q = current["q"]
        d = current["d"]
        confidence = float(current["confidence"])
        doubt = 1.0 - confidence
        uncertainty = normalized_entropy(q)
        ambivalence = ambivalence_index(q)
        previous_q = state["previous_q"]
        volatility = (
            None
            if previous_q is None
            else normalized_js_divergence(q, previous_q)
        )
        source_disagreement = _source_disagreement(row_diagnostics)
        if source_disagreement["value"] is not None:
            source_disagreement["status"] = diagnostic_status
        ordinary_innovation = _innovation(first_boundary=first_boundary)
        public_j_innovation = _innovation(first_boundary=first_boundary)
        load = _activation_load(
            uncertainty=uncertainty,
            volatility=volatility,
            disagreement=source_disagreement["value"],
            ordinary_innovation=ordinary_innovation["value"],
            public_j_innovation=public_j_innovation["value"],
        )

        candidate_phase = PHASE_BY_CLASS[current["decision_top"]]
        accepted = bool(current["accepted"])
        selected_phase = candidate_phase if accepted else None
        previous_phase = (
            state["previous_accepted_phase"]
            if previous_index is not None and request_index == previous_index + 1
            else None
        )
        events = [
            _primary_event(
                accepted=accepted,
                current_phase=candidate_phase,
                previous_phase=previous_phase,
            )
        ]

        load_percentile = None
        recovery_status = "unavailable_unfitted_authenticated_load_episode_rule"
        recovery_value: bool | None = None

        uncertainty_band = _band(uncertainty, edges)
        uncertainty_trend = _trend(
            uncertainty, state["previous_uncertainty"], trend_threshold
        )
        disagreement_value = source_disagreement["value"]
        disagreement_band = (
            None
            if disagreement_value is None
            else _band(float(disagreement_value), edges)
        )
        template_inputs = {
            "accepted": accepted,
            "candidate_phase": candidate_phase,
            "selected_phase": selected_phase,
            "decision_confidence": confidence,
            "diffuse_uncertainty": uncertainty,
            "uncertainty_band": uncertainty_band,
            "uncertainty_trend": uncertainty_trend,
            "previous_diffuse_uncertainty": state["previous_uncertainty"],
            "source_disagreement": disagreement_value,
            "source_disagreement_band": disagreement_band,
            "activation_load_raw": load["raw_index"],
            "activation_load_percentile": load_percentile,
            "activation_load_status": load["status"],
            "activation_load_component_count": load["available_component_count"],
            "previous_accepted_phase": previous_phase,
            "events": list(events),
            "evidence_status": evidence_level,
        }
        rationale = render_rationale(template_inputs)
        head_placeholders = _unfitted_heads(contract)
        head_placeholders["recovery_like_event"] = {
            "value": recovery_value,
            "kind": "trajectory_defined_descriptive_event_not_emotion",
            "status": recovery_status,
        }
        forecast_phase_probabilities = {
            PHASE_BY_CLASS[class_id]: q[class_id] for class_id in CLASSES
        }
        decision_phase_probabilities = {
            PHASE_BY_CLASS[class_id]: d[class_id] for class_id in CLASSES
        }

        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "claim_scope": dict(contract["claim_scope"]),
                "boundary": {
                    "row_id_sha256": hashlib.sha256(row_id.encode("utf-8")).hexdigest(),
                    "task_id_sha256": hashlib.sha256(task_id.encode("utf-8")).hexdigest(),
                    "request_index": request_index,
                    "causal_cutoff": contract["causal_cutoff"],
                },
                "phase_forecast": {
                    "classes": [PHASE_BY_CLASS[class_id] for class_id in CLASSES],
                    "probabilities_q": forecast_phase_probabilities,
                    "decision_probabilities_d": decision_phase_probabilities,
                    "forecast_top_phase": PHASE_BY_CLASS[current["forecast_top"]],
                    "candidate_phase": candidate_phase,
                    "selected_phase": selected_phase,
                    "decision_confidence": confidence,
                    "confidence_status": confidence_status,
                    "accepted": accepted,
                },
                "proxies": {
                    "forecast_doubt": {
                        "value": doubt,
                        "kind": (
                            "provisional_calibrated_probability_complement"
                            if authenticated
                            else "synthetic_probability_complement"
                        ),
                        "status": (
                            "available_authenticated_design_only"
                            if authenticated
                            else "available_synthetic_test_only"
                        ),
                    },
                    "diffuse_uncertainty": {
                        "value": uncertainty,
                        "kind": "normalized_forecast_entropy",
                        "status": "available_descriptive",
                    },
                    "ambivalence": {
                        "value": ambivalence,
                        "kind": "top_two_forecast_closeness",
                        "status": "available_descriptive",
                    },
                    "forecast_volatility": {
                        "value": volatility,
                        "kind": "normalized_jensen_shannon_divergence_from_prior_boundary",
                        "status": (
                            diagnostic_status
                            if volatility is not None
                            else "unavailable_first_boundary"
                        ),
                    },
                    "source_disagreement": source_disagreement,
                    "ordinary_activation_innovation": ordinary_innovation,
                    "public_jacobian_activation_innovation": public_j_innovation,
                    "forecast_decision_class_conflict": {
                        "value": current["forecast_top"] != current["decision_top"],
                        "kind": "categorical_forecast_policy_disagreement",
                        "status": "available_descriptive",
                    },
                    "activation_load_like": load,
                    **head_placeholders,
                },
                "emotion_inference": {
                    "status": "unsupported",
                    "value": None,
                    "reason": (
                        "no independently observable or experimentally validated "
                        "subjective-emotion target"
                    ),
                },
                "reasoning_trace": {
                    "kind": "high_level_readout_phase_forecast_trace_not_private_cot",
                    "events": list(events),
                    "template_id": "phase_confidence_uncertainty_transition_load_v1",
                    "rationale": rationale,
                    "template_inputs": template_inputs,
                    "evidence_fields": [
                        "candidate_phase",
                        "selected_phase",
                        "decision_confidence",
                        "diffuse_uncertainty",
                        "uncertainty_trend",
                        "source_disagreement",
                        "activation_load_raw",
                        "activation_load_percentile",
                        "previous_accepted_phase",
                        "events",
                        "evidence_status",
                    ],
                },
                "abstention": {
                    "abstained": not accepted,
                    "reasons": (
                        []
                        if accepted
                        else ["v4_selective_action_threshold_not_met"]
                    ),
                    "probability_vector_retained": True,
                },
                "reliability": {
                    "status": (
                        "unavailable_no_fresh_transport_confirmatory_bootstrap"
                    ),
                    "decision_confidence_interval": None,
                    "selective_accuracy_lower_bound": None,
                    "coverage_lower_bound": None,
                },
                "provenance": {
                    "protocol_id": PROTOCOL_ID,
                    "protocol_canonical_sha256": protocol_sha256,
                    "source_binding": binding,
                    "evidence_level": evidence_level,
                    "input_authenticated": authenticated,
                    "fresh_transport_confirmation": False,
                    "operational_reliability_claim": False,
                    "reserved_validation_opened": False,
                },
            }
        )

        state["previous_request_index"] = request_index
        state["previous_q"] = q
        state["previous_uncertainty"] = uncertainty
        state["previous_accepted_phase"] = candidate_phase if accepted else None

    extra_diagnostics = sorted(set(diagnostics) - seen_rows)
    _require(
        not extra_diagnostics,
        f"diagnostics contain unknown prediction row IDs: {extra_diagnostics}",
    )
    return result


__all__ = [
    "CLASSES",
    "DEFAULT_PROTOCOL",
    "DESIGN_ARTIFACT_PATH",
    "DESIGN_ARTIFACT_SHA256",
    "DIAGNOSTIC_ALLOWLIST",
    "FORBIDDEN_INFERENCE_FIELDS",
    "PHASE_BY_CLASS",
    "PREDICTION_CONSISTENCY_FIELDS",
    "PREDICTION_INPUT_ALLOWLIST",
    "PROTOCOL_ID",
    "PROTOCOL_STATUS",
    "SCHEMA_VERSION",
    "TRACE_EVENTS",
    "ambivalence_index",
    "build_reasoning_trace",
    "build_reasoning_trace_from_v4_design_artifact",
    "load_protocol",
    "normalized_entropy",
    "normalized_js_divergence",
    "render_rationale",
    "validate_protocol",
]
