#!/usr/bin/env python3
"""Fit and evaluate the frozen binary SWE phase interpreter.

The interpreter predicts whether the next known consequential Qwen Code action
is an edit or a check/finalization.  Inference is causal at a request boundary.
The evaluation label is not: it is derived offline by scanning the recorded
future trajectory.  Consequently this program emits a prediction for every
numerically stable, feature-complete replay row and computes metrics only for
the subset whose future-derived label is ascertainable.

Two matched readouts are always fitted and evaluated:

* ``j_compact`` uses the public Jacobian-lens action scores.
* ``l_compact`` uses ordinary logit-lens action scores.

Use ``fit`` on the frozen development cohort to produce one hash-bound joblib
bundle and its JSON manifest.  Use ``evaluate`` to apply that immutable bundle
to a fresh prompt/report pair.  Joblib files are executable pickle artifacts;
only evaluate a bundle whose SHA-256 is pinned by a trusted manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ANALYZER_PATH = ROOT / "scripts/analyze_swe_task_state_interpreter.py"
COHORT_CHECKER_PATH = ROOT / "scripts/check_swe_task_state_validation_cohort.py"
DEFAULT_PROTOCOL = ROOT / "configs/swe_binary_phase_interpreter_v2.json"
DEFAULT_SOURCE_PROTOCOL = ROOT / "configs/swe_task_state_interpreter_protocol.json"
DEFAULT_BEHAVIORAL_PROTOCOL = ROOT / "configs/swe_behavioral_readout_protocol.json"
DEFAULT_RESERVED_COHORT = ROOT / "configs/swe_task_state_validation_cohort.json"
DEFAULT_RESERVED_CAMPAIGN_A = (
    ROOT / "configs/swe_task_state_validation_a_campaign.json"
)
DEFAULT_RESERVED_CAMPAIGN_B = (
    ROOT / "configs/swe_task_state_validation_b_campaign.json"
)
DEFAULT_RESERVED_IMAGE_REGISTRY = (
    ROOT / "configs/swe_task_state_validation_image_digests.json"
)

SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
PROTOCOL_ID = "swe-binary-phase-interpreter-v2"
CLASSES = ("edit", "check_or_finish")
VARIANTS = ("j_compact", "l_compact")
SOURCE_ACTION_CLASSES = ("inspect", "edit", "validate", "finalize")
MILESTONE_ACTIONS = ("edit", "validate", "finalize")
FEATURE_WIDTH = 154
SOURCE_LAYER_COUNT = 24
SOURCE_CLASS_COUNT = 4
DEFAULT_SEED = 271828
DEFAULT_BOOTSTRAP_SEED = 44021
DEFAULT_BOOTSTRAP_SAMPLES = 5000
CANARY_PROBABILITY_DECIMAL_PLACES = 12
CANARY_PROBABILITY_ATOL = 1e-12
CANARY_PROBABILITY_RTOL = 0.0
GATE_EVIDENCE_DECIMAL_PLACES = 14
RESERVED_SUMMARY_PIN_KEY = "reserved_prompts_summary_sha256"
RESERVED_SUMMARY_PIN_JSON_POINTER = "/pins/reserved_prompts_summary_sha256"
CORE_PROTOCOL_HASH_ENCODING = "sha256_sorted_compact_ascii_json"


def _ml_dependencies():
    """Load serialization/training dependencies only for fit/evaluate paths."""
    try:
        import joblib
        from sklearn.ensemble import ExtraTreesClassifier
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "fit/evaluate requires the pinned readout-v2 environment "
            "with joblib and scikit-learn installed"
        ) from error
    return joblib, ExtraTreesClassifier


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be nonempty text")
    return value


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return int(value)


def finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value independently of whitespace and object key order."""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError(f"value is not canonical ASCII JSON: {error}") from error
    return sha256_bytes(payload)


def core_protocol_sha256(value: Any) -> str:
    """Hash the frozen protocol while normalizing its sole append-only pin."""
    protocol = mapping(value, "binary phase protocol")
    pins = mapping(protocol.get("pins"), "v2 pins")
    require(
        RESERVED_SUMMARY_PIN_KEY in pins,
        "reserved prompts summary pin is missing",
    )
    normalized = dict(protocol)
    normalized_pins = dict(pins)
    normalized_pins[RESERVED_SUMMARY_PIN_KEY] = None
    normalized["pins"] = normalized_pins
    return canonical_json_sha256(normalized)


def _validate_sha(value: Any, label: str) -> str:
    result = nonempty_string(value, label)
    require(
        len(result) == 64
        and all(character in "0123456789abcdef" for character in result),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return result


def read_json(path: Path, label: str) -> Any:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label} as JSON: {path}: {error}") from error


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_joblib_dump(path: Path, value: Any) -> None:
    joblib, _ = _ml_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        joblib.dump(value, temporary, compress=3, protocol=5)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_source_analyzer():
    name = "swe_task_state_interpreter_v1_for_binary_v2"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SOURCE_ANALYZER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE_ANALYZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SOURCE_ANALYZER = _load_source_analyzer()


def _load_cohort_checker():
    name = "swe_task_state_validation_cohort_for_binary_v2"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, COHORT_CHECKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {COHORT_CHECKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COHORT_CHECKER = _load_cohort_checker()


def _optional_nested(
    value: Mapping[str, Any], names: Sequence[str], default: Any = None
) -> Any:
    current: Any = value
    for name in names:
        if not isinstance(current, dict) or name not in current:
            return default
        current = current[name]
    return current


def _protocol_value(
    protocol: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
    label: str,
) -> Any:
    sentinel = object()
    for path in paths:
        result = _optional_nested(protocol, path, sentinel)
        if result is not sentinel:
            return result
    joined = " or ".join(".".join(path) for path in paths)
    raise ValueError(f"{label} is missing ({joined})")


def validate_v2_protocol(
    value: Any,
    *,
    protocol_sha256: str,
    source_protocol_sha256: str,
    behavioral_protocol_sha256: str,
) -> dict[str, Any]:
    """Validate the executable portions of the frozen v2 protocol.

    The JSON document remains the authoritative preregistration.  This
    validator deliberately checks every field that can change extraction,
    fitting, serialization, evaluation, or a pass/fail decision.
    """
    protocol = mapping(value, "binary phase protocol")
    require(protocol.get("schema_version") == 1, "binary protocol schema changed")
    require(protocol.get("id") == PROTOCOL_ID, "binary protocol identity changed")
    protocol_core_sha256 = core_protocol_sha256(protocol)

    class_contract = mapping(protocol.get("class_contract"), "class contract")
    classes = _protocol_value(
        class_contract,
        (("class_ids_in_order",), ("classes_in_order",), ("classes",)),
        "binary class order",
    )
    require(classes == list(CLASSES), "binary class order changed")
    require(
        class_contract.get("edit")
        in {
            "next known consequential milestone is edit",
            "next_known_consequential_milestone_is_edit",
        }
        or class_contract.get("edit_source_actions") == ["edit"],
        "edit target mapping changed",
    )
    check_sources = _protocol_value(
        class_contract,
        (("check_or_finish_source_actions",), ("check_or_finish", "source_actions")),
        "check-or-finish target mapping",
    )
    require(
        check_sources == ["validate", "finalize"],
        "check-or-finish target mapping changed",
    )

    pins = mapping(protocol.get("pins"), "v2 pins")
    source_pin = _protocol_value(
        pins,
        (
            ("source_task_state_protocol_sha256",),
            ("task_state_protocol_sha256",),
            ("v1_task_state_protocol_sha256",),
        ),
        "source protocol pin",
    )
    behavioral_pin = _protocol_value(
        pins,
        (("behavioral_protocol_sha256",),),
        "behavioral protocol pin",
    )
    require(
        _validate_sha(source_pin, "source protocol pin") == source_protocol_sha256,
        "source task-state protocol pin differs from supplied file",
    )
    require(
        _validate_sha(behavioral_pin, "behavioral protocol pin")
        == behavioral_protocol_sha256,
        "behavioral protocol pin differs from supplied file",
    )
    development_prompt_pin = _validate_sha(
        pins.get("development_prompt_bundle_sha256"), "development prompt pin"
    )
    development_report_pin = _validate_sha(
        pins.get("development_public_report_sha256"), "development report pin"
    )
    reserved_pins = {
        "cohort": _validate_sha(
            pins.get("reserved_cohort_sha256"), "reserved cohort pin"
        ),
        "campaign_a": _validate_sha(
            pins.get("reserved_campaign_a_sha256"), "reserved campaign A pin"
        ),
        "campaign_b": _validate_sha(
            pins.get("reserved_campaign_b_sha256"), "reserved campaign B pin"
        ),
        "image_registry": _validate_sha(
            pins.get("reserved_image_registry_sha256"),
            "reserved image registry pin",
        ),
    }
    cohort_checker_pin = _validate_sha(
        pins.get("cohort_checker_sha256"), "cohort checker implementation pin"
    )
    require(
        cohort_checker_pin == sha256_file(COHORT_CHECKER_PATH),
        "cohort checker implementation differs from protocol pin",
    )
    reserved_summary_pin_value = pins.get("reserved_prompts_summary_sha256")
    reserved_summary_pin = (
        None
        if reserved_summary_pin_value is None
        else _validate_sha(
            reserved_summary_pin_value, "reserved prompts summary pin"
        )
    )

    summary_contract = mapping(
        protocol.get("reserved_prompts_summary_contract"),
        "reserved prompts summary contract",
    )
    require(
        summary_contract
        == {
            "required_evaluate_input": True,
            "validator": "scripts/check_swe_task_state_validation_cohort.py::validate_materialized_bundle",
            "checker_implementation_sha256_is_protocol_pinned": True,
            "literal_summary_sha256_required_before_evaluate": True,
            "null_summary_pin_blocks_evaluate": True,
            "pinning_phase": "after_reserved_materialization_and_before_lens_replay_or_evaluation",
            "core_protocol_identity_normalizes_only_json_pointer": RESERVED_SUMMARY_PIN_JSON_POINTER,
            "core_protocol_hash_encoding": CORE_PROTOCOL_HASH_ENCODING,
            "fit_requires_null_summary_pin": True,
            "evaluation_requires_null_to_literal_sha256_transition": True,
            "core_protocol_sha256_must_match_across_transition": True,
            "any_other_protocol_mutation_forbidden": True,
            "post_materialization_model_refit_forbidden": True,
            "observed_summary_sha256_recorded_at_evaluation": True,
            "summary_must_bind_exact_prompt_bundle_sha256": True,
            "summary_must_bind_cohort_and_campaign_sha256s": True,
            "summary_must_bind_action_protocol_and_chat_template_sha256s": True,
            "every_prompt_payload_sha256_must_verify": True,
            "exact_prompt_ids_global_indices_all_probeable_fields_and_task_order_required": True,
        },
        "reserved prompts summary contract changed",
    )

    eligibility = mapping(
        protocol.get("eligibility_contract"), "v2 eligibility contract"
    )
    require(
        eligibility.get("prediction_requires_current_action_label") is False,
        "v2 inference must not require the current action label",
    )
    require(
        eligibility.get("future_label_controls_metric_eligibility_only") is True,
        "future labels must control metric eligibility only",
    )

    feature = mapping(protocol.get("feature_contract"), "v2 feature contract")
    source_layers = _protocol_value(
        feature,
        (("source_layers",), ("layers",)),
        "v2 feature layers",
    )
    require(
        source_layers == list(range(24, 48)),
        "v2 feature layers must remain 24 through 47",
    )
    source_classes = _protocol_value(
        feature,
        (("source_action_classes_in_order",), ("source_class_ids_in_order",)),
        "source action class order",
    )
    require(
        source_classes == list(SOURCE_ACTION_CLASSES),
        "source action class order changed",
    )
    variants = _protocol_value(
        feature,
        (("variants",), ("variants_in_order",)),
        "feature variants",
    )
    require(variants == list(VARIANTS), "v2 feature variants changed")
    width = integer(
        _protocol_value(feature, (("feature_width",), ("width",)), "feature width"),
        "feature width",
        minimum=1,
    )
    require(width == FEATURE_WIDTH, "v2 feature width changed")
    ema_alpha = finite(
        _protocol_value(feature, (("ema_alpha",),), "EMA alpha"), "EMA alpha"
    )
    require(ema_alpha == 0.5, "EMA alpha changed")

    model = mapping(protocol.get("model_contract"), "v2 model contract")
    family = _protocol_value(
        model, (("family",), ("classifier",)), "model family"
    )
    require(family == "ExtraTreesClassifier", "v2 classifier changed")
    params_value = model.get("parameters", model)
    params = mapping(params_value, "ExtraTrees parameters")
    seed = integer(
        _protocol_value(model, (("random_seed",), ("random_state",), ("seed",)), "model seed"),
        "model seed",
    )
    require(
        integer(params.get("n_estimators"), "n_estimators", minimum=1) == 100
        and integer(params.get("min_samples_leaf"), "min_samples_leaf", minimum=1)
        == 5
        and finite(params.get("max_features"), "max_features") == 0.5
        and seed == DEFAULT_SEED,
        "v2 ExtraTrees architecture or seed changed",
    )
    require(
        model.get("probability_class_order") == list(CLASSES),
        "model probability class order changed",
    )
    expected_estimator_parameters = {
        "bootstrap": False,
        "ccp_alpha": 0.0,
        "class_weight": None,
        "criterion": "gini",
        "max_depth": None,
        "max_features": 0.5,
        "max_leaf_nodes": None,
        "max_samples": None,
        "min_impurity_decrease": 0.0,
        "min_samples_leaf": 5,
        "min_samples_split": 2,
        "min_weight_fraction_leaf": 0.0,
        "monotonic_cst": None,
        "n_estimators": 100,
        "n_jobs": 8,
        "oob_score": False,
        "verbose": 0,
        "warm_start": False,
    }
    require(
        dict(params) == expected_estimator_parameters,
        "explicit ExtraTrees parameter contract changed",
    )

    calibration = mapping(
        protocol.get("calibration_and_abstention"),
        "calibration and abstention contract",
    )
    temperature = finite(
        _protocol_value(
            calibration,
            (("temperature",), ("fixed_temperature",)),
            "fixed temperature",
        ),
        "fixed temperature",
    )
    threshold = finite(
        _protocol_value(
            calibration,
            (("confidence_threshold",), ("fixed_confidence_threshold",), ("tau",)),
            "fixed confidence threshold",
        ),
        "fixed confidence threshold",
    )
    require(temperature == 1.0, "v2 temperature must remain 1.0")
    require(threshold == 0.0, "v2 confidence threshold must remain 0.0")

    horizon = mapping(protocol.get("horizon_reporting"), "horizon reporting contract")
    require(
        horizon.get("mandatory") is True
        and horizon.get("buckets")
        == [
            {"id": "h0", "minimum": 0, "maximum": 0},
            {"id": "h1_2", "minimum": 1, "maximum": 2},
            {"id": "h3_5", "minimum": 3, "maximum": 5},
            {"id": "h6_10", "minimum": 6, "maximum": 10},
            {"id": "h11_plus", "minimum": 11, "maximum": None},
        ],
        "mandatory horizon reporting contract changed",
    )

    bootstrap = mapping(protocol.get("bootstrap"), "v2 bootstrap contract")
    samples = integer(bootstrap.get("samples"), "bootstrap samples", minimum=1)
    bootstrap_seed = integer(bootstrap.get("seed"), "bootstrap seed")
    confidence_level = finite(
        bootstrap.get("confidence_level"), "bootstrap confidence level"
    )
    minimum_valid_fraction = finite(
        bootstrap.get("minimum_valid_fraction"), "bootstrap minimum valid fraction"
    )
    require(
        bootstrap.get("algorithm")
        in {
            "repository_then_task_cluster_percentile_conditional_on_oof_predictions",
            "hierarchical_repository_then_task_percentile_v1",
        }
        and bootstrap.get("models_refit_inside_bootstrap") is False
        and bootstrap.get("row_resampling_forbidden") is True
        and samples >= 1
        and 0.0 < confidence_level < 1.0
        and 0.0 < minimum_valid_fraction <= 1.0,
        "v2 bootstrap contract changed",
    )

    serialization = mapping(
        protocol.get("serialization_contract"), "serialization contract"
    )
    require(
        serialization.get("format") in {"joblib", "single_joblib_dictionary"}
        and serialization.get("single_bundle_contains_both_variants") is True
        and serialization.get("gate_evidence_float_decimal_places")
        == GATE_EVIDENCE_DECIMAL_PLACES
        and serialization.get("gate_evidence_hash_encoding")
        == "sorted_compact_ascii_json_after_recursive_decimal_rounding"
        and serialization.get(
            "published_bundle_sha256_is_an_integrity_pin_not_a_refit_byte_identity_claim"
        )
        is True
        and serialization.get(
            "semantic_reproducibility_uses_runtime_analyzer_feature_and_probability_canaries"
        )
        is True,
        "v2 serialization contract changed",
    )

    weighting = mapping(protocol.get("weighting_contract"), "weighting contract")
    require(
        weighting.get("training")
        in {
            "equal_task_then_equal_event_then_equal_prefix_then_binary_class_rebalance",
            "task_event_prefix_then_binary_class_rebalance",
        }
        and weighting.get("evaluation")
        in {
            "equal_task_then_equal_event_then_equal_prefix",
            "task_event_prefix",
        },
        "v2 weighting contract changed",
    )

    reliability = mapping(protocol.get("reliability_gates"), "reliability gates")
    development_gates = _gate_set(
        reliability.get("development", reliability.get("development_gates")),
        "development gates",
    )
    validation_gates = _gate_set(
        reliability.get("validation", reliability.get("validation_gates")),
        "validation gates",
    )

    return {
        "value": dict(protocol),
        "sha256": protocol_sha256,
        "core_sha256": protocol_core_sha256,
        "development_prompt_sha256": development_prompt_pin,
        "development_public_report_sha256": development_report_pin,
        "reserved_pins": reserved_pins,
        "cohort_checker_sha256": cohort_checker_pin,
        "reserved_prompts_summary_sha256": reserved_summary_pin,
        "classes": CLASSES,
        "variants": VARIANTS,
        "source_layers": tuple(source_layers),
        "source_classes": tuple(source_classes),
        "feature_width": width,
        "ema_alpha": ema_alpha,
        "model": {
            "n_estimators": 100,
            "min_samples_leaf": 5,
            "max_features": 0.5,
            "random_state": seed,
            "n_jobs": integer(params.get("n_jobs"), "n_jobs", minimum=1),
            "explicit_parameters": expected_estimator_parameters,
        },
        "temperature": temperature,
        "threshold": threshold,
        "bootstrap": {
            "samples": samples,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "minimum_valid_fraction": minimum_valid_fraction,
        },
        "gates": {
            "development": development_gates,
            "validation": validation_gates,
        },
    }


def build_fit_protocol_lifecycle(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Create the model-bound identity for the pre-materialization protocol."""
    require(
        protocol.get("reserved_prompts_summary_sha256") is None,
        "fit requires reserved prompts summary pin to remain null",
    )
    return {
        "schema_version": 1,
        "core_protocol_sha256": _validate_sha(
            protocol.get("core_sha256"), "core protocol SHA-256"
        ),
        "core_protocol_hash_encoding": CORE_PROTOCOL_HASH_ENCODING,
        "normalized_json_pointer": RESERVED_SUMMARY_PIN_JSON_POINTER,
        "fit_full_protocol_sha256": _validate_sha(
            protocol.get("sha256"), "fit full protocol SHA-256"
        ),
        "fit_reserved_prompts_summary_sha256": None,
        "fit_reserved_prompts_summary_state": "null_before_reserved_materialization",
        "allowed_evaluation_transition": {
            "field_json_pointer": RESERVED_SUMMARY_PIN_JSON_POINTER,
            "from": None,
            "to": "literal_lowercase_sha256",
            "requires_same_core_protocol_sha256": True,
            "forbids_any_other_protocol_mutation": True,
            "model_refit_after_transition": False,
        },
    }


def validate_fit_protocol_lifecycle(value: Any) -> dict[str, Any]:
    """Validate the exact lifecycle record stored in a manifest or bundle."""
    record = mapping(value, "fit protocol lifecycle")
    expected_keys = {
        "schema_version",
        "core_protocol_sha256",
        "core_protocol_hash_encoding",
        "normalized_json_pointer",
        "fit_full_protocol_sha256",
        "fit_reserved_prompts_summary_sha256",
        "fit_reserved_prompts_summary_state",
        "allowed_evaluation_transition",
    }
    require(set(record) == expected_keys, "fit protocol lifecycle fields changed")
    core_sha = _validate_sha(
        record.get("core_protocol_sha256"), "fit lifecycle core protocol SHA-256"
    )
    full_sha = _validate_sha(
        record.get("fit_full_protocol_sha256"),
        "fit lifecycle full protocol SHA-256",
    )
    require(
        record.get("schema_version") == 1
        and record.get("core_protocol_hash_encoding")
        == CORE_PROTOCOL_HASH_ENCODING
        and record.get("normalized_json_pointer")
        == RESERVED_SUMMARY_PIN_JSON_POINTER
        and record.get("fit_reserved_prompts_summary_sha256") is None
        and record.get("fit_reserved_prompts_summary_state")
        == "null_before_reserved_materialization"
        and record.get("allowed_evaluation_transition")
        == {
            "field_json_pointer": RESERVED_SUMMARY_PIN_JSON_POINTER,
            "from": None,
            "to": "literal_lowercase_sha256",
            "requires_same_core_protocol_sha256": True,
            "forbids_any_other_protocol_mutation": True,
            "model_refit_after_transition": False,
        },
        "fit protocol lifecycle contract changed",
    )
    return {
        **dict(record),
        "core_protocol_sha256": core_sha,
        "fit_full_protocol_sha256": full_sha,
    }


def fit_protocol_state_record(fit_lifecycle_value: Any) -> dict[str, Any]:
    fit_lifecycle = validate_fit_protocol_lifecycle(fit_lifecycle_value)
    return {
        "schema_version": 1,
        "phase": "fit",
        "core_protocol_sha256": fit_lifecycle["core_protocol_sha256"],
        "fit_full_protocol_sha256": fit_lifecycle["fit_full_protocol_sha256"],
        "current_full_protocol_sha256": fit_lifecycle[
            "fit_full_protocol_sha256"
        ],
        "fit_reserved_prompts_summary_sha256": None,
        "current_reserved_prompts_summary_sha256": None,
        "transition": {
            "kind": "fit_null_state",
            "field_json_pointer": RESERVED_SUMMARY_PIN_JSON_POINTER,
            "from": None,
            "to": None,
            "applied": False,
        },
        "same_core_protocol_sha256_verified": True,
        "sole_append_only_field_transition_verified": False,
        "model_refit_after_transition": False,
    }


def validate_protocol_lifecycle_transition(
    fit_lifecycle_value: Any,
    current_protocol: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """Validate fit identity or the sole permitted post-materialization change."""
    require(phase in {"fit", "evaluate"}, "protocol lifecycle phase changed")
    fit_lifecycle = validate_fit_protocol_lifecycle(fit_lifecycle_value)
    current_full_sha = _validate_sha(
        current_protocol.get("sha256"), "current full protocol SHA-256"
    )
    current_core_sha = _validate_sha(
        current_protocol.get("core_sha256"), "current core protocol SHA-256"
    )
    protocol_value = mapping(current_protocol.get("value"), "current protocol value")
    require(
        core_protocol_sha256(protocol_value) == current_core_sha,
        "current core protocol SHA-256 is internally inconsistent",
    )
    raw_current_pin = mapping(protocol_value.get("pins"), "current protocol pins").get(
        RESERVED_SUMMARY_PIN_KEY
    )
    require(
        raw_current_pin == current_protocol.get("reserved_prompts_summary_sha256"),
        "current reserved prompts summary pin is internally inconsistent",
    )
    require(
        current_core_sha == fit_lifecycle["core_protocol_sha256"],
        "core protocol SHA-256 differs; mutation outside the append-only reserved summary pin is forbidden",
    )

    if phase == "fit":
        require(
            raw_current_pin is None,
            "fit requires reserved prompts summary pin to remain null",
        )
        require(
            current_full_sha == fit_lifecycle["fit_full_protocol_sha256"],
            "fit full protocol SHA-256 differs from the model-bound lifecycle",
        )
        return fit_protocol_state_record(fit_lifecycle)
    else:
        current_pin = _validate_sha(
            raw_current_pin, "evaluation reserved prompts summary pin"
        )
        require(
            current_full_sha != fit_lifecycle["fit_full_protocol_sha256"],
            "evaluation requires a distinct full protocol with a null-to-literal summary pin transition",
        )
        transition = {
            "kind": "null_to_literal_lowercase_sha256",
            "field_json_pointer": RESERVED_SUMMARY_PIN_JSON_POINTER,
            "from": None,
            "to": current_pin,
            "applied": True,
        }

    return {
        "schema_version": 1,
        "phase": phase,
        "core_protocol_sha256": current_core_sha,
        "fit_full_protocol_sha256": fit_lifecycle["fit_full_protocol_sha256"],
        "current_full_protocol_sha256": current_full_sha,
        "fit_reserved_prompts_summary_sha256": None,
        "current_reserved_prompts_summary_sha256": raw_current_pin,
        "transition": transition,
        "same_core_protocol_sha256_verified": True,
        "sole_append_only_field_transition_verified": phase == "evaluate",
        "model_refit_after_transition": False,
    }


def _gate_set(value: Any, label: str) -> dict[str, Any]:
    gate_set = mapping(value, label)
    support = mapping(gate_set.get("support"), f"{label} support")
    normalized_support: dict[str, int | float] = {}
    for key, raw in support.items():
        require(isinstance(key, str) and key.startswith("minimum_"), f"invalid {label} support key")
        if key.endswith("_fraction"):
            normalized_support[key] = finite(raw, f"{label} {key}")
            require(
                0.0 <= normalized_support[key] <= 1.0,
                f"{label} {key} must be a fraction",
            )
        else:
            normalized_support[key] = integer(raw, f"{label} {key}", minimum=1)
    require(bool(normalized_support), f"{label} support gates are empty")
    absolute = [_metric_gate(item, paired=False) for item in sequence(gate_set.get("absolute"), f"{label} absolute")]
    paired = [_metric_gate(item, paired=True) for item in sequence(gate_set.get("paired"), f"{label} paired")]
    identifiers = [item["id"] for item in (*absolute, *paired)]
    require(len(identifiers) == len(set(identifiers)), f"{label} gate IDs are duplicated")
    return {"support": normalized_support, "absolute": absolute, "paired": paired}


def _metric_gate(value: Any, *, paired: bool) -> dict[str, Any]:
    gate = mapping(value, "metric gate")
    result = {
        "id": nonempty_string(gate.get("id"), "gate ID"),
        "metric": nonempty_string(gate.get("metric"), "gate metric"),
        "bound": nonempty_string(gate.get("bound"), "gate bound"),
        "operator": nonempty_string(gate.get("operator"), "gate operator"),
        "value": finite(gate.get("value"), "gate value"),
    }
    require(
        result["bound"] in {"point", "bootstrap_lower", "bootstrap_upper"},
        "unsupported gate bound",
    )
    require(
        result["operator"]
        in {
            "minimum_inclusive",
            "minimum_exclusive",
            "maximum_inclusive",
            "maximum_exclusive",
        },
        "unsupported gate operator",
    )
    if paired:
        candidate = nonempty_string(gate.get("candidate"), "gate candidate")
        reference = nonempty_string(gate.get("reference"), "gate reference")
        require(
            candidate in VARIANTS and reference in VARIANTS and candidate != reference,
            "paired gate variants are invalid",
        )
        result.update({"candidate": candidate, "reference": reference})
    else:
        variant = nonempty_string(gate.get("variant"), "gate variant")
        require(variant in VARIANTS, "absolute gate variant is invalid")
        result["variant"] = variant
    return result


def normalize_source_protocol(
    source_value: Any,
    behavioral_value: Any,
    *,
    behavioral_sha256: str,
) -> dict[str, Any]:
    # Validation inputs intentionally omit prompt/report hashes: the v1 prompt
    # pin names the development cohort and must not reject a fresh validation
    # cohort.  Model/lens/runtime provenance remains fully enforced.
    return SOURCE_ANALYZER.validate_protocol(
        source_value,
        behavioral_protocol_value=behavioral_value,
        behavioral_protocol_sha256=behavioral_sha256,
    )


def _expected_repository(instance_id: str) -> str:
    require("__" in instance_id, f"invalid reserved instance ID: {instance_id}")
    owner, issue_name = instance_id.split("__", 1)
    require("-" in issue_name, f"invalid reserved instance ID: {instance_id}")
    repository = issue_name.rsplit("-", 1)[0]
    require(bool(owner) and bool(repository), f"invalid reserved instance ID: {instance_id}")
    return f"{owner}/{repository}"


def validate_reserved_evaluation_inputs(
    prompts_value: Any,
    *,
    protocol: Mapping[str, Any],
    cohort_path: Path,
    campaign_a_path: Path,
    campaign_b_path: Path,
    image_registry_path: Path,
    prompts_path: Path,
    prompts_summary_path: Path,
) -> dict[str, Any]:
    """Bind evaluation prompts to the one preregistered reserved cohort."""
    supplied_paths = {
        "cohort": cohort_path,
        "campaign_a": campaign_a_path,
        "campaign_b": campaign_b_path,
        "image_registry": image_registry_path,
    }
    for label, path in supplied_paths.items():
        require(
            path.is_file() and not path.is_symlink(),
            f"reserved {label} is not a regular non-symlink file: {path}",
        )
    observed_hashes = {
        label: sha256_file(path) for label, path in supplied_paths.items()
    }
    require(
        observed_hashes == protocol["reserved_pins"],
        "reserved cohort, campaign, or image registry differs from protocol pins",
    )
    observed_summary_sha = sha256_file(prompts_summary_path)
    reserved_summary_pin = protocol.get("reserved_prompts_summary_sha256")
    require(
        isinstance(reserved_summary_pin, str),
        "reserved prompts summary SHA-256 is not pinned; evaluation is forbidden",
    )
    require(
        observed_summary_sha == reserved_summary_pin,
        "reserved prompts summary differs from protocol pin",
    )

    resolved_cohort = cohort_path.expanduser().resolve(strict=True)
    resolved_images = image_registry_path.expanduser().resolve(strict=True)
    cohort, image_registry, canonical_campaigns = (
        COHORT_CHECKER.load_and_validate_bindings(resolved_cohort, resolved_images)
    )
    strict_prompts = COHORT_CHECKER.strict_json_file(
        prompts_path, "reserved prompt bundle"
    )
    require(
        prompts_value == strict_prompts,
        "in-memory reserved prompts differ from the hash-bound prompt file",
    )
    supplied_campaigns = [
        mapping(
            COHORT_CHECKER.strict_json_file(path, f"reserved campaign {index}"),
            f"reserved campaign {index}",
        )
        for index, path in enumerate((campaign_a_path, campaign_b_path))
    ]
    require(
        supplied_campaigns == canonical_campaigns,
        "supplied reserved campaigns differ from cohort-bound campaigns",
    )
    materialized_bundle = COHORT_CHECKER.validate_materialized_bundle(
        cohort,
        canonical_campaigns,
        cohort_path=resolved_cohort,
        prompts_path=prompts_path.expanduser().absolute(),
        summary_path=prompts_summary_path.expanduser().absolute(),
    )
    cohort_rows = [
        mapping(value, f"reserved cohort row {index}")
        for index, value in enumerate(sequence(cohort.get("cohorts"), "cohort rows"))
    ]
    require(len(cohort_rows) == 2, "reserved cohort must contain two campaigns")
    expected_instance_ids = [
        nonempty_string(value, "reserved instance ID")
        for value in sequence(cohort.get("instance_ids"), "reserved instance IDs")
    ]
    require(
        len(expected_instance_ids) == 20
        and len(expected_instance_ids) == len(set(expected_instance_ids)),
        "reserved cohort instance coverage changed",
    )
    images = mapping(image_registry.get("images"), "reserved image registry")
    require(
        set(images) == set(expected_instance_ids),
        "reserved image registry task coverage differs from cohort",
    )

    cohort_sha = observed_hashes["cohort"]
    task_bindings: dict[str, dict[str, Any]] = {}
    for index, (cohort_row, campaign) in enumerate(
        zip(cohort_rows, supplied_campaigns, strict=True)
    ):
        instance_ids = [
            nonempty_string(value, f"reserved campaign {index} instance ID")
            for value in sequence(
                campaign.get("instance_ids"), f"reserved campaign {index} instances"
            )
        ]
        require(
            cohort_row.get("instance_ids") == instance_ids,
            f"reserved campaign {index} task order differs from cohort",
        )
        cohort_id = nonempty_string(cohort_row.get("id"), "reserved cohort ID")
        campaign_sha = observed_hashes[f"campaign_{'a' if index == 0 else 'b'}"]
        require(
            cohort_row.get("campaign_sha256") == campaign_sha,
            f"reserved campaign {index} internal hash pin differs",
        )
        for instance_id in instance_ids:
            require(instance_id not in task_bindings, "reserved campaigns overlap")
            task_bindings[instance_id] = {
                "cohort_index": index,
                "cohort_id": cohort_id,
                "campaign_sha256": campaign_sha,
                "source_task_instance_ids": instance_ids,
                "repository": _expected_repository(instance_id),
            }
    require(
        list(task_bindings) == expected_instance_ids,
        "reserved campaign task order differs from combined cohort",
    )

    prompts = sequence(prompts_value, "reserved prompt bundle")
    require(bool(prompts), "reserved prompt bundle is empty")
    first_seen_tasks: list[str] = []
    observed_by_task: dict[str, list[int]] = defaultdict(list)
    declared_by_task: dict[str, list[int]] = {}
    for prompt_index, prompt_value in enumerate(prompts):
        prompt = mapping(prompt_value, f"reserved prompt {prompt_index}")
        prompt_id = nonempty_string(prompt.get("id"), "reserved prompt ID")
        metadata = mapping(prompt.get("metadata"), f"{prompt_id} metadata")
        task = mapping(metadata.get("task"), f"{prompt_id} task")
        selection = mapping(metadata.get("selection"), f"{prompt_id} selection")
        cohort_metadata = mapping(metadata.get("cohort"), f"{prompt_id} cohort")
        instance_id = nonempty_string(task.get("instance_id"), "prompt task ID")
        require(
            instance_id in task_bindings,
            f"evaluation prompt contains a non-reserved task: {instance_id}",
        )
        binding = task_bindings[instance_id]
        require(
            task.get("repo") == binding["repository"],
            f"reserved task repository differs: {instance_id}",
        )
        require(
            cohort_metadata.get("id") == binding["cohort_id"]
            and cohort_metadata.get("index") == binding["cohort_index"]
            and cohort_metadata.get("campaign_sha256")
            == binding["campaign_sha256"]
            and cohort_metadata.get("cohort_manifest_sha256") == cohort_sha
            and cohort_metadata.get("source_task_instance_ids")
            == binding["source_task_instance_ids"],
            f"reserved prompt cohort membership differs: {prompt_id}",
        )
        request_index = integer(
            selection.get("task_request_index"),
            "reserved task request index",
            minimum=1,
        )
        declared = [
            integer(value, "reserved probeable request index", minimum=1)
            for value in sequence(
                task.get("probeable_request_indices"),
                "reserved probeable request indices",
            )
        ]
        require(
            bool(declared)
            and declared == list(range(1, len(declared) + 1)),
            f"reserved task probeable request contract differs: {instance_id}",
        )
        if instance_id not in observed_by_task:
            first_seen_tasks.append(instance_id)
            declared_by_task[instance_id] = declared
        else:
            require(
                declared_by_task[instance_id] == declared,
                f"reserved request declaration changed within {instance_id}",
            )
        observed_by_task[instance_id].append(request_index)
    require(
        first_seen_tasks == expected_instance_ids,
        "evaluation prompt task IDs or order differ from reserved cohort",
    )
    for instance_id in expected_instance_ids:
        require(
            observed_by_task[instance_id] == declared_by_task[instance_id],
            f"evaluation prompts do not cover every reserved request: {instance_id}",
        )
    return {
        "hashes": observed_hashes,
        "task_count": len(expected_instance_ids),
        "prompt_count": len(prompts),
        "prompts_summary_sha256": observed_summary_sha,
        "materialized_bundle": materialized_bundle,
        "instance_ids": expected_instance_ids,
        "repositories": sorted(
            {binding["repository"] for binding in task_bindings.values()}
        ),
        "cohort_ids": [str(row["id"]) for row in cohort_rows],
        "internal_bindings_validated": True,
        "prompt_payload_and_summary_hash_bindings_validated": True,
        "prompt_task_repository_and_cohort_membership_validated": True,
    }


def action_of(prompt: Mapping[str, Any]) -> str | None:
    metadata = mapping(prompt.get("metadata"), "prompt metadata")
    labels = mapping(metadata.get("labels"), "prompt labels")
    action = mapping(labels.get("action"), "prompt action label")
    class_id = action.get("class_id")
    if action.get("status") != "available" or class_id not in SOURCE_ACTION_CLASSES:
        return None
    return str(class_id)


def offline_binary_labels(prompts: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Assign fail-closed future labels to every prompt in a complete bundle."""
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        metadata = mapping(prompt.get("metadata"), "prompt metadata")
        task = mapping(metadata.get("task"), "prompt task")
        by_task[nonempty_string(task.get("instance_id"), "task ID")].append(prompt)

    labels: dict[str, dict[str, Any]] = {}
    task_records: dict[str, Any] = {}
    for task_id, task_prompts in sorted(by_task.items()):
        ordered = sorted(
            task_prompts,
            key=lambda row: integer(
                mapping(
                    mapping(row.get("metadata"), "prompt metadata").get("selection"),
                    "prompt selection",
                ).get("task_request_index"),
                "task request index",
                minimum=1,
            ),
        )
        observed = [
            integer(
                mapping(
                    mapping(row.get("metadata"), "prompt metadata").get("selection"),
                    "selection",
                ).get("task_request_index"),
                "task request index",
                minimum=1,
            )
            for row in ordered
        ]
        first_task = mapping(
            mapping(ordered[0].get("metadata"), "prompt metadata").get("task"),
            "task metadata",
        )
        declared = [
            integer(item, "probeable request index", minimum=1)
            for item in sequence(
                first_task.get("probeable_request_indices"),
                "probeable request indices",
            )
        ]
        complete = (
            declared == list(range(1, len(declared) + 1))
            and observed == declared
            and len(observed) == len(set(observed))
        )
        actions = [action_of(row) for row in ordered]
        counts: Counter[str] = Counter()
        for start, prompt in enumerate(ordered):
            record: dict[str, Any] = {
                "status": "censored",
                "label": None,
                "reason": None,
                "horizon_requests": None,
                "target_request_index": None,
                "target_source_action": None,
                "offline_future_derived": True,
            }
            if not complete:
                record["reason"] = "incomplete_nonconsecutive_bundle"
            else:
                for future_index in range(start, len(actions)):
                    action = actions[future_index]
                    if action is None:
                        record["reason"] = "unknown_before_next_milestone"
                        break
                    if action == "inspect":
                        continue
                    require(action in MILESTONE_ACTIONS, f"unhandled action: {action}")
                    label = "edit" if action == "edit" else "check_or_finish"
                    record.update(
                        {
                            "status": "available",
                            "label": label,
                            "reason": None,
                            "horizon_requests": future_index - start,
                            "target_request_index": observed[future_index],
                            "target_source_action": action,
                        }
                    )
                    break
                else:
                    record["reason"] = "no_observed_future_milestone"
            prompt_id = nonempty_string(prompt.get("id"), "prompt ID")
            labels[prompt_id] = record
            counts[str(record["label"] or record["reason"])] += 1
        task_records[task_id] = {
            "repo": nonempty_string(first_task.get("repo"), "task repository"),
            "complete_consecutive_bundle": complete,
            "request_count": len(ordered),
            "immediate_action_support": dict(
                sorted(Counter(action or "unknown" for action in actions).items())
            ),
            "offline_label_support": dict(sorted(counts.items())),
        }
    return labels, {"tasks": task_records}


def extract_stable_inference_rows(
    prompt_bundle_value: Any,
    report_value: Any,
    *,
    source_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract stable causal features without requiring an action label."""
    prompts = sequence(prompt_bundle_value, "prompt bundle")
    report = mapping(report_value, "public report")
    SOURCE_ANALYZER._validate_report_provenance(report, protocol=source_protocol)
    experiments = sequence(report.get("experiments"), "report experiments")
    require(len(prompts) == len(experiments), "prompt/report row counts differ")
    prompt_ids = [nonempty_string(row.get("id"), "prompt ID") for row in prompts]
    experiment_ids = [
        nonempty_string(row.get("id"), "experiment ID") for row in experiments
    ]
    require(len(prompt_ids) == len(set(prompt_ids)), "prompt IDs are duplicated")
    require(prompt_ids == experiment_ids, "prompt/report IDs or order differ")

    history_by_id, history_coverage = SOURCE_ANALYZER._causal_history_features(
        prompts, source_protocol["class_ids"]
    )
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    stable_before_feature_count = 0
    for prompt, experiment in zip(prompts, experiments, strict=True):
        prompt_id = str(prompt["id"])
        require(
            experiment.get("prompt") == prompt.get("text")
            and experiment.get("prompt_token_ids") == prompt.get("token_ids")
            and experiment.get("metadata") == prompt.get("metadata"),
            f"{prompt_id} report payload is not bound to supplied prompt",
        )
        token_ids = sequence(prompt.get("token_ids"), f"{prompt_id} token IDs")
        require(bool(token_ids), f"{prompt_id} token IDs are empty")
        expected_position = len(token_ids) - 1
        require(
            experiment.get("capture_positions_resolved") == [expected_position],
            f"{prompt_id} was not captured only at the final prompt token",
        )
        scored = mapping(
            experiment.get("scored_vocabulary"), f"{prompt_id} scored vocabulary"
        )
        require(
            scored.get("token_ids") == prompt.get("score_token_ids"),
            f"{prompt_id} scored vocabulary differs from prompt contract",
        )
        stable, details = SOURCE_ANALYZER._numerically_stable(
            experiment, source_protocol["eligibility"]
        )
        if not stable:
            counts["numerically_unstable"] += 1
            exclusions.append(
                {
                    "row_id": prompt_id,
                    "reason": "numerically_unstable",
                    "details": details,
                }
            )
            continue
        stable_before_feature_count += 1

        metadata = mapping(prompt.get("metadata"), f"{prompt_id} metadata")
        selection = mapping(metadata.get("selection"), f"{prompt_id} selection")
        if (
            source_protocol["eligibility"]["require_primary"]
            and selection.get("primary_for_action_evaluation") is not True
        ):
            counts["not_primary_selection"] += 1
            exclusions.append(
                {"row_id": prompt_id, "reason": "not_primary_selection", "details": []}
            )
            continue
        history = history_by_id.get(prompt_id)
        if history is None:
            counts["causal_history_unavailable"] += 1
            exclusions.append(
                {
                    "row_id": prompt_id,
                    "reason": "causal_history_unavailable",
                    "details": ["complete consecutive probe bundle required"],
                }
            )
            continue

        task = mapping(metadata.get("task"), f"{prompt_id} task")
        request_index = integer(
            selection.get("task_request_index"),
            "task request index",
            minimum=1,
        )
        progress = [float(request_index), math.log1p(request_index)]
        lexical = SOURCE_ANALYZER._lexical_features(
            prompt,
            class_ids=source_protocol["class_ids"],
            token_ids_by_class=source_protocol["token_ids_by_class"],
            token_texts_by_class=source_protocol["token_texts_by_class"],
        )
        ordinary = SOURCE_ANALYZER._layer_class_features(
            experiment,
            layers=source_protocol["layers"],
            class_ids=source_protocol["class_ids"],
            token_ids_by_class=source_protocol["token_ids_by_class"],
            method="ordinary_logit",
            expected_token_position=expected_position,
        )
        public = SOURCE_ANALYZER._layer_class_features(
            experiment,
            layers=source_protocol["layers"],
            class_ids=source_protocol["class_ids"],
            token_ids_by_class=source_protocol["token_ids_by_class"],
            method="public_jacobian",
            expected_token_position=expected_position,
        )
        history_context = [*history, *lexical, *progress]
        require(
            len(public) == len(ordinary) == SOURCE_LAYER_COUNT * SOURCE_CLASS_COUNT,
            f"{prompt_id} source readout width changed",
        )
        require(len(history_context) == 32, f"{prompt_id} history context width changed")
        require(
            all(math.isfinite(item) for item in (*public, *ordinary, *history_context)),
            f"{prompt_id} feature source contains nonfinite values",
        )
        cohort = metadata.get("cohort")
        cohort_id = (
            str(cohort["id"])
            if isinstance(cohort, dict) and isinstance(cohort.get("id"), str)
            else "unspecified"
        )
        action = mapping(
            mapping(metadata.get("labels"), "prompt labels").get("action"),
            "prompt action",
        )
        rows.append(
            {
                "row_id": prompt_id,
                "task_id": nonempty_string(task.get("instance_id"), "task ID"),
                "repo": nonempty_string(task.get("repo"), "task repository"),
                "cohort_id": cohort_id,
                "task_request_index": request_index,
                "checkpoint_ordinal": selection.get("checkpoint_ordinal"),
                "current_action_label_status": str(action.get("status", "unavailable")),
                "current_action_class_id": (
                    str(action["class_id"])
                    if action.get("status") == "available"
                    and action.get("class_id") in SOURCE_ACTION_CLASSES
                    else None
                ),
                "public_jacobian": public,
                "ordinary_logit": ordinary,
                "history_context": history_context,
            }
        )
    return {
        "rows": rows,
        "eligibility": {
            "prompt_count": len(prompts),
            "numerically_stable_before_feature_requirements": stable_before_feature_count,
            "inference_eligible_stable_row_count": len(rows),
            "excluded_row_count": len(exclusions),
            "exclusion_counts": dict(sorted(counts.items())),
            "exclusions": exclusions,
            "current_action_label_not_required": True,
            "causal_history": history_coverage,
        },
    }


def compact_layer_shape(values: Sequence[float]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64).reshape(
        SOURCE_LAYER_COUNT, SOURCE_CLASS_COUNT
    )
    x = np.arange(SOURCE_LAYER_COUNT, dtype=np.float64)
    centered = x - x.mean()
    slopes = centered @ matrix / float(centered @ centered)
    result = np.concatenate(
        [
            matrix.mean(axis=0),
            matrix.std(axis=0),
            matrix.min(axis=0),
            matrix.max(axis=0),
            matrix[:6].mean(axis=0),
            matrix[6:18].mean(axis=0),
            matrix[18:].mean(axis=0),
            slopes,
            matrix[-1] - matrix[0],
            np.argmax(matrix, axis=0).astype(np.float64) / 23.0,
        ]
    )
    require(result.shape == (40,), "compact layer summary width changed")
    return result


def build_feature_rows(
    stable_rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
    *,
    ema_alpha: float,
) -> list[dict[str, Any]]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in stable_rows:
        by_task[str(row["task_id"])].append(row)
    result_by_id: dict[str, dict[str, Any]] = {}
    for task_id, task_rows in sorted(by_task.items()):
        ordered = sorted(task_rows, key=lambda row: int(row["task_request_index"]))
        observed = [int(row["task_request_index"]) for row in ordered]
        require(len(observed) == len(set(observed)), f"duplicate stable request index in {task_id}")
        previous: dict[str, np.ndarray] = {}
        ema: dict[str, np.ndarray] = {}
        previous_index: int | None = None
        for row in ordered:
            request_index = int(row["task_request_index"])
            no_previous = previous_index is None
            gap = 0.0 if no_previous else math.log1p(request_index - int(previous_index))
            context = np.asarray(row["history_context"], dtype=np.float64)
            features: dict[str, list[float]] = {}
            for variant, source_name in (
                ("j_compact", "public_jacobian"),
                ("l_compact", "ordinary_logit"),
            ):
                current = np.asarray(row[source_name], dtype=np.float64)
                delta = (
                    np.zeros_like(current)
                    if source_name not in previous
                    else current - previous[source_name]
                )
                deviation = (
                    np.zeros_like(current)
                    if source_name not in ema
                    else current - ema[source_name]
                )
                feature = np.concatenate(
                    [
                        compact_layer_shape(current),
                        compact_layer_shape(delta),
                        compact_layer_shape(deviation),
                        context,
                        np.asarray([gap, float(no_previous)], dtype=np.float64),
                    ]
                )
                require(
                    feature.shape == (FEATURE_WIDTH,) and np.all(np.isfinite(feature)),
                    f"{row['row_id']} {variant} feature vector is invalid",
                )
                features[variant] = feature.tolist()
                previous[source_name] = current
                ema[source_name] = (
                    current.copy()
                    if source_name not in ema
                    else ema_alpha * current + (1.0 - ema_alpha) * ema[source_name]
                )
            previous_index = request_index
            assignment = mapping(assignments.get(str(row["row_id"])), "offline assignment")
            label_status = str(assignment["status"])
            label = str(assignment["label"]) if label_status == "available" else None
            event_id = (
                f"{row['task_id']}::{assignment['target_request_index']}::{label}"
                if label is not None
                else None
            )
            result_by_id[str(row["row_id"])] = {
                **{
                    key: row[key]
                    for key in (
                        "row_id",
                        "task_id",
                        "repo",
                        "cohort_id",
                        "task_request_index",
                        "checkpoint_ordinal",
                        "current_action_label_status",
                        "current_action_class_id",
                    )
                },
                "label_status": label_status,
                "label": label,
                "label_censor_reason": assignment.get("reason"),
                "label_offline_future_derived": True,
                "milestone_horizon_requests": assignment.get("horizon_requests"),
                "milestone_target_request_index": assignment.get("target_request_index"),
                "milestone_target_source_action": assignment.get("target_source_action"),
                "event_id": event_id,
                "features": features,
            }
    return [result_by_id[str(row["row_id"])] for row in stable_rows]


def labeled_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("label_status") == "available"]


def matrix_for(rows: Sequence[Mapping[str, Any]], variant: str) -> np.ndarray:
    values = np.asarray(
        [mapping(row.get("features"), "row features")[variant] for row in rows],
        dtype=np.float64,
    )
    require(
        values.shape == (len(rows), FEATURE_WIDTH) and np.all(np.isfinite(values)),
        f"{variant} feature matrix is invalid",
    )
    return values


def identifiers_for(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([str(row[key]) for row in rows])


def task_event_prefix_weights(
    tasks: np.ndarray,
    events: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    rebalance_classes: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(len(tasks) == len(events) and len(tasks) > 0, "weight inputs are invalid")
    weights = np.zeros(len(tasks), dtype=np.float64)
    unique_tasks = sorted(set(tasks.tolist()))
    for task_id in unique_tasks:
        task_indices = np.flatnonzero(tasks == task_id)
        unique_events = sorted(set(events[task_indices].tolist()))
        require(bool(unique_events), f"task {task_id} has no physical events")
        for event_id in unique_events:
            indices = task_indices[events[task_indices] == event_id]
            weights[indices] = 1.0 / (
                len(unique_tasks) * len(unique_events) * len(indices)
            )
    require(np.all(weights > 0.0), "base weights contain zero mass")
    weights /= weights.sum()
    pre_class_mass: dict[str, float] = {}
    if labels is not None:
        pre_class_mass = {
            class_id: float(weights[labels == class_id].sum()) for class_id in CLASSES
        }
    pre_task_mass = {
        task_id: float(weights[tasks == task_id].sum()) for task_id in unique_tasks
    }
    if rebalance_classes:
        require(labels is not None and len(labels) == len(tasks), "class rebalance requires labels")
        for class_id in CLASSES:
            mask = labels == class_id
            mass = float(weights[mask].sum())
            require(mass > 0.0, f"training split lacks class {class_id}")
            weights[mask] *= 1.0 / (len(CLASSES) * mass)
        weights /= weights.sum()
    post_class_mass: dict[str, float] = {}
    if labels is not None:
        post_class_mass = {
            class_id: float(weights[labels == class_id].sum()) for class_id in CLASSES
        }
    post_task_mass = {
        task_id: float(weights[tasks == task_id].sum()) for task_id in unique_tasks
    }
    diagnostics = {
        "algorithm": (
            "equal_task_then_equal_event_within_task_then_equal_prefix_within_event"
            + ("_then_binary_class_rebalance" if rebalance_classes else "")
        ),
        "normalization": "sum_to_one",
        "row_count": len(weights),
        "task_count": len(unique_tasks),
        "event_count": len(set(events.tolist())),
        "pre_class_mass": pre_class_mass,
        "post_class_mass": post_class_mass,
        "pre_task_mass_minimum": min(pre_task_mass.values()),
        "pre_task_mass_maximum": max(pre_task_mass.values()),
        "post_task_mass_minimum": min(post_task_mass.values()),
        "post_task_mass_maximum": max(post_task_mass.values()),
        "weight_float64_sha256": sha256_bytes(
            np.asarray(weights, dtype="<f8").tobytes(order="C")
        ),
    }
    return weights, diagnostics


def fit_model(
    x: np.ndarray,
    labels: np.ndarray,
    tasks: np.ndarray,
    events: np.ndarray,
    *,
    model_contract: Mapping[str, Any],
    seed: int,
) -> tuple[ExtraTreesClassifier, dict[str, Any]]:
    _, ExtraTreesClassifier = _ml_dependencies()
    weights, diagnostics = task_event_prefix_weights(
        tasks, events, labels=labels, rebalance_classes=True
    )
    model = ExtraTreesClassifier(
        bootstrap=False,
        ccp_alpha=0.0,
        class_weight=None,
        criterion="gini",
        max_depth=None,
        n_estimators=int(model_contract["n_estimators"]),
        min_samples_leaf=int(model_contract["min_samples_leaf"]),
        max_features=float(model_contract["max_features"]),
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        monotonic_cst=None,
        random_state=seed,
        n_jobs=int(model_contract["n_jobs"]),
        oob_score=False,
        verbose=0,
        warm_start=False,
    )
    model.fit(x, labels, sample_weight=weights * len(labels))
    require(set(model.classes_.tolist()) == set(CLASSES), "fitted model classes changed")
    diagnostics.update(
        {
            "sample_weight_scale_supplied_to_sklearn": len(labels),
            "scaled_weight_sum": float((weights * len(labels)).sum()),
            "row_identity_sha256": sha256_bytes(
                "\n".join(
                    f"{task}\0{event}\0{label}"
                    for task, event, label in zip(tasks, events, labels, strict=True)
                ).encode("utf-8")
            ),
            "estimator_get_params": model.get_params(deep=False),
        }
    )
    return model, diagnostics


def aligned_probabilities(model: ExtraTreesClassifier, x: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    result = np.zeros((len(x), len(CLASSES)), dtype=np.float64)
    for source_index, class_id in enumerate(model.classes_):
        require(str(class_id) in CLASSES, "model emitted an unknown class")
        result[:, CLASSES.index(str(class_id))] = raw[:, source_index]
    require(
        result.shape == (len(x), len(CLASSES))
        and np.all(np.isfinite(result))
        and np.all(result >= 0.0)
        and np.allclose(result.sum(axis=1), 1.0),
        "model probabilities are invalid",
    )
    return result


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    require(temperature > 0.0, "temperature must be positive")
    if temperature == 1.0:
        return probabilities.copy()
    logits = np.log(np.clip(probabilities, 1e-300, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def probability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    tasks: np.ndarray,
    events: np.ndarray,
    *,
    ece_bins: int = 10,
) -> dict[str, Any]:
    if not len(labels):
        return {
            "row_count": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "per_class_recall": {class_id: None for class_id in CLASSES},
            "recall_edit": None,
            "recall_check_or_finish": None,
            "multiclass_negative_log_likelihood": None,
            "multiclass_brier": None,
            "top_label_ece": None,
        }
    weights, _ = task_event_prefix_weights(tasks, events)
    y = np.asarray([CLASSES.index(str(label)) for label in labels], dtype=np.int64)
    predicted_indices = np.argmax(probabilities, axis=1)
    predicted = np.asarray(CLASSES)[predicted_indices]
    correct = predicted == labels
    recalls: dict[str, float | None] = {}
    for class_id in CLASSES:
        mask = labels == class_id
        mass = float(weights[mask].sum())
        recalls[class_id] = (
            float(np.sum(weights[mask] * correct[mask]) / mass) if mass else None
        )
    available_recalls = [value for value in recalls.values() if value is not None]
    true_probability = np.clip(
        probabilities[np.arange(len(y)), y], 1e-300, 1.0
    )
    one_hot = np.eye(len(CLASSES), dtype=np.float64)[y]
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        mask = (confidence >= lower) & (
            confidence < upper if bin_index < ece_bins - 1 else confidence <= upper
        )
        mass = float(weights[mask].sum())
        if mass:
            bin_accuracy = float(np.sum(weights[mask] * correct[mask]) / mass)
            bin_confidence = float(np.sum(weights[mask] * confidence[mask]) / mass)
            ece += mass * abs(bin_accuracy - bin_confidence)
    return {
        "row_count": len(labels),
        "accuracy": float(np.sum(weights * correct)),
        "balanced_accuracy": (
            float(np.mean(available_recalls))
            if len(available_recalls) == len(CLASSES)
            else None
        ),
        "per_class_recall": recalls,
        "recall_edit": recalls["edit"],
        "recall_check_or_finish": recalls["check_or_finish"],
        "multiclass_negative_log_likelihood": float(
            -np.sum(weights * np.log(true_probability))
        ),
        "multiclass_brier": float(
            np.sum(weights * np.sum((probabilities - one_hot) ** 2, axis=1))
        ),
        "top_label_ece": ece,
        "weighting": "equal_task_then_equal_event_then_equal_prefix",
    }


HORIZON_BUCKETS = ("h0", "h1_2", "h3_5", "h6_10", "h11_plus")


def horizon_bucket(horizon: int) -> str:
    require(horizon >= 0, "milestone horizon must be nonnegative")
    if horizon == 0:
        return "h0"
    if horizon <= 2:
        return "h1_2"
    if horizon <= 5:
        return "h3_5"
    if horizon <= 10:
        return "h6_10"
    return "h11_plus"


def horizon_stratified_metrics(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report performance decay as the labeled milestone moves farther away."""
    result: dict[str, Any] = {}
    for bucket in HORIZON_BUCKETS:
        rows = [
            row
            for row in predictions
            if row.get("metric_evaluable") is True
            and horizon_bucket(integer(row.get("milestone_horizon_requests"), "milestone horizon"))
            == bucket
        ]
        class_support = Counter(str(row["label"]) for row in rows)
        support = {
            "rows": len(rows),
            "physical_events": len({str(row["event_id"]) for row in rows}),
            "tasks": len({str(row["task_id"]) for row in rows}),
            "repositories": len({str(row["repo"]) for row in rows}),
            "rows_per_class": {
                class_id: int(class_support[class_id]) for class_id in CLASSES
            },
            "events_per_class": {
                class_id: len(
                    {
                        str(row["event_id"])
                        for row in rows
                        if row["label"] == class_id
                    }
                )
                for class_id in CLASSES
            },
        }
        if rows:
            metrics = probability_metrics(
                identifiers_for(rows, "label"),
                np.asarray([row["probabilities"] for row in rows], dtype=np.float64),
                identifiers_for(rows, "task_id"),
                identifiers_for(rows, "event_id"),
            )
        else:
            metrics = probability_metrics(
                np.asarray([]),
                np.zeros((0, len(CLASSES)), dtype=np.float64),
                np.asarray([]),
                np.asarray([]),
            )
        result[bucket] = {"support": support, "probability_metrics": metrics}
    return result


def _prediction_record(
    row: Mapping[str, Any],
    probability: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    values = [float(item) for item in probability]
    predicted = CLASSES[int(np.argmax(probability))]
    confidence = max(values)
    return {
        **{
            key: row.get(key)
            for key in (
                "row_id",
                "task_id",
                "repo",
                "cohort_id",
                "task_request_index",
                "checkpoint_ordinal",
                "current_action_label_status",
                "current_action_class_id",
                "label_status",
                "label",
                "label_censor_reason",
                "label_offline_future_derived",
                "milestone_horizon_requests",
                "milestone_target_request_index",
                "milestone_target_source_action",
                "event_id",
            )
        },
        "class_ids": list(CLASSES),
        "probabilities": values,
        "prediction": predicted,
        "confidence": confidence,
        "confidence_threshold": threshold,
        "accepted": bool(confidence >= threshold),
        "metric_evaluable": row.get("label_status") == "available",
    }


def support_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    eligibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evaluable = labeled_rows(rows)
    events = {str(row["event_id"]) for row in evaluable}
    all_tasks = {str(row["task_id"]) for row in rows}
    all_repos = {str(row["repo"]) for row in rows}
    prompt_count = (
        int(eligibility["prompt_count"])
        if eligibility is not None and "prompt_count" in eligibility
        else len(rows)
    )
    stable_before_features = (
        int(eligibility["numerically_stable_before_feature_requirements"])
        if eligibility is not None
        and "numerically_stable_before_feature_requirements" in eligibility
        else len(rows)
    )
    stability_retention = (
        stable_before_features / prompt_count if prompt_count else 0.0
    )
    return {
        "inference": {
            "replayed_prompt_rows": prompt_count,
            "numerically_stable_rows_before_feature_requirements": stable_before_features,
            "numerical_stability_retention_fraction": stability_retention,
            "numerical_stability_retention_ppm": int(
                math.floor(stability_retention * 1_000_000 + 1e-12)
            ),
            "stable_emitted_rows": len(rows),
            "tasks": len(all_tasks),
            "repositories": len(all_repos),
            "current_action_label_available_rows": sum(
                row.get("current_action_label_status") == "available" for row in rows
            ),
            "current_action_label_unavailable_rows": sum(
                row.get("current_action_label_status") != "available" for row in rows
            ),
        },
        "evaluation": {
            "future_label_available_rows": len(evaluable),
            "future_label_censored_rows": len(rows) - len(evaluable),
            "future_label_evaluable_fraction_of_emissions": (
                len(evaluable) / len(rows) if rows else 0.0
            ),
            "physical_events": len(events),
            "tasks": len({str(row["task_id"]) for row in evaluable}),
            "repositories": len({str(row["repo"]) for row in evaluable}),
            "rows_per_class": dict(
                sorted(Counter(str(row["label"]) for row in evaluable).items())
            ),
            "events_per_class": {
                class_id: len(
                    {
                        str(row["event_id"])
                        for row in evaluable
                        if row["label"] == class_id
                    }
                )
                for class_id in CLASSES
            },
            "tasks_per_class": {
                class_id: len(
                    {
                        str(row["task_id"])
                        for row in evaluable
                        if row["label"] == class_id
                    }
                )
                for class_id in CLASSES
            },
            "repositories_per_class": {
                class_id: len(
                    {
                        str(row["repo"])
                        for row in evaluable
                        if row["label"] == class_id
                    }
                )
                for class_id in CLASSES
            },
            "censor_reasons": dict(
                sorted(
                    Counter(
                        str(row["label_censor_reason"])
                        for row in rows
                        if row.get("label_status") != "available"
                    ).items()
                )
            ),
        },
    }


def evaluate_probabilities(
    rows: Sequence[Mapping[str, Any]],
    probabilities_by_variant: Mapping[str, np.ndarray],
    *,
    threshold: float,
) -> dict[str, Any]:
    evaluable_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["label_status"] == "available"],
        dtype=np.int64,
    )
    evaluable = [rows[int(index)] for index in evaluable_indices]
    labels = identifiers_for(evaluable, "label") if evaluable else np.asarray([])
    tasks = identifiers_for(evaluable, "task_id") if evaluable else np.asarray([])
    events = identifiers_for(evaluable, "event_id") if evaluable else np.asarray([])
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        probabilities = probabilities_by_variant[variant]
        require(
            probabilities.shape == (len(rows), len(CLASSES)),
            f"{variant} probability matrix shape changed",
        )
        predictions = [
            _prediction_record(row, probability, threshold=threshold)
            for row, probability in zip(rows, probabilities, strict=True)
        ]
        metrics = probability_metrics(
            labels,
            probabilities[evaluable_indices],
            tasks,
            events,
        )
        result[variant] = {
            "inference_row_count": len(rows),
            "accepted_inference_row_count": sum(row["accepted"] for row in predictions),
            "inference_coverage": (
                sum(row["accepted"] for row in predictions) / len(rows) if rows else 0.0
            ),
            "label_evaluable_row_count": len(evaluable),
            "label_evaluable_fraction_of_inference": (
                len(evaluable) / len(rows) if rows else 0.0
            ),
            "probability_metrics_on_label_evaluable_rows": metrics,
            "horizon_stratified_metrics": horizon_stratified_metrics(predictions),
            "predictions": predictions,
        }
    return result


def leave_one_repository_out(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    rows = list(labeled_rows(rows))
    require(bool(rows), "development cohort has no evaluable rows")
    labels = identifiers_for(rows, "label")
    tasks = identifiers_for(rows, "task_id")
    events = identifiers_for(rows, "event_id")
    repositories = identifiers_for(rows, "repo")
    repositories_in_order = sorted(set(repositories.tolist()))
    require(len(repositories_in_order) >= 2, "LORO requires at least two repositories")
    probabilities = {
        variant: np.zeros((len(rows), len(CLASSES)), dtype=np.float64)
        for variant in VARIANTS
    }
    covered = np.zeros(len(rows), dtype=bool)
    folds: list[dict[str, Any]] = []
    for outer_index, heldout in enumerate(repositories_in_order):
        train = repositories != heldout
        evaluation = repositories == heldout
        require(np.any(train) and np.any(evaluation), f"invalid outer fold: {heldout}")
        require(
            set(labels[train].tolist()) == set(CLASSES),
            f"outer training split for {heldout} lacks a class",
        )
        fold_seed = int(protocol["model"]["random_state"]) + outer_index * 1000 + 900
        fold_record: dict[str, Any] = {
            "heldout_repository": heldout,
            "training_rows": int(train.sum()),
            "evaluation_rows": int(evaluation.sum()),
            "random_state": fold_seed,
            "heldout_labels_used_for_fit_or_selection": False,
            "training_weight_diagnostics": {},
        }
        for variant in VARIANTS:
            model, diagnostics = fit_model(
                matrix_for([rows[index] for index in np.flatnonzero(train)], variant),
                labels[train],
                tasks[train],
                events[train],
                model_contract=protocol["model"],
                seed=fold_seed,
            )
            fold_record["training_weight_diagnostics"][variant] = diagnostics
            raw = aligned_probabilities(
                model,
                matrix_for([rows[index] for index in np.flatnonzero(evaluation)], variant),
            )
            probabilities[variant][evaluation] = apply_temperature(
                raw, float(protocol["temperature"])
            )
        covered[evaluation] = True
        folds.append(fold_record)
    require(np.all(covered), "LORO did not cover every evaluable development row")
    results: dict[str, Any] = {}
    for variant in VARIANTS:
        predictions = [
            _prediction_record(row, probability, threshold=float(protocol["threshold"]))
            for row, probability in zip(rows, probabilities[variant], strict=True)
        ]
        results[variant] = {
            "probability_metrics": probability_metrics(
                labels, probabilities[variant], tasks, events
            ),
            "horizon_stratified_metrics": horizon_stratified_metrics(predictions),
            "predictions": predictions,
        }
    return {
        "algorithm": "leave_one_repository_out",
        "fold_count": len(folds),
        "all_evaluable_rows_covered_once": True,
        "folds": folds,
        "results": results,
    }


def bootstrap_indices(
    repositories: np.ndarray,
    tasks: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_repo: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (repository, task) in enumerate(zip(repositories, tasks, strict=True)):
        by_repo[str(repository)][str(task)].append(index)
    repository_ids = sorted(by_repo)
    require(bool(repository_ids), "bootstrap has no repositories")
    indices: list[int] = []
    synthetic_repositories: list[str] = []
    synthetic_tasks: list[str] = []
    for repo_draw, source_repo_index in enumerate(
        rng.integers(0, len(repository_ids), size=len(repository_ids))
    ):
        source_repository = by_repo[repository_ids[int(source_repo_index)]]
        task_ids = sorted(source_repository)
        for task_draw, source_task_index in enumerate(
            rng.integers(0, len(task_ids), size=len(task_ids))
        ):
            source_indices = source_repository[task_ids[int(source_task_index)]]
            for source_index in source_indices:
                indices.append(source_index)
                synthetic_repositories.append(f"r{repo_draw}")
                synthetic_tasks.append(f"r{repo_draw}-t{task_draw}")
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(synthetic_repositories),
        np.asarray(synthetic_tasks),
    )


def hierarchical_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    predictions_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
    minimum_valid_fraction: float,
) -> dict[str, Any]:
    rows = list(rows)
    require(bool(rows), "bootstrap requires evaluable rows")
    labels = identifiers_for(rows, "label")
    tasks = identifiers_for(rows, "task_id")
    events = identifiers_for(rows, "event_id")
    repositories = identifiers_for(rows, "repo")
    probabilities = {
        variant: np.asarray(
            [prediction["probabilities"] for prediction in predictions_by_variant[variant]],
            dtype=np.float64,
        )
        for variant in VARIANTS
    }
    for variant in VARIANTS:
        require(
            [str(prediction["row_id"]) for prediction in predictions_by_variant[variant]]
            == [str(row["row_id"]) for row in rows],
            f"{variant} bootstrap predictions are not paired to rows",
        )
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "multiclass_negative_log_likelihood",
        "multiclass_brier",
        "top_label_ece",
    )
    draws = {
        variant: {metric: [] for metric in metric_names} for variant in VARIANTS
    }
    paired = {metric: [] for metric in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        indices, _, synthetic_tasks = bootstrap_indices(repositories, tasks, rng)
        synthetic_events = np.asarray(
            [
                f"{synthetic_tasks[offset]}::{events[source_index]}"
                for offset, source_index in enumerate(indices)
            ]
        )
        current: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            metrics = probability_metrics(
                labels[indices],
                probabilities[variant][indices],
                synthetic_tasks,
                synthetic_events,
            )
            current[variant] = metrics
            for metric in metric_names:
                value = metrics[metric]
                if value is not None and math.isfinite(float(value)):
                    draws[variant][metric].append(float(value))
        for metric in metric_names:
            left = current["j_compact"][metric]
            right = current["l_compact"][metric]
            if left is not None and right is not None:
                paired[metric].append(float(left) - float(right))

    tail = (1.0 - confidence_level) / 2.0

    def interval(values: Sequence[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if not len(array):
            return {
                "valid_samples": 0,
                "valid_fraction": 0.0,
                "minimum_valid_fraction_met": False,
                "lower": None,
                "median": None,
                "upper": None,
            }
        return {
            "valid_samples": len(array),
            "valid_fraction": len(array) / samples,
            "minimum_valid_fraction_met": len(array) / samples >= minimum_valid_fraction,
            "lower": float(np.quantile(array, tail)),
            "median": float(np.quantile(array, 0.5)),
            "upper": float(np.quantile(array, 1.0 - tail)),
        }

    return {
        "algorithm": "repository_then_task_cluster_percentile_conditional_on_frozen_predictions",
        "samples": samples,
        "seed": seed,
        "confidence_level": confidence_level,
        "minimum_valid_fraction": minimum_valid_fraction,
        "models_refit_inside_bootstrap": False,
        "row_resampling_forbidden": True,
        "intervals": {
            variant: {
                metric: interval(values) for metric, values in variant_draws.items()
            }
            for variant, variant_draws in draws.items()
        },
        "paired_differences": {
            "j_compact_minus_l_compact": {
                metric: interval(values) for metric, values in paired.items()
            }
        },
    }


def _support_value(support: Mapping[str, Any], key: str) -> int | float | None:
    evaluation = mapping(support.get("evaluation"), "evaluation support")
    inference = mapping(support.get("inference"), "inference support")
    aliases = {
        "minimum_rows": evaluation.get("future_label_available_rows"),
        "minimum_evaluable_rows": evaluation.get("future_label_available_rows"),
        "minimum_events": evaluation.get("physical_events"),
        "minimum_physical_events": evaluation.get("physical_events"),
        "minimum_tasks": evaluation.get("tasks"),
        "minimum_repositories": evaluation.get("repositories"),
        "minimum_inference_rows": inference.get("stable_emitted_rows"),
        "minimum_inference_tasks": inference.get("tasks"),
        "minimum_inference_repositories": inference.get("repositories"),
        "minimum_replay_certification_retention_ppm": inference.get(
            "numerical_stability_retention_ppm"
        ),
        "minimum_numerical_stability_retention_ppm": inference.get(
            "numerical_stability_retention_ppm"
        ),
        "minimum_numerical_stability_retention_fraction": inference.get(
            "numerical_stability_retention_fraction"
        ),
        "minimum_edit_events": mapping(evaluation.get("events_per_class"), "event class support").get("edit"),
        "minimum_check_or_finish_events": mapping(evaluation.get("events_per_class"), "event class support").get("check_or_finish"),
        "minimum_edit_tasks": mapping(evaluation.get("tasks_per_class"), "task class support").get("edit"),
        "minimum_check_or_finish_tasks": mapping(evaluation.get("tasks_per_class"), "task class support").get("check_or_finish"),
        "minimum_edit_repositories": mapping(
            evaluation.get("repositories_per_class"), "repository class support"
        ).get("edit"),
        "minimum_check_or_finish_repositories": mapping(
            evaluation.get("repositories_per_class"), "repository class support"
        ).get("check_or_finish"),
    }
    value = aliases.get(key)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _compare_gate(observed: float, operator: str, threshold: float) -> bool:
    if operator == "minimum_inclusive":
        return observed >= threshold
    if operator == "minimum_exclusive":
        return observed > threshold
    if operator == "maximum_inclusive":
        return observed <= threshold
    if operator == "maximum_exclusive":
        return observed < threshold
    raise AssertionError(operator)


def evaluate_gates(
    gate_contract: Mapping[str, Any],
    *,
    support: Mapping[str, Any],
    results: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for key, minimum in mapping(gate_contract.get("support"), "support gates").items():
        observed = _support_value(support, key)
        passed = observed is not None and float(observed) >= float(minimum)
        records.append(
            {
                "id": key,
                "kind": "support",
                "observed": observed,
                "operator": "minimum_inclusive",
                "threshold": minimum,
                "passed": passed,
            }
        )
    for gate in sequence(gate_contract.get("absolute"), "absolute gates"):
        variant = str(gate["variant"])
        metric = str(gate["metric"])
        point_metrics = mapping(
            mapping(results.get(variant), f"{variant} result").get(
                "probability_metrics_on_label_evaluable_rows",
                mapping(results.get(variant), f"{variant} result").get("probability_metrics"),
            ),
            f"{variant} metrics",
        )
        bootstrap_valid = True
        bootstrap_valid_fraction = None
        if gate["bound"] == "point":
            observed = point_metrics.get(metric)
        else:
            interval = mapping(
                mapping(bootstrap["intervals"].get(variant), "variant intervals").get(metric),
                "metric interval",
            )
            bootstrap_valid = interval.get("minimum_valid_fraction_met") is True
            bootstrap_valid_fraction = interval.get("valid_fraction")
            observed = (
                interval["lower" if gate["bound"] == "bootstrap_lower" else "upper"]
                if bootstrap_valid
                else None
            )
        passed = observed is not None and _compare_gate(
            float(observed), str(gate["operator"]), float(gate["value"])
        )
        records.append(
            {
                **gate,
                "kind": "absolute",
                "observed": observed,
                "bootstrap_minimum_valid_fraction_met": bootstrap_valid,
                "bootstrap_valid_fraction": bootstrap_valid_fraction,
                "passed": passed,
            }
        )
    paired_intervals = mapping(
        bootstrap.get("paired_differences"), "paired differences"
    )
    for gate in sequence(gate_contract.get("paired"), "paired gates"):
        candidate = str(gate["candidate"])
        reference = str(gate["reference"])
        metric = str(gate["metric"])
        candidate_metrics = mapping(
            mapping(results[candidate], "candidate result").get(
                "probability_metrics_on_label_evaluable_rows",
                mapping(results[candidate], "candidate result").get("probability_metrics"),
            ),
            "candidate metrics",
        )
        reference_metrics = mapping(
            mapping(results[reference], "reference result").get(
                "probability_metrics_on_label_evaluable_rows",
                mapping(results[reference], "reference result").get("probability_metrics"),
            ),
            "reference metrics",
        )
        comparison_id = f"{candidate}_minus_{reference}"
        bootstrap_valid = True
        bootstrap_valid_fraction = None
        if gate["bound"] == "point":
            left = candidate_metrics.get(metric)
            right = reference_metrics.get(metric)
            observed = None if left is None or right is None else float(left) - float(right)
        else:
            interval = mapping(
                mapping(paired_intervals.get(comparison_id), comparison_id).get(metric),
                "paired metric interval",
            )
            bootstrap_valid = interval.get("minimum_valid_fraction_met") is True
            bootstrap_valid_fraction = interval.get("valid_fraction")
            observed = (
                interval["lower" if gate["bound"] == "bootstrap_lower" else "upper"]
                if bootstrap_valid
                else None
            )
        passed = observed is not None and _compare_gate(
            float(observed), str(gate["operator"]), float(gate["value"])
        )
        records.append(
            {
                **gate,
                "kind": "paired",
                "observed": observed,
                "bootstrap_minimum_valid_fraction_met": bootstrap_valid,
                "bootstrap_valid_fraction": bootstrap_valid_fraction,
                "passed": passed,
            }
        )
    return _canonicalize_gate_evidence(
        {
        "passed": bool(records) and all(record["passed"] for record in records),
        "passed_count": sum(record["passed"] for record in records),
        "gate_count": len(records),
        "records": records,
        }
    )


def _canonicalize_gate_evidence(value: Any) -> Any:
    """Remove insignificant platform/thread float variance from gate evidence."""
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        require(math.isfinite(value), "gate evidence contains a nonfinite float")
        rounded = round(value, GATE_EVIDENCE_DECIMAL_PLACES)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, dict):
        return {str(key): _canonicalize_gate_evidence(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize_gate_evidence(item) for item in value]
    raise ValueError(f"unsupported gate evidence value: {type(value).__name__}")


def development_gate_decision_record(
    gate_results: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _canonicalize_gate_evidence(dict(gate_results))
    require(isinstance(normalized.get("passed"), bool), "development gate decision is invalid")
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return {
        "schema_version": 1,
        "float_metric_decimal_places": GATE_EVIDENCE_DECIMAL_PLACES,
        "decision": "deployable_for_reserved_evaluation" if normalized["passed"] else "development_stop",
        "passed": bool(normalized["passed"]),
        "passed_count": integer(normalized.get("passed_count"), "passed gate count"),
        "gate_count": integer(normalized.get("gate_count"), "gate count", minimum=1),
        "gate_results_canonical_json_sha256": sha256_bytes(canonical),
        "gate_results": normalized,
    }


def validate_development_gate_decision(value: Any) -> dict[str, Any]:
    decision = mapping(value, "development gate decision")
    gate_results = mapping(decision.get("gate_results"), "development gate results")
    require(
        gate_results == _canonicalize_gate_evidence(dict(gate_results)),
        "development gate results are not float-canonicalized",
    )
    passed = gate_results.get("passed")
    require(isinstance(passed, bool), "development gate pass flag is invalid")
    passed_count = integer(gate_results.get("passed_count"), "passed gate count")
    gate_count = integer(gate_results.get("gate_count"), "gate count", minimum=1)
    canonical = json.dumps(
        gate_results,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    expected_decision = "deployable_for_reserved_evaluation" if passed else "development_stop"
    require(
        decision.get("schema_version") == 1
        and decision.get("float_metric_decimal_places")
        == GATE_EVIDENCE_DECIMAL_PLACES
        and decision.get("decision") == expected_decision
        and decision.get("passed") is passed
        and decision.get("passed_count") == passed_count
        and decision.get("gate_count") == gate_count
        and passed_count <= gate_count
        and _validate_sha(
            decision.get("gate_results_canonical_json_sha256"),
            "development gate-results hash",
        )
        == sha256_bytes(canonical),
        "development gate decision record is inconsistent",
    )
    return dict(decision)


def _runtime_versions() -> dict[str, str]:
    packages = ("joblib", "numpy", "scikit-learn", "scipy", "threadpoolctl")
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def _float64_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return sha256_bytes(array.tobytes(order="C"))


def training_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered_row_ids = [str(row["row_id"]) for row in rows]
    ordered_event_ids = [str(row["event_id"]) for row in rows]
    ordered_labels = [str(row["label"]) for row in rows]

    def text_hash(values: Sequence[str]) -> str:
        return sha256_bytes("\n".join(values).encode("utf-8"))

    feature_hashes = {
        variant: _float64_sha256(matrix_for(rows, variant)) for variant in VARIANTS
    }
    identity = {
        "ordered_row_ids": ordered_row_ids,
        "ordered_event_ids": ordered_event_ids,
        "ordered_labels": ordered_labels,
        "ordered_row_ids_sha256": text_hash(ordered_row_ids),
        "ordered_event_ids_sha256": text_hash(ordered_event_ids),
        "ordered_labels_sha256": text_hash(ordered_labels),
        "feature_matrix_float64_sha256": feature_hashes,
    }
    identity["combined_canonical_json_sha256"] = sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    )
    return identity


def build_model_canary(
    rows: Sequence[Mapping[str, Any]],
    models: Mapping[str, Any],
    *,
    temperature: float,
) -> dict[str, Any]:
    require(bool(rows), "cannot build a canary without training rows")
    row = min(rows, key=lambda item: str(item["row_id"]))
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        feature = matrix_for([row], variant)
        probability = apply_temperature(
            aligned_probabilities(models[variant], feature), temperature
        )
        quantized_probability = np.round(
            probability, decimals=CANARY_PROBABILITY_DECIMAL_PLACES
        )
        variants[variant] = {
            "feature_values": feature[0].tolist(),
            "feature_float64_sha256": _float64_sha256(feature),
            "expected_probability_values": probability[0].tolist(),
            "expected_quantized_probability_values": quantized_probability[0].tolist(),
            "expected_quantized_probability_float64_sha256": _float64_sha256(
                quantized_probability
            ),
        }
    return {
        "row_id": str(row["row_id"]),
        "selection": "lexicographically_first_evaluable_training_row_id",
        "classes_in_order": list(CLASSES),
        "probability_comparison": {
            "decimal_places": CANARY_PROBABILITY_DECIMAL_PLACES,
            "absolute_tolerance": CANARY_PROBABILITY_ATOL,
            "relative_tolerance": CANARY_PROBABILITY_RTOL,
            "hash_encoding": "little_endian_float64_after_numpy_decimal_rounding",
            "reason": "parallel_tree_probability_reduction_can_vary_by_float64_roundoff",
        },
        "variants": variants,
    }


def verify_model_canary(
    canary: Mapping[str, Any],
    models: Mapping[str, Any],
    *,
    temperature: float,
) -> None:
    require(
        canary.get("classes_in_order") == list(CLASSES),
        "model canary class order changed",
    )
    require(
        canary.get("probability_comparison")
        == {
            "decimal_places": CANARY_PROBABILITY_DECIMAL_PLACES,
            "absolute_tolerance": CANARY_PROBABILITY_ATOL,
            "relative_tolerance": CANARY_PROBABILITY_RTOL,
            "hash_encoding": "little_endian_float64_after_numpy_decimal_rounding",
            "reason": "parallel_tree_probability_reduction_can_vary_by_float64_roundoff",
        },
        "model canary probability comparison contract changed",
    )
    variants = mapping(canary.get("variants"), "model canary variants")
    require(set(variants) == set(VARIANTS), "model canary variants changed")
    for variant in VARIANTS:
        record = mapping(variants[variant], f"{variant} canary")
        feature = np.asarray(record.get("feature_values"), dtype=np.float64).reshape(1, -1)
        require(
            feature.shape == (1, FEATURE_WIDTH)
            and _float64_sha256(feature)
            == _validate_sha(
                record.get("feature_float64_sha256"), f"{variant} canary feature hash"
            ),
            f"{variant} canary feature fingerprint changed",
        )
        observed = apply_temperature(
            aligned_probabilities(models[variant], feature), temperature
        )
        expected = np.asarray(
            record.get("expected_probability_values"), dtype=np.float64
        ).reshape(1, -1)
        expected_quantized = np.asarray(
            record.get("expected_quantized_probability_values"), dtype=np.float64
        ).reshape(1, -1)
        observed_quantized = np.round(
            observed, decimals=CANARY_PROBABILITY_DECIMAL_PLACES
        )
        require(
            expected.shape == (1, len(CLASSES))
            and expected_quantized.shape == (1, len(CLASSES))
            and np.array_equal(
                expected_quantized,
                np.round(expected, decimals=CANARY_PROBABILITY_DECIMAL_PLACES),
            )
            and _float64_sha256(expected_quantized)
            == _validate_sha(
                record.get("expected_quantized_probability_float64_sha256"),
                f"{variant} canary probability hash",
            )
            and np.allclose(
                observed,
                expected,
                rtol=CANARY_PROBABILITY_RTOL,
                atol=CANARY_PROBABILITY_ATOL,
            )
            and np.array_equal(observed_quantized, expected_quantized)
            and _float64_sha256(observed_quantized)
            == record["expected_quantized_probability_float64_sha256"],
            f"{variant} estimator canary prediction changed",
        )


def _prepare_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "prompts": args.prompts,
        "public_report": args.public_report,
        "protocol": args.protocol,
        "source_protocol": args.source_protocol,
        "behavioral_protocol": args.behavioral_protocol,
        "source_analyzer": SOURCE_ANALYZER_PATH,
        "binary_analyzer": Path(__file__).resolve(),
        "cohort_checker": COHORT_CHECKER_PATH,
    }
    if args.command == "evaluate":
        paths.update(
            {
                "reserved_cohort": args.reserved_cohort,
                "reserved_campaign_a": args.reserved_campaign_a,
                "reserved_campaign_b": args.reserved_campaign_b,
                "reserved_image_registry": args.reserved_image_registry,
                "reserved_prompts_summary": args.reserved_prompts_summary,
            }
        )
    for label, path in paths.items():
        require(path.is_file(), f"missing {label}: {path}")
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    prompts = (
        COHORT_CHECKER.strict_json_file(args.prompts, "reserved prompts")
        if args.command == "evaluate"
        else read_json(args.prompts, "prompts")
    )
    report = read_json(args.public_report, "public report")
    protocol_value = read_json(args.protocol, "binary protocol")
    source_value = read_json(args.source_protocol, "source task-state protocol")
    behavioral_value = read_json(args.behavioral_protocol, "behavioral protocol")
    protocol = validate_v2_protocol(
        protocol_value,
        protocol_sha256=hashes["protocol"],
        source_protocol_sha256=hashes["source_protocol"],
        behavioral_protocol_sha256=hashes["behavioral_protocol"],
    )
    fit_protocol_lifecycle = None
    if args.command == "fit":
        fit_protocol_lifecycle = build_fit_protocol_lifecycle(protocol)
        require(
            hashes["prompts"] == protocol["development_prompt_sha256"]
            and hashes["public_report"]
            == protocol["development_public_report_sha256"],
            "fit inputs differ from the frozen development prompt/report pins",
        )
    else:
        require(
            protocol["reserved_prompts_summary_sha256"] is not None,
            "evaluate requires a literal reserved prompts summary pin",
        )
    reserved_binding = None
    if args.command == "evaluate":
        reserved_binding = validate_reserved_evaluation_inputs(
            prompts,
            protocol=protocol,
            cohort_path=args.reserved_cohort,
            campaign_a_path=args.reserved_campaign_a,
            campaign_b_path=args.reserved_campaign_b,
            image_registry_path=args.reserved_image_registry,
            prompts_path=args.prompts,
            prompts_summary_path=args.reserved_prompts_summary,
        )
    source_protocol = normalize_source_protocol(
        source_value,
        behavioral_value,
        behavioral_sha256=hashes["behavioral_protocol"],
    )
    require(
        tuple(source_protocol["class_ids"]) == SOURCE_ACTION_CLASSES,
        "source action class order changed",
    )
    require(
        tuple(source_protocol["layers"]) == tuple(protocol["source_layers"]),
        "source and v2 layer contracts differ",
    )
    assignments, assignment_summary = offline_binary_labels(prompts)
    extracted = extract_stable_inference_rows(
        prompts, report, source_protocol=source_protocol
    )
    rows = build_feature_rows(
        extracted["rows"], assignments, ema_alpha=float(protocol["ema_alpha"])
    )
    require(
        len(rows) == extracted["eligibility"]["inference_eligible_stable_row_count"],
        "feature builder dropped a stable inference row",
    )
    return {
        "hashes": hashes,
        "protocol": protocol,
        "rows": rows,
        "eligibility": extracted["eligibility"],
        "assignment_summary": assignment_summary,
        "reserved_binding": reserved_binding,
        "fit_protocol_lifecycle": fit_protocol_lifecycle,
    }


def fit_command(args: argparse.Namespace) -> int:
    prepared = _prepare_inputs(args)
    protocol = prepared["protocol"]
    fit_protocol_lifecycle = mapping(
        prepared["fit_protocol_lifecycle"], "prepared fit protocol lifecycle"
    )
    fit_protocol_state = validate_protocol_lifecycle_transition(
        fit_protocol_lifecycle, protocol, phase="fit"
    )
    rows = prepared["rows"]
    support = support_summary(rows, eligibility=prepared["eligibility"])
    evaluable = list(labeled_rows(rows))
    require(bool(evaluable), "development inputs have no future-label-evaluable rows")

    loro = leave_one_repository_out(rows, protocol=protocol)
    bootstrap = hierarchical_bootstrap(
        evaluable,
        {
            variant: loro["results"][variant]["predictions"] for variant in VARIANTS
        },
        samples=int(protocol["bootstrap"]["samples"]),
        seed=int(protocol["bootstrap"]["seed"]),
        confidence_level=float(protocol["bootstrap"]["confidence_level"]),
        minimum_valid_fraction=float(
            protocol["bootstrap"]["minimum_valid_fraction"]
        ),
    )
    development_gate_results = evaluate_gates(
        protocol["gates"]["development"],
        support=support,
        results=loro["results"],
        bootstrap=bootstrap,
    )
    development_decision = development_gate_decision_record(
        development_gate_results
    )

    labels = identifiers_for(evaluable, "label")
    tasks = identifiers_for(evaluable, "task_id")
    events = identifiers_for(evaluable, "event_id")
    models: dict[str, ExtraTreesClassifier] = {}
    full_weight_diagnostics: dict[str, Any] = {}
    for variant in VARIANTS:
        model, diagnostics = fit_model(
            matrix_for(evaluable, variant),
            labels,
            tasks,
            events,
            model_contract=protocol["model"],
            seed=int(protocol["model"]["random_state"]),
        )
        models[variant] = model
        full_weight_diagnostics[variant] = diagnostics
    require(
        full_weight_diagnostics["j_compact"] == full_weight_diagnostics["l_compact"],
        "matched branches received different training weights",
    )
    training_identity_record = training_identity(evaluable)
    estimator_parameters = {
        variant: models[variant].get_params(deep=False) for variant in VARIANTS
    }
    canary = build_model_canary(
        evaluable, models, temperature=float(protocol["temperature"])
    )

    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "id": "swe-binary-phase-interpreter-v2-model-bundle",
        "classes_in_order": list(CLASSES),
        "variants_in_order": list(VARIANTS),
        "feature_width": FEATURE_WIDTH,
        "protocol_sha256": prepared["hashes"]["protocol"],
        "core_protocol_sha256": protocol["core_sha256"],
        "fit_reserved_prompts_summary_sha256": None,
        "protocol_lifecycle": dict(fit_protocol_lifecycle),
        "protocol_fit_state": fit_protocol_state,
        "source_protocol_sha256": prepared["hashes"]["source_protocol"],
        "behavioral_protocol_sha256": prepared["hashes"]["behavioral_protocol"],
        "binary_analyzer_sha256": prepared["hashes"]["binary_analyzer"],
        "source_analyzer_sha256": prepared["hashes"]["source_analyzer"],
        "cohort_checker_sha256": prepared["hashes"]["cohort_checker"],
        "development_prompt_sha256": prepared["hashes"]["prompts"],
        "development_public_report_sha256": prepared["hashes"]["public_report"],
        "temperature": float(protocol["temperature"]),
        "confidence_threshold": float(protocol["threshold"]),
        "model_contract": dict(protocol["model"]),
        "estimator_get_params": estimator_parameters,
        "training_weight_diagnostics": full_weight_diagnostics["j_compact"],
        "training_identity": training_identity_record,
        "model_canary": canary,
        "development_gate_decision": development_decision,
        "models": models,
    }
    atomic_joblib_dump(args.bundle, bundle)
    bundle_sha256 = sha256_file(args.bundle)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "id": "swe-binary-phase-interpreter-v2-model-manifest",
        "bundle": {
            "filename": args.bundle.name,
            "sha256": bundle_sha256,
            "joblib_top_level_type": "dictionary",
            "contains_variants": list(VARIANTS),
        },
        "protocol_sha256": prepared["hashes"]["protocol"],
        "core_protocol_sha256": protocol["core_sha256"],
        "fit_reserved_prompts_summary_sha256": None,
        "protocol_lifecycle": dict(fit_protocol_lifecycle),
        "protocol_fit_state": fit_protocol_state,
        "classes_in_order": list(CLASSES),
        "variants_in_order": list(VARIANTS),
        "feature_width": FEATURE_WIDTH,
        "temperature": float(protocol["temperature"]),
        "confidence_threshold": float(protocol["threshold"]),
        "model_contract": dict(protocol["model"]),
        "estimator_get_params": estimator_parameters,
        "development_inputs": {
            key: prepared["hashes"][key]
            for key in (
                "prompts",
                "public_report",
                "source_protocol",
                "behavioral_protocol",
                "source_analyzer",
                "binary_analyzer",
                "cohort_checker",
            )
        },
        "development_support": support,
        "training_weight_diagnostics": full_weight_diagnostics["j_compact"],
        "training_identity": training_identity_record,
        "model_canary": canary,
        "development_gate_decision": development_decision,
        "runtime_versions": _runtime_versions(),
        "security": {
            "joblib_is_pickle_and_must_not_be_loaded_before_hash_verification": True,
            "hash_verified_before_load_by_evaluate_command": True,
            "development_gate_stop_checked_before_joblib_load": True,
        },
    }
    atomic_write_json(args.manifest, manifest)
    validate_bundle_and_manifest(
        args.bundle,
        args.manifest,
        current_protocol=protocol,
        expected_source_protocol_sha256=prepared["hashes"]["source_protocol"],
        expected_behavioral_protocol_sha256=prepared["hashes"][
            "behavioral_protocol"
        ],
        protocol_phase="fit",
        require_development_pass=False,
    )
    manifest_sha256 = sha256_file(args.manifest)
    report = {
        "schema_version": SCHEMA_VERSION,
        "id": "swe-binary-phase-interpreter-v2-development-fit",
        "scope": "development_post_architecture_selection_not_operational_validation",
        "inputs": prepared["hashes"],
        "protocol_lifecycle": {
            "fit": fit_protocol_state,
            "allowed_evaluation_transition": fit_protocol_lifecycle[
                "allowed_evaluation_transition"
            ],
        },
        "eligibility": prepared["eligibility"],
        "support": support,
        "target_contract": {
            "classes_in_order": list(CLASSES),
            "scan_starts_at_current_request_action": True,
            "known_inspect_actions_skipped": True,
            "unknown_before_candidate_milestone": "censor",
            "no_observed_future_milestone": "censor",
            "future_trajectory_label_available_at_inference": False,
        },
        "inference_contract": {
            "prediction_emitted_for_every_stable_feature_complete_row": True,
            "current_action_label_required": False,
            "temperature": float(protocol["temperature"]),
            "confidence_threshold_tau": float(protocol["threshold"]),
            "inference_coverage_at_tau": 1.0,
            "label_evaluable_fraction_is_not_inference_coverage": True,
        },
        "development_loro": loro,
        "bootstrap": bootstrap,
        "development_gates": development_gate_results,
        "development_gate_decision": development_decision,
        "frozen_model": {
            "bundle_path": str(args.bundle),
            "bundle_sha256": bundle_sha256,
            "manifest_path": str(args.manifest),
            "manifest_sha256": manifest_sha256,
            "training_weight_diagnostics": full_weight_diagnostics,
            "post_serialization_hash_runtime_analyzer_and_canary_verified": True,
            "post_serialization_verification_required_development_pass": False,
        },
        "operational_reliability_claim": False,
        "limitations": [
            "The architecture and binary target were selected after inspecting this development cohort.",
            "Bootstrap intervals condition on frozen out-of-repository predictions and do not refit models.",
            "ExtraTrees min_samples_leaf counts raw rows despite event-equal sample weights.",
            "The public J-lens fit precision is unpublished and the lens is applied to NVFP4 activations.",
        ],
    }
    atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bundle": str(args.bundle),
                "bundle_sha256": bundle_sha256,
                "manifest": str(args.manifest),
                "stable_emissions": len(rows),
                "label_evaluable_rows": len(evaluable),
                "development_gates_passed": development_gate_results["passed"],
                "operational_reliability_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


def validate_bundle_and_manifest(
    bundle_path: Path,
    manifest_path: Path,
    *,
    current_protocol: Mapping[str, Any],
    expected_source_protocol_sha256: str,
    expected_behavioral_protocol_sha256: str,
    protocol_phase: str,
    require_development_pass: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = mapping(read_json(manifest_path, "model manifest"), "model manifest")
    require(
        manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION
        and manifest.get("id") == "swe-binary-phase-interpreter-v2-model-manifest",
        "model manifest identity changed",
    )
    bundle_record = mapping(manifest.get("bundle"), "manifest bundle record")
    expected_bundle_sha = _validate_sha(bundle_record.get("sha256"), "bundle SHA-256")
    observed_bundle_sha = sha256_file(bundle_path)
    require(observed_bundle_sha == expected_bundle_sha, "model bundle SHA-256 mismatch")
    manifest_lifecycle = validate_fit_protocol_lifecycle(
        manifest.get("protocol_lifecycle")
    )
    protocol_transition = validate_protocol_lifecycle_transition(
        manifest_lifecycle, current_protocol, phase=protocol_phase
    )
    expected_fit_state = fit_protocol_state_record(manifest_lifecycle)
    require(
        manifest.get("protocol_sha256")
        == manifest_lifecycle["fit_full_protocol_sha256"]
        and manifest.get("core_protocol_sha256")
        == manifest_lifecycle["core_protocol_sha256"]
        and manifest.get("fit_reserved_prompts_summary_sha256") is None
        and manifest.get("protocol_fit_state") == expected_fit_state
        and manifest.get("classes_in_order") == list(CLASSES)
        and manifest.get("variants_in_order") == list(VARIANTS)
        and manifest.get("feature_width") == FEATURE_WIDTH
        and manifest.get("temperature") == 1.0
        and manifest.get("confidence_threshold") == 0.0,
        "model manifest contract differs from current protocol",
    )
    development_inputs = mapping(
        manifest.get("development_inputs"), "manifest development inputs"
    )
    current_binary_analyzer_sha = sha256_file(Path(__file__).resolve())
    current_source_analyzer_sha = sha256_file(SOURCE_ANALYZER_PATH)
    current_cohort_checker_sha = sha256_file(COHORT_CHECKER_PATH)
    require(
        development_inputs.get("binary_analyzer") == current_binary_analyzer_sha
        and development_inputs.get("source_analyzer") == current_source_analyzer_sha
        and development_inputs.get("cohort_checker")
        == current_cohort_checker_sha,
        "analyzer implementation changed after model fitting",
    )
    current_versions = _runtime_versions()
    require(
        manifest.get("runtime_versions") == current_versions,
        "readout runtime versions differ from the frozen model manifest",
    )
    manifest_development_decision = validate_development_gate_decision(
        manifest.get("development_gate_decision")
    )
    if require_development_pass:
        require(
            manifest_development_decision["passed"] is True,
            "development gates failed; reserved evaluation is forbidden",
        )

    # Hash verification must remain above this executable pickle load.
    joblib, ExtraTreesClassifier = _ml_dependencies()
    bundle = joblib.load(bundle_path)
    require(isinstance(bundle, dict), "model bundle must be a dictionary")
    require(
        bundle.get("schema_version") == BUNDLE_SCHEMA_VERSION
        and bundle.get("id") == "swe-binary-phase-interpreter-v2-model-bundle"
        and bundle.get("classes_in_order") == list(CLASSES)
        and bundle.get("variants_in_order") == list(VARIANTS)
        and bundle.get("feature_width") == FEATURE_WIDTH
        and bundle.get("protocol_sha256")
        == manifest_lifecycle["fit_full_protocol_sha256"]
        and bundle.get("core_protocol_sha256")
        == manifest_lifecycle["core_protocol_sha256"]
        and bundle.get("fit_reserved_prompts_summary_sha256") is None
        and bundle.get("protocol_lifecycle") == manifest_lifecycle
        and bundle.get("protocol_fit_state") == expected_fit_state
        and bundle.get("source_protocol_sha256") == expected_source_protocol_sha256
        and bundle.get("behavioral_protocol_sha256")
        == expected_behavioral_protocol_sha256
        and bundle.get("binary_analyzer_sha256") == current_binary_analyzer_sha
        and bundle.get("source_analyzer_sha256") == current_source_analyzer_sha
        and bundle.get("cohort_checker_sha256") == current_cohort_checker_sha
        and bundle.get("temperature") == 1.0
        and bundle.get("confidence_threshold") == 0.0,
        "model bundle metadata differs from current protocol",
    )
    models = mapping(bundle.get("models"), "bundle models")
    require(set(models) == set(VARIANTS), "model bundle variants changed")
    for variant in VARIANTS:
        model = models[variant]
        require(
            isinstance(model, ExtraTreesClassifier)
            and int(model.n_features_in_) == FEATURE_WIDTH
            and set(str(item) for item in model.classes_) == set(CLASSES),
            f"{variant} bundled estimator is incompatible",
        )
    require(
        bundle.get("training_identity") == manifest.get("training_identity"),
        "bundle and manifest training identities differ",
    )
    require(
        bundle.get("estimator_get_params") == manifest.get("estimator_get_params"),
        "bundle and manifest estimator parameters differ",
    )
    for variant in VARIANTS:
        require(
            models[variant].get_params(deep=False)
            == mapping(manifest["estimator_get_params"], "manifest estimator parameters")[variant],
            f"{variant} estimator parameters differ from manifest",
        )
    require(
        bundle.get("model_canary") == manifest.get("model_canary"),
        "bundle and manifest model canaries differ",
    )
    require(
        bundle.get("development_gate_decision")
        == manifest_development_decision,
        "bundle and manifest development gate decisions differ",
    )
    verify_model_canary(
        mapping(bundle.get("model_canary"), "model canary"),
        models,
        temperature=float(bundle["temperature"]),
    )
    return dict(bundle), dict(manifest), protocol_transition


def evaluate_command(args: argparse.Namespace) -> int:
    prepared = _prepare_inputs(args)
    protocol = prepared["protocol"]
    bundle, manifest, protocol_transition = validate_bundle_and_manifest(
        args.bundle,
        args.manifest,
        current_protocol=protocol,
        expected_source_protocol_sha256=prepared["hashes"]["source_protocol"],
        expected_behavioral_protocol_sha256=prepared["hashes"]["behavioral_protocol"],
        protocol_phase="evaluate",
    )
    rows = prepared["rows"]
    models = mapping(bundle["models"], "bundle models")
    probabilities = {
        variant: apply_temperature(
            aligned_probabilities(models[variant], matrix_for(rows, variant)),
            float(protocol["temperature"]),
        )
        for variant in VARIANTS
    }
    results = evaluate_probabilities(
        rows, probabilities, threshold=float(protocol["threshold"])
    )
    evaluable = list(labeled_rows(rows))
    evaluable_predictions = {
        variant: [
            prediction
            for prediction in results[variant]["predictions"]
            if prediction["metric_evaluable"]
        ]
        for variant in VARIANTS
    }
    require(bool(evaluable), "validation inputs have no future-label-evaluable rows")
    bootstrap = hierarchical_bootstrap(
        evaluable,
        evaluable_predictions,
        samples=int(protocol["bootstrap"]["samples"]),
        seed=int(protocol["bootstrap"]["seed"]),
        confidence_level=float(protocol["bootstrap"]["confidence_level"]),
        minimum_valid_fraction=float(
            protocol["bootstrap"]["minimum_valid_fraction"]
        ),
    )
    support = support_summary(rows, eligibility=prepared["eligibility"])
    validation_gate_results = evaluate_gates(
        protocol["gates"]["validation"],
        support=support,
        results=results,
        bootstrap=bootstrap,
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "id": "swe-binary-phase-interpreter-v2-reserved-evaluation",
        "inputs": {
            **prepared["hashes"],
            "bundle": sha256_file(args.bundle),
            "manifest": sha256_file(args.manifest),
        },
        "frozen_model_manifest": manifest,
        "protocol_lifecycle": protocol_transition,
        "reserved_cohort_binding": prepared["reserved_binding"],
        "eligibility": prepared["eligibility"],
        "support": support,
        "inference_contract": {
            "prediction_emitted_for_every_stable_feature_complete_row": True,
            "current_action_label_required": False,
            "confidence_threshold_tau": float(protocol["threshold"]),
            "accepted_stable_emissions": len(rows),
            "stable_emission_count": len(rows),
            "inference_coverage_at_tau": 1.0,
            "future_label_evaluable_fraction": support["evaluation"][
                "future_label_evaluable_fraction_of_emissions"
            ],
            "future_label_evaluable_fraction_is_not_inference_coverage": True,
        },
        "results": results,
        "bootstrap": bootstrap,
        "validation_gates": validation_gate_results,
        "operational_reliability_claim": validation_gate_results["passed"],
        "interpretation": (
            "A passing result supports prediction of the next consequential work-cycle phase "
            "at stable Qwen Code request boundaries. It does not decode hidden prose or chain-of-thought."
        ),
    }
    atomic_write_json(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "stable_emissions": len(rows),
                "label_evaluable_rows": len(evaluable),
                "validation_gates_passed": validation_gate_results["passed"],
                "operational_reliability_claim": validation_gate_results["passed"],
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--prompts", type=Path, required=True)
        subparser.add_argument("--public-report", type=Path, required=True)
        subparser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        subparser.add_argument(
            "--source-protocol", type=Path, default=DEFAULT_SOURCE_PROTOCOL
        )
        subparser.add_argument(
            "--behavioral-protocol", type=Path, default=DEFAULT_BEHAVIORAL_PROTOCOL
        )
        subparser.add_argument("--bundle", type=Path, required=True)
        subparser.add_argument("--manifest", type=Path, required=True)
        subparser.add_argument("--output", type=Path, required=True)

    fit_parser = subparsers.add_parser("fit", help="fit and serialize on development data")
    common(fit_parser)
    fit_parser.set_defaults(handler=fit_command)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate a frozen bundle on a new cohort"
    )
    common(evaluate_parser)
    evaluate_parser.add_argument(
        "--reserved-cohort", type=Path, default=DEFAULT_RESERVED_COHORT
    )
    evaluate_parser.add_argument(
        "--reserved-campaign-a", type=Path, default=DEFAULT_RESERVED_CAMPAIGN_A
    )
    evaluate_parser.add_argument(
        "--reserved-campaign-b", type=Path, default=DEFAULT_RESERVED_CAMPAIGN_B
    )
    evaluate_parser.add_argument(
        "--reserved-image-registry",
        type=Path,
        default=DEFAULT_RESERVED_IMAGE_REGISTRY,
    )
    evaluate_parser.add_argument(
        "--reserved-prompts-summary", type=Path, required=True
    )
    evaluate_parser.set_defaults(handler=evaluate_command)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [
        args.prompts,
        args.public_report,
        args.protocol,
        args.source_protocol,
        args.behavioral_protocol,
    ]
    if args.command == "evaluate":
        paths.extend(
            [
                args.bundle,
                args.manifest,
                args.reserved_cohort,
                args.reserved_campaign_a,
                args.reserved_campaign_b,
                args.reserved_image_registry,
                args.reserved_prompts_summary,
            ]
        )
    for path in paths:
        require(path.is_file(), f"missing input: {path}")
    output_paths = [args.output]
    if args.command == "fit":
        output_paths.extend([args.bundle, args.manifest])
    require(
        len({path.resolve() for path in output_paths}) == len(output_paths),
        "output, bundle, and manifest paths must be distinct",
    )
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
