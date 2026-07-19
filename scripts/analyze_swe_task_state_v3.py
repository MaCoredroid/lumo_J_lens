#!/usr/bin/env python3
"""Analyze and fit the predeclared V3 current-action task-state interpreter.

V3 predicts the ensuing same-request completion's observable action at every stable,
feature-complete final-prompt boundary.  The three metric classes are
``inspect``, ``edit``, and ``check_or_finish``; the last class is the frozen
collapse of the source ``validate`` and ``finalize`` labels.  Unknown current
actions still receive predictions and are explicitly reported, but never enter
fit, calibration, threshold selection, or metric weights.

The implementation intentionally imports two small, hash-authenticated pieces
of the published analyzers without modifying them: V1's fourteen causal
prior-action fields and V2's forty-value compact layer-shape summary.  It does
not reuse V2's future label, deltas, EMA, lexical, progress, gap, or horizon
features.

Every forest prediction uses one estimator worker.  Multi-worker
``predict_proba`` accumulates floating-point tree probabilities in lock
acquisition order and is not bit-reproducible even when the fitted trees are
identical.  Fit-level concurrency is instead applied across the twenty frozen
variant/seed estimators, whose results are collected in declared order.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import Counter, defaultdict
import copy
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
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import zlib

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V1_ANALYZER_PATH = ROOT / "scripts/analyze_swe_task_state_interpreter.py"
V2_ANALYZER_PATH = ROOT / "scripts/analyze_swe_binary_phase_v2.py"
COHORT_CHECKER_PATH = ROOT / "scripts/check_swe_task_state_v3_development_cohort.py"
V3_MATERIALIZER_PATH = ROOT / "scripts/materialize_swe_state_interpreter_v3_probes.py"
HISTORICAL_MATERIALIZER_PATH = ROOT / "scripts/materialize_swe_behavioral_probes.py"
REPLAY_PIPELINE_PATH = ROOT / "scripts/swe_task_state_v3_replay_pipeline.py"
REPLAY_SHELL_WRAPPER_PATH = ROOT / "scripts/run_swe_task_state_v3_replay.sh"
V1_PROTOCOL_PATH = ROOT / "configs/swe_task_state_interpreter_protocol.json"
V2_PROTOCOL_PATH = ROOT / "configs/swe_binary_phase_interpreter_v2.json"
HISTORICAL_ACTION_PROTOCOL_PATH = ROOT / "configs/swe_stage_action_probes.json"
DEFAULT_ACTION_PROTOCOL = ROOT / "configs/swe_task_state_v3_action_probes.json"
BEHAVIORAL_PROTOCOL_PATH = ROOT / "configs/swe_behavioral_readout_protocol.json"
DEFAULT_PROTOCOL = ROOT / "configs/swe_task_state_interpreter_v3.json"
V3_REQUIREMENTS_PATH = ROOT / "requirements-v3-state-interpreter.txt"
DEFAULT_DEVELOPMENT_COHORT = (
    ROOT / "configs/swe_task_state_v3_development_cohort.json"
)
V3_DEVELOPMENT_ROOT = ROOT / ".cache/swe_state_interpreter_v3_development"
V3_REPLAY_ROOT = V3_DEVELOPMENT_ROOT / "replay"
V3_INTERPRETER_OUTPUT_ROOT = V3_DEVELOPMENT_ROOT / "interpreter"
DEFAULT_REPLAY_MERGE_RECEIPT = V3_REPLAY_ROOT / "merge-manifest.json"

V1_ANALYZER_SHA256 = "279d0a41742e9feeabd3dbd82b73f609326a4942c983cc2eee7125164ebd0594"
V2_ANALYZER_SHA256 = "eee2f7f49acbaaa5b51144410526f9d62526892c5f27cf929c858035ea07d72e"
V1_PROTOCOL_SHA256 = "a6441137828866e8aad9dc547fc0fee37706ece390f503a81fdd9e0f53ed409a"
V2_PROTOCOL_SHA256 = "40c74f9ba46b7ca0251e4cba7f23a871de5f6d2db9c08ff6722e9f8fbbac80c3"
HISTORICAL_ACTION_PROTOCOL_SHA256 = "bce204d03608e181456bb5c05a041c4bf4d305f48cb4b4e651ba34460d46d493"
V3_ACTION_PROTOCOL_SHA256 = "0ebd258a2b46beb2a9be3d42cab24680803a2f971cb21e96acecb78e19cd81bf"
BEHAVIORAL_PROTOCOL_SHA256 = "ae96a783a6e6736ec6a12fc8d3a7a50b3896c57cf759aa3becd3aa33e257dfa8"
COHORT_CHECKER_SHA256 = "0b0ddc053669fab6ef6c37ddd26ee523d66a135d7515bc9c6dece10ff979a21c"
V3_MATERIALIZER_SHA256 = "702998e411a2b8e1aa2564046a5aa4d5a90536cbf90899f8a50dc584bd6eb364"
HISTORICAL_MATERIALIZER_SHA256 = "c63fac2907b887d973920c8fc71adf219affa1d6373a0aeb8ac2fffd59940a4e"
REPLAY_PIPELINE_SHA256 = "b869a9554efa39b7062a714fa8a117f3d76708bcc29437c7fa06829b00e69ef3"
REPLAY_SHELL_WRAPPER_SHA256 = "157a59a1fc580d32b280e9d920490909d87b3d1eefca3399151240151b83fced"
V3_REQUIREMENTS_SHA256 = "71e8d7c1dc198ecf887493d455613c46e5098709ef7ea5a3134b854596725cf6"
DEVELOPMENT_SELECTION_PROOF_PATH = (
    ROOT / "validation/swe-task-state-v3-development-cohort-selection.json"
)
DEVELOPMENT_SELECTION_PROOF_SHA256 = "7adb31c20ae3b0fe8e0074e921afc0847f11e42d48e36863741cc09f4a86b9bf"

SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
PROTOCOL_ID = "swe-task-state-interpreter-v3"
CLASSES = ("inspect", "edit", "check_or_finish")
SOURCE_ACTION_CLASSES = ("inspect", "edit", "validate", "finalize")
COLLAPSE = {
    "inspect": "inspect",
    "edit": "edit",
    "validate": "check_or_finish",
    "finalize": "check_or_finish",
}
VARIANTS = ("history_only", "history_j", "history_logit", "history_logit_j")
VARIANT_WIDTHS = {
    "history_only": 14,
    "history_j": 54,
    "history_logit": 54,
    "history_logit_j": 94,
}
SOURCE_LAYERS = tuple(range(24, 48))
SOURCE_LAYER_COUNT = 24
SOURCE_CLASS_COUNT = 4
HISTORY_WIDTH = 14
COMPACT_WIDTH = 40
BOOTSTRAP_NUMPY_VERSION = "2.5.1"
BOOTSTRAP_BIT_GENERATOR = "PCG64"
BOOTSTRAP_QUANTILE_METHOD = "inverted_cdf"
BOOTSTRAP_ALGORITHM = (
    "hierarchical_bayesian_bootstrap_with_full_nested_loro_model_refit"
)
BOOTSTRAP_INTERVAL_INTERPRETATION = (
    "95_percent_equal_tail_hierarchical_bayesian_bootstrap_credible_interval"
)
BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION = 3
BOOTSTRAP_ROW_EVIDENCE_SCHEMA_VERSION = 1
BOOTSTRAP_PROBABILITY_ENCODING = (
    "little_endian_float64_c_order_zlib_level_9_base64"
)
BOOTSTRAP_ACCEPTANCE_ENCODING = (
    "numpy_packbits_bitorder_little_zlib_level_9_base64_zero_padding"
)
BOOTSTRAP_RETAINED_BYTES_PROVE = (
    "identity_bound_row_weight_probability_acceptance_metric_pair_and_interval_"
    "self_consistency"
)
BOOTSTRAP_REFIT_DECLARATION_SCOPE = (
    "in_process_execution_declaration_by_the_frozen_analyzer_not_persisted_refit_proof"
)
FROZEN_SKLEARN_VERSION = "1.9.0"
FROZEN_JOBLIB_VERSION = "1.5.3"
FROZEN_SCIPY_VERSION = "1.18.0"
FROZEN_THREADPOOLCTL_VERSION = "3.6.0"
FROZEN_IJSON_VERSION = "3.5.0"
FROZEN_ZLIB_VERSION = "1.3.2"

HISTORY_FEATURE_NAMES = (
    "log1p_prior_count_inspect",
    "log1p_prior_count_edit",
    "log1p_prior_count_validate",
    "log1p_prior_count_finalize",
    "previous_action_is_inspect",
    "previous_action_is_edit",
    "previous_action_is_validate",
    "previous_action_is_finalize",
    "log1p_prior_unknown_count",
    "previous_action_is_unknown",
    "has_edited",
    "has_validated",
    "turns_since_edit_or_minus_one",
    "turns_since_validate_or_minus_one",
)
COMPACT_SUMMARIES = (
    "mean_all_layers",
    "standard_deviation_all_layers",
    "minimum_all_layers",
    "maximum_all_layers",
    "mean_layers_24_to_29",
    "mean_layers_30_to_41",
    "mean_layers_42_to_47",
    "least_squares_layer_slope",
    "last_minus_first_layer",
    "argmax_layer_index_divided_by_23",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a JSON array")
    return value


def nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be a nonempty string")
    return value


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return int(value)


def finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _expected_bootstrap_evidence_scope() -> dict[str, Any]:
    return {
        "retained_bytes_prove": BOOTSTRAP_RETAINED_BYTES_PROVE,
        "models_refit_inside_bootstrap_field": BOOTSTRAP_REFIT_DECLARATION_SCOPE,
        "cryptographic_refit_attestation": False,
        "independent_refit_attestation": False,
        "label_informed_self_consistent_malicious_artifact_authors": "out_of_scope",
        "missing_or_internally_inconsistent_1000_draw_evidence_fails_bound_gates_closed": True,
        "reserved_validation_data_used_for_evidence_or_gates": False,
    }


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {path}: {error}") from error


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def atomic_write_json_no_clobber(path: Path, value: Any) -> None:
    """Atomically create a JSON artifact without ever replacing a destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise ValueError(f"refusing to clobber existing output: {path}") from error
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def atomic_joblib_dump(path: Path, value: Any) -> None:
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("joblib is required; use requirements-readout-v2.txt") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".joblib.tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        joblib.dump(value, temporary_name, compress=3)
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def atomic_joblib_dump_no_clobber(path: Path, value: Any) -> None:
    """Atomically create a joblib artifact without replacing a destination."""

    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("joblib is required; use requirements-readout-v2.txt") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".joblib.tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        joblib.dump(value, temporary_name, compress=3)
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise ValueError(f"refusing to clobber existing output: {path}") from error
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _load_pinned_module(name: str, path: Path, expected_sha256: str) -> Any:
    require(path.is_file(), f"missing historical helper: {path}")
    require(
        sha256_file(path) == expected_sha256,
        f"historical helper bytes changed: {path.relative_to(ROOT)}",
    )
    specification = importlib.util.spec_from_file_location(name, path)
    require(
        specification is not None and specification.loader is not None,
        f"cannot import historical helper: {path}",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


HISTORICAL_V1 = _load_pinned_module(
    "_swe_task_state_v3_pinned_v1", V1_ANALYZER_PATH, V1_ANALYZER_SHA256
)
HISTORICAL_V2 = _load_pinned_module(
    "_swe_task_state_v3_pinned_v2", V2_ANALYZER_PATH, V2_ANALYZER_SHA256
)
COHORT_CHECKER = _load_pinned_module(
    "_swe_task_state_v3_pinned_bundle_checker",
    COHORT_CHECKER_PATH,
    COHORT_CHECKER_SHA256,
)
REPLAY_PIPELINE = _load_pinned_module(
    "_swe_task_state_v3_pinned_replay_pipeline",
    REPLAY_PIPELINE_PATH,
    REPLAY_PIPELINE_SHA256,
)


def _validate_file_pin(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"missing pinned {label}: {path}")
    require(sha256_file(path) == expected, f"{label} SHA-256 changed")


def _expected_v3_action_protocol() -> dict[str, Any]:
    historical = copy.deepcopy(
        mapping(read_json(HISTORICAL_ACTION_PROTOCOL_PATH, "historical action protocol"), "historical action protocol")
    )
    additions = {
        "inspect": [(" grep", 20049), (" rg", 17815)],
        "edit": [(" write", 3165), (" replace", 8032)],
        "validate": [(" pytest", 26864), (" run", 1542)],
        "finalize": [(" answer", 4087), (" done", 2725)],
    }
    for record in sequence(historical.get("action_classes"), "historical action classes"):
        class_record = mapping(record, "historical action class")
        class_id = nonempty_string(class_record.get("id"), "historical action class ID")
        sequence(class_record.get("tokens"), "historical action tokens").extend(
            {"text": text, "token_id": token_id}
            for text, token_id in additions[class_id]
        )
    return historical


def validate_protocol(
    value: Any,
    *,
    action_protocol_value: Any | None = None,
) -> dict[str, Any]:
    """Validate every frozen V3 field and every historical byte/config pin."""

    protocol = mapping(value, "V3 protocol")
    canonical = mapping(read_json(DEFAULT_PROTOCOL, "canonical V3 protocol"), "canonical V3 protocol")
    require(protocol == canonical, "V3 protocol differs from the predeclared canonical contract")

    pinned_files = (
        (V1_ANALYZER_PATH, V1_ANALYZER_SHA256, "historical V1 analyzer"),
        (V2_ANALYZER_PATH, V2_ANALYZER_SHA256, "historical V2 analyzer"),
        (V1_PROTOCOL_PATH, V1_PROTOCOL_SHA256, "historical V1 protocol"),
        (V2_PROTOCOL_PATH, V2_PROTOCOL_SHA256, "historical V2 protocol"),
        (
            HISTORICAL_ACTION_PROTOCOL_PATH,
            HISTORICAL_ACTION_PROTOCOL_SHA256,
            "historical action protocol",
        ),
        (DEFAULT_ACTION_PROTOCOL, V3_ACTION_PROTOCOL_SHA256, "V3 action protocol"),
        (BEHAVIORAL_PROTOCOL_PATH, BEHAVIORAL_PROTOCOL_SHA256, "behavioral protocol"),
        (COHORT_CHECKER_PATH, COHORT_CHECKER_SHA256, "materialized bundle checker"),
        (V3_MATERIALIZER_PATH, V3_MATERIALIZER_SHA256, "V3 materializer"),
        (
            HISTORICAL_MATERIALIZER_PATH,
            HISTORICAL_MATERIALIZER_SHA256,
            "historical materializer",
        ),
        (REPLAY_PIPELINE_PATH, REPLAY_PIPELINE_SHA256, "V3 replay pipeline"),
        (
            REPLAY_SHELL_WRAPPER_PATH,
            REPLAY_SHELL_WRAPPER_SHA256,
            "V3 replay shell wrapper",
        ),
        (V3_REQUIREMENTS_PATH, V3_REQUIREMENTS_SHA256, "V3 requirements"),
        (
            DEVELOPMENT_SELECTION_PROOF_PATH,
            DEVELOPMENT_SELECTION_PROOF_SHA256,
            "V3 development selection proof",
        ),
    )
    for path, expected, label in pinned_files:
        _validate_file_pin(path, expected, label)

    pins = mapping(protocol.get("pins"), "V3 pins")
    require(
        pins.get("historical_v1_analyzer_sha256") == V1_ANALYZER_SHA256
        and pins.get("historical_v2_analyzer_sha256") == V2_ANALYZER_SHA256
        and pins.get("historical_v1_protocol_sha256") == V1_PROTOCOL_SHA256
        and pins.get("historical_v2_protocol_sha256") == V2_PROTOCOL_SHA256
        and pins.get("historical_action_protocol_sha256")
        == HISTORICAL_ACTION_PROTOCOL_SHA256
        and pins.get("v3_action_protocol_sha256") == V3_ACTION_PROTOCOL_SHA256
        and pins.get("behavioral_protocol_sha256") == BEHAVIORAL_PROTOCOL_SHA256,
        "V3 historical/config pins changed",
    )
    require(
        pins.get("materialized_bundle_checker_sha256") == COHORT_CHECKER_SHA256
        and pins.get("v3_materializer_sha256") == V3_MATERIALIZER_SHA256
        and pins.get("historical_materializer_sha256")
        == HISTORICAL_MATERIALIZER_SHA256
        and pins.get("replay_pipeline_sha256") == REPLAY_PIPELINE_SHA256
        and pins.get("replay_shell_wrapper_sha256")
        == REPLAY_SHELL_WRAPPER_SHA256,
        "V3 materialization/replay provenance pins changed",
    )
    require(
        pins.get("v3_requirements_sha256") == V3_REQUIREMENTS_SHA256,
        "V3 requirements pin changed",
    )
    require(
        pins.get("analyzer_runtime")
        == {
            "numpy": BOOTSTRAP_NUMPY_VERSION,
            "scikit-learn": FROZEN_SKLEARN_VERSION,
            "joblib": FROZEN_JOBLIB_VERSION,
            "scipy": FROZEN_SCIPY_VERSION,
            "threadpoolctl": FROZEN_THREADPOOLCTL_VERSION,
            "ijson": FROZEN_IJSON_VERSION,
            "zlib": FROZEN_ZLIB_VERSION,
        },
        "V3 analyzer runtime pins changed",
    )
    development_contract = mapping(
        protocol.get("development_data_contract"), "V3 development data contract"
    )
    require(
        development_contract.get("cohort_manifest_path")
        == "configs/swe_task_state_v3_development_cohort.json"
        and development_contract.get("selection_proof_path")
        == "validation/swe-task-state-v3-development-cohort-selection.json"
        and development_contract.get("selection_proof_sha256")
        == DEVELOPMENT_SELECTION_PROOF_SHA256
        and development_contract.get("declaration_validator")
        == (
            "scripts/check_swe_task_state_v3_development_cohort.py"
            "::validate_declaration"
        )
        and development_contract.get("validator")
        == (
            "scripts/check_swe_task_state_v3_development_cohort.py"
            "::validate_materialized_bundle"
        ),
        "V3 exact development declaration/bundle bindings changed",
    )
    streaming_contract = mapping(
        mapping(
            protocol.get("bounded_memory_replay_contract"),
            "V3 bounded-memory replay contract",
        ).get("production_analyzer_input"),
        "V3 production analyzer input contract",
    )
    require(
        streaming_contract.get("ijson_version") == "3.5.0"
        and streaming_contract.get("requirements_path")
        == "requirements-v3-state-interpreter.txt"
        and streaming_contract.get("requirements_sha256")
        == V3_REQUIREMENTS_SHA256
        and streaming_contract.get("prompt_report_feature_pass")
        == "bounded_memory_lockstep_streaming",
        "V3 bounded-memory analyzer dependency/input contract changed",
    )

    v1 = mapping(read_json(V1_PROTOCOL_PATH, "historical V1 protocol"), "historical V1 protocol")
    v2 = mapping(read_json(V2_PROTOCOL_PATH, "historical V2 protocol"), "historical V2 protocol")
    v1_pins = mapping(v1.get("input_pins"), "historical V1 input pins")
    v2_pins = mapping(v2.get("pins"), "historical V2 pins")
    require(
        pins.get("model") == v1_pins.get("model") == v2_pins.get("model"),
        "model pin differs across V1/V2/V3",
    )
    require(
        all(
            mapping(pins.get("public_lens"), "V3 lens pin").get(key) == value
            for key, value in mapping(v1_pins.get("public_lens"), "V1 lens pin").items()
        )
        and all(
            mapping(pins.get("public_lens"), "V3 lens pin").get(key) == value
            for key, value in mapping(v2_pins.get("public_lens"), "V2 lens pin").items()
        ),
        "public lens pin differs across V1/V2/V3",
    )
    require(
        all(
            mapping(pins.get("replay_runtime"), "V3 runtime pin").get(key) == value
            for key, value in mapping(v1_pins.get("replay_runtime"), "V1 runtime pin").items()
        ),
        "replay runtime pin differs from V1",
    )

    action_protocol = mapping(
        action_protocol_value
        if action_protocol_value is not None
        else read_json(DEFAULT_ACTION_PROTOCOL, "V3 action protocol"),
        "V3 action protocol",
    )
    require(
        action_protocol == _expected_v3_action_protocol(),
        "V3 action protocol is not the exact historical classifier plus requested forms",
    )
    action_records = sequence(action_protocol.get("action_classes"), "V3 action classes")
    require(
        [mapping(row, "action class").get("id") for row in action_records]
        == list(SOURCE_ACTION_CLASSES),
        "V3 source concept order changed",
    )
    token_ids_by_class: dict[str, list[int]] = {}
    token_texts_by_class: dict[str, list[str]] = {}
    all_token_ids: list[int] = []
    for raw_record in action_records:
        record = mapping(raw_record, "V3 action class")
        class_id = str(record["id"])
        tokens = [mapping(item, f"{class_id} token") for item in sequence(record.get("tokens"), f"{class_id} tokens")]
        require(len(tokens) == 8, f"{class_id} must have exactly eight token forms")
        token_ids = [integer(item.get("token_id"), f"{class_id} token ID") for item in tokens]
        token_texts = [nonempty_string(item.get("text"), f"{class_id} token text") for item in tokens]
        require(all(text.startswith(" ") for text in token_texts), "action forms must retain leading spaces")
        token_ids_by_class[class_id] = token_ids
        token_texts_by_class[class_id] = token_texts
        all_token_ids.extend(token_ids)
    require(len(all_token_ids) == len(set(all_token_ids)), "V3 action token IDs overlap")

    eligibility = mapping(protocol.get("eligibility_contract"), "eligibility contract")
    target = mapping(protocol.get("target_contract"), "target contract")
    require(
        target.get("temporal_alignment")
        == "ensuing_same_request_completion_observable_action_from_the_final_prompt_boundary"
        and target.get("target_is_prospective_ensuing_same_request_completion")
        is True
        and target.get("later_request_actions_used_for_target_or_features") is False,
        "V3 prospective same-request target semantics changed",
    )
    stability = mapping(eligibility.get("numerical_stability"), "stability contract")
    model = mapping(protocol.get("model_contract"), "model contract")
    parameters = mapping(model.get("parameters"), "ExtraTrees parameters")
    fit_execution = mapping(
        model.get("fit_execution"), "model fit execution contract"
    )
    prediction_execution = mapping(
        model.get("prediction_execution"), "model prediction execution contract"
    )
    model_seeds = [
        integer(seed, "ensemble seed")
        for seed in sequence(model.get("seeds_in_order"), "ensemble seeds")
    ]
    require(
        int(parameters.get("n_jobs", 0)) == 1
        and fit_execution.get("parallel_unit")
        == "one_variant_seed_estimator"
        and fit_execution.get("backend")
        == "sklearn_joblib_loky_processes"
        and integer(
            fit_execution.get("worker_count"),
            "variant/seed fit worker count",
            minimum=1,
        )
        == len(VARIANTS) * len(model_seeds)
        and integer(
            fit_execution.get("estimator_fit_n_jobs"),
            "fit-time estimator n_jobs",
            minimum=1,
        )
        == 1
        and integer(
            fit_execution.get("persisted_estimator_n_jobs"),
            "persisted estimator n_jobs",
            minimum=1,
        )
        == int(parameters["n_jobs"])
        and fit_execution.get("submission_order") == "variant_then_seed"
        and fit_execution.get("result_collection_order")
        == "variant_then_seed"
        and fit_execution.get("deterministic_ordered_collection") is True,
        "V3 model fit execution is not the frozen ordered 20-estimator schedule",
    )
    require(
        integer(
            prediction_execution.get("estimator_n_jobs"),
            "prediction estimator n_jobs",
            minimum=1,
        )
        == int(parameters["n_jobs"])
        and prediction_execution.get("tree_probability_reduction_order")
        == "serial_estimator_order"
        and prediction_execution.get("repeated_prediction_must_be_bitwise_identical")
        is True
        and prediction_execution.get("parallel_prediction_forbidden") is True,
        "V3 model prediction is not the frozen deterministic serial reduction",
    )
    weighting = mapping(protocol.get("weighting_contract"), "weighting contract")
    nested = mapping(protocol.get("nested_evaluation"), "nested evaluation")
    calibration = mapping(protocol.get("calibration"), "calibration")
    abstention = mapping(protocol.get("abstention"), "abstention")
    bootstrap = mapping(protocol.get("bootstrap"), "bootstrap")
    bootstrap_execution = mapping(
        bootstrap.get("execution"), "bootstrap execution"
    )
    draw_evidence = mapping(
        bootstrap.get("draw_record_row_prediction_evidence"),
        "bootstrap draw-record row prediction evidence",
    )
    evidence_scope = mapping(
        bootstrap.get("persisted_evidence_scope"),
        "bootstrap persisted evidence scope",
    )
    require(
        weighting.get("point_estimand")
        == (
            "equal_repository_then_equal_known_task_within_repository_then_"
            "equal_known_row_within_task"
        )
        and weighting.get(
            "point_weights_are_computed_once_on_all_known_development_rows_then_restricted_per_split"
        )
        is True
        and weighting.get(
            "bayesian_draw_weights_are_computed_once_on_all_known_development_rows_then_restricted_per_split"
        )
        is True
        and weighting.get("prevalidated_training_weight_fit_transport")
        == (
            "validate_positive_finite_float64_near_unit_mass_and_near_one_third_"
            "class_mass_then_preserve_every_float64_bit_without_second_normalization"
        )
        and finite(
            weighting.get(
                "prevalidated_training_weight_unit_mass_absolute_tolerance"
            ),
            "prevalidated training-weight unit-mass tolerance",
        )
        == 1e-12
        and weighting.get("prevalidated_crossfit_base_weight_transport")
        == (
            "validate_positive_finite_float64_and_near_unit_mass_then_preserve_"
            "every_float64_bit_before_inner_split_restriction"
        )
        and finite(
            weighting.get(
                "prevalidated_crossfit_base_weight_unit_mass_absolute_tolerance"
            ),
            "prevalidated crossfit base-weight unit-mass tolerance",
        )
        == 1e-12,
        "V3 hierarchical point/draw weighting estimand changed",
    )
    require(
        bootstrap.get("algorithm") == BOOTSTRAP_ALGORITHM
        and integer(bootstrap.get("samples"), "Bayesian bootstrap samples") == 1000
        and integer(bootstrap.get("seed"), "Bayesian bootstrap seed") == 918273
        and finite(bootstrap.get("confidence_level"), "bootstrap confidence")
        == 0.95
        and bootstrap.get("interval_interpretation")
        == BOOTSTRAP_INTERVAL_INTERPRETATION
        and bootstrap.get("numpy_version") == BOOTSTRAP_NUMPY_VERSION
        and bootstrap.get("bit_generator") == BOOTSTRAP_BIT_GENERATOR
        and bootstrap.get("seed_sequence_per_draw")
        == "SeedSequence([918273,draw_index])"
        and bootstrap.get("quantile_method") == BOOTSTRAP_QUANTILE_METHOD
        and bootstrap.get("all_original_repositories_tasks_and_rows_retained_every_draw")
        is True
        and bootstrap.get("strictly_positive_finite_float64_weights_required")
        is True
        and bootstrap.get("resampling_or_row_duplication_forbidden") is True
        and bootstrap.get("zero_retries_any_draw_error_aborts") is True
        and bootstrap.get("all_1000_draws_required") is True
        and integer(
            bootstrap_execution.get("checkpoint_schema_version"),
            "bootstrap checkpoint schema version",
            minimum=1,
        )
        == BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION
        and draw_evidence.get("schema_version")
        == BOOTSTRAP_ROW_EVIDENCE_SCHEMA_VERSION
        and draw_evidence.get("row_order")
        == "exact_analyzer_input_order_bound_by_ordered_row_identity_sha256"
        and draw_evidence.get("row_scope")
        == "all_stable_feature_complete_prediction_rows_including_unknown_actions"
        and draw_evidence.get("class_order") == list(CLASSES)
        and draw_evidence.get("probability_encoding")
        == BOOTSTRAP_PROBABILITY_ENCODING
        and draw_evidence.get("acceptance_encoding")
        == BOOTSTRAP_ACCEPTANCE_ENCODING
        and draw_evidence.get("validation")
        == (
            "decode_exact_row_level_evidence_recompute_probability_hash_and_all_"
            "variant_metrics_then_recompute_pairs_and_intervals"
        )
        and dict(evidence_scope) == _expected_bootstrap_evidence_scope(),
        "V3 hierarchical Bayesian-bootstrap contract changed",
    )
    output_contract = mapping(
        protocol.get("analyzer_output_contract"), "analyzer output contract"
    )
    require(
        output_contract.get("dedicated_root")
        == ".cache/swe_state_interpreter_v3_development/interpreter"
        and output_contract.get("new_outputs_are_no_clobber") is True
        and output_contract.get(
            "only_an_identity_bound_self_consistency_validated_matching_"
            "bootstrap_checkpoint_may_be_updated_on_resume"
        )
        is True,
        "V3 analyzer output contract changed",
    )
    return {
        "value": dict(protocol),
        "class_ids": list(CLASSES),
        "source_class_ids": list(SOURCE_ACTION_CLASSES),
        "layers": list(SOURCE_LAYERS),
        "token_ids_by_class": token_ids_by_class,
        "token_texts_by_class": token_texts_by_class,
        "report_helper_protocol": {
            "report_pins": {
                "model": dict(mapping(v1_pins.get("model"), "V1 model pin")),
                "public_lens": dict(mapping(v1_pins.get("public_lens"), "V1 lens pin")),
                "runtime": dict(mapping(v1_pins.get("replay_runtime"), "V1 runtime pin")),
            }
        },
        "eligibility": {
            "stable_rms": finite(
                stability.get("final_logits_rms_error_maximum_inclusive"),
                "stable RMS threshold",
            ),
            "stable_max": finite(
                stability.get("final_logits_max_abs_error_maximum_inclusive"),
                "stable max-abs threshold",
            ),
        },
        "model": {
            "seeds": model_seeds,
            "probability_floor": finite(model.get("probability_floor"), "probability floor"),
            "parameters": dict(parameters),
            "fit_execution": dict(fit_execution),
            "prediction_execution": dict(prediction_execution),
        },
        "weighting": dict(weighting),
        "nested": {
            "minimum_inner_repositories": integer(
                nested.get("minimum_inner_repositories"),
                "minimum inner repositories",
                minimum=2,
            )
        },
        "calibration": {
            "temperatures": [finite(item, "temperature") for item in sequence(calibration.get("temperature_grid"), "temperature grid")]
        },
        "abstention": {
            "thresholds": [finite(item, "threshold") for item in sequence(abstention.get("confidence_threshold_grid"), "threshold grid")],
            **dict(mapping(abstention.get("selection"), "threshold selection")),
        },
        "metrics": dict(mapping(protocol.get("metrics"), "metrics")),
        "bootstrap": dict(bootstrap),
        "gates": dict(mapping(protocol.get("reliability_gates"), "reliability gates")),
    }


def compact_feature_names(prefix: str) -> list[str]:
    return [
        f"{prefix}__{summary}__{concept}"
        for summary in COMPACT_SUMMARIES
        for concept in SOURCE_ACTION_CLASSES
    ]


def feature_names(variant: str) -> list[str]:
    require(variant in VARIANTS, f"unknown variant: {variant}")
    names = list(HISTORY_FEATURE_NAMES)
    if variant in {"history_logit", "history_logit_j"}:
        names.extend(compact_feature_names("current_ordinary_logit"))
    if variant in {"history_j", "history_logit_j"}:
        names.extend(compact_feature_names("current_public_jacobian"))
    require(len(names) == VARIANT_WIDTHS[variant], f"{variant} feature-name width changed")
    return names


def compact_layer_shape(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(HISTORICAL_V2.compact_layer_shape(values), dtype=np.float64)
    require(
        result.shape == (COMPACT_WIDTH,) and np.all(np.isfinite(result)),
        "historical V2 compact layer shape changed",
    )
    return result


def causal_history_features(
    prompts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    values, coverage = HISTORICAL_V1._causal_history_features(
        prompts, SOURCE_ACTION_CLASSES
    )
    for row_id, history in values.items():
        require(
            len(history) == HISTORY_WIDTH
            and all(math.isfinite(float(item)) for item in history),
            f"{row_id} historical causal history width changed",
        )
    return values, coverage


def build_variant_features(
    history: Sequence[float],
    current_public_jacobian: Sequence[float],
    current_ordinary_logit: Sequence[float],
) -> dict[str, list[float]]:
    history_array = np.asarray(history, dtype=np.float64)
    public_compact = compact_layer_shape(current_public_jacobian)
    logit_compact = compact_layer_shape(current_ordinary_logit)
    require(history_array.shape == (HISTORY_WIDTH,), "causal history width changed")
    blocks = {
        "history_only": history_array,
        "history_j": np.concatenate([history_array, public_compact]),
        "history_logit": np.concatenate([history_array, logit_compact]),
        "history_logit_j": np.concatenate(
            [history_array, logit_compact, public_compact]
        ),
    }
    result: dict[str, list[float]] = {}
    for variant in VARIANTS:
        values = np.asarray(blocks[variant], dtype=np.float64)
        require(
            values.shape == (VARIANT_WIDTHS[variant],)
            and np.all(np.isfinite(values)),
            f"{variant} feature vector is invalid",
        )
        result[variant] = values.tolist()
    return result


def _source_action(prompt: Mapping[str, Any]) -> tuple[str | None, str]:
    metadata = mapping(prompt.get("metadata"), "prompt metadata")
    labels = mapping(metadata.get("labels"), "prompt labels")
    action = mapping(labels.get("action"), "prompt action label")
    class_id = action.get("class_id")
    if action.get("status") == "available" and class_id in SOURCE_ACTION_CLASSES:
        return str(class_id), "available"
    return None, str(action.get("status", "unavailable"))


def auxiliary_diagnostic_labels(
    prompts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Derive the two frozen diagnostics without exposing them to features.

    ``milestone_within_2`` examines exactly the current request and its
    immediately following request.  A milestone at the current request wins
    without consulting the next request.  Otherwise an unknown/missing action
    before a milestone makes the diagnostic unknown; two known inspections
    yield ``none``.  No inspection is skipped beyond offset one.
    """

    by_task: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for prompt in prompts:
        metadata = mapping(prompt.get("metadata"), "prompt metadata")
        task = mapping(metadata.get("task"), "prompt task")
        task_id = nonempty_string(task.get("instance_id"), "task ID")
        selection = mapping(metadata.get("selection"), "prompt selection")
        request_index = integer(
            selection.get("task_request_index"), "task request index", minimum=1
        )
        require(
            request_index not in by_task[task_id],
            f"duplicate task request index in {task_id}",
        )
        by_task[task_id][request_index] = prompt

    result: dict[str, dict[str, Any]] = {}
    consequential_mapping = {
        "inspect": "none",
        "edit": "edit",
        "validate": "check_or_finish",
        "finalize": "check_or_finish",
    }
    for task_id, by_index in sorted(by_task.items()):
        for request_index, prompt in sorted(by_index.items()):
            current_action, _ = _source_action(prompt)
            current_consequential = (
                consequential_mapping[current_action]
                if current_action is not None
                else None
            )
            milestone: str | None = None
            status = "available"
            reason: str | None = None
            resolved_offset: int | None = None
            observed_actions: list[str | None] = []
            for offset in (0, 1):
                candidate = by_index.get(request_index + offset)
                if candidate is None:
                    status = "unknown"
                    reason = "incomplete_two_completion_window_before_milestone"
                    break
                action, _ = _source_action(candidate)
                observed_actions.append(action)
                if action is None:
                    status = "unknown"
                    reason = "unclassified_action_before_milestone"
                    break
                if action == "inspect":
                    continue
                milestone = "edit" if action == "edit" else "check_or_finish"
                resolved_offset = offset
                break
            else:
                milestone = "none"
            result[str(prompt["id"])] = {
                "current_consequential_source_type": {
                    "status": "available" if current_consequential is not None else "unknown",
                    "label": current_consequential,
                    "source_action": current_action,
                    "role": "diagnostic_only_not_an_operational_gate",
                },
                "milestone_within_2": {
                    "status": status,
                    "label": milestone if status == "available" else None,
                    "reason": reason,
                    "window_offsets_inclusive": [0, 1],
                    "resolved_offset": resolved_offset,
                    "observed_actions_until_resolution": observed_actions,
                    "role": "diagnostic_only_not_an_operational_gate",
                },
            }
    return result


def _extract_aligned_stable_rows(
    prompts_for_context: Sequence[Mapping[str, Any]],
    aligned_pairs: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    prompt_count: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Shared exact extractor for in-memory and lockstep-streaming inputs."""

    # Diagnostics are derived separately and are never passed to feature code.
    auxiliary_by_id = auxiliary_diagnostic_labels(prompts_for_context)
    # This helper records each row before reading/updating that row's action.
    history_by_id, history_coverage = causal_history_features(prompts_for_context)
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    stable_count = 0
    processed_count = 0
    known_source_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    for prompt, experiment in aligned_pairs:
        processed_count += 1
        prompt_id = str(prompt["id"])
        require(
            experiment.get("prompt") == prompt.get("text")
            and experiment.get("prompt_token_ids") == prompt.get("token_ids")
            and experiment.get("metadata") == prompt.get("metadata"),
            f"{prompt_id} report payload is not bound to supplied prompt",
        )
        prompt_token_ids = sequence(prompt.get("token_ids"), f"{prompt_id} token IDs")
        require(bool(prompt_token_ids), f"{prompt_id} token IDs are empty")
        expected_position = len(prompt_token_ids) - 1
        require(
            experiment.get("capture_positions_resolved") == [expected_position],
            f"{prompt_id} was not captured only at the final prompt token",
        )
        scored = mapping(experiment.get("scored_vocabulary"), f"{prompt_id} scored vocabulary")
        require(
            scored.get("token_ids") == prompt.get("score_token_ids"),
            f"{prompt_id} scored vocabulary differs from prompt contract",
        )
        stable, stability_reasons = HISTORICAL_V1._numerically_stable(
            experiment, protocol["eligibility"]
        )
        if not stable:
            exclusion_counts["numerically_unstable"] += 1
            exclusions.append(
                {
                    "row_id": prompt_id,
                    "reason": "numerically_unstable",
                    "details": stability_reasons,
                }
            )
            continue
        stable_count += 1
        history = history_by_id.get(prompt_id)
        if history is None:
            exclusion_counts["causal_history_unavailable"] += 1
            exclusions.append(
                {
                    "row_id": prompt_id,
                    "reason": "causal_history_unavailable",
                    "details": ["complete consecutive probe bundle required"],
                }
            )
            continue

        ordinary = HISTORICAL_V1._layer_class_features(
            experiment,
            layers=protocol["layers"],
            class_ids=protocol["source_class_ids"],
            token_ids_by_class=protocol["token_ids_by_class"],
            method="ordinary_logit",
            expected_token_position=expected_position,
        )
        public = HISTORICAL_V1._layer_class_features(
            experiment,
            layers=protocol["layers"],
            class_ids=protocol["source_class_ids"],
            token_ids_by_class=protocol["token_ids_by_class"],
            method="public_jacobian",
            expected_token_position=expected_position,
        )
        require(
            len(ordinary) == len(public) == SOURCE_LAYER_COUNT * SOURCE_CLASS_COUNT,
            f"{prompt_id} current score width changed",
        )
        source_action, action_status = _source_action(prompt)
        label = COLLAPSE[source_action] if source_action is not None else None
        if source_action is None:
            known_source_counts["unknown"] += 1
            target_counts["unknown_metric_ineligible"] += 1
        else:
            known_source_counts[source_action] += 1
            target_counts[str(label)] += 1

        metadata = mapping(prompt.get("metadata"), f"{prompt_id} metadata")
        task = mapping(metadata.get("task"), f"{prompt_id} task")
        selection = mapping(metadata.get("selection"), f"{prompt_id} selection")
        cohort = metadata.get("cohort")
        cohort_id = (
            str(cohort["id"])
            if isinstance(cohort, dict) and isinstance(cohort.get("id"), str)
            else "unspecified"
        )
        request_index = integer(
            selection.get("task_request_index"), "task request index", minimum=1
        )
        rows.append(
            {
                "row_id": prompt_id,
                "task_id": nonempty_string(task.get("instance_id"), "task ID"),
                "repo": nonempty_string(task.get("repo"), "task repository"),
                "cohort_id": cohort_id,
                "task_request_index": request_index,
                "checkpoint_ordinal": selection.get("checkpoint_ordinal"),
                "source_action_label_status": action_status,
                "source_action_class_id": source_action,
                "label_status": "available" if label is not None else "unknown_current_action",
                "label": label,
                "metric_evaluable": label is not None,
                "auxiliary_diagnostics": auxiliary_by_id[prompt_id],
                "features": build_variant_features(history, public, ordinary),
            }
        )

    require(
        processed_count == prompt_count,
        "aligned prompt/report stream count differs from declared prompt count",
    )
    known_count = sum(row["metric_evaluable"] for row in rows)
    return {
        "rows": rows,
        "eligibility": {
            "all_replayed_prompt_count": prompt_count,
            "numerically_stable_prompt_count": stable_count,
            "stable_feature_complete_prediction_count": len(rows),
            "known_current_action_prediction_count": known_count,
            "unknown_current_action_prediction_count": len(rows) - known_count,
            "numerical_stability_fraction": stable_count / prompt_count if prompt_count else 0.0,
            "stable_feature_complete_prediction_fraction": (
                len(rows) / stable_count if stable_count else 0.0
            ),
            "stable_feature_complete_prediction_fraction_numerator": len(rows),
            "stable_feature_complete_prediction_fraction_denominator": stable_count,
            "known_current_action_fraction_of_predictions": known_count / len(rows) if rows else 0.0,
            "predictions_emitted_for_unknown_current_actions": True,
            "current_action_used_only_after_causal_history_was_computed": True,
            "source_action_support": dict(sorted(known_source_counts.items())),
            "target_support": dict(sorted(target_counts.items())),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "exclusions": exclusions,
            "causal_history": history_coverage,
        },
    }


def extract_stable_rows(
    prompt_bundle_value: Any,
    report_value: Any,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """In-memory reference extractor used by focused fixtures and parity tests."""

    prompts = sequence(prompt_bundle_value, "prompt bundle")
    report = mapping(report_value, "public report")
    HISTORICAL_V1._validate_report_provenance(
        report, protocol=protocol["report_helper_protocol"]
    )
    experiments = sequence(report.get("experiments"), "report experiments")
    require(len(prompts) == len(experiments), "prompt/report row counts differ")
    prompt_ids = [nonempty_string(row.get("id"), "prompt ID") for row in prompts]
    experiment_ids = [
        nonempty_string(row.get("id"), "experiment ID") for row in experiments
    ]
    require(len(prompt_ids) == len(set(prompt_ids)), "prompt IDs are duplicated")
    require(
        len(experiment_ids) == len(set(experiment_ids)),
        "report experiment IDs are duplicated",
    )
    require(prompt_ids == experiment_ids, "prompt/report IDs or order differ")
    return _extract_aligned_stable_rows(
        prompts,
        zip(prompts, experiments, strict=True),
        prompt_count=len(prompts),
        protocol=protocol,
    )


def _ijson_dependency() -> Any:
    try:
        import ijson
    except ImportError as error:
        raise RuntimeError(
            "ijson 3.5.0 is required for bounded-memory V3 extraction; "
            "install requirements-v3-state-interpreter.txt"
        ) from error
    require(
        str(getattr(ijson, "__version__", "")) == "3.5.0",
        "bounded-memory extraction requires the frozen ijson 3.5.0 release",
    )
    return ijson


def _next_stream_event(events: Iterator[Any], label: str) -> tuple[str, str, Any]:
    try:
        prefix, event, value = next(events)
    except StopIteration as error:
        raise ValueError(f"{label} ended before its JSON value was complete") from error
    return str(prefix), str(event), value


def _build_stream_value(
    first_event: tuple[str, str, Any],
    events: Iterator[Any],
    *,
    label: str,
    ijson: Any,
) -> Any:
    _prefix, event, value = first_event
    builder = ijson.ObjectBuilder()
    builder.event(event, value)
    if event not in {"start_map", "start_array"}:
        require(
            event in {"null", "boolean", "integer", "double", "number", "string"},
            f"{label} has an invalid JSON value event",
        )
        return builder.value
    depth = 1
    while depth:
        _nested_prefix, nested_event, nested_value = _next_stream_event(
            events, label
        )
        builder.event(nested_event, nested_value)
        if nested_event in {"start_map", "start_array"}:
            depth += 1
        elif nested_event in {"end_map", "end_array"}:
            depth -= 1
    return builder.value


def _discard_stream_value(
    first_event: tuple[str, str, Any], events: Iterator[Any], *, label: str
) -> None:
    _prefix, event, _value = first_event
    if event not in {"start_map", "start_array"}:
        require(
            event in {"null", "boolean", "integer", "double", "number", "string"},
            f"{label} has an invalid JSON value event",
        )
        return
    depth = 1
    while depth:
        _nested_prefix, nested_event, _nested_value = _next_stream_event(
            events, label
        )
        if nested_event in {"start_map", "start_array"}:
            depth += 1
        elif nested_event in {"end_map", "end_array"}:
            depth -= 1


def _stream_json_array_objects(path: Path, *, label: str) -> Iterator[Mapping[str, Any]]:
    """Yield one root-array object at a time and reject extra JSON values."""

    ijson = _ijson_dependency()
    path = path.expanduser()
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        with path.open("rb") as handle:
            events = iter(ijson.parse(handle, use_float=True))
            require(
                _next_stream_event(events, label) == ("", "start_array", None),
                f"{label} root must be a JSON array",
            )
            while True:
                first_event = _next_stream_event(events, label)
                if first_event == ("", "end_array", None):
                    break
                require(
                    first_event[0] == "item" and first_event[1] == "start_map",
                    f"{label} rows must be JSON objects",
                )
                yield mapping(
                    _build_stream_value(
                        first_event, events, label=f"{label} row", ijson=ijson
                    ),
                    f"{label} row",
                )
            try:
                next(events)
            except StopIteration:
                pass
            else:
                raise ValueError(f"{label} contains trailing JSON values")
    except (OSError, UnicodeError, ijson.JSONError) as error:
        raise ValueError(f"could not stream {label}: {path}: {error}") from error


def _history_prompt_skeleton(prompt: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only the small fields used by the exact history/diagnostic helpers."""

    prompt_id = nonempty_string(prompt.get("id"), "prompt ID")
    metadata = mapping(prompt.get("metadata"), f"{prompt_id} metadata")
    task = mapping(metadata.get("task"), f"{prompt_id} task")
    selection = mapping(metadata.get("selection"), f"{prompt_id} selection")
    labels = mapping(metadata.get("labels"), f"{prompt_id} labels")
    action = mapping(labels.get("action"), f"{prompt_id} action label")
    return {
        "id": prompt_id,
        "metadata": {
            "task": {
                "instance_id": nonempty_string(
                    task.get("instance_id"), f"{prompt_id} task ID"
                ),
                "probeable_request_indices": list(
                    sequence(
                        task.get("probeable_request_indices"),
                        f"{prompt_id} probeable request indices",
                    )
                ),
            },
            "selection": {
                "task_request_index": integer(
                    selection.get("task_request_index"),
                    f"{prompt_id} task request index",
                    minimum=1,
                )
            },
            "labels": {"action": dict(action)},
        },
    }


def _stream_prompt_context(path: Path) -> list[dict[str, Any]]:
    skeletons: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for prompt in _stream_json_array_objects(path, label="prompt bundle"):
        skeleton = _history_prompt_skeleton(prompt)
        prompt_id = str(skeleton["id"])
        require(prompt_id not in seen_ids, "prompt IDs are duplicated")
        seen_ids.add(prompt_id)
        skeletons.append(skeleton)
    return skeletons


def _stream_report_experiments(
    path: Path, metadata: dict[str, Any]
) -> Iterator[Mapping[str, Any]]:
    """Yield experiments while collecting only small schema-3 root provenance."""

    ijson = _ijson_dependency()
    path = path.expanduser()
    require(
        path.is_file() and not path.is_symlink(),
        "public report is not a regular file",
    )
    retained_keys = {
        "schema_version",
        "score_encoding",
        "assertions",
        "model",
        "lens",
        "runtime",
    }
    seen_keys: set[str] = set()
    experiments_seen = False
    try:
        with path.open("rb") as handle:
            events = iter(ijson.parse(handle, use_float=True))
            require(
                _next_stream_event(events, "public report")
                == ("", "start_map", None),
                "public report root must be a JSON object",
            )
            while True:
                prefix, event, value = _next_stream_event(events, "public report")
                if (prefix, event, value) == ("", "end_map", None):
                    break
                require(
                    prefix == "" and event == "map_key" and isinstance(value, str),
                    "public report root structure is invalid",
                )
                key = value
                require(key not in seen_keys, f"public report repeats root key {key}")
                seen_keys.add(key)
                first_event = _next_stream_event(events, f"public report field {key}")
                if key == "experiments":
                    experiments_seen = True
                    require(
                        first_event == ("experiments", "start_array", None),
                        "public report experiments must be a JSON array",
                    )
                    while True:
                        experiment_event = _next_stream_event(
                            events, "public report experiments"
                        )
                        if experiment_event == ("experiments", "end_array", None):
                            break
                        require(
                            experiment_event[0] == "experiments.item"
                            and experiment_event[1] == "start_map",
                            "public report experiment rows must be JSON objects",
                        )
                        yield mapping(
                            _build_stream_value(
                                experiment_event,
                                events,
                                label="public report experiment",
                                ijson=ijson,
                            ),
                            "public report experiment",
                        )
                elif key in retained_keys:
                    metadata[key] = _build_stream_value(
                        first_event,
                        events,
                        label=f"public report field {key}",
                        ijson=ijson,
                    )
                else:
                    _discard_stream_value(
                        first_event,
                        events,
                        label=f"public report field {key}",
                    )
            require(experiments_seen, "public report lacks experiments")
            try:
                next(events)
            except StopIteration:
                pass
            else:
                raise ValueError("public report contains trailing JSON values")
    except (OSError, UnicodeError, ijson.JSONError) as error:
        raise ValueError(f"could not stream public report: {path}: {error}") from error


def extract_stable_rows_streaming(
    prompts_path: Path,
    report_path: Path,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Bounded-memory production extraction with exact lockstep row alignment."""

    prompt_context = _stream_prompt_context(prompts_path)
    expected_prompt_ids = [str(prompt["id"]) for prompt in prompt_context]
    report_metadata: dict[str, Any] = {}
    prompt_iterator = iter(
        _stream_json_array_objects(prompts_path, label="prompt bundle")
    )
    experiment_iterator = iter(
        _stream_report_experiments(report_path, report_metadata)
    )

    def aligned_pairs() -> Iterator[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        sentinel = object()
        seen_prompt_ids: set[str] = set()
        seen_experiment_ids: set[str] = set()
        index = 0
        try:
            while True:
                prompt = next(prompt_iterator, sentinel)
                experiment = next(experiment_iterator, sentinel)
                if prompt is sentinel and experiment is sentinel:
                    break
                require(
                    prompt is not sentinel,
                    "public report contains trailing experiment rows",
                )
                require(
                    experiment is not sentinel,
                    "prompt bundle contains trailing prompt rows",
                )
                prompt_row = mapping(prompt, "streamed prompt row")
                experiment_row = mapping(experiment, "streamed experiment row")
                prompt_id = nonempty_string(prompt_row.get("id"), "prompt ID")
                experiment_id = nonempty_string(
                    experiment_row.get("id"), "experiment ID"
                )
                require(prompt_id not in seen_prompt_ids, "prompt IDs are duplicated")
                require(
                    experiment_id not in seen_experiment_ids,
                    "report experiment IDs are duplicated",
                )
                seen_prompt_ids.add(prompt_id)
                seen_experiment_ids.add(experiment_id)
                require(
                    index < len(expected_prompt_ids)
                    and prompt_id == expected_prompt_ids[index],
                    "prompt order changed between bounded-memory passes",
                )
                require(
                    prompt_id == experiment_id,
                    "prompt/report IDs or order differ",
                )
                index += 1
                yield prompt_row, experiment_row
            require(
                index == len(expected_prompt_ids),
                "prompt stream count changed between bounded-memory passes",
            )
        finally:
            for iterator in (prompt_iterator, experiment_iterator):
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()

    result = _extract_aligned_stable_rows(
        prompt_context,
        aligned_pairs(),
        prompt_count=len(prompt_context),
        protocol=protocol,
    )
    HISTORICAL_V1._validate_report_provenance(
        report_metadata, protocol=protocol["report_helper_protocol"]
    )
    return result


def known_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("metric_evaluable") is True]


def matrix_for(rows: Sequence[Mapping[str, Any]], variant: str) -> np.ndarray:
    require(variant in VARIANTS, f"unknown variant: {variant}")
    values = np.asarray(
        [mapping(row.get("features"), "row features")[variant] for row in rows],
        dtype=np.float64,
    )
    require(
        values.shape == (len(rows), VARIANT_WIDTHS[variant])
        and np.all(np.isfinite(values)),
        f"{variant} feature matrix is invalid",
    )
    return values


def labels_for(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    labels = np.asarray([str(row["label"]) for row in rows])
    require(
        all(label in CLASSES for label in labels.tolist()),
        "metric rows contain an unknown class",
    )
    return labels


def _normalized_positive_weights(
    rows: Sequence[Mapping[str, Any]],
    weights: Sequence[float] | np.ndarray,
    label: str,
) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    require(
        values.dtype == np.float64
        and values.shape == (len(rows),)
        and np.all(np.isfinite(values))
        and np.all(values > 0.0),
        f"{label} must be strictly positive finite float64 with one value per row",
    )
    total = float(values.sum(dtype=np.float64))
    require(math.isfinite(total) and total > 0.0, f"{label} mass is invalid")
    result = values / total
    require(
        np.all(np.isfinite(result))
        and np.all(result > 0.0)
        and math.isclose(float(result.sum(dtype=np.float64)), 1.0),
        f"{label} normalization failed",
    )
    return result


def restrict_base_weights(
    rows: Sequence[Mapping[str, Any]],
    base_weights: Sequence[float] | np.ndarray,
    indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Restrict one master estimand vector to a split and renormalize it."""

    full = _normalized_positive_weights(rows, base_weights, "master base weights")
    selected = np.asarray(indices, dtype=np.int64)
    require(
        selected.ndim == 1
        and len(selected) > 0
        and np.all(selected >= 0)
        and np.all(selected < len(rows))
        and len(set(selected.tolist())) == len(selected),
        "weight restriction indices are invalid",
    )
    selected_rows = [rows[int(index)] for index in selected]
    return _normalized_positive_weights(
        selected_rows, full[selected], "split-restricted base weights"
    )


def hierarchical_equal_weights(
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Equal repository -> equal task -> equal row point-estimand weights."""

    require(bool(rows), "hierarchical weights require at least one row")
    repositories = sorted({str(row["repo"]) for row in rows})
    weights = np.zeros(len(rows), dtype=np.float64)
    for repository in repositories:
        repository_indices = [
            index
            for index, row in enumerate(rows)
            if str(row["repo"]) == repository
        ]
        tasks = sorted({str(rows[index]["task_id"]) for index in repository_indices})
        require(bool(tasks), f"repository {repository} has no weighted task")
        for task_id in tasks:
            indices = [
                index
                for index in repository_indices
                if str(rows[index]["task_id"]) == task_id
            ]
            mass = 1.0 / (len(repositories) * len(tasks))
            weights[np.asarray(indices, dtype=np.int64)] = mass / len(indices)
    return _normalized_positive_weights(rows, weights, "hierarchical point weights")


def task_equal_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Compatibility name for the frozen hierarchical point estimand."""

    return hierarchical_equal_weights(rows)


def ordered_row_identity_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Hash ordered IDs, repository/task/label fields, and every feature bit."""

    records: list[dict[str, Any]] = []
    for row in rows:
        features = mapping(row.get("features"), "row features")
        feature_hashes: dict[str, str] = {}
        for variant in VARIANTS:
            values = np.asarray(features.get(variant), dtype="<f8")
            require(
                values.shape == (VARIANT_WIDTHS[variant],)
                and np.all(np.isfinite(values)),
                f"{variant} row identity features are invalid",
            )
            feature_hashes[variant] = sha256_bytes(values.tobytes(order="C"))
        records.append(
            {
                "row_id": str(row["row_id"]),
                "task_id": str(row["task_id"]),
                "repo": str(row["repo"]),
                "cohort_id": row.get("cohort_id"),
                "task_request_index": row.get("task_request_index"),
                "checkpoint_ordinal": row.get("checkpoint_ordinal"),
                "source_action_label_status": row.get(
                    "source_action_label_status"
                ),
                "source_action_class_id": row.get("source_action_class_id"),
                "label_status": row.get("label_status"),
                "label": row.get("label"),
                "metric_evaluable": row.get("metric_evaluable"),
                "feature_float64_sha256": feature_hashes,
            }
        )
    return canonical_json_sha256(records)


def training_weights(
    rows: Sequence[Mapping[str, Any]],
    base_weights: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Restrict a base estimand vector, then exactly rebalance three classes."""

    labels = labels_for(rows)
    weights = (
        hierarchical_equal_weights(rows)
        if base_weights is None
        else _normalized_positive_weights(rows, base_weights, "training base weights")
    )
    base_hash = sha256_bytes(np.asarray(weights, dtype="<f8").tobytes(order="C"))
    pre_class_mass = {
        class_id: float(weights[labels == class_id].sum()) for class_id in CLASSES
    }
    for class_id in CLASSES:
        mask = labels == class_id
        mass = float(weights[mask].sum())
        require(mass > 0.0, f"training split lacks class {class_id}")
        weights[mask] *= 1.0 / (len(CLASSES) * mass)
    weights /= weights.sum()
    tasks = np.asarray([str(row["task_id"]) for row in rows])
    return weights, {
        "algorithm": (
            "split_restricted_hierarchical_base_weights_then_exact_global_"
            "three_class_rebalance"
        ),
        "row_count": len(rows),
        "task_count": len(set(tasks.tolist())),
        "repository_count": len({str(row["repo"]) for row in rows}),
        "base_weight_float64_sha256": base_hash,
        "pre_class_mass": pre_class_mass,
        "post_class_mass": {
            class_id: float(weights[labels == class_id].sum())
            for class_id in CLASSES
        },
        "weight_float64_sha256": sha256_bytes(
            np.asarray(weights, dtype="<f8").tobytes(order="C")
        ),
        "ordered_row_ids_sha256": canonical_json_sha256(
            [str(row["row_id"]) for row in rows]
        ),
    }


def apply_probability_floor(probabilities: np.ndarray, floor: float) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    require(
        values.ndim == 2
        and values.shape[1] == len(CLASSES)
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and 0.0 < floor < 1.0 / len(CLASSES),
        "probability floor inputs are invalid",
    )
    clipped = np.maximum(values, floor)
    clipped /= clipped.sum(axis=1, keepdims=True)
    require(
        np.all(clipped > 0.0)
        and np.all(np.isfinite(clipped))
        and np.allclose(clipped.sum(axis=1), 1.0),
        "floored probabilities are invalid",
    )
    return clipped


def _ml_dependencies() -> tuple[Any, Any, Any]:
    require(
        np.__version__ == BOOTSTRAP_NUMPY_VERSION
        and importlib.metadata.version("scikit-learn") == FROZEN_SKLEARN_VERSION
        and importlib.metadata.version("joblib") == FROZEN_JOBLIB_VERSION
        and importlib.metadata.version("scipy") == FROZEN_SCIPY_VERSION
        and importlib.metadata.version("threadpoolctl")
        == FROZEN_THREADPOOLCTL_VERSION,
        "model fitting requires the frozen NumPy/scikit-learn/joblib/scipy/threadpoolctl runtime",
    )
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.utils.parallel import Parallel, delayed
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn is required; use requirements-readout-v2.txt"
        ) from error
    return ExtraTreesClassifier, Parallel, delayed


def _validated_unit_weight_vector(
    rows: Sequence[Mapping[str, Any]],
    weights: Sequence[float] | np.ndarray,
    *,
    absolute_tolerance: float,
    label: str,
) -> np.ndarray:
    """Validate one near-unit vector while preserving every supplied float bit."""

    values = np.array(weights, dtype=np.float64, copy=True)
    total = float(values.sum(dtype=np.float64))
    require(
        values.shape == (len(rows),)
        and np.all(np.isfinite(values))
        and np.all(values > 0.0)
        and math.isfinite(total)
        and abs(total - 1.0) <= absolute_tolerance,
        f"{label} must be positive finite float64 with near-unit mass",
    )
    return values


def _validated_prebalanced_training_weights(
    rows: Sequence[Mapping[str, Any]],
    weights: Sequence[float] | np.ndarray,
    *,
    absolute_tolerance: float,
) -> np.ndarray:
    """Validate one class-balanced fit vector without renormalizing its bits."""

    values = _validated_unit_weight_vector(
        rows,
        weights,
        absolute_tolerance=absolute_tolerance,
        label="supplied ensemble weights",
    )
    labels = labels_for(rows)
    for class_id in CLASSES:
        class_mass = float(values[labels == class_id].sum(dtype=np.float64))
        require(
            abs(class_mass - 1.0 / len(CLASSES)) <= absolute_tolerance,
            "supplied ensemble weights must preserve exact three-class rebalance within roundoff tolerance",
        )
    return values


def _prepare_ensemble_fit(
    x: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Mapping[str, Any], list[int]]:
    labels = labels_for(rows)
    require(
        x.ndim == 2
        and x.shape[0] == len(rows)
        and x.shape[1] > 0
        and len(rows) > 0
        and np.all(np.isfinite(x)),
        "ensemble training matrix is invalid",
    )
    if weights is None:
        normalized_weights, diagnostics = training_weights(rows)
    else:
        tolerance = finite(
            protocol["weighting"].get(
                "prevalidated_training_weight_unit_mass_absolute_tolerance"
            ),
            "prevalidated training-weight unit-mass tolerance",
        )
        normalized_weights = _validated_prebalanced_training_weights(
            rows,
            weights,
            absolute_tolerance=tolerance,
        )
        labels_for_diagnostics = labels_for(rows)
        diagnostics = {
            "algorithm": "caller_supplied_prevalidated_training_weights",
            "row_count": len(rows),
            "task_count": len({str(row["task_id"]) for row in rows}),
            "repository_count": len({str(row["repo"]) for row in rows}),
            "post_class_mass": {
                class_id: float(
                    normalized_weights[labels_for_diagnostics == class_id].sum()
                )
                for class_id in CLASSES
            },
            "weight_float64_sha256": sha256_bytes(
                np.asarray(normalized_weights, dtype="<f8").tobytes(order="C")
            ),
            "ordered_row_ids_sha256": canonical_json_sha256(
                [str(row["row_id"]) for row in rows]
            ),
        }
    model_contract = mapping(protocol.get("model"), "normalized model contract")
    parameters = mapping(model_contract.get("parameters"), "ExtraTrees parameters")
    seeds = [
        integer(seed, "ensemble seed")
        for seed in sequence(model_contract.get("seeds"), "ensemble seeds")
    ]
    require(bool(seeds) and len(seeds) == len(set(seeds)), "ensemble seeds are invalid")
    return labels, normalized_weights, diagnostics, parameters, seeds


def _fit_seed_estimator(
    ExtraTreesClassifier: Any,
    x: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    parameters: Mapping[str, Any],
    seed: int,
    fit_n_jobs: int,
) -> Any:
    model = ExtraTreesClassifier(
        bootstrap=bool(parameters["bootstrap"]),
        ccp_alpha=float(parameters["ccp_alpha"]),
        class_weight=parameters["class_weight"],
        criterion=str(parameters["criterion"]),
        max_depth=parameters["max_depth"],
        max_features=float(parameters["max_features"]),
        max_leaf_nodes=parameters["max_leaf_nodes"],
        max_samples=parameters["max_samples"],
        min_impurity_decrease=float(parameters["min_impurity_decrease"]),
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        min_samples_split=int(parameters["min_samples_split"]),
        min_weight_fraction_leaf=float(parameters["min_weight_fraction_leaf"]),
        monotonic_cst=parameters["monotonic_cst"],
        n_estimators=int(parameters["n_estimators"]),
        n_jobs=fit_n_jobs,
        oob_score=bool(parameters["oob_score"]),
        random_state=seed,
        verbose=int(parameters["verbose"]),
        warm_start=bool(parameters["warm_start"]),
    )
    model.fit(x, labels, sample_weight=weights * len(labels))
    require(
        set(str(item) for item in model.classes_) == set(CLASSES),
        "seed estimator classes changed",
    )
    model.set_params(n_jobs=int(parameters["n_jobs"]))
    return model


def _ensemble_fit_diagnostics(
    models: Sequence[Any],
    diagnostics: Mapping[str, Any],
    seeds: Sequence[int],
    row_count: int,
) -> dict[str, Any]:
    return {
        **dict(diagnostics),
        "seeds_in_order": list(seeds),
        "estimator_count": len(models),
        "sample_weight_scale": row_count,
        "estimator_get_params": [model.get_params(deep=False) for model in models],
    }


def fit_ensemble(
    x: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    weights: np.ndarray | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    ExtraTreesClassifier, _, _ = _ml_dependencies()
    labels, normalized_weights, diagnostics, parameters, seeds = (
        _prepare_ensemble_fit(
            x, rows, protocol=protocol, weights=weights
        )
    )
    fit_execution = mapping(
        mapping(protocol.get("model"), "normalized model contract").get(
            "fit_execution"
        ),
        "normalized model fit execution",
    )
    fit_n_jobs = integer(
        fit_execution.get("estimator_fit_n_jobs"),
        "fit-time estimator n_jobs",
        minimum=1,
    )
    models = [
        _fit_seed_estimator(
            ExtraTreesClassifier,
            x,
            labels,
            normalized_weights,
            parameters=parameters,
            seed=seed,
            fit_n_jobs=fit_n_jobs,
        )
        for seed in seeds
    ]
    return models, _ensemble_fit_diagnostics(
        models, diagnostics, seeds, len(rows)
    )


def aligned_ensemble_probabilities(
    models: Sequence[Any],
    x: np.ndarray,
    *,
    probability_floor: float,
) -> np.ndarray:
    require(bool(models), "probability ensemble is empty")
    seed_probabilities: list[np.ndarray] = []
    for model in models:
        require(
            integer(getattr(model, "n_jobs", None), "prediction estimator n_jobs")
            == 1,
            "parallel tree-probability reduction is forbidden",
        )
        raw = np.asarray(model.predict_proba(x), dtype=np.float64)
        aligned = np.zeros((len(x), len(CLASSES)), dtype=np.float64)
        for source_index, class_id in enumerate(model.classes_):
            require(str(class_id) in CLASSES, "model emitted an unknown class")
            aligned[:, CLASSES.index(str(class_id))] = raw[:, source_index]
        require(
            np.all(np.isfinite(aligned))
            and np.all(aligned >= 0.0)
            and np.allclose(aligned.sum(axis=1), 1.0),
            "seed probabilities are invalid",
        )
        seed_probabilities.append(aligned)
    averaged = np.mean(np.stack(seed_probabilities, axis=0), axis=0)
    return apply_probability_floor(averaged, probability_floor)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    require(temperature > 0.0, "temperature must be positive")
    values = np.asarray(probabilities, dtype=np.float64)
    require(
        values.ndim == 2
        and values.shape[1] == len(CLASSES)
        and np.all(values > 0.0),
        "temperature inputs must be strictly positive class probabilities",
    )
    log_values = np.log(values) / temperature
    log_values -= log_values.max(axis=1, keepdims=True)
    result = np.exp(log_values)
    result /= result.sum(axis=1, keepdims=True)
    require(np.all(result > 0.0), "temperature scaling introduced zero probability")
    return result


def probability_metrics(
    rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    *,
    accepted: np.ndarray | None = None,
    ece_bins: int = 10,
    weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "recall_inspect": None,
            "recall_edit": None,
            "recall_check_or_finish": None,
            "multiclass_negative_log_likelihood": None,
            "multiclass_brier": None,
            "top_label_ece": None,
            "selected_coverage": None,
            "selected_accepted_accuracy": None,
            "per_class_accepted_coverage": {class_id: None for class_id in CLASSES},
        }
    values = np.asarray(probabilities, dtype=np.float64)
    require(
        values.shape == (len(rows), len(CLASSES))
        and np.all(values > 0.0)
        and np.allclose(values.sum(axis=1), 1.0),
        "metric probabilities are invalid",
    )
    labels = labels_for(rows)
    y = np.asarray([CLASSES.index(label) for label in labels], dtype=np.int64)
    metric_weights = (
        hierarchical_equal_weights(rows)
        if weights is None
        else _normalized_positive_weights(rows, weights, "metric weights")
    )
    predicted_indices = np.argmax(values, axis=1)
    correct = predicted_indices == y
    recalls: dict[str, float | None] = {}
    for class_id in CLASSES:
        mask = labels == class_id
        mass = float(metric_weights[mask].sum())
        recalls[class_id] = (
            float(np.sum(metric_weights[mask] * correct[mask]) / mass)
            if mass
            else None
        )
    balanced_accuracy = (
        float(np.mean([float(recalls[class_id]) for class_id in CLASSES]))
        if all(recalls[class_id] is not None for class_id in CLASSES)
        else None
    )
    true_probability = values[np.arange(len(rows)), y]
    one_hot = np.eye(len(CLASSES), dtype=np.float64)[y]
    confidence = values.max(axis=1)
    ece = 0.0
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        mask = (confidence >= lower) & (
            confidence < upper if bin_index < ece_bins - 1 else confidence <= upper
        )
        mass = float(metric_weights[mask].sum())
        if mass:
            bin_accuracy = float(
                np.sum(metric_weights[mask] * correct[mask]) / mass
            )
            bin_confidence = float(
                np.sum(metric_weights[mask] * confidence[mask]) / mass
            )
            ece += mass * abs(bin_accuracy - bin_confidence)
    accepted_mask = (
        np.ones(len(rows), dtype=bool)
        if accepted is None
        else np.asarray(accepted, dtype=bool)
    )
    require(accepted_mask.shape == (len(rows),), "accepted mask shape changed")
    coverage = float(metric_weights[accepted_mask].sum())
    accepted_accuracy = (
        float(
            np.sum(
                metric_weights[accepted_mask] * correct[accepted_mask]
            )
            / coverage
        )
        if coverage
        else None
    )
    per_class_coverage: dict[str, float | None] = {}
    for class_id in CLASSES:
        mask = labels == class_id
        mass = float(metric_weights[mask].sum())
        per_class_coverage[class_id] = (
            float(metric_weights[mask & accepted_mask].sum() / mass)
            if mass
            else None
        )
    return {
        "row_count": len(rows),
        "accuracy": float(np.sum(metric_weights * correct)),
        "balanced_accuracy": balanced_accuracy,
        "recall_inspect": recalls["inspect"],
        "recall_edit": recalls["edit"],
        "recall_check_or_finish": recalls["check_or_finish"],
        "multiclass_negative_log_likelihood": float(
            -np.sum(metric_weights * np.log(true_probability))
        ),
        "multiclass_brier": float(
            np.sum(
                metric_weights * np.sum((values - one_hot) ** 2, axis=1)
            )
        ),
        "top_label_ece": float(ece),
        "selected_coverage": coverage,
        "selected_accepted_accuracy": accepted_accuracy,
        "per_class_accepted_coverage": per_class_coverage,
    }


def select_temperature(
    rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    temperatures: Sequence[float],
    *,
    weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, float]] = []
    for temperature in temperatures:
        calibrated = apply_temperature(probabilities, float(temperature))
        metric = probability_metrics(rows, calibrated, weights=weights)[
            "multiclass_negative_log_likelihood"
        ]
        candidates.append({"temperature": float(temperature), "nll": float(metric)})
    selected = min(
        candidates,
        key=lambda row: (
            row["nll"],
            abs(row["temperature"] - 1.0),
            row["temperature"],
        ),
    )
    return {
        "temperature": selected["temperature"],
        "selection_metric": (
            "hierarchical_base_weighted_multiclass_negative_log_likelihood"
        ),
        "selected_nll": selected["nll"],
        "candidates": candidates,
    }


def select_threshold(
    rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    *,
    thresholds: Sequence[float],
    accuracy_minimum: float,
    coverage_minimum: float,
    minimum_accepted_rows_per_class: int,
    weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    labels = labels_for(rows)
    confidence = probabilities.max(axis=1)
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = confidence >= float(threshold)
        metrics = probability_metrics(
            rows, probabilities, accepted=accepted, weights=weights
        )
        per_class_counts = {
            class_id: int(np.sum(accepted & (labels == class_id)))
            for class_id in CLASSES
        }
        meets = (
            metrics["selected_accepted_accuracy"] is not None
            and float(metrics["selected_accepted_accuracy"]) >= accuracy_minimum
            and float(metrics["selected_coverage"]) >= coverage_minimum
            and all(
                count >= minimum_accepted_rows_per_class
                for count in per_class_counts.values()
            )
        )
        candidates.append(
            {
                "threshold": float(threshold),
                "coverage": metrics["selected_coverage"],
                "accepted_accuracy": metrics["selected_accepted_accuracy"],
                "accepted_rows_per_class": per_class_counts,
                "meets_floors": bool(meets),
            }
        )
    passing = [candidate for candidate in candidates if candidate["meets_floors"]]
    if passing:
        selected = max(
            passing,
            key=lambda row: (row["coverage"], -row["threshold"]),
        )
        fallback_used = False
    else:
        coverage_candidates = [
            candidate
            for candidate in candidates
            if candidate["coverage"] >= coverage_minimum
            and candidate["accepted_accuracy"] is not None
        ]
        pool = coverage_candidates or [
            candidate
            for candidate in candidates
            if candidate["accepted_accuracy"] is not None
        ]
        require(bool(pool), "threshold grid produced no accepted predictions")
        selected = max(
            pool,
            key=lambda row: (
                row["accepted_accuracy"],
                row["coverage"],
                -row["threshold"],
            ),
        )
        fallback_used = True
    return {
        "threshold": selected["threshold"],
        "selected_under_floors": selected["meets_floors"],
        "fallback_used": fallback_used,
        "accuracy_minimum": accuracy_minimum,
        "coverage_minimum": coverage_minimum,
        "minimum_accepted_rows_per_class": minimum_accepted_rows_per_class,
        "selected_metrics": selected,
        "candidates": candidates,
    }


def select_calibration_and_threshold(
    rows: Sequence[Mapping[str, Any]],
    raw_probabilities: np.ndarray,
    *,
    protocol: Mapping[str, Any],
    weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    calibration = select_temperature(
        rows,
        raw_probabilities,
        protocol["calibration"]["temperatures"],
        weights=weights,
    )
    calibrated = apply_temperature(
        raw_probabilities, float(calibration["temperature"])
    )
    abstention = mapping(protocol.get("abstention"), "normalized abstention")
    threshold = select_threshold(
        rows,
        calibrated,
        thresholds=abstention["thresholds"],
        accuracy_minimum=finite(
            abstention.get("accepted_accuracy_minimum"), "accuracy minimum"
        ),
        coverage_minimum=finite(
            abstention.get("coverage_minimum"), "coverage minimum"
        ),
        minimum_accepted_rows_per_class=integer(
            abstention.get("minimum_accepted_rows_per_class"),
            "minimum accepted rows per class",
            minimum=1,
        ),
        weights=weights,
    )
    return {"calibration": calibration, "abstention": threshold}


def _fit_all_variants(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    base_weights: Sequence[float] | np.ndarray | None = None,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    ExtraTreesClassifier, Parallel, delayed = _ml_dependencies()
    weights, shared_weight_diagnostics = training_weights(rows, base_weights)
    matrices: dict[str, np.ndarray] = {}
    contexts: dict[
        str,
        tuple[
            np.ndarray,
            np.ndarray,
            dict[str, Any],
            Mapping[str, Any],
            list[int],
        ],
    ] = {}
    for variant in VARIANTS:
        matrices[variant] = matrix_for(rows, variant)
        contexts[variant] = _prepare_ensemble_fit(
            matrices[variant], rows, protocol=protocol, weights=weights
        )
        require(
            contexts[variant][2]["weight_float64_sha256"]
            == shared_weight_diagnostics["weight_float64_sha256"],
            "matched branches did not use identical training weights",
        )
    execution = mapping(
        mapping(protocol.get("model"), "normalized model contract").get(
            "fit_execution"
        ),
        "normalized model fit execution",
    )
    worker_count = integer(
        execution.get("worker_count"), "variant/seed fit worker count", minimum=1
    )
    fit_n_jobs = integer(
        execution.get("estimator_fit_n_jobs"),
        "fit-time estimator n_jobs",
        minimum=1,
    )
    persisted_n_jobs = integer(
        execution.get("persisted_estimator_n_jobs"),
        "persisted estimator n_jobs",
        minimum=1,
    )
    require(
        execution.get("parallel_unit") == "one_variant_seed_estimator"
        and execution.get("backend") == "sklearn_joblib_loky_processes"
        and execution.get("submission_order") == "variant_then_seed"
        and execution.get("result_collection_order") == "variant_then_seed"
        and execution.get("deterministic_ordered_collection") is True,
        "model fit execution contract changed",
    )
    require(
        fit_n_jobs == 1
        and persisted_n_jobs
        == int(mapping(protocol["model"]["parameters"], "ExtraTrees parameters")["n_jobs"]),
        "fit-time or persisted estimator n_jobs changed",
    )
    specifications = [
        (variant, seed)
        for variant in VARIANTS
        for seed in contexts[variant][4]
    ]
    require(
        worker_count == len(specifications),
        "variant/seed worker count must equal the frozen estimator count",
    )
    fitted_in_order = Parallel(
        n_jobs=worker_count,
        backend="loky",
        pre_dispatch=worker_count,
    )(
        delayed(_fit_seed_estimator)(
            ExtraTreesClassifier,
            matrices[variant],
            contexts[variant][0],
            contexts[variant][1],
            parameters=contexts[variant][3],
            seed=seed,
            fit_n_jobs=fit_n_jobs,
        )
        for variant, seed in specifications
    )
    require(
        len(fitted_in_order) == len(specifications),
        "parallel estimator result count changed",
    )
    models: dict[str, list[Any]] = {variant: [] for variant in VARIANTS}
    for (variant, _), model in zip(
        specifications, fitted_in_order, strict=True
    ):
        models[variant].append(model)
    diagnostics = {
        variant: _ensemble_fit_diagnostics(
            models[variant],
            contexts[variant][2],
            contexts[variant][4],
            len(rows),
        )
        for variant in VARIANTS
    }
    return models, {
        "shared": shared_weight_diagnostics,
        "variants": diagnostics,
        "same_weights_and_seed_order_across_variants": True,
        "fit_execution": {
            "parallel_unit": "one_variant_seed_estimator",
            "backend": "sklearn_joblib_loky_processes",
            "worker_count": worker_count,
            "estimator_fit_n_jobs": fit_n_jobs,
            "persisted_estimator_n_jobs": persisted_n_jobs,
            "submission_order": "variant_then_seed",
            "result_collection_order": "variant_then_seed",
            "deterministic_ordered_collection": True,
            "estimator_n_jobs": int(
                mapping(
                    protocol["model"]["parameters"], "ExtraTrees parameters"
                )["n_jobs"]
            ),
        },
    }


def crossfit_raw_probabilities(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    base_weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Repository-crossfit raw probabilities with identical matched folds."""

    rows = list(rows)
    require(bool(rows) and len(known_rows(rows)) == len(rows), "crossfit requires known-label rows")
    repositories = np.asarray([str(row["repo"]) for row in rows])
    master_weights = (
        hierarchical_equal_weights(rows)
        if base_weights is None
        else _validated_unit_weight_vector(
            rows,
            base_weights,
            absolute_tolerance=finite(
                protocol["weighting"].get(
                    "prevalidated_crossfit_base_weight_unit_mass_absolute_tolerance"
                ),
                "prevalidated crossfit base-weight unit-mass tolerance",
            ),
            label="crossfit base weights",
        )
    )
    repositories_in_order = sorted(set(repositories.tolist()))
    require(
        len(repositories_in_order) >= int(protocol["nested"]["minimum_inner_repositories"]),
        "too few repositories for inner crossfit",
    )
    probabilities = {
        variant: np.full((len(rows), len(CLASSES)), np.nan, dtype=np.float64)
        for variant in VARIANTS
    }
    covered = np.zeros(len(rows), dtype=bool)
    folds: list[dict[str, Any]] = []
    floor = float(protocol["model"]["probability_floor"])
    for heldout_repository in repositories_in_order:
        train_indices = np.flatnonzero(repositories != heldout_repository)
        evaluation_indices = np.flatnonzero(repositories == heldout_repository)
        train_rows = [rows[int(index)] for index in train_indices]
        evaluation_rows = [rows[int(index)] for index in evaluation_indices]
        train_base_weights = restrict_base_weights(
            rows, master_weights, train_indices
        )
        require(
            set(labels_for(train_rows).tolist()) == set(CLASSES),
            f"inner training split for {heldout_repository} lacks a class",
        )
        models, weight_diagnostics = _fit_all_variants(
            train_rows, protocol=protocol, base_weights=train_base_weights
        )
        for variant in VARIANTS:
            probabilities[variant][evaluation_indices] = aligned_ensemble_probabilities(
                models[variant],
                matrix_for(evaluation_rows, variant),
                probability_floor=floor,
            )
        covered[evaluation_indices] = True
        folds.append(
            {
                "heldout_repository": heldout_repository,
                "training_repositories": sorted(
                    set(repositories[train_indices].tolist())
                ),
                "training_rows": len(train_indices),
                "evaluation_rows": len(evaluation_indices),
                "training_row_ids_sha256": canonical_json_sha256(
                    [str(row["row_id"]) for row in train_rows]
                ),
                "evaluation_row_ids_sha256": canonical_json_sha256(
                    [str(row["row_id"]) for row in evaluation_rows]
                ),
                "heldout_labels_used_for_fit_calibration_or_threshold_selection": False,
                "shared_training_weight_sha256": weight_diagnostics["shared"][
                    "weight_float64_sha256"
                ],
                "training_base_weight_sha256": sha256_bytes(
                    np.asarray(train_base_weights, dtype="<f8").tobytes(order="C")
                ),
                "seed_order": list(protocol["model"]["seeds"]),
            }
        )
    require(np.all(covered), "inner crossfit did not cover every row exactly once")
    for variant in VARIANTS:
        require(
            np.all(np.isfinite(probabilities[variant]))
            and np.all(probabilities[variant] > 0.0),
            f"{variant} inner crossfit probabilities are incomplete",
        )
    return {
        "probabilities": probabilities,
        "folds": folds,
        "repositories_in_order": repositories_in_order,
        "all_rows_covered_once": True,
    }


def _prediction_record(
    row: Mapping[str, Any],
    probability: np.ndarray,
    *,
    threshold: float,
    temperature: float,
) -> dict[str, Any]:
    predicted_index = int(np.argmax(probability))
    confidence = float(probability[predicted_index])
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
                "source_action_label_status",
                "source_action_class_id",
                "label_status",
                "label",
                "metric_evaluable",
                "auxiliary_diagnostics",
            )
        },
        "probabilities": {
            class_id: float(probability[index])
            for index, class_id in enumerate(CLASSES)
        },
        "predicted_class": CLASSES[predicted_index],
        "confidence": confidence,
        "temperature": float(temperature),
        "confidence_threshold": float(threshold),
        "accepted": bool(confidence >= threshold),
    }


def _per_repository_metrics(
    rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    accepted: np.ndarray,
    weights: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    master_weights = _normalized_positive_weights(
        rows, weights, "per-repository metric base weights"
    )
    repositories = sorted({str(row["repo"]) for row in rows})
    result: dict[str, Any] = {}
    for repository in repositories:
        indices = [index for index, row in enumerate(rows) if str(row["repo"]) == repository]
        repository_rows = [rows[index] for index in indices]
        repository_weights = restrict_base_weights(rows, master_weights, indices)
        result[repository] = probability_metrics(
            repository_rows,
            probabilities[np.asarray(indices, dtype=np.int64)],
            accepted=accepted[np.asarray(indices, dtype=np.int64)],
            weights=repository_weights,
        )
    return result


def nested_leave_one_repository_out(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    base_weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Run leakage-free nested LORO while predicting unknown-action rows."""

    rows = list(rows)
    require(bool(rows), "nested evaluation has no stable feature-complete rows")
    require(
        len({str(row["row_id"]) for row in rows}) == len(rows),
        "nested evaluation row IDs are not unique",
    )
    known_indices_in_all = [
        index
        for index, row in enumerate(rows)
        if row.get("metric_evaluable") is True
    ]
    known = [rows[index] for index in known_indices_in_all]
    require(bool(known), "nested evaluation has no known current actions")
    known_base_weights = (
        hierarchical_equal_weights(known)
        if base_weights is None
        else _normalized_positive_weights(
            known, base_weights, "nested known-row base weights"
        )
    )
    base_weight_source = (
        "hierarchical_equal_point_weights"
        if base_weights is None
        else "hierarchical_bayesian_bootstrap_draw_weights"
    )
    all_prediction_base_weights = hierarchical_equal_weights(rows)
    hierarchical_known_action_fraction = float(
        all_prediction_base_weights[
            np.asarray(
                [row.get("metric_evaluable") is True for row in rows], dtype=bool
            )
        ].sum()
    )
    repositories = np.asarray([str(row["repo"]) for row in rows])
    repositories_in_order = sorted(set(repositories.tolist()))
    require(len(repositories_in_order) >= 3, "nested LORO requires repositories")
    require(
        sorted({str(row["repo"]) for row in known}) == repositories_in_order,
        "every prediction repository must contain a known-action row",
    )

    probabilities = {
        variant: np.full((len(rows), len(CLASSES)), np.nan, dtype=np.float64)
        for variant in VARIANTS
    }
    raw_known_oof = {
        variant: np.full((len(rows), len(CLASSES)), np.nan, dtype=np.float64)
        for variant in VARIANTS
    }
    accepted = {
        variant: np.zeros(len(rows), dtype=bool) for variant in VARIANTS
    }
    temperatures_by_row = {
        variant: np.full(len(rows), np.nan, dtype=np.float64) for variant in VARIANTS
    }
    thresholds_by_row = {
        variant: np.full(len(rows), np.nan, dtype=np.float64) for variant in VARIANTS
    }
    covered = np.zeros(len(rows), dtype=bool)
    fold_records: list[dict[str, Any]] = []
    floor = float(protocol["model"]["probability_floor"])

    for heldout_repository in repositories_in_order:
        outer_train_known_indices = [
            index
            for index, row in enumerate(known)
            if str(row["repo"]) != heldout_repository
        ]
        outer_train_rows = [known[index] for index in outer_train_known_indices]
        outer_train_base_weights = restrict_base_weights(
            known, known_base_weights, outer_train_known_indices
        )
        evaluation_indices = np.flatnonzero(repositories == heldout_repository)
        evaluation_rows = [rows[int(index)] for index in evaluation_indices]
        require(
            set(labels_for(outer_train_rows).tolist()) == set(CLASSES),
            f"outer training split for {heldout_repository} lacks a class",
        )

        # These are the only labels used for this fold's temperature/threshold.
        inner = crossfit_raw_probabilities(
            outer_train_rows,
            protocol=protocol,
            base_weights=outer_train_base_weights,
        )
        settings = {
            variant: select_calibration_and_threshold(
                outer_train_rows,
                inner["probabilities"][variant],
                protocol=protocol,
                weights=outer_train_base_weights,
            )
            for variant in VARIANTS
        }
        models, weight_diagnostics = _fit_all_variants(
            outer_train_rows,
            protocol=protocol,
            base_weights=outer_train_base_weights,
        )
        heldout_known_count = 0
        for variant in VARIANTS:
            raw = aligned_ensemble_probabilities(
                models[variant],
                matrix_for(evaluation_rows, variant),
                probability_floor=floor,
            )
            temperature = float(settings[variant]["calibration"]["temperature"])
            threshold = float(settings[variant]["abstention"]["threshold"])
            calibrated = apply_temperature(raw, temperature)
            probabilities[variant][evaluation_indices] = calibrated
            accepted[variant][evaluation_indices] = calibrated.max(axis=1) >= threshold
            temperatures_by_row[variant][evaluation_indices] = temperature
            thresholds_by_row[variant][evaluation_indices] = threshold
            for local_index, global_index in enumerate(evaluation_indices):
                if rows[int(global_index)].get("metric_evaluable") is True:
                    raw_known_oof[variant][int(global_index)] = raw[local_index]
                    heldout_known_count += int(variant == VARIANTS[0])
        covered[evaluation_indices] = True
        selection_ids = [str(row["row_id"]) for row in outer_train_rows]
        heldout_ids = [str(row["row_id"]) for row in evaluation_rows]
        require(
            set(selection_ids).isdisjoint(heldout_ids),
            "outer heldout rows leaked into calibration or threshold selection",
        )
        fold_records.append(
            {
                "heldout_repository": heldout_repository,
                "heldout_prediction_rows": len(evaluation_rows),
                "heldout_known_action_rows": heldout_known_count,
                "outer_training_known_action_rows": len(outer_train_rows),
                "outer_training_repositories": sorted(
                    {str(row["repo"]) for row in outer_train_rows}
                ),
                "inner_fold_count": len(inner["folds"]),
                "inner_folds": inner["folds"],
                "inner_selection_row_ids_sha256": canonical_json_sha256(selection_ids),
                "heldout_row_ids_sha256": canonical_json_sha256(heldout_ids),
                "inner_and_heldout_row_ids_disjoint": True,
                "heldout_labels_used_for_fit_calibration_or_threshold_selection": False,
                "settings": settings,
                "shared_outer_training_weight_sha256": weight_diagnostics["shared"][
                    "weight_float64_sha256"
                ],
                "outer_training_base_weight_sha256": sha256_bytes(
                    np.asarray(outer_train_base_weights, dtype="<f8").tobytes(
                        order="C"
                    )
                ),
                "same_folds_weights_seed_order_and_hyperparameters_across_variants": True,
            }
        )

    require(np.all(covered), "outer LORO did not emit every stable prediction")
    evaluable_indices = np.asarray(known_indices_in_all, dtype=np.int64)
    evaluable_rows = [rows[int(index)] for index in evaluable_indices]
    results: dict[str, Any] = {}
    full_selection: dict[str, Any] = {}
    for variant in VARIANTS:
        require(
            np.all(np.isfinite(probabilities[variant]))
            and np.all(probabilities[variant] > 0.0)
            and np.all(np.isfinite(temperatures_by_row[variant]))
            and np.all(np.isfinite(thresholds_by_row[variant])),
            f"{variant} outer predictions are incomplete",
        )
        known_probabilities = probabilities[variant][evaluable_indices]
        known_accepted = accepted[variant][evaluable_indices]
        metrics = probability_metrics(
            evaluable_rows,
            known_probabilities,
            accepted=known_accepted,
            weights=known_base_weights,
        )
        metrics["known_action_fraction"] = hierarchical_known_action_fraction
        metrics[
            "selected_coverage_denominator"
        ] = "known_current_action_metric_rows_only_not_all_stable_emissions"
        predictions = [
            _prediction_record(
                row,
                probability,
                threshold=float(thresholds_by_row[variant][index]),
                temperature=float(temperatures_by_row[variant][index]),
            )
            for index, (row, probability) in enumerate(
                zip(rows, probabilities[variant], strict=True)
            )
        ]
        full_selection[variant] = select_calibration_and_threshold(
            evaluable_rows,
            raw_known_oof[variant][evaluable_indices],
            protocol=protocol,
            weights=known_base_weights,
        )
        results[variant] = {
            "inference_prediction_count": len(rows),
            "known_action_metric_row_count": len(evaluable_rows),
            "unknown_action_prediction_count": len(rows) - len(evaluable_rows),
            "accepted_inference_prediction_count": int(accepted[variant].sum()),
            "inference_acceptance_fraction": float(accepted[variant].mean()),
            "inference_acceptance_fraction_denominator": "all_stable_feature_complete_predictions",
            "metrics": metrics,
            "per_repository_metrics": _per_repository_metrics(
                evaluable_rows,
                known_probabilities,
                known_accepted,
                known_base_weights,
            ),
            "predictions": predictions,
            "probability_float64_sha256": sha256_bytes(
                np.asarray(probabilities[variant], dtype="<f8").tobytes(order="C")
            ),
        }
    full_oof = {
        variant: {
            "known_row_ids_sha256": canonical_json_sha256(
                [str(row["row_id"]) for row in evaluable_rows]
            ),
            "raw_probabilities": np.asarray(
                raw_known_oof[variant][evaluable_indices], dtype=np.float64
            ).tolist(),
            "probability_float64_sha256": sha256_bytes(
                np.asarray(
                    raw_known_oof[variant][evaluable_indices], dtype="<f8"
                ).tobytes(order="C")
            ),
        }
        for variant in VARIANTS
    }
    return {
        "algorithm": "nested_leave_one_repository_out",
        "ordered_row_identity_sha256": ordered_row_identity_sha256(rows),
        "known_ordered_row_identity_sha256": ordered_row_identity_sha256(known),
        "known_base_weight_float64_sha256": sha256_bytes(
            np.asarray(known_base_weights, dtype="<f8").tobytes(order="C")
        ),
        "point_estimand": (
            "equal_repository_then_equal_known_task_within_repository_then_"
            "equal_known_row_within_task"
        ),
        "base_weight_source": base_weight_source,
        "outer_fold_count": len(fold_records),
        "repositories_in_order": repositories_in_order,
        "all_stable_feature_complete_rows_predicted_once": True,
        "unknown_current_actions_received_predictions": True,
        "outer_heldout_labels_used_for_fit_calibration_or_threshold_selection": False,
        "folds": fold_records,
        "results": results,
        "full_development_selection": full_selection,
        "full_development_oof_raw_probabilities": full_oof,
    }


BOOTSTRAP_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "recall_inspect",
    "recall_edit",
    "recall_check_or_finish",
    "multiclass_negative_log_likelihood",
    "multiclass_brier",
    "top_label_ece",
    "selected_coverage",
    "selected_accepted_accuracy",
)
PAIRED_COMPARISONS = (
    ("history_j", "history_logit"),
    ("history_logit_j", "history_logit"),
)


def hierarchical_bayesian_bootstrap_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    draw_index: int,
    seed: int = 918273,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive one independent hierarchical Bayesian-bootstrap weight vector."""

    require(
        np.__version__ == BOOTSTRAP_NUMPY_VERSION,
        f"Bayesian bootstrap requires NumPy {BOOTSTRAP_NUMPY_VERSION}",
    )
    draw_index = integer(draw_index, "Bayesian bootstrap draw index", minimum=0)
    seed = integer(seed, "Bayesian bootstrap seed", minimum=0)
    require(bool(rows), "Bayesian bootstrap has no known rows")
    repositories = sorted({str(row["repo"]) for row in rows})
    require(
        all(
            any(str(row["repo"]) == repository for row in rows)
            for repository in repositories
        ),
        "Bayesian bootstrap repository support changed",
    )
    seed_sequence = np.random.SeedSequence([seed, draw_index])
    rng = np.random.Generator(np.random.PCG64(seed_sequence))
    repository_gamma = np.asarray(
        rng.gamma(shape=1.0, scale=1.0, size=len(repositories)),
        dtype=np.float64,
    )
    require(
        np.all(np.isfinite(repository_gamma))
        and np.all(repository_gamma > 0.0),
        "Bayesian bootstrap repository Gamma weights are invalid",
    )
    repository_weights = repository_gamma / repository_gamma.sum(dtype=np.float64)
    row_weights = np.zeros(len(rows), dtype=np.float64)
    task_weight_values: list[float] = []
    ordered_repository_tasks: list[dict[str, Any]] = []
    for repository_index, repository in enumerate(repositories):
        repository_row_indices = [
            index
            for index, row in enumerate(rows)
            if str(row["repo"]) == repository
        ]
        tasks = sorted(
            {str(rows[index]["task_id"]) for index in repository_row_indices}
        )
        require(bool(tasks), f"Bayesian bootstrap repository {repository} has no task")
        task_gamma = np.asarray(
            rng.gamma(shape=1.0, scale=1.0, size=len(tasks)),
            dtype=np.float64,
        )
        require(
            np.all(np.isfinite(task_gamma)) and np.all(task_gamma > 0.0),
            f"Bayesian bootstrap task Gamma weights are invalid in {repository}",
        )
        task_weights = task_gamma / task_gamma.sum(dtype=np.float64)
        for task_index, task_id in enumerate(tasks):
            indices = [
                index
                for index in repository_row_indices
                if str(rows[index]["task_id"]) == task_id
            ]
            require(bool(indices), "Bayesian bootstrap task has no known row")
            task_mass = float(repository_weights[repository_index]) * float(
                task_weights[task_index]
            )
            row_weights[np.asarray(indices, dtype=np.int64)] = task_mass / len(
                indices
            )
            task_weight_values.append(float(task_weights[task_index]))
            ordered_repository_tasks.append(
                {
                    "repo": repository,
                    "task_id": task_id,
                    "known_row_count": len(indices),
                }
            )
    normalized = _normalized_positive_weights(
        rows, row_weights, "Bayesian bootstrap known-row weights"
    )
    repository_little_endian = np.asarray(repository_weights, dtype="<f8")
    task_little_endian = np.asarray(task_weight_values, dtype="<f8")
    row_little_endian = np.asarray(normalized, dtype="<f8")
    return normalized, {
        "draw_index": draw_index,
        "seed_sequence_entropy": [seed, draw_index],
        "repository_count": len(repositories),
        "known_task_count": len(ordered_repository_tasks),
        "known_row_count": len(rows),
        "repositories_in_order_sha256": canonical_json_sha256(repositories),
        "repository_tasks_in_order_sha256": canonical_json_sha256(
            ordered_repository_tasks
        ),
        "repository_weight_float64_sha256": sha256_bytes(
            repository_little_endian.tobytes(order="C")
        ),
        "within_repository_task_weight_float64_sha256": sha256_bytes(
            task_little_endian.tobytes(order="C")
        ),
        "known_row_weight_float64_sha256": sha256_bytes(
            row_little_endian.tobytes(order="C")
        ),
        "all_original_repositories_tasks_and_rows_retained": True,
    }


def _credible_interval(
    values: Sequence[float], confidence_level: float, expected_draws: int
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(
        array.shape == (expected_draws,)
        and np.all(np.isfinite(array)),
        "credible interval requires every finite Bayesian-bootstrap draw",
    )
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "lower": float(
            np.quantile(array, alpha, method=BOOTSTRAP_QUANTILE_METHOD)
        ),
        "upper": float(
            np.quantile(array, 1.0 - alpha, method=BOOTSTRAP_QUANTILE_METHOD)
        ),
        "draw_count": expected_draws,
        "confidence_level": confidence_level,
        "interval_interpretation": BOOTSTRAP_INTERVAL_INTERPRETATION,
        "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
    }


def _valid_probability_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_metric_domain(metric: str, value: Any, label: str) -> float:
    number = finite(value, label)
    if metric in {
        "accuracy",
        "balanced_accuracy",
        "recall_inspect",
        "recall_edit",
        "recall_check_or_finish",
        "top_label_ece",
        "selected_coverage",
        "selected_accepted_accuracy",
    }:
        require(0.0 <= number <= 1.0, f"{label} is outside [0,1]")
    elif metric == "multiclass_negative_log_likelihood":
        require(number >= 0.0, f"{label} is negative")
    elif metric == "multiclass_brier":
        require(0.0 <= number <= 2.0, f"{label} is outside [0,2]")
    else:
        raise ValueError(f"unknown bootstrap metric: {metric}")
    return number


def _validate_selection_structure(
    value: Any,
    *,
    protocol: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    selection = mapping(value, label)
    calibration = mapping(selection.get("calibration"), f"{label} calibration")
    abstention = mapping(selection.get("abstention"), f"{label} abstention")
    temperature = finite(calibration.get("temperature"), f"{label} temperature")
    threshold = finite(abstention.get("threshold"), f"{label} threshold")
    require(
        temperature in [float(item) for item in protocol["calibration"]["temperatures"]],
        f"{label} temperature is off the frozen grid",
    )
    require(
        threshold in [float(item) for item in protocol["abstention"]["thresholds"]],
        f"{label} threshold is off the frozen grid",
    )
    require(
        bool(sequence(calibration.get("candidates"), f"{label} calibration candidates"))
        and bool(sequence(abstention.get("candidates"), f"{label} threshold candidates")),
        f"{label} selection candidates are empty",
    )
    return selection


def validate_nested_evidence(
    value: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    base_weights: Sequence[float] | np.ndarray | None = None,
    require_primary_selection_floors: bool = True,
) -> dict[str, Any]:
    """Recompute identity, coverage, probabilities, metrics, and full OOF settings."""

    nested = mapping(value, "nested evidence")
    rows = list(rows)
    known_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("metric_evaluable") is True
    ]
    known = [rows[index] for index in known_indices]
    require(bool(rows) and bool(known), "nested evidence source rows are empty")
    weights = (
        hierarchical_equal_weights(known)
        if base_weights is None
        else _normalized_positive_weights(
            known, base_weights, "nested evidence base weights"
        )
    )
    all_prediction_weights = hierarchical_equal_weights(rows)
    expected_known_action_fraction = float(
        all_prediction_weights[
            np.asarray(
                [row.get("metric_evaluable") is True for row in rows], dtype=bool
            )
        ].sum()
    )
    repositories = sorted({str(row["repo"]) for row in rows})
    require(
        nested.get("algorithm") == "nested_leave_one_repository_out"
        and nested.get("ordered_row_identity_sha256")
        == ordered_row_identity_sha256(rows)
        and nested.get("known_ordered_row_identity_sha256")
        == ordered_row_identity_sha256(known)
        and nested.get("known_base_weight_float64_sha256")
        == sha256_bytes(np.asarray(weights, dtype="<f8").tobytes(order="C"))
        and nested.get("point_estimand")
        == (
            "equal_repository_then_equal_known_task_within_repository_then_"
            "equal_known_row_within_task"
        )
        and nested.get("base_weight_source")
        == (
            "hierarchical_equal_point_weights"
            if base_weights is None
            else "hierarchical_bayesian_bootstrap_draw_weights"
        )
        and nested.get("repositories_in_order") == repositories,
        "nested evidence identity or base weights changed",
    )
    folds = [
        mapping(item, f"nested fold {index}")
        for index, item in enumerate(sequence(nested.get("folds"), "nested folds"))
    ]
    require(
        bool(folds)
        and len(folds) == len(repositories)
        and nested.get("outer_fold_count") == len(folds),
        "nested fold coverage is incomplete",
    )
    fold_settings_by_repository: dict[str, Mapping[str, Any]] = {}
    for fold, heldout_repository in zip(folds, repositories, strict=True):
        heldout_rows = [
            row for row in rows if str(row["repo"]) == heldout_repository
        ]
        outer_train_rows = [
            row for row in known if str(row["repo"]) != heldout_repository
        ]
        outer_train_indices = [
            index
            for index, row in enumerate(known)
            if str(row["repo"]) != heldout_repository
        ]
        outer_base_weights = restrict_base_weights(
            known, weights, outer_train_indices
        )
        _, outer_training_diagnostics = training_weights(
            outer_train_rows, outer_base_weights
        )
        require(
            fold.get("heldout_repository") == heldout_repository
            and fold.get("heldout_prediction_rows") == len(heldout_rows)
            and fold.get("heldout_known_action_rows")
            == sum(row.get("metric_evaluable") is True for row in heldout_rows)
            and fold.get("outer_training_known_action_rows")
            == len(outer_train_rows)
            and fold.get("outer_training_repositories")
            == sorted({str(row["repo"]) for row in outer_train_rows})
            and integer(fold.get("inner_fold_count"), "nested inner fold count")
            == len({str(row["repo"]) for row in outer_train_rows})
            and fold.get("inner_selection_row_ids_sha256")
            == canonical_json_sha256(
                [str(row["row_id"]) for row in outer_train_rows]
            )
            and fold.get("heldout_row_ids_sha256")
            == canonical_json_sha256(
                [str(row["row_id"]) for row in heldout_rows]
            )
            and fold.get("inner_and_heldout_row_ids_disjoint") is True
            and fold.get(
                "heldout_labels_used_for_fit_calibration_or_threshold_selection"
            )
            is False
            and fold.get(
                "same_folds_weights_seed_order_and_hyperparameters_across_variants"
            )
            is True,
            f"nested fold structure changed for {heldout_repository}",
        )
        require(
            fold.get("outer_training_base_weight_sha256")
            == sha256_bytes(
                np.asarray(outer_base_weights, dtype="<f8").tobytes(order="C")
            )
            and fold.get("shared_outer_training_weight_sha256")
            == outer_training_diagnostics["weight_float64_sha256"],
            f"nested fold weights changed for {heldout_repository}",
        )
        inner_folds = [
            mapping(item, f"{heldout_repository} inner fold {index}")
            for index, item in enumerate(
                sequence(fold.get("inner_folds"), "nested inner folds")
            )
        ]
        inner_repositories = sorted(
            {str(row["repo"]) for row in outer_train_rows}
        )
        require(
            len(inner_folds) == len(inner_repositories),
            f"nested inner fold records changed for {heldout_repository}",
        )
        outer_repository_array = np.asarray(
            [str(row["repo"]) for row in outer_train_rows]
        )
        for inner_fold, inner_heldout in zip(
            inner_folds, inner_repositories, strict=True
        ):
            inner_train_indices = np.flatnonzero(
                outer_repository_array != inner_heldout
            )
            inner_evaluation_indices = np.flatnonzero(
                outer_repository_array == inner_heldout
            )
            inner_train_rows = [
                outer_train_rows[int(index)] for index in inner_train_indices
            ]
            inner_evaluation_rows = [
                outer_train_rows[int(index)]
                for index in inner_evaluation_indices
            ]
            inner_base_weights = restrict_base_weights(
                outer_train_rows, outer_base_weights, inner_train_indices
            )
            _, inner_training_diagnostics = training_weights(
                inner_train_rows, inner_base_weights
            )
            require(
                inner_fold.get("heldout_repository") == inner_heldout
                and inner_fold.get("training_repositories")
                == sorted({str(row["repo"]) for row in inner_train_rows})
                and inner_fold.get("training_rows") == len(inner_train_rows)
                and inner_fold.get("evaluation_rows")
                == len(inner_evaluation_rows)
                and inner_fold.get("training_row_ids_sha256")
                == canonical_json_sha256(
                    [str(row["row_id"]) for row in inner_train_rows]
                )
                and inner_fold.get("evaluation_row_ids_sha256")
                == canonical_json_sha256(
                    [str(row["row_id"]) for row in inner_evaluation_rows]
                )
                and inner_fold.get(
                    "heldout_labels_used_for_fit_calibration_or_threshold_selection"
                )
                is False
                and inner_fold.get("training_base_weight_sha256")
                == sha256_bytes(
                    np.asarray(inner_base_weights, dtype="<f8").tobytes(
                        order="C"
                    )
                )
                and inner_fold.get("shared_training_weight_sha256")
                == inner_training_diagnostics["weight_float64_sha256"]
                and inner_fold.get("seed_order")
                == list(protocol["model"]["seeds"]),
                f"nested inner fold structure changed for {heldout_repository}/{inner_heldout}",
            )
        settings = mapping(fold.get("settings"), "nested fold settings")
        require(set(settings) == set(VARIANTS), "nested fold settings are incomplete")
        for variant in VARIANTS:
            _validate_selection_structure(
                settings[variant],
                protocol=protocol,
                label=f"{heldout_repository} {variant} selection",
            )
        fold_settings_by_repository[heldout_repository] = settings

    results = mapping(nested.get("results"), "nested results")
    require(set(results) == set(VARIANTS), "nested result variants changed")
    reconstructed_probabilities: dict[str, np.ndarray] = {}
    reconstructed_acceptance: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        result = mapping(results[variant], f"nested {variant} result")
        predictions = [
            mapping(item, f"{variant} prediction {index}")
            for index, item in enumerate(
                sequence(result.get("predictions"), f"{variant} predictions")
            )
        ]
        require(
            len(predictions) == len(rows)
            and result.get("inference_prediction_count") == len(rows)
            and result.get("known_action_metric_row_count") == len(known)
            and result.get("unknown_action_prediction_count")
            == len(rows) - len(known),
            f"{variant} prediction coverage changed",
        )
        probability_rows: list[list[float]] = []
        accepted_rows: list[bool] = []
        for row, prediction in zip(rows, predictions, strict=True):
            probabilities = mapping(
                prediction.get("probabilities"), f"{variant} prediction probabilities"
            )
            require(
                prediction.get("row_id") == row.get("row_id")
                and prediction.get("task_id") == row.get("task_id")
                and prediction.get("repo") == row.get("repo")
                and all(
                    prediction.get(key) == row.get(key)
                    for key in (
                        "cohort_id",
                        "task_request_index",
                        "checkpoint_ordinal",
                        "source_action_label_status",
                        "source_action_class_id",
                        "label_status",
                        "label",
                        "metric_evaluable",
                        "auxiliary_diagnostics",
                    )
                )
                and set(probabilities) == set(CLASSES),
                f"{variant} prediction row identity or class order changed",
            )
            vector = [
                finite(probabilities[class_id], f"{variant} {class_id} probability")
                for class_id in CLASSES
            ]
            require(
                all(item > 0.0 for item in vector)
                and math.isclose(sum(vector), 1.0, rel_tol=1e-12, abs_tol=1e-12),
                f"{variant} prediction probabilities are invalid",
            )
            predicted_index = int(np.argmax(np.asarray(vector, dtype=np.float64)))
            confidence = finite(prediction.get("confidence"), "prediction confidence")
            threshold = finite(
                prediction.get("confidence_threshold"), "prediction threshold"
            )
            temperature = finite(
                prediction.get("temperature"), "prediction temperature"
            )
            expected_fold_selection = mapping(
                fold_settings_by_repository[str(row["repo"])][variant],
                "prediction fold selection",
            )
            expected_temperature = finite(
                mapping(
                    expected_fold_selection.get("calibration"),
                    "prediction fold calibration",
                ).get("temperature"),
                "prediction fold temperature",
            )
            expected_threshold = finite(
                mapping(
                    expected_fold_selection.get("abstention"),
                    "prediction fold abstention",
                ).get("threshold"),
                "prediction fold threshold",
            )
            require(
                prediction.get("predicted_class") == CLASSES[predicted_index]
                and confidence == vector[predicted_index]
                and prediction.get("accepted") is (confidence >= threshold)
                and threshold
                in [float(item) for item in protocol["abstention"]["thresholds"]]
                and temperature
                in [float(item) for item in protocol["calibration"]["temperatures"]]
                and temperature == expected_temperature
                and threshold == expected_threshold,
                f"{variant} prediction decision fields changed",
            )
            probability_rows.append(vector)
            accepted_rows.append(bool(prediction["accepted"]))
        probability_array = np.asarray(probability_rows, dtype=np.float64)
        accepted_array = np.asarray(accepted_rows, dtype=bool)
        require(
            result.get("probability_float64_sha256")
            == sha256_bytes(
                np.asarray(probability_array, dtype="<f8").tobytes(order="C")
            )
            and result.get("accepted_inference_prediction_count")
            == int(accepted_array.sum())
            and finite(
                result.get("inference_acceptance_fraction"),
                f"{variant} inference acceptance fraction",
            )
            == float(accepted_array.mean()),
            f"{variant} probability hash or acceptance summary changed",
        )
        known_probability_array = probability_array[
            np.asarray(known_indices, dtype=np.int64)
        ]
        known_accepted_array = accepted_array[np.asarray(known_indices, dtype=np.int64)]
        expected_metrics = probability_metrics(
            known,
            known_probability_array,
            accepted=known_accepted_array,
            weights=weights,
        )
        stored_metrics = mapping(result.get("metrics"), f"{variant} metrics")
        for metric in BOOTSTRAP_METRICS:
            require(
                stored_metrics.get(metric) == expected_metrics.get(metric),
                f"{variant} nested metric {metric} does not reproduce",
            )
        expected_per_repository = _per_repository_metrics(
            known,
            known_probability_array,
            known_accepted_array,
            weights,
        )
        require(
            stored_metrics.get("known_action_fraction")
            == expected_known_action_fraction,
            f"{variant} known-action fraction changed",
        )
        require(
            result.get("per_repository_metrics") == expected_per_repository
            and result.get("inference_acceptance_fraction_denominator")
            == "all_stable_feature_complete_predictions"
            and stored_metrics.get("selected_coverage_denominator")
            == "known_current_action_metric_rows_only_not_all_stable_emissions",
            f"{variant} repository metrics or metric denominators changed",
        )
        reconstructed_probabilities[variant] = probability_array
        reconstructed_acceptance[variant] = accepted_array

    full_oof = mapping(
        nested.get("full_development_oof_raw_probabilities"),
        "full development OOF probabilities",
    )
    full_selection = mapping(
        nested.get("full_development_selection"), "full development selection"
    )
    require(
        set(full_oof) == set(VARIANTS)
        and set(full_selection) == set(VARIANTS),
        "full development OOF evidence or settings are incomplete",
    )
    known_ids_hash = canonical_json_sha256([str(row["row_id"]) for row in known])
    for variant in VARIANTS:
        record = mapping(full_oof[variant], f"{variant} full OOF record")
        raw = np.asarray(record.get("raw_probabilities"), dtype=np.float64)
        require(
            raw.shape == (len(known), len(CLASSES))
            and np.all(np.isfinite(raw))
            and np.all(raw > 0.0)
            and np.allclose(raw.sum(axis=1), 1.0)
            and record.get("known_row_ids_sha256") == known_ids_hash
            and record.get("probability_float64_sha256")
            == sha256_bytes(np.asarray(raw, dtype="<f8").tobytes(order="C")),
            f"{variant} full OOF probability evidence changed",
        )
        expected_selection = select_calibration_and_threshold(
            known, raw, protocol=protocol, weights=weights
        )
        selection = _validate_selection_structure(
            full_selection[variant],
            protocol=protocol,
            label=f"{variant} full development selection",
        )
        require(
            dict(selection) == expected_selection,
            f"{variant} full development selection does not reproduce from OOF probabilities",
        )
    primary = mapping(full_selection.get("history_j"), "primary full selection")
    primary_abstention = mapping(
        primary.get("abstention"), "primary full-selection abstention"
    )
    if require_primary_selection_floors:
        require(
            primary_abstention.get("selected_under_floors") is True
            and primary_abstention.get("fallback_used") is False,
            "primary full development selection did not meet frozen floors",
        )
    require(
        nested.get("all_stable_feature_complete_rows_predicted_once") is True
        and nested.get("unknown_current_actions_received_predictions") is True
        and nested.get(
            "outer_heldout_labels_used_for_fit_calibration_or_threshold_selection"
        )
        is False,
        "nested coverage/leakage declarations changed",
    )
    return {
        "ordered_row_identity_sha256": ordered_row_identity_sha256(rows),
        "known_ordered_row_identity_sha256": ordered_row_identity_sha256(known),
        "fold_count": len(folds),
        "prediction_count_per_variant": len(rows),
        "full_oof_selection_recomputed": True,
        "primary_selection_met_floors_without_fallback": bool(
            primary_abstention.get("selected_under_floors") is True
            and primary_abstention.get("fallback_used") is False
        ),
    }


def _zlib_base64_encode(value: bytes) -> str:
    """Encode exact binary evidence compactly with a frozen codec configuration."""

    return base64.b64encode(zlib.compress(value, level=9)).decode("ascii")


def _zlib_base64_decode_exact(value: Any, expected_size: int, label: str) -> bytes:
    """Decode bounded evidence while rejecting aliases, padding, and trailing data."""

    require(isinstance(value, str), f"{label} must be a base64 string")
    try:
        encoded = value.encode("ascii")
        compressed = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise ValueError(f"{label} is not canonical base64") from error
    require(
        base64.b64encode(compressed) == encoded,
        f"{label} is not canonical base64",
    )
    try:
        decompressor = zlib.decompressobj()
        decoded = decompressor.decompress(compressed, expected_size + 1)
    except zlib.error as error:
        raise ValueError(f"{label} is not valid zlib data") from error
    require(
        len(decoded) == expected_size
        and decompressor.eof
        and not decompressor.unconsumed_tail
        and not decompressor.unused_data
        and decompressor.flush() == b"",
        f"{label} decoded byte length or stream boundary changed",
    )
    require(
        _zlib_base64_encode(decoded) == value,
        f"{label} is not the canonical frozen level-9 zlib encoding",
    )
    return decoded


def _encode_bootstrap_row_prediction_evidence(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Mapping[str, np.ndarray],
    acceptance: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Losslessly retain all-row probabilities and a compact acceptance bitset."""

    row_count = len(rows)
    require(row_count > 0, "bootstrap row-prediction evidence has no rows")
    require(
        set(probabilities) == set(VARIANTS)
        and set(acceptance) == set(VARIANTS),
        "bootstrap row-prediction evidence variants changed",
    )
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        probability_array = np.asarray(probabilities[variant], dtype=np.float64)
        accepted_array = np.asarray(acceptance[variant], dtype=bool)
        require(
            probability_array.shape == (row_count, len(CLASSES))
            and np.all(np.isfinite(probability_array))
            and np.all(probability_array > 0.0)
            and np.allclose(
                probability_array.sum(axis=1),
                1.0,
                rtol=1e-12,
                atol=1e-12,
            )
            and accepted_array.shape == (row_count,),
            f"{variant} bootstrap row-prediction evidence is invalid",
        )
        probability_bytes = np.asarray(
            probability_array, dtype="<f8"
        ).tobytes(order="C")
        acceptance_bytes = np.packbits(
            accepted_array.astype(np.uint8), bitorder="little"
        ).tobytes(order="C")
        variants[variant] = {
            "probability_zlib_base64": _zlib_base64_encode(probability_bytes),
            "acceptance_zlib_base64": _zlib_base64_encode(acceptance_bytes),
        }
    return {
        "schema_version": BOOTSTRAP_ROW_EVIDENCE_SCHEMA_VERSION,
        "ordered_row_identity_sha256": ordered_row_identity_sha256(rows),
        "row_count": row_count,
        "class_order": list(CLASSES),
        "probability_shape": [row_count, len(CLASSES)],
        "probability_encoding": BOOTSTRAP_PROBABILITY_ENCODING,
        "acceptance_encoding": BOOTSTRAP_ACCEPTANCE_ENCODING,
        "variants": variants,
    }


def _decode_bootstrap_row_prediction_evidence(
    value: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    draw_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Decode and fully validate one draw's exact row-level retained evidence."""

    evidence = mapping(
        value, f"Bayesian-bootstrap draw {draw_index} row prediction evidence"
    )
    expected_fields = {
        "schema_version",
        "ordered_row_identity_sha256",
        "row_count",
        "class_order",
        "probability_shape",
        "probability_encoding",
        "acceptance_encoding",
        "variants",
    }
    row_count = len(rows)
    require(
        set(evidence) == expected_fields
        and evidence.get("schema_version")
        == BOOTSTRAP_ROW_EVIDENCE_SCHEMA_VERSION
        and evidence.get("ordered_row_identity_sha256")
        == ordered_row_identity_sha256(rows)
        and evidence.get("row_count") == row_count
        and evidence.get("class_order") == list(CLASSES)
        and evidence.get("probability_shape") == [row_count, len(CLASSES)]
        and evidence.get("probability_encoding")
        == BOOTSTRAP_PROBABILITY_ENCODING
        and evidence.get("acceptance_encoding")
        == BOOTSTRAP_ACCEPTANCE_ENCODING,
        f"Bayesian-bootstrap draw {draw_index} row prediction evidence contract changed",
    )
    variants = mapping(
        evidence.get("variants"),
        f"Bayesian-bootstrap draw {draw_index} row prediction evidence variants",
    )
    require(
        set(variants) == set(VARIANTS),
        f"Bayesian-bootstrap draw {draw_index} row prediction evidence variants changed",
    )
    probability_size = row_count * len(CLASSES) * np.dtype("<f8").itemsize
    acceptance_size = (row_count + 7) // 8
    decoded_probabilities: dict[str, np.ndarray] = {}
    decoded_acceptance: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        variant_evidence = mapping(
            variants[variant],
            f"Bayesian-bootstrap draw {draw_index} {variant} row prediction evidence",
        )
        require(
            set(variant_evidence)
            == {"probability_zlib_base64", "acceptance_zlib_base64"},
            f"Bayesian-bootstrap draw {draw_index} {variant} evidence fields changed",
        )
        probability_bytes = _zlib_base64_decode_exact(
            variant_evidence.get("probability_zlib_base64"),
            probability_size,
            f"draw {draw_index} {variant} probability evidence",
        )
        acceptance_bytes = _zlib_base64_decode_exact(
            variant_evidence.get("acceptance_zlib_base64"),
            acceptance_size,
            f"draw {draw_index} {variant} acceptance evidence",
        )
        probability_array = np.frombuffer(
            probability_bytes, dtype="<f8"
        ).reshape(row_count, len(CLASSES))
        packed_acceptance = np.frombuffer(acceptance_bytes, dtype=np.uint8)
        accepted_array = np.unpackbits(
            packed_acceptance, count=row_count, bitorder="little"
        ).astype(bool)
        canonical_acceptance_bytes = np.packbits(
            accepted_array.astype(np.uint8), bitorder="little"
        ).tobytes(order="C")
        require(
            canonical_acceptance_bytes == acceptance_bytes,
            f"draw {draw_index} {variant} acceptance padding bits changed",
        )
        require(
            np.all(np.isfinite(probability_array))
            and np.all(probability_array > 0.0)
            and np.allclose(
                probability_array.sum(axis=1),
                1.0,
                rtol=1e-12,
                atol=1e-12,
            ),
            f"draw {draw_index} {variant} decoded probabilities are invalid",
        )
        decoded_probabilities[variant] = probability_array
        decoded_acceptance[variant] = accepted_array
    return decoded_probabilities, decoded_acceptance


def _bootstrap_arrays_from_nested_results(
    value: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    draw_index: int,
) -> tuple[
    Mapping[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    """Extract the exact row outputs used to construct a compact draw record."""

    result_variants = mapping(
        mapping(value, "Bayesian-bootstrap nested result").get("results"),
        "Bayesian-bootstrap nested variant results",
    )
    require(
        set(result_variants) == set(VARIANTS),
        "Bayesian-bootstrap nested result variants changed",
    )
    probability_arrays: dict[str, np.ndarray] = {}
    acceptance_arrays: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        result = mapping(result_variants[variant], f"{variant} draw result")
        predictions = [
            mapping(item, f"draw {draw_index} {variant} prediction {index}")
            for index, item in enumerate(
                sequence(
                    result.get("predictions"),
                    f"draw {draw_index} {variant} predictions",
                )
            )
        ]
        require(
            len(predictions) == len(rows),
            f"draw {draw_index} {variant} prediction coverage changed",
        )
        probability_rows: list[list[float]] = []
        accepted_rows: list[bool] = []
        for row, prediction in zip(rows, predictions, strict=True):
            class_probabilities = mapping(
                prediction.get("probabilities"),
                f"draw {draw_index} {variant} prediction probabilities",
            )
            require(
                prediction.get("row_id") == row.get("row_id")
                and set(class_probabilities) == set(CLASSES),
                f"draw {draw_index} {variant} prediction identity changed",
            )
            vector = [
                finite(
                    class_probabilities[class_id],
                    f"draw {draw_index} {variant} {class_id} probability",
                )
                for class_id in CLASSES
            ]
            require(
                all(item > 0.0 for item in vector)
                and math.isclose(
                    sum(vector), 1.0, rel_tol=1e-12, abs_tol=1e-12
                ),
                f"draw {draw_index} {variant} prediction probabilities are invalid",
            )
            accepted_value = prediction.get("accepted")
            require(
                isinstance(accepted_value, bool),
                f"draw {draw_index} {variant} prediction acceptance is invalid",
            )
            probability_rows.append(vector)
            accepted_rows.append(accepted_value)
        probability_arrays[variant] = np.asarray(
            probability_rows, dtype=np.float64
        )
        acceptance_arrays[variant] = np.asarray(accepted_rows, dtype=bool)
    return result_variants, probability_arrays, acceptance_arrays


def _bootstrap_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    samples: int,
) -> dict[str, Any]:
    bootstrap = mapping(protocol.get("bootstrap"), "normalized bootstrap")
    return {
        "schema_version": BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
        "protocol_sha256": canonical_json_sha256(protocol["value"]),
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "requirements_sha256": V3_REQUIREMENTS_SHA256,
        "runtime_versions": _runtime_versions(),
        "ordered_row_identity_sha256": ordered_row_identity_sha256(rows),
        "known_ordered_row_identity_sha256": ordered_row_identity_sha256(
            known_rows(rows)
        ),
        "algorithm": BOOTSTRAP_ALGORITHM,
        "samples": samples,
        "seed": integer(bootstrap.get("seed"), "Bayesian bootstrap seed"),
        "confidence_level": finite(
            bootstrap.get("confidence_level"), "Bayesian bootstrap confidence"
        ),
        "interval_interpretation": BOOTSTRAP_INTERVAL_INTERPRETATION,
        "numpy_version": BOOTSTRAP_NUMPY_VERSION,
        "bit_generator": BOOTSTRAP_BIT_GENERATOR,
        "seed_sequence_per_draw": "SeedSequence([918273,draw_index])",
        "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        "persisted_evidence_scope": _expected_bootstrap_evidence_scope(),
        "all_draws_required": True,
        "variants_in_order": list(VARIANTS),
        "metrics_in_order": list(BOOTSTRAP_METRICS),
        "paired_comparisons_in_order": [
            f"{candidate}_minus_{reference}"
            for candidate, reference in PAIRED_COMPARISONS
        ],
    }


def _validated_draw_records(
    values: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    expected_count: int,
) -> list[dict[str, Any]]:
    rows = list(rows)
    records = [
        dict(mapping(item, f"Bayesian-bootstrap draw record {index}"))
        for index, item in enumerate(
            sequence(values, "Bayesian-bootstrap draw records")
        )
    ]
    require(
        len(records) == expected_count,
        "Bayesian-bootstrap draw record count changed",
    )
    known_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("metric_evaluable") is True
    ]
    known = [rows[index] for index in known_indices]
    known_index_array = np.asarray(known_indices, dtype=np.int64)
    seed = integer(protocol["bootstrap"].get("seed"), "Bayesian bootstrap seed")
    for draw_index, record in enumerate(records):
        require(
            record.get("draw_index") == draw_index
            and record.get("status") == "complete",
            "Bayesian-bootstrap draw indices are not contiguous and complete",
        )
        expected_draw_weights, expected_weight_record = (
            hierarchical_bayesian_bootstrap_weights(
                known, draw_index=draw_index, seed=seed
            )
        )
        for field in (
            "seed_sequence_entropy",
            "repository_count",
            "known_task_count",
            "known_row_count",
            "repositories_in_order_sha256",
            "repository_tasks_in_order_sha256",
            "repository_weight_float64_sha256",
            "within_repository_task_weight_float64_sha256",
            "known_row_weight_float64_sha256",
            "all_original_repositories_tasks_and_rows_retained",
        ):
            require(
                record.get(field) == expected_weight_record[field],
                f"Bayesian-bootstrap draw {draw_index} {field} changed",
            )
        decoded_probabilities, decoded_acceptance = (
            _decode_bootstrap_row_prediction_evidence(
                record.get("row_prediction_evidence"),
                rows=rows,
                draw_index=draw_index,
            )
        )
        probability_hashes = mapping(
            record.get("nested_probability_hashes"), "draw probability hashes"
        )
        require(
            set(probability_hashes) == set(VARIANTS),
            "draw probability hashes changed",
        )
        metrics = mapping(record.get("variant_metrics"), "draw variant metrics")
        require(set(metrics) == set(VARIANTS), "draw metric variants changed")
        normalized_metrics: dict[str, dict[str, float]] = {}
        for variant in VARIANTS:
            variant_metrics = mapping(metrics[variant], f"{variant} draw metrics")
            require(
                set(variant_metrics) == set(BOOTSTRAP_METRICS),
                f"{variant} draw metric names changed",
            )
            expected_metrics = probability_metrics(
                known,
                decoded_probabilities[variant][known_index_array],
                accepted=decoded_acceptance[variant][known_index_array],
                weights=expected_draw_weights,
            )
            normalized_metrics[variant] = {}
            for metric in BOOTSTRAP_METRICS:
                stored_metric = _validate_metric_domain(
                    metric,
                    variant_metrics[metric],
                    f"draw {draw_index} {variant} {metric}",
                )
                expected_metric = _validate_metric_domain(
                    metric,
                    expected_metrics[metric],
                    f"recomputed draw {draw_index} {variant} {metric}",
                )
                require(
                    stored_metric == expected_metric,
                    f"draw {draw_index} {variant} {metric} does not reproduce from row prediction evidence",
                )
                normalized_metrics[variant][metric] = expected_metric
            expected_probability_hash = sha256_bytes(
                np.asarray(
                    decoded_probabilities[variant], dtype="<f8"
                ).tobytes(order="C")
            )
            require(
                _valid_probability_hash(probability_hashes[variant])
                and probability_hashes[variant] == expected_probability_hash,
                f"draw {draw_index} {variant} probability hash does not reproduce from row prediction evidence",
            )
        pairs = mapping(record.get("paired_differences"), "draw paired differences")
        expected_pair_names = {
            f"{candidate}_minus_{reference}"
            for candidate, reference in PAIRED_COMPARISONS
        }
        require(set(pairs) == expected_pair_names, "draw paired comparisons changed")
        for candidate, reference in PAIRED_COMPARISONS:
            comparison = f"{candidate}_minus_{reference}"
            comparison_values = mapping(pairs[comparison], f"{comparison} draw values")
            require(
                set(comparison_values) == set(BOOTSTRAP_METRICS),
                f"{comparison} draw metric names changed",
            )
            for metric in BOOTSTRAP_METRICS:
                difference = finite(
                    comparison_values[metric],
                    f"draw {draw_index} {comparison} {metric}",
                )
                require(
                    difference
                    == normalized_metrics[candidate][metric]
                    - normalized_metrics[reference][metric],
                    f"draw {draw_index} {comparison} {metric} arithmetic changed",
                )
        stored_hash = record.get("draw_record_sha256")
        unhashed = dict(record)
        unhashed.pop("draw_record_sha256", None)
        require(
            stored_hash == canonical_json_sha256(unhashed),
            f"Bayesian-bootstrap draw {draw_index} record hash changed",
        )
    return records


def _bootstrap_intervals_from_records(
    records: Sequence[Mapping[str, Any]],
    confidence_level: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_count = len(records)
    intervals = {
        variant: {
            metric: _credible_interval(
                [
                    float(record["variant_metrics"][variant][metric])
                    for record in records
                ],
                confidence_level,
                sample_count,
            )
            for metric in BOOTSTRAP_METRICS
        }
        for variant in VARIANTS
    }
    paired = {
        f"{candidate}_minus_{reference}": {
            metric: _credible_interval(
                [
                    float(
                        record["paired_differences"][
                            f"{candidate}_minus_{reference}"
                        ][metric]
                    )
                    for record in records
                ],
                confidence_level,
                sample_count,
            )
            for metric in BOOTSTRAP_METRICS
        }
        for candidate, reference in PAIRED_COMPARISONS
    }
    return intervals, paired


def validate_bootstrap_evidence(
    value: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    require_production_samples: bool = True,
) -> dict[str, Any]:
    """Validate complete identity-bound draw records and recompute all bounds."""

    bootstrap = mapping(value, "Bayesian-bootstrap evidence")
    contract = mapping(protocol.get("bootstrap"), "normalized bootstrap")
    expected_samples = integer(contract.get("samples"), "bootstrap samples")
    samples = integer(bootstrap.get("samples"), "stored bootstrap samples", minimum=1)
    if require_production_samples:
        require(
            samples == expected_samples == 1000,
            "operational evidence requires exactly 1000 Bayesian-bootstrap draws",
        )
    identity = _bootstrap_identity(rows, protocol=protocol, samples=samples)
    require(
        bootstrap.get("status") == "complete_hierarchical_bayesian_bootstrap"
        and bootstrap.get("identity") == identity
        and bootstrap.get("algorithm") == BOOTSTRAP_ALGORITHM
        and bootstrap.get("protocol_sha256") == identity["protocol_sha256"]
        and bootstrap.get("analyzer_sha256") == identity["analyzer_sha256"]
        and bootstrap.get("requirements_sha256")
        == identity["requirements_sha256"]
        and bootstrap.get("runtime_versions") == identity["runtime_versions"]
        and bootstrap.get("ordered_row_identity_sha256")
        == identity["ordered_row_identity_sha256"]
        and bootstrap.get("seed") == identity["seed"]
        and bootstrap.get("confidence_level") == identity["confidence_level"]
        and bootstrap.get("interval_interpretation")
        == BOOTSTRAP_INTERVAL_INTERPRETATION
        and bootstrap.get("numpy_version") == BOOTSTRAP_NUMPY_VERSION
        and bootstrap.get("bit_generator") == BOOTSTRAP_BIT_GENERATOR
        and bootstrap.get("seed_sequence_per_draw")
        == "SeedSequence([918273,draw_index])"
        and bootstrap.get("quantile_method") == BOOTSTRAP_QUANTILE_METHOD
        and bootstrap.get("persisted_evidence_scope")
        == identity["persisted_evidence_scope"]
        and bootstrap.get("all_draws_complete") is True
        and bootstrap.get("models_refit_inside_bootstrap") is True
        and bootstrap.get(
            "models_refit_inside_bootstrap_is_in_process_execution_declaration_only"
        )
        is True
        and bootstrap.get(
            "calibration_and_threshold_reselected_inside_each_draw"
        )
        is True
        and bootstrap.get("same_draws_folds_weights_and_seed_order_across_variants")
        is True
        and bootstrap.get("all_original_rows_retained_every_draw") is True
        and bootstrap.get("retry_count") == 0,
        "Bayesian-bootstrap frozen evidence fields changed",
    )
    execution = mapping(
        bootstrap.get("execution"), "Bayesian-bootstrap execution evidence"
    )
    fit_execution = mapping(
        protocol["model"].get("fit_execution"), "model fit execution"
    )
    require(
        execution.get("draw_parallelism") == 1
        and execution.get("draw_order")
        == "serial_increasing_index_independent_seed_sequence"
        and execution.get("rng_state_stored") is False
        and execution.get("checkpoint_resume_supported") is True
        and integer(
            execution.get("resumed_from_draw"), "bootstrap resumed draw"
        )
        <= samples
        and execution.get("estimator_n_jobs")
        == int(protocol["model"]["parameters"]["n_jobs"])
        and execution.get("variant_seed_fit_parallelism")
        == int(fit_execution["worker_count"])
        and execution.get("variant_seed_fit_backend")
        == fit_execution["backend"]
        and execution.get("variant_seed_result_collection_order")
        == fit_execution["result_collection_order"],
        "Bayesian-bootstrap execution evidence changed",
    )
    records = _validated_draw_records(
        bootstrap.get("draw_records"),
        rows=rows,
        protocol=protocol,
        expected_count=samples,
    )
    require(
        bootstrap.get("draw_records_sha256") == canonical_json_sha256(records),
        "Bayesian-bootstrap draw-record array hash changed",
    )
    intervals, paired = _bootstrap_intervals_from_records(
        records, float(identity["confidence_level"])
    )
    require(
        bootstrap.get("intervals") == intervals
        and bootstrap.get("paired_differences") == paired,
        "Bayesian-bootstrap credible intervals do not reproduce from draw records",
    )
    return {
        "samples": samples,
        "draw_records_sha256": canonical_json_sha256(records),
        "ordered_row_identity_sha256": identity["ordered_row_identity_sha256"],
        "persisted_evidence_scope": identity["persisted_evidence_scope"],
        "all_draws_complete": True,
        "refit_execution_is_declared_not_independently_attested": True,
        "credible_intervals_recomputed_from_draw_records": True,
    }


def model_refit_hierarchical_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    samples: int | None = None,
    nested_runner: Callable[..., dict[str, Any]] = nested_leave_one_repository_out,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    validate_nested_draws: bool = True,
) -> dict[str, Any]:
    """Run full nested refits for every deterministic Bayesian-bootstrap draw."""

    require(
        np.__version__ == BOOTSTRAP_NUMPY_VERSION
        and zlib.ZLIB_RUNTIME_VERSION == FROZEN_ZLIB_VERSION,
        (
            f"Bayesian bootstrap requires NumPy {BOOTSTRAP_NUMPY_VERSION} "
            f"and zlib {FROZEN_ZLIB_VERSION}"
        ),
    )
    rows = list(rows)
    known_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("metric_evaluable") is True
    ]
    known = [rows[index] for index in known_indices]
    known_index_array = np.asarray(known_indices, dtype=np.int64)
    require(bool(rows) and bool(known), "Bayesian bootstrap source rows are empty")
    contract = mapping(protocol.get("bootstrap"), "normalized bootstrap")
    sample_count = integer(
        contract.get("samples") if samples is None else samples,
        "Bayesian bootstrap samples",
        minimum=1,
    )
    seed = integer(contract.get("seed"), "Bayesian bootstrap seed", minimum=0)
    confidence_level = finite(
        contract.get("confidence_level"), "Bayesian bootstrap confidence level"
    )
    require(
        checkpoint_path is not None or not resume,
        "Bayesian-bootstrap resume requires an explicit checkpoint path",
    )
    identity = _bootstrap_identity(
        rows, protocol=protocol, samples=sample_count
    )
    draw_records: list[dict[str, Any]] = []
    start_draw = 0
    if checkpoint_path is not None and resume:
        checkpoint = mapping(
            read_json(checkpoint_path, "Bayesian-bootstrap checkpoint"),
            "Bayesian-bootstrap checkpoint",
        )
        require(
            checkpoint.get("schema_version") == BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION
            and checkpoint.get("id")
            == "swe-task-state-interpreter-v3-bayesian-bootstrap-checkpoint"
            and checkpoint.get("identity") == identity,
            "Bayesian-bootstrap checkpoint identity differs from protocol/rows/samples/seed",
        )
        start_draw = integer(
            checkpoint.get("next_draw_index"),
            "Bayesian-bootstrap checkpoint next draw",
        )
        require(
            0 <= start_draw <= sample_count,
            "Bayesian-bootstrap checkpoint draw index exceeds sample count",
        )
        draw_records = _validated_draw_records(
            checkpoint.get("draw_records"),
            rows=rows,
            protocol=protocol,
            expected_count=start_draw,
        )
        require(
            checkpoint.get("draw_records_sha256")
            == canonical_json_sha256(draw_records),
            "Bayesian-bootstrap checkpoint draw-record array hash changed",
        )
        require(
            checkpoint.get("status")
            == ("complete" if start_draw == sample_count else "in_progress")
            and checkpoint.get("draw_records_are_sole_accumulator_source") is True
            and checkpoint.get("rng_state_stored") is False,
            "Bayesian-bootstrap checkpoint status or accumulator contract changed",
        )
    elif checkpoint_path is not None:
        require(
            not checkpoint_path.exists() and not checkpoint_path.is_symlink(),
            "Bayesian-bootstrap checkpoint already exists; resume explicitly",
        )

    def write_checkpoint(next_draw_index: int, status: str) -> None:
        if checkpoint_path is None:
            return
        atomic_write_json(
            checkpoint_path,
            {
                "schema_version": BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
                "id": "swe-task-state-interpreter-v3-bayesian-bootstrap-checkpoint",
                "status": status,
                "identity": identity,
                "next_draw_index": next_draw_index,
                "draw_records": draw_records,
                "draw_records_sha256": canonical_json_sha256(draw_records),
                "draw_records_are_sole_accumulator_source": True,
                "rng_state_stored": False,
            },
        )

    for draw_index in range(start_draw, sample_count):
        draw_weights, weight_record = hierarchical_bayesian_bootstrap_weights(
            known, draw_index=draw_index, seed=seed
        )
        draw_result = nested_runner(
            rows, protocol=protocol, base_weights=draw_weights
        )
        if validate_nested_draws:
            validate_nested_evidence(
                draw_result,
                rows=rows,
                protocol=protocol,
                base_weights=draw_weights,
                require_primary_selection_floors=False,
            )
        result_variants, probability_arrays, acceptance_arrays = (
            _bootstrap_arrays_from_nested_results(
                draw_result,
                rows=rows,
                draw_index=draw_index,
            )
        )
        row_prediction_evidence = _encode_bootstrap_row_prediction_evidence(
            rows, probability_arrays, acceptance_arrays
        )
        variant_metrics: dict[str, dict[str, float]] = {}
        probability_hashes: dict[str, str] = {}
        for variant in VARIANTS:
            result = mapping(result_variants[variant], f"{variant} draw result")
            metrics = mapping(result.get("metrics"), f"{variant} draw metrics")
            recomputed_metrics = probability_metrics(
                known,
                probability_arrays[variant][known_index_array],
                accepted=acceptance_arrays[variant][known_index_array],
                weights=draw_weights,
            )
            variant_metrics[variant] = {}
            for metric in BOOTSTRAP_METRICS:
                stored_metric = _validate_metric_domain(
                    metric,
                    metrics.get(metric),
                    f"draw {draw_index} {variant} {metric}",
                )
                recomputed_metric = _validate_metric_domain(
                    metric,
                    recomputed_metrics[metric],
                    f"recomputed draw {draw_index} {variant} {metric}",
                )
                require(
                    stored_metric == recomputed_metric,
                    f"draw {draw_index} {variant} {metric} does not reproduce from nested predictions",
                )
                variant_metrics[variant][metric] = recomputed_metric
            probability_hash = sha256_bytes(
                np.asarray(
                    probability_arrays[variant], dtype="<f8"
                ).tobytes(order="C")
            )
            require(
                result.get("probability_float64_sha256") == probability_hash,
                f"draw {draw_index} {variant} probability hash does not reproduce from nested predictions",
            )
            probability_hashes[variant] = probability_hash
        paired_differences = {
            f"{candidate}_minus_{reference}": {
                metric: variant_metrics[candidate][metric]
                - variant_metrics[reference][metric]
                for metric in BOOTSTRAP_METRICS
            }
            for candidate, reference in PAIRED_COMPARISONS
        }
        record: dict[str, Any] = {
            "draw_index": draw_index,
            "status": "complete",
            **weight_record,
            "variant_metrics": variant_metrics,
            "paired_differences": paired_differences,
            "nested_probability_hashes": probability_hashes,
            "row_prediction_evidence": row_prediction_evidence,
        }
        record["draw_record_sha256"] = canonical_json_sha256(record)
        draw_records.append(record)
        write_checkpoint(
            draw_index + 1,
            "complete" if draw_index + 1 == sample_count else "in_progress",
        )

    require(
        len(draw_records) == sample_count,
        "Bayesian bootstrap did not complete every requested draw",
    )
    validated_records = _validated_draw_records(
        draw_records,
        rows=rows,
        protocol=protocol,
        expected_count=sample_count,
    )
    intervals, paired = _bootstrap_intervals_from_records(
        validated_records, confidence_level
    )
    result = {
        "status": "complete_hierarchical_bayesian_bootstrap",
        "identity": identity,
        "algorithm": BOOTSTRAP_ALGORITHM,
        "protocol_sha256": identity["protocol_sha256"],
        "analyzer_sha256": identity["analyzer_sha256"],
        "requirements_sha256": identity["requirements_sha256"],
        "runtime_versions": identity["runtime_versions"],
        "ordered_row_identity_sha256": identity["ordered_row_identity_sha256"],
        "samples": sample_count,
        "seed": seed,
        "confidence_level": confidence_level,
        "interval_interpretation": BOOTSTRAP_INTERVAL_INTERPRETATION,
        "numpy_version": BOOTSTRAP_NUMPY_VERSION,
        "bit_generator": BOOTSTRAP_BIT_GENERATOR,
        "seed_sequence_per_draw": "SeedSequence([918273,draw_index])",
        "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        "persisted_evidence_scope": identity["persisted_evidence_scope"],
        "all_draws_complete": True,
        "models_refit_inside_bootstrap": True,
        "models_refit_inside_bootstrap_is_in_process_execution_declaration_only": True,
        "calibration_and_threshold_reselected_inside_each_draw": True,
        "same_draws_folds_weights_and_seed_order_across_variants": True,
        "all_original_rows_retained_every_draw": True,
        "retry_count": 0,
        "draw_records": validated_records,
        "draw_records_sha256": canonical_json_sha256(validated_records),
        "intervals": intervals,
        "paired_differences": paired,
        "execution": {
            "draw_parallelism": 1,
            "draw_order": "serial_increasing_index_independent_seed_sequence",
            "rng_state_stored": False,
            "checkpoint_resume_supported": True,
            "resumed_from_draw": start_draw,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "estimator_n_jobs": int(protocol["model"]["parameters"]["n_jobs"]),
            "variant_seed_fit_parallelism": int(
                protocol["model"]["fit_execution"]["worker_count"]
            ),
            "variant_seed_fit_backend": protocol["model"]["fit_execution"][
                "backend"
            ],
            "variant_seed_result_collection_order": protocol["model"][
                "fit_execution"
            ]["result_collection_order"],
        },
    }
    validate_bootstrap_evidence(
        result,
        rows=rows,
        protocol=protocol,
        require_production_samples=samples is None,
    )
    write_checkpoint(sample_count, "complete")
    return result


def missing_model_refit_bootstrap(reason: str) -> dict[str, Any]:
    """Return an explicit fail-closed diagnostic record, never fake intervals."""

    return {
        "status": "absent_fail_closed",
        "reason": reason,
        "models_refit_inside_bootstrap": False,
        "models_refit_inside_bootstrap_is_in_process_execution_declaration_only": True,
        "persisted_evidence_scope": _expected_bootstrap_evidence_scope(),
        "calibration_and_threshold_reselected_inside_each_draw": False,
        "all_draws_complete": False,
        "samples": 0,
        "draw_records": [],
        "intervals": {},
        "paired_differences": {},
        "operational_reliability_credible_intervals_available": False,
    }


def support_summary(
    rows: Sequence[Mapping[str, Any]], eligibility: Mapping[str, Any]
) -> dict[str, Any]:
    known = known_rows(rows)
    stable_final_prompt_rows = integer(
        eligibility.get("numerically_stable_prompt_count"),
        "numerically stable final-prompt rows",
    )
    feature_complete_rows = integer(
        eligibility.get("stable_feature_complete_prediction_count"),
        "stable feature-complete prediction rows",
    )
    require(
        feature_complete_rows == len(rows)
        and feature_complete_rows <= stable_final_prompt_rows,
        "stable feature-completion eligibility counts are inconsistent",
    )
    feature_completion_fraction = (
        feature_complete_rows / stable_final_prompt_rows
        if stable_final_prompt_rows
        else 0.0
    )
    declared_feature_completion_fraction = finite(
        eligibility.get("stable_feature_complete_prediction_fraction"),
        "stable feature-complete prediction fraction",
    )
    require(
        math.isclose(
            feature_completion_fraction,
            declared_feature_completion_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "stable feature-completion fraction differs from its eligibility counts",
    )
    all_row_weights = hierarchical_equal_weights(rows) if rows else np.asarray([])
    known_mask = np.asarray(
        [row.get("metric_evaluable") is True for row in rows], dtype=bool
    )
    hierarchical_known_fraction = (
        float(all_row_weights[known_mask].sum()) if rows else 0.0
    )
    known_tasks_by_class = {
        class_id: len(
            {
                str(row["task_id"])
                for row in known
                if row.get("label") == class_id
            }
        )
        for class_id in CLASSES
    }
    known_repositories_by_class = {
        class_id: len(
            {
                str(row["repo"])
                for row in known
                if row.get("label") == class_id
            }
        )
        for class_id in CLASSES
    }
    return {
        "stable_prediction_rows": len(rows),
        "numerically_stable_final_prompt_rows": stable_final_prompt_rows,
        "stable_feature_complete_prediction_rows": feature_complete_rows,
        "stable_feature_complete_prediction_fraction": feature_completion_fraction,
        "stable_feature_complete_prediction_fraction_numerator": feature_complete_rows,
        "stable_feature_complete_prediction_fraction_denominator": stable_final_prompt_rows,
        "known_action_rows": len(known),
        "unknown_action_rows": len(rows) - len(known),
        "prediction_tasks": len({str(row["task_id"]) for row in rows}),
        "prediction_repositories": len({str(row["repo"]) for row in rows}),
        "known_action_tasks": len({str(row["task_id"]) for row in known}),
        "known_action_repositories": len({str(row["repo"]) for row in known}),
        "known_action_fraction": len(known) / len(rows) if rows else 0.0,
        "hierarchical_known_action_fraction": hierarchical_known_fraction,
        "hierarchical_known_action_fraction_estimand": (
            "equal_repository_then_equal_prediction_task_within_repository_"
            "then_equal_prediction_row_within_task"
        ),
        "known_inspect_tasks": known_tasks_by_class["inspect"],
        "known_edit_tasks": known_tasks_by_class["edit"],
        "known_check_or_finish_tasks": known_tasks_by_class["check_or_finish"],
        "known_inspect_repositories": known_repositories_by_class["inspect"],
        "known_edit_repositories": known_repositories_by_class["edit"],
        "known_check_or_finish_repositories": known_repositories_by_class[
            "check_or_finish"
        ],
        "numerical_stability_fraction": float(
            eligibility["numerical_stability_fraction"]
        ),
        "target_class_rows": dict(
            sorted(Counter(str(row["label"]) for row in known).items())
        ),
        "unknown_actions_explicitly_counted": True,
    }


def _compare_gate(observed: float, operator: str, threshold: float) -> bool:
    if operator == "minimum_inclusive":
        return observed >= threshold
    if operator == "minimum_exclusive":
        return observed > threshold
    if operator == "maximum_inclusive":
        return observed <= threshold
    if operator == "maximum_exclusive":
        return observed < threshold
    raise ValueError(f"unsupported gate operator: {operator}")


def evaluate_gates(
    *,
    protocol: Mapping[str, Any],
    support: Mapping[str, Any],
    nested: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    gate_contract = mapping(protocol.get("gates"), "normalized gate contract")
    support_contract = mapping(gate_contract.get("support"), "support gates")
    support_key_map = {
        "minimum_stable_prediction_rows": "stable_prediction_rows",
        "minimum_known_action_rows": "known_action_rows",
        "minimum_prediction_tasks": "prediction_tasks",
        "minimum_prediction_repositories": "prediction_repositories",
        "minimum_known_action_tasks": "known_action_tasks",
        "minimum_known_action_repositories": "known_action_repositories",
        "minimum_hierarchical_known_action_fraction": (
            "hierarchical_known_action_fraction"
        ),
        "minimum_known_inspect_tasks": "known_inspect_tasks",
        "minimum_known_edit_tasks": "known_edit_tasks",
        "minimum_known_check_or_finish_tasks": "known_check_or_finish_tasks",
        "minimum_known_inspect_repositories": "known_inspect_repositories",
        "minimum_known_edit_repositories": "known_edit_repositories",
        "minimum_known_check_or_finish_repositories": (
            "known_check_or_finish_repositories"
        ),
        "minimum_numerical_stability_fraction": "numerical_stability_fraction",
        "minimum_stable_feature_complete_prediction_fraction": (
            "stable_feature_complete_prediction_fraction"
        ),
    }
    support_results: list[dict[str, Any]] = []
    for gate_key, support_key in support_key_map.items():
        threshold = finite(support_contract.get(gate_key), gate_key)
        observed = finite(support.get(support_key), support_key)
        support_results.append(
            {
                "id": gate_key,
                "observed": observed,
                "threshold": threshold,
                "operator": "minimum_inclusive",
                "passed": observed >= threshold,
            }
        )

    evidence_errors: list[str] = []
    nested_reliable = False
    bootstrap_reliable = False
    if rows is None:
        evidence_errors.append(
            "ordered source rows were not supplied for structural evidence validation"
        )
    else:
        try:
            validate_nested_evidence(
                nested, rows=rows, protocol=protocol
            )
            nested_reliable = True
        except (KeyError, TypeError, ValueError) as error:
            evidence_errors.append(f"nested evidence invalid: {error}")
        try:
            validate_bootstrap_evidence(
                bootstrap,
                rows=rows,
                protocol=protocol,
                require_production_samples=True,
            )
            bootstrap_reliable = True
        except (KeyError, TypeError, ValueError) as error:
            evidence_errors.append(f"Bayesian-bootstrap evidence invalid: {error}")
    results_by_variant = mapping(nested.get("results"), "nested results")
    absolute_results: list[dict[str, Any]] = []
    for raw_gate in sequence(gate_contract.get("absolute"), "absolute gates"):
        gate = mapping(raw_gate, "absolute gate")
        variant = nonempty_string(gate.get("variant"), "gate variant")
        metric = nonempty_string(gate.get("metric"), "gate metric")
        bound = nonempty_string(gate.get("bound"), "gate bound")
        threshold = finite(gate.get("value"), "gate threshold")
        operator = nonempty_string(gate.get("operator"), "gate operator")
        metrics = mapping(
            mapping(results_by_variant.get(variant), "variant result").get("metrics"),
            "variant metrics",
        )
        point = metrics.get(metric)
        interval_reliable = bound == "point" or bootstrap_reliable
        if bound == "point":
            observed = point
        else:
            interval = mapping(
                mapping(
                    mapping(bootstrap.get("intervals", {}), "bootstrap intervals").get(
                        variant, {}
                    ),
                    "variant intervals",
                ).get(metric, {}),
                "metric interval",
            )
            observed = interval.get(
                "lower" if bound == "bootstrap_lower" else "upper"
            )
            interval_reliable = (
                interval_reliable
                and observed is not None
            )
        passed = (
            interval_reliable
            and observed is not None
            and _compare_gate(float(observed), operator, threshold)
        )
        absolute_results.append(
            {
                **dict(gate),
                "point": point,
                "observed": observed,
                "model_refit_interval_required": bound != "point",
                "model_refit_interval_available_and_valid": interval_reliable,
                "credible_interval_interpretation": (
                    BOOTSTRAP_INTERVAL_INTERPRETATION
                    if bound != "point"
                    else None
                ),
                "passed": bool(passed),
            }
        )

    paired_results: list[dict[str, Any]] = []
    for raw_gate in sequence(gate_contract.get("paired"), "paired gates"):
        gate = mapping(raw_gate, "paired gate")
        candidate = nonempty_string(gate.get("candidate"), "candidate")
        reference = nonempty_string(gate.get("reference"), "reference")
        metric = nonempty_string(gate.get("metric"), "paired metric")
        bound = nonempty_string(gate.get("bound"), "paired bound")
        operator = nonempty_string(gate.get("operator"), "paired operator")
        threshold = finite(gate.get("value"), "paired threshold")
        candidate_point = results_by_variant[candidate]["metrics"].get(metric)
        reference_point = results_by_variant[reference]["metrics"].get(metric)
        point = (
            float(candidate_point) - float(reference_point)
            if candidate_point is not None and reference_point is not None
            else None
        )
        comparison = f"{candidate}_minus_{reference}"
        interval = mapping(
            mapping(
                mapping(
                    bootstrap.get("paired_differences", {}),
                    "paired bootstrap intervals",
                ).get(comparison, {}),
                "paired comparison intervals",
            ).get(metric, {}),
            "paired metric interval",
        )
        observed = interval.get(
            "lower" if bound == "bootstrap_lower" else "upper"
        )
        interval_reliable = (
            bootstrap_reliable
            and observed is not None
        )
        passed = (
            interval_reliable
            and _compare_gate(float(observed), operator, threshold)
        )
        paired_results.append(
            {
                **dict(gate),
                "point_candidate_minus_reference": point,
                "observed": observed,
                "model_refit_interval_available_and_valid": interval_reliable,
                "credible_interval_interpretation": (
                    BOOTSTRAP_INTERVAL_INTERPRETATION
                ),
                "passed": bool(passed),
            }
        )

    all_results = [*support_results, *absolute_results, *paired_results]
    blockers: list[str] = []
    if not bootstrap_reliable:
        blockers.append(
            "complete 1000-draw hierarchical Bayesian-bootstrap credible intervals are absent or invalid"
        )
    if not nested_reliable:
        blockers.append("nested LORO evidence is structurally invalid")
    blockers.extend(evidence_errors)
    passed = not blockers and all(result["passed"] for result in all_results)
    return {
        "passed": passed,
        "operational_reliability_claim": passed,
        "primary_branch": "history_j",
        "support": support_results,
        "absolute": absolute_results,
        "paired": paired_results,
        "model_refit_bootstrap_required": True,
        "model_refit_bootstrap_reliable": bootstrap_reliable,
        "nested_evidence_reliable": nested_reliable,
        "reliability_blockers": blockers,
        "failed_gate_ids": [
            str(result["id"]) for result in all_results if not result["passed"]
        ],
    }


def auxiliary_diagnostic_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    milestone_status: Counter[str] = Counter()
    milestone_labels: Counter[str] = Counter()
    consequential_status: Counter[str] = Counter()
    consequential_labels: Counter[str] = Counter()
    for row in rows:
        diagnostics = mapping(row.get("auxiliary_diagnostics"), "auxiliary diagnostics")
        milestone = mapping(diagnostics.get("milestone_within_2"), "milestone diagnostic")
        consequential = mapping(
            diagnostics.get("current_consequential_source_type"),
            "consequential diagnostic",
        )
        milestone_status[str(milestone.get("status"))] += 1
        milestone_labels[str(milestone.get("label") or milestone.get("reason"))] += 1
        consequential_status[str(consequential.get("status"))] += 1
        consequential_labels[str(consequential.get("label") or "unknown")] += 1
    return {
        "role": "diagnostic_only_not_an_operational_model_or_gate",
        "future_fields_enter_any_feature_fit_calibration_threshold_or_gate": False,
        "milestone_within_2": {
            "window_offsets_inclusive": [0, 1],
            "arbitrary_horizon_inspection_skip_forbidden": True,
            "status_support": dict(sorted(milestone_status.items())),
            "label_or_censor_support": dict(sorted(milestone_labels.items())),
        },
        "current_consequential_source_type": {
            "status_support": dict(sorted(consequential_status.items())),
            "label_support": dict(sorted(consequential_labels.items())),
        },
    }


def _runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "zlib": zlib.ZLIB_RUNTIME_VERSION,
    }
    for package in (
        "scikit-learn",
        "joblib",
        "scipy",
        "threadpoolctl",
        "ijson",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def validate_development_bundle(
    *,
    cohort_path: Path,
    prompts_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Authenticate the exact N60 declaration and every materialized payload."""

    require(
        cohort_path.expanduser().resolve(strict=True)
        == DEFAULT_DEVELOPMENT_COHORT.resolve(strict=True),
        "development analysis requires the exact V3 N60 cohort manifest path",
    )
    require(
        prompts_path.is_file() and not prompts_path.is_symlink(),
        "development prompts must be a regular file",
    )
    require(
        summary_path.is_file() and not summary_path.is_symlink(),
        "development prompts summary must be a regular file",
    )
    declaration = COHORT_CHECKER.validate_declaration(cohort_path)
    checker_result = COHORT_CHECKER.validate_materialized_bundle(
        declaration,
        prompts_path=prompts_path,
        summary_path=summary_path,
    )
    require(
        checker_result.get("cohort_count") == 2
        and checker_result.get("task_count") == 60
        and checker_result.get("cohort_manifest_sha256")
        == sha256_file(cohort_path)
        and checker_result.get("prompt_bundle_sha256") == sha256_file(prompts_path)
        and checker_result.get("summary_sha256") == sha256_file(summary_path),
        "materialized V3 bundle checker returned an inconsistent binding",
    )
    return {
        **dict(checker_result),
        "checker_sha256": COHORT_CHECKER_SHA256,
        "exact_n60_declaration_validated": True,
        "run_image_manifest_and_task_runner_image_bindings_validated": True,
        "materialized_source_image_manifest_provenance_validated": True,
        "every_prompt_payload_and_provenance_binding_validated": True,
        "task_cohort_campaign_order_validated": True,
    }


def validate_replay_merge_receipt(
    *,
    receipt_path: Path,
    public_report_path: Path,
    prompts_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Delegate full replay authentication to the pinned standalone pipeline."""

    require(
        receipt_path.is_file()
        and not receipt_path.is_symlink()
        and public_report_path.is_file()
        and not public_report_path.is_symlink()
        and prompts_path.is_file()
        and not prompts_path.is_symlink()
        and summary_path.is_file()
        and not summary_path.is_symlink(),
        "replay receipt/report and materialized inputs must be regular files",
    )
    validated = REPLAY_PIPELINE.validate_merge_receipt(
        report_path=public_report_path,
        merge_manifest_path=receipt_path,
        replay_root=receipt_path.parent,
        prompts_path=prompts_path,
        summary_path=summary_path,
    )
    require(
        Path(validated.report_path).resolve(strict=True)
        == public_report_path.resolve(strict=True)
        and Path(validated.merge_manifest_path).resolve(strict=True)
        == receipt_path.resolve(strict=True)
        and validated.report_sha256 == sha256_file(public_report_path)
        and validated.merge_manifest_sha256 == sha256_file(receipt_path)
        and validated.prompt_bundle_sha256 == sha256_file(prompts_path)
        and integer(validated.experiment_count, "validated replay experiment count", minimum=1)
        > 0
        and _valid_probability_hash(validated.materialization_receipt_sha256)
        and _valid_git_commit(validated.source_freeze_git_commit)
        and _valid_git_commit(validated.data_freeze_git_commit),
        "pinned replay pipeline returned an inconsistent authenticated receipt",
    )
    return {
        "validator": "pinned_standalone_replay_pipeline_validate_merge_receipt",
        "replay_pipeline_sha256": REPLAY_PIPELINE_SHA256,
        "replay_shell_wrapper_sha256": REPLAY_SHELL_WRAPPER_SHA256,
        "report_sha256": validated.report_sha256,
        "merge_manifest_sha256": validated.merge_manifest_sha256,
        "experiment_count": int(validated.experiment_count),
        "prompt_bundle_sha256": validated.prompt_bundle_sha256,
        "materialization_receipt_sha256": validated.materialization_receipt_sha256,
        "source_freeze_git_commit": validated.source_freeze_git_commit,
        "data_freeze_git_commit": validated.data_freeze_git_commit,
        "full_chunk_sources_and_lossless_merge_reauthenticated": True,
    }


def _prepare_inputs(args: argparse.Namespace) -> dict[str, Any]:
    expected_runtime = {
        "numpy": BOOTSTRAP_NUMPY_VERSION,
        "scikit-learn": FROZEN_SKLEARN_VERSION,
        "joblib": FROZEN_JOBLIB_VERSION,
        "scipy": FROZEN_SCIPY_VERSION,
        "threadpoolctl": FROZEN_THREADPOOLCTL_VERSION,
        "ijson": FROZEN_IJSON_VERSION,
        "zlib": FROZEN_ZLIB_VERSION,
    }
    try:
        installed_runtime = {
            "numpy": np.__version__,
            "zlib": zlib.ZLIB_RUNTIME_VERSION,
            **{
                package: importlib.metadata.version(package)
                for package in expected_runtime
                if package not in {"numpy", "zlib"}
            },
        }
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(
            "production analyzer is missing a frozen runtime dependency"
        ) from error
    require(
        installed_runtime == expected_runtime,
        "production analyzer requires the exact frozen model/analyzer runtime",
    )
    protocol_value = read_json(args.protocol, "V3 protocol")
    action_protocol_value = read_json(args.action_protocol, "V3 action protocol")
    require(
        sha256_file(args.protocol) == sha256_file(DEFAULT_PROTOCOL),
        "CLI V3 protocol bytes differ from the canonical predeclared file",
    )
    require(
        sha256_file(args.action_protocol) == V3_ACTION_PROTOCOL_SHA256,
        "CLI V3 action protocol SHA-256 changed",
    )
    protocol = validate_protocol(
        protocol_value, action_protocol_value=action_protocol_value
    )
    development_binding = validate_development_bundle(
        cohort_path=args.development_cohort,
        prompts_path=args.prompts,
        summary_path=args.prompts_summary,
    )
    replay_merge_binding = validate_replay_merge_receipt(
        receipt_path=args.replay_merge_receipt,
        public_report_path=args.public_report,
        prompts_path=args.prompts,
        summary_path=args.prompts_summary,
    )
    extracted = extract_stable_rows_streaming(
        args.prompts,
        args.public_report,
        protocol=protocol,
    )
    hashes = {
        "prompts": sha256_file(args.prompts),
        "public_report": sha256_file(args.public_report),
        "protocol": sha256_file(args.protocol),
        "action_protocol": sha256_file(args.action_protocol),
        "analyzer": sha256_file(Path(__file__).resolve()),
        "historical_v1_analyzer": sha256_file(V1_ANALYZER_PATH),
        "historical_v2_analyzer": sha256_file(V2_ANALYZER_PATH),
        "historical_v1_protocol": sha256_file(V1_PROTOCOL_PATH),
        "historical_v2_protocol": sha256_file(V2_PROTOCOL_PATH),
        "behavioral_protocol": sha256_file(BEHAVIORAL_PROTOCOL_PATH),
        "materialized_bundle_checker": sha256_file(COHORT_CHECKER_PATH),
        "v3_requirements": sha256_file(V3_REQUIREMENTS_PATH),
        "development_cohort_manifest": sha256_file(args.development_cohort),
        "development_prompts_summary": sha256_file(args.prompts_summary),
        "replay_merge_receipt": sha256_file(args.replay_merge_receipt),
    }
    return {
        "protocol": protocol,
        "rows": extracted["rows"],
        "eligibility": extracted["eligibility"],
        "hashes": hashes,
        "development_binding": development_binding,
        "replay_merge_binding": replay_merge_binding,
    }


def analyze_command(args: argparse.Namespace) -> int:
    require(
        not args.resume_bootstrap or args.bootstrap_checkpoint is not None,
        "--resume-bootstrap requires --bootstrap-checkpoint",
    )
    require(
        not args.diagnostic_without_model_refit_bootstrap
        or (args.bootstrap_checkpoint is None and not args.resume_bootstrap),
        "diagnostic no-bootstrap mode cannot create or resume a bootstrap checkpoint",
    )
    prepared = _prepare_inputs(args)
    rows = prepared["rows"]
    protocol = prepared["protocol"]
    nested = nested_leave_one_repository_out(rows, protocol=protocol)
    if args.diagnostic_without_model_refit_bootstrap:
        bootstrap = missing_model_refit_bootstrap(
            "explicit CLI diagnostic mode skipped the predeclared expensive model-refit bootstrap"
        )
    else:
        bootstrap = model_refit_hierarchical_bootstrap(
            rows,
            protocol=protocol,
            checkpoint_path=args.bootstrap_checkpoint,
            resume=args.resume_bootstrap,
        )
    support = support_summary(rows, prepared["eligibility"])
    gates = evaluate_gates(
        protocol=protocol,
        support=support,
        nested=nested,
        bootstrap=bootstrap,
        rows=rows,
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "id": "swe-task-state-interpreter-v3-development-analysis",
        "scope": "fresh_development_only_not_reserved_validation",
        "inputs": prepared["hashes"],
        "development_data_binding": prepared["development_binding"],
        "replay_merge_binding": prepared["replay_merge_binding"],
        "target_contract": {
            "classes_in_order": list(CLASSES),
            "source_actions_in_order": list(SOURCE_ACTION_CLASSES),
            "collapse": dict(COLLAPSE),
            "ensuing_same_request_completion_action": True,
            "unknown_actions_receive_predictions_but_are_metric_ineligible": True,
            "target_is_prospective_ensuing_same_request_completion": True,
            "later_request_actions_used_for_target_or_features": False,
        },
        "feature_contract": {
            "variants_in_order": list(VARIANTS),
            "variant_widths": dict(VARIANT_WIDTHS),
            "feature_names": {variant: feature_names(variant) for variant in VARIANTS},
            "history_computed_before_current_action": True,
            "excluded_temporal_or_lexical_families": list(
                prepared["protocol"]["value"]["feature_contract"][
                    "excluded_feature_families"
                ]
            ),
        },
        "eligibility": prepared["eligibility"],
        "support": support,
        "auxiliary_diagnostics": auxiliary_diagnostic_summary(rows),
        "nested_development_evaluation": nested,
        "bootstrap": bootstrap,
        "development_gates": gates,
        "operational_reliability_claim": gates["passed"],
        "reserved_validation_allowed": gates["passed"],
        "interpretation": (
            "Predictions concern the observable ensuing same-request completion action only. Auxiliary bounded "
            "diagnostics and all labels are excluded from hidden-prose or chain-of-thought claims."
        ),
    }
    atomic_write_json_no_clobber(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "stable_predictions": len(rows),
                "known_action_metric_rows": len(known_rows(rows)),
                "unknown_action_predictions": len(rows) - len(known_rows(rows)),
                "model_refit_bootstrap_status": bootstrap["status"],
                "development_gates_passed": gates["passed"],
                "operational_reliability_claim": gates["passed"],
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_analysis_for_fit(
    analysis: Any, *, prepared: Mapping[str, Any]
) -> Mapping[str, Any]:
    value = mapping(analysis, "development analysis")
    require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("id")
        == "swe-task-state-interpreter-v3-development-analysis",
        "development analysis identity changed",
    )
    require(
        value.get("inputs") == prepared["hashes"],
        "development analysis is not bound to the supplied inputs/analyzer",
    )
    require(
        value.get("development_data_binding") == prepared["development_binding"],
        "development analysis N60 cohort/summary binding changed",
    )
    require(
        value.get("replay_merge_binding") == prepared["replay_merge_binding"],
        "development analysis replay merge receipt binding changed",
    )
    require(
        value.get("eligibility") == prepared["eligibility"],
        "development analysis eligibility differs from streamed input extraction",
    )
    stored_support = mapping(value.get("support"), "development support")
    require(
        dict(stored_support)
        == support_summary(prepared["rows"], prepared["eligibility"]),
        "development support does not reproduce from streamed eligibility/rows",
    )
    stored_nested = mapping(
        value.get("nested_development_evaluation"),
        "nested development evaluation",
    )
    validate_nested_evidence(
        stored_nested,
        rows=prepared["rows"],
        protocol=prepared["protocol"],
    )
    validate_bootstrap_evidence(
        mapping(value.get("bootstrap"), "development bootstrap"),
        rows=prepared["rows"],
        protocol=prepared["protocol"],
        require_production_samples=True,
    )
    recomputed_nested = nested_leave_one_repository_out(
        prepared["rows"], protocol=prepared["protocol"]
    )
    require(
        recomputed_nested == dict(stored_nested),
        "nested development evidence does not reproduce from the authenticated rows",
    )
    gates = mapping(value.get("development_gates"), "development gates")
    recomputed_gates = evaluate_gates(
        protocol=prepared["protocol"],
        support=stored_support,
        nested=stored_nested,
        bootstrap=mapping(value.get("bootstrap"), "development bootstrap"),
        rows=prepared["rows"],
    )
    require(
        dict(gates) == recomputed_gates,
        "development gate record does not reproduce from stored evidence",
    )
    require(
        gates.get("passed") is True
        and gates.get("model_refit_bootstrap_reliable") is True
        and value.get("operational_reliability_claim") is True,
        "development gates failed or lack reliable model-refit intervals; fit is forbidden",
    )
    nested = mapping(
        value.get("nested_development_evaluation"), "nested development evaluation"
    )
    require(
        nested.get("all_stable_feature_complete_rows_predicted_once") is True,
        "development analysis has incomplete prediction coverage",
    )
    return value


def _model_canary(
    rows: Sequence[Mapping[str, Any]],
    models: Mapping[str, Sequence[Any]],
    settings: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    canary_rows = list(rows[: min(8, len(rows))])
    floor = float(protocol["model"]["probability_floor"])
    return {
        "ordered_row_ids": [str(row["row_id"]) for row in canary_rows],
        "feature_float64_sha256": {
            variant: sha256_bytes(
                np.asarray(matrix_for(canary_rows, variant), dtype="<f8").tobytes(
                    order="C"
                )
            )
            for variant in VARIANTS
        },
        "calibrated_probabilities_rounded_12dp": {
            variant: np.round(
                apply_temperature(
                    aligned_ensemble_probabilities(
                        models[variant],
                        matrix_for(canary_rows, variant),
                        probability_floor=floor,
                    ),
                    float(settings[variant]["calibration"]["temperature"]),
                ),
                12,
            ).tolist()
            for variant in VARIANTS
        },
    }


def fit_command(args: argparse.Namespace) -> int:
    prepared = _prepare_inputs(args)
    analysis_value = _validate_analysis_for_fit(
        read_json(args.analysis, "development analysis"), prepared=prepared
    )
    rows = known_rows(prepared["rows"])
    require(bool(rows), "fit has no known-action rows")
    nested = mapping(
        analysis_value.get("nested_development_evaluation"),
        "nested development evaluation",
    )
    settings = mapping(
        nested.get("full_development_selection"), "full development selection"
    )
    require(set(settings) == set(VARIANTS), "full development settings are incomplete")
    models, training_diagnostics = _fit_all_variants(
        rows, protocol=prepared["protocol"]
    )
    canary = _model_canary(
        rows, models, settings, protocol=prepared["protocol"]
    )
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "id": "swe-task-state-interpreter-v3-model-bundle",
        "classes_in_order": list(CLASSES),
        "source_actions_in_order": list(SOURCE_ACTION_CLASSES),
        "variants_in_order": list(VARIANTS),
        "variant_widths": dict(VARIANT_WIDTHS),
        "protocol_sha256": prepared["hashes"]["protocol"],
        "action_protocol_sha256": prepared["hashes"]["action_protocol"],
        "analyzer_sha256": prepared["hashes"]["analyzer"],
        "development_analysis_sha256": sha256_file(args.analysis),
        "development_inputs": prepared["hashes"],
        "development_data_binding": prepared["development_binding"],
        "replay_merge_binding": prepared["replay_merge_binding"],
        "settings": dict(settings),
        "training_diagnostics": training_diagnostics,
        "model_canary": canary,
        "models": models,
    }
    atomic_joblib_dump_no_clobber(args.bundle, bundle)
    bundle_sha256 = sha256_file(args.bundle)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "id": "swe-task-state-interpreter-v3-model-manifest",
        "bundle": {
            "filename": args.bundle.name,
            "sha256": bundle_sha256,
            "joblib_top_level_type": "dictionary",
        },
        "classes_in_order": list(CLASSES),
        "variants_in_order": list(VARIANTS),
        "variant_widths": dict(VARIANT_WIDTHS),
        "model_seeds_in_order": list(prepared["protocol"]["model"]["seeds"]),
        "probability_floor": prepared["protocol"]["model"]["probability_floor"],
        "protocol_sha256": prepared["hashes"]["protocol"],
        "action_protocol_sha256": prepared["hashes"]["action_protocol"],
        "analyzer_sha256": prepared["hashes"]["analyzer"],
        "development_analysis_sha256": sha256_file(args.analysis),
        "development_gate_evidence_sha256": canonical_json_sha256(
            analysis_value["development_gates"]
        ),
        "development_gates_passed": True,
        "training_diagnostics": training_diagnostics,
        "model_canary": canary,
        "runtime_versions": _runtime_versions(),
        "security": {
            "joblib_is_executable_pickle": True,
            "bundle_sha256_must_be_verified_before_load": True,
        },
    }
    atomic_write_json_no_clobber(args.manifest, manifest)
    output = {
        "schema_version": SCHEMA_VERSION,
        "id": "swe-task-state-interpreter-v3-development-fit",
        "scope": "fresh_development_fit_after_all_predeclared_gates_passed",
        "inputs": {
            **prepared["hashes"],
            "analysis": sha256_file(args.analysis),
        },
        "development_data_binding": prepared["development_binding"],
        "replay_merge_binding": prepared["replay_merge_binding"],
        "bundle": {"path": str(args.bundle), "sha256": bundle_sha256},
        "manifest": {
            "path": str(args.manifest),
            "sha256": sha256_file(args.manifest),
        },
        "known_action_training_rows": len(rows),
        "unknown_action_rows_excluded_from_fit_only": len(prepared["rows"]) - len(rows),
        "unknown_actions_will_still_receive_inference_predictions": True,
        "settings": dict(settings),
        "training_diagnostics": training_diagnostics,
        "model_canary": canary,
        "operational_reliability_claim": True,
        "reserved_validation_allowed_by_development_gate": True,
    }
    atomic_write_json_no_clobber(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bundle": str(args.bundle),
                "bundle_sha256": bundle_sha256,
                "manifest": str(args.manifest),
                "known_action_training_rows": len(rows),
                "development_gates_passed": True,
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
        subparser.add_argument(
            "--replay-merge-receipt",
            type=Path,
            default=DEFAULT_REPLAY_MERGE_RECEIPT,
            help="authenticated lossless bounded-memory replay merge manifest",
        )
        subparser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        subparser.add_argument(
            "--action-protocol", type=Path, default=DEFAULT_ACTION_PROTOCOL
        )
        subparser.add_argument(
            "--development-cohort",
            type=Path,
            default=DEFAULT_DEVELOPMENT_COHORT,
        )
        subparser.add_argument("--prompts-summary", type=Path, required=True)
        subparser.add_argument("--output", type=Path, required=True)

    analyze_parser = subparsers.add_parser(
        "analyze", help="run nested fresh-development evaluation and gates"
    )
    common(analyze_parser)
    analyze_parser.add_argument(
        "--diagnostic-without-model-refit-bootstrap",
        action="store_true",
        help=(
            "skip the expensive refit bootstrap and fail bound gates closed; "
            "never use this output as reliability evidence"
        ),
    )
    analyze_parser.add_argument(
        "--bootstrap-checkpoint",
        type=Path,
        help="write deterministic full-refit progress after every completed draw",
    )
    analyze_parser.add_argument(
        "--resume-bootstrap",
        action="store_true",
        help="resume only an exact protocol/row/sample/seed-matched checkpoint",
    )
    analyze_parser.set_defaults(handler=analyze_command)

    fit_parser = subparsers.add_parser(
        "fit", help="fit a bundle only after a hash-bound passing analysis"
    )
    common(fit_parser)
    fit_parser.add_argument("--analysis", type=Path, required=True)
    fit_parser.add_argument("--bundle", type=Path, required=True)
    fit_parser.add_argument("--manifest", type=Path, required=True)
    fit_parser.set_defaults(handler=fit_command)
    return parser.parse_args(argv)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_no_symlink_components(path: Path, label: str) -> None:
    absolute = path.expanduser().absolute()
    cursor = absolute
    while True:
        if cursor.exists() or cursor.is_symlink():
            require(not cursor.is_symlink(), f"{label} traverses a symlink: {cursor}")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent


def validate_cli_path_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Fail closed on aliases, clobbers, symlinks, and non-V3 output roots."""

    input_paths = [
        args.prompts,
        args.public_report,
        args.replay_merge_receipt,
        args.protocol,
        args.action_protocol,
        args.development_cohort,
        args.prompts_summary,
    ]
    if args.command == "fit":
        input_paths.append(args.analysis)
    for path in input_paths:
        require(
            path.is_file() and not path.is_symlink(),
            f"missing or unsafe canonical input: {path}",
        )
        _require_no_symlink_components(path, "input path")
    resolved_inputs = [path.resolve(strict=True) for path in input_paths]
    require(
        len(set(resolved_inputs)) == len(resolved_inputs),
        "canonical input paths must be distinct",
    )
    input_inodes = [(path.stat().st_dev, path.stat().st_ino) for path in input_paths]
    require(
        len(set(input_inodes)) == len(input_inodes),
        "input paths must not be hard-link aliases",
    )

    output_paths = [args.output]
    if args.command == "fit":
        output_paths.extend([args.bundle, args.manifest])
    checkpoint = getattr(args, "bootstrap_checkpoint", None)
    if checkpoint is not None:
        output_paths.append(checkpoint)
    root = V3_INTERPRETER_OUTPUT_ROOT.resolve(strict=False)
    resolved_outputs: list[Path] = []
    for path in output_paths:
        _require_no_symlink_components(path.parent, "output parent")
        resolved = path.expanduser().resolve(strict=False)
        require(
            _is_relative_to(resolved, root),
            "all analyzer outputs and checkpoints must be under the dedicated V3 interpreter root",
        )
        require(
            "reserved" not in {part.lower() for part in resolved.parts}
            and not _is_relative_to(resolved, (ROOT / "validation").resolve()),
            "reserved and validation paths are forbidden as analyzer outputs",
        )
        resolved_outputs.append(resolved)
    require(
        len(set(resolved_outputs)) == len(resolved_outputs)
        and set(resolved_outputs).isdisjoint(resolved_inputs),
        "canonical input, output, bundle, manifest, and checkpoint paths must be distinct",
    )
    for path in [args.output, *(
        [args.bundle, args.manifest] if args.command == "fit" else []
    )]:
        require(
            not path.exists() and not path.is_symlink(),
            f"new analyzer output already exists: {path}",
        )
    if checkpoint is not None:
        if args.resume_bootstrap:
            require(
                checkpoint.is_file() and not checkpoint.is_symlink(),
                "resume checkpoint must be an existing regular non-symlink file",
            )
            checkpoint_inode = (
                checkpoint.stat().st_dev,
                checkpoint.stat().st_ino,
            )
            require(
                checkpoint_inode not in set(input_inodes),
                "resume checkpoint must not be a hard-link alias of an input",
            )
        else:
            require(
                not checkpoint.exists() and not checkpoint.is_symlink(),
                "new bootstrap checkpoint already exists",
            )
    return {
        "dedicated_output_root": str(root),
        "input_count": len(input_paths),
        "output_or_checkpoint_count": len(output_paths),
        "canonical_paths_distinct": True,
        "symlinks_and_hardlink_aliases_rejected": True,
        "new_outputs_no_clobber": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_cli_path_contract(args)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
