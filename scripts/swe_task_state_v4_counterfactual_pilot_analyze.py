#!/usr/bin/env python3
"""Join a blinded V4 pilot only after capture/generation and emit descriptive markers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import swe_task_state_v4_counterfactual_generate as generation  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
SCHEMA_VERSION = 1
KIND = "swe_task_state_v4_counterfactual_pilot_descriptive_receipt"
FORBIDDEN_PATH_FRAGMENTS = ("reserved", "validation")


class AnalysisError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot load JSON {path}: {error}") from error


def lexical_path_preflight(paths: Iterable[Path]) -> None:
    for path in paths:
        normalized = os.path.normpath(os.path.abspath(os.fspath(path))).lower()
        if any(
            fragment in component
            for fragment in FORBIDDEN_PATH_FRAGMENTS
            for component in Path(normalized).parts
        ):
            raise AnalysisError(f"forbidden path rejected before filesystem access: {path}")


def canonical_path_preflight(
    *, input_paths: Iterable[Path], output_paths: Iterable[Path]
) -> None:
    for path, strict in [
        *((path, True) for path in input_paths),
        *((path, False) for path in output_paths),
    ]:
        resolved = path.resolve(strict=strict)
        if any(
            fragment in component.lower()
            for fragment in FORBIDDEN_PATH_FRAGMENTS
            for component in resolved.parts
        ):
            raise AnalysisError(f"forbidden canonical path rejected: {path}")


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        display = str(resolved.relative_to(ROOT))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": sha256_file(resolved), "size_bytes": resolved.stat().st_size}


def _bound_load(path: Path, expected_sha256: str, label: str) -> Any:
    _require(_is_sha256(expected_sha256), f"{label} expected hash is invalid")
    _require(sha256_file(path) == expected_sha256, f"{label} bytes changed")
    return load_json(path)


MARKER_PATTERNS: dict[str, re.Pattern[str]] = {
    "explicit_certainty_language": re.compile(
        r"\b(?:certain(?:ly)?|clearly|definitely|confident|conclusive|confirmed|established|obvious(?:ly)?|safe to conclude)\b",
        re.I,
    ),
    "explicit_uncertainty_language": re.compile(
        r"\b(?:uncertain|unclear|unsure|ambiguous|mixed|might|may|perhaps|likely|possib(?:le|ility)|not enough evidence|cannot conclude)\b",
        re.I,
    ),
    "recheck_language": re.compile(
        r"\b(?:recheck|re-check|double-check|inspect|investigate|rerun|re-run|retest|re-test|examine|diagnose|debug|check again)\b",
        re.I,
    ),
    "explicit_doubt_language": re.compile(
        r"\b(?:doubt(?:ful)?|hesitant|hesitate|question whether|not convinced)\b",
        re.I,
    ),
    "seek_information_language": re.compile(
        r"\b(?:need to inspect|need more information|gather more|look at|open the|search for|find out|determine whether|check the)\b",
        re.I,
    ),
    "multiple_alternatives_language": re.compile(
        r"\b(?:alternative|alternatives|either\b.{0,80}\bor|two possibilities|whether\b.{0,80}\bor|could\b.{0,80}\bor)\b",
        re.I | re.S,
    ),
    "advance_or_validate_language": re.compile(
        r"\b(?:proceed|implement|edit|patch|apply the fix|run (?:the )?tests?|validate|commit|continue with)\b",
        re.I,
    ),
    "rework_or_targeted_inspection_language": re.compile(
        r"\b(?:rework|revise|fix|inspect|investigate|recheck|re-check|diagnose|debug|examine|trace)\b",
        re.I,
    ),
    "explicit_pressure_language": re.compile(
        r"\b(?:time|token limit|deadline|hurry|quickly|immediately|urgent|urgency|pressure|little time)\b",
        re.I,
    ),
    "explicit_affect_or_stress_language": re.compile(
        r"\b(?:stress(?:ed|ful)?|anxious|anxiety|frustrat(?:ed|ing|ion)|worried|worry|concern(?:ed)?|fear(?:ful)?|relief|relieved|nervous|emotion|feel(?:ing)?|overwhelmed|panic)\b",
        re.I,
    ),
    "observation_language": re.compile(
        r"\b(?:because|given|evidence|diagnostic|result|shows?|indicates?|observed|found)\b",
        re.I,
    ),
    "conclusion_language": re.compile(
        r"\b(?:therefore|thus|so|means|conclude|suggests?|supports?|contradicts?)\b",
        re.I,
    ),
    "action_intent_language": re.compile(
        r"\b(?:I will|I'll|next|should|need to|plan to|the next step)\b",
        re.I,
    ),
}


def extract_blinded_markers(text: str, *, generated_token_count: int) -> dict[str, Any]:
    _require(isinstance(text, str), "completion text must be a string")
    markers = {name: bool(pattern.search(text)) for name, pattern in MARKER_PATTERNS.items()}
    markers["doubt_like_behavior_proxy"] = bool(
        markers["recheck_language"]
        or markers["explicit_doubt_language"]
        or markers["seek_information_language"]
        or markers["multiple_alternatives_language"]
    )
    positions: list[int] = []
    for name in ("observation_language", "conclusion_language", "action_intent_language"):
        match = MARKER_PATTERNS[name].search(text)
        positions.append(match.start() if match else -1)
    markers["visible_chain_language_proxy"] = bool(
        all(position >= 0 for position in positions)
        and positions[0] <= positions[1] <= positions[2]
    )
    markers["generated_token_count"] = int(generated_token_count)
    markers["visible_character_count"] = len(text)
    markers["nonempty_line_count"] = sum(bool(line.strip()) for line in text.splitlines())
    markers["sentence_terminal_count"] = len(re.findall(r"[.!?](?:\s|$)", text))
    return markers


def validate_completions(
    value: Any,
    *,
    capture_prompts_sha256: str,
    materialization_sha256: str,
    capture_manifest_sha256: str,
) -> list[dict[str, Any]]:
    _require(
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and value.get("kind") == generation.KIND
        and value.get("status") == "passed_blinded_generation_after_authenticated_capture"
        and value.get("scope")
        == "externally_observable_completion_text_only_no_labels_or_condition_key",
        "blinded completion identity changed",
    )
    inputs = value.get("inputs", {})
    _require(
        inputs.get("capture_prompts", {}).get("sha256") == capture_prompts_sha256
        and inputs.get("materialization_manifest", {}).get("sha256")
        == materialization_sha256
        and inputs.get("authenticated_capture_manifest", {}).get("sha256")
        == capture_manifest_sha256,
        "completion generation is not bound to the pilot inputs",
    )
    execution = value.get("execution_order", {})
    _require(
        execution.get("authenticated_capture_verified_before_model_load") is True
        and execution.get("condition_key_absent_from_process_inputs") is True
        and execution.get("completion_label_extraction_completed") is False,
        "completion generation did not preserve separation",
    )
    records = value.get("records")
    _require(
        value.get("completion_count") == 12
        and isinstance(records, list)
        and len(records) == 12,
        "completion count changed",
    )
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        _require(
            isinstance(row, dict)
            and row.get("index") == index
            and _is_sha256(row.get("prompt_id"))
            and _is_sha256(row.get("capture_source_id_sha256"))
            and isinstance(row.get("generated_text"), str)
            and sha256_bytes(row["generated_text"].encode("utf-8"))
            == row.get("generated_text_sha256")
            and isinstance(row.get("generated_token_ids"), list)
            and row.get("generated_token_count") == len(row["generated_token_ids"])
            and sha256_bytes(canonical_json_bytes(row["generated_token_ids"]))
            == row.get("generated_token_ids_sha256"),
            f"completion record {index} changed",
        )
        normalized.append(dict(row))
    _require(len({row["prompt_id"] for row in normalized}) == 12, "completion IDs duplicate")
    return normalized


def validate_condition_key(value: Any, prompt_ids: set[str]) -> dict[str, dict[str, Any]]:
    _require(
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and value.get("kind") == "swe_task_state_v4_counterfactual_pilot_condition_key"
        and value.get("status") == "assignment_only_completion_labels_not_extracted"
        and value.get("capture_and_activation_feature_process_must_not_read_this_file")
        is True
        and value.get("completion_label_extraction_completed") is False
        and value.get("subjective_state_labels_present") is False,
        "condition-key separation changed",
    )
    records = value.get("records")
    _require(isinstance(records, list) and len(records) == 12, "condition key count changed")
    by_id: dict[str, dict[str, Any]] = {}
    for row in records:
        _require(
            isinstance(row, dict)
            and _is_sha256(row.get("prompt_id"))
            and row["prompt_id"] not in by_id
            and row.get("evidence_level") in generation.EVIDENCE_LEVELS
            and row.get("pressure_level") in generation.PRESSURE_LEVELS
            and row.get("paraphrase_replica") in (0, 1),
            "condition-key record changed",
        )
        by_id[row["prompt_id"]] = dict(row)
    _require(set(by_id) == prompt_ids, "condition key does not exactly join completions")
    return by_id


def objective_behavior_proxy(markers: Mapping[str, Any], evidence: str) -> dict[str, bool]:
    if evidence == "clear_success":
        appropriate = bool(
            markers["advance_or_validate_language"]
            and not markers["rework_or_targeted_inspection_language"]
            and not markers["doubt_like_behavior_proxy"]
        )
    elif evidence == "clear_failure":
        appropriate = bool(markers["rework_or_targeted_inspection_language"])
    elif evidence == "contradictory_ambiguous":
        appropriate = bool(markers["doubt_like_behavior_proxy"])
    else:
        raise AnalysisError(f"unsupported evidence level: {evidence}")
    if evidence == "contradictory_ambiguous":
        calibration = bool(appropriate and markers["explicit_uncertainty_language"])
    else:
        calibration = bool(appropriate and not markers["explicit_uncertainty_language"])
    return {
        "objective_action_appropriateness_rule_proxy": appropriate,
        "objective_certainty_action_calibration_rule_proxy": calibration,
    }


def _load_activation_states(
    capture_manifest: Mapping[str, Any], capture_manifest_path: Path
) -> dict[str, dict[str, np.ndarray]]:
    try:
        from safetensors.numpy import load_file
    except ImportError as error:
        raise AnalysisError("safetensors is required for activation receipt") from error
    states: dict[str, dict[str, np.ndarray]] = {}
    root = capture_manifest_path.parent
    for boundary in capture_manifest["boundaries"]:
        source_id = boundary["source_id_sha256"]
        shard_record = boundary["shard"]
        shard_path = root / shard_record["path"]
        lexical_path_preflight((shard_path,))
        canonical_path_preflight(input_paths=(shard_path,), output_paths=())
        _require(
            shard_path.stat().st_size == shard_record["size_bytes"]
            and sha256_file(shard_path) == shard_record["sha256"],
            "activation shard byte binding changed",
        )
        loaded = load_file(shard_path)
        _require(
            set(loaded) == {"raw_residual", "public_j_state"}
            and all(value.shape == (24, 5120) for value in loaded.values())
            and all(value.dtype == np.float32 for value in loaded.values())
            and all(np.isfinite(value).all() for value in loaded.values()),
            "activation shard tensor contract changed",
        )
        states[source_id] = {name: value.copy() for name, value in loaded.items()}
    _require(len(states) == 12, "activation state count changed")
    return states


def _distance(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    difference = left.astype(np.float64) - right.astype(np.float64)
    per_layer_rms = np.sqrt(np.mean(difference * difference, axis=1))
    left_norm = np.linalg.norm(left.astype(np.float64), axis=1)
    right_norm = np.linalg.norm(right.astype(np.float64), axis=1)
    dot = np.sum(left.astype(np.float64) * right.astype(np.float64), axis=1)
    cosine_distance = 1.0 - dot / np.maximum(left_norm * right_norm, 1e-30)
    return {
        "mean_layer_rms_difference": float(np.mean(per_layer_rms)),
        "max_layer_rms_difference": float(np.max(per_layer_rms)),
        "mean_layer_cosine_distance": float(np.mean(cosine_distance)),
        "max_layer_cosine_distance": float(np.max(cosine_distance)),
    }


def descriptive_summary(joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    boolean_fields = [
        *MARKER_PATTERNS,
        "doubt_like_behavior_proxy",
        "visible_chain_language_proxy",
        "objective_action_appropriateness_rule_proxy",
        "objective_certainty_action_calibration_rule_proxy",
    ]
    by_evidence: dict[str, Any] = {}
    for evidence in generation.EVIDENCE_LEVELS:
        rows = [row for row in joined if row["evidence_level"] == evidence]
        by_evidence[evidence] = {
            "n": len(rows),
            "marker_positive_counts": {
                field: sum(bool(row["markers"][field]) for row in rows)
                for field in boolean_fields
            },
            "mean_generated_token_count": float(
                np.mean([row["markers"]["generated_token_count"] for row in rows])
            ),
        }
    pressure_pairs: list[dict[str, Any]] = []
    for evidence in generation.EVIDENCE_LEVELS:
        for replica in (0, 1):
            cells = {
                row["pressure_level"]: row
                for row in joined
                if row["evidence_level"] == evidence
                and row["paraphrase_replica"] == replica
            }
            _require(set(cells) == set(generation.PRESSURE_LEVELS), "pressure pair incomplete")
            neutral = cells["neutral"]
            pressure = cells["fixed_time_or_token_pressure"]
            pressure_pairs.append(
                {
                    "evidence_level": evidence,
                    "paraphrase_replica": replica,
                    "pressure_minus_neutral_generated_tokens": pressure["markers"][
                        "generated_token_count"
                    ]
                    - neutral["markers"]["generated_token_count"],
                    "marker_changes_pressure_minus_neutral": {
                        field: int(bool(pressure["markers"][field]))
                        - int(bool(neutral["markers"][field]))
                        for field in boolean_fields
                    },
                    "activation_pressure_minus_neutral_distances": pressure[
                        "activation_distance_from_neutral"
                    ],
                }
            )
    paraphrase_pairs: list[dict[str, Any]] = []
    for evidence in generation.EVIDENCE_LEVELS:
        for pressure in generation.PRESSURE_LEVELS:
            cells = {
                row["paraphrase_replica"]: row
                for row in joined
                if row["evidence_level"] == evidence and row["pressure_level"] == pressure
            }
            _require(set(cells) == {0, 1}, "paraphrase pair incomplete")
            paraphrase_pairs.append(
                {
                    "evidence_level": evidence,
                    "pressure_level": pressure,
                    "boolean_marker_agreement": {
                        field: bool(cells[0]["markers"][field])
                        == bool(cells[1]["markers"][field])
                        for field in boolean_fields
                    },
                }
            )
    return {
        "by_evidence": by_evidence,
        "pressure_pairs": pressure_pairs,
        "paraphrase_pairs": paraphrase_pairs,
        "descriptive_only_one_boundary_no_uncertainty_interval": True,
    }


def _adapter_split(
    capture_manifest: Mapping[str, Any], capture_manifest_path: Path
) -> dict[str, Any]:
    reference_record = capture_manifest["reference_report"]
    base_record = capture_manifest["base_report"]
    reference_path = ROOT / reference_record["path"]
    base_path = ROOT / base_record["path"]
    lexical_path_preflight((reference_path, base_path))
    canonical_path_preflight(input_paths=(reference_path, base_path), output_paths=())
    _require(
        sha256_file(reference_path) == reference_record["sha256"]
        and sha256_file(base_path) == base_record["sha256"],
        "base/reference report byte binding changed",
    )
    reference = load_json(reference_path)
    fresh = load_json(base_path)
    _require(
        reference.get("status") == fresh.get("status") == "failed"
        and len(reference.get("experiments", [])) == len(fresh.get("experiments", [])) == 12,
        "expected failed adapter canary pattern changed",
    )
    failed_rows: list[int] = []
    row_patterns: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(
        zip(reference["experiments"], fresh["experiments"], strict=True)
    ):
        left_pattern = {
            "top1_match": left["final_layer_top1_matches_greedy"],
            "norm_within_tolerance": left["final_norm_reconstruction"]["within_tolerance"],
            "logits_within_tolerance": left["final_logits_reconstruction"]["within_tolerance"],
            "norm_max_abs_error": left["final_norm_reconstruction"]["max_abs_error"],
            "logits_max_abs_error": left["final_logits_reconstruction"]["max_abs_error"],
        }
        right_pattern = {
            "top1_match": right["final_layer_top1_matches_greedy"],
            "norm_within_tolerance": right["final_norm_reconstruction"]["within_tolerance"],
            "logits_within_tolerance": right["final_logits_reconstruction"]["within_tolerance"],
            "norm_max_abs_error": right["final_norm_reconstruction"]["max_abs_error"],
            "logits_max_abs_error": right["final_logits_reconstruction"]["max_abs_error"],
        }
        _require(left_pattern == right_pattern, f"adapter pattern differs at row {index}")
        if not left_pattern["logits_within_tolerance"]:
            failed_rows.append(index)
        row_patterns.append({"index": index, **left_pattern})
    _require(failed_rows == [1, 3, 10], "adapter failure rows changed")
    _require(
        all(row["top1_match"] and row["norm_within_tolerance"] for row in row_patterns)
        and all(
            row["logits_max_abs_error"] == (0.125 if row["index"] in failed_rows else 0.0625)
            for row in row_patterns
        ),
        "adapter exact error pattern changed",
    )
    return {
        "raw_public_j_pre_vocabulary_capture_status": "passed_12_of_12",
        "reference_residual_manifest_equality": "passed_12_of_12",
        "safetensors_reload_verification": "passed_12_of_12",
        "vocabulary_adapter_status": "failed_3_of_12",
        "vocabulary_adapter_failed_rows": failed_rows,
        "reference_and_fresh_failure_pattern_exactly_equal": True,
        "top1_greedy_match": "passed_12_of_12",
        "final_norm_tolerance": "passed_12_of_12",
        "final_logits_tolerance": "failed_rows_1_3_10_max_abs_error_0.125",
        "vocabulary_level_claims_permitted": False,
        "raw_public_j_tensor_claims_limited_to_capture_contract": True,
        "reference_elapsed_seconds": reference["elapsed_seconds"],
        "capture_elapsed_seconds": fresh["elapsed_seconds"],
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-prompts", type=Path, required=True)
    parser.add_argument("--capture-prompts-sha256", required=True)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--materialization-manifest-sha256", required=True)
    parser.add_argument("--condition-key", type=Path, required=True)
    parser.add_argument("--condition-key-sha256", required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--capture-manifest-sha256", required=True)
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--completions-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    input_paths = (
        args.capture_prompts,
        args.materialization_manifest,
        args.condition_key,
        args.capture_manifest,
        args.completions,
        SCRIPT_PATH,
    )
    lexical_path_preflight((*input_paths, args.output))
    canonical_path_preflight(input_paths=input_paths, output_paths=(args.output,))
    capture_prompts = _bound_load(
        args.capture_prompts, args.capture_prompts_sha256, "capture prompts"
    )
    materialization = _bound_load(
        args.materialization_manifest,
        args.materialization_manifest_sha256,
        "materialization manifest",
    )
    capture_manifest = _bound_load(
        args.capture_manifest, args.capture_manifest_sha256, "capture manifest"
    )
    completions_value = _bound_load(
        args.completions, args.completions_sha256, "blinded completions"
    )
    prompt_rows = generation.validate_capture_bundle(capture_prompts)
    generation.validate_materialization(
        materialization,
        capture_path=args.capture_prompts,
        capture_sha256=args.capture_prompts_sha256,
    )
    generation.validate_capture_manifest(
        capture_manifest,
        capture_path=args.capture_prompts,
        capture_sha256=args.capture_prompts_sha256,
        prompt_rows=prompt_rows,
    )
    completions = validate_completions(
        completions_value,
        capture_prompts_sha256=args.capture_prompts_sha256,
        materialization_sha256=args.materialization_manifest_sha256,
        capture_manifest_sha256=args.capture_manifest_sha256,
    )

    # This entire pass is condition-blind.  The key is not opened until all markers exist.
    blinded_markers = {
        row["prompt_id"]: extract_blinded_markers(
            row["generated_text"], generated_token_count=row["generated_token_count"]
        )
        for row in completions
    }
    condition_value = _bound_load(args.condition_key, args.condition_key_sha256, "condition key")
    condition_by_id = validate_condition_key(condition_value, set(blinded_markers))
    states = _load_activation_states(capture_manifest, args.capture_manifest)
    completion_by_id = {row["prompt_id"]: row for row in completions}
    joined: list[dict[str, Any]] = []
    by_cell: dict[tuple[str, int, str], dict[str, Any]] = {}
    for prompt in prompt_rows:
        prompt_id = prompt["id"]
        condition = condition_by_id[prompt_id]
        markers = dict(blinded_markers[prompt_id])
        markers.update(objective_behavior_proxy(markers, condition["evidence_level"]))
        activation = states[condition["capture_source_id_sha256"]]
        activation_norms = {
            tensor_name: {
                "mean_layer_rms": float(np.mean(np.sqrt(np.mean(tensor.astype(np.float64) ** 2, axis=1)))),
                "max_layer_rms": float(np.max(np.sqrt(np.mean(tensor.astype(np.float64) ** 2, axis=1)))),
            }
            for tensor_name, tensor in activation.items()
        }
        record = {
            "prompt_id": prompt_id,
            "condition_id_sha256": condition["condition_id_sha256"],
            "condition_order_within_boundary": condition["condition_order_within_boundary"],
            "evidence_level": condition["evidence_level"],
            "pressure_level": condition["pressure_level"],
            "paraphrase_replica": condition["paraphrase_replica"],
            "generated_text": completion_by_id[prompt_id]["generated_text"],
            "generated_text_sha256": completion_by_id[prompt_id]["generated_text_sha256"],
            "markers": markers,
            "activation_norms": activation_norms,
            "activation_distance_from_neutral": None,
        }
        joined.append(record)
        by_cell[(record["evidence_level"], record["paraphrase_replica"], record["pressure_level"])] = record
    for evidence in generation.EVIDENCE_LEVELS:
        for replica in (0, 1):
            neutral_record = by_cell[(evidence, replica, "neutral")]
            pressure_record = by_cell[(evidence, replica, "fixed_time_or_token_pressure")]
            neutral_state = states[
                condition_by_id[neutral_record["prompt_id"]]["capture_source_id_sha256"]
            ]
            pressure_state = states[
                condition_by_id[pressure_record["prompt_id"]]["capture_source_id_sha256"]
            ]
            pressure_record["activation_distance_from_neutral"] = {
                name: _distance(pressure_state[name], neutral_state[name])
                for name in ("raw_residual", "public_j_state")
            }
            neutral_record["activation_distance_from_neutral"] = {
                name: {
                    "mean_layer_rms_difference": 0.0,
                    "max_layer_rms_difference": 0.0,
                    "mean_layer_cosine_distance": 0.0,
                    "max_layer_cosine_distance": 0.0,
                }
                for name in ("raw_residual", "public_j_state")
            }
    joined.sort(key=lambda row: int(row["condition_order_within_boundary"]))
    adapter = _adapter_split(capture_manifest, args.capture_manifest)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "descriptive_feasibility_receipt_one_boundary_no_inference",
        "scope": "observable_completion_markers_and_visible_condition_activation_distances_only",
        "inputs": {
            "capture_prompts": _file_record(args.capture_prompts),
            "materialization_manifest": _file_record(args.materialization_manifest),
            "condition_key": _file_record(args.condition_key),
            "authenticated_capture_manifest": _file_record(args.capture_manifest),
            "blinded_completions": _file_record(args.completions),
        },
        "implementation": _file_record(SCRIPT_PATH),
        "separation_audit": {
            "capture_and_generation_completed_before_condition_key_read": True,
            "completion_markers_extracted_before_condition_key_read": True,
            "condition_key_join_exact_12_of_12": True,
            "capture_prompt_allowed_difference_gates": materialization["matching_gates"],
            "condition_key_never_used_as_activation_feature": True,
            "subjective_state_labels_absent": True,
        },
        "capture_contract_split": adapter,
        "marker_definitions": {
            name: pattern.pattern for name, pattern in MARKER_PATTERNS.items()
        },
        "marker_limitations": {
            "regex_markers_are_not_semantic_annotations": True,
            "visible_chain_language_proxy_is_not_private_cot": True,
            "objective_action_and_calibration_are_rule_proxies_not_subjective_confidence": True,
            "explicit_affect_language_is_not_experienced_emotion": True,
            "pressure_cue_response_is_not_experienced_stress": True,
            "activation_distances_include_direct_visible_manipulation_encoding_and_are_not_a_hidden_state_readout": True,
        },
        "condition_count": len(joined),
        "condition_outputs": joined,
        "descriptive_summary": descriptive_summary(joined),
        "runtime_and_cost": {
            "reference_seconds": adapter["reference_elapsed_seconds"],
            "raw_public_j_capture_seconds": adapter["capture_elapsed_seconds"],
            "blinded_completion_generation_seconds": completions_value["runtime"]["elapsed_seconds"],
            "total_measured_three_pass_seconds": adapter["reference_elapsed_seconds"]
            + adapter["capture_elapsed_seconds"]
            + completions_value["runtime"]["elapsed_seconds"],
            "pilot_prefix_is_unique_shortest_not_representative": True,
            "full_1440_campaign_cost_extrapolation_permitted": False,
        },
        "claim_scope": {
            "prompt_matching_feasibility_established_for_one_block": True,
            "raw_public_j_capture_feasibility_established_for_one_block": True,
            "observable_completion_generation_feasibility_established_for_one_block": True,
            "factorial_effect_established": False,
            "outer_generalization_established": False,
            "vocabulary_adapter_valid_for_all_conditions": False,
            "private_chain_of_thought_reconstructed": False,
            "subjective_emotion_confidence_doubt_or_stress_inferred": False,
        },
        "reserved_validation_access_authorized": False,
    }
    _atomic_write_json(args.output, result)
    print(f"wrote descriptive one-boundary receipt to {args.output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
