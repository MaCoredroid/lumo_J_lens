#!/usr/bin/env python3
"""Verify the one-block V4 pilot capture and record its split pass/fail status."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import swe_task_state_v4_counterfactual_generate as generation  # noqa: E402
import swe_task_state_v4_counterfactual_pilot_analyze as analysis  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
KIND = "swe_task_state_v4_counterfactual_pilot_capture_receipt"
lexical_path_preflight = analysis.lexical_path_preflight


def validate_capture_split(value: Any) -> dict[str, Any]:
    analysis._require(isinstance(value, dict), "capture split must be an object")
    required = {
        "raw_public_j_pre_vocabulary_capture_status": "passed_12_of_12",
        "reference_residual_manifest_equality": "passed_12_of_12",
        "safetensors_reload_verification": "passed_12_of_12",
        "vocabulary_adapter_status": "failed_3_of_12",
        "vocabulary_adapter_failed_rows": [1, 3, 10],
        "reference_and_fresh_failure_pattern_exactly_equal": True,
        "top1_greedy_match": "passed_12_of_12",
        "final_norm_tolerance": "passed_12_of_12",
        "final_logits_tolerance": "failed_rows_1_3_10_max_abs_error_0.125",
        "vocabulary_level_claims_permitted": False,
        "raw_public_j_tensor_claims_limited_to_capture_contract": True,
    }
    for key, expected in required.items():
        analysis._require(value.get(key) == expected, f"capture split changed: {key}")
    for key in ("reference_elapsed_seconds", "capture_elapsed_seconds"):
        observed = value.get(key)
        analysis._require(
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and observed > 0
            and math.isfinite(float(observed)),
            f"capture split runtime changed: {key}",
        )
    return dict(value)


def _atomic_write_json(path: Path, value: Any) -> None:
    analysis._require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
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
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--capture-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    inputs = (
        args.capture_prompts,
        args.materialization_manifest,
        args.capture_manifest,
        SCRIPT_PATH,
    )
    lexical_path_preflight((*inputs, args.output))
    analysis.canonical_path_preflight(input_paths=inputs, output_paths=(args.output,))
    prompts_value = analysis._bound_load(
        args.capture_prompts, args.capture_prompts_sha256, "capture prompts"
    )
    materialization = analysis._bound_load(
        args.materialization_manifest,
        args.materialization_manifest_sha256,
        "materialization manifest",
    )
    capture_manifest = analysis._bound_load(
        args.capture_manifest, args.capture_manifest_sha256, "capture manifest"
    )
    prompt_rows = generation.validate_capture_bundle(prompts_value)
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
    adapter_split = validate_capture_split(
        analysis._adapter_split(capture_manifest, args.capture_manifest)
    )
    result = {
        "schema_version": 1,
        "kind": KIND,
        "status": "split_result_raw_capture_passed_vocabulary_adapter_failed",
        "scope": "one_complete_factorial_block_capture_feasibility_only_no_decoder_or_behavior_result",
        "inputs": {
            "capture_prompts": analysis._file_record(args.capture_prompts),
            "materialization_manifest": analysis._file_record(args.materialization_manifest),
            "authenticated_capture_manifest": analysis._file_record(args.capture_manifest),
        },
        "implementation": analysis._file_record(SCRIPT_PATH),
        "prompt_matching": materialization["matching_gates"],
        "capture_contract_split": adapter_split,
        "execution_state": {
            "blinded_reference_completed": True,
            "authenticated_raw_public_j_capture_completed": True,
            "blinded_counterfactual_completion_generation_completed": False,
            "completion_marker_extraction_completed": False,
            "condition_key_join_completed": False,
        },
        "claim_scope": {
            "prompt_matching_feasibility_established_for_one_block": True,
            "raw_public_j_capture_feasibility_established_for_one_block": True,
            "vocabulary_adapter_valid_for_all_conditions": False,
            "counterfactual_behavior_effect_established": False,
            "private_chain_of_thought_reconstructed": False,
            "subjective_emotion_confidence_doubt_or_stress_inferred": False,
        },
        "next_gate": "successful_blinded_completion_generation_then_condition_blind_marker_extraction_then_condition_key_join",
        "reserved_validation_access_authorized": False,
    }
    _atomic_write_json(args.output, result)
    print(f"wrote split capture receipt to {args.output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
