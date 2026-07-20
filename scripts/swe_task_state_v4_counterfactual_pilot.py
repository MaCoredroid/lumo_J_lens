#!/usr/bin/env python3
"""Materialize and verify one blinded V4 counterfactual feasibility block."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_counterfactual_pilot.json"
CONFIG_SHA256 = "7b4cda2f0b22901ce27f5c8d01c2d8d365c61ac4b5822e6814ce7de37d711989"
SCRIPT_PATH = Path(__file__).resolve()
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
MODEL_SNAPSHOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--nvidia--Qwen3.6-27B-NVFP4"
    / "snapshots"
    / MODEL_REVISION
)
CAPTURE_BUNDLE_NAME = "capture-prompts.json"
CONDITION_KEY_NAME = "condition-key.json"
MANIFEST_NAME = "materialization-manifest.json"
KIND = "swe_task_state_v4_counterfactual_pilot_materialization"
EVIDENCE_LEVELS = ("clear_success", "clear_failure", "contradictory_ambiguous")
PRESSURE_LEVELS = ("neutral", "fixed_time_or_token_pressure")
FORBIDDEN_PATH_FRAGMENTS = ("reserved", "validation")


class PilotError(ValueError):
    """Raised when the feasibility-pilot contract is not met."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PilotError(f"cannot load JSON {path}: {error}") from error


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


def lexical_path_preflight(paths: Iterable[Path]) -> None:
    for path in paths:
        normalized = os.path.normpath(os.path.abspath(os.fspath(path))).lower()
        if any(
            fragment in component
            for fragment in FORBIDDEN_PATH_FRAGMENTS
            for component in Path(normalized).parts
        ):
            raise PilotError(f"forbidden path rejected before filesystem access: {path}")


def canonical_path_preflight(
    *, input_paths: Iterable[Path], output_paths: Iterable[Path]
) -> None:
    for path, strict in [
        *((path, True) for path in input_paths),
        *((path, False) for path in output_paths),
    ]:
        try:
            resolved = path.resolve(strict=strict)
        except OSError as error:
            raise PilotError(f"cannot resolve path: {path}: {error}") from error
        if any(
            fragment in component.lower()
            for fragment in FORBIDDEN_PATH_FRAGMENTS
            for component in resolved.parts
        ):
            raise PilotError(f"forbidden canonical path rejected: {path}")


def _validate_bound_file(path: Path, record: Mapping[str, Any], label: str) -> None:
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    resolved = path.resolve(strict=True)
    _require(resolved.is_file(), f"{label} is not a regular file")
    _require(
        resolved.stat().st_size == record.get("size_bytes")
        and sha256_file(resolved) == record.get("sha256"),
        f"{label} byte binding changed",
    )


def validate_config(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "pilot config must be an object")
    config = dict(value)
    _require(
        set(config)
        == {
            "schema_version",
            "id",
            "status",
            "purpose",
            "sources",
            "pilot_selection",
            "rendering",
            "matching_gates",
            "execution_order",
            "observable_targets",
            "claim_scope",
            "reserved_validation_access_authorized",
        },
        "pilot config schema changed",
    )
    _require(
        config["schema_version"] == 1
        and config["id"] == "swe-task-state-v4-counterfactual-runtime-pilot-v1"
        and config["status"]
        == "prospective_development_feasibility_only_reserved_validation_closed",
        "pilot config identity changed",
    )
    selection = config["pilot_selection"]
    _require(
        isinstance(selection, dict)
        and _is_sha256(selection.get("source_id_sha256"))
        and selection.get("expected_original_prompt_token_count") == 12504
        and selection.get("expected_assignment_count") == 12
        and selection.get("labels_outcomes_completion_text_and_activation_values_forbidden")
        is True
        and selection.get("inferential_use_forbidden") is True,
        "pilot selection contract changed",
    )
    rendering = config["rendering"]
    _require(
        isinstance(rendering, dict)
        and rendering.get("generation_suffix") == "<|im_start|>assistant\n<think>\n"
        and rendering.get("tool_name_and_arguments_identical_across_all_conditions")
        is True
        and rendering.get("semantic_padding_tokens_used") is False,
        "pilot rendering contract changed",
    )
    for replica in ("0", "1"):
        _require(
            set(rendering["evidence_text"][replica]) == set(EVIDENCE_LEVELS)
            and set(rendering["pressure_text"][replica]) == set(PRESSURE_LEVELS),
            "pilot factorial rendering cells changed",
        )
    _require(
        all(value is True for value in config["matching_gates"].values())
        and config["reserved_validation_access_authorized"] is False,
        "pilot matching or reserved-data gate changed",
    )
    claims = config["claim_scope"]
    _require(
        claims.get("runtime_and_prompt_matching_feasibility_may_be_established") is True
        and all(
            claims.get(name) is False
            for name in (
                "factorial_effect_established_by_one_boundary",
                "outer_generalization_established",
                "private_chain_of_thought_reconstructed",
                "subjective_confidence_or_doubt_inferred",
                "experienced_stress_inferred",
                "experienced_emotion_inferred",
            )
        ),
        "pilot claim scope changed",
    )
    return config


def validate_assignment(value: Any, config: Mapping[str, Any]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]]
]:
    _require(isinstance(value, dict), "assignment manifest must be an object")
    _require(
        value.get("schema_version") == 1
        and value.get("kind") == "swe_task_state_v4_counterfactual_boundary_assignment"
        and value.get("status")
        == "passed_selection_and_assignment_only_generation_not_started"
        and value.get("reserved_validation_access_authorized") is False,
        "assignment identity changed",
    )
    execution = value.get("execution_state")
    _require(
        isinstance(execution, dict)
        and execution.get("generation_completed") is False
        and execution.get("activation_capture_completed") is False
        and execution.get("completion_label_extraction_completed") is False,
        "assignment is no longer prospective",
    )
    boundaries = value.get("selection", {}).get("boundaries")
    assignments = value.get("assignment", {}).get("assignments")
    _require(
        isinstance(boundaries, list)
        and len(boundaries) == 120
        and isinstance(assignments, list)
        and len(assignments) == 1440,
        "assignment counts changed",
    )
    selected_source = config["pilot_selection"]["source_id_sha256"]
    selected_boundaries = [
        dict(row)
        for row in boundaries
        if isinstance(row, dict) and row.get("source_id_sha256") == selected_source
    ]
    block = [
        dict(row)
        for row in assignments
        if isinstance(row, dict) and row.get("source_id_sha256") == selected_source
    ]
    _require(len(selected_boundaries) == 1 and len(block) == 12, "pilot block changed")
    _require(
        {
            (row.get("evidence_level"), row.get("pressure_level"), row.get("paraphrase_replica"))
            for row in block
        }
        == {
            (evidence, pressure, replica)
            for evidence in EVIDENCE_LEVELS
            for pressure in PRESSURE_LEVELS
            for replica in (0, 1)
        },
        "pilot block is not a complete 3x2x2 factorial",
    )
    _require(
        sorted(row.get("condition_order_within_boundary") for row in block)
        == list(range(12))
        and len({row.get("condition_id_sha256") for row in block}) == 12
        and all(_is_sha256(row.get("condition_id_sha256")) for row in block),
        "pilot condition identities or order changed",
    )
    return selected_boundaries, sorted(
        block, key=lambda row: int(row["condition_order_within_boundary"])
    )


def stream_selected_prompt(
    path: Path, *, boundary_sources: set[str], target_source: str
) -> tuple[dict[str, Any], dict[str, int]]:
    """Read only ID/text/token fields; never materialize source metadata or labels."""

    try:
        import ijson
    except ImportError as error:
        raise PilotError("ijson is required for bounded source streaming") from error

    current_id: str | None = None
    current_source: str | None = None
    current_text: str | None = None
    current_tokens: list[int] | None = None
    current_token_count = 0
    selected_counts: dict[str, int] = {}
    target: dict[str, Any] | None = None
    with path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle, use_float=True):
            if prefix == "item" and event == "start_map":
                current_id = None
                current_source = None
                current_text = None
                current_tokens = None
                current_token_count = 0
            elif prefix == "item.id" and event == "string":
                current_id = value
                current_source = sha256_text(value)
                if current_source == target_source:
                    current_tokens = []
            elif prefix == "item.text" and event == "string":
                if current_source == target_source:
                    current_text = value
            elif prefix == "item.token_ids.item" and event == "number":
                _require(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                    "source prompt contains an invalid token ID",
                )
                current_token_count += 1
                if current_tokens is not None:
                    current_tokens.append(value)
            elif prefix == "item" and event == "end_map":
                if current_source in boundary_sources:
                    _require(current_source not in selected_counts, "duplicate selected source")
                    selected_counts[current_source] = current_token_count
                if current_source == target_source:
                    _require(
                        current_id is not None
                        and isinstance(current_text, str)
                        and isinstance(current_tokens, list)
                        and len(current_tokens) == current_token_count,
                        "selected source is incomplete",
                    )
                    target = {
                        "id": current_id,
                        "text": current_text,
                        "token_ids": current_tokens,
                    }
    _require(len(selected_counts) == 120, "not all frozen boundaries were found")
    _require(target is not None, "pilot source was not found")
    return target, selected_counts


def load_tokenizer(config: Mapping[str, Any]) -> Any:
    from transformers import AutoTokenizer

    model = config["sources"]["model"]
    _require(MODEL_SNAPSHOT.resolve(strict=True).is_dir(), "pinned model snapshot is absent")
    for name, expected in (
        ("tokenizer.json", model["tokenizer_json_sha256"]),
        ("tokenizer_config.json", model["tokenizer_config_sha256"]),
    ):
        path = MODEL_SNAPSHOT / name
        _require(path.resolve(strict=True).is_file(), f"missing pinned {name}")
        _require(sha256_file(path) == expected, f"pinned {name} hash changed")
    return AutoTokenizer.from_pretrained(
        MODEL_SNAPSHOT, local_files_only=True, trust_remote_code=True
    )


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    return sha256_bytes(canonical_json_bytes(list(token_ids)))


def _encoded_prefix(tokenizer: Any, text: str, full_ids: Sequence[int], label: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    _require(list(full_ids[: len(encoded)]) == encoded, f"{label} is not token-prefix stable")
    return len(encoded)


def validate_pairwise_matching(
    rows: Sequence[Mapping[str, Any]], token_ids_by_prompt: Mapping[str, Sequence[int]]
) -> None:
    by_replica: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_replica[int(row["paraphrase_replica"])].append(row)
    _require(set(by_replica) == {0, 1}, "both paraphrase replicas are required")
    for replica, replica_rows in by_replica.items():
        _require(len(replica_rows) == 6, f"replica {replica} does not contain six cells")
        token_counts = {len(token_ids_by_prompt[str(row["prompt_id"])]) for row in replica_rows}
        evidence_ranges = {tuple(row["evidence_token_range"]) for row in replica_rows}
        pressure_ranges = {tuple(row["pressure_token_range"]) for row in replica_rows}
        _require(
            len(token_counts) == len(evidence_ranges) == len(pressure_ranges) == 1,
            f"replica {replica} matching geometry differs",
        )
        allowed: set[int] = set()
        for start, stop in (*evidence_ranges, *pressure_ranges):
            allowed.update(range(int(start), int(stop)))
        for left_index, left in enumerate(replica_rows):
            left_ids = list(token_ids_by_prompt[str(left["prompt_id"])])
            for right in replica_rows[left_index + 1 :]:
                right_ids = list(token_ids_by_prompt[str(right["prompt_id"])])
                _require(len(left_ids) == len(right_ids), "matched prompts differ in length")
                changed = {
                    index
                    for index, (left_id, right_id) in enumerate(zip(left_ids, right_ids))
                    if left_id != right_id
                }
                _require(
                    changed <= allowed,
                    "pairwise token difference escapes declared manipulation ranges",
                )


def render_block(
    *,
    config: Mapping[str, Any],
    tokenizer: Any,
    source: Mapping[str, Any],
    block: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_text = source["text"]
    source_ids = list(source["token_ids"])
    _require(
        tokenizer.encode(source_text, add_special_tokens=False) == source_ids,
        "source text does not retokenize to exact source token IDs",
    )
    rendering = config["rendering"]
    suffix = rendering["generation_suffix"]
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    _require(
        source_text.endswith(suffix)
        and source_ids[-len(suffix_ids) :] == suffix_ids,
        "source generation suffix text or tokens changed",
    )
    base_text = source_text[: -len(suffix)]
    base_ids = source_ids[: -len(suffix_ids)]
    _require(
        tokenizer.encode(base_text, add_special_tokens=False) == base_ids,
        "source base prefix is not independently token-stable",
    )

    capture_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    token_ids_by_prompt: dict[str, list[int]] = {}
    evidence_counts: dict[int, set[int]] = defaultdict(set)
    pressure_counts: dict[int, set[int]] = defaultdict(set)
    full_counts: dict[int, set[int]] = defaultdict(set)
    for assignment in block:
        replica = int(assignment["paraphrase_replica"])
        evidence = str(assignment["evidence_level"])
        pressure = str(assignment["pressure_level"])
        evidence_text = rendering["evidence_text"][str(replica)][evidence]
        pressure_text = rendering["pressure_text"][str(replica)][pressure]
        through_fixed_before = base_text + rendering["fixed_before_evidence"]
        through_evidence = through_fixed_before + evidence_text
        through_fixed_between = (
            through_evidence + rendering["fixed_between_evidence_and_pressure"]
        )
        through_pressure = through_fixed_between + pressure_text
        full_text = through_pressure + rendering["fixed_after_pressure"]
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        _require(full_ids[: len(base_ids)] == base_ids, "base prefix tokens changed")
        evidence_start = _encoded_prefix(
            tokenizer, through_fixed_before, full_ids, "fixed-before-evidence prefix"
        )
        evidence_stop = _encoded_prefix(
            tokenizer, through_evidence, full_ids, "through-evidence prefix"
        )
        pressure_start = _encoded_prefix(
            tokenizer, through_fixed_between, full_ids, "fixed-between prefix"
        )
        pressure_stop = _encoded_prefix(
            tokenizer, through_pressure, full_ids, "through-pressure prefix"
        )
        _require(
            full_ids[-len(suffix_ids) :] == suffix_ids,
            "counterfactual generation suffix changed",
        )
        evidence_counts[replica].add(evidence_stop - evidence_start)
        pressure_counts[replica].add(pressure_stop - pressure_start)
        full_counts[replica].add(len(full_ids))
        condition_id = str(assignment["condition_id_sha256"])
        prompt_id = sha256_text(
            "swe-task-state-v4-counterfactual-pilot-prompt-v1\0" + condition_id
        )
        capture_source_id = sha256_text(prompt_id)
        capture_rows.append({"id": prompt_id, "token_ids": full_ids})
        token_ids_by_prompt[prompt_id] = full_ids
        key_rows.append(
            {
                "prompt_id": prompt_id,
                "capture_source_id_sha256": capture_source_id,
                "condition_id_sha256": condition_id,
                "condition_order_within_boundary": int(
                    assignment["condition_order_within_boundary"]
                ),
                "source_id_sha256": assignment["source_id_sha256"],
                "evidence_level": evidence,
                "pressure_level": pressure,
                "paraphrase_replica": replica,
                "rendered_prompt_sha256": sha256_text(full_text),
                "prompt_token_ids_sha256": token_ids_sha256(full_ids),
                "prompt_token_count": len(full_ids),
                "base_prefix_token_count": len(base_ids),
                "evidence_token_range": [evidence_start, evidence_stop],
                "pressure_token_range": [pressure_start, pressure_stop],
            }
        )
    _require(
        all(len(evidence_counts[replica]) == 1 for replica in (0, 1))
        and all(len(pressure_counts[replica]) == 1 for replica in (0, 1))
        and all(len(full_counts[replica]) == 1 for replica in (0, 1)),
        "factorial prompt matching failed",
    )
    _require(
        {next(iter(evidence_counts[replica])) for replica in (0, 1)} == {19},
        "evidence templates are no longer exactly 19 tokens",
    )
    _require(
        next(iter(pressure_counts[0])) == 26
        and next(iter(pressure_counts[1])) == 23,
        "pressure templates are no longer 26/23 matched tokens",
    )
    validate_pairwise_matching(key_rows, token_ids_by_prompt)
    summary = {
        "source_prompt_token_count": len(source_ids),
        "base_prefix_token_count": len(base_ids),
        "generation_suffix_token_count": len(suffix_ids),
        "evidence_segment_token_count_by_replica": {
            str(replica): next(iter(evidence_counts[replica])) for replica in (0, 1)
        },
        "pressure_segment_token_count_by_replica": {
            str(replica): next(iter(pressure_counts[replica])) for replica in (0, 1)
        },
        "full_prompt_token_count_by_replica": {
            str(replica): next(iter(full_counts[replica])) for replica in (0, 1)
        },
        "source_text_sha256": sha256_text(source_text),
        "source_token_ids_sha256": token_ids_sha256(source_ids),
        "base_prefix_token_ids_sha256": token_ids_sha256(base_ids),
        "generation_suffix_token_ids_sha256": token_ids_sha256(suffix_ids),
        "pairwise_difference_check": "passed_within_replica_only_declared_ranges_differ",
        "semantic_padding_tokens_used": False,
    }
    return capture_rows, key_rows, summary


def _artifact(path: Path) -> dict[str, Any]:
    try:
        display = str(path.resolve(strict=True).relative_to(ROOT))
    except ValueError:
        display = str(path.resolve(strict=True))
    return {"path": display, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def build_artifacts(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    sources = config["sources"]
    assignment_path = ROOT / sources["counterfactual_assignment"]["path"]
    prompt_path = ROOT / sources["development_prompt_bundle"]["path"]
    raw_config_path = ROOT / sources["raw_capture_protocol"]["path"]
    lexical_path_preflight((assignment_path, prompt_path, raw_config_path))
    canonical_path_preflight(
        input_paths=(assignment_path, prompt_path, raw_config_path), output_paths=()
    )
    _validate_bound_file(
        assignment_path, sources["counterfactual_assignment"], "assignment manifest"
    )
    _validate_bound_file(prompt_path, sources["development_prompt_bundle"], "prompt bundle")
    _require(
        sha256_file(raw_config_path) == sources["raw_capture_protocol"]["sha256"],
        "raw-capture protocol hash changed",
    )
    assignment = load_json(assignment_path)
    boundaries, block = validate_assignment(assignment, config)
    boundary_sources = {
        str(row["source_id_sha256"])
        for row in assignment["selection"]["boundaries"]
    }
    target_source = config["pilot_selection"]["source_id_sha256"]
    source, selected_counts = stream_selected_prompt(
        prompt_path,
        boundary_sources=boundary_sources,
        target_source=target_source,
    )
    expected_count = config["pilot_selection"]["expected_original_prompt_token_count"]
    _require(
        selected_counts[target_source] == expected_count
        and min(selected_counts.values()) == expected_count
        and [
            source_id
            for source_id, count in selected_counts.items()
            if count == expected_count
        ]
        == [target_source],
        "pinned feasibility selection is not the unique minimum token-count boundary",
    )
    tokenizer = load_tokenizer(config)
    capture_rows, key_rows, matching = render_block(
        config=config, tokenizer=tokenizer, source=source, block=block
    )
    capture_bundle = capture_rows
    condition_key = {
        "schema_version": 1,
        "kind": "swe_task_state_v4_counterfactual_pilot_condition_key",
        "status": "assignment_only_completion_labels_not_extracted",
        "scope": "label_sidecar_for_post_capture_analysis_only",
        "source_id_sha256": target_source,
        "assignment_count": len(key_rows),
        "records": key_rows,
        "capture_and_activation_feature_process_must_not_read_this_file": True,
        "completion_label_extraction_completed": False,
        "subjective_state_labels_present": False,
        "reserved_validation_access_authorized": False,
    }
    provenance = {
        "boundary": boundaries[0],
        "matching": matching,
        "capture_prompt_schema": {
            "allowed_row_keys": ["id", "token_ids"],
            "opaque_ids": True,
            "text_or_metadata_present": False,
        },
        "selection_audit": {
            "candidate_boundary_count": len(selected_counts),
            "minimum_token_count": min(selected_counts.values()),
            "minimum_is_unique": True,
            "labels_outcomes_completion_text_and_activations_used": False,
            "inferential_use_forbidden": True,
        },
    }
    return capture_bundle, condition_key, provenance


def materialize(output_dir: Path) -> dict[str, Any]:
    lexical_path_preflight((CONFIG_PATH, output_dir))
    canonical_path_preflight(input_paths=(CONFIG_PATH,), output_paths=(output_dir,))
    _require(
        CONFIG_PATH.resolve(strict=True) == CONFIG_PATH
        and sha256_file(CONFIG_PATH) == CONFIG_SHA256,
        "only the exact pilot config is allowed",
    )
    _require(not output_dir.exists() and not output_dir.is_symlink(), "output exists")
    config = validate_config(load_json(CONFIG_PATH))
    capture_bundle, condition_key, provenance = build_artifacts(config)
    output_dir = output_dir.resolve(strict=False)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    _require(not temporary.exists() and not temporary.is_symlink(), "temporary output exists")
    temporary.mkdir()
    try:
        capture_path = temporary / CAPTURE_BUNDLE_NAME
        key_path = temporary / CONDITION_KEY_NAME
        _write_json(capture_path, capture_bundle)
        _write_json(key_path, condition_key)
        manifest = {
            "schema_version": 1,
            "kind": KIND,
            "status": "passed_materialization_only_capture_and_generation_not_started",
            "scope": "one_complete_factorial_block_runtime_feasibility_only_no_inference",
            "config": _artifact(CONFIG_PATH),
            "implementation": _artifact(SCRIPT_PATH),
            "sources": {
                name: dict(record) for name, record in config["sources"].items()
            },
            "outputs": {
                "capture_bundle": {
                    "path": CAPTURE_BUNDLE_NAME,
                    "sha256": sha256_file(capture_path),
                    "size_bytes": capture_path.stat().st_size,
                    "prompt_count": len(capture_bundle),
                    "allowed_row_keys": ["id", "token_ids"],
                },
                "condition_key": {
                    "path": CONDITION_KEY_NAME,
                    "sha256": sha256_file(key_path),
                    "size_bytes": key_path.stat().st_size,
                    "must_not_be_read_by_capture_or_activation_feature_builder": True,
                },
            },
            "provenance": provenance,
            "execution_state": {
                "reference_generation_completed": False,
                "authenticated_activation_capture_completed": False,
                "counterfactual_completion_generation_completed": False,
                "completion_label_extraction_completed": False,
            },
            "matching_gates": {
                name: "passed" for name in config["matching_gates"]
            },
            "claim_scope": dict(config["claim_scope"]),
            "forbidden_path_guard_passed": True,
            "reserved_validation_access_authorized": False,
        }
        _write_json(temporary / MANIFEST_NAME, manifest)
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def verify_existing(output_dir: Path) -> dict[str, Any]:
    paths = (
        CONFIG_PATH,
        output_dir,
        output_dir / CAPTURE_BUNDLE_NAME,
        output_dir / CONDITION_KEY_NAME,
        output_dir / MANIFEST_NAME,
    )
    lexical_path_preflight(paths)
    canonical_path_preflight(input_paths=paths, output_paths=())
    _require(sha256_file(CONFIG_PATH) == CONFIG_SHA256, "pilot config hash changed")
    config = validate_config(load_json(CONFIG_PATH))
    manifest = load_json(output_dir / MANIFEST_NAME)
    _require(
        isinstance(manifest, dict)
        and manifest.get("schema_version") == 1
        and manifest.get("kind") == KIND
        and manifest.get("status")
        == "passed_materialization_only_capture_and_generation_not_started",
        "pilot manifest identity changed",
    )
    expected_capture, expected_key, expected_provenance = build_artifacts(config)
    observed_capture = load_json(output_dir / CAPTURE_BUNDLE_NAME)
    observed_key = load_json(output_dir / CONDITION_KEY_NAME)
    _require(
        canonical_json_bytes(observed_capture) == canonical_json_bytes(expected_capture),
        "capture bundle differs from deterministic materialization",
    )
    _require(
        canonical_json_bytes(observed_key) == canonical_json_bytes(expected_key),
        "condition key differs from deterministic materialization",
    )
    _require(
        canonical_json_bytes(manifest.get("provenance"))
        == canonical_json_bytes(expected_provenance),
        "pilot provenance differs from deterministic materialization",
    )
    for name, filename in (
        ("capture_bundle", CAPTURE_BUNDLE_NAME),
        ("condition_key", CONDITION_KEY_NAME),
    ):
        record = manifest["outputs"][name]
        path = output_dir / filename
        _require(
            record["sha256"] == sha256_file(path)
            and record["size_bytes"] == path.stat().st_size,
            f"{name} binding changed",
        )
    _require(
        all(set(row) == {"id", "token_ids"} for row in observed_capture),
        "capture bundle contains a label, text, or metadata field",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_only:
        manifest = verify_existing(args.output_dir)
        action = "verified"
    else:
        manifest = materialize(args.output_dir)
        action = "materialized"
    print(
        f"{action} {manifest['outputs']['capture_bundle']['prompt_count']} blinded "
        f"counterfactual prompts under {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
