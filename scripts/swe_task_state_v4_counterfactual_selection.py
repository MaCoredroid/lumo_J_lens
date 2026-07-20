#!/usr/bin/env python3
"""Freeze label-independent V4 counterfactual boundaries and assignments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_counterfactual_state_protocol.json"
CONFIG_SHA256 = "7ad84e317960afb68d6d6ecd72acf206171748ef88cf52beb55d6a4ca0985327"
SCRIPT_PATH = Path(__file__).resolve()
KIND = "swe_task_state_v4_counterfactual_boundary_assignment"
FORBIDDEN_PATH_FRAGMENTS = ("reserved", "validation")
EVIDENCE_LEVELS = ("clear_success", "clear_failure", "contradictory_ambiguous")
PRESSURE_LEVELS = ("neutral", "fixed_time_or_token_pressure")


class SelectionError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionError(message)


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
        raise SelectionError(f"cannot load JSON: {path}: {error}") from error


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
            raise SelectionError(
                f"forbidden path rejected before filesystem access: {path}"
            )


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
            raise SelectionError(f"cannot resolve path: {path}: {error}") from error
        if any(
            fragment in component.lower()
            for fragment in FORBIDDEN_PATH_FRAGMENTS
            for component in resolved.parts
        ):
            raise SelectionError(f"forbidden canonical path rejected: {path}")


def validate_config(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "counterfactual config must be an object")
    config = dict(value)
    _require(
        set(config)
        == {
            "schema_version",
            "id",
            "status",
            "purpose",
            "sources",
            "boundary_selection",
            "factorial_design",
            "prompt_matching",
            "capture_and_generation",
            "externally_observable_targets",
            "estimands",
            "statistics",
            "separation_and_leakage_guards",
            "claim_scope",
        },
        "counterfactual config schema changed",
    )
    _require(
        config["schema_version"] == 1
        and config["id"]
        == "swe-task-state-v4-randomized-evidence-pressure-counterfactual-v1"
        and config["status"]
        == "prospective_development_protocol_reserved_validation_closed_not_executed",
        "counterfactual config identity changed",
    )
    boundary = config["boundary_selection"]
    _require(
        isinstance(boundary, dict)
        and boundary.get("stable_boundaries_per_task") == 2
        and boundary.get("algorithm")
        == "nearest_stable_boundary_to_one_third_and_two_thirds_of_each_complete_task_request_sequence_ties_to_lower_request"
        and boundary.get("labels_outcomes_completion_text_and_activation_values_forbidden")
        is True
        and boundary.get("expected_base_boundary_count") == 120,
        "counterfactual boundary-selection contract changed",
    )
    factorial = config["factorial_design"]
    _require(
        isinstance(factorial, dict)
        and factorial.get("evidence_factor", {}).get("levels")
        == list(EVIDENCE_LEVELS)
        and factorial.get("pressure_factor", {}).get("levels")
        == list(PRESSURE_LEVELS)
        and factorial.get("paraphrase_replicas_per_cell") == 2
        and factorial.get("cells_per_boundary") == 12
        and factorial.get("expected_prompt_count") == 1440
        and factorial.get("assignment", {}).get("seed") == 20260720,
        "counterfactual factorial contract changed",
    )
    _require(
        config["capture_and_generation"]
        == {
            "model_lens_and_layers": "exact_raw_capture_protocol",
            "capture_position": "final_prompt_token_before_counterfactual_completion",
            "raw_residual_and_public_j_state_capture_required": True,
            "completion_generation": "one_pinned_deterministic_generation_per_counterfactual_prompt",
            "activation_capture_must_precede_completion_label_extraction": True,
            "task_state_never_shared_across_counterfactual_conditions": True,
        },
        "counterfactual capture requirement changed",
    )
    guards = config["separation_and_leakage_guards"]
    _require(
        isinstance(guards, dict)
        and guards.get("assignment_condition_and_completion_labels_physically_absent_from_activation_feature_bundle")
        is True
        and guards.get("no_outcome_or_benchmark_gold_access") is True
        and guards.get("reserved_validation_access_authorized") is False,
        "counterfactual leakage guard changed",
    )
    return config


def validate_alignment(value: Any) -> list[dict[str, Any]]:
    _require(isinstance(value, dict), "alignment index must be an object")
    _require(
        value.get("schema_version") == 1
        and value.get("kind") == "swe_task_state_v4_label_free_alignment_index"
        and value.get("status") == "passed"
        and value.get("scope") == "grouping_order_and_stability_only_no_labels"
        and value.get("row_count") == 1708
        and value.get("stable_row_count") == 1606,
        "alignment identity or counts changed",
    )
    rows = value.get("rows")
    _require(isinstance(rows, list) and len(rows) == 1708, "alignment rows changed")
    expected_keys = {
        "global_index",
        "source_id_sha256",
        "task_id_sha256",
        "repository",
        "request_index",
        "stable_feature_eligible",
    }
    seen_sources: set[str] = set()
    seen_requests: set[tuple[str, str, int]] = set()
    previous_by_task: dict[tuple[str, str], int] = {}
    request_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    normalized: list[dict[str, Any]] = []
    for position, raw in enumerate(rows):
        _require(isinstance(raw, dict) and set(raw) == expected_keys, "alignment row schema changed")
        source = raw["source_id_sha256"]
        task = raw["task_id_sha256"]
        repository = raw["repository"]
        request = raw["request_index"]
        stable = raw["stable_feature_eligible"]
        _require(
            raw["global_index"] == position
            and _is_sha256(source)
            and _is_sha256(task)
            and isinstance(repository, str)
            and bool(repository)
            and isinstance(request, int)
            and not isinstance(request, bool)
            and request >= 1
            and isinstance(stable, bool),
            f"alignment row {position} is invalid",
        )
        key = (repository, task)
        identity = (*key, request)
        previous = previous_by_task.get(key)
        _require(
            source not in seen_sources
            and identity not in seen_requests
            and (previous is None or request > previous),
            "alignment identity or order changed",
        )
        seen_sources.add(source)
        seen_requests.add(identity)
        previous_by_task[key] = request
        request_indices[key].append(request)
        normalized.append(dict(raw))
    for values in request_indices.values():
        _require(values == list(range(1, len(values) + 1)), "task requests are incomplete")
    _require(
        sum(bool(row["stable_feature_eligible"]) for row in normalized) == 1606,
        "stable alignment support changed",
    )
    return normalized


def select_boundaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select two label-independent stable quantile boundaries per task."""

    by_task: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[(str(row["repository"]), str(row["task_id_sha256"]))].append(row)
    _require(len(by_task) == 60, "counterfactual source must contain exactly 60 tasks")
    selected: list[dict[str, Any]] = []
    for (repository, task), task_rows in sorted(by_task.items()):
        ordered = sorted(task_rows, key=lambda row: int(row["request_index"]))
        request_count = int(ordered[-1]["request_index"])
        stable = [row for row in ordered if row["stable_feature_eligible"] is True]
        internal = [
            row
            for row in stable
            if 1 < int(row["request_index"]) < request_count
        ]
        pool = internal if len(internal) >= 4 else stable
        _require(len(pool) >= 2, f"task has fewer than two selectable rows: {task}")
        chosen: set[int] = set()
        for quantile_name, numerator in (("one_third", 1), ("two_thirds", 2)):
            target_numerator = numerator * request_count
            candidates = [row for row in pool if int(row["global_index"]) not in chosen]
            row = min(
                candidates,
                key=lambda item: (
                    abs(3 * int(item["request_index"]) - target_numerator),
                    int(item["request_index"]),
                    int(item["global_index"]),
                ),
            )
            chosen.add(int(row["global_index"]))
            selected.append(
                {
                    "repository": repository,
                    "task_id_sha256": task,
                    "quantile": quantile_name,
                    "request_count": request_count,
                    "request_index": int(row["request_index"]),
                    "global_index": int(row["global_index"]),
                    "source_id_sha256": str(row["source_id_sha256"]),
                    "selection_pool": (
                        "stable_internal_requests" if pool is internal else "all_stable_requests"
                    ),
                }
            )
    _require(len(selected) == 120, "counterfactual selection count changed")
    _require(
        len({row["source_id_sha256"] for row in selected}) == 120,
        "counterfactual boundaries are duplicated",
    )
    return selected


def assign_conditions(
    selected: Sequence[Mapping[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    for boundary in selected:
        block = []
        for evidence in EVIDENCE_LEVELS:
            for pressure in PRESSURE_LEVELS:
                for replica in (0, 1):
                    identity = (
                        f"{seed}\0{boundary['source_id_sha256']}\0{evidence}\0"
                        f"{pressure}\0{replica}"
                    )
                    randomization_key = sha256_text("order\0" + identity)
                    block.append(
                        {
                            "condition_id_sha256": sha256_text("condition\0" + identity),
                            "randomization_key_sha256": randomization_key,
                            "evidence_level": evidence,
                            "pressure_level": pressure,
                            "paraphrase_replica": replica,
                        }
                    )
        block.sort(key=lambda item: item["randomization_key_sha256"])
        for position, item in enumerate(block):
            assignments.append(
                {
                    "source_id_sha256": boundary["source_id_sha256"],
                    "condition_order_within_boundary": position,
                    **item,
                }
            )
    _require(len(assignments) == 1440, "counterfactual assignment count changed")
    _require(
        len({item["condition_id_sha256"] for item in assignments}) == 1440,
        "counterfactual condition IDs are duplicated",
    )
    return assignments


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(ROOT)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_manifest(
    *, config: Mapping[str, Any], alignment_path: Path, alignment: Any
) -> dict[str, Any]:
    rows = validate_alignment(alignment)
    selected = select_boundaries(rows)
    seed = int(config["factorial_design"]["assignment"]["seed"])
    assignments = assign_conditions(selected, seed=seed)
    return {
        "schema_version": 1,
        "kind": KIND,
        "status": "passed_selection_and_assignment_only_generation_not_started",
        "scope": "label_independent_grouping_and_randomized_assignment_no_model_features_labels_outcomes_or_completions",
        "config": _artifact(CONFIG_PATH),
        "implementation": _artifact(SCRIPT_PATH),
        "alignment_index": _artifact(alignment_path),
        "selection": {
            "algorithm": config["boundary_selection"]["algorithm"],
            "task_count": 60,
            "boundary_count": len(selected),
            "boundaries": selected,
        },
        "assignment": {
            "seed": seed,
            "factorial_cells_per_boundary": 12,
            "assignment_count": len(assignments),
            "condition_order_is_deterministic_seeded_hash_order": True,
            "assignments": assignments,
        },
        "feature_boundary": {
            "condition_mapping_is_label_side_information_never_activation_feature": True,
            "selection_used_only_repository_task_request_and_stability_grouping": True,
            "labels_outcomes_completion_text_and_activation_values_used": False,
        },
        "execution_state": {
            "generation_completed": False,
            "activation_capture_completed": False,
            "completion_label_extraction_completed": False,
            "capture_requirement_is_prospective_not_a_completed_action": True,
        },
        "claim_scope": {
            "generation_or_capture_completed": False,
            "counterfactual_effect_established": False,
            "private_chain_of_thought_reconstructed": False,
            "subjective_emotion_confidence_doubt_or_stress_inferred": False,
        },
        "forbidden_path_guard_passed": True,
        "reserved_validation_access_authorized": False,
    }


def write_json_no_clobber(path: Path, value: Any) -> None:
    _require(
        not path.exists() and not path.is_symlink(),
        f"refusing to overwrite counterfactual selection: {path}",
    )
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
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    lexical_path_preflight((args.config, args.output))
    canonical_path_preflight(input_paths=(args.config,), output_paths=(args.output,))
    _require(
        args.config.resolve(strict=True) == CONFIG_PATH
        and sha256_file(CONFIG_PATH) == CONFIG_SHA256,
        "only the exact counterfactual protocol config is allowed",
    )
    config = validate_config(load_json(CONFIG_PATH))
    alignment_record = config["sources"]["label_free_alignment_index"]
    alignment_path = ROOT / alignment_record["path"]
    lexical_path_preflight((alignment_path,))
    canonical_path_preflight(input_paths=(alignment_path,), output_paths=())
    _require(
        alignment_path.stat().st_size == alignment_record["size_bytes"]
        and sha256_file(alignment_path) == alignment_record["sha256"],
        "alignment byte binding changed",
    )
    before = sha256_file(alignment_path)
    manifest = build_manifest(
        config=config,
        alignment_path=alignment_path,
        alignment=load_json(alignment_path),
    )
    _require(sha256_file(alignment_path) == before, "alignment changed during selection")
    write_json_no_clobber(args.output, manifest)
    print(
        f"wrote {manifest['selection']['boundary_count']} boundaries and "
        f"{manifest['assignment']['assignment_count']} assignments to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
