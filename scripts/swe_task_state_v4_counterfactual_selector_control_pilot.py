#!/usr/bin/env python3
"""Freeze and materialize the CPU-only V4 selector-control scaffold.

This module never loads the model weights and has no generation or activation-
capture command.  It deterministically freezes two task-disjoint development
stages, but materializes prompts for Stage A only.  Stage B remains an identity-
only holdout in the split manifest.

The Stage-A capture bundle contains only opaque IDs and token IDs.  Semantic
conditions, selector maps, source/task grouping, and conceptual capture-position
names live only in a physically separate condition-key sidecar that capture and
generation processes are forbidden to read.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_counterfactual_selector_control_pilot.json"
)
CONFIG_SHA256 = "d5619231cc924a12303dbf147dfef53e45787aa6136dbeb94b706c1788aa3361"
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

SPLIT_MANIFEST_NAME = "split-manifest.json"
CAPTURE_BUNDLE_NAME = "stage-a-capture-prompts.json"
GENERATION_BUNDLE_NAME = "stage-a-generation-prompts.json"
CONDITION_KEY_NAME = "stage-a-condition-key.json"
MATERIALIZATION_MANIFEST_NAME = "materialization-manifest.json"

KIND = "swe_task_state_v4_counterfactual_selector_control_materialization"
EVIDENCE_LEVELS = (
    "clear_success",
    "clear_failure",
    "contradictory_ambiguous",
)
PRESSURE_LEVELS = ("neutral", "fixed_time_or_token_pressure")
REPLICAS = (0, 1)
FORBIDDEN_PATH_FRAGMENTS = ("reserved", "validation")


class SelectorControlError(ValueError):
    """Raised when a selector-control contract is not met."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectorControlError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SelectorControlError(f"cannot load JSON {path}: {error}") from error


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


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    return sha256_bytes(canonical_json_bytes(list(token_ids)))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _path_forbidden(path: Path) -> bool:
    normalized = Path(os.path.normpath(os.path.abspath(os.fspath(path))))
    return any(
        fragment in component.lower()
        for component in normalized.parts
        for fragment in FORBIDDEN_PATH_FRAGMENTS
    )


def lexical_path_preflight(paths: Iterable[Path]) -> None:
    """Reject forbidden path text before stat, resolve, hash, read, or write."""

    for path in paths:
        if _path_forbidden(path):
            raise SelectorControlError(
                f"forbidden path rejected before filesystem access: {path}"
            )


def canonical_path_preflight(
    *, input_paths: Iterable[Path], output_paths: Iterable[Path]
) -> None:
    for path, is_input in [
        *((item, True) for item in input_paths),
        *((item, False) for item in output_paths),
    ]:
        try:
            resolved = (
                path.resolve(strict=True)
                if is_input
                else path.parent.resolve(strict=True) / path.name
            )
        except OSError as error:
            raise SelectorControlError(f"cannot resolve path {path}: {error}") from error
        if _path_forbidden(resolved):
            raise SelectorControlError(f"forbidden canonical path rejected: {path}")


def _bound_path(record: Mapping[str, Any]) -> Path:
    raw = record.get("path")
    _require(isinstance(raw, str) and bool(raw), "bound source path is invalid")
    return ROOT / raw


def _validate_bound_file(record: Mapping[str, Any], label: str) -> Path:
    path = _bound_path(record)
    lexical_path_preflight((path,))
    canonical_path_preflight(input_paths=(path,), output_paths=())
    _require(_is_sha256(record.get("sha256")), f"{label} SHA-256 is invalid")
    _require(
        isinstance(record.get("size_bytes"), int)
        and not isinstance(record.get("size_bytes"), bool)
        and int(record["size_bytes"]) >= 0,
        f"{label} size is invalid",
    )
    _require(path.stat().st_size == record["size_bytes"], f"{label} size changed")
    _require(sha256_file(path) == record["sha256"], f"{label} hash changed")
    return path


def validate_config(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "selector-control config must be an object")
    config = dict(value)
    if CONFIG_SHA256 != "TO_BE_FROZEN_AFTER_IMPLEMENTATION":
        _require(sha256_file(CONFIG_PATH) == CONFIG_SHA256, "config hash changed")
    _require(
        config.get("schema_version") == 1
        and config.get("id")
        == "swe-task-state-v4-counterfactual-selector-control-pilot-v1"
        and config.get("status")
        == "prospective_development_scaffold_reserved_validation_closed_no_runtime",
        "selector-control config identity changed",
    )
    split = config.get("split")
    _require(
        isinstance(split, dict)
        and split.get("selection_domain")
        == "swe-task-state-v4-counterfactual-selector-control-split-v1"
        and split.get("expected_boundary_count_per_stage") == 20
        and split.get("expected_distinct_task_count_per_stage") == 20
        and split.get("expected_repository_count_per_stage") == 10
        and split.get("exclude_entire_observed_pilot_task") is True
        and _is_sha256(split.get("expected_split_canonical_sha256")),
        "split contract changed",
    )
    forbidden_selection = set(split.get("selection_inputs_forbidden", []))
    _require(
        {
            "prompt_text",
            "prompt_token_count",
            "labels",
            "outcomes",
            "completion_text",
            "completion_markers",
            "activation_values",
            "task_difficulty",
        }
        <= forbidden_selection,
        "label-independent split guard changed",
    )
    factorial = config.get("factorial_design")
    _require(
        isinstance(factorial, dict)
        and factorial.get("evidence_levels_in_semantic_order")
        == list(EVIDENCE_LEVELS)
        and factorial.get("pressure_levels_in_semantic_order")
        == list(PRESSURE_LEVELS)
        and factorial.get("paraphrase_replicas") == list(REPLICAS)
        and factorial.get("cells_per_boundary") == 12
        and factorial.get("stage_a_completion_prompt_count") == 240,
        "factorial contract changed",
    )
    selectors = config.get("selector_balance")
    _require(
        isinstance(selectors, dict)
        and selectors.get("evidence_codes_in_order") == ["A", "B", "C"]
        and selectors.get("pressure_codes_in_order") == ["X", "Y"]
        and selectors.get("each_rendered_selector_must_be_exactly_one_pinned_token")
        is True
        and selectors.get("semantic_description_order_fixed_across_all_code_maps")
        is True
        and selectors.get("exactly_two_declared_variable_selector_token_positions_per_prompt")
        is True,
        "selector-balance contract changed",
    )
    rendering = config.get("rendering")
    _require(
        isinstance(rendering, dict)
        and rendering.get("generation_suffix")
        == "<|im_start|>assistant\n<think>\n"
        and rendering.get("expected_common_bridge_token_count") == 32
        and rendering.get("expected_identical_tokens_after_pressure_selector") == 42
        and rendering.get("semantic_padding_tokens_used") is False,
        "rendering contract changed",
    )
    capture = config.get("capture_scaffold")
    _require(
        isinstance(capture, dict)
        and capture.get("expected_pre_manipulation_prompt_count") == 20
        and capture.get("expected_selector_tail_prompt_count") == 240
        and capture.get("expected_post_bridge_prompt_count") == 240
        and capture.get("expected_total_capture_prompt_count") == 500
        and capture.get("expected_generation_prompt_count") == 240
        and capture.get("capture_bundle_row_allowlist") == ["id", "token_ids"]
        and capture.get("model_generation_or_activation_capture_performed_by_this_scaffold")
        is False,
        "capture scaffold changed",
    )
    separation = config.get("separation_and_chronology")
    _require(
        isinstance(separation, dict)
        and separation.get("stage_a_capture_and_generation_bundles_contain_only_opaque_ids_and_token_ids")
        is True
        and separation.get("stage_a_condition_key_is_a_physically_separate_sidecar")
        is True
        and separation.get("capture_and_generation_processes_must_not_read_condition_key")
        is True
        and separation.get("stage_b_has_no_prompt_bundle_condition_key_capture_generation_completion_or_annotation_artifact")
        is True
        and separation.get("reserved_validation_access_authorized") is False,
        "separation contract changed",
    )
    gates = config.get("go_stop_gates")
    claims = config.get("claim_scope")
    _require(
        isinstance(gates, dict)
        and gates.get("stage_a_scaffold_authorizes_gpu_runtime") is False
        and gates.get("stage_b_runtime_or_prompt_materialization_authorized") is False
        and gates.get("later_stage_a_runtime_requires_new_hash_bound_execution_config")
        is True
        and isinstance(claims, dict)
        and claims.get("stage_a_or_stage_b_runtime_completed") is False
        and claims.get("factorial_behavior_or_activation_effect_established") is False
        and claims.get("incremental_activation_readout_established") is False
        and claims.get("private_chain_of_thought_reconstructed") is False
        and claims.get("subjective_confidence_or_doubt_inferred") is False
        and claims.get("experienced_stress_inferred") is False
        and claims.get("experienced_emotion_inferred") is False,
        "go/stop or claim limit changed",
    )
    output = config.get("output")
    _require(
        isinstance(output, dict)
        and output.get("split_manifest_name") == SPLIT_MANIFEST_NAME
        and output.get("stage_a_capture_bundle_name") == CAPTURE_BUNDLE_NAME
        and output.get("stage_a_generation_bundle_name") == GENERATION_BUNDLE_NAME
        and output.get("stage_a_condition_key_name") == CONDITION_KEY_NAME
        and output.get("materialization_manifest_name")
        == MATERIALIZATION_MANIFEST_NAME
        and output.get("new_output_no_clobber") is True
        and output.get("stage_b_output_files_forbidden") is True,
        "output contract changed",
    )
    return config


def validate_sources(config: Mapping[str, Any]) -> dict[str, Path]:
    sources = config["sources"]
    paths: dict[str, Path] = {}
    for name in (
        "counterfactual_assignment",
        "development_prompt_bundle",
        "counterfactual_protocol",
        "raw_capture_protocol",
        "raw_capture_implementation",
        "observed_one_boundary_receipt",
    ):
        paths[name] = _validate_bound_file(sources[name], name)
    return paths


def validate_assignment(value: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _require(isinstance(value, dict), "assignment manifest must be an object")
    _require(
        value.get("schema_version") == 1
        and value.get("kind")
        == "swe_task_state_v4_counterfactual_boundary_assignment"
        and value.get("status")
        == "passed_selection_and_assignment_only_generation_not_started"
        and value.get("reserved_validation_access_authorized") is False,
        "assignment identity changed",
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
    normalized_boundaries: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for raw in boundaries:
        _require(isinstance(raw, dict), "boundary row must be an object")
        row = dict(raw)
        _require(
            _is_sha256(row.get("source_id_sha256"))
            and _is_sha256(row.get("task_id_sha256"))
            and isinstance(row.get("repository"), str)
            and bool(row["repository"])
            and row.get("quantile") in {"one_third", "two_thirds"}
            and row["source_id_sha256"] not in seen_sources,
            "boundary identity changed",
        )
        seen_sources.add(row["source_id_sha256"])
        normalized_boundaries.append(row)
    normalized_assignments: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_conditions: set[str] = set()
    for raw in assignments:
        _require(isinstance(raw, dict), "assignment row must be an object")
        row = dict(raw)
        condition = row.get("condition_id_sha256")
        _require(
            row.get("source_id_sha256") in seen_sources
            and _is_sha256(condition)
            and condition not in seen_conditions
            and row.get("evidence_level") in EVIDENCE_LEVELS
            and row.get("pressure_level") in PRESSURE_LEVELS
            and row.get("paraphrase_replica") in REPLICAS
            and isinstance(row.get("condition_order_within_boundary"), int)
            and not isinstance(row.get("condition_order_within_boundary"), bool),
            "assignment row identity changed",
        )
        seen_conditions.add(str(condition))
        normalized_assignments.append(row)
        by_source[str(row["source_id_sha256"])].append(row)
    expected_cells = {
        (evidence, pressure, replica)
        for evidence in EVIDENCE_LEVELS
        for pressure in PRESSURE_LEVELS
        for replica in REPLICAS
    }
    for source_id in seen_sources:
        block = by_source[source_id]
        _require(
            len(block) == 12
            and {
                (
                    row["evidence_level"],
                    row["pressure_level"],
                    row["paraphrase_replica"],
                )
                for row in block
            }
            == expected_cells
            and sorted(row["condition_order_within_boundary"] for row in block)
            == list(range(12)),
            "assignment block is not a complete ordered factorial",
        )
    return normalized_boundaries, normalized_assignments


def select_split(
    boundaries: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    split = config["split"]
    pilot_source = split["observed_pilot_source_id_sha256"]
    pilot_task = split["observed_pilot_task_id_sha256"]
    matches = [row for row in boundaries if row["source_id_sha256"] == pilot_source]
    _require(
        len(matches) == 1 and matches[0]["task_id_sha256"] == pilot_task,
        "observed pilot source/task exclusion identity changed",
    )
    by_repo: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in boundaries:
        repo = str(row["repository"])
        task = str(row["task_id_sha256"])
        quantile = str(row["quantile"])
        _require(
            quantile not in by_repo[repo][task],
            "duplicate repository/task/quantile boundary",
        )
        by_repo[repo][task][quantile] = row
    domain = str(split["selection_domain"])
    selection: dict[str, Any] = {
        "schema_version": 1,
        "domain": domain,
        "observed_pilot_task_excluded": pilot_task,
        "stage_a": [],
        "stage_b_holdout_unopened": [],
    }
    for repository, tasks in sorted(by_repo.items()):
        eligible = [
            task
            for task, quantiles in tasks.items()
            if task != pilot_task
            and {"one_third", "two_thirds"} <= set(quantiles)
        ]
        eligible.sort(
            key=lambda task: sha256_text(
                domain + "\0" + repository + "\0" + task
            )
        )
        _require(len(eligible) >= 4, f"repository {repository} lacks four tasks")
        choices = (
            ("stage_a", eligible[0], "one_third"),
            ("stage_a", eligible[1], "two_thirds"),
            ("stage_b_holdout_unopened", eligible[2], "one_third"),
            ("stage_b_holdout_unopened", eligible[3], "two_thirds"),
        )
        for stage, task, quantile in choices:
            row = tasks[task][quantile]
            selection[stage].append(
                {
                    "repository": repository,
                    "task_id_sha256": task,
                    "quantile": quantile,
                    "source_id_sha256": row["source_id_sha256"],
                }
            )
    expected_n = int(split["expected_boundary_count_per_stage"])
    for stage in ("stage_a", "stage_b_holdout_unopened"):
        rows = selection[stage]
        _require(
            len(rows) == expected_n
            and len({row["repository"] for row in rows})
            == split["expected_repository_count_per_stage"]
            and len({row["task_id_sha256"] for row in rows})
            == split["expected_distinct_task_count_per_stage"]
            and all(row["task_id_sha256"] != pilot_task for row in rows),
            f"{stage} counts or pilot-task exclusion changed",
        )
    stage_a_tasks = {row["task_id_sha256"] for row in selection["stage_a"]}
    stage_b_tasks = {
        row["task_id_sha256"] for row in selection["stage_b_holdout_unopened"]
    }
    _require(not (stage_a_tasks & stage_b_tasks), "Stage A/B tasks overlap")
    observed = sha256_bytes(canonical_json_bytes(selection))
    _require(
        observed == split["expected_split_canonical_sha256"],
        "frozen split canonical identity changed",
    )
    return selection


def build_code_maps(
    selection: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    selector = config["selector_balance"]
    evidence_permutations = list(permutations(selector["evidence_codes_in_order"]))
    pressure_permutations = list(permutations(selector["pressure_codes_in_order"]))
    _require(
        len(evidence_permutations) == 6 and len(pressure_permutations) == 2,
        "selector permutation geometry changed",
    )
    # One 12-unit cycle contains every evidence x pressure map exactly once.
    # Its first four units cover four distinct evidence maps and two of each
    # pressure map, so a 40-unit stage differs by at most one use per evidence
    # map and is exactly balanced across pressure maps.
    combination_order = (
        (0, 0),
        (1, 1),
        (2, 0),
        (3, 1),
        (0, 1),
        (1, 0),
        (2, 1),
        (3, 0),
        (4, 0),
        (4, 1),
        (5, 0),
        (5, 1),
    )
    domain = str(selector["mapping_domain"])
    all_maps: dict[tuple[str, int], dict[str, Any]] = {}
    stage_a_records: list[dict[str, Any]] = []
    for stage in ("stage_a", "stage_b_holdout_unopened"):
        units = [
            (str(row["source_id_sha256"]), replica)
            for row in selection[stage]
            for replica in REPLICAS
        ]
        units.sort(
            key=lambda item: sha256_text(
                domain + "\0" + stage + "\0" + item[0] + "\0" + str(item[1])
            )
        )
        _require(len(units) == 40, f"{stage} mapping-unit count changed")
        for index, (source_id, replica) in enumerate(units):
            evidence_index, pressure_index = combination_order[
                index % len(combination_order)
            ]
            evidence_codes = evidence_permutations[evidence_index]
            pressure_codes = pressure_permutations[pressure_index]
            record = {
                "source_id_sha256": source_id,
                "paraphrase_replica": replica,
                "evidence_code_by_level": dict(zip(EVIDENCE_LEVELS, evidence_codes)),
                "pressure_code_by_level": dict(zip(PRESSURE_LEVELS, pressure_codes)),
            }
            all_maps[(source_id, replica)] = record
            if stage == "stage_a":
                stage_a_records.append(record)
        evidence_counts: dict[tuple[str, ...], int] = defaultdict(int)
        pressure_counts: dict[tuple[str, ...], int] = defaultdict(int)
        for source_id, replica in units:
            item = all_maps[(source_id, replica)]
            evidence_counts[
                tuple(item["evidence_code_by_level"][level] for level in EVIDENCE_LEVELS)
            ] += 1
            pressure_counts[
                tuple(item["pressure_code_by_level"][level] for level in PRESSURE_LEVELS)
            ] += 1
        _require(
            set(evidence_counts.values()) <= {6, 7}
            and set(pressure_counts.values()) == {20},
            f"{stage} code-map balance changed",
        )
    stage_a_mapping = {
        "schema_version": 1,
        "domain": domain,
        "stage": "stage_a",
        "records": stage_a_records,
    }
    expected = selector["expected_stage_a_mapping_canonical_sha256"]
    observed = sha256_bytes(canonical_json_bytes(stage_a_mapping))
    if expected != "TO_BE_FROZEN_AFTER_IMPLEMENTATION":
        _require(observed == expected, "Stage-A mapping canonical identity changed")
    return all_maps, stage_a_mapping


def stream_selected_prompts(path: Path, selected_sources: set[str]) -> dict[str, dict[str, Any]]:
    """Read only id/text/token_ids for the already-frozen Stage-A identities."""

    try:
        import ijson
    except ImportError as error:
        raise SelectorControlError("ijson is required for bounded source streaming") from error
    _require(len(selected_sources) == 20, "only 20 frozen Stage-A sources may be read")
    found: dict[str, dict[str, Any]] = {}
    current_id: str | None = None
    current_source: str | None = None
    current_text: str | None = None
    current_tokens: list[int] | None = None
    with path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle, use_float=True):
            if prefix == "item" and event == "start_map":
                current_id = None
                current_source = None
                current_text = None
                current_tokens = None
            elif prefix == "item.id" and event == "string":
                current_id = value
                current_source = sha256_text(value)
                if current_source in selected_sources:
                    current_tokens = []
            elif prefix == "item.text" and event == "string":
                if current_source in selected_sources:
                    current_text = value
            elif prefix == "item.token_ids.item" and event == "number":
                if current_tokens is not None:
                    _require(
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value >= 0,
                        "selected prompt contains invalid token ID",
                    )
                    current_tokens.append(value)
            elif prefix == "item" and event == "end_map":
                if current_source in selected_sources:
                    _require(
                        current_source not in found
                        and current_id is not None
                        and isinstance(current_text, str)
                        and isinstance(current_tokens, list),
                        "selected prompt is duplicate or incomplete",
                    )
                    found[current_source] = {
                        "id": current_id,
                        "text": current_text,
                        "token_ids": current_tokens,
                    }
    _require(set(found) == selected_sources, "not all frozen Stage-A prompts were found")
    return found


def load_tokenizer(config: Mapping[str, Any]) -> Any:
    from transformers import AutoTokenizer

    model = config["sources"]["model"]
    _require(model["revision"] == MODEL_REVISION, "model revision changed")
    _require(MODEL_SNAPSHOT.resolve(strict=True).is_dir(), "model snapshot missing")
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


def _encoded_prefix(tokenizer: Any, text: str, full_ids: Sequence[int], label: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    _require(list(full_ids[: len(encoded)]) == encoded, f"{label} is not token-prefix stable")
    return len(encoded)


def _format_line(template: str, *, code: str, description: str) -> str:
    _require(
        template.count("{code}") == 1 and template.count("{description}") == 1,
        "selector line template placeholders changed",
    )
    return template.format(code=code, description=description)


def _opaque_id(kind: str, identity: str) -> str:
    return sha256_text(
        "swe-task-state-v4-counterfactual-selector-control-v1\0"
        + kind
        + "\0"
        + identity
    )


def render_stage_a(
    *,
    config: Mapping[str, Any],
    tokenizer: Any,
    selection: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    code_maps: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rendering = config["rendering"]
    selector_config = config["selector_balance"]
    suffix = str(rendering["generation_suffix"])
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    _require(bool(suffix_ids), "generation suffix has no tokens")
    bridge = str(rendering["common_bridge"])
    bridge_ids = tokenizer.encode(bridge, add_special_tokens=False)
    _require(
        len(bridge_ids) == rendering["expected_common_bridge_token_count"],
        "common bridge token count changed",
    )
    for rendered_selector in selector_config["rendered_selector_tokens"]:
        _require(
            len(tokenizer.encode(rendered_selector, add_special_tokens=False)) == 1,
            f"selector {rendered_selector!r} is not exactly one token",
        )
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in assignments:
        by_source[str(row["source_id_sha256"])].append(row)

    capture_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    matching_blocks: list[dict[str, Any]] = []
    capture_ids_seen: set[str] = set()
    post_rows_by_id: dict[str, dict[str, Any]] = {}

    for split_row in selection["stage_a"]:
        source_id = str(split_row["source_id_sha256"])
        source = sources[source_id]
        source_text = str(source["text"])
        source_ids = list(source["token_ids"])
        _require(
            tokenizer.encode(source_text, add_special_tokens=False) == source_ids,
            "source text does not retokenize to exact IDs",
        )
        _require(
            source_text.endswith(suffix)
            and source_ids[-len(suffix_ids) :] == suffix_ids,
            "source generation suffix changed",
        )
        base_text = source_text[: -len(suffix)]
        base_ids = source_ids[: -len(suffix_ids)]
        _require(
            tokenizer.encode(base_text, add_special_tokens=False) == base_ids,
            "source base prefix is not independently token-stable",
        )
        _require(
            str(rendering["fixed_probe_prefix"]).startswith(suffix),
            "manipulated prompt does not pass through original generation boundary",
        )
        pre_id = _opaque_id("pre", source_id)
        _require(pre_id not in capture_ids_seen, "duplicate pre capture ID")
        capture_ids_seen.add(pre_id)
        capture_rows.append({"id": pre_id, "token_ids": source_ids})

        block = sorted(
            by_source[source_id],
            key=lambda row: int(row["condition_order_within_boundary"]),
        )
        _require(len(block) == 12, "Stage-A assignment block changed")
        for replica in REPLICAS:
            replica_rows = [
                row for row in block if int(row["paraphrase_replica"]) == replica
            ]
            _require(len(replica_rows) == 6, "replica block changed")
            mapping = code_maps[(source_id, replica)]
            evidence_map = mapping["evidence_code_by_level"]
            pressure_map = mapping["pressure_code_by_level"]
            prefix = base_text + str(rendering["fixed_probe_prefix"])
            for level in EVIDENCE_LEVELS:
                prefix += _format_line(
                    str(rendering["evidence_line_template"]),
                    code=str(evidence_map[level]),
                    description=str(rendering["evidence_text"][str(replica)][level]),
                )
            prefix += str(rendering["between_tables"])
            for level in PRESSURE_LEVELS:
                prefix += _format_line(
                    str(rendering["pressure_line_template"]),
                    code=str(pressure_map[level]),
                    description=str(rendering["pressure_text"][str(replica)][level]),
                )
            prefix += str(rendering["selected_evidence_prefix"])

            rendered_rows: list[dict[str, Any]] = []
            post_ids_by_condition: dict[str, list[int]] = {}
            for assignment in replica_rows:
                condition_id = str(assignment["condition_id_sha256"])
                evidence = str(assignment["evidence_level"])
                pressure = str(assignment["pressure_level"])
                evidence_code = str(evidence_map[evidence])
                pressure_code = str(pressure_map[pressure])
                through_evidence = prefix + " " + evidence_code
                through_between = through_evidence + str(rendering["between_selectors"])
                selector_tail_text = through_between + " " + pressure_code
                post_text = (
                    selector_tail_text
                    + str(rendering["after_pressure_selector"])
                    + bridge
                    + str(rendering["final_after_bridge"])
                )
                selector_ids = tokenizer.encode(
                    selector_tail_text, add_special_tokens=False
                )
                post_ids = tokenizer.encode(post_text, add_special_tokens=False)
                _require(
                    post_ids[: len(selector_ids)] == selector_ids,
                    "selector-tail capture is not a causal prefix of post-bridge prompt",
                )
                evidence_start = _encoded_prefix(
                    tokenizer, prefix, post_ids, "pre-evidence-selector prefix"
                )
                evidence_stop = _encoded_prefix(
                    tokenizer, through_evidence, post_ids, "evidence selector"
                )
                pressure_start = _encoded_prefix(
                    tokenizer, through_between, post_ids, "pre-pressure-selector prefix"
                )
                pressure_stop = _encoded_prefix(
                    tokenizer, selector_tail_text, post_ids, "pressure selector"
                )
                _require(
                    evidence_stop - evidence_start == 1
                    and pressure_stop - pressure_start == 1,
                    "each selected code must occupy exactly one token",
                )
                _require(
                    len(post_ids) - pressure_stop
                    == rendering["expected_identical_tokens_after_pressure_selector"],
                    "post-selector identical suffix token count changed",
                )
                _require(
                    post_ids[-len(suffix_ids) :] == suffix_ids,
                    "post-bridge generation suffix changed",
                )
                selector_id = _opaque_id("selector_tail", condition_id)
                post_id = _opaque_id("post_bridge", condition_id)
                _require(
                    selector_id not in capture_ids_seen
                    and post_id not in capture_ids_seen,
                    "duplicate Stage-A capture ID",
                )
                capture_ids_seen.update((selector_id, post_id))
                selector_row = {"id": selector_id, "token_ids": selector_ids}
                post_row = {"id": post_id, "token_ids": post_ids}
                capture_rows.extend((selector_row, post_row))
                generation_rows.append(dict(post_row))
                post_rows_by_id[post_id] = post_row
                post_ids_by_condition[condition_id] = post_ids
                key_row = {
                    "prompt_id": post_id,
                    "condition_id_sha256": condition_id,
                    "condition_order_within_boundary": int(
                        assignment["condition_order_within_boundary"]
                    ),
                    "source_id_sha256": source_id,
                    "task_id_sha256": split_row["task_id_sha256"],
                    "repository": split_row["repository"],
                    "quantile": split_row["quantile"],
                    "evidence_level": evidence,
                    "pressure_level": pressure,
                    "paraphrase_replica": replica,
                    "evidence_code_by_level": dict(evidence_map),
                    "pressure_code_by_level": dict(pressure_map),
                    "selected_evidence_code": evidence_code,
                    "selected_pressure_code": pressure_code,
                    "capture_prompt_ids": {
                        "pre_manipulation": pre_id,
                        "selector_tail": selector_id,
                        "post_bridge": post_id,
                    },
                    "capture_token_counts": {
                        "pre_manipulation": len(source_ids),
                        "selector_tail": len(selector_ids),
                        "post_bridge": len(post_ids),
                    },
                    "selector_token_indices": {
                        "evidence": evidence_start,
                        "pressure": pressure_start,
                    },
                    "prompt_token_ids_sha256": token_ids_sha256(post_ids),
                    "rendered_prompt_sha256": sha256_text(post_text),
                    "source_prompt_token_ids_sha256": token_ids_sha256(source_ids),
                }
                key_rows.append(key_row)
                rendered_rows.append(key_row)

            full_lengths = {
                row["capture_token_counts"]["post_bridge"] for row in rendered_rows
            }
            selector_lengths = {
                row["capture_token_counts"]["selector_tail"] for row in rendered_rows
            }
            selector_geometry = {
                (
                    row["selector_token_indices"]["evidence"],
                    row["selector_token_indices"]["pressure"],
                )
                for row in rendered_rows
            }
            _require(
                len(full_lengths) == len(selector_lengths) == len(selector_geometry) == 1,
                "within-replica prompt geometry changed",
            )
            evidence_index, pressure_index = next(iter(selector_geometry))
            allowed = {evidence_index, pressure_index}
            for left_index, left in enumerate(rendered_rows):
                left_ids = post_ids_by_condition[left["condition_id_sha256"]]
                for right in rendered_rows[left_index + 1 :]:
                    right_ids = post_ids_by_condition[right["condition_id_sha256"]]
                    _require(len(left_ids) == len(right_ids), "matched lengths differ")
                    changed = {
                        index
                        for index, (left_id, right_id) in enumerate(
                            zip(left_ids, right_ids)
                        )
                        if left_id != right_id
                    }
                    expected_changed = int(
                        left["evidence_level"] != right["evidence_level"]
                    ) + int(left["pressure_level"] != right["pressure_level"])
                    _require(
                        changed <= allowed and len(changed) == expected_changed,
                        "pairwise changes are not exactly the changed-factor selectors",
                    )
            matching_blocks.append(
                {
                    "source_id_sha256": source_id,
                    "paraphrase_replica": replica,
                    "full_prompt_token_count": next(iter(full_lengths)),
                    "selector_tail_token_count": next(iter(selector_lengths)),
                    "selector_token_indices": {
                        "evidence": evidence_index,
                        "pressure": pressure_index,
                    },
                    "semantic_description_order": {
                        "evidence": list(EVIDENCE_LEVELS),
                        "pressure": list(PRESSURE_LEVELS),
                    },
                    "declared_variable_token_positions": 2,
                    "pairwise_difference_check": "passed_exactly_one_changed_token_per_changed_factor",
                    "post_pressure_selector_identical_suffix_tokens": rendering[
                        "expected_identical_tokens_after_pressure_selector"
                    ],
                }
            )

    _require(
        len(capture_rows) == config["capture_scaffold"]["expected_total_capture_prompt_count"]
        and len(generation_rows)
        == config["capture_scaffold"]["expected_generation_prompt_count"]
        and len(key_rows) == config["factorial_design"]["stage_a_completion_prompt_count"]
        and len(matching_blocks) == 40,
        "Stage-A output counts changed",
    )
    _require(
        len({row["id"] for row in capture_rows}) == len(capture_rows)
        and len({row["id"] for row in generation_rows}) == len(generation_rows)
        and all(set(row) == {"id", "token_ids"} for row in capture_rows)
        and all(set(row) == {"id", "token_ids"} for row in generation_rows),
        "opaque prompt bundle schema or identity changed",
    )
    _require(
        all(post_rows_by_id[row["id"]] == row for row in generation_rows),
        "generation prompts differ from post-bridge capture prompts",
    )
    forbidden_capture_values = {
        str(row["condition_id_sha256"]) for row in key_rows
    } | {str(row["source_id_sha256"]) for row in key_rows} | {
        str(row["task_id_sha256"]) for row in key_rows
    }
    _require(
        not ({row["id"] for row in capture_rows} & forbidden_capture_values),
        "opaque capture ID reuses a condition/source/task identity",
    )
    condition_key = {
        "schema_version": 1,
        "kind": "swe_task_state_v4_counterfactual_selector_control_stage_a_condition_key",
        "status": "assignment_and_mapping_only_no_capture_generation_or_annotation_run",
        "scope": "physically_separate_post_capture_post_annotation_join_sidecar",
        "record_count": len(key_rows),
        "records": key_rows,
        "capture_and_generation_processes_must_not_read_this_file": True,
        "completion_annotation_must_be_condition_blind": True,
        "subjective_state_labels_present": False,
        "stage_b_records_present": False,
        "reserved_validation_access_authorized": False,
    }
    matching = {
        "block_count": len(matching_blocks),
        "blocks": matching_blocks,
        "all_salient_descriptions_present_in_fixed_semantic_order_within_replica": True,
        "all_conditions_have_two_declared_one_token_selectors": True,
        "all_pairwise_differences_confined_to_changed_factor_selectors": True,
        "common_bridge_token_count": len(bridge_ids),
        "identical_tokens_after_pressure_selector": rendering[
            "expected_identical_tokens_after_pressure_selector"
        ],
        "semantic_padding_tokens_used": False,
    }
    return capture_rows, generation_rows, condition_key, matching


def build_artifacts(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = validate_sources(config)
    assignment = load_json(paths["counterfactual_assignment"])
    boundaries, assignments = validate_assignment(assignment)
    selection = select_split(boundaries, config)
    code_maps, stage_a_mapping = build_code_maps(selection, config)
    stage_a_sources = {
        str(row["source_id_sha256"]) for row in selection["stage_a"]
    }
    stage_b_sources = {
        str(row["source_id_sha256"])
        for row in selection["stage_b_holdout_unopened"]
    }
    _require(not (stage_a_sources & stage_b_sources), "Stage A/B sources overlap")
    # This is the only source-prompt read: Stage B identities are deliberately
    # never passed to the streamer and therefore never materialized as prompts.
    sources = stream_selected_prompts(paths["development_prompt_bundle"], stage_a_sources)
    tokenizer = load_tokenizer(config)
    capture, generation, condition_key, matching = render_stage_a(
        config=config,
        tokenizer=tokenizer,
        selection=selection,
        assignments=assignments,
        sources=sources,
        code_maps=code_maps,
    )
    split_manifest = {
        "schema_version": 1,
        "kind": "swe_task_state_v4_counterfactual_selector_control_split_manifest",
        "status": "stage_a_and_untouched_stage_b_identities_frozen_no_model_runtime",
        "selection": selection,
        "selection_canonical_sha256": sha256_bytes(canonical_json_bytes(selection)),
        "selection_inputs_used": config["split"]["selection_inputs_allowed"],
        "selection_inputs_forbidden": config["split"]["selection_inputs_forbidden"],
        "stage_a_prompt_materialization_allowed_by_this_scaffold": True,
        "stage_b_prompt_materialization_allowed_by_this_scaffold": False,
        "stage_b_prompt_completion_capture_annotation_or_condition_key_present": False,
        "reserved_validation_access_authorized": False,
    }
    provenance = {
        "stage_a_mapping_canonical_sha256": sha256_bytes(
            canonical_json_bytes(stage_a_mapping)
        ),
        "matching": matching,
        "capture_prompt_schema": {
            "allowed_row_keys": ["id", "token_ids"],
            "opaque_ids": True,
            "text_or_condition_metadata_present": False,
        },
        "generation_prompt_schema": {
            "allowed_row_keys": ["id", "token_ids"],
            "opaque_ids": True,
            "equals_post_bridge_capture_subset": True,
        },
        "stage_b": {
            "identities_frozen": True,
            "prompt_sources_streamed": False,
            "prompt_bundle_materialized": False,
            "condition_key_materialized": False,
            "capture_generation_or_annotation_run": False,
        },
        "runtime": {
            "model_loaded": False,
            "gpu_generation_run": False,
            "activation_capture_run": False,
            "completion_annotation_run": False,
            "condition_key_join_run": False,
        },
    }
    return {
        SPLIT_MANIFEST_NAME: split_manifest,
        CAPTURE_BUNDLE_NAME: capture,
        GENERATION_BUNDLE_NAME: generation,
        CONDITION_KEY_NAME: condition_key,
        "provenance": provenance,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _display_path(path: Path) -> str:
    absolute = path.absolute()
    try:
        return str(absolute.relative_to(ROOT))
    except ValueError:
        return str(absolute)


def _artifact(path: Path, *, advertised_path: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": _display_path(advertised_path or resolved),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def materialize(output_dir: Path) -> dict[str, Any]:
    config = validate_config(load_json(CONFIG_PATH))
    source_paths = [
        _bound_path(config["sources"][name])
        for name in (
            "counterfactual_assignment",
            "development_prompt_bundle",
            "counterfactual_protocol",
            "raw_capture_protocol",
            "raw_capture_implementation",
            "observed_one_boundary_receipt",
        )
    ]
    lexical_path_preflight((CONFIG_PATH, *source_paths, output_dir))
    canonical_path_preflight(
        input_paths=(CONFIG_PATH, *source_paths), output_paths=(output_dir,)
    )
    _require(not output_dir.exists(), f"output already exists: {output_dir}")
    artifacts = build_artifacts(config)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        for name in (
            SPLIT_MANIFEST_NAME,
            CAPTURE_BUNDLE_NAME,
            GENERATION_BUNDLE_NAME,
            CONDITION_KEY_NAME,
        ):
            _write_json(temporary / name, artifacts[name])
        manifest = {
            "schema_version": 1,
            "kind": KIND,
            "status": "passed_scaffold_materialization_only_no_model_runtime",
            "config": _artifact(CONFIG_PATH),
            "implementation": _artifact(SCRIPT_PATH),
            "inputs": {
                name: {
                    "path": config["sources"][name]["path"],
                    "sha256": config["sources"][name]["sha256"],
                    "size_bytes": config["sources"][name]["size_bytes"],
                }
                for name in (
                    "counterfactual_assignment",
                    "development_prompt_bundle",
                    "counterfactual_protocol",
                    "raw_capture_protocol",
                    "raw_capture_implementation",
                    "observed_one_boundary_receipt",
                )
            },
            "outputs": {
                "split_manifest": {
                    **_artifact(
                        temporary / SPLIT_MANIFEST_NAME,
                        advertised_path=output_dir / SPLIT_MANIFEST_NAME,
                    ),
                    "stage_a_boundary_count": 20,
                    "stage_b_identity_only_boundary_count": 20,
                },
                "stage_a_capture_bundle": {
                    **_artifact(
                        temporary / CAPTURE_BUNDLE_NAME,
                        advertised_path=output_dir / CAPTURE_BUNDLE_NAME,
                    ),
                    "prompt_count": len(artifacts[CAPTURE_BUNDLE_NAME]),
                    "allowed_row_keys": ["id", "token_ids"],
                },
                "stage_a_generation_bundle": {
                    **_artifact(
                        temporary / GENERATION_BUNDLE_NAME,
                        advertised_path=output_dir / GENERATION_BUNDLE_NAME,
                    ),
                    "prompt_count": len(artifacts[GENERATION_BUNDLE_NAME]),
                    "allowed_row_keys": ["id", "token_ids"],
                },
                "stage_a_condition_key": {
                    **_artifact(
                        temporary / CONDITION_KEY_NAME,
                        advertised_path=output_dir / CONDITION_KEY_NAME,
                    ),
                    "record_count": artifacts[CONDITION_KEY_NAME]["record_count"],
                    "must_not_be_read_by_capture_generation_or_condition_blind_annotation": True,
                },
            },
            "provenance": artifacts["provenance"],
            "execution_state": {
                "stage_a_reference_run": False,
                "stage_a_activation_capture_run": False,
                "stage_a_completion_generation_run": False,
                "stage_a_condition_blind_annotation_run": False,
                "stage_a_condition_key_join_run": False,
                "stage_b_prompt_materialization_run": False,
                "stage_b_any_runtime": False,
            },
            "claim_scope": config["claim_scope"],
            "reserved_validation_access_authorized": False,
        }
        _write_json(temporary / MATERIALIZATION_MANIFEST_NAME, manifest)
        os.replace(temporary, output_dir)
        return load_json(output_dir / MATERIALIZATION_MANIFEST_NAME)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_existing(output_dir: Path) -> dict[str, Any]:
    config = validate_config(load_json(CONFIG_PATH))
    paths = [
        output_dir / name
        for name in (
            SPLIT_MANIFEST_NAME,
            CAPTURE_BUNDLE_NAME,
            GENERATION_BUNDLE_NAME,
            CONDITION_KEY_NAME,
            MATERIALIZATION_MANIFEST_NAME,
        )
    ]
    lexical_path_preflight((CONFIG_PATH, output_dir, *paths))
    canonical_path_preflight(input_paths=(CONFIG_PATH, *paths), output_paths=())
    _require(
        {path.name for path in output_dir.iterdir()} == {path.name for path in paths},
        "selector-control output contains an unexpected or missing file",
    )
    expected = build_artifacts(config)
    for name in (
        SPLIT_MANIFEST_NAME,
        CAPTURE_BUNDLE_NAME,
        GENERATION_BUNDLE_NAME,
        CONDITION_KEY_NAME,
    ):
        observed = load_json(output_dir / name)
        _require(
            canonical_json_bytes(observed) == canonical_json_bytes(expected[name]),
            f"existing {name} differs from deterministic materialization",
        )
    manifest = load_json(output_dir / MATERIALIZATION_MANIFEST_NAME)
    _require(
        manifest.get("status")
        == "passed_scaffold_materialization_only_no_model_runtime"
        and manifest.get("reserved_validation_access_authorized") is False,
        "materialization manifest identity changed",
    )
    for key, name in (
        ("split_manifest", SPLIT_MANIFEST_NAME),
        ("stage_a_capture_bundle", CAPTURE_BUNDLE_NAME),
        ("stage_a_generation_bundle", GENERATION_BUNDLE_NAME),
        ("stage_a_condition_key", CONDITION_KEY_NAME),
    ):
        record = manifest["outputs"][key]
        _require(
            record["path"] == _display_path(output_dir / name)
            and record["sha256"] == sha256_file(output_dir / name)
            and record["size_bytes"] == (output_dir / name).stat().st_size,
            f"manifest binding changed for {name}",
        )
    _require(
        manifest["config"]["sha256"] == sha256_file(CONFIG_PATH)
        and manifest["implementation"]["sha256"] == sha256_file(SCRIPT_PATH)
        and all(value is False for value in manifest["execution_state"].values()),
        "implementation/config/runtime binding changed",
    )
    capture = load_json(output_dir / CAPTURE_BUNDLE_NAME)
    generation = load_json(output_dir / GENERATION_BUNDLE_NAME)
    _require(
        all(set(row) == {"id", "token_ids"} for row in capture)
        and all(set(row) == {"id", "token_ids"} for row in generation),
        "opaque prompt schema changed",
    )
    _require(
        not any("stage-b" in path.name.lower() for path in output_dir.iterdir()),
        "Stage-B prompt or condition artifact exists",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="nonreserved scaffold output directory; defaults to config output root",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="rebuild in memory and verify an existing scaffold without model runtime",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = validate_config(load_json(CONFIG_PATH))
    output_dir = args.output_dir or ROOT / config["output"]["default_root"]
    manifest = (
        verify_existing(output_dir)
        if args.verify_existing
        else materialize(output_dir)
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "stage_a_capture_prompts": manifest["outputs"][
                    "stage_a_capture_bundle"
                ]["prompt_count"],
                "stage_a_generation_prompts": manifest["outputs"][
                    "stage_a_generation_bundle"
                ]["prompt_count"],
                "stage_b_runtime": manifest["execution_state"]["stage_b_any_runtime"],
                "reserved_validation_access_authorized": manifest[
                    "reserved_validation_access_authorized"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
