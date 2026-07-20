#!/usr/bin/env python3
"""Build a physically separate sidecar of frozen observable SWE events."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_observable_events.json"
SCRIPT_PATH = Path(__file__).resolve()
KIND = "swe_task_state_v4_observable_event_label_sidecar"
INDEX_KIND = "swe_task_state_v4_label_free_alignment_index"
SCHEMA_VERSION = 1
FROZEN_CONFIG_CANONICAL_SHA256 = (
    "ecc1734adee9d3fd414211ab83159e103152128fb471e2842ce688bd948a4466"
)
ACTION_COLLAPSE = {
    "inspect": "inspect",
    "edit": "edit",
    "validate": "check",
    "finalize": "check",
}


class EventError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EventError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)


def validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict) or hashlib.sha256(
        canonical_json_bytes(config)
    ).hexdigest() != FROZEN_CONFIG_CANONICAL_SHA256:
        raise EventError("observable-event protocol registry changed")
    if not isinstance(config, dict) or set(config) != {
        "schema_version",
        "id",
        "status",
        "sources",
        "eligibility_source",
        "temporal_contract",
        "targets",
        "positive_controls",
        "expected_stable_support",
        "forbidden_inputs",
        "claim_scope",
    }:
        raise EventError("observable-event config schema changed")
    if config["schema_version"] != 1 or config["id"] != (
        "swe-task-state-v4-observable-event-label-sidecar"
    ) or config["status"] != "development_only_reserved_validation_closed":
        raise EventError("observable-event config identity changed")
    if [source["row_count"] for source in config["sources"]] != [517, 370, 414, 407]:
        raise EventError("observable-event source partition changed")
    if sum(source["row_count"] for source in config["sources"]) != 1708:
        raise EventError("observable-event source count changed")
    eligibility = config["eligibility_source"]
    if (eligibility["all_rows"], eligibility["stable_rows"], eligibility["numerically_unstable_rows"]) != (1708, 1606, 102):
        raise EventError("observable-event eligibility counts changed")
    expected_targets = {
        "action_phase",
        "observable_rationale_language_marker",
        "transition_kind",
        "milestone_within_2",
        "tool_outcome",
        "validation_outcome",
        "terminal_event",
        "successful_validation_event",
        "failed_validation_event",
        "tool_failure_event",
    }
    if set(config["targets"]) != expected_targets:
        raise EventError("observable-event target registry changed")
    temporal = config["temporal_contract"]
    if (
        temporal.get("index_symbol") != "t"
        or temporal.get("feature_cutoff")
        != "h_t is the final captured prompt token immediately before ensuing assistant completion t"
        or set(temporal.get("target_completion_offsets", {})) != expected_targets
        or temporal.get("latest_allowed_target_completion") != "t+1"
        or temporal.get("completion_t_or_later_as_feature_forbidden") is not True
    ):
        raise EventError("observable-event temporal contract changed")
    if set(config["positive_controls"]) != {
        "has_prior_edit",
        "has_prior_validate",
        "previous_tool_failure",
    }:
        raise EventError("observable-event positive controls changed")
    if set(config["expected_stable_support"]) != expected_targets | set(config["positive_controls"]):
        raise EventError("observable-event support registry changed")
    claims = config["claim_scope"]
    if not isinstance(claims, dict) or set(claims) != {
        "private_chain_of_thought_reconstructed",
        "hidden_thought_or_understanding_established",
        "emotion_decoding_established",
        "causal_interpretation_established",
        "repository_held_out_predictability_established",
        "incremental_value_over_visible_baselines_established",
    } or any(value is not False for value in claims.values()):
        raise EventError("observable-event extraction cannot establish any claim")
    return config


def _available_label(value: str | None, *, reason: str | None = None) -> dict[str, Any]:
    return {
        "status": "available" if value is not None else "unknown",
        "value": value,
        "reason": None if value is not None else reason,
    }


def _validated_class_record(
    record: Mapping[str, Any],
    *,
    label: str,
    available_classes: set[str],
    unavailable_statuses: set[str],
) -> str | None:
    if not {"status", "class_id", "derivation"} <= set(record):
        raise EventError(f"{label} label record is incomplete")
    status = record.get("status")
    class_id = record.get("class_id")
    derivation = record.get("derivation")
    if not isinstance(derivation, str) or not derivation:
        raise EventError(f"{label} derivation is invalid")
    if status == "available":
        if class_id not in available_classes:
            raise EventError(f"available {label} class is invalid")
        return str(class_id)
    if status not in unavailable_statuses or class_id is not None:
        raise EventError(f"unavailable {label} state is invalid")
    return None


def _validate_completion_state(
    *,
    prompt_id: str,
    action: Mapping[str, Any],
    tool: Mapping[str, Any],
    validation: Mapping[str, Any],
    terminal: Mapping[str, Any],
    next_completion: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None, bool, bool | None]:
    action_value = _validated_class_record(
        action,
        label="action",
        available_classes=set(ACTION_COLLAPSE),
        unavailable_statuses={"missing"},
    )
    tool_value = _validated_class_record(
        tool,
        label="tool execution",
        available_classes={"success", "failure"},
        unavailable_statuses={"missing", "not_applicable"},
    )
    validation_value = _validated_class_record(
        validation,
        label="validation",
        available_classes={"success", "failure"},
        unavailable_statuses={"not_applicable"},
    )
    expected_terminal_keys = {
        "finish_reason",
        "is_episode_endpoint",
        "is_probeable_endpoint",
        "is_terminal",
        "is_terminal_completion",
    }
    if set(terminal) != expected_terminal_keys:
        raise EventError(f"terminal record schema is invalid: {prompt_id}")
    terminal_tuple = (
        terminal.get("finish_reason"),
        terminal.get("is_episode_endpoint"),
        terminal.get("is_probeable_endpoint"),
        terminal.get("is_terminal"),
        terminal.get("is_terminal_completion"),
    )
    allowed_terminal_tuples = {
        ("tool_calls", False, False, False, False),
        ("tool_calls", True, True, False, False),
        ("length", True, True, False, False),
        ("stop", True, True, True, True),
    }
    if terminal_tuple not in allowed_terminal_tuples:
        raise EventError(f"terminal state is invalid: {prompt_id}")

    next_status = next_completion.get("status")
    if next_status not in {
        "materialized_in_following_request",
        "terminal",
        "truncated",
        "unobserved_after_task_end",
    }:
        raise EventError(f"next-completion status is invalid: {prompt_id}")
    diagnosis = next_completion.get("diagnosis_expressed")
    if next_status == "materialized_in_following_request":
        if not isinstance(diagnosis, bool):
            raise EventError(f"materialized rationale marker is invalid: {prompt_id}")
        regex_hits = next_completion.get("diagnosis_regex_hits")
        if not isinstance(regex_hits, list) or not all(
            isinstance(hit, str) for hit in regex_hits
        ):
            raise EventError(f"materialized rationale-regex audit is invalid: {prompt_id}")
        if bool(regex_hits) is not diagnosis:
            raise EventError(f"rationale marker and regex audit disagree: {prompt_id}")
        if (
            tool.get("status") != "available"
            or terminal_tuple != ("tool_calls", False, False, False, False)
        ):
            raise EventError(f"materialized completion state is inconsistent: {prompt_id}")
    else:
        if diagnosis is not None or "diagnosis_regex_hits" in next_completion:
            raise EventError(f"non-materialized rationale marker is invalid: {prompt_id}")
        expected_nonmaterialized = {
            "terminal": (
                "available",
                "finalize",
                "not_applicable",
                "not_applicable",
                ("stop", True, True, True, True),
            ),
            "truncated": (
                "missing",
                None,
                "not_applicable",
                "not_applicable",
                ("length", True, True, False, False),
            ),
            "unobserved_after_task_end": (
                "missing",
                None,
                "missing",
                "not_applicable",
                ("tool_calls", True, True, False, False),
            ),
        }[str(next_status)]
        observed_nonmaterialized = (
            action.get("status"),
            action_value,
            tool.get("status"),
            validation.get("status"),
            terminal_tuple,
        )
        if observed_nonmaterialized != expected_nonmaterialized:
            raise EventError(f"non-materialized completion state is inconsistent: {prompt_id}")
    terminal_value = bool(terminal["is_terminal_completion"])
    return action_value, tool_value, validation_value, terminal_value, (
        diagnosis if isinstance(diagnosis, bool) else None
    )


def _stream_skeletons(path: Path) -> Iterable[dict[str, Any]]:
    import ijson

    with path.open("rb") as handle:
        for row in ijson.items(handle, "item"):
            raw_prompt_id = row.get("id")
            metadata = row.get("metadata")
            if not isinstance(raw_prompt_id, str) or not raw_prompt_id or not isinstance(metadata, dict):
                raise EventError("source prompt id or metadata is invalid")
            prompt_id = raw_prompt_id
            task = metadata.get("task")
            selection = metadata.get("selection")
            labels = metadata.get("labels")
            provenance = metadata.get("provenance")
            if not all(isinstance(value, dict) for value in (task, selection, labels, provenance)):
                raise EventError(f"source prompt metadata is incomplete: {prompt_id}")
            action = labels.get("action")
            tool = labels.get("tool_execution")
            validation = labels.get("validation")
            terminal = labels.get("terminal")
            next_completion = provenance.get("next_completion")
            if not all(isinstance(value, dict) for value in (action, tool, validation, terminal, next_completion)):
                raise EventError(f"observable label fields are incomplete: {prompt_id}")
            (
                action_value,
                tool_value,
                validation_value,
                terminal_value,
                diagnosis,
            ) = _validate_completion_state(
                prompt_id=prompt_id,
                action=action,
                tool=tool,
                validation=validation,
                terminal=terminal,
                next_completion=next_completion,
            )
            materialized = (
                next_completion.get("status") == "materialized_in_following_request"
            )
            request_index = selection.get("task_request_index")
            if isinstance(request_index, bool) or not isinstance(request_index, int) or request_index < 1:
                raise EventError(f"task request index is invalid: {prompt_id}")
            instance_id = task.get("instance_id")
            repository = task.get("repo")
            if not isinstance(instance_id, str) or not instance_id or not isinstance(repository, str) or not repository:
                raise EventError(f"task grouping fields are invalid: {prompt_id}")
            yield {
                "prompt_id": prompt_id,
                "source_id_sha256": sha256_text(prompt_id),
                "task_id": instance_id,
                "task_id_sha256": sha256_text(instance_id),
                "repository": repository,
                "request_index": request_index,
                "action": action_value,
                "tool_outcome": tool_value,
                "validation_outcome": validation_value,
                "terminal": terminal_value,
                "materialized": materialized,
                "diagnosis": diagnosis if materialized else None,
                "next_status": next_completion.get("status"),
            }


def _transition(previous: str, current: str) -> str:
    previous_phase = ACTION_COLLAPSE[previous]
    current_phase = ACTION_COLLAPSE[current]
    if previous_phase == current_phase:
        return "continuation"
    if (previous_phase, current_phase) in {
        ("inspect", "edit"),
        ("inspect", "check"),
        ("edit", "check"),
    }:
        return "advance"
    return "rework"


def derive_rows(
    skeletons: list[dict[str, Any]], *, unstable_ids: set[str]
) -> list[dict[str, Any]]:
    expected_skeleton_keys = {
        "prompt_id",
        "source_id_sha256",
        "task_id",
        "task_id_sha256",
        "repository",
        "request_index",
        "action",
        "tool_outcome",
        "validation_outcome",
        "terminal",
        "materialized",
        "diagnosis",
        "next_status",
    }
    by_task: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    task_repositories: dict[str, str] = {}
    seen_ids: set[str] = set()
    for row in skeletons:
        if not isinstance(row, dict) or set(row) != expected_skeleton_keys:
            raise EventError("observable skeleton schema changed")
        prompt_id = row["prompt_id"]
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in seen_ids:
            raise EventError(f"duplicate prompt id: {prompt_id}")
        seen_ids.add(prompt_id)
        task_id = row["task_id"]
        repository = row["repository"]
        if task_id in task_repositories and task_repositories[task_id] != repository:
            raise EventError("one task identity maps to multiple repositories")
        task_repositories[task_id] = repository
        request_index = row["request_index"]
        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(repository, str)
            or not repository
            or isinstance(request_index, bool)
            or not isinstance(request_index, int)
            or request_index < 1
        ):
            raise EventError("observable skeleton grouping is invalid")
        expected_task_hash = sha256_text(task_id)
        if row["task_id_sha256"] != expected_task_hash:
            raise EventError("observable skeleton task hash changed")
        if row["source_id_sha256"] != sha256_text(prompt_id):
            raise EventError("observable skeleton source hash changed")
        next_status = row["next_status"]
        if row["materialized"] is True:
            if (
                next_status != "materialized_in_following_request"
                or row["terminal"] is not False
                or not isinstance(row["diagnosis"], bool)
                or row["tool_outcome"] not in {"success", "failure"}
            ):
                raise EventError("materialized observable skeleton is inconsistent")
        elif row["materialized"] is False:
            if (
                next_status not in {"terminal", "truncated", "unobserved_after_task_end"}
                or row["diagnosis"] is not None
                or row["tool_outcome"] is not None
                or row["validation_outcome"] is not None
                or (row["terminal"] is True) != (next_status == "terminal")
                or (next_status == "terminal" and row["action"] != "finalize")
                or (next_status != "terminal" and row["action"] is not None)
            ):
                raise EventError("non-materialized observable skeleton is inconsistent")
        else:
            raise EventError("observable skeleton materialization flag is invalid")
        if row["validation_outcome"] is not None and row["action"] != "validate":
            raise EventError("validation outcome requires a validation action")
        task_rows = by_task[(repository, task_id)]
        if request_index in task_rows:
            raise EventError("duplicate task request index")
        task_rows[request_index] = row
    if not unstable_ids <= seen_ids:
        raise EventError("unstable eligibility contains an unknown identity")

    controls_before: dict[tuple[str, str, int], dict[str, bool]] = {}
    for (repository, task_id), task_rows in by_task.items():
        observed_indices = sorted(task_rows)
        if observed_indices != list(range(1, len(observed_indices) + 1)):
            raise EventError("task requests must form a complete consecutive sequence")
        prior = {"edit": False, "validate": False}
        for request_index in observed_indices:
            controls_before[(repository, task_id, request_index)] = dict(prior)
            action = task_rows[request_index]["action"]
            if action == "edit":
                prior["edit"] = True
            if action in {"validate", "finalize"}:
                prior["validate"] = True

    result = []
    for global_index, row in enumerate(skeletons):
        task_key = (row["repository"], row["task_id"])
        task_rows = by_task[task_key]
        request_index = row["request_index"]
        previous = task_rows.get(request_index - 1)
        following = task_rows.get(request_index + 1)
        action = row["action"]
        phase = ACTION_COLLAPSE[action] if action is not None else None
        transition = (
            _transition(previous["action"], action)
            if previous is not None
            and previous["action"] is not None
            and action is not None
            else None
        )
        milestone = None
        milestone_reason = None
        for candidate in (row, following):
            if candidate is None:
                milestone_reason = "incomplete_two_completion_window_before_milestone"
                break
            candidate_action = candidate["action"]
            if candidate_action is None:
                milestone_reason = "unclassified_action_before_milestone"
                break
            if candidate_action == "inspect":
                continue
            milestone = "edit" if candidate_action == "edit" else "check"
            break
        else:
            milestone = "none"

        materialized = row["materialized"]
        validation_value = row["validation_outcome"]
        tool_value = row["tool_outcome"]
        controls = controls_before[(*task_key, request_index)]
        output = {
            "global_index": global_index,
            "source_id_sha256": row["source_id_sha256"],
            "task_id_sha256": row["task_id_sha256"],
            "repository": row["repository"],
            "request_index": request_index,
            "stable_feature_eligible": row["prompt_id"] not in unstable_ids,
            "targets": {
                "action_phase": _available_label(
                    phase, reason="unknown_current_action"
                ),
                "observable_rationale_language_marker": _available_label(
                    "yes" if row["diagnosis"] is True else "no"
                    if row["diagnosis"] is False
                    else None,
                    reason="ensuing_completion_not_materialized",
                ),
                "transition_kind": _available_label(
                    transition, reason="immediate_action_pair_unavailable"
                ),
                "milestone_within_2": _available_label(
                    milestone, reason=milestone_reason
                ),
                "tool_outcome": _available_label(
                    tool_value, reason="ensuing_tool_outcome_unavailable"
                ),
                "validation_outcome": _available_label(
                    validation_value, reason="not_an_available_validation_completion"
                ),
                "terminal_event": _available_label(
                    "yes"
                    if row["terminal"]
                    else "no"
                    if row["next_status"]
                    in {"materialized_in_following_request", "truncated"}
                    else None,
                    reason="ensuing_completion_unobserved_after_task_end",
                ),
                "successful_validation_event": _available_label(
                    "yes" if validation_value == "success" else "no"
                    if materialized else None,
                    reason="ensuing_completion_not_materialized",
                ),
                "failed_validation_event": _available_label(
                    "yes" if validation_value == "failure" else "no"
                    if materialized else None,
                    reason="ensuing_completion_not_materialized",
                ),
                "tool_failure_event": _available_label(
                    "yes" if tool_value == "failure" else "no"
                    if materialized else None,
                    reason="ensuing_completion_not_materialized",
                ),
            },
            "positive_controls": {
                "has_prior_edit": "yes" if controls["edit"] else "no",
                "has_prior_validate": "yes" if controls["validate"] else "no",
                "previous_tool_failure": (
                    "yes"
                    if previous is not None and previous["tool_outcome"] == "failure"
                    else "no"
                ),
            },
        }
        result.append(output)
    return result


def support_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    stable = [row for row in rows if row["stable_feature_eligible"]]
    result: dict[str, dict[str, int]] = {}
    for target in next(iter(stable))["targets"]:
        counts: Counter[str] = Counter()
        for row in stable:
            record = row["targets"][target]
            counts[record["value"] if record["status"] == "available" else "unknown"] += 1
        counts.setdefault("unknown", 0)
        result[target] = dict(counts)
    for control in next(iter(stable))["positive_controls"]:
        counts = Counter(row["positive_controls"][control] for row in stable)
        result[control] = dict(counts)
    return result


def _write_no_clobber(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise EventError(f"refusing to overwrite observable-event sidecar: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def build_sidecar(config: Mapping[str, Any]) -> dict[str, Any]:
    skeletons = []
    source_bindings = []
    for source in config["sources"]:
        path = (ROOT / source["path"]).resolve(strict=True)
        if sha256_file(path) != source["sha256"]:
            raise EventError(f"observable-event source hash changed: {path}")
        rows = list(_stream_skeletons(path))
        if len(rows) != source["row_count"]:
            raise EventError(f"observable-event source row count changed: {path}")
        skeletons.extend(rows)
        source_bindings.append(dict(source))
    eligibility_source = config["eligibility_source"]
    eligibility_path = (ROOT / eligibility_source["path"]).resolve(strict=True)
    if sha256_file(eligibility_path) != eligibility_source["sha256"]:
        raise EventError("eligibility artifact hash changed")
    eligibility = load_json(eligibility_path)["eligibility"]
    if eligibility.get("exclusion_counts") != {"numerically_unstable": 102}:
        raise EventError("eligibility exclusion contract changed")
    unstable_ids = {
        str(record["row_id"])
        for record in eligibility["exclusions"]
        if record.get("reason") == "numerically_unstable"
    }
    if len(unstable_ids) != eligibility_source["numerically_unstable_rows"]:
        raise EventError("unstable eligibility identity count changed")
    rows = derive_rows(skeletons, unstable_ids=unstable_ids)
    support = support_summary(rows)
    if support != config["expected_stable_support"]:
        raise EventError(
            "observable-event support changed: "
            + json.dumps(support, sort_keys=True)
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "passed",
        "scope": "labels_and_grouping_only_never_feature_input",
        "config": {
            "path": str(CONFIG_PATH.relative_to(ROOT)),
            "sha256": sha256_file(CONFIG_PATH),
        },
        "implementation": {
            "path": str(SCRIPT_PATH.relative_to(ROOT)),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "sources": source_bindings,
        "eligibility_source": dict(eligibility_source),
        "row_count": len(rows),
        "stable_row_count": sum(row["stable_feature_eligible"] for row in rows),
        "support": support,
        "temporal_contract": config["temporal_contract"],
        "target_contracts": config["targets"],
        "positive_control_contracts": config["positive_controls"],
        "forbidden_inputs": config["forbidden_inputs"],
        "claim_scope": config["claim_scope"],
        "rows": rows,
    }


def build_label_free_index(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    """Project grouping/order fields into a physically separate artifact."""

    rows = sidecar.get("rows")
    if not isinstance(rows, list):
        raise EventError("observable-event sidecar rows are invalid")
    index_rows = []
    for expected_index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("global_index") != expected_index:
            raise EventError("observable-event row order changed")
        index_rows.append(
            {
                "global_index": expected_index,
                "source_id_sha256": row["source_id_sha256"],
                "task_id_sha256": row["task_id_sha256"],
                "repository": row["repository"],
                "request_index": row["request_index"],
                "stable_feature_eligible": row["stable_feature_eligible"],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": INDEX_KIND,
        "status": "passed",
        "scope": "grouping_order_and_stability_only_no_labels",
        "config": dict(sidecar["config"]),
        "implementation": dict(sidecar["implementation"]),
        "sources": list(sidecar["sources"]),
        "eligibility_source": dict(sidecar["eligibility_source"]),
        "row_count": len(index_rows),
        "stable_row_count": sum(row["stable_feature_eligible"] for row in index_rows),
        "feature_use": {
            "allowed": [
                "task-local ordering for causal temporal transforms",
                "repository and task grouping for held-out splits and weights",
                "stable eligibility filtering",
            ],
            "forbidden": [
                "hashing or one-hot encoding IDs as model features",
                "repository or request index as semantic model features",
            ],
        },
        "rows": index_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    if (
        config_path != CONFIG_PATH
        or args.output.exists()
        or args.output.is_symlink()
        or args.index_output.exists()
        or args.index_output.is_symlink()
        or args.output.resolve(strict=False) == args.index_output.resolve(strict=False)
    ):
        raise EventError("observable-event paths changed or output exists")
    config = validate_config(load_json(config_path))
    sidecar = build_sidecar(config)
    _write_no_clobber(args.output.resolve(strict=False), sidecar)
    _write_no_clobber(
        args.index_output.resolve(strict=False), build_label_free_index(sidecar)
    )
    print(f"wrote {sidecar['stable_row_count']} stable observable-event rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
