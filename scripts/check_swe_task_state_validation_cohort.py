#!/usr/bin/env python3
"""Fail-closed validation for the fresh SWE task-state cohort."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = ROOT / "configs/swe_task_state_validation_cohort.json"
DEFAULT_IMAGE_CONFIG = ROOT / "configs/swe_task_state_validation_image_digests.json"
CAMPAIGN_KIND = "swe_verified_behavioral_trajectory_campaign"
COHORT_KIND = "swe_verified_behavioral_n20_cohort"
IMAGE_PREFIX = "swebench/sweb.eval.x86_64."
INSTANCE_RE = re.compile(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class CohortValidationError(ValueError):
    """Raised when the frozen cohort cannot be reproduced exactly."""


@dataclass(frozen=True)
class SelectionResult:
    candidates: frozenset[str]
    eligible_repositories: tuple[str, ...]
    quotas: Mapping[str, int]
    ordered_by_repository: Mapping[str, tuple[str, ...]]
    batches: tuple[tuple[str, ...], tuple[str, ...]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CohortValidationError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be nonempty text")
    return value


def positive_integer(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 1,
        f"{label} must be a positive integer",
    )
    return value


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_file(path: Path, label: str) -> Any:
    try:
        raw = path.read_bytes()
        return json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CohortValidationError(f"non-finite JSON number in {label}: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CohortValidationError(f"cannot read {label}: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_sorted_strings(values: Sequence[str]) -> str:
    return sha256_json(sorted(values))


def validate_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return value


def repository_file(logical: Any, label: str) -> Path:
    value = nonempty_string(logical, label)
    require("\\" not in value and "\x00" not in value, f"unsafe {label}")
    relative = PurePosixPath(value)
    require(
        not relative.is_absolute()
        and relative.as_posix() == value
        and all(piece not in ("", ".", "..") for piece in relative.parts),
        f"non-canonical {label}: {value!r}",
    )
    path = ROOT.joinpath(*relative.parts)
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    return path


def run_command(command: Sequence[str], label: str, *, allow_no_match: bool = False) -> str:
    result = subprocess.run(
        list(command),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    allowed = {0, 1} if allow_no_match else {0}
    require(
        result.returncode in allowed,
        f"{label} failed with status {result.returncode}: {result.stderr.strip()}",
    )
    return result.stdout


def load_official_instance_ids(
    dataset: Mapping[str, Any], *, dataset_parquet: Path | None = None
) -> frozenset[str]:
    repo_id = nonempty_string(dataset.get("repo_id"), "dataset repo ID")
    revision = nonempty_string(dataset.get("revision"), "dataset revision")
    if dataset_parquet is None:
        try:
            from huggingface_hub import snapshot_download

            snapshot = Path(
                snapshot_download(
                    repo_id,
                    repo_type="dataset",
                    revision=revision,
                    allow_patterns=["data/*.parquet"],
                    local_files_only=True,
                )
            )
        except Exception as error:
            raise CohortValidationError(
                f"pinned dataset snapshot is unavailable locally: {repo_id}@{revision}: {error}"
            ) from error
        parquet_paths = sorted((snapshot / "data").glob("*.parquet"))
    else:
        parquet_paths = [dataset_parquet.expanduser().resolve(strict=True)]
    require(bool(parquet_paths), "pinned dataset snapshot has no Parquet data")
    try:
        import pyarrow.parquet as parquet

        values: list[str] = []
        for path in parquet_paths:
            column = parquet.read_table(path, columns=["instance_id"]).column("instance_id")
            values.extend(column.to_pylist())
    except Exception as error:
        raise CohortValidationError(f"cannot read pinned dataset instance IDs: {error}") from error
    require(
        bool(values)
        and all(isinstance(value, str) and INSTANCE_RE.fullmatch(value) for value in values),
        "pinned dataset contains invalid instance IDs",
    )
    require(len(values) == len(set(values)), "pinned dataset repeats an instance ID")
    return frozenset(values)


def canonical_tag(instance_id: str) -> str:
    require(INSTANCE_RE.fullmatch(instance_id) is not None, f"invalid instance ID: {instance_id}")
    return f"{IMAGE_PREFIX}{instance_id.replace('__', '_1776_')}:latest"


def list_cached_image_tags(*, docker_bin: str) -> dict[str, str]:
    output = run_command(
        [docker_bin, "image", "ls", "--format", "{{.Repository}}\t{{.Tag}}"],
        "Docker image listing",
    )
    result: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        require(len(parts) == 2, f"malformed Docker image listing row: {line!r}")
        repository, tag = parts
        if not repository.startswith(IMAGE_PREFIX) or tag != "latest":
            continue
        encoded = repository.removeprefix(IMAGE_PREFIX)
        if encoded.count("_1776_") != 1:
            continue
        owner, name = encoded.split("_1776_", 1)
        instance_id = f"{owner}__{name}"
        if INSTANCE_RE.fullmatch(instance_id) is None:
            continue
        full_tag = f"{repository}:{tag}"
        require(instance_id not in result, f"cached canonical image repeats {instance_id}")
        result[instance_id] = full_tag
    require(bool(result), "Docker has no canonical cached SWE images")
    return result


def inspect_images(tags: Sequence[str], *, docker_bin: str) -> dict[str, Mapping[str, Any]]:
    require(bool(tags), "no Docker images were supplied for inspection")
    output = run_command(
        [docker_bin, "image", "inspect", *tags],
        "Docker image inspection",
    )
    try:
        records = json.loads(output)
    except json.JSONDecodeError as error:
        raise CohortValidationError(
            f"Docker image inspection emitted invalid JSON: {error}"
        ) from error
    require(isinstance(records, list), "Docker image inspection must emit an array")
    by_tag: dict[str, Mapping[str, Any]] = {}
    requested = set(tags)
    for index, raw_record in enumerate(records):
        record = mapping(raw_record, f"Docker image inspection row {index}")
        for tag in record.get("RepoTags") or []:
            if tag in requested:
                require(tag not in by_tag, f"Docker inspection repeats tag {tag}")
                by_tag[tag] = record
    require(set(by_tag) == requested, "Docker inspection did not cover every requested image")
    return by_tag


def prior_used_instance_ids(
    selection: Mapping[str, Any], *, official_ids: frozenset[str], git_bin: str
) -> frozenset[str]:
    commit = nonempty_string(selection.get("prior_use_git_commit"), "prior-use commit")
    require(COMMIT_RE.fullmatch(commit) is not None, "prior-use commit must be a full SHA-1")
    resolved = run_command(
        [git_bin, "rev-parse", f"{commit}^{{commit}}"],
        "prior-use commit resolution",
    ).strip()
    require(resolved == commit, "prior-use commit does not resolve exactly")
    roots = [
        nonempty_string(value, f"prior-use root {index}")
        for index, value in enumerate(
            sequence(selection.get("prior_use_exclusion_roots"), "prior-use roots")
        )
    ]
    require(bool(roots) and len(roots) == len(set(roots)), "prior-use roots are empty or repeat")
    for root in roots:
        path = PurePosixPath(root)
        require(
            not path.is_absolute()
            and path.as_posix() == root
            and all(piece not in ("", ".", "..") for piece in path.parts),
            f"unsafe prior-use root: {root!r}",
        )
    output = run_command(
        [
            git_bin,
            "grep",
            "-h",
            "-o",
            "-E",
            INSTANCE_RE.pattern,
            commit,
            "--",
            *roots,
        ],
        "prior-use artifact scan",
        allow_no_match=True,
    )
    mentioned = {value for value in output.splitlines() if INSTANCE_RE.fullmatch(value)}
    return frozenset(mentioned & official_ids)


def reproduce_selection(
    selection: Mapping[str, Any],
    *,
    official_ids: frozenset[str],
    cached_amd64_ids: frozenset[str],
    prior_used_ids: frozenset[str],
) -> SelectionResult:
    seed = nonempty_string(selection.get("seed_text"), "selection seed")
    minimum = positive_integer(
        selection.get("repository_minimum_candidates"), "repository minimum candidates"
    )
    base_quota = positive_integer(
        selection.get("base_quota_per_repository"), "base repository quota"
    )
    selected_count = positive_integer(
        selection.get("selected_task_count"), "selected task count"
    )
    require(base_quota == 2, "fresh-state-v1 requires a base quota of two")
    candidates = frozenset((official_ids & cached_amd64_ids) - prior_used_ids)
    require(
        len(candidates) == selection.get("candidate_count_after_exclusion"),
        "candidate count after prior-use exclusion changed",
    )
    expected_candidate_hash = validate_sha256(
        selection.get("candidate_instance_ids_sha256"),
        "candidate instance-set hash",
    )
    require(
        sha256_sorted_strings(tuple(candidates)) == expected_candidate_hash,
        "candidate instance-set identity changed",
    )
    grouped: dict[str, list[str]] = defaultdict(list)
    for instance_id in candidates:
        grouped[instance_id.split("__", 1)[0]].append(instance_id)
    eligible = {
        repository: values
        for repository, values in grouped.items()
        if len(values) >= minimum
    }
    require(
        len(eligible) == selection.get("repository_count"),
        "eligible repository count changed",
    )
    eligible_instance_ids = tuple(
        instance_id for values in eligible.values() for instance_id in values
    )
    expected_eligible_hash = validate_sha256(
        selection.get("eligible_instance_ids_sha256"),
        "eligible instance-set hash",
    )
    require(
        sha256_sorted_strings(eligible_instance_ids) == expected_eligible_hash,
        "eligible instance-set identity changed",
    )
    require(selected_count >= base_quota * len(eligible), "base quotas exceed task count")
    for repository, values in eligible.items():
        values.sort(
            key=lambda instance_id: (
                hashlib.sha256(
                    seed.encode("utf-8") + b"\x00" + instance_id.encode("ascii")
                ).digest(),
                instance_id,
            )
        )
    allocation_order = sorted(
        eligible, key=lambda repository: (-len(eligible[repository]), repository)
    )
    quotas = {repository: base_quota for repository in eligible}
    extra_allocation: list[str] = []
    remaining = selected_count - sum(quotas.values())
    while remaining:
        progressed = False
        for repository in allocation_order:
            if remaining == 0:
                break
            if quotas[repository] >= len(eligible[repository]):
                continue
            quotas[repository] += 1
            extra_allocation.append(repository)
            remaining -= 1
            progressed = True
        require(progressed, "candidate capacity cannot satisfy the selected task count")
    selected_by_repository = {
        repository: tuple(eligible[repository][: quotas[repository]])
        for repository in sorted(eligible)
    }

    batch_lists: list[list[str]] = [[], []]
    next_index: dict[str, int] = {}
    for repository, values in selected_by_repository.items():
        for index, instance_id in enumerate(values[:base_quota]):
            batch_lists[index % 2].append(instance_id)
        next_index[repository] = base_quota
    for repository in extra_allocation:
        index = next_index[repository]
        destination = min(range(2), key=lambda batch: (len(batch_lists[batch]), batch))
        batch_lists[destination].append(selected_by_repository[repository][index])
        next_index[repository] += 1
    require(
        all(next_index[repository] == quotas[repository] for repository in quotas),
        "batch partition did not consume every selected task",
    )
    order_key = {
        instance_id: (repository, index)
        for repository, values in selected_by_repository.items()
        for index, instance_id in enumerate(values)
    }
    for values in batch_lists:
        values.sort(key=order_key.__getitem__)
    require(
        len(batch_lists[0]) == len(batch_lists[1]) == selected_count // 2,
        "fresh-state-v1 did not produce two balanced batches",
    )
    return SelectionResult(
        candidates=candidates,
        eligible_repositories=tuple(sorted(eligible)),
        quotas=dict(sorted(quotas.items())),
        ordered_by_repository=selected_by_repository,
        batches=(tuple(batch_lists[0]), tuple(batch_lists[1])),
    )


def load_and_validate_bindings(
    cohort_path: Path, image_config_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Mapping[str, Any]]]:
    cohort = mapping(strict_json_file(cohort_path, "cohort manifest"), "cohort manifest")
    require(cohort.get("schema_version") == 1, "cohort schema mismatch")
    require(cohort.get("kind") == COHORT_KIND, "cohort kind mismatch")
    require(
        cohort.get("lens_outputs_used_for_selection") is False
        and cohort.get("official_outcomes_used_for_selection") is False,
        "cohort selection used forbidden post-generation evidence",
    )
    dataset = mapping(cohort.get("dataset"), "cohort dataset")
    pins = mapping(cohort.get("pins"), "cohort pins")
    for field, binding_name in (
        ("action_protocol", "action_protocol_sha256"),
        ("chat_template", "chat_template_sha256"),
    ):
        binding = mapping(cohort.get(field), f"cohort {field}")
        path = repository_file(binding.get("path"), f"cohort {field} path")
        expected = validate_sha256(binding.get("sha256"), f"cohort {field} hash")
        require(sha256_file(path) == expected, f"cohort {field} bytes changed")
        require(pins.get(binding_name) == expected, f"cohort {field} duplicate pin differs")
    image_hash = validate_sha256(
        pins.get("image_registry_sha256"), "cohort image registry hash"
    )
    require(sha256_file(image_config_path) == image_hash, "image registry bytes changed")
    image_config = mapping(
        strict_json_file(image_config_path, "image registry"), "image registry"
    )
    require(image_config.get("schema_version") == 1, "image registry schema mismatch")

    cohort_rows = [
        mapping(value, f"cohort row {index}")
        for index, value in enumerate(sequence(cohort.get("cohorts"), "cohort rows"))
    ]
    require(len(cohort_rows) == 2, "fresh-state-v1 requires exactly two campaigns")
    campaigns: list[Mapping[str, Any]] = []
    combined: list[str] = []
    for index, row in enumerate(cohort_rows):
        campaign_path = repository_file(
            row.get("campaign_path"), f"cohort campaign {index} path"
        )
        expected_hash = validate_sha256(
            row.get("campaign_sha256"), f"cohort campaign {index} hash"
        )
        require(sha256_file(campaign_path) == expected_hash, f"campaign {index} bytes changed")
        campaign = mapping(
            strict_json_file(campaign_path, f"campaign {index}"), f"campaign {index}"
        )
        require(campaign.get("schema_version") == 1, f"campaign {index} schema mismatch")
        require(campaign.get("kind") == CAMPAIGN_KIND, f"campaign {index} kind mismatch")
        require(campaign.get("dataset") == dataset, f"campaign {index} dataset pin differs")
        campaign_selection = mapping(campaign.get("selection"), f"campaign {index} selection")
        require(
            campaign_selection.get("lens_outputs_used") is False
            and campaign_selection.get("official_outcomes_used") is False,
            f"campaign {index} selection used forbidden evidence",
        )
        instance_ids = [
            nonempty_string(value, f"campaign {index} instance {item_index}")
            for item_index, value in enumerate(
                sequence(campaign.get("instance_ids"), f"campaign {index} instances")
            )
        ]
        require(len(instance_ids) == 10, f"campaign {index} must contain ten tasks")
        require(len(instance_ids) == len(set(instance_ids)), f"campaign {index} repeats a task")
        require(row.get("instance_ids") == instance_ids, f"campaign {index} row coverage differs")
        combined.extend(instance_ids)
        campaigns.append(campaign)
    require(len(combined) == len(set(combined)), "validation campaigns overlap")
    require(cohort.get("instance_ids") == combined, "combined cohort order changed")
    require(
        mapping(campaigns[0].get("generation"), "campaign generation")
        == mapping(campaigns[1].get("generation"), "campaign generation"),
        "campaign generation profiles differ",
    )
    return cohort, image_config, campaigns


def validate_image_pins(
    image_config: Mapping[str, Any],
    *,
    selected_ids: Sequence[str],
    cached_tags: Mapping[str, str],
    inspected: Mapping[str, Mapping[str, Any]],
) -> None:
    images = mapping(image_config.get("images"), "image registry images")
    require(set(images) == set(selected_ids), "image registry coverage differs from cohort")
    for instance_id in selected_ids:
        architecture_map = mapping(images.get(instance_id), f"image pin {instance_id}")
        require(
            set(architecture_map) == {"x86_64"},
            f"image pin architecture differs: {instance_id}",
        )
        pin = mapping(architecture_map.get("x86_64"), f"x86_64 image pin {instance_id}")
        require(set(pin) == {"reference", "image_id"}, f"image pin fields differ: {instance_id}")
        image_id = nonempty_string(pin.get("image_id"), f"image ID {instance_id}")
        reference = nonempty_string(pin.get("reference"), f"image reference {instance_id}")
        require(
            re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is not None,
            f"invalid image ID: {instance_id}",
        )
        require(
            cached_tags.get(instance_id) == canonical_tag(instance_id),
            f"image tag changed: {instance_id}",
        )
        record = inspected[canonical_tag(instance_id)]
        require(record.get("Architecture") == "amd64", f"image is not amd64: {instance_id}")
        require(record.get("Id") == image_id, f"local image bytes differ: {instance_id}")
        require(
            reference in (record.get("RepoDigests") or []),
            f"image digest is unproven: {instance_id}",
        )


def _positive_index(value: Any, label: str) -> int:
    return positive_integer(value, label)


def _prompt_payload_sha256(prompt: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(prompt))
    metadata = mapping(payload.get("metadata"), "prompt metadata")
    provenance = mapping(metadata.get("provenance"), "prompt provenance")
    provenance.pop("prompt_record_payload_sha256", None)
    return sha256_json(payload)


def validate_materialized_bundle(
    cohort: Mapping[str, Any],
    campaigns: Sequence[Mapping[str, Any]],
    *,
    cohort_path: Path,
    prompts_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Bind a dense bundle to every frozen task and all probeable requests."""
    require(
        prompts_path.is_file() and not prompts_path.is_symlink(),
        f"prompt bundle is not a regular file: {prompts_path}",
    )
    require(
        summary_path.is_file() and not summary_path.is_symlink(),
        f"prompt summary is not a regular file: {summary_path}",
    )
    prompts = [
        mapping(value, f"prompt row {index}")
        for index, value in enumerate(
            sequence(strict_json_file(prompts_path, "prompt bundle"), "prompt bundle")
        )
    ]
    summary = mapping(strict_json_file(summary_path, "prompt summary"), "prompt summary")
    require(bool(prompts), "materialized prompt bundle is empty")

    cohort_rows = [
        mapping(value, f"cohort row {index}")
        for index, value in enumerate(sequence(cohort.get("cohorts"), "cohort rows"))
    ]
    require(len(cohort_rows) == len(campaigns) == 2, "bundle cohort count changed")
    cohort_sha256 = sha256_file(cohort_path)
    expected_instance_ids = [
        nonempty_string(value, "frozen instance ID")
        for value in sequence(cohort.get("instance_ids"), "frozen instance IDs")
    ]
    expected_campaign_hashes = [
        validate_sha256(row.get("campaign_sha256"), f"campaign {index} hash")
        for index, row in enumerate(cohort_rows)
    ]
    expected_cohort_ids = [
        nonempty_string(row.get("id"), f"cohort {index} ID")
        for index, row in enumerate(cohort_rows)
    ]
    require(
        len(expected_cohort_ids) == len(set(expected_cohort_ids)),
        "cohort IDs repeat",
    )
    task_binding: dict[str, tuple[int, str, str, list[str]]] = {}
    for index, (row, campaign) in enumerate(zip(cohort_rows, campaigns, strict=True)):
        instance_ids = [
            nonempty_string(value, f"cohort {index} instance ID")
            for value in sequence(row.get("instance_ids"), f"cohort {index} instances")
        ]
        require(campaign.get("instance_ids") == instance_ids, "campaign task order changed")
        for instance_id in instance_ids:
            require(instance_id not in task_binding, "bundle cohort repeats a task")
            task_binding[instance_id] = (
                index,
                expected_cohort_ids[index],
                expected_campaign_hashes[index],
                instance_ids,
            )
    require(
        list(task_binding) == expected_instance_ids,
        "bundle task order differs from the frozen cohort",
    )

    prompt_ids: set[str] = set()
    global_indices: list[int] = []
    first_seen_tasks: list[str] = []
    observed_by_task: dict[str, list[int]] = defaultdict(list)
    probeable_by_task: dict[str, list[int]] = {}
    expected_summary_prompts: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        prompt_id = nonempty_string(prompt.get("id"), f"prompt {index} ID")
        require(prompt_id not in prompt_ids, "materialized prompt IDs repeat")
        prompt_ids.add(prompt_id)
        metadata = mapping(prompt.get("metadata"), f"prompt {prompt_id} metadata")
        task = mapping(metadata.get("task"), f"prompt {prompt_id} task")
        selection = mapping(
            metadata.get("selection"), f"prompt {prompt_id} selection"
        )
        provenance = mapping(
            metadata.get("provenance"), f"prompt {prompt_id} provenance"
        )
        cohort_metadata = mapping(
            metadata.get("cohort"), f"prompt {prompt_id} cohort"
        )
        instance_id = nonempty_string(task.get("instance_id"), "prompt instance ID")
        require(instance_id in task_binding, f"prompt contains an unfrozen task: {instance_id}")
        cohort_index, cohort_id, campaign_hash, cohort_instance_ids = task_binding[
            instance_id
        ]
        require(
            cohort_metadata.get("id") == cohort_id
            and cohort_metadata.get("index") == cohort_index
            and cohort_metadata.get("campaign_sha256") == campaign_hash
            and cohort_metadata.get("cohort_manifest_sha256") == cohort_sha256
            and cohort_metadata.get("source_task_instance_ids") == cohort_instance_ids,
            f"prompt cohort binding differs: {prompt_id}",
        )
        task_request_index = _positive_index(
            selection.get("task_request_index"), "task request index"
        )
        global_request_index = _positive_index(
            selection.get("global_request_index"), "global request index"
        )
        probeable = [
            _positive_index(value, "probeable request index")
            for value in sequence(
                task.get("probeable_request_indices"), "probeable request indices"
            )
        ]
        require(bool(probeable) and probeable == sorted(set(probeable)), "probeable request indices are invalid")
        require(
            selection.get("max_checkpoints") is None
            and selection.get("probeable_request_indices") == probeable
            and selection.get("candidate_count") == len(probeable)
            and selection.get("checkpoint_count") == len(probeable)
            and task.get("probeable_request_count") == len(probeable),
            f"prompt is not bound to the all-probeable contract: {prompt_id}",
        )
        if instance_id not in observed_by_task:
            first_seen_tasks.append(instance_id)
            probeable_by_task[instance_id] = probeable
        else:
            require(
                probeable_by_task[instance_id] == probeable,
                f"probeable request declaration changed within {instance_id}",
            )
        observed_by_task[instance_id].append(task_request_index)
        global_indices.append(global_request_index)
        payload_hash = validate_sha256(
            provenance.get("prompt_record_payload_sha256"),
            f"prompt {prompt_id} payload hash",
        )
        require(
            _prompt_payload_sha256(prompt) == payload_hash,
            f"prompt payload hash differs: {prompt_id}",
        )
        expected_summary_prompts.append(
            {
                "id": prompt_id,
                "cohort_id": cohort_id,
                "instance_id": instance_id,
                "global_request_index": global_request_index,
                "prompt_record_payload_sha256": payload_hash,
            }
        )

    require(first_seen_tasks == expected_instance_ids, "materialized task order changed")
    require(
        len(global_indices) == len(set(global_indices))
        and global_indices == sorted(global_indices),
        "materialized global request order changed",
    )
    for instance_id in expected_instance_ids:
        require(
            observed_by_task[instance_id] == probeable_by_task[instance_id],
            f"materialized bundle does not cover every probeable request: {instance_id}",
        )

    pins = mapping(cohort.get("pins"), "cohort pins")
    require(
        summary.get("schema_version") == 1
        and summary.get("kind") == "swe_verified_behavioral_probe_combination"
        and summary.get("cohort_manifest_sha256") == cohort_sha256
        and summary.get("source_campaign_sha256s") == expected_campaign_hashes
        and summary.get("campaign_sha256s") == expected_campaign_hashes
        and summary.get("action_protocol_sha256")
        == pins.get("action_protocol_sha256")
        and summary.get("chat_template_sha256") == pins.get("chat_template_sha256")
        and summary.get("cohort_count") == len(cohort_rows)
        and summary.get("task_count") == len(expected_instance_ids)
        and summary.get("prompt_count") == len(prompts)
        and summary.get("prompt_bundle_sha256") == sha256_file(prompts_path),
        "materialized summary identity or counts changed",
    )
    require(
        summary.get("prompts") == expected_summary_prompts,
        "materialized summary prompt binding changed",
    )
    summary_cohorts = [
        mapping(value, f"summary cohort {index}")
        for index, value in enumerate(sequence(summary.get("cohorts"), "summary cohorts"))
    ]
    require(len(summary_cohorts) == len(cohort_rows), "summary cohort count changed")
    for index, (row, summary_cohort) in enumerate(
        zip(cohort_rows, summary_cohorts, strict=True)
    ):
        require(
            summary_cohort.get("id") == expected_cohort_ids[index]
            and summary_cohort.get("index") == index
            and summary_cohort.get("campaign_sha256") == expected_campaign_hashes[index]
            and summary_cohort.get("cohort_manifest_sha256") == cohort_sha256
            and summary_cohort.get("source_task_instance_ids")
            == row.get("instance_ids")
            and summary_cohort.get("source_task_count")
            == len(sequence(row.get("instance_ids"), "cohort instances")),
            f"summary cohort binding changed: {index}",
        )

    audits = [
        mapping(value, f"task audit {index}")
        for index, value in enumerate(sequence(summary.get("task_audits"), "task audits"))
    ]
    require(len(audits) == len(expected_instance_ids), "task audit coverage changed")
    global_request_count = 0
    for index, (instance_id, audit) in enumerate(
        zip(expected_instance_ids, audits, strict=True)
    ):
        cohort_index, cohort_id, campaign_hash, _ = task_binding[instance_id]
        request_count = positive_integer(audit.get("request_count"), "task request count")
        global_request_count += request_count
        require(
            audit.get("instance_id") == instance_id
            and audit.get("selection_index") == index
            and audit.get("cohort_id") == cohort_id
            and audit.get("campaign_sha256") == campaign_hash
            and audit.get("probeable_request_indices") == probeable_by_task[instance_id]
            and audit.get("selected_request_indices") == observed_by_task[instance_id]
            and audit.get("selected_checkpoint_count")
            == len(observed_by_task[instance_id]),
            f"task audit binding changed: {instance_id}",
        )
        require(
            cohort_index == expected_cohort_ids.index(cohort_id),
            f"task audit cohort index changed: {instance_id}",
        )
    require(
        summary.get("global_request_count") == global_request_count,
        "summary global request count changed",
    )
    return {
        "prompt_count": len(prompts),
        "task_count": len(expected_instance_ids),
        "cohort_count": len(cohort_rows),
        "prompt_bundle_sha256": sha256_file(prompts_path),
        "summary_sha256": sha256_file(summary_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--image-config", type=Path, default=DEFAULT_IMAGE_CONFIG)
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--git-bin", default="git")
    parser.add_argument(
        "--validate-bundle",
        nargs=2,
        type=Path,
        metavar=("PROMPTS", "SUMMARY"),
        help="validate a materialized all-probeable bundle without Docker access",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cohort_path = args.cohort.expanduser().resolve(strict=True)
    image_config_path = args.image_config.expanduser().resolve(strict=True)
    cohort, image_config, campaigns = load_and_validate_bindings(
        cohort_path, image_config_path
    )
    if args.validate_bundle is not None:
        prompts_path, summary_path = (
            path.expanduser().absolute() for path in args.validate_bundle
        )
        result = validate_materialized_bundle(
            cohort,
            campaigns,
            cohort_path=cohort_path,
            prompts_path=prompts_path,
            summary_path=summary_path,
        )
        print(
            "validated frozen all-probeable bundle: "
            f"tasks={result['task_count']}, prompts={result['prompt_count']}, "
            f"sha256={result['prompt_bundle_sha256']}"
        )
        return 0
    dataset = mapping(cohort.get("dataset"), "cohort dataset")
    official_ids = load_official_instance_ids(dataset)
    cached_tags = list_cached_image_tags(docker_bin=args.docker_bin)
    verified_cached_tags = {
        instance_id: tag
        for instance_id, tag in cached_tags.items()
        if instance_id in official_ids
    }
    require(bool(verified_cached_tags), "Docker has no cached SWE-Verified images")
    inspected = inspect_images(
        list(verified_cached_tags.values()), docker_bin=args.docker_bin
    )
    cached_amd64_ids = frozenset(
        instance_id
        for instance_id, tag in verified_cached_tags.items()
        if inspected[tag].get("Architecture") == "amd64"
    )
    selection = mapping(cohort.get("selection"), "cohort selection")
    prior_used_ids = prior_used_instance_ids(
        selection, official_ids=official_ids, git_bin=args.git_bin
    )
    reproduced = reproduce_selection(
        selection,
        official_ids=official_ids,
        cached_amd64_ids=cached_amd64_ids,
        prior_used_ids=prior_used_ids,
    )
    configured_batches = tuple(
        tuple(sequence(campaign.get("instance_ids"), "campaign instances"))
        for campaign in campaigns
    )
    require(
        configured_batches == reproduced.batches,
        "frozen campaign order differs from deterministic fresh-state-v1 selection",
    )
    selected_ids = [value for batch in reproduced.batches for value in batch]
    validate_image_pins(
        image_config,
        selected_ids=selected_ids,
        cached_tags=cached_tags,
        inspected=inspected,
    )
    cached_verified_count = len(official_ids & frozenset(cached_tags))
    cached_exclusion_count = len((official_ids & cached_amd64_ids) & prior_used_ids)
    print(
        "validated fresh-state-v1 cohort: "
        f"official={len(official_ids)}, cached_verified={cached_verified_count}, "
        f"prior_used_cached={cached_exclusion_count}, candidates={len(reproduced.candidates)}, "
        f"repositories={len(reproduced.eligible_repositories)}, "
        f"batches={len(reproduced.batches[0])}+{len(reproduced.batches[1])}, "
        f"image_pins={len(selected_ids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
