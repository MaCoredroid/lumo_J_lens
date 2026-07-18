#!/usr/bin/env python3
"""Evaluate behavioral J-lens probes with repository-held-out calibration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/swe_behavioral_readout_protocol.json"
PROTOCOL_KIND = "swe_verified_behavioral_readout_protocol"
PROMPT_KIND = "swe_verified_behavioral_probe"
REPORT_SCHEMA_VERSION = 3
FIXED_LAYER_BAND = tuple(range(24, 48))
ALL_SOURCE_LAYERS = tuple(range(63))
METHODS = (
    "public_jacobian",
    "nf4_jacobian",
    "native_jacobian",
    "ordinary_logit",
)
REPORT_LABELS = ("public", "nf4", "native")
MAX_CHECKPOINTS = 8
RESIDUAL_MANIFEST_ALGORITHM = (
    "SHA-256 over length-prefixed canonical layer/shape/dtype/"
    "token-position/byte-count headers and logical row-major FP32 bytes"
)


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


def sha256_string(value: Any, label: str) -> str:
    result = nonempty_string(value, label)
    require(
        len(result) == 64 and all(character in "0123456789abcdef" for character in result),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return result


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return value


def finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    )


def materialized_json_sha256(value: Any) -> str:
    return sha256_bytes(
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
            "ascii"
        )
    )


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


def logmeanexp(values: Sequence[float]) -> float:
    require(bool(values), "logmeanexp input is empty")
    maximum = max(values)
    return maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in values) / len(values)
    )


def softmax(values: Sequence[float]) -> list[float]:
    require(bool(values), "softmax input is empty")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = math.fsum(weights)
    require(total > 0.0 and math.isfinite(total), "softmax normalization failed")
    return [value / total for value in weights]


def percentile(values: Sequence[float], probability: float) -> float:
    require(bool(values), "percentile input is empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def uniform_request_indices(count: int, *, limit: int = MAX_CHECKPOINTS) -> list[int]:
    integer(count, "request count")
    integer(limit, "checkpoint limit", minimum=1)
    if count <= limit:
        return list(range(1, count + 1))
    return [
        1 + math.floor(index * (count - 1) / (limit - 1))
        for index in range(limit)
    ]


def _validate_classes(
    value: Any, *, label: str, vocabulary_size: int, seen: set[int]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    class_ids: set[str] = set()
    sizes: set[int] = set()
    for raw_record in sequence(value, label):
        record = mapping(raw_record, label)
        class_id = nonempty_string(record.get("id"), f"{label} ID")
        require(class_id not in class_ids, f"duplicate {label} ID: {class_id}")
        class_ids.add(class_id)
        tokens: list[dict[str, Any]] = []
        for raw_token in sequence(record.get("tokens"), f"{label}.{class_id}.tokens"):
            token = mapping(raw_token, "class token")
            text = nonempty_string(token.get("text"), "class token text")
            token_id = integer(token.get("token_id"), "class token ID")
            require(
                text.startswith(" ")
                and not text.startswith("  ")
                and token_id < vocabulary_size
                and token_id not in seen,
                f"invalid or overlapping class token: {text!r}/{token_id}",
            )
            seen.add(token_id)
            tokens.append({"text": text, "token_id": token_id})
        require(bool(tokens), f"{label}.{class_id} has no tokens")
        sizes.add(len(tokens))
        result.append({"id": class_id, "tokens": tokens})
    require(bool(result) and len(sizes) == 1, f"{label} is empty or unbalanced")
    return result


def validate_protocol(protocol: Mapping[str, Any], *, protocol_sha256: str) -> dict[str, Any]:
    require(protocol.get("schema_version") == 1, "readout protocol schema mismatch")
    require(protocol.get("kind") == PROTOCOL_KIND, "readout protocol kind mismatch")
    require(
        protocol.get("analysis_version") == "task-held-out-v1"
        and protocol.get("lens_outputs_used_for_selection_or_labels") is False,
        "readout protocol version/selection contract mismatch",
    )
    pins = mapping(protocol.get("pins"), "protocol pins")
    model = mapping(pins.get("model"), "model pin")
    tokenizer = mapping(pins.get("tokenizer"), "tokenizer pin")
    require(
        model.get("repo_id") == "nvidia/Qwen3.6-27B-NVFP4"
        and model.get("revision")
        == "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
        and len(nonempty_string(model.get("config_sha256"), "model config SHA")) == 64
        and len(nonempty_string(model.get("index_sha256"), "model index SHA")) == 64,
        "model pin mismatch",
    )
    vocabulary_size = integer(tokenizer.get("vocabulary_size"), "tokenizer size", minimum=1)
    logit_size = integer(tokenizer.get("logit_vocabulary_size"), "logit size", minimum=1)
    require(logit_size >= vocabulary_size, "logit vocabulary is smaller than tokenizer")
    action_pin = mapping(pins.get("action_protocol"), "action protocol pin")
    action_path = (ROOT / nonempty_string(action_pin.get("path"), "action protocol path")).resolve(
        strict=True
    )
    require(action_path.is_relative_to(ROOT), "action protocol path escapes repository")
    action_sha = sha256_file(action_path)
    require(
        action_sha == action_pin.get("sha256"), "pinned action protocol hash mismatch"
    )
    action_value = mapping(json.loads(action_path.read_bytes()), "action protocol")
    band = mapping(protocol.get("fixed_layer_band"), "fixed layer band")
    require(
        band.get("start") == 24
        and band.get("end") == 47
        and band.get("end_inclusive") is True
        and band.get("layers") == list(FIXED_LAYER_BAND),
        "fixed layer band must be exactly layers 24 through 47",
    )
    reduction = mapping(protocol.get("class_score_reduction"), "score reduction")
    require(
        reduction
        == {
            "within_class": "logmeanexp_over_declared_token_logits",
            "across_layers": "arithmetic_mean_over_fixed_layers_24_through_47",
            "layer_selection": "none",
        },
        "class score reduction changed",
    )
    seen: set[int] = set()
    actions = _validate_classes(
        protocol.get("action_classes"),
        label="action classes",
        vocabulary_size=vocabulary_size,
        seen=seen,
    )
    outcomes = _validate_classes(
        protocol.get("outcome_classes"),
        label="outcome classes",
        vocabulary_size=vocabulary_size,
        seen=seen,
    )
    require(
        action_value.get("action_classes") == actions
        and action_value.get("outcome_classes") == outcomes,
        "readout/action protocol vocabularies differ",
    )
    primary = mapping(protocol.get("primary_cohort"), "primary cohort")
    require(
        primary
        == {
            "cohort": "uniform_probeable_request_index",
            "algorithm": "uniform_probeable_request_indices_v1",
            "primary_for_action_evaluation": True,
            "independent_of_next_action_label": True,
            "label_conditioned_lifecycle_checkpoints_are_primary": False,
        },
        "primary cohort contract changed",
    )
    prompt_context = mapping(protocol.get("prompt_context"), "prompt context contract")
    require(
        prompt_context
        == {
            "max_model_len": 65536,
            "reserved_generation_tokens": 1,
            "maximum_prompt_tokens": 65535,
            "over_limit_prompts_forbidden": True,
        },
        "prompt context contract changed",
    )
    crossfit = mapping(protocol.get("cross_fitting"), "cross-fitting contract")
    require(
        crossfit.get("algorithm")
        == "leave_one_repository_out_first_supported_calibration_subset_v1"
        and crossfit.get("group") == "repository_then_task"
        and crossfit.get("evaluation_group") == "repository"
        and crossfit.get("held_out_labels_used_for_split_or_fit") is False,
        "cross-fitting isolation contract changed",
    )
    calibrator = mapping(crossfit.get("calibrator"), "calibrator")
    require(
        calibrator.get("kind")
        == "per_class_additive_bias_then_scalar_temperature"
        and calibrator.get("bias_reference_class") == "last_declared_class"
        and calibrator.get("layer_selection") == "none",
        "calibrator form changed",
    )
    numerical = mapping(
        protocol.get("numerical_certification"), "numerical certification"
    )
    expected_numerical = {
        "final_norm_max_abs_tolerance": 0.125,
        "final_norm_rms_tolerance": 0.006,
        "final_logits_max_abs_tolerance": 0.0625,
        "final_logits_rms_tolerance": 0.01,
        "top_k_prefix": 5,
        "residual_tensor_count": 64,
        "residual_d_model": 5120,
        "residual_dtype_bytes": 4,
    }
    require(dict(numerical) == expected_numerical, "numerical certification changed")
    bootstrap = mapping(protocol.get("bootstrap"), "bootstrap contract")
    minimum_valid_fraction = finite(
        bootstrap.get("minimum_valid_fraction"), "bootstrap minimum valid fraction"
    )
    require(
        bootstrap.get("algorithm")
        == "hierarchical_repository_then_task_percentile_v1"
        and bootstrap.get("row_resampling_forbidden") is True,
        "bootstrap must cluster by repository and task",
    )
    require(
        0.0 < minimum_valid_fraction <= 1.0,
        "bootstrap minimum valid fraction must be in (0, 1]",
    )
    return {
        "sha256": protocol_sha256,
        "model": dict(model),
        "tokenizer": dict(tokenizer),
        "vocabulary_size": vocabulary_size,
        "logit_vocabulary_size": logit_size,
        "action_protocol_sha256": action_sha,
        "action_classes": actions,
        "outcome_classes": outcomes,
        "action_ids": [record["id"] for record in actions],
        "outcome_ids": [record["id"] for record in outcomes],
        "fixed_layers": list(FIXED_LAYER_BAND),
        "lens_pins": dict(mapping(pins.get("lenses"), "lens pins")),
        "primary_cohort": dict(primary),
        "prompt_context": dict(prompt_context),
        "crossfit": dict(crossfit),
        "bootstrap": dict(bootstrap),
        "numerical_certification": dict(numerical),
        "future_target": dict(mapping(protocol.get("future_target"), "future target")),
        "official_outcome": dict(
            mapping(protocol.get("official_outcome"), "official outcome")
        ),
        "development": dict(
            mapping(protocol.get("development_contract"), "development contract")
        ),
    }


def _forms(
    value: Any,
    *,
    label: str,
    logit_size: int,
    token_text: dict[int, str],
    seen: set[int],
) -> list[dict[str, Any]]:
    forms: list[dict[str, Any]] = []
    local_seen: set[int] = set()
    for raw_form in sequence(value, label):
        form = mapping(raw_form, label)
        text = nonempty_string(form.get("text"), f"{label} text")
        token_id = integer(form.get("token_id"), f"{label} token ID")
        require(
            token_id < logit_size and token_id not in local_seen,
            f"{label} token ID is invalid or duplicated: {token_id}",
        )
        previous = token_text.setdefault(token_id, text)
        require(previous == text, f"token {token_id} has conflicting text")
        local_seen.add(token_id)
        seen.add(token_id)
        forms.append({**dict(form), "text": text, "token_id": token_id})
    require(bool(forms), f"{label} must not be empty")
    return forms


def _aliases(value: Any, label: str) -> list[str]:
    aliases = [nonempty_string(item, label) for item in sequence(value, label)]
    require(aliases == list(dict.fromkeys(aliases)), f"{label} contains duplicates")
    return aliases


def _excluded_foil_ids(value: Any) -> list[str]:
    result: list[str] = []
    for item in sequence(value, "excluded foils"):
        if isinstance(item, str):
            foil_id = nonempty_string(item, "excluded foil ID")
        else:
            record = mapping(item, "excluded foil")
            foil_id = nonempty_string(
                record.get("foil_id", record.get("id")), "excluded foil ID"
            )
        require(foil_id not in result, "excluded foil IDs contain duplicates")
        result.append(foil_id)
    return result


def _prompt_payload_hash(prompt: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(prompt))
    metadata = mapping(payload.get("metadata"), "prompt metadata")
    provenance = mapping(metadata.get("provenance"), "prompt provenance")
    require(
        "prompt_record_payload_sha256" in provenance,
        "prompt payload self-hash field is missing",
    )
    del provenance["prompt_record_payload_sha256"]
    return sha256_json(payload)


def _cohort_metadata(value: Any, label: str) -> dict[str, Any]:
    cohort = mapping(value, label)
    expected_fields = {
        "index",
        "id",
        "campaign_sha256",
        "source_run_id",
        "source_run_label",
        "source_summary_sha256",
        "source_task_count",
        "source_task_instance_ids",
        "source_global_request_count",
        "source_prompt_count",
        "global_request_offset",
        "task_offset",
        "cohort_manifest_sha256",
    }
    require(set(cohort) == expected_fields, f"{label} fields changed")
    index = integer(cohort.get("index"), f"{label} index")
    cohort_id = nonempty_string(cohort.get("id"), f"{label} ID")
    require(
        cohort_id[0].isalpha()
        and cohort_id[0].isascii()
        and all(
            character.isascii() and (character.islower() or character.isdigit() or character in "_-")
            for character in cohort_id
        ),
        f"{label} ID is unsafe",
    )
    campaign_sha256 = sha256_string(cohort.get("campaign_sha256"), f"{label} campaign SHA")
    source_run_id = nonempty_string(cohort.get("source_run_id"), f"{label} source run ID")
    require(
        source_run_id.startswith("run-")
        and len(source_run_id) == 24
        and all(character in "0123456789abcdef" for character in source_run_id[4:]),
        f"{label} source run ID is invalid",
    )
    source_run_label = nonempty_string(
        cohort.get("source_run_label"), f"{label} source run label"
    )
    source_summary_sha256 = sha256_string(
        cohort.get("source_summary_sha256"), f"{label} source summary SHA"
    )
    source_task_count = integer(
        cohort.get("source_task_count"), f"{label} source task count", minimum=1
    )
    source_task_ids = [
        nonempty_string(item, f"{label} source task ID")
        for item in sequence(
            cohort.get("source_task_instance_ids"), f"{label} source task IDs"
        )
    ]
    require(
        len(source_task_ids) == source_task_count
        and len(source_task_ids) == len(set(source_task_ids)),
        f"{label} source task coverage is invalid",
    )
    return {
        "index": index,
        "id": cohort_id,
        "campaign_sha256": campaign_sha256,
        "source_run_id": source_run_id,
        "source_run_label": source_run_label,
        "source_summary_sha256": source_summary_sha256,
        "source_task_count": source_task_count,
        "source_task_instance_ids": source_task_ids,
        "source_global_request_count": integer(
            cohort.get("source_global_request_count"),
            f"{label} source global request count",
        ),
        "source_prompt_count": integer(
            cohort.get("source_prompt_count"), f"{label} source prompt count", minimum=1
        ),
        "global_request_offset": integer(
            cohort.get("global_request_offset"), f"{label} global request offset"
        ),
        "task_offset": integer(cohort.get("task_offset"), f"{label} task offset"),
        "cohort_manifest_sha256": sha256_string(
            cohort.get("cohort_manifest_sha256"), f"{label} manifest SHA"
        ),
    }


def validate_prompt_bundle(
    prompts_value: Any, *, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    prompts = sequence(prompts_value, "behavioral prompt bundle")
    require(bool(prompts), "behavioral prompt bundle is empty")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    campaign_sha256s: list[str] = []
    task_contracts: dict[str, dict[str, Any]] = {}
    global_request_indices: set[tuple[str, int]] = set()
    cohort_presence: bool | None = None
    cohort_contracts: dict[int, dict[str, Any]] = {}
    lifecycle_flags: list[dict[str, Any]] = []
    fixed_tokens = [
        token
        for family in (protocol["action_classes"], protocol["outcome_classes"])
        for record in family
        for token in record["tokens"]
    ]
    for index, raw_prompt in enumerate(prompts):
        prompt = mapping(raw_prompt, f"prompt[{index}]")
        require(
            set(prompt) == {"id", "text", "token_ids", "score_token_ids", "metadata"},
            f"prompt[{index}] fields changed",
        )
        prompt_id = nonempty_string(prompt.get("id"), "prompt ID")
        require(prompt_id not in ids, f"duplicate prompt ID: {prompt_id}")
        ids.add(prompt_id)
        text = nonempty_string(prompt.get("text"), f"{prompt_id}.text")
        token_ids = sequence(prompt.get("token_ids"), f"{prompt_id}.token_ids")
        require(
            bool(token_ids)
            and all(isinstance(item, int) and not isinstance(item, bool) for item in token_ids),
            f"{prompt_id} has invalid token IDs",
        )
        require(
            len(token_ids) <= protocol["prompt_context"]["maximum_prompt_tokens"],
            f"{prompt_id} exceeds the replayable prompt-token limit",
        )
        score_ids = sequence(prompt.get("score_token_ids"), f"{prompt_id}.score IDs")
        require(
            bool(score_ids)
            and len(score_ids) == len(set(score_ids))
            and all(
                isinstance(item, int)
                and not isinstance(item, bool)
                and 0 <= item < protocol["logit_vocabulary_size"]
                for item in score_ids
            ),
            f"{prompt_id} has invalid scored token IDs",
        )
        metadata = mapping(prompt.get("metadata"), f"{prompt_id}.metadata")
        require(
            metadata.get("kind") == PROMPT_KIND
            and metadata.get("schema_version") == 1,
            f"{prompt_id} behavioral metadata kind/schema mismatch",
        )
        current_campaign = sha256_string(metadata.get("campaign_sha256"), "campaign SHA")
        if current_campaign not in campaign_sha256s:
            campaign_sha256s.append(current_campaign)
        raw_cohort = metadata.get("cohort")
        has_cohort = raw_cohort is not None
        if cohort_presence is None:
            cohort_presence = has_cohort
        require(cohort_presence is has_cohort, "cohort metadata is partial across prompts")
        cohort: dict[str, Any] | None = None
        if has_cohort:
            cohort = _cohort_metadata(raw_cohort, f"{prompt_id}.cohort")
            require(
                cohort["campaign_sha256"] == current_campaign,
                f"{prompt_id} cohort/campaign binding mismatch",
            )
            previous_cohort = cohort_contracts.setdefault(cohort["index"], cohort)
            require(previous_cohort == cohort, f"{prompt_id} cohort metadata changed")
        require(
            metadata.get("action_protocol_sha256")
            == protocol["action_protocol_sha256"],
            f"{prompt_id} action protocol hash mismatch",
        )
        task = mapping(metadata.get("task"), f"{prompt_id}.task")
        task_id = nonempty_string(task.get("instance_id"), "task instance ID")
        repo = nonempty_string(task.get("repo"), "task repository")
        task_record = {
            "campaign_sha256": current_campaign,
            "selection_index": integer(task.get("selection_index"), "task selection index"),
            "instance_id": task_id,
            "repo": repo,
            "base_commit": nonempty_string(task.get("base_commit"), "task base commit"),
            "request_count": integer(task.get("request_count"), "task request count", minimum=1),
        }
        probeable_request_indices = [
            integer(value, "probeable request index", minimum=1)
            for value in sequence(
                task.get("probeable_request_indices"), "probeable request indices"
            )
        ]
        excluded_request_indices = [
            integer(value, "excluded request index", minimum=1)
            for value in sequence(
                task.get("excluded_request_indices"), "excluded request indices"
            )
        ]
        probeable_request_count = integer(
            task.get("probeable_request_count"),
            "probeable request count",
            minimum=1,
        )
        require(
            bool(probeable_request_indices)
            and probeable_request_indices == sorted(set(probeable_request_indices))
            and excluded_request_indices == sorted(set(excluded_request_indices))
            and set(probeable_request_indices).isdisjoint(excluded_request_indices)
            and sorted(probeable_request_indices + excluded_request_indices)
            == list(range(1, task_record["request_count"] + 1))
            and probeable_request_count == len(probeable_request_indices),
            f"{prompt_id} probeable/excluded request partition is invalid",
        )
        task_record.update(
            {
                "probeable_request_count": probeable_request_count,
                "probeable_request_indices": probeable_request_indices,
                "excluded_request_indices": excluded_request_indices,
            }
        )
        if cohort is None:
            require(
                "source_selection_index" not in task,
                f"{prompt_id} has combined task metadata without a cohort",
            )
        else:
            source_task_index = integer(
                task.get("source_selection_index"), "source task selection index"
            )
            require(
                source_task_index < cohort["source_task_count"]
                and task_record["selection_index"]
                == source_task_index + cohort["task_offset"]
                and cohort["source_task_instance_ids"][source_task_index] == task_id,
                f"{prompt_id} combined task offset/binding mismatch",
            )
        previous_task = task_contracts.setdefault(
            task_id,
            {"task": task_record, "official": None, "primary": [], "all_prompt_ids": []},
        )
        require(previous_task["task"] == task_record, f"{task_id} task metadata changed")
        previous_task["all_prompt_ids"].append(prompt_id)
        selection = mapping(metadata.get("selection"), f"{prompt_id}.selection")
        task_request_index = integer(
            selection.get("task_request_index"), "task request index", minimum=1
        )
        require(
            task_request_index <= task_record["request_count"],
            f"{prompt_id} request index exceeds task request count",
        )
        global_request_index = integer(
            selection.get("global_request_index"), "global request index", minimum=1
        )
        if cohort is None:
            require(
                "source_global_request_index" not in selection,
                f"{prompt_id} has a source global index without a cohort",
            )
            global_request_key = (current_campaign, global_request_index)
        else:
            source_global_request_index = integer(
                selection.get("source_global_request_index"),
                "source global request index",
                minimum=1,
            )
            require(
                source_global_request_index <= cohort["source_global_request_count"]
                and global_request_index
                == source_global_request_index + cohort["global_request_offset"],
                f"{prompt_id} combined global request offset mismatch",
            )
            global_request_key = ("combined", global_request_index)
        require(global_request_key not in global_request_indices, "duplicate global request index")
        global_request_indices.add(global_request_key)
        ordinal = integer(selection.get("checkpoint_ordinal"), "checkpoint ordinal")
        checkpoint_count = integer(
            selection.get("checkpoint_count"), "checkpoint count", minimum=1
        )
        max_checkpoints = integer(
            selection.get("max_checkpoints"), "maximum checkpoints", minimum=1
        )
        selection_probeable = [
            integer(value, "selection probeable request index", minimum=1)
            for value in sequence(
                selection.get("probeable_request_indices"),
                "selection probeable request indices",
            )
        ]
        selection_excluded = [
            integer(value, "selection excluded request index", minimum=1)
            for value in sequence(
                selection.get("excluded_request_indices"),
                "selection excluded request indices",
            )
        ]
        candidate_ordinal = integer(
            selection.get("candidate_ordinal"), "candidate ordinal", minimum=1
        )
        candidate_ordinal_base = integer(
            selection.get("candidate_ordinal_base"),
            "candidate ordinal base",
            minimum=1,
        )
        candidate_count = integer(
            selection.get("candidate_count"), "candidate count", minimum=1
        )
        max_prompt_tokens = integer(
            selection.get("max_prompt_tokens"),
            "maximum prompt tokens",
            minimum=1,
        )
        require(
            max_prompt_tokens == protocol["prompt_context"]["maximum_prompt_tokens"]
            and candidate_ordinal_base == 1
            and candidate_count == len(probeable_request_indices)
            and selection_probeable == probeable_request_indices
            and selection_excluded == excluded_request_indices
            and candidate_ordinal <= len(probeable_request_indices)
            and probeable_request_indices[candidate_ordinal - 1] == task_request_index,
            f"{prompt_id} probeable candidate binding mismatch",
        )
        primary = bool(
            selection.get("cohort") == protocol["primary_cohort"]["cohort"]
            and selection.get("algorithm") == protocol["primary_cohort"]["algorithm"]
            and max_checkpoints == MAX_CHECKPOINTS
            and selection.get("primary_for_action_evaluation") is True
            and selection.get("independent_of_next_action_label") is True
        )
        if selection.get("primary_for_action_evaluation") is True:
            require(primary, f"{prompt_id} claims primary status with a nonuniform cohort")
        if primary:
            previous_task["primary"].append(
                {
                    "prompt_id": prompt_id,
                    "request_index": task_request_index,
                    "ordinal": ordinal,
                    "checkpoint_count": checkpoint_count,
                    "candidate_ordinal": candidate_ordinal,
                }
            )
        else:
            lifecycle_flags.append(
                {
                    "prompt_id": prompt_id,
                    "instance_id": task_id,
                    "cohort": selection.get("cohort"),
                    "algorithm": selection.get("algorithm"),
                    "independent_of_next_action_label": selection.get(
                        "independent_of_next_action_label"
                    ),
                    "flag": "label_conditioned_or_nonuniform_lifecycle_checkpoint",
                }
            )
        labels = mapping(metadata.get("labels"), f"{prompt_id}.labels")
        action = mapping(labels.get("action"), "action label")
        action_status = action.get("status")
        action_class = action.get("class_id")
        require(
            (action_status == "available" and action_class in protocol["action_ids"])
            or (action_status == "missing" and action_class is None),
            f"{prompt_id} action label/status mismatch",
        )
        nonempty_string(action.get("derivation"), "action derivation")
        tool_execution = mapping(labels.get("tool_execution"), "tool-execution label")
        require(
            (
                tool_execution.get("status") == "available"
                and tool_execution.get("class_id") in protocol["outcome_ids"]
            )
            or (
                tool_execution.get("status") == "not_applicable"
                and tool_execution.get("class_id") is None
            )
            or (
                tool_execution.get("status") == "missing"
                and tool_execution.get("class_id") is None
            ),
            f"{prompt_id} tool-execution status/class mismatch",
        )
        if tool_execution.get("status") == "missing":
            require(
                tool_execution.get("class_id") is None,
                f"{prompt_id} missing tool execution was imputed",
            )
        require(
            tool_execution.get("status") in {"available", "not_applicable", "missing"},
            f"{prompt_id} tool-execution status is invalid",
        )
        nonempty_string(tool_execution.get("derivation"), "tool-execution derivation")
        validation = mapping(labels.get("validation"), "validation label")
        require(
            (
                validation.get("status") == "available"
                and validation.get("class_id") in protocol["outcome_ids"]
            )
            or (
                validation.get("status") == "not_applicable"
                and validation.get("class_id") is None
            )
            or (
                validation.get("status") == "missing"
                and validation.get("class_id") is None
            ),
            f"{prompt_id} validation label/status mismatch",
        )
        if validation.get("status") == "missing":
            require(
                validation.get("class_id") is None,
                f"{prompt_id} missing validation was imputed",
            )
        require(
            validation.get("status") in {"available", "not_applicable", "missing"},
            f"{prompt_id} validation status is invalid",
        )
        nonempty_string(validation.get("derivation"), "validation derivation")
        terminal = mapping(labels.get("terminal"), "terminal label")
        require(
            terminal.get("finish_reason") in {None, "tool_calls", "stop", "length"}
            and isinstance(terminal.get("is_terminal"), bool)
            and isinstance(terminal.get("is_terminal_completion"), bool)
            and isinstance(terminal.get("is_episode_endpoint"), bool)
            and isinstance(terminal.get("is_probeable_endpoint"), bool)
            and terminal.get("is_terminal") is terminal.get("is_terminal_completion")
            and terminal.get("is_terminal_completion")
            is (terminal.get("finish_reason") == "stop")
            and terminal.get("is_episode_endpoint")
            is (task_request_index == task_record["request_count"])
            and terminal.get("is_probeable_endpoint")
            is (task_request_index == probeable_request_indices[-1]),
            f"{prompt_id} terminal label mismatch",
        )
        official = mapping(labels.get("official_outcome"), "official outcome")
        official_status = official.get("status")
        official_class = official.get("class_id")
        require(
            (official_status == "available" and official_class in protocol["outcome_ids"])
            or (official_status == "missing" and official_class is None),
            f"{prompt_id} official outcome status/class mismatch",
        )
        verdict = official.get("verdict")
        require(
            (official_status == "available" and isinstance(verdict, str) and bool(verdict))
            or (
                official_status == "missing"
                and (verdict is None or (isinstance(verdict, str) and bool(verdict)))
            ),
            f"{prompt_id} official verdict/status mismatch",
        )
        official_derivation = nonempty_string(
            official.get("derivation"), "official outcome derivation"
        )
        if official_status == "available":
            require(
                verdict in {"resolved", "unresolved", "error", "empty"}
                and official_class
                == ("success" if verdict == "resolved" else "failure")
                and official_derivation == "bound_official_swe_bench_aggregate",
                f"{prompt_id} official aggregate label is inconsistent",
            )
            nonempty_string(
                official.get("official_outcomes_path"), "official outcomes path"
            )
            sha256_string(
                official.get("official_outcomes_sha256"), "official outcomes SHA"
            )
            sha256_string(
                official.get("outcome_record_sha256"), "official outcome record SHA"
            )
        else:
            if verdict is None:
                require(
                    official_derivation == "official_outcome_aggregate_absent"
                    and official.get("official_outcomes_path") is None
                    and official.get("official_outcomes_sha256") is None
                    and official.get("outcome_record_sha256") is None,
                    f"{prompt_id} absent official aggregate evidence is inconsistent",
                )
            else:
                require(
                    verdict in {"error", "empty"}
                    and official_derivation
                    == "official_nonbinary_infrastructure_or_empty_outcome",
                    f"{prompt_id} missing official outcome was imputed from a nonbinary verdict",
                )
                nonempty_string(
                    official.get("official_outcomes_path"), "official outcomes path"
                )
                sha256_string(
                    official.get("official_outcomes_sha256"), "official outcomes SHA"
                )
                sha256_string(
                    official.get("outcome_record_sha256"),
                    "official outcome record SHA",
                )
        if previous_task["official"] is None:
            previous_task["official"] = dict(official)
        require(
            previous_task["official"] == dict(official),
            f"{task_id} repeats inconsistent official outcomes",
        )
        token_text: dict[int, str] = {}
        dynamic_seen: set[int] = set()
        targets: list[dict[str, Any]] = []
        for target_index, raw_target in enumerate(
            sequence(metadata.get("targets"), f"{prompt_id}.targets")
        ):
            target = mapping(raw_target, f"target[{target_index}]")
            target_id = nonempty_string(target.get("id"), "target ID")
            require(
                target_id not in {item["id"] for item in targets},
                f"duplicate target ID: {target_id}",
            )
            target_forms = _forms(
                target.get("forms"),
                label=f"{target_id} forms",
                logit_size=protocol["logit_vocabulary_size"],
                token_text=token_text,
                seen=dynamic_seen,
            )
            aliases = _aliases(target.get("aliases"), f"{target_id} aliases")
            require(bool(aliases), f"{target_id} has no aliases")
            future_support = target.get("future_support")
            require(
                isinstance(future_support, dict)
                and bool(future_support)
                and future_support.get("benchmark_gold_used") is False
                and future_support.get("lens_output_used") is False,
                f"{target_id} future-support evidence used a forbidden source",
            )
            target_task_id = nonempty_string(
                target.get("task_instance_id"), "target task instance ID"
            )
            require(target_task_id == task_id, f"{target_id} belongs to another task")
            foils: list[dict[str, Any]] = []
            for foil_index, raw_foil in enumerate(sequence(target.get("foils"), "foils")):
                foil = mapping(raw_foil, f"foil[{foil_index}]")
                foil_id = nonempty_string(foil.get("id"), "foil ID")
                require(
                    foil_id not in {item["id"] for item in foils},
                    f"duplicate foil ID: {foil_id}",
                )
                foil_forms = _forms(
                    foil.get("forms"),
                    label=f"{foil_id} forms",
                    logit_size=protocol["logit_vocabulary_size"],
                    token_text=token_text,
                    seen=dynamic_seen,
                )
                foils.append(
                    {
                        **dict(foil),
                        "id": foil_id,
                        "task_instance_id": nonempty_string(
                            foil.get("task_instance_id"), "foil task instance ID"
                        ),
                        "target": nonempty_string(foil.get("target"), "foil target"),
                        "kind": nonempty_string(foil.get("kind"), "foil kind"),
                        "forms": foil_forms,
                        "aliases": _aliases(foil.get("aliases"), f"{foil_id} aliases"),
                    }
                )
            require(bool(foils), f"{target_id} has no same-task foil candidates")
            require(
                all(
                    foil["task_instance_id"] == task_id
                    and foil["kind"] == target.get("kind")
                    for foil in foils
                ),
                f"{target_id} has a cross-task foil or kind-mismatched foil",
            )
            targets.append(
                {
                    **dict(target),
                    "id": target_id,
                    "target": nonempty_string(target.get("target"), "target text"),
                    "kind": nonempty_string(target.get("kind"), "target kind"),
                    "forms": target_forms,
                    "aliases": aliases,
                    "foils": foils,
                }
            )
        eligibility_by_id: dict[str, dict[str, Any]] = {}
        for raw_eligibility in sequence(
            metadata.get("target_eligibility"), "target eligibility"
        ):
            eligibility = mapping(raw_eligibility, "target eligibility")
            target_id = nonempty_string(eligibility.get("target_id"), "eligible target ID")
            require(target_id not in eligibility_by_id, "duplicate target eligibility")
            target_exposed = eligibility.get("target_exposed")
            require(isinstance(target_exposed, bool), "target exposure flag is invalid")
            retained = [
                nonempty_string(item, "retained foil ID")
                for item in sequence(
                    eligibility.get("retained_hidden_foil_ids"), "retained hidden foils"
                )
            ]
            require(retained == list(dict.fromkeys(retained)), "retained foil IDs duplicate")
            excluded = _excluded_foil_ids(eligibility.get("excluded_foils"))
            status = eligibility.get("status")
            require(
                status in {"eligible", "target_exposed", "insufficient_hidden_foils"},
                "target eligibility status is invalid",
            )
            if status == "eligible":
                require(
                    target_exposed is False and bool(retained),
                    "eligible target is exposed or has no hidden foils",
                )
            elif status == "target_exposed":
                require(target_exposed is True, "target_exposed status has a false flag")
            else:
                require(
                    target_exposed is False and not retained,
                    "insufficient-hidden-foils status is inconsistent",
                )
            eligibility_by_id[target_id] = {
                **dict(eligibility),
                "retained_hidden_foil_ids": retained,
                "excluded_foil_ids": excluded,
            }
        require(
            set(eligibility_by_id) == {target["id"] for target in targets},
            "target eligibility does not exactly cover targets",
        )
        for target in targets:
            eligibility = eligibility_by_id[target["id"]]
            foil_by_id = {foil["id"]: foil for foil in target["foils"]}
            retained = eligibility["retained_hidden_foil_ids"]
            excluded = eligibility["excluded_foil_ids"]
            require(
                set(retained).isdisjoint(excluded)
                and set(retained) | set(excluded) == set(foil_by_id),
                f"{target['id']} retained/excluded foil partition is incomplete",
            )
            for foil_id in retained:
                require(
                    foil_by_id[foil_id]["task_instance_id"] == task_id,
                    f"{target['id']} retained a cross-task foil",
                )
            target["eligibility"] = eligibility
        expected_dynamic = [
            form["token_id"]
            for target in targets
            for forms in [target["forms"], *(foil["forms"] for foil in target["foils"])]
            for form in forms
        ]
        fixed_ids = [token["token_id"] for token in fixed_tokens]
        require(
            set(expected_dynamic).isdisjoint(fixed_ids),
            f"{prompt_id} dynamic tokens overlap action/outcome tokens",
        )
        expected_score_ids = list(dict.fromkeys(expected_dynamic + fixed_ids))
        require(score_ids == expected_score_ids, f"{prompt_id} scored vocabulary order mismatch")
        for token in fixed_tokens:
            previous = token_text.setdefault(token["token_id"], token["text"])
            require(previous == token["text"], "fixed token text conflicts")
        provenance = mapping(metadata.get("provenance"), f"{prompt_id}.provenance")
        require(
            provenance.get("rendered_prompt_sha256") == sha256_text(text)
            and provenance.get("token_ids_sha256") == sha256_json(token_ids)
            and provenance.get("prompt_token_count") == len(token_ids),
            f"{prompt_id} prompt text/token provenance mismatch",
        )
        for field in (
            "raw_request_sha256",
            "usage_sha256",
            "usage_record_sha256",
            "runner_metadata_sha256",
            "prompt_record_payload_sha256",
        ):
            value = nonempty_string(provenance.get(field), f"{prompt_id}.{field}")
            require(len(value) == 64, f"{prompt_id}.{field} is malformed")
        for field in (
            "raw_request_path",
            "usage_path",
            "runner_metadata_path",
        ):
            nonempty_string(provenance.get(field), f"{prompt_id}.{field}")
        official_provenance = mapping(
            provenance.get("official_outcomes"), f"{prompt_id}.official outcomes provenance"
        )
        require(
            official_provenance.get("status")
            == (
                "available"
                if official.get("official_outcomes_sha256") is not None
                else "missing"
            )
            and official_provenance.get("path")
            == official.get("official_outcomes_path")
            and official_provenance.get("sha256")
            == official.get("official_outcomes_sha256")
            and official_provenance.get("outcome_record_sha256")
            == official.get("outcome_record_sha256"),
            f"{prompt_id} official outcome provenance mismatch",
        )
        generated_patch_path = provenance.get("generated_patch_path")
        generated_patch_sha = provenance.get("generated_patch_sha256")
        require(
            (generated_patch_path is None and generated_patch_sha is None)
            or (
                isinstance(generated_patch_path, str)
                and bool(generated_patch_path)
                and isinstance(generated_patch_sha, str)
                and len(generated_patch_sha) == 64
            ),
            f"{prompt_id} generated-patch path/hash mismatch",
        )
        combination_value = provenance.get("combination")
        if cohort is None:
            require(
                combination_value is None,
                f"{prompt_id} has combination provenance without a cohort",
            )
        else:
            combination = mapping(combination_value, f"{prompt_id}.combination")
            require(
                set(combination)
                == {
                    "source_prompt_id",
                    "source_prompt_record_payload_sha256",
                    "source_campaign_global_request_index",
                    "combined_global_request_index",
                    "cohort_manifest_sha256",
                },
                f"{prompt_id} combination provenance fields changed",
            )
            nonempty_string(combination.get("source_prompt_id"), "source prompt ID")
            sha256_string(
                combination.get("source_prompt_record_payload_sha256"),
                "source prompt payload SHA",
            )
            require(
                combination.get("source_campaign_global_request_index")
                == selection["source_global_request_index"]
                and combination.get("combined_global_request_index")
                == global_request_index
                and combination.get("cohort_manifest_sha256")
                == cohort["cohort_manifest_sha256"],
                f"{prompt_id} combination provenance binding mismatch",
            )
        require(
            provenance.get("prompt_record_payload_sha256") == _prompt_payload_hash(prompt),
            f"{prompt_id} canonical prompt-record payload hash mismatch",
        )
        result.append(
            {
                "id": prompt_id,
                "text": text,
                "token_ids": list(token_ids),
                "score_token_ids": list(score_ids),
                "metadata": dict(metadata),
                "token_text": token_text,
                "campaign_sha256": current_campaign,
                "cohort": copy.deepcopy(cohort),
                "cohort_id": cohort["id"] if cohort is not None else "single_campaign",
                "task_id": task_id,
                "repo": repo,
                "task_request_index": task_request_index,
                "global_request_index": global_request_index,
                "checkpoint_ordinal": ordinal,
                "checkpoint_count": checkpoint_count,
                "primary": primary,
                "action_label": action_class,
                "action_status": action_status,
                "tool_execution_label": tool_execution["class_id"],
                "validation_label": validation.get("class_id"),
                "official_outcome": official_class,
                "official_outcome_status": official_status,
                "targets": targets,
            }
        )
    require(bool(campaign_sha256s), "campaign SHA was not initialized")
    combined = bool(cohort_presence)
    require(
        combined is (len(campaign_sha256s) > 1),
        "multiple campaigns require complete combined-cohort metadata",
    )
    cohort_rows: list[dict[str, Any]] = []
    selected_task_count = len(task_contracts)
    unprobed_task_ids: list[str] = []
    if combined:
        cohort_rows = [cohort_contracts[index] for index in sorted(cohort_contracts)]
        require(
            [cohort["index"] for cohort in cohort_rows] == list(range(len(cohort_rows)))
            and [cohort["campaign_sha256"] for cohort in cohort_rows]
            == campaign_sha256s
            and len({cohort["id"] for cohort in cohort_rows}) == len(cohort_rows)
            and len({cohort["source_run_id"] for cohort in cohort_rows})
            == len(cohort_rows)
            and len({cohort["cohort_manifest_sha256"] for cohort in cohort_rows}) == 1,
            "combined cohort order/identity is invalid",
        )
        expected_global_offset = 0
        expected_task_offset = 0
        seen_source_tasks: set[str] = set()
        for cohort in cohort_rows:
            require(
                cohort["global_request_offset"] == expected_global_offset
                and cohort["task_offset"] == expected_task_offset,
                f"cohort {cohort['id']} cumulative offsets are invalid",
            )
            source_task_ids = cohort["source_task_instance_ids"]
            require(
                not (set(source_task_ids) & seen_source_tasks),
                "task ID overlaps combined cohorts",
            )
            seen_source_tasks.update(source_task_ids)
            cohort_tasks = [
                contract["task"]
                for contract in task_contracts.values()
                if contract["task"]["campaign_sha256"] == cohort["campaign_sha256"]
            ]
            represented_task_ids = {task["instance_id"] for task in cohort_tasks}
            require(
                represented_task_ids <= set(source_task_ids)
                and sum(task["request_count"] for task in cohort_tasks)
                == cohort["source_global_request_count"]
                and sum(
                    prompt["campaign_sha256"] == cohort["campaign_sha256"]
                    for prompt in result
                )
                == cohort["source_prompt_count"],
                f"cohort {cohort['id']} source count/coverage mismatch",
            )
            unprobed_task_ids.extend(
                task_id for task_id in source_task_ids if task_id not in represented_task_ids
            )
            expected_global_offset += cohort["source_global_request_count"]
            expected_task_offset += cohort["source_task_count"]
        selected_task_count = expected_task_offset
    for task_id, contract in task_contracts.items():
        primary_rows = contract["primary"]
        require(bool(primary_rows), f"{task_id} has no uniform primary checkpoints")
        counts = {row["checkpoint_count"] for row in primary_rows}
        require(
            counts == {len(primary_rows)},
            f"{task_id} checkpoint_count does not match materialized primary prompts",
        )
        ordered_primary = sorted(primary_rows, key=lambda row: row["ordinal"])
        ordinals = [row["ordinal"] for row in ordered_primary]
        require(
            ordinals == list(range(len(primary_rows))),
            f"{task_id} checkpoint ordinals are not contiguous",
        )
        candidate_ordinals = [row["candidate_ordinal"] for row in ordered_primary]
        expected_candidate_ordinals = uniform_request_indices(
            contract["task"]["probeable_request_count"], limit=MAX_CHECKPOINTS
        )
        require(
            candidate_ordinals == expected_candidate_ordinals,
            f"{task_id} primary candidate ordinals differ from exact uniform quantiles",
        )
        request_indices = [row["request_index"] for row in ordered_primary]
        expected_request_indices = [
            contract["task"]["probeable_request_indices"][ordinal - 1]
            for ordinal in expected_candidate_ordinals
        ]
        require(
            request_indices == expected_request_indices,
            f"{task_id} primary requests differ from exact probeable quantiles",
        )
    task_repos: dict[str, str] = {}
    for task_id, contract in task_contracts.items():
        repo = contract["task"]["repo"]
        previous = task_repos.setdefault(task_id, repo)
        require(previous == repo, "task repository changed")
    return {
        "prompts": result,
        "prompt_bundle_sha256": materialized_json_sha256(prompts),
        "campaign_sha256": campaign_sha256s[0] if len(campaign_sha256s) == 1 else None,
        "campaign_sha256s": campaign_sha256s,
        "campaign_count": len(campaign_sha256s),
        "task_count": len(task_contracts),
        "selected_task_count": selected_task_count,
        "unprobed_task_ids": unprobed_task_ids,
        "cohorts": cohort_rows,
        "combined_cohort_manifest_sha256": (
            cohort_rows[0]["cohort_manifest_sha256"] if cohort_rows else None
        ),
        "repository_count": len(set(task_repos.values())),
        "primary_prompt_count": sum(prompt["primary"] for prompt in result),
        "lifecycle_checkpoint_flags": lifecycle_flags,
        "task_contracts": task_contracts,
    }


def _validate_lens(lens: Mapping[str, Any], label: str, protocol: Mapping[str, Any]) -> None:
    require(
        lens.get("d_model") == 5120
        and lens.get("source_layers") == list(ALL_SOURCE_LAYERS)
        and lens.get("tensor_shape") == [5120, 5120],
        f"{label} lens shape/source layers mismatch",
    )
    pins = protocol["lens_pins"]
    if label == "public":
        pin = mapping(pins.get("public"), "public lens pin")
        require(
            lens.get("repo_id") == pin.get("repo_id")
            and lens.get("revision") == pin.get("revision")
            and lens.get("sha256") == pin.get("sha256")
            and lens.get("n_prompts") == pin.get("n_prompts"),
            "public lens pin mismatch",
        )
    elif label == "nf4":
        pin = mapping(pins.get("nf4"), "NF4 lens pin")
        require(
            lens.get("kind") == "local_fit"
            and lens.get("sha256") == pin.get("sha256")
            and lens.get("provenance_sha256") == pin.get("provenance_sha256")
            and lens.get("n_prompts") == pin.get("n_prompts"),
            "NF4 lens pin mismatch",
        )
    elif label == "native":
        pin = mapping(pins.get("native_nvfp4_ste"), "native lens pin")
        require(
            lens.get("kind") == "native_nvfp4_ste_fit"
            and lens.get("sha256") == pin.get("sha256")
            and lens.get("provenance_sha256") == pin.get("provenance_sha256")
            and lens.get("state_sha256") == pin.get("state_sha256")
            and lens.get("n_prompts") == pin.get("n_prompts"),
            "native NVFP4-STE lens pin mismatch",
        )
    else:
        raise ValueError(f"unknown report label: {label}")


def _scored_evidence(
    value: Any,
    *,
    expected_ids: Sequence[int],
    token_text: Mapping[int, str],
    logit_size: int,
    label: str,
) -> dict[int, dict[str, Any]]:
    readout = mapping(value, label)
    records = sequence(readout.get("scored_tokens"), f"{label}.scored_tokens")
    require(len(records) == len(expected_ids), f"{label} scored-token count mismatch")
    result: dict[int, dict[str, Any]] = {}
    for raw_record, expected_id in zip(records, expected_ids, strict=True):
        record = mapping(raw_record, "scored token")
        require(record.get("token_id") == expected_id, f"{label} scored-token order mismatch")
        rank = integer(record.get("rank"), f"{label} rank", minimum=1)
        require(rank <= logit_size, f"{label} rank exceeds logit vocabulary")
        require(record.get("token") == token_text[expected_id], f"{label} token text mismatch")
        result[expected_id] = {
            "rank": rank,
            "score": finite(record.get("score"), f"{label} score"),
            "logprob": finite(record.get("logprob"), f"{label} logprob"),
        }
    return result


def _runtime_identity(runtime: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(runtime)
    model_load_seconds = result.pop("model_load_seconds", None)
    require(
        isinstance(model_load_seconds, (int, float))
        and not isinstance(model_load_seconds, bool)
        and float(model_load_seconds) > 0.0,
        "runtime model-load duration is invalid",
    )
    require(
        result.get("mtp_enabled") is False
        and result.get("enforce_eager") is True
        and result.get("language_model_only") is True
        and result.get("transport_dtype") == "torch.float32"
        and result.get("readout_dtype") == "torch.bfloat16",
        "behavioral residual-capture runtime is incompatible",
    )
    return result


def _nonnegative_diagnostic(value: Any, label: str) -> float:
    result = finite(value, label)
    require(result >= 0.0, f"{label} must be nonnegative")
    return result


def _readout_top_ids(
    value: Any,
    *,
    label: str,
    generated_token_id: int,
    top_k: int,
    logit_size: int,
) -> list[int]:
    records = sequence(value, label)
    require(len(records) == 1, f"{label} must contain exactly one final position")
    record = mapping(records[0], label)
    token_ids = sequence(record.get("token_ids"), f"{label}.token_ids")
    require(
        len(token_ids) >= top_k
        and all(
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and 0 <= token_id < logit_size
            for token_id in token_ids
        ),
        f"{label} top-token IDs are invalid",
    )
    target_rank = integer(record.get("target_rank"), f"{label} target rank", minimum=1)
    require(
        record.get("target_token_id") == generated_token_id
        and target_rank <= logit_size,
        f"{label} generated-token target binding changed",
    )
    return list(token_ids)


def _validate_numerical_diagnostics(
    experiment: Mapping[str, Any],
    *,
    generated_token_id: int,
    final_position: int,
    protocol: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    contract = protocol["numerical_certification"]
    final_norm = mapping(experiment.get("final_norm_reconstruction"), f"{label} final norm")
    norm_max = _nonnegative_diagnostic(final_norm.get("max_abs_error"), f"{label} norm max error")
    norm_rms = _nonnegative_diagnostic(final_norm.get("rms_error"), f"{label} norm RMS error")
    reference_rms = _nonnegative_diagnostic(
        final_norm.get("reference_rms"), f"{label} norm reference RMS"
    )
    require(reference_rms > 0.0, f"{label} norm reference RMS must be positive")
    relative_rms = _nonnegative_diagnostic(
        final_norm.get("relative_rms_error"), f"{label} norm relative RMS error"
    )
    require(
        math.isclose(relative_rms, norm_rms / reference_rms, rel_tol=1e-6, abs_tol=1e-12),
        f"{label} norm relative RMS error is inconsistent",
    )
    norm_max_tolerance = finite(
        final_norm.get("max_abs_tolerance"), f"{label} norm max tolerance"
    )
    norm_rms_tolerance = finite(
        final_norm.get("rms_tolerance"), f"{label} norm RMS tolerance"
    )
    require(
        norm_max_tolerance == contract["final_norm_max_abs_tolerance"]
        and norm_rms_tolerance == contract["final_norm_rms_tolerance"],
        f"{label} norm tolerances differ from the protocol",
    )
    norm_ok = norm_max <= norm_max_tolerance and norm_rms <= norm_rms_tolerance
    require(
        isinstance(final_norm.get("within_tolerance"), bool)
        and final_norm.get("within_tolerance") is norm_ok,
        f"{label} norm tolerance flag is inconsistent",
    )

    final_logits = mapping(
        experiment.get("final_logits_reconstruction"), f"{label} final logits"
    )
    logits_max = _nonnegative_diagnostic(
        final_logits.get("max_abs_error"), f"{label} logits max error"
    )
    logits_rms = _nonnegative_diagnostic(
        final_logits.get("rms_error"), f"{label} logits RMS error"
    )
    logits_max_tolerance = finite(
        final_logits.get("max_abs_tolerance"), f"{label} logits max tolerance"
    )
    logits_rms_tolerance = finite(
        final_logits.get("rms_tolerance"), f"{label} logits RMS tolerance"
    )
    top_k = integer(final_logits.get("top_k_prefix"), f"{label} top-k prefix", minimum=1)
    require(
        logits_max_tolerance == contract["final_logits_max_abs_tolerance"]
        and logits_rms_tolerance == contract["final_logits_rms_tolerance"]
        and top_k == contract["top_k_prefix"],
        f"{label} logits tolerances/top-k differ from the protocol",
    )
    reconstructed_ids = _readout_top_ids(
        experiment.get("final_model_readout"),
        label=f"{label} reconstructed final readout",
        generated_token_id=generated_token_id,
        top_k=top_k,
        logit_size=protocol["logit_vocabulary_size"],
    )
    captured_ids = _readout_top_ids(
        experiment.get("captured_final_model_readout"),
        label=f"{label} captured final readout",
        generated_token_id=generated_token_id,
        top_k=top_k,
        logit_size=protocol["logit_vocabulary_size"],
    )
    top_k_ok = reconstructed_ids[:top_k] == captured_ids[:top_k]
    require(
        isinstance(final_logits.get("top_k_prefix_token_ids_match"), bool)
        and final_logits.get("top_k_prefix_token_ids_match") is top_k_ok,
        f"{label} top-k parity flag is inconsistent",
    )
    top1_ok = reconstructed_ids[0] == generated_token_id and captured_ids[0] == generated_token_id
    require(
        isinstance(experiment.get("final_layer_top1_matches_greedy"), bool)
        and experiment.get("final_layer_top1_matches_greedy") is top1_ok,
        f"{label} greedy top-1 flag is inconsistent",
    )
    logits_ok = (
        logits_max <= logits_max_tolerance
        and logits_rms <= logits_rms_tolerance
        and top_k_ok
    )
    require(
        isinstance(final_logits.get("within_tolerance"), bool)
        and final_logits.get("within_tolerance") is logits_ok,
        f"{label} logits tolerance flag is inconsistent",
    )

    residual = dict(mapping(experiment.get("residual_capture_manifest"), f"{label} residual manifest"))
    residual_sha = nonempty_string(residual.get("sha256"), f"{label} residual SHA")
    expected_logical_bytes = (
        int(contract["residual_tensor_count"])
        * int(contract["residual_d_model"])
        * int(contract["residual_dtype_bytes"])
    )
    require(
        residual.get("algorithm") == RESIDUAL_MANIFEST_ALGORITHM
        and len(residual_sha) == 64
        and all(character in "0123456789abcdef" for character in residual_sha)
        and residual.get("tensor_count") == contract["residual_tensor_count"]
        and residual.get("logical_bytes") == expected_logical_bytes
        and residual.get("token_positions") == [final_position],
        f"{label} residual capture manifest is invalid",
    )
    return {
        "top1": top1_ok,
        "top_k": top_k_ok,
        "norm": dict(final_norm),
        "norm_ok": norm_ok,
        "logits": dict(final_logits),
        "logits_ok": logits_ok,
        "residual": residual,
        "certified": bool(top1_ok and norm_ok and logits_ok),
    }


def validate_report(
    report: Mapping[str, Any],
    *,
    label: str,
    prompt_contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    require(report.get("schema_version") == REPORT_SCHEMA_VERSION, f"{label} schema mismatch")
    require(report.get("score_encoding") == "unrounded-float32", f"{label} score encoding")
    _validate_lens(mapping(report.get("lens"), f"{label} lens"), label, protocol)
    model = mapping(report.get("model"), f"{label} model")
    pin = protocol["model"]
    require(
        model.get("repo_id") == pin["repo_id"]
        and model.get("revision") == pin["revision"]
        and model.get("config_sha256") == pin["config_sha256"]
        and model.get("index_sha256") == pin["index_sha256"],
        f"{label} model pin mismatch",
    )
    runtime_identity = _runtime_identity(mapping(report.get("runtime"), f"{label} runtime"))
    assertions = mapping(report.get("assertions"), f"{label} assertions")
    require(
        assertions.get("lens_hash_matches") is True
        and assertions.get("lens_metadata_matches") is True
        and assertions.get("model_architecture_matches") is True,
        f"{label} integrity assertions failed",
    )
    prompts = prompt_contract["prompts"]
    union_ids = list(
        dict.fromkeys(token_id for prompt in prompts for token_id in prompt["score_token_ids"])
    )
    union_text = [
        next(prompt["token_text"][token_id] for prompt in prompts if token_id in prompt["token_text"])
        for token_id in union_ids
    ]
    global_vocabulary = mapping(report.get("scored_vocabulary"), "report vocabulary")
    require(
        global_vocabulary.get("scope") == "global_plus_per_experiment"
        and global_vocabulary.get("token_ids") == []
        and global_vocabulary.get("tokens") == []
        and global_vocabulary.get("union_token_ids") == union_ids
        and global_vocabulary.get("union_tokens") == union_text,
        f"{label} global scored vocabulary mismatch",
    )
    experiments = sequence(report.get("experiments"), f"{label} experiments")
    require(len(experiments) == len(prompts), f"{label} experiment count mismatch")
    rows: list[dict[str, Any]] = []
    numerical_counts = {
        "experiment_count": len(experiments),
        "greedy_top1_match": 0,
        "final_top5_match": 0,
        "final_norm_within_tolerance": 0,
        "final_logits_within_tolerance": 0,
        "fully_certified": 0,
    }
    for experiment_index, (raw_experiment, prompt) in enumerate(
        zip(experiments, prompts, strict=True)
    ):
        experiment = mapping(raw_experiment, f"{label}.experiment[{experiment_index}]")
        require(experiment.get("id") == prompt["id"], f"{label} prompt ID mismatch")
        require(experiment.get("prompt") == prompt["text"], f"{label} prompt text mismatch")
        require(
            experiment.get("prompt_token_ids") == prompt["token_ids"],
            f"{label} prompt token IDs mismatch",
        )
        require(experiment.get("metadata") == prompt["metadata"], f"{label} metadata mismatch")
        final_position = len(prompt["token_ids"]) - 1
        require(
            experiment.get("positions_requested") == [-1]
            and experiment.get("positions_resolved") == [final_position]
            and experiment.get("capture_positions_resolved") == [final_position]
            and experiment.get("final_validation_position") == final_position,
            f"{label}/{prompt['id']} final-position contract mismatch",
        )
        vocabulary = mapping(experiment.get("scored_vocabulary"), "experiment vocabulary")
        require(
            vocabulary.get("token_ids") == prompt["score_token_ids"]
            and vocabulary.get("tokens")
            == [prompt["token_text"][item] for item in prompt["score_token_ids"]],
            f"{label}/{prompt['id']} scored vocabulary mismatch",
        )
        generated = integer(experiment.get("generated_token_id"), "generated token ID")
        require(generated < protocol["logit_vocabulary_size"], "generated token out of range")
        diagnostics = _validate_numerical_diagnostics(
            experiment,
            generated_token_id=generated,
            final_position=final_position,
            protocol=protocol,
            label=f"{label}/{prompt['id']}",
        )
        top1 = diagnostics["top1"]
        top5 = diagnostics["top_k"]
        norm_ok = diagnostics["norm_ok"]
        logits_ok = diagnostics["logits_ok"]
        certified = diagnostics["certified"]
        numerical_counts["greedy_top1_match"] += int(top1)
        numerical_counts["final_top5_match"] += int(top5)
        numerical_counts["final_norm_within_tolerance"] += int(norm_ok)
        numerical_counts["final_logits_within_tolerance"] += int(logits_ok)
        numerical_counts["fully_certified"] += int(certified)
        layers = sequence(experiment.get("layers"), f"{label} layers")
        layer_ids = [integer(mapping(item, "layer").get("layer"), "layer ID") for item in layers]
        require(
            len(layer_ids) == len(set(layer_ids))
            and set(FIXED_LAYER_BAND).issubset(layer_ids),
            f"{label}/{prompt['id']} lacks fixed layers 24 through 47",
        )
        evidence: dict[str, dict[int, dict[int, dict[str, Any]]]] = {
            "jacobian": {},
            "logit": {},
        }
        logit_fixed: list[dict[str, Any]] = []
        for raw_layer in layers:
            layer = mapping(raw_layer, "layer")
            layer_id = layer["layer"]
            positions = sequence(layer.get("positions"), "layer positions")
            require(len(positions) == 1, "each layer must contain one capture position")
            position = mapping(positions[0], "layer position")
            require(
                position.get("capture_index") == 0
                and position.get("token_position") == final_position,
                f"{label}/{prompt['id']}/layer-{layer_id} capture position mismatch",
            )
            for method, field in (("jacobian", "jacobian_lens"), ("logit", "logit_lens")):
                evidence[method][layer_id] = _scored_evidence(
                    position.get(field),
                    expected_ids=prompt["score_token_ids"],
                    token_text=prompt["token_text"],
                    logit_size=protocol["logit_vocabulary_size"],
                    label=f"{label}/{prompt['id']}/layer-{layer_id}/{method}",
                )
            if layer_id in FIXED_LAYER_BAND:
                logit_fixed.append(dict(mapping(position.get("logit_lens"), "logit lens")))
        residual = diagnostics["residual"]
        baseline_binding = {
            "id": prompt["id"],
            "prompt": experiment.get("prompt"),
            "prompt_token_ids": experiment.get("prompt_token_ids"),
            "metadata": experiment.get("metadata"),
            "scored_vocabulary": dict(vocabulary),
            "generated_token_id": generated,
            "residual_capture_manifest": residual,
            "fixed_band_logit_readouts": logit_fixed,
            "final_layer_top1_matches_greedy": top1,
            "final_norm_reconstruction": diagnostics["norm"],
            "final_logits_reconstruction": diagnostics["logits"],
            "final_model_readout": copy.deepcopy(experiment.get("final_model_readout")),
            "captured_final_model_readout": copy.deepcopy(
                experiment.get("captured_final_model_readout")
            ),
        }
        rows.append(
            {
                "prompt": prompt,
                "evidence": evidence,
                "layer_ids": layer_ids,
                "numerically_certified": certified,
                "baseline_binding": baseline_binding,
            }
        )
    return {
        "label": label,
        "rows": rows,
        "runtime_identity": runtime_identity,
        "numerical_eligibility": {
            **numerical_counts,
            "report_status": report.get("status"),
        },
    }


def validate_report_pairing(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require([report["label"] for report in reports] == list(REPORT_LABELS), "report order")
    reference = reports[0]
    for report in reports[1:]:
        require(
            report["runtime_identity"] == reference["runtime_identity"],
            "public/NF4/native replay runtime identities differ",
        )
        require(len(report["rows"]) == len(reference["rows"]), "report row counts differ")
        for left, right in zip(reference["rows"], report["rows"], strict=True):
            require(
                left["baseline_binding"] == right["baseline_binding"],
                f"final/logit/residual baselines differ: {left['prompt']['id']}",
            )
            require(left["layer_ids"] == right["layer_ids"], "layer coverage differs")
    return {
        "reports": list(REPORT_LABELS),
        "prompt_count": len(reference["rows"]),
        "runtime_identity_equal": True,
        "exact_prompt_id_text_tokens_metadata_equal": True,
        "residual_capture_manifests_equal": True,
        "fixed_band_ordinary_logit_readouts_equal": True,
        "final_model_readouts_equal": True,
        "final_reconstruction_evidence_equal": True,
    }


def _group_band_scores(
    evidence: Mapping[int, Mapping[int, Mapping[str, Any]]],
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    per_layer: list[dict[str, Any]] = []
    aggregate: dict[str, list[float]] = {str(group["id"]): [] for group in groups}
    for layer in FIXED_LAYER_BAND:
        tokens = mapping(evidence.get(layer), f"layer {layer} evidence")
        scores: dict[str, float] = {}
        for group in groups:
            group_id = str(group["id"])
            token_ids = [token["token_id"] for token in group["tokens"]]
            score = logmeanexp([float(tokens[token_id]["score"]) for token_id in token_ids])
            scores[group_id] = score
            aggregate[group_id].append(score)
        per_layer.append({"layer": layer, "class_scores": scores})
    band = {
        group_id: math.fsum(values) / len(values) for group_id, values in aggregate.items()
    }
    return {
        "layers": list(FIXED_LAYER_BAND),
        "per_layer": per_layer,
        "band_mean_class_scores": band,
    }


def _method_evidence(
    reports: Mapping[str, Mapping[str, Any]], row_index: int, method: str
) -> Mapping[int, Mapping[int, Mapping[str, Any]]]:
    if method == "public_jacobian":
        return reports["public"]["rows"][row_index]["evidence"]["jacobian"]
    if method == "nf4_jacobian":
        return reports["nf4"]["rows"][row_index]["evidence"]["jacobian"]
    if method == "native_jacobian":
        return reports["native"]["rows"][row_index]["evidence"]["jacobian"]
    if method == "ordinary_logit":
        return reports["public"]["rows"][row_index]["evidence"]["logit"]
    raise ValueError(f"unknown method: {method}")


def _prediction_record(
    *,
    prompt: Mapping[str, Any],
    label: str,
    class_ids: Sequence[str],
    band: Mapping[str, Any],
) -> dict[str, Any]:
    scores = [float(band["band_mean_class_scores"][class_id]) for class_id in class_ids]
    probabilities = softmax(scores)
    prediction = class_ids[max(range(len(class_ids)), key=lambda index: scores[index])]
    return {
        "row_id": prompt["id"],
        "prompt_id": prompt["id"],
        "task_id": prompt["task_id"],
        "repo": prompt["repo"],
        "cohort_id": prompt["cohort_id"],
        "label": label,
        "class_ids": list(class_ids),
        "scores": scores,
        "probabilities": probabilities,
        "prediction": prediction,
        "raw_fixed_band": band,
    }


def classification_metrics(
    records: Sequence[Mapping[str, Any]], class_ids: Sequence[str]
) -> dict[str, Any]:
    support = {class_id: 0 for class_id in class_ids}
    correct = {class_id: 0 for class_id in class_ids}
    confusion = {
        expected: {predicted: 0 for predicted in class_ids} for expected in class_ids
    }
    nll_values: list[float] = []
    brier_values: list[float] = []
    total_correct = 0
    for record in records:
        label = str(record["label"])
        require(label in class_ids, f"classification label is undeclared: {label}")
        probabilities = [float(value) for value in record["probabilities"]]
        require(
            len(probabilities) == len(class_ids)
            and all(math.isfinite(value) and value >= 0.0 for value in probabilities)
            and abs(math.fsum(probabilities) - 1.0) <= 1e-9,
            "classification probabilities are invalid",
        )
        prediction = str(record["prediction"])
        require(prediction in class_ids, "classification prediction is undeclared")
        label_index = class_ids.index(label)
        support[label] += 1
        confusion[label][prediction] += 1
        is_correct = prediction == label
        correct[label] += int(is_correct)
        total_correct += int(is_correct)
        nll_values.append(-math.log(max(probabilities[label_index], 1e-300)))
        brier_values.append(
            math.fsum(
                (probability - (1.0 if index == label_index else 0.0)) ** 2
                for index, probability in enumerate(probabilities)
            )
        )
    count = len(records)
    recalls = {
        class_id: (correct[class_id] / support[class_id] if support[class_id] else None)
        for class_id in class_ids
    }
    complete_support = all(support.values())
    macro = (
        math.fsum(float(recalls[class_id]) for class_id in class_ids) / len(class_ids)
        if complete_support
        else None
    )
    return {
        "status": "available" if count else "insufficient_no_rows",
        "row_count": count,
        "task_count": len({str(record["task_id"]) for record in records}),
        "repository_count": len({str(record["repo"]) for record in records}),
        "support": support,
        "confusion": confusion,
        "per_class_recall": recalls,
        "micro_accuracy": total_correct / count if count else None,
        "balanced_accuracy": macro,
        "macro_recall": macro,
        "balanced_accuracy_status": (
            "available_all_declared_classes"
            if complete_support
            else "insufficient_declared_class_support"
        ),
        "negative_log_likelihood": math.fsum(nll_values) / count if count else None,
        "multiclass_brier": math.fsum(brier_values) / count if count else None,
    }


def classification_metrics_by_cohort(
    records: Sequence[Mapping[str, Any]], class_ids: Sequence[str]
) -> dict[str, Any]:
    cohort_ids = sorted(
        {str(record.get("cohort_id", "unspecified")) for record in records}
    )
    return {
        "status": "descriptive_only_no_subgroup_refitting",
        "cohorts": {
            cohort_id: classification_metrics(
                [
                    record
                    for record in records
                    if str(record.get("cohort_id", "unspecified")) == cohort_id
                ],
                class_ids,
            )
            for cohort_id in cohort_ids
        },
    }


def _support_counts(
    rows: Sequence[Mapping[str, Any]], class_ids: Sequence[str]
) -> dict[str, int]:
    return {
        class_id: sum(str(row["label"]) == class_id for row in rows)
        for class_id in class_ids
    }


def _support_satisfies(counts: Mapping[str, int], minimum: int) -> bool:
    return all(count >= minimum for count in counts.values())


def _split_rank(seed: int, heldout_repo: str, repo: str) -> str:
    return sha256_text(f"{seed}\0{heldout_repo}\0{repo}")


def build_fold_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    heldout_repo: str,
    all_repositories: Sequence[str],
    class_ids: Sequence[str],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    other = [repo for repo in all_repositories if repo != heldout_repo]
    minimum_fit = int(contract["minimum_fit_repositories"])
    minimum_calibration = int(contract["minimum_calibration_repositories"])
    calibration_count = int(contract["calibration_repository_count"])
    if (
        heldout_repo not in all_repositories
        or calibration_count < minimum_calibration
        or len(other) - calibration_count < minimum_fit
    ):
        return {
            "status": "insufficient_repository_count",
            "heldout_repo": heldout_repo,
            "fit_repositories": [],
            "calibration_repositories": [],
        }
    ranked = sorted(
        other,
        key=lambda repo: (
            _split_rank(int(contract["split_seed"]), heldout_repo, repo),
            repo,
        ),
    )
    minimum_fit_examples = int(contract["minimum_fit_examples_per_class"])
    minimum_calibration_examples = int(
        contract["minimum_calibration_examples_per_class"]
    )
    attempts = 0
    for calibration_tuple in itertools.combinations(ranked, calibration_count):
        attempts += 1
        calibration_repositories = list(calibration_tuple)
        fit_repositories = [repo for repo in ranked if repo not in calibration_tuple]
        fit_rows = [row for row in rows if row["repo"] in fit_repositories]
        calibration_rows = [
            row for row in rows if row["repo"] in calibration_repositories
        ]
        fit_support = _support_counts(fit_rows, class_ids)
        calibration_support = _support_counts(calibration_rows, class_ids)
        if _support_satisfies(
            fit_support, minimum_fit_examples
        ) and _support_satisfies(calibration_support, minimum_calibration_examples):
            return {
                "status": "available",
                "heldout_repo": heldout_repo,
                "fit_repositories": fit_repositories,
                "calibration_repositories": calibration_repositories,
                "fit_support": fit_support,
                "calibration_support": calibration_support,
                "candidate_subsets_checked": attempts,
                "split_used_heldout_labels": False,
            }
    return {
        "status": "insufficient_class_support",
        "heldout_repo": heldout_repo,
        "fit_repositories": [],
        "calibration_repositories": [],
        "candidate_subsets_checked": attempts,
        "split_used_heldout_labels": False,
    }


def _bias_objective(
    scores: np.ndarray, labels: np.ndarray, theta: np.ndarray, l2: float
) -> float:
    full = np.concatenate([theta, np.zeros(1, dtype=np.float64)])
    logits = scores + full
    maximum = np.max(logits, axis=1, keepdims=True)
    logsum = maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))
    return float(np.mean(logsum - logits[np.arange(len(labels)), labels])) + float(
        0.5 * l2 * np.dot(theta, theta)
    )


def fit_class_bias(
    rows: Sequence[Mapping[str, Any]],
    class_ids: Sequence[str],
    *,
    l2: float,
    maximum_iterations: int = 100,
) -> dict[str, Any]:
    require(bool(rows), "cannot fit bias without rows")
    scores = np.asarray([row["scores"] for row in rows], dtype=np.float64)
    labels = np.asarray([class_ids.index(str(row["label"])) for row in rows], dtype=np.int64)
    class_count = len(class_ids)
    theta = np.zeros(class_count - 1, dtype=np.float64)
    converged = False
    iterations = 0
    objective = _bias_objective(scores, labels, theta, l2)
    for iterations in range(1, maximum_iterations + 1):
        full = np.concatenate([theta, np.zeros(1, dtype=np.float64)])
        logits = scores + full
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        targets = np.zeros_like(probabilities)
        targets[np.arange(len(labels)), labels] = 1.0
        reduced = probabilities[:, :-1]
        gradient = np.mean(reduced - targets[:, :-1], axis=0) + l2 * theta
        hessian = np.zeros((class_count - 1, class_count - 1), dtype=np.float64)
        for probability in probabilities:
            vector = probability[:-1]
            hessian += np.diag(vector) - np.outer(vector, vector)
        hessian /= len(labels)
        hessian += np.eye(class_count - 1, dtype=np.float64) * l2
        if float(np.max(np.abs(gradient))) < 1e-10:
            converged = True
            break
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise ValueError("bias calibration Hessian is singular") from error
        accepted = False
        scale = 1.0
        for _ in range(40):
            candidate = theta - scale * step
            candidate_objective = _bias_objective(scores, labels, candidate, l2)
            if candidate_objective <= objective - 1e-12:
                theta = candidate
                objective = candidate_objective
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            converged = float(np.max(np.abs(gradient))) < 1e-7
            break
        if float(np.max(np.abs(scale * step))) < 1e-9:
            converged = True
            break
    biases = np.concatenate([theta, np.zeros(1, dtype=np.float64)])
    require(np.all(np.isfinite(biases)), "bias calibration produced nonfinite parameters")
    return {
        "biases": [float(value) for value in biases],
        "reference_class": class_ids[-1],
        "iterations": iterations,
        "converged": converged,
        "fit_objective": objective,
        "fit_row_count": len(rows),
    }


def _temperature_objective(
    rows: Sequence[Mapping[str, Any]],
    class_ids: Sequence[str],
    biases: Sequence[float],
    temperature: float,
) -> float:
    losses: list[float] = []
    for row in rows:
        logits = [
            (float(score) + float(bias)) / temperature
            for score, bias in zip(row["scores"], biases, strict=True)
        ]
        probabilities = softmax(logits)
        losses.append(-math.log(max(probabilities[class_ids.index(str(row["label"]))], 1e-300)))
    return math.fsum(losses) / len(losses)


def fit_temperature(
    rows: Sequence[Mapping[str, Any]],
    class_ids: Sequence[str],
    biases: Sequence[float],
    *,
    minimum: float,
    maximum: float,
    iterations: int,
) -> dict[str, Any]:
    require(bool(rows) and 0.0 < minimum < maximum, "temperature fit inputs are invalid")
    left = math.log(minimum)
    right = math.log(maximum)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc = _temperature_objective(rows, class_ids, biases, math.exp(c))
    fd = _temperature_objective(rows, class_ids, biases, math.exp(d))
    for _ in range(iterations):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = _temperature_objective(rows, class_ids, biases, math.exp(c))
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = _temperature_objective(rows, class_ids, biases, math.exp(d))
    log_temperature = (left + right) / 2.0
    temperature = math.exp(log_temperature)
    return {
        "temperature": temperature,
        "calibration_objective": _temperature_objective(
            rows, class_ids, biases, temperature
        ),
        "calibration_row_count": len(rows),
        "search_iterations": iterations,
    }


def _apply_calibrator(
    row: Mapping[str, Any], class_ids: Sequence[str], biases: Sequence[float], temperature: float
) -> dict[str, Any]:
    logits = [
        (float(score) + float(bias)) / temperature
        for score, bias in zip(row["scores"], biases, strict=True)
    ]
    probabilities = softmax(logits)
    prediction_index = max(range(len(class_ids)), key=lambda index: logits[index])
    return {
        **{key: row[key] for key in ("row_id", "task_id", "repo", "label")},
        "cohort_id": str(row.get("cohort_id", "unspecified")),
        "class_ids": list(class_ids),
        "scores": list(row["scores"]),
        "calibrated_logits": logits,
        "probabilities": probabilities,
        "prediction": class_ids[prediction_index],
    }


def _fit_prior(
    rows: Sequence[Mapping[str, Any]], class_ids: Sequence[str], alpha: float
) -> list[float]:
    counts = _support_counts(rows, class_ids)
    denominator = len(rows) + alpha * len(class_ids)
    return [(counts[class_id] + alpha) / denominator for class_id in class_ids]


def _prior_prediction(
    row: Mapping[str, Any], class_ids: Sequence[str], probabilities: Sequence[float]
) -> dict[str, Any]:
    prediction_index = max(range(len(class_ids)), key=lambda index: probabilities[index])
    return {
        **{key: row[key] for key in ("row_id", "task_id", "repo", "label")},
        "cohort_id": str(row.get("cohort_id", "unspecified")),
        "class_ids": list(class_ids),
        "probabilities": list(probabilities),
        "prediction": class_ids[prediction_index],
    }


def crossfit_track(
    rows: Sequence[Mapping[str, Any]],
    *,
    class_ids: Sequence[str],
    all_repositories: Sequence[str],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    prior_predictions: list[dict[str, Any]] = []
    calibrator_contract = mapping(contract.get("calibrator"), "calibrator contract")
    prior_contract = mapping(contract.get("majority_baseline"), "majority baseline")
    for heldout_repo in sorted(all_repositories):
        evaluation_rows = [row for row in rows if row["repo"] == heldout_repo]
        if not evaluation_rows:
            folds.append(
                {
                    "heldout_repository": heldout_repo,
                    "status": "insufficient_no_evaluation_rows",
                }
            )
            continue
        split = build_fold_split(
            rows,
            heldout_repo=heldout_repo,
            all_repositories=all_repositories,
            class_ids=class_ids,
            contract=contract,
        )
        if split["status"] != "available":
            folds.append({"heldout_repository": heldout_repo, **split})
            continue
        fit_rows = [row for row in rows if row["repo"] in split["fit_repositories"]]
        calibration_rows = [
            row for row in rows if row["repo"] in split["calibration_repositories"]
        ]
        require(
            not ({row["repo"] for row in fit_rows} & {row["repo"] for row in calibration_rows})
            and heldout_repo not in {row["repo"] for row in fit_rows + calibration_rows},
            "cross-fitting repository isolation failed",
        )
        bias_fit = fit_class_bias(
            fit_rows,
            class_ids,
            l2=float(calibrator_contract["bias_l2"]),
        )
        temperature_fit = fit_temperature(
            calibration_rows,
            class_ids,
            bias_fit["biases"],
            minimum=float(calibrator_contract["temperature_min"]),
            maximum=float(calibrator_contract["temperature_max"]),
            iterations=int(calibrator_contract["temperature_search_iterations"]),
        )
        fold_predictions = [
            _apply_calibrator(
                row,
                class_ids,
                bias_fit["biases"],
                float(temperature_fit["temperature"]),
            )
            for row in evaluation_rows
        ]
        prior = _fit_prior(fit_rows, class_ids, float(prior_contract["laplace_alpha"]))
        fold_prior_predictions = [
            _prior_prediction(row, class_ids, prior) for row in evaluation_rows
        ]
        predictions.extend(fold_predictions)
        prior_predictions.extend(fold_prior_predictions)
        folds.append(
            {
                "heldout_repository": heldout_repo,
                "status": "available",
                "evaluation_task_ids": sorted(
                    {str(row["task_id"]) for row in evaluation_rows}
                ),
                "evaluation_row_count": len(evaluation_rows),
                "fit_repositories": split["fit_repositories"],
                "calibration_repositories": split["calibration_repositories"],
                "fit_support": split["fit_support"],
                "calibration_support": split["calibration_support"],
                "candidate_subsets_checked": split["candidate_subsets_checked"],
                "split_used_heldout_labels": False,
                "fit_task_ids": sorted({str(row["task_id"]) for row in fit_rows}),
                "calibration_task_ids": sorted(
                    {str(row["task_id"]) for row in calibration_rows}
                ),
                "bias_fit": bias_fit,
                "temperature_fit": temperature_fit,
                "majority_prior": {
                    class_id: prior[index] for index, class_id in enumerate(class_ids)
                },
                "fit_payload_sha256": sha256_json(
                    [
                        {
                            "row_id": row["row_id"],
                            "task_id": row["task_id"],
                            "repo": row["repo"],
                            "label": row["label"],
                            "scores": row["scores"],
                        }
                        for row in sorted(fit_rows, key=lambda item: str(item["row_id"]))
                    ]
                ),
                "calibration_payload_sha256": sha256_json(
                    [
                        {
                            "row_id": row["row_id"],
                            "task_id": row["task_id"],
                            "repo": row["repo"],
                            "label": row["label"],
                            "scores": row["scores"],
                        }
                        for row in sorted(
                            calibration_rows, key=lambda item: str(item["row_id"])
                        )
                    ]
                ),
            }
        )
    expected_row_ids = {str(row["row_id"]) for row in rows}
    predicted_row_ids = {str(row["row_id"]) for row in predictions}
    complete = (
        all(fold.get("status") == "available" for fold in folds)
        and predicted_row_ids == expected_row_ids
        and len(predictions) == len(rows)
    )
    metrics = classification_metrics(predictions, class_ids)
    prior_metrics = classification_metrics(prior_predictions, class_ids)
    support_pass = complete and metrics["balanced_accuracy"] is not None
    return {
        "status": "available" if support_pass else "insufficient_split_or_class_support",
        "algorithm": contract["algorithm"],
        "group": contract["group"],
        "evaluation_group": contract["evaluation_group"],
        "fold_count": len(folds),
        "successful_fold_count": sum(fold.get("status") == "available" for fold in folds),
        "complete_evaluation_row_coverage": predicted_row_ids == expected_row_ids,
        "all_split_and_support_rules_pass": support_pass,
        "folds": folds,
        "predictions": predictions,
        "metrics": metrics,
        "majority_baseline": {
            "status": "available" if support_pass else "insufficient_split_or_class_support",
            "predictions": prior_predictions,
            "metrics": prior_metrics,
        },
    }


def _hierarchical_sample(
    records: Sequence[Mapping[str, Any]], rng: random.Random
) -> list[Mapping[str, Any]]:
    by_repo: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for record in records:
        by_repo.setdefault(str(record["repo"]), {}).setdefault(
            str(record["task_id"]), []
        ).append(record)
    repositories = sorted(by_repo)
    sample: list[Mapping[str, Any]] = []
    for _ in repositories:
        repository = repositories[rng.randrange(len(repositories))]
        tasks = sorted(by_repo[repository])
        for _ in tasks:
            task = tasks[rng.randrange(len(tasks))]
            sample.extend(by_repo[repository][task])
    return sample


def bootstrap_classification(
    records: Sequence[Mapping[str, Any]],
    class_ids: Sequence[str],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
    minimum_valid_fraction: float,
) -> dict[str, Any]:
    require(
        0.0 < minimum_valid_fraction <= 1.0,
        "bootstrap minimum valid fraction must be in (0, 1]",
    )
    if samples <= 0 or not records:
        return {
            "status": "disabled" if samples <= 0 else "insufficient_no_rows",
            "samples_requested": samples,
            "samples_valid": 0,
            "valid_fraction": 0.0,
            "minimum_valid_fraction": minimum_valid_fraction,
            "unit": "hierarchical_repository_then_task_never_rows",
            "intervals": {},
            "metric_status": {},
        }
    rng = random.Random(seed)
    values: dict[str, list[float]] = {
        "micro_accuracy": [],
        "balanced_accuracy": [],
        "negative_log_likelihood": [],
        "multiclass_brier": [],
    }
    for _ in range(samples):
        metrics = classification_metrics(_hierarchical_sample(records, rng), class_ids)
        for key in values:
            value = metrics[key]
            if value is not None:
                values[key].append(float(value))
    alpha = (1.0 - confidence_level) / 2.0
    valid_fractions = {
        key: len(items) / samples for key, items in values.items()
    }
    metric_status = {
        key: (
            "available"
            if valid_fractions[key] >= minimum_valid_fraction
            else "insufficient_valid_bootstrap_fraction"
        )
        for key in values
    }
    intervals = {
        key: (
            {
                "lower": percentile(items, alpha),
                "upper": percentile(items, 1.0 - alpha),
                "valid_samples": len(items),
            }
            if items and metric_status[key] == "available"
            else None
        )
        for key, items in values.items()
    }
    return {
        "status": (
            "available"
            if all(status == "available" for status in metric_status.values())
            else "insufficient_valid_bootstrap_fraction"
        ),
        "samples_requested": samples,
        "samples_valid": min((len(items) for items in values.values()), default=0),
        "valid_fraction": min(valid_fractions.values(), default=0.0),
        "minimum_valid_fraction": minimum_valid_fraction,
        "seed": seed,
        "confidence_level": confidence_level,
        "unit": "hierarchical_repository_then_task_never_rows",
        "metric_status": metric_status,
        "valid_fraction_by_metric": valid_fractions,
        "intervals": intervals,
    }


def _add_bootstrap(
    result: dict[str, Any],
    *,
    class_ids: Sequence[str],
    bootstrap: Mapping[str, Any],
    samples: int,
    seed_offset: int,
) -> None:
    result["bootstrap"] = bootstrap_classification(
        result.get("predictions", []),
        class_ids,
        samples=samples,
        seed=int(bootstrap["seed"]) + seed_offset,
        confidence_level=float(bootstrap["confidence_level"]),
        minimum_valid_fraction=float(bootstrap["minimum_valid_fraction"]),
    )


def build_action_rows(
    prompt_contract: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    detailed: list[dict[str, Any]] = []
    features = {method: [] for method in METHODS}
    classes = protocol["action_classes"]
    class_ids = protocol["action_ids"]
    for row_index, prompt in enumerate(prompt_contract["prompts"]):
        if not prompt["primary"] or prompt["action_status"] != "available":
            continue
        certified = all(
            reports[label]["rows"][row_index]["numerically_certified"]
            for label in REPORT_LABELS
        )
        methods: dict[str, Any] = {}
        if certified:
            for method in METHODS:
                band = _group_band_scores(
                    _method_evidence(reports, row_index, method), classes
                )
                record = _prediction_record(
                    prompt=prompt,
                    label=str(prompt["action_label"]),
                    class_ids=class_ids,
                    band=band,
                )
                features[method].append(record)
                methods[method] = {
                    "prediction": record["prediction"],
                    "probabilities": record["probabilities"],
                    "raw_fixed_band": band,
                }
        detailed.append(
            {
                "prompt_id": prompt["id"],
                "task_id": prompt["task_id"],
                "repo": prompt["repo"],
                "cohort_id": prompt["cohort_id"],
                "task_request_index": prompt["task_request_index"],
                "expected_action": prompt["action_label"],
                "numerically_certified_across_all_reports": certified,
                "methods": methods,
            }
        )
    return detailed, features


def build_official_outcome_rows(
    prompt_contract: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_task: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for row_index, prompt in enumerate(prompt_contract["prompts"]):
        if not prompt["primary"]:
            continue
        previous = by_task.get(prompt["task_id"])
        if previous is None or prompt["task_request_index"] > previous[1]["task_request_index"]:
            by_task[prompt["task_id"]] = (row_index, prompt)
    detailed: list[dict[str, Any]] = []
    features = {method: [] for method in METHODS}
    classes = protocol["outcome_classes"]
    class_ids = protocol["outcome_ids"]
    observed_tasks: set[str] = set()
    for task_id in sorted(by_task):
        row_index, prompt = by_task[task_id]
        require(task_id not in observed_tasks, "official outcome repeated for a task")
        observed_tasks.add(task_id)
        if prompt["official_outcome_status"] != "available":
            detailed.append(
                {
                    "task_id": task_id,
                    "repo": prompt["repo"],
                    "cohort_id": prompt["cohort_id"],
                    "prompt_id": prompt["id"],
                    "status": "missing_not_imputed",
                    "methods": {},
                }
            )
            continue
        certified = all(
            reports[label]["rows"][row_index]["numerically_certified"]
            for label in REPORT_LABELS
        )
        methods: dict[str, Any] = {}
        if certified:
            for method in METHODS:
                band = _group_band_scores(
                    _method_evidence(reports, row_index, method), classes
                )
                record = _prediction_record(
                    prompt=prompt,
                    label=str(prompt["official_outcome"]),
                    class_ids=class_ids,
                    band=band,
                )
                features[method].append(record)
                methods[method] = {
                    "prediction": record["prediction"],
                    "probabilities": record["probabilities"],
                    "raw_fixed_band": band,
                }
        detailed.append(
            {
                "task_id": task_id,
                "repo": prompt["repo"],
                "cohort_id": prompt["cohort_id"],
                "prompt_id": prompt["id"],
                "status": "available" if certified else "numerically_uncertified",
                "official_outcome": prompt["official_outcome"],
                "observation_index_within_task": 1,
                "methods": methods,
            }
        )
    require(
        len(observed_tasks) == prompt_contract["task_count"],
        "official-outcome task selection does not cover every task exactly once",
    )
    return detailed, features


def _target_group(target: Mapping[str, Any]) -> dict[str, Any]:
    return {"id": "target", "tokens": target["forms"]}


def _foil_group(foil: Mapping[str, Any]) -> dict[str, Any]:
    return {"id": foil["id"], "tokens": foil["forms"]}


def build_future_target_rows(
    prompt_contract: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    detailed: list[dict[str, Any]] = []
    aggregate: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for row_index, prompt in enumerate(prompt_contract["prompts"]):
        if not prompt["primary"]:
            continue
        certified = all(
            reports[label]["rows"][row_index]["numerically_certified"]
            for label in REPORT_LABELS
        )
        for target in prompt["targets"]:
            eligibility = target["eligibility"]
            if eligibility["status"] != "eligible":
                continue
            retained = set(eligibility["retained_hidden_foil_ids"])
            foils = [foil for foil in target["foils"] if foil["id"] in retained]
            require(
                bool(foils)
                and all(foil["task_instance_id"] == prompt["task_id"] for foil in foils),
                "eligible future target lacks a same-task hidden foil",
            )
            methods: dict[str, Any] = {}
            if certified:
                for method in METHODS:
                    groups = [_target_group(target)] + [_foil_group(foil) for foil in foils]
                    band = _group_band_scores(
                        _method_evidence(reports, row_index, method), groups
                    )
                    scores = band["band_mean_class_scores"]
                    strongest = max(
                        foils, key=lambda foil: (scores[foil["id"]], foil["id"])
                    )
                    margin = float(scores["target"] - scores[strongest["id"]])
                    probability = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, margin))))
                    method_record = {
                        "strongest_hidden_foil_id": strongest["id"],
                        "strongest_hidden_foil": strongest["target"],
                        "target_minus_strongest_foil_margin": margin,
                        "target_probability": probability,
                        "target_preferred": margin > 0.0,
                        "raw_fixed_band": band,
                    }
                    methods[method] = method_record
                    aggregate[method].append(
                        {
                            "row_id": f"{prompt['id']}::{target['id']}",
                            "prompt_id": prompt["id"],
                            "target_id": target["id"],
                            "task_id": prompt["task_id"],
                            "repo": prompt["repo"],
                            "cohort_id": prompt["cohort_id"],
                            "margin": margin,
                            "probability": probability,
                            "correct": margin > 0.0,
                            "strongest_hidden_foil_id": strongest["id"],
                        }
                    )
            detailed.append(
                {
                    "prompt_id": prompt["id"],
                    "task_id": prompt["task_id"],
                    "repo": prompt["repo"],
                    "cohort_id": prompt["cohort_id"],
                    "target_id": target["id"],
                    "target": target["target"],
                    "eligible_hidden_foil_ids": sorted(retained),
                    "numerically_certified_across_all_reports": certified,
                    "methods": methods,
                }
            )
    return detailed, aggregate


def _future_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "status": "insufficient_no_eligible_rows",
            "row_count": 0,
            "target_preference_accuracy": None,
            "negative_log_likelihood": None,
            "binary_brier": None,
            "mean_margin": None,
        }
    return {
        "status": "descriptive_only_dynamic_target_vocabulary",
        "row_count": len(records),
        "task_count": len({record["task_id"] for record in records}),
        "repository_count": len({record["repo"] for record in records}),
        "target_preference_accuracy": sum(record["correct"] for record in records)
        / len(records),
        "negative_log_likelihood": math.fsum(
            -math.log(max(float(record["probability"]), 1e-300)) for record in records
        )
        / len(records),
        "binary_brier": math.fsum(
            (1.0 - float(record["probability"])) ** 2 for record in records
        )
        / len(records),
        "mean_margin": math.fsum(float(record["margin"]) for record in records)
        / len(records),
        "foil_rule": "strongest_hidden_same_task_foil_by_fixed_band_score",
    }


def _future_metrics_by_cohort(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cohort_ids = sorted({str(record.get("cohort_id", "unspecified")) for record in records})
    return {
        "status": "descriptive_only_no_subgroup_refitting",
        "cohorts": {
            cohort_id: _future_metrics(
                [
                    record
                    for record in records
                    if str(record.get("cohort_id", "unspecified")) == cohort_id
                ]
            )
            for cohort_id in cohort_ids
        },
    }


def _bootstrap_future(
    records: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    if samples <= 0 or not records:
        return {
            "status": "disabled" if samples <= 0 else "insufficient_no_rows",
            "samples_requested": samples,
            "unit": "hierarchical_repository_then_task_never_rows",
            "intervals": {},
        }
    rng = random.Random(seed)
    values = {key: [] for key in ("target_preference_accuracy", "negative_log_likelihood", "binary_brier", "mean_margin")}
    for _ in range(samples):
        metrics = _future_metrics(_hierarchical_sample(records, rng))
        for key in values:
            values[key].append(float(metrics[key]))
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "status": "available",
        "samples_requested": samples,
        "seed": seed,
        "unit": "hierarchical_repository_then_task_never_rows",
        "intervals": {
            key: {
                "lower": percentile(items, alpha),
                "upper": percentile(items, 1.0 - alpha),
                "valid_samples": len(items),
            }
            for key, items in values.items()
        },
    }


def _track_analysis(
    features: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    class_ids: Sequence[str],
    all_repositories: Sequence[str],
    protocol: Mapping[str, Any],
    bootstrap_samples: int,
    seed_base: int,
) -> dict[str, Any]:
    raw = {
        method: {
            "status": "descriptive_uncalibrated",
            "metrics": classification_metrics(rows, class_ids),
            "descriptive_cohort_metrics": classification_metrics_by_cohort(
                rows, class_ids
            ),
        }
        for method, rows in features.items()
    }
    crossfit: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        result = crossfit_track(
            features[method],
            class_ids=class_ids,
            all_repositories=all_repositories,
            contract=protocol["crossfit"],
        )
        _add_bootstrap(
            result,
            class_ids=class_ids,
            bootstrap=protocol["bootstrap"],
            samples=bootstrap_samples,
            seed_offset=seed_base + method_index,
        )
        result["descriptive_cohort_metrics"] = classification_metrics_by_cohort(
            result["predictions"], class_ids
        )
        result["majority_baseline"]["descriptive_cohort_metrics"] = (
            classification_metrics_by_cohort(
                result["majority_baseline"]["predictions"], class_ids
            )
        )
        if method == "ordinary_logit":
            _add_bootstrap(
                result["majority_baseline"],
                class_ids=class_ids,
                bootstrap=protocol["bootstrap"],
                samples=bootstrap_samples,
                seed_offset=seed_base + 100,
            )
        crossfit[method] = result
    split_support_pass = all(
        result["all_split_and_support_rules_pass"] for result in crossfit.values()
    )
    bootstrap_support_pass = all(
        result["bootstrap"]["status"] == "available" for result in crossfit.values()
    )
    return {
        "raw_descriptive": raw,
        "task_held_out_crossfit": crossfit,
        "majority_baseline": crossfit["ordinary_logit"]["majority_baseline"],
        "ordinary_logit_baseline": crossfit["ordinary_logit"],
        "all_method_split_support_rules_pass": split_support_pass,
        "all_method_bootstrap_valid_fraction_rules_pass": bootstrap_support_pass,
        "all_method_inference_support_rules_pass": (
            split_support_pass and bootstrap_support_pass
        ),
    }


def build_analysis(
    prompts_value: Any,
    public_report: Mapping[str, Any],
    nf4_report: Mapping[str, Any],
    native_report: Mapping[str, Any],
    protocol_value: Mapping[str, Any],
    *,
    protocol_sha256: str,
    input_hashes: Mapping[str, str],
    bootstrap_samples: int,
) -> dict[str, Any]:
    protocol = validate_protocol(protocol_value, protocol_sha256=protocol_sha256)
    prompt_contract = validate_prompt_bundle(prompts_value, protocol=protocol)
    reports_list = [
        validate_report(
            report,
            label=label,
            prompt_contract=prompt_contract,
            protocol=protocol,
        )
        for label, report in zip(
            REPORT_LABELS, (public_report, nf4_report, native_report), strict=True
        )
    ]
    pairing = validate_report_pairing(reports_list)
    reports = {report["label"]: report for report in reports_list}
    detailed_action, action_features = build_action_rows(
        prompt_contract, reports, protocol
    )
    detailed_outcome, outcome_features = build_official_outcome_rows(
        prompt_contract, reports, protocol
    )
    detailed_future, future_features = build_future_target_rows(prompt_contract, reports)
    all_repositories = sorted(
        {prompt["repo"] for prompt in prompt_contract["prompts"] if prompt["primary"]}
    )
    action_analysis = _track_analysis(
        action_features,
        class_ids=protocol["action_ids"],
        all_repositories=all_repositories,
        protocol=protocol,
        bootstrap_samples=bootstrap_samples,
        seed_base=0,
    )
    outcome_analysis = _track_analysis(
        outcome_features,
        class_ids=protocol["outcome_ids"],
        all_repositories=all_repositories,
        protocol=protocol,
        bootstrap_samples=bootstrap_samples,
        seed_base=1000,
    )
    future_summary: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        future_summary[method] = {
            "metrics": _future_metrics(future_features[method]),
            "descriptive_cohort_metrics": _future_metrics_by_cohort(
                future_features[method]
            ),
            "bootstrap": _bootstrap_future(
                future_features[method],
                samples=bootstrap_samples,
                seed=int(protocol["bootstrap"]["seed"]) + 2000 + method_index,
                confidence_level=float(protocol["bootstrap"]["confidence_level"]),
            ),
        }
    all_support = bool(
        action_analysis["all_method_inference_support_rules_pass"]
        and outcome_analysis["all_method_inference_support_rules_pass"]
        and not prompt_contract["unprobed_task_ids"]
    )
    task_count = prompt_contract["task_count"]
    minimum_nondevelopment = int(protocol["development"]["minimum_tasks_for_non_development_claim"])
    if task_count < minimum_nondevelopment:
        decision_status = (
            "development_support_complete"
            if all_support
            else "development_insufficient_split_or_class_support"
        )
    else:
        decision_status = (
            "held_out_evaluation_complete"
            if all_support
            else "insufficient_split_or_class_support"
        )
    return {
        "schema_version": 1,
        "kind": "swe_verified_behavioral_task_held_out_analysis",
        "analysis_version": "task-held-out-v1",
        "status": decision_status,
        "inputs": dict(input_hashes),
        "protocol": {
            "path": str(DEFAULT_PROTOCOL.relative_to(ROOT)),
            "sha256": protocol_sha256,
            "fixed_layers": list(FIXED_LAYER_BAND),
            "class_score_reduction": protocol_value["class_score_reduction"],
            "cross_fitting": protocol_value["cross_fitting"],
            "prompt_context": protocol_value["prompt_context"],
            "numerical_certification": protocol_value["numerical_certification"],
            "bootstrap": {
                **dict(protocol_value["bootstrap"]),
                "samples": bootstrap_samples,
            },
        },
        "campaign": {
            "campaign_sha256": prompt_contract["campaign_sha256"],
            "campaign_sha256s": prompt_contract["campaign_sha256s"],
            "campaign_count": prompt_contract["campaign_count"],
            "combined_cohort_manifest_sha256": prompt_contract[
                "combined_cohort_manifest_sha256"
            ],
            "cohorts": prompt_contract["cohorts"],
            "task_count_by_campaign": {
                campaign_sha256: sum(
                    contract["task"]["campaign_sha256"] == campaign_sha256
                    for contract in prompt_contract["task_contracts"].values()
                )
                for campaign_sha256 in prompt_contract["campaign_sha256s"]
            },
            "task_count": task_count,
            "selected_task_count": prompt_contract["selected_task_count"],
            "unprobed_task_count": len(prompt_contract["unprobed_task_ids"]),
            "unprobed_task_ids": prompt_contract["unprobed_task_ids"],
            "repository_count": prompt_contract["repository_count"],
            "prompt_count": len(prompt_contract["prompts"]),
            "primary_probeable_uniform_prompt_count": prompt_contract[
                "primary_prompt_count"
            ],
            "n10_policy": protocol["development"]["n10_policy"],
        },
        "cohort_audit": {
            "primary": protocol["primary_cohort"],
            "label_conditioned_or_nonuniform_checkpoint_count": len(
                prompt_contract["lifecycle_checkpoint_flags"]
            ),
            "flags": prompt_contract["lifecycle_checkpoint_flags"],
            "primary_decision_cohort": "uniform_probeable_request_index_only",
        },
        "pairing": pairing,
        "numerical_eligibility": {
            report["label"]: report["numerical_eligibility"] for report in reports_list
        },
        "tracks": {
            "next_action": {
                "observation_unit": "uniform_probeable_request_checkpoint",
                "rows": detailed_action,
                **action_analysis,
            },
            "official_outcome": {
                "observation_unit": "one_latest_uniform_probeable_checkpoint_per_task",
                "stage_repetition_forbidden": True,
                "rows": detailed_outcome,
                **outcome_analysis,
            },
            "future_target_vs_strongest_hidden_same_task_foil": {
                "observation_unit": "eligible_target_at_uniform_probeable_checkpoint",
                "selection": protocol["future_target"],
                "rows": detailed_future,
                "methods": future_summary,
            },
        },
        "decision_audit": {
            "all_split_and_support_rules_pass": all_support,
            "all_bootstrap_valid_fraction_rules_pass": bool(
                action_analysis["all_method_bootstrap_valid_fraction_rules_pass"]
                and outcome_analysis["all_method_bootstrap_valid_fraction_rules_pass"]
            ),
            "task_count": task_count,
            "selected_task_count": prompt_contract["selected_task_count"],
            "unprobed_task_count": len(prompt_contract["unprobed_task_ids"]),
            "minimum_tasks_for_non_development_claim": minimum_nondevelopment,
            "status": decision_status,
            "no_missing_fold_or_label_was_imputed": True,
            "evaluation_repositories_never_enter_fit_or_calibration": True,
            "fit_and_calibration_repositories_are_disjoint": True,
            "bootstrap_resamples_repositories_then_tasks_never_rows": True,
            "ordinary_logit_is_verified_identical_across_all_three_reports": True,
            "official_outcome_is_observed_once_per_task": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument("--nf4-report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {
        "prompts": args.prompts.expanduser().resolve(strict=True),
        "public_report": args.public_report.expanduser().resolve(strict=True),
        "nf4_report": args.nf4_report.expanduser().resolve(strict=True),
        "native_report": args.native_report.expanduser().resolve(strict=True),
        "protocol": args.protocol.expanduser().resolve(strict=True),
    }
    protocol_bytes = paths["protocol"].read_bytes()
    protocol_value = mapping(json.loads(protocol_bytes), "behavioral readout protocol")
    configured_samples = integer(
        mapping(protocol_value.get("bootstrap"), "bootstrap").get("samples"),
        "configured bootstrap samples",
    )
    bootstrap_samples = (
        configured_samples if args.bootstrap_samples is None else args.bootstrap_samples
    )
    require(bootstrap_samples >= 0, "bootstrap sample count must be nonnegative")
    analysis = build_analysis(
        json.loads(paths["prompts"].read_bytes()),
        mapping(json.loads(paths["public_report"].read_bytes()), "public report"),
        mapping(json.loads(paths["nf4_report"].read_bytes()), "NF4 report"),
        mapping(json.loads(paths["native_report"].read_bytes()), "native report"),
        protocol_value,
        protocol_sha256=sha256_bytes(protocol_bytes),
        input_hashes={
            key: sha256_file(path) for key, path in paths.items()
        },
        bootstrap_samples=bootstrap_samples,
    )
    output = args.output.expanduser().resolve()
    atomic_write_json(output, analysis)
    print(f"wrote {output} (sha256={sha256_file(output)}, status={analysis['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
