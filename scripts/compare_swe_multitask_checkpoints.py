#!/usr/bin/env python3
"""Compare exact-rank task-start and C1 multi-task SWE J-lens analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_swe_multitask_initial_probes import (  # noqa: E402
    CHECKPOINT_CONTRACTS,
    FIXED_MIDDLE_LAYERS,
    LOGIT_VOCABULARY_SIZE,
    MODEL_REPO,
    MODEL_REVISION,
    TOKENIZER_JSON_SHA256,
)


METHODS = ("public_jacobian", "native_jacobian", "logit_lens")
ANALYSIS_KINDS = {
    "C0": "exploratory_swe_verified_multitask_initial_probe_analysis",
    "C1": (
        "exploratory_swe_verified_multitask_"
        "post_repository_observation_probe_analysis"
    ),
    "C0M": (
        "exploratory_swe_verified_multitask_"
        "capture_matched_initial_probe_analysis"
    ),
}
BOOTSTRAP_SEED = 36_127
BOOTSTRAP_SAMPLES = 20_000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return value


def finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be a nonempty string")
    return value


def valid_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalized_utility(rank: int) -> float:
    require(1 <= rank <= LOGIT_VOCABULARY_SIZE, "rank outside LM-head vocabulary")
    return math.log(LOGIT_VOCABULARY_SIZE / rank) / math.log(
        LOGIT_VOCABULARY_SIZE
    )


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    require(bool(ordered), "quantile input must not be empty")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def derived_seed(label: str, seed: int) -> int:
    digest = hashlib.sha256(label.encode("ascii")).hexdigest()
    return seed + int(digest[:8], 16)


def _validate_target_score(value: Any, label: str) -> dict[str, Any]:
    score = mapping(value, label)
    scorable = score.get("scorable")
    require(isinstance(scorable, bool), f"{label}.scorable must be boolean")
    if not scorable:
        require(
            score.get("minimum_rank") is None and score.get("utility_u") is None,
            f"{label} unscorable rank fields must be null",
        )
        return {"scorable": False, "minimum_rank": None, "utility_u": None}
    rank = score.get("minimum_rank")
    require(
        isinstance(rank, int)
        and not isinstance(rank, bool)
        and 1 <= rank <= LOGIT_VOCABULARY_SIZE,
        f"{label}.minimum_rank must be an exact full-vocabulary rank",
    )
    utility = finite(score.get("utility_u"), f"{label}.utility_u")
    require(
        close(utility, normalized_utility(rank)),
        f"{label}.utility_u is inconsistent with its exact rank",
    )
    best_layer = score.get("best_layer")
    require(
        best_layer in FIXED_MIDDLE_LAYERS,
        f"{label}.best_layer is outside the fixed middle band",
    )
    best_token_id = score.get("best_token_id")
    require(
        isinstance(best_token_id, int) and not isinstance(best_token_id, bool),
        f"{label}.best_token_id is invalid",
    )
    nonempty_string(score.get("best_token"), f"{label}.best_token")
    return {
        "scorable": True,
        "minimum_rank": rank,
        "utility_u": utility,
        "best_layer": best_layer,
        "best_token_id": best_token_id,
        "best_token": score["best_token"],
    }


def _validate_method(value: Any, *, label: str) -> dict[str, Any]:
    method = mapping(value, label)
    tasks_value = sequence(method.get("tasks"), f"{label}.tasks")
    require(bool(tasks_value), f"{label}.tasks must not be empty")
    require(method.get("task_count") == len(tasks_value), f"{label}.task_count mismatch")
    tasks: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for task_index, raw_task in enumerate(tasks_value):
        task_label = f"{label}.tasks[{task_index}]"
        task = mapping(raw_task, task_label)
        identifier = nonempty_string(task.get("id"), f"{task_label}.id")
        instance_id = nonempty_string(
            task.get("instance_id"), f"{task_label}.instance_id"
        )
        require(instance_id not in seen_tasks, f"{label} has duplicate task {instance_id}")
        seen_tasks.add(instance_id)
        repo = nonempty_string(task.get("repo"), f"{task_label}.repo")
        generated_token_id = task.get("generated_token_id")
        require(
            isinstance(generated_token_id, int)
            and not isinstance(generated_token_id, bool),
            f"{task_label}.generated_token_id is invalid",
        )
        families_value = sequence(task.get("families"), f"{task_label}.families")
        require(bool(families_value), f"{task_label}.families must not be empty")
        families: list[dict[str, Any]] = []
        seen_families: set[str] = set()
        scorable_family_utilities: list[float] = []
        for family_index, raw_family in enumerate(families_value):
            family_label = f"{task_label}.families[{family_index}]"
            family = mapping(raw_family, family_label)
            family_name = nonempty_string(
                family.get("family"), f"{family_label}.family"
            )
            require(
                family_name not in seen_families,
                f"{task_label} has duplicate family {family_name}",
            )
            seen_families.add(family_name)
            concepts_value = sequence(
                family.get("concepts"), f"{family_label}.concepts"
            )
            require(bool(concepts_value), f"{family_label}.concepts must not be empty")
            concepts: list[dict[str, Any]] = []
            seen_concepts: set[str] = set()
            scorable_utilities: list[float] = []
            for concept_index, raw_concept in enumerate(concepts_value):
                concept_label = f"{family_label}.concepts[{concept_index}]"
                concept = mapping(raw_concept, concept_label)
                concept_id = nonempty_string(
                    concept.get("id"), f"{concept_label}.id"
                )
                require(
                    concept_id not in seen_concepts,
                    f"{family_label} has duplicate concept {concept_id}",
                )
                seen_concepts.add(concept_id)
                target = nonempty_string(
                    concept.get("target"), f"{concept_label}.target"
                )
                path = nonempty_string(concept.get("path"), f"{concept_label}.path")
                score = _validate_target_score(
                    concept.get("target_score"), f"{concept_label}.target_score"
                )
                if score["scorable"]:
                    scorable_utilities.append(score["utility_u"])
                concepts.append(
                    {
                        "id": concept_id,
                        "family": family_name,
                        "target": target,
                        "path": path,
                        "score": score,
                    }
                )
            require(
                family.get("concept_count") == len(concepts),
                f"{family_label}.concept_count mismatch",
            )
            require(
                family.get("scorable_target_concept_count")
                == len(scorable_utilities),
                f"{family_label} scorable concept count mismatch",
            )
            if scorable_utilities:
                family_utility = finite(
                    family.get("target_utility_u"), f"{family_label}.target_utility_u"
                )
                require(
                    close(family_utility, statistics.fmean(scorable_utilities)),
                    f"{family_label}.target_utility_u weighting mismatch",
                )
                scorable_family_utilities.append(family_utility)
            families.append(
                {"family": family_name, "concepts": concepts}
            )
        require(bool(scorable_family_utilities), f"{task_label} has no scorable family")
        require(
            task.get("family_count") == len(scorable_family_utilities),
            f"{task_label}.family_count mismatch",
        )
        task_utility = finite(task.get("target_utility_u"), f"{task_label}.target_utility_u")
        require(
            close(task_utility, statistics.fmean(scorable_family_utilities)),
            f"{task_label}.target_utility_u weighting mismatch",
        )
        tasks.append(
            {
                "id": identifier,
                "instance_id": instance_id,
                "repo": repo,
                "generated_token_id": generated_token_id,
                "families": families,
                "target_utility_u": task_utility,
            }
        )
    overall_utility = finite(method.get("target_utility_u"), f"{label}.target_utility_u")
    require(
        close(overall_utility, statistics.fmean(task["target_utility_u"] for task in tasks)),
        f"{label}.target_utility_u task weighting mismatch",
    )
    return {"tasks": tasks, "target_utility_u": overall_utility}


def _identity_signature(method: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            task["id"],
            task["instance_id"],
            task["repo"],
            task["generated_token_id"],
            family["family"],
            concept["id"],
            concept["target"],
            concept["path"],
            concept["score"]["scorable"],
        )
        for task in method["tasks"]
        for family in task["families"]
        for concept in family["concepts"]
    ]


def validate_analysis(value: Any, *, checkpoint_id: str) -> dict[str, Any]:
    analysis = mapping(value, f"{checkpoint_id} analysis")
    require(
        analysis.get("schema_version") == 1
        and analysis.get("kind") == ANALYSIS_KINDS[checkpoint_id],
        f"{checkpoint_id} analysis identity mismatch",
    )
    model = mapping(analysis.get("model"), f"{checkpoint_id}.model")
    require(
        model.get("repo_id") == MODEL_REPO
        and model.get("revision") == MODEL_REVISION
        and model.get("tokenizer_json_sha256") == TOKENIZER_JSON_SHA256
        and model.get("logit_vocabulary_size") == LOGIT_VOCABULARY_SIZE,
        f"{checkpoint_id} model identity mismatch",
    )
    source = mapping(
        analysis.get("source_bindings"), f"{checkpoint_id}.source_bindings"
    )
    protocol_sha256 = valid_sha256(
        source.get("protocol_sha256"), f"{checkpoint_id} protocol hash"
    )
    trajectory_bindings: list[dict[str, str]] = []
    if checkpoint_id in {"C0M", "C1"}:
        raw_bindings = sequence(
            source.get("trajectory_bindings"),
            f"{checkpoint_id}.source_bindings.trajectory_bindings",
        )
        seen_instances: set[str] = set()
        for index, raw_binding in enumerate(raw_bindings):
            binding = mapping(raw_binding, f"{checkpoint_id} trajectory binding {index}")
            instance_id = nonempty_string(
                binding.get("instance_id"),
                f"{checkpoint_id} trajectory binding {index} instance",
            )
            require(
                instance_id not in seen_instances,
                f"{checkpoint_id} trajectory bindings contain duplicate tasks",
            )
            seen_instances.add(instance_id)
            trajectory_bindings.append(
                {
                    "instance_id": instance_id,
                    "capture_manifest_sha256": valid_sha256(
                        binding.get("capture_manifest_sha256"),
                        f"{checkpoint_id} trajectory binding {index} capture",
                    ),
                    "first_request_sha256": valid_sha256(
                        binding.get("first_request_sha256"),
                        f"{checkpoint_id} trajectory binding {index} first request",
                    ),
                    "second_request_sha256": valid_sha256(
                        binding.get("second_request_sha256"),
                        f"{checkpoint_id} trajectory binding {index} second request",
                    ),
                }
            )
    evaluation = mapping(analysis.get("evaluation"), f"{checkpoint_id}.evaluation")
    require(
        evaluation.get("checkpoint_metadata") == CHECKPOINT_CONTRACTS[checkpoint_id],
        f"{checkpoint_id} exact checkpoint contract mismatch",
    )
    require(
        evaluation.get("fixed_middle_layers") == list(FIXED_MIDDLE_LAYERS),
        f"{checkpoint_id} fixed middle layers mismatch",
    )
    require(
        evaluation.get("rank_reduction")
        == "minimum over predeclared eligible forms and fixed layers",
        f"{checkpoint_id} rank reduction contract mismatch",
    )
    methods_value = mapping(analysis.get("methods"), f"{checkpoint_id}.methods")
    require(
        set(methods_value) == set(METHODS),
        f"{checkpoint_id} method set mismatch",
    )
    methods = {
        method: _validate_method(
            methods_value[method], label=f"{checkpoint_id}.methods.{method}"
        )
        for method in METHODS
    }
    reference_signature = _identity_signature(methods[METHODS[0]])
    for method in METHODS[1:]:
        require(
            _identity_signature(methods[method]) == reference_signature,
            f"{checkpoint_id} concept identity grid differs across methods",
        )
    coverage = mapping(analysis.get("coverage"), f"{checkpoint_id}.coverage")
    require(
        coverage.get("task_count") == len(methods[METHODS[0]]["tasks"]),
        f"{checkpoint_id} coverage task count mismatch",
    )
    return {
        "model": dict(model),
        "protocol_sha256": protocol_sha256,
        "layers": list(FIXED_MIDDLE_LAYERS),
        "methods": methods,
        "trajectory_bindings": trajectory_bindings,
    }


def _task_index(method: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    tasks = method["tasks"]
    result = {task["instance_id"]: task for task in tasks}
    require(len(result) == len(tasks), "task index contains duplicate identities")
    return result


def _concept_index(task: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    concepts = {
        (family["family"], concept["id"]): concept
        for family in task["families"]
        for concept in family["concepts"]
    }
    expected = sum(len(family["concepts"]) for family in task["families"])
    require(len(concepts) == expected, f"task {task['instance_id']} concept keys collide")
    return concepts


def compare_method(
    c0_method: Mapping[str, Any],
    c1_method: Mapping[str, Any],
    *,
    method_name: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    c0_tasks = _task_index(c0_method)
    c1_tasks = _task_index(c1_method)
    per_concept: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    excluded_pairs: list[dict[str, Any]] = []
    for instance_id, c1_task in c1_tasks.items():
        c0_task = c0_tasks.get(instance_id)
        require(c0_task is not None, f"C1 retained task {instance_id} is absent from C0")
        require(
            c0_task["repo"] == c1_task["repo"],
            f"retained task {instance_id} repository changed between checkpoints",
        )
        c0_concepts = _concept_index(c0_task)
        family_pairs: dict[str, list[dict[str, Any]]] = {}
        for c1_family in c1_task["families"]:
            family = c1_family["family"]
            for c1_concept in c1_family["concepts"]:
                key = (family, c1_concept["id"])
                c0_concept = c0_concepts.get(key)
                require(
                    c0_concept is not None,
                    (
                        f"C1 retained concept {instance_id}/{family}/"
                        f"{c1_concept['id']} is absent from C0"
                    ),
                )
                require(
                    c0_concept["target"] == c1_concept["target"]
                    and c0_concept["path"] == c1_concept["path"],
                    (
                        f"retained concept identity changed: {instance_id}/"
                        f"{family}/{c1_concept['id']}"
                    ),
                )
                c0_score = c0_concept["score"]
                c1_score = c1_concept["score"]
                row: dict[str, Any] = {
                    "instance_id": instance_id,
                    "repo": c1_task["repo"],
                    "family": family,
                    "concept_id": c1_concept["id"],
                    "target": c1_concept["target"],
                    "path": c1_concept["path"],
                    "paired_scorable": c0_score["scorable"] and c1_score["scorable"],
                    "c0_minimum_rank": c0_score["minimum_rank"],
                    "c1_minimum_rank": c1_score["minimum_rank"],
                    "c0_best_layer": c0_score.get("best_layer"),
                    "c1_best_layer": c1_score.get("best_layer"),
                }
                if row["paired_scorable"]:
                    c0_rank = c0_score["minimum_rank"]
                    c1_rank = c1_score["minimum_rank"]
                    pair = {
                        "c0_utility_u": c0_score["utility_u"],
                        "c1_utility_u": c1_score["utility_u"],
                        "c1_minus_c0_utility_u": (
                            c1_score["utility_u"] - c0_score["utility_u"]
                        ),
                        "c0_log_rank": math.log(c0_rank),
                        "c1_log_rank": math.log(c1_rank),
                        "c1_minus_c0_log_rank": math.log(c1_rank) - math.log(c0_rank),
                        "c1_minus_c0_rank": c1_rank - c0_rank,
                    }
                    row.update(pair)
                    family_pairs.setdefault(family, []).append(pair)
                else:
                    if not c0_score["scorable"] and not c1_score["scorable"]:
                        row["exclusion_reason"] = "unscorable at both checkpoints"
                    elif not c0_score["scorable"]:
                        row["exclusion_reason"] = "unscorable at C0"
                    else:
                        row["exclusion_reason"] = "unscorable at C1"
                    excluded_pairs.append(
                        {
                            "instance_id": instance_id,
                            "family": family,
                            "concept_id": c1_concept["id"],
                            "reason": row["exclusion_reason"],
                        }
                    )
                per_concept.append(row)
        if not family_pairs:
            excluded_pairs.append(
                {
                    "instance_id": instance_id,
                    "reason": "task has no concepts scorable at both checkpoints",
                }
            )
            continue
        family_rows = []
        for family, pairs in family_pairs.items():
            family_rows.append(
                {
                    "family": family,
                    "paired_concept_count": len(pairs),
                    "c0_utility_u": statistics.fmean(pair["c0_utility_u"] for pair in pairs),
                    "c1_utility_u": statistics.fmean(pair["c1_utility_u"] for pair in pairs),
                    "c0_mean_log_rank": statistics.fmean(pair["c0_log_rank"] for pair in pairs),
                    "c1_mean_log_rank": statistics.fmean(pair["c1_log_rank"] for pair in pairs),
                }
            )
        c0_utility = statistics.fmean(family["c0_utility_u"] for family in family_rows)
        c1_utility = statistics.fmean(family["c1_utility_u"] for family in family_rows)
        c0_log_rank = statistics.fmean(
            family["c0_mean_log_rank"] for family in family_rows
        )
        c1_log_rank = statistics.fmean(
            family["c1_mean_log_rank"] for family in family_rows
        )
        task_rows.append(
            {
                "instance_id": instance_id,
                "repo": c1_task["repo"],
                "paired_family_count": len(family_rows),
                "paired_concept_count": sum(
                    family["paired_concept_count"] for family in family_rows
                ),
                "c0_utility_u": c0_utility,
                "c1_utility_u": c1_utility,
                "c1_minus_c0_utility_u": c1_utility - c0_utility,
                "c0_mean_log_rank": c0_log_rank,
                "c1_mean_log_rank": c1_log_rank,
                "c1_minus_c0_log_rank": c1_log_rank - c0_log_rank,
                "families": family_rows,
            }
        )
    require(bool(task_rows), f"{method_name} has no paired-scorable retained C1 tasks")

    utility_bootstrap = paired_task_bootstrap(
        [task["c1_minus_c0_utility_u"] for task in task_rows],
        label=f"{method_name}:utility",
        seed=seed,
        samples=samples,
        direction="higher_is_better",
    )
    log_bootstrap = paired_task_bootstrap(
        [task["c1_minus_c0_log_rank"] for task in task_rows],
        label=f"{method_name}:log-rank",
        seed=seed,
        samples=samples,
        direction="lower_is_better",
    )
    c0_utility = statistics.fmean(task["c0_utility_u"] for task in task_rows)
    c1_utility = statistics.fmean(task["c1_utility_u"] for task in task_rows)
    c0_log_rank = statistics.fmean(task["c0_mean_log_rank"] for task in task_rows)
    c1_log_rank = statistics.fmean(task["c1_mean_log_rank"] for task in task_rows)
    log_interval = log_bootstrap["confidence_interval"]
    c0_gmean = math.exp(c0_log_rank)
    c1_gmean = math.exp(c1_log_rank)
    return {
        "matching": {
            "c0_available_task_count": len(c0_tasks),
            "c1_retained_task_count": len(c1_tasks),
            "paired_metric_task_count": len(task_rows),
            "retained_c1_concept_count": len(per_concept),
            "paired_scorable_concept_count": sum(
                row["paired_scorable"] for row in per_concept
            ),
            "unscorable_exclusions": excluded_pairs,
        },
        "normalized_utility_u": {
            "c0_task_macro": c0_utility,
            "c1_task_macro": c1_utility,
            "c1_minus_c0": c1_utility - c0_utility,
            "paired_task_bootstrap": utility_bootstrap,
        },
        "mean_log_rank": {
            "c0_task_macro": c0_log_rank,
            "c1_task_macro": c1_log_rank,
            "c1_minus_c0": c1_log_rank - c0_log_rank,
            "paired_task_bootstrap": log_bootstrap,
        },
        "geometric_mean_rank": {
            "c0_task_macro": c0_gmean,
            "c1_task_macro": c1_gmean,
            "c1_to_c0_ratio": c1_gmean / c0_gmean,
            "ratio_confidence_interval": {
                "confidence_level": 0.95,
                "lower": math.exp(log_interval["lower"]),
                "upper": math.exp(log_interval["upper"]),
                "derived_from": "paired task bootstrap of C1-minus-C0 mean log rank",
            },
        },
        "tasks": task_rows,
        "per_concept_ranks": per_concept,
    }


def paired_task_bootstrap(
    differences: Sequence[float],
    *,
    label: str,
    seed: int,
    samples: int,
    direction: str,
) -> dict[str, Any]:
    require(samples > 0, "bootstrap sample count must be positive")
    require(bool(differences), "paired task bootstrap requires at least one task")
    values = [finite(value, "task difference") for value in differences]
    metric_seed = derived_seed(label, seed)
    generator = random.Random(metric_seed)
    draws = [
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(samples)
    ]
    return {
        "method": "deterministic paired task-level percentile bootstrap",
        "sampling_unit": "retained SWE-Verified task",
        "direction": direction,
        "seed": metric_seed,
        "samples": samples,
        "task_count": len(values),
        "estimate": statistics.fmean(values),
        "confidence_interval": {
            "confidence_level": 0.95,
            "lower": quantile(draws, 0.025),
            "upper": quantile(draws, 0.975),
        },
        "positive_task_count": sum(value > 0.0 for value in values),
        "negative_task_count": sum(value < 0.0 for value in values),
        "tie_task_count": sum(value == 0.0 for value in values),
    }


def compare_stage_method_contrasts(
    methods: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    contrasts = {
        "public_minus_logit": ("public_jacobian", "logit_lens"),
        "native_minus_logit": ("native_jacobian", "logit_lens"),
        "native_minus_public": ("native_jacobian", "public_jacobian"),
    }
    result: dict[str, Any] = {}
    for label, (left_name, right_name) in contrasts.items():
        left = {task["instance_id"]: task for task in methods[left_name]["tasks"]}
        right = {task["instance_id"]: task for task in methods[right_name]["tasks"]}
        require(
            set(left) == set(right),
            f"{label} stage contrast has different paired task coverage",
        )
        task_rows = []
        for instance_id in left:
            utility = (
                left[instance_id]["c1_minus_c0_utility_u"]
                - right[instance_id]["c1_minus_c0_utility_u"]
            )
            log_rank = (
                left[instance_id]["c1_minus_c0_log_rank"]
                - right[instance_id]["c1_minus_c0_log_rank"]
            )
            task_rows.append(
                {
                    "instance_id": instance_id,
                    "c1_minus_c0_utility_difference": utility,
                    "c1_minus_c0_log_rank_difference": log_rank,
                }
            )
        result[label] = {
            "left_method": left_name,
            "right_method": right_name,
            "estimand": (
                "(C1 minus baseline) left-method change minus "
                "(C1 minus baseline) right-method change"
            ),
            "normalized_utility_u": paired_task_bootstrap(
                [row["c1_minus_c0_utility_difference"] for row in task_rows],
                label=f"stage-contrast:{label}:utility",
                seed=seed,
                samples=samples,
                direction="higher_is_better_for_left_method",
            ),
            "mean_log_rank": paired_task_bootstrap(
                [row["c1_minus_c0_log_rank_difference"] for row in task_rows],
                label=f"stage-contrast:{label}:log-rank",
                seed=seed,
                samples=samples,
                direction="lower_is_better_for_left_method",
            ),
            "tasks": task_rows,
        }
    return result


def compare(
    c0_value: Mapping[str, Any],
    c1_value: Mapping[str, Any],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    c0_checkpoint_id: str = "C0",
) -> dict[str, Any]:
    require(bootstrap_samples > 0, "bootstrap sample count must be positive")
    require(c0_checkpoint_id in {"C0", "C0M"}, "unsupported baseline checkpoint")
    c0 = validate_analysis(c0_value, checkpoint_id=c0_checkpoint_id)
    c1 = validate_analysis(c1_value, checkpoint_id="C1")
    require(c0["model"] == c1["model"], "C0 and C1 model contracts differ")
    require(
        c0["protocol_sha256"] == c1["protocol_sha256"],
        "C0 and C1 protocol SHA-256 bindings differ",
    )
    require(c0["layers"] == c1["layers"], "C0 and C1 fixed layer bands differ")
    if c0_checkpoint_id == "C0M":
        require(
            c0["trajectory_bindings"] == c1["trajectory_bindings"],
            "C0M and C1 trajectory request bindings differ",
        )
    methods = {
        method: compare_method(
            c0["methods"][method],
            c1["methods"][method],
            method_name=method,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        )
        for method in METHODS
    }
    reference_pairs = [
        (
            row["instance_id"],
            row["family"],
            row["concept_id"],
            row["paired_scorable"],
        )
        for row in methods[METHODS[0]]["per_concept_ranks"]
    ]
    for method in METHODS[1:]:
        require(
            [
                (
                    row["instance_id"],
                    row["family"],
                    row["concept_id"],
                    row["paired_scorable"],
                )
                for row in methods[method]["per_concept_ranks"]
            ]
            == reference_pairs,
            "paired C0/C1 identity coverage differs across methods",
        )
    stage_method_contrasts = compare_stage_method_contrasts(
        methods, seed=bootstrap_seed, samples=bootstrap_samples
    )
    return {
        "schema_version": 1,
        "kind": "exploratory_swe_verified_multitask_checkpoint_comparison",
        "label": (
            "EXPLORATORY STAGE-EMERGENCE COMPARISON: exact-rank association "
            f"change from {c0_checkpoint_id} to C1, not causal evidence"
        ),
        "contract": {
            "c0_checkpoint": CHECKPOINT_CONTRACTS[c0_checkpoint_id],
            "c1_checkpoint": CHECKPOINT_CONTRACTS["C1"],
            "trajectory_request_bindings_match": c0_checkpoint_id == "C0M",
            "protocol_sha256": c0["protocol_sha256"],
            "model": c0["model"],
            "fixed_middle_layers": c0["layers"],
            "matching_policy": (
                "C1-retained task/family/concept identities matched into the "
                "declared task-start baseline; baseline-only identities ignored"
            ),
            "rank_source": "target_score.minimum_rank from validated analysis JSON",
            "weighting": (
                "equal paired concepts within family, equal families within task, "
                "equal retained tasks overall"
            ),
        },
        "statistics": {
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_samples": bootstrap_samples,
            "sampling_unit": "retained SWE-Verified task",
        },
        "methods": methods,
        "stage_method_contrasts": stage_method_contrasts,
        "claims_gate": None,
        "limitations": [
            "This is an exploratory stage-emergence comparison, not a causal intervention.",
            "Minimum rank over forms and layers is an optimistic associative readout.",
            "Complete-case metrics exclude retained concepts unscorable at either checkpoint.",
        ],
    }


def read_json(path: Path) -> tuple[Any, dict[str, Any]]:
    raw = path.read_bytes()
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        display_path = f"external/{resolved.name}"
    return json.loads(raw), {
        "path": display_path,
        "size_bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-analysis", type=Path, required=True)
    parser.add_argument(
        "--c0-checkpoint", choices=("C0", "C0M"), default="C0"
    )
    parser.add_argument("--c1-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    c0, c0_source = read_json(args.c0_analysis)
    c1, c1_source = read_json(args.c1_analysis)
    require(
        isinstance(c0, dict) and isinstance(c1, dict),
        "C0 and C1 analysis JSON must be objects",
    )
    result = compare(
        c0,
        c1,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        c0_checkpoint_id=args.c0_checkpoint,
    )
    result["inputs"] = {"c0_analysis": c0_source, "c1_analysis": c1_source}
    atomic_write_json(args.output, result)
    for method in METHODS:
        utility = result["methods"][method]["normalized_utility_u"]
        interval = utility["paired_task_bootstrap"]["confidence_interval"]
        print(
            f"{method} C1-{args.c0_checkpoint} task-u: "
            f"{utility['c1_minus_c0']:+.6f} "
            f"[{interval['lower']:+.6f}, {interval['upper']:+.6f}]"
        )
    for label, contrast in result["stage_method_contrasts"].items():
        utility = contrast["normalized_utility_u"]
        interval = utility["confidence_interval"]
        print(
            f"{label} stage difference: {utility['estimate']:+.6f} "
            f"[{interval['lower']:+.6f}, {interval['upper']:+.6f}]"
        )
    print("claims gate: NONE (exploratory stage-emergence comparison)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
