#!/usr/bin/env python3
"""Explore a future SWE milestone target on development trajectories only.

The target is the next known edit, validation, or finalization action after a
checkpoint, skipping known inspections.  Labels are intentionally fail-closed:
an unknown action before the candidate milestone censors the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = ROOT / "scripts" / "analyze_swe_task_state_interpreter.py"
TASK_STATE_READOUT_PATH = ROOT / "scripts" / "swe_task_state_readout.py"
MILESTONES = ("edit", "validate", "finalize")
VARIANTS = (
    "progress_only",
    "history_context",
    "public_jacobian",
    "jacobian_context",
    "ordinary_logit",
    "logit_context",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_analyzer():
    spec = importlib.util.spec_from_file_location("task_state_analyzer", ANALYZER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load task-state analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def action_of(prompt: dict) -> str | None:
    action = prompt["metadata"]["labels"]["action"]
    class_id = action.get("class_id")
    if action.get("status") != "available" or class_id not in {
        "inspect",
        "edit",
        "validate",
        "finalize",
    }:
        return None
    return str(class_id)


def milestone_labels(prompts: list[dict]) -> tuple[dict[str, dict], dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for prompt in prompts:
        by_task[prompt["metadata"]["task"]["instance_id"]].append(prompt)

    labels: dict[str, dict] = {}
    task_summary: dict[str, dict] = {}
    for task_id, task_prompts in sorted(by_task.items()):
        ordered = sorted(
            task_prompts,
            key=lambda row: row["metadata"]["selection"]["task_request_index"],
        )
        observed = [row["metadata"]["selection"]["task_request_index"] for row in ordered]
        declared = ordered[0]["metadata"]["task"]["probeable_request_indices"]
        complete = (
            declared == list(range(1, len(declared) + 1))
            and observed == declared
            and len(observed) == len(set(observed))
        )
        actions = [action_of(row) for row in ordered]
        task_counts: Counter[str] = Counter()
        for index, prompt in enumerate(ordered):
            record = {
                "status": "censored",
                "label": None,
                "reason": None,
                "horizon_requests": None,
                "target_request_index": None,
            }
            if not complete:
                record["reason"] = "incomplete_nonconsecutive_bundle"
            else:
                for future_index in range(index, len(actions)):
                    action = actions[future_index]
                    if action is None:
                        record["reason"] = "unknown_before_next_milestone"
                        break
                    if action == "inspect":
                        continue
                    if action in MILESTONES:
                        record.update(
                            {
                                "status": "available",
                                "label": action,
                                "reason": None,
                                "horizon_requests": future_index - index,
                                "target_request_index": observed[future_index],
                            }
                        )
                        break
                    raise AssertionError(f"unhandled action {action}")
                else:
                    record["reason"] = "no_observed_future_milestone"
            labels[prompt["id"]] = record
            task_counts[record["label"] or record["reason"]] += 1
        task_summary[task_id] = {
            "repo": ordered[0]["metadata"]["task"]["repo"],
            "request_count": len(ordered),
            "immediate_action_support": dict(
                sorted(Counter(action or "unknown" for action in actions).items())
            ),
            "milestone_assignment_support": dict(sorted(task_counts.items())),
            "complete_consecutive_bundle": complete,
        }
    return labels, task_summary


def simple_metrics(predictions: list[dict]) -> dict:
    support = Counter(row["label"] for row in predictions)
    correct = Counter(
        row["label"] for row in predictions if row["prediction"] == row["label"]
    )
    recalls = {
        label: correct[label] / support[label] if support[label] else None
        for label in MILESTONES
    }
    available = [value for value in recalls.values() if value is not None]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        by_task[row["task_id"]].append(row)
    task_bas = []
    task_accs = []
    for rows in by_task.values():
        task_support = Counter(row["label"] for row in rows)
        task_correct = Counter(
            row["label"] for row in rows if row["prediction"] == row["label"]
        )
        task_recalls = [
            task_correct[label] / task_support[label]
            for label in MILESTONES
            if task_support[label]
        ]
        task_bas.append(float(np.mean(task_recalls)))
        task_accs.append(sum(task_correct.values()) / len(rows))
    return {
        "row_count": len(predictions),
        "support": dict(support),
        "accuracy": sum(correct.values()) / len(predictions),
        "balanced_accuracy": float(np.mean(available)),
        "per_class_recall": recalls,
        "task_macro_accuracy": float(np.mean(task_accs)),
        "task_macro_balanced_accuracy_over_present_classes": float(np.mean(task_bas)),
    }


def _hierarchical_sample_pairs(
    paired: list[tuple[dict, dict]], rng: np.random.Generator
) -> list[tuple[dict, dict]]:
    by_repo_task: dict[str, dict[str, list[tuple[dict, dict]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pair in paired:
        by_repo_task[pair[0]["repo"]][pair[0]["task_id"]].append(pair)
    repos = sorted(by_repo_task)
    sampled_pairs: list[tuple[dict, dict]] = []
    for repository_draw, repo_index in enumerate(
        rng.integers(0, len(repos), len(repos))
    ):
        repo = repos[int(repo_index)]
        tasks = sorted(by_repo_task[repo])
        bootstrap_repo = f"bootstrap-repository-{repository_draw}"
        for task_draw, task_index in enumerate(
            rng.integers(0, len(tasks), len(tasks))
        ):
            task = tasks[int(task_index)]
            bootstrap_task = f"{bootstrap_repo}-task-{task_draw}"
            sampled_pairs.extend(
                (
                    {
                        **left,
                        "repo": bootstrap_repo,
                        "task_id": bootstrap_task,
                    },
                    {
                        **right,
                        "repo": bootstrap_repo,
                        "task_id": bootstrap_task,
                    },
                )
                for left, right in by_repo_task[repo][task]
            )
    return sampled_pairs


def paired_hierarchical_bootstrap(
    predictions_a: list[dict], predictions_b: list[dict], *, samples: int = 5000, seed: int = 91337
) -> dict:
    if [row["row_id"] for row in predictions_a] != [row["row_id"] for row in predictions_b]:
        raise RuntimeError("predictions are not paired")
    paired = list(zip(predictions_a, predictions_b, strict=True))
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        sampled_pairs = _hierarchical_sample_pairs(paired, rng)
        a = simple_metrics([pair[0] for pair in sampled_pairs])
        b = simple_metrics([pair[1] for pair in sampled_pairs])
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "task_macro_accuracy",
            "task_macro_balanced_accuracy_over_present_classes",
        ):
            draws[metric].append(a[metric] - b[metric])

    def summarize(values: list[float], point: float) -> dict:
        return {
            "point": point,
            "ci_lower": float(np.quantile(values, 0.025)),
            "ci_upper": float(np.quantile(values, 0.975)),
            "positive_draw_fraction": float(np.mean(np.asarray(values) > 0.0)),
        }

    metrics_a = simple_metrics(predictions_a)
    metrics_b = simple_metrics(predictions_b)
    return {
        metric: summarize(draws[metric], metrics_a[metric] - metrics_b[metric])
        for metric in draws
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--behavioral-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=VARIANTS,
    )
    args = parser.parse_args()

    analyzer = load_analyzer()
    prompts = json.loads(args.prompts.read_text())
    report = json.loads(args.report.read_text())
    protocol_value = json.loads(args.protocol.read_text())
    behavioral_value = json.loads(args.behavioral_protocol.read_text())
    hashes = {
        "prompts": sha256_file(args.prompts),
        "public_report": sha256_file(args.report),
        "protocol": sha256_file(args.protocol),
        "behavioral_protocol": sha256_file(args.behavioral_protocol),
    }
    implementation_hashes = {
        "milestone_explorer": sha256_file(Path(__file__).resolve()),
        "task_state_analyzer": sha256_file(ANALYZER_PATH),
        "task_state_readout": sha256_file(TASK_STATE_READOUT_PATH),
    }
    protocol = analyzer.validate_protocol(
        protocol_value,
        behavioral_protocol_value=behavioral_value,
        behavioral_protocol_sha256=hashes["behavioral_protocol"],
        prompt_sha256=hashes["prompts"],
        report_sha256=hashes["public_report"],
    )
    extracted = analyzer.extract_rows(prompts, report, protocol=protocol)
    labels, task_summary = milestone_labels(prompts)

    rows = []
    stable_censor_reasons: Counter[str] = Counter()
    horizons: list[int] = []
    for row in extracted["rows"]:
        assignment = labels[row["row_id"]]
        if assignment["status"] != "available":
            stable_censor_reasons[assignment["reason"]] += 1
            continue
        relabeled = dict(row)
        relabeled["label"] = assignment["label"]
        relabeled["milestone_horizon_requests"] = assignment["horizon_requests"]
        relabeled["milestone_target_request_index"] = assignment["target_request_index"]
        rows.append(relabeled)
        horizons.append(assignment["horizon_requests"])

    variants = tuple(args.variants)
    evaluations = {}
    for variant in variants:
        print(f"evaluating {variant}", flush=True)
        evaluations[variant] = analyzer.nested_repository_evaluation(
            rows,
            variant=variant,
            class_ids=MILESTONES,
            c_grid=protocol["model"]["c_grid"],
            maximum_iterations=protocol["model"]["maximum_iterations"],
            minimum_valid_inner_folds=protocol["evaluation"]["minimum_inner"],
            temperatures=protocol["calibration"]["temperatures"],
            thresholds=protocol["abstention"]["thresholds"],
            threshold_contract=protocol["abstention"],
            ece_bins=protocol["metrics"]["ece_bins"],
        )

    comparisons = {}
    for candidate, reference in (
        ("jacobian_context", "history_context"),
        ("logit_context", "history_context"),
        ("jacobian_context", "logit_context"),
        ("public_jacobian", "ordinary_logit"),
        ("public_jacobian", "progress_only"),
    ):
        if (
            candidate in evaluations
            and reference in evaluations
            and evaluations[candidate]["status"] == evaluations[reference]["status"] == "available"
        ):
            comparisons[f"{candidate}_minus_{reference}"] = paired_hierarchical_bootstrap(
                evaluations[candidate]["predictions"], evaluations[reference]["predictions"]
            )

    assigned_all = [record for record in labels.values() if record["status"] == "available"]
    stable_events = {
        (row["task_id"], row["milestone_target_request_index"], row["label"])
        for row in rows
    }
    result = {
        "scope": "development_only_posthoc_exploration_not_a_reliability_claim",
        "inputs": hashes,
        "implementation_provenance": {
            "sha256": implementation_hashes,
            "all_implementations_hash_bound": True,
        },
        "label_contract": {
            "classes_in_order": list(MILESTONES),
            "scan_starts_at_current_request_action": True,
            "inspect_actions_skipped": True,
            "unknown_before_candidate_milestone": "censor",
            "no_observed_future_milestone": "censor",
            "requires_complete_consecutive_request_bundle": True,
            "future_trajectory_label_not_available_online": True,
        },
        "all_prompt_assignment": {
            "prompt_count": len(prompts),
            "available_count": len(assigned_all),
            "class_support": dict(Counter(record["label"] for record in assigned_all)),
            "distinct_target_event_count": len(
                {
                    (
                        prompt["metadata"]["task"]["instance_id"],
                        labels[prompt["id"]]["target_request_index"],
                        labels[prompt["id"]]["label"],
                    )
                    for prompt in prompts
                    if labels[prompt["id"]]["status"] == "available"
                }
            ),
            "censor_reasons": dict(
                Counter(
                    record["reason"]
                    for record in labels.values()
                    if record["status"] != "available"
                )
            ),
        },
        "stable_replay_assignment": {
            "source_stable_row_count": len(extracted["rows"]),
            "eligible_row_count": len(rows),
            "task_count": len({row["task_id"] for row in rows}),
            "repository_count": len({row["repo"] for row in rows}),
            "class_support": dict(Counter(row["label"] for row in rows)),
            "distinct_target_event_count": len(stable_events),
            "distinct_target_events_per_class": {
                label: sum(event[2] == label for event in stable_events)
                for label in MILESTONES
            },
            "tasks_per_class": {
                label: len({row["task_id"] for row in rows if row["label"] == label})
                for label in MILESTONES
            },
            "repositories_per_class": {
                label: len({row["repo"] for row in rows if row["label"] == label})
                for label in MILESTONES
            },
            "censor_reasons": dict(stable_censor_reasons),
            "horizon_requests": {
                "zero_count": sum(value == 0 for value in horizons),
                "positive_count": sum(value > 0 for value in horizons),
                "median": float(np.median(horizons)),
                "p90": float(np.quantile(horizons, 0.9)),
                "max": max(horizons),
            },
        },
        "task_summary": task_summary,
        "evaluations": {
            variant: {
                "status": evaluation["status"],
                "row_count": evaluation["row_count"],
                "repository_count": evaluation["repository_count"],
                "full": evaluation["metrics"]["full"],
                "primary_task_equal": evaluation["metrics"]["primary_task_equal"],
                "selected_abstention": evaluation["metrics"]["selected_abstention"],
                "fixed_threshold_sweep": evaluation["metrics"]["fixed_threshold_sweep"],
            }
            for variant, evaluation in evaluations.items()
        },
        "paired_hierarchical_bootstrap": comparisons,
        "scientific_limitations": [
            "The target is posthoc and derived from the same trajectories used for development.",
            "The target asks what milestone eventually occurs next, not the immediate next action.",
            "Repository-held-out predictions share training data; bootstrap "
            "intervals condition on frozen predictions and omit fit/selection "
            "uncertainty.",
            "A fresh frozen cohort was not inspected.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
