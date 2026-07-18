#!/usr/bin/env python3
"""Focused tests for task-held-out behavioral J-lens analysis."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYZE = load_module(
    "analyze_swe_behavioral_probes",
    ROOT / "scripts" / "analyze_swe_behavioral_probes.py",
)
PROTOCOL_BYTES = (ROOT / "configs" / "swe_behavioral_readout_protocol.json").read_bytes()
PROTOCOL_VALUE = json.loads(PROTOCOL_BYTES)
PROTOCOL = ANALYZE.validate_protocol(
    PROTOCOL_VALUE, protocol_sha256=ANALYZE.sha256_bytes(PROTOCOL_BYTES)
)


def feature_rows(class_ids: list[str], repositories: int = 8):
    rows = []
    for repository_index in range(repositories):
        for class_index, class_id in enumerate(class_ids):
            scores = [-0.25] * len(class_ids)
            scores[class_index] = 1.0 + repository_index * 0.01
            rows.append(
                {
                    "row_id": f"repo-{repository_index}-row-{class_id}",
                    "task_id": f"project{repository_index}__task-1",
                    "repo": f"project{repository_index}/repo",
                    "label": class_id,
                    "scores": scores,
                }
            )
    return rows


def behavioral_prompt(*, foil_task: str = "project__task-1", target_exposed=False):
    fixed_tokens = [
        token
        for family in (PROTOCOL["action_classes"], PROTOCOL["outcome_classes"])
        for record in family
        for token in record["tokens"]
    ]
    text = "exact behavioral prompt"
    token_ids = [11, 12, 13]
    prompt = {
        "id": "behavioral-000-project__task-1-q00",
        "text": text,
        "token_ids": token_ids,
        "score_token_ids": [100, 101]
        + [token["token_id"] for token in fixed_tokens],
        "metadata": {
            "kind": "swe_verified_behavioral_probe",
            "schema_version": 1,
            "campaign_sha256": "c" * 64,
            "action_protocol_sha256": PROTOCOL["action_protocol_sha256"],
            "selection": {
                "cohort": "uniform_probeable_request_index",
                "algorithm": "uniform_probeable_request_indices_v1",
                "max_checkpoints": 8,
                "max_prompt_tokens": 65535,
                "task_request_index": 1,
                "global_request_index": 1,
                "checkpoint_ordinal": 0,
                "checkpoint_count": 1,
                "candidate_ordinal": 1,
                "candidate_ordinal_base": 1,
                "candidate_count": 1,
                "probeable_request_indices": [1],
                "excluded_request_indices": [],
                "primary_for_action_evaluation": True,
                "independent_of_next_action_label": True,
            },
            "task": {
                "selection_index": 0,
                "instance_id": "project__task-1",
                "repo": "project/repo",
                "base_commit": "a" * 40,
                "request_count": 1,
                "probeable_request_count": 1,
                "probeable_request_indices": [1],
                "excluded_request_indices": [],
            },
            "labels": {
                "action": {
                    "status": "available",
                    "class_id": "inspect",
                    "derivation": "synthetic",
                },
                "tool_execution": {
                    "status": "available",
                    "class_id": "success",
                    "derivation": "synthetic",
                },
                "validation": {
                    "status": "not_applicable",
                    "class_id": None,
                    "derivation": "synthetic",
                },
                "terminal": {
                    "finish_reason": "stop",
                    "is_terminal": True,
                    "is_terminal_completion": True,
                    "is_episode_endpoint": True,
                    "is_probeable_endpoint": True,
                },
                "official_outcome": {
                    "status": "available",
                    "class_id": "success",
                    "verdict": "resolved",
                    "derivation": "bound_official_swe_bench_aggregate",
                    "official_outcomes_path": "official_score/official_outcomes.json",
                    "official_outcomes_sha256": "6" * 64,
                    "outcome_record_sha256": "7" * 64,
                },
            },
            "targets": [
                {
                    "id": "target-1",
                    "task_instance_id": "project__task-1",
                    "kind": "future_diagnosis",
                    "target": "target",
                    "forms": [{"text": " target", "token_id": 100}],
                    "aliases": ["target"],
                    "future_support": {
                        "request_index": 2,
                        "sha256": "f" * 64,
                        "benchmark_gold_used": False,
                        "lens_output_used": False,
                    },
                    "foils": [
                        {
                            "id": "foil-1",
                            "task_instance_id": foil_task,
                            "kind": "future_diagnosis",
                            "target": "foil",
                            "forms": [{"text": " foil", "token_id": 101}],
                            "aliases": ["foil"],
                            "source": {"kind": "same_task_alternative"},
                        }
                    ],
                }
            ],
            "target_eligibility": [
                {
                    "target_id": "target-1",
                    "target_exposed": target_exposed,
                    "retained_hidden_foil_ids": ["foil-1"],
                    "excluded_foils": [],
                    "status": "eligible",
                }
            ],
            "provenance": {
                "raw_request_path": "validation/capture/chat_0001.json",
                "raw_request_sha256": "1" * 64,
                "usage_path": "validation/capture/usage.jsonl",
                "usage_sha256": "2" * 64,
                "usage_record_sha256": "3" * 64,
                "official_outcomes": {
                    "status": "available",
                    "path": "official_score/official_outcomes.json",
                    "sha256": "6" * 64,
                    "outcome_record_sha256": "7" * 64,
                },
                "runner_metadata_path": "validation/capture/runner.json",
                "runner_metadata_sha256": "4" * 64,
                "generated_patch_path": "validation/capture/patch.diff",
                "generated_patch_sha256": "5" * 64,
                "rendered_prompt_sha256": ANALYZE.sha256_text(text),
                "token_ids_sha256": ANALYZE.sha256_json(token_ids),
                "prompt_token_count": len(token_ids),
                "next_completion": {"synthetic": True},
                "prompt_record_payload_sha256": "0" * 64,
            },
        },
    }
    prompt["metadata"]["provenance"]["prompt_record_payload_sha256"] = (
        ANALYZE._prompt_payload_hash(prompt)
    )
    return prompt


def uniform_task_prompts(
    request_indices: list[int], *, probeable_request_indices: list[int] | None = None
):
    candidates = probeable_request_indices or list(range(1, 11))
    excluded = [index for index in range(1, 11) if index not in candidates]
    prompts = []
    for ordinal, request_index in enumerate(request_indices):
        prompt = behavioral_prompt()
        prompt["id"] = f"behavioral-000-project__task-1-q{ordinal:02d}"
        prompt["metadata"]["task"].update(
            {
                "request_count": 10,
                "probeable_request_count": len(candidates),
                "probeable_request_indices": candidates,
                "excluded_request_indices": excluded,
            }
        )
        selection = prompt["metadata"]["selection"]
        selection["task_request_index"] = request_index
        selection["global_request_index"] = request_index
        selection["checkpoint_ordinal"] = ordinal
        selection["checkpoint_count"] = len(request_indices)
        selection["candidate_ordinal"] = candidates.index(request_index) + 1
        selection["candidate_count"] = len(candidates)
        selection["probeable_request_indices"] = candidates
        selection["excluded_request_indices"] = excluded
        terminal = prompt["metadata"]["labels"]["terminal"]
        terminal["finish_reason"] = "stop" if request_index == 10 else "tool_calls"
        terminal["is_terminal"] = request_index == 10
        terminal["is_terminal_completion"] = request_index == 10
        terminal["is_episode_endpoint"] = request_index == 10
        terminal["is_probeable_endpoint"] = request_index == candidates[-1]
        prompt["metadata"]["provenance"]["prompt_record_payload_sha256"] = (
            ANALYZE._prompt_payload_hash(prompt)
        )
        prompts.append(prompt)
    return prompts


def numerical_experiment():
    top_ids = [7, 8, 9, 10, 11]
    readout = {
        "token_ids": top_ids,
        "target_token_id": 7,
        "target_rank": 1,
    }
    return {
        "final_layer_top1_matches_greedy": True,
        "final_model_readout": [copy.deepcopy(readout)],
        "captured_final_model_readout": [copy.deepcopy(readout)],
        "final_norm_reconstruction": {
            "max_abs_error": 0.1,
            "rms_error": 0.005,
            "reference_rms": 2.0,
            "relative_rms_error": 0.0025,
            "max_abs_tolerance": 0.125,
            "rms_tolerance": 0.006,
            "within_tolerance": True,
        },
        "final_logits_reconstruction": {
            "max_abs_error": 0.05,
            "rms_error": 0.009,
            "max_abs_tolerance": 0.0625,
            "rms_tolerance": 0.01,
            "top_k_prefix": 5,
            "top_k_prefix_token_ids_match": True,
            "within_tolerance": True,
        },
        "residual_capture_manifest": {
            "algorithm": ANALYZE.RESIDUAL_MANIFEST_ALGORITHM,
            "sha256": "a" * 64,
            "tensor_count": 64,
            "logical_bytes": 64 * 5120 * 4,
            "token_positions": [2],
        },
    }


def combined_prompt(index: int):
    prompt = behavioral_prompt()
    campaign_sha256 = ("c" if index == 0 else "d") * 64
    task_id = f"project{index}__task-1"
    manifest_sha256 = "e" * 64
    prompt["id"] = f"swe-behavioral-cohort{index}-{campaign_sha256[:8]}-source-{index}"
    metadata = prompt["metadata"]
    metadata["campaign_sha256"] = campaign_sha256
    metadata["task"].update(
        {
            "selection_index": index,
            "source_selection_index": 0,
            "instance_id": task_id,
            "repo": f"project{index}/repo",
            "base_commit": str(index + 1) * 40,
        }
    )
    metadata["selection"].update(
        {
            "global_request_index": index + 1,
            "source_global_request_index": 1,
        }
    )
    metadata["targets"][0]["task_instance_id"] = task_id
    metadata["targets"][0]["foils"][0]["task_instance_id"] = task_id
    metadata["cohort"] = {
        "index": index,
        "id": f"cohort{index}",
        "campaign_sha256": campaign_sha256,
        "source_run_id": f"run-{index:020x}",
        "source_run_label": f"run-{index}",
        "source_summary_sha256": ("a" if index == 0 else "b") * 64,
        "source_task_count": 1,
        "source_task_instance_ids": [task_id],
        "source_global_request_count": 1,
        "source_prompt_count": 1,
        "global_request_offset": index,
        "task_offset": index,
        "cohort_manifest_sha256": manifest_sha256,
    }
    metadata["provenance"]["combination"] = {
        "source_prompt_id": f"source-{index}",
        "source_prompt_record_payload_sha256": ("8" if index == 0 else "9") * 64,
        "source_campaign_global_request_index": 1,
        "combined_global_request_index": index + 1,
        "cohort_manifest_sha256": manifest_sha256,
    }
    metadata["provenance"]["prompt_record_payload_sha256"] = (
        ANALYZE._prompt_payload_hash(prompt)
    )
    return prompt


class BehavioralPromptTests(unittest.TestCase):
    def test_valid_same_task_hidden_foil(self):
        result = ANALYZE.validate_prompt_bundle([behavioral_prompt()], protocol=PROTOCOL)
        self.assertEqual(result["task_count"], 1)
        self.assertEqual(result["primary_prompt_count"], 1)
        self.assertEqual(result["lifecycle_checkpoint_flags"], [])

    def test_terminal_missing_states_and_absent_patch_are_not_imputed(self):
        prompt = behavioral_prompt()
        prompt["metadata"]["labels"]["tool_execution"] = {
            "status": "not_applicable",
            "class_id": None,
            "derivation": "terminal_completion_has_no_tool_execution",
        }
        prompt["metadata"]["labels"]["official_outcome"] = {
            "status": "missing",
            "class_id": None,
            "verdict": None,
            "derivation": "official_outcome_aggregate_absent",
            "official_outcomes_path": None,
            "official_outcomes_sha256": None,
            "outcome_record_sha256": None,
        }
        provenance = prompt["metadata"]["provenance"]
        provenance["generated_patch_path"] = None
        provenance["generated_patch_sha256"] = None
        provenance["official_outcomes"] = {
            "status": "missing",
            "path": None,
            "sha256": None,
            "outcome_record_sha256": None,
        }
        provenance["prompt_record_payload_sha256"] = ANALYZE._prompt_payload_hash(prompt)
        result = ANALYZE.validate_prompt_bundle([prompt], protocol=PROTOCOL)
        row = result["prompts"][0]
        self.assertIsNone(row["tool_execution_label"])
        self.assertEqual(row["official_outcome_status"], "missing")

    def test_missing_usage_endpoint_labels_are_retained_without_imputation(self):
        prompt = behavioral_prompt()
        labels = prompt["metadata"]["labels"]
        labels["action"] = {
            "status": "missing",
            "class_id": None,
            "derivation": "no_usage_and_no_following_capture",
        }
        labels["tool_execution"] = {
            "status": "missing",
            "class_id": None,
            "derivation": "no_usage_and_no_following_capture",
        }
        labels["validation"] = {
            "status": "missing",
            "class_id": None,
            "derivation": "no_usage_and_no_following_capture",
        }
        labels["terminal"] = {
            "finish_reason": None,
            "is_terminal": False,
            "is_terminal_completion": False,
            "is_episode_endpoint": True,
            "is_probeable_endpoint": True,
        }
        prompt["metadata"]["provenance"]["prompt_record_payload_sha256"] = (
            ANALYZE._prompt_payload_hash(prompt)
        )
        result = ANALYZE.validate_prompt_bundle([prompt], protocol=PROTOCOL)
        row = result["prompts"][0]
        self.assertEqual(row["action_status"], "missing")
        self.assertIsNone(row["tool_execution_label"])
        self.assertIsNone(row["validation_label"])

    def test_nonbinary_official_scorer_state_is_missing_never_failure(self):
        prompt = behavioral_prompt()
        official = prompt["metadata"]["labels"]["official_outcome"]
        official.update({"verdict": "error", "class_id": "failure"})
        prompt["metadata"]["provenance"]["prompt_record_payload_sha256"] = (
            ANALYZE._prompt_payload_hash(prompt)
        )
        with self.assertRaisesRegex(ValueError, "aggregate label is inconsistent"):
            ANALYZE.validate_prompt_bundle([prompt], protocol=PROTOCOL)

        official.update(
            {
                "status": "missing",
                "class_id": None,
                "verdict": "error",
                "derivation": "official_nonbinary_infrastructure_or_empty_outcome",
            }
        )
        prompt["metadata"]["provenance"]["prompt_record_payload_sha256"] = (
            ANALYZE._prompt_payload_hash(prompt)
        )
        result = ANALYZE.validate_prompt_bundle([prompt], protocol=PROTOCOL)
        self.assertEqual(result["prompts"][0]["official_outcome_status"], "missing")
        self.assertIsNone(result["prompts"][0]["official_outcome"])

    def test_protocol_labels_pooled_scope_and_fixed_foil(self):
        self.assertEqual(
            PROTOCOL["probe_vs_refit"]["claim_scope"],
            "pooled_predeclared_development_plus_replication_repository_crossfit",
        )
        self.assertEqual(
            PROTOCOL["probe_vs_refit"]["replication_interpretation"],
            "cohort_subgroups_descriptive_only_not_an_independent_replication_test",
        )
        self.assertEqual(
            PROTOCOL["future_target"]["foil_reduction"],
            "fixed_hidden_foil_by_seeded_sha256_v1",
        )
        self.assertEqual(
            PROTOCOL["official_outcome"]["available_verdict_mapping"],
            {"resolved": "success", "unresolved": "failure"},
        )
        self.assertEqual(
            PROTOCOL["crossfit"]["majority_baseline"]["kind"],
            "fit_checkpoint_class_prior",
        )
        self.assertEqual(
            PROTOCOL["probe_vs_refit"]["next_action_estimand"][
                "point_estimate_weighting"
            ],
            "one_equal_weight_per_checkpoint_row",
        )
        legacy = copy.deepcopy(PROTOCOL_VALUE)
        legacy["analysis_version"] = "task-held-out-v1"
        with self.assertRaisesRegex(ValueError, "version/selection contract"):
            ANALYZE.validate_protocol(legacy, protocol_sha256="a" * 64)

    def test_cross_task_hidden_foil_fails_closed(self):
        prompt = behavioral_prompt(foil_task="other__task-2")
        prompt["metadata"]["provenance"]["prompt_record_payload_sha256"] = (
            ANALYZE._prompt_payload_hash(prompt)
        )
        with self.assertRaisesRegex(ValueError, "cross-task foil"):
            ANALYZE.validate_prompt_bundle([prompt], protocol=PROTOCOL)

    def test_exposed_target_cannot_be_eligible(self):
        with self.assertRaisesRegex(ValueError, "eligible target is exposed"):
            ANALYZE.validate_prompt_bundle(
                [behavioral_prompt(target_exposed=True)], protocol=PROTOCOL
            )

    def test_prompt_payload_hash_is_enforced(self):
        prompt = behavioral_prompt()
        prompt["text"] += " mutated"
        prompt["metadata"]["provenance"]["rendered_prompt_sha256"] = ANALYZE.sha256_text(
            prompt["text"]
        )
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            ANALYZE.validate_prompt_bundle([prompt], protocol=PROTOCOL)

    def test_prompt_token_count_must_leave_one_generation_token(self):
        prompt = behavioral_prompt()
        prompt["token_ids"] = [11] * 65535
        prompt["metadata"]["provenance"]["token_ids_sha256"] = ANALYZE.sha256_json(
            prompt["token_ids"]
        )
        prompt["metadata"]["provenance"]["prompt_token_count"] = len(
            prompt["token_ids"]
        )
        prompt["metadata"]["provenance"]["prompt_record_payload_sha256"] = (
            ANALYZE._prompt_payload_hash(prompt)
        )
        ANALYZE.validate_prompt_bundle([prompt], protocol=PROTOCOL)

        prompt["token_ids"].append(11)
        prompt["metadata"]["provenance"]["token_ids_sha256"] = ANALYZE.sha256_json(
            prompt["token_ids"]
        )
        prompt["metadata"]["provenance"]["prompt_token_count"] = len(
            prompt["token_ids"]
        )
        prompt["metadata"]["provenance"]["prompt_record_payload_sha256"] = (
            ANALYZE._prompt_payload_hash(prompt)
        )
        with self.assertRaisesRegex(ValueError, "replayable prompt-token limit"):
            ANALYZE.validate_prompt_bundle([prompt], protocol=PROTOCOL)

    def test_probeable_and_excluded_indices_must_partition_raw_requests(self):
        prompt = behavioral_prompt()
        prompt["metadata"]["task"]["excluded_request_indices"] = [1]
        prompt["metadata"]["selection"]["excluded_request_indices"] = [1]
        prompt["metadata"]["provenance"]["prompt_record_payload_sha256"] = (
            ANALYZE._prompt_payload_hash(prompt)
        )
        with self.assertRaisesRegex(ValueError, "request partition is invalid"):
            ANALYZE.validate_prompt_bundle([prompt], protocol=PROTOCOL)

    def test_primary_checkpoints_must_match_exact_probeable_quantiles(self):
        expected = [1, 2, 3, 4, 6, 7, 8, 10]
        result = ANALYZE.validate_prompt_bundle(
            uniform_task_prompts(expected), protocol=PROTOCOL
        )
        self.assertEqual(result["primary_prompt_count"], len(expected))

        nonuniform = uniform_task_prompts([1, 2, 3, 4, 5, 7, 8, 10])
        with self.assertRaisesRegex(ValueError, "exact uniform quantiles"):
            ANALYZE.validate_prompt_bundle(nonuniform, protocol=PROTOCOL)

        candidates = [1, 2, 3, 4, 6, 7, 8, 9, 10]
        excluded = uniform_task_prompts(
            expected, probeable_request_indices=candidates
        )
        result = ANALYZE.validate_prompt_bundle(excluded, protocol=PROTOCOL)
        self.assertEqual(
            [row["task_request_index"] for row in result["prompts"]], expected
        )
        self.assertNotIn(5, [row["task_request_index"] for row in result["prompts"]])

    def test_combined_campaigns_require_and_validate_cohort_offsets(self):
        prompts = [combined_prompt(0), combined_prompt(1)]
        result = ANALYZE.validate_prompt_bundle(prompts, protocol=PROTOCOL)
        self.assertEqual(result["campaign_count"], 2)
        self.assertEqual(result["task_count"], 2)
        self.assertEqual(result["selected_task_count"], 2)
        self.assertEqual(result["unprobed_task_ids"], [])
        self.assertEqual(
            [prompt["cohort_id"] for prompt in result["prompts"]],
            ["cohort0", "cohort1"],
        )

        bad_offset = [combined_prompt(0), combined_prompt(1)]
        bad_offset[1]["metadata"]["selection"]["global_request_index"] = 3
        bad_offset[1]["metadata"]["provenance"]["combination"][
            "combined_global_request_index"
        ] = 3
        bad_offset[1]["metadata"]["provenance"]["prompt_record_payload_sha256"] = (
            ANALYZE._prompt_payload_hash(bad_offset[1])
        )
        with self.assertRaisesRegex(ValueError, "combined global request offset"):
            ANALYZE.validate_prompt_bundle(bad_offset, protocol=PROTOCOL)

    def test_mixed_campaigns_without_combination_metadata_fail_closed(self):
        first = behavioral_prompt()
        second = behavioral_prompt()
        second["id"] = "behavioral-second"
        second["metadata"]["campaign_sha256"] = "d" * 64
        second["metadata"]["task"].update(
            {
                "selection_index": 1,
                "instance_id": "project2__task-2",
                "repo": "project2/repo",
                "base_commit": "b" * 40,
            }
        )
        second["metadata"]["selection"]["global_request_index"] = 2
        second["metadata"]["targets"][0]["task_instance_id"] = "project2__task-2"
        second["metadata"]["targets"][0]["foils"][0][
            "task_instance_id"
        ] = "project2__task-2"
        second["metadata"]["provenance"]["prompt_record_payload_sha256"] = (
            ANALYZE._prompt_payload_hash(second)
        )
        with self.assertRaisesRegex(ValueError, "combined-cohort metadata"):
            ANALYZE.validate_prompt_bundle([first, second], protocol=PROTOCOL)


class SplitIsolationTests(unittest.TestCase):
    def setUp(self):
        self.class_ids = ["success", "failure"]
        self.rows = feature_rows(self.class_ids)
        self.repositories = sorted({row["repo"] for row in self.rows})
        self.contract = copy.deepcopy(PROTOCOL["crossfit"])
        self.contract["calibration_repository_count"] = 2
        self.contract["minimum_fit_repositories"] = 4

    def test_heldout_labels_do_not_change_heldout_split_or_parameters(self):
        original = ANALYZE.crossfit_track(
            self.rows,
            class_ids=self.class_ids,
            all_repositories=self.repositories,
            contract=self.contract,
        )
        heldout = self.repositories[0]
        mutated = copy.deepcopy(self.rows)
        for row in mutated:
            if row["repo"] == heldout:
                row["label"] = self.class_ids[1 - self.class_ids.index(row["label"])]
                row["scores"] = [1000.0, -1000.0]
        changed = ANALYZE.crossfit_track(
            mutated,
            class_ids=self.class_ids,
            all_repositories=self.repositories,
            contract=self.contract,
        )
        original_fold = next(
            fold for fold in original["folds"] if fold["heldout_repository"] == heldout
        )
        changed_fold = next(
            fold for fold in changed["folds"] if fold["heldout_repository"] == heldout
        )
        self.assertEqual(original_fold["fit_repositories"], changed_fold["fit_repositories"])
        self.assertEqual(
            original_fold["calibration_repositories"],
            changed_fold["calibration_repositories"],
        )
        self.assertEqual(original_fold["fit_payload_sha256"], changed_fold["fit_payload_sha256"])
        self.assertEqual(
            original_fold["calibration_payload_sha256"],
            changed_fold["calibration_payload_sha256"],
        )
        self.assertEqual(original_fold["bias_fit"], changed_fold["bias_fit"])
        self.assertEqual(original_fold["temperature_fit"], changed_fold["temperature_fit"])

    def test_missing_class_support_is_not_imputed(self):
        rows = [row for row in self.rows if row["label"] == "success"]
        result = ANALYZE.crossfit_track(
            rows,
            class_ids=self.class_ids,
            all_repositories=self.repositories,
            contract=self.contract,
        )
        self.assertEqual(result["status"], "insufficient_split_or_class_support")
        self.assertEqual(result["successful_fold_count"], 0)
        self.assertEqual(result["predictions"], [])

    def test_fit_prior_and_metrics_are_training_only_baselines(self):
        fit = [
            {"label": "success"},
            {"label": "success"},
            {"label": "success"},
            {"label": "failure"},
        ]
        prior = ANALYZE._fit_prior(fit, self.class_ids, 1.0)
        self.assertEqual(prior, [4.0 / 6.0, 2.0 / 6.0])
        records = [
            {
                "row_id": "a",
                "task_id": "a__1",
                "repo": "a/repo",
                "label": "success",
                "prediction": "success",
                "probabilities": prior,
            },
            {
                "row_id": "b",
                "task_id": "b__1",
                "repo": "b/repo",
                "label": "failure",
                "prediction": "success",
                "probabilities": prior,
            },
        ]
        metrics = ANALYZE.classification_metrics(records, self.class_ids)
        self.assertEqual(metrics["micro_accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(
            metrics["negative_log_likelihood"],
            (-math.log(4.0 / 6.0) - math.log(2.0 / 6.0)) / 2.0,
        )

    def test_fit_prior_matches_checkpoint_row_estimand(self):
        fit = [
            {"task_id": "long", "label": "success"},
            {"task_id": "long", "label": "success"},
            {"task_id": "long", "label": "success"},
            {"task_id": "short", "label": "failure"},
        ]
        self.assertEqual(
            ANALYZE._fit_prior(fit, self.class_ids, 1.0),
            [4.0 / 6.0, 2.0 / 6.0],
        )

    def test_bootstrap_enforces_minimum_valid_fraction(self):
        records = [
            {
                "row_id": "success",
                "task_id": "success__1",
                "repo": "success/repo",
                "cohort_id": "development",
                "label": "success",
                "prediction": "success",
                "probabilities": [0.9, 0.1],
            },
            {
                "row_id": "failure",
                "task_id": "failure__1",
                "repo": "failure/repo",
                "cohort_id": "replication",
                "label": "failure",
                "prediction": "failure",
                "probabilities": [0.1, 0.9],
            },
        ]
        result = ANALYZE.bootstrap_classification(
            records,
            ["success", "failure"],
            samples=1000,
            seed=1,
            confidence_level=0.95,
            minimum_valid_fraction=0.8,
        )
        self.assertEqual(result["status"], "insufficient_valid_bootstrap_fraction")
        self.assertEqual(
            result["metric_status"]["balanced_accuracy"],
            "insufficient_valid_bootstrap_fraction",
        )
        self.assertIsNone(result["intervals"]["balanced_accuracy"])
        subgroup = ANALYZE.classification_metrics_by_cohort(
            records, ["success", "failure"]
        )
        self.assertEqual(
            set(subgroup["cohorts"]), {"development", "replication"}
        )

    def test_track_support_requires_enabled_valid_bootstrap(self):
        rows = feature_rows(PROTOCOL["outcome_ids"])
        for row in rows:
            row["probabilities"] = ANALYZE.softmax(row["scores"])
            row["prediction"] = PROTOCOL["outcome_ids"][
                max(range(len(row["scores"])), key=row["scores"].__getitem__)
            ]
        features = {method: copy.deepcopy(rows) for method in ANALYZE.METHODS}
        repositories = sorted({row["repo"] for row in rows})
        disabled = ANALYZE._track_analysis(
            features,
            class_ids=PROTOCOL["outcome_ids"],
            all_repositories=repositories,
            protocol=PROTOCOL,
            bootstrap_samples=0,
            seed_base=0,
            decision_track="official_outcome",
        )
        self.assertTrue(disabled["all_method_split_support_rules_pass"])
        self.assertFalse(disabled["all_method_bootstrap_valid_fraction_rules_pass"])
        self.assertFalse(disabled["all_method_inference_support_rules_pass"])

        enabled = ANALYZE._track_analysis(
            features,
            class_ids=PROTOCOL["outcome_ids"],
            all_repositories=repositories,
            protocol=PROTOCOL,
            bootstrap_samples=20,
            seed_base=0,
            decision_track="official_outcome",
        )
        self.assertTrue(enabled["all_method_bootstrap_valid_fraction_rules_pass"])
        self.assertTrue(enabled["all_method_inference_support_rules_pass"])


class NumericalCertificationTests(unittest.TestCase):
    def test_diagnostics_are_recomputed_against_pinned_tolerances(self):
        diagnostics = ANALYZE._validate_numerical_diagnostics(
            numerical_experiment(),
            generated_token_id=7,
            final_position=2,
            protocol=PROTOCOL,
            label="synthetic",
        )
        self.assertTrue(diagnostics["certified"])

        inflated = numerical_experiment()
        inflated["final_logits_reconstruction"]["max_abs_tolerance"] = 100.0
        with self.assertRaisesRegex(ValueError, "differ from the protocol"):
            ANALYZE._validate_numerical_diagnostics(
                inflated,
                generated_token_id=7,
                final_position=2,
                protocol=PROTOCOL,
                label="synthetic",
            )

    def test_false_report_flags_and_topk_claims_fail_closed(self):
        inconsistent = numerical_experiment()
        inconsistent["final_norm_reconstruction"]["within_tolerance"] = False
        with self.assertRaisesRegex(ValueError, "flag is inconsistent"):
            ANALYZE._validate_numerical_diagnostics(
                inconsistent,
                generated_token_id=7,
                final_position=2,
                protocol=PROTOCOL,
                label="synthetic",
            )

    def test_consistent_failed_diagnostics_are_retained_as_uncertified(self):
        experiment = numerical_experiment()
        captured = experiment["captured_final_model_readout"][0]
        captured["token_ids"] = [8, 7, 9, 10, 11]
        captured["target_rank"] = 2
        experiment["final_layer_top1_matches_greedy"] = False
        experiment["final_logits_reconstruction"][
            "top_k_prefix_token_ids_match"
        ] = False
        experiment["final_logits_reconstruction"]["within_tolerance"] = False
        diagnostics = ANALYZE._validate_numerical_diagnostics(
            experiment,
            generated_token_id=7,
            final_position=2,
            protocol=PROTOCOL,
            label="synthetic",
        )
        self.assertFalse(diagnostics["certified"])

        inconsistent = numerical_experiment()
        inconsistent["captured_final_model_readout"][0]["token_ids"][4] = 12
        with self.assertRaisesRegex(ValueError, "top-k parity flag is inconsistent"):
            ANALYZE._validate_numerical_diagnostics(
                inconsistent,
                generated_token_id=7,
                final_position=2,
                protocol=PROTOCOL,
                label="synthetic",
            )


class FutureTargetCohortTests(unittest.TestCase):
    def test_nonprimary_lifecycle_prompts_are_excluded(self):
        prompt = {
            "primary": False,
            "targets": [],
        }
        reports = {
            label: {"rows": [{"numerically_certified": True}]}
            for label in ANALYZE.REPORT_LABELS
        }
        detailed, aggregate = ANALYZE.build_future_target_rows(
            {"prompts": [prompt]}, reports, PROTOCOL
        )
        self.assertEqual(detailed, [])
        self.assertTrue(all(not rows for rows in aggregate.values()))

    def test_future_subgroups_are_descriptive(self):
        rows = [
            {
                "cohort_id": cohort_id,
                "task_id": f"{cohort_id}__1",
                "repo": f"{cohort_id}/repo",
                "correct": correct,
                "probability": 0.75 if correct else 0.25,
                "margin": 1.0 if correct else -1.0,
            }
            for cohort_id, correct in (("development", True), ("replication", False))
        ]
        result = ANALYZE._future_metrics_by_cohort(rows)
        self.assertEqual(result["status"], "descriptive_only_no_subgroup_refitting")
        self.assertEqual(
            set(result["cohorts"]), {"development", "replication"}
        )

    def test_fixed_foil_is_seeded_and_score_independent(self):
        target = {"id": "target-z"}
        foils = [{"id": "foil-b"}, {"id": "foil-a"}]
        expected = min(
            foils,
            key=lambda foil: (
                ANALYZE.sha256_text(
                    f"{PROTOCOL['future_target']['foil_selection_seed']}\0"
                    f"{target['id']}\0{foil['id']}"
                ),
                foil["id"],
            ),
        )
        selected = ANALYZE._fixed_hidden_foil(target, foils, PROTOCOL)
        self.assertEqual(selected["id"], expected["id"])
        self.assertNotIn("score", selected)

    def test_future_metrics_weight_tasks_not_target_rows(self):
        rows = [
            {
                "row_id": f"many-{index}",
                "task_id": "many",
                "repo": "a/repo",
                "cohort_id": "pooled",
                "correct": True,
                "probability": 0.9,
                "margin": 1.0,
            }
            for index in range(10)
        ] + [
            {
                "row_id": "one",
                "task_id": "one",
                "repo": "b/repo",
                "cohort_id": "pooled",
                "correct": False,
                "probability": 0.1,
                "margin": -1.0,
            }
        ]
        metrics = ANALYZE._future_metrics(rows)
        self.assertEqual(metrics["row_count"], 11)
        self.assertEqual(metrics["task_count"], 2)
        self.assertEqual(metrics["target_preference_accuracy"], 0.5)


class PairedInferenceTests(unittest.TestCase):
    @staticmethod
    def classification_rows(*, correct: bool):
        class_ids = ["success", "failure"]
        rows = []
        for repository in range(6):
            for class_index, label in enumerate(class_ids):
                predicted_index = class_index if correct else 1 - class_index
                probabilities = [0.1, 0.1]
                probabilities[predicted_index] = 0.9
                rows.append(
                    {
                        "row_id": f"r{repository}-{label}",
                        "task_id": f"task-{repository}",
                        "repo": f"repo-{repository}",
                        "cohort_id": "pooled",
                        "label": label,
                        "prediction": class_ids[predicted_index],
                        "probabilities": probabilities,
                    }
                )
        return rows

    def test_paired_bootstrap_uses_exact_identity_and_same_draw_delta(self):
        result = ANALYZE.bootstrap_paired_classification(
            self.classification_rows(correct=True),
            self.classification_rows(correct=False),
            ["success", "failure"],
            samples=200,
            seed=7,
            confidence_level=0.95,
            minimum_valid_fraction=0.8,
        )
        self.assertEqual(result["status"], "available")
        self.assertTrue(result["same_draw_for_candidate_and_reference"])
        self.assertTrue(result["pairing"]["exact_row_coverage"])
        self.assertEqual(
            result["observed_benefit_deltas"]["balanced_accuracy_gain"], 1.0
        )
        self.assertEqual(
            result["intervals"]["balanced_accuracy_gain"]["lower"], 1.0
        )

        mismatched = self.classification_rows(correct=False)[:-1]
        mismatch = ANALYZE.bootstrap_paired_classification(
            self.classification_rows(correct=True),
            mismatched,
            ["success", "failure"],
            samples=20,
            seed=7,
            confidence_level=0.95,
            minimum_valid_fraction=0.8,
        )
        self.assertEqual(mismatch["status"], "insufficient_unpaired_row_coverage")
        self.assertFalse(mismatch["pairing"]["exact_row_coverage"])

    def test_paired_future_is_task_averaged_and_uses_the_same_fixed_contrast(self):
        candidate = []
        reference = []
        for task in range(6):
            for target in range(task + 1):
                identity = {
                    "row_id": f"task-{task}-target-{target}",
                    "prompt_id": f"prompt-{task}",
                    "target_id": f"target-{target}",
                    "task_id": f"task-{task}",
                    "repo": f"repo-{task}",
                    "cohort_id": "pooled",
                    "fixed_hidden_foil_id": f"foil-{task}-{target}",
                }
                candidate.append(
                    {
                        **identity,
                        "correct": task != 0,
                        "probability": 0.9 if task != 0 else 0.1,
                        "margin": 1.0 if task != 0 else -1.0,
                    }
                )
                reference.append(
                    {
                        **identity,
                        "correct": False,
                        "probability": 0.1,
                        "margin": -1.0,
                    }
                )
        result = ANALYZE.bootstrap_paired_future(
            candidate,
            reference,
            samples=200,
            seed=8,
            confidence_level=0.95,
            minimum_valid_fraction=0.8,
        )
        self.assertEqual(result["status"], "available")
        self.assertTrue(result["task_averaged"])
        self.assertTrue(
            result["pairing"]["exact_fixed_foil_contrast_across_methods"]
        )
        self.assertEqual(result["pairing"]["paired_task_count"], 6)
        self.assertEqual(
            result["observed_benefit_deltas"][
                "target_preference_accuracy_gain"
            ],
            5.0 / 6.0,
        )

        missing = ANALYZE.bootstrap_paired_future(
            candidate,
            reference[:-1],
            samples=20,
            seed=8,
            confidence_level=0.95,
            minimum_valid_fraction=0.8,
        )
        self.assertEqual(
            missing["status"], "insufficient_unpaired_target_row_coverage"
        )

        wrong_foil = copy.deepcopy(reference)
        wrong_foil[0]["fixed_hidden_foil_id"] = "different-foil"
        with self.assertRaisesRegex(ValueError, "different fixed foils"):
            ANALYZE.bootstrap_paired_future(
                candidate,
                wrong_foil,
                samples=20,
                seed=8,
                confidence_level=0.95,
                minimum_valid_fraction=0.8,
            )


class ProbeVersusRefitDecisionTests(unittest.TestCase):
    @staticmethod
    def comparison(metric: str, point: float, lower: float, upper: float):
        return {
            "status": "available",
            "observed_benefit_deltas": {metric: point},
            "metric_status": {metric: "available"},
            "intervals": {
                metric: {"lower": lower, "upper": upper, "valid_samples": 5000}
            },
        }

    def comparisons(self):
        action = {
            "public_jacobian_vs_ordinary_logit": self.comparison(
                "balanced_accuracy_gain", 0.05, -0.02, 0.12
            ),
            "public_jacobian_vs_majority_baseline": self.comparison(
                "balanced_accuracy_gain", 0.04, -0.03, 0.11
            ),
            "public_jacobian_vs_native_jacobian": self.comparison(
                "balanced_accuracy_gain", 0.15, 0.02, 0.25
            ),
            "public_jacobian_vs_nf4_jacobian": self.comparison(
                "balanced_accuracy_gain", 0.03, -0.04, 0.10
            ),
        }
        future = {
            "public_jacobian_vs_ordinary_logit": self.comparison(
                "target_preference_accuracy_gain", 0.10, 0.01, 0.20
            ),
            "public_jacobian_vs_native_jacobian": self.comparison(
                "target_preference_accuracy_gain", 0.08, 0.01, 0.15
            ),
            "public_jacobian_vs_nf4_jacobian": self.comparison(
                "target_preference_accuracy_gain", 0.01, -0.06, 0.08
            ),
        }
        return action, future

    def decide(
        self,
        action,
        future,
        *,
        coverage=True,
        samples=5000,
        operational_status="held_out_evaluation_complete",
        outcome=None,
    ):
        return ANALYZE.build_probe_vs_refit_decision(
            operational_status=operational_status,
            joint_coverage={
                "all_predeclared_joint_coverage_gates_pass": coverage,
            },
            action_comparisons=action,
            outcome_comparisons=action if outcome is None else outcome,
            future_comparisons=future,
            protocol=PROTOCOL,
            bootstrap_samples=samples,
        )

    def test_refit_native_candidate_branch_and_status_separation(self):
        action, future = self.comparisons()
        result = self.decide(action, future)
        self.assertEqual(result["classification"], "refit_native_candidate")
        self.assertEqual(result["operational_status"], "held_out_evaluation_complete")
        self.assertTrue(result["operational_status_is_not_scientific_decision"])
        self.assertEqual(
            result["claim_scope"],
            "pooled_predeclared_development_plus_replication_repository_crossfit",
        )

    def test_no_refit_evidence_branch(self):
        action, future = self.comparisons()
        action["public_jacobian_vs_native_jacobian"] = self.comparison(
            "balanced_accuracy_gain", 0.05, -0.04, 0.13
        )
        result = self.decide(action, future)
        self.assertEqual(result["classification"], "no_refit_evidence")
        self.assertTrue(result["probe_valid"])

    def test_readout_or_task_problem_branch(self):
        action, future = self.comparisons()
        future["public_jacobian_vs_ordinary_logit"] = self.comparison(
            "target_preference_accuracy_gain", -0.05, -0.12, 0.0
        )
        result = self.decide(action, future)
        self.assertEqual(result["classification"], "readout_or_task_problem")
        self.assertFalse(result["probe_valid"])

    def test_insufficient_support_branches_never_recommend_refit(self):
        action, future = self.comparisons()
        coverage = self.decide(action, future, coverage=False)
        self.assertEqual(coverage["classification"], "insufficient_support")
        self.assertIn("do not refit", coverage["next_step"])

        samples = self.decide(action, future, samples=4999)
        self.assertEqual(samples["classification"], "insufficient_support")
        self.assertFalse(samples["support"]["paired_bootstrap_sample_gate_pass"])

        operational = self.decide(
            action,
            future,
            operational_status="insufficient_split_or_class_support",
        )
        self.assertEqual(operational["classification"], "insufficient_support")
        self.assertFalse(operational["support"]["operational_inference_gate_pass"])

        unavailable_outcome = copy.deepcopy(action)
        unavailable_outcome["public_jacobian_vs_native_jacobian"]["metric_status"][
            "balanced_accuracy_gain"
        ] = "insufficient_valid_bootstrap_fraction"
        unavailable_outcome["public_jacobian_vs_native_jacobian"]["intervals"][
            "balanced_accuracy_gain"
        ] = None
        outcome_gate = self.decide(
            action,
            future,
            outcome=unavailable_outcome,
        )
        self.assertEqual(outcome_gate["classification"], "insufficient_support")
        self.assertFalse(
            outcome_gate["support"]["required_metric_evidence_available"]
        )

    def test_joint_coverage_gates_pass_and_fail_explicitly(self):
        prompt_contract = {
            "primary_prompt_count": 80,
            "task_count": 20,
            "selected_task_count": 20,
            "unprobed_task_ids": [],
            "repository_count": 10,
        }
        action = []
        for task in range(20):
            for label in PROTOCOL["action_ids"]:
                action.append(
                    {
                        "row_id": f"a-{task}-{label}",
                        "task_id": f"task-{task}",
                        "repo": f"repo-{task % 10}",
                        "label": label,
                    }
                )
        outcome = [
            {
                "row_id": f"o-{task}",
                "task_id": f"task-{task}",
                "repo": f"repo-{task % 10}",
                "label": PROTOCOL["outcome_ids"][task % 2],
            }
            for task in range(20)
        ]
        future = [
            {
                "row_id": f"f-{task}",
                "task_id": f"task-{task}",
                "repo": f"repo-{task % 6}",
                "cohort_id": "pooled",
                "correct": True,
                "probability": 0.75,
                "margin": 1.0,
            }
            for task in range(10)
        ]
        passed = ANALYZE.build_joint_decision_coverage(
            prompt_contract=prompt_contract,
            action_records=action,
            outcome_records=outcome,
            future_records=future,
            protocol=PROTOCOL,
        )
        self.assertTrue(passed["all_predeclared_joint_coverage_gates_pass"])
        self.assertEqual(passed["next_action"]["selected_row_count"], 80)
        self.assertEqual(
            passed["next_action"]["jointly_certified_selected_row_fraction"], 1.0
        )
        self.assertTrue(all(passed["next_action"]["gates"].values()))
        self.assertTrue(all(passed["official_outcome"]["gates"].values()))
        self.assertTrue(all(passed["future_identifier"]["gates"].values()))

        action_failed = [
            row
            for row in action
            if row["label"] != PROTOCOL["action_ids"][-1]
            or int(row["task_id"].split("-")[-1]) < 4
        ]
        failed = ANALYZE.build_joint_decision_coverage(
            prompt_contract=prompt_contract,
            action_records=action_failed,
            outcome_records=outcome[:15],
            future_records=future[:9],
            protocol=PROTOCOL,
        )
        self.assertFalse(failed["all_predeclared_joint_coverage_gates_pass"])
        self.assertFalse(failed["next_action"]["gates"]["minimum_tasks_per_class"])
        self.assertFalse(
            failed["official_outcome"]["gates"]["minimum_jointly_certified_tasks"]
        )
        self.assertFalse(failed["future_identifier"]["gates"]["minimum_tasks"])

        below_fraction = ANALYZE.build_joint_decision_coverage(
            prompt_contract=prompt_contract,
            action_records=action[:63],
            outcome_records=outcome,
            future_records=future,
            protocol=PROTOCOL,
        )
        self.assertEqual(
            below_fraction["next_action"][
                "jointly_certified_selected_row_fraction"
            ],
            63 / 80,
        )
        self.assertFalse(
            below_fraction["next_action"]["gates"][
                "minimum_jointly_certified_selected_row_fraction"
            ]
        )


class BaselinePairingTests(unittest.TestCase):
    def test_final_logit_residual_baseline_difference_fails(self):
        binding = {
            "id": "p",
            "prompt": "text",
            "prompt_token_ids": [1],
            "metadata": {"kind": "synthetic"},
            "scored_vocabulary": {"token_ids": [2], "tokens": [" x"]},
            "generated_token_id": 3,
            "residual_capture_manifest": {"sha256": "a" * 64},
            "fixed_band_logit_readouts": [{"scored_tokens": [{"score": 1.0}]}],
            "final_layer_top1_matches_greedy": True,
            "final_norm_reconstruction": {"within_tolerance": True},
            "final_logits_reconstruction": {"within_tolerance": True},
            "final_model_readout": [{"scored_tokens": []}],
            "captured_final_model_readout": [{"scored_tokens": []}],
        }
        reports = [
            {
                "label": label,
                "runtime_identity": {"mtp_enabled": False},
                "rows": [
                    {
                        "prompt": {"id": "p"},
                        "baseline_binding": copy.deepcopy(binding),
                        "layer_ids": list(range(24, 48)),
                    }
                ],
            }
            for label in ANALYZE.REPORT_LABELS
        ]
        self.assertTrue(
            ANALYZE.validate_report_pairing(reports)[
                "fixed_band_ordinary_logit_readouts_equal"
            ]
        )
        reports[2]["rows"][0]["baseline_binding"]["fixed_band_logit_readouts"][0][
            "scored_tokens"
        ][0]["score"] = 2.0
        with self.assertRaisesRegex(ValueError, "baselines differ"):
            ANALYZE.validate_report_pairing(reports)


if __name__ == "__main__":
    unittest.main()
