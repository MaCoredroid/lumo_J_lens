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
            {"prompts": [prompt]}, reports
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
