#!/usr/bin/env python3
"""Focused tests for the selective SWE task-state interpreter."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module(
    "analyze_swe_task_state_interpreter",
    ROOT / "scripts/analyze_swe_task_state_interpreter.py",
)
BEHAVIORAL_PATH = ROOT / "configs/swe_behavioral_readout_protocol.json"
BEHAVIORAL = json.loads(BEHAVIORAL_PATH.read_bytes())


def protocol() -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": "swe-task-state-interpreter-v1",
        "decision_scope": "development_screen_only",
        "input_pins": {
            "behavioral_protocol_sha256": MODULE.sha256_file(BEHAVIORAL_PATH),
            "prompt_bundle_sha256": None,
            "public_report_sha256": None,
            "model": {
                "repo_id": "nvidia/Qwen3.6-27B-NVFP4",
                "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
                "config_sha256": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
                "index_sha256": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
            },
            "public_lens": {
                "repo_id": "neuronpedia/jacobian-lens",
                "revision": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
                "sha256": "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1",
                "n_prompts": 1000,
            },
            "replay_runtime": {
                "enforce_eager": True,
                "mtp_enabled": False,
                "max_model_len": 65536,
                "max_num_batched_tokens": 4096,
                "mamba_block_size": 1024,
                "kv_cache_dtype": "fp8_e4m3",
                "kv_offloading_size": 8.0,
                "kv_offloading_backend": "native",
                "stream_final_only": True,
            },
        },
        "scope": {
            "class_ids_in_order": [item["id"] for item in BEHAVIORAL["action_classes"]],
            "variants": list(MODULE.VARIANTS),
        },
        "feature_contract": {
            "layers": BEHAVIORAL["fixed_layer_band"]["layers"],
            "feature_order": "layer-major_then_class-order",
            "within_class_reduction": "logmeanexp_over_declared_token_logits",
            "progress_features": ["task_request_index", "log1p_task_request_index"],
            "lexical_features": [
                "log1p_exact_token_occurrence_count_per_class",
                "normalized_token_recency_per_class",
                "log1p_exact_string_occurrence_count_per_class",
                "normalized_string_recency_per_class",
            ],
            "history_features": [
                "log1p_cumulative_prior_action_count_per_class",
                "previous_action_one_hot_per_class",
                "log1p_cumulative_unknown_prior_action_count",
                "previous_action_unknown",
                "has_edited",
                "has_validated",
                "turns_since_edit_or_minus_one",
                "turns_since_validate_or_minus_one",
            ],
            "history_requires_complete_consecutive_probe_bundle": True,
            "future_trajectory_fields_forbidden": True,
            "variant_blocks": {
                "progress_only": ["progress"],
                "lexical_progress": ["lexical", "progress"],
                "history_context": ["history", "lexical", "progress"],
                "ordinary_logit": ["ordinary_logit"],
                "logit_context": ["ordinary_logit", "history", "lexical", "progress"],
                "public_jacobian": ["public_jacobian"],
                "jacobian_context": [
                    "public_jacobian",
                    "history",
                    "lexical",
                    "progress",
                ],
                "hybrid": [
                    "public_jacobian",
                    "ordinary_logit",
                    "history",
                    "lexical",
                    "progress",
                ],
            },
        },
        "eligibility": {
            "action_label_status": "available",
            "require_primary_selection": False,
            "require_finite_action_scores": True,
            "numerical_stability": {
                "final_layer_top1_matches_greedy": True,
                "final_norm_within_tolerance": True,
                "final_logits_top_k_prefix_token_ids_match": True,
                "final_logits_rms_error_maximum_inclusive": 0.02,
                "final_logits_max_abs_error_maximum_inclusive": 0.125,
            },
        },
        "model_contract": {
            "family": "multinomial_logistic_regression",
            "penalty": "L2",
            "solver": "lbfgs",
            "class_weight": "balanced_from_current_training_split_only",
            "fit_intercept": True,
            "scaler_fit": "current_training_split_only",
            "regularization_C_grid": [0.01],
            "maximum_iterations": 2000,
        },
        "outer_evaluation": {
            "algorithm": "leave_one_repository_out",
            "heldout_labels_used_for_fit_scaling_calibration_or_threshold_selection": False,
        },
        "inner_model_selection": {
            "algorithm": "leave_one_repository_out_within_outer_training",
            "selection_metric": "balanced_accuracy",
            "tie_break": "smallest_C",
            "complete_inner_prediction_coverage_required": True,
            "minimum_valid_inner_folds": 5,
        },
        "calibration": {
            "algorithm": "inner_repository_crossfit_temperature_grid",
            "temperature_grid": [0.5, 1.0, 2.0],
            "selection_metric": "multiclass_negative_log_likelihood",
            "tie_break": "closest_to_one_then_smallest",
            "outer_heldout_labels_used": False,
        },
        "abstention": {
            "confidence": "maximum_temperature_scaled_class_probability",
            "accept_when": "confidence_greater_than_or_equal_to_threshold",
            "confidence_threshold_grid": [0.0, 0.5, 0.8],
            "sweep_role": "descriptive_fixed_grid",
            "selection": {
                "algorithm": "maximum_coverage_meeting_floors",
                "accepted_accuracy_minimum": 0.5,
                "balanced_accepted_recall_minimum": 0.5,
                "coverage_minimum": 0.25,
                "minimum_accepted_rows_per_class": 1,
                "floor_weighting": "task_equal_primary",
                "tie_break": "lowest_threshold",
                "fallback": "maximum_balanced_accepted_recall_then_accuracy_then_coverage",
                "outer_heldout_labels_used": False,
            },
        },
        "metrics": {
            "ece_equal_width_bin_count": 10,
            "probability_metrics": [
                "multiclass_negative_log_likelihood",
                "multiclass_brier",
                "top_label_ece",
            ],
            "selective_metrics": [
                "coverage",
                "accepted_accuracy",
                "balanced_accepted_recall",
                "per_class_accepted_coverage",
            ],
        },
        "bootstrap": {
            "algorithm": "hierarchical_repository_then_task_percentile_v1",
            "samples": 50,
            "seed": 91,
            "confidence_level": 0.95,
            "minimum_valid_fraction": 0.5,
            "row_resampling_forbidden": True,
            "models_refit_inside_bootstrap": False,
            "interval_interpretation": "conditional_on_frozen_out_of_repository_predictions_excludes_fit_and_selection_uncertainty",
            "operational_reliability_proof": False,
        },
        "reliability_gates": {
            "support": {
                "minimum_rows": 1,
                "minimum_tasks": 1,
                "minimum_repositories": 2,
            },
            "require_all_outer_folds_available": True,
            "require_all_calibration_targets_met": False,
            "calibration_target_variants": ["hybrid"],
            "absolute": [
                {
                    "id": "hybrid_task_accuracy",
                    "variant": "hybrid",
                    "metric": "task_macro_overall_accuracy",
                    "bound": "point",
                    "operator": "minimum_inclusive",
                    "value": 0.5,
                }
            ],
            "paired": [
                {
                    "id": "hybrid_j_ablation",
                    "candidate": "hybrid",
                    "reference": "logit_context",
                    "metric": "task_macro_overall_accuracy",
                    "bound": "bootstrap_lower",
                    "operator": "minimum_exclusive",
                    "value": 0.0,
                }
            ],
        },
    }


def prompt_and_experiment(index: int, label: str, *, stable: bool = True):
    all_token_ids = [
        token["token_id"]
        for action_class in BEHAVIORAL["action_classes"]
        for token in action_class["tokens"]
    ]
    all_token_texts = [
        token["text"]
        for action_class in BEHAVIORAL["action_classes"]
        for token in action_class["tokens"]
    ]
    prompt_id = f"dense-task-r{index:04d}"
    prompt = {
        "id": prompt_id,
        "text": "".join(all_token_texts) + f" turn {index}",
        "token_ids": all_token_ids + [100000 + index],
        "score_token_ids": all_token_ids,
        "metadata": {
            "labels": {"action": {"status": "available", "class_id": label}},
            "selection": {
                "task_request_index": index,
                "checkpoint_ordinal": index - 1,
                "primary_for_action_evaluation": True,
            },
            "task": {
                "instance_id": "owner__repo-1",
                "repo": "owner/repo",
                "probeable_request_indices": [1, 2, 3],
            },
            "cohort": {"id": "dense"},
        },
    }
    layers = []
    for layer in BEHAVIORAL["fixed_layer_band"]["layers"]:
        scored = [
            {"token_id": token_id, "score": float(token_id % 17) / 10.0}
            for token_id in all_token_ids
        ]
        layers.append(
            {
                "layer": layer,
                "positions": [
                    {
                        "token_position": len(prompt["token_ids"]) - 1,
                        "logit_lens": {"scored_tokens": scored},
                        "jacobian_lens": {
                            "scored_tokens": [
                                {**item, "score": item["score"] + layer / 100.0}
                                for item in scored
                            ]
                        },
                    }
                ],
            }
        )
    experiment = {
        "id": prompt_id,
        "prompt": prompt["text"],
        "prompt_token_ids": prompt["token_ids"],
        "capture_positions_resolved": [len(prompt["token_ids"]) - 1],
        "metadata": prompt["metadata"],
        "scored_vocabulary": {"token_ids": prompt["score_token_ids"]},
        "final_layer_top1_matches_greedy": stable,
        "final_norm_reconstruction": {"within_tolerance": stable},
        "final_logits_reconstruction": {
            "top_k_prefix_token_ids_match": stable,
            "rms_error": 0.01,
            "max_abs_error": 0.1,
        },
        "layers": layers,
    }
    return prompt, experiment


def report(experiments):
    value = protocol()["input_pins"]
    return {
        "schema_version": 3,
        "score_encoding": "unrounded-float32",
        "assertions": {
            "lens_hash_matches": True,
            "lens_metadata_matches": True,
            "model_architecture_matches": True,
        },
        "model": value["model"],
        "lens": value["public_lens"],
        "runtime": value["replay_runtime"],
        "experiments": experiments,
    }


def feature_row(repository: int, class_index: int, repetition: int = 0):
    feature = np.zeros(12, dtype=np.float64)
    feature[class_index] = 3.0
    feature[4 + repository % 4] = 0.01
    return {
        "row_id": f"repo-{repository}-class-{class_index}-{repetition}",
        "task_id": f"repo-{repository}__task-{repetition}",
        "repo": f"owner-{repository}/repo",
        "cohort_id": "synthetic",
        "label": MODULE.READOUT.CLASS_IDS[class_index],
        "task_request_index": class_index + 1,
        "checkpoint_ordinal": class_index,
        "features": {"ordinary_logit": feature.tolist()},
    }


class TaskStateInterpreterTests(unittest.TestCase):
    def test_protocol_is_external_and_validated(self):
        value = protocol()
        result = MODULE.validate_protocol(
            value,
            behavioral_protocol_value=BEHAVIORAL,
            behavioral_protocol_sha256=MODULE.sha256_file(BEHAVIORAL_PATH),
            prompt_sha256="0" * 64,
            report_sha256="1" * 64,
        )
        self.assertEqual(result["class_ids"], list(MODULE.READOUT.CLASS_IDS))
        self.assertEqual(result["abstention"]["thresholds"], [0.0, 0.5, 0.8])
        broken = copy.deepcopy(value)
        broken["abstention"]["selection"]["accepted_accuracy_minimum"] = 1.1
        with self.assertRaisesRegex(ValueError, "must be probabilities"):
            MODULE.validate_protocol(
                broken,
                behavioral_protocol_value=BEHAVIORAL,
                behavioral_protocol_sha256=MODULE.sha256_file(BEHAVIORAL_PATH),
            )

    def test_report_payload_and_artifact_pins_are_fail_closed(self):
        prompt, experiment = prompt_and_experiment(1, "inspect")
        normalized = MODULE.validate_protocol(
            protocol(),
            behavioral_protocol_value=BEHAVIORAL,
            behavioral_protocol_sha256=MODULE.sha256_file(BEHAVIORAL_PATH),
        )
        altered = copy.deepcopy(experiment)
        altered["prompt_token_ids"][-1] += 1
        with self.assertRaisesRegex(ValueError, "not bound"):
            MODULE.extract_rows([prompt], report([altered]), protocol=normalized)

        wrong_position = copy.deepcopy(experiment)
        wrong_position["capture_positions_resolved"] = [len(prompt["token_ids"]) - 2]
        with self.assertRaisesRegex(ValueError, "final prompt token"):
            MODULE.extract_rows([prompt], report([wrong_position]), protocol=normalized)

        wrong_layer_position = copy.deepcopy(experiment)
        wrong_layer_position["layers"][0]["positions"][0]["token_position"] -= 1
        with self.assertRaisesRegex(ValueError, "final prompt token"):
            MODULE.extract_rows(
                [prompt], report([wrong_layer_position]), protocol=normalized
            )

        wrong_lens = report([experiment])
        wrong_lens["lens"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "lens pin differs"):
            MODULE.extract_rows([prompt], wrong_lens, protocol=normalized)

    def test_dense_extraction_builds_lexical_and_causal_history(self):
        pairs = [
            prompt_and_experiment(1, "inspect"),
            prompt_and_experiment(2, "edit"),
            prompt_and_experiment(3, "validate", stable=False),
        ]
        normalized = MODULE.validate_protocol(
            protocol(),
            behavioral_protocol_value=BEHAVIORAL,
            behavioral_protocol_sha256=MODULE.sha256_file(BEHAVIORAL_PATH),
        )
        result = MODULE.extract_rows(
            [item[0] for item in pairs],
            report([item[1] for item in pairs]),
            protocol=normalized,
        )
        self.assertEqual(result["eligibility"]["eligible_row_count"], 2)
        self.assertEqual(result["eligibility"]["exclusion_counts"], {"numerically_unstable": 1})
        self.assertEqual(result["eligibility"]["causal_history"]["complete_history_task_count"], 1)
        first, second = result["rows"]
        self.assertEqual(len(first["features"]["progress_only"]), 2)
        self.assertEqual(len(first["features"]["lexical_progress"]), 18)
        self.assertEqual(len(first["features"]["history_context"]), 32)
        self.assertEqual(len(first["features"]["ordinary_logit"]), 96)
        self.assertEqual(len(first["features"]["logit_context"]), 128)
        self.assertEqual(len(first["features"]["hybrid"]), 224)
        history_offset = 0
        second_history = second["features"]["history_context"][:14]
        self.assertAlmostEqual(second_history[history_offset], np.log(2.0))
        self.assertEqual(second_history[4:8], [1.0, 0.0, 0.0, 0.0])

    def test_sparse_bundle_never_claims_consecutive_history(self):
        first = prompt_and_experiment(1, "inspect")
        third = prompt_and_experiment(3, "validate")
        normalized = MODULE.validate_protocol(
            protocol(),
            behavioral_protocol_value=BEHAVIORAL,
            behavioral_protocol_sha256=MODULE.sha256_file(BEHAVIORAL_PATH),
        )
        result = MODULE.extract_rows(
            [first[0], third[0]],
            report([first[1], third[1]]),
            protocol=normalized,
        )
        self.assertEqual(result["eligibility"]["causal_history"]["complete_history_task_count"], 0)
        self.assertTrue(all(row["features"]["history_context"] is None for row in result["rows"]))
        self.assertTrue(all(row["features"]["logit_context"] is None for row in result["rows"]))
        self.assertTrue(all(row["features"]["jacobian_context"] is None for row in result["rows"]))
        self.assertTrue(all(row["features"]["hybrid"] is None for row in result["rows"]))

    def test_dense_history_records_unknown_action_without_imputation(self):
        pairs = [
            prompt_and_experiment(1, "inspect"),
            prompt_and_experiment(2, "edit"),
            prompt_and_experiment(3, "validate"),
        ]
        pairs[1][0]["metadata"]["labels"]["action"] = {
            "status": "missing",
            "class_id": None,
            "derivation": "unclassified_missing_not_imputed",
        }
        normalized = MODULE.validate_protocol(
            protocol(),
            behavioral_protocol_value=BEHAVIORAL,
            behavioral_protocol_sha256=MODULE.sha256_file(BEHAVIORAL_PATH),
        )
        result = MODULE.extract_rows(
            [item[0] for item in pairs],
            report([item[1] for item in pairs]),
            protocol=normalized,
        )
        self.assertEqual(result["eligibility"]["eligible_row_count"], 2)
        third_history = result["rows"][1]["features"]["history_context"][:14]
        self.assertAlmostEqual(third_history[8], np.log(2.0))
        self.assertEqual(third_history[9], 1.0)

    def test_nested_evaluation_is_repository_isolated(self):
        rows = [
            feature_row(repository, class_index, repetition)
            for repository in range(8)
            for class_index in range(4)
            for repetition in range(2)
        ]
        kwargs = {
            "variant": "ordinary_logit",
            "class_ids": list(MODULE.READOUT.CLASS_IDS),
            "c_grid": [0.01],
            "maximum_iterations": 2000,
            "minimum_valid_inner_folds": 5,
            "temperatures": [0.5, 1.0],
            "thresholds": [0.0, 0.5],
            "threshold_contract": {
                "accepted_accuracy_minimum": 0.5,
                "balanced_accepted_recall_minimum": 0.5,
                "coverage_minimum": 0.25,
                "minimum_accepted_rows_per_class": 1,
                "floor_weighting": "task_equal_primary",
            },
            "ece_bins": 10,
        }
        result = MODULE.nested_repository_evaluation(rows, **kwargs)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["metrics"]["full"]["balanced_accuracy"], 1.0)
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["repo"] == "owner-0/repo":
                current = MODULE.READOUT.CLASS_IDS.index(row["label"])
                row["label"] = MODULE.READOUT.CLASS_IDS[(current + 1) % 4]
        changed_result = MODULE.nested_repository_evaluation(changed, **kwargs)
        original_fold = next(item for item in result["folds"] if item["heldout_repository"] == "owner-0/repo")
        changed_fold = next(item for item in changed_result["folds"] if item["heldout_repository"] == "owner-0/repo")
        self.assertEqual(original_fold, changed_fold)
        original_probabilities = [row["probabilities"] for row in result["predictions"] if row["repo"] == "owner-0/repo"]
        changed_probabilities = [row["probabilities"] for row in changed_result["predictions"] if row["repo"] == "owner-0/repo"]
        self.assertEqual(original_probabilities, changed_probabilities)

    def test_balanced_accepted_recall_counts_abstentions_as_misses(self):
        classes = list(MODULE.READOUT.CLASS_IDS)
        predictions = []
        for class_id in classes:
            for accepted in (True, False):
                predictions.append(
                    {
                        "label": class_id,
                        "prediction": class_id,
                        "class_ids": classes,
                        "probabilities": [
                            1.0 if candidate == class_id else 0.0
                            for candidate in classes
                        ],
                        "confidence": 1.0,
                        "accepted": accepted,
                    }
                )
        metrics = MODULE._selective_metrics(predictions, classes, ece_bins=10)
        self.assertEqual(metrics["accepted_accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accepted_conditional_accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accepted_recall"], 0.5)

    def test_task_equal_summary_prevents_long_task_dominance(self):
        predictions = []
        for index in range(9):
            predictions.append(
                {
                    "row_id": f"long-{index}",
                    "task_id": "long",
                    "repo": "one/repo",
                    "label": "inspect",
                    "prediction": "edit",
                    "probabilities": [0.1, 0.7, 0.1, 0.1],
                    "confidence": 0.7,
                    "accepted": True,
                }
            )
        predictions.append(
            {
                "row_id": "short-0",
                "task_id": "short",
                "repo": "two/repo",
                "label": "inspect",
                "prediction": "inspect",
                "probabilities": [0.7, 0.1, 0.1, 0.1],
                "confidence": 0.7,
                "accepted": True,
            }
        )
        row_metrics = MODULE._probability_metrics(predictions, MODULE.READOUT.CLASS_IDS, ece_bins=10)
        task_metrics = MODULE._task_equal_metrics(predictions, MODULE.READOUT.CLASS_IDS, ece_bins=10)
        self.assertEqual(row_metrics["accuracy"], 0.1)
        self.assertEqual(task_metrics["task_macro_overall_accuracy"], 0.5)
        self.assertEqual(task_metrics["task_macro_selective_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
