from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_swe_task_state_v3.py"
PROTOCOL = ROOT / "configs/swe_task_state_interpreter_v3.json"
ACTION_PROTOCOL = ROOT / "configs/swe_task_state_v3_action_probes.json"
OLD_ACTION_PROTOCOL = ROOT / "configs/swe_stage_action_probes.json"
READOUT_PYTHON = ROOT / ".venv-readout-v2/bin/python"
READOUT_PYTHON_TARGET = READOUT_PYTHON.resolve()
MODEL_TEST_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ("sklearn", "joblib")
)


class TestReadoutEnvironmentBridge(unittest.TestCase):
    @unittest.skipIf(
        MODEL_TEST_DEPENDENCIES_AVAILABLE,
        "already running analyzer tests in a model-capable environment",
    )
    def test_analyzer_suite_runs_in_canonical_readout_environment(self):
        self.assertTrue(
            READOUT_PYTHON.is_file()
            and READOUT_PYTHON_TARGET.is_file()
            and os.access(READOUT_PYTHON_TARGET, os.X_OK),
            f"missing executable canonical readout interpreter: {READOUT_PYTHON}",
        )
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment["VIRTUAL_ENV"] = str(READOUT_PYTHON.parents[1])
        environment["PATH"] = os.pathsep.join(
            [str(READOUT_PYTHON.parent), environment.get("PATH", "")]
        )
        completed = subprocess.run(
            [
                str(READOUT_PYTHON),
                "-m",
                "unittest",
                "-v",
                "tests.test_analyze_swe_task_state_v3",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stdout + completed.stderr,
        )


def load_module():
    specification = importlib.util.spec_from_file_location("task_state_v3_test", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def prompt(
    row_id: str,
    request_index: int,
    action: str | None,
    *,
    task_id: str = "repo__task-1",
    repo: str = "repo/repo",
    declared: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "metadata": {
            "task": {
                "instance_id": task_id,
                "repo": repo,
                "probeable_request_indices": declared or [1, 2, 3],
            },
            "selection": {"task_request_index": request_index},
            "labels": {
                "action": {
                    "status": "available" if action is not None else "missing",
                    "class_id": action,
                }
            },
        },
    }


def small_extraction_fixture() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = [
        prompt("p1", 1, "inspect", declared=[1, 2]),
        prompt("p2", 2, None, declared=[1, 2]),
    ]
    for index, row in enumerate(prompts):
        row["text"] = f"prompt {index}"
        row["token_ids"] = [10, 20 + index]
        row["score_token_ids"] = [1, 2, 3]
        row["metadata"]["cohort"] = {"id": "development_a"}
    experiments = [
        {
            "id": row["id"],
            "prompt": row["text"],
            "prompt_token_ids": row["token_ids"],
            "metadata": row["metadata"],
            "capture_positions_resolved": [1],
            "scored_vocabulary": {"token_ids": row["score_token_ids"]},
        }
        for row in prompts
    ]
    return prompts, {"experiments": experiments}


def stub_extraction_helpers(module, monkeypatch) -> None:
    monkeypatch.setattr(
        module.HISTORICAL_V1, "_validate_report_provenance", lambda *a, **k: None
    )
    monkeypatch.setattr(
        module.HISTORICAL_V1, "_numerically_stable", lambda *a, **k: (True, [])
    )

    def fake_layer(*args, method, **kwargs):
        offset = 1.0 if method == "public_jacobian" else -1.0
        return (np.arange(96, dtype=np.float64) + offset).tolist()

    monkeypatch.setattr(module.HISTORICAL_V1, "_layer_class_features", fake_layer)


def synthetic_row(
    module,
    row_id: str,
    task_id: str,
    repo: str,
    label: str | None,
    class_index: int,
) -> dict[str, Any]:
    history = np.zeros(14, dtype=np.float64)
    history[0] = class_index
    j = np.arange(96, dtype=np.float64) * 0.01 + class_index
    logit = np.arange(96, dtype=np.float64) * -0.01 + class_index
    return {
        "row_id": row_id,
        "task_id": task_id,
        "repo": repo,
        "cohort_id": "synthetic",
        "task_request_index": 1,
        "checkpoint_ordinal": 1,
        "source_action_label_status": "available" if label else "missing",
        "source_action_class_id": (
            "validate" if label == "check_or_finish" else label
        ),
        "label_status": "available" if label else "unknown_current_action",
        "label": label,
        "metric_evaluable": label is not None,
        "auxiliary_diagnostics": {
            "milestone_within_2": {
                "status": "unknown",
                "label": None,
                "role": "diagnostic_only_not_an_operational_gate",
            },
            "current_consequential_source_type": {
                "status": "unknown" if label is None else "available",
                "label": label,
                "role": "diagnostic_only_not_an_operational_gate",
            },
        },
        "features": module.build_variant_features(history, j, logit),
    }


def passing_metrics() -> dict[str, Any]:
    return {
        "row_count": 1500,
        "accuracy": 0.9,
        "balanced_accuracy": 0.8,
        "recall_inspect": 0.8,
        "recall_edit": 0.7,
        "recall_check_or_finish": 0.8,
        "multiclass_negative_log_likelihood": 0.4,
        "multiclass_brier": 0.2,
        "top_label_ece": 0.05,
        "selected_coverage": 0.75,
        "selected_accepted_accuracy": 0.9,
        "known_action_fraction": 0.96,
    }


def synthetic_bootstrap_nested_result(module, rows, base_weights):
    """Return compact-test nested outputs with self-consistent row evidence."""

    known_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("metric_evaluable") is True
    ]
    known = [rows[index] for index in known_indices]
    known_index_array = np.asarray(known_indices, dtype=np.int64)
    results = {}
    for variant_index, variant in enumerate(module.VARIANTS):
        confidence = 0.76 + 0.02 * variant_index
        probability_rows = []
        predictions = []
        for row in rows:
            label = row.get("label")
            predicted_index = (
                module.CLASSES.index(label)
                if label in module.CLASSES
                else variant_index % len(module.CLASSES)
            )
            vector = np.full(
                len(module.CLASSES),
                (1.0 - confidence) / (len(module.CLASSES) - 1),
                dtype=np.float64,
            )
            vector[predicted_index] = confidence
            probability_rows.append(vector)
            predictions.append(
                {
                    "row_id": row["row_id"],
                    "probabilities": {
                        class_id: float(vector[class_index])
                        for class_index, class_id in enumerate(module.CLASSES)
                    },
                    "accepted": True,
                }
            )
        probability_array = np.asarray(probability_rows, dtype=np.float64)
        accepted_array = np.ones(len(rows), dtype=bool)
        metrics = module.probability_metrics(
            known,
            probability_array[known_index_array],
            accepted=accepted_array[known_index_array],
            weights=np.asarray(base_weights, dtype=np.float64),
        )
        results[variant] = {
            "metrics": metrics,
            "predictions": predictions,
            "probability_float64_sha256": module.sha256_bytes(
                np.asarray(probability_array, dtype="<f8").tobytes(order="C")
            ),
        }
    return {"results": results}


def synthetic_nested_for_gates(module, metric_overrides=None):
    metrics_by_variant = {}
    for variant in module.VARIANTS:
        metrics = passing_metrics()
        if variant == "history_logit":
            metrics.update(
                {
                    "accuracy": 0.84,
                    "balanced_accuracy": 0.76,
                    "multiclass_negative_log_likelihood": 0.48,
                    "multiclass_brier": 0.27,
                }
            )
        if variant == "history_logit_j":
            metrics.update(
                {
                    "accuracy": 0.91,
                    "balanced_accuracy": 0.81,
                    "multiclass_negative_log_likelihood": 0.38,
                    "multiclass_brier": 0.18,
                }
            )
        metrics_by_variant[variant] = {"metrics": metrics}
    if metric_overrides:
        metrics_by_variant["history_j"]["metrics"].update(metric_overrides)
    return {
        "all_stable_feature_complete_rows_predicted_once": True,
        "results": metrics_by_variant,
    }


def passing_support():
    return {
        "stable_prediction_rows": 1600,
        "numerically_stable_final_prompt_rows": 1700,
        "stable_feature_complete_prediction_rows": 1600,
        "stable_feature_complete_prediction_fraction": 1600 / 1700,
        "stable_feature_complete_prediction_fraction_numerator": 1600,
        "stable_feature_complete_prediction_fraction_denominator": 1700,
        "known_action_rows": 1540,
        "unknown_action_rows": 60,
        "prediction_tasks": 60,
        "prediction_repositories": 10,
        "known_action_tasks": 60,
        "known_action_repositories": 10,
        "known_action_fraction": 0.9625,
        "hierarchical_known_action_fraction": 0.9625,
        "known_inspect_tasks": 20,
        "known_edit_tasks": 20,
        "known_check_or_finish_tasks": 20,
        "known_inspect_repositories": 10,
        "known_edit_repositories": 10,
        "known_check_or_finish_repositories": 10,
        "numerical_stability_fraction": 0.95,
    }


def passing_bootstrap(module):
    intervals = {
        variant: {
            metric: {
                "lower": (
                    0.01
                    if metric == "top_label_ece"
                    else 0.1
                    if metric in {
                        "multiclass_negative_log_likelihood",
                        "multiclass_brier",
                    }
                    else 0.7
                ),
                "upper": (
                    0.6
                    if metric == "multiclass_negative_log_likelihood"
                    else 0.3
                    if metric == "multiclass_brier"
                    else 0.09
                    if metric == "top_label_ece"
                    else 0.98
                ),
            }
            for metric in module.BOOTSTRAP_METRICS
        }
        for variant in module.VARIANTS
    }
    intervals["history_j"]["balanced_accuracy"] = {
        "lower": 0.72,
        "upper": 0.86,
    }
    intervals["history_j"]["selected_accepted_accuracy"] = {
        "lower": 0.82,
        "upper": 0.95,
    }
    intervals["history_j"]["top_label_ece"] = {
        "lower": 0.02,
        "upper": 0.09,
    }
    pairs = {}
    for candidate, reference in module.PAIRED_COMPARISONS:
        name = f"{candidate}_minus_{reference}"
        pairs[name] = {
            "accuracy": {"lower": 0.01, "upper": 0.08},
            "balanced_accuracy": {"lower": 0.005, "upper": 0.09},
            "multiclass_negative_log_likelihood": {
                "lower": -0.15,
                "upper": -0.005,
            },
            "multiclass_brier": {"lower": -0.12, "upper": -0.004},
        }
    return {
        "status": "complete_hierarchical_bayesian_bootstrap",
        "models_refit_inside_bootstrap": True,
        "calibration_and_threshold_reselected_inside_each_draw": True,
        "intervals": intervals,
        "paired_differences": pairs,
    }


class _PatchManager:
    """Small unittest.mock-backed replacement for per-test attribute patching."""

    def __init__(self) -> None:
        self._patchers: list[Any] = []

    def setattr(self, target: object, name: str, value: object) -> None:
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        self._patchers.append(patcher)

    def stop(self) -> None:
        for patcher in reversed(self._patchers):
            patcher.stop()
        self._patchers.clear()


@unittest.skipUnless(
    MODEL_TEST_DEPENDENCIES_AVAILABLE,
    "analyzer model tests require the frozen readout environment",
)
class TestAnalyzeSweTaskStateV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.normalized = cls.module.validate_protocol(
            json.loads(PROTOCOL.read_text()),
            action_protocol_value=json.loads(ACTION_PROTOCOL.read_text()),
        )

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temporary_directory.name)
        self.monkeypatch = _PatchManager()
        self.addCleanup(self._temporary_directory.cleanup)
        self.addCleanup(self.monkeypatch.stop)

    def test_action_protocol_is_exact_semantic_copy_plus_requested_forms(self):
        module = self.module

        old = json.loads(OLD_ACTION_PROTOCOL.read_text())
        new = json.loads(ACTION_PROTOCOL.read_text())
        additions = {
            "inspect": [(" grep", 20049), (" rg", 17815)],
            "edit": [(" write", 3165), (" replace", 8032)],
            "validate": [(" pytest", 26864), (" run", 1542)],
            "finalize": [(" answer", 4087), (" done", 2725)],
        }
        for old_class, new_class in zip(
            old["action_classes"], new["action_classes"], strict=True
        ):
            assert new_class["tokens"][:6] == old_class["tokens"]
            assert [
                (row["text"], row["token_id"]) for row in new_class["tokens"][6:]
            ] == additions[old_class["id"]]
            assert len(new_class["tokens"]) == 8
        trimmed = copy.deepcopy(new)
        for action_class in trimmed["action_classes"]:
            del action_class["tokens"][6:]
        assert trimmed == old
        assert module.sha256_file(ACTION_PROTOCOL) == module.V3_ACTION_PROTOCOL_SHA256

    def test_protocol_authenticates_all_published_helpers_and_configs(self):
        module = self.module
        normalized = self.normalized

        assert normalized["class_ids"] == list(module.CLASSES)
        assert normalized["model"]["parameters"]["n_jobs"] == 1
        assert normalized["model"]["fit_execution"] == {
            "parallel_unit": "one_variant_seed_estimator",
            "backend": "sklearn_joblib_loky_processes",
            "worker_count": 20,
            "estimator_fit_n_jobs": 1,
            "persisted_estimator_n_jobs": 1,
            "submission_order": "variant_then_seed",
            "result_collection_order": "variant_then_seed",
            "deterministic_ordered_collection": True,
        }
        assert normalized["model"]["prediction_execution"][
            "tree_probability_reduction_order"
        ] == "serial_estimator_order"
        assert normalized["model"]["prediction_execution"][
            "parallel_prediction_forbidden"
        ] is True
        for path, expected in (
            (module.V1_ANALYZER_PATH, module.V1_ANALYZER_SHA256),
            (module.V2_ANALYZER_PATH, module.V2_ANALYZER_SHA256),
            (module.V1_PROTOCOL_PATH, module.V1_PROTOCOL_SHA256),
            (module.V2_PROTOCOL_PATH, module.V2_PROTOCOL_SHA256),
            (
                module.HISTORICAL_ACTION_PROTOCOL_PATH,
                module.HISTORICAL_ACTION_PROTOCOL_SHA256,
            ),
            (module.BEHAVIORAL_PROTOCOL_PATH, module.BEHAVIORAL_PROTOCOL_SHA256),
            (module.COHORT_CHECKER_PATH, module.COHORT_CHECKER_SHA256),
            (module.V3_MATERIALIZER_PATH, module.V3_MATERIALIZER_SHA256),
            (
                module.HISTORICAL_MATERIALIZER_PATH,
                module.HISTORICAL_MATERIALIZER_SHA256,
            ),
            (module.REPLAY_PIPELINE_PATH, module.REPLAY_PIPELINE_SHA256),
            (
                module.REPLAY_SHELL_WRAPPER_PATH,
                module.REPLAY_SHELL_WRAPPER_SHA256,
            ),
            (module.V3_REQUIREMENTS_PATH, module.V3_REQUIREMENTS_SHA256),
            (
                module.DEVELOPMENT_SELECTION_PROOF_PATH,
                module.DEVELOPMENT_SELECTION_PROOF_SHA256,
            ),
        ):
            assert module.sha256_file(path) == expected
        runtime_pins = json.loads(PROTOCOL.read_text())["pins"][
            "analyzer_runtime"
        ]
        assert runtime_pins == {
            "numpy": module.BOOTSTRAP_NUMPY_VERSION,
            "scikit-learn": module.FROZEN_SKLEARN_VERSION,
            "joblib": module.FROZEN_JOBLIB_VERSION,
            "scipy": module.FROZEN_SCIPY_VERSION,
            "threadpoolctl": module.FROZEN_THREADPOOLCTL_VERSION,
            "ijson": module.FROZEN_IJSON_VERSION,
            "zlib": module.FROZEN_ZLIB_VERSION,
        }
        runtime_tamper = json.loads(PROTOCOL.read_text())
        runtime_tamper["pins"]["analyzer_runtime"]["numpy"] = "0.0.0"
        with self.assertRaisesRegex(ValueError, "predeclared canonical"):
            module.validate_protocol(
                runtime_tamper,
                action_protocol_value=json.loads(ACTION_PROTOCOL.read_text()),
            )
        mutated = json.loads(PROTOCOL.read_text())
        mutated["feature_contract"]["history_width"] = 15
        with self.assertRaisesRegex(ValueError, "predeclared canonical"):
            module.validate_protocol(
                mutated, action_protocol_value=json.loads(ACTION_PROTOCOL.read_text())
            )

    def test_bootstrap_evidence_scope_discloses_honest_execution_trust_boundary(self):
        module = self.module
        protocol = json.loads(PROTOCOL.read_text())
        expected_scope = module._expected_bootstrap_evidence_scope()
        assert protocol["bootstrap"]["persisted_evidence_scope"] == expected_scope
        assert expected_scope["retained_bytes_prove"] == (
            "identity_bound_row_weight_probability_acceptance_metric_pair_and_"
            "interval_self_consistency"
        )
        assert expected_scope["models_refit_inside_bootstrap_field"] == (
            "in_process_execution_declaration_by_the_frozen_analyzer_not_"
            "persisted_refit_proof"
        )
        assert expected_scope["cryptographic_refit_attestation"] is False
        assert expected_scope["independent_refit_attestation"] is False
        assert (
            expected_scope[
                "label_informed_self_consistent_malicious_artifact_authors"
            ]
            == "out_of_scope"
        )
        assert (
            expected_scope[
                "missing_or_internally_inconsistent_1000_draw_evidence_fails_"
                "bound_gates_closed"
            ]
            is True
        )
        assert (
            expected_scope["reserved_validation_data_used_for_evidence_or_gates"]
            is False
        )
        output_contract = protocol["analyzer_output_contract"]
        assert output_contract[
            "only_an_identity_bound_self_consistency_validated_matching_"
            "bootstrap_checkpoint_may_be_updated_on_resume"
        ] is True
        assert not any("authenticated_matching_bootstrap" in key for key in output_contract)
        missing = module.missing_model_refit_bootstrap("test omission")
        assert missing["persisted_evidence_scope"] == expected_scope
        assert missing["models_refit_inside_bootstrap"] is False
        assert (
            missing[
                "models_refit_inside_bootstrap_is_in_process_execution_"
                "declaration_only"
            ]
            is True
        )

    def test_prepare_inputs_rejects_each_frozen_runtime_version_drift(self):
        module = self.module
        empty_args = module.argparse.Namespace()
        original_version = module.importlib.metadata.version

        with self.subTest(package="numpy"), mock.patch.object(
            module.np, "__version__", "0.0.0"
        ):
            with self.assertRaisesRegex(
                ValueError, "exact frozen model/analyzer runtime"
            ):
                module._prepare_inputs(empty_args)

        with self.subTest(package="zlib"), mock.patch.object(
            module.zlib, "ZLIB_RUNTIME_VERSION", "0.0.0"
        ):
            with self.assertRaisesRegex(
                ValueError, "exact frozen model/analyzer runtime"
            ):
                module._prepare_inputs(empty_args)

        for package in (
            "scikit-learn",
            "joblib",
            "scipy",
            "threadpoolctl",
            "ijson",
        ):
            with self.subTest(package=package):
                def drifted_version(name, *, drifted_package=package):
                    if name == drifted_package:
                        return "0.0.0"
                    return original_version(name)

                with mock.patch.object(
                    module.importlib.metadata, "version", drifted_version
                ):
                    with self.assertRaisesRegex(
                        ValueError, "exact frozen model/analyzer runtime"
                    ):
                        module._prepare_inputs(empty_args)

    def test_feature_order_and_widths_are_exact(self):
        module = self.module

        history = np.arange(14, dtype=np.float64)
        public = np.arange(96, dtype=np.float64) + 100.0
        logit = np.arange(96, dtype=np.float64) - 100.0
        features = module.build_variant_features(history, public, logit)
        public_compact = module.compact_layer_shape(public)
        logit_compact = module.compact_layer_shape(logit)
        assert list(features) == list(module.VARIANTS)
        assert features["history_only"] == history.tolist()
        np.testing.assert_array_equal(features["history_j"][:14], history)
        np.testing.assert_array_equal(features["history_j"][14:], public_compact)
        np.testing.assert_array_equal(features["history_logit"][:14], history)
        np.testing.assert_array_equal(features["history_logit"][14:], logit_compact)
        np.testing.assert_array_equal(features["history_logit_j"][:14], history)
        np.testing.assert_array_equal(
            features["history_logit_j"][14:54], logit_compact
        )
        np.testing.assert_array_equal(
            features["history_logit_j"][54:94], public_compact
        )
        assert {key: len(value) for key, value in features.items()} == module.VARIANT_WIDTHS
        assert module.feature_names("history_j")[14:18] == [
            "current_public_jacobian__mean_all_layers__inspect",
            "current_public_jacobian__mean_all_layers__edit",
            "current_public_jacobian__mean_all_layers__validate",
            "current_public_jacobian__mean_all_layers__finalize",
        ]

    def test_history_is_computed_before_current_action(self):
        module = self.module

        prompts = [
            prompt("p1", 1, "edit"),
            prompt("p2", 2, "validate"),
            prompt("p3", 3, None),
        ]
        history, coverage = module.causal_history_features(prompts)
        assert coverage["complete_history_task_count"] == 1
        assert history["p1"][:10] == [0.0] * 10
        assert history["p1"][10:] == [0.0, 0.0, -1.0, -1.0]
        self.assertAlmostEqual(history["p2"][1], math.log(2.0))
        assert history["p2"][5] == 1.0
        assert history["p2"][10] == 1.0
        assert history["p2"][12] == 1.0
        assert history["p2"][2] == 0.0
        assert history["p2"][6] == 0.0
        assert history["p2"][11] == 0.0

    def test_auxiliary_fixed_two_completion_boundaries_and_unknowns(self):
        module = self.module

        prompts = [
            prompt("p1", 1, "inspect", declared=[1, 2, 3, 4, 5]),
            prompt("p2", 2, "inspect", declared=[1, 2, 3, 4, 5]),
            prompt("p3", 3, "edit", declared=[1, 2, 3, 4, 5]),
            prompt("p4", 4, None, declared=[1, 2, 3, 4, 5]),
            prompt("p5", 5, "validate", declared=[1, 2, 3, 4, 5]),
        ]
        labels = module.auxiliary_diagnostic_labels(prompts)
        assert labels["p1"]["milestone_within_2"]["label"] == "none"
        assert labels["p1"]["milestone_within_2"]["observed_actions_until_resolution"] == [
            "inspect",
            "inspect",
        ]
        assert labels["p2"]["milestone_within_2"]["label"] == "edit"
        assert labels["p2"]["milestone_within_2"]["resolved_offset"] == 1
        assert labels["p3"]["milestone_within_2"]["label"] == "edit"
        assert labels["p3"]["milestone_within_2"]["resolved_offset"] == 0
        assert labels["p4"]["milestone_within_2"]["status"] == "unknown"
        assert (
            labels["p4"]["milestone_within_2"]["reason"]
            == "unclassified_action_before_milestone"
        )
        assert labels["p5"]["milestone_within_2"]["label"] == "check_or_finish"
        assert labels["p5"]["milestone_within_2"]["resolved_offset"] == 0
        assert labels["p1"]["current_consequential_source_type"]["label"] == "none"
        assert labels["p5"]["current_consequential_source_type"]["label"] == "check_or_finish"

    def test_auxiliary_never_skips_inspections_beyond_offset_one(self):
        module = self.module

        prompts = [
            prompt("p1", 1, "inspect"),
            prompt("p2", 2, "inspect"),
            prompt("p3", 3, "edit"),
        ]
        labels = module.auxiliary_diagnostic_labels(prompts)
        assert labels["p1"]["milestone_within_2"]["label"] == "none"
        assert labels["p2"]["milestone_within_2"]["label"] == "edit"
        lone = module.auxiliary_diagnostic_labels(
            [prompt("only", 1, "inspect", declared=[1])]
        )
        assert lone["only"]["milestone_within_2"]["status"] == "unknown"
        assert "incomplete" in lone["only"]["milestone_within_2"]["reason"]

    def test_future_diagnostic_changes_cannot_change_prior_features(self):
        module = self.module

        original = [
            prompt("p1", 1, "inspect"),
            prompt("p2", 2, "inspect"),
            prompt("p3", 3, "edit"),
        ]
        mutated = copy.deepcopy(original)
        mutated[2]["metadata"]["labels"]["action"] = {
            "status": "available",
            "class_id": "finalize",
        }
        before, _ = module.causal_history_features(original)
        after, _ = module.causal_history_features(mutated)
        assert before["p1"] == after["p1"]
        assert before["p2"] == after["p2"]
        assert (
            module.auxiliary_diagnostic_labels(original)["p2"]["milestone_within_2"][
                "label"
            ]
            == "edit"
        )
        assert (
            module.auxiliary_diagnostic_labels(mutated)["p2"]["milestone_within_2"][
                "label"
            ]
            == "check_or_finish"
        )

    def test_probability_floor_is_nonzero_normalized_and_deterministic(self):
        module = self.module

        raw = np.asarray([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]], dtype=np.float64)
        first = module.apply_probability_floor(raw, 1e-6)
        second = module.apply_probability_floor(raw, 1e-6)
        assert np.all(first > 0.0)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first.sum(axis=1), 1.0)

    def test_temperature_tie_break_is_numeric_and_deterministic(self):
        module = self.module

        rows = [
            synthetic_row(
                module,
                f"row-{index}",
                f"task-{index}",
                "repo/repo",
                label,
                index,
            )
            for index, label in enumerate(module.CLASSES)
        ]
        uniform = np.full((len(rows), len(module.CLASSES)), 1.0 / 3.0)
        first = module.select_temperature(rows, uniform, [1.25, 0.75])
        second = module.select_temperature(rows, uniform, [0.75, 1.25])
        assert first["candidates"][0]["nll"] == first["candidates"][1]["nll"]
        assert first["temperature"] == 0.75
        assert second["temperature"] == 0.75

    def test_extraction_emits_unknown_current_action_without_metric_label(self):
        module = self.module
        normalized = self.normalized
        monkeypatch = self.monkeypatch

        prompts, report = small_extraction_fixture()
        stub_extraction_helpers(module, monkeypatch)
        extracted = module.extract_stable_rows(
            prompts,
            report,
            protocol=normalized,
        )
        assert len(extracted["rows"]) == 2
        assert extracted["rows"][0]["label"] == "inspect"
        assert extracted["rows"][0]["metric_evaluable"] is True
        assert extracted["rows"][1]["label"] is None
        assert extracted["rows"][1]["metric_evaluable"] is False
        assert extracted["rows"][1]["label_status"] == "unknown_current_action"
        assert extracted["eligibility"]["known_current_action_prediction_count"] == 1
        assert extracted["eligibility"]["unknown_current_action_prediction_count"] == 1
        assert (
            extracted["eligibility"]["stable_feature_complete_prediction_fraction"]
            == 1.0
        )
        assert (
            extracted["eligibility"][
                "stable_feature_complete_prediction_fraction_numerator"
            ]
            == 2
        )
        assert (
            extracted["eligibility"][
                "stable_feature_complete_prediction_fraction_denominator"
            ]
            == 2
        )
        assert extracted["eligibility"]["predictions_emitted_for_unknown_current_actions"] is True

    def test_streaming_extraction_exactly_matches_in_memory_reference(self):
        module = self.module
        normalized = self.normalized
        monkeypatch = self.monkeypatch
        tmp_path = self.tmp_path

        prompts, report = small_extraction_fixture()
        report["chunk_provenance"] = {
            "source_chunk_sha256s": ["a" * 64, "b" * 64]
        }
        stub_extraction_helpers(module, monkeypatch)
        expected = module.extract_stable_rows(prompts, report, protocol=normalized)
        prompts_path = tmp_path / "prompts.json"
        report_path = tmp_path / "report.json"
        prompts_path.write_text(json.dumps(prompts))
        report_path.write_text(json.dumps(report))

        observed = module.extract_stable_rows_streaming(
            prompts_path, report_path, protocol=normalized
        )

        assert observed == expected

    def test_streaming_extraction_rejects_duplicate_rows(self):
        module = self.module
        normalized = self.normalized
        monkeypatch = self.monkeypatch
        tmp_path = self.tmp_path

        prompts, report = small_extraction_fixture()
        stub_extraction_helpers(module, monkeypatch)
        prompts_path = tmp_path / "prompts.json"
        report_path = tmp_path / "report.json"

        duplicate_prompts = copy.deepcopy(prompts)
        duplicate_prompts[1]["id"] = duplicate_prompts[0]["id"]
        prompts_path.write_text(json.dumps(duplicate_prompts))
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "prompt IDs are duplicated"):
            module.extract_stable_rows_streaming(
                prompts_path, report_path, protocol=normalized
            )

        duplicate_report = copy.deepcopy(report)
        duplicate_report["experiments"][1]["id"] = duplicate_report["experiments"][0][
            "id"
        ]
        prompts_path.write_text(json.dumps(prompts))
        report_path.write_text(json.dumps(duplicate_report))
        with self.assertRaisesRegex(ValueError, "experiment IDs are duplicated"):
            module.extract_stable_rows_streaming(
                prompts_path, report_path, protocol=normalized
            )

    def test_streaming_extraction_rejects_misaligned_and_trailing_rows(self):
        module = self.module
        normalized = self.normalized
        tmp_path = self.tmp_path

        prompts, report = small_extraction_fixture()
        prompts_path = tmp_path / "prompts.json"
        report_path = tmp_path / "report.json"
        prompts_path.write_text(json.dumps(prompts))

        misaligned = copy.deepcopy(report)
        misaligned["experiments"].reverse()
        report_path.write_text(json.dumps(misaligned))
        with self.assertRaisesRegex(ValueError, "IDs or order differ"):
            module.extract_stable_rows_streaming(
                prompts_path, report_path, protocol=normalized
            )

        report_path.write_text(json.dumps({"experiments": []}))
        with self.assertRaisesRegex(ValueError, "trailing prompt rows"):
            module.extract_stable_rows_streaming(
                prompts_path, report_path, protocol=normalized
            )

        prompts_path.write_text("[]\n")
        report_path.write_text(json.dumps({"experiments": report["experiments"]}))
        with self.assertRaisesRegex(ValueError, "trailing experiment rows"):
            module.extract_stable_rows_streaming(
                prompts_path, report_path, protocol=normalized
            )

    def test_streaming_extraction_rejects_trailing_json_document(self):
        module = self.module
        normalized = self.normalized
        tmp_path = self.tmp_path

        prompts, report = small_extraction_fixture()
        prompts_path = tmp_path / "prompts.json"
        report_path = tmp_path / "report.json"
        prompts_path.write_text(json.dumps(prompts) + "\n{}\n")
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "could not stream prompt bundle|trailing"):
            module.extract_stable_rows_streaming(
                prompts_path, report_path, protocol=normalized
            )

    def test_training_weights_are_task_deterministic_and_class_balanced(self):
        module = self.module

        rows = []
        for task_id, labels in (
            ("t1", ["inspect", "inspect", "edit"]),
            ("t2", ["inspect", "check_or_finish", "check_or_finish"]),
        ):
            for index, label in enumerate(labels):
                rows.append(
                    synthetic_row(
                        module,
                        f"{task_id}-{index}",
                        task_id,
                        "r/r",
                        label,
                        module.CLASSES.index(label),
                    )
                )
        first, diagnostics = module.training_weights(rows)
        second, diagnostics_second = module.training_weights(rows)
        np.testing.assert_array_equal(first, second)
        assert diagnostics == diagnostics_second
        labels = module.labels_for(rows)
        for class_id in module.CLASSES:
            self.assertAlmostEqual(first[labels == class_id].sum(), 1.0 / 3.0)

    def test_prebalanced_fit_weights_preserve_bits_without_second_normalization(self):
        module = self.module
        normalized = self.normalized
        rows = [
            synthetic_row(
                module,
                f"row-{index}",
                f"task-{index}",
                f"repo-{index}",
                module.CLASSES[index // 2],
                index // 2,
            )
            for index in range(6)
        ]
        supplied = np.full(6, 1.0 / 6.0, dtype=np.float64)
        self.assertNotEqual(float(supplied.sum(dtype=np.float64)), 1.0)

        _, observed, diagnostics, _, _ = module._prepare_ensemble_fit(
            module.matrix_for(rows, "history_j"),
            rows,
            protocol=normalized,
            weights=supplied,
        )
        np.testing.assert_array_equal(observed, supplied)
        self.assertIsNot(observed, supplied)
        supplied_copy = supplied.copy()
        observed[0] *= 0.5
        np.testing.assert_array_equal(supplied, supplied_copy)
        self.assertEqual(
            diagnostics["weight_float64_sha256"],
            module.sha256_bytes(np.asarray(supplied, dtype="<f8").tobytes(order="C")),
        )

        with self.assertRaisesRegex(ValueError, "near-unit mass"):
            module._prepare_ensemble_fit(
                module.matrix_for(rows, "history_j"),
                rows,
                protocol=normalized,
                weights=supplied * 2.0,
            )
        unbalanced = supplied.copy()
        unbalanced[0] += 1e-4
        unbalanced[2] -= 1e-4
        with self.assertRaisesRegex(ValueError, "three-class rebalance"):
            module._prepare_ensemble_fit(
                module.matrix_for(rows, "history_j"),
                rows,
                protocol=normalized,
                weights=unbalanced,
            )

    def test_crossfit_preserves_master_weight_bits_before_split_restriction(self):
        module = self.module
        local = copy.deepcopy(self.normalized)
        local["model"] = copy.deepcopy(local["model"])
        local["model"]["seeds"] = [271828]
        local["model"]["parameters"] = copy.deepcopy(local["model"]["parameters"])
        local["model"]["parameters"]["n_estimators"] = 4
        local["model"]["fit_execution"] = copy.deepcopy(
            local["model"]["fit_execution"]
        )
        local["model"]["fit_execution"]["worker_count"] = len(module.VARIANTS)
        rows = [
            synthetic_row(
                module,
                f"repo-{repo_index}-{class_id}",
                f"task-{repo_index}-{class_id}",
                f"repo-{repo_index}",
                class_id,
                class_index,
            )
            for repo_index in range(6)
            for class_index, class_id in enumerate(module.CLASSES)
        ]
        base_weights = np.full(len(rows), 1.0 / len(rows), dtype=np.float64)
        base_weights *= np.nextafter(1.0, np.inf)
        self.assertNotEqual(float(base_weights.sum(dtype=np.float64)), 1.0)

        observed = module.crossfit_raw_probabilities(
            rows, protocol=local, base_weights=base_weights
        )
        repositories = np.asarray([str(row["repo"]) for row in rows])
        for fold in observed["folds"]:
            train_indices = np.flatnonzero(
                repositories != str(fold["heldout_repository"])
            )
            expected = module.restrict_base_weights(
                rows, base_weights, train_indices
            )
            self.assertEqual(
                fold["training_base_weight_sha256"],
                module.sha256_bytes(
                    np.asarray(expected, dtype="<f8").tobytes(order="C")
                ),
            )

    def test_nested_loro_emits_unknown_predictions_and_isolates_outer_fold(self):
        module = self.module
        normalized = self.normalized
        monkeypatch = self.monkeypatch

        rows = []
        for repo_index in range(6):
            repo = f"repo{repo_index}/repo{repo_index}"
            for class_index, label in enumerate(module.CLASSES):
                rows.append(
                    synthetic_row(
                        module,
                        f"r{repo_index}-{label}",
                        f"t{repo_index}-{label}",
                        repo,
                        label,
                        class_index,
                    )
                )
            rows.append(
                synthetic_row(
                    module,
                    f"r{repo_index}-unknown",
                    f"t{repo_index}-unknown",
                    repo,
                    None,
                    0,
                )
            )

        inner_inputs: list[set[str]] = []

        def probabilities_from_rows(input_rows):
            values = np.full((len(input_rows), 3), 0.1, dtype=np.float64)
            for index, row in enumerate(input_rows):
                values[index, module.CLASSES.index(str(row["label"]))] = 0.8
            return values

        def fake_crossfit(input_rows, *, protocol, base_weights=None):
            inner_inputs.append({str(row["repo"]) for row in input_rows})
            probabilities = probabilities_from_rows(input_rows)
            repository_values = np.asarray(
                [str(row["repo"]) for row in input_rows]
            )
            master_weights = (
                module.hierarchical_equal_weights(input_rows)
                if base_weights is None
                else np.asarray(base_weights, dtype=np.float64)
            )
            folds = []
            for heldout_repository in sorted(set(repository_values.tolist())):
                train_indices = np.flatnonzero(
                    repository_values != heldout_repository
                )
                evaluation_indices = np.flatnonzero(
                    repository_values == heldout_repository
                )
                train_rows = [input_rows[int(index)] for index in train_indices]
                evaluation_rows = [
                    input_rows[int(index)] for index in evaluation_indices
                ]
                train_base_weights = module.restrict_base_weights(
                    input_rows, master_weights, train_indices
                )
                training_diagnostics = module.training_weights(
                    train_rows, train_base_weights
                )[1]
                folds.append(
                    {
                        "heldout_repository": heldout_repository,
                        "training_repositories": sorted(
                            set(repository_values[train_indices].tolist())
                        ),
                        "training_rows": len(train_indices),
                        "evaluation_rows": len(evaluation_indices),
                        "training_row_ids_sha256": module.canonical_json_sha256(
                            [str(row["row_id"]) for row in train_rows]
                        ),
                        "evaluation_row_ids_sha256": module.canonical_json_sha256(
                            [str(row["row_id"]) for row in evaluation_rows]
                        ),
                        "heldout_labels_used_for_fit_calibration_or_threshold_selection": False,
                        "shared_training_weight_sha256": training_diagnostics[
                            "weight_float64_sha256"
                        ],
                        "training_base_weight_sha256": module.sha256_bytes(
                            np.asarray(train_base_weights, dtype="<f8").tobytes(
                                order="C"
                            )
                        ),
                        "seed_order": list(protocol["model"]["seeds"]),
                    }
                )
            return {
                "probabilities": {
                    variant: probabilities.copy() for variant in module.VARIANTS
                },
                "folds": folds,
            }

        def fake_fit_all(input_rows, *, protocol, base_weights=None):
            digest = module.training_weights(input_rows, base_weights)[1][
                "weight_float64_sha256"
            ]
            return (
                {variant: [variant] for variant in module.VARIANTS},
                {"shared": {"weight_float64_sha256": digest}},
            )

        def fake_aligned(models, x, *, probability_floor):
            values = np.full((len(x), 3), 0.1, dtype=np.float64)
            for index, feature in enumerate(x):
                values[index, int(round(feature[0])) % 3] = 0.8
            return module.apply_probability_floor(values, probability_floor)

        monkeypatch.setattr(module, "crossfit_raw_probabilities", fake_crossfit)
        monkeypatch.setattr(module, "_fit_all_variants", fake_fit_all)
        monkeypatch.setattr(module, "aligned_ensemble_probabilities", fake_aligned)
        result = module.nested_leave_one_repository_out(rows, protocol=normalized)
        assert result["all_stable_feature_complete_rows_predicted_once"] is True
        assert len(result["folds"]) == 6
        assert len(inner_inputs) == 6
        for fold, repositories_used in zip(result["folds"], inner_inputs, strict=True):
            assert fold["heldout_repository"] not in repositories_used
            assert fold["inner_and_heldout_row_ids_disjoint"] is True
            assert fold["heldout_labels_used_for_fit_calibration_or_threshold_selection"] is False
        for variant in module.VARIANTS:
            variant_result = result["results"][variant]
            assert variant_result["inference_prediction_count"] == 24
            assert variant_result["known_action_metric_row_count"] == 18
            assert variant_result["unknown_action_prediction_count"] == 6
            assert (
                variant_result["metrics"]["selected_coverage_denominator"]
                == "known_current_action_metric_rows_only_not_all_stable_emissions"
            )
            assert (
                variant_result["inference_acceptance_fraction_denominator"]
                == "all_stable_feature_complete_predictions"
            )
            unknown_predictions = [
                row
                for row in variant_result["predictions"]
                if row["metric_evaluable"] is False
            ]
            assert len(unknown_predictions) == 6
            assert all(
                set(row["probabilities"]) == set(module.CLASSES)
                for row in unknown_predictions
            )

        validated = module.validate_nested_evidence(
            result,
            rows=rows,
            protocol=normalized,
            require_primary_selection_floors=False,
        )
        assert validated["full_oof_selection_recomputed"] is True
        assert validated["prediction_count_per_variant"] == len(rows)

        metric_tamper = copy.deepcopy(result)
        metric_tamper["results"]["history_j"]["metrics"]["accuracy"] += 0.01
        with self.assertRaisesRegex(ValueError, "nested metric accuracy"):
            module.validate_nested_evidence(
                metric_tamper,
                rows=rows,
                protocol=normalized,
                require_primary_selection_floors=False,
            )

        setting_tamper = copy.deepcopy(result)
        setting_tamper["folds"][0]["settings"]["history_j"]["calibration"][
            "temperature"
        ] = 123.0
        with self.assertRaisesRegex(ValueError, "temperature is off the frozen grid"):
            module.validate_nested_evidence(
                setting_tamper,
                rows=rows,
                protocol=normalized,
                require_primary_selection_floors=False,
            )

        full_selection_off_grid = copy.deepcopy(result)
        full_selection_off_grid["full_development_selection"]["history_j"][
            "abstention"
        ]["threshold"] = 0.123456
        with self.assertRaisesRegex(ValueError, "threshold is off the frozen grid"):
            module.validate_nested_evidence(
                full_selection_off_grid,
                rows=rows,
                protocol=normalized,
                require_primary_selection_floors=False,
            )

        fallback_protocol = copy.deepcopy(normalized)
        fallback_protocol["abstention"] = copy.deepcopy(normalized["abstention"])
        fallback_protocol["abstention"][
            "minimum_accepted_rows_per_class"
        ] = 100
        fallback_result = module.nested_leave_one_repository_out(
            rows, protocol=fallback_protocol
        )
        assert (
            fallback_result["full_development_selection"]["history_j"][
                "abstention"
            ]["fallback_used"]
            is True
        )
        with self.assertRaisesRegex(ValueError, "did not meet frozen floors"):
            module.validate_nested_evidence(
                fallback_result,
                rows=rows,
                protocol=fallback_protocol,
            )

    def test_real_multiseed_ensemble_is_deterministic(self):
        module = self.module
        normalized = self.normalized

        local = copy.deepcopy(normalized)
        local["model"] = copy.deepcopy(normalized["model"])
        local["model"]["seeds"] = [11, 29]
        local["model"]["parameters"] = copy.deepcopy(normalized["model"]["parameters"])
        local["model"]["parameters"]["n_estimators"] = 8
        local["model"]["parameters"]["n_jobs"] = 1
        rows = []
        for task_index in range(9):
            label = module.CLASSES[task_index % 3]
            rows.append(
                synthetic_row(
                    module,
                    f"row-{task_index}",
                    f"task-{task_index}",
                    f"repo-{task_index % 3}",
                    label,
                    task_index % 3,
                )
            )
        x = module.matrix_for(rows, "history_j")
        first_models, first_diagnostics = module.fit_ensemble(
            x, rows, protocol=local
        )
        second_models, second_diagnostics = module.fit_ensemble(
            x, rows, protocol=local
        )
        first = module.aligned_ensemble_probabilities(
            first_models, x, probability_floor=local["model"]["probability_floor"]
        )
        second = module.aligned_ensemble_probabilities(
            second_models, x, probability_floor=local["model"]["probability_floor"]
        )
        np.testing.assert_array_equal(first, second)
        assert first_diagnostics == second_diagnostics
        assert np.all(first > 0.0)

    def test_fit_thread_count_preserves_trees_and_serial_prediction_bits(self):
        module = self.module
        normalized = self.normalized

        local = copy.deepcopy(normalized)
        local["model"] = copy.deepcopy(normalized["model"])
        local["model"]["seeds"] = [271828]
        local["model"]["parameters"] = copy.deepcopy(
            normalized["model"]["parameters"]
        )
        local["model"]["parameters"]["n_estimators"] = 16
        rows = [
            synthetic_row(
                module,
                f"row-{task_index}",
                f"task-{task_index}",
                f"repo-{task_index % 3}",
                module.CLASSES[task_index % 3],
                task_index % 3,
            )
            for task_index in range(30)
        ]
        for task_index, row in enumerate(rows):
            row["features"]["history_j"][0] += task_index / 1000.0
            row["features"]["history_j"][-1] -= task_index / 2000.0
        x = module.matrix_for(rows, "history_j")
        labels, weights, _, parameters, seeds = module._prepare_ensemble_fit(
            x, rows, protocol=local, weights=None
        )
        ExtraTreesClassifier, _, _ = module._ml_dependencies()
        fit_parallel = module._fit_seed_estimator(
            ExtraTreesClassifier,
            x,
            labels,
            weights,
            parameters=parameters,
            seed=seeds[0],
            fit_n_jobs=8,
        )
        fit_serial = module._fit_seed_estimator(
            ExtraTreesClassifier,
            x,
            labels,
            weights,
            parameters=parameters,
            seed=seeds[0],
            fit_n_jobs=1,
        )
        assert fit_parallel.n_jobs == fit_serial.n_jobs == 1
        for parallel_tree, serial_tree in zip(
            fit_parallel.estimators_, fit_serial.estimators_, strict=True
        ):
            np.testing.assert_array_equal(
                parallel_tree.tree_.children_left, serial_tree.tree_.children_left
            )
            np.testing.assert_array_equal(
                parallel_tree.tree_.children_right, serial_tree.tree_.children_right
            )
            np.testing.assert_array_equal(
                parallel_tree.tree_.feature, serial_tree.tree_.feature
            )
            np.testing.assert_array_equal(
                parallel_tree.tree_.threshold, serial_tree.tree_.threshold
            )
            np.testing.assert_array_equal(
                parallel_tree.tree_.value, serial_tree.tree_.value
            )
            np.testing.assert_array_equal(
                parallel_tree.tree_.weighted_n_node_samples,
                serial_tree.tree_.weighted_n_node_samples,
            )
        predictions = [
            module.aligned_ensemble_probabilities(
                [model], x, probability_floor=local["model"]["probability_floor"]
            )
            for model in (fit_parallel, fit_parallel, fit_serial, fit_serial)
        ]
        for prediction in predictions[1:]:
            np.testing.assert_array_equal(prediction, predictions[0])
        fit_parallel.set_params(n_jobs=2)
        with self.assertRaisesRegex(ValueError, "parallel tree-probability"):
            module.aligned_ensemble_probabilities(
                [fit_parallel],
                x,
                probability_floor=local["model"]["probability_floor"],
            )

    def test_parallel_variant_seed_fit_is_bit_exact_to_sequential(self):
        module = self.module
        normalized = self.normalized

        local = copy.deepcopy(normalized)
        local["model"] = copy.deepcopy(normalized["model"])
        local["model"]["seeds"] = [11, 29]
        local["model"]["parameters"] = copy.deepcopy(
            normalized["model"]["parameters"]
        )
        local["model"]["parameters"]["n_estimators"] = 8
        local["model"]["parameters"]["n_jobs"] = 1
        local["model"]["fit_execution"] = copy.deepcopy(
            normalized["model"]["fit_execution"]
        )
        local["model"]["fit_execution"]["worker_count"] = (
            len(module.VARIANTS) * len(local["model"]["seeds"])
        )
        rows = [
            synthetic_row(
                module,
                f"row-{task_index}",
                f"task-{task_index}",
                f"repo-{task_index % 3}",
                module.CLASSES[task_index % 3],
                task_index % 3,
            )
            for task_index in range(18)
        ]
        weights, shared_diagnostics = module.training_weights(rows)
        sequential_models = {}
        sequential_diagnostics = {}
        for variant in module.VARIANTS:
            sequential_models[variant], sequential_diagnostics[variant] = (
                module.fit_ensemble(
                    module.matrix_for(rows, variant),
                    rows,
                    protocol=local,
                    weights=weights,
                )
            )

        parallel_models, parallel_diagnostics = module._fit_all_variants(
            rows, protocol=local
        )

        assert parallel_diagnostics["shared"] == shared_diagnostics
        assert parallel_diagnostics["variants"] == sequential_diagnostics
        assert parallel_diagnostics["fit_execution"] == {
            "parallel_unit": "one_variant_seed_estimator",
            "backend": "sklearn_joblib_loky_processes",
            "worker_count": 8,
            "estimator_fit_n_jobs": 1,
            "persisted_estimator_n_jobs": 1,
            "submission_order": "variant_then_seed",
            "result_collection_order": "variant_then_seed",
            "deterministic_ordered_collection": True,
            "estimator_n_jobs": 1,
        }
        for variant in module.VARIANTS:
            x = module.matrix_for(rows, variant)
            sequential_probabilities = module.aligned_ensemble_probabilities(
                sequential_models[variant],
                x,
                probability_floor=local["model"]["probability_floor"],
            )
            parallel_probabilities = module.aligned_ensemble_probabilities(
                parallel_models[variant],
                x,
                probability_floor=local["model"]["probability_floor"],
            )
            np.testing.assert_array_equal(
                parallel_probabilities, sequential_probabilities
            )

    def test_gates_fail_closed_without_model_refit_intervals(self):
        module = self.module
        normalized = self.normalized

        result = module.evaluate_gates(
            protocol=normalized,
            support=passing_support(),
            nested=synthetic_nested_for_gates(module),
            bootstrap=module.missing_model_refit_bootstrap("test omission"),
        )
        assert result["passed"] is False
        assert result["model_refit_bootstrap_reliable"] is False
        assert result["reliability_blockers"]
        assert "balanced_accuracy_lower" in result["failed_gate_ids"]
        assert "matched_accuracy" in result["failed_gate_ids"]

    def test_all_recovered_absolute_and_paired_gates_can_pass(self):
        module = self.module
        normalized = self.normalized
        monkeypatch = self.monkeypatch

        monkeypatch.setattr(
            module, "validate_nested_evidence", lambda *args, **kwargs: {}
        )
        monkeypatch.setattr(
            module, "validate_bootstrap_evidence", lambda *args, **kwargs: {}
        )

        result = module.evaluate_gates(
            protocol=normalized,
            support=passing_support(),
            nested=synthetic_nested_for_gates(module),
            bootstrap=passing_bootstrap(module),
            rows=[],
        )
        assert result["passed"] is True
        assert result["operational_reliability_claim"] is True
        assert result["failed_gate_ids"] == []
        assert len(result["absolute"]) == 16
        assert {
            "inspect_recall_lower",
            "edit_recall_lower",
            "check_or_finish_recall_lower",
            "accepted_coverage_lower",
            "nll_upper",
        }.issubset({row["id"] for row in result["absolute"]})
        assert len(result["paired"]) == 8
        assert {
            (row["candidate"], row["reference"]) for row in result["paired"]
        } == {
            ("history_j", "history_logit"),
            ("history_logit_j", "history_logit"),
        }

    def test_feature_completion_support_is_count_based_and_gated(self):
        module = self.module
        normalized = self.normalized

        rows = [synthetic_row(module, "row", "task", "repo", "inspect", 0)]
        support = module.support_summary(
            rows,
            {
                "numerically_stable_prompt_count": 2,
                "stable_feature_complete_prediction_count": 1,
                "stable_feature_complete_prediction_fraction": 0.5,
                "numerical_stability_fraction": 0.8,
            },
        )
        assert support["stable_feature_complete_prediction_fraction"] == 0.5
        assert support["stable_feature_complete_prediction_fraction_numerator"] == 1
        assert support["stable_feature_complete_prediction_fraction_denominator"] == 2

        failing = passing_support()
        failing["stable_feature_complete_prediction_fraction"] = 0.899
        result = module.evaluate_gates(
            protocol=normalized,
            support=failing,
            nested=synthetic_nested_for_gates(module),
            bootstrap=passing_bootstrap(module),
        )
        assert result["passed"] is False
        assert (
            "minimum_stable_feature_complete_prediction_fraction"
            in result["failed_gate_ids"]
        )

    def test_each_bound_gate_requires_structurally_valid_evidence(self):
        module = self.module
        normalized = self.normalized
        monkeypatch = self.monkeypatch

        bootstrap = passing_bootstrap(module)
        bootstrap["intervals"]["history_j"]["balanced_accuracy"]["lower"] = -1.0
        bootstrap["paired_differences"]["history_j_minus_history_logit"][
            "accuracy"
        ]["lower"] = -1.0
        monkeypatch.setattr(
            module, "validate_nested_evidence", lambda *args, **kwargs: {}
        )

        def reject_invalid_bootstrap(*args, **kwargs):
            raise ValueError("synthetic interval evidence is structurally invalid")

        monkeypatch.setattr(
            module, "validate_bootstrap_evidence", reject_invalid_bootstrap
        )
        result = module.evaluate_gates(
            protocol=normalized,
            support=passing_support(),
            nested=synthetic_nested_for_gates(module),
            bootstrap=bootstrap,
            rows=[],
        )
        assert result["passed"] is False
        assert "balanced_accuracy_lower" in result["failed_gate_ids"]
        assert "matched_accuracy" in result["failed_gate_ids"]

    def test_hierarchical_bootstrap_invokes_full_nested_refit_per_draw(self):
        module = self.module
        normalized = self.normalized

        rows = []
        for repo_index in range(6):
            for class_index, label in enumerate(module.CLASSES):
                rows.append(
                    synthetic_row(
                        module,
                        f"r{repo_index}-{label}",
                        f"t{repo_index}-{label}",
                        f"repo{repo_index}",
                        label,
                        class_index,
                    )
                )
        calls = []

        def fake_nested(sampled, *, protocol, base_weights):
            calls.append((sampled, np.asarray(base_weights).copy()))
            return synthetic_bootstrap_nested_result(
                module, sampled, base_weights
            )

        result = module.model_refit_hierarchical_bootstrap(
            rows,
            protocol=normalized,
            samples=3,
            nested_runner=fake_nested,
            validate_nested_draws=False,
        )
        assert len(calls) == 3
        assert result["status"] == "complete_hierarchical_bayesian_bootstrap"
        assert result["models_refit_inside_bootstrap"] is True
        assert (
            result[
                "models_refit_inside_bootstrap_is_in_process_execution_"
                "declaration_only"
            ]
            is True
        )
        assert result["persisted_evidence_scope"] == (
            module._expected_bootstrap_evidence_scope()
        )
        assert result["calibration_and_threshold_reselected_inside_each_draw"] is True
        assert result["same_draws_folds_weights_and_seed_order_across_variants"] is True
        assert result["samples"] == 3
        assert len(result["draw_records"]) == 3
        assert [record["draw_index"] for record in result["draw_records"]] == [0, 1, 2]
        assert all(
            [row["row_id"] for row in sampled] == [row["row_id"] for row in rows]
            for sampled, _ in calls
        )
        assert all(np.all(weights > 0.0) for _, weights in calls)
        assert module.validate_bootstrap_evidence(
            result,
            rows=rows,
            protocol=normalized,
            require_production_samples=False,
        )["credible_intervals_recomputed_from_draw_records"] is True

    def test_bootstrap_binary_evidence_rejects_noncanonical_zlib_level(self):
        module = self.module
        payload = b"row-probability-evidence-" * 64
        canonical = module._zlib_base64_encode(payload)
        level_one = module.base64.b64encode(
            module.zlib.compress(payload, level=1)
        ).decode("ascii")
        assert level_one != canonical
        with self.assertRaisesRegex(ValueError, "canonical frozen level-9"):
            module._zlib_base64_decode_exact(
                level_one, len(payload), "synthetic probability evidence"
            )

    def test_bootstrap_checkpoint_resume_is_identity_bound_and_deterministic(self):
        module = self.module
        normalized = self.normalized
        tmp_path = self.tmp_path

        rows = []
        for repo_index in range(6):
            for class_index, label in enumerate(module.CLASSES):
                rows.append(
                    synthetic_row(
                        module,
                        f"r{repo_index}-{label}",
                        f"t{repo_index}-{label}",
                        f"repo{repo_index}",
                        label,
                        class_index,
                    )
                )
        calls = 0

        def fake_nested(sampled, *, protocol, base_weights):
            nonlocal calls
            calls += 1
            return synthetic_bootstrap_nested_result(
                module, sampled, base_weights
            )

        checkpoint = tmp_path / "bootstrap-checkpoint.json"
        first = module.model_refit_hierarchical_bootstrap(
            rows,
            protocol=normalized,
            samples=2,
            nested_runner=fake_nested,
            checkpoint_path=checkpoint,
            validate_nested_draws=False,
        )
        assert calls == 2
        checkpoint_value = json.loads(checkpoint.read_text())
        assert checkpoint_value["status"] == "complete"
        assert checkpoint_value["next_draw_index"] == 2
        with self.assertRaisesRegex(ValueError, "already exists; resume explicitly"):
            module.model_refit_hierarchical_bootstrap(
                rows,
                protocol=normalized,
                samples=2,
                nested_runner=fake_nested,
                checkpoint_path=checkpoint,
                validate_nested_draws=False,
            )
        second = module.model_refit_hierarchical_bootstrap(
            rows,
            protocol=normalized,
            samples=2,
            nested_runner=fake_nested,
            checkpoint_path=checkpoint,
            resume=True,
            validate_nested_draws=False,
        )
        assert calls == 2
        assert first["intervals"] == second["intervals"]
        assert first["paired_differences"] == second["paired_differences"]
        assert second["execution"]["resumed_from_draw"] == 2
        assert second["samples"] == 2

        status_tamper = copy.deepcopy(checkpoint_value)
        status_tamper["status"] = "in_progress"
        checkpoint.write_text(json.dumps(status_tamper))
        with self.assertRaisesRegex(ValueError, "status or accumulator contract changed"):
            module.model_refit_hierarchical_bootstrap(
                rows,
                protocol=normalized,
                samples=2,
                nested_runner=fake_nested,
                checkpoint_path=checkpoint,
                resume=True,
                validate_nested_draws=False,
            )

        count_tamper = copy.deepcopy(checkpoint_value)
        count_tamper["next_draw_index"] = 1
        checkpoint.write_text(json.dumps(count_tamper))
        with self.assertRaisesRegex(ValueError, "draw record count changed"):
            module.model_refit_hierarchical_bootstrap(
                rows,
                protocol=normalized,
                samples=2,
                nested_runner=fake_nested,
                checkpoint_path=checkpoint,
                resume=True,
                validate_nested_draws=False,
            )

        metric_tamper = copy.deepcopy(checkpoint_value)
        del metric_tamper["draw_records"][0]["variant_metrics"]["history_j"][
            "accuracy"
        ]
        checkpoint.write_text(json.dumps(metric_tamper))
        with self.assertRaisesRegex(ValueError, "draw metric names changed"):
            module.model_refit_hierarchical_bootstrap(
                rows,
                protocol=normalized,
                samples=2,
                nested_runner=fake_nested,
                checkpoint_path=checkpoint,
                resume=True,
                validate_nested_draws=False,
            )

        for field, mutate_identity in (
            (
                "analyzer_sha256",
                lambda identity: identity.__setitem__("analyzer_sha256", "0" * 64),
            ),
            (
                "requirements_sha256",
                lambda identity: identity.__setitem__(
                    "requirements_sha256", "0" * 64
                ),
            ),
            (
                "runtime_versions",
                lambda identity: identity["runtime_versions"].__setitem__(
                    "numpy", "0.0.0"
                ),
            ),
        ):
            with self.subTest(checkpoint_identity_field=field):
                identity_tamper = copy.deepcopy(checkpoint_value)
                mutate_identity(identity_tamper["identity"])
                checkpoint.write_text(json.dumps(identity_tamper))
                with self.assertRaisesRegex(ValueError, "checkpoint identity differs"):
                    module.model_refit_hierarchical_bootstrap(
                        rows,
                        protocol=normalized,
                        samples=2,
                        nested_runner=fake_nested,
                        checkpoint_path=checkpoint,
                        resume=True,
                        validate_nested_draws=False,
                    )

        checkpoint.write_text(json.dumps(checkpoint_value))
        changed = copy.deepcopy(normalized)
        changed["value"] = copy.deepcopy(normalized["value"])
        changed["value"]["status"] = "changed"
        with self.assertRaisesRegex(ValueError, "checkpoint identity differs"):
            module.model_refit_hierarchical_bootstrap(
                rows,
                protocol=changed,
                samples=2,
                nested_runner=fake_nested,
                checkpoint_path=checkpoint,
                resume=True,
                validate_nested_draws=False,
            )

    def test_bootstrap_evidence_rejects_every_structural_tamper(self):
        module = self.module
        normalized = self.normalized

        rows = [
            synthetic_row(
                module,
                f"r{repo_index}-{label}",
                f"t{repo_index}-{label}",
                f"repo{repo_index}",
                label,
                class_index,
            )
            for repo_index in range(6)
            for class_index, label in enumerate(module.CLASSES)
        ]

        def fake_nested(sampled, *, protocol, base_weights):
            return synthetic_bootstrap_nested_result(
                module, sampled, base_weights
            )

        valid = module.model_refit_hierarchical_bootstrap(
            rows,
            protocol=normalized,
            samples=3,
            nested_runner=fake_nested,
            validate_nested_draws=False,
        )

        identity_tamper = copy.deepcopy(valid)
        identity_tamper["identity"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "frozen evidence fields changed"):
            module.validate_bootstrap_evidence(
                identity_tamper,
                rows=rows,
                protocol=normalized,
                require_production_samples=False,
            )

        scope_tamper = copy.deepcopy(valid)
        scope_tamper["persisted_evidence_scope"][
            "independent_refit_attestation"
        ] = True
        with self.assertRaisesRegex(ValueError, "frozen evidence fields changed"):
            module.validate_bootstrap_evidence(
                scope_tamper,
                rows=rows,
                protocol=normalized,
                require_production_samples=False,
            )

        order_tamper = copy.deepcopy(valid)
        order_tamper["draw_records"][0], order_tamper["draw_records"][1] = (
            order_tamper["draw_records"][1],
            order_tamper["draw_records"][0],
        )
        with self.assertRaisesRegex(ValueError, "indices are not contiguous"):
            module.validate_bootstrap_evidence(
                order_tamper,
                rows=rows,
                protocol=normalized,
                require_production_samples=False,
            )

        count_tamper = copy.deepcopy(valid)
        count_tamper["draw_records"].pop()
        with self.assertRaisesRegex(ValueError, "draw record count changed"):
            module.validate_bootstrap_evidence(
                count_tamper,
                rows=rows,
                protocol=normalized,
                require_production_samples=False,
            )

        metric_tamper = copy.deepcopy(valid)
        del metric_tamper["draw_records"][0]["variant_metrics"]["history_j"][
            "accuracy"
        ]
        with self.assertRaisesRegex(ValueError, "draw metric names changed"):
            module.validate_bootstrap_evidence(
                metric_tamper,
                rows=rows,
                protocol=normalized,
                require_production_samples=False,
            )

        pair_tamper = copy.deepcopy(valid)
        pair_tamper["draw_records"][0]["paired_differences"][
            "history_j_minus_history_logit"
        ]["accuracy"] += 0.01
        with self.assertRaisesRegex(ValueError, "arithmetic changed"):
            module.validate_bootstrap_evidence(
                pair_tamper,
                rows=rows,
                protocol=normalized,
                require_production_samples=False,
            )

        probability_hash_tamper = copy.deepcopy(valid)
        probability_hash_tamper["draw_records"][0]["nested_probability_hashes"][
            "history_j"
        ] = "not-a-sha256"
        with self.assertRaisesRegex(ValueError, "probability hash does not reproduce"):
            module.validate_bootstrap_evidence(
                probability_hash_tamper,
                rows=rows,
                protocol=normalized,
                require_production_samples=False,
            )

        recomputed_metric_tamper = copy.deepcopy(valid)
        recomputed_metric_tamper["draw_records"][0]["variant_metrics"][
            "history_only"
        ]["accuracy"] = 0.5
        record = recomputed_metric_tamper["draw_records"][0]
        unhashed = dict(record)
        unhashed.pop("draw_record_sha256")
        record["draw_record_sha256"] = module.canonical_json_sha256(unhashed)
        with self.assertRaisesRegex(ValueError, "does not reproduce from row prediction"):
            module.validate_bootstrap_evidence(
                recomputed_metric_tamper,
                rows=rows,
                protocol=normalized,
                require_production_samples=False,
            )

        recomputed_probability_tamper = copy.deepcopy(valid)
        probability_record = recomputed_probability_tamper["draw_records"][0]
        evidence_variants = probability_record["row_prediction_evidence"]["variants"]
        evidence_variants["history_j"]["probability_zlib_base64"] = (
            evidence_variants["history_only"]["probability_zlib_base64"]
        )
        unhashed = dict(probability_record)
        unhashed.pop("draw_record_sha256")
        probability_record["draw_record_sha256"] = module.canonical_json_sha256(
            unhashed
        )
        with self.assertRaisesRegex(
            ValueError,
            "does not reproduce from row prediction evidence|probability hash does not reproduce",
        ):
            module.validate_bootstrap_evidence(
                recomputed_probability_tamper,
                rows=rows,
                protocol=normalized,
                require_production_samples=False,
            )

        recomputed_acceptance_tamper = copy.deepcopy(valid)
        acceptance_record = recomputed_acceptance_tamper["draw_records"][0]
        probabilities, acceptance = (
            module._decode_bootstrap_row_prediction_evidence(
                acceptance_record["row_prediction_evidence"],
                rows=rows,
                draw_index=0,
            )
        )
        acceptance["history_j"][0] = False
        acceptance_record["row_prediction_evidence"] = (
            module._encode_bootstrap_row_prediction_evidence(
                rows, probabilities, acceptance
            )
        )
        unhashed = dict(acceptance_record)
        unhashed.pop("draw_record_sha256")
        acceptance_record["draw_record_sha256"] = module.canonical_json_sha256(
            unhashed
        )
        with self.assertRaisesRegex(ValueError, "does not reproduce from row prediction"):
            module.validate_bootstrap_evidence(
                recomputed_acceptance_tamper,
                rows=rows,
                protocol=normalized,
                require_production_samples=False,
            )

        record_hash_tamper = copy.deepcopy(valid)
        record_hash_tamper["draw_records"][0]["draw_record_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "record hash changed"):
            module.validate_bootstrap_evidence(
                record_hash_tamper,
                rows=rows,
                protocol=normalized,
                require_production_samples=False,
            )

        array_hash_tamper = copy.deepcopy(valid)
        array_hash_tamper["draw_records_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "array hash changed"):
            module.validate_bootstrap_evidence(
                array_hash_tamper,
                rows=rows,
                protocol=normalized,
                require_production_samples=False,
            )

        interval_tamper = copy.deepcopy(valid)
        interval_tamper["intervals"]["history_j"]["accuracy"]["lower"] += 0.01
        with self.assertRaisesRegex(ValueError, "credible intervals do not reproduce"):
            module.validate_bootstrap_evidence(
                interval_tamper,
                rows=rows,
                protocol=normalized,
                require_production_samples=False,
            )

    def test_bootstrap_rejects_self_consistent_fabricated_records_without_row_evidence(self):
        module = self.module
        normalized = self.normalized
        rows = [
            synthetic_row(
                module,
                f"r{repo_index}-{label}",
                f"t{repo_index}-{label}",
                f"repo{repo_index}",
                label,
                class_index,
            )
            for repo_index in range(6)
            for class_index, label in enumerate(module.CLASSES)
        ]

        def fake_nested(sampled, *, protocol, base_weights):
            return synthetic_bootstrap_nested_result(
                module, sampled, base_weights
            )

        fabricated = module.model_refit_hierarchical_bootstrap(
            rows,
            protocol=normalized,
            samples=3,
            nested_runner=fake_nested,
            validate_nested_draws=False,
        )
        for draw_record in fabricated["draw_records"]:
            draw_record.pop("row_prediction_evidence")
            draw_record["nested_probability_hashes"] = {
                variant: "0" * 64 for variant in module.VARIANTS
            }
            draw_record["variant_metrics"] = {
                variant: {
                    metric: (
                        0.1
                        if metric
                        in {
                            "multiclass_negative_log_likelihood",
                            "multiclass_brier",
                        }
                        else 0.02
                        if metric == "top_label_ece"
                        else 0.92
                    )
                    for metric in module.BOOTSTRAP_METRICS
                }
                for variant in module.VARIANTS
            }
            draw_record["paired_differences"] = {
                f"{candidate}_minus_{reference}": {
                    metric: (
                        draw_record["variant_metrics"][candidate][metric]
                        - draw_record["variant_metrics"][reference][metric]
                    )
                    for metric in module.BOOTSTRAP_METRICS
                }
                for candidate, reference in module.PAIRED_COMPARISONS
            }
            unhashed = dict(draw_record)
            unhashed.pop("draw_record_sha256")
            draw_record["draw_record_sha256"] = module.canonical_json_sha256(
                unhashed
            )
        fabricated["draw_records_sha256"] = module.canonical_json_sha256(
            fabricated["draw_records"]
        )
        fabricated["intervals"], fabricated["paired_differences"] = (
            module._bootstrap_intervals_from_records(
                fabricated["draw_records"], fabricated["confidence_level"]
            )
        )

        with self.assertRaisesRegex(ValueError, "row prediction evidence"):
            module.validate_bootstrap_evidence(
                fabricated,
                rows=rows,
                protocol=normalized,
                require_production_samples=False,
            )

    def test_development_bundle_uses_exact_n60_declaration_and_payload_checker(self):
        module = self.module
        monkeypatch = self.monkeypatch
        tmp_path = self.tmp_path

        cohort_path = tmp_path / "cohort.json"
        prompts = tmp_path / "prompts.json"
        summary = tmp_path / "summary.json"
        cohort_path.write_text("{}\n")
        prompts.write_text("[]\n")
        summary.write_text("{}\n")
        observed = {}
        declaration = object()

        class FakeChecker:
            @staticmethod
            def validate_declaration(supplied_path):
                observed["cohort_path"] = supplied_path
                return declaration

            @staticmethod
            def validate_materialized_bundle(
                supplied_declaration, *, prompts_path, summary_path
            ):
                observed["declaration"] = supplied_declaration
                return {
                    "task_count": 60,
                    "prompt_count": 0,
                    "cohort_count": 2,
                    "cohort_manifest_sha256": module.sha256_file(cohort_path),
                    "prompt_bundle_sha256": module.sha256_file(prompts_path),
                    "summary_sha256": module.sha256_file(summary_path),
                }

        monkeypatch.setattr(module, "DEFAULT_DEVELOPMENT_COHORT", cohort_path)
        monkeypatch.setattr(module, "COHORT_CHECKER", FakeChecker)
        result = module.validate_development_bundle(
            cohort_path=cohort_path,
            prompts_path=prompts,
            summary_path=summary,
        )
        assert result["task_count"] == 60
        assert result["exact_n60_declaration_validated"] is True
        assert result["every_prompt_payload_and_provenance_binding_validated"] is True
        assert observed["cohort_path"] == cohort_path
        assert observed["declaration"] is declaration

    def test_replay_merge_receipt_is_delegated_and_hash_bound(self):
        module = self.module
        monkeypatch = self.monkeypatch
        tmp_path = self.tmp_path

        receipt = tmp_path / "merge-manifest.json"
        report = tmp_path / "public-report.json"
        prompts = tmp_path / "prompts.json"
        summary = tmp_path / "prompts-summary.json"
        for path, payload in (
            (receipt, "{\"receipt\": true}\n"),
            (report, "{\"report\": true}\n"),
            (prompts, "[]\n"),
            (summary, "{}\n"),
        ):
            path.write_text(payload)
        observed = {}

        class FakeReplayPipeline:
            report_sha256 = module.sha256_file(report)

            @classmethod
            def validate_merge_receipt(
                cls,
                *,
                report_path,
                merge_manifest_path,
                replay_root,
                prompts_path,
                summary_path,
            ):
                observed.update(
                    {
                        "report_path": report_path,
                        "merge_manifest_path": merge_manifest_path,
                        "replay_root": replay_root,
                        "prompts_path": prompts_path,
                        "summary_path": summary_path,
                    }
                )
                return module.argparse.Namespace(
                    report_path=report_path,
                    report_sha256=cls.report_sha256,
                    merge_manifest_path=merge_manifest_path,
                    merge_manifest_sha256=module.sha256_file(
                        merge_manifest_path
                    ),
                    experiment_count=2,
                    prompt_bundle_sha256=module.sha256_file(prompts_path),
                    materialization_receipt_sha256="a" * 64,
                    source_freeze_git_commit="b" * 40,
                    data_freeze_git_commit="c" * 40,
                )

        monkeypatch.setattr(module, "REPLAY_PIPELINE", FakeReplayPipeline)
        result = module.validate_replay_merge_receipt(
            receipt_path=receipt,
            public_report_path=report,
            prompts_path=prompts,
            summary_path=summary,
        )
        assert result["report_sha256"] == module.sha256_file(report)
        assert result["experiment_count"] == 2
        assert result["full_chunk_sources_and_lossless_merge_reauthenticated"] is True
        assert observed == {
            "report_path": report,
            "merge_manifest_path": receipt,
            "replay_root": receipt.parent,
            "prompts_path": prompts,
            "summary_path": summary,
        }

        FakeReplayPipeline.report_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "inconsistent authenticated receipt"):
            module.validate_replay_merge_receipt(
                receipt_path=receipt,
                public_report_path=report,
                prompts_path=prompts,
                summary_path=summary,
            )

    def test_fit_refuses_nonpassing_or_non_refit_analysis(self):
        module = self.module
        normalized = self.normalized
        monkeypatch = self.monkeypatch

        bootstrap = module.missing_model_refit_bootstrap("test omission")
        nested = synthetic_nested_for_gates(module)
        support = passing_support()
        gates = module.evaluate_gates(
            protocol=normalized,
            support=support,
            nested=nested,
            bootstrap=bootstrap,
        )
        analysis = {
            "schema_version": module.SCHEMA_VERSION,
            "id": "swe-task-state-interpreter-v3-development-analysis",
            "inputs": {"x": "y"},
            "development_data_binding": {},
            "replay_merge_binding": {},
            "eligibility": {"source": "test"},
            "support": support,
            "nested_development_evaluation": nested,
            "bootstrap": bootstrap,
            "development_gates": gates,
            "operational_reliability_claim": False,
        }
        mismatched_support = copy.deepcopy(support)
        mismatched_support["stable_feature_complete_prediction_fraction"] = 0.5
        monkeypatch.setattr(
            module,
            "support_summary",
            lambda rows, eligibility: mismatched_support,
        )
        prepared = {
            "hashes": {"x": "y"},
            "development_binding": {},
            "replay_merge_binding": {},
            "eligibility": {"source": "test"},
            "rows": [],
            "protocol": normalized,
        }
        with self.assertRaisesRegex(ValueError, "support does not reproduce"):
            module._validate_analysis_for_fit(analysis, prepared=prepared)

        monkeypatch.setattr(module, "support_summary", lambda rows, eligibility: support)
        monkeypatch.setattr(
            module, "validate_nested_evidence", lambda *args, **kwargs: {}
        )
        monkeypatch.setattr(
            module, "validate_bootstrap_evidence", lambda *args, **kwargs: {}
        )
        monkeypatch.setattr(
            module,
            "nested_leave_one_repository_out",
            lambda *args, **kwargs: nested,
        )
        monkeypatch.setattr(
            module, "evaluate_gates", lambda *args, **kwargs: gates
        )
        with self.assertRaisesRegex(ValueError, "fit is forbidden"):
            module._validate_analysis_for_fit(analysis, prepared=prepared)

    def test_cli_path_contract_rejects_aliases_and_clobbers(self):
        module = self.module
        monkeypatch = self.monkeypatch
        tmp_path = self.tmp_path

        monkeypatch.setattr(module, "V3_INTERPRETER_OUTPUT_ROOT", tmp_path)
        inputs = {
            name: tmp_path / f"{name}.json"
            for name in (
                "prompts",
                "public_report",
                "replay_merge_receipt",
                "protocol",
                "action_protocol",
                "development_cohort",
                "prompts_summary",
            )
        }
        for path in inputs.values():
            path.write_text("{}\n")
        args = module.argparse.Namespace(
            command="analyze",
            output=tmp_path / "analysis.json",
            bootstrap_checkpoint=None,
            resume_bootstrap=False,
            **inputs,
        )
        validated = module.validate_cli_path_contract(args)
        assert validated["canonical_paths_distinct"] is True
        assert validated["new_outputs_no_clobber"] is True

        args.output = args.prompts
        with self.assertRaisesRegex(ValueError, "paths must be distinct"):
            module.validate_cli_path_contract(args)

        args.output = tmp_path / "existing-analysis.json"
        args.output.write_text("do not overwrite\n")
        with self.assertRaisesRegex(ValueError, "output already exists"):
            module.validate_cli_path_contract(args)

        args.output = tmp_path / "new-analysis.json"
        args.bootstrap_checkpoint = args.output
        with self.assertRaisesRegex(ValueError, "paths must be distinct"):
            module.validate_cli_path_contract(args)

        hardlinked_checkpoint = tmp_path / "hardlinked-checkpoint.json"
        os.link(args.prompts, hardlinked_checkpoint)
        args.bootstrap_checkpoint = hardlinked_checkpoint
        args.resume_bootstrap = True
        with self.assertRaisesRegex(ValueError, "hard-link alias"):
            module.validate_cli_path_contract(args)

    def test_cli_exposes_development_analyze_and_fit(self):
        module = self.module

        analyze = module.parse_args(
            [
                "analyze",
                "--prompts",
                "p.json",
                "--public-report",
                "r.json",
                "--prompts-summary",
                "s.json",
                "--output",
                "a.json",
                "--diagnostic-without-model-refit-bootstrap",
            ]
        )
        assert analyze.command == "analyze"
        assert analyze.diagnostic_without_model_refit_bootstrap is True
        fit = module.parse_args(
            [
                "fit",
                "--prompts",
                "p.json",
                "--public-report",
                "r.json",
                "--prompts-summary",
                "s.json",
                "--analysis",
                "a.json",
                "--bundle",
                "m.joblib",
                "--manifest",
                "m.json",
                "--output",
                "f.json",
            ]
        )
        assert fit.command == "fit"


if __name__ == "__main__":
    unittest.main()
