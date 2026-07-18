#!/usr/bin/env python3
"""Focused tests for exact-rank SWE checkpoint comparisons."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FIXTURES = load_module(
    "multitask_analyzer_fixtures_for_checkpoint_test",
    ROOT / "tests" / "test_analyze_swe_multitask_initial_probes.py",
)
ANALYZER = FIXTURES.MODULE
MODULE = load_module(
    "compare_swe_multitask_checkpoints",
    ROOT / "scripts" / "compare_swe_multitask_checkpoints.py",
)


def checkpoint_analyses(
    *, c0_checkpoint: str = "C0", retain_subset: bool = True
) -> tuple[dict[str, object], dict[str, object]]:
    prompts, public, native = FIXTURES.fixture()
    if c0_checkpoint == "C0M":
        for index, (prompt, public_experiment, native_experiment) in enumerate(
            zip(prompts, public["experiments"], native["experiments"], strict=True)
        ):
            prompt["metadata"]["checkpoint"] = copy.deepcopy(
                ANALYZER.CHECKPOINT_CONTRACTS["C0M"]
            )
            prompt["metadata"]["capture_match"] = {
                "capture_manifest_sha256": "d" * 64,
                "first_request_sha256": f"{index + 1:064x}",
                "second_request_sha256": f"{index + 101:064x}",
            }
            public_experiment["metadata"] = copy.deepcopy(prompt["metadata"])
            native_experiment["metadata"] = copy.deepcopy(prompt["metadata"])
    c0 = ANALYZER.analyze(
        prompts,
        public,
        native,
        bootstrap_seed=7,
        bootstrap_samples=10,
        expected_checkpoint=ANALYZER.CHECKPOINT_CONTRACTS[c0_checkpoint],
    )
    for index, (prompt, public_experiment, native_experiment) in enumerate(
        zip(
        prompts,
        public["experiments"],
        native["experiments"],
        strict=True,
        )
    ):
        prompt["metadata"]["checkpoint"] = copy.deepcopy(
            ANALYZER.CHECKPOINT_CONTRACTS["C1"]
        )
        prompt["metadata"]["observation_audit"] = {
            "capture_manifest_sha256": "d" * 64,
            "first_request_sha256": f"{index + 1:064x}",
            "second_request_sha256": f"{index + 101:064x}",
        }
        public_experiment["metadata"] = copy.deepcopy(prompt["metadata"])
        native_experiment["metadata"] = copy.deepcopy(prompt["metadata"])
    c1 = ANALYZER.analyze(
        prompts,
        public,
        native,
        bootstrap_seed=7,
        bootstrap_samples=10,
        expected_checkpoint=ANALYZER.CHECKPOINT_CONTRACTS["C1"],
    )
    retain_and_improve_c1(c1, retain_subset=retain_subset)
    return c0, c1


def recompute_method(method: dict[str, object]) -> None:
    for task in method["tasks"]:
        family_utilities = []
        for family in task["families"]:
            scores = [
                concept["target_score"]
                for concept in family["concepts"]
                if concept["target_score"]["scorable"]
            ]
            family["concept_count"] = len(family["concepts"])
            family["scorable_target_concept_count"] = len(scores)
            if scores:
                family["target_utility_u"] = statistics.fmean(
                    score["utility_u"] for score in scores
                )
                family_utilities.append(family["target_utility_u"])
        task["family_count"] = len(family_utilities)
        task["target_utility_u"] = statistics.fmean(family_utilities)
    method["task_count"] = len(method["tasks"])
    method["target_utility_u"] = statistics.fmean(
        task["target_utility_u"] for task in method["tasks"]
    )


def retain_and_improve_c1(
    c1: dict[str, object], *, retain_subset: bool = True
) -> None:
    for method in c1["methods"].values():
        if retain_subset:
            method["tasks"] = [
                task
                for task in method["tasks"]
                if task["instance_id"] != "project__project-1"
            ]
        first_task = method["tasks"][0]
        first_family = first_task["families"][0]
        first_family["concepts"] = [
            concept
            for concept in first_family["concepts"]
            if concept["id"] != "concept-1"
        ]
        for task in method["tasks"]:
            for family in task["families"]:
                for concept in family["concepts"]:
                    score = concept["target_score"]
                    if score["scorable"]:
                        score["minimum_rank"] = max(1, score["minimum_rank"] // 2)
                        score["utility_u"] = ANALYZER.normalized_utility(
                            score["minimum_rank"]
                        )
        recompute_method(method)
    reference_tasks = c1["methods"]["public_jacobian"]["tasks"]
    c1["coverage"]["task_count"] = len(reference_tasks)
    c1["coverage"]["repository_count"] = len(
        {task["repo"] for task in reference_tasks}
    )
    c1["coverage"]["concept_count"] = sum(
        len(family["concepts"])
        for task in reference_tasks
        for family in task["families"]
    )


class CompareSweMultitaskCheckpointsTest(unittest.TestCase):
    def test_retained_identity_task_macro_exact_rank_metrics(self) -> None:
        c0, c1 = checkpoint_analyses()
        result = MODULE.compare(
            c0, c1, bootstrap_seed=13, bootstrap_samples=200
        )
        repeated = MODULE.compare(
            c0, c1, bootstrap_seed=13, bootstrap_samples=200
        )
        self.assertEqual(result, repeated)
        self.assertIn("not causal", result["label"])
        self.assertEqual(result["statistics"]["bootstrap_samples"], 200)

        public = result["methods"]["public_jacobian"]
        self.assertEqual(public["matching"]["c0_available_task_count"], 4)
        self.assertEqual(public["matching"]["c1_retained_task_count"], 3)
        self.assertEqual(public["matching"]["retained_c1_concept_count"], 6)
        self.assertEqual(public["matching"]["paired_scorable_concept_count"], 6)
        utility = public["normalized_utility_u"]
        self.assertGreater(utility["c1_minus_c0"], 0.0)
        self.assertEqual(
            utility["paired_task_bootstrap"]["positive_task_count"], 2
        )
        self.assertEqual(utility["paired_task_bootstrap"]["tie_task_count"], 1)
        log_rank = public["mean_log_rank"]
        self.assertLess(log_rank["c1_minus_c0"], 0.0)
        self.assertEqual(
            log_rank["paired_task_bootstrap"]["negative_task_count"], 2
        )
        self.assertLess(public["geometric_mean_rank"]["c1_to_c0_ratio"], 1.0)
        concept = public["per_concept_ranks"][0]
        self.assertEqual(concept["c0_minimum_rank"], 1)
        self.assertEqual(concept["c1_minimum_rank"], 1)
        self.assertIn("c1_minus_c0_log_rank", concept)
        self.assertNotIn("pass_at_k", result["contract"]["rank_source"])
        self.assertIn("public_minus_logit", result["stage_method_contrasts"])
        self.assertEqual(
            result["stage_method_contrasts"]["public_minus_logit"][
                "normalized_utility_u"
            ]["task_count"],
            3,
        )

    def test_capture_matched_trajectory_contract_and_stage_contrasts(self) -> None:
        c0m, c1 = checkpoint_analyses(
            c0_checkpoint="C0M", retain_subset=False
        )
        result = MODULE.compare(
            c0m,
            c1,
            bootstrap_seed=13,
            bootstrap_samples=100,
            c0_checkpoint_id="C0M",
        )
        self.assertTrue(result["contract"]["trajectory_request_bindings_match"])
        self.assertEqual(result["contract"]["c0_checkpoint"]["id"], "C0M")
        self.assertIn("C0M", result["label"])

        c1["source_bindings"]["trajectory_bindings"][0][
            "first_request_sha256"
        ] = "f" * 64
        with self.assertRaisesRegex(ValueError, "trajectory request bindings"):
            MODULE.compare(
                c0m,
                c1,
                bootstrap_seed=13,
                bootstrap_samples=10,
                c0_checkpoint_id="C0M",
            )

    def test_rejects_protocol_model_layers_and_checkpoint_mismatches(self) -> None:
        mutators = (
            (
                lambda c0, c1: c1["source_bindings"].__setitem__(
                    "protocol_sha256", "f" * 64
                ),
                "protocol",
            ),
            (
                lambda c0, c1: c1["model"].__setitem__("revision", "changed"),
                "model",
            ),
            (
                lambda c0, c1: c1["evaluation"]["fixed_middle_layers"].pop(),
                "layers",
            ),
            (
                lambda c0, c1: c1["evaluation"]["checkpoint_metadata"].__setitem__(
                    "name", "changed"
                ),
                "checkpoint",
            ),
        )
        for mutate, message in mutators:
            with self.subTest(message=message):
                c0, c1 = checkpoint_analyses()
                mutate(c0, c1)
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.compare(c0, c1, bootstrap_seed=3, bootstrap_samples=10)

    def test_rejects_unmatched_retained_task_and_concept(self) -> None:
        c0, c1 = checkpoint_analyses()
        for method in c1["methods"].values():
            method["tasks"][0]["instance_id"] = "missing__task-1"
        with self.assertRaisesRegex(ValueError, "absent from C0"):
            MODULE.compare(c0, c1, bootstrap_seed=3, bootstrap_samples=10)

        c0, c1 = checkpoint_analyses()
        for method in c1["methods"].values():
            method["tasks"][0]["families"][0]["concepts"][0]["id"] = "missing"
        with self.assertRaisesRegex(ValueError, "absent from C0"):
            MODULE.compare(c0, c1, bootstrap_seed=3, bootstrap_samples=10)

    def test_rejects_missing_exact_rank_and_inconsistent_utility(self) -> None:
        c0, c1 = checkpoint_analyses()
        del c1["methods"]["public_jacobian"]["tasks"][0]["families"][0][
            "concepts"
        ][0]["target_score"]["minimum_rank"]
        with self.assertRaisesRegex(ValueError, "minimum_rank|rank"):
            MODULE.compare(c0, c1, bootstrap_seed=3, bootstrap_samples=10)

        c0, c1 = checkpoint_analyses()
        c1["methods"]["public_jacobian"]["tasks"][0]["families"][0][
            "concepts"
        ][0]["target_score"]["utility_u"] += 0.1
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MODULE.compare(c0, c1, bootstrap_seed=3, bootstrap_samples=10)

    def test_cli_writes_bound_output(self) -> None:
        c0, c1 = checkpoint_analyses()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0_path = root / "c0.json"
            c1_path = root / "c1.json"
            output = root / "comparison.json"
            c0_path.write_text(json.dumps(c0), encoding="utf-8")
            c1_path.write_text(json.dumps(c1), encoding="utf-8")
            status = MODULE.main(
                [
                    "--c0-analysis",
                    str(c0_path),
                    "--c1-analysis",
                    str(c1_path),
                    "--output",
                    str(output),
                    "--bootstrap-samples",
                    "20",
                ]
            )
            self.assertEqual(status, 0)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["statistics"]["bootstrap_samples"], 20)
            self.assertEqual(
                set(written["inputs"]), {"c0_analysis", "c1_analysis"}
            )
            self.assertEqual(len(written["inputs"]["c0_analysis"]["sha256"]), 64)
            self.assertEqual(
                written["inputs"]["c0_analysis"]["path"], "external/c0.json"
            )


if __name__ == "__main__":
    unittest.main()
