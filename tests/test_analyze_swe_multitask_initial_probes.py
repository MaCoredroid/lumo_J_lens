#!/usr/bin/env python3
"""Focused tests for the multi-task SWE task-start probe analyzer."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
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


MODULE = load_module(
    "analyze_swe_multitask_initial_probes",
    ROOT / "scripts" / "analyze_swe_multitask_initial_probes.py",
)

V = MODULE.LOGIT_VOCABULARY_SIZE
LAYERS = MODULE.FIXED_MIDDLE_LAYERS
PUBLIC_TARGET_RANKS = (
    (1, V, 1),
    (5, 5),
    (10, 10),
    (50, 50),
)
NATIVE_TARGET_RANKS = (
    (2, 2, 2),
    (10, 10),
    (20, 20),
    (100, 100),
)
LOGIT_TARGET_RANKS = (
    (10, 10, 10),
    (20, 20),
    (50, 50),
    (200, 200),
)


def forms(target: str, base: int) -> list[dict[str, object]]:
    return [
        {"text": target, "token_id": base},
        {"text": f" {target}", "token_id": base + 1},
    ]


def make_concept(task_index: int, concept_index: int, family: str) -> dict[str, object]:
    target = f"target_{task_index}_{concept_index}"
    foil_target = f"foil_{task_index}_{concept_index}"
    base = 1000 + task_index * 100 + concept_index * 10
    return {
        "id": f"concept-{concept_index}",
        "family": family,
        "target": target,
        "path": f"package/module_{task_index}_{concept_index}.py",
        "evidence": {"kind": "patch_path", "line": concept_index + 1},
        "visibility": "oracle_hidden",
        "forms": forms(target, base),
        "foils": [
            {
                "task_instance_id": f"external__task-{task_index}",
                "concept_id": f"foil-concept-{concept_index}",
                "family": family,
                "target": foil_target,
                "forms": forms(foil_target, base + 2),
            }
        ],
    }


def make_prompts() -> list[dict[str, object]]:
    result = []
    for task_index in range(4):
        families = (
            ("file_stem", "file_stem", "hunk_symbol")
            if task_index == 0
            else ("file_stem", "hunk_symbol")
        )
        concepts = [
            make_concept(task_index, concept_index, family)
            for concept_index, family in enumerate(families)
        ]
        score_ids = [
            form["token_id"]
            for concept in concepts
            for group in (concept["forms"], concept["foils"][0]["forms"])
            for form in group
        ]
        result.append(
            {
                "id": f"swe-initial-task-{task_index}",
                "text": f"rendered task-start prompt {task_index}",
                "token_ids": [50, 60 + task_index, 70],
                "score_token_ids": score_ids,
                "metadata": {
                    "kind": "swe_verified_multitask_initial_probe",
                    "protocol_sha256": "a" * 64,
                    "lens_outputs_used_for_selection": False,
                    "task": {
                        "instance_id": f"project__project-{task_index}",
                        "repo": "repo-a" if task_index < 2 else "repo-b",
                        "base_commit": f"commit-{task_index}",
                        "problem_statement_sha256": f"{task_index + 1:x}" * 64,
                        "patch_sha256": f"{task_index + 5:x}" * 64,
                        "test_patch_sha256": f"{task_index + 9:x}" * 64,
                    },
                    "checkpoint": {
                        "id": "C0",
                        "name": "task_start",
                        "visibility_boundary": "before_first_assistant_token",
                    },
                    "middle_band_layers": list(LAYERS),
                    "concepts": concepts,
                },
            }
        )
    return result


def token_text(prompt: dict[str, object]) -> dict[int, str]:
    return {
        form["token_id"]: form["text"]
        for concept in prompt["metadata"]["concepts"]
        for group in (concept["forms"], concept["foils"][0]["forms"])
        for form in group
    }


def lens(kind: str) -> dict[str, object]:
    common = {
        "d_model": 5120,
        "source_layers": list(MODULE.ALL_SOURCE_LAYERS),
        "tensor_shape": [5120, 5120],
    }
    if kind == "public":
        return {
            **common,
            "repo_id": MODULE.PUBLIC_LENS_REPO,
            "revision": MODULE.PUBLIC_LENS_REVISION,
            "sha256": MODULE.PUBLIC_LENS_SHA256,
            "n_prompts": 1000,
        }
    return {
        **common,
        "kind": "native_nvfp4_ste_fit",
        "sha256": MODULE.NATIVE_LENS_SHA256,
        "state_sha256": MODULE.NATIVE_STATE_SHA256,
        "provenance_sha256": MODULE.NATIVE_PROVENANCE_SHA256,
        "fit_model": MODULE.MODEL_REPO,
        "fit_model_revision": MODULE.MODEL_REVISION,
        "n_prompts": 10,
    }


def rank_values(
    prompt: dict[str, object], task_index: int, layer: int, method: str, kind: str
) -> dict[int, int]:
    values = {token_id: V for token_id in prompt["score_token_ids"]}
    if method == "logit":
        target_ranks = LOGIT_TARGET_RANKS[task_index]
        target_layer = 20
        foil_rank = 100
    elif kind == "public":
        target_ranks = PUBLIC_TARGET_RANKS[task_index]
        target_layer = 47
        foil_rank = 100
    else:
        target_ranks = NATIVE_TARGET_RANKS[task_index]
        target_layer = 30
        foil_rank = 100
    if layer == target_layer:
        for concept_index, concept in enumerate(prompt["metadata"]["concepts"]):
            # The first target form is the accepted token and must be ignored.
            values[concept["forms"][1]["token_id"]] = target_ranks[concept_index]
            values[concept["foils"][0]["forms"][1]["token_id"]] = foil_rank
    return values


def readout(
    prompt: dict[str, object], ranks: dict[int, int], generated_token_id: int
) -> dict[str, object]:
    text = token_text(prompt)
    return {
        "token_ids": [0],
        "tokens": ["!"],
        "scores": [1.0],
        "target_token_id": generated_token_id,
        "target_token": text[generated_token_id],
        "target_rank": 10,
        "target_score": 0.5,
        "target_logprob": -2.0,
        "scored_tokens": [
            {
                "token_id": token_id,
                "token": text[token_id],
                "rank": ranks[token_id],
                "score": 0.0,
                "logprob": -10.0,
            }
            for token_id in prompt["score_token_ids"]
        ],
    }


def make_report(prompts: list[dict[str, object]], kind: str) -> dict[str, object]:
    experiments = []
    for task_index, prompt in enumerate(prompts):
        generated_token_id = prompt["metadata"]["concepts"][0]["forms"][0]["token_id"]
        final_position = len(prompt["token_ids"]) - 1
        layers = []
        for layer in LAYERS:
            layers.append(
                {
                    "layer": layer,
                    "positions": [
                        {
                            "capture_index": 0,
                            "token_position": final_position,
                            "jacobian_lens": readout(
                                prompt,
                                rank_values(prompt, task_index, layer, "jacobian", kind),
                                generated_token_id,
                            ),
                            "logit_lens": readout(
                                prompt,
                                rank_values(prompt, task_index, layer, "logit", kind),
                                generated_token_id,
                            ),
                        }
                    ],
                }
            )
        text = token_text(prompt)
        experiments.append(
            {
                "id": prompt["id"],
                "prompt": prompt["text"],
                "prompt_token_ids": copy.deepcopy(prompt["token_ids"]),
                "prompt_tokens": ["a", "b", "c"],
                "metadata": copy.deepcopy(prompt["metadata"]),
                "generated_token_id": generated_token_id,
                "positions_requested": [-1],
                "positions_resolved": [final_position],
                "capture_positions_resolved": [final_position],
                "final_validation_position": final_position,
                "scored_vocabulary": {
                    "token_ids": copy.deepcopy(prompt["score_token_ids"]),
                    "tokens": [text[token_id] for token_id in prompt["score_token_ids"]],
                },
                "layers": layers,
                "residual_capture_manifest": {
                    "sha256": f"{task_index + 1:064x}",
                    "token_positions": [final_position],
                },
                "final_layer_top1_matches_greedy": True,
                "final_norm_reconstruction": {"within_tolerance": True},
                "final_logits_reconstruction": {
                    "within_tolerance": True,
                    "top_k_prefix_token_ids_match": True,
                },
            }
        )
    union_ids = list(
        dict.fromkeys(
            token_id for prompt in prompts for token_id in prompt["score_token_ids"]
        )
    )
    all_text = {
        token_id: value
        for prompt in prompts
        for token_id, value in token_text(prompt).items()
    }
    return {
        "schema_version": 3,
        "score_encoding": "unrounded-float32",
        "status": "passed",
        "model": {
            "repo_id": MODULE.MODEL_REPO,
            "revision": MODULE.MODEL_REVISION,
            "config_sha256": MODULE.MODEL_CONFIG_SHA256,
            "index_sha256": MODULE.MODEL_INDEX_SHA256,
        },
        "lens": lens(kind),
        "runtime": {
            "mtp_enabled": False,
            "enforce_eager": True,
            "language_model_only": True,
        },
        "scored_vocabulary": {
            "scope": "global_plus_per_experiment",
            "token_ids": [],
            "tokens": [],
            "union_token_ids": union_ids,
            "union_tokens": [all_text[token_id] for token_id in union_ids],
        },
        "assertions": {
            "lens_hash_matches": True,
            "lens_metadata_matches": True,
            "model_architecture_matches": True,
            "all_final_layer_top1_match_greedy": True,
            "all_final_adapter_reconstructions_within_tolerance": True,
        },
        "experiments": experiments,
    }


def fixture():
    prompts = make_prompts()
    return prompts, make_report(prompts, "public"), make_report(prompts, "native")


def u(rank: int) -> float:
    return math.log(V / rank) / math.log(V)


class AnalyzeSweMultitaskInitialProbesTest(unittest.TestCase):
    def test_external_input_path_is_logical_not_clone_root_dependent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps({"value": 1}), encoding="utf-8")
            value, source = MODULE.read_json(path)
        self.assertEqual(value, {"value": 1})
        self.assertEqual(source["path"], "external/input.json")

    def test_c1_requires_and_records_its_exact_checkpoint_contract(self) -> None:
        prompts, public, native = fixture()
        for prompt, public_experiment, native_experiment in zip(
            prompts,
            public["experiments"],
            native["experiments"],
            strict=True,
        ):
            prompt["metadata"]["checkpoint"] = copy.deepcopy(
                MODULE.CHECKPOINT_CONTRACTS["C1"]
            )
            prompt["metadata"]["observation_audit"] = {
                "capture_manifest_sha256": "d" * 64,
                "first_request_sha256": f"{len(prompt['id']):064x}",
                "second_request_sha256": f"{len(prompt['id']) + 1:064x}",
            }
            public_experiment["metadata"] = copy.deepcopy(prompt["metadata"])
            native_experiment["metadata"] = copy.deepcopy(prompt["metadata"])

        with self.assertRaisesRegex(ValueError, "expected C0 contract"):
            MODULE.analyze(
                prompts, public, native, bootstrap_seed=7, bootstrap_samples=10
            )
        result = MODULE.analyze(
            prompts,
            public,
            native,
            bootstrap_seed=7,
            bootstrap_samples=10,
            expected_checkpoint=MODULE.CHECKPOINT_CONTRACTS["C1"],
        )
        self.assertEqual(
            result["evaluation"]["checkpoint_metadata"],
            MODULE.CHECKPOINT_CONTRACTS["C1"],
        )
        self.assertIn("repository observation", result["evaluation"]["checkpoint"])
        self.assertEqual(result["coverage"]["task_count"], 4)
        self.assertEqual(
            result["kind"],
            (
                "exploratory_swe_verified_multitask_"
                "post_repository_observation_probe_analysis"
            ),
        )
        self.assertIn("post-repository-observation", result["label"])
        self.assertNotIn("task start", " ".join(result["evaluation"]["limitations"]))

    def test_c1_still_rejects_non_oracle_hidden_retained_concepts(self) -> None:
        prompts = make_prompts()
        for prompt in prompts:
            prompt["metadata"]["checkpoint"] = copy.deepcopy(
                MODULE.CHECKPOINT_CONTRACTS["C1"]
            )
            prompt["metadata"]["observation_audit"] = {
                "capture_manifest_sha256": "d" * 64,
                "first_request_sha256": "e" * 64,
                "second_request_sha256": "f" * 64,
            }
        prompts[0]["metadata"]["concepts"][0]["visibility"] = (
            "explicit_control_excluded"
        )
        with self.assertRaisesRegex(ValueError, "visibility mismatch"):
            MODULE.validate_prompt_bundle(
                prompts,
                expected_checkpoint=MODULE.CHECKPOINT_CONTRACTS["C1"],
            )

    def test_checkpoint_configuration_is_named_and_exact(self) -> None:
        args = MODULE.parse_args(
            [
                "--prompts",
                "prompts.json",
                "--public-report",
                "public.json",
                "--native-report",
                "native.json",
                "--output",
                "analysis.json",
                "--expected-checkpoint",
                "C1",
            ]
        )
        self.assertEqual(args.expected_checkpoint, "C1")
        prompts, _, _ = fixture()
        weakened = {
            "id": "C1",
            "name": "post_first_repository_observation",
            "visibility_boundary": "anything",
        }
        with self.assertRaisesRegex(ValueError, "supported exact contract"):
            MODULE.validate_prompt_bundle(
                prompts, expected_checkpoint=weakened
            )

    def test_capture_matched_checkpoint_records_trajectory_bindings(self) -> None:
        prompts, public, native = fixture()
        for index, (prompt, public_experiment, native_experiment) in enumerate(
            zip(prompts, public["experiments"], native["experiments"], strict=True)
        ):
            prompt["metadata"]["checkpoint"] = copy.deepcopy(
                MODULE.CHECKPOINT_CONTRACTS["C0M"]
            )
            prompt["metadata"]["capture_match"] = {
                "capture_manifest_sha256": "d" * 64,
                "first_request_sha256": f"{index + 1:064x}",
                "second_request_sha256": f"{index + 101:064x}",
            }
            public_experiment["metadata"] = copy.deepcopy(prompt["metadata"])
            native_experiment["metadata"] = copy.deepcopy(prompt["metadata"])

        result = MODULE.analyze(
            prompts,
            public,
            native,
            bootstrap_seed=7,
            bootstrap_samples=10,
            expected_checkpoint=MODULE.CHECKPOINT_CONTRACTS["C0M"],
        )
        self.assertEqual(
            result["kind"],
            "exploratory_swe_verified_multitask_capture_matched_initial_probe_analysis",
        )
        self.assertIn("same captured trajectory", result["evaluation"]["checkpoint"])
        self.assertEqual(len(result["source_bindings"]["trajectory_bindings"]), 4)
        self.assertEqual(
            result["source_bindings"]["trajectory_bindings"][0]["instance_id"],
            "project__project-0",
        )

    def test_family_then_task_weighting_bootstrap_foils_and_loro(self) -> None:
        prompts, public, native = fixture()
        result = MODULE.analyze(
            prompts,
            public,
            native,
            bootstrap_seed=7,
            bootstrap_samples=200,
        )
        repeated = MODULE.analyze(
            prompts,
            public,
            native,
            bootstrap_seed=7,
            bootstrap_samples=200,
        )
        self.assertEqual(result, repeated)
        self.assertEqual(
            result["kind"],
            "exploratory_swe_verified_multitask_initial_probe_analysis",
        )
        self.assertEqual(
            result["label"],
            (
                "EXPLORATORY MULTI-TASK PILOT: associative task-start concept "
                "readout, not chain-of-thought recovery or causal evidence"
            ),
        )

        public_method = result["methods"]["public_jacobian"]
        first_task = public_method["tasks"][0]
        self.assertAlmostEqual(first_task["target_utility_u"], 0.75)
        self.assertAlmostEqual(first_task["target_pass_at_k"]["10"], 0.75)
        self.assertEqual(
            first_task["families"][0]["concepts"][0]["target_score"]["best_layer"],
            47,
        )
        self.assertEqual(len(public_method["accepted_generated_token_exclusions"]), 4)
        self.assertEqual(
            public_method["accepted_generated_token_exclusions"][0]["role"],
            "target",
        )

        expected_public = statistics.fmean(
            (
                0.75,
                u(5),
                u(10),
                u(50),
            )
        )
        self.assertAlmostEqual(public_method["target_utility_u"], expected_public)
        comparison = result["comparisons"]["public_minus_logit"]
        self.assertEqual(comparison["task_utility_u"]["samples"], 200)
        self.assertEqual(comparison["task_utility_u"]["positive_task_count"], 3)
        self.assertEqual(comparison["task_utility_u"]["negative_task_count"], 1)
        self.assertEqual(comparison["task_utility_u"]["tie_task_count"], 0)
        self.assertEqual(len(comparison["leave_one_repo_out"]["task_utility_u"]), 2)
        self.assertEqual(set(comparison["pass_at_k"]), {str(k) for k in MODULE.PASS_K})

        foil = result["comparisons"]["target_minus_foil"]
        self.assertIsNotNone(foil)
        for method in ("public_jacobian", "native_jacobian", "logit_lens"):
            self.assertEqual(foil[method]["task_count"], 4)
            self.assertEqual(
                len(foil[method]["leave_one_repo_out"]["task_utility_u"]), 2
            )
        self.assertIsNone(result["evaluation"]["claims_gate"])
        self.assertTrue(result["pairing"]["accepted_generated_tokens_equal"])

    def test_rejects_prompt_text_token_metadata_and_vocabulary_mismatches(self) -> None:
        mutators = (
            (
                lambda prompts, public, native: public["experiments"][0].__setitem__(
                    "prompt", "changed"
                ),
                "prompt text",
            ),
            (
                lambda prompts, public, native: public["experiments"][0][
                    "prompt_token_ids"
                ].__setitem__(0, 99),
                "token IDs",
            ),
            (
                lambda prompts, public, native: public["experiments"][0]["metadata"].__setitem__(
                    "kind", "changed"
                ),
                "metadata",
            ),
            (
                lambda prompts, public, native: public["experiments"][0][
                    "scored_vocabulary"
                ]["token_ids"].pop(),
                "vocabulary",
            ),
        )
        for mutate, message in mutators:
            with self.subTest(message=message):
                prompts, public, native = fixture()
                mutate(prompts, public, native)
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.analyze(
                        prompts, public, native, bootstrap_seed=7, bootstrap_samples=10
                    )

    def test_rejects_layer_rank_generated_token_and_residual_mismatches(self) -> None:
        cases = (
            (
                lambda prompts, public, native: public["experiments"][0]["layers"].pop(),
                "fixed layers",
            ),
            (
                lambda prompts, public, native: public["experiments"][0]["layers"][0][
                    "positions"
                ][0]["jacobian_lens"]["scored_tokens"][0].pop("rank"),
                "rank",
            ),
            (
                lambda prompts, public, native: native["experiments"][0].__setitem__(
                    "generated_token_id",
                    native["experiments"][0]["generated_token_id"] + 1,
                ),
                "target|generated token|accepted",
            ),
            (
                lambda prompts, public, native: native["experiments"][0][
                    "residual_capture_manifest"
                ].__setitem__("sha256", "f" * 64),
                "residual_capture_manifest|residual",
            ),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                prompts, public, native = fixture()
                mutate(prompts, public, native)
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.analyze(
                        prompts, public, native, bootstrap_seed=7, bootstrap_samples=10
                    )

    def test_rejects_unfrozen_selection_score_union_and_target_override(self) -> None:
        cases = (
            (
                lambda prompts: prompts[0]["metadata"].__setitem__(
                    "lens_outputs_used_for_selection", True
                ),
                "frozen",
            ),
            (lambda prompts: prompts[0]["score_token_ids"].pop(), "form union"),
            (lambda prompts: prompts[0].__setitem__("target_token_id", 3), "override"),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                prompts, public, native = fixture()
                mutate(prompts)
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.analyze(
                        prompts, public, native, bootstrap_seed=7, bootstrap_samples=10
                    )


if __name__ == "__main__":
    unittest.main()
