from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_hidden_future_concepts.py"
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_hidden_future_concepts.json"

spec = importlib.util.spec_from_file_location(
    "swe_task_state_v4_hidden_future_concepts", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def absence_evidence(aliases: list[str]) -> dict:
    return {
        "normalization": "NFKC_then_casefold_with_snake_camel_identifier_segments_v1",
        "aliases": aliases,
        "identifier_hits": [],
        "scored_form_token_id_hits": [],
        "exposed": False,
    }


def form(kind: str, text: str, token_id: int) -> dict:
    return {"kind": kind, "text": text, "token_id": token_id}


def future_support() -> dict:
    return {
        "contract": "intersection_of_agent_generated_patch_mutation_completion_and_terminal_summary_v1",
        "benchmark_gold_used": False,
        "lens_output_used": False,
        "generated_patch": {
            "path": "generation/task/patch.diff",
            "sha256": "a" * 64,
            "source_path": "package/module.py",
            "patch_line_number": 10,
            "line_sha256": "b" * 64,
            "span": [1, 6],
        },
        "mutation_completion": {
            "completion_index": 3,
            "source_request_global_index": 10,
            "next_request_global_index": 11,
            "source_campaign_source_request_global_index": 4,
            "source_campaign_next_request_global_index": 5,
            "channel": "argument_text",
            "channel_text_sha256": "c" * 64,
            "span": [2, 7],
        },
        "terminal_summary": {
            "runner_metadata_path": "generation/task/runner_metadata.json",
            "runner_metadata_sha256": "d" * 64,
            "field": "/qwen/result_tail",
            "text_sha256": "e" * 64,
            "span": [3, 8],
        },
    }


def target_and_eligibility(*, foil_source: str = "generated_patch_removed_identifier"):
    target = {
        "id": "target-00-alpha",
        "kind": "identifier",
        "target": "alpha",
        "forms": [form("bare", "alpha", 11), form("leading_space", " alpha", 12)],
        "aliases": ["alpha"],
        "future_support": future_support(),
        "task_instance_id": "task-a",
        "foils": [
            {
                "id": "foil-00-beta",
                "task_instance_id": "task-a",
                "kind": "identifier",
                "target": "beta",
                "forms": [form("bare", "beta", 13)],
                "aliases": ["beta"],
                "source": {
                    "type": foil_source,
                    "path": "package/module.py",
                    "patch_line_number": 9,
                    "line_sha256": "f" * 64,
                },
            }
        ],
    }
    eligibility = {
        "target_id": "target-00-alpha",
        "target_exposed": False,
        "retained_hidden_foil_ids": ["foil-00-beta"],
        "excluded_foils": [],
        "status": "eligible",
        "target_channel_evidence": {
            "system": absence_evidence(["alpha"]),
            "user": absence_evidence(["alpha"]),
        },
        "target_rendered_evidence": absence_evidence(["alpha"]),
        "foil_evidence": [
            {
                "foil_id": "foil-00-beta",
                "exposed": False,
                "channel_evidence": {
                    "system": absence_evidence(["beta"]),
                    "user": absence_evidence(["beta"]),
                },
                "rendered_evidence": absence_evidence(["beta"]),
            }
        ],
    }
    return target, eligibility


def neutral_candidates() -> list[dict]:
    return [
        {
            "candidate_id": "candidate-a",
            "candidate_text": "alpha",
            "forms": [
                form("bare", "alpha", 1),
                form("leading_space", " alpha", 2),
            ],
        },
        {
            "candidate_id": "candidate-b",
            "candidate_text": "beta",
            "forms": [form("bare", "beta", 3)],
        },
    ]


def readout(scores: dict[int, float], ranks: dict[int, int]) -> dict:
    texts = {1: "alpha", 2: " alpha", 3: "beta"}
    rows = [
        {
            "token_id": token_id,
            "token": texts[token_id],
            "score": float(scores[token_id]),
            "logprob": float(scores[token_id] - 10.0),
            "rank": ranks[token_id],
        }
        for token_id in (1, 2, 3)
    ]
    return {
        "scored_tokens": rows,
        "scores": [row["score"] for row in rows],
        "token_ids": [row["token_id"] for row in rows],
        "tokens": [row["token"] for row in rows],
        "target_logprob": -1.0,
        "target_rank": 1,
        "target_score": 1.0,
        "target_token": "unused",
        "target_token_id": 999,
        "final_distribution_fidelity": {},
    }


def scoring_experiment() -> dict:
    layers = []
    for layer in module.LAYERS:
        layers.append(
            {
                "layer": layer,
                "positions": [
                    {
                        "token_position": 4,
                        "logit_lens": readout(
                            {1: 0.0, 2: 0.0, 3: -1.0}, {1: 5, 2: 10, 3: 20}
                        ),
                        "jacobian_lens": readout(
                            {1: -2.0, 2: -2.0, 3: 0.0}, {1: 50, 2: 40, 3: 2}
                        ),
                    }
                ],
            }
        )
    return {
        "layers": layers,
        "generated_text": "forbidden completion must be inert",
        "metadata": {"source_kind": "target"},
    }


def alignment_index(rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "kind": "swe_task_state_v4_label_free_alignment_index",
        "status": "passed",
        "scope": "grouping_order_and_stability_only_no_labels",
        "config": {},
        "implementation": {},
        "sources": [],
        "eligibility_source": {},
        "row_count": len(rows),
        "stable_row_count": sum(row["stable_feature_eligible"] for row in rows),
        "feature_use": {
            "allowed": [
                "task-local ordering for causal temporal transforms",
                "repository and task grouping for held-out splits and weights",
                "stable eligibility filtering",
            ],
            "forbidden": [
                "hashing or one-hot encoding IDs as model features",
                "repository or request index as semantic model features",
            ],
        },
        "rows": rows,
    }


def alignment_row(index: int, prompt_id: str, request: int, *, stable: bool = True) -> dict:
    return {
        "global_index": index,
        "source_id_sha256": module.sha256_text(prompt_id),
        "task_id_sha256": module.sha256_text("task-a"),
        "repository": "owner/repo",
        "request_index": request,
        "stable_feature_eligible": stable,
    }


def metric_row(
    repository: str,
    task: str,
    target: str,
    boundary: int,
    *,
    ordinary: float = 0.25,
    public: float = 0.25,
) -> dict:
    def method(value: float) -> dict:
        return {
            "evaluation": {"metrics": {name: value for name in module.ALL_METRICS}},
            "scored": {
                "predicted_candidate_id": "candidate-a",
                "candidates": [
                    {"candidate_id": "candidate-a", "candidate_text": "alpha"}
                ],
            },
        }

    return {
        "repository": repository,
        "task_id_sha256": task,
        "future_target_id": target,
        "global_index": boundary,
        "request_index": boundary + 1,
        "methods": {
            "ordinary_logit": method(ordinary),
            "public_jacobian": method(public),
        },
    }


class HiddenFutureConceptTests(unittest.TestCase):
    def config(self) -> dict:
        self.assertEqual(module.sha256_file(CONFIG_PATH), module.CONFIG_SHA256)
        return module.validate_config(module.load_json_strict(CONFIG_PATH))

    def test_config_freezes_sources_scoring_estimand_and_limitations(self):
        config = self.config()
        self.assertEqual(config["scoring"]["layers_in_order"], list(range(24, 48)))
        self.assertEqual(config["scoring"]["methods_in_order"], list(module.METHODS))
        self.assertEqual(config["source_contract"]["all_boundary_count"], 1708)
        self.assertEqual(
            config["source_contract"]["eligible_target_boundary_instances_before_stability_filter"],
            77,
        )
        self.assertEqual(config["mandatory_limitations"], list(module.MANDATORY_LIMITATIONS))
        self.assertFalse(config["claim_scope"]["private_chain_of_thought_reconstructed"])
        self.assertFalse(config["claim_scope"]["emotion_confidence_doubt_or_stress_decoded"])
        self.assertFalse(config["claim_scope"]["cot_or_cot_like_decoded"])
        self.assertFalse(config["claim_scope"]["semantic_concept_chain_decoded"])
        self.assertTrue(config["claim_scope"]["proposition_relation_supplied_not_decoded"])

    def test_positive_claims_are_only_identifier_surface_and_supplied_relation(self):
        config = self.config()
        positive = {
            key for key, value in config["claim_scope"].items() if value is True
        }
        self.assertEqual(
            positive,
            {
                "target_and_retained_foil_identifier_surface_forms_absent_from_visible_prefix",
                "fixed_future_derived_identifier_candidate_ranking_evaluated",
                "proposition_relation_supplied_not_decoded",
            },
        )
        template = config["renderer"]["identifier_ranking_template"].lower()
        self.assertIn("identifier surface forms", template)
        self.assertNotIn("concept", template)
        self.assertNotIn("proposition", template)
        self.assertTrue(
            config["renderer"]["rendered_sentence_is_fixed_wrapper_not_decoded_language"]
        )

    def test_config_mutations_fail_closed(self):
        config = self.config()
        mutations = []
        changed = copy.deepcopy(config)
        changed["scoring"]["layers_in_order"][0] = 23
        mutations.append(changed)
        changed = copy.deepcopy(config)
        changed["source_contract"]["benchmark_gold_used"] = True
        mutations.append(changed)
        changed = copy.deepcopy(config)
        changed["claim_scope"]["private_chain_of_thought_reconstructed"] = True
        mutations.append(changed)
        changed = copy.deepcopy(config)
        changed["claim_scope"]["cot_or_cot_like_decoded"] = True
        mutations.append(changed)
        changed = copy.deepcopy(config)
        changed["mandatory_limitations"].pop()
        mutations.append(changed)
        changed = copy.deepcopy(config)
        changed["code_dependencies"][0]["sha256"] = "0" * 64
        mutations.append(changed)
        for value in mutations:
            with self.subTest(value=value), self.assertRaises(module.HiddenFutureConceptError):
                module.validate_config(value)

    def test_candidate_builder_recomputes_prefix_absence_and_strips_source_kind(self):
        target, eligibility = target_and_eligibility()
        candidates, true_candidate_id = module.build_hidden_candidate_set(
            target=target,
            eligibility=eligibility,
            task_instance_id="task-a",
            prefix_text="unrelated visible prefix",
            prefix_token_ids=[90, 91],
        )
        self.assertEqual(len(candidates), 2)
        self.assertIn(true_candidate_id, {row["candidate_id"] for row in candidates})
        self.assertTrue(
            all(set(row) == {"candidate_id", "candidate_text", "forms"} for row in candidates)
        )
        serialized = json.dumps(candidates)
        self.assertNotIn("generated_patch", serialized)
        self.assertNotIn("target-00", serialized)
        self.assertNotIn("foil-00", serialized)

        context_target, context_eligibility = target_and_eligibility(
            foil_source="generated_patch_context_identifier"
        )
        repeated, _ = module.build_hidden_candidate_set(
            target=context_target,
            eligibility=context_eligibility,
            task_instance_id="task-a",
            prefix_text="unrelated visible prefix",
            prefix_token_ids=[90, 91],
        )
        self.assertEqual(candidates, repeated)

    def test_prefix_support_and_retained_foil_mutations_fail_closed(self):
        target, eligibility = target_and_eligibility()
        with self.assertRaisesRegex(module.HiddenFutureConceptError, "appears in"):
            module.build_hidden_candidate_set(
                target=target,
                eligibility=eligibility,
                task_instance_id="task-a",
                prefix_text="we will call alpha next",
                prefix_token_ids=[90],
            )
        with self.assertRaisesRegex(module.HiddenFutureConceptError, "token form"):
            module.build_hidden_candidate_set(
                target=target,
                eligibility=eligibility,
                task_instance_id="task-a",
                prefix_text="unrelated visible prefix",
                prefix_token_ids=[13],
            )
        leaked = copy.deepcopy(target)
        leaked["future_support"]["benchmark_gold_used"] = True
        with self.assertRaisesRegex(module.HiddenFutureConceptError, "forbidden evidence"):
            module.build_hidden_candidate_set(
                target=leaked,
                eligibility=eligibility,
                task_instance_id="task-a",
                prefix_text="unrelated visible prefix",
                prefix_token_ids=[90],
            )
        invalid = copy.deepcopy(eligibility)
        invalid["retained_hidden_foil_ids"] = []
        with self.assertRaisesRegex(module.HiddenFutureConceptError, "retained hidden foil"):
            module.build_hidden_candidate_set(
                target=target,
                eligibility=invalid,
                task_instance_id="task-a",
                prefix_text="unrelated visible prefix",
                prefix_token_ids=[90],
            )

    def test_score_reductions_predictions_and_open_vocabulary_ranks_are_exact(self):
        experiment = scoring_experiment()
        ordinary = module.score_candidate_set(
            experiment=experiment,
            candidates=neutral_candidates(),
            method="ordinary_logit",
            expected_token_position=4,
        )
        public = module.score_candidate_set(
            experiment=experiment,
            candidates=neutral_candidates(),
            method="public_jacobian",
            expected_token_position=4,
        )
        self.assertEqual(ordinary["predicted_candidate_id"], "candidate-a")
        self.assertEqual(public["predicted_candidate_id"], "candidate-b")
        by_id = {row["candidate_id"]: row for row in ordinary["candidates"]}
        self.assertAlmostEqual(by_id["candidate-a"]["candidate_score"], 0.0)
        self.assertAlmostEqual(by_id["candidate-b"]["candidate_score"], -1.0)
        expected_utility = math.log(module.VOCABULARY_SIZE / 5) / math.log(
            module.VOCABULARY_SIZE
        )
        self.assertAlmostEqual(
            by_id["candidate-a"]["open_vocabulary_rank_utility"], expected_utility
        )
        self.assertEqual(
            by_id["candidate-a"]["open_vocabulary_top_k_layer_fraction"]["10"],
            1.0,
        )

    def test_prediction_api_cannot_accept_target_labels_or_grouping(self):
        parameters = inspect.signature(module.score_candidate_set).parameters
        for forbidden in (
            "target",
            "true_target",
            "labels",
            "source_kind",
            "repository",
            "task",
            "completion",
        ):
            self.assertNotIn(forbidden, parameters)
        candidate = copy.deepcopy(neutral_candidates()[0])
        candidate["source_kind"] = "target"
        with self.assertRaisesRegex(module.HiddenFutureConceptError, "source-kind"):
            module.score_candidate_set(
                experiment=scoring_experiment(),
                candidates=[candidate, neutral_candidates()[1]],
                method="ordinary_logit",
                expected_token_position=4,
            )

    def test_true_target_and_completion_mutations_do_not_change_prediction(self):
        experiment = scoring_experiment()
        scored = module.score_candidate_set(
            experiment=experiment,
            candidates=neutral_candidates(),
            method="ordinary_logit",
            expected_token_position=4,
        )
        changed = copy.deepcopy(experiment)
        changed["generated_text"] = "a completely different future completion"
        changed["metadata"]["source_kind"] = "foil"
        repeated = module.score_candidate_set(
            experiment=changed,
            candidates=neutral_candidates(),
            method="ordinary_logit",
            expected_token_position=4,
        )
        self.assertEqual(scored, repeated)
        evaluation_a = module.evaluate_scored_candidate_set(
            scored, true_candidate_id="candidate-a"
        )
        evaluation_b = module.evaluate_scored_candidate_set(
            scored, true_candidate_id="candidate-b"
        )
        self.assertEqual(
            evaluation_a["predicted_candidate_id"], evaluation_b["predicted_candidate_id"]
        )
        self.assertNotEqual(
            evaluation_a["metrics"]["top1_accuracy"],
            evaluation_b["metrics"]["top1_accuracy"],
        )
        self.assertNotIn(
            "true_target",
            inspect.signature(module.render_identifier_ranking).parameters,
        )
        self.assertNotIn(
            "proposition",
            inspect.signature(module.render_identifier_ranking).parameters,
        )

    def test_alignment_and_prompt_report_bindings_are_strict(self):
        rows = [
            alignment_row(0, "p1", 1),
            alignment_row(1, "p2", 2, stable=False),
        ]
        observed = module.validate_alignment_index(
            alignment_index(rows), expected_total_count=2, expected_stable_count=1
        )
        self.assertEqual(observed, rows)
        changed = alignment_index(copy.deepcopy(rows))
        changed["rows"][0]["label"] = "target"
        with self.assertRaisesRegex(module.HiddenFutureConceptError, "schema changed"):
            module.validate_alignment_index(
                changed, expected_total_count=2, expected_stable_count=1
            )

        prompt = {
            "id": "p1",
            "text": "visible prefix",
            "token_ids": [4, 5],
            "score_token_ids": [1, 2, 3],
            "metadata": {
                "task": {"instance_id": "task-a", "repo": "owner/repo"},
                "selection": {"task_request_index": 1},
            },
        }
        experiment = {
            "id": "p1",
            "prompt": "visible prefix",
            "prompt_token_ids": [4, 5],
            "metadata": copy.deepcopy(prompt["metadata"]),
            "capture_positions_resolved": [1],
            "scored_vocabulary": {"token_ids": [1, 2, 3], "tokens": ["a", "b", "c"]},
            "generated_text": "ignored",
        }
        binding = module.validate_prompt_report_binding(
            prompt=prompt, experiment=experiment, alignment_row=rows[0], global_index=0
        )
        self.assertEqual(binding["expected_token_position"], 1)
        rebound = copy.deepcopy(experiment)
        rebound["metadata"]["selection"]["task_request_index"] = 2
        with self.assertRaisesRegex(module.HiddenFutureConceptError, "payload binding"):
            module.validate_prompt_report_binding(
                prompt=prompt, experiment=rebound, alignment_row=rows[0], global_index=0
            )

    def test_hierarchical_weights_equalize_every_declared_level(self):
        rows = [
            metric_row("repo-a", "task-a1", "target-1", 0),
            metric_row("repo-a", "task-a1", "target-1", 1),
            metric_row("repo-a", "task-a1", "target-2", 2),
            metric_row("repo-a", "task-a2", "target-1", 3),
            metric_row("repo-b", "task-b1", "target-1", 4),
        ]
        weights = module.hierarchical_row_weights(rows)
        np.testing.assert_allclose(weights, [0.0625, 0.0625, 0.125, 0.25, 0.5])
        self.assertAlmostEqual(float(weights[:4].sum()), 0.5)
        metrics = module.weighted_method_metrics(rows, weights)
        self.assertEqual(metrics["ordinary_logit"], metrics["public_jacobian"])

    def test_paired_cluster_bootstrap_is_deterministic_and_keeps_zero_delta(self):
        rows = [
            metric_row("repo-a", "task-a", "target-1", 0),
            metric_row("repo-a", "task-a", "target-1", 1),
            metric_row("repo-b", "task-b", "target-2", 2),
        ]
        weights = module.hierarchical_row_weights(rows)
        point = module.weighted_method_metrics(rows, weights)
        first = module.paired_repository_task_bootstrap(
            rows,
            point_metrics=point,
            draw_count=50,
            seed=7,
            confidence_level=0.95,
        )
        second = module.paired_repository_task_bootstrap(
            rows,
            point_metrics=point,
            draw_count=50,
            seed=7,
            confidence_level=0.95,
        )
        self.assertEqual(first, second)
        for result in first["results"].values():
            self.assertEqual(result["interval"], [0.0, 0.0])
            self.assertEqual(
                result["point_delta_public_jacobian_minus_ordinary_logit"], 0.0
            )
        self.assertTrue(first["complete_target_trajectories_retained"])

    def test_identifier_ranking_chains_are_fixed_sorted_and_target_independent(self):
        rows = [
            metric_row("repo-a", "task-a", "target-1", 2),
            metric_row("repo-a", "task-a", "target-1", 0),
        ]
        chains = module.build_identifier_ranking_chains(
            rows, template=module.EXPECTED_RENDERER["identifier_ranking_template"]
        )
        requests = [
            row["request_index"]
            for row in chains[0]["methods"]["ordinary_logit"]
        ]
        self.assertEqual(requests, [1, 3])
        rendered = chains[0]["methods"]["ordinary_logit"][0][
            "rendered_identifier_ranking"
        ]
        self.assertIn("ordinary_logit", rendered)
        self.assertIn("`alpha`", rendered)
        self.assertNotIn("concept", rendered.lower())
        self.assertNotIn("proposition", rendered.lower())
        self.assertTrue(
            all(
                not row["true_target_text_used_to_select_or_render"]
                and row["proposition_relation_supplied_not_decoded"]
                for row in chains[0]["methods"]["ordinary_logit"]
            )
        )

    def test_authenticated_record_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text("{}", encoding="utf-8")
            record = {
                "path": str(path),
                "sha256": hashlib.sha256(b"{}").hexdigest(),
                "size_bytes": 2,
            }
            self.assertEqual(module.authenticate_records([record], label="fixture"), [path])
            path.write_text("{ }", encoding="utf-8")
            with self.assertRaisesRegex(module.HiddenFutureConceptError, "byte binding"):
                module.authenticate_records([record], label="fixture")

    def test_forbidden_paths_precede_every_filesystem_operation(self):
        args = module.build_parser().parse_args(
            ["--output", "/tmp/validation/hidden-future.json"]
        )
        with mock.patch.object(
            module,
            "canonical_path_preflight",
            side_effect=AssertionError("resolve touched"),
        ), mock.patch.object(
            module, "sha256_file", side_effect=AssertionError("hash touched")
        ), mock.patch.object(
            module, "load_json_strict", side_effect=AssertionError("read touched")
        ):
            with self.assertRaisesRegex(module.HiddenFutureConceptError, "forbidden path"):
                module.run(args)

    def test_canonical_parent_symlink_and_output_clobber_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closed = root / "reserved_material"
            closed.mkdir()
            source = closed / "source.json"
            source.write_text("{}", encoding="utf-8")
            alias = root / "allowed_alias"
            alias.symlink_to(closed, target_is_directory=True)
            with self.assertRaisesRegex(module.HiddenFutureConceptError, "canonical path"):
                module.canonical_path_preflight(
                    input_paths=[alias / "source.json"], output_paths=[]
                )

            output = root / "report.json"
            module._write_json_no_clobber(output, {"status": "passed"})
            self.assertEqual(json.loads(output.read_text()), {"status": "passed"})
            with self.assertRaisesRegex(module.HiddenFutureConceptError, "overwrite"):
                module._write_json_no_clobber(output, {"status": "changed"})


if __name__ == "__main__":
    unittest.main()
