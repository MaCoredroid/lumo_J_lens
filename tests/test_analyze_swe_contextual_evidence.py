#!/usr/bin/env python3
"""Focused tests for the frozen contextual-evidence update analysis."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_swe_contextual_evidence",
    ROOT / "scripts" / "analyze_swe_contextual_evidence.py",
)
assert SPEC and SPEC.loader
ANALYZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZE
SPEC.loader.exec_module(ANALYZE)


def concept(task_index: int, concept_index: int, *, target: bool) -> dict:
    token_id = 100 + task_index * 10 + concept_index
    label = f"entity_{task_index}_{concept_index}"
    return {
        "id": f"concept-{task_index}-{concept_index}",
        "label": label,
        "aliases": [label],
        "exposure_normalization": "case_sensitive_identifier_boundary_v1",
        "forms": [{"kind": "leading_space", "text": f" {label}", "token_id": token_id}],
        "expected_exposure": {
            "before": {"present": False, "identifier_occurrences": 0},
            "after": {"present": target, "identifier_occurrences": int(target)},
        },
        "future_present": target,
    }


def make_protocol() -> dict:
    tasks = []
    repositories = ("owner/repo-a", "owner/repo-a", "owner/repo-b", "owner/repo-c")
    eligibility = (True, True, True, False)
    statuses = (
        "matched_newly_exposed_target_and_foils",
        "matched_newly_exposed_target_and_foils",
        "matched_novel_unexposed_target_and_foils",
        "descriptive_exposure_frequency_mismatch",
    )
    for index, (repo, eligible, status) in enumerate(
        zip(repositories, eligibility, statuses, strict=True)
    ):
        tasks.append(
            {
                "id": f"task-{index}",
                "instance_id": f"{repo.replace('/', '__')}-{index}",
                "repo": repo,
                "cohort": "development" if index < 2 else "replication",
                "after_global_request_index": 10 + index * 3,
                "after_task_request_index": 2 + index,
                "raw_sha256": {
                    "before": f"{index + 1:064x}",
                    "after": f"{index + 11:064x}",
                    "label": f"{index + 21:064x}",
                },
                "stratum": "novel_inference" if index == 2 else "evidence_reweighting",
                "primary_control_eligible": eligible,
                "control_match_status": status,
                "target": concept(index, 0, target=True),
                "foils": [concept(index, foil, target=False) for foil in range(1, 4)],
                "task_card": {
                    "why": f"entity_{index}_0 becomes relevant",
                    "where": f"module_{index}.py",
                    "evidence": f"bound transition {index}",
                    "next": f"inspect entity_{index}_0",
                    "claim_scope": "synthetic fixture",
                },
            }
        )
    return {
        "schema_version": 1,
        "kind": ANALYZE.PROTOCOL_KIND,
        "analysis_version": "paired-evidence-update-development-v1",
        "status": "development_protocol_frozen_before_contextual_replay",
        "lens_outputs_used_for_boundary_or_labels": False,
        "pins": {
            "model": {
                "repo_id": "nvidia/Qwen3.6-27B-NVFP4",
                "revision": "r" * 40,
                "config_sha256": "a" * 64,
                "index_sha256": "b" * 64,
            },
            "lenses": {
                "public": {"repo_id": "neuronpedia/jacobian-lens", "revision": "p" * 40, "sha256": "1" * 64},
                "nf4": {"sha256": "2" * 64},
                "native_nvfp4_ste": {"sha256": "3" * 64},
            },
        },
        "fixed_layer_band": {
            "start": 24,
            "end": 47,
            "end_inclusive": True,
            "layers": list(range(24, 48)),
        },
        "score_reduction": {
            "within_concept": "logmeanexp_over_declared_token_scores",
            "within_foil_set": "logmeanexp_over_three_concept_scores",
            "across_layers": "arithmetic_mean_over_fixed_layers_24_through_47",
            "layer_selection": "none",
        },
        "numerical_certification": {
            "primary_stable": {
                "final_norm_max_abs_tolerance": 0.125,
                "final_norm_rms_tolerance": 0.006,
                "final_logits_max_abs_tolerance": 0.125,
                "final_logits_rms_tolerance": 0.02,
                "top_k_prefix": 5,
            },
            "legacy_strict": {
                "final_norm_max_abs_tolerance": 0.125,
                "final_norm_rms_tolerance": 0.006,
                "final_logits_max_abs_tolerance": 0.0625,
                "final_logits_rms_tolerance": 0.01,
                "top_k_prefix": 5,
            },
        },
        "controls": {
            "copy_frequency": "paired_log1p_exact_form_token_count_margin",
            "copy_recency": "paired_negative_log1p_last_exact_form_token_distance_margin",
            "wrong_task": "score_every_task_target_on_every_task_pair_and_rank_own_target_by_after_minus_before_update",
        },
        "bootstrap": {
            "algorithm": "hierarchical_repository_then_task_percentile_v1",
            "seed": 73,
            "samples": 100,
            "confidence_level": 0.95,
        },
        "decision": {
            "minimum_primary_tasks": 3,
            "minimum_primary_repositories": 2,
        },
        "task_card_policy": {
            "observed_fields": ["evidence", "where", "next"],
            "lens_scored_field": "why",
        },
        "tasks": tasks,
    }


def document_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def exposure(task: dict, state: str, token_ids: list[int]) -> dict:
    def record(item: dict) -> dict:
        token_id = item["forms"][0]["token_id"]
        present = state == "after" and item is task["target"]
        indices = [index for index, observed in enumerate(token_ids) if observed == token_id]
        identifier = {
            "normalization": "case_sensitive_identifier_boundary_v1",
            "matching": "exact_alias_with_ascii_identifier_boundaries",
            "identifier_occurrences": int(present),
            "present": present,
            "per_alias": [{"alias": item["aliases"][0], "occurrences": int(present)}],
        }
        return {
            "id": item["id"],
            **identifier,
            "source": "newline_joined_recursive_string_values_from_raw_request_messages",
            "raw_messages_sha256": "e" * 64,
            "supplemental_rendered": {
                "case_sensitive": identifier,
                "nfkc_casefold": {
                    "normalization": "NFKC_then_casefold",
                    "matching": "exact_alias_with_unicode_identifier_boundaries",
                    "identifier_occurrences": int(present),
                    "present": present,
                    "per_alias": [
                        {
                            "alias": item["aliases"][0],
                            "normalized_alias": item["aliases"][0],
                            "occurrences": int(present),
                        }
                    ],
                },
            },
            "forms": [
                {
                    "kind": "leading_space",
                    "text": item["forms"][0]["text"],
                    "token_id": token_id,
                    "token_occurrences": len(indices),
                    "last_token_distance": len(token_ids) - 1 - indices[-1] if indices else None,
                }
            ],
        }

    return {
        "target": record(task["target"]),
        "foils": [record(item) for item in task["foils"]],
    }


def make_prompts_and_manifest(protocol: dict, protocol_sha: str) -> tuple[list[dict], dict]:
    score_ids = sorted(
        form["token_id"]
        for task in protocol["tasks"]
        for item in [task["target"], *task["foils"]]
        for form in item["forms"]
    )
    token_text = {
        form["token_id"]: form["text"]
        for task in protocol["tasks"]
        for item in [task["target"], *task["foils"]]
        for form in item["forms"]
    }
    prompts: list[dict] = []
    manifest_tasks = []
    for ordinal, task in enumerate(protocol["tasks"]):
        manifest_prompts = []
        concepts = {"target": task["target"], "foils": task["foils"]}
        task_identity = {
            field: task[field]
            for field in (
                "id",
                "instance_id",
                "repo",
                "cohort",
                "after_global_request_index",
                "after_task_request_index",
                "stratum",
                "primary_control_eligible",
                "control_match_status",
            )
        }
        for state in ("before", "after"):
            target_id = task["target"]["forms"][0]["token_id"]
            # The third task deliberately receives a wrong-task token after the boundary.
            added = (
                protocol["tasks"][0]["target"]["forms"][0]["token_id"]
                if ordinal == 2
                else target_id
            )
            token_ids = [7, 8] if state == "before" else [7, added, 8]
            prompt_text = f"synthetic task {ordinal} {state}"
            prompt_id = f"swe-contextual-evidence-task-{ordinal}-{state}"
            prompt_exposure = exposure(task, state, token_ids)
            prompt_provenance = {
                "sha256": ANALYZE.sha256_text(prompt_text),
                "token_ids_sha256": ANALYZE.sha256_json(token_ids),
                "token_count": len(token_ids),
                "normalized_messages_sha256": "f" * 64,
                "normalized_string_tool_call_arguments": 0,
            }
            prompts.append(
                {
                    "id": prompt_id,
                    "text": prompt_text,
                    "token_ids": token_ids,
                    "score_token_ids": score_ids,
                    "metadata": {
                        "kind": ANALYZE.PROMPT_KIND,
                        "analysis_version": "paired-evidence-update-development-v1",
                        "protocol_sha256": protocol_sha,
                        "lens_outputs_used_for_boundary_or_labels": False,
                        "state": state,
                        "task": task_identity,
                        "raw_sha256": task["raw_sha256"],
                        "prompt": prompt_provenance,
                        "concepts": concepts,
                        "exposure": prompt_exposure,
                        "task_card": task["task_card"],
                        "fixed_layer_band": protocol["fixed_layer_band"],
                        "score_reduction": protocol["score_reduction"],
                    },
                }
            )
            manifest_prompts.append(
                {
                    "id": prompt_id,
                    "state": state,
                    "prompt_sha256": prompt_provenance["sha256"],
                    "token_ids_sha256": prompt_provenance["token_ids_sha256"],
                    "prompt_token_count": len(token_ids),
                    "score_token_ids_sha256": ANALYZE.sha256_json(score_ids),
                    "score_token_count": len(score_ids),
                    "exposure": prompt_exposure,
                }
            )
        manifest_tasks.append(
            {
                "task_ordinal": ordinal,
                "task": task_identity,
                "raw_sources": {
                    state: {"path": f"chat_{ordinal}_{state}.json", "bytes": 10, "sha256": task["raw_sha256"][state]}
                    for state in ("before", "after", "label")
                },
                "boundary_audit": {"exact_prefix": True},
                "future_label_audit": {
                    "target_present": True,
                    "all_foils_absent": True,
                    "future_text_retained": False,
                },
                "concepts": concepts,
                "task_card": task["task_card"],
                "prompts": manifest_prompts,
            }
        )
    raw = document_bytes(prompts)
    manifest = {
        "schema_version": 1,
        "kind": ANALYZE.MANIFEST_KIND,
        "analysis_version": "paired-evidence-update-development-v1",
        "protocol": {"path": "protocol.json", "sha256": protocol_sha},
        "lens_outputs_used_for_boundary_or_labels": False,
        "fixed_layer_band": protocol["fixed_layer_band"],
        "score_reduction": protocol["score_reduction"],
        "score_vocabulary": {
            "token_ids": score_ids,
            "token_text": {str(token_id): token_text[token_id] for token_id in score_ids},
            "token_ids_sha256": ANALYZE.sha256_json(score_ids),
            "token_count": len(score_ids),
            "scope": "union_of_all_declared_target_and_foil_forms_on_every_prompt",
        },
        "task_count": len(protocol["tasks"]),
        "prompt_count": len(prompts),
        "tasks": manifest_tasks,
        "prompt_bundle": {
            "sha256": ANALYZE.sha256_bytes(raw),
            "bytes": len(raw),
            "count": len(prompts),
            "serialization": "indented_sorted_key_ascii_json_with_trailing_newline",
        },
    }
    return prompts, manifest


PUBLIC_UPDATES = (
    (3.0, 0.2, 0.0, 0.0),
    (0.1, 2.5, 0.0, 0.0),
    (2.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.1, 2.0),
)
LOGIT_UPDATES = (
    (0.8, 0.2, 0.0, 0.0),
    (0.1, 0.7, 0.0, 0.0),
    (0.5, 0.0, 0.1, 0.0),
    (0.0, 0.0, 0.1, 0.8),
)


def make_report(protocol: dict, prompts: list[dict], *, label: str = "public") -> dict:
    score_ids = prompts[0]["score_token_ids"]
    token_text = {
        form["token_id"]: form["text"]
        for task in protocol["tasks"]
        for item in [task["target"], *task["foils"]]
        for form in item["forms"]
    }
    target_id_to_index = {
        task["target"]["forms"][0]["token_id"]: index
        for index, task in enumerate(protocol["tasks"])
    }

    def scored(task_index: int, state: str, *, jacobian: bool) -> dict:
        matrix = PUBLIC_UPDATES if jacobian else LOGIT_UPDATES
        records = []
        for token_id in score_ids:
            score = 0.0
            if state == "after" and token_id in target_id_to_index:
                score = matrix[task_index][target_id_to_index[token_id]]
                if label == "nf4" and jacobian:
                    score *= 0.75
            records.append({"token_id": token_id, "token": token_text[token_id], "score": score})
        return {"scored_tokens": records}

    experiments = []
    for prompt in prompts:
        task_index = int(prompt["metadata"]["task"]["id"].split("-")[-1])
        state = prompt["metadata"]["state"]
        final_position = len(prompt["token_ids"]) - 1
        logits_max = 0.01
        logits_rms = 0.001
        if task_index == 1:
            logits_max, logits_rms = 0.1, 0.015
        if task_index == 3 and state == "after":
            logits_max, logits_rms = 0.2, 0.03
        recorded_logits_ok = logits_max <= 0.0625 and logits_rms <= 0.01
        final_readout = [{"token_ids": [999, 998, 997, 996, 995], "target_token_id": 999, "target_rank": 1}]
        experiments.append(
            {
                "id": prompt["id"],
                "prompt": prompt["text"],
                "prompt_token_ids": prompt["token_ids"],
                "metadata": prompt["metadata"],
                "positions_requested": [-1],
                "positions_resolved": [final_position],
                "capture_positions_resolved": [final_position],
                "final_validation_position": final_position,
                "generated_token_id": 999,
                "final_layer_top1_matches_greedy": True,
                "final_model_readout": final_readout,
                "captured_final_model_readout": copy.deepcopy(final_readout),
                "final_norm_reconstruction": {
                    "max_abs_error": 0.01,
                    "rms_error": 0.001,
                    "max_abs_tolerance": 0.125,
                    "rms_tolerance": 0.006,
                    "within_tolerance": True,
                },
                "final_logits_reconstruction": {
                    "max_abs_error": logits_max,
                    "rms_error": logits_rms,
                    "max_abs_tolerance": 0.0625,
                    "rms_tolerance": 0.01,
                    "top_k_prefix": 5,
                    "top_k_prefix_token_ids_match": True,
                    "within_tolerance": recorded_logits_ok,
                },
                "residual_capture_manifest": {
                    "sha256": f"{task_index * 2 + (state == 'after') + 100:064x}",
                    "tensor_count": 64,
                    "token_positions": [final_position],
                },
                "scored_vocabulary": {
                    "token_ids": score_ids,
                    "tokens": [token_text[token_id] for token_id in score_ids],
                },
                "layers": [
                    {
                        "layer": layer,
                        "positions": [
                            {
                                "capture_index": 0,
                                "token_position": final_position,
                                "jacobian_lens": scored(task_index, state, jacobian=True),
                                "logit_lens": scored(task_index, state, jacobian=False),
                            }
                        ],
                    }
                    for layer in range(24, 48)
                ],
            }
        )
    lens_key = ANALYZE.REPORT_LENS_KEYS[label]
    return {
        "schema_version": 3,
        "status": "failed" if label == "public" else "passed",
        "score_encoding": "unrounded-float32",
        "lens": {
            "sha256": protocol["pins"]["lenses"][lens_key]["sha256"],
            **{
                key: protocol["pins"]["lenses"][lens_key][key]
                for key in ("repo_id", "revision")
                if key in protocol["pins"]["lenses"][lens_key]
            },
        },
        "model": protocol["pins"]["model"],
        "runtime": {
            "runtime": "synthetic",
            "model_load_seconds": 1.0,
            "mtp_enabled": False,
            "enforce_eager": True,
            "language_model_only": True,
            "transport_dtype": "torch.float32",
            "readout_dtype": "torch.bfloat16",
        },
        "assertions": {
            "lens_hash_matches": True,
            "lens_metadata_matches": True,
            "model_architecture_matches": True,
        },
        "scored_vocabulary": {
            "union_token_ids": score_ids,
            "union_tokens": [token_text[token_id] for token_id in score_ids],
        },
        "experiments": experiments,
    }


def fixture() -> tuple[dict, str, list[dict], dict, dict]:
    protocol = make_protocol()
    protocol_sha = ANALYZE.sha256_bytes(document_bytes(protocol))
    prompts, manifest = make_prompts_and_manifest(protocol, protocol_sha)
    report = make_report(protocol, prompts)
    return protocol, protocol_sha, prompts, manifest, report


class ContextualEvidenceAnalysisTests(unittest.TestCase):
    def test_scores_ranks_primary_gates_and_context_swap(self) -> None:
        protocol, protocol_sha, prompts, manifest, report = fixture()
        analysis, cards = ANALYZE.analyze(
            protocol,
            manifest,
            prompts,
            report,
            protocol_sha256=protocol_sha,
            manifest_sha256=ANALYZE.sha256_json(manifest),
        )
        public = analysis["task_rows"]["public_jacobian"]
        self.assertAlmostEqual(public[0]["target_vs_foils_update"], 3.0)
        self.assertEqual(public[0]["own_target_rank_by_update"], 1.0)
        self.assertGreater(public[0]["own_minus_other_mean_context_selectivity"], 0.0)
        self.assertEqual(public[0]["same_repo_sibling_count"], 1)
        self.assertGreater(public[2]["own_target_rank_by_update"], 1.0)
        self.assertLess(public[2]["own_minus_other_mean_context_selectivity"], 0.0)

        stable = analysis["profiles"]["primary_stable"]
        strict = analysis["profiles"]["legacy_strict"]
        self.assertEqual(stable["methods"]["public_jacobian"]["support"]["task_count"], 3)
        self.assertEqual(strict["methods"]["public_jacobian"]["support"]["task_count"], 2)
        self.assertTrue(stable["public_usefulness_decision"]["passed"])
        self.assertEqual(
            stable["public_usefulness_decision"]["classification"],
            "frozen_directional_point_rule_pass",
        )
        self.assertFalse(strict["public_usefulness_decision"]["passed"])
        self.assertFalse(stable["copy_retrieval_diagnostic"]["passed"])
        self.assertEqual(analysis["cards_summary"], {"supported_lens_why_count": 2, "withheld_lens_why_count": 2})
        self.assertEqual(cards["cards"][0]["fields"]["WHY"]["observed"]["status"], "not_an_observed_card_field")
        self.assertEqual(cards["cards"][2]["lens_why_guard"]["status"], "withheld")
        self.assertEqual(
            analysis["inputs"]["reports"]["public"],
            {
                "sha256": ANALYZE.sha256_json(report),
                "basis": "canonical_sorted_compact_ascii_json_value",
            },
        )

    def test_copy_controls_detect_own_and_wrong_context(self) -> None:
        protocol, protocol_sha, prompts, manifest, report = fixture()
        analysis, _ = ANALYZE.analyze(
            protocol,
            manifest,
            prompts,
            report,
            protocol_sha256=protocol_sha,
            manifest_sha256=ANALYZE.sha256_json(manifest),
        )
        rows = analysis["copy_baselines"]["task_rows"]
        self.assertTrue(rows[0]["copy_frequency"]["own_target_top1_by_update"])
        self.assertGreater(rows[0]["copy_frequency"]["target_vs_foils_update"], 0.0)
        self.assertFalse(rows[2]["copy_frequency"]["own_target_top1_by_update"])
        self.assertLess(rows[2]["copy_frequency"]["own_minus_other_mean_context_selectivity"], 0.0)

    def test_rejects_incomplete_scored_ids_and_protocol_hash_mismatch(self) -> None:
        protocol, protocol_sha, prompts, manifest, report = fixture()
        broken = copy.deepcopy(report)
        broken["experiments"][0]["layers"][0]["positions"][0]["jacobian_lens"]["scored_tokens"].pop()
        with self.assertRaisesRegex(ValueError, "scored IDs are incomplete"):
            ANALYZE.analyze(
                protocol,
                manifest,
                prompts,
                broken,
                protocol_sha256=protocol_sha,
                manifest_sha256=ANALYZE.sha256_json(manifest),
            )
        wrong_manifest = copy.deepcopy(manifest)
        wrong_manifest["protocol"]["sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "manifest protocol hash mismatch"):
            ANALYZE.analyze(
                protocol,
                wrong_manifest,
                prompts,
                report,
                protocol_sha256=protocol_sha,
                manifest_sha256=ANALYZE.sha256_json(wrong_manifest),
            )

    def test_rejects_cross_report_residual_mismatch(self) -> None:
        protocol, protocol_sha, prompts, manifest, report = fixture()
        nf4 = make_report(protocol, prompts, label="nf4")
        valid_analysis, _ = ANALYZE.analyze(
            protocol,
            manifest,
            prompts,
            report,
            protocol_sha256=protocol_sha,
            manifest_sha256=ANALYZE.sha256_json(manifest),
            nf4_report_value=nf4,
        )
        self.assertIn("nf4_jacobian", valid_analysis["task_rows"])
        self.assertIn(
            "public_jacobian_minus_nf4_jacobian",
            valid_analysis["profiles"]["primary_stable"]["paired_comparisons"],
        )
        nf4["experiments"][1]["residual_capture_manifest"]["sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "residual_capture_manifest differs"):
            ANALYZE.analyze(
                protocol,
                manifest,
                prompts,
                report,
                protocol_sha256=protocol_sha,
                manifest_sha256=ANALYZE.sha256_json(manifest),
                nf4_report_value=nf4,
            )


if __name__ == "__main__":
    unittest.main()
