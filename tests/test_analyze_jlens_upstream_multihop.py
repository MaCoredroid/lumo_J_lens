#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_jlens_upstream_multihop",
    ROOT / "scripts/analyze_jlens_upstream_multihop.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _metadata(index: int) -> dict:
    if index == 0:
        intermediates = [
            {
                "text": "Alpha",
                "eligible_forms": [
                    {"form": "bare", "text": "Alpha", "token_id": 10},
                    {
                        "form": "leading_space",
                        "text": " Alpha",
                        "token_id": 11,
                    },
                ],
                "excluded_forms": [],
                "scorable": True,
            }
        ]
        name = "scorable"
    else:
        intermediates = [
            {
                "text": "12",
                "eligible_forms": [],
                "excluded_forms": [
                    {
                        "form": "bare",
                        "text": "12",
                        "token_ids": [1, 2],
                        "reason": "not_exactly_one_token",
                    },
                    {
                        "form": "leading_space",
                        "text": " 12",
                        "token_ids": [3, 1, 2],
                        "reason": "not_exactly_one_token",
                    },
                ],
                "scorable": False,
            }
        ]
        name = "excluded"
    return {
        "kind": "anthropic_jlens_multihop_qwen36_control",
        "upstream": {
            "commit": MODULE.UPSTREAM_COMMIT,
            "source_sha256": MODULE.UPSTREAM_SOURCE_SHA256,
            "item_index": index,
            "name": name,
            "target": f"target-{index}",
            "intermediates": intermediates,
        },
        "tokenizer": {
            "repo_id": MODULE.MODEL_REPO,
            "revision": MODULE.MODEL_REVISION,
            "tokenizer_json_sha256": MODULE.TOKENIZER_JSON_SHA256,
        },
    }


def _prompt_bundle(metadata: list[dict]) -> list[dict]:
    first = {
        "id": "upstream-multihop-000-scorable",
        "text": "Prompt zero ",
        "token_ids": [101, 102],
        "score_token_ids": [10, 11],
        "metadata": metadata[0],
    }
    second = {
        "id": "upstream-multihop-001-excluded",
        "text": "Prompt one ",
        "token_ids": [201, 202],
        "metadata": metadata[1],
    }
    return [first, second]


def _scored_records(jacobian: bool) -> list[dict]:
    ranks = (1, 2) if jacobian else (10, 11)
    return [
        {
            "token_id": token_id,
            "token": token,
            "rank": rank,
            "score": 0.0,
            "logprob": -1.0,
        }
        for token_id, token, rank in zip(
            (10, 11), ("Alpha", " Alpha"), ranks, strict=True
        )
    ]


def _report(metadata: list[dict], *, native: bool = False) -> dict:
    experiments = []
    prompt_bundle = _prompt_bundle(metadata)
    for index, prompt in enumerate(prompt_bundle):
        scored_ids = [10, 11] if index == 0 else []
        layers = []
        for layer in MODULE.ALL_LAYERS:
            if scored_ids:
                jacobian = {"scored_tokens": _scored_records(True)}
                logit = {"scored_tokens": _scored_records(False)}
            else:
                # The runner omits scored_tokens when a prompt has no eligible forms.
                jacobian = {}
                logit = {}
            layers.append(
                {
                    "layer": layer,
                    "layer_type": "full_attention",
                    "positions": [
                        {
                            "token_position": len(prompt["token_ids"]) - 1,
                            "jacobian_lens": jacobian,
                            "logit_lens": logit,
                        }
                    ],
                }
            )
        experiments.append(
            {
                "id": prompt["id"],
                "prompt": prompt["text"],
                "prompt_token_ids": prompt["token_ids"],
                "positions_requested": [-1],
                "positions_resolved": [len(prompt["token_ids"]) - 1],
                "metadata": prompt["metadata"],
                "scored_vocabulary": {
                    "token_ids": scored_ids,
                    "tokens": ["Alpha", " Alpha"] if scored_ids else [],
                },
                "residual_capture_manifest": {
                    "sha256": f"{index + 1:064x}",
                    "layers": list(MODULE.ALL_LAYERS),
                },
                "final_layer_top1_matches_greedy": True,
                "final_norm_reconstruction": {"within_tolerance": True},
                "final_logits_reconstruction": {
                    "within_tolerance": True,
                    "top_k_prefix_token_ids_match": True,
                },
                "layers": layers,
            }
        )
    lens = {
        "d_model": 5120,
        "source_layers": list(MODULE.ALL_LAYERS),
        "tensor_shape": [5120, 5120],
    }
    if native:
        lens.update({
            "kind": "native_nvfp4_ste_fit",
            "sha256": MODULE.NATIVE_LENS_SHA256,
            "state_sha256": MODULE.NATIVE_STATE_SHA256,
            "provenance_sha256": MODULE.NATIVE_PROVENANCE_SHA256,
            "fit_model": MODULE.MODEL_REPO,
            "fit_model_revision": MODULE.MODEL_REVISION,
        })
    else:
        lens.update({
            "repo_id": MODULE.PUBLIC_LENS_REPO,
            "revision": MODULE.PUBLIC_LENS_REVISION,
            "sha256": MODULE.PUBLIC_LENS_SHA256,
        })
    return {
        "schema_version": 3,
        "score_encoding": "unrounded-float32",
        "lens": lens,
        "model": {
            "repo_id": MODULE.MODEL_REPO,
            "revision": MODULE.MODEL_REVISION,
        },
        "runtime": {
            "mtp_enabled": False,
            "enforce_eager": True,
            "language_model_only": True,
        },
        "assertions": {
            "lens_hash_matches": True,
            "lens_metadata_matches": True,
            "model_architecture_matches": True,
            "all_final_layer_top1_match_greedy": True,
        },
        "scored_vocabulary": {
            "token_ids": [],
            "tokens": [],
            "scope": "global_plus_per_experiment",
            "union_token_ids": [10, 11],
            "union_tokens": ["Alpha", " Alpha"],
        },
        "experiments": experiments,
    }


def _manifest(metadata: list[dict]) -> dict:
    prompts = _prompt_bundle(metadata)
    return {
        "schema_version": 1,
        "kind": "anthropic_jlens_multihop_qwen36_materialization",
        "upstream": {
            "repository": "anthropics/jacobian-lens",
            "relative_path": "data/evaluations/lens-eval-multihop.json",
            "commit": MODULE.UPSTREAM_COMMIT,
            "source_sha256": MODULE.UPSTREAM_SOURCE_SHA256,
        },
        "model": {
            "repo_id": MODULE.MODEL_REPO,
            "revision": MODULE.MODEL_REVISION,
            "tokenizer_json_sha256": MODULE.TOKENIZER_JSON_SHA256,
            "tokenizer_vocabulary_size": MODULE.TOKENIZER_VOCABULARY_SIZE,
            "config_sha256": MODULE.MODEL_CONFIG_SHA256,
            "logit_vocabulary_size": MODULE.LOGIT_VOCABULARY_SIZE,
        },
        "metric_contract": {
            "fixed_middle_layers": list(MODULE.FIXED_MIDDLE_LAYERS),
            "secondary_all_layers": list(MODULE.ALL_LAYERS),
            "unscorable_intermediate_policy": (
                "count_as_miss_to_preserve_all_upstream_item/intermediate_denominators"
            ),
        },
        "coverage": {
            "item_count": 2,
            "intermediate_occurrence_count": 2,
            "scorable_intermediate_occurrence_count": 1,
            "excluded_intermediate_occurrence_count": 1,
        },
        "scored_vocabulary": {
            "token_ids": [10, 11],
            "tokens": ["Alpha", " Alpha"],
        },
        "outputs": {
            "source_copy": {"sha256": MODULE.UPSTREAM_SOURCE_SHA256},
            "prompts": {"sha256": MODULE.materialized_json_sha256(prompts)}
        },
    }


class AnalyzeUpstreamMultihopTest(unittest.TestCase):
    def setUp(self):
        self.metadata = [_metadata(0), _metadata(1)]
        self.manifest = _manifest(self.metadata)
        self.report = _report(self.metadata)

    def test_excluded_intermediate_is_a_miss_and_auc_favors_jacobian(self):
        result = MODULE.analyze(
            self.manifest, self.report, bootstrap_seed=7, bootstrap_samples=200
        )
        primary = result["primary"]["primary_fixed_middle_band"]
        jacobian = primary["jacobian_lens"]
        logit = primary["logit_lens"]

        self.assertEqual(jacobian["pass_at_k"]["1"], 0.5)
        self.assertEqual(logit["pass_at_k"]["1"], 0.0)
        self.assertEqual(jacobian["pass_at_k"]["10"], 0.5)
        self.assertEqual(logit["pass_at_k"]["10"], 0.5)
        self.assertEqual(
            jacobian["items"][1]["intermediates"][0]["minimum_rank"], None
        )
        self.assertAlmostEqual(jacobian["normalized_log_rank_auc"], 0.5)
        vocabulary_size = MODULE.LOGIT_VOCABULARY_SIZE
        expected_logit_auc = 0.5 * (
            (math.log(vocabulary_size) - math.log(10))
            / math.log(vocabulary_size)
        )
        self.assertAlmostEqual(logit["normalized_log_rank_auc"], expected_logit_auc)
        self.assertGreater(
            primary["jacobian_minus_logit"]["normalized_log_rank_auc"], 0.0
        )
        bootstrap = primary["paired_item_bootstrap"]
        self.assertEqual(bootstrap["seed"], 7)
        self.assertEqual(bootstrap["samples"], 200)
        self.assertAlmostEqual(
            bootstrap["normalized_log_rank_auc_gain"]["estimate"],
            primary["jacobian_minus_logit"]["normalized_log_rank_auc"],
        )
        self.assertEqual(
            bootstrap["normalized_log_rank_auc_gain"]["positive_item_count"], 1
        )
        self.assertEqual(
            bootstrap["normalized_log_rank_auc_gain"]["tie_item_count"], 1
        )

    def test_missing_exact_rank_is_rejected(self):
        del self.report["experiments"][0]["layers"][24]["positions"][0][
            "jacobian_lens"
        ]["scored_tokens"][0]["rank"]
        with self.assertRaisesRegex(ValueError, "exact rank invalid"):
            MODULE.analyze(self.manifest, self.report)

    def test_top_level_union_mismatch_is_rejected(self):
        self.report["scored_vocabulary"]["union_token_ids"] = [10]
        self.report["scored_vocabulary"]["union_tokens"] = ["Alpha"]
        with self.assertRaisesRegex(ValueError, "scored vocabulary.*mismatch"):
            MODULE.analyze(self.manifest, self.report)

    def test_pairing_rejects_residual_capture_mismatch(self):
        native = _report(self.metadata, native=True)
        native["experiments"][1]["residual_capture_manifest"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "residual_capture_manifest"):
            MODULE.analyze(self.manifest, self.report, native)

    def test_pairing_rejects_logit_lens_field_mismatch(self):
        native = _report(self.metadata, native=True)
        native["experiments"][0]["layers"][0]["positions"][0]["logit_lens"][
            "scored_tokens"
        ][0]["score"] = 1.0
        with self.assertRaisesRegex(ValueError, "logit_lens_sha256"):
            MODULE.analyze(self.manifest, self.report, native)


if __name__ == "__main__":
    unittest.main()
