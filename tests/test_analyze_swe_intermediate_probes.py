#!/usr/bin/env python3
"""Focused tests for the frozen SWE intermediate-concept analysis."""

from __future__ import annotations

import copy
import hashlib
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


MATERIALIZER = load_module(
    "materialize_swe_intermediate_probes_for_analysis_test",
    ROOT / "scripts" / "materialize_swe_intermediate_probes.py",
)
MODULE = load_module(
    "analyze_swe_intermediate_probes",
    ROOT / "scripts" / "analyze_swe_intermediate_probes.py",
)

VOCABULARY_SIZE = 248_320
LAYERS = tuple(range(16, 48))
PASS_K = (1, 5, 10, 50, 100, 1000)
CATEGORIES = {
    "task_explicit_baseline",
    "exact_pre_identifier_state",
    "post_tool_boundary_retention",
    "teacher_forced_lexical_control",
}
JACOBIAN_MINIMA = ([5, 1], [2], [6], [20], [60], [200], [1000], [VOCABULARY_SIZE])
LOGIT_MINIMA = ([10, 100], [1], [5], [10], [50], [100], [1000], [VOCABULARY_SIZE])


def materialized_sha256(value: object) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "ascii"
    )
    return hashlib.sha256(raw).hexdigest()


def category_fields(index: int) -> tuple[str, str]:
    if index < 2:
        return "localization", "task_explicit"
    if index < 4:
        return "identifier_correction", "tool_outcome_explicit"
    if index < 6:
        return "patch", "tool_outcome_explicit"
    return "final_summary", "teacher_forced_explicit_positive_control"


def intermediate(item_index: int, concept_index: int) -> dict[str, object]:
    base = 1000 + item_index * 10 + concept_index * 2
    key = f"item_{item_index}_concept_{concept_index}"
    return {
        "key": key,
        "forms": [
            {"text": f" {key}_a", "token_id": base},
            {"text": f" {key}_b", "token_id": base + 1},
        ],
    }


def item(index: int) -> dict[str, object]:
    event_family, leakage_class = category_fields(index)
    intermediates = [intermediate(index, concept) for concept in range(2 if index == 0 else 1)]
    return {
        "id": f"item-{index}",
        "event_family": event_family,
        "request_index": index + 1,
        "offset": 32 + index if event_family == "identifier_correction" else 0,
        "state": f"state-{index}",
        "rationale": f"synthetic rationale {index}",
        "leakage_class": leakage_class,
        "request_sha256": f"{index + 1:x}" * 64,
        "evidence": [
            {
                "kind": "tool_result",
                "content_sha256": f"{index + 8:x}" * 64,
                "supports": [record["key"] for record in intermediates],
            }
        ],
        "intermediates": intermediates,
    }


def config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "swe_verified_intermediate_concept_eval",
        "adaptation": {
            "status": "exploratory_one_task_adaptation",
            "lens_outputs_used_for_selection": False,
        },
        "model": {
            "repo_id": MATERIALIZER.MODEL_REPO,
            "revision": MATERIALIZER.MODEL_REVISION,
            "tokenizer_json_sha256": MATERIALIZER.TOKENIZER_JSON_SHA256,
        },
        "task": {"instance_id": "synthetic__task-1"},
        "source": {
            "trajectory_bundle_sha256": "a" * 64,
            "trajectory_prompt_count": MATERIALIZER.EXPECTED_TRAJECTORY_COUNT,
            "trace_sha256": "b" * 64,
            "dataset_sha256": "d" * 64,
            "prompt_provenance_id": "e" * 64,
        },
        "middle_band": {
            "layers": list(LAYERS),
            "fixed_before_scoring": True,
        },
        "metric": {
            "name": "intermediate_pass_at_k",
            "accepted_target_token_scored": False,
            "pass_at_k": list(PASS_K),
        },
        "items": [item(index) for index in range(8)],
    }


def trajectory_prompt(probe_item: dict[str, object]) -> dict[str, object]:
    request = probe_item["request_index"]
    offset = probe_item["offset"]
    target = 200_000 + request
    return {
        "id": f"trajectory-{request}-{offset}",
        "token_ids": [700 + request, 800 + request, 900 + request],
        "target_token_id": target,
        "metadata": {
            "kind": "certified_swe_teacher_forced_trajectory",
            "provenance_id": "e" * 64,
            "request_index": request,
            "source_hashes": {
                "request_sha256": probe_item["request_sha256"],
                "trace_sha256": "b" * 64,
                "tokenizer_json_sha256": MATERIALIZER.TOKENIZER_JSON_SHA256,
            },
            "trajectory": {
                "request_index": request,
                "offset": offset,
                "region": "reasoning",
                "events": ["synthetic_event"],
                "target_token_id": target,
            },
        },
    }


def probe_artifacts() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    value = config()
    selected = [trajectory_prompt(probe_item) for probe_item in value["items"]]
    fillers = []
    for offset in range(MATERIALIZER.EXPECTED_TRAJECTORY_COUNT - len(selected)):
        filler = {
            "id": f"filler-{offset}",
            "token_ids": [9, offset, 999],
            "target_token_id": 210_000 + offset,
            "metadata": {
                "provenance_id": "e" * 64,
                "request_index": 9,
                "source_hashes": {
                    "request_sha256": "9" * 64,
                    "trace_sha256": "b" * 64,
                    "tokenizer_json_sha256": MATERIALIZER.TOKENIZER_JSON_SHA256,
                },
                "trajectory": {
                    "request_index": 9,
                    "offset": offset,
                    "target_token_id": 210_000 + offset,
                },
            },
        }
        fillers.append(filler)
    prompts, summary = MATERIALIZER.build_probe_bundle(
        selected + fillers,
        value,
        config_sha256="c" * 64,
        trajectory_sha256="a" * 64,
    )
    summary["output_path"] = "/synthetic/prompts.json"
    summary["output_sha256"] = materialized_sha256(prompts)
    return value, summary, prompts


def vocabulary(config_value: dict[str, object]) -> dict[int, str]:
    return {
        form["token_id"]: form["text"]
        for probe_item in config_value["items"]
        for concept in probe_item["intermediates"]
        for form in concept["forms"]
    }


def rank_map(
    config_value: dict[str, object], item_index: int, layer: int, method: str
) -> dict[int, int]:
    values = {token_id: VOCABULARY_SIZE for token_id in vocabulary(config_value)}
    minima = JACOBIAN_MINIMA if method == "jacobian" else LOGIT_MINIMA
    for concept_index, concept in enumerate(config_value["items"][item_index]["intermediates"]):
        winning_layer = 47 if item_index == 0 and concept_index == 1 else 16 + (item_index * 3 + concept_index * 5) % len(LAYERS)
        if layer == winning_layer:
            values[concept["forms"][1]["token_id"]] = minima[item_index][concept_index]
    return values


def readout(values: dict[int, str], ranks: dict[int, int]) -> dict[str, object]:
    return {
        "scored_tokens": [
            {
                "token_id": token_id,
                "token": values[token_id],
                "rank": ranks[token_id],
                "score": 0.0,
                "logprob": -12.0,
            }
            for token_id in sorted(values)
        ]
    }


def lens(kind: str) -> dict[str, object]:
    common = {
        "d_model": 5120,
        "source_layers": list(range(63)),
        "tensor_shape": [5120, 5120],
    }
    if kind == "public":
        return {
            **common,
            "repo_id": "neuronpedia/jacobian-lens",
            "revision": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
            "sha256": "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1",
        }
    return {
        **common,
        "kind": "native_nvfp4_ste_fit",
        "sha256": "82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057",
        "state_sha256": "f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6",
        "provenance_sha256": "289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601",
        "fit_model": MATERIALIZER.MODEL_REPO,
        "fit_model_revision": MATERIALIZER.MODEL_REVISION,
    }


def report(
    config_value: dict[str, object], prompts: list[dict[str, object]], *, kind: str
) -> dict[str, object]:
    values = vocabulary(config_value)
    token_ids = sorted(values)
    experiments = []
    for item_index, prompt in enumerate(prompts):
        final_position = len(prompt["token_ids"]) - 1
        layers = []
        for layer in LAYERS:
            layers.append(
                {
                    "layer": layer,
                    "layer_type": "linear_attention",
                    "positions": [
                        {
                            "capture_index": 0,
                            "token_position": final_position,
                            "jacobian_lens": readout(
                                values,
                                rank_map(config_value, item_index, layer, "jacobian"),
                            ),
                            "logit_lens": readout(
                                values,
                                rank_map(config_value, item_index, layer, "logit"),
                            ),
                        }
                    ],
                }
            )
        target = prompt["target_token_id"]
        experiments.append(
            {
                "id": prompt["id"],
                "prompt": f"synthetic prompt {item_index}",
                "prompt_token_ids": copy.deepcopy(prompt["token_ids"]),
                "target_token_id_override": target,
                "generated_token_id": target,
                "positions_requested": [-1],
                "positions_resolved": [final_position],
                "capture_positions_resolved": [final_position],
                "final_validation_position": final_position,
                "metadata": copy.deepcopy(prompt["metadata"]),
                "scored_vocabulary": {
                    "token_ids": token_ids,
                    "tokens": [values[token_id] for token_id in token_ids],
                },
                "layers": layers,
                "residual_capture_manifest": {
                    "sha256": f"{item_index + 1:064x}",
                    "tensor_count": 64,
                    "token_positions": [final_position],
                },
                "final_layer_top1_matches_greedy": True,
                "final_norm_reconstruction": {"within_tolerance": True},
                "final_logits_reconstruction": {
                    "within_tolerance": item_index % 2 == 0,
                    "top_k_prefix_token_ids_match": True,
                },
            }
        )
    return {
        "schema_version": 3,
        "score_encoding": "unrounded-float32",
        "status": "failed",
        "model": {
            "repo_id": MATERIALIZER.MODEL_REPO,
            "revision": MATERIALIZER.MODEL_REVISION,
            "config_sha256": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
            "index_sha256": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
        },
        "lens": lens(kind),
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
            "all_final_adapter_reconstructions_within_tolerance": False,
        },
        "scored_vocabulary": {
            "scope": "global",
            "token_ids": token_ids,
            "tokens": [values[token_id] for token_id in token_ids],
            "union_token_ids": token_ids,
            "union_tokens": [values[token_id] for token_id in token_ids],
        },
        "experiments": experiments,
    }


def fixture():
    config_value, summary, prompts = probe_artifacts()
    public = report(config_value, prompts, kind="public")
    native = report(config_value, prompts, kind="native")
    return config_value, summary, prompts, public, native


def expected_auc(minima: tuple[list[int], ...]) -> float:
    denominator = math.log(VOCABULARY_SIZE)
    item_scores = []
    for item_minima in minima:
        item_scores.append(
            sum(math.log(VOCABULARY_SIZE / rank) / denominator for rank in item_minima)
            / len(item_minima)
        )
    return sum(item_scores) / len(item_scores)


class AnalyzeSweIntermediateProbesTest(unittest.TestCase):
    def test_exact_form_layer_minima_item_macro_auc_and_categories(self) -> None:
        config_value, summary, prompts, public, native = fixture()
        result = MODULE.analyze(config_value, summary, prompts, public, native)
        repeated = MODULE.analyze(config_value, summary, prompts, public, native)
        self.assertEqual(result, repeated)

        overall = result["public"]["overall"]
        self.assertEqual(
            overall["jacobian_lens"]["pass_at_k"],
            {
                "1": 0.0625,
                "5": 0.25,
                "10": 0.375,
                "50": 0.5,
                "100": 0.625,
                "1000": 0.875,
            },
        )
        self.assertEqual(
            overall["logit_lens"]["pass_at_k"],
            {
                "1": 0.125,
                "5": 0.25,
                "10": 0.4375,
                "50": 0.5625,
                "100": 0.75,
                "1000": 0.875,
            },
        )
        self.assertAlmostEqual(
            overall["jacobian_lens"]["normalized_log_rank_auc"],
            expected_auc(JACOBIAN_MINIMA),
        )
        self.assertAlmostEqual(
            overall["logit_lens"]["normalized_log_rank_auc"],
            expected_auc(LOGIT_MINIMA),
        )
        bootstrap = overall["paired_item_bootstrap"]
        for metric in ("normalized_log_rank_auc_gain", "pass_at_10_gain"):
            interval = bootstrap[metric]["confidence_interval"]
            self.assertIn("deterministic", interval["method"])
            self.assertIn("paired item", interval["method"])
            self.assertEqual(interval["confidence_level"], 0.95)
            self.assertIsInstance(interval["seed"], int)
            self.assertEqual(interval["samples"], MODULE.BOOTSTRAP_SAMPLES)
            self.assertLessEqual(interval["lower"], interval["upper"])
        self.assertEqual(set(result["public"]["categories"]), CATEGORIES)
        self.assertEqual(
            {key: value["item_count"] for key, value in result["public"]["categories"].items()},
            {key: 2 for key in CATEGORIES},
        )
        self.assertFalse(result["evaluation"]["claims_gate_preregistered"])
        self.assertIsNone(result["evaluation"]["claims_gate"])

    def test_rejects_missing_exact_rank(self) -> None:
        config_value, summary, prompts, public, native = fixture()
        del public["experiments"][0]["layers"][0]["positions"][0]["jacobian_lens"]["scored_tokens"][0]["rank"]
        with self.assertRaisesRegex(ValueError, "rank"):
            MODULE.analyze(config_value, summary, prompts, public, native)

    def test_rejects_incomplete_scored_vocabulary(self) -> None:
        config_value, summary, prompts, public, native = fixture()
        public["scored_vocabulary"]["token_ids"].pop()
        with self.assertRaisesRegex(ValueError, "vocab"):
            MODULE.analyze(config_value, summary, prompts, public, native)

    def test_rejects_missing_prompt_bundle_hash(self) -> None:
        config_value, summary, prompts, public, native = fixture()
        del summary["output_sha256"]
        with self.assertRaisesRegex(ValueError, "hash|SHA|sha256"):
            MODULE.analyze(config_value, summary, prompts, public, native)

    def test_rejects_paired_residual_mismatch(self) -> None:
        config_value, summary, prompts, public, native = fixture()
        native["experiments"][3]["residual_capture_manifest"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "residual_capture_manifest|residual"):
            MODULE.analyze(config_value, summary, prompts, public, native)


if __name__ == "__main__":
    unittest.main()
