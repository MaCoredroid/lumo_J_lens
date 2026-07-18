#!/usr/bin/env python3
"""Validate and summarize frozen SWE intermediate-concept J-lens probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from materialize_swe_intermediate_probes import (  # noqa: E402
    EXPECTED_LAYERS,
    MODEL_REPO,
    MODEL_REVISION,
    TOKENIZER_JSON_SHA256,
    validate_config,
)


MODEL_CONFIG_SHA256 = (
    "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338"
)
MODEL_INDEX_SHA256 = (
    "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2"
)
LOGIT_VOCABULARY_SIZE = 248_320
PUBLIC_LENS_REPO = "neuronpedia/jacobian-lens"
PUBLIC_LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
PUBLIC_LENS_SHA256 = (
    "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"
)
NATIVE_LENS_SHA256 = (
    "82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057"
)
NATIVE_STATE_SHA256 = (
    "f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6"
)
NATIVE_PROVENANCE_SHA256 = (
    "289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601"
)
ALL_SOURCE_LAYERS = tuple(range(63))
BOOTSTRAP_SEED = 36_027
BOOTSTRAP_SAMPLES = 20_000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return value


def finite(value: Any, label: str) -> float:
    require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def valid_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def materialized_json_sha256(value: Any) -> str:
    raw = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    return sha256_bytes(raw)


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    require(bool(ordered), "quantile input must not be empty")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def derived_seed(label: str, seed: int) -> int:
    return seed + int(hashlib.sha256(label.encode("ascii")).hexdigest()[:8], 16)


def validate_bundle(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    *,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    items, token_ids, pass_at_k = validate_config(config)
    require(
        summary.get("schema_version") == 1
        and summary.get("kind")
        == "swe_verified_intermediate_concept_probe_bundle",
        "prompt summary identity mismatch",
    )
    summary_config_sha = valid_sha256(
        summary.get("config_sha256"), "prompt summary config hash"
    )
    if config_sha256 is not None:
        require(
            summary_config_sha == config_sha256,
            "prompt summary config SHA-256 does not match the config file",
        )
    trajectory_sha = valid_sha256(
        summary.get("trajectory_bundle_sha256"),
        "prompt summary trajectory hash",
    )
    require(
        trajectory_sha
        == mapping(config.get("source"), "config.source").get(
            "trajectory_bundle_sha256"
        ),
        "prompt summary trajectory hash mismatch",
    )
    require(
        summary.get("adaptation_status") == "exploratory_one_task_adaptation"
        and summary.get("lens_outputs_used_for_selection") is False,
        "prompt summary adaptation contract mismatch",
    )
    require(
        summary.get("middle_band_layers") == list(EXPECTED_LAYERS),
        "prompt summary layer band mismatch",
    )
    require(
        summary.get("pass_at_k") == list(pass_at_k),
        "prompt summary pass@k mismatch",
    )
    require(
        summary.get("item_count") == len(items)
        and len(prompts) == len(items),
        "prompt summary item count mismatch",
    )
    expected_ids = [item["id"] for item in items]
    require(summary.get("item_ids") == expected_ids, "prompt summary item IDs mismatch")
    require(
        summary.get("intermediate_count")
        == sum(len(item["intermediates"]) for item in items),
        "prompt summary intermediate count mismatch",
    )
    require(
        summary.get("scored_token_ids") == list(token_ids),
        "prompt summary scored vocabulary mismatch",
    )
    output_sha = valid_sha256(
        summary.get("output_sha256"), "prompt bundle output hash"
    )
    require(
        materialized_json_sha256(prompts) == output_sha,
        "prompt bundle SHA-256 does not match the summary output hash",
    )

    token_text: dict[int, str] = {}
    normalized_prompts: list[dict[str, Any]] = []
    for index, (item, prompt_value) in enumerate(zip(items, prompts, strict=True)):
        prompt = mapping(prompt_value, f"prompt[{index}]")
        expected_id = f"swe-intermediate-{item['id']}"
        require(prompt.get("id") == expected_id, f"prompt {index} ID mismatch")
        prompt_token_ids = sequence(
            prompt.get("token_ids"), f"prompt {expected_id}.token_ids"
        )
        require(
            bool(prompt_token_ids)
            and all(
                isinstance(token_id, int)
                and not isinstance(token_id, bool)
                and token_id >= 0
                for token_id in prompt_token_ids
            ),
            f"prompt {expected_id} token IDs invalid",
        )
        target = prompt.get("target_token_id")
        require(
            isinstance(target, int) and not isinstance(target, bool) and target >= 0,
            f"prompt {expected_id} target invalid",
        )
        metadata = mapping(prompt.get("metadata"), f"prompt {expected_id}.metadata")
        trajectory = mapping(
            metadata.get("trajectory"), f"prompt {expected_id}.trajectory"
        )
        require(
            trajectory.get("request_index", metadata.get("request_index"))
            == item["request_index"]
            and trajectory.get("offset") == item["offset"],
            f"prompt {expected_id} request/offset mismatch",
        )
        require(
            trajectory.get("target_token_id") == target,
            f"prompt {expected_id} accepted target mismatch",
        )
        probe = mapping(
            metadata.get("intermediate_probe"),
            f"prompt {expected_id}.intermediate_probe",
        )
        for field in (
            "id",
            "event_family",
            "state",
            "rationale",
            "leakage_class",
            "evidence",
            "intermediates",
        ):
            require(
                probe.get(field) == item.get(field),
                f"prompt {expected_id} probe {field} mismatch",
            )
        require(
            probe.get("config_sha256") == summary_config_sha
            and probe.get("trajectory_bundle_sha256") == trajectory_sha,
            f"prompt {expected_id} source hash mismatch",
        )
        require(
            probe.get("middle_band_layers") == list(EXPECTED_LAYERS)
            and probe.get("lens_outputs_used_for_selection") is False,
            f"prompt {expected_id} frozen selection contract mismatch",
        )
        for intermediate in item["intermediates"]:
            for form in intermediate["forms"]:
                previous = token_text.setdefault(form["token_id"], form["text"])
                require(previous == form["text"], "configured vocabulary text mismatch")
        require(target not in token_ids, f"prompt {expected_id} target leaks into scored vocabulary")
        normalized_prompts.append(
            {
                "id": expected_id,
                "item": item,
                "token_ids": list(prompt_token_ids),
                "target_token_id": target,
                "metadata": metadata,
            }
        )

    require(sorted(token_text) == list(token_ids), "configured vocabulary coverage mismatch")
    return {
        "items": list(items),
        "prompts": normalized_prompts,
        "token_ids": list(token_ids),
        "token_text": token_text,
        "pass_at_k": list(pass_at_k),
        "config_sha256": summary_config_sha,
        "trajectory_sha256": trajectory_sha,
        "prompt_bundle_sha256": output_sha,
    }


def validate_lens(report: Mapping[str, Any], label: str) -> None:
    lens = mapping(report.get("lens"), f"{label}.lens")
    require(
        lens.get("d_model") == 5120
        and lens.get("source_layers") == list(ALL_SOURCE_LAYERS)
        and lens.get("tensor_shape") == [5120, 5120],
        f"{label} lens geometry mismatch",
    )
    if label == "public":
        require(
            lens.get("repo_id") == PUBLIC_LENS_REPO
            and lens.get("revision") == PUBLIC_LENS_REVISION
            and lens.get("sha256") == PUBLIC_LENS_SHA256
            and lens.get("n_prompts", 1000) == 1000,
            "public lens identity mismatch",
        )
    else:
        require(
            lens.get("kind") == "native_nvfp4_ste_fit"
            and lens.get("sha256") == NATIVE_LENS_SHA256
            and lens.get("state_sha256") == NATIVE_STATE_SHA256
            and lens.get("provenance_sha256") == NATIVE_PROVENANCE_SHA256
            and lens.get("fit_model") == MODEL_REPO
            and lens.get("fit_model_revision") == MODEL_REVISION
            and lens.get("n_prompts", 10) == 10,
            "native lens identity or provenance mismatch",
        )


def scored_ranks(
    value: Any,
    label: str,
    *,
    token_ids: Sequence[int],
    token_text: Mapping[int, str],
) -> dict[int, int]:
    records = sequence(
        mapping(value, label).get("scored_tokens"), f"{label}.scored_tokens"
    )
    result: dict[int, int] = {}
    for index, raw_record in enumerate(records):
        record = mapping(raw_record, f"{label}.scored_tokens[{index}]")
        token_id = record.get("token_id")
        rank = record.get("rank")
        require(
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and token_id not in result,
            f"{label} scored vocabulary token IDs invalid",
        )
        require(
            isinstance(rank, int)
            and not isinstance(rank, bool)
            and 1 <= rank <= LOGIT_VOCABULARY_SIZE,
            f"{label} exact rank invalid",
        )
        require(
            record.get("token") == token_text.get(token_id),
            f"{label} scored vocabulary token text mismatch",
        )
        finite(record.get("score"), f"{label}.token-{token_id}.score")
        finite(record.get("logprob"), f"{label}.token-{token_id}.logprob")
        result[token_id] = rank
    require(
        list(result) == list(token_ids),
        f"{label} scored vocabulary rank coverage/order mismatch",
    )
    return result


def category_for(item: Mapping[str, Any]) -> str:
    leakage = item["leakage_class"]
    if leakage == "task_explicit":
        return "task_explicit_baseline"
    if item["event_family"] == "identifier_correction" and item["offset"] > 0:
        return "exact_pre_identifier_state"
    if leakage.startswith("tool_outcome_") and item["offset"] == 0:
        return "post_tool_boundary_retention"
    if leakage == "teacher_forced_explicit_positive_control":
        return "teacher_forced_lexical_control"
    raise ValueError(f"item {item['id']} has no deterministic analysis category")


def validate_report(
    report: Mapping[str, Any], *, label: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    require(report.get("schema_version") == 3, f"{label} report schema mismatch")
    require(
        report.get("score_encoding") == "unrounded-float32",
        f"{label} score encoding mismatch",
    )
    validate_lens(report, label)
    model = mapping(report.get("model"), f"{label}.model")
    require(
        model.get("repo_id") == MODEL_REPO
        and model.get("revision") == MODEL_REVISION
        and model.get("config_sha256") == MODEL_CONFIG_SHA256
        and model.get("index_sha256") == MODEL_INDEX_SHA256,
        f"{label} pinned model mismatch",
    )
    runtime = mapping(report.get("runtime"), f"{label}.runtime")
    require(
        runtime.get("mtp_enabled") is False
        and runtime.get("enforce_eager") is True
        and runtime.get("language_model_only") is True,
        f"{label} replay runtime mismatch",
    )
    assertions = mapping(report.get("assertions"), f"{label}.assertions")
    for assertion in (
        "lens_hash_matches",
        "lens_metadata_matches",
        "model_architecture_matches",
    ):
        require(assertions.get(assertion) is True, f"{label} assertion failed: {assertion}")

    vocabulary = mapping(report.get("scored_vocabulary"), f"{label}.vocabulary")
    expected_ids = contract["token_ids"]
    expected_tokens = [contract["token_text"][token_id] for token_id in expected_ids]
    require(
        vocabulary.get("scope") == "global"
        and vocabulary.get("token_ids") == expected_ids
        and vocabulary.get("tokens") == expected_tokens,
        f"{label} global scored vocabulary mismatch",
    )
    for field in ("union_token_ids", "union_tokens"):
        if field in vocabulary:
            expected = expected_ids if field.endswith("ids") else expected_tokens
            require(vocabulary.get(field) == expected, f"{label} vocabulary {field} mismatch")

    experiments = sequence(report.get("experiments"), f"{label}.experiments")
    require(
        len(experiments) == len(contract["prompts"]),
        f"{label} experiment count mismatch",
    )
    normalized_items: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    eligibility = {
        "accepted_target_matches_greedy": 0,
        "adapter_top1_matches_greedy": 0,
        "final_top5_matches": 0,
        "final_norm_within_tolerance": 0,
        "final_logits_within_tolerance": 0,
        "strict_adapter_pass": 0,
    }
    all_top1 = True
    all_strict = True
    for index, (raw_experiment, prompt) in enumerate(
        zip(experiments, contract["prompts"], strict=True)
    ):
        experiment = mapping(raw_experiment, f"{label}.experiments[{index}]")
        identifier = prompt["id"]
        require(experiment.get("id") == identifier, f"{label} experiment ID mismatch")
        require(
            experiment.get("prompt_token_ids") == prompt["token_ids"],
            f"{label} {identifier} prompt token IDs mismatch",
        )
        require(
            experiment.get("metadata") == prompt["metadata"],
            f"{label} {identifier} prompt metadata mismatch",
        )
        target = prompt["target_token_id"]
        require(
            experiment.get("target_token_id_override") == target,
            f"{label} {identifier} accepted target mismatch",
        )
        final_position = len(prompt["token_ids"]) - 1
        require(
            experiment.get("positions_requested") == [-1]
            and experiment.get("positions_resolved") == [final_position]
            and experiment.get("capture_positions_resolved") == [final_position]
            and experiment.get("final_validation_position") == final_position,
            f"{label} {identifier} readout position mismatch",
        )
        experiment_vocabulary = mapping(
            experiment.get("scored_vocabulary"),
            f"{label} {identifier}.vocabulary",
        )
        require(
            experiment_vocabulary.get("token_ids") == expected_ids
            and experiment_vocabulary.get("tokens") == expected_tokens,
            f"{label} {identifier} scored vocabulary mismatch",
        )

        layers = sequence(experiment.get("layers"), f"{label} {identifier}.layers")
        layer_ids = [mapping(layer, "layer").get("layer") for layer in layers]
        require(
            layer_ids == list(EXPECTED_LAYERS),
            f"{label} {identifier} must contain fixed layers 16 through 47",
        )
        rank_maps: dict[str, dict[int, dict[int, int]]] = {
            "jacobian": {},
            "logit": {},
        }
        for layer_id, layer_value in zip(layer_ids, layers, strict=True):
            positions = sequence(
                mapping(layer_value, "layer").get("positions"),
                f"{label} {identifier}.layer-{layer_id}.positions",
            )
            require(len(positions) == 1, f"{label} {identifier} layer position count mismatch")
            position = mapping(positions[0], "position")
            require(
                position.get("capture_index") == 0
                and position.get("token_position") == final_position,
                f"{label} {identifier} layer {layer_id} position mismatch",
            )
            rank_maps["jacobian"][layer_id] = scored_ranks(
                position.get("jacobian_lens"),
                f"{label} {identifier}.layer-{layer_id}.jacobian",
                token_ids=expected_ids,
                token_text=contract["token_text"],
            )
            rank_maps["logit"][layer_id] = scored_ranks(
                position.get("logit_lens"),
                f"{label} {identifier}.layer-{layer_id}.logit",
                token_ids=expected_ids,
                token_text=contract["token_text"],
            )

        top1 = experiment.get("final_layer_top1_matches_greedy")
        require(isinstance(top1, bool), f"{label} {identifier} top1 flag invalid")
        final_norm = mapping(
            experiment.get("final_norm_reconstruction"),
            f"{label} {identifier}.final_norm",
        )
        final_logits = mapping(
            experiment.get("final_logits_reconstruction"),
            f"{label} {identifier}.final_logits",
        )
        norm_within = final_norm.get("within_tolerance")
        logits_within = final_logits.get("within_tolerance")
        top5 = final_logits.get("top_k_prefix_token_ids_match")
        require(
            all(isinstance(value, bool) for value in (norm_within, logits_within, top5)),
            f"{label} {identifier} adapter flags invalid",
        )
        strict = top1 and norm_within and logits_within
        accepted_matches = experiment.get("generated_token_id") == target
        eligibility["accepted_target_matches_greedy"] += int(accepted_matches)
        eligibility["adapter_top1_matches_greedy"] += int(top1)
        eligibility["final_top5_matches"] += int(top5)
        eligibility["final_norm_within_tolerance"] += int(norm_within)
        eligibility["final_logits_within_tolerance"] += int(logits_within)
        eligibility["strict_adapter_pass"] += int(strict)
        all_top1 = all_top1 and top1
        all_strict = all_strict and strict

        residual_manifest = mapping(
            experiment.get("residual_capture_manifest"),
            f"{label} {identifier}.residual_capture_manifest",
        )
        valid_sha256(
            residual_manifest.get("sha256"),
            f"{label} {identifier} residual manifest hash",
        )
        item = prompt["item"]
        normalized_items.append(
            {
                "id": item["id"],
                "event_family": item["event_family"],
                "request_index": item["request_index"],
                "offset": item["offset"],
                "state": item["state"],
                "leakage_class": item["leakage_class"],
                "category": category_for(item),
                "intermediates": item["intermediates"],
                "rank_maps": rank_maps,
            }
        )
        pair_rows.append(
            {
                "id": identifier,
                "prompt_token_ids": experiment.get("prompt_token_ids"),
                "metadata": experiment.get("metadata"),
                "target_token_id_override": experiment.get("target_token_id_override"),
                "residual_capture_manifest": dict(residual_manifest),
                "logit_rank_maps": rank_maps["logit"],
                "adapter_top1": top1,
                "final_top5": top5,
                "final_norm_within": norm_within,
                "final_logits_within": logits_within,
            }
        )

    require(
        assertions.get("all_final_layer_top1_match_greedy") is all_top1,
        f"{label} aggregate top1 assertion mismatch",
    )
    if "all_final_adapter_reconstructions_within_tolerance" in assertions:
        require(
            assertions.get("all_final_adapter_reconstructions_within_tolerance")
            is all_strict,
            f"{label} aggregate adapter assertion mismatch",
        )
    require(
        report.get("status") == ("passed" if all_top1 and all_strict else "failed"),
        f"{label} report status mismatch",
    )
    return {
        "items": normalized_items,
        "pair_rows": pair_rows,
        "numerical_eligibility": {
            "experiment_count": len(experiments),
            "counts": eligibility,
            "strict_report_status": report.get("status"),
            "interpretation": (
                "Intermediate ranks are descriptive on paired residuals. Strict "
                "adapter failures and accepted-target/greedy mismatches are retained."
            ),
        },
    }


def best_intermediate(
    item: Mapping[str, Any], method: str, intermediate: Mapping[str, Any]
) -> dict[str, Any]:
    candidates: list[tuple[int, int, int, str]] = []
    for layer in EXPECTED_LAYERS:
        for form in intermediate["forms"]:
            token_id = form["token_id"]
            rank = item["rank_maps"][method][layer][token_id]
            candidates.append((rank, layer, token_id, form["text"]))
    rank, layer, token_id, token = min(candidates)
    return {
        "key": intermediate["key"],
        "minimum_rank": rank,
        "best_layer": layer,
        "best_token_id": token_id,
        "best_token": token,
        "forms": [dict(form) for form in intermediate["forms"]],
    }


def method_metrics(
    items: Sequence[Mapping[str, Any]], method: str, pass_at_k: Sequence[int]
) -> dict[str, Any]:
    require(bool(items), "method metric item set must not be empty")
    item_results: list[dict[str, Any]] = []
    all_ranks: list[int] = []
    log_denominator = math.log(LOGIT_VOCABULARY_SIZE)
    for item in items:
        intermediates = [
            best_intermediate(item, method, intermediate)
            for intermediate in item["intermediates"]
        ]
        ranks = [row["minimum_rank"] for row in intermediates]
        all_ranks.extend(ranks)
        item_pass = {
            str(k): sum(rank <= k for rank in ranks) / len(ranks) for k in pass_at_k
        }
        item_auc = statistics.fmean(
            math.log(LOGIT_VOCABULARY_SIZE / rank) / log_denominator
            for rank in ranks
        )
        item_results.append(
            {
                "id": item["id"],
                "event_family": item["event_family"],
                "request_index": item["request_index"],
                "offset": item["offset"],
                "state": item["state"],
                "leakage_class": item["leakage_class"],
                "category": item["category"],
                "intermediates": intermediates,
                "pass_at_k": item_pass,
                "normalized_log_rank_auc": item_auc,
            }
        )
    return {
        "pass_at_k": {
            str(k): statistics.fmean(row["pass_at_k"][str(k)] for row in item_results)
            for k in pass_at_k
        },
        "normalized_log_rank_auc": statistics.fmean(
            row["normalized_log_rank_auc"] for row in item_results
        ),
        "rank_summary": {
            "intermediate_count": len(all_ranks),
            "minimum": min(all_ranks),
            "median": statistics.median(all_ranks),
            "geometric_mean": math.exp(statistics.fmean(math.log(rank) for rank in all_ranks)),
            "maximum": max(all_ranks),
        },
        "items": item_results,
    }


def paired_bootstrap(
    jacobian: Mapping[str, Any],
    logit: Mapping[str, Any],
    *,
    label: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    require(samples > 0, "bootstrap sample count must be positive")
    jacobian_items = sequence(jacobian.get("items"), "jacobian.items")
    logit_items = sequence(logit.get("items"), "logit.items")
    require(
        len(jacobian_items) == len(logit_items) and jacobian_items,
        "paired bootstrap item grid mismatch",
    )
    auc_differences: list[float] = []
    pass10_differences: list[float] = []
    for left, right in zip(jacobian_items, logit_items, strict=True):
        require(left["id"] == right["id"], "paired bootstrap item IDs differ")
        auc_differences.append(
            left["normalized_log_rank_auc"] - right["normalized_log_rank_auc"]
        )
        pass10_differences.append(
            left["pass_at_k"]["10"] - right["pass_at_k"]["10"]
        )

    interval_seed = derived_seed(label, seed)
    method = "deterministic paired item percentile bootstrap"

    def summarize(values: Sequence[float], *, metric_seed: int) -> dict[str, Any]:
        if len(values) == 1:
            confidence_interval: dict[str, Any] = {
                "method": "unavailable: paired item stratum contains one item",
                "confidence_level": None,
                "seed": metric_seed,
                "samples": 0,
                "lower": None,
                "upper": None,
            }
        else:
            generator = random.Random(metric_seed)
            draws = [
                statistics.fmean(generator.choice(values) for _ in values)
                for _ in range(samples)
            ]
            confidence_interval = {
                "method": method,
                "confidence_level": 0.95,
                "seed": metric_seed,
                "samples": samples,
                "lower": quantile(draws, 0.025),
                "upper": quantile(draws, 0.975),
            }
        return {
            "estimate": statistics.fmean(values),
            "confidence_interval": confidence_interval,
            "positive_item_count": sum(value > 0.0 for value in values),
            "negative_item_count": sum(value < 0.0 for value in values),
            "tie_item_count": sum(value == 0.0 for value in values),
        }

    return {
        "sampling_unit": "frozen SWE probe item",
        "item_count": len(auc_differences),
        "normalized_log_rank_auc_gain": summarize(
            auc_differences, metric_seed=interval_seed
        ),
        "pass_at_10_gain": summarize(
            pass10_differences, metric_seed=interval_seed + 1
        ),
    }


def metric_summary(
    items: Sequence[Mapping[str, Any]],
    pass_at_k: Sequence[int],
    *,
    label: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    jacobian = method_metrics(items, "jacobian", pass_at_k)
    logit = method_metrics(items, "logit", pass_at_k)
    return {
        "item_count": len(items),
        "jacobian_lens": jacobian,
        "logit_lens": logit,
        "jacobian_minus_logit": {
            "pass_at_k": {
                str(k): jacobian["pass_at_k"][str(k)] - logit["pass_at_k"][str(k)]
                for k in pass_at_k
            },
            "normalized_log_rank_auc": jacobian["normalized_log_rank_auc"]
            - logit["normalized_log_rank_auc"],
        },
        "paired_item_bootstrap": paired_bootstrap(
            jacobian,
            logit,
            label=label,
            seed=seed,
            samples=samples,
        ),
    }


def summarize_report(
    validated: Mapping[str, Any],
    pass_at_k: Sequence[int],
    *,
    label: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    items = validated["items"]
    categories: dict[str, Any] = {}
    for category in (
        "task_explicit_baseline",
        "exact_pre_identifier_state",
        "post_tool_boundary_retention",
        "teacher_forced_lexical_control",
    ):
        selected = [item for item in items if item["category"] == category]
        require(bool(selected), f"analysis category {category} is empty")
        categories[category] = metric_summary(
            selected,
            pass_at_k,
            label=f"{label}:{category}",
            seed=seed,
            samples=samples,
        )
    pre_output = [
        item
        for item in items
        if item["category"]
        in ("exact_pre_identifier_state", "post_tool_boundary_retention")
    ]
    return {
        "numerical_eligibility": validated["numerical_eligibility"],
        "overall": metric_summary(
            items,
            pass_at_k,
            label=f"{label}:overall",
            seed=seed,
            samples=samples,
        ),
        "categories": categories,
        "sensitivity_pre_output_nonbaseline": metric_summary(
            pre_output,
            pass_at_k,
            label=f"{label}:pre-output-nonbaseline",
            seed=seed,
            samples=samples,
        ),
    }


def validate_pair(primary: Mapping[str, Any], native: Mapping[str, Any]) -> dict[str, Any]:
    left = primary["pair_rows"]
    right = native["pair_rows"]
    require(len(left) == len(right), "paired report item count mismatch")
    residuals: list[Mapping[str, Any]] = []
    for public_row, native_row in zip(left, right, strict=True):
        for field in (
            "id",
            "prompt_token_ids",
            "metadata",
            "target_token_id_override",
            "residual_capture_manifest",
            "logit_rank_maps",
            "adapter_top1",
            "final_top5",
            "final_norm_within",
            "final_logits_within",
        ):
            require(
                public_row[field] == native_row[field],
                f"paired reports differ in {field}: {public_row['id']}",
            )
        residuals.append(public_row["residual_capture_manifest"])
    return {
        "item_count": len(left),
        "exact_prompt_residual_and_logit_pairing": True,
        "residual_manifest_list_sha256": materialized_json_sha256(residuals),
    }


def analyze(
    config_value: Mapping[str, Any],
    summary_value: Mapping[str, Any],
    prompts_value: Sequence[Mapping[str, Any]],
    public_report: Mapping[str, Any],
    native_report: Mapping[str, Any] | None = None,
    *,
    config_sha256: str | None = None,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    contract = validate_bundle(
        config_value,
        summary_value,
        prompts_value,
        config_sha256=config_sha256,
    )
    public = validate_report(public_report, label="public", contract=contract)
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "exploratory_swe_intermediate_concept_analysis",
        "label": (
            "EXPLORATORY ONE-TASK ADAPTATION: method-aligned intermediate "
            "concept recovery, not a benchmark-level or causal result"
        ),
        "model": {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "logit_vocabulary_size": LOGIT_VOCABULARY_SIZE,
        },
        "source_bindings": {
            "config_sha256": contract["config_sha256"],
            "trajectory_bundle_sha256": contract["trajectory_sha256"],
            "prompt_bundle_sha256": contract["prompt_bundle_sha256"],
        },
        "evaluation": {
            "middle_band_layers": list(EXPECTED_LAYERS),
            "pass_at_k": contract["pass_at_k"],
            "item_macro_weighting": True,
            "minimum_over_verified_forms_and_fixed_middle_layers": True,
            "accepted_target_token_scored": False,
            "claims_gate_preregistered": False,
            "claims_gate": None,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_samples": bootstrap_samples,
            "limitations": [
                "One easy, successful SWE-Verified task is not an independent-task sample.",
                "Most post-tool concepts are explicit in the preceding transcript.",
                "Minimum rank over synonyms and 32 layers creates multiple opportunities.",
                "Readout association alone does not establish causal use by the model.",
            ],
        },
        "public": summarize_report(
            public,
            contract["pass_at_k"],
            label="public",
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
    }
    if native_report is not None:
        native = validate_report(native_report, label="native", contract=contract)
        result["native"] = summarize_report(
            native,
            contract["pass_at_k"],
            label="native",
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        )
        result["pairing"] = validate_pair(public, native)
    else:
        result["pairing"] = {
            "exact_prompt_residual_and_logit_pairing": None,
            "reason": "no native report supplied",
        }
    return result


def read_json(path: Path) -> tuple[Any, dict[str, Any]]:
    raw = path.read_bytes()
    return json.loads(raw), {
        "path": str(path.resolve()),
        "size_bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_source = read_json(args.config)
    summary, summary_source = read_json(args.summary)
    prompts, prompts_source = read_json(args.prompts)
    public_report, public_source = read_json(args.report)
    native_report = native_source = None
    if args.native_report:
        native_report, native_source = read_json(args.native_report)
    require(
        isinstance(config, dict)
        and isinstance(summary, dict)
        and isinstance(prompts, list)
        and isinstance(public_report, dict),
        "config, summary, prompts, and report JSON types are invalid",
    )
    result = analyze(
        config,
        summary,
        prompts,
        public_report,
        native_report,
        config_sha256=config_source["sha256"],
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    result["inputs"] = {
        "config": config_source,
        "summary": summary_source,
        "prompts": prompts_source,
        "public_report": public_source,
        "native_report": native_source,
    }
    atomic_write_json(args.output, result)
    public = result["public"]["overall"]
    bootstrap = public["paired_item_bootstrap"]["normalized_log_rank_auc_gain"]
    interval = bootstrap["confidence_interval"]
    print(
        "public fixed-middle AUC J/logit: "
        f"{public['jacobian_lens']['normalized_log_rank_auc']:.6f}/"
        f"{public['logit_lens']['normalized_log_rank_auc']:.6f}; "
        f"gain {bootstrap['estimate']:+.6f} "
        f"[{interval['lower']:+.6f}, {interval['upper']:+.6f}]"
    )
    if "native" in result:
        native = result["native"]["overall"]
        native_bootstrap = native["paired_item_bootstrap"][
            "normalized_log_rank_auc_gain"
        ]
        native_interval = native_bootstrap["confidence_interval"]
        print(
            "native fixed-middle AUC J/logit: "
            f"{native['jacobian_lens']['normalized_log_rank_auc']:.6f}/"
            f"{native['logit_lens']['normalized_log_rank_auc']:.6f}; "
            f"gain {native_bootstrap['estimate']:+.6f} "
            f"[{native_interval['lower']:+.6f}, {native_interval['upper']:+.6f}]"
        )
    print("claims gate: NONE (exploratory one-task adaptation)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
