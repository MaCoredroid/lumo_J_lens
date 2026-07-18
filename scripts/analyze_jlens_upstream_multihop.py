#!/usr/bin/env python3
"""Analyze Qwen J-lens reports on Anthropic's pinned multihop control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
from typing import Any, Mapping, Sequence


UPSTREAM_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
UPSTREAM_SOURCE_SHA256 = "50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416"
MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
TOKENIZER_JSON_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
MODEL_CONFIG_SHA256 = "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338"
TOKENIZER_VOCABULARY_SIZE = 248_077
LOGIT_VOCABULARY_SIZE = 248_320
PUBLIC_LENS_REPO = "neuronpedia/jacobian-lens"
PUBLIC_LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
PUBLIC_LENS_SHA256 = "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"
NATIVE_LENS_SHA256 = "82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057"
NATIVE_STATE_SHA256 = "f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6"
NATIVE_PROVENANCE_SHA256 = "289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601"
FIXED_MIDDLE_LAYERS = tuple(range(24, 48))
ALL_LAYERS = tuple(range(63))
PASS_K = (1, 5, 10, 50, 100)
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def materialized_json_sha256(value: Any) -> str:
    return sha256_bytes((json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii"))


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


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    require(manifest.get("schema_version") == 1 and manifest.get("kind") == "anthropic_jlens_multihop_qwen36_materialization", "materialization manifest identity mismatch")
    upstream = mapping(manifest.get("upstream"), "manifest.upstream")
    require(upstream.get("repository") == "anthropics/jacobian-lens" and upstream.get("relative_path") == "data/evaluations/lens-eval-multihop.json", "upstream source location mismatch")
    require(upstream.get("commit") == UPSTREAM_COMMIT and upstream.get("source_sha256") == UPSTREAM_SOURCE_SHA256, "upstream source binding mismatch")
    model = mapping(manifest.get("model"), "manifest.model")
    require(model.get("repo_id") == MODEL_REPO and model.get("revision") == MODEL_REVISION, "manifest model mismatch")
    require(model.get("tokenizer_json_sha256") == TOKENIZER_JSON_SHA256, "manifest tokenizer mismatch")
    require(model.get("config_sha256") == MODEL_CONFIG_SHA256, "manifest model config mismatch")
    require(model.get("tokenizer_vocabulary_size") == TOKENIZER_VOCABULARY_SIZE, "manifest tokenizer vocabulary size mismatch")
    vocabulary_size = model.get("logit_vocabulary_size")
    require(vocabulary_size == LOGIT_VOCABULARY_SIZE, "manifest LM-head vocabulary size mismatch")
    metric = mapping(manifest.get("metric_contract"), "manifest.metric_contract")
    require(metric.get("fixed_middle_layers") == list(FIXED_MIDDLE_LAYERS), "fixed middle band changed")
    require(metric.get("secondary_all_layers") == list(ALL_LAYERS), "secondary layer band changed")
    require(metric.get("unscorable_intermediate_policy") == "count_as_miss_to_preserve_all_upstream_item/intermediate_denominators", "unscorable policy changed")
    scored = mapping(manifest.get("scored_vocabulary"), "manifest.scored_vocabulary")
    ids, tokens = scored.get("token_ids"), scored.get("tokens")
    require(isinstance(ids, list) and ids == sorted(set(ids)) and all(isinstance(item, int) and not isinstance(item, bool) for item in ids), "manifest scored vocabulary IDs invalid")
    require(isinstance(tokens, list) and len(tokens) == len(ids) and all(isinstance(item, str) for item in tokens), "manifest scored vocabulary tokens invalid")
    outputs = mapping(manifest.get("outputs"), "manifest.outputs")
    source_hash = mapping(outputs.get("source_copy"), "manifest.outputs.source_copy").get("sha256")
    require(source_hash == UPSTREAM_SOURCE_SHA256, "materialized source copy binding mismatch")
    prompts_hash = mapping(outputs.get("prompts"), "manifest.outputs.prompts").get("sha256")
    require(isinstance(prompts_hash, str) and len(prompts_hash) == 64, "manifest prompt hash invalid")
    coverage = mapping(manifest.get("coverage"), "manifest.coverage")
    coverage_values = [coverage.get(key) for key in ("item_count", "intermediate_occurrence_count", "scorable_intermediate_occurrence_count", "excluded_intermediate_occurrence_count")]
    require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in coverage_values), "manifest coverage invalid")
    item_count, occurrence_count, scorable_count, excluded_count = coverage_values
    require(item_count > 0 and occurrence_count >= item_count, "manifest coverage counts invalid")
    require(scorable_count + excluded_count == occurrence_count, "manifest intermediate coverage does not add up")
    return {"token_ids": ids, "tokens": tokens, "token_text": dict(zip(ids, tokens, strict=True)), "vocabulary_size": vocabulary_size, "prompts_sha256": prompts_hash, "coverage": dict(coverage)}


def ranks_from_readout(
    value: Any,
    label: str,
    expected_ids: Sequence[int],
    expected_tokens: Mapping[int, str],
    vocabulary_size: int,
) -> dict[int, int]:
    readout = mapping(value, label)
    raw_records = readout.get("scored_tokens")
    if not expected_ids:
        require(raw_records in (None, []), f"{label} unexpectedly contains scored tokens")
        return {}
    records = sequence(raw_records, f"{label}.scored_tokens")
    result: dict[int, int] = {}
    for index, raw in enumerate(records):
        record = mapping(raw, f"{label}.scored_tokens[{index}]")
        token_id, rank = record.get("token_id"), record.get("rank")
        require(isinstance(token_id, int) and not isinstance(token_id, bool) and token_id not in result, f"{label} token IDs invalid")
        require(isinstance(rank, int) and not isinstance(rank, bool) and 1 <= rank <= vocabulary_size, f"{label} exact rank invalid")
        require(record.get("token") == expected_tokens.get(token_id), f"{label} decoded scored token mismatch")
        result[token_id] = rank
    require(list(result) == list(expected_ids), f"{label} scored vocabulary order/coverage mismatch")
    return result


def validate_lens(report: Mapping[str, Any], label: str) -> None:
    lens = mapping(report.get("lens"), f"{label}.lens")
    require(lens.get("d_model") == 5120 and lens.get("source_layers") == list(ALL_LAYERS), f"{label} lens geometry mismatch")
    require(lens.get("tensor_shape") == [5120, 5120], f"{label} lens tensor shape mismatch")
    if label == "primary":
        require(lens.get("repo_id") == PUBLIC_LENS_REPO and lens.get("revision") == PUBLIC_LENS_REVISION, "primary public lens identity mismatch")
        require(lens.get("sha256") == PUBLIC_LENS_SHA256, "primary public lens hash mismatch")
    else:
        require(lens.get("kind") == "native_nvfp4_ste_fit" and lens.get("sha256") == NATIVE_LENS_SHA256, "native lens identity mismatch")
        require(lens.get("state_sha256") == NATIVE_STATE_SHA256 and lens.get("provenance_sha256") == NATIVE_PROVENANCE_SHA256, "native lens provenance mismatch")
        require(lens.get("fit_model") == MODEL_REPO and lens.get("fit_model_revision") == MODEL_REVISION, "native fit model mismatch")


def validate_report(report: Mapping[str, Any], *, label: str, manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    require(report.get("schema_version") == 3, f"{label} report schema mismatch")
    require(report.get("score_encoding") == "unrounded-float32", f"{label} score encoding mismatch")
    validate_lens(report, label)
    model = mapping(report.get("model"), f"{label}.model")
    require(model.get("repo_id") == MODEL_REPO and model.get("revision") == MODEL_REVISION, f"{label} model mismatch")
    runtime = mapping(report.get("runtime"), f"{label}.runtime")
    require(runtime.get("mtp_enabled") is False and runtime.get("enforce_eager") is True and runtime.get("language_model_only") is True, f"{label} NVFP4 runtime contract mismatch")
    assertions = mapping(report.get("assertions"), f"{label}.assertions")
    for assertion in ("lens_hash_matches", "lens_metadata_matches", "model_architecture_matches"):
        require(assertions.get(assertion) is True, f"{label} required assertion failed: {assertion}")
    require(
        isinstance(assertions.get("all_final_layer_top1_match_greedy"), bool),
        f"{label} greedy parity assertion must be boolean",
    )
    scored = mapping(report.get("scored_vocabulary"), f"{label}.scored_vocabulary")
    require(scored.get("token_ids") == [] and scored.get("tokens") == [], f"{label} must not use a global scored vocabulary")
    union_ids = scored.get("union_token_ids")
    union_tokens = scored.get("union_tokens")
    require(isinstance(union_ids, list) and len(union_ids) == len(set(union_ids)) and set(union_ids) == set(contract["token_ids"]), f"{label} scored vocabulary union mismatch")
    require(isinstance(union_tokens, list) and len(union_tokens) == len(union_ids), f"{label} scored vocabulary union tokens invalid")
    require(all(contract["token_text"].get(token_id) == token for token_id, token in zip(union_ids, union_tokens, strict=True)), f"{label} scored vocabulary union decoding mismatch")
    require(scored.get("scope") == "global_plus_per_experiment", f"{label} scored vocabulary scope mismatch")
    experiments = sequence(report.get("experiments"), f"{label}.experiments")
    coverage = mapping(manifest.get("coverage"), "manifest.coverage")
    require(len(experiments) == coverage.get("item_count"), f"{label} item count mismatch")
    prompt_bundle: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    observed_union: dict[int, str] = {}
    observed_occurrences = observed_scorable = 0
    eligibility_counts = {
        "greedy_top1_match": 0,
        "final_top5_match": 0,
        "final_norm_within_tolerance": 0,
        "final_logits_within_tolerance": 0,
    }
    for item_index, raw in enumerate(experiments):
        experiment = mapping(raw, f"{label}.experiments[{item_index}]")
        metadata = mapping(experiment.get("metadata"), f"{label}.item-{item_index}.metadata")
        upstream = mapping(metadata.get("upstream"), f"{label}.item-{item_index}.upstream")
        require(upstream.get("commit") == UPSTREAM_COMMIT and upstream.get("source_sha256") == UPSTREAM_SOURCE_SHA256 and upstream.get("item_index") == item_index, f"{label} item {item_index} upstream binding mismatch")
        expected_id = f"upstream-multihop-{item_index:03d}-{upstream.get('name')}"
        require(experiment.get("id") == expected_id, f"{label} item {item_index} ID mismatch")
        tokenizer = mapping(metadata.get("tokenizer"), f"{label}.item-{item_index}.tokenizer")
        require(tokenizer.get("repo_id") == MODEL_REPO and tokenizer.get("revision") == MODEL_REVISION and tokenizer.get("tokenizer_json_sha256") == TOKENIZER_JSON_SHA256, f"{label} item {item_index} tokenizer binding mismatch")
        require(metadata.get("kind") == "anthropic_jlens_multihop_qwen36_control", f"{label} item {item_index} metadata kind mismatch")
        require(isinstance(upstream.get("target"), str) and upstream.get("target"), f"{label} item {item_index} target invalid")
        intermediate_records = sequence(upstream.get("intermediates"), f"{label}.item-{item_index}.intermediates")
        require(bool(intermediate_records), f"{label} item {item_index} intermediates empty")
        item_score_ids: set[int] = set()
        for intermediate in intermediate_records:
            record = mapping(intermediate, "intermediate")
            intermediate_text = record.get("text")
            require(isinstance(intermediate_text, str) and intermediate_text, "intermediate text invalid")
            forms = sequence(record.get("eligible_forms"), "intermediate.eligible_forms")
            exclusions = sequence(record.get("excluded_forms"), "intermediate.excluded_forms")
            require(record.get("scorable") is bool(forms), "intermediate scorable flag mismatch")
            require(len(forms) + len(exclusions) == 2, "intermediate must bind bare and leading-space forms")
            observed_occurrences += 1
            observed_scorable += int(bool(forms))
            observed_kinds: set[str] = set()
            for raw_form in forms:
                form = mapping(raw_form, "form")
                form_kind, token_id = form.get("form"), form.get("token_id")
                require(form_kind in ("bare", "leading_space") and form_kind not in observed_kinds, "eligible form kind invalid")
                expected_text = intermediate_text if form_kind == "bare" else f" {intermediate_text}"
                require(form.get("text") == expected_text, "eligible form text invalid")
                require(isinstance(token_id, int) and not isinstance(token_id, bool) and contract["token_text"].get(token_id) == expected_text, "eligible form missing from scored vocabulary")
                observed_kinds.add(form_kind)
                item_score_ids.add(token_id)
                observed_union[token_id] = expected_text
            for raw_exclusion in exclusions:
                exclusion = mapping(raw_exclusion, "exclusion")
                form_kind = exclusion.get("form")
                require(form_kind in ("bare", "leading_space") and form_kind not in observed_kinds, "excluded form kind invalid")
                expected_text = intermediate_text if form_kind == "bare" else f" {intermediate_text}"
                require(exclusion.get("text") == expected_text and isinstance(exclusion.get("token_ids"), list), "excluded form record invalid")
                require(exclusion.get("reason") in ("not_exactly_one_token", "decode_roundtrip_mismatch"), "excluded form reason invalid")
                observed_kinds.add(form_kind)
            require(observed_kinds == {"bare", "leading_space"}, "intermediate form coverage invalid")
        expected_item_ids = sorted(item_score_ids)
        experiment_scored = mapping(experiment.get("scored_vocabulary"), f"{label}.item-{item_index}.scored_vocabulary")
        require(experiment_scored.get("token_ids") == expected_item_ids, f"{label} item {item_index} scored vocabulary mismatch")
        require(experiment_scored.get("tokens") == [contract["token_text"][token_id] for token_id in expected_item_ids], f"{label} item {item_index} scored vocabulary tokens mismatch")
        prompt_token_ids = experiment.get("prompt_token_ids")
        require(isinstance(prompt_token_ids, list) and prompt_token_ids and all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in prompt_token_ids), f"{label} item {item_index} prompt token IDs invalid")
        final_prompt_position = len(prompt_token_ids) - 1
        require(experiment.get("positions_requested") == [-1] and experiment.get("positions_resolved") == [final_prompt_position], f"{label} item {item_index} readout position mismatch")
        top1_match = experiment.get("final_layer_top1_matches_greedy")
        require(isinstance(top1_match, bool), f"{label} item {item_index} greedy parity must be boolean")
        final_norm = mapping(experiment.get("final_norm_reconstruction"), f"{label}.item-{item_index}.final_norm")
        final_logits = mapping(experiment.get("final_logits_reconstruction"), f"{label}.item-{item_index}.final_logits")
        norm_within = final_norm.get("within_tolerance")
        logits_within = final_logits.get("within_tolerance")
        top5_match = final_logits.get("top_k_prefix_token_ids_match")
        require(all(isinstance(value, bool) for value in (norm_within, logits_within, top5_match)), f"{label} item {item_index} adapter flags must be boolean")
        eligibility_counts["greedy_top1_match"] += int(top1_match)
        eligibility_counts["final_top5_match"] += int(top5_match)
        eligibility_counts["final_norm_within_tolerance"] += int(norm_within)
        eligibility_counts["final_logits_within_tolerance"] += int(logits_within)
        layers = sequence(experiment.get("layers"), f"{label}.item-{item_index}.layers")
        layer_ids = [mapping(layer, "layer").get("layer") for layer in layers]
        require(layer_ids == list(ALL_LAYERS), f"{label} item {item_index} must contain all 63 layers")
        layer_ranks: dict[int, dict[str, dict[int, int]]] = {}
        logit_lens_readouts: list[Mapping[str, Any]] = []
        for layer_id, layer in zip(layer_ids, layers, strict=True):
            positions = sequence(mapping(layer, "layer").get("positions"), "layer.positions")
            require(len(positions) == 1, f"{label} item {item_index} layer {layer_id} must have one position")
            position = mapping(positions[0], "position")
            require(position.get("token_position") == final_prompt_position, f"{label} item {item_index} layer {layer_id} position mismatch")
            logit_readout = mapping(
                position.get("logit_lens"),
                f"{label}.item-{item_index}.layer-{layer_id}.logit",
            )
            layer_ranks[layer_id] = {
                "jacobian": ranks_from_readout(position.get("jacobian_lens"), f"{label}.item-{item_index}.layer-{layer_id}.jacobian", expected_item_ids, contract["token_text"], contract["vocabulary_size"]),
                "logit": ranks_from_readout(logit_readout, f"{label}.item-{item_index}.layer-{layer_id}.logit", expected_item_ids, contract["token_text"], contract["vocabulary_size"]),
            }
            logit_lens_readouts.append(logit_readout)
        prompt_record = {"id": experiment["id"], "text": experiment.get("prompt"), "token_ids": prompt_token_ids, "metadata": metadata}
        if expected_item_ids:
            prompt_record["score_token_ids"] = expected_item_ids
        prompt_bundle.append(prompt_record)
        item_rows.append({"name": upstream.get("name"), "intermediates": intermediate_records, "layer_ranks": layer_ranks})
        pair_rows.append({
            "id": experiment["id"],
            "prompt_token_ids": experiment.get("prompt_token_ids"),
            "residual_capture_manifest": experiment.get("residual_capture_manifest"),
            "logit_lens_sha256": materialized_json_sha256(logit_lens_readouts),
            "greedy_top1_match": top1_match,
            "final_top5_match": top5_match,
            "final_norm_within_tolerance": norm_within,
            "final_logits_within_tolerance": logits_within,
        })
    require(materialized_json_sha256(prompt_bundle) == contract["prompts_sha256"], f"{label} report does not reconstruct the bound prompt bundle")
    require(observed_union == contract["token_text"], f"{label} eligible forms do not reconstruct the manifest scored vocabulary")
    require(observed_occurrences == coverage.get("intermediate_occurrence_count") and observed_scorable == coverage.get("scorable_intermediate_occurrence_count"), f"{label} intermediate coverage mismatch")
    require(observed_occurrences - observed_scorable == coverage.get("excluded_intermediate_occurrence_count"), f"{label} excluded intermediate coverage mismatch")
    return {
        "items": item_rows,
        "pair_rows": pair_rows,
        "report": report,
        "numerical_eligibility": {
            "experiment_count": len(experiments),
            "counts": eligibility_counts,
            "interpretation": (
                "Descriptive method comparison remains paired on identical "
                "residuals; imperfect final-output parity limits causal claims."
            ),
        },
    }


def method_metrics(items: Sequence[Mapping[str, Any]], method: str, layers: Sequence[int], vocabulary_size: int) -> dict[str, Any]:
    item_results: list[dict[str, Any]] = []
    for item in items:
        intermediate_results: list[dict[str, Any]] = []
        for intermediate in item["intermediates"]:
            form_ids = [form["token_id"] for form in intermediate["eligible_forms"]]
            minimum = min((item["layer_ranks"][layer][method][token_id] for layer in layers for token_id in form_ids), default=None)
            intermediate_results.append({"text": intermediate["text"], "scorable": bool(form_ids), "minimum_rank": minimum})
        item_results.append({"name": item["name"], "intermediates": intermediate_results})
    pass_at = {}
    for k in PASS_K:
        pass_at[str(k)] = statistics.fmean(
            sum(row["minimum_rank"] is not None and row["minimum_rank"] <= k for row in item["intermediates"]) / len(item["intermediates"])
            for item in item_results
        )
    log_denominator = math.log(vocabulary_size)
    auc = statistics.fmean(
        statistics.fmean(
            (math.log(vocabulary_size) - math.log(row["minimum_rank"])) / log_denominator
            if row["minimum_rank"] is not None and row["minimum_rank"] <= vocabulary_size else 0.0
            for row in item["intermediates"]
        )
        for item in item_results
    )
    return {"pass_at_k": pass_at, "normalized_log_rank_auc": auc, "items": item_results}


def item_metric(item: Mapping[str, Any], *, k: int, vocabulary_size: int) -> tuple[float, float]:
    intermediates = item["intermediates"]
    pass_at_k = sum(
        row["minimum_rank"] is not None and row["minimum_rank"] <= k
        for row in intermediates
    ) / len(intermediates)
    log_denominator = math.log(vocabulary_size)
    auc = statistics.fmean(
        (math.log(vocabulary_size) - math.log(row["minimum_rank"]))
        / log_denominator
        if row["minimum_rank"] is not None
        and row["minimum_rank"] <= vocabulary_size
        else 0.0
        for row in intermediates
    )
    return pass_at_k, auc


def paired_item_bootstrap(
    jacobian: Mapping[str, Any],
    logit: Mapping[str, Any],
    *,
    vocabulary_size: int,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    require(samples > 0, "bootstrap sample count must be positive")
    jacobian_items = sequence(jacobian.get("items"), "jacobian.items")
    logit_items = sequence(logit.get("items"), "logit.items")
    require(len(jacobian_items) == len(logit_items) and jacobian_items, "paired bootstrap item grid mismatch")
    auc_differences: list[float] = []
    pass10_differences: list[float] = []
    for jacobian_item, logit_item in zip(jacobian_items, logit_items, strict=True):
        require(jacobian_item.get("name") == logit_item.get("name"), "paired bootstrap item names differ")
        jacobian_pass10, jacobian_auc = item_metric(
            jacobian_item, k=10, vocabulary_size=vocabulary_size
        )
        logit_pass10, logit_auc = item_metric(
            logit_item, k=10, vocabulary_size=vocabulary_size
        )
        pass10_differences.append(jacobian_pass10 - logit_pass10)
        auc_differences.append(jacobian_auc - logit_auc)

    generator = random.Random(seed)
    auc_draws: list[float] = []
    pass10_draws: list[float] = []
    for _ in range(samples):
        indices = [generator.randrange(len(auc_differences)) for _ in auc_differences]
        auc_draws.append(statistics.fmean(auc_differences[index] for index in indices))
        pass10_draws.append(
            statistics.fmean(pass10_differences[index] for index in indices)
        )

    def interval(values: Sequence[float], draws: Sequence[float]) -> dict[str, Any]:
        return {
            "estimate": statistics.fmean(values),
            "confidence_interval": {
                "confidence_level": 0.95,
                "lower": quantile(draws, 0.025),
                "upper": quantile(draws, 0.975),
            },
            "positive_item_count": sum(value > 0.0 for value in values),
            "negative_item_count": sum(value < 0.0 for value in values),
            "tie_item_count": sum(value == 0.0 for value in values),
        }

    return {
        "method": "deterministic paired item-level percentile bootstrap",
        "seed": seed,
        "samples": samples,
        "normalized_log_rank_auc_gain": interval(auc_differences, auc_draws),
        "pass_at_10_gain": interval(pass10_differences, pass10_draws),
    }


def band_summary(
    validated: Mapping[str, Any],
    layers: Sequence[int],
    vocabulary_size: int,
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    jacobian = method_metrics(validated["items"], "jacobian", layers, vocabulary_size)
    logit = method_metrics(validated["items"], "logit", layers, vocabulary_size)
    return {
        "layers": list(layers),
        "jacobian_lens": jacobian,
        "logit_lens": logit,
        "jacobian_minus_logit": {
            "pass_at_k": {str(k): jacobian["pass_at_k"][str(k)] - logit["pass_at_k"][str(k)] for k in PASS_K},
            "normalized_log_rank_auc": jacobian["normalized_log_rank_auc"] - logit["normalized_log_rank_auc"],
        },
        "paired_item_bootstrap": paired_item_bootstrap(
            jacobian,
            logit,
            vocabulary_size=vocabulary_size,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
    }


def summarize(
    validated: Mapping[str, Any],
    vocabulary_size: int,
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    return {
        "numerical_eligibility": validated["numerical_eligibility"],
        "primary_fixed_middle_band": band_summary(
            validated,
            FIXED_MIDDLE_LAYERS,
            vocabulary_size,
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        ),
        "secondary_all_layers": band_summary(
            validated,
            ALL_LAYERS,
            vocabulary_size,
            bootstrap_seed=bootstrap_seed + 1,
            bootstrap_samples=bootstrap_samples,
        ),
    }


def validate_pair(primary: Mapping[str, Any], native: Mapping[str, Any]) -> None:
    require(len(primary["pair_rows"]) == len(native["pair_rows"]), "paired item count mismatch")
    for left, right in zip(primary["pair_rows"], native["pair_rows"], strict=True):
        for field in (
            "id",
            "prompt_token_ids",
            "residual_capture_manifest",
            "logit_lens_sha256",
            "greedy_top1_match",
            "final_top5_match",
            "final_norm_within_tolerance",
            "final_logits_within_tolerance",
        ):
            require(left[field] == right[field], f"paired reports differ in {field}: {left['id']}")


def analyze(
    manifest_value: Mapping[str, Any],
    report: Mapping[str, Any],
    native_report: Mapping[str, Any] | None = None,
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    contract = validate_manifest(manifest_value)
    primary = validate_report(report, label="primary", manifest=manifest_value, contract=contract)
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "anthropic_jlens_multihop_qwen36_control_analysis",
        "upstream": {"commit": UPSTREAM_COMMIT, "source_sha256": UPSTREAM_SOURCE_SHA256, "metric": "pass@k = item-macro mean fraction of intermediates with min-over-band/form exact rank <= k"},
        "coverage": contract["coverage"],
        "auc_contract": {"reported": True, "justification": "runner records exact full-vocabulary competition ranks", "x_axis": "natural log rank threshold from 1 through vocabulary size", "normalization": "exact area under the pass-threshold step function divided by log(vocabulary_size)"},
        "statistics": {
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_samples": bootstrap_samples,
            "sampling_unit": "upstream multihop item",
        },
        "primary": summarize(
            primary,
            contract["vocabulary_size"],
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        ),
    }
    if native_report is not None:
        native = validate_report(native_report, label="native", manifest=manifest_value, contract=contract)
        validate_pair(primary, native)
        result["native"] = summarize(
            native,
            contract["vocabulary_size"],
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        )
    return result


def read_json(path: Path) -> tuple[Any, dict[str, Any]]:
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.resolve()), "size_bytes": len(raw), "sha256": sha256_bytes(raw)}


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, manifest_source = read_json(args.manifest)
    report, report_source = read_json(args.report)
    native = native_source = None
    if args.native_report:
        native, native_source = read_json(args.native_report)
    require(isinstance(manifest, dict) and isinstance(report, dict), "manifest and report must be objects")
    result = analyze(
        manifest,
        report,
        native,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    result["inputs"] = {"manifest": manifest_source, "primary_report": report_source, "native_report": native_source}
    atomic_write_json(args.output, result)
    band = result["primary"]["primary_fixed_middle_band"]
    print("fixed-middle pass@k J/logit: " + ", ".join(f"{k}={band['jacobian_lens']['pass_at_k'][str(k)]:.3f}/{band['logit_lens']['pass_at_k'][str(k)]:.3f}" for k in PASS_K))
    print(f"normalized log-rank AUC J/logit: {band['jacobian_lens']['normalized_log_rank_auc']:.4f}/{band['logit_lens']['normalized_log_rank_auc']:.4f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
