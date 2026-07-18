#!/usr/bin/env python3
"""Validate and summarize a dense SWE Jacobian-lens trajectory report."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
from typing import Any, Mapping, Sequence


MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
MODEL_CONFIG_SHA256 = "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338"
MODEL_INDEX_SHA256 = "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2"
FIXED_MIDDLE_LAYERS = (24, 27, 28, 31, 32, 35, 36, 39, 40)
EXPECTED_REQUESTS = tuple(range(1, 10))
FINAL_NORM_MAX_ABS_TOLERANCE = 0.125
FINAL_NORM_RMS_TOLERANCE = 0.006
FINAL_LOGIT_MAX_ABS_TOLERANCE = 0.0625
FINAL_LOGIT_RMS_TOLERANCE = 0.01
FINAL_TOP_K = 5
BOOTSTRAP_SEED = 36027
BOOTSTRAP_SAMPLES = 20_000
NONINFERIORITY_MARGIN_NATS = -0.10


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
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(not isinstance(value, bool) and isinstance(value, int) and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


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


def bootstrap_mean(values: Sequence[float], *, seed: int, samples: int) -> dict[str, Any]:
    require(len(values) == 9, "bootstrap requires nine request-level values")
    require(samples > 0, "bootstrap sample count must be positive")
    generator = random.Random(seed)
    draws = [statistics.fmean(generator.choice(values) for _ in values) for _ in range(samples)]
    return {
        "method": "deterministic request-level percentile bootstrap",
        "confidence_level": 0.95,
        "seed": seed,
        "samples": samples,
        "lower": quantile(draws, 0.025),
        "upper": quantile(draws, 0.975),
    }


def bootstrap_relative_kl(
    pairs: Sequence[tuple[float, float]], *, seed: int, samples: int
) -> dict[str, Any]:
    require(len(pairs) == 9, "bootstrap requires nine request-level KL pairs")
    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sampled = [generator.choice(pairs) for _ in pairs]
        jacobian = statistics.fmean(pair[0] for pair in sampled)
        logit = statistics.fmean(pair[1] for pair in sampled)
        require(logit > 0.0, "logit-lens KL mean must be positive")
        draws.append((logit - jacobian) / logit)
    return {
        "method": "deterministic paired request-level percentile bootstrap",
        "confidence_level": 0.95,
        "seed": seed,
        "samples": samples,
        "lower": quantile(draws, 0.025),
        "upper": quantile(draws, 0.975),
    }


def exact_sign_flip_p(values: Sequence[float], *, null: float = 0.0) -> float:
    require(len(values) == 9, "sign-flip test requires nine request-level values")
    centered = [value - null for value in values]
    observed = statistics.fmean(centered)
    exceed = 0
    total = 1 << len(centered)
    for signs in itertools.product((-1.0, 1.0), repeat=len(centered)):
        statistic = statistics.fmean(sign * abs(value) for sign, value in zip(signs, centered, strict=True))
        if statistic >= observed - 1e-15:
            exceed += 1
    return exceed / total


def validate_fidelity(value: Any, label: str) -> dict[str, Any]:
    fidelity = mapping(value, label)
    require(fidelity.get("reference") == "captured_block_63_final_model", f"{label}.reference mismatch")
    kl = finite(fidelity.get("kl_final_to_readout"), f"{label}.kl_final_to_readout")
    require(kl >= 0.0, f"{label}.kl_final_to_readout must be nonnegative")
    top_k = integer(fidelity.get("top_k"), f"{label}.top_k", minimum=1)
    require(top_k == FINAL_TOP_K, f"{label}.top_k must be {FINAL_TOP_K}")
    count = integer(fidelity.get("top_k_overlap_count"), f"{label}.top_k_overlap_count")
    require(count <= top_k, f"{label}.top_k_overlap_count exceeds top_k")
    overlap = finite(fidelity.get("top_k_overlap_fraction"), f"{label}.top_k_overlap_fraction")
    require(math.isclose(overlap, count / top_k, abs_tol=1e-12), f"{label}.top_k overlap fields disagree")
    require(isinstance(fidelity.get("top1_matches_final"), bool), f"{label}.top1_matches_final must be boolean")
    return {"kl": kl, "overlap": overlap, "top1": fidelity["top1_matches_final"]}


def validate_readout(value: Any, label: str, target: int) -> dict[str, Any]:
    readout = mapping(value, label)
    require(readout.get("target_token_id") == target, f"{label}.target_token_id mismatch")
    rank = integer(readout.get("target_rank"), f"{label}.target_rank", minimum=1)
    logprob = finite(readout.get("target_logprob"), f"{label}.target_logprob")
    ids = sequence(readout.get("token_ids"), f"{label}.token_ids")
    require(len(ids) >= FINAL_TOP_K and all(isinstance(item, int) and not isinstance(item, bool) for item in ids), f"{label}.token_ids invalid")
    fidelity = validate_fidelity(readout.get("final_distribution_fidelity"), f"{label}.final_distribution_fidelity")
    return {"rank": rank, "logprob": logprob, **fidelity}


def validate_adapter(experiment: Mapping[str, Any], label: str) -> dict[str, bool]:
    norm = mapping(experiment.get("final_norm_reconstruction"), f"{label}.final_norm_reconstruction")
    norm_max = finite(norm.get("max_abs_error"), f"{label}.norm.max_abs_error")
    norm_rms = finite(norm.get("rms_error"), f"{label}.norm.rms_error")
    require(norm.get("max_abs_tolerance") == FINAL_NORM_MAX_ABS_TOLERANCE, f"{label}.norm max tolerance mismatch")
    require(norm.get("rms_tolerance") == FINAL_NORM_RMS_TOLERANCE, f"{label}.norm RMS tolerance mismatch")
    norm_expected = norm_max <= FINAL_NORM_MAX_ABS_TOLERANCE and norm_rms <= FINAL_NORM_RMS_TOLERANCE
    require(norm.get("within_tolerance") is norm_expected, f"{label}.norm status mismatch")

    logits = mapping(experiment.get("final_logits_reconstruction"), f"{label}.final_logits_reconstruction")
    logit_max = finite(logits.get("max_abs_error"), f"{label}.logits.max_abs_error")
    logit_rms = finite(logits.get("rms_error"), f"{label}.logits.rms_error")
    require(logits.get("max_abs_tolerance") == FINAL_LOGIT_MAX_ABS_TOLERANCE, f"{label}.logits max tolerance mismatch")
    require(logits.get("rms_tolerance") == FINAL_LOGIT_RMS_TOLERANCE, f"{label}.logits RMS tolerance mismatch")
    require(logits.get("top_k_prefix") == FINAL_TOP_K, f"{label}.logits top-k mismatch")
    top5 = logits.get("top_k_prefix_token_ids_match")
    require(isinstance(top5, bool), f"{label}.logits top-k status must be boolean")
    logits_expected = logit_max <= FINAL_LOGIT_MAX_ABS_TOLERANCE and logit_rms <= FINAL_LOGIT_RMS_TOLERANCE and top5
    require(logits.get("within_tolerance") is logits_expected, f"{label}.logits status mismatch")
    top1 = experiment.get("final_layer_top1_matches_greedy")
    require(isinstance(top1, bool), f"{label}.final_layer_top1_matches_greedy must be boolean")
    return {
        "top1": top1,
        "top5": top5,
        "norm_strict": norm_expected,
        "logits_strict": logits_expected,
        "norm_rms": norm_rms <= FINAL_NORM_RMS_TOLERANCE,
        "logit_rms": logit_rms <= FINAL_LOGIT_RMS_TOLERANCE,
    }


def validate_report(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    require(report.get("schema_version") == 3, f"{label} schema version mismatch")
    require(report.get("score_encoding") == "unrounded-float32", f"{label} score encoding mismatch")
    model = mapping(report.get("model"), f"{label}.model")
    require(model.get("repo_id") == MODEL_REPO and model.get("revision") == MODEL_REVISION, f"{label} pinned model mismatch")
    require(model.get("config_sha256") == MODEL_CONFIG_SHA256 and model.get("index_sha256") == MODEL_INDEX_SHA256, f"{label} model metadata hash mismatch")
    runtime = mapping(report.get("runtime"), f"{label}.runtime")
    require(runtime.get("mtp_enabled") is False and runtime.get("stream_final_only") is True, f"{label} runtime semantics mismatch")

    experiments = sequence(report.get("experiments"), f"{label}.experiments")
    require(bool(experiments), f"{label} experiments must not be empty")
    ids = [item.get("id") if isinstance(item, dict) else None for item in experiments]
    require(all(isinstance(item, str) and item for item in ids), f"{label} experiment IDs invalid")
    require(len(set(ids)) == len(ids) and ids == sorted(ids), f"{label} experiment IDs must be unique and sorted")

    observations: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    adapters: list[dict[str, bool]] = []
    layer_order: tuple[int, ...] | None = None
    coordinates: list[tuple[int, int]] = []
    for experiment_index, raw in enumerate(experiments):
        experiment = mapping(raw, f"{label}.experiments[{experiment_index}]")
        identifier = str(experiment["id"])
        metadata = mapping(experiment.get("metadata"), f"{identifier}.metadata")
        trajectory = mapping(metadata.get("trajectory"), f"{identifier}.metadata.trajectory")
        request_value = trajectory.get("request_index", metadata.get("request_index"))
        request = integer(request_value, f"{identifier}.request_index", minimum=1)
        stage = metadata.get("stage")
        if isinstance(stage, dict) and "request_index" in stage:
            require(stage["request_index"] == request, f"{identifier} stage request index mismatch")
        require(request in EXPECTED_REQUESTS, f"{identifier}.request_index outside 1..9")
        offset = integer(trajectory.get("offset"), f"{identifier}.offset")
        require(isinstance(trajectory.get("region"), str) and trajectory["region"], f"{identifier}.region invalid")
        events = trajectory.get("events")
        require(isinstance(events, list) and all(isinstance(event, str) for event in events), f"{identifier}.events invalid")
        target = integer(trajectory.get("target_token_id"), f"{identifier}.target_token_id")
        require(experiment.get("target_token_id_override") == target, f"{identifier} target override mismatch")
        coordinates.append((request, offset))

        prompt_ids = sequence(experiment.get("prompt_token_ids"), f"{identifier}.prompt_token_ids")
        require(bool(prompt_ids) and all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in prompt_ids), f"{identifier}.prompt_token_ids invalid")
        prompt_tokens = sequence(experiment.get("prompt_tokens"), f"{identifier}.prompt_tokens")
        require(len(prompt_tokens) == len(prompt_ids), f"{identifier}.prompt token fields disagree")
        final_position = len(prompt_ids) - 1
        require(experiment.get("positions_resolved") == [final_position], f"{identifier} must decode one final position")
        require(experiment.get("capture_positions_resolved") == [final_position], f"{identifier} capture position mismatch")
        require(experiment.get("final_validation_position") == final_position, f"{identifier} final validation position mismatch")

        for field in ("final_model_readout", "captured_final_model_readout"):
            final_rows = sequence(experiment.get(field), f"{identifier}.{field}")
            require(len(final_rows) == 1 and mapping(final_rows[0], field).get("target_token_id") == target, f"{identifier}.{field} target mismatch")

        layers = sequence(experiment.get("layers"), f"{identifier}.layers")
        current_order = tuple(integer(mapping(item, "layer").get("layer"), "layer") for item in layers)
        require(current_order == tuple(sorted(set(current_order))), f"{identifier} layers must be unique and sorted")
        if layer_order is None:
            layer_order = current_order
            require(set(FIXED_MIDDLE_LAYERS).issubset(layer_order), f"{label} omits fixed middle layers")
        else:
            require(current_order == layer_order, f"{identifier} layer grid changed")
        for layer_value in layers:
            layer = mapping(layer_value, f"{identifier}.layer")
            layer_id = int(layer["layer"])
            positions = sequence(layer.get("positions"), f"{identifier}.layer-{layer_id}.positions")
            require(len(positions) == 1, f"{identifier}.layer-{layer_id} must have one position")
            position = mapping(positions[0], f"{identifier}.layer-{layer_id}.position")
            require(position.get("capture_index") == 0 and position.get("token_position") == final_position, f"{identifier}.layer-{layer_id} position mismatch")
            jacobian = validate_readout(position.get("jacobian_lens"), f"{identifier}.layer-{layer_id}.jacobian", target)
            logit = validate_readout(position.get("logit_lens"), f"{identifier}.layer-{layer_id}.logit", target)
            observations.append({
                "id": identifier, "request": request, "offset": offset, "layer": layer_id,
                "j_kl": jacobian["kl"], "l_kl": logit["kl"],
                "logprob_gain": jacobian["logprob"] - logit["logprob"],
                "rank_gain": logit["rank"] - jacobian["rank"],
                "overlap_gain": jacobian["overlap"] - logit["overlap"],
            })
        adapters.append(validate_adapter(experiment, identifier))
        prompt_rows.append({"id": identifier, "request": request, "offset": offset, "tokens": prompt_ids, "target": target, "manifest": experiment.get("residual_capture_manifest"), "metadata": metadata})

    require(coordinates == sorted(set(coordinates)), f"{label} trajectory coordinates must be unique and sorted")
    require(tuple(sorted({request for request, _ in coordinates})) == EXPECTED_REQUESTS, f"{label} must cover all nine requests")
    for request in EXPECTED_REQUESTS:
        rows = [row for row in prompt_rows if row["request"] == request]
        base_lengths = {len(row["tokens"]) - row["offset"] for row in rows}
        require(len(base_lengths) == 1 and next(iter(base_lengths)) > 0, f"{label} request {request} base prompt pairing mismatch")
        longest = max(rows, key=lambda row: row["offset"])
        for row in rows:
            require(longest["tokens"][: len(row["tokens"])] == row["tokens"], f"{row['id']} is not an exact teacher-forced prefix")
            if row["offset"] < longest["offset"]:
                base = next(iter(base_lengths))
                require(longest["tokens"][base + row["offset"]] == row["target"], f"{row['id']} target does not pair with the trajectory")

    assertions = mapping(report.get("assertions"), f"{label}.assertions")
    all_top1 = all(item["top1"] for item in adapters)
    all_strict = all(item["norm_strict"] and item["logits_strict"] for item in adapters)
    require(assertions.get("all_final_layer_top1_match_greedy") is all_top1, f"{label} top1 assertion mismatch")
    require(assertions.get("all_final_adapter_reconstructions_within_tolerance") is all_strict, f"{label} adapter assertion mismatch")
    require(report.get("status") == ("passed" if all_top1 and all_strict else "failed"), f"{label} report status mismatch")
    adapter = {
        "experiment_count": len(adapters),
        "strict_pass_count": sum(item["top1"] and item["norm_strict"] and item["logits_strict"] for item in adapters),
        "strict_eligible": all_top1 and all_strict,
        "distribution_analysis_eligible": all(item["top1"] and item["top5"] and item["norm_rms"] and item["logit_rms"] for item in adapters),
    }
    return {"observations": observations, "prompts": prompt_rows, "layers": layer_order, "adapter": adapter, "report": report}


def _seed(label: str, seed: int) -> int:
    return seed + int(hashlib.sha256(label.encode("ascii")).hexdigest()[:8], 16)


def summarize_observations(rows: Sequence[Mapping[str, Any]], *, label: str, seed: int, samples: int) -> dict[str, Any]:
    require(bool(rows), f"{label} has no observations")
    requests: list[dict[str, Any]] = []
    for request in EXPECTED_REQUESTS:
        selected = [row for row in rows if row["request"] == request]
        require(bool(selected), f"{label} missing request {request}")
        requests.append({
            "request_index": request,
            "observation_count": len(selected),
            "jacobian_kl": statistics.fmean(row["j_kl"] for row in selected),
            "logit_kl": statistics.fmean(row["l_kl"] for row in selected),
            "kl_gain": statistics.fmean(row["l_kl"] - row["j_kl"] for row in selected),
            "target_logprob_gain": statistics.fmean(row["logprob_gain"] for row in selected),
            "target_rank_gain": statistics.fmean(row["rank_gain"] for row in selected),
            "top5_overlap_gain": statistics.fmean(row["overlap_gain"] for row in selected),
        })
    j_kl = statistics.fmean(row["jacobian_kl"] for row in requests)
    l_kl = statistics.fmean(row["logit_kl"] for row in requests)
    require(l_kl > 0.0, f"{label} logit KL must be positive")
    relative = (l_kl - j_kl) / l_kl
    relative_ci = bootstrap_relative_kl([(row["jacobian_kl"], row["logit_kl"]) for row in requests], seed=_seed(label + ":kl", seed), samples=samples)
    logprob_values = [row["target_logprob_gain"] for row in requests]
    return {
        "observation_count": len(rows),
        "request_count": 9,
        "macro_equal_request": {
            "jacobian_kl": j_kl,
            "logit_kl": l_kl,
            "kl_gain_nats": l_kl - j_kl,
            "relative_kl_reduction": relative,
            "relative_kl_reduction_ci": relative_ci,
            "kl_gain_exact_one_sided_sign_flip_p": exact_sign_flip_p([row["kl_gain"] for row in requests]),
            "target_logprob_gain_nats": statistics.fmean(logprob_values),
            "target_logprob_gain_ci": bootstrap_mean(logprob_values, seed=_seed(label + ":token", seed), samples=samples),
            "target_logprob_gain_exact_one_sided_sign_flip_p": exact_sign_flip_p(logprob_values),
            "target_rank_gain": statistics.fmean(row["target_rank_gain"] for row in requests),
            "top5_overlap_fraction_gain": statistics.fmean(row["top5_overlap_gain"] for row in requests),
        },
        "paired_wins": {
            "observation_kl": {"jacobian": sum(row["l_kl"] > row["j_kl"] for row in rows), "ties": sum(row["l_kl"] == row["j_kl"] for row in rows), "logit": sum(row["l_kl"] < row["j_kl"] for row in rows)},
            "request_mean_kl": {"jacobian": sum(row["kl_gain"] > 0 for row in requests), "ties": sum(row["kl_gain"] == 0 for row in requests), "logit": sum(row["kl_gain"] < 0 for row in requests)},
        },
        "request_level": requests,
    }


def summarize_validated(validated: Mapping[str, Any], *, seed: int, samples: int) -> dict[str, Any]:
    observations = validated["observations"]
    per_layer = {
        str(layer): summarize_observations([row for row in observations if row["layer"] == layer], label=f"layer-{layer}", seed=seed, samples=samples)
        for layer in validated["layers"]
    }
    primary_rows = [row for row in observations if row["layer"] in FIXED_MIDDLE_LAYERS]
    primary = summarize_observations(primary_rows, label="fixed-middle", seed=seed, samples=samples)
    macro = primary["macro_equal_request"]
    positive = primary["paired_wins"]["request_mean_kl"]["jacobian"]
    predictive_improvement = {
        "interpretation": (
            "Next-token distribution calibration only. Jacobian Lens is fit to "
            "future-summed causal transport, so this is not a lens-quality gate."
        ),
        "criteria": {
            "relative_kl_reduction_at_least_10_percent": macro["relative_kl_reduction"] >= 0.10,
            "bootstrap_ci_lower_above_zero": macro["relative_kl_reduction_ci"]["lower"] > 0.0,
            "at_least_7_of_9_positive_request_means": positive >= 7,
        }
    }
    predictive_improvement["passed"] = all(
        predictive_improvement["criteria"].values()
    )
    token_ci = macro["target_logprob_gain_ci"]
    noninferiority = {
        "interpretation": (
            "Teacher-forced next-token calibration only; failure does not imply "
            "failure on the Jacobian Lens intermediate-concept objective."
        ),
        "margin_nats": NONINFERIORITY_MARGIN_NATS,
        "mean_gain_nats": macro["target_logprob_gain_nats"],
        "confidence_interval": token_ci,
        "exact_one_sided_sign_flip_p_at_margin": exact_sign_flip_p([row["target_logprob_gain"] for row in primary["request_level"]], null=NONINFERIORITY_MARGIN_NATS),
        "passed": token_ci["lower"] > NONINFERIORITY_MARGIN_NATS,
    }
    return {
        "coverage": {"experiment_count": len(validated["prompts"]), "requests": list(EXPECTED_REQUESTS), "layers": list(validated["layers"]), "fixed_middle_layers": list(FIXED_MIDDLE_LAYERS)},
        "adapter_eligibility": validated["adapter"],
        "per_layer": per_layer,
        "fixed_middle_primary": primary,
        "predictive_improvement_diagnostic": predictive_improvement,
        "next_token_noninferiority_diagnostic": noninferiority,
        "predictive_diagnostic_passed": predictive_improvement["passed"]
        and noninferiority["passed"]
        and validated["adapter"]["distribution_analysis_eligible"],
    }


def validate_pair(primary: Mapping[str, Any], native: Mapping[str, Any]) -> None:
    require(primary["layers"] == native["layers"], "native layer grid differs")
    require(len(primary["prompts"]) == len(native["prompts"]), "native prompt count differs")
    for left, right in zip(primary["prompts"], native["prompts"], strict=True):
        for field in ("id", "tokens", "target", "manifest"):
            require(left[field] == right[field], f"native exact prompt pairing differs in {field}: {left['id']}")


def paired_native_comparison(primary: Mapping[str, Any], native: Mapping[str, Any], *, seed: int, samples: int) -> dict[str, Any]:
    left = {(row["id"], row["layer"]): row for row in primary["observations"] if row["layer"] in FIXED_MIDDLE_LAYERS}
    right = {(row["id"], row["layer"]): row for row in native["observations"] if row["layer"] in FIXED_MIDDLE_LAYERS}
    require(left.keys() == right.keys(), "native observation grid differs")
    values: list[float] = []
    request_values: list[dict[str, Any]] = []
    for request in EXPECTED_REQUESTS:
        gains = [(right[key]["l_kl"] - right[key]["j_kl"]) - (left[key]["l_kl"] - left[key]["j_kl"]) for key in left if left[key]["request"] == request]
        value = statistics.fmean(gains)
        values.append(value)
        request_values.append({"request_index": request, "native_minus_primary_kl_gain_nats": value})
    return {
        "interpretation": "positive values favor the native lens over the primary lens",
        "native_minus_primary_kl_gain_nats": statistics.fmean(values),
        "confidence_interval": bootstrap_mean(values, seed=_seed("native-pair", seed), samples=samples),
        "exact_one_sided_sign_flip_p": exact_sign_flip_p(values),
        "positive_request_count": sum(value > 0 for value in values),
        "request_level": request_values,
    }


def analyze(primary_report: Mapping[str, Any], native_report: Mapping[str, Any] | None = None, *, seed: int = BOOTSTRAP_SEED, samples: int = BOOTSTRAP_SAMPLES) -> dict[str, Any]:
    primary = validate_report(primary_report, "primary")
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "dense_swe_jlens_output_prediction_calibration",
        "label": (
            "CALIBRATION DIAGNOSTIC: comparison with captured final next-token "
            "distributions; not a Jacobian Lens quality or reproduction gate"
        ),
        "model": {"repo_id": MODEL_REPO, "revision": MODEL_REVISION},
        "statistics": {"request_macro_weighting": "equal across nine requests", "bootstrap_seed": seed, "bootstrap_samples": samples},
        "primary": summarize_validated(primary, seed=seed, samples=samples),
    }
    if native_report is not None:
        native = validate_report(native_report, "native")
        validate_pair(primary, native)
        result["native"] = summarize_validated(native, seed=seed, samples=samples)
        result["native_comparison"] = paired_native_comparison(primary, native, seed=seed, samples=samples)
    return result


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


def read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    require(isinstance(value, dict), f"{path} must contain an object")
    return value, {"path": str(path.resolve()), "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    primary, primary_source = read_json(args.report)
    native = native_source = None
    if args.native_report:
        native, native_source = read_json(args.native_report)
    result = analyze(primary, native, seed=args.bootstrap_seed, samples=args.bootstrap_samples)
    result["inputs"] = {"primary": primary_source, "native": native_source}
    atomic_write_json(args.output, result)
    fixed = result["primary"]["fixed_middle_primary"]["macro_equal_request"]
    diagnostic = result["primary"]["predictive_improvement_diagnostic"]
    token = result["primary"]["next_token_noninferiority_diagnostic"]
    adapter = result["primary"]["adapter_eligibility"]
    print(f"fixed-middle relative KL reduction: {fixed['relative_kl_reduction']:.3%} [{fixed['relative_kl_reduction_ci']['lower']:.3%}, {fixed['relative_kl_reduction_ci']['upper']:.3%}]")
    print(f"predictive improvement diagnostic: {'PASS' if diagnostic['passed'] else 'FAIL'}; next-token noninferiority diagnostic: {'PASS' if token['passed'] else 'FAIL'}; adapter analysis eligibility: {'PASS' if adapter['distribution_analysis_eligible'] else 'FAIL'}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
