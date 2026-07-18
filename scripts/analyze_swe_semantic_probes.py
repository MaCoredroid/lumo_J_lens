#!/usr/bin/env python3
"""Analyze frozen exploratory semantic contrasts on a SWE trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping, Sequence


MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
PRIMARY_LAYERS = (39, 40)
EXPECTED_PROBES = 5
EXPECTED_OBSERVATIONS = 10


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    require(config.get("schema_version") == 1, "semantic config schema mismatch")
    require(config.get("kind") == "swe_verified_event_semantic_contrasts", "semantic config kind mismatch")
    model = mapping(config.get("model"), "config.model")
    require(model.get("repo_id") == MODEL_REPO and model.get("revision") == MODEL_REVISION, "semantic config model mismatch")
    require(config.get("primary_layers") == list(PRIMARY_LAYERS), "semantic primary layers changed")
    probes = sequence(config.get("probes"), "config.probes")
    require(len(probes) == EXPECTED_PROBES, f"semantic config must contain {EXPECTED_PROBES} probes")
    ids: list[str] = []
    points: set[tuple[int, int]] = set()
    vocabulary: dict[int, str] = {}
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(probes):
        probe = mapping(raw, f"config.probes[{index}]")
        identifier = probe.get("id")
        require(isinstance(identifier, str) and identifier and identifier not in ids, f"probe {index} ID invalid")
        request = probe.get("request_index")
        offset = probe.get("offset")
        require(isinstance(request, int) and not isinstance(request, bool) and 1 <= request <= 9, f"probe {identifier} request invalid")
        require(isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0, f"probe {identifier} offset invalid")
        require((request, offset) not in points, f"probe point {(request, offset)} duplicated")
        groups: dict[str, list[int]] = {}
        for group in ("positive", "negative"):
            records = sequence(probe.get(group), f"probe {identifier}.{group}")
            require(bool(records), f"probe {identifier}.{group} empty")
            token_ids: list[int] = []
            for record_index, raw_record in enumerate(records):
                record = mapping(raw_record, f"probe {identifier}.{group}[{record_index}]")
                token_id = record.get("token_id")
                text = record.get("text")
                require(isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0 and token_id not in token_ids, f"probe {identifier}.{group} token invalid")
                require(isinstance(text, str) and text, f"probe {identifier}.{group} text invalid")
                if token_id in vocabulary:
                    require(vocabulary[token_id] == text, f"token {token_id} text changed across probes")
                vocabulary[token_id] = text
                token_ids.append(token_id)
            groups[group] = token_ids
        require(not set(groups["positive"]) & set(groups["negative"]), f"probe {identifier} contrast groups overlap")
        ids.append(identifier)
        points.add((request, offset))
        normalized.append({"id": identifier, "request": request, "offset": offset, **groups})
    return {"probes": normalized, "ids": ids, "vocabulary": vocabulary}


def scored_map(value: Any, label: str, vocabulary: Mapping[int, str]) -> dict[int, float]:
    records = sequence(mapping(value, label).get("scored_tokens"), f"{label}.scored_tokens")
    result: dict[int, float] = {}
    for index, raw in enumerate(records):
        record = mapping(raw, f"{label}.scored_tokens[{index}]")
        token_id = record.get("token_id")
        require(isinstance(token_id, int) and not isinstance(token_id, bool) and token_id not in result, f"{label} scored token IDs invalid")
        require(record.get("token") == vocabulary.get(token_id), f"{label} token {token_id} text mismatch")
        result[token_id] = finite(record.get("score"), f"{label}.token-{token_id}.score")
    require(set(result) == set(vocabulary), f"{label} does not cover the configured vocabulary")
    return result


def logmeanexp(values: Sequence[float]) -> float:
    require(bool(values), "logmeanexp input must not be empty")
    maximum = max(values)
    return maximum + math.log(statistics.fmean(math.exp(value - maximum) for value in values))


def contrast(scores: Mapping[int, float], probe: Mapping[str, Any]) -> float:
    positive = logmeanexp([scores[token] for token in probe["positive"]])
    negative = logmeanexp([scores[token] for token in probe["negative"]])
    return positive - negative


def validate_report(
    report: Mapping[str, Any], *, label: str, config: Mapping[str, Any], config_sha256: str
) -> dict[str, Any]:
    require(report.get("schema_version") == 3, f"{label} report schema mismatch")
    model = mapping(report.get("model"), f"{label}.model")
    require(model.get("repo_id") == MODEL_REPO and model.get("revision") == MODEL_REVISION, f"{label} model mismatch")
    scored_vocabulary = mapping(report.get("scored_vocabulary"), f"{label}.scored_vocabulary")
    expected_ids = sorted(config["vocabulary"])
    require(scored_vocabulary.get("token_ids") == expected_ids, f"{label} scored vocabulary IDs mismatch")
    require(scored_vocabulary.get("tokens") == [config["vocabulary"][item] for item in expected_ids], f"{label} scored vocabulary text mismatch")

    experiments = sequence(report.get("experiments"), f"{label}.experiments")
    expected_experiment_ids = [f"swe-semantic-{item}" for item in config["ids"]]
    require([item.get("id") if isinstance(item, dict) else None for item in experiments] == expected_experiment_ids, f"{label} semantic experiment IDs/order mismatch")
    observations: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    trajectory_hash: str | None = None
    for raw, probe in zip(experiments, config["probes"], strict=True):
        experiment = mapping(raw, f"{label}.{probe['id']}")
        identifier = str(experiment["id"])
        metadata = mapping(experiment.get("metadata"), f"{identifier}.metadata")
        trajectory = mapping(metadata.get("trajectory"), f"{identifier}.trajectory")
        request = trajectory.get("request_index", metadata.get("request_index"))
        require(request == probe["request"] and trajectory.get("offset") == probe["offset"], f"{identifier} request/offset mismatch")
        semantic = mapping(metadata.get("semantic_probe"), f"{identifier}.semantic_probe")
        require(semantic.get("id") == probe["id"], f"{identifier} probe ID mismatch")
        require(semantic.get("positive_token_ids") == probe["positive"] and semantic.get("negative_token_ids") == probe["negative"], f"{identifier} contrast token binding mismatch")
        require(semantic.get("config_sha256") == config_sha256, f"{identifier} config hash mismatch")
        current_trajectory_hash = semantic.get("trajectory_bundle_sha256")
        require(isinstance(current_trajectory_hash, str) and len(current_trajectory_hash) == 64, f"{identifier} trajectory hash invalid")
        if trajectory_hash is None:
            trajectory_hash = current_trajectory_hash
        require(current_trajectory_hash == trajectory_hash, f"{identifier} trajectory hash changed")

        layers = sequence(experiment.get("layers"), f"{identifier}.layers")
        layer_map = {mapping(item, "layer").get("layer"): item for item in layers}
        require(set(PRIMARY_LAYERS).issubset(layer_map), f"{identifier} primary layers missing")
        final_rows = sequence(experiment.get("captured_final_model_readout"), f"{identifier}.captured_final_model_readout")
        require(len(final_rows) == 1, f"{identifier} final readout count mismatch")
        final_scores = scored_map(final_rows[0], f"{identifier}.final", config["vocabulary"])
        final_margin = contrast(final_scores, probe)
        for layer_id in PRIMARY_LAYERS:
            positions = sequence(mapping(layer_map[layer_id], "layer").get("positions"), f"{identifier}.layer-{layer_id}.positions")
            require(len(positions) == 1, f"{identifier}.layer-{layer_id} position count mismatch")
            position = mapping(positions[0], "position")
            jacobian_scores = scored_map(position.get("jacobian_lens"), f"{identifier}.layer-{layer_id}.jacobian", config["vocabulary"])
            logit_scores = scored_map(position.get("logit_lens"), f"{identifier}.layer-{layer_id}.logit", config["vocabulary"])
            jacobian_margin = contrast(jacobian_scores, probe)
            logit_margin = contrast(logit_scores, probe)
            jacobian_error = abs(jacobian_margin - final_margin)
            logit_error = abs(logit_margin - final_margin)
            observations.append({
                "probe_id": probe["id"], "request_index": probe["request"], "offset": probe["offset"], "layer": layer_id,
                "jacobian_margin": jacobian_margin, "logit_margin": logit_margin, "final_margin": final_margin,
                "jacobian_absolute_error": jacobian_error, "logit_absolute_error": logit_error,
                "jacobian_closer": jacobian_error < logit_error,
            })
        prompts.append({
            "id": identifier,
            "prompt_token_ids": experiment.get("prompt_token_ids"),
            "residual_capture_manifest": experiment.get("residual_capture_manifest"),
            "request_index": request,
            "offset": probe["offset"],
        })
    require(len(observations) == EXPECTED_OBSERVATIONS, f"{label} must yield {EXPECTED_OBSERVATIONS} primary observations")
    return {"observations": observations, "prompts": prompts, "trajectory_bundle_sha256": trajectory_hash, "report": report}


def aggregate(validated: Mapping[str, Any]) -> dict[str, Any]:
    rows = validated["observations"]
    jacobian_error = statistics.fmean(row["jacobian_absolute_error"] for row in rows)
    logit_error = statistics.fmean(row["logit_absolute_error"] for row in rows)
    reduction = (logit_error - jacobian_error) / logit_error if logit_error > 0 else 0.0
    summary = {
        "observation_count": len(rows),
        "primary_layers": list(PRIMARY_LAYERS),
        "jacobian_positive_count": sum(row["jacobian_margin"] > 0 for row in rows),
        "logit_positive_count": sum(row["logit_margin"] > 0 for row in rows),
        "jacobian_closer_count": sum(row["jacobian_closer"] for row in rows),
        "mean_jacobian_absolute_error": jacobian_error,
        "mean_logit_absolute_error": logit_error,
        "mean_absolute_error_reduction": reduction,
        "observations": rows,
    }
    summary["final_margin_calibration_diagnostic"] = {
        "interpretation": (
            "Describes closeness to the final next-token contrast only; it is "
            "not a Jacobian Lens quality or reproduction gate."
        ),
        "criteria": {
            "jacobian_positive_at_least_8_of_10": summary["jacobian_positive_count"] >= 8,
            "jacobian_closer_at_least_7_of_10": summary["jacobian_closer_count"] >= 7,
            "mean_absolute_error_reduction_at_least_25_percent": reduction >= 0.25,
        }
    }
    summary["final_margin_calibration_diagnostic"]["passed"] = all(
        summary["final_margin_calibration_diagnostic"]["criteria"].values()
    )
    return summary


def validate_pair(primary: Mapping[str, Any], native: Mapping[str, Any]) -> dict[str, Any]:
    require(primary["trajectory_bundle_sha256"] == native["trajectory_bundle_sha256"], "paired trajectory bundle hash mismatch")
    require(len(primary["prompts"]) == len(native["prompts"]), "paired prompt count mismatch")
    for left, right in zip(primary["prompts"], native["prompts"], strict=True):
        for field in ("id", "prompt_token_ids", "residual_capture_manifest", "request_index", "offset"):
            require(left[field] == right[field], f"paired reports differ in {field}: {left['id']}")
    left_rows = {(row["probe_id"], row["layer"]): row for row in primary["observations"]}
    right_rows = {(row["probe_id"], row["layer"]): row for row in native["observations"]}
    require(left_rows.keys() == right_rows.keys(), "paired semantic observation grid mismatch")
    agreement = sum(
        (left_rows[key]["jacobian_margin"] > 0) == (right_rows[key]["jacobian_margin"] > 0)
        for key in left_rows
    )
    return {
        "observation_count": EXPECTED_OBSERVATIONS,
        "jacobian_margin_sign_agreement_count": agreement,
        "required_count": 8,
        "passed": agreement >= 8,
    }


def analyze(config_value: Mapping[str, Any], report: Mapping[str, Any], *, config_sha256: str, native_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = validate_config(config_value)
    primary = validate_report(report, label="primary", config=config, config_sha256=config_sha256)
    primary_summary = aggregate(primary)
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "exploratory_swe_semantic_probe_analysis",
        "label": "EXPLORATORY: event-aligned token contrasts; not a causal or confirmatory result",
        "model": {"repo_id": MODEL_REPO, "revision": MODEL_REVISION},
        "config_sha256": config_sha256,
        "trajectory_bundle_sha256": primary["trajectory_bundle_sha256"],
        "primary": primary_summary,
    }
    if native_report is not None:
        native = validate_report(native_report, label="native", config=config, config_sha256=config_sha256)
        result["native"] = aggregate(native)
        result["paired_sign_agreement"] = validate_pair(primary, native)
    else:
        result["paired_sign_agreement"] = {"passed": None, "reason": "no second report supplied"}
    pairing_pass = result["paired_sign_agreement"]["passed"]
    result["paired_calibration_diagnostic"] = {
        "final_margin_diagnostic_passed": primary_summary[
            "final_margin_calibration_diagnostic"
        ]["passed"],
        "paired_sign_agreement_passed": pairing_pass,
        "passed": primary_summary["final_margin_calibration_diagnostic"][
            "passed"
        ]
        and pairing_pass is True,
    }
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_source = read_json(args.config)
    report, report_source = read_json(args.report)
    native = native_source = None
    if args.native_report:
        native, native_source = read_json(args.native_report)
    require(isinstance(config, dict) and isinstance(report, dict), "config and report must be objects")
    result = analyze(config, report, config_sha256=config_source["sha256"], native_report=native)
    result["inputs"] = {"config": config_source, "primary_report": report_source, "native_report": native_source}
    atomic_write_json(args.output, result)
    primary = result["primary"]
    print(f"semantic probes: J positive {primary['jacobian_positive_count']}/10; logit positive {primary['logit_positive_count']}/10; J closer {primary['jacobian_closer_count']}/10")
    calibration = primary["final_margin_calibration_diagnostic"]
    print(f"mean final-margin absolute-error reduction: {primary['mean_absolute_error_reduction']:.2%}; calibration diagnostic: {'PASS' if calibration['passed'] else 'FAIL'}; paired sign agreement: {result['paired_sign_agreement']['passed']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
