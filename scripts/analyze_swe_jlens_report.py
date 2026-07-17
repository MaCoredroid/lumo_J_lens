#!/usr/bin/env python3
"""Validate and summarize the certified SWE Jacobian-Lens replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_jlens_nvfp4_reports import compare_reports, validate_paired_report_identities


VALIDATION = ROOT / "validation"
FIXED_MIDDLE_LAYERS = (24, 27, 28, 31, 32, 35, 36, 39, 40)
SEMANTIC_STAGES = (3, 4, 6, 7, 8)
SEMANTIC_LAYERS = (39, 40)
EXPECTED_SOURCE_LAYERS = tuple(range(63))
EXPECTED_PROMPT_COUNTS = (
    11861,
    12148,
    12743,
    13629,
    13883,
    14522,
    15073,
    15327,
    15678,
)
EXPECTED_PROVENANCE_ID = (
    "72da18d1cead29ce7c4fe2627608040599c5f43c9e2b589855f89efa39afe038"
)
CORRECT_CANDIDATE = "cothm"
BUGGY_CANDIDATE = "cotm"
METHODS = ("jacobian_lens", "logit_lens")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def materializer_json_sha256(value: Any) -> str:
    raw = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    return sha256_bytes(raw)


def read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    try:
        display_path = str(path.resolve().relative_to(ROOT))
    except ValueError:
        display_path = str(path.resolve())
    return value, {
        "path": display_path,
        "size_bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def experiments_by_id(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    experiments = report.get("experiments")
    require(isinstance(experiments, list), "report experiments must be a list")
    result: dict[str, dict[str, Any]] = {}
    for experiment in experiments:
        require(isinstance(experiment, dict), "experiment must be an object")
        identifier = experiment.get("id")
        require(isinstance(identifier, str), "experiment id must be a string")
        require(identifier not in result, f"duplicate experiment id {identifier}")
        result[identifier] = experiment
    return result


def layer_map(experiment: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    layers = experiment.get("layers")
    require(isinstance(layers, list), "experiment layers must be a list")
    result = {layer["layer"]: layer for layer in layers}
    require(tuple(sorted(result)) == EXPECTED_SOURCE_LAYERS, "source layers are incomplete")
    return result


def position_readout(
    experiment: Mapping[str, Any], layer: int, method: str
) -> dict[str, Any]:
    positions = layer_map(experiment)[layer]["positions"]
    require(len(positions) == 1, "SWE replay requires one position per layer")
    return positions[0][method]


def average_ranks(values: Sequence[int]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def spearman(left: Sequence[int], right: Sequence[int]) -> float:
    require(len(left) == len(right) and len(left) > 1, "invalid Spearman inputs")
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_mean = statistics.fmean(left_rank)
    right_mean = statistics.fmean(right_rank)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_rank, right_rank, strict=True)
    )
    left_norm = sum((x - left_mean) ** 2 for x in left_rank)
    right_norm = sum((y - right_mean) ** 2 for y in right_rank)
    require(left_norm > 0 and right_norm > 0, "undefined Spearman correlation")
    return numerator / math.sqrt(left_norm * right_norm)


def adapter_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    experiments = list(experiments_by_id(report).values())
    norm = [item["final_norm_reconstruction"]["within_tolerance"] for item in experiments]
    logits = [
        item["final_logits_reconstruction"]["within_tolerance"]
        for item in experiments
    ]
    top1 = [item["final_layer_top1_matches_greedy"] for item in experiments]
    top5 = [
        item["final_logits_reconstruction"]["top_k_prefix_token_ids_match"]
        for item in experiments
    ]
    return {
        "report_status": report["status"],
        "experiment_count": len(experiments),
        "combined_strict_pass_count": sum(
            bool(a and b and c and d)
            for a, b, c, d in zip(norm, logits, top1, top5, strict=True)
        ),
        "final_norm_pass_count": sum(bool(value) for value in norm),
        "final_logits_pass_count": sum(bool(value) for value in logits),
        "final_top1_pass_count": sum(bool(value) for value in top1),
        "final_top5_prefix_pass_count": sum(bool(value) for value in top5),
        "failed_experiment_ids": [
            experiment["id"]
            for experiment, passed in zip(experiments, logits, strict=True)
            if not passed
        ],
    }


def validate_stage_pair(
    native: Mapping[str, Any], public: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_paired_report_identities(native, public)
    native_by_id = experiments_by_id(native)
    public_by_id = experiments_by_id(public)
    expected_ids = [f"swe-sympy-13480-request-{index:02d}" for index in range(1, 10)]
    require(list(native_by_id) == expected_ids, "native stage IDs are not canonical")
    require(list(public_by_id) == expected_ids, "public stage IDs are not canonical")
    native_items = [native_by_id[identifier] for identifier in expected_ids]
    public_items = [public_by_id[identifier] for identifier in expected_ids]
    require(
        tuple(len(item["prompt_token_ids"]) for item in native_items)
        == EXPECTED_PROMPT_COUNTS,
        "stage prompt token counts do not match the certified ledger",
    )
    for native_item, public_item in zip(native_items, public_items, strict=True):
        for field in (
            "id",
            "prompt_token_ids",
            "metadata",
            "positions_resolved",
            "generated_token_id",
            "generated_token",
            "residual_capture_manifest",
            "final_model_readout",
        ):
            require(
                native_item[field] == public_item[field],
                f"stage pair differs in {field}: {native_item['id']}",
            )
        require(
            native_item["metadata"]["provenance_id"] == EXPECTED_PROVENANCE_ID,
            "stage provenance ID mismatch",
        )
        for layer in EXPECTED_SOURCE_LAYERS:
            require(
                position_readout(native_item, layer, "logit_lens")
                == position_readout(public_item, layer, "logit_lens"),
                f"stage logit baseline differs at {native_item['id']} layer {layer}",
            )
    return native_items, public_items


def validate_candidate_pair(
    native: Mapping[str, Any], public: Mapping[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    validate_paired_report_identities(native, public)

    def grouped(report: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for experiment in experiments_by_id(report).values():
            metadata = experiment["metadata"]
            require(metadata["provenance_id"] == EXPECTED_PROVENANCE_ID, "candidate provenance mismatch")
            candidate = metadata["candidate"]["identifier"]
            groups.setdefault(candidate, []).append(experiment)
        for items in groups.values():
            items.sort(key=lambda item: item["metadata"]["step"]["index"])
        require(set(groups) == {CORRECT_CANDIDATE, BUGGY_CANDIDATE}, "candidate set mismatch")
        require(len(groups[CORRECT_CANDIDATE]) == 3, "cothm must have three BPE steps")
        require(len(groups[BUGGY_CANDIDATE]) == 2, "cotm must have two BPE steps")
        return groups

    native_groups = grouped(native)
    public_groups = grouped(public)
    for candidate in (CORRECT_CANDIDATE, BUGGY_CANDIDATE):
        for native_item, public_item in zip(
            native_groups[candidate], public_groups[candidate], strict=True
        ):
            for field in (
                "id",
                "prompt_token_ids",
                "target_token_id_override",
                "metadata",
                "residual_capture_manifest",
                "final_model_readout",
            ):
                require(native_item[field] == public_item[field], f"candidate pair differs in {field}")
            for layer in EXPECTED_SOURCE_LAYERS:
                require(
                    position_readout(native_item, layer, "logit_lens")
                    == position_readout(public_item, layer, "logit_lens"),
                    f"candidate logit baseline differs at {native_item['id']} layer {layer}",
                )
    native_first = {
        candidate: native_groups[candidate][0]
        for candidate in (CORRECT_CANDIDATE, BUGGY_CANDIDATE)
    }
    require(
        native_first[CORRECT_CANDIDATE]["prompt_token_ids"]
        == native_first[BUGGY_CANDIDATE]["prompt_token_ids"],
        "first candidate alternatives do not share an exact context",
    )
    require(
        native_first[CORRECT_CANDIDATE]["residual_capture_manifest"]
        == native_first[BUGGY_CANDIDATE]["residual_capture_manifest"],
        "first candidate alternatives do not share a residual capture",
    )
    require(
        native_groups[CORRECT_CANDIDATE][0]["target_token_id_override"] == 981
        and native_groups[BUGGY_CANDIDATE][0]["target_token_id_override"] == 62317,
        "candidate first-token IDs are not co/cot",
    )
    return native_groups, public_groups


def candidate_layer_rows(
    native_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    public_groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    native_correct = native_groups[CORRECT_CANDIDATE][0]
    native_buggy = native_groups[BUGGY_CANDIDATE][0]
    public_correct = public_groups[CORRECT_CANDIDATE][0]
    public_buggy = public_groups[BUGGY_CANDIDATE][0]
    rows = []
    for layer in EXPECTED_SOURCE_LAYERS:
        row: dict[str, Any] = {"layer": layer}
        for label, correct, buggy, method in (
            ("native_jacobian", native_correct, native_buggy, "jacobian_lens"),
            ("public_jacobian", public_correct, public_buggy, "jacobian_lens"),
            ("logit_lens", native_correct, native_buggy, "logit_lens"),
        ):
            correct_readout = position_readout(correct, layer, method)
            buggy_readout = position_readout(buggy, layer, method)
            row[label] = {
                "correct_token": "co",
                "correct_rank": correct_readout["target_rank"],
                "correct_logprob": correct_readout["target_logprob"],
                "buggy_token": "cot",
                "buggy_rank": buggy_readout["target_rank"],
                "buggy_logprob": buggy_readout["target_logprob"],
                "correct_minus_buggy_logprob": (
                    correct_readout["target_logprob"]
                    - buggy_readout["target_logprob"]
                ),
            }
        rows.append(row)
    return rows


def margin_summary(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["layer"] in FIXED_MIDDLE_LAYERS]
    margins = [row[method]["correct_minus_buggy_logprob"] for row in selected]
    return {
        "layers": list(FIXED_MIDDLE_LAYERS),
        "mean_correct_minus_buggy_logprob": statistics.fmean(margins),
        "positive_layer_count": sum(value > 0 for value in margins),
        "layer_count": len(margins),
        "rows": [
            {"layer": row["layer"], **row[method]}
            for row in selected
        ],
    }


def sequence_totals(
    groups: Mapping[str, Sequence[Mapping[str, Any]]], method: str, layer: int
) -> dict[str, Any]:
    totals = {}
    for candidate in (CORRECT_CANDIDATE, BUGGY_CANDIDATE):
        values = [
            position_readout(experiment, layer, method)["target_logprob"]
            for experiment in groups[candidate]
        ]
        totals[candidate] = {
            "bpe_token_count": len(values),
            "sum_logprob": sum(values),
            "mean_logprob_per_bpe_token": statistics.fmean(values),
        }
    totals["correct_minus_buggy_sum_logprob"] = (
        totals[CORRECT_CANDIDATE]["sum_logprob"]
        - totals[BUGGY_CANDIDATE]["sum_logprob"]
    )
    return totals


def middle_pair_metrics(
    native_items: Sequence[Mapping[str, Any]],
    public_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    top1 = 0
    exact_top5 = 0
    overlaps = []
    logprob_deltas = []
    layer_spearman = []
    for layer in FIXED_MIDDLE_LAYERS:
        native_ranks = []
        public_ranks = []
        for native, public in zip(native_items, public_items, strict=True):
            left = position_readout(native, layer, "jacobian_lens")
            right = position_readout(public, layer, "jacobian_lens")
            top1 += left["token_ids"][0] == right["token_ids"][0]
            left_top5 = set(left["token_ids"][:5])
            right_top5 = set(right["token_ids"][:5])
            exact_top5 += left_top5 == right_top5
            overlaps.append(len(left_top5 & right_top5) / 5)
            logprob_deltas.append(abs(left["target_logprob"] - right["target_logprob"]))
            native_ranks.append(left["target_rank"])
            public_ranks.append(right["target_rank"])
        layer_spearman.append(spearman(native_ranks, public_ranks))
    count = len(FIXED_MIDDLE_LAYERS) * len(native_items)
    return {
        "observation_count": count,
        "top1_agreement_count": top1,
        "top1_agreement_rate": top1 / count,
        "top5_exact_set_agreement_count": exact_top5,
        "top5_exact_set_agreement_rate": exact_top5 / count,
        "top5_overlap_mean_fraction": statistics.fmean(overlaps),
        "macro_layer_target_rank_spearman": statistics.fmean(layer_spearman),
        "target_logprob_mean_absolute_delta": statistics.fmean(logprob_deltas),
    }


def semantic_readouts(
    native_items: Sequence[Mapping[str, Any]],
    public_items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for stage in SEMANTIC_STAGES:
        native = native_items[stage - 1]
        public = public_items[stage - 1]
        row = {
            "stage": stage,
            "stage_name": native["metadata"]["stage"]["name"],
            "original_sampled_first_token": native["metadata"]["sampled_next"]["first_token_text"],
            "layers": [],
        }
        for layer in SEMANTIC_LAYERS:
            native_readout = position_readout(native, layer, "jacobian_lens")
            public_readout = position_readout(public, layer, "jacobian_lens")
            row["layers"].append(
                {
                    "layer": layer,
                    "native_top5_tokens": native_readout["tokens"][:5],
                    "public_top5_tokens": public_readout["tokens"][:5],
                }
            )
        rows.append(row)
    return rows


def validate_preflight(
    report: Mapping[str, Any],
    expected_stage: Mapping[str, Any],
    expected_public_report: Mapping[str, Any],
    stage_identity: Mapping[str, Any],
) -> dict[str, Any]:
    require(report.get("schema_version") == 3, "preflight schema version mismatch")
    require(
        report.get("score_encoding") == expected_public_report.get("score_encoding"),
        "preflight score encoding mismatch",
    )
    require(report.get("model") == expected_public_report.get("model"), "preflight model identity mismatch")
    require(report.get("lens") == expected_public_report.get("lens"), "preflight public lens identity mismatch")
    runtime = report.get("runtime")
    require(isinstance(runtime, dict), "preflight runtime must be an object")
    for field, expected in stage_identity["runtime"].items():
        actual = runtime.get(field)
        require(
            type(actual) is type(expected) and actual == expected,
            f"preflight runtime identity mismatch: {field}",
        )
    experiments = list(experiments_by_id(report).values())
    require(report["status"] == "passed", "longest-context preflight did not pass")
    require(len(experiments) == 1, "preflight must contain one experiment")
    experiment = experiments[0]
    for field in (
        "id",
        "prompt",
        "prompt_token_ids",
        "prompt_tokens",
        "metadata",
        "positions_requested",
        "positions_resolved",
        "capture_positions_resolved",
        "final_validation_position",
        "generated_token_id",
        "generated_token",
    ):
        require(
            experiment.get(field) == expected_stage.get(field),
            f"preflight does not bind exact stage request 9 field: {field}",
        )
    require(len(experiment["prompt_token_ids"]) == 15678, "preflight token count mismatch")
    require(experiment["positions_resolved"] == [15677], "preflight position mismatch")
    require(
        (experiment["generated_token_id"], experiment["generated_token"])
        == (760, "The"),
        "preflight generated token mismatch",
    )
    preflight_layers = tuple(layer["layer"] for layer in experiment["layers"])
    require(preflight_layers == (31, 32), "preflight layer set mismatch")
    require(all(report["assertions"].values()), "preflight assertions did not all pass")
    manifest = experiment.get("residual_capture_manifest")
    require(isinstance(manifest, dict), "preflight residual manifest must be an object")
    require(
        manifest.get("tensor_count") == 64
        and manifest.get("logical_bytes") == 64 * 1 * 5120 * 4
        and manifest.get("token_positions") == [15677]
        and isinstance(manifest.get("sha256"), str)
        and len(manifest["sha256"]) == 64,
        "preflight residual manifest geometry mismatch",
    )
    stage_manifest = expected_stage["residual_capture_manifest"]
    return {
        "status": report["status"],
        "prompt_id": experiment["id"],
        "prompt_token_count": len(experiment["prompt_token_ids"]),
        "position": experiment["positions_resolved"][0],
        "layers": [31, 32],
        "generated_token": experiment["generated_token"],
        "assertions": report["assertions"],
        "exact_stage_context_fields_match": True,
        "independent_capture_manifest_sha256": manifest["sha256"],
        "all_layer_stage_capture_manifest_sha256": stage_manifest["sha256"],
        "capture_manifest_matches_all_layer_stage": manifest == stage_manifest,
    }


def build_analysis(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "native_stage": args.native_stage,
        "public_stage": args.public_stage,
        "paired_stage": args.paired_stage,
        "native_candidate": args.native_candidate,
        "public_candidate": args.public_candidate,
        "preflight": args.preflight,
        "prompt_provenance": args.prompt_provenance,
        "certified_run": args.certified_run,
        "patch": args.patch,
    }
    reports: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        if label == "patch":
            raw = path.read_bytes()
            reports[label] = {"text": raw.decode("utf-8")}
            records[label] = {
                "path": str(path.resolve().relative_to(ROOT)),
                "size_bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }
        else:
            reports[label], records[label] = read_json(path)

    stage_identity = validate_paired_report_identities(
        reports["native_stage"], reports["public_stage"]
    )
    candidate_identity = validate_paired_report_identities(
        reports["native_candidate"], reports["public_candidate"]
    )
    for field in ("native_lens", "public_lens", "model", "runtime", "host"):
        require(
            stage_identity[field] == candidate_identity[field],
            f"candidate reports do not reuse the stage {field} identity",
        )
    native_items, public_items = validate_stage_pair(
        reports["native_stage"], reports["public_stage"]
    )
    native_groups, public_groups = validate_candidate_pair(
        reports["native_candidate"], reports["public_candidate"]
    )
    candidate_rows = candidate_layer_rows(native_groups, public_groups)

    provenance = reports["prompt_provenance"]
    require(provenance["provenance_id"] == EXPECTED_PROVENANCE_ID, "prompt provenance ID mismatch")
    reconstructed_stages = [
        {
            "id": item["id"],
            "text": item["prompt"],
            "token_ids": item["prompt_token_ids"],
            "metadata": item["metadata"],
        }
        for item in native_items
    ]
    native_candidate_items = list(
        experiments_by_id(reports["native_candidate"]).values()
    )
    reconstructed_candidates = [
        {
            "id": item["id"],
            "text": item["prompt"],
            "token_ids": item["prompt_token_ids"],
            "target_token_id": item["target_token_id_override"],
            "metadata": item["metadata"],
        }
        for item in native_candidate_items
    ]
    require(
        materializer_json_sha256(reconstructed_stages)
        == provenance["outputs"]["stage_prompts"]["sha256"],
        "stage report does not reconstruct the bound materializer bundle",
    )
    require(
        materializer_json_sha256(reconstructed_candidates)
        == provenance["outputs"]["candidate_prompts"]["sha256"],
        "candidate report does not reconstruct the bound materializer bundle",
    )
    paired = reports["paired_stage"]
    require(paired["pairing"]["prompt_count"] == 9, "paired stage prompt count mismatch")
    require(paired["pairing"]["observation_count"] == 567, "paired stage observation count mismatch")
    require(
        paired["input_files"]["native"]["sha256"] == records["native_stage"]["sha256"]
        and paired["input_files"]["public"]["sha256"] == records["public_stage"]["sha256"],
        "paired report does not bind the supplied stage reports",
    )
    recomputed_pair = compare_reports(
        reports["native_stage"], reports["public_stage"]
    )
    for field in (
        "schema_version",
        "scope",
        "inputs",
        "pairing",
        "adapter_certificates",
        "metrics",
    ):
        require(
            paired.get(field) == recomputed_pair.get(field),
            f"paired stage report is stale or invalid: {field}",
        )

    certified = reports["certified_run"]
    require(certified["result"] == "PASS", "certified run is not PASS")
    official = certified["official_harness"]
    require(
        (official["submitted"], official["completed"], official["resolved"], official["errors"])
        == (1, 1, 1, 0),
        "official SWE result is not 1/1 resolved",
    )
    require(
        records["patch"]["sha256"] == certified["task"]["patch_sha256"],
        "patch hash does not match certified run",
    )
    patch_text = reports["patch"]["text"]
    require("-                    if cotm is S.ComplexInfinity:" in patch_text, "patch lacks buggy line")
    require("+                    if cothm is S.ComplexInfinity:" in patch_text, "patch lacks corrected line")

    original_match_count = sum(
        item["generated_token_id"]
        == item["metadata"]["sampled_next"]["first_token_id"]
        for item in native_items
    )
    final_correct = native_groups[CORRECT_CANDIDATE][0]["final_model_readout"][0]
    final_buggy = native_groups[BUGGY_CANDIDATE][0]["final_model_readout"][0]
    final_candidate = {
        "correct_token": "co",
        "correct_rank": final_correct["target_rank"],
        "correct_logprob": final_correct["target_logprob"],
        "buggy_token": "cot",
        "buggy_rank": final_buggy["target_rank"],
        "buggy_logprob": final_buggy["target_logprob"],
        "correct_minus_buggy_logprob": (
            final_correct["target_logprob"] - final_buggy["target_logprob"]
        ),
    }

    full_sequence_layers = list(FIXED_MIDDLE_LAYERS) + [62]
    return {
        "schema_version": 1,
        "scope": "validated descriptive replay of frozen certified SWE contexts; not original hidden-state capture or causal intervention",
        "input_files": records,
        "official_outcome": {
            "run_name": certified["run_name"],
            "task": certified["task"]["instance_id"],
            "result": certified["result"],
            "submitted": official["submitted"],
            "completed": official["completed"],
            "resolved": official["resolved"],
            "errors": official["errors"],
            "patch_change": "cotm -> cothm",
            "patch_sha256": records["patch"]["sha256"],
        },
        "prompt_binding": {
            "provenance_id": EXPECTED_PROVENANCE_ID,
            "prompt_token_counts": list(EXPECTED_PROMPT_COUNTS),
            "stage_bundle_sha256": provenance["outputs"]["stage_prompts"]["sha256"],
            "candidate_bundle_sha256": provenance["outputs"]["candidate_prompts"]["sha256"],
        },
        "artifacts": {
            "model": reports["native_stage"]["model"],
            "native_lens": reports["native_stage"]["lens"],
            "public_lens": reports["public_stage"]["lens"],
            "runtime": reports["native_stage"]["runtime"],
        },
        "replay": {
            "stage_count": 9,
            "source_layers": list(EXPECTED_SOURCE_LAYERS),
            "target_layer": 63,
            "mtp_enabled": False,
            "original_sampled_first_token_match_count": original_match_count,
            "original_sampled_first_token_total": 9,
            "longest_context_preflight": validate_preflight(
                reports["preflight"],
                public_items[-1],
                reports["public_stage"],
                stage_identity,
            ),
        },
        "adapter_certificates": {
            "native_stage": adapter_summary(reports["native_stage"]),
            "public_stage": adapter_summary(reports["public_stage"]),
            "native_candidate": adapter_summary(reports["native_candidate"]),
            "public_candidate": adapter_summary(reports["public_candidate"]),
            "interpretation": "failed strict certificates are preserved independently from descriptive lens metrics",
        },
        "middle_layers": {
            "fixed_exploratory_reporting_layers": list(FIXED_MIDDLE_LAYERS),
            "selected_stage_top5_readouts": semantic_readouts(native_items, public_items),
            "native_public_pairing": middle_pair_metrics(native_items, public_items),
            "whole_depth_native_public": paired["metrics"]["overall"]["comparisons"]["native_vs_public_jacobian_lens"],
        },
        "candidate_probe": {
            "context": "teacher-forced original turn-3 reasoning after source inspection, ending after the opening backtick in 'the variable is actually `'",
            "correct_identifier": CORRECT_CANDIDATE,
            "correct_token_ids": [981, 337, 76],
            "buggy_identifier": BUGGY_CANDIDATE,
            "buggy_token_ids": [62317, 76],
            "primary_comparison": "same-context first token co versus cot",
            "native_fixed_middle": margin_summary(candidate_rows, "native_jacobian"),
            "public_fixed_middle": margin_summary(candidate_rows, "public_jacobian"),
            "logit_lens_fixed_middle": margin_summary(candidate_rows, "logit_lens"),
            "all_source_layer_rows": candidate_rows,
            "final_model_control": final_candidate,
            "secondary_unequal_bpe_sequence_totals": {
                "warning": "cothm has three BPE tokens and cotm has two; totals are not the primary middle-layer comparison",
                "native_jacobian": {
                    str(layer): sequence_totals(native_groups, "jacobian_lens", layer)
                    for layer in full_sequence_layers
                },
                "public_jacobian": {
                    str(layer): sequence_totals(public_groups, "jacobian_lens", layer)
                    for layer in full_sequence_layers
                },
            },
            "interpretation": "relative token preference only; both alternatives are outside the middle-layer top 10",
        },
        "limitations": [
            "exact frozen-token eager replay, not retained activations from the original compiled MTP run",
            "native lens uses an identity-STE surrogate and only ten 128-token WikiText fit prompts",
            "public lens fit-time precision and quantization are unpublished",
            "11.9K-15.7K code-agent contexts are out of distribution for the native fit corpus",
            "the task prompt already supplied the file, line, exception, and undefined spelling",
            "the ordinary logit lens exposes the candidate preference at least as consistently as either J-lens",
            "the fixed nine-layer slice was selected for exploratory reporting after the run, not preregistered",
            "the isolated preflight residual digest differs from the sequential all-stage request-9 digest",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-stage", type=Path, default=VALIDATION / "jlens-swe-qwen-code-native-nvfp4-ste-2026-07-17.json")
    parser.add_argument("--public-stage", type=Path, default=VALIDATION / "jlens-swe-qwen-code-public-2026-07-17.json")
    parser.add_argument("--paired-stage", type=Path, default=VALIDATION / "jlens-swe-qwen-code-native-vs-public-2026-07-17.json")
    parser.add_argument("--native-candidate", type=Path, default=VALIDATION / "jlens-swe-qwen-code-candidate-probe-2026-07-17.json")
    parser.add_argument("--public-candidate", type=Path, default=VALIDATION / "jlens-swe-qwen-code-candidate-probe-public-2026-07-17.json")
    parser.add_argument("--preflight", type=Path, default=VALIDATION / "jlens-swe-qwen-code-longest-preflight-2026-07-17.json")
    parser.add_argument("--prompt-provenance", type=Path, default=VALIDATION / "jlens-swe-qwen-code-prompt-provenance-2026-07-17.json")
    parser.add_argument("--certified-run", type=Path, default=VALIDATION / "2026-07-15-publication-certified.json")
    parser.add_argument("--patch", type=Path, default=VALIDATION / "sympy__sympy-13480.patch")
    parser.add_argument("--output", type=Path, default=VALIDATION / "jlens-swe-qwen-code-analysis-2026-07-17.json")
    parser.add_argument("--check", action="store_true", help="compare the computed object with --output without writing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = build_analysis(args)
    if args.check:
        published, _ = read_json(args.output)
        require(published == analysis, f"published analysis is stale: {args.output}")
        print(f"SWE J-lens analysis passed: {args.output}")
        return
    atomic_write_json(args.output, analysis)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
