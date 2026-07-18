#!/usr/bin/env python3
"""Freeze method-aligned intermediate-concept probes from the SWE trajectory."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/swe_intermediate_concept_probes.json"
DEFAULT_TRAJECTORY = (
    ROOT / ".cache/swe_jlens_trajectory/swe_jlens_trajectory_prompts.json"
)
DEFAULT_OUTPUT = ROOT / ".cache/swe_jlens_intermediate/prompts.json"
MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
TOKENIZER_JSON_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
EXPECTED_LAYERS = tuple(range(16, 48))
EXPECTED_TRAJECTORY_COUNT = 293


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
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


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def validate_config(
    value: Mapping[str, Any], *, tokenizer: Any | None = None
) -> tuple[list[Mapping[str, Any]], tuple[int, ...], tuple[int, ...]]:
    if value.get("schema_version") != 1:
        raise ValueError("intermediate probe config schema must be 1")
    if value.get("kind") != "swe_verified_intermediate_concept_eval":
        raise ValueError("unexpected intermediate probe config kind")

    adaptation = require_mapping(value.get("adaptation"), "config adaptation")
    if adaptation.get("status") != "exploratory_one_task_adaptation":
        raise ValueError("config must label the evaluation as a one-task adaptation")
    if adaptation.get("lens_outputs_used_for_selection") is not False:
        raise ValueError("probe selection must be independent of lens outputs")

    model = require_mapping(value.get("model"), "config model")
    if model.get("repo_id") != MODEL_REPO or model.get("revision") != MODEL_REVISION:
        raise ValueError("intermediate probe config does not pin the expected model")
    if model.get("tokenizer_json_sha256") != TOKENIZER_JSON_SHA256:
        raise ValueError("intermediate probe config has the wrong tokenizer hash")

    source = require_mapping(value.get("source"), "config source")
    require_sha256(source.get("trajectory_bundle_sha256"), "trajectory bundle hash")
    require_sha256(source.get("trace_sha256"), "trace hash")
    require_sha256(source.get("dataset_sha256"), "dataset hash")
    require_sha256(source.get("prompt_provenance_id"), "prompt provenance ID")
    if source.get("trajectory_prompt_count") != EXPECTED_TRAJECTORY_COUNT:
        raise ValueError("config has the wrong source trajectory prompt count")

    band = require_mapping(value.get("middle_band"), "config middle band")
    raw_layers = band.get("layers")
    if not isinstance(raw_layers, list) or tuple(raw_layers) != EXPECTED_LAYERS:
        raise ValueError("middle band must be the fixed contiguous layers 16 through 47")
    if band.get("fixed_before_scoring") is not True:
        raise ValueError("middle band must be frozen before scoring")

    metric = require_mapping(value.get("metric"), "config metric")
    if metric.get("name") != "intermediate_pass_at_k":
        raise ValueError("unexpected intermediate probe metric")
    if metric.get("accepted_target_token_scored") is not False:
        raise ValueError("the accepted next-token target must not be scored")
    pass_at_k = metric.get("pass_at_k")
    if (
        not isinstance(pass_at_k, list)
        or not pass_at_k
        or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in pass_at_k)
        or pass_at_k != sorted(set(pass_at_k))
    ):
        raise ValueError("pass_at_k must be a sorted unique list of positive integers")

    raw_items = value.get("items")
    if not isinstance(raw_items, list) or not 8 <= len(raw_items) <= 12:
        raise ValueError("config requires 8 through 12 frozen items")

    items: list[Mapping[str, Any]] = []
    seen_item_ids: set[str] = set()
    seen_points: set[tuple[int, int]] = set()
    seen_intermediate_keys: set[str] = set()
    token_text_by_id: dict[int, str] = {}
    token_id_by_text: dict[str, int] = {}
    scored_token_ids: set[int] = set()
    for ordinal, raw_item in enumerate(raw_items, 1):
        item = require_mapping(raw_item, f"item {ordinal}")
        item_id = item.get("id")
        request_index = item.get("request_index")
        offset = item.get("offset")
        if not isinstance(item_id, str) or not item_id or item_id in seen_item_ids:
            raise ValueError(f"item {ordinal} has an invalid or duplicate ID")
        if (
            isinstance(request_index, bool)
            or not isinstance(request_index, int)
            or not 1 <= request_index <= 9
        ):
            raise ValueError(f"item {item_id} has an invalid request index")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(f"item {item_id} has an invalid offset")
        point = (request_index, offset)
        if point in seen_points:
            raise ValueError(f"duplicate intermediate probe point {point}")
        seen_item_ids.add(item_id)
        seen_points.add(point)
        require_sha256(item.get("request_sha256"), f"item {item_id} request hash")
        for field in ("event_family", "state", "rationale", "leakage_class"):
            if not isinstance(item.get(field), str) or not item[field]:
                raise ValueError(f"item {item_id} requires {field}")

        raw_intermediates = item.get("intermediates")
        if not isinstance(raw_intermediates, list) or not raw_intermediates:
            raise ValueError(f"item {item_id} requires semantic intermediates")
        item_keys: set[str] = set()
        for intermediate_ordinal, raw_intermediate in enumerate(raw_intermediates, 1):
            intermediate = require_mapping(
                raw_intermediate,
                f"item {item_id} intermediate {intermediate_ordinal}",
            )
            key = intermediate.get("key")
            if (
                not isinstance(key, str)
                or not key
                or key in item_keys
                or key in seen_intermediate_keys
            ):
                raise ValueError(f"item {item_id} has an invalid or duplicate concept key")
            item_keys.add(key)
            seen_intermediate_keys.add(key)
            raw_forms = intermediate.get("forms")
            if not isinstance(raw_forms, list) or not raw_forms:
                raise ValueError(f"intermediate {key} requires token forms")
            form_ids: set[int] = set()
            form_texts: set[str] = set()
            for form_ordinal, raw_form in enumerate(raw_forms, 1):
                form = require_mapping(raw_form, f"intermediate {key} form {form_ordinal}")
                text = form.get("text")
                token_id = form.get("token_id")
                if not isinstance(text, str) or not text or not text.startswith(" "):
                    raise ValueError(f"intermediate {key} has an invalid token surface")
                if (
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    or token_id in form_ids
                    or text in form_texts
                ):
                    raise ValueError(f"intermediate {key} has duplicate or invalid forms")
                previous_text = token_text_by_id.setdefault(token_id, text)
                if previous_text != text:
                    raise ValueError(
                        f"token ID {token_id} is pinned to both {previous_text!r} and {text!r}"
                    )
                previous_id = token_id_by_text.setdefault(text, token_id)
                if previous_id != token_id:
                    raise ValueError(
                        f"token surface {text!r} is pinned to both {previous_id} and {token_id}"
                    )
                if tokenizer is not None:
                    encoded = tokenizer.encode(text, add_special_tokens=False)
                    decoded = tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    if encoded != [token_id] or decoded != text:
                        raise ValueError(
                            f"token pin changed for {text!r}: "
                            f"encoded={encoded}, decoded={decoded!r}"
                        )
                form_ids.add(token_id)
                form_texts.add(text)
                scored_token_ids.add(token_id)

        raw_evidence = item.get("evidence")
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise ValueError(f"item {item_id} requires grounding evidence")
        supported: set[str] = set()
        for evidence_ordinal, raw_record in enumerate(raw_evidence, 1):
            record = require_mapping(
                raw_record, f"item {item_id} evidence {evidence_ordinal}"
            )
            require_sha256(
                record.get("content_sha256"),
                f"item {item_id} evidence content hash",
            )
            if "source_file_sha256" in record:
                require_sha256(
                    record.get("source_file_sha256"),
                    f"item {item_id} evidence source hash",
                )
            supports = record.get("supports")
            if (
                not isinstance(supports, list)
                or not supports
                or any(not isinstance(key, str) for key in supports)
            ):
                raise ValueError(f"item {item_id} evidence requires concept supports")
            if not set(supports) <= item_keys:
                raise ValueError(f"item {item_id} evidence names an unknown concept")
            supported.update(supports)
        if supported != item_keys:
            missing = sorted(item_keys - supported)
            raise ValueError(f"item {item_id} has ungrounded concepts: {missing}")
        items.append(item)

    return items, tuple(sorted(scored_token_ids)), tuple(pass_at_k)


def trajectory_point(prompt: Mapping[str, Any]) -> tuple[int, int]:
    metadata = require_mapping(prompt.get("metadata"), "trajectory metadata")
    trajectory = require_mapping(metadata.get("trajectory"), "trajectory coordinates")
    request_index = trajectory.get("request_index", metadata.get("request_index"))
    offset = trajectory.get("offset")
    if (
        isinstance(request_index, bool)
        or not isinstance(request_index, int)
        or isinstance(offset, bool)
        or not isinstance(offset, int)
    ):
        raise ValueError("trajectory prompt has invalid request/offset coordinates")
    return request_index, offset


def build_probe_bundle(
    trajectory_prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    trajectory_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items, scored_token_ids, pass_at_k = validate_config(config)
    source_config = require_mapping(config.get("source"), "config source")
    if trajectory_sha256 != source_config.get("trajectory_bundle_sha256"):
        raise ValueError("source trajectory bundle hash does not match the frozen config")
    if len(trajectory_prompts) != source_config.get("trajectory_prompt_count"):
        raise ValueError("source trajectory prompt count does not match the frozen config")

    by_point: dict[tuple[int, int], Mapping[str, Any]] = {}
    for prompt in trajectory_prompts:
        point = trajectory_point(prompt)
        if point in by_point:
            raise ValueError(f"trajectory contains duplicate point {point}")
        by_point[point] = prompt

    selected: list[dict[str, Any]] = []
    for item in items:
        point = (item["request_index"], item["offset"])
        source_prompt = by_point.get(point)
        if source_prompt is None:
            raise ValueError(f"trajectory does not contain intermediate probe point {point}")
        source_metadata = require_mapping(source_prompt.get("metadata"), "selected metadata")
        source_hashes = require_mapping(source_metadata.get("source_hashes"), "source hashes")
        source_trajectory = require_mapping(
            source_metadata.get("trajectory"), "source trajectory"
        )
        if source_hashes.get("request_sha256") != item.get("request_sha256"):
            raise ValueError(f"item {item['id']} request hash does not match the trajectory")
        if source_hashes.get("trace_sha256") != source_config.get("trace_sha256"):
            raise ValueError(f"item {item['id']} trace hash does not match the config")
        if source_hashes.get("tokenizer_json_sha256") != TOKENIZER_JSON_SHA256:
            raise ValueError(f"item {item['id']} tokenizer hash does not match the config")
        if source_metadata.get("provenance_id") != source_config.get("prompt_provenance_id"):
            raise ValueError(f"item {item['id']} provenance ID does not match the config")
        accepted_target = source_trajectory.get("target_token_id")
        if accepted_target != source_prompt.get("target_token_id"):
            raise ValueError(f"item {item['id']} has inconsistent accepted targets")
        if accepted_target in scored_token_ids:
            raise ValueError(
                f"accepted target token {accepted_target} is included in scored forms"
            )

        prompt = copy.deepcopy(dict(source_prompt))
        metadata = prompt["metadata"]
        prompt["id"] = f"swe-intermediate-{item['id']}"
        metadata["intermediate_probe"] = {
            "id": item["id"],
            "event_family": item["event_family"],
            "state": item["state"],
            "rationale": item["rationale"],
            "leakage_class": item["leakage_class"],
            "evidence": copy.deepcopy(item["evidence"]),
            "intermediates": copy.deepcopy(item["intermediates"]),
            "middle_band_layers": list(EXPECTED_LAYERS),
            "metric": {
                "name": config["metric"]["name"],
                "pass_at_k": list(pass_at_k),
                "accepted_target_token_scored": False,
            },
            "adaptation_status": config["adaptation"]["status"],
            "lens_outputs_used_for_selection": False,
            "config_sha256": config_sha256,
            "trajectory_bundle_sha256": trajectory_sha256,
        }
        selected.append(prompt)

    summary = {
        "schema_version": 1,
        "kind": "swe_verified_intermediate_concept_probe_bundle",
        "adaptation_status": config["adaptation"]["status"],
        "lens_outputs_used_for_selection": False,
        "model": copy.deepcopy(config["model"]),
        "task": copy.deepcopy(config["task"]),
        "middle_band_layers": list(EXPECTED_LAYERS),
        "pass_at_k": list(pass_at_k),
        "item_count": len(selected),
        "intermediate_count": sum(len(item["intermediates"]) for item in items),
        "item_ids": [item["id"] for item in items],
        "trajectory_points": [
            {"request_index": item["request_index"], "offset": item["offset"]}
            for item in items
        ],
        "scored_token_ids": list(scored_token_ids),
        "config_sha256": config_sha256,
        "trajectory_bundle_sha256": trajectory_sha256,
    }
    return selected, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trajectory-prompts", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--skip-tokenizer-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    trajectory_path = args.trajectory_prompts.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    summary_path = (
        args.summary_output.expanduser().resolve()
        if args.summary_output
        else output_path.with_name(f"{output_path.stem}.summary.json")
    )
    config = require_mapping(
        json.loads(config_path.read_text(encoding="utf-8")), "intermediate config"
    )
    if not args.skip_tokenizer_check:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        snapshot = Path(
            snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)
        )
        if sha256_file(snapshot / "tokenizer.json") != TOKENIZER_JSON_SHA256:
            raise ValueError("cached tokenizer.json does not match the frozen hash")
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        validate_config(config, tokenizer=tokenizer)

    trajectory_value = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(trajectory_value, list) or not trajectory_value:
        raise ValueError("trajectory prompt bundle must be a nonempty list")
    config_hash = sha256_file(config_path)
    trajectory_hash = sha256_file(trajectory_path)
    bundle, summary = build_probe_bundle(
        trajectory_value,
        config,
        config_sha256=config_hash,
        trajectory_sha256=trajectory_hash,
    )
    atomic_write_json(output_path, bundle)
    summary["output_path"] = str(output_path)
    summary["output_sha256"] = sha256_file(output_path)
    atomic_write_json(summary_path, summary)
    print(
        f"wrote {len(bundle)} items / {summary['intermediate_count']} intermediates "
        f"to {output_path}"
    )
    print(
        "score IDs with --score-token-ids "
        + ",".join(str(token_id) for token_id in summary["scored_token_ids"])
    )
    print(f"wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
