#!/usr/bin/env python3
"""Select event-aligned semantic probes from the certified SWE trajectory."""

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
DEFAULT_CONFIG = ROOT / "configs/swe_semantic_probes.json"
DEFAULT_TRAJECTORY = (
    ROOT / ".cache/swe_jlens_trajectory/swe_jlens_trajectory_prompts.json"
)
DEFAULT_OUTPUT = ROOT / ".cache/swe_jlens_semantic/prompts.json"
MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"


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


def validate_config(
    value: Mapping[str, Any], *, tokenizer: Any | None = None
) -> tuple[list[Mapping[str, Any]], tuple[int, ...]]:
    if value.get("schema_version") != 1:
        raise ValueError("semantic probe config schema must be 1")
    model = require_mapping(value.get("model"), "config model")
    if model.get("repo_id") != MODEL_REPO or model.get("revision") != MODEL_REVISION:
        raise ValueError("semantic probe config does not pin the expected model")
    raw_probes = value.get("probes")
    if not isinstance(raw_probes, list) or not raw_probes:
        raise ValueError("semantic probe config requires a nonempty probe list")

    probes: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    seen_points: set[tuple[int, int]] = set()
    all_token_ids: set[int] = set()
    for ordinal, raw_probe in enumerate(raw_probes, 1):
        probe = require_mapping(raw_probe, f"probe {ordinal}")
        probe_id = probe.get("id")
        request_index = probe.get("request_index")
        offset = probe.get("offset")
        if not isinstance(probe_id, str) or not probe_id or probe_id in seen_ids:
            raise ValueError(f"probe {ordinal} has an invalid or duplicate id")
        if (
            isinstance(request_index, bool)
            or not isinstance(request_index, int)
            or not 1 <= request_index <= 9
        ):
            raise ValueError(f"probe {probe_id} has an invalid request index")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(f"probe {probe_id} has an invalid offset")
        point = (request_index, offset)
        if point in seen_points:
            raise ValueError(f"duplicate semantic probe point {point}")
        seen_ids.add(probe_id)
        seen_points.add(point)

        group_ids: dict[str, set[int]] = {}
        for group in ("positive", "negative"):
            raw_records = probe.get(group)
            if not isinstance(raw_records, list) or not raw_records:
                raise ValueError(f"probe {probe_id} requires {group} tokens")
            ids: set[int] = set()
            for record_index, raw_record in enumerate(raw_records, 1):
                record = require_mapping(
                    raw_record, f"probe {probe_id} {group} token {record_index}"
                )
                text = record.get("text")
                token_id = record.get("token_id")
                if not isinstance(text, str) or not text:
                    raise ValueError(f"probe {probe_id} has an invalid token text")
                if (
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    or token_id in ids
                ):
                    raise ValueError(f"probe {probe_id} has invalid token IDs")
                if tokenizer is not None:
                    encoded = tokenizer.encode(text, add_special_tokens=False)
                    decoded = tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    if encoded != [token_id] or decoded != text:
                        raise ValueError(
                            f"probe {probe_id} token pin changed for {text!r}: "
                            f"encoded={encoded}, decoded={decoded!r}"
                        )
                ids.add(token_id)
                all_token_ids.add(token_id)
            group_ids[group] = ids
        if group_ids["positive"] & group_ids["negative"]:
            raise ValueError(f"probe {probe_id} has overlapping contrast tokens")
        probes.append(probe)
    return probes, tuple(sorted(all_token_ids))


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
    probes, scored_token_ids = validate_config(config)
    by_point: dict[tuple[int, int], Mapping[str, Any]] = {}
    for prompt in trajectory_prompts:
        point = trajectory_point(prompt)
        if point in by_point:
            raise ValueError(f"trajectory contains duplicate point {point}")
        by_point[point] = prompt

    selected: list[dict[str, Any]] = []
    for probe in probes:
        point = (probe["request_index"], probe["offset"])
        source = by_point.get(point)
        if source is None:
            raise ValueError(f"trajectory does not contain semantic probe point {point}")
        prompt = copy.deepcopy(dict(source))
        metadata = require_mapping(prompt.get("metadata"), "selected metadata")
        prompt["id"] = f"swe-semantic-{probe['id']}"
        metadata["semantic_probe"] = {
            "id": probe["id"],
            "state": probe.get("state"),
            "positive_token_ids": [record["token_id"] for record in probe["positive"]],
            "negative_token_ids": [record["token_id"] for record in probe["negative"]],
            "config_sha256": config_sha256,
            "trajectory_bundle_sha256": trajectory_sha256,
        }
        selected.append(prompt)
    summary = {
        "schema_version": 1,
        "kind": "swe_verified_event_semantic_probe_bundle",
        "model": copy.deepcopy(config["model"]),
        "task": config.get("task"),
        "primary_layers": copy.deepcopy(config.get("primary_layers")),
        "selection_note": config.get("selection_note"),
        "probe_count": len(selected),
        "probe_ids": [probe["id"] for probe in probes],
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
        json.loads(config_path.read_text(encoding="utf-8")), "semantic config"
    )
    if not args.skip_tokenizer_check:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        snapshot = snapshot_download(
            MODEL_REPO, revision=MODEL_REVISION, local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        validate_config(config, tokenizer=tokenizer)
    trajectory_value = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(trajectory_value, list) or not trajectory_value:
        raise ValueError("trajectory prompt bundle must be a nonempty list")
    bundle, summary = build_probe_bundle(
        trajectory_value,
        config,
        config_sha256=sha256_file(config_path),
        trajectory_sha256=sha256_file(trajectory_path),
    )
    atomic_write_json(output_path, bundle)
    summary["output_path"] = str(output_path)
    summary["output_sha256"] = sha256_file(output_path)
    atomic_write_json(summary_path, summary)
    print(
        f"wrote {len(bundle)} probes to {output_path}; "
        f"score IDs with --score-token-ids "
        f"{','.join(str(token_id) for token_id in summary['scored_token_ids'])}"
    )
    print(f"wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
