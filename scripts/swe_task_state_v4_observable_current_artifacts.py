#!/usr/bin/env python3
"""Authenticate and publish the sole current V4 observable artifact routing set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

try:
    from scripts import swe_task_state_v4_observable_decoder as DECODER
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    import swe_task_state_v4_observable_decoder as DECODER  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_observable_current_artifacts.json"
)
KIND = "swe_task_state_v4_observable_current_artifact_index"
STATUS_SCOPE = "artifact_authentication_and_current_routing_only_not_a_scientific_gate"
HISTORICAL_IDS = [
    "observable_feature_bundle_v2",
    "observable_feature_bundle_v2b",
    "observable_feature_bundle_v2c",
    "observable_feature_bundle_v2d",
    "observable_feature_bundle_v2e",
    "observable_rationale_marker_report_v2",
    "observable_rationale_marker_report_v2e",
    "observable_rationale_marker_nested_inference_v1",
]


class ArtifactIndexError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactIndexError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_record(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    record = dict(value)
    _require(
        set(record) == {"path", "sha256", "size_bytes"}
        and isinstance(record["path"], str)
        and bool(record["path"])
        and not Path(record["path"]).is_absolute()
        and _is_sha256(record["sha256"])
        and isinstance(record["size_bytes"], int)
        and not isinstance(record["size_bytes"], bool)
        and record["size_bytes"] > 0,
        f"{label} byte record is invalid",
    )
    DECODER.frozen_lexical_path_preflight([Path(record["path"])])
    return record


def validate_config(value: Any) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "current-artifact config must be an object")
    config = dict(value)
    _require(
        set(config)
        == {
            "schema_version",
            "id",
            "status",
            "status_scope",
            "path_guard",
            "current",
            "historical_superseded",
            "routing_policy",
        }
        and config["schema_version"] == 1
        and config["id"] == "swe-task-state-v4-observable-current-artifact-index"
        and config["status"] == "development_only_artifact_routing"
        and config["status_scope"] == STATUS_SCOPE
        and config["path_guard"]
        == {"forbidden_path_fragments": ["reserved", "validation"]},
        "current-artifact config identity changed",
    )
    current = config["current"]
    _require(
        isinstance(current, Mapping)
        and set(current) == {"feature_bundle", "decoder_report", "nested_inference"}
        and isinstance(current["feature_bundle"], Mapping)
        and set(current["feature_bundle"]) == {"manifest", "data"},
        "current artifact routing changed",
    )
    _validate_record(current["feature_bundle"]["manifest"], "current bundle manifest")
    _validate_record(current["feature_bundle"]["data"], "current bundle data")
    _validate_record(current["decoder_report"], "current decoder report")
    _validate_record(current["nested_inference"], "current nested inference")
    _require(
        current["feature_bundle"]["manifest"]["path"].endswith("-v2f.json")
        and current["feature_bundle"]["data"]["path"].endswith("-v2f.npz")
        and current["decoder_report"]["path"].endswith("-v2f.json")
        and current["nested_inference"]["path"].endswith("-v2.json"),
        "current artifact versions changed",
    )
    historical = config["historical_superseded"]
    _require(
        isinstance(historical, list)
        and [item.get("artifact_id") for item in historical if isinstance(item, Mapping)]
        == HISTORICAL_IDS,
        "historical artifact order or registry changed",
    )
    for position, value_item in enumerate(historical):
        _require(isinstance(value_item, Mapping), f"historical item {position} invalid")
        item = dict(value_item)
        common = {"artifact_id", "kind", "status", "superseded_by"}
        _require(
            item["status"] == "historical_superseded"
            and item["superseded_by"]
            in {
                "current.feature_bundle",
                "current.decoder_report",
                "current.nested_inference",
            },
            f"historical item {position} routing changed",
        )
        if item["kind"] == "feature_bundle":
            _require(set(item) == common | {"manifest", "data"}, "bundle history schema changed")
            _validate_record(item["manifest"], f"historical {item['artifact_id']} manifest")
            _validate_record(item["data"], f"historical {item['artifact_id']} data")
        else:
            _require(
                item["kind"] in {"decoder_report", "nested_inference"}
                and set(item) == common | {"artifact"},
                f"historical item {position} schema changed",
            )
            _validate_record(item["artifact"], f"historical {item['artifact_id']}")
    _require(
        config["routing_policy"]
        == {
            "current_consumers_must_use_only_current_records": True,
            "historical_records_are_not_current_fallbacks": True,
            "historical_artifacts_deleted": False,
            "reserved_validation_access_authorized": False,
        },
        "current artifact routing policy changed",
    )
    return config


def _flatten_records(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    current = config["current"]
    records = {
        "current_feature_bundle_manifest": dict(current["feature_bundle"]["manifest"]),
        "current_feature_bundle_data": dict(current["feature_bundle"]["data"]),
        "current_decoder_report": dict(current["decoder_report"]),
        "current_nested_inference": dict(current["nested_inference"]),
    }
    for item in config["historical_superseded"]:
        prefix = f"historical_{item['artifact_id']}"
        if item["kind"] == "feature_bundle":
            records[f"{prefix}_manifest"] = dict(item["manifest"])
            records[f"{prefix}_data"] = dict(item["data"])
        else:
            records[prefix] = dict(item["artifact"])
    return records


def authenticate_records(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[Path, dict[str, Any]]]:
    paths = [ROOT / record["path"] for record in records.values()]
    DECODER.frozen_lexical_path_preflight(paths)
    DECODER.frozen_canonical_path_preflight(input_paths=paths, output_paths=[])
    bindings: dict[str, tuple[Path, dict[str, Any]]] = {}
    for (label, expected_value), path in zip(records.items(), paths, strict=True):
        expected = dict(expected_value)
        observed = DECODER._artifact_byte_record(path, label=label)
        _require(observed == expected, f"authenticated artifact changed: {label}")
        bindings[label] = (path.resolve(strict=True), expected)
    return bindings


def validate_current_chain(config: Mapping[str, Any]) -> None:
    current = config["current"]
    bundle = DECODER.load_json(ROOT / current["feature_bundle"]["manifest"]["path"])
    decoder_report = DECODER.load_json(ROOT / current["decoder_report"]["path"])
    nested = DECODER.load_json(ROOT / current["nested_inference"]["path"])
    bundle_data = current["feature_bundle"]["data"]
    _require(
        bundle.get("kind") == DECODER.FEATURE_BUNDLE_KIND
        and bundle.get("data", {}).get("sha256") == bundle_data["sha256"]
        and bundle.get("data", {}).get("size_bytes") == bundle_data["size_bytes"]
        and bundle.get("data", {}).get("path") == Path(bundle_data["path"]).name,
        "current bundle manifest/data chain changed",
    )
    report_inputs = decoder_report.get("inputs", {})
    _require(
        decoder_report.get("kind") == DECODER.REPORT_KIND
        and decoder_report.get("status_scope") == DECODER.STATUS_SCOPE
        and report_inputs.get("feature_manifest")
        == current["feature_bundle"]["manifest"]
        and report_inputs.get("feature_data") == bundle_data,
        "current decoder/bundle chain changed",
    )
    nested_input = nested.get("inputs", {}).get("decoder_report", {})
    _require(
        nested.get("kind")
        == "swe_task_state_v4_observable_nested_fixed_oof_repository_inference_report"
        and nested_input.get("sha256") == current["decoder_report"]["sha256"]
        and nested_input.get("size_bytes") == current["decoder_report"]["size_bytes"]
        and Path(str(nested_input.get("path"))).resolve(strict=True)
        == (ROOT / current["decoder_report"]["path"]).resolve(strict=True),
        "current nested-inference/decoder chain changed",
    )


def _write_no_clobber(
    path: Path, value: Any, *, before_publish: Any
) -> None:
    _require(not path.exists() and not path.is_symlink(), "artifact index output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(path.parent.is_dir() and not path.parent.is_symlink(), "index parent invalid")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        before_publish()
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise ArtifactIndexError("artifact index output exists") from error
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def run(args: argparse.Namespace) -> int:
    DECODER.frozen_lexical_path_preflight([args.config, args.output])
    DECODER.frozen_canonical_path_preflight(
        input_paths=[args.config], output_paths=[args.output]
    )
    config_path = args.config.resolve(strict=True)
    _require(config_path == CONFIG_PATH, "artifact-index config path changed")
    config_bytes, config_record = DECODER._read_startup_bytes(
        config_path, label="artifact-index config"
    )
    _script_bytes, implementation_record = DECODER._read_startup_bytes(
        SCRIPT_PATH, label="artifact-index implementation"
    )
    config = validate_config(
        DECODER._load_json_bytes(config_bytes, label="current-artifact config")
    )
    bindings: dict[str, tuple[Path, Mapping[str, Any]]] = {
        "index_config": (config_path, config_record),
        "index_implementation": (SCRIPT_PATH, implementation_record),
        **authenticate_records(_flatten_records(config)),
    }
    validate_current_chain(config)
    report = {
        "schema_version": 1,
        "kind": KIND,
        "status": "passed",
        "status_scope": STATUS_SCOPE,
        "config": config_record,
        "implementation": implementation_record,
        "authentication": {
            "all_current_and_historical_records_authenticated": True,
            "all_records_rehashed_after_serialization_immediately_before_atomic_publish": True,
            "publication": "same_directory_fsynced_temporary_then_atomic_hard_link_no_clobber",
            "authenticated_record_count": len(bindings),
            "current_chain_bindings_verified": True,
            "reserved_validation_access_authorized": False,
        },
        "current": config["current"],
        "historical_superseded": config["historical_superseded"],
        "routing_policy": config["routing_policy"],
    }
    output = Path(os.path.abspath(os.fspath(args.output)))
    _require(not output.is_symlink(), "artifact index output must not be a symlink")
    _write_no_clobber(
        output,
        report,
        before_publish=lambda: DECODER._assert_byte_records_unchanged(bindings),
    )
    print(
        f"wrote current observable artifact index with "
        f"{len(config['historical_superseded'])} historical entries to {output}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    return parser


__all__ = [
    "ArtifactIndexError",
    "CONFIG_PATH",
    "HISTORICAL_IDS",
    "KIND",
    "STATUS_SCOPE",
    "authenticate_records",
    "run",
    "validate_config",
    "validate_current_chain",
]


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
