#!/usr/bin/env python3
"""Assemble four authenticated projection chunks into label-free causal tensors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURE_CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_activation_features.json"
FEATURE_MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_activation_features.py"
PROJECTION_CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_activation_projection.json"
)
PROJECTION_MODULE_PATH = (
    ROOT / "scripts" / "swe_task_state_v4_activation_projection.py"
)
SCRIPT_PATH = Path(__file__).resolve()
KIND = "swe_task_state_v4_label_free_activation_feature_campaign"
SCHEMA_VERSION = 1


def _import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_features = _import_module(
    "swe_task_state_v4_activation_features_for_campaign", FEATURE_MODULE_PATH
)
_projection = _import_module(
    "swe_task_state_v4_activation_projection_for_campaign", PROJECTION_MODULE_PATH
)


class CampaignError(ValueError):
    """Raised when campaign authentication or label-free boundaries fail closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CampaignError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise CampaignError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CampaignError(f"cannot resolve {label}: {path}: {exc}") from exc
    if not resolved.is_file():
        raise CampaignError(f"{label} must be a regular file: {path}")
    return resolved


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _display_path(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _logical_array_sha256(name: str, array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    header = canonical_json_bytes(
        {"name": name, "shape": list(values.shape), "dtype": values.dtype.str}
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _string_sequence_sha256(name: str, values: Sequence[str]) -> str:
    header = canonical_json_bytes(
        {"name": name, "count": len(values), "encoding": "ascii-length-prefixed"}
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_binding_record(
    value: Any,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
        raise CampaignError(f"{label} binding schema changed")
    if not isinstance(value["path"], str) or not value["path"]:
        raise CampaignError(f"{label} binding path is invalid")
    observed = _require_sha256(value["sha256"], f"{label} binding sha256")
    if expected_sha256 is not None and observed != expected_sha256:
        raise CampaignError(f"{label} binding SHA-256 changed")
    if (
        isinstance(value["size_bytes"], bool)
        or not isinstance(value["size_bytes"], int)
        or value["size_bytes"] < 1
    ):
        raise CampaignError(f"{label} binding size is invalid")
    if expected_size_bytes is not None and value["size_bytes"] != expected_size_bytes:
        raise CampaignError(f"{label} binding size changed")
    return value


def _validate_alignment_metadata(
    index: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any],
) -> None:
    required = campaign["required_alignment"]
    if (
        index.get("kind") != required["kind"]
        or index.get("schema_version") != required["schema_version"]
        or index.get("scope") != required["scope"]
    ):
        raise CampaignError("alignment identity changed")
    if index.get("feature_use") != {
        "allowed": [
            "task-local ordering for causal temporal transforms",
            "repository and task grouping for held-out splits and weights",
            "stable eligibility filtering",
        ],
        "forbidden": [
            "hashing or one-hot encoding IDs as model features",
            "repository or request index as semantic model features",
        ],
    }:
        raise CampaignError("alignment feature-use boundary changed")
    sources = index.get("sources")
    expected_counts = campaign["projection_chunk_boundary_counts"]
    if not isinstance(sources, list) or len(sources) != len(expected_counts):
        raise CampaignError("alignment source partition changed")
    for chunk_index, (source, expected_count, expected_sha256) in enumerate(
        zip(
            sources,
            expected_counts,
            required["source_sha256s"],
            strict=True,
        )
    ):
        if not isinstance(source, dict) or set(source) != {"path", "sha256", "row_count"}:
            raise CampaignError(f"alignment source {chunk_index} schema changed")
        if (
            not isinstance(source["path"], str)
            or not source["path"]
            or _require_sha256(
                source["sha256"], f"alignment source {chunk_index} sha256"
            )
            != expected_sha256
            or source["row_count"] != expected_count
        ):
            raise CampaignError(f"alignment source {chunk_index} changed")
    for label, expected_sha256 in (
        ("config", required["config_sha256"]),
        ("implementation", required["implementation_sha256"]),
    ):
        value = index.get(label)
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise CampaignError(f"alignment {label} binding schema changed")
        if not isinstance(value["path"], str) or not value["path"]:
            raise CampaignError(f"alignment {label} path is invalid")
        if _require_sha256(
            value["sha256"], f"alignment {label} sha256"
        ) != expected_sha256:
            raise CampaignError(f"alignment {label} SHA-256 changed")
    eligibility = index.get("eligibility_source")
    if not isinstance(eligibility, dict) or set(eligibility) != {
        "path",
        "sha256",
        "all_rows",
        "stable_rows",
        "numerically_unstable_rows",
    }:
        raise CampaignError("alignment eligibility binding schema changed")
    if _require_sha256(
        eligibility["sha256"], "alignment eligibility sha256"
    ) != required["eligibility_source_sha256"]:
        raise CampaignError("alignment eligibility SHA-256 changed")
    if (
        not isinstance(eligibility["path"], str)
        or not eligibility["path"]
        or eligibility["all_rows"] != campaign["total_boundary_count"]
        or eligibility["stable_rows"] != campaign["stable_feature_count"]
        or eligibility["numerically_unstable_rows"]
        != campaign["total_boundary_count"] - campaign["stable_feature_count"]
    ):
        raise CampaignError("alignment eligibility counts changed")


def _validate_projection_manifest(
    manifest: Any,
    *,
    expected_count: int,
    expected_projection_config_sha256: str,
    expected_projection_implementation_sha256: str,
    projection_config: Mapping[str, Any],
    expected_mapping_sha256s: Sequence[str],
) -> None:
    expected_keys = {
        "schema_version",
        "kind",
        "status",
        "projection_config",
        "capture_config",
        "implementation",
        "pre_and_post_projection_bindings_equal",
        "input_manifest",
        "mapping_sha256s",
        "boundary_count",
        "sketch_shape",
        "sketches_logical_sha256",
        "band_statistics_logical_sha256",
        "data",
        "sources",
        "bands",
        "projection",
        "band_statistics",
        "downstream_feature_boundary",
        "claim_scope",
        "forbidden_path_guard_passed",
        "reserved_validation_access_authorized",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise CampaignError("projection manifest top-level allowlist changed")
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["kind"] != "swe_task_state_v4_label_independent_activation_projection"
        or manifest["status"] != "passed"
        or manifest["pre_and_post_projection_bindings_equal"] is not True
        or manifest["forbidden_path_guard_passed"] is not True
        or manifest["reserved_validation_access_authorized"] is not False
    ):
        raise CampaignError("projection manifest identity or status changed")
    _validate_binding_record(
        manifest["projection_config"],
        label="projection config",
        expected_sha256=expected_projection_config_sha256,
        expected_size_bytes=PROJECTION_CONFIG_PATH.stat().st_size,
    )
    _validate_binding_record(
        manifest["implementation"],
        label="projection implementation",
        expected_sha256=expected_projection_implementation_sha256,
        expected_size_bytes=PROJECTION_MODULE_PATH.stat().st_size,
    )
    capture_binding = _validate_binding_record(
        manifest["capture_config"], label="capture config"
    )
    if capture_binding["sha256"] != projection_config["input"]["capture_config_sha256"]:
        raise CampaignError("projection capture-config binding changed")
    input_manifest = manifest["input_manifest"]
    if not isinstance(input_manifest, dict) or set(input_manifest) != {"path", "sha256"}:
        raise CampaignError("projection input-manifest binding schema changed")
    if not isinstance(input_manifest["path"], str) or not input_manifest["path"]:
        raise CampaignError("projection input-manifest path is invalid")
    _require_sha256(input_manifest["sha256"], "projection input-manifest sha256")
    if (
        manifest["boundary_count"] != expected_count
        or manifest["sketch_shape"] != [expected_count, 4, 2, 3, 256]
    ):
        raise CampaignError("projection chunk geometry changed")
    if manifest["mapping_sha256s"] != list(expected_mapping_sha256s):
        raise CampaignError("projection mapping hashes changed")
    for label in ("sketches_logical_sha256", "band_statistics_logical_sha256"):
        _require_sha256(manifest[label], f"projection {label}")
    if (
        manifest["sources"] != projection_config["input"]["sources"]
        or manifest["bands"] != projection_config["pooling"]["bands"]
        or manifest["projection"] != projection_config["projection"]
        or manifest["band_statistics"] != projection_config["band_statistics"]
        or manifest["downstream_feature_boundary"]
        != projection_config["downstream_feature_boundary"]
        or manifest["claim_scope"] != projection_config["claim_scope"]
        or any(value is not False for value in manifest["claim_scope"].values())
    ):
        raise CampaignError("projection feature or claim boundary changed")


def _validate_projection_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_count: int,
    manifest: Mapping[str, Any],
) -> None:
    expected_keys = {
        "sketches",
        "band_statistics",
        "source_id_sha256",
        "token_ids_sha256",
        "boundary_index",
    }
    if set(arrays) != expected_keys:
        raise CampaignError("projection NPZ array allowlist changed")
    sketches = np.asarray(arrays["sketches"])
    statistics = np.asarray(arrays["band_statistics"])
    source_ids = np.asarray(arrays["source_id_sha256"])
    token_ids = np.asarray(arrays["token_ids_sha256"])
    boundary_index = np.asarray(arrays["boundary_index"])
    if sketches.shape != (expected_count, 4, 2, 3, 256) or sketches.dtype != np.dtype(
        "<f4"
    ):
        raise CampaignError("projection sketch geometry or dtype changed")
    if statistics.shape != (expected_count, 2, 3, 3) or statistics.dtype != np.dtype(
        "<f4"
    ):
        raise CampaignError("projection statistic geometry or dtype changed")
    if (
        source_ids.shape != (expected_count,)
        or source_ids.dtype != np.dtype("<U64")
        or token_ids.shape != (expected_count,)
        or token_ids.dtype != np.dtype("<U64")
    ):
        raise CampaignError("projection authentication-array geometry changed")
    if boundary_index.shape != (expected_count,) or boundary_index.dtype != np.dtype(
        "<i8"
    ):
        raise CampaignError("projection boundary-index geometry changed")
    if not np.array_equal(boundary_index, np.arange(expected_count, dtype="<i8")):
        raise CampaignError("projection boundary order changed")
    if not np.all(np.isfinite(sketches)) or not np.all(np.isfinite(statistics)):
        raise CampaignError("projection tensor contains non-finite values")
    for label, values in (("source id", source_ids), ("token ids", token_ids)):
        for value in values.tolist():
            _require_sha256(value, label)
    if len(set(source_ids.tolist())) != expected_count:
        raise CampaignError("projection chunk contains duplicate source identities")
    if _projection._hash_float_array(sketches, name="sketches") != manifest[
        "sketches_logical_sha256"
    ]:
        raise CampaignError("projection sketches logical SHA-256 mismatch")
    if _projection._hash_float_array(
        statistics, name="band_statistics"
    ) != manifest["band_statistics_logical_sha256"]:
        raise CampaignError("projection band statistics logical SHA-256 mismatch")


def load_projection_bundle(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_count: int,
    projection_config: Mapping[str, Any],
    expected_mapping_sha256s: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
    """Load one authenticated projection artifact; labels are not accepted."""

    manifest_path = _require_regular_file(manifest_path, "projection manifest")
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256, "projection manifest sha256"
    )
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise CampaignError("projection manifest SHA-256 mismatch")
    manifest = _projection.load_json_strict(manifest_path)
    _validate_projection_manifest(
        manifest,
        expected_count=expected_count,
        expected_projection_config_sha256=sha256_file(PROJECTION_CONFIG_PATH),
        expected_projection_implementation_sha256=sha256_file(
            PROJECTION_MODULE_PATH
        ),
        projection_config=projection_config,
        expected_mapping_sha256s=expected_mapping_sha256s,
    )
    data = manifest["data"]
    if not isinstance(data, dict) or set(data) != {
        "path",
        "sha256",
        "size_bytes",
        "keys",
        "reload_verified",
    }:
        raise CampaignError("projection data binding schema changed")
    if (
        not isinstance(data["path"], str)
        or not data["path"]
        or Path(data["path"]).is_absolute()
        or data["keys"]
        != [
            "band_statistics",
            "boundary_index",
            "sketches",
            "source_id_sha256",
            "token_ids_sha256",
        ]
        or data["reload_verified"] is not True
    ):
        raise CampaignError("projection data binding changed")
    expected_data_sha256 = _require_sha256(data["sha256"], "projection data sha256")
    if (
        isinstance(data["size_bytes"], bool)
        or not isinstance(data["size_bytes"], int)
        or data["size_bytes"] < 1
    ):
        raise CampaignError("projection data size is invalid")
    data_path = (manifest_path.parent / data["path"]).resolve(strict=False)
    try:
        data_path.relative_to(manifest_path.parent)
    except ValueError as exc:
        raise CampaignError("projection data escapes its manifest directory") from exc
    _projection.frozen_lexical_path_preflight((data_path,))
    _projection.frozen_canonical_path_preflight(
        input_paths=(data_path,), output_paths=()
    )
    data_path = _require_regular_file(data_path, "projection data")
    if data_path.stat().st_size != data["size_bytes"]:
        raise CampaignError("projection data size changed")
    if sha256_file(data_path) != expected_data_sha256:
        raise CampaignError("projection data SHA-256 mismatch")
    try:
        with np.load(data_path, allow_pickle=False) as loaded:
            arrays = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
    except (OSError, ValueError, KeyError) as exc:
        raise CampaignError(f"cannot load projection NPZ: {data_path}: {exc}") from exc
    _validate_projection_arrays(
        arrays, expected_count=expected_count, manifest=manifest
    )
    if sha256_file(data_path) != expected_data_sha256:
        raise CampaignError("projection data changed during reload")
    return arrays, manifest, data_path


def validate_exact_coverage(
    *,
    projections: Mapping[str, np.ndarray],
    alignment_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_ids = np.asarray(projections.get("source_id_sha256"))
    expected_ids = [str(row["source_id_sha256"]) for row in alignment_rows]
    observed_ids = source_ids.tolist()
    if len(observed_ids) != len(expected_ids):
        raise CampaignError("projection/alignment coverage count differs")
    if len(set(observed_ids)) != len(observed_ids):
        raise CampaignError("assembled projections contain duplicate source identities")
    if observed_ids != expected_ids:
        mismatch = next(
            index
            for index, (observed, expected) in enumerate(
                zip(observed_ids, expected_ids, strict=True)
            )
            if observed != expected
        )
        raise CampaignError(
            f"projection source order differs from alignment at global index {mismatch}"
        )
    return {
        "source_id_order_matches_alignment": True,
        "source_id_coverage_exact": True,
        "source_ids_unique": True,
        "source_id_sequence_sha256": _string_sequence_sha256(
            "source_id_sha256", observed_ids
        ),
    }


def build_output_tensors(
    *,
    projections: Mapping[str, np.ndarray],
    alignment_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Construct the exact tensor allowlist; no labels or outcomes are accepted."""

    raw_current, public_j_current = _features.primary_current_matrices(projections)
    tensors = _features.build_causal_activation_features(
        alignment_rows=alignment_rows,
        raw_current=raw_current,
        public_j_current=public_j_current,
        ema_alpha=config["temporal"]["ema_alpha"],
    )
    expected_keys = set(config["campaign"]["output_tensor_keys"])
    if set(tensors) != expected_keys:
        raise CampaignError("output tensor allowlist changed")
    result: dict[str, np.ndarray] = {}
    for name in config["campaign"]["output_tensor_keys"]:
        dtype = "<i8" if name == "global_index" else "<f8"
        values = np.ascontiguousarray(tensors[name], dtype=dtype)
        if values.dtype.kind not in {"i", "f"} or (
            values.dtype.kind == "f" and not np.all(np.isfinite(values))
        ):
            raise CampaignError(f"output {name} is not a finite numeric tensor")
        result[name] = values
    expected_retained = np.asarray(
        [
            index
            for index, row in enumerate(alignment_rows)
            if row["stable_feature_eligible"]
        ],
        dtype="<i8",
    )
    if not np.array_equal(result["global_index"], expected_retained):
        raise CampaignError("retained global indices changed")
    return result


def _write_npz_no_clobber(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise CampaignError(f"refusing to overwrite campaign data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    if temporary.exists() or temporary.is_symlink():
        raise CampaignError(f"temporary campaign data exists: {temporary}")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _write_json_no_clobber(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise CampaignError(f"refusing to overwrite campaign manifest: {path}")
    rendered = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise CampaignError(f"temporary campaign manifest exists: {temporary}")
    temporary.write_text(rendered)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=FEATURE_CONFIG_PATH)
    parser.add_argument(
        "--projection-manifest", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--projection-manifest-sha256", action="append", required=True
    )
    parser.add_argument("--alignment-index", type=Path, required=True)
    parser.add_argument("--alignment-index-sha256", required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    manifest_paths = tuple(args.projection_manifest or ())
    manifest_hashes = tuple(args.projection_manifest_sha256 or ())
    _projection.frozen_lexical_path_preflight(
        (
            args.config,
            *manifest_paths,
            args.alignment_index,
            args.output_data,
            args.output_manifest,
        )
    )
    _projection.frozen_canonical_path_preflight(
        input_paths=(args.config, *manifest_paths, args.alignment_index),
        output_paths=(args.output_data, args.output_manifest),
    )
    config_path = _require_regular_file(args.config, "activation-feature config")
    alignment_path = _require_regular_file(args.alignment_index, "alignment index")
    config = _features.validate_config(_projection.load_json_strict(config_path))
    campaign = config["campaign"]
    if (
        len(manifest_paths) != campaign["projection_chunk_count"]
        or len(manifest_hashes) != campaign["projection_chunk_count"]
    ):
        raise CampaignError("exactly four ordered projection manifests and hashes are required")
    expected_alignment_sha256 = _require_sha256(
        args.alignment_index_sha256, "alignment index sha256"
    )
    if sha256_file(alignment_path) != expected_alignment_sha256:
        raise CampaignError("alignment index SHA-256 mismatch")
    alignment = _projection.load_json_strict(alignment_path)
    alignment_rows = _features.validate_alignment_index(alignment)
    _validate_alignment_metadata(alignment, campaign=campaign)

    projection_config = _projection.validate_config(
        _projection.load_json_strict(PROJECTION_CONFIG_PATH)
    )
    expected_mapping_sha256s = [
        _projection.countsketch_mapping(
            seed=seed,
            hidden_size=projection_config["input"]["hidden_size"],
            width=projection_config["projection"]["width"],
        )[2]
        for seed in projection_config["projection"]["seeds"]
    ]
    fixed_inputs = [
        config_path,
        FEATURE_MODULE_PATH,
        PROJECTION_CONFIG_PATH,
        PROJECTION_MODULE_PATH,
        SCRIPT_PATH,
        alignment_path,
    ]
    pre_hashes = {str(path): sha256_file(path) for path in fixed_inputs}
    chunks = []
    projection_records = []
    data_paths = []
    for chunk_index, (manifest_path, manifest_hash, expected_count) in enumerate(
        zip(
            manifest_paths,
            manifest_hashes,
            campaign["projection_chunk_boundary_counts"],
            strict=True,
        )
    ):
        manifest_path = _require_regular_file(
            manifest_path, f"projection manifest {chunk_index}"
        )
        arrays, manifest, data_path = load_projection_bundle(
            manifest_path,
            expected_manifest_sha256=manifest_hash,
            expected_count=expected_count,
            projection_config=projection_config,
            expected_mapping_sha256s=expected_mapping_sha256s,
        )
        chunks.append(arrays)
        data_paths.append(data_path)
        pre_hashes[str(manifest_path)] = sha256_file(manifest_path)
        pre_hashes[str(data_path)] = sha256_file(data_path)
        projection_records.append(
            {
                "chunk_index": chunk_index,
                "boundary_count": expected_count,
                "manifest": _artifact_record(manifest_path),
                "data": _artifact_record(data_path),
                "capture_manifest_sha256": manifest["input_manifest"]["sha256"],
                "source_id_sequence_sha256": _string_sequence_sha256(
                    "source_id_sha256", arrays["source_id_sha256"].tolist()
                ),
                "token_ids_sequence_sha256": _string_sequence_sha256(
                    "token_ids_sha256", arrays["token_ids_sha256"].tolist()
                ),
            }
        )
    projections = _features.concatenate_projection_arrays(chunks)
    if len(projections["source_id_sha256"]) != campaign["total_boundary_count"]:
        raise CampaignError("assembled projection count changed")
    coverage = validate_exact_coverage(
        projections=projections, alignment_rows=alignment_rows
    )
    tensors = build_output_tensors(
        projections=projections,
        alignment_rows=alignment_rows,
        config=config,
    )
    for path_text, expected_hash in pre_hashes.items():
        if sha256_file(Path(path_text)) != expected_hash:
            raise CampaignError(f"authenticated input changed during assembly: {path_text}")

    output_data = args.output_data.resolve(strict=False)
    output_manifest = args.output_manifest.resolve(strict=False)
    if output_data == output_manifest:
        raise CampaignError("campaign data and manifest paths must differ")
    if output_data.parent != output_manifest.parent:
        raise CampaignError("campaign data and manifest must share a directory")
    if output_data.exists() or output_data.is_symlink():
        raise CampaignError(f"campaign data output exists: {output_data}")
    if output_manifest.exists() or output_manifest.is_symlink():
        raise CampaignError(f"campaign manifest output exists: {output_manifest}")
    _write_npz_no_clobber(output_data, tensors)
    output_data_sha256 = sha256_file(output_data)
    try:
        with np.load(output_data, allow_pickle=False) as loaded:
            if set(loaded.files) != set(tensors):
                raise CampaignError("campaign NPZ reload keys changed")
            for name, expected in tensors.items():
                if not np.array_equal(loaded[name], expected, equal_nan=False):
                    raise CampaignError(f"campaign NPZ reload changed {name}")
    except (OSError, ValueError, KeyError) as exc:
        raise CampaignError(f"cannot reload campaign NPZ: {exc}") from exc
    for path_text, expected_hash in pre_hashes.items():
        if sha256_file(Path(path_text)) != expected_hash:
            raise CampaignError(f"authenticated input changed during output: {path_text}")

    output_arrays = {
        name: {
            "shape": list(values.shape),
            "dtype": (
                "little-endian-int64"
                if values.dtype == np.dtype("<i8")
                else "little-endian-float64"
            ),
            "logical_sha256": _logical_array_sha256(name, values),
        }
        for name, values in tensors.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "passed",
        "status_scope": "label_free_primary_seed_current_and_causal_sequence_tensors_only",
        "feature_config": _artifact_record(config_path),
        "implementation": _artifact_record(SCRIPT_PATH),
        "feature_implementation": _artifact_record(FEATURE_MODULE_PATH),
        "projection_config": _artifact_record(PROJECTION_CONFIG_PATH),
        "projection_implementation": _artifact_record(PROJECTION_MODULE_PATH),
        "pre_and_post_input_bindings_equal": True,
        "inputs": {
            "alignment_index": {
                **_artifact_record(alignment_path),
                "row_count": campaign["total_boundary_count"],
                "stable_row_count": campaign["stable_feature_count"],
            },
            "projection_chunks": projection_records,
        },
        "coverage": {
            **coverage,
            "chunk_order_exact": True,
            "boundary_count": campaign["total_boundary_count"],
            "stable_feature_count": campaign["stable_feature_count"],
        },
        "output": {
            "path": output_data.name,
            "sha256": output_data_sha256,
            "size_bytes": output_data.stat().st_size,
            "keys": sorted(tensors),
            "numeric_tensor_only": True,
            "reload_verified": True,
            "arrays": output_arrays,
        },
        "projection": config["projection"],
        "temporal": config["temporal"],
        "variants": config["variants"],
        "feature_boundary": config["feature_boundary"],
        "claim_scope": config["claim_scope"],
        "forbidden_path_guard_passed": True,
        "reserved_validation_access_authorized": False,
    }
    _write_json_no_clobber(output_manifest, manifest)
    print(
        f"wrote {campaign['stable_feature_count']} label-free activation feature rows "
        f"to {output_data}"
    )
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
