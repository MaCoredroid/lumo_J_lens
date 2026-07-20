#!/usr/bin/env python3
"""Project authenticated pre-vocabulary SWE states with a label-free sketch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_activation_projection.json"
CAPTURE_CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_raw_capture.json"
SCRIPT_PATH = Path(__file__).resolve()
KIND = "swe_task_state_v4_label_independent_activation_projection"
SCHEMA_VERSION = 1
FROZEN_FORBIDDEN_PATH_FRAGMENTS = ("reserved", "validation")


class ProjectionError(ValueError):
    """Raised when the projection contract or an input artifact fails closed."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProjectionError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> Any:
    try:
        return json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProjectionError(f"cannot read strict JSON {path}: {exc}") from exc


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
        raise ProjectionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ProjectionError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ProjectionError(f"{label} must be a regular file: {path}")
    return resolved


def _path_has_forbidden_fragment(path: Path, fragments: Iterable[str]) -> bool:
    parts = [part.lower() for part in path.resolve(strict=False).parts]
    return any(fragment.lower() in part for fragment in fragments for part in parts)


def frozen_lexical_path_preflight(paths: Iterable[Path | None]) -> None:
    """Reject forbidden path text before any filesystem operation."""

    for path in paths:
        if path is None:
            continue
        normalized = os.path.normpath(os.path.abspath(os.fspath(path))).lower()
        if any(
            fragment in component
            for fragment in FROZEN_FORBIDDEN_PATH_FRAGMENTS
            for component in Path(normalized).parts
        ):
            raise ProjectionError(
                f"forbidden path rejected before filesystem access: {path}"
            )


def frozen_canonical_path_preflight(
    *, input_paths: Iterable[Path | None], output_paths: Iterable[Path | None]
) -> None:
    """Resolve metadata, then reject forbidden canonical parents before reads."""

    for path, strict in [
        *((path, True) for path in input_paths),
        *((path, False) for path in output_paths),
    ]:
        if path is None:
            continue
        try:
            resolved = Path(path).resolve(strict=strict)
        except OSError as exc:
            raise ProjectionError(
                f"cannot resolve projection path metadata: {path}: {exc}"
            ) from exc
        lowered_parts = [part.lower() for part in resolved.parts]
        if any(
            fragment in component
            for fragment in FROZEN_FORBIDDEN_PATH_FRAGMENTS
            for component in lowered_parts
        ):
            raise ProjectionError(
                f"forbidden canonical path rejected before file read or hash: {path}"
            )


def validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ProjectionError("projection config must be an object")
    if set(config) != {
        "schema_version",
        "id",
        "status",
        "input",
        "pooling",
        "projection",
        "band_statistics",
        "authentication",
        "downstream_feature_boundary",
        "claim_scope",
    }:
        raise ProjectionError("projection config top-level keys changed")
    if config["schema_version"] != 1:
        raise ProjectionError("unsupported projection config schema")
    if config["id"] != "swe-task-state-v4-label-independent-activation-countsketch":
        raise ProjectionError("projection config id changed")
    if config["status"] != "development_only_reserved_validation_closed":
        raise ProjectionError("projection config status changed")
    capture_hash = sha256_file(CAPTURE_CONFIG_PATH)
    expected_input = {
        "kind": "swe_task_state_v4_label_independent_public_j_state_capture",
        "schema_version": 1,
        "capture_config_sha256": capture_hash,
        "sources": ["raw_residual", "public_j_state"],
        "layers": list(range(24, 48)),
        "hidden_size": 5120,
        "dtype": "little-endian-float32",
    }
    if config["input"] != expected_input:
        raise ProjectionError("projection input contract changed")
    expected_pooling = {
        "kind": "arithmetic_mean_state_within_inclusive_layer_band",
        "bands": [
            {"id": "early", "first_layer": 24, "last_layer": 29},
            {"id": "middle", "first_layer": 30, "last_layer": 41},
            {"id": "late", "first_layer": 42, "last_layer": 47},
        ],
        "normalization_before_pooling": "none",
        "preserve_pre_vocabulary_state_scale": True,
    }
    if config["pooling"] != expected_pooling:
        raise ProjectionError("projection pooling contract changed")
    expected_projection = {
        "kind": "countsketch",
        "width": 256,
        "mapping": (
            "bucket=uint64_be(sha256(seed + NUL + hidden_index)[0:8]) mod "
            "width; sign=+1 iff digest[8] bit0 else -1"
        ),
        "scale": "sqrt(width / hidden_size)",
        "accumulation_dtype": "float64",
        "output_dtype": "little-endian-float32",
        "seeds": [
            "swe-task-state-v4-activation-countsketch-primary-0",
            "swe-task-state-v4-activation-countsketch-sensitivity-1",
            "swe-task-state-v4-activation-countsketch-sensitivity-2",
            "swe-task-state-v4-activation-countsketch-sensitivity-3",
        ],
        "primary_seed_index": 0,
        "sensitivity_seed_indices": [1, 2, 3],
        "decoder_fold": (
            "for width 64, sum buckets with index mod 64 and divide by 2"
        ),
    }
    if config["projection"] != expected_projection:
        raise ProjectionError("CountSketch contract changed")
    if config["band_statistics"] != [
        "pooled_state_rms",
        "mean_layer_state_rms",
        "first_to_last_layer_delta_rms",
    ]:
        raise ProjectionError("band statistics changed")
    if config["authentication"] != {
        "require_input_manifest_sha256": True,
        "require_capture_status_passed": True,
        "require_all_boundary_capture_valid": True,
        "require_shard_sha256_and_reload": True,
        "forbidden_path_fragments": ["reserved", "validation"],
        "no_clobber": True,
    }:
        raise ProjectionError("projection authentication contract changed")
    if config["downstream_feature_boundary"] != {
        "primary_feature_array": "sketches",
        "diagnostic_feature_array": "band_statistics",
        "authentication_or_join_only_arrays": [
            "source_id_sha256",
            "token_ids_sha256",
            "boundary_index",
        ],
        "primary_current_feature": (
            "selected seed and source, 256-to-64 folded, band-major flatten"
        ),
        "authentication_fields_as_features_forbidden": True,
        "semantic_ids_as_features_forbidden": True,
        "base_report_access_forbidden": True,
    }:
        raise ProjectionError("downstream projection feature boundary changed")
    claims = config["claim_scope"]
    if set(claims) != {
        "private_chain_of_thought_reconstructed",
        "cot_like_observable_event_decoding_established",
        "emotion_decoding_established",
        "causal_interpretation_established",
        "incremental_value_over_visible_or_word_probe_baselines_established",
    } or any(value is not False for value in claims.values()):
        raise ProjectionError("projection-only claim scope must remain false")
    return config


def countsketch_mapping(
    *, seed: str, hidden_size: int, width: int
) -> tuple[np.ndarray, np.ndarray, str]:
    if not isinstance(seed, str) or not seed:
        raise ProjectionError("CountSketch seed must be nonempty text")
    if hidden_size < 1 or width < 1:
        raise ProjectionError("CountSketch dimensions must be positive")
    buckets = np.empty(hidden_size, dtype="<u4")
    signs = np.empty(hidden_size, dtype=np.int8)
    prefix = seed.encode("utf-8") + b"\0"
    for hidden_index in range(hidden_size):
        digest = hashlib.sha256(prefix + str(hidden_index).encode("ascii")).digest()
        buckets[hidden_index] = int.from_bytes(digest[:8], "big") % width
        signs[hidden_index] = 1 if digest[8] & 1 else -1
    header = canonical_json_bytes(
        {
            "algorithm": "countsketch_sha256_v1",
            "seed": seed,
            "hidden_size": hidden_size,
            "width": width,
            "bucket_dtype": "little-endian-uint32",
            "sign_dtype": "int8",
        }
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(buckets.tobytes(order="C"))
    digest.update(signs.tobytes(order="C"))
    return buckets, signs, digest.hexdigest()


def countsketch_vector(
    vector: np.ndarray,
    *,
    buckets: np.ndarray,
    signs: np.ndarray,
    width: int,
) -> np.ndarray:
    values = np.asarray(vector)
    if values.shape != buckets.shape or signs.shape != buckets.shape:
        raise ProjectionError("CountSketch vector or mapping geometry changed")
    if not np.issubdtype(values.dtype, np.floating) or not np.all(np.isfinite(values)):
        raise ProjectionError("CountSketch vector must contain finite floats")
    result = np.zeros(width, dtype=np.float64)
    np.add.at(result, buckets, values.astype(np.float64) * signs.astype(np.float64))
    result *= math.sqrt(width / len(values))
    projected = result.astype("<f4")
    if not np.all(np.isfinite(projected)):
        raise ProjectionError("CountSketch output contains non-finite values")
    return projected


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def project_state_pair(
    raw_residual: np.ndarray,
    public_j_state: np.ndarray,
    *,
    config: Mapping[str, Any],
    mappings: Sequence[tuple[np.ndarray, np.ndarray, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return sketches [seed, source, band, width] and scalar statistics."""

    layers = list(config["input"]["layers"])
    hidden_size = int(config["input"]["hidden_size"])
    expected_shape = (len(layers), hidden_size)
    sources = [np.asarray(raw_residual), np.asarray(public_j_state)]
    for source in sources:
        if source.shape != expected_shape or source.dtype != np.dtype("float32"):
            raise ProjectionError(
                f"state tensor must have shape {expected_shape} and float32 dtype"
            )
        if not np.all(np.isfinite(source)):
            raise ProjectionError("state tensor contains non-finite values")
    bands = config["pooling"]["bands"]
    width = int(config["projection"]["width"])
    sketches = np.empty(
        (len(mappings), len(sources), len(bands), width), dtype="<f4"
    )
    statistics = np.empty((len(sources), len(bands), 3), dtype="<f4")
    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    for source_index, source in enumerate(sources):
        for band_index, band in enumerate(bands):
            band_layers = list(range(band["first_layer"], band["last_layer"] + 1))
            try:
                band_values = source[[layer_to_index[layer] for layer in band_layers]]
            except KeyError as exc:
                raise ProjectionError(f"band references absent layer: {exc}") from exc
            pooled = np.mean(band_values.astype(np.float64), axis=0)
            statistics[source_index, band_index] = np.asarray(
                [
                    _rms(pooled),
                    float(np.mean([_rms(row) for row in band_values])),
                    _rms(band_values[-1].astype(np.float64) - band_values[0]),
                ],
                dtype="<f4",
            )
            for seed_index, (buckets, signs, _mapping_hash) in enumerate(mappings):
                sketches[seed_index, source_index, band_index] = countsketch_vector(
                    pooled, buckets=buckets, signs=signs, width=width
                )
    if not np.all(np.isfinite(sketches)) or not np.all(np.isfinite(statistics)):
        raise ProjectionError("projected features contain non-finite values")
    return sketches, statistics


def fold_projection_256_to_64(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[-1] != 256 or not np.issubdtype(array.dtype, np.floating):
        raise ProjectionError("fold input must have a trailing width of 256 floats")
    folded = (
        array[..., 0:64].astype(np.float64)
        + array[..., 64:128].astype(np.float64)
        + array[..., 128:192].astype(np.float64)
        + array[..., 192:256].astype(np.float64)
    ) / 2.0
    result = folded.astype("<f4")
    if not np.all(np.isfinite(result)):
        raise ProjectionError("folded projection contains non-finite values")
    return result


def _extract_current_features_for_seed(
    arrays: Mapping[str, np.ndarray],
    *,
    seed_index: int,
    source: str,
) -> np.ndarray:
    if source not in {"raw_residual", "public_j_state"}:
        raise ProjectionError(f"unknown activation source: {source}")
    sketches = np.asarray(arrays.get("sketches"))
    if sketches.ndim != 5 or sketches.shape[2:] != (2, 3, 256):
        raise ProjectionError("primary sketch array geometry changed")
    if (
        isinstance(seed_index, bool)
        or not isinstance(seed_index, int)
        or not 0 <= seed_index < sketches.shape[1]
    ):
        raise ProjectionError("seed index is out of range")
    if sketches.dtype != np.dtype("float32") or not np.all(np.isfinite(sketches)):
        raise ProjectionError("primary sketches must be finite float32")
    source_index = 0 if source == "raw_residual" else 1
    folded = fold_projection_256_to_64(sketches[:, seed_index, source_index])
    result = folded.reshape(folded.shape[0], 3 * 64).astype("<f4", copy=False)
    if result.shape != (sketches.shape[0], 192):
        raise ProjectionError("current activation feature width changed")
    return result.copy()


def extract_primary_current_features(
    arrays: Mapping[str, np.ndarray],
    *,
    seed_index: int,
    source: str,
) -> np.ndarray:
    """Extract the frozen headline seed only; IDs/fidelity never enter X."""

    if isinstance(seed_index, bool) or seed_index != 0:
        raise ProjectionError("headline primary extraction requires seed_index == 0")
    return _extract_current_features_for_seed(
        arrays, seed_index=seed_index, source=source
    )


def extract_sensitivity_current_features(
    arrays: Mapping[str, np.ndarray],
    *,
    seed_index: int,
    source: str,
) -> np.ndarray:
    """Extract one frozen sensitivity seed, never the headline primary seed."""

    if isinstance(seed_index, bool) or seed_index not in {1, 2, 3}:
        raise ProjectionError("sensitivity extraction requires seed_index in {1, 2, 3}")
    return _extract_current_features_for_seed(
        arrays, seed_index=seed_index, source=source
    )


def _validate_capture_manifest(
    manifest: Any,
    *,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise ProjectionError("capture manifest must be an object")
    expected_top_keys = {
        "schema_version",
        "kind",
        "status",
        "status_scope",
        "capture_config",
        "source_bundle",
        "reference_report",
        "base_report",
        "implementation",
        "normalized_cli_contract",
        "model",
        "lens",
        "capture",
        "feature_independence",
        "downstream_feature_boundary",
        "claim_scope",
        "summary",
        "boundaries",
    }
    if set(manifest) != expected_top_keys:
        raise ProjectionError("capture manifest top-level allowlist changed")
    if (
        manifest.get("schema_version") != config["input"]["schema_version"]
        or manifest.get("kind") != config["input"]["kind"]
        or manifest.get("status") != "passed"
    ):
        raise ProjectionError("capture manifest identity or status changed")
    capture_config = manifest.get("capture_config")
    if not isinstance(capture_config, dict) or capture_config.get("sha256") != config[
        "input"
    ]["capture_config_sha256"]:
        raise ProjectionError("capture-config hash binding changed")
    if manifest.get("capture") != {
        "layers": config["input"]["layers"],
        "position": "causal_prefix_tail_only",
        "positions_argument": [-1],
        "stream_final_only_required": True,
        "raw_tensor": "post_block_residual_before_final_norm",
        "transported_tensor": (
            "public_j_state_before_bfloat16_final_norm_or_vocabulary_projection"
        ),
        "storage_dtype": "little_endian_float32",
        "shard_format": "safetensors",
    }:
        raise ProjectionError("capture tensor contract changed")
    downstream = manifest.get("downstream_feature_boundary")
    if not isinstance(downstream, dict) or downstream.get(
        "allowed_tensor_features"
    ) != ["raw_residual", "public_j_state"] or downstream.get(
        "semantic_ids_as_features_forbidden"
    ) is not True or downstream.get("base_report_fields_as_features_forbidden") is not True:
        raise ProjectionError("capture downstream feature boundary changed")
    claims = manifest.get("claim_scope")
    if not isinstance(claims, dict) or any(value is not False for value in claims.values()):
        raise ProjectionError("capture manifest overclaims")
    boundaries = manifest.get("boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise ProjectionError("capture manifest must contain boundaries")
    for index, boundary in enumerate(boundaries):
        expected_boundary_keys = {
            "index",
            "source_id_sha256",
            "token_ids_sha256",
            "token_count",
            "token_position",
            "residual_capture_manifest",
            "reference_residual_manifest_equal",
            "shard",
            "capture_valid",
            "vocabulary_adapter_strict",
            "final_model_top1_matches_greedy",
            "final_norm_reconstruction_within_tolerance",
            "final_logits_reconstruction_within_tolerance",
        }
        if (
            not isinstance(boundary, dict)
            or set(boundary) != expected_boundary_keys
            or boundary.get("index") != index
            or boundary.get("capture_valid") is not True
            or boundary.get("reference_residual_manifest_equal") is not True
        ):
            raise ProjectionError(f"capture boundary {index} is not authenticated")
    return boundaries


def _load_verified_shard(
    capture_root: Path,
    boundary: Mapping[str, Any],
    *,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    from safetensors.numpy import load_file

    shard = boundary.get("shard")
    if (
        not isinstance(shard, dict)
        or set(shard) != {
            "path",
            "sha256",
            "size_bytes",
            "tensor_keys",
            "shape",
            "dtype",
            "raw_residual_logical_sha256",
            "public_j_state_logical_sha256",
            "reload_verified",
        }
        or shard.get("reload_verified") is not True
    ):
        raise ProjectionError("capture shard record is invalid")
    relative = shard.get("path")
    if not isinstance(relative, str):
        raise ProjectionError("capture shard path is invalid")
    shard_path = (capture_root / relative).resolve(strict=False)
    try:
        shard_path.relative_to(capture_root)
    except ValueError as exc:
        raise ProjectionError("capture shard escapes its manifest directory") from exc
    shard_path = _require_regular_file(shard_path, "capture shard")
    expected_sha = _require_sha256(shard.get("sha256"), "capture shard sha256")
    if sha256_file(shard_path) != expected_sha:
        raise ProjectionError("capture shard SHA-256 mismatch")
    loaded = load_file(shard_path)
    if set(loaded) != {"raw_residual", "public_j_state"}:
        raise ProjectionError("capture shard tensor keys changed")
    result = []
    for key in ("raw_residual", "public_j_state"):
        array = np.asarray(loaded[key])
        if array.shape != expected_shape or array.dtype != np.dtype("float32"):
            raise ProjectionError(f"capture shard {key} geometry or dtype changed")
        if not np.all(np.isfinite(array)):
            raise ProjectionError(f"capture shard {key} is non-finite")
        result.append(array)
    if sha256_file(shard_path) != expected_sha:
        raise ProjectionError("capture shard changed during reload")
    return result[0], result[1]


def _hash_float_array(array: np.ndarray, *, name: str) -> str:
    contiguous = np.asarray(array, dtype="<f4", order="C")
    header = canonical_json_bytes(
        {"name": name, "shape": list(contiguous.shape), "dtype": "little-endian-float32"}
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def build_projection_arrays(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = _require_regular_file(manifest_path, "capture manifest")
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256, "capture manifest sha256"
    )
    observed_manifest_sha256 = sha256_file(manifest_path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise ProjectionError("capture manifest SHA-256 mismatch")
    boundaries = _validate_capture_manifest(
        load_json_strict(manifest_path), config=config
    )
    hidden_size = int(config["input"]["hidden_size"])
    width = int(config["projection"]["width"])
    mappings = [
        countsketch_mapping(seed=seed, hidden_size=hidden_size, width=width)
        for seed in config["projection"]["seeds"]
    ]
    sketch_rows = []
    statistic_rows = []
    source_ids = []
    token_hashes = []
    expected_shape = (len(config["input"]["layers"]), hidden_size)
    for boundary in boundaries:
        raw, public_j = _load_verified_shard(
            manifest_path.parent, boundary, expected_shape=expected_shape
        )
        sketches, statistics = project_state_pair(
            raw, public_j, config=config, mappings=mappings
        )
        sketch_rows.append(sketches)
        statistic_rows.append(statistics)
        source_ids.append(_require_sha256(boundary.get("source_id_sha256"), "source id"))
        token_hashes.append(_require_sha256(boundary.get("token_ids_sha256"), "token ids"))
    arrays = {
        "sketches": np.asarray(sketch_rows, dtype="<f4"),
        "band_statistics": np.asarray(statistic_rows, dtype="<f4"),
        "source_id_sha256": np.asarray(source_ids, dtype="<U64"),
        "token_ids_sha256": np.asarray(token_hashes, dtype="<U64"),
        "boundary_index": np.arange(len(boundaries), dtype="<i8"),
    }
    expected_sketch_shape = (
        len(boundaries),
        len(mappings),
        len(config["input"]["sources"]),
        len(config["pooling"]["bands"]),
        width,
    )
    if arrays["sketches"].shape != expected_sketch_shape:
        raise ProjectionError("assembled sketch geometry changed")
    metadata = {
        "input_manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": observed_manifest_sha256,
        },
        "mapping_sha256s": [mapping[2] for mapping in mappings],
        "boundary_count": len(boundaries),
        "sketch_shape": list(expected_sketch_shape),
        "sketches_logical_sha256": _hash_float_array(
            arrays["sketches"], name="sketches"
        ),
        "band_statistics_logical_sha256": _hash_float_array(
            arrays["band_statistics"], name="band_statistics"
        ),
    }
    return arrays, metadata


def _write_npz_no_clobber(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise ProjectionError(f"refusing to overwrite projection data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    if temporary.exists() or temporary.is_symlink():
        raise ProjectionError(f"temporary projection data exists: {temporary}")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _write_json_no_clobber(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise ProjectionError(f"refusing to overwrite projection manifest: {path}")
    rendered = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(rendered)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--capture-manifest-sha256", required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    frozen_lexical_path_preflight(
        (
            args.config,
            args.capture_manifest,
            args.output_data,
            args.output_manifest,
        )
    )
    frozen_canonical_path_preflight(
        input_paths=(args.config, args.capture_manifest),
        output_paths=(args.output_data, args.output_manifest),
    )
    config_path = _require_regular_file(args.config, "projection config")
    input_manifest_path = _require_regular_file(
        args.capture_manifest, "capture manifest"
    )
    pre_run_bindings = {
        "projection_config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256_file(config_path),
            "size_bytes": config_path.stat().st_size,
        },
        "capture_config": {
            "path": str(CAPTURE_CONFIG_PATH.relative_to(ROOT)),
            "sha256": sha256_file(CAPTURE_CONFIG_PATH),
            "size_bytes": CAPTURE_CONFIG_PATH.stat().st_size,
        },
        "implementation": {
            "path": str(SCRIPT_PATH.relative_to(ROOT)),
            "sha256": sha256_file(SCRIPT_PATH),
            "size_bytes": SCRIPT_PATH.stat().st_size,
        },
        "input_manifest": {
            "path": str(input_manifest_path.relative_to(ROOT)),
            "sha256": sha256_file(input_manifest_path),
            "size_bytes": input_manifest_path.stat().st_size,
        },
    }
    config = validate_config(load_json_strict(config_path))
    if (
        sha256_file(config_path) != pre_run_bindings["projection_config"]["sha256"]
        or sha256_file(CAPTURE_CONFIG_PATH)
        != pre_run_bindings["capture_config"]["sha256"]
        or sha256_file(SCRIPT_PATH) != pre_run_bindings["implementation"]["sha256"]
        or sha256_file(input_manifest_path)
        != pre_run_bindings["input_manifest"]["sha256"]
    ):
        raise ProjectionError("projection input changed while being parsed")
    forbidden = config["authentication"]["forbidden_path_fragments"]
    for path in (
        config_path,
        args.capture_manifest,
        args.output_data,
        args.output_manifest,
    ):
        if _path_has_forbidden_fragment(path, forbidden):
            raise ProjectionError(f"reserved-validation path is forbidden: {path}")
    output_data = args.output_data.resolve(strict=False)
    output_manifest = args.output_manifest.resolve(strict=False)
    if output_data == output_manifest:
        raise ProjectionError("projection data and manifest paths must differ")
    if output_data.exists() or output_data.is_symlink():
        raise ProjectionError(f"projection data output exists: {output_data}")
    if output_manifest.exists() or output_manifest.is_symlink():
        raise ProjectionError(f"projection manifest output exists: {output_manifest}")
    if output_data.parent != output_manifest.parent:
        raise ProjectionError("projection data and manifest must share a directory")

    arrays, input_metadata = build_projection_arrays(
        input_manifest_path,
        expected_manifest_sha256=args.capture_manifest_sha256,
        config=config,
    )
    post_run_hashes = {
        "projection_config": sha256_file(config_path),
        "capture_config": sha256_file(CAPTURE_CONFIG_PATH),
        "implementation": sha256_file(SCRIPT_PATH),
        "input_manifest": sha256_file(input_manifest_path),
    }
    for label, observed_hash in post_run_hashes.items():
        if observed_hash != pre_run_bindings[label]["sha256"]:
            raise ProjectionError(f"{label} changed during projection")
    _write_npz_no_clobber(output_data, arrays)
    data_sha256 = sha256_file(output_data)
    reloaded = np.load(output_data, allow_pickle=False)
    if set(reloaded.files) != set(arrays):
        raise ProjectionError("projection NPZ reload keys changed")
    for key, expected in arrays.items():
        if not np.array_equal(reloaded[key], expected, equal_nan=False):
            raise ProjectionError(f"projection NPZ reload changed {key}")
    reloaded.close()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "passed",
        "projection_config": {
            **pre_run_bindings["projection_config"],
        },
        "capture_config": pre_run_bindings["capture_config"],
        "implementation": pre_run_bindings["implementation"],
        "pre_and_post_projection_bindings_equal": True,
        **input_metadata,
        "data": {
            "path": str(output_data.relative_to(output_manifest.parent)),
            "sha256": data_sha256,
            "size_bytes": output_data.stat().st_size,
            "keys": sorted(arrays),
            "reload_verified": True,
        },
        "sources": config["input"]["sources"],
        "bands": config["pooling"]["bands"],
        "projection": config["projection"],
        "band_statistics": config["band_statistics"],
        "downstream_feature_boundary": config["downstream_feature_boundary"],
        "claim_scope": config["claim_scope"],
        "forbidden_path_guard_passed": True,
        "reserved_validation_access_authorized": False,
    }
    _write_json_no_clobber(output_manifest, manifest)
    print(
        f"wrote {input_metadata['boundary_count']} label-independent activation "
        f"projections to {output_data}",
    )
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
