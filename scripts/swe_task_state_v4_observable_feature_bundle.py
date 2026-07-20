#!/usr/bin/env python3
"""Build the V4 observable feature bundle from authenticated source NPZs.

This producer has no caller-supplied feature-array seam.  It authenticates the
label-free alignment index and the visible/activation source manifests, opens
and validates the NPZ bound by each source manifest, and copies only the frozen
numeric block allowlist into a new bundle.  Labels, targets, outcomes, and
completion text are neither accepted nor opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scripts import swe_task_state_v4_observable_decoder as DECODER
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    import swe_task_state_v4_observable_decoder as DECODER  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = DECODER.CONFIG_PATH
SCHEMA_VERSION = 2
KIND = DECODER.FEATURE_BUNDLE_KIND
FORMAT = "npz_allow_pickle_false"
CONSTRUCTION_ALGORITHM = "authenticated_source_npz_exact_join_v1"

VISIBLE_ROLE = "visible_baselines"
ACTIVATION_ROLE = "activation_features"
VISIBLE_KIND = "swe_task_state_v4_visible_word_probe_baseline_features"
ACTIVATION_KIND = "swe_task_state_v4_label_free_activation_feature_campaign"

VISIBLE_KEYS = (
    "global_index",
    "history_only",
    "sequence_logit",
    "sequence_j",
    "sequence_logit_j",
)
ACTIVATION_KEYS = (
    "global_index",
    "public_j_activation_current",
    "public_j_activation_sequence",
    "raw_activation_current",
    "raw_activation_sequence",
    "raw_public_j_activation_sequence",
)
SOURCE_FEATURE_BLOCKS = {
    VISIBLE_ROLE: list(VISIBLE_KEYS[1:]),
    ACTIVATION_ROLE: list(ACTIVATION_KEYS[1:]),
}

VISIBLE_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "status",
    "status_scope",
    "pre_and_post_input_bindings_equal",
    "config",
    "implementation",
    "inputs",
    "local_code_dependencies",
    "coverage",
    "output",
    "variants",
    "feature_boundary",
    "claim_scope",
    "forbidden_path_guard_passed",
    "reserved_validation_access_authorized",
}
ACTIVATION_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "status",
    "status_scope",
    "feature_config",
    "implementation",
    "feature_implementation",
    "projection_config",
    "projection_implementation",
    "pre_and_post_input_bindings_equal",
    "inputs",
    "coverage",
    "output",
    "projection",
    "temporal",
    "variants",
    "feature_boundary",
    "claim_scope",
    "forbidden_path_guard_passed",
    "reserved_validation_access_authorized",
}


class BundleBuilderError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleBuilderError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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


def logical_array_sha256(name: str, array: np.ndarray) -> str:
    """Match the logical-array digest emitted by both source producers."""

    values = np.ascontiguousarray(array)
    header = canonical_json_bytes(
        {"name": name, "shape": list(values.shape), "dtype": values.dtype.str}
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json_strict(path: Path, label: str) -> dict[str, Any]:
    try:
        value = DECODER.load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, DECODER.DecoderError) as error:
        raise BundleBuilderError(f"cannot load strict {label} JSON: {error}") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return dict(value)


def _regular_input(path: Path, label: str) -> Path:
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BundleBuilderError(f"cannot resolve {label}: {path}: {error}") from error
    _require(resolved.is_file() and not resolved.is_symlink(), f"{label} is not a regular file")
    return resolved


def _root_record_path(path_text: str, label: str) -> Path:
    _require(isinstance(path_text, str) and bool(path_text), f"{label} path is invalid")
    raw = Path(path_text)
    candidate = raw if raw.is_absolute() else ROOT / raw
    DECODER.frozen_lexical_path_preflight([candidate])
    DECODER.frozen_canonical_path_preflight(input_paths=[candidate], output_paths=[])
    return _regular_input(candidate, label)


def _source_data_path(manifest_path: Path, path_text: Any, label: str) -> Path:
    _require(isinstance(path_text, str) and bool(path_text), f"{label} path is invalid")
    raw = Path(path_text)
    _require(not raw.is_absolute(), f"{label} path must be relative to its manifest")
    DECODER.frozen_lexical_path_preflight([raw])
    candidate = manifest_path.parent / raw
    DECODER.frozen_canonical_path_preflight(input_paths=[candidate], output_paths=[])
    resolved = _regular_input(candidate, label)
    try:
        resolved.relative_to(manifest_path.parent.resolve(strict=True))
    except ValueError as error:
        raise BundleBuilderError(f"{label} escapes its manifest directory") from error
    return resolved


def _validate_bound_artifact(record: Any, label: str) -> Path:
    _require(isinstance(record, Mapping), f"{label} binding must be an object")
    _require(
        set(record) == {"path", "sha256", "size_bytes"}
        and _is_sha256(record.get("sha256"))
        and isinstance(record.get("size_bytes"), int)
        and not isinstance(record.get("size_bytes"), bool)
        and int(record["size_bytes"]) > 0,
        f"{label} binding schema is invalid",
    )
    path = _root_record_path(str(record["path"]), label)
    _require(path.stat().st_size == record["size_bytes"], f"{label} size changed")
    _require(sha256_file(path) == record["sha256"], f"{label} hash changed")
    return path


def _validate_unopened_artifact_record(record: Any, label: str) -> None:
    """Validate an upstream provenance record without opening its large payload."""

    _require(isinstance(record, Mapping), f"{label} binding must be an object")
    _require(
        set(record) == {"path", "sha256", "size_bytes"}
        and isinstance(record.get("path"), str)
        and bool(record.get("path"))
        and _is_sha256(record.get("sha256"))
        and isinstance(record.get("size_bytes"), int)
        and not isinstance(record.get("size_bytes"), bool)
        and int(record["size_bytes"]) > 0,
        f"{label} binding schema is invalid",
    )
    DECODER.frozen_lexical_path_preflight([Path(str(record["path"]))])


def _expected_array_contract(name: str) -> tuple[tuple[int, ...], np.dtype[Any], str]:
    if name == "global_index":
        return (1606,), np.dtype("<i8"), "little-endian-int64"
    _require(name in DECODER.BASE_BLOCK_WIDTHS, f"unknown source feature block: {name}")
    return (
        (1606, int(DECODER.BASE_BLOCK_WIDTHS[name])),
        np.dtype("<f8"),
        "little-endian-float64",
    )


def _validate_source_output(
    *,
    manifest_path: Path,
    output: Any,
    expected_keys: Sequence[str],
    label: str,
) -> tuple[dict[str, np.ndarray], Path, dict[str, Any]]:
    _require(isinstance(output, Mapping), f"{label} output must be an object")
    _require(
        set(output)
        == {
            "path",
            "sha256",
            "size_bytes",
            "keys",
            "numeric_tensor_only",
            "reload_verified",
            "arrays",
        },
        f"{label} output schema changed",
    )
    expected = list(expected_keys)
    _require(
        output.get("keys") == expected
        and output.get("numeric_tensor_only") is True
        and output.get("reload_verified") is True
        and _is_sha256(output.get("sha256"))
        and isinstance(output.get("size_bytes"), int)
        and not isinstance(output.get("size_bytes"), bool)
        and int(output["size_bytes"]) > 0,
        f"{label} output contract changed",
    )
    array_records = output.get("arrays")
    _require(
        isinstance(array_records, Mapping) and set(array_records) == set(expected),
        f"{label} output array registry changed",
    )
    for name in expected:
        record = array_records[name]
        shape, _dtype, manifest_dtype = _expected_array_contract(name)
        _require(
            isinstance(record, Mapping)
            and set(record) == {"shape", "dtype", "logical_sha256"}
            and record.get("shape") == list(shape)
            and record.get("dtype") == manifest_dtype
            and _is_sha256(record.get("logical_sha256")),
            f"{label} manifest array contract changed: {name}",
        )

    data_path = _source_data_path(manifest_path, output.get("path"), f"{label} data")
    _require(data_path.stat().st_size == output["size_bytes"], f"{label} data size changed")
    expected_data_sha256 = str(output["sha256"])
    _require(sha256_file(data_path) == expected_data_sha256, f"{label} data hash changed")
    try:
        with np.load(data_path, allow_pickle=False) as archive:
            _require(archive.files == expected, f"{label} NPZ key order changed")
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise BundleBuilderError(f"cannot load authenticated {label} NPZ: {error}") from error
    _require(sha256_file(data_path) == expected_data_sha256, f"{label} data changed during reload")

    normalized_records: dict[str, Any] = {}
    for name in expected:
        values = arrays[name]
        shape, dtype, _manifest_dtype = _expected_array_contract(name)
        _require(
            values.shape == shape
            and values.dtype == dtype
            and values.dtype.kind != "O"
            and (values.dtype.kind != "f" or bool(np.all(np.isfinite(values)))),
            f"{label} NPZ array geometry, dtype, or finiteness changed: {name}",
        )
        observed_logical = logical_array_sha256(name, values)
        _require(
            observed_logical == array_records[name]["logical_sha256"],
            f"{label} logical array hash changed: {name}",
        )
        arrays[name] = np.ascontiguousarray(values)
        normalized_records[name] = {
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "logical_sha256": observed_logical,
        }
    return arrays, data_path, normalized_records


def _validate_alignment_binding(
    record: Any,
    *,
    alignment_path: Path,
    expected_alignment_sha256: str,
    label: str,
    with_counts: bool,
) -> None:
    required = {"path", "sha256", "size_bytes"}
    if with_counts:
        required |= {"row_count", "stable_row_count"}
    _require(isinstance(record, Mapping) and set(record) == required, f"{label} schema changed")
    _require(
        record.get("sha256") == expected_alignment_sha256
        and record.get("size_bytes") == alignment_path.stat().st_size
        and (not with_counts or record.get("row_count") == 1708)
        and (not with_counts or record.get("stable_row_count") == 1606),
        f"{label} identity changed",
    )
    bound_path = _root_record_path(str(record.get("path")), label)
    _require(bound_path == alignment_path, f"{label} path differs from requested alignment")


def _validate_common_source_identity(
    manifest: Mapping[str, Any],
    *,
    kind: str,
    status_scope: str,
    label: str,
) -> None:
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("kind") == kind
        and manifest.get("status") == "passed"
        and manifest.get("status_scope") == status_scope
        and manifest.get("pre_and_post_input_bindings_equal") is True
        and manifest.get("forbidden_path_guard_passed") is True
        and manifest.get("reserved_validation_access_authorized") is False,
        f"{label} identity, status, or authentication boundary changed",
    )
    claims = manifest.get("claim_scope")
    _require(
        isinstance(claims, Mapping)
        and bool(claims)
        and all(value is False for value in claims.values()),
        f"{label} claim scope changed",
    )


def _source_record(
    *,
    role: str,
    kind: str,
    manifest_path: Path,
    data_path: Path,
    array_records: Mapping[str, Any],
    selected_arrays: Sequence[str],
    expected_alignment_sha256: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "kind": kind,
        "manifest": _artifact_record(manifest_path),
        "data": {
            **_artifact_record(data_path),
            "format": FORMAT,
            "array_keys": list(selected_arrays),
            "arrays": {name: dict(array_records[name]) for name in selected_arrays},
        },
        "selected_arrays": list(selected_arrays),
        "alignment_index_sha256": expected_alignment_sha256,
    }


def _load_visible_source(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    alignment_path: Path,
    expected_alignment_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
    _require(sha256_file(manifest_path) == expected_manifest_sha256, "visible manifest hash changed")
    manifest = _load_json_strict(manifest_path, "visible source manifest")
    _require(set(manifest) == VISIBLE_MANIFEST_KEYS, "visible source manifest schema changed")
    _validate_common_source_identity(
        manifest,
        kind=VISIBLE_KIND,
        status_scope="authenticated_visible_precompletion_numeric_baselines_only",
        label="visible source",
    )
    _validate_bound_artifact(manifest["config"], "visible source config")
    _validate_bound_artifact(manifest["implementation"], "visible source implementation")
    dependencies = manifest["local_code_dependencies"]
    _require(isinstance(dependencies, list), "visible local dependency registry changed")
    for position, dependency in enumerate(dependencies):
        _require(
            isinstance(dependency, Mapping)
            and set(dependency) == {"role", "path", "sha256", "size_bytes"}
            and isinstance(dependency.get("role"), str)
            and bool(dependency.get("role")),
            f"visible local dependency {position} schema changed",
        )
        _validate_bound_artifact(
            {key: dependency[key] for key in ("path", "sha256", "size_bytes")},
            f"visible local dependency {position}",
        )
    inputs = manifest["inputs"]
    _require(
        isinstance(inputs, Mapping)
        and set(inputs)
        == {
            "development_prompts",
            "development_public_report",
            "label_free_alignment_index",
            "v3_action_protocol",
            "v3_protocol",
        },
        "visible source input registry changed",
    )
    for name, record in inputs.items():
        if name != "label_free_alignment_index":
            _validate_unopened_artifact_record(record, f"visible upstream input {name}")
    _validate_alignment_binding(
        inputs["label_free_alignment_index"],
        alignment_path=alignment_path,
        expected_alignment_sha256=expected_alignment_sha256,
        label="visible alignment binding",
        with_counts=False,
    )
    coverage = manifest["coverage"]
    _require(
        isinstance(coverage, Mapping)
        and coverage.get("all_boundary_count") == 1708
        and coverage.get("stable_row_count") == 1606
        and coverage.get("numerically_unstable_row_count") == 102
        and coverage.get("stable_source_identity_order_exact") is True,
        "visible source coverage changed",
    )
    variants = manifest["variants"]
    _require(isinstance(variants, Mapping) and set(variants) == set(VISIBLE_KEYS[1:]), "visible variants changed")
    for name in VISIBLE_KEYS[1:]:
        _require(
            isinstance(variants[name], Mapping)
            and variants[name].get("width") == DECODER.BASE_BLOCK_WIDTHS[name],
            f"visible variant width changed: {name}",
        )
    boundary = manifest["feature_boundary"]
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("label_sidecar_accepted") is False
        and boundary.get("semantic_ids_as_features_forbidden") is True
        and boundary.get("repository_as_feature_forbidden") is True,
        "visible source feature boundary changed",
    )
    arrays, data_path, records = _validate_source_output(
        manifest_path=manifest_path,
        output=manifest["output"],
        expected_keys=VISIBLE_KEYS,
        label="visible source",
    )
    _require(sha256_file(manifest_path) == expected_manifest_sha256, "visible manifest changed during reload")
    return arrays, _source_record(
        role=VISIBLE_ROLE,
        kind=VISIBLE_KIND,
        manifest_path=manifest_path,
        data_path=data_path,
        array_records=records,
        selected_arrays=VISIBLE_KEYS,
        expected_alignment_sha256=expected_alignment_sha256,
    ), data_path


def _load_activation_source(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    alignment_path: Path,
    expected_alignment_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
    _require(sha256_file(manifest_path) == expected_manifest_sha256, "activation manifest hash changed")
    manifest = _load_json_strict(manifest_path, "activation source manifest")
    _require(set(manifest) == ACTIVATION_MANIFEST_KEYS, "activation source manifest schema changed")
    _validate_common_source_identity(
        manifest,
        kind=ACTIVATION_KIND,
        status_scope="label_free_primary_seed_current_and_causal_sequence_tensors_only",
        label="activation source",
    )
    for key, label in (
        ("feature_config", "activation feature config"),
        ("implementation", "activation campaign implementation"),
        ("feature_implementation", "activation feature implementation"),
        ("projection_config", "activation projection config"),
        ("projection_implementation", "activation projection implementation"),
    ):
        _validate_bound_artifact(manifest[key], label)
    inputs = manifest["inputs"]
    _require(
        isinstance(inputs, Mapping)
        and set(inputs) == {"alignment_index", "projection_chunks"}
        and isinstance(inputs["projection_chunks"], list)
        and len(inputs["projection_chunks"]) == 4,
        "activation source input registry changed",
    )
    _validate_alignment_binding(
        inputs["alignment_index"],
        alignment_path=alignment_path,
        expected_alignment_sha256=expected_alignment_sha256,
        label="activation alignment binding",
        with_counts=True,
    )
    coverage = manifest["coverage"]
    _require(
        isinstance(coverage, Mapping)
        and coverage.get("boundary_count") == 1708
        and coverage.get("stable_feature_count") == 1606
        and coverage.get("source_id_order_matches_alignment") is True
        and coverage.get("source_id_coverage_exact") is True
        and coverage.get("chunk_order_exact") is True,
        "activation source coverage changed",
    )
    variants = manifest["variants"]
    _require(isinstance(variants, Mapping) and set(variants) == set(ACTIVATION_KEYS[1:]), "activation variants changed")
    for name in ACTIVATION_KEYS[1:]:
        _require(
            variants[name] == DECODER.BASE_BLOCK_WIDTHS[name],
            f"activation variant width changed: {name}",
        )
    boundary = manifest["feature_boundary"]
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("labels_or_outcomes_accepted") is False
        and boundary.get("semantic_ids_as_features_forbidden") is True
        and boundary.get("repository_as_feature_forbidden") is True,
        "activation source feature boundary changed",
    )
    arrays, data_path, records = _validate_source_output(
        manifest_path=manifest_path,
        output=manifest["output"],
        expected_keys=ACTIVATION_KEYS,
        label="activation source",
    )
    _require(sha256_file(manifest_path) == expected_manifest_sha256, "activation manifest changed during reload")
    return arrays, _source_record(
        role=ACTIVATION_ROLE,
        kind=ACTIVATION_KIND,
        manifest_path=manifest_path,
        data_path=data_path,
        array_records=records,
        selected_arrays=ACTIVATION_KEYS,
        expected_alignment_sha256=expected_alignment_sha256,
    ), data_path


def _bundle_array_records(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(np.asarray(arrays[name]).shape),
            "dtype": np.asarray(arrays[name]).dtype.str,
            "logical_sha256": logical_array_sha256(name, np.asarray(arrays[name])),
        }
        for name in sorted(arrays)
    }


def _reload_bundle_exact(path: Path, expected: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    expected_keys = sorted(expected)
    try:
        with np.load(path, allow_pickle=False) as archive:
            _require(archive.files == expected_keys, "bundle NPZ key order changed")
            observed = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise BundleBuilderError(f"cannot reload bundle NPZ: {error}") from error
    for name in expected_keys:
        left = observed[name]
        right = np.asarray(expected[name])
        _require(
            left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right, equal_nan=False),
            f"bundle NPZ reload changed array: {name}",
        )
    return observed


def _assert_source_bundle_equality(
    bundle: Mapping[str, np.ndarray],
    visible: Mapping[str, np.ndarray],
    activation: Mapping[str, np.ndarray],
) -> None:
    for source in (visible, activation):
        _require(
            np.array_equal(bundle["global_index"], source["global_index"]),
            "source global_index differs from bundle alignment",
        )
    for role, source in ((VISIBLE_ROLE, visible), (ACTIVATION_ROLE, activation)):
        for name in SOURCE_FEATURE_BLOCKS[role]:
            _require(
                np.asarray(bundle[name]).dtype == np.asarray(source[name]).dtype
                and np.array_equal(bundle[name], source[name], equal_nan=False)
                and logical_array_sha256(name, bundle[name])
                == logical_array_sha256(name, source[name]),
                f"source block differs from bundle: {role}/{name}",
            )


def build_feature_bundle_from_authenticated_sources(
    *,
    alignment_index_path: Path,
    expected_alignment_index_sha256: str,
    visible_manifest_path: Path,
    expected_visible_manifest_sha256: str,
    activation_manifest_path: Path,
    expected_activation_manifest_sha256: str,
    output_data_path: Path,
    output_manifest_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate both source NPZs and atomically write their exact joined bundle."""

    input_paths = [alignment_index_path, visible_manifest_path, activation_manifest_path]
    output_paths = [output_data_path, output_manifest_path]
    # This is intentionally the first operation on caller-controlled paths.
    DECODER.frozen_lexical_path_preflight([*input_paths, *output_paths])
    DECODER.frozen_canonical_path_preflight(
        input_paths=input_paths,
        output_paths=output_paths,
    )
    validated_config = DECODER.validate_config(config)
    for digest, label in (
        (expected_alignment_index_sha256, "alignment index"),
        (expected_visible_manifest_sha256, "visible source manifest"),
        (expected_activation_manifest_sha256, "activation source manifest"),
    ):
        _require(_is_sha256(digest), f"{label} expected SHA-256 is invalid")

    alignment_path = _regular_input(Path(alignment_index_path), "alignment index")
    visible_path = _regular_input(Path(visible_manifest_path), "visible source manifest")
    activation_path = _regular_input(Path(activation_manifest_path), "activation source manifest")
    data_path = Path(output_data_path).resolve(strict=False)
    manifest_path = Path(output_manifest_path).resolve(strict=False)
    _require(
        data_path != manifest_path
        and data_path.parent == manifest_path.parent
        and data_path.suffix == ".npz"
        and manifest_path.suffix == ".json",
        "bundle outputs must be distinct .npz/.json files in one directory",
    )
    _require(
        not data_path.exists()
        and not data_path.is_symlink()
        and not manifest_path.exists()
        and not manifest_path.is_symlink(),
        "refusing to overwrite feature bundle output",
    )
    _require(
        sha256_file(alignment_path) == expected_alignment_index_sha256,
        "alignment index hash changed",
    )
    alignment_value = _load_json_strict(alignment_path, "alignment index")
    try:
        alignment_rows = DECODER.validate_alignment_index(alignment_value)
    except DECODER.DecoderError as error:
        raise BundleBuilderError(f"alignment index contract changed: {error}") from error

    visible_arrays, visible_record, visible_data_path = _load_visible_source(
        manifest_path=visible_path,
        expected_manifest_sha256=expected_visible_manifest_sha256,
        alignment_path=alignment_path,
        expected_alignment_sha256=expected_alignment_index_sha256,
    )
    activation_arrays, activation_record, activation_data_path = _load_activation_source(
        manifest_path=activation_path,
        expected_manifest_sha256=expected_activation_manifest_sha256,
        alignment_path=alignment_path,
        expected_alignment_sha256=expected_alignment_index_sha256,
    )
    expected_global_index = np.asarray(
        [row["global_index"] for row in alignment_rows if row["stable_feature_eligible"]],
        dtype="<i8",
    )
    _require(
        np.array_equal(visible_arrays["global_index"], expected_global_index)
        and np.array_equal(activation_arrays["global_index"], expected_global_index),
        "source global_index order differs from stable alignment",
    )
    try:
        assembled = DECODER.assemble_label_free_feature_arrays(
            alignment_rows=alignment_rows,
            activation_features=activation_arrays,
            visible_baselines=visible_arrays,
        )
        validated = DECODER.validate_feature_arrays(
            assembled,
            alignment_rows=alignment_rows,
        )
    except DECODER.DecoderError as error:
        raise BundleBuilderError(f"source feature assembly failed: {error}") from error
    _assert_source_bundle_equality(validated, visible_arrays, activation_arrays)

    authenticated_inputs = [
        alignment_path,
        visible_path,
        visible_data_path,
        activation_path,
        activation_data_path,
        CONFIG_PATH,
        DECODER.SCRIPT_PATH,
        SCRIPT_PATH,
    ]
    pre_hashes = {path: sha256_file(path) for path in authenticated_inputs}
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_data = data_path.with_name(f".{data_path.stem}.tmp-{os.getpid()}.npz")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    _require(
        not temporary_data.exists()
        and not temporary_data.is_symlink()
        and not temporary_manifest.exists()
        and not temporary_manifest.is_symlink(),
        "bundle temporary output already exists",
    )
    data_committed = False
    manifest_committed = False
    try:
        ordered = {name: validated[name] for name in sorted(validated)}
        np.savez_compressed(temporary_data, **ordered)
        _reload_bundle_exact(temporary_data, ordered)
        os.replace(temporary_data, data_path)
        data_committed = True
        reloaded = _reload_bundle_exact(data_path, ordered)
        _assert_source_bundle_equality(reloaded, visible_arrays, activation_arrays)
        for path, expected_hash in pre_hashes.items():
            _require(sha256_file(path) == expected_hash, f"authenticated input changed during bundle construction: {path}")

        array_records = _bundle_array_records(reloaded)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "status": "passed",
            "scope": validated_config["artifacts"]["feature_bundle_scope"],
            "producer": _artifact_record(SCRIPT_PATH),
            "decoder_contract": {
                "config": _artifact_record(CONFIG_PATH),
                "implementation": _artifact_record(DECODER.SCRIPT_PATH),
            },
            "data": {
                "path": data_path.name,
                "sha256": sha256_file(data_path),
                "size_bytes": data_path.stat().st_size,
                "format": FORMAT,
                "array_keys": sorted(reloaded),
                "arrays": array_records,
                "reload_verified": True,
            },
            "alignment_index": _artifact_record(alignment_path),
            "sources": [visible_record, activation_record],
            "source_feature_blocks": {
                role: list(names) for role, names in SOURCE_FEATURE_BLOCKS.items()
            },
            "construction": {
                "algorithm": CONSTRUCTION_ALGORITHM,
                "join_key": "global_index",
                "stable_alignment_only": True,
                "caller_supplied_feature_arrays_accepted": False,
                "labels_targets_outcomes_or_completion_text_opened": False,
                "source_manifest_hashes_verified": True,
                "source_data_hashes_verified": True,
                "source_logical_hashes_verified": True,
                "source_global_indices_equal_alignment": True,
                "source_arrays_equal_bundle_arrays": True,
                "bundle_reload_verified": True,
                "pre_and_post_source_bindings_equal": True,
            },
            "row_count": 1606,
            "metadata_arrays": list(DECODER.METADATA_ARRAYS),
            "base_feature_blocks": dict(DECODER.BASE_BLOCK_WIDTHS),
            "forbidden_array_fields_absent": list(
                validated_config["artifacts"]["forbidden_array_fields"]
            ),
            "grouping_arrays_never_numeric_features": list(
                validated_config["artifacts"]["grouping_arrays_never_numeric_features"]
            ),
            "claim_scope": {
                "labels_or_outcomes_present": False,
                "semantic_grouping_fields_used_as_features": False,
                "private_chain_of_thought_reconstructed": False,
                "emotion_decoding_established": False,
            },
        }
        rendered = json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
        temporary_manifest.write_text(rendered, encoding="utf-8")
        for path, expected_hash in pre_hashes.items():
            _require(sha256_file(path) == expected_hash, f"authenticated input changed before manifest commit: {path}")
        os.replace(temporary_manifest, manifest_path)
        manifest_committed = True
        _require(
            _load_json_strict(manifest_path, "written bundle manifest") == manifest,
            "written bundle manifest reload changed",
        )
        return manifest
    finally:
        for temporary in (temporary_data, temporary_manifest):
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
        if data_committed and not manifest_committed and data_path.exists() and not data_path.is_symlink():
            data_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--alignment-index", type=Path, required=True)
    parser.add_argument("--expected-alignment-index-sha256", required=True)
    parser.add_argument("--visible-manifest", type=Path, required=True)
    parser.add_argument("--expected-visible-manifest-sha256", required=True)
    parser.add_argument("--activation-manifest", type=Path, required=True)
    parser.add_argument("--expected-activation-manifest-sha256", required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    # The lexical gate is intentionally the first post-parse operation.
    DECODER.frozen_lexical_path_preflight(
        [
            args.config,
            args.alignment_index,
            args.visible_manifest,
            args.activation_manifest,
            args.output_data,
            args.output_manifest,
        ]
    )
    DECODER.frozen_canonical_path_preflight(
        input_paths=[
            args.config,
            args.alignment_index,
            args.visible_manifest,
            args.activation_manifest,
        ],
        output_paths=[args.output_data, args.output_manifest],
    )
    config_path = _regular_input(args.config, "decoder config")
    _require(config_path == CONFIG_PATH, "decoder config path changed")
    config = _load_json_strict(config_path, "decoder config")
    manifest = build_feature_bundle_from_authenticated_sources(
        alignment_index_path=args.alignment_index,
        expected_alignment_index_sha256=args.expected_alignment_index_sha256,
        visible_manifest_path=args.visible_manifest,
        expected_visible_manifest_sha256=args.expected_visible_manifest_sha256,
        activation_manifest_path=args.activation_manifest,
        expected_activation_manifest_sha256=args.expected_activation_manifest_sha256,
        output_data_path=args.output_data,
        output_manifest_path=args.output_manifest,
        config=config,
    )
    print(
        f"wrote {manifest['row_count']} source-authenticated feature rows "
        f"to {args.output_data}"
    )
    return 0


__all__ = [
    "ACTIVATION_KEYS",
    "ACTIVATION_KIND",
    "ACTIVATION_ROLE",
    "BundleBuilderError",
    "CONSTRUCTION_ALGORITHM",
    "KIND",
    "SCHEMA_VERSION",
    "SOURCE_FEATURE_BLOCKS",
    "VISIBLE_KEYS",
    "VISIBLE_KIND",
    "VISIBLE_ROLE",
    "build_feature_bundle_from_authenticated_sources",
    "build_parser",
    "logical_array_sha256",
    "run",
    "sha256_file",
]


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
