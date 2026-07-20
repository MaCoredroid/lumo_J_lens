#!/usr/bin/env python3
"""Build authenticated, numeric-only V4 visible word-probe baselines.

The large development prompt/report pair is consumed only through the frozen
V4 streaming extractor.  This wrapper deliberately does not call V3's broad
``validate_protocol`` entry point: that validator authenticates a historical
selection proof in the closed validation tree.  Instead, this wrapper byte-
authenticates the exact V3 protocol and action registry and derives only the
small extraction view needed by ``extract_stable_rows_streaming``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from scripts import swe_task_state_v4_extract as EXTRACT
except ModuleNotFoundError as error:  # Support direct execution from scripts/.
    if error.name != "scripts":
        raise
    import swe_task_state_v4_extract as EXTRACT  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_visible_baselines.json"
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_SHA256 = "6ada08d478be598f606660acec3064dd659023060d4effbfec7989ad15ced58d"
SCHEMA_VERSION = 1
KIND = "swe_task_state_v4_visible_word_probe_baseline_features"
FORBIDDEN_PATH_FRAGMENTS = ("reserved", "validation")
VARIANTS = ("history_only", "sequence_logit", "sequence_j", "sequence_logit_j")
VARIANT_WIDTHS = {
    "history_only": 14,
    "sequence_logit": 136,
    "sequence_j": 136,
    "sequence_logit_j": 256,
}
OUTPUT_KEYS = ("global_index", *VARIANTS)


class BaselineError(ValueError):
    """Raised when an authentication or feature-boundary check fails closed."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BaselineError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineError(f"cannot read JSON: {path}: {exc}") from exc


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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BaselineError(f"{label} must be a lowercase SHA-256 digest")
    return value


def lexical_path_preflight(paths: Iterable[Path | None]) -> None:
    """Reject closed-tree text before any filesystem read, hash, or resolve."""

    for path in paths:
        if path is None:
            continue
        normalized = os.path.normpath(os.path.abspath(os.fspath(path))).lower()
        if any(
            fragment in component
            for fragment in FORBIDDEN_PATH_FRAGMENTS
            for component in Path(normalized).parts
        ):
            raise BaselineError(
                f"forbidden path rejected before filesystem access: {path}"
            )


def canonical_path_preflight(
    *, input_paths: Iterable[Path], output_paths: Iterable[Path]
) -> None:
    """Reject forbidden canonical parents after the lexical first gate."""

    for path, strict in [
        *((path, True) for path in input_paths),
        *((path, False) for path in output_paths),
    ]:
        try:
            resolved = path.resolve(strict=strict)
        except OSError as exc:
            raise BaselineError(f"cannot resolve path metadata: {path}: {exc}") from exc
        if any(
            fragment in component.lower()
            for fragment in FORBIDDEN_PATH_FRAGMENTS
            for component in resolved.parts
        ):
            raise BaselineError(f"forbidden canonical path rejected: {path}")


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise BaselineError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BaselineError(f"cannot resolve {label}: {path}: {exc}") from exc
    if not resolved.is_file():
        raise BaselineError(f"{label} must be a regular file: {path}")
    return resolved


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


def _validate_bound_record(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
        raise BaselineError(f"{label} binding schema changed")
    if (
        not isinstance(value["path"], str)
        or not value["path"]
        or Path(value["path"]).is_absolute()
        or isinstance(value["size_bytes"], bool)
        or not isinstance(value["size_bytes"], int)
        or value["size_bytes"] < 1
    ):
        raise BaselineError(f"{label} binding is invalid")
    _require_sha256(value["sha256"], f"{label} SHA-256")
    return value


def validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict) or set(config) != {
        "schema_version",
        "id",
        "status",
        "inputs",
        "local_code_dependencies",
        "campaign",
        "variants",
        "authentication",
        "feature_boundary",
        "claim_scope",
    }:
        raise BaselineError("visible-baseline config schema changed")
    if (
        config["schema_version"] != SCHEMA_VERSION
        or config["id"]
        != "swe-task-state-v4-visible-word-probe-baseline-features"
        or config["status"] != "development_only_reserved_validation_closed"
    ):
        raise BaselineError("visible-baseline config identity changed")
    inputs = config["inputs"]
    if not isinstance(inputs, dict) or tuple(inputs) != (
        "development_prompts",
        "development_public_report",
        "v3_protocol",
        "v3_action_protocol",
        "label_free_alignment_index",
    ):
        raise BaselineError("visible-baseline input registry changed")
    expected_input_hashes = {
        "development_prompts": "17f664b3029220458ff62b8e80a90ec5e796f0372217f02b727d6589df38d3d0",
        "development_public_report": "7c943132163749f69bd35e4fa2e52bcfee2318fe349fa77603324a37ffaabe46",
        "v3_protocol": "9d8b0a7d5c45dc192365429af27c6193de752cc160458eff8e21807d37662b1d",
        "v3_action_protocol": "0ebd258a2b46beb2a9be3d42cab24680803a2f971cb21e96acecb78e19cd81bf",
        "label_free_alignment_index": "c0d9e4bd6ad8f6962cf58776441ec4ef042eebbbae8a9959b4149e605c9ae38e",
    }
    for label, expected_hash in expected_input_hashes.items():
        record = _validate_bound_record(inputs[label], label=label)
        if record["sha256"] != expected_hash:
            raise BaselineError(f"{label} frozen SHA-256 changed")
    dependencies = config["local_code_dependencies"]
    expected_roles = (
        "v4_streaming_extractor",
        "v4_sequence_feature_builder",
        "frozen_v3_analyzer",
        "frozen_v1_analyzer",
        "frozen_v2_analyzer",
        "frozen_v3_bundle_checker_import",
        "frozen_v3_replay_import",
    )
    if (
        not isinstance(dependencies, list)
        or tuple(item.get("role") for item in dependencies if isinstance(item, dict))
        != expected_roles
    ):
        raise BaselineError("local code dependency registry changed")
    for position, dependency in enumerate(dependencies):
        if not isinstance(dependency, dict) or set(dependency) != {
            "role",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise BaselineError(f"local code dependency {position} schema changed")
        _validate_bound_record(
            {key: dependency[key] for key in ("path", "sha256", "size_bytes")},
            label=f"local code dependency {position}",
        )
    campaign = config["campaign"]
    if campaign != {
        "all_boundary_count": 1708,
        "stable_row_count": 1606,
        "numerically_unstable_row_count": 102,
        "output_tensor_keys": list(OUTPUT_KEYS),
        "output_dtypes": {
            "global_index": "little-endian-int64",
            **{name: "little-endian-float64" for name in VARIANTS},
        },
    }:
        raise BaselineError("visible-baseline campaign contract changed")
    variants = config["variants"]
    if not isinstance(variants, dict) or tuple(variants) != VARIANTS:
        raise BaselineError("visible-baseline variant order changed")
    for name, width in VARIANT_WIDTHS.items():
        value = variants[name]
        if (
            not isinstance(value, dict)
            or set(value) != {"width", "definition"}
            or value["width"] != width
            or not isinstance(value["definition"], str)
            or not value["definition"]
        ):
            raise BaselineError(f"visible-baseline {name} contract changed")
    authentication = config["authentication"]
    if (
        not isinstance(authentication, dict)
        or authentication.get("forbidden_path_fragments")
        != list(FORBIDDEN_PATH_FRAGMENTS)
        or any(
            value is not True
            for key, value in authentication.items()
            if key != "forbidden_path_fragments"
        )
    ):
        raise BaselineError("visible-baseline authentication contract changed")
    boundary = config["feature_boundary"]
    if (
        not isinstance(boundary, dict)
        or boundary.get("allowed_numeric_feature_inputs")
        != [f"frozen_extractor.features.{name}" for name in VARIANTS]
        or boundary.get("label_sidecar_accepted") is not False
        or boundary.get("semantic_ids_as_features_forbidden") is not True
        or boundary.get("repository_as_feature_forbidden") is not True
        or not isinstance(boundary.get("forbidden_as_features"), list)
        or "current source action label or collapsed target"
        not in boundary["forbidden_as_features"]
        or "current or future completion text" not in boundary["forbidden_as_features"]
        or "IDs hashes or repository indicators" not in boundary["forbidden_as_features"]
    ):
        raise BaselineError("visible-baseline feature boundary changed")
    claims = config["claim_scope"]
    if not isinstance(claims, dict) or set(claims) != {
        "private_chain_of_thought_reconstructed",
        "cot_like_observable_event_decoding_established",
        "emotion_decoding_established",
        "causal_interpretation_established",
        "repository_held_out_predictability_established",
        "incremental_value_over_visible_baselines_established",
    } or any(value is not False for value in claims.values()):
        raise BaselineError("visible baselines cannot establish interpretive claims")
    return config


def resolve_and_authenticate_records(
    records: Sequence[Mapping[str, Any]], *, label: str
) -> list[Path]:
    paths: list[Path] = []
    for position, record in enumerate(records):
        path = ROOT / str(record["path"])
        path = _require_regular_file(path, f"{label} {position}")
        if (
            path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise BaselineError(f"{label} {position} byte binding changed")
        paths.append(path)
    return paths


def build_extraction_protocol(
    protocol_value: Any, action_protocol_value: Any
) -> dict[str, Any]:
    """Derive the exact extraction-only view without opening closed artifacts."""

    if not isinstance(protocol_value, dict) or (
        protocol_value.get("schema_version"), protocol_value.get("id")
    ) != (1, "swe-task-state-interpreter-v3"):
        raise BaselineError("V3 protocol identity changed")
    if not isinstance(action_protocol_value, dict) or (
        action_protocol_value.get("schema_version"), action_protocol_value.get("kind")
    ) != (1, "swe_verified_stage_action_probe_protocol"):
        raise BaselineError("V3 action protocol identity changed")
    pins = protocol_value.get("pins")
    feature = protocol_value.get("feature_contract")
    target = protocol_value.get("target_contract")
    eligibility = protocol_value.get("eligibility_contract")
    if not all(isinstance(value, dict) for value in (pins, feature, target, eligibility)):
        raise BaselineError("V3 extraction contract is incomplete")
    assert isinstance(pins, dict)
    assert isinstance(feature, dict)
    assert isinstance(target, dict)
    assert isinstance(eligibility, dict)
    if pins.get("v3_action_protocol_sha256") != (
        "0ebd258a2b46beb2a9be3d42cab24680803a2f971cb21e96acecb78e19cd81bf"
    ):
        raise BaselineError("V3 action-protocol pin changed")
    layers = feature.get("source_layers")
    source_class_ids = target.get("source_action_classes_in_order")
    if layers != list(range(24, 48)) or source_class_ids != [
        "inspect",
        "edit",
        "validate",
        "finalize",
    ]:
        raise BaselineError("V3 word-probe source order changed")
    action_records = action_protocol_value.get("action_classes")
    if not isinstance(action_records, list) or [
        record.get("id") if isinstance(record, dict) else None
        for record in action_records
    ] != source_class_ids:
        raise BaselineError("V3 action class order changed")
    token_ids_by_class: dict[str, list[int]] = {}
    observed_token_ids: list[int] = []
    for record in action_records:
        assert isinstance(record, dict)
        tokens = record.get("tokens")
        if not isinstance(tokens, list) or len(tokens) != 8:
            raise BaselineError("each V3 action class must contain eight tokens")
        token_ids = [
            token.get("token_id") if isinstance(token, dict) else None
            for token in tokens
        ]
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in token_ids
        ):
            raise BaselineError("V3 action token IDs changed")
        class_id = str(record["id"])
        token_ids_by_class[class_id] = [int(token_id) for token_id in token_ids]
        observed_token_ids.extend(token_ids_by_class[class_id])
    if len(observed_token_ids) != len(set(observed_token_ids)):
        raise BaselineError("V3 action token IDs overlap")
    stability = eligibility.get("numerical_stability")
    if not isinstance(stability, dict) or (
        stability.get("final_logits_rms_error_maximum_inclusive"),
        stability.get("final_logits_max_abs_error_maximum_inclusive"),
    ) != (0.02, 0.125):
        raise BaselineError("V3 numerical-stability thresholds changed")
    model = pins.get("model")
    lens = pins.get("public_lens")
    runtime = pins.get("replay_runtime")
    if not all(isinstance(value, dict) for value in (model, lens, runtime)):
        raise BaselineError("V3 replay pins are incomplete")
    assert isinstance(model, dict)
    assert isinstance(lens, dict)
    assert isinstance(runtime, dict)
    model_keys = ("repo_id", "revision", "config_sha256", "index_sha256")
    lens_keys = ("repo_id", "revision", "sha256", "n_prompts")
    runtime_keys = (
        "enforce_eager",
        "mtp_enabled",
        "max_model_len",
        "max_num_batched_tokens",
        "mamba_block_size",
        "kv_cache_dtype",
        "kv_offloading_size",
        "kv_offloading_backend",
        "stream_final_only",
    )
    try:
        report_pins = {
            "model": {key: model[key] for key in model_keys},
            "public_lens": {key: lens[key] for key in lens_keys},
            "runtime": {key: runtime[key] for key in runtime_keys},
        }
    except KeyError as exc:
        raise BaselineError(f"V3 replay pin is missing: {exc}") from exc
    return {
        "source_class_ids": list(source_class_ids),
        "layers": list(layers),
        "token_ids_by_class": token_ids_by_class,
        "report_helper_protocol": {"report_pins": report_pins},
        "eligibility": {"stable_rms": 0.02, "stable_max": 0.125},
    }


def validate_alignment_index(
    index: Any, *, expected_total_count: int, expected_stable_count: int
) -> list[dict[str, Any]]:
    expected_top_keys = {
        "schema_version",
        "kind",
        "status",
        "scope",
        "config",
        "implementation",
        "sources",
        "eligibility_source",
        "row_count",
        "stable_row_count",
        "feature_use",
        "rows",
    }
    if not isinstance(index, dict) or set(index) != expected_top_keys:
        raise BaselineError("alignment index schema changed")
    if (
        index["schema_version"] != 1
        or index["kind"] != "swe_task_state_v4_label_free_alignment_index"
        or index["status"] != "passed"
        or index["scope"] != "grouping_order_and_stability_only_no_labels"
        or index["row_count"] != expected_total_count
        or index["stable_row_count"] != expected_stable_count
        or index["feature_use"]
        != {
            "allowed": [
                "task-local ordering for causal temporal transforms",
                "repository and task grouping for held-out splits and weights",
                "stable eligibility filtering",
            ],
            "forbidden": [
                "hashing or one-hot encoding IDs as model features",
                "repository or request index as semantic model features",
            ],
        }
    ):
        raise BaselineError("alignment index identity or feature-use boundary changed")
    rows = index["rows"]
    if not isinstance(rows, list) or len(rows) != expected_total_count:
        raise BaselineError("alignment row count changed")
    row_keys = {
        "global_index",
        "source_id_sha256",
        "task_id_sha256",
        "repository",
        "request_index",
        "stable_feature_eligible",
    }
    seen_sources: set[str] = set()
    seen_requests: set[tuple[str, str, int]] = set()
    previous_by_task: dict[tuple[str, str], int] = {}
    request_indices_by_task: dict[tuple[str, str], list[int]] = {}
    repository_by_task_id: dict[str, str] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != row_keys:
            raise BaselineError(f"alignment row {position} schema changed")
        source_id = _require_sha256(
            row["source_id_sha256"], f"alignment row {position} source ID"
        )
        task_id = _require_sha256(
            row["task_id_sha256"], f"alignment row {position} task ID"
        )
        repository = row["repository"]
        request_index = row["request_index"]
        stable = row["stable_feature_eligible"]
        if (
            row["global_index"] != position
            or not isinstance(repository, str)
            or not repository
            or isinstance(request_index, bool)
            or not isinstance(request_index, int)
            or request_index < 1
            or not isinstance(stable, bool)
        ):
            raise BaselineError(f"alignment row {position} is invalid")
        if (
            task_id in repository_by_task_id
            and repository_by_task_id[task_id] != repository
        ):
            raise BaselineError("one alignment task maps to multiple repositories")
        repository_by_task_id[task_id] = repository
        task_key = (repository, task_id)
        request_key = (*task_key, request_index)
        previous = previous_by_task.get(task_key)
        if (
            source_id in seen_sources
            or request_key in seen_requests
            or (previous is not None and request_index <= previous)
        ):
            raise BaselineError("alignment identities or task order changed")
        seen_sources.add(source_id)
        seen_requests.add(request_key)
        previous_by_task[task_key] = request_index
        request_indices_by_task.setdefault(task_key, []).append(request_index)
    for request_indices in request_indices_by_task.values():
        if request_indices != list(range(1, len(request_indices) + 1)):
            raise BaselineError("alignment task requests are not complete and consecutive")
    if sum(bool(row["stable_feature_eligible"]) for row in rows) != expected_stable_count:
        raise BaselineError("alignment stable count changed")
    return rows


def _finite_feature_vector(
    row: Mapping[str, Any], name: str, width: int, *, position: int
) -> np.ndarray:
    features = row.get("features")
    if not isinstance(features, Mapping) or set(features) != set(VARIANTS):
        raise BaselineError(f"extracted row {position} feature allowlist changed")
    try:
        values = np.asarray(features[name], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BaselineError(f"extracted row {position} {name} is not numeric") from exc
    if values.shape != (width,) or not np.all(np.isfinite(values)):
        raise BaselineError(
            f"extracted row {position} {name} width or finiteness changed"
        )
    return values


def build_baseline_tensors(
    *,
    extracted_rows: Sequence[Mapping[str, Any]],
    alignment_rows: Sequence[Mapping[str, Any]],
    expected_total_count: int,
    expected_stable_count: int,
) -> dict[str, np.ndarray]:
    """Select only the frozen numeric features; labels are never accepted."""

    if len(alignment_rows) != expected_total_count:
        raise BaselineError("alignment count differs from the frozen campaign")
    stable_pairs = [
        (position, row)
        for position, row in enumerate(alignment_rows)
        if row.get("stable_feature_eligible") is True
    ]
    if len(stable_pairs) != expected_stable_count or len(extracted_rows) != len(
        stable_pairs
    ):
        raise BaselineError("extracted/stable alignment coverage differs")
    tensors: dict[str, list[np.ndarray]] = {name: [] for name in VARIANTS}
    retained_indices: list[int] = []
    for stable_position, (global_index, alignment) in enumerate(stable_pairs):
        row = extracted_rows[stable_position]
        if not isinstance(row, Mapping):
            raise BaselineError(f"extracted row {stable_position} is not a mapping")
        row_id = row.get("row_id")
        task_id = row.get("task_id")
        repo = row.get("repo")
        request_index = row.get("task_request_index")
        if (
            not isinstance(row_id, str)
            or sha256_text(row_id) != alignment["source_id_sha256"]
            or not isinstance(task_id, str)
            or sha256_text(task_id) != alignment["task_id_sha256"]
            or repo != alignment["repository"]
            or request_index != alignment["request_index"]
        ):
            raise BaselineError(
                f"extracted row identity differs from alignment at stable row {stable_position}"
            )
        retained_indices.append(global_index)
        for name, width in VARIANT_WIDTHS.items():
            tensors[name].append(
                _finite_feature_vector(row, name, width, position=stable_position)
            )
    result: dict[str, np.ndarray] = {
        "global_index": np.ascontiguousarray(retained_indices, dtype="<i8")
    }
    for name in VARIANTS:
        result[name] = np.ascontiguousarray(np.stack(tensors[name]), dtype="<f8")
    if tuple(result) != OUTPUT_KEYS:
        raise BaselineError("numeric output allowlist changed")
    for name, values in result.items():
        expected_shape = (
            (expected_stable_count,)
            if name == "global_index"
            else (expected_stable_count, VARIANT_WIDTHS[name])
        )
        if (
            values.shape != expected_shape
            or values.dtype != np.dtype("<i8" if name == "global_index" else "<f8")
            or values.dtype.kind not in {"i", "f"}
            or (values.dtype.kind == "f" and not np.all(np.isfinite(values)))
        ):
            raise BaselineError(f"numeric output {name} contract changed")
    return result


def validate_extraction_coverage(
    extraction: Mapping[str, Any],
    *,
    alignment_rows: Sequence[Mapping[str, Any]],
    expected_total_count: int,
    expected_stable_count: int,
    expected_unstable_count: int,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    rows = extraction.get("rows")
    eligibility = extraction.get("eligibility")
    if not isinstance(rows, list) or not isinstance(eligibility, dict):
        raise BaselineError("streaming extraction result schema changed")
    if (
        eligibility.get("all_replayed_prompt_count") != expected_total_count
        or eligibility.get("numerically_stable_prompt_count")
        != expected_stable_count
        or eligibility.get("stable_feature_complete_prediction_count")
        != expected_stable_count
    ):
        raise BaselineError("streaming extraction eligibility counts changed")
    exclusions = eligibility.get("exclusions")
    if not isinstance(exclusions, list) or len(exclusions) != expected_unstable_count:
        raise BaselineError("streaming extraction exclusion count changed")
    if eligibility.get("exclusion_counts") != {
        "numerically_unstable": expected_unstable_count
    }:
        raise BaselineError("streaming extraction exclusion reasons changed")
    stable_expected = [
        row["source_id_sha256"]
        for row in alignment_rows
        if row["stable_feature_eligible"]
    ]
    unstable_expected = [
        row["source_id_sha256"]
        for row in alignment_rows
        if not row["stable_feature_eligible"]
    ]
    stable_observed = [
        sha256_text(str(row.get("row_id"))) if isinstance(row, Mapping) else ""
        for row in rows
    ]
    unstable_observed: list[str] = []
    for position, exclusion in enumerate(exclusions):
        if (
            not isinstance(exclusion, Mapping)
            or exclusion.get("reason") != "numerically_unstable"
            or not isinstance(exclusion.get("row_id"), str)
        ):
            raise BaselineError(f"streaming exclusion {position} changed")
        unstable_observed.append(sha256_text(str(exclusion["row_id"])))
    if stable_observed != stable_expected or unstable_observed != unstable_expected:
        raise BaselineError("streaming extraction identity order differs from alignment")
    return rows, eligibility


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
    digest = hashlib.sha256()
    header = canonical_json_bytes({"name": name, "count": len(values)})
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _write_npz_no_clobber(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.suffix != ".npz":
        raise BaselineError("baseline data output must end in .npz")
    if path.exists() or path.is_symlink():
        raise BaselineError(f"refusing to overwrite baseline data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    if temporary.exists() or temporary.is_symlink():
        raise BaselineError(f"temporary baseline data exists: {temporary}")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _write_json_no_clobber(path: Path, value: Any) -> None:
    if path.suffix != ".json":
        raise BaselineError("baseline manifest output must end in .json")
    if path.exists() or path.is_symlink():
        raise BaselineError(f"refusing to overwrite baseline manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise BaselineError(f"temporary baseline manifest exists: {temporary}")
    rendered = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    try:
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    # This must remain the first operation: even a stat/hash is forbidden first.
    lexical_path_preflight((args.config, args.output_data, args.output_manifest))
    canonical_path_preflight(
        input_paths=(args.config,),
        output_paths=(args.output_data, args.output_manifest),
    )
    config_path = _require_regular_file(args.config, "visible-baseline config")
    if config_path != CONFIG_PATH or sha256_file(config_path) != CONFIG_SHA256:
        raise BaselineError("only the exact frozen visible-baseline config is allowed")
    config = validate_config(load_json_strict(config_path))
    input_records = list(config["inputs"].values())
    code_records = config["local_code_dependencies"]
    input_paths_unresolved = [ROOT / record["path"] for record in input_records]
    code_paths_unresolved = [ROOT / record["path"] for record in code_records]
    # Config-derived path text is gated before any one of those files is touched.
    lexical_path_preflight((*input_paths_unresolved, *code_paths_unresolved))
    canonical_path_preflight(
        input_paths=(*input_paths_unresolved, *code_paths_unresolved),
        output_paths=(args.output_data, args.output_manifest),
    )
    input_paths = resolve_and_authenticate_records(input_records, label="input")
    code_paths = resolve_and_authenticate_records(code_records, label="code dependency")
    input_by_name = dict(zip(config["inputs"], input_paths, strict=True))

    output_data = args.output_data.resolve(strict=False)
    output_manifest = args.output_manifest.resolve(strict=False)
    if (
        output_data == output_manifest
        or output_data.parent != output_manifest.parent
        or output_data.exists()
        or output_data.is_symlink()
        or output_manifest.exists()
        or output_manifest.is_symlink()
    ):
        raise BaselineError("baseline outputs must be distinct new files in one directory")
    if output_data.suffix != ".npz" or output_manifest.suffix != ".json":
        raise BaselineError("baseline outputs must use .npz and .json suffixes")

    authenticated_paths = [config_path, SCRIPT_PATH, *input_paths, *code_paths]
    pre_hashes = {str(path): sha256_file(path) for path in authenticated_paths}
    protocol = build_extraction_protocol(
        load_json_strict(input_by_name["v3_protocol"]),
        load_json_strict(input_by_name["v3_action_protocol"]),
    )
    alignment = load_json_strict(input_by_name["label_free_alignment_index"])
    campaign = config["campaign"]
    alignment_rows = validate_alignment_index(
        alignment,
        expected_total_count=campaign["all_boundary_count"],
        expected_stable_count=campaign["stable_row_count"],
    )
    extraction = EXTRACT.extract_stable_rows_streaming(
        input_by_name["development_prompts"],
        input_by_name["development_public_report"],
        protocol=protocol,
    )
    extracted_rows, _eligibility = validate_extraction_coverage(
        extraction,
        alignment_rows=alignment_rows,
        expected_total_count=campaign["all_boundary_count"],
        expected_stable_count=campaign["stable_row_count"],
        expected_unstable_count=campaign["numerically_unstable_row_count"],
    )
    tensors = build_baseline_tensors(
        extracted_rows=extracted_rows,
        alignment_rows=alignment_rows,
        expected_total_count=campaign["all_boundary_count"],
        expected_stable_count=campaign["stable_row_count"],
    )
    for path_text, expected_hash in pre_hashes.items():
        if sha256_file(Path(path_text)) != expected_hash:
            raise BaselineError(f"authenticated input changed during extraction: {path_text}")

    _write_npz_no_clobber(output_data, tensors)
    output_sha256 = sha256_file(output_data)
    try:
        with np.load(output_data, allow_pickle=False) as loaded:
            if tuple(loaded.files) != OUTPUT_KEYS:
                raise BaselineError("baseline NPZ reload key order changed")
            for name, expected in tensors.items():
                observed = np.asarray(loaded[name])
                if observed.dtype.kind not in {"i", "f"} or not np.array_equal(
                    observed, expected, equal_nan=False
                ):
                    raise BaselineError(f"baseline NPZ reload changed {name}")
    except (OSError, ValueError, KeyError) as exc:
        raise BaselineError(f"cannot reload baseline NPZ: {exc}") from exc
    for path_text, expected_hash in pre_hashes.items():
        if sha256_file(Path(path_text)) != expected_hash:
            raise BaselineError(f"authenticated input changed during output: {path_text}")

    stable_source_ids = [
        row["source_id_sha256"]
        for row in alignment_rows
        if row["stable_feature_eligible"]
    ]
    unstable_source_ids = [
        row["source_id_sha256"]
        for row in alignment_rows
        if not row["stable_feature_eligible"]
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "passed",
        "status_scope": "authenticated_visible_precompletion_numeric_baselines_only",
        "pre_and_post_input_bindings_equal": True,
        "config": _artifact_record(config_path),
        "implementation": _artifact_record(SCRIPT_PATH),
        "inputs": {
            name: _artifact_record(path) for name, path in input_by_name.items()
        },
        "local_code_dependencies": [
            {"role": record["role"], **_artifact_record(path)}
            for record, path in zip(code_records, code_paths, strict=True)
        ],
        "coverage": {
            "all_boundary_count": campaign["all_boundary_count"],
            "stable_row_count": campaign["stable_row_count"],
            "numerically_unstable_row_count": campaign[
                "numerically_unstable_row_count"
            ],
            "stable_source_identity_order_exact": True,
            "unstable_source_identity_order_exact": True,
            "stable_source_identity_sequence_sha256": _string_sequence_sha256(
                "stable_source_id_sha256", stable_source_ids
            ),
            "unstable_source_identity_sequence_sha256": _string_sequence_sha256(
                "unstable_source_id_sha256", unstable_source_ids
            ),
        },
        "output": {
            "path": output_data.name,
            "sha256": output_sha256,
            "size_bytes": output_data.stat().st_size,
            "keys": list(OUTPUT_KEYS),
            "numeric_tensor_only": True,
            "reload_verified": True,
            "arrays": {
                name: {
                    "shape": list(values.shape),
                    "dtype": (
                        "little-endian-int64"
                        if name == "global_index"
                        else "little-endian-float64"
                    ),
                    "logical_sha256": _logical_array_sha256(name, values),
                }
                for name, values in tensors.items()
            },
        },
        "variants": config["variants"],
        "feature_boundary": config["feature_boundary"],
        "claim_scope": config["claim_scope"],
        "forbidden_path_guard_passed": True,
        "reserved_validation_access_authorized": False,
    }
    _write_json_no_clobber(output_manifest, manifest)
    print(
        f"wrote {campaign['stable_row_count']} authenticated visible-baseline rows "
        f"to {output_data}"
    )
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
