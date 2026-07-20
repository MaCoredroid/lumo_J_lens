#!/usr/bin/env python3
"""Capture label-independent raw and public-J states without changing V3 reports.

This V4-only wrapper imports the frozen NVFP4 runner, intercepts its exact
float32 Jacobian transport, and writes a separate authenticated safetensors
shard per causal prefix boundary.  The wrapped runner emits its standard JSON
schema and the wrapped readout return value is not extended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_jlens_nvfp4 as base  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_raw_capture.json"
WRAPPER_PATH = Path(__file__).resolve()
SHELL_WRAPPER_PATH = ROOT / "scripts" / "run_swe_task_state_v4_raw_capture.sh"
SCHEMA_VERSION = 1
KIND = "swe_task_state_v4_label_independent_public_j_state_capture"
FROZEN_FORBIDDEN_PATH_FRAGMENTS = ("reserved", "validation")


class CaptureError(ValueError):
    """Raised when the V4 capture contract is not met."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CaptureError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> Any:
    try:
        return json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read strict JSON {path}: {exc}") from exc


def load_json_bytes_strict(data: bytes, *, label: str) -> Any:
    try:
        return json.loads(data, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot parse strict JSON {label}: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hex_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CaptureError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise CaptureError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise CaptureError(f"{label} must be a regular file: {path}")
    return resolved


def _path_has_forbidden_fragment(path: Path, fragments: Iterable[str]) -> bool:
    lowered = [part.lower() for part in path.resolve(strict=False).parts]
    return any(fragment.lower() in part for fragment in fragments for part in lowered)


def frozen_lexical_path_preflight(paths: Iterable[Path | None]) -> None:
    """Reject forbidden path text without resolving, statting, or reading it."""

    for path in paths:
        if path is None:
            continue
        normalized = os.path.normpath(os.path.abspath(os.fspath(path))).lower()
        if any(
            fragment in component
            for fragment in FROZEN_FORBIDDEN_PATH_FRAGMENTS
            for component in Path(normalized).parts
        ):
            raise CaptureError(f"forbidden path rejected before filesystem access: {path}")


def frozen_canonical_path_preflight(
    *, input_paths: Iterable[Path | None], output_paths: Iterable[Path | None]
) -> None:
    """Resolve path metadata, then reject forbidden canonical parents before reads."""

    for path, strict in [
        *((path, True) for path in input_paths),
        *((path, False) for path in output_paths),
    ]:
        if path is None:
            continue
        try:
            resolved = Path(path).resolve(strict=strict)
        except OSError as exc:
            raise CaptureError(f"cannot resolve capture path metadata: {path}: {exc}") from exc
        lowered_parts = [part.lower() for part in resolved.parts]
        if any(
            fragment in component
            for fragment in FROZEN_FORBIDDEN_PATH_FRAGMENTS
            for component in lowered_parts
        ):
            raise CaptureError(
                f"forbidden canonical path rejected before file read or hash: {path}"
            )


def validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise CaptureError("capture config must be an object")
    expected_top = {
        "schema_version",
        "id",
        "status",
        "model",
        "lens",
        "capture",
        "authentication",
        "feature_independence",
        "downstream_feature_boundary",
        "claim_scope",
    }
    if set(config) != expected_top:
        raise CaptureError("capture config top-level keys changed")
    if config["schema_version"] != 1:
        raise CaptureError("unsupported capture config schema")
    if config["id"] != "swe-task-state-v4-label-independent-public-j-state-capture":
        raise CaptureError("capture config id changed")
    if config["status"] != "development_only_reserved_validation_closed":
        raise CaptureError("capture config status changed")

    model = config["model"]
    if model != {
        "repo_id": base.MODEL_REPO,
        "revision": base.MODEL_REVISION,
        "hidden_size": 5120,
        "layer_count": 64,
    }:
        raise CaptureError("model identity or geometry changed")
    lens = config["lens"]
    if lens != {
        "kind": "public",
        "sha256": base.LENS_SHA256,
        "transport": "float32(h_l) @ float32(J_l).T",
    }:
        raise CaptureError("public lens identity or transport changed")

    capture = config["capture"]
    expected_layers = list(range(24, 48))
    expected_capture = {
        "layers": expected_layers,
        "position": "causal_prefix_tail_only",
        "positions_argument": [-1],
        "stream_final_only_required": True,
        "raw_tensor": "post_block_residual_before_final_norm",
        "transported_tensor": (
            "public_j_state_before_bfloat16_final_norm_or_vocabulary_projection"
        ),
        "storage_dtype": "little_endian_float32",
        "shard_format": "safetensors",
    }
    if capture != expected_capture:
        raise CaptureError("capture tensor, layer, position, or storage contract changed")

    authentication = config["authentication"]
    if authentication != {
        "require_exact_source_bundle_sha256": True,
        "require_exact_reference_report_sha256": True,
        "require_reference_residual_manifest_equality_by_ordered_source_id": True,
        "require_all_64_layer_residual_manifest": True,
        "require_shard_reload_and_hash_verification": True,
        "forbidden_path_fragments": ["reserved", "validation"],
    }:
        raise CaptureError("capture authentication contract changed")

    independence = config["feature_independence"]
    expected_forbidden = [
        "prompt_text",
        "metadata",
        "target_token_id",
        "score_token_ids",
        "generated_token_id",
        "generated_text",
        "current_or_future_action_label",
        "current_or_future_concept_label",
        "task_outcome",
        "emotion_label",
        "private_reasoning",
    ]
    if independence != {
        "forward_inputs": ["prompt_token_ids"],
        "forbidden_feature_inputs": expected_forbidden,
        "source_ids_are_sha256_pseudonyms": True,
        "manifest_contains_prompt_tokens_or_text": False,
    }:
        raise CaptureError("feature-independence contract changed")

    downstream = config["downstream_feature_boundary"]
    if downstream != {
        "allowed_tensor_features": ["raw_residual", "public_j_state"],
        "allowed_causal_transforms": [
            "fixed_label_independent_projection",
            "current_state",
            "delta_from_immediately_previous_same_task_state",
            "deviation_from_prior_same_task_ema",
        ],
        "authentication_or_grouping_only": [
            "source_id_sha256",
            "token_ids_sha256",
            "token_count",
            "token_position",
            "paths",
            "hashes",
            "fidelity_fields",
            "base_report",
            "reference_report",
        ],
        "semantic_ids_as_features_forbidden": True,
        "base_report_fields_as_features_forbidden": True,
    }:
        raise CaptureError("downstream tensor-only feature boundary changed")

    claims = config["claim_scope"]
    if set(claims) != {
        "private_chain_of_thought_reconstructed",
        "cot_like_observable_event_decoding_established",
        "emotion_decoding_established",
        "confidence_or_calibration_established",
        "causal_interpretation_established",
        "incremental_value_over_raw_residual_or_word_probe_baselines_established",
    } or any(value is not False for value in claims.values()):
        raise CaptureError("capture-only claim scope must remain entirely false")
    return config


def _validate_token_ids(value: object, *, index: int) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in value
        )
    ):
        raise CaptureError(f"source prompt {index} requires nonempty integer token_ids")
    return list(value)


def sanitize_source_bundle(source: Any) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Retain only token IDs and pseudonymous IDs for the model forward."""

    if not isinstance(source, list) or not source:
        raise CaptureError("source bundle must be a nonempty JSON list")
    sanitized: list[dict[str, object]] = []
    identities: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(source):
        if not isinstance(row, dict):
            raise CaptureError(f"source prompt {index} must be an object")
        source_id = str(row.get("id", index))
        if source_id in seen_ids:
            raise CaptureError(f"duplicate source prompt id: {source_id!r}")
        seen_ids.add(source_id)
        token_ids = _validate_token_ids(row.get("token_ids"), index=index)
        source_id_sha256 = sha256_bytes(source_id.encode("utf-8"))
        token_ids_sha256 = sha256_bytes(canonical_json_bytes(token_ids))
        sanitized.append({"id": source_id_sha256, "token_ids": token_ids})
        identities.append(
            {
                "index": index,
                "source_id": source_id,
                "source_id_sha256": source_id_sha256,
                "token_ids_sha256": token_ids_sha256,
                "token_count": len(token_ids),
                "token_position": len(token_ids) - 1,
            }
        )
    return sanitized, identities


def _tensor_logical_sha256(tensor: Any, *, name: str, layers: list[int]) -> str:
    import torch

    if not torch.is_tensor(tensor):
        raise CaptureError(f"{name} is not a tensor")
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise CaptureError(f"{name} must be CPU float32")
    if tensor.ndim != 2 or list(tensor.shape) != [len(layers), 5120]:
        raise CaptureError(f"{name} geometry changed: {list(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise CaptureError(f"{name} contains non-finite values")
    contiguous = tensor.detach().contiguous()
    header = canonical_json_bytes(
        {
            "name": name,
            "layers": layers,
            "shape": list(contiguous.shape),
            "dtype": "little-endian-float32",
        }
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _manifest_is_all_layer_capture(
    value: object, *, expected_token_position: int | None = None
) -> bool:
    expected_algorithm = (
        "SHA-256 over length-prefixed canonical layer/shape/dtype/"
        "token-position/byte-count headers and logical row-major FP32 bytes"
    )
    if not isinstance(value, dict) or set(value) != {
        "algorithm",
        "sha256",
        "tensor_count",
        "logical_bytes",
        "token_positions",
    }:
        return False
    digest = value.get("sha256")
    token_positions = value.get("token_positions")
    return bool(
        value.get("algorithm") == expected_algorithm
        and value.get("tensor_count") == 64
        and value.get("logical_bytes") == 64 * 5120 * 4
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        and isinstance(token_positions, list)
        and len(token_positions) == 1
        and isinstance(token_positions[0], int)
        and not isinstance(token_positions[0], bool)
        and (
            expected_token_position is None
            or token_positions == [expected_token_position]
        )
    )


_ORIGINAL_READOUT = base._readout_captures
_ORIGINAL_TRANSPORT = base.transport_residual
_CAPTURE_CONTEXT: dict[str, Any] = {
    "armed": False,
    "layers": [],
    "transported": [],
    "records": [],
    "shards_dir": None,
}


def _capturing_transport(residual: Any, jacobian: Any) -> Any:
    transported = _ORIGINAL_TRANSPORT(residual, jacobian)
    if _CAPTURE_CONTEXT["armed"]:
        import torch

        copied = transported.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if copied.ndim != 2 or list(copied.shape) != [1, 5120]:
            raise CaptureError(
                f"transported state geometry changed: {list(copied.shape)}"
            )
        if not bool(torch.isfinite(copied).all()):
            raise CaptureError("transported state contains non-finite values")
        _CAPTURE_CONTEXT["transported"].append(copied[0])
    return transported


def _write_verified_shard(
    *,
    index: int,
    raw_residual: Any,
    public_j_state: Any,
    layers: list[int],
) -> dict[str, object]:
    import torch
    from safetensors.torch import load_file, save_file

    shards_dir = _CAPTURE_CONTEXT["shards_dir"]
    if not isinstance(shards_dir, Path):
        raise CaptureError("shard output directory was not armed")
    shard_path = shards_dir / f"boundary-{index:06d}.safetensors"
    if shard_path.exists() or shard_path.is_symlink():
        raise CaptureError(f"refusing to overwrite shard: {shard_path}")
    tensors = {
        "raw_residual": raw_residual.detach().contiguous(),
        "public_j_state": public_j_state.detach().contiguous(),
    }
    temporary_path = shard_path.with_name(f".{shard_path.name}.tmp-{os.getpid()}")
    if temporary_path.exists() or temporary_path.is_symlink():
        raise CaptureError(f"temporary shard already exists: {temporary_path}")
    save_file(
        tensors,
        temporary_path,
        metadata={
            "schema_version": str(SCHEMA_VERSION),
            "kind": KIND,
            "layers": ",".join(str(layer) for layer in layers),
        },
    )
    temporary_sha256 = sha256_file(temporary_path)
    loaded = load_file(temporary_path, device="cpu")
    if set(loaded) != set(tensors):
        raise CaptureError("safetensors reload keys changed")
    for key, expected in tensors.items():
        observed = loaded[key]
        if observed.dtype != torch.float32 or not torch.equal(observed, expected):
            raise CaptureError(f"safetensors reload changed {key}")
    if sha256_file(temporary_path) != temporary_sha256:
        raise CaptureError("safetensors shard changed during reload verification")
    os.replace(temporary_path, shard_path)
    shard_sha256 = sha256_file(shard_path)
    if shard_sha256 != temporary_sha256:
        raise CaptureError("atomic shard placement changed its bytes")
    return {
        "path": str(shard_path.relative_to(shards_dir.parent)),
        "sha256": shard_sha256,
        "size_bytes": shard_path.stat().st_size,
        "tensor_keys": ["public_j_state", "raw_residual"],
        "shape": [len(layers), 5120],
        "dtype": "little-endian-float32",
        "raw_residual_logical_sha256": _tensor_logical_sha256(
            raw_residual, name="raw_residual", layers=layers
        ),
        "public_j_state_logical_sha256": _tensor_logical_sha256(
            public_j_state, name="public_j_state", layers=layers
        ),
        "reload_verified": True,
    }


def _capturing_readout(
    model: Any,
    *,
    lens_path: str,
    layers: tuple[int, ...],
    top_k: int,
    target_token_ids: tuple[int, ...],
    score_token_ids: tuple[int, ...] = (),
) -> dict[str, object]:
    import torch

    configured_layers = list(_CAPTURE_CONTEXT["layers"])
    if list(layers) != configured_layers:
        raise CaptureError("runner layers differ from the frozen V4 capture layers")
    if tuple(model._jlens_positions) == () or len(model._jlens_positions) != 1:
        raise CaptureError("V4 raw capture requires exactly one causal tail position")
    if _CAPTURE_CONTEXT["armed"]:
        raise CaptureError("nested V4 capture is not supported")
    _CAPTURE_CONTEXT["transported"] = []
    _CAPTURE_CONTEXT["armed"] = True
    try:
        result = _ORIGINAL_READOUT(
            model,
            lens_path=lens_path,
            layers=layers,
            top_k=top_k,
            target_token_ids=target_token_ids,
            score_token_ids=score_token_ids,
        )
    finally:
        _CAPTURE_CONTEXT["armed"] = False

    transported_rows = _CAPTURE_CONTEXT["transported"]
    if len(transported_rows) != len(configured_layers):
        raise CaptureError(
            "captured Jacobian-state count changed: "
            f"expected {len(configured_layers)}, got {len(transported_rows)}"
        )
    raw_rows = []
    for layer in configured_layers:
        raw = model._jlens_captures.get(layer)
        if not torch.is_tensor(raw) or list(raw.shape) != [1, 5120]:
            raise CaptureError(f"raw residual geometry changed at layer {layer}")
        raw_rows.append(raw[0].detach().to(device="cpu", dtype=torch.float32))
    raw_tensor = torch.stack(raw_rows).contiguous()
    transported_tensor = torch.stack(transported_rows).contiguous()
    manifest = result.get("residual_capture_manifest")
    if not _manifest_is_all_layer_capture(manifest):
        raise CaptureError("base runner did not return an all-64-layer residual manifest")
    index = len(_CAPTURE_CONTEXT["records"])
    shard = _write_verified_shard(
        index=index,
        raw_residual=raw_tensor,
        public_j_state=transported_tensor,
        layers=configured_layers,
    )
    final_readout = result["final_model_readout"][0]
    captured_final = result["captured_final_model_readout"][0]
    final_top1_match = (
        final_readout["token_ids"][0] == final_readout["target_token_id"]
        and captured_final["token_ids"][0] == captured_final["target_token_id"]
    )
    _CAPTURE_CONTEXT["records"].append(
        {
            "index": index,
            "residual_capture_manifest": manifest,
            "shard": shard,
            "final_model_top1_matches_greedy": bool(final_top1_match),
            "final_norm_reconstruction": dict(result["final_norm_reconstruction"]),
            "final_logits_reconstruction": dict(result["final_logits_reconstruction"]),
        }
    )
    return result


def _reference_records(path: Path) -> list[dict[str, object]]:
    import ijson

    records: list[dict[str, object]] = []
    with path.open("rb") as handle:
        for index, experiment in enumerate(ijson.items(handle, "experiments.item")):
            manifest = experiment.get("residual_capture_manifest")
            if not _manifest_is_all_layer_capture(manifest):
                raise CaptureError(
                    f"reference experiment {index} lacks a valid residual manifest"
                )
            records.append(
                {
                    "index": index,
                    "source_id": str(experiment.get("id")),
                    "residual_capture_manifest": manifest,
                }
            )
    if not records:
        raise CaptureError("reference report has no experiments")
    return records


def _capture_selected_object(parser: Any, first_event: tuple[str, str, Any]) -> Any:
    """Build one selected ijson map/array from an already consumed start event."""

    import ijson

    _prefix, event, value = first_event
    if event not in {"start_map", "start_array"}:
        raise CaptureError("selected JSON object did not start with a container")
    builder = ijson.ObjectBuilder()
    builder.event(event, value)
    depth = 1
    for _inner_prefix, inner_event, inner_value in parser:
        builder.event(inner_event, inner_value)
        if inner_event in {"start_map", "start_array"}:
            depth += 1
        elif inner_event in {"end_map", "end_array"}:
            depth -= 1
            if depth == 0:
                return builder.value
    raise CaptureError("base report ended inside a selected JSON object")


def audit_base_report(path: Path) -> dict[str, Any]:
    """Stream a large base report while retaining only provenance metadata."""

    import ijson

    selected: dict[str, Any] = {}
    experiment_count = 0
    with path.open("rb") as handle:
        parser = iter(ijson.parse(handle, use_float=True))
        for prefix, event, value in parser:
            if prefix in {"schema_version", "status"} and event in {
                "string",
                "number",
            }:
                selected[prefix] = value
            elif prefix in {"model", "lens", "runtime", "assertions"} and event == "start_map":
                selected[prefix] = _capture_selected_object(
                    parser, (prefix, event, value)
                )
            elif prefix == "experiments.item" and event == "start_map":
                experiment_count += 1
    if set(selected) != {
        "schema_version",
        "status",
        "model",
        "lens",
        "runtime",
        "assertions",
    }:
        raise CaptureError("base report provenance fields are incomplete")
    selected["experiment_count"] = experiment_count
    return selected


def _file_binding(path: Path) -> dict[str, object]:
    resolved = _require_regular_file(path, "bound input")
    try:
        display_path = str(resolved.relative_to(ROOT))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _require_bindings_unchanged(
    bindings: Mapping[str, Mapping[str, object]], paths: Mapping[str, Path]
) -> None:
    for label, expected in bindings.items():
        observed = _file_binding(paths[label])
        if observed != dict(expected):
            raise CaptureError(f"{label} bytes or path changed during capture")


def normalized_cli_contract(
    args: argparse.Namespace,
    *,
    source_path: Path,
    reference_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "lens_kind": args.lens_kind,
        "layers": list(base.validate_layers(base.parse_integer_list(args.layers, allow_all=True))),
        "positions": base.parse_integer_list(args.positions),
        "top_k": args.top_k,
        "score_token_ids": [],
        "runtime": {
            **base._runtime_pins(args),
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "source_bundle": str(source_path.relative_to(ROOT)),
        "reference_report": str(reference_path.relative_to(ROOT)),
        "state_output_dir": str(output_dir.relative_to(ROOT)),
        "base_output": str(args.output.resolve(strict=False).relative_to(ROOT)),
    }


def _canonical_manifest_equality(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _merge_and_validate_records(
    identities: list[dict[str, object]],
    captures: list[dict[str, object]],
    references: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not (len(identities) == len(captures) == len(references)):
        raise CaptureError(
            "source/capture/reference counts differ: "
            f"{len(identities)}/{len(captures)}/{len(references)}"
        )
    merged: list[dict[str, object]] = []
    for index, (identity, capture, reference) in enumerate(
        zip(identities, captures, references, strict=True)
    ):
        if identity["index"] != index or capture["index"] != index or reference["index"] != index:
            raise CaptureError("ordered boundary indices changed")
        if identity["source_id"] != reference["source_id"]:
            raise CaptureError(f"reference source id changed at boundary {index}")
        if not _manifest_is_all_layer_capture(
            capture["residual_capture_manifest"],
            expected_token_position=int(identity["token_position"]),
        ) or not _manifest_is_all_layer_capture(
            reference["residual_capture_manifest"],
            expected_token_position=int(identity["token_position"]),
        ):
            raise CaptureError(
                f"residual manifest token position or exact schema changed at boundary {index}"
            )
        residual_match = _canonical_manifest_equality(
            capture["residual_capture_manifest"],
            reference["residual_capture_manifest"],
        )
        if not residual_match:
            raise CaptureError(f"fresh residual manifest differs at boundary {index}")
        vocab_adapter_strict = bool(
            capture["final_model_top1_matches_greedy"]
            and capture["final_norm_reconstruction"].get("within_tolerance")
            and capture["final_logits_reconstruction"].get("within_tolerance")
        )
        merged.append(
            {
                "index": index,
                "source_id_sha256": identity["source_id_sha256"],
                "token_ids_sha256": identity["token_ids_sha256"],
                "token_count": identity["token_count"],
                "token_position": identity["token_position"],
                "residual_capture_manifest": capture["residual_capture_manifest"],
                "reference_residual_manifest_equal": True,
                "shard": capture["shard"],
                "capture_valid": True,
                "vocabulary_adapter_strict": vocab_adapter_strict,
                "final_model_top1_matches_greedy": capture[
                    "final_model_top1_matches_greedy"
                ],
                "final_norm_reconstruction_within_tolerance": bool(
                    capture["final_norm_reconstruction"].get("within_tolerance")
                ),
                "final_logits_reconstruction_within_tolerance": bool(
                    capture["final_logits_reconstruction"].get("within_tolerance")
                ),
            }
        )
    return merged


def _write_json_no_clobber(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise CaptureError(f"refusing to overwrite output: {path}")
    rendered = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise CaptureError(f"temporary output already exists: {temporary}")
    temporary.write_text(rendered)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--capture-config",
        type=Path,
        default=CONFIG_PATH,
        help="exact V4 raw-capture contract",
    )
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--reference-report-sha256", required=True)
    parser.add_argument("--state-output-dir", type=Path, required=True)
    return parser


def _validate_cli_and_prepare(args: argparse.Namespace) -> tuple[
    dict[str, Any], Path, Path, Path, list[dict[str, object]], list[dict[str, object]]
]:
    config_path = _require_regular_file(args.capture_config, "capture config")
    config_bytes = config_path.read_bytes()
    config = validate_config(
        load_json_bytes_strict(config_bytes, label="capture config")
    )
    forbidden = config["authentication"]["forbidden_path_fragments"]
    if args.prompt or not args.prompts_file:
        raise CaptureError("V4 raw capture requires --prompts-file and forbids --prompt")
    if args.score_token_ids:
        raise CaptureError("V4 raw capture forbids --score-token-ids")
    if args.lens_kind != "public" or args.lens_path is not None:
        raise CaptureError("V4 raw capture is frozen to the pinned public lens")
    expected_layers = config["capture"]["layers"]
    observed_layers = base.validate_layers(base.parse_integer_list(args.layers, allow_all=True))
    if list(observed_layers) != expected_layers:
        raise CaptureError("--layers must exactly equal the frozen capture layers 24..47")
    if base.parse_integer_list(args.positions) != [-1]:
        raise CaptureError("--positions must be exactly -1")
    if not args.stream_final_only:
        raise CaptureError("--stream-final-only is required")
    if args.top_k < 5:
        raise CaptureError("--top-k must be at least 5 for the fidelity canary")

    source_path = _require_regular_file(args.prompts_file, "source bundle")
    reference_path = _require_regular_file(args.reference_report, "reference report")
    source_expected = _require_hex_sha256(
        args.source_bundle_sha256, "--source-bundle-sha256"
    )
    reference_expected = _require_hex_sha256(
        args.reference_report_sha256, "--reference-report-sha256"
    )
    for path in (config_path, source_path, reference_path, args.state_output_dir, args.output):
        if path is None or _path_has_forbidden_fragment(Path(path), forbidden):
            raise CaptureError(f"reserved-validation path is forbidden: {path}")
    source_bytes = source_path.read_bytes()
    if sha256_bytes(source_bytes) != source_expected:
        raise CaptureError("source bundle SHA-256 mismatch")
    if sha256_file(reference_path) != reference_expected:
        raise CaptureError("reference report SHA-256 mismatch")

    output_dir = args.state_output_dir.resolve(strict=False)
    if output_dir.exists() or output_dir.is_symlink():
        raise CaptureError(f"state output directory must not exist: {output_dir}")
    if args.output is None:
        raise CaptureError("V4 raw capture requires --output for the base report")
    base_output = args.output.resolve(strict=False)
    try:
        base_output.relative_to(output_dir)
    except ValueError as exc:
        raise CaptureError("base --output must be inside --state-output-dir") from exc
    if base_output.exists() or base_output.is_symlink():
        raise CaptureError(f"base output already exists: {base_output}")

    source = load_json_bytes_strict(source_bytes, label="source bundle")
    sanitized, identities = sanitize_source_bundle(source)
    del source, source_bytes, config_bytes
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "shards").mkdir()
    return config, config_path, source_path, reference_path, sanitized, identities


def run(args: argparse.Namespace) -> int:
    frozen_lexical_path_preflight(
        (
            args.capture_config,
            args.prompts_file,
            args.reference_report,
            args.state_output_dir,
            args.output,
        )
    )
    frozen_canonical_path_preflight(
        input_paths=(args.capture_config, args.prompts_file, args.reference_report),
        output_paths=(args.state_output_dir, args.output),
    )
    if args.prompts_file is None:
        raise CaptureError("V4 raw capture requires --prompts-file")
    preliminary_paths = {
        "capture_config": _require_regular_file(args.capture_config, "capture config"),
        "wrapper": _require_regular_file(WRAPPER_PATH, "wrapper"),
        "shell_wrapper": _require_regular_file(SHELL_WRAPPER_PATH, "shell wrapper"),
        "frozen_base_runner": _require_regular_file(
            Path(base.__file__).resolve(), "frozen base runner"
        ),
        "source_bundle": _require_regular_file(args.prompts_file, "source bundle"),
        "reference_report": _require_regular_file(
            args.reference_report, "reference report"
        ),
    }
    pre_run_bindings = {
        label: _file_binding(path) for label, path in preliminary_paths.items()
    }
    (
        config,
        config_path,
        source_path,
        reference_path,
        sanitized,
        identities,
    ) = _validate_cli_and_prepare(args)
    output_dir = args.state_output_dir.resolve()
    binding_paths = {
        "capture_config": config_path,
        "wrapper": WRAPPER_PATH,
        "shell_wrapper": SHELL_WRAPPER_PATH,
        "frozen_base_runner": Path(base.__file__).resolve(),
        "source_bundle": source_path,
        "reference_report": reference_path,
    }
    _require_bindings_unchanged(pre_run_bindings, binding_paths)
    cli_contract = normalized_cli_contract(
        args,
        source_path=source_path,
        reference_path=reference_path,
        output_dir=output_dir,
    )
    _CAPTURE_CONTEXT.update(
        {
            "armed": False,
            "layers": list(config["capture"]["layers"]),
            "transported": [],
            "records": [],
            "shards_dir": output_dir / "shards",
        }
    )
    base.transport_residual = _capturing_transport
    base._readout_captures = _capturing_readout

    with tempfile.TemporaryDirectory(prefix="swe-v4-label-free-") as temporary_dir:
        sanitized_path = Path(temporary_dir) / "prompts.json"
        sanitized_path.write_text(
            json.dumps(sanitized, separators=(",", ":"), ensure_ascii=True) + "\n"
        )
        args.prompts_file = sanitized_path
        del sanitized
        if args.prompt and args.prompts_file:
            raise CaptureError("pass at most one of --prompt and --prompts-file")
        if not 1 <= args.top_k <= 100:
            raise CaptureError("--top-k must be in 1..100")
        if not 0.70 <= args.gpu_memory_utilization <= 0.90:
            raise CaptureError("--gpu-memory-utilization must be in 0.70..0.90")
        base._runtime_pins(args)
        lens_mode = base.lens_artifact_mode(args)
        with ExitStack() as resources:
            base_exit_code = base._run(args, lens_mode=lens_mode, resources=resources)

    references = _reference_records(reference_path)
    _require_bindings_unchanged(pre_run_bindings, binding_paths)
    records = _merge_and_validate_records(
        identities, list(_CAPTURE_CONTEXT["records"]), references
    )
    base_output_path = _require_regular_file(args.output, "base report")
    base_report_audit = audit_base_report(base_output_path)
    if base_report_audit["schema_version"] != base.SCHEMA_VERSION:
        raise CaptureError("base report schema changed")
    if base_report_audit["status"] not in {"passed", "failed"}:
        raise CaptureError("base report status is invalid")
    if base_report_audit["experiment_count"] != len(records):
        raise CaptureError("base report experiment count changed")
    expected_runtime = {
        **base._runtime_pins(args),
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if any(
        base_report_audit["runtime"].get(key) != value
        for key, value in expected_runtime.items()
    ):
        raise CaptureError("base report runtime pins changed")
    if (
        base_report_audit["model"].get("repo_id") != base.MODEL_REPO
        or base_report_audit["model"].get("revision") != base.MODEL_REVISION
        or base_report_audit["lens"].get("sha256") != base.LENS_SHA256
    ):
        raise CaptureError("base report model or lens identity changed")
    base_report_binding = _file_binding(base_output_path)
    all_capture_valid = all(record["capture_valid"] for record in records)
    all_vocab_strict = all(record["vocabulary_adapter_strict"] for record in records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "passed" if all_capture_valid else "failed",
        "status_scope": "raw_and_public_j_pre_vocabulary_state_capture_only",
        "capture_config": pre_run_bindings["capture_config"],
        "source_bundle": {
            **pre_run_bindings["source_bundle"],
            "prompt_count": len(identities),
        },
        "reference_report": pre_run_bindings["reference_report"],
        "base_report": {
            **base_report_binding,
            "schema_version": base_report_audit["schema_version"],
            "status": base_report_audit["status"],
            "experiment_count": base_report_audit["experiment_count"],
            "runtime": base_report_audit["runtime"],
            "model_record_sha256": sha256_bytes(
                canonical_json_bytes(base_report_audit["model"])
            ),
            "lens_record_sha256": sha256_bytes(
                canonical_json_bytes(base_report_audit["lens"])
            ),
            "assertions": base_report_audit["assertions"],
            "feature_use_forbidden": True,
        },
        "implementation": {
            "wrapper": pre_run_bindings["wrapper"],
            "shell_wrapper": pre_run_bindings["shell_wrapper"],
            "frozen_base_runner": pre_run_bindings["frozen_base_runner"],
            "pre_and_post_run_bindings_equal": True,
        },
        "normalized_cli_contract": cli_contract,
        "model": config["model"],
        "lens": config["lens"],
        "capture": config["capture"],
        "feature_independence": config["feature_independence"],
        "downstream_feature_boundary": config["downstream_feature_boundary"],
        "claim_scope": config["claim_scope"],
        "summary": {
            "boundary_count": len(records),
            "all_capture_valid": all_capture_valid,
            "vocabulary_adapter_strict_count": sum(
                record["vocabulary_adapter_strict"] for record in records
            ),
            "all_vocabulary_adapters_strict": all_vocab_strict,
            "base_runner_exit_code": base_exit_code,
            "forbidden_path_guard_passed": True,
            "reserved_validation_access_authorized": False,
        },
        "boundaries": records,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_no_clobber(manifest_path, manifest)
    print(
        f"wrote {len(records)} authenticated raw/public-J state shards to "
        f"{output_dir}; strict vocab adapters "
        f"{manifest['summary']['vocabulary_adapter_strict_count']}/{len(records)}",
        file=sys.stderr,
    )
    return 0 if all_capture_valid else 1


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
