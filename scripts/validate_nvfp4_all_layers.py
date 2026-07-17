#!/usr/bin/env python3
"""Validate all-layer NVFP4/FP8-STE reverse replay on a real capture."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import torch

from fit_jlens_nvfp4_ste import SOURCE_LAYERS, valid_estimator_positions
from modelopt_checkpoint import (
    PINNED_METADATA_SHA256,
    PINNED_REVISION,
    ModelOptCheckpoint,
    default_pinned_snapshot,
)
from nvfp4_block_replay import (
    CapturedQwenBlockReplayFactory,
    DenseDequantFp8Backend,
    DenseDequantW4Backend,
    LiveFp8Backend,
    PackedNvFp4W4Backend,
    QWEN36_27B_REPLAY_SPEC,
    QwenReplaySpec,
)


SCHEMA_VERSION = 1
EXPECTED_TARGET_LAYERS = list(range(64))
EXPECTED_SHARED_TENSORS = 688
EXPECTED_OBSERVER_ONLY_TENSORS = 432
EXPECTED_REPLAY_PARAMETERS = 785
EXPECTED_OBSERVER_TENSORS = 1905
MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_IDENTITY_POLICY = "pinned-nvidia-modelopt-snapshot-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-report", type=Path)
    parser.add_argument("--observer-proof", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=default_pinned_snapshot())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-stop", type=int, default=1)
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--ste-policy", choices=("identity", "clipped"), default="identity")
    parser.add_argument("--checkpoint-interval", type=int, default=16)
    parser.add_argument(
        "--backend",
        choices=("dense", "packed-live"),
        default="dense",
    )
    parser.add_argument(
        "--compare-dense",
        action="store_true",
        help="also rerun dense validation and report elementwise row errors",
    )
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--max-comparison-relative-rms", type=float, default=0.03)
    parser.add_argument("--allow-unpinned-checkpoint", action="store_true")
    parser.add_argument(
        "--allow-unproven-capture",
        action="store_true",
        help="diagnostic escape hatch; output is explicitly non-authoritative",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_capture(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"failed to load tensor capture {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("capture payload root must be an object")
    targets = payload.get("target_layers")
    if targets != list(range(64)) or payload.get("target_profile") != "all":
        raise ValueError("validator requires an all-64-layer capture")
    return payload


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _require_artifact_binding(
    record: Any,
    path: Path,
    sha256: str,
    *,
    label: str,
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"observer proof lacks {label} artifact binding")
    if (
        Path(str(record.get("path", ""))).resolve() != path.resolve()
        or record.get("bytes") != path.stat().st_size
        or record.get("sha256") != sha256
    ):
        raise ValueError(f"observer proof {label} artifact binding mismatch")


def validate_capture_authority(
    capture_path: Path,
    payload: Mapping[str, Any],
    capture_report_path: Path,
    proof_path: Path,
    *,
    capture_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind an all-layer tensor payload to its isolated compiled proof."""

    capture_path = capture_path.resolve()
    capture_report_path = capture_report_path.resolve()
    proof_path = proof_path.resolve()
    capture_hash = capture_sha256 or sha256_file(capture_path)
    report_hash = sha256_file(capture_report_path)
    report = _load_json_object(capture_report_path, label="capture report")
    runtime = report.get("runtime")
    capture = report.get("capture")
    prompt = report.get("prompt")
    model = report.get("model")
    identity = model.get("identity") if isinstance(model, Mapping) else None
    if (
        report.get("schema_version") != 1
        or report.get("status") != "captured"
        or not isinstance(runtime, Mapping)
        or runtime.get("mode") != "compiled-observer"
        or runtime.get("target_profile") != "all"
        or runtime.get("target_layers") != EXPECTED_TARGET_LAYERS
        or runtime.get("mtp_enabled") is not False
        or runtime.get("language_model_only") is not True
        or not isinstance(model, Mapping)
        or model.get("repo_id") != MODEL_REPO
        or model.get("revision") != PINNED_REVISION
        or not isinstance(identity, Mapping)
        or identity.get("policy") != MODEL_IDENTITY_POLICY
        or identity.get("repo_id") != MODEL_REPO
        or identity.get("revision") != PINNED_REVISION
        or identity.get("metadata_sha256") != PINNED_METADATA_SHA256
        or identity.get("strict_pinned_validation") is not True
        or identity.get("validator")
        != "ModelOptCheckpoint(strict_pinned=True)"
        or identity.get("resolved_path") != model.get("resolved_path")
        or not isinstance(prompt, Mapping)
        or prompt.get("token_ids") != payload.get("prompt_token_ids")
        or not isinstance(capture, Mapping)
        or capture.get("missing_required") != []
        or capture.get("truncated") != []
        or len(capture.get("replay_parameter_provenance", {}))
        != EXPECTED_REPLAY_PARAMETERS
        or len(capture.get("tensor_summaries", {})) != EXPECTED_OBSERVER_TENSORS
        or len(capture.get("compile_visible_observer_tensors", []))
        != EXPECTED_OBSERVER_ONLY_TENSORS
        or Path(str(capture.get("tensor_output", ""))).resolve() != capture_path
    ):
        raise ValueError("compiled observer capture report scope is invalid")

    proof = _load_json_object(proof_path, label="observer proof")
    claim = proof.get("claim")
    configuration = proof.get("configuration")
    shared = proof.get("shared_internal_tensor_parity")
    replay = proof.get("replay_parameter_parity")
    completeness = proof.get("observer_capture_completeness")
    if (
        proof.get("schema_version") != 1
        or proof.get("status") != "passed"
        or not isinstance(claim, Mapping)
        or claim.get("mtp") != "off"
        or claim.get("observer_graph_modified") is not True
        or claim.get("observer_modification_discharged") is not True
        or proof.get("generation_record_parity", {}).get("exact") is not True
        or not isinstance(configuration, Mapping)
        or configuration.get("model_revision") != PINNED_REVISION
        or configuration.get("target_profile") != "all"
        or configuration.get("target_layers") != EXPECTED_TARGET_LAYERS
        or configuration.get("mtp_enabled") is not False
        or configuration.get("language_model_only") is not True
        or configuration.get("prompt", {}).get("token_ids")
        != payload.get("prompt_token_ids")
        or not isinstance(shared, Mapping)
        or shared.get("shared_tensor_count") != EXPECTED_SHARED_TENSORS
        or shared.get("observer_only_tensor_count")
        != EXPECTED_OBSERVER_ONLY_TENSORS
        or shared.get("all_shared_bit_exact") is not True
        or not isinstance(replay, Mapping)
        or replay.get("parameter_count") != EXPECTED_REPLAY_PARAMETERS
        or replay.get("all_names_equal") is not True
        or replay.get("all_shapes_equal") is not True
        or replay.get("all_dtypes_equal") is not True
        or replay.get("all_content_hashes_equal") is not True
        or replay.get("json_provenance_equal") is not True
        or not isinstance(completeness, Mapping)
        or completeness.get("required_missing") != []
        or completeness.get("truncated") != []
    ):
        raise ValueError("isolated compiled observer proof scope is invalid")

    artifacts = proof.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("observer proof lacks artifact bindings")
    _require_artifact_binding(
        artifacts.get("observer_json"),
        capture_report_path,
        report_hash,
        label="observer JSON",
    )
    _require_artifact_binding(
        artifacts.get("observer_tensors"),
        capture_path,
        capture_hash,
        label="observer tensors",
    )
    verifier_path = Path(__file__).resolve().with_name("prove_nvfp4_capture_pair.py")
    verifier = proof.get("verifier")
    if (
        not isinstance(verifier, Mapping)
        or Path(str(verifier.get("path", ""))).resolve() != verifier_path
        or verifier.get("bytes") != verifier_path.stat().st_size
        or verifier.get("sha256") != sha256_file(verifier_path)
    ):
        raise ValueError("observer proof verifier source binding mismatch")
    return {
        "status": "passed",
        "claim": "authoritative within this pinned prompt/runtime scope",
        "capture_report_path": str(capture_report_path),
        "capture_report_sha256": report_hash,
        "observer_proof_path": str(proof_path),
        "observer_proof_sha256": sha256_file(proof_path),
        "capture_sha256": capture_hash,
    }


def summarize_rows(rows: Mapping[int, torch.Tensor]) -> dict[str, Any]:
    if set(rows) != set(SOURCE_LAYERS):
        missing = sorted(set(SOURCE_LAYERS) - set(rows))
        extra = sorted(set(rows) - set(SOURCE_LAYERS))
        raise ValueError(f"Jacobian source coverage mismatch: missing={missing}, extra={extra}")
    records: dict[str, Any] = {}
    for layer in SOURCE_LAYERS:
        value = rows[layer].detach().float().contiguous().cpu()
        finite = bool(torch.isfinite(value).all())
        nonzero = bool((value != 0).any())
        raw = value.numpy().tobytes()
        records[str(layer)] = {
            "shape": list(value.shape),
            "finite": finite,
            "nonzero": nonzero,
            "rms": (
                float(value.square().mean().sqrt().item())
                if value.numel() and finite
                else math.inf
            ),
            "max_abs": (
                float(value.abs().max().item())
                if value.numel() and finite
                else math.inf
            ),
            "sha256_float32": hashlib.sha256(raw).hexdigest(),
        }
    return records


def load_reference_report(
    path: Path,
    payload: Mapping[str, Any],
    *,
    row_start: int,
    row_stop: int,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load reference report {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("reference report root must be an object")
    rows = value.get("rows")
    prompt = value.get("prompt")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "passed"
        or not isinstance(rows, Mapping)
        or rows.get("interval") != [row_start, row_stop]
        or rows.get("source_layer_count") != len(SOURCE_LAYERS)
        or not isinstance(rows.get("layers"), Mapping)
        or not isinstance(prompt, Mapping)
        or prompt.get("token_ids") != payload.get("prompt_token_ids")
    ):
        raise ValueError("reference report scope does not match this validation")
    if set(rows["layers"]) != {str(layer) for layer in SOURCE_LAYERS}:
        raise ValueError("reference report does not cover all 63 source layers")
    return value


def _make_factory(
    payload: Mapping[str, Any],
    checkpoint: ModelOptCheckpoint,
    spec: QwenReplaySpec,
    *,
    backend: str,
    ste_policy: str,
    checkpoint_interval: int,
) -> CapturedQwenBlockReplayFactory:
    if backend == "dense":
        w4_backend = DenseDequantW4Backend(checkpoint, dtype=torch.bfloat16)
        fp8_backend = DenseDequantFp8Backend(dtype=torch.bfloat16)
    elif backend == "packed-live":
        w4_backend = PackedNvFp4W4Backend(checkpoint)
        fp8_backend = LiveFp8Backend()
    else:
        raise ValueError(f"unknown replay backend: {backend}")
    return CapturedQwenBlockReplayFactory(
        payload,
        w4_backend,
        fp8_backend=fp8_backend,
        spec=spec,
        ste_policy=ste_policy,
        checkpoint_interval=checkpoint_interval,
    )


def _run_rows(
    payload: Mapping[str, Any],
    checkpoint: ModelOptCheckpoint,
    spec: QwenReplaySpec,
    positions: torch.Tensor,
    *,
    backend: str,
    device: torch.device,
    row_start: int,
    row_stop: int,
    ste_policy: str,
    checkpoint_interval: int,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    factory = _make_factory(
        payload,
        checkpoint,
        spec,
        backend=backend,
        ste_policy=ste_policy,
        checkpoint_interval=checkpoint_interval,
    )
    started = time.monotonic()
    chunk = factory.reverse_replay_rows(
        positions,
        row_start,
        row_stop,
        first_block=1,
        target_layer=63,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    rows = {
        layer: value.detach().float().contiguous().cpu()
        for layer, value in chunk.rows.items()
    }
    resources = {
        "device": str(device),
        "elapsed_seconds": elapsed,
        "max_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "max_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None
        ),
    }
    del chunk, factory
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, resources


def compare_rows(
    packed: Mapping[int, torch.Tensor],
    dense: Mapping[int, torch.Tensor],
    reference: Mapping[str, Any] | None,
    *,
    max_relative_rms: float,
) -> dict[str, Any]:
    if max_relative_rms < 0:
        raise ValueError("comparison tolerance must be nonnegative")
    if set(packed) != set(SOURCE_LAYERS) or set(dense) != set(SOURCE_LAYERS):
        raise ValueError("comparison rows do not cover all source layers")
    packed_summary = summarize_rows(packed)
    dense_summary = summarize_rows(dense)
    reference_layers = (
        reference["rows"]["layers"] if reference is not None else None
    )
    records: dict[str, Any] = {}
    dense_reference_matches = 0
    packed_reference_matches = 0
    max_observed_relative_rms = 0.0
    max_observed_abs = 0.0
    for layer in SOURCE_LAYERS:
        key = str(layer)
        actual = packed[layer].float()
        expected = dense[layer].float()
        difference = actual - expected
        finite = bool(torch.isfinite(difference).all())
        rms = (
            float(difference.square().mean().sqrt().item())
            if finite
            else math.inf
        )
        max_abs = float(difference.abs().max().item()) if finite else math.inf
        reference_rms = float(expected.square().mean().sqrt().item())
        relative_rms = rms / max(reference_rms, 1e-30)
        max_observed_relative_rms = max(max_observed_relative_rms, relative_rms)
        max_observed_abs = max(max_observed_abs, max_abs)
        reference_sha = (
            reference_layers[key]["sha256_float32"]
            if reference_layers is not None
            else None
        )
        dense_hash_match = (
            dense_summary[key]["sha256_float32"] == reference_sha
            if reference_sha is not None
            else None
        )
        packed_hash_match = (
            packed_summary[key]["sha256_float32"] == reference_sha
            if reference_sha is not None
            else None
        )
        dense_reference_matches += int(dense_hash_match is True)
        packed_reference_matches += int(packed_hash_match is True)
        records[key] = {
            "shape": list(actual.shape),
            "finite": finite,
            "exact": bool(torch.equal(actual, expected)),
            "max_abs": max_abs,
            "rms": rms,
            "relative_rms": relative_rms,
            "within_tolerance": finite and relative_rms <= max_relative_rms,
            "packed_sha256_float32": packed_summary[key]["sha256_float32"],
            "dense_sha256_float32": dense_summary[key]["sha256_float32"],
            "certificate_sha256_float32": reference_sha,
            "dense_certificate_hash_equal": dense_hash_match,
            "packed_certificate_hash_equal": packed_hash_match,
        }
    all_within = all(record["within_tolerance"] for record in records.values())
    certificate_pass = (
        reference_layers is None
        or dense_reference_matches == len(SOURCE_LAYERS)
    )
    return {
        "status": "passed" if all_within and certificate_pass else "failed",
        "contract": (
            "packed W4 + live FP8 rows compared elementwise with a fresh "
            "dense-dequant validation rerun"
        ),
        "max_relative_rms_tolerance": max_relative_rms,
        "max_observed_relative_rms": max_observed_relative_rms,
        "max_observed_abs": max_observed_abs,
        "all_layers_within_tolerance": all_within,
        "dense_certificate_hash_match_count": dense_reference_matches,
        "packed_certificate_hash_match_count": packed_reference_matches,
        "certificate_layer_count": (
            len(SOURCE_LAYERS) if reference_layers is not None else 0
        ),
        "layers": records,
    }


def validate_all_layers(
    payload: Mapping[str, Any],
    checkpoint: ModelOptCheckpoint,
    *,
    device: torch.device | str,
    row_start: int,
    row_stop: int,
    skip_first: int,
    ste_policy: str,
    checkpoint_interval: int,
    backend: str = "dense",
    compare_dense: bool = False,
    reference: Mapping[str, Any] | None = None,
    max_comparison_relative_rms: float = 0.03,
) -> dict[str, Any]:
    parsed_spec = QwenReplaySpec.from_model_config(checkpoint.config)
    if parsed_spec != QWEN36_27B_REPLAY_SPEC:
        raise ValueError("pinned checkpoint replay geometry changed")
    if not 0 <= row_start < row_stop <= parsed_spec.attention.hidden_size:
        raise ValueError("invalid row interval")
    token_ids = payload.get("prompt_token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("capture prompt token IDs are missing")

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested but unavailable")
    if compare_dense and backend != "packed-live":
        raise ValueError("--compare-dense requires --backend packed-live")
    positions = valid_estimator_positions(
        len(token_ids), skip_first, device=target_device
    )
    primary_rows, primary_resources = _run_rows(
        payload,
        checkpoint,
        parsed_spec,
        positions,
        backend=backend,
        device=target_device,
        row_start=row_start,
        row_stop=row_stop,
        ste_policy=ste_policy,
        checkpoint_interval=checkpoint_interval,
    )
    rows = summarize_rows(primary_rows)
    passed = all(record["finite"] and record["nonzero"] for record in rows.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "forward": "exact deployed compiled NVFP4/FP8 capture per block",
            "backward": (
                "dense-dequant W4/FP8 validation VJP"
                if backend == "dense"
                else "packed NVFP4 W4 + live runtime FP8 streaming VJP"
            ),
            "backend": backend,
            "literal_quantized_derivative": False,
            "ste_policy": ste_policy,
            "checkpoint_interval": checkpoint_interval,
            "estimator": "future-summed VJP; mean over matching source positions",
        },
        "prompt": {
            "text": payload.get("prompt"),
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "valid_positions": positions.detach().cpu().tolist(),
        },
        "rows": {
            "interval": [row_start, row_stop],
            "source_layer_count": len(rows),
            "layers": rows,
        },
        "resources": primary_resources,
    }
    if compare_dense:
        dense_rows, dense_resources = _run_rows(
            payload,
            checkpoint,
            parsed_spec,
            positions,
            backend="dense",
            device=target_device,
            row_start=row_start,
            row_stop=row_stop,
            ste_policy=ste_policy,
            checkpoint_interval=checkpoint_interval,
        )
        comparison = compare_rows(
            primary_rows,
            dense_rows,
            reference,
            max_relative_rms=max_comparison_relative_rms,
        )
        packed_allocated = primary_resources["max_cuda_allocated_bytes"]
        dense_allocated = dense_resources["max_cuda_allocated_bytes"]
        comparison["resources"] = {
            "packed_live": primary_resources,
            "dense_validation": dense_resources,
            "max_cuda_allocated_bytes_saved": (
                dense_allocated - packed_allocated
                if dense_allocated is not None and packed_allocated is not None
                else None
            ),
            "max_cuda_allocated_fraction_saved": (
                (dense_allocated - packed_allocated) / dense_allocated
                if dense_allocated and packed_allocated is not None
                else None
            ),
        }
        result["comparison"] = comparison
        if comparison["status"] != "passed":
            result["status"] = "failed"
    return result


def main() -> int:
    args = _parse_args()
    if not args.allow_unproven_capture and (
        args.capture_report is None or args.observer_proof is None
    ):
        raise ValueError(
            "authoritative validation requires --capture-report and --observer-proof"
        )
    if (args.capture_report is None) != (args.observer_proof is None):
        raise ValueError("--capture-report and --observer-proof must be supplied together")
    payload = load_capture(args.capture)
    capture_sha256 = sha256_file(args.capture)
    authority = (
        validate_capture_authority(
            args.capture,
            payload,
            args.capture_report,
            args.observer_proof,
            capture_sha256=capture_sha256,
        )
        if args.capture_report is not None and args.observer_proof is not None
        else {
            "status": "unproven",
            "claim": "diagnostic only; compiled observer proof was not supplied",
        }
    )
    reference = (
        load_reference_report(
            args.reference_report,
            payload,
            row_start=args.row_start,
            row_stop=args.row_stop,
        )
        if args.reference_report is not None
        else None
    )
    checkpoint = ModelOptCheckpoint(
        args.checkpoint,
        strict_pinned=not args.allow_unpinned_checkpoint,
    )
    result = validate_all_layers(
        payload,
        checkpoint,
        device=args.device,
        row_start=args.row_start,
        row_stop=args.row_stop,
        skip_first=args.skip_first,
        ste_policy=args.ste_policy,
        checkpoint_interval=args.checkpoint_interval,
        backend=args.backend,
        compare_dense=args.compare_dense,
        reference=reference,
        max_comparison_relative_rms=args.max_comparison_relative_rms,
    )
    if reference is not None:
        reference_source = reference.get("source")
        if (
            not isinstance(reference_source, Mapping)
            or reference_source.get("capture_sha256") != capture_sha256
        ):
            raise ValueError("reference report is not bound to the selected capture")
    result["source"] = {
        "capture_path": str(args.capture.resolve()),
        "capture_size": args.capture.stat().st_size,
        "capture_sha256": capture_sha256,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "capture_authority": authority,
    }
    if args.reference_report is not None:
        result["source"]["reference_report_path"] = str(
            args.reference_report.resolve()
        )
        result["source"]["reference_report_sha256"] = sha256_file(
            args.reference_report
        )
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "source_layers": result["rows"]["source_layer_count"],
                "row_interval": result["rows"]["interval"],
                "elapsed_seconds": result["resources"]["elapsed_seconds"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
