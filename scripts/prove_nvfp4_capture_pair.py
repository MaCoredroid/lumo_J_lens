#!/usr/bin/env python3
"""Prove parity between isolated compiled NVFP4 baseline and observer captures.

Large tensor payloads are processed by sequential mmap-backed subprocesses.
The verifier therefore never retains both multi-gigabyte payloads in one
process. SHA-256 comparisons use logical row-major tensor bytes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE_ROOT = ROOT / ".cache" / "runtime_capture"
DEFAULT_BASELINE_JSON = DEFAULT_CAPTURE_ROOT / "all_layers_compiled_baseline.json"
DEFAULT_BASELINE_TENSORS = DEFAULT_CAPTURE_ROOT / "all_layers_compiled_baseline.pt"
DEFAULT_OBSERVER_JSON = DEFAULT_CAPTURE_ROOT / "all_layers_compiled_observer.json"
DEFAULT_OBSERVER_TENSORS = DEFAULT_CAPTURE_ROOT / "all_layers_compiled_observer.pt"
DEFAULT_OUTPUT = DEFAULT_CAPTURE_ROOT / "all_layers_capture_proof.json"
EXPECTED_TARGET_LAYERS = list(range(64))
EXPECTED_SHARED_NON_REPLAY = 688
EXPECTED_OBSERVER_ONLY_NON_REPLAY = 432
EXPECTED_REPLAY_PARAMETERS = 785
SCHEMA_VERSION = 1


class VerificationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument(
        "--baseline-tensors", type=Path, default=DEFAULT_BASELINE_TENSORS
    )
    parser.add_argument("--observer-json", type=Path, default=DEFAULT_OBSERVER_JSON)
    parser.add_argument(
        "--observer-tensors", type=Path, default=DEFAULT_OBSERVER_TENSORS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-shared-non-replay",
        type=int,
        default=EXPECTED_SHARED_NON_REPLAY,
    )
    parser.add_argument(
        "--expected-observer-only-non-replay",
        type=int,
        default=EXPECTED_OBSERVER_ONLY_NON_REPLAY,
    )
    parser.add_argument(
        "--expected-replay-parameters",
        type=int,
        default=EXPECTED_REPLAY_PARAMETERS,
    )
    parser.add_argument("--hash-chunk-mib", type=int, default=32)
    parser.add_argument("--manifest-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--manifest-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_record(path: Path, *, chunk_bytes: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
            size += len(chunk)
    stat = path.stat()
    _require(size == stat.st_size, f"file changed while hashing: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "mtime_ns": stat.st_mtime_ns,
    }


def _tensor_sha256(tensor: Any, *, chunk_bytes: int) -> str:
    import torch

    source = tensor.detach().contiguous()
    if source.ndim == 0:
        source = source.reshape(1)
    logical = source.view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, logical.numel(), chunk_bytes):
        chunk = logical[start : start + chunk_bytes].numpy()
        digest.update(memoryview(chunk))
    return digest.hexdigest()


def _tensor_descriptor(tensor: Any, *, chunk_bytes: int) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "logical_bytes": tensor.numel() * tensor.element_size(),
        "logical_row_major_sha256": _tensor_sha256(
            tensor, chunk_bytes=chunk_bytes
        ),
    }


def _build_tensor_manifest(path: Path, *, chunk_bytes: int) -> dict[str, Any]:
    import torch

    artifact = _file_record(path, chunk_bytes=chunk_bytes)
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    _require(isinstance(payload, dict), f"tensor payload is not a mapping: {path}")
    tensors = payload.get("tensors")
    _require(isinstance(tensors, dict), f"tensor payload has no tensor map: {path}")

    descriptors: dict[str, Any] = {}
    for name in sorted(tensors):
        tensor = tensors[name]
        _require(isinstance(name, str), f"non-string tensor name in {path}")
        _require(torch.is_tensor(tensor), f"{name} is not a tensor in {path}")
        descriptors[name] = _tensor_descriptor(tensor, chunk_bytes=chunk_bytes)

    payload_metadata = {
        key: payload.get(key)
        for key in (
            "schema_version",
            "mode",
            "model_revision",
            "prompt",
            "prompt_token_ids",
            "target_profile",
            "target_layers",
        )
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact": artifact,
        "payload_metadata": payload_metadata,
        "tensor_count": len(descriptors),
        "logical_tensor_bytes": sum(
            descriptor["logical_bytes"] for descriptor in descriptors.values()
        ),
        "tensors": descriptors,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    del tensors
    del payload
    gc.collect()
    return manifest


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _run_manifest_worker(path: Path, output: Path, *, chunk_bytes: int) -> None:
    manifest = _build_tensor_manifest(path, chunk_bytes=chunk_bytes)
    _write_json_atomic(output, manifest)


def _build_manifest_in_subprocess(
    path: Path, output: Path, *, chunk_mib: int
) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--manifest-worker",
            str(path),
            "--manifest-output",
            str(output),
            "--hash-chunk-mib",
            str(chunk_mib),
        ],
        check=True,
    )
    return json.loads(output.read_text())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def _runtime_mode(metadata: dict[str, Any]) -> str | None:
    return metadata.get("runtime", {}).get("mode") or metadata.get("mode")


def _validate_json_contract(
    baseline: dict[str, Any], observer: dict[str, Any]
) -> dict[str, Any]:
    _require(_runtime_mode(baseline) == "compiled", "baseline mode is not compiled")
    _require(
        _runtime_mode(observer) == "compiled-observer",
        "observer mode is not compiled-observer",
    )
    for label, metadata in (("baseline", baseline), ("observer", observer)):
        runtime = metadata.get("runtime", {})
        _require(
            runtime.get("target_profile") == "all",
            f"{label} target profile is not all",
        )
        _require(
            runtime.get("target_layers") == EXPECTED_TARGET_LAYERS,
            f"{label} does not cover all 64 main-model layers",
        )
        _require(runtime.get("mtp_enabled") is False, f"{label} has MTP enabled")
        _require(
            runtime.get("language_model_only") is True,
            f"{label} is not main-model-only",
        )

    _require(
        baseline.get("authority", {}).get("compiled_endpoint") == "uninstrumented",
        "baseline endpoint is not labelled uninstrumented",
    )
    _require(
        observer.get("runtime", {}).get("compiled_observer") is True,
        "observer metadata does not identify the compiled observer",
    )
    _require(observer.get("status") == "captured", "observer status is not captured")
    _require(
        observer.get("authority", {}).get(
            "detailed_tensor_capture_is_unmodified_serving_graph"
        )
        is False,
        "observer graph modification is not explicitly disclosed",
    )

    baseline_generation = baseline.get("authoritative_compiled_generation")
    baseline_instrumented = baseline.get("instrumented_generation")
    observer_generation = observer.get("instrumented_generation")
    _require(
        isinstance(baseline_generation, dict),
        "baseline authoritative generation record is missing",
    )
    _require(
        baseline_generation == baseline_instrumented,
        "baseline authoritative and instrumented generation records differ",
    )
    _require(
        baseline_generation == observer_generation,
        "baseline and observer full generation records differ",
    )

    observer_capture = observer.get("capture", {})
    required_missing = observer_capture.get("missing_required")
    truncated = observer_capture.get("truncated")
    _require(required_missing == [], "observer has required missing tensors")
    _require(truncated == [], "observer has truncated tensors")

    _require(
        baseline.get("model", {}).get("revision")
        == observer.get("model", {}).get("revision"),
        "model revisions differ",
    )
    _require(
        baseline.get("prompt") == observer.get("prompt"),
        "prompt records differ",
    )
    return {
        "record": baseline_generation,
        "record_sha256": _canonical_json_sha256(baseline_generation),
        "records_compared": [
            "baseline.authoritative_compiled_generation",
            "baseline.instrumented_generation",
            "observer.instrumented_generation",
        ],
    }


def _validate_payload_metadata(
    label: str,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    payload = manifest["payload_metadata"]
    _require(payload["schema_version"] == metadata["schema_version"], f"{label} schema mismatch")
    _require(payload["mode"] == _runtime_mode(metadata), f"{label} mode mismatch")
    _require(
        payload["model_revision"] == metadata["model"]["revision"],
        f"{label} revision mismatch",
    )
    _require(payload["prompt"] == metadata["prompt"]["text"], f"{label} prompt mismatch")
    _require(
        payload["prompt_token_ids"] == metadata["prompt"]["token_ids"],
        f"{label} prompt-token mismatch",
    )
    _require(payload["target_profile"] == "all", f"{label} PT target is not all")
    _require(
        payload["target_layers"] == EXPECTED_TARGET_LAYERS,
        f"{label} PT does not cover all layers",
    )


def _validate_json_tensor_bindings(
    label: str,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    tensors = manifest["tensors"]
    non_replay = {name: value for name, value in tensors.items() if not name.startswith("replay.")}
    replay = {name: value for name, value in tensors.items() if name.startswith("replay.")}
    summaries = metadata.get("capture", {}).get("tensor_summaries", {})
    provenance = metadata.get("capture", {}).get("replay_parameter_provenance", {})
    _require(set(summaries) == set(non_replay), f"{label} JSON summary names do not bind PT")
    _require(set(provenance) == set(replay), f"{label} replay provenance names do not bind PT")

    for name, descriptor in non_replay.items():
        summary = summaries[name]
        _require(summary.get("shape") == descriptor["shape"], f"{label} {name} shape mismatch")
        _require(summary.get("dtype") == descriptor["dtype"], f"{label} {name} dtype mismatch")
        _require(
            summary.get("sha256") == descriptor["logical_row_major_sha256"],
            f"{label} {name} content hash mismatch against JSON",
        )
    for name, descriptor in replay.items():
        record = provenance[name]
        _require(record.get("shape") == descriptor["shape"], f"{label} {name} shape mismatch")
        _require(record.get("dtype") == descriptor["dtype"], f"{label} {name} dtype mismatch")


def _comparison_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        key: descriptor[key]
        for key in (
            "shape",
            "dtype",
            "numel",
            "logical_bytes",
            "logical_row_major_sha256",
        )
    }


def _category(name: str) -> str:
    if name.startswith("replay."):
        return "replay"
    if name.startswith("gdn."):
        return "gdn_internal"
    if name.startswith("attention."):
        return "full_attention_internal"
    if name.startswith("linear."):
        return "compiled_linear_output"
    if name.startswith("h") and "_post_block" in name:
        return "post_block_h"
    if name.endswith(".swiglu_output"):
        return "swiglu_output"
    return "other"


def _manifest_digest(
    names: list[str], tensors: dict[str, Any]
) -> tuple[str, str]:
    records = [
        {"name": name, **_comparison_descriptor(tensors[name])} for name in names
    ]
    return _canonical_json_sha256(names), _canonical_json_sha256(records)


def _compare_manifests(
    baseline: dict[str, Any],
    observer: dict[str, Any],
    *,
    expected_shared_non_replay: int,
    expected_observer_only_non_replay: int,
    expected_replay_parameters: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_tensors = baseline["tensors"]
    observer_tensors = observer["tensors"]
    baseline_non_replay = {
        name for name in baseline_tensors if not name.startswith("replay.")
    }
    observer_non_replay = {
        name for name in observer_tensors if not name.startswith("replay.")
    }
    shared = sorted(baseline_non_replay & observer_non_replay)
    baseline_only = sorted(baseline_non_replay - observer_non_replay)
    observer_only = sorted(observer_non_replay - baseline_non_replay)

    _require(
        len(shared) == expected_shared_non_replay,
        f"expected {expected_shared_non_replay} shared non-replay tensors, found {len(shared)}",
    )
    _require(not baseline_only, f"baseline-only non-replay tensors: {baseline_only[:5]}")
    _require(
        len(observer_only) == expected_observer_only_non_replay,
        f"expected {expected_observer_only_non_replay} observer-only tensors, found {len(observer_only)}",
    )

    shared_mismatches = [
        name
        for name in shared
        if _comparison_descriptor(baseline_tensors[name])
        != _comparison_descriptor(observer_tensors[name])
    ]
    _require(
        not shared_mismatches,
        f"shared non-replay tensors are not bit-exact: {shared_mismatches[:5]}",
    )
    shared_names_sha, shared_manifest_sha = _manifest_digest(shared, baseline_tensors)

    baseline_replay = {
        name for name in baseline_tensors if name.startswith("replay.")
    }
    observer_replay = {
        name for name in observer_tensors if name.startswith("replay.")
    }
    _require(
        len(baseline_replay) == expected_replay_parameters,
        f"baseline replay count is {len(baseline_replay)}, expected {expected_replay_parameters}",
    )
    _require(
        baseline_replay == observer_replay,
        "baseline and observer replay parameter name sets differ",
    )
    replay_names = sorted(baseline_replay)
    replay_mismatches = [
        name
        for name in replay_names
        if _comparison_descriptor(baseline_tensors[name])
        != _comparison_descriptor(observer_tensors[name])
    ]
    _require(
        not replay_mismatches,
        f"replay parameters differ: {replay_mismatches[:5]}",
    )
    replay_names_sha, replay_manifest_sha = _manifest_digest(
        replay_names, baseline_tensors
    )

    non_replay_result = {
        "baseline_tensor_count": len(baseline_non_replay),
        "observer_tensor_count": len(observer_non_replay),
        "shared_tensor_count": len(shared),
        "baseline_only_tensor_count": len(baseline_only),
        "observer_only_tensor_count": len(observer_only),
        "shared_category_counts": dict(
            sorted(Counter(_category(name) for name in shared).items())
        ),
        "observer_only_category_counts": dict(
            sorted(Counter(_category(name) for name in observer_only).items())
        ),
        "shared_logical_bytes": sum(
            baseline_tensors[name]["logical_bytes"] for name in shared
        ),
        "all_shared_bit_exact": True,
        "comparison": "shape, dtype, numel, logical byte count, and SHA-256 of logical row-major bytes",
        "shared_names_sha256": shared_names_sha,
        "shared_tensor_manifest_sha256": shared_manifest_sha,
        "mismatches": [],
    }
    replay_result = {
        "parameter_count": len(replay_names),
        "logical_bytes": sum(
            baseline_tensors[name]["logical_bytes"] for name in replay_names
        ),
        "all_names_equal": True,
        "all_shapes_equal": True,
        "all_dtypes_equal": True,
        "all_content_hashes_equal": True,
        "content_hash_algorithm": "SHA-256 over logical row-major bytes",
        "parameter_names_sha256": replay_names_sha,
        "parameter_manifest_sha256": replay_manifest_sha,
        "mismatches": [],
    }
    return non_replay_result, replay_result


def _build_proof(
    *,
    baseline_json_path: Path,
    baseline_tensors_path: Path,
    observer_json_path: Path,
    observer_tensors_path: Path,
    expected_shared_non_replay: int,
    expected_observer_only_non_replay: int,
    expected_replay_parameters: int,
    chunk_mib: int,
) -> dict[str, Any]:
    _require(chunk_mib > 0, "hash chunk size must be positive")
    chunk_bytes = chunk_mib * 1024 * 1024
    baseline_json = _load_json(baseline_json_path)
    observer_json = _load_json(observer_json_path)
    generation = _validate_json_contract(baseline_json, observer_json)
    baseline_json_record = _file_record(baseline_json_path, chunk_bytes=chunk_bytes)
    observer_json_record = _file_record(observer_json_path, chunk_bytes=chunk_bytes)

    with tempfile.TemporaryDirectory(prefix="nvfp4-pair-proof-") as temp:
        temp_path = Path(temp)
        baseline_manifest = _build_manifest_in_subprocess(
            baseline_tensors_path,
            temp_path / "baseline-manifest.json",
            chunk_mib=chunk_mib,
        )
        observer_manifest = _build_manifest_in_subprocess(
            observer_tensors_path,
            temp_path / "observer-manifest.json",
            chunk_mib=chunk_mib,
        )

    _validate_payload_metadata("baseline", baseline_json, baseline_manifest)
    _validate_payload_metadata("observer", observer_json, observer_manifest)
    _validate_json_tensor_bindings("baseline", baseline_json, baseline_manifest)
    _validate_json_tensor_bindings("observer", observer_json, observer_manifest)
    _require(
        baseline_json["capture"]["replay_parameter_provenance"]
        == observer_json["capture"]["replay_parameter_provenance"],
        "replay parameter provenance differs between captures",
    )
    non_replay, replay = _compare_manifests(
        baseline_manifest,
        observer_manifest,
        expected_shared_non_replay=expected_shared_non_replay,
        expected_observer_only_non_replay=expected_observer_only_non_replay,
        expected_replay_parameters=expected_replay_parameters,
    )

    observer_raw_missing = observer_json["capture"].get("missing", [])
    raw_missing_categories = Counter(
        "linear_input"
        if name.startswith("linear.") and name.endswith(".input")
        else "gdn_norm_hook"
        if name.startswith("gdn.")
        and name.rsplit(".", 1)[-1]
        in {"norm_core_input", "norm_z_input", "norm_output"}
        else "unexpected"
        for name in observer_raw_missing
    )
    _require(
        raw_missing_categories.get("unexpected", 0) == 0,
        "observer raw missing list contains an unexpected boundary",
    )
    baseline_non_replay_names = {
        name
        for name in baseline_manifest["tensors"]
        if not name.startswith("replay.")
    }
    observer_non_replay_names = {
        name
        for name in observer_manifest["tensors"]
        if not name.startswith("replay.")
    }
    observer_only_names = observer_non_replay_names - baseline_non_replay_names
    compile_visible_names = set(
        observer_json["capture"].get("compile_visible_observer_tensors", [])
    )
    _require(
        compile_visible_names == observer_only_names,
        "compile-visible observer name set does not match observer-only tensors",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "claim": {
            "scope": "exact compiled main-model prefill for the pinned prompt and runtime shape",
            "mtp": "off",
            "observer_graph_modified": True,
            "observer_modification_discharged": True,
            "discharge_basis": [
                "full generation-record equality with an isolated uninstrumented compiled baseline",
                f"bit-exact parity for all {expected_shared_non_replay} shared internal tensors",
                f"shape/dtype/content-hash parity for all {expected_replay_parameters} replay parameters",
            ],
            "not_claimed": [
                "MTP draft/decode parity",
                "an unmodified observer graph",
                "parity for prompts or runtime shapes not present in these artifacts",
            ],
        },
        "artifacts": {
            "baseline_json": baseline_json_record,
            "baseline_tensors": baseline_manifest["artifact"],
            "observer_json": observer_json_record,
            "observer_tensors": observer_manifest["artifact"],
        },
        "configuration": {
            "model_repo": baseline_json["model"]["repo_id"],
            "model_revision": baseline_json["model"]["revision"],
            "target_profile": "all",
            "target_layers": EXPECTED_TARGET_LAYERS,
            "main_model_layer_count": 64,
            "mtp_enabled": False,
            "language_model_only": True,
            "prompt": baseline_json["prompt"],
        },
        "generation_record_parity": {
            "exact": True,
            **generation,
        },
        "observer_capture_completeness": {
            "required_missing": [],
            "truncated": [],
            "diagnostic_postload_hook_buffers_not_observed": len(
                observer_raw_missing
            ),
            "diagnostic_missing_category_counts": dict(
                sorted(raw_missing_categories.items())
            ),
            "compile_visible_observer_tensor_count": len(compile_visible_names),
            "compile_visible_names_equal_observer_only_names": True,
            "diagnostic_note": (
                "The raw missing list contains ordinary post-load hook buffers "
                "that the cached compiled graph intentionally bypasses; all "
                "required compiled-observer boundaries are present."
            ),
        },
        "shared_internal_tensor_parity": non_replay,
        "replay_parameter_parity": {
            **replay,
            "json_provenance_equal": True,
        },
        "memory_bounded_verification": {
            "strategy": "two sequential isolated mmap-backed manifest workers",
            "maximum_concurrent_tensor_payloads": 1,
            "hash_chunk_bytes": chunk_bytes,
            "baseline_worker_peak_rss_bytes": baseline_manifest["peak_rss_bytes"],
            "observer_worker_peak_rss_bytes": observer_manifest["peak_rss_bytes"],
            "baseline_payload_tensor_count": baseline_manifest["tensor_count"],
            "observer_payload_tensor_count": observer_manifest["tensor_count"],
        },
        "verifier": _file_record(Path(__file__).resolve(), chunk_bytes=chunk_bytes),
    }


def main() -> None:
    args = _parse_args()
    _require(args.hash_chunk_mib > 0, "hash chunk size must be positive")
    if args.manifest_worker is not None:
        _require(args.manifest_output is not None, "manifest worker needs an output")
        _run_manifest_worker(
            args.manifest_worker,
            args.manifest_output,
            chunk_bytes=args.hash_chunk_mib * 1024 * 1024,
        )
        return
    _require(args.manifest_output is None, "manifest output is worker-only")
    proof = _build_proof(
        baseline_json_path=args.baseline_json,
        baseline_tensors_path=args.baseline_tensors,
        observer_json_path=args.observer_json,
        observer_tensors_path=args.observer_tensors,
        expected_shared_non_replay=args.expected_shared_non_replay,
        expected_observer_only_non_replay=args.expected_observer_only_non_replay,
        expected_replay_parameters=args.expected_replay_parameters,
        chunk_mib=args.hash_chunk_mib,
    )
    _write_json_atomic(args.output, proof)
    sys.stdout.write(json.dumps(proof, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
