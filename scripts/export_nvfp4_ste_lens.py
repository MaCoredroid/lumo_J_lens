#!/usr/bin/env python3
"""Export completed NVFP4 STE means as an upstream JacobianLens checkpoint.

The fit state deliberately publishes raw little-endian FP32 matrices so that
it can commit rows and prompt sums without pickle.  This exporter validates
that final directory, maps the matrices with ``torch.from_file``, and writes
the four-key checkpoint consumed by ``JacobianLens.load`` and
``JacobianLens.from_pretrained``::

    {"J", "n_prompts", "source_layers", "d_model"}

The source matrices stay file-backed while ``torch.save`` streams them into
the checkpoint.  The production path therefore does not first assemble the
63 matrices in anonymous RAM.  Export is fail-closed and no-clobber: source
hashes are checked before and after serialization and the completed temporary
file is hard-linked into place only if the destination is still absent.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping
import uuid

import numpy as np

try:
    from nvfp4_fit_state import (
        FINAL_ARTIFACT_SCHEMA_VERSION,
        FitLayout,
        PRODUCTION_LAYOUT,
        canonical_sha256,
        layer_path,
        read_json,
    )
except ModuleNotFoundError:  # Imported as ``scripts.export_nvfp4_ste_lens``.
    from scripts.nvfp4_fit_state import (
        FINAL_ARTIFACT_SCHEMA_VERSION,
        FitLayout,
        PRODUCTION_LAYOUT,
        canonical_sha256,
        layer_path,
        read_json,
    )


ARTIFACT_TYPE = "lumo-jlens-dense-fp32-means"
CHECKPOINT_KEYS = {"J", "n_prompts", "source_layers", "d_model"}
LAYER_RECORD_KEYS = {"layer", "filename", "shape", "dtype", "size", "sha256"}
FINAL_METADATA_KEYS = {
    "schema_version",
    "artifact_type",
    "created_at",
    "run_id",
    "contract_sha256",
    "layout",
    "n_prompts",
    "averaging",
    "layers",
    "layer_aggregate_sha256",
    "committed_prompts_sha256",
    "metadata",
    "metadata_sha256",
}
SHA256_CHUNK_BYTES = 16 * 1024 * 1024


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _open_regular_readonly(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise RuntimeError(f"input must not be a symlink: {path}") from None
        raise
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise RuntimeError(f"input must be a regular file: {path}")
    return descriptor


def _hash_file(
    path: Path,
    *,
    expected_size: int | None = None,
    require_finite_f32: bool = False,
) -> tuple[str, int]:
    """Hash one regular file, optionally checking FP32 finiteness in one scan."""

    descriptor = _open_regular_readonly(path)
    digest = hashlib.sha256()
    total = 0
    try:
        handle = os.fdopen(descriptor, "rb")
        descriptor = -1
        with handle:
            initial_size = os.fstat(handle.fileno()).st_size
            if expected_size is not None and initial_size != expected_size:
                raise RuntimeError(
                    f"unexpected file size for {path}: "
                    f"{initial_size} != {expected_size}"
                )
            while chunk := handle.read(SHA256_CHUNK_BYTES):
                total += len(chunk)
                digest.update(chunk)
                if require_finite_f32:
                    if len(chunk) % np.dtype("<f4").itemsize:
                        raise RuntimeError(f"unaligned FP32 data in {path}")
                    values = np.frombuffer(chunk, dtype="<f4")
                    if not bool(np.isfinite(values).all()):
                        raise FloatingPointError(f"non-finite matrix values in {path}")
            final_size = os.fstat(handle.fileno()).st_size
    finally:
        # ``fdopen`` owns the descriptor unless construction itself failed.
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if total != final_size or (expected_size is not None and total != expected_size):
        raise RuntimeError(f"file changed size while it was read: {path}")
    return digest.hexdigest(), total


def _parse_layout(value: Any) -> FitLayout:
    if not isinstance(value, dict):
        raise RuntimeError("final artifact layout must be a JSON object")
    required = {
        "hidden_size",
        "source_layers",
        "source_layer_count",
        "prompt_count",
        "io_rows",
        "matrix_dtype",
        "matrix_shape",
        "matrix_bytes",
        "matrix_set_bytes",
    }
    if set(value) != required:
        raise RuntimeError("final artifact layout fields do not match schema version 1")
    layers_value = value.get("source_layers")
    if not isinstance(layers_value, list) or any(
        not isinstance(layer, int) or isinstance(layer, bool) for layer in layers_value
    ):
        raise RuntimeError("final artifact source layers must be integers")
    for name in ("hidden_size", "prompt_count", "io_rows"):
        field = value.get(name)
        if not isinstance(field, int) or isinstance(field, bool):
            raise RuntimeError(f"final artifact layout {name} must be an integer")
    try:
        layout = FitLayout(
            hidden_size=value["hidden_size"],
            source_layers=tuple(layers_value),
            prompt_count=value["prompt_count"],
            io_rows=value["io_rows"],
        )
    except ValueError as error:
        raise RuntimeError(f"invalid final artifact layout: {error}") from error
    if value != layout.record():
        raise RuntimeError("final artifact layout contains inconsistent derived fields")
    return layout


def _validate_scalar_metadata(metadata: Mapping[str, Any], layout: FitLayout) -> None:
    if set(metadata) != FINAL_METADATA_KEYS:
        raise RuntimeError(
            "final artifact metadata fields do not match schema version 1"
        )
    if metadata.get("schema_version") != FINAL_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("unsupported final artifact schema version")
    if metadata.get("artifact_type") != ARTIFACT_TYPE:
        raise RuntimeError("unexpected final artifact type")
    if metadata.get("n_prompts") != layout.prompt_count:
        raise RuntimeError("final artifact prompt count does not match its layout")
    for name in ("created_at", "run_id"):
        if not isinstance(metadata.get(name), str) or not metadata[name]:
            raise RuntimeError(f"final artifact {name} must be a nonempty string")
    if metadata.get("averaging") != (
        "arithmetic mean of cumulative little-endian FP32 sums"
    ):
        raise RuntimeError("unexpected final artifact averaging rule")
    for name in (
        "contract_sha256",
        "layer_aggregate_sha256",
        "committed_prompts_sha256",
        "metadata_sha256",
    ):
        if not _is_sha256(metadata.get(name)):
            raise RuntimeError(f"final artifact {name} is not a SHA-256 digest")
    payload = metadata.get("metadata")
    if not isinstance(payload, dict):
        raise RuntimeError("final artifact metadata payload must be a JSON object")
    if metadata["metadata_sha256"] != canonical_sha256(payload):
        raise RuntimeError("final artifact metadata payload hash mismatch")


def validate_final_mean(
    directory: Path,
    *,
    expected_layout: FitLayout = PRODUCTION_LAYOUT,
) -> dict[str, Any]:
    """Validate a completed raw final-mean directory in bounded memory."""

    directory = directory.expanduser()
    try:
        directory_mode = directory.lstat().st_mode
    except FileNotFoundError:
        raise FileNotFoundError(
            f"final-mean directory does not exist: {directory}"
        ) from None
    if not stat.S_ISDIR(directory_mode):
        raise RuntimeError(
            f"final-mean input must be a real directory, not a symlink: {directory}"
        )
    directory = directory.resolve()
    metadata_path = directory / "metadata.json"
    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise RuntimeError("final artifact metadata must be a JSON object")
    layout = _parse_layout(metadata.get("layout"))
    if layout != expected_layout:
        raise RuntimeError(
            "final artifact layout does not match the requested export layout: "
            f"{layout.record()} != {expected_layout.record()}"
        )
    _validate_scalar_metadata(metadata, layout)

    records = metadata.get("layers")
    if not isinstance(records, list) or len(records) != len(layout.source_layers):
        raise RuntimeError("final artifact layer manifest is incomplete")
    validated_records: list[dict[str, Any]] = []
    for layer, record in zip(layout.source_layers, records, strict=True):
        path = layer_path(directory, layer)
        if not isinstance(record, dict) or set(record) != LAYER_RECORD_KEYS:
            raise RuntimeError(f"invalid final artifact layer record at layer {layer}")
        expected_record = {
            "layer": layer,
            "filename": path.name,
            "shape": [layout.hidden_size, layout.hidden_size],
            "dtype": "little-endian-float32",
            "size": layout.matrix_bytes,
        }
        for name, expected in expected_record.items():
            if record.get(name) != expected:
                raise RuntimeError(
                    f"final artifact layer {layer} has inconsistent {name}"
                )
        if not _is_sha256(record.get("sha256")):
            raise RuntimeError(f"final artifact layer {layer} has an invalid SHA-256")
        actual_sha256, _ = _hash_file(
            path,
            expected_size=layout.matrix_bytes,
            require_finite_f32=True,
        )
        if actual_sha256 != record["sha256"]:
            raise RuntimeError(f"final artifact hash mismatch at layer {layer}")
        validated_records.append(dict(record))

    if metadata["layer_aggregate_sha256"] != canonical_sha256(validated_records):
        raise RuntimeError("final artifact layer aggregate hash mismatch")
    metadata_sha256, metadata_size = _hash_file(metadata_path)
    return {
        "directory": directory,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "metadata_file_sha256": metadata_sha256,
        "metadata_file_size": metadata_size,
        "layout": layout,
        "layers": tuple(validated_records),
    }


def _map_jacobians(artifact: Mapping[str, Any]) -> dict[int, Any]:
    if sys.byteorder != "little":
        raise RuntimeError("raw little-endian matrices require a little-endian host")
    import torch

    layout: FitLayout = artifact["layout"]
    count = layout.hidden_size * layout.hidden_size
    result: dict[int, Any] = {}
    for layer in layout.source_layers:
        tensor = torch.from_file(
            str(layer_path(artifact["directory"], layer)),
            shared=False,
            size=count,
            dtype=torch.float32,
        )
        result[layer] = tensor.reshape(layout.hidden_size, layout.hidden_size)
    return result


def verify_exported_checkpoint(
    path: Path,
    *,
    layout: FitLayout,
    n_prompts: int,
) -> dict[str, Any]:
    """Check the serialized header and file-backed tensor geometry."""

    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != CHECKPOINT_KEYS:
        keys = sorted(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint)
        raise RuntimeError(f"unexpected exported checkpoint keys: {keys}")
    if checkpoint["n_prompts"] != n_prompts:
        raise RuntimeError("exported checkpoint prompt count mismatch")
    if checkpoint["source_layers"] != list(layout.source_layers):
        raise RuntimeError("exported checkpoint source layers mismatch")
    if checkpoint["d_model"] != layout.hidden_size:
        raise RuntimeError("exported checkpoint d_model mismatch")
    jacobians = checkpoint["J"]
    if not isinstance(jacobians, dict) or set(jacobians) != set(layout.source_layers):
        raise RuntimeError("exported checkpoint Jacobian keys mismatch")
    for layer in layout.source_layers:
        tensor = jacobians[layer]
        if (
            not torch.is_tensor(tensor)
            or tuple(tensor.shape) != (layout.hidden_size, layout.hidden_size)
            or tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
        ):
            raise RuntimeError(f"exported checkpoint tensor mismatch at layer {layer}")
    del checkpoint
    return {
        "checkpoint_keys": sorted(CHECKPOINT_KEYS),
        "d_model": layout.hidden_size,
        "n_prompts": n_prompts,
        "source_layers": list(layout.source_layers),
        "tensor_dtype": "torch.float32",
        "tensor_shape": [layout.hidden_size, layout.hidden_size],
    }


def _destination_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _revalidate_sources(artifact: Mapping[str, Any]) -> None:
    metadata_sha256, metadata_size = _hash_file(artifact["metadata_path"])
    if (
        metadata_sha256 != artifact["metadata_file_sha256"]
        or metadata_size != artifact["metadata_file_size"]
    ):
        raise RuntimeError("final artifact metadata changed during export")
    layout: FitLayout = artifact["layout"]
    for record in artifact["layers"]:
        path = layer_path(artifact["directory"], record["layer"])
        actual_sha256, _ = _hash_file(path, expected_size=layout.matrix_bytes)
        if actual_sha256 != record["sha256"]:
            raise RuntimeError(
                f"final artifact layer {record['layer']} changed during export"
            )


def export_lens(
    final_mean: Path,
    output: Path,
    *,
    expected_layout: FitLayout = PRODUCTION_LAYOUT,
) -> dict[str, Any]:
    """Validate and atomically export one upstream-compatible lens file."""

    artifact = validate_final_mean(final_mean, expected_layout=expected_layout)
    output = output.expanduser().resolve()
    if _destination_exists(output):
        raise FileExistsError(f"refusing to overwrite export destination: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output in {
        artifact["metadata_path"],
        *(
            layer_path(artifact["directory"], layer)
            for layer in expected_layout.source_layers
        ),
    }:
        raise RuntimeError("export destination aliases a final-mean input file")

    import torch

    temporary = output.with_name(
        f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        jacobians = _map_jacobians(artifact)
        checkpoint = {
            "J": jacobians,
            "n_prompts": artifact["metadata"]["n_prompts"],
            "source_layers": list(artifact["layout"].source_layers),
            "d_model": artifact["layout"].hidden_size,
        }
        with temporary.open("xb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        del checkpoint, jacobians

        verification = verify_exported_checkpoint(
            temporary,
            layout=artifact["layout"],
            n_prompts=artifact["metadata"]["n_prompts"],
        )
        _revalidate_sources(artifact)
        output_sha256, output_size = _hash_file(temporary)
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise FileExistsError(
                f"export destination appeared concurrently: {output}"
            ) from None
        _fsync_directory(output.parent)
        temporary.unlink()
        _fsync_directory(output.parent)
    finally:
        if _destination_exists(temporary):
            temporary.unlink()

    return {
        "artifact_type": "upstream-jacobian-lens-torch-checkpoint",
        "path": str(output),
        "size_bytes": output_size,
        "sha256": output_sha256,
        "source": {
            "final_mean": str(artifact["directory"]),
            "metadata_sha256": artifact["metadata_file_sha256"],
            "layer_aggregate_sha256": artifact["metadata"][
                "layer_aggregate_sha256"
            ],
            "contract_sha256": artifact["metadata"]["contract_sha256"],
        },
        **verification,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-mean",
        type=Path,
        required=True,
        help="completed FitStateStore final-mean directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new lens.pt destination (existing paths are never overwritten)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = export_lens(args.final_mean, args.output)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
