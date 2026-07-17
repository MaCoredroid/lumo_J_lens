#!/usr/bin/env python3
"""On-demand access to raw ModelOpt W4A16 tensors in a sharded checkpoint.

This loader is deliberately narrow.  It exposes only checkpoint modules marked
``W4A16_NVFP4`` and rejects FP8 modules because their raw shard scales do not
describe vLLM's post-load, fused/requantized runtime weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import torch

try:
    from scripts.nvfp4_ste import dequantize_nvfp4_weight
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from nvfp4_ste import dequantize_nvfp4_weight


PINNED_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
PINNED_METADATA_SHA256 = {
    "config.json": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
    "hf_quant_config.json": "fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1",
    "model.safetensors.index.json": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
}
PINNED_SHARDS = {
    "model-00001-of-00003.safetensors": (
        9_965_652_512,
        "b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d",
    ),
    "model-00002-of-00003.safetensors": (
        9_985_757_032,
        "06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d",
    ),
    "model-00003-of-00003.safetensors": (
        1_970_287_640,
        "e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845",
    ),
}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Hash a checkpoint file without materializing it in host memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    shard: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class NvFp4Metadata:
    """Non-materializing description of one raw or fused NVFP4 module."""

    module_name: str
    components: tuple[str, ...]
    packed_shape: tuple[int, int]
    block_scale_shape: tuple[int, int]
    block_size: int
    tensors: tuple[TensorMetadata, ...]


@dataclass(frozen=True)
class RawNvFp4Weight:
    """Raw, unswizzled ModelOpt weight suitable for surrogate W4 VJPs."""

    module_name: str
    components: tuple[str, ...]
    packed_weight: torch.Tensor
    block_scales: torch.Tensor
    global_scale: torch.Tensor
    block_size: int
    source_shards: tuple[str, ...]

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return the frozen effective ``[out, in]`` weight."""

        return dequantize_nvfp4_weight(
            self.packed_weight,
            self.block_scales,
            self.global_scale,
            block_size=self.block_size,
            dtype=dtype,
        )


def default_pinned_snapshot() -> Path:
    """Return the normal Hugging Face cache path for the pinned checkpoint."""

    hf_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ).expanduser()
    return (
        hf_home
        / "hub"
        / "models--nvidia--Qwen3.6-27B-NVFP4"
        / "snapshots"
        / PINNED_REVISION
    )


class ModelOptCheckpoint:
    """Read selected raw W4 tensors without materializing a whole checkpoint.

    Pinned integrity checks are enabled by default.  ``strict_pinned=False`` is
    intended for synthetic fixtures or an explicitly reviewed checkpoint.
    """

    def __init__(self, snapshot: str | Path, *, strict_pinned: bool = True) -> None:
        self.snapshot = Path(snapshot).expanduser().absolute()
        if not self.snapshot.is_dir():
            raise FileNotFoundError(f"checkpoint snapshot not found: {self.snapshot}")
        if strict_pinned:
            self._validate_pinned_files()

        self.config = self._read_json("config.json")
        self.quant_config = self._read_json("hf_quant_config.json")
        self.index = self._read_json("model.safetensors.index.json")
        self.weight_map = self._validate_index()
        self.quantized_layers = self._validate_quantization_config()

    def _read_json(self, filename: str) -> dict[str, Any]:
        path = self.snapshot / filename
        if not path.is_file():
            raise FileNotFoundError(f"required checkpoint file missing: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"invalid JSON in {path}") from error
        if not isinstance(value, dict):
            raise ValueError(f"checkpoint JSON root must be an object: {path}")
        return value

    def _validate_pinned_files(self) -> None:
        if self.snapshot.name != PINNED_REVISION:
            raise ValueError(
                f"snapshot revision {self.snapshot.name!r} != pinned {PINNED_REVISION}"
            )

        for filename, expected in PINNED_METADATA_SHA256.items():
            path = self.snapshot / filename
            if not path.is_file():
                raise FileNotFoundError(f"required checkpoint file missing: {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch for {filename}: {actual} != {expected}"
                )

        for filename, (expected_size, expected_blob) in PINNED_SHARDS.items():
            path = self.snapshot / filename
            if not path.is_file():
                raise FileNotFoundError(f"required checkpoint shard missing: {path}")
            if path.stat().st_size != expected_size:
                raise ValueError(
                    f"size mismatch for {filename}: {path.stat().st_size} "
                    f"!= {expected_size}"
                )
            if path.is_symlink():
                blob_name = Path(os.readlink(path)).name
                if blob_name != expected_blob:
                    raise ValueError(
                        f"Hugging Face blob mismatch for {filename}: "
                        f"{blob_name} != {expected_blob}"
                    )
            actual_sha256 = sha256_file(path)
            if actual_sha256 != expected_blob:
                raise ValueError(
                    f"SHA-256 mismatch for {filename}: "
                    f"{actual_sha256} != {expected_blob}"
                )

    def validate_pinned_integrity(self) -> None:
        """Rehash the pinned checkpoint after a long-running computation."""
        self._validate_pinned_files()

    def _validate_index(self) -> dict[str, str]:
        weight_map = self.index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("checkpoint index must contain a non-empty weight_map")

        checked: dict[str, str] = {}
        for tensor_name, shard in weight_map.items():
            if not isinstance(tensor_name, str) or not isinstance(shard, str):
                raise ValueError("checkpoint weight_map keys and values must be strings")
            shard_path = Path(shard)
            if shard_path.name != shard or shard_path.suffix != ".safetensors":
                raise ValueError(f"unsafe checkpoint shard name: {shard!r}")
            if not (self.snapshot / shard).is_file():
                raise FileNotFoundError(f"indexed checkpoint shard missing: {shard}")
            checked[tensor_name] = shard
        return checked

    def _validate_quantization_config(self) -> dict[str, Mapping[str, Any]]:
        producer = self.quant_config.get("producer", {})
        quantization = self.quant_config.get("quantization", {})
        if not isinstance(producer, dict) or producer.get("name") != "modelopt":
            raise ValueError("hf_quant_config.json is not a ModelOpt checkpoint")
        if not isinstance(quantization, dict):
            raise ValueError("hf_quant_config.json lacks quantization metadata")
        if quantization.get("quant_algo") != "MIXED_PRECISION":
            raise ValueError("checkpoint quant_algo must be MIXED_PRECISION")
        layers = quantization.get("quantized_layers")
        if not isinstance(layers, dict):
            raise ValueError("checkpoint lacks a quantized_layers mapping")

        embedded = self.config.get("quantization_config", {})
        if not isinstance(embedded, dict) or embedded.get("quant_method") != "modelopt":
            raise ValueError("config.json does not declare ModelOpt quantization")
        if embedded.get("quant_algo") != "MIXED_PRECISION":
            raise ValueError("config.json quant_algo must be MIXED_PRECISION")
        return layers

    @staticmethod
    def _component_names(module_name: str) -> tuple[str, ...]:
        if module_name.endswith(".gate_up_proj"):
            stem = module_name[: -len("gate_up_proj")]
            return stem + "gate_proj", stem + "up_proj"
        return (module_name,)

    def _block_size(self, module_name: str) -> int:
        metadata = self.quantized_layers.get(module_name)
        if not isinstance(metadata, Mapping):
            raise KeyError(f"module is not declared quantized: {module_name}")
        algorithm = metadata.get("quant_algo")
        if algorithm != "W4A16_NVFP4":
            detail = (
                "raw FP8 checkpoint weights are forbidden for runtime VJPs; "
                "capture the post-load/requantized vLLM weight"
                if algorithm == "FP8"
                else f"unsupported quantization algorithm {algorithm!r}"
            )
            raise ValueError(f"{module_name}: {detail}")
        group_size = metadata.get("group_size")
        if group_size != 16:
            raise ValueError(
                f"{module_name}: NVFP4 group_size {group_size!r} != expected 16"
            )
        return group_size

    def _tensor_metadata(self, name: str) -> TensorMetadata:
        try:
            shard = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"tensor absent from checkpoint index: {name}") from error
        path = self.snapshot / shard
        with safe_open(path, framework="pt", device="cpu") as handle:
            try:
                view = handle.get_slice(name)
            except Exception as error:
                raise ValueError(f"indexed tensor missing from shard: {name} in {shard}") from error
            return TensorMetadata(name, shard, tuple(view.get_shape()), view.get_dtype())

    def inspect_nvfp4(self, module_name: str) -> NvFp4Metadata:
        """Inspect shapes and dtypes without reading the tensor payloads."""

        components = self._component_names(module_name)
        block_sizes = {self._block_size(component) for component in components}
        if len(block_sizes) != 1:
            raise ValueError(f"fused module has inconsistent block sizes: {module_name}")
        block_size = block_sizes.pop()

        tensors: list[TensorMetadata] = []
        packed_rows = scale_rows = 0
        packed_k: int | None = None
        scale_k: int | None = None
        for component in components:
            packed = self._tensor_metadata(component + ".weight")
            scales = self._tensor_metadata(component + ".weight_scale")
            global_scale = self._tensor_metadata(component + ".weight_scale_2")
            tensors.extend((packed, scales, global_scale))
            if len(packed.shape) != 2 or packed.dtype != "U8":
                raise ValueError(f"invalid packed NVFP4 tensor metadata: {packed}")
            if len(scales.shape) != 2 or scales.dtype != "F8_E4M3":
                raise ValueError(f"invalid NVFP4 block-scale metadata: {scales}")
            if global_scale.shape not in {(), (1,)} or global_scale.dtype != "F32":
                raise ValueError(f"invalid NVFP4 global-scale metadata: {global_scale}")
            if scales.shape != (packed.shape[0], packed.shape[1] * 2 // block_size):
                raise ValueError(f"NVFP4 weight/scale shapes disagree for {component}")
            if packed_k is not None and packed.shape[1] != packed_k:
                raise ValueError(f"fused module input dimensions disagree: {module_name}")
            if scale_k is not None and scales.shape[1] != scale_k:
                raise ValueError(f"fused module scale dimensions disagree: {module_name}")
            packed_k, scale_k = packed.shape[1], scales.shape[1]
            packed_rows += packed.shape[0]
            scale_rows += scales.shape[0]

        assert packed_k is not None and scale_k is not None
        return NvFp4Metadata(
            module_name=module_name,
            components=components,
            packed_shape=(packed_rows, packed_k),
            block_scale_shape=(scale_rows, scale_k),
            block_size=block_size,
            tensors=tuple(tensors),
        )

    def _load_tensors(
        self, names: Sequence[str], *, device: str | int
    ) -> dict[str, torch.Tensor]:
        by_shard: dict[str, list[str]] = {}
        for name in names:
            try:
                shard = self.weight_map[name]
            except KeyError as error:
                raise KeyError(f"tensor absent from checkpoint index: {name}") from error
            by_shard.setdefault(shard, []).append(name)

        result: dict[str, torch.Tensor] = {}
        for shard, shard_names in by_shard.items():
            with safe_open(
                self.snapshot / shard, framework="pt", device=device
            ) as handle:
                for name in shard_names:
                    try:
                        result[name] = handle.get_tensor(name)
                    except Exception as error:
                        raise ValueError(
                            f"indexed tensor missing from shard: {name} in {shard}"
                        ) from error
        return result

    def load_nvfp4(
        self, module_name: str, *, device: str | int = "cpu"
    ) -> RawNvFp4Weight:
        """Load one raw W4 module or a ``gate_up_proj`` fused alias on demand."""

        metadata = self.inspect_nvfp4(module_name)
        names = [tensor.name for tensor in metadata.tensors]
        tensors = self._load_tensors(names, device=device)

        packed_parts: list[torch.Tensor] = []
        scale_parts: list[torch.Tensor] = []
        global_parts: list[torch.Tensor] = []
        for component in metadata.components:
            packed_parts.append(tensors[component + ".weight"])
            scale_parts.append(tensors[component + ".weight_scale"])
            global_parts.append(tensors[component + ".weight_scale_2"])

        for component, packed, scales, global_scale in zip(
            metadata.components,
            packed_parts,
            scale_parts,
            global_parts,
            strict=True,
        ):
            if packed.dtype != torch.uint8:
                raise TypeError(f"{component}.weight must be uint8")
            if scales.dtype != torch.float8_e4m3fn:
                raise TypeError(f"{component}.weight_scale must be float8_e4m3fn")
            if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
                raise TypeError(f"{component}.weight_scale_2 must be one float32 value")
            if not bool(torch.isfinite(global_scale).all()) or float(global_scale) <= 0:
                raise ValueError(f"{component}.weight_scale_2 must be finite and positive")

        first_global = global_parts[0].reshape(())
        if any(not torch.equal(part.reshape(()), first_global) for part in global_parts[1:]):
            raise ValueError(
                f"cannot fuse {module_name}: component global scales are unequal"
            )
        packed_weight = (
            packed_parts[0]
            if len(packed_parts) == 1
            else torch.cat(packed_parts, dim=0)
        )
        block_scales = (
            scale_parts[0]
            if len(scale_parts) == 1
            else torch.cat(scale_parts, dim=0)
        )
        source_shards = tuple(dict.fromkeys(tensor.shard for tensor in metadata.tensors))
        return RawNvFp4Weight(
            module_name=module_name,
            components=metadata.components,
            packed_weight=packed_weight,
            block_scales=block_scales,
            global_scale=first_global,
            block_size=metadata.block_size,
            source_shards=source_shards,
        )
