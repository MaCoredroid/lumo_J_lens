#!/usr/bin/env python3
"""Capture the deployed Qwen3.6 NVFP4 operator boundaries in vLLM.

The compiled run has two intentionally distinct phases:

1. An uninstrumented generation is the authoritative serving-graph endpoint.
2. Forward hooks are then installed and the same prompt is rerun after the
   prefix cache is reset. These tensors are useful for parity diagnostics, but
   are labelled as instrumented because hooks can invalidate a compiled graph
   or inhibit the RMSNorm/FP8-quant fusion pass.

Run with VLLM_ENABLE_V1_MULTIPROCESSING=0. ``--mode both`` launches isolated
eager, compiled, and compiled-observer child processes. MTP is intentionally
disabled: this checker covers the main-model prefill only. The proven
layers 61-63 remain the default; use ``--target-layers all`` only after the
memory preflight reports ``safe``.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
MODEL_IDENTITY_POLICY = "pinned-nvidia-modelopt-snapshot-v1"
PINNED_METADATA_SHA256 = {
    "config.json": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
    "hf_quant_config.json": "fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1",
    "model.safetensors.index.json": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
}
DEFAULT_PROMPT = "Fact: The currency used in the country shaped like a boot is"
EXPECTED_LAYER_COUNT = 64
LATE_TARGET_LAYERS = (61, 62, 63)
LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"
# The validated cap-64 late observer reserved 26.631 GiB versus 25.132 GiB
# for its engine budget plus capture buffers.
EMPIRICAL_RUNTIME_OVERHEAD_BYTES = 3 * 512 * 1024 * 1024
DEFAULT_SAFETY_MARGIN_BYTES = 512 * 1024 * 1024
FP8_MAX = 448.0
SCHEMA_VERSION = 1
DISABLED_KERNELS = (
    "FlashInferFP8ScaledMMLinearKernel,"
    "FlashInferCutlassNvFp4LinearKernel,"
    "FlashInferTrtllmNvFp4LinearKernel,"
    "FlashInferCudnnNvFp4LinearKernel"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("eager", "compiled", "compiled-observer", "both"),
        default="both",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help=(
            "diagnostic path override; accepted only when it resolves to the exact "
            "locally cached pinned NVIDIA snapshot"
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        help="frozen JSON prompt manifest; overrides --prompt",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=0,
        help="zero-based entry in --prompt-manifest",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensor-output", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--capture-capacity", type=int, default=128)
    parser.add_argument(
        "--target-layers",
        choices=("late", "all"),
        default="late",
        help="capture the proven layers 61-63 or every main-model layer",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="estimate capture memory from config and exit without loading vLLM",
    )
    parser.add_argument(
        "--total-gpu-memory-gib",
        type=float,
        help="override detected device memory for deterministic preflight checks",
    )
    parser.add_argument(
        "--allow-unsafe-capture-memory",
        action="store_true",
        help="run even when the conservative capture-memory estimate exceeds VRAM",
    )
    parser.add_argument(
        "--no-weight-digests",
        action="store_true",
        help="skip SHA-256 over the post-load FP8 tensors",
    )
    return parser.parse_args()


def _load_frozen_prompt(path: Path, index: int) -> dict[str, Any]:
    """Load one hash-checked prompt entry with its declared truncated IDs."""

    if index < 0:
        raise ValueError("prompt index must be non-negative")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("prompt manifest must be a schema-version-1 object")
    prompts = raw.get("prompts")
    if not isinstance(prompts, list) or index >= len(prompts):
        raise ValueError(f"prompt index {index} is outside the manifest")
    entry = prompts[index]
    if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
        raise ValueError(f"prompt manifest entry {index} has no text")
    token_ids = entry.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids or not all(
        isinstance(token, int) and token >= 0 for token in token_ids
    ):
        raise ValueError(f"prompt manifest entry {index} has invalid token_ids")
    if entry.get("token_count") != len(token_ids):
        raise ValueError(f"prompt manifest entry {index} token_count mismatch")
    text_sha256 = hashlib.sha256(entry["text"].encode("utf-8")).hexdigest()
    if entry.get("text_sha256") != text_sha256:
        raise ValueError(f"prompt manifest entry {index} text SHA-256 mismatch")
    return {
        "text": entry["text"],
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "text_sha256": text_sha256,
        "row_index": entry.get("row_index"),
        "manifest_path": str(path.resolve()),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "manifest_index": index,
    }


def _resolve_target_layers(profile: str, layer_count: int) -> tuple[int, ...]:
    if profile == "all":
        return tuple(range(layer_count))
    if profile == "late":
        if layer_count <= max(LATE_TARGET_LAYERS):
            raise ValueError(
                f"late target requires layer {max(LATE_TARGET_LAYERS)}, "
                f"but config has {layer_count} layers"
            )
        return LATE_TARGET_LAYERS
    raise ValueError(f"unknown target-layer profile: {profile}")


def _partition_target_layers(
    target_layers: tuple[int, ...], layer_types: list[str] | tuple[str, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    gdn_layers = tuple(
        index for index in target_layers if layer_types[index] == LINEAR_ATTENTION
    )
    full_attention_layers = tuple(
        index for index in target_layers if layer_types[index] == FULL_ATTENTION
    )
    unsupported = {
        layer_types[index]
        for index in target_layers
        if layer_types[index] not in {LINEAR_ATTENTION, FULL_ATTENTION}
    }
    if unsupported:
        raise ValueError(f"unsupported target layer types: {sorted(unsupported)}")
    return gdn_layers, full_attention_layers


def _load_text_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    required = {
        "hidden_size",
        "intermediate_size",
        "layer_types",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    }
    missing = sorted(required - set(text_config))
    if missing:
        raise ValueError(f"text config is missing capture geometry: {missing}")
    return text_config


def _capture_geometry(text_config: dict[str, Any]) -> dict[str, int]:
    hidden = int(text_config["hidden_size"])
    intermediate = int(text_config["intermediate_size"])
    attention_heads = int(text_config["num_attention_heads"])
    attention_head_dim = int(text_config["head_dim"])
    kv_heads = int(text_config["num_key_value_heads"])
    linear_key_heads = int(text_config["linear_num_key_heads"])
    linear_value_heads = int(text_config["linear_num_value_heads"])
    linear_key_dim = int(text_config["linear_key_head_dim"])
    linear_value_dim = int(text_config["linear_value_head_dim"])
    full_q = attention_heads * attention_head_dim
    full_kv = kv_heads * attention_head_dim
    linear_k = linear_key_heads * linear_key_dim
    linear_v = linear_value_heads * linear_value_dim
    output_gate = full_q if text_config.get("attn_output_gate", False) else 0
    return {
        "hidden": hidden,
        "intermediate": intermediate,
        "gate_up": 2 * intermediate,
        "attention_head_dim": attention_head_dim,
        "full_q": full_q,
        "full_kv": full_kv,
        "full_qkv": full_q + 2 * full_kv + output_gate,
        "linear_key_heads": linear_key_heads,
        "linear_value_heads": linear_value_heads,
        "linear_key_dim": linear_key_dim,
        "linear_value_dim": linear_value_dim,
        "linear_k": linear_k,
        "linear_v": linear_v,
        "gdn_mixed": 2 * linear_k + linear_v,
        "gdn_qkvz": 2 * linear_k + 2 * linear_v,
        "gdn_ba": 2 * linear_value_heads,
        "conv_kernel": int(text_config["linear_conv_kernel_dim"]),
    }


def _estimate_capture_memory(
    *,
    text_config: dict[str, Any],
    target_layers: tuple[int, ...],
    capture_capacity: int,
    gpu_memory_utilization: float,
    total_gpu_bytes: int,
    safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES,
) -> dict[str, Any]:
    """Return a conservative allocation model without importing vLLM."""
    if capture_capacity < 1:
        raise ValueError("capture capacity must be positive")
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu memory utilization must be between zero and one")
    if total_gpu_bytes < 1:
        raise ValueError("total GPU memory must be positive")

    layer_types = list(text_config["layer_types"])
    gdn_layers, full_attention_layers = _partition_target_layers(
        target_layers, layer_types
    )
    geometry = _capture_geometry(text_config)
    bf16_bytes = 2
    fp32_bytes = 4
    int32_bytes = 4
    fp8_bytes = 1

    hidden = geometry["hidden"]
    intermediate = geometry["intermediate"]
    gate_up = geometry["gate_up"]
    attention_head_dim = geometry["attention_head_dim"]
    full_q = geometry["full_q"]
    full_kv = geometry["full_kv"]
    full_qkv = geometry["full_qkv"]
    linear_key_heads = geometry["linear_key_heads"]
    linear_value_heads = geometry["linear_value_heads"]
    linear_key_dim = geometry["linear_key_dim"]
    linear_value_dim = geometry["linear_value_dim"]
    linear_k = geometry["linear_k"]
    linear_v = geometry["linear_v"]
    gdn_mixed = geometry["gdn_mixed"]
    gdn_qkvz = geometry["gdn_qkvz"]
    gdn_ba = geometry["gdn_ba"]

    # These output-only buffers are registered before LLM construction in
    # compiled-observer mode, so vLLM includes them in its engine profiling.
    observer_gdn_elements = (
        hidden
        + intermediate
        + gdn_mixed  # conv1d is a LinearBase even though forward is bypassed
        + gdn_qkvz
        + gdn_ba
        + hidden
        + gate_up
        + hidden
    )
    observer_full_elements = (
        hidden + intermediate + full_qkv + hidden + gate_up + hidden
    )
    observer_buffer_count = 8 * len(gdn_layers) + 6 * len(full_attention_layers)
    compile_visible_observer_bytes = (
        capture_capacity
        * bf16_bytes
        * (
            observer_gdn_elements * len(gdn_layers)
            + observer_full_elements * len(full_attention_layers)
        )
        + observer_buffer_count * int32_bytes
    )

    postload_gdn_bf16_elements = (
        hidden
        + intermediate
        + (hidden + gdn_qkvz)
        + (hidden + gdn_ba)
        + (linear_v + hidden)
        + (hidden + gate_up)
        + (intermediate + hidden)
        + 2 * gdn_mixed
        + 2 * linear_value_heads
        + 6 * linear_v
        + 2 * linear_k
    )
    postload_gdn_fp32_elements = 2 * linear_value_heads
    gdn_state_elements = linear_value_heads * linear_value_dim * linear_key_dim
    postload_full_bf16_elements = (
        hidden
        + intermediate
        + (hidden + full_qkv)
        + (full_q + hidden)
        + (hidden + gate_up)
        + (intermediate + hidden)
        + 2 * full_q
        + 2 * full_kv
    )
    postload_buffer_count = 28 * len(gdn_layers) + 14 * len(full_attention_layers)
    postload_capture_bytes = (
        capture_capacity
        * (
            bf16_bytes
            * (
                postload_gdn_bf16_elements * len(gdn_layers)
                + postload_full_bf16_elements * len(full_attention_layers)
            )
            + fp32_bytes * postload_gdn_fp32_elements * len(gdn_layers)
        )
        + fp32_bytes * 2 * gdn_state_elements * len(gdn_layers)
        + postload_buffer_count * int32_bytes
    )

    replay_gdn_bytes = (
        fp8_bytes * (hidden * gdn_qkvz + linear_v * hidden)
        + fp32_bytes * 4
        + bf16_bytes
        * (
            gdn_ba * hidden
            + gdn_mixed * geometry["conv_kernel"]
            + linear_value_heads
            + linear_value_dim
            + 2 * hidden
        )
        + fp32_bytes * linear_value_heads
    )
    replay_full_bytes = (
        fp8_bytes * (hidden * full_qkv + full_q * hidden)
        + fp32_bytes * 4
        + bf16_bytes * (2 * attention_head_dim + 2 * hidden)
    )
    replay_parameter_host_bytes = (
        replay_gdn_bytes * len(gdn_layers)
        + replay_full_bytes * len(full_attention_layers)
        + bf16_bytes * hidden
    )
    replay_parameter_tensor_count = (
        13 * len(gdn_layers) + 10 * len(full_attention_layers) + 1
    )

    engine_budget_bytes = int(total_gpu_bytes * gpu_memory_utilization)
    estimated_peak_gpu_bytes = (
        engine_budget_bytes
        + postload_capture_bytes
        + EMPIRICAL_RUNTIME_OVERHEAD_BYTES
        + safety_margin_bytes
    )
    headroom_bytes = total_gpu_bytes - estimated_peak_gpu_bytes
    safe = headroom_bytes >= 0

    def gib(value: int) -> float:
        return value / (1024**3)

    return {
        "target_layer_count": len(target_layers),
        "gdn_layer_count": len(gdn_layers),
        "full_attention_layer_count": len(full_attention_layers),
        "capture_capacity": capture_capacity,
        "total_gpu_bytes": total_gpu_bytes,
        "total_gpu_gib": gib(total_gpu_bytes),
        "gpu_memory_utilization": gpu_memory_utilization,
        "engine_profile_budget_bytes": engine_budget_bytes,
        "engine_profile_budget_gib": gib(engine_budget_bytes),
        "compile_visible_observer_buffer_count": observer_buffer_count,
        "compile_visible_observer_gpu_bytes": compile_visible_observer_bytes,
        "compile_visible_observer_gpu_gib": gib(compile_visible_observer_bytes),
        "postload_capture_buffer_count": postload_buffer_count,
        "postload_capture_gpu_bytes": postload_capture_bytes,
        "postload_capture_gpu_gib": gib(postload_capture_bytes),
        "empirical_runtime_overhead_bytes": EMPIRICAL_RUNTIME_OVERHEAD_BYTES,
        "empirical_runtime_overhead_gib": gib(EMPIRICAL_RUNTIME_OVERHEAD_BYTES),
        "safety_margin_bytes": safety_margin_bytes,
        "safety_margin_gib": gib(safety_margin_bytes),
        "estimated_peak_gpu_bytes": estimated_peak_gpu_bytes,
        "estimated_peak_gpu_gib": gib(estimated_peak_gpu_bytes),
        "estimated_gpu_headroom_bytes": headroom_bytes,
        "estimated_gpu_headroom_gib": gib(headroom_bytes),
        "replay_parameter_host_bytes": replay_parameter_host_bytes,
        "replay_parameter_host_gib": gib(replay_parameter_host_bytes),
        "replay_parameter_tensor_count": replay_parameter_tensor_count,
        "nvfp4_checkpoint_weights_in_replay_export": False,
        "safe": safe,
        "accounting_note": (
            "compile-visible observer buffers are present during vLLM profiling; "
            "the peak estimate adds post-load capture buffers, 1.5 GiB of "
            "allocator/compiler overhead measured in the validated late-layer "
            "run, and a separate safety margin to the configured engine budget"
        ),
    }


def _prepare_process_environment() -> None:
    # These must be set before importing vLLM.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_DISABLED_KERNELS", DISABLED_KERNELS)
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _text_model_parts(model: Any) -> tuple[Any, Any]:
    language_model = model.language_model if hasattr(model, "language_model") else model
    text_model = language_model.model
    if not hasattr(text_model, "layers"):
        raise TypeError(f"unsupported vLLM model layout: {type(model).__name__}")
    return language_model, text_model


def _class_name(value: Any) -> str | None:
    return None if value is None else f"{type(value).__module__}.{type(value).__name__}"


def _tensor_descriptor(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
        "numel": tensor.numel(),
    }


def _tensor_sha256(tensor: Any, target_bytes: int = 32 * 1024 * 1024) -> str:
    """Hash a CUDA tensor in logical row-major order with bounded host memory."""
    import torch

    digest = hashlib.sha256()
    if tensor.ndim == 0:
        chunks = (tensor.reshape(1),)
    else:
        row_bytes = max(1, tensor[0].numel() * tensor.element_size())
        rows = max(1, target_bytes // row_bytes)
        chunks = (tensor[start : start + rows] for start in range(0, tensor.shape[0], rows))
    for chunk in chunks:
        raw = chunk.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def _small_tensor_values(tensor: Any, limit: int = 256) -> list[float] | None:
    if tensor.numel() > limit:
        return None
    return tensor.detach().float().cpu().reshape(-1).tolist()


def _linear_inventory(name: str, module: Any, *, weight_digests: bool) -> dict[str, Any]:
    quant_method = getattr(module, "quant_method", None)
    kernel = getattr(quant_method, "fp8_linear", None)
    if kernel is None:
        kernel = getattr(quant_method, "kernel", None)
    record: dict[str, Any] = {
        "name": name,
        "module_class": _class_name(module),
        "quant_method_class": _class_name(quant_method),
        "kernel_class": _class_name(kernel),
        "input_size": getattr(module, "input_size", None),
        "input_size_per_partition": getattr(module, "input_size_per_partition", None),
        "output_size": getattr(module, "output_size", None),
        "output_size_per_partition": getattr(module, "output_size_per_partition", None),
        "logical_widths": list(getattr(module, "logical_widths", ())),
    }
    for parameter_name in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
        tensor = getattr(module, parameter_name, None)
        if tensor is None:
            continue
        descriptor = _tensor_descriptor(tensor)
        values = _small_tensor_values(tensor)
        if values is not None:
            descriptor["values"] = values
        if weight_digests and parameter_name == "weight" and "float8" in str(tensor.dtype):
            descriptor["logical_row_major_sha256"] = _tensor_sha256(tensor)
        record[parameter_name] = descriptor
    return record


def _compilation_record(model: Any) -> dict[str, Any]:
    try:
        language_model, _ = _text_model_parts(model)
        config = language_model.vllm_config.compilation_config
        return {
            "mode": str(config.mode),
            "backend": str(config.backend),
            "cudagraph_mode": str(config.cudagraph_mode),
            "custom_ops": list(config.custom_ops),
            "enabled_custom_ops": sorted(config.enabled_custom_ops),
            "disabled_custom_ops": sorted(config.disabled_custom_ops),
            "static_forward_context_count": len(config.static_forward_context),
        }
    except Exception as exc:  # diagnostic metadata must not abort capture
        return {"error": f"{type(exc).__name__}: {exc}"}


def _inventory_model(
    model: Any, *, weight_digests: bool, target_layers: tuple[int, ...]
) -> dict[str, Any]:
    from vllm.model_executor.layers.linear import LinearBase

    language_model, text_model = _text_model_parts(model)
    if len(text_model.layers) != EXPECTED_LAYER_COUNT:
        raise ValueError(
            f"expected {EXPECTED_LAYER_COUNT} main-model layers, "
            f"found {len(text_model.layers)}"
        )
    layer_types = [layer.layer_type for layer in text_model.layers]
    gdn_layers, full_attention_layers = _partition_target_layers(
        target_layers, layer_types
    )

    linears: list[dict[str, Any]] = []
    for layer_index in target_layers:
        layer = text_model.layers[layer_index]
        for relative_name, module in layer.named_modules():
            if isinstance(module, LinearBase):
                name = f"layers.{layer_index}.{relative_name}".rstrip(".")
                linears.append(
                    _linear_inventory(name, module, weight_digests=weight_digests)
                )

    gdn: dict[str, Any] = {}
    for layer_index in gdn_layers:
        attention = text_model.layers[layer_index].linear_attn
        chunk = attention.chunk_gated_delta_rule
        gdn[str(layer_index)] = {
            "attention_class": _class_name(attention),
            "forward_method": getattr(
                attention._forward_method,
                "__name__",
                repr(attention._forward_method),
            ),
            "prefill_backend": attention.gdn_prefill_backend,
            "chunk_class": _class_name(chunk),
            "chunk_forward_method": getattr(
                chunk._forward_method,
                "__name__",
                repr(chunk._forward_method),
            ),
            "num_k_heads": attention.num_k_heads,
            "num_v_heads": attention.num_v_heads,
            "head_k_dim": attention.head_k_dim,
            "head_v_dim": attention.head_v_dim,
            "conv_kernel_size": attention.conv_kernel_size,
            "conv_weight": _tensor_descriptor(attention.conv1d.weight),
            "A_log": _tensor_descriptor(attention.A_log),
            "dt_bias": _tensor_descriptor(attention.dt_bias),
        }

    full_attention = {
        str(index): {
            "module_class": _class_name(text_model.layers[index].self_attn.attn),
            "backend_impl_class": _class_name(
                text_model.layers[index].self_attn.attn.impl
            ),
            "kv_cache_dtype": text_model.layers[index].self_attn.attn.kv_cache_dtype,
        }
        for index in full_attention_layers
    }
    inventory = {
        "root_class": _class_name(model),
        "language_model_class": _class_name(language_model),
        "text_model_class": _class_name(text_model),
        "layer_count": len(text_model.layers),
        "target_layer_types": {
            str(index): text_model.layers[index].layer_type for index in target_layers
        },
        "compilation": _compilation_record(model),
        "linears": linears,
        "gdn": gdn,
        "full_attention": full_attention,
    }
    if "63" in full_attention:
        inventory["layer63_attention"] = full_attention["63"]
    return inventory


def _normalize_layer_prefix(prefix: str) -> str:
    marker = "layers."
    position = prefix.find(marker)
    return prefix[position:] if position >= 0 else prefix


def _is_target_prefix(prefix: str, target_layers: tuple[int, ...]) -> bool:
    normalized = _normalize_layer_prefix(prefix)
    return any(normalized.startswith(f"layers.{index}.") for index in target_layers)


def _install_compiled_observer_patches(
    capacity: int, target_layers: tuple[int, ...]
) -> dict[str, Any]:
    """Install output-only observers before model construction.

    The opaque custom op is part of the compiled graph and mutates only its
    dedicated destination buffers. It observes outputs after the original op;
    it never substitutes a value used by model inference.
    """
    import torch
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        LinearBase,
        ReplicatedLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
    from vllm.platforms import current_platform

    @torch.library.custom_op(
        "jlens_nvfp4::capture_output",
        mutates_args=("destination", "row_count"),
        schema="(Tensor source, Tensor(a!) destination, Tensor(b!) row_count) -> ()",
    )
    def capture_output(
        source: torch.Tensor,
        destination: torch.Tensor,
        row_count: torch.Tensor,
    ) -> None:
        source_2d = source.reshape(-1, source.shape[-1])
        copied = min(source_2d.shape[0], destination.shape[0])
        destination[:copied].copy_(source_2d[:copied])
        row_count.fill_(source_2d.shape[0])

    @capture_output.register_fake
    def _capture_output_fake(
        source: torch.Tensor,
        destination: torch.Tensor,
        row_count: torch.Tensor,
    ) -> None:
        return None

    original_linear_init = LinearBase.__init__

    @functools.wraps(original_linear_init)
    def observer_linear_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_linear_init(self, *args, **kwargs)
        if not _is_target_prefix(self.prefix, target_layers):
            return
        self.register_buffer(
            "_jlens_observer_output",
            torch.full(
                (capacity, int(self.output_size)),
                float("nan"),
                dtype=self.params_dtype,
                device=current_platform.current_device(),
            ),
            persistent=False,
        )
        self.register_buffer(
            "_jlens_observer_count",
            torch.zeros(
                (), dtype=torch.int32, device=current_platform.current_device()
            ),
            persistent=False,
        )
        self._jlens_observer_name = f"linear.{_normalize_layer_prefix(self.prefix)}.output"

    LinearBase.__init__ = observer_linear_init

    patched_linear_classes: list[str] = []
    for linear_class in (
        LinearBase,
        ReplicatedLinear,
        ColumnParallelLinear,
        RowParallelLinear,
    ):
        original_forward = linear_class.forward

        @functools.wraps(original_forward)
        def observer_linear_forward(
            self: Any,
            *args: Any,
            __original: Any = original_forward,
            **kwargs: Any,
        ) -> Any:
            result = __original(self, *args, **kwargs)
            if hasattr(self, "_jlens_observer_output"):
                output = result[0] if isinstance(result, tuple) else result
                capture_output(
                    output,
                    self._jlens_observer_output,
                    self._jlens_observer_count,
                )
            return result

        linear_class.forward = observer_linear_forward
        patched_linear_classes.append(linear_class.__name__)

    original_mlp_init = Qwen2MoeMLP.__init__

    @functools.wraps(original_mlp_init)
    def observer_mlp_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_mlp_init(self, *args, **kwargs)
        prefix = kwargs.get("prefix", "")
        if not _is_target_prefix(prefix, target_layers):
            return
        width = int(self.down_proj.input_size_per_partition)
        self.act_fn.register_buffer(
            "_jlens_observer_output",
            torch.full(
                (capacity, width),
                float("nan"),
                dtype=self.down_proj.params_dtype,
                device=current_platform.current_device(),
            ),
            persistent=False,
        )
        self.act_fn.register_buffer(
            "_jlens_observer_count",
            torch.zeros(
                (), dtype=torch.int32, device=current_platform.current_device()
            ),
            persistent=False,
        )
        self.act_fn._jlens_observer_name = (
            f"{_normalize_layer_prefix(prefix)}.swiglu_output"
        )

    Qwen2MoeMLP.__init__ = observer_mlp_init
    original_swiglu_forward = SiluAndMul.forward

    @functools.wraps(original_swiglu_forward)
    def observer_swiglu_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_swiglu_forward(self, *args, **kwargs)
        if hasattr(self, "_jlens_observer_output"):
            capture_output(
                result,
                self._jlens_observer_output,
                self._jlens_observer_count,
            )
        return result

    SiluAndMul.forward = observer_swiglu_forward

    original_decoder_init = Qwen3_5DecoderLayer.__init__

    @functools.wraps(original_decoder_init)
    def observer_decoder_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_decoder_init(self, *args, **kwargs)
        if self.layer_idx not in target_layers:
            return
        config = args[0] if args else kwargs["vllm_config"]
        dtype = config.model_config.dtype
        width = config.model_config.hf_text_config.hidden_size
        self.register_buffer(
            "_jlens_observer_post_block",
            torch.full(
                (capacity, width),
                float("nan"),
                dtype=dtype,
                device=current_platform.current_device(),
            ),
            persistent=False,
        )
        self.register_buffer(
            "_jlens_observer_post_block_count",
            torch.zeros(
                (), dtype=torch.int32, device=current_platform.current_device()
            ),
            persistent=False,
        )

    Qwen3_5DecoderLayer.__init__ = observer_decoder_init
    original_decoder_forward = Qwen3_5DecoderLayer.forward

    @functools.wraps(original_decoder_forward)
    def observer_decoder_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_decoder_forward(self, *args, **kwargs)
        if hasattr(self, "_jlens_observer_post_block"):
            branch, residual = result
            capture_output(
                branch + residual,
                self._jlens_observer_post_block,
                self._jlens_observer_post_block_count,
            )
        return result

    Qwen3_5DecoderLayer.forward = observer_decoder_forward
    return {
        "capacity": capacity,
        "target_layers": list(target_layers),
        "custom_op": "jlens_nvfp4::capture_output",
        "patched_linear_classes": patched_linear_classes,
        "post_output_only": True,
    }


def _allocate_record(state: dict[str, Any], name: str, shape: tuple[int, ...], dtype: Any) -> None:
    import torch

    device = state["device"]
    tensor = torch.empty(shape, dtype=dtype, device=device)
    if tensor.is_floating_point():
        tensor.fill_(float("nan"))
    else:
        tensor.zero_()
    state["buffers"][name] = tensor
    state["row_counts"][name] = torch.zeros((), dtype=torch.int32, device=device)
    state["token_axes"][name] = 0


def _allocate_token_axis_record(
    state: dict[str, Any],
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
    *,
    token_axis: int,
) -> None:
    _allocate_record(state, name, shape, dtype)
    state["token_axes"][name] = token_axis


def _record_tensor(state: dict[str, Any], name: str, source: Any) -> None:
    """Copy a runtime tensor into a fixed CUDA buffer; suitable for Dynamo tracing."""
    destination = state["buffers"][name]
    axis = state["token_axes"][name]
    count = source.shape[axis]
    capacity = destination.shape[axis]
    copied = min(count, capacity)
    destination_slice = [slice(None)] * destination.ndim
    source_slice = [slice(None)] * source.ndim
    destination_slice[axis] = slice(0, copied)
    source_slice[axis] = slice(0, copied)
    destination[tuple(destination_slice)].copy_(source.detach()[tuple(source_slice)])
    state["row_counts"][name].fill_(count)


def _install_capture(
    model: Any, *, capacity: int, target_layers: tuple[int, ...]
) -> dict[str, Any]:
    import torch
    from vllm.model_executor.layers.linear import LinearBase

    _, text_model = _text_model_parts(model)
    layer_types = [layer.layer_type for layer in text_model.layers]
    gdn_layers, full_attention_layers = _partition_target_layers(
        target_layers, layer_types
    )
    device = next(text_model.parameters()).device
    dtype = next(text_model.parameters()).dtype
    hidden_size = int(text_model.norm.weight.numel())
    state: dict[str, Any] = {
        "device": device,
        "dtype": dtype,
        "capacity": capacity,
        "target_layers": target_layers,
        "gdn_layers": gdn_layers,
        "full_attention_layers": full_attention_layers,
        "buffers": {},
        "row_counts": {},
        "token_axes": {},
        "handles": [],
        "wrapped_core": {},
        "wrapped_attention_impls": {},
        "linear_names": [],
    }
    model._nvfp4_runtime_capture = state

    for layer_index in target_layers:
        layer = text_model.layers[layer_index]
        residual_name = f"h{layer_index}_post_block"
        _allocate_record(state, residual_name, (capacity, hidden_size), dtype)

        def layer_hook(_module: Any, _inputs: Any, output: Any, *, name: str = residual_name) -> None:
            branch, residual = output
            _record_tensor(state, name, branch + residual)

        state["handles"].append(layer.register_forward_hook(layer_hook))

        for relative_name, module in layer.named_modules():
            if not isinstance(module, LinearBase):
                continue
            full_name = f"layers.{layer_index}.{relative_name}".rstrip(".")
            # conv1d is used as a raw depthwise weight, never through LinearBase.forward.
            if relative_name.endswith("conv1d"):
                continue
            input_width = int(getattr(module, "input_size_per_partition", module.input_size))
            output_width = int(
                getattr(module, "output_size_per_partition", module.output_size)
            )
            input_name = f"linear.{full_name}.input"
            output_name = f"linear.{full_name}.output"
            _allocate_record(state, input_name, (capacity, input_width), dtype)
            _allocate_record(state, output_name, (capacity, output_width), dtype)
            state["linear_names"].append(full_name)

            def linear_hook(
                _module: Any,
                inputs: Any,
                output: Any,
                *,
                x_name: str = input_name,
                y_name: str = output_name,
            ) -> None:
                x = inputs[0].reshape(-1, inputs[0].shape[-1])
                y = output[0] if isinstance(output, tuple) else output
                y = y.reshape(-1, y.shape[-1])
                _record_tensor(state, x_name, x)
                _record_tensor(state, y_name, y)

            state["handles"].append(module.register_forward_hook(linear_hook))

        swiglu_name = f"layers.{layer_index}.mlp.swiglu_output"
        swiglu_width = int(layer.mlp.down_proj.input_size_per_partition)
        _allocate_record(state, swiglu_name, (capacity, swiglu_width), dtype)

        def swiglu_hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            __name: str = swiglu_name,
        ) -> None:
            _record_tensor(state, __name, output.reshape(-1, output.shape[-1]))

        state["handles"].append(layer.mlp.act_fn.register_forward_hook(swiglu_hook))

    for layer_index in full_attention_layers:
        full_attention = text_model.layers[layer_index].self_attn
        attention_prefix = f"attention.layer{layer_index}"
        for suffix, width in (
            ("q_post_rope", full_attention.q_size),
            ("k_post_rope", full_attention.kv_size),
            ("v", full_attention.kv_size),
            ("core_output", full_attention.q_size),
        ):
            _allocate_record(
                state,
                f"{attention_prefix}.{suffix}",
                (capacity, int(width)),
                dtype,
            )

        def attention_pre_hook(
            _module: Any,
            args: Any,
            kwargs: dict[str, Any],
            *,
            __prefix: str = attention_prefix,
        ) -> None:
            query = args[0] if len(args) > 0 else kwargs["query"]
            key = args[1] if len(args) > 1 else kwargs["key"]
            value = args[2] if len(args) > 2 else kwargs["value"]
            _record_tensor(state, f"{__prefix}.q_post_rope", query)
            _record_tensor(state, f"{__prefix}.k_post_rope", key)
            _record_tensor(state, f"{__prefix}.v", value)

        def attention_hook(
            _module: Any,
            _args: Any,
            _kwargs: dict[str, Any],
            output: Any,
            *,
            __prefix: str = attention_prefix,
        ) -> None:
            _record_tensor(state, f"{__prefix}.core_output", output)

        state["handles"].append(
            full_attention.attn.register_forward_pre_hook(
                attention_pre_hook, with_kwargs=True
            )
        )
        state["handles"].append(
            full_attention.attn.register_forward_hook(
                attention_hook, with_kwargs=True
            )
        )

        # Compiled execution bypasses Attention.forward and calls the split
        # custom op, whose backend object is still resolved at runtime.
        original_attention_impl_forward = full_attention.attn.impl.forward

        @functools.wraps(original_attention_impl_forward)
        def wrapped_attention_impl_forward(
            *args: Any,
            __original: Any = original_attention_impl_forward,
            __prefix: str = attention_prefix,
            **kwargs: Any,
        ) -> Any:
            query = args[1] if len(args) > 1 else kwargs["query"]
            key = args[2] if len(args) > 2 else kwargs["key"]
            value = args[3] if len(args) > 3 else kwargs["value"]
            output = kwargs["output"]
            _record_tensor(
                state,
                f"{__prefix}.q_post_rope",
                query.reshape(query.shape[0], -1),
            )
            _record_tensor(
                state,
                f"{__prefix}.k_post_rope",
                key.reshape(key.shape[0], -1),
            )
            _record_tensor(
                state,
                f"{__prefix}.v",
                value.reshape(value.shape[0], -1),
            )
            result = __original(*args, **kwargs)
            _record_tensor(
                state,
                f"{__prefix}.core_output",
                output.reshape(output.shape[0], -1),
            )
            return result

        full_attention.attn.impl.forward = wrapped_attention_impl_forward
        state["wrapped_attention_impls"][layer_index] = (
            full_attention.attn,
            original_attention_impl_forward,
        )

    for layer_index in gdn_layers:
        attention = text_model.layers[layer_index].linear_attn
        heads_k = attention.num_k_heads // attention.tp_size
        heads_v = attention.num_v_heads // attention.tp_size
        qkv_width = heads_k * attention.head_k_dim * 2 + heads_v * attention.head_v_dim
        prefix = f"gdn.layer{layer_index}"

        for suffix, shape, tensor_dtype in (
            ("mixed_qkv", (capacity, qkv_width), dtype),
            ("conv_output_prefill", (capacity, qkv_width), dtype),
            ("b", (capacity, heads_v), dtype),
            ("a", (capacity, heads_v), dtype),
            ("core_output", (capacity, heads_v, attention.head_v_dim), dtype),
            ("norm_core_input", (capacity * heads_v, attention.head_v_dim), dtype),
            ("norm_z_input", (capacity * heads_v, attention.head_v_dim), dtype),
            ("norm_output", (capacity * heads_v, attention.head_v_dim), dtype),
        ):
            _allocate_record(state, f"{prefix}.{suffix}", shape, tensor_dtype)

        for suffix, shape, tensor_dtype, token_axis in (
            ("q", (1, capacity, heads_k, attention.head_k_dim), dtype, 1),
            ("k", (1, capacity, heads_k, attention.head_k_dim), dtype, 1),
            ("v", (1, capacity, heads_v, attention.head_v_dim), dtype, 1),
            ("log_g", (1, capacity, heads_v), torch.float32, 1),
            ("beta", (1, capacity, heads_v), torch.float32, 1),
            ("chunk_output", (1, capacity, heads_v, attention.head_v_dim), dtype, 1),
        ):
            _allocate_token_axis_record(
                state,
                f"{prefix}.{suffix}",
                shape,
                tensor_dtype,
                token_axis=token_axis,
            )

        state_shape = (1, heads_v, attention.head_v_dim, attention.head_k_dim)
        _allocate_record(state, f"{prefix}.initial_state", state_shape, torch.float32)
        _allocate_record(state, f"{prefix}.final_state", state_shape, torch.float32)
        state["token_axes"][f"{prefix}.initial_state"] = -1
        state["token_axes"][f"{prefix}.final_state"] = -1

        original_core = attention._forward_core

        @functools.wraps(original_core)
        def wrapped_core(*args: Any, __prefix: str = prefix, __original: Any = original_core, **kwargs: Any) -> Any:
            import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_module

            mixed_qkv = kwargs.get("mixed_qkv", args[0] if args else None)
            b = kwargs.get("b", args[1] if len(args) > 1 else None)
            a = kwargs.get("a", args[2] if len(args) > 2 else None)
            core = kwargs.get("core_attn_out", args[3] if len(args) > 3 else None)
            _record_tensor(state, f"{__prefix}.mixed_qkv", mixed_qkv)
            _record_tensor(state, f"{__prefix}.b", b)
            _record_tensor(state, f"{__prefix}.a", a)
            original_fused_post_conv_prep = gdn_module.fused_post_conv_prep

            @functools.wraps(original_fused_post_conv_prep)
            def capture_fused_post_conv_prep(*fused_args: Any, **fused_kwargs: Any) -> Any:
                conv_output = fused_kwargs.get(
                    "conv_output", fused_args[0] if fused_args else None
                )
                _record_tensor(
                    state, f"{__prefix}.conv_output_prefill", conv_output
                )
                return original_fused_post_conv_prep(*fused_args, **fused_kwargs)

            gdn_module.fused_post_conv_prep = capture_fused_post_conv_prep
            try:
                result = __original(*args, **kwargs)
            finally:
                gdn_module.fused_post_conv_prep = original_fused_post_conv_prep
            _record_tensor(state, f"{__prefix}.core_output", core)
            return result

        attention._forward_core = wrapped_core
        state["wrapped_core"][layer_index] = original_core

        def chunk_pre_hook(
            _module: Any,
            _args: Any,
            kwargs: dict[str, Any],
            *,
            __prefix: str = prefix,
        ) -> None:
            for argument, suffix in (
                ("q", "q"),
                ("k", "k"),
                ("v", "v"),
                ("g", "log_g"),
                ("beta", "beta"),
                ("initial_state", "initial_state"),
            ):
                _record_tensor(state, f"{__prefix}.{suffix}", kwargs[argument])

        def chunk_hook(
            _module: Any,
            _args: Any,
            _kwargs: dict[str, Any],
            output: Any,
            *,
            __prefix: str = prefix,
        ) -> None:
            chunk_output, final_state = output
            _record_tensor(state, f"{__prefix}.chunk_output", chunk_output)
            _record_tensor(state, f"{__prefix}.final_state", final_state)

        state["handles"].append(
            attention.chunk_gated_delta_rule.register_forward_pre_hook(
                chunk_pre_hook, with_kwargs=True
            )
        )
        state["handles"].append(
            attention.chunk_gated_delta_rule.register_forward_hook(
                chunk_hook, with_kwargs=True
            )
        )

        def norm_pre_hook(
            _module: Any,
            args: Any,
            *,
            __prefix: str = prefix,
        ) -> None:
            _record_tensor(state, f"{__prefix}.norm_core_input", args[0])
            _record_tensor(state, f"{__prefix}.norm_z_input", args[1])

        def norm_hook(
            _module: Any,
            _args: Any,
            output: Any,
            *,
            __prefix: str = prefix,
        ) -> None:
            _record_tensor(state, f"{__prefix}.norm_output", output)

        state["handles"].append(attention.norm.register_forward_pre_hook(norm_pre_hook))
        state["handles"].append(attention.norm.register_forward_hook(norm_hook))

    return {
        "capture_capacity": capacity,
        "target_layers": list(target_layers),
        "gdn_layers": list(gdn_layers),
        "full_attention_layers": list(full_attention_layers),
        "allocated_tensor_count": len(state["buffers"]),
        "linear_boundary_count": len(state["linear_names"]),
    }


def _collect_capture(model: Any) -> dict[str, Any]:
    state = model._nvfp4_runtime_capture
    tensors: dict[str, Any] = {}
    rows: dict[str, int] = {}
    truncated: list[str] = []
    missing: list[str] = []
    for name, buffer in state["buffers"].items():
        axis = state["token_axes"][name]
        raw_count = int(state["row_counts"][name].item())
        rows[name] = raw_count
        if raw_count == 0:
            missing.append(name)
            continue
        if axis == -1:
            tensors[name] = buffer.detach().cpu()
            continue
        count = min(raw_count, buffer.shape[axis])
        if raw_count > buffer.shape[axis]:
            truncated.append(name)
        slices = [slice(None)] * buffer.ndim
        slices[axis] = slice(0, count)
        tensors[name] = buffer[tuple(slices)].detach().cpu()

    # Compile-visible observer buffers are registered before model construction
    # in compiled-observer mode. They are independent of ordinary module hooks.
    _, text_model = _text_model_parts(model)
    observer_names: list[str] = []
    for layer_index in state["target_layers"]:
        layer = text_model.layers[layer_index]
        if hasattr(layer, "_jlens_observer_post_block"):
            count = int(layer._jlens_observer_post_block_count.item())
            name = f"h{layer_index}_post_block"
            if count > 0:
                tensors[name] = layer._jlens_observer_post_block[
                    : min(count, layer._jlens_observer_post_block.shape[0])
                ].detach().cpu()
                rows[name] = count
                observer_names.append(name)
                if count > layer._jlens_observer_post_block.shape[0]:
                    truncated.append(name)
        for module in layer.modules():
            name = getattr(module, "_jlens_observer_name", None)
            if name is None:
                continue
            count = int(module._jlens_observer_count.item())
            if count <= 0:
                continue
            tensors[name] = module._jlens_observer_output[
                : min(count, module._jlens_observer_output.shape[0])
            ].detach().cpu()
            rows[name] = count
            observer_names.append(name)
            if count > module._jlens_observer_output.shape[0]:
                truncated.append(name)

    # Saturation is computed from exact captured linear inputs and post-load scales.
    saturation: dict[str, Any] = {}
    for layer_index in state["target_layers"]:
        layer = text_model.layers[layer_index]
        for relative_name, module in layer.named_modules():
            full_name = f"layers.{layer_index}.{relative_name}".rstrip(".")
            input_name = f"linear.{full_name}.input"
            if input_name not in tensors or not hasattr(module, "input_scale"):
                continue
            scale_tensor = getattr(module, "input_scale", None)
            if scale_tensor is None or scale_tensor.numel() != 1:
                continue
            x = tensors[input_name].float()
            scale = float(scale_tensor.detach().float().cpu().item())
            limit = FP8_MAX * scale
            finite = x.isfinite()
            saturated = finite & (x.abs() >= limit)
            saturation[full_name] = {
                "input_scale": scale,
                "finite_values": int(finite.sum().item()),
                "nonfinite_values": int((~finite).sum().item()),
                "at_or_above_fp8_limit": int(saturated.sum().item()),
                "fraction_at_or_above_fp8_limit": float(
                    saturated.sum().item() / max(1, finite.sum().item())
                ),
                "max_abs": float(x[finite].abs().max().item()) if finite.any() else None,
                "fp8_limit": limit,
            }

    replay_tensors, replay_provenance = _collect_replay_parameters(
        model, target_layers=state["target_layers"]
    )
    tensors.update(replay_tensors)
    missing = [name for name in missing if name not in tensors]
    return {
        "tensors": tensors,
        "row_counts": rows,
        "missing": sorted(missing),
        "truncated": sorted(truncated),
        "fp8_activation_saturation": saturation,
        "compile_visible_observer_tensors": sorted(observer_names),
        "replay_parameter_tensor_count": len(replay_provenance),
        "replay_parameter_provenance": replay_provenance,
    }


def _collect_replay_parameters(
    model: Any, *, target_layers: tuple[int, ...]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy live deployed attention and norm parameters for target replay."""
    import torch

    _, text_model = _text_model_parts(model)
    layer_types = [layer.layer_type for layer in text_model.layers]
    gdn_layers, full_attention_layers = _partition_target_layers(
        target_layers, layer_types
    )
    tensors: dict[str, Any] = {}
    provenance: dict[str, Any] = {}

    def add(name: str, tensor: Any, *, role: str, orientation: str) -> None:
        key = f"replay.{name}"
        value = tensor.detach().contiguous().cpu()
        tensors[key] = value
        provenance[key] = {
            "role": role,
            "orientation": orientation,
            "source": "live post-load vLLM parameter",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    fp8_modules: list[tuple[str, Any]] = []
    for layer_index in gdn_layers:
        attention = text_model.layers[layer_index].linear_attn
        fp8_modules.extend(
            (
                (f"layers.{layer_index}.linear_attn.in_proj_qkvz", attention.in_proj_qkvz),
                (f"layers.{layer_index}.linear_attn.out_proj", attention.out_proj),
            )
        )
    for layer_index in full_attention_layers:
        full_attention = text_model.layers[layer_index].self_attn
        fp8_modules.extend(
            (
                (
                    f"layers.{layer_index}.self_attn.qkv_proj",
                    full_attention.qkv_proj,
                ),
                (f"layers.{layer_index}.self_attn.o_proj", full_attention.o_proj),
            )
        )
    for name, module in fp8_modules:
        if module.weight.dtype != torch.float8_e4m3fn:
            raise TypeError(f"{name} is not live FP8: {module.weight.dtype}")
        add(
            f"fp8.{name}.weight",
            module.weight,
            role="post-load requantized FP8 weight",
            orientation="[input_features, output_features] used by Cutlass scaled_mm",
        )
        for scale_name in ("weight_scale", "input_scale"):
            add(
                f"fp8.{name}.{scale_name}",
                getattr(module, scale_name),
                role=f"post-load scalar {scale_name}",
                orientation="scalar multiplicative scale",
            )

    for layer_index in target_layers:
        layer = text_model.layers[layer_index]
        for norm_name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(layer, norm_name)
            add(
                f"norm.layers.{layer_index}.{norm_name}.weight",
                norm.weight,
                role="RMSNorm weight",
                orientation="[hidden_size]",
            )
    add(
        "norm.final.weight",
        text_model.norm.weight,
        role="final RMSNorm weight",
        orientation="[hidden_size]",
    )

    for layer_index in gdn_layers:
        attention = text_model.layers[layer_index].linear_attn
        prefix = f"gdn.layers.{layer_index}"
        for name, tensor, role, orientation in (
            ("in_proj_ba.weight", attention.in_proj_ba.weight, "unquantized b/a projection", "[output_features, input_features]"),
            ("conv1d.weight", attention.conv1d.weight, "causal depthwise convolution", "[channels, 1, kernel_width]"),
            ("A_log", attention.A_log, "GDN log decay parameter", "[value_heads]"),
            ("dt_bias", attention.dt_bias, "GDN time-step bias", "[value_heads]"),
            ("norm.weight", attention.norm.weight, "GDN gated RMSNorm weight", "[value_head_dim]"),
        ):
            add(f"{prefix}.{name}", tensor, role=role, orientation=orientation)

    for layer_index in full_attention_layers:
        full_attention = text_model.layers[layer_index].self_attn
        for name, norm in (
            ("q_norm", full_attention.q_norm),
            ("k_norm", full_attention.k_norm),
        ):
            add(
                f"norm.layers.{layer_index}.self_attn.{name}.weight",
                norm.weight,
                role=f"layer {layer_index} {name} RMSNorm weight",
                orientation="[attention_head_dim]",
            )
    return tensors, provenance


def _remove_capture(model: Any) -> None:
    _, text_model = _text_model_parts(model)
    state = getattr(model, "_nvfp4_runtime_capture", None)
    if state is None:
        return
    for handle in state["handles"]:
        handle.remove()
    for layer_index, original in state["wrapped_core"].items():
        text_model.layers[layer_index].linear_attn._forward_core = original
    for attention, original in state["wrapped_attention_impls"].values():
        attention.impl.forward = original
    delattr(model, "_nvfp4_runtime_capture")


def _dynamo_counters() -> dict[str, dict[str, int]]:
    try:
        from torch._dynamo.utils import counters

        return {
            category: {str(key): int(value) for key, value in values.items()}
            for category, values in counters.items()
            if values
        }
    except Exception as exc:
        return {"error": {"message": f"{type(exc).__name__}: {exc}"}}  # type: ignore[dict-item]


def _counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for category in sorted(set(before) | set(after)):
        if category == "error":
            continue
        old_values = before.get(category, {})
        new_values = after.get(category, {})
        changed = {
            key: new_values.get(key, 0) - old_values.get(key, 0)
            for key in sorted(set(old_values) | set(new_values))
            if new_values.get(key, 0) != old_values.get(key, 0)
        }
        if changed:
            delta[category] = changed
    return delta


def _generation_record(output: Any) -> dict[str, Any]:
    completion = output.outputs[0]
    candidates: list[dict[str, Any]] = []
    if completion.logprobs:
        for token_id, value in completion.logprobs[0].items():
            candidates.append(
                {
                    "token_id": int(token_id),
                    "logprob": float(value.logprob),
                    "rank": value.rank,
                    "decoded_token": value.decoded_token,
                }
            )
        candidates.sort(key=lambda item: item["logprob"], reverse=True)
    return {
        "prompt_token_ids": list(output.prompt_token_ids),
        "generated_token_id": int(completion.token_ids[0]),
        "generated_text": completion.text,
        "finish_reason": completion.finish_reason,
        "top_logprobs": candidates,
    }


def _generation_parity(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_scores = {item["token_id"]: item["logprob"] for item in first["top_logprobs"]}
    second_scores = {item["token_id"]: item["logprob"] for item in second["top_logprobs"]}
    shared = sorted(set(first_scores) & set(second_scores))
    deltas = {str(token): second_scores[token] - first_scores[token] for token in shared}
    return {
        "generated_token_equal": first["generated_token_id"] == second["generated_token_id"],
        "shared_top_logprob_tokens": len(shared),
        "max_abs_shared_logprob_delta": max((abs(value) for value in deltas.values()), default=None),
        "logprob_deltas": deltas,
    }


def _summarize_tensor(tensor: Any) -> dict[str, Any]:
    import torch

    raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    return {
        **_tensor_descriptor(tensor),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": int(finite.sum().item()),
        "nonfinite": int((~finite).sum().item()),
        "min": float(values[finite].min().item()) if finite.any() else None,
        "max": float(values[finite].max().item()) if finite.any() else None,
        "rms": float(torch.sqrt(torch.mean(values[finite].square())).item()) if finite.any() else None,
    }


def _required_capture_names(
    target_layers: tuple[int, ...], layer_types: list[str] | tuple[str, ...]
) -> set[str]:
    gdn_layers, full_attention_layers = _partition_target_layers(
        target_layers, layer_types
    )
    names = {f"h{index}_post_block" for index in target_layers}
    for index in gdn_layers:
        prefix = f"gdn.layer{index}"
        names.update(
            f"{prefix}.{suffix}"
            for suffix in (
                "q",
                "k",
                "v",
                "log_g",
                "beta",
                "initial_state",
                "chunk_output",
                "final_state",
                "core_output",
                "conv_output_prefill",
            )
        )
    for index in full_attention_layers:
        names.update(
            f"attention.layer{index}.{suffix}"
            for suffix in ("q_post_rope", "k_post_rope", "v", "core_output")
        )
    names.update(
        f"layers.{index}.mlp.swiglu_output" for index in target_layers
    )
    return names


def _download_pinned_model_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)
    ).resolve()


def _metadata_identity(snapshot: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for filename, expected_sha256 in PINNED_METADATA_SHA256.items():
        metadata_path = snapshot / filename
        if not metadata_path.is_file():
            raise FileNotFoundError(f"required pinned metadata is missing: {metadata_path}")
        digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(
                f"pinned metadata SHA-256 mismatch for {filename}: "
                f"{digest} != {expected_sha256}"
            )
        actual[filename] = digest
    return actual


def _validate_pinned_checkpoint(snapshot: Path) -> dict[str, Any]:
    metadata_sha256 = _metadata_identity(snapshot)
    try:
        from scripts.modelopt_checkpoint import (
            ModelOptCheckpoint,
            PINNED_METADATA_SHA256 as CHECKPOINT_METADATA_SHA256,
            PINNED_REVISION as CHECKPOINT_REVISION,
        )
    except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
        from modelopt_checkpoint import (  # type: ignore[no-redef]
            ModelOptCheckpoint,
            PINNED_METADATA_SHA256 as CHECKPOINT_METADATA_SHA256,
            PINNED_REVISION as CHECKPOINT_REVISION,
        )

    if CHECKPOINT_REVISION != MODEL_REVISION:
        raise RuntimeError("runtime capture and ModelOpt checkpoint revisions diverged")
    if dict(CHECKPOINT_METADATA_SHA256) != PINNED_METADATA_SHA256:
        raise RuntimeError("runtime capture and ModelOpt metadata pins diverged")
    ModelOptCheckpoint(snapshot, strict_pinned=True)
    return {
        "policy": MODEL_IDENTITY_POLICY,
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "resolved_path": str(snapshot),
        "metadata_sha256": metadata_sha256,
        "strict_pinned_validation": True,
        "validator": "ModelOptCheckpoint(strict_pinned=True)",
    }


def _resolve_model_path(path: Path | None) -> tuple[Path, dict[str, Any]]:
    pinned_snapshot = _download_pinned_model_snapshot()
    candidate = pinned_snapshot if path is None else path.expanduser().resolve()
    if candidate != pinned_snapshot:
        raise ValueError(
            "--model-path must resolve to the exact pinned NVIDIA snapshot: "
            f"{candidate} != {pinned_snapshot}"
        )
    return candidate, _validate_pinned_checkpoint(candidate)


def _total_gpu_bytes(override_gib: float | None = None) -> int:
    if override_gib is not None:
        if override_gib <= 0:
            raise ValueError("total GPU memory override must be positive")
        return int(override_gib * 1024**3)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; pass --total-gpu-memory-gib for offline preflight"
        )
    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)


def _build_preflight(
    args: argparse.Namespace,
    *,
    model_path: Path,
    model_identity: dict[str, Any],
    text_config: dict[str, Any],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    layer_types = list(text_config["layer_types"])
    if len(layer_types) != EXPECTED_LAYER_COUNT:
        raise ValueError(
            f"expected {EXPECTED_LAYER_COUNT} main-model layers, "
            f"found {len(layer_types)} in config"
        )
    target_layers = _resolve_target_layers(args.target_layers, len(layer_types))
    estimate = _estimate_capture_memory(
        text_config=text_config,
        target_layers=target_layers,
        capture_capacity=args.capture_capacity,
        gpu_memory_utilization=args.gpu_memory_utilization,
        total_gpu_bytes=_total_gpu_bytes(args.total_gpu_memory_gib),
    )
    estimate.update(
        {
            "model_path": str(model_path),
            "model_revision": MODEL_REVISION,
            "model_identity": model_identity,
            "target_profile": args.target_layers,
            "target_layers": list(target_layers),
            "layer_types": {
                str(index): layer_types[index] for index in target_layers
            },
        }
    )
    return target_layers, estimate


def _run_preflight_only(args: argparse.Namespace) -> dict[str, Any]:
    model_path, model_identity = _resolve_model_path(args.model_path)
    text_config = _load_text_config(model_path)
    _, estimate = _build_preflight(
        args,
        model_path=model_path,
        model_identity=model_identity,
        text_config=text_config,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "safe" if estimate["safe"] else "unsafe",
        "mode": "preflight-only",
        "memory_preflight": estimate,
        "run_would_require_override": (
            not estimate["safe"] and not args.allow_unsafe_capture_memory
        ),
    }


def _run_single(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError("VLLM_ENABLE_V1_MULTIPROCESSING must be 0")
    if args.capture_capacity < 1:
        raise ValueError("capture capacity must be positive")
    if args.max_num_batched_tokens < 1568:
        raise ValueError(
            "Qwen3.6 Mamba align mode requires max_num_batched_tokens >= 1568 "
            "because its aligned attention/cache block is 1568 tokens"
        )

    model_path, model_identity = _resolve_model_path(args.model_path)
    text_config = _load_text_config(model_path)
    target_layers, memory_preflight = _build_preflight(
        args,
        model_path=model_path,
        model_identity=model_identity,
        text_config=text_config,
    )
    if not memory_preflight["safe"] and not args.allow_unsafe_capture_memory:
        raise MemoryError(
            "capture preflight exceeds GPU memory: estimated peak "
            f"{memory_preflight['estimated_peak_gpu_gib']:.2f} GiB > "
            f"{memory_preflight['total_gpu_gib']:.2f} GiB; reduce "
            "--capture-capacity/--gpu-memory-utilization or pass "
            "--allow-unsafe-capture-memory"
        )

    import torch
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt

    from transformers import AutoTokenizer

    tokenizer_for_count = AutoTokenizer.from_pretrained(
        model_path, revision=MODEL_REVISION, local_files_only=True
    )
    prompt_provenance = None
    prompt_text = args.prompt
    if args.prompt_manifest is not None:
        prompt_provenance = _load_frozen_prompt(
            args.prompt_manifest.resolve(), args.prompt_index
        )
        prompt_text = prompt_provenance["text"]
        prompt_token_ids = list(prompt_provenance["token_ids"])
        recomputed = tokenizer_for_count(
            prompt_text,
            add_special_tokens=True,
            truncation=True,
            max_length=len(prompt_token_ids),
            return_attention_mask=False,
        ).input_ids
        if list(recomputed) != prompt_token_ids:
            raise ValueError(
                "frozen prompt token IDs do not match the pinned NVFP4 tokenizer"
            )
    else:
        prompt_token_ids = tokenizer_for_count.encode(
            prompt_text, add_special_tokens=True
        )
    if len(prompt_token_ids) > args.capture_capacity:
        raise ValueError(
            f"prompt has {len(prompt_token_ids)} tokens but capture capacity is "
            f"{args.capture_capacity}"
        )

    observer_patch = None
    if args.mode == "compiled-observer":
        observer_patch = _install_compiled_observer_patches(
            args.capture_capacity, target_layers
        )

    started = datetime.now(timezone.utc)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        dtype="bfloat16",
        quantization="modelopt_fp4",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.mode == "eager",
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        language_model_only=True,
        gdn_prefill_backend="triton",
        mamba_cache_mode="align",
        mamba_block_size=1024,
        mamba_ssm_cache_dtype="float32",
        attention_backend="TRITON_ATTN",
        limit_mm_per_prompt={"image": 0, "video": 0},
        enable_flashinfer_autotune=False,
        async_scheduling=False,
        seed=0,
    )
    load_seconds = time.perf_counter() - load_started
    inventory = llm.apply_model(
        functools.partial(
            _inventory_model,
            weight_digests=not args.no_weight_digests,
            target_layers=target_layers,
        )
    )[0]

    sampling = SamplingParams(max_tokens=1, temperature=0, seed=0, logprobs=20)
    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids, prompt=prompt_text)
    authoritative_generation = None
    baseline_counters = None
    if args.mode == "compiled":
        before = _dynamo_counters()
        baseline_output = llm.generate([prompt], sampling, use_tqdm=False)[0]
        after = _dynamo_counters()
        authoritative_generation = _generation_record(baseline_output)
        baseline_counters = {
            "before": before,
            "after": after,
            "delta": _counter_delta(before, after),
        }
        if not llm.reset_prefix_cache():
            raise RuntimeError("vLLM refused to reset the prefix cache before capture")

    install = llm.apply_model(
        functools.partial(
            _install_capture,
            capacity=args.capture_capacity,
            target_layers=target_layers,
        )
    )[0]
    before_instrumented = _dynamo_counters()
    captured_output = llm.generate([prompt], sampling, use_tqdm=False)[0]
    after_instrumented = _dynamo_counters()
    captured_generation = _generation_record(captured_output)
    capture = llm.apply_model(_collect_capture)[0]
    llm.apply_model(_remove_capture)

    tensor_summaries = {
        name: _summarize_tensor(tensor)
        for name, tensor in capture["tensors"].items()
        if not name.startswith("replay.")
    }
    required = _required_capture_names(target_layers, text_config["layer_types"])
    missing_required = sorted(required - set(capture["tensors"]))
    tensor_payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "model_revision": MODEL_REVISION,
        "model_identity": model_identity,
        "prompt": prompt_text,
        "prompt_token_ids": prompt_token_ids,
        "prompt_provenance": prompt_provenance,
        "target_profile": args.target_layers,
        "target_layers": list(target_layers),
        "tensors": capture.pop("tensors"),
    }
    if args.tensor_output:
        args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor_payload, args.tensor_output)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "captured" if not missing_required and not capture["truncated"] else "incomplete",
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "cuda_device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "total_gpu_bytes": _total_gpu_bytes(),
        },
        "model": {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "resolved_path": str(model_path),
            "identity": model_identity,
            "load_seconds": load_seconds,
            "inventory": inventory,
        },
        "runtime": {
            "mode": args.mode,
            "enforce_eager": args.mode == "eager",
            "compiled_observer": args.mode == "compiled-observer",
            "compiled_observer_patch": observer_patch,
            "mtp_enabled": False,
            "mtp_scope_note": "main-model prefill only; speculative draft/decode is excluded",
            "language_model_only": True,
            "gdn_prefill_backend": "triton",
            "attention_backend": "TRITON_ATTN",
            "prefix_caching": True,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "disabled_kernels": os.environ.get("VLLM_DISABLED_KERNELS", ""),
            "target_profile": args.target_layers,
            "target_layers": list(target_layers),
        },
        "memory_preflight": memory_preflight,
        "prompt": {
            "text": prompt_text,
            "token_ids": prompt_token_ids,
            "token_count": len(prompt_token_ids),
            "provenance": prompt_provenance,
        },
        "capture": {
            **capture,
            "install": install,
            "tensor_output": str(args.tensor_output) if args.tensor_output else None,
            "tensor_summaries": tensor_summaries,
            "missing_required": missing_required,
        },
        "instrumented_generation": captured_generation,
        "instrumented_dynamo_counters": {
            "before": before_instrumented,
            "after": after_instrumented,
            "delta": _counter_delta(before_instrumented, after_instrumented),
        },
        "authority": {
            "detailed_tensor_capture": (
                "compile-visible output observer"
                if args.mode == "compiled-observer"
                else "instrumented"
            ),
            "detailed_tensor_capture_is_unmodified_serving_graph": False,
            "reason": (
                "The compiled-observer profile adds post-output opaque copy ops; "
                "ordinary post-load hooks are bypassed by the cached compiled graph."
                if args.mode == "compiled-observer"
                else "PyTorch forward hooks and copy buffers may invalidate Dynamo "
                "guards or inhibit compiled RMSNorm+FP8 quant fusion."
            ),
        },
    }
    if authoritative_generation is not None:
        result["authoritative_compiled_generation"] = authoritative_generation
        result["compiled_baseline_dynamo_counters"] = baseline_counters
        result["compiled_endpoint_parity"] = _generation_parity(
            authoritative_generation, captured_generation
        )
        result["authority"]["compiled_endpoint"] = "uninstrumented"

    result["gpu_memory"] = {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }

    llm.llm_engine.engine_core.shutdown()
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()
    torch.cuda.empty_cache()
    return result


def _tensor_parity(left: Any, right: Any) -> dict[str, Any]:
    import torch

    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "comparable": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
        }
    left_float = left.float()
    right_float = right.float()
    difference = right_float - left_float
    finite = torch.isfinite(left_float) & torch.isfinite(right_float)
    if finite.any():
        finite_difference = difference[finite]
        max_abs = float(finite_difference.abs().max().item())
        rms = float(torch.sqrt(torch.mean(finite_difference.square())).item())
        denominator = float(torch.sqrt(torch.mean(left_float[finite].square())).item())
    else:
        max_abs = math.inf
        rms = math.inf
        denominator = 0.0
    return {
        "comparable": True,
        "exact": torch.equal(left, right),
        "finite_pair_count": int(finite.sum().item()),
        "nonfinite_pair_count": int((~finite).sum().item()),
        "max_abs": max_abs,
        "rms": rms,
        "relative_rms": rms / max(denominator, 1e-30),
        "within_bf16_boundary_tolerance": max_abs <= 0.125 and rms <= 0.006,
    }


def _compare_tensor_payloads(eager_path: Path, compiled_path: Path) -> dict[str, Any]:
    import torch

    eager = torch.load(eager_path, map_location="cpu", weights_only=True)
    compiled = torch.load(compiled_path, map_location="cpu", weights_only=True)
    eager_tensors = eager["tensors"]
    compiled_tensors = compiled["tensors"]
    shared = sorted(
        name
        for name in set(eager_tensors) & set(compiled_tensors)
        if not name.startswith("replay.")
    )
    metrics = {
        name: _tensor_parity(eager_tensors[name], compiled_tensors[name]) for name in shared
    }
    comparable = [value for value in metrics.values() if value.get("comparable")]
    return {
        "eager_only": sorted(
            name
            for name in set(eager_tensors) - set(compiled_tensors)
            if not name.startswith("replay.")
        ),
        "compiled_only": sorted(
            name
            for name in set(compiled_tensors) - set(eager_tensors)
            if not name.startswith("replay.")
        ),
        "shared_tensor_count": len(shared),
        "all_exact": bool(comparable) and all(value["exact"] for value in comparable),
        "all_within_bf16_boundary_tolerance": bool(comparable)
        and all(value["within_bf16_boundary_tolerance"] for value in comparable),
        "metrics": metrics,
    }


def _child_command(
    args: argparse.Namespace, mode: str, output: Path, tensor_output: Path
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--output",
        str(output),
        "--tensor-output",
        str(tensor_output),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--capture-capacity",
        str(args.capture_capacity),
        "--target-layers",
        args.target_layers,
    ]
    if args.prompt_manifest is not None:
        command.extend(
            (
                "--prompt-manifest",
                str(args.prompt_manifest),
                "--prompt-index",
                str(args.prompt_index),
            )
        )
    else:
        command.extend(("--prompt", args.prompt))
    if args.model_path:
        command.extend(("--model-path", str(args.model_path)))
    if args.no_weight_digests:
        command.append("--no-weight-digests")
    if args.total_gpu_memory_gib is not None:
        command.extend(("--total-gpu-memory-gib", str(args.total_gpu_memory_gib)))
    if args.allow_unsafe_capture_memory:
        command.append("--allow-unsafe-capture-memory")
    return command


def _run_both(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    with tempfile.TemporaryDirectory(prefix="nvfp4-capture-") as temp:
        temp_path = Path(temp)
        eager_json = temp_path / "eager.json"
        eager_tensors = temp_path / "eager.pt"
        compiled_json = temp_path / "compiled.json"
        compiled_tensors = temp_path / "compiled.pt"
        observer_json = temp_path / "compiled-observer.json"
        observer_tensors = temp_path / "compiled-observer.pt"
        for mode, output, tensors in (
            ("eager", eager_json, eager_tensors),
            ("compiled", compiled_json, compiled_tensors),
            ("compiled-observer", observer_json, observer_tensors),
        ):
            subprocess.run(
                _child_command(args, mode, output, tensors),
                check=True,
                env=os.environ.copy(),
            )
        eager = json.loads(eager_json.read_text())
        compiled = json.loads(compiled_json.read_text())
        observer = json.loads(observer_json.read_text())
        compiled_parity = _compare_tensor_payloads(eager_tensors, compiled_tensors)
        observer_parity = _compare_tensor_payloads(eager_tensors, observer_tensors)
        endpoint_parity = _generation_parity(
            compiled["authoritative_compiled_generation"],
            observer["instrumented_generation"],
        )
        if args.tensor_output:
            args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "mode": "both",
                    "eager": torch.load(eager_tensors, map_location="cpu", weights_only=True),
                    "compiled": torch.load(
                        compiled_tensors, map_location="cpu", weights_only=True
                    ),
                    "compiled_observer": torch.load(
                        observer_tensors, map_location="cpu", weights_only=True
                    ),
                },
                args.tensor_output,
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "captured"
                if eager["status"] == "captured"
                and observer["status"] == "captured"
                and endpoint_parity["generated_token_equal"]
                and endpoint_parity["max_abs_shared_logprob_delta"] == 0.0
                else "incomplete"
            ),
            "mode": "both",
            "eager": eager,
            "compiled": compiled,
            "compiled_observer": observer,
            "eager_vs_compiled_split_boundary_parity": compiled_parity,
            "eager_vs_compiled_observer_parity": observer_parity,
            "uninstrumented_compiled_vs_compiled_observer_endpoint": endpoint_parity,
            "authority": {
                "compiled_endpoint": "uninstrumented authoritative main-model prefill",
                "boundary_parity": "instrumented diagnostic only",
                "mtp_enabled": False,
                "limitation": (
                    "No Python-hook capture can prove values inside the untouched compiled "
                    "graph; hooks can cause recompilation or block fusion."
                ),
            },
            "tensor_output": str(args.tensor_output) if args.tensor_output else None,
        }


def main() -> None:
    args = _parse_args()
    if args.mode == "compiled-observer":
        # The normal cache key intentionally tracks source files, not runtime
        # monkeypatches. Reusing an uninstrumented AOT graph would bypass the
        # observer ops entirely.
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    _prepare_process_environment()
    if args.preflight_only:
        result = _run_preflight_only(args)
    else:
        result = _run_both(args) if args.mode == "both" else _run_single(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
