#!/usr/bin/env python3
"""Offline validation of the captured Qwen3.6 NVFP4 late suffix.

This script replays decoder layer 62 (Gated DeltaNet) and layer 63 (full
attention) from a tensor payload emitted by ``check_nvfp4_runtime_capture.py``.
Runtime FP8 weights come only from the post-load capture.  MLP W4 weights are
read from the raw pinned ModelOpt checkpoint and dequantized for frozen-weight
VJPs.

Captured values are exact forward substitutions.  The backward is the named
NVFP4/FP8-STE surrogate, not the literal derivative of quantization.  The
outer residual additions and RMSNorm replay are reconstructed operations, so
their parity is reported separately from captured kernel boundaries.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Protocol

import torch

from modelopt_checkpoint import (
    ModelOptCheckpoint,
    PINNED_METADATA_SHA256,
    PINNED_REVISION,
    default_pinned_snapshot,
)
from nvfp4_attention import (
    QWEN36_27B_LAYER63,
    QwenBlockLinears,
    QwenFullAttentionConfig,
    replay_qwen_full_attention_suffix,
)
from nvfp4_gdn import (
    GdnCapture,
    GdnLayout,
    GdnWeights,
    replay_qwen_gdn_block,
)
from nvfp4_ste import (
    dequantize_runtime_fp8_weight,
    exact_fp8_linear_ste,
    exact_value,
    exact_w4a16_linear,
    gated_rms_norm,
)


SCHEMA_VERSION = 1
MODEL_REVISION = PINNED_REVISION
DEFAULT_GDN_LAYOUT = GdnLayout(
    key_heads=16,
    value_heads=48,
    key_dim=128,
    value_dim=128,
    norm_eps=1e-6,
)
CHECKPOINT_PREFIX = "model.language_model.layers"
GDN_PROOF_SUFFIXES = (
    "a",
    "b",
    "beta",
    "chunk_output",
    "conv_output_prefill",
    "core_output",
    "final_state",
    "initial_state",
    "k",
    "log_g",
    "mixed_qkv",
    "q",
    "v",
)
EXPECTED_SHARED_PROOF_NAMES = {
    *(f"gdn.layer{layer}.{suffix}" for layer in (61, 62) for suffix in GDN_PROOF_SUFFIXES),
    "attention.layer63.q_post_rope",
    "attention.layer63.k_post_rope",
    "attention.layer63.v",
    "attention.layer63.core_output",
}
EXPECTED_ALL_LAYER_SHARED_TENSORS = 688
EXPECTED_ALL_LAYER_OBSERVER_ONLY_TENSORS = 432
EXPECTED_ALL_LAYER_REPLAY_PARAMETERS = 785
MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_IDENTITY_POLICY = "pinned-nvidia-modelopt-snapshot-v1"


class CheckpointWeight(Protocol):
    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor: ...


class CheckpointReader(Protocol):
    def load_nvfp4(self, module_name: str) -> CheckpointWeight: ...


@dataclass(frozen=True)
class ErrorMetrics:
    shape: tuple[int, ...]
    finite: bool
    exact: bool
    max_abs: float
    rms: float
    relative_rms: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=default_pinned_snapshot())
    parser.add_argument("--capture-report", type=Path)
    parser.add_argument("--observer-proof", type=Path)
    parser.add_argument(
        "--require-authoritative",
        action="store_true",
        help="fail unless a valid isolated compiled-observer proof is supplied",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--ste-policy", choices=("identity", "clipped"), default="identity")
    parser.add_argument("--checkpoint-interval", type=int, default=16)
    parser.add_argument("--vjp-rows", type=int, default=2)
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--forward-atol", type=float, default=0.0)
    parser.add_argument("--forward-rtol", type=float, default=0.0)
    parser.add_argument("--vjp-atol", type=float, default=1e-4)
    parser.add_argument("--vjp-rtol", type=float, default=5e-4)
    parser.add_argument("--allow-unpinned-checkpoint", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


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


def load_capture_payload(path: Path) -> dict[str, Any]:
    """Load the tensor-only payload without permitting arbitrary pickle globals."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"failed to load capture payload {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("capture payload root must be a mapping")
    return payload


def _fp8_prefix(module_name: str) -> str:
    return f"replay.fp8.{module_name}"


def _required_tensor_names() -> set[str]:
    names = {"h61_post_block", "h62_post_block", "h63_post_block"}
    for module in (
        "layers.62.linear_attn.in_proj_qkvz",
        "layers.62.linear_attn.in_proj_ba",
        "layers.62.linear_attn.out_proj",
        "layers.62.mlp.gate_up_proj",
        "layers.62.mlp.down_proj",
        "layers.63.self_attn.qkv_proj",
        "layers.63.self_attn.o_proj",
        "layers.63.mlp.gate_up_proj",
        "layers.63.mlp.down_proj",
    ):
        names.add(f"linear.{module}.output")
    names.update(
        {
            "layers.62.mlp.swiglu_output",
            "layers.63.mlp.swiglu_output",
            "attention.layer63.q_post_rope",
            "attention.layer63.k_post_rope",
            "attention.layer63.v",
            "attention.layer63.core_output",
        }
    )
    names.update(
        f"gdn.layer62.{suffix}"
        for suffix in (
            "conv_output_prefill",
            "q",
            "k",
            "v",
            "log_g",
            "beta",
            "initial_state",
            "chunk_output",
            "final_state",
            "core_output",
        )
    )
    for module in (
        "layers.62.linear_attn.in_proj_qkvz",
        "layers.62.linear_attn.out_proj",
        "layers.63.self_attn.qkv_proj",
        "layers.63.self_attn.o_proj",
    ):
        prefix = _fp8_prefix(module)
        names.update(
            {
                f"{prefix}.weight",
                f"{prefix}.weight_scale",
                f"{prefix}.input_scale",
            }
        )
    for layer in (62, 63):
        names.update(
            {
                f"replay.norm.layers.{layer}.input_layernorm.weight",
                f"replay.norm.layers.{layer}.post_attention_layernorm.weight",
            }
        )
    names.update(
        {
            "replay.gdn.layers.62.in_proj_ba.weight",
            "replay.gdn.layers.62.conv1d.weight",
            "replay.gdn.layers.62.A_log",
            "replay.gdn.layers.62.dt_bias",
            "replay.gdn.layers.62.norm.weight",
            "replay.norm.layers.63.self_attn.q_norm.weight",
            "replay.norm.layers.63.self_attn.k_norm.weight",
        }
    )
    return names


def _optional_boundary_tensor_names() -> set[str]:
    names = {
        "gdn.layer62.norm_core_input",
        "gdn.layer62.norm_z_input",
        "gdn.layer62.norm_output",
    }
    for module in (
        "layers.62.linear_attn.in_proj_qkvz",
        "layers.62.linear_attn.in_proj_ba",
        "layers.62.linear_attn.out_proj",
        "layers.62.mlp.gate_up_proj",
        "layers.62.mlp.down_proj",
        "layers.63.self_attn.qkv_proj",
        "layers.63.self_attn.o_proj",
        "layers.63.mlp.gate_up_proj",
        "layers.63.mlp.down_proj",
    ):
        names.add(f"linear.{module}.input")
    return names


def validate_payload_schema(
    payload: Mapping[str, Any],
    *,
    attention_config: QwenFullAttentionConfig = QWEN36_27B_LAYER63,
    gdn_layout: GdnLayout = DEFAULT_GDN_LAYOUT,
) -> Mapping[str, torch.Tensor]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"capture schema {payload.get('schema_version')!r} != {SCHEMA_VERSION}"
        )
    if payload.get("model_revision") != MODEL_REVISION:
        raise ValueError("capture model revision does not match the pinned checkpoint")
    token_ids = payload.get("prompt_token_ids")
    if not isinstance(token_ids, list) or not token_ids or not all(
        isinstance(value, int) and value >= 0 for value in token_ids
    ):
        raise ValueError("prompt_token_ids must be a non-empty integer list")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("capture tensors must be a mapping")
    missing = sorted(_required_tensor_names() - set(tensors))
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(f"capture is missing {len(missing)} required tensors: {preview}{suffix}")
    non_tensors = sorted(name for name, value in tensors.items() if not isinstance(value, torch.Tensor))
    if non_tensors:
        raise TypeError(f"capture entries are not tensors: {non_tensors[:4]}")

    tokens = len(token_ids)
    hidden = attention_config.hidden_size
    expected = {
        "h61_post_block": (tokens, hidden),
        "h62_post_block": (tokens, hidden),
        "h63_post_block": (tokens, hidden),
        "attention.layer63.q_post_rope": (tokens, attention_config.query_size),
        "attention.layer63.k_post_rope": (tokens, attention_config.kv_size),
        "attention.layer63.v": (tokens, attention_config.kv_size),
        "attention.layer63.core_output": (tokens, attention_config.query_size),
        "gdn.layer62.q": (1, tokens, gdn_layout.key_heads, gdn_layout.key_dim),
        "gdn.layer62.k": (1, tokens, gdn_layout.key_heads, gdn_layout.key_dim),
        "gdn.layer62.v": (1, tokens, gdn_layout.value_heads, gdn_layout.value_dim),
        "gdn.layer62.log_g": (1, tokens, gdn_layout.value_heads),
        "gdn.layer62.beta": (1, tokens, gdn_layout.value_heads),
        "gdn.layer62.chunk_output": (
            1,
            tokens,
            gdn_layout.value_heads,
            gdn_layout.value_dim,
        ),
        "gdn.layer62.initial_state": (
            1,
            gdn_layout.value_heads,
            gdn_layout.value_dim,
            gdn_layout.key_dim,
        ),
        "gdn.layer62.final_state": (
            1,
            gdn_layout.value_heads,
            gdn_layout.value_dim,
            gdn_layout.key_dim,
        ),
    }
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(
                f"capture tensor {name} shape {tuple(tensors[name].shape)} != {shape}"
            )
    for name in _required_tensor_names():
        tensor = tensors[name]
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor.float()).all()):
            raise ValueError(f"capture tensor contains non-finite values: {name}")
    return tensors


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> ErrorMetrics:
    if actual.shape != expected.shape:
        raise ValueError(
            f"comparison shapes differ: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    left = actual.detach().float()
    right = expected.detach().to(device=actual.device).float()
    difference = left - right
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite:
        return ErrorMetrics(tuple(actual.shape), False, False, math.inf, math.inf, math.inf)
    rms = torch.sqrt(torch.mean(difference.square()))
    reference_rms = torch.sqrt(torch.mean(right.square()))
    return ErrorMetrics(
        shape=tuple(actual.shape),
        finite=True,
        exact=bool(torch.equal(actual.detach(), expected.detach().to(actual.device))),
        max_abs=float(difference.abs().max().item()),
        rms=float(rms.item()),
        relative_rms=float((rms / reference_rms.clamp_min(1e-12)).item()),
    )


def _metric_record(metrics: ErrorMetrics) -> dict[str, Any]:
    record = asdict(metrics)
    record["shape"] = list(metrics.shape)
    return record


def _within(metrics: ErrorMetrics, *, atol: float, rtol: float) -> bool:
    return (
        metrics.finite
        and metrics.max_abs <= atol
        and metrics.relative_rms <= rtol
    )


def compiled_residual_sum(
    logical_input: torch.Tensor,
    attention_branch: torch.Tensor,
    mlp_branch: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the observed compiled residual accumulation.

    The compiled-observer payload is bitwise consistent with a single FP32
    accumulation of the three BF16 branch values followed by one cast.  This
    differs from launching two eager BF16 additions.  The continuous
    surrogate derivative of either addition order is the same identity map.
    """

    if not (
        logical_input.shape == attention_branch.shape == mlp_branch.shape
    ):
        raise ValueError("compiled residual branches must have the same shape")
    return (
        logical_input.float()
        + attention_branch.float()
        + mlp_branch.float()
    ).to(logical_input.dtype)


def future_summed_vjp_rows(
    target: torch.Tensor,
    sources: Sequence[torch.Tensor],
    valid_positions: torch.Tensor,
    row_start: int,
    row_stop: int,
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    """Vectorized Anthropic-estimator rows for unbatched ``[T, D]`` activations."""

    if target.ndim != 2:
        raise ValueError("target must have shape [tokens, hidden]")
    tokens, hidden = target.shape
    if not 0 <= row_start < row_stop <= hidden:
        raise ValueError("invalid VJP row interval")
    if valid_positions.ndim != 1 or valid_positions.numel() == 0:
        raise ValueError("valid_positions must be non-empty")
    if int(valid_positions.min()) < 0 or int(valid_positions.max()) >= tokens:
        raise ValueError("valid position is outside the target sequence")
    for source in sources:
        if source.shape != target.shape or not source.requires_grad:
            raise ValueError("every VJP source must match target and require gradients")

    count = row_stop - row_start
    cotangents = torch.zeros(
        count,
        tokens,
        hidden,
        device=target.device,
        dtype=target.dtype,
    )
    rows = torch.arange(count, device=target.device)
    dimensions = row_start + rows
    cotangents[rows[:, None], valid_positions[None, :], dimensions[:, None]] = 1
    gradients = torch.autograd.grad(
        target,
        tuple(sources),
        grad_outputs=cotangents,
        retain_graph=retain_graph,
        is_grads_batched=True,
    )
    result = tuple(
        gradient[:, valid_positions].float().mean(dim=1) for gradient in gradients
    )
    if not all(bool(torch.isfinite(value).all()) for value in result):
        raise FloatingPointError("batched suffix VJP contains non-finite values")
    return result


def sequential_future_summed_vjp_rows(
    target: torch.Tensor,
    sources: Sequence[torch.Tensor],
    valid_positions: torch.Tensor,
    row_start: int,
    row_stop: int,
) -> tuple[torch.Tensor, ...]:
    rows = [
        torch.empty(
            row_stop - row_start,
            target.shape[-1],
            dtype=torch.float32,
            device=target.device,
        )
        for _ in sources
    ]
    for offset, dimension in enumerate(range(row_start, row_stop)):
        cotangent = torch.zeros_like(target)
        cotangent[valid_positions, dimension] = 1
        gradients = torch.autograd.grad(
            target,
            tuple(sources),
            grad_outputs=cotangent,
            retain_graph=dimension + 1 < row_stop,
        )
        for destination, gradient in zip(rows, gradients, strict=True):
            destination[offset] = gradient[valid_positions].float().mean(dim=0)
    result = tuple(rows)
    if not all(bool(torch.isfinite(value).all()) for value in result):
        raise FloatingPointError("sequential suffix VJP contains non-finite values")
    return result


def _runtime_fp8_weight(
    tensors: Mapping[str, torch.Tensor],
    module_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = _fp8_prefix(module_name)
    weight = dequantize_runtime_fp8_weight(
        tensors[f"{prefix}.weight"],
        tensors[f"{prefix}.weight_scale"],
        transposed=True,
        dtype=dtype,
    ).to(device)
    input_scale = tensors[f"{prefix}.input_scale"].to(device=device)
    return weight, input_scale


def _checkpoint_w4_weight(
    checkpoint: CheckpointReader,
    layer: int,
    module: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    name = f"{CHECKPOINT_PREFIX}.{layer}.mlp.{module}"
    return checkpoint.load_nvfp4(name).dequantize(dtype=dtype).to(device)


def _capture_tensor(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    device: torch.device,
) -> torch.Tensor:
    return tensors[name].to(device)


def _gdn_capture(
    tensors: Mapping[str, torch.Tensor],
    layout: GdnLayout,
    device: torch.device,
) -> GdnCapture:
    tokens = tensors["h61_post_block"].shape[0]
    qkvz = _capture_tensor(
        tensors,
        "linear.layers.62.linear_attn.in_proj_qkvz.output",
        device,
    )
    value_size = layout.value_heads * layout.value_dim
    z = qkvz[..., -value_size:].reshape(
        tokens, layout.value_heads, layout.value_dim
    )
    core_output = _capture_tensor(
        tensors, "gdn.layer62.chunk_output", device
    ).squeeze(0)
    if "gdn.layer62.norm_output" in tensors:
        gated_norm = _capture_tensor(
            tensors, "gdn.layer62.norm_output", device
        ).reshape(tokens, layout.value_heads, layout.value_dim)
    else:
        gated_norm = gated_rms_norm(
            core_output,
            z,
            _capture_tensor(
                tensors, "replay.gdn.layers.62.norm.weight", device
            ),
            layout.norm_eps,
        )
    return GdnCapture(
        qkvz=qkvz,
        ba=_capture_tensor(
            tensors,
            "linear.layers.62.linear_attn.in_proj_ba.output",
            device,
        ),
        conv_qkv=_capture_tensor(tensors, "gdn.layer62.conv_output_prefill", device),
        query=_capture_tensor(tensors, "gdn.layer62.q", device).squeeze(0),
        key=_capture_tensor(tensors, "gdn.layer62.k", device).squeeze(0),
        value=_capture_tensor(tensors, "gdn.layer62.v", device).squeeze(0),
        log_decay=_capture_tensor(tensors, "gdn.layer62.log_g", device).squeeze(0),
        beta=_capture_tensor(tensors, "gdn.layer62.beta", device).squeeze(0),
        core_output=core_output,
        final_state=_capture_tensor(tensors, "gdn.layer62.final_state", device).squeeze(0),
        gated_norm=gated_norm,
        branch_output=_capture_tensor(
            tensors,
            "linear.layers.62.linear_attn.out_proj.output",
            device,
        ),
    )


def _boundary_metrics(
    tensors: Mapping[str, torch.Tensor],
    device: torch.device,
    layer62: Any,
    layer63: Any,
    gdn_capture: GdnCapture,
) -> dict[str, ErrorMetrics]:
    captured = lambda name: _capture_tensor(tensors, name, device)
    comparisons: tuple[tuple[str, torch.Tensor, str], ...] = (
        (
            "layer62.input_norm_to_qkvz",
            layer62.attention_input,
            "linear.layers.62.linear_attn.in_proj_qkvz.input",
        ),
        (
            "layer62.input_norm_to_ba",
            layer62.attention_input,
            "linear.layers.62.linear_attn.in_proj_ba.input",
        ),
        (
            "layer62.chunk_to_gated_norm",
            gdn_capture.core_output.flatten(0, 1),
            "gdn.layer62.norm_core_input",
        ),
        (
            "layer62.mlp_norm_to_gate_up",
            layer62.mlp_input,
            "linear.layers.62.mlp.gate_up_proj.input",
        ),
        (
            "layer62.swiglu_to_down",
            layer62.activated,
            "linear.layers.62.mlp.down_proj.input",
        ),
        (
            "layer63.input_norm_to_qkv",
            layer63.attention_input,
            "linear.layers.63.self_attn.qkv_proj.input",
        ),
        (
            "layer63.gated_attention_to_o",
            layer63.attention.gated_output,
            "linear.layers.63.self_attn.o_proj.input",
        ),
        (
            "layer63.mlp_norm_to_gate_up",
            layer63.mlp_input,
            "linear.layers.63.mlp.gate_up_proj.input",
        ),
        (
            "layer63.swiglu_to_down",
            layer63.activated,
            "linear.layers.63.mlp.down_proj.input",
        ),
    )
    metrics = {
        "layer62.chunk_vs_core_capture": error_metrics(
            gdn_capture.core_output,
            captured("gdn.layer62.core_output"),
        ),
    }
    for name, actual, capture_name in comparisons:
        if capture_name in tensors:
            metrics[name] = error_metrics(actual, captured(capture_name))
    return metrics


def validate_suffix_payload(
    payload: Mapping[str, Any],
    checkpoint: CheckpointReader,
    *,
    device: torch.device | str,
    attention_config: QwenFullAttentionConfig = QWEN36_27B_LAYER63,
    gdn_layout: GdnLayout = DEFAULT_GDN_LAYOUT,
    weight_dtype: torch.dtype = torch.bfloat16,
    ste_policy: str = "identity",
    checkpoint_interval: int = 16,
    vjp_rows: int = 2,
    skip_first: int = 0,
    forward_atol: float = 0.0,
    forward_rtol: float = 0.0,
    vjp_atol: float = 1e-4,
    vjp_rtol: float = 5e-4,
) -> dict[str, Any]:
    """Materialize the captured suffix, compare it, and validate partial J61/J62."""

    tensors = validate_payload_schema(
        payload,
        attention_config=attention_config,
        gdn_layout=gdn_layout,
    )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested but CUDA is unavailable")
    if ste_policy not in {"identity", "clipped"}:
        raise ValueError("ste_policy must be identity or clipped")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if vjp_rows < 0 or vjp_rows > attention_config.hidden_size:
        raise ValueError("vjp_rows is outside the hidden width")
    tokens = len(payload["prompt_token_ids"])
    if skip_first < 0 or (vjp_rows and tokens <= skip_first + 1):
        raise ValueError("prompt is too short for the selected VJP position mask")

    started = time.monotonic()
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)

    qkvz_weight, qkvz_scale = _runtime_fp8_weight(
        tensors,
        "layers.62.linear_attn.in_proj_qkvz",
        device=target_device,
        dtype=weight_dtype,
    )
    gdn_out_weight, gdn_out_scale = _runtime_fp8_weight(
        tensors,
        "layers.62.linear_attn.out_proj",
        device=target_device,
        dtype=weight_dtype,
    )
    qkv_weight, qkv_scale = _runtime_fp8_weight(
        tensors,
        "layers.63.self_attn.qkv_proj",
        device=target_device,
        dtype=weight_dtype,
    )
    attention_out_weight, attention_out_scale = _runtime_fp8_weight(
        tensors,
        "layers.63.self_attn.o_proj",
        device=target_device,
        dtype=weight_dtype,
    )
    layer62_gate_up_weight = _checkpoint_w4_weight(
        checkpoint,
        62,
        "gate_up_proj",
        device=target_device,
        dtype=weight_dtype,
    )
    layer62_down_weight = _checkpoint_w4_weight(
        checkpoint,
        62,
        "down_proj",
        device=target_device,
        dtype=weight_dtype,
    )
    layer63_gate_up_weight = _checkpoint_w4_weight(
        checkpoint,
        63,
        "gate_up_proj",
        device=target_device,
        dtype=weight_dtype,
    )
    layer63_down_weight = _checkpoint_w4_weight(
        checkpoint,
        63,
        "down_proj",
        device=target_device,
        dtype=weight_dtype,
    )

    capture = _gdn_capture(tensors, gdn_layout, target_device)
    weights = GdnWeights(
        qkvz_out_in=qkvz_weight,
        qkvz_input_scale=qkvz_scale,
        ba_out_in=_capture_tensor(
            tensors, "replay.gdn.layers.62.in_proj_ba.weight", target_device
        ).to(weight_dtype),
        conv=_capture_tensor(
            tensors, "replay.gdn.layers.62.conv1d.weight", target_device
        ),
        a_log=_capture_tensor(tensors, "replay.gdn.layers.62.A_log", target_device),
        dt_bias=_capture_tensor(tensors, "replay.gdn.layers.62.dt_bias", target_device),
        norm=_capture_tensor(tensors, "replay.gdn.layers.62.norm.weight", target_device),
        out_out_in=gdn_out_weight,
        out_input_scale=gdn_out_scale,
    )

    captured = lambda name: _capture_tensor(tensors, name, target_device)
    h61 = captured("h61_post_block").detach().requires_grad_(True)
    layer62 = replay_qwen_gdn_block(
        h61,
        gdn_layout,
        weights,
        capture,
        input_norm_weight=captured(
            "replay.norm.layers.62.input_layernorm.weight"
        ),
        post_attention_norm_weight=captured(
            "replay.norm.layers.62.post_attention_layernorm.weight"
        ),
        gate_up_linear=lambda inputs: exact_w4a16_linear(
            inputs,
            captured("linear.layers.62.mlp.gate_up_proj.output"),
            layer62_gate_up_weight,
        ),
        down_linear=lambda inputs: exact_w4a16_linear(
            inputs,
            captured("linear.layers.62.mlp.down_proj.output"),
            layer62_down_weight,
        ),
        exact_swiglu_output=captured("layers.62.mlp.swiglu_output"),
        initial_state=captured("gdn.layer62.initial_state").squeeze(0),
        ste_policy=ste_policy,
        checkpoint_interval=checkpoint_interval,
    )
    h62_captured = captured("h62_post_block")
    h62_eager_metric = error_metrics(layer62.output, h62_captured)
    h62_reconstructed = compiled_residual_sum(
        h61,
        layer62.attention_output,
        layer62.mlp_output,
    )
    h62_metric = error_metrics(h62_reconstructed, h62_captured)
    h62 = exact_value(h62_reconstructed, h62_captured)

    layer63_linears = QwenBlockLinears(
        qkv=lambda inputs: exact_fp8_linear_ste(
            inputs,
            captured("linear.layers.63.self_attn.qkv_proj.output"),
            qkv_weight,
            qkv_scale,
            ste_policy=ste_policy,
        ),
        attention_out=lambda inputs: exact_fp8_linear_ste(
            inputs,
            captured("linear.layers.63.self_attn.o_proj.output"),
            attention_out_weight,
            attention_out_scale,
            ste_policy=ste_policy,
        ),
        gate_up=lambda inputs: exact_w4a16_linear(
            inputs,
            captured("linear.layers.63.mlp.gate_up_proj.output"),
            layer63_gate_up_weight,
        ),
        down=lambda inputs: exact_w4a16_linear(
            inputs,
            captured("linear.layers.63.mlp.down_proj.output"),
            layer63_down_weight,
        ),
    )
    positions = torch.arange(tokens, device=target_device, dtype=torch.long)
    layer63 = replay_qwen_full_attention_suffix(
        h62,
        positions,
        attention_config,
        input_norm_weight=captured(
            "replay.norm.layers.63.input_layernorm.weight"
        ),
        post_attention_norm_weight=captured(
            "replay.norm.layers.63.post_attention_layernorm.weight"
        ),
        q_norm_weight=captured(
            "replay.norm.layers.63.self_attn.q_norm.weight"
        ),
        k_norm_weight=captured(
            "replay.norm.layers.63.self_attn.k_norm.weight"
        ),
        linears=layer63_linears,
        exact_query=captured("attention.layer63.q_post_rope"),
        exact_key=captured("attention.layer63.k_post_rope"),
        exact_value=captured("attention.layer63.v"),
        exact_attention_output=captured("attention.layer63.core_output"),
        exact_swiglu_output=captured("layers.63.mlp.swiglu_output"),
    )
    h63_captured = captured("h63_post_block")
    h63_eager_metric = error_metrics(layer63.output, h63_captured)
    h63_reconstructed = compiled_residual_sum(
        h62,
        layer63.attention.output,
        layer63.hidden_states,
    )
    h63_metric = error_metrics(h63_reconstructed, h63_captured)
    target = exact_value(h63_reconstructed, h63_captured)

    boundary = _boundary_metrics(
        tensors,
        target_device,
        layer62,
        layer63,
        capture,
    )
    initial_state = captured("gdn.layer62.initial_state")
    forward_pass = _within(h62_metric, atol=forward_atol, rtol=forward_rtol) and _within(
        h63_metric, atol=forward_atol, rtol=forward_rtol
    )

    vjp_record: dict[str, Any]
    vjp_pass = True
    if vjp_rows:
        valid_positions = torch.arange(
            skip_first,
            tokens - 1,
            device=target_device,
            dtype=torch.long,
        )
        batched = future_summed_vjp_rows(
            target,
            (h61, h62),
            valid_positions,
            0,
            vjp_rows,
            retain_graph=True,
        )
        sequential = sequential_future_summed_vjp_rows(
            target,
            (h61, h62),
            valid_positions,
            0,
            vjp_rows,
        )
        sources: dict[str, Any] = {}
        for name, vectorized, reference in zip(
            ("J61", "J62"), batched, sequential, strict=True
        ):
            parity = error_metrics(vectorized, reference)
            passed = _within(parity, atol=vjp_atol, rtol=vjp_rtol)
            vjp_pass = vjp_pass and passed
            sources[name] = {
                "shape": list(vectorized.shape),
                "finite": bool(torch.isfinite(vectorized).all()),
                "nonzero": bool((vectorized != 0).any()),
                "rms": float(vectorized.square().mean().sqrt().item()),
                "sha256_float32": tensor_sha256(vectorized),
                "sequential_parity": _metric_record(parity),
                "passed": passed,
            }
        vjp_record = {
            "enabled": True,
            "rows": [0, vjp_rows],
            "valid_positions": valid_positions.detach().cpu().tolist(),
            "estimator": "future-summed target cotangent; mean over matching source positions",
            "sources": sources,
        }
    else:
        vjp_record = {"enabled": False, "reason": "--vjp-rows=0"}

    elapsed = time.monotonic() - started
    result = {
        "schema_version": 1,
        "status": "passed" if forward_pass and vjp_pass else "failed",
        "model_revision": payload["model_revision"],
        "capture": {
            "mode": payload.get("mode"),
            "prompt": payload.get("prompt"),
            "token_count": tokens,
            "tensor_payload_authority": (
                "captured/instrumented values only; this validator does not upgrade "
                "them to an unmodified compiled serving graph"
            ),
        },
        "authority": {
            "accepted": False,
            "forward_capture": "instrumented-only",
            "reason": (
                "No independently validated isolated stock-vs-observer proof "
                "is bound to this result"
            ),
            "derivative_claim": "surrogate only; never exact quantized derivative",
        },
        "contract": {
            "forward": "captured exact kernel/linear values plus observed compiled FP32 residual accumulation",
            "backward": "NVFP4 frozen-weight / FP8-STE surrogate",
            "ste_policy": ste_policy,
            "weight_dtype": str(weight_dtype),
            "checkpoint_interval": checkpoint_interval,
            "fresh_prefill_zero_conv_tail": True,
            "captured_initial_state_used": True,
            "gdn_gated_norm_forward_value": (
                "captured"
                if "gdn.layer62.norm_output" in tensors
                else "recomputed surrogate from captured core/z"
            ),
            "literal_quantized_derivative": False,
        },
        "forward": {
            "h62_reconstructed": _metric_record(h62_metric),
            "h63_reconstructed": _metric_record(h63_metric),
            "eager_bf16_addition_diagnostic": {
                "h62": _metric_record(h62_eager_metric),
                "h63": _metric_record(h63_eager_metric),
            },
            "residual_accumulation": "FP32(input + attention + MLP), one cast to activation dtype",
            "tolerance": {"max_abs": forward_atol, "relative_rms": forward_rtol},
            "passed": forward_pass,
        },
        "boundary_parity": {
            name: _metric_record(metrics) for name, metrics in boundary.items()
        },
        "unavailable_optional_boundaries": sorted(
            _optional_boundary_tensor_names() - set(tensors)
        ),
        "gdn_initial_state": {
            "shape": list(initial_state.shape),
            "all_zero": bool((initial_state == 0).all()),
            "max_abs": float(initial_state.float().abs().max().item()),
        },
        "vjp_validation": vjp_record,
        "resources": {
            "elapsed_seconds": elapsed,
            "device": str(target_device),
            "max_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(target_device))
                if target_device.type == "cuda"
                else None
            ),
            "max_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(target_device))
                if target_device.type == "cuda"
                else None
            ),
        },
    }
    return result


def _load_capture_report(path: Path | None, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("capture report root must be an object")
    if value.get("schema_version") != payload.get("schema_version"):
        raise ValueError("capture report schema does not match tensor payload")
    model = value.get("model", {})
    if not isinstance(model, dict) or model.get("revision") != payload.get("model_revision"):
        raise ValueError("capture report model revision does not match tensor payload")
    return value


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _canonical_json_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(rendered).hexdigest()


def _validate_manifest_pair_observer_proof(
    proof: Mapping[str, Any],
    proof_path: Path,
    *,
    capture_path: Path,
    capture_report_path: Path,
    payload: Mapping[str, Any],
    capture_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the memory-bounded proof emitted by the all-layer verifier."""

    claim = proof.get("claim")
    configuration = proof.get("configuration")
    shared = proof.get("shared_internal_tensor_parity")
    replay = proof.get("replay_parameter_parity")
    completeness = proof.get("observer_capture_completeness")
    if (
        not isinstance(claim, Mapping)
        or claim.get("mtp") != "off"
        or claim.get("observer_graph_modified") is not True
        or claim.get("observer_modification_discharged") is not True
        or not isinstance(configuration, Mapping)
        or configuration.get("model_revision") != payload.get("model_revision")
        or configuration.get("target_profile") != "all"
        or configuration.get("target_layers") != list(range(64))
        or configuration.get("mtp_enabled") is not False
        or configuration.get("language_model_only") is not True
        or configuration.get("prompt", {}).get("token_ids")
        != payload.get("prompt_token_ids")
        or proof.get("generation_record_parity", {}).get("exact") is not True
        or not isinstance(shared, Mapping)
        or shared.get("shared_tensor_count") != EXPECTED_ALL_LAYER_SHARED_TENSORS
        or shared.get("observer_only_tensor_count")
        != EXPECTED_ALL_LAYER_OBSERVER_ONLY_TENSORS
        or shared.get("all_shared_bit_exact") is not True
        or not isinstance(replay, Mapping)
        or replay.get("parameter_count") != EXPECTED_ALL_LAYER_REPLAY_PARAMETERS
        or replay.get("all_names_equal") is not True
        or replay.get("all_shapes_equal") is not True
        or replay.get("all_dtypes_equal") is not True
        or replay.get("all_content_hashes_equal") is not True
        or replay.get("json_provenance_equal") is not True
        or not isinstance(completeness, Mapping)
        or completeness.get("required_missing") != []
        or completeness.get("truncated") != []
    ):
        raise ValueError("manifest-pair observer proof scope or parity is invalid")

    selected_report = _load_json_object(
        capture_report_path, label="observer capture report"
    )
    if selected_report != capture_report:
        raise ValueError("selected capture report changed after it was loaded")
    runtime = selected_report.get("runtime")
    report_prompt = selected_report.get("prompt")
    report_model = selected_report.get("model")
    identity = (
        report_model.get("identity")
        if isinstance(report_model, Mapping)
        else None
    )
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("mode") != "compiled-observer"
        or runtime.get("target_profile") != "all"
        or runtime.get("target_layers") != list(range(64))
        or runtime.get("mtp_enabled") is not False
        or not isinstance(report_prompt, Mapping)
        or report_prompt.get("token_ids") != payload.get("prompt_token_ids")
        or not isinstance(report_model, Mapping)
        or report_model.get("repo_id") != MODEL_REPO
        or report_model.get("revision") != payload.get("model_revision")
        or not isinstance(identity, Mapping)
        or identity.get("policy") != MODEL_IDENTITY_POLICY
        or identity.get("repo_id") != MODEL_REPO
        or identity.get("revision") != MODEL_REVISION
        or identity.get("metadata_sha256") != PINNED_METADATA_SHA256
        or identity.get("strict_pinned_validation") is not True
        or identity.get("validator")
        != "ModelOptCheckpoint(strict_pinned=True)"
        or identity.get("resolved_path") != report_model.get("resolved_path")
    ):
        raise ValueError("manifest-pair observer report scope mismatch")

    artifacts = proof.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("manifest-pair observer proof lacks artifact records")
    actual_records: dict[str, dict[str, Any]] = {}
    for key, selected in (
        ("observer_json", capture_report_path.resolve()),
        ("observer_tensors", capture_path.resolve()),
    ):
        record = artifacts.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"manifest-pair proof lacks {key} binding")
        actual = {
            "path": str(selected),
            "bytes": selected.stat().st_size,
            "sha256": sha256_file(selected),
        }
        if (
            Path(str(record.get("path", ""))).resolve() != selected
            or record.get("bytes") != actual["bytes"]
            or record.get("sha256") != actual["sha256"]
        ):
            raise ValueError(f"manifest-pair proof {key} binding mismatch")
        actual_records[key] = actual

    verifier_path = Path(__file__).resolve().with_name(
        "prove_nvfp4_capture_pair.py"
    )
    verifier = proof.get("verifier")
    if (
        not isinstance(verifier, Mapping)
        or Path(str(verifier.get("path", ""))).resolve() != verifier_path
        or verifier.get("bytes") != verifier_path.stat().st_size
        or verifier.get("sha256") != sha256_file(verifier_path)
    ):
        raise ValueError("manifest-pair proof verifier source binding mismatch")

    return {
        "accepted": True,
        "scope": "pinned prompt/configuration, main-model prefill, MTP disabled",
        "proof_schema": "memory-bounded-manifest-pair-v1",
        "proof_path": str(proof_path),
        "proof_bytes": proof_path.stat().st_size,
        "proof_sha256": sha256_file(proof_path),
        "basis": claim.get("discharge_basis"),
        "endpoint_generation_exact": True,
        "shared_tensor_count": EXPECTED_ALL_LAYER_SHARED_TENSORS,
        "shared_tensors_all_exact_on_independent_verifier": True,
        "replay_parameter_count": EXPECTED_ALL_LAYER_REPLAY_PARAMETERS,
        "replay_parameters_all_exact_on_independent_verifier": True,
        "artifacts": actual_records,
        "forward_capture_claim": "authoritative within the pinned observer-proof scope",
        "derivative_claim": "surrogate only; observer proof makes no derivative exactness claim",
    }


def validate_observer_proof(
    proof_path: Path,
    *,
    capture_path: Path,
    capture_report_path: Path,
    payload: Mapping[str, Any],
    capture_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind and independently recheck the isolated compiled observer proof."""

    proof_path = proof_path.resolve()
    proof = _load_json_object(proof_path, label="observer proof")
    if proof.get("schema_version") != 1 or proof.get("status") != "passed":
        raise ValueError("observer proof is not a passed schema-v1 proof")
    if "claim" in proof:
        return _validate_manifest_pair_observer_proof(
            proof,
            proof_path,
            capture_path=capture_path,
            capture_report_path=capture_report_path,
            payload=payload,
            capture_report=capture_report,
        )
    authority = proof.get("authority")
    if not isinstance(authority, Mapping) or authority.get("accepted") is not True:
        raise ValueError("observer proof authority was not accepted")
    if authority.get("ordinary_post_init_hooks_authoritative") is not False:
        raise ValueError("observer proof must reject ordinary post-init hooks")

    scope = proof.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("observer proof lacks a scope object")
    prompt = scope.get("prompt")
    if not isinstance(prompt, Mapping):
        raise ValueError("observer proof lacks prompt scope")
    if (
        scope.get("model_revision") != payload.get("model_revision")
        or scope.get("main_model_prefill") is not True
        or scope.get("mtp_enabled") is not False
        or prompt.get("token_ids") != payload.get("prompt_token_ids")
        or prompt.get("token_count") != len(payload["prompt_token_ids"])
        or prompt.get("text") != payload.get("prompt")
    ):
        raise ValueError("observer proof scope does not match the capture payload")

    artifacts = proof.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("observer proof lacks artifact paths")
    path_keys = (
        "baseline_json",
        "baseline_tensors",
        "observer_json",
        "observer_tensors",
    )
    resolved: dict[str, Path] = {}
    for key in path_keys:
        raw_path = artifacts.get(key)
        if not isinstance(raw_path, str):
            raise ValueError(f"observer proof artifact {key} is not a path")
        artifact_path = Path(raw_path).expanduser().resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"observer proof artifact is missing: {artifact_path}")
        resolved[key] = artifact_path
    if resolved["observer_tensors"] != capture_path.resolve():
        raise ValueError("observer proof tensor artifact is not the selected capture")
    if resolved["observer_json"] != capture_report_path.resolve():
        raise ValueError("observer proof JSON artifact is not the selected capture report")
    for key, size_key in (
        ("baseline_tensors", "baseline_tensor_bytes"),
        ("observer_tensors", "observer_tensor_bytes"),
    ):
        expected_size = artifacts.get(size_key)
        actual_size = resolved[key].stat().st_size
        if not isinstance(expected_size, int) or actual_size != expected_size:
            raise ValueError(
                f"observer proof artifact size mismatch for {key}: "
                f"{actual_size} != {expected_size!r}"
            )

    baseline_report = _load_json_object(
        resolved["baseline_json"], label="baseline capture report"
    )
    observer_report = _load_json_object(
        resolved["observer_json"], label="observer capture report"
    )
    if observer_report != capture_report:
        raise ValueError("selected capture report changed after it was loaded")
    for report, expected_mode in (
        (baseline_report, "compiled"),
        (observer_report, "compiled-observer"),
    ):
        model = report.get("model")
        runtime = report.get("runtime")
        report_prompt = report.get("prompt")
        if (
            not isinstance(model, Mapping)
            or model.get("revision") != payload.get("model_revision")
            or not isinstance(runtime, Mapping)
            or runtime.get("mode") != expected_mode
            or not isinstance(report_prompt, Mapping)
            or report_prompt.get("token_ids") != payload.get("prompt_token_ids")
        ):
            raise ValueError(f"{expected_mode} proof report scope mismatch")
    baseline_generation = baseline_report.get("authoritative_compiled_generation")
    observer_generation = observer_report.get("instrumented_generation")
    if not isinstance(baseline_generation, Mapping) or baseline_generation != observer_generation:
        raise ValueError("stock and observer endpoint generations are not exactly equal")

    endpoint = proof.get("endpoint_parity")
    if (
        not isinstance(endpoint, Mapping)
        or endpoint.get("generated_token_equal") is not True
        or endpoint.get("max_abs_shared_logprob_delta") != 0.0
        or endpoint.get("shared_top_logprob_tokens") != 20
        or not isinstance(endpoint.get("logprob_deltas"), Mapping)
        or any(value != 0.0 for value in endpoint["logprob_deltas"].values())
    ):
        raise ValueError("observer proof does not establish exact endpoint parity")

    parity = proof.get("shared_compiled_boundary_parity")
    if not isinstance(parity, Mapping):
        raise ValueError("observer proof lacks shared tensor parity")
    metrics = parity.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != EXPECTED_SHARED_PROOF_NAMES:
        raise ValueError("observer proof shared tensor names do not match the required 30")
    if (
        parity.get("shared_tensor_count") != len(EXPECTED_SHARED_PROOF_NAMES)
        or parity.get("all_exact") is not True
        or parity.get("all_within_bf16_boundary_tolerance") is not True
        or parity.get("eager_only") != []
    ):
        raise ValueError("observer proof shared tensor aggregate did not pass exactly")
    for name, metric in metrics.items():
        if (
            not isinstance(metric, Mapping)
            or metric.get("comparable") is not True
            or metric.get("exact") is not True
            or metric.get("max_abs") != 0.0
            or metric.get("rms") != 0.0
            or metric.get("relative_rms") != 0.0
            or metric.get("nonfinite_pair_count") != 0
        ):
            raise ValueError(f"observer proof metric is not exact: {name}")

    baseline_payload = load_capture_payload(resolved["baseline_tensors"])
    if (
        baseline_payload.get("schema_version") != payload.get("schema_version")
        or baseline_payload.get("mode") != "compiled"
        or baseline_payload.get("model_revision") != payload.get("model_revision")
        or baseline_payload.get("prompt_token_ids") != payload.get("prompt_token_ids")
    ):
        raise ValueError("baseline tensor payload scope mismatch")
    baseline_tensors = baseline_payload.get("tensors")
    observer_tensors = payload.get("tensors")
    if not isinstance(baseline_tensors, Mapping) or not isinstance(observer_tensors, Mapping):
        raise ValueError("proof tensor payload lacks a tensor mapping")
    independently_checked = 0
    for name in sorted(EXPECTED_SHARED_PROOF_NAMES):
        left = baseline_tensors.get(name)
        right = observer_tensors.get(name)
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise ValueError(f"proof shared tensor is absent: {name}")
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise ValueError(f"proof shared tensor differs on independent reload: {name}")
        independently_checked += 1
    del baseline_payload, baseline_tensors

    artifact_records = {
        key: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for key, path in resolved.items()
    }
    return {
        "accepted": True,
        "scope": "pinned prompt/configuration, main-model prefill, MTP disabled",
        "proof_path": str(proof_path),
        "proof_bytes": proof_path.stat().st_size,
        "proof_sha256": sha256_file(proof_path),
        "basis": authority.get("basis"),
        "endpoint_generation_exact": True,
        "endpoint_generation_sha256": _canonical_json_sha256(baseline_generation),
        "shared_tensor_count": independently_checked,
        "shared_tensors_all_exact_on_independent_reload": True,
        "shared_tensor_names_sha256": _canonical_json_sha256(
            sorted(EXPECTED_SHARED_PROOF_NAMES)
        ),
        "artifacts": artifact_records,
        "forward_capture_claim": "authoritative within the pinned observer-proof scope",
        "derivative_claim": "surrogate only; observer proof makes no derivative exactness claim",
    }


def main() -> int:
    args = _parse_args()
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive")
    if min(args.forward_atol, args.forward_rtol, args.vjp_atol, args.vjp_rtol) < 0:
        raise ValueError("validation tolerances must be nonnegative")
    if args.require_authoritative and args.observer_proof is None:
        raise ValueError(
            "--require-authoritative requires --observer-proof and --capture-report"
        )
    if args.observer_proof is not None and args.capture_report is None:
        raise ValueError("--observer-proof requires --capture-report")
    payload = load_capture_payload(args.capture)
    capture_report = _load_capture_report(args.capture_report, payload)
    observer_proof = None
    if args.observer_proof is not None:
        assert args.capture_report is not None
        assert capture_report is not None
        observer_proof = validate_observer_proof(
            args.observer_proof,
            capture_path=args.capture,
            capture_report_path=args.capture_report,
            payload=payload,
            capture_report=capture_report,
        )
    checkpoint = ModelOptCheckpoint(
        args.checkpoint,
        strict_pinned=not args.allow_unpinned_checkpoint,
    )
    weight_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.weight_dtype]
    result = validate_suffix_payload(
        payload,
        checkpoint,
        device=args.device,
        weight_dtype=weight_dtype,
        ste_policy=args.ste_policy,
        checkpoint_interval=args.checkpoint_interval,
        vjp_rows=args.vjp_rows,
        skip_first=args.skip_first,
        forward_atol=args.forward_atol,
        forward_rtol=args.forward_rtol,
        vjp_atol=args.vjp_atol,
        vjp_rtol=args.vjp_rtol,
    )
    result["source"] = {
        "capture_path": str(args.capture.resolve()),
        "capture_sha256": sha256_file(args.capture),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "validator_path": str(Path(__file__).resolve()),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
    }
    if capture_report is not None:
        result["capture"]["report_path"] = str(args.capture_report.resolve())
        result["capture"]["report_sha256"] = sha256_file(args.capture_report)
        result["capture"]["report_status"] = capture_report.get("status")
        result["capture"]["reported_authority"] = capture_report.get("authority")
    if observer_proof is not None:
        result["observer_proof"] = observer_proof
        result["capture"]["tensor_payload_authority"] = (
            "authoritative within the independently rechecked, pinned "
            "stock-compiled-vs-observer proof scope"
        )
        result["authority"] = {
            "accepted": True,
            "forward_capture": observer_proof["forward_capture_claim"],
            "basis": observer_proof["basis"],
            "scope": observer_proof["scope"],
            "derivative_claim": observer_proof["derivative_claim"],
        }
    atomic_write_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "h62_max_abs": result["forward"]["h62_reconstructed"]["max_abs"],
        "h63_max_abs": result["forward"]["h63_reconstructed"]["max_abs"],
        "vjp": result["vjp_validation"].get("enabled"),
    }, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
