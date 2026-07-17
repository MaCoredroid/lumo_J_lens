#!/usr/bin/env python3
"""Memory-bounded input VJP for live post-load vLLM FP8 linears.

The deployed Cutlass weight is E4M3 ``[K, N]`` with one scalar weight scale.
Given ``grad_output[M, N]``, this module computes ``grad_input[M, K]`` using
BF16 tensor-core products and FP32 accumulation without materializing a full
dequantized ``[N, K]`` matrix.  The exact-forward wrapper also applies the
declared identity or clipped straight-through derivative for FP8 activation
quantization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only environments use the Torch fallback.
    triton = None
    tl = None


Backend = Literal["auto", "torch", "triton"]
StePolicy = Literal["identity", "clipped"]
FP8_E4M3_MAX = 448.0
DEFAULT_MODULE = "layers.62.linear_attn.in_proj_qkvz"
DEFAULT_WEIGHT_CAPTURE = Path(".cache/runtime_capture/compiled_observer_final.pt")
DEFAULT_FORWARD_CAPTURE = Path(".cache/runtime_capture/eager_smoke.pt")
SCHEMA_VERSION = 1
MAX_REAL_PROBE_RELATIVE_RMS = 1e-5
MAX_REAL_WRAPPER_RELATIVE_RMS = 5e-3


def _validate_inputs(
    grad_output: torch.Tensor,
    postload_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    dot_dtype: torch.dtype,
) -> tuple[int, int, int]:
    if grad_output.ndim != 2 or not grad_output.is_floating_point():
        raise TypeError("grad_output must be a rank-2 floating-point tensor")
    if postload_weight.ndim != 2 or postload_weight.dtype != torch.float8_e4m3fn:
        raise TypeError("postload_weight must be rank-2 float8_e4m3fn")
    if weight_scale.dtype != torch.float32 or weight_scale.numel() != 1:
        raise TypeError("weight_scale must contain one float32 value")
    if dot_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("dot_dtype must be bfloat16, float16, or float32")
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("output_dtype must be bfloat16, float16, or float32")

    m, n = grad_output.shape
    k, weight_n = postload_weight.shape
    if n != weight_n:
        raise ValueError(f"grad_output N={n} != post-load weight N={weight_n}")
    devices = {grad_output.device, postload_weight.device, weight_scale.device}
    if len(devices) != 1:
        raise ValueError(f"all tensors must share one device, got {devices}")
    if not bool(torch.isfinite(weight_scale).all()) or float(weight_scale) <= 0:
        raise ValueError("weight_scale must be finite and positive")
    return m, n, k


def fp8_activation_saturation_mask(
    inputs: torch.Tensor,
    input_scale: torch.Tensor,
    *,
    fp8_max: float = FP8_E4M3_MAX,
) -> torch.Tensor:
    """Return the clipped-STE mask for one deployed activation scale."""

    if input_scale.dtype != torch.float32 or input_scale.numel() != 1:
        raise TypeError("input_scale must contain one float32 value")
    if input_scale.device != inputs.device:
        raise ValueError("inputs and input_scale must share one device")
    if not bool(torch.isfinite(input_scale).all()) or float(input_scale) <= 0:
        raise ValueError("input_scale must be finite and positive")
    limit = input_scale.float() * fp8_max
    return inputs.detach().float().abs() <= limit


def live_fp8_input_vjp_torch(
    grad_output: torch.Tensor,
    postload_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.float32,
    dot_dtype: torch.dtype = torch.bfloat16,
    tile_k: int = 128,
) -> torch.Tensor:
    """K-tiled Torch fallback with no full dequantized weight matrix."""

    m, n, k = _validate_inputs(
        grad_output,
        postload_weight,
        weight_scale,
        output_dtype=output_dtype,
        dot_dtype=dot_dtype,
    )
    if tile_k < 1:
        raise ValueError("tile_k must be positive")

    result = torch.empty((m, k), dtype=torch.float32, device=grad_output.device)
    scale = weight_scale.float()
    for start in range(0, k, tile_k):
        stop = min(start + tile_k, k)
        weight_out_in = postload_weight[start:stop].T.float() * scale
        if dot_dtype == torch.float32:
            partial = grad_output.float() @ weight_out_in
        else:
            left = grad_output.to(dot_dtype)
            right = weight_out_in.to(dot_dtype)
            if grad_output.is_cuda:
                partial = torch.mm(left, right, out_dtype=torch.float32)
            else:
                # Explicit FP32 accumulation makes the fallback deterministic.
                partial = left.float() @ right.float()
        result[:, start:stop] = partial
    return result.to(output_dtype)


if triton is not None:

    @triton.jit
    def _live_fp8_input_vjp_kernel(
        grad_ptr,
        weight_ptr,
        scale_ptr,
        output_ptr,
        m_size: tl.constexpr,
        n_size: tl.constexpr,
        k_size: tl.constexpr,
        stride_gm: tl.constexpr,
        stride_gn: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_om: tl.constexpr,
        stride_ok: tl.constexpr,
        use_fp16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        weight_scale = tl.load(scale_ptr).to(tl.float32)

        for n_start in tl.range(
            0, n_size, BLOCK_N, loop_unroll_factor=1
        ):
            offsets_n = n_start + tl.arange(0, BLOCK_N)
            grad = tl.load(
                grad_ptr
                + offsets_m[:, None] * stride_gm
                + offsets_n[None, :] * stride_gn,
                mask=(offsets_m[:, None] < m_size)
                & (offsets_n[None, :] < n_size),
                other=0.0,
            )
            weight = tl.load(
                weight_ptr
                + offsets_k[None, :] * stride_wk
                + offsets_n[:, None] * stride_wn,
                mask=(offsets_k[None, :] < k_size)
                & (offsets_n[:, None] < n_size),
                other=0.0,
            ).to(tl.float32)
            weight *= weight_scale

            if use_fp16:
                grad = grad.to(tl.float16)
                weight = weight.to(tl.float16)
            else:
                grad = grad.to(tl.bfloat16)
                weight = weight.to(tl.bfloat16)
            accumulator += tl.dot(grad, weight)

        tl.store(
            output_ptr
            + offsets_m[:, None] * stride_om
            + offsets_k[None, :] * stride_ok,
            accumulator,
            mask=(offsets_m[:, None] < m_size)
            & (offsets_k[None, :] < k_size),
        )


def live_fp8_input_vjp_triton(
    grad_output: torch.Tensor,
    postload_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.float32,
    dot_dtype: torch.dtype = torch.bfloat16,
    block_m: int = 16,
    block_n: int = 32,
    block_k: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    """Triton live-FP8 VJP; only the final ``[M, K]`` is materialized."""

    if triton is None:
        raise RuntimeError("Triton is not installed")
    m, n, k = _validate_inputs(
        grad_output,
        postload_weight,
        weight_scale,
        output_dtype=output_dtype,
        dot_dtype=dot_dtype,
    )
    if not grad_output.is_cuda:
        raise ValueError("Triton backend requires CUDA tensors")
    if dot_dtype == torch.float32:
        raise ValueError(
            "Triton tensor-core backend supports bfloat16/float16 dot_dtype; "
            "use backend='torch' for an FP32 dot"
        )
    for name, value in (
        ("block_m", block_m),
        ("block_n", block_n),
        ("block_k", block_k),
    ):
        if value < 16 or value & (value - 1):
            raise ValueError(f"{name} must be a power of two >= 16")
    if block_n % 16 or block_k % 16:
        raise ValueError("block_n and block_k must be multiples of 16")

    output = torch.empty((m, k), dtype=output_dtype, device=grad_output.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(k, block_k))
    _live_fp8_input_vjp_kernel[grid](
        grad_output,
        postload_weight,
        weight_scale,
        output,
        m,
        n,
        k,
        grad_output.stride(0),
        grad_output.stride(1),
        postload_weight.stride(0),
        postload_weight.stride(1),
        output.stride(0),
        output.stride(1),
        dot_dtype == torch.float16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return output


def live_fp8_input_vjp(
    grad_output: torch.Tensor,
    postload_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.float32,
    dot_dtype: torch.dtype = torch.bfloat16,
    backend: Backend = "auto",
    tile_k: int = 128,
) -> torch.Tensor:
    """Dispatch to Triton on CUDA, otherwise to the streaming fallback."""

    if backend not in ("auto", "torch", "triton"):
        raise ValueError(f"invalid backend: {backend!r}")
    selected = (
        "triton"
        if backend == "auto" and grad_output.is_cuda and triton is not None
        else "torch"
        if backend == "auto"
        else backend
    )
    if selected == "triton":
        return live_fp8_input_vjp_triton(
            grad_output,
            postload_weight,
            weight_scale,
            output_dtype=output_dtype,
            dot_dtype=dot_dtype,
        )
    return live_fp8_input_vjp_torch(
        grad_output,
        postload_weight,
        weight_scale,
        output_dtype=output_dtype,
        dot_dtype=dot_dtype,
        tile_k=tile_k,
    )


@torch.library.custom_op(
    "lumo_jlens::live_fp8_input_vjp_bf16_v1",
    mutates_args=(),
)
def live_fp8_input_vjp_op(
    grad_output: torch.Tensor,
    postload_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Opaque first-order live-FP8 VJP op for the replay autograd graph."""

    return live_fp8_input_vjp(
        grad_output,
        postload_weight,
        weight_scale,
        output_dtype=torch.float32,
        dot_dtype=torch.bfloat16,
    )


@live_fp8_input_vjp_op.register_fake
def _live_fp8_input_vjp_fake(
    grad_output: torch.Tensor,
    postload_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    del weight_scale
    return grad_output.new_empty(
        (grad_output.shape[0], postload_weight.shape[0]),
        dtype=torch.float32,
    )


@live_fp8_input_vjp_op.register_vmap
def _live_fp8_input_vjp_vmap(
    _info: Any,
    in_dims: tuple[int | None, ...],
    grad_output: torch.Tensor,
    postload_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, int | None]:
    grad_dim, weight_dim, scale_dim = in_dims
    if weight_dim is not None or scale_dim is not None:
        raise ValueError("only grad_output may be batched in the live-FP8 VJP")
    if grad_dim is None:
        return live_fp8_input_vjp_op(
            grad_output, postload_weight, weight_scale
        ), None

    batched_grad = grad_output.movedim(grad_dim, 0)
    if batched_grad.ndim != 3:
        raise ValueError(
            "the live-FP8 VJP batching rule expects logical [M, N] inputs"
        )
    batch, rows, n = batched_grad.shape
    flattened = batched_grad.reshape(batch * rows, n)
    flattened_result = live_fp8_input_vjp_op(
        flattened, postload_weight, weight_scale
    )
    return flattened_result.reshape(batch, rows, postload_weight.shape[0]), 0


class _ExactLiveFp8Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        exact_output: torch.Tensor,
        postload_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor,
        clip_saturation: bool,
    ) -> torch.Tensor:
        if inputs.ndim < 1 or not inputs.is_floating_point():
            raise TypeError("inputs must be a floating-point tensor")
        if exact_output.ndim != inputs.ndim or not exact_output.is_floating_point():
            raise TypeError("exact_output must be floating point with input rank")
        if postload_weight.ndim != 2:
            raise TypeError("postload_weight must be rank two")
        k, n = postload_weight.shape
        if inputs.shape[-1] != k:
            raise ValueError(f"input K={inputs.shape[-1]} != post-load weight K={k}")
        expected_output_shape = (*inputs.shape[:-1], n)
        if tuple(exact_output.shape) != expected_output_shape:
            raise ValueError(
                f"exact_output shape {tuple(exact_output.shape)} "
                f"!= {expected_output_shape}"
            )
        _validate_inputs(
            exact_output.reshape(-1, n),
            postload_weight,
            weight_scale,
            output_dtype=torch.float32,
            dot_dtype=torch.bfloat16,
        )
        if inputs.device != exact_output.device:
            raise ValueError("inputs and exact_output must share one device")
        mask = (
            fp8_activation_saturation_mask(inputs, input_scale)
            if clip_saturation
            else torch.empty(0, dtype=torch.bool, device=inputs.device)
        )

        ctx.save_for_backward(postload_weight, weight_scale, mask)
        ctx.clip_saturation = clip_saturation
        ctx.input_shape = tuple(inputs.shape)
        ctx.input_dtype = inputs.dtype
        ctx.output_features = n
        return exact_output

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        postload_weight, weight_scale, mask = ctx.saved_tensors
        flat_grad = grad_output.reshape(-1, ctx.output_features)
        flat_input_grad = live_fp8_input_vjp_op(
            flat_grad, postload_weight, weight_scale
        )
        input_grad = flat_input_grad.reshape(ctx.input_shape).to(ctx.input_dtype)
        if ctx.clip_saturation:
            input_grad = input_grad * mask.to(ctx.input_dtype)
        return input_grad, None, None, None, None, None


def exact_live_fp8_linear(
    inputs: torch.Tensor,
    exact_output: torch.Tensor,
    postload_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    *,
    ste_policy: StePolicy = "identity",
) -> torch.Tensor:
    """Return the captured output and use a live-FP8 surrogate backward."""

    if ste_policy not in ("identity", "clipped"):
        raise ValueError("ste_policy must be 'identity' or 'clipped'")
    return _ExactLiveFp8Linear.apply(
        inputs,
        exact_output,
        postload_weight,
        weight_scale,
        input_scale,
        ste_policy == "clipped",
    )


def resource_estimate(
    *,
    m: int,
    n: int,
    k: int,
    tile_k: int = 128,
    output_element_size: int = 4,
) -> dict[str, int]:
    """Theoretical tensor bytes, excluding framework/kernel workspaces."""

    if min(m, n, k, tile_k) < 1:
        raise ValueError("dimensions must be positive")
    return {
        "live_fp8_weight_bytes": k * n,
        "live_weight_scale_bytes": 4,
        "grad_output_bytes_bf16": m * n * 2,
        "grad_input_bytes": m * k * output_element_size,
        "full_dequantized_weight_bytes_bf16_avoided": n * k * 2,
        "full_dequantized_weight_bytes_fp32_avoided": n * k * 4,
        "torch_fallback_max_weight_tile_bytes_fp32": min(tile_k, k) * n * 4,
        "triton_bf16_weight_tile_bytes_per_program": 32 * 64 * 2,
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    difference = actual.float() - expected.float()
    expected_rms = float(expected.float().square().mean().sqrt().item())
    rms = float(difference.square().mean().sqrt().item())
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": float(difference.abs().max().item()),
        "rms": rms,
        "reference_rms": expected_rms,
        "relative_rms": rms / max(expected_rms, 1e-30),
        "actual_sha256": _tensor_sha256(actual),
        "reference_sha256": _tensor_sha256(expected),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-real-layer62", action="store_true")
    parser.add_argument("--weight-capture", type=Path, default=DEFAULT_WEIGHT_CAPTURE)
    parser.add_argument("--forward-capture", type=Path, default=DEFAULT_FORWARD_CAPTURE)
    parser.add_argument("--module", default=DEFAULT_MODULE)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batched-probes", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _real_layer62_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("real layer62 probe requires CUDA")
    try:
        from scripts.nvfp4_ste import (
            dequantize_runtime_fp8_weight,
            exact_fp8_linear_ste,
        )
    except ModuleNotFoundError:
        from nvfp4_ste import dequantize_runtime_fp8_weight, exact_fp8_linear_ste

    weight_capture_path = args.weight_capture.resolve()
    forward_capture_path = args.forward_capture.resolve()
    started = datetime.now(timezone.utc)
    load_started = time.perf_counter()
    weight_payload = torch.load(
        weight_capture_path, map_location="cpu", weights_only=True, mmap=True
    )
    forward_payload = torch.load(
        forward_capture_path, map_location="cpu", weights_only=True, mmap=True
    )
    if weight_payload.get("model_revision") != forward_payload.get("model_revision"):
        raise ValueError("weight and forward captures have different model revisions")
    weight_tensors = weight_payload["tensors"]
    forward_tensors = forward_payload["tensors"]
    replay_prefix = f"replay.fp8.{args.module}"
    linear_prefix = f"linear.{args.module}"
    postload_cpu = weight_tensors[f"{replay_prefix}.weight"]
    weight_scale_cpu = weight_tensors[f"{replay_prefix}.weight_scale"]
    input_scale_cpu = weight_tensors[f"{replay_prefix}.input_scale"]
    captured_input_cpu = forward_tensors[f"{linear_prefix}.input"]
    captured_output_cpu = forward_tensors[f"{linear_prefix}.output"]
    cpu_load_seconds = time.perf_counter() - load_started

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    postload_weight = postload_cpu.to(device)
    weight_scale = weight_scale_cpu.to(device)
    input_scale = input_scale_cpu.to(device)
    k, n = postload_weight.shape
    generator = torch.Generator().manual_seed(20260717)
    grad_output = torch.randn(args.m, n, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )

    for _ in range(args.warmup):
        warmup_output = live_fp8_input_vjp_triton(
            grad_output, postload_weight, weight_scale
        )
    torch.cuda.synchronize()
    del warmup_output

    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    triton_times: list[float] = []
    actual = None
    for _ in range(args.repeats):
        if actual is not None:
            del actual
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        actual = live_fp8_input_vjp_triton(
            grad_output, postload_weight, weight_scale
        )
        end.record()
        end.synchronize()
        triton_times.append(begin.elapsed_time(end))
    assert actual is not None
    triton_peak_allocated = torch.cuda.max_memory_allocated()
    triton_peak_reserved = torch.cuda.max_memory_reserved()

    reference_baseline_allocated = torch.cuda.memory_allocated()
    reference_baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    reference_begin = torch.cuda.Event(enable_timing=True)
    reference_end = torch.cuda.Event(enable_timing=True)
    reference_begin.record()
    dense_weight = dequantize_runtime_fp8_weight(
        postload_weight,
        weight_scale,
        transposed=True,
        dtype=torch.bfloat16,
    )
    reference = torch.mm(grad_output, dense_weight, out_dtype=torch.float32)
    reference_end.record()
    reference_end.synchronize()
    reference_ms = reference_begin.elapsed_time(reference_end)
    reference_peak_allocated = torch.cuda.max_memory_allocated()
    reference_peak_reserved = torch.cuda.max_memory_reserved()
    direct_metrics = _error_metrics(actual, reference)
    direct_metrics["maximum_relative_rms"] = MAX_REAL_PROBE_RELATIVE_RMS
    direct_metrics["passed"] = (
        direct_metrics["finite"]
        and direct_metrics["relative_rms"] <= MAX_REAL_PROBE_RELATIVE_RMS
    )

    captured_input = captured_input_cpu.to(device).detach().requires_grad_(True)
    captured_output = captured_output_cpu.to(device)
    cotangents = torch.randn(
        args.batched_probes,
        *captured_output.shape,
        generator=generator,
    ).to(device=device, dtype=torch.bfloat16)
    wrapper_metrics: dict[str, Any] = {}
    for policy in ("identity", "clipped"):
        packed_input = captured_input.detach().clone().requires_grad_(True)
        dense_input = captured_input.detach().clone().requires_grad_(True)
        packed_forward = exact_live_fp8_linear(
            packed_input,
            captured_output,
            postload_weight,
            weight_scale,
            input_scale,
            ste_policy=policy,
        )
        dense_forward = exact_fp8_linear_ste(
            dense_input,
            captured_output,
            dense_weight,
            input_scale,
            ste_policy=policy,
        )
        (packed_gradient,) = torch.autograd.grad(
            packed_forward,
            packed_input,
            grad_outputs=cotangents,
            is_grads_batched=True,
        )
        (dense_gradient,) = torch.autograd.grad(
            dense_forward,
            dense_input,
            grad_outputs=cotangents,
            is_grads_batched=True,
        )
        metrics = _error_metrics(packed_gradient, dense_gradient)
        metrics.update(
            {
                "maximum_relative_rms": MAX_REAL_WRAPPER_RELATIVE_RMS,
                "forward_bitwise_exact": bool(torch.equal(packed_forward, captured_output)),
                "gradient_dtype": str(packed_gradient.dtype),
                "gradient_shape": list(packed_gradient.shape),
            }
        )
        metrics["passed"] = (
            metrics["finite"]
            and metrics["forward_bitwise_exact"]
            and metrics["gradient_dtype"] == str(captured_input.dtype)
            and metrics["relative_rms"] <= MAX_REAL_WRAPPER_RELATIVE_RMS
        )
        wrapper_metrics[policy] = metrics

    saturation_mask = fp8_activation_saturation_mask(
        captured_input.detach(), input_scale
    )
    finite_input = torch.isfinite(captured_input.detach())
    estimate = resource_estimate(m=args.m, n=n, k=k, tile_k=args.tile_k)
    passed = bool(direct_metrics["passed"]) and all(
        bool(metrics["passed"]) for metrics in wrapper_metrics.values()
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": triton.__version__ if triton is not None else None,
            "cuda_device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "captures": {
            "model_revision": weight_payload.get("model_revision"),
            "weight_capture": str(weight_capture_path),
            "weight_capture_sha256": _file_sha256(weight_capture_path),
            "forward_capture": str(forward_capture_path),
            "forward_capture_sha256": _file_sha256(forward_capture_path),
            "module": args.module,
            "cpu_load_seconds": cpu_load_seconds,
            "postload_weight_shape": list(postload_cpu.shape),
            "postload_weight_dtype": str(postload_cpu.dtype),
            "weight_scale": float(weight_scale_cpu),
            "input_scale": float(input_scale_cpu),
            "captured_input_shape": list(captured_input_cpu.shape),
            "captured_output_shape": list(captured_output_cpu.shape),
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "problem": {"m": args.m, "n": n, "k": k},
        "contract": {
            "runtime_weight_orientation": "[K, N]",
            "dot_dtype": "torch.bfloat16",
            "accumulator_dtype": "torch.float32",
            "output_dtype": "torch.float32",
            "full_weight_materialized_by_live_fp8_kernel": False,
            "full_weight_materialized_by_validation_reference": True,
            "dense_validation_weight_dtype": "torch.bfloat16",
            "activation_ste_policies": ["identity", "clipped"],
        },
        "numerics_vs_dense_bf16_dequant": direct_metrics,
        "captured_exact_forward_batched_autograd": wrapper_metrics,
        "activation_saturation": {
            "finite_values": int(finite_input.sum().item()),
            "nonfinite_values": int((~finite_input).sum().item()),
            "clipped_values": int((finite_input & ~saturation_mask).sum().item()),
            "fraction_clipped": float(
                (finite_input & ~saturation_mask).sum().item()
                / max(1, finite_input.sum().item())
            ),
            "fp8_limit": float(input_scale_cpu) * FP8_E4M3_MAX,
            "max_abs": float(captured_input_cpu.float().abs().max().item()),
        },
        "performance": {
            "triton_ms": triton_times,
            "triton_median_ms": sorted(triton_times)[len(triton_times) // 2],
            "cuda_allocated_before_output_bytes": baseline_allocated,
            "cuda_reserved_before_output_bytes": baseline_reserved,
            "triton_peak_allocated_bytes": triton_peak_allocated,
            "triton_peak_allocated_delta_bytes": (
                triton_peak_allocated - baseline_allocated
            ),
            "triton_peak_reserved_bytes": triton_peak_reserved,
            "triton_peak_reserved_delta_bytes": triton_peak_reserved - baseline_reserved,
            "dense_reference_ms": reference_ms,
            "dense_reference_baseline_allocated_bytes": reference_baseline_allocated,
            "dense_reference_baseline_reserved_bytes": reference_baseline_reserved,
            "dense_reference_peak_allocated_bytes": reference_peak_allocated,
            "dense_reference_peak_allocated_delta_bytes": (
                reference_peak_allocated - reference_baseline_allocated
            ),
            "dense_reference_peak_reserved_bytes": reference_peak_reserved,
            "dense_reference_peak_reserved_delta_bytes": (
                reference_peak_reserved - reference_baseline_reserved
            ),
        },
        "theoretical_resources": estimate,
    }


def main() -> None:
    args = _parse_args()
    if not args.probe_real_layer62:
        raise SystemExit("pass --probe-real-layer62 to run the captured real-layer probe")
    if min(args.m, args.tile_k, args.warmup, args.repeats, args.batched_probes) < 1:
        raise ValueError(
            "m, tile-k, warmup, repeats, and batched-probes must be positive"
        )
    result = _real_layer62_probe(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
