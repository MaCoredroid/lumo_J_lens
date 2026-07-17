#!/usr/bin/env python3
"""Memory-bounded input VJP for raw ModelOpt NVFP4 W4A16 weights.

Given ``grad_output[M, N]`` and a raw checkpoint weight stored as packed E2M1
``[N, K/2]`` plus FP8 block scales ``[N, K/16]``, compute
``grad_input = grad_output @ dequantized_weight`` without materializing the
full ``[N, K]`` weight.  The Triton path decodes a ``[BLOCK_N, BLOCK_K]`` tile
inside each program; the Torch fallback materializes at most ``tile_n`` rows.

This is a surrogate VJP for the deployed W4A16 forward.  The default dot dtype
is BF16, matching the A16 serving path, with FP32 accumulation/output.
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
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
DEFAULT_MODULE = "model.language_model.layers.62.mlp.down_proj"
SCHEMA_VERSION = 1
MAX_REAL_PROBE_RELATIVE_RMS = 1e-5


def _validate_inputs(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    block_size: int,
    output_dtype: torch.dtype,
    dot_dtype: torch.dtype,
) -> tuple[int, int, int]:
    if grad_output.ndim != 2 or not grad_output.is_floating_point():
        raise TypeError("grad_output must be a rank-2 floating-point tensor")
    if packed_weight.ndim != 2 or packed_weight.dtype != torch.uint8:
        raise TypeError("packed_weight must be a rank-2 uint8 tensor")
    if block_scales.ndim != 2 or block_scales.dtype != torch.float8_e4m3fn:
        raise TypeError("block_scales must be rank-2 float8_e4m3fn")
    if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
        raise TypeError("global_scale must contain one float32 value")
    if block_size <= 0 or block_size % 2:
        raise ValueError("block_size must be a positive even integer")
    if dot_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("dot_dtype must be bfloat16, float16, or float32")
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("output_dtype must be bfloat16, float16, or float32")

    m, n = grad_output.shape
    weight_n, packed_k = packed_weight.shape
    k = packed_k * 2
    if n != weight_n:
        raise ValueError(f"grad_output N={n} != packed weight N={weight_n}")
    if k % block_size:
        raise ValueError(f"K={k} must be divisible by block_size={block_size}")
    expected_scales = (n, k // block_size)
    if tuple(block_scales.shape) != expected_scales:
        raise ValueError(
            f"block_scales shape {tuple(block_scales.shape)} != {expected_scales}"
        )
    devices = {
        grad_output.device,
        packed_weight.device,
        block_scales.device,
        global_scale.device,
    }
    if len(devices) != 1:
        raise ValueError(f"all tensors must share one device, got {devices}")
    if not bool(torch.isfinite(global_scale).all()) or float(global_scale) <= 0:
        raise ValueError("global_scale must be finite and positive")
    return m, n, k


def _decode_e2m1_rows(packed: torch.Tensor) -> torch.Tensor:
    """Decode low-nibble-first E2M1 rows to FP32."""
    low = packed & 0x0F
    high = packed >> 4
    nibbles = torch.stack((low, high), dim=-1).reshape(
        packed.shape[0], packed.shape[1] * 2
    )
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    magnitude = table[(nibbles & 0x07).long()]
    sign = torch.where((nibbles & 0x08) != 0, -1.0, 1.0)
    return magnitude * sign


def packed_nvfp4_input_vjp_torch(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    block_size: int = 16,
    output_dtype: torch.dtype = torch.float32,
    dot_dtype: torch.dtype = torch.bfloat16,
    tile_n: int = 128,
) -> torch.Tensor:
    """Streaming Torch reference/fallback with no full weight materialization."""
    m, n, k = _validate_inputs(
        grad_output,
        packed_weight,
        block_scales,
        global_scale,
        block_size=block_size,
        output_dtype=output_dtype,
        dot_dtype=dot_dtype,
    )
    if tile_n < 1:
        raise ValueError("tile_n must be positive")

    accumulator = torch.zeros((m, k), dtype=torch.float32, device=grad_output.device)
    global_fp32 = global_scale.float()
    for start in range(0, n, tile_n):
        stop = min(start + tile_n, n)
        values = _decode_e2m1_rows(packed_weight[start:stop])
        scales = block_scales[start:stop].float() * global_fp32
        weight = (
            values.reshape(stop - start, -1, block_size)
            * scales.unsqueeze(-1)
        ).reshape(stop - start, k)

        if dot_dtype == torch.float32:
            partial = grad_output[:, start:stop].float() @ weight
        else:
            left = grad_output[:, start:stop].to(dot_dtype)
            right = weight.to(dot_dtype)
            if grad_output.is_cuda:
                partial = torch.mm(left, right, out_dtype=torch.float32)
            else:
                # CPU BF16/FP16 matmul may accumulate or round differently by
                # backend.  Explicit FP32 gives a deterministic fallback.
                partial = left.float() @ right.float()
        accumulator.add_(partial)
    return accumulator.to(output_dtype)


if triton is not None:

    @triton.jit
    def _nvfp4_packed_input_vjp_kernel(
        grad_ptr,
        packed_ptr,
        scale_ptr,
        global_ptr,
        output_ptr,
        m_size: tl.constexpr,
        n_size: tl.constexpr,
        k_size: tl.constexpr,
        stride_gm: tl.constexpr,
        stride_gn: tl.constexpr,
        stride_pn: tl.constexpr,
        stride_pk: tl.constexpr,
        stride_sn: tl.constexpr,
        stride_sk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_ok: tl.constexpr,
        block_size: tl.constexpr,
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
        global_scale = tl.load(global_ptr).to(tl.float32)

        # Keep N as a device loop.  A Python ``range`` sees constexpr N and
        # unrolls every reduction tile into the generated kernel.
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

            packed = tl.load(
                packed_ptr
                + offsets_n[:, None] * stride_pn
                + (offsets_k[None, :] // 2) * stride_pk,
                mask=(offsets_n[:, None] < n_size)
                & (offsets_k[None, :] < k_size),
                other=0,
            ).to(tl.int32)
            nibble = tl.where(
                (offsets_k[None, :] & 1) == 0,
                packed & 0x0F,
                (packed >> 4) & 0x0F,
            )
            code = nibble & 0x07
            magnitude = tl.where(
                code <= 4,
                code.to(tl.float32) * 0.5,
                tl.where(code == 5, 3.0, tl.where(code == 6, 4.0, 6.0)),
            )
            signed = tl.where((nibble & 0x08) != 0, -magnitude, magnitude)
            scale = tl.load(
                scale_ptr
                + offsets_n[:, None] * stride_sn
                + (offsets_k[None, :] // block_size) * stride_sk,
                mask=(offsets_n[:, None] < n_size)
                & (offsets_k[None, :] < k_size),
                other=0.0,
            ).to(tl.float32)
            weight = signed * scale * global_scale

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


def packed_nvfp4_input_vjp_triton(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    block_size: int = 16,
    output_dtype: torch.dtype = torch.float32,
    dot_dtype: torch.dtype = torch.bfloat16,
    block_m: int = 16,
    block_n: int = 32,
    block_k: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    """Triton packed VJP; only the final ``[M, K]`` output is materialized."""
    if triton is None:
        raise RuntimeError("Triton is not installed")
    m, n, k = _validate_inputs(
        grad_output,
        packed_weight,
        block_scales,
        global_scale,
        block_size=block_size,
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
    _nvfp4_packed_input_vjp_kernel[grid](
        grad_output,
        packed_weight,
        block_scales,
        global_scale,
        output,
        m,
        n,
        k,
        grad_output.stride(0),
        grad_output.stride(1),
        packed_weight.stride(0),
        packed_weight.stride(1),
        block_scales.stride(0),
        block_scales.stride(1),
        output.stride(0),
        output.stride(1),
        block_size,
        dot_dtype == torch.float16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return output


def packed_nvfp4_input_vjp(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    block_size: int = 16,
    output_dtype: torch.dtype = torch.float32,
    dot_dtype: torch.dtype = torch.bfloat16,
    backend: Backend = "auto",
    tile_n: int = 128,
) -> torch.Tensor:
    """Dispatch to Triton on CUDA, otherwise to the streaming Torch fallback."""
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
        return packed_nvfp4_input_vjp_triton(
            grad_output,
            packed_weight,
            block_scales,
            global_scale,
            block_size=block_size,
            output_dtype=output_dtype,
            dot_dtype=dot_dtype,
        )
    return packed_nvfp4_input_vjp_torch(
        grad_output,
        packed_weight,
        block_scales,
        global_scale,
        block_size=block_size,
        output_dtype=output_dtype,
        dot_dtype=dot_dtype,
        tile_n=tile_n,
    )


@torch.library.custom_op(
    "lumo_jlens::packed_nvfp4_input_vjp_bf16_v1",
    mutates_args=(),
)
def packed_nvfp4_input_vjp_op(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Opaque first-order packed VJP op used by the autograd replay graph."""

    return packed_nvfp4_input_vjp(
        grad_output,
        packed_weight,
        block_scales,
        global_scale,
        block_size=block_size,
        output_dtype=torch.float32,
        dot_dtype=torch.bfloat16,
    )


@packed_nvfp4_input_vjp_op.register_fake
def _packed_nvfp4_input_vjp_fake(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    del block_scales, global_scale, block_size
    return grad_output.new_empty(
        (grad_output.shape[0], packed_weight.shape[1] * 2),
        dtype=torch.float32,
    )


@packed_nvfp4_input_vjp_op.register_vmap
def _packed_nvfp4_input_vjp_vmap(
    _info: Any,
    in_dims: tuple[int | None, ...],
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, int | None]:
    grad_dim, packed_dim, scales_dim, global_dim, block_dim = in_dims
    if any(
        dim is not None
        for dim in (packed_dim, scales_dim, global_dim, block_dim)
    ):
        raise ValueError("only grad_output may be batched in the packed VJP")
    if grad_dim is None:
        return (
            packed_nvfp4_input_vjp_op(
                grad_output,
                packed_weight,
                block_scales,
                global_scale,
                block_size,
            ),
            None,
        )

    batched_grad = grad_output.movedim(grad_dim, 0)
    if batched_grad.ndim != 3:
        raise ValueError(
            "the packed VJP batching rule expects logical [M, N] inputs"
        )
    batch, rows, n = batched_grad.shape
    flattened = batched_grad.reshape(batch * rows, n)
    flattened_result = packed_nvfp4_input_vjp_op(
        flattened,
        packed_weight,
        block_scales,
        global_scale,
        block_size,
    )
    result = flattened_result.reshape(
        batch, rows, packed_weight.shape[1] * 2
    )
    return result, 0


class _ExactPackedNvFp4Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        exact_output: torch.Tensor,
        packed_weight: torch.Tensor,
        block_scales: torch.Tensor,
        global_scale: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        if inputs.ndim < 1 or not inputs.is_floating_point():
            raise TypeError("inputs must be a floating-point tensor")
        if exact_output.ndim != inputs.ndim or not exact_output.is_floating_point():
            raise TypeError("exact_output must be floating point with input rank")
        n, packed_k = packed_weight.shape
        k = packed_k * 2
        if inputs.shape[-1] != k:
            raise ValueError(f"input K={inputs.shape[-1]} != packed weight K={k}")
        expected_output_shape = (*inputs.shape[:-1], n)
        if tuple(exact_output.shape) != expected_output_shape:
            raise ValueError(
                f"exact_output shape {tuple(exact_output.shape)} "
                f"!= {expected_output_shape}"
            )
        _validate_inputs(
            exact_output.reshape(-1, n),
            packed_weight,
            block_scales,
            global_scale,
            block_size=block_size,
            output_dtype=torch.float32,
            dot_dtype=torch.bfloat16,
        )
        if inputs.device != exact_output.device:
            raise ValueError("inputs and exact_output must share one device")

        ctx.save_for_backward(packed_weight, block_scales, global_scale)
        ctx.block_size = block_size
        ctx.input_shape = tuple(inputs.shape)
        ctx.input_dtype = inputs.dtype
        ctx.output_features = n
        return exact_output

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        packed_weight, block_scales, global_scale = ctx.saved_tensors
        flat_grad = grad_output.reshape(-1, ctx.output_features)
        flat_input_grad = packed_nvfp4_input_vjp_op(
            flat_grad,
            packed_weight,
            block_scales,
            global_scale,
            ctx.block_size,
        )
        input_grad = flat_input_grad.reshape(ctx.input_shape).to(ctx.input_dtype)
        return input_grad, None, None, None, None, None


def exact_packed_nvfp4_linear(
    inputs: torch.Tensor,
    exact_output: torch.Tensor,
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    block_size: int = 16,
) -> torch.Tensor:
    """Return the captured forward value and use the packed NVFP4 input VJP."""

    return _ExactPackedNvFp4Linear.apply(
        inputs,
        exact_output,
        packed_weight,
        block_scales,
        global_scale,
        block_size,
    )


def resource_estimate(
    *,
    m: int,
    n: int,
    k: int,
    tile_n: int = 128,
    output_element_size: int = 4,
    dot_element_size: int = 2,
) -> dict[str, int]:
    """Theoretical tensor bytes, excluding framework/kernel workspaces."""
    if min(m, n, k, tile_n) < 1 or k % 16:
        raise ValueError("dimensions must be positive and K divisible by 16")
    packed = n * k // 2
    scales = n * k // 16
    return {
        "raw_packed_weight_bytes": packed,
        "raw_block_scale_bytes": scales,
        "raw_global_scale_bytes": 4,
        "grad_output_bytes_bf16": m * n * 2,
        "grad_input_bytes": m * k * output_element_size,
        "full_dequantized_weight_bytes_bf16_avoided": n * k * 2,
        "full_dequantized_weight_bytes_fp32_avoided": n * k * 4,
        "torch_fallback_max_weight_tile_bytes": min(tile_n, n)
        * k
        * dot_element_size,
        "triton_weight_tile_bytes_per_program": 32 * 64 * dot_element_size,
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-real-layer62", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--module", default=DEFAULT_MODULE)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _real_layer62_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("real layer62 probe requires CUDA")
    try:
        from scripts.modelopt_checkpoint import ModelOptCheckpoint, default_pinned_snapshot
        from scripts.nvfp4_ste import dequantize_nvfp4_weight
    except ModuleNotFoundError:
        from modelopt_checkpoint import ModelOptCheckpoint, default_pinned_snapshot
        from nvfp4_ste import dequantize_nvfp4_weight

    snapshot = args.snapshot.resolve() if args.snapshot else default_pinned_snapshot()
    started = datetime.now(timezone.utc)
    load_started = time.perf_counter()
    checkpoint = ModelOptCheckpoint(snapshot)
    metadata = checkpoint.inspect_nvfp4(args.module)
    raw = checkpoint.load_nvfp4(args.module, device="cpu")
    cpu_load_seconds = time.perf_counter() - load_started

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    packed = raw.packed_weight.to(device)
    scales = raw.block_scales.to(device)
    global_scale = raw.global_scale.to(device)
    n, packed_k = packed.shape
    k = packed_k * 2
    generator = torch.Generator(device="cpu").manual_seed(20260716)
    grad = torch.randn(args.m, n, generator=generator, dtype=torch.bfloat16).to(device)

    for _ in range(args.warmup):
        warmup_output = packed_nvfp4_input_vjp_triton(
            grad, packed, scales, global_scale
        )
    torch.cuda.synchronize()
    del warmup_output

    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    triton_times: list[float] = []
    actual = None
    for _ in range(args.repeats):
        # Do not retain the preceding timed output.  This keeps the measured
        # peak at one production output tensor instead of two Python results.
        if actual is not None:
            del actual
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        actual = packed_nvfp4_input_vjp_triton(grad, packed, scales, global_scale)
        end.record()
        end.synchronize()
        triton_times.append(begin.elapsed_time(end))
    assert actual is not None
    torch.cuda.synchronize()
    triton_peak_allocated = torch.cuda.max_memory_allocated()
    triton_peak_reserved = torch.cuda.max_memory_reserved()
    triton_peak_allocated_delta = triton_peak_allocated - baseline_allocated
    triton_peak_reserved_delta = triton_peak_reserved - baseline_reserved

    # Materialize a dense BF16 weight only after packed-kernel memory
    # measurement.  This is validation machinery, not part of the VJP path.
    reference_baseline_allocated = torch.cuda.memory_allocated()
    reference_baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    reference_begin = torch.cuda.Event(enable_timing=True)
    reference_end = torch.cuda.Event(enable_timing=True)
    reference_begin.record()
    dense_weight = dequantize_nvfp4_weight(
        packed,
        scales,
        global_scale,
        block_size=raw.block_size,
        dtype=torch.bfloat16,
    )
    reference = torch.mm(grad, dense_weight, out_dtype=torch.float32)
    reference_end.record()
    reference_end.synchronize()
    reference_ms = reference_begin.elapsed_time(reference_end)
    reference_peak_allocated = torch.cuda.max_memory_allocated()
    reference_peak_reserved = torch.cuda.max_memory_reserved()
    difference = actual.float() - reference.float()
    reference_rms = float(reference.float().square().mean().sqrt().item())
    rms = float(difference.square().mean().sqrt().item())
    relative_rms = rms / max(reference_rms, 1e-30)
    finite = bool(torch.isfinite(actual).all())
    numerics_passed = finite and relative_rms <= MAX_REAL_PROBE_RELATIVE_RMS
    estimate = resource_estimate(m=args.m, n=n, k=k, tile_n=args.tile_n)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if numerics_passed else "failed",
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
        "checkpoint": {
            "snapshot": str(snapshot),
            "module": args.module,
            "components": list(raw.components),
            "source_shards": list(raw.source_shards),
            "packed_shape": list(metadata.packed_shape),
            "block_scale_shape": list(metadata.block_scale_shape),
            "block_size": raw.block_size,
            "cpu_load_seconds": cpu_load_seconds,
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "problem": {"m": args.m, "n": n, "k": k},
        "contract": {
            "dot_dtype": "torch.bfloat16",
            "accumulator_dtype": "torch.float32",
            "output_dtype": "torch.float32",
            "full_weight_materialized_by_packed_kernel": False,
            "full_weight_materialized_by_validation_reference": True,
            "dense_validation_weight_dtype": "torch.bfloat16",
        },
        "numerics_vs_dense_bf16_dequant": {
            "passed": numerics_passed,
            "finite": finite,
            "maximum_relative_rms": MAX_REAL_PROBE_RELATIVE_RMS,
            "max_abs": float(difference.abs().max().item()),
            "rms": rms,
            "reference_rms": reference_rms,
            "relative_rms": relative_rms,
            "actual_sha256": _tensor_sha256(actual),
            "reference_sha256": _tensor_sha256(reference),
        },
        "performance": {
            "triton_ms": triton_times,
            "triton_median_ms": sorted(triton_times)[len(triton_times) // 2],
            "cuda_allocated_before_output_bytes": baseline_allocated,
            "cuda_reserved_before_output_bytes": baseline_reserved,
            "triton_peak_allocated_bytes": triton_peak_allocated,
            "triton_peak_allocated_delta_bytes": triton_peak_allocated_delta,
            "triton_peak_reserved_bytes": triton_peak_reserved,
            "triton_peak_reserved_delta_bytes": triton_peak_reserved_delta,
            "dense_reference_ms": reference_ms,
            "dense_reference_baseline_allocated_bytes": (
                reference_baseline_allocated
            ),
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
        raise SystemExit("pass --probe-real-layer62 to run the pinned real-weight probe")
    if min(args.m, args.tile_n, args.warmup, args.repeats) < 1:
        raise ValueError("m, tile-n, warmup, and repeats must be positive")
    result = _real_layer62_probe(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
