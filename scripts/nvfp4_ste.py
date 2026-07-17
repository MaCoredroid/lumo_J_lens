#!/usr/bin/env python3
"""Surrogate-backward primitives for the deployed Qwen3.6 NVFP4 graph.

The forward values used by the fitter come from instrumented vLLM kernels.
This module defines the smooth backward contract applied to those values:

* W4A16 linears use the frozen, dequantized ModelOpt NVFP4 weight.
* Static FP8 linears use the post-load/requantized effective weight and an
  explicitly selected straight-through activation derivative.
* Gated DeltaNet uses an explicit reverse recurrence in FP32.

These derivatives are a declared surrogate for quantized inference, not the
literal derivative of rounding.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F


E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP8_E4M3_MAX = 448.0


def decode_e2m1_bytes(packed: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Decode low-nibble-first packed E2M1 values without applying scales."""

    if packed.dtype != torch.uint8 or packed.ndim < 1:
        raise TypeError("packed E2M1 tensor must be uint8")
    flat = packed.reshape(-1)
    nibbles = torch.stack((flat & 0x0F, flat >> 4), dim=-1).reshape(-1)
    table = torch.tensor(E2M1_VALUES, device=packed.device, dtype=torch.float32)
    magnitudes = table[(nibbles & 0x07).long()]
    signs = torch.where((nibbles & 0x08) != 0, -1.0, 1.0)
    decoded = magnitudes * signs
    return decoded.reshape(*packed.shape[:-1], packed.shape[-1] * 2).to(dtype)


def dequantize_nvfp4_weight(
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    block_size: int = 16,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a raw, unswizzled ModelOpt W4A16 weight tensor.

    ``packed_weight`` is ``[N, K/2]`` with two E2M1 values per byte,
    ``block_scales`` is ``[N, K/block_size]`` FP8 E4M3, and
    ``global_scale`` is the raw ModelOpt ``weight_scale_2``.  This function
    intentionally does not accept vLLM's post-load Marlin-repacked tensors.
    """

    if packed_weight.ndim != 2 or packed_weight.dtype != torch.uint8:
        raise TypeError("packed_weight must be a rank-2 uint8 tensor")
    if block_size <= 0 or block_size % 2:
        raise ValueError("block_size must be a positive even integer")
    n_rows, packed_k = packed_weight.shape
    k = packed_k * 2
    expected_scales = (n_rows, k // block_size)
    if k % block_size or tuple(block_scales.shape) != expected_scales:
        raise ValueError(
            f"block_scales shape {tuple(block_scales.shape)} != {expected_scales}"
        )
    if global_scale.numel() != 1:
        raise ValueError("global_scale must contain exactly one value")

    values = decode_e2m1_bytes(packed_weight, dtype=torch.float32)
    scales = block_scales.to(torch.float32) * global_scale.to(torch.float32)
    values = values.reshape(n_rows, -1, block_size)
    return (values * scales.unsqueeze(-1)).reshape(n_rows, k).to(dtype)


def dequantize_runtime_fp8_weight(
    postload_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    transposed: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the effective ``[out, in]`` weight used by a vLLM FP8 linear.

    ModelOpt's post-load path may requantize fused shards to a common maximum
    scale.  The live tensor is therefore authoritative; raw checkpoint shard
    scales are not interchangeable with it.
    """

    if postload_weight.dtype != torch.float8_e4m3fn or postload_weight.ndim != 2:
        raise TypeError("postload_weight must be rank-2 float8_e4m3fn")
    if weight_scale.numel() != 1:
        raise ValueError("weight_scale must contain exactly one value")
    weight = postload_weight.T if transposed else postload_weight
    return (weight.to(torch.float32) * weight_scale.to(torch.float32)).to(dtype)


def fp8_saturation_mask(
    inputs: torch.Tensor,
    input_scale: torch.Tensor,
    *,
    fp8_max: float = FP8_E4M3_MAX,
) -> torch.Tensor:
    """Mask for the declared clipped STE through static FP8 activation quant."""

    if input_scale.numel() != 1 or not bool(torch.isfinite(input_scale).all()):
        raise ValueError("input_scale must be one finite scalar")
    scale = input_scale.to(device=inputs.device, dtype=torch.float32)
    if float(scale) <= 0:
        raise ValueError("input_scale must be positive")
    return inputs.detach().float().abs() <= scale * fp8_max


class _ExactLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        inputs: torch.Tensor,
        exact_output: torch.Tensor,
        weight_out_in: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.shape[:-1] != exact_output.shape[:-1]:
            raise ValueError("linear input and exact output leading shapes differ")
        if weight_out_in.shape != (exact_output.shape[-1], inputs.shape[-1]):
            raise ValueError("weight shape does not match linear input/output")
        ctx.input_dtype = inputs.dtype
        ctx.save_for_backward(weight_out_in)
        return exact_output

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor):
        (weight_out_in,) = ctx.saved_tensors
        grad_input = torch.matmul(
            grad_output.to(weight_out_in.dtype), weight_out_in
        )
        return grad_input.to(ctx.input_dtype), None, None


class _ExactFp8LinearSTE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        inputs: torch.Tensor,
        exact_output: torch.Tensor,
        weight_out_in: torch.Tensor,
        input_scale: torch.Tensor,
        clip_saturation: bool,
    ) -> torch.Tensor:
        if inputs.shape[:-1] != exact_output.shape[:-1]:
            raise ValueError("linear input and exact output leading shapes differ")
        if weight_out_in.shape != (exact_output.shape[-1], inputs.shape[-1]):
            raise ValueError("weight shape does not match linear input/output")
        mask = (
            fp8_saturation_mask(inputs, input_scale)
            if clip_saturation
            else torch.ones_like(inputs, dtype=torch.bool)
        )
        ctx.input_dtype = inputs.dtype
        ctx.save_for_backward(weight_out_in, mask)
        return exact_output

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor):
        weight_out_in, mask = ctx.saved_tensors
        grad_input = torch.matmul(
            grad_output.to(weight_out_in.dtype), weight_out_in
        )
        grad_input = grad_input * mask.to(grad_input.dtype)
        return grad_input.to(ctx.input_dtype), None, None, None, None


class _ExactValue(torch.autograd.Function):
    @staticmethod
    def forward(
        _ctx: object,
        surrogate_value: torch.Tensor,
        exact_value: torch.Tensor,
    ) -> torch.Tensor:
        if surrogate_value.shape != exact_value.shape:
            raise ValueError("surrogate and exact values must have the same shape")
        return exact_value

    @staticmethod
    def backward(_ctx: object, grad_output: torch.Tensor):
        return grad_output, None


def exact_frozen_linear(
    inputs: torch.Tensor,
    exact_output: torch.Tensor,
    weight_out_in: torch.Tensor,
) -> torch.Tensor:
    """Use an exact captured output and a frozen effective-weight VJP."""

    return _ExactLinear.apply(inputs, exact_output, weight_out_in)


def exact_w4a16_linear(
    inputs: torch.Tensor,
    exact_output: torch.Tensor,
    weight_out_in: torch.Tensor,
) -> torch.Tensor:
    """Use exact Marlin output and the dequantized NVFP4 VJP weight."""

    return exact_frozen_linear(inputs, exact_output, weight_out_in)


def exact_fp8_linear_ste(
    inputs: torch.Tensor,
    exact_output: torch.Tensor,
    weight_out_in: torch.Tensor,
    input_scale: torch.Tensor,
    *,
    ste_policy: str = "identity",
) -> torch.Tensor:
    """Use exact Cutlass output and the named FP8 STE in backward.

    ``identity`` treats rounding, cast, and saturation as identity for every
    finite input. ``clipped`` treats rounding/cast as identity but returns
    zero derivative outside ``+-448 * input_scale``.  Production fits must
    record this value and the measured saturation count.
    """

    if ste_policy not in {"identity", "clipped"}:
        raise ValueError("ste_policy must be 'identity' or 'clipped'")

    return _ExactFp8LinearSTE.apply(
        inputs,
        exact_output,
        weight_out_in,
        input_scale,
        ste_policy == "clipped",
    )


def exact_value(
    surrogate_value: torch.Tensor,
    captured_exact_value: torch.Tensor,
) -> torch.Tensor:
    """Return a captured kernel value while differentiating the surrogate."""

    return _ExactValue.apply(surrogate_value, captured_exact_value)


def rms_norm(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    gemma_offset: bool = False,
) -> torch.Tensor:
    """Differentiable FP32-reduction RMSNorm matching vLLM conventions."""

    x = inputs.float()
    w = weight.float() + (1.0 if gemma_offset else 0.0)
    variance = x.square().mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps) * w).to(inputs.dtype)


def gated_rms_norm(
    inputs: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Qwen GDN norm-before-gate: ``RMSNorm(x) * SiLU(gate)``."""

    normalized = rms_norm(inputs, weight, eps)
    return (normalized.float() * F.silu(gate.float())).to(inputs.dtype)


def causal_depthwise_conv1d_silu(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    initial_tail: torch.Tensor | None = None,
) -> torch.Tensor:
    """Width-K causal depthwise convolution followed by SiLU.

    ``inputs`` is ``[T, channels]`` and ``weight`` is either
    ``[channels, K]`` or vLLM's ``[channels, 1, K]``.  ``initial_tail`` may
    provide the preceding ``K-1`` rows; a fresh prefill uses zeros.
    """

    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [tokens, channels]")
    if weight.ndim == 3:
        if weight.shape[1] != 1:
            raise ValueError("rank-3 depthwise weight must have singleton axis 1")
        weight = weight[:, 0]
    if weight.ndim != 2 or weight.shape[0] != inputs.shape[1]:
        raise ValueError("weight must have shape [channels, kernel_width]")
    width = weight.shape[1]
    if initial_tail is None:
        initial_tail = inputs.new_zeros((width - 1, inputs.shape[1]))
    if initial_tail.shape != (width - 1, inputs.shape[1]):
        raise ValueError("initial_tail shape does not match convolution")
    padded = torch.cat((initial_tail, inputs), dim=0).T.unsqueeze(0)
    convolved = F.conv1d(
        padded,
        weight.unsqueeze(1),
        groups=inputs.shape[1],
    )
    return F.silu(convolved.squeeze(0).T)


class GdnForwardTrace(NamedTuple):
    output: torch.Tensor
    final_state: torch.Tensor
    states_pre: tuple[torch.Tensor, ...]
    states_decayed: tuple[torch.Tensor, ...]
    deltas: tuple[torch.Tensor, ...]
    repeated_query: torch.Tensor
    repeated_key: torch.Tensor


def gdn_reference_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    scale: float | None = None,
) -> GdnForwardTrace:
    """FP32 recurrent oracle for vLLM's chunked Gated DeltaNet prefill."""

    if query.ndim != 3 or key.shape != query.shape:
        raise ValueError("query and key must share shape [T, Hk, Dk]")
    if value.ndim != 3:
        raise ValueError("value must have shape [T, Hv, Dv]")
    tokens, key_heads, key_dim = query.shape
    if value.shape[0] != tokens:
        raise ValueError("query and value token counts differ")
    value_heads, value_dim = value.shape[1:]
    if value_heads % key_heads:
        raise ValueError("value head count must be divisible by key head count")
    if log_decay.shape != (tokens, value_heads) or beta.shape != log_decay.shape:
        raise ValueError("log_decay and beta must have shape [T, Hv]")
    repeat_factor = value_heads // key_heads
    q = query.float().repeat_interleave(repeat_factor, dim=1)
    k = key.float().repeat_interleave(repeat_factor, dim=1)
    q = q * (key_dim**-0.5 if scale is None else scale)
    v = value.float()
    alpha = log_decay.float().exp()
    beta_f = beta.float()
    if initial_state is None:
        state = torch.zeros(
            value_heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device=query.device,
        )
    else:
        expected = (value_heads, value_dim, key_dim)
        if initial_state.shape != expected:
            raise ValueError(f"initial_state shape {initial_state.shape} != {expected}")
        state = initial_state.float()

    states_pre: list[torch.Tensor] = []
    states_decayed: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    for token in range(tokens):
        states_pre.append(state)
        decayed = state * alpha[token, :, None, None]
        memory = (decayed * k[token, :, None, :]).sum(dim=-1)
        delta = (v[token] - memory) * beta_f[token, :, None]
        state = decayed + delta[:, :, None] * k[token, :, None, :]
        output = (state * q[token, :, None, :]).sum(dim=-1)
        states_decayed.append(decayed)
        deltas.append(delta)
        outputs.append(output)
    return GdnForwardTrace(
        output=torch.stack(outputs, dim=0),
        final_state=state,
        states_pre=tuple(states_pre),
        states_decayed=tuple(states_decayed),
        deltas=tuple(deltas),
        repeated_query=q,
        repeated_key=k,
    )


class GdnVjp(NamedTuple):
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    log_decay: torch.Tensor
    beta: torch.Tensor
    initial_state: torch.Tensor


def gdn_vjp_batched(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    grad_output: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    grad_final_state: torch.Tensor | None = None,
    checkpoint_interval: int = 16,
) -> GdnVjp:
    """Analytic reverse scan for ``C`` independent GDN cotangents.

    Primals are unbatched ``[T, ...]``. ``grad_output`` is
    ``[C, T, Hv, Dv]`` and the returned gradients retain that leading
    cotangent dimension.  Gradients are with respect to vLLM's log-space
    decay gate, not the already exponentiated decay.
    """

    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if query.ndim != 3 or key.shape != query.shape:
        raise ValueError("query and key must share shape [T, Hk, Dk]")
    if value.ndim != 3:
        raise ValueError("value must have shape [T, Hv, Dv]")
    tokens, key_heads, key_dim = query.shape
    if value.shape[0] != tokens:
        raise ValueError("query and value token counts differ")
    value_heads, value_dim = value.shape[1:]
    if value_heads % key_heads:
        raise ValueError("value head count must be divisible by key head count")
    if log_decay.shape != (tokens, value_heads) or beta.shape != log_decay.shape:
        raise ValueError("log_decay and beta must have shape [T, Hv]")
    if grad_output.ndim != 4 or grad_output.shape[1:] != (
        tokens,
        value_heads,
        value_dim,
    ):
        raise ValueError("grad_output must have shape [C, T, Hv, Dv]")
    cotangents = grad_output.shape[0]
    repeat_factor = value_heads // key_heads
    q_scale = key_dim**-0.5 if scale is None else scale
    q = query.float().repeat_interleave(repeat_factor, dim=1) * q_scale
    k = key.float().repeat_interleave(repeat_factor, dim=1)
    alpha = log_decay.float().exp()
    beta_f = beta.float()
    v = value.float()
    dy = grad_output.float()

    expected_state_shape = (value_heads, value_dim, key_dim)
    if initial_state is None:
        state = torch.zeros(
            expected_state_shape,
            dtype=torch.float32,
            device=query.device,
        )
    else:
        if initial_state.shape != expected_state_shape:
            raise ValueError(
                f"initial_state shape {initial_state.shape} != "
                f"{expected_state_shape}"
            )
        state = initial_state.float()

    # Only retain chunk-boundary states.  At Qwen3.6 dimensions one full state
    # is about 3 MiB, so saving every token would dominate the live model's
    # remaining VRAM.  The reverse scan recomputes each short chunk once.
    checkpoints: list[torch.Tensor] = []
    for token in range(tokens):
        if token % checkpoint_interval == 0:
            checkpoints.append(state)
        decayed = state * alpha[token, :, None, None]
        memory = (decayed * k[token, :, None, :]).sum(dim=-1)
        delta = (v[token] - memory) * beta_f[token, :, None]
        state = decayed + delta[:, :, None] * k[token, :, None, :]

    if grad_final_state is None:
        state_bar = torch.zeros(
            cotangents,
            value_heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device=query.device,
        )
    else:
        expected = (cotangents, value_heads, value_dim, key_dim)
        if grad_final_state.shape != expected:
            raise ValueError(
                f"grad_final_state shape {grad_final_state.shape} != {expected}"
            )
        state_bar = grad_final_state.float()

    dq: list[torch.Tensor] = []
    dk: list[torch.Tensor] = []
    dv: list[torch.Tensor] = []
    dg: list[torch.Tensor] = []
    dbeta: list[torch.Tensor] = []
    last_chunk = ((tokens - 1) // checkpoint_interval) * checkpoint_interval
    for chunk_start in range(last_chunk, -1, -checkpoint_interval):
        chunk_end = min(tokens, chunk_start + checkpoint_interval)
        state = checkpoints[chunk_start // checkpoint_interval]
        states_pre: list[torch.Tensor] = []
        for token in range(chunk_start, chunk_end):
            states_pre.append(state)
            decayed = state * alpha[token, :, None, None]
            memory = (decayed * k[token, :, None, :]).sum(dim=-1)
            delta = (v[token] - memory) * beta_f[token, :, None]
            state = decayed + delta[:, :, None] * k[token, :, None, :]

        for token in range(chunk_end - 1, chunk_start - 1, -1):
            state_pre = states_pre[token - chunk_start]
            k_t = k[token]
            q_t = q[token]
            state_decayed = state_pre * alpha[token, :, None, None]
            memory = (state_decayed * k_t[:, None, :]).sum(dim=-1)
            delta = (v[token] - memory) * beta_f[token, :, None]
            state_post = state_decayed + delta[:, :, None] * k_t[:, None, :]
            dy_t = dy[:, token]

            state_bar = state_bar + dy_t[:, :, :, None] * q_t[None, :, None, :]
            dq_t = (dy_t[:, :, :, None] * state_post[None]).sum(dim=-2)

            delta_bar = (state_bar * k_t[None, :, None, :]).sum(dim=-1)
            dk_t = (state_bar * delta[None, :, :, None]).sum(dim=-2)
            decayed_bar = state_bar

            dv_t = delta_bar * beta_f[token][None, :, None]
            dbeta_t = (delta_bar * (v[token] - memory)[None]).sum(dim=-1)
            memory_bar = -delta_bar * beta_f[token][None, :, None]
            decayed_bar = (
                decayed_bar
                + memory_bar[:, :, :, None] * k_t[None, :, None, :]
            )
            dk_t = dk_t + (
                memory_bar[:, :, :, None] * state_decayed[None]
            ).sum(dim=-2)

            dg_t = alpha[token][None] * (
                decayed_bar * state_pre[None]
            ).sum(dim=(-1, -2))
            state_bar = decayed_bar * alpha[token][None, :, None, None]

            dq.append(dq_t)
            dk.append(dk_t)
            dv.append(dv_t)
            dg.append(dg_t)
            dbeta.append(dbeta_t)

    dq_r = torch.stack(dq[::-1], dim=1)
    dk_r = torch.stack(dk[::-1], dim=1)
    dq_r = dq_r.reshape(
        cotangents, tokens, key_heads, repeat_factor, key_dim
    ).sum(dim=3)
    dk_r = dk_r.reshape(
        cotangents, tokens, key_heads, repeat_factor, key_dim
    ).sum(dim=3)
    return GdnVjp(
        query=dq_r * q_scale,
        key=dk_r,
        value=torch.stack(dv[::-1], dim=1),
        log_decay=torch.stack(dg[::-1], dim=1),
        beta=torch.stack(dbeta[::-1], dim=1),
        initial_state=state_bar,
    )


class _ExactGdnRecurrence(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        exact_output: torch.Tensor,
        exact_final_state: torch.Tensor,
        scale: float,
        checkpoint_interval: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_output = (query.shape[0], value.shape[1], value.shape[2])
        expected_state = (value.shape[1], value.shape[2], query.shape[2])
        if exact_output.shape != expected_output:
            raise ValueError(
                f"exact_output shape {exact_output.shape} != {expected_output}"
            )
        if initial_state.shape != expected_state:
            raise ValueError(
                f"initial_state shape {initial_state.shape} != {expected_state}"
            )
        if exact_final_state.shape != expected_state:
            raise ValueError(
                "exact_final_state shape "
                f"{exact_final_state.shape} != {expected_state}"
            )
        ctx.save_for_backward(query, key, value, log_decay, beta, initial_state)
        ctx.input_dtypes = tuple(
            tensor.dtype
            for tensor in (query, key, value, log_decay, beta, initial_state)
        )
        ctx.scale = scale
        ctx.checkpoint_interval = checkpoint_interval
        return exact_output, exact_final_state

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
        grad_final_state: torch.Tensor,
    ):
        query, key, value, log_decay, beta, initial_state = ctx.saved_tensors
        vjp = gdn_vjp_batched(
            query,
            key,
            value,
            log_decay,
            beta,
            grad_output.unsqueeze(0),
            initial_state,
            scale=ctx.scale,
            grad_final_state=grad_final_state.unsqueeze(0),
            checkpoint_interval=ctx.checkpoint_interval,
        )
        gradients = tuple(
            gradient.squeeze(0).to(dtype)
            for gradient, dtype in zip(vjp, ctx.input_dtypes, strict=True)
        )
        return (*gradients, None, None, None, None)


def exact_gdn_recurrence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    exact_output: torch.Tensor,
    exact_final_state: torch.Tensor,
    *,
    scale: float | None = None,
    checkpoint_interval: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use captured Triton recurrence values and the analytic GDN VJP."""

    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    effective_scale = query.shape[-1] ** -0.5 if scale is None else scale
    return _ExactGdnRecurrence.apply(
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state,
        exact_output,
        exact_final_state,
        effective_scale,
        checkpoint_interval,
    )


def l2_normalize(inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FLA-compatible L2 normalization used before the GDN recurrence."""

    x = inputs.float()
    return (x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + eps)).to(
        inputs.dtype
    )


def gdn_log_decay(
    a: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor
) -> torch.Tensor:
    """Compute vLLM's FP32 log-space GDN decay gate."""

    return -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())
