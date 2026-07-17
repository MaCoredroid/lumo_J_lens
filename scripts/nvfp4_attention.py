#!/usr/bin/env python3
"""Differentiable Qwen3.6 full-attention suffix for NVFP4 J-lens fits.

The deployed projection values are injected by caller-supplied linear
functions.  Those functions can therefore use the exact-forward W4A16/FP8
wrappers from :mod:`nvfp4_ste`, while this module supplies the continuous
RMSNorm, MRoPE, causal attention, gating, residual, and SwiGLU derivatives.

The attention oracle is a single-sequence prefill implementation.  It is not
a replacement for paged decode attention or vLLM's packed multi-request
metadata handling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F


Tensor = torch.Tensor
LinearFn = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class QwenFullAttentionConfig:
    """Shape and numerical parameters needed to replay one full-attention block."""

    hidden_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    rms_norm_eps: float
    mrope_section: tuple[int, int, int] | None = None
    mrope_interleaved: bool = True
    attention_output_gate: bool = True

    def __post_init__(self) -> None:
        if min(
            self.hidden_size,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
        ) <= 0:
            raise ValueError("attention dimensions must be positive")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_dim:
            raise ValueError("rotary_dim must be in [1, head_dim]")
        if self.rotary_dim % 2:
            raise ValueError("rotary_dim must be even")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be finite and positive")
        if self.mrope_section is not None:
            if len(self.mrope_section) != 3 or any(x < 0 for x in self.mrope_section):
                raise ValueError("mrope_section must contain three nonnegative sizes")
            if sum(self.mrope_section) != self.rotary_dim // 2:
                raise ValueError("mrope_section must sum to rotary_dim / 2")

    @property
    def query_size(self) -> int:
        return self.num_query_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def qkv_projection_size(self) -> int:
        query_multiplier = 2 if self.attention_output_gate else 1
        return query_multiplier * self.query_size + 2 * self.kv_size


QWEN36_27B_LAYER63 = QwenFullAttentionConfig(
    hidden_size=5120,
    num_query_heads=24,
    num_kv_heads=4,
    head_dim=256,
    rotary_dim=64,
    rope_theta=10_000_000.0,
    rms_norm_eps=1e-6,
    mrope_section=(11, 11, 10),
    mrope_interleaved=True,
    attention_output_gate=True,
)


class _ExactValue(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        surrogate: Tensor,
        exact_output: Tensor,
    ) -> Tensor:
        if surrogate.shape != exact_output.shape:
            raise ValueError(
                "surrogate and exact output shapes differ: "
                f"{tuple(surrogate.shape)} != {tuple(exact_output.shape)}"
            )
        if surrogate.device != exact_output.device:
            raise ValueError("surrogate and exact output must be on the same device")
        ctx.surrogate_dtype = surrogate.dtype
        return exact_output

    @staticmethod
    def backward(ctx: object, grad_output: Tensor):
        return grad_output.to(ctx.surrogate_dtype), None


def replace_forward_value(surrogate: Tensor, exact_output: Tensor) -> Tensor:
    """Return ``exact_output`` while using ``surrogate`` for every VJP.

    This is useful for non-linear fused kernels as well as linear projection
    closures.  The exact tensor is treated as captured data, never as a
    differentiable input.
    """

    return _ExactValue.apply(surrogate, exact_output)


def qwen_rms_norm(inputs: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Qwen3.6/Gemma RMSNorm, including the learned-weight ``+1`` offset."""

    if inputs.shape[-1] != weight.numel():
        raise ValueError("RMSNorm weight does not match the input width")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm epsilon must be finite and positive")
    normalized = inputs.float() * torch.rsqrt(
        inputs.float().square().mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * (1.0 + weight.float())).to(inputs.dtype)


def qwen_fused_add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor | None,
    weight: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Logical equivalent of vLLM's fused residual add plus Gemma RMSNorm.

    vLLM carries the current branch and accumulated residual separately.  The
    returned pair is ``(normalized_sum, accumulated_sum)``.
    """

    if residual is None:
        accumulated = hidden_states
    else:
        if residual.shape != hidden_states.shape:
            raise ValueError("hidden_states and residual shapes differ")
        accumulated = hidden_states + residual
    return qwen_rms_norm(accumulated, weight, eps), accumulated


def logical_hidden_states(hidden_states: Tensor, residual: Tensor | None) -> Tensor:
    """Reconstruct the logical HF-style hidden state from vLLM's split pair."""

    if residual is None:
        return hidden_states
    if hidden_states.shape != residual.shape:
        raise ValueError("hidden_states and residual shapes differ")
    return hidden_states + residual


def qwen_mrope_cos_sin(
    positions: Tensor,
    *,
    rotary_dim: int,
    rope_theta: float,
    mrope_section: tuple[int, int, int] | None = None,
    interleaved: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Build vLLM-compatible Qwen rotary rows for text-only fitting."""

    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive and even")
    if positions.ndim not in {1, 2}:
        raise ValueError("positions must have shape [T] or [3, T]")
    if positions.ndim == 2 and positions.shape[0] != 3:
        raise ValueError("multimodal positions must have shape [3, T]")
    if bool((positions < 0).any()):
        raise ValueError("positions must be nonnegative")
    if positions.ndim == 2:
        if mrope_section is None or sum(mrope_section) != rotary_dim // 2:
            raise ValueError("valid mrope_section is required for 3-axis positions")
        if not (
            torch.equal(positions[0], positions[1])
            and torch.equal(positions[0], positions[2])
        ):
            raise NotImplementedError(
                "divergent 3-axis multimodal MRoPE positions are outside the "
                "text-only NVFP4 fit contract"
            )
        positions = positions[0]

    exponents = torch.arange(
        0,
        rotary_dim,
        2,
        dtype=torch.float32,
        device=positions.device,
    ) / rotary_dim
    inverse_frequency = rope_theta ** (-exponents)
    frequencies = positions.to(torch.float32).unsqueeze(-1) * inverse_frequency

    return frequencies.cos().to(dtype), frequencies.sin().to(dtype)


def apply_neox_rotary(
    inputs: Tensor,
    cos: Tensor,
    sin: Tensor,
    *,
    rotary_dim: int,
) -> Tensor:
    """Apply NeoX-style partial rotary embedding to ``[T, H, D]`` inputs."""

    if inputs.ndim != 3:
        raise ValueError("rotary inputs must have shape [T, H, D]")
    if rotary_dim <= 0 or rotary_dim > inputs.shape[-1] or rotary_dim % 2:
        raise ValueError("invalid rotary_dim")
    expected = (inputs.shape[0], rotary_dim // 2)
    if cos.shape != expected or sin.shape != expected:
        raise ValueError(f"cos and sin must both have shape {expected}")

    rotating, passthrough = inputs[..., :rotary_dim], inputs[..., rotary_dim:]
    first, second = rotating.chunk(2, dim=-1)
    cos_heads = cos.to(inputs.dtype).unsqueeze(1)
    sin_heads = sin.to(inputs.dtype).unsqueeze(1)
    rotated = torch.cat(
        (
            first * cos_heads - second * sin_heads,
            second * cos_heads + first * sin_heads,
        ),
        dim=-1,
    )
    return torch.cat((rotated, passthrough), dim=-1)


def apply_qwen_mrope(
    query: Tensor,
    key: Tensor,
    positions: Tensor,
    config: QwenFullAttentionConfig,
) -> tuple[Tensor, Tensor]:
    """Apply the checkpoint's partial MRoPE to query and key heads."""

    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query and key must have shape [T, H, D]")
    if query.shape[0] != key.shape[0] or query.shape[-1] != config.head_dim:
        raise ValueError("query/key token or query head dimensions are invalid")
    if key.shape[-1] != config.head_dim:
        raise ValueError("key head dimension is invalid")
    cos, sin = qwen_mrope_cos_sin(
        positions,
        rotary_dim=config.rotary_dim,
        rope_theta=config.rope_theta,
        mrope_section=config.mrope_section,
        interleaved=config.mrope_interleaved,
        dtype=query.dtype,
    )
    return (
        apply_neox_rotary(query, cos, sin, rotary_dim=config.rotary_dim),
        apply_neox_rotary(key, cos, sin, rotary_dim=config.rotary_dim),
    )


def _validate_gqa_inputs(query: Tensor, key: Tensor, value: Tensor) -> int:
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key, and value must have shape [T, H, D]")
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("query, key, and value token counts differ")
    if key.shape[1] != value.shape[1]:
        raise ValueError("key and value head counts differ")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions differ")
    if query.shape[1] % key.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    return query.shape[1] // key.shape[1]


def _accumulator(tensor: Tensor) -> Tensor:
    return tensor if tensor.dtype == torch.float64 else tensor.float()


def _gqa_probabilities(query: Tensor, key: Tensor, scale: float) -> tuple[Tensor, Tensor]:
    groups = _validate_gqa_inputs(query, key, key)
    repeated_key = _accumulator(key).repeat_interleave(groups, dim=1)
    scores = torch.einsum("thd,shd->hts", _accumulator(query), repeated_key)
    scores = scores * scale
    tokens = query.shape[0]
    causal = torch.ones(tokens, tokens, dtype=torch.bool, device=query.device).tril()
    probabilities = torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1)
    return probabilities, repeated_key


def canonical_gqa_causal_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Differentiable single-sequence GQA prefill oracle.

    Inputs use vLLM's flattened-token layout ``[T, heads, head_dim]``.  KV
    heads are repeated in contiguous query-head groups, matching Qwen/vLLM.
    """

    groups = _validate_gqa_inputs(query, key, value)
    actual_scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(actual_scale):
        raise ValueError("attention scale must be finite")
    probabilities, _ = _gqa_probabilities(query, key, actual_scale)
    repeated_value = _accumulator(value).repeat_interleave(groups, dim=1)
    output = torch.einsum("hts,shv->thv", probabilities, repeated_value)
    return output.to(value.dtype)


class AttentionVjp(NamedTuple):
    query: Tensor
    key: Tensor
    value: Tensor


def gqa_causal_attention_vjp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    *,
    scale: float | None = None,
) -> AttentionVjp:
    """Explicit full-sequence VJP for canonical grouped-query attention.

    ``grad_output`` may be ``[T, Hq, Dv]`` or contain arbitrary leading
    cotangent-batch dimensions.  No position averaging or per-token Jacobian
    approximation is performed.
    """

    groups = _validate_gqa_inputs(query, key, value)
    expected_tail = (query.shape[0], query.shape[1], value.shape[-1])
    if grad_output.shape[-3:] != expected_tail:
        raise ValueError(f"grad_output must end in {expected_tail}")
    actual_scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    probabilities, repeated_key = _gqa_probabilities(query, key, actual_scale)
    repeated_value = _accumulator(value).repeat_interleave(groups, dim=1)
    cotangent = _accumulator(grad_output)

    grad_probabilities = torch.einsum(
        "...thv,shv->...hts", cotangent, repeated_value
    )
    centered = grad_probabilities - (
        grad_probabilities * probabilities
    ).sum(dim=-1, keepdim=True)
    grad_scores = probabilities * centered
    grad_query = actual_scale * torch.einsum(
        "...hts,shd->...thd", grad_scores, repeated_key
    )
    grad_key_repeated = actual_scale * torch.einsum(
        "...hts,thd->...shd", grad_scores, _accumulator(query)
    )
    grad_value_repeated = torch.einsum(
        "hts,...thv->...shv", probabilities, cotangent
    )

    leading = grad_output.shape[:-3]
    tokens, kv_heads = key.shape[:2]
    grad_key = grad_key_repeated.reshape(
        *leading, tokens, kv_heads, groups, key.shape[-1]
    ).sum(dim=-2)
    grad_value = grad_value_repeated.reshape(
        *leading, tokens, kv_heads, groups, value.shape[-1]
    ).sum(dim=-2)
    return AttentionVjp(
        grad_query.to(query.dtype),
        grad_key.to(key.dtype),
        grad_value.to(value.dtype),
    )


def exact_gqa_causal_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    exact_output: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Use an exact deployed attention value and canonical attention VJPs."""

    surrogate = canonical_gqa_causal_attention(
        query,
        key,
        value,
        scale=scale,
    )
    return replace_forward_value(surrogate, exact_output)


def split_qwen_qkv(
    qkv: Tensor,
    config: QwenFullAttentionConfig,
) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
    """Decode vLLM's fused ``q(+gate), k, v`` projection layout."""

    if qkv.ndim != 2 or qkv.shape[-1] != config.qkv_projection_size:
        raise ValueError(
            f"qkv must have shape [T, {config.qkv_projection_size}]"
        )
    tokens = qkv.shape[0]
    q_width = config.query_size * (2 if config.attention_output_gate else 1)
    q_part, key, value = qkv.split(
        (q_width, config.kv_size, config.kv_size), dim=-1
    )
    if config.attention_output_gate:
        q_and_gate = q_part.reshape(tokens, config.num_query_heads, 2 * config.head_dim)
        query, gate = q_and_gate.chunk(2, dim=-1)
    else:
        query = q_part.reshape(tokens, config.num_query_heads, config.head_dim)
        gate = None
    return (
        query,
        gate,
        key.reshape(tokens, config.num_kv_heads, config.head_dim),
        value.reshape(tokens, config.num_kv_heads, config.head_dim),
    )


def qwen_swiglu(gate_up: Tensor, *, exact_output: Tensor | None = None) -> Tensor:
    """Continuous surrogate for vLLM's fused ``silu_and_mul`` operation."""

    if gate_up.shape[-1] % 2:
        raise ValueError("gate_up width must be even")
    gate, up = gate_up.chunk(2, dim=-1)
    activated = (F.silu(gate.float()) * up.float()).to(gate_up.dtype)
    if exact_output is not None:
        activated = replace_forward_value(activated, exact_output)
    return activated


@dataclass(frozen=True)
class QwenBlockLinears:
    """Projection callables in vLLM execution order."""

    qkv: LinearFn
    attention_out: LinearFn
    gate_up: LinearFn
    down: LinearFn


@dataclass
class QwenAttentionReplay:
    output: Tensor
    qkv: Tensor
    query: Tensor
    key: Tensor
    value: Tensor
    gate: Tensor | None
    core_output: Tensor
    gated_output: Tensor


def _inject_exact_heads(
    surrogate: Tensor,
    exact_output: Tensor | None,
    *,
    name: str,
) -> Tensor:
    if exact_output is None:
        return surrogate
    if exact_output.shape == (
        surrogate.shape[0],
        surrogate.shape[1] * surrogate.shape[2],
    ):
        exact_output = exact_output.reshape_as(surrogate)
    if exact_output.shape != surrogate.shape:
        raise ValueError(
            f"exact {name} must have shape {tuple(surrogate.shape)} or "
            f"[{surrogate.shape[0]}, {surrogate.shape[1] * surrogate.shape[2]}]"
        )
    return replace_forward_value(surrogate, exact_output)


def replay_qwen_full_attention(
    normalized_hidden_states: Tensor,
    positions: Tensor,
    config: QwenFullAttentionConfig,
    *,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    qkv_linear: LinearFn,
    attention_out_linear: LinearFn,
    exact_query: Tensor | None = None,
    exact_key: Tensor | None = None,
    exact_value: Tensor | None = None,
    exact_attention_output: Tensor | None = None,
) -> QwenAttentionReplay:
    """Replay Qwen full attention with caller-controlled deployed values.

    ``exact_query``, ``exact_key``, and ``exact_value`` are the inputs captured
    at vLLM's ``Attention`` module boundary.  They may use either headed or
    flattened layout.  Supplying them evaluates the canonical attention VJP
    at the actual deployed values while retaining the surrogate RMSNorm/RoPE
    derivative back to the fused QKV projection.
    """

    if normalized_hidden_states.ndim != 2:
        raise ValueError("normalized_hidden_states must have shape [T, hidden]")
    if normalized_hidden_states.shape[-1] != config.hidden_size:
        raise ValueError("normalized hidden width does not match config")
    qkv = qkv_linear(normalized_hidden_states)
    query, gate, key, value = split_qwen_qkv(qkv, config)
    query = qwen_rms_norm(query, q_norm_weight, config.rms_norm_eps)
    key = qwen_rms_norm(key, k_norm_weight, config.rms_norm_eps)
    query, key = apply_qwen_mrope(query, key, positions, config)
    query = _inject_exact_heads(query, exact_query, name="query")
    key = _inject_exact_heads(key, exact_key, name="key")
    value = _inject_exact_heads(value, exact_value, name="value")
    if exact_attention_output is None:
        core = canonical_gqa_causal_attention(query, key, value)
    else:
        if exact_attention_output.shape == (
            query.shape[0],
            config.query_size,
        ):
            exact_attention_output = exact_attention_output.reshape_as(query)
        core = exact_gqa_causal_attention(
            query,
            key,
            value,
            exact_attention_output,
        )
    flattened = core.reshape(core.shape[0], -1)
    if gate is not None:
        flattened_gate = gate.reshape(gate.shape[0], -1)
        gated = flattened * torch.sigmoid(flattened_gate)
    else:
        gated = flattened
    output = attention_out_linear(gated)
    return QwenAttentionReplay(
        output=output,
        qkv=qkv,
        query=query,
        key=key,
        value=value,
        gate=gate,
        core_output=core,
        gated_output=gated,
    )


@dataclass
class QwenBlockReplay:
    """Logical output plus vLLM's final branch/residual representation."""

    output: Tensor
    hidden_states: Tensor
    residual: Tensor
    attention_input: Tensor
    attention: QwenAttentionReplay
    after_attention: Tensor
    mlp_input: Tensor
    gate_up: Tensor
    activated: Tensor


def replay_qwen_full_attention_suffix(
    logical_input: Tensor,
    positions: Tensor,
    config: QwenFullAttentionConfig,
    *,
    input_norm_weight: Tensor,
    post_attention_norm_weight: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    linears: QwenBlockLinears,
    exact_query: Tensor | None = None,
    exact_key: Tensor | None = None,
    exact_value: Tensor | None = None,
    exact_attention_output: Tensor | None = None,
    exact_swiglu_output: Tensor | None = None,
) -> QwenBlockReplay:
    """Replay a complete Qwen full-attention block from its logical input.

    The returned ``hidden_states`` and ``residual`` are exactly the two values
    vLLM passes to the next block.  Their sum is the HF-style ``output``.
    """

    if logical_input.ndim != 2 or logical_input.shape[-1] != config.hidden_size:
        raise ValueError("logical_input must have shape [T, hidden_size]")
    attention_input = qwen_rms_norm(
        logical_input,
        input_norm_weight,
        config.rms_norm_eps,
    )
    attention = replay_qwen_full_attention(
        attention_input,
        positions,
        config,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qkv_linear=linears.qkv,
        attention_out_linear=linears.attention_out,
        exact_query=exact_query,
        exact_key=exact_key,
        exact_value=exact_value,
        exact_attention_output=exact_attention_output,
    )
    after_attention = logical_input + attention.output
    mlp_input = qwen_rms_norm(
        after_attention,
        post_attention_norm_weight,
        config.rms_norm_eps,
    )
    gate_up = linears.gate_up(mlp_input)
    activated = qwen_swiglu(gate_up, exact_output=exact_swiglu_output)
    mlp_output = linears.down(activated)
    output = after_attention + mlp_output
    return QwenBlockReplay(
        output=output,
        hidden_states=mlp_output,
        residual=after_attention,
        attention_input=attention_input,
        attention=attention,
        after_attention=after_attention,
        mlp_input=mlp_input,
        gate_up=gate_up,
        activated=activated,
    )
