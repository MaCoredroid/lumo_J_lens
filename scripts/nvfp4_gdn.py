#!/usr/bin/env python3
"""Exact-forward surrogate replay for one Qwen3.6 Gated DeltaNet branch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from nvfp4_attention import qwen_rms_norm, qwen_swiglu
from nvfp4_ste import (
    causal_depthwise_conv1d_silu,
    exact_fp8_linear_ste,
    exact_frozen_linear,
    exact_gdn_recurrence,
    exact_value,
    gated_rms_norm,
    gdn_log_decay,
    l2_normalize,
)


@dataclass(frozen=True)
class GdnLayout:
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    norm_eps: float = 1e-6


@dataclass(frozen=True)
class GdnWeights:
    qkvz_out_in: torch.Tensor | None
    qkvz_input_scale: torch.Tensor | None
    ba_out_in: torch.Tensor
    conv: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    norm: torch.Tensor
    out_out_in: torch.Tensor | None
    out_input_scale: torch.Tensor | None


@dataclass(frozen=True)
class GdnCapture:
    qkvz: torch.Tensor
    ba: torch.Tensor
    conv_qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    log_decay: torch.Tensor
    beta: torch.Tensor
    core_output: torch.Tensor
    final_state: torch.Tensor
    gated_norm: torch.Tensor
    branch_output: torch.Tensor


@dataclass(frozen=True)
class GdnBlockReplay:
    output: torch.Tensor
    attention_input: torch.Tensor
    attention_output: torch.Tensor
    after_attention: torch.Tensor
    mlp_input: torch.Tensor
    gate_up: torch.Tensor
    activated: torch.Tensor
    mlp_output: torch.Tensor


def _check_capture(
    hidden: torch.Tensor,
    layout: GdnLayout,
    capture: GdnCapture,
) -> None:
    tokens = hidden.shape[0]
    q_size = layout.key_heads * layout.key_dim
    v_size = layout.value_heads * layout.value_dim
    qkv_size = q_size * 2 + v_size
    expected = {
        "qkvz": (tokens, qkv_size + v_size),
        "ba": (tokens, layout.value_heads * 2),
        "conv_qkv": (tokens, qkv_size),
        "query": (tokens, layout.key_heads, layout.key_dim),
        "key": (tokens, layout.key_heads, layout.key_dim),
        "value": (tokens, layout.value_heads, layout.value_dim),
        "log_decay": (tokens, layout.value_heads),
        "beta": (tokens, layout.value_heads),
        "core_output": (tokens, layout.value_heads, layout.value_dim),
        "final_state": (
            layout.value_heads,
            layout.value_dim,
            layout.key_dim,
        ),
        "gated_norm": (tokens, layout.value_heads, layout.value_dim),
        "branch_output": hidden.shape,
    }
    for name, shape in expected.items():
        actual = getattr(capture, name).shape
        if actual != shape:
            raise ValueError(f"capture {name} shape {actual} != {shape}")


def replay_gdn_branch(
    normalized_hidden: torch.Tensor,
    layout: GdnLayout,
    weights: GdnWeights,
    capture: GdnCapture,
    *,
    qkvz_linear: Callable[[torch.Tensor], torch.Tensor] | None = None,
    out_linear: Callable[[torch.Tensor], torch.Tensor] | None = None,
    initial_state: torch.Tensor | None = None,
    initial_conv_tail: torch.Tensor | None = None,
    ste_policy: str = "identity",
    checkpoint_interval: int = 16,
) -> torch.Tensor:
    """Replay a GDN branch with captured kernel values in every forward step.

    The return value is bit-for-bit the captured vLLM branch output. Gradients
    follow the declared FP8 STE, frozen effective weights, smooth conv/gates,
    and analytic GDN recurrence.
    """

    if normalized_hidden.ndim != 2:
        raise ValueError("normalized_hidden must have shape [tokens, hidden]")
    _check_capture(normalized_hidden, layout, capture)

    q_size = layout.key_heads * layout.key_dim
    v_size = layout.value_heads * layout.value_dim
    qkv_size = q_size * 2 + v_size

    if qkvz_linear is None:
        if weights.qkvz_out_in is None or weights.qkvz_input_scale is None:
            raise ValueError("dense GDN qkvz weights are absent")
        qkvz = exact_fp8_linear_ste(
            normalized_hidden,
            capture.qkvz,
            weights.qkvz_out_in,
            weights.qkvz_input_scale,
            ste_policy=ste_policy,
        )
    else:
        qkvz = qkvz_linear(normalized_hidden)
    ba = exact_frozen_linear(
        normalized_hidden,
        capture.ba,
        weights.ba_out_in,
    )
    mixed_qkv, z_flat = qkvz.split((qkv_size, v_size), dim=-1)
    z = z_flat.reshape(
        normalized_hidden.shape[0], layout.value_heads, layout.value_dim
    )

    conv_qkv = exact_value(
        causal_depthwise_conv1d_silu(
            mixed_qkv,
            weights.conv,
            initial_tail=initial_conv_tail,
        ),
        capture.conv_qkv,
    )
    query_raw, key_raw, value_raw = conv_qkv.split(
        (q_size, q_size, v_size), dim=-1
    )
    query_raw = query_raw.reshape(
        normalized_hidden.shape[0], layout.key_heads, layout.key_dim
    )
    key_raw = key_raw.reshape(
        normalized_hidden.shape[0], layout.key_heads, layout.key_dim
    )
    value_raw = value_raw.reshape(
        normalized_hidden.shape[0], layout.value_heads, layout.value_dim
    )
    query = exact_value(l2_normalize(query_raw), capture.query)
    key = exact_value(l2_normalize(key_raw), capture.key)
    value = exact_value(value_raw, capture.value)

    b, a = ba.chunk(2, dim=-1)
    log_decay = exact_value(
        gdn_log_decay(a, weights.a_log, weights.dt_bias),
        capture.log_decay,
    )
    beta = exact_value(torch.sigmoid(b.float()), capture.beta)

    if initial_state is None:
        initial_state = torch.zeros_like(capture.final_state)
    core_output, _ = exact_gdn_recurrence(
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state,
        capture.core_output,
        capture.final_state,
        checkpoint_interval=checkpoint_interval,
    )
    normalized = exact_value(
        gated_rms_norm(core_output, z, weights.norm, layout.norm_eps),
        capture.gated_norm,
    )
    if out_linear is not None:
        return out_linear(normalized.flatten(-2))
    if weights.out_out_in is None or weights.out_input_scale is None:
        raise ValueError("dense GDN output weights are absent")
    return exact_fp8_linear_ste(
        normalized.flatten(-2),
        capture.branch_output,
        weights.out_out_in,
        weights.out_input_scale,
        ste_policy=ste_policy,
    )


def replay_qwen_gdn_block(
    logical_input: torch.Tensor,
    layout: GdnLayout,
    weights: GdnWeights,
    capture: GdnCapture,
    *,
    input_norm_weight: torch.Tensor,
    post_attention_norm_weight: torch.Tensor,
    gate_up_linear: Callable[[torch.Tensor], torch.Tensor],
    down_linear: Callable[[torch.Tensor], torch.Tensor],
    qkvz_linear: Callable[[torch.Tensor], torch.Tensor] | None = None,
    out_linear: Callable[[torch.Tensor], torch.Tensor] | None = None,
    exact_swiglu_output: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    initial_conv_tail: torch.Tensor | None = None,
    ste_policy: str = "identity",
    checkpoint_interval: int = 16,
) -> GdnBlockReplay:
    """Replay a complete Qwen3.6 GDN decoder block from a logical residual."""

    attention_input = qwen_rms_norm(
        logical_input, input_norm_weight, layout.norm_eps
    )
    attention_output = replay_gdn_branch(
        attention_input,
        layout,
        weights,
        capture,
        qkvz_linear=qkvz_linear,
        out_linear=out_linear,
        initial_state=initial_state,
        initial_conv_tail=initial_conv_tail,
        ste_policy=ste_policy,
        checkpoint_interval=checkpoint_interval,
    )
    after_attention = logical_input + attention_output
    mlp_input = qwen_rms_norm(
        after_attention, post_attention_norm_weight, layout.norm_eps
    )
    gate_up = gate_up_linear(mlp_input)
    activated = qwen_swiglu(gate_up, exact_output=exact_swiglu_output)
    mlp_output = down_linear(activated)
    return GdnBlockReplay(
        output=after_attention + mlp_output,
        attention_input=attention_input,
        attention_output=attention_output,
        after_attention=after_attention,
        mlp_input=mlp_input,
        gate_up=gate_up,
        activated=activated,
        mlp_output=mlp_output,
    )
