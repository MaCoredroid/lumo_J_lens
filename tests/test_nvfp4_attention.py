#!/usr/bin/env python3
"""Tests for the Qwen3.6 full-attention surrogate suffix."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "nvfp4_attention", ROOT / "scripts" / "nvfp4_attention.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RotaryEmbeddingTest(unittest.TestCase):
    def test_text_positions_equal_triplicated_mrope_positions(self) -> None:
        positions = torch.arange(5)
        text = MODULE.qwen_mrope_cos_sin(
            positions,
            rotary_dim=12,
            rope_theta=10_000.0,
            mrope_section=(2, 2, 2),
        )
        multimodal = MODULE.qwen_mrope_cos_sin(
            positions.repeat(3, 1),
            rotary_dim=12,
            rope_theta=10_000.0,
            mrope_section=(2, 2, 2),
        )
        for actual, expected in zip(multimodal, text, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_divergent_multimodal_mrope_positions(self) -> None:
        positions = torch.stack(
            [torch.arange(5), torch.arange(5) + 1, torch.arange(5) + 2]
        )
        with self.assertRaisesRegex(NotImplementedError, "text-only"):
            MODULE.qwen_mrope_cos_sin(
                positions,
                rotary_dim=12,
                rope_theta=10_000.0,
                mrope_section=(2, 2, 2),
            )

    def test_partial_rotary_preserves_norm_and_passthrough(self) -> None:
        generator = torch.Generator().manual_seed(10)
        inputs = torch.randn(4, 3, 8, generator=generator)
        cos, sin = MODULE.qwen_mrope_cos_sin(
            torch.arange(4), rotary_dim=4, rope_theta=1000.0
        )
        actual = MODULE.apply_neox_rotary(inputs, cos, sin, rotary_dim=4)
        torch.testing.assert_close(actual[..., 4:], inputs[..., 4:], rtol=0, atol=0)
        torch.testing.assert_close(
            actual[..., :4].square().sum(-1),
            inputs[..., :4].square().sum(-1),
            rtol=2e-6,
            atol=2e-6,
        )


class GroupedQueryAttentionTest(unittest.TestCase):
    def make_primals(self):
        generator = torch.Generator().manual_seed(123)
        query = torch.randn(5, 6, 4, generator=generator, dtype=torch.float64)
        key = torch.randn(5, 2, 4, generator=generator, dtype=torch.float64)
        value = torch.randn(5, 2, 3, generator=generator, dtype=torch.float64)
        return query, key, value

    def test_gqa_shapes(self) -> None:
        query, key, value = self.make_primals()
        output = MODULE.canonical_gqa_causal_attention(query, key, value)
        self.assertEqual(output.shape, (5, 6, 3))
        grads = MODULE.gqa_causal_attention_vjp(
            query, key, value, torch.randn(7, 5, 6, 3, dtype=torch.float64)
        )
        self.assertEqual(grads.query.shape, (7, 5, 6, 4))
        self.assertEqual(grads.key.shape, (7, 5, 2, 4))
        self.assertEqual(grads.value.shape, (7, 5, 2, 3))

    def test_future_key_and_value_do_not_affect_earlier_outputs(self) -> None:
        query, key, value = self.make_primals()
        baseline = MODULE.canonical_gqa_causal_attention(query, key, value)
        changed_key = key.clone()
        changed_value = value.clone()
        changed_key[-1].add_(100.0)
        changed_value[-1].sub_(100.0)
        changed = MODULE.canonical_gqa_causal_attention(
            query, changed_key, changed_value
        )
        torch.testing.assert_close(changed[:-1], baseline[:-1], rtol=0, atol=0)

    def test_batched_explicit_vjp_matches_autograd(self) -> None:
        query, key, value = self.make_primals()
        leaves = tuple(x.detach().requires_grad_(True) for x in (query, key, value))
        output = MODULE.canonical_gqa_causal_attention(*leaves)
        cotangents = torch.randn(4, *output.shape, dtype=torch.float64)
        expected = torch.autograd.grad(
            output,
            leaves,
            grad_outputs=cotangents,
            is_grads_batched=True,
        )
        actual = MODULE.gqa_causal_attention_vjp(
            query, key, value, cotangents
        )
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=2e-12, atol=2e-12)

    def test_vjp_satisfies_adjoint_identity(self) -> None:
        primals = self.make_primals()
        directions = tuple(torch.randn_like(x) for x in primals)
        cotangent = torch.randn(5, 6, 3, dtype=torch.float64)
        _, tangent = torch.autograd.functional.jvp(
            lambda q, k, v: MODULE.canonical_gqa_causal_attention(q, k, v),
            primals,
            directions,
        )
        vjp = MODULE.gqa_causal_attention_vjp(*primals, cotangent)
        lhs = (tangent * cotangent).sum()
        rhs = sum((direction * grad).sum() for direction, grad in zip(directions, vjp))
        torch.testing.assert_close(lhs, rhs, rtol=3e-12, atol=3e-12)

    def test_exact_attention_value_keeps_canonical_derivative(self) -> None:
        query, key, value = self.make_primals()
        leaves = tuple(x.detach().requires_grad_(True) for x in (query, key, value))
        exact = torch.randn(5, 6, 3, dtype=torch.float64)
        output = MODULE.exact_gqa_causal_attention(*leaves, exact)
        torch.testing.assert_close(output, exact, rtol=0, atol=0)
        cotangent = torch.randn_like(exact)
        actual = torch.autograd.grad(output, leaves, cotangent)
        expected = MODULE.gqa_causal_attention_vjp(*leaves, cotangent)
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=2e-12, atol=2e-12)

    def test_batched_vjp_passes_through_exact_attention_wrapper(self) -> None:
        query, key, value = self.make_primals()
        leaves = tuple(x.detach().requires_grad_(True) for x in (query, key, value))
        exact = torch.randn(5, 6, 3, dtype=torch.float64)
        output = MODULE.exact_gqa_causal_attention(*leaves, exact)
        cotangents = torch.randn(4, *exact.shape, dtype=torch.float64)
        actual = torch.autograd.grad(
            output,
            leaves,
            grad_outputs=cotangents,
            is_grads_batched=True,
        )
        expected = MODULE.gqa_causal_attention_vjp(*leaves, cotangents)
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=2e-12, atol=2e-12)


class QwenSuffixReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MODULE.QwenFullAttentionConfig(
            hidden_size=8,
            num_query_heads=2,
            num_kv_heads=1,
            head_dim=4,
            rotary_dim=4,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            mrope_section=(1, 1, 0),
        )

    def test_generic_exact_value_is_exact_with_identity_vjp(self) -> None:
        surrogate = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        exact = torch.randn(3, 4, dtype=torch.float32)
        output = MODULE.replace_forward_value(surrogate, exact)
        self.assertEqual(output.dtype, exact.dtype)
        torch.testing.assert_close(output, exact, rtol=0, atol=0)
        cotangent = torch.randn_like(output)
        output.backward(cotangent)
        torch.testing.assert_close(
            surrogate.grad, cotangent.to(torch.float64), rtol=0, atol=0
        )

    def test_split_residual_matches_logical_residual_sequence(self) -> None:
        generator = torch.Generator().manual_seed(9)
        branch = torch.randn(3, 8, generator=generator)
        residual = torch.randn(3, 8, generator=generator)
        weight = torch.randn(8, generator=generator)
        normalized, accumulated = MODULE.qwen_fused_add_rms_norm(
            branch, residual, weight, 1e-6
        )
        torch.testing.assert_close(accumulated, branch + residual)
        torch.testing.assert_close(
            normalized, MODULE.qwen_rms_norm(branch + residual, weight, 1e-6)
        )
        torch.testing.assert_close(
            MODULE.logical_hidden_states(branch, residual), branch + residual
        )

    def test_suffix_replay_shapes_and_vllm_residual_contract(self) -> None:
        generator = torch.Generator().manual_seed(51)
        tokens, hidden, intermediate = 4, 8, 6
        qkv_weight = torch.randn(
            self.config.qkv_projection_size, hidden, generator=generator
        ) * 0.1
        out_weight = torch.randn(hidden, self.config.query_size, generator=generator) * 0.1
        gate_up_weight = torch.randn(2 * intermediate, hidden, generator=generator) * 0.1
        down_weight = torch.randn(hidden, intermediate, generator=generator) * 0.1

        linears = MODULE.QwenBlockLinears(
            qkv=lambda x: x @ qkv_weight.T,
            attention_out=lambda x: x @ out_weight.T,
            gate_up=lambda x: x @ gate_up_weight.T,
            down=lambda x: x @ down_weight.T,
        )
        logical_input = torch.randn(tokens, hidden, generator=generator)
        zeros = torch.zeros(hidden)
        trace = MODULE.replay_qwen_full_attention_suffix(
            logical_input,
            torch.arange(tokens),
            self.config,
            input_norm_weight=zeros,
            post_attention_norm_weight=zeros,
            q_norm_weight=torch.zeros(self.config.head_dim),
            k_norm_weight=torch.zeros(self.config.head_dim),
            linears=linears,
        )
        self.assertEqual(trace.output.shape, (tokens, hidden))
        self.assertEqual(trace.attention.query.shape, (tokens, 2, 4))
        self.assertEqual(trace.attention.key.shape, (tokens, 1, 4))
        self.assertEqual(trace.gate_up.shape, (tokens, 2 * intermediate))
        torch.testing.assert_close(
            trace.output,
            MODULE.logical_hidden_states(trace.hidden_states, trace.residual),
        )

    def test_attention_uses_captured_flat_query_key_value(self) -> None:
        generator = torch.Generator().manual_seed(61)
        tokens, hidden = 3, self.config.hidden_size
        qkv_weight = torch.randn(
            self.config.qkv_projection_size, hidden, generator=generator
        ) * 0.1
        inputs = torch.randn(
            tokens, hidden, generator=generator, requires_grad=True
        )
        exact_query = torch.randn(
            tokens, self.config.query_size, generator=generator
        )
        exact_key = torch.randn(tokens, self.config.kv_size, generator=generator)
        exact_value = torch.randn(tokens, self.config.kv_size, generator=generator)
        trace = MODULE.replay_qwen_full_attention(
            inputs,
            torch.arange(tokens),
            self.config,
            q_norm_weight=torch.zeros(self.config.head_dim),
            k_norm_weight=torch.zeros(self.config.head_dim),
            qkv_linear=lambda x: x @ qkv_weight.T,
            attention_out_linear=lambda x: x,
            exact_query=exact_query,
            exact_key=exact_key,
            exact_value=exact_value,
        )
        expected = MODULE.canonical_gqa_causal_attention(
            exact_query.reshape(
                tokens, self.config.num_query_heads, self.config.head_dim
            ),
            exact_key.reshape(
                tokens, self.config.num_kv_heads, self.config.head_dim
            ),
            exact_value.reshape(
                tokens, self.config.num_kv_heads, self.config.head_dim
            ),
        )
        torch.testing.assert_close(trace.core_output, expected, rtol=0, atol=0)
        trace.output.square().sum().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)

    def test_suffix_accepts_exact_attention_and_swiglu_values(self) -> None:
        generator = torch.Generator().manual_seed(77)
        tokens, hidden, intermediate = 3, 8, 5
        qkv_weight = torch.randn(
            self.config.qkv_projection_size, hidden, generator=generator
        ) * 0.05
        out_weight = torch.randn(hidden, self.config.query_size, generator=generator) * 0.05
        gate_up_weight = torch.randn(2 * intermediate, hidden, generator=generator) * 0.05
        down_weight = torch.randn(hidden, intermediate, generator=generator) * 0.05
        linears = MODULE.QwenBlockLinears(
            qkv=lambda x: x @ qkv_weight.T,
            attention_out=lambda x: x @ out_weight.T,
            gate_up=lambda x: x @ gate_up_weight.T,
            down=lambda x: x @ down_weight.T,
        )
        exact_attention = torch.randn(
            tokens, self.config.query_size, generator=generator
        )
        exact_swiglu = torch.randn(tokens, intermediate, generator=generator)
        exact_query = torch.randn(
            tokens, self.config.query_size, generator=generator
        )
        exact_key = torch.randn(tokens, self.config.kv_size, generator=generator)
        exact_value = torch.randn(tokens, self.config.kv_size, generator=generator)
        trace = MODULE.replay_qwen_full_attention_suffix(
            torch.randn(tokens, hidden, generator=generator, requires_grad=True),
            torch.arange(tokens),
            self.config,
            input_norm_weight=torch.zeros(hidden),
            post_attention_norm_weight=torch.zeros(hidden),
            q_norm_weight=torch.zeros(self.config.head_dim),
            k_norm_weight=torch.zeros(self.config.head_dim),
            linears=linears,
            exact_query=exact_query,
            exact_key=exact_key,
            exact_value=exact_value,
            exact_attention_output=exact_attention,
            exact_swiglu_output=exact_swiglu,
        )
        torch.testing.assert_close(
            trace.attention.core_output,
            exact_attention.reshape(
                tokens, self.config.num_query_heads, self.config.head_dim
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(trace.activated, exact_swiglu, rtol=0, atol=0)
        torch.testing.assert_close(
            trace.attention.query,
            exact_query.reshape(
                tokens, self.config.num_query_heads, self.config.head_dim
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            trace.attention.key,
            exact_key.reshape(
                tokens, self.config.num_kv_heads, self.config.head_dim
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            trace.attention.value,
            exact_value.reshape(
                tokens, self.config.num_kv_heads, self.config.head_dim
            ),
            rtol=0,
            atol=0,
        )
        trace.output.square().sum().backward()


if __name__ == "__main__":
    unittest.main()
