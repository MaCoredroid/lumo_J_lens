#!/usr/bin/env python3
"""Unit tests for the NVFP4/FP8-STE surrogate primitives."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "nvfp4_ste", ROOT / "scripts" / "nvfp4_ste.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NvFp4DecodeTest(unittest.TestCase):
    def test_low_nibble_is_first_and_sign_bit_is_applied(self) -> None:
        packed = torch.tensor([[0x10, 0xF8]], dtype=torch.uint8)
        actual = MODULE.decode_e2m1_bytes(packed, dtype=torch.float32)
        expected = torch.tensor([[0.0, 0.5, -0.0, -6.0]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_per_block_and_global_scales_are_multiplied(self) -> None:
        packed = torch.tensor([[0x21, 0x43]], dtype=torch.uint8)
        scales = torch.tensor([[2.0, 3.0]], dtype=torch.float8_e4m3fn)
        global_scale = torch.tensor(0.5)
        actual = MODULE.dequantize_nvfp4_weight(
            packed, scales, global_scale, block_size=2
        )
        expected = torch.tensor([[0.5, 1.0, 2.25, 3.0]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class ExactForwardSteTest(unittest.TestCase):
    def test_w4_forward_is_exact_and_backward_uses_effective_weight(self) -> None:
        x = torch.tensor([[1.0, 2.0]], requires_grad=True)
        exact = torch.tensor([[7.25, -9.5]])
        weight = torch.tensor([[1.0, 3.0], [-2.0, 4.0]])
        output = MODULE.exact_w4a16_linear(x, exact, weight)
        torch.testing.assert_close(output, exact, rtol=0, atol=0)
        output.backward(torch.tensor([[2.0, -1.0]]))
        torch.testing.assert_close(x.grad, torch.tensor([[4.0, 2.0]]))

    def test_fp8_identity_ste_includes_saturated_inputs(self) -> None:
        x = torch.tensor([[0.0, 447.0, 449.0]], requires_grad=True)
        exact = torch.tensor([[123.0]])
        weight = torch.ones(1, 3)
        output = MODULE.exact_fp8_linear_ste(
            x, exact, weight, torch.tensor(1.0)
        )
        output.backward()
        torch.testing.assert_close(x.grad, torch.tensor([[1.0, 1.0, 1.0]]))

    def test_fp8_clipped_ste_masks_saturated_inputs(self) -> None:
        x = torch.tensor([[0.0, 447.0, 449.0]], requires_grad=True)
        exact = torch.tensor([[123.0]])
        weight = torch.ones(1, 3)
        output = MODULE.exact_fp8_linear_ste(
            x,
            exact,
            weight,
            torch.tensor(1.0),
            ste_policy="clipped",
        )
        output.backward()
        torch.testing.assert_close(x.grad, torch.tensor([[1.0, 1.0, 0.0]]))

    def test_batched_vjp_works_through_exact_forward_function(self) -> None:
        x = torch.randn(3, 2, requires_grad=True)
        exact = torch.randn(3, 4)
        weight = torch.randn(4, 2)
        output = MODULE.exact_w4a16_linear(x, exact, weight)
        cotangents = torch.randn(5, 3, 4)
        (actual,) = torch.autograd.grad(
            output,
            x,
            grad_outputs=cotangents,
            is_grads_batched=True,
        )
        expected = cotangents @ weight
        torch.testing.assert_close(actual, expected)

    def test_backward_returns_the_input_dtype(self) -> None:
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        exact = torch.randn(2, 4, dtype=torch.float32)
        weight = torch.randn(4, 3, dtype=torch.float32)
        MODULE.exact_w4a16_linear(x, exact, weight).sum().backward()
        self.assertEqual(x.grad.dtype, torch.float64)
        torch.testing.assert_close(
            x.grad,
            weight.sum(dim=0).to(torch.float64).expand_as(x),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_exact_value_uses_captured_forward_and_surrogate_backward(self) -> None:
        surrogate = torch.tensor([1.0, 2.0], requires_grad=True)
        exact = torch.tensor([8.0, 9.0])
        output = MODULE.exact_value(surrogate.square(), exact)
        torch.testing.assert_close(output, exact, rtol=0, atol=0)
        output.sum().backward()
        torch.testing.assert_close(surrogate.grad, torch.tensor([2.0, 4.0]))


class GatedDeltaRuleTest(unittest.TestCase):
    def make_primals(self, *, repeat_factor: int = 3):
        generator = torch.Generator().manual_seed(123)
        tokens, key_heads, key_dim, value_dim = 5, 2, 4, 3
        value_heads = key_heads * repeat_factor
        query = torch.randn(tokens, key_heads, key_dim, generator=generator)
        key = torch.randn(tokens, key_heads, key_dim, generator=generator)
        value = torch.randn(tokens, value_heads, value_dim, generator=generator)
        log_decay = -torch.rand(tokens, value_heads, generator=generator)
        beta = torch.sigmoid(torch.randn(tokens, value_heads, generator=generator))
        state = torch.randn(
            value_heads, value_dim, key_dim, generator=generator
        ) * 0.1
        return query, key, value, log_decay, beta, state

    def test_batched_analytic_vjp_matches_torch_autograd(self) -> None:
        primals = self.make_primals()
        cotangents = torch.randn(3, 5, 6, 3)
        actual = MODULE.gdn_vjp_batched(*primals[:5], cotangents, primals[5])

        references = [[] for _ in range(6)]
        for cotangent in cotangents:
            leaves = [value.detach().requires_grad_(True) for value in primals]
            trace = MODULE.gdn_reference_forward(*leaves[:5], leaves[5])
            grads = torch.autograd.grad(trace.output, leaves, cotangent)
            for bucket, grad in zip(references, grads, strict=True):
                bucket.append(grad)

        expected = [torch.stack(bucket) for bucket in references]
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)

    def test_final_state_cotangent_is_included(self) -> None:
        primals = self.make_primals(repeat_factor=1)
        grad_output = torch.zeros(2, 5, 2, 3)
        grad_final = torch.randn(2, 2, 3, 4)
        actual = MODULE.gdn_vjp_batched(
            *primals[:5],
            grad_output,
            primals[5],
            grad_final_state=grad_final,
        )
        references = [[] for _ in range(6)]
        for cotangent in grad_final:
            leaves = [value.detach().requires_grad_(True) for value in primals]
            trace = MODULE.gdn_reference_forward(*leaves[:5], leaves[5])
            grads = torch.autograd.grad(
                trace.final_state,
                leaves,
                cotangent,
                allow_unused=True,
            )
            grads = tuple(
                torch.zeros_like(leaf) if grad is None else grad
                for leaf, grad in zip(leaves, grads, strict=True)
            )
            for bucket, grad in zip(references, grads, strict=True):
                bucket.append(grad)
        for got, bucket in zip(actual, references, strict=True):
            torch.testing.assert_close(
                got, torch.stack(bucket), rtol=2e-5, atol=2e-5
            )

    def test_repeat_factor_reduces_query_and_key_gradients(self) -> None:
        primals = self.make_primals(repeat_factor=3)
        trace = MODULE.gdn_reference_forward(*primals[:5], primals[5])
        self.assertEqual(trace.output.shape, (5, 6, 3))
        actual = MODULE.gdn_vjp_batched(
            *primals[:5], torch.ones(1, 5, 6, 3), primals[5]
        )
        self.assertEqual(actual.query.shape, (1, 5, 2, 4))
        self.assertEqual(actual.key.shape, (1, 5, 2, 4))

    def test_adjoint_identity(self) -> None:
        primals = self.make_primals()
        directions = [torch.randn_like(value) for value in primals]
        cotangent = torch.randn(1, 5, 6, 3)
        vjp = MODULE.gdn_vjp_batched(
            *primals[:5], cotangent, primals[5]
        )
        lhs = sum(
            (gradient[0] * direction).sum()
            for gradient, direction in zip(vjp, directions, strict=True)
        )

        def output_for(*values):
            return MODULE.gdn_reference_forward(
                *values[:5], values[5]
            ).output

        _, tangent = torch.func.jvp(output_for, primals, tuple(directions))
        rhs = (tangent * cotangent[0]).sum()
        torch.testing.assert_close(lhs, rhs, rtol=3e-5, atol=3e-5)

    def test_checkpoint_intervals_are_equivalent(self) -> None:
        primals = self.make_primals()
        cotangents = torch.randn(2, 5, 6, 3)
        expected = MODULE.gdn_vjp_batched(
            *primals[:5],
            cotangents,
            primals[5],
            checkpoint_interval=1,
        )
        for interval in (2, 16):
            actual = MODULE.gdn_vjp_batched(
                *primals[:5],
                cotangents,
                primals[5],
                checkpoint_interval=interval,
            )
            for got, want in zip(actual, expected, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_exact_recurrence_uses_captured_forward_and_analytic_backward(self) -> None:
        primals = self.make_primals()
        trace = MODULE.gdn_reference_forward(*primals[:5], primals[5])
        exact_output = torch.randn_like(trace.output)
        exact_final = torch.randn_like(trace.final_state)
        leaves = [value.detach().requires_grad_(True) for value in primals]
        output, final_state = MODULE.exact_gdn_recurrence(
            *leaves[:5],
            leaves[5],
            exact_output,
            exact_final,
        )
        torch.testing.assert_close(output, exact_output, rtol=0, atol=0)
        torch.testing.assert_close(final_state, exact_final, rtol=0, atol=0)

        grad_output = torch.randn_like(output)
        grad_final = torch.randn_like(final_state)
        actual = torch.autograd.grad(
            (output, final_state), leaves, (grad_output, grad_final)
        )
        expected = MODULE.gdn_vjp_batched(
            *primals[:5],
            grad_output.unsqueeze(0),
            primals[5],
            grad_final_state=grad_final.unsqueeze(0),
        )
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want.squeeze(0))

    def test_exact_recurrence_supports_batched_autograd_vjps(self) -> None:
        primals = self.make_primals()
        trace = MODULE.gdn_reference_forward(*primals[:5], primals[5])
        query = primals[0].detach().requires_grad_(True)
        output, _ = MODULE.exact_gdn_recurrence(
            query,
            *primals[1:5],
            primals[5],
            trace.output,
            trace.final_state,
        )
        cotangents = torch.randn(3, *output.shape)
        (actual,) = torch.autograd.grad(
            output,
            query,
            cotangents,
            is_grads_batched=True,
        )
        expected = MODULE.gdn_vjp_batched(
            *primals[:5], cotangents, primals[5]
        ).query
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_chunk_boundary_lengths_match_autograd(self) -> None:
        generator = torch.Generator().manual_seed(991)
        for tokens in (1, 15, 16, 17, 63, 64, 65):
            primals = (
                torch.randn(tokens, 1, 2, generator=generator),
                torch.randn(tokens, 1, 2, generator=generator),
                torch.randn(tokens, 1, 3, generator=generator),
                -torch.rand(tokens, 1, generator=generator),
                torch.sigmoid(torch.randn(tokens, 1, generator=generator)),
                torch.randn(1, 3, 2, generator=generator) * 0.1,
            )
            cotangent = torch.randn(1, tokens, 1, 3, generator=generator)
            actual = MODULE.gdn_vjp_batched(
                *primals[:5], cotangent, primals[5]
            )
            leaves = [value.detach().requires_grad_(True) for value in primals]
            output = MODULE.gdn_reference_forward(
                *leaves[:5], leaves[5]
            ).output
            expected = torch.autograd.grad(output, leaves, cotangent[0])
            for got, want in zip(actual, expected, strict=True):
                torch.testing.assert_close(
                    got.squeeze(0), want, rtol=3e-5, atol=3e-5
                )

    def test_qwen_real_head_dimensions_single_token(self) -> None:
        generator = torch.Generator().manual_seed(36)
        query = torch.randn(1, 16, 128, generator=generator)
        key = torch.randn(1, 16, 128, generator=generator)
        value = torch.randn(1, 48, 128, generator=generator)
        log_decay = -torch.rand(1, 48, generator=generator)
        beta = torch.sigmoid(torch.randn(1, 48, generator=generator))
        cotangent = torch.randn(1, 1, 48, 128, generator=generator)
        actual = MODULE.gdn_vjp_batched(
            query, key, value, log_decay, beta, cotangent
        )
        self.assertEqual(actual.query.shape, (1, 1, 16, 128))
        self.assertEqual(actual.key.shape, (1, 1, 16, 128))
        self.assertEqual(actual.value.shape, (1, 1, 48, 128))
        self.assertEqual(actual.initial_state.shape, (1, 48, 128, 128))
        for tensor in actual:
            self.assertTrue(bool(torch.isfinite(tensor).all()))


class SurroundingOpsTest(unittest.TestCase):
    def test_causal_conv_has_no_future_dependency(self) -> None:
        x = torch.arange(1.0, 6.0).reshape(5, 1).requires_grad_(True)
        weight = torch.tensor([[1.0, 2.0, 3.0]])
        output = MODULE.causal_depthwise_conv1d_silu(x, weight)
        output[2].backward()
        self.assertEqual(float(x.grad[3:].abs().sum()), 0.0)

    def test_gemma_rms_norm_uses_one_plus_weight(self) -> None:
        x = torch.tensor([[3.0, 4.0]])
        weight = torch.zeros(2)
        actual = MODULE.rms_norm(x, weight, 0.0, gemma_offset=True)
        expected = x / torch.sqrt(x.square().mean())
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
