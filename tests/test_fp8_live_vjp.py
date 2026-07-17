from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fp8_live_vjp", ROOT / "scripts" / "fp8_live_vjp.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

STE_SPEC = importlib.util.spec_from_file_location(
    "nvfp4_ste_for_live_fp8_vjp_test", ROOT / "scripts" / "nvfp4_ste.py"
)
assert STE_SPEC and STE_SPEC.loader
STE_MODULE = importlib.util.module_from_spec(STE_SPEC)
sys.modules[STE_SPEC.name] = STE_MODULE
STE_SPEC.loader.exec_module(STE_MODULE)


def make_case(*, m: int = 3, n: int = 16, k: int = 32):
    generator = torch.Generator().manual_seed(3617)
    values = torch.randn(k, n, generator=generator) * 2.0
    weight = values.to(torch.float8_e4m3fn)
    weight_scale = torch.tensor(0.25, dtype=torch.float32)
    input_scale = torch.tensor(0.01, dtype=torch.float32)
    grad = torch.randn(m, n, generator=generator, dtype=torch.float32)
    return grad, weight, weight_scale, input_scale


def dense_reference(grad, weight, weight_scale, *, dot_dtype):
    dense = STE_MODULE.dequantize_runtime_fp8_weight(
        weight,
        weight_scale,
        transposed=True,
        dtype=dot_dtype,
    )
    return grad.to(dot_dtype).float() @ dense.float()


class LiveFp8VjpTest(unittest.TestCase):
    def assert_batched_exact_linear_matches_dense(
        self,
        device: torch.device | str,
        *,
        policy: str,
    ) -> None:
        _grad, weight, weight_scale, input_scale = make_case()
        weight = weight.to(device)
        weight_scale = weight_scale.to(device)
        input_scale = input_scale.to(device)
        dense_weight = STE_MODULE.dequantize_runtime_fp8_weight(
            weight,
            weight_scale,
            transposed=True,
            dtype=torch.bfloat16,
        )

        generator = torch.Generator().manual_seed(9914)
        input_values = torch.randn(3, 32, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        input_values[0, 0] = 5.0
        input_values[1, 1] = -5.0
        exact = torch.randn(3, 16, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        cotangents = torch.randn(5, 3, 16, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        live_input = input_values.detach().clone().requires_grad_(True)
        dense_input = input_values.detach().clone().requires_grad_(True)

        live_output = MODULE.exact_live_fp8_linear(
            live_input,
            exact,
            weight,
            weight_scale,
            input_scale,
            ste_policy=policy,
        )
        dense_output = STE_MODULE.exact_fp8_linear_ste(
            dense_input,
            exact,
            dense_weight,
            input_scale,
            ste_policy=policy,
        )
        torch.testing.assert_close(live_output, exact, rtol=0, atol=0)
        torch.testing.assert_close(dense_output, exact, rtol=0, atol=0)
        (actual,) = torch.autograd.grad(
            live_output,
            live_input,
            grad_outputs=cotangents,
            is_grads_batched=True,
        )
        (expected,) = torch.autograd.grad(
            dense_output,
            dense_input,
            grad_outputs=cotangents,
            is_grads_batched=True,
        )

        self.assertEqual(actual.dtype, live_input.dtype)
        self.assertEqual(tuple(actual.shape), (5, 3, 32))
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        if policy == "clipped":
            torch.testing.assert_close(
                actual[:, 0, 0], torch.zeros_like(actual[:, 0, 0]), rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual[:, 1, 1], torch.zeros_like(actual[:, 1, 1]), rtol=0, atol=0
            )

    def test_torch_fp32_matches_dense_reference_exactly(self) -> None:
        grad, weight, weight_scale, _input_scale = make_case()
        expected = dense_reference(
            grad, weight, weight_scale, dot_dtype=torch.float32
        )
        actual = MODULE.live_fp8_input_vjp_torch(
            grad,
            weight,
            weight_scale,
            dot_dtype=torch.float32,
            tile_k=weight.shape[0],
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_torch_bf16_matches_dense_quantized_contract(self) -> None:
        grad, weight, weight_scale, _input_scale = make_case()
        expected = dense_reference(
            grad, weight, weight_scale, dot_dtype=torch.bfloat16
        )
        actual = MODULE.live_fp8_input_vjp_torch(
            grad,
            weight,
            weight_scale,
            dot_dtype=torch.bfloat16,
            tile_k=weight.shape[0],
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_k_tiling_is_exact(self) -> None:
        grad, weight, weight_scale, _input_scale = make_case(k=48)
        whole = MODULE.live_fp8_input_vjp_torch(
            grad, weight, weight_scale, dot_dtype=torch.float32, tile_k=48
        )
        tiled = MODULE.live_fp8_input_vjp_torch(
            grad, weight, weight_scale, dot_dtype=torch.float32, tile_k=7
        )
        torch.testing.assert_close(tiled, whole, rtol=0, atol=0)

    def test_resource_estimate_accounts_for_avoided_dense_weight(self) -> None:
        estimate = MODULE.resource_estimate(m=8, n=16384, k=5120, tile_k=128)
        self.assertEqual(estimate["live_fp8_weight_bytes"], 5120 * 16384)
        self.assertEqual(
            estimate["full_dequantized_weight_bytes_bf16_avoided"],
            5120 * 16384 * 2,
        )
        self.assertLess(
            estimate["torch_fallback_max_weight_tile_bytes_fp32"],
            estimate["full_dequantized_weight_bytes_fp32_avoided"],
        )

    def test_validation_rejects_transposed_weight(self) -> None:
        grad, weight, weight_scale, _input_scale = make_case()
        with self.assertRaisesRegex(ValueError, "post-load weight N"):
            MODULE.live_fp8_input_vjp_torch(grad, weight.T, weight_scale)

    def test_exact_wrapper_identity_cpu(self) -> None:
        self.assert_batched_exact_linear_matches_dense("cpu", policy="identity")

    def test_exact_wrapper_clipped_cpu(self) -> None:
        self.assert_batched_exact_linear_matches_dense("cpu", policy="clipped")

    def test_exact_wrapper_preserves_float64_gradient_dtype(self) -> None:
        _grad, weight, weight_scale, input_scale = make_case(m=2, n=16, k=32)
        inputs = torch.randn(2, 32, dtype=torch.float64, requires_grad=True)
        exact = torch.randn(2, 16, dtype=torch.float32)
        cotangents = torch.randn(3, 2, 16, dtype=torch.float32)
        output = MODULE.exact_live_fp8_linear(
            inputs,
            exact,
            weight,
            weight_scale,
            input_scale,
        )
        (actual,) = torch.autograd.grad(
            output,
            inputs,
            grad_outputs=cotangents,
            is_grads_batched=True,
        )
        self.assertEqual(actual.dtype, torch.float64)
        self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_exact_wrapper_rejects_unknown_policy(self) -> None:
        _grad, weight, weight_scale, input_scale = make_case()
        with self.assertRaisesRegex(ValueError, "ste_policy"):
            MODULE.exact_live_fp8_linear(
                torch.randn(3, 32),
                torch.randn(3, 16),
                weight,
                weight_scale,
                input_scale,
                ste_policy="opaque",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_exact_wrapper_identity_cuda(self) -> None:
        self.assert_batched_exact_linear_matches_dense("cuda", policy="identity")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_exact_wrapper_clipped_cuda(self) -> None:
        self.assert_batched_exact_linear_matches_dense("cuda", policy="clipped")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_triton_matches_exact_representable_case(self) -> None:
        m, n, k = 2, 16, 32
        weight = torch.full(
            (k, n), 2.0, dtype=torch.float8_e4m3fn, device="cuda"
        )
        weight_scale = torch.tensor(0.5, dtype=torch.float32, device="cuda")
        grad = torch.tensor(
            [[1.0] * n, [0.5] * n], dtype=torch.bfloat16, device="cuda"
        )
        actual = MODULE.live_fp8_input_vjp_triton(grad, weight, weight_scale)
        expected = torch.tensor(
            [[16.0] * k, [8.0] * k], dtype=torch.float32, device="cuda"
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_triton_matches_random_dense_bf16_reference(self) -> None:
        grad, weight, weight_scale, _input_scale = make_case(m=7, n=73, k=96)
        grad = grad.to(torch.bfloat16).cuda()
        weight = weight.cuda()
        weight_scale = weight_scale.cuda()
        dense = STE_MODULE.dequantize_runtime_fp8_weight(
            weight,
            weight_scale,
            transposed=True,
            dtype=torch.bfloat16,
        )
        expected = torch.mm(grad, dense, out_dtype=torch.float32)
        actual = MODULE.live_fp8_input_vjp_triton(grad, weight, weight_scale)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
