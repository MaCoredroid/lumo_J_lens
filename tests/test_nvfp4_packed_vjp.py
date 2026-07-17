from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nvfp4_packed_vjp as MODULE  # noqa: E402

STE_SPEC = importlib.util.spec_from_file_location(
    "nvfp4_ste_for_packed_vjp_test", ROOT / "scripts" / "nvfp4_ste.py"
)
assert STE_SPEC and STE_SPEC.loader
STE_MODULE = importlib.util.module_from_spec(STE_SPEC)
sys.modules[STE_SPEC.name] = STE_MODULE
STE_SPEC.loader.exec_module(STE_MODULE)


def make_case(*, m: int = 3, n: int = 5, k: int = 32):
    generator = torch.Generator().manual_seed(1716)
    packed = torch.randint(0, 256, (n, k // 2), generator=generator, dtype=torch.uint8)
    scale_values = torch.tensor(
        [0.5, 0.75, 1.0, 1.5], dtype=torch.float32
    )
    indices = torch.randint(
        0, len(scale_values), (n, k // 16), generator=generator
    )
    scales = scale_values[indices].to(torch.float8_e4m3fn)
    global_scale = torch.tensor(0.25, dtype=torch.float32)
    grad = torch.randn(m, n, generator=generator, dtype=torch.float32)
    return grad, packed, scales, global_scale


def dense_reference(grad, packed, scales, global_scale, *, dot_dtype):
    values = MODULE._decode_e2m1_rows(packed)
    weight = (
        values.reshape(packed.shape[0], -1, 16)
        * (scales.float() * global_scale).unsqueeze(-1)
    ).reshape(packed.shape[0], -1)
    return grad.to(dot_dtype).float() @ weight.to(dot_dtype).float()


class PackedNvFp4VjpTest(unittest.TestCase):
    def assert_batched_exact_linear_matches_dense(
        self, device: torch.device | str
    ) -> None:
        grad, packed, scales, global_scale = make_case(m=3, n=16, k=32)
        del grad
        packed = packed.to(device)
        scales = scales.to(device)
        global_scale = global_scale.to(device)
        dense_weight = STE_MODULE.dequantize_nvfp4_weight(
            packed,
            scales,
            global_scale,
            dtype=torch.bfloat16,
        )

        generator = torch.Generator().manual_seed(8241)
        input_values = torch.randn(3, 32, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        exact = torch.randn(3, 16, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        cotangents = torch.randn(5, 3, 16, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        packed_input = input_values.detach().clone().requires_grad_(True)
        dense_input = input_values.detach().clone().requires_grad_(True)

        packed_output = MODULE.exact_packed_nvfp4_linear(
            packed_input,
            exact,
            packed,
            scales,
            global_scale,
        )
        dense_output = STE_MODULE.exact_w4a16_linear(
            dense_input, exact, dense_weight
        )
        torch.testing.assert_close(packed_output, exact, rtol=0, atol=0)
        torch.testing.assert_close(dense_output, exact, rtol=0, atol=0)
        (actual,) = torch.autograd.grad(
            packed_output,
            packed_input,
            grad_outputs=cotangents,
            is_grads_batched=True,
        )
        (expected,) = torch.autograd.grad(
            dense_output,
            dense_input,
            grad_outputs=cotangents,
            is_grads_batched=True,
        )

        self.assertEqual(actual.dtype, packed_input.dtype)
        self.assertEqual(tuple(actual.shape), (5, 3, 32))
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_low_nibble_first_decode_and_sign(self) -> None:
        packed = torch.tensor([[0x10, 0xF8]], dtype=torch.uint8)
        actual = MODULE._decode_e2m1_rows(packed)
        expected = torch.tensor([[0.0, 0.5, -0.0, -6.0]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_streaming_fp32_matches_dense_reference_exactly(self) -> None:
        grad, packed, scales, global_scale = make_case()
        expected = dense_reference(
            grad, packed, scales, global_scale, dot_dtype=torch.float32
        )
        actual = MODULE.packed_nvfp4_input_vjp_torch(
            grad,
            packed,
            scales,
            global_scale,
            dot_dtype=torch.float32,
            tile_n=packed.shape[0],
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_streaming_bf16_matches_dense_quantized_contract(self) -> None:
        grad, packed, scales, global_scale = make_case()
        expected = dense_reference(
            grad, packed, scales, global_scale, dot_dtype=torch.bfloat16
        )
        actual = MODULE.packed_nvfp4_input_vjp_torch(
            grad,
            packed,
            scales,
            global_scale,
            dot_dtype=torch.bfloat16,
            tile_n=packed.shape[0],
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_streaming_chunking_is_numerically_stable(self) -> None:
        grad, packed, scales, global_scale = make_case(n=7)
        whole = MODULE.packed_nvfp4_input_vjp_torch(
            grad,
            packed,
            scales,
            global_scale,
            dot_dtype=torch.float32,
            tile_n=7,
        )
        tiled = MODULE.packed_nvfp4_input_vjp_torch(
            grad,
            packed,
            scales,
            global_scale,
            dot_dtype=torch.float32,
            tile_n=2,
        )
        torch.testing.assert_close(tiled, whole, rtol=1e-6, atol=1e-6)

    def test_resource_estimate_accounts_for_avoided_dense_weight(self) -> None:
        estimate = MODULE.resource_estimate(m=8, n=5120, k=17408, tile_n=128)
        self.assertEqual(
            estimate["full_dequantized_weight_bytes_bf16_avoided"],
            5120 * 17408 * 2,
        )
        self.assertEqual(
            estimate["raw_packed_weight_bytes"], 5120 * 17408 // 2
        )
        self.assertLess(
            estimate["torch_fallback_max_weight_tile_bytes"],
            estimate["full_dequantized_weight_bytes_bf16_avoided"],
        )

    def test_shape_validation_rejects_wrong_scale_grid(self) -> None:
        grad, packed, scales, global_scale = make_case()
        with self.assertRaisesRegex(ValueError, "block_scales shape"):
            MODULE.packed_nvfp4_input_vjp_torch(
                grad, packed, scales[:, :-1], global_scale
            )

    def test_exact_packed_linear_batched_autograd_cpu(self) -> None:
        self.assert_batched_exact_linear_matches_dense("cpu")

    def test_exact_packed_linear_preserves_float64_gradient_dtype(self) -> None:
        _grad, packed, scales, global_scale = make_case(m=2, n=5, k=32)
        inputs = torch.randn(2, 32, dtype=torch.float64, requires_grad=True)
        exact = torch.randn(2, 5, dtype=torch.float32)
        cotangents = torch.randn(3, 2, 5, dtype=torch.float32)
        output = MODULE.exact_packed_nvfp4_linear(
            inputs, exact, packed, scales, global_scale
        )
        (actual,) = torch.autograd.grad(
            output,
            inputs,
            grad_outputs=cotangents,
            is_grads_batched=True,
        )
        self.assertEqual(actual.dtype, torch.float64)
        self.assertTrue(bool(torch.isfinite(actual).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_exact_packed_linear_batched_autograd_cuda(self) -> None:
        self.assert_batched_exact_linear_matches_dense("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_triton_matches_exact_representable_small_case(self) -> None:
        # Powers-of-two values keep both BF16 products and the short reduction exact.
        m, n, k = 2, 16, 32
        packed = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda")
        scales = torch.ones(
            (n, k // 16), dtype=torch.float8_e4m3fn, device="cuda"
        )
        global_scale = torch.tensor(0.5, dtype=torch.float32, device="cuda")
        grad = torch.tensor(
            [[1.0] * n, [0.5] * n], dtype=torch.bfloat16, device="cuda"
        )
        actual = MODULE.packed_nvfp4_input_vjp_triton(
            grad, packed, scales, global_scale
        )
        expected = torch.tensor(
            [[8.0] * k, [4.0] * k], dtype=torch.float32, device="cuda"
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_triton_matches_random_dense_bf16_reference(self) -> None:
        grad, packed, scales, global_scale = make_case(m=7, n=73, k=96)
        grad = grad.to(torch.bfloat16).cuda()
        packed = packed.cuda()
        scales = scales.cuda()
        global_scale = global_scale.cuda()
        expected = dense_reference(
            grad, packed, scales, global_scale, dot_dtype=torch.bfloat16
        )
        actual = MODULE.packed_nvfp4_input_vjp_triton(
            grad, packed, scales, global_scale
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
