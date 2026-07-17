#!/usr/bin/env python3
"""Tests for the blockwise NVFP4/FP8-STE Jacobian estimator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fit_jlens_nvfp4_ste", ROOT / "scripts" / "fit_jlens_nvfp4_ste.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ReverseReplayTest(unittest.TestCase):
    def test_blockwise_rows_match_retained_graph_autograd(self) -> None:
        torch.manual_seed(4)
        tokens, hidden = 4, 5
        weights = {
            1: torch.randn(hidden, hidden) / 4,
            2: torch.randn(hidden, hidden) / 4,
            3: torch.randn(hidden, hidden) / 4,
        }
        h0 = torch.randn(tokens, hidden)
        captures = {0: h0}
        current = h0
        for layer in range(1, 4):
            current = current + torch.tanh(current @ weights[layer])
            captures[layer] = current.detach()

        def replay(layer: int, logical_input: torch.Tensor) -> torch.Tensor:
            canonical = logical_input + torch.tanh(logical_input @ weights[layer])
            exact = captures[layer].to(logical_input.device)
            return canonical + (exact - canonical).detach()

        positions = torch.tensor([1, 2])
        actual = MODULE.reverse_replay_rows(
            captures,
            replay,
            positions,
            1,
            4,
            target_layer=3,
        )

        rooted = h0.detach().requires_grad_(True)
        retained = {0: rooted}
        current = rooted
        for layer in range(1, 4):
            current = current + torch.tanh(current @ weights[layer])
            retained[layer] = current
        cotangents = MODULE.target_cotangent_rows(
            tokens=tokens,
            hidden=hidden,
            valid_positions=positions,
            row_start=1,
            row_stop=4,
            dtype=current.dtype,
            device=current.device,
        )
        for source in range(3):
            (gradient,) = torch.autograd.grad(
                current,
                retained[source],
                grad_outputs=cotangents,
                is_grads_batched=True,
                retain_graph=True,
            )
            expected = gradient[:, positions].mean(dim=1)
            torch.testing.assert_close(actual.rows[source], expected)

    def test_replay_must_reproduce_captured_forward_exactly(self) -> None:
        captures = {0: torch.zeros(2, 3), 1: torch.ones(2, 3)}
        with self.assertRaisesRegex(RuntimeError, "not exact"):
            MODULE.reverse_replay_rows(
                captures,
                lambda _layer, value: value + 0.5,
                torch.tensor([0]),
                0,
                1,
                target_layer=1,
            )

    def test_matrix_row_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            matrices = MODULE.prepare_matrix_files(
                Path(temporary), source_layers=(0, 1), hidden=3
            )
            chunk = MODULE.ReverseRows(
                row_start=1,
                row_stop=3,
                rows={
                    0: torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    1: torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
                },
                propagated_cotangent=torch.empty(0),
            )
            MODULE.write_reverse_rows(matrices, chunk)
            np.testing.assert_array_equal(
                np.asarray(matrices[0][1:3]), chunk.rows[0].numpy()
            )
            np.testing.assert_array_equal(
                np.asarray(matrices[1][1:3]), chunk.rows[1].numpy()
            )


class ContractTest(unittest.TestCase):
    def test_position_mask_skips_final_token(self) -> None:
        self.assertEqual(
            MODULE.valid_estimator_positions(6, 2).tolist(),
            [2, 3, 4],
        )

    def test_target_cotangent_is_future_summed(self) -> None:
        positions = torch.tensor([1, 3])
        value = MODULE.target_cotangent_rows(
            tokens=4,
            hidden=3,
            valid_positions=positions,
            row_start=1,
            row_stop=3,
            dtype=torch.float32,
            device="cpu",
        )
        self.assertEqual(tuple(value.shape), (2, 4, 3))
        self.assertEqual(value[0, :, 1].tolist(), [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(value[1, :, 2].tolist(), [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(float(value.sum()), 4.0)


if __name__ == "__main__":
    unittest.main()
