#!/usr/bin/env python3
"""Tests for the captured real-GDN parity certificate."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nvfp4_ste as STE  # noqa: E402
import validate_nvfp4_gdn_capture as VALIDATE  # noqa: E402


class GdnCaptureValidationTest(unittest.TestCase):
    def make_payload(self, path: Path, *, perturb: float = 0.0) -> None:
        generator = torch.Generator().manual_seed(4)
        tensors = {}
        tokens = 2
        for layer in VALIDATE.LAYERS:
            q = torch.randn(tokens, 16, 128, generator=generator)
            k = torch.randn(tokens, 16, 128, generator=generator)
            v = torch.randn(tokens, 48, 128, generator=generator)
            g = -torch.rand(tokens, 48, generator=generator)
            beta = torch.sigmoid(torch.randn(tokens, 48, generator=generator))
            state = torch.zeros(48, 128, 128)
            trace = STE.gdn_reference_forward(q, k, v, g, beta, state)
            output = trace.output + perturb
            final = trace.final_state + perturb
            prefix = f"gdn.layer{layer}."
            tensors.update(
                {
                    prefix + "q": q.unsqueeze(0),
                    prefix + "k": k.unsqueeze(0),
                    prefix + "v": v.unsqueeze(0),
                    prefix + "log_g": g.unsqueeze(0),
                    prefix + "beta": beta.unsqueeze(0),
                    prefix + "initial_state": state.unsqueeze(0),
                    prefix + "chunk_output": output.unsqueeze(0),
                    prefix + "core_output": output,
                    prefix + "final_state": final.unsqueeze(0),
                }
            )
        torch.save(
            {
                "model_revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
                "mode": "eager",
                "prompt": "test",
                "prompt_token_ids": [1, 2],
                "tensors": tensors,
            },
            path,
        )

    def test_exact_synthetic_capture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.pt"
            self.make_payload(path)
            result = VALIDATE.validate_capture(path)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["layers"]["61"]["core_equals_chunk_bitwise"])

    def test_large_recurrence_error_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.pt"
            self.make_payload(path, perturb=10.0)
            result = VALIDATE.validate_capture(path)
        self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
