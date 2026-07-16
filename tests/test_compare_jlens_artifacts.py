#!/usr/bin/env python3
"""Focused tests for streamed Jacobian Lens matrix comparison."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
if torch is not None:
    SPEC = importlib.util.spec_from_file_location(
        "compare_jlens_artifacts", SCRIPTS / "compare_jlens_artifacts.py"
    )
    assert SPEC and SPEC.loader
    MODULE = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = MODULE
    SPEC.loader.exec_module(MODULE)
else:
    MODULE = None


@unittest.skipIf(torch is None, "torch is installed in .venv-fit")
class LayerComparisonTest(unittest.TestCase):
    def test_identical_matrix_metrics(self) -> None:
        matrix = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        record, totals = MODULE.compare_layer_matrices(
            matrix, matrix, layer=0, d_model=2, row_chunk=1
        )
        self.assertEqual(record["relative_frobenius_difference"], 0.0)
        self.assertAlmostEqual(record["frobenius_cosine"], 1.0)
        self.assertAlmostEqual(record["local"]["trace"], 5.0)
        self.assertAlmostEqual(record["local"]["frobenius_norm"], math.sqrt(15))
        self.assertAlmostEqual(
            record["local"]["best_scalar_identity"]["scalar"], 2.5
        )
        self.assertEqual(record["row_wise_cosine"]["quantiles"]["p050"], 1.0)
        self.assertEqual(totals.difference_norm_squared, 0.0)

    def test_known_difference_and_row_cosines(self) -> None:
        local = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
        public = torch.eye(2)
        record, _ = MODULE.compare_layer_matrices(
            local, public, layer=4, d_model=2, row_chunk=2
        )
        self.assertAlmostEqual(record["frobenius_cosine"], -0.5)
        self.assertAlmostEqual(record["relative_frobenius_difference"], math.sqrt(3))
        row = record["row_wise_cosine"]
        self.assertAlmostEqual(row["quantiles"]["p000"], -1.0)
        self.assertAlmostEqual(row["quantiles"]["p050"], -0.5)
        self.assertAlmostEqual(row["quantiles"]["p100"], 0.0)

    def test_nonfinite_and_zero_rows_fail_closed(self) -> None:
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            MODULE.compare_layer_matrices(
                torch.tensor([[1.0, float("nan")], [0.0, 1.0]]),
                torch.eye(2),
                layer=0,
                d_model=2,
                row_chunk=1,
            )
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            MODULE.compare_layer_matrices(
                torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
                torch.eye(2),
                layer=0,
                d_model=2,
                row_chunk=1,
            )


@unittest.skipIf(torch is None, "torch is installed in .venv-fit")
class ArtifactComparisonTest(unittest.TestCase):
    def write_checkpoint(
        self, path: Path, matrices: dict[int, torch.Tensor]
    ) -> None:
        torch.save(
            {
                "J": matrices,
                "d_model": 2,
                "n_prompts": 1,
                "source_layers": [0, 1],
            },
            path,
        )

    def test_tiny_memory_mapped_artifacts_and_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public = root / "public.pt"
            local = root / "local.pt"
            self.write_checkpoint(public, {0: torch.eye(2), 1: torch.eye(2)})
            self.write_checkpoint(
                local, {0: torch.eye(2), 1: 2.0 * torch.eye(2)}
            )
            layers, aggregate = MODULE.compare_artifact_matrices(
                local,
                public,
                d_model=2,
                source_layers=(0, 1),
                row_chunk=1,
            )

        self.assertEqual([record["layer"] for record in layers], [0, 1])
        self.assertEqual(layers[0]["relative_frobenius_difference"], 0.0)
        self.assertEqual(layers[1]["relative_frobenius_difference"], 1.0)
        self.assertAlmostEqual(
            aggregate["global_relative_frobenius_difference"], math.sqrt(0.5)
        )
        self.assertAlmostEqual(aggregate["global_frobenius_cosine"], 3 / math.sqrt(10))
        self.assertEqual(aggregate["all_rows_cosine"]["count"], 4)

    def test_checkpoint_layer_contract_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.pt"
            second = root / "second.pt"
            torch.save(
                {"J": {0: torch.eye(2)}, "d_model": 2, "n_prompts": 1},
                first,
            )
            self.write_checkpoint(second, {0: torch.eye(2), 1: torch.eye(2)})
            with self.assertRaisesRegex(ValueError, "source layers mismatch"):
                MODULE.compare_artifact_matrices(
                    first,
                    second,
                    d_model=2,
                    source_layers=(0, 1),
                    row_chunk=1,
                )

    def test_artifact_verifiers_are_mandatory(self) -> None:
        calls = []

        def local_verifier(path, **kwargs):
            calls.append(("local", path, kwargs))
            return {"sha256": "a" * 64, "n_prompts": 10}

        def public_file_verifier(path):
            calls.append(("public_file", path, {}))
            return {"sha256": "b" * 64, "size_bytes": 123}

        def public_checkpoint_verifier(path, **kwargs):
            calls.append(("public_checkpoint", path, kwargs))
            return {"d_model": 5120, "source_layers": list(range(63))}

        records = MODULE.verify_artifacts(
            local_path=Path("local.pt"),
            local_sha256="a" * 64,
            local_provenance=Path("local.provenance.json"),
            public_path=Path("public.pt"),
            local_verifier=local_verifier,
            public_file_verifier=public_file_verifier,
            public_checkpoint_verifier=public_checkpoint_verifier,
        )
        self.assertEqual(records["local"]["sha256"], "a" * 64)
        self.assertEqual(records["public"]["sha256"], "b" * 64)
        self.assertEqual(calls[0][2]["expected_sha256"], "a" * 64)
        self.assertTrue(calls[0][2]["check_finite"])
        self.assertFalse(calls[2][2]["check_finite"])


if __name__ == "__main__":
    unittest.main()
