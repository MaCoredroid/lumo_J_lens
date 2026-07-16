#!/usr/bin/env python3
"""Focused tests for the vectorized NF4 Jacobian estimator."""

from __future__ import annotations

import importlib.util
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # The fitter lives in the optional Torch environment.
    torch = None
    nn = None


ROOT = Path(__file__).resolve().parents[1]
if torch is not None:
    SPEC = importlib.util.spec_from_file_location(
        "fit_jlens_nf4", ROOT / "scripts" / "fit_jlens_nf4.py"
    )
    assert SPEC and SPEC.loader
    MODULE = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = MODULE
    SPEC.loader.exec_module(MODULE)
else:
    MODULE = None


if torch is not None:
    class CausalBlock(nn.Module):
        def __init__(self, width: int, seed: int) -> None:
            super().__init__()
            generator = torch.Generator().manual_seed(seed)
            self.weight = nn.Parameter(
                torch.randn(width, width, generator=generator, dtype=torch.float64)
                / width
            )

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            mixed = torch.cumsum(hidden, dim=1) @ self.weight.T
            return hidden + torch.tanh(mixed)


    class TinyLensModel(nn.Module):
        def __init__(self, width: int = 6, layers: int = 4) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                CausalBlock(width, 100 + i) for i in range(layers)
            )
            for parameter in self.parameters():
                parameter.requires_grad_(False)

        def activations(self, inputs: torch.Tensor) -> list[torch.Tensor]:
            result: list[torch.Tensor] = []
            hidden = inputs
            for index, layer in enumerate(self.layers):
                hidden = layer(hidden)
                if index == 0:
                    hidden.requires_grad_(True)
                result.append(hidden)
            return result


    def sequential_estimator(
        target: torch.Tensor, sources: list[torch.Tensor], valid: torch.Tensor
    ) -> list[torch.Tensor]:
        width = target.shape[-1]
        matrices = [torch.zeros(width, width, dtype=torch.float32) for _ in sources]
        for dimension in range(width):
            cotangent = torch.zeros_like(target)
            cotangent[0, valid, dimension] = 1
            gradients = torch.autograd.grad(
                target,
                sources,
                grad_outputs=cotangent,
                retain_graph=dimension + 1 < width,
            )
            for matrix, gradient in zip(matrices, gradients, strict=True):
                matrix[dimension] = gradient[0, valid].float().mean(dim=0)
        return matrices


@unittest.skipIf(torch is None, "torch is installed in the optional fit environment")
class FutureSummedEstimatorTest(unittest.TestCase):
    def test_normative_position_mask_excludes_final_token(self) -> None:
        positions = MODULE.valid_estimator_positions(128, 16)
        self.assertEqual(positions.tolist(), list(range(16, 127)))
        self.assertNotIn(127, positions.tolist())

    def test_vectorized_rows_match_sequential_autograd(self) -> None:
        torch.manual_seed(11)
        inputs = torch.randn(1, 7, 6, dtype=torch.float64)
        valid = torch.arange(2, 7)

        vector_model = TinyLensModel()
        vector_activations = vector_model.activations(inputs)
        actual = [torch.zeros(6, 6) for _ in range(3)]
        for start, stop, chunks in MODULE.iter_future_summed_row_chunks(
            vector_activations[3],
            vector_activations[:3],
            valid,
            cotangent_batch=2,
        ):
            for matrix, chunk in zip(actual, chunks, strict=True):
                matrix[start:stop] = chunk

        reference_model = TinyLensModel()
        reference_activations = reference_model.activations(inputs)
        expected = sequential_estimator(
            reference_activations[3], reference_activations[:3], valid
        )
        for vectorized, sequential in zip(actual, expected, strict=True):
            torch.testing.assert_close(vectorized, sequential, rtol=0, atol=0)

    def test_source_position_mean_and_future_sum_are_both_applied(self) -> None:
        target = torch.tensor(
            [[[1.0], [2.0], [3.0]]], dtype=torch.float64, requires_grad=True
        )
        # Every later target depends on all earlier source positions.
        target = torch.cumsum(target, dim=1)
        source = target.clone()
        # Rebuild a causal target downstream of the captured source.
        final = torch.cumsum(source, dim=1)
        valid = torch.tensor([1, 2])
        row = MODULE.future_summed_vjp_rows(
            final, [source], valid, 0, 1, retain_graph=False
        )[0]
        # d(sum(final[1:])) / d(source) is [2, 2, 1]; average source
        # positions 1 and 2 -> (2 + 1) / 2.
        torch.testing.assert_close(row, torch.tensor([[1.5]], dtype=torch.float32))

    def test_invalid_target_batch_is_rejected(self) -> None:
        target = torch.zeros(2, 3, 4, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "shape"):
            MODULE.future_summed_vjp_rows(
                target, [target], torch.tensor([1]), 0, 1, retain_graph=False
            )

    def test_same_graph_sequential_oracle_matches_batched_rows(self) -> None:
        inputs = torch.randn(1, 7, 6, dtype=torch.float64)
        activations = TinyLensModel().activations(inputs)
        valid = torch.arange(2, 7)
        batched = [torch.empty(3, 6) for _ in range(2)]
        for start, stop, chunks in MODULE.iter_future_summed_row_chunks(
            activations[3],
            activations[1:3],
            valid,
            row_stop=3,
            cotangent_batch=2,
            retain_graph_after=True,
        ):
            for matrix, rows in zip(batched, chunks, strict=True):
                matrix[start:stop] = rows
        sequential = MODULE.sequential_future_summed_rows(
            activations[3], activations[1:3], valid, 0, 3
        )
        for actual, expected in zip(batched, sequential, strict=True):
            metrics = MODULE.vectorization_error_metrics(actual, expected)
            self.assertEqual(metrics["max_abs"], 0.0)
            self.assertEqual(metrics["relative_rms"], 0.0)


@unittest.skipIf(torch is None, "torch is installed in the optional fit environment")
class IntegrityAndPublicationTest(unittest.TestCase):
    def make_state(self, work_dir: Path) -> dict:
        now = MODULE.utc_now()
        return {
            "schema_version": MODULE.STATE_SCHEMA,
            "run_id": "test-run",
            "started_at": now,
            "updated_at": now,
            "status": "running",
            "contract": {"source_identity": {}},
            "contract_sha256": "1" * 64,
            "prompt_count": 1,
            "n_done": 1,
            "next_prompt": 1,
            "sum_generation": 1,
            "sum_integrity": None,
            "current": None,
            "elapsed_seconds": 0.0,
            "total_wall_seconds": 0.0,
            "invocations": [
                {
                    "index": 0,
                    "argv": ["--test"],
                    "started_at": now,
                    "last_checkpoint_at": now,
                    "last_checkpoint_unix": time.time(),
                    "elapsed_seconds": 0.0,
                    "status": "running",
                    "source_observation": {},
                }
            ],
            "start_source_observation": {},
            "model_execution": {
                "quantized_weights": {"aggregate_sha256": "2" * 64}
            },
            "model_execution_sha256": "3" * 64,
            "publication": None,
            "max_cuda_allocated_bytes": 0,
            "max_cuda_reserved_bytes": 0,
        }

    def make_args(self, root: Path, *, resume: bool) -> argparse.Namespace:
        return argparse.Namespace(
            work_dir=root / "work",
            output=root / "published.pt",
            provenance=root / "published.json",
            output_dtype="float32",
            resume=resume,
        )

    def test_quantized_manifest_hashes_every_state_tensor(self) -> None:
        state2 = SimpleNamespace(
            blocksize=256,
            absmax=torch.tensor([0.5], dtype=torch.float32),
            code=torch.arange(256, dtype=torch.float32),
            dtype=torch.float32,
        )
        state = SimpleNamespace(
            blocksize=64,
            nested=True,
            state2=state2,
            absmax=torch.tensor([1, 2], dtype=torch.uint8),
            code=torch.linspace(-1, 1, 16),
            offset=torch.tensor(0.25),
            quant_type="nf4",
            shape=(2, 4),
            dtype=torch.bfloat16,
        )
        weight = SimpleNamespace(
            data=torch.tensor([[1, 2, 3, 4]], dtype=torch.uint8),
            quant_state=state,
            blocksize=64,
        )
        module = SimpleNamespace(weight=weight, in_features=4, out_features=2)
        first = MODULE.nf4_weights_manifest([("linear", module)])
        second = MODULE.nf4_weights_manifest([("linear", module)])
        self.assertEqual(first, second)
        self.assertEqual(first["module_count"], 1)
        self.assertEqual(first["modules"][0]["nested_blocksize"], 256)
        state.state2.blocksize = 128
        with self.assertRaisesRegex(RuntimeError, "block structure"):
            MODULE.nf4_weights_manifest([("linear", module)])

    def test_current_prefix_and_sum_corruption_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(MODULE, "D_MODEL", 3), mock.patch.object(
                MODULE, "SOURCE_LAYERS", (0, 1)
            ):
                matrices = []
                for layer in (0, 1):
                    path = MODULE.layer_path(root, layer)
                    matrix = np.memmap(
                        path, mode="w+", dtype=MODULE.F32_DTYPE, shape=(3, 3)
                    )
                    matrix[:] = layer + np.arange(9).reshape(3, 3)
                    matrix.flush()
                    matrices.append(matrix)
                chunk = {
                    "start": 0,
                    "stop": 2,
                    "sha256": MODULE.current_chunk_sha256(matrices, 0, 2),
                }
                MODULE.validate_current_integrity(
                    matrices, {"next_row": 2, "chunks": [chunk]}
                )
                matrices[1][0, 0] += 1
                matrices[1].flush()
                with self.assertRaisesRegex(RuntimeError, "prefix integrity"):
                    MODULE.validate_current_integrity(
                        matrices, {"next_row": 2, "chunks": [chunk]}
                    )

                records = []
                for layer in (0, 1):
                    path = MODULE.layer_path(root, layer)
                    records.append(
                        {
                            "layer": layer,
                            "size": path.stat().st_size,
                            "sha256": MODULE.sha256_file(path),
                        }
                    )
                state = {
                    "sum_generation": 1,
                    "sum_integrity": {"generation": 1, "layers": records},
                }
                MODULE.validate_sum_integrity(root, state)
                with MODULE.layer_path(root, 0).open("r+b") as handle:
                    handle.seek(0)
                    handle.write(b"xxxx")
                with self.assertRaisesRegex(RuntimeError, "sum hash mismatch"):
                    MODULE.validate_sum_integrity(root, state)

    def test_exclusive_publication_recovers_owned_partial_and_rejects_unrelated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            destination = root / "destination"
            publish_temp = root / ".owned-temp"
            candidate.write_bytes(b"complete artifact")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            publish_temp.write_bytes(b"partial")
            MODULE.publish_candidate_exclusive(
                candidate,
                destination,
                publish_temp,
                digest,
                allow_matching_existing=True,
            )
            self.assertEqual(destination.read_bytes(), b"complete artifact")
            candidate.unlink()
            MODULE.publish_candidate_exclusive(
                candidate,
                destination,
                publish_temp,
                digest,
                allow_matching_existing=True,
            )
            destination.write_bytes(b"unrelated")
            with self.assertRaisesRegex(RuntimeError, "persisted publication hash"):
                MODULE.publish_candidate_exclusive(
                    candidate,
                    destination,
                    publish_temp,
                    digest,
                    allow_matching_existing=True,
                )

    def test_fp16_build_records_authoritative_and_published_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            summed = work / "sum-000001"
            summed.mkdir(parents=True)
            with mock.patch.object(MODULE, "D_MODEL", 2), mock.patch.object(
                MODULE, "SOURCE_LAYERS", (0, 1)
            ), mock.patch.object(MODULE, "TARGET_LAYER", 2):
                for layer in (0, 1):
                    matrix = np.memmap(
                        MODULE.layer_path(summed, layer),
                        mode="w+",
                        dtype=MODULE.F32_DTYPE,
                        shape=(2, 2),
                    )
                    matrix[:] = np.array(
                        [[0.1 + layer, 0.2], [0.3, 0.4]], dtype=np.float32
                    )
                    matrix.flush()
                    del matrix
                state = self.make_state(work)
                output = root / "lens-fp16.pt"
                _digest, statistics = MODULE.build_lens(
                    work, state, output, output_dtype="float16"
                )
                checkpoint = torch.load(output, map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["J"][0].dtype, torch.float16)
                for record in statistics:
                    self.assertEqual(record["dtype"], "float32")
                    self.assertEqual(record["published"]["dtype"], "float16")
                    self.assertNotEqual(record["sha256"], record["published"]["sha256"])

    def test_final_publication_resumes_after_artifact_and_sidecar_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root, resume=False)
            args.work_dir.mkdir()
            state = self.make_state(args.work_dir)
            prompt_metadata = {"prompts": [{"id": "p"}]}

            def fake_build(_work, _state, path, *, output_dtype):
                self.assertEqual(output_dtype, "float32")
                path.write_bytes(b"lens")
                return MODULE.sha256_file(path), [{"layer": 0}]

            def fake_provenance(path, *_args):
                path.write_text('{"status":"completed"}\n', encoding="utf-8")
                return MODULE.sha256_file(path)

            with mock.patch.object(MODULE, "build_lens", side_effect=fake_build), mock.patch.object(
                MODULE, "publish_provenance", side_effect=fake_provenance
            ), mock.patch.object(MODULE, "environment_metadata", return_value={}), mock.patch.object(
                MODULE, "validate_sum_integrity"
            ):
                publication = MODULE.stage_publication(
                    args.work_dir,
                    state,
                    args.output,
                    args.provenance,
                    args.output_dtype,
                    resume=False,
                )
                candidate = Path(publication["artifact_candidate"])
                digest, stats = fake_build(
                    args.work_dir, state, candidate, output_dtype="float32"
                )
                publication.update(
                    artifact_sha256=digest,
                    artifact_size=candidate.stat().st_size,
                    layer_statistics=stats,
                    status="artifact_ready",
                )
                MODULE.save_state(args.work_dir, state)
                MODULE.publish_candidate_exclusive(
                    candidate,
                    args.output,
                    Path(publication["artifact_publish_temp"]),
                    digest,
                    allow_matching_existing=False,
                )

                args.resume = True
                lens_hash, provenance_hash = MODULE.finalize_publication(
                    args, state, prompt_metadata, args.provenance
                )
                self.assertEqual(state["status"], "completed")
                self.assertEqual(MODULE.sha256_file(args.output), lens_hash)
                self.assertEqual(MODULE.sha256_file(args.provenance), provenance_hash)

                # This is the crash window after both files exist but before the
                # final state write. An explicit resume verifies rather than replaces.
                state["status"] = "running"
                state["publication"]["status"] = "provenance_ready"
                state["invocations"][-1]["status"] = "running"
                MODULE.finalize_publication(args, state, prompt_metadata, args.provenance)
                self.assertEqual(state["status"], "completed")


if __name__ == "__main__":
    unittest.main()
