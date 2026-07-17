#!/usr/bin/env python3
"""Tests for the on-demand ModelOpt W4 checkpoint reader."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "modelopt_checkpoint", ROOT / "scripts" / "modelopt_checkpoint.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SyntheticModelOptCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.snapshot = Path(self.temp.name)
        self.gate = "model.language_model.layers.0.mlp.gate_proj"
        self.up = "model.language_model.layers.0.mlp.up_proj"
        self.fp8 = "model.language_model.layers.0.linear_attn.out_proj"

        gate_weight = torch.tensor(
            [[0x10] * 8, [0x21] * 8], dtype=torch.uint8
        )
        up_weight = torch.tensor(
            [[0x32] * 8, [0x43] * 8], dtype=torch.uint8
        )
        block_scales = torch.ones(2, 1, dtype=torch.float8_e4m3fn)
        global_scale = torch.tensor(0.5, dtype=torch.float32)
        shard_a = {
            self.gate + ".weight": gate_weight,
            self.up + ".weight": up_weight,
        }
        shard_b = {
            self.gate + ".weight_scale": block_scales,
            self.gate + ".weight_scale_2": global_scale,
            self.up + ".weight_scale": block_scales.clone(),
            self.up + ".weight_scale_2": global_scale.clone(),
        }
        save_file(shard_a, self.snapshot / "model-00001-of-00002.safetensors")
        save_file(shard_b, self.snapshot / "model-00002-of-00002.safetensors")

        weight_map = {
            name: "model-00001-of-00002.safetensors" for name in shard_a
        }
        weight_map.update(
            {name: "model-00002-of-00002.safetensors" for name in shard_b}
        )
        (self.snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
        )
        layers = {
            self.gate: {"quant_algo": "W4A16_NVFP4", "group_size": 16},
            self.up: {"quant_algo": "W4A16_NVFP4", "group_size": 16},
            self.fp8: {"quant_algo": "FP8"},
        }
        (self.snapshot / "hf_quant_config.json").write_text(
            json.dumps(
                {
                    "producer": {"name": "modelopt", "version": "test"},
                    "quantization": {
                        "quant_algo": "MIXED_PRECISION",
                        "quantized_layers": layers,
                    },
                }
            ),
            encoding="utf-8",
        )
        (self.snapshot / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "quantization_config": {
                        "quant_method": "modelopt",
                        "quant_algo": "MIXED_PRECISION",
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def open(self):
        return MODULE.ModelOptCheckpoint(self.snapshot, strict_pinned=False)

    def test_reads_only_indexed_tensors_and_dequantizes(self) -> None:
        weight = self.open().load_nvfp4(self.gate)
        self.assertEqual(weight.components, (self.gate,))
        self.assertEqual(tuple(weight.packed_weight.shape), (2, 8))
        self.assertEqual(tuple(weight.block_scales.shape), (2, 1))
        self.assertEqual(weight.source_shards, (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ))
        expected = torch.tensor(
            [[0.0, 0.25] * 8, [0.25, 0.5] * 8], dtype=torch.float32
        )
        torch.testing.assert_close(weight.dequantize(), expected, rtol=0, atol=0)

    def test_runtime_gate_up_alias_fuses_checkpoint_rows(self) -> None:
        alias = "model.language_model.layers.0.mlp.gate_up_proj"
        metadata = self.open().inspect_nvfp4(alias)
        self.assertEqual(metadata.components, (self.gate, self.up))
        self.assertEqual(metadata.packed_shape, (4, 8))
        self.assertEqual(metadata.block_scale_shape, (4, 1))

        fused = self.open().load_nvfp4(alias)
        self.assertEqual(tuple(fused.packed_weight.shape), (4, 8))
        self.assertTrue(torch.equal(fused.packed_weight[:2], self.open().load_nvfp4(self.gate).packed_weight))
        self.assertTrue(torch.equal(fused.packed_weight[2:], self.open().load_nvfp4(self.up).packed_weight))

    def test_rejects_raw_fp8_runtime_vjp_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "raw FP8 checkpoint weights are forbidden"):
            self.open().inspect_nvfp4(self.fp8)

    def test_pinned_validation_is_not_silently_disabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "snapshot revision"):
            MODULE.ModelOptCheckpoint(self.snapshot)

    def test_strict_validation_hashes_shard_content_not_only_size(self) -> None:
        metadata_hashes = {
            filename: hashlib.sha256((self.snapshot / filename).read_bytes()).hexdigest()
            for filename in (
                "config.json",
                "hf_quant_config.json",
                "model.safetensors.index.json",
            )
        }
        shard_names = (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        )
        shard_pins = {
            filename: (
                (self.snapshot / filename).stat().st_size,
                hashlib.sha256((self.snapshot / filename).read_bytes()).hexdigest(),
            )
            for filename in shard_names
        }
        with (
            mock.patch.object(MODULE, "PINNED_REVISION", self.snapshot.name),
            mock.patch.object(MODULE, "PINNED_METADATA_SHA256", metadata_hashes),
            mock.patch.object(MODULE, "PINNED_SHARDS", shard_pins),
        ):
            checkpoint = MODULE.ModelOptCheckpoint(self.snapshot)
            shard = self.snapshot / shard_names[0]
            payload = bytearray(shard.read_bytes())
            payload[-1] ^= 1
            shard.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                checkpoint.validate_pinned_integrity()
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                MODULE.ModelOptCheckpoint(self.snapshot)

    def test_index_rejects_path_traversal(self) -> None:
        index_path = self.snapshot / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        first = next(iter(index["weight_map"]))
        index["weight_map"][first] = "../outside.safetensors"
        index_path.write_text(json.dumps(index), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsafe checkpoint shard"):
            self.open()


class RealPinnedCheckpointProbeTest(unittest.TestCase):
    def test_layer_62_gate_metadata_matches_pinned_checkpoint(self) -> None:
        snapshot = MODULE.default_pinned_snapshot()
        if not snapshot.is_dir():
            self.skipTest(f"pinned checkpoint is not present at {snapshot}")
        checkpoint = MODULE.ModelOptCheckpoint(snapshot)
        module_name = "model.language_model.layers.62.mlp.gate_proj"
        metadata = checkpoint.inspect_nvfp4(module_name)
        self.assertEqual(metadata.packed_shape, (17_408, 2_560))
        self.assertEqual(metadata.block_scale_shape, (17_408, 320))
        self.assertEqual(metadata.block_size, 16)
        self.assertEqual(
            {tensor.shard for tensor in metadata.tensors},
            {"model-00003-of-00003.safetensors"},
        )


if __name__ == "__main__":
    unittest.main()
