from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import nvfp4_fit_state as STATE


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_nvfp4_ste_lens", ROOT / "scripts" / "export_nvfp4_ste_lens.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from jlens import JacobianLens
except ModuleNotFoundError:
    JacobianLens = None


@unittest.skipIf(torch is None, "torch is required for lens export")
class ExportNVFP4STELensTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.layout = MODULE.FitLayout(
            hidden_size=3,
            source_layers=(0, 2),
            prompt_count=2,
            io_rows=2,
        )

    def _rows(self, value: float, count: int) -> dict[int, np.ndarray]:
        return {
            layer: np.full(
                (count, self.layout.hidden_size),
                value + layer,
                dtype=np.float32,
            )
            for layer in self.layout.source_layers
        }

    def materialize_final_mean(self, name: str = "work") -> Path:
        work = self.root / name
        with STATE.FitStateStore.create(
            work,
            {"model_revision": "test", "estimator": "test"},
            layout=self.layout,
        ) as store:
            for prompt_index, value in enumerate((1.0, 3.0)):
                store.begin_prompt({"id": f"prompt-{prompt_index}"})
                store.write_current_chunk(self._rows(value, 2))
                store.write_current_chunk(self._rows(value, 1))
                store.commit_current_prompt()
            store.finalize_means(
                {"fit_quantization": "nvfp4-forward-surrogate-backward"}
            )
        return work / STATE.FINAL_DIRECTORY

    def test_exports_exact_upstream_checkpoint_from_file_backed_tensors(self) -> None:
        final_mean = self.materialize_final_mean("tampered-work")
        output = self.root / "lens.pt"
        with mock.patch.object(torch, "from_file", wraps=torch.from_file) as mapper:
            result = MODULE.export_lens(
                final_mean,
                output,
                expected_layout=self.layout,
            )

        self.assertEqual(mapper.call_count, 2)
        for call in mapper.call_args_list:
            self.assertFalse(call.kwargs["shared"])
            self.assertEqual(call.kwargs["size"], 9)
            self.assertEqual(call.kwargs["dtype"], torch.float32)
        self.assertEqual(result["sha256"], MODULE._hash_file(output)[0])

        checkpoint = torch.load(
            output, map_location="cpu", weights_only=True, mmap=True
        )
        self.assertEqual(set(checkpoint), MODULE.CHECKPOINT_KEYS)
        self.assertEqual(checkpoint["n_prompts"], 2)
        self.assertEqual(checkpoint["d_model"], 3)
        self.assertEqual(checkpoint["source_layers"], [0, 2])
        for layer in self.layout.source_layers:
            self.assertEqual(checkpoint["J"][layer].dtype, torch.float32)
            np.testing.assert_array_equal(
                checkpoint["J"][layer].numpy(),
                np.full((3, 3), 2.0 + layer, dtype=np.float32),
            )

    @unittest.skipIf(JacobianLens is None, "jacobian-lens is installed in .venv-fit")
    def test_upstream_load_and_from_pretrained_accept_export(self) -> None:
        final_mean = self.materialize_final_mean("clean-work")
        output = self.root / "lens.pt"
        MODULE.export_lens(final_mean, output, expected_layout=self.layout)

        loaded = JacobianLens.load(str(output))
        from_directory = JacobianLens.from_pretrained(str(self.root))
        for lens in (loaded, from_directory):
            self.assertEqual(lens.n_prompts, 2)
            self.assertEqual(lens.d_model, 3)
            self.assertEqual(lens.source_layers, [0, 2])
            np.testing.assert_array_equal(
                lens.jacobians[2].numpy(), np.full((3, 3), 4.0, dtype=np.float32)
            )

    def test_tampered_hash_wrong_layout_and_existing_output_fail_closed(self) -> None:
        final_mean = self.materialize_final_mean("tampered-work")
        layer_zero = MODULE.layer_path(final_mean, 0)
        with layer_zero.open("r+b") as handle:
            handle.write(np.float32(99.0).tobytes())
        with self.assertRaisesRegex(RuntimeError, "hash mismatch at layer 0"):
            MODULE.export_lens(
                final_mean,
                self.root / "tampered.pt",
                expected_layout=self.layout,
            )

        final_mean = self.materialize_final_mean("clean-work")
        wrong_layout = MODULE.FitLayout(
            hidden_size=3,
            source_layers=(0, 2),
            prompt_count=3,
            io_rows=2,
        )
        with self.assertRaisesRegex(RuntimeError, "requested export layout"):
            MODULE.validate_final_mean(final_mean, expected_layout=wrong_layout)

        symlink = self.root / "linked-final-mean"
        symlink.symlink_to(final_mean, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "real directory"):
            MODULE.validate_final_mean(symlink, expected_layout=self.layout)

        output = self.root / "existing.pt"
        output.write_bytes(b"owned")
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            MODULE.export_lens(final_mean, output, expected_layout=self.layout)
        self.assertEqual(output.read_bytes(), b"owned")

    def test_nonfinite_values_fail_even_when_manifest_hash_is_rewritten(self) -> None:
        final_mean = self.materialize_final_mean()
        path = MODULE.layer_path(final_mean, 2)
        with path.open("r+b") as handle:
            handle.write(np.float32(float("nan")).tobytes())
        metadata_path = final_mean / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["layers"][1]["sha256"] = MODULE._hash_file(path)[0]
        metadata["layer_aggregate_sha256"] = MODULE.canonical_sha256(
            metadata["layers"]
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            MODULE.export_lens(
                final_mean,
                self.root / "nonfinite.pt",
                expected_layout=self.layout,
            )

    def test_source_change_during_save_removes_temporary_and_output(self) -> None:
        final_mean = self.materialize_final_mean()
        output = self.root / "raced.pt"
        original_save = torch.save

        def mutate_after_save(value, handle):
            original_save(value, handle)
            path = MODULE.layer_path(final_mean, 0)
            with path.open("r+b") as source:
                source.write(np.float32(123.0).tobytes())

        with mock.patch.object(torch, "save", side_effect=mutate_after_save):
            with self.assertRaisesRegex(RuntimeError, "changed during export"):
                MODULE.export_lens(
                    final_mean,
                    output,
                    expected_layout=self.layout,
                )
        self.assertFalse(os.path.lexists(output))
        self.assertEqual(list(self.root.glob(".raced.pt.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
