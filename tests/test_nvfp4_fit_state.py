from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "nvfp4_fit_state", ROOT / "scripts" / "nvfp4_fit_state.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FitStateStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.work = self.root / "work"
        self.layout = MODULE.FitLayout(
            hidden_size=3,
            source_layers=(0, 2),
            prompt_count=2,
            io_rows=2,
        )
        self.contract = {
            "model_revision": "test-revision",
            "estimator": "future-summed-vjp",
        }

    def layer_rows(self, value: float, rows: int) -> dict[int, np.ndarray]:
        return {
            layer: np.full(
                (rows, self.layout.hidden_size),
                value + layer,
                dtype=np.float32,
            )
            for layer in self.layout.source_layers
        }

    def fill_prompt(
        self,
        store: MODULE.FitStateStore,
        prompt_index: int,
        value: float,
    ) -> dict:
        store.begin_prompt({"id": f"prompt-{prompt_index}", "index": prompt_index})
        store.write_current_chunk(self.layer_rows(value, 2))
        store.write_current_chunk(self.layer_rows(value, 1))
        return store.commit_current_prompt()

    def test_production_layout_is_63_dense_5120_matrices_and_ten_prompts(self) -> None:
        layout = MODULE.PRODUCTION_LAYOUT
        self.assertEqual(layout.hidden_size, 5120)
        self.assertEqual(layout.source_layers, tuple(range(63)))
        self.assertEqual(layout.prompt_count, 10)
        self.assertEqual(layout.matrix_bytes, 5120 * 5120 * 4)
        self.assertEqual(layout.matrix_set_bytes, 63 * 5120 * 5120 * 4)
        self.assertEqual(layout.record()["matrix_dtype"], "little-endian-float32")

    def test_create_resume_contract_and_exclusive_lock(self) -> None:
        store = MODULE.FitStateStore.create(
            self.work, self.contract, layout=self.layout
        )
        self.addCleanup(store.close)
        with self.assertRaisesRegex(RuntimeError, "already locked"):
            MODULE.FitStateStore.resume(
                self.work, self.contract, layout=self.layout
            )
        store.close()

        stale_state_temporary = self.work / ".state.json.tmp.123.crashed"
        stale_state_temporary.write_text("partial")
        with MODULE.FitStateStore.resume(
            self.work, self.contract, layout=self.layout
        ) as resumed:
            self.assertEqual(resumed.state["contract"], self.contract)
            self.assertFalse(stale_state_temporary.exists())
        with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
            MODULE.FitStateStore.resume(
                self.work, {"different": True}, layout=self.layout
            )
        other_layout = MODULE.FitLayout(
            hidden_size=4,
            source_layers=(0, 2),
            prompt_count=2,
            io_rows=2,
        )
        with self.assertRaisesRegex(RuntimeError, "layout mismatch"):
            MODULE.FitStateStore.resume(
                self.work, self.contract, layout=other_layout
            )

    def test_unrecorded_rows_are_overwritten_after_resume(self) -> None:
        with MODULE.FitStateStore.create(
            self.work, self.contract, layout=self.layout
        ) as store:
            store.begin_prompt({"id": "prompt-0"})
            chunk = store.write_current_chunk(self.layer_rows(1.0, 2))
            self.assertEqual((chunk["start"], chunk["stop"]), (0, 2))

        path = MODULE.layer_path(self.work / MODULE.CURRENT_DIRECTORY, 0)
        uncommitted = MODULE.open_matrix(path, self.layout, mode="r+")
        uncommitted[2] = 999.0
        uncommitted.flush()
        del uncommitted

        with MODULE.FitStateStore.resume(
            self.work, self.contract, layout=self.layout
        ) as resumed:
            self.assertEqual(resumed.state["current"]["next_row"], 2)
            resumed.write_current_chunk(self.layer_rows(1.0, 1))
            resumed.commit_current_prompt()
            summed = MODULE.open_matrix(
                MODULE.layer_path(MODULE.sum_directory(self.work, 1), 0),
                self.layout,
                mode="r",
            )
            np.testing.assert_array_equal(summed, np.ones((3, 3), dtype=np.float32))
            del summed

    def test_discard_current_prompt_removes_rows_and_allows_restart(self) -> None:
        with MODULE.FitStateStore.create(
            self.work, self.contract, layout=self.layout
        ) as store:
            store.begin_prompt({"id": "prompt-0"})
            store.write_current_chunk(self.layer_rows(1.0, 2))
            store.discard_current_prompt()
            self.assertIsNone(store.state["current"])
            self.assertFalse((self.work / MODULE.CURRENT_DIRECTORY).exists())
            store.begin_prompt({"id": "prompt-0"})
            self.assertEqual(store.state["current"]["next_row"], 0)

    def test_chunk_writes_hash_each_row_once_then_commit_revalidates(self) -> None:
        with MODULE.FitStateStore.create(
            self.work, self.contract, layout=self.layout
        ) as store:
            store.begin_prompt({"id": "prompt-0"})
            with mock.patch.object(
                MODULE,
                "current_chunk_sha256",
                wraps=MODULE.current_chunk_sha256,
            ) as hasher:
                store.write_current_chunk(self.layer_rows(1.0, 2))
                store.write_current_chunk(self.layer_rows(1.0, 1))
                self.assertEqual(hasher.call_count, 2)
                store.commit_current_prompt()
                self.assertEqual(hasher.call_count, 4)

    def test_recorded_row_corruption_is_rejected_on_resume(self) -> None:
        with MODULE.FitStateStore.create(
            self.work, self.contract, layout=self.layout
        ) as store:
            store.begin_prompt({"id": "prompt-0"})
            store.write_current_chunk(self.layer_rows(1.0, 1))

        path = MODULE.layer_path(self.work / MODULE.CURRENT_DIRECTORY, 2)
        matrix = MODULE.open_matrix(path, self.layout, mode="r+")
        matrix[0, 0] += 1.0
        matrix.flush()
        del matrix
        with self.assertRaisesRegex(RuntimeError, "prefix integrity"):
            MODULE.FitStateStore.resume(
                self.work, self.contract, layout=self.layout
            )

    def test_commit_rejects_incomplete_prompt(self) -> None:
        with MODULE.FitStateStore.create(
            self.work, self.contract, layout=self.layout
        ) as store:
            store.begin_prompt({"id": "prompt-0"})
            store.write_current_chunk(self.layer_rows(1.0, 2))
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                store.commit_current_prompt()

    def test_commit_recovers_crash_before_state_advance(self) -> None:
        one_prompt = MODULE.FitLayout(
            hidden_size=2,
            source_layers=(0,),
            prompt_count=1,
            io_rows=1,
        )
        with MODULE.FitStateStore.create(
            self.work, self.contract, layout=one_prompt
        ) as store:
            store.begin_prompt({"id": "prompt-0"})
            store.write_current_chunk([np.full((2, 2), 3.0, dtype=np.float32)])
            with mock.patch.object(
                store, "_write_state", side_effect=OSError("simulated crash")
            ):
                with self.assertRaisesRegex(OSError, "simulated crash"):
                    store.commit_current_prompt()
            self.assertEqual(store.state["n_done"], 0)
            self.assertTrue(MODULE.sum_directory(self.work, 1).is_dir())

        with MODULE.FitStateStore.resume(
            self.work, self.contract, layout=one_prompt
        ) as resumed:
            self.assertFalse(MODULE.sum_directory(self.work, 1).exists())
            self.assertEqual(resumed.state["current"]["next_row"], 2)
            resumed.commit_current_prompt()
            self.assertEqual(resumed.state["n_done"], 1)

    def test_commit_recovers_crash_after_state_advance(self) -> None:
        one_prompt = MODULE.FitLayout(
            hidden_size=2,
            source_layers=(0,),
            prompt_count=1,
            io_rows=1,
        )
        store = MODULE.FitStateStore.create(
            self.work, self.contract, layout=one_prompt
        )
        store.begin_prompt({"id": "prompt-0"})
        store.write_current_chunk([np.full((2, 2), 4.0, dtype=np.float32)])
        original_rmtree = MODULE.shutil.rmtree

        def fail_current_cleanup(path, *args, **kwargs):
            if Path(path).name == MODULE.CURRENT_DIRECTORY:
                raise OSError("simulated cleanup crash")
            return original_rmtree(path, *args, **kwargs)

        with mock.patch.object(MODULE.shutil, "rmtree", side_effect=fail_current_cleanup):
            with self.assertRaisesRegex(OSError, "cleanup crash"):
                store.commit_current_prompt()
        self.assertEqual(store.state["n_done"], 1)
        store.close()

        with MODULE.FitStateStore.resume(
            self.work, self.contract, layout=one_prompt
        ) as resumed:
            self.assertEqual(resumed.state["n_done"], 1)
            self.assertFalse((self.work / MODULE.CURRENT_DIRECTORY).exists())
            MODULE.validate_sum_integrity(
                MODULE.sum_directory(self.work, 1), resumed.state, one_prompt
            )

    def test_cumulative_sum_and_final_mean_metadata_are_resumable(self) -> None:
        with MODULE.FitStateStore.create(
            self.work, self.contract, layout=self.layout
        ) as store:
            first = self.fill_prompt(store, 0, 1.0)
            second = self.fill_prompt(store, 1, 3.0)
            self.assertEqual(first["sum_generation"], 1)
            self.assertEqual(second["sum_generation"], 2)
            self.assertFalse(MODULE.sum_directory(self.work, 1).exists())

            metadata = store.finalize_means(
                {"fit_quantization": "nvfp4-forward-surrogate-backward"}
            )
            self.assertEqual(metadata["n_prompts"], 2)
            self.assertEqual(metadata["layout"], self.layout.record())
            self.assertEqual(store.state["status"], "completed")
            for layer in self.layout.source_layers:
                path = MODULE.layer_path(self.work / MODULE.FINAL_DIRECTORY, layer)
                mean = MODULE.open_matrix(path, self.layout, mode="r")
                np.testing.assert_array_equal(
                    mean,
                    np.full((3, 3), 2.0 + layer, dtype=np.float32),
                )
                del mean
            first_value = MODULE.layer_path(
                self.work / MODULE.FINAL_DIRECTORY, 0
            ).read_bytes()[:4]
            self.assertEqual(struct.unpack("<f", first_value)[0], 2.0)
            repeated = store.finalize_means(metadata["metadata"])
            self.assertEqual(repeated, metadata)

        with MODULE.FitStateStore.resume(
            self.work, self.contract, layout=self.layout
        ) as resumed:
            verified = resumed.validate_final_artifact()
            self.assertEqual(
                verified["layer_aggregate_sha256"],
                metadata["layer_aggregate_sha256"],
            )

    def test_committed_sum_corruption_is_rejected_on_resume(self) -> None:
        with MODULE.FitStateStore.create(
            self.work, self.contract, layout=self.layout
        ) as store:
            self.fill_prompt(store, 0, 2.0)
        path = MODULE.layer_path(MODULE.sum_directory(self.work, 1), 2)
        with path.open("r+b") as handle:
            handle.seek(0)
            handle.write(b"xxxx")
        with self.assertRaisesRegex(RuntimeError, "sum hash mismatch"):
            MODULE.FitStateStore.resume(
                self.work, self.contract, layout=self.layout
            )

    def test_finalization_recovers_renamed_artifact_before_state_advance(self) -> None:
        one_prompt = MODULE.FitLayout(
            hidden_size=2,
            source_layers=(0,),
            prompt_count=1,
            io_rows=1,
        )
        with MODULE.FitStateStore.create(
            self.work, self.contract, layout=one_prompt
        ) as store:
            store.begin_prompt({"id": "prompt-0"})
            store.write_current_chunk([np.full((2, 2), 6.0, dtype=np.float32)])
            store.commit_current_prompt()
            with mock.patch.object(
                store, "_write_state", side_effect=OSError("simulated crash")
            ):
                with self.assertRaisesRegex(OSError, "simulated crash"):
                    store.finalize_means({"kind": "test"})
            self.assertIsNone(store.state["final_artifact"])
            self.assertTrue((self.work / MODULE.FINAL_DIRECTORY).is_dir())

        with MODULE.FitStateStore.resume(
            self.work, self.contract, layout=one_prompt
        ) as resumed:
            self.assertEqual(resumed.state["status"], "completed")
            self.assertEqual(resumed.validate_final_artifact()["metadata"], {"kind": "test"})

    def test_final_matrix_corruption_is_rejected(self) -> None:
        one_prompt = MODULE.FitLayout(
            hidden_size=2,
            source_layers=(0,),
            prompt_count=1,
            io_rows=1,
        )
        with MODULE.FitStateStore.create(
            self.work, self.contract, layout=one_prompt
        ) as store:
            store.begin_prompt({"id": "prompt-0"})
            store.write_current_chunk([np.ones((2, 2), dtype=np.float32)])
            store.commit_current_prompt()
            store.finalize_means({"kind": "test"})
        path = MODULE.layer_path(self.work / MODULE.FINAL_DIRECTORY, 0)
        with path.open("r+b") as handle:
            handle.seek(0)
            handle.write(b"xxxx")
        with self.assertRaisesRegex(RuntimeError, "artifact hash mismatch"):
            MODULE.FitStateStore.resume(
                self.work, self.contract, layout=one_prompt
            )

    def test_atomic_json_failure_preserves_previous_state(self) -> None:
        path = self.root / "atomic.json"
        MODULE.atomic_write_json(path, {"generation": 1})
        with mock.patch.object(MODULE.os, "replace", side_effect=OSError("crash")):
            with self.assertRaisesRegex(OSError, "crash"):
                MODULE.atomic_write_json(path, {"generation": 2})
        self.assertEqual(json.loads(path.read_text()), {"generation": 1})
        self.assertEqual(list(self.root.glob(".atomic.json.tmp.*")), [])

    def test_state_module_has_no_pickle_loader(self) -> None:
        source = (ROOT / "scripts" / "nvfp4_fit_state.py").read_text()
        self.assertNotIn("import pickle", source)
        self.assertNotIn("pickle.load", source)
        self.assertNotIn("torch.load", source)


if __name__ == "__main__":
    unittest.main()
