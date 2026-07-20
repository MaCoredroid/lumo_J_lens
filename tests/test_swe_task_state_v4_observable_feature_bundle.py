from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_observable_feature_bundle.py"
spec = importlib.util.spec_from_file_location(
    "swe_task_state_v4_observable_feature_bundle", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _alignment_rows() -> list[dict[str, object]]:
    task = "f" * 64
    return [
        {
            "global_index": index,
            "source_id_sha256": f"{index + 1:064x}",
            "task_id_sha256": task,
            "repository": "owner/repo",
            "request_index": index + 1,
            "stable_feature_eligible": index < 1606,
        }
        for index in range(1708)
    ]


def _array_manifest(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, values in arrays.items():
        result[name] = {
            "shape": list(values.shape),
            "dtype": (
                "little-endian-int64"
                if name == "global_index"
                else "little-endian-float64"
            ),
            "logical_sha256": module.logical_array_sha256(name, values),
        }
    return result


def _source_output(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": module.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "keys": list(arrays),
        "numeric_tensor_only": True,
        "reload_verified": True,
        "arrays": _array_manifest(arrays),
    }


def _fake_upstream(path: str, digest: str) -> dict[str, object]:
    return {"path": path, "sha256": digest, "size_bytes": 1}


def _actual_record(path: str) -> dict[str, object]:
    return module._artifact_record(ROOT / path)


def make_fixture(
    root: Path,
    *,
    activation_global_index: np.ndarray | None = None,
    activation_extra_key: bool = False,
    activation_nonfinite: bool = False,
) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    alignment_path = root / "alignment-index.json"
    alignment = {
        "schema_version": 1,
        "kind": "swe_task_state_v4_label_free_alignment_index",
        "status": "passed",
        "scope": "grouping_order_and_stability_only_no_labels",
        "config": {},
        "implementation": {},
        "sources": [],
        "eligibility_source": {},
        "row_count": 1708,
        "stable_row_count": 1606,
        "feature_use": {},
        "rows": _alignment_rows(),
    }
    _write_json(alignment_path, alignment)
    alignment_sha256 = module.sha256_file(alignment_path)
    alignment_record = module._artifact_record(alignment_path)

    global_index = np.arange(1606, dtype="<i8")
    visible_arrays: dict[str, np.ndarray] = {"global_index": global_index}
    for position, name in enumerate(module.VISIBLE_KEYS[1:], start=1):
        visible_arrays[name] = np.full(
            (1606, module.DECODER.BASE_BLOCK_WIDTHS[name]),
            float(position),
            dtype="<f8",
        )
    visible_data_path = root / "visible.npz"
    np.savez_compressed(visible_data_path, **visible_arrays)
    visible_manifest = {
        "schema_version": 1,
        "kind": module.VISIBLE_KIND,
        "status": "passed",
        "status_scope": "authenticated_visible_precompletion_numeric_baselines_only",
        "pre_and_post_input_bindings_equal": True,
        "config": _actual_record("configs/swe_task_state_v4_visible_baselines.json"),
        "implementation": _actual_record("scripts/swe_task_state_v4_visible_baselines.py"),
        "inputs": {
            "development_prompts": _fake_upstream(".cache/fake-prompts.json", "1" * 64),
            "development_public_report": _fake_upstream(".cache/fake-public.json", "2" * 64),
            "label_free_alignment_index": alignment_record,
            "v3_action_protocol": _fake_upstream("configs/fake-action.json", "3" * 64),
            "v3_protocol": _fake_upstream("configs/fake-protocol.json", "4" * 64),
        },
        "local_code_dependencies": [],
        "coverage": {
            "all_boundary_count": 1708,
            "stable_row_count": 1606,
            "numerically_unstable_row_count": 102,
            "stable_source_identity_order_exact": True,
        },
        "output": _source_output(visible_data_path, visible_arrays),
        "variants": {
            name: {
                "width": module.DECODER.BASE_BLOCK_WIDTHS[name],
                "definition": name,
            }
            for name in module.VISIBLE_KEYS[1:]
        },
        "feature_boundary": {
            "label_sidecar_accepted": False,
            "semantic_ids_as_features_forbidden": True,
            "repository_as_feature_forbidden": True,
        },
        "claim_scope": {"private_chain_of_thought_reconstructed": False},
        "forbidden_path_guard_passed": True,
        "reserved_validation_access_authorized": False,
    }
    visible_manifest_path = root / "visible.json"
    _write_json(visible_manifest_path, visible_manifest)

    activation_arrays: dict[str, np.ndarray] = {
        "global_index": (
            global_index.copy()
            if activation_global_index is None
            else np.asarray(activation_global_index, dtype="<i8")
        )
    }
    for position, name in enumerate(module.ACTIVATION_KEYS[1:], start=11):
        activation_arrays[name] = np.full(
            (1606, module.DECODER.BASE_BLOCK_WIDTHS[name]),
            float(position),
            dtype="<f8",
        )
    if activation_nonfinite:
        activation_arrays["raw_activation_current"][0, 0] = np.inf
    if activation_extra_key:
        activation_arrays["unexpected"] = np.zeros((1606, 1), dtype="<f8")
    activation_data_path = root / "activation.npz"
    np.savez_compressed(activation_data_path, **activation_arrays)
    activation_manifest = {
        "schema_version": 1,
        "kind": module.ACTIVATION_KIND,
        "status": "passed",
        "status_scope": "label_free_primary_seed_current_and_causal_sequence_tensors_only",
        "feature_config": _actual_record("configs/swe_task_state_v4_activation_features.json"),
        "implementation": _actual_record("scripts/swe_task_state_v4_activation_feature_campaign.py"),
        "feature_implementation": _actual_record("scripts/swe_task_state_v4_activation_features.py"),
        "projection_config": _actual_record("configs/swe_task_state_v4_activation_projection.json"),
        "projection_implementation": _actual_record("scripts/swe_task_state_v4_activation_projection.py"),
        "pre_and_post_input_bindings_equal": True,
        "inputs": {
            "alignment_index": {
                **alignment_record,
                "row_count": 1708,
                "stable_row_count": 1606,
            },
            "projection_chunks": [{}, {}, {}, {}],
        },
        "coverage": {
            "boundary_count": 1708,
            "stable_feature_count": 1606,
            "source_id_order_matches_alignment": True,
            "source_id_coverage_exact": True,
            "chunk_order_exact": True,
        },
        "output": _source_output(activation_data_path, activation_arrays),
        "projection": {},
        "temporal": {},
        "variants": {
            name: module.DECODER.BASE_BLOCK_WIDTHS[name]
            for name in module.ACTIVATION_KEYS[1:]
        },
        "feature_boundary": {
            "labels_or_outcomes_accepted": False,
            "semantic_ids_as_features_forbidden": True,
            "repository_as_feature_forbidden": True,
        },
        "claim_scope": {"private_chain_of_thought_reconstructed": False},
        "forbidden_path_guard_passed": True,
        "reserved_validation_access_authorized": False,
    }
    activation_manifest_path = root / "activation.json"
    _write_json(activation_manifest_path, activation_manifest)
    return {
        "alignment": alignment_path,
        "alignment_sha256": alignment_sha256,
        "visible_manifest": visible_manifest_path,
        "visible_manifest_sha256": module.sha256_file(visible_manifest_path),
        "visible_data": visible_data_path,
        "visible_arrays": visible_arrays,
        "activation_manifest": activation_manifest_path,
        "activation_manifest_sha256": module.sha256_file(activation_manifest_path),
        "activation_data": activation_data_path,
        "activation_arrays": activation_arrays,
    }


def build(fixture: dict[str, object], output_root: Path) -> dict[str, object]:
    config = module.DECODER.load_json(module.CONFIG_PATH)
    return module.build_feature_bundle_from_authenticated_sources(
        alignment_index_path=Path(fixture["alignment"]),
        expected_alignment_index_sha256=str(fixture["alignment_sha256"]),
        visible_manifest_path=Path(fixture["visible_manifest"]),
        expected_visible_manifest_sha256=str(fixture["visible_manifest_sha256"]),
        activation_manifest_path=Path(fixture["activation_manifest"]),
        expected_activation_manifest_sha256=str(fixture["activation_manifest_sha256"]),
        output_data_path=output_root / "bundle.npz",
        output_manifest_path=output_root / "bundle.json",
        config=config,
    )


class ObservableFeatureBundleTests(unittest.TestCase):
    def test_public_builder_has_no_caller_array_or_label_seam(self) -> None:
        parameters = inspect.signature(
            module.build_feature_bundle_from_authenticated_sources
        ).parameters
        self.assertEqual(
            list(parameters),
            [
                "alignment_index_path",
                "expected_alignment_index_sha256",
                "visible_manifest_path",
                "expected_visible_manifest_sha256",
                "activation_manifest_path",
                "expected_activation_manifest_sha256",
                "output_data_path",
                "output_manifest_path",
                "config",
            ],
        )
        for forbidden in (
            "arrays",
            "features",
            "labels",
            "targets",
            "outcomes",
            "completion_text",
        ):
            self.assertNotIn(forbidden, parameters)

    def test_authenticated_sources_build_exact_schema_v2_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = make_fixture(root / "inputs")
            output = root / "output"
            manifest = build(fixture, output)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["kind"], module.KIND)
            self.assertEqual(
                [source["role"] for source in manifest["sources"]],
                [module.VISIBLE_ROLE, module.ACTIVATION_ROLE],
            )
            self.assertFalse(
                manifest["construction"]["caller_supplied_feature_arrays_accepted"]
            )
            self.assertTrue(
                manifest["construction"]["source_arrays_equal_bundle_arrays"]
            )
            with np.load(output / "bundle.npz", allow_pickle=False) as archive:
                self.assertEqual(archive.files, sorted(archive.files))
                np.testing.assert_array_equal(
                    archive["history_only"], fixture["visible_arrays"]["history_only"]
                )
                np.testing.assert_array_equal(
                    archive["raw_activation_current"],
                    fixture["activation_arrays"]["raw_activation_current"],
                )
                self.assertNotIn("label", archive.files)
                self.assertNotIn("target", archive.files)
                self.assertNotIn("outcome", archive.files)

    def test_source_npz_tamper_is_rejected_before_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = make_fixture(root / "inputs")
            with Path(fixture["visible_data"]).open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(module.BundleBuilderError, "data size changed"):
                build(fixture, root / "output")

    def test_source_logical_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = make_fixture(root / "inputs")
            path = Path(fixture["visible_manifest"])
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["output"]["arrays"]["history_only"]["logical_sha256"] = "0" * 64
            _write_json(path, manifest)
            fixture["visible_manifest_sha256"] = module.sha256_file(path)
            with self.assertRaisesRegex(module.BundleBuilderError, "logical array hash changed"):
                build(fixture, root / "output")

    def test_cross_source_global_index_reordering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = make_fixture(
                root / "inputs",
                activation_global_index=np.arange(1605, -1, -1, dtype="<i8"),
            )
            with self.assertRaisesRegex(module.BundleBuilderError, "stable alignment"):
                build(fixture, root / "output")

    def test_source_extra_array_and_nonfinite_array_are_rejected(self) -> None:
        for option, message in (
            ({"activation_extra_key": True}, "output contract changed"),
            ({"activation_nonfinite": True}, "finiteness changed"),
        ):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = make_fixture(root / "inputs", **option)
                with self.assertRaisesRegex(module.BundleBuilderError, message):
                    build(fixture, root / "output")

    def test_source_internal_alignment_binding_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = make_fixture(root / "inputs")
            path = Path(fixture["activation_manifest"])
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["inputs"]["alignment_index"]["sha256"] = "0" * 64
            _write_json(path, manifest)
            fixture["activation_manifest_sha256"] = module.sha256_file(path)
            with self.assertRaisesRegex(module.BundleBuilderError, "alignment binding identity"):
                build(fixture, root / "output")

    def test_source_output_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = make_fixture(root / "inputs")
            escaped = root / "escaped-visible.npz"
            escaped.write_bytes(Path(fixture["visible_data"]).read_bytes())
            path = Path(fixture["visible_manifest"])
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["output"]["path"] = "../escaped-visible.npz"
            manifest["output"]["sha256"] = module.sha256_file(escaped)
            manifest["output"]["size_bytes"] = escaped.stat().st_size
            _write_json(path, manifest)
            fixture["visible_manifest_sha256"] = module.sha256_file(path)
            with self.assertRaisesRegex(module.BundleBuilderError, "escapes its manifest"):
                build(fixture, root / "output")

    def test_manifest_hash_and_duplicate_json_key_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = make_fixture(root / "inputs")
            fixture["visible_manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(module.BundleBuilderError, "manifest hash changed"):
                build(fixture, root / "output-a")

            fixture = make_fixture(root / "inputs-b")
            path = Path(fixture["visible_manifest"])
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace('"status": "passed",', '"status": "passed",\n  "status": "passed",', 1), encoding="utf-8")
            fixture["visible_manifest_sha256"] = module.sha256_file(path)
            with self.assertRaisesRegex(module.BundleBuilderError, "duplicate JSON key"):
                build(fixture, root / "output-b")

    def test_no_clobber_and_atomic_temporary_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = make_fixture(root / "inputs")
            output = root / "output"
            build(fixture, output)
            with self.assertRaisesRegex(module.BundleBuilderError, "overwrite"):
                build(fixture, output)
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["bundle.json", "bundle.npz"],
            )


if __name__ == "__main__":
    unittest.main()
