from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_activation_feature_campaign.py"
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_activation_features.json"
PROJECTION_CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_activation_projection.json"
)

spec = importlib.util.spec_from_file_location(
    "swe_task_state_v4_activation_feature_campaign", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class ActivationFeatureCampaignTests(unittest.TestCase):
    def feature_config(self):
        return module._features.validate_config(
            module._projection.load_json_strict(CONFIG_PATH)
        )

    def projection_config(self):
        return module._projection.validate_config(
            module._projection.load_json_strict(PROJECTION_CONFIG_PATH)
        )

    def mappings(self, config):
        return [
            module._projection.countsketch_mapping(
                seed=seed,
                hidden_size=config["input"]["hidden_size"],
                width=config["projection"]["width"],
            )[2]
            for seed in config["projection"]["seeds"]
        ]

    def write_projection_bundle(self, root: Path, *, count: int = 2):
        config = self.projection_config()
        source_ids = np.asarray([f"{index + 1:064x}" for index in range(count)], dtype="<U64")
        token_ids = np.asarray([f"{index + 101:064x}" for index in range(count)], dtype="<U64")
        arrays = {
            "sketches": np.zeros((count, 4, 2, 3, 256), dtype="<f4"),
            "band_statistics": np.zeros((count, 2, 3, 3), dtype="<f4"),
            "source_id_sha256": source_ids,
            "token_ids_sha256": token_ids,
            "boundary_index": np.arange(count, dtype="<i8"),
        }
        data_path = root / "projection.npz"
        np.savez_compressed(data_path, **arrays)
        manifest = {
            "schema_version": 1,
            "kind": "swe_task_state_v4_label_independent_activation_projection",
            "status": "passed",
            "projection_config": {
                "path": str(PROJECTION_CONFIG_PATH.relative_to(ROOT)),
                "sha256": module.sha256_file(PROJECTION_CONFIG_PATH),
                "size_bytes": PROJECTION_CONFIG_PATH.stat().st_size,
            },
            "capture_config": {
                "path": "configs/swe_task_state_v4_raw_capture.json",
                "sha256": config["input"]["capture_config_sha256"],
                "size_bytes": 1,
            },
            "implementation": {
                "path": str(module.PROJECTION_MODULE_PATH.relative_to(ROOT)),
                "sha256": module.sha256_file(module.PROJECTION_MODULE_PATH),
                "size_bytes": module.PROJECTION_MODULE_PATH.stat().st_size,
            },
            "pre_and_post_projection_bindings_equal": True,
            "input_manifest": {"path": "capture/manifest.json", "sha256": "a" * 64},
            "mapping_sha256s": self.mappings(config),
            "boundary_count": count,
            "sketch_shape": [count, 4, 2, 3, 256],
            "sketches_logical_sha256": module._projection._hash_float_array(
                arrays["sketches"], name="sketches"
            ),
            "band_statistics_logical_sha256": module._projection._hash_float_array(
                arrays["band_statistics"], name="band_statistics"
            ),
            "data": {
                "path": data_path.name,
                "sha256": module.sha256_file(data_path),
                "size_bytes": data_path.stat().st_size,
                "keys": sorted(arrays),
                "reload_verified": True,
            },
            "sources": config["input"]["sources"],
            "bands": config["pooling"]["bands"],
            "projection": config["projection"],
            "band_statistics": config["band_statistics"],
            "downstream_feature_boundary": config["downstream_feature_boundary"],
            "claim_scope": config["claim_scope"],
            "forbidden_path_guard_passed": True,
            "reserved_validation_access_authorized": False,
        }
        manifest_path = root / "projection.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        return manifest_path, arrays

    def test_config_freezes_four_chunks_tensor_allowlist_and_false_claims(self):
        config = self.feature_config()
        campaign = config["campaign"]
        self.assertEqual(campaign["projection_chunk_boundary_counts"], [517, 370, 414, 407])
        self.assertEqual(campaign["total_boundary_count"], 1708)
        self.assertEqual(campaign["stable_feature_count"], 1606)
        self.assertEqual(
            set(campaign["output_tensor_keys"]),
            {
                "global_index",
                "raw_activation_current",
                "public_j_activation_current",
                "raw_activation_sequence",
                "public_j_activation_sequence",
                "raw_public_j_activation_sequence",
            },
        )
        self.assertTrue(all(value is False for value in config["claim_scope"].values()))
        changed = copy.deepcopy(config)
        changed["campaign"]["projection_chunk_boundary_counts"][0] -= 1
        with self.assertRaises(module._features.FeatureError):
            module._features.validate_config(changed)

    def test_projection_bundle_verifies_manifest_data_and_tensor_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, expected = self.write_projection_bundle(root)
            projection_config = self.projection_config()
            arrays, manifest, data_path = module.load_projection_bundle(
                manifest_path,
                expected_manifest_sha256=module.sha256_file(manifest_path),
                expected_count=2,
                projection_config=projection_config,
                expected_mapping_sha256s=self.mappings(projection_config),
            )
            self.assertEqual(data_path, (root / "projection.npz").resolve())
            self.assertEqual(manifest["boundary_count"], 2)
            for name in expected:
                np.testing.assert_array_equal(arrays[name], expected[name])
            changed = dict(arrays)
            changed["boundary_index"] = np.asarray([1, 0], dtype="<i8")
            with self.assertRaises(module.CampaignError):
                module._validate_projection_arrays(
                    changed, expected_count=2, manifest=manifest
                )

    def test_alignment_metadata_is_pinned_not_merely_well_formed(self):
        campaign = self.feature_config()["campaign"]
        required = campaign["required_alignment"]
        index = {
            "schema_version": required["schema_version"],
            "kind": required["kind"],
            "scope": required["scope"],
            "feature_use": {
                "allowed": [
                    "task-local ordering for causal temporal transforms",
                    "repository and task grouping for held-out splits and weights",
                    "stable eligibility filtering",
                ],
                "forbidden": [
                    "hashing or one-hot encoding IDs as model features",
                    "repository or request index as semantic model features",
                ],
            },
            "sources": [
                {"path": f"source-{position}", "sha256": digest, "row_count": count}
                for position, (digest, count) in enumerate(
                    zip(
                        required["source_sha256s"],
                        campaign["projection_chunk_boundary_counts"],
                        strict=True,
                    )
                )
            ],
            "config": {"path": "config", "sha256": required["config_sha256"]},
            "implementation": {
                "path": "implementation",
                "sha256": required["implementation_sha256"],
            },
            "eligibility_source": {
                "path": "eligibility",
                "sha256": required["eligibility_source_sha256"],
                "all_rows": 1708,
                "stable_rows": 1606,
                "numerically_unstable_rows": 102,
            },
        }
        module._validate_alignment_metadata(index, campaign=campaign)
        changed = copy.deepcopy(index)
        changed["eligibility_source"]["sha256"] = "f" * 64
        with self.assertRaises(module.CampaignError):
            module._validate_alignment_metadata(changed, campaign=campaign)
            with self.assertRaises(module.CampaignError):
                module.load_projection_bundle(
                    manifest_path,
                    expected_manifest_sha256="0" * 64,
                    expected_count=2,
                    projection_config=projection_config,
                    expected_mapping_sha256s=self.mappings(projection_config),
                )

    def test_coverage_requires_exact_identity_order_and_uniqueness(self):
        identities = ["1" * 64, "2" * 64, "3" * 64]
        rows = [{"source_id_sha256": value} for value in identities]
        projections = {"source_id_sha256": np.asarray(identities, dtype="<U64")}
        observed = module.validate_exact_coverage(
            projections=projections, alignment_rows=rows
        )
        self.assertTrue(observed["source_id_order_matches_alignment"])
        changed = {"source_id_sha256": projections["source_id_sha256"][[1, 0, 2]]}
        with self.assertRaises(module.CampaignError):
            module.validate_exact_coverage(projections=changed, alignment_rows=rows)
        duplicated = {
            "source_id_sha256": np.asarray([identities[0]] * 3, dtype="<U64")
        }
        with self.assertRaises(module.CampaignError):
            module.validate_exact_coverage(projections=duplicated, alignment_rows=rows)

    def test_builder_api_has_no_label_target_or_outcome_argument(self):
        annotations = module.build_output_tensors.__annotations__
        for forbidden in ("labels", "targets", "outcomes", "metadata"):
            self.assertNotIn(forbidden, annotations)

    def test_output_is_exact_numeric_tensor_allowlist(self):
        config = self.feature_config()
        projections = {
            "sketches": np.zeros((1708, 4, 2, 3, 256), dtype="<f4"),
            "band_statistics": np.zeros((1708, 2, 3, 3), dtype="<f4"),
            "source_id_sha256": np.asarray(
                [f"{index + 1:064x}" for index in range(1708)], dtype="<U64"
            ),
            "token_ids_sha256": np.asarray(
                [f"{index + 2000:064x}" for index in range(1708)], dtype="<U64"
            ),
            "boundary_index": np.arange(1708, dtype="<i8"),
        }
        rows = [
            {
                "global_index": index,
                "source_id_sha256": projections["source_id_sha256"][index],
                "task_id_sha256": "f" * 64,
                "repository": "grouping-only",
                "request_index": index + 1,
                "stable_feature_eligible": index < 1606,
            }
            for index in range(1708)
        ]
        tensors = module.build_output_tensors(
            projections=projections, alignment_rows=rows, config=config
        )
        self.assertEqual(set(tensors), set(config["campaign"]["output_tensor_keys"]))
        self.assertEqual(tensors["global_index"].dtype, np.dtype("<i8"))
        for name, values in tensors.items():
            self.assertIn(values.dtype.kind, {"i", "f"}, name)
            self.assertNotIn(values.dtype.kind, {"O", "S", "U"}, name)
        np.testing.assert_array_equal(
            tensors["global_index"], np.arange(1606, dtype="<i8")
        )

    def test_run_rejects_forbidden_path_before_filesystem_access(self):
        args = module.build_parser().parse_args(
            [
                "--projection-manifest",
                "/tmp/validation/projection.json",
                "--projection-manifest-sha256",
                "a" * 64,
                "--alignment-index",
                "/tmp/alignment.json",
                "--alignment-index-sha256",
                "b" * 64,
                "--output-data",
                "/tmp/features.npz",
                "--output-manifest",
                "/tmp/features.json",
            ]
        )
        with mock.patch.object(
            module, "_require_regular_file", side_effect=AssertionError("filesystem touched")
        ), mock.patch.object(
            module, "sha256_file", side_effect=AssertionError("hash touched")
        ):
            with self.assertRaises(module._projection.ProjectionError):
                module.run(args)


if __name__ == "__main__":
    unittest.main()
