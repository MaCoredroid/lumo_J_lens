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
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_activation_projection.py"
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_activation_projection.json"

spec = importlib.util.spec_from_file_location(
    "swe_task_state_v4_activation_projection", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class ActivationProjectionTests(unittest.TestCase):
    def config(self):
        return module.load_json_strict(CONFIG_PATH)

    def test_checked_in_config_and_false_claim_scope_are_exact(self):
        config = module.validate_config(self.config())
        self.assertEqual(config["input"]["layers"], list(range(24, 48)))
        self.assertEqual(config["projection"]["width"], 256)
        self.assertEqual(len(config["projection"]["seeds"]), 4)
        self.assertEqual(config["projection"]["primary_seed_index"], 0)
        self.assertEqual(
            config["projection"]["sensitivity_seed_indices"], [1, 2, 3]
        )
        self.assertTrue(all(value is False for value in config["claim_scope"].values()))
        self.assertTrue(
            config["downstream_feature_boundary"][
                "authentication_fields_as_features_forbidden"
            ]
        )

    def test_config_rejects_capture_hash_seed_pooling_and_claim_drift(self):
        mutations = []
        changed = self.config()
        changed["input"]["capture_config_sha256"] = "0" * 64
        mutations.append(changed)
        changed = self.config()
        changed["projection"]["seeds"][0] = "label-derived"
        mutations.append(changed)
        changed = self.config()
        changed["pooling"]["normalization_before_pooling"] = "final_rmsnorm"
        mutations.append(changed)
        changed = self.config()
        changed["claim_scope"]["private_chain_of_thought_reconstructed"] = True
        mutations.append(changed)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(module.ProjectionError):
                    module.validate_config(mutation)

    def test_countsketch_mapping_is_deterministic_seeded_and_label_free(self):
        first = module.countsketch_mapping(seed="fixed", hidden_size=17, width=5)
        second = module.countsketch_mapping(seed="fixed", hidden_size=17, width=5)
        changed = module.countsketch_mapping(seed="changed", hidden_size=17, width=5)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertNotEqual(first[2], changed[2])
        self.assertTrue(np.all(first[0] < 5))
        self.assertEqual(set(first[1].tolist()), {-1, 1})

    def test_countsketch_hand_calculation_and_scale(self):
        vector = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buckets = np.asarray([0, 1, 0, 1], dtype=np.uint32)
        signs = np.asarray([1, -1, 1, 1], dtype=np.int8)
        observed = module.countsketch_vector(
            vector, buckets=buckets, signs=signs, width=2
        )
        scale = np.sqrt(2.0 / 4.0)
        expected = np.asarray([(1.0 + 3.0) * scale, (-2.0 + 4.0) * scale], dtype=np.float32)
        np.testing.assert_array_equal(observed, expected)

    def test_band_pooling_statistics_and_source_separation_are_exact(self):
        config = self.config()
        hidden = config["input"]["hidden_size"]
        raw = np.empty((24, hidden), dtype=np.float32)
        public_j = np.empty((24, hidden), dtype=np.float32)
        for index in range(24):
            raw[index] = index + 1
            public_j[index] = 2 * (index + 1)
        mappings = [module.countsketch_mapping(seed="test", hidden_size=hidden, width=256)]
        sketches, stats = module.project_state_pair(
            raw, public_j, config=config, mappings=mappings
        )
        self.assertEqual(sketches.shape, (1, 2, 3, 256))
        self.assertEqual(stats.shape, (2, 3, 3))
        np.testing.assert_allclose(sketches[0, 1], 2 * sketches[0, 0], rtol=1e-6)
        np.testing.assert_allclose(stats[1], 2 * stats[0], rtol=1e-6)
        self.assertAlmostEqual(float(stats[0, 0, 0]), 3.5, places=6)
        self.assertAlmostEqual(float(stats[0, 0, 1]), 3.5, places=6)
        self.assertAlmostEqual(float(stats[0, 0, 2]), 5.0, places=6)

    def test_projection_has_no_label_metadata_or_target_argument(self):
        parameters = module.project_state_pair.__annotations__
        self.assertNotIn("labels", parameters)
        self.assertNotIn("metadata", parameters)
        self.assertNotIn("target_token_id", parameters)
        self.assertNotIn("outcome", parameters)

    def test_256_to_64_fold_is_exact(self):
        values = np.arange(256, dtype=np.float32)
        observed = module.fold_projection_256_to_64(values)
        expected = (values[:64] + values[64:128] + values[128:192] + values[192:]) / 2
        np.testing.assert_array_equal(observed, expected.astype(np.float32))

    def test_primary_extractor_is_bit_invariant_to_all_authentication_fields(self):
        rng = np.random.default_rng(7)
        sketches = rng.standard_normal((3, 4, 2, 3, 256)).astype(np.float32)
        arrays = {
            "sketches": sketches,
            "band_statistics": rng.standard_normal((3, 2, 3, 3)).astype(np.float32),
            "source_id_sha256": np.asarray(["a" * 64, "b" * 64, "c" * 64]),
            "token_ids_sha256": np.asarray(["d" * 64, "e" * 64, "f" * 64]),
            "boundary_index": np.arange(3),
        }
        first = module.extract_primary_current_features(
            arrays, seed_index=0, source="public_j_state"
        )
        changed = copy.deepcopy(arrays)
        changed["source_id_sha256"] = changed["source_id_sha256"][::-1]
        changed["token_ids_sha256"] = np.asarray(["0" * 64] * 3)
        changed["boundary_index"] = np.asarray([99, 98, 97])
        changed["band_statistics"] *= -100
        changed["metadata"] = np.asarray(["label", "outcome", "target"])
        second = module.extract_primary_current_features(
            changed, seed_index=0, source="public_j_state"
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (3, 192))

    def test_primary_and_sensitivity_seed_apis_fail_closed(self):
        rng = np.random.default_rng(19)
        arrays = {
            "sketches": rng.standard_normal((2, 4, 2, 3, 256)).astype(np.float32)
        }
        primary = module.extract_primary_current_features(
            arrays, seed_index=0, source="raw_residual"
        )
        sensitivity = module.extract_sensitivity_current_features(
            arrays, seed_index=1, source="raw_residual"
        )
        self.assertEqual(primary.shape, (2, 192))
        self.assertEqual(sensitivity.shape, (2, 192))
        self.assertFalse(np.array_equal(primary, sensitivity))
        for rejected in (True, 1, 2, 3, 4, -1):
            with self.subTest(primary_seed=rejected):
                with self.assertRaises(module.ProjectionError):
                    module.extract_primary_current_features(
                        arrays, seed_index=rejected, source="raw_residual"
                    )
        for rejected in (False, 0, 4, -1):
            with self.subTest(sensitivity_seed=rejected):
                with self.assertRaises(module.ProjectionError):
                    module.extract_sensitivity_current_features(
                        arrays, seed_index=rejected, source="raw_residual"
                    )

    def test_capture_manifest_validation_rejects_overclaim_or_invalid_boundary(self):
        config = self.config()
        manifest = {
            "schema_version": 1,
            "kind": config["input"]["kind"],
            "status": "passed",
            "status_scope": "raw_and_public_j_pre_vocabulary_state_capture_only",
            "capture_config": {"sha256": config["input"]["capture_config_sha256"]},
            "source_bundle": {},
            "reference_report": {},
            "base_report": {},
            "implementation": {},
            "normalized_cli_contract": {},
            "model": {},
            "lens": {},
            "capture": {
                "layers": list(range(24, 48)),
                "position": "causal_prefix_tail_only",
                "positions_argument": [-1],
                "stream_final_only_required": True,
                "raw_tensor": "post_block_residual_before_final_norm",
                "transported_tensor": "public_j_state_before_bfloat16_final_norm_or_vocabulary_projection",
                "storage_dtype": "little_endian_float32",
                "shard_format": "safetensors",
            },
            "feature_independence": {},
            "downstream_feature_boundary": {
                "allowed_tensor_features": ["raw_residual", "public_j_state"],
                "semantic_ids_as_features_forbidden": True,
                "base_report_fields_as_features_forbidden": True,
            },
            "claim_scope": {"private_chain_of_thought_reconstructed": False},
            "summary": {},
            "boundaries": [
                {
                    "index": 0,
                    "source_id_sha256": "a" * 64,
                    "token_ids_sha256": "b" * 64,
                    "token_count": 5,
                    "token_position": 4,
                    "residual_capture_manifest": {},
                    "capture_valid": True,
                    "reference_residual_manifest_equal": True,
                    "shard": {},
                    "vocabulary_adapter_strict": False,
                    "final_model_top1_matches_greedy": True,
                    "final_norm_reconstruction_within_tolerance": True,
                    "final_logits_reconstruction_within_tolerance": False,
                }
            ],
        }
        self.assertEqual(len(module._validate_capture_manifest(manifest, config=config)), 1)
        changed = copy.deepcopy(manifest)
        changed["claim_scope"]["private_chain_of_thought_reconstructed"] = True
        with self.assertRaises(module.ProjectionError):
            module._validate_capture_manifest(changed, config=config)
        changed = copy.deepcopy(manifest)
        changed["boundaries"][0]["capture_valid"] = False
        with self.assertRaises(module.ProjectionError):
            module._validate_capture_manifest(changed, config=config)

    def test_strict_json_reserved_path_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"a":1,"a":2}')
            with self.assertRaises(module.ProjectionError):
                module.load_json_strict(duplicate)
            self.assertTrue(
                module._path_has_forbidden_fragment(
                    root / "reserved_validation" / "x", ["reserved", "validation"]
                )
            )
            output = root / "features.npz"
            module._write_npz_no_clobber(output, {"x": np.asarray([1], dtype=np.int64)})
            with self.assertRaises(module.ProjectionError):
                module._write_npz_no_clobber(output, {"x": np.asarray([2], dtype=np.int64)})

    def test_run_rejects_forbidden_path_before_any_filesystem_helper(self):
        args = module.build_parser().parse_args(
            [
                "--capture-manifest",
                "/tmp/validation_capture.json",
                "--capture-manifest-sha256",
                "a" * 64,
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
            with self.assertRaises(module.ProjectionError):
                module.run(args)

    def test_parent_symlink_to_forbidden_path_is_rejected_before_hash_or_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forbidden = root / "validation_material"
            forbidden.mkdir()
            capture = forbidden / "capture.json"
            capture.write_text("{}")
            alias = root / "alias"
            alias.symlink_to(forbidden, target_is_directory=True)
            args = module.build_parser().parse_args(
                [
                    "--capture-manifest",
                    str(alias / "capture.json"),
                    "--capture-manifest-sha256",
                    "a" * 64,
                    "--output-data",
                    str(root / "features.npz"),
                    "--output-manifest",
                    str(root / "features.json"),
                ]
            )
            with mock.patch.object(
                module, "_require_regular_file", side_effect=AssertionError("file opened")
            ), mock.patch.object(
                module, "sha256_file", side_effect=AssertionError("hash touched")
            ):
                with self.assertRaises(module.ProjectionError):
                    module.run(args)


if __name__ == "__main__":
    unittest.main()
