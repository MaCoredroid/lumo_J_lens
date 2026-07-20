from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_swe_task_state_v4_raw_capture.py"
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_raw_capture.json"

spec = importlib.util.spec_from_file_location("swe_task_state_v4_raw_capture", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class RawCaptureContractTests(unittest.TestCase):
    def config(self):
        return module.load_json_strict(CONFIG_PATH)

    def test_checked_in_config_is_exact_and_claims_remain_false(self):
        config = module.validate_config(self.config())
        self.assertTrue(all(value is False for value in config["claim_scope"].values()))
        self.assertEqual(config["capture"]["layers"], list(range(24, 48)))
        self.assertEqual(config["lens"]["sha256"], module.base.LENS_SHA256)
        self.assertEqual(
            config["downstream_feature_boundary"]["allowed_tensor_features"],
            ["raw_residual", "public_j_state"],
        )
        self.assertTrue(
            config["downstream_feature_boundary"][
                "semantic_ids_as_features_forbidden"
            ]
        )

    def test_config_rejects_layer_lens_input_and_claim_drift(self):
        mutations = []
        changed = self.config()
        changed["capture"]["layers"][0] = 23
        mutations.append(changed)
        changed = self.config()
        changed["lens"]["sha256"] = "0" * 64
        mutations.append(changed)
        changed = self.config()
        changed["feature_independence"]["forward_inputs"].append("metadata")
        mutations.append(changed)
        changed = self.config()
        changed["claim_scope"]["private_chain_of_thought_reconstructed"] = True
        mutations.append(changed)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(module.CaptureError):
                    module.validate_config(mutation)

    def test_source_sanitizer_is_label_and_text_inert(self):
        source = [
            {
                "id": "boundary-a",
                "token_ids": [1, 2, 3],
                "text": "secret visible prefix",
                "metadata": {"action": "verification", "outcome": "pass"},
                "target_token_id": 99,
                "score_token_ids": [88],
                "emotion": "stress",
                "private_reasoning": "not a feature",
            }
        ]
        sanitized, identities = module.sanitize_source_bundle(source)
        changed = copy.deepcopy(source)
        changed[0]["text"] = "completely changed"
        changed[0]["metadata"] = {"action": "implementation", "outcome": "fail"}
        changed[0]["target_token_id"] = 7
        changed[0]["score_token_ids"] = [6]
        changed[0]["emotion"] = "joy"
        changed[0]["private_reasoning"] = "changed"
        sanitized_changed, identities_changed = module.sanitize_source_bundle(changed)
        self.assertEqual(sanitized, sanitized_changed)
        self.assertEqual(identities, identities_changed)
        self.assertEqual(sanitized, [{"id": hashlib.sha256(b"boundary-a").hexdigest(), "token_ids": [1, 2, 3]}])
        self.assertNotIn("text", sanitized[0])
        self.assertNotIn("metadata", sanitized[0])

    def test_token_or_source_id_mutation_changes_authenticated_identity(self):
        _, first = module.sanitize_source_bundle([{"id": "a", "token_ids": [1, 2]}])
        _, second = module.sanitize_source_bundle([{"id": "a", "token_ids": [1, 3]}])
        _, third = module.sanitize_source_bundle([{"id": "b", "token_ids": [1, 2]}])
        self.assertNotEqual(first[0]["token_ids_sha256"], second[0]["token_ids_sha256"])
        self.assertNotEqual(first[0]["source_id_sha256"], third[0]["source_id_sha256"])

    def test_source_sanitizer_rejects_missing_bad_or_duplicate_rows(self):
        bad_values = [
            [],
            [{}],
            [{"token_ids": []}],
            [{"token_ids": [True]}],
            [{"token_ids": [-1]}],
            [{"id": "a", "token_ids": [1]}, {"id": "a", "token_ids": [2]}],
        ]
        for value in bad_values:
            with self.subTest(value=value):
                with self.assertRaises(module.CaptureError):
                    module.sanitize_source_bundle(value)

    def test_strict_json_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"a":1,"a":2}')
            with self.assertRaises(module.CaptureError):
                module.load_json_strict(path)

    def test_reserved_validation_path_guard_is_fail_closed(self):
        fragments = ["reserved", "validation"]
        self.assertTrue(
            module._path_has_forbidden_fragment(
                ROOT / ".cache" / "reserved_validation" / "x.json", fragments
            )
        )
        self.assertFalse(
            module._path_has_forbidden_fragment(
                ROOT / ".cache" / "swe_jlens_intermediate" / "x.json", fragments
            )
        )

    def test_run_rejects_forbidden_path_before_any_filesystem_helper(self):
        args = module.build_parser().parse_args(
            [
                "--prompts-file",
                "/tmp/reserved_source.json",
                "--source-bundle-sha256",
                "a" * 64,
                "--reference-report",
                "/tmp/reference.json",
                "--reference-report-sha256",
                "b" * 64,
                "--state-output-dir",
                "/tmp/output",
                "--output",
                "/tmp/output/report.json",
            ]
        )
        with mock.patch.object(
            module, "_require_regular_file", side_effect=AssertionError("filesystem touched")
        ), mock.patch.object(
            module, "sha256_file", side_effect=AssertionError("hash touched")
        ):
            with self.assertRaises(module.CaptureError):
                module.run(args)

    def test_parent_symlink_to_forbidden_path_is_rejected_before_hash_or_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forbidden = root / "reserved_material"
            forbidden.mkdir()
            source = forbidden / "source.json"
            source.write_text("[]")
            reference = root / "reference.json"
            reference.write_text("{}")
            alias = root / "alias"
            alias.symlink_to(forbidden, target_is_directory=True)
            args = module.build_parser().parse_args(
                [
                    "--prompts-file",
                    str(alias / "source.json"),
                    "--source-bundle-sha256",
                    "a" * 64,
                    "--reference-report",
                    str(reference),
                    "--reference-report-sha256",
                    "b" * 64,
                    "--state-output-dir",
                    str(root / "output"),
                    "--output",
                    str(root / "output" / "report.json"),
                ]
            )
            with mock.patch.object(
                module, "_require_regular_file", side_effect=AssertionError("file opened")
            ), mock.patch.object(
                module, "sha256_file", side_effect=AssertionError("hash touched")
            ):
                with self.assertRaises(module.CaptureError):
                    module.run(args)

    def test_reference_merge_requires_exact_order_id_and_residual_manifest(self):
        manifest = {
            "algorithm": (
                "SHA-256 over length-prefixed canonical layer/shape/dtype/"
                "token-position/byte-count headers and logical row-major FP32 bytes"
            ),
            "sha256": "a" * 64,
            "tensor_count": 64,
            "logical_bytes": 64 * 5120 * 4,
            "token_positions": [4],
        }
        identities = [
            {
                "index": 0,
                "source_id": "a",
                "source_id_sha256": "b" * 64,
                "token_ids_sha256": "c" * 64,
                "token_count": 5,
                "token_position": 4,
            }
        ]
        captures = [
            {
                "index": 0,
                "residual_capture_manifest": manifest,
                "shard": {"reload_verified": True},
                "final_model_top1_matches_greedy": True,
                "final_norm_reconstruction": {"within_tolerance": True},
                "final_logits_reconstruction": {"within_tolerance": False},
            }
        ]
        references = [{"index": 0, "source_id": "a", "residual_capture_manifest": manifest}]
        merged = module._merge_and_validate_records(identities, captures, references)
        self.assertTrue(merged[0]["capture_valid"])
        self.assertFalse(merged[0]["vocabulary_adapter_strict"])
        changed = copy.deepcopy(references)
        changed[0]["residual_capture_manifest"]["sha256"] = "d" * 64
        with self.assertRaises(module.CaptureError):
            module._merge_and_validate_records(identities, captures, changed)
        changed = copy.deepcopy(references)
        changed[0]["source_id"] = "other"
        with self.assertRaises(module.CaptureError):
            module._merge_and_validate_records(identities, captures, changed)

    def test_no_clobber_writer_and_finite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            module._write_json_no_clobber(path, {"x": 1.25})
            self.assertEqual(json.loads(path.read_text()), {"x": 1.25})
            with self.assertRaises(module.CaptureError):
                module._write_json_no_clobber(path, {"x": 2})

    def test_base_report_stream_audit_retains_only_provenance(self):
        report = {
            "schema_version": module.base.SCHEMA_VERSION,
            "status": "failed",
            "model": {"repo_id": "model", "checkpoint_integrity": {"x": 1}},
            "lens": {"sha256": "a" * 64},
            "runtime": {"max_model_len": 16, "gpu_memory_utilization": 0.78},
            "assertions": {"adapter": False},
            "experiments": [
                {"id": "a", "generated_text": "forbidden"},
                {"id": "b", "prompt": "also forbidden"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report, sort_keys=True))
            observed = module.audit_base_report(path)
        self.assertEqual(observed["experiment_count"], 2)
        self.assertEqual(observed["runtime"], report["runtime"])
        self.assertNotIn("experiments", observed)
        self.assertNotIn("generated_text", json.dumps(observed))

    def test_interception_writes_exact_states_returns_unmodified_result_and_counts(self):
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is available only in the pinned vLLM test lane")
        import torch

        layers = list(range(24, 48))
        algorithm = (
            "SHA-256 over length-prefixed canonical layer/shape/dtype/"
            "token-position/byte-count headers and logical row-major FP32 bytes"
        )

        class FakeModel:
            _jlens_positions = (7,)
            _jlens_captures = {
                layer: torch.full((1, 5120), float(layer), dtype=torch.float32)
                for layer in layers
            }

        result = {
            "residual_capture_manifest": {
                "algorithm": algorithm,
                "sha256": "a" * 64,
                "tensor_count": 64,
                "logical_bytes": 64 * 5120 * 4,
                "token_positions": [7],
            },
            "final_model_readout": [{"token_ids": [3], "target_token_id": 3}],
            "captured_final_model_readout": [
                {"token_ids": [3], "target_token_id": 3}
            ],
            "final_norm_reconstruction": {"within_tolerance": True},
            "final_logits_reconstruction": {"within_tolerance": True},
        }

        old_readout = module._ORIGINAL_READOUT
        old_transport = module._ORIGINAL_TRANSPORT
        old_base_transport = module.base.transport_residual
        try:
            module._ORIGINAL_TRANSPORT = lambda residual, _jacobian: residual + 0.25

            def fake_readout(model, *, layers, **_kwargs):
                for layer in layers:
                    module.base.transport_residual(model._jlens_captures[layer], None)
                return result

            module._ORIGINAL_READOUT = fake_readout
            module.base.transport_residual = module._capturing_transport
            with tempfile.TemporaryDirectory() as directory:
                shards = Path(directory) / "shards"
                shards.mkdir()
                module._CAPTURE_CONTEXT.update(
                    {
                        "armed": False,
                        "layers": layers,
                        "transported": [],
                        "records": [],
                        "shards_dir": shards,
                    }
                )
                observed = module._capturing_readout(
                    FakeModel(),
                    lens_path="unused",
                    layers=tuple(layers),
                    top_k=5,
                    target_token_ids=(3,),
                )
                self.assertIs(observed, result)
                self.assertEqual(len(module._CAPTURE_CONTEXT["records"]), 1)
                self.assertTrue((shards / "boundary-000000.safetensors").is_file())
                self.assertFalse(any(".tmp-" in path.name for path in shards.iterdir()))

                def short_readout(model, *, layers, **_kwargs):
                    for layer in layers[:-1]:
                        module.base.transport_residual(model._jlens_captures[layer], None)
                    return result

                module._ORIGINAL_READOUT = short_readout
                module._CAPTURE_CONTEXT["records"] = []
                with self.assertRaises(module.CaptureError):
                    module._capturing_readout(
                        FakeModel(),
                        lens_path="unused",
                        layers=tuple(layers),
                        top_k=5,
                        target_token_ids=(3,),
                    )
        finally:
            module._ORIGINAL_READOUT = old_readout
            module._ORIGINAL_TRANSPORT = old_transport
            module.base.transport_residual = old_base_transport


if __name__ == "__main__":
    unittest.main()
