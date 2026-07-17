#!/usr/bin/env python3
"""Focused tests for packed/dense all-layer comparison reporting."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_nvfp4_all_layers as VALIDATOR  # noqa: E402


class RowComparisonTest(unittest.TestCase):
    def _rows(self) -> dict[int, torch.Tensor]:
        return {
            layer: torch.tensor(
                [[1.0 + layer / 100.0, -0.5]], dtype=torch.float32
            )
            for layer in VALIDATOR.SOURCE_LAYERS
        }

    def _reference(
        self, rows: dict[int, torch.Tensor]
    ) -> dict[str, object]:
        return {
            "rows": {
                "layers": VALIDATOR.summarize_rows(rows),
            }
        }

    def test_all_dense_hashes_and_small_packed_errors_pass(self) -> None:
        dense = self._rows()
        packed = {
            layer: value + torch.tensor([[1e-4, -1e-4]])
            for layer, value in dense.items()
        }
        result = VALIDATOR.compare_rows(
            packed,
            dense,
            self._reference(dense),
            max_relative_rms=1e-3,
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["dense_certificate_hash_match_count"], 63)
        self.assertEqual(result["packed_certificate_hash_match_count"], 0)
        self.assertTrue(result["all_layers_within_tolerance"])

    def test_dense_certificate_hash_mismatch_fails_closed(self) -> None:
        dense = self._rows()
        reference = self._reference(dense)
        reference["rows"]["layers"]["17"]["sha256_float32"] = "0" * 64
        result = VALIDATOR.compare_rows(
            dense,
            dense,
            reference,
            max_relative_rms=0.0,
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["dense_certificate_hash_match_count"], 62)

    def test_one_layer_outside_error_tolerance_fails(self) -> None:
        dense = self._rows()
        packed = {layer: value.clone() for layer, value in dense.items()}
        packed[42] += 0.25
        result = VALIDATOR.compare_rows(
            packed,
            dense,
            self._reference(dense),
            max_relative_rms=0.01,
        )
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["layers"]["42"]["within_tolerance"])


class CaptureAuthorityTest(unittest.TestCase):
    def test_proof_binds_report_payload_and_verifier(self) -> None:
        token_ids = [10, 11, 12]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_path = root / "observer.pt"
            report_path = root / "observer.json"
            proof_path = root / "proof.json"
            verifier_path = root / "prove_nvfp4_capture_pair.py"
            tensor_path.write_bytes(b"captured tensors")
            verifier_path.write_text("# pinned verifier\n", encoding="utf-8")
            report = {
                "schema_version": 1,
                "status": "captured",
                "runtime": {
                    "mode": "compiled-observer",
                    "target_profile": "all",
                    "target_layers": list(range(64)),
                    "mtp_enabled": False,
                    "language_model_only": True,
                },
                "model": {
                    "repo_id": VALIDATOR.MODEL_REPO,
                    "revision": VALIDATOR.PINNED_REVISION,
                    "resolved_path": "/cache/snapshots/"
                    + VALIDATOR.PINNED_REVISION,
                    "identity": {
                        "policy": VALIDATOR.MODEL_IDENTITY_POLICY,
                        "repo_id": VALIDATOR.MODEL_REPO,
                        "revision": VALIDATOR.PINNED_REVISION,
                        "resolved_path": "/cache/snapshots/"
                        + VALIDATOR.PINNED_REVISION,
                        "metadata_sha256": VALIDATOR.PINNED_METADATA_SHA256,
                        "strict_pinned_validation": True,
                        "validator": "ModelOptCheckpoint(strict_pinned=True)",
                    },
                },
                "prompt": {"token_ids": token_ids},
                "capture": {
                    "missing_required": [],
                    "truncated": [],
                    "replay_parameter_provenance": {
                        str(index): {} for index in range(785)
                    },
                    "tensor_summaries": {str(index): {} for index in range(1905)},
                    "compile_visible_observer_tensors": [
                        str(index) for index in range(432)
                    ],
                    "tensor_output": str(tensor_path),
                },
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")

            def record(path: Path) -> dict[str, object]:
                return {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            proof = {
                "schema_version": 1,
                "status": "passed",
                "claim": {
                    "mtp": "off",
                    "observer_graph_modified": True,
                    "observer_modification_discharged": True,
                },
                "configuration": {
                    "model_revision": VALIDATOR.PINNED_REVISION,
                    "target_profile": "all",
                    "target_layers": list(range(64)),
                    "mtp_enabled": False,
                    "language_model_only": True,
                    "prompt": {"token_ids": token_ids},
                },
                "generation_record_parity": {"exact": True},
                "shared_internal_tensor_parity": {
                    "shared_tensor_count": 688,
                    "observer_only_tensor_count": 432,
                    "all_shared_bit_exact": True,
                },
                "replay_parameter_parity": {
                    "parameter_count": 785,
                    "all_names_equal": True,
                    "all_shapes_equal": True,
                    "all_dtypes_equal": True,
                    "all_content_hashes_equal": True,
                    "json_provenance_equal": True,
                },
                "observer_capture_completeness": {
                    "required_missing": [],
                    "truncated": [],
                },
                "artifacts": {
                    "observer_json": record(report_path),
                    "observer_tensors": record(tensor_path),
                },
                "verifier": record(verifier_path),
            }
            proof_path.write_text(json.dumps(proof), encoding="utf-8")
            fake_module_path = root / "validate_nvfp4_all_layers.py"
            fake_module_path.write_text("# validator\n", encoding="utf-8")
            with mock.patch.object(VALIDATOR, "__file__", str(fake_module_path)):
                authority = VALIDATOR.validate_capture_authority(
                    tensor_path,
                    {"prompt_token_ids": token_ids},
                    report_path,
                    proof_path,
                )
                self.assertEqual(authority["status"], "passed")
                proof["generation_record_parity"]["exact"] = False
                proof_path.write_text(json.dumps(proof), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "proof scope"):
                    VALIDATOR.validate_capture_authority(
                        tensor_path,
                        {"prompt_token_ids": token_ids},
                        report_path,
                        proof_path,
                    )


if __name__ == "__main__":
    unittest.main()
