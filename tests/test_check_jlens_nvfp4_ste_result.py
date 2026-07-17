#!/usr/bin/env python3
"""Mutation tests for the final native NVFP4/FP8-STE evidence checker."""

from __future__ import annotations

import copy
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_jlens_nvfp4_ste_result",
    ROOT / "scripts" / "check_jlens_nvfp4_ste_result.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NativeNvfp4SteEvidenceCheckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = MODULE.load_evidence()

    def altered(self, field: str):
        return replace(
            self.evidence,
            **{field: copy.deepcopy(getattr(self.evidence, field))},
        )

    def test_committed_evidence_validates_with_adapter_status_separate(self) -> None:
        summary = MODULE.validate_evidence(self.evidence)
        self.assertEqual(summary["fit"]["prompts"], 10)
        self.assertEqual(summary["fit"]["chunks_per_prompt"], 20)
        self.assertEqual(summary["heldout"]["observation_count"], 1008)
        self.assertEqual(
            summary["heldout"]["adapter_status"],
            {"native": "failed", "public": "failed"},
        )
        self.assertNotIn("adapter_certificates", summary["heldout"]["overall"])

    def test_state_digest_is_hard_bound(self) -> None:
        digests = dict(self.evidence.file_sha256)
        digests["fit_state"] = "0" * 64
        evidence = replace(self.evidence, file_sha256=digests)
        with self.assertRaisesRegex(ValueError, "fit_state.*SHA-256"):
            MODULE._check_fit_evidence(evidence)

    def test_capture_exact_forward_proof_cannot_be_erased(self) -> None:
        evidence = self.altered("final_metadata")
        proof = evidence.final_metadata["metadata"]["progress"]["prompts"]["0"][
            "capture_binding"
        ]["proof_claim"]
        proof["shared_internal_tensor_parity"]["all_shared_bit_exact"] = False
        with self.assertRaisesRegex(ValueError, "provenance payload hash"):
            MODULE._check_fit_evidence(evidence)

    def test_full_twenty_chunk_progress_is_required(self) -> None:
        evidence = self.altered("run_progress")
        evidence.run_progress["prompts"]["4"]["chunks"].pop()
        with self.assertRaisesRegex(ValueError, "standalone run progress"):
            MODULE._check_fit_evidence(evidence)

    def test_fit_source_files_are_rehashed_from_the_checkout(self) -> None:
        original = MODULE.artifact_verify._hash_fd

        def changed_source(opened, *, label):
            digest, size = original(opened, label=label)
            if label.startswith("native source "):
                return "0" * 64, size
            return digest, size

        with mock.patch.object(
            MODULE.artifact_verify, "_hash_fd", side_effect=changed_source
        ):
            with self.assertRaisesRegex(ValueError, "native source .* SHA-256"):
                MODULE._check_fit_evidence(self.evidence)

    def test_native_artifact_hash_is_bound_to_report(self) -> None:
        evidence = self.altered("native_report")
        evidence.native_report["lens"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "artifact field sha256"):
            MODULE._check_heldout_evidence(evidence)

    def test_geometry_is_bound_to_exact_native_artifact(self) -> None:
        evidence = self.altered("geometry")
        evidence.geometry["artifacts"]["local"]["state_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "exact native artifact"):
            MODULE._check_geometry(evidence)

    def test_geometry_must_be_finite_but_has_no_quality_threshold(self) -> None:
        evidence = self.altered("geometry")
        evidence.geometry["layers"][0]["frobenius_cosine"] = math.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            MODULE._check_geometry(evidence)

    def test_frozen_heldout_token_ids_cannot_change(self) -> None:
        evidence = self.altered("eval_prompts")
        evidence.eval_prompts["prompts"][0]["token_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "frozen token IDs"):
            MODULE._check_heldout_evidence(evidence)

    def test_public_fit_precision_cannot_be_invented(self) -> None:
        evidence = self.altered("public_report")
        evidence.public_report["lens"]["fit_time_model_precision"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "fit_time_model_precision"):
            MODULE._check_heldout_evidence(evidence)

    def test_all_three_modelopt_shards_are_pinned(self) -> None:
        evidence = self.altered("public_report")
        shards = evidence.public_report["model"]["checkpoint_integrity"]["shards"]
        shards["model-00002-of-00003.safetensors"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint_integrity"):
            MODULE._check_heldout_evidence(evidence)

    def test_unrounded_score_encoding_is_required(self) -> None:
        evidence = self.altered("native_report")
        evidence.native_report["score_encoding"] = "rounded-decimal"
        with self.assertRaisesRegex(ValueError, "score_encoding"):
            MODULE._check_heldout_evidence(evidence)

    def test_residual_capture_manifests_must_be_identical(self) -> None:
        evidence = self.altered("public_report")
        evidence.public_report["experiments"][2]["residual_capture_manifest"][
            "sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "diagnostic mismatch"):
            MODULE._check_heldout_evidence(evidence)

    def test_paired_metrics_are_independently_recomputed(self) -> None:
        evidence = self.altered("paired_report")
        comparison = evidence.paired_report["metrics"]["overall"]["comparisons"][
            "native_vs_public_jacobian_lens"
        ]
        comparison["top1_agreement_count"] += 1
        with self.assertRaisesRegex(ValueError, "paired report metrics"):
            MODULE._check_heldout_evidence(evidence)


if __name__ == "__main__":
    unittest.main()
