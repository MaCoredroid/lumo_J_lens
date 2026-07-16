#!/usr/bin/env python3
"""Focused mutation tests for the offline NF4 evidence checker."""

from __future__ import annotations

import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "check_jlens_nf4_result", SCRIPTS / "check_jlens_nf4_result.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NF4EvidenceCheckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = MODULE.load_evidence()

    def altered(self, field: str):
        return replace(
            self.evidence,
            **{field: copy.deepcopy(getattr(self.evidence, field))},
        )

    def test_committed_failed_pair_is_valid_and_metrics_are_derived(self) -> None:
        metrics = MODULE.validate_evidence(self.evidence)
        self.assertEqual(metrics, MODULE.EXPECTED_CROSS_MODEL)
        self.assertEqual(self.evidence.local_nvfp4["status"], "failed")
        self.assertFalse(
            self.evidence.local_nvfp4["assertions"][
                "all_final_adapter_reconstructions_within_tolerance"
            ]
        )

    def test_provenance_digest_is_bound_to_verification_summary(self) -> None:
        evidence = replace(self.evidence, provenance_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "provenance SHA-256"):
            MODULE.validate_evidence(evidence)

    def test_geometry_aggregate_is_derived_from_layers(self) -> None:
        evidence = self.altered("geometry")
        evidence.geometry["aggregate"]["local_total_frobenius_norm"] += 1.0
        with self.assertRaisesRegex(ValueError, "geometry local norm"):
            MODULE.validate_evidence(evidence)

    def test_evaluation_summary_is_derived_from_readouts(self) -> None:
        evidence = self.altered("evaluation")
        evidence.evaluation["summary"][0]["methods"]["logit_lens"][
            "target_top1_count"
        ] += 1
        with self.assertRaisesRegex(ValueError, "evaluation summary"):
            MODULE.validate_evidence(evidence)

    def test_paired_prompt_identity_must_match(self) -> None:
        evidence = self.altered("local_nvfp4")
        evidence.local_nvfp4["experiments"][0]["prompt_token_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "paired prompt_token_ids"):
            MODULE.validate_evidence(evidence)

    def test_paired_reconstruction_diagnostics_must_be_identical(self) -> None:
        evidence = self.altered("public_nvfp4")
        evidence.public_nvfp4["experiments"][0]["final_logits_reconstruction"][
            "max_abs_error"
        ] = 0.0
        with self.assertRaisesRegex(ValueError, "paired logit diagnostics differ"):
            MODULE.validate_evidence(evidence)

    def test_reconstruction_must_be_the_only_failed_assertion(self) -> None:
        local = copy.deepcopy(self.evidence.local_nvfp4)
        public = copy.deepcopy(self.evidence.public_nvfp4)
        local["assertions"]["lens_hash_matches"] = False
        public["assertions"]["lens_hash_matches"] = False
        evidence = replace(self.evidence, local_nvfp4=local, public_nvfp4=public)
        with self.assertRaisesRegex(ValueError, "paired assertions"):
            MODULE.validate_evidence(evidence)

    def test_row18_prefix_mismatch_cannot_be_erased(self) -> None:
        local = copy.deepcopy(self.evidence.local_nvfp4)
        public = copy.deepcopy(self.evidence.public_nvfp4)
        for report in (local, public):
            report["experiments"][1]["final_logits_reconstruction"][
                "top_k_prefix_token_ids_match"
            ] = True
        evidence = replace(self.evidence, local_nvfp4=local, public_nvfp4=public)
        with self.assertRaisesRegex(ValueError, "top-k prefix flag mismatch"):
            MODULE.validate_evidence(evidence)


if __name__ == "__main__":
    unittest.main()
