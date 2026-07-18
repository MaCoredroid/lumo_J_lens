#!/usr/bin/env python3
"""Focused tests for the contextual-evidence publication verifier."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/check_swe_contextual_evidence_publication.py"
SPEC = importlib.util.spec_from_file_location(
    "check_swe_contextual_evidence_publication", MODULE_PATH
)
assert SPEC and SPEC.loader
CHECK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECK
SPEC.loader.exec_module(CHECK)


def load(name: str) -> dict:
    return json.loads((CHECK.BUNDLE / name).read_bytes())


class ContextualEvidencePublicationTest(unittest.TestCase):
    def test_real_publication_passes(self) -> None:
        self.assertEqual(
            CHECK.verify_publication(),
            {
                "task_count": 12,
                "primary_task_count": 8,
                "supported_card_count": 3,
                "classification": "frozen_directional_point_rule_pass",
                "operational_usefulness": False,
            },
        )

    def test_claim_boundary_cannot_be_upgraded(self) -> None:
        manifest = load("run_manifest.json")
        manifest["claim_boundary"]["classification_is_not_operational_usefulness"] = False
        with self.assertRaisesRegex(CHECK.PublicationError, "claim boundary"):
            CHECK.verify_manifest(manifest)

    def test_directional_classification_cannot_be_called_operational(self) -> None:
        manifest = load("run_manifest.json")
        analysis = load("analysis.json")
        decision = analysis["profiles"]["primary_stable"]["public_usefulness_decision"]
        decision["classification_is_not_operational_usefulness"] = False
        with self.assertRaisesRegex(CHECK.PublicationError, "directional decision"):
            CHECK.verify_analysis(analysis, manifest)

    def test_copy_failure_and_uncertainty_are_required(self) -> None:
        manifest = load("run_manifest.json")
        analysis = load("analysis.json")
        stable = analysis["profiles"]["primary_stable"]
        changed_copy = copy.deepcopy(analysis)
        changed_copy["profiles"]["primary_stable"]["copy_retrieval_diagnostic"][
            "passed"
        ] = True
        with self.assertRaisesRegex(CHECK.PublicationError, "copy diagnostic"):
            CHECK.verify_analysis(changed_copy, manifest)

        stable["public_usefulness_decision"]["uncertainty_diagnostic"][
            "all_four_confidence_interval_lowers_positive"
        ] = True
        with self.assertRaisesRegex(CHECK.PublicationError, "uncertainty boundary"):
            CHECK.verify_analysis(analysis, manifest)

    def test_withheld_card_cannot_expose_a_lens_claim(self) -> None:
        cards = load("cards.json")
        withheld = next(
            card
            for card in cards["cards"]
            if card["lens_why_guard"]["status"] == "withheld"
        )
        withheld["fields"]["WHY"]["lens"]["claim"] = "unsupported hidden thought"
        with self.assertRaisesRegex(CHECK.PublicationError, "withheld WHY"):
            CHECK.verify_cards(cards)

    def test_paths_fail_closed(self) -> None:
        with self.assertRaisesRegex(CHECK.PublicationError, "non-canonical"):
            CHECK.canonical_relative("../analysis.json", "test path")


if __name__ == "__main__":
    unittest.main()
