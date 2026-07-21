from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/swe_task_state_v4_general_concept_forms.json"

_spec = importlib.util.spec_from_file_location(
    "general_concept_forms", ROOT / "scripts/swe_task_state_v4_general_concept_forms.py"
)
gcf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gcf)

# The 3 families the concept-chain marks unscorable are intentionally absent.
_UNSCORABLE = {"typographical_error", "repair_success", "repair_summary"}


class GeneralConceptFormsTests(unittest.TestCase):
    def setUp(self):
        if not CONFIG.exists():
            self.skipTest("general concept-form config not built")
        self.doc = json.loads(CONFIG.read_text())

    def test_all_fourteen_scorable_families(self):
        fams = self.doc["families"]
        self.assertEqual(len(fams), 14)
        for excluded in _UNSCORABLE:
            self.assertNotIn(excluded, fams)

    def test_each_family_has_two_plus_forms(self):
        for family, forms in self.doc["families"].items():
            self.assertGreaterEqual(len(forms), 2, family)

    def test_token_ids_globally_unique(self):
        ids = [f["token_id"] for forms in self.doc["families"].values() for f in forms]
        self.assertEqual(len(ids), len(set(ids)), "a token is shared across families")

    def test_candidate_families_match_the_concept_chain_scorable_set(self):
        # the module's candidates must not name any unscorable family
        self.assertEqual(set(gcf.CANDIDATES) & _UNSCORABLE, set())
        self.assertEqual(len(gcf.CANDIDATES), 14)


if __name__ == "__main__":
    unittest.main()
