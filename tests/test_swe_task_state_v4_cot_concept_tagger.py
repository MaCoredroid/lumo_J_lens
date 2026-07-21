from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "cot_concept_tagger", ROOT / "scripts/swe_task_state_v4_cot_concept_tagger.py"
)
tagger = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tagger)


class TagSpaceTests(unittest.TestCase):
    def test_fourteen_scorable_families(self):
        self.assertEqual(len(tagger.CONCEPT_FAMILIES), 14)
        for excluded in ("typographical_error", "repair_success", "repair_summary"):
            self.assertNotIn(excluded, tagger.CONCEPT_FAMILIES)

    def test_prompt_lists_every_family_and_the_text(self):
        msgs = tagger.build_prompt("Let me inspect hyperbolic.py")
        user = msgs[-1]["content"]
        for fid in tagger.CONCEPT_FAMILIES:
            self.assertIn(fid, user)
        self.assertIn("hyperbolic.py", user)
        self.assertIn(tagger.ABSTAIN, user)


class ParseTests(unittest.TestCase):
    def test_clean_family_id(self):
        self.assertEqual(tagger.parse_response("source_localization"), "source_localization")

    def test_tolerates_punctuation_and_quotes(self):
        self.assertEqual(tagger.parse_response(' "task_resolution". '), "task_resolution")

    def test_abstain(self):
        self.assertEqual(tagger.parse_response("none"), tagger.ABSTAIN)

    def test_rejects_ambiguous(self):
        with self.assertRaises(ValueError):
            tagger.parse_response("maybe source_edit or repair")

    def test_rejects_unknown(self):
        with self.assertRaises(ValueError):
            tagger.parse_response("refactoring")


if __name__ == "__main__":
    unittest.main()
