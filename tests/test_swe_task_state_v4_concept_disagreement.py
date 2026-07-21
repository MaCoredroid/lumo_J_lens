from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dis = _load("concept_disagreement", "scripts/swe_task_state_v4_concept_disagreement.py")
scorer = _load("general_concept_scorer", "scripts/swe_task_state_v4_general_concept_scorer.py")


class ConceptDisagreementTests(unittest.TestCase):
    def setUp(self):
        if not scorer.GENERAL_VOCAB.exists():
            self.skipTest("vocab not built")
        self.fine = scorer.family_token_ids(dis.V2)

    def test_concepts_partition_all_fine_families(self):
        covered = [m for members in dis.CONCEPTS.values() for m in members]
        self.assertEqual(len(covered), len(set(covered)), "a fine family is in two super-concepts")
        self.assertEqual(set(covered), set(self.fine), "super-concepts must partition all 14 fine families")

    def test_super_of_maps_members(self):
        self.assertEqual(dis._super_of("source_edit"), "modify_code")
        self.assertEqual(dis._super_of("verification"), "assess")
        self.assertEqual(dis._super_of("source_localization"), "locate")
        self.assertIsNone(dis._super_of("nonexistent"))

    def _readout(self, exps):
        p = Path(tempfile.NamedTemporaryFile("w", suffix=".json", delete=False).name)
        p.write_text(json.dumps({"experiments": exps}))
        return p

    def _exp(self, exp_id, boost_family, tag):
        logp = {}
        for fam, ids in self.fine.items():
            lp = -1.0 if fam == boost_family else -9.0
            for tid in ids:
                logp[tid] = lp
        scored = [{"token_id": t, "logprob": v} for t, v in logp.items()]
        layer = {"layer": 44, "positions": [{"jacobian_lens": {"scored_tokens": scored}}]}
        return {"id": exp_id, "metadata": {"task": "t", "turn": 1, "tag": tag}, "layers": [layer]}

    def test_compute_structure_and_agreement(self):
        # two turns whose boosted concept matches their tag's super-concept -> high agreement
        exps = [
            self._exp("e1", "source_edit", "source_edit"),       # modify_code
            self._exp("e2", "source_localization", "source_localization"),  # locate
        ]
        r = dis.compute(self._readout(exps), n_examples=0)
        self.assertEqual(set(r["concepts"]), {"locate", "modify_code", "assess"})
        self.assertEqual(r["n_turns"], 2)
        self.assertGreaterEqual(r["agreement_rate"], 0.0)
        self.assertLessEqual(r["agreement_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
