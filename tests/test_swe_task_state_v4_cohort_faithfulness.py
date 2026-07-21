from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


faith = _load("cohort_faithfulness", "scripts/swe_task_state_v4_cohort_faithfulness.py")
scorer = _load("general_concept_scorer", "scripts/swe_task_state_v4_general_concept_scorer.py")


def _experiment(exp_id: str, boosted_family: str, tag: str, fam_ids, n_layers: int = 2) -> dict:
    logprobs = {}
    for fam, ids in fam_ids.items():
        lp = -1.0 if fam == boosted_family else -9.0
        for tid in ids:
            logprobs[tid] = lp
    scored = [{"token_id": tid, "logprob": lp} for tid, lp in logprobs.items()]
    layer = {"positions": [{"jacobian_lens": {"scored_tokens": scored}}]}
    return {"id": exp_id, "metadata": {"task": "t", "turn": 1, "tag": tag}, "layers": [layer] * n_layers}


class CohortFaithfulnessTests(unittest.TestCase):
    def setUp(self):
        if not scorer.GENERAL_VOCAB.exists():
            self.skipTest("general vocab not built")
        self.fam_ids = scorer.family_token_ids()

    def _report(self, experiments) -> Path:
        p = Path(tempfile.NamedTemporaryFile("w", suffix=".json", delete=False).name)
        p.write_text(json.dumps({"experiments": experiments}))
        return p

    def test_hit_and_miss_gives_half(self):
        exps = [
            _experiment("e1", "verification", "verification", self.fam_ids),  # hit
            _experiment("e2", "repair", "source_localization", self.fam_ids),  # miss
        ]
        result = faith.compute(self._report(exps))
        self.assertEqual(result["n_turns_compared"], 2)
        self.assertAlmostEqual(result["faithfulness_raw"], 0.5)

    def test_none_tags_excluded(self):
        exps = [
            _experiment("e1", "verification", "verification", self.fam_ids),
            _experiment("e2", "repair", "none", self.fam_ids),  # excluded
        ]
        result = faith.compute(self._report(exps))
        self.assertEqual(result["n_turns_compared"], 1)

    def test_per_family_recall_present(self):
        exps = [_experiment(f"e{i}", "verification", "verification", self.fam_ids) for i in range(3)]
        result = faith.compute(self._report(exps))
        self.assertIn("verification", result["per_family"])
        self.assertEqual(result["per_family"]["verification"]["n"], 3)
        self.assertEqual(result["per_family"]["verification"]["recall_raw"], 1.0)


if __name__ == "__main__":
    unittest.main()
