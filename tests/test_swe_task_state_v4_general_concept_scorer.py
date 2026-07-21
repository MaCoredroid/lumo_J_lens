from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "general_concept_scorer", ROOT / "scripts/swe_task_state_v4_general_concept_scorer.py"
)
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)


def _experiment(exp_id: str, logprob_by_id: dict[int, float], n_layers: int = 2) -> dict:
    scored = [{"token_id": tid, "logprob": lp} for tid, lp in logprob_by_id.items()]
    layer = {"positions": [{"jacobian_lens": {"scored_tokens": scored}}]}
    return {"id": exp_id, "layers": [layer for _ in range(n_layers)]}


class GeneralScorerTests(unittest.TestCase):
    def setUp(self):
        if not scorer.GENERAL_VOCAB.exists():
            self.skipTest("general vocab not built")
        self.fam_ids = scorer.family_token_ids()

    def test_family_score_is_mean_over_forms_and_layers(self):
        fam = "verification"
        ids = self.fam_ids[fam]
        exp = _experiment("e", {ids[0]: -2.0, ids[1]: -4.0}, n_layers=3)
        scores = scorer.score_experiment(exp, self.fam_ids)
        # mean over 2 forms x 3 layers of {-2,-4} = -3.0
        self.assertAlmostEqual(scores[fam], -3.0)

    def test_top1_is_the_boosted_family(self):
        boost = "task_resolution"
        logprobs = {}
        for fam, ids in self.fam_ids.items():
            lp = -1.0 if fam == boost else -10.0
            for tid in ids:
                logprobs[tid] = lp
        exp = _experiment("e", logprobs)
        scores = scorer.score_experiment(exp, self.fam_ids)
        self.assertEqual(scorer._top1(scores), boost)

    def test_score_report_and_centering(self):
        def boosted(fam):
            lp = {}
            for f, ids in self.fam_ids.items():
                for tid in ids:
                    lp[tid] = -1.0 if f == fam else -8.0
            return lp

        report = {
            "experiments": [
                _experiment("b1", boosted("source_localization")),
                _experiment("b2", boosted("repair")),
            ]
        }
        p = Path(tempfile.NamedTemporaryFile("w", suffix=".json", delete=False).name)
        p.write_text(json.dumps(report))
        rows = scorer.score_report(p)
        self.assertEqual(rows[0]["top1"], "source_localization")
        self.assertEqual(rows[1]["top1"], "repair")
        # centering present + valid family
        self.assertIn(rows[0]["top1_centered"], self.fam_ids)


if __name__ == "__main__":
    unittest.main()
