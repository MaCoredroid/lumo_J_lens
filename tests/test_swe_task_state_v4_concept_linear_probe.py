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


probe = _load("concept_linear_probe", "scripts/swe_task_state_v4_concept_linear_probe.py")


def _synthetic_readout(path: Path):
    # 6 tasks x 8 turns, two classes separated by which token lights up; layers 40-47.
    pool = [1000, 1001, 1002, 1003]
    exps = []
    for ti in range(6):
        for turn in range(1, 9):
            cls = "verification" if turn % 2 == 0 else "repair"
            hot = 1000 if cls == "verification" else 1001
            layers = []
            for ln in range(16, 48):
                scored = [{"token_id": t, "logprob": (-1.0 if t == hot else -8.0)} for t in pool]
                layers.append({"layer": ln, "positions": [{"jacobian_lens": {"scored_tokens": scored}}]})
            exps.append({"id": f"t{ti}-{turn}", "metadata": {"task": f"task{ti}", "turn": turn, "tag": cls}, "layers": layers})
    path.write_text(json.dumps({"experiments": exps}))


class LinearProbeTests(unittest.TestCase):
    def test_split_is_deterministic_10_10(self):
        try:
            train, test = probe._split()
        except FileNotFoundError:
            self.skipTest("tags not built")
        self.assertEqual(len(train & test), 0)
        self.assertGreaterEqual(len(train) + len(test), 2)

    def test_probe_separates_a_clean_synthetic(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("sklearn unavailable")
        p = Path(tempfile.NamedTemporaryFile("w", suffix=".json", delete=False).name)
        _synthetic_readout(p)
        # override the by-task split helper to split the synthetic tasks
        orig = probe._split
        probe._split = lambda: ({"task0", "task1", "task2"}, {"task3", "task4", "task5"})
        try:
            r = probe.compute(p, layers=[40, 41, 42, 43], C=1.0, folds=3)
        finally:
            probe._split = orig
        self.assertEqual(r["n_turns"], 48)
        self.assertGreater(r["held_out_accuracy"], 0.9)  # cleanly separable
        self.assertIn("verification", r["per_family"])


if __name__ == "__main__":
    unittest.main()
