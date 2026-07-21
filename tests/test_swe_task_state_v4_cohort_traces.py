from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "cohort_traces", ROOT / "scripts/swe_task_state_v4_cohort_traces.py"
)
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)


def _write(text: str) -> Path:
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    fh.write(text)
    fh.close()
    return Path(fh.name)


class FormatRobustnessTests(unittest.TestCase):
    def test_json_list(self):
        p = _write(json.dumps([{"type": "assistant"}, {"type": "user"}]))
        self.assertEqual(len(ct.read_trace_entries(p)), 2)

    def test_jsonl(self):
        p = _write('{"type":"assistant"}\n{"type":"user"}\n')
        self.assertEqual(len(ct.read_trace_entries(p)), 2)

    def test_empty_and_garbage_yield_nothing(self):
        self.assertEqual(ct.read_trace_entries(_write("")), [])
        self.assertEqual(ct.read_trace_entries(_write("not json\n")), [])

    def test_thinking_extraction_both_shapes(self):
        entry = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "diagnose the bug"}],
            },
        }
        p = _write(json.dumps([entry]))
        self.assertEqual(ct.task_thinking_blocks(p), ["diagnose the bug"])


class CohortSurveyTests(unittest.TestCase):
    def test_survey_finds_usable_tasks(self):
        if not ct.RUNS.exists():
            self.skipTest("runs/ not present")
        s = ct.survey()
        self.assertGreater(s["n_usable_tasks"], 1)  # a real cohort, not n=1
        self.assertGreater(s["total_usable_turns"], s["n_usable_tasks"])
        for row in s["usable"]:
            self.assertGreater(row["n_turns"], 0)


if __name__ == "__main__":
    unittest.main()
