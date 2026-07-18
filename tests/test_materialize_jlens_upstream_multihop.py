#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("materialize_jlens_upstream_multihop", ROOT / "scripts/materialize_jlens_upstream_multihop.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    def __len__(self):
        return 100

    def encode(self, text, *, add_special_tokens):
        values = {"Prompt ": [1, 2], "Alpha": [10], " Alpha": [11], "12": [1, 2], " 12": [3, 1, 2]}
        return values[text]

    def decode(self, token_ids, **_kwargs):
        values = {(1, 2): "Prompt ", (10,): "Alpha", (11,): " Alpha"}
        return values.get(tuple(token_ids), "not-the-input")


class MaterializeUpstreamMultihopTest(unittest.TestCase):
    def test_forms_keep_exact_single_tokens_and_record_exclusions(self):
        eligible = MODULE.verified_forms(FakeTokenizer(), "Alpha")
        self.assertTrue(eligible["scorable"])
        self.assertEqual([row["token_id"] for row in eligible["eligible_forms"]], [10, 11])
        excluded = MODULE.verified_forms(FakeTokenizer(), "12")
        self.assertFalse(excluded["scorable"])
        self.assertEqual(len(excluded["excluded_forms"]), 2)
        self.assertTrue(all(row["reason"] == "not_exactly_one_token" for row in excluded["excluded_forms"]))

    def test_bundle_preserves_upstream_denominator_and_union(self):
        source = {"items": [{"name": "item", "prompt": "Prompt ", "target": "Answer", "intermediates": ["Alpha", "12"]}]}
        prompts, manifest = MODULE.build_bundle(source, FakeTokenizer())
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]["token_ids"], [1, 2])
        self.assertEqual(prompts[0]["score_token_ids"], [10, 11])
        self.assertEqual(manifest["coverage"]["intermediate_occurrence_count"], 2)
        self.assertEqual(manifest["coverage"]["scorable_intermediate_occurrence_count"], 1)
        self.assertEqual(manifest["coverage"]["excluded_intermediate_occurrence_count"], 1)
        self.assertEqual(manifest["scored_vocabulary"]["token_ids"], [10, 11])
        self.assertIn("count_as_miss", manifest["metric_contract"]["unscorable_intermediate_policy"])

    def test_fully_excluded_prompt_omits_empty_runner_vocabulary(self):
        source = {"items": [{"name": "item", "prompt": "Prompt ", "target": "Answer", "intermediates": ["12"]}]}
        prompts, manifest = MODULE.build_bundle(source, FakeTokenizer())
        self.assertNotIn("score_token_ids", prompts[0])
        self.assertEqual(manifest["scored_vocabulary"]["token_ids"], [])

    def test_cli_requires_one_portable_source_mode(self):
        with self.assertRaisesRegex(ValueError, "select exactly one"):
            MODULE.main(["--output-dir", "/tmp/not-used"])


if __name__ == "__main__":
    unittest.main()
