from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_activation_features.py"
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_activation_features.json"

spec = importlib.util.spec_from_file_location("swe_task_state_v4_activation_features", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class ActivationFeatureTests(unittest.TestCase):
    def test_config_is_exact_and_claims_false(self):
        config = module.validate_config(module.load_json(CONFIG_PATH))
        self.assertEqual(config["projection"]["primary_seed_index"], 0)
        self.assertEqual(config["variants"]["raw_public_j_activation_sequence"], 1154)
        self.assertTrue(all(value is False for value in config["claim_scope"].values()))

    def test_temporal_features_are_preupdate_and_unstable_rows_do_not_update(self):
        rows = []
        for index in range(1708):
            rows.append(
                {
                    "global_index": index,
                    "source_id_sha256": f"{index:064x}",
                    "task_id_sha256": "1" * 64,
                    "repository": "repo",
                    "request_index": index + 1,
                    "stable_feature_eligible": index < 2 or index >= 104,
                }
            )
        # Exactly 102 unstable rows: indices 2..103.  The next stable row must
        # compare against index 1, not any skipped row.
        raw = np.repeat(np.arange(1708, dtype=np.float64)[:, None], 192, axis=1)
        public_j = 2 * raw
        observed = module.build_causal_activation_features(
            alignment_rows=rows, raw_current=raw, public_j_current=public_j
        )
        self.assertEqual(observed["raw_activation_sequence"].shape, (1606, 578))
        first = observed["raw_activation_sequence"][0]
        second = observed["raw_activation_sequence"][1]
        after_gap = observed["raw_activation_sequence"][2]
        np.testing.assert_array_equal(first[192:576], np.zeros(384))
        np.testing.assert_array_equal(second[192:384], np.ones(192))
        np.testing.assert_array_equal(after_gap[192:384], np.full(192, 103.0))
        self.assertEqual(first[-1], 1.0)
        self.assertEqual(second[-1], 0.0)
        self.assertAlmostEqual(after_gap[-2], np.log1p(103))

    def test_future_mutation_cannot_change_earlier_features_and_tasks_are_isolated(self):
        rows = []
        for index in range(1708):
            task = "1" * 64 if index % 2 == 0 else "2" * 64
            rows.append(
                {
                    "global_index": index,
                    "source_id_sha256": f"{index:064x}",
                    "task_id_sha256": task,
                    "repository": "repo",
                    "request_index": index // 2 + 1,
                    "stable_feature_eligible": index < 1606,
                }
            )
        raw = np.zeros((1708, 192), dtype=np.float64)
        public_j = np.zeros_like(raw)
        raw[:, 0] = np.arange(1708)
        public_j[:, 0] = -np.arange(1708)
        first = module.build_causal_activation_features(
            alignment_rows=rows, raw_current=raw, public_j_current=public_j
        )
        changed_raw = raw.copy()
        changed_raw[1000:] += 9999
        second = module.build_causal_activation_features(
            alignment_rows=rows, raw_current=changed_raw, public_j_current=public_j
        )
        np.testing.assert_array_equal(
            first["raw_activation_sequence"][:1000],
            second["raw_activation_sequence"][:1000],
        )
        # Index 2 is the second row for task 1 and its delta is 2, not 1.
        self.assertEqual(first["raw_activation_sequence"][2, 192], 2.0)

    def test_api_has_no_label_argument(self):
        annotations = module.build_causal_activation_features.__annotations__
        self.assertNotIn("labels", annotations)
        self.assertNotIn("targets", annotations)
        self.assertNotIn("outcomes", annotations)

    def test_alignment_rejects_cross_repository_task_and_request_gaps(self):
        rows = [
            {
                "global_index": index,
                "source_id_sha256": f"{index:064x}",
                "task_id_sha256": "1" * 64,
                "repository": "repo-a",
                "request_index": index + 1,
                "stable_feature_eligible": index < 1606,
            }
            for index in range(1708)
        ]
        index = {
            "schema_version": 1,
            "kind": "swe_task_state_v4_label_free_alignment_index",
            "status": "passed",
            "scope": "grouping_order_and_stability_only_no_labels",
            "config": {},
            "implementation": {},
            "sources": [],
            "eligibility_source": {},
            "row_count": 1708,
            "stable_row_count": 1606,
            "feature_use": {},
            "rows": rows,
        }
        changed_repo = {**index, "rows": [dict(row) for row in rows]}
        changed_repo["rows"][1]["repository"] = "repo-b"
        with self.assertRaises(module.FeatureError):
            module.validate_alignment_index(changed_repo)
        gapped = {**index, "rows": [dict(row) for row in rows]}
        for row in gapped["rows"][1:]:
            row["request_index"] += 1
        with self.assertRaises(module.FeatureError):
            module.validate_alignment_index(gapped)


if __name__ == "__main__":
    unittest.main()
