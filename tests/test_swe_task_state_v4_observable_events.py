from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_observable_events.py"
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_observable_events.json"

spec = importlib.util.spec_from_file_location("swe_task_state_v4_observable_events", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def skeleton(
    prompt_id,
    request_index,
    *,
    action="inspect",
    tool_outcome="success",
    validation_outcome=None,
    terminal=False,
    materialized=True,
    diagnosis=False,
    next_status="materialized_in_following_request",
    task_id="task",
    repository="repo",
):
    return {
        "prompt_id": prompt_id,
        "source_id_sha256": module.sha256_text(prompt_id),
        "task_id": task_id,
        "task_id_sha256": module.sha256_text(task_id),
        "repository": repository,
        "request_index": request_index,
        "action": action,
        "tool_outcome": tool_outcome,
        "validation_outcome": validation_outcome,
        "terminal": terminal,
        "materialized": materialized,
        "diagnosis": diagnosis,
        "next_status": next_status,
    }


class ObservableEventTests(unittest.TestCase):
    def test_config_is_exact_and_claims_are_false(self):
        config = module.validate_config(module.load_json(CONFIG_PATH))
        self.assertEqual(sum(row["row_count"] for row in config["sources"]), 1708)
        self.assertTrue(all(value is False for value in config["claim_scope"].values()))
        self.assertIn("observable_rationale_language_marker", config["targets"])
        self.assertIn(
            "not a diagnosis, reasoning, understanding, or hidden/private chain-of-thought",
            config["targets"]["observable_rationale_language_marker"]["cot_like_scope"],
        )
        self.assertEqual(
            config["temporal_contract"]["target_completion_offsets"]
            ["milestone_within_2"],
            ["t", "t+1"],
        )
        changed = copy.deepcopy(config)
        changed["claim_scope"] = {}
        with self.assertRaises(module.EventError):
            module.validate_config(changed)

    def test_transition_mapping_is_exact(self):
        expected = {
            ("inspect", "inspect"): "continuation",
            ("inspect", "edit"): "advance",
            ("inspect", "validate"): "advance",
            ("edit", "validate"): "advance",
            ("edit", "inspect"): "rework",
            ("validate", "inspect"): "rework",
            ("validate", "edit"): "rework",
            ("validate", "finalize"): "continuation",
        }
        for pair, value in expected.items():
            self.assertEqual(module._transition(*pair), value)

    def test_unknown_diagnosis_is_not_coerced_to_no_and_future_is_immediate_only(self):
        rows = [
            skeleton("a1", 1, action="inspect", diagnosis=False),
            skeleton("a2", 2, action="edit", tool_outcome="failure", diagnosis=True),
            skeleton(
                "a3",
                3,
                action="finalize",
                tool_outcome=None,
                terminal=True,
                materialized=False,
                diagnosis=None,
                next_status="terminal",
            ),
        ]
        observed = module.derive_rows(rows, unstable_ids={"a2"})
        self.assertEqual(
            observed[0]["targets"]["observable_rationale_language_marker"]["value"],
            "no",
        )
        self.assertEqual(
            observed[1]["targets"]["observable_rationale_language_marker"]["value"],
            "yes",
        )
        self.assertEqual(observed[0]["targets"]["milestone_within_2"]["value"], "edit")
        self.assertEqual(observed[1]["targets"]["transition_kind"]["value"], "advance")
        self.assertFalse(observed[1]["stable_feature_eligible"])
        truncated = skeleton(
            "z1",
            1,
            action=None,
            tool_outcome=None,
            materialized=False,
            diagnosis=None,
            next_status="truncated",
        )
        unknown = module.derive_rows([truncated], unstable_ids=set())[0]
        marker = unknown["targets"]["observable_rationale_language_marker"]
        self.assertEqual(marker["status"], "unknown")
        self.assertIsNone(marker["value"])
        self.assertEqual(unknown["targets"]["terminal_event"]["value"], "no")
        unobserved = dict(truncated)
        unobserved["prompt_id"] = "z2"
        unobserved["source_id_sha256"] = module.sha256_text("z2")
        unobserved["next_status"] = "unobserved_after_task_end"
        terminal_marker = module.derive_rows([unobserved], unstable_ids=set())[0][
            "targets"
        ]["terminal_event"]
        self.assertEqual(terminal_marker["status"], "unknown")

    def test_t_plus_2_mutation_cannot_change_row_t(self):
        rows = [
            skeleton("a1", 1, action="inspect"),
            skeleton("a2", 2, action="inspect"),
            skeleton("a3", 3, action="edit", diagnosis=True),
            skeleton(
                "a4",
                4,
                action="finalize",
                tool_outcome=None,
                terminal=True,
                materialized=False,
                diagnosis=None,
                next_status="terminal",
            ),
        ]
        changed = copy.deepcopy(rows)
        changed[2].update(
            action="validate", validation_outcome="failure", diagnosis=False
        )
        first = module.derive_rows(rows, unstable_ids=set())[0]
        second = module.derive_rows(changed, unstable_ids=set())[0]
        self.assertEqual(first, second)

    def test_history_is_derived_by_repository_task_and_request_index(self):
        rows = [
            skeleton("a2", 2, action="inspect"),
            skeleton("a1", 1, action="edit"),
        ]
        observed = module.derive_rows(rows, unstable_ids=set())
        self.assertEqual(observed[0]["positive_controls"]["has_prior_edit"], "yes")
        self.assertEqual(observed[0]["targets"]["transition_kind"]["value"], "rework")

    def test_malformed_completion_state_and_cross_repo_task_are_rejected(self):
        malformed = skeleton("a1", 1, terminal=True)
        with self.assertRaises(module.EventError):
            module.derive_rows([malformed], unstable_ids=set())
        first = skeleton("a1", 1, task_id="shared", repository="repo-a")
        second = skeleton("a2", 1, task_id="shared", repository="repo-b")
        with self.assertRaises(module.EventError):
            module.derive_rows([first, second], unstable_ids=set())

    def test_label_mutation_does_not_change_grouping_or_identity_fields(self):
        row = skeleton("a", 1)
        changed = copy.deepcopy(row)
        changed.update(
            action="validate",
            tool_outcome="failure",
            validation_outcome="failure",
            diagnosis=True,
        )
        first = module.derive_rows([row], unstable_ids=set())[0]
        second = module.derive_rows([changed], unstable_ids=set())[0]
        for key in (
            "global_index",
            "source_id_sha256",
            "task_id_sha256",
            "repository",
            "request_index",
            "stable_feature_eligible",
        ):
            self.assertEqual(first[key], second[key])

    def test_label_free_index_contains_no_targets_or_controls(self):
        sidecar = {
            "config": {"sha256": "a" * 64},
            "implementation": {"sha256": "b" * 64},
            "sources": [],
            "eligibility_source": {},
            "rows": [
                {
                    "global_index": 0,
                    "source_id_sha256": "c" * 64,
                    "task_id_sha256": "d" * 64,
                    "repository": "repo",
                    "request_index": 1,
                    "stable_feature_eligible": True,
                    "targets": {"observable_rationale_language_marker": {"value": "yes"}},
                    "positive_controls": {"has_prior_edit": "no"},
                }
            ],
        }
        index = module.build_label_free_index(sidecar)
        self.assertEqual(index["kind"], module.INDEX_KIND)
        rendered = str(index)
        self.assertNotIn("observable_rationale_language_marker", rendered)
        self.assertNotIn("has_prior_edit", rendered)
        self.assertNotIn("targets", index["rows"][0])


if __name__ == "__main__":
    unittest.main()
