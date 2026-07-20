from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_counterfactual_selection.py"
SPEC = importlib.util.spec_from_file_location(
    "swe_task_state_v4_counterfactual_selection", MODULE_PATH
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def synthetic_rows() -> list[dict[str, object]]:
    rows = []
    global_index = 0
    for task_index in range(60):
        repository = f"repo-{task_index % 10}"
        task = f"{task_index + 1:064x}"
        for request_index in range(1, 7):
            rows.append(
                {
                    "global_index": global_index,
                    "source_id_sha256": f"{global_index + 1000:064x}",
                    "task_id_sha256": task,
                    "repository": repository,
                    "request_index": request_index,
                    "stable_feature_eligible": True,
                }
            )
            global_index += 1
    return rows


class CounterfactualSelectionTests(unittest.TestCase):
    def test_config_separates_observable_proxies_from_subjective_claims(self) -> None:
        self.assertEqual(
            module.sha256_file(module.CONFIG_PATH),
            module.CONFIG_SHA256,
        )
        config = module.validate_config(module.load_json(module.CONFIG_PATH))
        self.assertEqual(config["factorial_design"]["expected_prompt_count"], 1440)
        capture = config["capture_and_generation"]
        self.assertTrue(capture["raw_residual_and_public_j_state_capture_required"])
        self.assertTrue(
            capture["activation_capture_must_precede_completion_label_extraction"]
        )
        self.assertNotIn("raw_residual_and_public_j_state_captured", capture)
        self.assertNotIn("activation_capture_completed_before_completion_label_extraction", capture)
        claims = config["claim_scope"]
        self.assertFalse(claims["private_chain_of_thought_reconstructed"])
        self.assertFalse(claims["subjective_confidence_or_doubt_inferred"])
        self.assertFalse(claims["experienced_stress_inferred"])
        self.assertFalse(claims["experienced_emotion_inferred"])

    def test_selection_uses_fixed_internal_quantiles_and_no_labels(self) -> None:
        rows = synthetic_rows()
        selected = module.select_boundaries(rows)
        self.assertEqual(len(selected), 120)
        by_task: dict[str, list[dict[str, object]]] = {}
        for row in selected:
            by_task.setdefault(str(row["task_id_sha256"]), []).append(row)
        self.assertEqual(len(by_task), 60)
        for task_rows in by_task.values():
            self.assertEqual(
                [row["request_index"] for row in task_rows],
                [2, 4],
            )
            self.assertTrue(
                all(row["selection_pool"] == "stable_internal_requests" for row in task_rows)
            )

    def test_selection_is_invariant_to_unread_extra_labels(self) -> None:
        rows = synthetic_rows()
        changed = [dict(row, label="future", outcome="success") for row in rows]
        self.assertEqual(
            module.select_boundaries(rows),
            module.select_boundaries(changed),
        )

    def test_assignment_is_complete_deterministic_and_unique(self) -> None:
        selected = module.select_boundaries(synthetic_rows())
        left = module.assign_conditions(selected, seed=20260720)
        right = module.assign_conditions(selected, seed=20260720)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 1440)
        self.assertEqual(len({row["condition_id_sha256"] for row in left}), 1440)
        first_source = selected[0]["source_id_sha256"]
        block = [row for row in left if row["source_id_sha256"] == first_source]
        self.assertEqual(len(block), 12)
        self.assertEqual(
            {
                (row["evidence_level"], row["pressure_level"], row["paraphrase_replica"])
                for row in block
            },
            {
                (evidence, pressure, replica)
                for evidence in module.EVIDENCE_LEVELS
                for pressure in module.PRESSURE_LEVELS
                for replica in (0, 1)
            },
        )

    def test_forbidden_paths_fail_before_access(self) -> None:
        with self.assertRaisesRegex(module.SelectionError, "before filesystem access"):
            module.lexical_path_preflight((Path("/tmp/reserved-state/input.json"),))

    def test_assignment_manifest_marks_execution_not_started(self) -> None:
        config = module.validate_config(module.load_json(module.CONFIG_PATH))
        selected = [{} for _ in range(120)]
        assignments = [
            {"condition_id_sha256": f"{index:064x}"} for index in range(1440)
        ]
        with (
            mock.patch.object(module, "validate_alignment", return_value=[]),
            mock.patch.object(module, "select_boundaries", return_value=selected),
            mock.patch.object(module, "assign_conditions", return_value=assignments),
            mock.patch.object(
                module,
                "_artifact",
                return_value={"path": "fixture", "sha256": "0" * 64, "size_bytes": 1},
            ),
        ):
            manifest = module.build_manifest(
                config=config,
                alignment_path=Path("fixture.json"),
                alignment={},
            )
        self.assertEqual(
            manifest["execution_state"],
            {
                "generation_completed": False,
                "activation_capture_completed": False,
                "completion_label_extraction_completed": False,
                "capture_requirement_is_prospective_not_a_completed_action": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
