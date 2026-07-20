from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_counterfactual_generate.py"
SPEC = importlib.util.spec_from_file_location(
    "swe_task_state_v4_counterfactual_generate", MODULE_PATH
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def blinded_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(12):
        count = 12694 if index % 2 else 12697
        rows.append({"id": f"{index + 1:064x}", "token_ids": [1] * count})
    return rows


class CounterfactualGenerateTests(unittest.TestCase):
    def test_capture_bundle_is_exactly_blinded(self) -> None:
        observed = module.validate_capture_bundle(blinded_rows())
        self.assertEqual(len(observed), 12)
        self.assertTrue(all(set(row) == {"id", "token_ids"} for row in observed))

    def test_capture_bundle_rejects_condition_field(self) -> None:
        rows = blinded_rows()
        rows[0]["evidence_level"] = "clear_success"
        with self.assertRaisesRegex(module.GenerationError, "not blinded"):
            module.validate_capture_bundle(rows)

    def test_materialization_requires_pre_generation_state(self) -> None:
        path = Path("capture-prompts.json")
        value = {
            "schema_version": 1,
            "kind": "swe_task_state_v4_counterfactual_pilot_materialization",
            "status": "passed_materialization_only_capture_and_generation_not_started",
            "outputs": {
                "capture_bundle": {
                    "sha256": "a" * 64,
                    "size_bytes": 7,
                    "prompt_count": 12,
                    "allowed_row_keys": ["id", "token_ids"],
                }
            },
            "execution_state": {
                "counterfactual_completion_generation_completed": False,
                "completion_label_extraction_completed": False,
            },
        }
        with mock.patch.object(Path, "stat") as stat:
            stat.return_value.st_size = 7
            module.validate_materialization(value, capture_path=path, capture_sha256="a" * 64)
        value["execution_state"]["completion_label_extraction_completed"] = True
        with mock.patch.object(Path, "stat") as stat:
            stat.return_value.st_size = 7
            with self.assertRaisesRegex(module.GenerationError, "pre-generation"):
                module.validate_materialization(value, capture_path=path, capture_sha256="a" * 64)

    def test_forbidden_paths_fail_before_access(self) -> None:
        with self.assertRaisesRegex(module.GenerationError, "before filesystem access"):
            module.lexical_path_preflight((Path("/tmp/validation-hidden/input.json"),))


if __name__ == "__main__":
    unittest.main()
