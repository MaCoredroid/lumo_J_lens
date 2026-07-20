from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
MODULE_PATH = SCRIPTS / "swe_task_state_v4_counterfactual_pilot_analyze.py"
SPEC = importlib.util.spec_from_file_location(
    "swe_task_state_v4_counterfactual_pilot_analyze", MODULE_PATH
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class CounterfactualPilotAnalyzeTests(unittest.TestCase):
    def test_markers_keep_objective_behavior_separate_from_affect(self) -> None:
        markers = module.extract_blinded_markers(
            "Given the mixed diagnostic evidence, either path A or path B remains possible. "
            "Therefore, the next step should recheck and inspect the test output.",
            generated_token_count=31,
        )
        self.assertTrue(markers["explicit_uncertainty_language"])
        self.assertTrue(markers["recheck_language"])
        self.assertTrue(markers["multiple_alternatives_language"])
        self.assertTrue(markers["doubt_like_behavior_proxy"])
        self.assertTrue(markers["visible_chain_language_proxy"])
        self.assertFalse(markers["explicit_affect_or_stress_language"])

    def test_explicit_affect_is_language_not_experience(self) -> None:
        markers = module.extract_blinded_markers(
            "The deadline feels stressful, but I will run the tests immediately.",
            generated_token_count=14,
        )
        self.assertTrue(markers["explicit_affect_or_stress_language"])
        self.assertTrue(markers["explicit_pressure_language"])
        self.assertTrue(markers["advance_or_validate_language"])

    def test_objective_action_proxy_depends_on_assigned_evidence(self) -> None:
        base = {name: False for name in module.MARKER_PATTERNS}
        base["doubt_like_behavior_proxy"] = False
        success = dict(base, advance_or_validate_language=True)
        failure = dict(base, rework_or_targeted_inspection_language=True)
        ambiguous = dict(
            base,
            doubt_like_behavior_proxy=True,
            explicit_uncertainty_language=True,
        )
        self.assertTrue(
            module.objective_behavior_proxy(success, "clear_success")[
                "objective_action_appropriateness_rule_proxy"
            ]
        )
        self.assertTrue(
            module.objective_behavior_proxy(failure, "clear_failure")[
                "objective_action_appropriateness_rule_proxy"
            ]
        )
        ambiguous_result = module.objective_behavior_proxy(
            ambiguous, "contradictory_ambiguous"
        )
        self.assertTrue(ambiguous_result["objective_action_appropriateness_rule_proxy"])
        self.assertTrue(
            ambiguous_result["objective_certainty_action_calibration_rule_proxy"]
        )

    def test_activation_distance_is_zero_only_for_equal_states(self) -> None:
        left = np.ones((24, 5120), dtype=np.float32)
        same = module._distance(left, left.copy())
        changed = left.copy()
        changed[3, 9] = 2.0
        different = module._distance(changed, left)
        self.assertEqual(same["mean_layer_rms_difference"], 0.0)
        self.assertAlmostEqual(same["mean_layer_cosine_distance"], 0.0, places=12)
        self.assertGreater(different["mean_layer_rms_difference"], 0.0)
        self.assertGreater(different["mean_layer_cosine_distance"], 0.0)

    def test_condition_key_must_join_exactly(self) -> None:
        ids = {f"{index + 1:064x}" for index in range(12)}
        records = []
        for index, prompt_id in enumerate(sorted(ids)):
            records.append(
                {
                    "prompt_id": prompt_id,
                    "evidence_level": module.generation.EVIDENCE_LEVELS[index % 3],
                    "pressure_level": module.generation.PRESSURE_LEVELS[index % 2],
                    "paraphrase_replica": index % 2,
                }
            )
        value = {
            "schema_version": 1,
            "kind": "swe_task_state_v4_counterfactual_pilot_condition_key",
            "status": "assignment_only_completion_labels_not_extracted",
            "capture_and_activation_feature_process_must_not_read_this_file": True,
            "completion_label_extraction_completed": False,
            "subjective_state_labels_present": False,
            "records": records,
        }
        self.assertEqual(set(module.validate_condition_key(value, ids)), ids)
        with self.assertRaisesRegex(module.AnalysisError, "exactly join"):
            module.validate_condition_key(value, ids | {"f" * 64})


if __name__ == "__main__":
    unittest.main()
