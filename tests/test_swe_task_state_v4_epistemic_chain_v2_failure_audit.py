import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "swe_task_state_v4_epistemic_chain_v2_failure_audit.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EpistemicChainV2FailureAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_immutable_run_bindings_match(self):
        run = self.config["immutable_run"]
        for key in ("final_gate_receipt", "dual_lock_receipt"):
            binding = run[key]
            self.assertEqual(sha256_file(ROOT / binding["path"]), binding["sha256"])
        for role in run["roles"].values():
            self.assertEqual(sha256_file(ROOT / role["manifest_path"]), role["manifest_sha256"])
            self.assertEqual(sha256_file(ROOT / role["records_path"]), role["records_sha256"])
        for report in run["reports"].values():
            self.assertEqual(sha256_file(ROOT / report["path"]), report["sha256"])

    def test_corrected_invalid_accounting(self):
        counts = self.config["corrected_counts"]
        self.assertEqual(counts["independent_a"]["own_invalid_total"], 30)
        self.assertEqual(counts["independent_b"]["own_invalid_total"], 10)
        self.assertEqual(counts["adjudicator"]["invalid"], 12)
        self.assertEqual(
            counts["independent_a"]["completion_invalid"] + counts["adjudicator"]["invalid"],
            34,
        )
        self.assertEqual(
            counts["independent_b"]["completion_invalid"] + counts["adjudicator"]["invalid"],
            22,
        )

    def test_v3_remediation_removes_observed_interface_failures(self):
        requirements = self.config["prospective_v3_requirements"]
        for key in (
            "decision_specific_one_of_schemas",
            "model_generated_quote_strings_forbidden",
            "novelty_validity_included_in_invalid_output_gate",
            "task_shaped_per_family_preflight_before_sealing",
            "real_trajectory_selection_annotation_and_decoder_fit_forbidden_until_v3_gate_passes",
        ):
            self.assertIs(requirements[key], True)
        self.assertEqual(
            {row["classification"] for row in self.config["root_causes"]},
            {
                "interface_design_failure",
                "adjudication_interface_and_runtime_compatibility_failure",
                "scoring_diagnostic_failure",
                "model_semantic_error",
            },
        )

    def test_all_scientific_claims_remain_false(self):
        self.assertTrue(all(value is False for value in self.config["claim_boundary"].values()))
        self.assertIs(self.config["scope"]["reserved_validation_closed"], True)
        self.assertIs(self.config["scope"]["reserved_validation_accessed"], False)


if __name__ == "__main__":
    unittest.main()
