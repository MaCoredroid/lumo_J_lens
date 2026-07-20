from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_counterfactual_pilot_capture_receipt.py"
SPEC = importlib.util.spec_from_file_location(
    "swe_task_state_v4_counterfactual_pilot_capture_receipt", MODULE_PATH
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def exact_split() -> dict[str, object]:
    return {
        "capture_elapsed_seconds": 41.260575,
        "final_logits_tolerance": "failed_rows_1_3_10_max_abs_error_0.125",
        "final_norm_tolerance": "passed_12_of_12",
        "raw_public_j_pre_vocabulary_capture_status": "passed_12_of_12",
        "raw_public_j_tensor_claims_limited_to_capture_contract": True,
        "reference_and_fresh_failure_pattern_exactly_equal": True,
        "reference_elapsed_seconds": 49.399777,
        "reference_residual_manifest_equality": "passed_12_of_12",
        "safetensors_reload_verification": "passed_12_of_12",
        "top1_greedy_match": "passed_12_of_12",
        "vocabulary_adapter_failed_rows": [1, 3, 10],
        "vocabulary_adapter_status": "failed_3_of_12",
        "vocabulary_level_claims_permitted": False,
    }


class CounterfactualPilotCaptureReceiptTests(unittest.TestCase):
    def test_exact_split_is_accepted_without_promoting_vocabulary_claim(self) -> None:
        observed = module.validate_capture_split(exact_split())
        self.assertEqual(
            observed["raw_public_j_pre_vocabulary_capture_status"],
            "passed_12_of_12",
        )
        self.assertEqual(observed["vocabulary_adapter_status"], "failed_3_of_12")
        self.assertFalse(observed["vocabulary_level_claims_permitted"])

    def test_split_fails_closed_if_vocabulary_failure_is_hidden(self) -> None:
        changed = exact_split()
        changed["vocabulary_adapter_status"] = "passed_12_of_12"
        changed["vocabulary_adapter_failed_rows"] = []
        with self.assertRaisesRegex(module.analysis.AnalysisError, "capture split changed"):
            module.validate_capture_split(changed)

    def test_split_fails_closed_if_exact_failure_rows_or_error_change(self) -> None:
        for key, value in (
            ("vocabulary_adapter_failed_rows", [1, 10]),
            ("final_logits_tolerance", "failed_rows_1_10_max_abs_error_0.125"),
            ("reference_and_fresh_failure_pattern_exactly_equal", False),
        ):
            with self.subTest(key=key):
                changed = exact_split()
                changed[key] = value
                with self.assertRaisesRegex(
                    module.analysis.AnalysisError, "capture split changed"
                ):
                    module.validate_capture_split(changed)

    def test_input_hash_mismatch_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cf-receipt-") as directory:
            root = Path(directory)
            capture_prompts = root / "capture-prompts.json"
            materialization = root / "materialization.json"
            capture_manifest = root / "capture-manifest.json"
            output = root / "receipt.json"
            for path in (capture_prompts, materialization, capture_manifest):
                path.write_text("{}\n", encoding="utf-8")
            args = argparse.Namespace(
                capture_prompts=capture_prompts,
                capture_prompts_sha256="0" * 64,
                materialization_manifest=materialization,
                materialization_manifest_sha256="0" * 64,
                capture_manifest=capture_manifest,
                capture_manifest_sha256="0" * 64,
                output=output,
            )
            with self.assertRaisesRegex(module.analysis.AnalysisError, "bytes changed"):
                module.run(args)
            self.assertFalse(output.exists())

    def test_forbidden_paths_fail_before_access(self) -> None:
        with self.assertRaisesRegex(module.analysis.AnalysisError, "before filesystem access"):
            module.lexical_path_preflight((Path("/tmp/reserved-receipt/input.json"),))


if __name__ == "__main__":
    unittest.main()
