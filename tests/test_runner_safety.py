import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_swe_verified import (  # noqa: E402
    _container_image_for,
    _prediction_completeness_exit_code,
    _safe_task_dir,
    _validated_commit,
)


ROOT = Path(__file__).resolve().parents[1]


class RunnerSafetyTests(unittest.TestCase):
    def test_official_instance_id_stays_below_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tasks"
            self.assertEqual(_safe_task_dir(root, "sympy__sympy-13480").parent, root.resolve())

    def test_path_traversal_instance_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _safe_task_dir(Path(tmp), "../../outside")

    def test_absolute_instance_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _safe_task_dir(Path(tmp), "/tmp/other")

    def test_sha_commit_is_accepted(self):
        commit = "f57fe3f4b3f2cab225749e1b3b38ae1bf80b62f0"
        self.assertEqual(_validated_commit(commit), commit)

    def test_shell_expression_commit_is_rejected(self):
        with self.assertRaises(ValueError):
            _validated_commit("HEAD; touch /tmp/bad")

    def test_verified_launcher_keeps_endpoint_and_proxy_context_limits_equal(self):
        launcher = (ROOT / "scripts" / "run_verified_task.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}", launcher)
        self.assertIn('--max-model-len "$MAX_MODEL_LEN"', launcher)
        self.assertIn('--proxy-context-limit "$MAX_MODEL_LEN"', launcher)

    def test_submission_campaign_rejects_missing_predictions(self):
        self.assertEqual(
            _prediction_completeness_exit_code(
                ["sympy__sympy-13480"], allow_empty_predictions=False
            ),
            2,
        )

    def test_behavioral_campaign_retains_missing_predictions(self):
        self.assertEqual(
            _prediction_completeness_exit_code(
                ["sympy__sympy-13480"], allow_empty_predictions=True
            ),
            0,
        )

    def test_behavioral_launcher_uses_one_context_limit(self):
        launcher = (ROOT / "scripts" / "run_swe_behavioral_campaign.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}", launcher)
        self.assertIn('--max-model-len "$MAX_MODEL_LEN"', launcher)
        self.assertIn('--proxy-context-limit "$MAX_MODEL_LEN"', launcher)
        self.assertIn("--allow-empty-predictions", launcher)

    def test_runner_accepts_a_separate_pinned_image_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "images.json"
            reference = "example.invalid/sympy@sha256:" + "a" * 64
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "images": {
                            "sympy__sympy-13480": {
                                "x86_64": {
                                    "reference": reference,
                                    "image_id": "sha256:" + "a" * 64,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ, {"SWE_IMAGE_CONFIG": str(registry)}, clear=False
            ):
                self.assertEqual(_container_image_for("sympy__sympy-13480"), reference)
