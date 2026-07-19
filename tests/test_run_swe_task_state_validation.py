from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/run_swe_task_state_validation.sh"


def write_executable(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body).lstrip(), encoding="ascii")
    path.chmod(0o755)


def assignment(source: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}=([^\n]+)$", source, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing shell assignment: {name}")
    return match.group(1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TaskStateValidationWrapperTests(unittest.TestCase):
    def test_defaults_to_dense_materializer_adapter(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn(
            "$ROOT/scripts/materialize_swe_dense_behavioral_probes.py", source
        )

    def test_core_helpers_and_configs_are_canonical_and_content_pinned(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        pinned = {
            "CHECKER_SHA256": ROOT
            / "scripts/check_swe_task_state_validation_cohort.py",
            "CAMPAIGN_RUNNER_SHA256": ROOT
            / "scripts/run_swe_behavioral_campaign.sh",
            "MATERIALIZER_SHA256": ROOT
            / "scripts/materialize_swe_dense_behavioral_probes.py",
            "HISTORICAL_MATERIALIZER_SHA256": ROOT
            / "scripts/materialize_swe_behavioral_probes.py",
            "COHORT_MANIFEST_SHA256": ROOT
            / "configs/swe_task_state_validation_cohort.json",
            "IMAGE_CONFIG_SHA256": ROOT
            / "configs/swe_task_state_validation_image_digests.json",
            "CAMPAIGN_A_SHA256": ROOT
            / "configs/swe_task_state_validation_a_campaign.json",
            "CAMPAIGN_B_SHA256": ROOT
            / "configs/swe_task_state_validation_b_campaign.json",
            "ACTION_PROTOCOL_SHA256": ROOT / "configs/swe_stage_action_probes.json",
            "CHAT_TEMPLATE_SHA256": ROOT / "configs/qwen3-openai-codex.jinja",
        }
        for variable, path in pinned.items():
            with self.subTest(variable=variable):
                self.assertEqual(assignment(source, variable), sha256(path))
        for variable in (
            "CHECKER",
            "CAMPAIGN_RUNNER",
            "MATERIALIZER",
            "COHORT_MANIFEST",
            "IMAGE_CONFIG",
            "CAMPAIGN_A",
            "CAMPAIGN_B",
        ):
            with self.subTest(variable=variable):
                self.assertNotIn(f"{variable}=${{{variable}:-", source)
        self.assertIn('--validate-bundle "$PROMPTS_OUTPUT" "$SUMMARY_OUTPUT"', source)

    def test_identity_environment_drift_is_ignored_in_check_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            call_log = temporary / "calls.log"
            fake_python = temporary / "fake-python"
            write_executable(
                fake_python,
                r"""
                #!/usr/bin/env bash
                set -euo pipefail
                printf '%s\n' "$*" >>"$CALL_LOG"
                """,
            )
            environment = {
                **os.environ,
                "CALL_LOG": str(call_log),
                "SWE_PYTHON": str(fake_python),
                "VLLM_PYTHON": str(fake_python),
                "CHECKER": str(temporary / "checker.py"),
                "CAMPAIGN_RUNNER": str(temporary / "runner"),
                "MATERIALIZER": str(temporary / "materializer.py"),
                "COHORT_MANIFEST": str(temporary / "cohort.json"),
                "IMAGE_CONFIG": str(temporary / "images.json"),
                "CAMPAIGN_A": str(temporary / "drift-a.json"),
                "CAMPAIGN_B": str(temporary / "drift-b.json"),
                "RUN_NAME_A": "drift-a",
                "RUN_NAME_B": "drift-b",
                "PORT_A": "19052",
                "PORT_B": "19052",
                "PROXY_PORT_A": "31052",
                "PROXY_PORT_B": "31053",
            }

            result = subprocess.run(
                [str(WRAPPER), "--check-only"],
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            call = call_log.read_text(encoding="ascii").strip()
            self.assertIn(
                f"{ROOT}/scripts/check_swe_task_state_validation_cohort.py",
                call,
            )
            self.assertIn(
                f"--cohort {ROOT}/configs/swe_task_state_validation_cohort.json",
                call,
            )
            self.assertIn(
                "--image-config "
                f"{ROOT}/configs/swe_task_state_validation_image_digests.json",
                call,
            )
            self.assertNotIn(str(temporary / "checker.py"), call)

    def test_rejects_distinct_generation_ports_before_running_checker(self) -> None:
        environment = {
            **os.environ,
            "PORT_A": "19998",
            "PORT_B": "19999",
        }
        result = subprocess.run(
            [str(WRAPPER), "--check-only"],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("campaign endpoint ports must match", result.stderr)

    def test_full_run_rejects_preexisting_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            existing = temporary / "swe_task_state_validation_a_20260718"
            existing.mkdir()
            result = subprocess.run(
                [str(WRAPPER)],
                cwd=ROOT,
                env={
                    **os.environ,
                    "RUN_ROOT_A": str(existing),
                    "RUN_ROOT_B": str(
                        temporary / "swe_task_state_validation_b_20260718"
                    ),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("run root already exists", result.stderr)


if __name__ == "__main__":
    unittest.main()
