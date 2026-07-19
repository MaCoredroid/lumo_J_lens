#!/usr/bin/env python3
"""Focused contract tests for the dense task-state interpreter wrapper."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_swe_task_state_interpreter.sh"
PROTOCOL = ROOT / "configs/swe_task_state_interpreter_protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assignment(source: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}=([^\n]+)$", source, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing shell assignment: {name}")
    return match.group(1)


def write_executable(path: Path, source: str) -> None:
    path.write_text(textwrap.dedent(source).lstrip(), encoding="ascii")
    path.chmod(0o755)


class RunSweTaskStateInterpreterTest(unittest.TestCase):
    def test_shell_syntax_help_and_exclusive_modes(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        help_result = subprocess.run(
            [str(SCRIPT), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--prepare-only", help_result.stdout)
        self.assertIn("--analyze-only", help_result.stdout)
        rejected = subprocess.run(
            [str(SCRIPT), "--prepare-only", "--analyze-only"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(rejected.returncode, 1)
        self.assertIn("mutually exclusive", rejected.stderr)

    def test_protocol_and_every_execution_helper_are_content_pinned(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        pinned = {
            "PROTOCOL_SHA256": PROTOCOL,
            "BEHAVIORAL_PROTOCOL_SHA256": ROOT
            / "configs/swe_behavioral_readout_protocol.json",
            "COHORT_MANIFEST_SHA256": ROOT / "configs/swe_behavioral_n20_cohort.json",
            "PRIMARY_CAMPAIGN_SHA256": ROOT / "configs/swe_behavioral_campaign.json",
            "REPLICATION_CAMPAIGN_SHA256": ROOT
            / "configs/swe_behavioral_replication_campaign.json",
            "ACTION_PROTOCOL_SHA256": ROOT / "configs/swe_stage_action_probes.json",
            "CHAT_TEMPLATE_SHA256": ROOT / "configs/qwen3-openai-codex.jinja",
            "MATERIALIZER_SHA256": ROOT
            / "scripts/materialize_swe_dense_behavioral_probes.py",
            "ANALYZER_SHA256": ROOT / "scripts/analyze_swe_task_state_interpreter.py",
            "TASK_STATE_READOUT_SHA256": ROOT / "scripts/swe_task_state_readout.py",
            "JLENS_RUNNER_SHA256": ROOT / "scripts/run_jlens_nvfp4.sh",
            "JLENS_PYTHON_RUNNER_SHA256": ROOT / "scripts/run_jlens_nvfp4.py",
            "MODEL_CHECKPOINT_VERIFIER_SHA256": ROOT / "scripts/modelopt_checkpoint.py",
        }
        for variable, path in pinned.items():
            with self.subTest(variable=variable):
                self.assertEqual(assignment(source, variable), sha256(path))

        protocol = json.loads(PROTOCOL.read_bytes())
        self.assertEqual(
            assignment(source, "PROMPTS_SHA256"),
            protocol["input_pins"]["prompt_bundle_sha256"],
        )

    def test_exact_dense_materialization_and_public_replay_contract_is_present(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        required = (
            '--cohort "$PRIMARY_CAMPAIGN" "$PRIMARY_RUN_ROOT"',
            '--cohort "$REPLICATION_CAMPAIGN" "$REPLICATION_RUN_ROOT"',
            '--cohort-manifest "$COHORT_MANIFEST"',
            '--action-protocol "$ACTION_PROTOCOL"',
            '--template "$CHAT_TEMPLATE"',
            "--require-official-outcomes",
            "--all-probeable",
            "\"prompt_count\": 698",
            "\"task_count\": 20",
            "\"global_request_count\": 699",
            "ensure_serving_endpoint_offline",
            '--lens-kind public',
            '--prompts-file "$PROMPTS"',
            '--layers "$LAYERS"',
            "--positions=-1",
            '--top-k "$TOP_K"',
            '--max-model-len "$MAX_MODEL_LEN"',
            '--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"',
            '--mamba-block-size "$MAMBA_BLOCK_SIZE"',
            "--enable-prefix-caching",
            "--kv-cache-dtype fp8_e4m3",
            '--kv-offloading-size "$KV_OFFLOADING_SIZE"',
            "--kv-offloading-backend native",
            "--stream-final-only",
            '--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"',
            '--public-report "$PUBLIC_REPORT"',
            '--behavioral-protocol "$BEHAVIORAL_PROTOCOL"',
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

        offline_call = source.index("  ensure_serving_endpoint_offline\n")
        replay_call = source.index('  "$JLENS_RUNNER" \\\n')
        self.assertLess(offline_call, replay_call)

    def test_prepare_only_fails_closed_on_a_nonfrozen_prompt_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            output_root = temporary / "output"
            fake_python = temporary / "fake-vllm-python"
            call_log = temporary / "calls.log"
            write_executable(
                fake_python,
                r"""
                #!/usr/bin/env bash
                set -euo pipefail
                printf '%s\n' "$*" >>"$CALL_LOG"
                shift
                output=
                summary=
                while (($#)); do
                  case "$1" in
                    --output) output=$2; shift 2 ;;
                    --summary) summary=$2; shift 2 ;;
                    *) shift ;;
                  esac
                done
                printf '[]\n' >"$output"
                printf '{}\n' >"$summary"
                """,
            )
            environment = {
                **os.environ,
                "CALL_LOG": str(call_log),
                "OUT_DIR": str(output_root),
                "VLLM_PY": str(fake_python),
            }
            result = subprocess.run(
                [str(SCRIPT), "--prepare-only"],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("materialized prompts do not match", result.stderr)
            call = call_log.read_text(encoding="ascii")
            self.assertIn("--all-probeable", call)
            self.assertIn("--require-official-outcomes", call)
            self.assertFalse((output_root / "checksums/prompts.sha256").exists())

    def test_analyze_only_rejects_a_self_consistent_but_unpinned_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "output"
            checksums = output_root / "checksums"
            checksums.mkdir(parents=True)
            (output_root / "logs").mkdir()
            prompts = output_root / "prompts.json"
            summary = output_root / "prompts-summary.json"
            report = output_root / "public-report.json"
            prompts.write_text("[]\n", encoding="ascii")
            summary.write_text("{}\n", encoding="ascii")
            report.write_text("{}\n", encoding="ascii")
            for path, sidecar_name in (
                (prompts, "prompts.sha256"),
                (summary, "prompts-summary.sha256"),
                (report, "public-report.sha256"),
            ):
                (checksums / sidecar_name).write_text(
                    f"{sha256(path)}  {path.name}\n", encoding="ascii"
                )
            fake_python = output_root / "fake-vllm-python"
            write_executable(
                fake_python,
                """
                #!/usr/bin/env bash
                echo "analyzer must not run" >&2
                exit 97
                """,
            )
            result = subprocess.run(
                [str(SCRIPT), "--analyze-only"],
                cwd=ROOT,
                env={
                    **os.environ,
                    "OUT_DIR": str(output_root),
                    "VLLM_PY": str(fake_python),
                },
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("dense N=20 prompt bundle SHA-256 mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
