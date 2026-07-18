#!/usr/bin/env python3
"""Static contract tests for the frozen contextual-evidence replay wrapper."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_swe_contextual_evidence_pilot.sh"
PROTOCOL = ROOT / "configs/swe_contextual_evidence_protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assignment(source: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}=([^\n]+)$", source, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing shell assignment: {name}")
    return match.group(1)


class RunSweContextualEvidencePilotTest(unittest.TestCase):
    def test_shell_syntax_help_and_exclusive_modes(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        help_result = subprocess.run(
            [str(SCRIPT), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--prepare-only", help_result.stdout)
        self.assertIn("--public-only", help_result.stdout)
        rejected = subprocess.run(
            [str(SCRIPT), "--prepare-only", "--public-only"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(rejected.returncode, 1)
        self.assertIn("mutually exclusive", rejected.stderr)

    def test_protocol_model_and_lens_pins_are_exact(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        protocol = json.loads(PROTOCOL.read_bytes())
        self.assertEqual(assignment(source, "PROTOCOL_SHA256"), sha256(PROTOCOL))
        self.assertEqual(
            protocol["pins"]["model"],
            {
                "repo_id": assignment(source, "MODEL_REPO"),
                "revision": assignment(source, "MODEL_REVISION"),
                "config_sha256": assignment(source, "MODEL_CONFIG_SHA256"),
                "index_sha256": assignment(source, "MODEL_INDEX_SHA256"),
            },
        )
        lenses = protocol["pins"]["lenses"]
        self.assertEqual(lenses["public"]["sha256"], assignment(source, "PUBLIC_LENS_SHA256"))
        self.assertEqual(lenses["nf4"]["sha256"], assignment(source, "NF4_LENS_SHA256"))
        self.assertEqual(
            lenses["nf4"]["provenance_sha256"],
            assignment(source, "NF4_PROVENANCE_SHA256"),
        )
        self.assertEqual(
            lenses["native_nvfp4_ste"]["sha256"],
            assignment(source, "NATIVE_LENS_SHA256"),
        )
        self.assertEqual(
            lenses["native_nvfp4_ste"]["provenance_sha256"],
            assignment(source, "NATIVE_PROVENANCE_SHA256"),
        )
        self.assertEqual(
            lenses["native_nvfp4_ste"]["state_sha256"],
            assignment(source, "NATIVE_STATE_SHA256"),
        )
        expected_layers = ",".join(str(layer) for layer in protocol["fixed_layer_band"]["layers"])
        self.assertEqual(assignment(source, "LAYERS"), expected_layers)

    def test_runner_and_checkpoint_helpers_are_content_pinned(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        pinned = {
            "JLENS_RUNNER_SHA256": ROOT / "scripts/run_jlens_nvfp4.sh",
            "JLENS_PYTHON_RUNNER_SHA256": ROOT / "scripts/run_jlens_nvfp4.py",
            "MODEL_CHECKPOINT_VERIFIER_SHA256": ROOT / "scripts/modelopt_checkpoint.py",
        }
        for variable, path in pinned.items():
            with self.subTest(variable=variable):
                self.assertEqual(assignment(source, variable), sha256(path))

    def test_exact_capture_and_analysis_contract_is_present(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        required = (
            "--output-prompts \"$PROMPTS\"",
            "--output-manifest \"$MATERIALIZATION\"",
            'PROMPTS_SHA256=0662a327c4d13e4f359935e5766df53803939917095d00f1d5122865b425497d',
            "--positions=-1",
            "--top-k \"$TOP_K\"",
            "--max-model-len \"$MAX_MODEL_LEN\"",
            "--max-num-batched-tokens \"$MAX_NUM_BATCHED_TOKENS\"",
            "--mamba-block-size \"$MAMBA_BLOCK_SIZE\"",
            "--enable-prefix-caching",
            "--kv-cache-dtype fp8_e4m3",
            "--kv-offloading-size \"$KV_OFFLOADING_SIZE\"",
            "--kv-offloading-backend native",
            "--stream-final-only",
            "--gpu-memory-utilization \"$GPU_MEMORY_UTILIZATION\"",
            "--lens-kind public",
            "--lens-kind nf4",
            "--lens-kind nvfp4-ste",
            "--manifest \"$MATERIALIZATION\"",
            "--public-report \"$PUBLIC_REPORT\"",
            "--nf4-report \"$NF4_REPORT\"",
            "--native-report \"$NATIVE_REPORT\"",
            '"mtp_enabled": False',
            '"enforce_eager": True',
            '"language_model_only": True',
            'expected_status = "passed" if process_status == "0" else "failed"',
            'if process_status not in {"0", "1"}',
            'if ((status != 0 && status != 1)); then',
            "validate_json_report \"$report\" \"$label\" \"$status\"",
            "write_sha256_sidecar \"$report\" \"$checksum\"",
            "ensure_capture_resources_idle \"$label\"",
            "attempts/attempt-",
            "emit_run_manifest failed",
            "promote_attempt latest-public",
            "promote_attempt latest",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

    def test_failed_numerical_gate_is_accepted_only_as_fresh_valid_evidence(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        fresh_check = source.index('if [[ ! -s "$report" ]]')
        status_check = source.index("if ((status != 0 && status != 1)); then")
        semantic_check = source.index(
            'if ! validate_json_report "$report" "$label" "$status"'
        )
        checksum_write = source.index('write_sha256_sidecar "$report" "$checksum"')
        accepted = source.index('accepted runner exit $status with terminal report')
        self.assertLess(fresh_check, status_check)
        self.assertLess(status_check, semantic_check)
        self.assertLess(semantic_check, checksum_write)
        self.assertLess(checksum_write, accepted)

    def test_initialization_failure_preserves_phase_status_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "pilot"
            environment = os.environ.copy()
            environment.update(
                {
                    "OUT_DIR": str(output_root),
                    "JLENS_RUNNER": "/dev/null",
                }
            )
            result = subprocess.run(
                [str(SCRIPT), "--prepare-only"],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 1)
            attempts = list((output_root / "attempts").iterdir())
            self.assertEqual(len(attempts), 1)
            attempt = attempts[0]
            self.assertEqual(
                (attempt / "initialization.exit_status").read_text(encoding="ascii"),
                "initialization-failed\n",
            )
            manifest = json.loads((attempt / "run_manifest.json").read_bytes())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["phase"], "initialization")
            self.assertEqual(manifest["process_status"], 1)
            self.assertFalse((output_root / ".pilot.lock").exists())


if __name__ == "__main__":
    unittest.main()
