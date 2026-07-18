#!/usr/bin/env python3
"""Static and prepare/reuse smoke tests for the multi-stage pilot wrapper."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_swe_multistage_pilot.sh"


FAKE_PYTHON = r"""#!/usr/bin/env bash
set -euo pipefail
script=$1
shift
if [[ "$script" == - ]]; then
  exec python3 - "$@"
fi
printf '%s' "$(basename "$script")" >>"$FAKE_CALLS"
for argument in "$@"; do
  printf ' %s' "$argument" >>"$FAKE_CALLS"
done
printf '\n' >>"$FAKE_CALLS"

output=
summary=
while (($#)); do
  case "$1" in
    --output)
      output=$2
      shift 2
      ;;
    --summary)
      summary=$2
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -n "$output" ]]; then
  mkdir -p "$(dirname "$output")"
  case "$(basename "$script")" in
    analyze_swe_multistage_probes.py)
      printf '{"status":"ok"}\n' >"$output"
      ;;
    augment_swe_multistage_action_probes.py)
      printf '[{"id":"fake-probe","text":"abc","token_ids":[1,2],"score_token_ids":[3],"metadata":{"kind":"fake"}}]\n' >"$output"
      ;;
    *)
      printf '[]\n' >"$output"
      ;;
  esac
fi
if [[ -n "$summary" ]]; then
  mkdir -p "$(dirname "$summary")"
  printf '{}\n' >"$summary"
fi
"""


class RunSweMultistagePilotTest(unittest.TestCase):
    def test_shell_syntax_and_exact_runtime_contract(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        text = SCRIPT.read_text(encoding="utf-8")
        required_fragments = (
            "--layers \"$LAYERS\"",
            "LAYERS=16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47",
            "--positions=-1",
            "--top-k 10",
            "--max-model-len 49152",
            "--max-num-batched-tokens 4096",
            "--mamba-block-size 1024",
            "--enable-prefix-caching",
            "--kv-cache-dtype fp8_e4m3",
            "--kv-offloading-size 8",
            "--kv-offloading-backend native",
            "--stream-final-only",
            "--gpu-memory-utilization 0.78",
            "--lens-kind public",
            "--lens-kind nf4",
            "--lens-kind nvfp4-ste",
            "--nf4-report \"$NF4_REPORT\"",
            "MODEL_CONFIG_SHA256=c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
            "MODEL_INDEX_SHA256=7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
            "REPORT_SCHEMA_VERSION=3",
            "TORCH_VERSION=2.11.0+cu130",
            "VLLM_VERSION=0.23.0",
            "TRANSFORMERS_VERSION=5.12.1",
            "--lifecycle-protocol \"$LIFECYCLE_PROTOCOL\"",
            "emit_run_manifest prepared prepare_only",
            "emit_run_manifest complete",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, text)

    def test_exact_local_lens_and_provenance_pins(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        pins = (
            "NF4_LENS_PATH=$ROOT/.cache/Qwen3.6-27B-jlens-nf4-n10-fp32.pt",
            "NF4_LENS_SHA256=54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f",
            "NF4_PROVENANCE_PATH=$ROOT/.cache/Qwen3.6-27B-jlens-nf4-n10-fp32.provenance.json",
            "NF4_PROVENANCE_SHA256=08dfe022cc74bab226308dcd77a3901ec18806f8832dc8dec46524374fa3b1e7",
            "NATIVE_LENS_PATH=$ROOT/.cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt",
            "NATIVE_LENS_SHA256=82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057",
            "NATIVE_PROVENANCE_PATH=$ROOT/.cache/nvfp4_ste_fit/final-mean/metadata.json",
            "NATIVE_PROVENANCE_SHA256=289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601",
            "NATIVE_STATE_PATH=$ROOT/.cache/nvfp4_ste_fit/state.json",
            "NATIVE_STATE_SHA256=f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6",
        )
        for pin in pins:
            self.assertIn(pin, text)
        self.assertIn("if ((status != 0 && status != 1)); then", text)
        self.assertIn('if [[ ! -s "$report" ]]; then', text)
        self.assertIn('rm -f "$report"', text)
        self.assertIn('>"$stdout_file" 2>"$log"', text)
        self.assertIn('rm -f "$stdout_file"', text)
        self.assertIn(
            'die "reused $label report is stale or violates the pinned runtime contract"',
            text,
        )

    def _fake_environment(self, directory: Path) -> dict[str, str]:
        fake_python = directory / "fake-python"
        fake_python.write_text(FAKE_PYTHON, encoding="utf-8")
        fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
        environment = os.environ.copy()
        environment.update(
            {
                "VLLM_PY": str(fake_python),
                "OUT_DIR": str(directory / "out"),
                "FAKE_CALLS": str(directory / "calls.log"),
            }
        )
        return environment

    def _report(self, lens_kind: str) -> dict:
        prompt = {
            "id": "fake-probe",
            "text": "abc",
            "token_ids": [1, 2],
            "score_token_ids": [3],
            "metadata": {"kind": "fake"},
        }
        scored = [{"token_id": 3}]
        layers = []
        for layer in range(16, 48):
            layers.append(
                {
                    "layer": layer,
                    "positions": [
                        {
                            "token_position": 1,
                            "logit_lens": {
                                "tokens": [str(index) for index in range(10)],
                                "scored_tokens": scored,
                            },
                            "jacobian_lens": {
                                "tokens": [str(index) for index in range(10)],
                                "scored_tokens": scored,
                            },
                        }
                    ],
                }
            )
        if lens_kind == "public":
            lens = {
                "repo_id": "neuronpedia/jacobian-lens",
                "revision": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
                "sha256": "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1",
                "n_prompts": 1000,
            }
        elif lens_kind == "nf4":
            lens = {
                "kind": "local_fit",
                "sha256": "54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f",
                "provenance_sha256": "08dfe022cc74bab226308dcd77a3901ec18806f8832dc8dec46524374fa3b1e7",
                "n_prompts": 10,
                "fit_quantization": "bitsandbytes-nf4-double-quant-bfloat16",
            }
        else:
            lens = {
                "kind": "native_nvfp4_ste_fit",
                "sha256": "82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057",
                "provenance_sha256": "289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601",
                "state_sha256": "f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6",
                "n_prompts": 10,
                "fit_quantization": "nvidia-modelopt-nvfp4-fp8-exact-forward-identity-ste",
            }
        return {
            "schema_version": 3,
            "score_encoding": "unrounded-float32",
            "status": "failed",
            "host": {
                "python": "3.12.13",
                "packages": {
                    "huggingface-hub": "1.21.0",
                    "torch": "2.11.0+cu130",
                    "transformers": "5.12.1",
                    "triton": "3.6.0",
                    "vllm": "0.23.0",
                },
                "gpu": {
                    "name": "NVIDIA GeForce RTX 5090",
                    "compute_capability": "12.0",
                },
            },
            "model": {
                "repo_id": "nvidia/Qwen3.6-27B-NVFP4",
                "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
                "config_sha256": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
                "index_sha256": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
            },
            "lens": lens,
            "runtime": {
                "mtp_enabled": False,
                "enforce_eager": True,
                "language_model_only": True,
                "max_model_len": 49152,
                "max_num_batched_tokens": 4096,
                "mamba_block_size": 1024,
                "enable_prefix_caching": True,
                "kv_cache_dtype": "fp8_e4m3",
                "kv_offloading_size": 8.0,
                "kv_offloading_backend": "native",
                "stream_final_only": True,
                "gpu_memory_utilization": 0.78,
                "capture_adapter": "vLLM apply_model forward hooks",
                "transport_dtype": "torch.float32",
                "readout_dtype": "torch.bfloat16",
                "timing_scope": "artifact resolution and validation through readout",
                "model_load_seconds": 1.0,
            },
            "assertions": {
                "lens_hash_matches": True,
                "lens_metadata_matches": True,
                "model_architecture_matches": True,
            },
            "scored_vocabulary": {"union_token_ids": [3]},
            "experiments": [
                {
                    "id": prompt["id"],
                    "prompt": prompt["text"],
                    "prompt_token_ids": prompt["token_ids"],
                    "metadata": prompt["metadata"],
                    "scored_vocabulary": {"token_ids": prompt["score_token_ids"]},
                    "positions_requested": [-1],
                    "positions_resolved": [1],
                    "capture_positions_resolved": [1],
                    "layers": layers,
                }
            ],
        }

    def _write_reuse_reports(self, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=True)
        for lens_kind, name in (
            ("public", "public-report.json"),
            ("nf4", "nf4-report.json"),
            ("native", "native-report.json"),
        ):
            (output / name).write_text(
                json.dumps(self._report(lens_kind)) + "\n", encoding="ascii"
            )

    def test_prepare_only_smoke_honors_out_dir_and_stops_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment = self._fake_environment(directory)
            output = Path(environment["OUT_DIR"])
            output.mkdir(parents=True)
            (output / "prompts.json").write_text("stale\n", encoding="ascii")
            (output / "analysis.json").write_text("stale\n", encoding="ascii")
            (output / "public.exit_status").write_text("stale\n", encoding="ascii")
            result = subprocess.run(
                [str(SCRIPT), "--prepare-only"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "prompts.json").is_file())
            self.assertTrue((output / "prompts_summary.json").is_file())
            self.assertTrue((output / "action_prompts.json").is_file())
            self.assertTrue((output / "action_prompts_summary.json").is_file())
            calls = (directory / "calls.log").read_text(encoding="utf-8")
            self.assertEqual(len(calls.splitlines()), 2)
            self.assertIn("materialize_swe_multistage_probes.py", calls)
            self.assertIn("augment_swe_multistage_action_probes.py", calls)
            self.assertNotIn("run_jlens_nvfp4", calls)
            self.assertFalse((output / "analysis.json").exists())
            self.assertFalse((output / "public.exit_status").exists())
            manifest_path = output / "run_manifest.json"
            checksum_path = output / "run_manifest.sha256"
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(manifest["mode"], "prepare_only")
            self.assertEqual(set(manifest["phases"]), {"materialize", "augment"})
            self.assertFalse(manifest["path_contract"]["absolute_paths_embedded"])
            for collection in (manifest["inputs"], manifest["artifacts"]):
                for record in collection.values():
                    self.assertFalse(Path(record["path"]).is_absolute())
                    self.assertIn(
                        record["path_base"],
                        {"repository_root", "output_directory"},
                    )
            expected_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            checksum_digest, checksum_target = checksum_path.read_text().split()
            self.assertEqual(checksum_digest, expected_digest)
            self.assertEqual(checksum_target, manifest_path.name)
            self.assertEqual(
                hashlib.sha256((output / checksum_target).read_bytes()).hexdigest(),
                checksum_digest,
            )

    def test_reuse_reports_smoke_skips_replay_and_runs_three_report_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment = self._fake_environment(directory)
            output = Path(environment["OUT_DIR"])
            self._write_reuse_reports(output)
            result = subprocess.run(
                [str(SCRIPT), "--reuse-reports"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "analysis.json").is_file())
            calls = (directory / "calls.log").read_text(encoding="utf-8")
            self.assertEqual(len(calls.splitlines()), 3)
            analyzer_call = calls.splitlines()[-1]
            self.assertIn("analyze_swe_multistage_probes.py", analyzer_call)
            self.assertIn(f"--public-report {output / 'public-report.json'}", analyzer_call)
            self.assertIn(f"--nf4-report {output / 'nf4-report.json'}", analyzer_call)
            self.assertIn(f"--native-report {output / 'native-report.json'}", analyzer_call)
            self.assertNotIn("run_jlens_nvfp4", calls)
            manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="ascii")
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["mode"], "reuse_reports")
            self.assertEqual(manifest["schema_version"], 2)
            self.assertFalse(manifest["path_contract"]["absolute_paths_embedded"])
            self.assertEqual(manifest["phases"]["public"]["status"], "reused")
            self.assertEqual(manifest["phases"]["nf4"]["status"], "reused")
            self.assertEqual(manifest["phases"]["native"]["status"], "reused")

    def test_reuse_rejects_report_bound_to_stale_prompt_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment = self._fake_environment(directory)
            output = Path(environment["OUT_DIR"])
            self._write_reuse_reports(output)
            stale = json.loads(
                (output / "public-report.json").read_text(encoding="ascii")
            )
            stale["experiments"][0]["prompt"] = "stale"
            (output / "public-report.json").write_text(
                json.dumps(stale) + "\n", encoding="ascii"
            )

            result = subprocess.run(
                [str(SCRIPT), "--reuse-reports"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stale or violates", result.stderr)
            self.assertFalse((output / "public.exit_status").exists())
            self.assertFalse((output / "analysis.json").exists())
            self.assertFalse((output / "run_manifest.json").exists())

    def test_mutually_exclusive_modes_fail_closed(self) -> None:
        result = subprocess.run(
            [str(SCRIPT), "--prepare-only", "--reuse-reports"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutually exclusive", result.stderr)


if __name__ == "__main__":
    unittest.main()
