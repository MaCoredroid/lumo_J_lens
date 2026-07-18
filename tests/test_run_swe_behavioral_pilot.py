#!/usr/bin/env python3
"""Static and failure-preservation tests for the behavioral N=20 pilot."""

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
SCRIPT = ROOT / "scripts/run_swe_behavioral_pilot.sh"
PRIMARY_CAMPAIGN = ROOT / "configs/swe_behavioral_campaign.json"
REPLICATION_CAMPAIGN = ROOT / "configs/swe_behavioral_replication_campaign.json"
MODEL_ID = "qwen3.6-27b-nvfp4-mtp::qwen-code-0.19.4"
SWEBENCH_VERSION = "4.1.0"
SPEC_DECODING_LINES = (
    "(APIServer pid=1234) INFO 07-18 00:00:10 [metrics.py:101] "
    "SpecDecoding metrics: Mean acceptance length: 1.90, "
    "Accepted throughput: 9.00 tokens/s, Drafted throughput: 10.00 tokens/s, "
    "Accepted: 90 tokens, Drafted: 100 tokens, "
    "Per-position acceptance rate: 0.900, Avg Draft acceptance rate: 90.0%",
    "(APIServer pid=1234) INFO 07-18 00:00:20 [metrics.py:101] "
    "SpecDecoding metrics: Mean acceptance length: 1.80, "
    "Accepted throughput: 8.00 tokens/s, Drafted throughput: 10.00 tokens/s, "
    "Accepted: 80 tokens, Drafted: 100 tokens, "
    "Per-position acceptance rate: 0.800, Avg Draft acceptance rate: 80.0%",
)
SERVER_LOG_BYTES = ("\n".join(SPEC_DECODING_LINES) + "\n").encode("ascii")


FAKE_PYTHON = r"""#!/usr/bin/env bash
set -euo pipefail
script=$1
shift
if [[ "$script" == - ]]; then
  if [[ "${1:-}" == behavioral-pilot-manifest-v1 || \
        "${1:-}" == behavioral-pilot-prompt-length-v1 || \
        "${1:-}" == behavioral-pilot-campaign-v1 ]]; then
    exec python3 - "$@"
  fi
  cat >/dev/null
  exit 0
fi

printf '%s' "$(basename "$script")" >>"$FAKE_CALLS"
for argument in "$@"; do
  printf ' %s' "$argument" >>"$FAKE_CALLS"
done
printf '\n' >>"$FAKE_CALLS"

if [[ "$(basename "$script")" == materialize_swe_behavioral_probes.py ]]; then
  if [[ "${FAKE_MATERIALIZER_STATUS:-0}" != 0 ]]; then
    exit "$FAKE_MATERIALIZER_STATUS"
  fi
fi

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
  if [[ "$(basename "$script")" == materialize_swe_behavioral_probes.py && \
        "${FAKE_MATERIALIZER_NO_OUTPUT:-0}" == 1 ]]; then
    :
  elif [[ "$(basename "$script")" == analyze_swe_behavioral_probes.py ]]; then
    printf '{"status":"insufficient_split_or_class_support"}\n' >"$output"
  elif [[ "$(basename "$script")" == materialize_swe_behavioral_probes.py && \
          "${FAKE_OVERSIZED_PROMPT:-0}" == 1 ]]; then
    python3 -c 'import json, sys; json.dump([{"id": "too-long", "token_ids": [0] * 65536}], open(sys.argv[1], "w", encoding="ascii"))' "$output"
  else
    printf '[]\n' >"$output"
  fi
fi
if [[ -n "$summary" ]]; then
  mkdir -p "$(dirname "$summary")"
  printf '{}\n' >"$summary"
fi
"""


class RunSweBehavioralPilotTest(unittest.TestCase):
    @staticmethod
    def _canonical_bytes(value: object) -> bytes:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")

    def test_shell_syntax_and_exact_capture_contract(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        text = SCRIPT.read_text(encoding="utf-8")
        required = (
            "--cohort \"$PRIMARY_CAMPAIGN\" \"$PRIMARY_RUN_ROOT\"",
            "--cohort \"$REPLICATION_CAMPAIGN\" \"$REPLICATION_RUN_ROOT\"",
            "--cohort-manifest \"$COHORT_MANIFEST\"",
            "--require-official-outcomes",
            "LAYERS=24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47",
            "--positions=-1",
            "--top-k 10",
            "MAX_MODEL_LEN=65536",
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
            "--bootstrap-samples \"$BOOTSTRAP_SAMPLES\"",
            'COHORT_MANIFEST=${COHORT_MANIFEST:-',
            'ensure_capture_resources_idle "$label"',
            "--query-compute-apps=pid,process_name,used_memory",
            "--query-gpu=memory.used",
            '"gnome-control-center": 96',
            '"ptyxis": 64',
            '"nautilus": 64',
            "aggregate_process_cap_mib = 192",
            "total_gpu_memory_cap_mib = 640",
            '"$SERVER_ENDPOINT"',
            'server_log_path = run_root / "server.log"',
            '"SpecDecoding metrics:"',
            '"weighted_acceptance_rate"',
            "attempts/",
            "promote_attempt latest",
            '"mtp_enabled": False',
            '"enforce_eager": True',
            '"language_model_only": True',
            '"trajectory_generation": {',
            '"enabled": True',
            "emit_run_manifest failed",
            "validate_json_report",
            "validate_analysis",
        )
        for fragment in required:
            self.assertIn(fragment, text)

    def test_exact_model_runner_and_lens_pins(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        pins = (
            "MODEL_REVISION=0893e1606ff3d5f97a441f405d5fc541a6bdf404",
            "MODEL_CONFIG_SHA256=c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
            "MODEL_INDEX_SHA256=7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
            "MODEL_CHECKPOINT_VERIFIER_SHA256=f02b4cdf84d800b13165dfc559c244bdce8c693e939d64d05a5e26351ce57fb6",
            "JLENS_RUNNER_SHA256=89763dc20b09394e52f2296654b58408de40c902cb5dddc0a109026964eb2331",
            "JLENS_PYTHON_RUNNER_SHA256=18697bd8adc1159b7228390c526b9c418e87c25c64e7fa5d121d9f95906f7ee5",
            "PUBLIC_LENS_SHA256=1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1",
            "NF4_LENS_SHA256=54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f",
            "NF4_PROVENANCE_SHA256=08dfe022cc74bab226308dcd77a3901ec18806f8832dc8dec46524374fa3b1e7",
            "NATIVE_LENS_SHA256=82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057",
            "NATIVE_PROVENANCE_SHA256=289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601",
            "NATIVE_STATE_SHA256=f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6",
        )
        for pin in pins:
            self.assertIn(pin, text)
        self.assertIn("PINNED_METADATA_SHA256, PINNED_SHARDS", text)
        self.assertIn('"validated_after_evaluation": True', text)

    def _campaign_run(self, path: Path, campaign_path: Path) -> None:
        campaign_bytes = campaign_path.read_bytes()
        campaign_sha = hashlib.sha256(campaign_bytes).hexdigest()
        campaign = json.loads(campaign_bytes)
        instance_ids = campaign["instance_ids"]
        logical_campaign = campaign_path.relative_to(ROOT).as_posix()
        path.mkdir(parents=True)
        (path / "server.log").write_bytes(SERVER_LOG_BYTES)
        (path / "generation_sources.sha256").write_text(
            f"{campaign_sha}  {logical_campaign}\n", encoding="ascii"
        )
        dataset = [{"instance_id": instance_id} for instance_id in instance_ids]
        dataset_bytes = (
            json.dumps(dataset, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("ascii")
        (path / "dataset.json").write_bytes(dataset_bytes)
        images = [
            {
                "instance_id": instance_id,
                "image_id": f"sha256:{index + 1:064x}",
            }
            for index, instance_id in enumerate(instance_ids)
        ]
        image_manifest = {
            "schema_version": 1,
            "kind": "swe_verified_behavioral_campaign_image_manifest",
            "campaign_config_sha256": campaign_sha,
            "images": images,
        }
        image_bytes = (
            json.dumps(image_manifest, indent=2, sort_keys=True, ensure_ascii=True)
            + "\n"
        ).encode("ascii")
        (path / "image_manifest.json").write_bytes(image_bytes)

        prediction_rows = [
            {
                "instance_id": instance_id,
                "model_name_or_path": MODEL_ID,
                "model_patch": f"patch for {instance_id}\n",
            }
            for instance_id in instance_ids
        ]
        source_bytes = b"".join(
            self._canonical_bytes(row) + b"\n" for row in prediction_rows
        )
        source_path = path / "generation/verified/predictions.jsonl"
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(source_bytes)
        for instance_id in instance_ids:
            metadata_path = (
                path
                / "generation/verified/per_task"
                / instance_id
                / "runner_metadata.json"
            )
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "ended_at": "2026-07-18T00:00:00+00:00",
                        "qwen": {
                            "parsed": True,
                            "exit_code": 0,
                            "timed_out": False,
                            "num_turns": 2,
                        },
                        "eval_report": {"model_id": MODEL_ID},
                    }
                ),
                encoding="ascii",
            )
        proxy_root = path / "proxy_dumps"
        proxy_root.mkdir()
        for index in range(1, len(instance_ids) * 2 + 1):
            (proxy_root / f"chat_{index:04d}.json").write_text(
                '{}\n', encoding="ascii"
            )

        complete_bytes = b"".join(
            self._canonical_bytes(row) + b"\n" for row in prediction_rows
        )
        hashes = {
            "campaign_config_sha256": campaign_sha,
            "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "image_manifest_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "source_predictions_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "complete_predictions_sha256": hashlib.sha256(complete_bytes).hexdigest(),
        }
        evidence_payload = {
            "hashes": hashes,
            "instance_ids": instance_ids,
            "model_name_or_path": MODEL_ID,
            "swebench_version": SWEBENCH_VERSION,
        }
        evidence_id = hashlib.sha256(
            self._canonical_bytes(evidence_payload)
        ).hexdigest()
        run_name = path.name
        run_id = f"lumo_j_lens_{run_name}_behavioral_{evidence_id[:16]}"
        outcomes = []
        for row in prediction_rows:
            patch_bytes = row["model_patch"].encode("utf-8")
            outcomes.append(
                {
                    "instance_id": row["instance_id"],
                    "outcome": "unresolved",
                    "patch_bytes": len(patch_bytes),
                    "patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
                }
            )
        official = {
            "schema_version": 1,
            "kind": "swe_verified_behavioral_official_outcomes",
            "run_name": run_name,
            "run_id": run_id,
            "evidence_id": evidence_id,
            "instance_ids": instance_ids,
            "counts": {
                "resolved": 0,
                "unresolved": len(instance_ids),
                "error": 0,
                "empty": 0,
            },
            "outcomes": outcomes,
            "inputs": {
                "hashes": hashes,
                "score_input_manifest_sha256": "0" * 64,
                "missing_prediction_ids": [],
                "empty_prediction_ids": [],
                "model_name_or_path": MODEL_ID,
            },
            "official_harness": {
                "swebench_version": SWEBENCH_VERSION,
                "max_workers": 1,
                "exit_status": 0,
            },
        }
        official_path = path / "official_score/official_outcomes.json"
        official_path.parent.mkdir(parents=True)
        official_path.write_text(
            json.dumps(official, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="ascii",
        )

    def _environment(self, directory: Path) -> tuple[dict[str, str], Path]:
        fake_python = directory / "fake-python"
        fake_python.write_text(FAKE_PYTHON, encoding="utf-8")
        fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
        primary = directory / "primary"
        replication = directory / "replication"
        self._campaign_run(primary, PRIMARY_CAMPAIGN)
        self._campaign_run(replication, REPLICATION_CAMPAIGN)
        output = directory / "out"
        environment = os.environ.copy()
        environment.update(
            {
                "VLLM_PY": str(fake_python),
                "OUT_DIR": str(output),
                "PRIMARY_RUN_ROOT": str(primary),
                "REPLICATION_RUN_ROOT": str(replication),
                "FAKE_CALLS": str(directory / "calls.log"),
                "SERVER_UNIT": "lumo-jlens-test-unit-does-not-exist",
            }
        )
        return environment, output

    def _attempts(self, output: Path) -> list[Path]:
        return sorted(path for path in (output / "attempts").iterdir() if path.is_dir())

    def _only_attempt(self, output: Path) -> Path:
        attempts = self._attempts(output)
        self.assertEqual(len(attempts), 1)
        return attempts[0]

    def _run_preflight(
        self, directory: Path, *, process_rows: str, total_memory_mib: str
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        output = directory / "preflight-output"
        (output / "logs").mkdir(parents=True)
        environment = os.environ.copy()
        environment.update(
            {
                "FAKE_NVIDIA_PROCESS_ROWS": process_rows,
                "FAKE_NVIDIA_TOTAL_MEMORY": total_memory_mib,
            }
        )
        command = r'''
set -Eeuo pipefail
OUT_DIR=$2
LOG_DIR=$OUT_DIR/logs
SERVER_UNIT=lumo-jlens-test-unit-does-not-exist
SERVER_ENDPOINT=http://127.0.0.1:1/v1/models
MANIFEST_PY=python3
CURRENT_PHASE=
systemctl() { return 3; }
curl() { return 7; }
nvidia-smi() {
  case "$*" in
    *--query-compute-apps=pid,process_name,used_memory*)
      printf '%s' "$FAKE_NVIDIA_PROCESS_ROWS"
      ;;
    *--query-gpu=memory.used*)
      printf '%s\n' "$FAKE_NVIDIA_TOTAL_MEMORY"
      ;;
    *)
      return 2
      ;;
  esac
}
show_failure_log() {
  [[ ! -f "$1" ]] || cat "$1" >&2
}
die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}
source <(
  awk '
    /^ensure_capture_resources_idle\(\) \{/ { capture = 1 }
    /^stage_reused_reports_impl\(\) \{/ { capture = 0 }
    capture { print }
  ' "$1"
)
ensure_capture_resources_idle public
'''
        result = subprocess.run(
            ["bash", "-c", command, "preflight-test", str(SCRIPT), str(output)],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        return result, output

    def _seed_latest_reports(self, output: Path, *, stale_public: bool = False) -> Path:
        attempt = output / "attempts/prior-complete"
        checksum_dir = attempt / "checksums"
        checksum_dir.mkdir(parents=True)
        (attempt / "run_manifest.json").write_text(
            '{"status":"complete"}\n', encoding="ascii"
        )
        for label in ("public", "nf4", "native"):
            report = attempt / f"{label}-report.json"
            report.write_text("{}\n", encoding="ascii")
            digest = hashlib.sha256(report.read_bytes()).hexdigest()
            if label == "public" and stale_public:
                digest = "f" * 64
            (checksum_dir / f"{label}-report.sha256").write_text(
                f"{digest}  {report.name}\n", encoding="ascii"
            )
        (output / "latest").symlink_to("attempts/prior-complete")
        return attempt

    def _make_fatal_turn_limit(self, run_root: Path, instance_id: str) -> None:
        task_root = run_root / "generation/verified/per_task" / instance_id
        metadata_path = task_root / "runner_metadata.json"
        metadata = json.loads(metadata_path.read_bytes())
        metadata["qwen"] = {
            "elapsed_s": 1.0,
            "exit_code": 53,
            "timed_out": False,
            "cli_exit_is_verdict": False,
            "parsed": False,
            "subtype": None,
            "num_turns": None,
            "duration_api_ms": None,
            "usage": None,
            "tool_calls": None,
            "tool_by_name": None,
            "result_tail": "",
        }
        metadata_path.write_text(json.dumps(metadata), encoding="ascii")
        predictions = [
            json.loads(line)
            for line in (run_root / "generation/verified/predictions.jsonl")
            .read_text(encoding="ascii")
            .splitlines()
        ]
        patch = next(
            row["model_patch"] for row in predictions if row["instance_id"] == instance_id
        )
        (task_root / "patch.diff").write_text(patch, encoding="utf-8")
        (task_root / "qwen_trace.json").write_bytes(b"")
        (task_root / "qwen_stderr.log").write_text(
            json.dumps(
                {
                    "error": {
                        "type": "FatalTurnLimitedError",
                        "message": (
                            "Reached max session turns for this session. Increase the "
                            "number of turns by specifying maxSessionTurns in settings.json."
                        ),
                        "code": 53,
                    }
                }
            ),
            encoding="ascii",
        )
        usage_path = task_root / "qwen_home/.qwen/usage_record.jsonl"
        usage_path.parent.mkdir(parents=True)
        usage_path.write_text(
            json.dumps({"models": {"qwen3.6-27b-nvfp4": {"requests": 50}}}) + "\n",
            encoding="ascii",
        )
        proxy_root = run_root / "proxy_dumps"
        for path in proxy_root.glob("chat_*.json"):
            path.unlink()
        request_count = 0
        for path in sorted(
            (run_root / "generation/verified/per_task").glob("*/runner_metadata.json")
        ):
            qwen = json.loads(path.read_bytes())["qwen"]
            request_count += qwen["num_turns"] if qwen["parsed"] else 50
        for index in range(1, request_count + 1):
            (proxy_root / f"chat_{index:04d}.json").write_text(
                '{}\n', encoding="ascii"
            )

    def _empty_prediction_and_rebind_official(
        self, run_root: Path, instance_id: str
    ) -> None:
        source_path = run_root / "generation/verified/predictions.jsonl"
        rows = [json.loads(line) for line in source_path.read_text().splitlines()]
        for row in rows:
            if row["instance_id"] == instance_id:
                row["model_patch"] = ""
        source_bytes = b"".join(self._canonical_bytes(row) + b"\n" for row in rows)
        source_path.write_bytes(source_bytes)
        task_patch = (
            run_root / "generation/verified/per_task" / instance_id / "patch.diff"
        )
        task_patch.write_bytes(b"")

        official_path = run_root / "official_score/official_outcomes.json"
        official = json.loads(official_path.read_bytes())
        hashes = official["inputs"]["hashes"]
        hashes["source_predictions_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        hashes["complete_predictions_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        official["inputs"]["empty_prediction_ids"] = [instance_id]
        for row in official["outcomes"]:
            if row["instance_id"] == instance_id:
                row.update(
                    {
                        "outcome": "empty",
                        "patch_bytes": 0,
                        "patch_sha256": hashlib.sha256(b"").hexdigest(),
                    }
                )
        official["counts"] = {
            "resolved": 0,
            "unresolved": len(rows) - 1,
            "error": 0,
            "empty": 1,
        }
        evidence_payload = {
            "hashes": hashes,
            "instance_ids": official["instance_ids"],
            "model_name_or_path": MODEL_ID,
            "swebench_version": SWEBENCH_VERSION,
        }
        evidence_id = hashlib.sha256(
            self._canonical_bytes(evidence_payload)
        ).hexdigest()
        official["evidence_id"] = evidence_id
        official["run_id"] = (
            f"lumo_j_lens_{official['run_name']}_behavioral_{evidence_id[:16]}"
        )
        official_path.write_text(
            json.dumps(official, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="ascii",
        )

    def test_idle_gpu_preflight_allows_pinned_desktop_baseline_and_logs_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            rows = "\n".join(
                (
                    "10197, /usr/bin/gnome-control-center, 60",
                    "2098863, /usr/bin/ptyxis, 44",
                    "2098966, /usr/bin/nautilus, 52",
                )
            )
            result, output = self._run_preflight(
                directory, process_rows=rows, total_memory_mib="500"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (output / "public_preflight.exit_status").read_text().strip(), "0"
            )
            log = (output / "logs/public_preflight.log").read_text()
            self.assertIn("gnome-control-center<=96 MiB (max 1)", log)
            self.assertIn(
                "pid=10197, process_name='/usr/bin/gnome-control-center', "
                "basename=gnome-control-center, memory=60 MiB, cap=96 MiB",
                log,
            )
            self.assertIn(
                "count=3, memory=156 MiB, cap=192 MiB",
                log,
            )
            self.assertIn("memory=500 MiB, cap=640 MiB", log)

    def test_idle_gpu_preflight_rejects_nonbaseline_or_excessive_use(self) -> None:
        cases = {
            "python": (
                "4001, /usr/bin/python3, 10",
                "500",
                "basename='python3'",
            ),
            "vllm": (
                "4002, /opt/vllm/bin/vllm, 10",
                "500",
                "basename='vllm'",
            ),
            "unknown": (
                "4003, /usr/bin/unknown-gpu-client, 10",
                "500",
                "basename='unknown-gpu-client'",
            ),
            "per_process_cap": (
                "4004, /usr/bin/gnome-control-center, 97",
                "500",
                "uses 97 MiB, cap is 96 MiB",
            ),
            "aggregate_cap": (
                "\n".join(
                    (
                        "4005, /usr/bin/gnome-control-center, 96",
                        "4006, /usr/bin/ptyxis, 64",
                        "4007, /usr/bin/nautilus, 64",
                    )
                ),
                "500",
                "224 MiB > 192 MiB",
            ),
            "total_memory_cap": (
                "4008, /usr/bin/nautilus, 52",
                "641",
                "641 MiB > 640 MiB",
            ),
            "duplicate_basename": (
                "\n".join(
                    (
                        "4009, /usr/bin/nautilus, 52",
                        "4010, /usr/bin/nautilus, 52",
                    )
                ),
                "500",
                "duplicate allowed idle desktop GPU process: nautilus",
            ),
        }
        for name, (rows, total_memory, expected) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as raw_directory:
                result, output = self._run_preflight(
                    Path(raw_directory),
                    process_rows=rows,
                    total_memory_mib=total_memory,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(
                    (output / "public_preflight.exit_status").read_text().strip(),
                    "capture-resource-busy-or-unverifiable",
                )
                log = (output / "logs/public_preflight.log").read_text()
                self.assertIn(expected, log)

    def test_campaign_mtp_evidence_rejects_missing_or_invalid_server_log_proof(
        self,
    ) -> None:
        invalid_total_line = SPEC_DECODING_LINES[0].replace(
            "Accepted: 90 tokens, Drafted: 100 tokens",
            "Accepted: 101 tokens, Drafted: 100 tokens",
        )
        cases = {
            "symlink": "not a regular non-symlink file",
            "changed_metric_format": "SpecDecoding metrics line 1 changed format",
            "invalid_token_totals": (
                "speculative-decoding token totals are invalid: "
                "accepted=101, drafted=100"
            ),
        }
        for name, expected in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as raw_directory:
                directory = Path(raw_directory)
                environment, output = self._environment(directory)
                server_log = Path(environment["PRIMARY_RUN_ROOT"]) / "server.log"
                if name == "symlink":
                    target = directory / "server-log-target"
                    target.write_bytes(SERVER_LOG_BYTES)
                    server_log.unlink()
                    server_log.symlink_to(target)
                elif name == "changed_metric_format":
                    server_log.write_text(
                        SPEC_DECODING_LINES[0].replace(
                            "Mean acceptance length", "Mean accepted length"
                        )
                        + "\n",
                        encoding="ascii",
                    )
                else:
                    server_log.write_text(invalid_total_line + "\n", encoding="ascii")
                result = subprocess.run(
                    [str(SCRIPT), "--prepare-only"],
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
                attempt = self._only_attempt(output)
                self.assertFalse((attempt / "campaign_evidence.json").exists())
                self.assertEqual(
                    (attempt / "campaign_validation.exit_status").read_text().strip(),
                    "invalid-campaign-evidence",
                )

    def test_prepare_only_emits_portable_hash_bound_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment, output = self._environment(directory)
            result = subprocess.run(
                [str(SCRIPT), "--prepare-only"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            attempt = self._only_attempt(output)
            self.assertEqual((output / "latest-prepared").resolve(), attempt.resolve())
            self.assertTrue((attempt / "prompts.json").is_file())
            self.assertTrue((attempt / "prompts_summary.json").is_file())
            self.assertTrue((attempt / "checksums/prompts.sha256").is_file())
            self.assertTrue((attempt / "campaign_evidence.json").is_file())
            evidence = json.loads((attempt / "campaign_evidence.json").read_bytes())
            development_evidence = next(
                row for row in evidence["cohorts"] if row["cohort"] == "development"
            )
            mtp = development_evidence["mtp_speculative_decoding"]
            self.assertEqual(mtp["server_log"]["path"], "server.log")
            self.assertEqual(mtp["server_log"]["bytes"], len(SERVER_LOG_BYTES))
            self.assertEqual(
                mtp["server_log"]["sha256"],
                hashlib.sha256(SERVER_LOG_BYTES).hexdigest(),
            )
            self.assertEqual(mtp["metric_line_count"], 2)
            self.assertEqual(mtp["accepted_tokens"], 170)
            self.assertEqual(mtp["drafted_tokens"], 200)
            self.assertEqual(mtp["weighted_acceptance_rate"], 0.85)
            calls = (directory / "calls.log").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 1)
            self.assertIn("materialize_swe_behavioral_probes.py", calls[0])
            self.assertIn("--cohort", calls[0])
            self.assertNotIn("analyze_swe_behavioral_probes.py", calls[0])

            manifest_path = attempt / "run_manifest.json"
            checksum_path = attempt / "run_manifest.sha256"
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(manifest["scientific_status"], "not_run")
            self.assertEqual(manifest["mode"], "prepare_only")
            self.assertEqual([row["name"] for row in manifest["cohorts"]], ["development", "replication"])
            self.assertFalse(manifest["path_contract"]["absolute_paths_embedded"])
            self.assertEqual(manifest["runtime_contract"]["max_model_len"], 65536)
            self.assertEqual(manifest["runtime_contract"]["layers"], list(range(24, 48)))
            self.assertTrue(
                manifest["runtime_contract"]["mtp_scope"]["trajectory_generation"]["enabled"]
            )
            self.assertFalse(
                manifest["runtime_contract"]["mtp_scope"]["eager_residual_capture"]["enabled"]
            )
            expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            digest, target = checksum_path.read_text(encoding="ascii").split()
            self.assertEqual((digest, target), (expected, manifest_path.name))

    def test_materializer_failure_preserves_log_status_and_failed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment, output = self._environment(directory)
            prior = self._seed_latest_reports(output)
            environment["FAKE_MATERIALIZER_STATUS"] = "23"
            result = subprocess.run(
                [str(SCRIPT), "--prepare-only"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            attempts = self._attempts(output)
            self.assertEqual(len(attempts), 2)
            attempt = next(path for path in attempts if path != prior)
            self.assertEqual(
                (attempt / "materialize.exit_status").read_text(encoding="ascii").strip(),
                "23",
            )
            self.assertTrue((attempt / "logs/materialize.log").is_file())
            manifest = json.loads((attempt / "run_manifest.json").read_text(encoding="ascii"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failure"]["phase"], "materialize")
            self.assertEqual(manifest["phases"]["materialize"]["status"], "23")
            self.assertNotIn("analysis", manifest["artifacts"])
            self.assertEqual((output / "latest").resolve(), prior.resolve())

    def test_oversized_prompt_fails_before_any_lens_process(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment, output = self._environment(directory)
            environment["FAKE_OVERSIZED_PROMPT"] = "1"
            result = subprocess.run(
                [str(SCRIPT)],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            attempt = self._only_attempt(output)
            self.assertIn("too-long: 65536 tokens", result.stderr)
            self.assertIn("max_prompt_tokens=65535", result.stderr)
            self.assertEqual(
                (attempt / "materialize.exit_status").read_text(encoding="ascii").strip(),
                "invalid-materialized-bundle",
            )
            calls = (directory / "calls.log").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 1)
            self.assertIn("materialize_swe_behavioral_probes.py", calls[0])
            self.assertFalse(any("run_jlens_nvfp4" in call for call in calls))
            manifest = json.loads((attempt / "run_manifest.json").read_text(encoding="ascii"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failure"]["phase"], "materialize")

    def test_reuse_rejects_report_with_stale_checksum_before_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment, output = self._environment(directory)
            prior = self._seed_latest_reports(output, stale_public=True)
            result = subprocess.run(
                [str(SCRIPT), "--reuse-reports"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            calls = (directory / "calls.log").read_text(encoding="utf-8")
            self.assertIn("materialize_swe_behavioral_probes.py", calls)
            self.assertNotIn("analyze_swe_behavioral_probes.py", calls)
            attempt = next(path for path in self._attempts(output) if path != prior)
            manifest = json.loads((attempt / "run_manifest.json").read_text(encoding="ascii"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failure"]["phase"], "public")
            self.assertIn("public_report", manifest["artifacts"])
            self.assertEqual(
                (attempt / "public.exit_status").read_text(encoding="ascii").strip(),
                "invalid-reused-report-checksum",
            )
            self.assertEqual((output / "latest").resolve(), prior.resolve())

    def test_missing_materializer_output_never_records_success_status(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment, output = self._environment(directory)
            environment["FAKE_MATERIALIZER_NO_OUTPUT"] = "1"
            result = subprocess.run(
                [str(SCRIPT), "--prepare-only"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            attempt = self._only_attempt(output)
            status = (attempt / "materialize.exit_status").read_text().strip()
            self.assertEqual(status, "missing-output-after-process-status-0")
            manifest = json.loads((attempt / "run_manifest.json").read_bytes())
            self.assertEqual(
                manifest["phases"]["materialize"]["status"],
                "missing-output-after-process-status-0",
            )

    def test_real_campaign_validator_rejects_post_score_dataset_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment, output = self._environment(directory)
            dataset_path = Path(environment["PRIMARY_RUN_ROOT"]) / "dataset.json"
            dataset = json.loads(dataset_path.read_bytes())
            dataset[0]["drift"] = True
            dataset_path.write_text(json.dumps(dataset), encoding="ascii")
            result = subprocess.run(
                [str(SCRIPT), "--prepare-only"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("official score input/hash contract changed", result.stderr)
            attempt = self._only_attempt(output)
            self.assertFalse((attempt / "materialize.exit_status").exists())
            self.assertEqual(
                (attempt / "campaign_validation.exit_status").read_text().strip(),
                "invalid-campaign-evidence",
            )

    def test_missing_campaign_root_still_emits_failed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment, output = self._environment(directory)
            environment["PRIMARY_RUN_ROOT"] = str(directory / "missing-primary")
            result = subprocess.run(
                [str(SCRIPT), "--prepare-only"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            attempt = self._only_attempt(output)
            manifest = json.loads((attempt / "run_manifest.json").read_bytes())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failure"]["phase"], "campaign_validation")
            self.assertIsNone(manifest["cohorts"][0]["dataset"])
            self.assertTrue((attempt / "run_manifest.sha256").is_file())

    def test_exact_fatal_turn_limit_is_accepted_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment, output = self._environment(directory)
            replication = Path(environment["REPLICATION_RUN_ROOT"])
            instance_id = json.loads(REPLICATION_CAMPAIGN.read_bytes())["instance_ids"][0]
            self._make_fatal_turn_limit(replication, instance_id)
            result = subprocess.run(
                [str(SCRIPT), "--prepare-only"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            attempt = self._only_attempt(output)
            evidence = json.loads((attempt / "campaign_evidence.json").read_bytes())
            replication_evidence = next(
                row for row in evidence["cohorts"] if row["cohort"] == "replication"
            )
            fatal = replication_evidence["fatal_turn_limit_tasks"]
            self.assertEqual([row["instance_id"] for row in fatal], [instance_id])
            self.assertEqual(fatal[0]["proxy_request_count"], 50)
            self.assertEqual(fatal[0]["empty_trace"]["bytes"], 0)
            self.assertGreater(fatal[0]["patch"]["bytes"], 0)

    def test_generic_unparsed_failure_is_not_accepted_as_turn_limit(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment, output = self._environment(directory)
            replication = Path(environment["REPLICATION_RUN_ROOT"])
            instance_id = json.loads(REPLICATION_CAMPAIGN.read_bytes())["instance_ids"][0]
            metadata_path = (
                replication
                / "generation/verified/per_task"
                / instance_id
                / "runner_metadata.json"
            )
            metadata = json.loads(metadata_path.read_bytes())
            metadata["qwen"] = {
                "parsed": False,
                "exit_code": 1,
                "timed_out": False,
                "num_turns": None,
            }
            metadata_path.write_text(json.dumps(metadata), encoding="ascii")
            result = subprocess.run(
                [str(SCRIPT), "--prepare-only"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsupported unparsed termination", result.stderr)
            attempt = self._only_attempt(output)
            self.assertFalse((attempt / "campaign_evidence.json").exists())

    def test_exact_fatal_turn_limit_retains_an_empty_patch_without_imputation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment, output = self._environment(directory)
            replication = Path(environment["REPLICATION_RUN_ROOT"])
            instance_id = json.loads(REPLICATION_CAMPAIGN.read_bytes())["instance_ids"][0]
            self._make_fatal_turn_limit(replication, instance_id)
            self._empty_prediction_and_rebind_official(replication, instance_id)
            result = subprocess.run(
                [str(SCRIPT), "--prepare-only"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            attempt = self._only_attempt(output)
            evidence = json.loads((attempt / "campaign_evidence.json").read_bytes())
            replication_evidence = next(
                row for row in evidence["cohorts"] if row["cohort"] == "replication"
            )
            fatal = replication_evidence["fatal_turn_limit_tasks"][0]
            self.assertEqual(fatal["instance_id"], instance_id)
            self.assertEqual(fatal["patch"]["bytes"], 0)
            self.assertEqual(fatal["patch"]["sha256"], hashlib.sha256(b"").hexdigest())

    def test_reuse_completion_preserves_scientific_status_and_promotes_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment, output = self._environment(directory)
            prior = self._seed_latest_reports(output)
            result = subprocess.run(
                [str(SCRIPT), "--reuse-reports"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            latest = (output / "latest").resolve()
            self.assertNotEqual(latest, prior.resolve())
            self.assertTrue(prior.is_dir())
            manifest = json.loads((latest / "run_manifest.json").read_bytes())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["execution_status"], "complete")
            self.assertEqual(
                manifest["scientific_status"],
                "insufficient_split_or_class_support",
            )


if __name__ == "__main__":
    unittest.main()
