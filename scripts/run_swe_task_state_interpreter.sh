#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VLLM_PY=${VLLM_PY:-$ROOT/.venv-vllm/bin/python}
MANIFEST_PY=${MANIFEST_PY:-python3}

PROTOCOL=$ROOT/configs/swe_task_state_interpreter_protocol.json
BEHAVIORAL_PROTOCOL=$ROOT/configs/swe_behavioral_readout_protocol.json
COHORT_MANIFEST=$ROOT/configs/swe_behavioral_n20_cohort.json
PRIMARY_CAMPAIGN=$ROOT/configs/swe_behavioral_campaign.json
REPLICATION_CAMPAIGN=$ROOT/configs/swe_behavioral_replication_campaign.json
ACTION_PROTOCOL=$ROOT/configs/swe_stage_action_probes.json
CHAT_TEMPLATE=$ROOT/configs/qwen3-openai-codex.jinja
MATERIALIZER=$ROOT/scripts/materialize_swe_dense_behavioral_probes.py
ANALYZER=$ROOT/scripts/analyze_swe_task_state_interpreter.py
TASK_STATE_READOUT=$ROOT/scripts/swe_task_state_readout.py
JLENS_RUNNER=$ROOT/scripts/run_jlens_nvfp4.sh
JLENS_PYTHON_RUNNER=$ROOT/scripts/run_jlens_nvfp4.py
MODEL_CHECKPOINT_VERIFIER=$ROOT/scripts/modelopt_checkpoint.py

# These pins make a helper or protocol edit an explicit pipeline revision.
PROTOCOL_SHA256=a6441137828866e8aad9dc547fc0fee37706ece390f503a81fdd9e0f53ed409a
BEHAVIORAL_PROTOCOL_SHA256=ae96a783a6e6736ec6a12fc8d3a7a50b3896c57cf759aa3becd3aa33e257dfa8
COHORT_MANIFEST_SHA256=35dd5ff6c693256da840beb1bfe29a1841d9b87a9efc4c5b6ac7b1d6ec9e2ab9
PRIMARY_CAMPAIGN_SHA256=24d1e885f830c1738294a3e1cdfda50fce307a942bb0badb6f9b1fa55f1de1a0
REPLICATION_CAMPAIGN_SHA256=c861a0f13a9a8225855696a301d51000538b9f78bee1f14f0a028963644708ae
ACTION_PROTOCOL_SHA256=bce204d03608e181456bb5c05a041c4bf4d305f48cb4b4e651ba34460d46d493
CHAT_TEMPLATE_SHA256=c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da
MATERIALIZER_SHA256=a1c9b75b885fd9e50f59363c09d5aa47566eb7bbf7a91fe5aad6a8dccfd1cc5e
ANALYZER_SHA256=279d0a41742e9feeabd3dbd82b73f609326a4942c983cc2eee7125164ebd0594
TASK_STATE_READOUT_SHA256=e8015b51a68ed2001f3351e38d6beaccbcf7dbe507eeeb2e2b608c4f276e2e77
JLENS_RUNNER_SHA256=89763dc20b09394e52f2296654b58408de40c902cb5dddc0a109026964eb2331
JLENS_PYTHON_RUNNER_SHA256=18697bd8adc1159b7228390c526b9c418e87c25c64e7fa5d121d9f95906f7ee5
MODEL_CHECKPOINT_VERIFIER_SHA256=f02b4cdf84d800b13165dfc559c244bdce8c693e939d64d05a5e26351ce57fb6
PROMPTS_SHA256=c89a8dc95455633d1e25c62c3393c063ead3184e84d33a9a00b55ab390b76c27

PRIMARY_RUN_ROOT=${PRIMARY_RUN_ROOT:-$ROOT/runs/swe_behavioral_n10_20260718}
REPLICATION_RUN_ROOT=${REPLICATION_RUN_ROOT:-$ROOT/runs/swe_behavioral_replication_n10_20260718}
OUTPUT_ROOT=${OUT_DIR:-$ROOT/.cache/swe_task_state_interpreter}
PROMPTS=$OUTPUT_ROOT/prompts.json
PROMPTS_SUMMARY=$OUTPUT_ROOT/prompts-summary.json
PUBLIC_REPORT=$OUTPUT_ROOT/public-report.json
ANALYSIS=$OUTPUT_ROOT/analysis.json
CHECKSUM_DIR=$OUTPUT_ROOT/checksums
LOG_DIR=$OUTPUT_ROOT/logs
PROMPTS_CHECKSUM=$CHECKSUM_DIR/prompts.sha256
PROMPTS_SUMMARY_CHECKSUM=$CHECKSUM_DIR/prompts-summary.sha256
PUBLIC_REPORT_CHECKSUM=$CHECKSUM_DIR/public-report.sha256
ANALYSIS_CHECKSUM=$CHECKSUM_DIR/analysis.sha256

LAYERS=24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47
MAX_MODEL_LEN=65536
MAX_NUM_BATCHED_TOKENS=4096
MAMBA_BLOCK_SIZE=1024
KV_OFFLOADING_SIZE=8
GPU_MEMORY_UTILIZATION=0.78
TOP_K=10
SERVER_UNIT=${SERVER_UNIT:-lumo_j_lens_qwen27b}
SERVER_ENDPOINT=${SERVER_ENDPOINT:-http://127.0.0.1:9952/v1/models}

mode=full

usage() {
  cat <<'EOF'
Usage: scripts/run_swe_task_state_interpreter.sh [--prepare-only | --analyze-only]

The default mode reproduces the frozen dense N=20 prompt bundle, replays the
public Jacobian lens on NVFP4 with MTP disabled, and evaluates the selective
task-state interpreter.

Options:
  --prepare-only  Materialize and hash-check all 698 probeable prompts only
  --analyze-only  Reuse hash-checked prompts and public report, then analyze

Environment:
  OUT_DIR                   Artifact directory (default: .cache/swe_task_state_interpreter)
  PRIMARY_RUN_ROOT          Existing development N=10 Qwen Code run root
  REPLICATION_RUN_ROOT      Existing replication N=10 Qwen Code run root
  VLLM_PY                   Pinned vLLM Python interpreter
  SERVER_ENDPOINT           Generation endpoint, which must be offline for replay
  SERVER_UNIT               Generation systemd unit, which must be inactive
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha256() {
  local path=$1
  local expected=$2
  local label=$3
  [[ -s "$path" ]] || {
    echo "$label is missing or empty: $path" >&2
    return 1
  }
  local actual
  actual=$(sha256_file "$path") || {
    echo "could not hash $label: $path" >&2
    return 1
  }
  [[ "$actual" == "$expected" ]] || {
    echo "$label SHA-256 mismatch: expected $expected, got $actual" >&2
    return 1
  }
}

write_sha256_sidecar() {
  local path=$1
  local sidecar=$2
  local temporary=$sidecar.tmp.$$
  printf '%s  %s\n' "$(sha256_file "$path")" "$(basename "$path")" >"$temporary"
  mv -f "$temporary" "$sidecar"
}

verify_sha256_sidecar() {
  local path=$1
  local sidecar=$2
  local label=$3
  [[ -s "$path" && -s "$sidecar" ]] || {
    die "$label or its SHA-256 sidecar is missing"
  }
  local expected target extra actual
  read -r expected target extra <"$sidecar" || die "could not read $label sidecar"
  [[ -z "${extra:-}" && "$target" == "$(basename "$path")" \
    && "$expected" =~ ^[0-9a-f]{64}$ ]] || {
    die "$label sidecar is malformed or names a different file"
  }
  actual=$(sha256_file "$path") || die "could not hash $label"
  [[ "$actual" == "$expected" ]] || {
    die "$label SHA-256 mismatch: expected $expected, got $actual"
  }
}

show_failure_log() {
  local log=$1
  if [[ -f "$log" ]]; then
    echo "--- tail of $log ---" >&2
    tail -80 "$log" >&2 || true
  fi
}

validate_initialization_inputs() {
  command -v "$MANIFEST_PY" >/dev/null 2>&1 || {
    echo "manifest Python is missing: $MANIFEST_PY"
    return 1
  }
  [[ -x "$VLLM_PY" ]] || {
    echo "vLLM Python is missing or not executable: $VLLM_PY"
    return 1
  }
  local path
  for path in \
    "$PROTOCOL" "$BEHAVIORAL_PROTOCOL" "$COHORT_MANIFEST" \
    "$PRIMARY_CAMPAIGN" "$REPLICATION_CAMPAIGN" "$ACTION_PROTOCOL" \
    "$CHAT_TEMPLATE" "$MATERIALIZER" "$ANALYZER" "$TASK_STATE_READOUT" "$JLENS_RUNNER" \
    "$JLENS_PYTHON_RUNNER" "$MODEL_CHECKPOINT_VERIFIER"; do
    [[ -s "$path" ]] || {
      echo "required input is missing or empty: $path"
      return 1
    }
  done

  require_sha256 "$PROTOCOL" "$PROTOCOL_SHA256" "task-state protocol" || return 1
  require_sha256 "$BEHAVIORAL_PROTOCOL" "$BEHAVIORAL_PROTOCOL_SHA256" \
    "behavioral protocol" || return 1
  require_sha256 "$COHORT_MANIFEST" "$COHORT_MANIFEST_SHA256" \
    "N=20 cohort manifest" || return 1
  require_sha256 "$PRIMARY_CAMPAIGN" "$PRIMARY_CAMPAIGN_SHA256" \
    "development campaign" || return 1
  require_sha256 "$REPLICATION_CAMPAIGN" "$REPLICATION_CAMPAIGN_SHA256" \
    "replication campaign" || return 1
  require_sha256 "$ACTION_PROTOCOL" "$ACTION_PROTOCOL_SHA256" \
    "action protocol" || return 1
  require_sha256 "$CHAT_TEMPLATE" "$CHAT_TEMPLATE_SHA256" \
    "chat template" || return 1
  require_sha256 "$MATERIALIZER" "$MATERIALIZER_SHA256" \
    "dense prompt materializer" || return 1
  require_sha256 "$ANALYZER" "$ANALYZER_SHA256" \
    "task-state analyzer" || return 1
  require_sha256 "$TASK_STATE_READOUT" "$TASK_STATE_READOUT_SHA256" \
    "task-state readout" || return 1
  require_sha256 "$JLENS_RUNNER" "$JLENS_RUNNER_SHA256" \
    "J-lens shell runner" || return 1
  require_sha256 "$JLENS_PYTHON_RUNNER" "$JLENS_PYTHON_RUNNER_SHA256" \
    "J-lens Python runner" || return 1
  require_sha256 "$MODEL_CHECKPOINT_VERIFIER" \
    "$MODEL_CHECKPOINT_VERIFIER_SHA256" "model checkpoint verifier" || return 1

  "$MANIFEST_PY" - \
    "$PROTOCOL" "$BEHAVIORAL_PROTOCOL_SHA256" "$PROMPTS_SHA256" "$LAYERS" \
    "$MAX_MODEL_LEN" "$MAX_NUM_BATCHED_TOKENS" "$MAMBA_BLOCK_SIZE" \
    "$KV_OFFLOADING_SIZE" <<'PY'
import json
from pathlib import Path
import re
import sys

(
    protocol_arg,
    behavioral_sha,
    prompt_sha,
    layers_arg,
    max_model_len,
    max_batched_tokens,
    mamba_block_size,
    kv_offloading_size,
) = sys.argv[1:]
protocol = json.loads(Path(protocol_arg).read_bytes())
if protocol.get("schema_version") != 1 or protocol.get("id") != "swe-task-state-interpreter-v1":
    raise SystemExit("task-state protocol identity changed")
pins = protocol.get("input_pins", {})
if pins.get("behavioral_protocol_sha256") != behavioral_sha:
    raise SystemExit("task-state behavioral protocol pin changed")
if pins.get("prompt_bundle_sha256") != prompt_sha:
    raise SystemExit("task-state dense prompt pin changed")
if pins.get("public_report_sha256") is not None:
    value = pins["public_report_sha256"]
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SystemExit("task-state public report pin is malformed")
expected_model = {
    "repo_id": "nvidia/Qwen3.6-27B-NVFP4",
    "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
    "config_sha256": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
    "index_sha256": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
}
expected_lens = {
    "repo_id": "neuronpedia/jacobian-lens",
    "revision": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
    "sha256": "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1",
    "n_prompts": 1000,
}
expected_runtime = {
    "enforce_eager": True,
    "mtp_enabled": False,
    "max_model_len": int(max_model_len),
    "max_num_batched_tokens": int(max_batched_tokens),
    "mamba_block_size": int(mamba_block_size),
    "kv_cache_dtype": "fp8_e4m3",
    "kv_offloading_size": float(kv_offloading_size),
    "kv_offloading_backend": "native",
    "stream_final_only": True,
}
if pins.get("model") != expected_model:
    raise SystemExit("task-state model pin differs from the NVFP4 replay contract")
if pins.get("public_lens") != expected_lens:
    raise SystemExit("task-state public lens pin differs from the replay contract")
if pins.get("replay_runtime") != expected_runtime:
    raise SystemExit("task-state replay runtime pin differs from the runner arguments")
layers = [int(item) for item in layers_arg.split(",")]
if protocol.get("feature_contract", {}).get("layers") != layers:
    raise SystemExit("task-state feature layers differ from the replay layer band")
PY
}

validate_prepared_bundle() {
  require_sha256 "$PROMPTS" "$PROMPTS_SHA256" "dense N=20 prompt bundle" \
    || return 1
  "$MANIFEST_PY" - \
    "$PROMPTS" "$PROMPTS_SUMMARY" "$PROMPTS_SHA256" \
    "$COHORT_MANIFEST_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

prompts_arg, summary_arg, prompt_sha, cohort_sha = sys.argv[1:]
prompts_path = Path(prompts_arg)
summary_path = Path(summary_arg)
if not summary_path.is_file():
    raise SystemExit("dense prompt summary is missing")
summary = json.loads(summary_path.read_bytes())
if {
    "schema_version": summary.get("schema_version"),
    "kind": summary.get("kind"),
    "prompt_count": summary.get("prompt_count"),
    "task_count": summary.get("task_count"),
    "cohort_count": summary.get("cohort_count"),
    "global_request_count": summary.get("global_request_count"),
    "prompt_bundle_sha256": summary.get("prompt_bundle_sha256"),
    "cohort_manifest_sha256": summary.get("cohort_manifest_sha256"),
} != {
    "schema_version": 1,
    "kind": "swe_verified_behavioral_probe_combination",
    "prompt_count": 698,
    "task_count": 20,
    "cohort_count": 2,
    "global_request_count": 699,
    "prompt_bundle_sha256": prompt_sha,
    "cohort_manifest_sha256": cohort_sha,
}:
    raise SystemExit("dense prompt summary differs from the frozen all-probeable contract")
digest = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
if digest != prompt_sha:
    raise SystemExit("dense prompt bytes changed while validating the summary")
PY
}

materialize_dense_bundle() {
  [[ -d "$PRIMARY_RUN_ROOT" ]] || die "development run root is missing: $PRIMARY_RUN_ROOT"
  [[ -d "$REPLICATION_RUN_ROOT" ]] || {
    die "replication run root is missing: $REPLICATION_RUN_ROOT"
  }
  rm -f \
    "$PROMPTS" "$PROMPTS_SUMMARY" "$PROMPTS_CHECKSUM" \
    "$PROMPTS_SUMMARY_CHECKSUM" "$PUBLIC_REPORT" "$PUBLIC_REPORT_CHECKSUM" \
    "$ANALYSIS" "$ANALYSIS_CHECKSUM"
  echo "[prepare] reproducing the frozen dense N=20 all-probeable bundle"
  if ! "$VLLM_PY" "$MATERIALIZER" \
    --cohort "$PRIMARY_CAMPAIGN" "$PRIMARY_RUN_ROOT" \
    --cohort "$REPLICATION_CAMPAIGN" "$REPLICATION_RUN_ROOT" \
    --cohort-manifest "$COHORT_MANIFEST" \
    --action-protocol "$ACTION_PROTOCOL" \
    --template "$CHAT_TEMPLATE" \
    --require-official-outcomes \
    --all-probeable \
    --output "$PROMPTS" \
    --summary "$PROMPTS_SUMMARY" \
    >"$LOG_DIR/materialize.log" 2>&1; then
    show_failure_log "$LOG_DIR/materialize.log"
    die "dense prompt materialization failed"
  fi
  if ! validate_prepared_bundle >>"$LOG_DIR/materialize.log" 2>&1; then
    show_failure_log "$LOG_DIR/materialize.log"
    die "materialized prompts do not match the frozen dense bundle"
  fi
  write_sha256_sidecar "$PROMPTS" "$PROMPTS_CHECKSUM"
  write_sha256_sidecar "$PROMPTS_SUMMARY" "$PROMPTS_SUMMARY_CHECKSUM"
  verify_sha256_sidecar "$PROMPTS" "$PROMPTS_CHECKSUM" "dense prompts"
  verify_sha256_sidecar \
    "$PROMPTS_SUMMARY" "$PROMPTS_SUMMARY_CHECKSUM" "dense prompt summary"
}

ensure_serving_endpoint_offline() {
  if command -v systemctl >/dev/null 2>&1 \
    && systemctl is-active --quiet "$SERVER_UNIT"; then
    die "generation server unit is active: $SERVER_UNIT"
  fi
  command -v curl >/dev/null 2>&1 || {
    die "curl is required to verify that the generation endpoint is offline"
  }
  if curl --silent --output /dev/null --connect-timeout 1 --max-time 2 \
    "$SERVER_ENDPOINT"; then
    die "generation endpoint is live: $SERVER_ENDPOINT"
  fi
  echo "[preflight] generation endpoint is offline: $SERVER_ENDPOINT"
}

run_public_replay() {
  rm -f \
    "$PUBLIC_REPORT" "$PUBLIC_REPORT_CHECKSUM" "$ANALYSIS" \
    "$ANALYSIS_CHECKSUM" "$LOG_DIR/public.log"
  ensure_serving_endpoint_offline
  validate_prepared_bundle || die "dense prompt bundle changed before replay"
  verify_sha256_sidecar "$PROMPTS" "$PROMPTS_CHECKSUM" "dense prompts"
  verify_sha256_sidecar \
    "$PROMPTS_SUMMARY" "$PROMPTS_SUMMARY_CHECKSUM" "dense prompt summary"
  echo "[replay] public Jacobian lens on the dense NVFP4 prompt bundle"
  set +e
  "$JLENS_RUNNER" \
    --lens-kind public \
    --prompts-file "$PROMPTS" \
    --layers "$LAYERS" \
    --positions=-1 \
    --top-k "$TOP_K" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --mamba-block-size "$MAMBA_BLOCK_SIZE" \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e4m3 \
    --kv-offloading-size "$KV_OFFLOADING_SIZE" \
    --kv-offloading-backend native \
    --stream-final-only \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --output "$PUBLIC_REPORT" \
    >/dev/null 2>"$LOG_DIR/public.log"
  local status=$?
  set -e
  if [[ ! -s "$PUBLIC_REPORT" ]]; then
    show_failure_log "$LOG_DIR/public.log"
    die "public replay exited $status without a fresh report"
  fi
  if ((status != 0 && status != 1)); then
    show_failure_log "$LOG_DIR/public.log"
    die "public replay emitted a report but exited with unsupported status $status"
  fi
  write_sha256_sidecar "$PUBLIC_REPORT" "$PUBLIC_REPORT_CHECKSUM"
  verify_sha256_sidecar \
    "$PUBLIC_REPORT" "$PUBLIC_REPORT_CHECKSUM" "public replay report"
  echo "[replay] accepted terminal report with runner exit status $status"
}

verify_analysis_inputs() {
  validate_prepared_bundle
  verify_sha256_sidecar "$PROMPTS" "$PROMPTS_CHECKSUM" "dense prompts"
  verify_sha256_sidecar \
    "$PROMPTS_SUMMARY" "$PROMPTS_SUMMARY_CHECKSUM" "dense prompt summary"
  verify_sha256_sidecar \
    "$PUBLIC_REPORT" "$PUBLIC_REPORT_CHECKSUM" "public replay report"

  # The development protocol permits a fresh report; its sidecar and the
  # analysis input record bind the exact bytes used. Publication must pin it.
  "$MANIFEST_PY" - "$PROTOCOL" "$PUBLIC_REPORT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

protocol_arg, report_arg = sys.argv[1:]
pin = json.loads(Path(protocol_arg).read_bytes())["input_pins"]["public_report_sha256"]
observed = hashlib.sha256(Path(report_arg).read_bytes()).hexdigest()
if pin is not None and observed != pin:
    raise SystemExit(
        f"public report differs from protocol pin: expected {pin}, got {observed}"
    )
PY
}

validate_analysis_output() {
  "$MANIFEST_PY" - \
    "$ANALYSIS" "$PROMPTS" "$PUBLIC_REPORT" "$PROTOCOL" \
    "$BEHAVIORAL_PROTOCOL" "$ANALYZER" "$TASK_STATE_READOUT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    analysis_arg,
    prompts_arg,
    report_arg,
    protocol_arg,
    behavioral_arg,
    analyzer_arg,
    task_state_readout_arg,
) = sys.argv[1:]
analysis = json.loads(Path(analysis_arg).read_bytes())
digest = lambda value: hashlib.sha256(Path(value).read_bytes()).hexdigest()
expected_inputs = {
    "prompts": digest(prompts_arg),
    "public_report": digest(report_arg),
    "protocol": digest(protocol_arg),
    "behavioral_protocol": digest(behavioral_arg),
    "analyzer_implementation": digest(analyzer_arg),
    "task_state_readout_implementation": digest(task_state_readout_arg),
}
if analysis.get("schema_version") != 1:
    raise SystemExit("task-state analysis schema changed")
if analysis.get("kind") != "swe_task_state_selective_interpreter_analysis":
    raise SystemExit("task-state analysis kind changed")
if analysis.get("inputs") != expected_inputs:
    raise SystemExit("task-state analysis is not bound to the supplied artifacts")
if analysis.get("interpretation_scope") != (
    "observable next-action class only; this is not a decoder of hidden chain-of-thought"
):
    raise SystemExit("task-state interpretation scope changed")
PY
}

run_analysis() {
  verify_analysis_inputs
  rm -f "$ANALYSIS" "$ANALYSIS_CHECKSUM" "$LOG_DIR/analyze.log"
  echo "[analyze] repository-held-out calibrated selective interpreter"
  if ! "$VLLM_PY" "$ANALYZER" \
    --prompts "$PROMPTS" \
    --public-report "$PUBLIC_REPORT" \
    --protocol "$PROTOCOL" \
    --behavioral-protocol "$BEHAVIORAL_PROTOCOL" \
    --output "$ANALYSIS" \
    >"$LOG_DIR/analyze.log" 2>&1; then
    show_failure_log "$LOG_DIR/analyze.log"
    die "task-state interpreter analysis failed"
  fi
  [[ -s "$ANALYSIS" ]] || die "analyzer did not emit a fresh analysis"
  if ! validate_analysis_output >>"$LOG_DIR/analyze.log" 2>&1; then
    show_failure_log "$LOG_DIR/analyze.log"
    die "task-state analysis does not bind the exact replay evidence"
  fi
  write_sha256_sidecar "$ANALYSIS" "$ANALYSIS_CHECKSUM"
  verify_sha256_sidecar "$ANALYSIS" "$ANALYSIS_CHECKSUM" "task-state analysis"
}

while (($#)); do
  case "$1" in
    --prepare-only)
      [[ "$mode" == full ]] || die "--prepare-only and --analyze-only are mutually exclusive"
      mode=prepare
      ;;
    --analyze-only)
      [[ "$mode" == full ]] || die "--prepare-only and --analyze-only are mutually exclusive"
      mode=analyze
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
  shift
done

[[ ! -L "$OUTPUT_ROOT" ]] || die "output root must not be a symlink: $OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT" "$CHECKSUM_DIR" "$LOG_DIR"
LOCK_DIR=$OUTPUT_ROOT/.interpreter.lock
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  die "another task-state interpreter run owns $LOCK_DIR"
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if ! validate_initialization_inputs >"$LOG_DIR/initialization.log" 2>&1; then
  show_failure_log "$LOG_DIR/initialization.log"
  die "task-state interpreter initialization failed"
fi

if [[ "$mode" != analyze ]]; then
  materialize_dense_bundle
fi
if [[ "$mode" == prepare ]]; then
  echo "prepared dense task-state prompts: $PROMPTS"
  exit 0
fi
if [[ "$mode" == full ]]; then
  run_public_replay
fi
run_analysis

echo "task-state analysis: $ANALYSIS"
