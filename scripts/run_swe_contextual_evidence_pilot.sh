#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VLLM_PY=${VLLM_PY:-$ROOT/.venv-vllm/bin/python}
OUTPUT_ROOT=${OUT_DIR:-$ROOT/.cache/swe_contextual_evidence_pilot}
MANIFEST_PY=${MANIFEST_PY:-python3}

PROTOCOL=$ROOT/configs/swe_contextual_evidence_protocol.json
MATERIALIZER=$ROOT/scripts/materialize_swe_contextual_evidence.py
ANALYZER=$ROOT/scripts/analyze_swe_contextual_evidence.py
JLENS_RUNNER=${JLENS_RUNNER:-$ROOT/scripts/run_jlens_nvfp4.sh}
JLENS_PYTHON_RUNNER=$ROOT/scripts/run_jlens_nvfp4.py
MODEL_CHECKPOINT_VERIFIER=$ROOT/scripts/modelopt_checkpoint.py

PROTOCOL_SHA256=d2e32b4aa027ed387f1c8105046b2a102c3cfbe51d7879dbe86ebd804b58160a
PROMPTS_SHA256=0662a327c4d13e4f359935e5766df53803939917095d00f1d5122865b425497d
JLENS_RUNNER_SHA256=89763dc20b09394e52f2296654b58408de40c902cb5dddc0a109026964eb2331
JLENS_PYTHON_RUNNER_SHA256=18697bd8adc1159b7228390c526b9c418e87c25c64e7fa5d121d9f95906f7ee5
MODEL_CHECKPOINT_VERIFIER_SHA256=f02b4cdf84d800b13165dfc559c244bdce8c693e939d64d05a5e26351ce57fb6

MODEL_REPO=nvidia/Qwen3.6-27B-NVFP4
MODEL_REVISION=0893e1606ff3d5f97a441f405d5fc541a6bdf404
MODEL_CONFIG_SHA256=c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338
MODEL_INDEX_SHA256=7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2
PUBLIC_LENS_REPO=neuronpedia/jacobian-lens
PUBLIC_LENS_REVISION=a4114d7752d11eb546e6cf372213d7e75526d3a1
PUBLIC_LENS_SHA256=1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1
NF4_LENS_PATH=$ROOT/.cache/Qwen3.6-27B-jlens-nf4-n10-fp32.pt
NF4_LENS_SHA256=54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f
NF4_PROVENANCE_PATH=$ROOT/.cache/Qwen3.6-27B-jlens-nf4-n10-fp32.provenance.json
NF4_PROVENANCE_SHA256=08dfe022cc74bab226308dcd77a3901ec18806f8832dc8dec46524374fa3b1e7
NATIVE_LENS_PATH=$ROOT/.cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt
NATIVE_LENS_SHA256=82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057
NATIVE_PROVENANCE_PATH=$ROOT/.cache/nvfp4_ste_fit/final-mean/metadata.json
NATIVE_PROVENANCE_SHA256=289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601
NATIVE_STATE_PATH=$ROOT/.cache/nvfp4_ste_fit/state.json
NATIVE_STATE_SHA256=f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6

LAYERS=24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47
MAX_MODEL_LEN=65536
MAX_NUM_BATCHED_TOKENS=4096
MAMBA_BLOCK_SIZE=1024
KV_OFFLOADING_SIZE=8
GPU_MEMORY_UTILIZATION=0.78
TOP_K=10
TORCH_VERSION=2.11.0+cu130
VLLM_VERSION=0.23.0
TRANSFORMERS_VERSION=5.12.1
HUGGINGFACE_HUB_VERSION=1.21.0
TRITON_VERSION=3.6.0
PYTHON_VERSION=3.12.13
SERVER_UNIT=${SERVER_UNIT:-lumo_j_lens_qwen27b}
SERVER_ENDPOINT=${SERVER_ENDPOINT:-http://127.0.0.1:9952/v1/models}

usage() {
  cat <<'EOF'
Usage: scripts/run_swe_contextual_evidence_pilot.sh [--prepare-only | --public-only]

The default mode materializes the frozen paired prompt bundle, replays the
public, NF4, and native NVFP4-STE lenses sequentially, and analyzes all three.

Options:
  --prepare-only  Validate and materialize the exact 24-prompt bundle only
  --public-only   Replay and analyze the public lens plus ordinary logit control

Environment:
  OUT_DIR          Immutable-attempt output root
  VLLM_PY          Pinned vLLM Python interpreter
  JLENS_RUNNER     J-lens shell runner (must match the pinned runner hash)
  SERVER_ENDPOINT  Generation endpoint, which must be offline before replay
EOF
}

die() {
  FAILURE_REASON=$*
  echo "ERROR: $*" >&2
  exit 1
}

show_failure_log() {
  local log=$1
  if [[ -f "$log" ]]; then
    echo "--- tail of $log ---" >&2
    tail -80 "$log" >&2 || true
  fi
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
  actual=$(sha256_file "$path") || return 1
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
  [[ -s "$path" && -s "$sidecar" ]] || return 1
  local expected target extra actual
  read -r expected target extra <"$sidecar" || return 1
  [[ -z "${extra:-}" && "$target" == "$(basename "$path")" \
    && "$expected" =~ ^[0-9a-f]{64}$ ]] || return 1
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] || {
    echo "$label SHA-256 mismatch: expected $expected, got $actual" >&2
    return 1
  }
}

validate_initialization_inputs() {
  [[ -x "$VLLM_PY" ]] || {
    echo "vLLM Python is missing or not executable: $VLLM_PY"
    return 1
  }
  [[ -x "$JLENS_RUNNER" ]] || {
    echo "J-lens runner is missing or not executable: $JLENS_RUNNER"
    return 1
  }
  local path
  for path in "$PROTOCOL" "$MATERIALIZER" "$ANALYZER" \
    "$JLENS_PYTHON_RUNNER" "$MODEL_CHECKPOINT_VERIFIER"; do
    [[ -s "$path" ]] || {
      echo "required input is missing or empty: $path"
      return 1
    }
  done
  require_sha256 "$PROTOCOL" "$PROTOCOL_SHA256" "frozen contextual protocol" \
    || return 1
  require_sha256 "$JLENS_RUNNER" "$JLENS_RUNNER_SHA256" "J-lens shell runner" \
    || return 1
  require_sha256 "$JLENS_PYTHON_RUNNER" "$JLENS_PYTHON_RUNNER_SHA256" \
    "J-lens Python runner" || return 1
  require_sha256 "$MODEL_CHECKPOINT_VERIFIER" "$MODEL_CHECKPOINT_VERIFIER_SHA256" \
    "model checkpoint verifier" || return 1

  "$VLLM_PY" - "$PROTOCOL" \
    "$MODEL_REPO" "$MODEL_REVISION" "$MODEL_CONFIG_SHA256" "$MODEL_INDEX_SHA256" \
    "$PUBLIC_LENS_REPO" "$PUBLIC_LENS_REVISION" "$PUBLIC_LENS_SHA256" \
    "$NF4_LENS_PATH" "$NF4_LENS_SHA256" "$NF4_PROVENANCE_SHA256" \
    "$NATIVE_LENS_PATH" "$NATIVE_LENS_SHA256" \
    "$NATIVE_PROVENANCE_SHA256" "$NATIVE_STATE_SHA256" "$LAYERS" <<'PY'
import json
from pathlib import Path
import sys

(
    protocol_arg,
    model_repo,
    model_revision,
    model_config_sha,
    model_index_sha,
    public_repo,
    public_revision,
    public_sha,
    nf4_path,
    nf4_sha,
    nf4_provenance_sha,
    native_path,
    native_sha,
    native_provenance_sha,
    native_state_sha,
    layers_arg,
) = sys.argv[1:]
protocol = json.loads(Path(protocol_arg).read_bytes())
expected_model = {
    "repo_id": model_repo,
    "revision": model_revision,
    "config_sha256": model_config_sha,
    "index_sha256": model_index_sha,
}
if protocol.get("pins", {}).get("model") != expected_model:
    raise SystemExit("protocol model pin differs from the replay contract")
lenses = protocol["pins"]["lenses"]
expected_lenses = {
    "public": {
        "repo_id": public_repo,
        "revision": public_revision,
        "sha256": public_sha,
        "n_prompts": 1000,
    },
    "nf4": {
        "path": str(Path(nf4_path).resolve().relative_to(Path(protocol_arg).resolve().parents[1])),
        "sha256": nf4_sha,
        "provenance_sha256": nf4_provenance_sha,
        "n_prompts": 10,
    },
    "native_nvfp4_ste": {
        "path": str(Path(native_path).resolve().relative_to(Path(protocol_arg).resolve().parents[1])),
        "sha256": native_sha,
        "provenance_sha256": native_provenance_sha,
        "state_sha256": native_state_sha,
        "n_prompts": 10,
    },
}
if lenses != expected_lenses:
    raise SystemExit("protocol lens pins differ from the replay contract")
layers = [int(item) for item in layers_arg.split(",")]
if protocol.get("fixed_layer_band", {}).get("layers") != layers:
    raise SystemExit("protocol fixed layer band differs from the replay contract")
context = protocol.get("prompt_context", {})
if context != {
    "max_model_len": 65536,
    "reserved_generation_tokens": 1,
    "maximum_prompt_tokens": 65535,
    "positions": [-1],
}:
    raise SystemExit("protocol prompt context differs from the replay contract")
PY
}

validate_local_lenses() {
  require_sha256 "$NF4_LENS_PATH" "$NF4_LENS_SHA256" "NF4 lens" || return 1
  require_sha256 "$NF4_PROVENANCE_PATH" "$NF4_PROVENANCE_SHA256" "NF4 provenance" \
    || return 1
  require_sha256 "$NATIVE_LENS_PATH" "$NATIVE_LENS_SHA256" "native NVFP4 lens" \
    || return 1
  require_sha256 "$NATIVE_PROVENANCE_PATH" "$NATIVE_PROVENANCE_SHA256" \
    "native NVFP4 provenance" || return 1
  require_sha256 "$NATIVE_STATE_PATH" "$NATIVE_STATE_SHA256" "native NVFP4 state" \
    || return 1
}

run_logged() {
  local label=$1
  local log=$2
  shift 2
  local outputs=()
  while (($#)) && [[ "$1" != -- ]]; do
    outputs+=("$1")
    shift
  done
  [[ "${1:-}" == -- ]] || die "$label has no command delimiter"
  shift
  CURRENT_PHASE=$label
  local status_file=$OUT_DIR/$label.exit_status
  local wall_file=$OUT_DIR/$label.wall_seconds
  local output
  rm -f "$log" "$status_file" "$wall_file" "${outputs[@]}"
  local started ended status
  started=$(date +%s)
  set +e
  "$@" >"$log" 2>&1
  status=$?
  set -e
  ended=$(date +%s)
  printf '%s\n' "$status" >"$status_file"
  printf '%s\n' "$((ended - started))" >"$wall_file"
  if ((status != 0)); then
    show_failure_log "$log"
    die "$label failed with exit status $status"
  fi
  for output in "${outputs[@]}"; do
    [[ -s "$output" ]] || die "$label did not emit a fresh nonempty output: $output"
  done
}

validate_materialization() {
  "$VLLM_PY" - "$ROOT" "$PROTOCOL" "$PROMPTS" "$MATERIALIZATION" \
    "$PROTOCOL_SHA256" "$PROMPTS_SHA256" "$MAX_MODEL_LEN" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root_arg, protocol_arg, prompts_arg, manifest_arg, protocol_sha, prompt_sha, max_len = sys.argv[1:]
root = Path(root_arg)
sys.path.insert(0, str(root / "scripts"))
import analyze_swe_contextual_evidence as analyzer

def read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), raw

protocol_value, protocol_raw = read(protocol_arg)
prompts_value, prompts_raw = read(prompts_arg)
manifest_value, manifest_raw = read(manifest_arg)
if hashlib.sha256(protocol_raw).hexdigest() != protocol_sha:
    raise SystemExit("protocol bytes changed during materialization")
if hashlib.sha256(prompts_raw).hexdigest() != prompt_sha:
    raise SystemExit("materialized prompt bytes differ from the frozen bundle")
protocol = analyzer.validate_protocol(protocol_value, protocol_sha256=protocol_sha)
manifest = analyzer.validate_manifest(
    manifest_value,
    protocol=protocol,
    protocol_sha256=protocol_sha,
    manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
)
prompts = analyzer.validate_prompt_bundle(prompts_value, protocol=protocol, manifest=manifest)
if len(prompts["prompts"]) != 24 or len(protocol["tasks"]) != 12:
    raise SystemExit("frozen contextual bundle must contain 12 pairs / 24 prompts")
limit = int(max_len) - 1
oversized = [
    (prompt["id"], len(prompt["token_ids"]))
    for prompt in prompts["prompts"]
    if len(prompt["token_ids"]) > limit
]
if oversized:
    raise SystemExit(f"contextual prompts exceed {limit} tokens: {oversized}")
PY
}

ensure_capture_resources_idle() {
  local label=$1
  local phase=${label}_preflight
  local log=$LOG_DIR/$phase.log
  local status_file=$OUT_DIR/$phase.exit_status
  CURRENT_PHASE=$phase
  rm -f "$log" "$status_file"
  if ! {
    if command -v systemctl >/dev/null 2>&1 \
      && systemctl is-active --quiet "$SERVER_UNIT"; then
      echo "generation server unit is active: $SERVER_UNIT"
      false
    elif ! command -v curl >/dev/null 2>&1; then
      echo "curl is required to verify that the generation endpoint is offline"
      false
    elif curl --silent --output /dev/null --connect-timeout 1 --max-time 2 \
      "$SERVER_ENDPOINT"; then
      echo "generation endpoint is live: $SERVER_ENDPOINT"
      false
    elif ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "nvidia-smi is required for exclusive capture preflight"
      false
    else
      local processes total_memory
      processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits) || false
      total_memory=$(nvidia-smi --query-gpu=memory.used \
        --format=csv,noheader,nounits) || false
      "$MANIFEST_PY" - "$processes" "$total_memory" <<'PY'
import csv
import io
from pathlib import PurePosixPath
import re
import sys

process_rows, total_rows = sys.argv[1:]
caps = {"gnome-control-center": 96, "ptyxis": 64, "nautilus": 64}
seen = set()
aggregate = 0
for row_number, row in enumerate(csv.reader(io.StringIO(process_rows)), 1):
    if not row:
        continue
    if len(row) != 3:
        raise SystemExit(f"malformed GPU process row {row_number}")
    pid, process_name, memory = (item.strip() for item in row)
    basename = PurePosixPath(process_name).name
    if re.fullmatch(r"[1-9][0-9]*", pid) is None or basename not in caps:
        raise SystemExit(f"non-idle GPU compute process: pid={pid}, name={process_name!r}")
    if basename in seen or re.fullmatch(r"[0-9]+", memory) is None:
        raise SystemExit(f"invalid or duplicate idle GPU process: {basename}")
    observed = int(memory)
    if observed > caps[basename]:
        raise SystemExit(f"idle GPU process {basename} uses {observed} MiB > {caps[basename]} MiB")
    seen.add(basename)
    aggregate += observed
if aggregate > 192:
    raise SystemExit(f"idle GPU process aggregate uses {aggregate} MiB > 192 MiB")
values = [line.strip() for line in total_rows.splitlines() if line.strip()]
if len(values) != 1 or re.fullmatch(r"[0-9]+", values[0]) is None:
    raise SystemExit("total GPU memory query is not one integer")
total = int(values[0])
if total < aggregate or total > 640:
    raise SystemExit(f"GPU memory is not idle: total={total} MiB, processes={aggregate} MiB")
print(f"exclusive replay preflight passed: {len(seen)} allowed processes, {total} MiB total")
PY
    fi
  } >"$log" 2>&1; then
    printf 'capture-resource-busy-or-unverifiable\n' >"$status_file"
    show_failure_log "$log"
    die "$label model load refused because capture resources are not exclusive"
  fi
  printf '0\n' >"$status_file"
}

validate_json_report() {
  local report=$1
  local lens_label=$2
  local process_status=$3
  "$VLLM_PY" - "$ROOT" "$PROTOCOL" "$PROMPTS" "$MATERIALIZATION" \
    "$report" "$lens_label" "$process_status" \
    "$MODEL_REPO" "$MODEL_REVISION" "$MODEL_CONFIG_SHA256" "$MODEL_INDEX_SHA256" \
    "$LAYERS" "$MAX_MODEL_LEN" "$MAX_NUM_BATCHED_TOKENS" "$MAMBA_BLOCK_SIZE" \
    "$KV_OFFLOADING_SIZE" "$GPU_MEMORY_UTILIZATION" "$TOP_K" \
    "$TORCH_VERSION" "$VLLM_VERSION" "$TRANSFORMERS_VERSION" \
    "$HUGGINGFACE_HUB_VERSION" "$TRITON_VERSION" "$PYTHON_VERSION" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    root_arg, protocol_arg, prompts_arg, manifest_arg, report_arg, lens_label,
    process_status, model_repo, model_revision, model_config_sha, model_index_sha,
    layers_arg, max_len, max_batched, mamba_block, offload_size, gpu_util, top_k,
    torch_version, vllm_version, transformers_version, hub_version, triton_version,
    python_version,
) = sys.argv[1:]
root = Path(root_arg)
sys.path.insert(0, str(root / "scripts"))
import analyze_swe_contextual_evidence as analyzer
from modelopt_checkpoint import PINNED_METADATA_SHA256, PINNED_SHARDS

def read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), raw

protocol_value, protocol_raw = read(protocol_arg)
prompts_value, _ = read(prompts_arg)
manifest_value, manifest_raw = read(manifest_arg)
report_value, _ = read(report_arg)
protocol_sha = hashlib.sha256(protocol_raw).hexdigest()
protocol = analyzer.validate_protocol(protocol_value, protocol_sha256=protocol_sha)
manifest = analyzer.validate_manifest(
    manifest_value,
    protocol=protocol,
    protocol_sha256=protocol_sha,
    manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
)
prompts = analyzer.validate_prompt_bundle(prompts_value, protocol=protocol, manifest=manifest)
validated = analyzer.validate_report(
    report_value, label=lens_label, protocol=protocol, prompts=prompts
)
expected_status = "passed" if process_status == "0" else "failed"
if process_status not in {"0", "1"} or report_value.get("status") != expected_status:
    raise SystemExit("report terminal status does not correspond to runner exit status")

expected_runtime = {
    "mtp_enabled": False,
    "enforce_eager": True,
    "language_model_only": True,
    "max_model_len": int(max_len),
    "max_num_batched_tokens": int(max_batched),
    "mamba_block_size": int(mamba_block),
    "enable_prefix_caching": True,
    "kv_cache_dtype": "fp8_e4m3",
    "kv_offloading_size": float(offload_size),
    "kv_offloading_backend": "native",
    "stream_final_only": True,
    "gpu_memory_utilization": float(gpu_util),
    "capture_adapter": "vLLM apply_model forward hooks",
    "transport_dtype": "torch.float32",
    "readout_dtype": "torch.bfloat16",
    "timing_scope": "artifact resolution and validation through readout",
}
runtime = dict(report_value.get("runtime", {}))
model_load = runtime.pop("model_load_seconds", None)
if runtime != expected_runtime or not isinstance(model_load, (int, float)) or model_load <= 0:
    raise SystemExit("report runtime differs from the frozen MTP-off eager replay contract")

model = report_value.get("model", {})
expected_model = {
    "repo_id": model_repo,
    "revision": model_revision,
    "config_sha256": model_config_sha,
    "index_sha256": model_index_sha,
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
}
if {key: model.get(key) for key in expected_model} != expected_model:
    raise SystemExit("report model differs from the pinned NVFP4 checkpoint")
expected_checkpoint = {
    "policy": "ModelOptCheckpoint(strict_pinned=True)",
    "validated_before_model_load": True,
    "validated_after_evaluation": True,
    "metadata_sha256": dict(PINNED_METADATA_SHA256),
    "shards": {
        filename: {"bytes": size, "sha256": sha}
        for filename, (size, sha) in PINNED_SHARDS.items()
    },
}
if model.get("checkpoint_integrity") != expected_checkpoint:
    raise SystemExit("report lacks full before/after checkpoint integrity evidence")
host = report_value.get("host", {})
if host.get("python") != python_version or host.get("packages") != {
    "huggingface-hub": hub_version,
    "torch": torch_version,
    "transformers": transformers_version,
    "triton": triton_version,
    "vllm": vllm_version,
}:
    raise SystemExit("report host packages differ from the pinned environment")
gpu = host.get("gpu", {})
if gpu.get("name") != "NVIDIA GeForce RTX 5090" or gpu.get("compute_capability") != "12.0":
    raise SystemExit("report was not captured on the pinned RTX 5090")
expected_layers = [int(item) for item in layers_arg.split(",")]
for experiment in report_value.get("experiments", []):
    if [row.get("layer") for row in experiment.get("layers", [])] != expected_layers:
        raise SystemExit("report layer grid differs from the frozen band")
    for row in experiment["layers"]:
        position = row["positions"][0]
        for method in ("jacobian_lens", "logit_lens"):
            readout = position[method]
            for field in ("tokens", "token_ids", "scores"):
                if not isinstance(readout.get(field), list) or len(readout[field]) != int(top_k):
                    raise SystemExit(f"report {method}.{field} differs from top-k={top_k}")
if validated["runtime"] != report_value["runtime"]:
    raise SystemExit("analyzer/runtime validation disagreement")
PY
}

run_lens() {
  local label=$1
  local report=$2
  local log=$3
  shift 3
  CURRENT_PHASE=$label
  local status_file=$OUT_DIR/$label.exit_status
  local wall_file=$OUT_DIR/$label.wall_seconds
  local stdout_file=$OUT_DIR/.$label.stdout
  local checksum=$CHECKSUM_DIR/$label-report.sha256
  ensure_capture_resources_idle "$label"
  CURRENT_PHASE=$label
  rm -f "$report" "$log" "$status_file" "$wall_file" "$stdout_file" "$checksum"
  local started ended status
  started=$(date +%s)
  set +e
  "$@" --output "$report" >"$stdout_file" 2>"$log"
  status=$?
  set -e
  ended=$(date +%s)
  printf '%s\n' "$status" >"$status_file"
  printf '%s\n' "$((ended - started))" >"$wall_file"
  if [[ ! -s "$report" ]]; then
    [[ ! -s "$stdout_file" ]] || tail -80 "$stdout_file" >>"$log" || true
    printf 'missing-report-after-process-status-%s\n' "$status" >"$status_file"
    show_failure_log "$log"
    die "$label exited $status without a fresh terminal report"
  fi
  if ((status != 0 && status != 1)); then
    printf 'unsupported-process-status-%s\n' "$status" >"$status_file"
    show_failure_log "$log"
    die "$label emitted a report but exited with unsupported status $status"
  fi
  if ! validate_json_report "$report" "$label" "$status" >>"$log" 2>&1; then
    printf 'invalid-report-after-process-status-%s\n' "$status" >"$status_file"
    show_failure_log "$log"
    die "$label report violates the frozen contextual replay contract"
  fi
  rm -f "$stdout_file"
  write_sha256_sidecar "$report" "$checksum"
  verify_sha256_sidecar "$report" "$checksum" "$label report" \
    || die "$label report checksum verification failed"
  echo "[$label] accepted runner exit $status with terminal report: $report"
}

validate_analysis_outputs() {
  "$VLLM_PY" - "$ANALYSIS" "$CARDS" "$PROTOCOL" "$MATERIALIZATION" \
    "$PROMPTS" "$RUN_MODE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

analysis_arg, cards_arg, protocol_arg, manifest_arg, prompts_arg, mode = sys.argv[1:]
analysis = json.loads(Path(analysis_arg).read_bytes())
cards = json.loads(Path(cards_arg).read_bytes())
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
if analysis.get("schema_version") != 1 or analysis.get("kind") != "swe_contextual_evidence_update_analysis":
    raise SystemExit("contextual analysis kind/schema mismatch")
if analysis.get("protocol_sha256") != digest(protocol_arg):
    raise SystemExit("analysis protocol hash mismatch")
if analysis.get("manifest_sha256") != digest(manifest_arg):
    raise SystemExit("analysis materialization hash mismatch")
if analysis.get("prompt_bundle_sha256") != digest(prompts_arg):
    raise SystemExit("analysis prompt hash mismatch")
expected_reports = {"public"} if mode == "public_only" else {"public", "nf4", "native"}
if set(analysis.get("report_status", {})) != expected_reports:
    raise SystemExit("analysis report set differs from the selected replay mode")
if analysis.get("task_count") != 12:
    raise SystemExit("analysis task count differs from the frozen protocol")
if cards.get("schema_version") != 1 or cards.get("kind") != "swe_contextual_evidence_cards":
    raise SystemExit("contextual cards kind/schema mismatch")
if len(cards.get("cards", [])) != 12:
    raise SystemExit("contextual cards do not cover all frozen tasks")
PY
}

emit_run_manifest() {
  local status=$1
  local process_status=$2
  "$MANIFEST_PY" - "$RUN_MANIFEST" "$status" "$process_status" "$RUN_MODE" \
    "$CURRENT_PHASE" "$FAILURE_REASON" "$ROOT" "$OUTPUT_ROOT" "$OUT_DIR" \
    "$PROTOCOL" "$MATERIALIZER" "$ANALYZER" "$JLENS_RUNNER" \
    "$PROMPTS" "$MATERIALIZATION" "$PUBLIC_REPORT" "$NF4_REPORT" "$NATIVE_REPORT" \
    "$ANALYSIS" "$CARDS" "$LAYERS" "$MAX_MODEL_LEN" "$TOP_K" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

(
    output_arg, status, process_status, mode, phase, failure_reason, root_arg,
    output_root_arg, attempt_arg, protocol_arg, materializer_arg, analyzer_arg,
    runner_arg, prompts_arg, materialization_arg, public_arg, nf4_arg, native_arg,
    analysis_arg, cards_arg, layers_arg, max_len, top_k,
) = sys.argv[1:]
attempt = Path(attempt_arg)

def record(path_arg):
    path = Path(path_arg)
    result = {"path": str(path)}
    if path.is_file():
        raw = path.read_bytes()
        result.update({"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    else:
        result["missing"] = True
    return result

phases = {}
for path in sorted(attempt.glob("*.exit_status")):
    phases.setdefault(path.name.removesuffix(".exit_status"), {})["exit_status"] = path.read_text().strip()
for path in sorted(attempt.glob("*.wall_seconds")):
    phases.setdefault(path.name.removesuffix(".wall_seconds"), {})["wall_seconds"] = path.read_text().strip()
value = {
    "schema_version": 1,
    "kind": "swe_contextual_evidence_pilot_run",
    "status": status,
    "process_status": int(process_status),
    "mode": mode,
    "phase": phase,
    "failure_reason": failure_reason or None,
    "root": str(Path(root_arg)),
    "output_root": str(Path(output_root_arg)),
    "attempt": str(attempt),
    "runtime_contract": {
        "layers": [int(item) for item in layers_arg.split(",")],
        "positions": [-1],
        "top_k": int(top_k),
        "max_model_len": int(max_len),
        "mtp_enabled": False,
        "enforce_eager": True,
        "language_model_only": True,
    },
    "phases": phases,
    "inputs": {
        "protocol": record(protocol_arg),
        "materializer": record(materializer_arg),
        "analyzer": record(analyzer_arg),
        "runner": record(runner_arg),
    },
    "artifacts": {
        "prompts": record(prompts_arg),
        "materialization": record(materialization_arg),
        "public_report": record(public_arg),
        "nf4_report": record(nf4_arg),
        "native_report": record(native_arg),
        "analysis": record(analysis_arg),
        "cards": record(cards_arg),
    },
}
path = Path(output_arg)
path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
except BaseException:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
    raise
PY
  write_sha256_sidecar "$RUN_MANIFEST" "$RUN_MANIFEST_SHA256"
}

promote_attempt() {
  local link_name=$1
  "$MANIFEST_PY" - "$OUTPUT_ROOT" "$OUT_DIR" "$link_name" <<'PY'
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
attempt = Path(sys.argv[2]).resolve(strict=True)
link_name = sys.argv[3]
attempts = (root / "attempts").resolve(strict=True)
if attempt.parent != attempts:
    raise SystemExit("attempt is outside the immutable attempts directory")
destination = root / link_name
if destination.exists() and not destination.is_symlink():
    raise SystemExit(f"promotion target is not a symlink: {destination}")
temporary = root / f".{link_name}.tmp.{os.getpid()}"
try:
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(Path("attempts") / attempt.name)
    os.replace(temporary, destination)
finally:
    temporary.unlink(missing_ok=True)
PY
}

prepare_only=0
public_only=0
while (($#)); do
  case "$1" in
    --prepare-only)
      prepare_only=1
      ;;
    --public-only)
      public_only=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      echo "ERROR: unknown argument: $1" >&2
      exit 1
      ;;
  esac
  shift
done
if ((prepare_only && public_only)); then
  echo "ERROR: --prepare-only and --public-only are mutually exclusive" >&2
  exit 1
fi

command -v "$MANIFEST_PY" >/dev/null 2>&1 || {
  echo "ERROR: manifest Python is missing: $MANIFEST_PY" >&2
  exit 1
}
[[ ! -L "$OUTPUT_ROOT" ]] || {
  echo "ERROR: output root must not be a symlink: $OUTPUT_ROOT" >&2
  exit 1
}
mkdir -p "$OUTPUT_ROOT/attempts"
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd)
LOCK_DIR=$OUTPUT_ROOT/.pilot.lock
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "ERROR: another contextual pilot owns $LOCK_DIR" >&2
  exit 1
fi
OUT_DIR=$(mktemp -d "$OUTPUT_ROOT/attempts/attempt-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
LOG_DIR=$OUT_DIR/logs
CHECKSUM_DIR=$OUT_DIR/checksums
mkdir -p "$LOG_DIR" "$CHECKSUM_DIR"

PROMPTS=$OUT_DIR/prompts.json
MATERIALIZATION=$OUT_DIR/materialization.json
PUBLIC_REPORT=$OUT_DIR/public-report.json
NF4_REPORT=$OUT_DIR/nf4-report.json
NATIVE_REPORT=$OUT_DIR/native-report.json
ANALYSIS=$OUT_DIR/analysis.json
CARDS=$OUT_DIR/task-cards.json
RUN_MANIFEST=$OUT_DIR/run_manifest.json
RUN_MANIFEST_SHA256=$OUT_DIR/run_manifest.sha256

CURRENT_PHASE=initialization
FAILURE_REASON=
MANIFEST_EMITTED=0
if ((prepare_only)); then
  RUN_MODE=prepare_only
elif ((public_only)); then
  RUN_MODE=public_only
else
  RUN_MODE=full
fi

finish() {
  local process_status=$?
  trap - EXIT
  if ((MANIFEST_EMITTED == 0)); then
    emit_run_manifest failed "$process_status" >&2 || true
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
  exit "$process_status"
}
trap finish EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if ! validate_initialization_inputs >"$LOG_DIR/initialization.log" 2>&1; then
  printf 'initialization-failed\n' >"$OUT_DIR/initialization.exit_status"
  show_failure_log "$LOG_DIR/initialization.log"
  die "contextual pilot initialization failed"
fi
printf '0\n' >"$OUT_DIR/initialization.exit_status"

run_logged materialize "$LOG_DIR/materialize.log" "$PROMPTS" "$MATERIALIZATION" -- \
  "$VLLM_PY" "$MATERIALIZER" \
  --protocol "$PROTOCOL" \
  --output-prompts "$PROMPTS" \
  --output-manifest "$MATERIALIZATION"
if ! validate_materialization >>"$LOG_DIR/materialize.log" 2>&1; then
  printf 'invalid-materialized-bundle\n' >"$OUT_DIR/materialize.exit_status"
  show_failure_log "$LOG_DIR/materialize.log"
  die "contextual materialization violates the frozen protocol"
fi
write_sha256_sidecar "$PROMPTS" "$CHECKSUM_DIR/prompts.sha256"
write_sha256_sidecar "$MATERIALIZATION" "$CHECKSUM_DIR/materialization.sha256"

if ((prepare_only)); then
  CURRENT_PHASE=complete
  emit_run_manifest prepared 0
  promote_attempt latest-prepared
  MANIFEST_EMITTED=1
  echo "prepared prompts: $PROMPTS"
  echo "latest prepared attempt: $OUTPUT_ROOT/latest-prepared"
  exit 0
fi

if ((public_only == 0)); then
  CURRENT_PHASE=lens_artifact_validation
  if ! validate_local_lenses >"$LOG_DIR/lens_artifact_validation.log" 2>&1; then
    printf 'lens-artifact-validation-failed\n' >"$OUT_DIR/lens_artifact_validation.exit_status"
    show_failure_log "$LOG_DIR/lens_artifact_validation.log"
    die "local lens artifact validation failed"
  fi
  printf '0\n' >"$OUT_DIR/lens_artifact_validation.exit_status"
fi

COMMON_ARGS=(
  --prompts-file "$PROMPTS"
  --layers "$LAYERS"
  --positions=-1
  --top-k "$TOP_K"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --mamba-block-size "$MAMBA_BLOCK_SIZE"
  --enable-prefix-caching
  --kv-cache-dtype fp8_e4m3
  --kv-offloading-size "$KV_OFFLOADING_SIZE"
  --kv-offloading-backend native
  --stream-final-only
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)

run_lens public "$PUBLIC_REPORT" "$LOG_DIR/public.log" \
  "$JLENS_RUNNER" --lens-kind public "${COMMON_ARGS[@]}"

if ((public_only == 0)); then
  run_lens nf4 "$NF4_REPORT" "$LOG_DIR/nf4.log" \
    "$JLENS_RUNNER" \
    --lens-kind nf4 \
    --lens-path "$NF4_LENS_PATH" \
    --lens-sha256 "$NF4_LENS_SHA256" \
    --lens-provenance "$NF4_PROVENANCE_PATH" \
    "${COMMON_ARGS[@]}"

  run_lens native "$NATIVE_REPORT" "$LOG_DIR/native.log" \
    "$JLENS_RUNNER" \
    --lens-kind nvfp4-ste \
    --lens-path "$NATIVE_LENS_PATH" \
    --lens-sha256 "$NATIVE_LENS_SHA256" \
    --lens-provenance "$NATIVE_PROVENANCE_PATH" \
    --lens-state "$NATIVE_STATE_PATH" \
    --lens-state-sha256 "$NATIVE_STATE_SHA256" \
    "${COMMON_ARGS[@]}"
fi

ANALYZER_ARGS=(
  --protocol "$PROTOCOL"
  --manifest "$MATERIALIZATION"
  --public-report "$PUBLIC_REPORT"
  --output "$ANALYSIS"
  --cards-output "$CARDS"
)
if ((public_only == 0)); then
  ANALYZER_ARGS+=(--nf4-report "$NF4_REPORT" --native-report "$NATIVE_REPORT")
fi
run_logged analyze "$LOG_DIR/analyze.log" "$ANALYSIS" "$CARDS" -- \
  "$VLLM_PY" "$ANALYZER" "${ANALYZER_ARGS[@]}"
if ! validate_analysis_outputs >>"$LOG_DIR/analyze.log" 2>&1; then
  printf 'invalid-analysis\n' >"$OUT_DIR/analyze.exit_status"
  show_failure_log "$LOG_DIR/analyze.log"
  die "contextual analysis does not bind the exact replay evidence"
fi
write_sha256_sidecar "$ANALYSIS" "$CHECKSUM_DIR/analysis.sha256"
write_sha256_sidecar "$CARDS" "$CHECKSUM_DIR/task-cards.sha256"

CURRENT_PHASE=complete
emit_run_manifest complete 0
if ((public_only)); then
  promote_attempt latest-public
  PROMOTED=$OUTPUT_ROOT/latest-public
else
  promote_attempt latest
  PROMOTED=$OUTPUT_ROOT/latest
fi
MANIFEST_EMITTED=1
echo "analysis: $ANALYSIS"
echo "task cards: $CARDS"
echo "latest complete attempt: $PROMOTED"
