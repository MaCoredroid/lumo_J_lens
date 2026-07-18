#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VLLM_PY=${VLLM_PY:-$ROOT/.venv-vllm/bin/python}
OUT_DIR=${OUT_DIR:-$ROOT/.cache/swe_multistage_pilot}

PILOT_SCRIPT=$ROOT/scripts/run_swe_multistage_pilot.sh
MATERIALIZER=$ROOT/scripts/materialize_swe_multistage_probes.py
AUGMENTER=$ROOT/scripts/augment_swe_multistage_action_probes.py
ANALYZER=$ROOT/scripts/analyze_swe_multistage_probes.py
JLENS_RUNNER=${JLENS_RUNNER:-$ROOT/scripts/run_jlens_nvfp4.sh}
JLENS_PYTHON_RUNNER=$ROOT/scripts/run_jlens_nvfp4.py
LIFECYCLE_PROTOCOL=$ROOT/configs/swe_multistage_protocol.json
TRAJECTORY_MANIFEST=$ROOT/configs/swe_multistage_trajectory_manifest.json
ACTION_PROTOCOL=$ROOT/configs/swe_stage_action_probes.json

MODEL_REPO=nvidia/Qwen3.6-27B-NVFP4
MODEL_REVISION=0893e1606ff3d5f97a441f405d5fc541a6bdf404
MODEL_CONFIG_SHA256=c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338
MODEL_INDEX_SHA256=7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2
PUBLIC_LENS_REPO=neuronpedia/jacobian-lens
PUBLIC_LENS_REVISION=a4114d7752d11eb546e6cf372213d7e75526d3a1
PUBLIC_LENS_SHA256=1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1
REPORT_SCHEMA_VERSION=3
TORCH_VERSION=2.11.0+cu130
VLLM_VERSION=0.23.0
TRANSFORMERS_VERSION=5.12.1
HUGGINGFACE_HUB_VERSION=1.21.0
TRITON_VERSION=3.6.0
PYTHON_VERSION=3.12.13

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

LAYERS=16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47

usage() {
  cat <<'EOF'
Usage: scripts/run_swe_multistage_pilot.sh [--prepare-only | --reuse-reports]

Environment:
  OUT_DIR    Output directory (default: .cache/swe_multistage_pilot)
  VLLM_PY    Pinned vLLM Python interpreter
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  local path=$1
  local label=$2
  [[ -f "$path" ]] || die "$label is missing: $path"
}

require_sha256() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
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
  local status_file=$OUT_DIR/$label.exit_status
  rm -f "$log" "$status_file" "${outputs[@]}"
  echo "[$label] log: $log"
  set +e
  "$@" >"$log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" >"$status_file"
  if ((status != 0)); then
    show_failure_log "$log"
    die "$label failed with exit status $status"
  fi
  local output
  for output in "${outputs[@]}"; do
    [[ -s "$output" ]] || die "$label did not emit a fresh nonempty output: $output"
  done
}

validate_json_report() {
  local report=$1
  local lens_kind=$2
  local process_status=${3:-reused}
  "$VLLM_PY" - \
    "$report" "$lens_kind" "$process_status" "$ACTION_PROMPTS" \
    "$REPORT_SCHEMA_VERSION" "$LAYERS" \
    "$MODEL_REPO" "$MODEL_REVISION" "$MODEL_CONFIG_SHA256" "$MODEL_INDEX_SHA256" \
    "$PUBLIC_LENS_REPO" "$PUBLIC_LENS_REVISION" "$PUBLIC_LENS_SHA256" \
    "$NF4_LENS_SHA256" "$NF4_PROVENANCE_SHA256" \
    "$NATIVE_LENS_SHA256" "$NATIVE_PROVENANCE_SHA256" "$NATIVE_STATE_SHA256" \
    "$TORCH_VERSION" "$VLLM_VERSION" "$TRANSFORMERS_VERSION" \
    "$HUGGINGFACE_HUB_VERSION" "$TRITON_VERSION" "$PYTHON_VERSION" <<'PY'
import json
import sys
from pathlib import Path

(
    report_arg,
    lens_kind,
    process_status,
    prompts_arg,
    schema_arg,
    layers_arg,
    model_repo,
    model_revision,
    model_config_sha256,
    model_index_sha256,
    public_lens_repo,
    public_lens_revision,
    public_lens_sha256,
    nf4_lens_sha256,
    nf4_provenance_sha256,
    native_lens_sha256,
    native_provenance_sha256,
    native_state_sha256,
    torch_version,
    vllm_version,
    transformers_version,
    hub_version,
    triton_version,
    python_version,
) = sys.argv[1:]


def fail(message):
    raise SystemExit(message)


def expect(actual, expected, label):
    if actual != expected:
        fail(f"{label} mismatch: expected {expected!r}, got {actual!r}")


path = Path(report_arg)
prompts_path = Path(prompts_arg)
try:
    value = json.loads(path.read_bytes())
    prompts = json.loads(prompts_path.read_bytes())
except (OSError, json.JSONDecodeError) as error:
    fail(f"cannot load report/prompt JSON: {error}")
if not isinstance(value, dict):
    fail(f"report must be a JSON object: {path}")
if not isinstance(prompts, list) or not prompts:
    fail("action prompt bundle must be a nonempty list")
if value.get("status") not in {"passed", "failed"}:
    fail(f"report has an invalid status: {path}")
if process_status in {"0", "1"}:
    expect(value["status"], "passed" if process_status == "0" else "failed", "status/exit")
expect(value.get("schema_version"), int(schema_arg), "report schema")
expect(value.get("score_encoding"), "unrounded-float32", "score encoding")

model = value.get("model")
if not isinstance(model, dict):
    fail("report.model must be an object")
model_expected = {
    "repo_id": model_repo,
    "revision": model_revision,
    "config_sha256": model_config_sha256,
    "index_sha256": model_index_sha256,
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
}
for key, expected in model_expected.items():
    expect(model.get(key), expected, f"model.{key}")

host = value.get("host")
if not isinstance(host, dict):
    fail("report.host must be an object")
expect(host.get("python"), python_version, "host.python")
expect(
    host.get("packages"),
    {
        "huggingface-hub": hub_version,
        "torch": torch_version,
        "transformers": transformers_version,
        "triton": triton_version,
        "vllm": vllm_version,
    },
    "host.packages",
)
gpu = host.get("gpu")
if not isinstance(gpu, dict):
    fail("report.host.gpu must be an object")
expect(gpu.get("name"), "NVIDIA GeForce RTX 5090", "host.gpu.name")
expect(gpu.get("compute_capability"), "12.0", "host.gpu.compute_capability")

runtime = value.get("runtime")
if not isinstance(runtime, dict):
    fail("report.runtime must be an object")
runtime_expected = {
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
}
expect(set(runtime), set(runtime_expected) | {"model_load_seconds"}, "runtime fields")
for key, expected in runtime_expected.items():
    expect(runtime.get(key), expected, f"runtime.{key}")
if not isinstance(runtime.get("model_load_seconds"), (int, float)) or runtime["model_load_seconds"] <= 0:
    fail("runtime.model_load_seconds must be positive")

lens = value.get("lens")
if not isinstance(lens, dict):
    fail("report.lens must be an object")
if lens_kind == "public":
    lens_expected = {
        "repo_id": public_lens_repo,
        "revision": public_lens_revision,
        "sha256": public_lens_sha256,
        "n_prompts": 1000,
    }
elif lens_kind == "nf4":
    lens_expected = {
        "kind": "local_fit",
        "sha256": nf4_lens_sha256,
        "provenance_sha256": nf4_provenance_sha256,
        "n_prompts": 10,
        "fit_quantization": "bitsandbytes-nf4-double-quant-bfloat16",
    }
elif lens_kind == "native":
    lens_expected = {
        "kind": "native_nvfp4_ste_fit",
        "sha256": native_lens_sha256,
        "provenance_sha256": native_provenance_sha256,
        "state_sha256": native_state_sha256,
        "n_prompts": 10,
        "fit_quantization": "nvidia-modelopt-nvfp4-fp8-exact-forward-identity-ste",
    }
else:
    fail(f"unknown expected lens kind: {lens_kind}")
for key, expected in lens_expected.items():
    expect(lens.get(key), expected, f"lens.{key}")

assertions = value.get("assertions")
if not isinstance(assertions, dict):
    fail("report.assertions must be an object")
for key in ("lens_hash_matches", "lens_metadata_matches", "model_architecture_matches"):
    expect(assertions.get(key), True, f"assertions.{key}")

layers = [int(item) for item in layers_arg.split(",")]
experiments = value.get("experiments")
if not isinstance(experiments, list):
    fail("report.experiments must be a list")
expect(len(experiments), len(prompts), "experiment count")
score_union = []
for prompt in prompts:
    if not isinstance(prompt, dict):
        fail("prompt records must be objects")
    for token_id in prompt.get("score_token_ids", []):
        if token_id not in score_union:
            score_union.append(token_id)
for index, (prompt, experiment) in enumerate(zip(prompts, experiments, strict=True)):
    label = f"experiment[{index}]"
    if not isinstance(experiment, dict):
        fail(f"{label} must be an object")
    expect(experiment.get("id"), str(prompt.get("id", index)), f"{label}.id")
    expect(experiment.get("prompt"), prompt.get("text"), f"{label}.prompt")
    expect(experiment.get("prompt_token_ids"), prompt.get("token_ids"), f"{label}.prompt_token_ids")
    expect(experiment.get("metadata"), prompt.get("metadata"), f"{label}.metadata")
    score_ids = prompt.get("score_token_ids")
    expect(
        experiment.get("scored_vocabulary", {}).get("token_ids"),
        score_ids,
        f"{label}.scored_vocabulary.token_ids",
    )
    final_position = len(prompt["token_ids"]) - 1
    expect(experiment.get("positions_requested"), [-1], f"{label}.positions_requested")
    expect(experiment.get("positions_resolved"), [final_position], f"{label}.positions_resolved")
    expect(
        experiment.get("capture_positions_resolved"),
        [final_position],
        f"{label}.capture_positions_resolved",
    )
    layer_records = experiment.get("layers")
    if not isinstance(layer_records, list):
        fail(f"{label}.layers must be a list")
    expect([record.get("layer") for record in layer_records], layers, f"{label}.layers")
    for record in layer_records:
        positions = record.get("positions")
        if not isinstance(positions, list) or len(positions) != 1:
            fail(f"{label} layer positions changed")
        position = positions[0]
        expect(position.get("token_position"), final_position, f"{label}.token_position")
        for method in ("logit_lens", "jacobian_lens"):
            readout = position.get(method)
            if not isinstance(readout, dict):
                fail(f"{label}.{method} is missing")
            expect(len(readout.get("tokens", [])), 10, f"{label}.{method}.top_k")
            expect(
                [entry.get("token_id") for entry in readout.get("scored_tokens", [])],
                score_ids,
                f"{label}.{method}.scored_tokens",
            )
expect(value.get("scored_vocabulary", {}).get("union_token_ids"), score_union, "score union")
PY
}

run_lens() {
  local label=$1
  local report=$2
  local log=$3
  shift 3

  local status_file=$OUT_DIR/$label.exit_status
  local stdout_file=$OUT_DIR/.$label.stdout
  rm -f "$report" "$log" "$status_file" "$stdout_file"
  echo "[$label] log: $log"
  set +e
  "$@" --output "$report" >"$stdout_file" 2>"$log"
  local status=$?
  set -e
  printf '%s\n' "$status" >"$status_file"

  if [[ ! -s "$report" ]]; then
    if [[ -s "$stdout_file" ]]; then
      echo "--- tail of $stdout_file ---" >>"$log"
      tail -80 "$stdout_file" >>"$log" || true
    fi
    show_failure_log "$log"
    die "$label exited $status without a fresh report"
  fi
  rm -f "$stdout_file"
  if ((status != 0 && status != 1)); then
    show_failure_log "$log"
    die "$label emitted a report but exited with unsupported status $status"
  fi
  if ! validate_json_report "$report" "$label" "$status" >>"$log" 2>&1; then
    show_failure_log "$log"
    die "$label emitted an invalid report"
  fi
  echo "[$label] accepted exit status $status with report $report"
}

reuse_lens_report() {
  local label=$1
  local report=$2
  local log=$LOG_DIR/$label.log
  local status_file=$OUT_DIR/$label.exit_status
  rm -f "$log" "$status_file"
  if ! validate_json_report "$report" "$label" reused >"$log" 2>&1; then
    show_failure_log "$log"
    die "reused $label report is stale or violates the pinned runtime contract"
  fi
  printf 'reused\n' >"$status_file"
  echo "[$label] reused exact prompt/runtime-bound report: $report"
}

emit_run_manifest() {
  local final_status=$1
  local mode=$2
  rm -f "$RUN_MANIFEST" "$RUN_MANIFEST_SHA256"
  "$VLLM_PY" - \
    "$RUN_MANIFEST" "$final_status" "$mode" "$ROOT" "$OUT_DIR" \
    "$PILOT_SCRIPT" "$MATERIALIZER" "$AUGMENTER" "$ANALYZER" \
    "$JLENS_RUNNER" "$JLENS_PYTHON_RUNNER" \
    "$LIFECYCLE_PROTOCOL" "$TRAJECTORY_MANIFEST" "$ACTION_PROTOCOL" \
    "$PROMPTS" "$PROMPTS_SUMMARY" "$ACTION_PROMPTS" "$ACTION_SUMMARY" \
    "$PUBLIC_REPORT" "$NF4_REPORT" "$NATIVE_REPORT" "$ANALYSIS" \
    "$MODEL_REPO" "$MODEL_REVISION" "$MODEL_CONFIG_SHA256" "$MODEL_INDEX_SHA256" \
    "$PUBLIC_LENS_SHA256" "$NF4_LENS_SHA256" "$NATIVE_LENS_SHA256" \
    "$LAYERS" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

(
    manifest_arg,
    final_status,
    mode,
    root_arg,
    out_arg,
    pilot_arg,
    materializer_arg,
    augmenter_arg,
    analyzer_arg,
    runner_arg,
    python_runner_arg,
    lifecycle_arg,
    trajectory_arg,
    action_arg,
    prompts_arg,
    prompts_summary_arg,
    action_prompts_arg,
    action_summary_arg,
    public_arg,
    nf4_arg,
    native_arg,
    analysis_arg,
    model_repo,
    model_revision,
    model_config_sha256,
    model_index_sha256,
    public_lens_sha256,
    nf4_lens_sha256,
    native_lens_sha256,
    layers_arg,
) = sys.argv[1:]

root = Path(root_arg).resolve(strict=True)
out = Path(out_arg).resolve(strict=True)
manifest_path = Path(manifest_arg)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def record(path_arg, *, required=True):
    path = Path(path_arg)
    if not path.is_file() or path.stat().st_size == 0:
        if required:
            raise SystemExit(f"manifest artifact is missing or empty: {path}")
        return None
    resolved = path.resolve(strict=True)
    if resolved.is_relative_to(root):
        path_base = "repository_root"
        logical_path = resolved.relative_to(root).as_posix()
    elif resolved.is_relative_to(out):
        path_base = "output_directory"
        logical_path = resolved.relative_to(out).as_posix()
    else:
        raise SystemExit(f"manifest artifact is outside repository/output roots: {path}")
    result = {
        "path": logical_path,
        "path_base": path_base,
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }
    if path.suffix == ".json":
        value = json.loads(path.read_bytes())
        if isinstance(value, dict) and value.get("status") in {"passed", "failed"}:
            result["report_status"] = value["status"]
    return result


inputs = {
    "pilot": record(pilot_arg),
    "materializer": record(materializer_arg),
    "augmenter": record(augmenter_arg),
    "analyzer": record(analyzer_arg),
    "lifecycle_protocol": record(lifecycle_arg),
    "trajectory_manifest": record(trajectory_arg),
    "action_protocol": record(action_arg),
}
if mode != "prepare_only":
    inputs["jlens_runner"] = record(runner_arg)
    inputs["jlens_python_runner"] = record(python_runner_arg)

artifacts = {
    "prompts": record(prompts_arg),
    "prompts_summary": record(prompts_summary_arg),
    "action_prompts": record(action_prompts_arg),
    "action_prompts_summary": record(action_summary_arg),
}
if final_status == "complete":
    artifacts.update(
        {
            "public_report": record(public_arg),
            "nf4_report": record(nf4_arg),
            "native_report": record(native_arg),
            "analysis": record(analysis_arg),
        }
    )

phases = {}
for label in ("materialize", "augment", "public", "nf4", "native", "analyze"):
    status_path = out / f"{label}.exit_status"
    if status_path.is_file():
        phases[label] = {
            "status": status_path.read_text(encoding="ascii").strip(),
            "status_file_sha256": digest(status_path),
        }

value = {
    "schema_version": 2,
    "kind": "swe_verified_multistage_pilot_run",
    "status": final_status,
    "mode": mode,
    "path_contract": {
        "repository_root": "directory containing this repository",
        "output_directory": "directory containing this manifest",
        "absolute_paths_embedded": False,
    },
    "inputs": inputs,
    "artifacts": artifacts,
    "phases": phases,
    "runtime_contract": {
        "model": {
            "repo_id": model_repo,
            "revision": model_revision,
            "config_sha256": model_config_sha256,
            "index_sha256": model_index_sha256,
        },
        "lens_sha256": {
            "public": public_lens_sha256,
            "nf4": nf4_lens_sha256,
            "native": native_lens_sha256,
        },
        "layers": [int(item) for item in layers_arg.split(",")],
        "positions": [-1],
        "top_k": 10,
        "max_model_len": 49152,
        "max_num_batched_tokens": 4096,
        "mamba_block_size": 1024,
        "enable_prefix_caching": True,
        "kv_cache_dtype": "fp8_e4m3",
        "kv_offloading_size_gib": 8,
        "kv_offloading_backend": "native",
        "stream_final_only": True,
        "gpu_memory_utilization": 0.78,
        "mtp_enabled": False,
    },
}
rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary = tempfile.mkstemp(prefix=f".{manifest_path.name}.", dir=manifest_path.parent)
try:
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest_path)
except BaseException:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
    raise
PY
  local digest
  digest=$(sha256sum "$RUN_MANIFEST" | awk '{print $1}')
  printf '%s  %s\n' "$digest" "$(basename "$RUN_MANIFEST")" >"$RUN_MANIFEST_SHA256"
  echo "run manifest: $RUN_MANIFEST (sha256=$digest)"
}

prepare_only=0
reuse_reports=0
while (($#)); do
  case "$1" in
    --prepare-only)
      prepare_only=1
      ;;
    --reuse-reports)
      reuse_reports=1
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

if ((prepare_only && reuse_reports)); then
  die "--prepare-only and --reuse-reports are mutually exclusive"
fi

[[ -x "$VLLM_PY" ]] || die "vLLM Python is missing or not executable: $VLLM_PY"
require_file "$MATERIALIZER" "multi-stage materializer"
require_file "$AUGMENTER" "action-vocabulary augmenter"
require_file "$ANALYZER" "multi-stage analyzer"
require_file "$LIFECYCLE_PROTOCOL" "lifecycle protocol"
require_file "$TRAJECTORY_MANIFEST" "trajectory manifest"
require_file "$ACTION_PROTOCOL" "action protocol"

mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)
LOG_DIR=$OUT_DIR/logs
mkdir -p "$LOG_DIR"
LOCK_DIR=$OUT_DIR/.pilot.lock
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  die "another pilot process owns $LOCK_DIR"
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

PROMPTS=$OUT_DIR/prompts.json
PROMPTS_SUMMARY=$OUT_DIR/prompts_summary.json
ACTION_PROMPTS=$OUT_DIR/action_prompts.json
ACTION_SUMMARY=$OUT_DIR/action_prompts_summary.json
PUBLIC_REPORT=$OUT_DIR/public-report.json
NF4_REPORT=$OUT_DIR/nf4-report.json
NATIVE_REPORT=$OUT_DIR/native-report.json
ANALYSIS=$OUT_DIR/analysis.json
RUN_MANIFEST=$OUT_DIR/run_manifest.json
RUN_MANIFEST_SHA256=$OUT_DIR/run_manifest.sha256

rm -f \
  "$PROMPTS" \
  "$PROMPTS_SUMMARY" \
  "$ACTION_PROMPTS" \
  "$ACTION_SUMMARY" \
  "$ANALYSIS" \
  "$OUT_DIR/materialize.exit_status" \
  "$OUT_DIR/augment.exit_status" \
  "$OUT_DIR/public.exit_status" \
  "$OUT_DIR/nf4.exit_status" \
  "$OUT_DIR/native.exit_status" \
  "$OUT_DIR/analyze.exit_status" \
  "$RUN_MANIFEST" \
  "$RUN_MANIFEST_SHA256"
rm -f "$LOG_DIR"/{materialize,augment,public,nf4,native,analyze}.log
rm -f "$OUT_DIR"/.{public,nf4,native}.stdout
if ((!reuse_reports && !prepare_only)); then
  rm -f "$PUBLIC_REPORT" "$NF4_REPORT" "$NATIVE_REPORT"
fi

run_logged materialize "$LOG_DIR/materialize.log" \
  "$PROMPTS" "$PROMPTS_SUMMARY" -- \
  "$VLLM_PY" "$MATERIALIZER" \
  --lifecycle-protocol "$LIFECYCLE_PROTOCOL" \
  --manifest "$TRAJECTORY_MANIFEST" \
  --artifact-root "$ROOT" \
  --output "$PROMPTS" \
  --summary "$PROMPTS_SUMMARY"

run_logged augment "$LOG_DIR/augment.log" \
  "$ACTION_PROMPTS" "$ACTION_SUMMARY" -- \
  "$VLLM_PY" "$AUGMENTER" \
  --input "$PROMPTS" \
  --action-protocol "$ACTION_PROTOCOL" \
  --lifecycle-protocol "$LIFECYCLE_PROTOCOL" \
  --artifact-root "$ROOT" \
  --output "$ACTION_PROMPTS" \
  --summary "$ACTION_SUMMARY"

if ((prepare_only)); then
  emit_run_manifest prepared prepare_only
  echo "prepared prompts: $ACTION_PROMPTS"
  exit 0
fi

if ((reuse_reports)); then
  require_file "$PUBLIC_REPORT" "reused public report"
  require_file "$NF4_REPORT" "reused NF4 report"
  require_file "$NATIVE_REPORT" "reused native report"
  [[ -s "$PUBLIC_REPORT" && -s "$NF4_REPORT" && -s "$NATIVE_REPORT" ]] || {
    die "all reused reports must be nonempty"
  }
  reuse_lens_report public "$PUBLIC_REPORT"
  reuse_lens_report nf4 "$NF4_REPORT"
  reuse_lens_report native "$NATIVE_REPORT"
else
  [[ -x "$JLENS_RUNNER" ]] || die "J-lens runner is missing: $JLENS_RUNNER"
  require_file "$JLENS_PYTHON_RUNNER" "J-lens Python runner"
  require_file "$NF4_LENS_PATH" "NF4 lens"
  require_file "$NF4_PROVENANCE_PATH" "NF4 provenance"
  require_file "$NATIVE_LENS_PATH" "native NVFP4 lens"
  require_file "$NATIVE_PROVENANCE_PATH" "native NVFP4 provenance"
  require_file "$NATIVE_STATE_PATH" "native NVFP4 state"
  require_sha256 "$NF4_LENS_PATH" "$NF4_LENS_SHA256" "NF4 lens"
  require_sha256 \
    "$NF4_PROVENANCE_PATH" "$NF4_PROVENANCE_SHA256" "NF4 provenance"
  require_sha256 "$NATIVE_LENS_PATH" "$NATIVE_LENS_SHA256" "native lens"
  require_sha256 \
    "$NATIVE_PROVENANCE_PATH" "$NATIVE_PROVENANCE_SHA256" "native provenance"
  require_sha256 "$NATIVE_STATE_PATH" "$NATIVE_STATE_SHA256" "native state"

  COMMON_ARGS=(
    --prompts-file "$ACTION_PROMPTS"
    --layers "$LAYERS"
    --positions=-1
    --top-k 10
    --max-model-len 49152
    --max-num-batched-tokens 4096
    --mamba-block-size 1024
    --enable-prefix-caching
    --kv-cache-dtype fp8_e4m3
    --kv-offloading-size 8
    --kv-offloading-backend native
    --stream-final-only
    --gpu-memory-utilization 0.78
  )

  run_lens public "$PUBLIC_REPORT" "$LOG_DIR/public.log" \
    "$JLENS_RUNNER" \
    --lens-kind public \
    "${COMMON_ARGS[@]}"

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

run_logged analyze "$LOG_DIR/analyze.log" \
  "$ANALYSIS" -- \
  "$VLLM_PY" "$ANALYZER" \
  --prompts "$ACTION_PROMPTS" \
  --action-protocol "$ACTION_PROTOCOL" \
  --public-report "$PUBLIC_REPORT" \
  --nf4-report "$NF4_REPORT" \
  --native-report "$NATIVE_REPORT" \
  --output "$ANALYSIS"

emit_run_manifest complete "$([[ $reuse_reports == 1 ]] && echo reuse_reports || echo fresh_replay)"
echo "analysis: $ANALYSIS"
