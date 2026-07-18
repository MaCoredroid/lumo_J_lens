#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi
CONFIG=${CONFIG:-$ROOT/configs/swe_behavioral_campaign.json}
IMAGE_CONFIG=${IMAGE_CONFIG:-$ROOT/configs/swe_behavioral_image_digests.json}
RUN_NAME=${RUN_NAME:-swe_behavioral_dev_20260718}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/$RUN_NAME}
SWE_PYTHON=${SWE_PYTHON:-$ROOT/.venv-swe/bin/python}
QWEN_BIN=${QWEN_BIN:-$ROOT/node_modules/.bin/qwen}
PORT=${PORT:-9952}
PROXY_PORT=${PROXY_PORT:-30042}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
MODEL=${SERVED_MODEL_NAME:-qwen3.6-27b-nvfp4}
ENDPOINT=${ENDPOINT:-http://127.0.0.1:$PORT/v1}
SUBSET=$RUN_ROOT/subset.json
DATASET_JSON=$RUN_ROOT/dataset.json
IMAGE_MANIFEST=$RUN_ROOT/image_manifest.json
OUT_ROOT=$RUN_ROOT/generation
SERVER_LOG=${SERVER_LOG:-$RUN_ROOT/server.log}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -f "$CONFIG" ]] || die "campaign config is missing: $CONFIG"
[[ -f "$IMAGE_CONFIG" ]] || die "campaign image registry is missing: $IMAGE_CONFIG"
[[ -x "$SWE_PYTHON" ]] || die "SWE Python is missing: $SWE_PYTHON"
[[ -x "$QWEN_BIN" ]] || die "Qwen Code is missing: $QWEN_BIN"
[[ $($QWEN_BIN --version) == 0.19.4 ]] || die "Qwen Code 0.19.4 is required"
mkdir -p "$RUN_ROOT"

"$SWE_PYTHON" - \
  "$CONFIG" "$IMAGE_CONFIG" "$SUBSET" "$DATASET_JSON" "$IMAGE_MANIFEST" \
  "$MAX_MODEL_LEN" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
image_config_path = Path(sys.argv[2])
subset_path = Path(sys.argv[3])
dataset_path = Path(sys.argv[4])
image_manifest_path = Path(sys.argv[5])
max_model_len = int(sys.argv[6])
config_bytes = config_path.read_bytes()
config = json.loads(config_bytes)
image_config_bytes = image_config_path.read_bytes()
image_config = json.loads(image_config_bytes)
if config.get("schema_version") != 1:
    raise SystemExit("unsupported behavioral campaign schema")
if config.get("generation", {}).get("max_model_len") != max_model_len:
    raise SystemExit("MAX_MODEL_LEN does not match the frozen campaign config")
instance_ids = config.get("instance_ids")
if (
    not isinstance(instance_ids, list)
    or len(instance_ids) != 10
    or len(set(instance_ids)) != len(instance_ids)
    or not all(isinstance(value, str) and "__" in value for value in instance_ids)
):
    raise SystemExit("campaign must contain ten unique SWE instance IDs")

images = []
for instance_id in instance_ids:
    tag = (
        "swebench/sweb.eval.x86_64."
        + instance_id.replace("__", "_1776_")
        + ":latest"
    )
    try:
        inspected = json.loads(
            subprocess.check_output(["docker", "image", "inspect", tag], text=True)
        )
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"required cached image is missing: {tag}") from error
    if len(inspected) != 1 or inspected[0].get("Architecture") != "amd64":
        raise SystemExit(f"unexpected image inspection result: {tag}")
    pin = image_config.get("images", {}).get(instance_id, {}).get("x86_64")
    if not isinstance(pin, dict) or pin.get("image_id") != inspected[0]["Id"]:
        raise SystemExit(f"campaign image pin does not match local bytes: {tag}")
    if pin.get("reference") not in (inspected[0].get("RepoDigests") or []):
        raise SystemExit(f"campaign image digest reference is not locally proven: {tag}")
    images.append(
        {
            "instance_id": instance_id,
            "tag": tag,
            "image_id": inspected[0]["Id"],
            "repo_digests": sorted(inspected[0].get("RepoDigests") or []),
        }
    )

from datasets import load_dataset

dataset = load_dataset(
    config["dataset"]["repo_id"],
    split="test",
    revision=config["dataset"]["revision"],
)
selected = [dict(row) for row in dataset if row["instance_id"] in set(instance_ids)]
selected.sort(key=lambda row: instance_ids.index(row["instance_id"]))
if [row["instance_id"] for row in selected] != instance_ids:
    raise SystemExit("frozen dataset revision does not contain the exact campaign cohort")

subset = {
    "dataset_name": str(dataset_path.resolve()),
    "instance_ids": instance_ids,
}
manifest = {
    "schema_version": 1,
    "kind": "swe_verified_behavioral_campaign_image_manifest",
    "campaign_config_path": str(config_path),
    "campaign_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    "image_config_path": str(image_config_path),
    "image_config_sha256": hashlib.sha256(image_config_bytes).hexdigest(),
    "dataset": config["dataset"],
    "images": images,
}
for path, value in (
    (subset_path, subset),
    (dataset_path, selected),
    (image_manifest_path, manifest),
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
PY

server_started=0
cleanup() {
  status=$?
  if ((server_started)); then
    "$ROOT/scripts/stop_server.sh" || true
  fi
  exit "$status"
}
trap cleanup EXIT

if "$SWE_PYTHON" "$ROOT/scripts/check_endpoint.py" "$ENDPOINT" \
  --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" --quiet >/dev/null 2>&1; then
  echo "using compatible endpoint: $ENDPOINT"
elif curl -fsS "$ENDPOINT/models" >/dev/null 2>&1; then
  die "an incompatible endpoint already owns $ENDPOINT"
else
  echo "starting the frozen NVFP4 + MTP generation profile"
  MAX_MODEL_LEN=$MAX_MODEL_LEN \
  MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096} \
  MAX_NUM_SEQS=${MAX_NUM_SEQS:-2} \
  NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-1} \
  KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3} \
  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85} \
  LOG_PATH=$SERVER_LOG \
    "$ROOT/scripts/start_server.sh"
  server_started=1
fi

export SWE_IMAGE_CONFIG="$IMAGE_CONFIG"
export LUMO_ENABLE_THINKING=${LUMO_ENABLE_THINKING:-true}
export LUMO_PROXY_FORCE_TEMPERATURE=${LUMO_PROXY_FORCE_TEMPERATURE:-1.0}
export LUMO_PROXY_FORCE_TOP_P=${LUMO_PROXY_FORCE_TOP_P:-0.95}
export LUMO_PROXY_FORCE_TOP_K=${LUMO_PROXY_FORCE_TOP_K:-20}
export LUMO_PROXY_FORCE_MIN_P=${LUMO_PROXY_FORCE_MIN_P:-0.0}
export LUMO_PROXY_FORCE_PRESENCE_PENALTY=${LUMO_PROXY_FORCE_PRESENCE_PENALTY:-0.0}
export LUMO_PROXY_FORCE_SEED=${LUMO_PROXY_FORCE_SEED:-880001234}
export SWE_EMPTY_PATCH_RETRIES=${SWE_EMPTY_PATCH_RETRIES:-0}

"$SWE_PYTHON" "$ROOT/scripts/run_swe_verified.py" \
  --subset "$SUBSET" \
  --out-root "$OUT_ROOT" \
  --dataset-tag verified \
  --runtime container \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --model-name "qwen3.6-27b-nvfp4-mtp::qwen-code-0.19.4" \
  --repo-cache "$ROOT/.cache/swe_bench_repos" \
  --eval-mode skip \
  --agent-wall-s 900 \
  --qwen-max-wall 840s \
  --max-session-turns 50 \
  --proxy-port "$PROXY_PORT" \
  --proxy-max-tokens 8192 \
  --proxy-context-limit "$MAX_MODEL_LEN" \
  --proxy-script "$ROOT/scripts/qwen_code_proxy.py" \
  --proxy-dump-dir "$RUN_ROOT/proxy_dumps" \
  --container-name-prefix swe_ep_lumo_behavioral \
  --proxy-tool-choice "" \
  --proxy-required-tools run_shell_command \
  --core-tool run_shell_command \
  --exclude-tools tool_search \
  --exclude-tools web_fetch \
  --exclude-tools web_search \
  --exclude-tools agent \
  --exclude-tools skill \
  --exclude-tools task_stop \
  --exclude-tools send_message \
  --exclude-tools enter_worktree \
  --exclude-tools exit_worktree \
  --exclude-tools notebook_edit \
  --exclude-tools todo_write \
  --exclude-tools ask_user_question \
  --exclude-tools exit_plan_mode \
  --exclude-tools enter_plan_mode \
  --allow-empty-predictions \
  --qwen-bin "$QWEN_BIN"

printf '%s\n' "$RUN_NAME" >"$ROOT/runs/LATEST_BEHAVIORAL_RUN"
echo "behavioral generation complete: $OUT_ROOT/verified"
