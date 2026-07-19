#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ ! -e "$ROOT/.env" && ! -L "$ROOT/.env" ]] || {
  die "the certified V3 campaign forbids $ROOT/.env, including symlinks"
}
export LUMO_V3_CERTIFIED_NO_DOTENV=1

CONFIG=${CONFIG:-}
IMAGE_CONFIG=${IMAGE_CONFIG:-}
SELECTION_PROOF=${SELECTION_PROOF:-}
RUN_ROOT=${RUN_ROOT:-}
SWE_PYTHON=${SWE_PYTHON:-$ROOT/.venv-swe/bin/python}
QWEN_BIN=${QWEN_BIN:-$ROOT/node_modules/.bin/qwen}
PORT=${PORT:-9952}
PROXY_PORT=${PROXY_PORT:-}
MODEL_PATH=${MODEL_PATH:-nvidia/Qwen3.6-27B-NVFP4}
MODEL_REVISION=${MODEL_REVISION:-0893e1606ff3d5f97a441f405d5fc541a6bdf404}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.6-27b-nvfp4}
QUANTIZATION=${QUANTIZATION:-modelopt_fp4}
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-0}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-TRITON_ATTN}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$ROOT/configs/qwen3-openai-codex.jinja}
ENDPOINT=http://127.0.0.1:$PORT/v1
V3_RUNS_ROOT=$ROOT/runs/swe_state_interpreter_v3_development
CONTRACT=$ROOT/scripts/swe_state_interpreter_v3_campaign_contract.py

[[ -n "$CONFIG" ]] || die "CONFIG is required"
[[ -n "$IMAGE_CONFIG" ]] || die "IMAGE_CONFIG is required"
[[ -n "$SELECTION_PROOF" ]] || die "SELECTION_PROOF is required"
[[ -n "$RUN_ROOT" ]] || die "RUN_ROOT is required"
[[ -n "$PROXY_PORT" ]] || die "PROXY_PORT is required"
[[ "$PORT" =~ ^[0-9]+$ && "$PORT" -ge 1 && "$PORT" -le 65535 ]] || {
  die "PORT must be an integer from 1 through 65535"
}
[[ "$PROXY_PORT" =~ ^[0-9]+$ && "$PROXY_PORT" -ge 1 && "$PROXY_PORT" -le 65535 ]] || {
  die "PROXY_PORT must be an integer from 1 through 65535"
}
[[ "$PORT" != "$PROXY_PORT" ]] || die "PORT and PROXY_PORT must differ"
[[ "$MODEL_PATH" == nvidia/Qwen3.6-27B-NVFP4 ]] || die "MODEL_PATH changed"
[[ "$MODEL_REVISION" == 0893e1606ff3d5f97a441f405d5fc541a6bdf404 ]] || {
  die "MODEL_REVISION changed"
}
[[ "$SERVED_MODEL_NAME" == qwen3.6-27b-nvfp4 ]] || die "SERVED_MODEL_NAME changed"
[[ "$QUANTIZATION" == modelopt_fp4 ]] || die "QUANTIZATION changed"
[[ "$KV_OFFLOAD_GB" == 0 ]] || die "KV_OFFLOAD_GB changed"
[[ "$ATTENTION_BACKEND" == TRITON_ATTN ]] || die "ATTENTION_BACKEND changed"
[[ "$CHAT_TEMPLATE" == "$ROOT/configs/qwen3-openai-codex.jinja" \
  && -f "$CHAT_TEMPLATE" && ! -L "$CHAT_TEMPLATE" ]] || {
  die "CHAT_TEMPLATE changed"
}
[[ -x "$SWE_PYTHON" ]] || die "SWE Python is missing: $SWE_PYTHON"
[[ -x "$QWEN_BIN" ]] || die "Qwen Code is missing: $QWEN_BIN"
[[ -f "$CONTRACT" && ! -L "$CONTRACT" ]] || die "V3 campaign contract is missing"
[[ $($QWEN_BIN --version) == 0.19.4 ]] || die "Qwen Code 0.19.4 is required"

"$SWE_PYTHON" "$CONTRACT" \
  --config "$CONFIG" \
  --image-config "$IMAGE_CONFIG" \
  --selection-proof "$SELECTION_PROOF" \
  --run-root "$RUN_ROOT"

CONFIG=$(realpath "$CONFIG")
IMAGE_CONFIG=$(realpath "$IMAGE_CONFIG")
SELECTION_PROOF=$(realpath "$SELECTION_PROOF")
RUN_ROOT=$(realpath "$RUN_ROOT")
SUBSET=$RUN_ROOT/subset.json
DATASET_JSON=$RUN_ROOT/dataset.json
OUT_ROOT=$RUN_ROOT/generation
SERVER_LOG=$RUN_ROOT/server.log

server_started=0
cleanup() {
  status=$?
  if ((server_started)); then
    "$ROOT/scripts/stop_server.sh" || true
  fi
  exit "$status"
}
trap cleanup EXIT

echo "starting the frozen NVFP4 + MTP generation profile through the managed launcher"
PORT=$PORT \
MODEL_PATH=$MODEL_PATH \
MODEL_REVISION=$MODEL_REVISION \
SERVED_MODEL_NAME=$SERVED_MODEL_NAME \
CHAT_TEMPLATE=$CHAT_TEMPLATE \
QUANTIZATION=$QUANTIZATION \
KV_OFFLOAD_GB=$KV_OFFLOAD_GB \
ATTENTION_BACKEND=$ATTENTION_BACKEND \
MAX_MODEL_LEN=65536 \
MAX_NUM_BATCHED_TOKENS=4096 \
MAX_NUM_SEQS=2 \
NUM_SPEC_TOKENS=1 \
KV_CACHE_DTYPE=fp8_e4m3 \
GPU_MEMORY_UTILIZATION=0.85 \
LOG_PATH=$SERVER_LOG \
  "$ROOT/scripts/start_server.sh"
server_started=1

export SWE_IMAGE_CONFIG=$IMAGE_CONFIG
export LUMO_ENABLE_THINKING=true
export LUMO_PROXY_FORCE_TEMPERATURE=1.0
export LUMO_PROXY_FORCE_TOP_P=0.95
export LUMO_PROXY_FORCE_TOP_K=20
export LUMO_PROXY_FORCE_MIN_P=0.0
export LUMO_PROXY_FORCE_PRESENCE_PENALTY=0.0
export LUMO_PROXY_FORCE_SEED=880001234
export SWE_EMPTY_PATCH_RETRIES=0

"$SWE_PYTHON" "$ROOT/scripts/run_swe_verified.py" \
  --subset "$SUBSET" \
  --out-root "$OUT_ROOT" \
  --dataset-tag verified \
  --runtime container \
  --endpoint "$ENDPOINT" \
  --model "$SERVED_MODEL_NAME" \
  --model-name "qwen3.6-27b-nvfp4-mtp::qwen-code-0.19.4" \
  --repo-cache "$ROOT/.cache/swe_bench_repos" \
  --eval-mode skip \
  --agent-wall-s 900 \
  --qwen-max-wall 840s \
  --max-session-turns 50 \
  --proxy-port "$PROXY_PORT" \
  --proxy-max-tokens 8192 \
  --proxy-context-limit 65536 \
  --proxy-script "$ROOT/scripts/qwen_code_proxy.py" \
  --proxy-dump-dir "$RUN_ROOT/proxy_dumps" \
  --container-name-prefix swe_ep_lumo_state_v3 \
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

echo "V3 behavioral generation complete: $OUT_ROOT/verified"
