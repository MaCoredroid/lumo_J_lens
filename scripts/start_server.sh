#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${LUMO_V3_CERTIFIED_NO_DOTENV:-0} != 1 && -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi
UNIT=${UNIT:-lumo_j_lens_qwen27b}
PORT=${PORT:-9952}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.6-27b-nvfp4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-620}
LOG_PATH=${LOG_PATH:-$ROOT/runs/server.log}
CHECK_PYTHON=${VLLM_PY:-$ROOT/.venv-vllm/bin/python}
booting=0
mkdir -p "$(dirname "$LOG_PATH")"
[[ -x "$CHECK_PYTHON" ]] || { echo "missing endpoint-check Python: $CHECK_PYTHON" >&2; exit 1; }

cleanup_failed_boot() {
  status=$?
  if (( booting )); then
    systemctl --user stop "$UNIT.service" >/dev/null 2>&1 || true
    rm -f "$ROOT/.server-state"
  fi
  return "$status"
}
trap cleanup_failed_boot EXIT

check_endpoint() {
  "$CHECK_PYTHON" "$ROOT/scripts/check_endpoint.py" "http://127.0.0.1:$PORT/v1" \
    --model "$SERVED_MODEL_NAME" --max-model-len "$MAX_MODEL_LEN" "$@"
}

if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  if [[ -f "$ROOT/.server-state" ]] && systemctl --user is-active --quiet "$UNIT.service"; then
    echo "restarting the existing managed server to reapply the certified profile"
    systemctl --user stop "$UNIT.service"
    rm -f "$ROOT/.server-state"
    for _ in {1..36}; do
      curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || break
      sleep 5
    done
    if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "managed endpoint did not stop cleanly" >&2
      exit 1
    fi
  else
    if check_endpoint >/dev/null 2>&1; then
      echo "a compatible but unmanaged server already owns port $PORT; refusing to adopt it" >&2
    else
      echo "an incompatible unmanaged server already owns port $PORT; refusing to replace it" >&2
    fi
    exit 1
  fi
fi

systemctl --user reset-failed "$UNIT.service" >/dev/null 2>&1 || true

env_args=()
for name in VLLM_BIN VLLM_PY MODEL_PATH MODEL_REVISION SERVED_MODEL_NAME PORT MAX_MODEL_LEN \
  MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS NUM_SPEC_TOKENS QUANTIZATION KV_CACHE_DTYPE \
  KV_OFFLOAD_GB ATTENTION_BACKEND GPU_MEMORY_UTILIZATION HF_HUB_OFFLINE \
  LUMO_V3_CERTIFIED_NO_DOTENV; do
  [[ -v "$name" ]] && env_args+=(--setenv="$name=${!name}")
done

booting=1
systemd-run --user --unit="$UNIT" --collect \
  -p MemoryMax=22G -p MemorySwapMax=4G \
  -p "WorkingDirectory=$ROOT" \
  -p "StandardOutput=append:$LOG_PATH" \
  -p "StandardError=append:$LOG_PATH" \
  "${env_args[@]}" \
  "$ROOT/scripts/serve_qwen36_27b_nvfp4_mtp.sh" >/dev/null

deadline=$((SECONDS + BOOT_TIMEOUT))
sleep 5
while (( SECONDS < deadline )); do
  if check_endpoint --quiet >/dev/null 2>&1; then
    printf 'UNIT=%q\nPORT=%q\nLOG_PATH=%q\n' "$UNIT" "$PORT" "$LOG_PATH" > "$ROOT/.server-state"
    booting=0
    echo "server ready: http://127.0.0.1:$PORT/v1"
    echo "log: $LOG_PATH"
    exit 0
  fi
  state=$(systemctl --user show -p ActiveState --value "$UNIT.service" 2>/dev/null || true)
  if [[ "$state" == inactive || "$state" == failed ]]; then
    echo "server service stopped during boot (state=$state)" >&2
    tail -100 "$LOG_PATH" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "server did not become ready within ${BOOT_TIMEOUT}s" >&2
tail -100 "$LOG_PATH" >&2 || true
systemctl --user stop "$UNIT.service" >/dev/null 2>&1 || true
exit 1
