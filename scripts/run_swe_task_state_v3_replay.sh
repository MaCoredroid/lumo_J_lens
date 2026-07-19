#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VLLM_PY=${VLLM_PY:-$ROOT/.venv-vllm/bin/python}
PIPELINE=$ROOT/scripts/swe_task_state_v3_replay_pipeline.py

[[ -x "$VLLM_PY" ]] || {
  echo "ERROR: pinned vLLM Python is missing or not executable: $VLLM_PY" >&2
  exit 1
}
[[ -f "$PIPELINE" && ! -L "$PIPELINE" ]] || {
  echo "ERROR: V3 replay pipeline is missing or unsafe: $PIPELINE" >&2
  exit 1
}

exec "$VLLM_PY" "$PIPELINE" "$@"
