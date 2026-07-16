#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TASK_ID=${1:-sympy__sympy-13480}
server_started=0

cleanup() {
  if (( server_started )); then
    "$ROOT/scripts/stop_server.sh" || true
  fi
}
trap cleanup EXIT INT TERM

"$ROOT/scripts/preflight.sh"
server_started=1
"$ROOT/scripts/start_server.sh"
"$ROOT/scripts/run_verified_task.sh" "$TASK_ID"
"$ROOT/scripts/stop_server.sh"
server_started=0
"$ROOT/scripts/score_verified.sh" "$TASK_ID"
