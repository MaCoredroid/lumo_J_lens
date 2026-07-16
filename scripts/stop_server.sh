#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi
UNIT=${UNIT:-lumo_j_lens_qwen27b}
if [[ -f "$ROOT/.server-state" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.server-state"
fi

systemctl --user stop "$UNIT.service" >/dev/null 2>&1 || true
rm -f "$ROOT/.server-state"

deadline=$((SECONDS + 180))
while (( SECONDS < deadline )); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if (( used < 8000 )); then
    echo "server stopped; GPU settled at ${used} MiB"
    exit 0
  fi
  sleep 5
done

echo "warning: GPU did not settle below 8000 MiB" >&2
exit 1
