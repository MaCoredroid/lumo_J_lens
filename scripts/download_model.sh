#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi
MODEL_PATH=${MODEL_PATH:-nvidia/Qwen3.6-27B-NVFP4}
MODEL_REVISION=${MODEL_REVISION:-0893e1606ff3d5f97a441f405d5fc541a6bdf404}
HF_BIN=${HF_BIN:-$ROOT/.venv-vllm/bin/hf}

[[ -x "$HF_BIN" ]] || { echo "missing $HF_BIN; run scripts/setup.sh first" >&2; exit 1; }

echo "Downloading $MODEL_PATH (about 22 GB of weights plus metadata)."
exec "$HF_BIN" download "$MODEL_PATH" --revision "$MODEL_REVISION"
