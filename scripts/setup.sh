#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON_VERSION=3.12.13
UV_VERSION=0.11.24
NODE_VERSION=v22.23.1
NPM_VERSION=10.9.8

for cmd in uv node npm docker git curl; do
  command -v "$cmd" >/dev/null || { echo "missing required command: $cmd" >&2; exit 1; }
done
[[ $(uv --version | awk '{print $2}') == "$UV_VERSION" ]] || {
  echo "certified setup requires uv $UV_VERSION" >&2; exit 1;
}
[[ $(node --version) == "$NODE_VERSION" ]] || {
  echo "certified setup requires Node.js $NODE_VERSION" >&2; exit 1;
}
[[ $(npm --version) == "$NPM_VERSION" ]] || {
  echo "certified setup requires npm $NPM_VERSION" >&2; exit 1;
}

vllm_constraints=$(mktemp)
trap 'rm -f "$vllm_constraints"' EXIT
# uv resolves the CUDA wheel correctly as torch==2.11.0; the installed package
# reports 2.11.0+cu130. Strip only that local suffix from the observed freeze.
sed 's/+cu130//' validation/vllm-freeze.txt > "$vllm_constraints"

if [[ ! -x .venv-vllm/bin/python ]]; then
  uv venv --python "$PYTHON_VERSION" .venv-vllm
fi
[[ $(.venv-vllm/bin/python -c 'import platform; print(platform.python_version())') == "$PYTHON_VERSION" ]] || {
  echo "remove .venv-vllm and rerun: certified Python is $PYTHON_VERSION" >&2; exit 1;
}
uv pip sync --python .venv-vllm/bin/python "$vllm_constraints" \
  --strict --torch-backend cu130

if [[ ! -x .venv-swe/bin/python ]]; then
  uv venv --python "$PYTHON_VERSION" .venv-swe
fi
[[ $(.venv-swe/bin/python -c 'import platform; print(platform.python_version())') == "$PYTHON_VERSION" ]] || {
  echo "remove .venv-swe and rerun: certified Python is $PYTHON_VERSION" >&2; exit 1;
}
uv pip sync --python .venv-swe/bin/python validation/swe-freeze.txt --strict
uv pip check --python .venv-vllm/bin/python
uv pip check --python .venv-swe/bin/python

npm ci
"$ROOT/scripts/fetch_chat_template.sh"

echo "installed:"
.venv-vllm/bin/python -c 'import vllm; print("  vllm", vllm.__version__)'
.venv-swe/bin/python -c 'import importlib.metadata as m; print("  swebench", m.version("swebench"))'
node_modules/.bin/qwen --version | sed 's/^/  qwen-code /'

if [[ "${1:-}" == "--download-model" ]]; then
  "$ROOT/scripts/download_model.sh"
fi
