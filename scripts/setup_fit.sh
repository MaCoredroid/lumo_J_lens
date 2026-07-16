#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON_VERSION=3.12.13
UV_VERSION=0.11.24

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: scripts/setup_fit.sh"
  exit 0
fi
if (($#)); then
  echo "setup_fit.sh does not accept arguments" >&2
  exit 2
fi

command -v uv >/dev/null || {
  echo "missing required command: uv" >&2
  exit 1
}
command -v git >/dev/null || {
  echo "missing required command: git" >&2
  exit 1
}
[[ $(uv --version | awk '{print $2}') == "$UV_VERSION" ]] || {
  echo "fit setup requires uv $UV_VERSION" >&2
  exit 1
}

constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
# The freeze records the installed CUDA wheel's local suffix. uv resolves the
# same wheel from the pinned CUDA 13.0 backend using the public version.
sed 's/+cu130//' validation/fit-freeze.txt > "$constraints"

if [[ ! -x .venv-fit/bin/python ]]; then
  uv venv --python "$PYTHON_VERSION" .venv-fit
fi
[[ $(.venv-fit/bin/python -c 'import platform; print(platform.python_version())') == "$PYTHON_VERSION" ]] || {
  echo "remove .venv-fit and rerun: required Python is $PYTHON_VERSION" >&2
  exit 1
}
uv pip sync --python .venv-fit/bin/python "$constraints" \
  --strict --torch-backend cu130
uv pip check --python .venv-fit/bin/python

.venv-fit/bin/python - <<'PY'
import importlib.metadata as metadata
import torch

print("installed fit environment:")
print("  torch", torch.__version__)
for package in ("transformers", "bitsandbytes", "datasets", "jlens"):
    print(f"  {package}", metadata.version(package))
PY
