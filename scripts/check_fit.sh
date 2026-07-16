#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON_VERSION=3.12.13
UV_VERSION=0.11.24
FIT_PYTHON="$ROOT/.venv-fit/bin/python"

command -v uv >/dev/null || {
  echo "missing required command: uv" >&2
  exit 1
}
[[ $(uv --version | awk '{print $2}') == "$UV_VERSION" ]] || {
  echo "fit checks require uv $UV_VERSION" >&2
  exit 1
}
if [[ ! -x "$FIT_PYTHON" ]]; then
  echo "missing .venv-fit; run scripts/setup_fit.sh" >&2
  exit 1
fi
[[ $("$FIT_PYTHON" -c 'import platform; print(platform.python_version())') == "$PYTHON_VERSION" ]] || {
  echo "fit checks require Python $PYTHON_VERSION" >&2
  exit 1
}

bash -n scripts/setup_fit.sh scripts/check_fit.sh
"$FIT_PYTHON" -m py_compile \
  scripts/compare_jlens_artifacts.py \
  scripts/evaluate_jlens_nf4.py \
  scripts/fit_jlens_nf4.py \
  scripts/materialize_jlens_fit_prompts.py \
  tests/test_compare_jlens_artifacts.py \
  tests/test_evaluate_jlens_nf4.py \
  tests/test_fit_jlens_nf4.py \
  tests/test_materialize_jlens_fit_prompts.py \
  tests/test_local_jlens_artifact.py
"$FIT_PYTHON" -c 'import bitsandbytes, datasets, torch, transformers'
"$FIT_PYTHON" -m unittest -v \
  tests/test_compare_jlens_artifacts.py \
  tests/test_evaluate_jlens_nf4.py \
  tests/test_fit_jlens_nf4.py \
  tests/test_materialize_jlens_fit_prompts.py \
  tests/test_local_jlens_artifact.py

uv pip check --python "$FIT_PYTHON"
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
uv pip freeze --python "$FIT_PYTHON" | sed 's/+cu130//' | sort \
  > "$tmp_dir/actual.txt"
sed 's/+cu130//' validation/fit-freeze.txt | sort > "$tmp_dir/expected.txt"
diff -u "$tmp_dir/expected.txt" "$tmp_dir/actual.txt"

echo "Fit environment and tests passed."
