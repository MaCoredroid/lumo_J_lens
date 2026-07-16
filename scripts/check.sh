#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

for script in "$ROOT"/scripts/*.sh; do
  bash -n "$script"
done

"$PYTHON_BIN" -m py_compile \
  "$ROOT/scripts/qwen_code_proxy.py" \
  "$ROOT/scripts/check_endpoint.py" \
  "$ROOT/scripts/check_validation.py" \
  "$ROOT/scripts/materialize_verified_dataset.py" \
  "$ROOT/scripts/resolve_swe_image.py" \
  "$ROOT/scripts/run_swe_verified.py" \
  "$ROOT/tests/test_proxy_envelope.py" \
  "$ROOT/tests/test_proxy_context_clamp.py" \
  "$ROOT/tests/test_check_endpoint.py" \
  "$ROOT/tests/test_runner_safety.py"
"$PYTHON_BIN" "$ROOT/tests/test_proxy_envelope.py"
"$PYTHON_BIN" -m unittest discover -s "$ROOT/tests" -p 'test_*.py'
"$PYTHON_BIN" "$ROOT/scripts/check_validation.py"

if [[ -x "$ROOT/node_modules/.bin/qwen" ]]; then
  version=$("$ROOT/node_modules/.bin/qwen" --version)
  [[ "$version" == 0.19.4 ]] || {
    echo "unexpected Qwen Code version: $version" >&2
    exit 1
  }
else
  echo "INFO Qwen Code is not installed; run scripts/setup.sh to verify its pin"
fi

if command -v uv >/dev/null && [[ -x "$ROOT/.venv-vllm/bin/python" ]]; then
  uv pip check --python "$ROOT/.venv-vllm/bin/python"
  actual=$tmp_dir/vllm-actual.txt
  expected=$tmp_dir/vllm-expected.txt
  uv pip freeze --python "$ROOT/.venv-vllm/bin/python" | sed 's/+cu130//' | sort > "$actual"
  sed 's/+cu130//' "$ROOT/validation/vllm-freeze.txt" | sort > "$expected"
  diff -u "$expected" "$actual"
fi
if command -v uv >/dev/null && [[ -x "$ROOT/.venv-swe/bin/python" ]]; then
  uv pip check --python "$ROOT/.venv-swe/bin/python"
  actual_swe=$tmp_dir/swe-actual.txt
  uv pip freeze --python "$ROOT/.venv-swe/bin/python" | sort > "$actual_swe"
  diff -u <(sort "$ROOT/validation/swe-freeze.txt") "$actual_swe"
fi

echo "Static checks passed."
