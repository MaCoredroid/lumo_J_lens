#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z ${PYTHON_BIN+x} ]]; then
  if [[ -x "$ROOT/.venv-vllm/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv-vllm/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

for script in "$ROOT"/scripts/*.sh; do
  bash -n "$script"
done

mapfile -t python_sources < <(
  find "$ROOT/scripts" "$ROOT/tests" -maxdepth 1 -type f -name '*.py' -print | sort
)
"$PYTHON_BIN" -m py_compile "${python_sources[@]}"
"$PYTHON_BIN" "$ROOT/tests/test_proxy_envelope.py"
"$PYTHON_BIN" -m unittest discover -s "$ROOT/tests" -p 'test_*.py'
"$PYTHON_BIN" "$ROOT/scripts/check_validation.py"
"$PYTHON_BIN" "$ROOT/scripts/check_historical_validation.py"
"$PYTHON_BIN" "$ROOT/scripts/check_swe_multistage_publication.py"
"$PYTHON_BIN" "$ROOT/scripts/check_swe_behavioral_n20_publication.py"
"$PYTHON_BIN" "$ROOT/scripts/check_jlens_result.py"
"$PYTHON_BIN" "$ROOT/scripts/check_jlens_nf4_result.py"
"$PYTHON_BIN" "$ROOT/scripts/check_jlens_nvfp4_ste_result.py"
"$PYTHON_BIN" "$ROOT/scripts/analyze_swe_jlens_report.py" --check
(
  cd "$ROOT"
  sha256sum --check validation/jlens-nvfp4-2026-07-16.sha256
  sha256sum --check validation/jlens-nvfp4-ste-prefit-2026-07-16.sha256
  sha256sum --check validation/jlens-nvfp4-ste-evidence-2026-07-17.sha256
  sha256sum --check validation/jlens-swe-qwen-code-evidence-2026-07-17.sha256
  sha256sum --check validation/jlens-nf4-evidence.sha256
)

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
