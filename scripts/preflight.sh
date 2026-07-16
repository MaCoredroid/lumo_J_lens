#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi
VLLM_BIN=${VLLM_BIN:-$ROOT/.venv-vllm/bin/vllm}
VLLM_PY=${VLLM_PY:-$(dirname "$VLLM_BIN")/python}
SWE_PYTHON=${SWE_PYTHON:-$ROOT/.venv-swe/bin/python}
QWEN_BIN=${QWEN_BIN:-$ROOT/node_modules/.bin/qwen}
MODEL_PATH=${MODEL_PATH:-nvidia/Qwen3.6-27B-NVFP4}
MODEL_REVISION=${MODEL_REVISION:-0893e1606ff3d5f97a441f405d5fc541a6bdf404}
TASK_ID=${TASK_ID:-sympy__sympy-13480}
EXPECTED_PYTHON_VERSION=3.12.13
EXPECTED_VLLM_VERSION=0.23.0
EXPECTED_SWEBENCH_VERSION=4.1.0
EXPECTED_DATASETS_VERSION=5.0.0
EXPECTED_DOCKER_PY_VERSION=7.1.0
EXPECTED_QWEN_VERSION=0.19.4
EXPECTED_UV_VERSION=0.11.24
EXPECTED_NODE_VERSION=v22.23.1
EXPECTED_NPM_VERSION=10.9.8
MIN_COMPUTE_CAPABILITY=${MIN_COMPUTE_CAPABILITY:-12.0}
MIN_RAM_KIB=${MIN_RAM_KIB:-$((30 * 1024 * 1024))}
MIN_SWAP_KIB=${MIN_SWAP_KIB:-$((4 * 1024 * 1024))}
fail=0

check_cmd() {
  if command -v "$1" >/dev/null; then
    printf 'PASS command %-12s %s\n' "$1" "$(command -v "$1")"
  else
    printf 'FAIL command %-12s missing\n' "$1" >&2
    fail=1
  fi
}

for cmd in nvidia-smi docker curl git node npm systemctl systemd-run uv; do check_cmd "$cmd"; done

if command -v uv >/dev/null; then
  uv_version=$(uv --version | awk '{print $2}')
  [[ "$uv_version" == "$EXPECTED_UV_VERSION" ]] || {
    echo "FAIL uv $uv_version; expected $EXPECTED_UV_VERSION" >&2; fail=1;
  }
fi
if command -v node >/dev/null; then
  node_version=$(node --version)
  [[ "$node_version" == "$EXPECTED_NODE_VERSION" ]] || {
    echo "FAIL Node.js $node_version; expected $EXPECTED_NODE_VERSION" >&2; fail=1;
  }
fi
if command -v npm >/dev/null; then
  npm_version=$(npm --version)
  [[ "$npm_version" == "$EXPECTED_NPM_VERSION" ]] || {
    echo "FAIL npm $npm_version; expected $EXPECTED_NPM_VERSION" >&2; fail=1;
  }
fi

if [[ -x "$ROOT/.venv-vllm/bin/python" ]]; then
  uv pip check --python "$ROOT/.venv-vllm/bin/python" || fail=1
fi
if [[ -x "$ROOT/.venv-swe/bin/python" ]]; then
  uv pip check --python "$ROOT/.venv-swe/bin/python" || fail=1
fi

if [[ -x "$VLLM_BIN" && -x "$VLLM_PY" ]]; then
  vllm_python_version=$("$VLLM_PY" -c 'import platform; print(platform.python_version())')
  [[ "$vllm_python_version" == "$EXPECTED_PYTHON_VERSION" ]] || {
    echo "FAIL vLLM Python $vllm_python_version; expected $EXPECTED_PYTHON_VERSION" >&2; fail=1;
  }
  vllm_version=$("$VLLM_BIN" --version)
  [[ "$vllm_version" == "$EXPECTED_VLLM_VERSION" ]] || {
    echo "FAIL vLLM $vllm_version; expected $EXPECTED_VLLM_VERSION" >&2; fail=1;
  }
  printf 'PASS vllm %s\n' "$vllm_version"
else
  echo "FAIL vLLM binary or Python missing: $VLLM_BIN / $VLLM_PY" >&2; fail=1
fi

if [[ -x "$SWE_PYTHON" ]]; then
  if ! "$SWE_PYTHON" - "$EXPECTED_PYTHON_VERSION" "$EXPECTED_SWEBENCH_VERSION" \
      "$EXPECTED_DATASETS_VERSION" "$EXPECTED_DOCKER_PY_VERSION" <<'PY'
import importlib.metadata as m
import platform
import sys
expected = dict(zip(("python", "swebench", "datasets", "docker"), sys.argv[1:]))
actual = {
    "python": platform.python_version(),
    "swebench": m.version("swebench"),
    "datasets": m.version("datasets"),
    "docker": m.version("docker"),
}
for name, version in actual.items():
    print("PASS", "docker-py" if name == "docker" else name, version)
if actual != expected:
    print(f"version mismatch: expected {expected}, got {actual}", file=sys.stderr)
    raise SystemExit(1)
PY
  then
    fail=1
  fi
else
  echo "FAIL SWE Python missing: $SWE_PYTHON" >&2; fail=1
fi

if [[ -x "$QWEN_BIN" ]]; then
  qwen_version=$("$QWEN_BIN" --version)
  [[ "$qwen_version" == "$EXPECTED_QWEN_VERSION" ]] || {
    echo "FAIL Qwen Code $qwen_version; expected $EXPECTED_QWEN_VERSION" >&2; fail=1;
  }
  printf 'PASS qwen-code %s\n' "$qwen_version"
else
  echo "FAIL Qwen Code missing: $QWEN_BIN" >&2; fail=1
fi

if command -v nvidia-smi >/dev/null; then
  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  IFS=, read -r total used compute driver < <(
    nvidia-smi --query-gpu=memory.total,memory.used,compute_cap,driver_version \
      --format=csv,noheader,nounits | head -1
  )
  total=${total// /}; used=${used// /}; compute=${compute// /}; driver=${driver// /}
  printf 'INFO GPU %s, total=%s MiB used=%s MiB compute=%s driver=%s\n' \
    "$gpu_name" "$total" "$used" "$compute" "$driver"
  (( total >= 32000 )) || { echo "FAIL this profile requires about 32 GB VRAM" >&2; fail=1; }
  awk -v actual="$compute" -v minimum="$MIN_COMPUTE_CAPABILITY" \
    'BEGIN { exit !((actual + 0) >= (minimum + 0)) }' || {
      echo "FAIL this profile requires compute capability >= $MIN_COMPUTE_CAPABILITY" >&2
      fail=1
    }
fi

read -r mem_kib swap_kib < <(
  awk '/^MemTotal:/ {mem=$2} /^SwapTotal:/ {swap=$2} END {print mem+0, swap+0}' /proc/meminfo
)
printf 'INFO host memory=%s MiB swap=%s MiB\n' "$((mem_kib / 1024))" "$((swap_kib / 1024))"
(( mem_kib >= MIN_RAM_KIB )) || {
  echo "FAIL this profile requires at least $((MIN_RAM_KIB / 1024 / 1024)) GiB RAM" >&2
  fail=1
}
(( swap_kib >= MIN_SWAP_KIB )) || {
  echo "FAIL this profile requires at least $((MIN_SWAP_KIB / 1024 / 1024)) GiB swap" >&2
  fail=1
}

if docker info >/dev/null 2>&1; then
  echo "PASS Docker daemon is usable without sudo"
else
  echo "FAIL Docker daemon unavailable to current user" >&2; fail=1
fi

image=$("$SWE_PYTHON" "$ROOT/scripts/resolve_swe_image.py" "$TASK_ID" --field reference)
pinned=$("$SWE_PYTHON" "$ROOT/scripts/resolve_swe_image.py" "$TASK_ID" --field pinned)
if [[ "$pinned" != true && ${ALLOW_UNPINNED_SWE_IMAGE:-0} != 1 ]]; then
  echo "FAIL no certified task-image digest for $TASK_ID" >&2
  fail=1
fi
if docker image inspect "$image" >/dev/null 2>&1; then
  echo "PASS task image cached: $image"
else
  echo "INFO task image not cached; run: scripts/pull_verified_image.sh $TASK_ID"
fi

if [[ -f "$MODEL_PATH/config.json" ]]; then
  echo "PASS local model snapshot: $MODEL_PATH"
elif [[ "$MODEL_PATH" == */* ]]; then
  if snapshot=$("$VLLM_PY" - "$MODEL_PATH" "$MODEL_REVISION" <<'PY' 2>/dev/null
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True))
PY
  ); then
    echo "PASS pinned model snapshot cached: $snapshot"
  else
    echo "FAIL pinned model snapshot is not cached; run scripts/download_model.sh" >&2
    fail=1
  fi
else
  echo "FAIL local model path has no config.json: $MODEL_PATH" >&2; fail=1
fi

template=$ROOT/configs/qwen3-openai-codex.jinja
template_sha=c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da
if [[ -f "$template" && $(sha256sum "$template" | awk '{print $1}') == "$template_sha" ]]; then
  echo "PASS chat template SHA-256: $template_sha"
else
  echo "FAIL chat template missing or unverified; run scripts/fetch_chat_template.sh" >&2
  fail=1
fi

(( fail == 0 )) || exit 1
echo "Preflight passed."
