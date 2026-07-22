#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VLLM_PY=${VLLM_PY:-$ROOT/.venv-vllm/bin/python}
[[ -x "$VLLM_PY" ]] || { echo "missing vLLM Python: $VLLM_PY" >&2; exit 1; }

PURELIB=$($VLLM_PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export CUDA_HOME=${CUDA_HOME:-$PURELIB/nvidia/cu13}
CU13_LIB=$CUDA_HOME/lib
CUDA_DEV_LINKS=${CUDA_DEV_LINKS:-$ROOT/.cache/cuda_dev_links}
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "CUDA toolkit missing under $CUDA_HOME" >&2; exit 1; }

mkdir -p "$CUDA_DEV_LINKS"
find "$CUDA_DEV_LINKS" -maxdepth 1 -type l -delete
for lib in "$CU13_LIB"/*.so.*; do
  [[ -e "$lib" ]] || continue
  base=$(basename "$lib")
  link=${base%%.so.*}.so
  ln -s "$lib" "$CUDA_DEV_LINKS/$link"
done

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=${VLLM_DISABLED_KERNELS:-FlashInferFP8ScaledMMLinearKernel,FlashInferCutlassNvFp4LinearKernel,FlashInferTrtllmNvFp4LinearKernel,FlashInferCudnnNvFp4LinearKernel}
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export LIBRARY_PATH="$CUDA_DEV_LINKS:$CU13_LIB${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CU13_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

kv_offload=0
for argument in "$@"; do
  if [[ "$argument" == --kv-offloading-size || "$argument" == --kv-offloading-size=* ]]; then
    kv_offload=1
    break
  fi
done
if ((kv_offload)); then
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
fi

exec "$VLLM_PY" "$ROOT/scripts/run_swe_action_probe.py" "$@"
