#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${LUMO_V3_CERTIFIED_NO_DOTENV:-0} != 1 && -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi
VLLM_BIN=${VLLM_BIN:-$ROOT/.venv-vllm/bin/vllm}
VLLM_PY=${VLLM_PY:-$(dirname "$VLLM_BIN")/python}
MODEL_PATH=${MODEL_PATH:-nvidia/Qwen3.6-27B-NVFP4}
MODEL_REVISION=${MODEL_REVISION:-0893e1606ff3d5f97a441f405d5fc541a6bdf404}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.6-27b-nvfp4}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$ROOT/configs/qwen3-openai-codex.jinja}

PORT=${PORT:-9952}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-1}
QUANTIZATION=${QUANTIZATION:-modelopt_fp4}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-0}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-TRITON_ATTN}

[[ -x "$VLLM_BIN" ]] || { echo "missing vLLM: $VLLM_BIN" >&2; exit 1; }
[[ -x "$VLLM_PY" ]] || { echo "missing vLLM Python: $VLLM_PY" >&2; exit 1; }
[[ -f "$CHAT_TEMPLATE" ]] || { echo "missing chat template: $CHAT_TEMPLATE" >&2; exit 1; }

if [[ ${GPU_MEMORY_UTILIZATION:-auto} == auto ]]; then
  read -r gpu_used gpu_total < <(
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits |
      awk -F', *' 'NR==1 {print $1, $2}'
  )
  GPU_MEMORY_UTILIZATION=$(
    "$VLLM_PY" -c "print(f'{min(0.85, ($gpu_total-$gpu_used-1800)/$gpu_total):.2f}')"
  )
fi
"$VLLM_PY" -c "raise SystemExit(0 if float('$GPU_MEMORY_UTILIZATION') >= 0.74 else 1)" || {
  echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION is below certified floor 0.74" >&2
  exit 1
}

KV_OFFLOAD_GB=$(
  "$VLLM_PY" -c "v=max(0,min(float('$KV_OFFLOAD_GB'),8)); print(int(v) if v==int(v) else v)"
)

PURELIB=$("$VLLM_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
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

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_DISABLED_KERNELS=${VLLM_DISABLED_KERNELS:-FlashInferFP8ScaledMMLinearKernel,FlashInferCutlassNvFp4LinearKernel,FlashInferTrtllmNvFp4LinearKernel,FlashInferCudnnNvFp4LinearKernel}
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export LIBRARY_PATH="$CUDA_DEV_LINKS:$CU13_LIB${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CU13_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

kv_offload_args=()
if "$VLLM_PY" -c "raise SystemExit(0 if float('$KV_OFFLOAD_GB') > 0 else 1)"; then
  unset PYTORCH_CUDA_ALLOC_CONF
  kv_offload_args=(--kv-offloading-size "$KV_OFFLOAD_GB" --kv-offloading-backend native)
else
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
fi

revision_args=()
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  revision_args=(--revision "$MODEL_REVISION")
fi

echo "Serving $MODEL_PATH as $SERVED_MODEL_NAME on 127.0.0.1:$PORT" >&2
echo "profile: NVFP4, MTP=$NUM_SPEC_TOKENS, seqs=$MAX_NUM_SEQS, len=$MAX_MODEL_LEN, gmu=$GPU_MEMORY_UTILIZATION, kv=$KV_CACHE_DTYPE, offload=${KV_OFFLOAD_GB}G" >&2

# These are launcher-selection variables, not vLLM configuration variables.
# Keep their values in this shell but do not pass them to vLLM's env scanner.
export -n VLLM_BIN VLLM_PY 2>/dev/null || true

exec "$VLLM_BIN" serve "$MODEL_PATH" \
  "${revision_args[@]}" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --quantization "$QUANTIZATION" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gdn-prefill-backend triton \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-block-size 1024 \
  --mamba-ssm-cache-dtype float32 \
  "${kv_offload_args[@]}" \
  --speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}" \
  --attention-backend "$ATTENTION_BACKEND" \
  --no-enable-flashinfer-autotune \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
