#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SWE_PYTHON=${SWE_PYTHON:-$ROOT/.venv-swe/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-$ROOT/.venv-vllm/bin/python}
CHECKER=$ROOT/scripts/check_swe_task_state_validation_cohort.py
CAMPAIGN_RUNNER=$ROOT/scripts/run_swe_behavioral_campaign.sh
MATERIALIZER=$ROOT/scripts/materialize_swe_dense_behavioral_probes.py
HISTORICAL_MATERIALIZER=$ROOT/scripts/materialize_swe_behavioral_probes.py

COHORT_MANIFEST=$ROOT/configs/swe_task_state_validation_cohort.json
IMAGE_CONFIG=$ROOT/configs/swe_task_state_validation_image_digests.json
CAMPAIGN_A=$ROOT/configs/swe_task_state_validation_a_campaign.json
CAMPAIGN_B=$ROOT/configs/swe_task_state_validation_b_campaign.json
ACTION_PROTOCOL=$ROOT/configs/swe_stage_action_probes.json
CHAT_TEMPLATE=$ROOT/configs/qwen3-openai-codex.jinja

CHECKER_SHA256=9cb4a509ab1faa29d2ff0e64336e3fb283f803812037eaf6644f0d8ee25e9d6b
CAMPAIGN_RUNNER_SHA256=8792d039534b4c742ed0052c70d1c072ca1fe2a35e13ec369e81f8209a9008ef
MATERIALIZER_SHA256=a1c9b75b885fd9e50f59363c09d5aa47566eb7bbf7a91fe5aad6a8dccfd1cc5e
HISTORICAL_MATERIALIZER_SHA256=c63fac2907b887d973920c8fc71adf219affa1d6373a0aeb8ac2fffd59940a4e
COHORT_MANIFEST_SHA256=f0c2adfb562494362f359a60aa37a63f289af9f1b8b833805c07e2173aea6cbd
IMAGE_CONFIG_SHA256=87b5491b73fc77d8023b9855b6c7410ef78c77d6e0ed70a50c42f186fd289b17
CAMPAIGN_A_SHA256=eff5d681db4f1e51a4361e21baa6c3142400fcf68ec8d58b10d7193ea8e91651
CAMPAIGN_B_SHA256=d6596962167ed48e322638ae4a9758975b7a6630204cadc84ea6bead14bda4df
ACTION_PROTOCOL_SHA256=bce204d03608e181456bb5c05a041c4bf4d305f48cb4b4e651ba34460d46d493
CHAT_TEMPLATE_SHA256=c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da

RUN_NAME_A=swe_task_state_validation_a_20260718
RUN_NAME_B=swe_task_state_validation_b_20260718
RUN_ROOT_A=${RUN_ROOT_A:-$ROOT/runs/$RUN_NAME_A}
RUN_ROOT_B=${RUN_ROOT_B:-$ROOT/runs/$RUN_NAME_B}
PORT_A=${PORT_A:-9952}
PORT_B=${PORT_B:-$PORT_A}
PROXY_PORT_A=${PROXY_PORT_A:-30052}
PROXY_PORT_B=${PROXY_PORT_B:-30053}

OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/.cache/swe_task_state_validation_20260718}
PROMPTS_OUTPUT=${PROMPTS_OUTPUT:-$OUTPUT_ROOT/prompts.json}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-$OUTPUT_ROOT/prompts_summary.json}

mode=full

usage() {
  cat <<'EOF'
Usage: scripts/run_swe_task_state_validation.sh [--check-only | --materialize-only]

The default mode validates the frozen cohort, generates campaign A and then
campaign B with distinct run roots and proxy ports, and materializes one combined
all-probeable prompt bundle. --materialize-only reuses completed run roots.
--check-only performs no generation or materialization.

Relevant environment overrides:
  RUN_ROOT_A, RUN_ROOT_B       Frozen campaign output roots
  PORT_A, PORT_B               Shared vLLM endpoint port (must match)
  PROXY_PORT_A, PROXY_PORT_B   Sequential Qwen proxy ports
  OUTPUT_ROOT                  Combined prompt bundle directory
  SWE_PYTHON                   SWE environment Python executable
  VLLM_PYTHON                  tokenizer/materializer Python executable
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha256() {
  local path=$1
  local expected=$2
  local label=$3
  [[ -f "$path" && ! -L "$path" ]] || die "$label is not a regular file: $path"
  local actual
  actual=$(sha256_file "$path") || die "could not hash $label: $path"
  [[ "$actual" == "$expected" ]] || {
    die "$label SHA-256 mismatch: expected $expected, got $actual"
  }
}

while (($#)); do
  case "$1" in
    --check-only)
      [[ "$mode" == full ]] || die "only one mode flag may be supplied"
      mode=check
      ;;
    --materialize-only)
      [[ "$mode" == full ]] || die "only one mode flag may be supplied"
      mode=materialize
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
  shift
done

[[ -x "$SWE_PYTHON" ]] || die "SWE Python is missing or not executable: $SWE_PYTHON"
[[ -x "$VLLM_PYTHON" ]] || die "vLLM Python is missing or not executable: $VLLM_PYTHON"
require_sha256 "$CHECKER" "$CHECKER_SHA256" "validation cohort checker"
require_sha256 "$CAMPAIGN_RUNNER" "$CAMPAIGN_RUNNER_SHA256" "campaign runner"
require_sha256 "$MATERIALIZER" "$MATERIALIZER_SHA256" "dense materializer"
require_sha256 \
  "$HISTORICAL_MATERIALIZER" "$HISTORICAL_MATERIALIZER_SHA256" \
  "historical materializer dependency"
require_sha256 "$COHORT_MANIFEST" "$COHORT_MANIFEST_SHA256" "cohort manifest"
require_sha256 "$IMAGE_CONFIG" "$IMAGE_CONFIG_SHA256" "image registry"
require_sha256 "$CAMPAIGN_A" "$CAMPAIGN_A_SHA256" "campaign A config"
require_sha256 "$CAMPAIGN_B" "$CAMPAIGN_B_SHA256" "campaign B config"
require_sha256 "$ACTION_PROTOCOL" "$ACTION_PROTOCOL_SHA256" "action protocol"
require_sha256 "$CHAT_TEMPLATE" "$CHAT_TEMPLATE_SHA256" "chat template"
[[ "$RUN_ROOT_A" != "$RUN_ROOT_B" ]] || die "campaign run roots must differ"
[[ "$PORT_A" == "$PORT_B" ]] || {
  die "campaign endpoint ports must match to prevent two resident model servers"
}
[[ "$PROXY_PORT_A" != "$PROXY_PORT_B" ]] || die "campaign proxy ports must differ"
[[ "$(basename "$RUN_ROOT_A")" == "$RUN_NAME_A" ]] || {
  die "RUN_ROOT_A basename must match RUN_NAME_A"
}
[[ "$(basename "$RUN_ROOT_B")" == "$RUN_NAME_B" ]] || {
  die "RUN_ROOT_B basename must match RUN_NAME_B"
}
if [[ "$mode" == full ]]; then
  [[ ! -e "$RUN_ROOT_A" ]] || {
    die "campaign A run root already exists; use a fresh root or --materialize-only: $RUN_ROOT_A"
  }
  [[ ! -e "$RUN_ROOT_B" ]] || {
    die "campaign B run root already exists; use a fresh root or --materialize-only: $RUN_ROOT_B"
  }
fi

echo "[check] reproducing the frozen cohort from Docker and prior-use Git state"
"$SWE_PYTHON" "$CHECKER" \
  --cohort "$COHORT_MANIFEST" \
  --image-config "$IMAGE_CONFIG"

if [[ "$mode" == check ]]; then
  exit 0
fi

if [[ "$mode" == full ]]; then
  echo "[generate-a] run root: $RUN_ROOT_A; endpoint port: $PORT_A"
  CONFIG="$CAMPAIGN_A" \
  IMAGE_CONFIG="$IMAGE_CONFIG" \
  RUN_NAME="$RUN_NAME_A" \
  RUN_ROOT="$RUN_ROOT_A" \
  PORT="$PORT_A" \
  PROXY_PORT="$PROXY_PORT_A" \
  SERVER_LOG="$RUN_ROOT_A/server.log" \
    "$CAMPAIGN_RUNNER"

  echo "[generate-b] run root: $RUN_ROOT_B; endpoint port: $PORT_B"
  CONFIG="$CAMPAIGN_B" \
  IMAGE_CONFIG="$IMAGE_CONFIG" \
  RUN_NAME="$RUN_NAME_B" \
  RUN_ROOT="$RUN_ROOT_B" \
  PORT="$PORT_B" \
  PROXY_PORT="$PROXY_PORT_B" \
  SERVER_LOG="$RUN_ROOT_B/server.log" \
    "$CAMPAIGN_RUNNER"
fi

[[ -d "$RUN_ROOT_A" ]] || die "campaign A run root is missing: $RUN_ROOT_A"
[[ -d "$RUN_ROOT_B" ]] || die "campaign B run root is missing: $RUN_ROOT_B"
mkdir -p "$OUTPUT_ROOT"

echo "[materialize] combined all-probeable bundle: $PROMPTS_OUTPUT"
"$VLLM_PYTHON" "$MATERIALIZER" \
  --cohort "$CAMPAIGN_A" "$RUN_ROOT_A" \
  --cohort "$CAMPAIGN_B" "$RUN_ROOT_B" \
  --cohort-manifest "$COHORT_MANIFEST" \
  --action-protocol "$ACTION_PROTOCOL" \
  --template "$CHAT_TEMPLATE" \
  --all-probeable \
  --output "$PROMPTS_OUTPUT" \
  --summary "$SUMMARY_OUTPUT"

"$SWE_PYTHON" "$CHECKER" \
  --cohort "$COHORT_MANIFEST" \
  --image-config "$IMAGE_CONFIG" \
  --validate-bundle "$PROMPTS_OUTPUT" "$SUMMARY_OUTPUT"
echo "task-state validation bundle complete: $PROMPTS_OUTPUT"
