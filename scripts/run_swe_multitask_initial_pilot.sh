#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SWE_PY=${SWE_PY:-"$ROOT/.venv-swe/bin/python"}
VLLM_PY=${VLLM_PY:-"$ROOT/.venv-vllm/bin/python"}
OUT_DIR=${OUT_DIR:-"$ROOT/.cache/swe_multitask_initial"}
PROTOCOL=${PROTOCOL:-"$ROOT/configs/swe_multitask_initial_protocol.json"}
CANDIDATES="$ROOT/.cache/swe_verified_initial_probe_candidates.json"
PROMPTS="$OUT_DIR/prompts.json"
PROMPT_SUMMARY="$OUT_DIR/prompts_summary.json"
PUBLIC_REPORT="$OUT_DIR/public-report.json"
NATIVE_REPORT="$OUT_DIR/native-report.json"
ANALYSIS="$OUT_DIR/analysis.json"
PREPARE_ONLY=0
REUSE_REPORTS=0

usage() {
  cat <<'EOF'
Usage: scripts/run_swe_multitask_initial_pilot.sh [--prepare-only] [--reuse-reports]

  --prepare-only    Regenerate and validate the frozen prompt bundle, then stop.
  --reuse-reports   Skip GPU replay and reanalyze the existing paired reports.
EOF
}

while (($#)); do
  case "$1" in
    --prepare-only) PREPARE_ONLY=1 ;;
    --reuse-reports) REUSE_REPORTS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

for executable in "$SWE_PY" "$VLLM_PY"; do
  [[ -x "$executable" ]] || {
    echo "missing Python environment: $executable" >&2
    exit 1
  }
done
mkdir -p "$OUT_DIR"

"$SWE_PY" "$ROOT/scripts/materialize_swe_verified_probe_candidates.py" \
  --output "$CANDIDATES"
"$VLLM_PY" "$ROOT/scripts/materialize_swe_multitask_initial_probes.py" \
  --protocol "$PROTOCOL" \
  --candidates "$CANDIDATES" \
  --output "$PROMPTS" \
  --summary "$PROMPT_SUMMARY"

if ((PREPARE_ONLY)); then
  exit 0
fi

run_lens() {
  local label=$1
  local output=$2
  shift 2
  local status

  set +e
  "$ROOT/scripts/run_jlens_nvfp4.sh" \
    "$@" \
    --prompts-file "$PROMPTS" \
    --layers "$(seq -s, 16 47)" \
    --positions=-1 \
    --top-k 10 \
    --max-model-len 16384 \
    --max-num-batched-tokens 4096 \
    --mamba-block-size 1024 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e4m3 \
    --stream-final-only \
    --gpu-memory-utilization 0.78 \
    --output "$output" \
    >/dev/null 2>"$OUT_DIR/$label-run.log"
  status=$?
  set -e

  # Exit 1 is the runner's documented strict reconstruction-certificate result.
  # The analyzer validates every reconstruction and pairing field from the report.
  if ((status > 1)); then
    echo "$label replay failed with status $status; see $OUT_DIR/$label-run.log" >&2
    exit "$status"
  fi
  if [[ ! -s "$output" ]]; then
    echo "$label replay did not write $output; see $OUT_DIR/$label-run.log" >&2
    exit 1
  fi
}

if ((!REUSE_REPORTS)); then
  run_lens public "$PUBLIC_REPORT" --lens-kind public
  run_lens native "$NATIVE_REPORT" \
    --lens-kind nvfp4-ste \
    --lens-path "$ROOT/.cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt" \
    --lens-sha256 82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057 \
    --lens-provenance "$ROOT/.cache/nvfp4_ste_fit/final-mean/metadata.json" \
    --lens-state "$ROOT/.cache/nvfp4_ste_fit/state.json" \
    --lens-state-sha256 f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6
fi

for report in "$PUBLIC_REPORT" "$NATIVE_REPORT"; do
  [[ -s "$report" ]] || {
    echo "missing replay report: $report" >&2
    exit 1
  }
done

"$VLLM_PY" "$ROOT/scripts/analyze_swe_multitask_initial_probes.py" \
  --prompts "$PROMPTS" \
  --public-report "$PUBLIC_REPORT" \
  --native-report "$NATIVE_REPORT" \
  --output "$ANALYSIS"

echo "analysis: $ANALYSIS"
