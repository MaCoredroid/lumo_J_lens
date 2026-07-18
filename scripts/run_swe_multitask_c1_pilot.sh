#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VLLM_PY=${VLLM_PY:-"$ROOT/.venv-vllm/bin/python"}
OUT_DIR=${OUT_DIR:-"$ROOT/.cache/swe_multitask_c1"}
C0M_OUT_DIR=${C0M_OUT_DIR:-"$ROOT/.cache/swe_multitask_c0m"}
PROMPTS="$OUT_DIR/prompts.json"
PROMPT_SUMMARY="$OUT_DIR/prompts_summary.json"
PUBLIC_REPORT="$OUT_DIR/public-report.json"
NATIVE_REPORT="$OUT_DIR/native-report.json"
ANALYSIS="$OUT_DIR/analysis.json"
C0M_PROMPTS="$C0M_OUT_DIR/prompts.json"
C0M_PROMPT_SUMMARY="$C0M_OUT_DIR/prompts_summary.json"
C0M_PUBLIC_REPORT="$C0M_OUT_DIR/public-report.json"
C0M_NATIVE_REPORT="$C0M_OUT_DIR/native-report.json"
C0M_ANALYSIS="$C0M_OUT_DIR/analysis.json"
COMPARISON="$OUT_DIR/c0m-c1-comparison.json"
PREPARE_ONLY=0
REUSE_REPORTS=0
REUSE_C1_REPORTS=0

usage() {
  cat <<'EOF'
Usage: scripts/run_swe_multitask_c1_pilot.sh [--prepare-only] [--reuse-c1-reports] [--reuse-reports]

  --prepare-only       Rebuild and validate the matched C0M/C1 bundles, then stop.
  --reuse-c1-reports   Reuse C1 reports but run the capture-matched C0M replay.
  --reuse-reports      Skip all GPU replay and reanalyze existing paired reports.
EOF
}

while (($#)); do
  case "$1" in
    --prepare-only) PREPARE_ONLY=1 ;;
    --reuse-c1-reports) REUSE_C1_REPORTS=1 ;;
    --reuse-reports) REUSE_REPORTS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ -x "$VLLM_PY" ]] || {
  echo "missing Python environment: $VLLM_PY" >&2
  exit 1
}
mkdir -p "$OUT_DIR" "$C0M_OUT_DIR"

"$VLLM_PY" "$ROOT/scripts/materialize_swe_multitask_c1_probes.py" \
  --capture-root "$ROOT" \
  --output "$PROMPTS" \
  --summary "$PROMPT_SUMMARY"

"$VLLM_PY" "$ROOT/scripts/materialize_swe_multitask_capture_matched_probes.py" \
  --capture-root "$ROOT" \
  --c1-prompts "$PROMPTS" \
  --output "$C0M_PROMPTS" \
  --summary "$C0M_PROMPT_SUMMARY"

if ((PREPARE_ONLY)); then
  exit 0
fi

run_lens() {
  local label=$1
  local output=$2
  local prompts=$3
  shift 3
  local status

  set +e
  "$ROOT/scripts/run_jlens_nvfp4.sh" \
    "$@" \
    --prompts-file "$prompts" \
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

  # Exit 1 preserves the runner's strict reconstruction-certificate result.
  # The analyzer validates each certificate and the public/native pairing.
  if ((status > 1)); then
    echo "$label replay failed with status $status; see $OUT_DIR/$label-run.log" >&2
    exit "$status"
  fi
  [[ -s "$output" ]] || {
    echo "$label replay did not write $output; see $OUT_DIR/$label-run.log" >&2
    exit 1
  }
}

if ((!REUSE_REPORTS)); then
  run_lens c0m-public "$C0M_PUBLIC_REPORT" "$C0M_PROMPTS" --lens-kind public
  run_lens c0m-native "$C0M_NATIVE_REPORT" "$C0M_PROMPTS" \
    --lens-kind nvfp4-ste \
    --lens-path "$ROOT/.cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt" \
    --lens-sha256 82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057 \
    --lens-provenance "$ROOT/.cache/nvfp4_ste_fit/final-mean/metadata.json" \
    --lens-state "$ROOT/.cache/nvfp4_ste_fit/state.json" \
    --lens-state-sha256 f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6
  if ((!REUSE_C1_REPORTS)); then
    run_lens c1-public "$PUBLIC_REPORT" "$PROMPTS" --lens-kind public
    run_lens c1-native "$NATIVE_REPORT" "$PROMPTS" \
      --lens-kind nvfp4-ste \
      --lens-path "$ROOT/.cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt" \
      --lens-sha256 82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057 \
      --lens-provenance "$ROOT/.cache/nvfp4_ste_fit/final-mean/metadata.json" \
      --lens-state "$ROOT/.cache/nvfp4_ste_fit/state.json" \
      --lens-state-sha256 f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6
  fi
fi

for report in \
  "$C0M_PUBLIC_REPORT" "$C0M_NATIVE_REPORT" \
  "$PUBLIC_REPORT" "$NATIVE_REPORT"; do
  [[ -s "$report" ]] || {
    echo "missing replay report: $report" >&2
    exit 1
  }
done

"$VLLM_PY" "$ROOT/scripts/analyze_swe_multitask_initial_probes.py" \
  --prompts "$C0M_PROMPTS" \
  --public-report "$C0M_PUBLIC_REPORT" \
  --native-report "$C0M_NATIVE_REPORT" \
  --expected-checkpoint C0M \
  --output "$C0M_ANALYSIS"

"$VLLM_PY" "$ROOT/scripts/analyze_swe_multitask_initial_probes.py" \
  --prompts "$PROMPTS" \
  --public-report "$PUBLIC_REPORT" \
  --native-report "$NATIVE_REPORT" \
  --expected-checkpoint C1 \
  --output "$ANALYSIS"

"$VLLM_PY" "$ROOT/scripts/compare_swe_multitask_checkpoints.py" \
  --c0-analysis "$C0M_ANALYSIS" \
  --c0-checkpoint C0M \
  --c1-analysis "$ANALYSIS" \
  --output "$COMPARISON"

echo "C0M analysis: $C0M_ANALYSIS"
echo "C1 analysis: $ANALYSIS"
echo "capture-matched checkpoint comparison: $COMPARISON"
