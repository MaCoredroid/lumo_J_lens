#!/usr/bin/env bash
# Drive the PRIMARY lanes (independent_a, independent_b) of a sealed-control
# suite to completion, idempotently. Mirrors the exact hand-driven recipe the
# Codex sessions used (freeze-primary -> run-primary -> finalize-primary per
# role/round), threading each stage's output-artifact sha256 from disk.
#
# Usage:  drive_sealed_primary.sh <suite-tag>       e.g.  drive_sealed_primary.sh r5
#
# Idempotent: a stage whose output artifact already exists is skipped.
# Exit codes:
#   0  all primary lanes finalized (ready for adjudicator)
#   3  a stage FAILED CLOSED (immutable failure) -> caller should fork a new suite
#   2  usage / precondition error
set -uo pipefail

TAG="${1:?usage: drive_sealed_primary.sh <suite-tag e.g. r5>}"
REPO=/home/mark/lumo_J_lens
ROOT=/home/mark/.cache/swe_task_state_v4_raw_capture/n60-final/sealed-control-v3-20260720-${TAG}
RUN=scripts/swe_task_state_v4_epistemic_chain_sealed_control_run_v3.py
cd "$REPO" || exit 2
[ -f "$ROOT/suite-init-receipt.json" ] || { echo "FATAL: no suite-init at $ROOT"; exit 2; }

ENVV=(env -u CUDA_HOME HF_HOME=/home/mark/.cache/huggingface HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel,FlashInferCutlassNvFp4LinearKernel,FlashInferTrtllmNvFp4LinearKernel,FlashInferCudnnNvFp4LinearKernel \
  CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0)
PY=.venv-vllm/bin/python
sha(){ sha256sum "$1" | awk '{print $1}'; }

# Stable source/config hashes (validated to match r5 suite-init).
EXEC_CFG=$(sha configs/swe_task_state_v4_epistemic_chain_sealed_control_executor_v3.json)
EXEC_SRC=$(sha scripts/swe_task_state_v4_epistemic_chain_sealed_control_executor_v3.py)
CTRL_SRC=$(sha scripts/swe_task_state_v4_epistemic_chain_sealed_control_run_v3.py)
ADPT_CFG=$(sha configs/swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3.json)
ADPT_SRC=$(sha scripts/swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3.py)
SI_FILE=$ROOT/suite-init-receipt.json
SI=$(sha "$SI_FILE")

echo "=== drive_sealed_primary $TAG @ $(date -u +%H:%M:%S)Z ==="
echo "controller=$CTRL_SRC executor=$EXEC_SRC adapter_src=$ADPT_SRC adapter_cfg=$ADPT_CFG"

# runner: execute a controller subcommand, tee to per-stage log, detect fail-closed
stage(){
  local name="$1"; shift
  local log="$ROOT/.driver-${name}.log"
  echo "--- STAGE $name @ $(date -u +%H:%M:%S)Z ---"
  if "${ENVV[@]}" $PY "$RUN" "$@" >"$log" 2>&1; then
    echo "    OK ($name)"; tail -2 "$log" | sed 's/^/    /'
    return 0
  else
    local rc=$?
    echo "    FAILED CLOSED ($name) rc=$rc"; tail -25 "$log" | sed 's/^/    /'
    return 3
  fi
}

freeze_initial(){ # role
  local role=$1; local s=${role#independent_}
  local out=$ROOT/independent-$s-initial
  stage "freeze-${role}-initial" freeze-primary --role "$role" --round initial \
    --suite-init "$SI_FILE" --suite-init-sha256 "$SI" \
    --executor-config-sha256 "$EXEC_CFG" --executor-source-sha256 "$EXEC_SRC" \
    --adapter-config-sha256 "$ADPT_CFG" --adapter-source-sha256 "$ADPT_SRC" \
    --controller-source-sha256 "$CTRL_SRC" --output-directory "$out"
}
run_round(){ # role round
  local role=$1; local round=$2; local s=${role#independent_}
  local fm=$ROOT/independent-$s-$round/freeze-manifest.json
  stage "run-${role}-${round}" run-primary \
    --freeze-manifest "$fm" --freeze-manifest-sha256 "$(sha "$fm")" \
    --suite-init "$SI_FILE" --suite-init-sha256 "$SI" \
    --executor-config-sha256 "$EXEC_CFG" --executor-source-sha256 "$EXEC_SRC" \
    --adapter-config-sha256 "$ADPT_CFG" --adapter-source-sha256 "$ADPT_SRC" \
    --trace-journal-sha256 "$(sha "$ROOT/read-trace.jsonl")"
}
freeze_detail(){ # role
  local role=$1; local s=${role#independent_}
  local out=$ROOT/independent-$s-detail
  local ib=$ROOT/independent-$s-initial/batch-artifact.json
  stage "freeze-${role}-detail" freeze-primary --role "$role" --round detail \
    --suite-init "$SI_FILE" --suite-init-sha256 "$SI" \
    --executor-config-sha256 "$EXEC_CFG" --executor-source-sha256 "$EXEC_SRC" \
    --adapter-config-sha256 "$ADPT_CFG" --adapter-source-sha256 "$ADPT_SRC" \
    --controller-source-sha256 "$CTRL_SRC" --output-directory "$out" \
    --initial-batch "$ib" --initial-batch-sha256 "$(sha "$ib")"
}
finalize(){ # role
  local role=$1; local s=${role#independent_}
  local ib=$ROOT/independent-$s-initial/batch-artifact.json
  local dfm=$ROOT/independent-$s-detail/freeze-manifest.json
  local db=$ROOT/independent-$s-detail/batch-artifact.json
  stage "finalize-${role}" finalize-primary --role "$role" \
    --suite-init "$SI_FILE" --suite-init-sha256 "$SI" \
    --initial-batch "$ib" --initial-batch-sha256 "$(sha "$ib")" \
    --detail-freeze-manifest "$dfm" --detail-freeze-manifest-sha256 "$(sha "$dfm")" \
    --detail-batch "$db" --detail-batch-sha256 "$(sha "$db")" \
    --adapter-config-sha256 "$ADPT_CFG" --adapter-source-sha256 "$ADPT_SRC" \
    --trace-journal-sha256 "$(sha "$ROOT/read-trace.jsonl")"
}

# Ordered primary-lane plan: (done-marker, action)
for role in independent_a independent_b; do
  s=${role#independent_}
  # initial freeze
  [ -f "$ROOT/independent-$s-initial/freeze-manifest.json" ] || freeze_initial "$role" || exit 3
  # initial run
  [ -f "$ROOT/independent-$s-initial/batch-artifact.json" ]  || run_round "$role" initial || exit 3
  # detail freeze
  [ -f "$ROOT/independent-$s-detail/freeze-manifest.json" ]  || freeze_detail "$role" || exit 3
  # detail run
  [ -f "$ROOT/independent-$s-detail/batch-artifact.json" ]   || run_round "$role" detail || exit 3
  # finalize / lock
  [ -f "$ROOT/${role}-primary-receipt.json" ]                || finalize "$role" || exit 3
  echo "=== PRIMARY LANE $role LOCKED ==="
done

echo "=== ALL PRIMARY LANES FINALIZED for $TAG — ready for adjudicator ==="
exit 0
