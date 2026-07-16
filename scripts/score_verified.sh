#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi
TASK_ID=${1:-sympy__sympy-13480}
RUN_NAME=${RUN_NAME:-$(cat "$ROOT/runs/LATEST_RUN" 2>/dev/null || true)}
[[ -n "$RUN_NAME" ]] || { echo "RUN_NAME is required when runs/LATEST_RUN is absent" >&2; exit 1; }
SWE_PYTHON=${SWE_PYTHON:-$ROOT/.venv-swe/bin/python}
PREDICTIONS=${PREDICTIONS:-$ROOT/runs/$RUN_NAME/generation/verified/predictions.jsonl}
DATASET_JSON=${DATASET_JSON:-$ROOT/runs/$RUN_NAME/dataset.json}
REPORT_DIR=${REPORT_DIR:-$ROOT/runs/$RUN_NAME/official_score}
RUN_ID=${SWE_RUN_ID:-lumo_j_lens_${RUN_NAME//[^A-Za-z0-9_.-]/_}}

[[ -x "$SWE_PYTHON" ]] || { echo "missing SWE Python: $SWE_PYTHON" >&2; exit 1; }
[[ -s "$PREDICTIONS" ]] || { echo "missing predictions: $PREDICTIONS" >&2; exit 1; }
[[ -s "$DATASET_JSON" ]] || { echo "missing pinned dataset: $DATASET_JSON" >&2; exit 1; }
if curl -fsS "http://127.0.0.1:${PORT:-9952}/v1/models" >/dev/null 2>&1; then
  echo "refusing to score while the 27B server is active; stop it to protect host RAM" >&2
  exit 1
fi

mkdir -p "$REPORT_DIR"
"$ROOT/scripts/pull_verified_image.sh" "$TASK_ID"
(
  cd "$REPORT_DIR"
  "$SWE_PYTHON" -m swebench.harness.run_evaluation \
    --dataset_name "$DATASET_JSON" \
    --split test \
    --predictions_path "$PREDICTIONS" \
    --instance_ids "$TASK_ID" \
    --run_id "$RUN_ID" \
    --namespace swebench \
    --cache_level env \
    --clean False \
    --max_workers 1 \
    --timeout 1800 \
    --report_dir "$REPORT_DIR"
)

report=$(find "$REPORT_DIR" -maxdepth 1 -type f -name "*.${RUN_ID}.json" | head -1)
[[ -n "$report" ]] || { echo "official harness did not produce a report" >&2; exit 1; }
"$SWE_PYTHON" - "$report" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("official report:", sys.argv[1])
print("resolved_ids:", r.get("resolved_ids", []))
print("unresolved_ids:", r.get("unresolved_ids", []))
print("error_ids:", r.get("error_ids", []))
raise SystemExit(0 if r.get("resolved_ids") else 2)
PY
