#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi
TASK_ID=${1:-sympy__sympy-13480}
ENDPOINT=${ENDPOINT:-http://127.0.0.1:${PORT:-9952}/v1}
MODEL=${SERVED_MODEL_NAME:-qwen3.6-27b-nvfp4}
SWE_PYTHON=${SWE_PYTHON:-$ROOT/.venv-swe/bin/python}
QWEN_BIN=${QWEN_BIN:-$ROOT/node_modules/.bin/qwen}
RUN_NAME=${RUN_NAME:-$(date -u +%Y%m%dT%H%M%SZ)_${TASK_ID//[^A-Za-z0-9_.-]/_}}
OUT_ROOT=${OUT_ROOT:-$ROOT/runs/$RUN_NAME/generation}
SUBSET=$ROOT/runs/$RUN_NAME/subset.json
DATASET_JSON=$ROOT/runs/$RUN_NAME/dataset.json
SWE_DATASET=${SWE_DATASET:-princeton-nlp/SWE-bench_Verified}
SWE_DATASET_REVISION=${SWE_DATASET_REVISION:-c104f840cc67f8b6eec6f759ebc8b2693d585d4a}

[[ -x "$SWE_PYTHON" ]] || { echo "missing SWE Python: $SWE_PYTHON" >&2; exit 1; }
[[ -x "$QWEN_BIN" ]] || { echo "missing Qwen Code: $QWEN_BIN" >&2; exit 1; }
qwen_version=$("$QWEN_BIN" --version)
[[ "$qwen_version" == 0.19.4 ]] || {
  echo "certified runner requires Qwen Code 0.19.4, got: $qwen_version" >&2
  exit 1
}
"$SWE_PYTHON" "$ROOT/scripts/check_endpoint.py" "$ENDPOINT" \
  --model "$MODEL" --max-model-len "${MAX_MODEL_LEN:-32768}"

"$ROOT/scripts/pull_verified_image.sh" "$TASK_ID"

mkdir -p "$(dirname "$SUBSET")" "$OUT_ROOT"
"$SWE_PYTHON" "$ROOT/scripts/materialize_verified_dataset.py" \
  --dataset "$SWE_DATASET" --revision "$SWE_DATASET_REVISION" \
  --instance-id "$TASK_ID" --output "$DATASET_JSON"
"$SWE_PYTHON" - "$DATASET_JSON" "$SUBSET" "$TASK_ID" <<'PY'
import json, sys
json.dump({"dataset_name": sys.argv[1], "instance_ids": [sys.argv[3]]}, open(sys.argv[2], "w"), indent=2)
open(sys.argv[2], "a").write("\n")
PY

export SWE_DOCKER_CMD=${SWE_DOCKER_CMD:-docker}
export LUMO_ENABLE_THINKING=${LUMO_ENABLE_THINKING:-true}
export LUMO_PROXY_FORCE_TEMPERATURE=${LUMO_PROXY_FORCE_TEMPERATURE:-1.0}
export LUMO_PROXY_FORCE_TOP_P=${LUMO_PROXY_FORCE_TOP_P:-0.95}
export LUMO_PROXY_FORCE_TOP_K=${LUMO_PROXY_FORCE_TOP_K:-20}
export LUMO_PROXY_FORCE_MIN_P=${LUMO_PROXY_FORCE_MIN_P:-0.0}
export LUMO_PROXY_FORCE_PRESENCE_PENALTY=${LUMO_PROXY_FORCE_PRESENCE_PENALTY:-0.0}
export LUMO_PROXY_FORCE_SEED=${LUMO_PROXY_FORCE_SEED:-880001234}
export SWE_EMPTY_PATCH_RETRIES=${SWE_EMPTY_PATCH_RETRIES:-1}

"$SWE_PYTHON" "$ROOT/scripts/run_swe_verified.py" \
  --subset "$SUBSET" \
  --out-root "$OUT_ROOT" \
  --dataset-tag verified \
  --runtime container \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --model-name "qwen3.6-27b-nvfp4-mtp::qwen-code-0.19.4" \
  --repo-cache "$ROOT/.cache/swe_bench_repos" \
  --eval-mode skip \
  --agent-wall-s 900 \
  --qwen-max-wall 840s \
  --max-session-turns 50 \
  --proxy-port "${PROXY_PORT:-30032}" \
  --proxy-max-tokens 8192 \
  --proxy-script "$ROOT/scripts/qwen_code_proxy.py" \
  --proxy-dump-dir "$ROOT/runs/$RUN_NAME/proxy_dumps" \
  --container-name-prefix "swe_ep_lumo" \
  --proxy-tool-choice "" \
  --proxy-required-tools run_shell_command \
  --core-tool run_shell_command \
  --exclude-tools tool_search \
  --exclude-tools web_fetch \
  --exclude-tools web_search \
  --exclude-tools agent \
  --exclude-tools skill \
  --exclude-tools task_stop \
  --exclude-tools send_message \
  --exclude-tools enter_worktree \
  --exclude-tools exit_worktree \
  --exclude-tools notebook_edit \
  --exclude-tools todo_write \
  --exclude-tools ask_user_question \
  --exclude-tools exit_plan_mode \
  --exclude-tools enter_plan_mode \
  --qwen-bin "$QWEN_BIN"

PREDICTIONS=$OUT_ROOT/verified/predictions.jsonl
"$SWE_PYTHON" - "$PREDICTIONS" "$TASK_ID" "$ROOT/runs/$RUN_NAME/proxy_dumps" <<'PY'
import glob, json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
matches = [row for row in rows if row.get("instance_id") == sys.argv[2]]
if len(matches) != 1 or not str(matches[0].get("model_patch") or "").strip():
    raise SystemExit(f"missing unique nonempty prediction for {sys.argv[2]}")
dump_files = sorted(glob.glob(sys.argv[3] + "/chat_*.json"))
if not dump_files:
    raise SystemExit("proxy produced no request dumps")
for path in dump_files:
    payload = json.load(open(path))
    tools = payload.get("tools")
    valid = (
        isinstance(tools, list)
        and len(tools) == 1
        and isinstance(tools[0], dict)
        and tools[0].get("type") == "function"
        and isinstance(tools[0].get("function"), dict)
        and tools[0]["function"].get("name") == "run_shell_command"
    )
    if not valid:
        raise SystemExit(f"unsafe tool boundary in {path}")
print(f"validated nonempty prediction: {sys.argv[2]}")
print(f"validated tool boundary across {len(dump_files)} request(s): run_shell_command only")
PY

echo "$RUN_NAME" > "$ROOT/runs/LATEST_RUN"
echo "generation complete: $PREDICTIONS"
echo "stop the model server before official scoring"
