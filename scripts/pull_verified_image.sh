#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi
TASK_ID=${1:-sympy__sympy-13480}
PYTHON_BIN=${SWE_PYTHON:-$ROOT/.venv-swe/bin/python}
[[ -x "$PYTHON_BIN" ]] || { echo "missing SWE Python: $PYTHON_BIN" >&2; exit 1; }

tag=$("$PYTHON_BIN" "$ROOT/scripts/resolve_swe_image.py" "$TASK_ID" --field tag)
reference=$("$PYTHON_BIN" "$ROOT/scripts/resolve_swe_image.py" "$TASK_ID" --field reference)
expected_id=$("$PYTHON_BIN" "$ROOT/scripts/resolve_swe_image.py" "$TASK_ID" --field image_id)
current_id=$(docker image inspect "$tag" --format '{{.Id}}' 2>/dev/null || true)

if [[ -n "$expected_id" && "$current_id" == "$expected_id" ]]; then
  echo "verified pinned task image: $tag ($expected_id)"
  exit 0
fi
if [[ -z "$expected_id" && ${ALLOW_UNPINNED_SWE_IMAGE:-0} != 1 ]]; then
  echo "no certified image digest for $TASK_ID; set ALLOW_UNPINNED_SWE_IMAGE=1 to opt in" >&2
  exit 1
fi
if [[ -z "$expected_id" && -n "$current_id" ]]; then
  echo "using explicitly allowed, unpinned task image: $tag" >&2
  exit 0
fi

echo "pulling $reference"
docker pull "$reference"
if [[ "$reference" != "$tag" ]]; then
  docker tag "$reference" "$tag"
fi
current_id=$(docker image inspect "$tag" --format '{{.Id}}')
[[ -z "$expected_id" || "$current_id" == "$expected_id" ]] || {
  echo "task image ID mismatch: expected $expected_id, got $current_id" >&2
  exit 1
}
echo "task image ready: $tag ($current_id)"
