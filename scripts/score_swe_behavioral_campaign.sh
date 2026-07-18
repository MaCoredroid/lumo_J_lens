#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi

CONFIG=${CONFIG:-$ROOT/configs/swe_behavioral_campaign.json}
RUN_NAME=${RUN_NAME:-$(cat "$ROOT/runs/LATEST_BEHAVIORAL_RUN" 2>/dev/null || true)}
[[ -n "$RUN_NAME" ]] || {
  echo "ERROR: RUN_NAME is required when runs/LATEST_BEHAVIORAL_RUN is absent" >&2
  exit 1
}
[[ "$RUN_NAME" =~ ^[A-Za-z0-9_.-]+$ && "$RUN_NAME" != "." && "$RUN_NAME" != ".." ]] || {
  echo "ERROR: RUN_NAME must be a safe path/run identifier" >&2
  exit 1
}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/$RUN_NAME}
SWE_PYTHON=${SWE_PYTHON:-$ROOT/.venv-swe/bin/python}
DATASET_JSON=${DATASET_JSON:-$RUN_ROOT/dataset.json}
IMAGE_MANIFEST=${IMAGE_MANIFEST:-$RUN_ROOT/image_manifest.json}
SOURCE_PREDICTIONS=${SOURCE_PREDICTIONS:-$RUN_ROOT/generation/verified/predictions.jsonl}
REPORT_ROOT=${REPORT_ROOT:-$RUN_ROOT/official_score}
RUN_ROOT=$(realpath -m -- "$RUN_ROOT")
SWE_PYTHON=$(cd "$(dirname "$SWE_PYTHON")" && pwd -P)/$(basename "$SWE_PYTHON")
DATASET_JSON=$(realpath -m -- "$DATASET_JSON")
IMAGE_MANIFEST=$(realpath -m -- "$IMAGE_MANIFEST")
SOURCE_PREDICTIONS=$(realpath -m -- "$SOURCE_PREDICTIONS")
REPORT_ROOT=$(realpath -m -- "$REPORT_ROOT")
NORMALIZED_OUTCOMES=$REPORT_ROOT/official_outcomes.json
ENDPOINT=${ENDPOINT:-http://127.0.0.1:${PORT:-9952}/v1}
SWE_TIMEOUT=${SWE_TIMEOUT:-1800}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -f "$CONFIG" && ! -L "$CONFIG" ]] || die "campaign config is missing or is a symlink: $CONFIG"
[[ -f "$DATASET_JSON" && ! -L "$DATASET_JSON" ]] || die "pinned local dataset is missing or is a symlink: $DATASET_JSON"
[[ -f "$IMAGE_MANIFEST" && ! -L "$IMAGE_MANIFEST" ]] || die "image manifest is missing or is a symlink: $IMAGE_MANIFEST"
[[ -x "$SWE_PYTHON" ]] || die "SWE Python is missing: $SWE_PYTHON"
[[ "$SWE_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "SWE_TIMEOUT must be a positive integer"
[[ ! -L "$REPORT_ROOT" ]] || die "report root must not be a symlink: $REPORT_ROOT"

mkdir -p "$REPORT_ROOT" "$REPORT_ROOT/attempts" "$REPORT_ROOT/archive"
LOCK_DIR=$REPORT_ROOT/.score.lock
mkdir "$LOCK_DIR" 2>/dev/null || die "another behavioral scoring process holds $LOCK_DIR"
PREP_DIR=$(mktemp -d "$REPORT_ROOT/.prepare.XXXXXX")
lock_held=1
cleanup() {
  status=$?
  if ((lock_held)); then
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
  trap - EXIT
  exit "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

COMPLETE_PREDICTIONS=$PREP_DIR/predictions.complete.jsonl
INPUT_MANIFEST=$PREP_DIR/score_inputs.json
RUN_ID_FILE=$PREP_DIR/run_id.txt
PREFLIGHT_STDOUT=$PREP_DIR/preflight.stdout.log
PREFLIGHT_STDERR=$PREP_DIR/preflight.stderr.log
PREFLIGHT_STATUS=$PREP_DIR/preflight.exit_status

set +e
"$SWE_PYTHON" - \
  "$CONFIG" "$DATASET_JSON" "$IMAGE_MANIFEST" "$SOURCE_PREDICTIONS" \
  "$COMPLETE_PREDICTIONS" "$INPUT_MANIFEST" "$RUN_ID_FILE" "$RUN_NAME" <<'PY' \
  >"$PREFLIGHT_STDOUT" 2>"$PREFLIGHT_STDERR"
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(message)


def read_bytes(path: Path, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        fail(f"{label} is not a regular non-symlink file: {path}")
    return path.read_bytes()


def load_json(path: Path, label: str):
    try:
        return json.loads(read_bytes(path, label))
    except json.JSONDecodeError as error:
        fail(f"{label} is not valid JSON: {error}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


config_path = Path(sys.argv[1]).resolve()
dataset_path = Path(sys.argv[2]).resolve()
image_manifest_path = Path(sys.argv[3]).resolve()
source_predictions_path = Path(sys.argv[4]).resolve()
complete_predictions_path = Path(sys.argv[5])
input_manifest_path = Path(sys.argv[6])
run_id_path = Path(sys.argv[7])
run_name = sys.argv[8]

try:
    swebench_version = importlib.metadata.version("swebench")
except importlib.metadata.PackageNotFoundError:
    fail("swebench is not installed in SWE_PYTHON")
if swebench_version != "4.1.0":
    fail(f"SWE-bench 4.1.0 is required, got {swebench_version}")

config_bytes = read_bytes(config_path, "campaign config")
config = load_json(config_path, "campaign config")
if config.get("schema_version") != 1:
    fail("unsupported behavioral campaign schema")
if config.get("kind") != "swe_verified_behavioral_trajectory_campaign":
    fail("unexpected behavioral campaign kind")
instance_ids = config.get("instance_ids")
instance_pattern = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+$")
if (
    not isinstance(instance_ids, list)
    or len(instance_ids) != 10
    or len(set(instance_ids)) != len(instance_ids)
    or not all(isinstance(value, str) and instance_pattern.fullmatch(value) for value in instance_ids)
):
    fail("campaign must contain ten unique safe SWE instance IDs")

dataset_bytes = read_bytes(dataset_path, "pinned local dataset")
dataset = load_json(dataset_path, "pinned local dataset")
if not isinstance(dataset, list) or len(dataset) != len(instance_ids):
    fail("pinned local dataset must contain exactly ten rows")
dataset_ids = []
for index, row in enumerate(dataset):
    if not isinstance(row, dict) or not isinstance(row.get("instance_id"), str):
        fail(f"dataset row {index} has no string instance_id")
    dataset_ids.append(row["instance_id"])
if dataset_ids != instance_ids:
    fail("pinned local dataset order/coverage does not match campaign config")

image_manifest_bytes = read_bytes(image_manifest_path, "image manifest")
image_manifest = load_json(image_manifest_path, "image manifest")
if image_manifest.get("schema_version") != 1:
    fail("unsupported image manifest schema")
if image_manifest.get("kind") != "swe_verified_behavioral_campaign_image_manifest":
    fail("unexpected image manifest kind")
if image_manifest.get("campaign_config_sha256") != sha256_bytes(config_bytes):
    fail("image manifest campaign config hash mismatch")
if image_manifest.get("dataset") != config.get("dataset"):
    fail("image manifest dataset pin mismatch")
images = image_manifest.get("images")
if not isinstance(images, list) or [item.get("instance_id") for item in images if isinstance(item, dict)] != instance_ids:
    fail("image manifest order/coverage does not match campaign config")
docker = shutil.which("docker")
if docker is None:
    fail("docker is unavailable")
for instance_id, image in zip(instance_ids, images, strict=True):
    if not isinstance(image, dict):
        fail(f"invalid image manifest record for {instance_id}")
    expected_tag = (
        "swebench/sweb.eval.x86_64."
        + instance_id.replace("__", "_1776_")
        + ":latest"
    )
    if image.get("tag") != expected_tag:
        fail(f"image tag mismatch for {instance_id}")
    image_id = image.get("image_id")
    repo_digests = image.get("repo_digests")
    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        fail(f"invalid image ID for {instance_id}")
    if not isinstance(repo_digests, list) or not repo_digests or not all(isinstance(value, str) for value in repo_digests):
        fail(f"missing immutable repository digest for {instance_id}")
    try:
        inspected = json.loads(
            subprocess.check_output(
                [docker, "image", "inspect", expected_tag], text=True
            )
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        fail(f"cannot inspect pinned image for {instance_id}: {error}")
    if len(inspected) != 1 or inspected[0].get("Architecture") != "amd64":
        fail(f"unexpected local image inspection for {instance_id}")
    if inspected[0].get("Id") != image_id:
        fail(f"local image ID changed for {instance_id}")
    if sorted(inspected[0].get("RepoDigests") or []) != sorted(repo_digests):
        fail(f"local image repository digests changed for {instance_id}")

generation = config.get("generation")
if not isinstance(generation, dict):
    fail("campaign generation contract is missing")
served_model = generation.get("served_model")
qwen_version = generation.get("qwen_code_version")
if not isinstance(served_model, str) or not served_model:
    fail("campaign served model is invalid")
if not isinstance(qwen_version, str) or not qwen_version:
    fail("campaign Qwen Code version is invalid")
model_prefix = served_model if served_model.endswith("-mtp") else served_model + "-mtp"
expected_model_name = f"{model_prefix}::qwen-code-{qwen_version}"

if source_predictions_path.is_symlink():
    fail("source predictions path must not be a symlink")
source_exists = source_predictions_path.is_file()
source_bytes = source_predictions_path.read_bytes() if source_exists else b""
if source_predictions_path.exists() and not source_exists:
    fail("source predictions path exists but is not a regular non-symlink file")
source_rows = {}
if source_exists:
    for line_number, raw_line in enumerate(source_bytes.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            fail(f"source predictions line {line_number} is invalid JSON: {error}")
        if not isinstance(row, dict):
            fail(f"source predictions line {line_number} is not an object")
        instance_id = row.get("instance_id")
        if instance_id not in instance_ids:
            fail(f"source predictions contains an unconfigured instance: {instance_id!r}")
        if instance_id in source_rows:
            fail(f"source predictions contains duplicate instance: {instance_id}")
        patch = row.get("model_patch")
        model_name = row.get("model_name_or_path")
        if not isinstance(patch, str):
            fail(f"source prediction patch is not a string for {instance_id}")
        if model_name != expected_model_name:
            fail(f"source prediction model identity mismatch for {instance_id}")
        source_rows[instance_id] = row

complete_rows = []
missing_ids = []
empty_ids = []
for instance_id in instance_ids:
    source = source_rows.get(instance_id)
    if source is None:
        missing_ids.append(instance_id)
        patch = ""
    else:
        patch = source["model_patch"]
    if not patch.strip():
        patch = ""
        empty_ids.append(instance_id)
    complete_rows.append(
        {
            "instance_id": instance_id,
            "model_name_or_path": expected_model_name,
            "model_patch": patch,
        }
    )

complete_bytes = b"".join(canonical_bytes(row) + b"\n" for row in complete_rows)
complete_predictions_path.write_bytes(complete_bytes)
input_hashes = {
    "campaign_config_sha256": sha256_bytes(config_bytes),
    "dataset_sha256": sha256_bytes(dataset_bytes),
    "image_manifest_sha256": sha256_bytes(image_manifest_bytes),
    "source_predictions_sha256": sha256_bytes(source_bytes) if source_exists else None,
    "complete_predictions_sha256": sha256_bytes(complete_bytes),
}
evidence_payload = {
    "hashes": input_hashes,
    "instance_ids": instance_ids,
    "model_name_or_path": expected_model_name,
    "swebench_version": swebench_version,
}
evidence_id = hashlib.sha256(canonical_bytes(evidence_payload)).hexdigest()
safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("._-")
if not safe_run_name:
    fail("RUN_NAME has no safe run-ID characters")
safe_run_name = safe_run_name[:64]
run_id = f"lumo_j_lens_{safe_run_name}_behavioral_{evidence_id[:16]}"
manifest = {
    "schema_version": 1,
    "kind": "swe_verified_behavioral_official_score_inputs",
    "run_name": run_name,
    "run_id": run_id,
    "evidence_id": evidence_id,
    "swebench_version": swebench_version,
    "instance_ids": instance_ids,
    "expected_model_name_or_path": expected_model_name,
    "missing_prediction_ids": missing_ids,
    "empty_prediction_ids": empty_ids,
    "paths": {
        "campaign_config": str(config_path),
        "dataset": str(dataset_path),
        "image_manifest": str(image_manifest_path),
        "source_predictions": str(source_predictions_path),
    },
    "source_predictions_exists": source_exists,
    "hashes": input_hashes,
}
input_manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    encoding="ascii",
)
run_id_path.write_text(run_id + "\n", encoding="ascii")
PY
preflight_status=$?
set -e
printf '%s\n' "$preflight_status" >"$PREFLIGHT_STATUS"
if ((preflight_status != 0)); then
  die "behavioral score preflight failed; see $PREFLIGHT_STDERR"
fi

RUN_ID=$(<"$RUN_ID_FILE")
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || die "preflight emitted an unsafe run ID"
ATTEMPT_BASE=$REPORT_ROOT/attempts/$RUN_ID
mkdir -p "$ATTEMPT_BASE"
attempt_number=1
while :; do
  ATTEMPT_DIR=$ATTEMPT_BASE/$(printf 'attempt-%04d' "$attempt_number")
  if mkdir "$ATTEMPT_DIR" 2>/dev/null; then
    break
  elif [[ ! -e "$ATTEMPT_DIR" ]]; then
    die "cannot create fresh attempt directory: $ATTEMPT_DIR"
  fi
  ((attempt_number += 1))
done
mv "$PREP_DIR"/* "$ATTEMPT_DIR"/
rmdir "$PREP_DIR"
PREP_DIR=
COMPLETE_PREDICTIONS=$ATTEMPT_DIR/predictions.complete.jsonl
INPUT_MANIFEST=$ATTEMPT_DIR/score_inputs.json

# A fresh invocation never consumes a previous official result. Archive the
# stable output before running so a failed rerun cannot leave stale evidence at
# the canonical path.
if [[ -e "$NORMALIZED_OUTCOMES" || -L "$NORMALIZED_OUTCOMES" ]]; then
  [[ -f "$NORMALIZED_OUTCOMES" && ! -L "$NORMALIZED_OUTCOMES" ]] || \
    die "normalized outcome path is not a regular file: $NORMALIZED_OUTCOMES"
  archive_number=1
  while :; do
    archive_path=$REPORT_ROOT/archive/$(printf 'official_outcomes-%04d.json' "$archive_number")
    if [[ ! -e "$archive_path" ]]; then
      mv "$NORMALIZED_OUTCOMES" "$archive_path"
      break
    elif [[ -L "$archive_path" ]]; then
      die "archive outcome path must not be a symlink: $archive_path"
    fi
    ((archive_number += 1))
  done
fi

SERVER_STDOUT=$ATTEMPT_DIR/server_stop.stdout.log
SERVER_STDERR=$ATTEMPT_DIR/server_stop.stderr.log
SERVER_STATUS=$ATTEMPT_DIR/server_stop.exit_status
if curl -fsS --max-time 2 "$ENDPOINT/models" >/dev/null 2>&1; then
  set +e
  "$ROOT/scripts/stop_server.sh" >"$SERVER_STDOUT" 2>"$SERVER_STDERR"
  server_status=$?
  set -e
  printf '%s\n' "$server_status" >"$SERVER_STATUS"
  if ((server_status != 0)); then
    die "model server stop did not settle cleanly; see $SERVER_STDERR"
  fi
  sleep 2
  if curl -fsS --max-time 2 "$ENDPOINT/models" >/dev/null 2>&1; then
    die "model server remains live at $ENDPOINT; refusing official scoring"
  fi
else
  : >"$SERVER_STDOUT"
  : >"$SERVER_STDERR"
  printf 'not_running\n' >"$SERVER_STATUS"
fi

mapfile -t INSTANCE_IDS < <(
  "$SWE_PYTHON" - "$INPUT_MANIFEST" <<'PY'
import json
import sys

for instance_id in json.load(open(sys.argv[1], encoding="ascii"))["instance_ids"]:
    print(instance_id)
PY
)
[[ ${#INSTANCE_IDS[@]} -eq 10 ]] || die "preflight instance list changed"

HARNESS_ROOT=$ATTEMPT_DIR/harness
# SWE-bench 4.1 writes the aggregate report in its working directory even
# when --report_dir is supplied. Keep both locations identical and sealed.
HARNESS_REPORT_DIR=$HARNESS_ROOT
mkdir -p "$HARNESS_ROOT"
HARNESS_STDOUT=$ATTEMPT_DIR/harness.stdout.log
HARNESS_STDERR=$ATTEMPT_DIR/harness.stderr.log
HARNESS_STATUS=$ATTEMPT_DIR/harness.exit_status
HARNESS_COMMAND=$ATTEMPT_DIR/harness_command.json
harness_argv=(
  "$SWE_PYTHON" -m swebench.harness.run_evaluation
  --dataset_name "$DATASET_JSON"
  --split test
  --predictions_path "$COMPLETE_PREDICTIONS"
  --instance_ids "${INSTANCE_IDS[@]}"
  --run_id "$RUN_ID"
  --namespace swebench
  --cache_level env
  --clean False
  --max_workers 1
  --timeout "$SWE_TIMEOUT"
  --report_dir "$HARNESS_REPORT_DIR"
)
"$SWE_PYTHON" - "$HARNESS_COMMAND" "${harness_argv[@]}" <<'PY'
import json
import sys

with open(sys.argv[1], "w", encoding="ascii") as handle:
    json.dump(sys.argv[2:], handle, indent=2, ensure_ascii=True)
    handle.write("\n")
PY

set +e
(
  cd "$HARNESS_ROOT"
  "${harness_argv[@]}"
) >"$HARNESS_STDOUT" 2>"$HARNESS_STDERR"
harness_status=$?
set -e
printf '%s\n' "$harness_status" >"$HARNESS_STATUS"
if ((harness_status != 0)); then
  die "official SWE-bench harness failed with status $harness_status; see $HARNESS_STDERR"
fi

mapfile -t aggregate_reports < <(
  find "$HARNESS_REPORT_DIR" -maxdepth 1 -type f -name "*.${RUN_ID}.json" -print | sort
)
[[ ${#aggregate_reports[@]} -eq 1 ]] || \
  die "official harness must emit exactly one aggregate report for $RUN_ID"
AGGREGATE_REPORT=${aggregate_reports[0]}

VALIDATION_STDOUT=$ATTEMPT_DIR/validation.stdout.log
VALIDATION_STDERR=$ATTEMPT_DIR/validation.stderr.log
VALIDATION_STATUS=$ATTEMPT_DIR/validation.exit_status
set +e
"$SWE_PYTHON" - \
  "$INPUT_MANIFEST" "$COMPLETE_PREDICTIONS" "$AGGREGATE_REPORT" \
  "$HARNESS_ROOT" "$ATTEMPT_DIR" "$NORMALIZED_OUTCOMES" "$SWE_TIMEOUT" <<'PY' \
  >"$VALIDATION_STDOUT" 2>"$VALIDATION_STDERR"
import hashlib
import json
import os
import re
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str):
    if not path.is_file() or path.is_symlink():
        fail(f"{label} is not a regular non-symlink file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail(f"{label} is invalid JSON: {error}")


input_manifest_path = Path(sys.argv[1]).resolve()
complete_predictions_path = Path(sys.argv[2]).resolve()
aggregate_path = Path(sys.argv[3]).resolve()
harness_root = Path(sys.argv[4]).resolve()
attempt_dir = Path(sys.argv[5]).resolve()
normalized_path = Path(sys.argv[6]).resolve()
timeout = int(sys.argv[7])

inputs = load_json(input_manifest_path, "score input manifest")
if inputs.get("schema_version") != 1:
    fail("unsupported score input manifest")
instance_ids = inputs.get("instance_ids")
if not isinstance(instance_ids, list) or len(instance_ids) != 10:
    fail("score input manifest no longer contains ten instances")
expected_set = set(instance_ids)
hashes = inputs.get("hashes")
paths = inputs.get("paths")
if not isinstance(hashes, dict) or not isinstance(paths, dict):
    fail("score input hash/path bindings are missing")
for key, hash_key in (
    ("campaign_config", "campaign_config_sha256"),
    ("dataset", "dataset_sha256"),
    ("image_manifest", "image_manifest_sha256"),
):
    path = Path(paths[key])
    if sha256_file(path) != hashes[hash_key]:
        fail(f"{key} changed after score preflight")
source_path = Path(paths["source_predictions"])
if inputs.get("source_predictions_exists"):
    if sha256_file(source_path) != hashes["source_predictions_sha256"]:
        fail("source predictions changed after score preflight")
elif source_path.exists() or source_path.is_symlink():
    fail("source predictions appeared after score preflight")
if sha256_file(complete_predictions_path) != hashes["complete_predictions_sha256"]:
    fail("completed predictions changed after score preflight")

evidence_payload = {
    "hashes": hashes,
    "instance_ids": instance_ids,
    "model_name_or_path": inputs.get("expected_model_name_or_path"),
    "swebench_version": inputs.get("swebench_version"),
}
evidence_id = hashlib.sha256(
    json.dumps(
        evidence_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
).hexdigest()
if inputs.get("evidence_id") != evidence_id:
    fail("score input evidence ID mismatch")
safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", inputs.get("run_name", "")).strip("._-")[:64]
expected_run_id = f"lumo_j_lens_{safe_run_name}_behavioral_{evidence_id[:16]}"
if not safe_run_name or inputs.get("run_id") != expected_run_id:
    fail("score input run ID mismatch")

prediction_rows = []
for line_number, line in enumerate(
    complete_predictions_path.read_text(encoding="ascii").splitlines(), start=1
):
    if not line.strip():
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        fail(f"completed prediction line {line_number} is invalid: {error}")
    prediction_rows.append(row)
if [row.get("instance_id") for row in prediction_rows] != instance_ids:
    fail("completed prediction order/coverage changed")
if any(row.get("model_name_or_path") != inputs["expected_model_name_or_path"] for row in prediction_rows):
    fail("completed prediction model identity changed")
empty_ids = {
    row["instance_id"] for row in prediction_rows if row.get("model_patch") == ""
}
if empty_ids != set(inputs.get("empty_prediction_ids") or []):
    fail("completed prediction empty-patch coverage changed")

aggregate = load_json(aggregate_path, "official aggregate report")
if aggregate.get("schema_version") != 2:
    fail("unexpected official aggregate schema")
list_fields = (
    "completed_ids",
    "incomplete_ids",
    "empty_patch_ids",
    "submitted_ids",
    "resolved_ids",
    "unresolved_ids",
    "error_ids",
)
sets = {}
for field in list_fields:
    values = aggregate.get(field)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        fail(f"official aggregate field is invalid: {field}")
    if len(values) != len(set(values)):
        fail(f"official aggregate field contains duplicates: {field}")
    sets[field] = set(values)
    if not sets[field] <= expected_set:
        fail(f"official aggregate field contains unexpected IDs: {field}")
if aggregate.get("total_instances") != 10:
    fail("official aggregate total instance count mismatch")
if sets["submitted_ids"] != expected_set or aggregate.get("submitted_instances") != 10:
    fail("official aggregate submitted coverage mismatch")
if sets["incomplete_ids"]:
    fail("official aggregate reports incomplete IDs despite completed predictions")
if sets["empty_patch_ids"] != empty_ids:
    fail("official aggregate empty-patch coverage mismatch")
for count_field, id_field in (
    ("completed_instances", "completed_ids"),
    ("resolved_instances", "resolved_ids"),
    ("unresolved_instances", "unresolved_ids"),
    ("empty_patch_instances", "empty_patch_ids"),
    ("error_instances", "error_ids"),
):
    if aggregate.get(count_field) != len(sets[id_field]):
        fail(f"official aggregate count mismatch: {count_field}")
category_sets = {
    "resolved": sets["resolved_ids"],
    "unresolved": sets["unresolved_ids"],
    "error": sets["error_ids"],
    "empty": sets["empty_patch_ids"],
}
for instance_id in instance_ids:
    memberships = [name for name, values in category_sets.items() if instance_id in values]
    if len(memberships) != 1:
        fail(f"official outcome is not unique/complete for {instance_id}: {memberships}")
if set().union(*category_sets.values()) != expected_set:
    fail("official outcome categories do not cover the campaign")
if not (sets["resolved_ids"] | sets["unresolved_ids"]) <= sets["completed_ids"]:
    fail("resolved/unresolved instances are missing completed coverage")
if empty_ids & sets["completed_ids"]:
    fail("empty-patch instances must not be marked completed")

model_dir = inputs["expected_model_name_or_path"].replace("/", "__")
outcomes = []
for row in prediction_rows:
    instance_id = row["instance_id"]
    outcome = next(name for name, values in category_sets.items() if instance_id in values)
    patch_bytes = len(row["model_patch"].encode("utf-8"))
    patch_sha256 = hashlib.sha256(row["model_patch"].encode("utf-8")).hexdigest()
    instance_report = (
        harness_root
        / "logs"
        / "run_evaluation"
        / inputs["run_id"]
        / model_dir
        / instance_id
        / "report.json"
    )
    instance_record = None
    if instance_report.is_file() and not instance_report.is_symlink():
        instance_record = {
            "path": str(instance_report.relative_to(attempt_dir)),
            "sha256": sha256_file(instance_report),
        }
    if outcome in {"resolved", "unresolved"}:
        if instance_record is None:
            fail(f"official instance report is missing for {instance_id}")
        report = load_json(instance_report, f"official instance report for {instance_id}")
        try:
            resolved = report[instance_id]["resolved"]
        except (KeyError, TypeError):
            fail(f"official instance report coverage is invalid for {instance_id}")
        if not isinstance(resolved, bool) or resolved != (outcome == "resolved"):
            fail(f"official instance report outcome mismatch for {instance_id}")
    outcomes.append(
        {
            "instance_id": instance_id,
            "outcome": outcome,
            "patch_bytes": patch_bytes,
            "patch_sha256": patch_sha256,
            "official_instance_report": instance_record,
        }
    )

def relative_artifact(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        fail(f"required score artifact is missing: {path}")
    return {
        "path": str(path.resolve().relative_to(attempt_dir)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }

normalized = {
    "schema_version": 1,
    "kind": "swe_verified_behavioral_official_outcomes",
    "run_name": inputs["run_name"],
    "run_id": inputs["run_id"],
    "evidence_id": inputs["evidence_id"],
    "instance_ids": instance_ids,
    "counts": {name: len(values) for name, values in category_sets.items()},
    "outcomes": outcomes,
    "inputs": {
        "hashes": hashes,
        "score_input_manifest_sha256": sha256_file(input_manifest_path),
        "missing_prediction_ids": inputs["missing_prediction_ids"],
        "empty_prediction_ids": inputs["empty_prediction_ids"],
        "model_name_or_path": inputs["expected_model_name_or_path"],
    },
    "official_harness": {
        "swebench_version": inputs["swebench_version"],
        "max_workers": 1,
        "timeout_seconds": timeout,
        "exit_status": 0,
        "aggregate_report": relative_artifact(aggregate_path),
        "stdout": relative_artifact(attempt_dir / "harness.stdout.log"),
        "stderr": relative_artifact(attempt_dir / "harness.stderr.log"),
        "status": relative_artifact(attempt_dir / "harness.exit_status"),
    },
    "attempt_path": str(attempt_dir),
}
encoded = (
    json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
).encode("ascii")
attempt_output = attempt_dir / "official_outcomes.json"
attempt_output.write_bytes(encoded)
normalized_path.parent.mkdir(parents=True, exist_ok=True)
temporary_output = normalized_path.with_name(normalized_path.name + f".tmp.{os.getpid()}")
temporary_output.write_bytes(encoded)
os.replace(temporary_output, normalized_path)
print(f"validated {len(instance_ids)} official outcomes")
PY
validation_status=$?
set -e
printf '%s\n' "$validation_status" >"$VALIDATION_STATUS"
if ((validation_status != 0)); then
  die "official score validation failed; see $VALIDATION_STDERR"
fi

echo "official behavioral outcomes: $NORMALIZED_OUTCOMES"
