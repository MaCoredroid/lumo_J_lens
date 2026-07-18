#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VLLM_PY=${VLLM_PY:-$ROOT/.venv-vllm/bin/python}
OUTPUT_ROOT=${OUT_DIR:-$ROOT/.cache/swe_behavioral_pilot_n20}
MANIFEST_PY=${MANIFEST_PY:-python3}

PRIMARY_CAMPAIGN=${PRIMARY_CAMPAIGN:-$ROOT/configs/swe_behavioral_campaign.json}
PRIMARY_RUN_ROOT=${PRIMARY_RUN_ROOT:-$ROOT/runs/swe_behavioral_n10_20260718}
REPLICATION_CAMPAIGN=${REPLICATION_CAMPAIGN:-$ROOT/configs/swe_behavioral_replication_campaign.json}
REPLICATION_RUN_ROOT=${REPLICATION_RUN_ROOT:-$ROOT/runs/swe_behavioral_replication_n10_20260718}

PILOT_SCRIPT=$ROOT/scripts/run_swe_behavioral_pilot.sh
MATERIALIZER=$ROOT/scripts/materialize_swe_behavioral_probes.py
ANALYZER=$ROOT/scripts/analyze_swe_behavioral_probes.py
JLENS_RUNNER=${JLENS_RUNNER:-$ROOT/scripts/run_jlens_nvfp4.sh}
JLENS_PYTHON_RUNNER=$ROOT/scripts/run_jlens_nvfp4.py
MODEL_CHECKPOINT_VERIFIER=$ROOT/scripts/modelopt_checkpoint.py
ACTION_PROTOCOL=$ROOT/configs/swe_stage_action_probes.json
READOUT_PROTOCOL=$ROOT/configs/swe_behavioral_readout_protocol.json
CHAT_TEMPLATE=$ROOT/configs/qwen3-openai-codex.jinja
COHORT_MANIFEST=${COHORT_MANIFEST:-$ROOT/configs/swe_behavioral_n20_cohort.json}

MODEL_REPO=nvidia/Qwen3.6-27B-NVFP4
MODEL_REVISION=0893e1606ff3d5f97a441f405d5fc541a6bdf404
MODEL_CONFIG_SHA256=c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338
MODEL_INDEX_SHA256=7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2
MODEL_CHECKPOINT_VERIFIER_SHA256=f02b4cdf84d800b13165dfc559c244bdce8c693e939d64d05a5e26351ce57fb6
JLENS_RUNNER_SHA256=89763dc20b09394e52f2296654b58408de40c902cb5dddc0a109026964eb2331
JLENS_PYTHON_RUNNER_SHA256=18697bd8adc1159b7228390c526b9c418e87c25c64e7fa5d121d9f95906f7ee5

PUBLIC_LENS_REPO=neuronpedia/jacobian-lens
PUBLIC_LENS_REVISION=a4114d7752d11eb546e6cf372213d7e75526d3a1
PUBLIC_LENS_SHA256=1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1
NF4_LENS_PATH=$ROOT/.cache/Qwen3.6-27B-jlens-nf4-n10-fp32.pt
NF4_LENS_SHA256=54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f
NF4_PROVENANCE_PATH=$ROOT/.cache/Qwen3.6-27B-jlens-nf4-n10-fp32.provenance.json
NF4_PROVENANCE_SHA256=08dfe022cc74bab226308dcd77a3901ec18806f8832dc8dec46524374fa3b1e7
NATIVE_LENS_PATH=$ROOT/.cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt
NATIVE_LENS_SHA256=82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057
NATIVE_PROVENANCE_PATH=$ROOT/.cache/nvfp4_ste_fit/final-mean/metadata.json
NATIVE_PROVENANCE_SHA256=289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601
NATIVE_STATE_PATH=$ROOT/.cache/nvfp4_ste_fit/state.json
NATIVE_STATE_SHA256=f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6

LAYERS=24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47
REPORT_SCHEMA_VERSION=3
MAX_MODEL_LEN=65536
MAX_NUM_BATCHED_TOKENS=4096
MAMBA_BLOCK_SIZE=1024
KV_OFFLOADING_SIZE=8
GPU_MEMORY_UTILIZATION=0.78
BOOTSTRAP_SAMPLES=5000
TORCH_VERSION=2.11.0+cu130
VLLM_VERSION=0.23.0
TRANSFORMERS_VERSION=5.12.1
HUGGINGFACE_HUB_VERSION=1.21.0
TRITON_VERSION=3.6.0
PYTHON_VERSION=3.12.13
SERVER_UNIT=${SERVER_UNIT:-lumo_j_lens_qwen27b}
SERVER_ENDPOINT=${SERVER_ENDPOINT:-http://127.0.0.1:9952/v1/models}
SWEBENCH_VERSION=4.1.0

usage() {
  cat <<'EOF'
Usage: scripts/run_swe_behavioral_pilot.sh [--prepare-only | --reuse-reports]

The default experiment is the predeclared development plus replication N=20
cohort. Override PRIMARY_* or REPLICATION_* only to replay another explicitly
frozen pair of behavioral campaigns.

Environment:
  PRIMARY_CAMPAIGN       First frozen campaign config
  PRIMARY_RUN_ROOT       First completed and officially scored campaign root
  REPLICATION_CAMPAIGN   Second frozen campaign config
  REPLICATION_RUN_ROOT   Second completed and officially scored campaign root
  COHORT_MANIFEST        Manifest binding the ordered campaign pair
  OUT_DIR                Root containing immutable attempts and the latest link
  VLLM_PY                Pinned vLLM Python interpreter
  SERVER_ENDPOINT        Generation endpoint which must be offline for capture
EOF
}

die() {
  FAILURE_REASON=$*
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  local path=$1
  local label=$2
  [[ -f "$path" ]] || die "$label is missing: $path"
}

require_nonempty_file() {
  local path=$1
  local label=$2
  [[ -s "$path" ]] || die "$label is missing or empty: $path"
}

require_sha256() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    die "$label SHA-256 mismatch: expected $expected, got $actual"
  }
}

validate_initialization_inputs() {
  [[ -x "$VLLM_PY" ]] || {
    echo "vLLM Python is missing or not executable: $VLLM_PY"
    return 1
  }
  local path label
  for pair in \
    "$MATERIALIZER:behavioral materializer" \
    "$ANALYZER:behavioral analyzer" \
    "$JLENS_RUNNER:J-lens shell runner" \
    "$JLENS_PYTHON_RUNNER:J-lens Python runner" \
    "$MODEL_CHECKPOINT_VERIFIER:model checkpoint verifier" \
    "$ACTION_PROTOCOL:action protocol" \
    "$READOUT_PROTOCOL:readout protocol" \
    "$CHAT_TEMPLATE:chat template" \
    "$COHORT_MANIFEST:N=20 cohort manifest" \
    "$PRIMARY_CAMPAIGN:development campaign" \
    "$REPLICATION_CAMPAIGN:replication campaign"; do
    path=${pair%%:*}
    label=${pair#*:}
    [[ -f "$path" ]] || {
      echo "$label is missing: $path"
      return 1
    }
  done
  [[ -x "$JLENS_RUNNER" ]] || {
    echo "J-lens shell runner is not executable: $JLENS_RUNNER"
    return 1
  }
  local actual
  for pair in \
    "$JLENS_RUNNER:$JLENS_RUNNER_SHA256:J-lens shell runner" \
    "$JLENS_PYTHON_RUNNER:$JLENS_PYTHON_RUNNER_SHA256:J-lens Python runner" \
    "$MODEL_CHECKPOINT_VERIFIER:$MODEL_CHECKPOINT_VERIFIER_SHA256:model checkpoint verifier"; do
    path=${pair%%:*}
    local remainder=${pair#*:}
    local expected=${remainder%%:*}
    label=${remainder#*:}
    actual=$(sha256sum "$path" | awk '{print $1}') || return 1
    [[ "$actual" == "$expected" ]] || {
      echo "$label SHA-256 mismatch: expected $expected, got $actual"
      return 1
    }
  done
}

validate_lens_artifact_inputs() {
  local path label actual remainder expected
  for pair in \
    "$NF4_LENS_PATH:NF4 lens" \
    "$NF4_PROVENANCE_PATH:NF4 provenance" \
    "$NATIVE_LENS_PATH:native NVFP4 lens" \
    "$NATIVE_PROVENANCE_PATH:native NVFP4 provenance" \
    "$NATIVE_STATE_PATH:native NVFP4 state"; do
    path=${pair%%:*}
    label=${pair#*:}
    [[ -f "$path" ]] || {
      echo "$label is missing: $path"
      return 1
    }
  done
  for pair in \
    "$NF4_LENS_PATH:$NF4_LENS_SHA256:NF4 lens" \
    "$NF4_PROVENANCE_PATH:$NF4_PROVENANCE_SHA256:NF4 provenance" \
    "$NATIVE_LENS_PATH:$NATIVE_LENS_SHA256:native NVFP4 lens" \
    "$NATIVE_PROVENANCE_PATH:$NATIVE_PROVENANCE_SHA256:native NVFP4 provenance" \
    "$NATIVE_STATE_PATH:$NATIVE_STATE_SHA256:native NVFP4 state"; do
    path=${pair%%:*}
    remainder=${pair#*:}
    expected=${remainder%%:*}
    label=${remainder#*:}
    actual=$(sha256sum "$path" | awk '{print $1}') || return 1
    [[ "$actual" == "$expected" ]] || {
      echo "$label SHA-256 mismatch: expected $expected, got $actual"
      return 1
    }
  done
}

write_sha256_sidecar() {
  local path=$1
  local sidecar=$2
  local digest temporary
  digest=$(sha256sum "$path" | awk '{print $1}')
  temporary=$sidecar.tmp.$$
  printf '%s  %s\n' "$digest" "$(basename "$path")" >"$temporary"
  mv -f "$temporary" "$sidecar"
}

verify_sha256_sidecar() {
  local path=$1
  local sidecar=$2
  local label=$3
  [[ -s "$path" ]] || {
    echo "$label is missing or empty: $path" >&2
    return 1
  }
  [[ -s "$sidecar" ]] || {
    echo "$label checksum is missing or empty: $sidecar" >&2
    return 1
  }
  local expected target extra actual
  read -r expected target extra <"$sidecar" || {
    echo "cannot read $label checksum" >&2
    return 1
  }
  [[ -z "${extra:-}" && "$target" == "$(basename "$path")" ]] || {
    echo "$label checksum target is invalid" >&2
    return 1
  }
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || {
    echo "$label checksum is malformed" >&2
    return 1
  }
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "$label SHA-256 mismatch: expected $expected, got $actual" >&2
    return 1
  }
}

show_failure_log() {
  local log=$1
  if [[ -f "$log" ]]; then
    echo "--- tail of $log ---" >&2
    tail -80 "$log" >&2 || true
  fi
}

run_logged() {
  local label=$1
  local log=$2
  shift 2
  local outputs=()
  while (($#)) && [[ "$1" != -- ]]; do
    outputs+=("$1")
    shift
  done
  [[ "${1:-}" == -- ]] || die "$label has no command delimiter"
  shift
  CURRENT_PHASE=$label
  local status_file=$OUT_DIR/$label.exit_status
  rm -f "$log" "$status_file" "${outputs[@]}"
  echo "[$label] log: $log"
  set +e
  "$@" >"$log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" >"$status_file"
  if ((status != 0)); then
    show_failure_log "$log"
    die "$label failed with exit status $status"
  fi
  local output
  for output in "${outputs[@]}"; do
    if [[ ! -s "$output" ]]; then
      printf 'missing-output-after-process-status-%s\n' "$status" >"$status_file"
      die "$label did not emit a fresh nonempty output: $output"
    fi
  done
}

validate_campaign_evidence() {
  "$VLLM_PY" - behavioral-pilot-campaign-v1 \
    "$ROOT" \
    "$PRIMARY_CAMPAIGN" "$PRIMARY_RUN_ROOT" \
    "$REPLICATION_CAMPAIGN" "$REPLICATION_RUN_ROOT" \
    "$MODEL_REPO" "$MODEL_REVISION" "$MAX_MODEL_LEN" "$SWEBENCH_VERSION" \
    "$CAMPAIGN_EVIDENCE" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

if sys.argv[1] != "behavioral-pilot-campaign-v1":
    raise SystemExit("campaign validator marker mismatch")
(
    root_arg,
    primary_campaign_arg,
    primary_run_arg,
    replication_campaign_arg,
    replication_run_arg,
    model_repo,
    model_revision,
    max_model_len_arg,
    swebench_version,
    evidence_output_arg,
) = sys.argv[2:]
root = Path(root_arg).resolve(strict=True)
max_model_len = int(max_model_len_arg)
expected_model_id = "qwen3.6-27b-nvfp4-mtp::qwen-code-0.19.4"
max_session_turns = 50
fatal_error = {
    "error": {
        "type": "FatalTurnLimitedError",
        "message": (
            "Reached max session turns for this session. Increase the number of "
            "turns by specifying maxSessionTurns in settings.json."
        ),
        "code": 53,
    }
}
metric_number = r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"
spec_decoding_pattern = re.compile(
    rf"\(APIServer pid=[1-9][0-9]*\) INFO "
    rf"[0-9]{{2}}-[0-9]{{2}} [0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}} "
    rf"\[metrics\.py:[1-9][0-9]*\] SpecDecoding metrics: "
    rf"Mean acceptance length: (?P<mean>{metric_number}), "
    rf"Accepted throughput: (?P<accepted_throughput>{metric_number}) tokens/s, "
    rf"Drafted throughput: (?P<drafted_throughput>{metric_number}) tokens/s, "
    rf"Accepted: (?P<accepted>[0-9]+) tokens, "
    rf"Drafted: (?P<drafted>[0-9]+) tokens, "
    rf"Per-position acceptance rate: (?P<position>{metric_number}), "
    rf"Avg Draft acceptance rate: (?P<average>{metric_number})%"
)


def fail(message):
    raise SystemExit(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def regular_bytes(path, label, *, required=True):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        fail(f"{label} is not a regular non-symlink file")
    if not path.is_file():
        if required:
            fail(f"{label} is missing: {path}")
        return None
    return path.read_bytes()


all_ids = []
cohort_audits = []
for cohort, campaign_arg, run_arg in (
    ("development", primary_campaign_arg, primary_run_arg),
    ("replication", replication_campaign_arg, replication_run_arg),
):
    campaign_path = Path(campaign_arg).resolve(strict=True)
    run_root = Path(run_arg).resolve(strict=True)
    campaign_bytes = campaign_path.read_bytes()
    campaign_sha = hashlib.sha256(campaign_bytes).hexdigest()
    campaign = json.loads(campaign_bytes)
    generation = campaign.get("generation", {})
    instance_ids = campaign.get("instance_ids")
    if (
        campaign.get("schema_version") != 1
        or campaign.get("kind") != "swe_verified_behavioral_trajectory_campaign"
        or generation.get("model_repo_id") != model_repo
        or generation.get("model_revision") != model_revision
        or generation.get("max_model_len") != max_model_len
        or generation.get("max_session_turns") != max_session_turns
        or not isinstance(instance_ids, list)
        or len(instance_ids) != 10
        or len(instance_ids) != len(set(instance_ids))
    ):
        fail(f"{cohort} campaign contract changed")
    if set(instance_ids) & set(all_ids):
        fail("development and replication task IDs overlap")
    all_ids.extend(instance_ids)

    server_log_path = run_root / "server.log"
    server_log_bytes = regular_bytes(server_log_path, f"{cohort} server log")
    try:
        server_log_text = server_log_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        fail(f"{cohort} server log is not UTF-8: {error}")
    metric_lines = []
    accepted_tokens = 0
    drafted_tokens = 0
    for line_number, line in enumerate(server_log_text.splitlines(), 1):
        if "SpecDecoding metrics:" not in line:
            continue
        match = spec_decoding_pattern.fullmatch(line)
        if match is None:
            fail(
                f"{cohort} server log SpecDecoding metrics line "
                f"{line_number} changed format"
            )
        metric_lines.append(line)
        accepted_tokens += int(match.group("accepted"))
        drafted_tokens += int(match.group("drafted"))
    if not metric_lines:
        fail(f"{cohort} server log has no exact SpecDecoding metrics evidence")
    if (
        accepted_tokens <= 0
        or drafted_tokens <= 0
        or accepted_tokens > drafted_tokens
    ):
        fail(
            f"{cohort} speculative-decoding token totals are invalid: "
            f"accepted={accepted_tokens}, drafted={drafted_tokens}"
        )
    metric_lines_bytes = b"".join(
        line.encode("utf-8") + b"\n" for line in metric_lines
    )
    mtp_evidence = {
        "server_log": {
            "path": server_log_path.relative_to(run_root).as_posix(),
            "bytes": len(server_log_bytes),
            "sha256": hashlib.sha256(server_log_bytes).hexdigest(),
        },
        "metric_format": "vllm_spec_decoding_metrics_v1",
        "metric_line_count": len(metric_lines),
        "metric_lines_sha256": hashlib.sha256(metric_lines_bytes).hexdigest(),
        "accepted_tokens": accepted_tokens,
        "drafted_tokens": drafted_tokens,
        "weighted_acceptance_rate": accepted_tokens / drafted_tokens,
        "weighted_acceptance_definition": "sum(accepted_tokens)/sum(drafted_tokens)",
    }

    source_seal = run_root / "generation_sources.sha256"
    if not source_seal.is_file():
        fail(f"{cohort} generation source seal is missing")
    sealed = {}
    for line_number, line in enumerate(source_seal.read_text(encoding="ascii").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\0]+)", line)
        if match is None:
            fail(f"{cohort} source seal line {line_number} is malformed")
        expected, logical = match.groups()
        path = (root / logical).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            fail(f"{cohort} source seal path escapes the repository: {logical}")
        actual = digest(path)
        if actual != expected:
            fail(f"{cohort} sealed source changed: {logical}")
        sealed[logical] = expected
    try:
        campaign_logical = campaign_path.relative_to(root).as_posix()
    except ValueError:
        fail(f"{cohort} campaign config is outside the repository")
    if sealed.get(campaign_logical) != campaign_sha:
        fail(f"{cohort} campaign config is not bound by its generation source seal")

    image_path = run_root / "image_manifest.json"
    dataset_path = run_root / "dataset.json"
    image_bytes = regular_bytes(image_path, f"{cohort} image manifest")
    dataset_bytes = regular_bytes(dataset_path, f"{cohort} dataset")
    image_manifest = json.loads(image_bytes)
    dataset = json.loads(dataset_bytes)
    image_rows = image_manifest.get("images", [])
    if (
        image_manifest.get("schema_version") != 1
        or image_manifest.get("kind")
        != "swe_verified_behavioral_campaign_image_manifest"
        or image_manifest.get("campaign_config_sha256") != campaign_sha
        or not isinstance(image_rows, list)
        or [row.get("instance_id") for row in image_rows] != instance_ids
        or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("image_id"))) is None
            for row in image_rows
        )
        or [row.get("instance_id") for row in dataset] != instance_ids
    ):
        fail(f"{cohort} image/dataset evidence is not bound to the campaign")

    official_path = run_root / "official_score/official_outcomes.json"
    official = json.loads(regular_bytes(official_path, f"{cohort} official outcomes"))
    outcomes = official.get("outcomes")
    if (
        official.get("schema_version") != 1
        or official.get("kind") != "swe_verified_behavioral_official_outcomes"
        or official.get("instance_ids") != instance_ids
        or not isinstance(outcomes, list)
        or [row.get("instance_id") for row in outcomes] != instance_ids
        or any(row.get("outcome") not in {"resolved", "unresolved", "error", "empty"} for row in outcomes)
    ):
        fail(f"{cohort} official outcomes do not exactly cover the campaign")

    source_path = run_root / "generation/verified/predictions.jsonl"
    source_bytes = regular_bytes(
        source_path, f"{cohort} source predictions", required=False
    )
    source_rows = {}
    if source_bytes is not None:
        for line_number, raw_line in enumerate(source_bytes.splitlines(), 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                fail(f"{cohort} source prediction line {line_number} is invalid: {error}")
            instance_id = row.get("instance_id")
            if instance_id not in instance_ids or instance_id in source_rows:
                fail(f"{cohort} source prediction coverage is invalid: {instance_id!r}")
            if (
                row.get("model_name_or_path") != expected_model_id
                or not isinstance(row.get("model_patch"), str)
            ):
                fail(f"{cohort} source prediction contract changed: {instance_id}")
            source_rows[instance_id] = row

    missing_ids = []
    empty_ids = []
    complete_rows = []
    complete_patches = {}
    fatal_records = []
    task_request_counts = []
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
        complete_patches[instance_id] = patch
        complete_rows.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": expected_model_id,
                "model_patch": patch,
            }
        )
    complete_bytes = b"".join(canonical_bytes(row) + b"\n" for row in complete_rows)
    expected_hashes = {
        "campaign_config_sha256": campaign_sha,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "image_manifest_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "source_predictions_sha256": (
            hashlib.sha256(source_bytes).hexdigest() if source_bytes is not None else None
        ),
        "complete_predictions_sha256": hashlib.sha256(complete_bytes).hexdigest(),
    }
    official_inputs = official.get("inputs")
    official_harness = official.get("official_harness")
    if (
        not isinstance(official_inputs, dict)
        or official_inputs.get("hashes") != expected_hashes
        or official_inputs.get("missing_prediction_ids") != missing_ids
        or official_inputs.get("empty_prediction_ids") != empty_ids
        or official_inputs.get("model_name_or_path") != expected_model_id
        or not isinstance(official_harness, dict)
        or official_harness.get("swebench_version") != swebench_version
        or official_harness.get("max_workers") != 1
        or official_harness.get("exit_status") != 0
    ):
        fail(f"{cohort} official score input/hash contract changed")
    evidence_payload = {
        "hashes": expected_hashes,
        "instance_ids": instance_ids,
        "model_name_or_path": expected_model_id,
        "swebench_version": swebench_version,
    }
    evidence_id = hashlib.sha256(canonical_bytes(evidence_payload)).hexdigest()
    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(official.get("run_name", ""))).strip("._-")[:64]
    expected_run_id = f"lumo_j_lens_{safe_run_name}_behavioral_{evidence_id[:16]}"
    if (
        not safe_run_name
        or official.get("evidence_id") != evidence_id
        or official.get("run_id") != expected_run_id
    ):
        fail(f"{cohort} official evidence/run identity changed")

    counts = {name: 0 for name in ("resolved", "unresolved", "error", "empty")}
    for row in outcomes:
        instance_id = row["instance_id"]
        outcome = row["outcome"]
        counts[outcome] += 1
        patch_bytes = complete_patches[instance_id].encode("utf-8")
        if (
            row.get("patch_bytes") != len(patch_bytes)
            or row.get("patch_sha256") != hashlib.sha256(patch_bytes).hexdigest()
        ):
            fail(f"{cohort} official patch binding changed: {instance_id}")
    if official.get("counts") != counts:
        fail(f"{cohort} official outcome counts changed")

    for instance_id in instance_ids:
        metadata_path = (
            run_root
            / "generation/verified/per_task"
            / instance_id
            / "runner_metadata.json"
        )
        if not metadata_path.is_file():
            fail(f"{cohort} task is incomplete: {instance_id}")
        metadata = json.loads(metadata_path.read_bytes())
        qwen = metadata.get("qwen")
        model_id = metadata.get("eval_report", {}).get("model_id")
        common_ok = (
            metadata.get("instance_id") != instance_id
            or not isinstance(qwen, dict)
            or not isinstance(metadata.get("ended_at"), str)
            or model_id != expected_model_id
        )
        if common_ok:
            fail(f"{cohort} runner metadata is incomplete or not MTP generation: {instance_id}")
        parsed_turns = qwen.get("num_turns")
        parsed_ok = (
            qwen.get("parsed") is True
            and qwen.get("exit_code") == 0
            and qwen.get("timed_out") is False
            and isinstance(parsed_turns, int)
            and not isinstance(parsed_turns, bool)
            and 1 <= parsed_turns <= max_session_turns
        )
        fatal_ok = (
            qwen.get("parsed") is False
            and qwen.get("exit_code") == 53
            and qwen.get("timed_out") is False
            and qwen.get("cli_exit_is_verdict") is False
            and qwen.get("num_turns") is None
            and qwen.get("subtype") is None
            and qwen.get("result_tail") == ""
            and qwen.get("usage") is None
            and qwen.get("tool_calls") is None
            and qwen.get("tool_by_name") is None
        )
        if parsed_ok:
            task_request_counts.append(parsed_turns)
            continue
        if not fatal_ok:
            fail(f"{cohort} runner metadata has an unsupported unparsed termination: {instance_id}")

        task_root = metadata_path.parent
        stderr_path = task_root / "qwen_stderr.log"
        trace_path = task_root / "qwen_trace.json"
        patch_path = task_root / "patch.diff"
        usage_path = task_root / "qwen_home/.qwen/usage_record.jsonl"
        stderr_bytes = regular_bytes(stderr_path, f"{cohort}/{instance_id} fatal stderr")
        trace_bytes = regular_bytes(trace_path, f"{cohort}/{instance_id} fatal trace")
        patch_bytes = regular_bytes(patch_path, f"{cohort}/{instance_id} fatal patch")
        usage_bytes = regular_bytes(usage_path, f"{cohort}/{instance_id} fatal usage")
        try:
            stderr_value = json.loads(stderr_bytes)
        except json.JSONDecodeError as error:
            fail(f"{cohort}/{instance_id} fatal stderr is invalid JSON: {error}")
        usage_lines = [line for line in usage_bytes.splitlines() if line.strip()]
        if len(usage_lines) != 1:
            fail(f"{cohort}/{instance_id} fatal usage must contain exactly one record")
        try:
            usage_value = json.loads(usage_lines[0])
        except json.JSONDecodeError as error:
            fail(f"{cohort}/{instance_id} fatal usage is invalid JSON: {error}")
        models = usage_value.get("models") if isinstance(usage_value, dict) else None
        requests = None
        if isinstance(models, dict) and models:
            request_values = [
                value.get("requests") if isinstance(value, dict) else None
                for value in models.values()
            ]
            if all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in request_values
            ):
                requests = sum(request_values)
        if (
            stderr_value != fatal_error
            or trace_bytes != b""
            or requests != max_session_turns
            or complete_patches[instance_id].encode("utf-8") != patch_bytes
        ):
            fail(f"{cohort}/{instance_id} fatal turn-limit evidence is incomplete or changed")
        fatal_records.append(
            {
                "instance_id": instance_id,
                "termination": "exact_fatal_turn_limit_53",
                "max_session_turns": max_session_turns,
                "proxy_request_count": requests,
                "stderr": {
                    "path": stderr_path.relative_to(run_root).as_posix(),
                    "sha256": hashlib.sha256(stderr_bytes).hexdigest(),
                },
                "empty_trace": {
                    "path": trace_path.relative_to(run_root).as_posix(),
                    "bytes": 0,
                    "sha256": hashlib.sha256(trace_bytes).hexdigest(),
                },
                "patch": {
                    "path": patch_path.relative_to(run_root).as_posix(),
                    "bytes": len(patch_bytes),
                    "sha256": hashlib.sha256(patch_bytes).hexdigest(),
                },
                "usage": {
                    "path": usage_path.relative_to(run_root).as_posix(),
                    "sha256": hashlib.sha256(usage_bytes).hexdigest(),
                },
            }
        )
        task_request_counts.append(requests)
    expected_proxy_requests = sum(task_request_counts)
    proxy_root = run_root / "proxy_dumps"
    chat_paths = sorted(proxy_root.glob("chat_*.json"))
    expected_names = [
        f"chat_{index:04d}.json" for index in range(1, expected_proxy_requests + 1)
    ]
    if [path.name for path in chat_paths] != expected_names:
        fail(
            f"{cohort} proxy capture coverage differs from task request counts: "
            f"expected {expected_proxy_requests}, got {len(chat_paths)}"
        )
    proxy_digest = hashlib.sha256()
    for path in chat_paths:
        payload = regular_bytes(path, f"{cohort} proxy capture {path.name}")
        proxy_digest.update(path.name.encode("ascii") + b"\0")
        proxy_digest.update(len(payload).to_bytes(8, "big"))
        proxy_digest.update(payload)
    fatal_by_id = {record["instance_id"]: record for record in fatal_records}
    global_start = 1
    for instance_id, request_count in zip(instance_ids, task_request_counts, strict=True):
        if instance_id in fatal_by_id:
            fatal_by_id[instance_id]["proxy_request_range"] = {
                "start_inclusive": global_start,
                "end_inclusive": global_start + request_count - 1,
                "count": request_count,
            }
        global_start += request_count
    cohort_audits.append(
        {
            "cohort": cohort,
            "campaign_sha256": campaign_sha,
            "instance_ids": instance_ids,
            "proxy_capture_binding": {
                "algorithm": "filename_length_payload_sha256_v1",
                "request_count": expected_proxy_requests,
                "sha256": proxy_digest.hexdigest(),
            },
            "mtp_speculative_decoding": mtp_evidence,
            "fatal_turn_limit_tasks": fatal_records,
        }
    )

if len(all_ids) != 20:
    fail("combined behavioral cohort must contain exactly 20 unique tasks")
evidence_output = Path(evidence_output_arg)
evidence_output.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "kind": "swe_behavioral_campaign_replay_evidence",
            "task_count": len(all_ids),
            "cohorts": cohort_audits,
        },
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    )
    + "\n",
    encoding="ascii",
)
PY
}

validate_materialized_bundle() {
  local prompt_sha
  prompt_sha=$(sha256sum "$PROMPTS" | awk '{print $1}')
  "$VLLM_PY" - \
    "$ROOT" "$PROMPTS" "$PROMPTS_SUMMARY" "$READOUT_PROTOCOL" \
    "$PRIMARY_CAMPAIGN" "$REPLICATION_CAMPAIGN" "$COHORT_MANIFEST" \
    "$prompt_sha" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    root_arg,
    prompts_arg,
    summary_arg,
    protocol_arg,
    primary_campaign_arg,
    replication_campaign_arg,
    cohort_manifest_arg,
    expected_prompt_sha,
) = sys.argv[1:]
root = Path(root_arg).resolve(strict=True)
sys.path.insert(0, str(root / "scripts"))
import analyze_swe_behavioral_probes as analyzer


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


prompts_path = Path(prompts_arg)
summary_path = Path(summary_arg)
protocol_path = Path(protocol_arg)
prompts = json.loads(prompts_path.read_bytes())
summary = json.loads(summary_path.read_bytes())
protocol_bytes = protocol_path.read_bytes()
protocol = analyzer.validate_protocol(
    json.loads(protocol_bytes), protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest()
)
contract = analyzer.validate_prompt_bundle(prompts, protocol=protocol)
campaign_hashes = [
    sha256(Path(primary_campaign_arg)),
    sha256(Path(replication_campaign_arg)),
]
if sha256(prompts_path) != expected_prompt_sha:
    raise SystemExit("prompt bundle changed during validation")
if summary.get("prompt_bundle_sha256") != expected_prompt_sha:
    raise SystemExit("materializer summary does not bind the exact prompt bytes")
if contract.get("prompt_bundle_sha256") != expected_prompt_sha:
    raise SystemExit("analyzer prompt contract hash differs from the file hash")
if contract.get("task_count") != 20:
    raise SystemExit("combined behavioral prompt bundle does not contain 20 tasks")
if summary.get("task_count") != 20 or summary.get("prompt_count") != len(prompts):
    raise SystemExit("materializer summary count mismatch")
summary_campaigns = summary.get("source_campaign_sha256s")
if summary_campaigns != campaign_hashes:
    raise SystemExit("materializer summary does not preserve both source campaign hashes")
if summary.get("cohort_manifest_sha256") != sha256(Path(cohort_manifest_arg)):
    raise SystemExit("materializer summary does not bind the frozen N=20 cohort manifest")
PY
}

validate_prompt_lengths() {
  "$VLLM_PY" - behavioral-pilot-prompt-length-v1 \
    "$PROMPTS" "$MAX_MODEL_LEN" <<'PY'
import json
from pathlib import Path
import sys

if sys.argv[1] != "behavioral-pilot-prompt-length-v1":
    raise SystemExit("prompt-length validator marker mismatch")
prompts_path = Path(sys.argv[2])
max_model_len = int(sys.argv[3])
max_prompt_tokens = max_model_len - 1
prompts = json.loads(prompts_path.read_bytes())
if not isinstance(prompts, list):
    raise SystemExit("materialized prompts must be a JSON list")

oversized = []
for index, prompt in enumerate(prompts):
    if not isinstance(prompt, dict):
        raise SystemExit(f"materialized prompt {index} is not an object")
    prompt_id = prompt.get("id")
    if not isinstance(prompt_id, str) or not prompt_id:
        prompt_id = f"index-{index}"
    token_ids = prompt.get("token_ids")
    if not isinstance(token_ids, list):
        raise SystemExit(f"materialized prompt {prompt_id} has no token_ids list")
    token_count = len(token_ids)
    if token_count > max_prompt_tokens:
        oversized.append((prompt_id, token_count))

if oversized:
    details = "; ".join(
        f"{prompt_id}: {token_count} tokens" for prompt_id, token_count in oversized
    )
    raise SystemExit(
        "materialized prompts exceed the replay context generation-slot contract "
        f"(max_model_len={max_model_len}, max_prompt_tokens={max_prompt_tokens}): "
        f"{details}"
    )
PY
}

validate_json_report() {
  local report=$1
  local lens_label=$2
  local process_status=${3:-reused}
  local prompt_sha
  prompt_sha=$(sha256sum "$PROMPTS" | awk '{print $1}')
  "$VLLM_PY" - \
    "$ROOT" "$report" "$lens_label" "$process_status" "$PROMPTS" \
    "$READOUT_PROTOCOL" "$prompt_sha" "$LAYERS" \
    "$MODEL_REPO" "$MODEL_REVISION" "$MODEL_CONFIG_SHA256" "$MODEL_INDEX_SHA256" \
    "$MAX_MODEL_LEN" "$MAX_NUM_BATCHED_TOKENS" "$MAMBA_BLOCK_SIZE" \
    "$KV_OFFLOADING_SIZE" "$GPU_MEMORY_UTILIZATION" \
    "$TORCH_VERSION" "$VLLM_VERSION" "$TRANSFORMERS_VERSION" \
    "$HUGGINGFACE_HUB_VERSION" "$TRITON_VERSION" "$PYTHON_VERSION" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    root_arg,
    report_arg,
    lens_label,
    process_status,
    prompts_arg,
    protocol_arg,
    expected_prompt_sha,
    layers_arg,
    model_repo,
    model_revision,
    model_config_sha,
    model_index_sha,
    max_model_len_arg,
    max_num_batched_tokens_arg,
    mamba_block_size_arg,
    kv_offloading_size_arg,
    gpu_memory_utilization_arg,
    torch_version,
    vllm_version,
    transformers_version,
    hub_version,
    triton_version,
    python_version,
) = sys.argv[1:]
root = Path(root_arg).resolve(strict=True)
sys.path.insert(0, str(root / "scripts"))
import analyze_swe_behavioral_probes as analyzer
from modelopt_checkpoint import PINNED_METADATA_SHA256, PINNED_SHARDS


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


report_path = Path(report_arg)
prompts_path = Path(prompts_arg)
protocol_path = Path(protocol_arg)
if digest(prompts_path) != expected_prompt_sha:
    raise SystemExit("prompt bytes changed before report validation")
prompts = json.loads(prompts_path.read_bytes())
report = json.loads(report_path.read_bytes())
protocol_bytes = protocol_path.read_bytes()
protocol = analyzer.validate_protocol(
    json.loads(protocol_bytes), protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest()
)
prompt_contract = analyzer.validate_prompt_bundle(prompts, protocol=protocol)
validated = analyzer.validate_report(
    report,
    label=lens_label,
    prompt_contract=prompt_contract,
    protocol=protocol,
)

if process_status in {"0", "1"}:
    expected_status = "passed" if process_status == "0" else "failed"
    if report.get("status") != expected_status:
        raise SystemExit("report status does not match the replay exit status")

expected_layers = [int(value) for value in layers_arg.split(",")]
for experiment in report.get("experiments", []):
    actual_layers = [record.get("layer") for record in experiment.get("layers", [])]
    if actual_layers != expected_layers:
        raise SystemExit("report contains layers outside or out of order from the fixed band")
    for layer in experiment.get("layers", []):
        positions = layer.get("positions", [])
        if len(positions) != 1:
            raise SystemExit("report layer does not contain exactly one requested position")
        for method in ("logit_lens", "jacobian_lens"):
            readout = positions[0].get(method, {})
            if not all(
                isinstance(readout.get(field), list) and len(readout[field]) == 10
                for field in ("tokens", "token_ids", "scores")
            ):
                raise SystemExit("report top-k readout differs from the pinned value 10")

expected_runtime = {
    "mtp_enabled": False,
    "enforce_eager": True,
    "language_model_only": True,
    "max_model_len": int(max_model_len_arg),
    "max_num_batched_tokens": int(max_num_batched_tokens_arg),
    "mamba_block_size": int(mamba_block_size_arg),
    "enable_prefix_caching": True,
    "kv_cache_dtype": "fp8_e4m3",
    "kv_offloading_size": float(kv_offloading_size_arg),
    "kv_offloading_backend": "native",
    "stream_final_only": True,
    "gpu_memory_utilization": float(gpu_memory_utilization_arg),
    "capture_adapter": "vLLM apply_model forward hooks",
    "transport_dtype": "torch.float32",
    "readout_dtype": "torch.bfloat16",
    "timing_scope": "artifact resolution and validation through readout",
}
if validated.get("runtime_identity") != expected_runtime:
    raise SystemExit("report runtime differs from the exact behavioral replay contract")

model = report.get("model", {})
if {
    "repo_id": model.get("repo_id"),
    "revision": model.get("revision"),
    "config_sha256": model.get("config_sha256"),
    "index_sha256": model.get("index_sha256"),
    "quant_method": model.get("quant_method"),
    "quant_algo": model.get("quant_algo"),
} != {
    "repo_id": model_repo,
    "revision": model_revision,
    "config_sha256": model_config_sha,
    "index_sha256": model_index_sha,
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
}:
    raise SystemExit("report model identity differs from the pinned NVFP4 checkpoint")
checkpoint = model.get("checkpoint_integrity")
expected_checkpoint = {
    "policy": "ModelOptCheckpoint(strict_pinned=True)",
    "validated_before_model_load": True,
    "validated_after_evaluation": True,
    "metadata_sha256": dict(PINNED_METADATA_SHA256),
    "shards": {
        filename: {"bytes": size, "sha256": sha256}
        for filename, (size, sha256) in PINNED_SHARDS.items()
    },
}
if checkpoint != expected_checkpoint:
    raise SystemExit("full checkpoint integrity record is incomplete or changed")

host = report.get("host", {})
if host.get("python") != python_version or host.get("packages") != {
    "huggingface-hub": hub_version,
    "torch": torch_version,
    "transformers": transformers_version,
    "triton": triton_version,
    "vllm": vllm_version,
}:
    raise SystemExit("report host package versions differ from the pinned environment")
gpu = host.get("gpu", {})
if gpu.get("name") != "NVIDIA GeForce RTX 5090" or gpu.get("compute_capability") != "12.0":
    raise SystemExit("report was not produced on the pinned RTX 5090 host")
PY
}

run_lens() {
  local label=$1
  local report=$2
  local log=$3
  shift 3
  CURRENT_PHASE=$label
  local status_file=$OUT_DIR/$label.exit_status
  local stdout_file=$OUT_DIR/.$label.stdout
  local checksum=$CHECKSUM_DIR/$label-report.sha256
  ensure_capture_resources_idle "$label"
  CURRENT_PHASE=$label
  rm -f "$report" "$log" "$status_file" "$stdout_file" "$checksum"
  echo "[$label] log: $log"
  set +e
  "$@" --output "$report" >"$stdout_file" 2>"$log"
  local status=$?
  set -e
  printf '%s\n' "$status" >"$status_file"

  if [[ ! -s "$report" ]]; then
    if [[ -s "$stdout_file" ]]; then
      echo "--- tail of stdout ---" >>"$log"
      tail -80 "$stdout_file" >>"$log" || true
    fi
    printf 'missing-report-after-process-status-%s\n' "$status" >"$status_file"
    show_failure_log "$log"
    die "$label exited $status without a fresh report"
  fi
  if ((status != 0 && status != 1)); then
    printf 'unsupported-process-status-%s\n' "$status" >"$status_file"
    show_failure_log "$log"
    die "$label emitted a report but exited with unsupported status $status"
  fi
  if ! validate_json_report "$report" "$label" "$status" >>"$log" 2>&1; then
    printf 'invalid-report-after-process-status-%s\n' "$status" >"$status_file"
    show_failure_log "$log"
    die "$label emitted a report that violates the pinned behavioral contract"
  fi
  rm -f "$stdout_file"
  write_sha256_sidecar "$report" "$checksum"
  verify_sha256_sidecar "$report" "$checksum" "$label report"
  echo "[$label] accepted exit status $status with report $report"
}

reuse_lens_report() {
  local label=$1
  local report=$2
  local checksum=$CHECKSUM_DIR/$label-report.sha256
  local log=$LOG_DIR/$label.log
  local status_file=$OUT_DIR/$label.exit_status
  CURRENT_PHASE=$label
  rm -f "$log" "$status_file"
  if ! verify_sha256_sidecar "$report" "$checksum" "reused $label report" >"$log" 2>&1; then
    printf 'invalid-reused-report-checksum\n' >"$status_file"
    show_failure_log "$log"
    die "reused $label report checksum is stale or invalid"
  fi
  if ! validate_json_report "$report" "$label" reused >"$log" 2>&1; then
    printf 'invalid-reused-report\n' >"$status_file"
    show_failure_log "$log"
    die "reused $label report is stale or violates the pinned behavioral contract"
  fi
  printf 'reused\n' >"$status_file"
  echo "[$label] reused exact prompt/runtime-bound report: $report"
}

ensure_capture_resources_idle() {
  local label=$1
  local phase=${label}_preflight
  local log=$LOG_DIR/$phase.log
  local status_file=$OUT_DIR/$phase.exit_status
  CURRENT_PHASE=$phase
  rm -f "$log" "$status_file"
  if ! {
    if command -v systemctl >/dev/null 2>&1 \
      && systemctl is-active --quiet "$SERVER_UNIT"; then
      echo "generation server unit is active: $SERVER_UNIT"
      false
    elif ! command -v curl >/dev/null 2>&1; then
      echo "curl is required to verify that the generation endpoint is offline"
      false
    elif curl --silent --output /dev/null --connect-timeout 1 --max-time 2 \
      "$SERVER_ENDPOINT"; then
      echo "generation endpoint is live: $SERVER_ENDPOINT"
      false
    elif ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "nvidia-smi is required for exclusive capture preflight"
      false
    else
      local processes total_memory
      if ! processes=$(nvidia-smi \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits); then
        echo "cannot query active GPU compute processes"
        false
      elif ! total_memory=$(nvidia-smi \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits); then
        echo "cannot query total GPU memory use"
        false
      elif ! "$MANIFEST_PY" - "$processes" "$total_memory" <<'PY'
import csv
import io
from pathlib import PurePosixPath
import re
import sys

process_rows, total_memory_rows = sys.argv[1:]
process_caps_mib = {
    "gnome-control-center": 96,
    "ptyxis": 64,
    "nautilus": 64,
}
aggregate_process_cap_mib = 192
total_gpu_memory_cap_mib = 640


def fail(message):
    raise SystemExit(message)


records = []
seen = set()
if process_rows.strip():
    for row_number, row in enumerate(csv.reader(io.StringIO(process_rows)), 1):
        if len(row) != 3:
            fail(f"GPU compute-process row {row_number} is malformed")
        pid_text, process_name, memory_text = (field.strip() for field in row)
        if re.fullmatch(r"[1-9][0-9]*", pid_text) is None:
            fail(f"GPU compute-process row {row_number} has an invalid PID")
        if not process_name or "\\" in process_name:
            fail(f"GPU compute-process row {row_number} has an invalid process name")
        basename = PurePosixPath(process_name).name
        if basename not in process_caps_mib:
            fail(
                "GPU compute process is not in the pinned idle desktop allowlist: "
                f"pid={pid_text}, process_name={process_name!r}, basename={basename!r}"
            )
        if basename in seen:
            fail(f"duplicate allowed idle desktop GPU process: {basename}")
        if re.fullmatch(r"[0-9]+", memory_text) is None:
            fail(f"GPU compute-process row {row_number} has invalid memory use")
        memory_mib = int(memory_text)
        cap_mib = process_caps_mib[basename]
        if memory_mib > cap_mib:
            fail(
                f"allowed idle desktop GPU process exceeds its memory cap: "
                f"{basename} uses {memory_mib} MiB, cap is {cap_mib} MiB"
            )
        seen.add(basename)
        records.append((int(pid_text), process_name, basename, memory_mib, cap_mib))

aggregate_memory_mib = sum(record[3] for record in records)
if aggregate_memory_mib > aggregate_process_cap_mib:
    fail(
        "allowed idle desktop GPU processes exceed the aggregate memory cap: "
        f"{aggregate_memory_mib} MiB > {aggregate_process_cap_mib} MiB"
    )

total_rows = [line.strip() for line in total_memory_rows.splitlines() if line.strip()]
if len(total_rows) != 1 or re.fullmatch(r"[0-9]+", total_rows[0]) is None:
    fail("total GPU memory query must return exactly one integer MiB value")
total_memory_mib = int(total_rows[0])
if total_memory_mib < aggregate_memory_mib:
    fail(
        "total GPU memory is smaller than compute-process memory: "
        f"{total_memory_mib} MiB < {aggregate_memory_mib} MiB"
    )
if total_memory_mib > total_gpu_memory_cap_mib:
    fail(
        "total GPU memory exceeds the pinned idle cap: "
        f"{total_memory_mib} MiB > {total_gpu_memory_cap_mib} MiB"
    )

print(
    "idle desktop GPU allowlist policy: "
    "gnome-control-center<=96 MiB (max 1), "
    "ptyxis<=64 MiB (max 1), nautilus<=64 MiB (max 1)"
)
for pid, process_name, basename, memory_mib, cap_mib in records:
    print(
        "allowed idle desktop GPU process: "
        f"pid={pid}, process_name={process_name!r}, basename={basename}, "
        f"memory={memory_mib} MiB, cap={cap_mib} MiB"
    )
print(
    "allowed idle desktop GPU process aggregate: "
    f"count={len(records)}, memory={aggregate_memory_mib} MiB, "
    f"cap={aggregate_process_cap_mib} MiB"
)
print(
    "observed total GPU memory: "
    f"memory={total_memory_mib} MiB, cap={total_gpu_memory_cap_mib} MiB"
)
PY
      then
        false
      fi
    fi
  } >"$log" 2>&1; then
    printf 'capture-resource-busy-or-unverifiable\n' >"$status_file"
    show_failure_log "$log"
    die "$label model load refused because capture resources are not exclusive"
  fi
  printf '0\n' >"$status_file"
}

stage_reused_reports_impl() {
  [[ -L "$LATEST_LINK" ]] || {
    echo "no atomically promoted complete attempt exists at $LATEST_LINK"
    return 1
  }
  local source_attempt
  source_attempt=$(readlink -f -- "$LATEST_LINK") || return 1
  [[ -n "$source_attempt" && -d "$source_attempt" \
    && "$source_attempt" == "$ATTEMPTS_DIR"/* \
    && "$source_attempt" != "$OUT_DIR" ]] || {
    echo "latest attempt link is unsafe or invalid: $LATEST_LINK"
    return 1
  }
  "$MANIFEST_PY" - "$source_attempt/run_manifest.json" <<'PY' || return 1
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
value = json.loads(path.read_bytes())
if value.get("status") != "complete":
    raise SystemExit("latest attempt is not an operationally complete replay")
PY
  local name
  for name in \
    public-report.json nf4-report.json native-report.json \
    checksums/public-report.sha256 checksums/nf4-report.sha256 checksums/native-report.sha256; do
    [[ -s "$source_attempt/$name" ]] || {
      echo "latest attempt lacks reusable artifact: $name"
      return 1
    }
    mkdir -p "$(dirname "$OUT_DIR/$name")" || return 1
    cp --reflink=auto --preserve=mode,timestamps -- \
      "$source_attempt/$name" "$OUT_DIR/$name" || return 1
  done
}

stage_reused_reports() {
  local status_file=$OUT_DIR/reuse_staging.exit_status
  local log=$LOG_DIR/reuse_staging.log
  CURRENT_PHASE=reuse_staging
  rm -f "$status_file" "$log"
  if ! stage_reused_reports_impl >"$log" 2>&1; then
    printf 'reuse-staging-failed\n' >"$status_file"
    show_failure_log "$log"
    die "could not stage reports from the latest complete attempt"
  fi
  printf '0\n' >"$status_file"
}

promote_attempt() {
  local link_name=$1
  CURRENT_PHASE=promotion
  local status_file=$OUT_DIR/promotion.exit_status
  local log=$LOG_DIR/promotion.log
  rm -f "$status_file" "$log"
  if ! "$MANIFEST_PY" - \
    "$OUTPUT_ROOT" "$OUT_DIR" "$link_name" >"$log" 2>&1 <<'PY'
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
attempt = Path(sys.argv[2]).resolve(strict=True)
link_name = sys.argv[3]
attempts = (root / "attempts").resolve(strict=True)
if not attempt.is_relative_to(attempts) or attempt.parent != attempts:
    raise SystemExit("attempt is outside the immutable attempts directory")
target = Path("attempts") / attempt.name
destination = root / link_name
if destination.exists() and not destination.is_symlink():
    raise SystemExit(f"promotion target is not a symlink: {destination}")
temporary = root / f".{link_name}.tmp.{os.getpid()}"
try:
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, destination)
finally:
    temporary.unlink(missing_ok=True)
PY
  then
    printf 'promotion-failed\n' >"$status_file"
    show_failure_log "$log"
    die "failed to atomically promote attempt through $link_name"
  fi
  printf '0\n' >"$status_file"
}

validate_analysis() {
  "$VLLM_PY" - \
    "$ANALYSIS" "$PROMPTS" "$PUBLIC_REPORT" "$NF4_REPORT" "$NATIVE_REPORT" \
    "$READOUT_PROTOCOL" "$BOOTSTRAP_SAMPLES" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

analysis_arg, prompts_arg, public_arg, nf4_arg, native_arg, protocol_arg, samples_arg = sys.argv[1:]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


paths = {
    "prompts": Path(prompts_arg),
    "public_report": Path(public_arg),
    "nf4_report": Path(nf4_arg),
    "native_report": Path(native_arg),
    "protocol": Path(protocol_arg),
}
analysis = json.loads(Path(analysis_arg).read_bytes())
if analysis.get("schema_version") != 1 or analysis.get("kind") != "swe_verified_behavioral_task_held_out_analysis":
    raise SystemExit("behavioral analysis kind/schema mismatch")
allowed_statuses = {
    "development_support_complete",
    "development_insufficient_split_or_class_support",
    "held_out_evaluation_complete",
    "insufficient_split_or_class_support",
}
if analysis.get("status") not in allowed_statuses:
    raise SystemExit("behavioral analysis scientific status is missing or unsupported")
if analysis.get("inputs") != {key: digest(path) for key, path in paths.items()}:
    raise SystemExit("analysis does not bind the exact prompt/report/protocol hashes")
if analysis.get("campaign", {}).get("task_count") != 20:
    raise SystemExit("analysis is not the combined N=20 cohort")
if analysis.get("protocol", {}).get("fixed_layers") != list(range(24, 48)):
    raise SystemExit("analysis changed the fixed layer band")
if analysis.get("protocol", {}).get("bootstrap", {}).get("samples") != int(samples_arg):
    raise SystemExit("analysis bootstrap count differs from the pinned contract")
decision = analysis.get("decision_audit", {})
if decision.get("status") != analysis.get("status"):
    raise SystemExit("analysis and decision-audit scientific statuses differ")
for field in (
    "no_missing_fold_or_label_was_imputed",
    "evaluation_repositories_never_enter_fit_or_calibration",
    "fit_and_calibration_repositories_are_disjoint",
    "bootstrap_resamples_repositories_then_tasks_never_rows",
    "ordinary_logit_is_verified_identical_across_all_three_reports",
    "official_outcome_is_observed_once_per_task",
):
    if decision.get(field) is not True:
        raise SystemExit(f"analysis decision audit failed: {field}")
PY
}

emit_run_manifest() {
  local status=$1
  local mode=$2
  local process_status=$3
  "$MANIFEST_PY" - behavioral-pilot-manifest-v1 \
    "$RUN_MANIFEST" "$status" "$mode" "$process_status" \
    "$CURRENT_PHASE" "$FAILURE_REASON" \
    "$ROOT" "$OUTPUT_ROOT" "$OUT_DIR" "$PRIMARY_RUN_ROOT" "$REPLICATION_RUN_ROOT" \
    "$PILOT_SCRIPT" "$MATERIALIZER" "$ANALYZER" \
    "$JLENS_RUNNER" "$JLENS_PYTHON_RUNNER" "$MODEL_CHECKPOINT_VERIFIER" \
    "$ACTION_PROTOCOL" "$READOUT_PROTOCOL" "$CHAT_TEMPLATE" "$COHORT_MANIFEST" \
    "$PRIMARY_CAMPAIGN" "$REPLICATION_CAMPAIGN" \
    "$NF4_LENS_PATH" "$NF4_PROVENANCE_PATH" \
    "$NATIVE_LENS_PATH" "$NATIVE_PROVENANCE_PATH" "$NATIVE_STATE_PATH" \
    "$CAMPAIGN_EVIDENCE" \
    "$PROMPTS" "$PROMPTS_SUMMARY" "$PROMPTS_CHECKSUM" \
    "$PUBLIC_REPORT" "$NF4_REPORT" "$NATIVE_REPORT" \
    "$PUBLIC_CHECKSUM" "$NF4_CHECKSUM" "$NATIVE_CHECKSUM" \
    "$ANALYSIS" "$ANALYSIS_CHECKSUM" \
    "$MODEL_REPO" "$MODEL_REVISION" "$MODEL_CONFIG_SHA256" "$MODEL_INDEX_SHA256" \
    "$PUBLIC_LENS_REPO" "$PUBLIC_LENS_REVISION" "$PUBLIC_LENS_SHA256" \
    "$NF4_LENS_SHA256" "$NF4_PROVENANCE_SHA256" \
    "$NATIVE_LENS_SHA256" "$NATIVE_PROVENANCE_SHA256" "$NATIVE_STATE_SHA256" \
    "$LAYERS" "$MAX_MODEL_LEN" "$MAX_NUM_BATCHED_TOKENS" \
    "$MAMBA_BLOCK_SIZE" "$KV_OFFLOADING_SIZE" "$GPU_MEMORY_UTILIZATION" \
    "$BOOTSTRAP_SAMPLES" "$ANALYSIS_SCIENTIFIC_STATUS" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

if sys.argv[1] != "behavioral-pilot-manifest-v1":
    raise SystemExit("manifest marker mismatch")
(
    manifest_arg,
    status,
    mode,
    process_status,
    phase,
    failure_reason,
    root_arg,
    output_root_arg,
    out_arg,
    primary_run_arg,
    replication_run_arg,
    pilot_arg,
    materializer_arg,
    analyzer_arg,
    runner_arg,
    python_runner_arg,
    checkpoint_verifier_arg,
    action_protocol_arg,
    readout_protocol_arg,
    template_arg,
    cohort_manifest_arg,
    primary_campaign_arg,
    replication_campaign_arg,
    nf4_lens_arg,
    nf4_provenance_arg,
    native_lens_arg,
    native_provenance_arg,
    native_state_arg,
    campaign_evidence_arg,
    prompts_arg,
    prompts_summary_arg,
    prompts_checksum_arg,
    public_arg,
    nf4_arg,
    native_arg,
    public_checksum_arg,
    nf4_checksum_arg,
    native_checksum_arg,
    analysis_arg,
    analysis_checksum_arg,
    model_repo,
    model_revision,
    model_config_sha,
    model_index_sha,
    public_lens_repo,
    public_lens_revision,
    public_lens_sha,
    nf4_lens_sha,
    nf4_provenance_sha,
    native_lens_sha,
    native_provenance_sha,
    native_state_sha,
    layers_arg,
    max_model_len_arg,
    max_num_batched_tokens_arg,
    mamba_block_size_arg,
    kv_offloading_size_arg,
    gpu_memory_utilization_arg,
    bootstrap_samples_arg,
    scientific_status,
) = sys.argv[2:]

root = Path(root_arg).resolve(strict=True)
output_root = Path(output_root_arg).resolve(strict=True)
out = Path(out_arg).resolve(strict=True)
primary_run = Path(primary_run_arg).resolve(strict=False)
replication_run = Path(replication_run_arg).resolve(strict=False)
manifest_path = Path(manifest_arg)
required_inputs = status != "failed"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def record(path_arg, *, required=False):
    path = Path(path_arg)
    try:
        usable = path.is_file() and path.stat().st_size > 0
    except OSError:
        usable = False
    if not usable:
        if required:
            raise SystemExit(f"manifest artifact is missing or empty: {path}")
        return None
    resolved = path.resolve(strict=True)
    if resolved.is_relative_to(out):
        path_base = "output_directory"
        logical_path = resolved.relative_to(out).as_posix()
    elif resolved.is_relative_to(primary_run):
        path_base = "primary_campaign_run_root"
        logical_path = resolved.relative_to(primary_run).as_posix()
    elif resolved.is_relative_to(replication_run):
        path_base = "replication_campaign_run_root"
        logical_path = resolved.relative_to(replication_run).as_posix()
    elif resolved.is_relative_to(root):
        path_base = "repository_root"
        logical_path = resolved.relative_to(root).as_posix()
    else:
        if required:
            raise SystemExit(f"manifest artifact is outside all declared roots: {path}")
        return None
    result = {
        "path": logical_path,
        "path_base": path_base,
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }
    if path.suffix == ".json":
        try:
            value = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError):
            value = None
        if isinstance(value, dict) and isinstance(value.get("status"), str):
            result["report_status"] = value["status"]
    return result


def cohort_record(name, campaign_arg, run):
    return {
        "name": name,
        "campaign": record(campaign_arg, required=required_inputs),
        "generation_sources": record(
            run / "generation_sources.sha256", required=required_inputs
        ),
        "dataset": record(run / "dataset.json", required=required_inputs),
        "image_manifest": record(run / "image_manifest.json", required=required_inputs),
        "official_outcomes": record(
            run / "official_score/official_outcomes.json", required=required_inputs
        ),
    }


inputs = {
    "pilot": record(pilot_arg, required=required_inputs),
    "materializer": record(materializer_arg, required=required_inputs),
    "analyzer": record(analyzer_arg, required=required_inputs),
    "jlens_runner": record(runner_arg, required=required_inputs),
    "jlens_python_runner": record(python_runner_arg, required=required_inputs),
    "model_checkpoint_verifier": record(
        checkpoint_verifier_arg, required=required_inputs
    ),
    "action_protocol": record(action_protocol_arg, required=required_inputs),
    "readout_protocol": record(readout_protocol_arg, required=required_inputs),
    "chat_template": record(template_arg, required=required_inputs),
    "cohort_manifest": record(cohort_manifest_arg, required=required_inputs),
    "nf4_lens": record(nf4_lens_arg),
    "nf4_provenance": record(nf4_provenance_arg),
    "native_lens": record(native_lens_arg),
    "native_provenance": record(native_provenance_arg),
    "native_state": record(native_state_arg),
}
artifacts = {
    "campaign_evidence": record(campaign_evidence_arg, required=required_inputs),
    "prompts": record(prompts_arg),
    "prompts_summary": record(prompts_summary_arg),
    "prompts_checksum": record(prompts_checksum_arg),
    "public_report": record(public_arg),
    "nf4_report": record(nf4_arg),
    "native_report": record(native_arg),
    "public_report_checksum": record(public_checksum_arg),
    "nf4_report_checksum": record(nf4_checksum_arg),
    "native_report_checksum": record(native_checksum_arg),
    "analysis": record(analysis_arg),
    "analysis_checksum": record(analysis_checksum_arg),
}
artifacts = {key: value for key, value in artifacts.items() if value is not None}

phases = {}
for label in (
    "initialization",
    "campaign_validation",
    "materialize",
    "reuse_staging",
    "lens_artifact_validation",
    "public_preflight",
    "public",
    "nf4_preflight",
    "nf4",
    "native_preflight",
    "native",
    "analyze",
    "promotion",
):
    status_path = out / f"{label}.exit_status"
    log_path = out / "logs" / f"{label}.log"
    if status_path.is_file():
        phases[label] = {
            "status": status_path.read_text(encoding="ascii").strip(),
            "status_artifact": record(status_path, required=True),
            "log_artifact": record(log_path),
        }

failure = None
if status == "failed":
    portable_reason = failure_reason
    for absolute, replacement in sorted(
        (
            (str(out), "$OUTPUT_DIRECTORY"),
            (str(primary_run), "$PRIMARY_CAMPAIGN_RUN_ROOT"),
            (str(replication_run), "$REPLICATION_CAMPAIGN_RUN_ROOT"),
            (str(root), "$REPOSITORY_ROOT"),
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        portable_reason = portable_reason.replace(absolute, replacement)
    failure = {
        "phase": phase,
        "process_exit_status": int(process_status),
        "reason": portable_reason or "phase exited before a more specific error was recorded",
    }

value = {
    "schema_version": 1,
    "kind": "swe_verified_behavioral_combined_n20_pilot_run",
    "status": status,
    "execution_status": status,
    "scientific_status": scientific_status,
    "mode": mode,
    "failure": failure,
    "path_contract": {
        "repository_root": "directory containing this repository",
        "primary_campaign_run_root": "first completed campaign root",
        "replication_campaign_run_root": "second completed campaign root",
        "output_root": "directory containing immutable attempts and promotion links",
        "output_directory": "immutable attempt directory containing this manifest",
        "absolute_paths_embedded": False,
    },
    "attempt": {
        "id": out.name,
        "path": f"attempts/{out.name}",
    },
    "cohorts": [
        cohort_record("development", primary_campaign_arg, primary_run),
        cohort_record("replication", replication_campaign_arg, replication_run),
    ],
    "inputs": inputs,
    "artifacts": artifacts,
    "phases": phases,
    "runtime_contract": {
        "model": {
            "repo_id": model_repo,
            "revision": model_revision,
            "config_sha256": model_config_sha,
            "index_sha256": model_index_sha,
        },
        "lenses": {
            "public": {
                "repo_id": public_lens_repo,
                "revision": public_lens_revision,
                "sha256": public_lens_sha,
            },
            "nf4": {
                "sha256": nf4_lens_sha,
                "provenance_sha256": nf4_provenance_sha,
            },
            "native": {
                "sha256": native_lens_sha,
                "provenance_sha256": native_provenance_sha,
                "state_sha256": native_state_sha,
            },
        },
        "layers": [int(value) for value in layers_arg.split(",")],
        "positions": [-1],
        "top_k": 10,
        "max_model_len": int(max_model_len_arg),
        "max_num_batched_tokens": int(max_num_batched_tokens_arg),
        "mamba_block_size": int(mamba_block_size_arg),
        "enable_prefix_caching": True,
        "kv_cache_dtype": "fp8_e4m3",
        "kv_offloading_size_gib": int(kv_offloading_size_arg),
        "kv_offloading_backend": "native",
        "stream_final_only": True,
        "gpu_memory_utilization": float(gpu_memory_utilization_arg),
        "bootstrap_samples": int(bootstrap_samples_arg),
        "mtp_scope": {
            "trajectory_generation": {
                "enabled": True,
                "evidence": "campaign evidence binds server.log and exact vLLM SpecDecoding metrics token totals",
            },
            "eager_residual_capture": {
                "enabled": False,
                "language_model_only": True,
                "reason": "capture hidden states from the target language model without draft-model execution",
            },
        },
    },
    "integrity_contract": {
        "campaign_generation_sources_rehashed": True,
        "full_model_checkpoint_rehashed_before_and_after_each_replay": True,
        "prompt_bundle_bound_by_materializer_summary_and_sha256_sidecar": True,
        "reports_bound_to_exact_prompts_and_sha256_sidecars": True,
        "analysis_bound_to_exact_prompt_report_protocol_hashes": True,
    },
}

rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary = tempfile.mkstemp(prefix=f".{manifest_path.name}.", dir=manifest_path.parent)
try:
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest_path)
except BaseException:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
    raise
PY
  write_sha256_sidecar "$RUN_MANIFEST" "$RUN_MANIFEST_SHA256"
  echo "run manifest: $RUN_MANIFEST"
}

prepare_only=0
reuse_reports=0
while (($#)); do
  case "$1" in
    --prepare-only)
      prepare_only=1
      ;;
    --reuse-reports)
      reuse_reports=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      echo "ERROR: unknown argument: $1" >&2
      exit 1
      ;;
  esac
  shift
done
if ((prepare_only && reuse_reports)); then
  echo "ERROR: --prepare-only and --reuse-reports are mutually exclusive" >&2
  exit 1
fi

command -v "$MANIFEST_PY" >/dev/null 2>&1 || {
  echo "ERROR: manifest Python is missing: $MANIFEST_PY" >&2
  exit 1
}
[[ ! -L "$OUTPUT_ROOT" ]] || {
  echo "ERROR: output root must not be a symlink: $OUTPUT_ROOT" >&2
  exit 1
}
mkdir -p "$OUTPUT_ROOT/attempts"
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd)
ATTEMPTS_DIR=$OUTPUT_ROOT/attempts
LATEST_LINK=$OUTPUT_ROOT/latest
LOCK_DIR=$OUTPUT_ROOT/.pilot.lock
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "ERROR: another behavioral pilot owns $LOCK_DIR" >&2
  exit 1
fi
OUT_DIR=$(mktemp -d "$ATTEMPTS_DIR/attempt-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
LOG_DIR=$OUT_DIR/logs
CHECKSUM_DIR=$OUT_DIR/checksums
mkdir -p "$LOG_DIR" "$CHECKSUM_DIR"

PROMPTS=$OUT_DIR/prompts.json
CAMPAIGN_EVIDENCE=$OUT_DIR/campaign_evidence.json
PROMPTS_SUMMARY=$OUT_DIR/prompts_summary.json
PROMPTS_CHECKSUM=$CHECKSUM_DIR/prompts.sha256
PUBLIC_REPORT=$OUT_DIR/public-report.json
NF4_REPORT=$OUT_DIR/nf4-report.json
NATIVE_REPORT=$OUT_DIR/native-report.json
PUBLIC_CHECKSUM=$CHECKSUM_DIR/public-report.sha256
NF4_CHECKSUM=$CHECKSUM_DIR/nf4-report.sha256
NATIVE_CHECKSUM=$CHECKSUM_DIR/native-report.sha256
ANALYSIS=$OUT_DIR/analysis.json
ANALYSIS_CHECKSUM=$CHECKSUM_DIR/analysis.sha256
RUN_MANIFEST=$OUT_DIR/run_manifest.json
RUN_MANIFEST_SHA256=$OUT_DIR/run_manifest.sha256

CURRENT_PHASE=initialization
FAILURE_REASON=
ANALYSIS_SCIENTIFIC_STATUS=not_run
if ((prepare_only)); then
  RUN_MODE=prepare_only
elif ((reuse_reports)); then
  RUN_MODE=reuse_reports
else
  RUN_MODE=fresh_replay
fi
MANIFEST_EMITTED=0

finish() {
  local process_status=$?
  trap - EXIT
  if ((MANIFEST_EMITTED == 0)); then
    emit_run_manifest failed "$RUN_MODE" "$process_status" >&2 || true
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
  exit "$process_status"
}
trap finish EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if ! validate_initialization_inputs >"$LOG_DIR/initialization.log" 2>&1; then
  printf 'initialization-failed\n' >"$OUT_DIR/initialization.exit_status"
  show_failure_log "$LOG_DIR/initialization.log"
  die "behavioral pilot initialization failed"
fi
printf '0\n' >"$OUT_DIR/initialization.exit_status"

CURRENT_PHASE=campaign_validation
if ! validate_campaign_evidence >"$LOG_DIR/campaign_validation.log" 2>&1; then
  printf 'invalid-campaign-evidence\n' >"$OUT_DIR/campaign_validation.exit_status"
  show_failure_log "$LOG_DIR/campaign_validation.log"
  die "combined N=20 campaign evidence validation failed"
fi
if [[ ! -s "$CAMPAIGN_EVIDENCE" ]]; then
  printf 'missing-campaign-evidence-output\n' >"$OUT_DIR/campaign_validation.exit_status"
  die "campaign validation did not emit structured evidence"
fi
printf '0\n' >"$OUT_DIR/campaign_validation.exit_status"

run_logged materialize "$LOG_DIR/materialize.log" \
  "$PROMPTS" "$PROMPTS_SUMMARY" -- \
  "$VLLM_PY" "$MATERIALIZER" \
  --cohort "$PRIMARY_CAMPAIGN" "$PRIMARY_RUN_ROOT" \
  --cohort "$REPLICATION_CAMPAIGN" "$REPLICATION_RUN_ROOT" \
  --cohort-manifest "$COHORT_MANIFEST" \
  --action-protocol "$ACTION_PROTOCOL" \
  --template "$CHAT_TEMPLATE" \
  --require-official-outcomes \
  --output "$PROMPTS" \
  --summary "$PROMPTS_SUMMARY"
if ! validate_prompt_lengths >>"$LOG_DIR/materialize.log" 2>&1 \
  || ! validate_materialized_bundle >>"$LOG_DIR/materialize.log" 2>&1; then
  printf 'invalid-materialized-bundle\n' >"$OUT_DIR/materialize.exit_status"
  show_failure_log "$LOG_DIR/materialize.log"
  die "combined behavioral materialization violates the pinned schema"
fi
write_sha256_sidecar "$PROMPTS" "$PROMPTS_CHECKSUM"
verify_sha256_sidecar "$PROMPTS" "$PROMPTS_CHECKSUM" "behavioral prompts"

if ((prepare_only)); then
  CURRENT_PHASE=complete
  emit_run_manifest prepared "$RUN_MODE" 0
  trap '' HUP INT TERM
  promote_attempt latest-prepared
  MANIFEST_EMITTED=1
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM
  echo "prepared combined N=20 prompts: $PROMPTS"
  echo "latest prepared attempt: $OUTPUT_ROOT/latest-prepared"
  exit 0
fi

if ((reuse_reports)); then
  stage_reused_reports
  reuse_lens_report public "$PUBLIC_REPORT"
  reuse_lens_report nf4 "$NF4_REPORT"
  reuse_lens_report native "$NATIVE_REPORT"
else
  CURRENT_PHASE=lens_artifact_validation
  if ! validate_lens_artifact_inputs >"$LOG_DIR/lens_artifact_validation.log" 2>&1; then
    printf 'lens-artifact-validation-failed\n' \
      >"$OUT_DIR/lens_artifact_validation.exit_status"
    show_failure_log "$LOG_DIR/lens_artifact_validation.log"
    die "lens artifact validation failed"
  fi
  printf '0\n' >"$OUT_DIR/lens_artifact_validation.exit_status"

  COMMON_ARGS=(
    --prompts-file "$PROMPTS"
    --layers "$LAYERS"
    --positions=-1
    --top-k 10
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --mamba-block-size "$MAMBA_BLOCK_SIZE"
    --enable-prefix-caching
    --kv-cache-dtype fp8_e4m3
    --kv-offloading-size "$KV_OFFLOADING_SIZE"
    --kv-offloading-backend native
    --stream-final-only
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  )

  run_lens public "$PUBLIC_REPORT" "$LOG_DIR/public.log" \
    "$JLENS_RUNNER" \
    --lens-kind public \
    "${COMMON_ARGS[@]}"

  run_lens nf4 "$NF4_REPORT" "$LOG_DIR/nf4.log" \
    "$JLENS_RUNNER" \
    --lens-kind nf4 \
    --lens-path "$NF4_LENS_PATH" \
    --lens-sha256 "$NF4_LENS_SHA256" \
    --lens-provenance "$NF4_PROVENANCE_PATH" \
    "${COMMON_ARGS[@]}"

  run_lens native "$NATIVE_REPORT" "$LOG_DIR/native.log" \
    "$JLENS_RUNNER" \
    --lens-kind nvfp4-ste \
    --lens-path "$NATIVE_LENS_PATH" \
    --lens-sha256 "$NATIVE_LENS_SHA256" \
    --lens-provenance "$NATIVE_PROVENANCE_PATH" \
    --lens-state "$NATIVE_STATE_PATH" \
    --lens-state-sha256 "$NATIVE_STATE_SHA256" \
    "${COMMON_ARGS[@]}"
fi

verify_sha256_sidecar "$PUBLIC_REPORT" "$PUBLIC_CHECKSUM" "public report"
verify_sha256_sidecar "$NF4_REPORT" "$NF4_CHECKSUM" "NF4 report"
verify_sha256_sidecar "$NATIVE_REPORT" "$NATIVE_CHECKSUM" "native report"
verify_sha256_sidecar "$PROMPTS" "$PROMPTS_CHECKSUM" "behavioral prompts"

run_logged analyze "$LOG_DIR/analyze.log" "$ANALYSIS" -- \
  "$VLLM_PY" "$ANALYZER" \
  --prompts "$PROMPTS" \
  --public-report "$PUBLIC_REPORT" \
  --nf4-report "$NF4_REPORT" \
  --native-report "$NATIVE_REPORT" \
  --protocol "$READOUT_PROTOCOL" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --output "$ANALYSIS"
if ! validate_analysis >>"$LOG_DIR/analyze.log" 2>&1; then
  printf 'invalid-analysis\n' >"$OUT_DIR/analyze.exit_status"
  show_failure_log "$LOG_DIR/analyze.log"
  die "behavioral analysis does not bind the exact replay evidence"
fi
if ! ANALYSIS_SCIENTIFIC_STATUS=$("$MANIFEST_PY" - "$ANALYSIS" <<'PY'
import json
from pathlib import Path
import sys

print(json.loads(Path(sys.argv[1]).read_bytes())["status"])
PY
); then
  printf 'scientific-status-read-failed\n' >"$OUT_DIR/analyze.exit_status"
  die "could not preserve the validated analysis scientific status"
fi
write_sha256_sidecar "$ANALYSIS" "$ANALYSIS_CHECKSUM"
verify_sha256_sidecar "$ANALYSIS" "$ANALYSIS_CHECKSUM" "behavioral analysis"

CURRENT_PHASE=complete
emit_run_manifest complete "$RUN_MODE" 0
trap '' HUP INT TERM
promote_attempt latest
MANIFEST_EMITTED=1
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
echo "analysis: $ANALYSIS"
echo "latest complete attempt: $LATEST_LINK"
