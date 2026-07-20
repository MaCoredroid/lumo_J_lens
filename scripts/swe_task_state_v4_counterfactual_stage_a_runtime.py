#!/usr/bin/env python3
"""CPU-only preflight and artifact verifier for the V4 Stage-A runtime.

This module deliberately has no model-loading, generation, activation-capture,
condition-join, or Stage-B command.  It authenticates the prospective runtime
contract and its two opaque token-ID bundles, records a no-runtime preflight,
and can later verify complete target-model capture/generation artifacts.  The
post-run verifier requires the actual safetensors shards and generated token
records; a JSON object that merely resembles the declared schema cannot pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_counterfactual_stage_a_runtime.json"
)
# Frozen after the additive config is finalized.  Keeping both byte and
# canonical-object digests catches byte edits and in-memory mutation in tests.
CONFIG_SHA256 = "e4ee185167732363e3e048b9b2ea7cbe85acb2f15e09692cec53ae87186cdb64"
CONFIG_CANONICAL_SHA256 = "2e7eceb34d88c9767d163c43a9d1858b85aa7a308b0920548f0bc2c8f8a7c61e"
SCRIPT_PATH = Path(__file__).resolve()

SCHEMA_VERSION = 1
PREFLIGHT_KIND = "swe_task_state_v4_counterfactual_stage_a_runtime_preflight"
CAPTURE_VERIFICATION_KIND = (
    "swe_task_state_v4_counterfactual_stage_a_capture_verification"
)
GENERATION_VERIFICATION_KIND = (
    "swe_task_state_v4_counterfactual_stage_a_generation_verification"
)
CAPTURE_KIND = "swe_task_state_v4_counterfactual_stage_a_authenticated_capture"
GENERATION_KIND = (
    "swe_task_state_v4_counterfactual_stage_a_authenticated_generation"
)
COMPLETION_KIND = (
    "swe_task_state_v4_counterfactual_stage_a_visible_completion_bundle"
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OPAQUE_ID_RE = SHA256_RE
HARD_FORBIDDEN_PATH_FRAGMENTS = (
    "reserved",
    "validation",
    "split-manifest",
    "condition-key",
    "stage-b",
    "semantic-answer",
    "expectation",
)
OUTPUT_FALSE_CLAIMS = {
    "private_chain_of_thought_reconstructed": False,
    "cot_or_cot_like_decoding_established": False,
    "subjective_confidence_inferred": False,
    "subjective_doubt_inferred": False,
    "experienced_stress_inferred": False,
    "experienced_emotion_inferred": False,
    "causal_affect_or_state_effect_established": False,
    "incremental_activation_readout_established": False,
    "outer_or_reserved_validation_generalization_established": False,
}


class StageARuntimeError(ValueError):
    """Raised before an unsafe or unauthenticated Stage-A operation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageARuntimeError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StageARuntimeError(f"cannot load strict JSON {path}: {error}") from error


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as error:
        raise StageARuntimeError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    _require(set(value) == set(expected), f"{label} keys changed")


def _path_forbidden_lexically(path: Path) -> bool:
    normalized = Path(os.path.normpath(os.path.abspath(os.fspath(path))))
    return any(
        fragment in component.lower()
        for component in normalized.parts
        for fragment in HARD_FORBIDDEN_PATH_FRAGMENTS
    )


def lexical_path_preflight(paths: Iterable[Path | None]) -> None:
    """Reject forbidden path text before resolve/stat/hash/read/write."""

    for path in paths:
        if path is not None and _path_forbidden_lexically(Path(path)):
            raise StageARuntimeError(
                f"forbidden path rejected before filesystem access: {path}"
            )


def canonical_path_preflight(
    *, input_paths: Iterable[Path | None], output_paths: Iterable[Path | None]
) -> None:
    for path, is_input in [
        *((item, True) for item in input_paths if item is not None),
        *((item, False) for item in output_paths if item is not None),
    ]:
        try:
            resolved = (
                Path(path).resolve(strict=True)
                if is_input
                else Path(path).parent.resolve(strict=True) / Path(path).name
            )
        except OSError as error:
            raise StageARuntimeError(f"cannot resolve path {path}: {error}") from error
        if _path_forbidden_lexically(resolved):
            raise StageARuntimeError(f"forbidden canonical path rejected: {path}")


def _root_path(raw: str, *, label: str) -> Path:
    _require(isinstance(raw, str) and bool(raw), f"{label} path is invalid")
    path = ROOT / raw
    lexical_path_preflight((path,))
    return path


def _regular_file(path: Path, label: str, *, allow_symlink: bool = False) -> Path:
    if path.is_symlink() and not allow_symlink:
        raise StageARuntimeError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise StageARuntimeError(f"{label} is unavailable: {path}: {error}") from error
    _require(resolved.is_file(), f"{label} must be a regular file: {path}")
    return resolved


def _file_record(path: Path, *, display_root: Path = ROOT) -> dict[str, Any]:
    resolved = _regular_file(path, "bound file", allow_symlink=True)
    try:
        display = str(path.resolve(strict=False).relative_to(display_root.resolve()))
    except ValueError:
        display = str(path)
    return {
        "path": display,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _validate_binding_shape(
    value: Any, label: str, *, extra_keys: Iterable[str] = ()
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} binding must be an object")
    binding = dict(value)
    _exact_keys(
        binding,
        ("path", "sha256", "size_bytes", *tuple(extra_keys)),
        f"{label} binding",
    )
    _require(
        isinstance(binding["path"], str)
        and bool(binding["path"])
        and _is_sha256(binding["sha256"])
        and isinstance(binding["size_bytes"], int)
        and not isinstance(binding["size_bytes"], bool)
        and binding["size_bytes"] > 0,
        f"{label} binding is invalid",
    )
    return binding


def _verify_bound_file(
    value: Mapping[str, Any], label: str, *, extra_keys: Iterable[str] = ()
) -> Path:
    binding = _validate_binding_shape(value, label, extra_keys=extra_keys)
    path = _root_path(binding["path"], label=label)
    canonical_path_preflight(input_paths=(path,), output_paths=())
    resolved = _regular_file(path, label)
    _require(resolved.stat().st_size == binding["size_bytes"], f"{label} size changed")
    _require(sha256_file(resolved) == binding["sha256"], f"{label} hash changed")
    return path


def _canonical_config_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def validate_config(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "Stage-A runtime config must be an object")
    config = dict(value)
    _require(sha256_file(CONFIG_PATH) == CONFIG_SHA256, "runtime config byte hash changed")
    _require(
        _canonical_config_sha256(config) == CONFIG_CANONICAL_SHA256,
        "runtime config object changed",
    )
    _exact_keys(
        config,
        (
            "schema_version",
            "id",
            "status",
            "purpose",
            "chronology",
            "frozen_inputs",
            "downstream_visible_annotation_contract_reference_only",
            "model",
            "public_j_lens",
            "tokenization",
            "runtime",
            "capture",
            "generation",
            "environment",
            "output_contracts",
            "postrun_authentication",
            "forbidden_access",
            "claim_scope",
            "output",
        ),
        "runtime config",
    )
    _require(
        config["schema_version"] == 1
        and config["id"]
        == "swe-task-state-v4-counterfactual-selector-control-stage-a-runtime-v1"
        and config["status"]
        == "prospective_stage_a_execution_contract_cpu_preflight_only_no_model_runtime",
        "runtime config identity changed",
    )
    chronology = config["chronology"]
    _require(
        chronology["gpu_execution_authorized_by_this_artifact"] is False
        and chronology["stage_b_materialization_or_runtime_authorized"] is False
        and chronology["condition_join_or_effect_analysis_authorized"] is False
        and chronology["reference_must_finish_before_capture"] is True
        and chronology["capture_must_verify_before_generation_model_load"] is True,
        "execution chronology or authorization changed",
    )
    model = config["model"]
    _require(
        model["repo_id"] == "nvidia/Qwen3.6-27B-NVFP4"
        and model["revision"] == "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
        and model["snapshot_tree_sha256"]
        == "9e81d31df546344ad68696c3cfd6cadce4ad6d3952710a3dc2021c1c2d42414d"
        and model["snapshot_file_count"] == 17
        and model["snapshot_size_bytes"] == 21941623844
        and model["hidden_size"] == 5120
        and model["layer_count"] == 64
        and model["vocabulary_size"] == 248320
        and model["maximum_position_embeddings"] == 262144
        and model["local_files_only"] is True,
        "target model binding changed",
    )
    runtime = config["runtime"]
    _require(
        runtime["max_model_len"] == 49152
        and runtime["max_num_batched_tokens"] == 4096
        and runtime["max_num_seqs"] == 1
        and runtime["input_truncation"] is False
        and runtime["seed"] == 0,
        "runtime context, batching, truncation, or seed changed",
    )
    capture = config["capture"]
    _require(
        capture["prompt_count"] == 500
        and capture["layers"] == list(range(24, 48))
        and capture["positions_argument"] == [-1]
        and capture["stream_final_only"] is True
        and capture["tensor_shape_per_boundary"] == [24, 5120]
        and capture["tensor_storage_dtype"] == "little_endian_float32"
        and capture["expected_shard_count"] == 500
        and all(
            capture[key] is True
            for key in (
                "require_safetensors_reload_exact_equality",
                "require_all_64_layer_residual_manifest_per_boundary",
                "require_reference_capture_residual_manifest_exact_equality",
                "require_final_model_top1_greedy_parity",
                "require_final_norm_reconstruction_within_tolerance",
                "require_final_logits_reconstruction_within_tolerance",
            )
        ),
        "capture layers, positions, geometry, or authentication changed",
    )
    generation = config["generation"]
    _require(
        generation["prompt_count"] == 240
        and generation["input_truncation"] is False
        and generation["sampling"]
        == {
            "max_new_tokens": 256,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "skip_special_tokens": True,
            "spaces_between_special_tokens": True,
        }
        and generation["requires_authenticated_capture_manifest_and_all_shards_cpu_verified"]
        is True
        and generation["completion_annotation_or_label_extraction_in_runtime"] is False,
        "generation sampling, order, or blinding changed",
    )
    forbidden = config["forbidden_access"]
    _require(
        tuple(forbidden["forbidden_path_fragments_checked_before_filesystem_access"])
        == HARD_FORBIDDEN_PATH_FRAGMENTS
        and forbidden["no_condition_or_code_map_join_in_this_runtime"] is True,
        "forbidden-access guard changed",
    )
    claims = config["claim_scope"]
    _require(
        claims["cpu_preflight_contract_implemented"] is True
        and all(
            value is False
            for key, value in claims.items()
            if key != "cpu_preflight_contract_implemented"
        ),
        "prospective claim scope changed",
    )
    return config


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    lexical_path_preflight((path,))
    canonical_path_preflight(input_paths=(path,), output_paths=())
    _require(path.resolve(strict=True) == CONFIG_PATH.resolve(strict=True), "only pinned config allowed")
    return validate_config(load_json(path))


def _validate_source_bindings(config: Mapping[str, Any]) -> dict[str, Path]:
    frozen = config["frozen_inputs"]
    simple = (
        "selector_scaffold_config",
        "selector_scaffold_implementation",
        "materialization_manifest",
        "raw_capture_protocol",
        "reference_runner",
        "capture_runner",
        "runtime_shell_wrapper",
        "runtime_requirements",
    )
    paths: dict[str, Path] = {}
    for name in simple:
        paths[name] = _verify_bound_file(frozen[name], name, extra_keys=("access",))
    for name in ("stage_a_capture_prompts", "stage_a_generation_prompts"):
        record = frozen[name]
        extra = (
            "record_count",
            "allowed_row_keys",
            "ordered_id_list_sha256",
            "row_contract_sha256",
            "minimum_token_count",
            "maximum_token_count",
        )
        if name == "stage_a_generation_prompts":
            extra = (*extra, "every_row_must_equal_one_capture_row_by_id_and_tokens")
        paths[name] = _verify_bound_file(record, name, extra_keys=extra)
    return paths


def _validate_schema_bindings(config: Mapping[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, record in config["output_contracts"].items():
        if name == "prospective_instance_hash_policy":
            continue
        path = _verify_bound_file(record, name, extra_keys=("schema_id",))
        schema = load_json(path)
        _require(
            isinstance(schema, dict)
            and schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
            and schema.get("$id") == record["schema_id"],
            f"{name} schema identity changed",
        )
        result[name] = path
    _require(
        set(result)
        == {
            "capture_manifest_schema",
            "generation_manifest_schema",
            "visible_completion_bundle_schema",
        },
        "output schema set changed",
    )
    return result


def snapshot_inventory(snapshot_path: Path) -> dict[str, Any]:
    """Hash exact resolved snapshot files without importing model code."""

    lexical_path_preflight((snapshot_path,))
    _require(snapshot_path.is_dir(), f"model snapshot is unavailable: {snapshot_path}")
    entries: list[dict[str, Any]] = []
    for path in sorted(
        snapshot_path.rglob("*"), key=lambda item: item.relative_to(snapshot_path).as_posix()
    ):
        if path.is_dir():
            continue
        _require(path.is_file(), f"snapshot entry is not a file: {path}")
        resolved = path.resolve(strict=True)
        entries.append(
            {
                "path": path.relative_to(snapshot_path).as_posix(),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    _require(bool(entries), "model snapshot is empty")
    return {
        "tree_sha256": sha256_bytes(canonical_json_bytes(entries)),
        "file_count": len(entries),
        "size_bytes": sum(item["size_bytes"] for item in entries),
        "files": entries,
    }


def resolve_and_verify_model(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    spec = config["model"]
    snapshot = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / spec["local_snapshot_cache_relative_path"]
    )
    lexical_path_preflight((snapshot,))
    canonical_path_preflight(input_paths=(snapshot,), output_paths=())
    inventory = snapshot_inventory(snapshot)
    _require(
        inventory["tree_sha256"] == spec["snapshot_tree_sha256"]
        and inventory["file_count"] == spec["snapshot_file_count"]
        and inventory["size_bytes"] == spec["snapshot_size_bytes"],
        "target model snapshot inventory changed",
    )
    for filename, key in (
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "model_index_sha256"),
        ("tokenizer.json", "tokenizer_json_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
        ("generation_config.json", "generation_config_sha256"),
    ):
        _require(
            sha256_file(snapshot / filename) == spec[key],
            f"target model {filename} changed",
        )
    model_config = load_json(snapshot / "config.json")
    text = model_config.get("text_config", {})
    _require(
        text.get("hidden_size") == spec["hidden_size"]
        and text.get("num_hidden_layers") == spec["layer_count"]
        and text.get("vocab_size") == spec["vocabulary_size"]
        and text.get("max_position_embeddings") == spec["maximum_position_embeddings"],
        "target model geometry changed",
    )
    return snapshot, inventory


def resolve_and_verify_lens(config: Mapping[str, Any]) -> Path:
    spec = config["public_j_lens"]
    snapshot = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--" + spec["repo_id"].replace("/", "--"))
        / "snapshots"
        / spec["revision"]
    )
    path = snapshot / spec["filename"]
    lexical_path_preflight((path,))
    canonical_path_preflight(input_paths=(path,), output_paths=())
    resolved = _regular_file(path, "public-J lens", allow_symlink=True)
    _require(
        resolved.stat().st_size == spec["size_bytes"]
        and sha256_file(resolved) == spec["sha256"],
        "public-J lens bytes changed",
    )
    return path


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    return sha256_bytes(canonical_json_bytes(list(token_ids)))


def prompt_row_contract(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "token_ids_sha256": token_ids_sha256(row["token_ids"]),
            "token_count": len(row["token_ids"]),
        }
        for row in rows
    ]


def validate_prompt_bundle(
    value: Any, *, spec: Mapping[str, Any], label: str, vocabulary_size: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(
        isinstance(value, list) and len(value) == spec["record_count"],
        f"{label} record count changed",
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        _require(isinstance(raw, dict), f"{label} row {index} is not an object")
        _exact_keys(raw, spec["allowed_row_keys"], f"{label} row {index}")
        prompt_id = raw.get("id")
        token_ids = raw.get("token_ids")
        _require(
            isinstance(prompt_id, str)
            and OPAQUE_ID_RE.fullmatch(prompt_id) is not None
            and prompt_id not in seen,
            f"{label} row {index} opaque ID changed or duplicated",
        )
        _require(
            isinstance(token_ids, list)
            and bool(token_ids)
            and all(
                isinstance(token_id, int)
                and not isinstance(token_id, bool)
                and 0 <= token_id < vocabulary_size
                for token_id in token_ids
            ),
            f"{label} row {index} token IDs are invalid",
        )
        seen.add(prompt_id)
        rows.append({"id": prompt_id, "token_ids": list(token_ids)})
    lengths = [len(row["token_ids"]) for row in rows]
    id_list_sha256 = sha256_bytes(
        canonical_json_bytes([row["id"] for row in rows])
    )
    contract = prompt_row_contract(rows)
    contract_sha256 = sha256_bytes(canonical_json_bytes(contract))
    _require(
        min(lengths) == spec["minimum_token_count"]
        and max(lengths) == spec["maximum_token_count"]
        and id_list_sha256 == spec["ordered_id_list_sha256"]
        and contract_sha256 == spec["row_contract_sha256"],
        f"{label} lengths, ordering, or row contract changed",
    )
    return rows, {
        "record_count": len(rows),
        "minimum_token_count": min(lengths),
        "maximum_token_count": max(lengths),
        "ordered_id_list_sha256": id_list_sha256,
        "row_contract_sha256": contract_sha256,
        "opaque_ids_unique_and_format_valid": True,
        "record_field_allowlist_passed": True,
    }


def load_and_validate_prompt_bundles(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frozen = config["frozen_inputs"]
    capture_rows, capture_summary = validate_prompt_bundle(
        load_json(paths["stage_a_capture_prompts"]),
        spec=frozen["stage_a_capture_prompts"],
        label="Stage-A capture prompts",
        vocabulary_size=config["model"]["vocabulary_size"],
    )
    generation_rows, generation_summary = validate_prompt_bundle(
        load_json(paths["stage_a_generation_prompts"]),
        spec=frozen["stage_a_generation_prompts"],
        label="Stage-A generation prompts",
        vocabulary_size=config["model"]["vocabulary_size"],
    )
    capture_by_id = {row["id"]: row["token_ids"] for row in capture_rows}
    _require(
        all(
            row["id"] in capture_by_id
            and row["token_ids"] == capture_by_id[row["id"]]
            for row in generation_rows
        ),
        "generation prompts are not exact capture rows",
    )
    _require(
        len(capture_by_id) - len(generation_rows)
        == frozen["capture_prompt_rows_not_used_for_generation"],
        "capture/generation prompt subset cardinality changed",
    )
    _require(
        capture_summary["maximum_token_count"] + 1
        <= config["runtime"]["max_model_len"],
        "capture forward would truncate or exceed max_model_len",
    )
    _require(
        generation_summary["maximum_token_count"]
        + config["generation"]["sampling"]["max_new_tokens"]
        <= config["runtime"]["max_model_len"],
        "generation would truncate or exceed max_model_len",
    )
    return capture_rows, generation_rows, {
        "capture": capture_summary,
        "generation": generation_summary,
        "generation_rows_are_exact_capture_rows": True,
        "capture_only_row_count": len(capture_rows) - len(generation_rows),
        "no_input_truncation_capacity_passed": True,
    }


def _nvidia_smi() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise StageARuntimeError(f"cannot query pinned GPU environment: {error}") from error
    rows = [line for line in output.splitlines() if line.strip()]
    _require(len(rows) == 1, "runtime contract requires exactly one visible GPU")
    values = [item.strip() for item in rows[0].split(",")]
    _require(len(values) == 4, "unexpected nvidia-smi output")
    return dict(
        zip(
            ("name", "driver_version", "memory_total_mib", "compute_capability"),
            values,
            strict=True,
        )
    )


def verify_runtime_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["environment"]
    expected_python = (ROOT / spec["python_executable_repo_relative"]).resolve()
    _require(Path(sys.executable).resolve() == expected_python, "wrong Python executable")
    _require(platform.python_version() == spec["python_version"], "Python version changed")
    observed_packages: dict[str, str] = {}
    for package, expected in spec["packages"].items():
        try:
            observed = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise StageARuntimeError(f"required package is missing: {package}") from error
        _require(observed == expected, f"package version changed: {package}")
        observed_packages[package] = observed
    cuda_home = (ROOT / spec["cuda_toolkit_repo_relative"]).resolve()
    nvcc = cuda_home / "bin" / "nvcc"
    resolved_nvcc = _regular_file(nvcc, "pinned nvcc", allow_symlink=True)
    _require(
        resolved_nvcc.stat().st_size == spec["nvcc_size_bytes"]
        and sha256_file(resolved_nvcc) == spec["nvcc_sha256"],
        "pinned nvcc bytes changed",
    )
    observed_environment: dict[str, str] = {}
    for name, expected in spec["required_environment"].items():
        actual = os.environ.get(name)
        if name == "CUDA_HOME":
            expected = str((ROOT / expected).resolve())
            actual = str(Path(actual).resolve()) if actual else actual
        _require(actual == expected, f"required environment variable changed: {name}")
        observed_environment[name] = str(actual)
    gpu = _nvidia_smi()
    _require(gpu == spec["gpu"], "GPU or driver environment changed")
    return {
        "python_executable": str(expected_python),
        "python_version": platform.python_version(),
        "packages": observed_packages,
        "cuda_home": str(cuda_home),
        "nvcc_sha256": spec["nvcc_sha256"],
        "required_environment": observed_environment,
        "gpu": gpu,
        "exact_runtime_environment_passed": True,
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    lexical_path_preflight((path,))
    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path_preflight(input_paths=(), output_paths=(path,))
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _require(
        not temporary.exists() and not temporary.is_symlink(),
        f"temporary output already exists: {temporary}",
    )
    try:
        temporary.write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def build_preflight_receipt(
    config: Mapping[str, Any],
    *,
    source_paths: Mapping[str, Path],
    schema_paths: Mapping[str, Path],
    prompt_summary: Mapping[str, Any],
    snapshot_path: Path,
    snapshot_inventory_record: Mapping[str, Any],
    lens_path: Path,
    environment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    frozen = config["frozen_inputs"]
    verified_inputs: dict[str, Any] = {}
    for name, path in source_paths.items():
        record = frozen[name]
        verified_inputs[name] = {
            "path": record["path"],
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
            "access": record.get("access", "opaque_token_ids_only"),
            "hash_verified": True,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PREFLIGHT_KIND,
        "status": "passed_cpu_preflight_only_no_model_runtime",
        "runtime_config": {
            "path": str(CONFIG_PATH.relative_to(ROOT)),
            "sha256": CONFIG_SHA256,
            "canonical_sha256": CONFIG_CANONICAL_SHA256,
            "size_bytes": CONFIG_PATH.stat().st_size,
        },
        "implementation": {
            **_file_record(SCRIPT_PATH),
            "gpu_execution_command_present": False,
        },
        "verified_inputs": verified_inputs,
        "prompt_contract": dict(prompt_summary),
        "model": {
            "repo_id": config["model"]["repo_id"],
            "revision": config["model"]["revision"],
            "snapshot_path": str(snapshot_path.resolve()),
            "snapshot_tree_sha256": snapshot_inventory_record["tree_sha256"],
            "snapshot_file_count": snapshot_inventory_record["file_count"],
            "snapshot_size_bytes": snapshot_inventory_record["size_bytes"],
            "weights_loaded": False,
        },
        "public_j_lens": {
            "path": str(lens_path.resolve()),
            "sha256": config["public_j_lens"]["sha256"],
            "size_bytes": config["public_j_lens"]["size_bytes"],
            "loaded": False,
        },
        "runtime_contract": {
            "parameters": config["runtime"],
            "capture": config["capture"],
            "generation": config["generation"],
            "environment_strictly_checked": environment is not None,
            "environment": environment,
        },
        "output_schemas": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": config["output_contracts"][name]["sha256"],
                "size_bytes": config["output_contracts"][name]["size_bytes"],
            }
            for name, path in schema_paths.items()
        },
        "safety": {
            "runtime_input_allowlist_enforced": True,
            "hash_only_materialization_manifest_not_parsed": True,
            "split_manifest_read": False,
            "condition_key_read": False,
            "stage_b_read_or_materialized": False,
            "semantic_answers_read": False,
            "join_conditions_read": False,
            "reserved_validation_access_authorized": False,
        },
        "execution_state": {
            "target_model_loaded": False,
            "reference_forward_run": False,
            "activation_capture_run": False,
            "completion_generation_run": False,
            "visible_annotation_run": False,
            "condition_join_run": False,
            "stage_b_any_runtime": False,
        },
        "claim_scope": config["claim_scope"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run_preflight(args: argparse.Namespace) -> int:
    lexical_path_preflight((args.config, args.output))
    config = load_and_validate_config(args.config)
    source_paths = _validate_source_bindings(config)
    schema_paths = _validate_schema_bindings(config)
    snapshot_path, inventory = resolve_and_verify_model(config)
    lens_path = resolve_and_verify_lens(config)
    _capture_rows, _generation_rows, prompt_summary = load_and_validate_prompt_bundles(
        config, source_paths
    )
    environment = verify_runtime_environment(config) if args.require_runtime_environment else None
    receipt = build_preflight_receipt(
        config,
        source_paths=source_paths,
        schema_paths=schema_paths,
        prompt_summary=prompt_summary,
        snapshot_path=snapshot_path,
        snapshot_inventory_record=inventory,
        lens_path=lens_path,
        environment=environment,
    )
    _atomic_write_json(args.output, receipt)
    print(
        f"wrote CPU-only Stage-A preflight for 500 captures and 240 generations to {args.output}",
        file=sys.stderr,
    )
    return 0


def _manifest_file_path(root: Path, raw: Any, label: str) -> Path:
    _require(isinstance(raw, str) and bool(raw), f"{label} path is invalid")
    relative = Path(raw)
    _require(not relative.is_absolute() and ".." not in relative.parts, f"{label} path escapes root")
    path = root / relative
    lexical_path_preflight((path,))
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise StageARuntimeError(f"{label} path escapes its output root") from error
    return path


def _verify_manifest_file_binding(
    root: Path, value: Any, label: str
) -> tuple[Path, dict[str, Any]]:
    binding = _validate_binding_shape(value, label)
    path = _manifest_file_path(root, binding["path"], label)
    resolved = _regular_file(path, label)
    _require(
        resolved.stat().st_size == binding["size_bytes"]
        and sha256_file(resolved) == binding["sha256"],
        f"{label} file binding changed",
    )
    return path, binding


def _expected_prompt_binding(
    config: Mapping[str, Any], name: str
) -> dict[str, Any]:
    spec = config["frozen_inputs"][name]
    return {
        "path": spec["path"],
        "sha256": spec["sha256"],
        "size_bytes": spec["size_bytes"],
        "record_count": spec["record_count"],
        "row_contract_sha256": spec["row_contract_sha256"],
    }


def _expected_implementation(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = config["frozen_inputs"]
    return {
        output_name: {
            "path": frozen[input_name]["path"],
            "sha256": frozen[input_name]["sha256"],
            "size_bytes": frozen[input_name]["size_bytes"],
        }
        for output_name, input_name in (
            ("reference_runner", "reference_runner"),
            ("capture_runner", "capture_runner"),
            ("shell_wrapper", "runtime_shell_wrapper"),
        )
    }


def _validate_runtime_output_record(
    value: Any, config: Mapping[str, Any], *, include_gpu: bool = True
) -> None:
    _require(isinstance(value, dict), "runtime output record must be an object")
    expected_keys = {"environment", "parameters", "started_at", "completed_at"}
    if include_gpu:
        expected_keys.add("gpu")
    _exact_keys(value, expected_keys, "runtime output record")
    environment = value["environment"]
    _require(
        isinstance(environment, dict)
        and environment.get("python_version") == config["environment"]["python_version"]
        and environment.get("packages") == config["environment"]["packages"]
        and environment.get("required_environment")
        == config["environment"]["required_environment"],
        "runtime output environment does not match contract",
    )
    _require(value["parameters"] == config["runtime"], "runtime output parameters changed")
    if include_gpu:
        _require(value["gpu"] == config["environment"]["gpu"], "runtime GPU record changed")
    for name in ("started_at", "completed_at"):
        _require(isinstance(value[name], str) and bool(value[name]), f"{name} missing")
        try:
            datetime.fromisoformat(value[name].replace("Z", "+00:00"))
        except ValueError as error:
            raise StageARuntimeError(f"{name} is not an ISO date-time") from error


def _validate_model_output_record(
    value: Any, config: Mapping[str, Any], *, phase: str
) -> None:
    _require(isinstance(value, dict), "model output record must be an object")
    before_key = (
        "checkpoint_validated_before_model_load"
        if phase == "capture"
        else f"checkpoint_validated_before_{phase}"
    )
    after_key = f"checkpoint_validated_after_{phase}"
    expected_keys = {
        "repo_id",
        "revision",
        "snapshot_tree_sha256_before",
        "snapshot_tree_sha256_after",
        before_key,
        after_key,
    }
    if phase == "capture":
        expected_keys |= {"snapshot_file_count", "snapshot_size_bytes"}
    _exact_keys(value, expected_keys, "model output record")
    model = config["model"]
    _require(
        value["repo_id"] == model["repo_id"]
        and value["revision"] == model["revision"]
        and value["snapshot_tree_sha256_before"] == model["snapshot_tree_sha256"]
        and value["snapshot_tree_sha256_after"] == model["snapshot_tree_sha256"]
        and value[before_key] is True
        and value[after_key] is True,
        "model execution lineage changed",
    )
    if phase == "capture":
        _require(
            value["snapshot_file_count"] == model["snapshot_file_count"]
            and value["snapshot_size_bytes"] == model["snapshot_size_bytes"],
            "model snapshot inventory summary changed",
        )


def _validate_residual_manifest(
    value: Any, *, expected_position: int, label: str
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    _exact_keys(
        value,
        ("sha256", "tensor_count", "logical_bytes", "token_position"),
        label,
    )
    _require(
        _is_sha256(value["sha256"])
        and value["tensor_count"] == 64
        and value["logical_bytes"] == 64 * 5120 * 4
        and value["token_position"] == expected_position,
        f"{label} identity or geometry changed",
    )
    return dict(value)


def _logical_tensor_sha256(array: Any, *, name: str, layers: Sequence[int]) -> str:
    import numpy as np

    _require(
        isinstance(array, np.ndarray)
        and list(array.shape) == [len(layers), 5120]
        and array.dtype == np.dtype("float32")
        and bool(np.isfinite(array).all()),
        f"{name} safetensors value, shape, dtype, or finiteness changed",
    )
    little = np.asarray(array, dtype="<f4", order="C")
    header = canonical_json_bytes(
        {
            "name": name,
            "layers": list(layers),
            "shape": list(little.shape),
            "dtype": "little-endian-float32",
        }
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(little.tobytes(order="C"))
    return digest.hexdigest()


def _verify_safetensors_shard(
    path: Path,
    record: Mapping[str, Any],
    *,
    layers: Sequence[int],
) -> None:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise StageARuntimeError(
            "safetensors is required for complete capture verification"
        ) from error
    _require(not path.is_symlink(), f"capture shard must not be a symlink: {path}")
    resolved = _regular_file(path, "capture shard")
    _require(
        resolved.stat().st_size == record["size_bytes"]
        and sha256_file(resolved) == record["sha256"],
        f"capture shard bytes changed: {path}",
    )
    try:
        with safe_open(resolved, framework="np", device="cpu") as handle:
            keys = sorted(handle.keys())
            _require(keys == ["public_j_state", "raw_residual"], "shard tensor keys changed")
            public_j = handle.get_tensor("public_j_state")
            raw = handle.get_tensor("raw_residual")
    except (OSError, RuntimeError, ValueError) as error:
        raise StageARuntimeError(f"cannot reload safetensors shard {path}: {error}") from error
    _require(
        _logical_tensor_sha256(raw, name="raw_residual", layers=layers)
        == record["raw_residual_logical_sha256"],
        f"raw residual logical hash changed: {path}",
    )
    _require(
        _logical_tensor_sha256(public_j, name="public_j_state", layers=layers)
        == record["public_j_state_logical_sha256"],
        f"public-J logical hash changed: {path}",
    )


def _capture_boundary_contract(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": row["index"],
            "prompt_id": row["prompt_id"],
            "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
            "prompt_token_count": row["prompt_token_count"],
            "token_position": row["token_position"],
            "reference_residual_manifest_sha256": row["reference_residual_manifest"][
                "sha256"
            ],
            "capture_residual_manifest_sha256": row["capture_residual_manifest"][
                "sha256"
            ],
            "shard_sha256": row["shard"]["sha256"],
            "raw_residual_logical_sha256": row["shard"][
                "raw_residual_logical_sha256"
            ],
            "public_j_state_logical_sha256": row["shard"][
                "public_j_state_logical_sha256"
            ],
            "generated_token_id": row["forward_canary"]["generated_token_id"],
        }
        for row in rows
    ]


def verify_capture_artifacts(
    *,
    config: Mapping[str, Any],
    capture_rows: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    capture_root: Path,
    verify_tensor_values: bool = True,
) -> dict[str, Any]:
    lexical_path_preflight((manifest_path, capture_root))
    canonical_path_preflight(input_paths=(manifest_path, capture_root), output_paths=())
    _require(manifest_path.parent.resolve() == capture_root.resolve(), "capture manifest must be in capture root")
    manifest = load_json(_regular_file(manifest_path, "capture manifest"))
    _require(isinstance(manifest, dict), "capture manifest must be an object")
    _exact_keys(
        manifest,
        (
            "schema_version",
            "kind",
            "status",
            "runtime_config_sha256",
            "capture_prompt_bundle",
            "implementation",
            "model",
            "lens",
            "runtime",
            "execution_order",
            "boundaries",
            "aggregate",
            "claim_scope",
            "reserved_validation_access_authorized",
        ),
        "capture manifest",
    )
    _require(
        manifest["schema_version"] == 1
        and manifest["kind"] == CAPTURE_KIND
        and manifest["status"] == "passed_authenticated_target_model_capture"
        and manifest["runtime_config_sha256"] == CONFIG_SHA256,
        "capture manifest identity changed",
    )
    _require(
        manifest["capture_prompt_bundle"]
        == _expected_prompt_binding(config, "stage_a_capture_prompts"),
        "capture manifest prompt binding changed",
    )
    _require(
        manifest["implementation"] == _expected_implementation(config),
        "capture implementation lineage changed",
    )
    _validate_model_output_record(manifest["model"], config, phase="capture")
    lens_expected = {
        key: config["public_j_lens"][key]
        for key in ("repo_id", "revision", "filename", "sha256", "size_bytes")
    }
    _require(manifest["lens"] == lens_expected, "capture public-J lens binding changed")
    _validate_runtime_output_record(manifest["runtime"], config)
    order = manifest["execution_order"]
    _require(isinstance(order, dict), "capture execution-order record missing")
    _exact_keys(
        order,
        (
            "reference_completed_before_capture",
            "reference_report",
            "reference_and_capture_residual_manifests_equal",
            "condition_key_read",
            "split_manifest_read",
            "stage_b_read",
            "semantic_answers_read",
            "join_conditions_read",
        ),
        "capture execution order",
    )
    _require(
        order["reference_completed_before_capture"] is True
        and order["reference_and_capture_residual_manifests_equal"] is True
        and all(
            order[key] is False
            for key in (
                "condition_key_read",
                "split_manifest_read",
                "stage_b_read",
                "semantic_answers_read",
                "join_conditions_read",
            )
        ),
        "capture chronology or forbidden-access record changed",
    )
    _verify_manifest_file_binding(capture_root, order["reference_report"], "reference report")
    boundaries = manifest["boundaries"]
    _require(
        isinstance(boundaries, list)
        and len(boundaries) == len(capture_rows) == config["capture"]["prompt_count"],
        "capture boundary count changed",
    )
    validated: list[dict[str, Any]] = []
    shard_set: list[dict[str, Any]] = []
    layers = config["capture"]["layers"]
    for index, (prompt, raw_boundary) in enumerate(
        zip(capture_rows, boundaries, strict=True)
    ):
        _require(isinstance(raw_boundary, dict), f"capture boundary {index} invalid")
        boundary = dict(raw_boundary)
        _exact_keys(
            boundary,
            (
                "index",
                "prompt_id",
                "prompt_token_ids_sha256",
                "prompt_token_count",
                "token_position",
                "reference_residual_manifest",
                "capture_residual_manifest",
                "reference_residual_manifest_equal",
                "shard",
                "forward_canary",
                "capture_valid",
            ),
            f"capture boundary {index}",
        )
        expected_token_hash = token_ids_sha256(prompt["token_ids"])
        expected_position = len(prompt["token_ids"]) - 1
        _require(
            boundary["index"] == index
            and boundary["prompt_id"] == prompt["id"]
            and boundary["prompt_token_ids_sha256"] == expected_token_hash
            and boundary["prompt_token_count"] == len(prompt["token_ids"])
            and boundary["token_position"] == expected_position
            and boundary["reference_residual_manifest_equal"] is True
            and boundary["capture_valid"] is True,
            f"capture boundary {index} does not bind the exact prompt",
        )
        reference = _validate_residual_manifest(
            boundary["reference_residual_manifest"],
            expected_position=expected_position,
            label=f"reference residual manifest {index}",
        )
        captured = _validate_residual_manifest(
            boundary["capture_residual_manifest"],
            expected_position=expected_position,
            label=f"capture residual manifest {index}",
        )
        _require(
            canonical_json_bytes(reference) == canonical_json_bytes(captured),
            f"reference/capture residual manifests differ at boundary {index}",
        )
        shard = boundary["shard"]
        _require(isinstance(shard, dict), f"capture shard record {index} invalid")
        _exact_keys(
            shard,
            (
                "path",
                "sha256",
                "size_bytes",
                "tensor_keys",
                "shape",
                "dtype",
                "raw_residual_logical_sha256",
                "public_j_state_logical_sha256",
                "reload_verified",
            ),
            f"capture shard record {index}",
        )
        expected_shard_path = f"shards/boundary-{index:06d}.safetensors"
        _require(
            shard["path"] == expected_shard_path
            and _is_sha256(shard["sha256"])
            and isinstance(shard["size_bytes"], int)
            and not isinstance(shard["size_bytes"], bool)
            and shard["size_bytes"] > 2 * 24 * 5120 * 4
            and shard["tensor_keys"] == ["public_j_state", "raw_residual"]
            and shard["shape"] == [24, 5120]
            and shard["dtype"] == "little-endian-float32"
            and _is_sha256(shard["raw_residual_logical_sha256"])
            and _is_sha256(shard["public_j_state_logical_sha256"])
            and shard["reload_verified"] is True,
            f"capture shard record {index} changed",
        )
        canary = boundary["forward_canary"]
        _require(isinstance(canary, dict), f"forward canary {index} invalid")
        _exact_keys(
            canary,
            (
                "generated_token_id",
                "final_model_top1_matches_greedy",
                "final_norm_reconstruction_within_tolerance",
                "final_logits_reconstruction_within_tolerance",
            ),
            f"forward canary {index}",
        )
        _require(
            isinstance(canary["generated_token_id"], int)
            and not isinstance(canary["generated_token_id"], bool)
            and 0 <= canary["generated_token_id"] < config["model"]["vocabulary_size"]
            and all(value is True for key, value in canary.items() if key != "generated_token_id"),
            f"forward canary {index} failed",
        )
        shard_path = _manifest_file_path(capture_root, shard["path"], f"capture shard {index}")
        _require(shard_path.exists(), f"capture shard is missing: {shard_path}")
        if verify_tensor_values:
            _verify_safetensors_shard(shard_path, shard, layers=layers)
        else:
            resolved = _regular_file(shard_path, f"capture shard {index}")
            _require(
                resolved.stat().st_size == shard["size_bytes"]
                and sha256_file(resolved) == shard["sha256"],
                f"capture shard bytes changed: {shard_path}",
            )
        shard_set.append(
            {"path": shard["path"], "sha256": shard["sha256"], "size_bytes": shard["size_bytes"]}
        )
        validated.append(boundary)
    aggregate = manifest["aggregate"]
    _require(isinstance(aggregate, dict), "capture aggregate missing")
    _exact_keys(
        aggregate,
        (
            "boundary_count",
            "all_capture_valid",
            "all_shards_reload_verified",
            "all_forward_canaries_passed",
            "boundary_contract_sha256",
            "shard_set_sha256",
        ),
        "capture aggregate",
    )
    boundary_contract_sha256 = sha256_bytes(
        canonical_json_bytes(_capture_boundary_contract(validated))
    )
    shard_set_sha256 = sha256_bytes(canonical_json_bytes(shard_set))
    _require(
        aggregate["boundary_count"] == 500
        and aggregate["all_capture_valid"] is True
        and aggregate["all_shards_reload_verified"] is True
        and aggregate["all_forward_canaries_passed"] is True
        and aggregate["boundary_contract_sha256"] == boundary_contract_sha256
        and aggregate["shard_set_sha256"] == shard_set_sha256,
        "capture aggregate lineage changed",
    )
    _require(manifest["claim_scope"] == OUTPUT_FALSE_CLAIMS, "capture claim scope changed")
    _require(
        manifest["reserved_validation_access_authorized"] is False,
        "capture reserved-validation authorization changed",
    )
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "boundary_count": len(validated),
        "boundary_contract_sha256": boundary_contract_sha256,
        "shard_set_sha256": shard_set_sha256,
        "all_tensor_values_verified": verify_tensor_values,
        "schema_only_bundle_rejected_without_shards": True,
    }


def _generation_record_contract(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": row["index"],
            "prompt_id": row["prompt_id"],
            "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
            "generated_token_ids_sha256": row["generated_token_ids_sha256"],
            "completion_text_sha256": row["completion_text_sha256"],
            "completion_status": row["completion_status"],
            "finish_reason": row["finish_reason"],
        }
        for row in rows
    ]


def _load_pinned_tokenizer(snapshot_path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise StageARuntimeError("transformers is required for generation verification") from error
    try:
        return AutoTokenizer.from_pretrained(
            snapshot_path,
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception as error:
        raise StageARuntimeError(f"cannot load pinned tokenizer: {error}") from error


def verify_generation_artifacts(
    *,
    config: Mapping[str, Any],
    generation_rows: Sequence[Mapping[str, Any]],
    capture_verification: Mapping[str, Any],
    capture_manifest_path: Path,
    generation_manifest_path: Path,
    completion_bundle_path: Path,
    output_root: Path,
    snapshot_path: Path,
) -> dict[str, Any]:
    lexical_path_preflight(
        (capture_manifest_path, generation_manifest_path, completion_bundle_path, output_root)
    )
    canonical_path_preflight(
        input_paths=(
            capture_manifest_path,
            generation_manifest_path,
            completion_bundle_path,
            output_root,
        ),
        output_paths=(),
    )
    manifest = load_json(_regular_file(generation_manifest_path, "generation manifest"))
    surface = load_json(_regular_file(completion_bundle_path, "visible completion bundle"))
    _require(isinstance(manifest, dict), "generation manifest must be an object")
    _exact_keys(
        manifest,
        (
            "schema_version",
            "kind",
            "status",
            "runtime_config_sha256",
            "generation_prompt_bundle",
            "authenticated_capture_manifest",
            "surface_completion_bundle",
            "model",
            "runtime",
            "sampling",
            "execution_order",
            "records",
            "aggregate",
            "claim_scope",
            "reserved_validation_access_authorized",
        ),
        "generation manifest",
    )
    _require(
        manifest["schema_version"] == 1
        and manifest["kind"] == GENERATION_KIND
        and manifest["status"] == "passed_generation_after_authenticated_capture"
        and manifest["runtime_config_sha256"] == CONFIG_SHA256,
        "generation manifest identity changed",
    )
    _require(
        manifest["generation_prompt_bundle"]
        == _expected_prompt_binding(config, "stage_a_generation_prompts"),
        "generation prompt binding changed",
    )
    capture_binding = manifest["authenticated_capture_manifest"]
    _require(isinstance(capture_binding, dict), "capture manifest binding missing")
    _exact_keys(capture_binding, ("path", "sha256", "size_bytes"), "capture manifest binding")
    _require(
        capture_binding["sha256"] == capture_verification["manifest_sha256"]
        and capture_binding["size_bytes"] == capture_verification["manifest_size_bytes"]
        and capture_manifest_path.resolve()
        == _manifest_file_path(output_root, capture_binding["path"], "capture manifest binding").resolve(),
        "generation does not bind the verified capture manifest",
    )
    surface_binding = manifest["surface_completion_bundle"]
    surface_bound_path, _ = _verify_manifest_file_binding(
        output_root, surface_binding, "surface completion bundle"
    )
    _require(
        surface_bound_path.resolve() == completion_bundle_path.resolve(),
        "generation manifest binds a different completion bundle",
    )
    _validate_model_output_record(manifest["model"], config, phase="generation")
    _validate_runtime_output_record(manifest["runtime"], config)
    expected_sampling = {
        **config["generation"]["sampling"],
        "no_input_truncation": True,
    }
    _require(manifest["sampling"] == expected_sampling, "generation sampling changed")
    order = manifest["execution_order"]
    _require(isinstance(order, dict), "generation execution order missing")
    _exact_keys(
        order,
        (
            "capture_verified_before_model_load",
            "capture_shards_rehashed_before_generation",
            "condition_key_read",
            "split_manifest_read",
            "stage_b_read",
            "semantic_answers_read",
            "join_conditions_read",
            "completion_annotation_run",
        ),
        "generation execution order",
    )
    _require(
        order["capture_verified_before_model_load"] is True
        and order["capture_shards_rehashed_before_generation"] is True
        and all(
            order[key] is False
            for key in (
                "condition_key_read",
                "split_manifest_read",
                "stage_b_read",
                "semantic_answers_read",
                "join_conditions_read",
                "completion_annotation_run",
            )
        ),
        "generation chronology or forbidden-access record changed",
    )
    records = manifest["records"]
    _require(
        isinstance(records, list)
        and len(records) == len(generation_rows) == config["generation"]["prompt_count"],
        "generation record count changed",
    )
    tokenizer = _load_pinned_tokenizer(snapshot_path)
    validated: list[dict[str, Any]] = []
    for index, (prompt, raw_record) in enumerate(
        zip(generation_rows, records, strict=True)
    ):
        _require(isinstance(raw_record, dict), f"generation record {index} invalid")
        record = dict(raw_record)
        _exact_keys(
            record,
            (
                "index",
                "prompt_id",
                "prompt_token_ids_sha256",
                "prompt_token_count",
                "generated_token_ids",
                "generated_token_ids_sha256",
                "generated_token_count",
                "completion_text",
                "completion_text_sha256",
                "completion_status",
                "finish_reason",
                "stop_reason",
                "vllm_prompt_tokens_equal_input",
                "completion_text_equals_pinned_tokenizer_visible_decode",
            ),
            f"generation record {index}",
        )
        generated = record["generated_token_ids"]
        _require(
            record["index"] == index
            and record["prompt_id"] == prompt["id"]
            and record["prompt_token_ids_sha256"] == token_ids_sha256(prompt["token_ids"])
            and record["prompt_token_count"] == len(prompt["token_ids"])
            and isinstance(generated, list)
            and len(generated) <= config["generation"]["sampling"]["max_new_tokens"]
            and all(
                isinstance(token_id, int)
                and not isinstance(token_id, bool)
                and 0 <= token_id < config["model"]["vocabulary_size"]
                for token_id in generated
            )
            and record["generated_token_ids_sha256"] == token_ids_sha256(generated)
            and record["generated_token_count"] == len(generated)
            and isinstance(record["completion_text"], str)
            and record["completion_text_sha256"]
            == sha256_bytes(record["completion_text"].encode("utf-8"))
            and record["vllm_prompt_tokens_equal_input"] is True
            and record["completion_text_equals_pinned_tokenizer_visible_decode"] is True,
            f"generation record {index} token or prompt lineage changed",
        )
        visible_decode = tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        _require(
            visible_decode == record["completion_text"],
            f"generation record {index} fails pinned tokenizer replay",
        )
        status = record["completion_status"]
        finish = record["finish_reason"]
        _require(
            status in {"complete", "truncated", "empty"}
            and finish in {"stop", "length"},
            f"generation record {index} completion status changed",
        )
        if status == "complete":
            _require(bool(record["completion_text"]) and finish == "stop", f"complete record {index} invalid")
        elif status == "truncated":
            _require(
                bool(record["completion_text"])
                and finish == "length"
                and len(generated) == config["generation"]["sampling"]["max_new_tokens"],
                f"truncated record {index} invalid",
            )
        else:
            _require(record["completion_text"] == "" and finish == "stop", f"empty record {index} invalid")
        validated.append(record)
    aggregate = manifest["aggregate"]
    _require(isinstance(aggregate, dict), "generation aggregate missing")
    _exact_keys(
        aggregate,
        ("record_count", "generation_error_count", "record_contract_sha256"),
        "generation aggregate",
    )
    record_contract_sha256 = sha256_bytes(
        canonical_json_bytes(_generation_record_contract(validated))
    )
    _require(
        aggregate["record_count"] == 240
        and aggregate["generation_error_count"] == 0
        and aggregate["record_contract_sha256"] == record_contract_sha256,
        "generation aggregate lineage changed",
    )
    _require(manifest["claim_scope"] == OUTPUT_FALSE_CLAIMS, "generation claim scope changed")
    _require(
        manifest["reserved_validation_access_authorized"] is False,
        "generation reserved-validation authorization changed",
    )
    _require(isinstance(surface, dict), "surface completion bundle must be an object")
    _exact_keys(
        surface,
        (
            "schema_version",
            "kind",
            "status",
            "generation_prompt_bundle_sha256",
            "surface_policy",
            "records",
            "stage_b_records_present",
            "reserved_validation_access_authorized",
        ),
        "surface completion bundle",
    )
    _require(
        surface["schema_version"] == 1
        and surface["kind"] == COMPLETION_KIND
        and surface["status"] == "stage_a_generation_complete_condition_key_not_joined"
        and surface["generation_prompt_bundle_sha256"]
        == config["frozen_inputs"]["stage_a_generation_prompts"]["sha256"]
        and surface["surface_policy"]
        == "exact_experiment_completion_string_only_no_prompt_prefix_no_external_task_outcome"
        and surface["stage_b_records_present"] is False
        and surface["reserved_validation_access_authorized"] is False,
        "surface completion bundle identity changed",
    )
    expected_surface_records = [
        {
            "prompt_id": row["prompt_id"],
            "completion_status": row["completion_status"],
            "completion_text": row["completion_text"],
        }
        for row in validated
    ]
    _require(
        surface["records"] == expected_surface_records,
        "surface completion records differ from authenticated generation",
    )
    return {
        "generation_manifest_sha256": sha256_file(generation_manifest_path),
        "generation_manifest_size_bytes": generation_manifest_path.stat().st_size,
        "surface_completion_bundle_sha256": sha256_file(completion_bundle_path),
        "surface_completion_bundle_size_bytes": completion_bundle_path.stat().st_size,
        "record_count": len(validated),
        "record_contract_sha256": record_contract_sha256,
        "pinned_tokenizer_replay_passed": True,
        "verified_capture_manifest_sha256": capture_verification["manifest_sha256"],
        "schema_only_bundle_rejected_without_capture_shards_or_token_replay": True,
    }


def _prepare_common_verification(
    config_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    Path,
]:
    config = load_and_validate_config(config_path)
    source_paths = _validate_source_bindings(config)
    _validate_schema_bindings(config)
    snapshot_path, _inventory = resolve_and_verify_model(config)
    resolve_and_verify_lens(config)
    capture_rows, generation_rows, _summary = load_and_validate_prompt_bundles(
        config, source_paths
    )
    return config, capture_rows, generation_rows, snapshot_path


def run_verify_capture(args: argparse.Namespace) -> int:
    lexical_path_preflight((args.config, args.capture_manifest, args.capture_root, args.output))
    config, capture_rows, _generation_rows, _snapshot = _prepare_common_verification(
        args.config
    )
    verification = verify_capture_artifacts(
        config=config,
        capture_rows=capture_rows,
        manifest_path=args.capture_manifest,
        capture_root=args.capture_root,
        verify_tensor_values=True,
    )
    receipt = {
        "schema_version": 1,
        "kind": CAPTURE_VERIFICATION_KIND,
        "status": "passed_complete_cpu_capture_artifact_verification",
        "runtime_config_sha256": CONFIG_SHA256,
        "capture": verification,
        "forbidden_access": {
            "condition_key_read": False,
            "split_manifest_read": False,
            "stage_b_read": False,
            "semantic_answers_read": False,
            "join_conditions_read": False,
            "reserved_validation_access_authorized": False,
        },
        "claim_scope": OUTPUT_FALSE_CLAIMS,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(args.output, receipt)
    print(f"verified 500 Stage-A capture shards into {args.output}", file=sys.stderr)
    return 0


def run_verify_generation(args: argparse.Namespace) -> int:
    lexical_path_preflight(
        (
            args.config,
            args.capture_manifest,
            args.capture_root,
            args.generation_manifest,
            args.completion_bundle,
            args.output_root,
            args.output,
        )
    )
    config, capture_rows, generation_rows, snapshot_path = _prepare_common_verification(
        args.config
    )
    capture_verification = verify_capture_artifacts(
        config=config,
        capture_rows=capture_rows,
        manifest_path=args.capture_manifest,
        capture_root=args.capture_root,
        verify_tensor_values=True,
    )
    generation_verification = verify_generation_artifacts(
        config=config,
        generation_rows=generation_rows,
        capture_verification=capture_verification,
        capture_manifest_path=args.capture_manifest,
        generation_manifest_path=args.generation_manifest,
        completion_bundle_path=args.completion_bundle,
        output_root=args.output_root,
        snapshot_path=snapshot_path,
    )
    receipt = {
        "schema_version": 1,
        "kind": GENERATION_VERIFICATION_KIND,
        "status": "passed_complete_cpu_generation_lineage_verification",
        "runtime_config_sha256": CONFIG_SHA256,
        "capture": capture_verification,
        "generation": generation_verification,
        "forbidden_access": {
            "condition_key_read": False,
            "split_manifest_read": False,
            "stage_b_read": False,
            "semantic_answers_read": False,
            "join_conditions_read": False,
            "reserved_validation_access_authorized": False,
        },
        "claim_scope": OUTPUT_FALSE_CLAIMS,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(args.output, receipt)
    print(f"verified 240 Stage-A completion lineages into {args.output}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight", help="CPU-only hash, prompt, snapshot, capacity, and environment preflight"
    )
    preflight.add_argument("--config", type=Path, default=CONFIG_PATH)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--require-runtime-environment", action="store_true")
    preflight.set_defaults(handler=run_preflight)

    verify_capture = subparsers.add_parser(
        "verify-capture", help="CPU-verify all 500 authenticated capture shards"
    )
    verify_capture.add_argument("--config", type=Path, default=CONFIG_PATH)
    verify_capture.add_argument("--capture-manifest", type=Path, required=True)
    verify_capture.add_argument("--capture-root", type=Path, required=True)
    verify_capture.add_argument("--output", type=Path, required=True)
    verify_capture.set_defaults(handler=run_verify_capture)

    verify_generation = subparsers.add_parser(
        "verify-generation",
        help="CPU-verify capture lineage, 240 generated-token records, and surface bundle",
    )
    verify_generation.add_argument("--config", type=Path, default=CONFIG_PATH)
    verify_generation.add_argument("--capture-manifest", type=Path, required=True)
    verify_generation.add_argument("--capture-root", type=Path, required=True)
    verify_generation.add_argument("--generation-manifest", type=Path, required=True)
    verify_generation.add_argument("--completion-bundle", type=Path, required=True)
    verify_generation.add_argument("--output-root", type=Path, required=True)
    verify_generation.add_argument("--output", type=Path, required=True)
    verify_generation.set_defaults(handler=run_verify_generation)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
