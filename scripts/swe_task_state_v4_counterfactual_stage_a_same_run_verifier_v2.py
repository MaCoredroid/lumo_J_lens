#!/usr/bin/env python3
"""CPU-only preflight and verifier for Stage-A V2 same-run artifacts.

There is deliberately no model import, model load, GPU command, or producer
subcommand here.  The future producer must already have run under the frozen
external filesystem tracer.  This verifier authenticates the two immutable
pre-generation receipts, parses the *content* of the reference report, reloads
every tensor shard, checks exact prompt echo and first-token parity, checks the
condition-blind completion surface, and rejects mock or schema-only output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_counterfactual_stage_a_same_run_runtime_v2.json"
CONFIG_SHA256 = "67217b2086124a5430cdd02e3fc2ebcf61f470d95cbba7ef90f18e104eccac0f"
CONFIG_CANONICAL_SHA256 = "b2d5ae2c27bc18614f99450da73153c57938a7ebee8b7e6f96649de23ad11283"
PRODUCER_PATH = ROOT / "scripts" / "swe_task_state_v4_counterfactual_stage_a_same_run_producer_v2.py"
PRODUCER_SHA256 = "8ca712959a36945f6cc6aee6d42a5cbd7a1d22e3dcaafb08fe112438985660fb"
SCRIPT_PATH = Path(__file__).resolve()
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_FRAGMENTS = (
    "reserved",
    "validation",
    "split-manifest",
    "condition-key",
    "stage-b",
    "semantic-answer",
    "expectation",
    "task-outcome",
    "visible-state-annotation",
)
FALSE_CLAIMS = {
    "private_chain_of_thought_reconstructed": False,
    "cot_or_cot_like_decoding_established": False,
    "latent_cot_like_concept_trajectory_established": False,
    "subjective_confidence_inferred": False,
    "subjective_doubt_inferred": False,
    "experienced_stress_inferred": False,
    "experienced_emotion_inferred": False,
    "causal_affect_or_state_effect_established": False,
    "incremental_activation_readout_established": False,
    "outer_or_reserved_validation_generalization_established": False,
}
# Current verifier intentionally cannot emit a real receipt.  Config mutation
# is insufficient; future real execution requires a new versioned code freeze.
REAL_FINAL_VERIFICATION_CODE_FREEZE_COMPLETE = False


class SameRunVerificationError(ValueError):
    """Raised when prospective or produced lineage is not exact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SameRunVerificationError(message)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SameRunVerificationError(f"cannot load strict JSON {path}: {error}") from error


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
        raise SameRunVerificationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    return sha256_bytes(canonical_json_bytes(list(token_ids)))


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _exact_keys(value: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    _require(set(value) == set(keys), f"{label} keys changed")


def _path_forbidden(path: Path) -> bool:
    normalized = Path(os.path.normpath(os.path.abspath(os.fspath(path))))
    return any(fragment in part.lower() for part in normalized.parts for fragment in FORBIDDEN_FRAGMENTS)


def lexical_path_preflight(paths: Iterable[Path | None]) -> None:
    """Reject forbidden lexical paths before stat, resolve, read, or write."""

    for path in paths:
        if path is not None and _path_forbidden(Path(path)):
            raise SameRunVerificationError(
                f"forbidden path rejected before filesystem access: {path}"
            )


def _regular_file(path: Path, label: str) -> Path:
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SameRunVerificationError(f"{label} unavailable: {error}") from error
    _require(resolved.is_file(), f"{label} is not a regular file")
    return resolved


def _binding(path: Path, *, display_root: Path | None = None) -> dict[str, Any]:
    resolved = _regular_file(path, "bound file")
    display = str(resolved)
    if display_root is not None:
        try:
            display = str(resolved.relative_to(display_root.resolve()))
        except ValueError:
            pass
    return {"path": display, "sha256": sha256_file(resolved), "size_bytes": resolved.stat().st_size}


def _verify_binding(record: Mapping[str, Any], *, label: str, root: Path = ROOT) -> Path:
    _require(isinstance(record, dict), f"{label} binding missing")
    _require({"path", "sha256", "size_bytes"}.issubset(record), f"{label} binding incomplete")
    _require(
        isinstance(record["path"], str)
        and _is_sha(record["sha256"])
        and isinstance(record["size_bytes"], int)
        and not isinstance(record["size_bytes"], bool)
        and record["size_bytes"] > 0,
        f"{label} binding invalid",
    )
    raw = Path(record["path"])
    path = raw if raw.is_absolute() else root / raw
    lexical_path_preflight((path,))
    resolved = _regular_file(path, label)
    _require(
        resolved.stat().st_size == record["size_bytes"]
        and sha256_file(resolved) == record["sha256"],
        f"{label} bytes changed",
    )
    return resolved


def _atomic_json(path: Path, value: Any) -> None:
    lexical_path_preflight((path,))
    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _require(not temporary.exists(), f"temporary output exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def validate_config(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "V2 config must be an object")
    _require(sha256_file(CONFIG_PATH) == CONFIG_SHA256, "V2 config byte hash changed")
    _require(
        sha256_bytes(canonical_json_bytes(value)) == CONFIG_CANONICAL_SHA256,
        "V2 config object changed",
    )
    config = dict(value)
    _require(
        config["schema_version"] == 2
        and config["id"] == "swe-task-state-v4-counterfactual-selector-control-stage-a-same-run-v2"
        and config["status"] == "prospective_cpu_contract_only_real_backend_absent_no_gpu_or_model_run",
        "V2 config identity changed",
    )
    _require(
        config["authorization"] == {
            "cpu_preflight_and_mock_tests_authorized": True,
            "real_gpu_producer_authorized_by_this_config": False,
            "real_backend_implementation_present": False,
            "real_traced_launcher_present": False,
            "raw_strace_normalizer_present": False,
            "raw_vs_normalized_trace_equivalence_established": False,
            "filesystem_alias_closure_established": False,
            "real_final_verification_receipt_emission_authorized": False,
            "current_producer_code_freeze_real_mode_enabled": False,
            "current_verifier_code_freeze_real_receipt_enabled": False,
            "stage_b_materialization_or_runtime_authorized": False,
            "condition_key_or_split_join_authorized": False,
            "reserved_validation_access_authorized": False,
        },
        "V2 authorization changed",
    )
    _require(config["generation_prompt_count"] == 240, "generation count changed")
    _require(config["implementation"]["producer"]["sha256"] == PRODUCER_SHA256, "producer pin changed")
    _require(sha256_file(PRODUCER_PATH) == PRODUCER_SHA256, "producer source changed")
    _require(
        config["runtime"]["max_model_len"] == 49152
        and config["runtime"]["max_num_seqs"] == 1
        and config["runtime"]["enable_prefix_caching"] is False
        and config["runtime"]["request_cache_reuse_allowed"] is False
        and config["runtime"]["seed"] == 0,
        "runtime/cache/seed contract changed",
    )
    sampling = config["sampling"]
    _require(
        sampling["max_new_tokens"] == sampling["min_new_tokens"] == 256
        and sampling["temperature"] == 0
        and sampling["seed"] == 0
        and sampling["ignore_eos"] is True
        and sampling["stop"] == []
        and sampling["stop_token_ids"] == []
        and sampling["truncate_prompt_tokens"] is None,
        "exact-256 sampler changed",
    )
    _require(
        config["capture"]["layers"] == list(range(24, 48))
        and config["capture"]["replay_between_capture_and_first_decode"] is False
        and config["capture"]["generation_must_consume_the_same_kv_and_mamba_cache"] is True,
        "same-run capture contract changed",
    )
    claims = config["claim_scope"]
    _require(claims["cpu_contract_and_mock_tests_implemented"] is True, "CPU claim changed")
    _require(
        all(value is False for key, value in claims.items() if key != "cpu_contract_and_mock_tests_implemented"),
        "an unearned claim became true",
    )
    _require(
        config["boundary_limits"]["captured_causal_boundaries_per_generation"] == 1
        and config["boundary_limits"]["one_boundary_can_establish_temporal_cot_trajectory"] is False
        and config["boundary_limits"]["one_boundary_can_establish_private_or_verbatim_cot"] is False,
        "one-boundary claim limit changed",
    )
    _require(
        config["implementation"]["production_backend"] is None
        and config["implementation"]["real_traced_launcher"] is None
        and config["implementation"]["raw_strace_normalizer"] is None
        and config["blocking_state"]["current_config_can_authorize_real_production_mode"] is False
        and config["blocking_state"]["current_verifier_can_emit_a_real_final_verification_receipt"] is False
        and config["filesystem_trace"]["raw_vs_normalized_trace_equivalence_verifier_present"] is False
        and config["filesystem_trace"]["hardlink_symlink_alias_closure_present"] is False
        and config["filesystem_trace"]["normalized_trace_alone_is_gate_sufficient"] is False
        and config["blocking_state"]["current_producer_code_freeze_has_false_real_mode_sentinel"] is True
        and config["blocking_state"]["current_verifier_code_freeze_has_false_real_receipt_sentinel"] is True
        and REAL_FINAL_VERIFICATION_CODE_FREEZE_COMPLETE is False,
        "current real-runtime blocker state changed",
    )
    return config


def _require_real_verification_authorization(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reject this frozen config before any producer/output artifact access."""

    authorization = config.get("authorization")
    _require(isinstance(authorization, Mapping), "real verification authorization is absent")
    required_true = (
        "real_gpu_producer_authorized_by_this_config",
        "real_backend_implementation_present",
        "real_traced_launcher_present",
        "raw_strace_normalizer_present",
        "raw_vs_normalized_trace_equivalence_established",
        "filesystem_alias_closure_established",
        "real_final_verification_receipt_emission_authorized",
    )
    _require(
        all(authorization.get(key) is True for key in required_true),
        "real same-run verification is not authorized by this frozen config",
    )
    # Check the immutable source sentinel before implementation lookup or any
    # binding/path operation.  A mutated config cannot turn this verifier into
    # an arbitrary-file oracle.
    _require(
        REAL_FINAL_VERIFICATION_CODE_FREEZE_COMPLETE is True,
        "real verification requires a new versioned verifier code freeze",
    )
    implementation = config.get("implementation")
    _require(isinstance(implementation, Mapping), "implementation contract is absent")
    resolved: dict[str, Any] = {}
    for key, label in (
        ("production_backend", "production backend source"),
        ("real_traced_launcher", "real traced launcher source"),
        ("raw_strace_normalizer", "raw strace normalizer source"),
    ):
        binding = implementation.get(key)
        _require(
            isinstance(binding, dict)
            and set(binding) == {"path", "sha256", "size_bytes"},
            f"{label} binding is absent or not exact",
        )
        _verify_binding(binding, label=label)
        resolved[key] = dict(binding)
    return resolved


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    lexical_path_preflight((path,))
    _require(path.resolve(strict=True) == CONFIG_PATH.resolve(strict=True), "only the pinned V2 config is allowed")
    return validate_config(load_json(path))


def verify_frozen_bindings(config: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for group in ("frozen_inputs", "schemas"):
        for name, record in config[group].items():
            paths[name] = _verify_binding(record, label=f"{group}.{name}")
    paths["producer"] = _verify_binding(config["implementation"]["producer"], label="producer")
    return paths


def validate_prompt_rows(
    value: Any, *, spec: Mapping[str, Any], vocabulary_size: int, label: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(isinstance(value, list) and len(value) == spec["record_count"], f"{label} count changed")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        _require(isinstance(raw, dict) and set(raw) == set(spec["allowed_row_keys"]), f"{label} row {index} fields changed")
        prompt_id, token_ids = raw["id"], raw["token_ids"]
        _require(isinstance(prompt_id, str) and SHA256_RE.fullmatch(prompt_id) and prompt_id not in seen, f"{label} row {index} ID changed")
        _require(
            isinstance(token_ids, list)
            and bool(token_ids)
            and all(isinstance(item, int) and not isinstance(item, bool) and 0 <= item < vocabulary_size for item in token_ids),
            f"{label} row {index} token IDs changed",
        )
        seen.add(prompt_id)
        rows.append({"id": prompt_id, "token_ids": list(token_ids)})
    contract = [
        {"id": row["id"], "token_ids_sha256": token_ids_sha256(row["token_ids"]), "token_count": len(row["token_ids"])}
        for row in rows
    ]
    id_hash = sha256_bytes(canonical_json_bytes([row["id"] for row in rows]))
    contract_hash = sha256_bytes(canonical_json_bytes(contract))
    _require(id_hash == spec["ordered_id_list_sha256"] and contract_hash == spec["row_contract_sha256"], f"{label} order or content changed")
    if "minimum_token_count" in spec:
        lengths = [len(row["token_ids"]) for row in rows]
        _require(min(lengths) == spec["minimum_token_count"] and max(lengths) == spec["maximum_token_count"], f"{label} lengths changed")
    return rows, {"record_count": len(rows), "ordered_id_list_sha256": id_hash, "row_contract_sha256": contract_hash}


def load_production_prompt_rows(config: Mapping[str, Any], paths: Mapping[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    capture, _ = validate_prompt_rows(
        load_json(paths["capture_prompts_for_prerequisite_verification"]),
        spec=config["frozen_inputs"]["capture_prompts_for_prerequisite_verification"],
        vocabulary_size=config["model"]["vocabulary_size"],
        label="capture prompts",
    )
    generation, _ = validate_prompt_rows(
        load_json(paths["generation_prompts"]),
        spec=config["frozen_inputs"]["generation_prompts"],
        vocabulary_size=config["model"]["vocabulary_size"],
        label="generation prompts",
    )
    capture_by_id = {row["id"]: row["token_ids"] for row in capture}
    _require(all(capture_by_id.get(row["id"]) == row["token_ids"] for row in generation), "generation rows are not exact capture rows")
    _require(max(len(row["token_ids"]) for row in generation) + 256 <= config["runtime"]["max_model_len"], "generation would truncate")
    return capture, generation


def expected_model_and_lens_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
    hf_home = Path(config["environment"]["exact_values"]["HF_HOME"])
    model = hf_home / config["model"]["snapshot_relative_to_hf_home"]
    lens_spec = config["public_j_lens"]
    lens = (
        hf_home / "hub" / ("models--" + lens_spec["repo_id"].replace("/", "--"))
        / "snapshots" / lens_spec["revision"] / lens_spec["filename"]
    )
    lexical_path_preflight((model, lens))
    return model, lens


def verify_model_and_lens_bytes(
    config: Mapping[str, Any], *, model_snapshot: Path, lens_path: Path
) -> dict[str, Any]:
    """Rehash the complete pinned model tree and the complete public-J file."""

    expected_model, expected_lens = expected_model_and_lens_paths(config)
    _require(
        model_snapshot.resolve(strict=True) == expected_model.resolve(strict=True)
        and lens_path.resolve(strict=True) == expected_lens.resolve(strict=True),
        "model or lens path differs from the pinned local snapshots",
    )
    entries: list[dict[str, Any]] = []
    for path in sorted(model_snapshot.rglob("*"), key=lambda item: item.relative_to(model_snapshot).as_posix()):
        if path.is_dir():
            continue
        resolved = path.resolve(strict=True)
        _require(resolved.is_file(), f"model snapshot entry is not a file: {path}")
        entries.append(
            {
                "path": path.relative_to(model_snapshot).as_posix(),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    model = config["model"]
    _require(
        len(entries) == model["snapshot_file_count"]
        and sum(item["size_bytes"] for item in entries) == model["snapshot_size_bytes"]
        and sha256_bytes(canonical_json_bytes(entries)) == model["snapshot_tree_sha256"],
        "complete model snapshot inventory changed",
    )
    by_name = {item["path"]: item for item in entries}
    _require(
        by_name["config.json"]["sha256"] == model["config_sha256"]
        and by_name["model.safetensors.index.json"]["sha256"] == model["model_index_sha256"]
        and by_name["tokenizer.json"]["sha256"] == config["tokenizer"]["tokenizer_json_sha256"]
        and by_name["tokenizer_config.json"]["sha256"] == config["tokenizer"]["tokenizer_config_sha256"]
        and by_name["generation_config.json"]["sha256"] == config["tokenizer"]["generation_config_sha256"],
        "model/tokenizer metadata bytes changed",
    )
    for name, expected in model["checkpoint_shards"].items():
        _require(
            by_name[name]["sha256"] == expected["sha256"]
            and by_name[name]["size_bytes"] == expected["size_bytes"],
            f"checkpoint shard changed: {name}",
        )
    lens = _regular_file(lens_path, "public-J lens")
    lens_spec = config["public_j_lens"]
    _require(
        lens.stat().st_size == lens_spec["size_bytes"]
        and sha256_file(lens) == lens_spec["sha256"],
        "public-J lens bytes changed",
    )
    return {
        "model_snapshot_path": str(model_snapshot.resolve()),
        "model_snapshot_tree_sha256": model["snapshot_tree_sha256"],
        "model_snapshot_file_count": len(entries),
        "model_snapshot_size_bytes": model["snapshot_size_bytes"],
        "public_j_lens": _binding(lens_path),
        "all_bytes_rehashed": True,
    }


def _validate_residual_manifest(value: Any, *, position: int, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} missing")
    _exact_keys(value, ("algorithm", "logical_bytes", "sha256", "tensor_count", "token_positions"), label)
    _require(
        value["algorithm"] == "SHA-256 over length-prefixed canonical layer/shape/dtype/token-position/byte-count headers and logical row-major FP32 bytes"
        and value["logical_bytes"] == 64 * 5120 * 4
        and _is_sha(value["sha256"])
        and value["tensor_count"] == 64
        and value["token_positions"] == [position],
        f"{label} content changed",
    )
    return dict(value)


def _validate_reference_metadata(report: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    expected_top = {
        "schema_version", "score_encoding", "status", "started_at", "completed_at",
        "elapsed_seconds", "host", "model", "lens", "runtime", "scored_vocabulary",
        "assertions", "experiments",
    }
    _exact_keys(report, expected_top, "reference report")
    auth = config["reference_report_authentication"]
    _require(
        report["schema_version"] == auth["expected_schema_version"]
        and report["status"] == auth["expected_status"]
        and report["score_encoding"] == auth["expected_score_encoding"],
        "reference report schema/status/score encoding changed",
    )
    model = report["model"]
    spec = config["model"]
    _require(isinstance(model, dict), "reference model record missing")
    _require(
        model.get("repo_id") == spec["repo_id"]
        and model.get("revision") == spec["revision"]
        and model.get("config_sha256") == spec["config_sha256"]
        and model.get("index_sha256") == spec["model_index_sha256"]
        and model.get("quant_method") == "modelopt"
        and model.get("quant_algo") == "MIXED_PRECISION",
        "reference model identity changed",
    )
    info = model.get("model_info", {})
    _require(
        info.get("hidden_size") == 5120
        and info.get("layer_count") == 64
        and info.get("language_model_class") == "Qwen3_5ForCausalLM",
        "reference model geometry changed",
    )
    checkpoint = model.get("checkpoint_integrity", {})
    _require(
        checkpoint.get("policy") == "ModelOptCheckpoint(strict_pinned=True)"
        and checkpoint.get("validated_before_model_load") is True
        and checkpoint.get("validated_after_evaluation") is True,
        "reference checkpoint chronology changed",
    )
    observed_shards = checkpoint.get("shards", {})
    expected_shards = {
        name: {"bytes": item["size_bytes"], "sha256": item["sha256"]}
        for name, item in spec["checkpoint_shards"].items()
    }
    _require(observed_shards == expected_shards, "reference checkpoint shard integrity changed")
    lens = report["lens"]
    lens_spec = config["public_j_lens"]
    _require(
        isinstance(lens, dict)
        and lens.get("repo_id") == lens_spec["repo_id"]
        and lens.get("revision") == lens_spec["revision"]
        and lens.get("filename") == lens_spec["filename"]
        and lens.get("sha256") == lens_spec["sha256"],
        "reference public-J identity changed",
    )
    runtime = report["runtime"]
    expected_runtime = config["reference_report_authentication"]["reference_runtime_exact"]
    _require(
        isinstance(runtime, dict)
        and runtime.get("max_model_len") == expected_runtime["max_model_len"]
        and runtime.get("max_num_batched_tokens") == expected_runtime["max_num_batched_tokens"]
        and runtime.get("mamba_block_size") == expected_runtime["mamba_block_size"]
        and runtime.get("enable_prefix_caching") is expected_runtime["enable_prefix_caching"]
        and runtime.get("kv_cache_dtype") == expected_runtime["kv_cache_dtype"]
        and runtime.get("stream_final_only") is True
        and runtime.get("enforce_eager") is True
        and runtime.get("language_model_only") is True
        and runtime.get("capture_adapter") == "vLLM apply_model forward hooks",
        "reference runtime/cache record changed",
    )
    assertions = report["assertions"]
    _require(
        assertions == {
            "lens_hash_matches": True,
            "lens_metadata_matches": True,
            "model_architecture_matches": True,
            "all_final_layer_top1_match_greedy": True,
            "all_final_adapter_reconstructions_within_tolerance": True,
        },
        "reference forward assertions failed",
    )


def authenticate_reference_report_content(
    report: Any,
    *,
    capture_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate report semantics and every prompt/canary, never hash only."""

    _require(isinstance(report, dict), "reference report must be an object")
    _validate_reference_metadata(report, config)
    experiments = report["experiments"]
    _require(
        isinstance(experiments, list)
        and len(experiments) == len(capture_rows)
        and len(experiments) == config["reference_report_authentication"]["expected_experiment_count"],
        "reference experiment count changed",
    )
    contracts: list[dict[str, Any]] = []
    for index, (experiment, prompt) in enumerate(zip(experiments, capture_rows, strict=True)):
        _require(isinstance(experiment, dict), f"reference experiment {index} invalid")
        required = {
            "id", "prompt_token_ids", "positions_requested", "positions_resolved",
            "capture_positions_resolved", "final_validation_position", "generated_token_id",
            "final_layer_top1_matches_greedy", "final_model_readout",
            "captured_final_model_readout", "final_norm_reconstruction",
            "final_logits_reconstruction", "residual_capture_manifest",
        }
        _require(required.issubset(experiment), f"reference experiment {index} content incomplete")
        position = len(prompt["token_ids"]) - 1
        generated = experiment["generated_token_id"]
        _require(
            experiment["id"] == prompt["id"]
            and experiment["prompt_token_ids"] == prompt["token_ids"]
            and experiment["positions_requested"] == [-1]
            and experiment["positions_resolved"] == [position]
            and experiment["capture_positions_resolved"] == [position]
            and experiment["final_validation_position"] == position
            and isinstance(generated, int)
            and not isinstance(generated, bool)
            and 0 <= generated < config["model"]["vocabulary_size"]
            and experiment["final_layer_top1_matches_greedy"] is True,
            f"reference experiment {index} prompt/tail identity changed",
        )
        residual = _validate_residual_manifest(
            experiment["residual_capture_manifest"], position=position, label=f"reference residual {index}"
        )
        final_model = experiment["final_model_readout"]
        captured_model = experiment["captured_final_model_readout"]
        _require(
            isinstance(final_model, list) and len(final_model) == 1
            and isinstance(captured_model, list) and len(captured_model) == 1
            and final_model[0].get("target_token_id") == generated
            and captured_model[0].get("target_token_id") == generated
            and final_model[0].get("token_ids", [None])[0] == generated
            and captured_model[0].get("token_ids", [None])[0] == generated,
            f"reference experiment {index} greedy token canary changed",
        )
        _require(
            experiment["final_norm_reconstruction"].get("within_tolerance") is True
            and experiment["final_logits_reconstruction"].get("within_tolerance") is True
            and experiment["final_logits_reconstruction"].get("top_k_prefix_token_ids_match") is True,
            f"reference experiment {index} adapter canary failed",
        )
        contracts.append(
            {
                "index": index,
                "prompt_id": prompt["id"],
                "prompt_token_ids_sha256": token_ids_sha256(prompt["token_ids"]),
                "prompt_token_count": len(prompt["token_ids"]),
                "token_position": position,
                "residual_manifest_sha256": residual["sha256"],
                "generated_token_id": generated,
            }
        )
    return {
        "content_authenticated_not_hash_only": True,
        "experiment_count": len(contracts),
        "selected_content_contract_sha256": sha256_bytes(canonical_json_bytes(contracts)),
        "all_prompt_ids_and_token_ids_exact": True,
        "all_residual_and_forward_canaries_passed": True,
    }


def build_preflight_receipt(
    config: Mapping[str, Any], *, bindings: Mapping[str, Path], generation_rows: Sequence[Mapping[str, Any]], model_and_lens: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "kind": config["pre_generation_receipts"]["preflight_kind"],
        "status": "passed_cpu_preflight_no_model_or_gpu_runtime",
        "runtime_config": _binding(CONFIG_PATH, display_root=ROOT),
        "producer": _binding(PRODUCER_PATH, display_root=ROOT),
        "verified_frozen_input_sha256": {
            name: sha256_file(path) for name, path in sorted(bindings.items())
        },
        "generation_prompt_contract": {
            "record_count": len(generation_rows),
            "row_contract_sha256": config["frozen_inputs"]["generation_prompts"]["row_contract_sha256"],
            "minimum_token_count": min(len(row["token_ids"]) for row in generation_rows),
            "maximum_token_count": max(len(row["token_ids"]) for row in generation_rows),
            "exact_256_token_capacity_passed": True,
        },
        "model_and_lens": dict(model_and_lens),
        "execution_state": {
            "model_loaded": False,
            "gpu_called": False,
            "producer_called": False,
            "stage_b_read": False,
            "condition_key_read": False,
            "reserved_validation_read": False,
        },
        "claims": FALSE_CLAIMS,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_capture_lock_receipt(
    *,
    config: Mapping[str, Any],
    preflight_path: Path,
    v1_capture_verification_path: Path,
    v1_capture_manifest_path: Path,
    v1_capture_root: Path,
    reference_report_path: Path,
    capture_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the immutable pre-generation lock after complete V1 verification.

    The caller must have run the pinned V1 complete shard verifier.  We
    authenticate that receipt and independently parse the reference report's
    content, closing V1's former hash-only reference-report gap.
    """

    for path in (preflight_path, v1_capture_verification_path, v1_capture_manifest_path, v1_capture_root, reference_report_path):
        lexical_path_preflight((path,))
    preflight = load_json(_regular_file(preflight_path, "preflight receipt"))
    _require(
        preflight.get("kind") == config["pre_generation_receipts"]["preflight_kind"]
        and preflight.get("status") == "passed_cpu_preflight_no_model_or_gpu_runtime"
        and preflight.get("runtime_config", {}).get("sha256") == CONFIG_SHA256,
        "preflight receipt identity changed",
    )
    v1 = load_json(_regular_file(v1_capture_verification_path, "V1 capture verification receipt"))
    _require(
        isinstance(v1, dict)
        and v1.get("kind") == config["pre_generation_receipts"]["v1_capture_verification_kind"]
        and v1.get("status") == config["pre_generation_receipts"]["v1_capture_verification_status"]
        and v1.get("runtime_config_sha256") == config["frozen_inputs"]["v1_stage_a_runtime_config"]["sha256"]
        and v1.get("capture", {}).get("boundary_count") == 500
        and v1.get("capture", {}).get("all_tensor_values_verified") is True
        and v1.get("forbidden_access", {}).get("reserved_validation_access_authorized") is False,
        "V1 complete capture verification receipt changed",
    )
    manifest_binding = _binding(v1_capture_manifest_path)
    _require(
        v1["capture"].get("manifest_sha256") == manifest_binding["sha256"]
        and v1["capture"].get("manifest_size_bytes") == manifest_binding["size_bytes"],
        "V1 receipt does not bind the supplied capture manifest",
    )
    # Re-run the exact pinned V1 verifier over all 500 present safetensors
    # shards.  Merely presenting a JSON receipt with the expected strings is
    # insufficient for the V2 pre-generation lock.
    v1_path = _verify_binding(
        config["frozen_inputs"]["v1_stage_a_runtime_verifier"],
        label="pinned V1 complete-capture verifier",
    )
    module_spec = importlib.util.spec_from_file_location(
        "_stage_a_runtime_v1_pinned_for_v2", v1_path
    )
    _require(
        module_spec is not None and module_spec.loader is not None,
        "cannot load pinned V1 verifier",
    )
    v1_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(v1_module)
    v1_config_path = _verify_binding(
        config["frozen_inputs"]["v1_stage_a_runtime_config"],
        label="pinned V1 runtime config",
    )
    v1_config = v1_module.load_and_validate_config(v1_config_path)
    live_verification = v1_module.verify_capture_artifacts(
        config=v1_config,
        capture_rows=capture_rows,
        manifest_path=v1_capture_manifest_path,
        capture_root=v1_capture_root,
        verify_tensor_values=True,
    )
    _require(
        live_verification == v1["capture"],
        "V1 receipt differs from a fresh complete 500-shard verification",
    )
    reference = load_json(_regular_file(reference_report_path, "reference report"))
    content = authenticate_reference_report_content(reference, capture_rows=capture_rows, config=config)
    return {
        "schema_version": 2,
        "kind": config["pre_generation_receipts"]["capture_lock_kind"],
        "status": "passed_pre_generation_capture_and_reference_content_lock",
        "runtime_config_sha256": CONFIG_SHA256,
        "preflight_receipt": _binding(preflight_path),
        "v1_capture_verification_receipt": _binding(v1_capture_verification_path),
        "v1_capture_manifest": manifest_binding,
        "v1_capture_root": str(v1_capture_root.resolve(strict=True)),
        "reference_report": _binding(reference_report_path),
        "reference_report_content": content,
        "chronology": {
            "preflight_existed_before_capture_lock": True,
            "all_500_capture_shards_verified_before_capture_lock": True,
            "all_500_capture_shards_freshly_reverified_by_pinned_v1_code": True,
            "reference_content_authenticated_before_producer": True,
            "model_loaded_for_v2_producer": False,
            "producer_called": False,
        },
        "claims": FALSE_CLAIMS,
        "locked_at": datetime.now(timezone.utc).isoformat(),
    }


def _read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _regular_file(path, "normalized filesystem trace").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            _require(line.endswith("\n"), f"trace line {line_number} is not newline terminated")
            try:
                value = json.loads(line, object_pairs_hook=_reject_duplicates)
            except json.JSONDecodeError as error:
                raise SameRunVerificationError(f"invalid trace JSON line {line_number}: {error}") from error
            _require(isinstance(value, dict), f"trace line {line_number} is not an object")
            rows.append(value)
    _require(bool(rows), "normalized filesystem trace is empty")
    return rows


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def verify_filesystem_trace(
    *,
    config: Mapping[str, Any],
    trace_path: Path,
    raw_trace_archive_path: Path,
    output_root: Path,
    exact_input_paths: Sequence[Path],
    model_snapshot_path: Path,
    lens_path: Path,
    preflight_path: Path,
    capture_lock_path: Path,
) -> dict[str, Any]:
    """Verify normalized external tracing and exact allowlisted access."""

    lexical_path_preflight((trace_path, raw_trace_archive_path, output_root, *exact_input_paths, model_snapshot_path, lens_path, preflight_path, capture_lock_path))
    rows = _read_jsonl_strict(trace_path)
    header = rows[0]
    _exact_keys(
        header,
        (
            "schema_version", "kind", "tracer", "producer_pid",
            "capture_started_before_producer_exec", "all_descendants_traced",
            "lost_event_count", "sanitized_environment",
        ),
        "trace header",
    )
    _require(
        header["schema_version"] == 2
        and header["kind"] == "swe_task_state_v4_stage_a_same_run_filesystem_trace_header_v2"
        and isinstance(header["producer_pid"], int)
        and not isinstance(header["producer_pid"], bool)
        and header["capture_started_before_producer_exec"] is True
        and header["all_descendants_traced"] is True
        and header["lost_event_count"] == 0,
        "filesystem tracer coverage changed",
    )
    _require(header["sanitized_environment"] == config["environment"]["exact_values"], "producer environment changed or contains extra keys")
    _require(
        header["tracer"] == config["filesystem_trace"]["tracer"],
        "tracer binary, version, or flags changed",
    )
    exact = {path.resolve(strict=True) for path in exact_input_paths}
    model_root = model_snapshot_path.resolve(strict=True)
    lens_resolved = lens_path.resolve(strict=True)
    output_resolved = output_root.resolve(strict=True)
    runtime_roots: list[Path] = []
    runtime_files: set[Path] = set()
    for raw_root in config["filesystem_trace"]["runtime_read_only_roots"]:
        candidate = Path(raw_root)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidate = Path(os.path.normpath(os.fspath(candidate)))
        if candidate.is_file():
            runtime_files.add(candidate)
        else:
            runtime_roots.append(candidate)
    preflight_sequence: int | None = None
    capture_sequence: int | None = None
    first_model_or_lens_sequence: int | None = None
    successful_read_count = 0
    for expected_sequence, event in enumerate(rows[1:]):
        _exact_keys(event, ("sequence", "pid", "operation", "path", "result"), f"trace event {expected_sequence}")
        _require(event["sequence"] == expected_sequence, "trace event sequence is not contiguous")
        _require(isinstance(event["pid"], int) and not isinstance(event["pid"], bool), "trace PID invalid")
        _require(event["operation"] in {"read", "write", "stat", "readlink", "exec"}, "trace operation invalid")
        _require(event["result"] in {"success", "enoent", "eacces"}, "trace result invalid")
        _require(isinstance(event["path"], str) and Path(event["path"]).is_absolute(), "trace path must be absolute")
        path = Path(os.path.normpath(event["path"]))
        _require(not _path_forbidden(path), f"forbidden trace path observed: {path}")
        if event["result"] != "success":
            continue
        if event["operation"] in {"read", "stat", "readlink", "exec"}:
            successful_read_count += 1
            allowed = (
                path in exact
                or _is_within(path, model_root)
                or path == lens_resolved
                or path in runtime_files
                or any(_is_within(path, root) for root in runtime_roots)
                or _is_within(path, output_resolved)
            )
            _require(allowed, f"successful read outside allowlist: {path}")
            if path == preflight_path.resolve(strict=True):
                preflight_sequence = expected_sequence if preflight_sequence is None else preflight_sequence
            if path == capture_lock_path.resolve(strict=True):
                capture_sequence = expected_sequence if capture_sequence is None else capture_sequence
            if _is_within(path, model_root) or path == lens_resolved:
                first_model_or_lens_sequence = (
                    expected_sequence if first_model_or_lens_sequence is None else first_model_or_lens_sequence
                )
        else:
            _require(_is_within(path, output_resolved), f"write outside exact output root: {path}")
    _require(preflight_sequence is not None and capture_sequence is not None, "pre-generation receipt reads absent from trace")
    _require(first_model_or_lens_sequence is not None, "model/lens reads absent from trace")
    _require(
        preflight_sequence < first_model_or_lens_sequence
        and capture_sequence < first_model_or_lens_sequence,
        "pre-generation receipts were not read before model/lens access",
    )
    raw_binding = _binding(raw_trace_archive_path)
    return {
        "normalized_trace": _binding(trace_path),
        "raw_trace_archive": raw_binding,
        "tracer": header["tracer"],
        "event_count": len(rows) - 1,
        "successful_read_count": successful_read_count,
        "lost_event_count": 0,
        "all_successful_reads_allowlisted": True,
        "receipts_read_before_model_or_lens": True,
        "sanitized_environment_exact": True,
    }


def _logical_array_sha256(
    array: Any, *, name: str, shape: Sequence[int], layer_ids: Sequence[int] | None = None
) -> str:
    import numpy as np

    _require(isinstance(array, np.ndarray) and list(array.shape) == list(shape), f"{name} shape changed")
    _require(array.dtype == np.dtype("float32") and bool(np.isfinite(array).all()), f"{name} dtype or finiteness changed")
    little = np.asarray(array, dtype="<f4", order="C")
    header: dict[str, Any] = {"name": name, "shape": list(shape), "dtype": "little-endian-float32"}
    if layer_ids is not None:
        header["layer_ids"] = list(layer_ids)
    rendered = canonical_json_bytes(header)
    digest = hashlib.sha256()
    digest.update(len(rendered).to_bytes(8, "big"))
    digest.update(rendered)
    digest.update(little.tobytes(order="C"))
    return digest.hexdigest()


def _load_tensors(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise SameRunVerificationError("safetensors is required") from error
    with safe_open(_regular_file(path, "same-run tensor shard"), framework="np", device="cpu") as handle:
        keys = sorted(handle.keys())
        expected = ["all_layer_residual", "prompt_tail_final_state", "prompt_tail_logits", "public_j_state", "raw_residual"]
        _require(keys == expected, "tensor shard keys changed")
        return {key: handle.get_tensor(key) for key in keys}


def _load_tokenizer(snapshot: Path) -> Any:
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=True)
    except Exception as error:
        raise SameRunVerificationError(f"cannot load pinned tokenizer: {error}") from error


def _verify_record(
    *,
    record: Mapping[str, Any],
    prompt: Mapping[str, Any],
    index: int,
    config: Mapping[str, Any],
    output_root: Path,
    tokenizer: Any,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version", "kind", "status", "runtime_config_sha256", "producer",
        "backend", "index", "prompt_id", "prompt", "same_run", "state",
        "forward_canary", "generation", "claims", "boundary_limits",
        "reserved_validation_access_authorized",
    }
    _exact_keys(record, expected_keys, f"record {index}")
    _require(
        record["schema_version"] == 2
        and record["kind"] == "swe_task_state_v4_counterfactual_stage_a_same_run_shard_v2"
        and record["status"] == "same_autoregressive_request_complete"
        and record["runtime_config_sha256"] == CONFIG_SHA256
        and record["index"] == index
        and record["prompt_id"] == prompt["id"]
        and record["producer"] == config["implementation"]["producer"],
        f"record {index} identity changed",
    )
    backend = record["backend"]
    _require(
        backend.get("mode") == backend.get("kind") == "real_local_pinned_vllm_online_capture"
        and isinstance(backend.get("implementation"), dict)
        and _is_sha(backend.get("implementation", {}).get("sha256"))
        and _is_sha(backend.get("model_instance_id")),
        f"record {index} is mock or backend-unbound",
    )
    prompt_record = record["prompt"]
    submitted = prompt_record.get("submitted_token_ids")
    engine = prompt_record.get("engine_returned_token_ids")
    _require(
        submitted == prompt["token_ids"]
        and engine == prompt["token_ids"]
        and prompt_record.get("exact_equal") is True
        and prompt_record.get("submitted_token_ids_sha256") == token_ids_sha256(prompt["token_ids"])
        and prompt_record.get("engine_returned_token_ids_sha256") == token_ids_sha256(prompt["token_ids"])
        and prompt_record.get("token_count") == len(prompt["token_ids"]),
        f"record {index} submitted/engine prompt IDs differ",
    )
    same = record["same_run"]
    tail = len(prompt["token_ids"]) - 1
    _require(
        same.get("model_instance_id") == backend["model_instance_id"]
        and all(_is_sha(same.get(key)) for key in ("request_id", "model_instance_id", "prefill_forward_id", "kv_cache_sequence_id"))
        and same.get("prompt_tail_position_id") == tail
        and same.get("prompt_tail_cache_write_index") == tail
        and same.get("first_decode_cache_length") == len(prompt["token_ids"])
        and same.get("same_backend_call") is True
        and same.get("capture_was_online_in_prefill") is True
        and same.get("generation_consumed_same_kv_cache") is True
        and same.get("replay_between_capture_and_first_decode") is False
        and isinstance(same.get("capture_completed_monotonic_ns"), int)
        and same["capture_completed_monotonic_ns"] <= same.get("first_decode_started_monotonic_ns", -1),
        f"record {index} same-cache chronology changed",
    )
    shard_record = record["state"]["shard"]
    shard_path = output_root / shard_record["path"]
    _require(_binding(shard_path, display_root=output_root) | {"tensor_keys": shard_record["tensor_keys"]} == shard_record, f"record {index} shard binding changed")
    tensors = _load_tensors(shard_path)
    shapes = {
        "all_layer_residual": [64, 5120], "raw_residual": [24, 5120],
        "public_j_state": [24, 5120], "prompt_tail_final_state": [5120],
        "prompt_tail_logits": [248320],
    }
    layer_ids = {
        "all_layer_residual": list(range(64)), "raw_residual": list(range(24, 48)),
        "public_j_state": list(range(24, 48)), "prompt_tail_final_state": None,
        "prompt_tail_logits": None,
    }
    hashes = {
        name: _logical_array_sha256(array, name=name, shape=shapes[name], layer_ids=layer_ids[name])
        for name, array in tensors.items()
    }
    _require(hashes == record["state"]["tensor_logical_sha256"], f"record {index} tensor logical hashes changed")
    import numpy as np
    _require(np.array_equal(tensors["all_layer_residual"][24:48], tensors["raw_residual"]), f"record {index} selected raw state mismatch")
    generation = record["generation"]
    generated = generation.get("token_ids")
    argmax = int(np.argmax(tensors["prompt_tail_logits"]))
    _require(
        isinstance(generated, list) and len(generated) == 256
        and generation.get("token_count") == 256
        and generation.get("token_ids_sha256") == token_ids_sha256(generated)
        and generated[0] == argmax
        and generation.get("finish_reason") == "length"
        and generation.get("stop_reason") is None
        and generation.get("sampling") == config["sampling"]
        and generation.get("completion_text_sha256") == sha256_bytes(generation.get("completion_text", "").encode("utf-8")),
        f"record {index} generation/stop/first-token contract changed",
    )
    canary = record["forward_canary"]
    _require(
        canary == {
            "prompt_tail_logits_argmax_token_id": argmax,
            "first_generated_token_id": generated[0],
            "first_generated_token_matches_prompt_tail_argmax": True,
            "final_norm_reconstruction_within_tolerance": True,
            "final_logits_reconstruction_within_tolerance": True,
        },
        f"record {index} forward canary changed",
    )
    decoded = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    _require(decoded == generation["completion_text"], f"record {index} pinned tokenizer decode changed")
    _require(record["claims"] == FALSE_CLAIMS, f"record {index} claims changed")
    _require(
        record["boundary_limits"] == {
            "captured_causal_boundary_count": 1,
            "one_prompt_tail_boundary_is_a_temporal_cot_trajectory": False,
            "private_or_verbatim_cot_ground_truth_present": False,
            "this_record_can_establish_private_cot": False,
        }
        and record["reserved_validation_access_authorized"] is False,
        f"record {index} one-boundary limit changed",
    )
    return {
        "prompt_id": prompt["id"], "request_id": same["request_id"],
        "completion_status": "forced_exact_256_token_window",
        "completion_text": generation["completion_text"], "tensor_shard_sha256": shard_record["sha256"],
    }


def verify_producer_bundle(
    *,
    config: Mapping[str, Any],
    generation_rows: Sequence[Mapping[str, Any]],
    producer_result_path: Path,
    output_root: Path,
    model_snapshot_path: Path,
    preflight_path: Path,
    capture_lock_path: Path,
) -> dict[str, Any]:
    # This authorization check is intentionally first: under the current
    # frozen config, no producer path, output root, receipt, tokenizer, shard,
    # or completion artifact may be resolved or read.
    production_bindings = _require_real_verification_authorization(config)
    producer_result = load_json(_regular_file(producer_result_path, "producer result"))
    _require(isinstance(producer_result, dict), "producer result missing")
    expected_keys = {
        "schema_version", "kind", "status", "runtime_config_sha256", "producer",
        "backend", "execution_provenance", "pre_generation_receipts", "records", "completion_bundle",
        "aggregate", "claims", "boundary_limits", "gate_eligible",
        "reserved_validation_access_authorized",
    }
    _exact_keys(producer_result, expected_keys, "producer result")
    _require(
        producer_result["schema_version"] == 2
        and producer_result["kind"] == "swe_task_state_v4_counterfactual_stage_a_same_run_producer_result_v2"
        and producer_result["status"] == "producer_outputs_complete_pending_external_trace_and_cpu_verification"
        and producer_result["runtime_config_sha256"] == CONFIG_SHA256
        and producer_result["producer"] == config["implementation"]["producer"]
        and producer_result["gate_eligible"] is True
        and producer_result["backend"].get("mode") == "real_local_pinned_vllm_online_capture"
        and producer_result["backend"].get("kind") == "real_local_pinned_vllm_online_capture",
        "producer result is mock, non-gate, or unbound",
    )
    _require(
        producer_result["backend"].get("implementation")
        == production_bindings["production_backend"],
        "producer result backend does not equal the frozen source binding",
    )
    provenance = producer_result["execution_provenance"]
    model = config["model"]
    tokenizer_spec = config["tokenizer"]
    _require(
        provenance == {
            "model": {
                "repo_id": model["repo_id"],
                "revision": model["revision"],
                "snapshot_tree_sha256_before": model["snapshot_tree_sha256"],
                "snapshot_tree_sha256_after": model["snapshot_tree_sha256"],
                "snapshot_file_count": model["snapshot_file_count"],
                "snapshot_size_bytes": model["snapshot_size_bytes"],
                "checkpoint_rehashed_before_model_load": True,
                "checkpoint_rehashed_after_engine_shutdown": True,
            },
            "tokenizer": {
                "tokenizer_json_sha256": tokenizer_spec["tokenizer_json_sha256"],
                "tokenizer_config_sha256": tokenizer_spec["tokenizer_config_sha256"],
                "generation_config_sha256": tokenizer_spec["generation_config_sha256"],
                "same_snapshot_as_model": True,
            },
            "runtime": config["runtime"],
            "sanitized_environment": config["environment"]["exact_values"],
            "model_instance_id": producer_result["backend"]["model_instance_id"],
            "model_loaded_once": True,
            "engine_seed": 0,
            "started_at": provenance.get("started_at"),
            "completed_at": provenance.get("completed_at"),
        }
        and isinstance(provenance.get("started_at"), str)
        and bool(provenance["started_at"])
        and isinstance(provenance.get("completed_at"), str)
        and bool(provenance["completed_at"]),
        "producer model/tokenizer/runtime/cache/environment provenance changed",
    )
    receipts = producer_result["pre_generation_receipts"]
    _require(
        receipts.get("preflight") == _binding(preflight_path)
        and receipts.get("capture_verification_lock") == _binding(capture_lock_path)
        and receipts.get("both_received_before_first_backend_call") is True,
        "producer did not bind immutable pre-generation receipts",
    )
    capture_lock = load_json(capture_lock_path)
    _require(
        capture_lock.get("kind") == config["pre_generation_receipts"]["capture_lock_kind"]
        and capture_lock.get("status") == "passed_pre_generation_capture_and_reference_content_lock"
        and capture_lock.get("runtime_config_sha256") == CONFIG_SHA256
        and capture_lock.get("reference_report_content", {}).get("content_authenticated_not_hash_only") is True,
        "capture lock receipt content changed",
    )
    records = producer_result["records"]
    _require(isinstance(records, list) and len(records) == len(generation_rows) == 240, "producer record count changed")
    tokenizer = _load_tokenizer(model_snapshot_path)
    verified: list[dict[str, Any]] = []
    model_instance_id = producer_result["backend"].get("model_instance_id")
    seen_requests: set[str] = set()
    for index, (summary, prompt) in enumerate(zip(records, generation_rows, strict=True)):
        _exact_keys(summary, ("index", "prompt_id", "request_id", "record", "tensor_shard"), f"producer summary {index}")
        _require(summary["index"] == index and summary["prompt_id"] == prompt["id"] and _is_sha(summary["request_id"]), f"producer summary {index} changed")
        record_path = _verify_binding(summary["record"], label=f"same-run record {index}", root=output_root)
        record = load_json(record_path)
        observed = _verify_record(
            record=record, prompt=prompt, index=index, config=config,
            output_root=output_root, tokenizer=tokenizer,
        )
        _require(observed["request_id"] == summary["request_id"] and summary["request_id"] not in seen_requests, f"request {index} duplicated or mismatched")
        seen_requests.add(summary["request_id"])
        _require(record["backend"]["model_instance_id"] == model_instance_id, "more than one model instance observed")
        _require(record["state"]["shard"] == summary["tensor_shard"], f"shard summary {index} differs")
        verified.append(observed)
    aggregate = producer_result["aggregate"]
    _require(
        aggregate == {
            "record_count": 240, "backend_call_count": 240, "unique_request_count": 240,
            "single_model_instance": True, "exact_generated_tokens_per_record": 256,
            "first_token_parity_all": True,
        },
        "producer aggregate changed",
    )
    _require(producer_result["claims"] == FALSE_CLAIMS, "producer claims changed")
    _require(
        producer_result["boundary_limits"] == {
            "captured_causal_boundary_count_per_generation": 1,
            "one_prompt_tail_boundary_is_a_temporal_cot_trajectory": False,
            "private_or_verbatim_cot_ground_truth_present": False,
            "this_bundle_can_establish_private_cot": False,
        }
        and producer_result["reserved_validation_access_authorized"] is False,
        "producer one-boundary claim limit changed",
    )
    completion_path = _verify_binding(producer_result["completion_bundle"], label="completion bundle", root=output_root)
    completion = load_json(completion_path)
    _exact_keys(
        completion,
        (
            "schema_version", "kind", "status", "runtime_config_sha256",
            "surface_policy", "records", "stage_b_records_present",
            "condition_key_joined", "reserved_validation_access_authorized",
        ),
        "completion bundle",
    )
    _require(
        completion["schema_version"] == 2
        and completion["kind"] == "swe_task_state_v4_counterfactual_stage_a_same_run_completion_bundle_v2"
        and completion["status"] == "stage_a_same_run_generation_complete_condition_blind"
        and completion["runtime_config_sha256"] == CONFIG_SHA256
        and completion["surface_policy"] == "exact_256_token_same_run_completion_only_no_prompt_or_condition_join"
        and completion["records"] == [
            {"prompt_id": row["prompt_id"], "completion_status": row["completion_status"], "completion_text": row["completion_text"]}
            for row in verified
        ]
        and completion["stage_b_records_present"] is False
        and completion["condition_key_joined"] is False
        and completion["reserved_validation_access_authorized"] is False,
        "completion bundle differs from verified same-run surfaces",
    )
    return {
        "producer_result": _binding(producer_result_path),
        "completion_bundle": _binding(completion_path),
        "record_count": 240,
        "record_contract_sha256": sha256_bytes(canonical_json_bytes(verified)),
        "all_tensor_shards_reloaded": True,
        "all_prompt_echoes_exact": True,
        "all_first_tokens_match_prompt_tail_logits_argmax": True,
        "all_stop_reasons_exact_length_only": True,
        "pinned_tokenizer_replay_passed": True,
        "one_model_instance": True,
    }


def run_preflight(args: argparse.Namespace) -> int:
    lexical_path_preflight((args.config, args.output))
    config = load_and_validate_config(args.config)
    bindings = verify_frozen_bindings(config)
    _capture_rows, generation_rows = load_production_prompt_rows(config, bindings)
    model_path, lens_path = expected_model_and_lens_paths(config)
    model_and_lens = verify_model_and_lens_bytes(
        config, model_snapshot=model_path, lens_path=lens_path
    )
    receipt = build_preflight_receipt(
        config, bindings=bindings, generation_rows=generation_rows,
        model_and_lens=model_and_lens,
    )
    _atomic_json(args.output, receipt)
    print(f"wrote CPU-only no-model V2 preflight to {args.output}", file=sys.stderr)
    return 0


def run_lock_capture(args: argparse.Namespace) -> int:
    lexical_path_preflight(
        (
            args.config, args.preflight, args.v1_capture_verification,
            args.v1_capture_manifest, args.v1_capture_root, args.reference_report, args.output,
        )
    )
    config = load_and_validate_config(args.config)
    bindings = verify_frozen_bindings(config)
    capture_rows, _generation_rows = load_production_prompt_rows(config, bindings)
    receipt = build_capture_lock_receipt(
        config=config,
        preflight_path=args.preflight,
        v1_capture_verification_path=args.v1_capture_verification,
        v1_capture_manifest_path=args.v1_capture_manifest,
        v1_capture_root=args.v1_capture_root,
        reference_report_path=args.reference_report,
        capture_rows=capture_rows,
    )
    _atomic_json(args.output, receipt)
    print(f"wrote immutable V2 pre-generation capture lock to {args.output}", file=sys.stderr)
    return 0


def run_verify(args: argparse.Namespace) -> int:
    lexical_path_preflight(
        (
            args.config, args.preflight, args.capture_lock, args.producer_result,
            args.producer_output_root, args.normalized_trace, args.raw_trace_archive,
            args.model_snapshot, args.lens, args.output,
        )
    )
    config = load_and_validate_config(args.config)
    # Fail under this prospective config before resolving any producer/output
    # artifacts.  A new versioned config and verifier freeze is mandatory.
    _require_real_verification_authorization(config)
    bindings = verify_frozen_bindings(config)
    _capture_rows, generation_rows = load_production_prompt_rows(config, bindings)
    model_and_lens = verify_model_and_lens_bytes(
        config, model_snapshot=args.model_snapshot, lens_path=args.lens
    )
    preflight_before = _binding(args.preflight)
    capture_lock_before = _binding(args.capture_lock)
    bundle = verify_producer_bundle(
        config=config, generation_rows=generation_rows,
        producer_result_path=args.producer_result,
        output_root=args.producer_output_root,
        model_snapshot_path=args.model_snapshot,
        preflight_path=args.preflight, capture_lock_path=args.capture_lock,
    )
    trace = verify_filesystem_trace(
        config=config, trace_path=args.normalized_trace,
        raw_trace_archive_path=args.raw_trace_archive,
        output_root=args.producer_output_root,
        exact_input_paths=[CONFIG_PATH, PRODUCER_PATH, *bindings.values(), args.preflight, args.capture_lock],
        model_snapshot_path=args.model_snapshot, lens_path=args.lens,
        preflight_path=args.preflight, capture_lock_path=args.capture_lock,
    )
    _require(_binding(args.preflight) == preflight_before and _binding(args.capture_lock) == capture_lock_before, "pre-generation receipts changed during verification")
    receipt = {
        "schema_version": 2,
        "kind": "swe_task_state_v4_counterfactual_stage_a_same_run_final_verification_v2",
        "status": "passed_artifact_lineage_same_model_single_prompt_tail_state_only",
        "runtime_config_sha256": CONFIG_SHA256,
        "pre_generation_receipts": {"preflight": preflight_before, "capture_lock": capture_lock_before},
        "producer_bundle": bundle,
        "model_and_lens": model_and_lens,
        "filesystem_trace": trace,
        "claims": FALSE_CLAIMS,
        "boundary_limits": {
            "captured_causal_boundary_count_per_generation": 1,
            "one_prompt_tail_boundary_is_a_temporal_cot_trajectory": False,
            "private_or_verbatim_cot_ground_truth_present": False,
            "this_receipt_establishes_private_cot": False,
        },
        "reserved_validation_access_authorized": False,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(args.output, receipt)
    print(f"verified 240 same-run Stage-A records into {args.output}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", type=Path, default=CONFIG_PATH)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.set_defaults(handler=run_preflight)
    lock = sub.add_parser("lock-capture")
    lock.add_argument("--config", type=Path, default=CONFIG_PATH)
    lock.add_argument("--preflight", type=Path, required=True)
    lock.add_argument("--v1-capture-verification", type=Path, required=True)
    lock.add_argument("--v1-capture-manifest", type=Path, required=True)
    lock.add_argument("--v1-capture-root", type=Path, required=True)
    lock.add_argument("--reference-report", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    lock.set_defaults(handler=run_lock_capture)
    verify = sub.add_parser("verify")
    verify.add_argument("--config", type=Path, default=CONFIG_PATH)
    verify.add_argument("--preflight", type=Path, required=True)
    verify.add_argument("--capture-lock", type=Path, required=True)
    verify.add_argument("--producer-result", type=Path, required=True)
    verify.add_argument("--producer-output-root", type=Path, required=True)
    verify.add_argument("--normalized-trace", type=Path, required=True)
    verify.add_argument("--raw-trace-archive", type=Path, required=True)
    verify.add_argument("--model-snapshot", type=Path, required=True)
    verify.add_argument("--lens", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.set_defaults(handler=run_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
