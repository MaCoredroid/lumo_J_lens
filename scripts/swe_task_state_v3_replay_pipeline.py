#!/usr/bin/env python3
"""Split, run, and losslessly merge the frozen V3 public J-lens replay.

The production CLI is intentionally tied to the exact V3 N=60 declaration and
its dedicated output namespace.  The lower-level functions accept explicit
roots so their path, partition, report, and merge invariants can be tested with
small synthetic bundles without touching development or reserved evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence, TextIO


ROOT = Path(__file__).resolve().parents[1]
V3_OUTPUT_ROOT = ROOT / ".cache/swe_state_interpreter_v3_development"
V3_REPLAY_ROOT = V3_OUTPUT_ROOT / "replay"
DEFAULT_PROMPTS = V3_OUTPUT_ROOT / "prompts.json"
DEFAULT_SUMMARY = V3_OUTPUT_ROOT / "prompts-summary.json"
DEFAULT_PROTOCOL = ROOT / "configs/swe_task_state_interpreter_v3.json"
DEFAULT_ACTION_PROTOCOL = ROOT / "configs/swe_task_state_v3_action_probes.json"
DEFAULT_COHORT = ROOT / "configs/swe_task_state_v3_development_cohort.json"
DEFAULT_CHECKER = ROOT / "scripts/check_swe_task_state_v3_development_cohort.py"
DEFAULT_MATERIALIZER = ROOT / "scripts/materialize_swe_state_interpreter_v3_probes.py"
DEFAULT_HISTORICAL_MATERIALIZER = ROOT / "scripts/materialize_swe_behavioral_probes.py"
DEFAULT_MATERIALIZATION_RECEIPT = (
    ROOT / "validation/swe-task-state-v3-development-materialization.json"
)
JLENS_SHELL_RUNNER = ROOT / "scripts/run_jlens_nvfp4.sh"
JLENS_PYTHON_RUNNER = ROOT / "scripts/run_jlens_nvfp4.py"
REPLAY_SHELL_WRAPPER = ROOT / "scripts/run_swe_task_state_v3_replay.sh"

CHUNK_MANIFEST_NAME = "chunk-manifest.json"
RUN_MANIFEST_NAME = "run-manifest.json"
MERGED_REPORT_NAME = "public-report.json"
MERGE_MANIFEST_NAME = "merge-manifest.json"
STAGED_MERGED_REPORT_NAME = ".public-report.validated-stage.json"
CHUNK_MANIFEST_SCHEMA = 1
RUN_MANIFEST_SCHEMA = 1
MERGE_MANIFEST_SCHEMA = 1
REPORT_SCHEMA = 3
SCORE_ENCODING = "unrounded-float32"
MAX_TASKS_PER_CHUNK = 15
LAYERS = tuple(range(24, 48))
POSITIONS = (-1,)
TOP_K = 10
GPU_MEMORY_UTILIZATION = 0.78

EXPECTED_MODEL_PIN = {
    "repo_id": "nvidia/Qwen3.6-27B-NVFP4",
    "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
    "config_sha256": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
    "index_sha256": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
}
EXPECTED_PUBLIC_LENS_PIN = {
    "repo_id": "neuronpedia/jacobian-lens",
    "revision": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
    "sha256": "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1",
    "n_prompts": 1000,
}
EXPECTED_REPLAY_RUNTIME_PIN = {
    "enforce_eager": True,
    "mtp_enabled": False,
    "max_model_len": 65536,
    "max_num_batched_tokens": 4096,
    "mamba_block_size": 1024,
    "kv_cache_dtype": "fp8_e4m3",
    "kv_offloading_size": 8.0,
    "kv_offloading_backend": "native",
    "stream_final_only": True,
}
EXPECTED_REPORT_RUNTIME = {
    **EXPECTED_REPLAY_RUNTIME_PIN,
    "enable_prefix_caching": True,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    "language_model_only": True,
}
REPORT_ASSERTION_KEYS = {
    "lens_hash_matches",
    "lens_metadata_matches",
    "model_architecture_matches",
    "all_final_layer_top1_match_greedy",
    "all_final_adapter_reconstructions_within_tolerance",
}
REPORT_ROOT_KEYS = {
    "schema_version",
    "score_encoding",
    "status",
    "started_at",
    "completed_at",
    "elapsed_seconds",
    "host",
    "model",
    "lens",
    "runtime",
    "scored_vocabulary",
    "assertions",
    "experiments",
}
EXPERIMENT_KEYS = {
    "id",
    "prompt",
    "prompt_token_ids",
    "prompt_tokens",
    "positions_requested",
    "positions_resolved",
    "capture_positions_resolved",
    "final_validation_position",
    "position_tokens",
    "generated_token_id",
    "generated_token",
    "generated_text",
    "generation_seconds",
    "final_layer_top1_matches_greedy",
    "scored_vocabulary",
    "layers",
    "final_model_readout",
    "captured_final_model_readout",
    "final_norm_reconstruction",
    "final_logits_reconstruction",
    "cuda_max_memory_allocated_bytes",
    "cuda_max_memory_reserved_bytes",
    "readout_seconds",
    "residual_capture_manifest",
    "metadata",
}


class ReplayValidationError(ValueError):
    """Raised when any frozen replay or path invariant fails."""


@dataclass(frozen=True)
class AuthenticatedBundle:
    inputs: Mapping[str, Any]
    task_ids: tuple[str, ...]
    prompt_count: int
    protocol: Mapping[str, Any]
    materialization_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class ValidatedPartition:
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    replay_root: Path
    prompts_path: Path
    prompt_records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ReportAudit:
    path: Path
    sha256: str
    experiment_count: int
    experiment_ids: tuple[str, ...]
    experiment_payload_sha256s: tuple[str, ...]
    metadata: Mapping[str, Any]
    assertions: Mapping[str, bool]
    status: str
    scored_vocabulary: Mapping[str, Any]


@dataclass(frozen=True)
class ValidatedReplayMerge:
    replay_root: Path
    report_path: Path
    report_sha256: str
    merge_manifest_path: Path
    merge_manifest_sha256: str
    experiment_count: int
    prompt_bundle_sha256: str
    materialization_receipt_sha256: str
    source_freeze_git_commit: str
    data_freeze_git_commit: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayValidationError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def text(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be nonempty text")
    return value


def integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    if minimum is not None:
        require(value >= minimum, f"{label} must be at least {minimum}")
    return value


def finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    result = float(value)
    if minimum is not None:
        require(result >= minimum, f"{label} must be at least {minimum}")
    return result


def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} fields changed")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ReplayValidationError(f"value is not canonical JSON: {error}") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_value_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's int/float or bool/int coercions."""

    return canonical_json_bytes(left) == canonical_json_bytes(right)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_file(path: Path, label: str) -> Any:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        return json.loads(
            path.read_bytes(),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ReplayValidationError(f"non-finite JSON number in {label}: {token}")
            ),
        )
    except ReplayValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReplayValidationError(f"cannot read {label}: {error}") from error


def atomic_write_json(path: Path, value: Any) -> None:
    require(path.parent.is_dir() and not path.parent.is_symlink(), "output parent is unsafe")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    require(not temporary.exists() and not temporary.is_symlink(), "temporary output already exists")
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def atomic_write_new_json(path: Path, value: Any) -> None:
    """Atomically publish a new JSON file while refusing every overwrite race."""

    require(path.parent.is_dir() and not path.parent.is_symlink(), "output parent is unsafe")
    require(not path.exists() and not path.is_symlink(), "new JSON output already exists")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    require(not temporary.exists() and not temporary.is_symlink(), "temporary output already exists")
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _ijson() -> Any:
    try:
        import ijson
    except ImportError as error:
        raise ReplayValidationError(
            "bounded-memory V3 replay requires pinned ijson 3.5.0"
        ) from error
    require(getattr(ijson, "__version__", None) == "3.5.0", "ijson version must be exactly 3.5.0")
    return ijson


def _next_event(events: Iterator[tuple[str, str, Any]], label: str) -> tuple[str, str, Any]:
    try:
        return next(events)
    except StopIteration as error:
        raise ReplayValidationError(f"unexpected end of JSON while reading {label}") from error


def _build_stream_value(
    first: tuple[str, str, Any],
    events: Iterator[tuple[str, str, Any]],
    *,
    label: str,
) -> Any:
    _prefix, event, value = first
    if event in {"string", "boolean", "null", "integer", "number", "double"}:
        if event in {"number", "double"}:
            finite(value, label)
        return value
    if event == "start_array":
        result: list[Any] = []
        while True:
            item = _next_event(events, label)
            if item[1] == "end_array":
                return result
            result.append(_build_stream_value(item, events, label=label))
    if event == "start_map":
        result_map: dict[str, Any] = {}
        while True:
            item = _next_event(events, label)
            if item[1] == "end_map":
                return result_map
            require(item[1] == "map_key" and isinstance(item[2], str), f"{label} object is malformed")
            key = item[2]
            require(key not in result_map, f"duplicate JSON key in {label}: {key}")
            result_map[key] = _build_stream_value(
                _next_event(events, f"{label}.{key}"), events, label=f"{label}.{key}"
            )
    raise ReplayValidationError(f"unsupported JSON event in {label}: {event}")


def iter_strict_json_array(path: Path, label: str) -> Iterator[Mapping[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    ijson = _ijson()
    try:
        with path.open("rb") as handle:
            events = iter(ijson.parse(handle, use_float=True))
            require(_next_event(events, label) == ("", "start_array", None), f"{label} must be one JSON array")
            while True:
                event = _next_event(events, label)
                if event == ("", "end_array", None):
                    break
                require(event[1] == "start_map", f"{label} rows must be objects")
                yield mapping(_build_stream_value(event, events, label=f"{label} row"), f"{label} row")
            try:
                next(events)
            except StopIteration:
                pass
            else:
                raise ReplayValidationError(f"{label} contains trailing JSON values")
    except ReplayValidationError:
        raise
    except (OSError, UnicodeError, ijson.JSONError) as error:
        raise ReplayValidationError(f"cannot stream {label}: {error}") from error


def _assert_root(root: Path, label: str) -> Path:
    require(root.is_dir() and not root.is_symlink(), f"{label} is not a regular directory")
    absolute = root.absolute()
    require(root.resolve(strict=True) == absolute, f"{label} traverses a symlink")
    return absolute


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    raw = text(value, label)
    relative = PurePosixPath(raw)
    require(
        not relative.is_absolute()
        and raw == relative.as_posix()
        and bool(relative.parts)
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"{label} is not a canonical safe relative path",
    )
    return relative


def safe_regular_file(root: Path, relative_value: Any, label: str) -> Path:
    absolute_root = _assert_root(root, f"{label} root")
    relative = _safe_relative(relative_value, f"{label} relative path")
    path = root.joinpath(*relative.parts)
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    resolved = path.resolve(strict=True)
    require(
        resolved.is_relative_to(absolute_root) and resolved == path.absolute(),
        f"{label} traverses a symlink or escapes its root",
    )
    return path


def _safe_direct_child(root: Path, name: str, label: str) -> Path:
    _assert_root(root, f"{label} root")
    relative = _safe_relative(name, f"{label} name")
    require(len(relative.parts) == 1, f"{label} must be a direct child")
    result = root / relative.name
    require(not result.is_symlink(), f"{label} must not be a symlink")
    return result


def _repository_record(path: Path) -> dict[str, str]:
    require(path.is_file() and not path.is_symlink(), f"pinned input is not regular: {path}")
    resolved = path.resolve(strict=True)
    require(
        resolved == path.absolute()
        and resolved.is_relative_to(ROOT.resolve(strict=True)),
        f"pinned input traverses a symlink or escapes repository: {path}",
    )
    return {"path": resolved.relative_to(ROOT.resolve(strict=True)).as_posix(), "sha256": sha256_file(resolved)}


def validate_protocol_contract(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    require(protocol.get("schema_version") == 1 and protocol.get("id") == "swe-task-state-interpreter-v3", "V3 protocol identity changed")
    pins = mapping(protocol.get("pins"), "V3 pins")
    require(canonical_value_equal(mapping(pins.get("model"), "model pin"), EXPECTED_MODEL_PIN), "V3 model pin changed")
    public_lens = mapping(pins.get("public_lens"), "public lens pin")
    require(canonical_value_equal({key: public_lens.get(key) for key in EXPECTED_PUBLIC_LENS_PIN}, EXPECTED_PUBLIC_LENS_PIN), "V3 public lens pin changed")
    runtime = mapping(pins.get("replay_runtime"), "replay runtime pin")
    require(canonical_value_equal({key: runtime.get(key) for key in EXPECTED_REPLAY_RUNTIME_PIN}, EXPECTED_REPLAY_RUNTIME_PIN), "V3 replay runtime pin changed")
    for pin_name, path in (
        ("v3_materializer_sha256", DEFAULT_MATERIALIZER),
        ("historical_materializer_sha256", DEFAULT_HISTORICAL_MATERIALIZER),
        ("replay_pipeline_sha256", Path(__file__).resolve()),
        ("replay_shell_wrapper_sha256", REPLAY_SHELL_WRAPPER),
    ):
        require(
            path.is_file()
            and not path.is_symlink()
            and path.resolve(strict=True) == path.absolute()
            and pins.get(pin_name) == sha256_file(path),
            f"V3 {pin_name} differs from exact implementation bytes",
        )
    feature = mapping(protocol.get("feature_contract"), "feature contract")
    require(feature.get("source_layers") == list(LAYERS), "V3 source layer band changed")
    replay = mapping(protocol.get("bounded_memory_replay_contract"), "bounded-memory replay contract")
    require(
        replay.get("maximum_tasks_per_replay_chunk_inclusive") == MAX_TASKS_PER_CHUNK
        and replay.get("combined_report_schema_version") == REPORT_SCHEMA
        and replay.get("master_prompt_order") == "exact_materialized_N60_prompt_bundle_order"
        and replay.get("chunk_partition") == "task_disjoint_and_collectively_exhaustive_in_master_task_order"
        and replay.get("experiment_coverage") == "every_master_prompt_experiment_exactly_once_in_master_prompt_order"
        and replay.get("experiment_payload_merge") == "semantic_and_canonical_payload_unchanged_with_each_per_record_canonical_sha256_preserved"
        and replay.get("score_recomputation_or_averaging") == "forbidden",
        "V3 bounded-memory replay contract changed",
    )
    return protocol


def _load_checker(expected_sha256: str) -> Any:
    require(DEFAULT_CHECKER.is_file() and not DEFAULT_CHECKER.is_symlink(), "V3 checker is missing or unsafe")
    require(sha256_file(DEFAULT_CHECKER) == expected_sha256, "V3 checker differs from the protocol pin")
    name = "_swe_task_state_v3_replay_pinned_checker"
    specification = importlib.util.spec_from_file_location(name, DEFAULT_CHECKER)
    require(specification is not None and specification.loader is not None, "cannot import V3 checker")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _load_materializer(expected_sha256: str) -> Any:
    require(
        DEFAULT_MATERIALIZER.is_file() and not DEFAULT_MATERIALIZER.is_symlink(),
        "V3 materializer is missing or unsafe",
    )
    require(
        sha256_file(DEFAULT_MATERIALIZER) == expected_sha256,
        "V3 materializer differs from the protocol pin",
    )
    name = "_swe_task_state_v3_replay_pinned_materializer"
    specification = importlib.util.spec_from_file_location(name, DEFAULT_MATERIALIZER)
    require(
        specification is not None and specification.loader is not None,
        "cannot import V3 materializer",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    require(
        callable(getattr(module, "verify_frozen_materialization", None)),
        "V3 materializer verification API changed",
    )
    return module


def authenticate_production_bundle(
    prompts_path: Path,
    summary_path: Path,
    *,
    verify_rematerialization: bool = False,
) -> AuthenticatedBundle:
    for supplied, expected, label in (
        (prompts_path, DEFAULT_PROMPTS, "master prompt bundle"),
        (summary_path, DEFAULT_SUMMARY, "master prompt summary"),
        (DEFAULT_PROTOCOL, DEFAULT_PROTOCOL, "V3 protocol"),
        (DEFAULT_COHORT, DEFAULT_COHORT, "V3 cohort"),
        (
            DEFAULT_MATERIALIZATION_RECEIPT,
            DEFAULT_MATERIALIZATION_RECEIPT,
            "V3 materialization receipt",
        ),
    ):
        require(not supplied.is_symlink(), f"{label} must not be a symlink")
        require(supplied.resolve(strict=True) == expected.resolve(strict=True), f"{label} path is not canonical")
    protocol = validate_protocol_contract(mapping(strict_json_file(DEFAULT_PROTOCOL, "V3 protocol"), "V3 protocol"))
    pins = mapping(protocol.get("pins"), "V3 pins")
    checker_hash = text(pins.get("materialized_bundle_checker_sha256"), "checker pin")
    require(sha256_file(DEFAULT_ACTION_PROTOCOL) == pins.get("v3_action_protocol_sha256"), "V3 action protocol differs from its pin")
    checker = _load_checker(checker_hash)
    declaration = checker.validate_declaration(DEFAULT_COHORT)
    checked = checker.validate_materialized_bundle(
        declaration,
        prompts_path=prompts_path,
        summary_path=summary_path,
    )
    require(
        checked.get("task_count") == 60
        and checked.get("cohort_count") == 2
        and checked.get("prompt_bundle_sha256") == sha256_file(prompts_path)
        and checked.get("summary_sha256") == sha256_file(summary_path),
        "pinned V3 checker returned an inconsistent N60 binding",
    )
    materialization_receipt = mapping(
        strict_json_file(DEFAULT_MATERIALIZATION_RECEIPT, "V3 materialization receipt"),
        "V3 materialization receipt",
    )
    receipt_audit = checker.validate_materialization_receipt(
        declaration,
        prompts_path=prompts_path,
        summary_path=summary_path,
        receipt_path=DEFAULT_MATERIALIZATION_RECEIPT,
        require_git_frozen=True,
    )
    require(
        receipt_audit.prompt_bundle_sha256 == checked.get("prompt_bundle_sha256")
        and receipt_audit.summary_sha256 == checked.get("summary_sha256")
        and isinstance(receipt_audit.data_freeze_git_commit, str),
        "Git-frozen materialization receipt returned inconsistent output bindings",
    )
    if verify_rematerialization:
        materializer = _load_materializer(
            text(pins.get("v3_materializer_sha256"), "V3 materializer pin")
        )
        verification = materializer.verify_frozen_materialization(
            checker=checker,
            declaration=declaration,
            receipt=materialization_receipt,
            prompts_path=prompts_path,
            summary_path=summary_path,
        )
        require(
            mapping(verification, "materialization verification").get("exact_match")
            is True,
            "deterministic materialization verification failed",
        )
    task_ids = tuple((*declaration.campaign_ids[0], *declaration.campaign_ids[1]))
    require(len(task_ids) == 60 and len(set(task_ids)) == 60, "authenticated V3 declaration is not exact N60")
    inputs = {
        "master_prompt_bundle": _repository_record(prompts_path),
        "master_prompt_summary": _repository_record(summary_path),
        "cohort_manifest": _repository_record(DEFAULT_COHORT),
        "interpreter_protocol": _repository_record(DEFAULT_PROTOCOL),
        "action_protocol": _repository_record(DEFAULT_ACTION_PROTOCOL),
        "materialized_bundle_checker": _repository_record(DEFAULT_CHECKER),
        "v3_materializer": _repository_record(DEFAULT_MATERIALIZER),
        "historical_materializer": _repository_record(DEFAULT_HISTORICAL_MATERIALIZER),
        "materialization_receipt": _repository_record(DEFAULT_MATERIALIZATION_RECEIPT),
        "materialization_freeze": {
            "source_freeze_git_commit": receipt_audit.source_freeze_git_commit,
            "data_freeze_git_commit": receipt_audit.data_freeze_git_commit,
            "exact_child_receipt_only_commit_validated": True,
            "deterministic_rematerialization_required_before_split": True,
        },
        "replay_pipeline": _repository_record(Path(__file__).resolve()),
        "replay_shell_wrapper": _repository_record(REPLAY_SHELL_WRAPPER),
        "jlens_shell_runner": _repository_record(JLENS_SHELL_RUNNER),
        "jlens_python_runner": _repository_record(JLENS_PYTHON_RUNNER),
    }
    return AuthenticatedBundle(
        inputs=inputs,
        task_ids=task_ids,
        prompt_count=integer(checked.get("prompt_count"), "checked prompt count", minimum=1),
        protocol=protocol,
        materialization_receipt=dict(materialization_receipt),
    )


def _prompt_task_id(prompt: Mapping[str, Any]) -> str:
    metadata = mapping(prompt.get("metadata"), "prompt metadata")
    task = mapping(metadata.get("task"), "prompt task")
    return text(task.get("instance_id"), "prompt task ID")


def _prompt_payload_pin(prompt: Mapping[str, Any]) -> str:
    provenance = mapping(mapping(prompt.get("metadata"), "prompt metadata").get("provenance"), "prompt provenance")
    value = text(provenance.get("prompt_record_payload_sha256"), "prompt payload SHA-256")
    require(len(value) == 64 and all(char in "0123456789abcdef" for char in value), "prompt payload SHA-256 is malformed")
    return value


def _open_chunk(path: Path) -> TextIO:
    handle = path.open("x", encoding="utf-8", newline="\n")
    handle.write("[\n")
    return handle


def _close_chunk(handle: TextIO) -> None:
    handle.write("\n]\n")
    handle.flush()
    os.fsync(handle.fileno())
    handle.close()


def create_chunk_manifest(
    *,
    prompts_path: Path,
    replay_root: Path,
    task_ids: Sequence[str],
    inputs: Mapping[str, Any],
    max_tasks_per_chunk: int = MAX_TASKS_PER_CHUNK,
) -> Path:
    """Stream one authenticated master array into whole-task chunk arrays."""

    require(prompts_path.is_file() and not prompts_path.is_symlink(), "master prompts are not regular")
    require(task_ids and len(task_ids) == len(set(task_ids)), "master task IDs are empty or duplicated")
    require(1 <= max_tasks_per_chunk <= MAX_TASKS_PER_CHUNK, "chunk task bound exceeds frozen maximum")
    parent = replay_root.parent
    _assert_root(parent, "replay parent")
    require(not replay_root.exists() and not replay_root.is_symlink(), "replay root already exists")
    temporary = Path(tempfile.mkdtemp(prefix=".v3-replay-build-", dir=parent))
    try:
        chunks_root = temporary / "chunks"
        chunks_root.mkdir()
        chunks: list[dict[str, Any]] = []
        all_prompt_ids: list[str] = []
        all_prompt_hashes: list[str] = []
        seen_prompt_ids: set[str] = set()
        task_cursor = 0
        task_prompt_count = 0
        prompt_index = 0
        chunk_index = -1
        chunk_handle: TextIO | None = None
        chunk_record: dict[str, Any] | None = None
        first_in_chunk = True

        def begin_chunk() -> tuple[TextIO, dict[str, Any]]:
            nonlocal chunk_index
            chunk_index += 1
            chunk_id = f"chunk-{chunk_index:03d}"
            relative = f"chunks/{chunk_id}-prompts.json"
            record: dict[str, Any] = {
                "index": chunk_index,
                "id": chunk_id,
                "prompts_path": relative,
                "prompts_sha256": None,
                "master_task_range_inclusive": [task_cursor, min(task_cursor + max_tasks_per_chunk - 1, len(task_ids) - 1)],
                "master_prompt_range_inclusive": [prompt_index, None],
                "task_count": 0,
                "prompt_count": 0,
                "tasks": [],
                "prompts": [],
            }
            return _open_chunk(temporary / relative), record

        chunk_handle, chunk_record = begin_chunk()
        for raw_prompt in iter_strict_json_array(prompts_path, "master prompt bundle"):
            prompt = dict(raw_prompt)
            prompt_id = text(prompt.get("id"), "prompt ID")
            require(prompt_id not in seen_prompt_ids, "master prompt IDs repeat")
            seen_prompt_ids.add(prompt_id)
            instance_id = _prompt_task_id(prompt)
            require(task_cursor < len(task_ids), "master prompt bundle contains a trailing task")
            if instance_id != task_ids[task_cursor]:
                require(task_prompt_count >= 1, f"master task has no prompts: {task_ids[task_cursor]}")
                task_cursor += 1
                task_prompt_count = 0
                require(task_cursor < len(task_ids) and instance_id == task_ids[task_cursor], "master task order changed")
                if task_cursor % max_tasks_per_chunk == 0:
                    require(chunk_handle is not None and chunk_record is not None, "chunk writer state is invalid")
                    _close_chunk(chunk_handle)
                    chunk_record["master_prompt_range_inclusive"][1] = prompt_index - 1
                    chunk_record["prompts_sha256"] = sha256_file(temporary / chunk_record["prompts_path"])
                    chunks.append(chunk_record)
                    chunk_handle, chunk_record = begin_chunk()
                    first_in_chunk = True
            require(chunk_record is not None and chunk_handle is not None, "chunk writer state is invalid")
            task_rows = chunk_record["tasks"]
            if not task_rows or task_rows[-1]["instance_id"] != instance_id:
                task_rows.append(
                    {
                        "instance_id": instance_id,
                        "master_task_index": task_cursor,
                        "master_prompt_range_inclusive": [prompt_index, None],
                        "prompt_count": 0,
                        "ordered_prompt_ids_sha256": None,
                        "ordered_prompt_canonical_sha256s_sha256": None,
                        "ordered_declared_payload_sha256s_sha256": None,
                    }
                )
                chunk_record["task_count"] += 1
            if not first_in_chunk:
                chunk_handle.write(",\n")
            chunk_handle.write(json.dumps(prompt, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
            first_in_chunk = False
            prompt_hash = canonical_sha256(prompt)
            payload_hash = _prompt_payload_pin(prompt)
            prompt_record = {
                "master_prompt_index": prompt_index,
                "id": prompt_id,
                "instance_id": instance_id,
                "canonical_sha256": prompt_hash,
                "declared_payload_sha256": payload_hash,
            }
            chunk_record["prompts"].append(prompt_record)
            chunk_record["prompt_count"] += 1
            task_record = task_rows[-1]
            task_record["prompt_count"] += 1
            task_record["master_prompt_range_inclusive"][1] = prompt_index
            all_prompt_ids.append(prompt_id)
            all_prompt_hashes.append(prompt_hash)
            task_prompt_count += 1
            prompt_index += 1

        require(prompt_index >= 1, "master prompt bundle is empty")
        require(task_prompt_count >= 1, f"master task has no prompts: {task_ids[task_cursor]}")
        require(task_cursor == len(task_ids) - 1, "master prompt bundle does not cover every task")
        require(chunk_handle is not None and chunk_record is not None, "chunk writer did not start")
        _close_chunk(chunk_handle)
        chunk_record["master_prompt_range_inclusive"][1] = prompt_index - 1
        chunk_record["prompts_sha256"] = sha256_file(temporary / chunk_record["prompts_path"])
        chunks.append(chunk_record)

        for chunk in chunks:
            prompts = sequence(chunk["prompts"], "chunk prompt records")
            for task_record in sequence(chunk["tasks"], "chunk task records"):
                start, end = task_record["master_prompt_range_inclusive"]
                rows = [row for row in prompts if start <= row["master_prompt_index"] <= end]
                require(len(rows) == task_record["prompt_count"], "task prompt manifest count changed")
                task_record["ordered_prompt_ids_sha256"] = canonical_sha256([row["id"] for row in rows])
                task_record["ordered_prompt_canonical_sha256s_sha256"] = canonical_sha256([row["canonical_sha256"] for row in rows])
                task_record["ordered_declared_payload_sha256s_sha256"] = canonical_sha256([row["declared_payload_sha256"] for row in rows])

        manifest = {
            "schema_version": CHUNK_MANIFEST_SCHEMA,
            "id": "swe-task-state-v3-bounded-memory-replay-chunks",
            "status": "split_from_checker_authenticated_exact_n60_master",
            "inputs": dict(inputs),
            "partition_contract": {
                "master_order": "exact_materialized_N60_prompt_bundle_order",
                "maximum_tasks_per_chunk_inclusive": MAX_TASKS_PER_CHUNK,
                "selected_tasks_per_chunk_inclusive": max_tasks_per_chunk,
                "whole_tasks_only": True,
                "task_disjoint": True,
                "collectively_exhaustive": True,
                "prompt_values_unchanged": True,
            },
            "task_count": len(task_ids),
            "prompt_count": prompt_index,
            "chunk_count": len(chunks),
            "ordered_task_ids": list(task_ids),
            "ordered_task_ids_sha256": canonical_sha256(list(task_ids)),
            "ordered_prompt_ids_sha256": canonical_sha256(all_prompt_ids),
            "ordered_prompt_canonical_sha256s_sha256": canonical_sha256(all_prompt_hashes),
            "chunks": chunks,
        }
        atomic_write_json(temporary / CHUNK_MANIFEST_NAME, manifest)
        temporary.rename(replay_root)
        return replay_root / CHUNK_MANIFEST_NAME
    except Exception:
        if temporary.exists() and not temporary.is_symlink() and temporary.parent == parent:
            shutil.rmtree(temporary)
        raise


def _manifest_inputs(value: Any, expected: Mapping[str, Any]) -> Mapping[str, Any]:
    inputs = mapping(value, "chunk manifest inputs")
    require(inputs == expected, "chunk manifest upstream pins changed")
    return inputs


def validate_chunk_manifest(
    *,
    manifest_path: Path,
    replay_root: Path,
    prompts_path: Path,
    expected_inputs: Mapping[str, Any],
    expected_task_ids: Sequence[str],
) -> ValidatedPartition:
    _assert_root(replay_root, "replay root")
    require(
        manifest_path == replay_root / CHUNK_MANIFEST_NAME
        and manifest_path.is_file()
        and not manifest_path.is_symlink(),
        "chunk manifest is not the exact regular replay-root child",
    )
    manifest = mapping(strict_json_file(manifest_path, "chunk manifest"), "chunk manifest")
    exact_keys(
        manifest,
        {
            "schema_version", "id", "status", "inputs", "partition_contract",
            "task_count", "prompt_count", "chunk_count", "ordered_task_ids",
            "ordered_task_ids_sha256", "ordered_prompt_ids_sha256",
            "ordered_prompt_canonical_sha256s_sha256", "chunks",
        },
        "chunk manifest",
    )
    require(
        manifest.get("schema_version") == CHUNK_MANIFEST_SCHEMA
        and manifest.get("id") == "swe-task-state-v3-bounded-memory-replay-chunks"
        and manifest.get("status") == "split_from_checker_authenticated_exact_n60_master",
        "chunk manifest identity changed",
    )
    _manifest_inputs(manifest.get("inputs"), expected_inputs)
    contract = mapping(manifest.get("partition_contract"), "partition contract")
    exact_keys(
        contract,
        {
            "master_order", "maximum_tasks_per_chunk_inclusive",
            "selected_tasks_per_chunk_inclusive", "whole_tasks_only",
            "task_disjoint", "collectively_exhaustive", "prompt_values_unchanged",
        },
        "partition contract",
    )
    selected_bound = integer(contract.get("selected_tasks_per_chunk_inclusive"), "selected chunk bound", minimum=1)
    require(
        contract.get("master_order") == "exact_materialized_N60_prompt_bundle_order"
        and contract.get("maximum_tasks_per_chunk_inclusive") == MAX_TASKS_PER_CHUNK
        and selected_bound <= MAX_TASKS_PER_CHUNK
        and all(contract.get(key) is True for key in ("whole_tasks_only", "task_disjoint", "collectively_exhaustive", "prompt_values_unchanged")),
        "partition contract is not frozen and fail-closed",
    )
    task_ids = list(expected_task_ids)
    require(
        manifest.get("task_count") == len(task_ids)
        and manifest.get("ordered_task_ids") == task_ids
        and manifest.get("ordered_task_ids_sha256") == canonical_sha256(task_ids),
        "chunk manifest task identity/order changed",
    )
    require(prompts_path.is_file() and not prompts_path.is_symlink(), "master prompt bundle is unsafe")
    master_pin = mapping(expected_inputs.get("master_prompt_bundle"), "master prompt pin")
    require(sha256_file(prompts_path) == master_pin.get("sha256"), "master prompt bundle bytes changed")

    chunks = [mapping(row, f"chunk {index}") for index, row in enumerate(sequence(manifest.get("chunks"), "chunks"))]
    require(manifest.get("chunk_count") == len(chunks) and bool(chunks), "chunk count changed")
    flat_prompt_records: list[Mapping[str, Any]] = []
    flat_task_ids: list[str] = []
    chunk_paths: list[Path] = []
    expected_task_cursor = 0
    expected_prompt_cursor = 0
    for index, chunk in enumerate(chunks):
        exact_keys(
            chunk,
            {
                "index", "id", "prompts_path", "prompts_sha256",
                "master_task_range_inclusive", "master_prompt_range_inclusive",
                "task_count", "prompt_count", "tasks", "prompts",
            },
            f"chunk {index}",
        )
        require(chunk.get("index") == index and chunk.get("id") == f"chunk-{index:03d}", f"chunk {index} identity changed")
        relative = _safe_relative(chunk.get("prompts_path"), f"chunk {index} prompt path")
        require(relative.parent == PurePosixPath("chunks") and relative.name == f"chunk-{index:03d}-prompts.json", f"chunk {index} prompt path changed")
        path = safe_regular_file(replay_root, relative.as_posix(), f"chunk {index} prompts")
        require(path not in chunk_paths, "chunk prompt path repeats")
        chunk_paths.append(path)
        require(sha256_file(path) == chunk.get("prompts_sha256"), f"chunk {index} prompt bytes changed")
        task_range = sequence(chunk.get("master_task_range_inclusive"), f"chunk {index} task range")
        prompt_range = sequence(chunk.get("master_prompt_range_inclusive"), f"chunk {index} prompt range")
        require(
            len(task_range) == len(prompt_range) == 2
            and task_range[0] == expected_task_cursor
            and prompt_range[0] == expected_prompt_cursor,
            f"chunk {index} ranges are noncontiguous",
        )
        tasks = [mapping(row, f"chunk {index} task") for row in sequence(chunk.get("tasks"), f"chunk {index} tasks")]
        prompt_records = [mapping(row, f"chunk {index} prompt record") for row in sequence(chunk.get("prompts"), f"chunk {index} prompt records")]
        require(
            1 <= len(tasks) <= selected_bound
            and chunk.get("task_count") == len(tasks)
            and chunk.get("prompt_count") == len(prompt_records)
            and bool(prompt_records)
            and task_range[1] == expected_task_cursor + len(tasks) - 1
            and prompt_range[1] == expected_prompt_cursor + len(prompt_records) - 1,
            f"chunk {index} counts/ranges changed",
        )
        for local_prompt_index, record in enumerate(prompt_records):
            exact_keys(record, {"master_prompt_index", "id", "instance_id", "canonical_sha256", "declared_payload_sha256"}, f"chunk {index} prompt record")
            master_prompt_index = integer(record.get("master_prompt_index"), "master prompt index", minimum=0)
            canonical_hash = text(record.get("canonical_sha256"), "prompt canonical SHA-256")
            payload_hash = text(record.get("declared_payload_sha256"), "declared payload SHA-256")
            require(
                master_prompt_index == expected_prompt_cursor + local_prompt_index
                and len(canonical_hash) == len(payload_hash) == 64
                and all(char in "0123456789abcdef" for char in canonical_hash + payload_hash),
                "chunk prompt indices or hashes changed",
            )
            text(record.get("id"), "manifest prompt ID")
            text(record.get("instance_id"), "manifest prompt task ID")
        for local_task_index, task_record in enumerate(tasks):
            exact_keys(
                task_record,
                {
                    "instance_id", "master_task_index", "master_prompt_range_inclusive",
                    "prompt_count", "ordered_prompt_ids_sha256",
                    "ordered_prompt_canonical_sha256s_sha256",
                    "ordered_declared_payload_sha256s_sha256",
                },
                f"chunk {index} task record",
            )
            master_task_index = expected_task_cursor + local_task_index
            require(master_task_index < len(task_ids), "chunk contains a trailing task")
            instance_id = task_ids[master_task_index]
            require(
                task_record.get("instance_id") == instance_id
                and task_record.get("master_task_index") == master_task_index,
                "chunk task identity/order changed",
            )
            task_prompt_range = sequence(task_record.get("master_prompt_range_inclusive"), "task prompt range")
            require(len(task_prompt_range) == 2, "task prompt range must have two endpoints")
            task_prompt_start = integer(task_prompt_range[0], "task prompt range start", minimum=0)
            task_prompt_end = integer(task_prompt_range[1], "task prompt range end", minimum=task_prompt_start)
            rows = [
                row
                for row in prompt_records
                if task_prompt_start <= row["master_prompt_index"] <= task_prompt_end
            ]
            require(
                bool(rows)
                and task_record.get("prompt_count") == len(rows)
                and [row.get("instance_id") for row in rows] == [instance_id] * len(rows)
                and task_record.get("ordered_prompt_ids_sha256") == canonical_sha256([row.get("id") for row in rows])
                and task_record.get("ordered_prompt_canonical_sha256s_sha256") == canonical_sha256([row.get("canonical_sha256") for row in rows])
                and task_record.get("ordered_declared_payload_sha256s_sha256") == canonical_sha256([row.get("declared_payload_sha256") for row in rows]),
                f"chunk task prompt identity hashes changed: {instance_id}",
            )
            flat_task_ids.append(instance_id)
        flat_prompt_records.extend(prompt_records)
        expected_task_cursor += len(tasks)
        expected_prompt_cursor += len(prompt_records)
    require(flat_task_ids == task_ids, "chunks are not task-disjoint/exhaustive in master order")
    require(
        manifest.get("prompt_count") == expected_prompt_cursor
        and manifest.get("ordered_prompt_ids_sha256") == canonical_sha256([row.get("id") for row in flat_prompt_records])
        and manifest.get("ordered_prompt_canonical_sha256s_sha256") == canonical_sha256([row.get("canonical_sha256") for row in flat_prompt_records]),
        "chunk prompt coverage/order hashes changed",
    )

    def chunk_prompt_stream() -> Iterator[Mapping[str, Any]]:
        for index, path in enumerate(chunk_paths):
            yield from iter_strict_json_array(path, f"chunk {index} prompt bundle")

    master_iterator = iter(iter_strict_json_array(prompts_path, "master prompt bundle"))
    chunk_iterator = iter(chunk_prompt_stream())
    sentinel = object()
    seen_ids: set[str] = set()
    for index, expected in enumerate(flat_prompt_records):
        master = next(master_iterator, sentinel)
        chunk_prompt = next(chunk_iterator, sentinel)
        require(master is not sentinel and chunk_prompt is not sentinel, "chunk or master prompt row is missing")
        require(master == chunk_prompt, f"chunk prompt value changed at master index {index}")
        prompt_id = text(master.get("id"), "prompt ID")
        require(prompt_id not in seen_ids and prompt_id == expected.get("id"), "chunk prompt ID duplicated or reordered")
        seen_ids.add(prompt_id)
        require(
            _prompt_task_id(master) == expected.get("instance_id")
            and canonical_sha256(master) == expected.get("canonical_sha256")
            and _prompt_payload_pin(master) == expected.get("declared_payload_sha256"),
            f"chunk prompt hash/provenance changed: {prompt_id}",
        )
    require(next(master_iterator, sentinel) is sentinel, "master prompt bundle has trailing rows")
    require(next(chunk_iterator, sentinel) is sentinel, "chunk prompt bundles have trailing rows")
    return ValidatedPartition(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        manifest=manifest,
        replay_root=replay_root,
        prompts_path=prompts_path,
        prompt_records=tuple(flat_prompt_records),
    )


def _stream_report_experiments(path: Path, retained: dict[str, Any], *, allow_combined: bool) -> Iterator[Mapping[str, Any]]:
    require(path.is_file() and not path.is_symlink(), "report is not a regular file")
    ijson = _ijson()
    seen: set[str] = set()
    experiments_seen = False
    try:
        with path.open("rb") as handle:
            events = iter(ijson.parse(handle, use_float=True))
            require(_next_event(events, "report") == ("", "start_map", None), "report root must be an object")
            while True:
                event = _next_event(events, "report")
                if event == ("", "end_map", None):
                    break
                require(event[0] == "" and event[1] == "map_key" and isinstance(event[2], str), "report root is malformed")
                key = event[2]
                require(key not in seen, f"report repeats root key: {key}")
                seen.add(key)
                first = _next_event(events, f"report.{key}")
                if key == "experiments":
                    experiments_seen = True
                    require(first == ("experiments", "start_array", None), "report experiments must be an array")
                    while True:
                        row_event = _next_event(events, "report experiments")
                        if row_event == ("experiments", "end_array", None):
                            break
                        require(row_event[1] == "start_map", "report experiment rows must be objects")
                        yield mapping(_build_stream_value(row_event, events, label="report experiment"), "report experiment")
                else:
                    retained[key] = _build_stream_value(first, events, label=f"report.{key}")
            allowed = set(REPORT_ROOT_KEYS)
            if allow_combined:
                allowed.add("combined_chunk_provenance")
            require(seen == allowed and experiments_seen, "report root fields changed or report is incomplete")
            try:
                next(events)
            except StopIteration:
                pass
            else:
                raise ReplayValidationError("report contains trailing JSON values")
    except ReplayValidationError:
        raise
    except (OSError, UnicodeError, ijson.JSONError) as error:
        raise ReplayValidationError(f"cannot stream report: {error}") from error


def _validate_report_pins(retained: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    validate_protocol_contract(protocol)
    require(
        integer(retained.get("schema_version"), "report schema version") == REPORT_SCHEMA
        and retained.get("score_encoding") == SCORE_ENCODING,
        "report schema or score encoding changed",
    )
    model = mapping(retained.get("model"), "report model")
    lens = mapping(retained.get("lens"), "report lens")
    runtime = mapping(retained.get("runtime"), "report runtime")
    mapping(retained.get("host"), "report host")
    text(retained.get("started_at"), "report start timestamp")
    text(retained.get("completed_at"), "report completion timestamp")
    finite(retained.get("elapsed_seconds"), "report elapsed seconds", minimum=0.0)
    require(canonical_value_equal({key: model.get(key) for key in EXPECTED_MODEL_PIN}, EXPECTED_MODEL_PIN), "report model pin changed")
    require(canonical_value_equal({key: lens.get(key) for key in EXPECTED_PUBLIC_LENS_PIN}, EXPECTED_PUBLIC_LENS_PIN), "report public lens pin changed")
    require(canonical_value_equal({key: runtime.get(key) for key in EXPECTED_REPORT_RUNTIME}, EXPECTED_REPORT_RUNTIME), "report runtime pin changed")


def _validate_fidelity(value: Any, label: str) -> None:
    fidelity = mapping(value, label)
    exact_keys(
        fidelity,
        {
            "reference", "kl_final_to_readout", "kl_readout_to_final",
            "jensen_shannon_divergence", "total_variation_distance",
            "top1_matches_final", "top_k", "top_k_overlap_count",
            "top_k_overlap_fraction",
        },
        label,
    )
    require(fidelity.get("reference") == "captured_block_63_final_model", f"{label} reference changed")
    require(isinstance(fidelity.get("top1_matches_final"), bool), f"{label} top-1 flag is invalid")
    overlap_k = integer(fidelity.get("top_k"), f"{label} top-k", minimum=1)
    overlap = integer(fidelity.get("top_k_overlap_count"), f"{label} overlap", minimum=0)
    require(overlap_k == 5 and overlap <= overlap_k, f"{label} top-k overlap changed")
    for key in (
        "kl_final_to_readout", "kl_readout_to_final", "jensen_shannon_divergence",
        "total_variation_distance", "top_k_overlap_fraction",
    ):
        finite(fidelity.get(key), f"{label} {key}", minimum=0.0)


def _validate_readout(
    value: Any,
    *,
    label: str,
    scored_ids: Sequence[int],
    scored_tokens: Sequence[str],
    fidelity_required: bool,
) -> Mapping[str, Any]:
    readout = mapping(value, label)
    keys = {
        "token_ids", "scores", "target_token_id", "target_score",
        "target_logprob", "target_rank", "scored_tokens", "tokens",
        "target_token",
    }
    if fidelity_required:
        keys.add("final_distribution_fidelity")
    exact_keys(readout, keys, label)
    top_ids = sequence(readout.get("token_ids"), f"{label} top token IDs")
    scores = sequence(readout.get("scores"), f"{label} top scores")
    tokens = sequence(readout.get("tokens"), f"{label} top tokens")
    require(len(top_ids) == len(scores) == len(tokens) == TOP_K, f"{label} top-k width changed")
    require(len(set(top_ids)) == TOP_K, f"{label} top token IDs repeat")
    for index, (token_id, score, token) in enumerate(zip(top_ids, scores, tokens, strict=True)):
        integer(token_id, f"{label} top token {index}", minimum=0)
        finite(score, f"{label} top score {index}")
        require(isinstance(token, str), f"{label} top token text is invalid")
    integer(readout.get("target_token_id"), f"{label} target token ID", minimum=0)
    finite(readout.get("target_score"), f"{label} target score")
    finite(readout.get("target_logprob"), f"{label} target logprob")
    integer(readout.get("target_rank"), f"{label} target rank", minimum=1)
    require(isinstance(readout.get("target_token"), str), f"{label} target token text is invalid")
    scored_rows = [mapping(row, f"{label} scored row") for row in sequence(readout.get("scored_tokens"), f"{label} scored rows")]
    require(len(scored_rows) == len(scored_ids), f"{label} scored-token width changed")
    for index, (row, token_id, token) in enumerate(zip(scored_rows, scored_ids, scored_tokens, strict=True)):
        exact_keys(row, {"token_id", "score", "logprob", "rank", "token"}, f"{label} scored row {index}")
        require(row.get("token_id") == token_id and row.get("token") == token, f"{label} scored-token identity changed")
        finite(row.get("score"), f"{label} scored score {index}")
        finite(row.get("logprob"), f"{label} scored logprob {index}")
        integer(row.get("rank"), f"{label} scored rank {index}", minimum=1)
    if fidelity_required:
        _validate_fidelity(readout.get("final_distribution_fidelity"), f"{label} fidelity")
    return readout


def _validate_reconstruction_canaries(
    experiment: Mapping[str, Any], *, label: str
) -> bool:
    final_norm = mapping(experiment.get("final_norm_reconstruction"), f"{label} final norm canary")
    exact_keys(
        final_norm,
        {
            "max_abs_error", "rms_error", "reference_rms", "relative_rms_error",
            "max_abs_tolerance", "rms_tolerance", "within_tolerance",
        },
        f"{label} final norm canary",
    )
    norm_max = finite(final_norm.get("max_abs_error"), f"{label} norm max error", minimum=0.0)
    norm_rms = finite(final_norm.get("rms_error"), f"{label} norm RMS error", minimum=0.0)
    finite(final_norm.get("reference_rms"), f"{label} norm reference RMS", minimum=0.0)
    finite(final_norm.get("relative_rms_error"), f"{label} norm relative RMS", minimum=0.0)
    require(
        final_norm.get("max_abs_tolerance") == 0.125
        and final_norm.get("rms_tolerance") == 0.006
        and isinstance(final_norm.get("within_tolerance"), bool)
        and final_norm.get("within_tolerance") is (norm_max <= 0.125 and norm_rms <= 0.006),
        f"{label} final norm canary is internally inconsistent",
    )
    final_logits = mapping(experiment.get("final_logits_reconstruction"), f"{label} final logits canary")
    exact_keys(
        final_logits,
        {
            "max_abs_error", "max_abs_tolerance", "rms_error", "rms_tolerance",
            "top_k_prefix", "top_k_prefix_token_ids_match", "within_tolerance",
        },
        f"{label} final logits canary",
    )
    logit_max = finite(final_logits.get("max_abs_error"), f"{label} logit max error", minimum=0.0)
    logit_rms = finite(final_logits.get("rms_error"), f"{label} logit RMS error", minimum=0.0)
    prefix_match = final_logits.get("top_k_prefix_token_ids_match")
    require(
        final_logits.get("max_abs_tolerance") == 0.0625
        and final_logits.get("rms_tolerance") == 0.01
        and integer(final_logits.get("top_k_prefix"), f"{label} logit top-k prefix") == 5
        and isinstance(prefix_match, bool)
        and isinstance(final_logits.get("within_tolerance"), bool)
        and final_logits.get("within_tolerance") is (
            logit_max <= 0.0625 and logit_rms <= 0.01 and prefix_match
        ),
        f"{label} final logits canary is internally inconsistent",
    )
    return bool(final_norm["within_tolerance"] and final_logits["within_tolerance"])


def _validate_experiment(experiment: Mapping[str, Any], prompt: Mapping[str, Any], expected: Mapping[str, Any]) -> tuple[str, bool, bool, list[tuple[int, str]]]:
    expected_keys = set(EXPERIMENT_KEYS)
    if "target_token_id" in prompt:
        expected_keys.add("target_token_id_override")
    exact_keys(experiment, expected_keys, f"experiment {expected.get('id')}")
    prompt_id = text(prompt.get("id"), "prompt ID")
    require(
        experiment.get("id") == prompt_id == expected.get("id")
        and experiment.get("prompt") == prompt.get("text")
        and canonical_value_equal(experiment.get("prompt_token_ids"), prompt.get("token_ids"))
        and canonical_value_equal(experiment.get("metadata"), prompt.get("metadata")),
        f"report experiment is not value-bound to prompt: {prompt_id}",
    )
    token_ids = sequence(prompt.get("token_ids"), f"{prompt_id} token IDs")
    require(bool(token_ids), f"{prompt_id} token IDs are empty")
    final_position = len(token_ids) - 1
    require(
        canonical_value_equal(experiment.get("positions_requested"), list(POSITIONS))
        and canonical_value_equal(experiment.get("positions_resolved"), [final_position])
        and canonical_value_equal(experiment.get("capture_positions_resolved"), [final_position])
        and integer(experiment.get("final_validation_position"), f"{prompt_id} final validation position") == final_position,
        f"{prompt_id} was not replayed only at the exact final boundary",
    )
    scored = mapping(experiment.get("scored_vocabulary"), f"{prompt_id} scored vocabulary")
    scored_ids = sequence(scored.get("token_ids"), f"{prompt_id} scored token IDs")
    scored_tokens = sequence(scored.get("tokens"), f"{prompt_id} scored token strings")
    require(
        scored_ids == prompt.get("score_token_ids")
        and len(scored_ids) == len(scored_tokens)
        and len(scored_ids) == len(set(scored_ids))
        and all(isinstance(token, str) for token in scored_tokens),
        f"{prompt_id} scored vocabulary changed",
    )
    layers = [mapping(row, f"{prompt_id} layer") for row in sequence(experiment.get("layers"), f"{prompt_id} layers")]
    require(
        [integer(row.get("layer"), f"{prompt_id} layer number") for row in layers]
        == list(LAYERS),
        f"{prompt_id} replay layer band/order changed",
    )
    for layer in layers:
        exact_keys(layer, {"layer", "layer_type", "positions"}, f"{prompt_id} layer {layer.get('layer')}")
        require(isinstance(layer.get("layer_type"), str) and bool(layer.get("layer_type")), f"{prompt_id} layer type is invalid")
        positions = [mapping(row, f"{prompt_id} layer position") for row in sequence(layer.get("positions"), f"{prompt_id} layer positions")]
        require(len(positions) == 1, f"{prompt_id} layer does not contain exactly the final position")
        position = positions[0]
        exact_keys(position, {"capture_index", "token_position", "logit_lens", "jacobian_lens"}, f"{prompt_id} layer position")
        require(
            integer(position.get("capture_index"), f"{prompt_id} capture index") == 0
            and integer(position.get("token_position"), f"{prompt_id} token position") == final_position,
            f"{prompt_id} layer capture position changed",
        )
        _validate_readout(
            position.get("logit_lens"),
            label=f"{prompt_id} layer {layer.get('layer')} logit readout",
            scored_ids=scored_ids,
            scored_tokens=scored_tokens,
            fidelity_required=True,
        )
        _validate_readout(
            position.get("jacobian_lens"),
            label=f"{prompt_id} layer {layer.get('layer')} Jacobian readout",
            scored_ids=scored_ids,
            scored_tokens=scored_tokens,
            fidelity_required=True,
        )
    final_rows = [mapping(row, f"{prompt_id} reconstructed final readout") for row in sequence(experiment.get("final_model_readout"), f"{prompt_id} final readout")]
    captured_rows = [mapping(row, f"{prompt_id} captured final readout") for row in sequence(experiment.get("captured_final_model_readout"), f"{prompt_id} captured final readout")]
    require(len(final_rows) == len(captured_rows) == 1, f"{prompt_id} final readout coverage changed")
    reconstructed = _validate_readout(
        final_rows[0],
        label=f"{prompt_id} reconstructed final readout",
        scored_ids=scored_ids,
        scored_tokens=scored_tokens,
        fidelity_required=True,
    )
    captured = _validate_readout(
        captured_rows[0],
        label=f"{prompt_id} captured final readout",
        scored_ids=scored_ids,
        scored_tokens=scored_tokens,
        fidelity_required=False,
    )
    final_top1 = experiment.get("final_layer_top1_matches_greedy")
    generated_token_id = integer(experiment.get("generated_token_id"), f"{prompt_id} generated token ID", minimum=0)
    require(
        isinstance(experiment.get("generated_token"), str)
        and isinstance(experiment.get("generated_text"), str)
        and sequence(experiment.get("prompt_tokens"), f"{prompt_id} prompt tokens")
        and len(experiment["prompt_tokens"]) == len(token_ids)
        and all(isinstance(token, str) for token in experiment["prompt_tokens"])
        and len(sequence(experiment.get("position_tokens"), f"{prompt_id} position tokens")) == 1
        and isinstance(experiment["position_tokens"][0], str),
        f"{prompt_id} decoded token fields are incomplete",
    )
    expected_target_token_id = prompt.get("target_token_id", generated_token_id)
    for readout in (reconstructed, captured):
        require(
            readout.get("target_token_id") == expected_target_token_id,
            f"{prompt_id} final readout target token changed",
        )
    require(
        isinstance(final_top1, bool)
        and final_top1 is (
            reconstructed["token_ids"][0] == generated_token_id
            and captured["token_ids"][0] == generated_token_id
        ),
        f"{prompt_id} final top-1 canary is internally inconsistent",
    )
    reconstruction_within_tolerance = _validate_reconstruction_canaries(experiment, label=prompt_id)
    for field in ("generation_seconds", "readout_seconds"):
        finite(experiment.get(field), f"{prompt_id} {field}", minimum=0.0)
    for field in ("cuda_max_memory_allocated_bytes", "cuda_max_memory_reserved_bytes"):
        integer(experiment.get(field), f"{prompt_id} {field}", minimum=0)
    residual = mapping(experiment.get("residual_capture_manifest"), f"{prompt_id} residual manifest")
    exact_keys(residual, {"algorithm", "sha256", "tensor_count", "logical_bytes", "token_positions"}, f"{prompt_id} residual manifest")
    residual_sha = text(residual.get("sha256"), f"{prompt_id} residual SHA-256")
    require(
        residual.get("algorithm") == (
            "SHA-256 over length-prefixed canonical layer/shape/dtype/"
            "token-position/byte-count headers and logical row-major FP32 bytes"
        )
        and len(residual_sha) == 64
        and all(char in "0123456789abcdef" for char in residual_sha)
        and integer(residual.get("tensor_count"), f"{prompt_id} residual tensor count") == 64
        and integer(residual.get("logical_bytes"), f"{prompt_id} residual logical bytes", minimum=1) == 64 * 5120 * 4
        and residual.get("token_positions") == [final_position],
        f"{prompt_id} residual manifest is incomplete",
    )
    payload_hash = canonical_sha256(experiment)
    require(
        canonical_sha256(prompt) == expected.get("canonical_sha256")
        and _prompt_payload_pin(prompt) == expected.get("declared_payload_sha256"),
        f"{prompt_id} prompt manifest hash changed during report validation",
    )
    return (
        payload_hash,
        final_top1,
        reconstruction_within_tolerance,
        [(integer(token_id, f"{prompt_id} scored token ID", minimum=0), str(token)) for token_id, token in zip(scored_ids, scored_tokens, strict=True)],
    )


def validate_report_file(
    *,
    report_path: Path,
    prompts_path: Path,
    prompt_records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    expected_exit_code: int | None,
    expected_sha256: str | None = None,
    allow_combined: bool = False,
) -> ReportAudit:
    require(report_path.is_file() and not report_path.is_symlink(), "report path is unsafe")
    observed_sha = sha256_file(report_path)
    if expected_sha256 is not None:
        require(observed_sha == expected_sha256, "report SHA-256 changed")
    retained: dict[str, Any] = {}
    prompt_iterator = iter(iter_strict_json_array(prompts_path, "report source prompts"))
    experiment_iterator = iter(_stream_report_experiments(report_path, retained, allow_combined=allow_combined))
    sentinel = object()
    experiment_ids: list[str] = []
    payload_hashes: list[str] = []
    top1_values: list[bool] = []
    reconstruction_values: list[bool] = []
    union_ids: list[int] = []
    union_tokens: list[str] = []
    token_strings: dict[int, str] = {}
    for index, expected in enumerate(prompt_records):
        prompt = next(prompt_iterator, sentinel)
        experiment = next(experiment_iterator, sentinel)
        require(prompt is not sentinel and experiment is not sentinel, "report or source prompt row is missing")
        payload_hash, top1, reconstruction, scored_pairs = _validate_experiment(experiment, prompt, expected)
        prompt_id = str(expected.get("id"))
        require(prompt_id not in experiment_ids, "report experiment IDs repeat")
        experiment_ids.append(prompt_id)
        payload_hashes.append(payload_hash)
        top1_values.append(top1)
        reconstruction_values.append(reconstruction)
        for token_id, token in scored_pairs:
            if token_id in token_strings:
                require(token_strings[token_id] == token, "one scored token ID decodes inconsistently")
            else:
                token_strings[token_id] = token
                union_ids.append(token_id)
                union_tokens.append(token)
    require(next(prompt_iterator, sentinel) is sentinel, "source prompt bundle has trailing rows")
    require(next(experiment_iterator, sentinel) is sentinel, "report has trailing experiment rows")
    _validate_report_pins(retained, protocol)
    assertions = mapping(retained.get("assertions"), "report assertions")
    exact_keys(assertions, REPORT_ASSERTION_KEYS, "report assertions")
    require(
        assertions.get("lens_hash_matches") is True
        and assertions.get("lens_metadata_matches") is True
        and assertions.get("model_architecture_matches") is True
        and assertions.get("all_final_layer_top1_match_greedy") is all(top1_values)
        and assertions.get("all_final_adapter_reconstructions_within_tolerance") is all(reconstruction_values),
        "report assertions do not aggregate experiment canaries exactly",
    )
    status = text(retained.get("status"), "report status")
    expected_status = "passed" if all(top1_values) and all(reconstruction_values) else "failed"
    require(status == expected_status, "report status does not match strict canaries")
    if expected_exit_code is not None:
        require(expected_exit_code in {0, 1}, "runner exit code is unsupported")
        require(
            (expected_exit_code == 0 and status == "passed")
            or (expected_exit_code == 1 and status == "failed"),
            "runner exit code does not match complete schema-3 strict-canary status",
        )
    scored_root = mapping(retained.get("scored_vocabulary"), "report scored vocabulary")
    exact_keys(scored_root, {"token_ids", "tokens", "scope", "union_token_ids", "union_tokens"}, "report scored vocabulary")
    require(
        scored_root.get("token_ids") == []
        and scored_root.get("tokens") == []
        and scored_root.get("scope") == "global_plus_per_experiment"
        and canonical_value_equal(scored_root.get("union_token_ids"), union_ids)
        and canonical_value_equal(scored_root.get("union_tokens"), union_tokens),
        "report scored-vocabulary union changed",
    )
    require(len(experiment_ids) == len(prompt_records), "report experiment count changed")
    require(sha256_file(report_path) == observed_sha, "report bytes changed during validation")
    return ReportAudit(
        path=report_path,
        sha256=observed_sha,
        experiment_count=len(experiment_ids),
        experiment_ids=tuple(experiment_ids),
        experiment_payload_sha256s=tuple(payload_hashes),
        metadata=dict(retained),
        assertions={key: bool(assertions[key]) for key in sorted(REPORT_ASSERTION_KEYS)},
        status=status,
        scored_vocabulary=dict(scored_root),
    )


def replay_command_arguments(chunk_path: Path, report_path: Path) -> list[str]:
    return [
        str(JLENS_SHELL_RUNNER),
        "--lens-kind", "public",
        "--prompts-file", str(chunk_path),
        "--layers", ",".join(str(layer) for layer in LAYERS),
        "--positions=-1",
        "--top-k", str(TOP_K),
        "--max-model-len", str(EXPECTED_REPLAY_RUNTIME_PIN["max_model_len"]),
        "--max-num-batched-tokens", str(EXPECTED_REPLAY_RUNTIME_PIN["max_num_batched_tokens"]),
        "--mamba-block-size", str(EXPECTED_REPLAY_RUNTIME_PIN["mamba_block_size"]),
        "--enable-prefix-caching",
        "--kv-cache-dtype", str(EXPECTED_REPLAY_RUNTIME_PIN["kv_cache_dtype"]),
        "--kv-offloading-size", str(int(EXPECTED_REPLAY_RUNTIME_PIN["kv_offloading_size"])),
        "--kv-offloading-backend", str(EXPECTED_REPLAY_RUNTIME_PIN["kv_offloading_backend"]),
        "--stream-final-only",
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--output", str(report_path),
    ]


def _chunk_prompt_records(chunk: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(mapping(row, "chunk prompt record") for row in sequence(chunk.get("prompts"), "chunk prompt records"))


def _run_manifest_base(partition: ValidatedPartition) -> dict[str, Any]:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "id": "swe-task-state-v3-bounded-memory-replay-run",
        "status": "running",
        "chunk_manifest": {
            "path": CHUNK_MANIFEST_NAME,
            "sha256": partition.manifest_sha256,
        },
        "runner": {
            "orchestrator": dict(mapping(partition.manifest["inputs"]["replay_pipeline"], "replay pipeline pin")),
            "shell_wrapper": dict(mapping(partition.manifest["inputs"]["replay_shell_wrapper"], "replay wrapper pin")),
            "shell": dict(mapping(partition.manifest["inputs"]["jlens_shell_runner"], "shell runner pin")),
            "python": dict(mapping(partition.manifest["inputs"]["jlens_python_runner"], "Python runner pin")),
            "stdout": "redirected_to_devnull_to_avoid_duplicate_enormous_report_output",
            "sequential": True,
            "interrupted_current_chunk_recovery": (
                "fail_closed_on_unrecorded_report_or_log; preserve orphan bytes for audit; "
                "manual move-aside is required before retry; never overwrite, delete, skip, or auto-accept"
            ),
            "accepted_exit_codes": {
                "0": "complete_schema3_report_with_passed_strict_canaries",
                "1": "complete_schema3_report_with_expected_failed_strict_canary",
            },
        },
        "chunk_count": partition.manifest["chunk_count"],
        "completed_chunk_count": 0,
        "chunks": [],
    }


def _validate_run_entry(
    entry: Mapping[str, Any],
    *,
    chunk: Mapping[str, Any],
    partition: ValidatedPartition,
    protocol: Mapping[str, Any],
) -> ReportAudit:
    exact_keys(
        entry,
        {
            "index", "id", "prompts", "report", "stderr_log", "exit_code",
            "accepted_terminal", "report_status", "report_assertions",
            "experiment_count", "master_task_range_inclusive",
            "master_prompt_range_inclusive", "experiment_payload_sha256s",
            "experiment_payload_sha256s_sha256",
        },
        "run chunk entry",
    )
    index = integer(entry.get("index"), "run chunk index", minimum=0)
    require(index == chunk.get("index") and entry.get("id") == chunk.get("id"), "run entry chunk identity changed")
    prompts_record = mapping(entry.get("prompts"), "run prompt record")
    report_record = mapping(entry.get("report"), "run report record")
    log_record = mapping(entry.get("stderr_log"), "run log record")
    exact_keys(prompts_record, {"path", "sha256"}, "run prompt record")
    exact_keys(report_record, {"path", "sha256"}, "run report record")
    exact_keys(log_record, {"path", "sha256"}, "run stderr-log record")
    require(prompts_record == {"path": chunk.get("prompts_path"), "sha256": chunk.get("prompts_sha256")}, "run entry prompt pin changed")
    require(
        report_record.get("path") == f"reports/{chunk.get('id')}-report.json"
        and log_record.get("path") == f"logs/{chunk.get('id')}.stderr.log",
        f"chunk {index} report/log path changed",
    )
    report_path = safe_regular_file(partition.replay_root, report_record.get("path"), f"chunk {index} report")
    log_path = safe_regular_file(partition.replay_root, log_record.get("path"), f"chunk {index} stderr log")
    require(sha256_file(log_path) == log_record.get("sha256"), f"chunk {index} stderr log changed")
    chunk_path = safe_regular_file(partition.replay_root, chunk.get("prompts_path"), f"chunk {index} prompts")
    audit = validate_report_file(
        report_path=report_path,
        prompts_path=chunk_path,
        prompt_records=_chunk_prompt_records(chunk),
        protocol=protocol,
        expected_exit_code=integer(entry.get("exit_code"), "runner exit code"),
        expected_sha256=text(report_record.get("sha256"), "report SHA-256"),
    )
    require(
        entry.get("accepted_terminal") is True
        and entry.get("report_status") == audit.status
        and entry.get("report_assertions") == audit.assertions
        and entry.get("experiment_count") == audit.experiment_count
        and entry.get("master_task_range_inclusive") == chunk.get("master_task_range_inclusive")
        and entry.get("master_prompt_range_inclusive") == chunk.get("master_prompt_range_inclusive")
        and entry.get("experiment_payload_sha256s") == list(audit.experiment_payload_sha256s)
        and entry.get("experiment_payload_sha256s_sha256") == canonical_sha256(list(audit.experiment_payload_sha256s)),
        f"run entry report provenance changed: chunk {index}",
    )
    return audit


def validate_run_manifest(
    *,
    partition: ValidatedPartition,
    protocol: Mapping[str, Any],
    require_complete: bool,
) -> tuple[Mapping[str, Any], tuple[ReportAudit, ...]]:
    path = partition.replay_root / RUN_MANIFEST_NAME
    value = mapping(strict_json_file(path, "run manifest"), "run manifest")
    exact_keys(value, set(_run_manifest_base(partition)), "run manifest")
    require(
        value.get("schema_version") == RUN_MANIFEST_SCHEMA
        and value.get("id") == "swe-task-state-v3-bounded-memory-replay-run"
        and value.get("chunk_manifest") == {"path": CHUNK_MANIFEST_NAME, "sha256": partition.manifest_sha256}
        and value.get("runner") == _run_manifest_base(partition)["runner"]
        and value.get("chunk_count") == partition.manifest.get("chunk_count"),
        "run manifest identity or frozen runner binding changed",
    )
    entries = [mapping(row, f"run chunk {index}") for index, row in enumerate(sequence(value.get("chunks"), "run chunks"))]
    require(value.get("completed_chunk_count") == len(entries), "run manifest completed count changed")
    chunks = sequence(partition.manifest.get("chunks"), "partition chunks")
    require(len(entries) <= len(chunks), "run manifest has trailing chunks")
    audits = tuple(
        _validate_run_entry(entry, chunk=mapping(chunks[index], f"chunk {index}"), partition=partition, protocol=protocol)
        for index, entry in enumerate(entries)
    )
    if audits:
        invariant_model = mapping(audits[0].metadata.get("model"), "chunk 0 model metadata")
        invariant_lens = mapping(audits[0].metadata.get("lens"), "chunk 0 lens metadata")
        invariant_runtime = dict(mapping(audits[0].metadata.get("runtime"), "chunk 0 runtime metadata"))
        invariant_runtime.pop("model_load_seconds", None)
        for index, audit in enumerate(audits[1:], start=1):
            runtime = dict(mapping(audit.metadata.get("runtime"), f"chunk {index} runtime metadata"))
            runtime.pop("model_load_seconds", None)
            require(
                canonical_value_equal(mapping(audit.metadata.get("model"), f"chunk {index} model metadata"), invariant_model),
                f"chunk {index} invariant model metadata differs from chunk 0",
            )
            require(
                canonical_value_equal(mapping(audit.metadata.get("lens"), f"chunk {index} lens metadata"), invariant_lens),
                f"chunk {index} invariant lens metadata differs from chunk 0",
            )
            require(canonical_value_equal(runtime, invariant_runtime), f"chunk {index} invariant runtime metadata differs from chunk 0")
    if require_complete:
        require(value.get("status") == "complete" and len(entries) == len(chunks), "run manifest is incomplete")
    else:
        expected_status = "complete" if len(entries) == len(chunks) else "running"
        require(value.get("status") == expected_status, "run manifest status changed")
    return value, audits


def run_replay_chunks(*, partition: ValidatedPartition, protocol: Mapping[str, Any]) -> Path:
    """Run unrecorded chunks sequentially; an authenticated prefix is resumable."""

    reports_root = partition.replay_root / "reports"
    logs_root = partition.replay_root / "logs"
    for directory in (reports_root, logs_root):
        if not directory.exists():
            directory.mkdir()
        _assert_root(directory, f"{directory.name} root")
    run_path = _safe_direct_child(partition.replay_root, RUN_MANIFEST_NAME, "run manifest")
    if run_path.exists():
        run_manifest, _audits = validate_run_manifest(partition=partition, protocol=protocol, require_complete=False)
        run_value = dict(run_manifest)
    else:
        run_value = _run_manifest_base(partition)
        atomic_write_json(run_path, run_value)
    chunks = [mapping(row, f"chunk {index}") for index, row in enumerate(sequence(partition.manifest.get("chunks"), "chunks"))]
    completed = integer(run_value.get("completed_chunk_count"), "completed chunks", minimum=0)
    for chunk in chunks[completed:]:
        index = integer(chunk.get("index"), "chunk index", minimum=0)
        chunk_id = text(chunk.get("id"), "chunk ID")
        chunk_path = safe_regular_file(partition.replay_root, chunk.get("prompts_path"), f"chunk {index} prompts")
        report_relative = f"reports/{chunk_id}-report.json"
        log_relative = f"logs/{chunk_id}.stderr.log"
        report_path = partition.replay_root / report_relative
        log_path = partition.replay_root / log_relative
        require(
            not report_path.exists() and not report_path.is_symlink(),
            f"chunk {index} has an unrecorded/orphaned report; fail-closed recovery requires preserving it for audit and manually moving it aside before retry",
        )
        require(
            not log_path.exists() and not log_path.is_symlink(),
            f"chunk {index} has an unrecorded/orphaned stderr log; fail-closed recovery requires preserving it for audit and manually moving it aside before retry",
        )
        command = replay_command_arguments(chunk_path, report_path)
        with log_path.open("xb") as stderr_handle, Path(os.devnull).open("wb") as stdout_handle:
            completed_process = subprocess.run(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        exit_code = completed_process.returncode
        require(exit_code in {0, 1}, f"chunk {index} runner exited unsupported status {exit_code}")
        require(report_path.is_file() and not report_path.is_symlink(), f"chunk {index} runner did not emit a regular report")
        audit = validate_report_file(
            report_path=report_path,
            prompts_path=chunk_path,
            prompt_records=_chunk_prompt_records(chunk),
            protocol=protocol,
            expected_exit_code=exit_code,
        )
        entry = {
            "index": index,
            "id": chunk_id,
            "prompts": {"path": chunk.get("prompts_path"), "sha256": chunk.get("prompts_sha256")},
            "report": {"path": report_relative, "sha256": audit.sha256},
            "stderr_log": {"path": log_relative, "sha256": sha256_file(log_path)},
            "exit_code": exit_code,
            "accepted_terminal": True,
            "report_status": audit.status,
            "report_assertions": dict(audit.assertions),
            "experiment_count": audit.experiment_count,
            "master_task_range_inclusive": chunk.get("master_task_range_inclusive"),
            "master_prompt_range_inclusive": chunk.get("master_prompt_range_inclusive"),
            "experiment_payload_sha256s": list(audit.experiment_payload_sha256s),
            "experiment_payload_sha256s_sha256": canonical_sha256(list(audit.experiment_payload_sha256s)),
        }
        sequence(run_value["chunks"], "run chunks").append(entry)
        run_value["completed_chunk_count"] = len(run_value["chunks"])
        run_value["status"] = "complete" if len(run_value["chunks"]) == len(chunks) else "running"
        atomic_write_json(run_path, run_value)
    validate_run_manifest(partition=partition, protocol=protocol, require_complete=True)
    return run_path


def _source_provenance(
    *,
    partition: ValidatedPartition,
    run_manifest: Mapping[str, Any],
    audits: Sequence[ReportAudit],
) -> dict[str, Any]:
    entries = sequence(run_manifest.get("chunks"), "run chunks")
    chunks = sequence(partition.manifest.get("chunks"), "partition chunks")
    sources: list[dict[str, Any]] = []
    invariant_model = mapping(audits[0].metadata.get("model"), "invariant model metadata")
    invariant_lens = mapping(audits[0].metadata.get("lens"), "invariant lens metadata")
    invariant_runtime = dict(mapping(audits[0].metadata.get("runtime"), "invariant runtime metadata"))
    invariant_runtime.pop("model_load_seconds", None)
    for entry, chunk, audit in zip(entries, chunks, audits, strict=True):
        sources.append(
            {
                "index": entry["index"],
                "id": entry["id"],
                "prompts": dict(mapping(entry["prompts"], "source prompts")),
                "report": dict(mapping(entry["report"], "source report")),
                "stderr_log": dict(mapping(entry["stderr_log"], "source log")),
                "exit_code": entry["exit_code"],
                "status": audit.status,
                "assertions": dict(audit.assertions),
                "task_count": chunk["task_count"],
                "experiment_count": audit.experiment_count,
                "master_task_range_inclusive": list(chunk["master_task_range_inclusive"]),
                "master_prompt_range_inclusive": list(chunk["master_prompt_range_inclusive"]),
                "started_at": audit.metadata["started_at"],
                "completed_at": audit.metadata["completed_at"],
                "elapsed_seconds": audit.metadata["elapsed_seconds"],
                "model_frozen_pin_sha256": canonical_sha256(EXPECTED_MODEL_PIN),
                "lens_frozen_pin_sha256": canonical_sha256(EXPECTED_PUBLIC_LENS_PIN),
                "runtime_frozen_pin_sha256": canonical_sha256(EXPECTED_REPORT_RUNTIME),
                "full_invariant_model_metadata_sha256": canonical_sha256(invariant_model),
                "full_invariant_lens_metadata_sha256": canonical_sha256(invariant_lens),
                "full_invariant_runtime_metadata_excluding_model_load_seconds_sha256": canonical_sha256(invariant_runtime),
                "experiment_payload_sha256s": list(audit.experiment_payload_sha256s),
                "experiment_payload_sha256s_sha256": canonical_sha256(list(audit.experiment_payload_sha256s)),
            }
        )
    return {
        "schema_version": 1,
        "kind": "swe-task-state-v3-lossless-bounded-memory-chunk-merge",
        "chunk_manifest": {"path": CHUNK_MANIFEST_NAME, "sha256": partition.manifest_sha256},
        "run_manifest": {"path": RUN_MANIFEST_NAME, "sha256": sha256_file(partition.replay_root / RUN_MANIFEST_NAME)},
        "master_prompt_bundle": dict(mapping(partition.manifest["inputs"]["master_prompt_bundle"], "master prompt pin")),
        "source_chunk_count": len(sources),
        "source_experiment_count": sum(source["experiment_count"] for source in sources),
        "source_chunks": sources,
        "cross_chunk_invariant_metadata": {
            "model": "full_value_identical",
            "lens": "full_value_identical",
            "runtime": "full_value_identical_except_model_load_seconds",
            "model_sha256": canonical_sha256(invariant_model),
            "lens_sha256": canonical_sha256(invariant_lens),
            "runtime_excluding_model_load_seconds_sha256": canonical_sha256(invariant_runtime),
        },
        "merge_contract": {
            "source_experiment_order": "exact_master_prompt_order",
            "experiment_values": "value_identical_to_source_chunk_records",
            "canonical_experiment_payload_sha256s_preserved": True,
            "scores_metadata_and_token_ids_recomputed": False,
            "scores_averaged": False,
        },
    }


def _aggregate_scored_vocabulary(audits: Sequence[ReportAudit]) -> dict[str, Any]:
    union_ids: list[int] = []
    union_tokens: list[str] = []
    token_strings: dict[int, str] = {}
    for audit in audits:
        scored = audit.scored_vocabulary
        for token_id, token in zip(scored["union_token_ids"], scored["union_tokens"], strict=True):
            token_id = integer(token_id, "union token ID", minimum=0)
            token = str(token)
            if token_id in token_strings:
                require(token_strings[token_id] == token, "chunk reports decode one token inconsistently")
            else:
                token_strings[token_id] = token
                union_ids.append(token_id)
                union_tokens.append(token)
    return {
        "token_ids": [],
        "tokens": [],
        "scope": "global_plus_per_experiment",
        "union_token_ids": union_ids,
        "union_tokens": union_tokens,
    }


def _write_streamed_combined_report(
    *,
    output_path: Path,
    prefix: Mapping[str, Any],
    partition: ValidatedPartition,
    audits: Sequence[ReportAudit],
) -> tuple[list[str], list[str]]:
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    require(not output_path.exists() and not output_path.is_symlink(), "combined report already exists")
    require(not temporary.exists() and not temporary.is_symlink(), "combined report temporary exists")
    merged_ids: list[str] = []
    merged_hashes: list[str] = []
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write("{\n")
            items = list(prefix.items())
            for key, value in items:
                handle.write(json.dumps(str(key), ensure_ascii=False))
                handle.write(":")
                handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
                handle.write(",\n")
            handle.write('"experiments":[\n')
            first = True
            chunks = sequence(partition.manifest.get("chunks"), "partition chunks")
            for chunk, audit in zip(chunks, audits, strict=True):
                report_record = mapping(sequence(strict_json_file(partition.replay_root / RUN_MANIFEST_NAME, "run manifest")["chunks"], "run chunks")[chunk["index"]]["report"], "report record")
                report_path = safe_regular_file(partition.replay_root, report_record["path"], "source chunk report")
                retained: dict[str, Any] = {}
                source_hashes = iter(audit.experiment_payload_sha256s)
                for experiment in _stream_report_experiments(report_path, retained, allow_combined=False):
                    expected_hash = next(source_hashes, None)
                    observed_hash = canonical_sha256(experiment)
                    require(expected_hash == observed_hash, "source experiment changed between validation and merge")
                    if not first:
                        handle.write(",\n")
                    handle.write(json.dumps(experiment, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
                    first = False
                    merged_ids.append(text(experiment.get("id"), "merged experiment ID"))
                    merged_hashes.append(observed_hash)
                require(next(source_hashes, None) is None, "source report lost experiments during merge")
                require(sha256_file(report_path) == audit.sha256, "source report bytes changed during merge")
            handle.write("\n]}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output_path, follow_symlinks=False)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return merged_ids, merged_hashes


def _combined_report_prefix(
    *,
    partition: ValidatedPartition,
    run_manifest: Mapping[str, Any],
    audits: Sequence[ReportAudit],
) -> dict[str, Any]:
    require(bool(audits), "no chunk reports are available to merge")
    first_metadata = audits[0].metadata
    assertions = {
        "lens_hash_matches": True,
        "lens_metadata_matches": True,
        "model_architecture_matches": True,
        "all_final_layer_top1_match_greedy": all(audit.assertions["all_final_layer_top1_match_greedy"] for audit in audits),
        "all_final_adapter_reconstructions_within_tolerance": all(audit.assertions["all_final_adapter_reconstructions_within_tolerance"] for audit in audits),
    }
    status = "passed" if assertions["all_final_layer_top1_match_greedy"] and assertions["all_final_adapter_reconstructions_within_tolerance"] else "failed"
    provenance = _source_provenance(partition=partition, run_manifest=run_manifest, audits=audits)
    return {
        "schema_version": REPORT_SCHEMA,
        "score_encoding": SCORE_ENCODING,
        "status": status,
        "started_at": first_metadata["started_at"],
        "completed_at": audits[-1].metadata["completed_at"],
        "elapsed_seconds": round(sum(finite(audit.metadata["elapsed_seconds"], "chunk elapsed", minimum=0.0) for audit in audits), 6),
        "host": first_metadata["host"],
        "model": first_metadata["model"],
        "lens": first_metadata["lens"],
        "runtime": first_metadata["runtime"],
        "scored_vocabulary": _aggregate_scored_vocabulary(audits),
        "assertions": assertions,
        "combined_chunk_provenance": provenance,
    }


def _validate_combined_report_exact(
    *,
    report_path: Path,
    prefix: Mapping[str, Any],
    partition: ValidatedPartition,
    protocol: Mapping[str, Any],
    audits: Sequence[ReportAudit],
) -> ReportAudit:
    expected_ids = [str(row.get("id")) for row in partition.prompt_records]
    expected_hashes = [value for audit in audits for value in audit.experiment_payload_sha256s]
    combined_audit = validate_report_file(
        report_path=report_path,
        prompts_path=partition.prompts_path,
        prompt_records=partition.prompt_records,
        protocol=protocol,
        expected_exit_code=None,
        allow_combined=True,
    )
    require(
        combined_audit.experiment_ids == tuple(expected_ids)
        and combined_audit.experiment_payload_sha256s == tuple(expected_hashes)
        and combined_audit.status == prefix.get("status")
        and canonical_value_equal(combined_audit.metadata, prefix),
        "combined report does not reproduce exact source experiments",
    )
    return combined_audit


def _expected_merge_manifest(
    *,
    partition: ValidatedPartition,
    combined_audit: ReportAudit,
    audits: Sequence[ReportAudit],
) -> dict[str, Any]:
    expected_ids = [str(row.get("id")) for row in partition.prompt_records]
    expected_hashes = [value for audit in audits for value in audit.experiment_payload_sha256s]
    return {
        "schema_version": MERGE_MANIFEST_SCHEMA,
        "id": "swe-task-state-v3-bounded-memory-replay-merge",
        "status": "complete",
        "chunk_manifest": {"path": CHUNK_MANIFEST_NAME, "sha256": partition.manifest_sha256},
        "run_manifest": {"path": RUN_MANIFEST_NAME, "sha256": sha256_file(partition.replay_root / RUN_MANIFEST_NAME)},
        "combined_report": {"path": MERGED_REPORT_NAME, "sha256": combined_audit.sha256, "schema_version": REPORT_SCHEMA, "status": combined_audit.status},
        "task_count": partition.manifest["task_count"],
        "experiment_count": combined_audit.experiment_count,
        "ordered_experiment_ids_sha256": canonical_sha256(expected_ids),
        "ordered_experiment_payload_sha256s": expected_hashes,
        "ordered_experiment_payload_sha256s_sha256": canonical_sha256(expected_hashes),
        "source_chunk_count": len(audits),
        "value_identical_lossless_merge_validated": True,
        "scores_metadata_and_token_ids_recomputed": False,
    }


def _publish_new_file(staged: Path, target: Path, label: str) -> None:
    require(
        staged.is_file()
        and not staged.is_symlink()
        and staged.parent == target.parent
        and not target.exists()
        and not target.is_symlink(),
        f"{label} staging/publication paths are unsafe",
    )
    os.link(staged, target, follow_symlinks=False)


def merge_replay_reports(*, partition: ValidatedPartition, protocol: Mapping[str, Any]) -> tuple[Path, Path]:
    run_manifest, audits = validate_run_manifest(partition=partition, protocol=protocol, require_complete=True)
    require(bool(audits), "no chunk reports are available to merge")
    prefix = _combined_report_prefix(
        partition=partition,
        run_manifest=run_manifest,
        audits=audits,
    )
    output_path = _safe_direct_child(partition.replay_root, MERGED_REPORT_NAME, "combined report")
    merge_path = _safe_direct_child(partition.replay_root, MERGE_MANIFEST_NAME, "merge manifest")
    staged_path = _safe_direct_child(
        partition.replay_root,
        STAGED_MERGED_REPORT_NAME,
        "staged combined report",
    )
    require(
        not merge_path.exists() or output_path.exists(),
        "merge manifest exists without its combined report",
    )
    if output_path.exists():
        require(
            output_path.is_file() and not output_path.is_symlink(),
            "combined report orphan is unsafe",
        )
        combined_audit = _validate_combined_report_exact(
            report_path=output_path,
            prefix=prefix,
            partition=partition,
            protocol=protocol,
            audits=audits,
        )
    else:
        require(not merge_path.exists(), "merge manifest exists before combined report publication")
        if not staged_path.exists():
            merged_ids, merged_hashes = _write_streamed_combined_report(
                output_path=staged_path,
                prefix=prefix,
                partition=partition,
                audits=audits,
            )
            expected_ids = [str(row.get("id")) for row in partition.prompt_records]
            expected_hashes = [
                value for audit in audits for value in audit.experiment_payload_sha256s
            ]
            require(
                merged_ids == expected_ids and merged_hashes == expected_hashes,
                "staged experiment order or canonical payload hashes changed",
            )
        require(
            staged_path.is_file() and not staged_path.is_symlink(),
            "staged combined report orphan is unsafe",
        )
        combined_audit = _validate_combined_report_exact(
            report_path=staged_path,
            prefix=prefix,
            partition=partition,
            protocol=protocol,
            audits=audits,
        )
        _publish_new_file(staged_path, output_path, "combined report")
        require(
            sha256_file(output_path) == combined_audit.sha256,
            "published combined report differs from validated staging bytes",
        )
    merge_manifest = _expected_merge_manifest(
        partition=partition,
        combined_audit=combined_audit,
        audits=audits,
    )
    if merge_path.exists():
        existing = mapping(strict_json_file(merge_path, "merge manifest"), "merge manifest")
        require(existing == merge_manifest, "existing merge manifest differs from full revalidation")
    else:
        atomic_write_new_json(merge_path, merge_manifest)
    require(
        sha256_file(output_path) == combined_audit.sha256
        and mapping(strict_json_file(merge_path, "merge manifest"), "merge manifest")
        == merge_manifest,
        "published merge receipt changed after publication",
    )
    if staged_path.exists() and not staged_path.is_symlink():
        require(
            sha256_file(staged_path) == combined_audit.sha256,
            "staged combined report changed after publication",
        )
        staged_path.unlink()
    return output_path, merge_path


def _production_partition(authenticated: AuthenticatedBundle) -> ValidatedPartition:
    return validate_chunk_manifest(
        manifest_path=V3_REPLAY_ROOT / CHUNK_MANIFEST_NAME,
        replay_root=V3_REPLAY_ROOT,
        prompts_path=DEFAULT_PROMPTS,
        expected_inputs=authenticated.inputs,
        expected_task_ids=authenticated.task_ids,
    )


def validate_merge_receipt(
    *,
    report_path: Path = V3_REPLAY_ROOT / MERGED_REPORT_NAME,
    merge_manifest_path: Path = V3_REPLAY_ROOT / MERGE_MANIFEST_NAME,
    replay_root: Path = V3_REPLAY_ROOT,
    prompts_path: Path = DEFAULT_PROMPTS,
    summary_path: Path = DEFAULT_SUMMARY,
) -> ValidatedReplayMerge:
    """Revalidate the entire canonical split/run/merge chain for analyzer use."""

    require(
        replay_root.absolute() == V3_REPLAY_ROOT.absolute()
        and replay_root.resolve(strict=True) == V3_REPLAY_ROOT.absolute(),
        "merge receipt replay root is not the exact canonical V3 replay namespace",
    )
    _assert_root(replay_root, "canonical replay root")
    expected_report = replay_root / MERGED_REPORT_NAME
    expected_merge = replay_root / MERGE_MANIFEST_NAME
    require(
        report_path.absolute() == expected_report.absolute()
        and report_path.resolve(strict=True) == expected_report.absolute()
        and merge_manifest_path.absolute() == expected_merge.absolute()
        and merge_manifest_path.resolve(strict=True) == expected_merge.absolute(),
        "analyzer report or merge-manifest path is not canonical",
    )
    report_path = safe_regular_file(replay_root, MERGED_REPORT_NAME, "combined report")
    merge_manifest_path = safe_regular_file(
        replay_root, MERGE_MANIFEST_NAME, "merge manifest"
    )
    observed_merge_sha256 = sha256_file(merge_manifest_path)
    staged_path = replay_root / STAGED_MERGED_REPORT_NAME
    require(
        not staged_path.exists() and not staged_path.is_symlink(),
        "validated merge still has an unpublished staged report",
    )
    authenticated = authenticate_production_bundle(prompts_path, summary_path)
    partition = _production_partition(authenticated)
    run_manifest, audits = validate_run_manifest(
        partition=partition,
        protocol=authenticated.protocol,
        require_complete=True,
    )
    require(bool(audits), "merge receipt has no authenticated source chunk reports")
    prefix = _combined_report_prefix(
        partition=partition,
        run_manifest=run_manifest,
        audits=audits,
    )
    combined_audit = _validate_combined_report_exact(
        report_path=report_path,
        prefix=prefix,
        partition=partition,
        protocol=authenticated.protocol,
        audits=audits,
    )
    expected_manifest = _expected_merge_manifest(
        partition=partition,
        combined_audit=combined_audit,
        audits=audits,
    )
    observed_manifest = mapping(
        strict_json_file(merge_manifest_path, "merge manifest"),
        "merge manifest",
    )
    require(
        observed_manifest == expected_manifest,
        "merge manifest differs from full source-report and combined-report revalidation",
    )
    receipt_input = mapping(
        authenticated.inputs.get("materialization_receipt"),
        "chunk materialization receipt input",
    )
    freeze_input = mapping(
        authenticated.inputs.get("materialization_freeze"),
        "chunk materialization freeze input",
    )
    require(
        mapping(partition.manifest.get("inputs"), "chunk manifest inputs")
        == authenticated.inputs
        and receipt_input.get("sha256") == sha256_file(DEFAULT_MATERIALIZATION_RECEIPT)
        and freeze_input.get("exact_child_receipt_only_commit_validated") is True
        and freeze_input.get("deterministic_rematerialization_required_before_split")
        is True,
        "merge chain is not bound to the exact frozen materialization receipt",
    )
    source_commit = text(
        freeze_input.get("source_freeze_git_commit"), "source-freeze commit"
    )
    data_commit = text(
        freeze_input.get("data_freeze_git_commit"), "data-freeze commit"
    )
    require(
        len(source_commit) == len(data_commit) == 40,
        "materialization freeze commits are malformed",
    )
    require(
        sha256_file(report_path) == combined_audit.sha256
        and sha256_file(merge_manifest_path) == observed_merge_sha256,
        "merge outputs changed during final validation",
    )
    return ValidatedReplayMerge(
        replay_root=replay_root,
        report_path=report_path,
        report_sha256=combined_audit.sha256,
        merge_manifest_path=merge_manifest_path,
        merge_manifest_sha256=observed_merge_sha256,
        experiment_count=combined_audit.experiment_count,
        prompt_bundle_sha256=text(
            mapping(authenticated.inputs["master_prompt_bundle"], "master prompt pin").get(
                "sha256"
            ),
            "master prompt SHA-256",
        ),
        materialization_receipt_sha256=text(
            receipt_input.get("sha256"), "materialization receipt SHA-256"
        ),
        source_freeze_git_commit=source_commit,
        data_freeze_git_commit=data_commit,
    )


def split_command(args: argparse.Namespace) -> int:
    authenticated = authenticate_production_bundle(
        args.prompts,
        args.prompts_summary,
        verify_rematerialization=True,
    )
    require(args.replay_root.absolute() == V3_REPLAY_ROOT.absolute(), "production replay root must be the dedicated V3 namespace")
    manifest_path = create_chunk_manifest(
        prompts_path=args.prompts,
        replay_root=args.replay_root,
        task_ids=authenticated.task_ids,
        inputs=authenticated.inputs,
        max_tasks_per_chunk=args.tasks_per_chunk,
    )
    partition = _production_partition(authenticated)
    require(partition.manifest_path == manifest_path, "fresh chunk manifest path changed")
    print(json.dumps({"chunk_manifest": str(manifest_path), "sha256": partition.manifest_sha256, "chunks": partition.manifest["chunk_count"], "tasks": partition.manifest["task_count"], "prompts": partition.manifest["prompt_count"]}, sort_keys=True))
    return 0


def run_command(args: argparse.Namespace) -> int:
    authenticated = authenticate_production_bundle(args.prompts, args.prompts_summary)
    partition = _production_partition(authenticated)
    path = run_replay_chunks(partition=partition, protocol=authenticated.protocol)
    value, _audits = validate_run_manifest(partition=partition, protocol=authenticated.protocol, require_complete=True)
    print(json.dumps({"run_manifest": str(path), "sha256": sha256_file(path), "status": value["status"], "chunks": value["completed_chunk_count"]}, sort_keys=True))
    return 0


def merge_command(args: argparse.Namespace) -> int:
    authenticated = authenticate_production_bundle(args.prompts, args.prompts_summary)
    partition = _production_partition(authenticated)
    report_path, manifest_path = merge_replay_reports(partition=partition, protocol=authenticated.protocol)
    print(json.dumps({"combined_report": str(report_path), "combined_report_sha256": sha256_file(report_path), "merge_manifest": str(manifest_path), "merge_manifest_sha256": sha256_file(manifest_path)}, sort_keys=True))
    return 0


def validate_command(args: argparse.Namespace) -> int:
    validated = validate_merge_receipt(
        report_path=args.report,
        merge_manifest_path=args.merge_manifest,
        replay_root=args.replay_root,
        prompts_path=args.prompts,
        summary_path=args.prompts_summary,
    )
    print(
        json.dumps(
            {
                "combined_report": str(validated.report_path),
                "combined_report_sha256": validated.report_sha256,
                "merge_manifest": str(validated.merge_manifest_path),
                "merge_manifest_sha256": validated.merge_manifest_sha256,
                "materialization_receipt_sha256": validated.materialization_receipt_sha256,
                "experiments": validated.experiment_count,
            },
            sort_keys=True,
        )
    )
    return 0


def all_command(args: argparse.Namespace) -> int:
    split_command(args)
    run_command(args)
    return merge_command(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("split", split_command),
        ("run", run_command),
        ("merge", merge_command),
        ("validate", validate_command),
        ("all", all_command),
    ):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
        subparser.add_argument("--prompts-summary", type=Path, default=DEFAULT_SUMMARY)
        subparser.set_defaults(handler=handler)
        if name in {"split", "all"}:
            subparser.add_argument("--replay-root", type=Path, default=V3_REPLAY_ROOT)
            subparser.add_argument("--tasks-per-chunk", type=int, default=MAX_TASKS_PER_CHUNK)
        elif name == "validate":
            subparser.add_argument("--replay-root", type=Path, default=V3_REPLAY_ROOT)
            subparser.add_argument(
                "--report",
                type=Path,
                default=V3_REPLAY_ROOT / MERGED_REPORT_NAME,
            )
            subparser.add_argument(
                "--merge-manifest",
                type=Path,
                default=V3_REPLAY_ROOT / MERGE_MANIFEST_NAME,
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ReplayValidationError, FileNotFoundError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
