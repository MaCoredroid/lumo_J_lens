#!/usr/bin/env python3
"""Fail-closed native vLLM adapter for the bounded V3 annotation runner.

The production entrypoint in this module owns every runtime import, tokenizer,
vLLM engine, and generation session.  It accepts no backend, callback, engine,
tokenizer, or output object from a caller.  The checked-in configuration is
deliberately false-state and grants no authority.  The entrypoint can proceed
only after authenticating both those exact config bytes and a separate external
launch file; it reaches no model snapshot, tokenizer, GPU, or output beforehand.

Future execution requires a separately reviewed launch-authorization file and
an out-of-band expected SHA-256 supplied to the entrypoint.  The adapter config
remains false-state and cannot authorize execution by mutation or flag flip.

The real path submits the bounded runner's exact ``submitted_prompt_token_ids``
as ``vllm.inputs.TokensPrompt(prompt_token_ids=...)``.  It never decodes,
renders, re-tokenizes, or truncates the input.  Output IDs are authoritative:
direct-output roles require exact tokenizer-decode/candidate-text parity, while
GPT-OSS additionally requires strict OpenAI Harmony parsing with exactly one
final channel, no unknown channel, and excluded analysis content.

No function writes an artifact.  The returned preflight/runtime receipts are
in-memory values that an independently frozen executor may persist only after
separate output authorization.
"""

from __future__ import annotations

import base64
import copy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import swe_task_state_v4_epistemic_chain_annotation_runner_v3 as runner_v3  # noqa: E402


CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3.json"
)
SOURCE_PATH = Path(__file__).resolve()
SCHEMA_VERSION = 1
CONFIG_KIND = "swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3"
LAUNCH_KIND = "swe_task_state_v4_epistemic_chain_native_vllm_launch_authorization_v3"
PREFLIGHT_KIND = "swe_task_state_v4_epistemic_chain_native_vllm_preflight_receipt_v3"
RUNTIME_RECEIPT_KIND = (
    "swe_task_state_v4_epistemic_chain_native_vllm_runtime_receipt_v3"
)
NON_GATE_PLAN_KIND = "swe_task_state_v4_epistemic_chain_native_vllm_non_gate_plan_v3"
NON_GATE_TEST_RECORD_KIND = (
    "swe_task_state_v4_epistemic_chain_native_vllm_non_gate_test_record_v3"
)
NATIVE_REQUEST_KIND = "swe_task_state_v4_native_generation_request_v3"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# ``generation.max_num_seqs`` is vLLM's in-flight scheduler concurrency, not a
# limit on the number of prompts accepted by ``LLM.generate``. Keep a distinct
# fail-closed cap for one externally authenticated request batch so a complete
# control round can share one model load while vLLM schedules it in waves.
MAX_AUTHENTICATED_BATCH_REQUESTS = 256

FALSE_NON_GATE_CLAIMS = MappingProxyType(
    {
        "actual_model_execution": False,
        "gate_eligible": False,
        "production_receipt": False,
        "sealed_control_evidence": False,
        "output_token_text_parity_observed_on_real_model": False,
    }
)


def _fresh_false_non_gate_claims() -> dict[str, bool]:
    """Return literal false claims, independent of any exported name rebinding."""

    return {
        "actual_model_execution": False,
        "gate_eligible": False,
        "production_receipt": False,
        "sealed_control_evidence": False,
        "output_token_text_parity_observed_on_real_model": False,
    }


V2_ROLE_FIELDS = {
    "execution_mode",
    "base_model_lineage",
    "repo_id",
    "revision",
    "snapshot_tree_sha256",
    "snapshot_file_count",
    "snapshot_size_bytes",
    "quantization",
    "dtype",
    "seed",
    "chat_template_kwargs",
    "output_extraction",
    "vllm_engine_kwargs",
}

V2_GENERATION_FIELDS = {
    "engine",
    "temperature",
    "top_p",
    "max_output_tokens",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "no_input_truncation",
    "vllm_use_flashinfer_sampler",
    "vllm_enable_v1_multiprocessing",
    "vllm_disabled_kernels",
    "structured_outputs_config",
    "cuda_home_override",
    "prompt_token_accounting",
}

MODEL_IDENTITY_FIELDS = {
    "base_model_lineage",
    "repo_id",
    "revision",
    "snapshot_tree_sha256",
    "quantization",
    "dtype",
}

TOKENIZER_IDENTITY_FIELDS = {
    "repo_id",
    "revision",
    "snapshot_tree_sha256",
    "tokenizer_mode",
    "tokenizer_class",
    "vocab_identity_sha256",
}


class NativeVllmAdapterError(ValueError):
    """Raised when any production or non-gate contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NativeVllmAdapterError(message)


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


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise NativeVllmAdapterError(f"cannot hash file {path}: {error}") from error
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be an array",
    )
    return list(value)


def _token_ids(value: Any, label: str, *, allow_empty: bool = False) -> list[int]:
    items = _sequence(value, label)
    _require(
        (allow_empty or bool(items))
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in items
        ),
        f"{label} are invalid",
    )
    return list(items)


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = value.decode("utf-8")
        parsed = json.loads(decoded, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise NativeVllmAdapterError(f"cannot parse {label}: {error}") from error
    return _mapping(parsed, label)


def _read_regular_file(path: Path, label: str, *, reject_symlink: bool = True) -> bytes:
    _require(
        not reject_symlink or not path.is_symlink(), f"{label} must not be a symlink"
    )
    try:
        resolved = path.resolve(strict=True)
        _require(resolved.is_file(), f"{label} must be a regular file")
        return resolved.read_bytes()
    except OSError as error:
        raise NativeVllmAdapterError(f"cannot read {label}: {error}") from error


def _resolve_bound_root_path(relative_path: str, label: str) -> Path:
    _require(
        isinstance(relative_path, str)
        and bool(relative_path)
        and not Path(relative_path).is_absolute(),
        f"{label} path must be a nonempty repository-relative path",
    )
    candidate = ROOT / relative_path
    try:
        resolved = candidate.resolve(strict=True)
        root = ROOT.resolve(strict=True)
    except OSError as error:
        raise NativeVllmAdapterError(f"cannot resolve {label}: {error}") from error
    _require(
        resolved == root or root in resolved.parents, f"{label} escapes repository root"
    )
    return resolved


@dataclass(frozen=True)
class AuthenticatedAdapterConfig:
    value: Mapping[str, Any]
    path: Path
    config_sha256: str
    source_sha256: str
    runner_sha256: str
    draft_config_sha256: str
    draft_source_sha256: str
    v2_config_sha256: str


def _validate_authorization(value: Any) -> dict[str, Any]:
    authorization = _mapping(value, "authorization")
    expected_fields = {
        "execution_authorized",
        "model_access_authorized",
        "gpu_access_authorized",
        "output_authorized",
        "production_receipt_authorized",
        "gate_eligible_execution_authorized",
        "authorized_launch_binding_path",
        "authorized_launch_binding_sha256",
        "future_enablement_requires_separately_frozen_launch_binding",
    }
    _require(set(authorization) == expected_fields, "authorization fields changed")
    flag_names = {
        "execution_authorized",
        "model_access_authorized",
        "gpu_access_authorized",
        "output_authorized",
        "production_receipt_authorized",
        "gate_eligible_execution_authorized",
    }
    _require(
        all(isinstance(authorization[name], bool) for name in flag_names),
        "authorization flags must be booleans",
    )
    _require(
        authorization["future_enablement_requires_separately_frozen_launch_binding"]
        is True,
        "separate launch binding requirement removed",
    )
    _require(
        all(authorization[name] is False for name in flag_names)
        and authorization["authorized_launch_binding_path"] is None
        and authorization["authorized_launch_binding_sha256"] is None,
        "adapter config must remain false-state; only an external launch may authorize",
    )
    return authorization


def validate_adapter_config(value: Any) -> dict[str, Any]:
    config = _mapping(value, "native vLLM adapter config")
    expected_fields = {
        "schema_version",
        "kind",
        "id",
        "status",
        "scope",
        "authorization",
        "implementation",
        "frozen_inputs",
        "package_contract",
        "environment_contract",
        "gpu_contract",
        "roles",
        "generation",
        "snapshot_contract",
        "tokenizer_contract",
        "launch_binding_contract",
        "receipt_contract",
        "non_gate_test_route",
        "claim_scope",
        "remaining_enablement_blockers",
    }
    _require(set(config) == expected_fields, "adapter config fields changed")
    _require(
        config["schema_version"] == SCHEMA_VERSION
        and config["kind"] == CONFIG_KIND
        and config["status"]
        == "production_source_implemented_execution_unauthorized_no_runtime_evidence",
        "adapter config identity invalid",
    )
    _require(
        _mapping(config["scope"], "scope")
        == {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "model_or_gpu_execution_performed": False,
            "runtime_artifacts_emitted": False,
            "source_implementation_is_not_execution_evidence": True,
        },
        "adapter scope changed",
    )
    _validate_authorization(config["authorization"])

    implementation = _mapping(config["implementation"], "implementation")
    _require(
        set(implementation)
        == {
            "source_path",
            "source_sha256",
            "production_entrypoint",
            "backend_ownership",
            "test_helper_kind",
        }
        and implementation["source_path"]
        == "scripts/swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3.py"
        and (
            implementation["source_sha256"] is None
            or SHA256_RE.fullmatch(str(implementation["source_sha256"])) is not None
        )
        and implementation["production_entrypoint"] == "execute_production_native_batch"
        and implementation["backend_ownership"]
        == "module_owned_imports_tokenizer_engine_and_session_no_injected_backend"
        and implementation["test_helper_kind"] == NON_GATE_TEST_RECORD_KIND,
        "implementation contract invalid",
    )

    frozen = _mapping(config["frozen_inputs"], "frozen inputs")
    _require(
        set(frozen)
        == {
            "source_only_draft_config",
            "source_only_draft_implementation",
            "bounded_v3_runner",
            "historical_v2_runtime_config",
        },
        "frozen input set changed",
    )
    expected_frozen = {
        "source_only_draft_config": (
            "configs/swe_task_state_v4_epistemic_chain_native_adapter_draft_v3.json",
            "156244aa7508306b0791f913ce52461cdde79526e50a03873801a6e38d6d5eeb",
        ),
        "source_only_draft_implementation": (
            "scripts/swe_task_state_v4_epistemic_chain_native_adapter_draft_v3.py",
            "50868a75d5c1dcd1181db4e655f61d7614007a15d60ad6fc7715fb11fbfc8d97",
        ),
        "bounded_v3_runner": (
            "scripts/swe_task_state_v4_epistemic_chain_annotation_runner_v3.py",
            "c35af19c9f3f1e38208ba7f23467c386ebff2e5fb6572582f9a620f51f394aeb",
        ),
        "historical_v2_runtime_config": (
            "configs/swe_task_state_v4_epistemic_chain_annotation_runner_v2.json",
            "a22604d78938917972dc23c517c942fb9022e867e17ce0b5a060189dd0dd1b4d",
        ),
    }
    for name, (path, digest) in expected_frozen.items():
        _require(
            _mapping(frozen[name], name) == {"path": path, "sha256": digest},
            f"frozen input binding changed: {name}",
        )

    package = _mapping(config["package_contract"], "package contract")
    _require(
        _mapping(package.get("interpreter"), "interpreter")
        == {
            "implementation": "cpython",
            "version": "3.12.13",
            "executable": "/home/mark/lumo_J_lens/.venv-vllm/bin/python",
        }
        and package.get("platform") == "Linux-7.0.0-27-generic-x86_64-with-glibc2.43"
        and package.get("distributions")
        == {
            "vllm": "0.23.0",
            "torch": "2.11.0+cu130",
            "transformers": "5.12.1",
            "openai-harmony": "0.0.8",
            "xgrammar": "0.2.2",
        }
        and package.get("distribution_identity_algorithm")
        == "sha256_canonical_json_name_version_metadata_sha256_record_sha256"
        and package.get("launch_binding_must_supply_package_bundle_sha256") is True
        and package.get("runtime_import_verification")
        == {
            "root_specs_verified_before_import": True,
            "security_module_files_verified_after_import": True,
            "module_file_must_be_under_distribution_root": True,
            "record_sha256_and_size_required": True,
        },
        "package contract changed",
    )

    environment = _mapping(config["environment_contract"], "environment contract")
    exact_env = _mapping(environment.get("exact_values"), "exact environment")
    _require(
        exact_env
        == {
            "HF_HOME": "/home/mark/.cache/huggingface",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_DISABLED_KERNELS": "FlashInferFP8ScaledMMLinearKernel,FlashInferCutlassNvFp4LinearKernel,FlashInferTrtllmNvFp4LinearKernel,FlashInferCudnnNvFp4LinearKernel",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONHASHSEED": "0",
        }
        and environment.get("must_be_absent") == ["CUDA_HOME"]
        and environment.get("mutation_before_authentication_forbidden") is True
        and environment.get("launch_binding_must_supply_environment_identity_sha256")
        is True,
        "environment contract changed",
    )

    gpu = _mapping(config["gpu_contract"], "GPU contract")
    _require(
        gpu.get("expected_visible_device_count") == 1
        and gpu.get("expected_device_index") == 0
        and gpu.get("expected_device_name") == "NVIDIA GeForce RTX 5090"
        and gpu.get("identity_fields")
        == [
            "torch_cuda_version",
            "cudnn_version",
            "visible_device_count",
            "device_index",
            "device_name",
            "compute_capability",
            "total_memory_bytes",
        ]
        and gpu.get("launch_binding_must_supply_gpu_identity_sha256") is True
        and gpu.get("gpu_query_forbidden_before_authorization") is True,
        "GPU contract changed",
    )

    roles = _mapping(config["roles"], "roles")
    _require(
        set(roles) == {"independent_a", "independent_b", "adjudicator"},
        "role roster changed",
    )
    expected_tokenizers = {
        "independent_a": (
            "auto",
            "transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer",
            248077,
            "420ab4b96193b4325156f56cd7c3876a8a0d46f515fe4f711a7a6bf5553bf8fa",
        ),
        "independent_b": (
            "auto",
            "transformers.tokenization_utils_tokenizers.TokenizersBackend",
            200019,
            "c473d7dea3722ad7fe8f6abf41274a326ee1ad25d1aba9034a36a9c663145d0e",
        ),
        "adjudicator": (
            "mistral",
            "transformers.tokenization_mistral_common.MistralCommonBackend",
            130044,
            "02f7bdb5b17ed8abcfb48ea188eac98a07d766c9e215c675baf357fbdee44786",
        ),
    }
    common_role_fields = V2_ROLE_FIELDS | {
        "snapshot_relative_to_hf_home",
        "tokenizer_mode",
        "tokenizer_class",
        "vocab_size",
        "vocab_identity_sha256",
    }
    for role_name, value in roles.items():
        role = _mapping(value, f"role {role_name}")
        expected_fields_for_role = set(common_role_fields)
        if role_name == "independent_b":
            expected_fields_for_role.add("harmony")
        _require(
            set(role) == expected_fields_for_role, f"role {role_name} fields changed"
        )
        tokenizer_mode, tokenizer_class, vocab_size, vocab_hash = expected_tokenizers[
            role_name
        ]
        _require(
            role["execution_mode"] == "local_model"
            and SHA256_RE.fullmatch(str(role["snapshot_tree_sha256"])) is not None
            and isinstance(role["snapshot_file_count"], int)
            and role["snapshot_file_count"] > 0
            and isinstance(role["snapshot_size_bytes"], int)
            and role["snapshot_size_bytes"] > 0
            and role["tokenizer_mode"] == tokenizer_mode
            and role["tokenizer_class"] == tokenizer_class
            and role["vocab_size"] == vocab_size
            and role["vocab_identity_sha256"] == vocab_hash,
            f"role {role_name} tokenizer or snapshot identity changed",
        )
    _require(
        roles["independent_a"]["output_extraction"] == "direct_structured_text"
        and roles["independent_b"]["output_extraction"]
        == "openai_harmony_final_channel"
        and roles["adjudicator"]["output_extraction"] == "direct_structured_text"
        and roles["independent_b"]["harmony"]
        == {
            "encoding": "HARMONY_GPT_OSS",
            "strict": True,
            "assistant_role": True,
            "exactly_one_final_channel": True,
            "unknown_channels_forbidden": True,
            "analysis_excluded": True,
            "max_output_tokens": 2048,
        },
        "output extraction contract changed",
    )

    generation = _mapping(config["generation"], "generation")
    _require(
        {key: generation.get(key) for key in V2_GENERATION_FIELDS}
        == {
            "engine": "vllm_offline_structured_outputs",
            "temperature": 0,
            "top_p": 1.0,
            "max_output_tokens": 768,
            "max_model_len": 90112,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.9,
            "no_input_truncation": True,
            "vllm_use_flashinfer_sampler": "0",
            "vllm_enable_v1_multiprocessing": "0",
            "vllm_disabled_kernels": [
                "FlashInferFP8ScaledMMLinearKernel",
                "FlashInferCutlassNvFp4LinearKernel",
                "FlashInferTrtllmNvFp4LinearKernel",
                "FlashInferCudnnNvFp4LinearKernel",
            ],
            "structured_outputs_config": {
                "backend": "xgrammar",
                "disable_any_whitespace": True,
            },
            "cuda_home_override": None,
            "prompt_token_accounting": "vllm_request_output.prompt_token_ids",
        },
        "historical generation settings changed",
    )
    _require(
        generation.get("enable_chunked_prefill") is True
        and generation.get("enable_prefix_caching") is True
        and generation.get("language_model_only") is True
        and generation.get("limit_mm_per_prompt") == {"image": 0, "video": 0}
        and generation.get("async_scheduling") is False
        and generation.get("per_request_structured_outputs")
        == {
            "type": "StructuredOutputsParams",
            "json_from_native_request": True,
            "disable_any_whitespace": True,
        }
        and generation.get("per_request_sampling_seed_source")
        == "native_generation_request.seed_bound_in_launch_request_batch"
        and generation.get("prompt_input")
        == {
            "type": "vllm.inputs.TokensPrompt",
            "only_field": "prompt_token_ids",
            "decode_before_submission": False,
            "render_before_submission": False,
            "retokenize_before_submission": False,
            "truncate_prompt_tokens": None,
        }
        and generation.get("output_decode")
        == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
            "candidate_text_exact_parity_required": True,
        },
        "native exact-token generation settings changed",
    )

    _require(
        _mapping(config["snapshot_contract"], "snapshot contract")
        == {
            "root_source": "environment.HF_HOME",
            "algorithm": "sha256_canonical_json_sorted_relative_path_size_sha256_entries",
            "resolved_regular_files_only": True,
            "tree_hash_file_count_and_total_size_all_required": True,
            "verify_before_and_after_generation": True,
            "network_forbidden": True,
        },
        "snapshot contract changed",
    )
    _require(
        _mapping(config["tokenizer_contract"], "tokenizer contract")
        == {
            "loader": "transformers.AutoTokenizer.from_pretrained",
            "local_files_only": True,
            "trust_remote_code": False,
            "vocab_identity_algorithm": "sha256_canonical_json_get_vocab_mapping",
            "class_is_fully_qualified_type_name": True,
            "vocab_size_and_hash_must_match_role": True,
            "runner_authenticated_context_required": True,
            "native_apply_chat_template_tokenize_true_required": True,
        },
        "tokenizer contract changed",
    )
    launch_contract = _mapping(config["launch_binding_contract"], "launch contract")
    _require(
        launch_contract.get("kind") == LAUNCH_KIND
        and all(
            launch_contract.get(name) is True
            for name in (
                "external_expected_file_sha256_required",
                "launch_binds_adapter_config_sha256_required",
                "execution_authorized_true_required",
                "adapter_config_source_runner_and_draft_hashes_required",
                "role_model_snapshot_tokenizer_package_runtime_environment_gpu_hashes_required",
                "request_batch_hash_required",
                "nonce_sha256_required",
                "self_hash_without_external_expected_hash_is_not_authorization",
            )
        ),
        "launch binding contract changed",
    )
    _require(
        _mapping(config["receipt_contract"], "receipt contract")
        == {
            "preflight_kind": PREFLIGHT_KIND,
            "runtime_kind": RUNTIME_RECEIPT_KIND,
            "all_identity_objects_and_external_hash_bindings_required": True,
            "prompt_ids_output_ids_finish_reason_schema_sampling_and_decode_parity_required": True,
            "in_memory_builder_only_no_file_writer": True,
        },
        "receipt contract changed",
    )
    _require(
        _mapping(config["non_gate_test_route"], "non-gate route")
        == {
            "record_kind": NON_GATE_TEST_RECORD_KIND,
            "actual_model_execution": False,
            "gate_eligible": False,
            "production_receipt": False,
            "may_not_be_promoted_or_rebound": True,
        },
        "non-gate route changed",
    )
    claims = _mapping(config["claim_scope"], "claim scope")
    _require(
        bool(claims) and all(item is False for item in claims.values()),
        "claim scope is not false",
    )
    blockers = _sequence(config["remaining_enablement_blockers"], "blockers")
    _require(
        len(blockers) >= 5
        and all(isinstance(item, str) and bool(item) for item in blockers),
        "enablement blockers missing",
    )
    return copy.deepcopy(config)


def authenticate_adapter_config(
    *, path: Path, expected_config_sha256: str
) -> AuthenticatedAdapterConfig:
    expected = _sha256(expected_config_sha256, "expected adapter config hash")
    config_bytes = _read_regular_file(Path(path), "adapter config")
    observed = sha256_bytes(config_bytes)
    _require(
        observed == expected, "adapter config differs from external authenticated hash"
    )
    config = validate_adapter_config(_strict_json_bytes(config_bytes, "adapter config"))

    frozen_hashes: dict[str, str] = {}
    for name, binding_value in config["frozen_inputs"].items():
        binding = _mapping(binding_value, f"frozen input {name}")
        bound_path = _resolve_bound_root_path(binding["path"], f"frozen input {name}")
        digest = sha256_bytes(_read_regular_file(bound_path, f"frozen input {name}"))
        _require(
            digest == _sha256(binding["sha256"], f"frozen input {name} hash"),
            f"frozen input changed: {name}",
        )
        frozen_hashes[name] = digest

    source_binding = config["implementation"]
    source_path = _resolve_bound_root_path(
        source_binding["source_path"], "adapter source"
    )
    _require(source_path == SOURCE_PATH, "adapter source path changed")
    source_hash = sha256_bytes(_read_regular_file(source_path, "adapter source"))
    configured_source_hash = source_binding["source_sha256"]
    if configured_source_hash is not None:
        _require(
            source_hash == _sha256(configured_source_hash, "adapter source hash"),
            "adapter source differs from configured frozen hash",
        )

    draft_config_path = _resolve_bound_root_path(
        config["frozen_inputs"]["source_only_draft_config"]["path"],
        "draft config",
    )
    draft = _strict_json_bytes(
        _read_regular_file(draft_config_path, "draft config"), "draft config"
    )
    v2_path = _resolve_bound_root_path(
        config["frozen_inputs"]["historical_v2_runtime_config"]["path"],
        "V2 runtime config",
    )
    v2 = _strict_json_bytes(_read_regular_file(v2_path, "V2 config"), "V2 config")
    role_projection = {
        role_name: {key: copy.deepcopy(role[key]) for key in V2_ROLE_FIELDS}
        for role_name, role in config["roles"].items()
    }
    generation_projection = {
        key: copy.deepcopy(config["generation"][key]) for key in V2_GENERATION_FIELDS
    }
    _require(
        role_projection == v2.get("roles") == draft.get("historical_v2_roles"),
        "role roster differs from full frozen V2/draft role objects",
    )
    _require(
        generation_projection
        == v2.get("generation")
        == draft.get("historical_v2_generation"),
        "generation settings differ from full frozen V2/draft generation objects",
    )
    return AuthenticatedAdapterConfig(
        value=copy.deepcopy(config),
        path=Path(path).resolve(strict=True),
        config_sha256=observed,
        source_sha256=source_hash,
        runner_sha256=frozen_hashes["bounded_v3_runner"],
        draft_config_sha256=frozen_hashes["source_only_draft_config"],
        draft_source_sha256=frozen_hashes["source_only_draft_implementation"],
        v2_config_sha256=frozen_hashes["historical_v2_runtime_config"],
    )


def load_adapter_config(
    *, path: Path = CONFIG_PATH, expected_config_sha256: str
) -> dict[str, Any]:
    return copy.deepcopy(
        dict(
            authenticate_adapter_config(
                path=path, expected_config_sha256=expected_config_sha256
            ).value
        )
    )


def snapshot_inventory(snapshot_path: Path) -> dict[str, Any]:
    """Hash each resolved regular file in one exact local HF snapshot."""

    snapshot = Path(snapshot_path)
    _require(snapshot.exists() and snapshot.is_dir(), "model snapshot is unavailable")
    _require(not snapshot.is_symlink(), "model snapshot root must not be a symlink")
    resolved_snapshot = snapshot.resolve(strict=True)
    entries: list[dict[str, Any]] = []
    for path in sorted(
        snapshot.rglob("*"), key=lambda item: item.relative_to(snapshot).as_posix()
    ):
        if path.is_dir():
            _require(not path.is_symlink(), "snapshot contains a symlinked directory")
            continue
        _require(path.is_file(), f"snapshot entry is not a regular file: {path}")
        resolved = path.resolve(strict=True)
        _require(resolved.is_file(), f"snapshot target is not a regular file: {path}")
        entries.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    _require(bool(entries), "model snapshot contains no files")
    body = {
        "snapshot_path": str(resolved_snapshot),
        "files": entries,
        "tree_sha256": sha256_value(entries),
        "file_count": len(entries),
        "size_bytes": sum(item["size_bytes"] for item in entries),
    }
    return {**body, "inventory_sha256": sha256_value(body)}


def resolve_and_verify_role_snapshot(
    *, config: Mapping[str, Any], role: str
) -> tuple[Path, dict[str, Any]]:
    roles = _mapping(config["roles"], "roles")
    _require(role in roles, "role invalid")
    role_spec = _mapping(roles[role], f"role {role}")
    hf_home = Path(config["environment_contract"]["exact_values"]["HF_HOME"])
    expected = hf_home / role_spec["snapshot_relative_to_hf_home"]
    _require(not expected.is_symlink(), "snapshot root must not be a symlink")
    try:
        snapshot = expected.resolve(strict=True)
        hf_root = hf_home.resolve(strict=True)
    except OSError as error:
        raise NativeVllmAdapterError(
            f"cannot resolve local snapshot: {error}"
        ) from error
    _require(hf_root in snapshot.parents, "snapshot escapes authenticated HF_HOME")
    inventory = snapshot_inventory(snapshot)
    _require(
        inventory["tree_sha256"] == role_spec["snapshot_tree_sha256"]
        and inventory["file_count"] == role_spec["snapshot_file_count"]
        and inventory["size_bytes"] == role_spec["snapshot_size_bytes"],
        "model snapshot tree/file-count/size identity changed",
    )
    return snapshot, inventory


def _distribution_file_bytes(distribution: Any, basename: str) -> bytes:
    candidates = [
        item
        for item in (distribution.files or ())
        if str(item).replace("\\", "/").endswith(".dist-info/" + basename)
    ]
    _require(len(candidates) == 1, f"distribution {basename} identity file missing")
    path = Path(distribution.locate_file(candidates[0]))
    return _read_regular_file(path, f"distribution {basename}", reject_symlink=False)


def package_identity_bundle(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read distribution metadata only; this imports no runtime package."""

    expected = _mapping(config["package_contract"]["distributions"], "package versions")
    identities: dict[str, Any] = {}
    for name, expected_version in sorted(expected.items()):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise NativeVllmAdapterError(
                f"required distribution unavailable: {name}"
            ) from error
        observed_name = str(distribution.metadata.get("Name", ""))
        version = str(distribution.version)
        _require(version == expected_version, f"package version changed: {name}")
        metadata_bytes = _distribution_file_bytes(distribution, "METADATA")
        record_bytes = _distribution_file_bytes(distribution, "RECORD")
        identity = {
            "requested_name": name,
            "distribution_name": observed_name,
            "version": version,
            "metadata_sha256": sha256_bytes(metadata_bytes),
            "record_sha256": sha256_bytes(record_bytes),
        }
        identities[name] = {**identity, "identity_sha256": sha256_value(identity)}
    body = {
        "algorithm": config["package_contract"]["distribution_identity_algorithm"],
        "distributions": identities,
    }
    return {**body, "package_bundle_sha256": sha256_value(body)}


def _verify_distribution_record_file(
    *,
    distribution_name: str,
    file_path: Path,
    expected_identity: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Bind one imported file to its authenticated distribution RECORD entry."""

    identity = _mapping(expected_identity, f"{label} distribution identity")
    try:
        distribution = importlib.metadata.distribution(distribution_name)
        distribution_root = Path(distribution.locate_file("")).resolve(strict=True)
        resolved_file = Path(file_path).resolve(strict=True)
    except (importlib.metadata.PackageNotFoundError, OSError) as error:
        raise NativeVllmAdapterError(
            f"cannot resolve {label} distribution file: {error}"
        ) from error
    _require(
        distribution_root in resolved_file.parents and resolved_file.is_file(),
        f"{label} is outside its authenticated distribution root",
    )
    _require(
        str(distribution.version) == identity.get("version")
        and str(distribution.metadata.get("Name", ""))
        == identity.get("distribution_name")
        and sha256_bytes(_distribution_file_bytes(distribution, "METADATA"))
        == identity.get("metadata_sha256")
        and sha256_bytes(_distribution_file_bytes(distribution, "RECORD"))
        == identity.get("record_sha256"),
        f"{label} distribution identity changed before import verification",
    )
    matching_entries = []
    for entry in distribution.files or ():
        try:
            located = Path(distribution.locate_file(entry)).resolve(strict=True)
        except OSError:
            continue
        if located == resolved_file:
            matching_entries.append(entry)
    _require(
        len(matching_entries) == 1,
        f"{label} file is not uniquely represented in distribution RECORD",
    )
    entry = matching_entries[0]
    record_hash = getattr(entry, "hash", None)
    record_size = getattr(entry, "size", None)
    _require(
        record_hash is not None
        and getattr(record_hash, "mode", None) == "sha256"
        and isinstance(getattr(record_hash, "value", None), str)
        and isinstance(record_size, int)
        and not isinstance(record_size, bool),
        f"{label} RECORD entry lacks a strict SHA-256 and size",
    )
    file_bytes = _read_regular_file(resolved_file, label)
    encoded_digest = (
        base64.urlsafe_b64encode(hashlib.sha256(file_bytes).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    _require(
        len(file_bytes) == record_size and encoded_digest == record_hash.value,
        f"{label} bytes differ from authenticated distribution RECORD",
    )
    return {
        "distribution": distribution_name,
        "module_file": str(resolved_file.relative_to(distribution_root)),
        "module_file_sha256": sha256_bytes(file_bytes),
        "record_sha256_urlsafe_base64": encoded_digest,
        "record_size_bytes": record_size,
    }


def _verify_imported_distribution_module(
    *,
    module: Any,
    expected_module_name: str,
    distribution_name: str,
    package_identity: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        getattr(module, "__name__", None) == expected_module_name,
        f"imported module name changed: {expected_module_name}",
    )
    module_file = getattr(module, "__file__", None)
    _require(
        isinstance(module_file, str) and bool(module_file),
        f"imported module file missing: {expected_module_name}",
    )
    distributions = _mapping(
        package_identity.get("distributions"), "authenticated package identities"
    )
    _require(
        distribution_name in distributions,
        f"distribution identity missing: {distribution_name}",
    )
    return _verify_distribution_record_file(
        distribution_name=distribution_name,
        file_path=Path(module_file),
        expected_identity=distributions[distribution_name],
        label=f"imported module {expected_module_name}",
    )


def _validate_runtime_import_specs(package_identity: Mapping[str, Any]) -> None:
    """Reject root-package shadowing before any heavyweight runtime import."""

    distributions = _mapping(
        package_identity.get("distributions"), "authenticated package identities"
    )
    for distribution_name, module_name in (
        ("vllm", "vllm"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("xgrammar", "xgrammar"),
        ("openai-harmony", "openai_harmony"),
    ):
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            module_file = getattr(loaded, "__file__", None)
        else:
            try:
                spec = importlib.util.find_spec(module_name)
            except (ImportError, AttributeError, ValueError) as error:
                raise NativeVllmAdapterError(
                    f"cannot resolve runtime import spec {module_name}: {error}"
                ) from error
            module_file = None if spec is None else spec.origin
        _require(
            isinstance(module_file, str) and bool(module_file),
            f"runtime import spec missing: {module_name}",
        )
        _verify_distribution_record_file(
            distribution_name=distribution_name,
            file_path=Path(module_file),
            expected_identity=distributions[distribution_name],
            label=f"runtime import spec {module_name}",
        )


def runtime_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = config["package_contract"]
    executable = str(Path(sys.executable).resolve(strict=True))
    identity = {
        "implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "executable": executable,
        "platform": platform.platform(),
    }
    _require(
        identity
        == {
            "implementation": expected["interpreter"]["implementation"],
            "python_version": expected["interpreter"]["version"],
            "executable": str(
                Path(expected["interpreter"]["executable"]).resolve(strict=True)
            ),
            "platform": expected["platform"],
        },
        "Python runtime identity changed",
    )
    return {**identity, "runtime_identity_sha256": sha256_value(identity)}


def environment_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = config["environment_contract"]
    expected = _mapping(contract["exact_values"], "exact environment")
    observed = {name: os.environ.get(name) for name in sorted(expected)}
    absent = {name: os.environ.get(name) for name in contract["must_be_absent"]}
    _require(
        observed == {name: expected[name] for name in sorted(expected)},
        "environment values changed",
    )
    _require(
        all(value is None for value in absent.values()),
        "forbidden environment value is present",
    )
    identity = {"exact_values": observed, "absent_values": absent}
    return {**identity, "environment_identity_sha256": sha256_value(identity)}


@dataclass(frozen=True)
class NativeRequestSpec:
    """All exact inputs needed to rebuild one runner-native request."""

    messages: Sequence[Mapping[str, str]]
    schema: Mapping[str, Any]
    seed: int
    stage: str
    annotation_pass: str
    packet_id_sha256: str
    source_id_sha256: str
    lineage_bindings: Mapping[str, Any]
    expected_native_request_sha256: str


def _validate_request_spec(value: Any, *, position: int) -> NativeRequestSpec:
    _require(
        isinstance(value, NativeRequestSpec), f"request spec {position} type invalid"
    )
    messages = _sequence(value.messages, f"request spec {position} messages")
    _require(bool(messages), f"request spec {position} messages empty")
    normalized_messages: list[dict[str, str]] = []
    for message_position, item in enumerate(messages):
        message = _mapping(item, f"request spec {position} message {message_position}")
        _require(
            set(message) == {"role", "content"}
            and isinstance(message["role"], str)
            and bool(message["role"])
            and isinstance(message["content"], str),
            f"request spec {position} message invalid",
        )
        normalized_messages.append(message)
    try:
        schema = runner_v3.validate_executable_response_schema(value.schema)
    except Exception as error:
        raise NativeVllmAdapterError(
            f"request spec {position} schema invalid: {error}"
        ) from error
    _require(
        isinstance(value.seed, int) and not isinstance(value.seed, bool),
        f"request spec {position} seed invalid",
    )
    _require(
        isinstance(value.stage, str)
        and bool(value.stage)
        and value.annotation_pass in runner_v3.PASSES,
        f"request spec {position} stage/pass invalid",
    )
    _sha256(value.packet_id_sha256, f"request spec {position} packet ID")
    _sha256(value.source_id_sha256, f"request spec {position} source ID")
    lineage = _mapping(value.lineage_bindings, f"request spec {position} lineage")
    try:
        runner_v3._validate_hash_lineage(lineage)  # exact frozen runner contract
    except Exception as error:
        raise NativeVllmAdapterError(
            f"request spec {position} lineage invalid: {error}"
        ) from error
    _sha256(
        value.expected_native_request_sha256,
        f"request spec {position} expected native request hash",
    )
    return NativeRequestSpec(
        messages=tuple(copy.deepcopy(normalized_messages)),
        schema=copy.deepcopy(schema),
        seed=value.seed,
        stage=value.stage,
        annotation_pass=value.annotation_pass,
        packet_id_sha256=value.packet_id_sha256,
        source_id_sha256=value.source_id_sha256,
        lineage_bindings=copy.deepcopy(lineage),
        expected_native_request_sha256=value.expected_native_request_sha256,
    )


def request_batch_descriptor(
    *, role: str, request_specs: Sequence[NativeRequestSpec], config: Mapping[str, Any]
) -> dict[str, Any]:
    roles = _mapping(config["roles"], "roles")
    _require(role in roles, "role invalid")
    specs = _sequence(request_specs, "request specs")
    _require(bool(specs), "request batch is empty")
    _require(
        len(specs) <= MAX_AUTHENTICATED_BATCH_REQUESTS,
        "request batch exceeds authenticated batch cap",
    )
    validated = [
        _validate_request_spec(item, position=position)
        for position, item in enumerate(specs)
    ]
    rows: list[dict[str, Any]] = []
    for position, spec in enumerate(validated):
        row = {
            "position": position,
            "expected_native_request_sha256": spec.expected_native_request_sha256,
            "messages_sha256": sha256_value(list(spec.messages)),
            "response_schema_sha256": sha256_value(spec.schema),
            "seed": spec.seed,
            "stage": spec.stage,
            "annotation_pass": spec.annotation_pass,
            "packet_id_sha256": spec.packet_id_sha256,
            "source_id_sha256": spec.source_id_sha256,
            "lineage_bindings_sha256": sha256_value(spec.lineage_bindings),
        }
        rows.append(row)
    body = {"role": role, "request_count": len(rows), "requests": rows}
    return {**body, "request_batch_sha256": sha256_value(body)}


def _model_identity(role_spec: Mapping[str, Any]) -> dict[str, Any]:
    return {field: copy.deepcopy(role_spec[field]) for field in MODEL_IDENTITY_FIELDS}


def _configured_tokenizer_identity(role_spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "repo_id": role_spec["repo_id"],
        "revision": role_spec["revision"],
        "snapshot_tree_sha256": role_spec["snapshot_tree_sha256"],
        "tokenizer_mode": role_spec["tokenizer_mode"],
        "tokenizer_class": role_spec["tokenizer_class"],
        "vocab_identity_sha256": role_spec["vocab_identity_sha256"],
    }


@dataclass(frozen=True)
class AuthenticatedLaunchBinding:
    value: Mapping[str, Any]
    path: Path
    file_sha256: str


LAUNCH_FIELDS = {
    "schema_version",
    "kind",
    "execution_authorized",
    "model_access_authorized",
    "gpu_access_authorized",
    "output_authorized",
    "production_receipt_authorized",
    "gate_eligible_execution_authorized",
    "adapter_config_sha256",
    "adapter_source_sha256",
    "runner_sha256",
    "draft_config_sha256",
    "draft_source_sha256",
    "v2_config_sha256",
    "role",
    "model_identity",
    "model_identity_sha256",
    "snapshot_inventory_sha256",
    "tokenizer_identity",
    "tokenizer_identity_sha256",
    "package_bundle_sha256",
    "runtime_identity_sha256",
    "environment_identity_sha256",
    "gpu_identity_sha256",
    "request_batch_sha256",
    "authorization_nonce_sha256",
}


def authenticate_launch_binding(
    *,
    authenticated_config: AuthenticatedAdapterConfig,
    launch_binding_path: Path,
    expected_launch_binding_sha256: str,
    role: str,
    request_batch_sha256: str,
) -> AuthenticatedLaunchBinding:
    """Authenticate one out-of-band launch file against an external hash."""

    _require(
        isinstance(authenticated_config, AuthenticatedAdapterConfig),
        "adapter config is not authenticated",
    )
    config = dict(authenticated_config.value)
    _validate_authorization(config["authorization"])
    configured_source_hash = config["implementation"]["source_sha256"]
    _require(
        configured_source_hash is not None
        and configured_source_hash == authenticated_config.source_sha256,
        "authorized execution requires a frozen exact adapter source hash",
    )
    expected_launch_hash = _sha256(
        expected_launch_binding_sha256, "external expected launch binding hash"
    )
    launch_path = Path(launch_binding_path)
    _require(not launch_path.is_symlink(), "launch binding must not be a symlink")
    try:
        launch_path = launch_path.resolve(strict=True)
    except OSError as error:
        raise NativeVllmAdapterError(
            f"cannot resolve launch binding: {error}"
        ) from error
    launch_bytes = _read_regular_file(launch_path, "authorized launch binding")
    observed = sha256_bytes(launch_bytes)
    _require(
        observed == expected_launch_hash,
        "launch binding differs from external authenticated hash",
    )
    launch = _strict_json_bytes(launch_bytes, "launch binding")
    _require(set(launch) == LAUNCH_FIELDS, "launch binding fields invalid")
    _require(
        launch["schema_version"] == SCHEMA_VERSION and launch["kind"] == LAUNCH_KIND,
        "launch binding identity invalid",
    )
    for flag in (
        "execution_authorized",
        "model_access_authorized",
        "gpu_access_authorized",
        "output_authorized",
        "production_receipt_authorized",
        "gate_eligible_execution_authorized",
    ):
        _require(launch[flag] is True, f"launch binding does not authorize {flag}")
    expected_hashes = {
        "adapter_config_sha256": authenticated_config.config_sha256,
        "adapter_source_sha256": authenticated_config.source_sha256,
        "runner_sha256": authenticated_config.runner_sha256,
        "draft_config_sha256": authenticated_config.draft_config_sha256,
        "draft_source_sha256": authenticated_config.draft_source_sha256,
        "v2_config_sha256": authenticated_config.v2_config_sha256,
        "request_batch_sha256": _sha256(request_batch_sha256, "request batch hash"),
    }
    _require(
        all(launch.get(name) == digest for name, digest in expected_hashes.items()),
        "launch binding source/config/runner/draft/V2/request hashes changed",
    )
    for name in (
        "model_identity_sha256",
        "snapshot_inventory_sha256",
        "tokenizer_identity_sha256",
        "package_bundle_sha256",
        "runtime_identity_sha256",
        "environment_identity_sha256",
        "gpu_identity_sha256",
        "authorization_nonce_sha256",
    ):
        _sha256(launch[name], f"launch {name}")
    roles = config["roles"]
    _require(role in roles and launch["role"] == role, "launch role changed")
    model = _model_identity(roles[role])
    tokenizer_identity = _configured_tokenizer_identity(roles[role])
    _require(
        launch["model_identity"] == model
        and launch["model_identity_sha256"] == sha256_value(model),
        "launch model identity changed",
    )
    _require(
        launch["tokenizer_identity"] == tokenizer_identity
        and launch["tokenizer_identity_sha256"] == sha256_value(tokenizer_identity),
        "launch tokenizer identity changed",
    )
    return AuthenticatedLaunchBinding(
        value=copy.deepcopy(launch), path=launch_path, file_sha256=observed
    )


@dataclass(frozen=True)
class PreflightReceipt:
    body: Mapping[str, Any]
    receipt_sha256: str


@dataclass(frozen=True)
class RuntimeReceipt:
    body: Mapping[str, Any]
    receipt_sha256: str


@dataclass(frozen=True)
class ProductionBatch:
    requests: Sequence[runner_v3.NativeGenerationRequest]
    results: Sequence[runner_v3.NativeGenerationResult]
    preflight_receipt: PreflightReceipt
    runtime_receipt: RuntimeReceipt


_PRODUCTION_AUTHORITY_TOKEN = object()


@dataclass(frozen=True)
class _ProductionAuthority:
    config: AuthenticatedAdapterConfig
    launch: AuthenticatedLaunchBinding
    request_batch: Mapping[str, Any]
    token: Any


def _production_authority(
    *,
    authenticated_config: AuthenticatedAdapterConfig,
    launch: AuthenticatedLaunchBinding,
    request_batch: Mapping[str, Any],
) -> _ProductionAuthority:
    _require(
        isinstance(launch, AuthenticatedLaunchBinding)
        and launch.value["execution_authorized"] is True,
        "production launch is not authenticated",
    )
    return _ProductionAuthority(
        config=authenticated_config,
        launch=launch,
        request_batch=copy.deepcopy(dict(request_batch)),
        token=_PRODUCTION_AUTHORITY_TOKEN,
    )


def _require_production_authority(value: Any) -> _ProductionAuthority:
    _require(
        isinstance(value, _ProductionAuthority)
        and value.token is _PRODUCTION_AUTHORITY_TOKEN,
        "production authority absent",
    )
    return value


def _build_preflight_receipt(
    *,
    authority: _ProductionAuthority,
    role: str,
    runtime: Mapping[str, Any],
    environment: Mapping[str, Any],
    packages: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    gpu_identity: Mapping[str, Any],
) -> PreflightReceipt:
    authority = _require_production_authority(authority)
    launch = authority.launch.value
    model_identity = _model_identity(authority.config.value["roles"][role])
    comparisons = {
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "environment_identity_sha256": environment["environment_identity_sha256"],
        "package_bundle_sha256": packages["package_bundle_sha256"],
        "snapshot_inventory_sha256": snapshot["inventory_sha256"],
        "tokenizer_identity_sha256": sha256_value(tokenizer_identity),
        "gpu_identity_sha256": gpu_identity["gpu_identity_sha256"],
        "model_identity_sha256": sha256_value(model_identity),
    }
    _require(
        all(launch[name] == value for name, value in comparisons.items()),
        "launch binding differs from observed preflight identity",
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": PREFLIGHT_KIND,
        "adapter_config_sha256": authority.config.config_sha256,
        "adapter_source_sha256": authority.config.source_sha256,
        "runner_sha256": authority.config.runner_sha256,
        "launch_binding_sha256": authority.launch.file_sha256,
        "role": role,
        "request_batch_sha256": authority.request_batch["request_batch_sha256"],
        "runtime_identity": copy.deepcopy(dict(runtime)),
        "environment_identity": copy.deepcopy(dict(environment)),
        "package_identity": copy.deepcopy(dict(packages)),
        "snapshot_inventory": copy.deepcopy(dict(snapshot)),
        "model_identity": model_identity,
        "model_identity_sha256": sha256_value(model_identity),
        "tokenizer_identity": copy.deepcopy(dict(tokenizer_identity)),
        "tokenizer_identity_sha256": sha256_value(tokenizer_identity),
        "gpu_identity": copy.deepcopy(dict(gpu_identity)),
        "authorization_nonce_sha256": launch["authorization_nonce_sha256"],
        "model_loaded": False,
        "generation_performed": False,
        "production_execution_authorized": True,
        "gate_evidence_established": False,
    }
    return PreflightReceipt(body=body, receipt_sha256=sha256_value(body))


def _validate_preflight_identity_objects(
    *, body: Mapping[str, Any], authority: _ProductionAuthority
) -> None:
    """Recompute every nested preflight identity, not just its named hash."""

    config = authority.config.value
    launch = authority.launch.value
    runtime = _mapping(body["runtime_identity"], "preflight runtime identity")
    _require(
        set(runtime)
        == {
            "implementation",
            "python_version",
            "executable",
            "platform",
            "runtime_identity_sha256",
        }
        and runtime["runtime_identity_sha256"]
        == sha256_value(
            {
                key: value
                for key, value in runtime.items()
                if key != "runtime_identity_sha256"
            }
        )
        == launch["runtime_identity_sha256"],
        "preflight runtime identity self-hash invalid",
    )
    expected_interpreter = config["package_contract"]["interpreter"]
    _require(
        runtime["implementation"] == expected_interpreter["implementation"]
        and runtime["python_version"] == expected_interpreter["version"]
        and runtime["executable"]
        == str(Path(expected_interpreter["executable"]).resolve(strict=True))
        and runtime["platform"] == config["package_contract"]["platform"],
        "preflight runtime identity differs from config",
    )

    environment = _mapping(
        body["environment_identity"], "preflight environment identity"
    )
    _require(
        set(environment)
        == {"exact_values", "absent_values", "environment_identity_sha256"}
        and environment["environment_identity_sha256"]
        == sha256_value(
            {
                key: value
                for key, value in environment.items()
                if key != "environment_identity_sha256"
            }
        )
        == launch["environment_identity_sha256"]
        and environment["exact_values"]
        == config["environment_contract"]["exact_values"]
        and environment["absent_values"]
        == {name: None for name in config["environment_contract"]["must_be_absent"]},
        "preflight environment identity invalid",
    )

    packages = _mapping(body["package_identity"], "preflight package identity")
    distributions = _mapping(
        packages.get("distributions"), "preflight distribution identities"
    )
    expected_versions = config["package_contract"]["distributions"]
    valid_distributions = set(distributions) == set(expected_versions)
    for name, raw_identity in distributions.items():
        identity = _mapping(raw_identity, f"preflight distribution {name}")
        valid_distributions = valid_distributions and (
            set(identity)
            == {
                "requested_name",
                "distribution_name",
                "version",
                "metadata_sha256",
                "record_sha256",
                "identity_sha256",
            }
            and identity["requested_name"] == name
            and isinstance(identity["distribution_name"], str)
            and bool(identity["distribution_name"])
            and identity["version"] == expected_versions.get(name)
            and isinstance(identity["metadata_sha256"], str)
            and SHA256_RE.fullmatch(identity["metadata_sha256"]) is not None
            and isinstance(identity["record_sha256"], str)
            and SHA256_RE.fullmatch(identity["record_sha256"]) is not None
            and identity["identity_sha256"]
            == sha256_value(
                {
                    key: value
                    for key, value in identity.items()
                    if key != "identity_sha256"
                }
            )
        )
    _require(
        set(packages) == {"algorithm", "distributions", "package_bundle_sha256"}
        and packages["algorithm"]
        == config["package_contract"]["distribution_identity_algorithm"]
        and valid_distributions
        and packages["package_bundle_sha256"]
        == sha256_value(
            {
                key: value
                for key, value in packages.items()
                if key != "package_bundle_sha256"
            }
        )
        == launch["package_bundle_sha256"],
        "preflight package identity invalid",
    )

    snapshot = _mapping(body["snapshot_inventory"], "preflight snapshot inventory")
    files = _sequence(snapshot.get("files"), "preflight snapshot files")
    valid_files = bool(files)
    file_paths: list[str] = []
    for item in files:
        entry = _mapping(item, "preflight snapshot file")
        path_value = entry.get("path")
        canonical_path = (
            isinstance(path_value, str)
            and bool(path_value)
            and str(PurePosixPath(path_value)) == path_value
            and not PurePosixPath(path_value).is_absolute()
            and ".." not in PurePosixPath(path_value).parts
            and "\\" not in path_value
        )
        valid_files = valid_files and (
            set(entry) == {"path", "size_bytes", "sha256"}
            and canonical_path
            and isinstance(entry["size_bytes"], int)
            and not isinstance(entry["size_bytes"], bool)
            and entry["size_bytes"] >= 0
            and isinstance(entry["sha256"], str)
            and SHA256_RE.fullmatch(entry["sha256"]) is not None
        )
        if isinstance(path_value, str):
            file_paths.append(path_value)
    snapshot_without_hash = {
        key: value for key, value in snapshot.items() if key != "inventory_sha256"
    }
    _require(
        set(snapshot)
        == {
            "snapshot_path",
            "files",
            "tree_sha256",
            "file_count",
            "size_bytes",
            "inventory_sha256",
        }
        and isinstance(snapshot["snapshot_path"], str)
        and bool(snapshot["snapshot_path"])
        and Path(snapshot["snapshot_path"]).is_absolute()
        and valid_files
        and file_paths == sorted(file_paths)
        and len(file_paths) == len(set(file_paths))
        and snapshot["tree_sha256"] == sha256_value(files)
        and snapshot["file_count"] == len(files)
        and snapshot["size_bytes"]
        == sum(_mapping(item, "snapshot file")["size_bytes"] for item in files)
        and snapshot["inventory_sha256"]
        == sha256_value(snapshot_without_hash)
        == launch["snapshot_inventory_sha256"],
        "preflight snapshot identity invalid",
    )

    role = launch["role"]
    model_identity = _mapping(body["model_identity"], "preflight model identity")
    expected_model = _model_identity(config["roles"][role])
    tokenizer_identity = _mapping(
        body["tokenizer_identity"], "preflight tokenizer identity"
    )
    expected_tokenizer = _configured_tokenizer_identity(config["roles"][role])
    _require(
        model_identity == expected_model == launch["model_identity"]
        and body["model_identity_sha256"]
        == sha256_value(model_identity)
        == launch["model_identity_sha256"]
        and tokenizer_identity == expected_tokenizer == launch["tokenizer_identity"]
        and body["tokenizer_identity_sha256"]
        == sha256_value(tokenizer_identity)
        == launch["tokenizer_identity_sha256"],
        "preflight model/tokenizer identity invalid",
    )

    gpu = _mapping(body["gpu_identity"], "preflight GPU identity")
    gpu_hash_body = {
        key: value for key, value in gpu.items() if key != "gpu_identity_sha256"
    }
    _require(
        set(gpu)
        == set(config["gpu_contract"]["identity_fields"]) | {"gpu_identity_sha256"}
        and isinstance(gpu["torch_cuda_version"], str)
        and bool(gpu["torch_cuda_version"])
        and isinstance(gpu["cudnn_version"], int)
        and not isinstance(gpu["cudnn_version"], bool)
        and gpu["cudnn_version"] > 0
        and isinstance(gpu["visible_device_count"], int)
        and not isinstance(gpu["visible_device_count"], bool)
        and isinstance(gpu["device_index"], int)
        and not isinstance(gpu["device_index"], bool)
        and gpu["visible_device_count"]
        == config["gpu_contract"]["expected_visible_device_count"]
        and gpu["device_index"] == config["gpu_contract"]["expected_device_index"]
        and gpu["device_name"] == config["gpu_contract"]["expected_device_name"]
        and isinstance(gpu["compute_capability"], list)
        and len(gpu["compute_capability"]) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in gpu["compute_capability"]
        )
        and isinstance(gpu["total_memory_bytes"], int)
        and not isinstance(gpu["total_memory_bytes"], bool)
        and gpu["total_memory_bytes"] > 0
        and gpu["gpu_identity_sha256"]
        == sha256_value(gpu_hash_body)
        == launch["gpu_identity_sha256"],
        "preflight GPU identity invalid",
    )


def validate_preflight_receipt(
    receipt: Any, *, authority: _ProductionAuthority, expected_receipt_sha256: str
) -> dict[str, Any]:
    authority = _require_production_authority(authority)
    _require(isinstance(receipt, PreflightReceipt), "preflight receipt type invalid")
    body = _mapping(receipt.body, "preflight receipt body")
    expected_fields = {
        "schema_version",
        "kind",
        "adapter_config_sha256",
        "adapter_source_sha256",
        "runner_sha256",
        "launch_binding_sha256",
        "role",
        "request_batch_sha256",
        "runtime_identity",
        "environment_identity",
        "package_identity",
        "snapshot_inventory",
        "model_identity",
        "model_identity_sha256",
        "tokenizer_identity",
        "tokenizer_identity_sha256",
        "gpu_identity",
        "authorization_nonce_sha256",
        "model_loaded",
        "generation_performed",
        "production_execution_authorized",
        "gate_evidence_established",
    }
    launch = authority.launch.value
    runtime = _mapping(body.get("runtime_identity"), "preflight runtime identity")
    environment = _mapping(
        body.get("environment_identity"), "preflight environment identity"
    )
    packages = _mapping(body.get("package_identity"), "preflight package identity")
    snapshot = _mapping(body.get("snapshot_inventory"), "preflight snapshot inventory")
    model_identity = _mapping(body.get("model_identity"), "preflight model identity")
    tokenizer_identity = _mapping(
        body.get("tokenizer_identity"), "preflight tokenizer identity"
    )
    gpu = _mapping(body.get("gpu_identity"), "preflight GPU identity")
    _require(
        set(body) == expected_fields
        and body.get("schema_version") == SCHEMA_VERSION
        and body.get("kind") == PREFLIGHT_KIND
        and body.get("adapter_config_sha256") == authority.config.config_sha256
        and body.get("adapter_source_sha256") == authority.config.source_sha256
        and body.get("runner_sha256") == authority.config.runner_sha256
        and body.get("launch_binding_sha256") == authority.launch.file_sha256
        and body.get("request_batch_sha256")
        == authority.request_batch["request_batch_sha256"]
        and body.get("role") == launch["role"]
        and body.get("authorization_nonce_sha256")
        == launch["authorization_nonce_sha256"]
        and runtime.get("runtime_identity_sha256") == launch["runtime_identity_sha256"]
        and environment.get("environment_identity_sha256")
        == launch["environment_identity_sha256"]
        and packages.get("package_bundle_sha256") == launch["package_bundle_sha256"]
        and snapshot.get("inventory_sha256") == launch["snapshot_inventory_sha256"]
        and model_identity == launch["model_identity"]
        and body.get("model_identity_sha256")
        == sha256_value(model_identity)
        == launch["model_identity_sha256"]
        and tokenizer_identity == launch["tokenizer_identity"]
        and body.get("tokenizer_identity_sha256")
        == sha256_value(tokenizer_identity)
        == launch["tokenizer_identity_sha256"]
        and gpu.get("gpu_identity_sha256") == launch["gpu_identity_sha256"]
        and body.get("model_loaded") is False
        and body.get("generation_performed") is False
        and body.get("production_execution_authorized") is True
        and body.get("gate_evidence_established") is False,
        "preflight receipt identity or claim invalid",
    )
    _validate_preflight_identity_objects(body=body, authority=authority)
    observed = sha256_value(body)
    _require(
        receipt.receipt_sha256 == observed
        and observed
        == _sha256(expected_receipt_sha256, "expected preflight receipt hash"),
        "preflight receipt differs from external authenticated hash",
    )
    return copy.deepcopy(body)


@dataclass(frozen=True)
class NonGateExactTokenPlan:
    body: Mapping[str, Any]
    plan_sha256: str


@dataclass(frozen=True)
class NonGateTestRecord:
    body: Mapping[str, Any]
    record_sha256: str


def build_non_gate_exact_token_plan(
    *, request: runner_v3.NativeGenerationRequest, config: Mapping[str, Any]
) -> NonGateExactTokenPlan:
    """Build a test-only description; it cannot initialize or call a backend."""

    try:
        request_body = runner_v3.validate_native_generation_request(request)
    except Exception as error:
        raise NativeVllmAdapterError(f"native request invalid: {error}") from error
    generation = config["generation"]
    _require(
        isinstance(request_body["seed"], int)
        and not isinstance(request_body["seed"], bool),
        "native request seed invalid",
    )
    sampling = {
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "seed": request_body["seed"],
        "max_tokens": generation["max_output_tokens"],
        "structured_outputs": {
            "json": copy.deepcopy(request_body["response_schema"]),
            "disable_any_whitespace": True,
        },
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": NON_GATE_PLAN_KIND,
        "native_request_sha256": request.request_sha256,
        "tokens_prompt": {
            "type": "vllm.inputs.TokensPrompt",
            "prompt_token_ids": list(request_body["submitted_prompt_token_ids"]),
        },
        "sampling_params": sampling,
        "input_transformations": {
            "decode": False,
            "render": False,
            "retokenize": False,
            "truncate": False,
        },
        "claims": {
            "actual_model_execution": False,
            "gate_eligible": False,
            "production_receipt": False,
            "sealed_control_evidence": False,
            "output_token_text_parity_observed_on_real_model": False,
        },
    }
    return NonGateExactTokenPlan(body=body, plan_sha256=sha256_value(body))


def build_non_gate_test_record(
    *,
    plan: NonGateExactTokenPlan,
    engine_prompt_token_ids: Sequence[int],
    output_token_ids: Sequence[int],
    finish_reason: str,
) -> NonGateTestRecord:
    _require(isinstance(plan, NonGateExactTokenPlan), "non-gate plan type invalid")
    _require(plan.plan_sha256 == sha256_value(plan.body), "non-gate plan hash invalid")
    plan_body = _mapping(plan.body, "non-gate plan")
    submitted = _token_ids(
        plan_body["tokens_prompt"]["prompt_token_ids"], "submitted prompt IDs"
    )
    engine = _token_ids(engine_prompt_token_ids, "engine prompt IDs")
    output = _token_ids(output_token_ids, "output token IDs")
    _require(
        engine == submitted, "non-gate engine prompt IDs differ from submitted IDs"
    )
    _require(finish_reason == "stop", "non-gate finish reason is not stop")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": NON_GATE_TEST_RECORD_KIND,
        "plan_sha256": plan.plan_sha256,
        "engine_prompt_token_ids": engine,
        "engine_prompt_token_ids_sha256": sha256_value(engine),
        "output_token_ids": output,
        "output_token_ids_sha256": sha256_value(output),
        "finish_reason": finish_reason,
        "claims": {
            "actual_model_execution": False,
            "gate_eligible": False,
            "production_receipt": False,
            "sealed_control_evidence": False,
            "output_token_text_parity_observed_on_real_model": False,
        },
    }
    return NonGateTestRecord(body=body, record_sha256=sha256_value(body))


def validate_non_gate_test_record(
    record: Any,
    *,
    plan: NonGateExactTokenPlan,
    expected_record_sha256: str,
) -> dict[str, Any]:
    _require(isinstance(record, NonGateTestRecord), "non-gate record type invalid")
    _require(isinstance(plan, NonGateExactTokenPlan), "non-gate plan type invalid")
    body = _mapping(record.body, "non-gate record")
    _require(
        set(body)
        == {
            "schema_version",
            "kind",
            "plan_sha256",
            "engine_prompt_token_ids",
            "engine_prompt_token_ids_sha256",
            "output_token_ids",
            "output_token_ids_sha256",
            "finish_reason",
            "claims",
        }
        and body["schema_version"] == SCHEMA_VERSION
        and body["kind"] == NON_GATE_TEST_RECORD_KIND
        and body["plan_sha256"] == plan.plan_sha256
        and set(_mapping(body["claims"], "non-gate claims"))
        == {
            "actual_model_execution",
            "gate_eligible",
            "production_receipt",
            "sealed_control_evidence",
            "output_token_text_parity_observed_on_real_model",
        }
        and all(
            item is False
            for item in _mapping(body["claims"], "non-gate claims").values()
        ),
        "non-gate record identity or claims invalid",
    )
    engine = _token_ids(body["engine_prompt_token_ids"], "engine prompt IDs")
    output = _token_ids(body["output_token_ids"], "output token IDs")
    _require(
        body["engine_prompt_token_ids_sha256"] == sha256_value(engine)
        and body["output_token_ids_sha256"] == sha256_value(output)
        and body["finish_reason"] == "stop",
        "non-gate record token hashes or finish reason invalid",
    )
    observed = sha256_value(body)
    _require(
        record.record_sha256 == observed
        and observed
        == _sha256(expected_record_sha256, "expected non-gate record hash"),
        "non-gate record differs from external authenticated hash",
    )
    return copy.deepcopy(body)


def _qualified_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _load_authenticated_tokenizer_and_context(
    *,
    authority: _ProductionAuthority,
    role: str,
    snapshot_path: Path,
) -> tuple[Any, runner_v3.AuthenticatedNativeGenerationContext, dict[str, Any]]:
    """Owned local-only tokenizer construction after production authorization."""

    authority = _require_production_authority(authority)
    config = authority.config.value
    role_spec = config["roles"][role]

    # This import is intentionally owned by the production module and occurs
    # only after config and launch authorization.  No loader is injectable.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer_class = _qualified_type_name(tokenizer)
    _require(
        tokenizer_class == role_spec["tokenizer_class"],
        "loaded tokenizer class differs from frozen role identity",
    )
    try:
        raw_vocab = tokenizer.get_vocab()
    except Exception as error:
        raise NativeVllmAdapterError(
            f"cannot obtain tokenizer vocabulary: {error}"
        ) from error
    vocab = _mapping(raw_vocab, "tokenizer vocabulary")
    _require(
        len(vocab) == role_spec["vocab_size"]
        and all(
            isinstance(token, str)
            and isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and token_id >= 0
            for token, token_id in vocab.items()
        ),
        "tokenizer vocabulary shape changed",
    )
    vocab_hash = sha256_value(vocab)
    _require(
        vocab_hash == role_spec["vocab_identity_sha256"],
        "tokenizer vocabulary identity changed",
    )
    tokenizer_identity = {
        "repo_id": role_spec["repo_id"],
        "revision": role_spec["revision"],
        "snapshot_tree_sha256": role_spec["snapshot_tree_sha256"],
        "tokenizer_mode": role_spec["tokenizer_mode"],
        "tokenizer_class": tokenizer_class,
        "vocab_identity_sha256": vocab_hash,
    }
    _require(
        tokenizer_identity == authority.launch.value["tokenizer_identity"]
        and sha256_value(tokenizer_identity)
        == authority.launch.value["tokenizer_identity_sha256"],
        "loaded tokenizer differs from authenticated launch identity",
    )
    model_identity = _model_identity(role_spec)
    try:
        context = runner_v3.authenticate_native_generation_context(
            tokenizer=tokenizer,
            model_identity=model_identity,
            expected_model_identity_sha256=sha256_value(model_identity),
            tokenizer_identity=tokenizer_identity,
            expected_tokenizer_identity_sha256=sha256_value(tokenizer_identity),
            chat_template_kwargs=copy.deepcopy(role_spec["chat_template_kwargs"]),
        )
    except Exception as error:
        raise NativeVllmAdapterError(
            f"cannot construct authenticated V3 generation context: {error}"
        ) from error
    return tokenizer, context, tokenizer_identity


def _build_native_requests(
    *,
    context: runner_v3.AuthenticatedNativeGenerationContext,
    specs: Sequence[NativeRequestSpec],
) -> list[runner_v3.NativeGenerationRequest]:
    results: list[runner_v3.NativeGenerationRequest] = []
    for position, raw_spec in enumerate(specs):
        spec = _validate_request_spec(raw_spec, position=position)
        try:
            request = runner_v3.build_native_generation_request(
                context=context,
                messages=copy.deepcopy(list(spec.messages)),
                schema=copy.deepcopy(dict(spec.schema)),
                seed=spec.seed,
                stage=spec.stage,
                annotation_pass=spec.annotation_pass,
                packet_id_sha256=spec.packet_id_sha256,
                source_id_sha256=spec.source_id_sha256,
                lineage_bindings=copy.deepcopy(dict(spec.lineage_bindings)),
            )
            request_body = runner_v3.validate_native_generation_request(
                request, context=context
            )
        except Exception as error:
            raise NativeVllmAdapterError(
                f"native request {position} construction failed: {error}"
            ) from error
        _require(
            request.request_sha256 == spec.expected_native_request_sha256,
            f"native request {position} differs from externally expected hash",
        )
        _require(
            request_body["seed"] == spec.seed
            and request_body["string_round_trip_used"] is False,
            f"native request {position} seed or rendering provenance changed",
        )
        results.append(request)
    return results


def build_vllm_engine_kwargs(
    *, config: Mapping[str, Any], role: str, snapshot_path: Path
) -> dict[str, Any]:
    roles = _mapping(config["roles"], "roles")
    _require(role in roles, "role invalid")
    role_spec = roles[role]
    generation = config["generation"]
    supplied = _mapping(role_spec["vllm_engine_kwargs"], "role vLLM kwargs")
    runner_owned = {
        "model",
        "tokenizer",
        "dtype",
        "quantization",
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "language_model_only",
        "limit_mm_per_prompt",
        "async_scheduling",
        "seed",
        "structured_outputs_config",
    }
    _require(
        not (set(supplied) & runner_owned),
        "role kwargs override adapter-owned engine fields",
    )
    result: dict[str, Any] = {
        "model": str(Path(snapshot_path).resolve(strict=True)),
        "tokenizer": str(Path(snapshot_path).resolve(strict=True)),
        "dtype": role_spec["dtype"],
        "quantization": role_spec["quantization"],
        "gpu_memory_utilization": generation["gpu_memory_utilization"],
        "max_model_len": generation["max_model_len"],
        "max_num_batched_tokens": generation["max_num_batched_tokens"],
        "max_num_seqs": generation["max_num_seqs"],
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "language_model_only": True,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "async_scheduling": False,
        "seed": role_spec["seed"],
        "structured_outputs_config": {
            "backend": "xgrammar",
            "disable_any_whitespace": True,
        },
    }
    result.update(copy.deepcopy(supplied))
    return result


def _gpu_identity(torch_module: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Query the sole visible GPU only after authenticated authorization."""

    gpu = config["gpu_contract"]
    _require(torch_module.cuda.is_available(), "CUDA is unavailable")
    count = int(torch_module.cuda.device_count())
    index = int(gpu["expected_device_index"])
    _require(count == gpu["expected_visible_device_count"], "visible GPU count changed")
    properties = torch_module.cuda.get_device_properties(index)
    capability = tuple(
        int(item) for item in torch_module.cuda.get_device_capability(index)
    )
    identity = {
        "torch_cuda_version": str(torch_module.version.cuda),
        "cudnn_version": int(torch_module.backends.cudnn.version()),
        "visible_device_count": count,
        "device_index": index,
        "device_name": str(torch_module.cuda.get_device_name(index)),
        "compute_capability": list(capability),
        "total_memory_bytes": int(properties.total_memory),
    }
    _require(
        identity["device_name"] == gpu["expected_device_name"], "GPU model changed"
    )
    return {**identity, "gpu_identity_sha256": sha256_value(identity)}


def decode_output_ids_with_exact_parity(
    *, tokenizer: Any, output_token_ids: Sequence[int], candidate_text: str
) -> tuple[str, dict[str, Any]]:
    """Reconstruct engine text from output IDs and require exact equality."""

    token_ids = _token_ids(output_token_ids, "output token IDs")
    _require(isinstance(candidate_text, str), "engine candidate text invalid")
    try:
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except Exception as error:
        raise NativeVllmAdapterError(
            f"cannot decode output token IDs: {error}"
        ) from error
    _require(
        isinstance(decoded, str) and bool(decoded),
        "authenticated output decode is empty",
    )
    _require(
        decoded == candidate_text,
        "engine candidate text differs from authenticated token decode",
    )
    provenance = {
        "decode_source": "authenticated_tokenizer.decode(output_token_ids)",
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
        "output_token_ids_sha256": sha256_value(token_ids),
        "decoded_text_sha256": sha256_bytes(decoded.encode("utf-8")),
        "candidate_text_sha256": sha256_bytes(candidate_text.encode("utf-8")),
        "candidate_text_exact_parity": True,
    }
    return decoded, provenance


def _valid_harmony_final_recipient(value: Any) -> bool:
    return value is None or (
        isinstance(value, str)
        and re.fullmatch(r"<\|constrain\|>[A-Za-z0-9_.:-]{1,128}", value)
        is not None
    )


def extract_openai_harmony_final_channel(
    token_ids: Sequence[int],
) -> tuple[str, dict[str, Any]]:
    """Strictly parse one GPT-OSS final channel and discard analysis content."""

    completion_ids = _token_ids(token_ids, "Harmony completion token IDs")
    try:
        from openai_harmony import (
            HarmonyEncodingName,
            Role,
            TextContent,
            load_harmony_encoding,
        )

        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        messages = encoding.parse_messages_from_completion_tokens(
            completion_ids, role=Role.ASSISTANT, strict=True
        )
    except Exception as error:
        raise NativeVllmAdapterError(
            f"strict OpenAI Harmony completion parsing failed: {error}"
        ) from error
    final_messages = [message for message in messages if message.channel == "final"]
    analysis_messages = [
        message for message in messages if message.channel == "analysis"
    ]
    unknown_channels = [
        str(message.channel)
        for message in messages
        if message.channel not in {"analysis", "final"}
    ]
    _require(
        len(final_messages) == 1 and not unknown_channels,
        "Harmony output must contain exactly one final and no unknown channel",
    )
    final_message = final_messages[0]
    final_recipient = getattr(final_message, "recipient", None)
    _require(
        _valid_harmony_final_recipient(final_recipient),
        "Harmony final channel recipient is not a valid constraint sentinel; "
        f"observed={final_recipient!r}",
    )
    _require(
        len(final_message.content) == 1
        and isinstance(final_message.content[0], TextContent),
        "Harmony final channel must contain exactly one text item",
    )
    final_text = str(final_message.content[0].text)
    _require(bool(final_text), "Harmony final channel is empty")
    provenance = {
        "mode": "openai_harmony_final_channel",
        "parser_package": "openai-harmony",
        "parser_version": importlib.metadata.version("openai-harmony"),
        "parser_strict": True,
        "assistant_message_count": len(messages),
        "analysis_message_count": len(analysis_messages),
        "analysis_content_excluded": True,
        "unknown_channel_count": 0,
        "final_message_count": 1,
        "final_recipient": final_recipient,
        "final_text_sha256": sha256_bytes(final_text.encode("utf-8")),
        "completion_token_ids_sha256": sha256_value(completion_ids),
    }
    return final_text, provenance


def _extract_schema_text(
    *,
    role_spec: Mapping[str, Any],
    tokenizer: Any,
    output_token_ids: Sequence[int],
    candidate_text: str,
) -> tuple[str, dict[str, Any]]:
    decoded, decode_provenance = decode_output_ids_with_exact_parity(
        tokenizer=tokenizer,
        output_token_ids=output_token_ids,
        candidate_text=candidate_text,
    )
    mode = role_spec["output_extraction"]
    if mode == "direct_structured_text":
        return decoded, {
            "mode": "direct_structured_text",
            "schema_text_source": "authenticated_output_token_decode",
            "output_token_decode_parity": copy.deepcopy(decode_provenance),
            "analysis_content_excluded": True,
        }
    _require(mode == "openai_harmony_final_channel", "output extraction mode invalid")
    final_text, harmony = extract_openai_harmony_final_channel(output_token_ids)
    return final_text, {
        "mode": "openai_harmony_final_channel",
        "schema_text_source": "strict_harmony_final_channel_from_output_token_ids",
        "output_token_decode_parity": copy.deepcopy(decode_provenance),
        "harmony": harmony,
        "analysis_content_excluded": True,
    }


def _validate_imported_runtime_versions(
    *,
    config: Mapping[str, Any],
    package_identity: Mapping[str, Any],
    vllm_module: Any,
    torch_module: Any,
    transformers_module: Any,
    xgrammar_module: Any,
    openai_harmony_module: Any,
    additional_distribution_modules: Sequence[tuple[str, str, Any]] = (),
) -> None:
    expected = config["package_contract"]["distributions"]
    observed = {
        "vllm": str(vllm_module.__version__),
        "torch": str(torch_module.__version__),
        "transformers": str(transformers_module.__version__),
        "xgrammar": importlib.metadata.version("xgrammar"),
        "openai-harmony": importlib.metadata.version("openai-harmony"),
    }
    _require(observed == expected, "imported runtime package versions changed")
    root_modules = (
        ("vllm", "vllm", vllm_module),
        ("torch", "torch", torch_module),
        ("transformers", "transformers", transformers_module),
        ("xgrammar", "xgrammar", xgrammar_module),
        ("openai-harmony", "openai_harmony", openai_harmony_module),
    )
    for distribution_name, module_name, module in (
        *root_modules,
        *tuple(additional_distribution_modules),
    ):
        _verify_imported_distribution_module(
            module=module,
            expected_module_name=module_name,
            distribution_name=distribution_name,
            package_identity=package_identity,
        )


def _role_max_output_tokens(
    *, role_spec: Mapping[str, Any], generation: Mapping[str, Any]
) -> int:
    value = generation["max_output_tokens"]
    if role_spec["output_extraction"] == "openai_harmony_final_channel":
        value = _mapping(role_spec["harmony"], "Harmony role contract")[
            "max_output_tokens"
        ]
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        "role output-token cap invalid",
    )
    return value


def _sampling_descriptor(
    *,
    request_body: Mapping[str, Any],
    generation: Mapping[str, Any],
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    if max_output_tokens is None:
        max_output_tokens = int(generation["max_output_tokens"])
    return {
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "seed": request_body["seed"],
        "max_tokens": max_output_tokens,
        "structured_outputs": {
            "json": copy.deepcopy(request_body["response_schema"]),
            "disable_any_whitespace": True,
        },
        "truncate_prompt_tokens": None,
    }


def validate_sampling_params_against_native_request(
    *,
    params: Any,
    request_body: Mapping[str, Any],
    generation: Mapping[str, Any],
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Bind the actual vLLM sampling object before the engine is called."""

    expected = _sampling_descriptor(
        request_body=request_body,
        generation=generation,
        max_output_tokens=max_output_tokens,
    )
    structured = getattr(params, "structured_outputs", None)
    _require(structured is not None, "actual sampling params omit structured outputs")
    observed_schema = copy.deepcopy(getattr(structured, "json", None))
    observed_disable_whitespace = getattr(structured, "disable_any_whitespace", None)
    observed_temperature = getattr(params, "temperature", None)
    observed_top_p = getattr(params, "top_p", None)
    observed_seed = getattr(params, "seed", None)
    observed_max_tokens = getattr(params, "max_tokens", None)
    _require(
        isinstance(observed_temperature, (int, float))
        and not isinstance(observed_temperature, bool)
        and observed_temperature == expected["temperature"]
        and isinstance(observed_top_p, (int, float))
        and not isinstance(observed_top_p, bool)
        and observed_top_p == expected["top_p"]
        and isinstance(observed_seed, int)
        and not isinstance(observed_seed, bool)
        and observed_seed == expected["seed"]
        and isinstance(observed_max_tokens, int)
        and not isinstance(observed_max_tokens, bool)
        and observed_max_tokens == expected["max_tokens"]
        and getattr(params, "truncate_prompt_tokens", None) is None
        and observed_schema == expected["structured_outputs"]["json"]
        and observed_disable_whitespace is True,
        "actual sampling params differ from exact seed/schema/xgrammar/no-truncation contract",
    )
    return copy.deepcopy(expected)


def _build_runtime_receipt(
    *,
    authority: _ProductionAuthority,
    role: str,
    preflight: PreflightReceipt,
    engine_kwargs: Mapping[str, Any],
    request_records: Sequence[Mapping[str, Any]],
    snapshot_before: Mapping[str, Any],
    snapshot_after: Mapping[str, Any],
) -> RuntimeReceipt:
    authority = _require_production_authority(authority)
    _require(
        snapshot_after == snapshot_before,
        "model snapshot changed during generation",
    )
    records = [copy.deepcopy(dict(item)) for item in request_records]
    _require(bool(records), "runtime receipt has no request records")
    _require(
        all(
            item.get("engine_prompt_matches_submitted") is True
            and item.get("candidate_text_token_decode_parity") is True
            and item.get("finish_reason") == "stop"
            for item in records
        ),
        "runtime request evidence incomplete",
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUNTIME_RECEIPT_KIND,
        "adapter_config_sha256": authority.config.config_sha256,
        "adapter_source_sha256": authority.config.source_sha256,
        "runner_sha256": authority.config.runner_sha256,
        "launch_binding_sha256": authority.launch.file_sha256,
        "preflight_receipt_sha256": preflight.receipt_sha256,
        "role": role,
        "request_batch_sha256": authority.request_batch["request_batch_sha256"],
        "engine_kwargs_sha256": sha256_value(engine_kwargs),
        "request_records": records,
        "request_records_sha256": sha256_value(records),
        "snapshot_inventory_before_sha256": snapshot_before["inventory_sha256"],
        "snapshot_inventory_after_sha256": snapshot_after["inventory_sha256"],
        "snapshot_unchanged": True,
        "input_string_round_trip_used": False,
        "input_truncation_used": False,
        "actual_model_execution": True,
        "production_receipt": True,
        "gate_eligible": bool(
            authority.launch.value["gate_eligible_execution_authorized"]
        ),
        "reserved_validation_accessed": False,
    }
    return RuntimeReceipt(body=body, receipt_sha256=sha256_value(body))


def _validate_output_extraction_receipt(
    *,
    value: Any,
    record: Mapping[str, Any],
    role_spec: Mapping[str, Any],
    package_versions: Mapping[str, Any],
) -> None:
    """Validate token-decode and role-specific extraction provenance in full."""

    extraction = _mapping(value, "runtime output extraction")
    parity = _mapping(
        extraction.get("output_token_decode_parity"),
        "runtime output token decode parity",
    )
    parity_fields = {
        "decode_source",
        "skip_special_tokens",
        "clean_up_tokenization_spaces",
        "output_token_ids_sha256",
        "decoded_text_sha256",
        "candidate_text_sha256",
        "candidate_text_exact_parity",
    }
    _require(
        set(parity) == parity_fields
        and parity.get("decode_source")
        == "authenticated_tokenizer.decode(output_token_ids)"
        and parity.get("skip_special_tokens") is True
        and parity.get("clean_up_tokenization_spaces") is False
        and parity.get("output_token_ids_sha256") == record["output_token_ids_sha256"]
        and SHA256_RE.fullmatch(str(parity.get("decoded_text_sha256"))) is not None
        and parity.get("decoded_text_sha256") == parity.get("candidate_text_sha256")
        and parity.get("candidate_text_exact_parity") is True,
        "runtime output token decode provenance invalid",
    )

    mode = role_spec["output_extraction"]
    _require(
        extraction.get("mode") == mode
        and extraction.get("analysis_content_excluded") is True,
        "runtime output extraction role or analysis claim invalid",
    )
    if mode == "direct_structured_text":
        _require(
            set(extraction)
            == {
                "mode",
                "schema_text_source",
                "output_token_decode_parity",
                "analysis_content_excluded",
            }
            and extraction.get("schema_text_source")
            == "authenticated_output_token_decode"
            and record["output_text_sha256"] == parity["decoded_text_sha256"],
            "direct structured-output extraction provenance invalid",
        )
        return

    _require(
        mode == "openai_harmony_final_channel"
        and set(extraction)
        == {
            "mode",
            "schema_text_source",
            "output_token_decode_parity",
            "harmony",
            "analysis_content_excluded",
        }
        and extraction.get("schema_text_source")
        == "strict_harmony_final_channel_from_output_token_ids",
        "Harmony extraction provenance invalid",
    )
    harmony = _mapping(extraction.get("harmony"), "runtime Harmony provenance")
    harmony_fields = {
        "mode",
        "parser_package",
        "parser_version",
        "parser_strict",
        "assistant_message_count",
        "analysis_message_count",
        "analysis_content_excluded",
        "unknown_channel_count",
        "final_message_count",
        "final_recipient",
        "final_text_sha256",
        "completion_token_ids_sha256",
    }
    assistant_count = harmony.get("assistant_message_count")
    analysis_count = harmony.get("analysis_message_count")
    _require(
        set(harmony) == harmony_fields
        and harmony.get("mode") == "openai_harmony_final_channel"
        and harmony.get("parser_package") == "openai-harmony"
        and harmony.get("parser_version") == package_versions["openai-harmony"]
        and harmony.get("parser_strict") is True
        and isinstance(assistant_count, int)
        and not isinstance(assistant_count, bool)
        and isinstance(analysis_count, int)
        and not isinstance(analysis_count, bool)
        and analysis_count >= 0
        and assistant_count == analysis_count + 1
        and harmony.get("analysis_content_excluded") is True
        and harmony.get("unknown_channel_count") == 0
        and harmony.get("final_message_count") == 1
        and _valid_harmony_final_recipient(harmony.get("final_recipient"))
        and harmony.get("final_text_sha256") == record["output_text_sha256"]
        and harmony.get("completion_token_ids_sha256")
        == record["output_token_ids_sha256"],
        "Harmony channel provenance invalid",
    )


def validate_runtime_receipt(
    receipt: Any,
    *,
    authority: _ProductionAuthority,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    authority = _require_production_authority(authority)
    _require(isinstance(receipt, RuntimeReceipt), "runtime receipt type invalid")
    body = _mapping(receipt.body, "runtime receipt body")
    expected_fields = {
        "schema_version",
        "kind",
        "adapter_config_sha256",
        "adapter_source_sha256",
        "runner_sha256",
        "launch_binding_sha256",
        "preflight_receipt_sha256",
        "role",
        "request_batch_sha256",
        "engine_kwargs_sha256",
        "request_records",
        "request_records_sha256",
        "snapshot_inventory_before_sha256",
        "snapshot_inventory_after_sha256",
        "snapshot_unchanged",
        "input_string_round_trip_used",
        "input_truncation_used",
        "actual_model_execution",
        "production_receipt",
        "gate_eligible",
        "reserved_validation_accessed",
    }
    launch = authority.launch.value
    _require(
        set(body) == expected_fields
        and body.get("schema_version") == SCHEMA_VERSION
        and body.get("kind") == RUNTIME_RECEIPT_KIND
        and body.get("adapter_config_sha256") == authority.config.config_sha256
        and body.get("adapter_source_sha256") == authority.config.source_sha256
        and body.get("runner_sha256") == authority.config.runner_sha256
        and body.get("launch_binding_sha256") == authority.launch.file_sha256
        and body.get("request_batch_sha256")
        == authority.request_batch["request_batch_sha256"]
        and body.get("role") == launch["role"]
        and body.get("snapshot_inventory_before_sha256")
        == body.get("snapshot_inventory_after_sha256")
        == launch["snapshot_inventory_sha256"]
        and isinstance(body.get("preflight_receipt_sha256"), str)
        and SHA256_RE.fullmatch(body["preflight_receipt_sha256"]) is not None
        and isinstance(body.get("engine_kwargs_sha256"), str)
        and SHA256_RE.fullmatch(body["engine_kwargs_sha256"]) is not None
        and body.get("snapshot_unchanged") is True
        and body.get("input_string_round_trip_used") is False
        and body.get("input_truncation_used") is False
        and body.get("actual_model_execution") is True
        and body.get("production_receipt") is True
        and body.get("gate_eligible") is True
        and body.get("gate_eligible") is launch["gate_eligible_execution_authorized"]
        and body.get("reserved_validation_accessed") is False,
        "runtime receipt identity or claim invalid",
    )
    records = _sequence(body.get("request_records"), "runtime request records")
    expected_max_output_tokens = _role_max_output_tokens(
        role_spec=authority.config.value["roles"][launch["role"]],
        generation=authority.config.value["generation"],
    )
    record_fields = {
        "position",
        "native_request_sha256",
        "native_result_sha256",
        "response_schema_sha256",
        "sampling_params",
        "sampling_params_sha256",
        "submitted_prompt_token_ids_sha256",
        "engine_prompt_token_ids_sha256",
        "engine_prompt_matches_submitted",
        "output_token_ids_sha256",
        "output_text_sha256",
        "candidate_text_token_decode_parity",
        "output_extraction",
        "finish_reason",
    }
    _require(
        len(records) == authority.request_batch["request_count"],
        "runtime request count differs from launch-bound batch",
    )
    validated_record_shape = True
    for position, item in enumerate(records):
        record = _mapping(item, "runtime request record")
        bound_request = authority.request_batch["requests"][position]
        sampling = _mapping(record.get("sampling_params"), "runtime sampling params")
        structured = _mapping(
            sampling.get("structured_outputs"), "runtime structured outputs"
        )
        validated_record_shape = validated_record_shape and (
            set(record) == record_fields
            and record.get("position") == position
            and record.get("native_request_sha256")
            == bound_request["expected_native_request_sha256"]
            and record.get("response_schema_sha256")
            == bound_request["response_schema_sha256"]
            and all(
                isinstance(record.get(name), str)
                and SHA256_RE.fullmatch(record[name]) is not None
                for name in (
                    "native_request_sha256",
                    "native_result_sha256",
                    "response_schema_sha256",
                    "sampling_params_sha256",
                    "submitted_prompt_token_ids_sha256",
                    "engine_prompt_token_ids_sha256",
                    "output_token_ids_sha256",
                    "output_text_sha256",
                )
            )
            and set(sampling)
            == {
                "temperature",
                "top_p",
                "seed",
                "max_tokens",
                "structured_outputs",
                "truncate_prompt_tokens",
            }
            and sampling["temperature"]
            == authority.config.value["generation"]["temperature"]
            and isinstance(sampling["temperature"], (int, float))
            and not isinstance(sampling["temperature"], bool)
            and sampling["top_p"] == authority.config.value["generation"]["top_p"]
            and isinstance(sampling["top_p"], (int, float))
            and not isinstance(sampling["top_p"], bool)
            and sampling["seed"] == bound_request["seed"]
            and isinstance(sampling["seed"], int)
            and not isinstance(sampling["seed"], bool)
            and sampling["max_tokens"] == expected_max_output_tokens
            and isinstance(sampling["max_tokens"], int)
            and not isinstance(sampling["max_tokens"], bool)
            and sampling["truncate_prompt_tokens"] is None
            and set(structured) == {"json", "disable_any_whitespace"}
            and structured["disable_any_whitespace"] is True
            and sha256_value(structured["json"]) == record["response_schema_sha256"]
            and record["sampling_params_sha256"] == sha256_value(sampling)
            and record["engine_prompt_matches_submitted"] is True
            and record["engine_prompt_token_ids_sha256"]
            == record["submitted_prompt_token_ids_sha256"]
            and record["candidate_text_token_decode_parity"] is True
            and record["finish_reason"] == "stop"
        )
        if validated_record_shape:
            _validate_output_extraction_receipt(
                value=record["output_extraction"],
                record=record,
                role_spec=authority.config.value["roles"][launch["role"]],
                package_versions=authority.config.value["package_contract"][
                    "distributions"
                ],
            )
    _require(
        bool(records)
        and validated_record_shape
        and body.get("request_records_sha256") == sha256_value(records),
        "runtime request records invalid",
    )
    observed = sha256_value(body)
    _require(
        receipt.receipt_sha256 == observed
        and observed
        == _sha256(expected_receipt_sha256, "expected runtime receipt hash"),
        "runtime receipt differs from external authenticated hash",
    )
    return copy.deepcopy(body)


def _execute_authorized_production(
    *,
    authority: _ProductionAuthority,
    role: str,
    request_specs: Sequence[NativeRequestSpec],
) -> ProductionBatch:
    """Owned real runtime.  No caller-supplied backend object crosses here."""

    authority = _require_production_authority(authority)
    config = authority.config.value
    role_spec = config["roles"][role]
    generation = config["generation"]
    max_output_tokens = _role_max_output_tokens(
        role_spec=role_spec, generation=generation
    )

    runtime = runtime_identity(config)
    environment = environment_identity(config)
    packages = package_identity_bundle(config)
    _validate_runtime_import_specs(packages)
    snapshot_path, snapshot_before = resolve_and_verify_role_snapshot(
        config=config, role=role
    )
    _require(
        snapshot_before["inventory_sha256"]
        == authority.launch.value["snapshot_inventory_sha256"],
        "snapshot inventory differs from launch binding",
    )
    tokenizer, context, tokenizer_identity = _load_authenticated_tokenizer_and_context(
        authority=authority, role=role, snapshot_path=snapshot_path
    )
    requests = _build_native_requests(context=context, specs=request_specs)
    _require(
        [request.request_sha256 for request in requests]
        == [
            row["expected_native_request_sha256"]
            for row in authority.request_batch["requests"]
        ],
        "built request order or hashes differ from launch-bound batch",
    )
    request_bodies = [
        runner_v3.validate_native_generation_request(request, context=context)
        for request in requests
    ]
    _require(
        all(
            len(body["submitted_prompt_token_ids"])
            + max_output_tokens
            <= int(generation["max_model_len"])
            for body in request_bodies
        ),
        "native request would exceed context without truncation",
    )

    # All heavyweight imports, GPU queries, and the engine session are owned by
    # this authorized path.  There is no callback or injectable backend.
    import torch
    import transformers
    import vllm
    import xgrammar
    import openai_harmony
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import StructuredOutputsParams

    additional_modules: dict[tuple[str, str], Any] = {}
    for distribution_name, symbol, label in (
        ("transformers", type(tokenizer), "loaded tokenizer type"),
        ("vllm", LLM, "vLLM LLM"),
        ("vllm", SamplingParams, "vLLM SamplingParams"),
        ("vllm", TokensPrompt, "vLLM TokensPrompt"),
        (
            "vllm",
            StructuredOutputsParams,
            "vLLM StructuredOutputsParams",
        ),
    ):
        module_name = getattr(symbol, "__module__", None)
        _require(
            isinstance(module_name, str) and bool(module_name),
            f"{label} defining module missing",
        )
        module = sys.modules.get(module_name)
        _require(module is not None, f"{label} defining module is not imported")
        additional_modules[(distribution_name, module_name)] = module
    for module_name, module in (
        ("torch.cuda", torch.cuda),
        ("torch.backends.cudnn", torch.backends.cudnn),
    ):
        additional_modules[("torch", module_name)] = module
    _validate_imported_runtime_versions(
        config=config,
        package_identity=packages,
        vllm_module=vllm,
        torch_module=torch,
        transformers_module=transformers,
        xgrammar_module=xgrammar,
        openai_harmony_module=openai_harmony,
        additional_distribution_modules=tuple(
            (distribution_name, module_name, module)
            for (distribution_name, module_name), module in sorted(
                additional_modules.items()
            )
        ),
    )
    gpu = _gpu_identity(torch, config)
    preflight = _build_preflight_receipt(
        authority=authority,
        role=role,
        runtime=runtime,
        environment=environment,
        packages=packages,
        snapshot=snapshot_before,
        tokenizer_identity=tokenizer_identity,
        gpu_identity=gpu,
    )
    validate_preflight_receipt(
        preflight,
        authority=authority,
        expected_receipt_sha256=preflight.receipt_sha256,
    )

    engine_kwargs = build_vllm_engine_kwargs(
        config=config, role=role, snapshot_path=snapshot_path
    )
    llm = LLM(**engine_kwargs)

    # Exact runner-native prompt IDs are copied directly into TokensPrompt.
    # No string field is provided and no decode/render/tokenize call intervenes.
    token_prompts = [
        TokensPrompt(prompt_token_ids=list(body["submitted_prompt_token_ids"]))
        for body in request_bodies
    ]
    sampling_params = [
        SamplingParams(
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            seed=int(body["seed"]),
            max_tokens=max_output_tokens,
            structured_outputs=StructuredOutputsParams(
                json=copy.deepcopy(dict(body["response_schema"])),
                disable_any_whitespace=True,
            ),
        )
        for body in request_bodies
    ]
    authenticated_sampling = [
        validate_sampling_params_against_native_request(
            params=params,
            request_body=body,
            generation=generation,
            max_output_tokens=max_output_tokens,
        )
        for params, body in zip(sampling_params, request_bodies, strict=True)
    ]
    outputs = llm.generate(
        token_prompts,
        sampling_params,
        use_tqdm=False,
    )
    _require(
        len(outputs) == len(requests), "vLLM output count differs from request count"
    )

    native_results: list[runner_v3.NativeGenerationResult] = []
    runtime_records: list[dict[str, Any]] = []
    for position, (request, request_body, params, sampling, output) in enumerate(
        zip(
            requests,
            request_bodies,
            sampling_params,
            authenticated_sampling,
            outputs,
            strict=True,
        )
    ):
        engine_prompt_ids = _token_ids(
            output.prompt_token_ids,
            f"engine prompt token IDs {position}",
        )
        submitted_ids = list(request_body["submitted_prompt_token_ids"])
        _require(
            engine_prompt_ids == submitted_ids,
            f"engine prompt IDs {position} differ from exact submitted IDs",
        )
        candidates = _sequence(output.outputs, f"engine candidates {position}")
        _require(
            len(candidates) == 1, f"engine returned multiple candidates at {position}"
        )
        candidate = candidates[0]
        output_ids = _token_ids(candidate.token_ids, f"output token IDs {position}")
        finish_reason = candidate.finish_reason
        _require(
            isinstance(finish_reason, str) and finish_reason == "stop",
            f"generation {position} did not finish with stop; "
            f"observed={finish_reason!r}",
        )
        text, extraction = _extract_schema_text(
            role_spec=role_spec,
            tokenizer=tokenizer,
            output_token_ids=output_ids,
            candidate_text=candidate.text,
        )
        try:
            result = runner_v3.build_native_generation_result(
                request=request,
                text=text,
                submitted_prompt_token_ids=submitted_ids,
                engine_prompt_token_ids=engine_prompt_ids,
                output_token_ids=output_ids,
                finish_reason=finish_reason,
            )
            result_body = runner_v3.validate_native_generation_result(
                request=request, result=result
            )
        except Exception as error:
            raise NativeVllmAdapterError(
                f"native result {position} construction failed: {error}"
            ) from error
        native_results.append(result)
        _require(
            validate_sampling_params_against_native_request(
                params=params,
                request_body=request_body,
                generation=generation,
            )
            == sampling,
            f"sampling parameters {position} changed during engine execution",
        )
        runtime_records.append(
            {
                "position": position,
                "native_request_sha256": request.request_sha256,
                "native_result_sha256": result.result_sha256,
                "response_schema_sha256": request_body["response_schema_sha256"],
                "sampling_params": sampling,
                "sampling_params_sha256": sha256_value(sampling),
                "submitted_prompt_token_ids_sha256": request_body[
                    "submitted_prompt_token_ids_sha256"
                ],
                "engine_prompt_token_ids_sha256": result_body[
                    "engine_prompt_token_ids_sha256"
                ],
                "engine_prompt_matches_submitted": True,
                "output_token_ids_sha256": result_body["output_token_ids_sha256"],
                "output_text_sha256": result_body["text_sha256"],
                "candidate_text_token_decode_parity": True,
                "output_extraction": extraction,
                "finish_reason": finish_reason,
            }
        )

    snapshot_after = snapshot_inventory(snapshot_path)
    _require(
        snapshot_after["tree_sha256"] == role_spec["snapshot_tree_sha256"]
        and snapshot_after["file_count"] == role_spec["snapshot_file_count"]
        and snapshot_after["size_bytes"] == role_spec["snapshot_size_bytes"],
        "model snapshot changed after generation",
    )
    runtime_receipt = _build_runtime_receipt(
        authority=authority,
        role=role,
        preflight=preflight,
        engine_kwargs=engine_kwargs,
        request_records=runtime_records,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )
    validate_runtime_receipt(
        runtime_receipt,
        authority=authority,
        expected_receipt_sha256=runtime_receipt.receipt_sha256,
    )
    return ProductionBatch(
        requests=tuple(requests),
        results=tuple(native_results),
        preflight_receipt=preflight,
        runtime_receipt=runtime_receipt,
    )


def execute_production_native_batch(
    *,
    expected_config_sha256: str,
    launch_binding_path: Path,
    expected_launch_binding_sha256: str,
    role: str,
    request_specs: Sequence[NativeRequestSpec],
) -> ProductionBatch:
    """Execute one authorized role batch using only module-owned runtime state.

    The checked-in config remains false-state.  A separately audited launch
    file, independently supplied expected file hash, exact request-batch hash,
    and all runtime identities are required before model/tokenizer/GPU/output
    access.  No execution callback or backend object is accepted.
    """

    authenticated_config = authenticate_adapter_config(
        path=CONFIG_PATH, expected_config_sha256=expected_config_sha256
    )
    _validate_authorization(authenticated_config.value["authorization"])

    # Request specs are used only to derive the exact launch-bound batch hash;
    # no model, tokenizer, runtime package, GPU, or output is touched here.
    request_batch = request_batch_descriptor(
        role=role,
        request_specs=request_specs,
        config=authenticated_config.value,
    )
    launch = authenticate_launch_binding(
        authenticated_config=authenticated_config,
        launch_binding_path=launch_binding_path,
        expected_launch_binding_sha256=expected_launch_binding_sha256,
        role=role,
        request_batch_sha256=request_batch["request_batch_sha256"],
    )
    authority = _production_authority(
        authenticated_config=authenticated_config,
        launch=launch,
        request_batch=request_batch,
    )
    return _execute_authorized_production(
        authority=authority,
        role=role,
        request_specs=request_specs,
    )


__all__ = [
    "AuthenticatedAdapterConfig",
    "AuthenticatedLaunchBinding",
    "CONFIG_KIND",
    "CONFIG_PATH",
    "FALSE_NON_GATE_CLAIMS",
    "LAUNCH_KIND",
    "NON_GATE_PLAN_KIND",
    "NON_GATE_TEST_RECORD_KIND",
    "NativeRequestSpec",
    "NativeVllmAdapterError",
    "NonGateExactTokenPlan",
    "NonGateTestRecord",
    "PREFLIGHT_KIND",
    "PreflightReceipt",
    "ProductionBatch",
    "RUNTIME_RECEIPT_KIND",
    "RuntimeReceipt",
    "authenticate_adapter_config",
    "build_non_gate_exact_token_plan",
    "build_non_gate_test_record",
    "build_vllm_engine_kwargs",
    "canonical_json_bytes",
    "decode_output_ids_with_exact_parity",
    "execute_production_native_batch",
    "extract_openai_harmony_final_channel",
    "load_adapter_config",
    "package_identity_bundle",
    "request_batch_descriptor",
    "resolve_and_verify_role_snapshot",
    "runtime_identity",
    "sha256_bytes",
    "sha256_file",
    "sha256_value",
    "snapshot_inventory",
    "validate_adapter_config",
    "validate_non_gate_test_record",
    "validate_preflight_receipt",
    "validate_runtime_receipt",
    "validate_sampling_params_against_native_request",
]
