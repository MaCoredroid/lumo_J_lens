#!/usr/bin/env python3
"""Source-only V3 native exact-token adapter contract.

This additive draft has no vLLM, tokenizer, model, GPU, output, or receipt
implementation.  Its production and gate entry points unconditionally reject;
a future reviewed source freeze must replace them.  The executable CPU surface
only derives and validates an exact-token invocation plan and can exercise a
strictly separate scripted, irrevocably non-gate record route.

The contract consumes the bounded V3 runner's ``NativeGenerationRequest``
token IDs directly.  A future adapter must put those IDs in a vLLM
``TokensPrompt`` without decoding, re-rendering, re-tokenizing, or truncating
them, and must authenticate the engine-observed prompt IDs and output IDs.
This draft does not prove that reported output text is the authenticated decode
of those IDs; a future real adapter must reconstruct that parity with its bound
tokenizer before any real or gate-eligible claim.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

# This local runner import is source-only and has no model/runtime imports.
import swe_task_state_v4_epistemic_chain_annotation_runner_v3 as runner_v3  # noqa: E402


CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_native_adapter_draft_v3.json"
)
SCHEMA_VERSION = 1
CONFIG_KIND = "swe_task_state_v4_epistemic_chain_native_adapter_draft_v3"
DRAFT_PLAN_KIND = "swe_task_state_v4_epistemic_chain_native_invocation_plan_draft_v3"
DRAFT_OBSERVATION_KIND = (
    "swe_task_state_v4_epistemic_chain_engine_observation_draft_v3"
)
SCRIPTED_NON_GATE_RECORD_KIND = (
    "swe_task_state_v4_epistemic_chain_scripted_cpu_non_gate_record_v3"
)
NATIVE_REQUEST_KIND = "swe_task_state_v4_native_generation_request_v3"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# No config mutation can authorize this source.  A future adapter requires a
# new versioned source file and review, not a runtime flag flip.
PRODUCTION_CODE_FREEZE_COMPLETE = False

# Display-only immutable view.  Record construction and validation deliberately
# use their own literal mappings below, so rebinding or attempting to mutate
# this exported convenience value can never become an execution trust source.
FALSE_EXECUTION_CLAIMS = MappingProxyType({
    "actual_model_execution": False,
    "gate_eligible": False,
    "production_receipt": False,
    "sealed_control_evidence": False,
    "annotation_interface_readiness_established": False,
    "output_text_token_decode_parity_established": False,
})

RUNTIME_IDENTITY_FIELDS = {
    "engine",
    "package",
    "package_version",
    "engine_build_sha256",
    "python_version",
    "cuda_runtime_version",
    "platform",
}
ADAPTER_IDENTITY_FIELDS = {
    "kind",
    "source_sha256",
    "entrypoint",
    "tokens_prompt_api",
    "runner_sha256",
}
MODEL_IDENTITY_FIELDS = {
    "base_model_lineage",
    "repo_id",
    "revision",
    "snapshot_tree_sha256",
    "quantization",
    "dtype",
}


class NativeAdapterDraftError(ValueError):
    """Raised when the prospective source-only contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NativeAdapterDraftError(message)


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


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _token_ids(value: Any, label: str, *, allow_empty: bool = False) -> list[int]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be an array",
    )
    result = list(value)
    _require(
        (allow_empty or bool(result))
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in result
        ),
        f"{label} are invalid",
    )
    return result


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be SHA-256",
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
        raise NativeAdapterDraftError(f"cannot parse {label}: {error}") from error
    return _mapping(parsed, label)


def _read_file_bytes(path: Path, label: str) -> bytes:
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        _require(resolved.is_file(), f"{label} is not a regular file")
        return resolved.read_bytes()
    except OSError as error:
        raise NativeAdapterDraftError(f"cannot read {label}: {error}") from error


@dataclass(frozen=True)
class AuthenticatedDraftConfig:
    """Draft bytes plus exact bound V2/V3 source authentication."""

    value: Mapping[str, Any]
    path: Path
    file_sha256: str
    v2_config_sha256: str
    v3_runner_sha256: str


def authenticate_draft_config(
    *, path: Path, expected_draft_config_sha256: str
) -> AuthenticatedDraftConfig:
    """Authenticate draft bytes before reading any request or caller identity."""

    expected = _sha256(expected_draft_config_sha256, "expected draft config hash")
    draft_bytes = _read_file_bytes(Path(path), "draft config")
    observed = sha256_bytes(draft_bytes)
    _require(observed == expected, "draft config differs from external authenticated hash")
    config = validate_draft_config(_strict_json_bytes(draft_bytes, "draft config"))

    historical = config["frozen_historical_inputs"]
    v2_binding = historical["v2_runtime_config"]
    v2_path = ROOT / v2_binding["path"]
    _require(
        v2_path.resolve(strict=False)
        == (
            ROOT
            / "configs"
            / "swe_task_state_v4_epistemic_chain_annotation_runner_v2.json"
        ).resolve(strict=False),
        "bound V2 config path changed",
    )
    v2_bytes = _read_file_bytes(v2_path, "bound V2 config")
    v2_hash = sha256_bytes(v2_bytes)
    _require(
        v2_hash == _sha256(v2_binding["sha256"], "bound V2 config hash"),
        "bound V2 config bytes changed",
    )
    v2 = _strict_json_bytes(v2_bytes, "bound V2 config")
    _require(
        config["historical_v2_roles"] == v2.get("roles"),
        "historical roles differ from complete bound V2 roles object",
    )
    _require(
        config["historical_v2_generation"] == v2.get("generation"),
        "historical generation differs from complete bound V2 generation object",
    )

    v3_binding = historical["v3_bounded_id_runner"]
    v3_path = ROOT / v3_binding["path"]
    _require(
        v3_path.resolve(strict=False)
        == (
            ROOT
            / "scripts"
            / "swe_task_state_v4_epistemic_chain_annotation_runner_v3.py"
        ).resolve(strict=False),
        "bound V3 runner path changed",
    )
    v3_hash = sha256_bytes(_read_file_bytes(v3_path, "bound V3 runner"))
    _require(
        v3_hash == _sha256(v3_binding["sha256"], "bound V3 runner hash"),
        "bound V3 runner bytes changed",
    )
    return AuthenticatedDraftConfig(
        value=copy.deepcopy(config),
        path=Path(path).resolve(strict=True),
        file_sha256=observed,
        v2_config_sha256=v2_hash,
        v3_runner_sha256=v3_hash,
    )


def load_draft_config(
    *, path: Path = CONFIG_PATH, expected_draft_config_sha256: str
) -> dict[str, Any]:
    """Load only after an independently supplied exact byte hash succeeds."""

    authenticated = authenticate_draft_config(
        path=path, expected_draft_config_sha256=expected_draft_config_sha256
    )
    return copy.deepcopy(dict(authenticated.value))


def validate_draft_config(value: Any) -> dict[str, Any]:
    config = _mapping(value, "draft config")
    expected_fields = {
        "schema_version",
        "kind",
        "id",
        "status",
        "scope",
        "observed_read_only_preflight",
        "authorization",
        "implementation",
        "authentication_contract",
        "frozen_historical_inputs",
        "historical_v2_roles",
        "historical_v2_generation",
        "future_exact_token_contract",
        "non_gate_cpu_route",
        "claim_scope",
        "blockers",
    }
    _require(set(config) == expected_fields, "draft config fields changed")
    _require(
        config["schema_version"] == SCHEMA_VERSION
        and config["kind"] == CONFIG_KIND
        and config["status"] == "source_only_draft_not_sealed_not_gate_ready",
        "draft config identity invalid",
    )

    scope = _mapping(config["scope"], "scope")
    _require(
        scope
        == {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "model_or_gpu_execution_performed": False,
            "artifacts_or_receipts_emitted": False,
        },
        "draft scope is not closed",
    )
    preflight = _mapping(
        config["observed_read_only_preflight"], "observed read-only preflight"
    )
    _require(
        preflight
        == {
            "provenance": "parent_reported_read_only_preflight_2026-07-20",
            "evidence_receipt_sha256": None,
            "authenticated_runtime_identity": False,
            "authorizes_execution": False,
            "gpu": {
                "model": "NVIDIA GeForce RTX 5090",
                "memory_total_gib": 32.0,
                "memory_free_gib_approx": 31.6,
            },
            "packages": {
                "vllm": "0.23.0",
                "torch": "2.11.0+cu130",
                "transformers": "5.12.1",
                "openai-harmony": "0.0.8",
                "xgrammar": "0.2.2",
            },
            "all_three_exact_snapshot_revisions_present": True,
            "used_by_source_only_plan_identity": False,
        },
        "preflight facts were changed or promoted to authenticated execution evidence",
    )
    authorization = _mapping(config["authorization"], "authorization")
    _require(
        set(authorization)
        == {
            "production_execution_authorized",
            "gate_eligible_execution_authorized",
            "model_or_gpu_access_authorized",
            "artifact_output_authorized",
            "production_receipt_emission_authorized",
        }
        and all(item is False for item in authorization.values()),
        "current draft authorizes execution or output",
    )

    implementation = _mapping(config["implementation"], "implementation")
    scaffold = _mapping(
        implementation.get("source_only_contract_scaffold"), "source-only scaffold"
    )
    _require(
        scaffold
        == {
            "path": "scripts/swe_task_state_v4_epistemic_chain_native_adapter_draft_v3.py",
            "hash_frozen": False,
            "sha256": None,
        },
        "source-only scaffold is incorrectly frozen",
    )
    expected_absent = {
        "production_native_adapter_present": False,
        "production_native_adapter_path": None,
        "production_native_adapter_sha256": None,
        "production_runtime_present": False,
        "production_runtime_path": None,
        "production_runtime_sha256": None,
        "production_receipt_emitter_present": False,
        "production_receipt_emitter_path": None,
        "production_receipt_emitter_sha256": None,
    }
    _require(
        {key: implementation.get(key) for key in expected_absent} == expected_absent
        and set(implementation) == {"source_only_contract_scaffold", *expected_absent},
        "a production implementation or implementation hash is present",
    )
    authentication_contract = _mapping(
        config["authentication_contract"], "authentication contract"
    )
    _require(
        authentication_contract
        == {
            "independent_expected_draft_config_sha256_required": True,
            "draft_config_bytes_authenticated_before_request_access": True,
            "bound_v2_config_path_and_sha256_required": True,
            "historical_roles_full_object_equality_to_bound_v2_required": True,
            "historical_generation_full_object_equality_to_bound_v2_required": True,
            "independent_expected_native_request_sha256_required": True,
            "authenticated_generation_context_required": True,
            "exact_native_request_reconstruction_required": True,
            "independent_expected_observation_sha256_required": True,
            "independent_expected_cpu_record_sha256_required": True,
        },
        "authentication contract weakened",
    )

    historical = _mapping(config["frozen_historical_inputs"], "historical inputs")
    _require(
        set(historical) == {"v2_runtime_config", "v3_bounded_id_runner"},
        "historical input fields invalid",
    )
    v2_binding = _mapping(historical["v2_runtime_config"], "V2 binding")
    v3_binding = _mapping(historical["v3_bounded_id_runner"], "V3 binding")
    _require(
        v2_binding
        == {
            "path": "configs/swe_task_state_v4_epistemic_chain_annotation_runner_v2.json",
            "sha256": "a22604d78938917972dc23c517c942fb9022e867e17ce0b5a060189dd0dd1b4d",
            "use": "historical_role_roster_and_engine_settings_only",
        }
        and v3_binding
        == {
            "path": "scripts/swe_task_state_v4_epistemic_chain_annotation_runner_v3.py",
            "sha256": "c35af19c9f3f1e38208ba7f23467c386ebff2e5fb6572582f9a620f51f394aeb",
            "native_request_kind": NATIVE_REQUEST_KIND,
        },
        "historical input binding changed",
    )

    roles = _mapping(config["historical_v2_roles"], "role roster")
    _require(
        set(roles) == {"independent_a", "independent_b", "adjudicator"},
        "historical role roster invalid",
    )
    for role_name, value in roles.items():
        role = _mapping(value, f"role {role_name}")
        _require(
            set(role)
            == {
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
            and role["execution_mode"] == "local_model"
            and SHA256_RE.fullmatch(str(role["snapshot_tree_sha256"])) is not None
            and isinstance(role["seed"], int)
            and not isinstance(role["seed"], bool)
            and role["output_extraction"]
            in {"direct_structured_text", "openai_harmony_final_channel"},
            f"role {role_name} identity invalid",
        )
    _require(
        roles["independent_b"]["output_extraction"]
        == "openai_harmony_final_channel"
        and roles["adjudicator"]["vllm_engine_kwargs"].get("tokenizer_mode")
        == "mistral",
        "family-specific role contract changed",
    )

    generation = _mapping(config["historical_v2_generation"], "generation")
    _require(
        generation.get("engine") == "vllm_offline_structured_outputs"
        and generation.get("temperature") == 0
        and generation.get("top_p") == 1.0
        and generation.get("max_output_tokens") == 768
        and generation.get("no_input_truncation") is True
        and generation.get("structured_outputs_config")
        == {"backend": "xgrammar", "disable_any_whitespace": True}
        and generation.get("prompt_token_accounting")
        == "vllm_request_output.prompt_token_ids",
        "historical generation contract changed",
    )
    exact = _mapping(config["future_exact_token_contract"], "exact-token contract")
    required_true = {
        "request_token_ids_passed_directly",
        "decode_before_submission_forbidden",
        "chat_template_rerender_forbidden",
        "retokenization_forbidden",
        "string_round_trip_forbidden",
        "input_truncation_forbidden",
        "engine_prompt_token_ids_exact_equality_required",
        "output_token_ids_required",
    }
    _require(
        exact.get("accepted_request_type") == "NativeGenerationRequest"
        and exact.get("accepted_request_kind") == NATIVE_REQUEST_KIND
        and exact.get("token_prompt_api") == "vllm.inputs.TokensPrompt"
        and exact.get("token_prompt_field") == "prompt_token_ids"
        and all(exact.get(key) is True for key in required_true)
        and exact.get("finish_reason_required") == "stop"
        and exact.get("sampling_seed_source") == "native_request.seed"
        and exact.get("response_schema_source") == "native_request.response_schema"
        and exact.get("structured_output_backend_required") == "xgrammar",
        "future exact-token contract weakened",
    )
    _require(
        exact.get("output_text_matches_authenticated_token_decode_required_for_real")
        is True
        and exact.get("source_only_output_text_token_decode_parity_verified")
        is False,
        "output token/text parity scope changed",
    )
    harmony = _mapping(exact.get("gpt_oss_extraction"), "GPT Harmony contract")
    _require(
        harmony
        == {
            "mode": "openai_harmony_final_channel",
            "parser_package": "openai-harmony",
            "strict": True,
            "exactly_one_final_channel_required": True,
        },
        "GPT Harmony extraction is not strict",
    )
    route = _mapping(config["non_gate_cpu_route"], "non-gate route")
    _require(
        route
        == {
            "api": "execute_scripted_cpu_non_gate",
            "record_kind": SCRIPTED_NON_GATE_RECORD_KIND,
            "actual_model_execution": False,
            "gate_eligible": False,
            "production_receipt": False,
        },
        "CPU route could be mistaken for production",
    )
    claims = _mapping(config["claim_scope"], "claim scope")
    _require(bool(claims) and all(item is False for item in claims.values()), "claim scope is not all false")
    blockers = config["blockers"]
    _require(
        isinstance(blockers, list)
        and len(blockers) >= 4
        and all(isinstance(item, str) and bool(item) for item in blockers),
        "draft blockers absent",
    )
    return copy.deepcopy(config)


def _authenticate_identity(
    value: Any,
    *,
    expected_sha256: str,
    expected_fields: set[str],
    label: str,
) -> tuple[dict[str, Any], str]:
    identity = _mapping(value, label)
    _require(set(identity) == expected_fields, f"{label} fields invalid")
    _require(
        all(isinstance(item, str) and bool(item) for item in identity.values()),
        f"{label} values invalid",
    )
    for field, item in identity.items():
        if field.endswith("_sha256"):
            _sha256(item, f"{label}.{field}")
    observed = sha256_value(identity)
    _require(
        _sha256(expected_sha256, f"expected {label} hash") == observed,
        f"{label} differs from external authenticated hash",
    )
    return identity, observed


@dataclass(frozen=True)
class DraftInvocationPlan:
    """Non-executing plan that binds a future adapter call byte-for-byte."""

    body: Mapping[str, Any]
    plan_sha256: str


@dataclass(frozen=True)
class NativeRequestReconstruction:
    """Exact host inputs used to independently rebuild one runner request."""

    generation_context: runner_v3.AuthenticatedNativeGenerationContext
    messages: Sequence[Mapping[str, str]]
    schema: Mapping[str, Any]
    seed: int
    stage: str
    annotation_pass: str
    packet_id_sha256: str
    source_id_sha256: str
    lineage_bindings: Mapping[str, Any]


def _authenticate_native_request(
    request: Any,
    *,
    expected_native_request_sha256: str,
    reconstruction: NativeRequestReconstruction,
) -> dict[str, Any]:
    expected = _sha256(
        expected_native_request_sha256, "expected native request hash"
    )
    _require(
        isinstance(request, runner_v3.NativeGenerationRequest),
        "native request type invalid",
    )
    _require(
        request.request_sha256 == expected,
        "native request differs from external authenticated hash",
    )
    _require(
        isinstance(reconstruction, NativeRequestReconstruction),
        "native request reconstruction inputs absent",
    )
    _require(
        isinstance(
            reconstruction.generation_context,
            runner_v3.AuthenticatedNativeGenerationContext,
        ),
        "authenticated native generation context absent",
    )
    try:
        body = runner_v3.validate_native_generation_request(
            request, context=reconstruction.generation_context
        )
    except Exception as error:
        raise NativeAdapterDraftError(f"native request invalid: {error}") from error
    body = copy.deepcopy(body)
    try:
        rebuilt = runner_v3.build_native_generation_request(
            context=reconstruction.generation_context,
            messages=copy.deepcopy(list(reconstruction.messages)),
            schema=copy.deepcopy(dict(reconstruction.schema)),
            seed=reconstruction.seed,
            stage=reconstruction.stage,
            annotation_pass=reconstruction.annotation_pass,
            packet_id_sha256=reconstruction.packet_id_sha256,
            source_id_sha256=reconstruction.source_id_sha256,
            lineage_bindings=copy.deepcopy(dict(reconstruction.lineage_bindings)),
        )
        rebuilt_body = runner_v3.validate_native_generation_request(
            rebuilt, context=reconstruction.generation_context
        )
        post_reconstruction_body = copy.deepcopy(
            runner_v3.validate_native_generation_request(
                request, context=reconstruction.generation_context
            )
        )
    except Exception as error:
        raise NativeAdapterDraftError(
            f"native request reconstruction failed: {error}"
        ) from error
    _require(
        rebuilt.request_sha256 == expected
        and rebuilt.request_sha256 == request.request_sha256
        and rebuilt_body == body,
        "native request differs from exact independently reconstructed request",
    )
    _require(
        post_reconstruction_body == body and request.request_sha256 == expected,
        "native request mutated during independent reconstruction",
    )
    _require(body.get("kind") == NATIVE_REQUEST_KIND, "native request kind changed")
    for identity_label in ("model_identity", "tokenizer_identity"):
        identity = _mapping(body[identity_label], f"request {identity_label}")
        for field, item in identity.items():
            if field.endswith("_sha256"):
                _sha256(item, f"request {identity_label}.{field}")
        _require(
            body[f"{identity_label}_sha256"] == sha256_value(identity),
            f"request {identity_label} exact hash invalid",
        )
    return body


def _expected_tokenizer_mode(role: Mapping[str, Any]) -> str:
    return str(role["vllm_engine_kwargs"].get("tokenizer_mode", "auto"))


def derive_source_only_invocation_plan(
    *,
    config_path: Path,
    expected_draft_config_sha256: str,
    role: str,
    request: Any,
    expected_native_request_sha256: str,
    request_reconstruction: NativeRequestReconstruction,
    runtime_identity: Mapping[str, Any],
    expected_runtime_identity_sha256: str,
    adapter_identity: Mapping[str, Any],
    expected_adapter_identity_sha256: str,
) -> DraftInvocationPlan:
    """Derive, but never execute, the exact future vLLM token invocation."""

    authenticated_config = authenticate_draft_config(
        path=config_path,
        expected_draft_config_sha256=expected_draft_config_sha256,
    )
    request_body = _authenticate_native_request(
        request,
        expected_native_request_sha256=expected_native_request_sha256,
        reconstruction=request_reconstruction,
    )
    return _derive_authenticated_source_only_invocation_plan(
        authenticated_config=authenticated_config,
        role=role,
        request=request,
        request_body=request_body,
        runtime_identity=runtime_identity,
        expected_runtime_identity_sha256=expected_runtime_identity_sha256,
        adapter_identity=adapter_identity,
        expected_adapter_identity_sha256=expected_adapter_identity_sha256,
    )


def _derive_authenticated_source_only_invocation_plan(
    *,
    authenticated_config: AuthenticatedDraftConfig,
    role: str,
    request: Any,
    request_body: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    expected_runtime_identity_sha256: str,
    adapter_identity: Mapping[str, Any],
    expected_adapter_identity_sha256: str,
) -> DraftInvocationPlan:
    validated_config = dict(authenticated_config.value)
    _require(role in validated_config["historical_v2_roles"], "role invalid")
    role_contract = validated_config["historical_v2_roles"][role]

    expected_model = {
        field: role_contract[field] for field in MODEL_IDENTITY_FIELDS
    }
    _require(
        request_body["model_identity"] == expected_model
        and request_body["model_identity_sha256"] == sha256_value(expected_model),
        "native request model identity differs from frozen role",
    )
    tokenizer_identity = _mapping(
        request_body["tokenizer_identity"], "request tokenizer identity"
    )
    _require(
        tokenizer_identity.get("repo_id") == role_contract["repo_id"]
        and tokenizer_identity.get("revision") == role_contract["revision"]
        and tokenizer_identity.get("snapshot_tree_sha256")
        == role_contract["snapshot_tree_sha256"]
        and tokenizer_identity.get("tokenizer_mode")
        == _expected_tokenizer_mode(role_contract)
        and request_body["tokenizer_identity_sha256"]
        == sha256_value(tokenizer_identity),
        "native request tokenizer identity differs from frozen role",
    )
    _require(
        request_body["chat_template_kwargs_sha256"]
        == sha256_value(role_contract["chat_template_kwargs"]),
        "native request chat-template kwargs differ from frozen role",
    )

    runtime, runtime_hash = _authenticate_identity(
        runtime_identity,
        expected_sha256=expected_runtime_identity_sha256,
        expected_fields=RUNTIME_IDENTITY_FIELDS,
        label="runtime identity",
    )
    _require(
        runtime["engine"] == "vllm_offline_structured_outputs"
        and runtime["package"] == "vllm",
        "future runtime identity is not vLLM offline structured output",
    )
    adapter, adapter_hash = _authenticate_identity(
        adapter_identity,
        expected_sha256=expected_adapter_identity_sha256,
        expected_fields=ADAPTER_IDENTITY_FIELDS,
        label="adapter identity",
    )
    _require(
        adapter["kind"] == "reviewed_production_native_adapter_v3"
        and adapter["entrypoint"] == "execute_native_token_request"
        and adapter["tokens_prompt_api"] == "vllm.inputs.TokensPrompt"
        and adapter["runner_sha256"]
        == validated_config["frozen_historical_inputs"]["v3_bounded_id_runner"][
            "sha256"
        ],
        "future adapter identity contract invalid",
    )

    prompt_ids = _token_ids(
        request_body["submitted_prompt_token_ids"], "request prompt token IDs"
    )
    schema = copy.deepcopy(request_body["response_schema"])
    generation = validated_config["historical_v2_generation"]
    sampling = {
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "max_tokens": generation["max_output_tokens"],
        "seed": request_body["seed"],
        "truncate_prompt_tokens": None,
        "structured_outputs": {"json": schema},
    }
    extraction_mode = role_contract["output_extraction"]
    extraction = (
        {
            "mode": "openai_harmony_final_channel",
            "parser_package": "openai-harmony",
            "strict": True,
            "exactly_one_final_channel_required": True,
            "analysis_or_tool_channels_excluded_from_text": True,
        }
        if extraction_mode == "openai_harmony_final_channel"
        else {"mode": "direct_structured_text"}
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": DRAFT_PLAN_KIND,
        "status": "source_only_non_executing_contract",
        "role": role,
        "draft_config_sha256": authenticated_config.file_sha256,
        "bound_v2_config_sha256": authenticated_config.v2_config_sha256,
        "bound_v3_runner_sha256": authenticated_config.v3_runner_sha256,
        "native_request_sha256": request.request_sha256,
        "native_request_kind": NATIVE_REQUEST_KIND,
        "native_request_exactly_reconstructed": True,
        "model_identity": copy.deepcopy(expected_model),
        "model_identity_sha256": request_body["model_identity_sha256"],
        "tokenizer_identity": copy.deepcopy(tokenizer_identity),
        "tokenizer_identity_sha256": request_body["tokenizer_identity_sha256"],
        "runtime_identity": copy.deepcopy(runtime),
        "runtime_identity_sha256": runtime_hash,
        "adapter_identity": copy.deepcopy(adapter),
        "adapter_identity_sha256": adapter_hash,
        "token_prompt": {
            "api": "vllm.inputs.TokensPrompt",
            "field": "prompt_token_ids",
            "prompt_token_ids": prompt_ids,
            "prompt_token_ids_sha256": sha256_value(prompt_ids),
        },
        "input_transformations": {
            "decoded_before_submission": False,
            "chat_template_rerendered": False,
            "retokenized": False,
            "string_round_trip_used": False,
            "truncation_allowed": False,
        },
        "engine": {
            "engine": generation["engine"],
            "role_vllm_engine_kwargs": copy.deepcopy(
                role_contract["vllm_engine_kwargs"]
            ),
            "max_model_len": generation["max_model_len"],
            "max_num_seqs": generation["max_num_seqs"],
            "max_num_batched_tokens": generation["max_num_batched_tokens"],
            "gpu_memory_utilization": generation["gpu_memory_utilization"],
            "structured_outputs_config": copy.deepcopy(
                generation["structured_outputs_config"]
            ),
            "vllm_use_flashinfer_sampler": generation[
                "vllm_use_flashinfer_sampler"
            ],
            "vllm_enable_v1_multiprocessing": generation[
                "vllm_enable_v1_multiprocessing"
            ],
            "vllm_disabled_kernels": copy.deepcopy(
                generation["vllm_disabled_kernels"]
            ),
        },
        "sampling_params": sampling,
        "sampling_params_sha256": sha256_value(sampling),
        "response_schema_sha256": request_body["response_schema_sha256"],
        "output_contract": {
            "engine_prompt_token_ids_must_equal_submitted": True,
            "output_token_ids_required": True,
            "finish_reason": "stop",
            "prompt_truncated": False,
            "output_text_token_decode_parity_verified": False,
            "future_real_adapter_requires_authenticated_tokenizer_reconstruction": True,
        },
        "output_extraction": extraction,
        "actual_model_execution": False,
        "gate_eligible": False,
        "production_receipt": False,
    }
    return DraftInvocationPlan(body=body, plan_sha256=sha256_value(body))


def validate_source_only_invocation_plan(
    plan: Any,
    *,
    config_path: Path,
    expected_draft_config_sha256: str,
    role: str,
    request: Any,
    expected_native_request_sha256: str,
    request_reconstruction: NativeRequestReconstruction,
    runtime_identity: Mapping[str, Any],
    expected_runtime_identity_sha256: str,
    adapter_identity: Mapping[str, Any],
    expected_adapter_identity_sha256: str,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    _config, _request, body = _authenticate_and_validate_plan(
        plan=plan,
        config_path=config_path,
        expected_draft_config_sha256=expected_draft_config_sha256,
        role=role,
        request=request,
        expected_native_request_sha256=expected_native_request_sha256,
        request_reconstruction=request_reconstruction,
        runtime_identity=runtime_identity,
        expected_runtime_identity_sha256=expected_runtime_identity_sha256,
        adapter_identity=adapter_identity,
        expected_adapter_identity_sha256=expected_adapter_identity_sha256,
        expected_plan_sha256=expected_plan_sha256,
    )
    return copy.deepcopy(body)


def _authenticate_and_validate_plan(
    *,
    plan: Any,
    config_path: Path,
    expected_draft_config_sha256: str,
    role: str,
    request: Any,
    expected_native_request_sha256: str,
    request_reconstruction: NativeRequestReconstruction,
    runtime_identity: Mapping[str, Any],
    expected_runtime_identity_sha256: str,
    adapter_identity: Mapping[str, Any],
    expected_adapter_identity_sha256: str,
    expected_plan_sha256: str,
) -> tuple[AuthenticatedDraftConfig, dict[str, Any], dict[str, Any]]:
    # Authentication order is deliberate: reject unauthenticated config bytes
    # before reading any request, context, identity, plan, backend, or output.
    authenticated_config = authenticate_draft_config(
        path=config_path,
        expected_draft_config_sha256=expected_draft_config_sha256,
    )
    request_body = _authenticate_native_request(
        request,
        expected_native_request_sha256=expected_native_request_sha256,
        reconstruction=request_reconstruction,
    )
    expected = _derive_authenticated_source_only_invocation_plan(
        authenticated_config=authenticated_config,
        role=role,
        request=request,
        request_body=request_body,
        runtime_identity=runtime_identity,
        expected_runtime_identity_sha256=expected_runtime_identity_sha256,
        adapter_identity=adapter_identity,
        expected_adapter_identity_sha256=expected_adapter_identity_sha256,
    )
    _require(isinstance(plan, DraftInvocationPlan), "draft plan type invalid")
    body = _mapping(plan.body, "draft plan body")
    _require(
        plan.plan_sha256 == sha256_value(body), "draft plan internal hash invalid"
    )
    _require(
        _sha256(expected_plan_sha256, "expected plan hash") == plan.plan_sha256,
        "draft plan differs from external authenticated hash",
    )
    _require(
        body == expected.body and plan.plan_sha256 == expected.plan_sha256,
        "draft plan differs from deterministic exact-token contract",
    )
    return authenticated_config, request_body, copy.deepcopy(body)


def _validate_authenticated_plan_shape(
    *,
    plan: DraftInvocationPlan,
    request: Any,
    request_body: Mapping[str, Any],
    authenticated_config: AuthenticatedDraftConfig,
    expected_native_request_sha256: str,
    expected_plan_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(isinstance(plan, DraftInvocationPlan), "draft plan type invalid")
    plan_body = _mapping(plan.body, "draft plan body")
    _require(
        plan.plan_sha256 == sha256_value(plan_body)
        and plan.plan_sha256 == _sha256(expected_plan_sha256, "expected plan hash"),
        "draft plan authentication failed",
    )
    _require(
        plan_body.get("kind") == DRAFT_PLAN_KIND
        and plan_body.get("status") == "source_only_non_executing_contract"
        and plan_body.get("draft_config_sha256")
        == authenticated_config.file_sha256
        and plan_body.get("bound_v2_config_sha256")
        == authenticated_config.v2_config_sha256
        and plan_body.get("bound_v3_runner_sha256")
        == authenticated_config.v3_runner_sha256
        and plan_body.get("native_request_sha256")
        == _sha256(expected_native_request_sha256, "expected native request hash")
        == request.request_sha256
        and plan_body.get("native_request_exactly_reconstructed") is True
        and plan_body.get("actual_model_execution") is False
        and plan_body.get("gate_eligible") is False
        and plan_body.get("production_receipt") is False,
        "draft plan identity or non-gate status invalid",
    )
    _require(
        plan_body.get("token_prompt", {}).get("prompt_token_ids")
        == request_body["submitted_prompt_token_ids"]
        and plan_body.get("token_prompt", {}).get("prompt_token_ids_sha256")
        == request_body["submitted_prompt_token_ids_sha256"],
        "draft plan does not preserve exact request token IDs",
    )
    expected_transformations = {
        "decoded_before_submission": False,
        "chat_template_rerendered": False,
        "retokenized": False,
        "string_round_trip_used": False,
        "truncation_allowed": False,
    }
    _require(
        plan_body.get("input_transformations") == expected_transformations,
        "draft plan permits an input transformation",
    )
    _require(
        plan_body.get("output_contract", {}).get(
            "output_text_token_decode_parity_verified"
        )
        is False
        and plan_body.get("output_contract", {}).get(
            "future_real_adapter_requires_authenticated_tokenizer_reconstruction"
        )
        is True,
        "source-only token/text parity was overstated",
    )
    return plan_body, request_body


def validate_draft_engine_observation(
    observation: Any,
    *,
    expected_observation_sha256: str,
    plan: DraftInvocationPlan,
    config_path: Path,
    expected_draft_config_sha256: str,
    role: str,
    request: Any,
    expected_native_request_sha256: str,
    request_reconstruction: NativeRequestReconstruction,
    runtime_identity: Mapping[str, Any],
    expected_runtime_identity_sha256: str,
    adapter_identity: Mapping[str, Any],
    expected_adapter_identity_sha256: str,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Validate a future-shaped observation without claiming model execution."""

    authenticated_config, request_body, plan_body = _authenticate_and_validate_plan(
        plan=plan,
        config_path=config_path,
        expected_draft_config_sha256=expected_draft_config_sha256,
        role=role,
        request=request,
        expected_native_request_sha256=expected_native_request_sha256,
        request_reconstruction=request_reconstruction,
        runtime_identity=runtime_identity,
        expected_runtime_identity_sha256=expected_runtime_identity_sha256,
        adapter_identity=adapter_identity,
        expected_adapter_identity_sha256=expected_adapter_identity_sha256,
        expected_plan_sha256=expected_plan_sha256,
    )
    _validate_authenticated_plan_shape(
        plan=plan,
        request=request,
        request_body=request_body,
        authenticated_config=authenticated_config,
        expected_native_request_sha256=expected_native_request_sha256,
        expected_plan_sha256=expected_plan_sha256,
    )
    value = _mapping(observation, "draft engine observation")
    _require(
        sha256_value(value)
        == _sha256(expected_observation_sha256, "expected observation hash"),
        "draft observation differs from external authenticated hash",
    )
    expected_fields = {
        "schema_version",
        "kind",
        "plan_sha256",
        "native_request_sha256",
        "response_schema_sha256",
        "seed",
        "sampling_params",
        "sampling_params_sha256",
        "structured_outputs_config",
        "submitted_prompt_token_ids",
        "submitted_prompt_token_ids_sha256",
        "engine_prompt_token_ids",
        "engine_prompt_token_ids_sha256",
        "prompt_truncated",
        "output_token_ids",
        "output_token_ids_sha256",
        "finish_reason",
        "text",
        "text_sha256",
        "output_extraction",
        "output_text_token_decode_parity_verified",
        "authenticated_tokenizer_decode_reconstruction_sha256",
        "actual_model_execution",
        "gate_eligible",
        "production_receipt",
    }
    _require(set(value) == expected_fields, "draft observation fields invalid")
    _require(
        value["schema_version"] == SCHEMA_VERSION
        and value["kind"] == DRAFT_OBSERVATION_KIND
        and value["plan_sha256"] == plan.plan_sha256
        and value["native_request_sha256"] == request.request_sha256
        and value["response_schema_sha256"]
        == request_body["response_schema_sha256"]
        and value["seed"] == request_body["seed"],
        "draft observation request, schema, or seed binding invalid",
    )
    _require(
        value["sampling_params"] == plan_body["sampling_params"]
        and value["sampling_params_sha256"]
        == sha256_value(value["sampling_params"])
        == plan_body["sampling_params_sha256"]
        and value["structured_outputs_config"]
        == plan_body["engine"]["structured_outputs_config"]
        == {"backend": "xgrammar", "disable_any_whitespace": True},
        "draft observation sampling or xgrammar schema contract changed",
    )
    submitted = _token_ids(
        value["submitted_prompt_token_ids"], "submitted prompt token IDs"
    )
    engine = _token_ids(value["engine_prompt_token_ids"], "engine prompt token IDs")
    output = _token_ids(value["output_token_ids"], "output token IDs")
    expected_prompt = list(request_body["submitted_prompt_token_ids"])
    _require(
        submitted == expected_prompt
        and engine == expected_prompt
        and value["submitted_prompt_token_ids_sha256"]
        == sha256_value(submitted)
        == request_body["submitted_prompt_token_ids_sha256"]
        and value["engine_prompt_token_ids_sha256"] == sha256_value(engine),
        "engine prompt token IDs differ from exact submitted request IDs",
    )
    _require(
        value["prompt_truncated"] is False
        and plan_body["sampling_params"]["truncate_prompt_tokens"] is None,
        "input truncation occurred or was enabled",
    )
    _require(
        value["output_token_ids_sha256"] == sha256_value(output),
        "output token IDs are not bound",
    )
    _require(value["finish_reason"] == "stop", "finish reason is not stop")
    text = value["text"]
    _require(
        isinstance(text, str)
        and bool(text)
        and value["text_sha256"] == sha256_bytes(text.encode("utf-8")),
        "output text authentication invalid",
    )
    _require(
        value["actual_model_execution"] is False
        and value["gate_eligible"] is False
        and value["production_receipt"] is False
        and value["output_text_token_decode_parity_verified"] is False
        and value["authenticated_tokenizer_decode_reconstruction_sha256"] is None,
        "draft observation asserts a real or gate-eligible execution",
    )

    extraction = _mapping(value["output_extraction"], "output extraction")
    expected_mode = plan_body["output_extraction"]["mode"]
    if expected_mode == "openai_harmony_final_channel":
        _require(
            extraction
            == {
                "mode": "openai_harmony_final_channel",
                "parser_package": "openai-harmony",
                "strict": True,
                "final_channel_count": 1,
                "analysis_or_tool_channels_excluded_from_text": True,
                "final_text_sha256": sha256_bytes(text.encode("utf-8")),
                "output_token_ids_sha256": sha256_value(output),
            },
            "GPT Harmony final-channel extraction is not strict and exact",
        )
    else:
        _require(
            extraction
            == {
                "mode": "direct_structured_text",
                "engine_text_used_directly": True,
                "text_sha256": sha256_bytes(text.encode("utf-8")),
            },
            "direct structured-text extraction invalid",
        )

    # Exercise the frozen runner's exact prompt/result token and finish checks.
    try:
        native_result = runner_v3.build_native_generation_result(
            request=request,
            text=text,
            submitted_prompt_token_ids=submitted,
            engine_prompt_token_ids=engine,
            output_token_ids=output,
            finish_reason=value["finish_reason"],
        )
        runner_v3.validate_native_generation_result(
            request=request, result=native_result
        )
    except Exception as error:
        raise NativeAdapterDraftError(
            f"frozen V3 native result contract failed: {error}"
        ) from error
    return copy.deepcopy(value)


@dataclass(frozen=True)
class ScriptedCpuNonGateRecord:
    """CPU-only record whose type and claims cannot represent a real run."""

    body: Mapping[str, Any]
    record_sha256: str


ScriptedObservation = Callable[[Mapping[str, Any], Any], Mapping[str, Any]]


def execute_scripted_cpu_non_gate(
    *,
    plan: DraftInvocationPlan,
    config_path: Path,
    expected_draft_config_sha256: str,
    role: str,
    request: Any,
    expected_native_request_sha256: str,
    request_reconstruction: NativeRequestReconstruction,
    runtime_identity: Mapping[str, Any],
    expected_runtime_identity_sha256: str,
    adapter_identity: Mapping[str, Any],
    expected_adapter_identity_sha256: str,
    expected_plan_sha256: str,
    expected_observation_sha256: str,
    expected_record_sha256: str,
    scripted_observation: ScriptedObservation,
) -> ScriptedCpuNonGateRecord:
    """Run one injected CPU callback; output remains irrevocably non-gate."""

    _config, _request_value, plan_body = _authenticate_and_validate_plan(
        plan=plan,
        config_path=config_path,
        expected_draft_config_sha256=expected_draft_config_sha256,
        role=role,
        request=request,
        expected_native_request_sha256=expected_native_request_sha256,
        request_reconstruction=request_reconstruction,
        runtime_identity=runtime_identity,
        expected_runtime_identity_sha256=expected_runtime_identity_sha256,
        adapter_identity=adapter_identity,
        expected_adapter_identity_sha256=expected_adapter_identity_sha256,
        expected_plan_sha256=expected_plan_sha256,
    )
    _require(callable(scripted_observation), "scripted CPU callback is not callable")
    callback_value = scripted_observation(copy.deepcopy(plan_body), request)
    observation = validate_draft_engine_observation(
        callback_value,
        expected_observation_sha256=expected_observation_sha256,
        plan=plan,
        config_path=config_path,
        expected_draft_config_sha256=expected_draft_config_sha256,
        role=role,
        request=request,
        expected_native_request_sha256=expected_native_request_sha256,
        request_reconstruction=request_reconstruction,
        runtime_identity=runtime_identity,
        expected_runtime_identity_sha256=expected_runtime_identity_sha256,
        adapter_identity=adapter_identity,
        expected_adapter_identity_sha256=expected_adapter_identity_sha256,
        expected_plan_sha256=expected_plan_sha256,
    )
    # Detect mutation through the plan dataclass's contained mappings.
    _authenticate_and_validate_plan(
        plan=plan,
        config_path=config_path,
        expected_draft_config_sha256=expected_draft_config_sha256,
        role=role,
        request=request,
        expected_native_request_sha256=expected_native_request_sha256,
        request_reconstruction=request_reconstruction,
        runtime_identity=runtime_identity,
        expected_runtime_identity_sha256=expected_runtime_identity_sha256,
        adapter_identity=adapter_identity,
        expected_adapter_identity_sha256=expected_adapter_identity_sha256,
        expected_plan_sha256=expected_plan_sha256,
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": SCRIPTED_NON_GATE_RECORD_KIND,
        "execution_mode": "scripted_cpu_non_gate",
        "plan_sha256": plan.plan_sha256,
        "native_request_sha256": request.request_sha256,
        "observation": observation,
        "observation_sha256": sha256_value(observation),
        "claims": {
            "actual_model_execution": False,
            "gate_eligible": False,
            "production_receipt": False,
            "sealed_control_evidence": False,
            "annotation_interface_readiness_established": False,
            "output_text_token_decode_parity_established": False,
        },
    }
    record = ScriptedCpuNonGateRecord(body=body, record_sha256=sha256_value(body))
    validate_scripted_cpu_non_gate_record(
        record,
        plan=plan,
        config_path=config_path,
        expected_draft_config_sha256=expected_draft_config_sha256,
        role=role,
        request=request,
        expected_native_request_sha256=expected_native_request_sha256,
        request_reconstruction=request_reconstruction,
        runtime_identity=runtime_identity,
        expected_runtime_identity_sha256=expected_runtime_identity_sha256,
        adapter_identity=adapter_identity,
        expected_adapter_identity_sha256=expected_adapter_identity_sha256,
        expected_plan_sha256=expected_plan_sha256,
        expected_observation_sha256=expected_observation_sha256,
        expected_record_sha256=expected_record_sha256,
    )
    return record


def validate_scripted_cpu_non_gate_record(
    record: Any,
    *,
    plan: DraftInvocationPlan,
    config_path: Path,
    expected_draft_config_sha256: str,
    role: str,
    request: Any,
    expected_native_request_sha256: str,
    request_reconstruction: NativeRequestReconstruction,
    runtime_identity: Mapping[str, Any],
    expected_runtime_identity_sha256: str,
    adapter_identity: Mapping[str, Any],
    expected_adapter_identity_sha256: str,
    expected_plan_sha256: str,
    expected_observation_sha256: str,
    expected_record_sha256: str,
) -> dict[str, Any]:
    # As with all plan-consuming validators, authenticate config and request
    # before inspecting the supplied record.
    _authenticate_and_validate_plan(
        plan=plan,
        config_path=config_path,
        expected_draft_config_sha256=expected_draft_config_sha256,
        role=role,
        request=request,
        expected_native_request_sha256=expected_native_request_sha256,
        request_reconstruction=request_reconstruction,
        runtime_identity=runtime_identity,
        expected_runtime_identity_sha256=expected_runtime_identity_sha256,
        adapter_identity=adapter_identity,
        expected_adapter_identity_sha256=expected_adapter_identity_sha256,
        expected_plan_sha256=expected_plan_sha256,
    )
    _require(
        isinstance(record, ScriptedCpuNonGateRecord), "scripted record type invalid"
    )
    body = _mapping(record.body, "scripted record body")
    _require(
        record.record_sha256 == sha256_value(body)
        and record.record_sha256
        == _sha256(expected_record_sha256, "expected scripted record hash"),
        "scripted record differs from external authenticated hash",
    )
    _require(
        set(body)
        == {
            "schema_version",
            "kind",
            "execution_mode",
            "plan_sha256",
            "native_request_sha256",
            "observation",
            "observation_sha256",
            "claims",
        },
        "scripted record fields invalid",
    )
    claims = _mapping(body["claims"], "scripted record claims")
    false_claim_fields = {
        "actual_model_execution",
        "gate_eligible",
        "production_receipt",
        "sealed_control_evidence",
        "annotation_interface_readiness_established",
        "output_text_token_decode_parity_established",
    }
    _require(
        body["schema_version"] == SCHEMA_VERSION
        and body["kind"] == SCRIPTED_NON_GATE_RECORD_KIND
        and body["execution_mode"] == "scripted_cpu_non_gate"
        and body["plan_sha256"] == plan.plan_sha256
        and body["native_request_sha256"] == request.request_sha256
        and set(claims) == false_claim_fields
        and all(claims[field] is False for field in false_claim_fields),
        "scripted record identity or false claims invalid",
    )
    observation = validate_draft_engine_observation(
        body["observation"],
        expected_observation_sha256=expected_observation_sha256,
        plan=plan,
        config_path=config_path,
        expected_draft_config_sha256=expected_draft_config_sha256,
        role=role,
        request=request,
        expected_native_request_sha256=expected_native_request_sha256,
        request_reconstruction=request_reconstruction,
        runtime_identity=runtime_identity,
        expected_runtime_identity_sha256=expected_runtime_identity_sha256,
        adapter_identity=adapter_identity,
        expected_adapter_identity_sha256=expected_adapter_identity_sha256,
        expected_plan_sha256=expected_plan_sha256,
    )
    _require(
        body["observation_sha256"] == sha256_value(observation),
        "scripted observation hash invalid",
    )
    return copy.deepcopy(body)


def execute_production_native_adapter(*_args: Any, **_kwargs: Any) -> None:
    """Unavailable in this source freeze; reject before touching any argument."""

    raise NativeAdapterDraftError(
        "production native adapter is absent and unauthorized in this source freeze"
    )


def execute_gate_eligible_native_adapter(*_args: Any, **_kwargs: Any) -> None:
    """Unavailable in this source freeze; reject before touching any argument."""

    raise NativeAdapterDraftError(
        "gate-eligible native execution is absent and unauthorized in this source freeze"
    )


def emit_production_gate_receipt(*_args: Any, **_kwargs: Any) -> None:
    """Unavailable in this source freeze; no receipt can be emitted."""

    raise NativeAdapterDraftError(
        "production or gate receipt emission is absent and unauthorized"
    )


__all__ = [
    "ADAPTER_IDENTITY_FIELDS",
    "AuthenticatedDraftConfig",
    "CONFIG_KIND",
    "CONFIG_PATH",
    "DRAFT_OBSERVATION_KIND",
    "DRAFT_PLAN_KIND",
    "DraftInvocationPlan",
    "FALSE_EXECUTION_CLAIMS",
    "NativeAdapterDraftError",
    "NativeRequestReconstruction",
    "PRODUCTION_CODE_FREEZE_COMPLETE",
    "RUNTIME_IDENTITY_FIELDS",
    "SCRIPTED_NON_GATE_RECORD_KIND",
    "ScriptedCpuNonGateRecord",
    "authenticate_draft_config",
    "derive_source_only_invocation_plan",
    "emit_production_gate_receipt",
    "execute_gate_eligible_native_adapter",
    "execute_production_native_adapter",
    "execute_scripted_cpu_non_gate",
    "load_draft_config",
    "sha256_value",
    "validate_draft_config",
    "validate_draft_engine_observation",
    "validate_scripted_cpu_non_gate_record",
    "validate_source_only_invocation_plan",
]
