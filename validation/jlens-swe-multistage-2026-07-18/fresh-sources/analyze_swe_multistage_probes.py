#!/usr/bin/env python3
"""Analyze paired gold, next-action, and outcome readouts across SWE stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/swe_stage_action_probes.json"
MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
MODEL_CONFIG_SHA256 = (
    "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338"
)
MODEL_INDEX_SHA256 = (
    "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2"
)
TOKENIZER_JSON_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
TOKENIZER_VOCABULARY_SIZE = 248_077
LOGIT_VOCABULARY_SIZE = 248_320
PUBLIC_LENS_REPO = "neuronpedia/jacobian-lens"
PUBLIC_LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
PUBLIC_LENS_SHA256 = (
    "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"
)
NF4_LENS_SHA256 = (
    "54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f"
)
NF4_PROVENANCE_SHA256 = (
    "08dfe022cc74bab226308dcd77a3901ec18806f8832dc8dec46524374fa3b1e7"
)
NATIVE_LENS_SHA256 = (
    "82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057"
)
NATIVE_STATE_SHA256 = (
    "f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6"
)
NATIVE_PROVENANCE_SHA256 = (
    "289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601"
)
ACTION_IDS = ("inspect", "edit", "validate", "finalize")
OUTCOME_IDS = ("success", "failure")
FIXED_LAYER_BAND = tuple(range(24, 48))
CAPTURE_LAYERS = tuple(range(16, 48))
ALL_SOURCE_LAYERS = tuple(range(63))
BOOTSTRAP_SEED = 36_027
DEFAULT_BOOTSTRAP_SAMPLES = 20_000
PROMPT_KIND = "swe_verified_multistage_probe"
ACTION_METADATA_KIND = "swe_verified_stage_action_probe_binding"
PROTOCOL_KIND = "swe_verified_stage_action_probe_protocol"
REPLAY_RUNTIME_IDENTITY = {
    "capture_adapter": "vLLM apply_model forward hooks",
    "enable_prefix_caching": True,
    "enforce_eager": True,
    "gpu_memory_utilization": 0.78,
    "kv_cache_dtype": "fp8_e4m3",
    "kv_offloading_backend": "native",
    "kv_offloading_size": 8.0,
    "language_model_only": True,
    "mamba_block_size": 1024,
    "max_model_len": 49_152,
    "max_num_batched_tokens": 4_096,
    "mtp_enabled": False,
    "readout_dtype": "torch.bfloat16",
    "stream_final_only": True,
    "timing_scope": "artifact resolution and validation through readout",
    "transport_dtype": "torch.float32",
}
REASONING_STATE_CONTRACT = {
    "included_in_action_accuracy": False,
    "labels": ["diagnosis_expressed"],
    "lens_scored_vocabulary": False,
    "representation": "independent_binary_multilabel_metadata",
}
ACCEPTED_TOKEN_POLICY = (
    "exclude_from_all_class_forms_and_mark_unscorable_if_a_contrast_cannot_be_retained"
)
CLASS_MARGIN_REDUCTION = (
    "expected_class_logmeanexp_minus_max_competing_class_logmeanexp"
)
POOLED_SENSITIVITY_ROLE = "sensitivity_only_not_a_decision_margin"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return value


def nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be a nonempty string")
    return value


def finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    )


def materialized_json_sha256(value: Any) -> str:
    return sha256_bytes(
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
            "ascii"
        )
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def logmeanexp(values: Sequence[float]) -> float:
    require(bool(values), "logmeanexp input must not be empty")
    maximum = max(values)
    return maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in values) / len(values)
    )


def quantile(values: Sequence[float], probability: float) -> float:
    require(bool(values), "quantile input must not be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def normalized_rank_utility(rank: int) -> float:
    return math.log(LOGIT_VOCABULARY_SIZE / rank) / math.log(LOGIT_VOCABULARY_SIZE)


def _class_records(
    value: Any, *, label: str, expected_ids: Sequence[str], global_ids: set[int]
) -> list[dict[str, Any]]:
    classes = sequence(value, label)
    require(
        [mapping(record, label).get("id") for record in classes] == list(expected_ids),
        f"{label} IDs/order mismatch",
    )
    result: list[dict[str, Any]] = []
    sizes: set[int] = set()
    for raw_record in classes:
        record = mapping(raw_record, label)
        class_id = str(record["id"])
        tokens: list[dict[str, Any]] = []
        for raw_token in sequence(record.get("tokens"), f"{label}.{class_id}.tokens"):
            token = mapping(raw_token, "class token")
            text = nonempty_string(token.get("text"), "class token text")
            token_id = token.get("token_id")
            require(
                text.startswith(" ")
                and not text.startswith("  ")
                and isinstance(token_id, int)
                and not isinstance(token_id, bool)
                and 0 <= token_id < TOKENIZER_VOCABULARY_SIZE,
                "class token is not a valid leading-space tokenizer token",
            )
            require(token_id not in global_ids, "action/outcome token IDs overlap")
            global_ids.add(token_id)
            tokens.append({"text": text, "token_id": token_id})
        require(bool(tokens), f"{label}.{class_id} has no tokens")
        sizes.add(len(tokens))
        result.append({"id": class_id, "tokens": tokens})
    require(len(sizes) == 1, f"{label} vocabulary sizes differ")
    return result


def validate_action_protocol(
    protocol: Mapping[str, Any], *, protocol_sha256: str
) -> dict[str, Any]:
    require(protocol.get("schema_version") == 1, "action protocol schema mismatch")
    require(protocol.get("kind") == PROTOCOL_KIND, "action protocol kind mismatch")
    require(
        protocol.get("lens_outputs_used_for_labels") is False,
        "action labels used lens outputs",
    )
    pins = mapping(protocol.get("pins"), "protocol pins")
    model = mapping(pins.get("model"), "model pin")
    tokenizer = mapping(pins.get("tokenizer"), "tokenizer pin")
    require(
        model.get("repo_id") == MODEL_REPO and model.get("revision") == MODEL_REVISION,
        "protocol model pin mismatch",
    )
    require(
        tokenizer.get("json_sha256") == TOKENIZER_JSON_SHA256
        and tokenizer.get("vocabulary_size") == TOKENIZER_VOCABULARY_SIZE,
        "protocol tokenizer pin mismatch",
    )
    band = mapping(protocol.get("fixed_layer_band"), "fixed layer band")
    require(
        band.get("start") == 24
        and band.get("end") == 47
        and band.get("end_inclusive") is True
        and band.get("layers") == list(FIXED_LAYER_BAND),
        "fixed layer band mismatch",
    )
    require(
        protocol.get("class_score_reduction") == "logmeanexp_over_class_tokens",
        "class score reduction mismatch",
    )
    require(
        protocol.get("class_margin_reduction") == CLASS_MARGIN_REDUCTION,
        "class margin reduction mismatch",
    )
    require(
        protocol.get("pooled_one_vs_rest_role") == POOLED_SENSITIVITY_ROLE,
        "pooled one-vs-rest role mismatch",
    )
    require(
        protocol.get("accepted_generated_token_policy") == ACCEPTED_TOKEN_POLICY,
        "accepted generated-token policy mismatch",
    )
    require(
        mapping(protocol.get("reasoning_state_contract"), "reasoning-state contract")
        == REASONING_STATE_CONTRACT,
        "reasoning-state contract mismatch",
    )
    classifier = mapping(
        protocol.get("next_completion_classifier"), "next-completion classifier"
    )
    require(
        classifier.get("action_precedence")
        == [
            "terminal_no_tool_response",
            "mutating_source_command",
            "test_command",
            "read_or_search_command",
            "validation_intent_assistant_text",
        ]
        and classifier.get("unclassified_policy") == "missing_not_imputed",
        "next-completion classifier contract mismatch",
    )
    diagnosis_regexes = sequence(
        classifier.get("diagnosis_assistant_text_regexes"),
        "diagnosis assistant-text regexes",
    )
    require(
        bool(diagnosis_regexes)
        and all(isinstance(value, str) and bool(value) for value in diagnosis_regexes),
        "diagnosis reasoning-state regex contract is invalid",
    )
    outcome_contract = mapping(
        protocol.get("outcome_control_contract"), "outcome-control contract"
    )
    require(
        mapping(
            outcome_contract.get("official_verdict_mapping"),
            "official verdict mapping",
        )
        == {
            "error": "failure",
            "incomplete": "failure",
            "resolved": "success",
            "unresolved": "failure",
        },
        "official verdict mapping mismatch",
    )
    global_ids: set[int] = set()
    actions = _class_records(
        protocol.get("action_classes"),
        label="action classes",
        expected_ids=ACTION_IDS,
        global_ids=global_ids,
    )
    outcomes = _class_records(
        protocol.get("outcome_classes"),
        label="outcome classes",
        expected_ids=OUTCOME_IDS,
        global_ids=global_ids,
    )
    return {
        "sha256": protocol_sha256,
        "action_classes": actions,
        "outcome_classes": outcomes,
        "action_ids": [token["token_id"] for row in actions for token in row["tokens"]],
        "outcome_ids": [token["token_id"] for row in outcomes for token in row["tokens"]],
        "reasoning_state_contract": dict(REASONING_STATE_CONTRACT),
    }


def _forms(
    value: Any,
    *,
    label: str,
    token_text: dict[int, str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw_form in sequence(value, label):
        form = mapping(raw_form, label)
        text = nonempty_string(form.get("text"), f"{label}.text")
        token_id = form.get("token_id")
        require(
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and 0 <= token_id < LOGIT_VOCABULARY_SIZE
            and token_id not in seen,
            f"{label} token ID is invalid or duplicated",
        )
        previous = token_text.setdefault(token_id, text)
        require(previous == text, f"token {token_id} has conflicting text")
        seen.add(token_id)
        result.append({"text": text, "token_id": token_id})
    require(bool(result), f"{label} must not be empty")
    return result


def _visibility_is_hidden(metadata: Mapping[str, Any]) -> bool:
    if metadata.get("analysis_role") != "oracle_hidden":
        return False
    audit = mapping(metadata.get("visibility_audit"), "visibility audit")
    records = sequence(audit.get("records"), "visibility records")
    require(bool(records), "hidden prompt has no visibility records")
    require(
        all(mapping(record, "visibility record").get("exposed") is False for record in records),
        "oracle_hidden prompt contains exposed target/foil evidence",
    )
    return True


def validate_prompt_bundle(
    prompts_value: Any,
    *,
    action_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    prompts = sequence(prompts_value, "prompt bundle")
    require(bool(prompts), "prompt bundle must not be empty")
    result: list[dict[str, Any]] = []
    prompt_ids: set[str] = set()
    task_stages: set[tuple[str, str]] = set()
    source_bundle_sha256: str | None = None
    reconstructed_source_prompts: list[dict[str, Any]] = []
    for index, raw_prompt in enumerate(prompts):
        prompt = mapping(raw_prompt, f"prompt[{index}]")
        prompt_id = nonempty_string(prompt.get("id"), f"prompt[{index}].id")
        require(prompt_id not in prompt_ids, f"duplicate prompt ID: {prompt_id}")
        prompt_ids.add(prompt_id)
        text = nonempty_string(prompt.get("text"), f"prompt {prompt_id}.text")
        token_ids = sequence(prompt.get("token_ids"), f"prompt {prompt_id}.token_ids")
        score_ids = sequence(
            prompt.get("score_token_ids"), f"prompt {prompt_id}.score_token_ids"
        )
        require(
            bool(token_ids)
            and all(isinstance(value, int) and not isinstance(value, bool) for value in token_ids),
            f"prompt {prompt_id} token IDs are invalid",
        )
        require(
            bool(score_ids)
            and len(score_ids) == len(set(score_ids))
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value < LOGIT_VOCABULARY_SIZE
                for value in score_ids
            ),
            f"prompt {prompt_id} scored token IDs are invalid",
        )
        metadata = mapping(prompt.get("metadata"), f"prompt {prompt_id}.metadata")
        require(metadata.get("kind") == PROMPT_KIND, f"prompt {prompt_id} kind mismatch")
        task = mapping(metadata.get("task"), "task metadata")
        stage = mapping(metadata.get("stage"), "stage metadata")
        instance_id = nonempty_string(task.get("instance_id"), "task instance ID")
        stage_id = nonempty_string(stage.get("id"), "stage ID")
        require(
            (instance_id, stage_id) not in task_stages,
            f"duplicate task/stage prompt: {instance_id}/{stage_id}",
        )
        task_stages.add((instance_id, stage_id))
        binding = mapping(metadata.get("stage_action_probe"), "stage-action binding")
        require(
            binding.get("schema_version") == 1
            and binding.get("kind") == ACTION_METADATA_KIND
            and binding.get("action_protocol_sha256") == action_protocol["sha256"]
            and binding.get("lifecycle_protocol_sha256")
            == metadata.get("lifecycle_protocol_sha256")
            and binding.get("exact_prompt_text_preserved") is True
            and binding.get("exact_prompt_token_ids_preserved") is True
            and binding.get("fixed_layer_band") == list(FIXED_LAYER_BAND),
            f"prompt {prompt_id} stage-action binding mismatch",
        )
        this_source_hash = nonempty_string(
            binding.get("source_prompt_bundle_sha256"), "source prompt bundle SHA"
        )
        if source_bundle_sha256 is None:
            source_bundle_sha256 = this_source_hash
        require(this_source_hash == source_bundle_sha256, "source prompt bundle hashes differ")
        original_score_ids = sequence(
            binding.get("source_score_token_ids"), "source scored token IDs"
        )
        require(
            binding.get("source_score_token_ids_sha256") == sha256_json(original_score_ids),
            "source scored-token hash mismatch",
        )
        expected_score_ids = list(
            dict.fromkeys(
                original_score_ids
                + action_protocol["action_ids"]
                + action_protocol["outcome_ids"]
            )
        )
        require(score_ids == expected_score_ids, "augmented scored vocabulary mismatch")
        require(
            binding.get("augmented_score_token_ids_sha256") == sha256_json(score_ids),
            "augmented scored-token hash mismatch",
        )
        source_prompt = json.loads(json.dumps(prompt))
        source_prompt["score_token_ids"] = original_score_ids
        del source_prompt["metadata"]["stage_action_probe"]
        require(
            binding.get("source_prompt_record_sha256") == sha256_json(source_prompt),
            "source prompt record hash mismatch",
        )
        reconstructed_source_prompts.append(source_prompt)
        vocabulary = mapping(binding.get("scored_vocabulary"), "action vocabulary")
        require(
            vocabulary.get("action_classes") == action_protocol["action_classes"]
            and vocabulary.get("outcome_classes") == action_protocol["outcome_classes"]
            and mapping(
                binding.get("reasoning_state_contract"), "bound reasoning-state contract"
            )
            == action_protocol["reasoning_state_contract"],
            "prompt action vocabulary differs from protocol",
        )
        next_completion = mapping(binding.get("next_completion"), "next completion label")
        expected_action = next_completion.get("expected_action_class")
        label_status = next_completion.get("label_status")
        require(
            (label_status == "available" and expected_action in ACTION_IDS)
            or (label_status == "missing" and expected_action is None),
            "next-action label/status mismatch",
        )
        require(
            next_completion.get("transition_outcome_class") in OUTCOME_IDS
            and next_completion.get("official_task_outcome_class") in OUTCOME_IDS
            and next_completion.get("lens_outputs_used_for_label") is False,
            "outcome labels are invalid",
        )
        diagnosis_hits = sequence(
            next_completion.get("diagnosis_regex_hits"),
            "next-completion diagnosis regex hits",
        )
        require(
            all(isinstance(value, str) and bool(value) for value in diagnosis_hits)
            and diagnosis_hits == sorted(set(diagnosis_hits)),
            "diagnosis regex hits are invalid or not canonical",
        )
        diagnosis_expressed = next_completion.get("diagnosis_expressed")
        require(
            isinstance(diagnosis_expressed, bool)
            and diagnosis_expressed is bool(diagnosis_hits),
            "diagnosis reasoning-state label does not match its regex evidence",
        )
        token_text: dict[int, str] = {}
        concepts: list[dict[str, Any]] = []
        for concept_index, raw_concept in enumerate(
            sequence(metadata.get("concepts"), "prompt concepts")
        ):
            concept = mapping(raw_concept, f"concept[{concept_index}]")
            target_forms = _forms(
                concept.get("forms"), label="target forms", token_text=token_text
            )
            foils: list[dict[str, Any]] = []
            for foil_index, raw_foil in enumerate(sequence(concept.get("foils"), "foils")):
                foil = mapping(raw_foil, f"foil[{foil_index}]")
                foil_forms = _forms(
                    foil.get("forms"), label="foil forms", token_text=token_text
                )
                foils.append({**dict(foil), "forms": foil_forms})
            require(bool(foils), "each gold concept requires at least one matched foil")
            concepts.append(
                {**dict(concept), "forms": target_forms, "foils": foils}
            )
        # The materializer orders each target immediately before its foils.
        concept_score_ids = [
            form["token_id"]
            for concept in concepts
            for forms in (
                [concept["forms"]] + [foil["forms"] for foil in concept["foils"]]
            )
            for form in forms
        ]
        require(
            original_score_ids == concept_score_ids,
            "source scored vocabulary is not the declared target/foil order",
        )
        for class_record in action_protocol["action_classes"] + action_protocol["outcome_classes"]:
            for token in class_record["tokens"]:
                previous = token_text.setdefault(token["token_id"], token["text"])
                require(previous == token["text"], "gold/action token text conflicts")
        hidden = _visibility_is_hidden(metadata)
        result.append(
            {
                "id": prompt_id,
                "text": text,
                "token_ids": list(token_ids),
                "score_token_ids": list(score_ids),
                "metadata": dict(metadata),
                "instance_id": instance_id,
                "stage_id": stage_id,
                "analysis_role": metadata.get("analysis_role"),
                "hidden_gold_eligible": hidden,
                "concepts": concepts,
                "token_text": token_text,
                "expected_action": expected_action,
                "transition_outcome": next_completion["transition_outcome_class"],
                "official_outcome": next_completion["official_task_outcome_class"],
                "reasoning_states": {
                    "diagnosis_expressed": diagnosis_expressed,
                    "diagnosis_regex_hits": list(diagnosis_hits),
                    "included_in_action_accuracy": False,
                },
            }
        )
    require(
        materialized_json_sha256(reconstructed_source_prompts) == source_bundle_sha256,
        "reconstructed source prompt bundle hash mismatch",
    )
    return {
        "prompts": result,
        "augmented_prompt_bundle_sha256": materialized_json_sha256(prompts),
        "source_prompt_bundle_sha256": source_bundle_sha256,
        "task_count": len({prompt["instance_id"] for prompt in result}),
    }


def validate_lens(lens: Mapping[str, Any], label: str) -> None:
    common = (
        lens.get("d_model") == 5120
        and lens.get("source_layers") == list(ALL_SOURCE_LAYERS)
        and lens.get("tensor_shape") == [5120, 5120]
    )
    require(common, f"{label} lens shape/layers mismatch")
    if label == "public":
        require(
            lens.get("repo_id") == PUBLIC_LENS_REPO
            and lens.get("revision") == PUBLIC_LENS_REVISION
            and lens.get("sha256") == PUBLIC_LENS_SHA256
            and lens.get("n_prompts") == 1000,
            "public n=1000 lens pin mismatch",
        )
    elif label == "nf4":
        require(
            lens.get("kind") == "local_fit"
            and lens.get("sha256") == NF4_LENS_SHA256
            and lens.get("provenance_sha256") == NF4_PROVENANCE_SHA256
            and lens.get("n_prompts") == 10
            and lens.get("fit_quantization") == "bitsandbytes-nf4-double-quant-bfloat16",
            "NF4 n=10 lens pin mismatch",
        )
    elif label == "native":
        require(
            lens.get("kind") == "native_nvfp4_ste_fit"
            and lens.get("sha256") == NATIVE_LENS_SHA256
            and lens.get("state_sha256") == NATIVE_STATE_SHA256
            and lens.get("provenance_sha256") == NATIVE_PROVENANCE_SHA256
            and lens.get("n_prompts") == 10
            and lens.get("fit_model") == MODEL_REPO
            and lens.get("fit_model_revision") == MODEL_REVISION,
            "native NVFP4 n=10 lens pin mismatch",
        )
    else:
        raise ValueError(f"unsupported report label {label!r}")


def _scored_evidence(
    readout: Any,
    *,
    label: str,
    expected_ids: Sequence[int],
    token_text: Mapping[int, str],
) -> dict[int, dict[str, Any]]:
    value = mapping(readout, label)
    records = sequence(value.get("scored_tokens"), f"{label}.scored_tokens")
    require(len(records) == len(expected_ids), f"{label} scored-token count mismatch")
    result: dict[int, dict[str, Any]] = {}
    for index, (raw_record, expected_id) in enumerate(zip(records, expected_ids, strict=True)):
        record = mapping(raw_record, f"{label}.scored_tokens[{index}]")
        token_id = record.get("token_id")
        rank = record.get("rank")
        require(token_id == expected_id, f"{label} scored-token order mismatch")
        require(
            isinstance(rank, int)
            and not isinstance(rank, bool)
            and 1 <= rank <= LOGIT_VOCABULARY_SIZE,
            f"{label} rank is invalid",
        )
        require(record.get("token") == token_text[token_id], f"{label} token text mismatch")
        result[token_id] = {
            "rank": rank,
            "score": finite(record.get("score"), f"{label}.score"),
            "logprob": finite(record.get("logprob"), f"{label}.logprob"),
        }
    return result


def validate_report(
    report: Mapping[str, Any],
    *,
    label: str,
    prompt_contract: Mapping[str, Any],
) -> dict[str, Any]:
    require(report.get("schema_version") == 3, f"{label} report schema mismatch")
    require(
        report.get("score_encoding") == "unrounded-float32",
        f"{label} score encoding mismatch",
    )
    validate_lens(mapping(report.get("lens"), f"{label}.lens"), label)
    model = mapping(report.get("model"), f"{label}.model")
    require(
        model.get("repo_id") == MODEL_REPO
        and model.get("revision") == MODEL_REVISION
        and model.get("config_sha256") == MODEL_CONFIG_SHA256
        and model.get("index_sha256") == MODEL_INDEX_SHA256,
        f"{label} model pin mismatch",
    )
    runtime = mapping(report.get("runtime"), f"{label}.runtime")
    runtime_identity = {
        field: runtime.get(field) for field in REPLAY_RUNTIME_IDENTITY
    }
    require(
        runtime_identity == REPLAY_RUNTIME_IDENTITY,
        f"{label} replay runtime identity mismatch",
    )
    assertions = mapping(report.get("assertions"), f"{label}.assertions")
    require(
        assertions.get("lens_hash_matches") is True
        and assertions.get("lens_metadata_matches") is True
        and assertions.get("model_architecture_matches") is True,
        f"{label} required integrity assertion failed",
    )
    experiments = sequence(report.get("experiments"), f"{label}.experiments")
    prompts = prompt_contract["prompts"]
    require(len(experiments) == len(prompts), f"{label} experiment count mismatch")
    union_ids = list(
        dict.fromkeys(token for prompt in prompts for token in prompt["score_token_ids"])
    )
    union_text = [
        next(prompt["token_text"][token] for prompt in prompts if token in prompt["token_text"])
        for token in union_ids
    ]
    vocabulary = mapping(report.get("scored_vocabulary"), f"{label}.scored_vocabulary")
    require(
        vocabulary.get("scope") == "global_plus_per_experiment"
        and vocabulary.get("token_ids") == []
        and vocabulary.get("tokens") == []
        and vocabulary.get("union_token_ids") == union_ids
        and vocabulary.get("union_tokens") == union_text,
        f"{label} report-level vocabulary mismatch",
    )
    rows: list[dict[str, Any]] = []
    eligibility = {
        "experiment_count": len(experiments),
        "greedy_top1_match": 0,
        "final_top5_match": 0,
        "final_norm_within_tolerance": 0,
        "final_logits_within_tolerance": 0,
    }
    all_top1 = True
    all_reconstruction = True
    for experiment_index, (raw_experiment, prompt) in enumerate(
        zip(experiments, prompts, strict=True)
    ):
        experiment = mapping(raw_experiment, f"{label}.experiment[{experiment_index}]")
        require(experiment.get("id") == prompt["id"], f"{label} prompt ID mismatch")
        require(experiment.get("prompt") == prompt["text"], f"{label} prompt text mismatch")
        require(
            experiment.get("prompt_token_ids") == prompt["token_ids"],
            f"{label} prompt token IDs mismatch",
        )
        require(experiment.get("metadata") == prompt["metadata"], f"{label} metadata mismatch")
        expected_position = len(prompt["token_ids"]) - 1
        require(
            experiment.get("positions_requested") == [-1]
            and experiment.get("positions_resolved") == [expected_position]
            and experiment.get("capture_positions_resolved") == [expected_position]
            and experiment.get("final_validation_position") == expected_position,
            f"{label} final-position contract mismatch",
        )
        scored = mapping(experiment.get("scored_vocabulary"), "experiment vocabulary")
        require(
            scored.get("token_ids") == prompt["score_token_ids"]
            and scored.get("tokens")
            == [prompt["token_text"][token] for token in prompt["score_token_ids"]],
            f"{label} per-prompt vocabulary mismatch",
        )
        generated = experiment.get("generated_token_id")
        require(
            isinstance(generated, int)
            and not isinstance(generated, bool)
            and 0 <= generated < LOGIT_VOCABULARY_SIZE,
            f"{label} generated token is invalid",
        )
        top1 = experiment.get("final_layer_top1_matches_greedy")
        final_norm = mapping(experiment.get("final_norm_reconstruction"), "final norm")
        final_logits = mapping(
            experiment.get("final_logits_reconstruction"), "final logits"
        )
        norm_within = final_norm.get("within_tolerance")
        logits_within = final_logits.get("within_tolerance")
        top5 = final_logits.get("top_k_prefix_token_ids_match")
        require(
            all(isinstance(value, bool) for value in (top1, norm_within, logits_within, top5)),
            f"{label} reconstruction eligibility flags are invalid",
        )
        eligibility["greedy_top1_match"] += int(top1)
        eligibility["final_top5_match"] += int(top5)
        eligibility["final_norm_within_tolerance"] += int(norm_within)
        eligibility["final_logits_within_tolerance"] += int(logits_within)
        all_top1 = all_top1 and top1
        all_reconstruction = all_reconstruction and norm_within and logits_within
        layers = sequence(experiment.get("layers"), f"{label}.layers")
        layer_ids = [mapping(layer, "layer").get("layer") for layer in layers]
        require(
            layer_ids == list(CAPTURE_LAYERS),
            f"{label} does not exactly cover capture layers 16 through 47",
        )
        evidence = {"jacobian": {}, "logit": {}}
        logit_readouts: list[dict[str, Any]] = []
        for raw_layer in layers:
            layer = mapping(raw_layer, "layer")
            layer_id = layer["layer"]
            positions = sequence(layer.get("positions"), "layer positions")
            require(len(positions) == 1, f"{label} layer position count mismatch")
            position = mapping(positions[0], "position")
            require(
                position.get("capture_index") == 0
                and position.get("token_position") == expected_position,
                f"{label} layer position mismatch",
            )
            for method, field in (("jacobian", "jacobian_lens"), ("logit", "logit_lens")):
                evidence[method][layer_id] = _scored_evidence(
                    position.get(field),
                    label=f"{label}.{prompt['id']}.layer-{layer_id}.{method}",
                    expected_ids=prompt["score_token_ids"],
                    token_text=prompt["token_text"],
                )
            if layer_id in FIXED_LAYER_BAND:
                logit_readouts.append(dict(mapping(position.get("logit_lens"), "logit readout")))
        residual = mapping(experiment.get("residual_capture_manifest"), "residual manifest")
        residual_sha = nonempty_string(residual.get("sha256"), "residual SHA")
        require(len(residual_sha) == 64, "residual SHA is malformed")
        rows.append(
            {
                "prompt": prompt,
                "generated_token_id": generated,
                "evidence": evidence,
                "layer_ids": layer_ids,
                "numerically_certified": bool(
                    top1 and norm_within and logits_within and top5
                ),
                "pair_binding": {
                    "id": prompt["id"],
                    "prompt": experiment.get("prompt"),
                    "prompt_token_ids": experiment.get("prompt_token_ids"),
                    "metadata": experiment.get("metadata"),
                    "scored_vocabulary": dict(scored),
                    "generated_token_id": generated,
                    "residual_capture_manifest": dict(residual),
                    "logit_fixed_band_sha256": sha256_json(logit_readouts),
                    "final_layer_top1_matches_greedy": top1,
                    "final_norm_reconstruction": dict(final_norm),
                    "final_logits_reconstruction": dict(final_logits),
                },
            }
        )
    require(
        assertions.get("all_final_layer_top1_match_greedy") is all_top1,
        f"{label} aggregate greedy assertion mismatch",
    )
    require(
        assertions.get("all_final_adapter_reconstructions_within_tolerance")
        is all_reconstruction,
        f"{label} aggregate adapter assertion mismatch",
    )
    expected_status = "passed" if all_top1 and all_reconstruction else "failed"
    require(report.get("status") == expected_status, f"{label} report status mismatch")
    return {
        "label": label,
        "rows": rows,
        "runtime_identity": runtime_identity,
        "numerical_eligibility": {
            **eligibility,
            "report_status": report.get("status"),
            "all_greedy_top1_match": all_top1,
            "all_adapter_reconstructions_within_tolerance": all_reconstruction,
        },
    }


def validate_pairing(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(reports) >= 2, "public and native reports are required")
    reference = reports[0]
    for report in reports[1:]:
        require(
            report["runtime_identity"] == reference["runtime_identity"],
            "paired runtime identity mismatch",
        )
        require(len(report["rows"]) == len(reference["rows"]), "paired row count mismatch")
        for left, right in zip(reference["rows"], report["rows"], strict=True):
            require(
                left["pair_binding"] == right["pair_binding"],
                f"reports differ in prompt/residual/logit evidence: {left['prompt']['id']}",
            )
            require(left["layer_ids"] == right["layer_ids"], "paired layer coverage differs")
    return {
        "report_labels": [report["label"] for report in reports],
        "prompt_count": len(reference["rows"]),
        "exact_prompt_token_metadata_vocabulary_pairing": True,
        "residual_capture_manifests_equal": True,
        "fixed_band_logit_readouts_equal": True,
        "runtime_identity_equal": True,
        "runtime_identity": dict(reference["runtime_identity"]),
    }


def _group_lme(
    evidence: Mapping[int, Mapping[str, float]],
    forms: Sequence[Mapping[str, Any]],
    field: str,
) -> float:
    return logmeanexp([evidence[int(form["token_id"])][field] for form in forms])


def _stability(values: Sequence[float]) -> dict[str, Any]:
    positives = sum(value > 0.0 for value in values)
    negatives = sum(value < 0.0 for value in values)
    zeroes = len(values) - positives - negatives
    return {
        "layer_count": len(values),
        "mean": statistics.fmean(values),
        "population_standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "positive_layer_fraction": positives / len(values),
        "negative_layer_fraction": negatives / len(values),
        "zero_layer_fraction": zeroes / len(values),
        "dominant_sign_fraction": max(positives, negatives, zeroes) / len(values),
    }


def gold_metric(
    evidence: Mapping[int, Mapping[int, Mapping[str, Any]]],
    concepts: Sequence[Mapping[str, Any]],
    *,
    generated_token_id: int,
) -> dict[str, Any]:
    retained: list[dict[str, Any]] = []
    exclusion_audit: list[dict[str, Any]] = []
    for concept in concepts:
        target_forms = [
            form for form in concept["forms"] if form["token_id"] != generated_token_id
        ]
        foils = []
        for foil in concept["foils"]:
            foil_forms = [
                form for form in foil["forms"] if form["token_id"] != generated_token_id
            ]
            if foil_forms:
                foils.append({**dict(foil), "forms": foil_forms})
        exclusion_audit.append(
            {
                "concept_id": concept.get("id"),
                "excluded_target_form_count": len(concept["forms"]) - len(target_forms),
                "retained_target_form_count": len(target_forms),
                "retained_foil_count": len(foils),
            }
        )
        if target_forms and foils:
            retained.append({**dict(concept), "forms": target_forms, "foils": foils})
    if not retained:
        return {
            "scorable": False,
            "reason": "no target/foil contrast remains after accepted-token exclusion",
            "accepted_generated_token_id": generated_token_id,
            "accepted_token_exclusion_audit": exclusion_audit,
            "fixed_layer_band": list(FIXED_LAYER_BAND),
        }
    curve: list[dict[str, Any]] = []
    for layer in FIXED_LAYER_BAND:
        concept_score_margins: list[float] = []
        concept_logprob_margins: list[float] = []
        for concept in retained:
            target_score = _group_lme(evidence[layer], concept["forms"], "score")
            target_logprob = _group_lme(evidence[layer], concept["forms"], "logprob")
            foil_scores = [
                _group_lme(evidence[layer], foil["forms"], "score")
                for foil in concept["foils"]
            ]
            foil_logprobs = [
                _group_lme(evidence[layer], foil["forms"], "logprob")
                for foil in concept["foils"]
            ]
            concept_score_margins.append(target_score - statistics.fmean(foil_scores))
            concept_logprob_margins.append(
                target_logprob - statistics.fmean(foil_logprobs)
            )
        curve.append(
            {
                "layer": layer,
                "score_margin": statistics.fmean(concept_score_margins),
                "logprob_margin": statistics.fmean(concept_logprob_margins),
            }
        )
    legacy_contrasts: list[float] = []
    for concept in retained:
        target_rank = min(
            evidence[layer][form["token_id"]]["rank"]
            for layer in FIXED_LAYER_BAND
            for form in concept["forms"]
        )
        foil_utilities = []
        for foil in concept["foils"]:
            foil_rank = min(
                evidence[layer][form["token_id"]]["rank"]
                for layer in FIXED_LAYER_BAND
                for form in foil["forms"]
            )
            foil_utilities.append(normalized_rank_utility(foil_rank))
        legacy_contrasts.append(
            normalized_rank_utility(target_rank) - statistics.fmean(foil_utilities)
        )
    score_values = [row["score_margin"] for row in curve]
    logprob_values = [row["logprob_margin"] for row in curve]
    return {
        "scorable": True,
        "metric_role": "primary_hidden_gold_same_layer_target_vs_matched_foil",
        "fixed_layer_band": list(FIXED_LAYER_BAND),
        "concept_count": len(retained),
        "declared_concept_count": len(concepts),
        "accepted_generated_token_id": generated_token_id,
        "accepted_token_exclusion_audit": exclusion_audit,
        "score_margin": statistics.fmean(score_values),
        "logprob_margin": statistics.fmean(logprob_values),
        "per_layer_curve": curve,
        "layer_stability": {
            "score_margin": _stability(score_values),
            "logprob_margin": _stability(logprob_values),
        },
        "legacy_min_rank_sensitivity": {
            "metric_role": "sensitivity_not_primary",
            "normalized_utility_target_minus_foil": statistics.fmean(legacy_contrasts),
        },
        "legacy_min_rank_utility_margin": statistics.fmean(legacy_contrasts),
    }


def class_metric(
    evidence: Mapping[int, Mapping[int, Mapping[str, Any]]],
    classes: Sequence[Mapping[str, Any]],
    expected_class: str,
    *,
    role: str,
    generated_token_id: int,
) -> dict[str, Any]:
    declared_class_order = [record["id"] for record in classes]
    require(expected_class in declared_class_order, "expected class is absent from vocabulary")
    retained_classes: list[dict[str, Any]] = []
    exclusion_audit: list[dict[str, Any]] = []
    for record in classes:
        retained_tokens = [
            token
            for token in record["tokens"]
            if token["token_id"] != generated_token_id
        ]
        exclusion_audit.append(
            {
                "class_id": record["id"],
                "excluded_form_count": len(record["tokens"]) - len(retained_tokens),
                "retained_form_count": len(retained_tokens),
            }
        )
        if retained_tokens:
            retained_classes.append({**dict(record), "tokens": retained_tokens})
    class_order = [record["id"] for record in retained_classes]
    if (
        len(retained_classes) != len(classes)
        or expected_class not in class_order
        or len(class_order) < 2
    ):
        return {
            "scorable": False,
            "reason": "accepted-token exclusion removed a declared class contrast",
            "metric_role": role,
            "expected_class": expected_class,
            "fixed_layer_band": list(FIXED_LAYER_BAND),
            "accepted_generated_token_id": generated_token_id,
            "accepted_token_exclusion_audit": exclusion_audit,
        }
    curve: list[dict[str, Any]] = []
    band_class_scores: dict[str, list[float]] = {class_id: [] for class_id in class_order}
    band_class_logprobs: dict[str, list[float]] = {class_id: [] for class_id in class_order}
    for layer in FIXED_LAYER_BAND:
        class_scores = {
            record["id"]: _group_lme(evidence[layer], record["tokens"], "score")
            for record in retained_classes
        }
        class_logprobs = {
            record["id"]: _group_lme(evidence[layer], record["tokens"], "logprob")
            for record in retained_classes
        }
        alternative_tokens = [
            token
            for record in retained_classes
            if record["id"] != expected_class
            for token in record["tokens"]
        ]
        competing_order = [class_id for class_id in class_order if class_id != expected_class]
        strongest_competing = max(
            competing_order, key=lambda class_id: class_scores[class_id]
        )
        pooled_alternative_score = _group_lme(
            evidence[layer], alternative_tokens, "score"
        )
        pooled_alternative_logprob = _group_lme(
            evidence[layer], alternative_tokens, "logprob"
        )
        predicted = max(class_order, key=lambda class_id: class_scores[class_id])
        for class_id in class_order:
            band_class_scores[class_id].append(class_scores[class_id])
            band_class_logprobs[class_id].append(class_logprobs[class_id])
        curve.append(
            {
                "layer": layer,
                "expected_score_margin": (
                    class_scores[expected_class] - class_scores[strongest_competing]
                ),
                "expected_logprob_margin": (
                    class_logprobs[expected_class]
                    - class_logprobs[strongest_competing]
                ),
                "strongest_competing_class": strongest_competing,
                "pooled_one_vs_rest_score_margin": (
                    class_scores[expected_class] - pooled_alternative_score
                ),
                "pooled_one_vs_rest_logprob_margin": (
                    class_logprobs[expected_class] - pooled_alternative_logprob
                ),
                "predicted_class": predicted,
                "correct": predicted == expected_class,
                "class_scores": class_scores,
                "class_logprobs": class_logprobs,
            }
        )
    aggregate_scores = {
        class_id: statistics.fmean(values) for class_id, values in band_class_scores.items()
    }
    aggregate_logprobs = {
        class_id: statistics.fmean(values)
        for class_id, values in band_class_logprobs.items()
    }
    predicted = max(class_order, key=lambda class_id: aggregate_scores[class_id])
    competing_order = [class_id for class_id in class_order if class_id != expected_class]
    strongest_competing = max(
        competing_order, key=lambda class_id: aggregate_scores[class_id]
    )
    score_margins = [row["expected_score_margin"] for row in curve]
    logprob_margins = [row["expected_logprob_margin"] for row in curve]
    pooled_score_margins = [row["pooled_one_vs_rest_score_margin"] for row in curve]
    pooled_logprob_margins = [
        row["pooled_one_vs_rest_logprob_margin"] for row in curve
    ]
    return {
        "scorable": True,
        "metric_role": role,
        "expected_class": expected_class,
        "fixed_layer_band": list(FIXED_LAYER_BAND),
        "class_score_reduction": "logmeanexp_over_class_tokens",
        "primary_margin_reduction": CLASS_MARGIN_REDUCTION,
        "accepted_generated_token_id": generated_token_id,
        "accepted_token_exclusion_audit": exclusion_audit,
        "accepted_generated_token_overlapped_class_vocabulary": any(
            audit["excluded_form_count"] for audit in exclusion_audit
        ),
        "declared_class_ids": declared_class_order,
        "retained_class_ids": class_order,
        "expected_score_margin": (
            aggregate_scores[expected_class] - aggregate_scores[strongest_competing]
        ),
        "expected_logprob_margin": (
            aggregate_logprobs[expected_class]
            - aggregate_logprobs[strongest_competing]
        ),
        "band_strongest_competing_class": strongest_competing,
        "band_class_scores": aggregate_scores,
        "band_class_logprobs": aggregate_logprobs,
        "band_predicted_class": predicted,
        "band_correct": predicted == expected_class,
        "per_layer_accuracy": statistics.fmean(float(row["correct"]) for row in curve),
        "per_layer_expected_score_margin_mean": statistics.fmean(score_margins),
        "per_layer_expected_logprob_margin_mean": statistics.fmean(logprob_margins),
        "pooled_one_vs_rest_sensitivity": {
            "metric_role": POOLED_SENSITIVITY_ROLE,
            "reduction": "logmeanexp_over_all_tokens_in_other_classes",
            "expected_score_margin": statistics.fmean(pooled_score_margins),
            "expected_logprob_margin": statistics.fmean(pooled_logprob_margins),
        },
        "per_layer_curve": curve,
        "layer_stability": {
            "expected_score_margin": _stability(score_margins),
            "expected_logprob_margin": _stability(logprob_margins),
        },
    }


def analyze_rows(
    reports: Sequence[Mapping[str, Any]], action_protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    report_by_label = {report["label"]: report for report in reports}
    methods: list[tuple[str, str, str]] = [
        ("public_jacobian", "public", "jacobian"),
    ]
    if "nf4" in report_by_label:
        methods.append(("nf4_jacobian", "nf4", "jacobian"))
    methods.extend(
        [
            ("native_jacobian", "native", "jacobian"),
            ("logit", "public", "logit"),
        ]
    )
    details: list[dict[str, Any]] = []
    for row_index, public_row in enumerate(report_by_label["public"]["rows"]):
        prompt = public_row["prompt"]
        method_metrics: dict[str, Any] = {}
        method_certification: dict[str, bool] = {}
        for method, report_label, readout_kind in methods:
            report_row = report_by_label[report_label]["rows"][row_index]
            evidence = report_row["evidence"][readout_kind]
            generated_token_id = report_row["generated_token_id"]
            metrics: dict[str, Any] = {
                "gold_hidden": (
                    gold_metric(
                        evidence,
                        prompt["concepts"],
                        generated_token_id=generated_token_id,
                    )
                    if prompt["hidden_gold_eligible"]
                    else None
                ),
                "next_action": (
                    class_metric(
                        evidence,
                        action_protocol["action_classes"],
                        prompt["expected_action"],
                        role="next_captured_completion_action",
                        generated_token_id=generated_token_id,
                    )
                    if prompt["expected_action"] is not None
                    else None
                ),
                "transition_outcome_control": class_metric(
                    evidence,
                    action_protocol["outcome_classes"],
                    prompt["transition_outcome"],
                    role="captured_transition_outcome_control",
                    generated_token_id=generated_token_id,
                ),
                "official_outcome_control": class_metric(
                    evidence,
                    action_protocol["outcome_classes"],
                    prompt["official_outcome"],
                    role="future_official_swe_verified_outcome_control",
                    generated_token_id=generated_token_id,
                ),
            }
            method_metrics[method] = metrics
            method_certification[method] = bool(report_row["numerically_certified"])
        details.append(
            {
                "id": prompt["id"],
                "instance_id": prompt["instance_id"],
                "stage_id": prompt["stage_id"],
                "analysis_role": prompt["analysis_role"],
                "hidden_gold_eligible": prompt["hidden_gold_eligible"],
                "expected_action_class": prompt["expected_action"],
                "transition_outcome_class": prompt["transition_outcome"],
                "official_outcome_class": prompt["official_outcome"],
                "reasoning_states": dict(prompt["reasoning_states"]),
                "method_numerical_certification": method_certification,
                "methods": method_metrics,
            }
        )
    return details


def _bootstrap_difference(
    pairs: Sequence[tuple[float, float]], *, label: str, samples: int
) -> dict[str, Any]:
    differences = [left - right for left, right in pairs]
    point = statistics.fmean(differences)
    if len(differences) == 1 or samples == 0:
        return {
            "paired_task_count": len(differences),
            "mean_difference": point,
            "confidence_interval_95": None,
            "inference": "descriptive_only",
        }
    seed = BOOTSTRAP_SEED + int(hashlib.sha256(label.encode("ascii")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    estimates = [
        statistics.fmean(rng.choice(differences) for _ in differences)
        for _ in range(samples)
    ]
    return {
        "paired_task_count": len(differences),
        "mean_difference": point,
        "confidence_interval_95": [quantile(estimates, 0.025), quantile(estimates, 0.975)],
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "resampling_unit": "task",
        "inference": "paired_task_level_percentile_bootstrap",
    }


def aggregate_track(
    details: Sequence[Mapping[str, Any]],
    *,
    track: str,
    fields: Sequence[str],
    bootstrap_samples: int,
    certified_only: bool,
) -> dict[str, Any]:
    stages = sorted({str(row["stage_id"]) for row in details})
    stage_sets: dict[str, list[Mapping[str, Any]]] = {
        stage: [row for row in details if row["stage_id"] == stage] for stage in stages
    }
    stage_sets["ALL_AVAILABLE_STAGES"] = list(details)
    output: dict[str, Any] = {}
    for stage, rows in stage_sets.items():
        methods = list(rows[0]["methods"]) if rows else []
        field_output: dict[str, Any] = {}
        for field in fields:
            method_task_values: dict[str, dict[str, dict[str, float]]] = {
                method: {} for method in methods
            }
            eligible_row_counts = {method: 0 for method in methods}
            uncertified_row_counts = {method: 0 for method in methods}
            for row in rows:
                task = str(row["instance_id"])
                for method in methods:
                    metric = row["methods"][method][track]
                    if metric is None or not metric.get("scorable", True):
                        continue
                    certified = bool(row["method_numerical_certification"][method])
                    if not certified:
                        uncertified_row_counts[method] += 1
                    if certified_only and not certified:
                        continue
                    value = float(metric[field])
                    prompt_values = method_task_values[method].setdefault(task, {})
                    require(row["id"] not in prompt_values, "duplicate aggregate prompt ID")
                    prompt_values[str(row["id"])] = value
                    eligible_row_counts[method] += 1
            collapsed = {
                method: {
                    task: statistics.fmean(values.values())
                    for task, values in task_values.items()
                }
                for method, task_values in method_task_values.items()
            }
            method_summary = {
                method: {
                    "task_count": len(task_values),
                    "eligible_row_count": eligible_row_counts[method],
                    "excluded_uncertified_row_count": (
                        uncertified_row_counts[method] if certified_only else 0
                    ),
                    "equal_task_mean": (
                        statistics.fmean(task_values.values()) if task_values else None
                    ),
                    "within_task_reduction": (
                        "equal mean over available stages"
                        if stage == "ALL_AVAILABLE_STAGES"
                        else "one prompt per task/stage"
                    ),
                }
                for method, task_values in collapsed.items()
            }
            cohort_curves: dict[str, list[dict[str, Any]]] = {}
            for method in methods:
                task_layer_values: dict[str, dict[int, list[float]]] = {}
                for row in rows:
                    if certified_only and not row["method_numerical_certification"][method]:
                        continue
                    metric = row["methods"][method][track]
                    if metric is None or not metric.get("scorable", True):
                        continue
                    curve = metric.get("per_layer_curve")
                    if not isinstance(curve, list):
                        continue
                    task = str(row["instance_id"])
                    for layer_row in curve:
                        layer = int(layer_row["layer"])
                        if field == "band_correct":
                            value = float(layer_row["correct"])
                        elif field in layer_row:
                            value = float(layer_row[field])
                        else:
                            continue
                        task_layer_values.setdefault(task, {}).setdefault(layer, []).append(
                            value
                        )
                curve_rows: list[dict[str, Any]] = []
                for layer in FIXED_LAYER_BAND:
                    task_values = [
                        statistics.fmean(values_by_layer[layer])
                        for values_by_layer in task_layer_values.values()
                        if layer in values_by_layer
                    ]
                    if task_values:
                        curve_rows.append(
                            {
                                "layer": layer,
                                "equal_task_mean": statistics.fmean(task_values),
                                "task_count": len(task_values),
                            }
                        )
                cohort_curves[method] = curve_rows
            comparisons: dict[str, Any] = {}

            def paired_task_values(
                left_method: str, right_method: str
            ) -> tuple[list[tuple[float, float]], int]:
                pairs: list[tuple[float, float]] = []
                paired_prompt_count = 0
                common_tasks = sorted(
                    set(method_task_values[left_method])
                    & set(method_task_values[right_method])
                )
                for task in common_tasks:
                    left_prompts = method_task_values[left_method][task]
                    right_prompts = method_task_values[right_method][task]
                    common_prompts = sorted(set(left_prompts) & set(right_prompts))
                    if not common_prompts:
                        continue
                    pairs.append(
                        (
                            statistics.fmean(left_prompts[prompt] for prompt in common_prompts),
                            statistics.fmean(right_prompts[prompt] for prompt in common_prompts),
                        )
                    )
                    paired_prompt_count += len(common_prompts)
                return pairs, paired_prompt_count

            for method in methods:
                if method == "logit":
                    continue
                pairs, paired_prompt_count = paired_task_values(method, "logit")
                if pairs:
                    comparison = _bootstrap_difference(
                        pairs,
                        label=f"{track}:{stage}:{field}:{method}",
                        samples=bootstrap_samples,
                    )
                    comparison["paired_prompt_count"] = paired_prompt_count
                    comparisons[f"{method}_minus_logit"] = comparison
            direct_pairs = (
                ("native_jacobian", "public_jacobian"),
                ("nf4_jacobian", "public_jacobian"),
                ("native_jacobian", "nf4_jacobian"),
            )
            for left_method, right_method in direct_pairs:
                if left_method not in collapsed or right_method not in collapsed:
                    continue
                pairs, paired_prompt_count = paired_task_values(left_method, right_method)
                if pairs:
                    comparison = _bootstrap_difference(
                        pairs,
                        label=(
                            f"{track}:{stage}:{field}:{left_method}:"
                            f"minus:{right_method}"
                        ),
                        samples=bootstrap_samples,
                    )
                    comparison["paired_prompt_count"] = paired_prompt_count
                    comparisons[f"{left_method}_minus_{right_method}"] = comparison
            field_output[field] = {
                "methods": method_summary,
                "paired_comparisons": comparisons,
                "cohort_per_layer_curves": cohort_curves,
                "cohort_layer_stability": {
                    method: (
                        _stability([row["equal_task_mean"] for row in curve])
                        if curve
                        else None
                    )
                    for method, curve in cohort_curves.items()
                },
            }
        output[stage] = {
            "metrics": field_output,
            "numerical_row_policy": (
                "certified_rows_only"
                if certified_only
                else "includes_uncertified_rows_sensitivity_only"
            ),
        }
    return output


def _classification_summary(
    details: Sequence[Mapping[str, Any]],
    *,
    track: str,
    class_ids: Sequence[str],
    certified_only: bool,
) -> dict[str, Any]:
    methods = list(details[0]["methods"]) if details else []
    output: dict[str, Any] = {}
    for method in methods:
        support = {class_id: 0 for class_id in class_ids}
        predicted_support = {class_id: 0 for class_id in class_ids}
        confusion = {
            expected: {predicted: 0 for predicted in class_ids}
            for expected in class_ids
        }
        score_margins = {class_id: [] for class_id in class_ids}
        logprob_margins = {class_id: [] for class_id in class_ids}
        task_correct = {class_id: {} for class_id in class_ids}
        task_score_margins = {class_id: {} for class_id in class_ids}
        task_logprob_margins = {class_id: {} for class_id in class_ids}
        eligible_rows = 0
        correct_rows = 0
        excluded_uncertified_rows = 0
        task_ids: set[str] = set()
        for row in details:
            metric = row["methods"][method][track]
            if metric is None or not metric.get("scorable", True):
                continue
            certified = bool(row["method_numerical_certification"][method])
            if certified_only and not certified:
                excluded_uncertified_rows += 1
                continue
            expected = str(metric["expected_class"])
            predicted = str(metric["band_predicted_class"])
            require(expected in support, f"{track} expected class is outside its contract")
            require(predicted in support, f"{track} predicted class is outside its contract")
            support[expected] += 1
            predicted_support[predicted] += 1
            confusion[expected][predicted] += 1
            score_margins[expected].append(float(metric["expected_score_margin"]))
            logprob_margins[expected].append(float(metric["expected_logprob_margin"]))
            task = str(row["instance_id"])
            task_correct[expected].setdefault(task, []).append(float(expected == predicted))
            task_score_margins[expected].setdefault(task, []).append(
                float(metric["expected_score_margin"])
            )
            task_logprob_margins[expected].setdefault(task, []).append(
                float(metric["expected_logprob_margin"])
            )
            eligible_rows += 1
            correct_rows += int(expected == predicted)
            task_ids.add(task)
        observed = [class_id for class_id in class_ids if support[class_id] > 0]
        missing = [class_id for class_id in class_ids if support[class_id] == 0]
        per_class_recall = {
            class_id: (
                confusion[class_id][class_id] / support[class_id]
                if support[class_id]
                else None
            )
            for class_id in class_ids
        }
        recalls = [float(per_class_recall[class_id]) for class_id in observed]
        per_class_score_margin = {
            class_id: (
                statistics.fmean(score_margins[class_id])
                if score_margins[class_id]
                else None
            )
            for class_id in class_ids
        }
        per_class_logprob_margin = {
            class_id: (
                statistics.fmean(logprob_margins[class_id])
                if logprob_margins[class_id]
                else None
            )
            for class_id in class_ids
        }
        per_class_equal_task_accuracy = {
            class_id: (
                statistics.fmean(
                    statistics.fmean(values)
                    for values in task_correct[class_id].values()
                )
                if task_correct[class_id]
                else None
            )
            for class_id in class_ids
        }
        per_class_equal_task_score_margin = {
            class_id: (
                statistics.fmean(
                    statistics.fmean(values)
                    for values in task_score_margins[class_id].values()
                )
                if task_score_margins[class_id]
                else None
            )
            for class_id in class_ids
        }
        per_class_equal_task_logprob_margin = {
            class_id: (
                statistics.fmean(
                    statistics.fmean(values)
                    for values in task_logprob_margins[class_id].values()
                )
                if task_logprob_margins[class_id]
                else None
            )
            for class_id in class_ids
        }
        if eligible_rows:
            majority_class = max(class_ids, key=lambda class_id: support[class_id])
            majority_accuracy = support[majority_class] / eligible_rows
            micro_accuracy = correct_rows / eligible_rows
            observed_macro_recall = statistics.fmean(recalls)
            equal_class_accuracy = statistics.fmean(
                float(per_class_equal_task_accuracy[class_id])
                for class_id in observed
            )
            equal_class_score_margin = statistics.fmean(
                float(per_class_equal_task_score_margin[class_id])
                for class_id in observed
            )
            equal_class_logprob_margin = statistics.fmean(
                float(per_class_equal_task_logprob_margin[class_id])
                for class_id in observed
            )
            raw_micro_score_margin = statistics.fmean(
                value for values in score_margins.values() for value in values
            )
            raw_micro_logprob_margin = statistics.fmean(
                value for values in logprob_margins.values() for value in values
            )
        else:
            majority_class = None
            majority_accuracy = None
            micro_accuracy = None
            observed_macro_recall = None
            equal_class_accuracy = None
            equal_class_score_margin = None
            equal_class_logprob_margin = None
            raw_micro_score_margin = None
            raw_micro_logprob_margin = None
        output[method] = {
            "inference": "descriptive_only",
            "task_count": len(task_ids),
            "eligible_row_count": eligible_rows,
            "excluded_uncertified_row_count": excluded_uncertified_rows,
            "class_support": support,
            "class_task_support": {
                class_id: len(task_correct[class_id]) for class_id in class_ids
            },
            "predicted_class_support": predicted_support,
            "confusion_matrix_expected_by_predicted": confusion,
            "observed_class_ids": observed,
            "missing_class_ids": missing,
            "per_class_recall": per_class_recall,
            "majority_class": majority_class,
            "majority_baseline_accuracy": majority_accuracy,
            "raw_micro_accuracy_secondary": micro_accuracy,
            "raw_micro_accuracy_minus_majority_baseline": (
                micro_accuracy - majority_accuracy
                if micro_accuracy is not None and majority_accuracy is not None
                else None
            ),
            "observed_class_macro_recall": observed_macro_recall,
            "balanced_accuracy_observed_classes": (
                observed_macro_recall if len(observed) >= 2 else None
            ),
            "balanced_accuracy_status": (
                "available_observed_classes_only"
                if len(observed) >= 2
                else "unavailable_fewer_than_two_observed_classes"
            ),
            "equal_observed_class_accuracy_primary": equal_class_accuracy,
            "per_class_equal_task_accuracy": per_class_equal_task_accuracy,
            "per_class_expected_score_margin_raw_rows": per_class_score_margin,
            "per_class_expected_logprob_margin_raw_rows": per_class_logprob_margin,
            "per_class_equal_task_expected_score_margin": (
                per_class_equal_task_score_margin
            ),
            "per_class_equal_task_expected_logprob_margin": (
                per_class_equal_task_logprob_margin
            ),
            "equal_observed_class_expected_score_margin_primary": (
                equal_class_score_margin
            ),
            "equal_observed_class_expected_logprob_margin_primary": (
                equal_class_logprob_margin
            ),
            "raw_micro_expected_score_margin_secondary": raw_micro_score_margin,
            "raw_micro_expected_logprob_margin_secondary": raw_micro_logprob_margin,
            "classification_identifiability": (
                "observed_multiclass"
                if len(observed) >= 2
                else "degenerate_single_or_no_observed_class"
            ),
        }
    return {
        "class_order": list(class_ids),
        "primary_aggregation": (
            "equal mean within task and class, across tasks within class, then across "
            "observed classes"
        ),
        "secondary_aggregation": "raw_micro_over_eligible_rows",
        "missing_classes_are_reported_not_imputed": True,
        "numerical_row_policy": (
            "certified_rows_only"
            if certified_only
            else "includes_uncertified_rows_sensitivity_only"
        ),
        "methods": output,
    }


def classification_track_summary(
    details: Sequence[Mapping[str, Any]],
    *,
    track: str,
    class_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "certified_primary": _classification_summary(
            details,
            track=track,
            class_ids=class_ids,
            certified_only=True,
        ),
        "uncertified_inclusive_sensitivity": _classification_summary(
            details,
            track=track,
            class_ids=class_ids,
            certified_only=False,
        ),
    }


def build_analysis(
    *,
    prompt_contract: Mapping[str, Any],
    action_protocol: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
    pairing: Mapping[str, Any],
    bootstrap_samples: int,
) -> dict[str, Any]:
    details = analyze_rows(reports, action_protocol)
    task_count = prompt_contract["task_count"]
    def aggregate_views(track: str, fields: Sequence[str]) -> dict[str, Any]:
        return {
            "certified_primary": aggregate_track(
                details,
                track=track,
                fields=fields,
                bootstrap_samples=bootstrap_samples,
                certified_only=True,
            ),
            "uncertified_inclusive_sensitivity": aggregate_track(
                details,
                track=track,
                fields=fields,
                bootstrap_samples=bootstrap_samples,
                certified_only=False,
            ),
        }

    aggregates = {
        "gold_hidden": aggregate_views(
            "gold_hidden",
            (
                "score_margin",
                "logprob_margin",
                "legacy_min_rank_utility_margin",
            ),
        ),
        "next_action": {
            **aggregate_views(
                "next_action",
                ("expected_score_margin", "expected_logprob_margin", "band_correct"),
            ),
            "all_available_stages_classification": classification_track_summary(
                details,
                track="next_action",
                class_ids=ACTION_IDS,
            ),
        },
        "transition_outcome_control": {
            **aggregate_views(
                "transition_outcome_control",
                ("expected_score_margin", "band_correct"),
            ),
            "all_available_stages_classification": classification_track_summary(
                details,
                track="transition_outcome_control",
                class_ids=OUTCOME_IDS,
            ),
        },
        "official_outcome_control": {
            **aggregate_views(
                "official_outcome_control",
                ("expected_score_margin", "band_correct"),
            ),
            "all_available_stages_classification": classification_track_summary(
                details,
                track="official_outcome_control",
                class_ids=OUTCOME_IDS,
            ),
        },
    }
    certified_row_count = sum(
        int(row["numerically_certified"])
        for report in reports
        for row in report["rows"]
    )
    total_report_row_count = sum(len(report["rows"]) for report in reports)
    numerical_status = (
        "certified"
        if certified_row_count == total_report_row_count
        else "partially_uncertified_primary_excludes_failed_rows"
    )
    return {
        "schema_version": 2,
        "kind": "swe_verified_multistage_probe_analysis",
        "status": "descriptive_n1_development" if task_count == 1 else "multi_task_pilot",
        "numerical_status": numerical_status,
        "interpretation_contract": {
            "n1_is_descriptive_only": task_count == 1,
            "task_count": task_count,
            "primary_gold_metric": (
                "same-layer target-vs-matched-foil score/logprob margin, equal mean "
                "over fixed layers 24..47"
            ),
            "gold_eligibility": (
                "only analysis_role=oracle_hidden with all leakage records exposed=false"
            ),
            "accepted_token_policy": (
                "exclude target/foil/action/outcome forms equal to the accepted generated "
                "token; mark unscorable rather than impute"
            ),
            "action_metric": (
                "expected-class logmeanexp minus the maximum competing-class logmeanexp; "
                "pooled one-vs-rest is sensitivity only"
            ),
            "action_classes": list(ACTION_IDS),
            "reasoning_state_labels": (
                "diagnosis_expressed is independent metadata and is excluded from action "
                "classification and accuracy"
            ),
            "classification_primary_aggregation": (
                "equal mean over observed classes with explicit missing-class support"
            ),
            "classification_secondary_aggregation": "raw micro over eligible rows",
            "numerical_certification": (
                "primary aggregates use only rows with greedy, norm, logits, and top-five "
                "reconstruction checks passing; inclusive results are sensitivity only"
            ),
            "legacy_min_rank": "reported only as a sensitivity metric; never primary",
            "cohort_aggregation": "equal mean within task, then equal mean across tasks",
            "labels_selected_from_lens_outputs": False,
        },
        "fixed_layer_band": list(FIXED_LAYER_BAND),
        "capture_layers": list(CAPTURE_LAYERS),
        "action_protocol_sha256": action_protocol["sha256"],
        "augmented_prompt_bundle_sha256": prompt_contract[
            "augmented_prompt_bundle_sha256"
        ],
        "source_prompt_bundle_sha256": prompt_contract["source_prompt_bundle_sha256"],
        "pairing": dict(pairing),
        "numerical_certification": {
            "certified_report_row_count": certified_row_count,
            "total_report_row_count": total_report_row_count,
            "all_report_rows_certified": certified_row_count == total_report_row_count,
            "uncertified_rows_retained_only_in_sensitivity_views": True,
        },
        "report_lenses": {
            report["label"]: {
                "n_prompts": (
                    1000 if report["label"] == "public" else 10
                ),
                "numerical_eligibility": report["numerical_eligibility"],
            }
            for report in reports
        },
        "bootstrap": {
            "samples": bootstrap_samples,
            "base_seed": BOOTSTRAP_SEED,
            "resampling_unit": "task",
        },
        "aggregates": aggregates,
        "prompt_details": details,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--action-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--nf4-report", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require(args.bootstrap_samples >= 0, "bootstrap sample count must be nonnegative")
    protocol_bytes = args.action_protocol.expanduser().resolve(strict=True).read_bytes()
    protocol = validate_action_protocol(
        mapping(json.loads(protocol_bytes), "action protocol"),
        protocol_sha256=sha256_bytes(protocol_bytes),
    )
    prompt_path = args.prompts.expanduser().resolve(strict=True)
    prompt_contract = validate_prompt_bundle(
        json.loads(prompt_path.read_bytes()), action_protocol=protocol
    )
    report_paths = [("public", args.public_report)]
    if args.nf4_report is not None:
        report_paths.append(("nf4", args.nf4_report))
    report_paths.append(("native", args.native_report))
    reports = [
        validate_report(
            mapping(json.loads(path.expanduser().resolve(strict=True).read_bytes()), label),
            label=label,
            prompt_contract=prompt_contract,
        )
        for label, path in report_paths
    ]
    pairing = validate_pairing(reports)
    analysis = build_analysis(
        prompt_contract=prompt_contract,
        action_protocol=protocol,
        reports=reports,
        pairing=pairing,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = args.output.expanduser().resolve()
    atomic_write_json(output, analysis)
    print(
        f"wrote {output} ({analysis['status']}, {len(analysis['prompt_details'])} prompts, "
        f"sha256={sha256_bytes(output.read_bytes())})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
