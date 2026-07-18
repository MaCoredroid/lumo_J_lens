#!/usr/bin/env python3
"""Validate and analyze a frozen multi-task SWE checkpoint J-lens pilot.

The input prompt bundle is the runner input itself.  Its metadata is the frozen
analysis contract: task identity, fixed layer band, target concepts, eligible
one-token forms, and optional matched foils.  Reports are rejected unless they
reproduce the prompts and per-prompt scored vocabularies exactly.
"""

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
LOGIT_VOCABULARY_SIZE = 248_320
PUBLIC_LENS_REPO = "neuronpedia/jacobian-lens"
PUBLIC_LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
PUBLIC_LENS_SHA256 = (
    "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"
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
ALL_SOURCE_LAYERS = tuple(range(63))
FIXED_MIDDLE_LAYERS = tuple(range(16, 48))
PASS_K = (1, 5, 10, 50, 100, 1000)
CONCEPT_FAMILIES = frozenset(
    ("file_stem", "module_dir", "hunk_symbol", "replacement")
)
BOOTSTRAP_SEED = 36_027
BOOTSTRAP_SAMPLES = 20_000
CHECKPOINT_CONTRACTS = {
    "C0": {
        "id": "C0",
        "name": "task_start",
        "visibility_boundary": "before_first_assistant_token",
    },
    "C0M": {
        "id": "C0M",
        "name": "capture_matched_task_start",
        "visibility_boundary": "same_captured_trajectory_before_first_assistant_token",
    },
    "C1": {
        "id": "C1",
        "name": "post_first_repository_observation",
        "visibility_boundary": (
            "after_successful_repository_read_or_search_before_second_assistant_token"
        ),
    },
}


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


def valid_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def valid_token_ids(value: Any, label: str, *, nonempty: bool = True) -> list[int]:
    result = sequence(value, label)
    require(not nonempty or bool(result), f"{label} must not be empty")
    require(
        all(
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and 0 <= token_id < LOGIT_VOCABULARY_SIZE
            for token_id in result
        ),
        f"{label} must contain valid token IDs",
    )
    return result


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def materialized_json_sha256(value: Any) -> str:
    raw = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    return sha256_bytes(raw)


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    require(bool(ordered), "quantile input must not be empty")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def derived_seed(label: str, seed: int) -> int:
    digest = hashlib.sha256(label.encode("ascii")).hexdigest()
    return seed + int(digest[:8], 16)


def normalized_utility(rank: int) -> float:
    require(1 <= rank <= LOGIT_VOCABULARY_SIZE, "rank is outside the LM-head vocabulary")
    return math.log(LOGIT_VOCABULARY_SIZE / rank) / math.log(
        LOGIT_VOCABULARY_SIZE
    )


def _validate_forms(
    value: Any,
    *,
    label: str,
    target: str,
    token_text: dict[int, str],
) -> list[dict[str, Any]]:
    forms = sequence(value, label)
    require(bool(forms), f"{label} must not be empty")
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, raw_form in enumerate(forms):
        form = mapping(raw_form, f"{label}[{index}]")
        text = nonempty_string(form.get("text"), f"{label}[{index}].text")
        token_id = form.get("token_id")
        require(
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and 0 <= token_id < LOGIT_VOCABULARY_SIZE
            and token_id not in seen,
            f"{label}[{index}].token_id is invalid or duplicated",
        )
        require(
            text in (target, f" {target}"),
            f"{label}[{index}] is not a bare or leading-space target form",
        )
        previous = token_text.setdefault(token_id, text)
        require(previous == text, f"token {token_id} has conflicting decoded forms")
        seen.add(token_id)
        result.append({"text": text, "token_id": token_id})
    return result


def validate_prompt_bundle(
    prompts_value: Sequence[Mapping[str, Any]],
    *,
    expected_checkpoint: Mapping[str, str] = CHECKPOINT_CONTRACTS["C0"],
) -> dict[str, Any]:
    prompts = sequence(prompts_value, "prompt bundle")
    require(bool(prompts), "prompt bundle must not be empty")
    require(
        dict(expected_checkpoint) in CHECKPOINT_CONTRACTS.values(),
        "expected checkpoint metadata is not a supported exact contract",
    )
    normalized: list[dict[str, Any]] = []
    prompt_ids: set[str] = set()
    task_ids: set[str] = set()
    protocol_hash: str | None = None
    global_token_text: dict[int, str] = {}

    for prompt_index, raw_prompt in enumerate(prompts):
        prompt = mapping(raw_prompt, f"prompt[{prompt_index}]")
        identifier = nonempty_string(prompt.get("id"), f"prompt[{prompt_index}].id")
        require(identifier not in prompt_ids, f"duplicate prompt ID: {identifier}")
        prompt_ids.add(identifier)
        text = nonempty_string(prompt.get("text"), f"prompt {identifier}.text")
        token_ids = valid_token_ids(prompt.get("token_ids"), f"prompt {identifier}.token_ids")
        score_token_ids = valid_token_ids(
            prompt.get("score_token_ids"), f"prompt {identifier}.score_token_ids"
        )
        require(
            len(score_token_ids) == len(set(score_token_ids)),
            f"prompt {identifier}.score_token_ids contains duplicates",
        )
        require(
            "target_token_id" not in prompt,
            f"prompt {identifier} must not override the accepted checkpoint token",
        )

        metadata = mapping(prompt.get("metadata"), f"prompt {identifier}.metadata")
        require(
            metadata.get("kind") == "swe_verified_multitask_initial_probe",
            f"prompt {identifier} metadata kind mismatch",
        )
        current_protocol = valid_sha256(
            metadata.get("protocol_sha256"), f"prompt {identifier}.protocol_sha256"
        )
        if protocol_hash is None:
            protocol_hash = current_protocol
        require(
            current_protocol == protocol_hash,
            f"prompt {identifier} protocol SHA-256 differs from the bundle",
        )
        require(
            metadata.get("lens_outputs_used_for_selection") is False,
            f"prompt {identifier} was not frozen independently of lens outputs",
        )
        require(
            metadata.get("middle_band_layers") == list(FIXED_MIDDLE_LAYERS),
            f"prompt {identifier} fixed middle layer contract changed",
        )
        checkpoint = mapping(
            metadata.get("checkpoint"), f"prompt {identifier}.checkpoint"
        )
        require(
            checkpoint == expected_checkpoint,
            (
                f"prompt {identifier} checkpoint metadata does not match "
                f"the expected {expected_checkpoint['id']} contract"
            ),
        )
        trajectory_binding = None
        if checkpoint["id"] in {"C0M", "C1"}:
            audit_field = (
                "capture_match" if checkpoint["id"] == "C0M" else "observation_audit"
            )
            observation_audit = mapping(
                metadata.get(audit_field),
                f"prompt {identifier}.{audit_field}",
            )
            trajectory_binding = {
                "instance_id": None,
                "capture_manifest_sha256": valid_sha256(
                    observation_audit.get("capture_manifest_sha256"),
                    f"prompt {identifier} capture manifest",
                ),
                "first_request_sha256": valid_sha256(
                    observation_audit.get("first_request_sha256"),
                    f"prompt {identifier} first request",
                ),
                "second_request_sha256": valid_sha256(
                    observation_audit.get("second_request_sha256"),
                    f"prompt {identifier} second request",
                ),
            }
        task = mapping(metadata.get("task"), f"prompt {identifier}.task")
        instance_id = nonempty_string(
            task.get("instance_id"), f"prompt {identifier}.task.instance_id"
        )
        require(instance_id not in task_ids, f"duplicate SWE task: {instance_id}")
        task_ids.add(instance_id)
        if trajectory_binding is not None:
            trajectory_binding["instance_id"] = instance_id
        repo = nonempty_string(task.get("repo"), f"prompt {identifier}.task.repo")
        nonempty_string(
            task.get("base_commit"), f"prompt {identifier}.task.base_commit"
        )
        for hash_field in (
            "problem_statement_sha256",
            "patch_sha256",
            "test_patch_sha256",
        ):
            valid_sha256(task.get(hash_field), f"prompt {identifier}.task.{hash_field}")

        concepts_value = sequence(
            metadata.get("concepts"), f"prompt {identifier}.concepts"
        )
        require(bool(concepts_value), f"prompt {identifier} has no concepts")
        concepts: list[dict[str, Any]] = []
        concept_ids: set[str] = set()
        prompt_token_text: dict[int, str] = {}
        expected_score_ids: set[int] = set()
        for concept_index, raw_concept in enumerate(concepts_value):
            concept = mapping(
                raw_concept, f"prompt {identifier}.concepts[{concept_index}]"
            )
            concept_id = nonempty_string(
                concept.get("id"),
                f"prompt {identifier}.concepts[{concept_index}].id",
            )
            require(
                concept_id not in concept_ids,
                f"prompt {identifier} has duplicate concept ID {concept_id}",
            )
            concept_ids.add(concept_id)
            family = concept.get("family")
            require(
                family in CONCEPT_FAMILIES,
                f"prompt {identifier} concept {concept_id} family is invalid",
            )
            target = nonempty_string(
                concept.get("target"),
                f"prompt {identifier} concept {concept_id}.target",
            )
            path = nonempty_string(
                concept.get("path"), f"prompt {identifier} concept {concept_id}.path"
            )
            require(
                concept.get("evidence") is not None,
                f"prompt {identifier} concept {concept_id}.evidence is missing",
            )
            require(
                concept.get("visibility") == "oracle_hidden",
                f"prompt {identifier} concept {concept_id} visibility mismatch",
            )
            forms = _validate_forms(
                concept.get("forms"),
                label=f"prompt {identifier} concept {concept_id}.forms",
                target=target,
                token_text=prompt_token_text,
            )
            expected_score_ids.update(form["token_id"] for form in forms)

            foils: list[dict[str, Any]] = []
            foil_keys: set[tuple[str, str]] = set()
            for foil_index, raw_foil in enumerate(
                sequence(
                    concept.get("foils", []),
                    f"prompt {identifier} concept {concept_id}.foils",
                )
            ):
                foil = mapping(
                    raw_foil,
                    f"prompt {identifier} concept {concept_id}.foils[{foil_index}]",
                )
                foil_task = nonempty_string(
                    foil.get("task_instance_id"),
                    f"prompt {identifier} concept {concept_id} foil task",
                )
                foil_concept = nonempty_string(
                    foil.get("concept_id"),
                    f"prompt {identifier} concept {concept_id} foil concept",
                )
                foil_key = (foil_task, foil_concept)
                require(
                    foil_task != instance_id and foil_key not in foil_keys,
                    f"prompt {identifier} concept {concept_id} foil identity is invalid",
                )
                foil_keys.add(foil_key)
                require(
                    foil.get("family") == family,
                    f"prompt {identifier} concept {concept_id} foil family mismatch",
                )
                foil_target = nonempty_string(
                    foil.get("target"),
                    f"prompt {identifier} concept {concept_id} foil target",
                )
                foil_forms = _validate_forms(
                    foil.get("forms"),
                    label=(
                        f"prompt {identifier} concept {concept_id} "
                        f"foil {foil_task}/{foil_concept}.forms"
                    ),
                    target=foil_target,
                    token_text=prompt_token_text,
                )
                expected_score_ids.update(form["token_id"] for form in foil_forms)
                foils.append(
                    {
                        "task_instance_id": foil_task,
                        "concept_id": foil_concept,
                        "family": family,
                        "target": foil_target,
                        "forms": foil_forms,
                    }
                )
            concepts.append(
                {
                    "id": concept_id,
                    "family": family,
                    "target": target,
                    "path": path,
                    "evidence": concept["evidence"],
                    "visibility": "oracle_hidden",
                    "forms": forms,
                    "foils": foils,
                }
            )

        require(
            set(score_token_ids) == expected_score_ids,
            f"prompt {identifier}.score_token_ids is not the target/foil form union",
        )
        for token_id, token in prompt_token_text.items():
            previous = global_token_text.setdefault(token_id, token)
            require(previous == token, f"bundle token {token_id} has conflicting text")
        normalized.append(
            {
                "id": identifier,
                "text": text,
                "token_ids": list(token_ids),
                "score_token_ids": list(score_token_ids),
                "token_text": prompt_token_text,
                "metadata": metadata,
                "instance_id": instance_id,
                "repo": repo,
                "task": dict(task),
                "concepts": concepts,
                "trajectory_binding": trajectory_binding,
            }
        )

    # Foils that name a selected task must bind exactly to its selected concept.
    target_index = {
        (prompt["instance_id"], concept["id"]): concept
        for prompt in normalized
        for concept in prompt["concepts"]
    }
    for prompt in normalized:
        for concept in prompt["concepts"]:
            for foil in concept["foils"]:
                counterpart = target_index.get(
                    (foil["task_instance_id"], foil["concept_id"])
                )
                if counterpart is None:
                    continue
                require(
                    foil["family"] == counterpart["family"]
                    and foil["target"] == counterpart["target"]
                    and foil["forms"] == counterpart["forms"],
                    (
                        f"foil {foil['task_instance_id']}/{foil['concept_id']} "
                        "does not match its selected target concept"
                    ),
                )

    require(protocol_hash is not None, "prompt bundle protocol hash is missing")
    union_ids = list(
        dict.fromkeys(
            token_id
            for prompt in normalized
            for token_id in prompt["score_token_ids"]
        )
    )
    return {
        "prompts": normalized,
        "protocol_sha256": protocol_hash,
        "token_text": global_token_text,
        "union_token_ids": union_ids,
        "bundle_sha256": materialized_json_sha256(prompts_value),
        "checkpoint": dict(expected_checkpoint),
        "trajectory_bindings": [
            prompt["trajectory_binding"]
            for prompt in normalized
            if prompt["trajectory_binding"] is not None
        ],
    }


def validate_lens(report: Mapping[str, Any], label: str) -> None:
    lens = mapping(report.get("lens"), f"{label}.lens")
    require(
        lens.get("d_model") == 5120
        and lens.get("source_layers") == list(ALL_SOURCE_LAYERS)
        and lens.get("tensor_shape") == [5120, 5120],
        f"{label} lens geometry mismatch",
    )
    if label == "public":
        require(
            lens.get("repo_id") == PUBLIC_LENS_REPO
            and lens.get("revision") == PUBLIC_LENS_REVISION
            and lens.get("sha256") == PUBLIC_LENS_SHA256
            and lens.get("n_prompts", 1000) == 1000,
            "public lens identity mismatch",
        )
    elif label == "native":
        require(
            lens.get("kind") == "native_nvfp4_ste_fit"
            and lens.get("sha256") == NATIVE_LENS_SHA256
            and lens.get("state_sha256") == NATIVE_STATE_SHA256
            and lens.get("provenance_sha256") == NATIVE_PROVENANCE_SHA256
            and lens.get("fit_model") == MODEL_REPO
            and lens.get("fit_model_revision") == MODEL_REVISION
            and lens.get("n_prompts", 10) == 10,
            "native lens identity or fit provenance mismatch",
        )
    else:
        raise ValueError(f"unknown report label: {label}")


def scored_ranks(
    value: Any,
    *,
    label: str,
    expected_ids: Sequence[int],
    token_text: Mapping[int, str],
    generated_token_id: int,
) -> dict[int, int]:
    readout = mapping(value, label)
    require(
        readout.get("target_token_id") == generated_token_id,
        f"{label} target is not the accepted generated token",
    )
    rank = readout.get("target_rank")
    require(
        isinstance(rank, int)
        and not isinstance(rank, bool)
        and 1 <= rank <= LOGIT_VOCABULARY_SIZE,
        f"{label} accepted-token rank is invalid",
    )
    finite(readout.get("target_score"), f"{label}.target_score")
    if "target_logprob" in readout:
        finite(readout.get("target_logprob"), f"{label}.target_logprob")

    records = sequence(readout.get("scored_tokens"), f"{label}.scored_tokens")
    result: dict[int, int] = {}
    for index, raw_record in enumerate(records):
        record = mapping(raw_record, f"{label}.scored_tokens[{index}]")
        token_id = record.get("token_id")
        token_rank = record.get("rank")
        require(
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and token_id not in result,
            f"{label} scored token ID is invalid or duplicated",
        )
        require(
            isinstance(token_rank, int)
            and not isinstance(token_rank, bool)
            and 1 <= token_rank <= LOGIT_VOCABULARY_SIZE,
            f"{label} exact rank is invalid",
        )
        require(
            record.get("token") == token_text.get(token_id),
            f"{label} scored token text mismatch",
        )
        finite(record.get("score"), f"{label}.token-{token_id}.score")
        finite(record.get("logprob"), f"{label}.token-{token_id}.logprob")
        result[token_id] = token_rank
    require(
        list(result) == list(expected_ids),
        f"{label} scored vocabulary rank coverage/order mismatch",
    )
    return result


def validate_report(
    report: Mapping[str, Any], *, label: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    require(report.get("schema_version") == 3, f"{label} report schema mismatch")
    require(
        report.get("score_encoding") == "unrounded-float32",
        f"{label} score encoding mismatch",
    )
    validate_lens(report, label)
    model = mapping(report.get("model"), f"{label}.model")
    require(
        model.get("repo_id") == MODEL_REPO
        and model.get("revision") == MODEL_REVISION
        and model.get("config_sha256") == MODEL_CONFIG_SHA256
        and model.get("index_sha256") == MODEL_INDEX_SHA256,
        f"{label} pinned model mismatch",
    )
    runtime = mapping(report.get("runtime"), f"{label}.runtime")
    require(
        runtime.get("mtp_enabled") is False
        and runtime.get("enforce_eager") is True
        and runtime.get("language_model_only") is True,
        f"{label} NVFP4 replay runtime mismatch",
    )
    assertions = mapping(report.get("assertions"), f"{label}.assertions")
    for assertion in (
        "lens_hash_matches",
        "lens_metadata_matches",
        "model_architecture_matches",
    ):
        require(
            assertions.get(assertion) is True,
            f"{label} required assertion failed: {assertion}",
        )

    vocabulary = mapping(report.get("scored_vocabulary"), f"{label}.scored_vocabulary")
    union_ids = contract["union_token_ids"]
    union_tokens = [contract["token_text"][token_id] for token_id in union_ids]
    require(
        vocabulary.get("scope") == "global_plus_per_experiment"
        and vocabulary.get("token_ids") == []
        and vocabulary.get("tokens") == []
        and vocabulary.get("union_token_ids") == union_ids
        and vocabulary.get("union_tokens") == union_tokens,
        f"{label} report-level scored vocabulary mismatch",
    )

    experiments = sequence(report.get("experiments"), f"{label}.experiments")
    require(
        len(experiments) == len(contract["prompts"]),
        f"{label} experiment count mismatch",
    )
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    eligibility = {
        "greedy_top1_match": 0,
        "final_top5_match": 0,
        "final_norm_within_tolerance": 0,
        "final_logits_within_tolerance": 0,
    }
    all_top1 = True
    all_reconstruction = True
    for index, (raw_experiment, prompt) in enumerate(
        zip(experiments, contract["prompts"], strict=True)
    ):
        experiment = mapping(raw_experiment, f"{label}.experiments[{index}]")
        identifier = prompt["id"]
        require(experiment.get("id") == identifier, f"{label} experiment ID mismatch")
        require(
            experiment.get("prompt") == prompt["text"],
            f"{label} {identifier} exact prompt text mismatch",
        )
        require(
            experiment.get("prompt_token_ids") == prompt["token_ids"],
            f"{label} {identifier} exact prompt token IDs mismatch",
        )
        prompt_tokens = sequence(
            experiment.get("prompt_tokens"), f"{label} {identifier}.prompt_tokens"
        )
        require(
            len(prompt_tokens) == len(prompt["token_ids"])
            and all(isinstance(token, str) for token in prompt_tokens),
            f"{label} {identifier} prompt token decoding is invalid",
        )
        require(
            experiment.get("metadata") == prompt["metadata"],
            f"{label} {identifier} exact metadata mismatch",
        )
        require(
            "target_token_id_override" not in experiment,
            f"{label} {identifier} unexpectedly overrides the accepted token",
        )
        generated_token_id = experiment.get("generated_token_id")
        require(
            isinstance(generated_token_id, int)
            and not isinstance(generated_token_id, bool)
            and 0 <= generated_token_id < LOGIT_VOCABULARY_SIZE,
            f"{label} {identifier} generated token ID is invalid",
        )
        final_position = len(prompt["token_ids"]) - 1
        require(
            experiment.get("positions_requested") == [-1]
            and experiment.get("positions_resolved") == [final_position]
            and experiment.get("capture_positions_resolved") == [final_position]
            and experiment.get("final_validation_position") == final_position,
            f"{label} {identifier} checkpoint readout position mismatch",
        )
        scored_vocabulary = mapping(
            experiment.get("scored_vocabulary"),
            f"{label} {identifier}.scored_vocabulary",
        )
        expected_ids = prompt["score_token_ids"]
        expected_tokens = [prompt["token_text"][token_id] for token_id in expected_ids]
        require(
            scored_vocabulary.get("token_ids") == expected_ids
            and scored_vocabulary.get("tokens") == expected_tokens,
            f"{label} {identifier} per-prompt scored vocabulary mismatch",
        )

        layers = sequence(experiment.get("layers"), f"{label} {identifier}.layers")
        layer_ids = [mapping(layer, "layer").get("layer") for layer in layers]
        require(
            layer_ids == list(FIXED_MIDDLE_LAYERS),
            f"{label} {identifier} must contain exactly fixed layers 16 through 47",
        )
        rank_maps: dict[str, dict[int, dict[int, int]]] = {
            "jacobian": {},
            "logit": {},
        }
        logit_readouts: list[Mapping[str, Any]] = []
        for layer_id, raw_layer in zip(layer_ids, layers, strict=True):
            layer = mapping(raw_layer, f"{label} {identifier}.layer-{layer_id}")
            positions = sequence(
                layer.get("positions"),
                f"{label} {identifier}.layer-{layer_id}.positions",
            )
            require(
                len(positions) == 1,
                f"{label} {identifier} layer {layer_id} position count mismatch",
            )
            position = mapping(positions[0], "position")
            require(
                position.get("capture_index") == 0
                and position.get("token_position") == final_position,
                f"{label} {identifier} layer {layer_id} position mismatch",
            )
            jacobian_readout = position.get("jacobian_lens")
            logit_readout = position.get("logit_lens")
            rank_maps["jacobian"][layer_id] = scored_ranks(
                jacobian_readout,
                label=f"{label} {identifier}.layer-{layer_id}.jacobian",
                expected_ids=expected_ids,
                token_text=prompt["token_text"],
                generated_token_id=generated_token_id,
            )
            rank_maps["logit"][layer_id] = scored_ranks(
                logit_readout,
                label=f"{label} {identifier}.layer-{layer_id}.logit",
                expected_ids=expected_ids,
                token_text=prompt["token_text"],
                generated_token_id=generated_token_id,
            )
            logit_readouts.append(mapping(logit_readout, "logit readout"))

        top1 = experiment.get("final_layer_top1_matches_greedy")
        require(isinstance(top1, bool), f"{label} {identifier} top1 flag is invalid")
        final_norm = mapping(
            experiment.get("final_norm_reconstruction"),
            f"{label} {identifier}.final_norm_reconstruction",
        )
        final_logits = mapping(
            experiment.get("final_logits_reconstruction"),
            f"{label} {identifier}.final_logits_reconstruction",
        )
        norm_within = final_norm.get("within_tolerance")
        logits_within = final_logits.get("within_tolerance")
        top5 = final_logits.get("top_k_prefix_token_ids_match")
        require(
            all(isinstance(value, bool) for value in (norm_within, logits_within, top5)),
            f"{label} {identifier} reconstruction flags are invalid",
        )
        eligibility["greedy_top1_match"] += int(top1)
        eligibility["final_top5_match"] += int(top5)
        eligibility["final_norm_within_tolerance"] += int(norm_within)
        eligibility["final_logits_within_tolerance"] += int(logits_within)
        all_top1 = all_top1 and top1
        all_reconstruction = all_reconstruction and norm_within and logits_within
        residual = mapping(
            experiment.get("residual_capture_manifest"),
            f"{label} {identifier}.residual_capture_manifest",
        )
        valid_sha256(
            residual.get("sha256"), f"{label} {identifier} residual manifest hash"
        )

        rows.append(
            {
                "id": identifier,
                "instance_id": prompt["instance_id"],
                "repo": prompt["repo"],
                "concepts": prompt["concepts"],
                "generated_token_id": generated_token_id,
                "rank_maps": rank_maps,
            }
        )
        pair_rows.append(
            {
                "id": identifier,
                "prompt": experiment.get("prompt"),
                "prompt_token_ids": experiment.get("prompt_token_ids"),
                "prompt_tokens": prompt_tokens,
                "metadata": experiment.get("metadata"),
                "scored_vocabulary": dict(scored_vocabulary),
                "generated_token_id": generated_token_id,
                "residual_capture_manifest": dict(residual),
                "logit_readouts_sha256": materialized_json_sha256(logit_readouts),
                "top1": top1,
                "top5": top5,
                "norm_within": norm_within,
                "logits_within": logits_within,
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
    require(
        report.get("status")
        == ("passed" if all_top1 and all_reconstruction else "failed"),
        f"{label} report status mismatch",
    )
    return {
        "rows": rows,
        "pair_rows": pair_rows,
        "numerical_eligibility": {
            "experiment_count": len(experiments),
            "counts": eligibility,
            "report_status": report.get("status"),
        },
    }


def validate_report_pair(
    public: Mapping[str, Any], native: Mapping[str, Any]
) -> dict[str, Any]:
    public_rows = sequence(public.get("pair_rows"), "public.pair_rows")
    native_rows = sequence(native.get("pair_rows"), "native.pair_rows")
    require(len(public_rows) == len(native_rows), "paired report task count mismatch")
    residuals: list[Mapping[str, Any]] = []
    for left, right in zip(public_rows, native_rows, strict=True):
        for field in (
            "id",
            "prompt",
            "prompt_token_ids",
            "prompt_tokens",
            "metadata",
            "scored_vocabulary",
            "generated_token_id",
            "residual_capture_manifest",
            "logit_readouts_sha256",
            "top1",
            "top5",
            "norm_within",
            "logits_within",
        ):
            require(
                left[field] == right[field],
                f"paired reports differ in {field}: {left['id']}",
            )
        residuals.append(left["residual_capture_manifest"])
    return {
        "task_count": len(public_rows),
        "exact_prompt_token_metadata_vocabulary_pairing": True,
        "accepted_generated_tokens_equal": True,
        "residual_and_logit_readouts_equal": True,
        "residual_manifest_list_sha256": materialized_json_sha256(residuals),
    }


def _best_form_rank(
    rank_maps: Mapping[int, Mapping[int, int]],
    forms: Sequence[Mapping[str, Any]],
    *,
    generated_token_id: int,
) -> dict[str, Any]:
    included_forms = [form for form in forms if form["token_id"] != generated_token_id]
    excluded_forms = [form for form in forms if form["token_id"] == generated_token_id]
    if not included_forms:
        return {
            "scorable": False,
            "reason": "all eligible forms equal the accepted generated token",
            "excluded_accepted_forms": [dict(form) for form in excluded_forms],
            "minimum_rank": None,
            "utility_u": None,
        }
    candidates = [
        (rank_maps[layer][form["token_id"]], layer, form["token_id"], form["text"])
        for layer in FIXED_MIDDLE_LAYERS
        for form in included_forms
    ]
    rank, layer, token_id, token = min(candidates)
    return {
        "scorable": True,
        "minimum_rank": rank,
        "utility_u": normalized_utility(rank),
        "best_layer": layer,
        "best_token_id": token_id,
        "best_token": token,
        "eligible_forms_after_accepted_token_exclusion": [
            dict(form) for form in included_forms
        ],
        "excluded_accepted_forms": [dict(form) for form in excluded_forms],
        "pass_at_k": {str(k): int(rank <= k) for k in PASS_K},
    }


def score_method(
    rows: Sequence[Mapping[str, Any]], *, method: str
) -> dict[str, Any]:
    task_results: list[dict[str, Any]] = []
    accepted_exclusions: list[dict[str, Any]] = []
    all_target_ranks: list[int] = []
    for row in rows:
        family_concepts: dict[str, list[dict[str, Any]]] = {}
        rank_maps = row["rank_maps"][method]
        for concept in row["concepts"]:
            target = _best_form_rank(
                rank_maps,
                concept["forms"],
                generated_token_id=row["generated_token_id"],
            )
            for form in target["excluded_accepted_forms"]:
                accepted_exclusions.append(
                    {
                        "instance_id": row["instance_id"],
                        "concept_id": concept["id"],
                        "role": "target",
                        **form,
                    }
                )
            if target["scorable"]:
                all_target_ranks.append(target["minimum_rank"])
            foils: list[dict[str, Any]] = []
            for foil in concept["foils"]:
                foil_score = _best_form_rank(
                    rank_maps,
                    foil["forms"],
                    generated_token_id=row["generated_token_id"],
                )
                for form in foil_score["excluded_accepted_forms"]:
                    accepted_exclusions.append(
                        {
                            "instance_id": row["instance_id"],
                            "concept_id": concept["id"],
                            "role": "foil",
                            "foil_task_instance_id": foil["task_instance_id"],
                            "foil_concept_id": foil["concept_id"],
                            **form,
                        }
                    )
                foils.append({**foil, "score": foil_score})
            scorable_foils = [foil for foil in foils if foil["score"]["scorable"]]
            contrast = None
            if target["scorable"] and scorable_foils:
                foil_u = statistics.fmean(
                    foil["score"]["utility_u"] for foil in scorable_foils
                )
                foil_pass = {
                    str(k): statistics.fmean(
                        foil["score"]["pass_at_k"][str(k)] for foil in scorable_foils
                    )
                    for k in PASS_K
                }
                contrast = {
                    "foil_count": len(scorable_foils),
                    "target_utility_u": target["utility_u"],
                    "foil_utility_u": foil_u,
                    "target_minus_foil_u": target["utility_u"] - foil_u,
                    "target_pass_at_k": dict(target["pass_at_k"]),
                    "foil_pass_at_k": foil_pass,
                    "target_minus_foil_pass_at_k": {
                        str(k): target["pass_at_k"][str(k)] - foil_pass[str(k)]
                        for k in PASS_K
                    },
                }
            family_concepts.setdefault(concept["family"], []).append(
                {
                    "id": concept["id"],
                    "target": concept["target"],
                    "path": concept["path"],
                    "target_score": target,
                    "foils": foils,
                    "target_minus_foil": contrast,
                }
            )

        families: list[dict[str, Any]] = []
        for family, concepts in family_concepts.items():
            scorable = [concept for concept in concepts if concept["target_score"]["scorable"]]
            contrasts = [
                concept["target_minus_foil"]
                for concept in concepts
                if concept["target_minus_foil"] is not None
            ]
            family_row: dict[str, Any] = {
                "family": family,
                "concept_count": len(concepts),
                "scorable_target_concept_count": len(scorable),
                "concepts": concepts,
            }
            if scorable:
                family_row["target_utility_u"] = statistics.fmean(
                    concept["target_score"]["utility_u"] for concept in scorable
                )
                family_row["target_pass_at_k"] = {
                    str(k): statistics.fmean(
                        concept["target_score"]["pass_at_k"][str(k)]
                        for concept in scorable
                    )
                    for k in PASS_K
                }
            if contrasts:
                family_row["contrast_concept_count"] = len(contrasts)
                family_row["contrast_target_utility_u"] = statistics.fmean(
                    contrast["target_utility_u"] for contrast in contrasts
                )
                family_row["contrast_foil_utility_u"] = statistics.fmean(
                    contrast["foil_utility_u"] for contrast in contrasts
                )
                family_row["target_minus_foil_u"] = statistics.fmean(
                    contrast["target_minus_foil_u"] for contrast in contrasts
                )
                family_row["contrast_target_pass_at_k"] = {
                    str(k): statistics.fmean(
                        contrast["target_pass_at_k"][str(k)] for contrast in contrasts
                    )
                    for k in PASS_K
                }
                family_row["contrast_foil_pass_at_k"] = {
                    str(k): statistics.fmean(
                        contrast["foil_pass_at_k"][str(k)] for contrast in contrasts
                    )
                    for k in PASS_K
                }
            families.append(family_row)

        target_families = [family for family in families if "target_utility_u" in family]
        require(
            bool(target_families),
            f"task {row['instance_id']} has no target forms after accepted-token exclusion",
        )
        task_result: dict[str, Any] = {
            "id": row["id"],
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "generated_token_id": row["generated_token_id"],
            "family_count": len(target_families),
            "target_utility_u": statistics.fmean(
                family["target_utility_u"] for family in target_families
            ),
            "target_pass_at_k": {
                str(k): statistics.fmean(
                    family["target_pass_at_k"][str(k)] for family in target_families
                )
                for k in PASS_K
            },
            "families": families,
        }
        contrast_families = [family for family in families if "target_minus_foil_u" in family]
        if contrast_families:
            task_result["foil_contrast"] = {
                "family_count": len(contrast_families),
                "target_utility_u": statistics.fmean(
                    family["contrast_target_utility_u"] for family in contrast_families
                ),
                "foil_utility_u": statistics.fmean(
                    family["contrast_foil_utility_u"] for family in contrast_families
                ),
                "target_minus_foil_u": statistics.fmean(
                    family["target_minus_foil_u"] for family in contrast_families
                ),
                "target_pass_at_k": {
                    str(k): statistics.fmean(
                        family["contrast_target_pass_at_k"][str(k)]
                        for family in contrast_families
                    )
                    for k in PASS_K
                },
                "foil_pass_at_k": {
                    str(k): statistics.fmean(
                        family["contrast_foil_pass_at_k"][str(k)]
                        for family in contrast_families
                    )
                    for k in PASS_K
                },
            }
        task_results.append(task_result)

    foil_tasks = [task for task in task_results if "foil_contrast" in task]
    result: dict[str, Any] = {
        "task_count": len(task_results),
        "target_utility_u": statistics.fmean(
            task["target_utility_u"] for task in task_results
        ),
        "target_pass_at_k": {
            str(k): statistics.fmean(
                task["target_pass_at_k"][str(k)] for task in task_results
            )
            for k in PASS_K
        },
        "rank_summary": {
            "aggregation": "unweighted over scorable retained target concepts",
            "target_concept_count": len(all_target_ranks),
            "minimum": min(all_target_ranks),
            "median": statistics.median(all_target_ranks),
            "geometric_mean": math.exp(
                statistics.fmean(math.log(rank) for rank in all_target_ranks)
            ),
            "maximum": max(all_target_ranks),
        },
        "accepted_generated_token_exclusions": accepted_exclusions,
        "tasks": task_results,
    }
    if foil_tasks:
        result["target_minus_foil"] = {
            "task_count": len(foil_tasks),
            "target_utility_u": statistics.fmean(
                task["foil_contrast"]["target_utility_u"] for task in foil_tasks
            ),
            "foil_utility_u": statistics.fmean(
                task["foil_contrast"]["foil_utility_u"] for task in foil_tasks
            ),
            "difference_u": statistics.fmean(
                task["foil_contrast"]["target_minus_foil_u"] for task in foil_tasks
            ),
        }
    return result


def paired_task_bootstrap(
    values: Sequence[Mapping[str, Any]],
    *,
    label: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    require(samples > 0, "bootstrap sample count must be positive")
    require(bool(values), "paired task bootstrap requires at least one task")
    differences = [finite(value.get("difference"), "task difference") for value in values]
    generator = random.Random(derived_seed(label, seed))
    draws = [
        statistics.fmean(generator.choice(differences) for _ in differences)
        for _ in range(samples)
    ]
    return {
        "method": "deterministic paired task-level percentile bootstrap",
        "sampling_unit": "independent SWE-Verified task",
        "seed": derived_seed(label, seed),
        "samples": samples,
        "task_count": len(differences),
        "estimate": statistics.fmean(differences),
        "confidence_interval": {
            "confidence_level": 0.95,
            "lower": quantile(draws, 0.025),
            "upper": quantile(draws, 0.975),
        },
        "positive_task_count": sum(value > 0.0 for value in differences),
        "negative_task_count": sum(value < 0.0 for value in differences),
        "tie_task_count": sum(value == 0.0 for value in differences),
    }


def leave_one_repo_out(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    repositories = sorted({nonempty_string(value.get("repo"), "task repo") for value in values})
    result: list[dict[str, Any]] = []
    for repository in repositories:
        retained = [value for value in values if value["repo"] != repository]
        if not retained:
            result.append(
                {
                    "omitted_repo": repository,
                    "omitted_task_count": len(values),
                    "remaining_task_count": 0,
                    "estimate": None,
                    "reason": "unavailable: contrastable tasks span only one repository",
                }
            )
            continue
        result.append(
            {
                "omitted_repo": repository,
                "omitted_task_count": len(values) - len(retained),
                "remaining_task_count": len(retained),
                "estimate": statistics.fmean(value["difference"] for value in retained),
            }
        )
    return result


def _task_map(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    tasks = sequence(summary.get("tasks"), "method.tasks")
    result = {task["instance_id"]: task for task in tasks}
    require(len(result) == len(tasks), "method summary has duplicate tasks")
    return result


def compare_methods(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    label: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    left_tasks = _task_map(left)
    right_tasks = _task_map(right)
    require(left_tasks.keys() == right_tasks.keys(), f"{label} task grid mismatch")
    utility_values: list[dict[str, Any]] = []
    pass_values: dict[str, list[dict[str, Any]]] = {str(k): [] for k in PASS_K}
    for instance_id, left_task in left_tasks.items():
        right_task = right_tasks[instance_id]
        require(
            left_task["repo"] == right_task["repo"],
            f"{label} repository pairing mismatch for {instance_id}",
        )
        base = {"instance_id": instance_id, "repo": left_task["repo"]}
        utility_values.append(
            {
                **base,
                "difference": left_task["target_utility_u"]
                - right_task["target_utility_u"],
            }
        )
        for k in PASS_K:
            pass_values[str(k)].append(
                {
                    **base,
                    "difference": left_task["target_pass_at_k"][str(k)]
                    - right_task["target_pass_at_k"][str(k)],
                }
            )
    return {
        "left_minus_right": label,
        "task_count": len(utility_values),
        "task_utility_u": paired_task_bootstrap(
            utility_values,
            label=f"{label}:utility",
            seed=seed,
            samples=samples,
        ),
        "pass_at_k": {
            str(k): paired_task_bootstrap(
                pass_values[str(k)],
                label=f"{label}:pass@{k}",
                seed=seed,
                samples=samples,
            )
            for k in PASS_K
        },
        "leave_one_repo_out": {
            "task_utility_u": leave_one_repo_out(utility_values),
            "pass_at_k": {
                str(k): leave_one_repo_out(pass_values[str(k)]) for k in PASS_K
            },
        },
    }


def target_minus_foil_comparison(
    method: Mapping[str, Any],
    *,
    label: str,
    seed: int,
    samples: int,
) -> dict[str, Any] | None:
    tasks = [task for task in sequence(method.get("tasks"), "method.tasks") if "foil_contrast" in task]
    if not tasks:
        return None
    utility_values = [
        {
            "instance_id": task["instance_id"],
            "repo": task["repo"],
            "difference": task["foil_contrast"]["target_minus_foil_u"],
        }
        for task in tasks
    ]
    pass_values = {
        str(k): [
            {
                "instance_id": task["instance_id"],
                "repo": task["repo"],
                "difference": task["foil_contrast"]["target_pass_at_k"][str(k)]
                - task["foil_contrast"]["foil_pass_at_k"][str(k)],
            }
            for task in tasks
        ]
        for k in PASS_K
    }
    return {
        "contrast": "target_minus_matched_foil",
        "task_count": len(tasks),
        "task_utility_u": paired_task_bootstrap(
            utility_values,
            label=f"{label}:target-minus-foil:utility",
            seed=seed,
            samples=samples,
        ),
        "pass_at_k": {
            str(k): paired_task_bootstrap(
                pass_values[str(k)],
                label=f"{label}:target-minus-foil:pass@{k}",
                seed=seed,
                samples=samples,
            )
            for k in PASS_K
        },
        "leave_one_repo_out": {
            "task_utility_u": leave_one_repo_out(utility_values),
            "pass_at_k": {
                str(k): leave_one_repo_out(pass_values[str(k)]) for k in PASS_K
            },
        },
    }


def analyze(
    prompts_value: Sequence[Mapping[str, Any]],
    public_report: Mapping[str, Any],
    native_report: Mapping[str, Any],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    expected_checkpoint: Mapping[str, str] = CHECKPOINT_CONTRACTS["C0"],
) -> dict[str, Any]:
    require(bootstrap_samples > 0, "bootstrap sample count must be positive")
    contract = validate_prompt_bundle(
        prompts_value, expected_checkpoint=expected_checkpoint
    )
    public = validate_report(public_report, label="public", contract=contract)
    native = validate_report(native_report, label="native", contract=contract)
    pairing = validate_report_pair(public, native)

    public_jacobian = score_method(public["rows"], method="jacobian")
    logit = score_method(public["rows"], method="logit")
    native_jacobian = score_method(native["rows"], method="jacobian")
    native_logit = score_method(native["rows"], method="logit")
    require(
        logit == native_logit,
        "paired reports produced different ordinary logit-lens summaries",
    )
    checkpoint_id = contract["checkpoint"]["id"]
    if checkpoint_id == "C0":
        analysis_kind = "exploratory_swe_verified_multitask_initial_probe_analysis"
        analysis_label = (
            "EXPLORATORY MULTI-TASK PILOT: associative task-start concept "
            "readout, not chain-of-thought recovery or causal evidence"
        )
        causal_limitation = (
            "Association at task start does not establish that the model uses a "
            "concept causally."
        )
    elif checkpoint_id == "C1":
        analysis_kind = (
            "exploratory_swe_verified_multitask_"
            "post_repository_observation_probe_analysis"
        )
        analysis_label = (
            "EXPLORATORY MULTI-TASK PILOT: associative post-repository-observation "
            "concept readout, not chain-of-thought recovery or causal evidence"
        )
        causal_limitation = (
            "Association after the first successful repository observation does not "
            "establish that the model uses a concept causally."
        )
    else:
        require(checkpoint_id == "C0M", "unsupported checkpoint analysis contract")
        analysis_kind = (
            "exploratory_swe_verified_multitask_"
            "capture_matched_initial_probe_analysis"
        )
        analysis_label = (
            "EXPLORATORY MULTI-TASK PILOT: associative capture-matched task-start "
            "concept readout, not chain-of-thought recovery or causal evidence"
        )
        causal_limitation = (
            "Association at a capture-matched task start does not establish that "
            "the model uses a concept causally."
        )

    comparisons = {
        "public_minus_logit": compare_methods(
            public_jacobian,
            logit,
            label="public_jacobian_minus_logit",
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
        "native_minus_logit": compare_methods(
            native_jacobian,
            logit,
            label="native_jacobian_minus_logit",
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
        "native_minus_public": compare_methods(
            native_jacobian,
            public_jacobian,
            label="native_jacobian_minus_public_jacobian",
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
    }
    target_foil = {
        "public_jacobian": target_minus_foil_comparison(
            public_jacobian,
            label="public_jacobian",
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
        "native_jacobian": target_minus_foil_comparison(
            native_jacobian,
            label="native_jacobian",
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
        "logit_lens": target_minus_foil_comparison(
            logit,
            label="logit_lens",
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
    }
    comparisons["target_minus_foil"] = (
        target_foil if any(value is not None for value in target_foil.values()) else None
    )

    return {
        "schema_version": 1,
        "kind": analysis_kind,
        "label": analysis_label,
        "model": {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "logit_vocabulary_size": LOGIT_VOCABULARY_SIZE,
        },
        "source_bindings": {
            "prompt_bundle_sha256": contract["bundle_sha256"],
            "protocol_sha256": contract["protocol_sha256"],
            "trajectory_bindings": contract["trajectory_bindings"],
        },
        "evaluation": {
            "checkpoint": {
                "C0": "C0: before the first assistant token",
                "C0M": (
                    "C0M: same captured trajectory before the first assistant token"
                ),
                "C1": "C1: after the first successful repository observation",
            }[contract["checkpoint"]["id"]],
            "checkpoint_metadata": contract["checkpoint"],
            "fixed_middle_layers": list(FIXED_MIDDLE_LAYERS),
            "pass_at_k": list(PASS_K),
            "rank_reduction": "minimum over predeclared eligible forms and fixed layers",
            "utility_u": "log(V / minimum_rank) / log(V)",
            "weighting": (
                "utility and pass@k: equal concepts within family, equal families "
                "within task, equal independent tasks overall; rank_summary: "
                "unweighted over scorable retained target concepts"
            ),
            "accepted_generated_token_policy": (
                "forms identical to the paired accepted generated token are "
                "excluded after report pairing and recorded explicitly"
            ),
            "bootstrap": {
                "sampling_unit": "independent SWE-Verified task",
                "seed": bootstrap_seed,
                "samples": bootstrap_samples,
            },
            "claims_gate": None,
            "claims_gate_reason": (
                "the frozen prompt metadata binds a protocol hash but does not "
                "embed preregistered decision thresholds"
            ),
            "limitations": [
                "Minimum over multiple forms and 32 layers is an optimistic readout statistic.",
                causal_limitation,
                "Oracle-hidden patch concepts test localization, not full SWE task completion.",
            ],
        },
        "coverage": {
            "task_count": len(contract["prompts"]),
            "repository_count": len({prompt["repo"] for prompt in contract["prompts"]}),
            "concept_count": sum(
                len(prompt["concepts"]) for prompt in contract["prompts"]
            ),
            "concept_family_counts": {
                family: sum(
                    concept["family"] == family
                    for prompt in contract["prompts"]
                    for concept in prompt["concepts"]
                )
                for family in sorted(CONCEPT_FAMILIES)
            },
            "foil_occurrence_count": sum(
                len(concept["foils"])
                for prompt in contract["prompts"]
                for concept in prompt["concepts"]
            ),
        },
        "numerical_eligibility": {
            "public": public["numerical_eligibility"],
            "native": native["numerical_eligibility"],
        },
        "pairing": pairing,
        "methods": {
            "public_jacobian": public_jacobian,
            "native_jacobian": native_jacobian,
            "logit_lens": logit,
        },
        "comparisons": comparisons,
    }


def read_json(path: Path) -> tuple[Any, dict[str, Any]]:
    raw = path.read_bytes()
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        display_path = f"external/{resolved.name}"
    return json.loads(raw), {
        "path": display_path,
        "size_bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint",
        choices=tuple(CHECKPOINT_CONTRACTS),
        default="C0",
        help="exact frozen checkpoint metadata contract (default: C0)",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prompts, prompts_source = read_json(args.prompts)
    public_report, public_source = read_json(args.public_report)
    native_report, native_source = read_json(args.native_report)
    require(
        isinstance(prompts, list)
        and isinstance(public_report, dict)
        and isinstance(native_report, dict),
        "prompt and report JSON top-level types are invalid",
    )
    result = analyze(
        prompts,
        public_report,
        native_report,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        expected_checkpoint=CHECKPOINT_CONTRACTS[args.expected_checkpoint],
    )
    result["inputs"] = {
        "prompts": prompts_source,
        "public_report": public_source,
        "native_report": native_source,
    }
    atomic_write_json(args.output, result)
    for comparison_name in (
        "public_minus_logit",
        "native_minus_logit",
        "native_minus_public",
    ):
        metric = result["comparisons"][comparison_name]["task_utility_u"]
        interval = metric["confidence_interval"]
        print(
            f"{comparison_name} task-u: {metric['estimate']:+.6f} "
            f"[{interval['lower']:+.6f}, {interval['upper']:+.6f}]"
        )
    print("claims gate: NONE (exploratory pilot; no embedded thresholds)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
