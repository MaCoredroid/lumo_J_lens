#!/usr/bin/env python3
"""Analyze frozen before/after contextual-evidence probes on SWE trajectories.

The analysis is deliberately narrower than a natural-language "thought decoder".
It measures whether fixed, task-specific concepts gain evidence across an observed
Qwen Code transition, relative to same-task foils and targets from other tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_KIND = "swe_contextual_evidence_update_protocol"
MANIFEST_KIND = "swe_contextual_evidence_materialization"
PROMPT_KIND = "swe_contextual_evidence_prompt"
ANALYSIS_KIND = "swe_contextual_evidence_update_analysis"
CARDS_KIND = "swe_contextual_evidence_cards"
REPORT_SCHEMA_VERSION = 3
FIXED_LAYERS = tuple(range(24, 48))
REPORT_LENS_KEYS = {
    "public": "public",
    "nf4": "nf4",
    "native": "native_nvfp4_ste",
}
METHOD_SPECS = {
    "public_jacobian": ("public", "jacobian"),
    "ordinary_logit": ("public", "logit"),
    "nf4_jacobian": ("nf4", "jacobian"),
    "native_jacobian": ("native", "jacobian"),
}
CARD_FIELDS = ("WHY", "WHERE", "EVIDENCE", "NEXT")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def text(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be nonempty text")
    return value


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
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


def digest(value: Any, label: str) -> str:
    result = text(value, label)
    require(
        len(result) == 64
        and all(character in "0123456789abcdef" for character in result),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return result


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    )


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
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


def logmeanexp(values: Sequence[float]) -> float:
    require(bool(values), "logmeanexp input is empty")
    maximum = max(values)
    return maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in values) / len(values)
    )


def percentile(values: Sequence[float], probability: float) -> float:
    require(bool(values), "percentile input is empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _normal_form(value: Any, label: str) -> dict[str, Any]:
    form = mapping(value, label)
    token_id = integer(form.get("token_id"), f"{label}.token_id")
    token_text = text(form.get("text"), f"{label}.text")
    result = dict(form)
    result["token_id"] = token_id
    result["text"] = token_text
    return result


def _normal_concept(value: Any, label: str) -> dict[str, Any]:
    concept = mapping(value, label)
    identifier = text(concept.get("id"), f"{label}.id")
    forms = [
        _normal_form(item, f"{label}.forms[{index}]")
        for index, item in enumerate(sequence(concept.get("forms"), f"{label}.forms"))
    ]
    require(bool(forms), f"{label} must have at least one single-token form")
    token_ids = [item["token_id"] for item in forms]
    require(len(token_ids) == len(set(token_ids)), f"{label} form token IDs overlap")
    result = dict(concept)
    result["id"] = identifier
    result["forms"] = forms
    return result


def _normal_task(value: Any, label: str) -> dict[str, Any]:
    task = mapping(value, label)
    result = dict(task)
    for field in ("id", "instance_id", "repo", "cohort", "stratum"):
        result[field] = text(task.get(field), f"{label}.{field}")
    result["after_global_request_index"] = integer(
        task.get("after_global_request_index"), f"{label}.after_global_request_index", minimum=1
    )
    result["after_task_request_index"] = integer(
        task.get("after_task_request_index"), f"{label}.after_task_request_index", minimum=1
    )
    raw_sha = mapping(task.get("raw_sha256"), f"{label}.raw_sha256")
    require(set(raw_sha) == {"before", "after", "label"}, f"{label}.raw_sha256 fields changed")
    result["raw_sha256"] = {
        state: digest(raw_sha.get(state), f"{label}.raw_sha256.{state}")
        for state in ("before", "after", "label")
    }
    result["target"] = _normal_concept(task.get("target"), f"{label}.target")
    result["foils"] = [
        _normal_concept(item, f"{label}.foils[{index}]")
        for index, item in enumerate(sequence(task.get("foils"), f"{label}.foils"))
    ]
    require(len(result["foils"]) == 3, f"{label} must have exactly three foils")
    concept_ids = [result["target"]["id"]] + [item["id"] for item in result["foils"]]
    require(len(concept_ids) == len(set(concept_ids)), f"{label} concept IDs overlap")
    result["task_card"] = dict(mapping(task.get("task_card"), f"{label}.task_card"))
    require(
        isinstance(task.get("primary_control_eligible"), bool),
        f"{label}.primary_control_eligible must be boolean",
    )
    result["primary_control_eligible"] = task["primary_control_eligible"]
    result["control_match_status"] = text(
        task.get("control_match_status"), f"{label}.control_match_status"
    )
    require(
        result["primary_control_eligible"]
        is result["control_match_status"].startswith("matched_"),
        f"{label} control eligibility/status mismatch",
    )
    return result


def _normal_profile(value: Any, label: str) -> dict[str, Any]:
    profile = mapping(value, label)
    result = {
        "final_norm_max_abs_tolerance": finite(
            profile.get("final_norm_max_abs_tolerance"), f"{label}.norm max"
        ),
        "final_norm_rms_tolerance": finite(
            profile.get("final_norm_rms_tolerance"), f"{label}.norm RMS"
        ),
        "final_logits_max_abs_tolerance": finite(
            profile.get("final_logits_max_abs_tolerance"), f"{label}.logits max"
        ),
        "final_logits_rms_tolerance": finite(
            profile.get("final_logits_rms_tolerance"), f"{label}.logits RMS"
        ),
        "top_k_prefix": integer(profile.get("top_k_prefix"), f"{label}.top-k", minimum=1),
    }
    require(
        all(result[key] >= 0.0 for key in result if key != "top_k_prefix"),
        f"{label} tolerances must be nonnegative",
    )
    return result


def validate_protocol(value: Any, *, protocol_sha256: str) -> dict[str, Any]:
    protocol = mapping(value, "protocol")
    require(protocol.get("schema_version") == 1, "protocol schema mismatch")
    require(protocol.get("kind") == PROTOCOL_KIND, "protocol kind mismatch")
    require(
        protocol.get("analysis_version") == "paired-evidence-update-development-v1"
        and protocol.get("lens_outputs_used_for_boundary_or_labels") is False,
        "protocol analysis version or label-isolation flag changed",
    )
    digest(protocol_sha256, "protocol SHA")

    band = mapping(protocol.get("fixed_layer_band"), "fixed_layer_band")
    require(
        band.get("layers") == list(FIXED_LAYERS),
        "fixed layer band must be exactly layers 24 through 47",
    )
    reduction = mapping(protocol.get("score_reduction"), "score_reduction")
    require(
        reduction
        == {
            "within_concept": "logmeanexp_over_declared_token_scores",
            "within_foil_set": "logmeanexp_over_three_concept_scores",
            "across_layers": "arithmetic_mean_over_fixed_layers_24_through_47",
            "layer_selection": "none",
        },
        "score reduction contract changed",
    )
    numerical = mapping(protocol.get("numerical_certification"), "numerical certification")
    profiles = {
        name: _normal_profile(numerical.get(name), f"numerical_certification.{name}")
        for name in ("primary_stable", "legacy_strict")
    }
    bootstrap = mapping(protocol.get("bootstrap"), "bootstrap")
    require(
        bootstrap.get("algorithm")
        == "hierarchical_repository_then_task_percentile_v1",
        "bootstrap algorithm changed",
    )
    bootstrap_value = {
        "algorithm": bootstrap["algorithm"],
        "seed": integer(bootstrap.get("seed"), "bootstrap.seed"),
        "samples": integer(bootstrap.get("samples"), "bootstrap.samples", minimum=1),
        "confidence_level": finite(
            bootstrap.get("confidence_level"), "bootstrap.confidence_level"
        ),
    }
    require(
        0.0 < bootstrap_value["confidence_level"] < 1.0,
        "bootstrap confidence level must be in (0, 1)",
    )

    pins = mapping(protocol.get("pins"), "pins")
    lenses = mapping(pins.get("lenses"), "pins.lenses")
    lens_pins: dict[str, dict[str, Any]] = {}
    for key in REPORT_LENS_KEYS.values():
        pin = dict(mapping(lenses.get(key), f"pins.lenses.{key}"))
        pin["sha256"] = digest(pin.get("sha256"), f"pins.lenses.{key}.sha256")
        lens_pins[key] = pin

    tasks = [
        _normal_task(item, f"tasks[{index}]")
        for index, item in enumerate(sequence(protocol.get("tasks"), "tasks"))
    ]
    require(bool(tasks), "protocol has no tasks")
    controls = mapping(protocol.get("controls"), "controls")
    require(
        controls.get("copy_frequency")
        == "paired_log1p_exact_form_token_count_margin"
        and controls.get("copy_recency")
        == "paired_negative_log1p_last_exact_form_token_distance_margin"
        and controls.get("wrong_task")
        == "score_every_task_target_on_every_task_pair_and_rank_own_target_by_after_minus_before_update",
        "copy or wrong-task control contract changed",
    )
    card_policy = mapping(protocol.get("task_card_policy"), "task_card_policy")
    decision = mapping(protocol.get("decision"), "decision")
    observed_fields = [
        text(item, "task_card_policy observed field").upper()
        for item in sequence(card_policy.get("observed_fields"), "task_card_policy.observed_fields")
    ]
    require(
        observed_fields == ["EVIDENCE", "WHERE", "NEXT"]
        and str(card_policy.get("lens_scored_field")).upper() == "WHY",
        "task-card observation/lens boundary changed",
    )
    for task in tasks:
        card_fields = {str(key).upper(): value for key, value in task["task_card"].items()}
        require(
            all(field in card_fields and card_fields[field] is not None for field in observed_fields),
            f"{task['id']} task card lacks an observed EVIDENCE/WHERE/NEXT field",
        )
    for field in ("id", "instance_id"):
        values = [task[field] for task in tasks]
        require(len(values) == len(set(values)), f"protocol task {field}s overlap")
    concept_ids = [
        concept["id"]
        for task in tasks
        for concept in [task["target"], *task["foils"]]
    ]
    require(len(concept_ids) == len(set(concept_ids)), "protocol concept IDs overlap")
    token_text: dict[int, str] = {}
    for task in tasks:
        for concept in [task["target"], *task["foils"]]:
            for form in concept["forms"]:
                prior = token_text.setdefault(form["token_id"], form["text"])
                require(prior == form["text"], f"token {form['token_id']} text changed")

    model_pin = dict(mapping(pins.get("model"), "pins.model")) if "model" in pins else {}
    return {
        "sha256": protocol_sha256,
        "raw": dict(protocol),
        "fixed_layers": list(FIXED_LAYERS),
        "profiles": profiles,
        "bootstrap": bootstrap_value,
        "lens_pins": lens_pins,
        "model_pin": model_pin,
        "tasks": tasks,
        "task_by_id": {task["id"]: task for task in tasks},
        "token_text": token_text,
        "context_token_ids": sorted(token_text),
        "controls": dict(controls),
        "card_policy": dict(card_policy),
        "observed_card_fields": observed_fields,
        "decision": dict(decision),
    }


def validate_manifest(
    value: Any,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    manifest = mapping(value, "manifest")
    require(manifest.get("schema_version") == 1, "manifest schema mismatch")
    require(manifest.get("kind") == MANIFEST_KIND, "manifest kind mismatch")
    require(
        manifest.get("analysis_version") == "paired-evidence-update-development-v1"
        and manifest.get("lens_outputs_used_for_boundary_or_labels") is False,
        "manifest analysis version or label-isolation flag changed",
    )
    digest(manifest_sha256, "manifest SHA")
    binding = mapping(manifest.get("protocol"), "manifest.protocol")
    require(
        binding.get("sha256") == protocol_sha256,
        "manifest protocol hash mismatch",
    )
    if "path" in binding:
        text(binding.get("path"), "manifest.protocol.path")
    require(
        manifest.get("fixed_layer_band") == protocol["raw"]["fixed_layer_band"]
        and manifest.get("score_reduction") == protocol["raw"]["score_reduction"],
        "manifest fixed layer or score reduction binding mismatch",
    )
    score_vocabulary = dict(mapping(manifest.get("score_vocabulary"), "manifest.score_vocabulary"))
    union_ids = [
        integer(item, "manifest score token ID")
        for item in sequence(score_vocabulary.get("token_ids"), "manifest.score_vocabulary.token_ids")
    ]
    require(union_ids == sorted(set(union_ids)), "manifest score IDs must be sorted and unique")
    require(union_ids == protocol["context_token_ids"], "manifest score IDs differ from protocol")
    require(
        score_vocabulary.get("token_ids_sha256") == sha256_json(union_ids)
        and score_vocabulary.get("token_count") == len(union_ids)
        and score_vocabulary.get("scope")
        == "union_of_all_declared_target_and_foil_forms_on_every_prompt",
        "manifest union score token hash mismatch",
    )
    token_text = mapping(score_vocabulary.get("token_text"), "manifest.score_vocabulary.token_text")
    require(
        token_text == {str(token_id): protocol["token_text"][token_id] for token_id in union_ids},
        "manifest score token text differs from protocol",
    )
    prompt_bundle = dict(mapping(manifest.get("prompt_bundle"), "manifest.prompt_bundle"))
    prompt_bundle["sha256"] = digest(
        prompt_bundle.get("sha256"), "manifest.prompt_bundle.sha256"
    )
    prompt_bundle["bytes"] = integer(prompt_bundle.get("bytes"), "manifest.prompt_bundle.bytes", minimum=1)
    prompt_bundle["count"] = integer(prompt_bundle.get("count"), "manifest.prompt_bundle.count", minimum=1)
    require(
        prompt_bundle.get("serialization")
        == "indented_sorted_key_ascii_json_with_trailing_newline",
        "manifest prompt serialization changed",
    )
    if "path" in prompt_bundle:
        text(prompt_bundle.get("path"), "manifest.prompt_bundle.path")

    task_values = sequence(manifest.get("tasks"), "manifest.tasks")
    require(len(task_values) == len(protocol["tasks"]), "manifest task count mismatch")
    tasks: list[dict[str, Any]] = []
    prompt_ids: set[str] = set()
    for index, (raw_task, declared) in enumerate(
        zip(task_values, protocol["tasks"], strict=True)
    ):
        label = f"manifest.tasks[{index}]"
        task = dict(mapping(raw_task, label))
        require(task.get("task_ordinal") == index, f"{label}.task_ordinal mismatch")
        task_identity = mapping(task.get("task"), f"{label}.task")
        expected_identity = {
            field: declared[field]
            for field in (
                "id",
                "instance_id",
                "repo",
                "cohort",
                "after_global_request_index",
                "after_task_request_index",
                "stratum",
                "primary_control_eligible",
                "control_match_status",
            )
        }
        require(task_identity == expected_identity, f"{label}.task identity mismatch")
        sources = mapping(task.get("raw_sources"), f"{label}.raw_sources")
        require(set(sources) == {"before", "after", "label"}, f"{label}.raw_sources changed")
        for state in ("before", "after", "label"):
            source = mapping(sources.get(state), f"{label}.raw_sources.{state}")
            text(source.get("path"), f"{label}.raw_sources.{state}.path")
            integer(source.get("bytes"), f"{label}.raw_sources.{state}.bytes", minimum=1)
            require(
                source.get("sha256") == declared["raw_sha256"][state],
                f"{label}.raw_sources.{state} SHA mismatch",
            )
        mapping(task.get("boundary_audit"), f"{label}.boundary_audit")
        future = mapping(task.get("future_label_audit"), f"{label}.future_label_audit")
        require(
            future.get("target_present") is True
            and future.get("all_foils_absent") is True
            and future.get("future_text_retained") is False,
            f"{label} future-label audit failed",
        )
        prompt_records = sequence(task.get("prompts"), f"{label}.prompts")
        require(len(prompt_records) == 2, f"{label} must have two prompt manifest records")
        normal_prompts: dict[str, dict[str, Any]] = {}
        for prompt_index, raw_prompt in enumerate(prompt_records):
            prompt_label = f"{label}.prompts[{prompt_index}]"
            prompt_record = dict(mapping(raw_prompt, prompt_label))
            state = prompt_record.get("state")
            require(state in {"before", "after"} and state not in normal_prompts, f"{prompt_label} state mismatch")
            prompt_id = text(prompt_record.get("id"), f"{prompt_label}.id")
            require(prompt_id not in prompt_ids, f"{prompt_label} prompt ID overlaps")
            prompt_ids.add(prompt_id)
            prompt_record["prompt_sha256"] = digest(
                prompt_record.get("prompt_sha256"), f"{prompt_label}.prompt_sha256"
            )
            prompt_record["token_ids_sha256"] = digest(
                prompt_record.get("token_ids_sha256"), f"{prompt_label}.token_ids_sha256"
            )
            require(
                prompt_record.get("score_token_ids_sha256") == sha256_json(union_ids)
                and prompt_record.get("score_token_count") == len(union_ids),
                f"{prompt_label} score vocabulary mismatch",
            )
            integer(prompt_record.get("prompt_token_count"), f"{prompt_label}.prompt_token_count", minimum=1)
            mapping(prompt_record.get("exposure"), f"{prompt_label}.exposure")
            normal_prompts[state] = prompt_record
        require(
            set(normal_prompts) == {"before", "after"},
            f"{label} must bind exact before/after prompt records",
        )
        concepts = mapping(task.get("concepts"), f"{label}.concepts")
        require(
            concepts.get("target") == declared["target"]
            and concepts.get("foils") == declared["foils"],
            f"{label} concepts differ from protocol",
        )
        require(task.get("task_card") == declared["task_card"], f"{label}.task_card mismatch")
        task.update(
            {
                "task": dict(task_identity),
                "prompt_by_state": normal_prompts,
                "concepts": dict(concepts),
            }
        )
        tasks.append(task)
    expected_count = 2 * len(tasks)
    require(
        manifest.get("task_count") == len(tasks)
        and manifest.get("prompt_count") == expected_count
        and prompt_bundle["count"] == expected_count
        and len(prompt_ids) == expected_count,
        "manifest task or prompt count mismatch",
    )
    return {
        "sha256": manifest_sha256,
        "raw": dict(manifest),
        "protocol_binding": dict(binding),
        "score_vocabulary": score_vocabulary,
        "prompt_bundle": prompt_bundle,
        "tasks": tasks,
        "task_by_id": {task["task"]["id"]: task for task in tasks},
        "prompt_ids": prompt_ids,
    }


def _prompt_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    bundle = mapping(value, "prompt bundle")
    return sequence(bundle.get("prompts"), "prompt bundle.prompts")


def _validate_identifier_evidence(
    value: Any,
    label: str,
    *,
    normalization: str,
    matching: str,
) -> dict[str, Any]:
    evidence = dict(mapping(value, label))
    count = integer(evidence.get("identifier_occurrences"), f"{label}.identifier_occurrences")
    require(
        evidence.get("normalization") == normalization
        and evidence.get("matching") == matching
        and evidence.get("present") is (count > 0),
        f"{label} identifier evidence is inconsistent",
    )
    per_alias = sequence(evidence.get("per_alias"), f"{label}.per_alias")
    require(
        math.fsum(
            integer(mapping(item, f"{label}.per_alias").get("occurrences"), f"{label} alias count")
            for item in per_alias
        )
        == count,
        f"{label} per-alias counts do not sum to the total",
    )
    return evidence


def _validate_concept_exposure(
    value: Any,
    *,
    concept: Mapping[str, Any],
    state: str,
    prompt_token_ids: Sequence[int],
    label: str,
) -> None:
    evidence = mapping(value, label)
    require(evidence.get("id") == concept["id"], f"{label} concept ID mismatch")
    _validate_identifier_evidence(
        evidence,
        label,
        normalization="case_sensitive_identifier_boundary_v1",
        matching="exact_alias_with_ascii_identifier_boundaries",
    )
    require(
        evidence.get("source")
        == "newline_joined_recursive_string_values_from_raw_request_messages",
        f"{label} exposure source changed",
    )
    digest(evidence.get("raw_messages_sha256"), f"{label}.raw_messages_sha256")
    expected = mapping(concept.get("expected_exposure"), f"{label} expected exposure")
    expected_state = mapping(expected.get(state), f"{label} expected exposure {state}")
    require(
        evidence.get("present") == expected_state.get("present")
        and evidence.get("identifier_occurrences")
        == expected_state.get("identifier_occurrences"),
        f"{label} differs from the protocol exposure declaration",
    )
    supplemental = mapping(evidence.get("supplemental_rendered"), f"{label}.supplemental_rendered")
    _validate_identifier_evidence(
        supplemental.get("case_sensitive"),
        f"{label}.supplemental_rendered.case_sensitive",
        normalization="case_sensitive_identifier_boundary_v1",
        matching="exact_alias_with_ascii_identifier_boundaries",
    )
    _validate_identifier_evidence(
        supplemental.get("nfkc_casefold"),
        f"{label}.supplemental_rendered.nfkc_casefold",
        normalization="NFKC_then_casefold",
        matching="exact_alias_with_unicode_identifier_boundaries",
    )
    form_records = sequence(evidence.get("forms"), f"{label}.forms")
    require(len(form_records) == len(concept["forms"]), f"{label} form count mismatch")
    for index, (raw_form, declared_form) in enumerate(
        zip(form_records, concept["forms"], strict=True)
    ):
        form = mapping(raw_form, f"{label}.forms[{index}]")
        token_id = declared_form["token_id"]
        indices = [
            position for position, observed in enumerate(prompt_token_ids) if observed == token_id
        ]
        require(
            form.get("kind") == declared_form.get("kind")
            and form.get("text") == declared_form["text"]
            and form.get("token_id") == token_id
            and form.get("token_occurrences") == len(indices)
            and form.get("last_token_distance")
            == (len(prompt_token_ids) - 1 - indices[-1] if indices else None),
            f"{label}.forms[{index}] exact-token exposure mismatch",
        )


def _validate_prompt_exposure(
    value: Any,
    *,
    task: Mapping[str, Any],
    state: str,
    prompt_token_ids: Sequence[int],
    label: str,
) -> None:
    exposure = mapping(value, label)
    _validate_concept_exposure(
        exposure.get("target"),
        concept=task["target"],
        state=state,
        prompt_token_ids=prompt_token_ids,
        label=f"{label}.target",
    )
    foils = sequence(exposure.get("foils"), f"{label}.foils")
    require(len(foils) == 3, f"{label} must contain three foil exposure records")
    for index, (record, concept) in enumerate(zip(foils, task["foils"], strict=True)):
        _validate_concept_exposure(
            record,
            concept=concept,
            state=state,
            prompt_token_ids=prompt_token_ids,
            label=f"{label}.foils[{index}]",
        )


def validate_prompt_bundle(
    value: Any,
    *,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    raw_prompts = _prompt_array(value)
    require(len(raw_prompts) == manifest["prompt_bundle"]["count"], "prompt count mismatch")
    prompts: list[dict[str, Any]] = []
    seen: set[str] = set()
    contextual_ids = set(protocol["context_token_ids"])
    union_ids: list[int] = []
    for index, raw_prompt in enumerate(raw_prompts):
        label = f"prompts[{index}]"
        prompt = dict(mapping(raw_prompt, label))
        prompt_id = text(prompt.get("id"), f"{label}.id")
        require(prompt_id not in seen, f"duplicate prompt ID: {prompt_id}")
        seen.add(prompt_id)
        prompt_text = prompt.get("text")
        require(isinstance(prompt_text, str), f"{label}.text must be text")
        token_ids = [
            integer(item, f"{label} token ID")
            for item in sequence(prompt.get("token_ids"), f"{label}.token_ids")
        ]
        require(bool(token_ids), f"{label} has no prompt tokens")
        score_ids = [
            integer(item, f"{label} score token ID")
            for item in sequence(prompt.get("score_token_ids"), f"{label}.score_token_ids")
        ]
        require(len(score_ids) == len(set(score_ids)), f"{label} score IDs overlap")
        require(contextual_ids.issubset(score_ids), f"{label} lacks union-scored contextual IDs")
        require(score_ids == protocol["context_token_ids"], f"{label} score vocabulary differs from manifest")
        union_ids.extend(item for item in score_ids if item not in union_ids)
        metadata = mapping(prompt.get("metadata"), f"{label}.metadata")
        require(metadata.get("kind") == PROMPT_KIND, f"{label} prompt kind mismatch")
        require(
            metadata.get("analysis_version") == "paired-evidence-update-development-v1"
            and metadata.get("protocol_sha256") == protocol["sha256"]
            and metadata.get("lens_outputs_used_for_boundary_or_labels") is False,
            f"{label} protocol hash mismatch",
        )
        state = metadata.get("state")
        require(state in {"before", "after"}, f"{label} state must be before or after")
        task_identity = mapping(metadata.get("task"), f"{label}.metadata.task")
        task_id = text(task_identity.get("id"), f"{label}.metadata.task.id")
        require(task_id in protocol["task_by_id"], f"{label} has unknown task")
        declared = protocol["task_by_id"][task_id]
        materialized = manifest["task_by_id"][task_id]
        prompt_manifest = materialized["prompt_by_state"][state]
        require(prompt_id == prompt_manifest["id"], f"{label} ID/state binding mismatch")
        expected_task_identity = {
            field: declared[field]
            for field in (
                "id",
                "instance_id",
                "repo",
                "cohort",
                "after_global_request_index",
                "after_task_request_index",
                "stratum",
                "primary_control_eligible",
                "control_match_status",
            )
        }
        require(task_identity == expected_task_identity, f"{label} task identity mismatch")
        require(metadata.get("raw_sha256") == declared["raw_sha256"], f"{label} raw SHA binding mismatch")
        require(
            metadata.get("concepts") == materialized["concepts"],
            f"{label} concept binding mismatch",
        )
        require(
            metadata.get("task_card") == declared["task_card"],
            f"{label} task card mismatch",
        )
        require(
            metadata.get("exposure") == prompt_manifest["exposure"],
            f"{label} exposure binding mismatch",
        )
        _validate_prompt_exposure(
            metadata.get("exposure"),
            task=declared,
            state=state,
            prompt_token_ids=token_ids,
            label=f"{label}.metadata.exposure",
        )
        require(
            metadata.get("fixed_layer_band") == protocol["raw"]["fixed_layer_band"]
            and metadata.get("score_reduction") == protocol["raw"]["score_reduction"],
            f"{label} layer/reduction binding mismatch",
        )
        prompt_provenance = mapping(metadata.get("prompt"), f"{label}.metadata.prompt")
        require(
            prompt_provenance.get("sha256") == sha256_text(prompt_text)
            and prompt_provenance.get("sha256") == prompt_manifest["prompt_sha256"],
            f"{label} rendered prompt hash mismatch",
        )
        require(
            prompt_provenance.get("token_ids_sha256") == sha256_json(token_ids)
            and prompt_provenance.get("token_ids_sha256")
            == prompt_manifest["token_ids_sha256"]
            and prompt_provenance.get("token_count") == len(token_ids)
            and prompt_manifest["prompt_token_count"] == len(token_ids),
            f"{label} token IDs hash mismatch",
        )
        prompt.update(
            {
                "id": prompt_id,
                "text": prompt_text,
                "token_ids": token_ids,
                "score_token_ids": score_ids,
                "metadata": dict(metadata),
                "task_id": task_id,
                "state": state,
            }
        )
        prompts.append(prompt)
    require(seen == manifest["prompt_ids"], "prompt ID coverage differs from manifest")
    for task in protocol["tasks"]:
        matched = [prompt for prompt in prompts if prompt["task_id"] == task["id"]]
        require(
            len(matched) == 2 and {prompt["state"] for prompt in matched} == {"before", "after"},
            f"{task['id']} must have exact before/after prompts",
        )
    require(
        union_ids == manifest["score_vocabulary"]["token_ids"],
        "prompt union score IDs differ from manifest",
    )
    return {
        "prompts": prompts,
        "by_id": {prompt["id"]: prompt for prompt in prompts},
        "by_task_state": {
            (prompt["task_id"], prompt["state"]): prompt for prompt in prompts
        },
        "union_score_token_ids": union_ids,
    }


def _scored_map(
    value: Any,
    *,
    expected_ids: Sequence[int],
    context_token_text: Mapping[int, str],
    label: str,
) -> dict[int, float]:
    record = mapping(value, label)
    values = sequence(record.get("scored_tokens"), f"{label}.scored_tokens")
    result: dict[int, float] = {}
    for index, raw in enumerate(values):
        item = mapping(raw, f"{label}.scored_tokens[{index}]")
        token_id = integer(item.get("token_id"), f"{label} scored token ID")
        require(token_id not in result, f"{label} repeats token {token_id}")
        if token_id in context_token_text:
            require(
                item.get("token") == context_token_text[token_id],
                f"{label} token {token_id} text mismatch",
            )
        result[token_id] = finite(item.get("score"), f"{label} token {token_id} score")
    require(list(result) == list(expected_ids), f"{label} scored IDs are incomplete or reordered")
    return result


def _top_ids(value: Any, label: str) -> list[int]:
    rows = sequence(value, label)
    require(len(rows) == 1, f"{label} must have one final position")
    return [
        integer(item, f"{label} top token ID")
        for item in sequence(mapping(rows[0], label).get("token_ids"), f"{label}.token_ids")
    ]


def _numerical_diagnostics(experiment: Mapping[str, Any], label: str) -> dict[str, Any]:
    generated = integer(experiment.get("generated_token_id"), f"{label} generated token")
    reconstructed = _top_ids(experiment.get("final_model_readout"), f"{label} reconstructed")
    captured = _top_ids(
        experiment.get("captured_final_model_readout"), f"{label} captured"
    )
    require(reconstructed and captured, f"{label} final readouts are empty")
    explicit_top1 = reconstructed[0] == generated and captured[0] == generated
    require(
        experiment.get("final_layer_top1_matches_greedy") is explicit_top1,
        f"{label} greedy top-1 flag is inconsistent",
    )
    norm = mapping(experiment.get("final_norm_reconstruction"), f"{label} final norm")
    logits = mapping(
        experiment.get("final_logits_reconstruction"), f"{label} final logits"
    )
    norm_max = finite(norm.get("max_abs_error"), f"{label} norm max error")
    norm_rms = finite(norm.get("rms_error"), f"{label} norm RMS error")
    logits_max = finite(logits.get("max_abs_error"), f"{label} logits max error")
    logits_rms = finite(logits.get("rms_error"), f"{label} logits RMS error")
    require(
        min(norm_max, norm_rms, logits_max, logits_rms) >= 0.0,
        f"{label} numerical errors must be nonnegative",
    )
    recorded_k = integer(logits.get("top_k_prefix"), f"{label} recorded top-k", minimum=1)
    require(
        len(reconstructed) >= recorded_k and len(captured) >= recorded_k,
        f"{label} final readout is shorter than recorded top-k",
    )
    top_k_match = reconstructed[:recorded_k] == captured[:recorded_k]
    require(
        logits.get("top_k_prefix_token_ids_match") is top_k_match,
        f"{label} top-k flag is inconsistent",
    )
    recorded_norm_ok = norm_max <= finite(
        norm.get("max_abs_tolerance"), f"{label} recorded norm max tolerance"
    ) and norm_rms <= finite(
        norm.get("rms_tolerance"), f"{label} recorded norm RMS tolerance"
    )
    require(
        norm.get("within_tolerance") is recorded_norm_ok,
        f"{label} recorded norm gate is inconsistent",
    )
    recorded_logits_ok = (
        logits_max
        <= finite(logits.get("max_abs_tolerance"), f"{label} recorded logits max tolerance")
        and logits_rms
        <= finite(logits.get("rms_tolerance"), f"{label} recorded logits RMS tolerance")
        and top_k_match
    )
    require(
        logits.get("within_tolerance") is recorded_logits_ok,
        f"{label} recorded logits gate is inconsistent",
    )
    return {
        "generated_token_id": generated,
        "top1_match": explicit_top1,
        "reconstructed_top_ids": reconstructed,
        "captured_top_ids": captured,
        "recorded_top_k": recorded_k,
        "norm_max_abs_error": norm_max,
        "norm_rms_error": norm_rms,
        "logits_max_abs_error": logits_max,
        "logits_rms_error": logits_rms,
    }


def _profile_gate(diagnostics: Mapping[str, Any], profile: Mapping[str, Any]) -> bool:
    top_k = profile["top_k_prefix"]
    reconstructed = diagnostics["reconstructed_top_ids"]
    captured = diagnostics["captured_top_ids"]
    return bool(
        diagnostics["top1_match"]
        and len(reconstructed) >= top_k
        and len(captured) >= top_k
        and reconstructed[:top_k] == captured[:top_k]
        and diagnostics["norm_max_abs_error"]
        <= profile["final_norm_max_abs_tolerance"]
        and diagnostics["norm_rms_error"] <= profile["final_norm_rms_tolerance"]
        and diagnostics["logits_max_abs_error"]
        <= profile["final_logits_max_abs_tolerance"]
        and diagnostics["logits_rms_error"]
        <= profile["final_logits_rms_tolerance"]
    )


def _runtime_identity(value: Any, label: str) -> dict[str, Any]:
    runtime = dict(mapping(value, label))
    load_seconds = finite(runtime.pop("model_load_seconds", None), f"{label}.model_load_seconds")
    require(load_seconds > 0.0, f"{label}.model_load_seconds must be positive")
    require(
        runtime.get("mtp_enabled") is False
        and runtime.get("enforce_eager") is True
        and runtime.get("language_model_only") is True
        and runtime.get("transport_dtype") == "torch.float32"
        and runtime.get("readout_dtype") == "torch.bfloat16",
        f"{label} is not the frozen residual-capture runtime",
    )
    return runtime


def validate_report(
    value: Any,
    *,
    label: str,
    protocol: Mapping[str, Any],
    prompts: Mapping[str, Any],
) -> dict[str, Any]:
    report = mapping(value, f"{label} report")
    require(report.get("schema_version") == REPORT_SCHEMA_VERSION, f"{label} report schema mismatch")
    require(report.get("score_encoding") == "unrounded-float32", f"{label} score encoding mismatch")
    require(
        report.get("status")
        in {"failed", "passed", "complete", "completed", "success", "succeeded"},
        f"{label} report status is not a complete terminal status",
    )
    pin = protocol["lens_pins"][REPORT_LENS_KEYS[label]]
    lens = mapping(report.get("lens"), f"{label}.lens")
    require(lens.get("sha256") == pin["sha256"], f"{label} lens SHA pin mismatch")
    for field in ("repo_id", "revision"):
        if field in pin:
            require(lens.get(field) == pin[field], f"{label} lens {field} pin mismatch")
    assertions = mapping(report.get("assertions"), f"{label}.assertions")
    for field in ("lens_hash_matches", "lens_metadata_matches", "model_architecture_matches"):
        require(assertions.get(field) is True, f"{label} integrity assertion failed: {field}")
    model = mapping(report.get("model"), f"{label}.model")
    for field, expected in protocol["model_pin"].items():
        if field in model and isinstance(expected, (str, int, bool)):
            require(model.get(field) == expected, f"{label} model {field} pin mismatch")

    global_vocabulary = mapping(report.get("scored_vocabulary"), f"{label} vocabulary")
    require(
        global_vocabulary.get("union_token_ids") == prompts["union_score_token_ids"],
        f"{label} global scored vocabulary IDs mismatch",
    )
    union_tokens = sequence(
        global_vocabulary.get("union_tokens"), f"{label} vocabulary.union_tokens"
    )
    require(
        len(union_tokens) == len(prompts["union_score_token_ids"]),
        f"{label} global scored vocabulary text count mismatch",
    )
    for token_id, token_value in zip(
        prompts["union_score_token_ids"], union_tokens, strict=True
    ):
        if token_id in protocol["token_text"]:
            require(
                token_value == protocol["token_text"][token_id],
                f"{label} global token {token_id} text mismatch",
            )

    experiments = sequence(report.get("experiments"), f"{label}.experiments")
    require(len(experiments) == len(prompts["prompts"]), f"{label} experiment count mismatch")
    rows: dict[str, dict[str, Any]] = {}
    for index, (raw_experiment, prompt) in enumerate(
        zip(experiments, prompts["prompts"], strict=True)
    ):
        experiment_label = f"{label}.experiments[{index}]/{prompt['id']}"
        experiment = mapping(raw_experiment, experiment_label)
        require(experiment.get("id") == prompt["id"], f"{experiment_label} prompt ID mismatch")
        require(experiment.get("prompt") == prompt["text"], f"{experiment_label} prompt text mismatch")
        require(
            experiment.get("prompt_token_ids") == prompt["token_ids"],
            f"{experiment_label} prompt token IDs mismatch",
        )
        require(
            experiment.get("metadata") == prompt["metadata"],
            f"{experiment_label} prompt metadata mismatch",
        )
        require(
            prompt["metadata"]["prompt"]["sha256"] == sha256_text(experiment["prompt"])
            and prompt["metadata"]["prompt"]["token_ids_sha256"]
            == sha256_json(experiment["prompt_token_ids"]),
            f"{experiment_label} experiment prompt/token hashes mismatch",
        )
        final_position = len(prompt["token_ids"]) - 1
        require(
            experiment.get("positions_requested") == [-1]
            and experiment.get("positions_resolved") == [final_position]
            and experiment.get("capture_positions_resolved") == [final_position]
            and experiment.get("final_validation_position") == final_position,
            f"{experiment_label} final-position binding mismatch",
        )
        vocabulary = mapping(
            experiment.get("scored_vocabulary"), f"{experiment_label}.scored_vocabulary"
        )
        require(
            vocabulary.get("token_ids") == prompt["score_token_ids"],
            f"{experiment_label} score IDs mismatch",
        )
        vocabulary_tokens = sequence(
            vocabulary.get("tokens"), f"{experiment_label}.scored_vocabulary.tokens"
        )
        require(
            len(vocabulary_tokens) == len(prompt["score_token_ids"]),
            f"{experiment_label} scored token text count mismatch",
        )
        for token_id, token_value in zip(
            prompt["score_token_ids"], vocabulary_tokens, strict=True
        ):
            if token_id in protocol["token_text"]:
                require(
                    token_value == protocol["token_text"][token_id],
                    f"{experiment_label} token {token_id} text mismatch",
                )

        diagnostics = _numerical_diagnostics(experiment, experiment_label)
        profile_gates = {
            name: _profile_gate(diagnostics, profile)
            for name, profile in protocol["profiles"].items()
        }
        residual = dict(
            mapping(
                experiment.get("residual_capture_manifest"),
                f"{experiment_label}.residual_capture_manifest",
            )
        )
        digest(residual.get("sha256"), f"{experiment_label} residual SHA")
        require(
            integer(residual.get("tensor_count"), f"{experiment_label} residual count", minimum=1)
            == 64
            and residual.get("token_positions") == [final_position],
            f"{experiment_label} residual manifest binding mismatch",
        )

        layers = sequence(experiment.get("layers"), f"{experiment_label}.layers")
        require(
            [mapping(item, "layer").get("layer") for item in layers]
            == list(FIXED_LAYERS),
            f"{experiment_label} layers must be exactly 24 through 47",
        )
        evidence: dict[str, dict[int, dict[int, float]]] = {"jacobian": {}, "logit": {}}
        for raw_layer in layers:
            layer = mapping(raw_layer, f"{experiment_label}.layer")
            layer_id = integer(layer.get("layer"), f"{experiment_label} layer")
            positions = sequence(
                layer.get("positions"), f"{experiment_label}.layer-{layer_id}.positions"
            )
            require(len(positions) == 1, f"{experiment_label} layer {layer_id} position count")
            position = mapping(positions[0], f"{experiment_label}.layer-{layer_id}.position")
            require(
                position.get("capture_index") == 0
                and position.get("token_position") == final_position,
                f"{experiment_label} layer {layer_id} capture binding mismatch",
            )
            evidence["jacobian"][layer_id] = _scored_map(
                position.get("jacobian_lens"),
                expected_ids=prompt["score_token_ids"],
                context_token_text=protocol["token_text"],
                label=f"{experiment_label}.layer-{layer_id}.jacobian",
            )
            evidence["logit"][layer_id] = _scored_map(
                position.get("logit_lens"),
                expected_ids=prompt["score_token_ids"],
                context_token_text=protocol["token_text"],
                label=f"{experiment_label}.layer-{layer_id}.logit",
            )
        rows[prompt["id"]] = {
            "prompt": prompt,
            "evidence": evidence,
            "diagnostics": diagnostics,
            "profile_gates": profile_gates,
            "residual_capture_manifest": residual,
        }
    return {
        "label": label,
        "status": report.get("status"),
        "lens": dict(lens),
        "model": dict(model),
        "runtime": _runtime_identity(report.get("runtime"), f"{label}.runtime"),
        "rows": rows,
    }


def validate_report_pairing(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    reference = reports["public"]
    for label, report in reports.items():
        if label == "public":
            continue
        require(report["model"] == reference["model"], f"public/{label} model identities differ")
        require(report["runtime"] == reference["runtime"], f"public/{label} runtime identities differ")
        require(report["rows"].keys() == reference["rows"].keys(), f"public/{label} prompt grids differ")
        for prompt_id, left in reference["rows"].items():
            right = report["rows"][prompt_id]
            require(
                right["prompt"] == left["prompt"],
                f"public/{label} prompt binding differs: {prompt_id}",
            )
            require(
                right["residual_capture_manifest"] == left["residual_capture_manifest"],
                f"public/{label} residual_capture_manifest differs: {prompt_id}",
            )
            require(
                right["evidence"]["logit"] == left["evidence"]["logit"],
                f"public/{label} ordinary logit readout differs: {prompt_id}",
            )
            require(
                right["diagnostics"] == left["diagnostics"],
                f"public/{label} numerical diagnostics differ: {prompt_id}",
            )
    return {
        "report_labels": list(reports),
        "prompt_count": len(reference["rows"]),
        "exact_prompt_pairing": True,
        "residual_capture_manifests_equal": True,
        "ordinary_logit_readouts_equal": True,
        "numerical_diagnostics_equal": True,
    }


def _concept_score(
    evidence: Mapping[int, Mapping[int, float]], concept: Mapping[str, Any]
) -> float:
    token_ids = [form["token_id"] for form in concept["forms"]]
    layer_scores = [
        logmeanexp([evidence[layer][token_id] for token_id in token_ids])
        for layer in FIXED_LAYERS
    ]
    return math.fsum(layer_scores) / len(layer_scores)


def _average_tie_rank(values: Mapping[str, float], own_id: str) -> tuple[float, bool, int]:
    own = values[own_id]
    greater = sum(value > own for value in values.values())
    tied = sum(value == own for value in values.values())
    rank = 1.0 + greater + (tied - 1) / 2.0
    return rank, greater == 0, tied


def _task_method_row(
    task: Mapping[str, Any],
    *,
    method: str,
    report: Mapping[str, Any],
    prompts: Mapping[str, Any],
    all_tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    readout = "logit" if method == "ordinary_logit" else "jacobian"
    state_scores: dict[str, dict[str, Any]] = {}
    for state in ("before", "after"):
        prompt = prompts["by_task_state"][(task["id"], state)]
        evidence = report["rows"][prompt["id"]]["evidence"][readout]
        target_score = _concept_score(evidence, task["target"])
        foil_scores = [_concept_score(evidence, foil) for foil in task["foils"]]
        target_candidates = {
            candidate["id"]: _concept_score(evidence, candidate["target"])
            for candidate in all_tasks
        }
        state_scores[state] = {
            "target_score": target_score,
            "foil_scores": foil_scores,
            "foil_logmeanexp": logmeanexp(foil_scores),
            "target_vs_foils": target_score - logmeanexp(foil_scores),
            "task_target_scores": target_candidates,
        }
    updates = {
        candidate["id"]: state_scores["after"]["task_target_scores"][candidate["id"]]
        - state_scores["before"]["task_target_scores"][candidate["id"]]
        for candidate in all_tasks
    }
    own_update = updates[task["id"]]
    other_updates = [value for task_id, value in updates.items() if task_id != task["id"]]
    require(bool(other_updates), "context selectivity requires at least two tasks")
    context_selectivity = own_update - math.fsum(other_updates) / len(other_updates)
    same_repo_updates = [
        updates[candidate["id"]]
        for candidate in all_tasks
        if candidate["id"] != task["id"] and candidate["repo"] == task["repo"]
    ]
    same_repo_selectivity = (
        own_update - math.fsum(same_repo_updates) / len(same_repo_updates)
        if same_repo_updates
        else None
    )
    rank, top1, tie_count = _average_tie_rank(updates, task["id"])
    target_token_ids = {form["token_id"] for form in task["target"]["forms"]}
    immediate_matches = {
        state: report["rows"][prompts["by_task_state"][(task["id"], state)]["id"]][
            "diagnostics"
        ]["generated_token_id"]
        in target_token_ids
        for state in ("before", "after")
    }
    gate_by_profile = {
        profile: bool(
            report["rows"][prompts["by_task_state"][(task["id"], "before")]["id"]][
                "profile_gates"
            ][profile]
            and report["rows"][
                prompts["by_task_state"][(task["id"], "after")]["id"]
            ]["profile_gates"][profile]
        )
        for profile in ("primary_stable", "legacy_strict")
    }
    return {
        "task_id": task["id"],
        "instance_id": task["instance_id"],
        "repo": task["repo"],
        "cohort": task["cohort"],
        "primary_control_eligible": task["primary_control_eligible"],
        "control_match_status": task["control_match_status"],
        "method": method,
        "before": state_scores["before"],
        "after": state_scores["after"],
        "target_score_update": own_update,
        "target_vs_foils_update": state_scores["after"]["target_vs_foils"]
        - state_scores["before"]["target_vs_foils"],
        "task_target_updates": updates,
        "own_target_rank_by_update": rank,
        "own_target_reciprocal_rank": 1.0 / rank,
        "own_target_top1_by_update": top1,
        "own_target_tie_count": tie_count,
        "own_minus_other_mean_context_selectivity": context_selectivity,
        "same_repo_sibling_count": len(same_repo_updates),
        "own_minus_same_repo_sibling_mean": same_repo_selectivity,
        "immediate_next_token_target_match": immediate_matches,
        "immediate_next_token_control_pass": not any(immediate_matches.values()),
        "gate_by_profile": gate_by_profile,
    }


def _copy_state_score(prompt_token_ids: Sequence[int], concept: Mapping[str, Any]) -> dict[str, Any]:
    form_ids = list(dict.fromkeys(form["token_id"] for form in concept["forms"]))
    frequency = sum(prompt_token_ids.count(token_id) for token_id in form_ids)
    latest = max(
        (
            index
            for index, token_id in enumerate(prompt_token_ids)
            if token_id in set(form_ids)
        ),
        default=-1,
    )
    last_distance = len(prompt_token_ids) - 1 - latest if latest >= 0 else len(prompt_token_ids)
    return {
        "exact_form_token_count": frequency,
        "log1p_exact_form_token_count": math.log1p(frequency),
        "last_occurrence_index": latest,
        "last_exact_form_token_distance": last_distance,
        "negative_log1p_last_exact_form_token_distance": -math.log1p(last_distance),
        "prompt_token_count": len(prompt_token_ids),
    }


def _copy_baseline_row(
    task: Mapping[str, Any],
    *,
    prompts: Mapping[str, Any],
    all_tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    states: dict[str, dict[str, dict[str, Any]]] = {}
    for state in ("before", "after"):
        token_ids = prompts["by_task_state"][(task["id"], state)]["token_ids"]
        states[state] = {
            candidate["id"]: _copy_state_score(token_ids, candidate["target"])
            for candidate in all_tasks
        }
    result: dict[str, Any] = {
        "task_id": task["id"],
        "instance_id": task["instance_id"],
        "repo": task["repo"],
        "cohort": task["cohort"],
        "primary_control_eligible": task["primary_control_eligible"],
        "control_match_status": task["control_match_status"],
        "states": states,
    }
    copy_metrics = {
        "copy_frequency": "log1p_exact_form_token_count",
        "copy_recency": "negative_log1p_last_exact_form_token_distance",
    }
    for baseline, metric in copy_metrics.items():
        updates = {
            candidate["id"]: states["after"][candidate["id"]][metric]
            - states["before"][candidate["id"]][metric]
            for candidate in all_tasks
        }
        rank, top1, tie_count = _average_tie_rank(updates, task["id"])
        others = [value for task_id, value in updates.items() if task_id != task["id"]]
        local_state: dict[str, Any] = {}
        for state in ("before", "after"):
            target_score = states[state][task["id"]][metric]
            foil_scores = [
                _copy_state_score(
                    prompts["by_task_state"][(task["id"], state)]["token_ids"], foil
                )[metric]
                for foil in task["foils"]
            ]
            local_state[state] = {
                "target_score": target_score,
                "foil_scores": foil_scores,
                "foil_logmeanexp": logmeanexp(foil_scores),
                "target_vs_foils": target_score - logmeanexp(foil_scores),
            }
        result[baseline] = {
            "definition": (
                "paired_log1p_exact_form_token_count_margin"
                if baseline == "copy_frequency"
                else "paired_negative_log1p_last_exact_form_token_distance_margin"
            ),
            "before": local_state["before"],
            "after": local_state["after"],
            "target_vs_foils_update": local_state["after"]["target_vs_foils"]
            - local_state["before"]["target_vs_foils"],
            "updates": updates,
            "own_update": updates[task["id"]],
            "own_target_rank_by_update": rank,
            "own_target_reciprocal_rank": 1.0 / rank,
            "own_target_top1_by_update": top1,
            "own_target_tie_count": tie_count,
            "own_minus_other_mean_context_selectivity": updates[task["id"]]
            - math.fsum(others) / len(others),
        }
    return result


METRIC_PATHS = {
    "target_vs_foils_before": ("before", "target_vs_foils"),
    "target_vs_foils_after": ("after", "target_vs_foils"),
    "target_vs_foils_update": ("target_vs_foils_update",),
    "target_score_before": ("before", "target_score"),
    "target_score_after": ("after", "target_score"),
    "target_score_update": ("target_score_update",),
    "own_target_reciprocal_rank": ("own_target_reciprocal_rank",),
    "own_target_top1": ("own_target_top1_by_update",),
    "context_selectivity": ("own_minus_other_mean_context_selectivity",),
    "same_repo_context_selectivity": ("own_minus_same_repo_sibling_mean",),
}


def _path_value(row: Mapping[str, Any], path: Sequence[str]) -> float | None:
    value: Any = row
    for field in path:
        value = mapping(value, "metric row").get(field)
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    return finite(value, "metric value")


def _hierarchical_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    values: Mapping[str, float],
    *,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    require(bool(rows), "bootstrap rows are empty")
    repositories: dict[str, list[str]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        repositories.setdefault(str(row["repo"]), []).append(task_id)
    repo_ids = sorted(repositories)
    rng = random.Random(bootstrap["seed"])
    samples: list[float] = []
    for _ in range(bootstrap["samples"]):
        selected: list[float] = []
        for repo in (rng.choice(repo_ids) for _ in repo_ids):
            task_ids = repositories[repo]
            selected.extend(values[rng.choice(task_ids)] for _ in task_ids)
        samples.append(math.fsum(selected) / len(selected))
    alpha = (1.0 - bootstrap["confidence_level"]) / 2.0
    return {
        "algorithm": bootstrap["algorithm"],
        "seed": bootstrap["seed"],
        "samples": bootstrap["samples"],
        "confidence_level": bootstrap["confidence_level"],
        "confidence_interval": {
            "lower": percentile(samples, alpha),
            "upper": percentile(samples, 1.0 - alpha),
        },
    }


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    path: Sequence[str],
    *,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    selected: list[Mapping[str, Any]] = []
    values: dict[str, float] = {}
    for row in rows:
        value = _path_value(row, path)
        if value is not None:
            selected.append(row)
            values[str(row["task_id"])] = value
    if not selected:
        return {"status": "insufficient_no_tasks", "task_count": 0}
    estimate = math.fsum(values.values()) / len(values)
    return {
        "status": "available",
        "task_count": len(selected),
        "repository_count": len({str(row["repo"]) for row in selected}),
        "estimate": estimate,
        "weighting": "one_equal_weight_per_task",
        "bootstrap": _hierarchical_bootstrap(selected, values, bootstrap=bootstrap),
    }


def _profile_method_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    descriptively_eligible = [
        row
        for row in rows
        if row["gate_by_profile"][profile] and row["immediate_next_token_control_pass"]
    ]
    eligible = [row for row in descriptively_eligible if row["primary_control_eligible"]]
    return {
        "support": {
            "task_count": len(eligible),
            "repository_count": len({row["repo"] for row in eligible}),
            "numerically_excluded_task_ids": [row["task_id"] for row in rows if not row["gate_by_profile"][profile]],
            "control_ineligible_task_ids": [row["task_id"] for row in rows if not row["primary_control_eligible"]],
            "immediate_next_token_excluded_task_ids": [
                row["task_id"] for row in rows if not row["immediate_next_token_control_pass"]
            ],
            "before_and_after_gate_required": True,
            "primary_control_eligible_required": True,
        },
        "metrics": {
            name: _metric_summary(eligible, path, bootstrap=bootstrap)
            for name, path in METRIC_PATHS.items()
        },
        "descriptive_all_control_matches": {
            "label": "descriptive_only_includes_control_mismatches",
            "task_count": len(descriptively_eligible),
            "repository_count": len({row["repo"] for row in descriptively_eligible}),
            "metrics": {
                name: _metric_summary(descriptively_eligible, path, bootstrap=bootstrap)
                for name, path in METRIC_PATHS.items()
            },
        },
    }


def _paired_comparison(
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = {row["task_id"]: row for row in candidate_rows}
    reference = {row["task_id"]: row for row in reference_rows}
    require(candidate.keys() == reference.keys(), "paired method task grids differ")
    paired_rows: list[dict[str, Any]] = []
    for task_id in candidate:
        left = candidate[task_id]
        right = reference[task_id]
        if not (
            left["primary_control_eligible"]
            and right["primary_control_eligible"]
            and left["immediate_next_token_control_pass"]
            and right["immediate_next_token_control_pass"]
            and left["gate_by_profile"][profile]
            and right["gate_by_profile"][profile]
        ):
            continue
        row: dict[str, Any] = {
            "task_id": task_id,
            "repo": left["repo"],
            "cohort": left["cohort"],
        }
        require(left["repo"] == right["repo"], "paired method repository mismatch")
        for metric, path in METRIC_PATHS.items():
            left_value = _path_value(left, path)
            right_value = _path_value(right, path)
            row[metric] = (
                left_value - right_value
                if left_value is not None and right_value is not None
                else None
            )
        paired_rows.append(row)
    return {
        "support": {
            "task_count": len(paired_rows),
            "repository_count": len({row["repo"] for row in paired_rows}),
            "joint_before_and_after_gate_required": True,
            "primary_control_eligible_required": True,
            "immediate_next_token_target_matches_excluded": True,
        },
        "metrics": {
            metric: _metric_summary(paired_rows, (metric,), bootstrap=bootstrap)
            for metric in METRIC_PATHS
        },
    }


def _copy_summary(
    rows: Sequence[Mapping[str, Any]], *, bootstrap: Mapping[str, Any], primary_only: bool = True
) -> dict[str, Any]:
    selected = (
        [row for row in rows if row["primary_control_eligible"]]
        if primary_only
        else list(rows)
    )
    result: dict[str, Any] = {}
    for baseline in ("copy_frequency", "copy_recency"):
        result[baseline] = {
            metric: _metric_summary(
                selected,
                (baseline, field),
                bootstrap=bootstrap,
            )
            for metric, field in {
                "target_vs_foils_update": "target_vs_foils_update",
                "own_target_reciprocal_rank": "own_target_reciprocal_rank",
                "own_target_top1": "own_target_top1_by_update",
                "context_selectivity": "own_minus_other_mean_context_selectivity",
            }.items()
        }
    return result


def _copy_retrieval_diagnostic(
    public_rows: Sequence[Mapping[str, Any]],
    copy_rows: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    copy_by_task = {row["task_id"]: row for row in copy_rows}
    rows: list[dict[str, Any]] = []
    for public in public_rows:
        if not (
            public["primary_control_eligible"]
            and public["immediate_next_token_control_pass"]
            and public["gate_by_profile"][profile]
        ):
            continue
        copy_row = copy_by_task[public["task_id"]]
        row: dict[str, Any] = {
            "task_id": public["task_id"],
            "repo": public["repo"],
            "cohort": public["cohort"],
        }
        for baseline in ("copy_frequency", "copy_recency"):
            row[f"mrr_minus_{baseline}"] = public["own_target_reciprocal_rank"] - copy_row[
                baseline
            ]["own_target_reciprocal_rank"]
            row[f"top1_minus_{baseline}"] = float(public["own_target_top1_by_update"]) - float(
                copy_row[baseline]["own_target_top1_by_update"]
            )
        rows.append(row)
    metrics = {
        name: _metric_summary(rows, (name,), bootstrap=bootstrap)
        for name in (
            "mrr_minus_copy_frequency",
            "top1_minus_copy_frequency",
            "mrr_minus_copy_recency",
            "top1_minus_copy_recency",
        )
    }
    positive = {
        name: record.get("status") == "available" and record.get("estimate", 0.0) > 0.0
        for name, record in metrics.items()
    }
    passed = all(positive.values())
    return {
        "role": "descriptive retrieval-scale control; does not alter the frozen protocol point rule",
        "task_count": len(rows),
        "metrics": metrics,
        "positive_point_delta_criteria": positive,
        "passed": passed,
        "interpretation": (
            "public J-lens beats both exact-token copy retrieval baselines"
            if passed
            else "public J-lens does not beat both exact-token copy retrieval baselines"
        ),
    }


def _estimate(summary: Mapping[str, Any], metric: str) -> float | None:
    record = mapping(mapping(summary.get("metrics"), "summary.metrics").get(metric), metric)
    return float(record["estimate"]) if record.get("status") == "available" else None


def _interval_lower(summary: Mapping[str, Any], metric: str) -> float | None:
    record = mapping(mapping(summary.get("metrics"), "summary.metrics").get(metric), metric)
    if record.get("status") != "available":
        return None
    interval = mapping(mapping(record.get("bootstrap"), "metric bootstrap").get("confidence_interval"), "confidence interval")
    return finite(interval.get("lower"), "confidence interval lower")


def _public_usefulness_decision(
    method_summary: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    decision_contract: Mapping[str, Any],
) -> dict[str, Any]:
    support = mapping(method_summary.get("support"), "public method support")
    public_margin = _estimate(method_summary, "target_vs_foils_update")
    public_context = _estimate(method_summary, "context_selectivity")
    delta_margin = _estimate(comparison, "target_vs_foils_update")
    delta_context = _estimate(comparison, "context_selectivity")
    criteria = {
        "minimum_primary_tasks": support.get("task_count", 0)
        >= integer(decision_contract.get("minimum_primary_tasks"), "minimum primary tasks", minimum=1),
        "minimum_primary_repositories": support.get("repository_count", 0)
        >= integer(
            decision_contract.get("minimum_primary_repositories"),
            "minimum primary repositories",
            minimum=1,
        ),
        "positive_mean_target_margin_update": public_margin is not None and public_margin > 0.0,
        "positive_public_j_minus_ordinary_logit_target_margin_update": delta_margin is not None
        and delta_margin > 0.0,
        "positive_mean_wrong_task_context_selectivity": public_context is not None
        and public_context > 0.0,
        "positive_public_j_minus_ordinary_logit_context_selectivity": delta_context is not None
        and delta_context > 0.0,
    }
    passed = all(criteria.values())
    interval_criteria = {
        "public_target_margin_update_ci_lower_positive": (
            (_interval_lower(method_summary, "target_vs_foils_update") or 0.0) > 0.0
        ),
        "public_minus_logit_target_margin_update_ci_lower_positive": (
            (_interval_lower(comparison, "target_vs_foils_update") or 0.0) > 0.0
        ),
        "public_context_selectivity_ci_lower_positive": (
            (_interval_lower(method_summary, "context_selectivity") or 0.0) > 0.0
        ),
        "public_minus_logit_context_selectivity_ci_lower_positive": (
            (_interval_lower(comparison, "context_selectivity") or 0.0) > 0.0
        ),
    }
    interval_supported = all(interval_criteria.values())
    return {
        "classification": (
            "frozen_directional_point_rule_pass"
            if passed
            else "frozen_directional_point_rule_fail"
        ),
        "classification_is_not_operational_usefulness": True,
        "passed": passed,
        "criteria": criteria,
        "failed_criteria": [name for name, value in criteria.items() if not value],
        "point_estimates": {
            "public_target_margin_update": public_margin,
            "public_context_selectivity": public_context,
            "public_minus_logit_target_margin_update": delta_margin,
            "public_minus_logit_context_selectivity": delta_context,
        },
        "uncertainty_diagnostic": {
            "role": "reported separately because the frozen usefulness rule is point-estimate directional",
            "criteria": interval_criteria,
            "all_four_confidence_interval_lowers_positive": interval_supported,
            "interpretation": (
                "directional effects are interval-supported"
                if interval_supported
                else "directional point-rule result is not interval-supported"
            ),
        },
        "scope": "development pilot; broad claims require independent preregistered tasks",
    }


def _card_value(task_card: Mapping[str, Any], field: str) -> Any:
    for key, value in task_card.items():
        if str(key).upper() == field:
            return value
    raise ValueError(f"task card lacks {field}")


def build_cards(
    protocol: Mapping[str, Any],
    method_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    public = {row["task_id"]: row for row in method_rows["public_jacobian"]}
    ordinary = {row["task_id"]: row for row in method_rows["ordinary_logit"]}
    cards: list[dict[str, Any]] = []
    for task in protocol["tasks"]:
        jacobian = public[task["id"]]
        logit = ordinary[task["id"]]
        criteria = {
            "public_before_and_after_primary_stable_gate": jacobian["gate_by_profile"][
                "primary_stable"
            ],
            "primary_control_eligible": jacobian["primary_control_eligible"],
            "immediate_next_token_control_pass": jacobian[
                "immediate_next_token_control_pass"
            ],
            "public_target_margin_update_positive": jacobian["target_vs_foils_update"] > 0.0,
            "public_context_selectivity_positive": jacobian[
                "own_minus_other_mean_context_selectivity"
            ]
            > 0.0,
            "public_minus_logit_target_margin_update_positive": jacobian[
                "target_vs_foils_update"
            ]
            - logit["target_vs_foils_update"]
            > 0.0,
            "public_minus_logit_context_selectivity_positive": jacobian[
                "own_minus_other_mean_context_selectivity"
            ]
            - logit["own_minus_other_mean_context_selectivity"]
            > 0.0,
        }
        supported = all(criteria.values())
        reasons = [name for name, passed in criteria.items() if not passed]
        target_label = next(
            (
                task["target"].get(key)
                for key in ("text", "target", "label", "value", "id")
                if task["target"].get(key)
            ),
            task["target"]["id"],
        )
        fields: dict[str, Any] = {}
        for field in CARD_FIELDS:
            observed = field in protocol["observed_card_fields"]
            fields[field] = {
                "observed": {
                    "status": (
                        "observed_from_bound_qwen_code_trajectory"
                        if observed
                        else "not_an_observed_card_field"
                    ),
                    "value": _card_value(task["task_card"], field) if observed else None,
                    "lens_inference": False,
                },
                "lens": {
                    "status": (
                        "supported_contextual_evidence_summary"
                        if field == "WHY" and supported
                        else "withheld"
                    ),
                    "claim": (
                        f"Fixed-band public J-lens evidence for {target_label!r} increased "
                        "more than its same-task foils, task-external targets, and the "
                        "corresponding ordinary-logit effects."
                        if field == "WHY" and supported
                        else None
                    ),
                    "withheld_reasons": (
                        []
                        if field == "WHY" and supported
                        else reasons
                        if field == "WHY"
                        else ["field_is_observed_context_not_inferred_from_lens"]
                    ),
                },
            }
        cards.append(
            {
                "task_id": task["id"],
                "instance_id": task["instance_id"],
                "repo": task["repo"],
                "cohort": task["cohort"],
                "target_concept_id": task["target"]["id"],
                "fields": fields,
                "lens_why_guard": {
                    "status": "supported" if supported else "withheld",
                    "criteria": criteria,
                    "withheld_reasons": reasons,
                    "scores": {
                        "public_target_margin_update": jacobian["target_vs_foils_update"],
                        "public_context_selectivity": jacobian[
                            "own_minus_other_mean_context_selectivity"
                        ],
                        "public_minus_logit_target_margin_update": jacobian[
                            "target_vs_foils_update"
                        ]
                        - logit["target_vs_foils_update"],
                        "public_minus_logit_context_selectivity": jacobian[
                            "own_minus_other_mean_context_selectivity"
                        ]
                        - logit["own_minus_other_mean_context_selectivity"],
                    },
                },
            }
        )
    return {
        "schema_version": 1,
        "kind": CARDS_KIND,
        "label": "OBSERVED trajectory fields plus guarded lens evidence; not recovered chain of thought",
        "cards": cards,
        "limitations": [
            "The lens scores predeclared concept/entity tokens; it does not recover private chain-of-thought text.",
            "Concept alignment is not evidence for a causal relation, semantic proposition, or why the model acted.",
            "Targets exposed in the prompt measure contextual evidence reweighting, not hidden-future prediction.",
        ],
    }


def analyze(
    protocol_value: Any,
    manifest_value: Any,
    prompts_value: Any,
    public_report_value: Any,
    *,
    protocol_sha256: str,
    manifest_sha256: str,
    nf4_report_value: Any | None = None,
    native_report_value: Any | None = None,
    report_sha256s: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = validate_protocol(protocol_value, protocol_sha256=protocol_sha256)
    manifest = validate_manifest(
        manifest_value,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha256,
    )
    prompts = validate_prompt_bundle(prompts_value, protocol=protocol, manifest=manifest)
    reports = {
        "public": validate_report(
            public_report_value,
            label="public",
            protocol=protocol,
            prompts=prompts,
        )
    }
    if nf4_report_value is not None:
        reports["nf4"] = validate_report(
            nf4_report_value,
            label="nf4",
            protocol=protocol,
            prompts=prompts,
        )
    if native_report_value is not None:
        reports["native"] = validate_report(
            native_report_value,
            label="native",
            protocol=protocol,
            prompts=prompts,
        )
    pairing = validate_report_pairing(reports)
    if report_sha256s is None:
        report_hashes = {
            label: {
                "sha256": sha256_json(
                    {
                        "public": public_report_value,
                        "nf4": nf4_report_value,
                        "native": native_report_value,
                    }[label]
                ),
                "basis": "canonical_sorted_compact_ascii_json_value",
            }
            for label in reports
        }
    else:
        require(set(report_sha256s) == set(reports), "report SHA label coverage mismatch")
        report_hashes = {
            label: {
                "sha256": digest(report_sha256s[label], f"{label} report SHA"),
                "basis": "exact_input_file_bytes",
            }
            for label in reports
        }

    method_rows: dict[str, list[dict[str, Any]]] = {}
    for method, (report_label, _) in METHOD_SPECS.items():
        if report_label not in reports:
            continue
        method_rows[method] = [
            _task_method_row(
                task,
                method=method,
                report=reports[report_label],
                prompts=prompts,
                all_tasks=protocol["tasks"],
            )
            for task in protocol["tasks"]
        ]
    copy_rows = [
        _copy_baseline_row(task, prompts=prompts, all_tasks=protocol["tasks"])
        for task in protocol["tasks"]
    ]
    profiles: dict[str, Any] = {}
    for profile in ("primary_stable", "legacy_strict"):
        method_summaries = {
            method: _profile_method_summary(
                rows, profile=profile, bootstrap=protocol["bootstrap"]
            )
            for method, rows in method_rows.items()
        }
        comparisons = {
            f"{method}_minus_ordinary_logit": _paired_comparison(
                rows,
                method_rows["ordinary_logit"],
                profile=profile,
                bootstrap=protocol["bootstrap"],
            )
            for method, rows in method_rows.items()
            if method != "ordinary_logit"
        }
        if "native_jacobian" in method_rows:
            comparisons["public_jacobian_minus_native_jacobian"] = _paired_comparison(
                method_rows["public_jacobian"],
                method_rows["native_jacobian"],
                profile=profile,
                bootstrap=protocol["bootstrap"],
            )
        if "nf4_jacobian" in method_rows:
            comparisons["public_jacobian_minus_nf4_jacobian"] = _paired_comparison(
                method_rows["public_jacobian"],
                method_rows["nf4_jacobian"],
                profile=profile,
                bootstrap=protocol["bootstrap"],
            )
        profiles[profile] = {
            "numerical_profile": protocol["profiles"][profile],
            "methods": method_summaries,
            "paired_comparisons": comparisons,
            "public_usefulness_decision": _public_usefulness_decision(
                method_summaries["public_jacobian"],
                comparisons["public_jacobian_minus_ordinary_logit"],
                decision_contract=protocol["decision"],
            ),
            "copy_retrieval_diagnostic": _copy_retrieval_diagnostic(
                method_rows["public_jacobian"],
                copy_rows,
                profile=profile,
                bootstrap=protocol["bootstrap"],
            ),
        }

    cards = build_cards(protocol, method_rows)
    result = {
        "schema_version": 1,
        "kind": ANALYSIS_KIND,
        "label": "FROZEN CONTEXTUAL EVIDENCE UPDATE; concept readout, not chain-of-thought recovery",
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "prompt_bundle_sha256": manifest["prompt_bundle"]["sha256"],
        "task_count": len(protocol["tasks"]),
        "repository_count": len({task["repo"] for task in protocol["tasks"]}),
        "report_pairing": pairing,
        "inputs": {
            "protocol": {"sha256": protocol_sha256},
            "manifest": {"sha256": manifest_sha256},
            "prompt_bundle": {"sha256": manifest["prompt_bundle"]["sha256"]},
            "reports": report_hashes,
        },
        "report_status": {label: report["status"] for label, report in reports.items()},
        "score_reduction": dict(protocol["raw"]["score_reduction"]),
        "rank_tie_policy": "average rank among exact ties; top1 when no candidate is strictly greater",
        "profiles": profiles,
        "copy_baselines": {
            "definitions": {
                "copy_frequency": "paired log1p exact-form token-count target-versus-three-foil margin",
                "copy_recency": "paired negative log1p last exact-form token-distance target-versus-three-foil margin; absent distance equals prompt length",
                "comparison": "rank own target by AFTER minus BEFORE among all task targets",
            },
            "task_rows": copy_rows,
            "task_equal_summary": _copy_summary(
                copy_rows, bootstrap=protocol["bootstrap"]
            ),
            "descriptive_all_control_matches_summary": _copy_summary(
                copy_rows, bootstrap=protocol["bootstrap"], primary_only=False
            ),
        },
        "task_rows": method_rows,
        "cards_summary": {
            "supported_lens_why_count": sum(
                card["lens_why_guard"]["status"] == "supported" for card in cards["cards"]
            ),
            "withheld_lens_why_count": sum(
                card["lens_why_guard"]["status"] == "withheld" for card in cards["cards"]
            ),
        },
        "limitations": cards["limitations"],
    }
    return result, cards


def _read_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    return json.loads(raw), raw


def _resolve_prompts_path(path_value: str, manifest_path: Path) -> Path:
    supplied = Path(path_value)
    candidates = (
        [supplied]
        if supplied.is_absolute()
        else [Path.cwd() / supplied, ROOT / supplied, manifest_path.parent / supplied]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"materialized prompt bundle not found: {path_value}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument("--nf4-report", type=Path)
    parser.add_argument("--native-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cards-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_value, protocol_raw = _read_json(args.protocol)
    manifest_value, manifest_raw = _read_json(args.manifest)
    manifest_prompts = mapping(manifest_value, "manifest").get("prompt_bundle")
    prompt_binding = mapping(manifest_prompts, "manifest.prompt_bundle")
    prompts_path = _resolve_prompts_path(
        text(prompt_binding.get("path"), "manifest.prompt_bundle.path"), args.manifest
    )
    prompts_value, prompts_raw = _read_json(prompts_path)
    require(
        sha256_bytes(prompts_raw) == prompt_binding.get("sha256"),
        "materialized prompt bundle file hash mismatch",
    )
    require(
        len(prompts_raw) == prompt_binding.get("bytes"),
        "materialized prompt bundle byte count mismatch",
    )
    public_value, public_raw = _read_json(args.public_report)
    nf4_value, nf4_raw = _read_json(args.nf4_report) if args.nf4_report else (None, None)
    native_value, native_raw = (
        _read_json(args.native_report) if args.native_report else (None, None)
    )
    report_sha256s = {"public": sha256_bytes(public_raw)}
    if nf4_raw is not None:
        report_sha256s["nf4"] = sha256_bytes(nf4_raw)
    if native_raw is not None:
        report_sha256s["native"] = sha256_bytes(native_raw)
    analysis, cards = analyze(
        protocol_value,
        manifest_value,
        prompts_value,
        public_value,
        protocol_sha256=sha256_bytes(protocol_raw),
        manifest_sha256=sha256_bytes(manifest_raw),
        nf4_report_value=nf4_value,
        native_report_value=native_value,
        report_sha256s=report_sha256s,
    )
    atomic_write_json(args.output, analysis)
    atomic_write_json(args.cards_output, cards)
    print(
        f"wrote {args.output} and {args.cards_output} "
        f"({analysis['cards_summary']['supported_lens_why_count']} supported lens WHY cards)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
