#!/usr/bin/env python3
"""Materialize capture-matched task-start (C0M) SWE probe prompts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import materialize_swe_jlens_prompts as RENDER
import materialize_swe_multitask_c1_probes as C1
import materialize_swe_multitask_initial_probes as C0


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/swe_multitask_initial_protocol.json"
DEFAULT_C1_PROMPTS = (
    ROOT / "validation/jlens-swe-multitask-c1-prompts-2026-07-18.json"
)
DEFAULT_TEMPLATE = ROOT / "configs/qwen3-openai-codex.jinja"
DEFAULT_OUTPUT = ROOT / ".cache/swe_multitask_c0m/prompts.json"
EXPECTED_PROTOCOL_SHA256 = C1.EXPECTED_PROTOCOL_SHA256
EXPECTED_C1_PROMPT_BUNDLE_SHA256 = (
    "45a00ae55b61ef2370a88cf2d177b8fa703a70abf04b9c9f4aa0ca10d1ebb5e6"
)
C0M_CHECKPOINT = {
    "id": "C0M",
    "name": "capture_matched_task_start",
    "visibility_boundary": "same_captured_trajectory_before_first_assistant_token",
}


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def retained_subjects(concepts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    subjects: list[dict[str, Any]] = []
    for concept in concepts:
        subjects.append(
            {
                "subject": "target",
                "task_instance_id": None,
                "concept_id": concept["id"],
                "family": concept["family"],
                "target": concept["target"],
                "forms": copy.deepcopy(C1.require_list(concept.get("forms"), "target forms")),
            }
        )
        for raw_foil in C1.require_list(concept.get("foils"), "concept foils"):
            foil = C1.require_mapping(raw_foil, "concept foil")
            subjects.append(
                {
                    "subject": "foil",
                    "task_instance_id": foil["task_instance_id"],
                    "concept_id": foil["concept_id"],
                    "family": foil["family"],
                    "target": foil["target"],
                    "forms": copy.deepcopy(C1.require_list(foil.get("forms"), "foil forms")),
                }
            )
    return subjects


def validate_c1_concepts(
    c1_prompt: Mapping[str, Any],
    task: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[int]]:
    metadata = C1.require_mapping(c1_prompt.get("metadata"), "C1 prompt metadata")
    concepts = [
        dict(C1.require_mapping(value, "C1 retained concept"))
        for value in C1.require_list(metadata.get("concepts"), "C1 retained concepts")
    ]
    if not concepts:
        raise ValueError("C1 prompt has no retained concepts")

    protocol_concepts = {
        concept["id"]: concept
        for value in C1.require_list(task.get("concepts"), "protocol task concepts")
        for concept in [C1.require_mapping(value, "protocol task concept")]
    }
    if len(protocol_concepts) != len(task["concepts"]):
        raise ValueError("protocol task concept IDs are not unique")

    scored_ids: set[int] = set()
    for concept in concepts:
        source = protocol_concepts.get(concept.get("id"))
        if source is None:
            raise ValueError("C1 retained concept is absent from the frozen protocol task")
        expected = {
            "id": source["id"],
            "family": source["family"],
            "target": source["target"],
            "path": source["path"],
            "evidence": source["sources"],
            "visibility": "oracle_hidden",
            "forms": source["forms"],
        }
        for field, expected_value in expected.items():
            require_equal(concept.get(field), expected_value, f"C1 concept {field}")

        source_foils = {
            (foil["task_instance_id"], foil["concept_id"]): foil
            for value in C1.require_list(source.get("foils"), "protocol concept foils")
            for foil in [C1.require_mapping(value, "protocol concept foil")]
        }
        c1_foils = C1.require_list(concept.get("foils"), "C1 concept foils")
        for raw_foil in c1_foils:
            foil = C1.require_mapping(raw_foil, "C1 concept foil")
            identity = (foil.get("task_instance_id"), foil.get("concept_id"))
            require_equal(foil, source_foils.get(identity), "C1 retained foil")

        for subject in retained_subjects([concept]):
            forms = C1.require_list(subject["forms"], "retained scored forms")
            if not forms:
                raise ValueError("C1 retained target or foil has no token forms")
            for raw_form in forms:
                form = C1.require_mapping(raw_form, "retained scored form")
                _, token_id = C0._validate_form(form, tokenizer, "C0M retained form")
                scored_ids.add(token_id)

    score_token_ids = C1.require_list(c1_prompt.get("score_token_ids"), "C1 score token IDs")
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in score_token_ids)
        or len(score_token_ids) != len(set(score_token_ids))
        or set(score_token_ids) != scored_ids
    ):
        raise ValueError("C1 score vocabulary does not exactly cover retained forms")
    return copy.deepcopy(concepts), list(score_token_ids)


def audit_hidden_surfaces(
    concepts: Sequence[Mapping[str, Any]], rendered: str, tokenizer: Any, instance_id: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for subject in retained_subjects(concepts):
        if subject["subject"] == "target":
            subject["task_instance_id"] = instance_id
        evidence = C1.exposure_evidence(
            rendered,
            str(subject["target"]),
            C1.require_list(subject["forms"], "visibility forms"),
            tokenizer,
        )
        record = {**subject, "full_rendered_prompt": evidence, "hidden": not evidence["exposed"]}
        records.append(record)
        if evidence["exposed"]:
            raise ValueError(
                f"C0M full rendered prompt exposes {subject['subject']} "
                f"{subject['target']!r} for {instance_id}"
            )
    return records


def validate_usage(
    usage_record: Mapping[str, Any], *, request_index: int, prompt_token_count: int
) -> None:
    usage = C1.require_mapping(usage_record.get("usage"), "capture usage values")
    if usage_record.get("idx") != request_index:
        raise ValueError("capture usage request index mismatch")
    if usage.get("prompt_tokens") != prompt_token_count:
        raise ValueError("rendered request disagrees with the pinned usage token count")


def build_c0m_bundle(
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    c1_prompts: Sequence[Mapping[str, Any]],
    c1_prompt_bundle_sha256: str,
    requests: Sequence[Mapping[str, Any]],
    request_sources: Sequence[Mapping[str, Any]],
    usage_records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    template: str,
    capture_manifest_sha256: str,
    usage_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    C0._validate_protocol_header(protocol)
    tasks = C1.require_list(protocol.get("tasks"), "protocol tasks")
    if len(requests) != 2 * len(tasks) or len(request_sources) != len(requests):
        raise ValueError("capture must contain exactly two requests per frozen protocol task")
    if len(usage_records) != len(requests):
        raise ValueError("usage ledger and capture request counts differ")

    tasks_by_instance: dict[str, Mapping[str, Any]] = {}
    for task_index, raw_task in enumerate(tasks):
        task = C1.require_mapping(raw_task, f"protocol task {task_index}")
        require_equal(task.get("selection_index"), task_index, "protocol task selection order")
        instance_id = task.get("instance_id")
        if not isinstance(instance_id, str) or instance_id in tasks_by_instance:
            raise ValueError("protocol task instance IDs are invalid or duplicated")
        tasks_by_instance[instance_id] = task

    prompts: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    seen_instances: set[str] = set()
    previous_task_index = -1
    for raw_c1_prompt in c1_prompts:
        c1_prompt = C1.require_mapping(raw_c1_prompt, "C1 prompt")
        c1_metadata = C1.require_mapping(c1_prompt.get("metadata"), "C1 prompt metadata")
        require_equal(
            c1_metadata.get("kind"),
            "swe_verified_multitask_initial_probe",
            "C1 prompt kind",
        )
        require_equal(c1_metadata.get("protocol_sha256"), protocol_sha256, "C1 protocol hash")
        require_equal(c1_metadata.get("checkpoint"), C1.C1_CHECKPOINT, "C1 checkpoint")
        require_equal(
            c1_metadata.get("lens_outputs_used_for_selection"),
            False,
            "C1 lens-output selection flag",
        )
        require_equal(
            c1_metadata.get("middle_band_layers"),
            list(C0.MIDDLE_BAND_LAYERS),
            "C1 middle-band layers",
        )
        c1_task = C1.require_mapping(c1_metadata.get("task"), "C1 task metadata")
        instance_id = c1_task.get("instance_id")
        if not isinstance(instance_id, str) or instance_id in seen_instances:
            raise ValueError("C1 task instance IDs are invalid or duplicated")
        seen_instances.add(instance_id)
        task = tasks_by_instance.get(instance_id)
        if task is None:
            raise ValueError("C1 prompt task is absent from the frozen protocol")
        task_index = int(task["selection_index"])
        if task_index <= previous_task_index:
            raise ValueError("C1 prompts are not in frozen protocol order")
        previous_task_index = task_index

        expected_task = {
            "instance_id": task["instance_id"],
            "repo": task["repo"],
            "base_commit": task["base_commit"],
            "problem_statement_sha256": C1.sha256_text(task["problem_statement"]),
            "patch_sha256": task["patch_sha256"],
            "test_patch_sha256": task["test_patch_sha256"],
        }
        require_equal(c1_task, expected_task, "C1 task provenance")
        concepts, score_token_ids = validate_c1_concepts(c1_prompt, task, tokenizer)

        first_index = task_index * 2 + 1
        second_index = first_index + 1
        first = C1.require_mapping(requests[first_index - 1], "first capture request")
        second = C1.require_mapping(requests[second_index - 1], "second capture request")
        for source in request_sources[first_index - 1 : second_index]:
            require_equal(source.get("capture_instance_id"), instance_id, "capture task binding")
        C1.validate_pair(first, second, first_index=first_index, task=task)

        audit = C1.require_mapping(c1_metadata.get("observation_audit"), "C1 observation audit")
        bindings = {
            "capture_manifest_sha256": capture_manifest_sha256,
            "usage_manifest_sha256": usage_manifest_sha256,
            "first_request_index": first_index,
            "second_request_index": second_index,
            "first_request_sha256": request_sources[first_index - 1]["sha256"],
            "second_request_sha256": request_sources[second_index - 1]["sha256"],
        }
        for field, expected_value in bindings.items():
            require_equal(audit.get(field), expected_value, f"C1 observation {field}")

        first_rendered, first_token_ids, first_normalized_count, first_messages = C1.render_request(
            tokenizer, request=first, template=template
        )
        second_rendered, second_token_ids, second_normalized_count, second_messages = C1.render_request(
            tokenizer, request=second, template=template
        )
        validate_usage(
            C1.require_mapping(usage_records[first_index - 1], "first-request usage"),
            request_index=first_index,
            prompt_token_count=len(first_token_ids),
        )
        validate_usage(
            C1.require_mapping(usage_records[second_index - 1], "second-request usage"),
            request_index=second_index,
            prompt_token_count=len(second_token_ids),
        )
        if len(first_token_ids) > C1.MAX_PROMPT_TOKENS:
            raise ValueError(f"C0M request {first_index} exceeds the model context limit")

        require_equal(c1_prompt.get("text"), second_rendered, "paired C1 rendered prompt")
        require_equal(c1_prompt.get("token_ids"), second_token_ids, "paired C1 token IDs")
        require_equal(
            audit.get("normalized_messages_sha256"),
            C1.sha256_json(second_messages),
            "paired C1 normalized messages hash",
        )
        require_equal(
            audit.get("normalized_string_tool_call_arguments"),
            second_normalized_count,
            "paired C1 normalized tool-call argument count",
        )
        require_equal(
            first_normalized_count,
            0,
            "first-request normalized tool-call argument count",
        )
        first_raw_messages = C1.require_list(first.get("messages"), "first request messages")
        second_raw_messages = C1.require_list(second.get("messages"), "second request messages")
        if second_raw_messages[: len(first_raw_messages)] != first_raw_messages:
            raise ValueError("first request messages are not the exact C1 message prefix")
        if second_messages[: len(first_messages)] != first_messages:
            raise ValueError("normalized first messages are not the exact C1 message prefix")
        if not second_rendered.startswith(first_rendered):
            raise ValueError("rendered C0M prompt is not the exact C1 text prefix")
        if second_token_ids[: len(first_token_ids)] != first_token_ids:
            raise ValueError("C0M token IDs are not the exact C1 token prefix")

        visibility = audit_hidden_surfaces(concepts, first_rendered, tokenizer, instance_id)
        c1_prompt_id = c1_prompt.get("id")
        if not isinstance(c1_prompt_id, str) or not c1_prompt_id:
            raise ValueError("paired C1 prompt has no ID")
        capture_match = {
            **bindings,
            "c1_prompt_bundle_sha256": c1_prompt_bundle_sha256,
            "paired_c1_prompt_id": c1_prompt_id,
            "paired_c1_prompt_sha256": C1.sha256_text(second_rendered),
            "paired_c1_token_ids_sha256": C1.sha256_json(second_token_ids),
            "first_request_messages_sha256": C1.sha256_json(first_raw_messages),
            "first_normalized_messages_sha256": C1.sha256_json(first_messages),
            "paired_c1_normalized_messages_sha256": C1.sha256_json(second_messages),
            "first_normalized_string_tool_call_arguments": first_normalized_count,
            "paired_c1_normalized_string_tool_call_arguments": second_normalized_count,
            "rendered_text_prefix_bytes": len(first_rendered.encode("utf-8")),
            "token_id_prefix_count": len(first_token_ids),
            "first_request_usage_prompt_tokens": len(first_token_ids),
            "exact_message_prefix": True,
            "exact_normalized_message_prefix": True,
            "exact_rendered_text_prefix": True,
            "exact_token_id_prefix": True,
        }
        prompt = {
            "id": f"swe-c0m-{first_index:02d}-{instance_id}",
            "text": first_rendered,
            "token_ids": first_token_ids,
            "score_token_ids": score_token_ids,
            "metadata": {
                "kind": "swe_verified_multitask_initial_probe",
                "protocol_sha256": protocol_sha256,
                "lens_outputs_used_for_selection": False,
                "task": copy.deepcopy(dict(c1_task)),
                "checkpoint": copy.deepcopy(C0M_CHECKPOINT),
                "middle_band_layers": list(C0.MIDDLE_BAND_LAYERS),
                "concepts": concepts,
                "capture_match": capture_match,
                "visibility_audit": {
                    "scope": "full_rendered_prompt_text_and_token_ids",
                    "all_retained_forms_hidden": True,
                    "records": visibility,
                },
            },
        }
        prompts.append(prompt)
        matches.append(
            {
                "id": prompt["id"],
                "instance_id": instance_id,
                "paired_c1_prompt_id": c1_prompt_id,
                "first_request_index": first_index,
                "second_request_index": second_index,
                "first_request_sha256": bindings["first_request_sha256"],
                "second_request_sha256": bindings["second_request_sha256"],
                "prompt_sha256": C1.sha256_text(first_rendered),
                "token_ids_sha256": C1.sha256_json(first_token_ids),
                "prompt_token_count": len(first_token_ids),
                "score_token_ids_sha256": C1.sha256_json(score_token_ids),
                "concept_count": len(concepts),
                "foil_count": sum(len(concept["foils"]) for concept in concepts),
                "visibility_record_count": len(visibility),
            }
        )

    summary = {
        "schema_version": 1,
        "kind": "swe_verified_multitask_c0m_materialization",
        "checkpoint": copy.deepcopy(C0M_CHECKPOINT),
        "protocol_sha256": protocol_sha256,
        "c1_prompt_bundle_sha256": c1_prompt_bundle_sha256,
        "capture_manifest_sha256": capture_manifest_sha256,
        "usage_manifest_sha256": usage_manifest_sha256,
        "chat_template_sha256": C1.sha256_text(template),
        "tokenizer_json_sha256": C0.TOKENIZER_JSON_SHA256,
        "lens_outputs_used_for_selection": False,
        "selected_prompt_count": len(prompts),
        "selected_task_count": len(prompts),
        "retained_primary_concept_count": sum(
            len(prompt["metadata"]["concepts"]) for prompt in prompts
        ),
        "retained_foil_count": sum(
            len(concept["foils"])
            for prompt in prompts
            for concept in prompt["metadata"]["concepts"]
        ),
        "prefix_contract": {
            "raw_messages": "exact_prefix_of_paired_c1_request",
            "normalized_messages": "exact_prefix_of_paired_c1_request",
            "rendered_text": "exact_prefix_of_paired_c1_prompt",
            "token_ids": "exact_prefix_of_paired_c1_prompt",
        },
        "visibility_contract": "all_retained_target_and_foil_forms_hidden_in_full_rendered_prompt",
        "prompts": matches,
    }
    return prompts, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--c1-prompts", type=Path, default=DEFAULT_C1_PROMPTS)
    parser.add_argument("--capture-root", type=Path, default=ROOT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    protocol_path = args.protocol.expanduser().resolve(strict=True)
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha256 = C1.sha256_bytes(protocol_bytes)
    require_equal(protocol_sha256, EXPECTED_PROTOCOL_SHA256, "frozen protocol SHA-256")
    protocol = C1.require_mapping(json.loads(protocol_bytes), "frozen protocol")

    c1_path = args.c1_prompts.expanduser().resolve(strict=True)
    c1_sha256 = C1.sha256_file(c1_path)
    require_equal(c1_sha256, EXPECTED_C1_PROMPT_BUNDLE_SHA256, "frozen C1 prompt bundle SHA-256")
    c1_prompts = C1.require_list(json.loads(c1_path.read_bytes()), "frozen C1 prompts")
    require_equal(len(c1_prompts), 8, "frozen C1 prompt count")

    capture_root = args.capture_root.expanduser().resolve(strict=True)
    requests, sources, usage, capture_sha256, usage_sha256 = C1.load_merged_capture(
        capture_root
    )
    template_path = args.template.expanduser().resolve(strict=True)
    require_equal(
        C1.sha256_file(template_path),
        RENDER.EXPECTED_TEMPLATE_SHA256,
        "pinned Qwen chat template SHA-256",
    )
    template = template_path.read_text(encoding="utf-8")
    snapshot = Path(
        args.model_snapshot
        or snapshot_download(
            RENDER.MODEL_REPO, revision=RENDER.MODEL_REVISION, local_files_only=True
        )
    ).expanduser().resolve(strict=True)
    require_equal(snapshot.name, RENDER.MODEL_REVISION, "pinned model snapshot revision")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    RENDER.validate_tokenizer(tokenizer, snapshot)
    _, _, model_pin, _ = C0._validate_protocol_header(protocol)
    C0._validate_model_pin(model_pin, tokenizer, snapshot)

    prompts, summary = build_c0m_bundle(
        protocol,
        protocol_sha256=protocol_sha256,
        c1_prompts=c1_prompts,
        c1_prompt_bundle_sha256=c1_sha256,
        requests=requests,
        request_sources=sources,
        usage_records=usage,
        tokenizer=tokenizer,
        template=template,
        capture_manifest_sha256=capture_sha256,
        usage_manifest_sha256=usage_sha256,
    )
    output = args.output.expanduser().resolve()
    summary_path = (
        args.summary or output.with_name(f"{output.stem}_summary.json")
    ).expanduser().resolve()
    C1.atomic_write_json(output, prompts)
    summary["prompt_bundle_sha256"] = C1.sha256_file(output)
    C1.atomic_write_json(summary_path, summary)
    print(
        f"wrote {output} ({len(prompts)} C0M prompts, "
        f"sha256={summary['prompt_bundle_sha256']})"
    )
    print(f"wrote {summary_path} (sha256={C1.sha256_file(summary_path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
