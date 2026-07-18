#!/usr/bin/env python3
"""Materialize exact-prefix lifecycle probes from complete SWE trajectories."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import materialize_swe_jlens_prompts as RENDER
import materialize_swe_multitask_c1_probes as C1
import materialize_swe_multitask_initial_probes as C0


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIFECYCLE_PROTOCOL = ROOT / "configs/swe_multistage_protocol.json"
DEFAULT_MANIFEST = ROOT / "configs/swe_multistage_trajectory_manifest.json"
DEFAULT_CONCEPT_REGISTRY = ROOT / "configs/swe_multitask_initial_protocol.json"
DEFAULT_IMAGE_REGISTRY = ROOT / "configs/swe_image_digests.json"
DEFAULT_TEMPLATE = ROOT / "configs/qwen3-openai-codex.jinja"
DEFAULT_OUTPUT = ROOT / ".cache/swe_multistage/prompts.json"
MANIFEST_KIND = "swe_verified_multistage_trajectory_manifest"
PROTOCOL_KIND = "swe_verified_multistage_probe_protocol"
PROMPT_KIND = "swe_verified_multistage_probe"
HISTORICAL_IMAGE_LIMITATION = (
    "the generation runner recorded only a mutable :latest reference; the certified "
    "digest was verified for the later official score, not proven for generation"
)
EVIDENCE_ARTIFACTS = (
    "generated_patch",
    "official_patch",
    "official_eval_script",
    "official_test_output",
    "official_run_log",
    "official_score_log",
    "official_report",
    "official_instance_report",
)
CONCEPT_REGISTRY_FIELDS = (
    "id",
    "family",
    "path",
    "target",
    "kind",
    "contrast",
    "forms",
    "foils",
    "sources",
)
EXPECTED_STAGES = (
    (
        "S0",
        "task_start",
        "task_start_request",
        "classify_from_leakage",
    ),
    (
        "S1",
        "first_successful_repository_orientation",
        "first_successful_repository_orientation",
        "classify_from_leakage",
    ),
    (
        "S2",
        "first_successful_oracle_source_body_read",
        "first_successful_oracle_source_body_read",
        "classify_from_leakage",
    ),
    (
        "S3",
        "diagnosis_pre_edit",
        "first_diagnosis_after_source_read_before_edit",
        "classify_from_leakage",
    ),
    (
        "S4",
        "last_filesystem_pre_edit",
        "last_request_before_first_successful_oracle_source_edit",
        "classify_from_leakage",
    ),
    (
        "S5",
        "first_post_edit_prefix",
        "first_successful_oracle_source_edit",
        "explicit_contaminated_control",
    ),
    (
        "S6",
        "first_successful_post_edit_validation",
        "first_successful_validation_after_source_edit",
        "explicit_contaminated_control",
    ),
    (
        "S7",
        "finalization_after_last_successful_validation",
        "terminal_response_after_last_post_edit_successful_validation",
        "explicit_contaminated_control",
    ),
)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def require_sha256(value: Any, label: str) -> str:
    text = require_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def repository_relative_path(value: Any, label: str) -> Path:
    text = require_string(value, label)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be repository-relative")
    return path


def resolve_bound_path(root: Path, value: Any, label: str) -> tuple[Path, Path]:
    relative = repository_relative_path(value, label)
    resolved_root = root.resolve(strict=True)
    resolved = (resolved_root / relative).resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} escapes the artifact root")
    return relative, resolved


def load_hash_ledger(path: Path, *, artifact_root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (\S.*)", line)
        if match is None:
            raise ValueError(f"evidence ledger line {line_number} is malformed")
        digest, relative_text = match.groups()
        if relative_text in entries:
            raise ValueError("evidence ledger contains duplicate paths")
        _, evidence_path = resolve_bound_path(
            artifact_root, relative_text, f"evidence ledger line {line_number} path"
        )
        require_equal(
            C1.sha256_file(evidence_path),
            digest,
            f"evidence ledger file SHA-256 for {relative_text}",
        )
        entries[relative_text] = digest
    if not entries:
        raise ValueError("evidence ledger must not be empty")
    return entries


def compile_regexes(values: Any, label: str) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for index, raw_value in enumerate(C1.require_list(values, label)):
        value = require_string(raw_value, f"{label} regex {index}")
        try:
            patterns.append(re.compile(value))
        except re.error as exc:
            raise ValueError(f"{label} regex {index} is invalid: {exc}") from exc
    if not patterns:
        raise ValueError(f"{label} must declare at least one regex")
    return patterns


def validate_lifecycle_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    require_equal(protocol.get("schema_version"), 1, "lifecycle protocol schema")
    require_equal(protocol.get("kind"), PROTOCOL_KIND, "lifecycle protocol kind")
    require_equal(
        protocol.get("lens_outputs_used_for_selection"),
        False,
        "lifecycle lens-output selection flag",
    )
    stages = C1.require_list(protocol.get("stages"), "lifecycle stages")
    require_equal(len(stages), len(EXPECTED_STAGES), "lifecycle stage count")
    normalized_stages: list[dict[str, Any]] = []
    for index, (raw_stage, expected) in enumerate(zip(stages, EXPECTED_STAGES, strict=True)):
        stage = C1.require_mapping(raw_stage, f"lifecycle stage {index}")
        require_equal(
            set(stage),
            {"id", "name", "selector", "visibility_contract"},
            f"lifecycle stage {index} fields",
        )
        actual = tuple(stage[field] for field in ("id", "name", "selector", "visibility_contract"))
        require_equal(actual, expected, f"lifecycle stage {index}")
        normalized_stages.append(copy.deepcopy(dict(stage)))

    pins = C1.require_mapping(protocol.get("pins"), "lifecycle pins")
    template_pin = C1.require_mapping(pins.get("chat_template"), "chat-template pin")
    model_pin = C1.require_mapping(pins.get("model"), "model pin")
    require_equal(
        template_pin.get("path"),
        str(DEFAULT_TEMPLATE.relative_to(ROOT)),
        "template path",
    )
    require_equal(
        require_sha256(template_pin.get("sha256"), "template SHA-256"),
        RENDER.EXPECTED_TEMPLATE_SHA256,
        "template SHA-256",
    )
    require_equal(model_pin.get("repo_id"), RENDER.MODEL_REPO, "model repository")
    require_equal(model_pin.get("revision"), RENDER.MODEL_REVISION, "model revision")
    require_equal(model_pin.get("served_model"), RENDER.SERVED_MODEL, "served model")
    require_equal(
        require_sha256(model_pin.get("tokenizer_json_sha256"), "tokenizer SHA-256"),
        C0.TOKENIZER_JSON_SHA256,
        "tokenizer SHA-256",
    )

    events = C1.require_mapping(protocol.get("event_contract"), "event contract")
    require_equal(
        set(events),
        {
            "diagnosis",
            "repository_orientation",
            "source_edit",
            "source_read",
            "successful_validation",
        },
        "event-contract fields",
    )
    orientation = C1.require_mapping(events["repository_orientation"], "orientation event")
    require_equal(
        orientation.get("algorithm"),
        "successful_nonmutating_repository_read_or_search_v1",
        "orientation algorithm",
    )
    source_read = C1.require_mapping(events["source_read"], "source-read event")
    require_equal(
        source_read.get("algorithm"),
        "successful_nonmutating_read_referencing_oracle_path_and_body_v1",
        "source-read algorithm",
    )
    declaration_templates = C1.require_list(
        source_read.get("declaration_regex_templates"),
        "source declaration regex templates",
    )
    if not declaration_templates or any(
        not isinstance(value, str) or value.count("{symbol}") != 1
        for value in declaration_templates
    ):
        raise ValueError("source declaration templates must each contain one {symbol}")

    diagnosis = C1.require_mapping(events["diagnosis"], "diagnosis event")
    require_equal(
        diagnosis.get("must_precede_first_source_edit"),
        True,
        "diagnosis pre-edit contract",
    )
    source_edit = C1.require_mapping(events["source_edit"], "source-edit event")
    require_equal(
        source_edit.get("algorithm"),
        "successful_path_targeted_mutation_command_v1",
        "source-edit algorithm",
    )
    require_equal(
        source_edit.get("must_reference_oracle_source_path"),
        True,
        "source-edit oracle-path contract",
    )
    require_equal(
        source_edit.get("requires_successful_tool_result"),
        True,
        "source-edit success contract",
    )
    validation = C1.require_mapping(events["successful_validation"], "validation event")
    require_equal(
        validation.get("requires_successful_tool_result"),
        True,
        "validation success contract",
    )
    return {
        "stages": normalized_stages,
        "diagnosis_regexes": compile_regexes(
            diagnosis.get("assistant_text_regexes"), "diagnosis event"
        ),
        "source_edit_regexes": compile_regexes(
            source_edit.get("command_regexes"), "source-edit event"
        ),
        "validation_regexes": compile_regexes(
            validation.get("command_regexes"), "validation event"
        ),
        "validation_positive_regexes": compile_regexes(
            validation.get("positive_output_regexes"), "validation positive output"
        ),
        "validation_negative_regexes": compile_regexes(
            validation.get("negative_output_regexes"), "validation negative output"
        ),
        "declaration_templates": list(declaration_templates),
    }


def validate_source_locations(oracle: Mapping[str, Any]) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw_location in enumerate(
        C1.require_list(oracle.get("source_locations"), "oracle source locations")
    ):
        location = C1.require_mapping(raw_location, f"oracle source location {index}")
        path = require_string(location.get("path"), f"source location {index} path")
        if path.startswith("/") or ".." in Path(path).parts or path in seen_paths:
            raise ValueError("oracle source paths must be unique repository-relative paths")
        seen_paths.add(path)
        symbols = [
            require_string(value, f"source location {index} symbol")
            for value in C1.require_list(location.get("symbols"), "source symbols")
        ]
        markers = [
            require_string(value, f"source location {index} body marker")
            for value in C1.require_list(location.get("body_markers", []), "body markers")
        ]
        if not symbols and not markers:
            raise ValueError("each oracle source location needs a symbol or body marker")
        if len(symbols) != len(set(symbols)) or len(markers) != len(set(markers)):
            raise ValueError("oracle source symbols and markers must be unique")
        locations.append({"path": path, "symbols": symbols, "body_markers": markers})
    if not locations:
        raise ValueError("oracle must declare at least one relevant source location")
    return locations


def validate_concepts(
    oracle: Mapping[str, Any], tokenizer: Any, source_paths: set[str]
) -> tuple[list[dict[str, Any]], list[int]]:
    concepts: list[dict[str, Any]] = []
    score_ids: list[int] = []
    seen_ids: set[str] = set()
    seen_token_ids: set[int] = set()
    for concept_index, raw_concept in enumerate(
        C1.require_list(oracle.get("concepts"), "oracle concepts")
    ):
        concept = C1.require_mapping(raw_concept, f"oracle concept {concept_index}")
        concept_id = require_string(concept.get("id"), f"concept {concept_index} ID")
        if concept_id in seen_ids:
            raise ValueError("oracle concept IDs must be unique within a trajectory")
        seen_ids.add(concept_id)
        path = require_string(concept.get("path"), f"concept {concept_index} path")
        if path not in source_paths:
            raise ValueError("every oracle concept path must be a declared source location")
        target = require_string(concept.get("target"), f"concept {concept_index} target")
        family = require_string(concept.get("family"), f"concept {concept_index} family")
        forms: list[dict[str, Any]] = []
        for form_index, raw_form in enumerate(
            C1.require_list(concept.get("forms"), "target forms")
        ):
            form = C1.require_mapping(raw_form, f"target form {form_index}")
            _, token_id = C0._validate_form(form, tokenizer, "multistage target")
            if token_id in seen_token_ids:
                raise ValueError("oracle scored token IDs must be globally unique per task")
            seen_token_ids.add(token_id)
            score_ids.append(token_id)
            forms.append(copy.deepcopy(dict(form)))
        if not forms:
            raise ValueError("each oracle target must declare a scored token form")

        foils: list[dict[str, Any]] = []
        seen_foils: set[tuple[str, str]] = set()
        for foil_index, raw_foil in enumerate(
            C1.require_list(concept.get("foils"), "concept foils")
        ):
            foil = C1.require_mapping(raw_foil, f"concept foil {foil_index}")
            identity = (
                require_string(foil.get("task_instance_id"), "foil task instance ID"),
                require_string(foil.get("concept_id"), "foil concept ID"),
            )
            if identity in seen_foils:
                raise ValueError("oracle foil identities must be unique per concept")
            seen_foils.add(identity)
            foil_target = require_string(foil.get("target"), "foil target")
            foil_family = require_string(foil.get("family"), "foil family")
            foil_forms: list[dict[str, Any]] = []
            for form_index, raw_form in enumerate(
                C1.require_list(foil.get("forms"), "foil forms")
            ):
                form = C1.require_mapping(raw_form, f"foil form {form_index}")
                _, token_id = C0._validate_form(form, tokenizer, "multistage foil")
                if token_id in seen_token_ids:
                    raise ValueError("oracle scored token IDs must be globally unique per task")
                seen_token_ids.add(token_id)
                score_ids.append(token_id)
                foil_forms.append(copy.deepcopy(dict(form)))
            if not foil_forms:
                raise ValueError("each oracle foil must declare a scored token form")
            foils.append(
                {
                    **copy.deepcopy(dict(foil)),
                    "task_instance_id": identity[0],
                    "concept_id": identity[1],
                    "target": foil_target,
                    "family": foil_family,
                    "forms": foil_forms,
                }
            )
        if not foils:
            raise ValueError("each oracle target must declare at least one matched foil")
        concepts.append(
            {
                **copy.deepcopy(dict(concept)),
                "id": concept_id,
                "path": path,
                "target": target,
                "family": family,
                "forms": forms,
                "foils": foils,
            }
        )
    if not concepts:
        raise ValueError("oracle must declare at least one target concept")
    return concepts, score_ids


def validate_manifest_header(
    manifest: Mapping[str, Any], *, lifecycle_protocol_sha256: str
) -> list[Mapping[str, Any]]:
    require_equal(manifest.get("schema_version"), 1, "trajectory manifest schema")
    require_equal(manifest.get("kind"), MANIFEST_KIND, "trajectory manifest kind")
    require_equal(
        manifest.get("lens_outputs_used_for_selection"),
        False,
        "trajectory lens-output selection flag",
    )
    for field, expected_path in (
        ("concept_registry", DEFAULT_CONCEPT_REGISTRY),
        ("image_registry", DEFAULT_IMAGE_REGISTRY),
    ):
        binding = C1.require_mapping(manifest.get(field), f"manifest {field}")
        require_equal(
            repository_relative_path(binding.get("path"), f"{field} path").as_posix(),
            expected_path.relative_to(ROOT).as_posix(),
            f"manifest {field} path",
        )
        require_sha256(binding.get("sha256"), f"manifest {field} SHA-256")
    require_equal(
        require_sha256(
            manifest.get("lifecycle_protocol_sha256"), "manifest lifecycle-protocol SHA-256"
        ),
        lifecycle_protocol_sha256,
        "manifest lifecycle-protocol SHA-256",
    )
    trajectories = [
        C1.require_mapping(value, f"trajectory {index}")
        for index, value in enumerate(
            C1.require_list(manifest.get("trajectories"), "manifest trajectories")
        )
    ]
    if not trajectories:
        raise ValueError("trajectory manifest must not be empty")
    seen_instances: set[str] = set()
    for index, trajectory in enumerate(trajectories):
        require_equal(trajectory.get("selection_index"), index, "trajectory selection index")
        instance_id = require_string(trajectory.get("instance_id"), "trajectory instance ID")
        if instance_id in seen_instances:
            raise ValueError("trajectory instance IDs must be unique")
        seen_instances.add(instance_id)
        for field in (
            "repo",
            "base_commit",
            "dataset_path",
            "problem_statement_sha256",
            "dataset_sha256",
            "patch_sha256",
            "test_patch_sha256",
            "proxy_dir",
            "usage_path",
            "official_report_path",
            "official_instance_report_path",
            "expected_official_verdict",
        ):
            require_string(trajectory.get(field), f"trajectory {instance_id} {field}")
        for field in (
            "dataset_sha256",
            "problem_statement_sha256",
            "patch_sha256",
            "test_patch_sha256",
            "request_manifest_sha256",
            "usage_sha256",
            "official_report_sha256",
            "official_instance_report_sha256",
        ):
            require_sha256(trajectory.get(field), f"trajectory {instance_id} {field}")
        require_int(trajectory.get("expected_request_count"), "expected request count", minimum=1)
        require_int(trajectory.get("max_prompt_tokens"), "maximum prompt tokens", minimum=1)
        if trajectory.get("expected_official_verdict") not in {
            "resolved",
            "unresolved",
            "incomplete",
            "error",
        }:
            raise ValueError("expected official verdict is not recognized")
        C1.require_mapping(trajectory.get("oracle"), "trajectory oracle")
        evidence = C1.require_mapping(
            trajectory.get("evidence_binding"), "trajectory evidence binding"
        )
        require_equal(
            set(evidence),
            {
                "manifest_path",
                "manifest_sha256",
                *(
                    field
                    for artifact in EVIDENCE_ARTIFACTS
                    for field in (f"{artifact}_path", f"{artifact}_sha256")
                ),
            },
            "trajectory evidence-binding fields",
        )
        repository_relative_path(evidence.get("manifest_path"), "evidence manifest path")
        require_sha256(evidence.get("manifest_sha256"), "evidence manifest SHA-256")
        for artifact_name in EVIDENCE_ARTIFACTS:
            repository_relative_path(
                evidence.get(f"{artifact_name}_path"),
                f"{artifact_name} path",
            )
            require_sha256(
                evidence.get(f"{artifact_name}_sha256"),
                f"{artifact_name} SHA-256",
            )
        require_equal(
            evidence.get("generated_patch_sha256"),
            evidence.get("official_patch_sha256"),
            "generated/official patch SHA-256",
        )
        image = C1.require_mapping(
            trajectory.get("image_binding"), "trajectory image binding"
        )
        require_equal(
            set(image),
            {
                "architecture",
                "historical_runner_reference",
                "historical_generation_digest_proven",
                "historical_generation_limitation",
            },
            "trajectory image-binding fields",
        )
        require_string(image.get("architecture"), "image architecture")
        require_string(
            image.get("historical_runner_reference"), "historical runner image reference"
        )
        require_equal(
            image.get("historical_generation_digest_proven"),
            False,
            "historical generation digest proof flag",
        )
        require_equal(
            image.get("historical_generation_limitation"),
            HISTORICAL_IMAGE_LIMITATION,
            "historical generation image limitation",
        )
        terminal = C1.require_mapping(
            trajectory.get("terminal_binding"), "trajectory terminal binding"
        )
        require_string(terminal.get("runner_metadata_path"), "runner metadata path")
        require_sha256(terminal.get("runner_metadata_sha256"), "runner metadata SHA-256")
        require_int(terminal.get("expected_final_request_index"), "final request index", minimum=1)
        require_equal(
            terminal.get("expected_finish_reason"), "stop", "terminal finish reason"
        )
        require_int(terminal.get("expected_num_turns"), "terminal turn count", minimum=1)
        require_string(
            terminal.get("expected_dataset_path_suffix"),
            "terminal dataset path suffix",
        )
        require_equal(terminal.get("expected_cli_exit_code"), 0, "terminal CLI exit code")
        require_equal(terminal.get("expected_parsed"), True, "terminal parsed flag")
        require_equal(terminal.get("expected_subtype"), "success", "terminal subtype")
    return trajectories


def request_manifest_sha256(sources: Sequence[Mapping[str, Any]]) -> str:
    return C1.sha256_json(
        [
            {
                "index": source["index"],
                "path": source["path"],
                "sha256": source["sha256"],
            }
            for source in sources
        ]
    )


def load_bound_json(
    artifact_root: Path, binding: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], str]:
    _, path = resolve_bound_path(artifact_root, binding.get("path"), f"{label} path")
    digest = C1.sha256_file(path)
    require_equal(digest, binding.get("sha256"), f"{label} SHA-256")
    value = C1.require_mapping(json.loads(path.read_bytes()), label)
    return copy.deepcopy(dict(value)), digest


def bind_concept_registry(
    registry: Mapping[str, Any],
    *,
    registry_sha256: str,
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    require_equal(registry.get("schema_version"), 1, "concept registry schema")
    require_equal(
        registry.get("kind"), "swe_verified_initial_probe_protocol", "concept registry kind"
    )
    require_equal(
        registry.get("status"),
        "exploratory_development_pilot",
        "concept registry status",
    )
    require_equal(
        registry.get("lens_outputs_used_for_selection"),
        False,
        "concept registry lens-selection flag",
    )
    tasks = [
        C1.require_mapping(value, "concept registry task")
        for value in C1.require_list(registry.get("tasks"), "concept registry tasks")
        if isinstance(value, dict) and value.get("instance_id") == trajectory["instance_id"]
    ]
    if len(tasks) != 1:
        raise ValueError("concept registry must contain exactly one trajectory task")
    task = tasks[0]
    for field in ("instance_id", "repo", "base_commit", "patch_sha256", "test_patch_sha256"):
        require_equal(task.get(field), trajectory.get(field), f"concept registry task {field}")
    registry_concepts = [
        C1.require_mapping(value, "concept registry concept")
        for value in C1.require_list(task.get("concepts"), "concept registry concepts")
    ]
    manifest_concepts = [
        C1.require_mapping(value, "manifest oracle concept")
        for value in C1.require_list(
            C1.require_mapping(trajectory.get("oracle"), "trajectory oracle").get(
                "concepts"
            ),
            "manifest oracle concepts",
        )
    ]
    expected_by_id = {str(concept.get("id")): concept for concept in registry_concepts}
    if len(expected_by_id) != len(registry_concepts):
        raise ValueError("concept registry task has duplicate concept IDs")
    if {str(concept.get("id")) for concept in manifest_concepts} != set(expected_by_id):
        raise ValueError("manifest concept IDs do not exactly match the frozen task registry")
    for concept in manifest_concepts:
        concept_id = str(concept["id"])
        expected = expected_by_id[concept_id]
        require_equal(
            {field: concept.get(field) for field in CONCEPT_REGISTRY_FIELDS},
            {field: expected.get(field) for field in CONCEPT_REGISTRY_FIELDS},
            f"frozen concept/foil registry entry {concept_id}",
        )
    expected_score_ids = C1.require_list(
        task.get("score_token_ids"), "concept registry score token IDs"
    )
    actual_score_ids = sorted(
        form["token_id"]
        for concept in manifest_concepts
        for forms in [
            concept["forms"],
            *(foil["forms"] for foil in concept["foils"]),
        ]
        for form in forms
    )
    require_equal(
        actual_score_ids,
        sorted(expected_score_ids),
        "manifest/frozen-registry score token IDs",
    )
    return {
        "path": DEFAULT_CONCEPT_REGISTRY.relative_to(ROOT).as_posix(),
        "sha256": registry_sha256,
        "task_selection_index": task.get("selection_index"),
        "instance_id": task["instance_id"],
        "concept_ids": [concept["id"] for concept in manifest_concepts],
        "exact_concept_foil_registry_match": True,
    }


def bind_image_evidence(
    registry: Mapping[str, Any],
    *,
    registry_sha256: str,
    trajectory: Mapping[str, Any],
    runner_metadata: Mapping[str, Any],
    evidence_binding: Mapping[str, Any],
) -> dict[str, Any]:
    require_equal(registry.get("schema_version"), 1, "image registry schema")
    images = C1.require_mapping(registry.get("images"), "image registry entries")
    instance_images = C1.require_mapping(
        images.get(trajectory["instance_id"]), "trajectory image registry entry"
    )
    manifest_binding = C1.require_mapping(
        trajectory.get("image_binding"), "trajectory image binding"
    )
    architecture = require_string(
        manifest_binding.get("architecture"), "trajectory image architecture"
    )
    pinned = C1.require_mapping(
        instance_images.get(architecture), "trajectory architecture image pin"
    )
    require_equal(set(pinned), {"reference", "image_id"}, "image registry pin fields")
    reference = require_string(pinned.get("reference"), "certified image reference")
    image_id = require_string(pinned.get("image_id"), "certified image ID")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise ValueError("certified image ID must be a sha256 digest")
    if not reference.endswith(f"@{image_id}"):
        raise ValueError("certified image reference does not end in its image ID")

    historical_reference = require_string(
        manifest_binding.get("historical_runner_reference"),
        "historical runner image reference",
    )
    require_equal(
        runner_metadata.get("image"),
        historical_reference,
        "runner historical image reference",
    )
    expected_score_line = (
        f"verified pinned task image: {historical_reference} ({image_id})"
    )
    require_equal(
        evidence_binding.get("official_score_log_first_line"),
        expected_score_line,
        "official score image proof",
    )
    return {
        "registry_path": DEFAULT_IMAGE_REGISTRY.relative_to(ROOT).as_posix(),
        "registry_sha256": registry_sha256,
        "architecture": architecture,
        "certified_reference": reference,
        "certified_image_id": image_id,
        "historical_runner_reference": historical_reference,
        "generation_digest_proven": False,
        "official_score_digest_proven": True,
        "limitation": HISTORICAL_IMAGE_LIMITATION,
    }


def bind_dataset_oracle(
    dataset: Any, trajectory: Mapping[str, Any]
) -> dict[str, Any]:
    rows = dataset if isinstance(dataset, list) else [dataset]
    matches = [
        C1.require_mapping(row, "dataset row")
        for row in rows
        if isinstance(row, dict) and row.get("instance_id") == trajectory["instance_id"]
    ]
    if len(matches) != 1:
        raise ValueError("dataset must contain exactly one row for the trajectory task")
    row = matches[0]
    for field in ("instance_id", "repo", "base_commit"):
        require_equal(row.get(field), trajectory[field], f"dataset task {field}")
    problem = require_string(row.get("problem_statement"), "dataset problem statement")
    patch = require_string(row.get("patch"), "dataset gold patch")
    test_patch = require_string(row.get("test_patch"), "dataset test patch")
    require_equal(
        C1.sha256_text(problem), trajectory["problem_statement_sha256"], "problem hash"
    )
    require_equal(C1.sha256_text(patch), trajectory["patch_sha256"], "gold-patch hash")
    require_equal(
        C1.sha256_text(test_patch), trajectory["test_patch_sha256"], "test-patch hash"
    )

    sections: dict[str, list[str]] = {}
    current_path: str | None = None
    for line in patch.splitlines():
        match = re.fullmatch(r"diff --git a/(.+) b/(.+)", line)
        if match is not None:
            if match.group(1) != match.group(2):
                raise ValueError("renamed gold files require an explicit oracle protocol extension")
            current_path = match.group(2)
            if current_path in sections:
                raise ValueError("gold patch repeats a file section")
            sections[current_path] = [line]
        elif current_path is not None:
            sections[current_path].append(line)
    oracle = C1.require_mapping(trajectory["oracle"], "trajectory oracle")
    locations = validate_source_locations(oracle)
    for location in locations:
        path = str(location["path"])
        if path not in sections:
            raise ValueError(f"oracle source path {path!r} is absent from the gold patch")
        section_text = "\n".join(sections[path])
        for symbol in location["symbols"]:
            if not C1.surface_present(section_text, str(symbol)):
                raise ValueError(
                    f"oracle source symbol {symbol!r} is absent from its gold patch section"
                )
    for raw_concept in C1.require_list(oracle.get("concepts"), "oracle concepts"):
        concept = C1.require_mapping(raw_concept, "oracle concept")
        path = str(concept["path"])
        added_text = "\n".join(
            line[1:]
            for line in sections[path]
            if line.startswith("+") and not line.startswith("+++")
        )
        if not C1.surface_present(added_text, str(concept["target"])):
            raise ValueError("oracle target is absent from added gold-patch lines")
    return {
        "dataset_sha256": trajectory["dataset_sha256"],
        "instance_id": row["instance_id"],
        "version": row.get("version"),
        "gold_source_paths": sorted(sections),
    }


def load_trajectory_artifacts(
    artifact_root: Path, trajectories: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    artifact_root = artifact_root.resolve(strict=True)
    artifacts: list[dict[str, Any]] = []
    for trajectory in trajectories:
        instance_id = str(trajectory["instance_id"])
        evidence_contract = C1.require_mapping(
            trajectory.get("evidence_binding"), "trajectory evidence binding"
        )
        evidence_manifest_relative, evidence_manifest_path = resolve_bound_path(
            artifact_root,
            evidence_contract.get("manifest_path"),
            f"{instance_id} evidence manifest path",
        )
        evidence_manifest_sha256 = C1.sha256_file(evidence_manifest_path)
        require_equal(
            evidence_manifest_sha256,
            evidence_contract.get("manifest_sha256"),
            f"{instance_id} evidence manifest SHA-256",
        )
        ledger_entries = load_hash_ledger(
            evidence_manifest_path, artifact_root=artifact_root
        )
        evidence_paths: dict[str, Path] = {}
        bound_evidence: dict[str, Any] = {
            "manifest_path": evidence_manifest_relative.as_posix(),
            "manifest_sha256": evidence_manifest_sha256,
            "manifest_entry_count": len(ledger_entries),
        }
        for artifact_name in EVIDENCE_ARTIFACTS:
            relative, path = resolve_bound_path(
                artifact_root,
                evidence_contract.get(f"{artifact_name}_path"),
                f"{instance_id} {artifact_name} path",
            )
            digest = C1.sha256_file(path)
            require_equal(
                digest,
                evidence_contract.get(f"{artifact_name}_sha256"),
                f"{instance_id} {artifact_name} SHA-256",
            )
            require_equal(
                ledger_entries.get(relative.as_posix()),
                digest,
                f"{instance_id} {artifact_name} evidence-ledger entry",
            )
            if path.stat().st_size == 0:
                raise ValueError(f"{instance_id} {artifact_name} evidence is empty")
            evidence_paths[artifact_name] = path
            bound_evidence[f"{artifact_name}_path"] = relative.as_posix()
            bound_evidence[f"{artifact_name}_sha256"] = digest
        generated_patch = evidence_paths["generated_patch"].read_bytes()
        official_patch = evidence_paths["official_patch"].read_bytes()
        require_equal(
            generated_patch,
            official_patch,
            f"{instance_id} generated/official patch bytes",
        )
        bound_evidence["generated_and_official_patch_bytes_equal"] = True
        score_lines = evidence_paths["official_score_log"].read_text(
            encoding="utf-8"
        ).splitlines()
        if not score_lines:
            raise ValueError(f"{instance_id} official score log is empty")
        bound_evidence["official_score_log_first_line"] = score_lines[0]

        dataset_relative, dataset_path = resolve_bound_path(
            artifact_root,
            trajectory.get("dataset_path"),
            f"{instance_id} dataset path",
        )
        dataset_sha256 = C1.sha256_file(dataset_path)
        require_equal(
            dataset_sha256,
            trajectory["dataset_sha256"],
            f"{instance_id} dataset SHA-256",
        )
        require_equal(
            ledger_entries.get(dataset_relative.as_posix()),
            dataset_sha256,
            f"{instance_id} dataset evidence-ledger entry",
        )
        dataset_binding = bind_dataset_oracle(
            json.loads(dataset_path.read_bytes()), trajectory
        )
        proxy_relative, proxy_dir = resolve_bound_path(
            artifact_root,
            trajectory.get("proxy_dir"),
            f"{instance_id} proxy directory",
        )
        if not proxy_dir.is_dir():
            raise ValueError(f"{instance_id} proxy path must be a directory")
        request_paths = sorted(proxy_dir.glob("chat_*.json"))
        expected_count = int(trajectory["expected_request_count"])
        if len(request_paths) != expected_count:
            raise ValueError(f"{instance_id} request count does not match its manifest pin")
        requests: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        for index, path in enumerate(request_paths, start=1):
            require_equal(path.name, f"chat_{index:04d}.json", "contiguous request filename")
            requests.append(dict(C1.require_mapping(json.loads(path.read_bytes()), path.name)))
            source_relative = f"{proxy_relative.as_posix()}/{path.name}"
            source_sha256 = C1.sha256_file(path)
            require_equal(
                ledger_entries.get(source_relative),
                source_sha256,
                f"{instance_id} request {index} evidence-ledger entry",
            )
            sources.append(
                {
                    "index": index,
                    "path": source_relative,
                    "bytes": path.stat().st_size,
                    "sha256": source_sha256,
                }
            )
        require_equal(
            request_manifest_sha256(sources),
            trajectory["request_manifest_sha256"],
            f"{instance_id} request-manifest SHA-256",
        )

        usage_relative, usage_path = resolve_bound_path(
            artifact_root,
            trajectory.get("usage_path"),
            f"{instance_id} usage path",
        )
        usage_sha256 = C1.sha256_file(usage_path)
        require_equal(
            usage_sha256,
            trajectory["usage_sha256"],
            f"{instance_id} usage SHA-256",
        )
        require_equal(
            ledger_entries.get(usage_relative.as_posix()),
            usage_sha256,
            f"{instance_id} usage evidence-ledger entry",
        )
        usage_records = [
            dict(C1.require_mapping(json.loads(line), f"{usage_path.name} line {index}"))
            for index, line in enumerate(
                usage_path.read_text(encoding="utf-8").splitlines(), start=1
            )
            if line.strip()
        ]

        require_equal(
            evidence_contract.get("official_report_path"),
            trajectory.get("official_report_path"),
            f"{instance_id} official report evidence path",
        )
        require_equal(
            evidence_contract.get("official_report_sha256"),
            trajectory.get("official_report_sha256"),
            f"{instance_id} official report evidence SHA-256",
        )
        report_path = evidence_paths["official_report"]
        require_equal(
            C1.sha256_file(report_path),
            trajectory["official_report_sha256"],
            f"{instance_id} official-report SHA-256",
        )
        report = C1.require_mapping(json.loads(report_path.read_bytes()), "official score report")
        require_equal(
            evidence_contract.get("official_instance_report_path"),
            trajectory.get("official_instance_report_path"),
            f"{instance_id} official instance report evidence path",
        )
        require_equal(
            evidence_contract.get("official_instance_report_sha256"),
            trajectory.get("official_instance_report_sha256"),
            f"{instance_id} official instance report evidence SHA-256",
        )
        instance_report_path = evidence_paths["official_instance_report"]
        require_equal(
            C1.sha256_file(instance_report_path),
            trajectory["official_instance_report_sha256"],
            f"{instance_id} official instance-report SHA-256",
        )
        instance_report = C1.require_mapping(
            json.loads(instance_report_path.read_bytes()), "official instance report"
        )
        terminal = C1.require_mapping(
            trajectory["terminal_binding"], "trajectory terminal binding"
        )
        runner_relative, runner_path = resolve_bound_path(
            artifact_root,
            terminal.get("runner_metadata_path"),
            f"{instance_id} runner metadata path",
        )
        runner_sha256 = C1.sha256_file(runner_path)
        require_equal(
            runner_sha256,
            terminal["runner_metadata_sha256"],
            f"{instance_id} runner-metadata SHA-256",
        )
        require_equal(
            ledger_entries.get(runner_relative.as_posix()),
            runner_sha256,
            f"{instance_id} runner-metadata evidence-ledger entry",
        )
        runner_metadata = C1.require_mapping(
            json.loads(runner_path.read_bytes()), "runner metadata"
        )
        artifacts.append(
            {
                "requests": requests,
                "request_sources": sources,
                "usage_records": usage_records,
                "official_report": copy.deepcopy(dict(report)),
                "official_instance_report": copy.deepcopy(dict(instance_report)),
                "runner_metadata": copy.deepcopy(dict(runner_metadata)),
                "dataset_binding": dataset_binding,
                "evidence_binding": bound_evidence,
            }
        )
    return artifacts


def tool_success(result: Mapping[str, Any]) -> bool:
    return (
        result["exit_code"] == 0
        and result["signal"] == 0
        and result["error"] == "(none)"
    )


def text_references_path(text: str, path: str) -> bool:
    normalized = text.replace("\\", "/")
    return path in normalized


def command_mutates_repository_path(command: str, path: str) -> bool:
    normalized = command.replace("\\", "/")
    normalized_path = path.replace("\\", "/").lstrip("/")

    def target_matches(raw_target: str) -> bool:
        target = raw_target.strip().strip("'\"").rstrip(",;)").replace("\\", "/")
        return target == normalized_path or target.endswith(f"/{normalized_path}")

    target_token = r"(?P<target>['\"][^'\"]+['\"]|[^\s;&|]+)"
    for line in normalized.splitlines():
        if re.search(r"(?i)\b(?:cat|echo|printf)\b", line):
            for match in re.finditer(rf"(?<![<>])>>?\s*{target_token}", line):
                if target_matches(match.group("target")):
                    return True
        for match in re.finditer(
            rf"(?i)\btee\b(?:\s+--?[A-Za-z-]+)*\s+{target_token}", line
        ):
            if target_matches(match.group("target")):
                return True
        if re.search(r"(?i)\bsed\b[^\n;|]*\s-i(?:[.A-Za-z0-9_-]*)?\b", line):
            target_match = re.search(rf"{target_token}\s*$", line)
            if target_match is not None and target_matches(target_match.group("target")):
                return True

    open_pattern = re.compile(
        r"\bopen\s*\(\s*(?P<quote>['\"])(?P<target>.*?)(?P=quote)\s*,\s*"
        r"(?P<mode_quote>['\"])(?P<mode>[^'\"]+)(?P=mode_quote)"
    )
    for match in open_pattern.finditer(normalized):
        if any(flag in match.group("mode") for flag in "wax+") and target_matches(
            match.group("target")
        ):
            return True

    write_method_pattern = re.compile(
        r"(?:Path\s*\(\s*)?(?P<quote>['\"])(?P<target>.*?)(?P=quote)\s*\)?"
        r"\s*\.write_(?:text|bytes)\s*\("
    )
    if any(
        target_matches(match.group("target"))
        for match in write_method_pattern.finditer(normalized)
    ):
        return True

    if re.search(r"(?i)\b(?:apply_patch|git\s+apply|patch\s+-p\d+)\b", normalized):
        if re.search(r"(?i)\b(?:--check|--dry-run)\b", normalized):
            return False
        patch_paths = (
            rf"(?m)^\*\*\* (?:Update|Add|Delete) File:\s*(?P<target>\S.*)$",
            rf"(?m)^diff --git a/(?P<target>{re.escape(normalized_path)})\s+b/",
            rf"(?m)^\+\+\+\s+b/(?P<target>{re.escape(normalized_path)})\s*$",
        )
        for pattern in patch_paths:
            for match in re.finditer(pattern, normalized):
                if target_matches(match.group("target")):
                    return True
    return False


def source_body_match(
    output: str,
    location: Mapping[str, Any],
    declaration_templates: Sequence[str],
) -> dict[str, Any] | None:
    candidates: list[tuple[str, re.Match[str]]] = []
    for symbol in location["symbols"]:
        for template in declaration_templates:
            pattern = re.compile(template.replace("{symbol}", re.escape(str(symbol))))
            match = pattern.search(output)
            if match is not None:
                candidates.append((f"symbol:{symbol}", match))
    for marker in location["body_markers"]:
        match = re.search(re.escape(str(marker)), output)
        if match is not None:
            candidates.append((f"marker:{marker}", match))
    for label, match in candidates:
        if any(line.strip() for line in output[match.end() :].splitlines()[1:]):
            return {"path": location["path"], "body_match": label}
    return None


def parse_transition(
    previous_messages: Sequence[Mapping[str, Any]],
    current_messages: Sequence[Mapping[str, Any]],
    *,
    source_locations: Sequence[Mapping[str, Any]],
    event_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if current_messages[: len(previous_messages)] != list(previous_messages):
        raise ValueError("trajectory request messages are not an exact raw prefix")
    extension = current_messages[len(previous_messages) :]
    if not extension:
        raise ValueError("successive trajectory requests must append a response")
    assistant = C1.require_mapping(extension[0], "appended assistant message")
    if assistant.get("role") != "assistant":
        raise ValueError("request extension must begin with an assistant message")
    tool_messages = [C1.require_mapping(value, "appended tool result") for value in extension[1:]]
    if any(message.get("role") != "tool" for message in tool_messages):
        raise ValueError("request extension may only append assistant and tool messages")
    calls = C1.require_list(assistant.get("tool_calls"), "appended assistant tool calls")
    if not calls or len(calls) != len(tool_messages):
        raise ValueError("non-final request extension must pair tool calls and results")

    assistant_text = C1.assistant_channel_text(assistant)
    diagnosis_hits = [
        pattern.pattern
        for pattern in event_contract["diagnosis_regexes"]
        if pattern.search(assistant_text)
    ]
    observations: list[dict[str, Any]] = []
    has_orientation = False
    source_reads: list[dict[str, Any]] = []
    source_edits: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for call_index, (raw_call, message) in enumerate(zip(calls, tool_messages, strict=True)):
        call = C1.require_mapping(raw_call, f"tool call {call_index}")
        function = C1.require_mapping(call.get("function"), f"tool call {call_index} function")
        require_equal(call.get("type"), "function", "tool-call type")
        name = require_string(function.get("name"), "tool-call name")
        require_equal(message.get("tool_call_id"), call.get("id"), "tool-call result ID")
        raw_arguments = require_string(function.get("arguments"), "raw tool-call arguments")
        try:
            arguments = C1.require_mapping(json.loads(raw_arguments), "tool-call arguments")
        except json.JSONDecodeError as exc:
            raise ValueError("tool-call arguments are invalid JSON") from exc
        result_text = C1.flatten_text_content(message.get("content"), "tool result content")
        observation: dict[str, Any] = {
            "call_index": call_index,
            "tool_call_id": call.get("id"),
            "tool_name": name,
            "arguments_sha256": C1.sha256_text(raw_arguments),
            "result_sha256": C1.sha256_text(result_text),
            "successful_repository_orientation": False,
            "oracle_source_body_reads": [],
            "oracle_source_edits": [],
            "successful_validation": False,
        }
        if name == "run_shell_command":
            command = require_string(arguments.get("command"), "shell command")
            result = C1.parse_tool_result(result_text)
            require_equal(result["command"], command, "tool-result shell command")
            success = tool_success(result)
            orientation = C1.successful_repository_observation(command, result)
            observation.update(
                {
                    "command_sha256": C1.sha256_text(command),
                    "exit_code": result["exit_code"],
                    "signal": result["signal"],
                    "successful_repository_orientation": orientation,
                }
            )
            has_orientation = has_orientation or orientation
            if success and C1.is_read_search_command(command):
                for location in source_locations:
                    if not text_references_path(command, str(location["path"])):
                        continue
                    body_match = source_body_match(
                        str(result["output"]),
                        location,
                        event_contract["declaration_templates"],
                    )
                    if body_match is not None:
                        observation["oracle_source_body_reads"].append(body_match)
                        source_reads.append(body_match)
            if success and any(
                pattern.search(command) for pattern in event_contract["source_edit_regexes"]
            ):
                for location in source_locations:
                    if command_mutates_repository_path(command, str(location["path"])):
                        match = {
                            "path": location["path"],
                            "derivation": "successful_path_targeted_mutation_command_v1",
                        }
                        observation["oracle_source_edits"].append(match)
                        source_edits.append(match)
            if success and any(
                pattern.search(command) for pattern in event_contract["validation_regexes"]
            ):
                validation_output = str(result["output"])
                positive_hits = [
                    pattern.pattern
                    for pattern in event_contract["validation_positive_regexes"]
                    if pattern.search(validation_output)
                ]
                negative_hits = [
                    pattern.pattern
                    for pattern in event_contract["validation_negative_regexes"]
                    if pattern.search(validation_output)
                ]
                observation["validation_positive_output_regex_hits"] = positive_hits
                observation["validation_negative_output_regex_hits"] = negative_hits
                if positive_hits and not negative_hits:
                    observation["successful_validation"] = True
                    validations.append({"command_sha256": C1.sha256_text(command)})
        observations.append(observation)
    return {
        "assistant_text_sha256": C1.sha256_text(assistant_text),
        "diagnosis_regex_hits": diagnosis_hits,
        "successful_repository_orientation": has_orientation,
        "oracle_source_body_reads": source_reads,
        "oracle_source_edits": source_edits,
        "successful_validations": validations,
        "observations": observations,
    }


def request_channels(request: Mapping[str, Any]) -> dict[str, str]:
    channels: dict[str, list[str]] = {
        "system": [],
        "user": [],
        "assistant_text": [],
        "tool_call_args": [],
        "tool_outputs": [],
    }
    for message_index, raw_message in enumerate(
        C1.require_list(request.get("messages"), "request messages")
    ):
        message = C1.require_mapping(raw_message, f"request message {message_index}")
        role = message.get("role")
        if role in {"system", "user"}:
            channels[str(role)].append(
                C1.flatten_text_content(message.get("content"), f"{role} message content")
            )
        elif role == "assistant":
            channels["assistant_text"].append(C1.assistant_channel_text(message))
            for raw_call in C1.require_list(message.get("tool_calls", []), "assistant tool calls"):
                call = C1.require_mapping(raw_call, "assistant tool call")
                function = C1.require_mapping(call.get("function"), "assistant tool function")
                arguments = require_string(function.get("arguments"), "tool-call arguments")
                try:
                    decoded = C1.require_mapping(json.loads(arguments), "tool-call arguments")
                except json.JSONDecodeError as exc:
                    raise ValueError("tool-call arguments are invalid JSON") from exc
                channels["tool_call_args"].extend(C1.flatten_string_values(decoded))
        elif role == "tool":
            channels["tool_outputs"].append(
                C1.flatten_text_content(message.get("content"), "tool result content")
            )
        else:
            raise ValueError(f"unsupported request message role {role!r}")
    return {name: "\n".join(values) for name, values in channels.items()}


def transition_event_labels(transition: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    if transition["successful_repository_orientation"]:
        labels.append("inspect")
    if transition["oracle_source_body_reads"]:
        labels.append("read_oracle_source")
    if transition["diagnosis_regex_hits"]:
        labels.append("diagnose")
    if transition["oracle_source_edits"]:
        labels.append("edit")
    if transition["successful_validations"]:
        labels.append("validate")
    return labels or ["other_tool_action"]


def concept_subjects(
    concepts: Sequence[Mapping[str, Any]], instance_id: str
) -> list[dict[str, Any]]:
    subjects: list[dict[str, Any]] = []
    for concept in concepts:
        semantic_aliases = [str(concept["target"])]
        contrast = concept.get("contrast")
        if isinstance(contrast, str) and contrast and contrast not in semantic_aliases:
            semantic_aliases.append(contrast)
        subjects.append(
            {
                "subject": "target",
                "task_instance_id": instance_id,
                "concept_id": concept["id"],
                "family": concept["family"],
                "target": concept["target"],
                "forms": copy.deepcopy(concept["forms"]),
                "semantic_aliases": semantic_aliases,
            }
        )
        for raw_foil in concept["foils"]:
            foil = C1.require_mapping(raw_foil, "concept foil")
            subjects.append(
                {
                    "subject": "foil",
                    "task_instance_id": foil["task_instance_id"],
                    "concept_id": foil["concept_id"],
                    "family": foil["family"],
                    "target": foil["target"],
                    "forms": copy.deepcopy(foil["forms"]),
                    "semantic_aliases": [str(foil["target"])],
                }
            )
    return subjects


IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
IDENTIFIER_SEGMENT_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)


def identifier_segments(value: str) -> tuple[str, ...]:
    return tuple(
        segment.casefold()
        for component in value.strip("_").split("_")
        for segment in IDENTIFIER_SEGMENT_RE.findall(component)
    )


def semantic_identifier_hits(
    text: str, aliases: Sequence[str]
) -> list[dict[str, Any]]:
    requested = [
        (alias, identifier_segments(alias))
        for alias in aliases
        if alias and identifier_segments(alias)
    ]
    counts: dict[tuple[str, str, str], int] = {}
    for match in IDENTIFIER_RE.finditer(text):
        identifier = match.group(0)
        segments = identifier_segments(identifier)
        for alias, alias_segments in requested:
            width = len(alias_segments)
            match_kind = None
            if identifier.casefold().strip("_") == alias.casefold().strip("_"):
                match_kind = "casefold_full_identifier"
            elif any(
                segments[index : index + width] == alias_segments
                for index in range(len(segments) - width + 1)
            ):
                match_kind = "casefold_identifier_segment"
            if match_kind is not None:
                key = (alias, identifier, match_kind)
                counts[key] = counts.get(key, 0) + 1
    return [
        {
            "alias": alias,
            "identifier": identifier,
            "match_kind": match_kind,
            "occurrences": count,
        }
        for (alias, identifier, match_kind), count in sorted(counts.items())
    ]


def semantic_exposure_evidence(
    text: str,
    target: str,
    forms: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
    tokenizer: Any,
) -> dict[str, Any]:
    evidence = C1.exposure_evidence(text, target, forms, tokenizer)
    identifier_hits = semantic_identifier_hits(text, aliases)
    return {
        **evidence,
        "semantic_aliases": list(aliases),
        "casefold_identifier_hits": identifier_hits,
        "semantic_exposed": bool(identifier_hits),
        "exposed": bool(evidence["exposed"] or identifier_hits),
    }


def visibility_audit(
    *,
    concepts: Sequence[Mapping[str, Any]],
    instance_id: str,
    request: Mapping[str, Any],
    rendered: str,
    tokenizer: Any,
    visibility_contract: str,
) -> tuple[list[dict[str, Any]], str]:
    channels = request_channels(request)
    records: list[dict[str, Any]] = []
    any_exposed = False
    for subject in concept_subjects(concepts, instance_id):
        forms = C1.require_list(subject["forms"], "visibility forms")
        aliases = C1.require_list(subject["semantic_aliases"], "semantic aliases")
        channel_evidence = {
            name: semantic_exposure_evidence(
                text,
                str(subject["target"]),
                forms,
                [str(alias) for alias in aliases],
                tokenizer,
            )
            for name, text in channels.items()
        }
        rendered_evidence = semantic_exposure_evidence(
            rendered,
            str(subject["target"]),
            forms,
            [str(alias) for alias in aliases],
            tokenizer,
        )
        channel_exposed = any(value["exposed"] for value in channel_evidence.values())
        if channel_exposed and not rendered_evidence["exposed"]:
            raise ValueError("channel leakage is absent from the canonical rendered prompt")
        surface_exposed = bool(
            rendered_evidence["canonical_identifier_boundary_hit"]
            or rendered_evidence["case_sensitive_compound_identifier_hits"]
            or rendered_evidence["semantic_exposed"]
        )
        token_id_exposed = bool(rendered_evidence["scored_form_token_id_hits"])
        exposed = bool(rendered_evidence["exposed"])
        any_exposed = any_exposed or exposed
        records.append(
            {
                **subject,
                "surface_exposed": surface_exposed,
                "token_id_exposed": token_id_exposed,
                "exposed": exposed,
                "visible_channels": [
                    name for name, evidence in channel_evidence.items() if evidence["exposed"]
                ],
                "channel_evidence": channel_evidence,
                "full_rendered_prompt": rendered_evidence,
            }
        )
    if visibility_contract == "oracle_hidden_required":
        if any_exposed:
            exposed_labels = [
                f"{record['subject']}:{record['target']}" for record in records if record["exposed"]
            ]
            raise ValueError(
                "oracle-hidden stage exposes scored concepts: " + ", ".join(exposed_labels)
            )
        analysis_role = "oracle_hidden"
    elif visibility_contract == "classify_from_leakage":
        analysis_role = "explicit_contaminated_control" if any_exposed else "oracle_hidden"
    elif visibility_contract == "explicit_contaminated_control":
        analysis_role = "explicit_contaminated_control"
    else:
        raise ValueError(f"unknown visibility contract {visibility_contract!r}")
    for record in records:
        if analysis_role == "oracle_hidden":
            record["analysis_role"] = (
                "primary_hidden" if record["subject"] == "target" else "matched_hidden_foil"
            )
        elif record["exposed"]:
            record["analysis_role"] = "explicit_contaminated_control"
        else:
            record["analysis_role"] = "stage_explicit_control_surface_hidden"
    return records, analysis_role


def bind_official_verdict(
    report: Mapping[str, Any],
    *,
    instance_id: str,
    expected_verdict: str,
    report_sha256: str,
) -> dict[str, Any]:
    require_equal(report.get("schema_version"), 2, "official report schema")
    list_fields = (
        "submitted_ids",
        "completed_ids",
        "incomplete_ids",
        "empty_patch_ids",
        "resolved_ids",
        "unresolved_ids",
        "error_ids",
    )
    values: dict[str, list[str]] = {}
    for field in list_fields:
        ids = C1.require_list(report.get(field), f"official report {field}")
        if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(
            set(ids)
        ):
            raise ValueError(f"official report {field} contains invalid IDs")
        values[field] = list(ids)
    count_bindings = {
        "submitted_instances": "submitted_ids",
        "completed_instances": "completed_ids",
        "empty_patch_instances": "empty_patch_ids",
        "resolved_instances": "resolved_ids",
        "unresolved_instances": "unresolved_ids",
        "error_instances": "error_ids",
    }
    for count_field, ids_field in count_bindings.items():
        require_equal(report.get(count_field), len(values[ids_field]), f"official {count_field}")
    if instance_id not in values["submitted_ids"]:
        raise ValueError("trajectory task is absent from the official submitted set")
    statuses = {
        "resolved": instance_id in values["resolved_ids"],
        "unresolved": instance_id in values["unresolved_ids"],
        "incomplete": instance_id in values["incomplete_ids"],
        "error": instance_id in values["error_ids"],
    }
    selected = [name for name, present in statuses.items() if present]
    if len(selected) != 1:
        raise ValueError("official report must assign exactly one primary verdict to the task")
    verdict = selected[0]
    require_equal(verdict, expected_verdict, "official verdict")
    if verdict in {"resolved", "unresolved"} and instance_id not in values["completed_ids"]:
        raise ValueError("resolved/unresolved official task is absent from completed IDs")
    return {
        "report_sha256": report_sha256,
        "verdict": verdict,
        "submitted": True,
        "completed": instance_id in values["completed_ids"],
        "empty_patch": instance_id in values["empty_patch_ids"],
    }


def bind_official_instance_report(
    report: Mapping[str, Any],
    *,
    instance_id: str,
    expected_verdict: str,
    report_sha256: str,
) -> dict[str, Any]:
    if set(report) != {instance_id}:
        raise ValueError("official instance report must bind exactly the trajectory task")
    result = C1.require_mapping(report[instance_id], "official per-instance result")
    for field in ("patch_is_None", "patch_exists", "patch_successfully_applied", "resolved"):
        if not isinstance(result.get(field), bool):
            raise ValueError(f"official per-instance field {field} must be boolean")
    verdict = "resolved" if result["resolved"] else "unresolved"
    if expected_verdict in {"resolved", "unresolved"}:
        require_equal(verdict, expected_verdict, "per-instance official verdict")
    if verdict == "resolved" and (
        result["patch_is_None"]
        or not result["patch_exists"]
        or not result["patch_successfully_applied"]
    ):
        raise ValueError("resolved per-instance report has an invalid patch contract")
    tests = C1.require_mapping(result.get("tests_status"), "official test status")
    expected_categories = {"FAIL_TO_PASS", "PASS_TO_PASS", "FAIL_TO_FAIL", "PASS_TO_FAIL"}
    require_equal(set(tests), expected_categories, "official test-status categories")
    test_counts: dict[str, dict[str, int]] = {}
    for category in sorted(expected_categories):
        status = C1.require_mapping(tests[category], f"official {category} status")
        require_equal(set(status), {"success", "failure"}, f"official {category} fields")
        successes = C1.require_list(status["success"], f"official {category} successes")
        failures = C1.require_list(status["failure"], f"official {category} failures")
        if any(not isinstance(value, str) or not value for value in [*successes, *failures]):
            raise ValueError("official per-instance test names must be nonempty strings")
        test_counts[category] = {"success": len(successes), "failure": len(failures)}
    if verdict == "resolved":
        if test_counts["FAIL_TO_PASS"]["success"] < 1:
            raise ValueError("resolved task lacks a passing FAIL_TO_PASS test")
        if any(counts["failure"] for counts in test_counts.values()):
            raise ValueError("resolved task has official test failures")
    return {
        "report_sha256": report_sha256,
        "verdict": verdict,
        "patch_exists": result["patch_exists"],
        "patch_successfully_applied": result["patch_successfully_applied"],
        "test_counts": test_counts,
    }


def bind_terminal_run(
    runner_metadata: Mapping[str, Any],
    usage_records: Sequence[Mapping[str, Any]],
    *,
    trajectory: Mapping[str, Any],
    terminal_contract: Mapping[str, Any],
) -> dict[str, Any]:
    instance_id = str(trajectory["instance_id"])
    require_equal(runner_metadata.get("instance_id"), instance_id, "runner task instance ID")
    require_equal(runner_metadata.get("repo"), trajectory["repo"], "runner repository")
    require_equal(
        runner_metadata.get("base_commit"), trajectory["base_commit"], "runner base commit"
    )
    require_equal(runner_metadata.get("agent"), "qwen_code", "runner agent")
    require_equal(runner_metadata.get("runtime"), "container", "runner runtime")
    require_equal(runner_metadata.get("eval_mode"), "skip", "generation evaluation mode")
    dataset_name = require_string(runner_metadata.get("dataset_name"), "runner dataset name")
    dataset_suffix = require_string(
        terminal_contract.get("expected_dataset_path_suffix"),
        "runner dataset path suffix",
    ).replace("\\", "/")
    normalized_dataset_name = dataset_name.replace("\\", "/")
    if not (
        normalized_dataset_name == dataset_suffix
        or normalized_dataset_name.endswith(f"/{dataset_suffix}")
    ):
        raise ValueError("runner dataset name does not match the pinned dataset suffix")
    image_contract = C1.require_mapping(
        trajectory.get("image_binding"), "trajectory image binding"
    )
    require_equal(
        runner_metadata.get("image"),
        image_contract.get("historical_runner_reference"),
        "runner historical image reference",
    )
    qwen = C1.require_mapping(runner_metadata.get("qwen"), "runner Qwen metadata")
    bindings = {
        "num_turns": terminal_contract["expected_num_turns"],
        "exit_code": terminal_contract["expected_cli_exit_code"],
        "parsed": terminal_contract["expected_parsed"],
        "subtype": terminal_contract["expected_subtype"],
    }
    for field, expected in bindings.items():
        require_equal(qwen.get(field), expected, f"runner Qwen {field}")
    final_index = int(terminal_contract["expected_final_request_index"])
    require_equal(final_index, len(usage_records), "terminal final request index")
    final_usage = C1.require_mapping(usage_records[-1], "terminal usage record")
    require_equal(final_usage.get("idx"), final_index, "terminal usage request index")
    require_equal(
        final_usage.get("finish_reason"),
        terminal_contract["expected_finish_reason"],
        "terminal usage finish reason",
    )
    return {
        "runner_metadata_sha256": terminal_contract["runner_metadata_sha256"],
        "final_request_index": final_index,
        "finish_reason": final_usage["finish_reason"],
        "num_turns": qwen["num_turns"],
        "cli_exit_code": qwen["exit_code"],
        "parsed": qwen["parsed"],
        "subtype": qwen["subtype"],
        "repo": runner_metadata["repo"],
        "base_commit": runner_metadata["base_commit"],
        "dataset_name": dataset_name,
        "dataset_path_suffix": dataset_suffix,
        "agent": runner_metadata["agent"],
        "runtime": runner_metadata["runtime"],
        "generation_eval_mode": runner_metadata["eval_mode"],
        "historical_image_reference": runner_metadata["image"],
    }


def select_stages(
    *,
    stages: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any] | None],
    usage_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    orientations = [
        offset + 1
        for offset, transition in enumerate(transitions)
        if offset > 0 and transition and transition["successful_repository_orientation"]
    ]
    source_reads = [
        offset + 1
        for offset, transition in enumerate(transitions)
        if offset > 0 and transition and transition["oracle_source_body_reads"]
    ]
    source_edits = [
        offset + 1
        for offset, transition in enumerate(transitions)
        if offset > 0 and transition and transition["oracle_source_edits"]
    ]
    validations = [
        offset + 1
        for offset, transition in enumerate(transitions)
        if offset > 0 and transition and transition["successful_validations"]
    ]
    first_edit = min(source_edits) if source_edits else None
    selections: list[dict[str, Any]] = []
    chosen: dict[str, int | None] = {}
    for stage in stages:
        stage_id = str(stage["id"])
        selector = stage["selector"]
        request_index: int | None = None
        evidence_request_index: int | None = None
        missing_reason: str | None = None
        if selector == "task_start_request":
            request_index = 1
        elif selector == "first_successful_repository_orientation":
            if orientations:
                request_index = min(orientations)
                evidence_request_index = request_index
            else:
                missing_reason = "no_successful_repository_orientation"
        elif selector == "first_successful_oracle_source_body_read":
            candidates = [
                index
                for index in source_reads
                if chosen.get("S1") is not None and index >= int(chosen["S1"])
            ]
            if chosen.get("S1") is None:
                missing_reason = "prerequisite_S1_missing"
            elif candidates:
                request_index = min(candidates)
                evidence_request_index = request_index
            else:
                missing_reason = "no_successful_oracle_source_body_read"
        elif selector == "first_diagnosis_after_source_read_before_edit":
            if chosen.get("S2") is None:
                missing_reason = "prerequisite_S2_missing"
            else:
                upper = first_edit if first_edit is not None else len(transitions)
                candidates = [
                    offset + 1
                    for offset, transition in enumerate(transitions)
                    if offset + 1 > int(chosen["S2"])
                    and offset + 1 < upper
                    and transition
                    and transition["diagnosis_regex_hits"]
                    and not transition["oracle_source_edits"]
                ]
                if candidates:
                    request_index = min(candidates)
                    evidence_request_index = request_index
                else:
                    missing_reason = "no_diagnosis_after_source_read_before_edit"
        elif selector == "last_request_before_first_successful_oracle_source_edit":
            if chosen.get("S2") is None:
                missing_reason = "prerequisite_S2_missing"
            elif source_edits:
                first_source_edit = min(source_edits)
                if first_source_edit <= 1:
                    missing_reason = "source_edit_has_no_preceding_request"
                elif first_source_edit <= int(chosen["S2"]):
                    missing_reason = "source_edit_does_not_follow_S2"
                else:
                    request_index = first_source_edit - 1
                    evidence_request_index = first_source_edit
            else:
                missing_reason = "no_successful_oracle_source_edit"
        elif selector == "first_successful_oracle_source_edit":
            if chosen.get("S4") is None:
                missing_reason = "prerequisite_S4_missing"
            elif source_edits:
                request_index = min(source_edits)
                evidence_request_index = request_index
                require_equal(
                    request_index,
                    int(chosen["S4"]) + 1,
                    "S4/S5 source-edit boundary",
                )
            else:
                missing_reason = "no_successful_oracle_source_edit"
        elif selector == "first_successful_validation_after_source_edit":
            if chosen.get("S5") is None:
                missing_reason = "prerequisite_S5_missing"
            else:
                post_edit_validations = [
                    index for index in validations if index > int(chosen["S5"])
                ]
                if not post_edit_validations:
                    missing_reason = "no_post_edit_successful_validation"
                else:
                    request_index = min(post_edit_validations)
                    evidence_request_index = request_index
        elif selector == "terminal_response_after_last_post_edit_successful_validation":
            if chosen.get("S6") is None:
                missing_reason = "prerequisite_S6_missing"
            else:
                post_edit_validations = [
                    index for index in validations if index >= int(chosen["S6"])
                ]
                if not post_edit_validations:
                    missing_reason = "no_post_edit_successful_validation"
                else:
                    last_validation = max(post_edit_validations)
                    candidates = [
                        index
                        for index, record in enumerate(usage_records, start=1)
                        if index >= last_validation and record.get("finish_reason") == "stop"
                    ]
                    if candidates:
                        request_index = min(candidates)
                        evidence_request_index = last_validation
                    else:
                        missing_reason = "no_terminal_response_after_last_successful_validation"
        else:
            raise ValueError(f"unsupported stage selector {selector!r}")
        prior_available_indices = [
            int(value) for value in chosen.values() if value is not None
        ]
        if (
            request_index is not None
            and prior_available_indices
            and request_index < prior_available_indices[-1]
        ):
            raise ValueError(
                f"stage {stage_id} request index regresses behind the prior available stage"
            )
        chosen[stage_id] = request_index
        selections.append(
            {
                "stage": copy.deepcopy(dict(stage)),
                "status": "available" if request_index is not None else "missing",
                "request_index": request_index,
                "evidence_request_index": evidence_request_index,
                "missing_reason": missing_reason,
            }
        )
    return selections


def validate_request_sequence(
    *,
    trajectory: Mapping[str, Any],
    artifact: Mapping[str, Any],
    tokenizer: Any,
    template: str,
    source_locations: Sequence[Mapping[str, Any]],
    event_contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any] | None]]:
    requests = C1.require_list(artifact.get("requests"), "trajectory requests")
    sources = C1.require_list(artifact.get("request_sources"), "request sources")
    usage_records = C1.require_list(artifact.get("usage_records"), "usage records")
    expected_count = int(trajectory["expected_request_count"])
    if len(requests) != expected_count or len(sources) != expected_count:
        raise ValueError("trajectory requests and source bindings do not match expected count")
    if len(usage_records) != expected_count:
        raise ValueError("usage ledger does not exactly cover trajectory requests")
    rendered_records: list[dict[str, Any]] = []
    transitions: list[dict[str, Any] | None] = [None]
    previous_raw_messages: list[Mapping[str, Any]] = []
    previous_normalized_messages: list[dict[str, Any]] = []
    previous_rendered = ""
    previous_token_ids: list[int] = []
    first_request: Mapping[str, Any] | None = None
    for offset, (raw_request, raw_source, raw_usage) in enumerate(
        zip(requests, sources, usage_records, strict=True), start=1
    ):
        request = C1.require_mapping(raw_request, f"request {offset}")
        source = C1.require_mapping(raw_source, f"request source {offset}")
        usage_record = C1.require_mapping(raw_usage, f"usage record {offset}")
        require_equal(source.get("index"), offset, "request source index")
        require_sha256(source.get("sha256"), "raw request SHA-256")
        C1.validate_capture_request(request, request_index=offset)
        messages = [
            C1.require_mapping(value, f"request {offset} message")
            for value in C1.require_list(request.get("messages"), "request messages")
        ]
        if offset == 1:
            if [message.get("role") for message in messages] != ["system", "user"]:
                raise ValueError("task-start request must contain exactly system and user messages")
            if str(trajectory["instance_id"]) not in json.dumps(
                messages, sort_keys=True, ensure_ascii=False
            ):
                raise ValueError("task-start request does not bind its trajectory instance ID")
            first_request = request
        else:
            assert first_request is not None
            require_equal(request.get("model"), first_request.get("model"), "trajectory model")
            require_equal(request.get("tools"), first_request.get("tools"), "trajectory tools")
            transitions.append(
                parse_transition(
                    previous_raw_messages,
                    messages,
                    source_locations=source_locations,
                    event_contract=event_contract,
                )
            )
        rendered, token_ids, normalized_count, normalized_messages = C1.render_request(
            tokenizer, request=request, template=template
        )
        usage = C1.require_mapping(usage_record.get("usage"), "usage values")
        require_equal(usage_record.get("idx"), offset, "usage request index")
        require_equal(usage.get("prompt_tokens"), len(token_ids), "usage prompt token count")
        if len(token_ids) > int(trajectory["max_prompt_tokens"]):
            raise ValueError(f"request {offset} exceeds the manifest context limit")
        finish_reason = usage_record.get("finish_reason")
        if finish_reason not in {"tool_calls", "stop", "length"}:
            raise ValueError("usage finish reason is not recognized")
        if offset < expected_count and finish_reason != "tool_calls":
            raise ValueError("every preterminal usage record must finish with tool_calls")
        if offset > 1:
            if (
                normalized_messages[: len(previous_normalized_messages)]
                != previous_normalized_messages
            ):
                raise ValueError("normalized request messages are not an exact prefix")
            if not rendered.startswith(previous_rendered):
                raise ValueError("canonical rendered requests are not exact text prefixes")
            if token_ids[: len(previous_token_ids)] != previous_token_ids:
                raise ValueError("canonical request token IDs are not exact prefixes")
        rendered_records.append(
            {
                "request": request,
                "source": source,
                "usage_record": usage_record,
                "text": rendered,
                "token_ids": token_ids,
                "normalized_count": normalized_count,
                "normalized_messages": normalized_messages,
            }
        )
        previous_raw_messages = messages
        previous_normalized_messages = normalized_messages
        previous_rendered = rendered
        previous_token_ids = token_ids
    return rendered_records, transitions


def build_multistage_bundle(
    lifecycle_protocol: Mapping[str, Any],
    *,
    lifecycle_protocol_sha256: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    concept_registry: Mapping[str, Any],
    concept_registry_sha256: str,
    image_registry: Mapping[str, Any],
    image_registry_sha256: str,
    artifacts: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    template: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    compiled_protocol = validate_lifecycle_protocol(lifecycle_protocol)
    trajectories = validate_manifest_header(
        manifest, lifecycle_protocol_sha256=lifecycle_protocol_sha256
    )
    concept_registry_contract = C1.require_mapping(
        manifest.get("concept_registry"), "manifest concept registry"
    )
    image_registry_contract = C1.require_mapping(
        manifest.get("image_registry"), "manifest image registry"
    )
    require_equal(
        concept_registry_sha256,
        concept_registry_contract.get("sha256"),
        "frozen concept registry SHA-256",
    )
    require_equal(
        image_registry_sha256,
        image_registry_contract.get("sha256"),
        "image registry SHA-256",
    )
    if len(artifacts) != len(trajectories):
        raise ValueError("trajectory artifact count does not match the manifest")
    prompts: list[dict[str, Any]] = []
    task_audits: list[dict[str, Any]] = []
    for trajectory_index, (trajectory, raw_artifact) in enumerate(
        zip(trajectories, artifacts, strict=True)
    ):
        artifact = C1.require_mapping(raw_artifact, f"trajectory artifact {trajectory_index}")
        oracle = C1.require_mapping(trajectory.get("oracle"), "trajectory oracle")
        concept_registry_binding = bind_concept_registry(
            concept_registry,
            registry_sha256=concept_registry_sha256,
            trajectory=trajectory,
        )
        source_locations = validate_source_locations(oracle)
        concepts, score_token_ids = validate_concepts(
            oracle, tokenizer, {str(location["path"]) for location in source_locations}
        )
        dataset_binding = C1.require_mapping(
            artifact.get("dataset_binding"), "dataset binding"
        )
        require_equal(
            dataset_binding.get("dataset_sha256"),
            trajectory["dataset_sha256"],
            "artifact dataset SHA-256",
        )
        require_equal(
            dataset_binding.get("instance_id"), trajectory["instance_id"], "artifact dataset task"
        )
        gold_paths = C1.require_list(
            dataset_binding.get("gold_source_paths"), "dataset gold source paths"
        )
        if not {location["path"] for location in source_locations}.issubset(set(gold_paths)):
            raise ValueError("oracle source paths are absent from the dataset binding")
        evidence_contract = C1.require_mapping(
            trajectory.get("evidence_binding"), "trajectory evidence binding"
        )
        evidence_binding = C1.require_mapping(
            artifact.get("evidence_binding"), "artifact evidence binding"
        )
        for field in ("manifest_path", "manifest_sha256"):
            require_equal(
                evidence_binding.get(field),
                evidence_contract.get(field),
                f"artifact evidence {field}",
            )
        require_int(
            evidence_binding.get("manifest_entry_count"),
            "artifact evidence manifest entry count",
            minimum=1,
        )
        for artifact_name in EVIDENCE_ARTIFACTS:
            for suffix in ("path", "sha256"):
                field = f"{artifact_name}_{suffix}"
                require_equal(
                    evidence_binding.get(field),
                    evidence_contract.get(field),
                    f"artifact evidence {field}",
                )
        require_equal(
            evidence_binding.get("generated_and_official_patch_bytes_equal"),
            True,
            "generated/official patch byte equality",
        )
        require_string(
            evidence_binding.get("official_score_log_first_line"),
            "official score image-proof line",
        )
        official_report = C1.require_mapping(
            artifact.get("official_report"), "official report"
        )
        verdict = bind_official_verdict(
            official_report,
            instance_id=str(trajectory["instance_id"]),
            expected_verdict=str(trajectory["expected_official_verdict"]),
            report_sha256=str(trajectory["official_report_sha256"]),
        )
        instance_report = C1.require_mapping(
            artifact.get("official_instance_report"), "official instance report"
        )
        instance_verdict = bind_official_instance_report(
            instance_report,
            instance_id=str(trajectory["instance_id"]),
            expected_verdict=str(trajectory["expected_official_verdict"]),
            report_sha256=str(trajectory["official_instance_report_sha256"]),
        )
        require_equal(
            instance_verdict["verdict"], verdict["verdict"], "aggregate/instance verdict"
        )
        runner_metadata = C1.require_mapping(
            artifact.get("runner_metadata"), "runner metadata"
        )
        image_binding = bind_image_evidence(
            image_registry,
            registry_sha256=image_registry_sha256,
            trajectory=trajectory,
            runner_metadata=runner_metadata,
            evidence_binding=evidence_binding,
        )
        rendered_records, transitions = validate_request_sequence(
            trajectory=trajectory,
            artifact=artifact,
            tokenizer=tokenizer,
            template=template,
            source_locations=source_locations,
            event_contract=compiled_protocol,
        )
        terminal_run = bind_terminal_run(
            runner_metadata,
            C1.require_list(artifact.get("usage_records"), "usage records"),
            trajectory=trajectory,
            terminal_contract=C1.require_mapping(
                trajectory["terminal_binding"], "terminal binding"
            ),
        )
        selections = select_stages(
            stages=compiled_protocol["stages"],
            transitions=transitions,
            usage_records=artifact["usage_records"],
        )
        stage_audits: list[dict[str, Any]] = []
        previous_available: dict[str, Any] | None = None
        for selection in selections:
            stage = selection["stage"]
            if selection["status"] == "missing":
                stage_audits.append(copy.deepcopy(selection))
                continue
            request_index = int(selection["request_index"])
            record = rendered_records[request_index - 1]
            visibility, analysis_role = visibility_audit(
                concepts=concepts,
                instance_id=str(trajectory["instance_id"]),
                request=record["request"],
                rendered=record["text"],
                tokenizer=tokenizer,
                visibility_contract=str(stage["visibility_contract"]),
            )
            prefix_binding: dict[str, Any] | None = None
            if previous_available is not None:
                previous_record = previous_available["record"]
                if request_index < int(previous_available["request_index"]):
                    raise ValueError("selected lifecycle stages regress in request chronology")
                previous_messages = C1.require_list(
                    previous_record["request"].get("messages"),
                    "prior selected-stage request messages",
                )
                current_messages = C1.require_list(
                    record["request"].get("messages"),
                    "current selected-stage request messages",
                )
                raw_message_prefix = (
                    current_messages[: len(previous_messages)] == previous_messages
                )
                previous_normalized = previous_record["normalized_messages"]
                current_normalized = record["normalized_messages"]
                normalized_message_prefix = (
                    current_normalized[: len(previous_normalized)] == previous_normalized
                )
                rendered_text_prefix = record["text"].startswith(previous_record["text"])
                token_id_prefix = (
                    record["token_ids"][: len(previous_record["token_ids"])]
                    == previous_record["token_ids"]
                )
                if not all(
                    (
                        raw_message_prefix,
                        normalized_message_prefix,
                        rendered_text_prefix,
                        token_id_prefix,
                    )
                ):
                    raise ValueError("selected lifecycle stages are not exact prompt prefixes")
                prefix_binding = {
                    "prior_stage_id": previous_available["stage_id"],
                    "raw_message_prefix": raw_message_prefix,
                    "normalized_message_prefix": normalized_message_prefix,
                    "rendered_text_prefix": rendered_text_prefix,
                    "token_id_prefix": token_id_prefix,
                    "prior_request_index": previous_available["request_index"],
                    "prior_rendered_bytes": len(previous_record["text"].encode("utf-8")),
                    "prior_token_count": len(previous_record["token_ids"]),
                }
            source = record["source"]
            usage_record = record["usage_record"]
            included_transition = transitions[request_index - 1]
            evidence_request_index = selection["evidence_request_index"]
            selection_transition = (
                transitions[int(evidence_request_index) - 1]
                if evidence_request_index is not None
                else None
            )
            if request_index < len(rendered_records):
                next_transition = transitions[request_index]
                if next_transition is None:
                    raise ValueError("next completion transition is unexpectedly absent")
                next_source = rendered_records[request_index]["source"]
                next_completion = {
                    "completion_index": request_index,
                    "materialized_in_request_index": request_index + 1,
                    "materialized_in_raw_request_sha256": next_source["sha256"],
                    "event_labels": transition_event_labels(next_transition),
                    "terminal_response": False,
                    "transition_sha256": C1.sha256_json(next_transition),
                    "transition": copy.deepcopy(next_transition),
                }
            else:
                finish_reason = usage_record.get("finish_reason")
                next_completion = {
                    "completion_index": request_index,
                    "materialized_in_request_index": None,
                    "materialized_in_raw_request_sha256": None,
                    "event_labels": (
                        ["finalize"] if finish_reason == "stop" else ["unobserved_completion"]
                    ),
                    "terminal_response": finish_reason == "stop",
                    "transition_sha256": None,
                    "transition": None,
                    "usage_finish_reason": finish_reason,
                    "usage_record_sha256": C1.sha256_json(usage_record),
                }
            prompt_id = (
                f"swe-{stage['id'].lower()}-{trajectory_index:03d}-"
                f"{trajectory['instance_id']}"
            )
            task_metadata = {
                field: copy.deepcopy(trajectory[field])
                for field in (
                    "instance_id",
                    "repo",
                    "base_commit",
                    "problem_statement_sha256",
                    "patch_sha256",
                    "test_patch_sha256",
                )
            }
            prompt = {
                "id": prompt_id,
                "text": record["text"],
                "token_ids": record["token_ids"],
                "score_token_ids": copy.deepcopy(score_token_ids),
                "metadata": {
                    "kind": PROMPT_KIND,
                    "lifecycle_protocol_sha256": lifecycle_protocol_sha256,
                    "trajectory_manifest_sha256": manifest_sha256,
                    "lens_outputs_used_for_selection": False,
                    "task": task_metadata,
                    "dataset_binding": copy.deepcopy(dict(dataset_binding)),
                    "concept_registry_binding": copy.deepcopy(concept_registry_binding),
                    "image_binding": copy.deepcopy(image_binding),
                    "stage": copy.deepcopy(stage),
                    "analysis_role": analysis_role,
                    "concepts": copy.deepcopy(concepts),
                    "visibility_audit": {
                        "scope": "all_request_channels_and_full_canonical_render",
                        "records": visibility,
                    },
                    "provenance": {
                        "raw_request_index": request_index,
                        "raw_request_path": source["path"],
                        "raw_request_sha256": source["sha256"],
                        "rendered_prompt_sha256": C1.sha256_text(record["text"]),
                        "token_ids_sha256": C1.sha256_json(record["token_ids"]),
                        "prompt_token_count": len(record["token_ids"]),
                        "normalized_messages_sha256": C1.sha256_json(
                            record["normalized_messages"]
                        ),
                        "normalized_string_tool_call_arguments": record["normalized_count"],
                        "usage_file_sha256": trajectory["usage_sha256"],
                        "usage_record_sha256": C1.sha256_json(usage_record),
                        "usage": copy.deepcopy(usage_record),
                        "evidence_binding": copy.deepcopy(dict(evidence_binding)),
                        "official_verdict": copy.deepcopy(verdict),
                        "official_instance_verdict": copy.deepcopy(instance_verdict),
                        "terminal_run": copy.deepcopy(terminal_run),
                        "request_boundary": {
                            "semantics": "chat_N_is_the_exact_prefix_before_completion_N",
                            "prompt_request_index": request_index,
                            "prompt_precedes_completion_index": request_index,
                            "latest_included_completion_index": request_index - 1,
                            "selection_evidence_request_index": evidence_request_index,
                            "selection_event_completion_index": (
                                int(evidence_request_index) - 1
                                if evidence_request_index is not None
                                else None
                            ),
                        },
                        "latest_included_transition": copy.deepcopy(included_transition),
                        "selection_event": copy.deepcopy(selection_transition),
                        "next_completion_transition": {
                            "contract": (
                                "completion_N_is_materialized_by_chat_N_plus_1_or_bound_by_"
                                "terminal_usage"
                            ),
                            "used_only_for_declared_stage_selection_or_action_analysis": True,
                            **next_completion,
                        },
                        "exact_prefix_from_prior_available_stage": prefix_binding,
                    },
                },
            }
            prompts.append(prompt)
            stage_audits.append(
                {
                    **copy.deepcopy(selection),
                    "prompt_id": prompt_id,
                    "analysis_role": analysis_role,
                    "raw_request_sha256": source["sha256"],
                    "rendered_prompt_sha256": C1.sha256_text(record["text"]),
                    "prompt_token_count": len(record["token_ids"]),
                    "official_verdict": verdict["verdict"],
                }
            )
            previous_available = {
                "stage_id": stage["id"],
                "request_index": request_index,
                "record": record,
            }
        task_audits.append(
            {
                "selection_index": trajectory_index,
                "instance_id": trajectory["instance_id"],
                "request_count": len(rendered_records),
                "official_verdict": verdict,
                "official_instance_verdict": instance_verdict,
                "terminal_run": terminal_run,
                "concept_registry_binding": concept_registry_binding,
                "image_binding": image_binding,
                "evidence_binding": copy.deepcopy(dict(evidence_binding)),
                "source_locations": copy.deepcopy(source_locations),
                "stage_audits": stage_audits,
                "available_stage_count": sum(
                    audit["status"] == "available" for audit in stage_audits
                ),
                "missing_stage_count": sum(audit["status"] == "missing" for audit in stage_audits),
            }
        )
    summary = {
        "schema_version": 1,
        "kind": "swe_verified_multistage_probe_materialization",
        "lifecycle_protocol_sha256": lifecycle_protocol_sha256,
        "trajectory_manifest_sha256": manifest_sha256,
        "concept_registry_sha256": concept_registry_sha256,
        "image_registry_sha256": image_registry_sha256,
        "chat_template_sha256": C1.sha256_text(template),
        "tokenizer_json_sha256": C0.TOKENIZER_JSON_SHA256,
        "lens_outputs_used_for_selection": False,
        "trajectory_count": len(trajectories),
        "prompt_count": len(prompts),
        "available_stage_count": sum(
            task["available_stage_count"] for task in task_audits
        ),
        "missing_stage_count": sum(task["missing_stage_count"] for task in task_audits),
        "hidden_prompt_count": sum(
            prompt["metadata"]["analysis_role"] == "oracle_hidden" for prompt in prompts
        ),
        "explicit_control_prompt_count": sum(
            prompt["metadata"]["analysis_role"] == "explicit_contaminated_control"
            for prompt in prompts
        ),
        "stage_contract": copy.deepcopy(compiled_protocol["stages"]),
        "task_audits": task_audits,
        "prompts": [
            {
                "id": prompt["id"],
                "instance_id": prompt["metadata"]["task"]["instance_id"],
                "stage_id": prompt["metadata"]["stage"]["id"],
                "analysis_role": prompt["metadata"]["analysis_role"],
                "prompt_sha256": C1.sha256_text(prompt["text"]),
                "token_ids_sha256": C1.sha256_json(prompt["token_ids"]),
                "prompt_token_count": len(prompt["token_ids"]),
                "score_token_ids_sha256": C1.sha256_json(prompt["score_token_ids"]),
            }
            for prompt in prompts
        ],
    }
    return prompts, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--lifecycle-protocol", type=Path, default=DEFAULT_LIFECYCLE_PROTOCOL)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    lifecycle_path = args.lifecycle_protocol.expanduser().resolve(strict=True)
    lifecycle_bytes = lifecycle_path.read_bytes()
    lifecycle_sha256 = C1.sha256_bytes(lifecycle_bytes)
    lifecycle = C1.require_mapping(json.loads(lifecycle_bytes), "lifecycle protocol")
    validate_lifecycle_protocol(lifecycle)

    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = C1.sha256_bytes(manifest_bytes)
    manifest = C1.require_mapping(json.loads(manifest_bytes), "trajectory manifest")
    trajectories = validate_manifest_header(
        manifest, lifecycle_protocol_sha256=lifecycle_sha256
    )
    artifact_root = args.artifact_root.expanduser().resolve(strict=True)
    concept_registry, concept_registry_sha256 = load_bound_json(
        artifact_root,
        C1.require_mapping(manifest.get("concept_registry"), "manifest concept registry"),
        label="frozen concept registry",
    )
    image_registry, image_registry_sha256 = load_bound_json(
        artifact_root,
        C1.require_mapping(manifest.get("image_registry"), "manifest image registry"),
        label="image registry",
    )
    artifacts = load_trajectory_artifacts(artifact_root, trajectories)

    template_path = args.template.expanduser().resolve(strict=True)
    require_equal(
        C1.sha256_file(template_path),
        RENDER.EXPECTED_TEMPLATE_SHA256,
        "pinned chat-template SHA-256",
    )
    template = template_path.read_text(encoding="utf-8")
    snapshot = Path(
        args.model_snapshot
        or snapshot_download(
            RENDER.MODEL_REPO,
            revision=RENDER.MODEL_REVISION,
            local_files_only=True,
        )
    ).expanduser().resolve(strict=True)
    require_equal(snapshot.name, RENDER.MODEL_REVISION, "pinned model snapshot revision")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    RENDER.validate_tokenizer(tokenizer, snapshot)
    require_equal(
        C1.sha256_file(snapshot / "tokenizer.json"),
        C0.TOKENIZER_JSON_SHA256,
        "pinned tokenizer SHA-256",
    )

    prompts, summary = build_multistage_bundle(
        lifecycle,
        lifecycle_protocol_sha256=lifecycle_sha256,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        concept_registry=concept_registry,
        concept_registry_sha256=concept_registry_sha256,
        image_registry=image_registry,
        image_registry_sha256=image_registry_sha256,
        artifacts=artifacts,
        tokenizer=tokenizer,
        template=template,
    )
    output = args.output.expanduser().resolve()
    summary_path = (
        args.summary or output.with_name(f"{output.stem}_summary.json")
    ).expanduser().resolve()
    C1.atomic_write_json(output, prompts)
    summary["prompt_bundle_sha256"] = C1.sha256_file(output)
    C1.atomic_write_json(summary_path, summary)
    print(
        f"wrote {output} ({len(prompts)} lifecycle prompts, "
        f"sha256={summary['prompt_bundle_sha256']})"
    )
    print(f"wrote {summary_path} (sha256={C1.sha256_file(summary_path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
