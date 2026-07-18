#!/usr/bin/env python3
"""Materialize frozen, leakage-audited SWE task-start probe prompts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from swe_task_contract import render_agents_md


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/swe_multitask_initial_protocol.json"
DEFAULT_CANDIDATES = ROOT / ".cache/swe_verified_initial_probe_candidates.json"
DEFAULT_SOURCE_REPORT = (
    ROOT / "validation/jlens-swe-qwen-code-native-nvfp4-ste-2026-07-17.json"
)
DEFAULT_OUTPUT = ROOT / ".cache/swe_multitask_initial/prompts.json"
MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
TOKENIZER_JSON_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
MIDDLE_BAND_LAYERS = tuple(range(16, 48))
MAX_PROMPT_TOKENS = 16_383
LOGIT_VOCABULARY_SIZE = 248_320


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_token_ids(token_ids: Sequence[int]) -> str:
    encoded = json.dumps(list(token_ids), separators=(",", ":"), ensure_ascii=True)
    return sha256_text(encoded)


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


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _identifier_surface_present(text: str, surface: str) -> bool:
    if not isinstance(surface, str) or not surface:
        raise ValueError("canonical target and foil surfaces must be non-empty strings")
    pattern = rf"(?<!\w){re.escape(surface)}(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _extract_template_span(
    prompt: str, start_marker: str, end_marker: str
) -> tuple[str, str, str]:
    if not start_marker or not end_marker:
        raise ValueError("AGENTS markers must be non-empty")
    if prompt.count(start_marker) != 1 or prompt.count(end_marker) != 1:
        raise ValueError("certified prompt must contain each AGENTS marker exactly once")
    start = prompt.index(start_marker)
    block_start = start + len(start_marker)
    end = prompt.index(end_marker, block_start)
    return prompt[:start], prompt[block_start:end], prompt[end + len(end_marker) :]


def _validate_form(form: Mapping[str, Any], tokenizer: Any, label: str) -> tuple[str, int]:
    text = form.get("text")
    token_id = form.get("token_id")
    if (
        not isinstance(text, str)
        or not text
        or isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
    ):
        raise ValueError(f"{label} has an invalid token form")
    encoded = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if encoded != [token_id] or decoded != text:
        raise ValueError(
            f"{label} token pin changed for {text!r}: encoded={encoded}, decoded={decoded!r}"
        )
    return text, token_id


def _candidate_concept_key(concept: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        concept.get("kind"),
        concept.get("path"),
        concept.get("target"),
        concept.get("contrast"),
    )


def _validate_protocol_header(protocol: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    require_equal(protocol.get("schema_version"), 1, "protocol schema version")
    require_equal(
        protocol.get("kind"),
        "swe_verified_initial_probe_protocol",
        "protocol kind",
    )
    require_equal(
        protocol.get("status"), "exploratory_development_pilot", "protocol status"
    )
    require_equal(
        protocol.get("lens_outputs_used_for_selection"),
        False,
        "lens-output selection flag",
    )
    pins = require_mapping(protocol.get("pins"), "protocol pins")
    candidate_pin = require_mapping(pins.get("candidate_manifest"), "candidate pin")
    template_pin = require_mapping(pins.get("template"), "template pin")
    model_pin = require_mapping(pins.get("model"), "model pin")
    metric = require_mapping(protocol.get("metric_contract"), "metric contract")
    require_equal(
        metric.get("middle_band_layers"),
        list(MIDDLE_BAND_LAYERS),
        "middle-band layers",
    )
    return candidate_pin, template_pin, model_pin, metric


def _validate_candidate_pin(
    candidate_pin: Mapping[str, Any], candidates: Mapping[str, Any], candidate_sha256: str
) -> None:
    require_equal(candidate_pin.get("sha256"), candidate_sha256, "candidate manifest hash")
    for field in (
        "schema_version",
        "kind",
        "source",
        "extraction",
        "task_count",
        "concept_count",
    ):
        require_equal(candidates.get(field), candidate_pin.get(field), f"candidate {field} pin")


def _validate_model_pin(model_pin: Mapping[str, Any], tokenizer: Any, snapshot: Path) -> None:
    require_equal(model_pin.get("repo_id"), MODEL_REPO, "model repository")
    require_equal(model_pin.get("revision"), MODEL_REVISION, "model revision")
    require_equal(
        model_pin.get("tokenizer_json_sha256"),
        TOKENIZER_JSON_SHA256,
        "tokenizer pin",
    )
    require_equal(sha256_file(snapshot / "tokenizer.json"), TOKENIZER_JSON_SHA256, "tokenizer file hash")
    require_equal(
        len(tokenizer), model_pin.get("tokenizer_vocabulary_size"), "tokenizer vocabulary size"
    )
    require_equal(
        model_pin.get("logit_vocabulary_size"),
        LOGIT_VOCABULARY_SIZE,
        "logit vocabulary size",
    )


def _validate_template(
    *,
    report: Mapping[str, Any],
    report_sha256: str,
    template_pin: Mapping[str, Any],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[str, str, str, str, str]:
    require_equal(template_pin.get("report_sha256"), report_sha256, "template report hash")
    experiment_index = template_pin.get("experiment_index")
    if isinstance(experiment_index, bool) or not isinstance(experiment_index, int):
        raise ValueError("template experiment index must be an integer")
    experiments = require_list(report.get("experiments"), "source report experiments")
    try:
        experiment = require_mapping(experiments[experiment_index], "template experiment")
    except IndexError as exc:
        raise ValueError("template experiment index is out of range") from exc
    require_equal(experiment.get("id"), template_pin.get("experiment_id"), "template experiment ID")
    prompt = experiment.get("prompt")
    prompt_token_ids = experiment.get("prompt_token_ids")
    if not isinstance(prompt, str) or not isinstance(prompt_token_ids, list):
        raise ValueError("template experiment lacks prompt or prompt token IDs")
    require_equal(sha256_text(prompt), template_pin.get("rendered_prompt_sha256"), "rendered prompt hash")
    require_equal(
        sha256_token_ids(prompt_token_ids),
        template_pin.get("prompt_token_ids_sha256"),
        "template token-ID hash",
    )
    require_equal(
        tokenizer.encode(prompt, add_special_tokens=False),
        prompt_token_ids,
        "certified prompt tokenization",
    )
    start_marker = template_pin.get("agents_start_marker")
    end_marker = template_pin.get("agents_end_marker")
    if not isinstance(start_marker, str) or not isinstance(end_marker, str):
        raise ValueError("template AGENTS markers must be strings")
    prefix, block, suffix = _extract_template_span(prompt, start_marker, end_marker)
    require_equal(sha256_text(block), template_pin.get("agents_block_sha256"), "AGENTS block hash")
    require_equal(sha256_text(prefix + suffix), template_pin.get("remainder_sha256"), "template remainder hash")

    source_id = template_pin.get("base_instance_id")
    source_slug = template_pin.get("base_hyphenated_project_slug")
    if not isinstance(source_id, str) or not isinstance(source_slug, str):
        raise ValueError("template source identifiers must be strings")
    require_equal(source_slug, source_id.replace("_", "-"), "source project slug")
    source_task = candidates_by_id.get(source_id)
    if source_task is None:
        raise ValueError("template source task is absent from the candidate manifest")
    expected_block = "\n" + render_agents_md(dict(source_task)).rstrip("\n") + "\n"
    require_equal(block, expected_block, "certified AGENTS byte contract")
    return prefix, suffix, source_id, source_slug, start_marker + end_marker


def _score_vocabulary(
    concepts: Sequence[Mapping[str, Any]], tokenizer: Any
) -> tuple[list[int], list[dict[str, Any]]]:
    score_ids: list[int] = []
    seen_ids: set[int] = set()
    materialized: list[dict[str, Any]] = []
    for concept_index, raw_concept in enumerate(concepts, 1):
        concept = require_mapping(raw_concept, f"concept {concept_index}")
        concept_id = concept.get("id")
        if not isinstance(concept_id, str) or not concept_id:
            raise ValueError(f"concept {concept_index} has an invalid ID")
        forms = require_list(concept.get("forms"), f"concept {concept_id} forms")
        if not forms:
            raise ValueError(f"concept {concept_id} has no target forms")
        for form_index, raw_form in enumerate(forms, 1):
            form = require_mapping(raw_form, f"concept {concept_id} form {form_index}")
            _, token_id = _validate_form(form, tokenizer, f"concept {concept_id}")
            if token_id not in seen_ids:
                seen_ids.add(token_id)
                score_ids.append(token_id)

        foils = require_list(concept.get("foils"), f"concept {concept_id} foils")
        for foil_index, raw_foil in enumerate(foils, 1):
            foil = require_mapping(raw_foil, f"concept {concept_id} foil {foil_index}")
            foil_forms = require_list(foil.get("forms"), f"concept {concept_id} foil forms")
            if not foil_forms:
                raise ValueError(f"concept {concept_id} has a foil without forms")
            for raw_form in foil_forms:
                form = require_mapping(raw_form, f"concept {concept_id} foil form")
                _, token_id = _validate_form(form, tokenizer, f"concept {concept_id} foil")
                if token_id not in seen_ids:
                    seen_ids.add(token_id)
                    score_ids.append(token_id)
        materialized.append(copy.deepcopy(dict(concept)))
    return score_ids, materialized


def build_probe_bundle(
    protocol: Mapping[str, Any],
    candidates: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    protocol_sha256: str,
    candidate_sha256: str,
    report_sha256: str,
    tokenizer: Any,
    snapshot: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_pin, template_pin, model_pin, _ = _validate_protocol_header(protocol)
    _validate_candidate_pin(candidate_pin, candidates, candidate_sha256)
    _validate_model_pin(model_pin, tokenizer, snapshot)
    candidate_tasks = require_list(candidates.get("tasks"), "candidate tasks")
    candidates_by_id = {
        task["instance_id"]: require_mapping(task, "candidate task")
        for task in candidate_tasks
    }
    if len(candidates_by_id) != len(candidate_tasks):
        raise ValueError("candidate manifest contains duplicate task IDs")
    prefix, suffix, source_id, source_slug, marker_pair = _validate_template(
        report=report,
        report_sha256=report_sha256,
        template_pin=template_pin,
        candidates_by_id=candidates_by_id,
        tokenizer=tokenizer,
    )
    start_marker = template_pin["agents_start_marker"]
    end_marker = template_pin["agents_end_marker"]

    tasks = require_list(protocol.get("tasks"), "protocol tasks")
    if not tasks:
        raise ValueError("protocol selects no tasks")
    bundle: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    global_score_ids: list[int] = []
    seen_global_score_ids: set[int] = set()
    for ordinal, raw_task in enumerate(tasks, 1):
        task = require_mapping(raw_task, f"protocol task {ordinal}")
        require_equal(
            task.get("selection_index"), ordinal - 1, f"protocol task {ordinal} selection index"
        )
        instance_id = task.get("instance_id")
        if not isinstance(instance_id, str) or instance_id in seen_task_ids:
            raise ValueError(f"protocol task {ordinal} has an invalid or duplicate ID")
        seen_task_ids.add(instance_id)
        candidate = candidates_by_id.get(instance_id)
        if candidate is None:
            raise ValueError(f"selected task {instance_id} is absent from candidates")
        for field in (
            "repo",
            "instance_id",
            "base_commit",
            "version",
            "problem_statement",
            "patch_sha256",
            "test_patch_sha256",
            "source_provenance",
        ):
            require_equal(task.get(field), candidate.get(field), f"task {instance_id} {field}")
        candidate_concepts = {
            _candidate_concept_key(require_mapping(value, "candidate concept")): value
            for value in require_list(candidate.get("concepts"), "candidate concepts")
        }
        concepts = require_list(task.get("concepts"), f"task {instance_id} concepts")
        for raw_concept in concepts:
            concept = require_mapping(raw_concept, f"task {instance_id} concept")
            source = candidate_concepts.get(_candidate_concept_key(concept))
            if source is None:
                raise ValueError(f"task {instance_id} contains a concept not derived from its gold patch")
            require_equal(concept.get("sources"), source.get("sources"), "concept evidence")

        agents_block = "\n" + render_agents_md(dict(task)).rstrip("\n") + "\n"
        target_slug = instance_id.replace("_", "-")
        rendered = prefix + start_marker + agents_block + end_marker + suffix
        rendered = rendered.replace(source_id, instance_id).replace(source_slug, target_slug)
        if source_id in rendered or source_slug in rendered:
            raise ValueError(f"source identifiers remain after rendering {instance_id}")
        if marker_pair in rendered:
            raise ValueError("AGENTS markers collapsed during rendering")

        for concept in concepts:
            concept_id = concept["id"]
            if _identifier_surface_present(rendered, concept.get("target")):
                raise ValueError(f"target leakage in {instance_id}/{concept_id}")
            for foil in require_list(concept.get("foils"), f"concept {concept_id} foils"):
                foil_map = require_mapping(foil, f"concept {concept_id} foil")
                if _identifier_surface_present(rendered, foil_map.get("target")):
                    raise ValueError(f"foil leakage in {instance_id}/{concept_id}")

        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        if len(token_ids) > MAX_PROMPT_TOKENS:
            raise ValueError(f"task {instance_id} prompt has {len(token_ids)} tokens")
        require_equal(
            len(token_ids),
            task.get("projected_prompt_token_count"),
            f"task {instance_id} projected token count",
        )
        require_equal(
            sha256_text(rendered),
            task.get("projected_prompt_sha256"),
            f"task {instance_id} projected prompt hash",
        )
        score_ids, materialized_concepts = _score_vocabulary(concepts, tokenizer)
        frozen_score_ids = require_list(
            task.get("score_token_ids"), f"task {instance_id} score token IDs"
        )
        if set(score_ids) != set(frozen_score_ids) or len(frozen_score_ids) != len(
            set(frozen_score_ids)
        ):
            raise ValueError(f"task {instance_id} score vocabulary does not match its concepts")
        score_ids = list(frozen_score_ids)
        for token_id in score_ids:
            if token_id not in seen_global_score_ids:
                seen_global_score_ids.add(token_id)
                global_score_ids.append(token_id)
        metadata_concepts = []
        for concept in materialized_concepts:
            metadata_concepts.append(
                {
                    "id": concept["id"],
                    "family": concept["family"],
                    "target": concept["target"],
                    "path": concept["path"],
                    "evidence": copy.deepcopy(concept["sources"]),
                    "visibility": "oracle_hidden",
                    "forms": copy.deepcopy(concept["forms"]),
                    "foils": copy.deepcopy(concept["foils"]),
                }
            )
        bundle.append(
            {
                "id": f"swe-initial-{ordinal:02d}-{instance_id}",
                "text": rendered,
                "token_ids": token_ids,
                "score_token_ids": score_ids,
                "metadata": {
                    "kind": "swe_verified_multitask_initial_probe",
                    "protocol_sha256": protocol_sha256,
                    "lens_outputs_used_for_selection": False,
                    "task": {
                        "instance_id": instance_id,
                        "repo": task["repo"],
                        "base_commit": task["base_commit"],
                        "problem_statement_sha256": sha256_text(task["problem_statement"]),
                        "patch_sha256": task["patch_sha256"],
                        "test_patch_sha256": task["test_patch_sha256"],
                    },
                    "checkpoint": {
                        "id": "C0",
                        "name": "task_start",
                        "visibility_boundary": "before_first_assistant_token",
                    },
                    "middle_band_layers": list(MIDDLE_BAND_LAYERS),
                    "concepts": metadata_concepts,
                },
            }
        )

    scored_vocabulary = require_mapping(
        protocol.get("scored_vocabulary"), "protocol scored vocabulary"
    )
    frozen_global_ids = require_list(
        scored_vocabulary.get("token_ids"), "global score token IDs"
    )
    if set(global_score_ids) != set(frozen_global_ids) or len(frozen_global_ids) != len(
        set(frozen_global_ids)
    ):
        raise ValueError("global score vocabulary does not match selected task concepts")

    summary = {
        "schema_version": 1,
        "kind": "swe_verified_multitask_initial_probe_materialization",
        "protocol_sha256": protocol_sha256,
        "candidate_manifest_sha256": candidate_sha256,
        "source_report_sha256": report_sha256,
        "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
        "prompt_count": len(bundle),
        "checkpoint": "C0",
        "middle_band_layers": list(MIDDLE_BAND_LAYERS),
        "lens_outputs_used_for_selection": False,
        "prompts": [
            {
                "id": item["id"],
                "instance_id": item["metadata"]["task"]["instance_id"],
                "prompt_sha256": sha256_text(item["text"]),
                "token_ids_sha256": sha256_token_ids(item["token_ids"]),
                "prompt_token_count": len(item["token_ids"]),
                "score_token_ids_sha256": sha256_token_ids(item["score_token_ids"]),
                "score_token_count": len(item["score_token_ids"]),
            }
            for item in bundle
        ],
    }
    return bundle, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    protocol_bytes = args.protocol.read_bytes()
    candidate_bytes = args.candidates.read_bytes()
    report_bytes = args.source_report.read_bytes()
    protocol = require_mapping(json.loads(protocol_bytes), "protocol")
    candidates = require_mapping(json.loads(candidate_bytes), "candidate manifest")
    report = require_mapping(json.loads(report_bytes), "source report")
    snapshot = Path(
        args.model_snapshot
        or snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)
    ).resolve(strict=True)
    if snapshot.name != MODEL_REVISION:
        raise ValueError(f"model snapshot must end in pinned revision {MODEL_REVISION}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    bundle, summary = build_probe_bundle(
        protocol,
        candidates,
        report,
        protocol_sha256=sha256_bytes(protocol_bytes),
        candidate_sha256=sha256_bytes(candidate_bytes),
        report_sha256=sha256_bytes(report_bytes),
        tokenizer=tokenizer,
        snapshot=snapshot,
    )
    output = args.output.resolve()
    summary_path = (args.summary or output.with_name(f"{output.stem}_summary.json")).resolve()
    atomic_write_json(output, bundle)
    summary["prompt_bundle_sha256"] = sha256_file(output)
    atomic_write_json(summary_path, summary)
    print(f"wrote {output} ({len(bundle)} prompts, sha256={summary['prompt_bundle_sha256']})")
    print(f"wrote {summary_path} (sha256={sha256_file(summary_path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
