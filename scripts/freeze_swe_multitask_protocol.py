#!/usr/bin/env python3
"""Freeze a leakage-resistant exploratory SWE Verified J-lens protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import keyword
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from swe_task_contract import render_agents_md  # noqa: E402


DEFAULT_CANDIDATES = ROOT / ".cache/swe_verified_initial_probe_candidates.json"
DEFAULT_TEMPLATE_REPORT = (
    ROOT / "validation/jlens-swe-qwen-code-native-nvfp4-ste-2026-07-17.json"
)
DEFAULT_OUTPUT = ROOT / "configs/swe_multitask_initial_protocol.json"
MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
TOKENIZER_JSON_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
TOKENIZER_VOCABULARY_SIZE = 248_077
LOGIT_VOCABULARY_SIZE = 248_320
DEFAULT_MODEL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4"
    / "snapshots"
    / MODEL_REVISION
)
TEMPLATE_REPORT_SHA256 = (
    "94f3963be698b0e6e86d3878ba69efa53a94059e5520f292a97f344d6bed6fab"
)
TEMPLATE_PROMPT_SHA256 = (
    "4226f190e941bf79d4cbb09fe64e9a692f831806d3752f5489dbdd9246a3d92a"
)
TEMPLATE_EXPERIMENT_INDEX = 0
TEMPLATE_BASE_INSTANCE_ID = "sympy__sympy-13480"
AGENTS_START_MARKER = "--- Context from: AGENTS.md ---"
AGENTS_END_MARKER = "--- End of Context from: AGENTS.md ---"
PILOT_TASK_COUNT = 10
MINIMUM_REPO_COUNT = 5
MAXIMUM_TASKS_PER_REPO = 2
MAXIMUM_CONCEPTS_PER_TASK = 3
MAXIMUM_PROMPT_TOKENS = 16_383
SELECTION_SEED = 36_027
BOOTSTRAP_SEED = 36_027
BOOTSTRAP_SAMPLES = 20_000
MIDDLE_BAND_LAYERS = tuple(range(16, 48))
PASS_K = (1, 5, 10, 50, 100, 1_000)
KIND_PRIORITY = {
    "identifier_replacement": 0,
    "symbol": 1,
    "file_stem": 2,
    "module_dir": 3,
}
FAMILY_BY_KIND = {
    "identifier_replacement": "replacement",
    "symbol": "hunk_symbol",
    "file_stem": "file_stem",
    "module_dir": "module_dir",
}
COMMON_GENERIC_CODING_TARGETS = frozenset(
    {
        "api",
        "app",
        "apps",
        "arg",
        "args",
        "base",
        "bool",
        "build",
        "cache",
        "class",
        "cli",
        "client",
        "cls",
        "code",
        "common",
        "config",
        "configuration",
        "context",
        "core",
        "data",
        "default",
        "dict",
        "error",
        "errors",
        "exception",
        "exceptions",
        "file",
        "files",
        "generic",
        "get",
        "helper",
        "helpers",
        "int",
        "list",
        "main",
        "manager",
        "model",
        "models",
        "module",
        "modules",
        "object",
        "objects",
        "option",
        "options",
        "parser",
        "path",
        "paths",
        "process",
        "property",
        "read",
        "request",
        "requests",
        "response",
        "responses",
        "result",
        "results",
        "runner",
        "server",
        "service",
        "services",
        "self",
        "set",
        "settings",
        "setup",
        "state",
        "str",
        "string",
        "strings",
        "task",
        "tasks",
        "test",
        "tests",
        "tool",
        "tools",
        "type",
        "types",
        "util",
        "utils",
        "value",
        "values",
        "version",
        "view",
        "views",
        "write",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def compact_json_sha256(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


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


def require_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{label} must be a{' possibly empty' if allow_empty else ' nonempty'} string")
    return value


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return (Path("$HOME") / resolved.relative_to(Path.home().resolve())).as_posix()
        except ValueError:
            return resolved.as_posix()


def seeded_digest(seed: int, *parts: str) -> str:
    payload = "\x1f".join((str(seed), *parts))
    return sha256_text(payload)


def surface_pattern(surface: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)", re.IGNORECASE)


def surface_present(surface: str, text: str) -> bool:
    return surface_pattern(surface).search(text) is not None


def validate_candidate_manifest(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    if value.get("schema_version") != 1:
        raise ValueError("candidate manifest schema_version must be 1")
    if value.get("kind") != "swe_verified_initial_probe_candidates":
        raise ValueError("candidate manifest kind mismatch")
    require_mapping(value.get("source"), "candidate source")
    require_mapping(value.get("extraction"), "candidate extraction")
    raw_tasks = value.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("candidate manifest requires a nonempty tasks list")
    if value.get("task_count") != len(raw_tasks):
        raise ValueError("candidate task_count mismatch")

    tasks: list[dict[str, Any]] = []
    instances: set[str] = set()
    concept_count = 0
    for task_index, raw_task in enumerate(raw_tasks):
        task = require_mapping(raw_task, f"task {task_index}")
        instance_id = require_string(task.get("instance_id"), f"task {task_index}.instance_id")
        if instance_id in instances:
            raise ValueError(f"duplicate candidate instance_id: {instance_id}")
        instances.add(instance_id)
        for field in ("repo", "base_commit", "version", "problem_statement"):
            require_string(task.get(field), f"task {instance_id}.{field}")
        for field in ("patch_sha256", "test_patch_sha256"):
            digest = require_string(task.get(field), f"task {instance_id}.{field}")
            if SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"task {instance_id}.{field} is not a SHA-256")
        require_mapping(task.get("source_provenance"), f"task {instance_id}.source_provenance")
        raw_concepts = task.get("concepts")
        if not isinstance(raw_concepts, list):
            raise ValueError(f"task {instance_id}.concepts must be a list")
        seen_concepts: set[tuple[str, str, str, str | None]] = set()
        for concept_index, raw_concept in enumerate(raw_concepts):
            concept = require_mapping(
                raw_concept, f"task {instance_id}.concept {concept_index}"
            )
            path = require_string(concept.get("path"), "concept.path")
            parsed_path = PurePosixPath(path)
            if parsed_path.is_absolute() or ".." in parsed_path.parts or parsed_path.suffix != ".py":
                raise ValueError(f"task {instance_id} has unsafe/non-Python concept path {path!r}")
            kind = concept.get("kind")
            if kind not in KIND_PRIORITY:
                raise ValueError(f"task {instance_id} has unsupported concept kind {kind!r}")
            target = require_string(concept.get("target"), "concept.target")
            contrast = concept.get("contrast")
            if contrast is not None:
                require_string(contrast, "concept.contrast")
            sources = concept.get("sources")
            if not isinstance(sources, list) or not sources or any(
                not isinstance(source, dict) for source in sources
            ):
                raise ValueError(f"task {instance_id} concept sources are invalid")
            key = (path, kind, target, contrast)
            if key in seen_concepts:
                raise ValueError(f"task {instance_id} contains a duplicate concept")
            seen_concepts.add(key)
        concept_count += len(raw_concepts)
        tasks.append(copy.deepcopy(dict(task)))
    if value.get("concept_count") != concept_count:
        raise ValueError("candidate concept_count mismatch")
    return tasks


def exact_token_forms(tokenizer: Any, target: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    forms: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for kind, text in (("bare", target), ("leading_space", f" {target}")):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            exclusions.append(
                {"kind": kind, "text": text, "token_ids": token_ids, "reason": "not_exactly_one_token"}
            )
            continue
        token_id = token_ids[0]
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != text:
            exclusions.append(
                {
                    "kind": kind,
                    "text": text,
                    "token_ids": token_ids,
                    "decoded": decoded,
                    "reason": "decode_roundtrip_mismatch",
                }
            )
            continue
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < LOGIT_VOCABULARY_SIZE
        ):
            exclusions.append(
                {
                    "kind": kind,
                    "text": text,
                    "token_ids": token_ids,
                    "decoded": decoded,
                    "reason": "outside_lm_head_vocabulary",
                }
            )
            continue
        forms.append({"kind": kind, "text": text, "token_id": token_id})
    return forms, exclusions


def extract_template(
    report_bytes: bytes,
    *,
    expected_report_sha256: str,
    expected_prompt_sha256: str,
) -> dict[str, Any]:
    report_sha256 = sha256_bytes(report_bytes)
    if report_sha256 != expected_report_sha256:
        raise ValueError(
            f"template report SHA-256 mismatch: {report_sha256} != {expected_report_sha256}"
        )
    report = require_mapping(json.loads(report_bytes), "template report")
    experiments = report.get("experiments")
    if not isinstance(experiments, list) or len(experiments) <= TEMPLATE_EXPERIMENT_INDEX:
        raise ValueError("template report does not contain experiment 0")
    experiment = require_mapping(
        experiments[TEMPLATE_EXPERIMENT_INDEX], "template experiment 0"
    )
    prompt = require_string(experiment.get("prompt"), "template rendered prompt")
    prompt_sha256 = sha256_text(prompt)
    if prompt_sha256 != expected_prompt_sha256:
        raise ValueError(
            f"template rendered prompt SHA-256 mismatch: {prompt_sha256} != {expected_prompt_sha256}"
        )
    prompt_token_ids = experiment.get("prompt_token_ids")
    if not isinstance(prompt_token_ids, list) or not prompt_token_ids or any(
        isinstance(item, bool) or not isinstance(item, int) for item in prompt_token_ids
    ):
        raise ValueError("template prompt_token_ids are invalid")
    if prompt.count(AGENTS_START_MARKER) != 1 or prompt.count(AGENTS_END_MARKER) != 1:
        raise ValueError("template must contain exactly one AGENTS marker pair")
    start = prompt.index(AGENTS_START_MARKER)
    content_start = start + len(AGENTS_START_MARKER)
    end = prompt.index(AGENTS_END_MARKER, content_start)
    if end <= content_start:
        raise ValueError("template AGENTS marker order is invalid")
    span_end = end + len(AGENTS_END_MARKER)
    block = prompt[content_start:end]
    remainder = prompt[:start] + prompt[span_end:]
    return {
        "report_sha256": report_sha256,
        "experiment_id": experiment.get("id"),
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "prompt_token_ids": list(prompt_token_ids),
        "prompt_token_ids_sha256": compact_json_sha256(prompt_token_ids),
        "agents_content_start": content_start,
        "agents_content_end": end,
        "agents_block": block,
        "agents_block_sha256": sha256_text(block),
        "remainder": remainder,
        "remainder_sha256": sha256_text(remainder),
    }


def project_prompt(
    template: Mapping[str, Any],
    base_task: Mapping[str, Any],
    target_task: Mapping[str, Any],
    *,
    renderer: Callable[[dict[str, Any]], str] = render_agents_md,
) -> str:
    prompt = str(template["prompt"])
    content_start = int(template["agents_content_start"])
    content_end = int(template["agents_content_end"])
    base_agents = renderer(dict(base_task))
    expected_block = "\n" + base_agents.rstrip("\n") + "\n"
    if prompt[content_start:content_end] != expected_block:
        raise ValueError("certified template AGENTS block does not match the base dataset row")
    target_agents = renderer(dict(target_task))
    replacement_block = "\n" + target_agents.rstrip("\n") + "\n"
    projected = prompt[:content_start] + replacement_block + prompt[content_end:]

    base_instance = str(base_task["instance_id"])
    target_instance = str(target_task["instance_id"])
    projected = projected.replace(base_instance, target_instance)
    base_slug = base_instance.replace("_", "-")
    target_slug = target_instance.replace("_", "-")
    projected = projected.replace(base_slug, target_slug)
    if base_instance != target_instance and base_instance in projected:
        raise ValueError("base instance ID remains after projected-prompt substitution")
    if base_slug != target_slug and base_slug in projected:
        raise ValueError("base hyphenated project slug remains after substitution")
    return projected


def concept_eligibility(
    concept: Mapping[str, Any],
    *,
    task_visible_text: str,
    template_remainder: str,
    tokenizer: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    target = str(concept["target"])
    if not target.isidentifier():
        return None, "not_python_identifier"
    if keyword.iskeyword(target):
        return None, "python_keyword"
    if len(target.strip("_")) < 3:
        return None, "target_canonical_too_short"
    if target.strip("_").casefold() in COMMON_GENERIC_CODING_TARGETS:
        return None, "generic_coding_stopword"
    if surface_present(target, task_visible_text):
        return None, "target_visible_in_task_agents"
    if surface_present(target, template_remainder):
        return None, "target_visible_in_template_remainder"
    forms, exclusions = exact_token_forms(tokenizer, target)
    if not forms:
        return None, "no_exact_single_token_form"
    result = copy.deepcopy(dict(concept))
    result["family"] = FAMILY_BY_KIND[str(result["kind"])]
    result["forms"] = forms
    result["form_exclusions"] = exclusions
    return result, None


def _concept_tie_key(seed: int, instance_id: str, concept: Mapping[str, Any]) -> tuple[str, str, str]:
    identity = "\x1f".join(
        (
            str(concept["kind"]),
            str(concept["path"]),
            str(concept["target"]),
            str(concept.get("contrast", "")),
        )
    )
    return seeded_digest(seed, "concept", instance_id, identity), str(concept["path"]), str(concept["target"])


def select_task_concepts(
    task: Mapping[str, Any],
    *,
    template_remainder: str,
    tokenizer: Any,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    instance_id = str(task["instance_id"])
    visible = render_agents_md(dict(task))
    eligible: list[dict[str, Any]] = []
    exclusions: dict[str, int] = {}
    for raw_concept in task["concepts"]:
        concept, reason = concept_eligibility(
            raw_concept,
            task_visible_text=visible,
            template_remainder=template_remainder,
            tokenizer=tokenizer,
        )
        if concept is None:
            assert reason is not None
            exclusions[reason] = exclusions.get(reason, 0) + 1
        else:
            eligible.append(concept)

    by_family: dict[str, list[dict[str, Any]]] = {}
    for concept in eligible:
        by_family.setdefault(str(concept["family"]), []).append(concept)
    family_winners: list[dict[str, Any]] = []
    for family, concepts in by_family.items():
        concepts.sort(key=lambda item: _concept_tie_key(seed, instance_id, item))
        family_winners.append(concepts[0])
        if len(concepts) > 1:
            exclusions["same_family_deduplication"] = (
                exclusions.get("same_family_deduplication", 0) + len(concepts) - 1
            )
    family_winners.sort(
        key=lambda item: (
            KIND_PRIORITY[str(item["kind"])],
            *_concept_tie_key(seed, instance_id, item),
        )
    )
    selected = family_winners[:MAXIMUM_CONCEPTS_PER_TASK]
    if len(family_winners) > len(selected):
        exclusions["per_task_concept_cap"] = len(family_winners) - len(selected)
    return selected, exclusions


def _task_quality(task: Mapping[str, Any], seed: int) -> tuple[Any, ...]:
    concepts = task["_selected_concepts"]
    priorities = [KIND_PRIORITY[str(concept["kind"])] for concept in concepts]
    strong_count = sum(priority <= KIND_PRIORITY["symbol"] for priority in priorities)
    return (
        min(priorities),
        -strong_count,
        -len(concepts),
        seeded_digest(seed, "task", str(task["repo"]), str(task["instance_id"])),
        str(task["instance_id"]),
    )


def repo_stratified_order(
    tasks: Sequence[Mapping[str, Any]], *, seed: int
) -> tuple[list[Mapping[str, Any]], list[str]]:
    by_repo: dict[str, list[Mapping[str, Any]]] = {}
    for task in tasks:
        by_repo.setdefault(str(task["repo"]), []).append(task)
    for repo_tasks in by_repo.values():
        repo_tasks.sort(key=lambda task: _task_quality(task, seed))
    repo_order = sorted(
        by_repo,
        key=lambda repo: (
            _task_quality(by_repo[repo][0], seed)[:3],
            seeded_digest(seed, "repo", repo),
            repo,
        ),
    )
    ordered: list[Mapping[str, Any]] = []
    maximum_depth = max(len(tasks_in_repo) for tasks_in_repo in by_repo.values())
    for depth in range(maximum_depth):
        for repo in repo_order:
            repo_tasks = by_repo[repo]
            if depth < len(repo_tasks):
                ordered.append(repo_tasks[depth])
    return ordered, repo_order


def assign_foils(
    tasks: list[dict[str, Any]], *, template_remainder: str, seed: int
) -> None:
    pool: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (task, concept) for task in tasks for concept in task["concepts"]
    ]
    for receiver in tasks:
        visible = render_agents_md(receiver)
        for concept in receiver["concepts"]:
            candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for source_task, source_concept in pool:
                if source_task["instance_id"] == receiver["instance_id"]:
                    continue
                if source_concept["family"] != concept["family"]:
                    continue
                if str(source_concept["target"]).casefold() == str(concept["target"]).casefold():
                    continue
                foil_target = str(source_concept["target"])
                if surface_present(foil_target, visible) or surface_present(
                    foil_target, template_remainder
                ):
                    continue
                candidates.append((source_task, source_concept))
            candidates.sort(
                key=lambda pair: (
                    seeded_digest(
                        seed,
                        "foil",
                        str(receiver["instance_id"]),
                        str(concept["id"]),
                        str(pair[0]["instance_id"]),
                        str(pair[1]["id"]),
                    ),
                    str(pair[0]["instance_id"]),
                    str(pair[1]["id"]),
                )
            )
            if not candidates:
                concept["foils"] = []
                concept["foil_status"] = "unavailable"
                concept["foil_unavailable_reason"] = (
                    "no_nonleaking_same_family_selected_cross_task_concept"
                )
                continue
            source_task, source_concept = candidates[0]
            concept["foils"] = [
                {
                    "task_instance_id": source_task["instance_id"],
                    "concept_id": source_concept["id"],
                    "family": source_concept["family"],
                    "kind": source_concept["kind"],
                    "target": source_concept["target"],
                    "forms": copy.deepcopy(source_concept["forms"]),
                }
            ]
            concept["foil_status"] = "assigned"
            concept["foil_unavailable_reason"] = None


def freeze_protocol(
    candidate_manifest: Mapping[str, Any],
    *,
    candidate_manifest_sha256: str,
    candidate_manifest_path: str,
    template: Mapping[str, Any],
    template_report_path: str,
    tokenizer: Any,
    model_snapshot_path: str,
    selection_seed: int = SELECTION_SEED,
) -> dict[str, Any]:
    candidate_tasks = validate_candidate_manifest(candidate_manifest)
    base_matches = [
        task for task in candidate_tasks if task["instance_id"] == TEMPLATE_BASE_INSTANCE_ID
    ]
    if len(base_matches) != 1:
        raise ValueError("candidate manifest must contain the certified template base task once")
    base_task = base_matches[0]

    base_ids = tokenizer.encode(str(template["prompt"]), add_special_tokens=False)
    if base_ids != template["prompt_token_ids"]:
        raise ValueError("pinned tokenizer does not reproduce template prompt_token_ids")
    decoded = tokenizer.decode(
        base_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != template["prompt"]:
        raise ValueError("pinned tokenizer does not round-trip the template prompt")
    if len(tokenizer) != TOKENIZER_VOCABULARY_SIZE:
        raise ValueError(
            f"tokenizer vocabulary size mismatch: {len(tokenizer)} != {TOKENIZER_VOCABULARY_SIZE}"
        )

    exclusion_totals: dict[str, int] = {}
    eligible_tasks: list[dict[str, Any]] = []
    over_budget_count = 0
    for raw_task in candidate_tasks:
        selected_concepts, exclusions = select_task_concepts(
            raw_task,
            template_remainder=str(template["remainder"]),
            tokenizer=tokenizer,
            seed=selection_seed,
        )
        for reason, count in exclusions.items():
            exclusion_totals[reason] = exclusion_totals.get(reason, 0) + count
        if not selected_concepts:
            continue
        projected = project_prompt(template, base_task, raw_task)
        projected_ids = tokenizer.encode(projected, add_special_tokens=False)
        if len(projected_ids) > MAXIMUM_PROMPT_TOKENS:
            over_budget_count += 1
            continue
        task = copy.deepcopy(raw_task)
        task["_selected_concepts"] = selected_concepts
        task["_projected_prompt_sha256"] = sha256_text(projected)
        task["_projected_prompt_token_count"] = len(projected_ids)
        eligible_tasks.append(task)

    # A repeated canonical would count the same easy label more than once. Freeze a
    # single owner before cohort selection, using kind priority and the seeded
    # concept identity as the predeclared tie-break. Empty tasks naturally yield to
    # the next ranked task from their repository.
    canonical_occurrences: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for task in eligible_tasks:
        for concept in task["_selected_concepts"]:
            canonical_occurrences.setdefault(
                str(concept["target"]).casefold(), []
            ).append((task, concept))
    duplicate_drop_count = 0
    for canonical, occurrences in canonical_occurrences.items():
        if len(occurrences) < 2:
            continue
        occurrences.sort(
            key=lambda pair: (
                KIND_PRIORITY[str(pair[1]["kind"])],
                seeded_digest(
                    selection_seed,
                    "cross-task-target",
                    canonical,
                    str(pair[0]["instance_id"]),
                    str(pair[1]["kind"]),
                    str(pair[1]["path"]),
                    str(pair[1]["target"]),
                ),
                str(pair[0]["instance_id"]),
            )
        )
        for losing_task, losing_concept in occurrences[1:]:
            losing_task["_selected_concepts"].remove(losing_concept)
            duplicate_drop_count += 1
    if duplicate_drop_count:
        exclusion_totals["cross_task_target_deduplication"] = duplicate_drop_count
    eligible_tasks = [task for task in eligible_tasks if task["_selected_concepts"]]

    ordered, repo_order = repo_stratified_order(eligible_tasks, seed=selection_seed)
    selected_raw: list[Mapping[str, Any]] = []
    repo_counts: dict[str, int] = {}
    for task in ordered:
        repo = str(task["repo"])
        if repo_counts.get(repo, 0) >= MAXIMUM_TASKS_PER_REPO:
            continue
        selected_raw.append(task)
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
        if len(selected_raw) == PILOT_TASK_COUNT:
            break
    if len(selected_raw) != PILOT_TASK_COUNT:
        raise ValueError(
            f"eligible cohort cannot fill {PILOT_TASK_COUNT} tasks under the repo cap"
        )
    selected_repos = {str(task["repo"]) for task in selected_raw}
    if len(selected_repos) < MINIMUM_REPO_COUNT:
        raise ValueError("selected cohort does not meet minimum repository diversity")
    selected_ids = {str(task["instance_id"]) for task in selected_raw}

    tasks: list[dict[str, Any]] = []
    for selection_index, raw_task in enumerate(selected_raw):
        task = {
            "selection_index": selection_index,
            "repo": raw_task["repo"],
            "instance_id": raw_task["instance_id"],
            "base_commit": raw_task["base_commit"],
            "version": raw_task["version"],
            "problem_statement": raw_task["problem_statement"],
            "patch_sha256": raw_task["patch_sha256"],
            "test_patch_sha256": raw_task["test_patch_sha256"],
            "source_provenance": copy.deepcopy(raw_task["source_provenance"]),
            "projected_prompt_sha256": raw_task["_projected_prompt_sha256"],
            "projected_prompt_token_count": raw_task["_projected_prompt_token_count"],
            "concepts": copy.deepcopy(raw_task["_selected_concepts"]),
        }
        for concept_index, concept in enumerate(task["concepts"]):
            concept["id"] = f"task-{selection_index:02d}-concept-{concept_index:02d}"
        tasks.append(task)

    assign_foils(
        tasks, template_remainder=str(template["remainder"]), seed=selection_seed
    )
    scored: dict[int, str] = {}
    for task in tasks:
        task_ids: set[int] = set()
        for concept in task["concepts"]:
            records = list(concept["forms"])
            for foil in concept["foils"]:
                records.extend(foil["forms"])
            for form in records:
                token_id, text = int(form["token_id"]), str(form["text"])
                if token_id in scored and scored[token_id] != text:
                    raise ValueError(f"token ID {token_id} has inconsistent decoded forms")
                scored[token_id] = text
                task_ids.add(token_id)
        task["score_token_ids"] = sorted(task_ids)

    replacement_order = [
        {
            "rank": rank,
            "repo": task["repo"],
            "instance_id": task["instance_id"],
            "projected_prompt_token_count": task["_projected_prompt_token_count"],
        }
        for rank, task in enumerate(
            (task for task in ordered if str(task["instance_id"]) not in selected_ids),
            1,
        )
    ]
    strong_task_count = sum(
        any(concept["kind"] in ("identifier_replacement", "symbol") for concept in task["concepts"])
        for task in tasks
    )
    if strong_task_count == 0:
        raise ValueError("selected cohort has no task with a strong concept")
    single_repo_round = len({str(task["repo"]) for task in eligible_tasks}) >= PILOT_TASK_COUNT

    return {
        "schema_version": 1,
        "kind": "swe_verified_initial_probe_protocol",
        "status": "exploratory_development_pilot",
        "lens_outputs_used_for_selection": False,
        "pins": {
            "candidate_manifest": {
                "path": candidate_manifest_path,
                "sha256": candidate_manifest_sha256,
                "schema_version": candidate_manifest["schema_version"],
                "kind": candidate_manifest["kind"],
                "source": copy.deepcopy(candidate_manifest["source"]),
                "extraction": copy.deepcopy(candidate_manifest["extraction"]),
                "task_count": candidate_manifest["task_count"],
                "concept_count": candidate_manifest["concept_count"],
            },
            "template": {
                "report_path": template_report_path,
                "report_sha256": template["report_sha256"],
                "experiment_index": TEMPLATE_EXPERIMENT_INDEX,
                "experiment_id": template["experiment_id"],
                "rendered_prompt_sha256": template["prompt_sha256"],
                "prompt_token_ids_sha256": template["prompt_token_ids_sha256"],
                "prompt_token_ids_sha256_encoding": "utf8_compact_sorted_json",
                "agents_start_marker": AGENTS_START_MARKER,
                "agents_end_marker": AGENTS_END_MARKER,
                "agents_block_sha256": template["agents_block_sha256"],
                "agents_block_sha256_scope": "utf8_exact_text_between_markers",
                "remainder_sha256": template["remainder_sha256"],
                "remainder_sha256_scope": "utf8_prompt_with_full_AGENTS_marker_span_removed",
                "base_instance_id": TEMPLATE_BASE_INSTANCE_ID,
                "base_hyphenated_project_slug": TEMPLATE_BASE_INSTANCE_ID.replace("_", "-"),
            },
            "model": {
                "repo_id": MODEL_REPO,
                "revision": MODEL_REVISION,
                "model_snapshot_path": model_snapshot_path,
                "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
                "tokenizer_vocabulary_size": TOKENIZER_VOCABULARY_SIZE,
                "logit_vocabulary_size": LOGIT_VOCABULARY_SIZE,
            },
        },
        "selection_contract": {
            "selection_seed": selection_seed,
            "ranking_algorithm": "sha256_seeded_repo_stratified_round_robin",
            "pilot_task_count": PILOT_TASK_COUNT,
            "minimum_repo_count": MINIMUM_REPO_COUNT,
            "maximum_tasks_per_repo": MAXIMUM_TASKS_PER_REPO,
            "maximize_first_round_repo_diversity": True,
            "one_task_per_repo_first_round_used": single_repo_round,
            "maximum_concepts_per_task": MAXIMUM_CONCEPTS_PER_TASK,
            "concept_priority": list(KIND_PRIORITY),
            "concept_family_mapping": dict(FAMILY_BY_KIND),
            "family_deduplication": "at most one retained concept per mapped family per task",
            "cross_task_target_deduplication": "one casefolded target canonical globally, keeping the kind-priority then seeded winner",
            "strong_concept_kinds": ["identifier_replacement", "symbol"],
            "target_form_candidates": ["target", "leading-space target"],
            "target_form_requirement": "one or more exact decode-roundtripping single tokens below LM-head vocabulary size",
            "target_surface_boundary": "case-insensitive (?<!\\w)surface(?!\\w)",
            "target_visible_text": "render_agents_md(task) plus certified template remainder",
            "generic_stopword_normalization": "target.strip('_').casefold()",
            "generic_coding_stopwords": sorted(COMMON_GENERIC_CODING_TARGETS),
            "filter_order": [
                "valid Python identifier",
                "not Python keyword",
                "canonical length after stripping underscores is at least three",
                "not generic coding stopword",
                "absent from task-visible AGENTS text at identifier boundary",
                "absent from template remainder at identifier boundary",
                "has exact single-token form",
                "family deduplication",
                "per-task concept cap",
                "projected prompt token budget",
            ],
            "maximum_projected_prompt_tokens": MAXIMUM_PROMPT_TOKENS,
            "projected_generation_token_reserve": 1,
            "foil_assignment": "one seeded same-family selected cross-task concept when nonleaking; otherwise explicit empty foils",
            "replacement_rule": "consume replacement_order in rank order while preserving task count and maximum two tasks per repo",
        },
        "eligibility_audit": {
            "candidate_task_count": len(candidate_tasks),
            "candidate_concept_count": sum(len(task["concepts"]) for task in candidate_tasks),
            "eligible_task_count": len(eligible_tasks),
            "eligible_repo_count": len({str(task["repo"]) for task in eligible_tasks}),
            "eligible_repo_ranking": repo_order,
            "concept_exclusion_counts": dict(sorted(exclusion_totals.items())),
            "over_prompt_budget_task_count": over_budget_count,
        },
        "metric_contract": {
            "middle_band_layers": list(MIDDLE_BAND_LAYERS),
            "rank_reduction": "minimum exact rank across retained forms and fixed middle-band layers per concept",
            "task_aggregation": "equal mean across retained concepts within task",
            "cohort_aggregation": "equal mean across tasks",
            "pass_at_k": list(PASS_K),
            "normalized_log_rank_auc": "(log(logit_vocabulary_size)-log(minimum_rank))/log(logit_vocabulary_size); zero if unscorable",
            "comparisons": [
                "public_jacobian_minus_logit",
                "native_jacobian_minus_logit",
                "native_jacobian_minus_public_jacobian",
            ],
            "bootstrap": {
                "method": "deterministic paired task-level percentile bootstrap",
                "resampling_unit": "task",
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "confidence_level": 0.95,
            },
        },
        "coverage": {
            "selected_task_count": len(tasks),
            "selected_repo_count": len(selected_repos),
            "selected_concept_count": sum(len(task["concepts"]) for task in tasks),
            "strong_task_count": strong_task_count,
            "assigned_foil_count": sum(
                len(concept["foils"]) for task in tasks for concept in task["concepts"]
            ),
            "unavailable_foil_count": sum(
                not concept["foils"] for task in tasks for concept in task["concepts"]
            ),
        },
        "replacement_order": replacement_order,
        "scored_vocabulary": {
            "scope": "selected task targets plus assigned receiving-task foils",
            "token_ids": sorted(scored),
            "tokens": [scored[token_id] for token_id in sorted(scored)],
        },
        "tasks": tasks,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--template-report", type=Path, default=DEFAULT_TEMPLATE_REPORT)
    parser.add_argument(
        "--template-report-sha256", default=TEMPLATE_REPORT_SHA256
    )
    parser.add_argument(
        "--template-prompt-sha256", default=TEMPLATE_PROMPT_SHA256
    )
    parser.add_argument("--model-snapshot", type=Path, default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_path = args.candidates.expanduser().resolve(strict=True)
    template_path = args.template_report.expanduser().resolve(strict=True)
    snapshot_path = args.model_snapshot.expanduser().resolve(strict=True)
    tokenizer_path = snapshot_path / "tokenizer.json"
    if sha256_file(tokenizer_path) != TOKENIZER_JSON_SHA256:
        raise ValueError("Qwen tokenizer.json SHA-256 mismatch")
    if SHA256_RE.fullmatch(args.template_report_sha256) is None:
        raise ValueError("--template-report-sha256 must be a lowercase SHA-256")
    if SHA256_RE.fullmatch(args.template_prompt_sha256) is None:
        raise ValueError("--template-prompt-sha256 must be a lowercase SHA-256")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
    candidate_bytes = candidate_path.read_bytes()
    candidate_manifest = require_mapping(
        json.loads(candidate_bytes), "candidate manifest"
    )
    template = extract_template(
        template_path.read_bytes(),
        expected_report_sha256=args.template_report_sha256,
        expected_prompt_sha256=args.template_prompt_sha256,
    )
    protocol = freeze_protocol(
        candidate_manifest,
        candidate_manifest_sha256=sha256_bytes(candidate_bytes),
        candidate_manifest_path=portable_path(candidate_path),
        template=template,
        template_report_path=portable_path(template_path),
        tokenizer=tokenizer,
        model_snapshot_path=portable_path(snapshot_path),
    )
    output_path = args.output.expanduser().resolve()
    atomic_write_json(output_path, protocol)
    coverage = protocol["coverage"]
    print(
        f"froze {coverage['selected_task_count']} tasks from "
        f"{coverage['selected_repo_count']} repos with "
        f"{coverage['selected_concept_count']} concepts: {output_path}"
    )
    print(f"protocol sha256={sha256_file(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
