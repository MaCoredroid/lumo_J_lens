#!/usr/bin/env python3
"""Materialize the exact frozen V3 N=60 all-probeable prompt bundle.

This is a narrow compatibility adapter around the historical behavioral
materializer.  The checked-in V3 declaration checker authenticates every
immutable input and every generated-run image binding before the historical
implementation is allowed to read the runs.  The adapter changes only the V3
manifest identity, dense checkpoint selection, and image-manifest provenance.
"""

from __future__ import annotations

import builtins
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import types
from typing import Any, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PINNED_HISTORICAL_SOURCES = {
    "swe_task_contract": (
        ROOT / "scripts/swe_task_contract.py",
        "b87f58d83921354eb555b8a1937e8028996c2d5ce0b42fe9b14136e9c061cda5",
    ),
    "materialize_swe_multitask_initial_probes": (
        ROOT / "scripts/materialize_swe_multitask_initial_probes.py",
        "531580c4b825a2bfc25e79b86612752b7ee90c120255c0164a95ab52df11a207",
    ),
    "materialize_swe_jlens_prompts": (
        ROOT / "scripts/materialize_swe_jlens_prompts.py",
        "9998c4fb6f16d17e97aad9ee609547c6f9657d246211b8b0c28586374244bfe0",
    ),
    "materialize_swe_multitask_c1_probes": (
        ROOT / "scripts/materialize_swe_multitask_c1_probes.py",
        "9bde3c0b9b0505652551e16488044afae38a1b4b114dc0ff1d416977cbfc9bc9",
    ),
    "materialize_swe_behavioral_probes": (
        ROOT / "scripts/materialize_swe_behavioral_probes.py",
        "c63fac2907b887d973920c8fc71adf219affa1d6373a0aeb8ac2fffd59940a4e",
    ),
}
PINNED_HISTORICAL_LOAD_ORDER = (
    "swe_task_contract",
    "materialize_swe_multitask_initial_probes",
    "materialize_swe_jlens_prompts",
    "materialize_swe_multitask_c1_probes",
    "materialize_swe_behavioral_probes",
)


def _load_pinned_historical() -> types.ModuleType:
    """Execute the verified historical closure as fresh, isolated modules."""

    payloads: dict[str, bytes] = {}
    if tuple(PINNED_HISTORICAL_SOURCES) != PINNED_HISTORICAL_LOAD_ORDER:
        raise RuntimeError("pinned historical source load order changed")
    for name in PINNED_HISTORICAL_LOAD_ORDER:
        path, expected_sha256 = PINNED_HISTORICAL_SOURCES[name]
        if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path.absolute():
            raise RuntimeError(f"pinned historical source is missing or unsafe: {path}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise RuntimeError(f"pinned historical source SHA-256 changed: {path}")
        payloads[name] = payload

    private_names = {
        name: (
            f"_swe_state_v3_pinned_{name}_"
            f"{hashlib.sha256(payloads[name]).hexdigest()}"
        )
        for name in PINNED_HISTORICAL_LOAD_ORDER
    }
    missing = object()
    original_modules = {
        name: sys.modules.get(name, missing) for name in private_names.values()
    }
    caller_modules = {
        name: sys.modules.get(name, missing) for name in PINNED_HISTORICAL_LOAD_ORDER
    }
    modules: dict[str, types.ModuleType] = {}

    def isolated_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if level == 0 and name in PINNED_HISTORICAL_SOURCES:
            dependency = modules.get(name)
            if dependency is None:
                raise ImportError(f"pinned historical dependency loaded out of order: {name}")
            return dependency
        return builtins.__import__(name, globals, locals, fromlist, level)

    try:
        for name in PINNED_HISTORICAL_LOAD_ORDER:
            path, _expected_sha256 = PINNED_HISTORICAL_SOURCES[name]
            private_name = private_names[name]
            specification = importlib.util.spec_from_file_location(private_name, path)
            if specification is None or specification.loader is None:
                raise RuntimeError(f"cannot load pinned historical source: {name}")
            module = importlib.util.module_from_spec(specification)
            modules[name] = module
            private_builtins = dict(vars(builtins))
            private_builtins["__import__"] = isolated_import
            module.__dict__["__builtins__"] = private_builtins
            # Keep the fresh module available under only its private identity.
            # Its absolute closure imports are routed by the private builtins
            # hook, so caller-owned public sys.modules entries are never hidden.
            sys.modules[private_name] = module
            exec(compile(payloads[name], str(path), "exec"), module.__dict__)
            if sys.modules.get(private_name) is not module:
                raise RuntimeError(f"pinned historical module identity changed: {name}")
            if any(
                sys.modules.get(public_name, missing) is not original
                for public_name, original in caller_modules.items()
            ):
                raise RuntimeError("historical closure changed caller-owned module state")

        for name in PINNED_HISTORICAL_LOAD_ORDER:
            path, _expected_sha256 = PINNED_HISTORICAL_SOURCES[name]
            loaded = modules[name]
            loaded_path = Path(str(getattr(loaded, "__file__", ""))).resolve(strict=True)
            if loaded_path != path.resolve(strict=True):
                raise RuntimeError(f"historical dependency loaded from wrong path: {name}")
            if path.read_bytes() != payloads[name]:
                raise RuntimeError(f"historical dependency changed during import: {name}")

        contract = modules["swe_task_contract"]
        initial = modules["materialize_swe_multitask_initial_probes"]
        renderer = modules["materialize_swe_jlens_prompts"]
        c1 = modules["materialize_swe_multitask_c1_probes"]
        historical_module = modules["materialize_swe_behavioral_probes"]
        if getattr(initial, "render_agents_md", None) is not getattr(
            contract, "render_agents_md", None
        ):
            raise RuntimeError("historical C0 did not bind the private task contract")
        if getattr(c1, "C0", None) is not initial or getattr(c1, "RENDER", None) is not renderer:
            raise RuntimeError("historical C1 did not bind the private C0/render closure")
        if (
            getattr(historical_module, "C0", None) is not initial
            or getattr(historical_module, "C1", None) is not c1
        ):
            raise RuntimeError("historical materializer did not bind the private C0/C1 closure")
        return historical_module
    finally:
        for name, original in original_modules.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


historical = _load_pinned_historical()

ALL_PROBEABLE_FLAG = "--all-probeable"
V3_COHORT_MANIFEST_KIND = "swe_verified_state_interpreter_v3_n60_cohort"
HISTORICAL_COHORT_MANIFEST_KIND = "swe_verified_behavioral_n20_cohort"
CHECKER_PATH = ROOT / "scripts/check_swe_task_state_v3_development_cohort.py"
# Updated only after the declaration/checker/protocol byte chain is finalized.
CHECKER_SHA256 = "0b0ddc053669fab6ef6c37ddd26ee523d66a135d7515bc9c6dece10ff979a21c"
V3_COHORT_PATH = ROOT / "configs/swe_task_state_v3_development_cohort.json"
V3_ACTION_PROTOCOL_PATH = ROOT / "configs/swe_task_state_v3_action_probes.json"
V3_TEMPLATE_PATH = ROOT / "configs/qwen3-openai-codex.jinja"
V3_RUNS_ROOT = ROOT / "runs/swe_state_interpreter_v3_development"
V3_CACHE_ROOT = ROOT / ".cache/swe_state_interpreter_v3_development"
V3_PROMPTS_PATH = V3_CACHE_ROOT / "prompts.json"
V3_SUMMARY_PATH = V3_CACHE_ROOT / "prompts-summary.json"
V3_RECEIPT_PATH = ROOT / "validation/swe-task-state-v3-development-materialization.json"


def _required_option_value(argv: Sequence[str], option: str) -> str:
    """Return one explicit CLI option value while rejecting ambiguity."""

    values: list[str] = []
    for index, value in enumerate(argv):
        if value == option:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise SystemExit(f"{option} requires a path")
            values.append(argv[index + 1])
        elif value.startswith(f"{option}="):
            option_value = value.partition("=")[2]
            if not option_value:
                raise SystemExit(f"{option} requires a path")
            values.append(option_value)
    if len(values) != 1:
        raise SystemExit(f"V3 materialization requires exactly one explicit {option}")
    return values[0]


def _option_count(argv: Sequence[str], option: str) -> int:
    return sum(value == option or value.startswith(f"{option}=") for value in argv)


def _lexical_absolute(path: Path) -> Path:
    candidate = path.expanduser()
    historical.require(".." not in candidate.parts, f"non-canonical path: {path}")
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return Path(os.path.abspath(candidate))


def _require_exact_path(
    path: Path,
    expected: Path,
    label: str,
    *,
    directory: bool = False,
) -> Path:
    """Require an exact, existing path with no symlink in its traversal."""

    candidate = _lexical_absolute(path)
    expected_absolute = _lexical_absolute(expected)
    historical.require(candidate == expected_absolute, f"{label} is not the exact frozen path")
    historical.require(candidate.exists() and not candidate.is_symlink(), f"{label} is missing or symlinked")
    historical.require(
        candidate.resolve(strict=True) == candidate,
        f"{label} traverses a symlink",
    )
    if directory:
        historical.require(candidate.is_dir(), f"{label} is not a directory")
    else:
        historical.require(candidate.is_file(), f"{label} is not a regular file")
    return candidate


def _open_directory_chain(path: Path) -> int:
    """Open an absolute directory without following any path-component symlink."""

    absolute = _lexical_absolute(path)
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for piece in absolute.parts[1:]:
            child = os.open(piece, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _ensure_cache_root() -> None:
    """Create the one V3 cache directory through non-symlink directory FDs."""

    try:
        parent_fd = _open_directory_chain(V3_CACHE_ROOT.parent)
    except OSError as error:
        raise ValueError(f"dedicated V3 cache path is unsafe: {error}") from error
    try:
        try:
            os.mkdir(V3_CACHE_ROOT.name, mode=0o755, dir_fd=parent_fd)
        except FileExistsError:
            pass
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        cache_fd = os.open(V3_CACHE_ROOT.name, flags, dir_fd=parent_fd)
        try:
            mode = os.fstat(cache_fd).st_mode
            historical.require(stat.S_ISDIR(mode), "dedicated V3 cache is not a directory")
        finally:
            os.close(cache_fd)
    except OSError as error:
        raise ValueError(f"dedicated V3 cache path is unsafe: {error}") from error
    finally:
        os.close(parent_fd)


def _validate_output_path(path: Path, label: str) -> Path:
    candidate = _lexical_absolute(path)
    historical.require(
        candidate.parent == V3_CACHE_ROOT,
        f"{label} must be a direct child of the dedicated V3 cache",
    )
    historical.require(candidate.suffix == ".json", f"{label} must be a JSON file")
    historical.require(
        not os.path.lexists(candidate),
        f"{label} target already exists; canonical materialization is no-clobber",
    )
    return candidate


def _validate_new_receipt_path() -> Path:
    candidate = _lexical_absolute(V3_RECEIPT_PATH)
    expected = ROOT / "validation/swe-task-state-v3-development-materialization.json"
    historical.require(candidate == expected.absolute(), "V3 receipt path changed")
    historical.require(
        candidate.parent.is_dir()
        and not candidate.parent.is_symlink()
        and candidate.parent.resolve(strict=True) == candidate.parent.absolute(),
        "V3 receipt parent is unsafe",
    )
    historical.require(
        not os.path.lexists(candidate),
        "V3 materialization receipt already exists; receipt creation is no-clobber",
    )
    return candidate


def _write_new_json(path: Path, value: Any) -> None:
    """Publish one JSON file atomically without any overwrite race."""

    rendered = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
    historical.require(not os.path.lexists(temporary), "receipt staging path already exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = _open_directory_chain(path.parent)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def _load_pinned_checker() -> types.ModuleType:
    """Execute only the exact expected checker bytes."""

    checker_path = _require_exact_path(CHECKER_PATH, CHECKER_PATH, "V3 declaration checker")
    payload = checker_path.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    historical.require(observed == CHECKER_SHA256, "V3 declaration checker SHA-256 changed")
    module_name = f"_pinned_swe_state_v3_checker_{observed}"
    module = types.ModuleType(module_name)
    module.__file__ = str(checker_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(payload, str(checker_path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    historical.require(callable(getattr(module, "validate_declaration", None)), "V3 checker API changed")
    historical.require(callable(getattr(module, "validate_run_image_provenance", None)), "V3 run-image checker API changed")
    historical.require(callable(getattr(module, "validate_materialized_bundle", None)), "V3 bundle checker API changed")
    historical.require(callable(getattr(module, "capture_clean_source_freeze", None)), "V3 source-freeze checker API changed")
    historical.require(callable(getattr(module, "build_materialization_receipt", None)), "V3 receipt builder API changed")
    historical.require(callable(getattr(module, "validate_materialization_receipt", None)), "V3 receipt checker API changed")
    return module


def _prepare_delegated_argv(
    argv: Sequence[str], *, declaration: Any
) -> tuple[list[str], Path, Path]:
    """Require the exact declared two-cohort invocation and explicit outputs."""

    delegated = list(argv)
    all_probeable_indices = [
        index for index, value in enumerate(delegated) if value == ALL_PROBEABLE_FLAG
    ]
    if len(all_probeable_indices) != 1:
        raise SystemExit("V3 dense materialization requires exactly one --all-probeable flag")
    delegated.pop(all_probeable_indices[0])

    for option in (
        "--cohort-manifest",
        "--action-protocol",
        "--template",
        "--output",
        "--summary",
    ):
        _required_option_value(delegated, option)
    if _option_count(delegated, "--cohort") != 2:
        raise SystemExit("V3 materialization requires exactly two explicit --cohort pairs")
    if any(value.startswith("--cohort=") for value in delegated):
        raise SystemExit("V3 --cohort pairs must use the explicit two-argument form")
    if _option_count(delegated, "--run-root") or _option_count(delegated, "--campaign"):
        raise SystemExit("V3 materialization accepts only the frozen two-cohort form")
    if _option_count(delegated, "--model-snapshot"):
        raise SystemExit(
            "V3 materialization resolves only the frozen campaign model revision; "
            "an arbitrary --model-snapshot is forbidden"
        )
    if _option_count(delegated, "--require-official-outcomes") > 1:
        raise SystemExit("V3 materialization accepts at most one --require-official-outcomes flag")

    args = historical.parse_args(delegated)
    historical.require(args.cohort is not None and len(args.cohort) == 2, "V3 cohort parse changed")
    _require_exact_path(args.cohort_manifest, declaration.cohort_path, "V3 cohort manifest")
    _require_exact_path(args.action_protocol, V3_ACTION_PROTOCOL_PATH, "V3 action protocol")
    _require_exact_path(args.template, V3_TEMPLATE_PATH, "V3 chat template")

    rows = [historical.mapping(row, f"V3 cohort row {index}") for index, row in enumerate(declaration.cohort["cohorts"])]
    historical.require(len(rows) == 2, "V3 declaration does not contain exactly A/B")
    for index, ((campaign_path, run_root), declared_campaign, row) in enumerate(
        zip(args.cohort, declaration.campaign_paths, rows, strict=True)
    ):
        _require_exact_path(campaign_path, declared_campaign, f"V3 campaign {index}")
        expected_run = V3_RUNS_ROOT / str(row["run_label"])
        _require_exact_path(run_root, expected_run, f"V3 run root {index}", directory=True)

    output = _validate_output_path(args.output, "V3 prompt output")
    summary = _validate_output_path(args.summary, "V3 prompt summary")
    historical.require(output != summary, "V3 prompt output and summary must differ")
    historical.require(
        output == V3_CACHE_ROOT / "prompts.json"
        and summary == V3_CACHE_ROOT / "prompts-summary.json",
        "V3 materialization outputs are not the exact canonical filenames",
    )
    return delegated, output, summary


def _translate_v3_manifest(
    manifest: Mapping[str, Any], *, action_protocol_logical_path: str
) -> dict[str, Any]:
    """Authenticate V3 identity, then translate only legacy expectations."""

    historical.require(
        manifest.get("schema_version") == 1
        and manifest.get("kind") == V3_COHORT_MANIFEST_KIND,
        "V3 cohort manifest schema mismatch",
    )
    action_binding = historical.mapping(manifest.get("action_protocol"), "V3 cohort action_protocol")
    historical.require(
        action_binding.get("path") == action_protocol_logical_path,
        "V3 cohort action protocol path mismatch",
    )
    translated = copy.deepcopy(dict(manifest))
    translated["kind"] = HISTORICAL_COHORT_MANIFEST_KIND
    translated_action = dict(historical.mapping(translated.get("action_protocol"), "translated cohort action_protocol"))
    translated_action["path"] = historical.DEFAULT_ACTION_PROTOCOL.relative_to(historical.ROOT).as_posix()
    translated["action_protocol"] = translated_action
    return translated


@contextmanager
def _v3_manifest_validation_patch(action_protocol_path: Path) -> Iterator[None]:
    """Route the already-authenticated V3 identity through the legacy validator."""

    resolved_protocol = _require_exact_path(
        action_protocol_path, V3_ACTION_PROTOCOL_PATH, "V3 action protocol"
    )
    logical_protocol = resolved_protocol.relative_to(ROOT).as_posix()
    original_validator = historical.validate_cohort_manifest

    def validate_v3_cohort_manifest(
        manifest: Mapping[str, Any], **kwargs: Any
    ) -> list[dict[str, Any]]:
        translated = _translate_v3_manifest(
            manifest, action_protocol_logical_path=logical_protocol
        )
        return original_validator(translated, **kwargs)

    historical.validate_cohort_manifest = validate_v3_cohort_manifest
    try:
        yield
    finally:
        historical.validate_cohort_manifest = original_validator


@contextmanager
def _all_probeable_patch() -> Iterator[None]:
    """Disable only uniform checkpoint thinning for this delegated call."""

    original_selector = historical.select_probeable_requests
    original_max_checkpoints = historical.MAX_CHECKPOINTS

    def select_all_probeable_requests(
        task: Mapping[str, Any], *, max_prompt_tokens: int, **_: Any
    ) -> dict[str, Any]:
        captures = task.get("captures")
        capture_count = len(captures) if isinstance(captures, list) else 0
        return original_selector(
            task,
            max_prompt_tokens=max_prompt_tokens,
            limit=max(1, capture_count),
        )

    historical.select_probeable_requests = select_all_probeable_requests
    historical.MAX_CHECKPOINTS = None
    try:
        yield
    finally:
        historical.select_probeable_requests = original_selector
        historical.MAX_CHECKPOINTS = original_max_checkpoints


@contextmanager
def _no_clobber_output_patch() -> Iterator[None]:
    """Replace the historical replace-on-write helper for this V3-only call."""

    original_writer = historical.atomic_write_json

    def write_new(path: Path, value: Any) -> None:
        candidate = _lexical_absolute(path)
        historical.require(
            candidate.parent == V3_CACHE_ROOT,
            "V3 materializer attempted to write outside the dedicated cache root",
        )
        _write_new_json(candidate, value)

    historical.atomic_write_json = write_new
    try:
        yield
    finally:
        historical.atomic_write_json = original_writer


@contextmanager
def _image_provenance_patch(
    image_manifest_sha256_by_run: Mapping[Path, str],
) -> Iterator[None]:
    """Carry authenticated run-image bytes through source and combined records."""

    original_builder = historical.build_behavioral_bundle
    original_combiner = historical.combine_behavioral_bundles

    def build_with_image_provenance(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompts, summary = original_builder(**kwargs)
        run_root = Path(kwargs["run_root"]).resolve(strict=True)
        image_sha256 = image_manifest_sha256_by_run.get(run_root)
        historical.require(image_sha256 is not None, "source run lacks authenticated image provenance")
        historical.require("source_image_manifest_sha256" not in summary, "source image provenance field already exists")
        summary["source_image_manifest_sha256"] = image_sha256
        return prompts, summary

    def combine_with_image_provenance(
        sources: Sequence[Mapping[str, Any]], *, cohort_manifest_sha256: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompts, summary = original_combiner(
            sources, cohort_manifest_sha256=cohort_manifest_sha256
        )
        cohort_rows = [historical.mapping(value, "combined cohort") for value in summary["cohorts"]]
        source_hashes: list[str] = []
        for index, source in enumerate(sources):
            source_summary = historical.mapping(source.get("summary"), f"source summary {index}")
            image_sha256 = source_summary.get("source_image_manifest_sha256")
            historical.require(
                isinstance(image_sha256, str) and len(image_sha256) == 64,
                f"source summary {index} lacks image-manifest SHA-256",
            )
            historical.require(
                "source_image_manifest_sha256" not in cohort_rows[index],
                "combined image provenance field already exists",
            )
            cohort_rows[index]["source_image_manifest_sha256"] = image_sha256
            source_hashes.append(image_sha256)

        for prompt in prompts:
            metadata = historical.mapping(prompt.get("metadata"), "combined prompt metadata")
            cohort = historical.mapping(metadata.get("cohort"), "combined prompt cohort")
            cohort_index = cohort.get("index")
            historical.require(
                isinstance(cohort_index, int)
                and not isinstance(cohort_index, bool)
                and 0 <= cohort_index < len(source_hashes),
                "combined prompt cohort index is invalid",
            )
            image_sha256 = source_hashes[cohort_index]
            cohort["source_image_manifest_sha256"] = image_sha256
            provenance = historical.mapping(metadata.get("provenance"), "combined prompt provenance")
            combination = historical.mapping(provenance.get("combination"), "combined prompt combination")
            combination["source_image_manifest_sha256"] = image_sha256
            provenance["prompt_record_payload_sha256"] = historical._prompt_record_payload_sha256(prompt)

        summary_rows = [historical.mapping(value, "combined prompt summary row") for value in summary["prompts"]]
        historical.require(len(summary_rows) == len(prompts), "combined prompt summary coverage changed")
        for prompt, row in zip(prompts, summary_rows, strict=True):
            historical.require(row.get("id") == prompt.get("id"), "combined prompt summary order changed")
            row["prompt_record_payload_sha256"] = historical.mapping(
                historical.mapping(prompt.get("metadata"), "combined prompt metadata").get("provenance"),
                "combined prompt provenance",
            )["prompt_record_payload_sha256"]
        return prompts, summary

    historical.build_behavioral_bundle = build_with_image_provenance
    historical.combine_behavioral_bundles = combine_with_image_provenance
    try:
        yield
    finally:
        historical.build_behavioral_bundle = original_builder
        historical.combine_behavioral_bundles = original_combiner


def _run_historical_materialization(
    *, checker: Any, declaration: Any, delegated: Sequence[str]
) -> int:
    run_images = checker.validate_run_image_provenance(declaration)
    rows = [
        historical.mapping(row, f"V3 cohort row {index}")
        for index, row in enumerate(declaration.cohort["cohorts"])
    ]
    image_hash_by_run = {
        (V3_RUNS_ROOT / str(row["run_label"])).resolve(strict=True): image_sha256
        for row, image_sha256 in zip(
            rows, run_images.image_manifest_sha256s, strict=True
        )
    }
    with (
        _v3_manifest_validation_patch(V3_ACTION_PROTOCOL_PATH),
        _all_probeable_patch(),
        _image_provenance_patch(image_hash_by_run),
        _no_clobber_output_patch(),
    ):
        result = historical.main(list(delegated))
    historical.require(result == 0, "historical materializer returned a failure status")
    return int(result)


def _verification_argv(
    declaration: Any,
    *,
    output: Path,
    summary: Path,
    require_official_outcomes: bool,
) -> list[str]:
    rows = [
        historical.mapping(row, f"V3 cohort row {index}")
        for index, row in enumerate(declaration.cohort["cohorts"])
    ]
    historical.require(len(rows) == len(declaration.campaign_paths) == 2, "V3 A/B declaration changed")
    result: list[str] = []
    for row, campaign_path in zip(rows, declaration.campaign_paths, strict=True):
        result.extend(
            [
                "--cohort",
                str(campaign_path),
                str(V3_RUNS_ROOT / str(row["run_label"])),
            ]
        )
    result.extend(
        [
            "--cohort-manifest",
            str(declaration.cohort_path),
            "--action-protocol",
            str(V3_ACTION_PROTOCOL_PATH),
            "--template",
            str(V3_TEMPLATE_PATH),
            "--output",
            str(output),
            "--summary",
            str(summary),
        ]
    )
    if require_official_outcomes:
        result.append("--require-official-outcomes")
    return result


def verify_frozen_materialization(
    *,
    checker: Any,
    declaration: Any,
    receipt: Mapping[str, Any],
    prompts_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Independently rematerialize to unique staging files and compare exact bytes."""

    historical.require(
        prompts_path.resolve(strict=True) == (V3_CACHE_ROOT / "prompts.json").absolute()
        and summary_path.resolve(strict=True)
        == (V3_CACHE_ROOT / "prompts-summary.json").absolute(),
        "verification inputs are not the exact canonical materialization outputs",
    )
    invocation = historical.mapping(receipt.get("invocation"), "receipt invocation")
    historical.require(
        set(invocation) == {"all_probeable", "require_official_outcomes"}
        and invocation.get("all_probeable") is True
        and isinstance(invocation.get("require_official_outcomes"), bool),
        "receipt invocation changed",
    )
    outputs = historical.mapping(receipt.get("outputs"), "receipt outputs")
    expected_prompts = historical.mapping(
        outputs.get("prompt_bundle"), "receipt prompt bundle"
    )
    expected_summary = historical.mapping(
        outputs.get("prompt_summary"), "receipt prompt summary"
    )
    token = f"{os.getpid()}-{secrets.token_hex(16)}"
    staged_prompts = V3_CACHE_ROOT / f".materialization-verification-{token}-prompts.json"
    staged_summary = V3_CACHE_ROOT / f".materialization-verification-{token}-summary.json"
    for path in (staged_prompts, staged_summary):
        historical.require(not os.path.lexists(path), "materialization verification staging collision")
    try:
        delegated = _verification_argv(
            declaration,
            output=staged_prompts,
            summary=staged_summary,
            require_official_outcomes=bool(invocation["require_official_outcomes"]),
        )
        _run_historical_materialization(
            checker=checker,
            declaration=declaration,
            delegated=delegated,
        )
        checked = checker.validate_materialized_bundle(
            declaration,
            prompts_path=staged_prompts,
            summary_path=staged_summary,
        )
        staged_prompt_sha = historical.sha256_file(staged_prompts)
        staged_summary_sha = historical.sha256_file(staged_summary)
        historical.require(
            staged_prompt_sha
            == historical.sha256_file(prompts_path)
            == expected_prompts.get("sha256")
            == checked.get("prompt_bundle_sha256")
            and staged_summary_sha
            == historical.sha256_file(summary_path)
            == expected_summary.get("sha256")
            == checked.get("summary_sha256"),
            "deterministic rematerialization differs from the Git-frozen receipt outputs",
        )
        return {
            "algorithm": "trusted_pinned_v3_materializer_exact_byte_rematerialization_v1",
            "prompt_bundle_sha256": staged_prompt_sha,
            "prompt_summary_sha256": staged_summary_sha,
            "source_freeze_git_commit": receipt.get("source_freeze_git_commit"),
            "exact_match": True,
        }
    finally:
        for path in (staged_prompts, staged_summary):
            if path.is_file() and not path.is_symlink() and path.parent == V3_CACHE_ROOT:
                path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    checker = _load_pinned_checker()
    historical.require(
        Path(checker.V3_RUNS_ROOT) == V3_RUNS_ROOT
        and Path(checker.V3_OUTPUT_ROOT) == V3_CACHE_ROOT,
        "V3 checker mutable namespaces changed",
    )
    declaration = checker.validate_declaration(
        _require_exact_path(V3_COHORT_PATH, V3_COHORT_PATH, "V3 cohort manifest")
    )
    delegated, output, summary = _prepare_delegated_argv(
        sys.argv[1:] if argv is None else argv,
        declaration=declaration,
    )
    _ensure_cache_root()
    receipt_path = _validate_new_receipt_path()
    source_freeze_git_commit = checker.capture_clean_source_freeze()
    result = _run_historical_materialization(
        checker=checker,
        declaration=declaration,
        delegated=delegated,
    )
    checker.validate_materialized_bundle(
        declaration,
        prompts_path=output,
        summary_path=summary,
    )
    receipt = checker.build_materialization_receipt(
        declaration,
        prompts_path=output,
        summary_path=summary,
        invocation={
            "all_probeable": True,
            "require_official_outcomes": "--require-official-outcomes" in delegated,
        },
        source_freeze_git_commit=source_freeze_git_commit,
    )
    _write_new_json(receipt_path, receipt)
    checker.validate_materialization_receipt(
        declaration,
        prompts_path=output,
        summary_path=summary,
        receipt_path=receipt_path,
        require_git_frozen=False,
    )
    print(
        "wrote Git-freeze materialization receipt "
        f"{receipt_path} (sha256={historical.sha256_file(receipt_path)}); "
        "commit it as the sole change in the direct child of the recorded source-freeze HEAD before replay"
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
