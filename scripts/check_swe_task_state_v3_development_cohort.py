#!/usr/bin/env python3
"""Fail-closed validation for the identity-only V3 SWE development cohort."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
V3_RUNS_ROOT = ROOT / "runs/swe_state_interpreter_v3_development"
V3_OUTPUT_ROOT = ROOT / ".cache/swe_state_interpreter_v3_development"
DEFAULT_COHORT = ROOT / "configs/swe_task_state_v3_development_cohort.json"
DEFAULT_PROMPTS = V3_OUTPUT_ROOT / "prompts.json"
DEFAULT_PROMPT_SUMMARY = V3_OUTPUT_ROOT / "prompts-summary.json"
DEFAULT_MATERIALIZATION_RECEIPT = (
    ROOT / "validation/swe-task-state-v3-development-materialization.json"
)
EXPECTED_COHORT_PATH = "configs/swe_task_state_v3_development_cohort.json"
EXPECTED_PROOF_PATH = "validation/swe-task-state-v3-development-cohort-selection.json"
EXPECTED_ACTION_PATH = "configs/swe_task_state_v3_action_probes.json"
EXPECTED_TEMPLATE_PATH = "configs/qwen3-openai-codex.jinja"
EXPECTED_INTERPRETER_PATH = "configs/swe_task_state_interpreter_v3.json"
EXPECTED_ANALYZER_PATH = "scripts/analyze_swe_task_state_v3.py"
EXPECTED_V3_MATERIALIZER_PATH = "scripts/materialize_swe_state_interpreter_v3_probes.py"
EXPECTED_HISTORICAL_MATERIALIZER_PATH = "scripts/materialize_swe_behavioral_probes.py"
EXPECTED_REPLAY_PIPELINE_PATH = "scripts/swe_task_state_v3_replay_pipeline.py"
EXPECTED_REPLAY_WRAPPER_PATH = "scripts/run_swe_task_state_v3_replay.sh"
EXPECTED_MATERIALIZATION_RECEIPT_PATH = (
    "validation/swe-task-state-v3-development-materialization.json"
)
EXPECTED_IMAGE_PATHS = (
    "configs/swe_task_state_v3_development_a_image_digests.json",
    "configs/swe_task_state_v3_development_b_image_digests.json",
)
EXPECTED_IMAGE_SHA256S = (
    "7faf123a36533c91a344bb9ce1d5d77d22c05d1944a5ef5b4ca13b9932d1c56a",
    "9f2e70deb46db24033b1f359331f1a85fcf0a220cb2597e4c3beda964aaec404",
)
COHORT_KIND = "swe_verified_state_interpreter_v3_n60_cohort"
MATERIALIZATION_RECEIPT_KIND = (
    "swe_verified_state_interpreter_v3_n60_materialization_receipt"
)
PROOF_KIND = "swe_verified_state_interpreter_v3_n60_selection_proof"
CAMPAIGN_KIND = "swe_verified_behavioral_trajectory_campaign"
SEED = "jlens-state-interpreter-v3-development-20260718"
FROZEN_GIT_COMMIT = "06ddaf7583024571857ce3f7fbd0cb586a7ba90e"
ACTION_SHA256 = "0ebd258a2b46beb2a9be3d42cab24680803a2f971cb21e96acecb78e19cd81bf"
TEMPLATE_SHA256 = "c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da"
OFFICIAL_SET_SHA256 = "33e18be7a9bd9f674790b63ed4d0b3fb17c176994802e3062b7d5a430a4e7d16"
PRIOR_SET_SHA256 = "a1758311fab453f2508d6faee3d0825539faf19ea5e664d9ff5dfeb957583a1f"
UNUSED_SET_SHA256 = "a394e7df516fec0c5e577b3b6b08b2d7ada5616553020964cf022ef805368bbe"
ELIGIBLE_SET_SHA256 = "690414f93355255ab1a320d36809b716a52b7a2ba1d143305edc873625133674"
A_ORDERED_SHA256 = "c3b98512bf1ba4aa0e271f919eef8c3d3a7293d8920acc12f55b56e809d2bbd2"
A_SET_SHA256 = "01b42cf9ee057bb62513093ae22e55be0aa4ea8e91ee9071712ee1f8e962ae29"
B_ORDERED_SHA256 = "470bb5bfa9045b5a6ee2742550c7643b84054ccd7c7a44e7d2fa314158d6c5e4"
B_SET_SHA256 = "7d53617bb995a1b3c8171c2cd5e91e399349a80005345d68f3f0e1c004134e6d"
COMBINED_ORDERED_SHA256 = "6f91acea742ed3bb58b6276fdfee59a9b42e82071ada596b4034873d2da36388"
COMBINED_SET_SHA256 = "9d7c5af77bc6e477cb9297927373cbf943ead89ba1876c8c1b146dfc4948b7de"
TRACKED_PATHS_SHA256 = "503607a729eb120e2e6e7eb8e2118a05493760e9503cce372bbf9c041d5ae22d"
TRACKED_TREE_SHA256 = "1e355b922654105eb695451b0aeb99a1eeafc8a8483890ddbea10e73697c55c1"
INSTANCE_RE = re.compile(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40,64}")

SOURCE_FREEZE_PATHS = (
    "package.json",
    "package-lock.json",
    EXPECTED_COHORT_PATH,
    EXPECTED_PROOF_PATH,
    EXPECTED_ACTION_PATH,
    EXPECTED_TEMPLATE_PATH,
    EXPECTED_INTERPRETER_PATH,
    EXPECTED_ANALYZER_PATH,
    "configs/swe_task_state_interpreter_protocol.json",
    "configs/swe_binary_phase_interpreter_v2.json",
    "configs/swe_stage_action_probes.json",
    "configs/swe_behavioral_readout_protocol.json",
    EXPECTED_IMAGE_PATHS[0],
    EXPECTED_IMAGE_PATHS[1],
    "configs/swe_task_state_v3_development_a_campaign.json",
    "configs/swe_task_state_v3_development_b_campaign.json",
    "requirements-v3-state-interpreter.txt",
    "requirements-readout-v2.txt",
    "scripts/check_swe_task_state_v3_development_cohort.py",
    "scripts/analyze_swe_task_state_interpreter.py",
    "scripts/analyze_swe_binary_phase_v2.py",
    "scripts/swe_task_state_readout.py",
    "scripts/check_swe_task_state_validation_cohort.py",
    EXPECTED_V3_MATERIALIZER_PATH,
    EXPECTED_HISTORICAL_MATERIALIZER_PATH,
    "scripts/materialize_swe_multitask_c1_probes.py",
    "scripts/materialize_swe_multitask_initial_probes.py",
    "scripts/materialize_swe_jlens_prompts.py",
    "scripts/swe_task_contract.py",
    "scripts/run_swe_state_interpreter_v3_campaign.sh",
    "scripts/swe_state_interpreter_v3_campaign_contract.py",
    EXPECTED_REPLAY_PIPELINE_PATH,
    EXPECTED_REPLAY_WRAPPER_PATH,
    "scripts/run_jlens_nvfp4.sh",
    "scripts/run_jlens_nvfp4.py",
    "scripts/download_jlens.py",
    "scripts/verify_nvfp4_ste_artifact.py",
    "scripts/modelopt_checkpoint.py",
    "scripts/nvfp4_ste.py",
    "scripts/run_swe_verified.py",
    "scripts/qwen_code_proxy.py",
    "scripts/check_endpoint.py",
    "scripts/resolve_swe_image.py",
    "scripts/start_server.sh",
    "scripts/stop_server.sh",
    "scripts/serve_qwen36_27b_nvfp4_mtp.sh",
    "validation/vllm-freeze.txt",
    "validation/swe-freeze.txt",
)

EXPECTED_DATASET = {
    "repo_id": "princeton-nlp/SWE-bench_Verified",
    "revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
    "split": "test",
    "cached_arrow_relative_path": (
        "princeton-nlp___swe-bench_verified/default/0.0.0/"
        "c104f840cc67f8b6eec6f759ebc8b2693d585d4a/"
        "swe-bench_verified-test.arrow"
    ),
    "cached_arrow_sha256": "0d119efe73413554335bd410a04d82fd4a586bfd312cee677ee40af5de2ac46e",
    "official_instance_count": 500,
    "official_instance_ids_set_sha256": OFFICIAL_SET_SHA256,
}
EXPECTED_CAMPAIGN_DATASET = {
    "repo_id": EXPECTED_DATASET["repo_id"],
    "revision": EXPECTED_DATASET["revision"],
}
EXPECTED_GENERATION = {
    "model_repo_id": "nvidia/Qwen3.6-27B-NVFP4",
    "model_revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
    "served_model": "qwen3.6-27b-nvfp4",
    "qwen_code_version": "0.19.4",
    "max_model_len": 65536,
    "max_session_turns": 50,
    "agent_wall_seconds": 900,
    "retain_empty_predictions": True,
}
EXPECTED_REPOSITORY_COUNTS = {
    "astropy/astropy": 22,
    "django/django": 231,
    "matplotlib/matplotlib": 34,
    "mwaskom/seaborn": 2,
    "pallets/flask": 1,
    "psf/requests": 8,
    "pydata/xarray": 22,
    "pylint-dev/pylint": 10,
    "pytest-dev/pytest": 19,
    "scikit-learn/scikit-learn": 32,
    "sphinx-doc/sphinx": 44,
    "sympy/sympy": 75,
}
EXPECTED_ALLOCATION = [
    {"repository": "astropy/astropy", "unused_count": 13, "selected_quota": 6, "campaign_a_count": 3, "campaign_b_count": 3},
    {"repository": "django/django", "unused_count": 184, "selected_quota": 7, "campaign_a_count": 4, "campaign_b_count": 3},
    {"repository": "matplotlib/matplotlib", "unused_count": 22, "selected_quota": 6, "campaign_a_count": 3, "campaign_b_count": 3},
    {"repository": "psf/requests", "unused_count": 6, "selected_quota": 6, "campaign_a_count": 3, "campaign_b_count": 3},
    {"repository": "pydata/xarray", "unused_count": 11, "selected_quota": 6, "campaign_a_count": 3, "campaign_b_count": 3},
    {"repository": "pylint-dev/pylint", "unused_count": 5, "selected_quota": 5, "campaign_a_count": 2, "campaign_b_count": 3},
    {"repository": "pytest-dev/pytest", "unused_count": 8, "selected_quota": 6, "campaign_a_count": 3, "campaign_b_count": 3},
    {"repository": "scikit-learn/scikit-learn", "unused_count": 22, "selected_quota": 6, "campaign_a_count": 3, "campaign_b_count": 3},
    {"repository": "sphinx-doc/sphinx", "unused_count": 29, "selected_quota": 6, "campaign_a_count": 3, "campaign_b_count": 3},
    {"repository": "sympy/sympy", "unused_count": 45, "selected_quota": 6, "campaign_a_count": 3, "campaign_b_count": 3},
]
EXPECTED_DISJOINTNESS = {
    "campaign_a_intersection_campaign_b": 0,
    "campaign_a_intersection_prior_reserved": 0,
    "campaign_b_intersection_prior_reserved": 0,
    "combined_intersection_prior_reserved": 0,
    "combined_missing_from_official_set": 0,
}
EXPECTED_HASH_CONTRACT = {
    "algorithm": "sha256",
    "canonical_json": "sort_keys_true_separators_comma_colon_ensure_ascii_true",
    "encoding": "compact_ascii_canonical_json",
    "ordered_list_hash_input": "declared_instance_id_array",
    "set_hash_input": "lexically_sorted_unique_instance_id_array",
}


class CohortValidationError(ValueError):
    """Raised whenever any frozen identity or provenance check fails."""


@dataclass(frozen=True)
class DeclarationBundle:
    cohort_path: Path
    cohort: Mapping[str, Any]
    proof_path: Path
    proof: Mapping[str, Any]
    campaigns: tuple[Mapping[str, Any], Mapping[str, Any]]
    campaign_paths: tuple[Path, Path]
    campaign_ids: tuple[tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True)
class HistoricalResult:
    tracked_paths: tuple[str, ...]
    tracked_tree: tuple[Mapping[str, str], ...]
    matched_sources: tuple[Mapping[str, Any], ...]
    prior_ids: frozenset[str]


@dataclass(frozen=True)
class SelectionResult:
    official_ids: frozenset[str]
    repository_counts: Mapping[str, int]
    unused_ids: frozenset[str]
    eligible_ids: frozenset[str]
    allocation: tuple[Mapping[str, Any], ...]
    campaign_a: tuple[str, ...]
    campaign_b: tuple[str, ...]


@dataclass(frozen=True)
class RunImageProvenance:
    image_manifest_sha256s: tuple[str, str]
    runner_metadata: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class MaterializationReceiptAudit:
    receipt_path: Path
    receipt_sha256: str
    source_freeze_git_commit: str
    data_freeze_git_commit: str | None
    prompt_bundle_sha256: str
    summary_sha256: str
    source_file_count: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CohortValidationError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be nonempty text")
    return value


def positive_integer(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 1,
        f"{label} must be a positive integer",
    )
    return value


def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} fields changed")


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_file(path: Path, label: str) -> Any:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        return json.loads(
            path.read_bytes(),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CohortValidationError(f"non-finite JSON number in {label}: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CohortValidationError(f"cannot read {label}: {error}") from error


class _DuplicateTrackingDict(dict[str, Any]):
    def __init__(self, duplicates: list[str]) -> None:
        super().__init__()
        self._duplicates = duplicates

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self:
            self._duplicates.append(key)
        super().__setitem__(key, value)


class _HashingReader:
    def __init__(self, handle: Any, digest: Any) -> None:
        self._handle = handle
        self._digest = digest

    def read(self, size: int = -1) -> bytes:
        value = self._handle.read(size)
        self._digest.update(value)
        return value

    def readinto(self, buffer: Any) -> int:
        count = self._handle.readinto(buffer)
        if count:
            self._digest.update(memoryview(buffer)[:count])
        return count


def iter_strict_json_array(
    path: Path,
    label: str,
    *,
    sha256_sink: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Yield one top-level array item at a time with strict JSON rejection."""

    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        import ijson
    except ImportError as error:
        raise CohortValidationError(
            "streaming prompt validation requires pinned ijson 3.5.0"
        ) from error
    require(
        getattr(ijson, "__version__", None) == "3.5.0",
        "streaming prompt validation requires exact ijson 3.5.0",
    )
    duplicates: list[str] = []

    def map_factory() -> _DuplicateTrackingDict:
        return _DuplicateTrackingDict(duplicates)

    try:
        with path.open("rb") as handle:
            first_non_whitespace: int | None = None
            while first_non_whitespace is None:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                for byte in chunk:
                    if byte not in b" \t\r\n":
                        first_non_whitespace = byte
                        break
            require(
                first_non_whitespace == ord("["),
                f"{label} must be one top-level JSON array",
            )
            handle.seek(0)
            digest = hashlib.sha256()
            hashing_reader = _HashingReader(handle, digest)
            for value in ijson.items(
                hashing_reader,
                "item",
                map_type=map_factory,
                use_float=True,
            ):
                require(
                    not duplicates,
                    f"duplicate JSON key in {label}: {duplicates[0] if duplicates else '<none>'}",
                )
                yield value
            require(
                not duplicates,
                f"duplicate JSON key in {label}: {duplicates[0] if duplicates else '<none>'}",
            )
            if sha256_sink is not None:
                sha256_sink["sha256"] = digest.hexdigest()
    except CohortValidationError:
        raise
    except Exception as error:
        raise CohortValidationError(f"cannot stream strict {label}: {error}") from error


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise CohortValidationError(f"value is not canonical JSON: {error}") from error


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_ordered(values: Sequence[str]) -> str:
    return sha256_json(list(values))


def sha256_set(values: Sequence[str] | set[str] | frozenset[str]) -> str:
    return sha256_json(sorted(set(values)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return value


def repository_path(logical: Any, label: str, *, must_exist: bool = True) -> Path:
    value = nonempty_string(logical, label)
    require("\\" not in value and "\x00" not in value, f"unsafe {label}")
    relative = PurePosixPath(value)
    require(
        not relative.is_absolute()
        and relative.as_posix() == value
        and all(piece not in ("", ".", "..") for piece in relative.parts),
        f"non-canonical {label}: {value!r}",
    )
    path = ROOT.joinpath(*relative.parts)
    resolved = path.resolve(strict=must_exist)
    require(
        resolved.is_relative_to(ROOT)
        and resolved == path.absolute(),
        f"{label} traverses a symlinked or external path",
    )
    if must_exist:
        require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    else:
        require(not path.is_symlink(), f"{label} must not be a symlink")
    return resolved


def _root_relative_file(path: Path, label: str) -> str:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    resolved_root = ROOT.resolve(strict=True)
    resolved = path.resolve(strict=True)
    require(
        resolved == path.absolute() and resolved.is_relative_to(resolved_root),
        f"{label} traverses a symlink or escapes the repository",
    )
    return resolved.relative_to(resolved_root).as_posix()


def _receipt_git_output(arguments: Sequence[str], label: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise CohortValidationError(f"cannot execute Git for {label}: {error}") from error
    require(
        result.returncode == 0,
        f"Git {label} failed: {result.stderr.decode('utf-8', errors='replace').strip()}",
    )
    return result.stdout


def _git_head() -> str:
    value = _receipt_git_output(["rev-parse", "--verify", "HEAD^{commit}"], "HEAD lookup").decode(
        "ascii"
    ).strip()
    require(re.fullmatch(r"[0-9a-f]{40}", value) is not None, "Git HEAD is not SHA-1")
    return value


def _require_clean_tracked_worktree() -> None:
    status = _receipt_git_output(
        ["status", "--porcelain=v1", "--untracked-files=no"],
        "tracked-worktree status",
    )
    require(status == b"", "tracked working tree or index differs from HEAD")


def _git_file_at_commit(commit: str, logical_path: str) -> bytes:
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid source-freeze commit")
    repository_path(logical_path, f"source-freeze file {logical_path}")
    return _receipt_git_output(
        ["show", f"{commit}:{logical_path}"],
        f"read {logical_path} at source-freeze commit",
    )


def _repository_source_records(source_freeze_git_commit: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    require(
        len(SOURCE_FREEZE_PATHS) == len(set(SOURCE_FREEZE_PATHS)),
        "source-freeze path declaration repeats",
    )
    for logical_path in SOURCE_FREEZE_PATHS:
        path = repository_path(logical_path, f"source-freeze file {logical_path}")
        current = path.read_bytes()
        frozen = _git_file_at_commit(source_freeze_git_commit, logical_path)
        require(current == frozen, f"source file differs from source-freeze commit: {logical_path}")
        records.append(
            {
                "path": logical_path,
                "sha256": hashlib.sha256(current).hexdigest(),
                "bytes": len(current),
            }
        )
    return records


def capture_clean_source_freeze() -> str:
    """Return HEAD only when all tracked source bytes are already frozen there."""

    _require_clean_tracked_worktree()
    commit = _git_head()
    _repository_source_records(commit)
    return commit


def _inventory_regular_tree(root: Path, label: str) -> list[dict[str, Any]]:
    require(root.is_dir() and not root.is_symlink(), f"{label} is not a regular directory")
    resolved_root = ROOT.resolve(strict=True)
    require(
        root.resolve(strict=True) == root.absolute()
        and root.resolve(strict=True).is_relative_to(resolved_root),
        f"{label} traverses a symlink or escapes the repository",
    )
    records: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        require(
            directory.is_dir()
            and not directory.is_symlink()
            and directory.resolve(strict=True) == directory.absolute(),
            f"{label} contains a symlinked directory",
        )
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise CohortValidationError(f"cannot enumerate {label}: {error}") from error
        for entry in entries:
            path = Path(entry.path)
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as error:
                raise CohortValidationError(f"cannot stat {label} entry {path}: {error}") from error
            require(not stat.S_ISLNK(mode), f"{label} contains a symlink: {path}")
            if stat.S_ISDIR(mode):
                visit(path)
            elif stat.S_ISREG(mode):
                logical = _root_relative_file(path, f"{label} file")
                records.append(
                    {
                        "path": logical,
                        "sha256": sha256_file(path),
                        "bytes": path.stat(follow_symlinks=False).st_size,
                    }
                )
            else:
                raise CohortValidationError(f"{label} contains a non-regular entry: {path}")

    visit(root)
    require(bool(records), f"{label} contains no regular files")
    require(
        [record["path"] for record in records]
        == sorted({str(record["path"]) for record in records}),
        f"{label} inventory order or uniqueness changed",
    )
    return records


def _validate_binding(
    value: Any, *, label: str, expected_path: str, expected_sha256: str | None = None
) -> tuple[Path, str]:
    binding = mapping(value, label)
    exact_keys(binding, {"path", "sha256"}, label)
    require(binding.get("path") == expected_path, f"{label} path changed")
    declared_hash = validate_sha256(binding.get("sha256"), f"{label} SHA-256")
    if expected_sha256 is not None:
        require(declared_hash == expected_sha256, f"{label} frozen SHA-256 changed")
    path = repository_path(binding.get("path"), f"{label} path")
    require(sha256_file(path) == declared_hash, f"{label} bytes changed")
    return path, declared_hash


def _instance_ids(value: Any, label: str, *, count: int) -> tuple[str, ...]:
    values = sequence(value, label)
    require(len(values) == count, f"{label} must contain exactly {count} values")
    require(
        all(isinstance(item, str) and INSTANCE_RE.fullmatch(item) for item in values),
        f"{label} contains an invalid SWE instance ID",
    )
    require(len(values) == len(set(values)), f"{label} repeats an instance ID")
    return tuple(values)


def _validate_image_binding(
    value: Any,
    *,
    index: int,
    label: str,
    selected_ids: Sequence[str],
) -> None:
    binding = mapping(value, label)
    exact_keys(binding, {"path", "sha256", "generation_authorized"}, label)
    require(
        binding
        == {
            "path": EXPECTED_IMAGE_PATHS[index],
            "sha256": EXPECTED_IMAGE_SHA256S[index],
            "generation_authorized": True,
        },
        f"{label} is not the exact finalized image gate",
    )
    path = repository_path(binding["path"], f"{label} path")
    require(sha256_file(path) == binding["sha256"], f"{label} bytes changed")
    registry = mapping(strict_json_file(path, label), label)
    exact_keys(registry, {"schema_version", "images"}, label)
    require(registry.get("schema_version") == 1, f"{label} schema changed")
    images = mapping(registry.get("images"), f"{label} images")
    require(set(images) == set(selected_ids), f"{label} coverage differs from campaign")
    for instance_id in selected_ids:
        architectures = mapping(images.get(instance_id), f"{label} {instance_id}")
        exact_keys(architectures, {"x86_64"}, f"{label} {instance_id}")
        pin = mapping(architectures.get("x86_64"), f"{label} {instance_id} x86_64")
        exact_keys(pin, {"reference", "image_id"}, f"{label} {instance_id} x86_64")
        image_id = nonempty_string(pin.get("image_id"), f"{label} {instance_id} image ID")
        reference = nonempty_string(pin.get("reference"), f"{label} {instance_id} reference")
        require(
            re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is not None,
            f"{label} {instance_id} image ID is invalid",
        )
        expected_prefix = (
            "swebench/sweb.eval.x86_64."
            + instance_id.replace("__", "_1776_")
            + "@sha256:"
        )
        require(
            reference.startswith(expected_prefix)
            and re.fullmatch(r"swebench/sweb\.eval\.x86_64\.[A-Za-z0-9_.-]+@sha256:[0-9a-f]{64}", reference) is not None,
            f"{label} {instance_id} RepoDigest is invalid",
        )


def validate_campaign_declaration(
    campaign: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    campaign_index: int,
) -> tuple[str, ...]:
    """Validate one immutable N=30 campaign and its manifest-row binding."""

    label = f"campaign {campaign_index}"
    exact_keys(
        campaign,
        {"schema_version", "kind", "dataset", "selection", "generation", "instance_ids"},
        label,
    )
    require(campaign.get("schema_version") == 1, f"{label} schema changed")
    require(campaign.get("kind") == CAMPAIGN_KIND, f"{label} kind changed")
    require(campaign.get("dataset") == EXPECTED_CAMPAIGN_DATASET, f"{label} dataset pin changed")
    require(campaign.get("generation") == EXPECTED_GENERATION, f"{label} generation pins changed")
    selection = mapping(campaign.get("selection"), f"{label} selection")
    exact_keys(
        selection,
        {
            "lens_outputs_used",
            "official_outcomes_used",
            "gold_patches_used",
            "problem_statements_used",
            "seed_text",
            "rule",
            "purpose",
            "ordered_instance_ids_sha256",
            "instance_ids_set_sha256",
        },
        f"{label} selection",
    )
    require(
        selection.get("lens_outputs_used") is False
        and selection.get("official_outcomes_used") is False
        and selection.get("gold_patches_used") is False
        and selection.get("problem_statements_used") is False,
        f"{label} selection used forbidden non-identity evidence",
    )
    require(selection.get("seed_text") == SEED, f"{label} seed changed")
    instance_ids = _instance_ids(campaign.get("instance_ids"), f"{label} instances", count=30)
    ordered_hash = sha256_ordered(instance_ids)
    set_hash = sha256_set(instance_ids)
    require(
        selection.get("ordered_instance_ids_sha256") == ordered_hash
        and row.get("ordered_instance_ids_sha256") == ordered_hash,
        f"{label} ordered instance hash changed",
    )
    require(
        selection.get("instance_ids_set_sha256") == set_hash
        and row.get("instance_ids_set_sha256") == set_hash,
        f"{label} instance-set hash changed",
    )
    expected_hashes = (
        (A_ORDERED_SHA256, A_SET_SHA256),
        (B_ORDERED_SHA256, B_SET_SHA256),
    )[campaign_index]
    require((ordered_hash, set_hash) == expected_hashes, f"{label} frozen membership changed")
    require(row.get("instance_ids") == list(instance_ids), f"{label} manifest coverage changed")
    return instance_ids


def _validate_interpreter_upstream_contract(
    interpreter: Mapping[str, Any], *, proof_sha256: str
) -> None:
    """Enforce the acyclic proof/checker -> protocol dependency direction."""

    require(interpreter.get("schema_version") == 1, "V3 interpreter protocol schema changed")
    require(interpreter.get("id") == "swe-task-state-interpreter-v3", "V3 interpreter protocol ID changed")
    pins = mapping(interpreter.get("pins"), "V3 interpreter pins")
    require(
        pins.get("v3_action_protocol_sha256") == ACTION_SHA256,
        "V3 interpreter action-protocol pin changed",
    )
    require(
        pins.get("materialized_bundle_checker_sha256")
        == sha256_file(Path(__file__).resolve()),
        "V3 interpreter does not pin the exact cohort checker bytes",
    )
    for pin_name, logical_path in (
        ("v3_materializer_sha256", EXPECTED_V3_MATERIALIZER_PATH),
        ("historical_materializer_sha256", EXPECTED_HISTORICAL_MATERIALIZER_PATH),
        ("replay_pipeline_sha256", EXPECTED_REPLAY_PIPELINE_PATH),
        ("replay_shell_wrapper_sha256", EXPECTED_REPLAY_WRAPPER_PATH),
    ):
        path = repository_path(logical_path, f"V3 {pin_name} path")
        require(
            pins.get(pin_name) == sha256_file(path),
            f"V3 interpreter {pin_name} does not pin exact bytes",
        )
    contract = mapping(
        interpreter.get("development_data_contract"),
        "V3 interpreter development data contract",
    )
    require(
        contract.get("cohort_manifest_path") == EXPECTED_COHORT_PATH
        and contract.get("cohort_kind") == COHORT_KIND
        and contract.get("selection_proof_path") == EXPECTED_PROOF_PATH
        and contract.get("selection_proof_sha256") == proof_sha256
        and contract.get("validator")
        == (
            "scripts/check_swe_task_state_v3_development_cohort.py"
            "::validate_materialized_bundle"
        )
        and contract.get("task_count") == 60
        and contract.get("cohort_ids_in_order") == ["development_a", "development_b"]
        and contract.get("campaign_paths_in_order")
        == [
            "configs/swe_task_state_v3_development_a_campaign.json",
            "configs/swe_task_state_v3_development_b_campaign.json",
        ]
        and contract.get("campaign_sha256s_in_order")
        == [
            "4379a32a60d3772421239c5bd2c27fa07e56089810a0adcd71bfb951caa0a0a2",
            "98dab25c7f11e46fe7b29f72a15535d11c276f4d68b9b68c01ba7f1bfad53387",
        ]
        and contract.get("combined_order") == "campaign_a_then_campaign_b"
        and contract.get("combined_ordered_instance_ids_sha256")
        == COMBINED_ORDERED_SHA256
        and contract.get("combined_instance_ids_set_sha256") == COMBINED_SET_SHA256
        and contract.get(
            "every_prompt_payload_hash_and_full_task_cohort_campaign_order_must_validate"
        )
        is True,
        "V3 interpreter development data/checker contract changed",
    )


def _validate_proof_declaration(
    proof: Mapping[str, Any],
    *,
    campaign_rows: Sequence[Mapping[str, Any]],
    campaign_paths: Sequence[Path],
    campaign_ids: Sequence[tuple[str, ...]],
    action_binding: Mapping[str, Any],
) -> None:
    exact_keys(
        proof,
        {
            "schema_version",
            "kind",
            "status",
            "hash_contract",
            "dataset",
            "identity_only_selection",
            "historical_exclusion",
            "selection",
            "campaigns",
            "disjointness",
            "bindings",
        },
        "selection proof",
    )
    require(proof.get("schema_version") == 1, "selection proof schema changed")
    require(proof.get("kind") == PROOF_KIND, "selection proof kind changed")
    require(
        proof.get("status") == "frozen_before_v3_development_generation_or_lens_replay",
        "selection proof freeze status changed",
    )
    require(proof.get("hash_contract") == EXPECTED_HASH_CONTRACT, "selection proof hash contract changed")
    proof_dataset = mapping(proof.get("dataset"), "selection proof dataset")
    require(
        proof_dataset
        == {
            **EXPECTED_DATASET,
            "official_repository_counts": EXPECTED_REPOSITORY_COUNTS,
        },
        "selection proof dataset identity changed",
    )
    require(
        proof.get("identity_only_selection")
        == {
            "columns_read": ["instance_id", "repo"],
            "gold_patches_read": False,
            "lens_outputs_read": False,
            "official_outcomes_read": False,
            "problem_statements_read": False,
        },
        "selection proof identity-only declaration changed",
    )
    require(proof.get("disjointness") == EXPECTED_DISJOINTNESS, "selection proof disjointness declaration changed")

    historical = mapping(proof.get("historical_exclusion"), "historical exclusion")
    exact_keys(
        historical,
        {
            "git_commit",
            "git_tree_root",
            "method",
            "tracked_config_count",
            "tracked_config_paths_sha256",
            "tracked_config_tree_sha256",
            "matched_source_files",
            "prior_reserved_instance_count",
            "prior_reserved_instance_ids",
            "prior_reserved_instance_ids_set_sha256",
            "working_tree_files_read",
        },
        "historical exclusion",
    )
    require(
        historical.get("git_commit") == FROZEN_GIT_COMMIT
        and historical.get("git_tree_root") == "configs"
        and historical.get("tracked_config_count") == 26
        and historical.get("tracked_config_paths_sha256") == TRACKED_PATHS_SHA256
        and historical.get("tracked_config_tree_sha256") == TRACKED_TREE_SHA256
        and historical.get("working_tree_files_read") is False,
        "historical exclusion frozen tree identity changed",
    )
    prior_ids = _instance_ids(
        historical.get("prior_reserved_instance_ids"),
        "prior/reserved instance IDs",
        count=154,
    )
    require(list(prior_ids) == sorted(prior_ids), "prior/reserved IDs are not canonical")
    require(
        historical.get("prior_reserved_instance_count") == 154
        and historical.get("prior_reserved_instance_ids_set_sha256") == PRIOR_SET_SHA256
        and sha256_set(prior_ids) == PRIOR_SET_SHA256,
        "prior/reserved instance identity changed",
    )
    matched = [
        mapping(row, f"matched source {index}")
        for index, row in enumerate(sequence(historical.get("matched_source_files"), "matched sources"))
    ]
    matched_paths: list[str] = []
    for index, row in enumerate(matched):
        exact_keys(row, {"path", "git_blob", "match_count"}, f"matched source {index}")
        path = nonempty_string(row.get("path"), f"matched source {index} path")
        require(path.startswith("configs/"), f"matched source {index} escaped configs")
        require(
            isinstance(row.get("git_blob"), str)
            and GIT_OBJECT_RE.fullmatch(row["git_blob"]) is not None,
            f"matched source {index} Git blob is invalid",
        )
        positive_integer(row.get("match_count"), f"matched source {index} count")
        matched_paths.append(path)
    require(matched_paths == sorted(set(matched_paths)), "matched source order or uniqueness changed")

    selection = mapping(proof.get("selection"), "selection proof rule")
    expected_selection = {
        "seed_text": SEED,
        "target_instance_count": 60,
        "eligible_repository_rule": "at least four unused tasks so both campaigns can receive at least two",
        "eligible_repository_count": 10,
        "base_quota_per_repository": 6,
        "remaining_quota_allocation": "unused count descending then repository lexical order",
        "within_repository_order": "ascending SHA256(seed_text + NUL + instance_id), then instance_id",
        "campaign_assignment": "repository lexical order; each repository starts in currently smaller campaign (A wins ties), then alternates ranked tasks",
        "unused_instance_count": 346,
        "unused_instance_ids_set_sha256": UNUSED_SET_SHA256,
        "eligible_common_campaign_instance_count": 345,
        "eligible_common_campaign_instance_ids_set_sha256": ELIGIBLE_SET_SHA256,
        "repository_allocation": EXPECTED_ALLOCATION,
    }
    require(selection == expected_selection, "selection proof deterministic rule changed")

    combined = (*campaign_ids[0], *campaign_ids[1])
    proof_campaigns = mapping(proof.get("campaigns"), "selection proof campaigns")
    exact_keys(
        proof_campaigns,
        {"a", "b", "combined_instance_count", "combined_order", "combined_ordered_instance_ids_sha256", "combined_instance_ids_set_sha256"},
        "selection proof campaigns",
    )
    for index, key in enumerate(("a", "b")):
        proof_row = mapping(proof_campaigns.get(key), f"selection proof campaign {key}")
        exact_keys(
            proof_row,
            {"path", "sha256", "instance_count", "ordered_instance_ids_sha256", "instance_ids_set_sha256", "image_registry"},
            f"selection proof campaign {key}",
        )
        require(
            proof_row.get("path") == campaign_rows[index].get("campaign_path")
            and proof_row.get("sha256") == sha256_file(campaign_paths[index])
            and proof_row.get("instance_count") == 30
            and proof_row.get("ordered_instance_ids_sha256") == sha256_ordered(campaign_ids[index])
            and proof_row.get("instance_ids_set_sha256") == sha256_set(campaign_ids[index]),
            f"selection proof campaign {key} binding changed",
        )
        _validate_image_binding(
            proof_row.get("image_registry"),
            index=index,
            label=f"selection proof campaign {key} image registry",
            selected_ids=campaign_ids[index],
        )
        require(
            proof_row.get("image_registry") == campaign_rows[index].get("image_registry"),
            f"campaign {key} image gate differs between proof and cohort",
        )
    require(
        proof_campaigns.get("combined_instance_count") == 60
        and proof_campaigns.get("combined_order") == "campaign_a_then_campaign_b"
        and proof_campaigns.get("combined_ordered_instance_ids_sha256") == sha256_ordered(combined) == COMBINED_ORDERED_SHA256
        and proof_campaigns.get("combined_instance_ids_set_sha256") == sha256_set(combined) == COMBINED_SET_SHA256,
        "selection proof combined campaign identity changed",
    )
    require(not (set(campaign_ids[0]) & set(campaign_ids[1])), "campaigns overlap")
    require(not (set(combined) & set(prior_ids)), "campaigns overlap prior/reserved IDs")

    bindings = mapping(proof.get("bindings"), "selection proof bindings")
    exact_keys(bindings, {"action_protocol"}, "selection proof bindings")
    require(bindings.get("action_protocol") == action_binding, "proof action-protocol binding differs")


def validate_declaration(cohort_path: Path = DEFAULT_COHORT) -> DeclarationBundle:
    """Validate every checked-in selection, campaign, and scientific binding."""

    supplied_cohort_path = cohort_path.expanduser()
    require(not supplied_cohort_path.is_symlink(), "cohort manifest must not be a symlink")
    cohort_path = supplied_cohort_path.resolve(strict=True)
    require(
        cohort_path == (ROOT / EXPECTED_COHORT_PATH).resolve(strict=True),
        "cohort manifest path is not the exact checked-in V3 N=60 manifest",
    )
    require(cohort_path.is_file() and not cohort_path.is_symlink(), "cohort manifest is not a regular file")
    cohort = mapping(strict_json_file(cohort_path, "V3 cohort manifest"), "V3 cohort manifest")
    exact_keys(
        cohort,
        {
            "schema_version",
            "kind",
            "status",
            "lens_outputs_used_for_selection",
            "official_outcomes_used_for_selection",
            "dataset",
            "selection",
            "disjointness",
            "pins",
            "selection_proof",
            "action_protocol",
            "chat_template",
            "scientific_artifacts",
            "cohorts",
            "instance_ids",
        },
        "V3 cohort manifest",
    )
    require(cohort.get("schema_version") == 1, "V3 cohort schema changed")
    require(cohort.get("kind") == COHORT_KIND, "V3 cohort kind changed")
    require(
        cohort.get("status") == "frozen_before_v3_development_generation_or_lens_replay",
        "V3 cohort freeze status changed",
    )
    require(
        cohort.get("lens_outputs_used_for_selection") is False
        and cohort.get("official_outcomes_used_for_selection") is False,
        "V3 cohort selection used forbidden post-generation evidence",
    )
    require(cohort.get("dataset") == EXPECTED_DATASET, "V3 cohort dataset/cache identity changed")
    require(cohort.get("disjointness") == EXPECTED_DISJOINTNESS, "V3 cohort disjointness declaration changed")
    expected_selection = {
        "seed_text": SEED,
        "target_instance_count": 60,
        "prior_reserved_instance_count": 154,
        "prior_reserved_instance_ids_set_sha256": PRIOR_SET_SHA256,
        "unused_instance_count": 346,
        "unused_instance_ids_set_sha256": UNUSED_SET_SHA256,
        "eligible_common_campaign_instance_count": 345,
        "eligible_common_campaign_instance_ids_set_sha256": ELIGIBLE_SET_SHA256,
        "campaign_a_ordered_instance_ids_sha256": A_ORDERED_SHA256,
        "campaign_a_instance_ids_set_sha256": A_SET_SHA256,
        "campaign_b_ordered_instance_ids_sha256": B_ORDERED_SHA256,
        "campaign_b_instance_ids_set_sha256": B_SET_SHA256,
        "combined_order": "campaign_a_then_campaign_b",
        "combined_ordered_instance_ids_sha256": COMBINED_ORDERED_SHA256,
        "combined_instance_ids_set_sha256": COMBINED_SET_SHA256,
        "identity_only_columns": ["instance_id", "repo"],
        "gold_patches_used": False,
        "problem_statements_used": False,
    }
    require(cohort.get("selection") == expected_selection, "V3 cohort selection identity changed")

    action_path, action_hash = _validate_binding(
        cohort.get("action_protocol"),
        label="cohort action protocol",
        expected_path=EXPECTED_ACTION_PATH,
        expected_sha256=ACTION_SHA256,
    )
    del action_path
    _template_path, template_hash = _validate_binding(
        cohort.get("chat_template"),
        label="cohort chat template",
        expected_path=EXPECTED_TEMPLATE_PATH,
        expected_sha256=TEMPLATE_SHA256,
    )
    scientific = mapping(cohort.get("scientific_artifacts"), "cohort scientific artifacts")
    exact_keys(scientific, {"interpreter_protocol", "analyzer"}, "cohort scientific artifacts")
    interpreter_path, interpreter_hash = _validate_binding(
        scientific.get("interpreter_protocol"),
        label="V3 interpreter protocol",
        expected_path=EXPECTED_INTERPRETER_PATH,
    )
    _analyzer_path, analyzer_hash = _validate_binding(
        scientific.get("analyzer"),
        label="V3 analyzer",
        expected_path=EXPECTED_ANALYZER_PATH,
    )

    rows = [
        mapping(row, f"cohort row {index}")
        for index, row in enumerate(sequence(cohort.get("cohorts"), "cohort rows"))
    ]
    require(len(rows) == 2, "V3 cohort must contain exactly two campaigns")
    expected_rows = (
        ("development_a", "configs/swe_task_state_v3_development_a_campaign.json", "swe_task_state_v3_development_a_20260719"),
        ("development_b", "configs/swe_task_state_v3_development_b_campaign.json", "swe_task_state_v3_development_b_20260719"),
    )
    campaigns: list[Mapping[str, Any]] = []
    campaign_paths: list[Path] = []
    campaign_ids: list[tuple[str, ...]] = []
    for index, (row, expected) in enumerate(zip(rows, expected_rows, strict=True)):
        exact_keys(
            row,
            {"id", "campaign_path", "campaign_sha256", "run_label", "ordered_instance_ids_sha256", "instance_ids_set_sha256", "image_registry", "instance_ids"},
            f"cohort row {index}",
        )
        require(
            (row.get("id"), row.get("campaign_path"), row.get("run_label")) == expected,
            f"cohort row {index} identity changed",
        )
        campaign_path = repository_path(row.get("campaign_path"), f"campaign {index} path")
        campaign_hash = validate_sha256(row.get("campaign_sha256"), f"campaign {index} SHA-256")
        require(sha256_file(campaign_path) == campaign_hash, f"campaign {index} bytes changed")
        campaign = mapping(strict_json_file(campaign_path, f"campaign {index}"), f"campaign {index}")
        ids = validate_campaign_declaration(campaign, row=row, campaign_index=index)
        _validate_image_binding(
            row.get("image_registry"),
            index=index,
            label=f"campaign {index} image registry",
            selected_ids=ids,
        )
        campaigns.append(campaign)
        campaign_paths.append(campaign_path)
        campaign_ids.append(ids)
    combined = (*campaign_ids[0], *campaign_ids[1])
    require(len(set(combined)) == 60, "V3 campaigns overlap")
    require(cohort.get("instance_ids") == list(combined), "combined V3 task order changed")
    require(
        sha256_ordered(combined) == COMBINED_ORDERED_SHA256
        and sha256_set(combined) == COMBINED_SET_SHA256,
        "combined V3 membership changed",
    )

    proof_binding = mapping(cohort.get("selection_proof"), "cohort selection proof")
    proof_path, proof_hash = _validate_binding(
        proof_binding,
        label="cohort selection proof",
        expected_path=EXPECTED_PROOF_PATH,
    )
    proof = mapping(strict_json_file(proof_path, "selection proof"), "selection proof")
    _validate_proof_declaration(
        proof,
        campaign_rows=rows,
        campaign_paths=campaign_paths,
        campaign_ids=campaign_ids,
        action_binding=mapping(cohort.get("action_protocol"), "cohort action protocol"),
    )
    interpreter = mapping(
        strict_json_file(interpreter_path, "V3 interpreter protocol"),
        "V3 interpreter protocol",
    )
    _validate_interpreter_upstream_contract(interpreter, proof_sha256=proof_hash)
    pins = mapping(cohort.get("pins"), "cohort pins")
    exact_keys(
        pins,
        {
            "action_protocol_sha256",
            "chat_template_sha256",
            "selection_proof_sha256",
            "interpreter_protocol_sha256",
            "analyzer_sha256",
            "development_a_image_registry_sha256",
            "development_b_image_registry_sha256",
        },
        "cohort pins",
    )
    require(
        pins
        == {
            "action_protocol_sha256": action_hash,
            "chat_template_sha256": template_hash,
            "selection_proof_sha256": proof_hash,
            "interpreter_protocol_sha256": interpreter_hash,
            "analyzer_sha256": analyzer_hash,
            "development_a_image_registry_sha256": EXPECTED_IMAGE_SHA256S[0],
            "development_b_image_registry_sha256": EXPECTED_IMAGE_SHA256S[1],
        },
        "cohort duplicate pins differ from their byte bindings",
    )
    return DeclarationBundle(
        cohort_path=cohort_path,
        cohort=cohort,
        proof_path=proof_path,
        proof=proof,
        campaigns=(campaigns[0], campaigns[1]),
        campaign_paths=(campaign_paths[0], campaign_paths[1]),
        campaign_ids=(campaign_ids[0], campaign_ids[1]),
    )


def scan_historical_config_payloads(
    entries: Sequence[tuple[str, str, bytes]],
    *,
    official_ids: frozenset[str],
) -> HistoricalResult:
    """Derive the prior universe from supplied frozen Git blobs, never live files."""

    normalized = sorted(entries, key=lambda entry: entry[0])
    paths = tuple(path for path, _blob, _payload in normalized)
    require(len(paths) == len(set(paths)), "frozen config tree repeats a path")
    tree: list[Mapping[str, str]] = []
    matched_sources: list[Mapping[str, Any]] = []
    prior: set[str] = set()
    encoded_ids = [(instance_id, instance_id.encode("ascii")) for instance_id in sorted(official_ids)]
    for path, blob, payload in normalized:
        require(
            path.startswith("configs/")
            and isinstance(blob, str)
            and GIT_OBJECT_RE.fullmatch(blob) is not None
            and isinstance(payload, bytes),
            f"invalid frozen config entry: {path!r}",
        )
        tree.append({"path": path, "git_blob": blob})
        matches = [instance_id for instance_id, token in encoded_ids if token in payload]
        if matches:
            prior.update(matches)
            matched_sources.append(
                {"path": path, "git_blob": blob, "match_count": len(matches)}
            )
    return HistoricalResult(
        tracked_paths=paths,
        tracked_tree=tuple(tree),
        matched_sources=tuple(matched_sources),
        prior_ids=frozenset(prior),
    )


def _git_output(git_bin: str, arguments: Sequence[str], label: str) -> bytes:
    result = subprocess.run(
        [git_bin, *arguments],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(
        result.returncode == 0,
        f"{label} failed with status {result.returncode}: "
        f"{result.stderr.decode('utf-8', errors='replace').strip()}",
    )
    return result.stdout


def load_frozen_historical(
    *,
    official_ids: frozenset[str],
    git_bin: str = "git",
    commit: str = FROZEN_GIT_COMMIT,
) -> HistoricalResult:
    """Read only blobs under configs at the exact pinned commit."""

    resolved = _git_output(
        git_bin, ["rev-parse", f"{commit}^{{commit}}"], "frozen commit resolution"
    ).decode("ascii").strip()
    require(resolved == commit, "frozen historical commit does not resolve exactly")
    raw_tree = _git_output(
        git_bin,
        ["ls-tree", "-r", "-z", "--full-tree", commit, "--", "configs"],
        "frozen config tree listing",
    )
    entries: list[tuple[str, str, bytes]] = []
    for record in raw_tree.rstrip(b"\x00").split(b"\x00"):
        require(bool(record), "frozen config tree contains an empty row")
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, raw_blob = metadata.split(b" ", 2)
            path = raw_path.decode("utf-8")
            blob = raw_blob.decode("ascii")
        except (ValueError, UnicodeError) as error:
            raise CohortValidationError(f"malformed frozen config tree row: {error}") from error
        require(mode == b"100644" and object_type == b"blob", f"unsupported frozen config entry: {path}")
        payload = _git_output(git_bin, ["cat-file", "blob", blob], f"frozen blob {path}")
        entries.append((path, blob, payload))
    return scan_historical_config_payloads(entries, official_ids=official_ids)


def reproduce_selection(
    identity_rows: Sequence[tuple[str, str]],
    *,
    prior_ids: frozenset[str],
    seed: str = SEED,
    target_count: int = 60,
    repository_minimum: int = 4,
    base_quota: int = 6,
) -> SelectionResult:
    """Reproduce repository quotas, hash ranks, and balanced A/B assignment."""

    require(bool(identity_rows), "official identity rows are empty")
    require(target_count >= 2 and target_count % 2 == 0, "target count must be positive and even")
    require(repository_minimum >= 2 and base_quota >= 1, "selection thresholds are invalid")
    identities: dict[str, str] = {}
    repository_counts: Counter[str] = Counter()
    for index, (instance_id, repository) in enumerate(identity_rows):
        require(
            isinstance(instance_id, str) and INSTANCE_RE.fullmatch(instance_id) is not None,
            f"official identity row {index} has an invalid instance ID",
        )
        require(isinstance(repository, str) and bool(repository), f"official identity row {index} has no repository")
        require(instance_id not in identities, f"official identity rows repeat {instance_id}")
        identities[instance_id] = repository
        repository_counts[repository] += 1
    official = frozenset(identities)
    require(prior_ids <= official, "prior/reserved set contains a non-official ID")
    unused = frozenset(official - prior_ids)
    grouped: dict[str, list[str]] = defaultdict(list)
    for instance_id in unused:
        grouped[identities[instance_id]].append(instance_id)
    eligible = {
        repository: values
        for repository, values in grouped.items()
        if len(values) >= repository_minimum
    }
    eligible_ids = frozenset(
        instance_id for values in eligible.values() for instance_id in values
    )
    require(eligible, "selection has no eligible repositories")
    for repository, values in eligible.items():
        values.sort(
            key=lambda instance_id: (
                hashlib.sha256(
                    seed.encode("utf-8") + b"\x00" + instance_id.encode("ascii")
                ).digest(),
                instance_id,
            )
        )
    quotas = {
        repository: min(base_quota, len(values))
        for repository, values in eligible.items()
    }
    remaining = target_count - sum(quotas.values())
    require(remaining >= 0, "base quotas exceed the target count")
    allocation_order = sorted(eligible, key=lambda repository: (-len(eligible[repository]), repository))
    while remaining:
        progressed = False
        for repository in allocation_order:
            if remaining == 0:
                break
            if quotas[repository] >= len(eligible[repository]):
                continue
            quotas[repository] += 1
            remaining -= 1
            progressed = True
        require(progressed, "eligible capacity cannot satisfy the target count")

    campaign_a: list[str] = []
    campaign_b: list[str] = []
    allocation: list[Mapping[str, Any]] = []
    for repository in sorted(eligible):
        selected = eligible[repository][: quotas[repository]]
        destination = 0 if len(campaign_a) <= len(campaign_b) else 1
        counts = [0, 0]
        for index, instance_id in enumerate(selected):
            current = (destination + index) % 2
            (campaign_a if current == 0 else campaign_b).append(instance_id)
            counts[current] += 1
        allocation.append(
            {
                "repository": repository,
                "unused_count": len(eligible[repository]),
                "selected_quota": quotas[repository],
                "campaign_a_count": counts[0],
                "campaign_b_count": counts[1],
            }
        )
    require(
        len(campaign_a) == len(campaign_b) == target_count // 2,
        "deterministic partition did not produce balanced campaigns",
    )
    return SelectionResult(
        official_ids=official,
        repository_counts=dict(sorted(repository_counts.items())),
        unused_ids=unused,
        eligible_ids=eligible_ids,
        allocation=tuple(allocation),
        campaign_a=tuple(campaign_a),
        campaign_b=tuple(campaign_b),
    )


def load_identity_rows(arrow_path: Path) -> list[tuple[str, str]]:
    """Lazily load only identity columns from the pinned cached Arrow file."""

    require(arrow_path.is_file() and not arrow_path.is_symlink(), "cached Arrow source is not a regular file")
    require(sha256_file(arrow_path) == EXPECTED_DATASET["cached_arrow_sha256"], "cached Arrow bytes changed")
    try:
        from datasets import Dataset
    except ImportError as error:
        raise CohortValidationError(
            "full reproduction requires the datasets package; declaration-only validation does not"
        ) from error
    try:
        dataset = Dataset.from_file(str(arrow_path))
        require(
            "instance_id" in dataset.column_names and "repo" in dataset.column_names,
            "cached Arrow source lacks identity columns",
        )
        identity = dataset.select_columns(["instance_id", "repo"])
        rows = [(row["instance_id"], row["repo"]) for row in identity]
    except CohortValidationError:
        raise
    except Exception as error:
        raise CohortValidationError(f"cannot read cached Arrow identity columns: {error}") from error
    return rows


def validate_full_reproduction(
    declaration: DeclarationBundle,
    *,
    arrow_path: Path,
    git_bin: str = "git",
) -> SelectionResult:
    """Recompute the official, prior, unused, eligible, and selected identities."""

    rows = load_identity_rows(arrow_path.expanduser().resolve(strict=True))
    official_ids = frozenset(instance_id for instance_id, _repository in rows)
    historical = load_frozen_historical(official_ids=official_ids, git_bin=git_bin)
    reproduced = reproduce_selection(rows, prior_ids=historical.prior_ids)
    proof_dataset = mapping(declaration.proof.get("dataset"), "selection proof dataset")
    require(
        len(reproduced.official_ids) == proof_dataset.get("official_instance_count") == 500
        and sha256_set(reproduced.official_ids) == proof_dataset.get("official_instance_ids_set_sha256") == OFFICIAL_SET_SHA256
        and reproduced.repository_counts == proof_dataset.get("official_repository_counts") == EXPECTED_REPOSITORY_COUNTS,
        "official dataset identity or repository counts changed",
    )
    historical_proof = mapping(declaration.proof.get("historical_exclusion"), "historical exclusion")
    require(
        len(historical.tracked_paths) == historical_proof.get("tracked_config_count") == 26
        and sha256_ordered(historical.tracked_paths) == historical_proof.get("tracked_config_paths_sha256") == TRACKED_PATHS_SHA256
        and sha256_json(list(historical.tracked_tree)) == historical_proof.get("tracked_config_tree_sha256") == TRACKED_TREE_SHA256,
        "frozen tracked config tree changed",
    )
    require(
        list(historical.matched_sources) == historical_proof.get("matched_source_files"),
        "frozen historical match sources changed",
    )
    require(
        len(historical.prior_ids) == historical_proof.get("prior_reserved_instance_count") == 154
        and sorted(historical.prior_ids) == historical_proof.get("prior_reserved_instance_ids")
        and sha256_set(historical.prior_ids) == historical_proof.get("prior_reserved_instance_ids_set_sha256") == PRIOR_SET_SHA256,
        "frozen prior/reserved identity changed",
    )
    proof_selection = mapping(declaration.proof.get("selection"), "selection proof rule")
    require(
        len(reproduced.unused_ids) == proof_selection.get("unused_instance_count") == 346
        and sha256_set(reproduced.unused_ids) == proof_selection.get("unused_instance_ids_set_sha256") == UNUSED_SET_SHA256,
        "unused official identity changed",
    )
    require(
        len(reproduced.eligible_ids) == proof_selection.get("eligible_common_campaign_instance_count") == 345
        and sha256_set(reproduced.eligible_ids) == proof_selection.get("eligible_common_campaign_instance_ids_set_sha256") == ELIGIBLE_SET_SHA256
        and list(reproduced.allocation) == proof_selection.get("repository_allocation") == EXPECTED_ALLOCATION,
        "eligible repository allocation changed",
    )
    require(
        reproduced.campaign_a == declaration.campaign_ids[0]
        and reproduced.campaign_b == declaration.campaign_ids[1],
        "campaign membership/order differs from deterministic reproduction",
    )
    require(
        not (set(reproduced.campaign_a) & set(reproduced.campaign_b))
        and not (set((*reproduced.campaign_a, *reproduced.campaign_b)) & historical.prior_ids)
        and set((*reproduced.campaign_a, *reproduced.campaign_b)) <= reproduced.official_ids,
        "reproduced cohort disjointness failed",
    )
    return reproduced


def _regular_descendant(path: Path, *, root: Path, label: str) -> Path:
    require(root.is_dir() and not root.is_symlink(), f"{label} root is not a regular directory")
    resolved_root = root.resolve(strict=True)
    require(resolved_root == root.absolute(), f"{label} root traverses a symlink")
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    resolved = path.resolve(strict=True)
    require(
        resolved.is_relative_to(resolved_root) and resolved == path.absolute(),
        f"{label} traverses a symlink or escapes its run root",
    )
    return resolved


def _bundle_output_file(path: Path, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = Path(os.path.abspath(candidate))
    require(
        V3_OUTPUT_ROOT.is_dir() and not V3_OUTPUT_ROOT.is_symlink(),
        "dedicated V3 output namespace is not a regular directory",
    )
    output_root = V3_OUTPUT_ROOT.absolute()
    require(
        V3_OUTPUT_ROOT.resolve(strict=True) == output_root,
        "dedicated V3 output namespace traverses a symlink",
    )
    require(
        candidate.parent == output_root and candidate.suffix == ".json",
        f"{label} is not a direct JSON child of the dedicated V3 output namespace",
    )
    require(
        candidate.is_file() and not candidate.is_symlink(),
        f"{label} is not a regular non-symlink file",
    )
    require(
        candidate.resolve(strict=True) == candidate,
        f"{label} traverses a symlink",
    )
    return candidate


def validate_run_image_provenance(
    declaration: DeclarationBundle,
) -> RunImageProvenance:
    """Bind exact run image manifests and runner metadata to finalized registries."""

    require(
        V3_RUNS_ROOT.is_dir() and not V3_RUNS_ROOT.is_symlink(),
        "dedicated V3 run namespace is not a regular directory",
    )
    resolved_runs_root = V3_RUNS_ROOT.resolve(strict=True)
    require(
        resolved_runs_root == V3_RUNS_ROOT.absolute(),
        "dedicated V3 run namespace traverses a symlink",
    )
    cohort_rows = [
        mapping(row, f"cohort row {index}")
        for index, row in enumerate(
            sequence(declaration.cohort.get("cohorts"), "cohort rows")
        )
    ]
    manifest_hashes: list[str] = []
    runner_bindings: dict[str, Mapping[str, str]] = {}
    for cohort_index, row in enumerate(cohort_rows):
        run_label = nonempty_string(row.get("run_label"), f"cohort {cohort_index} run label")
        run_root = V3_RUNS_ROOT / run_label
        require(
            run_root.is_dir()
            and not run_root.is_symlink()
            and run_root.parent.resolve(strict=True) == resolved_runs_root
            and run_root.resolve(strict=True) == run_root.absolute(),
            f"cohort {cohort_index} run root is not the exact dedicated direct child",
        )
        image_binding = mapping(row.get("image_registry"), f"cohort {cohort_index} image registry")
        registry_path = repository_path(
            image_binding.get("path"), f"cohort {cohort_index} image registry path"
        )
        registry = mapping(
            strict_json_file(registry_path, f"cohort {cohort_index} image registry"),
            f"cohort {cohort_index} image registry",
        )
        registry_images = mapping(
            registry.get("images"), f"cohort {cohort_index} registry images"
        )
        instance_ids = declaration.campaign_ids[cohort_index]
        manifest_path = _regular_descendant(
            run_root / "image_manifest.json",
            root=run_root,
            label=f"cohort {cohort_index} run image manifest",
        )
        manifest_sha256 = sha256_file(manifest_path)
        manifest = mapping(
            strict_json_file(manifest_path, f"cohort {cohort_index} run image manifest"),
            f"cohort {cohort_index} run image manifest",
        )
        exact_keys(
            manifest,
            {
                "schema_version",
                "kind",
                "campaign_config_path",
                "campaign_config_sha256",
                "image_config_path",
                "image_config_sha256",
                "selection_proof_path",
                "selection_proof_sha256",
                "dataset",
                "images",
            },
            f"cohort {cohort_index} run image manifest",
        )
        require(
            manifest.get("schema_version") == 1
            and manifest.get("kind") == "swe_verified_behavioral_campaign_image_manifest"
            and manifest.get("campaign_config_path")
            == str(declaration.campaign_paths[cohort_index])
            and manifest.get("campaign_config_sha256")
            == row.get("campaign_sha256")
            and manifest.get("image_config_path") == str(registry_path)
            and manifest.get("image_config_sha256")
            == image_binding.get("sha256")
            and manifest.get("selection_proof_path") == str(declaration.proof_path)
            and manifest.get("selection_proof_sha256")
            == sha256_file(declaration.proof_path)
            and manifest.get("dataset") == EXPECTED_CAMPAIGN_DATASET,
            f"cohort {cohort_index} run image manifest upstream binding changed",
        )
        image_rows = [
            mapping(value, f"cohort {cohort_index} image row {image_index}")
            for image_index, value in enumerate(
                sequence(manifest.get("images"), f"cohort {cohort_index} image rows")
            )
        ]
        require(
            [image.get("instance_id") for image in image_rows]
            == list(instance_ids),
            f"cohort {cohort_index} run image order/coverage changed",
        )
        for instance_id, image in zip(instance_ids, image_rows, strict=True):
            exact_keys(
                image,
                {"instance_id", "tag", "image_id", "repo_digests"},
                f"run image {instance_id}",
            )
            pin = mapping(
                mapping(registry_images.get(instance_id), f"registry {instance_id}").get(
                    "x86_64"
                ),
                f"registry {instance_id} x86_64",
            )
            reference = nonempty_string(pin.get("reference"), f"registry {instance_id} reference")
            expected_tag = (
                "swebench/sweb.eval.x86_64."
                + instance_id.replace("__", "_1776_")
                + ":latest"
            )
            repo_digests = sequence(image.get("repo_digests"), f"run image {instance_id} RepoDigests")
            require(
                image.get("tag") == expected_tag
                and image.get("image_id") == pin.get("image_id")
                and all(isinstance(value, str) for value in repo_digests)
                and reference in repo_digests,
                f"run image {instance_id} differs from finalized registry",
            )
            metadata_relative = PurePosixPath(
                "generation/verified/per_task"
            ) / instance_id / "runner_metadata.json"
            metadata_path = _regular_descendant(
                run_root.joinpath(*metadata_relative.parts),
                root=run_root,
                label=f"runner metadata {instance_id}",
            )
            metadata = mapping(
                strict_json_file(metadata_path, f"runner metadata {instance_id}"),
                f"runner metadata {instance_id}",
            )
            require(
                metadata.get("instance_id") == instance_id
                and metadata.get("image") == reference
                and metadata.get("dataset_name")
                == str((run_root / "dataset.json").resolve(strict=True)),
                f"runner metadata image/dataset binding changed: {instance_id}",
            )
            require(instance_id not in runner_bindings, "runner metadata task repeats")
            runner_bindings[instance_id] = {
                "path": metadata_relative.as_posix(),
                "sha256": sha256_file(metadata_path),
                "image_reference": reference,
            }
        manifest_hashes.append(manifest_sha256)
    require(len(manifest_hashes) == 2 and len(runner_bindings) == 60, "run image provenance coverage changed")
    return RunImageProvenance(
        image_manifest_sha256s=(manifest_hashes[0], manifest_hashes[1]),
        runner_metadata=runner_bindings,
    )


def _prompt_payload_sha256(prompt: Mapping[str, Any]) -> str:
    metadata = mapping(prompt.get("metadata"), "prompt metadata")
    provenance = mapping(metadata.get("provenance"), "prompt provenance")
    require(isinstance(provenance, dict), "prompt provenance is not mutable")
    sentinel = object()
    stored = provenance.pop("prompt_record_payload_sha256", sentinel)
    try:
        return sha256_json(prompt)
    finally:
        if stored is not sentinel:
            provenance["prompt_record_payload_sha256"] = stored


def validate_materialized_bundle(
    declaration: DeclarationBundle,
    *,
    prompts_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Authenticate exact N=60 all-probeable prompt and summary provenance."""

    prompts_path = _bundle_output_file(prompts_path, "prompt bundle")
    summary_path = _bundle_output_file(summary_path, "prompt summary")
    require(prompts_path != summary_path, "prompt bundle and summary paths are identical")
    summary = mapping(strict_json_file(summary_path, "prompt summary"), "prompt summary")

    cohort = declaration.cohort
    cohort_rows = [
        mapping(value, f"cohort row {index}")
        for index, value in enumerate(sequence(cohort.get("cohorts"), "cohort rows"))
    ]
    require(len(cohort_rows) == len(declaration.campaigns) == 2, "bundle cohort count changed")
    run_images = validate_run_image_provenance(declaration)
    cohort_sha256 = sha256_file(declaration.cohort_path)
    expected_instance_ids = list((*declaration.campaign_ids[0], *declaration.campaign_ids[1]))
    require(len(expected_instance_ids) == 60, "bundle declaration is not N=60")
    expected_campaign_hashes = [
        validate_sha256(row.get("campaign_sha256"), f"campaign {index} hash")
        for index, row in enumerate(cohort_rows)
    ]
    expected_cohort_ids = [
        nonempty_string(row.get("id"), f"cohort {index} ID")
        for index, row in enumerate(cohort_rows)
    ]
    require(len(expected_cohort_ids) == len(set(expected_cohort_ids)), "cohort IDs repeat")
    task_binding: dict[str, tuple[int, str, str, list[str], int]] = {}
    for index, row in enumerate(cohort_rows):
        instance_ids = list(declaration.campaign_ids[index])
        require(row.get("instance_ids") == instance_ids, "campaign task order changed")
        for instance_id in instance_ids:
            require(instance_id not in task_binding, "bundle cohort repeats a task")
            task_binding[instance_id] = (
                index,
                expected_cohort_ids[index],
                expected_campaign_hashes[index],
                instance_ids,
                len(task_binding),
            )
    require(list(task_binding) == expected_instance_ids, "bundle task order differs from frozen N=60 cohort")

    pins = mapping(cohort.get("pins"), "cohort pins")
    prompt_ids: set[str] = set()
    global_indices: list[int] = []
    first_seen_tasks: list[str] = []
    observed_by_task: dict[str, list[int]] = defaultdict(list)
    probeable_by_task: dict[str, list[int]] = {}
    prompt_facts_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected_summary_prompts: list[dict[str, Any]] = []
    prompt_stream_hash: dict[str, str] = {}
    prompt_count = 0
    for index, raw_prompt in enumerate(
        iter_strict_json_array(
            prompts_path,
            "prompt bundle",
            sha256_sink=prompt_stream_hash,
        )
    ):
        prompt = mapping(raw_prompt, f"prompt row {index}")
        prompt_count += 1
        prompt_id = nonempty_string(prompt.get("id"), f"prompt {index} ID")
        require(prompt_id not in prompt_ids, "materialized prompt IDs repeat")
        prompt_ids.add(prompt_id)
        metadata = mapping(prompt.get("metadata"), f"prompt {prompt_id} metadata")
        task = mapping(metadata.get("task"), f"prompt {prompt_id} task")
        selection = mapping(metadata.get("selection"), f"prompt {prompt_id} selection")
        provenance = mapping(metadata.get("provenance"), f"prompt {prompt_id} provenance")
        cohort_metadata = mapping(metadata.get("cohort"), f"prompt {prompt_id} cohort")
        instance_id = nonempty_string(task.get("instance_id"), "prompt instance ID")
        require(instance_id in task_binding, f"prompt contains an unfrozen task: {instance_id}")
        cohort_index, cohort_id, campaign_hash, source_ids, task_selection_index = task_binding[
            instance_id
        ]
        require(
            metadata.get("campaign_sha256") == campaign_hash
            and metadata.get("action_protocol_sha256")
            == pins.get("action_protocol_sha256")
            and metadata.get("chat_template_sha256")
            == pins.get("chat_template_sha256"),
            f"prompt top-level provenance binding differs: {prompt_id}",
        )
        require(
            cohort_metadata.get("id") == cohort_id
            and cohort_metadata.get("index") == cohort_index
            and cohort_metadata.get("campaign_sha256") == campaign_hash
            and cohort_metadata.get("cohort_manifest_sha256") == cohort_sha256
            and cohort_metadata.get("source_task_instance_ids") == source_ids,
            f"prompt cohort binding differs: {prompt_id}",
        )
        require(
            cohort_metadata.get("source_image_manifest_sha256")
            == run_images.image_manifest_sha256s[cohort_index],
            f"prompt cohort image-manifest binding differs: {prompt_id}",
        )
        require(
            task.get("selection_index") == task_selection_index
            and task.get("source_selection_index")
            == task_selection_index - (30 * cohort_index),
            f"prompt task selection index differs: {prompt_id}",
        )
        task_request_index = positive_integer(selection.get("task_request_index"), "task request index")
        global_request_index = positive_integer(selection.get("global_request_index"), "global request index")
        source_global_request_index = positive_integer(
            selection.get("source_global_request_index"),
            "source global request index",
        )
        probeable = [
            positive_integer(value, "probeable request index")
            for value in sequence(task.get("probeable_request_indices"), "probeable request indices")
        ]
        require(bool(probeable) and probeable == sorted(set(probeable)), "probeable request indices are invalid")
        require(
            selection.get("max_checkpoints") is None
            and selection.get("probeable_request_indices") == probeable
            and selection.get("candidate_count") == len(probeable)
            and selection.get("checkpoint_count") == len(probeable)
            and task.get("probeable_request_count") == len(probeable),
            f"prompt is not bound to the all-probeable contract: {prompt_id}",
        )
        if instance_id not in observed_by_task:
            first_seen_tasks.append(instance_id)
            probeable_by_task[instance_id] = probeable
        else:
            require(
                probeable_by_task[instance_id] == probeable,
                f"probeable request declaration changed within {instance_id}",
            )
        observed_by_task[instance_id].append(task_request_index)
        global_indices.append(global_request_index)
        combination = mapping(
            provenance.get("combination"),
            f"prompt {prompt_id} combination provenance",
        )
        require(
            combination.get("cohort_manifest_sha256") == cohort_sha256
            and combination.get("combined_global_request_index")
            == global_request_index
            and combination.get("source_campaign_global_request_index")
            == source_global_request_index
            and combination.get("source_image_manifest_sha256")
            == run_images.image_manifest_sha256s[cohort_index],
            f"prompt combination provenance differs: {prompt_id}",
        )
        runner_binding = mapping(
            run_images.runner_metadata.get(instance_id),
            f"runner binding {instance_id}",
        )
        require(
            provenance.get("runner_metadata_path") == runner_binding.get("path")
            and provenance.get("runner_metadata_sha256")
            == runner_binding.get("sha256"),
            f"prompt runner-metadata provenance differs: {prompt_id}",
        )
        prompt_facts_by_task[instance_id].append(
            {
                "id": prompt_id,
                "task_request_index": task_request_index,
                "global_request_index": global_request_index,
                "source_global_request_index": source_global_request_index,
                "task_request_count": task.get("request_count"),
                "cohort_metadata": dict(cohort_metadata),
            }
        )
        payload_hash = validate_sha256(
            provenance.get("prompt_record_payload_sha256"),
            f"prompt {prompt_id} payload hash",
        )
        require(_prompt_payload_sha256(prompt) == payload_hash, f"prompt payload hash differs: {prompt_id}")
        expected_summary_prompts.append(
            {
                "id": prompt_id,
                "cohort_id": cohort_id,
                "instance_id": instance_id,
                "global_request_index": global_request_index,
                "prompt_record_payload_sha256": payload_hash,
            }
        )

    require(prompt_count >= 1, "materialized prompt bundle is empty")
    require(first_seen_tasks == expected_instance_ids, "materialized task order/coverage changed")
    require(
        len(global_indices) == len(set(global_indices))
        and global_indices == sorted(global_indices),
        "materialized global request indices repeat or are out of order",
    )
    for instance_id in expected_instance_ids:
        require(
            observed_by_task[instance_id] == probeable_by_task[instance_id],
            f"materialized bundle does not cover every probeable request: {instance_id}",
        )

    prompt_bundle_sha256 = validate_sha256(
        prompt_stream_hash.get("sha256"), "streamed prompt bundle SHA-256"
    )
    require(
        summary.get("schema_version") == 1
        and summary.get("kind") == "swe_verified_behavioral_probe_combination"
        and summary.get("cohort_manifest_sha256") == cohort_sha256
        and summary.get("source_campaign_sha256s") == expected_campaign_hashes
        and summary.get("campaign_sha256s") == expected_campaign_hashes
        and summary.get("action_protocol_sha256") == pins.get("action_protocol_sha256")
        and summary.get("chat_template_sha256") == pins.get("chat_template_sha256")
        and summary.get("cohort_count") == 2
        and summary.get("task_count") == 60
        and summary.get("prompt_count") == prompt_count
        and summary.get("prompt_bundle_sha256") == prompt_bundle_sha256,
        "materialized summary identity, provenance, or counts changed",
    )
    require(summary.get("prompts") == expected_summary_prompts, "materialized summary prompt binding changed")
    summary_cohorts = [
        mapping(value, f"summary cohort {index}")
        for index, value in enumerate(sequence(summary.get("cohorts"), "summary cohorts"))
    ]
    require(len(summary_cohorts) == 2, "summary cohort count changed")
    for index, (row, summary_cohort) in enumerate(zip(cohort_rows, summary_cohorts, strict=True)):
        require(
            summary_cohort.get("id") == expected_cohort_ids[index]
            and summary_cohort.get("index") == index
            and summary_cohort.get("campaign_sha256") == expected_campaign_hashes[index]
            and summary_cohort.get("cohort_manifest_sha256") == cohort_sha256
            and summary_cohort.get("source_task_instance_ids") == row.get("instance_ids")
            and summary_cohort.get("source_task_count") == 30
            and summary_cohort.get("source_image_manifest_sha256")
            == run_images.image_manifest_sha256s[index],
            f"summary cohort binding changed: {index}",
        )
    audits = [
        mapping(value, f"task audit {index}")
        for index, value in enumerate(sequence(summary.get("task_audits"), "task audits"))
    ]
    require(len(audits) == 60, "task audit coverage changed")
    global_cursor = 1
    source_cursors = [1, 1]
    source_task_indices = [0, 0]
    cohort_prompt_counts = [0, 0]
    for index, (instance_id, audit) in enumerate(zip(expected_instance_ids, audits, strict=True)):
        cohort_index, cohort_id, campaign_hash, _source_ids, task_selection_index = task_binding[
            instance_id
        ]
        request_count = positive_integer(audit.get("request_count"), "task request count")
        global_start = positive_integer(
            audit.get("global_request_start"), "task global request start"
        )
        global_end = positive_integer(
            audit.get("global_request_end"), "task global request end"
        )
        source_start = positive_integer(
            audit.get("source_global_request_start"),
            "task source global request start",
        )
        source_end = positive_integer(
            audit.get("source_global_request_end"),
            "task source global request end",
        )
        require(
            audit.get("instance_id") == instance_id
            and audit.get("selection_index") == index == task_selection_index
            and audit.get("source_selection_index")
            == source_task_indices[cohort_index]
            and audit.get("cohort_id") == cohort_id
            and audit.get("campaign_sha256") == campaign_hash
            and audit.get("probeable_request_indices") == probeable_by_task[instance_id]
            and audit.get("selected_request_indices") == observed_by_task[instance_id]
            and audit.get("selected_checkpoint_count") == len(observed_by_task[instance_id])
            and cohort_index == expected_cohort_ids.index(cohort_id)
            and global_start == global_cursor
            and global_end == global_start + request_count - 1
            and source_start == source_cursors[cohort_index]
            and source_end == source_start + request_count - 1,
            f"task audit binding changed: {instance_id}",
        )
        for fact in prompt_facts_by_task[instance_id]:
            task_request_index = fact["task_request_index"]
            require(
                fact["task_request_count"] == request_count
                and 1 <= task_request_index <= request_count
                and fact["global_request_index"]
                == global_start + task_request_index - 1
                and fact["source_global_request_index"]
                == source_start + task_request_index - 1
                and fact["cohort_metadata"] == dict(summary_cohorts[cohort_index]),
                f"prompt request range or cohort offset differs: {fact['id']}",
            )
        cohort_prompt_counts[cohort_index] += len(prompt_facts_by_task[instance_id])
        global_cursor = global_end + 1
        source_cursors[cohort_index] = source_end + 1
        source_task_indices[cohort_index] += 1
    require(
        summary.get("global_request_count") == global_cursor - 1,
        "summary global request count changed",
    )
    expected_global_offset = 0
    for index, summary_cohort in enumerate(summary_cohorts):
        require(
            summary_cohort.get("task_offset") == 30 * index
            and summary_cohort.get("global_request_offset") == expected_global_offset
            and summary_cohort.get("source_global_request_count")
            == source_cursors[index] - 1
            and summary_cohort.get("source_prompt_count")
            == cohort_prompt_counts[index]
            and summary_cohort.get("source_run_label")
            == cohort_rows[index].get("run_label"),
            f"summary cohort request offsets/counts changed: {index}",
        )
        expected_global_offset += source_cursors[index] - 1
    require(
        expected_global_offset == global_cursor - 1,
        "summary cohort request ranges do not cover the combined request space",
    )
    return {
        "cohort_manifest_sha256": cohort_sha256,
        "cohort_count": 2,
        "task_count": 60,
        "prompt_count": prompt_count,
        "prompt_bundle_sha256": prompt_bundle_sha256,
        "summary_sha256": sha256_file(summary_path),
    }


def _materialization_protocol_pins(declaration: DeclarationBundle) -> Mapping[str, Any]:
    scientific = mapping(
        declaration.cohort.get("scientific_artifacts"),
        "cohort scientific artifacts",
    )
    interpreter_binding = mapping(
        scientific.get("interpreter_protocol"),
        "interpreter protocol binding",
    )
    interpreter_path = repository_path(
        interpreter_binding.get("path"), "interpreter protocol path"
    )
    require(
        interpreter_binding.get("path") == EXPECTED_INTERPRETER_PATH
        and interpreter_binding.get("sha256") == sha256_file(interpreter_path),
        "interpreter protocol materialization binding changed",
    )
    interpreter = mapping(
        strict_json_file(interpreter_path, "V3 interpreter protocol"),
        "V3 interpreter protocol",
    )
    return mapping(interpreter.get("pins"), "V3 interpreter pins")


def _validate_materialization_invocation(value: Any) -> dict[str, bool]:
    invocation = mapping(value, "materialization invocation")
    exact_keys(
        invocation,
        {"all_probeable", "require_official_outcomes"},
        "materialization invocation",
    )
    require(invocation.get("all_probeable") is True, "materialization was not all-probeable")
    require(
        isinstance(invocation.get("require_official_outcomes"), bool),
        "official-outcome materialization flag is not Boolean",
    )
    return {
        "all_probeable": True,
        "require_official_outcomes": bool(invocation["require_official_outcomes"]),
    }


def _run_source_inventory(
    declaration: DeclarationBundle,
) -> tuple[list[dict[str, Any]], int]:
    cohort_rows = [
        mapping(value, f"cohort row {index}")
        for index, value in enumerate(sequence(declaration.cohort.get("cohorts"), "cohort rows"))
    ]
    require(len(cohort_rows) == 2, "materialization source inventory is not A/B")
    run_records: list[dict[str, Any]] = []
    total_files = 0
    for cohort_index, row in enumerate(cohort_rows):
        run_label = nonempty_string(row.get("run_label"), f"cohort {cohort_index} run label")
        run_root = V3_RUNS_ROOT / run_label
        files = _inventory_regular_tree(run_root, f"cohort {cohort_index} run root")
        logical_paths = [str(record["path"]) for record in files]
        path_set = set(logical_paths)
        run_prefix = _root_relative_file(
            run_root / "dataset.json", f"cohort {cohort_index} dataset"
        ).rsplit("/dataset.json", 1)[0]
        required = {
            f"{run_prefix}/dataset.json",
            f"{run_prefix}/image_manifest.json",
            f"{run_prefix}/proxy_dumps/usage.jsonl",
        }
        for instance_id in declaration.campaign_ids[cohort_index]:
            task_prefix = f"{run_prefix}/generation/verified/per_task/{instance_id}"
            required.update(
                {
                    f"{task_prefix}/runner_metadata.json",
                    f"{task_prefix}/qwen_trace.json",
                    f"{task_prefix}/patch.diff",
                }
            )
        missing = sorted(required - path_set)
        require(
            not missing,
            f"cohort {cohort_index} source inventory lacks required artifacts: {missing[:3]}",
        )
        chat_prefix = f"{run_prefix}/proxy_dumps/chat_"
        chat_paths = [
            logical for logical in logical_paths if logical.startswith(chat_prefix) and logical.endswith(".json")
        ]
        require(bool(chat_paths), f"cohort {cohort_index} has no raw proxy chat captures")
        require(
            chat_paths
            == [f"{run_prefix}/proxy_dumps/chat_{index:04d}.json" for index in range(1, len(chat_paths) + 1)],
            f"cohort {cohort_index} raw proxy chat capture sequence changed",
        )
        run_records.append(
            {
                "cohort_index": cohort_index,
                "cohort_id": nonempty_string(row.get("id"), f"cohort {cohort_index} ID"),
                "run_label": run_label,
                "root": run_prefix,
                "file_count": len(files),
                "files_sha256": sha256_json(files),
                "raw_proxy_chat_count": len(chat_paths),
                "files": files,
            }
        )
        total_files += len(files)
    return run_records, total_files


def build_materialization_receipt(
    declaration: DeclarationBundle,
    *,
    prompts_path: Path,
    summary_path: Path,
    invocation: Mapping[str, Any],
    source_freeze_git_commit: str | None = None,
) -> dict[str, Any]:
    """Recompute the exact source/output receipt without trusting prompt declarations."""

    checked = validate_materialized_bundle(
        declaration,
        prompts_path=prompts_path,
        summary_path=summary_path,
    )
    require(
        prompts_path.resolve(strict=True) == DEFAULT_PROMPTS.absolute()
        and summary_path.resolve(strict=True) == DEFAULT_PROMPT_SUMMARY.absolute(),
        "materialization receipt outputs are not the exact canonical prompt paths",
    )
    frozen_invocation = _validate_materialization_invocation(invocation)
    if source_freeze_git_commit is None:
        source_freeze_git_commit = capture_clean_source_freeze()
    else:
        require(
            re.fullmatch(r"[0-9a-f]{40}", source_freeze_git_commit) is not None,
            "materialization source-freeze commit is invalid",
        )
        _receipt_git_output(
            ["cat-file", "-e", f"{source_freeze_git_commit}^{{commit}}"],
            "source-freeze commit validation",
        )
    repository_sources = _repository_source_records(source_freeze_git_commit)
    repository_sources_by_path = {
        str(record["path"]): record for record in repository_sources
    }
    pins = _materialization_protocol_pins(declaration)
    implementation_paths = {
        "checker": "scripts/check_swe_task_state_v3_development_cohort.py",
        "v3_materializer": EXPECTED_V3_MATERIALIZER_PATH,
        "historical_materializer": EXPECTED_HISTORICAL_MATERIALIZER_PATH,
        "replay_pipeline": EXPECTED_REPLAY_PIPELINE_PATH,
        "replay_shell_wrapper": EXPECTED_REPLAY_WRAPPER_PATH,
    }
    implementations = {
        name: dict(repository_sources_by_path[path])
        for name, path in implementation_paths.items()
    }
    require(
        implementations["checker"]["sha256"]
        == pins.get("materialized_bundle_checker_sha256")
        and implementations["v3_materializer"]["sha256"]
        == pins.get("v3_materializer_sha256")
        and implementations["historical_materializer"]["sha256"]
        == pins.get("historical_materializer_sha256")
        and implementations["replay_pipeline"]["sha256"]
        == pins.get("replay_pipeline_sha256")
        and implementations["replay_shell_wrapper"]["sha256"]
        == pins.get("replay_shell_wrapper_sha256"),
        "materialization/replay implementation bytes differ from protocol pins",
    )
    run_sources, run_source_file_count = _run_source_inventory(declaration)
    generation = mapping(
        declaration.campaigns[0].get("generation"), "campaign generation"
    )
    tokenizer = mapping(pins.get("tokenizer"), "V3 tokenizer pin")
    prompt_record = {
        "path": _root_relative_file(prompts_path, "canonical prompt bundle"),
        "sha256": checked["prompt_bundle_sha256"],
        "bytes": prompts_path.stat(follow_symlinks=False).st_size,
    }
    summary_record = {
        "path": _root_relative_file(summary_path, "canonical prompt summary"),
        "sha256": checked["summary_sha256"],
        "bytes": summary_path.stat(follow_symlinks=False).st_size,
    }
    return {
        "schema_version": 1,
        "kind": MATERIALIZATION_RECEIPT_KIND,
        "status": "materialized_before_public_lens_replay_for_exact_child_git_freeze",
        "source_freeze_git_commit": source_freeze_git_commit,
        "freeze_contract": {
            "source_freeze_tracked_tree_clean_at_materialization": True,
            "receipt_first_add_commit_must_be_direct_child_of_source_freeze": True,
            "receipt_add_commit_may_change_only_this_receipt": True,
            "receipt_and_source_bytes_must_remain_tracked_and_clean": True,
            "public_lens_replay_before_receipt_git_freeze": False,
        },
        "invocation": frozen_invocation,
        "tokenizer": {
            "model_repo_id": generation.get("model_repo_id"),
            "model_revision": generation.get("model_revision"),
            "tokenizer_json_sha256": tokenizer.get("json_sha256"),
            "vocabulary_size": tokenizer.get("vocabulary_size"),
        },
        "declaration": {
            "cohort_manifest": {
                "path": EXPECTED_COHORT_PATH,
                "sha256": sha256_file(declaration.cohort_path),
            },
            "selection_proof": {
                "path": EXPECTED_PROOF_PATH,
                "sha256": sha256_file(declaration.proof_path),
            },
            "ordered_instance_ids_sha256": COMBINED_ORDERED_SHA256,
            "instance_ids_set_sha256": COMBINED_SET_SHA256,
            "task_count": 60,
        },
        "implementations": implementations,
        "repository_source_file_count": len(repository_sources),
        "repository_source_files_sha256": sha256_json(repository_sources),
        "repository_source_files": repository_sources,
        "run_source_file_count": run_source_file_count,
        "run_source_files_sha256": sha256_json(run_sources),
        "runs": run_sources,
        "outputs": {
            "prompt_bundle": prompt_record,
            "prompt_summary": summary_record,
            "task_count": checked["task_count"],
            "cohort_count": checked["cohort_count"],
            "prompt_count": checked["prompt_count"],
        },
    }


def _validate_receipt_git_freeze(receipt_path: Path, receipt: Mapping[str, Any]) -> str:
    _require_clean_tracked_worktree()
    source_commit = nonempty_string(
        receipt.get("source_freeze_git_commit"), "source-freeze Git commit"
    )
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None, "invalid source-freeze commit")
    logical_path = _root_relative_file(receipt_path, "materialization receipt")
    require(
        logical_path == EXPECTED_MATERIALIZATION_RECEIPT_PATH,
        "materialization receipt path changed",
    )
    current_bytes = receipt_path.read_bytes()
    head = _git_head()
    require(
        _git_file_at_commit(head, logical_path) == current_bytes,
        "materialization receipt differs from current HEAD",
    )
    additions = [
        value
        for value in _receipt_git_output(
            ["log", "--format=%H", "--diff-filter=A", "--", logical_path],
            "receipt introduction lookup",
        )
        .decode("ascii")
        .splitlines()
        if value
    ]
    require(len(additions) == 1, "receipt does not have one unique first-add commit")
    data_commit = additions[0]
    parents = _receipt_git_output(
        ["rev-list", "--parents", "-n", "1", data_commit],
        "receipt commit parents",
    ).decode("ascii").split()
    require(
        parents == [data_commit, source_commit],
        "receipt data-freeze commit is not the direct child of source-freeze HEAD",
    )
    changed = _receipt_git_output(
        ["diff-tree", "--no-commit-id", "--name-status", "-r", data_commit],
        "receipt commit diff",
    ).decode("utf-8").splitlines()
    require(
        changed == [f"A\t{logical_path}"],
        "receipt data-freeze commit changed files other than the new receipt",
    )
    require(
        _git_file_at_commit(data_commit, logical_path) == current_bytes,
        "current receipt bytes differ from the exact data-freeze commit",
    )
    _receipt_git_output(
        ["merge-base", "--is-ancestor", data_commit, head],
        "receipt commit ancestry",
    )
    return data_commit


def validate_materialization_receipt(
    declaration: DeclarationBundle,
    *,
    prompts_path: Path,
    summary_path: Path,
    receipt_path: Path = DEFAULT_MATERIALIZATION_RECEIPT,
    require_git_frozen: bool = True,
) -> MaterializationReceiptAudit:
    """Recompute every source/output hash and, for replay, prove exact Git chronology."""

    supplied = receipt_path.expanduser()
    require(not supplied.is_symlink(), "materialization receipt must not be a symlink")
    resolved = supplied.resolve(strict=True)
    require(
        resolved == (ROOT / EXPECTED_MATERIALIZATION_RECEIPT_PATH).resolve(strict=True),
        "materialization receipt is not the exact repository path",
    )
    receipt = mapping(
        strict_json_file(resolved, "materialization receipt"),
        "materialization receipt",
    )
    source_commit = nonempty_string(
        receipt.get("source_freeze_git_commit"), "source-freeze Git commit"
    )
    invocation = mapping(receipt.get("invocation"), "materialization invocation")
    expected = build_materialization_receipt(
        declaration,
        prompts_path=prompts_path,
        summary_path=summary_path,
        invocation=invocation,
        source_freeze_git_commit=source_commit,
    )
    require(receipt == expected, "materialization receipt differs from recomputed exact sources/outputs")
    data_commit = _validate_receipt_git_freeze(resolved, receipt) if require_git_frozen else None
    outputs = mapping(receipt.get("outputs"), "materialization receipt outputs")
    prompt_record = mapping(outputs.get("prompt_bundle"), "receipt prompt bundle")
    summary_record = mapping(outputs.get("prompt_summary"), "receipt prompt summary")
    return MaterializationReceiptAudit(
        receipt_path=resolved,
        receipt_sha256=sha256_file(resolved),
        source_freeze_git_commit=source_commit,
        data_freeze_git_commit=data_commit,
        prompt_bundle_sha256=validate_sha256(
            prompt_record.get("sha256"), "receipt prompt SHA-256"
        ),
        summary_sha256=validate_sha256(
            summary_record.get("sha256"), "receipt summary SHA-256"
        ),
        source_file_count=positive_integer(
            receipt.get("repository_source_file_count"),
            "repository source file count",
        )
        + positive_integer(receipt.get("run_source_file_count"), "run source file count"),
    )


def _default_arrow_path() -> Path:
    cache_root = Path(
        os.environ.get(
            "HF_DATASETS_CACHE",
            str(Path.home() / ".cache/huggingface/datasets"),
        )
    )
    return cache_root / str(EXPECTED_DATASET["cached_arrow_relative_path"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--dataset-arrow", type=Path, default=_default_arrow_path())
    parser.add_argument("--git-bin", default="git")
    parser.add_argument(
        "--declaration-only",
        action="store_true",
        help="validate checked-in bindings without importing datasets or reading Git blobs",
    )
    parser.add_argument(
        "--validate-bundle",
        nargs=2,
        type=Path,
        metavar=("PROMPTS", "SUMMARY"),
        help="also authenticate one materialized N=60 all-probeable bundle",
    )
    parser.add_argument(
        "--validate-materialization-receipt",
        action="store_true",
        help="require the canonical bundle's exact-child Git-frozen source receipt",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        declaration = validate_declaration(args.cohort)
        reproduced: SelectionResult | None = None
        if not args.declaration_only:
            reproduced = validate_full_reproduction(
                declaration,
                arrow_path=args.dataset_arrow,
                git_bin=args.git_bin,
            )
        bundle: dict[str, Any] | None = None
        if args.validate_bundle is not None:
            prompts_path, summary_path = args.validate_bundle
            bundle = validate_materialized_bundle(
                declaration,
                prompts_path=prompts_path,
                summary_path=summary_path,
            )
            if args.validate_materialization_receipt:
                validate_materialization_receipt(
                    declaration,
                    prompts_path=prompts_path,
                    summary_path=summary_path,
                    receipt_path=DEFAULT_MATERIALIZATION_RECEIPT,
                    require_git_frozen=True,
                )
        elif args.validate_materialization_receipt:
            raise CohortValidationError(
                "--validate-materialization-receipt requires --validate-bundle PROMPTS SUMMARY"
            )
    except (CohortValidationError, FileNotFoundError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if reproduced is None:
        print("validated frozen V3 N=60 declaration and two exact generation-authorized image registries")
    else:
        print(
            "reproduced frozen V3 N=60 identity-only cohort: "
            f"official={len(reproduced.official_ids)}, prior={len(reproduced.official_ids - reproduced.unused_ids)}, "
            f"unused={len(reproduced.unused_ids)}, eligible={len(reproduced.eligible_ids)}, campaigns=30+30"
        )
    if bundle is not None:
        print(
            "validated materialized N=60 all-probeable bundle: "
            f"tasks={bundle['task_count']}, prompts={bundle['prompt_count']}, "
            f"sha256={bundle['prompt_bundle_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
