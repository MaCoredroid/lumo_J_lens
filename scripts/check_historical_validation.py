#!/usr/bin/env python3
"""Verify legacy source ties without rebinding them to current code."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
PIN_FILE = ROOT / "validation" / "historical-validation-pins.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class HistoricalValidationError(ValueError):
    """Raised when historical provenance cannot be verified exactly."""


@dataclass(frozen=True)
class HybridManifest:
    path: str
    historical_prefixes: tuple[str, ...]
    current_prefixes: tuple[str, ...]


@dataclass(frozen=True)
class HistoricalPolicy:
    commit: str
    source_manifests: tuple[str, ...]
    hybrid_manifests: tuple[HybridManifest, ...]
    external_files: Mapping[str, str]


BlobReader = Callable[[Path, str, str], bytes]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise HistoricalValidationError(f"{label} must be a non-empty string")
    if "\\" in value or "\x00" in value:
        raise HistoricalValidationError(f"{label} is not a canonical POSIX path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise HistoricalValidationError(f"unsafe {label}: {value!r}")
    if path.as_posix() != value:
        raise HistoricalValidationError(f"non-canonical {label}: {value!r}")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise HistoricalValidationError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _string_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise HistoricalValidationError(f"{label} must be a non-empty list")
    result = tuple(
        validate_relative_path(item, label=f"{label} entry") for item in value
    )
    if len(set(result)) != len(result):
        raise HistoricalValidationError(f"{label} contains duplicates")
    return result


def _prefix_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise HistoricalValidationError(f"{label} must be a non-empty list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.endswith("/"):
            raise HistoricalValidationError(f"{label} entries must end with '/'")
        base = validate_relative_path(item[:-1], label=f"{label} entry")
        result.append(f"{base}/")
    if len(set(result)) != len(result):
        raise HistoricalValidationError(f"{label} contains duplicates")
    return tuple(result)


def load_policy(path: Path = PIN_FILE) -> HistoricalPolicy:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HistoricalValidationError(f"cannot read historical pin file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HistoricalValidationError("historical pin file must contain an object")
    _exact_keys(
        value,
        {
            "schema_version",
            "historical_commit",
            "source_manifests",
            "hybrid_manifests",
            "external_files",
        },
        label="historical pin file",
    )
    if value["schema_version"] != 1:
        raise HistoricalValidationError("unsupported historical pin schema")
    commit = value["historical_commit"]
    if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
        raise HistoricalValidationError("historical_commit must be a full lowercase Git SHA")

    source_manifests = _string_list(
        value["source_manifests"], label="source_manifests"
    )
    hybrid_value = value["hybrid_manifests"]
    if not isinstance(hybrid_value, list) or not hybrid_value:
        raise HistoricalValidationError("hybrid_manifests must be a non-empty list")
    hybrid_manifests: list[HybridManifest] = []
    for index, item in enumerate(hybrid_value):
        if not isinstance(item, dict):
            raise HistoricalValidationError(f"hybrid_manifests[{index}] must be an object")
        _exact_keys(
            item,
            {"path", "historical_prefixes", "current_prefixes"},
            label=f"hybrid_manifests[{index}]",
        )
        historical_prefixes = _prefix_list(
            item["historical_prefixes"],
            label=f"hybrid_manifests[{index}].historical_prefixes",
        )
        current_prefixes = _prefix_list(
            item["current_prefixes"],
            label=f"hybrid_manifests[{index}].current_prefixes",
        )
        for historical in historical_prefixes:
            for current in current_prefixes:
                if historical.startswith(current) or current.startswith(historical):
                    raise HistoricalValidationError(
                        f"hybrid_manifests[{index}] has overlapping prefixes"
                    )
        hybrid_manifests.append(
            HybridManifest(
                path=validate_relative_path(
                    item["path"], label=f"hybrid_manifests[{index}].path"
                ),
                historical_prefixes=historical_prefixes,
                current_prefixes=current_prefixes,
            )
        )
    manifest_paths = source_manifests + tuple(item.path for item in hybrid_manifests)
    if len(set(manifest_paths)) != len(manifest_paths):
        raise HistoricalValidationError("manifest paths must be unique")

    external_value = value["external_files"]
    if not isinstance(external_value, list):
        raise HistoricalValidationError("external_files must be a list")
    external_files: dict[str, str] = {}
    for index, item in enumerate(external_value):
        if not isinstance(item, dict):
            raise HistoricalValidationError(f"external_files[{index}] must be an object")
        _exact_keys(item, {"path", "sha256"}, label=f"external_files[{index}]")
        external_path = validate_relative_path(
            item["path"], label=f"external_files[{index}].path"
        )
        digest = item["sha256"]
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise HistoricalValidationError(
                f"external_files[{index}].sha256 must be lowercase SHA-256"
            )
        if external_path in external_files:
            raise HistoricalValidationError("external_files contains duplicate paths")
        external_files[external_path] = digest

    return HistoricalPolicy(
        commit=commit,
        source_manifests=source_manifests,
        hybrid_manifests=tuple(hybrid_manifests),
        external_files=external_files,
    )


def ensure_historical_commit(root: Path, commit: str) -> None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise HistoricalValidationError(f"cannot execute git: {exc}") from exc
    if result.returncode != 0:
        raise HistoricalValidationError(
            f"historical commit {commit} is unavailable. This checkout may be "
            "shallow; fetch history with `git fetch --unshallow origin` or fetch "
            f"the exact commit with `git fetch origin {commit}`."
        )


def read_git_blob(root: Path, commit: str, relative: str) -> bytes:
    validate_relative_path(relative, label="Git blob path")
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise HistoricalValidationError(f"cannot execute git: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise HistoricalValidationError(
            f"historical blob is unavailable at {commit}:{relative}: {detail}"
        )
    return result.stdout


def _read_current(root: Path, relative: str) -> bytes:
    validate_relative_path(relative, label="checkout path")
    target = root / relative
    try:
        resolved = target.resolve(strict=True)
    except OSError as exc:
        raise HistoricalValidationError(f"cannot read checkout path {relative}: {exc}") from exc
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise HistoricalValidationError(f"unsafe or non-file checkout path: {relative}")
    try:
        return resolved.read_bytes()
    except OSError as exc:
        raise HistoricalValidationError(f"cannot read checkout path {relative}: {exc}") from exc


def parse_manifest(value: bytes, *, label: str) -> tuple[tuple[str, str], ...]:
    try:
        text = value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise HistoricalValidationError(f"{label} is not ASCII") from exc
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise HistoricalValidationError(f"{label}:{line_number} is empty")
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2 or SHA256_RE.fullmatch(parts[0]) is None:
            raise HistoricalValidationError(f"invalid manifest row at {label}:{line_number}")
        relative = validate_relative_path(
            parts[1], label=f"{label}:{line_number} path"
        )
        if relative in seen:
            raise HistoricalValidationError(f"duplicate manifest path in {label}: {relative}")
        seen.add(relative)
        rows.append((parts[0], relative))
    if not rows:
        raise HistoricalValidationError(f"{label} is empty")
    return tuple(rows)


def _verify_external(
    root: Path,
    relative: str,
    expected: str,
    external_files: Mapping[str, str],
) -> None:
    pinned = external_files.get(relative)
    if pinned is None:
        raise HistoricalValidationError(f"unapproved external source path: {relative}")
    if pinned != expected:
        raise HistoricalValidationError(
            f"external pin disagrees with manifest for {relative}: {pinned} != {expected}"
        )
    actual = sha256_bytes(_read_current(root, relative))
    if actual != expected:
        raise HistoricalValidationError(
            f"external source SHA-256 mismatch for {relative}: {actual}"
        )


def _historical_manifest_rows(
    root: Path,
    policy: HistoricalPolicy,
    relative: str,
    *,
    blob_reader: BlobReader,
) -> tuple[tuple[str, str], ...]:
    historical = blob_reader(root, policy.commit, relative)
    current = _read_current(root, relative)
    if current != historical:
        raise HistoricalValidationError(
            f"legacy manifest changed since {policy.commit}: {relative}"
        )
    return parse_manifest(historical, label=relative)


def verify_source_manifest(
    root: Path,
    policy: HistoricalPolicy,
    relative: str,
    *,
    blob_reader: BlobReader = read_git_blob,
) -> int:
    if relative not in policy.source_manifests:
        raise HistoricalValidationError(f"unconfigured source manifest: {relative}")
    rows = _historical_manifest_rows(
        root, policy, relative, blob_reader=blob_reader
    )
    for expected, target in rows:
        if target in policy.external_files:
            _verify_external(root, target, expected, policy.external_files)
            continue
        actual = sha256_bytes(blob_reader(root, policy.commit, target))
        if actual != expected:
            raise HistoricalValidationError(
                f"historical source SHA-256 mismatch for {target}: {actual}"
            )
    return len(rows)


def verify_hybrid_manifest(
    root: Path,
    policy: HistoricalPolicy,
    manifest: HybridManifest,
    *,
    blob_reader: BlobReader = read_git_blob,
) -> int:
    rows = _historical_manifest_rows(
        root, policy, manifest.path, blob_reader=blob_reader
    )
    for expected, target in rows:
        if target in policy.external_files:
            _verify_external(root, target, expected, policy.external_files)
            continue
        historical_matches = sum(
            target.startswith(prefix) for prefix in manifest.historical_prefixes
        )
        current_matches = sum(
            target.startswith(prefix) for prefix in manifest.current_prefixes
        )
        if historical_matches + current_matches != 1:
            raise HistoricalValidationError(
                f"hybrid manifest path has ambiguous or missing policy: {target}"
            )
        value = (
            blob_reader(root, policy.commit, target)
            if historical_matches
            else _read_current(root, target)
        )
        actual = sha256_bytes(value)
        if actual != expected:
            source = "historical source" if historical_matches else "current evidence"
            raise HistoricalValidationError(
                f"{source} SHA-256 mismatch for {target}: {actual}"
            )
    return len(rows)


def verify_policy(
    root: Path = ROOT,
    policy: HistoricalPolicy | None = None,
    *,
    blob_reader: BlobReader = read_git_blob,
) -> dict[str, int | str]:
    policy = policy or load_policy()
    ensure_historical_commit(root, policy.commit)
    source_rows = sum(
        verify_source_manifest(
            root, policy, manifest, blob_reader=blob_reader
        )
        for manifest in policy.source_manifests
    )
    hybrid_rows = sum(
        verify_hybrid_manifest(root, policy, manifest, blob_reader=blob_reader)
        for manifest in policy.hybrid_manifests
    )
    return {
        "commit": policy.commit,
        "source_manifests": len(policy.source_manifests),
        "source_rows": source_rows,
        "hybrid_manifests": len(policy.hybrid_manifests),
        "hybrid_rows": hybrid_rows,
    }


def main() -> int:
    try:
        result = verify_policy()
    except HistoricalValidationError as exc:
        raise SystemExit(f"Historical validation failed: {exc}") from exc
    print(
        "Historical validation passed: "
        f"{result['source_manifests']} source manifests and "
        f"{result['hybrid_manifests']} hybrid manifest at "
        f"{str(result['commit'])[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
