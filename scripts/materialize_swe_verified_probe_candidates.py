#!/usr/bin/env python3
"""Extract deterministic, untokenized probe candidates from SWE Verified patches."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import keyword
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import textwrap
import tokenize
from typing import Any, Callable, Iterable, Mapping, Sequence


DATASET_REPO = "princeton-nlp/SWE-bench_Verified"
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
DATASET_SPLIT = "test"
KIND_ORDER = {
    "file_stem": 0,
    "module_dir": 1,
    "symbol": 2,
    "identifier_replacement": 3,
}
TEST_DIRECTORY_NAMES = frozenset({"test", "tests", "testing"})
SYMBOL_RE = re.compile(
    r"^\s*(?:async\s+)?(?P<symbol_type>def|class)\s+"
    r"(?P<name>[^\W\d]\w*)\b"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


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


def _require_string(row: Mapping[str, Any], field: str, instance: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ValueError(f"row {instance!r} field {field!r} must be a string")
    return value


def _patch_path(patched_file: Any) -> str:
    raw_path = patched_file.target_file
    if raw_path == "/dev/null":
        raw_path = patched_file.source_file
    if raw_path.startswith(("a/", "b/")):
        raw_path = raw_path[2:]
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe path in gold patch: {raw_path!r}")
    return path.as_posix()


def is_non_test_python_path(path: str) -> bool:
    parsed = PurePosixPath(path)
    if parsed.suffix != ".py":
        return False
    lowered_parts = tuple(part.lower() for part in parsed.parts)
    filename = lowered_parts[-1]
    if any(part in TEST_DIRECTORY_NAMES for part in lowered_parts[:-1]):
        return False
    return not (
        filename == "conftest.py"
        or filename in {"test.py", "tests.py"}
        or filename.startswith("test_")
        or filename.endswith("_test.py")
    )


def _identifier_counts(source: str) -> Counter[str] | None:
    counts: Counter[str] = Counter()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(textwrap.dedent(source)).readline)
        for token in tokens:
            if token.type == tokenize.NAME and not keyword.iskeyword(token.string):
                counts[token.string] += 1
    except (IndentationError, tokenize.TokenError):
        return None
    return counts


def exact_identifier_replacement(
    removed_lines: Sequence[str], added_lines: Sequence[str]
) -> tuple[str, str] | None:
    """Return (added, removed) only for one exact lexical name replacement."""
    if not removed_lines or not added_lines:
        return None
    removed = _identifier_counts("".join(removed_lines))
    added = _identifier_counts("".join(added_lines))
    if removed is None or added is None:
        return None
    removed_only = removed - added
    added_only = added - removed
    if sum(removed_only.values()) != 1 or sum(added_only.values()) != 1:
        return None
    contrast = next(iter(removed_only))
    target = next(iter(added_only))
    if target == contrast:
        return None
    return target, contrast


def _change_blocks(hunk: Any) -> Iterable[tuple[int, list[str], list[str]]]:
    block_index = 0
    removed: list[str] = []
    added: list[str] = []
    for line in hunk:
        if line.is_context:
            if removed or added:
                yield block_index, removed, added
                block_index += 1
                removed, added = [], []
            continue
        if line.is_removed:
            removed.append(line.value)
        elif line.is_added:
            added.append(line.value)
    if removed or added:
        yield block_index, removed, added


def _indentation(source: str) -> int:
    expanded = source.expandtabs(8)
    return len(expanded) - len(expanded.lstrip(" "))


def _hunk_symbols(hunk: Any) -> list[str]:
    """Return declarations that name the hunk or enclose a changed block."""
    symbols: list[str] = []

    def add(source: str) -> re.Match[str] | None:
        match = SYMBOL_RE.match(source or "")
        if match is not None and match.group("name") not in symbols:
            symbols.append(match.group("name"))
        return match

    add(hunk.section_header)
    active: list[tuple[int, str]] = []
    lines = list(hunk)
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.is_context:
            stripped = line.value.strip()
            if stripped and not stripped.startswith(("#", "@")):
                indentation = _indentation(line.value)
                active = [scope for scope in active if scope[0] < indentation]
                match = SYMBOL_RE.match(line.value)
                if match is not None:
                    active.append((indentation, match.group("name")))
            index += 1
            continue

        changed: list[Any] = []
        while index < len(lines) and not lines[index].is_context:
            changed.append(lines[index])
            index += 1
        meaningful = [
            changed_line.value
            for changed_line in changed
            if changed_line.value.strip()
            and not changed_line.value.lstrip().startswith("#")
        ]
        if meaningful:
            change_indent = min(_indentation(source) for source in meaningful)
            active = [scope for scope in active if scope[0] < change_indent]
        for _, name in active:
            if name not in symbols:
                symbols.append(name)
        for changed_line in changed:
            match = add(changed_line.value)
            if match is not None and changed_line.is_added:
                indentation = _indentation(changed_line.value)
                active = [scope for scope in active if scope[0] < indentation]
                active.append((indentation, match.group("name")))
    return symbols


def _source_key(source: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        source["file_index"],
        source.get("hunk_index", -1),
        source.get("change_block_index", -1),
        source["derivation"],
    )


def _concept_key(concept: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        concept["path"],
        KIND_ORDER[concept["kind"]],
        concept["target"],
        concept.get("contrast", ""),
    )


def extract_patch_concepts(patch: str) -> list[dict[str, Any]]:
    try:
        from unidiff import PatchSet
    except ImportError as exc:  # pragma: no cover - exercised by the CLI environment
        raise RuntimeError(
            "unidiff is required; run this script with .venv-swe/bin/python"
        ) from exc

    parsed = PatchSet(patch)
    concepts: dict[tuple[str, str, str, str | None], dict[str, Any]] = {}

    def add_concept(
        *,
        path: str,
        kind: str,
        target: str,
        source: dict[str, Any],
        contrast: str | None = None,
    ) -> None:
        key = (path, kind, target, contrast)
        existing = concepts.get(key)
        if existing is None:
            existing = {"path": path, "kind": kind, "target": target, "sources": []}
            if contrast is not None:
                existing["contrast"] = contrast
            concepts[key] = existing
        if source not in existing["sources"]:
            existing["sources"].append(source)

    for file_index, patched_file in enumerate(parsed):
        path = _patch_path(patched_file)
        if not is_non_test_python_path(path):
            continue
        parsed_path = PurePosixPath(path)
        path_source = {
            "artifact": "patch",
            "file_index": file_index,
            "derivation": "file_path",
        }
        add_concept(
            path=path,
            kind="file_stem",
            target=parsed_path.stem,
            source=path_source,
        )
        if parsed_path.parent != PurePosixPath("."):
            add_concept(
                path=path,
                kind="module_dir",
                target=parsed_path.parent.name,
                source=path_source,
            )

        for hunk_index, hunk in enumerate(patched_file):
            for symbol in _hunk_symbols(hunk):
                add_concept(
                    path=path,
                    kind="symbol",
                    target=symbol,
                    source={
                        "artifact": "patch",
                        "file_index": file_index,
                        "hunk_index": hunk_index,
                        "derivation": "hunk_def_or_class",
                    },
                )

            for block_index, removed, added in _change_blocks(hunk):
                replacement = exact_identifier_replacement(removed, added)
                if replacement is None:
                    continue
                target, contrast = replacement
                add_concept(
                    path=path,
                    kind="identifier_replacement",
                    target=target,
                    contrast=contrast,
                    source={
                        "artifact": "patch",
                        "file_index": file_index,
                        "hunk_index": hunk_index,
                        "change_block_index": block_index,
                        "derivation": "exact_identifier_replacement",
                    },
                )

    result = sorted(concepts.values(), key=_concept_key)
    for concept in result:
        concept["sources"].sort(key=_source_key)
    return result


def build_candidate_manifest(
    rows: Sequence[Mapping[str, Any]], *, source: Mapping[str, Any]
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    seen_instances: set[str] = set()
    for row_index, row in enumerate(rows):
        provisional_instance = row.get("instance_id", f"row-{row_index}")
        if not isinstance(provisional_instance, str):
            provisional_instance = f"row-{row_index}"
        instance_id = _require_string(row, "instance_id", provisional_instance)
        if not instance_id:
            raise ValueError(f"row {row_index} has an empty instance_id")
        if instance_id in seen_instances:
            raise ValueError(f"duplicate instance_id: {instance_id}")
        seen_instances.add(instance_id)

        patch = _require_string(row, "patch", instance_id)
        test_patch = _require_string(row, "test_patch", instance_id)
        try:
            concepts = extract_patch_concepts(patch)
        except Exception as exc:
            raise ValueError(f"failed to parse gold patch for {instance_id}: {exc}") from exc
        tasks.append(
            {
                "repo": _require_string(row, "repo", instance_id),
                "instance_id": instance_id,
                "base_commit": _require_string(row, "base_commit", instance_id),
                "version": _require_string(row, "version", instance_id),
                "problem_statement": _require_string(
                    row, "problem_statement", instance_id
                ),
                "patch_sha256": sha256_text(patch),
                "test_patch_sha256": sha256_text(test_patch),
                "source_provenance": {
                    "dataset_row_index": row_index,
                    "gold_patch_field": "patch",
                    "test_patch_field": "test_patch",
                },
                "concepts": concepts,
            }
        )

    tasks.sort(key=lambda task: task["instance_id"])
    return {
        "schema_version": 1,
        "kind": "swe_verified_initial_probe_candidates",
        "source": dict(source),
        "extraction": {
            "parser": "unidiff.PatchSet",
            "artifact": "gold patch field only",
            "python_path_rule": "suffix .py",
            "test_directory_names_excluded": sorted(TEST_DIRECTORY_NAMES),
            "test_filename_rules": [
                "test.py",
                "tests.py",
                "conftest.py",
                "test_*.py",
                "*_test.py",
            ],
            "concept_kinds": list(KIND_ORDER),
            "identifier_rule": (
                "one added and one removed non-keyword tokenize.NAME after "
                "multiset cancellation within one contiguous change block"
            ),
            "tokenized_for_model": False,
            "final_tasks_selected": False,
            "task_order": "instance_id ascending",
        },
        "task_count": len(tasks),
        "concept_count": sum(len(task["concepts"]) for task in tasks),
        "tasks": tasks,
    }


def load_local_rows(path: Path) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("local dataset JSON must be a top-level array of objects")
    return value, {
        "mode": "local_json",
        "path": path.as_posix(),
        "sha256": sha256_bytes(raw),
    }


def load_pinned_rows(
    loader: Callable[..., Any] | None = None,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset
    dataset = loader(
        DATASET_REPO,
        revision=DATASET_REVISION,
        split=DATASET_SPLIT,
    )
    return [dict(row) for row in dataset], {
        "mode": "huggingface_dataset",
        "repo_id": DATASET_REPO,
        "revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-json",
        type=Path,
        help="offline JSON array used instead of the pinned Hugging Face dataset",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dataset_json is None:
        rows, source = load_pinned_rows()
    else:
        rows, source = load_local_rows(args.dataset_json)
    manifest = build_candidate_manifest(rows, source=source)
    atomic_write_json(args.output, manifest)
    digest = sha256_bytes(args.output.read_bytes())
    print(
        f"wrote {args.output} ({manifest['task_count']} tasks, "
        f"{manifest['concept_count']} concepts, sha256={digest})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
