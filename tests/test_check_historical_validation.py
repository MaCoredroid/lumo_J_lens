#!/usr/bin/env python3
"""Tests for commit-pinned historical validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_historical_validation as historical  # noqa: E402
import check_validation  # noqa: E402


COMMIT = "a" * 40


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def manifest(*rows: tuple[bytes, str]) -> bytes:
    return "".join(f"{digest(value)}  {path}\n" for value, path in rows).encode()


class HistoricalValidationTests(unittest.TestCase):
    def policy(
        self,
        *,
        source_manifests: tuple[str, ...] = ("validation/source.sha256",),
        hybrid_manifests: tuple[historical.HybridManifest, ...] = (),
        external_files: dict[str, str] | None = None,
    ) -> historical.HistoricalPolicy:
        return historical.HistoricalPolicy(
            commit=COMMIT,
            source_manifests=source_manifests,
            hybrid_manifests=hybrid_manifests,
            external_files=external_files or {},
        )

    def test_committed_policy_loads_with_full_commit_pin(self) -> None:
        policy = historical.load_policy()
        self.assertEqual(
            policy.commit,
            "b69c3f3004965520c0707755f412cf764f5fc27b",
        )
        self.assertIn(
            "validation/runtime-source-manifest.sha256",
            policy.source_manifests,
        )

    def test_historical_source_uses_git_blob_not_changed_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "validation").mkdir()
            (root / "scripts").mkdir()
            old_source = b"old source\n"
            source_manifest = manifest((old_source, "scripts/tool.py"))
            (root / "validation/source.sha256").write_bytes(source_manifest)
            (root / "scripts/tool.py").write_bytes(b"new source\n")
            blobs = {
                "validation/source.sha256": source_manifest,
                "scripts/tool.py": old_source,
            }

            count = historical.verify_source_manifest(
                root,
                self.policy(),
                "validation/source.sha256",
                blob_reader=lambda _root, _commit, path: blobs[path],
            )

            self.assertEqual(count, 1)

    def test_changed_legacy_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "validation").mkdir()
            old_manifest = manifest((b"old", "scripts/tool.py"))
            (root / "validation/source.sha256").write_bytes(
                manifest((b"changed", "scripts/tool.py"))
            )
            blobs = {
                "validation/source.sha256": old_manifest,
                "scripts/tool.py": b"old",
            }
            with self.assertRaisesRegex(
                historical.HistoricalValidationError, "legacy manifest changed"
            ):
                historical.verify_source_manifest(
                    root,
                    self.policy(),
                    "validation/source.sha256",
                    blob_reader=lambda _root, _commit, path: blobs[path],
                )

    def test_hybrid_manifest_checks_current_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "validation").mkdir()
            (root / "scripts").mkdir()
            source = b"old source"
            evidence = b"evidence"
            mixed = manifest(
                (source, "scripts/tool.py"),
                (evidence, "validation/report.json"),
            )
            (root / "validation/mixed.sha256").write_bytes(mixed)
            (root / "validation/report.json").write_bytes(evidence)
            blobs = {
                "validation/mixed.sha256": mixed,
                "scripts/tool.py": source,
            }
            policy = self.policy(
                source_manifests=(),
                hybrid_manifests=(
                    historical.HybridManifest(
                        path="validation/mixed.sha256",
                        historical_prefixes=("scripts/",),
                        current_prefixes=("validation/",),
                    ),
                ),
            )
            (root / "validation/report.json").write_bytes(b"tampered")

            with self.assertRaisesRegex(
                historical.HistoricalValidationError, "current evidence SHA-256"
            ):
                historical.verify_hybrid_manifest(
                    root,
                    policy,
                    policy.hybrid_manifests[0],
                    blob_reader=lambda _root, _commit, path: blobs[path],
                )

    def test_external_file_requires_independent_pin_and_current_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "validation").mkdir()
            (root / "configs").mkdir()
            external = b"template"
            source_manifest = manifest((external, "configs/template.jinja"))
            (root / "validation/source.sha256").write_bytes(source_manifest)
            (root / "configs/template.jinja").write_bytes(b"tampered")
            policy = self.policy(
                external_files={"configs/template.jinja": digest(external)}
            )
            with self.assertRaisesRegex(
                historical.HistoricalValidationError, "external source SHA-256"
            ):
                historical.verify_source_manifest(
                    root,
                    policy,
                    "validation/source.sha256",
                    blob_reader=lambda _root, _commit, path: {
                        "validation/source.sha256": source_manifest
                    }[path],
                )

    def test_hybrid_unknown_prefix_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "validation").mkdir()
            mixed = manifest((b"unknown", "docs/result.txt"))
            (root / "validation/mixed.sha256").write_bytes(mixed)
            policy = self.policy(
                source_manifests=(),
                hybrid_manifests=(
                    historical.HybridManifest(
                        path="validation/mixed.sha256",
                        historical_prefixes=("scripts/",),
                        current_prefixes=("validation/",),
                    ),
                ),
            )
            with self.assertRaisesRegex(
                historical.HistoricalValidationError, "missing policy"
            ):
                historical.verify_hybrid_manifest(
                    root,
                    policy,
                    policy.hybrid_manifests[0],
                    blob_reader=lambda _root, _commit, path: {
                        "validation/mixed.sha256": mixed
                    }[path],
                )

    def test_pin_parser_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pin = Path(tmp) / "pins.json"
            pin.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "historical_commit": COMMIT,
                        "source_manifests": ["../outside.sha256"],
                        "hybrid_manifests": [
                            {
                                "path": "validation/mixed.sha256",
                                "historical_prefixes": ["scripts/"],
                                "current_prefixes": ["validation/"],
                            }
                        ],
                        "external_files": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                historical.HistoricalValidationError, "unsafe"
            ):
                historical.load_policy(pin)

    def test_missing_commit_reports_shallow_clone_recovery(self) -> None:
        result = subprocess.CompletedProcess([], 1, stderr="missing")
        with mock.patch.object(historical.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(
                historical.HistoricalValidationError, "shallow.*git fetch"
            ):
                historical.ensure_historical_commit(Path("/repo"), COMMIT)

    def test_registry_extension_preserves_historical_entries(self) -> None:
        old = {"schema_version": 1, "images": {"task-a": {"x86_64": "pin"}}}
        new = {
            "schema_version": 1,
            "images": {
                "task-a": {"x86_64": "pin"},
                "task-b": {"x86_64": "new-pin"},
            },
        }
        self.assertEqual(
            check_validation.validate_registry_extension(
                json.dumps(old).encode(), json.dumps(new).encode()
            ),
            1,
        )

    def test_registry_extension_rejects_rebound_historical_entry(self) -> None:
        old = {"schema_version": 1, "images": {"task-a": {"x86_64": "pin"}}}
        new = {"schema_version": 1, "images": {"task-a": {"x86_64": "other"}}}
        with self.assertRaisesRegex(SystemExit, "changed historical entry"):
            check_validation.validate_registry_extension(
                json.dumps(old).encode(), json.dumps(new).encode()
            )


if __name__ == "__main__":
    unittest.main()
