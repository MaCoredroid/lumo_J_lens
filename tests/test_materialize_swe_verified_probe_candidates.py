#!/usr/bin/env python3
"""Tests for deterministic SWE Verified initial-probe candidate extraction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_swe_verified_probe_candidates",
    ROOT / "scripts" / "materialize_swe_verified_probe_candidates.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
HAS_UNIDIFF = importlib.util.find_spec("unidiff") is not None


SOURCE_PATCH = """\
diff --git a/src/pkg/engine.py b/src/pkg/engine.py
index 1111111..2222222 100644
--- a/src/pkg/engine.py
+++ b/src/pkg/engine.py
@@ -1,5 +1,5 @@
 class Engine:
     def compute(self, value):
-        result = old_helper(value)
+        result = new_helper(value)
         return result
 def untouched():
@@ -10 +10 @@ def rewrite(left, right):
-    return old_left(left) + old_right(right)
+    return new_left(left) + new_right(right)
diff --git a/tests/test_engine.py b/tests/test_engine.py
index 3333333..4444444 100644
--- a/tests/test_engine.py
+++ b/tests/test_engine.py
@@ -1 +1 @@
-def test_old(): pass
+def test_new(): pass
diff --git a/README.md b/README.md
index 5555555..6666666 100644
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old_helper
+new_helper
"""


def row(instance_id: str = "owner__repo-1") -> dict[str, str]:
    return {
        "repo": "owner/repo",
        "instance_id": instance_id,
        "base_commit": "a" * 40,
        "version": "1.0",
        "problem_statement": "Replace the obsolete helper.",
        "patch": SOURCE_PATCH,
        "test_patch": "test patch bytes\n",
    }


@unittest.skipUnless(HAS_UNIDIFF, "unidiff is installed in .venv-swe")
class CandidateExtractionTest(unittest.TestCase):
    def test_extracts_only_strict_non_test_python_concepts(self) -> None:
        concepts = MODULE.extract_patch_concepts(SOURCE_PATCH)
        compact = [
            (
                concept["path"],
                concept["kind"],
                concept["target"],
                concept.get("contrast"),
            )
            for concept in concepts
        ]
        self.assertEqual(
            compact,
            [
                ("src/pkg/engine.py", "file_stem", "engine", None),
                ("src/pkg/engine.py", "module_dir", "pkg", None),
                ("src/pkg/engine.py", "symbol", "Engine", None),
                ("src/pkg/engine.py", "symbol", "compute", None),
                ("src/pkg/engine.py", "symbol", "rewrite", None),
                (
                    "src/pkg/engine.py",
                    "identifier_replacement",
                    "new_helper",
                    "old_helper",
                ),
            ],
        )
        self.assertTrue(all(concept["sources"] for concept in concepts))
        self.assertFalse(any("test_engine.py" in item[0] for item in compact))
        self.assertFalse(any("README.md" in item[0] for item in compact))

    def test_identifier_pair_is_lexical_and_requires_exactly_one_each(self) -> None:
        self.assertEqual(
            MODULE.exact_identifier_replacement(
                ["return old_name(value)  # removed_comment_name\n"],
                ["return new_name(value)  # added_comment_name\n"],
            ),
            ("new_name", "old_name"),
        )
        self.assertIsNone(
            MODULE.exact_identifier_replacement(
                ['message = "old_name"\n'], ['message = "new_name"\n']
            )
        )
        self.assertIsNone(
            MODULE.exact_identifier_replacement(
                ["return old_a(left) + old_b(right)\n"],
                ["return new_a(left) + new_b(right)\n"],
            )
        )
        self.assertIsNone(
            MODULE.exact_identifier_replacement(["if ready:\n"], ["while ready:\n"])
        )

    def test_manifest_preserves_fields_hashes_provenance_and_canonical_order(self) -> None:
        first = row("owner__repo-2")
        second = row("owner__repo-1")
        manifest = MODULE.build_candidate_manifest(
            [first, second], source={"mode": "synthetic", "sha256": "f" * 64}
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["kind"], "swe_verified_initial_probe_candidates")
        self.assertFalse(manifest["extraction"]["tokenized_for_model"])
        self.assertFalse(manifest["extraction"]["final_tasks_selected"])
        self.assertEqual(manifest["task_count"], 2)
        self.assertEqual(manifest["concept_count"], 12)
        self.assertEqual(
            [task["instance_id"] for task in manifest["tasks"]],
            ["owner__repo-1", "owner__repo-2"],
        )
        task = manifest["tasks"][0]
        self.assertEqual(task["repo"], second["repo"])
        self.assertEqual(task["base_commit"], second["base_commit"])
        self.assertEqual(task["version"], second["version"])
        self.assertEqual(task["problem_statement"], second["problem_statement"])
        self.assertEqual(
            task["patch_sha256"],
            hashlib.sha256(second["patch"].encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            task["test_patch_sha256"],
            hashlib.sha256(second["test_patch"].encode("utf-8")).hexdigest(),
        )
        self.assertEqual(task["source_provenance"]["dataset_row_index"], 1)

    def test_local_json_cli_is_offline_and_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.json"
            first_output = root / "first.json"
            second_output = root / "second.json"
            dataset.write_text(
                json.dumps([row("owner__repo-2"), row("owner__repo-1")]),
                encoding="utf-8",
            )
            with mock.patch.object(
                MODULE,
                "load_pinned_rows",
                side_effect=AssertionError("local mode attempted a network load"),
            ):
                self.assertEqual(
                    MODULE.main(
                        ["--dataset-json", str(dataset), "--output", str(first_output)]
                    ),
                    0,
                )
                self.assertEqual(
                    MODULE.main(
                        ["--dataset-json", str(dataset), "--output", str(second_output)]
                    ),
                    0,
                )
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
            manifest = json.loads(first_output.read_text(encoding="ascii"))
            self.assertEqual(manifest["source"]["mode"], "local_json")
            self.assertEqual(
                manifest["source"]["sha256"],
                hashlib.sha256(dataset.read_bytes()).hexdigest(),
            )

    def test_rejects_duplicate_instances_and_non_array_local_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate instance_id"):
            MODULE.build_candidate_manifest(
                [row(), row()], source={"mode": "synthetic"}
            )
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset.json"
            dataset.write_text('{"not": "an array"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "top-level array"):
                MODULE.load_local_rows(dataset)


class PinnedSourceTest(unittest.TestCase):
    def test_remote_loader_receives_the_immutable_dataset_pin(self) -> None:
        calls: list[tuple[str, dict[str, str]]] = []

        def loader(repo: str, **kwargs: str) -> list[dict[str, str]]:
            calls.append((repo, kwargs))
            return []

        rows, source = MODULE.load_pinned_rows(loader)
        self.assertEqual(rows, [])
        self.assertEqual(
            calls,
            [
                (
                    MODULE.DATASET_REPO,
                    {
                        "revision": MODULE.DATASET_REVISION,
                        "split": MODULE.DATASET_SPLIT,
                    },
                )
            ],
        )
        self.assertEqual(source["repo_id"], MODULE.DATASET_REPO)
        self.assertEqual(source["revision"], MODULE.DATASET_REVISION)
        self.assertEqual(source["split"], MODULE.DATASET_SPLIT)


if __name__ == "__main__":
    unittest.main()
