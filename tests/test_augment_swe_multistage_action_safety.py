#!/usr/bin/env python3
"""Focused safety tests for action/outcome label derivation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/augment_swe_multistage_action_probes.py"
SPEC = importlib.util.spec_from_file_location("augment_swe_multistage_safety", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUGMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUGMENT)


def shell_result(command: str, output: str, *, exit_code: int = 0) -> str:
    return (
        f"Command: {command}\n"
        "Directory: /testbed\n"
        f"Output: {output}\n"
        "Error: (none)\n"
        f"Exit Code: {exit_code}\n"
        "Signal: 0\n"
        "Process Group PGID: 123"
    )


class AugmentSweMultistageActionSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lifecycle = json.loads(
            (ROOT / "configs/swe_multistage_protocol.json").read_text(encoding="utf-8")
        )
        validated = AUGMENT.validate_lifecycle_protocol(lifecycle)
        cls.positive = validated["validation_positive_regexes"]
        cls.negative = validated["validation_negative_regexes"]

    def outcome(self, output: str, *, exit_code: int = 0) -> tuple[bool, dict]:
        command = "python -m pytest tests/test_module.py -q 2>&1 | tail -20"
        return AUGMENT._tool_success(
            "run_shell_command",
            shell_result(command, output, exit_code=exit_code),
            command,
            test_command=True,
            validation_positive_regexes=self.positive,
            validation_negative_regexes=self.negative,
        )

    def test_test_pipeline_requires_positive_evidence(self) -> None:
        success, evidence = self.outcome("2 passed in 0.10s")
        self.assertTrue(success)
        self.assertTrue(evidence["validation_positive_output_regex_hits"])
        self.assertFalse(evidence["validation_negative_output_regex_hits"])

    def test_exit_zero_masked_test_failures_are_failures(self) -> None:
        rejected = (
            "FAILED tests/test_module.py::test_case",
            "Traceback (most recent call last):\nRuntimeError: broken",
            "1 failed, 2 passed in 0.10s",
            "0 passed in 0.01s",
            "0 tests passed",
            "no tests ran in 0.01s",
            "collected 0 items",
            "command completed",
        )
        for output in rejected:
            with self.subTest(output=output):
                success, _ = self.outcome(output)
                self.assertFalse(success)

    def test_nonzero_test_exit_fails_despite_positive_output(self) -> None:
        success, _ = self.outcome("2 passed in 0.10s", exit_code=1)
        self.assertFalse(success)

    def test_non_repository_sinks_are_not_source_edits(self) -> None:
        mutation_regexes = [
            AUGMENT.re.compile(
                r"(?im)(?:^|[;&|]\s*|\n\s*)(?:cat|echo|printf)\b[^\n]*(?:>>?|\btee\b)"
            )
        ]
        source_paths = ["src/module.py"]
        commands = (
            "cat src/module.py > /dev/null",
            "cat src/module.py > /tmp/output.py",
            "cat src/module.py > /var/tmp/output.py",
            "cat src/module.py > $HOME/output.py",
            "cat src/module.py > ${HOME}/output.py",
            "cat src/module.py > ~/output.py",
            "cat src/module.py > /home/mark/output.py",
        )
        for command in commands:
            with self.subTest(command=command):
                mutated, hits = AUGMENT._source_mutation(
                    command,
                    source_paths=source_paths,
                    mutation_regexes=mutation_regexes,
                )
                self.assertTrue(hits)
                self.assertFalse(mutated)

    def test_repository_relative_and_known_absolute_targets_are_source_edits(self) -> None:
        mutation_regexes = [AUGMENT.re.compile(r">{1,2}")]
        for command in (
            "cat src/module.py > tests/test_regression.py",
            "cat replacement.py > /testbed/src/module.py",
        ):
            with self.subTest(command=command):
                mutated, _ = AUGMENT._source_mutation(
                    command,
                    source_paths=["src/module.py"],
                    mutation_regexes=mutation_regexes,
                )
                self.assertTrue(mutated)

    def test_source_in_place_edit_remains_edit_when_output_is_discarded(self) -> None:
        mutated, _ = AUGMENT._source_mutation(
            "sed -i 's/old/new/' /testbed/src/module.py > /dev/null",
            source_paths=["src/module.py"],
            mutation_regexes=[AUGMENT.re.compile(r"\bsed\b[^\n;|]*\s-i\b")],
        )
        self.assertTrue(mutated)

    def test_patch_dry_runs_are_not_source_edits(self) -> None:
        mutation_regexes = [
            AUGMENT.re.compile(
                r"(?im)\b(?:apply_patch|git\s+apply|patch\b[^\n;|&]*\s-p\d+)\b"
            )
        ]
        for command in (
            "git apply --check fix.patch",
            "git apply --stat fix.patch",
            "git apply --numstat fix.patch",
            "git apply --summary fix.patch",
            "patch --dry-run -p1 < fix.patch",
            "patch -p1 --dry-run < fix.patch",
        ):
            with self.subTest(command=command):
                mutated, hits = AUGMENT._source_mutation(
                    command,
                    source_paths=["src/module.py"],
                    mutation_regexes=mutation_regexes,
                )
                self.assertFalse(mutated)

    def test_real_patch_applications_remain_source_edits(self) -> None:
        mutation_regexes = [
            AUGMENT.re.compile(
                r"(?im)\b(?:apply_patch|git\s+apply|patch\b[^\n;|&]*\s-p\d+)\b"
            )
        ]
        for command in (
            "git apply fix.patch",
            "patch -p1 < fix.patch",
            "apply_patch < fix.patch",
            "git apply --check fix.patch && git apply fix.patch",
            "git apply --check first.patch | git apply second.patch",
            "git apply --check first.patch & git apply second.patch",
        ):
            with self.subTest(command=command):
                mutated, hits = AUGMENT._source_mutation(
                    command,
                    source_paths=["src/module.py"],
                    mutation_regexes=mutation_regexes,
                )
                self.assertTrue(hits)
                self.assertTrue(mutated)


if __name__ == "__main__":
    unittest.main()
