#!/usr/bin/env python3
"""Tests for the portable SWE AGENTS contract renderer."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from swe_task_contract import render_agents_md  # noqa: E402


class SweTaskContractTest(unittest.TestCase):
    def test_renderer_is_byte_stable(self) -> None:
        rendered = render_agents_md(
            {
                "instance_id": "owner__repo-1",
                "repo": "owner/repo",
                "base_commit": "a" * 40,
                "version": "1.2",
                "problem_statement": "Fix the failing behavior.\r\nKeep CRLF.",
            }
        )

        self.assertTrue(rendered.endswith("\n"))
        self.assertIn("# SWE-Bench task: owner__repo-1\n", rendered)
        self.assertIn("**Repo:** `owner/repo`  \n", rendered)
        self.assertIn("**Version:** `1.2`  \n", rendered)
        self.assertIn(
            "## Problem statement\n\nFix the failing behavior.\r\nKeep CRLF.\n",
            rendered,
        )
        self.assertEqual(rendered.count("Do NOT modify any test files."), 1)


if __name__ == "__main__":
    unittest.main()
