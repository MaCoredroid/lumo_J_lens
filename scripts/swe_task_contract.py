#!/usr/bin/env python3
"""Render the byte-stable SWE task contract injected into Qwen Code."""

from __future__ import annotations

from typing import Any, Mapping


def render_agents_md(instance: Mapping[str, Any]) -> str:
    body: list[str] = []
    body.append(f"# SWE-Bench task: {instance['instance_id']}")
    body.append("")
    body.append(f"**Repo:** `{instance['repo']}`  ")
    body.append(f"**Base commit:** `{instance['base_commit']}`  ")
    if instance.get("version"):
        body.append(f"**Version:** `{instance['version']}`  ")
    body.append("")
    body.append("## Problem statement")
    body.append("")
    body.append(instance.get("problem_statement") or "(empty problem statement)")
    body.append("")
    body.append("## Required behavior")
    body.append("")
    body.append(
        "Implement the fix described in the problem statement by editing the "
        "source files in this workspace. Do NOT modify any test files. The "
        "hidden grader will apply its own test patch and run the test suite; "
        "your code must make those tests pass without breaking existing ones."
    )
    body.append("")
    body.append("## How to work (important)")
    body.append("")
    body.append(
        "- Reason carefully and thoroughly before each tool call. First inspect "
        "the relevant source files to confirm your understanding of the bug, "
        "then make the minimal correct edit.\n"
        "- Do NOT spend your time trying to `pip install` or build/conda the "
        "project -- the grader runs in its own prepared environment. If an "
        "install/build command fails, do not retry it; just edit the source.\n"
        "- You MUST finish by leaving an actual code change in the working tree. "
        "Do not stop until you have edited the source files to implement the fix."
    )
    body.append("")
    return "\n".join(body) + "\n"
