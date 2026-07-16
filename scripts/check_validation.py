#!/usr/bin/env python3
"""Verify published evidence, dependency locks, and certified runtime sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"
RECORD = VALIDATION / "2026-07-15-publication-certified.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch for {path.relative_to(ROOT)}: {actual}")


def main() -> int:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    checks = {
        VALIDATION / "sympy__sympy-13480.patch": record["task"]["patch_sha256"],
        VALIDATION / "official-report.json": record["official_harness"]["report_sha256"],
        VALIDATION / "runtime-source-manifest.sha256": record["runtime_source_manifest_sha256"],
        VALIDATION / "vllm-freeze.txt": record["dependency_lock_sha256"]["vllm_freeze"],
        VALIDATION / "swe-freeze.txt": record["dependency_lock_sha256"]["swe_freeze"],
        ROOT / "package-lock.json": record["dependency_lock_sha256"]["package_lock"],
        ROOT / "configs" / "swe_image_digests.json":
            record["dependency_lock_sha256"]["image_digest_map"],
    }
    for path, expected in checks.items():
        require_hash(path, expected)

    manifest = VALIDATION / "runtime-source-manifest.sha256"
    for line in manifest.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        target = (ROOT / relative).resolve()
        if not target.is_relative_to(ROOT.resolve()):
            raise SystemExit(f"unsafe manifest path: {relative}")
        require_hash(target, expected)

    request = json.loads((VALIDATION / "first-forwarded-request.json").read_text())
    expected_envelope = {
        "model": record["model"]["served_name"],
        "temperature": record["agent"]["temperature"],
        "top_p": record["agent"]["top_p"],
        "top_k": record["agent"]["top_k"],
        "min_p": record["agent"]["min_p"],
        "presence_penalty": record["agent"]["presence_penalty"],
        "seed": record["agent"]["initial_seed"],
        "max_tokens": 8192,
        "chat_template_kwargs": {"enable_thinking": record["agent"]["thinking"]},
        "tools": record["agent_boundary"]["declared_tools"],
    }
    if request != expected_envelope:
        raise SystemExit("first-forwarded-request.json does not match the certified record")

    report = json.loads((VALIDATION / "official-report.json").read_text())
    if report.get("resolved_ids") != [record["task"]["instance_id"]]:
        raise SystemExit("official report does not resolve the certified task")
    print(f"Validation integrity passed: {len(checks)} evidence/lock hashes and runtime manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
