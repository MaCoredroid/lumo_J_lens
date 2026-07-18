#!/usr/bin/env python3
"""Verify published evidence, dependency locks, and certified runtime sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from check_historical_validation import (
    HistoricalValidationError,
    ensure_historical_commit,
    load_policy,
    read_git_blob,
    verify_source_manifest,
)


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


def require_bytes_hash(value: bytes, expected: str, *, label: str) -> None:
    actual = hashlib.sha256(value).hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch for {label}: {actual}")


def validate_registry_extension(historical_bytes: bytes, current_bytes: bytes) -> int:
    try:
        historical = json.loads(historical_bytes)
        current = json.loads(current_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid image digest registry JSON: {exc}") from exc
    expected_keys = {"schema_version", "images"}
    for label, value in (("historical", historical), ("current", current)):
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise SystemExit(f"{label} image digest registry has an invalid schema")
        if not isinstance(value["schema_version"], int) or isinstance(
            value["schema_version"], bool
        ):
            raise SystemExit(f"{label} image digest registry schema version is invalid")
        if not isinstance(value["images"], dict):
            raise SystemExit(f"{label} image digest registry images must be an object")
    if current["schema_version"] != historical["schema_version"]:
        raise SystemExit("current image digest registry changed the historical schema")
    for instance_id, image in historical["images"].items():
        if current["images"].get(instance_id) != image:
            raise SystemExit(
                f"current image digest registry changed historical entry {instance_id}"
            )
    return len(historical["images"])


def main() -> int:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    policy = load_policy()
    try:
        ensure_historical_commit(ROOT, policy.commit)
    except HistoricalValidationError as exc:
        raise SystemExit(str(exc)) from exc
    checks = {
        VALIDATION / "sympy__sympy-13480.patch": record["task"]["patch_sha256"],
        VALIDATION / "official-report.json": record["official_harness"]["report_sha256"],
        VALIDATION / "runtime-source-manifest.sha256": record["runtime_source_manifest_sha256"],
        VALIDATION / "vllm-freeze.txt": record["dependency_lock_sha256"]["vllm_freeze"],
        VALIDATION / "swe-freeze.txt": record["dependency_lock_sha256"]["swe_freeze"],
        ROOT / "package-lock.json": record["dependency_lock_sha256"]["package_lock"],
    }
    for path, expected in checks.items():
        require_hash(path, expected)

    historical_registry = read_git_blob(
        ROOT, policy.commit, "configs/swe_image_digests.json"
    )
    require_bytes_hash(
        historical_registry,
        record["dependency_lock_sha256"]["image_digest_map"],
        label=f"{policy.commit}:configs/swe_image_digests.json",
    )
    historical_image_count = validate_registry_extension(
        historical_registry,
        (ROOT / "configs" / "swe_image_digests.json").read_bytes(),
    )
    try:
        verify_source_manifest(
            ROOT,
            policy,
            "validation/runtime-source-manifest.sha256",
        )
    except HistoricalValidationError as exc:
        raise SystemExit(str(exc)) from exc

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
    print(
        f"Validation integrity passed: {len(checks)} evidence/lock hashes, "
        f"{historical_image_count} preserved image pin, and historical runtime manifest"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
