#!/usr/bin/env python3
"""Resolve a SWE-bench instance image, preferring a certified digest pin."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "swe_image_digests.json"


def docker_arch() -> str:
    return "arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "x86_64"


def resolve_image(instance_id: str, config: Path = DEFAULT_CONFIG) -> dict[str, str | bool]:
    arch = docker_arch()
    slug = instance_id.replace("__", "_1776_")
    tag = f"swebench/sweb.eval.{arch}.{slug}:latest"
    pinned: dict[str, str] | None = None
    if config.is_file():
        payload = json.loads(config.read_text(encoding="utf-8"))
        value = payload.get("images", {}).get(instance_id, {}).get(arch)
        if isinstance(value, dict):
            pinned = value
    return {
        "tag": tag,
        "reference": pinned.get("reference", tag) if pinned else tag,
        "image_id": pinned.get("image_id", "") if pinned else "",
        "pinned": bool(pinned),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance_id")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--field", choices=("tag", "reference", "image_id", "pinned"))
    args = parser.parse_args()
    result = resolve_image(args.instance_id, args.config)
    if args.field:
        print(str(result[args.field]).lower() if isinstance(result[args.field], bool) else result[args.field])
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
