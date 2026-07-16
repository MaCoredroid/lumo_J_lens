#!/usr/bin/env python3
"""Validate the identity and context contract of an OpenAI models endpoint."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def validate_models_payload(payload: object, model: str, max_model_len: int) -> tuple[bool, str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return False, "response has no data array"
    matches = [item for item in payload["data"] if isinstance(item, dict) and item.get("id") == model]
    if not matches:
        ids = [item.get("id") for item in payload["data"] if isinstance(item, dict)]
        return False, f"expected model {model!r}; endpoint serves {ids!r}"
    observed = matches[0].get("max_model_len")
    try:
        observed_int = int(observed)
    except (TypeError, ValueError):
        return False, f"model has invalid max_model_len={observed!r}"
    if observed_int != max_model_len:
        return False, f"expected max_model_len={max_model_len}; observed {observed_int}"
    return True, f"model={model} max_model_len={observed_int}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint", help="OpenAI-compatible base URL, normally ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", required=True, type=int)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    url = args.endpoint.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=args.timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        if not args.quiet:
            print(f"endpoint check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    ok, detail = validate_models_payload(payload, args.model, args.max_model_len)
    if not args.quiet or not ok:
        print(("PASS " if ok else "FAIL ") + detail, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
