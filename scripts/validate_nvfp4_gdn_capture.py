#!/usr/bin/env python3
"""Validate analytic GDN recurrence against a captured vLLM NVFP4 prefill."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from nvfp4_ste import gdn_reference_forward


LAYERS = (61, 62)
KEY_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    difference = actual.float() - expected.float()
    rms = float(difference.square().mean().sqrt())
    reference_rms = float(expected.float().square().mean().sqrt())
    return {
        "exact_after_dtype_cast": torch.equal(actual.to(expected.dtype), expected),
        "max_abs": float(difference.abs().max()),
        "rms": rms,
        "reference_rms": reference_rms,
        "relative_rms": rms / max(reference_rms, 1e-30),
        "finite": bool(torch.isfinite(actual).all()),
    }


def validate_capture(
    path: Path,
    *,
    max_output_relative_rms: float = 0.01,
    max_state_relative_rms: float = 0.01,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("model_revision") != "0893e1606ff3d5f97a441f405d5fc541a6bdf404":
        raise ValueError("capture model revision is not the pinned NVIDIA revision")
    tensors = payload.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError("capture lacks a tensor mapping")

    layer_results: dict[str, Any] = {}
    all_pass = True
    for layer in LAYERS:
        prefix = f"gdn.layer{layer}."
        required = {
            name: tensors[prefix + name]
            for name in (
                "q",
                "k",
                "v",
                "log_g",
                "beta",
                "initial_state",
                "chunk_output",
                "core_output",
                "final_state",
            )
            if prefix + name in tensors
        }
        missing = sorted(
            set(
                (
                    "q",
                    "k",
                    "v",
                    "log_g",
                    "beta",
                    "initial_state",
                    "chunk_output",
                    "core_output",
                    "final_state",
                )
            )
            - set(required)
        )
        if missing:
            raise ValueError(f"layer {layer} capture is missing {missing}")

        query = required["q"].squeeze(0)
        key = required["k"].squeeze(0)
        value = required["v"].squeeze(0)
        log_decay = required["log_g"].squeeze(0)
        beta = required["beta"].squeeze(0)
        initial_state = required["initial_state"].squeeze(0)
        chunk_output = required["chunk_output"].squeeze(0)
        core_output = required["core_output"]
        final_state = required["final_state"].squeeze(0)
        tokens = query.shape[0]
        expected_shapes = {
            "q": (tokens, KEY_HEADS, HEAD_DIM),
            "k": (tokens, KEY_HEADS, HEAD_DIM),
            "v": (tokens, VALUE_HEADS, HEAD_DIM),
            "log_g": (tokens, VALUE_HEADS),
            "beta": (tokens, VALUE_HEADS),
            "initial_state": (VALUE_HEADS, HEAD_DIM, HEAD_DIM),
            "chunk_output": (tokens, VALUE_HEADS, HEAD_DIM),
            "core_output": (tokens, VALUE_HEADS, HEAD_DIM),
            "final_state": (VALUE_HEADS, HEAD_DIM, HEAD_DIM),
        }
        actual_shapes = {
            "q": tuple(query.shape),
            "k": tuple(key.shape),
            "v": tuple(value.shape),
            "log_g": tuple(log_decay.shape),
            "beta": tuple(beta.shape),
            "initial_state": tuple(initial_state.shape),
            "chunk_output": tuple(chunk_output.shape),
            "core_output": tuple(core_output.shape),
            "final_state": tuple(final_state.shape),
        }
        if actual_shapes != expected_shapes:
            raise ValueError(
                f"layer {layer} capture shapes differ: "
                f"{actual_shapes} != {expected_shapes}"
            )
        if not torch.equal(core_output, chunk_output):
            raise ValueError(f"layer {layer} core and chunk outputs are not identical")

        trace = gdn_reference_forward(
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state,
        )
        output_metrics = _metrics(trace.output, chunk_output)
        state_metrics = _metrics(trace.final_state, final_state)
        passed = (
            output_metrics["finite"]
            and state_metrics["finite"]
            and output_metrics["relative_rms"] <= max_output_relative_rms
            and state_metrics["relative_rms"] <= max_state_relative_rms
        )
        all_pass &= passed
        layer_results[str(layer)] = {
            "status": "passed" if passed else "failed",
            "tokens": tokens,
            "core_equals_chunk_bitwise": True,
            "output": output_metrics,
            "final_state": state_metrics,
        }

    return {
        "schema_version": 1,
        "status": "passed" if all_pass else "failed",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "capture": {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "mode": payload.get("mode"),
            "model_revision": payload.get("model_revision"),
            "prompt": payload.get("prompt"),
            "prompt_token_ids": payload.get("prompt_token_ids"),
        },
        "contract": {
            "recurrence": "FP32 sequential GDN surrogate",
            "deployed_reference": "vLLM Triton chunk_gated_delta_rule",
            "max_output_relative_rms": max_output_relative_rms,
            "max_state_relative_rms": max_state_relative_rms,
            "bit_equality_required": False,
        },
        "layers": layer_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-output-relative-rms", type=float, default=0.01)
    parser.add_argument("--max-state-relative-rms", type=float, default=0.01)
    args = parser.parse_args()
    result = validate_capture(
        args.capture,
        max_output_relative_rms=args.max_output_relative_rms,
        max_state_relative_rms=args.max_state_relative_rms,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
