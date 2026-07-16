#!/usr/bin/env python3
"""Download and verify the pinned Qwen3.6-27B Jacobian lens."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
LENS_FILENAME = (
    "qwen3.6-27b/jlens/Salesforce-wikitext/"
    "Qwen3.6-27B_jacobian_lens_n1000.pt"
)
LENS_SHA256 = "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"
LENS_SIZE = 3_303_032_772


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path) -> dict[str, object]:
    size = path.stat().st_size
    if size != LENS_SIZE:
        raise ValueError(f"lens size mismatch: expected {LENS_SIZE}, got {size}")
    digest = sha256_file(path)
    if digest != LENS_SHA256:
        raise ValueError(f"lens SHA-256 mismatch: expected {LENS_SHA256}, got {digest}")
    return {"path": str(path.resolve()), "size_bytes": size, "sha256": digest}


def verify_checkpoint(path: Path, *, check_finite: bool) -> dict[str, object]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if set(checkpoint) != {"J", "d_model", "n_prompts", "source_layers"}:
        raise ValueError(f"unexpected checkpoint keys: {sorted(checkpoint)}")
    expected_layers = list(range(63))
    if checkpoint["d_model"] != 5120:
        raise ValueError(f"unexpected d_model: {checkpoint['d_model']}")
    if checkpoint["n_prompts"] != 1000:
        raise ValueError(f"unexpected n_prompts: {checkpoint['n_prompts']}")
    if checkpoint["source_layers"] != expected_layers:
        raise ValueError("source_layers are not exactly 0..62")
    if sorted(checkpoint["J"]) != expected_layers:
        raise ValueError("Jacobian keys are not exactly 0..62")

    for layer in expected_layers:
        jacobian = checkpoint["J"][layer]
        if jacobian.shape != (5120, 5120):
            raise ValueError(f"layer {layer} has shape {tuple(jacobian.shape)}")
        if jacobian.dtype != torch.float16:
            raise ValueError(f"layer {layer} has dtype {jacobian.dtype}")
        if check_finite and not bool(torch.isfinite(jacobian).all()):
            raise ValueError(f"layer {layer} contains non-finite values")

    return {
        "checkpoint_keys": sorted(checkpoint),
        "d_model": checkpoint["d_model"],
        "n_prompts": checkpoint["n_prompts"],
        "source_layers": expected_layers,
        "tensor_dtype": "torch.float16",
        "tensor_shape": [5120, 5120],
        "finite_checked": check_finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-finite-check",
        action="store_true",
        help="skip the full 3.3 GB tensor finiteness scan",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=LENS_REPO,
            filename=LENS_FILENAME,
            revision=LENS_REVISION,
        )
    )
    result = {
        "repo_id": LENS_REPO,
        "revision": LENS_REVISION,
        "filename": LENS_FILENAME,
        **verify_file(path),
        **verify_checkpoint(path, check_finite=not args.no_finite_check),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
