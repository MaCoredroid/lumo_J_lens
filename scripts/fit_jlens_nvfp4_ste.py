#!/usr/bin/env python3
"""Fit a Jacobian Lens from exact NVFP4 forward captures and surrogate VJPs.

The production runner is intentionally split into two phases.  vLLM first
captures the deployed ModelOpt forward.  This module then walks the captured
decoder blocks in reverse, one block at a time.  Quantized linears provide
their declared frozen-weight/STE VJP while every forward value is replaced by
the corresponding runtime capture.

This is an NVFP4/FP8-STE Jacobian Lens.  It is not the derivative of discrete
FP4 or FP8 rounding.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Protocol

import numpy as np
import torch


SCHEMA_VERSION = 1
NUM_BLOCKS = 64
D_MODEL = 5120
SOURCE_LAYERS = tuple(range(63))
TARGET_LAYER = 63
F32_DTYPE = np.dtype("<f4")


class BlockReplay(Protocol):
    """Materialize one exact-forward surrogate decoder block."""

    def __call__(self, layer: int, logical_input: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class ReverseRows:
    """One output-row chunk for every fitted source layer."""

    row_start: int
    row_stop: int
    rows: Mapping[int, torch.Tensor]
    propagated_cotangent: torch.Tensor


def valid_estimator_positions(
    sequence: int,
    skip_first: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Match the Anthropic estimator: skip sinks and the final token."""

    if skip_first < 0 or sequence <= skip_first + 1:
        raise ValueError(
            f"prompt too short: sequence={sequence}, need > {skip_first + 1}"
        )
    return torch.arange(skip_first, sequence - 1, device=device, dtype=torch.long)


def target_cotangent_rows(
    *,
    tokens: int,
    hidden: int,
    valid_positions: torch.Tensor,
    row_start: int,
    row_stop: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Construct future-summed basis cotangents with shape ``[R,T,D]``."""

    if not 0 <= row_start < row_stop <= hidden:
        raise ValueError("invalid output row interval")
    if valid_positions.ndim != 1 or valid_positions.numel() == 0:
        raise ValueError("valid_positions must be a non-empty vector")
    positions = valid_positions.to(device=device, dtype=torch.long)
    if int(positions.min()) < 0 or int(positions.max()) >= tokens:
        raise ValueError("valid position is outside the target sequence")

    count = row_stop - row_start
    cotangents = torch.zeros(
        count,
        tokens,
        hidden,
        dtype=dtype,
        device=device,
    )
    rows = torch.arange(count, device=device)
    dimensions = row_start + rows
    cotangents[rows[:, None], positions[None, :], dimensions[:, None]] = 1
    return cotangents


def block_input_vjp(
    output: torch.Tensor,
    logical_input: torch.Tensor,
    cotangents: torch.Tensor,
) -> torch.Tensor:
    """Propagate a batch of cotangents through one replayed block."""

    if output.ndim != 2 or output.shape != logical_input.shape:
        raise ValueError("block input/output must have matching [tokens, hidden] shapes")
    if not logical_input.requires_grad:
        raise ValueError("logical_input must require gradients")
    if cotangents.shape[1:] != output.shape:
        raise ValueError("cotangent tail does not match block output")
    (gradient,) = torch.autograd.grad(
        output,
        logical_input,
        grad_outputs=cotangents,
        is_grads_batched=True,
    )
    if gradient.shape != cotangents.shape:
        raise RuntimeError(
            f"block VJP shape {tuple(gradient.shape)} != {tuple(cotangents.shape)}"
        )
    if not bool(torch.isfinite(gradient.float()).all()):
        raise FloatingPointError("block VJP contains non-finite values")
    return gradient


def reverse_replay_rows(
    hidden_states: Mapping[int, torch.Tensor],
    replay_block: BlockReplay,
    valid_positions: torch.Tensor,
    row_start: int,
    row_stop: int,
    *,
    first_block: int = 1,
    target_layer: int = TARGET_LAYER,
    device: torch.device | str | None = None,
) -> ReverseRows:
    """Compute one Jacobian row chunk by replaying blocks in reverse.

    ``hidden_states[i]`` is the logical output of decoder block ``i``.  Source
    layer 62 is therefore the input to block 63, and source layer 0 is the
    input to block 1.  Each replay graph is freed before the previous block is
    materialized, so memory does not grow with suffix depth.
    """

    if not 1 <= first_block <= target_layer:
        raise ValueError("invalid first_block/target_layer interval")
    required = set(range(first_block - 1, target_layer + 1))
    missing = sorted(required - set(hidden_states))
    if missing:
        raise ValueError(f"missing captured hidden states: {missing}")

    target = hidden_states[target_layer]
    if target.ndim != 2:
        raise ValueError("captured hidden states must have shape [tokens, hidden]")
    target_device = target.device if device is None else torch.device(device)
    tokens, hidden = target.shape
    cotangent = target_cotangent_rows(
        tokens=tokens,
        hidden=hidden,
        valid_positions=valid_positions,
        row_start=row_start,
        row_stop=row_stop,
        dtype=target.dtype,
        device=target_device,
    )
    positions = valid_positions.to(device=target_device, dtype=torch.long)
    rows_by_source: dict[int, torch.Tensor] = {}

    for block in range(target_layer, first_block - 1, -1):
        logical_input = (
            hidden_states[block - 1]
            .to(device=target_device)
            .detach()
            .requires_grad_(True)
        )
        output = replay_block(block, logical_input)
        expected = hidden_states[block].to(device=target_device)
        if output.shape != expected.shape:
            raise ValueError(
                f"replayed block {block} shape {tuple(output.shape)} != "
                f"capture {tuple(expected.shape)}"
            )
        if not torch.equal(output.detach(), expected):
            difference = output.detach().float() - expected.float()
            raise RuntimeError(
                f"replayed block {block} is not exact: "
                f"max_abs={float(difference.abs().max())}"
            )
        cotangent = block_input_vjp(output, logical_input, cotangent).detach()
        rows_by_source[block - 1] = cotangent[:, positions].float().mean(dim=1)

    return ReverseRows(
        row_start=row_start,
        row_stop=row_stop,
        rows=rows_by_source,
        propagated_cotangent=cotangent,
    )


def iter_reverse_replay_rows(
    hidden_states: Mapping[int, torch.Tensor],
    replay_block: BlockReplay,
    valid_positions: torch.Tensor,
    *,
    row_start: int = 0,
    row_stop: int | None = None,
    cotangent_batch: int = 32,
    first_block: int = 1,
    target_layer: int = TARGET_LAYER,
    device: torch.device | str | None = None,
):
    """Yield memory-bounded row chunks for all requested source layers."""

    if cotangent_batch <= 0:
        raise ValueError("cotangent_batch must be positive")
    hidden = hidden_states[target_layer].shape[-1]
    stop = hidden if row_stop is None else row_stop
    if not 0 <= row_start <= stop <= hidden:
        raise ValueError("invalid row_start/row_stop")
    for start in range(row_start, stop, cotangent_batch):
        yield reverse_replay_rows(
            hidden_states,
            replay_block,
            valid_positions,
            start,
            min(start + cotangent_batch, stop),
            first_block=first_block,
            target_layer=target_layer,
            device=device,
        )


def matrix_path(directory: Path, layer: int) -> Path:
    return directory / f"layer-{layer:02d}.f32"


def prepare_matrix_files(
    directory: Path,
    *,
    source_layers: Sequence[int] = SOURCE_LAYERS,
    hidden: int = D_MODEL,
) -> dict[int, np.memmap]:
    """Create or reopen dense little-endian FP32 matrix files."""

    directory.mkdir(parents=True, exist_ok=True)
    expected_bytes = hidden * hidden * F32_DTYPE.itemsize
    matrices: dict[int, np.memmap] = {}
    for layer in source_layers:
        path = matrix_path(directory, layer)
        if path.exists() and path.stat().st_size != expected_bytes:
            raise RuntimeError(f"invalid matrix file size: {path}")
        mode = "r+" if path.exists() else "w+"
        matrices[layer] = np.memmap(
            path,
            mode=mode,
            dtype=F32_DTYPE,
            shape=(hidden, hidden),
        )
    return matrices


def write_reverse_rows(
    matrices: Mapping[int, np.memmap],
    chunk: ReverseRows,
) -> None:
    """Persist one row chunk after validating layer and shape coverage."""

    expected = (chunk.row_stop - chunk.row_start, next(iter(matrices.values())).shape[1])
    if set(chunk.rows) != set(matrices):
        raise ValueError("row chunk source layers do not match matrix files")
    for layer, matrix in matrices.items():
        rows = chunk.rows[layer]
        if tuple(rows.shape) != expected:
            raise ValueError(
                f"layer {layer} row shape {tuple(rows.shape)} != {expected}"
            )
        matrix[chunk.row_start : chunk.row_stop] = (
            rows.detach().cpu().numpy().astype(F32_DTYPE, copy=False)
        )
        matrix.flush()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--describe-contract",
        action="store_true",
        help="print the machine-readable estimator contract",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.describe_contract:
        raise SystemExit(
            "production capture integration is not yet selected; use "
            "--describe-contract for the implemented reverse estimator contract"
        )
    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_layers": list(SOURCE_LAYERS),
        "target_layer": TARGET_LAYER,
        "estimator": "future-summed VJP, mean over matching source positions",
        "forward": "exact deployed NVFP4/FP8 captures",
        "backward": "frozen dequantized NVFP4 weights and explicit FP8 STE",
        "literal_rounding_derivative": False,
        "blockwise_reverse_replay": True,
    }
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
