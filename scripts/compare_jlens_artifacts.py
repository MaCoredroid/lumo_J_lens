#!/usr/bin/env python3
"""Compare a verified local Qwen Jacobian Lens with the pinned public lens.

This command compares matrix geometry only. It does not run held-out prompts,
decode logits, or make a behavioral-equivalence claim.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Sequence

from download_jlens import (
    LENS_FILENAME,
    LENS_REPO,
    LENS_REVISION,
    LENS_SHA256,
    verify_checkpoint,
    verify_file,
    verify_local_fit_artifact,
)
from verify_nvfp4_ste_artifact import (
    open_held_regular_file,
    open_verified_nvfp4_ste_artifact,
    verify_nvfp4_ste_artifact,
)


SCHEMA_VERSION = 1
D_MODEL = 5120
SOURCE_LAYERS = tuple(range(63))
ROW_QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


@dataclass(frozen=True)
class LayerTotals:
    local_norm_squared: float
    public_norm_squared: float
    difference_norm_squared: float
    inner_product: float
    row_cosines: Any


def _finite_float(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"non-finite metric: {label}")
    return result


def _quantile_key(quantile: float) -> str:
    return f"p{round(quantile * 100):03d}"


def quantile_summary(values: Any) -> dict[str, Any]:
    import torch

    values = torch.as_tensor(values, dtype=torch.float64, device="cpu").flatten()
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("quantile values must be nonempty and finite")
    quantiles = torch.tensor(ROW_QUANTILES, dtype=torch.float64)
    measured = torch.quantile(values, quantiles)
    return {
        "count": values.numel(),
        "mean": _finite_float(values.mean(), label="quantile mean"),
        "standard_deviation": _finite_float(
            values.std(unbiased=False), label="quantile standard deviation"
        ),
        "quantiles": {
            _quantile_key(quantile): _finite_float(value, label="quantile")
            for quantile, value in zip(ROW_QUANTILES, measured, strict=True)
        },
    }


def scalar_identity_component(
    *, trace: float, norm_squared: float, d_model: int
) -> dict[str, float]:
    if d_model <= 0 or norm_squared <= 0:
        raise ValueError("identity component requires positive dimensions and norm")
    scalar = trace / d_model
    identity_norm_squared = trace * trace / d_model
    residual_norm_squared = max(norm_squared - identity_norm_squared, 0.0)
    return {
        "scalar": _finite_float(scalar, label="identity scalar"),
        "frobenius_norm": _finite_float(
            math.sqrt(identity_norm_squared), label="identity norm"
        ),
        "energy_fraction": _finite_float(
            identity_norm_squared / norm_squared, label="identity energy fraction"
        ),
        "residual_relative_norm": _finite_float(
            math.sqrt(residual_norm_squared / norm_squared),
            label="identity residual relative norm",
        ),
    }


def compare_layer_matrices(
    local: Any,
    public: Any,
    *,
    layer: int,
    d_model: int,
    row_chunk: int,
) -> tuple[dict[str, Any], LayerTotals]:
    """Compare one layer without materializing either complete matrix."""
    import torch

    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    expected_shape = (d_model, d_model)
    for label, matrix in (("local", local), ("public", public)):
        if not torch.is_tensor(matrix) or tuple(matrix.shape) != expected_shape:
            shape = tuple(matrix.shape) if torch.is_tensor(matrix) else type(matrix)
            raise ValueError(f"layer {layer} {label} shape mismatch: {shape}")

    local_norm_squared = 0.0
    public_norm_squared = 0.0
    difference_norm_squared = 0.0
    inner_product = 0.0
    local_trace = 0.0
    public_trace = 0.0
    row_cosine_chunks: list[Any] = []

    for start in range(0, d_model, row_chunk):
        stop = min(start + row_chunk, d_model)
        local_rows = local[start:stop].to(device="cpu", dtype=torch.float64)
        public_rows = public[start:stop].to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(local_rows).all()):
            raise FloatingPointError(f"layer {layer} local matrix is non-finite")
        if not bool(torch.isfinite(public_rows).all()):
            raise FloatingPointError(f"layer {layer} public matrix is non-finite")

        difference = local_rows - public_rows
        local_row_norm_squared = local_rows.square().sum(dim=1)
        public_row_norm_squared = public_rows.square().sum(dim=1)
        zero_rows = (local_row_norm_squared == 0) | (public_row_norm_squared == 0)
        if bool(zero_rows.any()):
            rows = (torch.nonzero(zero_rows).flatten() + start).tolist()
            raise ValueError(f"layer {layer} has zero-norm comparison rows: {rows}")
        row_inner = (local_rows * public_rows).sum(dim=1)
        row_cosines = row_inner / torch.sqrt(
            local_row_norm_squared * public_row_norm_squared
        )
        row_cosine_chunks.append(row_cosines.clamp(-1.0, 1.0))

        local_norm_squared += float(local_row_norm_squared.sum())
        public_norm_squared += float(public_row_norm_squared.sum())
        difference_norm_squared += float(difference.square().sum())
        inner_product += float(row_inner.sum())
        diagonal = torch.arange(stop - start)
        local_trace += float(local_rows[diagonal, diagonal + start].sum())
        public_trace += float(public_rows[diagonal, diagonal + start].sum())

    for label, value in (
        ("local norm", local_norm_squared),
        ("public norm", public_norm_squared),
        ("difference norm", difference_norm_squared),
        ("inner product", inner_product),
        ("local trace", local_trace),
        ("public trace", public_trace),
    ):
        _finite_float(value, label=f"layer {layer} {label}")
    if local_norm_squared <= 0 or public_norm_squared <= 0:
        raise ValueError(f"layer {layer} matrix norm must be positive")

    local_norm = math.sqrt(local_norm_squared)
    public_norm = math.sqrt(public_norm_squared)
    difference_norm = math.sqrt(difference_norm_squared)
    frobenius_cosine = max(
        -1.0, min(1.0, inner_product / (local_norm * public_norm))
    )
    row_cosines = torch.cat(row_cosine_chunks)
    record = {
        "layer": layer,
        "difference_frobenius_norm": difference_norm,
        "relative_frobenius_difference": difference_norm / public_norm,
        "relative_difference_denominator": "public_frobenius_norm",
        "frobenius_cosine": frobenius_cosine,
        "row_wise_cosine": quantile_summary(row_cosines),
        "local": {
            "frobenius_norm": local_norm,
            "trace": local_trace,
            "best_scalar_identity": scalar_identity_component(
                trace=local_trace,
                norm_squared=local_norm_squared,
                d_model=d_model,
            ),
        },
        "public": {
            "frobenius_norm": public_norm,
            "trace": public_trace,
            "best_scalar_identity": scalar_identity_component(
                trace=public_trace,
                norm_squared=public_norm_squared,
                d_model=d_model,
            ),
        },
    }
    totals = LayerTotals(
        local_norm_squared=local_norm_squared,
        public_norm_squared=public_norm_squared,
        difference_norm_squared=difference_norm_squared,
        inner_product=inner_product,
        row_cosines=row_cosines,
    )
    return record, totals


def _load_checkpoint(
    path: Path, *, d_model: int, source_layers: Sequence[int]
) -> tuple[dict[str, Any], dict[int, Any]]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("d_model") != d_model:
        raise ValueError(f"checkpoint width mismatch: {path}")
    expected_layers = list(source_layers)
    if checkpoint.get("source_layers") != expected_layers:
        raise ValueError(f"checkpoint source layers mismatch: {path}")
    jacobians = checkpoint.get("J")
    if not isinstance(jacobians, dict) or set(jacobians) != set(source_layers):
        raise ValueError(f"checkpoint Jacobian keys mismatch: {path}")
    return checkpoint, jacobians


def compare_artifact_matrices(
    local_path: Path,
    public_path: Path,
    *,
    d_model: int = D_MODEL,
    source_layers: Sequence[int] = SOURCE_LAYERS,
    row_chunk: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Memory-map two verified checkpoints and compare them layer by layer."""
    import torch

    if d_model <= 0 or not source_layers:
        raise ValueError("d_model and source_layers must be nonempty")
    source_layers = tuple(source_layers)
    local_checkpoint, local_jacobians = _load_checkpoint(
        local_path, d_model=d_model, source_layers=source_layers
    )
    public_checkpoint, public_jacobians = _load_checkpoint(
        public_path, d_model=d_model, source_layers=source_layers
    )

    records: list[dict[str, Any]] = []
    totals: list[LayerTotals] = []
    for layer in source_layers:
        record, layer_totals = compare_layer_matrices(
            local_jacobians[layer],
            public_jacobians[layer],
            layer=layer,
            d_model=d_model,
            row_chunk=row_chunk,
        )
        records.append(record)
        totals.append(layer_totals)

    local_norm_squared = sum(item.local_norm_squared for item in totals)
    public_norm_squared = sum(item.public_norm_squared for item in totals)
    difference_norm_squared = sum(item.difference_norm_squared for item in totals)
    inner_product = sum(item.inner_product for item in totals)
    aggregate = {
        "layer_count": len(records),
        "global_relative_frobenius_difference": math.sqrt(
            difference_norm_squared / public_norm_squared
        ),
        "global_frobenius_cosine": max(
            -1.0,
            min(
                1.0,
                inner_product
                / math.sqrt(local_norm_squared * public_norm_squared),
            ),
        ),
        "per_layer_relative_frobenius_difference": quantile_summary(
            [record["relative_frobenius_difference"] for record in records]
        ),
        "per_layer_frobenius_cosine": quantile_summary(
            [record["frobenius_cosine"] for record in records]
        ),
        "all_rows_cosine": quantile_summary(
            torch.cat([item.row_cosines for item in totals])
        ),
        "local_total_frobenius_norm": math.sqrt(local_norm_squared),
        "public_total_frobenius_norm": math.sqrt(public_norm_squared),
        "difference_total_frobenius_norm": math.sqrt(difference_norm_squared),
    }
    # Keep the memory-mapped checkpoint objects alive until every chunk is read.
    del local_checkpoint, public_checkpoint
    return records, aggregate


def verify_artifacts(
    *,
    local_path: Path,
    local_sha256: str,
    local_provenance: Path,
    local_state: Path | None = None,
    local_state_sha256: str | None = None,
    public_path: Path,
    local_kind: str = "nf4",
    local_verifier: Callable[..., dict[str, object]] | None = None,
    nf4_verifier: Callable[..., dict[str, object]] = verify_local_fit_artifact,
    native_verifier: Callable[..., dict[str, object]] = verify_nvfp4_ste_artifact,
    public_file_verifier: Callable[..., dict[str, object]] = verify_file,
    public_checkpoint_verifier: Callable[..., dict[str, object]] = verify_checkpoint,
) -> dict[str, dict[str, object]]:
    if local_kind not in {"nf4", "nvfp4-ste"}:
        raise ValueError(f"unsupported local lens kind: {local_kind}")
    if local_verifier is None:
        local_verifier = (
            nf4_verifier if local_kind == "nf4" else native_verifier
        )
    local_kwargs: dict[str, object] = {
        "expected_sha256": local_sha256,
        "provenance_path": local_provenance,
        "check_finite": True,
    }
    if local_kind == "nvfp4-ste":
        if local_state is None or local_state_sha256 is None:
            raise ValueError("native comparison requires local fit state and SHA-256")
        local_kwargs.update(
            state_path=local_state,
            expected_state_sha256=local_state_sha256,
        )
    elif local_state is not None or local_state_sha256 is not None:
        raise ValueError("NF4 comparison does not accept native fit-state metadata")
    local = local_verifier(local_path, **local_kwargs)
    public = {
        "kind": "pinned_public",
        "repo_id": LENS_REPO,
        "revision": LENS_REVISION,
        "filename": LENS_FILENAME,
        **public_file_verifier(public_path),
        **public_checkpoint_verifier(public_path, check_finite=False),
    }
    return {"local": local, "public": public}


def atomic_write_json(path: Path, value: Any) -> None:
    rendered = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-path", type=Path, required=True)
    parser.add_argument("--local-sha256", required=True)
    parser.add_argument("--local-provenance", type=Path, required=True)
    parser.add_argument("--local-state", type=Path)
    parser.add_argument("--local-state-sha256")
    parser.add_argument(
        "--local-kind", choices=("nf4", "nvfp4-ste"), default="nf4"
    )
    parser.add_argument("--public-path", type=Path)
    parser.add_argument("--row-chunk", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.row_chunk <= D_MODEL:
        raise ValueError(f"--row-chunk must be in 1..{D_MODEL}")

    if args.public_path is None:
        from huggingface_hub import hf_hub_download

        public_path = Path(
            hf_hub_download(
                repo_id=LENS_REPO,
                filename=LENS_FILENAME,
                revision=LENS_REVISION,
                local_files_only=True,
            )
        ).resolve(strict=True)
    else:
        public_path = args.public_path
    with ExitStack() as resources:
        held_public = resources.enter_context(
            open_held_regular_file(
                public_path,
                label="public lens checkpoint",
                expected_sha256=LENS_SHA256,
            )
        )
        public_comparison_path = Path(held_public.fd_path)
        local_path = args.local_path
        local_provenance = args.local_provenance
        local_verifier: Callable[..., dict[str, object]] | None = None
        comparison_path = local_path.resolve()
        if args.local_kind == "nvfp4-ste":
            if args.local_state is None or args.local_state_sha256 is None:
                raise ValueError(
                    "native comparison requires --local-state and --local-state-sha256"
                )
            native = resources.enter_context(
                open_verified_nvfp4_ste_artifact(
                    local_path,
                    expected_sha256=args.local_sha256,
                    provenance_path=local_provenance,
                    state_path=args.local_state,
                    expected_state_sha256=args.local_state_sha256,
                    check_finite=True,
                )
            )
            comparison_path = Path(native.fd_path)
            local_verifier = lambda *_args, **_kwargs: dict(native.record)

        artifacts = verify_artifacts(
            local_path=local_path,
            local_sha256=args.local_sha256,
            local_provenance=local_provenance,
            local_state=args.local_state,
            local_state_sha256=args.local_state_sha256,
            public_path=public_comparison_path,
            local_kind=args.local_kind,
            local_verifier=local_verifier,
        )
        layers, aggregate = compare_artifact_matrices(
            comparison_path,
            public_comparison_path,
            row_chunk=args.row_chunk,
        )
        if args.local_kind == "nvfp4-ste":
            native.require_unchanged()
        held_public.require_unchanged()
        artifacts["public"]["path"] = str(held_public.path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "numeric matrix comparison only; no held-out token or logit evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "configuration": {
            "d_model": D_MODEL,
            "source_layers": list(SOURCE_LAYERS),
            "row_chunk": args.row_chunk,
            "relative_difference_reference": "public lens",
            "row_cosine_quantiles": list(ROW_QUANTILES),
        },
        "aggregate": aggregate,
        "layers": layers,
    }
    atomic_write_json(args.output, report)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
