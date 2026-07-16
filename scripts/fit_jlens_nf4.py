#!/usr/bin/env python3
"""Fit a new Qwen3.6-27B Jacobian Lens through a differentiable NF4 model.

This is deliberately separate from the NVFP4 serving/readout path.  It loads
the pinned official BF16 checkpoint as the text-only Hugging Face causal LM,
quantizes every decoder linear to bitsandbytes NF4, forces Qwen's ordinary
PyTorch Gated DeltaNet implementation, and fits the Anthropic future-summed
estimator.  The quantized forward and its activation VJPs are therefore from
NF4/BF16 execution; no pre-fitted lens is involved.

The default invocation is fail-closed: model files must already be in the
Hugging Face cache unless ``--allow-download`` is passed, prompts must encode
to exactly 128 tokens, existing state must be resumed explicitly, and every
official weight shard is content-hash verified before model construction.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
import contextlib
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import stat
import subprocess
import sys
import time
from typing import Any
import uuid

import numpy as np
import torch
from torch import nn


MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
JLENS_REFERENCE_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
JLENS_REFERENCE_URL = "https://github.com/anthropics/jacobian-lens.git"
PROMPT_MANIFEST_SHA256 = "2c36f17dee7287c096f7d1fdb7f8d7ecb8372c6cf8d13d7af8cfaed820439d3b"
MODEL_CONFIG_SHA256 = "69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c"
MODEL_INDEX_SHA256 = "a8ad2c26fb707ff8c245806315b03e3b4b74595528492423af5dae0ce39b4d9b"

# Hub LFS SHA-256 and exact file size at MODEL_REVISION.  These are checked
# before any untrusted tensor bytes are loaded.
MODEL_SHARDS: dict[str, tuple[int, str]] = {
    "model-00001-of-00015.safetensors": (
        3_968_861_352,
        "5f21d4e349aef6c74bedef7b3835dc8c11a16dd5ce72f4437e2284f1e83736e9",
    ),
    "model-00002-of-00015.safetensors": (
        3_921_677_136,
        "03de44dc7e933025498d72f8d7ea32d5cde16eb7a85b35ff1fa49a737f4b2242",
    ),
    "model-00003-of-00015.safetensors": (
        3_921_677_128,
        "5c3a68304dabeaa5a0eb70a5e383d2b8134997ef6d5c1e8afdcd236091c847f2",
    ),
    "model-00004-of-00015.safetensors": (
        3_921_677_128,
        "ba8b0849cb4c4c97e674709bfe56a9c008d7563a68ac57120ceb2855a73a9944",
    ),
    "model-00005-of-00015.safetensors": (
        3_921_677_112,
        "a5abc1d5e9583409193e2cd58a671a4ed98467a6eaf883cd4377e3cc02021c22",
    ),
    "model-00006-of-00015.safetensors": (
        3_900_710_888,
        "160d914e2e4704a401a3eab9e9eec3380e5723a35981c9169bedfb82ea32e6d6",
    ),
    "model-00007-of-00015.safetensors": (
        3_994_391_976,
        "0bcd0ce28c7d2cc6f5ab2c21902cd032fec0cc5f29e0897e3bb0967add388e47",
    ),
    "model-00008-of-00015.safetensors": (
        3_879_219_776,
        "584a0ed8018d3b19ce8e533d08a460454650027901be6323b50b591da143dca6",
    ),
    "model-00009-of-00015.safetensors": (
        3_921_677_136,
        "e7e3e1a17a2673340eb6bba95c3c6f07b12e7746ee738dde56551cbde27656f5",
    ),
    "model-00010-of-00015.safetensors": (
        3_921_677_128,
        "e8934789f4742c11da88f937da84bdbb8764c8a7f7ebe9e7b056bcbd41b14285",
    ),
    "model-00011-of-00015.safetensors": (
        3_921_677_136,
        "44e8fe06d2d609bf20b16b1d2f42348ca2dc99179721f76035ff50db678f9a8a",
    ),
    "model-00012-of-00015.safetensors": (
        3_921_677_136,
        "33c5d7d18e1b3f661334dff736fc7d00561f1f1bb9d2970c669b40d2e582974e",
    ),
    "model-00013-of-00015.safetensors": (
        3_995_081_848,
        "68db2ebb03231238c2a114f3642df967a970a1d96b029adc427838cb5cf7f27a",
    ),
    "model-00014-of-00015.safetensors": (
        3_942_652_952,
        "26c114fb6d5d4131ab227552d489d7a97c8c958a9ed94f6d3d1584f6cfe4b9fb",
    ),
    "model-00015-of-00015.safetensors": (
        508_670_568,
        "b84b5b1315e865c9a19a444045d422a73e3e2e31ce3766797cffd3507c68c9c9",
    ),
}

TOKENIZER_FILES: dict[str, str] = {
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
    "chat_template.jinja": "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}

NUM_LAYERS = 64
D_MODEL = 5120
SOURCE_LAYERS = tuple(range(63))
TARGET_LAYER = 63
EXPECTED_LINEAR_LAYERS = tuple(i for i in range(NUM_LAYERS) if i % 4 != 3)
EXPECTED_FULL_LAYERS = tuple(i for i in range(NUM_LAYERS) if i % 4 == 3)
EXPECTED_NF4_LINEARS = 496
STATE_SCHEMA = 2
PROMPT_MANIFEST_SCHEMA = 1
F32_DTYPE = np.dtype("<f4")
NF4_BLOCKSIZE = 64
NF4_NESTED_BLOCKSIZE = 256
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
SOURCE_TREE_FILES = (
    "configs/jlens_nf4_fit_prompts.json",
    "docs/NF4_FIT_CONTRACT.md",
    "requirements-fit.txt",
    "scripts/check_fit.sh",
    "scripts/compare_jlens_artifacts.py",
    "scripts/download_jlens.py",
    "scripts/fit_jlens_nf4.py",
    "scripts/materialize_jlens_fit_prompts.py",
    "scripts/setup_fit.sh",
    "validation/fit-freeze.txt",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular_file(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} does not exist: {path}") from None
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")


def tensor_content_sha256(
    tensor: torch.Tensor, *, chunk_bytes: int = 16 * 1024 * 1024
) -> str:
    """Hash tensor storage in logical contiguous order with bounded host memory."""

    value = tensor.detach().contiguous().reshape(-1)
    elements = max(1, chunk_bytes // max(1, value.element_size()))
    digest = hashlib.sha256()
    for start in range(0, value.numel(), elements):
        cpu = value[start : start + elements].to(device="cpu", non_blocking=False)
        digest.update(cpu.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_hash_record(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "nbytes": tensor.numel() * tensor.element_size(),
        "sha256": tensor_content_sha256(tensor),
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def source_tree_identity(
    root: Path, *, require_clean: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    commit = _run_git(root, "rev-parse", "HEAD")
    status_porcelain = _run_git(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if require_clean and status_porcelain:
        raise RuntimeError(
            "a new fit must start from a clean Git worktree; commit all source files first"
        )
    files: list[dict[str, Any]] = []
    for relative in SOURCE_TREE_FILES:
        path = root / relative
        require_regular_file(path, label="source-tree input")
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    identity = {
        "git_commit": commit,
        "files": files,
        "manifest_sha256": canonical_sha256(files),
    }
    observation = {
        "git_commit": commit,
        "git_status_porcelain": status_porcelain,
        "git_clean": not bool(status_porcelain),
        "observed_at": utc_now(),
        "manifest_sha256": identity["manifest_sha256"],
    }
    return identity, observation


def layer_path(directory: Path, layer: int) -> Path:
    return directory / f"layer-{layer:02d}.f32"


def validate_dense_file(path: Path, d_model: int | None = None) -> None:
    if d_model is None:
        d_model = D_MODEL
    expected = d_model * d_model * F32_DTYPE.itemsize
    if not path.is_file() or path.stat().st_size != expected:
        actual = path.stat().st_size if path.exists() else None
        raise RuntimeError(f"invalid dense matrix file {path}: {actual} != {expected}")


@dataclass(frozen=True)
class FrozenPrompt:
    prompt_id: str
    text_sha256: str
    token_ids: tuple[int, ...]
    row_index: int | None = None

    def record(self) -> dict[str, Any]:
        record = {
            "id": self.prompt_id,
            "text_sha256": self.text_sha256,
            "token_count": len(self.token_ids),
            "token_ids": list(self.token_ids),
        }
        if self.row_index is not None:
            record["row_index"] = self.row_index
        return record


class ActivationCapture:
    """Capture block outputs while rooting the graph at one block output."""

    def __init__(
        self, blocks: Sequence[nn.Module], indices: Sequence[int], graph_root: int
    ) -> None:
        self.blocks = blocks
        self.indices = tuple(sorted(set(indices)))
        self.graph_root = graph_root
        self.activations: dict[int, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, index: int):
        def capture(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            if not torch.is_tensor(tensor):
                raise TypeError(f"block {index} did not return a tensor first")
            if index == self.graph_root:
                if tensor.requires_grad:
                    raise RuntimeError(
                        "graph-root activation already requires grad; model parameters "
                        "must be frozen before fitting"
                    )
                tensor.requires_grad_(True)
            self.activations[index] = tensor

        return capture

    def __enter__(self) -> "ActivationCapture":
        try:
            for index in self.indices:
                self.handles.append(
                    self.blocks[index].register_forward_hook(self._hook(index))
                )
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_exc: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def future_summed_vjp_rows(
    target: torch.Tensor,
    sources: Sequence[torch.Tensor],
    valid_positions: torch.Tensor,
    row_start: int,
    row_stop: int,
    *,
    retain_graph: bool,
) -> list[torch.Tensor]:
    """Return an exact chunk of Anthropic-estimator rows for every source.

    A single prompt forward has batch size one.  ``is_grads_batched=True``
    adds an independent cotangent axis without replicating the saved forward
    graph.  This is mathematically identical to the reference implementation's
    independent prompt replicas because decoder samples do not interact.
    """

    if target.ndim != 3 or target.shape[0] != 1:
        raise ValueError("target must have shape [1, sequence, d_model]")
    sequence, d_model = target.shape[1:]
    if not 0 <= row_start < row_stop <= d_model:
        raise ValueError("invalid output row interval")
    if valid_positions.ndim != 1 or valid_positions.numel() == 0:
        raise ValueError("valid_positions must be a non-empty vector")
    valid_positions = valid_positions.to(device=target.device, dtype=torch.long)
    if int(valid_positions.min()) < 0 or int(valid_positions.max()) >= sequence:
        raise ValueError("valid position outside target sequence")
    for source in sources:
        if source.shape != target.shape or not source.requires_grad:
            raise ValueError("every source must match target and require gradients")

    n_rows = row_stop - row_start
    cotangents = torch.zeros(
        (n_rows, *target.shape), device=target.device, dtype=target.dtype
    )
    rows = torch.arange(n_rows, device=target.device)
    output_dimensions = row_start + rows
    cotangents[
        rows[:, None],
        0,
        valid_positions[None, :],
        output_dimensions[:, None],
    ] = 1

    gradients = torch.autograd.grad(
        outputs=target,
        inputs=tuple(sources),
        grad_outputs=cotangents,
        retain_graph=retain_graph,
        is_grads_batched=True,
    )
    reduced: list[torch.Tensor] = []
    for gradient in gradients:
        expected = (n_rows, 1, sequence, d_model)
        if gradient.shape != expected:
            raise RuntimeError(f"unexpected batched VJP shape {gradient.shape} != {expected}")
        result = gradient[:, 0, valid_positions, :].float().mean(dim=1)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("non-finite Jacobian row from batched VJP")
        reduced.append(result)
    return reduced


def iter_future_summed_row_chunks(
    target: torch.Tensor,
    sources: Sequence[torch.Tensor],
    valid_positions: torch.Tensor,
    *,
    row_start: int = 0,
    row_stop: int | None = None,
    cotangent_batch: int = 4,
    retain_graph_after: bool = False,
) -> Iterator[tuple[int, int, list[torch.Tensor]]]:
    if cotangent_batch <= 0:
        raise ValueError("cotangent_batch must be positive")
    d_model = target.shape[-1]
    effective_stop = d_model if row_stop is None else row_stop
    if not 0 <= row_start <= effective_stop <= d_model:
        raise ValueError("invalid row_start/row_stop")
    for start in range(row_start, effective_stop, cotangent_batch):
        stop = min(start + cotangent_batch, effective_stop)
        yield start, stop, future_summed_vjp_rows(
            target,
            sources,
            valid_positions,
            start,
            stop,
            retain_graph=retain_graph_after or stop < effective_stop,
        )


def sequential_future_summed_rows(
    target: torch.Tensor,
    sources: Sequence[torch.Tensor],
    valid_positions: torch.Tensor,
    row_start: int,
    row_stop: int,
) -> list[torch.Tensor]:
    matrices = [
        torch.empty(
            (row_stop - row_start, target.shape[-1]),
            device=target.device,
            dtype=torch.float32,
        )
        for _source in sources
    ]
    for offset, dimension in enumerate(range(row_start, row_stop)):
        cotangent = torch.zeros_like(target)
        cotangent[0, valid_positions, dimension] = 1
        gradients = torch.autograd.grad(
            target,
            tuple(sources),
            grad_outputs=cotangent,
            retain_graph=dimension + 1 < row_stop,
        )
        for matrix, gradient in zip(matrices, gradients, strict=True):
            matrix[offset] = gradient[0, valid_positions].float().mean(dim=0)
    return matrices


def vectorization_error_metrics(
    batched: torch.Tensor, sequential: torch.Tensor
) -> dict[str, float]:
    difference = batched.float() - sequential.float()
    rms = torch.sqrt(torch.mean(torch.square(difference)))
    reference_rms = torch.sqrt(torch.mean(torch.square(sequential.float())))
    return {
        "max_abs": float(torch.max(torch.abs(difference))),
        "rms": float(rms),
        "relative_rms": float(rms / torch.clamp(reference_rms, min=1e-12)),
    }


def valid_estimator_positions(
    seq_len: int, skip_first: int, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Match upstream ``valid_position_mask``: skip sinks and final token."""

    if skip_first < 0 or seq_len <= skip_first + 1:
        raise ValueError(
            f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens"
        )
    return torch.arange(skip_first, seq_len - 1, device=device, dtype=torch.long)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def tested_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "bitsandbytes": _package_version("bitsandbytes"),
        "accelerate": _package_version("accelerate"),
        "jlens": _package_version("jlens"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "numpy": np.__version__,
    }


def require_runtime() -> dict[str, Any]:
    versions = tested_versions()
    required = {
        "accelerate": "1.14.0",
        "transformers": "5.12.1",
        "bitsandbytes": "0.49.2",
        "jlens": "0.1.0",
    }
    mismatches = {
        name: (versions[name], expected)
        for name, expected in required.items()
        if versions[name] != expected
    }
    if not versions["torch"].startswith("2.11.0"):
        mismatches["torch"] = (versions["torch"], "2.11.0+CUDA")
    if mismatches:
        raise RuntimeError(f"untested package versions: {mismatches}")
    distribution = importlib.metadata.distribution("jlens")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("jlens must be installed from the pinned Git VCS URL")
    direct_url = json.loads(direct_url_text)
    vcs_info = direct_url.get("vcs_info", {})
    if (
        direct_url.get("url") != JLENS_REFERENCE_URL
        or vcs_info.get("commit_id") != JLENS_REFERENCE_COMMIT
    ):
        raise RuntimeError(f"unexpected jlens direct_url provenance: {direct_url}")
    versions["jlens_direct_url"] = direct_url
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 30 * 2**30:
        raise RuntimeError("at least 30 GiB CUDA memory is required")
    return versions


def require_model_memory() -> None:
    free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < 29 * 2**30:
        raise RuntimeError(
            f"at least 29 GiB CUDA memory must be free before loading; got {free_bytes / 2**30:.2f} GiB"
        )


def configure_determinism(seed: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be fixed before CUDA initialization"
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def runtime_identity(versions: dict[str, Any], seed: int) -> dict[str, Any]:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

    gpu = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if driver.returncode or not driver.stdout.strip():
        raise RuntimeError(f"failed to identify NVIDIA driver: {driver.stderr.strip()}")
    qwen_source = Path(q35.__file__).resolve()
    require_regular_file(qwen_source, label="Qwen modeling source")
    return {
        "versions": versions,
        "cuda_runtime": torch.version.cuda,
        "nvidia_driver": driver.stdout.strip(),
        "gpu": {
            "name": gpu.name,
            "compute_capability": [gpu.major, gpu.minor],
            "total_memory_bytes": gpu.total_memory,
        },
        "qwen_modeling_source": {
            "path": str(qwen_source),
            "size": qwen_source.stat().st_size,
            "sha256": sha256_file(qwen_source),
        },
        "determinism": {
            "seed": seed,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }


def resolve_and_verify_snapshot(
    cache_dir: Path | None, allow_download: bool
) -> tuple[Path, list[dict[str, Any]]]:
    from huggingface_hub import snapshot_download

    required_files = [
        "config.json",
        "model.safetensors.index.json",
        *MODEL_SHARDS,
        *TOKENIZER_FILES,
    ]
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(cache_dir) if cache_dir else None,
            allow_patterns=required_files,
            local_files_only=not allow_download,
        )
    ).resolve()
    for filename in required_files:
        if not (snapshot / filename).is_file():
            raise FileNotFoundError(f"pinned snapshot is missing {filename}")

    config_hash = sha256_file(snapshot / "config.json")
    index_hash = sha256_file(snapshot / "model.safetensors.index.json")
    if config_hash != MODEL_CONFIG_SHA256 or index_hash != MODEL_INDEX_SHA256:
        raise RuntimeError(
            f"pinned metadata hash mismatch: config={config_hash}, index={index_hash}"
        )
    index = read_json(snapshot / "model.safetensors.index.json")
    indexed_shards = set(index.get("weight_map", {}).values())
    if indexed_shards != set(MODEL_SHARDS):
        raise RuntimeError("model index shard set does not match the pinned manifest")

    records: list[dict[str, Any]] = [
        {"filename": "config.json", "size": (snapshot / "config.json").stat().st_size, "sha256": config_hash},
        {
            "filename": "model.safetensors.index.json",
            "size": (snapshot / "model.safetensors.index.json").stat().st_size,
            "sha256": index_hash,
        },
    ]
    for filename, expected_hash in TOKENIZER_FILES.items():
        path = snapshot / filename
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"tokenizer hash mismatch for {filename}")
        records.append(
            {"filename": filename, "size": path.stat().st_size, "sha256": actual_hash}
        )
    for filename, (expected_size, expected_hash) in MODEL_SHARDS.items():
        path = snapshot / filename
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"weight size mismatch for {filename}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"weight hash mismatch for {filename}")
        records.append(
            {"filename": filename, "size": expected_size, "sha256": actual_hash}
        )
    return snapshot, records


def load_tokenizer(snapshot: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.truncation_side != "right":
        raise RuntimeError("the pinned tokenizer must use right truncation")
    return tokenizer


def freeze_prompts(
    manifest_path: Path,
    tokenizer: Any,
    *,
    max_seq_len: int,
    require_exact_length: bool,
) -> tuple[list[FrozenPrompt], dict[str, Any]]:
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != PROMPT_MANIFEST_SHA256:
        raise ValueError(
            f"prompt manifest SHA-256 {manifest_hash} does not match pinned {PROMPT_MANIFEST_SHA256}"
        )
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != PROMPT_MANIFEST_SCHEMA:
        raise ValueError(f"prompt manifest schema_version must be {PROMPT_MANIFEST_SCHEMA}")
    entries = manifest.get("prompts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("prompt manifest must contain a non-empty prompts list")
    expected_dataset = {
        "repo": "Salesforce/wikitext",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "config": "wikitext-103-raw-v1",
        "split": "train",
    }
    expected_tokenizer = {
        "repo": MODEL_ID,
        "revision": MODEL_REVISION,
        "add_special_tokens": True,
        "force_bos_when_supported": True,
        "truncation": "right",
    }
    if manifest.get("dataset") != expected_dataset or manifest.get("tokenizer") != expected_tokenizer:
        raise ValueError("prompt manifest dataset/tokenizer provenance is not the pinned contract")

    frozen: list[FrozenPrompt] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            raise ValueError(f"prompt {index} must be an object with text")
        text = entry["text"]
        row_index = entry.get("row_index")
        if not isinstance(row_index, int) or row_index < 0:
            raise ValueError(f"prompt {index} has an invalid dataset row_index")
        prompt_id = str(entry.get("id", f"row-{row_index}"))
        if not text or prompt_id in seen_ids:
            raise ValueError(f"empty text or duplicate prompt id {prompt_id!r}")
        seen_ids.add(prompt_id)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        declared_hash = entry.get("text_sha256")
        if declared_hash is not None and declared_hash != text_hash:
            raise ValueError(f"text SHA-256 mismatch for prompt {prompt_id}")
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_len,
            return_attention_mask=False,
        ).input_ids
        token_ids = tuple(int(token) for token in encoded)
        if require_exact_length and len(token_ids) != max_seq_len:
            raise ValueError(
                f"prompt {prompt_id} encoded to {len(token_ids)} tokens, expected exactly {max_seq_len}"
            )
        declared_ids = entry.get("token_ids")
        if declared_ids is not None and tuple(declared_ids) != token_ids:
            raise ValueError(f"token IDs mismatch for prompt {prompt_id}")
        if entry.get("token_count") != len(token_ids):
            raise ValueError(f"token count mismatch for prompt {prompt_id}")
        frozen.append(FrozenPrompt(prompt_id, text_hash, token_ids, row_index))

    records = [prompt.record() for prompt in frozen]
    metadata = {
        "input_manifest_path": str(manifest_path.resolve()),
        "input_manifest_sha256": manifest_hash,
        "dataset": manifest.get("dataset"),
        "selection": manifest.get("selection"),
        "tokenizer": manifest.get("tokenizer"),
        "prompts": records,
        "frozen_prompt_sha256": canonical_sha256(records),
    }
    return frozen, metadata


def nf4_module_record(name: str, module: nn.Module) -> dict[str, Any]:
    state = module.weight.quant_state
    if (
        state is None
        or not state.nested
        or state.state2 is None
        or state.blocksize != NF4_BLOCKSIZE
        or state.state2.blocksize != NF4_NESTED_BLOCKSIZE
    ):
        raise RuntimeError(f"unexpected NF4 block structure in {name}")
    tensors = {
        "packed_weight": tensor_hash_record(module.weight.data),
        "absmax": tensor_hash_record(state.absmax),
        "quant_map": tensor_hash_record(state.code),
        "nested_absmax": tensor_hash_record(state.state2.absmax),
        "nested_quant_map": tensor_hash_record(state.state2.code),
        "nested_offset": tensor_hash_record(state.offset),
    }
    record = {
        "name": name,
        "in_features": module.in_features,
        "out_features": module.out_features,
        "weight_blocksize": module.weight.blocksize,
        "quant_type": state.quant_type,
        "blocksize": state.blocksize,
        "nested": state.nested,
        "nested_blocksize": state.state2.blocksize,
        "original_shape": list(state.shape),
        "original_dtype": str(state.dtype),
        "nested_dtype": str(state.state2.dtype),
        "tensors": tensors,
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def nf4_weights_manifest(
    modules: Sequence[tuple[str, nn.Module]],
) -> dict[str, Any]:
    records = [nf4_module_record(name, module) for name, module in sorted(modules)]
    return {
        "schema_version": 1,
        "module_count": len(records),
        "modules": records,
        "aggregate_sha256": canonical_sha256(records),
    }


def load_nf4_model(snapshot: Path):
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8,
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="eager",
        quantization_config=quantization,
    )
    if type(model).__name__ != "Qwen3_5ForCausalLM":
        raise RuntimeError(f"expected text-only Qwen3_5ForCausalLM, got {type(model).__name__}")
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Transformers did not mark the model as loaded in 4-bit")

    # The fit targets block 63 and never unembeds.  Keeping this 2.54 GB
    # matrix on the GPU would only reduce the retained-graph budget.
    if model.config.tie_word_embeddings:
        raise RuntimeError("the pinned 27B text config must not tie embeddings and LM head")
    if (
        model.lm_head.weight is model.model.embed_tokens.weight
        or model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    ):
        raise RuntimeError("the pinned 27B LM head unexpectedly shares embedding storage")
    model.lm_head.to(device="cpu")

    gdn_metadata = force_and_verify_torch_gdn(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    modules = dict(model.named_modules())
    nf4 = [(name, module) for name, module in modules.items() if isinstance(module, bnb.nn.Linear4bit)]
    if len(nf4) != EXPECTED_NF4_LINEARS:
        raise RuntimeError(f"expected {EXPECTED_NF4_LINEARS} NF4 linears, got {len(nf4)}")
    for name, module in nf4:
        state = getattr(module.weight, "quant_state", None)
        if (
            module.compute_dtype != torch.bfloat16
            or module.weight.dtype != torch.uint8
            or state is None
            or state.quant_type != "nf4"
            or not state.nested
            or state.state2 is None
            or state.blocksize != NF4_BLOCKSIZE
            or state.state2.blocksize != NF4_NESTED_BLOCKSIZE
            or module.weight.blocksize != NF4_BLOCKSIZE
            or not module.weight.compress_statistics
        ):
            raise RuntimeError(f"invalid NF4 double-quant state in {name}")
    quantized_weights = nf4_weights_manifest(nf4)
    if (
        isinstance(model.lm_head, bnb.nn.Linear4bit)
        or model.lm_head.weight.dtype != torch.bfloat16
        or model.lm_head.weight.device.type != "cpu"
    ):
        raise RuntimeError("LM head must remain an unquantized BF16 CPU Linear")
    if model.model.embed_tokens.weight.dtype != torch.bfloat16:
        raise RuntimeError("token embeddings must remain BF16")
    if model.model.embed_tokens.weight.device.type != "cuda":
        raise RuntimeError("token embeddings must remain on CUDA")
    if any(module.weight.device.type != "cuda" for _name, module in nf4):
        raise RuntimeError("every NF4 decoder linear must remain on CUDA")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("all model parameters must be frozen")
    return model, {
        "class": type(model).__name__,
        "nf4_linear_count": len(nf4),
        "lm_head_dtype": str(model.lm_head.weight.dtype),
        "lm_head_device": str(model.lm_head.weight.device),
        "lm_head_used_during_fit": False,
        "tie_word_embeddings": bool(model.config.tie_word_embeddings),
        "embedding_dtype": str(model.model.embed_tokens.weight.dtype),
        "memory_footprint_bytes": int(model.get_memory_footprint()),
        "pure_torch_gdn": gdn_metadata,
        "quantized_weights": quantized_weights,
    }


def force_and_verify_torch_gdn(model: nn.Module) -> dict[str, Any]:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

    layers = model.model.layers
    if len(layers) != NUM_LAYERS or model.config.hidden_size != D_MODEL:
        raise RuntimeError("Qwen architecture dimensions do not match the pinned target")
    layer_types = tuple(model.config.layer_types)
    expected = tuple(
        "linear_attention" if i in EXPECTED_LINEAR_LAYERS else "full_attention"
        for i in range(NUM_LAYERS)
    )
    if layer_types != expected:
        raise RuntimeError("Qwen layer-type schedule does not match the pinned target")

    replaced_norms = 0
    for index, layer in enumerate(layers):
        if index in EXPECTED_LINEAR_LAYERS:
            if not hasattr(layer, "linear_attn") or hasattr(layer, "self_attn"):
                raise RuntimeError(f"block {index} is not the expected GDN block")
            attention = layer.linear_attn
            attention.causal_conv1d_fn = None
            attention.causal_conv1d_update = q35.torch_causal_conv1d_update
            attention.chunk_gated_delta_rule = q35.torch_chunk_gated_delta_rule
            attention.recurrent_gated_delta_rule = q35.torch_recurrent_gated_delta_rule
            if type(attention.norm) is not q35.Qwen3_5RMSNormGated:
                old_norm = attention.norm
                replacement = q35.Qwen3_5RMSNormGated(
                    attention.head_v_dim, eps=attention.layer_norm_epsilon
                ).to(device=old_norm.weight.device, dtype=old_norm.weight.dtype)
                with torch.no_grad():
                    replacement.weight.copy_(old_norm.weight)
                attention.norm = replacement
                replaced_norms += 1
            checks = (
                attention.causal_conv1d_fn is None,
                attention.causal_conv1d_update is q35.torch_causal_conv1d_update,
                attention.chunk_gated_delta_rule is q35.torch_chunk_gated_delta_rule,
                attention.recurrent_gated_delta_rule is q35.torch_recurrent_gated_delta_rule,
                type(attention.norm) is q35.Qwen3_5RMSNormGated,
            )
            if not all(checks):
                raise RuntimeError(f"failed to force the pure-Torch GDN path at block {index}")
        elif not hasattr(layer, "self_attn") or hasattr(layer, "linear_attn"):
            raise RuntimeError(f"block {index} is not the expected full-attention block")
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("full attention must use the eager PyTorch implementation")
    return {
        "linear_attention_blocks": list(EXPECTED_LINEAR_LAYERS),
        "full_attention_blocks": list(EXPECTED_FULL_LAYERS),
        "replaced_fused_gdn_norms": replaced_norms,
        "gdn_chunk_function": inspect.getsourcefile(q35.torch_chunk_gated_delta_rule),
        "qwen_modeling_source_sha256": sha256_file(Path(q35.__file__).resolve()),
    }


def state_contract(
    args: argparse.Namespace,
    prompt_metadata: dict[str, Any],
    versions: dict[str, Any],
    artifact_records: list[dict[str, Any]],
    runtime: dict[str, Any],
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_artifacts": artifact_records,
        "quantization": {
            "method": "bitsandbytes",
            "type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
            "storage_dtype": "uint8",
            "blocksize": NF4_BLOCKSIZE,
            "nested_blocksize": NF4_NESTED_BLOCKSIZE,
        },
        "estimator": {
            "name": "anthropic_future_summed_vjp",
            "source_layers": list(SOURCE_LAYERS),
            "target_layer": TARGET_LAYER,
            "max_seq_len": args.max_seq_len,
            "skip_first": args.skip_first,
            "cotangent_batch": args.cotangent_batch,
            "row_limit": args.row_limit,
            "input_batch": 1,
            "is_grads_batched": True,
        },
        "prompts_sha256": prompt_metadata["frozen_prompt_sha256"],
        "prompt_manifest_sha256": prompt_metadata["input_manifest_sha256"],
        "seed": args.seed,
        "versions": versions,
        "runtime_identity": runtime,
        "source_identity": source_identity,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }


def initialize_or_resume_state(
    work_dir: Path,
    contract: dict[str, Any],
    prompt_count: int,
    *,
    resume: bool,
    source_observation: dict[str, Any],
) -> dict[str, Any]:
    state_path = work_dir / "state.json"
    fingerprint = canonical_sha256(contract)
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError("--resume requires an existing state.json")
        state = read_json(state_path)
        if state.get("schema_version") != STATE_SCHEMA:
            raise RuntimeError("unsupported resume-state schema")
        if state.get("contract_sha256") != fingerprint or state.get("contract") != contract:
            raise RuntimeError("resume contract mismatch")
        if state.get("prompt_count") != prompt_count:
            raise RuntimeError("resume prompt count mismatch")
        return state

    existing = [path for path in work_dir.iterdir() if path.name != ".fit.lock"]
    if state_path.exists() or existing:
        raise FileExistsError("new fit work directory must be empty; use --resume")
    state = {
        "schema_version": STATE_SCHEMA,
        "run_id": str(uuid.uuid4()),
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        "contract": contract,
        "contract_sha256": fingerprint,
        "prompt_count": prompt_count,
        "n_done": 0,
        "next_prompt": 0,
        "sum_generation": 0,
        "sum_integrity": None,
        "current": None,
        "elapsed_seconds": 0.0,
        "total_wall_seconds": 0.0,
        "invocations": [],
        "start_source_observation": source_observation,
        "model_execution": None,
        "model_execution_sha256": None,
        "publication": None,
        "max_cuda_allocated_bytes": 0,
        "max_cuda_reserved_bytes": 0,
    }
    atomic_write_json(state_path, state)
    return state


def _checkpoint_active_invocation(state: dict[str, Any]) -> None:
    invocations = state.get("invocations", [])
    if not invocations or invocations[-1].get("status") != "running":
        return
    invocation = invocations[-1]
    now_unix = time.time()
    previous = float(invocation["last_checkpoint_unix"])
    increment = max(0.0, now_unix - previous)
    invocation["elapsed_seconds"] += increment
    invocation["last_checkpoint_unix"] = now_unix
    invocation["last_checkpoint_at"] = utc_now()
    state["total_wall_seconds"] += increment


def begin_invocation(
    state: dict[str, Any], argv: Sequence[str], source_observation: dict[str, Any]
) -> int:
    if state["invocations"] and state["invocations"][-1].get("status") == "running":
        state["invocations"][-1]["status"] = "interrupted"
        state["invocations"][-1]["ended_at"] = state["invocations"][-1][
            "last_checkpoint_at"
        ]
    now = utc_now()
    now_unix = time.time()
    index = len(state["invocations"])
    state["invocations"].append(
        {
            "index": index,
            "argv": list(argv),
            "started_at": now,
            "last_checkpoint_at": now,
            "last_checkpoint_unix": now_unix,
            "elapsed_seconds": 0.0,
            "status": "running",
            "source_observation": source_observation,
        }
    )
    return index


def finish_invocation(state: dict[str, Any], status: str) -> None:
    _checkpoint_active_invocation(state)
    invocation = state["invocations"][-1]
    invocation["status"] = status
    invocation["ended_at"] = utc_now()


def save_state(
    work_dir: Path, state: dict[str, Any], *, checkpoint_invocation: bool = True
) -> None:
    if checkpoint_invocation:
        _checkpoint_active_invocation(state)
    state["updated_at"] = utc_now()
    atomic_write_json(work_dir / "state.json", state)


def cleanup_unreferenced(work_dir: Path, state: dict[str, Any]) -> None:
    current = work_dir / "current"
    if state["current"] is None and current.exists():
        shutil.rmtree(current)
    for path in work_dir.glob("sum-*.tmp"):
        shutil.rmtree(path)
    referenced = (
        work_dir / f"sum-{state['sum_generation']:06d}"
        if state["sum_generation"]
        else None
    )
    for path in work_dir.glob("sum-[0-9][0-9][0-9][0-9][0-9][0-9]"):
        if path != referenced:
            shutil.rmtree(path)
    if referenced is not None:
        validate_sum_integrity(referenced, state)


def bind_model_execution(
    work_dir: Path, state: dict[str, Any], model_metadata: dict[str, Any]
) -> None:
    fingerprint = canonical_sha256(model_metadata)
    if state["model_execution"] is None:
        state["model_execution"] = model_metadata
        state["model_execution_sha256"] = fingerprint
        save_state(work_dir, state)
        return
    if (
        state["model_execution_sha256"] != fingerprint
        or state["model_execution"] != model_metadata
    ):
        raise RuntimeError("loaded NF4 model execution manifest changed across resume")


def current_chunk_sha256(
    matrices: Sequence[np.memmap], start: int, stop: int
) -> str:
    digest = hashlib.sha256()
    for layer, matrix in zip(SOURCE_LAYERS, matrices, strict=True):
        digest.update(f"{layer}:{start}:{stop}\n".encode("ascii"))
        digest.update(np.asarray(matrix[start:stop], dtype=F32_DTYPE).tobytes(order="C"))
    return digest.hexdigest()


def validate_current_integrity(
    matrices: Sequence[np.memmap], current_state: dict[str, Any]
) -> None:
    cursor = 0
    for chunk in current_state.get("chunks", []):
        start, stop = chunk.get("start"), chunk.get("stop")
        if start != cursor or not isinstance(stop, int) or stop <= start:
            raise RuntimeError("current row-integrity chunks are not contiguous")
        actual = current_chunk_sha256(matrices, start, stop)
        if actual != chunk.get("sha256"):
            raise RuntimeError(f"current row-prefix integrity mismatch at rows {start}:{stop}")
        cursor = stop
    if cursor != current_state["next_row"]:
        raise RuntimeError("current row-integrity prefix does not reach next_row")


def validate_sum_integrity(sum_dir: Path, state: dict[str, Any]) -> None:
    integrity = state.get("sum_integrity")
    if (
        not isinstance(integrity, dict)
        or integrity.get("generation") != state["sum_generation"]
    ):
        raise RuntimeError("committed sum generation has no matching integrity manifest")
    records = integrity.get("layers")
    if not isinstance(records, list) or len(records) != len(SOURCE_LAYERS):
        raise RuntimeError("committed sum integrity manifest is incomplete")
    for layer, record in zip(SOURCE_LAYERS, records, strict=True):
        path = layer_path(sum_dir, layer)
        validate_dense_file(path)
        if record.get("layer") != layer or record.get("size") != path.stat().st_size:
            raise RuntimeError(f"committed sum metadata mismatch at layer {layer}")
        if sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"committed sum hash mismatch at layer {layer}")


def prepare_current(work_dir: Path, state: dict[str, Any]) -> list[np.memmap]:
    current = work_dir / "current"
    if state["current"] is None:
        if current.exists():
            shutil.rmtree(current)
        temporary = work_dir / "current.init"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        for layer in SOURCE_LAYERS:
            matrix = np.memmap(
                layer_path(temporary, layer), mode="w+", dtype=F32_DTYPE, shape=(D_MODEL, D_MODEL)
            )
            matrix.flush()
            del matrix
        os.replace(temporary, current)
        state["current"] = {
            "prompt_index": state["next_prompt"],
            "next_row": 0,
            "chunks": [],
        }
        save_state(work_dir, state)
    elif state["current"]["prompt_index"] != state["next_prompt"]:
        raise RuntimeError("current prompt index disagrees with next_prompt")

    matrices: list[np.memmap] = []
    for layer in SOURCE_LAYERS:
        path = layer_path(current, layer)
        validate_dense_file(path)
        matrices.append(np.memmap(path, mode="r+", dtype=F32_DTYPE, shape=(D_MODEL, D_MODEL)))
    validate_current_integrity(matrices, state["current"])
    return matrices


def commit_current_prompt(work_dir: Path, state: dict[str, Any]) -> None:
    if state["current"] is None or state["current"]["next_row"] != D_MODEL:
        raise RuntimeError("cannot commit an incomplete prompt")
    old_generation = state["sum_generation"]
    new_generation = old_generation + 1
    old_dir = work_dir / f"sum-{old_generation:06d}" if old_generation else None
    current = work_dir / "current"
    temporary = work_dir / f"sum-{new_generation:06d}.tmp"
    final = work_dir / f"sum-{new_generation:06d}"
    if temporary.exists():
        shutil.rmtree(temporary)
    if final.exists():
        shutil.rmtree(final)
    temporary.mkdir()
    sum_records: list[dict[str, Any]] = []
    for layer in SOURCE_LAYERS:
        current_matrix = np.memmap(
            layer_path(current, layer), mode="r", dtype=F32_DTYPE, shape=(D_MODEL, D_MODEL)
        )
        old_matrix = (
            np.memmap(layer_path(old_dir, layer), mode="r", dtype=F32_DTYPE, shape=(D_MODEL, D_MODEL))
            if old_dir
            else None
        )
        new_matrix = np.memmap(
            layer_path(temporary, layer), mode="w+", dtype=F32_DTYPE, shape=(D_MODEL, D_MODEL)
        )
        for row in range(0, D_MODEL, 64):
            stop = min(row + 64, D_MODEL)
            if old_matrix is None:
                new_matrix[row:stop] = current_matrix[row:stop]
            else:
                np.add(old_matrix[row:stop], current_matrix[row:stop], out=new_matrix[row:stop])
        new_matrix.flush()
        del current_matrix, old_matrix, new_matrix
        new_path = layer_path(temporary, layer)
        sum_records.append(
            {
                "layer": layer,
                "size": new_path.stat().st_size,
                "sha256": sha256_file(new_path),
            }
        )
    os.replace(temporary, final)
    state["n_done"] = new_generation
    state["next_prompt"] = new_generation
    state["sum_generation"] = new_generation
    state["sum_integrity"] = {
        "generation": new_generation,
        "layers": sum_records,
        "aggregate_sha256": canonical_sha256(sum_records),
    }
    state["current"] = None
    save_state(work_dir, state)
    shutil.rmtree(current)
    if old_dir:
        shutil.rmtree(old_dir)


def fit_one_prompt(
    model: nn.Module,
    prompt: FrozenPrompt,
    work_dir: Path,
    state: dict[str, Any],
    *,
    skip_first: int,
    cotangent_batch: int,
    row_limit: int = D_MODEL,
    commit: bool = True,
    validate_vectorization: bool = False,
) -> None:
    matrices = prepare_current(work_dir, state)
    row_start = int(state["current"]["next_row"])
    input_ids = torch.tensor(prompt.token_ids, device="cuda:0", dtype=torch.long)[None, :]
    valid_positions = valid_estimator_positions(
        input_ids.shape[1], skip_first, device=input_ids.device
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with ActivationCapture(
        model.model.layers,
        [*SOURCE_LAYERS, TARGET_LAYER],
        graph_root=SOURCE_LAYERS[0],
    ) as capture, torch.enable_grad():
        model.model(input_ids=input_ids, use_cache=False)
        missing = set((*SOURCE_LAYERS, TARGET_LAYER)) - set(capture.activations)
        if missing:
            raise RuntimeError(f"activation hooks did not fire for blocks {sorted(missing)}")
        target = capture.activations[TARGET_LAYER]
        sources = [capture.activations[layer] for layer in SOURCE_LAYERS]
        for start, stop, rows_by_layer in iter_future_summed_row_chunks(
            target,
            sources,
            valid_positions,
            row_start=row_start,
            row_stop=row_limit,
            cotangent_batch=cotangent_batch,
            retain_graph_after=validate_vectorization,
        ):
            for matrix, rows in zip(matrices, rows_by_layer, strict=True):
                matrix[start:stop] = rows.detach().cpu().numpy().astype(F32_DTYPE, copy=False)
                matrix.flush()
            state["current"]["chunks"].append(
                {
                    "start": start,
                    "stop": stop,
                    "sha256": current_chunk_sha256(matrices, start, stop),
                }
            )
            state["current"]["next_row"] = stop
            state["max_cuda_allocated_bytes"] = max(
                state["max_cuda_allocated_bytes"], torch.cuda.max_memory_allocated()
            )
            state["max_cuda_reserved_bytes"] = max(
                state["max_cuda_reserved_bytes"], torch.cuda.max_memory_reserved()
            )
            state["elapsed_seconds"] += time.monotonic() - started
            started = time.monotonic()
            save_state(work_dir, state)
            print(
                f"prompt {state['next_prompt'] + 1}/{state['prompt_count']} "
                f"rows {start}:{stop}/{D_MODEL}",
                flush=True,
            )
        if validate_vectorization:
            validation_layers = (61, 62)
            if not set(validation_layers).issubset(SOURCE_LAYERS):
                raise RuntimeError("diagnostic validation layers are not captured sources")
            validation_sources = [
                capture.activations[layer] for layer in validation_layers
            ]
            sequential = sequential_future_summed_rows(
                target,
                validation_sources,
                valid_positions,
                0,
                row_limit,
            )
            parity_records: list[dict[str, Any]] = []
            for layer, reference in zip(
                validation_layers, sequential, strict=True
            ):
                matrix_index = SOURCE_LAYERS.index(layer)
                batched = torch.from_numpy(
                    np.array(matrices[matrix_index][:row_limit], copy=True)
                ).to(device=reference.device)
                metrics = vectorization_error_metrics(batched, reference)
                record = {
                    "layer": layer,
                    "rows": row_limit,
                    **metrics,
                    "max_abs_tolerance": 1e-4,
                    "relative_rms_tolerance": 1e-4,
                }
                if metrics["max_abs"] > 1e-4 or metrics["relative_rms"] > 1e-4:
                    raise RuntimeError(
                        f"batched/sequential diagnostic mismatch at source layer {layer}: {metrics}"
                    )
                parity_records.append(record)
            state["diagnostic_vectorization_parity"] = {
                "method": "same_graph_sequential_autograd",
                "target_layer": TARGET_LAYER,
                "layers": parity_records,
                "passed": True,
            }
            save_state(work_dir, state)
    for matrix in matrices:
        matrix.flush()
    del matrices
    if commit:
        if row_limit != D_MODEL:
            raise RuntimeError("a row-limited diagnostic cannot commit a prompt")
        commit_current_prompt(work_dir, state)


def publish_incomplete_diagnostic(
    path: Path,
    args: argparse.Namespace,
    state: dict[str, Any],
    prompt_metadata: dict[str, Any],
    model_metadata: dict[str, Any],
) -> str:
    diagnostic = {
        "schema_version": 1,
        "status": "incomplete_diagnostic",
        "complete": False,
        "lens_exported": False,
        "run_id": state["run_id"],
        "started_at": state["started_at"],
        "stopped_at": utc_now(),
        "model": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            **model_metadata,
        },
        "contract": state["contract"],
        "contract_sha256": state["contract_sha256"],
        "prompts": prompt_metadata,
        "diagnostic": {
            "prompt_index": state["next_prompt"],
            "rows_computed": state["current"]["next_row"],
            "rows_required_for_complete_matrix": D_MODEL,
            "source_layers": list(SOURCE_LAYERS),
            "target_layer": TARGET_LAYER,
            "vectorization_parity": state.get("diagnostic_vectorization_parity"),
        },
        "resources": {
            "max_cuda_allocated_bytes": state["max_cuda_allocated_bytes"],
            "max_cuda_reserved_bytes": state["max_cuda_reserved_bytes"],
        },
        "source": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_write_json(path, diagnostic)
    return sha256_file(path)


def matrix_statistics(
    path: Path,
    d_model: int | None = None,
    *,
    dtype: np.dtype[Any] = F32_DTYPE,
    dtype_name: str = "float32",
) -> dict[str, Any]:
    if d_model is None:
        d_model = D_MODEL
    matrix = np.memmap(path, mode="r", dtype=dtype, shape=(d_model, d_model))
    finite = 0
    minimum = math.inf
    maximum = -math.inf
    squared_norm = 0.0
    for row in range(0, d_model, 64):
        chunk = np.asarray(matrix[row : min(row + 64, d_model)])
        finite += int(np.isfinite(chunk).sum())
        minimum = min(minimum, float(chunk.min()))
        maximum = max(maximum, float(chunk.max()))
        squared_norm += float(np.square(chunk.astype(np.float64)).sum())
    if finite != d_model * d_model:
        raise FloatingPointError(f"non-finite final matrix {path}")
    trace = float(np.trace(matrix, dtype=np.float64))
    del matrix
    return {
        "shape": [d_model, d_model],
        "dtype": dtype_name,
        "finite_count": finite,
        "min": minimum,
        "max": maximum,
        "frobenius_norm": math.sqrt(squared_norm),
        "trace": trace,
        "sha256": sha256_file(path),
    }


def build_lens(
    work_dir: Path,
    state: dict[str, Any],
    output: Path,
    *,
    output_dtype: str,
) -> tuple[str, list[dict[str, Any]]]:
    if state["n_done"] != state["prompt_count"] or state["current"] is not None:
        raise RuntimeError("cannot publish before every prompt is committed")
    sum_dir = work_dir / f"sum-{state['sum_generation']:06d}"
    mean_dir = work_dir / "final-mean"
    if mean_dir.exists():
        shutil.rmtree(mean_dir)
    mean_dir.mkdir()
    publish_dtype = np.dtype("<f4" if output_dtype == "float32" else "<f2")
    statistics: list[dict[str, Any]] = []
    publish_files: list[Path] = []
    for layer in SOURCE_LAYERS:
        summed = np.memmap(
            layer_path(sum_dir, layer), mode="r", dtype=F32_DTYPE, shape=(D_MODEL, D_MODEL)
        )
        mean_path = layer_path(mean_dir, layer)
        mean = np.memmap(mean_path, mode="w+", dtype=F32_DTYPE, shape=(D_MODEL, D_MODEL))
        for row in range(0, D_MODEL, 64):
            stop = min(row + 64, D_MODEL)
            np.divide(summed[row:stop], state["n_done"], out=mean[row:stop])
        mean.flush()
        del summed, mean
        layer_stats = matrix_statistics(mean_path)
        layer_stats["layer"] = layer
        if output_dtype == "float32":
            publish_files.append(mean_path)
            layer_stats["published"] = {
                key: value for key, value in layer_stats.items() if key != "layer"
            }
        else:
            if max(abs(layer_stats["min"]), abs(layer_stats["max"])) > np.finfo(np.float16).max:
                raise FloatingPointError(f"layer {layer} overflows FP16")
            fp16_path = mean_dir / f"layer-{layer:02d}.f16"
            src = np.memmap(mean_path, mode="r", dtype=F32_DTYPE, shape=(D_MODEL, D_MODEL))
            dst = np.memmap(fp16_path, mode="w+", dtype=publish_dtype, shape=(D_MODEL, D_MODEL))
            for row in range(0, D_MODEL, 64):
                stop = min(row + 64, D_MODEL)
                dst[row:stop] = src[row:stop]
            dst.flush()
            if not bool(np.isfinite(dst).all()):
                raise FloatingPointError(f"non-finite FP16 publication layer {layer}")
            del src, dst
            publish_files.append(fp16_path)
            layer_stats["published"] = matrix_statistics(
                fp16_path,
                dtype=np.dtype("<f2"),
                dtype_name="float16",
            )
        statistics.append(layer_stats)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    mappings: list[np.memmap] = []
    jacobians: dict[int, torch.Tensor] = {}
    for layer, path in zip(SOURCE_LAYERS, publish_files, strict=True):
        mapping = np.memmap(path, mode="r+", dtype=publish_dtype, shape=(D_MODEL, D_MODEL))
        mappings.append(mapping)
        jacobians[layer] = torch.from_numpy(mapping)
    checkpoint = {
        "J": jacobians,
        "n_prompts": state["n_done"],
        "d_model": D_MODEL,
        "source_layers": list(SOURCE_LAYERS),
        "metadata": {
            "fit_model": MODEL_ID,
            "fit_model_revision": MODEL_REVISION,
            "fit_quantization": "bitsandbytes-nf4-double-quant-bfloat16",
            "estimator": "anthropic-future-summed",
            "source_layers": list(SOURCE_LAYERS),
            "target_layer": TARGET_LAYER,
            "contract_sha256": state["contract_sha256"],
            "storage_dtype": output_dtype,
            "nf4_aggregate_sha256": state["model_execution"]["quantized_weights"][
                "aggregate_sha256"
            ],
        },
    }
    torch.save(checkpoint, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    mappings.clear()
    return sha256_file(output), statistics


def _resolved(path: Path) -> str:
    return str(path.expanduser().resolve())


def stage_publication(
    work_dir: Path,
    state: dict[str, Any],
    output: Path,
    provenance: Path,
    output_dtype: str,
    *,
    resume: bool,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    provenance = provenance.expanduser().resolve()
    publication = state.get("publication")
    if publication is not None:
        expected = (_resolved(output), _resolved(provenance), output_dtype)
        actual = (
            publication.get("output_path"),
            publication.get("provenance_path"),
            publication.get("output_dtype"),
        )
        if actual != expected:
            raise RuntimeError("publication paths or dtype changed across resume")
        if not resume and publication.get("status") != "staged":
            raise RuntimeError("an interrupted publication requires explicit --resume")
        return publication

    if output.exists() or provenance.exists():
        raise FileExistsError(
            "publication destination already exists without a matching persisted transaction"
        )
    directory = work_dir / f"publication-{state['run_id']}"
    if directory.exists():
        raise FileExistsError(f"unowned publication staging directory exists: {directory}")
    directory.mkdir()
    publication = {
        "schema_version": 1,
        "status": "staged",
        "staged_at": utc_now(),
        "output_path": str(output),
        "provenance_path": str(provenance),
        "output_dtype": output_dtype,
        "staging_directory": str(directory.resolve()),
        "artifact_candidate": str((directory / "lens.pt").resolve()),
        "artifact_publish_temp": str(
            (output.parent / f".{output.name}.publish-{state['run_id']}").resolve()
        ),
        "provenance_candidate": str((directory / "provenance.json").resolve()),
        "provenance_publish_temp": str(
            (
                provenance.parent
                / f".{provenance.name}.publish-{state['run_id']}"
            ).resolve()
        ),
    }
    state["publication"] = publication
    save_state(work_dir, state)
    return publication


def publish_candidate_exclusive(
    candidate: Path,
    destination: Path,
    temporary: Path,
    expected_sha256: str,
    *,
    allow_matching_existing: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        require_regular_file(destination, label="publication destination")
        actual = sha256_file(destination)
        if not allow_matching_existing:
            raise FileExistsError(f"refusing to overwrite existing {destination}")
        if actual != expected_sha256:
            raise RuntimeError(
                f"persisted publication hash mismatch for {destination}: "
                f"{actual} != {expected_sha256}"
            )
        return

    require_regular_file(candidate, label="publication candidate")
    if sha256_file(candidate) != expected_sha256:
        raise RuntimeError(f"publication candidate hash mismatch: {candidate}")
    if temporary.exists() or temporary.is_symlink():
        require_regular_file(temporary, label="declared publication temporary")
        if sha256_file(temporary) != expected_sha256:
            # This exact temporary path is owned by the persisted run transaction.
            temporary.unlink()
    if not temporary.exists():
        with candidate.open("rb") as source, temporary.open("xb") as target:
            shutil.copyfileobj(source, target, 16 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
    if sha256_file(temporary) != expected_sha256:
        raise RuntimeError(f"publication temporary hash mismatch: {temporary}")
    try:
        os.link(temporary, destination)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise
        if not allow_matching_existing:
            raise FileExistsError(f"refusing to overwrite existing {destination}") from error
        require_regular_file(destination, label="publication destination")
        if sha256_file(destination) != expected_sha256:
            raise RuntimeError(f"publication destination changed concurrently: {destination}")
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    temporary.unlink(missing_ok=True)


def git_metadata(root: Path) -> dict[str, Any]:
    def run(*command: str) -> str:
        result = subprocess.run(
            command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False
        )
        return result.stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD") or None,
        "status_porcelain": run("git", "status", "--short"),
    }


def environment_metadata() -> dict[str, Any]:
    packages = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    gpu = torch.cuda.get_device_properties(0)
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.strip()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "gpu": {
            "name": gpu.name,
            "compute_capability": [gpu.major, gpu.minor],
            "total_memory_bytes": gpu.total_memory,
            "nvidia_smi": smi,
        },
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def publish_provenance(
    path: Path,
    args: argparse.Namespace,
    state: dict[str, Any],
    prompt_metadata: dict[str, Any],
    lens_sha256: str,
    layer_statistics: list[dict[str, Any]],
    output: Path,
    completion_snapshot: dict[str, Any],
) -> str:
    provenance = {
        "schema_version": 1,
        "status": "completed",
        "complete": True,
        "run_id": state["run_id"],
        "started_at": state["started_at"],
        "completed_at": completion_snapshot["completed_at"],
        "estimator_elapsed_seconds": state["elapsed_seconds"],
        "total_wall_seconds": completion_snapshot["total_wall_seconds"],
        "invocations": completion_snapshot["invocations"],
        "model": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "loader": "AutoModelForCausalLM text-only",
            "execution_manifest_sha256": state["model_execution_sha256"],
            **state["model_execution"],
        },
        "contract": state["contract"],
        "contract_sha256": state["contract_sha256"],
        "prompts": prompt_metadata,
        "result": {
            "path": str(output.resolve()),
            "sha256": lens_sha256,
            "storage_dtype": args.output_dtype,
            "n_prompts": state["n_done"],
            "d_model": D_MODEL,
            "layers": layer_statistics,
        },
        "resources": {
            "max_cuda_allocated_bytes": state["max_cuda_allocated_bytes"],
            "max_cuda_reserved_bytes": state["max_cuda_reserved_bytes"],
        },
        "environment": completion_snapshot["environment"],
        "source": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "identity": state["contract"]["source_identity"],
            "start_observation": state["start_source_observation"],
        },
    }
    atomic_write_json(path, provenance)
    return sha256_file(path)


def finalize_publication(
    args: argparse.Namespace,
    state: dict[str, Any],
    prompt_metadata: dict[str, Any],
    provenance_path: Path,
) -> tuple[str, str]:
    if state["model_execution"] is None:
        raise RuntimeError("cannot publish without a persisted NF4 execution manifest")
    validate_sum_integrity(
        args.work_dir / f"sum-{state['sum_generation']:06d}", state
    )
    publication = stage_publication(
        args.work_dir,
        state,
        args.output,
        provenance_path,
        args.output_dtype,
        resume=args.resume,
    )
    artifact_candidate = Path(publication["artifact_candidate"])
    output = Path(publication["output_path"])
    if "artifact_sha256" not in publication:
        if artifact_candidate.exists() or artifact_candidate.is_symlink():
            require_regular_file(artifact_candidate, label="owned artifact candidate")
            artifact_candidate.unlink()
        lens_sha256, layer_statistics = build_lens(
            args.work_dir,
            state,
            artifact_candidate,
            output_dtype=args.output_dtype,
        )
        publication.update(
            {
                "artifact_sha256": lens_sha256,
                "artifact_size": artifact_candidate.stat().st_size,
                "layer_statistics": layer_statistics,
                "status": "artifact_ready",
                "artifact_ready_at": utc_now(),
            }
        )
        save_state(args.work_dir, state)
    lens_sha256 = publication["artifact_sha256"]
    publish_candidate_exclusive(
        artifact_candidate,
        output,
        Path(publication["artifact_publish_temp"]),
        lens_sha256,
        allow_matching_existing=args.resume,
    )
    publication["status"] = "artifact_published"
    publication["artifact_published_at"] = utc_now()
    save_state(args.work_dir, state)

    if "completion_snapshot" not in publication:
        _checkpoint_active_invocation(state)
        completed_at = utc_now()
        invocations = copy.deepcopy(state["invocations"])
        invocations[-1]["status"] = "completed"
        invocations[-1]["ended_at"] = completed_at
        publication["completion_snapshot"] = {
            "completed_at": completed_at,
            "total_wall_seconds": state["total_wall_seconds"],
            "invocations": invocations,
            "environment": environment_metadata(),
        }
        save_state(args.work_dir, state, checkpoint_invocation=False)
    provenance_candidate = Path(publication["provenance_candidate"])
    if "provenance_sha256" not in publication:
        if provenance_candidate.exists() or provenance_candidate.is_symlink():
            require_regular_file(provenance_candidate, label="owned provenance candidate")
            provenance_candidate.unlink()
        provenance_sha256 = publish_provenance(
            provenance_candidate,
            args,
            state,
            prompt_metadata,
            lens_sha256,
            publication["layer_statistics"],
            output,
            publication["completion_snapshot"],
        )
        publication.update(
            {
                "provenance_sha256": provenance_sha256,
                "provenance_size": provenance_candidate.stat().st_size,
                "status": "provenance_ready",
                "provenance_ready_at": utc_now(),
            }
        )
        save_state(args.work_dir, state)
    provenance_sha256 = publication["provenance_sha256"]
    publish_candidate_exclusive(
        provenance_candidate,
        Path(publication["provenance_path"]),
        Path(publication["provenance_publish_temp"]),
        provenance_sha256,
        allow_matching_existing=args.resume,
    )
    publication["status"] = "completed"
    publication["provenance_published_at"] = utc_now()
    state["status"] = "completed"
    state["output"] = {
        "path": publication["output_path"],
        "sha256": lens_sha256,
        "provenance_path": publication["provenance_path"],
        "provenance_sha256": provenance_sha256,
    }
    finish_invocation(state, "completed")
    save_state(args.work_dir, state, checkpoint_invocation=False)
    return lens_sha256, provenance_sha256


@contextlib.contextmanager
def exclusive_work_lock(work_dir: Path) -> Iterator[None]:
    lock_path = work_dir / ".fit.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another fitter owns {lock_path}") from error
        yield


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "jlens_nf4_fit_prompts.json",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-short-prompts", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--cotangent-batch", type=int, default=4)
    parser.add_argument(
        "--row-limit",
        type=int,
        default=D_MODEL,
        help="diagnostic output-row limit; values below 5120 can never export a lens",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        help="stop after this many total committed manifest prompts; resume may raise it",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dtype", choices=("float32", "float16"), default="float32")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_seq_len <= 0 or not 0 <= args.skip_first < args.max_seq_len:
        raise ValueError("invalid max-seq-len/skip-first")
    if not 1 <= args.cotangent_batch <= 64:
        raise ValueError("cotangent-batch must be in [1, 64]")
    if not 1 <= args.row_limit <= D_MODEL:
        raise ValueError(f"row-limit must be in [1, {D_MODEL}]")
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise ValueError("max-prompts must be positive")
    if not args.prompt_manifest.is_file():
        raise FileNotFoundError(args.prompt_manifest)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    provenance = args.provenance or args.output.with_suffix(args.output.suffix + ".provenance.json")
    if args.output.expanduser().resolve() == provenance.expanduser().resolve():
        raise ValueError("output and provenance paths must be distinct")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if provenance.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {provenance}")
    stat = shutil.disk_usage(args.work_dir)
    if stat.free < 32 * 2**30:
        raise RuntimeError("at least 32 GiB free disk is required in work-dir")


def main(argv: Sequence[str] | None = None) -> int:
    # cuBLAS reads this when its CUDA handle is initialized, which can happen
    # during the runtime checks below.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    invocation_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(invocation_argv)
    repo_root = Path(__file__).resolve().parents[1]
    source_identity, source_observation = source_tree_identity(
        repo_root, require_clean=not args.resume
    )
    validate_args(args)
    provenance_path = args.provenance or args.output.with_suffix(args.output.suffix + ".provenance.json")
    with exclusive_work_lock(args.work_dir):
        versions = require_runtime()
        configure_determinism(args.seed)
        runtime = runtime_identity(versions, args.seed)
        snapshot, artifact_records = resolve_and_verify_snapshot(args.cache_dir, args.allow_download)
        tokenizer = load_tokenizer(snapshot)
        prompts, prompt_metadata = freeze_prompts(
            args.prompt_manifest,
            tokenizer,
            max_seq_len=args.max_seq_len,
            require_exact_length=not args.allow_short_prompts,
        )
        contract = state_contract(
            args,
            prompt_metadata,
            versions,
            artifact_records,
            runtime,
            source_identity,
        )
        state = initialize_or_resume_state(
            args.work_dir,
            contract,
            len(prompts),
            resume=args.resume,
            source_observation=source_observation,
        )
        begin_invocation(state, invocation_argv, source_observation)
        save_state(args.work_dir, state)
        if state["status"] == "completed":
            lens_sha256, provenance_sha256 = finalize_publication(
                args, state, prompt_metadata, provenance_path
            )
            print(f"lens already completed and verified: {args.output} ({lens_sha256})")
            print(f"provenance: {provenance_path} ({provenance_sha256})")
            return 0
        state["status"] = "running"
        save_state(args.work_dir, state)
        cleanup_unreferenced(args.work_dir, state)

        if args.max_prompts is not None and args.max_prompts > len(prompts):
            raise ValueError("max-prompts exceeds the frozen manifest count")
        if state["next_prompt"] == len(prompts):
            lens_sha256, provenance_sha256 = finalize_publication(
                args, state, prompt_metadata, provenance_path
            )
            print(f"lens: {args.output} ({lens_sha256})")
            print(f"provenance: {provenance_path} ({provenance_sha256})")
            return 0

        require_model_memory()
        model, model_metadata = load_nf4_model(snapshot)
        bind_model_execution(args.work_dir, state, model_metadata)
        if args.row_limit < D_MODEL:
            if state["next_prompt"] != 0 or state["n_done"] != 0:
                raise RuntimeError("row-limit diagnostics require an uncommitted first prompt")
            fit_one_prompt(
                model,
                prompts[0],
                args.work_dir,
                state,
                skip_first=args.skip_first,
                cotangent_batch=args.cotangent_batch,
                row_limit=args.row_limit,
                commit=False,
                validate_vectorization=True,
            )
            state["status"] = "incomplete_diagnostic"
            finish_invocation(state, "incomplete_diagnostic")
            save_state(args.work_dir, state, checkpoint_invocation=False)
            diagnostic_hash = publish_incomplete_diagnostic(
                provenance_path, args, state, prompt_metadata, model_metadata
            )
            state["diagnostic"] = {
                "provenance_path": str(provenance_path.resolve()),
                "provenance_sha256": diagnostic_hash,
                "lens_exported": False,
            }
            save_state(args.work_dir, state, checkpoint_invocation=False)
            print(f"incomplete diagnostic only: {provenance_path} ({diagnostic_hash})")
            print("no lens was exported", flush=True)
            return 0

        stop_prompt = args.max_prompts if args.max_prompts is not None else len(prompts)
        while state["next_prompt"] < stop_prompt:
            fit_one_prompt(
                model,
                prompts[state["next_prompt"]],
                args.work_dir,
                state,
                skip_first=args.skip_first,
                cotangent_batch=args.cotangent_batch,
            )
        if state["next_prompt"] < len(prompts):
            state["status"] = "paused_incomplete"
            finish_invocation(state, "paused_incomplete")
            save_state(args.work_dir, state, checkpoint_invocation=False)
            print(
                f"paused after {state['n_done']}/{len(prompts)} prompts; "
                "no lens or completion provenance was exported",
                flush=True,
            )
            return 0
        del model
        torch.cuda.empty_cache()
        lens_sha256, provenance_sha256 = finalize_publication(
            args, state, prompt_metadata, provenance_path
        )
        print(f"lens: {args.output} ({lens_sha256})")
        print(f"provenance: {provenance_path} ({provenance_sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
