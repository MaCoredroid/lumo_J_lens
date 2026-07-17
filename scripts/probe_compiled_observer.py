#!/usr/bin/env python3
"""Prove a compile-visible observer on the deployed Qwen3.6 NVFP4 graph.

The ``both`` mode launches two isolated, cold-compiled vLLM processes.  The
stock child is untouched.  The observer child installs its custom operator and
constructor/forward patches before constructing ``LLM`` (and therefore before
the model is traced or compiled).  The observer is a side effect after each
selected GEMM; the tensor used by the model is returned unchanged.

MTP is deliberately disabled.  This probe covers the main-model prefill that
produces the first generated token.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
MODEL_IDENTITY_POLICY = "pinned-nvidia-modelopt-snapshot-v1"
PINNED_METADATA_SHA256 = {
    "config.json": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
    "hf_quant_config.json": "fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1",
    "model.safetensors.index.json": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
}
DEFAULT_PROMPT = "Fact: The currency used in the country shaped like a boot is"
TARGET_LAYERS = (61, 62, 63)
CAPTURE_CAPACITY = 128
SCHEMA_VERSION = 1
OBSERVER_OP_NAME = "lumo_compiled_observe_v1"
DISABLED_KERNELS = (
    "FlashInferFP8ScaledMMLinearKernel,"
    "FlashInferCutlassNvFp4LinearKernel,"
    "FlashInferTrtllmNvFp4LinearKernel,"
    "FlashInferCudnnNvFp4LinearKernel"
)

TARGET_LINEAR_SUFFIXES = {
    61: (
        "linear_attn.in_proj_qkvz",
        "linear_attn.in_proj_ba",
        "linear_attn.out_proj",
        "mlp.gate_up_proj",
        "mlp.down_proj",
    ),
    62: (
        "linear_attn.in_proj_qkvz",
        "linear_attn.in_proj_ba",
        "linear_attn.out_proj",
        "mlp.gate_up_proj",
        "mlp.down_proj",
    ),
    63: (
        "self_attn.qkv_proj",
        "self_attn.o_proj",
        "mlp.gate_up_proj",
        "mlp.down_proj",
    ),
}
EXPECTED_LINEAR_NAMES = tuple(
    f"layers.{layer}.{suffix}"
    for layer in TARGET_LAYERS
    for suffix in TARGET_LINEAR_SUFFIXES[layer]
)
_LAYER_PREFIX = re.compile(r"(?:^|\.)layers\.(61|62|63)\.(.+)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("stock", "observer", "both"), default="both"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help=(
            "diagnostic path override; accepted only when it resolves to the exact "
            "locally cached pinned NVIDIA snapshot"
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensor-output", type=Path)
    parser.add_argument("--debug-dump-root", type=Path)
    parser.add_argument(
        "--observer-scope",
        default="full",
        help=(
            "full, h-only, linears-only, h:<61|62|63>, or "
            "linear:<canonical layers.N... name>"
        ),
    )
    parser.add_argument("--capture-capacity", type=int, default=CAPTURE_CAPACITY)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    return parser.parse_args()


def _prepare_process_environment() -> None:
    # Set these before importing vLLM.  A cold compile makes the stock and
    # observer fusion counts independently meaningful.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_DISABLED_KERNELS", DISABLED_KERNELS)
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _canonical_linear_name(prefix: str) -> str | None:
    match = _LAYER_PREFIX.search(prefix)
    if match is None:
        return None
    layer = int(match.group(1))
    suffix = match.group(2)
    if suffix not in TARGET_LINEAR_SUFFIXES[layer]:
        return None
    return f"layers.{layer}.{suffix}"


def _parse_observer_scope(scope: str) -> tuple[set[str], set[int]]:
    if scope == "full":
        return set(EXPECTED_LINEAR_NAMES), set(TARGET_LAYERS)
    if scope == "h-only":
        return set(), set(TARGET_LAYERS)
    if scope == "linears-only":
        return set(EXPECTED_LINEAR_NAMES), set()
    if scope.startswith("h:"):
        try:
            layer = int(scope.removeprefix("h:"))
        except ValueError as exc:
            raise ValueError(f"invalid observer scope: {scope!r}") from exc
        if layer not in TARGET_LAYERS:
            raise ValueError(f"observer h layer must be one of {TARGET_LAYERS}")
        return set(), {layer}
    if scope.startswith("linear:"):
        name = scope.removeprefix("linear:")
        if name not in EXPECTED_LINEAR_NAMES:
            raise ValueError(
                f"unknown observer linear {name!r}; expected one of "
                f"{EXPECTED_LINEAR_NAMES}"
            )
        return {name}, set()
    raise ValueError(f"invalid observer scope: {scope!r}")


def _observe_impl(source: Any, destination: Any, row_count: Any) -> None:
    """Copy a flattened token axis without changing ``source``."""
    source = source.reshape(-1, source.shape[-1])
    copied = min(source.shape[0], destination.shape[0])
    destination[:copied].copy_(source[:copied])
    row_count.fill_(source.shape[0])


def _observe_fake(source: Any, destination: Any, row_count: Any) -> None:
    return None


def _register_observer_op() -> Any:
    import torch
    from vllm.utils.torch_utils import direct_register_custom_op

    # Keep torch out of module import so allocator environment variables can be
    # set first, then provide concrete types for torch.library schema inference.
    tensor_annotations = {
        "source": torch.Tensor,
        "destination": torch.Tensor,
        "row_count": torch.Tensor,
        "return": None,
    }
    _observe_impl.__annotations__ = tensor_annotations
    _observe_fake.__annotations__ = tensor_annotations
    direct_register_custom_op(
        OBSERVER_OP_NAME,
        _observe_impl,
        mutates_args=["destination", "row_count"],
        fake_impl=_observe_fake,
    )
    return getattr(torch.ops.vllm, OBSERVER_OP_NAME).default


def _allocate_observer_buffers(
    module: Any,
    *,
    width: int,
    capacity: int,
    dtype: Any,
    canonical_name: str,
) -> None:
    import torch

    module.register_buffer(
        "_lumo_observer_buffer",
        torch.full((capacity, width), float("nan"), dtype=dtype),
        persistent=False,
    )
    module.register_buffer(
        "_lumo_observer_row_count",
        torch.zeros((), dtype=torch.int32),
        persistent=False,
    )
    module._lumo_observer_name = canonical_name


def _install_observer_before_model_construction(
    capacity: int, *, scope: str
) -> dict[str, Any]:
    """Patch constructor-visible vLLM classes before ``LLM`` is constructed."""
    if capacity < 1:
        raise ValueError("capture capacity must be positive")

    from vllm.model_executor.layers import linear as linear_module
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.models import qwen3_5

    observer_op = _register_observer_op()
    selected_linears, selected_h_layers = _parse_observer_scope(scope)
    original_linear_init = LinearBase.__init__

    @functools.wraps(original_linear_init)
    def linear_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_linear_init(self, *args, **kwargs)
        canonical_name = _canonical_linear_name(self.prefix)
        if canonical_name not in selected_linears:
            return
        width = int(getattr(self, "output_size_per_partition", self.output_size))
        _allocate_observer_buffers(
            self,
            width=width,
            capacity=capacity,
            dtype=self.params_dtype,
            canonical_name=canonical_name,
        )

    LinearBase.__init__ = linear_init

    patched_linear_classes: list[str] = []
    for value in vars(linear_module).values():
        if not isinstance(value, type) or not issubclass(value, LinearBase):
            continue
        if "forward" not in value.__dict__:
            continue
        original_forward = value.forward

        @functools.wraps(original_forward)
        def linear_forward(
            self: Any,
            *args: Any,
            __original: Any = original_forward,
            **kwargs: Any,
        ) -> Any:
            result = __original(self, *args, **kwargs)
            if hasattr(self, "_lumo_observer_buffer"):
                output = result[0] if isinstance(result, tuple) else result
                observer_op(
                    output.reshape(-1, output.shape[-1]),
                    self._lumo_observer_buffer,
                    self._lumo_observer_row_count,
                )
            return result

        value.forward = linear_forward
        patched_linear_classes.append(value.__name__)

    decoder_class = qwen3_5.Qwen3_5DecoderLayer
    original_decoder_init = decoder_class.__init__
    original_decoder_forward = decoder_class.forward

    @functools.wraps(original_decoder_init)
    def decoder_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_decoder_init(self, *args, **kwargs)
        if self.layer_idx not in selected_h_layers:
            return
        vllm_config = args[0] if args else kwargs["vllm_config"]
        config = vllm_config.model_config.hf_text_config
        _allocate_observer_buffers(
            self,
            width=int(config.hidden_size),
            capacity=capacity,
            dtype=vllm_config.model_config.dtype,
            canonical_name=f"h{self.layer_idx}_post_block",
        )

    @functools.wraps(original_decoder_forward)
    def decoder_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_decoder_forward(self, *args, **kwargs)
        if hasattr(self, "_lumo_observer_buffer"):
            branch, residual = result
            logical_post_block = branch + residual
            observer_op(
                logical_post_block.reshape(-1, logical_post_block.shape[-1]),
                self._lumo_observer_buffer,
                self._lumo_observer_row_count,
            )
        return result

    decoder_class.__init__ = decoder_init
    decoder_class.forward = decoder_forward
    return {
        "installed_before_llm_construction": True,
        "custom_op": f"torch.ops.vllm.{OBSERVER_OP_NAME}.default",
        "custom_op_mutates": ["destination", "row_count"],
        "patched_linear_classes": sorted(patched_linear_classes),
        "patched_decoder_class": decoder_class.__name__,
        "capture_capacity": capacity,
        "scope": scope,
        "selected_linear_names": sorted(selected_linears),
        "selected_h_layers": sorted(selected_h_layers),
        "expected_names": sorted(
            selected_linears
            | {f"h{layer}_post_block" for layer in selected_h_layers}
        ),
        "insertion": "post-linear side effect and decoder-return side branch",
        "model_output_identity": "observer receives output; original result is returned",
    }


def _text_model(model: Any) -> Any:
    language_model = model.language_model if hasattr(model, "language_model") else model
    return language_model.model


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    cpu = tensor.detach().contiguous().cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    values = cpu.float()
    finite = torch.isfinite(values)
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": int(finite.sum().item()),
        "nonfinite": int((~finite).sum().item()),
        "min": float(values[finite].min().item()) if finite.any() else None,
        "max": float(values[finite].max().item()) if finite.any() else None,
        "rms": float(values[finite].square().mean().sqrt().item())
        if finite.any()
        else None,
    }


def _collect_observer(model: Any, *, expected_names: tuple[str, ...]) -> dict[str, Any]:
    import torch

    text_model = _text_model(model)
    records: dict[str, Any] = {}
    tensors: dict[str, Any] = {}
    for module in text_model.modules():
        name = getattr(module, "_lumo_observer_name", None)
        if name is None:
            continue
        raw_count = int(module._lumo_observer_row_count.item())
        capacity = int(module._lumo_observer_buffer.shape[0])
        copied = min(raw_count, capacity)
        tensor = module._lumo_observer_buffer[:copied].detach().cpu()
        records[name] = {
            "raw_row_count": raw_count,
            "copied_row_count": copied,
            "capacity": capacity,
            "truncated": raw_count > capacity,
            "buffer_device": str(module._lumo_observer_buffer.device),
            "buffer_persistent": "_lumo_observer_buffer" in module.state_dict(),
            "summary": _tensor_summary(tensor),
        }
        tensors[name] = tensor

    expected = set(expected_names)
    missing = sorted(expected - set(records))
    unexpected = sorted(set(records) - expected)
    zero_rows = sorted(name for name, value in records.items() if value["raw_row_count"] == 0)
    truncated = sorted(name for name, value in records.items() if value["truncated"])
    return {
        "records": records,
        "tensors": tensors,
        "expected_count": len(expected),
        "observed_count": len(records),
        "missing": missing,
        "unexpected": unexpected,
        "zero_rows": zero_rows,
        "truncated": truncated,
        "cuda_synchronized_before_collection": True,
        "torch_cuda_device": torch.cuda.current_device(),
    }


def _model_inventory(model: Any) -> dict[str, Any]:
    text_model = _text_model(model)
    language_model = model.language_model if hasattr(model, "language_model") else model
    config = language_model.vllm_config.compilation_config
    pass_config = config.pass_config
    kernel_config = language_model.vllm_config.kernel_config
    return {
        "root_class": type(model).__name__,
        "text_model_class": type(text_model).__name__,
        "layer_count": len(text_model.layers),
        "target_layer_types": {
            str(layer): text_model.layers[layer].layer_type for layer in TARGET_LAYERS
        },
        "compilation": {
            "mode": str(config.mode),
            "backend": str(config.backend),
            "cudagraph_mode": str(config.cudagraph_mode),
            "use_inductor_graph_partition": config.use_inductor_graph_partition,
            "custom_ops": list(config.custom_ops),
            "enabled_custom_ops": sorted(config.enabled_custom_ops),
            "disabled_custom_ops": sorted(config.disabled_custom_ops),
            "pass_config": {
                "fuse_norm_quant": pass_config.fuse_norm_quant,
                "fuse_act_quant": pass_config.fuse_act_quant,
                "fuse_attn_quant": pass_config.fuse_attn_quant,
            },
            "ir_op_priority": {
                "rms_norm": list(kernel_config.ir_op_priority.rms_norm),
                "fused_add_rms_norm": list(
                    kernel_config.ir_op_priority.fused_add_rms_norm
                ),
            },
        },
    }


def _generation_record(output: Any) -> dict[str, Any]:
    completion = output.outputs[0]
    candidates: list[dict[str, Any]] = []
    if completion.logprobs:
        for token_id, value in completion.logprobs[0].items():
            candidates.append(
                {
                    "token_id": int(token_id),
                    "logprob": float(value.logprob),
                    "rank": value.rank,
                    "decoded_token": value.decoded_token,
                }
            )
        candidates.sort(key=lambda item: (-item["logprob"], item["token_id"]))
    return {
        "prompt_token_ids": list(output.prompt_token_ids),
        "generated_token_id": int(completion.token_ids[0]),
        "generated_text": completion.text,
        "finish_reason": completion.finish_reason,
        "top_logprobs": candidates,
    }


def _compilation_diagnostics() -> dict[str, Any]:
    from vllm.compilation.counter import compilation_counter
    from vllm.compilation.passes.vllm_inductor_pass import get_match_table

    return {
        "counter": dataclasses.asdict(compilation_counter.clone()),
        "fusion_match_table": get_match_table(),
    }


def _scan_debug_dump(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False, "path": str(path) if path else None}
    needles = (
        f"vllm.{OBSERVER_OP_NAME}",
        "rms_norm_static_fp8_quant",
        "fused_add_rms_norm_static_fp8_quant",
        "static_scaled_fp8_quant",
        "vllm_ir.rms_norm",
        "vllm_ir.fused_add_rms_norm",
    )
    counts = {needle: 0 for needle in needles}
    scanned: list[str] = []
    pre_grad_files: list[Path] = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        if file_path.stat().st_size > 64 * 1024 * 1024:
            continue
        try:
            text = file_path.read_text(errors="strict")
        except (OSError, UnicodeError):
            continue
        scanned.append(str(file_path.relative_to(path)))
        if ".BEFORE_PRE_GRAD." in file_path.name and file_path.suffix == ".py":
            pre_grad_files.append(file_path)
        for needle in needles:
            counts[needle] += text.count(needle)

    pre_grad_node_needles = {
        "observer": f"torch.ops.vllm.{OBSERVER_OP_NAME}.default(",
        "cutlass_scaled_mm": "torch.ops._C.cutlass_scaled_mm(",
        "marlin_gemm": "torch.ops._C.marlin_gemm(",
        "rms_norm_ir": "torch.ops.vllm_ir.rms_norm.default(",
        "fused_add_rms_norm_ir": "torch.ops.vllm_ir.fused_add_rms_norm.default(",
        "rms_norm_static_fp8_quant": "torch.ops._C.rms_norm_static_fp8_quant.default(",
        "fused_add_rms_norm_static_fp8_quant": (
            "torch.ops._C.fused_add_rms_norm_static_fp8_quant.default("
        ),
    }
    pre_grad_node_counts = {
        name: sum(file_path.read_text().count(needle) for file_path in pre_grad_files)
        for name, needle in pre_grad_node_needles.items()
    }
    observer_nodes_by_file = {
        str(file_path.relative_to(path)): file_path.read_text().count(
            pre_grad_node_needles["observer"]
        )
        for file_path in pre_grad_files
        if pre_grad_node_needles["observer"] in file_path.read_text()
    }
    return {
        "available": True,
        "path": str(path),
        "text_files_scanned": len(scanned),
        "files": scanned,
        "string_occurrences": counts,
        "occurrences_are_diagnostic_not_unique_fx_nodes": True,
        "pre_grad_fx_file_count": len(pre_grad_files),
        "pre_grad_fx_node_occurrences": pre_grad_node_counts,
        "pre_grad_observer_nodes_by_file": observer_nodes_by_file,
    }


def _download_pinned_model_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)
    ).resolve()


def _metadata_identity(snapshot: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for filename, expected_sha256 in PINNED_METADATA_SHA256.items():
        metadata_path = snapshot / filename
        if not metadata_path.is_file():
            raise FileNotFoundError(f"required pinned metadata is missing: {metadata_path}")
        digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(
                f"pinned metadata SHA-256 mismatch for {filename}: "
                f"{digest} != {expected_sha256}"
            )
        actual[filename] = digest
    return actual


def _modelopt_checkpoint_api() -> tuple[Any, str, dict[str, str]]:
    try:
        from scripts.modelopt_checkpoint import (
            ModelOptCheckpoint,
            PINNED_METADATA_SHA256 as CHECKPOINT_METADATA_SHA256,
            PINNED_REVISION as CHECKPOINT_REVISION,
        )
    except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
        from modelopt_checkpoint import (  # type: ignore[no-redef]
            ModelOptCheckpoint,
            PINNED_METADATA_SHA256 as CHECKPOINT_METADATA_SHA256,
            PINNED_REVISION as CHECKPOINT_REVISION,
        )
    return ModelOptCheckpoint, CHECKPOINT_REVISION, dict(CHECKPOINT_METADATA_SHA256)


def _validate_pinned_checkpoint(snapshot: Path) -> dict[str, Any]:
    metadata_sha256 = _metadata_identity(snapshot)
    checkpoint_type, checkpoint_revision, checkpoint_metadata = (
        _modelopt_checkpoint_api()
    )
    if checkpoint_revision != MODEL_REVISION:
        raise RuntimeError("observer probe and ModelOpt checkpoint revisions diverged")
    if checkpoint_metadata != PINNED_METADATA_SHA256:
        raise RuntimeError("observer probe and ModelOpt metadata pins diverged")
    checkpoint_type(snapshot, strict_pinned=True)
    return {
        "policy": MODEL_IDENTITY_POLICY,
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "resolved_path": str(snapshot),
        "metadata_sha256": metadata_sha256,
        "strict_pinned_validation": True,
        "validator": "ModelOptCheckpoint(strict_pinned=True)",
    }


def _resolve_model_path(path: Path | None) -> tuple[Path, dict[str, Any]]:
    pinned_snapshot = _download_pinned_model_snapshot()
    candidate = pinned_snapshot if path is None else path.expanduser().resolve()
    if candidate != pinned_snapshot:
        raise ValueError(
            "--model-path must resolve to the exact pinned NVIDIA snapshot: "
            f"{candidate} != {pinned_snapshot}"
        )
    return candidate, _validate_pinned_checkpoint(candidate)


def _shutdown(llm: Any) -> None:
    import torch

    llm.llm_engine.engine_core.shutdown()
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()
    torch.cuda.empty_cache()


def _run_child(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode not in ("stock", "observer"):
        raise ValueError("child mode must be stock or observer")
    if args.capture_capacity < 1:
        raise ValueError("capture capacity must be positive")
    if args.max_num_batched_tokens < 1568:
        raise ValueError(
            "Qwen3.6 Mamba align mode requires max_num_batched_tokens >= 1568"
        )

    model_path, model_identity = _resolve_model_path(args.model_path)

    observer_install = None
    if args.mode == "observer":
        observer_install = _install_observer_before_model_construction(
            args.capture_capacity, scope=args.observer_scope
        )

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=MODEL_REVISION, local_files_only=True
    )
    prompt_token_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    if len(prompt_token_ids) > args.capture_capacity:
        raise ValueError(
            f"prompt has {len(prompt_token_ids)} tokens; observer capacity is "
            f"{args.capture_capacity}"
        )

    debug_dump = args.debug_dump_root.resolve() if args.debug_dump_root else None
    compilation_config: dict[str, Any] = {}
    if debug_dump is not None:
        compilation_config["debug_dump_path"] = str(debug_dump)

    started = datetime.now(timezone.utc)
    load_started = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        dtype="bfloat16",
        quantization="modelopt_fp4",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=False,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        language_model_only=True,
        gdn_prefill_backend="triton",
        mamba_cache_mode="align",
        mamba_block_size=1024,
        mamba_ssm_cache_dtype="float32",
        attention_backend="TRITON_ATTN",
        limit_mm_per_prompt={"image": 0, "video": 0},
        enable_flashinfer_autotune=False,
        async_scheduling=False,
        seed=0,
        compilation_config=compilation_config,
    )
    load_seconds = time.perf_counter() - load_started
    inventory = llm.apply_model(_model_inventory)[0]

    sampling = SamplingParams(max_tokens=1, temperature=0, seed=0, logprobs=20)
    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids, prompt=args.prompt)
    generation_started = time.perf_counter()
    output = llm.generate([prompt], sampling, use_tqdm=False)[0]
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started

    capture = None
    tensor_payload = None
    if args.mode == "observer":
        capture = llm.apply_model(
            functools.partial(
                _collect_observer,
                expected_names=tuple(observer_install["expected_names"]),
            )
        )[0]
        tensor_payload = capture.pop("tensors")
        if capture["missing"] or capture["zero_rows"] or capture["truncated"]:
            status = "incomplete"
        else:
            status = "observed"
    else:
        status = "stock"

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "mode": args.mode,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "cuda_device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "model": {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "resolved_path": str(model_path),
            "identity": model_identity,
            "load_seconds": load_seconds,
            "inventory": inventory,
        },
        "runtime": {
            "compiled": True,
            "enforce_eager": False,
            "cold_compile": os.environ.get("VLLM_DISABLE_COMPILE_CACHE") == "1",
            "mtp_enabled": False,
            "language_model_only": True,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "gdn_prefill_backend": "triton",
            "attention_backend": "TRITON_ATTN",
            "disabled_kernels": os.environ.get("VLLM_DISABLED_KERNELS", ""),
            "observer_scope": args.observer_scope,
        },
        "prompt": {
            "text": args.prompt,
            "token_ids": prompt_token_ids,
            "token_count": len(prompt_token_ids),
        },
        "generation": _generation_record(output),
        "generation_seconds": generation_seconds,
        "observer_install": observer_install,
        "observer_capture": capture,
        "compilation_diagnostics": _compilation_diagnostics(),
        "debug_dump": _scan_debug_dump(debug_dump),
    }

    if args.tensor_output is not None and tensor_payload is not None:
        args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "model_revision": MODEL_REVISION,
                "model_identity": model_identity,
                "prompt_token_ids": prompt_token_ids,
                "tensors": tensor_payload,
            },
            args.tensor_output,
        )
        result["tensor_output"] = str(args.tensor_output)

    _shutdown(llm)
    return result


def _endpoint_parity(stock: dict[str, Any], observer: dict[str, Any]) -> dict[str, Any]:
    left = stock["generation"]
    right = observer["generation"]
    left_scores = {item["token_id"]: item["logprob"] for item in left["top_logprobs"]}
    right_scores = {
        item["token_id"]: item["logprob"] for item in right["top_logprobs"]
    }
    shared = sorted(set(left_scores) & set(right_scores))
    deltas = {
        str(token): right_scores[token] - left_scores[token] for token in shared
    }
    exact_top_logprobs = left["top_logprobs"] == right["top_logprobs"]
    exact = (
        left["prompt_token_ids"] == right["prompt_token_ids"]
        and left["generated_token_id"] == right["generated_token_id"]
        and left["generated_text"] == right["generated_text"]
        and exact_top_logprobs
    )
    return {
        "exact": exact,
        "prompt_token_ids_exact": left["prompt_token_ids"]
        == right["prompt_token_ids"],
        "generated_token_exact": left["generated_token_id"]
        == right["generated_token_id"],
        "generated_text_exact": left["generated_text"] == right["generated_text"],
        "top_logprobs_exact": exact_top_logprobs,
        "stock_top_logprob_count": len(left["top_logprobs"]),
        "observer_top_logprob_count": len(right["top_logprobs"]),
        "shared_top_logprob_tokens": len(shared),
        "max_abs_shared_logprob_delta": max(
            (abs(value) for value in deltas.values()), default=None
        ),
        "logprob_deltas": deltas,
    }


def _child_command(
    args: argparse.Namespace,
    *,
    mode: str,
    output: Path,
    tensor_output: Path | None,
    debug_dump: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--prompt",
        args.prompt,
        "--output",
        str(output),
        "--debug-dump-root",
        str(debug_dump),
        "--capture-capacity",
        str(args.capture_capacity),
        "--observer-scope",
        args.observer_scope,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]
    if args.model_path is not None:
        command.extend(("--model-path", str(args.model_path)))
    if tensor_output is not None:
        command.extend(("--tensor-output", str(tensor_output)))
    return command


def _run_both(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="compiled-observer-") as temp:
        temporary = Path(temp)
        stock_json = temporary / "stock.json"
        observer_json = temporary / "observer.json"
        dump_root = (
            args.debug_dump_root.resolve()
            if args.debug_dump_root is not None
            else temporary / "debug"
        )
        stock_dump = dump_root / "stock"
        observer_dump = dump_root / "observer"
        observer_tensors = (
            args.tensor_output.resolve() if args.tensor_output is not None else None
        )

        for mode, output, tensor_output, debug_dump in (
            ("stock", stock_json, None, stock_dump),
            ("observer", observer_json, observer_tensors, observer_dump),
        ):
            subprocess.run(
                _child_command(
                    args,
                    mode=mode,
                    output=output,
                    tensor_output=tensor_output,
                    debug_dump=debug_dump,
                ),
                check=True,
                env=os.environ.copy(),
            )

        stock = json.loads(stock_json.read_text())
        observer = json.loads(observer_json.read_text())
        parity = _endpoint_parity(stock, observer)
        capture_complete = observer["status"] == "observed"
        stock_nodes = stock["debug_dump"].get("pre_grad_fx_node_occurrences", {})
        observer_nodes = observer["debug_dump"].get(
            "pre_grad_fx_node_occurrences", {}
        )
        expected_observer_nodes = observer["observer_capture"]["expected_count"]
        actual_observer_nodes = observer_nodes.get("observer", 0)
        non_observer_names = sorted((set(stock_nodes) | set(observer_nodes)) - {"observer"})
        non_observer_node_counts_equal = all(
            stock_nodes.get(name) == observer_nodes.get(name)
            for name in non_observer_names
        )
        compile_visible = actual_observer_nodes == expected_observer_nodes
        fusion = {
            "stock": stock["compilation_diagnostics"]["fusion_match_table"],
            "observer": observer["compilation_diagnostics"]["fusion_match_table"],
        }
        fusion["same_match_table"] = fusion["stock"] == fusion["observer"]
        status = (
            "proved"
            if parity["exact"] and capture_complete and compile_visible
            else "failed"
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "mode": "both",
            "stock": stock,
            "observer": observer,
            "compiled_endpoint_parity": parity,
            "compiled_observer_visibility": {
                "proved": compile_visible,
                "expected_pre_grad_fx_nodes": expected_observer_nodes,
                "actual_pre_grad_fx_nodes": actual_observer_nodes,
                "nodes_by_file": observer["debug_dump"].get(
                    "pre_grad_observer_nodes_by_file", {}
                ),
                "stock_non_observer_node_counts": {
                    name: stock_nodes.get(name) for name in non_observer_names
                },
                "observer_non_observer_node_counts": {
                    name: observer_nodes.get(name) for name in non_observer_names
                },
                "non_observer_node_counts_equal": non_observer_node_counts_equal,
            },
            "fusion_comparison": fusion,
            "claim": (
                "The observer is compile-visible and endpoint-nonintrusive for this "
                "pinned prompt/configuration. This is not a proof for every input shape."
                if status == "proved"
                else "The observer experiment did not satisfy exact endpoint and capture checks."
            ),
        }


def main() -> None:
    args = _parse_args()
    _prepare_process_environment()
    result = _run_both(args) if args.mode == "both" else _run_child(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
