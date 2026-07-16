#!/usr/bin/env python3
"""Apply the pinned Qwen3.6 Jacobian lens to the NVIDIA NVFP4 model."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from download_jlens import (
    LENS_FILENAME,
    LENS_REPO,
    LENS_REVISION,
    verify_checkpoint,
    verify_file,
    verify_local_fit_artifact,
)

MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
MODEL_CONFIG_SHA256 = "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338"
MODEL_INDEX_SHA256 = "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2"
DEFAULT_PROMPT = "Fact: The currency used in the country shaped like a boot is"
SOURCE_LAYERS = tuple(range(63))
CAPTURE_LAYERS = tuple(range(64))
SCHEMA_VERSION = 2
FINAL_NORM_MAX_ABS_TOLERANCE = 0.125
FINAL_NORM_RMS_TOLERANCE = 0.006
FINAL_LOGIT_MAX_ABS_TOLERANCE = 0.0625
FINAL_LOGIT_RMS_TOLERANCE = 0.01
FINAL_TOPK_PARITY_K = 5
MAX_CAPTURE_POSITIONS = 8


def parse_integer_list(value: str, *, allow_all: bool = False) -> list[int]:
    if allow_all and value.strip().lower() == "all":
        return list(SOURCE_LAYERS)
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value!r}") from exc
    if not result:
        raise argparse.ArgumentTypeError("integer list must not be empty")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("integer list contains duplicates")
    return result


def validate_layers(layers: list[int]) -> list[int]:
    invalid = [layer for layer in layers if layer not in SOURCE_LAYERS]
    if invalid:
        raise ValueError(f"lens layers must be in 0..62; got {invalid}")
    return sorted(layers)


def resolve_positions(positions: list[int], token_count: int) -> list[int]:
    resolved = [position + token_count if position < 0 else position for position in positions]
    invalid = [position for position in resolved if not 0 <= position < token_count]
    if invalid:
        raise ValueError(
            f"positions resolve outside a {token_count}-token prompt: {invalid}"
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError("positions resolve to duplicates")
    return resolved


def target_token_ids_for_positions(
    prompt_token_ids: list[int],
    resolved_positions: list[int],
    generated_token_id: int,
) -> tuple[int, ...]:
    final_position = len(prompt_token_ids) - 1
    if final_position not in resolved_positions:
        raise ValueError("resolved positions must include the prompt's final position")
    return tuple(
        generated_token_id
        if position == final_position
        else prompt_token_ids[position + 1]
        for position in resolved_positions
    )


def capture_positions_with_final(
    requested_positions: list[int], token_count: int
) -> list[int]:
    if len(requested_positions) > MAX_CAPTURE_POSITIONS:
        raise ValueError(
            f"at most {MAX_CAPTURE_POSITIONS} positions may be decoded per run"
        )
    result = list(requested_positions)
    final_position = token_count - 1
    if final_position not in result:
        result.append(final_position)
    return result


def reconstruct_post_block(block_output: tuple[Any, Any]) -> Any:
    branch_output, residual = block_output
    return branch_output + residual


def transport_residual(residual: Any, jacobian: Any) -> Any:
    return residual @ jacobian.T


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_model_parts(model: Any) -> tuple[Any, Any]:
    language_model = model.language_model if hasattr(model, "language_model") else model
    text_model = language_model.model
    if not hasattr(text_model, "layers") or not hasattr(text_model, "norm"):
        raise TypeError(f"unsupported vLLM model layout: {type(model).__name__}")
    return language_model, text_model


def _install_capture_hooks(model: Any) -> dict[str, object]:
    import torch

    # apply_model callbacks run outside vLLM's normal inference context.
    torch.set_grad_enabled(False)
    _, text_model = _text_model_parts(model)
    if len(text_model.layers) != 64:
        raise ValueError(f"expected 64 main layers, found {len(text_model.layers)}")

    model._jlens_captures = {}
    model._jlens_positions = ()
    model._jlens_handles = []
    model._jlens_capture_active = False
    model._jlens_final_normalized = None

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not model._jlens_positions:
                return
            branch_output, residual = output
            post_block = reconstruct_post_block((branch_output, residual))
            max_position = max(model._jlens_positions)
            if max_position >= post_block.shape[0]:
                raise RuntimeError(
                    f"capture position {max_position} exceeds forward rows "
                    f"{post_block.shape[0]}"
                )
            model._jlens_captures[layer] = (
                post_block[list(model._jlens_positions)].detach().float().cpu()
            )

        return hook

    for layer in CAPTURE_LAYERS:
        handle = text_model.layers[layer].register_forward_hook(make_hook(layer))
        model._jlens_handles.append(handle)

    def final_norm_hook(_module: Any, _inputs: Any, output: Any) -> None:
        if not model._jlens_capture_active:
            return
        normalized = output if torch.is_tensor(output) else output[0]
        model._jlens_final_normalized = (
            normalized[list(model._jlens_positions)].detach().float().cpu()
        )

    model._jlens_handles.append(text_model.norm.register_forward_hook(final_norm_hook))

    language_model, _ = _text_model_parts(model)
    return {
        "root_class": type(model).__name__,
        "language_model_class": type(language_model).__name__,
        "layer_count": len(text_model.layers),
        "hidden_size": text_model.config.hidden_size,
        "final_norm_class": type(text_model.norm).__name__,
        "lm_head_class": type(language_model.lm_head).__name__,
    }


def _prepare_capture(model: Any, *, positions: tuple[int, ...]) -> None:
    model._jlens_captures = {}
    model._jlens_positions = positions
    model._jlens_final_normalized = None
    model._jlens_capture_active = True


def _freeze_capture(model: Any) -> None:
    model._jlens_capture_active = False
    if model._jlens_final_normalized is None:
        raise RuntimeError("final norm hook did not capture the model forward")


def _remove_capture_hooks(model: Any) -> None:
    for handle in getattr(model, "_jlens_handles", []):
        handle.remove()
    model._jlens_handles = []


def _compact_topk(
    logits: Any,
    *,
    top_k: int,
    target_token_id: int,
) -> dict[str, object]:
    import torch

    logits = logits.detach().float()
    values, token_ids = torch.topk(logits, top_k)
    target_score = logits[target_token_id]
    target_rank = int((logits > target_score).sum().item()) + 1
    return {
        "token_ids": token_ids.cpu().tolist(),
        "scores": [round(float(value), 6) for value in values.cpu().tolist()],
        "target_token_id": target_token_id,
        "target_score": round(float(target_score), 6),
        "target_rank": target_rank,
    }


def _readout_captures(
    model: Any,
    *,
    lens_path: str,
    layers: tuple[int, ...],
    top_k: int,
    target_token_ids: tuple[int, ...],
) -> dict[str, object]:
    import torch

    language_model, text_model = _text_model_parts(model)
    captures = model._jlens_captures
    missing = sorted(set(CAPTURE_LAYERS) - set(captures))
    if missing:
        raise RuntimeError(f"hooks did not capture layers: {missing}")

    started = time.perf_counter()
    checkpoint = torch.load(
        lens_path, map_location="cpu", weights_only=True, mmap=True
    )
    device = next(text_model.norm.parameters()).device
    labels: list[tuple[str, int, int]] = []
    vectors: list[Any] = []

    for layer in layers:
        raw = captures[layer].to(device=device, dtype=torch.float32)
        vectors.extend(raw.unbind(0))
        labels.extend(("logit", layer, position) for position in range(raw.shape[0]))

        jacobian = checkpoint["J"][layer].to(device=device, dtype=torch.float32)
        transported = transport_residual(raw, jacobian)
        if not bool(torch.isfinite(transported).all()):
            raise RuntimeError(f"non-finite transported residual at layer {layer}")
        vectors.extend(transported.unbind(0))
        labels.extend(("jacobian", layer, position) for position in range(raw.shape[0]))
        del jacobian, transported

    final_residuals = captures[63].to(device=device, dtype=torch.float32)
    vectors.extend(final_residuals.unbind(0))
    labels.extend(("final", 63, position) for position in range(final_residuals.shape[0]))

    stacked = torch.stack(vectors).to(dtype=torch.bfloat16)
    normalized = text_model.norm(stacked)
    captured_final_normalized = model._jlens_final_normalized.to(
        device=device, dtype=torch.bfloat16
    )
    logits = language_model.compute_logits(
        torch.cat((normalized, captured_final_normalized), dim=0)
    )
    if logits is None:
        raise RuntimeError("vLLM compute_logits returned None")

    final_start = len(labels) - final_residuals.shape[0]
    reconstructed_final_normalized = normalized[final_start:].detach().float().cpu()
    captured_final_normalized_cpu = model._jlens_final_normalized
    final_norm_difference = (
        reconstructed_final_normalized - captured_final_normalized_cpu
    )
    final_norm_max_abs_error = float(final_norm_difference.abs().max())
    final_norm_rms_error = float(final_norm_difference.square().mean().sqrt())
    final_norm_reference_rms = float(
        captured_final_normalized_cpu.square().mean().sqrt()
    )
    final_norm_relative_rms_error = final_norm_rms_error / final_norm_reference_rms
    final_norm_within_tolerance = (
        final_norm_max_abs_error <= FINAL_NORM_MAX_ABS_TOLERANCE
        and final_norm_rms_error <= FINAL_NORM_RMS_TOLERANCE
    )

    positions = captures[63].shape[0]
    if len(target_token_ids) != positions:
        raise ValueError("target token count does not match captured positions")

    records: dict[tuple[str, int, int], dict[str, object]] = {}
    for index, label in enumerate(labels):
        records[label] = _compact_topk(
            logits[index],
            top_k=top_k,
            target_token_id=target_token_ids[label[2]],
        )
    captured_records = [
        _compact_topk(
            logits[len(labels) + position],
            top_k=top_k,
            target_token_id=target_token_ids[position],
        )
        for position in range(positions)
    ]
    reconstructed_final_logits = logits[final_start : final_start + positions].float()
    captured_final_logits = logits[len(labels) : len(labels) + positions].float()
    final_logit_difference = reconstructed_final_logits - captured_final_logits
    final_logit_max_abs_error = float(final_logit_difference.abs().max())
    final_logit_rms_error = float(final_logit_difference.square().mean().sqrt())
    topk_parity_k = min(FINAL_TOPK_PARITY_K, top_k)
    final_topk_prefix_ids_match = all(
        records[("final", 63, position)]["token_ids"][:topk_parity_k]
        == captured_records[position]["token_ids"][:topk_parity_k]
        for position in range(positions)
    )
    final_logits_within_tolerance = (
        final_logit_max_abs_error <= FINAL_LOGIT_MAX_ABS_TOLERANCE
        and final_logit_rms_error <= FINAL_LOGIT_RMS_TOLERANCE
        and final_topk_prefix_ids_match
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    per_layer: list[dict[str, object]] = []
    for layer in layers:
        per_layer.append(
            {
                "layer": layer,
                "layer_type": text_model.config.layer_types[layer],
                "positions": [
                    {
                        "capture_index": position,
                        "token_position": model._jlens_positions[position],
                        "logit_lens": records[("logit", layer, position)],
                        "jacobian_lens": records[("jacobian", layer, position)],
                    }
                    for position in range(positions)
                ],
            }
        )

    return {
        "layers": per_layer,
        "final_model_readout": [
            records[("final", 63, position)] for position in range(positions)
        ],
        "captured_final_model_readout": captured_records,
        "final_norm_reconstruction": {
            "max_abs_error": round(final_norm_max_abs_error, 9),
            "rms_error": round(final_norm_rms_error, 9),
            "reference_rms": round(final_norm_reference_rms, 9),
            "relative_rms_error": round(final_norm_relative_rms_error, 9),
            "max_abs_tolerance": FINAL_NORM_MAX_ABS_TOLERANCE,
            "rms_tolerance": FINAL_NORM_RMS_TOLERANCE,
            "within_tolerance": final_norm_within_tolerance,
        },
        "final_logits_reconstruction": {
            "max_abs_error": round(final_logit_max_abs_error, 9),
            "max_abs_tolerance": FINAL_LOGIT_MAX_ABS_TOLERANCE,
            "rms_error": round(final_logit_rms_error, 9),
            "rms_tolerance": FINAL_LOGIT_RMS_TOLERANCE,
            "top_k_prefix": topk_parity_k,
            "top_k_prefix_token_ids_match": final_topk_prefix_ids_match,
            "within_tolerance": final_logits_within_tolerance,
        },
        "readout_seconds": round(elapsed, 6),
        "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _decode_topk(tokenizer: Any, compact: dict[str, object]) -> dict[str, object]:
    token_ids = compact["token_ids"]
    return {
        **compact,
        "tokens": [tokenizer.decode([token_id]) for token_id in token_ids],
        "target_token": tokenizer.decode([compact["target_token_id"]]),
    }


def _decorate_readout(tokenizer: Any, readout: dict[str, object]) -> None:
    for layer in readout["layers"]:
        for position in layer["positions"]:
            position["logit_lens"] = _decode_topk(tokenizer, position["logit_lens"])
            position["jacobian_lens"] = _decode_topk(
                tokenizer, position["jacobian_lens"]
            )
    readout["final_model_readout"] = [
        _decode_topk(tokenizer, result)
        for result in readout["final_model_readout"]
    ]
    readout["captured_final_model_readout"] = [
        _decode_topk(tokenizer, result)
        for result in readout["captured_final_model_readout"]
    ]


def _load_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompt:
        return [{"id": "cli", "text": args.prompt}]
    if args.prompts_file:
        data = json.loads(args.prompts_file.read_text())
        if not isinstance(data, list) or not data:
            raise ValueError("prompts file must contain a nonempty JSON list")
        prompts = []
        for index, item in enumerate(data):
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError(f"invalid prompt entry at index {index}")
            prompts.append({"id": str(item.get("id", index)), "text": item["text"]})
        return prompts
    return [{"id": "currency_boot", "text": DEFAULT_PROMPT}]


def lens_artifact_mode(args: argparse.Namespace) -> str:
    """Classify lens CLI arguments without inspecting artifact bytes."""
    has_path = args.lens_path is not None
    has_sha256 = args.lens_sha256 is not None
    has_provenance = args.lens_provenance is not None
    if has_sha256 != has_provenance:
        raise ValueError(
            "local lenses require both --lens-sha256 and --lens-provenance"
        )
    if (has_sha256 or has_provenance) and not has_path:
        raise ValueError("local lens metadata requires --lens-path")
    if has_path and has_sha256 and has_provenance:
        return "local_fit"
    return "public"


def _package_versions() -> dict[str, str]:
    packages = ["vllm", "torch", "transformers", "huggingface-hub", "triton"]
    return {name: importlib.metadata.version(name) for name in packages}


def _nvidia_smi() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    values = subprocess.check_output(command, text=True).strip().split(", ")
    return dict(
        zip(
            ["name", "driver_version", "memory_total_mib", "memory_used_mib", "compute_capability"],
            values,
            strict=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="single raw completion prompt")
    parser.add_argument("--prompts-file", type=Path, help="JSON list of prompt objects")
    parser.add_argument("--layers", default="all", help="comma list in 0..62, or all")
    parser.add_argument("--positions", default="-1", help="comma list of token positions")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--lens-path",
        type=Path,
        help="pinned public artifact override, or a local fit with both metadata flags",
    )
    parser.add_argument(
        "--lens-sha256",
        help="required expected SHA-256 when --lens-path is a local fitted lens",
    )
    parser.add_argument(
        "--lens-provenance",
        type=Path,
        help="required completed fit provenance when --lens-path is a local fitted lens",
    )
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.prompt and args.prompts_file:
        raise ValueError("pass at most one of --prompt and --prompts-file")
    if not 1 <= args.top_k <= 100:
        raise ValueError("--top-k must be in 1..100")
    if not 0.70 <= args.gpu_memory_utilization <= 0.90:
        raise ValueError("--gpu-memory-utilization must be in 0.70..0.90")
    lens_mode = lens_artifact_mode(args)

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    layers = validate_layers(parse_integer_list(args.layers, allow_all=True))
    positions = parse_integer_list(args.positions)
    prompts = _load_prompts(args)

    from huggingface_hub import hf_hub_download, snapshot_download

    if lens_mode == "local_fit":
        lens_path = args.lens_path.resolve()
        lens_record = verify_local_fit_artifact(
            lens_path,
            expected_sha256=args.lens_sha256,
            provenance_path=args.lens_provenance.resolve(),
            check_finite=True,
        )
        lens_application = (
            f"{lens_record['fit_quantization']} fitted lens applied to "
            "pinned NVIDIA NVFP4/FP8 residuals"
        )
    else:
        if args.lens_path:
            lens_path = args.lens_path.resolve()
        else:
            lens_path = Path(
                hf_hub_download(
                    LENS_REPO,
                    LENS_FILENAME,
                    revision=LENS_REVISION,
                    local_files_only=True,
                )
            )
        lens_record = {
            "repo_id": LENS_REPO,
            "revision": LENS_REVISION,
            "filename": LENS_FILENAME,
            **verify_file(lens_path),
            **verify_checkpoint(lens_path, check_finite=False),
        }
        lens_application = "BF16-fitted lens applied to NVFP4/FP8 residuals"

    model_path = Path(
        snapshot_download(
            MODEL_REPO,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    model_config_path = model_path / "config.json"
    model_index_path = model_path / "model.safetensors.index.json"
    model_config_sha256 = sha256_file(model_config_path)
    model_index_sha256 = sha256_file(model_index_path)
    if model_config_sha256 != MODEL_CONFIG_SHA256:
        raise ValueError(
            f"model config SHA-256 mismatch: expected {MODEL_CONFIG_SHA256}, "
            f"got {model_config_sha256}"
        )
    if model_index_sha256 != MODEL_INDEX_SHA256:
        raise ValueError(
            f"model index SHA-256 mismatch: expected {MODEL_INDEX_SHA256}, "
            f"got {model_index_sha256}"
        )
    model_config = json.loads(model_config_path.read_text())
    text_config = model_config["text_config"]
    if text_config["hidden_size"] != 5120 or text_config["num_hidden_layers"] != 64:
        raise ValueError("model is not the expected 64-layer, width-5120 architecture")
    if text_config["layer_types"].count("linear_attention") != 48:
        raise ValueError("model does not have 48 GDN layers")
    if text_config["layer_types"].count("full_attention") != 16:
        raise ValueError("model does not have 16 full-attention layers")

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError("run through scripts/run_jlens_nvfp4.sh")

    import torch
    from vllm import LLM, SamplingParams, TokensPrompt

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        dtype="bfloat16",
        quantization="modelopt_fp4",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        language_model_only=True,
        gdn_prefill_backend="triton",
        mamba_cache_mode="align",
        mamba_block_size=args.max_model_len,
        mamba_ssm_cache_dtype="float32",
        attention_backend="TRITON_ATTN",
        limit_mm_per_prompt={"image": 0, "video": 0},
        enable_flashinfer_autotune=False,
        async_scheduling=False,
        seed=0,
    )
    model_load_seconds = time.perf_counter() - load_started
    model_info = llm.apply_model(_install_capture_hooks)[0]
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(max_tokens=1, temperature=0, seed=0)

    experiment_results = []
    all_final_top1_matches = True
    all_final_norm_matches = True
    for prompt_spec in prompts:
        token_ids = tokenizer.encode(prompt_spec["text"], add_special_tokens=True)
        if len(token_ids) + 1 > args.max_model_len:
            raise ValueError(
                f"prompt {prompt_spec['id']} has {len(token_ids)} tokens, "
                f"which leaves no generation slot under max {args.max_model_len}"
            )
        resolved = resolve_positions(positions, len(token_ids))
        final_token_position = len(token_ids) - 1
        capture_positions = capture_positions_with_final(resolved, len(token_ids))
        llm.apply_model(
            functools.partial(_prepare_capture, positions=tuple(capture_positions))
        )

        prompt_started = time.perf_counter()
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=token_ids, prompt=prompt_spec["text"])],
            sampling,
            use_tqdm=False,
        )
        generation_seconds = time.perf_counter() - prompt_started
        output = outputs[0]
        if output.prompt_token_ids != token_ids:
            raise RuntimeError("vLLM prompt token IDs differ from the frozen input")
        generated_token_id = output.outputs[0].token_ids[0]
        llm.apply_model(_freeze_capture)
        target_token_ids = target_token_ids_for_positions(
            token_ids, capture_positions, generated_token_id
        )

        readout = llm.apply_model(
            functools.partial(
                _readout_captures,
                lens_path=str(lens_path),
                layers=tuple(layers),
                top_k=args.top_k,
                target_token_ids=target_token_ids,
            )
        )[0]
        _decorate_readout(tokenizer, readout)
        final_position_index = capture_positions.index(final_token_position)
        final_matches = (
            readout["final_model_readout"][final_position_index]["token_ids"][0]
            == generated_token_id
            and readout["captured_final_model_readout"][final_position_index][
                "token_ids"
            ][0]
            == generated_token_id
        )
        final_norm_matches = (
            readout["final_norm_reconstruction"]["within_tolerance"]
            and readout["final_logits_reconstruction"]["within_tolerance"]
        )
        all_final_top1_matches = all_final_top1_matches and final_matches
        all_final_norm_matches = all_final_norm_matches and final_norm_matches
        for layer_result in readout["layers"]:
            layer_result["positions"] = [
                position
                for position in layer_result["positions"]
                if position["token_position"] in resolved
            ]

        experiment_results.append(
            {
                "id": prompt_spec["id"],
                "prompt": prompt_spec["text"],
                "prompt_token_ids": token_ids,
                "prompt_tokens": [tokenizer.decode([token_id]) for token_id in token_ids],
                "positions_requested": positions,
                "positions_resolved": resolved,
                "capture_positions_resolved": capture_positions,
                "final_validation_position": final_token_position,
                "position_tokens": [tokenizer.decode([token_ids[pos]]) for pos in resolved],
                "generated_token_id": generated_token_id,
                "generated_token": tokenizer.decode([generated_token_id]),
                "generated_text": output.outputs[0].text,
                "generation_seconds": round(generation_seconds, 6),
                "final_layer_top1_matches_greedy": final_matches,
                **readout,
            }
        )

    llm.apply_model(_remove_capture_hooks)
    runtime_gpu = _nvidia_smi()
    completed_at = datetime.now(timezone.utc)
    all_validations_passed = all_final_top1_matches and all_final_norm_matches
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all_validations_passed else "failed",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": runtime_gpu,
            "packages": _package_versions(),
        },
        "model": {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "resolved_path": str(model_path),
            "config_sha256": model_config_sha256,
            "index_sha256": model_index_sha256,
            "quant_method": model_config["quantization_config"]["quant_method"],
            "quant_algo": model_config["quantization_config"]["quant_algo"],
            "model_info": model_info,
        },
        "lens": {
            **lens_record,
            "application": lens_application,
        },
        "runtime": {
            "mtp_enabled": False,
            "enforce_eager": True,
            "language_model_only": True,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "capture_adapter": "vLLM apply_model forward hooks",
            "transport_dtype": "torch.float32",
            "readout_dtype": "torch.bfloat16",
            "model_load_seconds": round(model_load_seconds, 6),
            "timing_scope": "artifact resolution and validation through readout",
        },
        "assertions": {
            "lens_hash_matches": True,
            "lens_metadata_matches": True,
            "model_architecture_matches": True,
            "all_final_layer_top1_match_greedy": all_final_top1_matches,
            "all_final_adapter_reconstructions_within_tolerance": all_final_norm_matches,
        },
        "experiments": experiment_results,
    }

    # In single-process mode the caller owns both the engine and process group.
    llm.llm_engine.engine_core.shutdown()
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()
    torch.cuda.empty_cache()

    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"wrote {args.output}", file=sys.stderr)
    print(rendered, end="")
    return 0 if all_validations_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
