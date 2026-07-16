#!/usr/bin/env python3
"""Evaluate local and public Jacobian lenses on the pinned NF4 Qwen model.

The command is offline and fail-closed. It performs held-out readout evaluation,
not fitting, and applies the model final norm exactly once per readout.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Sequence

from compare_jlens_artifacts import verify_artifacts
from download_jlens import LENS_FILENAME, LENS_REPO, LENS_REVISION
import fit_jlens_nf4 as fitter


SCHEMA_VERSION = 1
PROMPTS_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "jlens_nf4_eval_prompts.json"
)
PROMPTS_SHA256 = "cd0fe64e800c7b937fcd891196eed6d7c30a8ff1246b9555dc2962bf61c9a56b"
SOURCE_LAYERS = tuple(range(63))
MAX_POSITIONS = 8
MIN_POSITION = 16
PROMPT_TOKEN_COUNT = 128
MIN_PROMPTS = 4
DATASET_REPO = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONFIG = "wikitext-103-raw-v1"


def parse_integer_list(value: str, *, allow_all: bool = False) -> list[int]:
    if allow_all and value.strip().lower() == "all":
        return list(SOURCE_LAYERS)
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value!r}") from exc
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("integer list must be nonempty and unique")
    return values


def validate_layers(layers: Sequence[int]) -> list[int]:
    invalid = [layer for layer in layers if layer not in SOURCE_LAYERS]
    if invalid:
        raise ValueError(f"layers must be in 0..62; got {invalid}")
    return sorted(layers)


def resolve_positions(positions: Sequence[int], token_count: int) -> list[int]:
    resolved = [position + token_count if position < 0 else position for position in positions]
    if len(resolved) > MAX_POSITIONS:
        raise ValueError(f"at most {MAX_POSITIONS} positions may be evaluated")
    if len(resolved) != len(set(resolved)):
        raise ValueError("positions resolve to duplicates")
    invalid = [
        position
        for position in resolved
        if not MIN_POSITION <= position < token_count - 1
    ]
    if invalid:
        raise ValueError(
            f"positions must resolve to {MIN_POSITION}..{token_count - 2} "
            f"for teacher-forced evaluation; got {invalid}"
        )
    return resolved


def load_frozen_prompts(
    path: Path,
    tokenizer: Any,
    *,
    expected_sha256: str = PROMPTS_SHA256,
) -> tuple[list[dict[str, Any]], str]:
    fitter.require_regular_file(path, label="held-out prompt file")
    digest = fitter.sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"held-out prompt SHA-256 mismatch: {digest}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("held-out prompt manifest must use schema_version 1")
    dataset = raw.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("held-out dataset provenance does not match the pinned corpus")
    expected_dataset = {
        "repo": DATASET_REPO,
        "revision": DATASET_REVISION,
        "config": DATASET_CONFIG,
        "split": dataset.get("split"),
    }
    if dataset != expected_dataset:
        raise ValueError("held-out dataset provenance does not match the pinned corpus")
    if dataset["split"] not in {"validation", "test"}:
        raise ValueError("held-out prompts must come from validation or test")
    expected_tokenizer = {
        "repo": fitter.MODEL_ID,
        "revision": fitter.MODEL_REVISION,
        "add_special_tokens": True,
        "force_bos_when_supported": True,
        "truncation": "right",
    }
    if raw.get("tokenizer") != expected_tokenizer:
        raise ValueError("held-out tokenizer provenance does not match the pinned model")
    selection = raw.get("selection")
    expected_selection = {
        "order": "dataset row order",
        "minimum_stripped_characters": 600,
        "required_token_count": PROMPT_TOKEN_COUNT,
        "take": MIN_PROMPTS,
    }
    if selection != expected_selection:
        raise ValueError("held-out selection must freeze exactly four 128-token prompts")
    entries = raw.get("prompts")
    if not isinstance(entries, list) or len(entries) != MIN_PROMPTS:
        raise ValueError("held-out manifest must contain exactly four prompts")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_rows: set[int] = set()
    for index, item in enumerate(entries):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError(f"invalid held-out prompt at index {index}")
        row_index = item.get("row_index")
        if (
            not isinstance(row_index, int)
            or isinstance(row_index, bool)
            or row_index < 0
            or row_index in seen_rows
        ):
            raise ValueError(f"invalid or duplicate dataset row at index {index}")
        seen_rows.add(row_index)
        prompt_id = item.get("id", f"{dataset['split']}-row-{row_index}")
        text = item["text"]
        if (
            not isinstance(prompt_id, str)
            or not prompt_id
            or prompt_id in seen
            or not text
        ):
            raise ValueError(f"invalid or duplicate prompt id at index {index}")
        seen.add(prompt_id)
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if item.get("text_sha256") != text_sha256:
            raise ValueError(f"prompt {prompt_id} text SHA-256 mismatch")
        raw_text = item.get("raw_text")
        if raw_text is not None:
            if not isinstance(raw_text, str) or raw_text.strip() != text:
                raise ValueError(f"prompt {prompt_id} raw text transform mismatch")
            if item.get("raw_text_sha256") != hashlib.sha256(
                raw_text.encode("utf-8")
            ).hexdigest():
                raise ValueError(f"prompt {prompt_id} raw text SHA-256 mismatch")
        token_ids = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=PROMPT_TOKEN_COUNT,
            return_attention_mask=False,
        )["input_ids"]
        declared_ids = item.get("token_ids")
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != PROMPT_TOKEN_COUNT
            or item.get("token_count") != PROMPT_TOKEN_COUNT
            or not isinstance(declared_ids, list)
            or len(declared_ids) != PROMPT_TOKEN_COUNT
            or any(
                not isinstance(token, int) or isinstance(token, bool)
                for token in declared_ids
            )
            or [int(token) for token in token_ids] != declared_ids
        ):
            raise ValueError(f"prompt {prompt_id} frozen token IDs do not match")
        records.append(
            {
                "id": prompt_id,
                "text": text,
                "text_sha256": text_sha256,
                "dataset_split": dataset["split"],
                "dataset_row_index": row_index,
                "token_ids": declared_ids,
            }
        )
    return records, digest


class PostBlockCapture(AbstractContextManager["PostBlockCapture"]):
    def __init__(self, blocks: Sequence[Any], layers: Sequence[int], positions: Any):
        self.blocks = blocks
        self.layers = tuple(layers)
        self.positions = positions
        self.activations: dict[int, Any] = {}
        self.handles: list[Any] = []

    def _hook(self, layer: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = output if torch.is_tensor(output) else output[0]
            if not torch.is_tensor(hidden) or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise TypeError(f"block {layer} returned an unexpected activation")
            self.activations[layer] = hidden[0].index_select(0, self.positions).detach().clone()

        return capture

    def __enter__(self) -> "PostBlockCapture":
        try:
            for layer in self.layers:
                self.handles.append(
                    self.blocks[layer].register_forward_hook(self._hook(layer))
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_exc: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def target_token_ids(
    prompt_token_ids: Sequence[int], positions: Sequence[int]
) -> tuple[list[int], list[str]]:
    targets = [int(prompt_token_ids[position + 1]) for position in positions]
    return targets, ["teacher_forced_next_token"] * len(targets)


def compact_topk(
    logits: Any, target_ids: Sequence[int], tokenizer: Any, *, top_k: int
) -> list[dict[str, Any]]:
    import torch

    logits = logits.detach().float().cpu()
    if logits.ndim != 2 or logits.shape[0] != len(target_ids):
        raise ValueError("logit rows do not match target IDs")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("readout logits are non-finite")
    records = []
    for row, target_id in zip(logits, target_ids, strict=True):
        values, token_ids = torch.topk(row, top_k)
        target_score = row[target_id]
        records.append(
            {
                "token_ids": token_ids.tolist(),
                "tokens": [tokenizer.decode([int(token)]) for token in token_ids],
                "scores": [round(float(value), 6) for value in values],
                "target_token_id": target_id,
                "target_token": tokenizer.decode([target_id]),
                "target_score": round(float(target_score), 6),
                "target_rank": int((row > target_score).sum().item()) + 1,
            }
        )
    return records


def readout_logits(
    residuals: Any,
    final_norm: Any,
    lm_head: Any,
    *,
    jacobian: Any | None = None,
) -> Any:
    """Apply optional J transport, then final norm once, then the LM head."""
    import torch

    hidden = residuals
    if jacobian is not None:
        if jacobian.ndim != 2 or jacobian.shape != (hidden.shape[-1], hidden.shape[-1]):
            raise ValueError("Jacobian shape does not match residual width")
        matrix = jacobian.to(device=hidden.device, dtype=torch.float32)
        hidden = hidden.float() @ matrix.T
        del matrix
    hidden = hidden.to(dtype=lm_head.weight.dtype)
    normalized = final_norm(hidden)
    if normalized.dtype != lm_head.weight.dtype:
        raise RuntimeError("final norm and LM head dtypes disagree")
    logits = lm_head(normalized)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("readout produced non-finite logits")
    return logits


def prepare_eval_lm_head(
    model: Any,
    device: Any,
    *,
    require_cuda_embedding: bool = True,
) -> dict[str, Any]:
    import torch

    head = model.lm_head
    embedding = model.model.embed_tokens
    if model.config.tie_word_embeddings:
        raise RuntimeError("evaluation requires an untied LM head")
    if head.weight is embedding.weight:
        raise RuntimeError("LM head unexpectedly aliases token embeddings")
    if head.weight.device.type != "cpu" or head.weight.dtype != torch.bfloat16:
        raise RuntimeError("fitter must leave an unquantized BF16 LM head on CPU")
    if require_cuda_embedding and embedding.weight.device.type != "cuda":
        raise RuntimeError("token embeddings must remain on CUDA")
    target = torch.device(device)
    nbytes = head.weight.numel() * head.weight.element_size()
    if target.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(target)
        reserve = nbytes + fitter.D_MODEL * fitter.D_MODEL * 4 + 512 * 2**20
        if free_bytes < reserve:
            raise RuntimeError("insufficient CUDA memory for LM head and streamed Jacobian")
    head.to(device=target)
    if head.weight.device != target or any(
        parameter.requires_grad for parameter in head.parameters()
    ):
        raise RuntimeError("failed to prepare frozen LM head for evaluation")
    return {
        "untied": True,
        "dtype": str(head.weight.dtype),
        "device": str(head.weight.device),
        "nbytes": nbytes,
    }


def load_lens_checkpoints(
    local_path: Path,
    public_path: Path,
    *,
    d_model: int = fitter.D_MODEL,
    source_layers: Sequence[int] = SOURCE_LAYERS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    expected_layers = list(source_layers)
    checkpoints = []
    for label, path in (("local", local_path), ("public", public_path)):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("d_model") != d_model
            or checkpoint.get("source_layers") != expected_layers
            or not isinstance(checkpoint.get("J"), dict)
            or set(checkpoint["J"]) != set(source_layers)
        ):
            raise ValueError(f"{label} lens checkpoint contract mismatch")
        checkpoints.append(checkpoint)
    return checkpoints[0], checkpoints[1]


def evaluate_prompt(
    model: Any,
    tokenizer: Any,
    prompt: dict[str, Any],
    *,
    layers: Sequence[int],
    positions: Sequence[int],
    top_k: int,
    local_jacobians: dict[int, Any],
    public_jacobians: dict[int, Any],
) -> dict[str, Any]:
    import torch

    token_ids = prompt["token_ids"]
    resolved = resolve_positions(positions, len(token_ids))
    device = model.model.embed_tokens.weight.device
    input_ids = torch.tensor(token_ids, device=device, dtype=torch.long)[None, :]
    position_index = torch.tensor(resolved, device=device, dtype=torch.long)

    with torch.inference_mode(), PostBlockCapture(
        model.model.layers, layers, position_index
    ) as capture:
        output = model.model(input_ids=input_ids, use_cache=False, return_dict=True)
        missing = set(layers) - set(capture.activations)
        if missing:
            raise RuntimeError(f"post-block hooks did not fire: {sorted(missing)}")
        final_normalized = output.last_hidden_state[0].index_select(0, position_index)
        final_logits = model.lm_head(final_normalized)
        targets, target_sources = target_token_ids(token_ids, resolved)
        final_records = compact_topk(final_logits, targets, tokenizer, top_k=top_k)

        layer_records = []
        for layer in layers:
            residuals = capture.activations[layer]
            vanilla = compact_topk(
                readout_logits(residuals, model.model.norm, model.lm_head),
                targets,
                tokenizer,
                top_k=top_k,
            )
            local = compact_topk(
                readout_logits(
                    residuals,
                    model.model.norm,
                    model.lm_head,
                    jacobian=local_jacobians[layer],
                ),
                targets,
                tokenizer,
                top_k=top_k,
            )
            public = compact_topk(
                readout_logits(
                    residuals,
                    model.model.norm,
                    model.lm_head,
                    jacobian=public_jacobians[layer],
                ),
                targets,
                tokenizer,
                top_k=top_k,
            )
            layer_records.append(
                {
                    "layer": layer,
                    "layer_type": model.config.layer_types[layer],
                    "positions": [
                        {
                            "token_position": position,
                            "vanilla_logit_lens": vanilla[index],
                            "local_jacobian_lens": local[index],
                            "public_jacobian_lens": public[index],
                        }
                        for index, position in enumerate(resolved)
                    ],
                }
            )

    return {
        **prompt,
        "tokens": [tokenizer.decode([token]) for token in token_ids],
        "positions_requested": list(positions),
        "positions_resolved": resolved,
        "targets": [
            {
                "token_position": position,
                "target_token_id": target,
                "target_token": tokenizer.decode([target]),
                "source": source,
                "final_model": final_records[index],
            }
            for index, (position, target, source) in enumerate(
                zip(resolved, targets, target_sources, strict=True)
            )
        ],
        "layers": layer_records,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-based average ranks, assigning tied values their mean rank."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def spearman_target_rank(
    left: Sequence[int], right: Sequence[int]
) -> dict[str, Any]:
    """Spearman correlation between paired target-token ranks."""
    import math

    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman correlation needs at least two paired ranks")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks, strict=True)
    )
    left_squared = sum((value - left_mean) ** 2 for value in left_ranks)
    right_squared = sum((value - right_mean) ** 2 for value in right_ranks)
    denominator = math.sqrt(left_squared * right_squared)
    coefficient = None if denominator == 0 else numerator / denominator
    return {
        "observations": len(left),
        "defined": coefficient is not None,
        "coefficient": coefficient,
    }


def method_target_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("method metrics need at least one observation")
    ranks = [int(record["target_rank"]) for record in records]
    top1 = sum(rank == 1 for rank in ranks)
    top5 = sum(rank <= 5 for rank in ranks)
    return {
        "observations": len(ranks),
        "mean_target_rank": sum(ranks) / len(ranks),
        "target_top1_count": top1,
        "target_top1_rate": top1 / len(ranks),
        "target_top5_count": top5,
        "target_top5_rate": top5 / len(ranks),
    }


def compare_method_records(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Compare two readouts on top-k predictions and target-token ranks."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("method comparison needs at least two paired observations")
    top1_agreement = 0
    top5_exact = 0
    top5_overlap = 0
    left_ranks: list[int] = []
    right_ranks: list[int] = []
    for left_record, right_record in zip(left, right, strict=True):
        if left_record.get("target_token_id") != right_record.get("target_token_id"):
            raise ValueError("method comparison target-token mismatch")
        left_tokens = left_record.get("token_ids")
        right_tokens = right_record.get("token_ids")
        if (
            not isinstance(left_tokens, list)
            or not isinstance(right_tokens, list)
            or len(left_tokens) < 5
            or len(right_tokens) < 5
        ):
            raise ValueError("method comparison requires at least five predictions")
        left_top5 = set(left_tokens[:5])
        right_top5 = set(right_tokens[:5])
        top1_agreement += left_tokens[0] == right_tokens[0]
        top5_exact += left_top5 == right_top5
        top5_overlap += len(left_top5 & right_top5)
        left_ranks.append(int(left_record["target_rank"]))
        right_ranks.append(int(right_record["target_rank"]))
    observations = len(left)
    return {
        "observations": observations,
        "top1_agreement_count": top1_agreement,
        "top1_agreement_rate": top1_agreement / observations,
        "top5_exact_set_agreement_count": top5_exact,
        "top5_exact_set_agreement_rate": top5_exact / observations,
        "top5_overlap_count": top5_overlap,
        "top5_overlap_mean_fraction": top5_overlap / (5 * observations),
        "spearman_target_rank": spearman_target_rank(left_ranks, right_ranks),
    }


def summarize(
    experiments: Sequence[dict[str, Any]], layers: Sequence[int]
) -> list[dict[str, Any]]:
    summaries = []
    for layer in layers:
        methods: dict[str, list[dict[str, Any]]] = {
            "logit_lens": [],
            "local_jacobian_lens": [],
            "public_jacobian_lens": [],
        }
        for experiment in experiments:
            layer_record = next(
                item for item in experiment["layers"] if item["layer"] == layer
            )
            targets = experiment["targets"]
            if len(layer_record["positions"]) != len(targets):
                raise ValueError("layer and target observation counts differ")
            for position, target in zip(
                layer_record["positions"], targets, strict=True
            ):
                if position["token_position"] != target["token_position"]:
                    raise ValueError("layer and target positions differ")
                methods["logit_lens"].append(position["vanilla_logit_lens"])
                methods["local_jacobian_lens"].append(
                    position["local_jacobian_lens"]
                )
                methods["public_jacobian_lens"].append(
                    position["public_jacobian_lens"]
                )
        summaries.append(
            {
                "layer": layer,
                "methods": {
                    method: method_target_metrics(records)
                    for method, records in methods.items()
                },
                "comparisons": {
                    "local_vs_public": compare_method_records(
                        methods["local_jacobian_lens"],
                        methods["public_jacobian_lens"],
                    ),
                    "logit_vs_public": compare_method_records(
                        methods["logit_lens"],
                        methods["public_jacobian_lens"],
                    ),
                },
            }
        )
    return summaries


def require_matching_nf4_aggregate(
    provenance_path: Path, model_metadata: dict[str, Any]
) -> str:
    """Bind evaluation to the exact packed NF4 weights used for fitting."""
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    try:
        fitted = provenance["model"]["quantized_weights"]["aggregate_sha256"]
        evaluated = model_metadata["quantized_weights"]["aggregate_sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "NF4 aggregate SHA-256 is missing from fit or model metadata"
        ) from exc
    for label, value in (("fitted", fitted), ("evaluated", evaluated)):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{label} NF4 aggregate SHA-256 is invalid")
    if evaluated != fitted:
        raise ValueError(
            "evaluation NF4 aggregate does not match the weights used for fitting: "
            f"fit={fitted}, evaluation={evaluated}"
        )
    return fitted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-path", type=Path, required=True)
    parser.add_argument("--local-sha256", required=True)
    parser.add_argument("--local-provenance", type=Path, required=True)
    parser.add_argument("--public-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--positions", default="16,32,64,96")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = fitter.CUBLAS_WORKSPACE_CONFIG
    args = build_parser().parse_args()
    layers = validate_layers(parse_integer_list(args.layers, allow_all=True))
    positions = parse_integer_list(args.positions)
    if not 5 <= args.top_k <= 100:
        raise ValueError("--top-k must be in 5..100 for top-5 agreement")
    fitter.configure_determinism(0)

    from huggingface_hub import hf_hub_download
    import torch

    public_path = args.public_path.resolve() if args.public_path else Path(
        hf_hub_download(
            repo_id=LENS_REPO,
            filename=LENS_FILENAME,
            revision=LENS_REVISION,
            local_files_only=True,
        )
    )
    local_path = args.local_path.resolve()
    provenance_path = args.local_provenance.resolve()
    artifacts = verify_artifacts(
        local_path=local_path,
        local_sha256=args.local_sha256,
        local_provenance=provenance_path,
        public_path=public_path,
    )
    local_checkpoint, public_checkpoint = load_lens_checkpoints(local_path, public_path)

    versions = fitter.require_runtime()
    snapshot, snapshot_records = fitter.resolve_and_verify_snapshot(args.cache_dir, False)
    tokenizer = fitter.load_tokenizer(snapshot)
    prompts, prompt_sha256 = load_frozen_prompts(PROMPTS_PATH, tokenizer)
    model, model_metadata = fitter.load_nf4_model(snapshot)
    quantized_weights = model_metadata["quantized_weights"]
    fit_nf4_sha256 = require_matching_nf4_aggregate(provenance_path, model_metadata)
    head_metadata = prepare_eval_lm_head(model, torch.device("cuda:0"))

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    experiments = [
        evaluate_prompt(
            model,
            tokenizer,
            prompt,
            layers=layers,
            positions=positions,
            top_k=args.top_k,
            local_jacobians=local_checkpoint["J"],
            public_jacobians=public_checkpoint["J"],
        )
        for prompt in prompts
    ]
    torch.cuda.synchronize()
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scope": "held-out NF4 readout evaluation; no fitting or NVFP4 equivalence claim",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "artifacts": artifacts,
        "model": {
            "repo_id": fitter.MODEL_ID,
            "revision": fitter.MODEL_REVISION,
            "snapshot": str(snapshot),
            "source_artifacts": snapshot_records,
            "nf4_linear_count": model_metadata["nf4_linear_count"],
            "nf4_weights": {
                "module_count": quantized_weights["module_count"],
                "aggregate_sha256": quantized_weights["aggregate_sha256"],
                "fit_aggregate_sha256": fit_nf4_sha256,
                "matches_fit": True,
            },
            "lm_head": head_metadata,
            "final_norm_contract": "exactly once before each intermediate readout",
        },
        "prompts": {
            "path": str(PROMPTS_PATH),
            "sha256": prompt_sha256,
            "count": len(prompts),
            "token_count": PROMPT_TOKEN_COUNT,
            "dataset": {
                "repo": DATASET_REPO,
                "revision": DATASET_REVISION,
                "config": DATASET_CONFIG,
                "split": prompts[0]["dataset_split"],
            },
        },
        "configuration": {
            "layers": layers,
            "positions": positions,
            "top_k": args.top_k,
            "target_semantics": "teacher_forced_next_token",
            "comparison_metrics": {
                "pairs": ["local_vs_public", "logit_vs_public"],
                "top1": "exact top-token agreement",
                "top5": "set overlap fraction and exact-set agreement",
                "rank_correlation": "Spearman correlation of paired target-token ranks",
            },
            "seed": 0,
            "deterministic_algorithms": True,
            "offline": True,
        },
        "host": {
            "platform": platform.platform(),
            "versions": versions,
            "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "summary": summarize(experiments, layers),
        "experiments": experiments,
    }
    fitter.atomic_write_json(args.output, report)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
