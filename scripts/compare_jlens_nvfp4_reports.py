#!/usr/bin/env python3
"""Compare paired native/public Jacobian Lens readout reports offline.

The inputs are two schema-v3 reports emitted by ``run_jlens_nvfp4.py`` for
the same target-model prompts. Adapter reconstruction is retained as an
independent certificate; a failed adapter certificate does not suppress lens
metrics or become a lens-quality verdict.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import tempfile
from typing import Any, Mapping, Sequence

from download_jlens import (
    LENS_FILENAME,
    LENS_REPO,
    LENS_REVISION,
    LENS_SHA256,
    LENS_SIZE,
)
from verify_nvfp4_ste_artifact import (
    D_MODEL,
    FIT_ESTIMATOR_LABEL,
    FIT_QUANTIZATION_LABEL,
    MODEL_METADATA_SHA256,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_SHARDS,
    N_PROMPTS,
    PRODUCTION_CONTRACT_SHA256,
    SOURCE_LAYERS,
    TARGET_LAYER,
)


SCHEMA_VERSION = 1
INPUT_SCHEMA_VERSION = 3
SCORE_ENCODING = "unrounded-float32"
MIN_TOP_K = 5
FINAL_NORM_MAX_ABS_TOLERANCE = 0.125
FINAL_NORM_RMS_TOLERANCE = 0.006
FINAL_LOGIT_MAX_ABS_TOLERANCE = 0.0625
FINAL_LOGIT_RMS_TOLERANCE = 0.01
FINAL_TOPK_PARITY_K = 5
TARGET_VALUE_FIELDS = ("target_logprob", "target_score")
NATIVE_LENS_KIND = "native_nvfp4_ste_fit"
PUBLIC_LENS_KIND = "pinned_public"
PUBLIC_N_PROMPTS = 1000
PUBLIC_FIT_TIME_MODEL_PRECISION = "unpublished"
PUBLIC_FIT_TIME_QUANTIZATION = "unpublished"
PUBLIC_LENS_APPLICATION = (
    "public Qwen3.6-27B FP16 lens with unpublished fit-time precision and "
    "quantization applied to NVFP4/FP8 residuals"
)
MODEL_CONFIG_SHA256 = "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338"
MODEL_INDEX_SHA256 = "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2"
EXPECTED_PLATFORM = "Linux-7.0.0-27-generic-x86_64-with-glibc2.43"
EXPECTED_PYTHON = "3.12.13"
EXPECTED_GPU_IDENTITY = {
    "name": "NVIDIA GeForce RTX 5090",
    "driver_version": "595.71.05",
    "memory_total_mib": "32607",
    "compute_capability": "12.0",
}
EXPECTED_PACKAGES = {
    "huggingface-hub": "1.21.0",
    "torch": "2.11.0+cu130",
    "transformers": "5.12.1",
    "triton": "3.6.0",
    "vllm": "0.23.0",
}
EXPECTED_RUNTIME_SEMANTICS = {
    "mtp_enabled": False,
    "enforce_eager": True,
    "language_model_only": True,
    "capture_adapter": "vLLM apply_model forward hooks",
    "transport_dtype": "torch.float32",
    "readout_dtype": "torch.bfloat16",
}
REQUIRED_TRUE_ASSERTIONS = (
    "lens_hash_matches",
    "lens_metadata_matches",
    "model_architecture_matches",
    "all_final_layer_top1_match_greedy",
)
FINAL_TOP1_ASSERTION = "all_final_layer_top1_match_greedy"
ADAPTER_ASSERTION = "all_final_adapter_reconstructions_within_tolerance"
ADAPTER_DIAGNOSTIC_FIELDS = (
    "final_layer_top1_matches_greedy",
    "final_norm_reconstruction",
    "final_logits_reconstruction",
    "final_model_readout",
    "captured_final_model_readout",
    "residual_capture_manifest",
)
RESIDUAL_CAPTURE_ALGORITHM = (
    "SHA-256 over length-prefixed canonical layer/shape/dtype/"
    "token-position/byte-count headers and logical row-major FP32 bytes"
)
PROMPT_IDENTITY_FIELDS = (
    "id",
    "prompt",
    "prompt_token_ids",
    "prompt_tokens",
    "positions_requested",
    "positions_resolved",
    "capture_positions_resolved",
    "final_validation_position",
    "position_tokens",
    "generated_token_id",
    "generated_token",
    "generated_text",
)
MODEL_IDENTITY_FIELDS = (
    "repo_id",
    "revision",
    "config_sha256",
    "index_sha256",
    "quant_method",
    "quant_algo",
    "model_info",
    "checkpoint_integrity",
)
RUNTIME_IDENTITY_FIELDS = (
    "mtp_enabled",
    "enforce_eager",
    "language_model_only",
    "max_model_len",
    "gpu_memory_utilization",
    "capture_adapter",
    "transport_dtype",
    "readout_dtype",
)
EXPECTED_CHECKPOINT_INTEGRITY = {
    "policy": "ModelOptCheckpoint(strict_pinned=True)",
    "validated_before_model_load": True,
    "validated_after_evaluation": True,
    "metadata_sha256": MODEL_METADATA_SHA256,
    "shards": {
        filename: {
            "bytes": record["bytes"],
            "sha256": record["blob_sha256"],
        }
        for filename, record in MODEL_SHARDS.items()
    },
}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validate_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        _finite_number(value, label)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_tree(item, f"{label}.{key}")
        return
    raise ValueError(f"{label} contains an unsupported JSON value")


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _read_report_file(path: Path) -> dict[str, Any]:
    """Read, hash, and parse exactly one regular-file byte sequence."""

    path = Path(os.path.abspath(path.expanduser()))
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"report must be a regular non-symlink file: {path}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"report must be a regular file: {path}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError(f"report changed before it was opened: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 16 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        or total != opened.st_size
    ):
        raise RuntimeError(f"report changed while it was read: {path}")

    raw = b"".join(chunks)
    try:
        value = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"report is not UTF-8: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON report: {path}") from error
    report = _mapping(value, str(path))
    _validate_finite_tree(report, str(path))
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": total,
        "device": opened.st_dev,
        "inode": opened.st_ino,
        "report": report,
    }


def load_report(path: Path) -> dict[str, Any]:
    return _read_report_file(path)["report"]


def _identity(record: Mapping[str, Any], fields: Sequence[str], label: str) -> dict[str, Any]:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"{label} is missing identity fields: {missing}")
    return {field: record[field] for field in fields}


def _average_ranks(values: Sequence[int]) -> list[float]:
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


def spearman_target_rank(left: Sequence[int], right: Sequence[int]) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("target-rank correlation requires paired observations")
    if len(left) < 2:
        return {"observations": len(left), "defined": False, "coefficient": None}
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks, strict=True)
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


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("numeric summary requires observations")
    return {
        "observations": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _validate_readout(
    record: Any,
    label: str,
    *,
    expected_target_id: int,
    expected_target_token: str,
    expected_top_k: int | None,
    allow_legacy_rounded_ties: bool = False,
) -> dict[str, Any]:
    result = _mapping(record, label)
    token_ids = _list(result.get("token_ids"), f"{label}.token_ids")
    tokens = _list(result.get("tokens"), f"{label}.tokens")
    scores = _list(result.get("scores"), f"{label}.scores")
    top_k = len(token_ids)
    if top_k < MIN_TOP_K or len(tokens) != top_k or len(scores) != top_k:
        raise ValueError(f"{label} must contain aligned top-{MIN_TOP_K} or larger readout lists")
    if expected_top_k is not None and top_k != expected_top_k:
        raise ValueError(f"{label} top-k differs from the paired report grid")
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in token_ids
    ):
        raise ValueError(f"{label}.token_ids must contain nonnegative integers")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(f"{label}.token_ids contains duplicates")
    if any(not isinstance(token, str) for token in tokens):
        raise ValueError(f"{label}.tokens must contain strings")
    normalized_scores = [
        _finite_number(score, f"{label}.scores[{index}]")
        for index, score in enumerate(scores)
    ]
    if any(
        normalized_scores[index] < normalized_scores[index + 1]
        for index in range(top_k - 1)
    ):
        raise ValueError(f"{label}.scores must be in descending order")
    target_id = result.get("target_token_id")
    target_rank = result.get("target_rank")
    if isinstance(target_id, bool) or not isinstance(target_id, int) or target_id < 0:
        raise ValueError(f"{label}.target_token_id must be a nonnegative integer")
    if target_id != expected_target_id:
        raise ValueError(f"{label}.target_token_id does not match the paired prompt")
    target_token = result.get("target_token")
    if not isinstance(target_token, str) or target_token != expected_target_token:
        raise ValueError(f"{label}.target_token does not match the paired prompt")
    if isinstance(target_rank, bool) or not isinstance(target_rank, int) or target_rank < 1:
        raise ValueError(f"{label}.target_rank must be a positive integer")
    if "target_score" not in result:
        raise ValueError(f"{label}.target_score is required")
    target_score = _finite_number(result["target_score"], f"{label}.target_score")
    greater_in_top_k = sum(score > target_score for score in normalized_scores)
    tied_in_top_k = sum(score == target_score for score in normalized_scores)
    if target_rank <= top_k:
        if allow_legacy_rounded_ties:
            maximum_rank = greater_in_top_k + tied_in_top_k + (
                0 if target_id in token_ids else 1
            )
            rank_matches = greater_in_top_k + 1 <= target_rank <= maximum_rank
        else:
            rank_matches = target_rank == greater_in_top_k + 1
        if target_score not in normalized_scores or not rank_matches:
            raise ValueError(f"{label} target_rank/target_score is inconsistent with top-k")
    elif allow_legacy_rounded_ties:
        if any(score < target_score for score in normalized_scores):
            raise ValueError(f"{label} target_rank/target_score is inconsistent with top-k")
    elif greater_in_top_k != top_k:
        raise ValueError(f"{label} target_rank/target_score is inconsistent with top-k")
    if target_id in token_ids:
        target_index = token_ids.index(target_id)
        if (
            target_rank > top_k
            or normalized_scores[target_index] != target_score
            or tokens[target_index] != target_token
        ):
            raise ValueError(f"{label} target entry is inconsistent with top-k")
    for field in TARGET_VALUE_FIELDS:
        if field in result:
            _finite_number(result[field], f"{label}.{field}")
    return result


def method_target_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("method metrics require observations")
    ranks = [int(record["target_rank"]) for record in records]
    result: dict[str, Any] = {
        "observations": len(records),
        "target_rank": _summary([float(rank) for rank in ranks]),
        "target_top1_count": sum(rank == 1 for rank in ranks),
        "target_top1_rate": sum(rank == 1 for rank in ranks) / len(ranks),
        "target_top5_count": sum(rank <= 5 for rank in ranks),
        "target_top5_rate": sum(rank <= 5 for rank in ranks) / len(ranks),
        "target_values": {},
    }
    for field in TARGET_VALUE_FIELDS:
        present = [field in record for record in records]
        if any(present) and not all(present):
            raise ValueError(f"{field} availability differs within one method")
        if all(present):
            result["target_values"][field] = _summary(
                [float(record[field]) for record in records]
            )
    return result


def compare_method_records(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("method comparison requires paired observations")
    top1 = 0
    top5_exact = 0
    top5_overlap = 0
    left_ranks: list[int] = []
    right_ranks: list[int] = []
    for index, (left_record, right_record) in enumerate(zip(left, right, strict=True)):
        if left_record["target_token_id"] != right_record["target_token_id"]:
            raise ValueError(f"observation {index}: target-token mismatch")
        left_ids = left_record["token_ids"]
        right_ids = right_record["token_ids"]
        if len(left_ids) != len(right_ids):
            raise ValueError(f"observation {index}: readout top-k mismatch")
        left_top5 = set(left_ids[:5])
        right_top5 = set(right_ids[:5])
        top1 += left_ids[0] == right_ids[0]
        top5_exact += left_top5 == right_top5
        top5_overlap += len(left_top5 & right_top5)
        left_ranks.append(int(left_record["target_rank"]))
        right_ranks.append(int(right_record["target_rank"]))

    rank_deltas = [
        float(a - b) for a, b in zip(left_ranks, right_ranks, strict=True)
    ]
    result: dict[str, Any] = {
        "observations": len(left),
        "top1_agreement_count": top1,
        "top1_agreement_rate": top1 / len(left),
        "top5_exact_set_agreement_count": top5_exact,
        "top5_exact_set_agreement_rate": top5_exact / len(left),
        "top5_overlap_count": top5_overlap,
        "top5_overlap_mean_fraction": top5_overlap / (5 * len(left)),
        "target_rank": {
            "left": _summary([float(value) for value in left_ranks]),
            "right": _summary([float(value) for value in right_ranks]),
            "mean_signed_delta_left_minus_right": statistics.fmean(rank_deltas),
            "mean_absolute_delta": statistics.fmean(abs(value) for value in rank_deltas),
            "spearman": spearman_target_rank(left_ranks, right_ranks),
        },
        "target_values": {},
    }
    for field in TARGET_VALUE_FIELDS:
        left_present = [field in record for record in left]
        right_present = [field in record for record in right]
        if left_present != right_present or (any(left_present) and not all(left_present)):
            raise ValueError(f"paired {field} availability mismatch")
        if all(left_present):
            left_values = [float(record[field]) for record in left]
            right_values = [float(record[field]) for record in right]
            deltas = [a - b for a, b in zip(left_values, right_values, strict=True)]
            result["target_values"][field] = {
                "left": _summary(left_values),
                "right": _summary(right_values),
                "mean_signed_delta_left_minus_right": statistics.fmean(deltas),
                "mean_absolute_delta": statistics.fmean(abs(value) for value in deltas),
                "root_mean_square_delta": math.sqrt(
                    statistics.fmean(value * value for value in deltas)
                ),
            }
    return result


def _adapter_certificate(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    status = report.get("status")
    if status not in {"passed", "failed"}:
        raise ValueError(f"{label} report status must be passed or failed")
    assertions = _mapping(report.get("assertions"), f"{label}.assertions")
    for name in REQUIRED_TRUE_ASSERTIONS:
        if name == FINAL_TOP1_ASSERTION:
            continue
        if assertions.get(name) is not True:
            raise ValueError(f"{label}.assertions.{name} must be true")
    final_top1_assertion = assertions.get(FINAL_TOP1_ASSERTION)
    if not isinstance(final_top1_assertion, bool):
        raise ValueError(f"{label}.assertions.{FINAL_TOP1_ASSERTION} must be boolean")
    adapter_assertion = assertions.get(ADAPTER_ASSERTION)
    if not isinstance(adapter_assertion, bool):
        raise ValueError(f"{label}.assertions.{ADAPTER_ASSERTION} must be boolean")
    experiments = _list(report.get("experiments"), f"{label}.experiments")
    diagnostics = []
    for index, experiment_value in enumerate(experiments):
        experiment = _mapping(experiment_value, f"{label}.experiments[{index}]")
        prompt_id = experiment.get("id")
        prompt_label = f"{label}.{prompt_id}"
        diagnostic = {"id": prompt_id}
        for field in ADAPTER_DIAGNOSTIC_FIELDS:
            if field not in experiment:
                raise ValueError(f"{prompt_label} is missing {field}")
            diagnostic[field] = deepcopy(experiment[field])

        norm_label = f"{prompt_label}.final_norm_reconstruction"
        norm = _mapping(diagnostic["final_norm_reconstruction"], norm_label)
        norm_fields = {
            "max_abs_error",
            "rms_error",
            "reference_rms",
            "relative_rms_error",
            "max_abs_tolerance",
            "rms_tolerance",
            "within_tolerance",
        }
        if set(norm) != norm_fields:
            raise ValueError(f"{norm_label} fields do not match the diagnostic schema")
        norm_max_error = _finite_number(norm["max_abs_error"], f"{norm_label}.max_abs_error")
        norm_rms_error = _finite_number(norm["rms_error"], f"{norm_label}.rms_error")
        norm_reference_rms = _finite_number(norm["reference_rms"], f"{norm_label}.reference_rms")
        norm_relative_rms = _finite_number(
            norm["relative_rms_error"], f"{norm_label}.relative_rms_error"
        )
        if min(norm_max_error, norm_rms_error, norm_relative_rms) < 0 or norm_reference_rms <= 0:
            raise ValueError(f"{norm_label} errors/reference must be nonnegative/positive")
        expected_relative_rms = norm_rms_error / norm_reference_rms
        if not math.isclose(
            norm_relative_rms,
            expected_relative_rms,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"{norm_label}.relative_rms_error does not match rms_error/reference_rms"
            )
        norm_max_tolerance = _finite_number(
            norm["max_abs_tolerance"], f"{norm_label}.max_abs_tolerance"
        )
        norm_rms_tolerance = _finite_number(
            norm["rms_tolerance"], f"{norm_label}.rms_tolerance"
        )
        if (
            norm_max_tolerance != FINAL_NORM_MAX_ABS_TOLERANCE
            or norm_rms_tolerance != FINAL_NORM_RMS_TOLERANCE
        ):
            raise ValueError(f"{norm_label} tolerances do not match the pinned constants")
        reported_norm_pass = norm.get("within_tolerance")
        if not isinstance(reported_norm_pass, bool):
            raise ValueError(f"{norm_label}.within_tolerance must be boolean")
        derived_norm_pass = (
            norm_max_error <= FINAL_NORM_MAX_ABS_TOLERANCE
            and norm_rms_error <= FINAL_NORM_RMS_TOLERANCE
        )
        if reported_norm_pass != derived_norm_pass:
            raise ValueError(
                f"{norm_label}.within_tolerance does not match the derived result"
            )
        norm["within_tolerance"] = derived_norm_pass

        logits_label = f"{prompt_label}.final_logits_reconstruction"
        logits = _mapping(diagnostic["final_logits_reconstruction"], logits_label)
        logits_fields = {
            "max_abs_error",
            "max_abs_tolerance",
            "rms_error",
            "rms_tolerance",
            "top_k_prefix",
            "top_k_prefix_token_ids_match",
            "within_tolerance",
        }
        if set(logits) != logits_fields:
            raise ValueError(f"{logits_label} fields do not match the diagnostic schema")
        logit_max_error = _finite_number(
            logits["max_abs_error"], f"{logits_label}.max_abs_error"
        )
        logit_rms_error = _finite_number(
            logits["rms_error"], f"{logits_label}.rms_error"
        )
        if min(logit_max_error, logit_rms_error) < 0:
            raise ValueError(f"{logits_label} errors must be nonnegative")
        logit_max_tolerance = _finite_number(
            logits["max_abs_tolerance"], f"{logits_label}.max_abs_tolerance"
        )
        logit_rms_tolerance = _finite_number(
            logits["rms_tolerance"], f"{logits_label}.rms_tolerance"
        )
        if (
            logit_max_tolerance != FINAL_LOGIT_MAX_ABS_TOLERANCE
            or logit_rms_tolerance != FINAL_LOGIT_RMS_TOLERANCE
        ):
            raise ValueError(f"{logits_label} tolerances do not match the pinned constants")
        prefix = logits.get("top_k_prefix")
        if (
            isinstance(prefix, bool)
            or not isinstance(prefix, int)
            or prefix != FINAL_TOPK_PARITY_K
        ):
            raise ValueError(f"{logits_label}.top_k_prefix must equal {FINAL_TOPK_PARITY_K}")

        capture_positions = _list(
            experiment.get("capture_positions_resolved"),
            f"{prompt_label}.capture_positions_resolved",
        )
        final_readouts = _list(
            diagnostic["final_model_readout"], f"{prompt_label}.final_model_readout"
        )
        captured_readouts = _list(
            diagnostic["captured_final_model_readout"],
            f"{prompt_label}.captured_final_model_readout",
        )
        if (
            not capture_positions
            or len(final_readouts) != len(capture_positions)
            or len(captured_readouts) != len(capture_positions)
        ):
            raise ValueError(f"{prompt_label} final readouts do not match captures")

        final_token_ids: list[list[int]] = []
        captured_token_ids: list[list[int]] = []
        for readout_index, (final_value, captured_value) in enumerate(
            zip(final_readouts, captured_readouts, strict=True)
        ):
            final_record = _mapping(
                final_value, f"{prompt_label}.final_model_readout[{readout_index}]"
            )
            captured_record = _mapping(
                captured_value,
                f"{prompt_label}.captured_final_model_readout[{readout_index}]",
            )
            final_ids = _list(
                final_record.get("token_ids"),
                f"{prompt_label}.final_model_readout[{readout_index}].token_ids",
            )
            captured_ids = _list(
                captured_record.get("token_ids"),
                f"{prompt_label}.captured_final_model_readout[{readout_index}].token_ids",
            )
            for ids, ids_label in (
                (final_ids, "final_model_readout"),
                (captured_ids, "captured_final_model_readout"),
            ):
                if len(ids) < FINAL_TOPK_PARITY_K or any(
                    isinstance(token, bool) or not isinstance(token, int) or token < 0
                    for token in ids
                ):
                    raise ValueError(
                        f"{prompt_label}.{ids_label}[{readout_index}].token_ids "
                        "cannot establish the pinned prefix"
                    )
            final_token_ids.append(final_ids)
            captured_token_ids.append(captured_ids)

        derived_prefix_match = all(
            final_ids[:FINAL_TOPK_PARITY_K]
            == captured_ids[:FINAL_TOPK_PARITY_K]
            for final_ids, captured_ids in zip(
                final_token_ids, captured_token_ids, strict=True
            )
        )
        reported_prefix_match = logits.get("top_k_prefix_token_ids_match")
        if not isinstance(reported_prefix_match, bool):
            raise ValueError(
                f"{logits_label}.top_k_prefix_token_ids_match must be boolean"
            )
        if reported_prefix_match != derived_prefix_match:
            raise ValueError(
                f"{logits_label}.top_k_prefix_token_ids_match does not match "
                "the derived token-ID prefixes"
            )
        logits["top_k_prefix_token_ids_match"] = derived_prefix_match
        reported_logits_pass = logits.get("within_tolerance")
        if not isinstance(reported_logits_pass, bool):
            raise ValueError(f"{logits_label}.within_tolerance must be boolean")
        derived_logits_pass = (
            logit_max_error <= FINAL_LOGIT_MAX_ABS_TOLERANCE
            and logit_rms_error <= FINAL_LOGIT_RMS_TOLERANCE
            and derived_prefix_match
        )
        if reported_logits_pass != derived_logits_pass:
            raise ValueError(
                f"{logits_label}.within_tolerance does not match the derived result"
            )
        logits["within_tolerance"] = derived_logits_pass

        final_position = experiment.get("final_validation_position")
        if final_position not in capture_positions:
            raise ValueError(f"{prompt_label}.final_validation_position is not captured")
        final_capture_index = capture_positions.index(final_position)
        generated_token_id = experiment.get("generated_token_id")
        if (
            isinstance(generated_token_id, bool)
            or not isinstance(generated_token_id, int)
            or generated_token_id < 0
        ):
            raise ValueError(f"{prompt_label}.generated_token_id is invalid")
        derived_final_top1 = (
            final_token_ids[final_capture_index][0] == generated_token_id
            and captured_token_ids[final_capture_index][0] == generated_token_id
        )
        reported_final_top1 = diagnostic["final_layer_top1_matches_greedy"]
        if not isinstance(reported_final_top1, bool):
            raise ValueError(
                f"{prompt_label}.final_layer_top1_matches_greedy must be boolean"
            )
        if reported_final_top1 != derived_final_top1:
            raise ValueError(
                f"{prompt_label}.final_layer_top1_matches_greedy does not match "
                "the derived final-position token IDs"
            )
        diagnostic["final_layer_top1_matches_greedy"] = derived_final_top1
        diagnostics.append(diagnostic)

    derived_final_top1 = all(
        item["final_layer_top1_matches_greedy"] for item in diagnostics
    )
    if final_top1_assertion != derived_final_top1:
        raise ValueError(
            f"{label}.assertions.{FINAL_TOP1_ASSERTION} does not match the derived result"
        )
    if not derived_final_top1:
        raise ValueError(f"{label} final-layer top-1 validation failed")
    diagnostics_pass = all(
        item["final_norm_reconstruction"]["within_tolerance"]
        and item["final_logits_reconstruction"]["within_tolerance"]
        for item in diagnostics
    )
    if adapter_assertion != diagnostics_pass:
        raise ValueError(f"{label} adapter assertion does not match its diagnostics")
    expected_status = "passed" if derived_final_top1 and diagnostics_pass else "failed"
    if status != expected_status:
        raise ValueError(f"{label} report status does not match validated assertions")
    return {
        "report_status": status,
        "assertions": deepcopy(assertions),
        "experiments": diagnostics,
    }


def _lens_identity(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    lens = _mapping(report.get("lens"), f"{label}.lens")
    fields = (
        "kind",
        "application",
        "sha256",
        "size_bytes",
        "n_prompts",
        "d_model",
        "source_layers",
        "target_layer",
        "tensor_dtype",
        "tensor_shape",
        "contract_sha256",
        "fit_model",
        "fit_model_revision",
        "fit_quantization",
        "fit_estimator",
        "fit_time_model_precision",
        "fit_time_quantization",
        "repo_id",
        "revision",
        "filename",
        "provenance_sha256",
        "provenance_size_bytes",
        "layer_aggregate_sha256",
        "committed_prompts_sha256",
        "verification_scope",
        "state_sha256",
        "state_size_bytes",
        "run_id",
        "surrogate_backward",
    )
    return {field: deepcopy(lens[field]) for field in fields if field in lens}


def _require_exact_fields(
    record: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"{label}.{field} does not match the pinned identity")


def _validate_lens_roles(
    native: Mapping[str, Any], public: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    native_lens = _mapping(native.get("lens"), "native.lens")
    public_lens = _mapping(public.get("lens"), "public.lens")
    expected_layers = list(SOURCE_LAYERS)
    _require_exact_fields(
        native_lens,
        {
            "kind": NATIVE_LENS_KIND,
            "fit_model": MODEL_REPO,
            "fit_model_revision": MODEL_REVISION,
            "fit_quantization": FIT_QUANTIZATION_LABEL,
            "fit_estimator": FIT_ESTIMATOR_LABEL,
            "contract_sha256": PRODUCTION_CONTRACT_SHA256,
            "n_prompts": N_PROMPTS,
            "d_model": D_MODEL,
            "source_layers": expected_layers,
            "target_layer": TARGET_LAYER,
            "tensor_dtype": "torch.float32",
            "tensor_shape": [D_MODEL, D_MODEL],
            "finite_checked": True,
            "checkpoint_keys": ["J", "d_model", "n_prompts", "source_layers"],
            "verification_scope": (
                "exact pinned production run; not a generic portable fit"
            ),
            "surrogate_backward": (
                "identity STE; not the literal derivative of quantized rounding"
            ),
        },
        "native.lens",
    )
    _sha256(native_lens.get("sha256"), "native.lens.sha256")
    _sha256(native_lens.get("provenance_sha256"), "native.lens.provenance_sha256")
    _sha256(
        native_lens.get("layer_aggregate_sha256"),
        "native.lens.layer_aggregate_sha256",
    )
    _sha256(
        native_lens.get("committed_prompts_sha256"),
        "native.lens.committed_prompts_sha256",
    )
    _sha256(native_lens.get("state_sha256"), "native.lens.state_sha256")
    _positive_integer(native_lens.get("size_bytes"), "native.lens.size_bytes")
    _positive_integer(
        native_lens.get("provenance_size_bytes"),
        "native.lens.provenance_size_bytes",
    )
    _positive_integer(
        native_lens.get("state_size_bytes"), "native.lens.state_size_bytes"
    )
    if not isinstance(native_lens.get("run_id"), str) or not native_lens["run_id"]:
        raise ValueError("native.lens.run_id must be a nonempty string")

    public_kind = public_lens.get("kind")
    if public_kind not in (None, PUBLIC_LENS_KIND):
        raise ValueError("public.lens.kind does not identify the pinned public lens")
    _require_exact_fields(
        public_lens,
        {
            "application": PUBLIC_LENS_APPLICATION,
            "repo_id": LENS_REPO,
            "revision": LENS_REVISION,
            "filename": LENS_FILENAME,
            "fit_time_model_precision": PUBLIC_FIT_TIME_MODEL_PRECISION,
            "fit_time_quantization": PUBLIC_FIT_TIME_QUANTIZATION,
            "sha256": LENS_SHA256,
            "size_bytes": LENS_SIZE,
            "n_prompts": PUBLIC_N_PROMPTS,
            "d_model": D_MODEL,
            "source_layers": expected_layers,
            "tensor_dtype": "torch.float16",
            "tensor_shape": [D_MODEL, D_MODEL],
            "finite_checked": False,
            "checkpoint_keys": ["J", "d_model", "n_prompts", "source_layers"],
        },
        "public.lens",
    )
    if native_lens["sha256"] == public_lens["sha256"]:
        raise ValueError("native and public lens artifacts must be distinct")
    return native_lens, public_lens


def _validate_model_identity(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    model = _identity(
        _mapping(report.get("model"), f"{label}.model"),
        MODEL_IDENTITY_FIELDS,
        f"{label}.model",
    )
    _require_exact_fields(
        model,
        {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "config_sha256": MODEL_CONFIG_SHA256,
            "index_sha256": MODEL_INDEX_SHA256,
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
        },
        f"{label}.model",
    )
    info = _mapping(model["model_info"], f"{label}.model.model_info")
    if info.get("hidden_size") != D_MODEL or info.get("layer_count") != TARGET_LAYER + 1:
        raise ValueError(f"{label}.model.model_info geometry mismatch")
    integrity = _mapping(
        model["checkpoint_integrity"], f"{label}.model.checkpoint_integrity"
    )
    if integrity != EXPECTED_CHECKPOINT_INTEGRITY:
        raise ValueError(
            f"{label}.model.checkpoint_integrity does not match the exact pinned "
            "metadata, shards, and pre/post validation contract"
        )
    model["checkpoint_integrity"] = deepcopy(integrity)
    return model


def _validate_runtime_identity(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    runtime = _identity(
        _mapping(report.get("runtime"), f"{label}.runtime"),
        RUNTIME_IDENTITY_FIELDS,
        f"{label}.runtime",
    )
    _require_exact_fields(
        runtime, EXPECTED_RUNTIME_SEMANTICS, f"{label}.runtime"
    )
    max_model_len = _positive_integer(
        runtime["max_model_len"], f"{label}.runtime.max_model_len"
    )
    gpu_utilization = _finite_number(
        runtime["gpu_memory_utilization"],
        f"{label}.runtime.gpu_memory_utilization",
    )
    if not 0.70 <= gpu_utilization <= 0.90:
        raise ValueError(
            f"{label}.runtime.gpu_memory_utilization must be in 0.70..0.90"
        )
    runtime["max_model_len"] = max_model_len
    runtime["gpu_memory_utilization"] = gpu_utilization
    return runtime


def _validate_host_identity(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    host = _mapping(report.get("host"), f"{label}.host")
    if host.get("platform") != EXPECTED_PLATFORM or host.get("python") != EXPECTED_PYTHON:
        raise ValueError(f"{label}.host platform/Python identity mismatch")
    gpu = _mapping(host.get("gpu"), f"{label}.host.gpu")
    _require_exact_fields(gpu, EXPECTED_GPU_IDENTITY, f"{label}.host.gpu")
    memory_used = gpu.get("memory_used_mib")
    if (
        not isinstance(memory_used, str)
        or not memory_used.isdigit()
        or not 0 <= int(memory_used) <= int(EXPECTED_GPU_IDENTITY["memory_total_mib"])
    ):
        raise ValueError(f"{label}.host.gpu.memory_used_mib is invalid")
    packages = _mapping(host.get("packages"), f"{label}.host.packages")
    if packages != EXPECTED_PACKAGES:
        raise ValueError(f"{label}.host package identity mismatch")
    return {
        "platform": host["platform"],
        "python": host["python"],
        "gpu": deepcopy(EXPECTED_GPU_IDENTITY),
        "packages": deepcopy(packages),
    }


def _validate_residual_capture_manifest(
    value: Any,
    label: str,
    *,
    token_positions: Sequence[int],
) -> dict[str, Any]:
    manifest = _mapping(value, label)
    expected_fields = {
        "sha256",
        "tensor_count",
        "logical_bytes",
        "token_positions",
        "algorithm",
    }
    if set(manifest) != expected_fields:
        raise ValueError(f"{label} fields do not match the residual manifest schema")
    _sha256(manifest["sha256"], f"{label}.sha256")
    tensor_count = _positive_integer(manifest["tensor_count"], f"{label}.tensor_count")
    logical_bytes = _positive_integer(manifest["logical_bytes"], f"{label}.logical_bytes")
    if tensor_count != TARGET_LAYER + 1:
        raise ValueError(f"{label}.tensor_count must cover all model residual layers")
    expected_bytes = tensor_count * len(token_positions) * D_MODEL * 4
    if logical_bytes != expected_bytes:
        raise ValueError(f"{label}.logical_bytes does not match captured FP32 geometry")
    if manifest["token_positions"] != list(token_positions):
        raise ValueError(f"{label}.token_positions does not match the capture grid")
    if manifest["algorithm"] != RESIDUAL_CAPTURE_ALGORITHM:
        raise ValueError(f"{label}.algorithm does not match the pinned digest algorithm")
    return manifest


def _macro_comparisons(layers: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = tuple(layers[0]["comparisons"])
    result: dict[str, Any] = {}
    for name in names:
        comparisons = [layer["comparisons"][name] for layer in layers]
        coefficients = [
            item["target_rank"]["spearman"]["coefficient"]
            for item in comparisons
            if item["target_rank"]["spearman"]["defined"]
        ]
        result[name] = {
            "layer_count": len(comparisons),
            "mean_top1_agreement_rate": statistics.fmean(
                item["top1_agreement_rate"] for item in comparisons
            ),
            "mean_top5_exact_set_agreement_rate": statistics.fmean(
                item["top5_exact_set_agreement_rate"] for item in comparisons
            ),
            "mean_top5_overlap_fraction": statistics.fmean(
                item["top5_overlap_mean_fraction"] for item in comparisons
            ),
            "defined_target_rank_spearman_layers": len(coefficients),
            "mean_defined_target_rank_spearman": (
                statistics.fmean(coefficients) if coefficients else None
            ),
        }
    return result


def compare_reports(native: Mapping[str, Any], public: Mapping[str, Any]) -> dict[str, Any]:
    for label, report in (("native", native), ("public", public)):
        if report.get("schema_version") != INPUT_SCHEMA_VERSION:
            raise ValueError(f"{label} report schema_version must be {INPUT_SCHEMA_VERSION}")
        if report.get("score_encoding") != SCORE_ENCODING:
            raise ValueError(
                f"{label} report score_encoding must be {SCORE_ENCODING!r}"
            )

    native_lens, public_lens = _validate_lens_roles(native, public)
    native_model = _validate_model_identity(native, "native")
    public_model = _validate_model_identity(public, "public")
    if native_model != public_model:
        raise ValueError("paired model identity mismatch")
    native_runtime = _validate_runtime_identity(native, "native")
    public_runtime = _validate_runtime_identity(public, "public")
    if native_runtime != public_runtime:
        raise ValueError("paired runtime identity mismatch")
    native_host = _validate_host_identity(native, "native")
    public_host = _validate_host_identity(public, "public")
    if native_host != public_host:
        raise ValueError("paired pinned host identity mismatch")

    native_experiments = _list(native.get("experiments"), "native.experiments")
    public_experiments = _list(public.get("experiments"), "public.experiments")
    if len(native_experiments) != len(public_experiments) or not native_experiments:
        raise ValueError("paired prompt count mismatch")

    layer_observations: dict[int, dict[str, list[dict[str, Any]]]] = {}
    layer_types: dict[int, str] = {}
    prompt_ids: list[str] = []
    positions_by_prompt: dict[str, list[int]] = {}
    residual_manifests: dict[str, dict[str, Any]] = {}
    expected_layer_order = list(SOURCE_LAYERS)
    top_k: int | None = None

    def validate_readout(
        record: Any,
        label: str,
        *,
        target_id: int,
        target_token: str,
    ) -> dict[str, Any]:
        nonlocal top_k
        result = _validate_readout(
            record,
            label,
            expected_target_id=target_id,
            expected_target_token=target_token,
            expected_top_k=top_k,
        )
        if top_k is None:
            top_k = len(result["token_ids"])
        return result

    for prompt_index, (native_value, public_value) in enumerate(
        zip(native_experiments, public_experiments, strict=True)
    ):
        native_experiment = _mapping(native_value, f"native.experiments[{prompt_index}]")
        public_experiment = _mapping(public_value, f"public.experiments[{prompt_index}]")
        for field in PROMPT_IDENTITY_FIELDS:
            if field not in native_experiment or field not in public_experiment:
                raise ValueError(f"paired prompt is missing identity field: {field}")
            if native_experiment[field] != public_experiment[field]:
                raise ValueError(f"paired prompt {field} mismatch at index {prompt_index}")
        prompt_id = native_experiment["id"]
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in prompt_ids:
            raise ValueError("prompt IDs must be unique nonempty strings")
        prompt_ids.append(prompt_id)
        if not isinstance(native_experiment["prompt"], str):
            raise ValueError(f"{prompt_id}.prompt must be a string")

        prompt_token_ids = _list(
            native_experiment["prompt_token_ids"], f"{prompt_id}.prompt_token_ids"
        )
        if not prompt_token_ids or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in prompt_token_ids
        ):
            raise ValueError(
                f"{prompt_id}.prompt_token_ids must contain nonnegative integers"
            )
        if len(prompt_token_ids) + 1 > native_runtime["max_model_len"]:
            raise ValueError(
                f"{prompt_id} leaves no generation slot under runtime.max_model_len"
            )
        prompt_tokens = _list(native_experiment["prompt_tokens"], f"{prompt_id}.prompt_tokens")
        if len(prompt_tokens) != len(prompt_token_ids) or any(
            not isinstance(token, str) for token in prompt_tokens
        ):
            raise ValueError(f"{prompt_id}.prompt_tokens must align with prompt_token_ids")
        generated_token_id = native_experiment["generated_token_id"]
        if (
            isinstance(generated_token_id, bool)
            or not isinstance(generated_token_id, int)
            or generated_token_id < 0
        ):
            raise ValueError(f"{prompt_id}.generated_token_id must be nonnegative")
        generated_token = native_experiment["generated_token"]
        if not isinstance(generated_token, str) or not isinstance(
            native_experiment["generated_text"], str
        ):
            raise ValueError(f"{prompt_id} generated token/text must be strings")

        requested_positions = _list(
            native_experiment["positions_requested"], f"{prompt_id}.positions_requested"
        )
        if not requested_positions or any(
            isinstance(position, bool) or not isinstance(position, int)
            for position in requested_positions
        ):
            raise ValueError(f"{prompt_id}.positions_requested must contain integers")
        resolved_positions = _list(
            native_experiment["positions_resolved"], f"{prompt_id}.positions_resolved"
        )
        if not resolved_positions or any(
            isinstance(position, bool) or not isinstance(position, int)
            for position in resolved_positions
        ) or len(set(resolved_positions)) != len(resolved_positions):
            raise ValueError(f"{prompt_id}.positions_resolved must be unique integers")
        calculated_positions = [
            position + len(prompt_token_ids) if position < 0 else position
            for position in requested_positions
        ]
        if calculated_positions != resolved_positions or any(
            not 0 <= position < len(prompt_token_ids) for position in resolved_positions
        ):
            raise ValueError(f"{prompt_id}.positions_resolved does not match the request")
        positions_by_prompt[prompt_id] = list(resolved_positions)

        final_position = native_experiment["final_validation_position"]
        if final_position != len(prompt_token_ids) - 1:
            raise ValueError(f"{prompt_id}.final_validation_position is invalid")
        capture_positions = _list(
            native_experiment["capture_positions_resolved"],
            f"{prompt_id}.capture_positions_resolved",
        )
        if any(
            isinstance(position, bool) or not isinstance(position, int)
            for position in capture_positions
        ) or len(set(capture_positions)) != len(capture_positions):
            raise ValueError(
                f"{prompt_id}.capture_positions_resolved must be unique integers"
            )
        expected_capture_positions = list(resolved_positions)
        if final_position not in expected_capture_positions:
            expected_capture_positions.append(final_position)
        if capture_positions != expected_capture_positions:
            raise ValueError(f"{prompt_id}.capture_positions_resolved is invalid")
        position_tokens = _list(
            native_experiment["position_tokens"], f"{prompt_id}.position_tokens"
        )
        if position_tokens != [prompt_tokens[position] for position in resolved_positions]:
            raise ValueError(f"{prompt_id}.position_tokens does not match the prompt")

        def expected_target(position: int) -> tuple[int, str]:
            if position == final_position:
                return generated_token_id, generated_token
            return prompt_token_ids[position + 1], prompt_tokens[position + 1]

        for field in ADAPTER_DIAGNOSTIC_FIELDS:
            if native_experiment.get(field) != public_experiment.get(field):
                raise ValueError(
                    f"paired lens-independent diagnostic mismatch at prompt {prompt_id}: {field}"
                )
        native_manifest = _validate_residual_capture_manifest(
            native_experiment.get("residual_capture_manifest"),
            f"native.{prompt_id}.residual_capture_manifest",
            token_positions=capture_positions,
        )
        _validate_residual_capture_manifest(
            public_experiment.get("residual_capture_manifest"),
            f"public.{prompt_id}.residual_capture_manifest",
            token_positions=capture_positions,
        )
        residual_manifests[prompt_id] = deepcopy(native_manifest)
        for side, experiment in (
            ("native", native_experiment),
            ("public", public_experiment),
        ):
            for field in ("final_model_readout", "captured_final_model_readout"):
                readouts = _list(experiment.get(field), f"{side}.{prompt_id}.{field}")
                if len(readouts) != len(capture_positions):
                    raise ValueError(f"{side}.{prompt_id}.{field} does not match captures")
                for capture_index, (position, readout) in enumerate(
                    zip(capture_positions, readouts, strict=True)
                ):
                    target_id, target_token = expected_target(position)
                    validate_readout(
                        readout,
                        f"{side}.{prompt_id}.{field}[{capture_index}]",
                        target_id=target_id,
                        target_token=target_token,
                    )

        native_layers = _list(native_experiment.get("layers"), f"native.{prompt_id}.layers")
        public_layers = _list(public_experiment.get("layers"), f"public.{prompt_id}.layers")
        native_layer_records = [
            _mapping(layer, f"native.{prompt_id}.layer") for layer in native_layers
        ]
        public_layer_records = [
            _mapping(layer, f"public.{prompt_id}.layer") for layer in public_layers
        ]
        native_order = [layer.get("layer") for layer in native_layer_records]
        public_order = [layer.get("layer") for layer in public_layer_records]
        if native_order != public_order:
            raise ValueError(f"paired layer order mismatch for prompt {prompt_id}")
        if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in native_order):
            raise ValueError(f"layer IDs must be integers for prompt {prompt_id}")
        if native_order != expected_layer_order:
            raise ValueError(
                f"observed layer grid does not match lens source_layers at prompt {prompt_id}"
            )

        for native_layer, public_layer in zip(
            native_layer_records, public_layer_records, strict=True
        ):
            layer = native_layer["layer"]
            if native_layer.get("layer_type") != public_layer.get("layer_type"):
                raise ValueError(f"paired layer type mismatch at prompt {prompt_id}, layer {layer}")
            layer_type = native_layer.get("layer_type")
            if not isinstance(layer_type, str) or not layer_type:
                raise ValueError(f"invalid layer type at prompt {prompt_id}, layer {layer}")
            if layer in layer_types and layer_types[layer] != layer_type:
                raise ValueError(f"layer type changed across prompts at layer {layer}")
            layer_types[layer] = layer_type
            native_positions = _list(
                native_layer.get("positions"), f"native.{prompt_id}.layer-{layer}.positions"
            )
            public_positions = _list(
                public_layer.get("positions"), f"public.{prompt_id}.layer-{layer}.positions"
            )
            if len(native_positions) != len(public_positions):
                raise ValueError(
                    f"paired position count mismatch at prompt {prompt_id}, layer {layer}"
                )
            native_position_records = [
                _mapping(
                    item,
                    f"native.{prompt_id}.layer-{layer}.position-{index}",
                )
                for index, item in enumerate(native_positions)
            ]
            public_position_records = [
                _mapping(
                    item,
                    f"public.{prompt_id}.layer-{layer}.position-{index}",
                )
                for index, item in enumerate(public_positions)
            ]
            if [
                item.get("token_position") for item in native_position_records
            ] != resolved_positions:
                raise ValueError(
                    f"native position set mismatch at prompt {prompt_id}, layer {layer}"
                )
            if [
                item.get("token_position") for item in public_position_records
            ] != resolved_positions:
                raise ValueError(
                    f"public position set mismatch at prompt {prompt_id}, layer {layer}"
                )

            methods = layer_observations.setdefault(
                layer,
                {
                    "native_jacobian_lens": [],
                    "public_jacobian_lens": [],
                    "native_logit_lens": [],
                    "public_logit_lens": [],
                },
            )
            for native_position, public_position in zip(
                native_position_records, public_position_records, strict=True
            ):
                identity = (
                    native_position.get("capture_index"),
                    native_position.get("token_position"),
                )
                public_identity = (
                    public_position.get("capture_index"),
                    public_position.get("token_position"),
                )
                if identity != public_identity:
                    raise ValueError(
                        f"paired position mismatch at prompt {prompt_id}, layer {layer}"
                    )
                capture_index, token_position = identity
                if (
                    isinstance(capture_index, bool)
                    or not isinstance(capture_index, int)
                    or not 0 <= capture_index < len(capture_positions)
                    or capture_positions[capture_index] != token_position
                ):
                    raise ValueError(f"invalid capture index at prompt {prompt_id}, layer {layer}")
                if not 0 <= token_position < len(prompt_token_ids):
                    raise ValueError(f"invalid token position at prompt {prompt_id}, layer {layer}")
                target_id, target_token = expected_target(token_position)
                native_jacobian = validate_readout(
                    native_position.get("jacobian_lens"),
                    f"native.{prompt_id}.layer-{layer}.jacobian_lens",
                    target_id=target_id,
                    target_token=target_token,
                )
                public_jacobian = validate_readout(
                    public_position.get("jacobian_lens"),
                    f"public.{prompt_id}.layer-{layer}.jacobian_lens",
                    target_id=target_id,
                    target_token=target_token,
                )
                methods["native_jacobian_lens"].append(native_jacobian)
                methods["public_jacobian_lens"].append(public_jacobian)

                if "logit_lens" not in native_position or "logit_lens" not in public_position:
                    raise ValueError(
                        "paired exact logit baseline is required at "
                        f"prompt {prompt_id}, layer {layer}"
                    )
                native_logit = validate_readout(
                    native_position["logit_lens"],
                    f"native.{prompt_id}.layer-{layer}.logit_lens",
                    target_id=target_id,
                    target_token=target_token,
                )
                public_logit = validate_readout(
                    public_position["logit_lens"],
                    f"public.{prompt_id}.layer-{layer}.logit_lens",
                    target_id=target_id,
                    target_token=target_token,
                )
                if native_logit != public_logit:
                    raise ValueError(
                        f"paired logit baseline differs at prompt {prompt_id}, layer {layer}"
                    )
                methods["native_logit_lens"].append(native_logit)
                methods["public_logit_lens"].append(public_logit)

    if native_lens["source_layers"] != expected_layer_order or public_lens[
        "source_layers"
    ] != expected_layer_order:
        raise ValueError("lens source_layers do not match the observed layer grid")
    if native_lens["d_model"] != native_model["model_info"]["hidden_size"] or public_lens[
        "d_model"
    ] != public_model["model_info"]["hidden_size"]:
        raise ValueError("lens d_model does not match the observed model geometry")

    layer_summaries = []
    all_methods = {name: [] for name in layer_observations[expected_layer_order[0]]}
    for layer in expected_layer_order:
        methods = layer_observations[layer]
        for name, records in methods.items():
            all_methods[name].extend(records)
        method_summary = {
            "native_jacobian_lens": method_target_metrics(methods["native_jacobian_lens"]),
            "public_jacobian_lens": method_target_metrics(methods["public_jacobian_lens"]),
        }
        comparisons = {
            "native_vs_public_jacobian_lens": compare_method_records(
                methods["native_jacobian_lens"], methods["public_jacobian_lens"]
            )
        }
        method_summary.update(
            {
                "native_logit_lens": method_target_metrics(methods["native_logit_lens"]),
                "public_logit_lens": method_target_metrics(methods["public_logit_lens"]),
            }
        )
        comparisons.update(
            {
                "native_jacobian_vs_native_logit": compare_method_records(
                    methods["native_jacobian_lens"], methods["native_logit_lens"]
                ),
                "public_jacobian_vs_public_logit": compare_method_records(
                    methods["public_jacobian_lens"], methods["public_logit_lens"]
                ),
                "native_vs_public_logit_lens": compare_method_records(
                    methods["native_logit_lens"], methods["public_logit_lens"]
                ),
            }
        )
        layer_summaries.append(
            {
                "layer": layer,
                "layer_type": layer_types[layer],
                "observations": len(methods["native_jacobian_lens"]),
                "methods": method_summary,
                "comparisons": comparisons,
            }
        )

    overall_methods = {
        "native_jacobian_lens": method_target_metrics(all_methods["native_jacobian_lens"]),
        "public_jacobian_lens": method_target_metrics(all_methods["public_jacobian_lens"]),
    }
    overall_comparisons = {
        "native_vs_public_jacobian_lens": compare_method_records(
            all_methods["native_jacobian_lens"], all_methods["public_jacobian_lens"]
        )
    }
    overall_methods.update(
        {
            "native_logit_lens": method_target_metrics(all_methods["native_logit_lens"]),
            "public_logit_lens": method_target_metrics(all_methods["public_logit_lens"]),
        }
    )
    overall_comparisons.update(
        {
            "native_jacobian_vs_native_logit": compare_method_records(
                all_methods["native_jacobian_lens"], all_methods["native_logit_lens"]
            ),
            "public_jacobian_vs_public_logit": compare_method_records(
                all_methods["public_jacobian_lens"], all_methods["public_logit_lens"]
            ),
            "native_vs_public_logit_lens": compare_method_records(
                all_methods["native_logit_lens"], all_methods["public_logit_lens"]
            ),
        }
    )

    native_adapter = _adapter_certificate(native, "native")
    public_adapter = _adapter_certificate(public, "public")
    if native_adapter["experiments"] != public_adapter["experiments"]:
        raise ValueError("paired lens-independent adapter diagnostics differ")
    if native_adapter["report_status"] != public_adapter["report_status"]:
        raise ValueError("paired report status differs")
    if native_adapter["assertions"] != public_adapter["assertions"]:
        raise ValueError("paired report assertions differ")
    observation_count = len(all_methods["native_jacobian_lens"])
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "offline paired NVFP4 readout comparison; adapter certificates are "
            "preserved independently from lens metrics"
        ),
        "inputs": {
            "native_lens": _lens_identity(native, "native"),
            "public_lens": _lens_identity(public, "public"),
        },
        "pairing": {
            "model": deepcopy(native_model),
            "runtime": deepcopy(native_runtime),
            "host": deepcopy(native_host),
            "prompt_ids": prompt_ids,
            "prompt_count": len(prompt_ids),
            "source_layers": expected_layer_order,
            "layer_count": len(expected_layer_order),
            "positions_by_prompt": positions_by_prompt,
            "residual_capture_manifests": residual_manifests,
            "top_k": top_k,
            "score_encoding": SCORE_ENCODING,
            "observation_count": observation_count,
            "logit_baseline_present": True,
        },
        "adapter_certificates": {
            "native": native_adapter,
            "public": public_adapter,
            "paired_report_status_identical": (
                native_adapter["report_status"] == public_adapter["report_status"]
            ),
            "paired_assertions_identical": (
                native_adapter["assertions"] == public_adapter["assertions"]
            ),
            "paired_diagnostics_identical": True,
        },
        "metrics": {
            "overall": {
                "observations": observation_count,
                "methods": overall_methods,
                "comparisons": overall_comparisons,
            },
            "macro_layer_comparisons": _macro_comparisons(layer_summaries),
            "layers": layer_summaries,
        },
    }


def compare_report_files(native_path: Path, public_path: Path) -> dict[str, Any]:
    native_record = _read_report_file(native_path)
    public_record = _read_report_file(public_path)
    if (native_record["device"], native_record["inode"]) == (
        public_record["device"],
        public_record["inode"],
    ):
        raise ValueError("native and public report inputs must be different files")
    if native_record["sha256"] == public_record["sha256"]:
        raise ValueError("native and public report inputs must have different bytes")
    result = compare_reports(native_record["report"], public_record["report"])
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["input_files"] = {
        "native": {
            "path": native_record["path"],
            "sha256": native_record["sha256"],
            "size_bytes": native_record["size_bytes"],
        },
        "public": {
            "path": public_record["path"],
            "sha256": public_record["sha256"],
            "size_bytes": public_record["size_bytes"],
        },
    }
    return result


def atomic_write_json(path: Path, value: Any) -> None:
    rendered = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    path = path.resolve()
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
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output.resolve() in {
        args.native_report.resolve(),
        args.public_report.resolve(),
    }:
        raise ValueError("output must not overwrite an input report")
    result = compare_report_files(args.native_report, args.public_report)
    atomic_write_json(args.output, result)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
