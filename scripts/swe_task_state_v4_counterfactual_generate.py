#!/usr/bin/env python3
"""Generate blinded V4 counterfactual completions only after capture passes."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_jlens_nvfp4 as base  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
SHELL_WRAPPER_PATH = ROOT / "scripts" / "run_swe_task_state_v4_counterfactual_generate.sh"
SCHEMA_VERSION = 1
KIND = "swe_task_state_v4_counterfactual_blinded_completions"
MAX_NEW_TOKENS = 256
FORBIDDEN_PATH_FRAGMENTS = ("reserved", "validation")
EVIDENCE_LEVELS = ("clear_success", "clear_failure", "contradictory_ambiguous")
PRESSURE_LEVELS = ("neutral", "fixed_time_or_token_pressure")


class GenerationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GenerationError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def lexical_path_preflight(paths: Iterable[Path]) -> None:
    for path in paths:
        normalized = os.path.normpath(os.path.abspath(os.fspath(path))).lower()
        if any(
            fragment in component
            for fragment in FORBIDDEN_PATH_FRAGMENTS
            for component in Path(normalized).parts
        ):
            raise GenerationError(f"forbidden path rejected before filesystem access: {path}")


def canonical_path_preflight(
    *, input_paths: Iterable[Path], output_paths: Iterable[Path]
) -> None:
    for path, strict in [
        *((path, True) for path in input_paths),
        *((path, False) for path in output_paths),
    ]:
        resolved = path.resolve(strict=strict)
        if any(
            fragment in component.lower()
            for fragment in FORBIDDEN_PATH_FRAGMENTS
            for component in resolved.parts
        ):
            raise GenerationError(f"forbidden canonical path rejected: {path}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GenerationError(f"cannot load JSON {path}: {error}") from error


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        display = str(resolved.relative_to(ROOT))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": sha256_file(resolved), "size_bytes": resolved.stat().st_size}


def validate_capture_bundle(value: Any) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and len(value) == 12, "capture bundle must have 12 rows")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(value):
        _require(
            isinstance(row, dict) and set(row) == {"id", "token_ids"},
            f"capture row {index} is not blinded",
        )
        prompt_id = row["id"]
        token_ids = row["token_ids"]
        _require(
            _is_sha256(prompt_id)
            and prompt_id not in seen
            and isinstance(token_ids, list)
            and 12000 <= len(token_ids) <= 14000
            and all(
                isinstance(token_id, int)
                and not isinstance(token_id, bool)
                and token_id >= 0
                for token_id in token_ids
            ),
            f"capture row {index} identity or tokens changed",
        )
        seen.add(prompt_id)
        rows.append({"id": prompt_id, "token_ids": list(token_ids)})
    _require({len(row["token_ids"]) for row in rows} == {12694, 12697}, "prompt matching changed")
    return rows


def validate_materialization(
    value: Any, *, capture_path: Path, capture_sha256: str
) -> None:
    _require(
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and value.get("kind") == "swe_task_state_v4_counterfactual_pilot_materialization"
        and value.get("status")
        == "passed_materialization_only_capture_and_generation_not_started",
        "materialization manifest identity changed",
    )
    record = value.get("outputs", {}).get("capture_bundle", {})
    _require(
        record.get("sha256") == capture_sha256
        and record.get("size_bytes") == capture_path.stat().st_size
        and record.get("prompt_count") == 12
        and record.get("allowed_row_keys") == ["id", "token_ids"],
        "materialization does not bind the blinded capture bundle",
    )
    execution = value.get("execution_state", {})
    _require(
        execution.get("counterfactual_completion_generation_completed") is False
        and execution.get("completion_label_extraction_completed") is False,
        "materialization is no longer pre-generation",
    )


def validate_capture_manifest(
    value: Any,
    *,
    capture_path: Path,
    capture_sha256: str,
    prompt_rows: Sequence[Mapping[str, Any]],
) -> None:
    _require(
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and value.get("kind") == "swe_task_state_v4_label_independent_public_j_state_capture"
        and value.get("status") == "passed"
        and value.get("status_scope") == "raw_and_public_j_pre_vocabulary_state_capture_only",
        "authenticated capture has not passed",
    )
    source = value.get("source_bundle", {})
    _require(
        source.get("sha256") == capture_sha256
        and source.get("size_bytes") == capture_path.stat().st_size
        and source.get("prompt_count") == 12,
        "capture manifest does not bind the blinded prompt bundle",
    )
    summary = value.get("summary", {})
    boundaries = value.get("boundaries")
    _require(
        summary.get("all_capture_valid") is True
        and summary.get("boundary_count") == 12
        and summary.get("reserved_validation_access_authorized") is False
        and isinstance(boundaries, list)
        and len(boundaries) == 12,
        "authenticated capture completeness changed",
    )
    for index, (prompt, boundary) in enumerate(zip(prompt_rows, boundaries, strict=True)):
        expected_source = sha256_bytes(str(prompt["id"]).encode("utf-8"))
        expected_tokens = sha256_bytes(canonical_json_bytes(prompt["token_ids"]))
        _require(
            isinstance(boundary, dict)
            and boundary.get("index") == index
            and boundary.get("source_id_sha256") == expected_source
            and boundary.get("token_ids_sha256") == expected_tokens
            and boundary.get("token_count") == len(prompt["token_ids"])
            and boundary.get("reference_residual_manifest_equal") is True
            and boundary.get("capture_valid") is True
            and boundary.get("shard", {}).get("reload_verified") is True,
            f"authenticated capture boundary {index} changed",
        )
    cli = value.get("normalized_cli_contract", {})
    _require(
        cli.get("source_bundle") == str(capture_path.resolve().relative_to(ROOT))
        and all("condition-key" not in str(item) for item in cli.values()),
        "capture CLI was not condition-key blind",
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-prompts", type=Path, required=True)
    parser.add_argument("--capture-prompts-sha256", required=True)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--materialization-manifest-sha256", required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--capture-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    inputs = (
        args.capture_prompts,
        args.materialization_manifest,
        args.capture_manifest,
        SCRIPT_PATH,
        SHELL_WRAPPER_PATH,
    )
    lexical_path_preflight((*inputs, args.output))
    canonical_path_preflight(input_paths=inputs, output_paths=(args.output,))
    for value, label in (
        (args.capture_prompts_sha256, "capture prompt hash"),
        (args.materialization_manifest_sha256, "materialization hash"),
        (args.capture_manifest_sha256, "capture manifest hash"),
    ):
        _require(_is_sha256(value), f"{label} is invalid")
    _require(
        sha256_file(args.capture_prompts) == args.capture_prompts_sha256,
        "capture prompt bytes changed",
    )
    _require(
        sha256_file(args.materialization_manifest)
        == args.materialization_manifest_sha256,
        "materialization bytes changed",
    )
    _require(
        sha256_file(args.capture_manifest) == args.capture_manifest_sha256,
        "capture manifest bytes changed",
    )
    rows = validate_capture_bundle(load_json(args.capture_prompts))
    validate_materialization(
        load_json(args.materialization_manifest),
        capture_path=args.capture_prompts,
        capture_sha256=args.capture_prompts_sha256,
    )
    validate_capture_manifest(
        load_json(args.capture_manifest),
        capture_path=args.capture_prompts,
        capture_sha256=args.capture_prompts_sha256,
        prompt_rows=rows,
    )

    from huggingface_hub import snapshot_download
    import torch
    from vllm import LLM, SamplingParams, TokensPrompt

    model_path = Path(
        snapshot_download(base.MODEL_REPO, revision=base.MODEL_REVISION, local_files_only=True)
    )
    checkpoint, checkpoint_record = base.open_pinned_model_checkpoint(model_path)
    torch.cuda.reset_peak_memory_stats()
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    load_started = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        dtype="bfloat16",
        quantization="modelopt_fp4",
        gpu_memory_utilization=0.78,
        max_model_len=16384,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        language_model_only=True,
        gdn_prefill_backend="triton",
        mamba_block_size=1024,
        mamba_cache_mode="align",
        mamba_ssm_cache_dtype="float32",
        kv_cache_dtype="fp8_e4m3",
        attention_backend="TRITON_ATTN",
        limit_mm_per_prompt={"image": 0, "video": 0},
        enable_flashinfer_autotune=False,
        async_scheduling=False,
        seed=0,
    )
    model_load_seconds = time.perf_counter() - load_started
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0, seed=0)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prompt_started = time.perf_counter()
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=row["token_ids"])], sampling, use_tqdm=False
        )
        generation_seconds = time.perf_counter() - prompt_started
        output = outputs[0]
        _require(output.prompt_token_ids == row["token_ids"], "vLLM changed prompt tokens")
        completion = output.outputs[0]
        generated_ids = list(completion.token_ids)
        _require(generated_ids, "empty counterfactual completion")
        generated_text = completion.text
        _require(isinstance(generated_text, str), "vLLM completion text is invalid")
        full_token_decode = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        visible_token_decode = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        records.append(
            {
                "index": index,
                "prompt_id": row["id"],
                "capture_source_id_sha256": sha256_bytes(row["id"].encode("utf-8")),
                "prompt_token_ids_sha256": sha256_bytes(canonical_json_bytes(row["token_ids"])),
                "prompt_token_count": len(row["token_ids"]),
                "generated_token_ids": generated_ids,
                "generated_token_ids_sha256": sha256_bytes(canonical_json_bytes(generated_ids)),
                "generated_token_count": len(generated_ids),
                "generated_text": generated_text,
                "generated_text_sha256": sha256_bytes(generated_text.encode("utf-8")),
                "generated_full_token_decode": full_token_decode,
                "generated_full_token_decode_sha256": sha256_bytes(
                    full_token_decode.encode("utf-8")
                ),
                "generated_visible_token_decode": visible_token_decode,
                "generated_visible_token_decode_sha256": sha256_bytes(
                    visible_token_decode.encode("utf-8")
                ),
                "vllm_text_equals_full_token_decode": generated_text == full_token_decode,
                "vllm_text_equals_visible_token_decode": generated_text
                == visible_token_decode,
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
                "generation_seconds": round(generation_seconds, 6),
            }
        )
    base.revalidate_pinned_model_checkpoint(checkpoint, checkpoint_record)
    completed_at = datetime.now(timezone.utc)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "passed_blinded_generation_after_authenticated_capture",
        "scope": "externally_observable_completion_text_only_no_labels_or_condition_key",
        "inputs": {
            "capture_prompts": _file_record(args.capture_prompts),
            "materialization_manifest": _file_record(args.materialization_manifest),
            "authenticated_capture_manifest": _file_record(args.capture_manifest),
        },
        "implementation": {
            "generator": _file_record(SCRIPT_PATH),
            "shell_wrapper": _file_record(SHELL_WRAPPER_PATH),
        },
        "model": {
            "repo_id": base.MODEL_REPO,
            "revision": base.MODEL_REVISION,
            "checkpoint_integrity": checkpoint_record,
        },
        "generation": {
            "temperature": 0,
            "seed": 0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt_order": "frozen_randomized_condition_order",
            "max_num_seqs": 1,
            "condition_key_read": False,
            "completion_labels_read_or_extracted": False,
        },
        "runtime": {
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "model_load_seconds": round(model_load_seconds, 6),
            "gpu": base._nvidia_smi(),
            "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "completion_count": len(records),
        "records": records,
        "execution_order": {
            "authenticated_capture_verified_before_model_load": True,
            "condition_key_absent_from_process_inputs": True,
            "completion_label_extraction_completed": False,
        },
        "claim_scope": {
            "observable_completions_generated": True,
            "counterfactual_effect_established": False,
            "private_chain_of_thought_reconstructed": False,
            "subjective_emotion_confidence_doubt_or_stress_inferred": False,
        },
        "reserved_validation_access_authorized": False,
    }
    _atomic_write_json(args.output, result)
    print(
        f"wrote {len(records)} blinded post-capture completions to {args.output}",
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
