#!/usr/bin/env python3
"""Materialize the pinned Wikitext corpus slice used for the local NF4 fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


DATASET_REPO = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_SPLIT = "train"
DATASET_SPLITS = ("train", "validation", "test")
MODEL_REPO = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_rows(
    rows: Any,
    tokenizer: Any,
    *,
    count: int,
    sequence_length: int,
    min_chars: int,
) -> list[dict[str, Any]]:
    """Select the first qualifying rows and freeze their exact token IDs."""
    selected: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        text = row["text"].strip()
        if len(text) < min_chars:
            continue
        token_ids = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=sequence_length,
        )["input_ids"]
        if len(token_ids) != sequence_length:
            continue
        selected.append(
            {
                "row_index": row_index,
                "text": text,
                "text_sha256": sha256_text(text),
                "token_count": len(token_ids),
                "token_ids": token_ids,
            }
        )
        if len(selected) == count:
            return selected
    raise RuntimeError(
        f"dataset ended after {len(selected)} qualifying rows; expected {count}"
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/jlens_nf4_fit_prompts.json"),
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--min-chars", type=int, default=600)
    parser.add_argument(
        "--split",
        choices=DATASET_SPLITS,
        default=DATASET_SPLIT,
        help="pinned Wikitext split; use validation/test for held-out evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0 or args.sequence_length <= 17 or args.min_chars <= 0:
        raise SystemExit("count/min-chars must be positive and sequence-length must exceed 17")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
    )
    if (
        getattr(tokenizer, "bos_token_id", None) is not None
        and hasattr(tokenizer, "add_bos_token")
    ):
        tokenizer.add_bos_token = True

    rows = load_dataset(
        DATASET_REPO,
        DATASET_CONFIG,
        split=args.split,
        revision=DATASET_REVISION,
        streaming=True,
    )
    prompts = select_rows(
        rows,
        tokenizer,
        count=args.count,
        sequence_length=args.sequence_length,
        min_chars=args.min_chars,
    )
    manifest = {
        "schema_version": 1,
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "config": DATASET_CONFIG,
            "split": args.split,
        },
        "tokenizer": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "add_special_tokens": True,
            "force_bos_when_supported": True,
            "truncation": "right",
        },
        "selection": {
            "order": "dataset row order",
            "minimum_stripped_characters": args.min_chars,
            "required_token_count": args.sequence_length,
            "take": args.count,
        },
        "prompts": prompts,
    }
    atomic_write_json(args.output, manifest)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output} ({len(prompts)} prompts, sha256={digest})")


if __name__ == "__main__":
    main()
