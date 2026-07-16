#!/usr/bin/env python3
"""Materialize selected SWE-bench rows from an immutable dataset revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested = set(args.instance_id)
    dataset = load_dataset(args.dataset, revision=args.revision, split=args.split)
    rows = [dict(row) for row in dataset if row["instance_id"] in requested]
    found = {row["instance_id"] for row in rows}
    missing = sorted(requested - found)
    if missing:
        raise SystemExit(f"instances missing at revision {args.revision}: {missing}")
    rows.sort(key=lambda row: args.instance_id.index(row["instance_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"materialized {len(rows)} row(s) at {args.revision}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
