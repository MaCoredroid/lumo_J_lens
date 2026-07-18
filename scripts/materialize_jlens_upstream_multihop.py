#!/usr/bin/env python3
"""Materialize Anthropic's pinned multihop lens evaluation for Qwen3.6."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
UPSTREAM_SOURCE_SHA256 = "50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416"
UPSTREAM_RELATIVE_SOURCE = Path("data/evaluations/lens-eval-multihop.json")
DEFAULT_OUTPUT = ROOT / ".cache/jlens_upstream_multihop"
MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
TOKENIZER_JSON_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
MODEL_CONFIG_SHA256 = "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338"
LOGIT_VOCABULARY_SIZE = 248_320


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
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


def verified_forms(tokenizer: Any, intermediate: str) -> dict[str, Any]:
    forms: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for form_kind, text in (("bare", intermediate), ("leading_space", f" {intermediate}")):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            exclusions.append({"form": form_kind, "text": text, "token_ids": token_ids, "reason": "not_exactly_one_token"})
            continue
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != text:
            exclusions.append({"form": form_kind, "text": text, "token_ids": token_ids, "decoded": decoded, "reason": "decode_roundtrip_mismatch"})
            continue
        forms.append({"form": form_kind, "text": text, "token_id": token_ids[0]})
    return {"text": intermediate, "eligible_forms": forms, "excluded_forms": exclusions, "scorable": bool(forms)}


def build_bundle(source: Mapping[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = source.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("upstream multihop source requires nonempty items")
    prompts: list[dict[str, Any]] = []
    names: set[str] = set()
    union: dict[int, str] = {}
    occurrence_count = scorable_count = 0
    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise ValueError(f"upstream item {index} must be an object")
        name, text, target, intermediates = (raw.get(key) for key in ("name", "prompt", "target", "intermediates"))
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"upstream item {index} has invalid name")
        if not isinstance(text, str) or not text or not isinstance(target, str) or not target:
            raise ValueError(f"upstream item {name} has invalid prompt/target")
        if not isinstance(intermediates, list) or not intermediates or any(not isinstance(item, str) or not item for item in intermediates):
            raise ValueError(f"upstream item {name} has invalid intermediates")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != text:
            raise ValueError(f"upstream prompt {name} does not round-trip exactly")
        intermediate_records = [verified_forms(tokenizer, item) for item in intermediates]
        item_score_token_ids: set[int] = set()
        for record in intermediate_records:
            occurrence_count += 1
            scorable_count += int(record["scorable"])
            for form in record["eligible_forms"]:
                token_id = form["token_id"]
                item_score_token_ids.add(token_id)
                if token_id in union and union[token_id] != form["text"]:
                    # One vocabulary ID has one canonical decode; aliases must agree.
                    raise ValueError(f"token {token_id} has inconsistent decoded forms")
                union[token_id] = form["text"]
        names.add(name)
        prompt = {
            "id": f"upstream-multihop-{index:03d}-{name}",
            "text": text,
            "token_ids": token_ids,
            "metadata": {
                "kind": "anthropic_jlens_multihop_qwen36_control",
                "upstream": {"commit": UPSTREAM_COMMIT, "source_sha256": UPSTREAM_SOURCE_SHA256, "item_index": index, "name": name, "target": target, "intermediates": intermediate_records},
                "tokenizer": {"repo_id": MODEL_REPO, "revision": MODEL_REVISION, "tokenizer_json_sha256": TOKENIZER_JSON_SHA256},
            },
        }
        if item_score_token_ids:
            prompt["score_token_ids"] = sorted(item_score_token_ids)
        prompts.append(prompt)
    manifest = {
        "schema_version": 1,
        "kind": "anthropic_jlens_multihop_qwen36_materialization",
        "upstream": {"repository": "anthropics/jacobian-lens", "commit": UPSTREAM_COMMIT, "relative_path": str(UPSTREAM_RELATIVE_SOURCE), "source_sha256": UPSTREAM_SOURCE_SHA256},
        "model": {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "tokenizer_vocabulary_size": len(tokenizer),
            "config_sha256": MODEL_CONFIG_SHA256,
            "logit_vocabulary_size": LOGIT_VOCABULARY_SIZE,
        },
        "metric_contract": {"fixed_middle_layers": list(range(24, 48)), "secondary_all_layers": list(range(63)), "unscorable_intermediate_policy": "count_as_miss_to_preserve_all_upstream_item/intermediate_denominators"},
        "coverage": {"item_count": len(prompts), "intermediate_occurrence_count": occurrence_count, "scorable_intermediate_occurrence_count": scorable_count, "excluded_intermediate_occurrence_count": occurrence_count - scorable_count},
        "scored_vocabulary": {"token_ids": sorted(union), "tokens": [union[token_id] for token_id in sorted(union)]},
    }
    return prompts, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, help="local pinned multihop JSON")
    parser.add_argument("--upstream-repo", type=Path, help="local anthropics/jacobian-lens checkout")
    parser.add_argument("--allow-download", action="store_true", help="explicitly allow fetching the file from the pinned raw GitHub commit")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected_sources = sum((args.source_file is not None, args.upstream_repo is not None, args.allow_download))
    if selected_sources != 1:
        raise ValueError("select exactly one of --source-file, --upstream-repo, or --allow-download")
    if args.source_file is not None:
        source_bytes = args.source_file.expanduser().resolve(strict=True).read_bytes()
    elif args.upstream_repo is not None:
        upstream = args.upstream_repo.expanduser().resolve(strict=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
        if commit != UPSTREAM_COMMIT:
            raise ValueError(f"upstream commit mismatch: {commit}")
        source_bytes = (upstream / UPSTREAM_RELATIVE_SOURCE).resolve(strict=True).read_bytes()
    else:
        from urllib.request import urlopen
        url = f"https://raw.githubusercontent.com/anthropics/jacobian-lens/{UPSTREAM_COMMIT}/{UPSTREAM_RELATIVE_SOURCE}"
        with urlopen(url, timeout=30) as response:
            source_bytes = response.read()
    if hashlib.sha256(source_bytes).hexdigest() != UPSTREAM_SOURCE_SHA256:
        raise ValueError("upstream multihop source SHA-256 mismatch")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    snapshot = Path(args.model_snapshot or snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)).resolve(strict=True)
    if sha256_file(snapshot / "tokenizer.json") != TOKENIZER_JSON_SHA256:
        raise ValueError("Qwen tokenizer.json SHA-256 mismatch")
    config_path = snapshot / "config.json"
    if sha256_file(config_path) != MODEL_CONFIG_SHA256:
        raise ValueError("Qwen config.json SHA-256 mismatch")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    if model_config["text_config"]["vocab_size"] != LOGIT_VOCABULARY_SIZE:
        raise ValueError("Qwen LM-head vocabulary size changed")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    source = json.loads(source_bytes)
    prompts, manifest = build_bundle(source, tokenizer)
    output = args.output_dir.expanduser().resolve()
    prompts_path = output / "prompts.json"
    source_copy = output / "lens-eval-multihop.upstream.json"
    manifest_path = output / "manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    source_copy.write_bytes(source_bytes)
    atomic_write_json(prompts_path, prompts)
    manifest["outputs"] = {
        "source_copy": {"path": source_copy.name, "size_bytes": source_copy.stat().st_size, "sha256": sha256_file(source_copy)},
        "prompts": {"path": prompts_path.name, "size_bytes": prompts_path.stat().st_size, "sha256": sha256_file(prompts_path)},
    }
    atomic_write_json(manifest_path, manifest)
    print(f"wrote {len(prompts)} prompts, {len(manifest['scored_vocabulary']['token_ids'])} scored token IDs: {output}")
    print(f"excluded single-token intermediate occurrences: {manifest['coverage']['excluded_intermediate_occurrence_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
