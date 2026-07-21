#!/usr/bin/env python3
"""Lever 3: bounded-context per-turn prompts for ALL 535 cohort turns (P7c).

The v3 build reconstructed the FULL conversation per turn, so 242 long/late turns exceeded
the 32768 window and were dropped -- biasing the cohort toward early turns. Here we keep the
same faithful reconstruction but BOUND the context: system + task + the most-recent whole
turn-blocks that fit a token cap + the current turn's exact thinking. Truncation is only at
turn boundaries (never orphaning a tool_result from its tool_call), so every turn -- including
the 242 -- yields a valid boundary. The end-of-thinking residual is dominated by recent context
+ the thinking, so bounded context is a faithful-enough approximation; the shared turns let us
check bounded-vs-full directly. Renders the codex template locally (no server); uses the v2
vocab as scored tokens. CPU-only, no GPU.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, BaseLoader
from jinja2.exceptions import TemplateError


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/qwen3-openai-codex.jinja"
VOCAB_V2 = ROOT / "configs/swe_task_state_v4_general_concept_forms_v2.json"
TAGS = ROOT / ".cache/swe_jlens_cohort/cohort_cot_tags.json"
DEFAULT_OUTPUT = ROOT / ".cache/swe_jlens_cohort/cohort-perturn-prompts-bounded.json"
CAP = 30000  # keep every boundary under the 32768 capture window with headroom


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BUILD = _load("build_cohort_prompts", "scripts/swe_task_state_v4_build_cohort_prompts.py")
_CT = _load("cohort_traces", "scripts/swe_task_state_v4_cohort_traces.py")


def _renderer():
    env = Environment(loader=BaseLoader())
    env.filters["from_json"] = json.loads
    env.filters["tojson"] = lambda v, indent=None, ensure_ascii=False: json.dumps(
        v, indent=indent, ensure_ascii=ensure_ascii
    )
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(TemplateError(m))
    return env.from_string(TEMPLATE.read_text())


def _tokenizer():
    from transformers import AutoTokenizer

    snap = next(Path.home().glob(".cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/*"))
    return AutoTokenizer.from_pretrained(str(snap), trust_remote_code=False)


def _blocks(body: list[dict]) -> list[list[dict]]:
    """Split reconstructed body messages into assistant-led turn blocks."""
    blocks: list[list[dict]] = []
    for m in body:
        if m.get("role") == "assistant" or not blocks:
            blocks.append([m])
        else:
            blocks[-1].append(m)
    return blocks


def _approx_cost(tokenizer, m: dict) -> int:
    c = m.get("content")
    text = c if isinstance(c, str) else json.dumps(c)
    n = len(tokenizer.encode(text, add_special_tokens=False)) + 8
    for tc in m.get("tool_calls", []) or []:
        n += len(tokenizer.encode(json.dumps(tc.get("function", {})), add_special_tokens=False)) + 8
    return n


def _bounded_ids(render, tokenizer, messages, tools, thinking, cap: int = CAP) -> tuple[list[int], int]:
    """Render system+task + most-recent whole blocks that fit, + thinking. Returns (ids, n_blocks_kept)."""
    head, body = messages[:2], messages[2:]
    blocks = _blocks(body)
    think_ids = tokenizer.encode(thinking, add_special_tokens=False)

    def render_ids(kept_blocks):
        msgs = head + [m for b in kept_blocks for m in b]
        text = render.render(messages=msgs, tools=tools, add_generation_prompt=True, enable_thinking=True)
        return tokenizer.encode(text, add_special_tokens=False) + think_ids

    # approximate budget: keep a suffix of blocks, then shrink until the exact render fits
    kept: list[list[dict]] = []
    budget = cap - (_approx_cost(tokenizer, head[0]) + _approx_cost(tokenizer, head[1]) + len(think_ids) + 256)
    running = 0
    for blk in reversed(blocks):
        cost = sum(_approx_cost(tokenizer, m) for m in blk)
        if running + cost > budget and kept:
            break
        running += cost
        kept.insert(0, blk)
    ids = render_ids(kept)
    while len(ids) > cap and kept:
        kept = kept[1:]
        ids = render_ids(kept)
    return ids, len(kept)


def build(output: Path = DEFAULT_OUTPUT, *, cap: int = CAP) -> dict[str, Any]:
    render = _renderer()
    tokenizer = _tokenizer()
    vocab = json.loads(VOCAB_V2.read_text())
    score_ids = sorted({f["token_id"] for forms in vocab["families"].values() for f in forms})
    framing = _BUILD._framing_by_task()
    tags = {t["task"]: t for t in json.loads(TAGS.read_text())["tasks"]}
    survey = {v["task"]: v for v in _CT.survey()["usable"]}

    prompts, lengths, blocks_kept, dropped = [], [], [], 0
    for task in tags:
        if task not in framing or task not in survey:
            continue
        entries = _CT.read_trace_entries(Path(survey[task]["trace"]))
        boundaries = _BUILD.reconstruct_boundaries(entries, framing[task])
        tag_by_turn = {r["turn"]: r["tag"] for r in tags[task]["turns"]}
        for b in boundaries:
            ids, nkept = _bounded_ids(render, tokenizer, b["messages"], framing[task]["tools"], b["thinking"], cap)
            if len(ids) > cap:  # even system+task+thinking alone overflow (rare)
                dropped += 1
                continue
            lengths.append(len(ids))
            blocks_kept.append(nkept)
            prompts.append(
                {
                    "id": f"{task}::turn{b['turn']:03d}",
                    "token_ids": ids,
                    "score_token_ids": score_ids,
                    "metadata": {
                        "task": task,
                        "turn": b["turn"],
                        "tag": tag_by_turn.get(b["turn"]),
                        "n_tokens": len(ids),
                        "blocks_kept": nkept,
                        "boundary": "end_of_thinking",
                        "context": "bounded",
                    },
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(prompts))
    lengths.sort()
    return {
        "n_prompts": len(prompts),
        "dropped": dropped,
        "cap": cap,
        "n_score_tokens": len(score_ids),
        "len_p50": lengths[len(lengths) // 2] if lengths else 0,
        "len_max": lengths[-1] if lengths else 0,
        "blocks_kept_min": min(blocks_kept) if blocks_kept else 0,
        "blocks_kept_p50": sorted(blocks_kept)[len(blocks_kept) // 2] if blocks_kept else 0,
        "output": str(output),
    }


def main() -> int:
    print(json.dumps(build(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
