#!/usr/bin/env python3
"""Build the per-turn cohort boundary prompts-file for the VJP capture (P7c final).

Reconstructs, for every tagged turn of the 20 cohort tasks, the exact end-of-thinking
boundary the model reasoned to, then emits a run_jlens_nvfp4 prompts-file whose scored
tokens are the task-independent general concept-form vocab.

Each turn's prompt is stitched from two aligned sources:
  * fixed agent framing (the 42.5k-char qwen-code system prompt + the task's initial
    user message) taken from the rerun proxy dumps -- run-independent, so it faithfully
    stands in for the original run's framing, which the trace omits; and
  * the original (thinking-on, already-tagged) trace's turn-by-turn assistant actions
    and tool results, walked in order, with the current turn's exact thinking appended
    to the assistant generation prompt to land the boundary on the last thinking token.
Prior-turn thinking is dropped from history -- the qwen-code agent strips it (confirmed
by the dumps), so the reconstruction matches what the model actually saw. Read-only over
run artifacts; needs the tokenizer only (no GPU).
"""

from __future__ import annotations

import importlib.util
import json
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERAL_VOCAB = ROOT / "configs/swe_task_state_v4_general_concept_forms.json"
TAGS = ROOT / ".cache/swe_jlens_cohort/cohort_cot_tags.json"
RERUN_DUMPS = ROOT / "runs/swe_jlens_rerun/proxy_dumps"
RERUN_LOG = ROOT / ".cache/sealed-run-driver-logs/latest-rerun.txt"
DEFAULT_OUTPUT = ROOT / ".cache/swe_jlens_cohort/cohort-perturn-prompts.json"
TOKENIZE_URL = "http://127.0.0.1:9952/tokenize"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CT = _load("cohort_traces", "scripts/swe_task_state_v4_cohort_traces.py")


def _tokenizer():
    from transformers import AutoTokenizer

    snap = next(
        Path.home().glob(
            ".cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/*"
        )
    )
    return AutoTokenizer.from_pretrained(str(snap), trust_remote_code=False)


def _rerun_task_order() -> list[str]:
    import re

    log = Path(RERUN_LOG.read_text().strip())
    order, seen = [], set()
    for line in log.read_text().splitlines():
        m = re.search(r"\] -> ([a-zA-Z0-9_.\-]+__[a-zA-Z0-9_.\-]+-\d+)", line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            order.append(m.group(1))
    return order


def _framing_by_task() -> dict[str, dict[str, Any]]:
    """task -> {system, user, tools} from the rerun dumps (system is run-independent)."""
    dumps = sorted(RERUN_DUMPS.glob("chat_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    order = _rerun_task_order()
    segments: list[list[Path]] = []
    for dp in dumps:
        n = len(json.loads(dp.read_text()).get("messages", []))
        if n <= 2 or not segments:
            segments.append([dp])
        else:
            segments[-1].append(dp)
    framing = {}
    for i, seg in enumerate(segments):
        if i >= len(order):
            break
        d0 = json.loads(seg[0].read_text())
        m = d0["messages"]
        framing[order[i]] = {"system": m[0], "user": m[1], "tools": d0.get("tools")}
    return framing


def reconstruct_boundaries(entries: list[dict], framing: dict[str, Any]) -> list[dict]:
    """Walk a trace -> [{turn, messages, thinking}] boundaries (prior msgs + this thinking)."""
    msgs: list[dict] = [framing["system"], framing["user"]]
    out: list[dict] = []
    cur: dict | None = None

    def flush():
        nonlocal cur
        if cur is not None:
            msgs.append(cur)
            cur = None

    turn = 0
    for e in entries:
        t = e.get("type")
        content = e.get("message", {}).get("content") if isinstance(e.get("message"), dict) else None
        if t == "assistant" and isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                pt = p.get("type")
                if pt == "thinking" and p.get("thinking", "").strip():
                    flush()
                    turn += 1
                    out.append({"turn": turn, "messages": list(msgs), "thinking": p["thinking"].strip()})
                    cur = {"role": "assistant", "content": ""}
                elif pt == "text":
                    cur = cur or {"role": "assistant", "content": ""}
                    cur["content"] += p.get("text", "")
                elif pt == "tool_use":
                    cur = cur or {"role": "assistant", "content": ""}
                    cur.setdefault("tool_calls", []).append(
                        {
                            "id": p.get("id"),
                            "type": "function",
                            "function": {"name": p.get("name"), "arguments": json.dumps(p.get("input", {}))},
                        }
                    )
        elif t == "user" and isinstance(content, list):
            flush()
            for p in content:
                if isinstance(p, dict) and p.get("type") == "tool_result":
                    c = p.get("content", "")
                    msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": p.get("tool_use_id"),
                            "content": c if isinstance(c, str) else json.dumps(c),
                        }
                    )
    return out


def _tokenize_messages(messages, tools) -> list[int]:
    """Render + tokenize a chat prompt via the vLLM server (exact production template)."""
    body = json.dumps(
        {
            "messages": messages,
            "tools": tools,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    ).encode()
    req = urllib.request.Request(TOKENIZE_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["tokens"]


def _boundary_ids(tokenizer, messages, tools, thinking) -> list[int]:
    prompt_ids = _tokenize_messages(messages, tools)
    think_ids = tokenizer.encode(thinking, add_special_tokens=False)
    return prompt_ids + think_ids


def build(output: Path = DEFAULT_OUTPUT, *, max_len: int = 32768) -> dict[str, Any]:
    tokenizer = _tokenizer()
    vocab = json.loads(GENERAL_VOCAB.read_text())
    score_ids = sorted({f["token_id"] for forms in vocab["families"].values() for f in forms})
    framing = _framing_by_task()
    tags = {t["task"]: t for t in json.loads(TAGS.read_text())["tasks"]}
    survey = {v["task"]: v for v in _CT.survey()["usable"]}

    prompts, lengths, dropped = [], [], 0
    for task in tags:
        if task not in framing or task not in survey:
            continue
        entries = _CT.read_trace_entries(Path(survey[task]["trace"]))
        boundaries = reconstruct_boundaries(entries, framing[task])
        tag_by_turn = {r["turn"]: r["tag"] for r in tags[task]["turns"]}
        for b in boundaries:
            ids = _boundary_ids(tokenizer, b["messages"], framing[task]["tools"], b["thinking"])
            lengths.append(len(ids))
            if len(ids) > max_len:
                dropped += 1
                continue
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
                        "boundary": "end_of_thinking",
                    },
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(prompts))
    lengths.sort()
    n = len(lengths)

    def pct(q: float) -> int:
        return lengths[min(n - 1, int(q * n))] if n else 0

    return {
        "n_prompts": len(prompts),
        "n_turns_total": n,
        "dropped_over_maxlen": dropped,
        "max_len": max_len,
        "len_p50": pct(0.50),
        "len_p90": pct(0.90),
        "len_p99": pct(0.99),
        "len_max": lengths[-1] if lengths else 0,
        "output": str(output),
    }


def main() -> int:
    print(json.dumps(build(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
