#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import re
from pathlib import Path
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Monotonic per-request index for the forced per-request seed. itertools.count()
# .__next__ is atomic under CPython, so it is safe to advance from the
# ThreadingHTTPServer worker threads without a lock.
_SEED_COUNTER = itertools.count()

# Env vars -> chat/completions sampling keys for the reference-envelope pins.
# (name, payload_key, caster) — mirrors the flywheel inference_proxy.py
# LUMO_PROXY_FORCE_* pattern, ported onto the OpenAI chat schema's top level.
_ENVELOPE_PINS = (
    ("LUMO_PROXY_FORCE_TEMPERATURE", "temperature", float),
    ("LUMO_PROXY_FORCE_TOP_P", "top_p", float),
    ("LUMO_PROXY_FORCE_TOP_K", "top_k", int),
    ("LUMO_PROXY_FORCE_MIN_P", "min_p", float),
    ("LUMO_PROXY_FORCE_PRESENCE_PENALTY", "presence_penalty", float),
)


def apply_reference_envelope(payload, *, env=None, seed_index=None):
    """Env-gated reference-envelope sampling pins (flywheel LUMO_PROXY_FORCE_*
    pattern, ported to the OpenAI /v1/chat/completions schema).

    The certified Qwen3.6 thinking envelope is temperature 1.0 / top_p 0.95 /
    top_k 20 / min_p 0 / presence_penalty 0, seeded per request. qwen-code
    itself sends different defaults, so the pins overwrite whatever the client
    sent. The values remain environment-driven to keep this proxy reusable.

    Default-OFF contract: with NONE of the LUMO_PROXY_FORCE_* vars set this
    returns the payload UNCHANGED (byte-identical passthrough), which preserves
    the greedy A/B baseline the corrected-ladder premise is measured against.

    Per-request seed: a client-supplied `seed` is left untouched (passthrough).
    Otherwise, if LUMO_PROXY_FORCE_SEED is set, stamps base_seed + seed_index so
    every request is distinct AND reproducible (temp>0 seeded contract).
    """
    env = os.environ if env is None else env
    out = dict(payload)
    for name, key, cast in _ENVELOPE_PINS:
        raw = env.get(name)
        if raw in (None, ""):
            continue
        try:
            out[key] = cast(raw)
        except (TypeError, ValueError):
            pass  # a malformed pin is ignored, never crashes the proxy
    # Per-request seed. Client seed wins (passthrough); else stamp base + index.
    if out.get("seed") is None:
        base = env.get("LUMO_PROXY_FORCE_SEED")
        if base not in (None, ""):
            try:
                out["seed"] = int(base) + int(seed_index or 0)
            except (TypeError, ValueError):
                pass
    return out


def request_tool_names(payload):
    """Return declared OpenAI tool names, preserving request order."""
    names = []
    for tool in payload.get("tools") or []:
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
            name = tool["function"].get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def has_exact_function_tools(payload, required_names):
    """Require one well-formed OpenAI function entry per expected tool name."""
    tools = payload.get("tools")
    if not isinstance(tools, list) or len(tools) != len(required_names):
        return False
    for tool, expected_name in zip(tools, required_names):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            return False
        function = tool.get("function")
        if not isinstance(function, dict) or function.get("name") != expected_name:
            return False
    return True


# ---------------------------------------------------------------------------
# Context-fit clamp (2026-07-08 datagen defect fix).
#
# vLLM rejects a /v1/chat/completions request with HTTP 400 when
# prompt_tokens + max_tokens > max_model_len (32768 for the epoch-2 27B
# teacher). Big-repo SWE prompts (>24k input tokens) plus the proxy's static
# 8192 max_tokens cap overflow the window, so the episode gets 0 tokens / 0
# tool calls and dies instantly. There is no tokenizer in the proxy, so we fit
# max_tokens two ways:
#   1. PREEMPTIVE: a cheap char/4 heuristic estimate of the prompt, reduce
#      max_tokens so input+output very likely fits (saves a round-trip).
#   2. EXACT FALLBACK: on a 400-with-context-overflow, parse vLLM's own
#      reported context limit + input-token count and RETRY ONCE with a
#      max_tokens computed to fit exactly.
# The heuristic is deliberately rough; the exact retry is the source of truth.
# ---------------------------------------------------------------------------

# vLLM's error text: "This model's maximum context length is 32768 tokens.
# However, you requested 8192 output tokens and your prompt contains at least
# 24577 input tokens, for a total of at least 32769 tokens. ...
# (parameter=input_tokens, value=24577)".
_CTX_LIMIT_RE = re.compile(r"maximum context length is (\d+)")
_INPUT_TOK_RE = re.compile(r"contains at least (\d+) input tokens")
_PARAM_VAL_RE = re.compile(r"parameter=input_tokens,\s*value=(\d+)")

# Never starve thinking-mode below this many output tokens UNLESS the prompt
# itself leaves less room (then we clamp to what remains minus the margin).
_MIN_OUTPUT_FLOOR = 1024
# Safety margin subtracted from the exact room so tokenizer/BOS/template drift
# does not re-trip the 400 on the retry.
_FIT_MARGIN = 64
# If the prompt leaves fewer than this many tokens of room, the instance is
# genuinely too big for the served context window: give up (let the 400 stand;
# the episode records as env-limited) rather than send a doomed sub-256 budget.
_MIN_ROOM = 256


def fit_max_tokens(input_tokens, requested_max, context_limit,
                   *, margin=_FIT_MARGIN, min_room=_MIN_ROOM):
    """Largest max_tokens that keeps input_tokens + max_tokens <= context_limit.

    Returns None when the prompt leaves < min_room tokens of room (env-limited:
    too big for this context window; the caller should let the 400 stand).

    Contract (preserve thinking-mode room): the result is >= _MIN_OUTPUT_FLOOR
    whenever the prompt leaves that much room; it drops below the floor ONLY
    when the prompt genuinely leaves less (then it is `room - margin`). It never
    exceeds requested_max — this is a downward clamp, never an inflation.
    """
    room = int(context_limit) - int(input_tokens)
    if room < min_room:
        return None
    target = room - margin
    if target < 1:
        return None
    return min(int(requested_max), target)


def parse_context_overflow(error_body):
    """Parse vLLM's context-overflow 400 body -> (context_limit, input_tokens).

    Returns None for any body that is not a context-length overflow (a bad-JSON
    400, an unrelated error, garbage) so the caller does NOT retry those.
    """
    if isinstance(error_body, (bytes, bytearray)):
        text = bytes(error_body).decode("utf-8", errors="replace")
    else:
        text = error_body or ""
    msg = text
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        err = obj.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            msg = err["message"]
        elif isinstance(err, str):
            msg = err
    m_ctx = _CTX_LIMIT_RE.search(msg)
    if not m_ctx:
        return None
    m_in = _INPUT_TOK_RE.search(msg) or _PARAM_VAL_RE.search(msg)
    if not m_in:
        return None
    return int(m_ctx.group(1)), int(m_in.group(1))


def estimate_prompt_tokens(payload):
    """Cheap char/4 prompt-size estimate (the proxy has no tokenizer).

    Deliberately rough: over-estimating merely trims a bit of output headroom,
    under-estimating is caught by the exact 400-retry. Covers messages + tools,
    which is where SWE prompts put the bulk of their bytes.
    """
    parts = []
    for key in ("messages", "tools"):
        val = payload.get(key)
        if val is not None:
            parts.append(json.dumps(val, ensure_ascii=False))
    if not parts:
        return 0
    return sum(len(p) for p in parts) // 4


# Bound the 400-retry backoff. Each retry is an INSTANT 400 (vLLM rejects
# pre-generation in ~50ms), so a handful is cheap and does NOT resemble the
# turn-consuming re-drive storm the datagen orchestrator worried about.
_MAX_CONTEXT_RETRIES = 8
# Below this many output tokens the instance is genuinely too big for the served
# window: stop backing off and let the 400 stand (episode records as env-limited)
# rather than send a doomed sub-floor budget.
_GIVEUP_OUTPUT = 128


def build_context_retry_body(body, error_body, attempt=0):
    """Given the CURRENT forwarded body (bytes) and vLLM's 400 error body,
    return a re-serialized body (bytes) with max_tokens reduced to fit, or None
    when no retry should be attempted (not an overflow, unparseable body, or the
    instance is genuinely too big -> let the 400 stand).

    CRUCIAL: vLLM's error reports input_tokens as a LOWER BOUND, not the true
    prompt length — it is exactly (context_limit - requested_max_tokens + 1),
    independent of the real prompt size. So a single "fit to the reported number"
    retry is insufficient whenever the true prompt exceeds that floor (it merely
    trims max_tokens by ~margin and 400s again). We therefore:
      * attempt 0 (first retry): OPTIMISTIC — size to the reported number. This
        is exact and wastes no output headroom IF the server reports the true
        count (some vLLM builds do), and costs one instant 400 if it does not.
      * attempts 1+: GEOMETRIC BACKOFF — halve the current max_tokens. This is
        the term that actually converges when the reported number is a floor:
        the fixed prompt + a halving output budget must eventually fit (or we
        cross the give-up floor and record env-limited).
    """
    if not body:
        return None
    parsed = parse_context_overflow(error_body)
    if parsed is None:
        return None
    context_limit, input_tokens = parsed
    try:
        payload = json.loads(bytes(body).decode("utf-8"))
    except (ValueError, TypeError, AttributeError):
        return None
    if not isinstance(payload, dict):
        return None
    requested = int(payload.get("max_tokens") or context_limit)
    # Optimistic ceiling from the reported (possibly-floor) numbers.
    optimistic = fit_max_tokens(input_tokens, requested, context_limit)
    if attempt <= 0:
        new_max = optimistic
    else:
        # Halve the budget that just failed; never let a loose optimistic
        # ceiling undo the guaranteed progress of halving.
        halved = requested // 2
        new_max = halved if optimistic is None else min(optimistic, halved)
    if new_max is None or new_max < _GIVEUP_OUTPUT or new_max >= requested:
        return None
    payload["max_tokens"] = new_max
    return json.dumps(payload).encode("utf-8")


class ProxyHandler(BaseHTTPRequestHandler):
    upstream = "http://127.0.0.1:30000/v1"
    max_tokens = 512
    # Served context window (vLLM --max-model-len). Drives the PREEMPTIVE cap;
    # the exact 400-retry reads the true limit from the error, so a stale value
    # here only costs an extra round-trip, never correctness. 0 disables the
    # preemptive cap (retry still fires).
    context_limit = 32768
    dump_dir = None
    tool_choice = ""
    tool_choice_turns = 0
    chat_count = 0
    request_count = 0
    health_token = ""
    required_tools = None

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "token": self.health_token}).encode("utf-8"))
            return
        self.forward(None)

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON")
            return

        is_chat = self.path.rstrip("/") == "/v1/chat/completions"
        if is_chat:
            if self.required_tools is not None:
                observed_tools = request_tool_names(payload)
                if not has_exact_function_tools(payload, self.required_tools):
                    self.send_response(422)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": {
                            "message": "agent tool boundary mismatch",
                            "expected": self.required_tools,
                            "observed": observed_tools,
                            "observed_entries": len(payload.get("tools", []))
                            if isinstance(payload.get("tools"), list) else None,
                        }
                    }).encode("utf-8"))
                    return
            type(self).chat_count += 1
            # THINKING MODE (env-gated; CONFIG_DELTAS.md D6). Default-OFF preserves
            # the historical byte-identical behavior for the non-thinking 9B teacher
            # (LUMO_ENABLE_THINKING unset -> False). The Qwen3.6-27B thinking teacher
            # sets LUMO_ENABLE_THINKING=true (Regime T) so <think> traces are emitted.
            _enable_thinking = (
                os.environ.get("LUMO_ENABLE_THINKING", "false").strip().lower() == "true"
            )
            payload["chat_template_kwargs"] = {
                **(payload.get("chat_template_kwargs") or {}),
                "enable_thinking": _enable_thinking,
            }
            if self.max_tokens and int(payload.get("max_tokens") or 0) > self.max_tokens:
                payload["max_tokens"] = self.max_tokens
            # PREEMPTIVE context-fit cap (char/4 heuristic; the exact 400-retry
            # in forward() is the fallback). Reduce max_tokens so a big-repo
            # prompt + output very likely fits the served context window instead
            # of 400-ing and losing the whole episode. Never inflates; only
            # reduces when the estimate says the request would overflow.
            if self.context_limit and int(payload.get("max_tokens") or 0) > 0:
                requested = int(payload["max_tokens"])
                capped = fit_max_tokens(
                    estimate_prompt_tokens(payload), requested, self.context_limit
                )
                if capped is not None and capped < requested:
                    payload["max_tokens"] = capped
            within_tool_choice_turns = (
                not self.tool_choice_turns or type(self).chat_count <= self.tool_choice_turns
            )
            if (
                self.tool_choice
                and within_tool_choice_turns
                and payload.get("tools")
                and not payload.get("tool_choice")
            ):
                payload["tool_choice"] = self.tool_choice
            # Reference-envelope sampling pins (LUMO_PROXY_FORCE_*, env-gated,
            # default-OFF -> byte-identical passthrough). Applied LAST so the
            # dumped body below is exactly what is forwarded upstream (proxy-side
            # evidence of the envelope the server received).
            payload = apply_reference_envelope(payload, seed_index=next(_SEED_COUNTER))
            if self.dump_dir:
                type(self).request_count += 1
                self._capture_idx = type(self).request_count
                dump_path = self.dump_dir / f"chat_{type(self).request_count:04d}.json"
                dump_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.forward(json.dumps(payload).encode("utf-8"), allow_context_retry=is_chat)

    def _record_usage(self, idx, tail_bytes):
        """Append the streamed response's final usage + finish_reason to
        dump_dir/usage.jsonl. Exact per-request tokens + a timestamp, captured
        server-side so they survive ANY qwen-code exit mode (turn-limit 53 /
        loop-detect 1 / budget 55 write nothing to the CLI's stdout)."""
        import time as _t
        text = tail_bytes.decode("utf-8", errors="replace")
        usage = None
        finish = None
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                if isinstance(obj.get("usage"), dict):
                    usage = obj["usage"]
                for ch in (obj.get("choices") or []):
                    if isinstance(ch, dict) and ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        rec = {"idx": idx, "ts": round(_t.time(), 3), "usage": usage, "finish_reason": finish}
        try:
            with (self.dump_dir / "usage.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:  # never let telemetry break the proxy
            pass

    def forward(self, body, allow_context_retry=False):
        url = self.upstream.rstrip("/") + self.path.removeprefix("/v1")
        headers = {"Content-Type": "application/json"}
        if self.headers.get("Authorization"):
            headers["Authorization"] = self.headers["Authorization"]
        capture_idx = getattr(self, "_capture_idx", None)
        tail = b""
        # Context-fit backoff loop. urlopen raises HTTPError BEFORE we send any
        # bytes to the client, so on a context-overflow 400 we can silently
        # re-forward a body with a smaller max_tokens and only the finally-
        # successful response reaches the client. Each retry is an instant
        # (pre-generation) rejection, so the bounded loop is cheap.
        attempt = 0
        cur_body = body
        while True:
            req = urllib.request.Request(url, data=cur_body, headers=headers, method=self.command)
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    self.send_response(resp.status)
                    content_type = resp.headers.get("Content-Type", "application/json")
                    self.send_header("Content-Type", content_type)
                    self.end_headers()
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        if capture_idx is not None and self.dump_dir is not None:
                            tail = (tail + chunk)[-16384:]
                if capture_idx is not None and self.dump_dir is not None:
                    self._record_usage(capture_idx, tail)
                return
            except urllib.error.HTTPError as exc:
                error_body = exc.read()
                if (
                    exc.code == 400
                    and allow_context_retry
                    and attempt < _MAX_CONTEXT_RETRIES
                ):
                    retry_body = build_context_retry_body(cur_body, error_body, attempt)
                    if retry_body is not None:
                        new_max = json.loads(retry_body).get("max_tokens")
                        sys.stderr.write(
                            f"context-overflow 400 for {self.path}: retry "
                            f"{attempt + 1}/{_MAX_CONTEXT_RETRIES} with "
                            f"max_tokens={new_max}\n"
                        )
                        attempt += 1
                        cur_body = retry_body
                        continue
                sys.stderr.write(
                    f"upstream HTTP {exc.code} for {self.path}: "
                    f"{error_body.decode('utf-8', errors='replace')}\n"
                )
                self.send_response(exc.code)
                self.send_header("Content-Type", exc.headers.get("Content-Type", "text/plain"))
                self.end_headers()
                self.wfile.write(error_body)
                return
            except Exception as exc:
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"{type(exc).__name__}: {exc}".encode("utf-8"))
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--upstream", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--context-limit", type=int, default=32768,
        help="Served context window (vLLM --max-model-len). Drives the "
             "preemptive context-fit cap; 0 disables it (the exact 400-retry "
             "still fires). The retry reads the true limit from vLLM's error.",
    )
    parser.add_argument("--dump-dir", type=Path, default=None)
    parser.add_argument("--tool-choice", default="")
    parser.add_argument("--tool-choice-turns", type=int, default=0)
    parser.add_argument(
        "--required-tools", default=None,
        help="comma-separated exact tool list required on every chat request",
    )
    args = parser.parse_args()

    ProxyHandler.upstream = args.upstream
    ProxyHandler.max_tokens = args.max_tokens
    ProxyHandler.context_limit = args.context_limit
    ProxyHandler.tool_choice = args.tool_choice
    ProxyHandler.tool_choice_turns = args.tool_choice_turns
    ProxyHandler.health_token = os.environ.get("LUMO_PROXY_HEALTH_TOKEN", "")
    if args.required_tools is not None:
        ProxyHandler.required_tools = [name for name in args.required_tools.split(",") if name]
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        ProxyHandler.dump_dir = args.dump_dir
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    print(f"qwen-code OpenAI compatibility proxy listening on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
