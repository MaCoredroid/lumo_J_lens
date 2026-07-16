#!/usr/bin/env python3
"""Unit tests for the Qwen Code proxy context-fit clamp (2026-07-08 fix).

Covers the pure helpers that turn a vLLM context-overflow 400 into a retry body
with a shrunk max_tokens: fit_max_tokens / parse_context_overflow /
estimate_prompt_tokens / build_context_retry_body. No network; the vLLM 400 is
mocked from its real error-string shape.

Run: .venv-swe/bin/python tests/test_proxy_context_clamp.py
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qwen_code_proxy as P  # noqa: E402


# The real vLLM 400 body shape (from the monitor forensics).
def vllm_overflow_body(limit=32768, requested=8192, input_tokens=24577):
    total = requested + input_tokens
    msg = (
        f"This model's maximum context length is {limit} tokens. However, you "
        f"requested {requested} output tokens and your prompt contains at least "
        f"{input_tokens} input tokens, for a total of at least {total} tokens. "
        f"Please reduce the length of the messages or completion. "
        f"(parameter=input_tokens, value={input_tokens})"
    )
    return json.dumps({"error": {"message": msg, "type": "BadRequestError", "code": 400}}).encode("utf-8")


class FitMaxTokens(unittest.TestCase):
    def test_downward_clamp_leaves_room(self):
        # 24577 input, 32768 limit -> room 8191, minus 64 margin = 8127.
        self.assertEqual(P.fit_max_tokens(24577, 8192, 32768), 8127)

    def test_never_inflates(self):
        # Plenty of room, but requested is small: clamp never raises it.
        self.assertEqual(P.fit_max_tokens(1000, 512, 32768), 512)

    def test_preserves_thinking_floor_when_room_is_large(self):
        # Small prompt, large request -> result comfortably above the 1024 floor.
        self.assertGreaterEqual(P.fit_max_tokens(2000, 8192, 32768), P._MIN_OUTPUT_FLOOR)

    def test_env_limited_returns_none(self):
        # Prompt leaves < _MIN_ROOM (256): genuinely too big -> None (let 400 stand).
        self.assertIsNone(P.fit_max_tokens(32700, 8192, 32768))

    def test_below_floor_only_when_prompt_leaves_less(self):
        # Room = 32768-32000 = 768 (>=256), minus 64 = 704 (< 1024 floor but valid).
        self.assertEqual(P.fit_max_tokens(32000, 8192, 32768), 704)


class ParseOverflow(unittest.TestCase):
    def test_parses_real_body(self):
        self.assertEqual(
            P.parse_context_overflow(vllm_overflow_body()), (32768, 24577)
        )

    def test_parses_param_value_fallback(self):
        # Body missing the "contains at least" phrase but with the parameter= tail.
        msg = ("This model's maximum context length is 32768 tokens. "
               "(parameter=input_tokens, value=30000)")
        body = json.dumps({"error": {"message": msg}}).encode()
        self.assertEqual(P.parse_context_overflow(body), (32768, 30000))

    def test_non_overflow_returns_none(self):
        body = json.dumps({"error": {"message": "invalid tool_choice value"}}).encode()
        self.assertIsNone(P.parse_context_overflow(body))

    def test_garbage_returns_none(self):
        self.assertIsNone(P.parse_context_overflow(b"\x00 not json"))

    def test_plain_string_body(self):
        msg = ("This model's maximum context length is 32768 tokens. However, "
               "your prompt contains at least 25000 input tokens.")
        self.assertEqual(P.parse_context_overflow(msg), (32768, 25000))


class EstimatePromptTokens(unittest.TestCase):
    def test_roughly_char_over_4(self):
        payload = {"messages": [{"role": "user", "content": "x" * 4000}]}
        est = P.estimate_prompt_tokens(payload)
        self.assertGreater(est, 900)  # ~1000+ from content plus json overhead

    def test_empty(self):
        self.assertEqual(P.estimate_prompt_tokens({"max_tokens": 8192}), 0)


class BuildRetryBody(unittest.TestCase):
    def test_mock_400_to_retry_body(self):
        # THE core scenario: 25k-token prompt + max_tokens 8192 -> clamped retry.
        orig = json.dumps({
            "model": "qwen3.6-27b-nvfp4",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "hello"}],
        }).encode()
        retry = P.build_context_retry_body(orig, vllm_overflow_body())
        self.assertIsNotNone(retry)
        obj = json.loads(retry)
        self.assertEqual(obj["max_tokens"], 8127)  # room 8191 - 64 margin
        self.assertLess(obj["max_tokens"], 8192)   # strictly reduced
        # Other fields preserved.
        self.assertEqual(obj["model"], "qwen3.6-27b-nvfp4")
        self.assertEqual(obj["messages"][0]["content"], "hello")

    def test_non_overflow_400_no_retry(self):
        orig = json.dumps({"max_tokens": 8192, "messages": []}).encode()
        body = json.dumps({"error": {"message": "bad tool_choice"}}).encode()
        self.assertIsNone(P.build_context_retry_body(orig, body))

    def test_env_limited_no_retry(self):
        # Prompt so big no usable room remains -> None (400 stands, env-limited).
        orig = json.dumps({"max_tokens": 8192, "messages": []}).encode()
        body = vllm_overflow_body(limit=32768, requested=8192, input_tokens=32760)
        self.assertIsNone(P.build_context_retry_body(orig, body))

    def test_no_reduction_needed_no_retry(self):
        # If the already-requested max_tokens fits, do not spuriously retry.
        orig = json.dumps({"max_tokens": 100, "messages": []}).encode()
        body = vllm_overflow_body(limit=32768, requested=100, input_tokens=1000)
        # room is huge; fit returns 100 == requested -> no retry.
        self.assertIsNone(P.build_context_retry_body(orig, body))

    def test_empty_body_no_retry(self):
        self.assertIsNone(P.build_context_retry_body(b"", vllm_overflow_body()))

    def test_attempt1_halves(self):
        # On the SECOND retry (attempt=1) the budget that just failed is halved,
        # which is the term that converges when vLLM's reported input_tokens is a
        # lower bound (== limit - max + 1) rather than the true prompt length.
        orig = json.dumps({"max_tokens": 8127, "messages": []}).encode()
        # vLLM re-reports the floor for the NEW max_tokens (8127): 32768-8127+1.
        body = vllm_overflow_body(limit=32768, requested=8127, input_tokens=32768 - 8127 + 1)
        retry = P.build_context_retry_body(orig, body, attempt=1)
        self.assertEqual(json.loads(retry)["max_tokens"], 8127 // 2)

    def test_backoff_converges_against_floor_only_server(self):
        # Simulate a vLLM that ONLY ever reports the floor (limit-max+1), with a
        # true prompt of 28000 tokens. The forward() loop must converge to a
        # max_tokens that actually fits (28000 + max <= 32768 -> max <= 4768).
        limit, true_prompt = 32768, 28000
        max_tokens = 8192
        cur = json.dumps({"max_tokens": max_tokens, "messages": []}).encode()
        fit_max = None
        for attempt in range(P._MAX_CONTEXT_RETRIES):
            # This request overflows iff true_prompt + max_tokens > limit.
            if true_prompt + max_tokens <= limit:
                fit_max = max_tokens
                break
            body = vllm_overflow_body(limit=limit, requested=max_tokens,
                                      input_tokens=limit - max_tokens + 1)
            nxt = P.build_context_retry_body(cur, body, attempt=attempt)
            self.assertIsNotNone(nxt, f"gave up too early at attempt {attempt}")
            cur = nxt
            max_tokens = json.loads(nxt)["max_tokens"]
        self.assertIsNotNone(fit_max, "never converged to a fitting max_tokens")
        self.assertLessEqual(true_prompt + fit_max, limit)
        self.assertGreater(fit_max, 0)

    def test_backoff_gives_up_when_prompt_exceeds_window(self):
        # A prompt that alone fills the window can never fit: the loop must give
        # up within the bound (returns None) rather than spin -> env-limited.
        limit, true_prompt = 32768, 32700  # leaves only 68 tokens of room
        max_tokens = 8192
        cur = json.dumps({"max_tokens": max_tokens, "messages": []}).encode()
        gave_up = False
        for attempt in range(P._MAX_CONTEXT_RETRIES + 2):
            if true_prompt + max_tokens <= limit:
                self.fail("unexpectedly fit a window-filling prompt")
            body = vllm_overflow_body(limit=limit, requested=max_tokens,
                                      input_tokens=limit - max_tokens + 1)
            nxt = P.build_context_retry_body(cur, body, attempt=attempt)
            if nxt is None:
                gave_up = True
                break
            cur = nxt
            max_tokens = json.loads(nxt)["max_tokens"]
        self.assertTrue(gave_up, "should have given up (env-limited) but did not")


if __name__ == "__main__":
    unittest.main(verbosity=2)
