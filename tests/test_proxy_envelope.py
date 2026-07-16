#!/usr/bin/env python3
"""Unit tests for the Qwen Code proxy reference-envelope transform.

Exercises apply_reference_envelope directly (pure function over an env dict +
seed index) so no server is needed. Run:

    .venv-swe/bin/python tests/test_proxy_envelope.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen_code_proxy import (  # noqa: E402
    apply_reference_envelope,
    has_exact_function_tools,
    request_tool_names,
)

FULL_ENV = {
    "LUMO_PROXY_FORCE_TEMPERATURE": "1.0",
    "LUMO_PROXY_FORCE_TOP_P": "0.95",
    "LUMO_PROXY_FORCE_TOP_K": "20",
    "LUMO_PROXY_FORCE_SEED": "1000",
}

_failures = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    if not cond:
        _failures.append(f"{name}: {detail}")
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))


def test_passthrough_when_unset():
    """No LUMO_PROXY_FORCE_* -> byte-identical passthrough (the greedy A/B floor)."""
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}
    out = apply_reference_envelope(body, env={}, seed_index=7)
    check("passthrough.equal", out == body, f"{out!r} != {body!r}")
    check("passthrough.no_seed", "seed" not in out, f"seed leaked: {out.get('seed')}")
    check("passthrough.not_aliased", out is not body, "returned the same object (mutation risk)")


def test_input_not_mutated():
    """The caller's dict must not be mutated in place."""
    body = {"temperature": 0, "top_p": 1.0}
    snapshot = dict(body)
    apply_reference_envelope(body, env=FULL_ENV, seed_index=0)
    check("no_mutate.input_unchanged", body == snapshot, f"{body!r} != {snapshot!r}")


def test_envelope_overrides_greedy():
    """temp/top_p/top_k pins overwrite the client's greedy values, right types."""
    body = {"temperature": 0, "top_p": 1.0, "messages": []}
    out = apply_reference_envelope(body, env=FULL_ENV, seed_index=0)
    check("override.temp", out["temperature"] == 1.0 and isinstance(out["temperature"], float),
          f"temperature={out.get('temperature')!r}")
    check("override.top_p", out["top_p"] == 0.95, f"top_p={out.get('top_p')!r}")
    check("override.top_k", out["top_k"] == 20 and isinstance(out["top_k"], int),
          f"top_k={out.get('top_k')!r}")


def test_seed_stamped_per_request():
    """No client seed + FORCE_SEED -> base + seed_index (distinct, reproducible)."""
    out0 = apply_reference_envelope({}, env=FULL_ENV, seed_index=0)
    out5 = apply_reference_envelope({}, env=FULL_ENV, seed_index=5)
    check("seed.base", out0["seed"] == 1000, f"seed={out0.get('seed')!r}")
    check("seed.indexed", out5["seed"] == 1005, f"seed={out5.get('seed')!r}")
    check("seed.distinct", out0["seed"] != out5["seed"], "per-request seeds collided")


def test_client_seed_passthrough():
    """A client-supplied seed is left UNTOUCHED even with FORCE_SEED set."""
    out = apply_reference_envelope({"seed": 42}, env=FULL_ENV, seed_index=9)
    check("client_seed.preserved", out["seed"] == 42, f"seed={out.get('seed')!r}")


def test_no_seed_without_force():
    """Envelope pins set but FORCE_SEED unset -> no seed injected."""
    env = {k: v for k, v in FULL_ENV.items() if k != "LUMO_PROXY_FORCE_SEED"}
    out = apply_reference_envelope({"temperature": 0}, env=env, seed_index=3)
    check("no_force_seed.absent", "seed" not in out, f"seed leaked: {out.get('seed')}")
    check("no_force_seed.temp_still_set", out["temperature"] == 1.0, f"temp={out.get('temperature')!r}")


def test_malformed_pin_ignored():
    """A malformed pin is ignored (no crash, key falls back to whatever was sent)."""
    body = {"temperature": 0, "top_k": 3}
    out = apply_reference_envelope(
        body,
        env={"LUMO_PROXY_FORCE_TEMPERATURE": "not-a-float", "LUMO_PROXY_FORCE_TOP_K": "20"},
        seed_index=0,
    )
    check("malformed.temp_untouched", out["temperature"] == 0, f"temp={out.get('temperature')!r}")
    check("malformed.top_k_applied", out["top_k"] == 20, f"top_k={out.get('top_k')!r}")


def test_optional_knobs():
    """min_p / presence_penalty are supported (parity with the flywheel), default-off."""
    out = apply_reference_envelope(
        {},
        env={"LUMO_PROXY_FORCE_MIN_P": "0.0", "LUMO_PROXY_FORCE_PRESENCE_PENALTY": "1.5"},
        seed_index=0,
    )
    check("optional.min_p", out["min_p"] == 0.0, f"min_p={out.get('min_p')!r}")
    check("optional.presence_penalty", out["presence_penalty"] == 1.5,
          f"presence_penalty={out.get('presence_penalty')!r}")


def test_empty_string_env_is_off():
    """Empty-string env value counts as unset (shells often export '')."""
    out = apply_reference_envelope({"temperature": 0}, env={"LUMO_PROXY_FORCE_TEMPERATURE": ""},
                                   seed_index=0)
    check("empty_env.off", out["temperature"] == 0, f"temp={out.get('temperature')!r}")


def test_request_tool_names():
    payload = {"tools": [
        {"type": "function", "function": {"name": "run_shell_command"}},
        {"type": "function", "function": {"name": "edit"}},
    ]}
    check("tools.names", request_tool_names(payload) == ["run_shell_command", "edit"])


def test_exact_tool_boundary():
    allowed = {"tools": [{
        "type": "function",
        "function": {"name": "run_shell_command", "parameters": {"type": "object"}},
    }]}
    check("tools.exact_allowed", has_exact_function_tools(allowed, ["run_shell_command"]))
    bypass = {"tools": allowed["tools"] + [{"type": "computer_use_preview"}]}
    check("tools.reject_nonfunction_extra",
          not has_exact_function_tools(bypass, ["run_shell_command"]))
    duplicate = {"tools": allowed["tools"] * 2}
    check("tools.reject_duplicate",
          not has_exact_function_tools(duplicate, ["run_shell_command"]))
    malformed = {"tools": [{"function": {"name": "run_shell_command"}}]}
    check("tools.reject_missing_type",
          not has_exact_function_tools(malformed, ["run_shell_command"]))
    check("tools.reject_nonlist",
          not has_exact_function_tools({"tools": {}}, ["run_shell_command"]))


if __name__ == "__main__":
    for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
        globals()[fn]()
    if _failures:
        print(f"\n{len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nALL PROXY-ENVELOPE UNIT TESTS PASSED")
