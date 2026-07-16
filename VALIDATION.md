# Validation

## Publication-Certified Run

Date: 2026-07-15 (America/Los_Angeles)

Run: `publication_certified_v2_20260715`

Result: **PASS**

This run validates the extracted repository itself, including the final security
boundary. It closes the historical gap between the July 8 gate, which used Qwen
Code 0.19.2 and a temperature-0.6 envelope, and the final Qwen Code 0.19.4,
temperature-1.0 thinking profile.

| Item | Certified result |
|---|---|
| Model | `nvidia/Qwen3.6-27B-NVFP4` at revision `0893e160...` |
| Server | vLLM 0.23.0, ModelOpt mixed NVFP4, FP8 E4M3 KV |
| MTP | native one-token draft model detected and active |
| Cache | prefix caching, chunked prefill, Mamba `align`/FP32 |
| Agent | Qwen Code 0.19.4, thinking enabled |
| Boundary | isolated HOME/CWD; exact tool schema `[run_shell_command]` |
| Runtime | pinned official task image, container network `none` |
| Task | `sympy__sympy-13480` |
| Qwen Code | exit 0, 9 turns/requests, 32.406 s |
| Patch | 590 bytes; SHA-256 `98e80f91393e...` |
| Official score | 1/1 resolved; zero unresolved, empty, or error |

### Setup and server evidence

- `.venv-vllm` exactly matched the 190-package freeze; `uv pip check` passed.
- `.venv-swe` exactly matched the 72-package freeze; `uv pip check` passed.
- `npm ci` installed Qwen Code 0.19.4 with zero reported vulnerabilities.
- The server passed the exact-name and 32,768-context endpoint gate.
- vLLM resolved `modelopt_fp4` to `modelopt_mixed` and KV to `fp8_e4m3`.
- The engine detected and shared weights with the native MTP draft model.
- With a warm compilation cache, engine initialization took 10.00 seconds,
  including 5.14 seconds compilation.
- GPU KV capacity was 83,012 tokens.
- MTP acceptance samples were 92.8% and 93.3%.
- Prefix-cache hit samples rose from 51.7% to 66.3%.
- Generation-throughput samples were 48.6 and 51.7 tokens/second.
- GPU memory returned to 383 MiB after server shutdown.

### Agent and isolation evidence

All nine forwarded requests declared exactly one well-formed function tool:
`run_shell_command`. The proxy rejects extra, malformed, non-function, or
duplicate entries. The agent executed nine shell calls successfully through the
Docker shim and attempted no other tool. The task container used `--network
none`, a 6 GiB memory ceiling, 8 GiB memory-plus-swap ceiling, and 1,024-PID
ceiling. Qwen ran with a scrubbed environment, fresh HOME, and an isolated CWD
containing only `AGENTS.md`.

The first forwarded envelope was:

```json
{
  "model": "qwen3.6-27b-nvfp4",
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 0.0,
  "seed": 880001234,
  "max_tokens": 8192,
  "chat_template_kwargs": {"enable_thinking": true},
  "tools": ["run_shell_command"]
}
```

Seeds increased monotonically through `880001242`. Aggregate model usage was
124,864 input tokens and 1,439 output tokens.

### Task and scorer evidence

The model corrected the undefined `cotm` reference to the existing `cothm`
variable in `sympy/functions/elementary/hyperbolic.py`. The official SWE-bench
4.1.0 harness, using the same materialized pinned dataset row, reported:

```json
{
  "submitted_instances": 1,
  "completed_instances": 1,
  "resolved_instances": 1,
  "unresolved_instances": 0,
  "empty_patch_instances": 0,
  "error_instances": 0,
  "resolved_ids": ["sympy__sympy-13480"]
}
```

Generation deliberately ran with local evaluation skipped, so its internal
campaign summary contains no resolved verdict. The separate official report
above is the benchmark verdict, as required by the memory-safe lifecycle.

Published evidence:

- `validation/2026-07-15-publication-certified.json`: sanitized machine record
- `validation/first-forwarded-request.json`: sanitized request envelope
- `validation/sympy__sympy-13480.patch`: exact generated patch
- `validation/official-report.json`: exact official aggregate report
- `validation/runtime-source-manifest.sha256`: exact certified runtime sources

Artifact hashes:

| Artifact | SHA-256 |
|---|---|
| Patch | `98e80f91393e76fc6323eeb0a7582aa89f1db717932ea5742eed467d692d024d` |
| Predictions JSONL | `7ea5811e0765855a1a817a06e975893368cc4304e24d093cfb4cc7d857fab4bb` |
| Materialized dataset | `e8d93ec886c5d367e412c2ec04c7a7696af3c821ab9fb3f0e03261a7743445e2` |
| Official report | `6f0e1c2ce449d411df0f2f8ffaf2680f1178ced057cee156c989a53732b0f7f8` |
| Runtime source manifest | `9537252ade61d1c58350b77b182180d8e09ea2a445d42d98218a210641f2bfcd` |

## Static Verification

After the final documentation-alignment changes, `scripts/check.sh` passed:

- all shell scripts passed `bash -n`;
- all Python entry points and tests passed `py_compile`;
- 24 standalone envelope/tool-boundary assertions passed;
- 29 discovered unit tests passed;
- both Python environments passed compatibility and exact-freeze checks; and
- the installed Qwen Code version was 0.19.4.

## Historical July 8 Gate

The source session's earlier three-gate run established broader serving evidence:

- format: zero schema mismatches and identical `qwen3_xml` structure;
- argument grounding: 63/64 source-verbatim arguments, zero malformed;
- live tasks: 4/4 resolved, including 2/2 official Verified; and
- MTP A/B: 109.4 versus 68.8 tokens/second, 1.59x at 93.02% acceptance.

The historical Verified tasks were `django__django-10914` and
`sympy__sympy-13480`. Those results are not presented as final-stack
certification; the publication-certified run above is that evidence.
