# Validation

## Jacobian Lens Fit And Transfer

Dates: 2026-07-16 to 2026-07-17 (America/Los_Angeles)

The July 16 fresh-fit result is a **completed exact dense NF4 fit**, not a
native NVFP4 fit. The July 17 native NVIDIA implementation has separately
passed its operation, late-suffix, and exploratory one-row all-layer hardware
gates using compiled NVFP4/FP8 forward values and an identity-STE surrogate
backward. Its strict per-prompt production captures and complete `n=10`
artifact are **pending**. No final native artifact hash, held-out score, or
completed-reproduction claim exists yet.

| Claim | Result | Evidence |
|---|---|---|
| Real-model batched-VJP diagnostics (`C=4,8,32`) | PASS; exact equality to sequential autograd on source layers 61/62 | [`c4`](validation/jlens-nf4-diagnostic-c4-2026-07-16.json), [`c8`](validation/jlens-nf4-diagnostic-c8-2026-07-16.json), [`c32`](validation/jlens-nf4-diagnostic-c32-2026-07-16.json) |
| Exact `n=10` NF4 fit, layers `0..62` to target 63 | PASS; status `completed`, `complete=true` | [`fit provenance`](validation/jlens-nf4-fit-provenance-2026-07-16.json) |
| Published local artifact | PASS; 63 finite FP32 `[5120,5120]` matrices | [`artifact verification`](validation/jlens-nf4-artifact-verification-2026-07-16.json) |
| Held-out NF4 readout evaluation | COMPLETED; four prompts, four positions, all 63 layers | [`evaluation`](validation/jlens-nf4-eval-2026-07-16.json) |
| Dense local/public matrix comparison | REPORTED; no post-hoc similarity threshold | [`comparison`](validation/jlens-nf4-vs-public-2026-07-16.json) |
| Local NF4 lens applied to NVFP4, strict paired adapter gate | **FAIL**; cross-application is not certified | [`local lens run`](validation/jlens-nf4-on-nvfp4-2026-07-16.json) |
| Public lens on the same four NVFP4 prompts, control gate | **FAIL** with the same adapter errors | [`public control`](validation/jlens-public-on-nvfp4-heldout-2026-07-16.json) |
| Public lens on the original two semantic prompts | PASS after recertification | [`public baseline`](validation/jlens-nvfp4-2026-07-16.json) |
| Exploratory native compiled baseline/observer proof | PASS engineering evidence; 688 shared tensors directly bit-exact, 432 observer-only boundaries complete, 785 replay parameters content-equal; predates strict identity binding | [`native report`](docs/JLENS_NVFP4_STE_EXPERIMENT.md) |
| Native packed/live VJPs, analytic GDN, and exploratory all-layer reverse replay | PASS engineering gates; one real estimator row for each source layer `0..62` | [`native report`](docs/JLENS_NVFP4_STE_EXPERIMENT.md) |
| Dense native NVFP4/FP8-STE `n=10` artifact | **PENDING**; ten prompts and all 5,120 rows for all 63 matrices have not completed | [`native runbook`](docs/JLENS_NVFP4_STE_EXPERIMENT.md) |

The exploratory native capture ran the local pinned vLLM/ModelOpt graph; this
RTX 5090 resolved the W4 operations to the observed weight-only Marlin fallback
and the FP8 operations to Cutlass. Its artifact predates the final
`model.identity` and shard-hash fields, so the hardened production runner will
not adopt it and must recapture/reprove every prompt. MTP was disabled because
the proof covers main-model prefill, not speculative draft/decode. Of the
capture boundaries, 688 GDN/attention tensors were directly compared
bit-for-bit. The 432 linear/SwiGLU/post-block tensors exist only in the observer
graph and are supported indirectly by exact endpoint generation parity plus
the direct shared-tensor proof; they are not claimed as direct baseline
equality. Real packed-W4 and live-FP8 probes measured relative RMS `1.2549e-7`
and `7.8404e-7` against dense dequantization. The batch-256 estimator
measurement projects to about 12.7 hours for ten prompts, before capture,
hashing, commit, finalization, and export overhead.

The fit used `Qwen/Qwen3.6-27B` revision `6a9e13bd...`, 496
bitsandbytes NF4 linears, double quantization, BF16 compute, a 128-token
sequence, `skip_first=16`, cotangent batch 32, and the full 5,120 rows for all
63 source matrices. The estimator took 6,367.555 seconds; cumulative
in-process invocation time was 6,496.513 seconds (1:48:16.5). The timestamps
span 6,566.933 seconds including the pause before resume. Peak CUDA
allocation/reservation was 23.252/24.201 GiB. The resulting 6,606,048,039-byte
artifact has SHA-256
`54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f`.

The measured fit commands were:

```bash
scripts/setup_fit.sh
scripts/check_fit.sh

.venv-fit/bin/python scripts/fit_jlens_nf4.py \
  --work-dir .cache/jlens-nf4-production-n10-c32 \
  --output .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.pt \
  --provenance .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.provenance.json \
  --max-prompts 1 --output-dtype float32

.venv-fit/bin/python scripts/fit_jlens_nf4.py \
  --work-dir .cache/jlens-nf4-production-n10-c32 \
  --output .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.pt \
  --provenance .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.provenance.json \
  --resume --output-dtype float32
```

The first invocation committed one prompt in 657.335 seconds and stopped by
contract. The resumed invocation completed the other nine in 5,839.178
seconds. Held-out NF4 evaluation then completed in 15.363 seconds and matched
the fit's aggregate NF4 weight hash exactly. The dense comparison with the
public `n=1000` FP16 lens of unpublished fit precision measured global
Frobenius cosine `0.750216`, mean
per-layer cosine `0.820655`, and global relative Frobenius difference
`0.865690`; these are descriptive measurements, not equivalence gates.

The local artifact was successfully loaded and read through every NVFP4 layer,
but the strict certificate ended with status `failed` after 33.328 seconds.
On Wikitext validation rows 3, 18, and 49, full-logit max error was `0.125`,
above the `0.0625` limit; row 42 was exactly at the limit. All four full-logit
RMS errors were below `0.01`, and all four reconstructed top-1 tokens matched
greedy generation. Row 18 also exceeded the final-norm max limit
(`0.25 > 0.125`) and missed the exact top-5 prefix. Repeating the run with the
public lens produced the identical reconstruction errors, demonstrating that
this strict failure is in the residual adapter check rather than the fitted
lens. It does not make the local NVFP4 cross-application certified.

As a descriptive cross-model readout comparison over the same 1,008
layer/position observations, local-lens NF4 versus NVFP4 outputs averaged
`0.769841` top-1 agreement, `0.816468` top-5 overlap, and `0.974066` Spearman
target-rank correlation. The public-lens control measured `0.755952`,
`0.818452`, and `0.979986`. These measurements do not override the failed
adapter certificate.

The original two-prompt public-lens baseline remains a separate passing run.
It was recertified in 16.300 seconds, including an 8.846-second model load,
with 25.618/27.596 GiB peak CUDA allocated/reserved. Both prompts had full-logit
max error `0.0625` at the allowed boundary, RMS errors `0.008794` and
`0.008464`, exact top-5 prefixes, and greedy top-1 agreement.

The detailed method, commands, result interpretation, and limitations are in
[`docs/JLENS_NF4_EXPERIMENT.md`](docs/JLENS_NF4_EXPERIMENT.md). Evidence-file
hashes are pinned in
[`validation/jlens-nf4-evidence.sha256`](validation/jlens-nf4-evidence.sha256).
The corresponding fit, evaluation, runner, checker, test, contract, and freeze
files are pinned in
[`validation/jlens-nf4-source-manifest.sha256`](validation/jlens-nf4-source-manifest.sha256).

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
