# Troubleshooting

This file records the actual failure sequence from the source session.

## `invalid literal for int()` on KV offload

Symptom:

```text
ValueError: invalid literal for int() with base 10: '8.0'
```

Cause: vLLM's human-readable integer parser expects an integer representation.

Fix: serialize integral values as `8`, not `8.0`. The launcher clamps the value
to `[0, 8]` GiB and emits an integer when possible.

## `FLASH_ATTN` fails with FP8 KV

Cause: the selected path did not support the checkpoint's FP8 KV cache.

Fix: explicitly use `ATTENTION_BACKEND=TRITON_ATTN`. Do not rely on the launcher's
older `FLASH_ATTN` fallback.

## FlashInfer reports CUDA compiler/header incompatibility

Symptom contains:

```text
CUDA compiler and CUDA toolkit headers are incompatible
```

Cause: the CUDA wheel's nvcc reports 13.2 while its headers declare 13.0. CCCL
rejects the minor-version skew during FlashInfer JIT compilation.

Fix:

```bash
export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
```

Also disable the four FlashInfer FP8/NVFP4 linear kernels listed in `SPEC.md` so
vLLM selects compiled Cutlass/Marlin alternatives.

## JIT linker cannot find `-lcudart`

Cause: CUDA wheel libraries are versioned (`libcudart.so.13`) but do not include
the unversioned development symlink (`libcudart.so`).

Fix: let the launcher generate `.cache/cuda_dev_links/*.so` and add the link
directory plus the CUDA wheel lib directory to `LIBRARY_PATH`. Add the CUDA lib
directory to `LD_LIBRARY_PATH`.

The launcher deletes and rebuilds these links on every start. If they point into
another checkout or an old virtual environment, do not preserve them; a stale
absolute target can make the server appear portable while silently using the
wrong CUDA wheel.

## OffloadingConnector rejects expandable segments

Cause: CUDA VMM remapping under `expandable_segments:True` can invalidate the
pinned host KV-offload buffer.

Fix: the launcher clears `PYTORCH_CUDA_ALLOC_CONF` whenever `KV_OFFLOAD_GB > 0`.
It uses expandable segments only for the frozen offload-zero path.

## Model boots but MTP is absent

Check the checkpoint index and config. Required evidence:

- `text_config.mtp_num_hidden_layers=1`
- 15 `mtp.*` tensors
- vLLM resolves a `Qwen3_5MTP` draft architecture
- runtime logs report speculative acceptance

A quantized checkpoint without these tensors cannot be repaired with a serving
flag. Download the official NVIDIA checkpoint.

## HTTP 400: prompt plus output exceeds 32k

The proxy should pre-clamp output and retry exact context-overflow 400 responses.
Inspect `proxy.log` for the retry ladder. If the prompt leaves fewer than 256
output tokens, the task is genuinely outside this served context and is labeled
`ctx_overflow`.

Do not label this as an honest model give-up or empty patch.

## Server unit is stale or already loaded

The start script runs:

```bash
systemctl --user reset-failed lumo_j_lens_qwen27b.service
```

before creating the transient service. If manual intervention is needed:

```bash
systemctl --user stop lumo_j_lens_qwen27b.service
systemctl --user reset-failed lumo_j_lens_qwen27b.service
```

## Server reaches ready and immediately exits

This occurred during extraction when a background `systemd-run --scope` remained
tied to the short-lived launch shell. The final standalone wrapper uses a
persistent transient `.service` with the same memory limits. Do not replace it
with a background scope unless the owning orchestrator itself stays alive.

## Official task imports fail in the agent episode

Cause: the agent is operating on a bare host checkout or a bind mount that hid
the image's prepared `/testbed` artifacts.

Fix: use `--runtime container`. The runner copies `/testbed` from the official
image before bind-mounting it back. Shell commands must flow through the generated
Docker-exec shim.

## Proxy rejects Qwen Code with HTTP 422

The publication profile requires every forwarded request to declare exactly one
tool named `run_shell_command`. A 422 response reporting a tool-boundary mismatch
means Qwen Code exposed an unexpected native, extension, web, or discovery tool.
Keep the core-tool allowlist and exclusion list from
`scripts/run_verified_task.sh`; do not weaken the proxy check to make the
request pass.

## Task image is present but rejected as unpinned

The certified resolver compares the official image reference and local image ID
to `configs/swe_image_digests.json`. Pull the digest-qualified image with
`scripts/pull_verified_image.sh <instance_id>`. For a new task, independently
resolve its official digest and add it to the map before unattended execution.
`ALLOW_UNPINNED_SWE_IMAGE=1` is a deliberate non-certified escape hatch, not a
fix for a missing pin.

## Python packages drift from the freeze

Both virtual environments are exact syncs, not minimum-version installs. Rerun
`scripts/setup.sh` so `uv pip sync --strict` removes extra packages and restores
the recorded versions. `scripts/check.sh` fails if either environment's sorted
freeze differs or `uv pip check` reports an incompatibility.

## NF4 loader fails before model construction because Accelerate is missing

Symptom: `fit_jlens_nf4.py` writes initial state, but `model_execution` remains
null and Transformers stops while processing `device_map={"": 0}` or
`low_cpu_mem_usage=True`. The abandoned first diagnostic records:

```json
{"versions": {"accelerate": "missing"}}
```

Cause: Transformers' quantized single-device loader delegates placement to
Accelerate. The initial fit environment omitted it.

Fix:

```bash
scripts/setup_fit.sh
.venv-fit/bin/python -c 'import accelerate; print(accelerate.__version__)'
scripts/check_fit.sh
```

The pinned version is `accelerate==1.14.0`. Start the retry with a fresh
`--work-dir`; do not resume the pre-fix state. Fit resume intentionally binds
the package contract, so installing Accelerate changes the runtime identity.

## `jlens.fit` cannot backpropagate through the NVIDIA NVFP4 checkpoint

Cause: the vLLM ModelOpt/Marlin NVFP4 and FP8 deployment kernels expose the
serving forward but not the residual activation backward required by the
Anthropic estimator. FP8 activation rounding also needs an explicitly declared
surrogate derivative. Disabling MTP does not create autograd support.

There are now two distinct resolutions:

- For a differentiable fallback, fit the pinned BF16 source checkpoint after
  load-time bitsandbytes NF4 quantization. Use `scripts/setup_fit.sh` and
  `scripts/fit_jlens_nf4.py`, and label the result NF4. Applying it to NVFP4
  activations remains cross-quantization.
- For the native NVIDIA path, use `scripts/run_nvfp4_ste_fit.py`. It captures
  the exact compiled NVFP4/FP8 forward, then replays packed ModelOpt W4 and
  live post-load FP8 input VJPs with identity STE and analytic GDN. Label the
  result `NVFP4/FP8-STE`; it is not the literal derivative of rounding. The
  measured production run recaptured all ten prompts under hardened identity,
  completed 63 matrices, and passed exact artifact plus upstream-loader checks.
  Do not substitute the old exploratory capture evidence, which predates that
  binding and remains non-reusable.

Do not point the native runner at a different quantized checkpoint. Its model
identity, metadata, all three shard hashes, prompt manifest, and source contract
are pinned and revalidated before prompt commit.

## Native NVFP4 capture is large or appears to run twice

This is expected. Every prompt uses isolated unmodified compiled-baseline and
compiled-observer processes. The baseline proves the serving endpoint; the
observer adds the 432 linear/SwiGLU/post-block boundaries needed by reverse
replay. The verifier directly compares 688 shared GDN/attention tensors and all
785 replay parameters, then deletes the large baseline tensor payload after
recording its hash. Only the observer payload is retained for fitting.

Do not interpret the 432 observer-only tensors as direct baseline equality.
They are absent from the baseline by design. Their evidence is required
completeness plus exact endpoint generation parity and the 688 direct internal
comparisons for the same prompt and runtime shape.

## Native NVFP4 fit was interrupted

Resume the same work directory:

```bash
.venv-vllm/bin/python scripts/run_nvfp4_ste_fit.py \
  --work-dir .cache/nvfp4_ste_fit \
  --resume
```

Do not delete or edit `state.json`, `current/`, `sum-NNNNNN/`, or the retained
prompt capture. Resume validates the contract, completed row-chunk hashes,
committed sums, checkpoint shards, and capture binding before doing more work.
A contract mismatch is a fail-closed provenance error, not a reason to edit the
state file.

For the completed reference run, `state.json` must report `status: completed`,
`n_done: 10`, `next_prompt: 10`, and `sum_generation: 10`. Its exact SHA-256 is
`f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6`.
The final metadata SHA-256 is
`289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601`,
and the exported checkpoint SHA-256 is
`82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057`.
If any value differs, treat it as a different run rather than editing evidence
to match.

## Native fitter rejects multimodal MRoPE positions

The production contract is text-only. Equal triplicated MRoPE axes reduce to
the text position vector, but divergent three-axis image/video positions are
rejected because that derivative path has not been validated. Use text-only
128-token fit prompts; do not bypass the check and call the result reproduced.

## Strict NVFP4 adapter certificate fails on held-out prompts

The historical July 16 four-prompt NF4/public held-out run completed readout,
but returned status `failed`.
Rows 3, 18, and 49 had full-logit max error `0.125`, above the `0.0625` gate;
row 42 was exactly at the allowed maximum. All logit RMS errors were below
`0.01`, and all greedy top-1 checks passed. Row 18 also had final-norm max
error `0.25 > 0.125` and a nonmatching top-5 prefix.

This is not specific to the local NF4 lens. Repeating the identical prompts
with the public lens produced the same reconstruction errors because adapter
parity is evaluated before applying a lens matrix. Do not weaken the gates or
mark the local cross-application certified. Preserve both failed records:

- `validation/jlens-nf4-on-nvfp4-2026-07-16.json`
- `validation/jlens-public-on-nvfp4-heldout-2026-07-16.json`

The original two-prompt public-lens baseline is a distinct passing regression,
not evidence that every prompt satisfies the adapter's strict max-error bound.

The July 17 native NVFP4/FP8-STE and public schema-3 reports independently
reproduced those same adapter values. Their four residual-capture manifests and
all logit-lens baseline fields matched exactly. Preserve these records too:

- `validation/jlens-native-nvfp4-ste-on-nvfp4-heldout-2026-07-17.json`
- `validation/jlens-public-schema3-on-nvfp4-heldout-2026-07-17.json`
- `validation/jlens-nvfp4-ste-vs-public-heldout-2026-07-17.json`

The paired report still contains the lens measurements: target-rank Spearman
`0.902843`, top-1 agreement `0.412698`, top-5 overlap `0.493651`, and
target-score RMSE `2.780105` over 1,008 observations. Adapter parity is
lens-independent and must be reported separately from these lens-quality
metrics.

## Scorer has no report

Confirm:

- The model server is stopped.
- `predictions.jsonl` contains only IDs present in the selected dataset.
- `model_patch` is nonempty.
- The exact official instance image exists.
- The current user can access Docker without sudo.
- The run ID is unique or old task containers/reports were cleaned.

Mixed SWE-Gym and Verified predictions must be partitioned before scoring; the
official harness validates every prediction ID against its dataset before it
applies the `--instance_ids` filter.

## Expected successful-run warnings

These are known and recorded in the validation:

- MTP method name deprecation/normalization.
- Experimental Mamba align prefix caching.
- FP8 KV scale fallback to 1.0.
- Marlin weight-only FP4 fallback.
- Full CUDA graph reduced to piecewise for this attention/spec combination.
- A resource-tracker semaphore warning during forced process teardown.

They should not be generalized away. Recheck them after any vLLM, CUDA, driver,
or checkpoint change.
