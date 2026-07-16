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
surrogate derivative. Disabling MTP does not change this.

Current resolution: fit against the pinned BF16 source checkpoint quantized at
load time to bitsandbytes NF4, freeze all weights, and force the ordinary
differentiable PyTorch GDN path. Use `scripts/setup_fit.sh` and
`scripts/fit_jlens_nf4.py`. Label the result as an NF4 fit. Native fitting
through `nvidia/Qwen3.6-27B-NVFP4` remains unreproduced; applying an NF4-fitted
lens to NVFP4 activations is cross-quantization, not a fix for the missing
native backward.

## Strict NVFP4 adapter certificate fails on held-out prompts

The four-prompt held-out run completed readout, but returned status `failed`.
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
