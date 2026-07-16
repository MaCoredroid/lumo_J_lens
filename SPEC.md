# Qwen3.6-27B NVFP4 + MTP + Qwen Code SWE-bench Specification

## 1. Purpose

This specification defines a reproducible, single-GPU path for:

1. Serving the NVIDIA NVFP4 checkpoint of Qwen3.6-27B through vLLM.
2. Enabling the checkpoint's native MTP head for speculative decoding.
3. Using GDN-aware prefix reuse with Mamba cache `align` mode.
4. Driving the model with Qwen Code in headless agent mode.
5. Giving the agent the exact official SWE-bench per-instance runtime.
6. Scoring the resulting patch with the official SWE-bench harness.

The target is correctness and reproducibility on one RTX 5090. This is not a
generic multi-GPU deployment guide and does not claim that every flag is optimal
on another GPU, CUDA toolchain, or vLLM release.

## 2. Acceptance Criteria

The setup is successful only when all of the following hold:

- `/v1/models` returns the served name and a 32,768-token context.
- The model, dataset, template, and certified task image match immutable pins.
- Startup resolves quantization to ModelOpt mixed NVFP4 and KV cache to FP8 E4M3.
- The engine reports an MTP draft model and nonzero speculative acceptance.
- Prefix caching, chunked prefill, Mamba `align`, and Triton GDN prefill are on.
- Qwen Code uses version 0.19.4 and can emit parsed `qwen3_xml` tool calls.
- Shell tools run inside the official task container, not in an incomplete host checkout.
- A nonempty `predictions.jsonl` is produced.
- The official harness completes with no infrastructure error.

The publication-certified reference run satisfies all of these criteria and resolves
`sympy__sympy-13480`.

## 3. Frozen Bill of Materials

### Host

| Component | Validated value |
|---|---|
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-27 |
| CPU architecture | x86_64 |
| GPU | NVIDIA GeForce RTX 5090 |
| VRAM | 32,607 MiB |
| Compute capability | 12.0 (`sm_120`) |
| Driver | 595.71.05 |
| Host memory | 30 GiB RAM, 8 GiB swap |
| Docker | 29.1.3 |
| Node/npm | 22.23.1 / 10.9.8 |
| uv | 0.11.24 |

### Inference environment

| Package | Version |
|---|---|
| Python | 3.12.13 |
| vLLM | 0.23.0 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.12.1 |
| FlashInfer | 0.6.12 |
| Triton | 3.6.0 |
| compressed-tensors | 0.17.0 |
| CUDA toolkit wheel | 13.0.2; bundled nvcc reports 13.2 |

The complete observed environment is in `validation/vllm-freeze.txt`; setup
uses it as the exact `uv pip sync` input rather than allowing transitive
packages to drift.

### Agent and scorer environment

| Package | Version |
|---|---|
| Qwen Code | 0.19.4 |
| SWE-bench | 4.1.0 |
| datasets | 5.0.0 |
| docker-py | 7.1.0 |
| huggingface-hub | 1.22.0 |

The complete scorer environment is in `validation/swe-freeze.txt`, which is
also its exact `uv pip sync` input.

## 4. Model Contract

Use the official checkpoint:

```text
nvidia/Qwen3.6-27B-NVFP4
revision/snapshot: 0893e1606ff3d5f97a441f405d5fc541a6bdf404
```

The reconstructed session verified:

- Three safetensor shards and 2,194 indexed tensors.
- Indexed tensor bytes: 21,921,428,072.
- `text_config.mtp_num_hidden_layers=1`.
- Fifteen `mtp.*` tensors are present in BF16.
- `hf_quant_config.exclude_modules` excludes `mtp*` and `mtp.layers.0*` from NVFP4.
- The checkpoint declares NVFP4/FP8 ModelOpt quantization.

Do not substitute a community quantization without inspecting its tensor index.
The community checkpoint evaluated in the source session did not contain the MTP
weights and therefore could not provide the required native speculative path.

## 5. Installation

The inference environment and scorer environment are intentionally separate.
This prevents SWE-bench dependencies from perturbing vLLM's tightly coupled
Torch/CUDA packages.

```bash
uv venv --python 3.12.13 .venv-vllm
sed 's/+cu130//' validation/vllm-freeze.txt > /tmp/vllm-constraints.txt
uv pip sync --python .venv-vllm/bin/python /tmp/vllm-constraints.txt \
  --strict --torch-backend cu130

uv venv --python 3.12.13 .venv-swe
uv pip sync --python .venv-swe/bin/python validation/swe-freeze.txt --strict

npm ci
scripts/fetch_chat_template.sh
.venv-vllm/bin/hf download nvidia/Qwen3.6-27B-NVFP4 \
  --revision 0893e1606ff3d5f97a441f405d5fc541a6bdf404
```

The equivalent automated command is:

```bash
scripts/setup.sh --download-model
```

No Conda installation is needed on the host. Each official SWE-bench task image
already contains its prepared `testbed` Conda environment.

## 6. Frozen Server Profile

The normative vLLM arguments are:

```text
--dtype bfloat16
--quantization modelopt_fp4
--kv-cache-dtype auto
--gpu-memory-utilization 0.85 (dynamic, see below)
--max-model-len 32768
--max-num-batched-tokens 4096
--max-num-seqs 2
--gdn-prefill-backend triton
--enable-chunked-prefill
--enable-prefix-caching
--mamba-cache-mode align
--mamba-block-size 1024
--mamba-ssm-cache-dtype float32
--speculative-config {"method":"qwen3_5_mtp","num_speculative_tokens":1}
--attention-backend TRITON_ATTN
--no-enable-flashinfer-autotune
--enable-auto-tool-choice
--tool-call-parser qwen3_xml
--reasoning-parser qwen3
```

vLLM 0.23.0 warns that `qwen3_5_mtp` is deprecated and normalizes it to `mtp`.
The older spelling is retained because it was the exact checkpoint-family path
selected and certified by the source session.

### Dynamic GPU utilization

Desktop VRAM use varies. The launcher computes:

```text
gmu = min(0.85, (total_mib - used_mib - 1800) / total_mib)
```

It refuses to boot below `0.74`. The 1,800 MiB reserve covers the difference
between `nvidia-smi` free memory and the free memory observed by Torch during
vLLM's startup check. A persistent desktop compositor is not mistaken for a
leaked model server.

### Why two sequences and no host KV offload

The original grid tested sequence counts 2 and 3 with KV offload 0, 4, and 8 GiB.
The frozen selection is `seqs=2`, `offload=0`:

| Profile | GPU KV capacity | Aggregate smoke | Cage RSS | Decision |
|---|---:|---:|---:|---|
| seq2/off0 | 83,012 tokens, 2.53 x 32k | 102.6 tok/s | 11.51 GiB historical | Frozen |
| seq2/off4 | 77,550 tokens, 2.37 x 32k | 100.7 tok/s | 14.55 GiB | Net negative |
| seq2/off8 | 77,550 tokens, 2.37 x 32k | 98.5 tok/s | 20.74 GiB | Too little RAM headroom |
| seq3/off0 | 83,012 tokens, 2.53 x 32k | 200.5 tok/s | 19.39 GiB | Oversubscribes three max contexts |

Offload remains an opt-in experiment, hard-capped at 8 GiB. It is not the
default and its value must serialize as an integer when integral (`8`, not `8.0`).

### Process and memory isolation

`start_server.sh` launches a transient user service with:

```text
MemoryMax=22G
MemorySwapMax=4G
```

Only one heavy model process may run. An earlier July 15 capture of the same
server profile reached 19.61 GiB service memory, 20.88 GiB peak, and 0.97 GiB
swap while remaining within the cgroup. GPU memory was 28,261 MiB after boot
and 29,747 MiB after the task.
The start and task wrappers reject an endpoint unless `/v1/models` contains the
exact served name and a 32,768-token context. A compatible server not owned by
the transient user service is not silently adopted by `reproduce_one.sh`.

## 7. Required `sm_120` CUDA Workarounds

The Python CUDA wheel is internally version-skewed on the validated host: nvcc
reports 13.2 while `cuda.h` reports CUDA 13.0. FlashInfer's vendored CCCL rejects
that mismatch, and the wheel ships versioned runtime libraries without the
unversioned developer links needed by JIT linking.

The launcher therefore applies all of the following:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS="FlashInferFP8ScaledMMLinearKernel,FlashInferCutlassNvFp4LinearKernel,FlashInferTrtllmNvFp4LinearKernel,FlashInferCudnnNvFp4LinearKernel"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
```

It also creates unversioned `.so` symlinks below `.cache/cuda_dev_links` and
adds that directory plus the CUDA wheel's library directory to `LIBRARY_PATH`.
The CUDA library directory is added to `LD_LIBRARY_PATH`.

The disabled FlashInfer entries are linear kernels. On this machine vLLM falls
back to Cutlass FP8 and Marlin weight-only FP4. Full-attention prefill may still
select a FlashInfer attention kernel and JIT it with the compatibility define.

When KV offload is disabled, use:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

When KV offload is enabled, clear that variable. The OffloadingConnector rejects
expandable segments because CUDA virtual-memory remapping can invalidate the
pinned host buffer.

## 8. Prefix Cache and MTP Semantics

Qwen3.6 is a hybrid GDN/full-attention model. Correct prefix reuse must restore
both attention KV state and recurrent Mamba/GDN state. The validated serving
configuration is:

- Prefix caching enabled.
- Chunked prefill enabled.
- Mamba cache mode `align`.
- Mamba block size 1024.
- Mamba SSM cache dtype FP32.
- GDN prefill backend Triton.

Mode `all` was not supported with the hybrid MTP path. vLLM marks Mamba `align`
prefix caching experimental; the warning is expected, but output must still be
validated on the target workload.

The native MTP head drafts one token. Historical A/B measurements were 109.4
tok/s with MTP versus 68.8 tok/s without it, a 1.59x increase at 93.02% draft
acceptance. The final source-tied publication run observed acceptance windows of
92.8% and 93.3%.

## 9. Qwen Code Pairing

Qwen Code is pinned in `package-lock.json` at 0.19.4. The episode runner creates
an isolated writable HOME per task and launches:

```text
qwen --yolo --output-format json --max-session-turns 50 --max-wall-time 840s -p <prompt>
```

The source flywheel profile used an effectively unlimited 100,000-turn cap. The
certified data-generation gate intentionally bounded a single task at 50 turns,
900 seconds in the parent runner, and 840 seconds in Qwen Code. Those bounded
values are the defaults here.

Qwen Code receives:

```text
OPENAI_API_KEY=EMPTY
OPENAI_BASE_URL=http://127.0.0.1:<proxy>/v1
OPENAI_MODEL=qwen3.6-27b-nvfp4
QWEN_MODEL=qwen3.6-27b-nvfp4
QWEN_CODE_MAX_OUTPUT_TOKENS=32768
QWEN_STREAM_IDLE_TIMEOUT_MS=240000
QWEN_CODE_SUPPRESS_YOLO_WARNING=1
QWEN_SERVE_CDP_TUNNEL_OVER_WS=1
NO_BROWSER=1
NO_COLOR=1
HOME=<fresh per-task qwen_home>
```

`EMPTY` is a literal compatibility placeholder. It is not a credential and the
local proxy ignores authentication.

The child process does not inherit the operator's environment. It receives only
basic locale/process variables and the explicit values above, uses the new HOME,
and has a one-item core-tool allowlist: `run_shell_command`. Native file/search/
edit tools plus web, subagent, skill, notebook, planning, and tool-discovery
tools are excluded. The only remaining tool is redirected into the official
container, which is the actual filesystem boundary.

### Compatibility proxy

Qwen Code does not directly send the final certified sampling/thinking settings.
`qwen_code_proxy.py` forwards OpenAI-compatible chat requests and injects:

```json
{
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 0.0,
  "chat_template_kwargs": {"enable_thinking": true}
}
```

It adds a deterministic base seed plus a monotonic request index unless the
client already supplied a seed. A per-launch random health token also prevents
a stale process on the proxy port from being mistaken for the new proxy.

### Context-fit clamp

The served context is 32,768 tokens, while Qwen Code may request 8,192 output
tokens. Large SWE prompts can overflow. The proxy uses two defenses:

1. A preemptive character-count estimate reduces `max_tokens` before forwarding.
2. If vLLM returns a context-length HTTP 400, the proxy parses vLLM's limit/input
   report and retries with a smaller budget. Later retries halve the prior budget
   because some vLLM errors report only a lower bound for input tokens.

The exact helper contract is covered by 20 unit tests. A request with 24,577
input tokens, requested output 8,192, context 32,768, and margin 64 clamps to
8,127. If fewer than 256 output tokens remain, the proxy preserves the upstream
error and the runner labels the episode `ctx_overflow` instead of an honest miss.

## 10. Official SWE Runtime Alignment

Running the agent in a bare host checkout is invalid for SWE-bench. The official
image includes an editable install, dependencies, compiled artifacts, and a
prepared `testbed` Conda environment.

The runner uses a workspace-mount plus `docker exec` pattern:

1. Resolve the official image from the instance ID and certified digest map. The
   reference task uses:

   ```text
   swebench/sweb.eval.x86_64.sympy_1776_sympy-13480@sha256:3a985bfd2f3430af337ad8f34793964d1f8845485fcee129c3f166dfba6c5e43
   ```

2. Materialize the selected dataset row from revision
   `c104f840cc67f8b6eec6f759ebc8b2693d585d4a` into the run directory.
3. Create a temporary container and copy its `/testbed` tree to the host.
4. Bind-mount that seeded tree back onto `/testbed` in a long-lived task container
   with network disabled, 6 GiB memory, 8 GiB memory-plus-swap, and 1,024 PIDs.
5. Create an isolated host CWD containing only a copy of `AGENTS.md`, then run
   Qwen Code there with the one-item `run_shell_command` core-tool allowlist.
6. Put a generated `bash` shim first on `PATH`; every inspection, edit, and test
   is forwarded through the sole declared tool to:

   ```text
   docker exec <container> /bin/bash -lc \
     'source /opt/miniconda3/bin/activate testbed && cd /testbed && ...'
   ```

7. Extract `git diff --binary <base_commit>` from inside the container.
8. Restore ownership and remove the task container/workspace.

The task container is launched with `--network none`. This lets the agent inspect,
edit, import, and test the project in the same environment the official evaluator
uses without exposing the repository to Qwen Code's native host-side file tools.

## 11. Generation and Scoring Lifecycle

Do not score while the 27B server is active on a 30 GiB host. The official task
container and scorer can create additional memory pressure. The required order is:

```bash
scripts/start_server.sh
scripts/run_verified_task.sh sympy__sympy-13480
scripts/stop_server.sh
scripts/score_verified.sh sympy__sympy-13480
```

The scorer command is equivalent to:

```bash
.venv-swe/bin/python -m swebench.harness.run_evaluation \
  --dataset_name runs/<run>/dataset.json \
  --split test \
  --predictions_path runs/<run>/generation/verified/predictions.jsonl \
  --instance_ids sympy__sympy-13480 \
  --run_id <unique-id> \
  --namespace swebench \
  --cache_level env \
  --clean False \
  --max_workers 1 \
  --timeout 1800
```

The Qwen CLI exit code is never the benchmark verdict. A loop detector or hard
wall can exit nonzero after useful edits. Only the official harness report is
used to call a task resolved.

## 12. Operator Controls

The serving and evaluation scripts load an optional repository-root `.env`
before applying defaults. Setup and template fetch consume only explicitly
exported environment variables. The certified profile needs no overrides, but
these controls are supported:

| Variable | Default / meaning |
|---|---|
| `MODEL_PATH` | `nvidia/Qwen3.6-27B-NVFP4` |
| `MODEL_REVISION` | pinned snapshot `0893e160...` |
| `SERVED_MODEL_NAME` | `qwen3.6-27b-nvfp4` |
| `PORT` / `ENDPOINT` | `9952` / derived loopback OpenAI endpoint |
| `CHAT_TEMPLATE` | fetched `configs/qwen3-openai-codex.jinja` |
| `MAX_MODEL_LEN` | `32768` |
| `MAX_NUM_BATCHED_TOKENS` | `4096` |
| `MAX_NUM_SEQS` | `2` |
| `NUM_SPEC_TOKENS` | `1` |
| `QUANTIZATION` | `modelopt_fp4` |
| `KV_CACHE_DTYPE` | `auto`, resolved to checkpoint FP8 E4M3 |
| `KV_OFFLOAD_GB` | `0`, accepted range 0-8 |
| `ATTENTION_BACKEND` | `TRITON_ATTN` |
| `GPU_MEMORY_UTILIZATION` | `auto`, bounded calculation in section 6 |
| `UNIT` | `lumo_j_lens_qwen27b` user-service name |
| `BOOT_TIMEOUT` | `620` seconds |
| `LOG_PATH` | `runs/server.log` unless a run wrapper overrides it |
| `RUN_NAME` | UTC timestamp plus task ID |
| `OUT_ROOT` | `runs/<run>/generation` |
| `PROXY_PORT` | `30032` for the certified wrapper |
| `SWE_DATASET` | `princeton-nlp/SWE-bench_Verified` |
| `SWE_DATASET_REVISION` | pinned revision `c104f840...` |
| `SWE_RUN_ID` | sanitized `lumo_j_lens_<run>` |
| `PREDICTIONS` / `DATASET_JSON` / `REPORT_DIR` | derived paths under the selected run |
| `VLLM_BIN` / `VLLM_PY` | inference-environment executable overrides |
| `SWE_PYTHON` | scorer/runner Python override |
| `QWEN_BIN` / `HF_BIN` | Qwen Code and Hugging Face CLI overrides |
| `ALLOW_UNPINNED_SWE_IMAGE` | `0`; setting `1` makes the run non-certified |
| `SWE_CONTAINER_MEMORY` | `6g` task-container memory ceiling |
| `SWE_CONTAINER_MEMORY_SWAP` | `8g` total task-container memory-plus-swap ceiling |
| `SWE_CONTAINER_PIDS_LIMIT` | `1024` task-container PID ceiling |

The sampling controls are `LUMO_ENABLE_THINKING`,
`LUMO_PROXY_FORCE_TEMPERATURE`, `LUMO_PROXY_FORCE_TOP_P`,
`LUMO_PROXY_FORCE_TOP_K`, `LUMO_PROXY_FORCE_MIN_P`,
`LUMO_PROXY_FORCE_PRESENCE_PENALTY`, and `LUMO_PROXY_FORCE_SEED`. Their
certified values are in `.env.example`; changing any of them creates a different
evaluation profile.

`CHAT_TEMPLATE_URL` and `CHAT_TEMPLATE_SHA256` exist for controlled template
replacement. The fetch still refuses a hash mismatch. `HF_HUB_OFFLINE=1` may be
passed after the pinned snapshot is cached. Minimum preflight gates can be raised
with `MIN_COMPUTE_CAPABILITY`, `MIN_RAM_KIB`, and `MIN_SWAP_KIB`; lowering them
leaves the certified hardware envelope.

## 13. Reproducibility and Security Rules

- Keep the server bound to loopback unless an authenticated reverse proxy is added.
- Never commit `.env`, Hugging Face credentials, GitHub tokens, or Qwen state.
- Use the official model snapshot and official per-instance SWE image.
- Refuse mutable image tags unless the operator explicitly opts out of certification.
- Materialize and score the same row from the pinned dataset revision.
- Never pass the operator's full environment into Qwen Code.
- Give Qwen Code an isolated HOME and CWD; declare only `run_shell_command`.
- Reject forwarded requests unless their tool schema is exactly the one shell tool.
- Run the unattended task container with no network.
- Keep generation and scoring artifacts separate.
- Preserve `model_patch` exactly; do not normalize it before scoring.
- Use a unique scorer run ID for every attempt.
- Run one model server at a time.
- Record the Qwen Code version in `model_name_or_path`.
- Treat mock/gold comparison as plumbing only, never as a benchmark score.
- Treat any modified or unpinned task prompt as untrusted; do not run it unattended.

## 14. Known Warnings

The following warnings occurred in a successful certified run and are documented,
not silently suppressed:

- `qwen3_5_mtp` is deprecated and normalized to `mtp`.
- Mamba `align` prefix caching is experimental.
- FP8 KV scaling falls back to 1.0 because explicit scales are absent.
- The 5090 path uses Marlin weight-only FP4 after FlashInfer linear kernels are disabled.
- Full CUDA graph mode may be reduced to piecewise for FlashInfer attention plus spec decode.
- `min_p` has no effect under this vLLM speculative-decoding implementation; the
  pinned value is zero, so this does not change the effective distribution.
- Shutdown may report one leaked Python semaphore after forced worker teardown.

Any different error, a missing MTP model, malformed tool calls, or zero draft
acceptance is a failed preflight/acceptance gate, not an expected warning.
