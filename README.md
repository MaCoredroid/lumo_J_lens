# lumo_J_lens

Reproducible serving, evaluation, and Jacobian Lens inspection for
`nvidia/Qwen3.6-27B-NVFP4` on one RTX 5090. The serving path uses native MTP
speculative decoding, GDN-aware prefix caching, Qwen Code, and the official
SWE-bench Verified container runtime. The public and locally fitted lens paths
read all 63 source layers through the packed NVFP4/FP8 target model. The
repository now also contains a native exact-forward NVFP4/FP8-STE fitter with
packed/live input VJPs, analytic GDN, and transactional dense output. Its
complete ten-prompt `n=10` fit, 63-matrix FP32 artifact, exact verifier, and
upstream `JacobianLens` load checks passed on July 17. A separate
differentiable NF4 path also produced an `n=10` lens.

This repository is the cleaned, standalone extraction of a July 8, 2026 Claude
Code setup session. It includes the fixes found during that session, not just
the final command. On July 15, 2026 the extracted stack was rerun end to end:

- vLLM reached ready on an RTX 5090 with an 83,012-token GPU KV pool.
- Qwen Code 0.19.4 produced a 590-byte patch for `sympy__sympy-13480` in nine turns.
- The official SWE-bench 4.1.0 harness resolved the task: `1/1`, zero errors.
- Live MTP draft acceptance ranged from 92.8% to 93.3% during the episode.

See [VALIDATION.md](VALIDATION.md) for the evidence and [SPEC.md](SPEC.md) for
the serving/SWE setup rationale. See
[docs/JLENS_NVFP4_REPRODUCTION.md](docs/JLENS_NVFP4_REPRODUCTION.md) for the
public FP16 lens of unpublished fit precision applied to NVFP4,
[docs/JLENS_NVFP4_STE_EXPERIMENT.md](docs/JLENS_NVFP4_STE_EXPERIMENT.md) for
the completed native NVIDIA fit contract and production evidence, and
[docs/JLENS_NF4_EXPERIMENT.md](docs/JLENS_NF4_EXPERIMENT.md) for the fresh-fit
experiment.

## Requirements

The frozen profile was tested on:

- Ubuntu 26.04, x86_64, user systemd available
- NVIDIA RTX 5090, 32,607 MiB VRAM, compute capability 12.0
- NVIDIA driver 595.71.05
- 30 GiB host RAM plus 8 GiB swap
- Docker usable by the current user without `sudo`
- uv 0.11.24, Node.js 22.23.1, npm 10.9.8, Git, curl, and about 35 GB free
  before pulling a task image

Inference runs directly on the host. Docker is CPU-side and is used only for the
official SWE-bench task environment and scorer. An NVIDIA Docker runtime is not
required.

## Quick Start

```bash
git clone https://github.com/MaCoredroid/lumo_J_lens.git
cd lumo_J_lens

scripts/setup.sh --download-model
scripts/pull_verified_image.sh sympy__sympy-13480
scripts/reproduce_one.sh sympy__sympy-13480
```

`reproduce_one.sh` enforces the safe lifecycle:

1. Preflight the host.
2. Start one 27B server in a transient user service capped at 22 GB RAM and 4 GB swap.
3. Run Qwen Code against the official per-instance container.
4. Stop the model server and wait for GPU memory to settle.
5. Run the official SWE-bench scorer.

The default model endpoint binds only to `127.0.0.1:9952`.
The reference task is reproducible because the model revision, dataset revision,
chat-template hash, dependency graph, and SymPy task-image digest are pinned.
Other task IDs fail closed until their image digest is added to
`configs/swe_image_digests.json`; `ALLOW_UNPINNED_SWE_IMAGE=1` is an explicit,
non-certified escape hatch.

Run the portable static checks independently of the GPU workflow:

```bash
scripts/check.sh
```

## Jacobian Lens On The RTX 5090

Download and fully verify the pinned 1,000-prompt lens, then apply it to every
source layer for the two frozen completion prompts:

```bash
.venv-vllm/bin/python scripts/download_jlens.py
scripts/run_jlens_nvfp4.sh \
  --prompts-file configs/jlens_prompts.json \
  --layers all --positions=-1 --top-k 10 \
  --output validation/jlens-nvfp4-local.json
```

The recertified July 16 reference run passed the independent final-residual
parity gate for both prompts, showed the expected `Italy`/`Italian` to `euro`
layer progression, allocated/reserved 25.62/27.60 GiB peak CUDA memory, loaded
the model in 8.846 seconds, and completed its measured artifact-gate through
readout lifecycle in 16.300 seconds. It uses the
exact NVIDIA ModelOpt checkpoint but disables MTP for eager residual capture;
MTP is a separate draft-token serving optimization, not part of the 64-layer
target-model lens. This applies a public FP16 lens whose fit-time model
precision and quantization were not published to quantized activations; it is
not an NVFP4 refit.

### Native NVFP4/FP8-STE fit

The NVIDIA checkpoint cannot be passed directly to `jlens.fit`: vLLM executes
packed ModelOpt/Marlin W4 and Cutlass FP8 serving kernels without the residual
activation backward, and FP8 rounding needs an explicit surrogate. The native
path instead captures the actual compiled NVFP4/FP8 forward and supplies
packed W4, live post-load FP8, analytic GDN, and full-attention input VJPs. It
uses identity STE for FP8 activation quantization, so this is an exact-forward
surrogate Jacobian rather than the literal derivative of rounding.

The completed production run recaptured all ten prompts under the hardened
checkpoint identity. For every prompt, endpoint generation was exact, 688/688
shared internal tensors were bit-exact, all 432 observer-only compiled
boundaries were present, and all 785 replay parameters matched by name, shape,
dtype, and content hash. Each prompt committed 20 row chunks. The 432
observer-only values cannot be directly compared to absent baseline tensors;
exact endpoint generation parity and the 688 direct comparisons provide the
bounded indirect evidence for the observer graph.

Plan, run, or resume the frozen ten-prompt production contract with:

```bash
.venv-vllm/bin/python scripts/run_nvfp4_ste_fit.py \
  --work-dir .cache/nvfp4_ste_fit --plan-only

.venv-vllm/bin/python scripts/run_nvfp4_ste_fit.py \
  --work-dir .cache/nvfp4_ste_fit

.venv-vllm/bin/python scripts/run_nvfp4_ste_fit.py \
  --work-dir .cache/nvfp4_ste_fit --resume
```

The contract is ten frozen 128-token prompts, `skip_first=16`, all 63 source
matrices, target block 63, and 5,120 rows per matrix. MTP is disabled because
it is a draft-token serving optimization outside the target-model lens. The
completed run took 47,577.883 seconds (13:12:57.9) and peaked at
8,936,882,688/11,404,312,576 CUDA bytes allocated/reserved. The authoritative
raw mean contains 63 little-endian FP32 `[5120,5120]` matrices totaling
6,606,028,800 bytes. The exported 6,606,046,478-byte checkpoint has SHA-256
`82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057`.
Both upstream `JacobianLens.load` and `JacobianLens.from_pretrained` passed.

Against the public `n=1000` FP16 lens, the native artifact measured global
Frobenius cosine `0.732877`, mean per-layer cosine `0.822360`, and global
relative Frobenius difference `0.934596`; these are descriptive, not post-hoc
pass thresholds. Over 1,008 paired held-out layer/position observations,
native/public Jacobian readouts measured target-rank Spearman `0.902843`, top-1
agreement `0.412698`, top-5 overlap `0.493651`, and target-score RMSE `2.780105`.
Both independently run adapter certificates failed with identical residual
manifests and reconstruction values. That certificate is evaluated before
either lens is applied and remains separate from lens-quality metrics.

### Fresh NF4 fit

A new exact, dense `n=10` lens was fitted successfully on the RTX 5090 against
`Qwen/Qwen3.6-27B` revision `6a9e13bd...`, quantized at load time to
bitsandbytes NF4 with double quantization and BF16 compute. The run used the
Anthropic future-summed VJP estimator, source layers `0..62`, target block 63,
128-token prompts, `skip_first=16`, and cotangent batches of 32. It completed
in 6,367.6 estimator seconds and 6,496.5 cumulative in-process invocation
seconds (1:48:16.5), with 23.25/24.20 GiB peak CUDA allocated/reserved. The
start/completion timestamps span 6,566.9 seconds including the deliberate
pause before resume.

The measured run deliberately stopped after its first committed prompt and
then exercised deterministic resume:

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

The 6.152 GiB FP32 artifact has SHA-256
`54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f`;
all 63 `[5120, 5120]` matrices passed the finite, shape, metadata, and provenance
gates. Held-out NF4 evaluation also completed. The local lens was then applied
to NVFP4 residuals, but that paired strict adapter certificate **failed**, as
did the public lens on the same four prompts: three full-logit max errors were
`0.125`, above the `0.0625` limit. All full-logit RMS checks and all greedy
top-1 checks passed; row 18 additionally failed the final-norm max and top-5
prefix gates. Because the adapter failures are identical with either lens,
they do not diagnose lens quality, but they do prevent certification of this
cross-quantization application.

The successful NF4 result remains an NF4 fit, not an NVFP4 fit. It is separate
from the native exact-forward NVFP4/FP8-STE implementation above. Exact NF4
evidence and interpretation are in [VALIDATION.md](VALIDATION.md) and the
[NF4 experiment report](docs/JLENS_NF4_EXPERIMENT.md); native-path evidence and
the completed production status are in the
[NVFP4/FP8-STE report](docs/JLENS_NVFP4_STE_EXPERIMENT.md).

## Manual Lifecycle

```bash
scripts/preflight.sh
scripts/start_server.sh
scripts/run_verified_task.sh sympy__sympy-13480
scripts/stop_server.sh
scripts/score_verified.sh sympy__sympy-13480
```

Use `.env.example` for the certified profile's primary overrides. Operational
scripts also expose local run names, paths, ports, timeouts, and debug controls
directly in their source. To create a local override file:

```bash
cp .env.example .env
```

The serving and evaluation scripts automatically load `.env`; setup and template
fetch use their explicit environment controls instead. Do not put API keys or
GitHub tokens in tracked files. Neither the local vLLM endpoint nor this public
model needs a real OpenAI API key; Qwen Code receives the literal placeholder
`EMPTY`.

## Outputs

Runtime artifacts are written below `runs/<timestamp>_<instance>/` and are
gitignored. Important files are:

- `generation/verified/predictions.jsonl`: official scorer input
- `dataset.json`: rows materialized from the pinned dataset revision
- `generation/verified/per_task/<id>/patch.diff`: generated patch
- `generation/verified/per_task/<id>/runner_metadata.json`: turns, tools, timing
- `proxy_dumps/`: exact forwarded sampling envelope and per-request usage
- `official_score/*.json`: official SWE-bench report

Treat the entire `runs/` tree as sensitive local state. Proxy dumps contain full
prompts and conversations; Qwen HOME state may contain local session/install
identifiers and paths. Do not ZIP, tar, or upload the working directory. Publish
only Git-tracked files after reviewing `git status` and `git ls-files`.

## Repository Map

- `scripts/serve_qwen36_27b_nvfp4_mtp.sh`: frozen vLLM server profile
- `scripts/download_jlens.py`: immutable 3.3 GB lens download and tensor gate
- `scripts/run_jlens_nvfp4.sh`: CUDA environment and offline lens launcher
- `scripts/run_jlens_nvfp4.py`: vLLM residual adapter and all-layer readout
- `scripts/run_nvfp4_ste_fit.py`: pinned, resumable native NVFP4/FP8-STE fitter
- `scripts/export_nvfp4_ste_lens.py`: validated upstream-compatible lens exporter
- `scripts/capture_nvfp4_fit_prompt.py`: isolated compiled capture/proof orchestration
- `scripts/nvfp4_packed_vjp.py`: memory-bounded raw ModelOpt W4 input VJP
- `scripts/fp8_live_vjp.py`: memory-bounded post-load FP8 input VJP
- `scripts/nvfp4_gdn.py`: analytic Gated DeltaNet recurrence and VJP
- `scripts/fit_jlens_nf4.py`: resumable exact dense NF4 fitter
- `scripts/evaluate_jlens_nf4.py`: held-out NF4 local/public/logit evaluation
- `scripts/compare_jlens_artifacts.py`: dense local/public matrix comparison
- `scripts/qwen_code_proxy.py`: thinking/envelope injection and context-fit retry
- `scripts/run_swe_verified.py`: Qwen Code plus official-container episode runner
- `scripts/score_verified.sh`: official SWE-bench scoring
- `scripts/fetch_chat_template.sh`: immutable, SHA-verified template fetch
- `configs/swe_image_digests.json`: certified task-image digest map
- `validation/`: exact package freezes and sanitized certified-run evidence
- `validation/jlens-nvfp4-2026-07-16.json`: complete lens experiment output
- `validation/jlens-nf4-fit-provenance-2026-07-16.json`: completed `n=10` fit certificate
- `validation/jlens-nf4-evidence.sha256`: hashes for the fresh-fit evidence set
- `validation/jlens-nf4-source-manifest.sha256`: hashes for fit/evaluation sources
- `validation/jlens-nvfp4-ste-source-manifest.sha256`: hashes for native fitter sources
- `validation/jlens-nvfp4-ste-fit-state-2026-07-17.json`: completed ten-prompt transactional state
- `validation/jlens-nvfp4-ste-final-metadata-2026-07-17.json`: 63-matrix raw-mean provenance
- `validation/jlens-nvfp4-ste-artifact-verification-2026-07-17.json`: exact exported-artifact verification
- `validation/jlens-nvfp4-ste-upstream-load-2026-07-17.json`: upstream loader smoke checks
- `validation/jlens-nvfp4-ste-vs-public-2026-07-17.json`: native/public matrix geometry
- `validation/jlens-native-nvfp4-ste-on-nvfp4-heldout-2026-07-17.json`: native schema-3 readout
- `validation/jlens-public-schema3-on-nvfp4-heldout-2026-07-17.json`: paired public control
- `validation/jlens-nvfp4-ste-vs-public-heldout-2026-07-17.json`: offline paired metrics
- `validation/jlens-source-manifest.sha256`: lens runner/evidence source tie
- `validation/runtime-source-manifest.sha256`: hashes for the certified runtime
- `docs/SESSION_RECONSTRUCTION.md`: source-session and commit chronology
- `docs/JLENS_NVFP4_STE_EXPERIMENT.md`: native fit contract, evidence, and runbook
- `docs/TROUBLESHOOTING.md`: every boot/runtime failure found and its fix

## Security Boundary

Qwen Code runs unattended only on the pinned official dataset row. It receives a
scrubbed environment with no inherited tokens and a fresh HOME. Its core-tool
allowlist contains only `run_shell_command`, and the shell shim sends every such
command into the task container. Native host file/search/edit tools, web,
subagent, skill, notebook, planning, and tool-discovery tools are excluded.
The agent CWD contains only `AGENTS.md`, the container runs with no network, and
the proxy rejects any request whose declared tool schema is not exactly one
well-formed shell function. The certified run forwarded that exact schema on all
nine requests, and all nine shell calls executed inside the container.
Do not use `--yolo` with untrusted or locally modified task prompts.
