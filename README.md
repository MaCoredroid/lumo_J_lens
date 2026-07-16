# lumo_J_lens

Reproducible serving and evaluation for `nvidia/Qwen3.6-27B-NVFP4` with native
MTP speculative decoding, GDN-aware prefix caching, Qwen Code, and the
official SWE-bench Verified container runtime.

This repository is the cleaned, standalone extraction of a July 8, 2026 Claude
Code setup session. It includes the fixes found during that session, not just
the final command. On July 15, 2026 the extracted stack was rerun end to end:

- vLLM reached ready on an RTX 5090 with an 83,012-token GPU KV pool.
- Qwen Code 0.19.4 produced a 590-byte patch for `sympy__sympy-13480` in nine turns.
- The official SWE-bench 4.1.0 harness resolved the task: `1/1`, zero errors.
- Live MTP draft acceptance ranged from 92.8% to 93.3% during the episode.

See [VALIDATION.md](VALIDATION.md) for the evidence and [SPEC.md](SPEC.md) for
the complete setup rationale.

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
- `scripts/qwen_code_proxy.py`: thinking/envelope injection and context-fit retry
- `scripts/run_swe_verified.py`: Qwen Code plus official-container episode runner
- `scripts/score_verified.sh`: official SWE-bench scoring
- `scripts/fetch_chat_template.sh`: immutable, SHA-verified template fetch
- `configs/swe_image_digests.json`: certified task-image digest map
- `validation/`: exact package freezes and sanitized certified-run evidence
- `validation/runtime-source-manifest.sha256`: hashes for the certified runtime
- `docs/SESSION_RECONSTRUCTION.md`: source-session and commit chronology
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
