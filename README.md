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
[docs/JLENS_SWE_QWEN_CODE_EXPERIMENT.md](docs/JLENS_SWE_QWEN_CODE_EXPERIMENT.md)
for the layer-by-layer replay of the certified Qwen Code episode,
[docs/JLENS_SWE_MULTITASK_CHECKPOINTS_2026-07-18.md](docs/JLENS_SWE_MULTITASK_CHECKPOINTS_2026-07-18.md)
for the leakage-audited multi-task C0/C1 probe, and
[docs/JLENS_SWE_MULTISTAGE_2026-07-18.md](docs/JLENS_SWE_MULTISTAGE_2026-07-18.md)
for the eight-stage lifecycle replay and next-step decision,
[docs/JLENS_SWE_BEHAVIORAL_N20_2026-07-18.md](docs/JLENS_SWE_BEHAVIORAL_N20_2026-07-18.md)
for the 20-task behavioral replay and probe-versus-refit decision,
[docs/JLENS_SWE_CONTEXTUAL_EVIDENCE_2026-07-18.md](docs/JLENS_SWE_CONTEXTUAL_EVIDENCE_2026-07-18.md)
for the paired contextual-evidence pilot and its guarded, non-COT task cards,
[docs/JLENS_SWE_TASK_STATE_INTERPRETER_2026-07-18.md](docs/JLENS_SWE_TASK_STATE_INTERPRETER_2026-07-18.md)
for the dense 698-request task-state readout and its negative reliability
decision,
[docs/JLENS_SWE_BINARY_PHASE_V2_2026-07-18.md](docs/JLENS_SWE_BINARY_PHASE_V2_2026-07-18.md)
for the corrected edit-versus-check/finish phase readout, serialized model, and
failed J-specific development gate,
and
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

For a fast task timeline, reuse that public `n=1000` lens and replay every
agent-loop completion in one model load:

```bash
scripts/quick_swe_jlens.py
```

Here a request means one Qwen Code chat completion between tool executions,
not a separate user task. The default extracts all nine exact completion
contexts, reads layers `24,31,32,39,40,62` at each final prompt boundary, and
writes a compact timeline under `.cache/swe_jlens_quick/`. Use
`--requests 3` or `--requests 3-6` for a smaller slice, and `--dry-run` to
inspect the exact one-load command without touching the GPU. The summary keeps
the evaluated request count separate from the lens-fit prompt count and
preserves a failed strict adapter status rather than treating it as a crash.
On the reference RTX 5090, the final verified invocation took 38.60 seconds
wall time; its measured runner lifecycle was 35.381 seconds, including an
8.613-second model load.

For a fail-closed task-wide action readout over every probeable request in the
20-task behavioral corpus, run:

```bash
scripts/run_swe_task_state_interpreter.sh
```

The dense development screen retained 639 stable labeled rows from 698
prefixes. Public J-only balanced accuracy was `0.6680`, versus `0.6717` for
ordinary logits. Adding J to the ordinary-logit-plus-context model changed
balanced accuracy by `-0.0036`, conditional 95% interval
`[-0.0532, +0.0471]`; the reliability gate failed. The lens contains an
action-phase signal, but it did not add reliable value beyond ordinary logits
and is not a hidden-COT decoder. The complete result and next-milestone
exploration are in the task-state report linked above.

The follow-up binary phase model uses the layer-resolved current, delta, and
EMA-deviation patterns to predict `edit` versus `check_or_finish`. Its corrected
operational extraction emits at 650 stable boundaries and has truth for 620.
Repository-held-out public-J accuracy/balanced accuracy is `0.8451/0.8178`,
versus `0.8235/0.7950` for the exactly matched ordinary-logit branch. The
`+0.0216` J accuracy difference has 95% interval
`[-0.0127, +0.0552]`; paired NLL and Brier intervals also cross zero. The
absolute phase signal is strong, but the frozen J-specific development screen
fails. Trajectories for the untouched reserved 20-task cohort were therefore
not generated. Reserved evaluation also requires a checker-validated prompt
summary and permits only a provenance-only null-to-literal-SHA protocol
transition, with the frozen core protocol and model unchanged.

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

### What the J-lens objective measures

The Anthropic J-lens is not trained to reconstruct the current next-token
distribution. It averages the causal Jacobian from a source residual to
final-layer residuals at present and future positions (`t' >= t`) and decodes
`J_l h_l` without a current-context affine intercept. Anthropic explicitly
reports that J-lens has the worst next-token KL
through most of the network and calls that a feature of the method. Therefore,
next-token KL, accepted-token NLL, and final-margin similarity are calibration
diagnostics, not J-lens quality or reproduction gates. The primary evaluation
is recovery of known single-token intermediate concepts, scored by minimum
rank over a fixed layer band and summarized by pass-at-k and normalized
log-rank AUC. See the pinned [method](https://transformer-circuits.pub/2026/workspace/index.html#methods-jlens),
[objective comparison](https://transformer-circuits.pub/2026/workspace/index.html#methods-compare),
[quantitative appendix](https://transformer-circuits.pub/2026/workspace/index.html#app-quant),
and [evaluation conventions](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/data/evaluations/README.md).

On Anthropic's pinned 93-item multihop set, the public lens over fixed layers
24 through 47 achieved AUC `0.62374` versus logit-lens `0.47073`, a gain of
`+0.15300` with paired item-bootstrap 95% CI `[0.11490, 0.19011]`.
Pass-at-10 was `0.29032` versus `0.13441`, a gain of `+0.15591` with CI
`[0.06452, 0.24731]`. The native NVFP4/FP8-STE lens nearly reproduced it:
AUC `0.61902`, gain `+0.14829` with CI `[0.10986, 0.18626]`, and pass-at-10
`0.27957`, gain `+0.14516` with CI `[0.05376, 0.23656]`. These are paired
descriptive results on identical residuals, not adapter certification: only
72/93 prompts passed the strict full-final-logit tolerance in either run.
Compact evidence is in
[`validation/jlens-upstream-multihop-control-analysis-2026-07-17.json`](validation/jlens-upstream-multihop-control-analysis-2026-07-17.json).

### J-lens on the certified SWE episode

The later 12-task contextual-evidence pilot compares adjacent Qwen Code
boundaries after task evidence enters the prompt. The public lens passed its
frozen directional point rule, but all four decision confidence-interval lower
bounds were nonpositive, exact-token copy baselines retrieved task targets far
better, and only 3/12 guarded entity-level `WHY` fields were emitted. Public,
NF4, and native NVFP4/FP8-STE lenses were statistically indistinguishable in
this small sample. See the
[contextual-evidence report](docs/JLENS_SWE_CONTEXTUAL_EVIDENCE_2026-07-18.md);
it explicitly does not claim chain-of-thought recovery.

The exact nine request contexts from the successful Qwen Code run were
re-rendered with the pinned template and tokenizer. Their lengths exactly
match the server ledger (`11,861` through `15,678` tokens). The native
NVFP4/FP8-STE lens and the public `n=1000` control were applied to all 63 source
layers at each next-token boundary; independently replayed residual manifests
and the logit-lens baselines match exactly.

Selected later-half top-10 readouts contain task-relevant vocabulary. After
the source was inspected, native layers 39/40 include `obvious` and `clearly`;
after reproducing the error they include `confirm` and `confirming`.
Subsequent stages contain repair/success terms and then `lack`/`unavailable`
when `pytest` is missing. These are vocabulary readouts, not literal hidden
thoughts.

At the exact original reasoning boundary after `the variable is actually `,
the same-context candidate probe compares the correct first token `co`
(`cothm`) with buggy `cot` (`cotm`). The native J-lens log-probability margin is
`+2.0664` at layer 31, `+1.9922` at layer 32, `+3.0391` at layer 35, and
`+2.3984` at layer 40. The public lens also favors `co` at every layer in the
fixed exploratory reporting slice. Both tokens are still outside the top 10
in that slice, so this
is a relative preference rather than a direct decode of `cothm`. At layer 62
the native/public margins reach `+13.5625`/`+14.25`; the final model margin is
`+13.50` and ranks the generated token `co` first. The original trace then
continues to `cothm`. Independently, the official scorer resolved the original
run's one-character `cotm` to `cothm` patch: `1/1`.

The ordinary logit-lens baseline also favors `co` at all nine middle layers,
with a larger mean margin (`+3.2440`) than either J-lens. The preference is
therefore not uniquely revealed by Jacobian transport. The dense 293-state
next-token trajectory and the earlier ten-state semantic final-margin analysis
are retained as calibration diagnostics, not quality gates: J-lens performed
worse than logit lens on their final-output targets, consistent with the
objective mismatch and Anthropic's reported behavior.

A method-aligned exploratory probe instead froze ten episode states, 17
single-token semantic intermediates, and layers 16 through 47 before scoring.
The public lens reached AUC `0.71543` versus logit-lens `0.69182`; the native
lens reached `0.69167`. Their AUC gains over logit lens were `+0.02361` with
paired item-bootstrap 95% CI `[-0.07363, 0.10226]` and `-0.00016` with CI
`[-0.09693, 0.08518]`. Public/native pass-at-10 were `0.35`/`0.25`, versus
`0.40` for logit lens. Seven items follow tool results whose concepts are
mostly already explicit in the transcript, and the exact pre-identifier item
was weak: public/native semantic ranks were `[191,525]`/`[175,787]`, versus
logit-lens `[3,463]`. This one-task adaptation had no preregistered claims
gate, so it does not establish latent reasoning or SWE-bench generalization.
Both probe reports retain `status: failed`: all ten rows pass model-greedy
top-1, top-5, and final-norm checks, but only 5/10 pass the strict full-logit
tolerance.
See the [full experiment report](docs/JLENS_SWE_QWEN_CODE_EXPERIMENT.md) and
[`validation/jlens-swe-qwen-code-intermediate-analysis-2026-07-17.json`](validation/jlens-swe-qwen-code-intermediate-analysis-2026-07-17.json).

### Multi-task SWE checkpoint probe

A second experiment freezes gold-patch concept candidates across ten
independent SWE-Verified tasks. At task start (C0), neither J-lens beats the
ordinary logit lens. The historical C1 exact-token audit retained eight tasks
and nine concepts, but a later case-folded identifier audit invalidated the
Django `lazy` row: `SimpleLazyObject` is present at both C0M and C1. The clean
candidate cohort is therefore seven tasks and eight concepts. Historical
aggregate JSON was not recomputed; its C1 point estimates (`U=0.28738` native,
`0.26247` public, and `0.25771` logit) include the exposed Django row and must
not be cited as a clean hidden-concept evaluation. All paired task-bootstrap
method-difference intervals already cross zero.

An audit caught that the original C0 and captured C1 startup contexts differed,
so those preliminary stage deltas were discarded. The corrected C0M baseline
uses exact token prefixes from the same eight trajectories. Matched C0M-to-C1
utility changes are `-0.06344/-0.04395/-0.02921` for logit/public/native. The
direct public-minus-logit and native-minus-logit stage contrasts are `+0.01949`
CI `[-0.08392, 0.12446]` and `+0.03423` CI `[-0.07364, 0.15884]`; neither is
detected. Native ranks Django's exposed `lazy` control 607th at C0M and 209th at
C1 while logit worsens from 929th to 44,630th; this is compatible with lexical
association and is not hidden-concept evidence. The next experiment must first
apply the semantic audit and same-task foil rule, then add an oracle-labeled C2
after reading the relevant source function and before target leakage, retaining
C0M and C1 as controls. The current data does not justify another fit.
See the [multi-task report](docs/JLENS_SWE_MULTITASK_CHECKPOINTS_2026-07-18.md).

This is an eager, MTP-disabled replay of frozen contexts because the original
compiled server did not retain hidden states. The strict multi-stage adapter
status is `failed`: five stages reached BF16 max logit error `0.125` against a
`0.0625` threshold. All nine final top-1 tokens, top-5 prefixes, final-norm RMS
checks, and full-logit RMS checks passed, and the longest-context preflight
passed the complete strict gate. Treat the lens result as a reproducible
descriptive analysis, not a strict adapter certification.

### Full-lifecycle SWE pilot

The one-command pilot freezes eight exact request prefixes from a separately
resolved Django trajectory, replays all of them through the public `n=1000`,
NF4 `n=10`, and native NVFP4/FP8-STE `n=10` lenses in three model loads, and
compares each readout with a same-residual logit lens:

```bash
OUT_DIR=validation/jlens-swe-multistage-2026-07-18 \
  scripts/run_swe_multistage_pilot.sh
```

Here one request is one Qwen Code model invocation in the agent/tool loop. The
eight stages map to requests `1,2,5,6,13,14,16,25`; they are correlated
checkpoints from one task, not eight independent samples. The corrected
case-folded identifier audit finds **zero hidden concept rows**. The task prompt
already contains `SimpleLazyObject`, which exposes the target `lazy` from S0;
all eight stages are therefore semantic-exposure controls. The earlier positive
`lazy`-versus-`_dims` margins are archived as a lexically primed, cross-domain
association, not evidence of hidden reasoning. `lazy` also comes from the
benchmark gold patch rather than Qwen's resolved repair, while `_dims` is an
unrelated xarray foil.

The action result remains negative: on the six certified rows, every J-lens
reaches `50.0%` micro accuracy versus a `66.7%` inspect majority baseline;
including both uncertified rows leaves J-lens at `50.0%` versus a `62.5%`
majority baseline. Edit and finalize are both missed. The outcome controls are
also non-diagnostic because the official outcome is always success and only one
transition is labeled failure. Immediate-next-action is not the J-lens training
objective: its future-summed transport can weight later validation content, so
the observed `validate` bias is a readout failure, not by itself evidence that
the fitted matrices are wrong.

The replay reports retain `status: failed` because stages S5 and S6 have
maximum final-logit error `0.125` against the strict `0.0625` bound. Their RMS,
top-five, greedy-token, and final-norm checks pass. Primary metrics exclude
those rows; inclusive values are sensitivity only. The result is therefore
`N=1` development evidence, not a generalization claim.

The next step is to refine the probe, then collect more requests and tasks with
the lenses frozen. Derive a target from the agent's actual subsequent diagnosis
or edit, reject it if any case-folded token, identifier segment, or semantic
alias is already visible, and compare it with a plausible same-task,
same-family foil. First use the existing resolved SymPy trajectory as an `N=2`
pipeline smoke test, then automate roughly 10-20 independent tasks with balanced
actions and both successful and failed validation transitions. Calibrate the
action vocabulary or train a small readout with task-held-out splits. Refit only
if method-aligned positive controls pass and the frozen lens then fails
consistently across held-out tasks. See the
[full multistage report](docs/JLENS_SWE_MULTISTAGE_2026-07-18.md).

### Twenty-task SWE behavioral study

The follow-on run automated 20 predeclared SWE-bench Verified tasks across 11
repositories. Qwen Code made 699 model requests with native one-token MTP and
produced 20 nonempty patches; the official scorer resolved 9/20. The lens
replay then sampled eight uniform request prefixes per task, 160 checkpoints
total, and applied the public `n=1000`, NF4 `n=10`, and native
NVFP4/FP8-STE `n=10` lenses to exactly paired eager NVFP4 residuals at fixed
layers 24-47.

The preregistered result is `insufficient_support`, not a hidden-reasoning
claim. Only 68/160 checkpoints passed the joint strict adapter gate, and the
future-identifier control retained four rows from one task. A disclosed
post-public-numerical-diagnostic sensitivity retained 155/160 checkpoints, but
ordinary logit lens beat the public J-lens on transport of the replay model's
own greedy token. The public lens still beat both local `n=10` artifacts, which
is compatible with a shared small-fit limitation but does not justify a refit
without a valid public-lens SWE positive control. A descriptive nested
leave-one-repository-out action classifier recovered all four action classes,
but strict support remained 66 rows and ordinary logit features reached
`0.8073` balanced accuracy versus `0.6823` for public J-lens features. A
post-hoc checkpoint-ordinal prior reached `0.6354`, exposing a strong task-phase
confound. Both action-readout support gates still failed. The next step is to
improve strict capture coverage and collect balanced, visibility-audited task
controls with the current lenses frozen. See
the [N20 report](docs/JLENS_SWE_BEHAVIORAL_N20_2026-07-18.md) and the compact
[publication evidence](validation/jlens-swe-behavioral-n20-2026-07-18/publication/),
including the exact [next-token transport result](validation/jlens-swe-behavioral-n20-2026-07-18/publication/next-token-transport-analysis.json)
and [learned action-layer result](validation/jlens-swe-behavioral-n20-2026-07-18/publication/action-layer-readout.json).

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
only Git-tracked files after reviewing `git status` and `git ls-files`. The
tracked publication subsets under `validation/`, including the full Django
trajectory used by the multistage pilot, were separately reviewed for
credentials and are intentionally reversible so the visibility and lifecycle
audits can be reproduced.

## Repository Map

- `scripts/serve_qwen36_27b_nvfp4_mtp.sh`: frozen vLLM server profile
- `scripts/download_jlens.py`: immutable 3.3 GB lens download and tensor gate
- `scripts/run_jlens_nvfp4.sh`: CUDA environment and offline lens launcher
- `scripts/run_jlens_nvfp4.py`: vLLM residual adapter and all-layer readout
- `scripts/quick_swe_jlens.py`: one-load replay of selected Qwen Code completions
- `scripts/analyze_swe_binary_phase_v2.py`: frozen fit/evaluate CLI for the matched binary phase readout
- `artifacts/swe-binary-phase-v2.manifest.json`: hash-bound model, environment, training, and canary contract
- `scripts/run_nvfp4_ste_fit.py`: pinned, resumable native NVFP4/FP8-STE fitter
- `scripts/export_nvfp4_ste_lens.py`: validated upstream-compatible lens exporter
- `scripts/capture_nvfp4_fit_prompt.py`: isolated compiled capture/proof orchestration
- `scripts/nvfp4_packed_vjp.py`: memory-bounded raw ModelOpt W4 input VJP
- `scripts/fp8_live_vjp.py`: memory-bounded post-load FP8 input VJP
- `scripts/nvfp4_gdn.py`: analytic Gated DeltaNet recurrence and VJP
- `scripts/fit_jlens_nf4.py`: resumable exact dense NF4 fitter
- `scripts/evaluate_jlens_nf4.py`: held-out NF4 local/public/logit evaluation
- `scripts/compare_jlens_artifacts.py`: dense local/public matrix comparison
- `scripts/materialize_swe_jlens_prompts.py`: exact certified-request renderer
- `scripts/analyze_swe_jlens_report.py`: bound middle-layer/candidate analysis
- `scripts/materialize_jlens_upstream_multihop.py`: pinned Anthropic control renderer
- `scripts/analyze_jlens_upstream_multihop.py`: exact-rank multihop evaluator
- `scripts/materialize_swe_jlens_trajectory.py`: dense teacher-forced state renderer
- `scripts/materialize_swe_intermediate_probes.py`: frozen SWE concept-probe renderer
- `scripts/analyze_swe_intermediate_probes.py`: paired pass-at-k/AUC evaluator
- `scripts/materialize_swe_multitask_c1_probes.py`: hash-pinned, token-aware C1 renderer
- `scripts/materialize_swe_multitask_capture_matched_probes.py`: exact C0M prefix renderer
- `scripts/analyze_swe_multitask_initial_probes.py`: C0/C0M/C1 exact-rank evaluator
- `scripts/compare_swe_multitask_checkpoints.py`: matched C0M-to-C1 stage comparator
- `scripts/run_swe_multitask_c1_pilot.sh`: one-command public/native C1 replay
- `scripts/run_swe_multistage_pilot.sh`: one-command eight-stage, three-lens replay
- `scripts/materialize_swe_multistage_probes.py`: exact lifecycle-prefix renderer
- `scripts/augment_swe_multistage_action_probes.py`: evidence-derived action labels
- `scripts/analyze_swe_multistage_probes.py`: paired concept/action/outcome analysis
- `configs/swe_intermediate_concept_probes.json`: evidence-grounded one-task probe set
- `scripts/qwen_code_proxy.py`: thinking/envelope injection and context-fit retry
- `scripts/run_swe_verified.py`: Qwen Code plus official-container episode runner
- `scripts/score_verified.sh`: official SWE-bench scoring
- `scripts/fetch_chat_template.sh`: immutable, SHA-verified template fetch
- `configs/swe_image_digests.json`: certified task-image digest map
- `validation/`: exact freezes, sanitized certified records, and publication-reviewed
  SWE J-lens reports whose exact prompts and token IDs are intentionally reversible
- `validation/jlens-nvfp4-2026-07-16.json`: complete lens experiment output
- `validation/jlens-nf4-fit-provenance-2026-07-16.json`: completed `n=10` fit certificate
- `validation/jlens-nf4-evidence.sha256`: hashes for the fresh-fit evidence set
- `validation/jlens-nf4-source-manifest.sha256`: hashes for fit/evaluation sources
- `validation/jlens-nvfp4-ste-source-manifest.sha256`: hashes for native fitter sources
- `validation/jlens-nvfp4-ste-fit-state-2026-07-17.json`: completed ten-prompt transactional state
- `validation/jlens-nvfp4-ste-final-metadata-2026-07-17.json`: 63-matrix raw-mean provenance
- `validation/jlens-nvfp4-ste-artifact-verification-2026-07-17.json`: exact exported-artifact verification
- `validation/jlens-upstream-multihop-control-analysis-2026-07-17.json`: external method-aligned control
- `validation/jlens-swe-qwen-code-intermediate-analysis-2026-07-17.json`: one-task exact-rank probe
- `validation/jlens-swe-multitask-c1-analysis-2026-07-18.json`: eight-task C1 exact-rank probe
- `validation/jlens-swe-multitask-c0m-analysis-2026-07-18.json`: matched task-start probe
- `validation/jlens-swe-multitask-c0m-c1-comparison-2026-07-18.json`: exact-prefix stage changes
- `validation/jlens-swe-multitask-evidence-2026-07-18.sha256`: C0/C1 evidence hashes
- `validation/jlens-swe-multistage-2026-07-18/`: complete three-lens lifecycle replay
- `validation/swe-multistage-django-13297/`: frozen trajectory and official score proof
- `validation/jlens-nvfp4-ste-upstream-load-2026-07-17.json`: upstream loader smoke checks
- `validation/jlens-nvfp4-ste-vs-public-2026-07-17.json`: native/public matrix geometry
- `validation/jlens-native-nvfp4-ste-on-nvfp4-heldout-2026-07-17.json`: native schema-3 readout
- `validation/jlens-public-schema3-on-nvfp4-heldout-2026-07-17.json`: paired public control
- `validation/jlens-nvfp4-ste-vs-public-heldout-2026-07-17.json`: offline paired metrics
- `validation/jlens-swe-qwen-code-analysis-2026-07-17.json`: compact SWE layer findings
- `validation/jlens-swe-qwen-code-native-nvfp4-ste-2026-07-17.json`: native all-layer replay
- `validation/jlens-swe-qwen-code-public-2026-07-17.json`: public all-layer control
- `validation/jlens-source-manifest.sha256`: lens runner/evidence source tie
- `validation/jlens-swe-qwen-code-intermediate-prompts-2026-07-17.json`: exact public ten-state SWE replay inputs
- `validation/jlens-swe-qwen-code-intermediate-prompts-summary-2026-07-17.json`: fixed 58-token scoring contract
- `validation/runtime-source-manifest.sha256`: hashes for the certified runtime
- `docs/SESSION_RECONSTRUCTION.md`: source-session and commit chronology
- `docs/JLENS_NVFP4_STE_EXPERIMENT.md`: native fit contract, evidence, and runbook
- `docs/JLENS_SWE_QWEN_CODE_EXPERIMENT.md`: certified SWE layer analysis and runbook
- `docs/JLENS_SWE_MULTITASK_CHECKPOINTS_2026-07-18.md`: C0/C1 multi-task report
- `docs/JLENS_SWE_MULTISTAGE_2026-07-18.md`: eight-stage lifecycle report
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
