# Qwen3.6 NVFP4 J-Lens SWE Behavioral N20 Study

**Date:** 2026-07-18
**Execution status:** complete
**Preregistered scientific decision:** `insufficient_support`

## Bottom Line

The engineering reproduction succeeded. Qwen Code ran 20 SWE-bench Verified
tasks through the pinned `nvidia/Qwen3.6-27B-NVFP4` server with one-token MTP,
produced a nonempty patch for every task, and completed 699 model requests. The
official SWE-bench 4.1.0 scorer resolved 9/20 patches. Eight uniformly spaced
request prefixes per task, 160 checkpoints total, were then replayed through
the same pinned main-model checkpoint in eager mode with MTP disabled. Public
`n=1000`, local NF4 `n=10`, and local native NVFP4/FP8-STE `n=10` J-lenses were
applied to exactly paired residual captures.

The scientific result is not a successful semantic SWE-stage readout. Under
the frozen primary protocol, only 68/160 checkpoints passed the joint strict
adapter gate, only 66 of those had action labels, only 8/20 tasks contributed a
strictly certified official-outcome row, and the future-identifier track had
four eligible rows from one task. Those values miss the preregistered support
thresholds. The required decision is therefore `insufficient_support`, with
the next step `collect the missing jointly certified task-level controls; do
not refit`.

Two supplemental analyses sharpen that answer without overriding it:

- A reconstruction-sensitivity amendment, frozen after public numerical
  diagnostics but before any lens ranks or log probabilities were inspected,
  retained 155/160 checkpoints. On the replay model's own greedy next token,
  ordinary logit lens decisively beat the public J-lens. This means that the
  control does not validate the J-lens readout, even though the public lens
  ranks the target better than both local `n=10` lenses.
- A repository-held-out learned action readout tests whether fixed vocabulary
  scores may retain useful layer structure that band averaging can discard. Its result is
  reported separately below. It is a classifier over lens features, not a new
  J-lens fit and not evidence of hidden prose.

The practical conclusion is to refine the measurement and data collection
before another 13-hour native lens fit. The existing public and native lenses
already show a fixed, paired advantage on an external, method-aligned multihop
control. What is missing is a well-supported SWE-specific target and readout.

## What Was Evaluated

A **Qwen Code request** is one `/v1/chat/completions` model completion in the
agent/tool loop. It is not an independent coding task. Completion N may issue
one or more tool calls; their results enter request N+1. The 20 trajectories
contain 699 requests: 267 in the development cohort and 432 in the replication
cohort.

The lens did not run at every request or every token. The materializer selected
eight uniform probeable request prefixes per task without using the action
label or lens output. It replayed the final prompt position (`-1`) at each of
those 160 checkpoints and scored the fixed source-layer band 24 through 47.
The complete agent trajectories determine the prefixes, subsequent-action
labels, generated patches, and official outcomes; the lens analysis is a
checkpoint sample across the whole task.

| Unit | Count | Interpretation |
|---|---:|---|
| SWE-bench Verified tasks | 20 | Distinct task trajectories across 11 repositories |
| Development / replication tasks | 10 / 10 | Predeclared cohorts; subgroup values are descriptive, not two independent tests |
| Qwen Code requests | 699 | Model completions inside the 20 agent loops |
| Selected checkpoints | 160 | Eight uniform request prefixes per task |
| Scored source layers | 24 | Fixed layers 24-47, with no best-layer selection |
| Prompt tokens replayed | 4,639,486 | 12,445 minimum, 26,449 median, 64,970 maximum |
| Available action labels | 154 | Inspect 101, edit 12, validate 26, finalize 15; six missing |

The 64,970-token maximum is below the frozen 65,535-token one-token replay
ceiling. One unserved Seaborn request above that ceiling was excluded before
uniform selection. Four replication trajectories reached the exact Qwen Code
50-turn ceiling after producing real patches. Their 50-request prefix chains
were recovered fail closed from raw proxy captures and usage records; all four
official outcomes were unresolved.

## Generation And Replay Are Different Runs

The production agent run used compiled vLLM, the pinned NVIDIA ModelOpt
NVFP4/FP8 checkpoint, and its one-token MTP draft block. Across both cohorts,
91,789 of 105,033 drafted speculative tokens were accepted, a weighted rate of
`0.873906`. The development and replication rates were `0.864763` and
`0.879037` respectively.

The J-lens replay used eager vLLM hooks and loaded only the 64-block main
language model. MTP was disabled. MTP proposes tokens to accelerate serving;
it is not another target-model source layer, and accepted speculative tokens
are verified by the target main model. The replay reconstructs the frozen chat
prefixes through the same pinned target-model weights, but it changes the
execution schedule and kernels. It is not a retrospective capture of
generation-time activations and does not reproduce the original stochastic
continuation.

At every checkpoint the replay generated one deterministic greedy token for
adapter checks. The primary action label still comes from the actual next Qwen
Code completion, and the terminal task label comes from the official scorer.
The supplemental next-token control explicitly targets the replay model's
greedy token and does not claim that token was the original agent continuation.

## What The Lens Means

The Anthropic J-lens estimates average causal transport from a source residual
to final-block residuals at the present and future positions. For source layer
`l`, prompt `p`, and output coordinate `i`, the local fit implements:

```text
J_l[p][i, :] = mean over source positions s=16..126 of
               d(sum over target positions t=16..126 h_63[t, i])
               / d h_l[s, :]

J_l = mean over fit prompts of J_l[p]
logits = lm_head(final_norm(h_l @ J_l.T))
```

This is a vocabulary readout. A token such as `validate` receiving a higher
score means the transported residual aligns more strongly with that token
under final norm and the LM head. It does not mean that the model literally
said an English sentence internally. The experiment does not recover hidden
prose, private chain of thought, or a complete causal explanation of the
agent's behavior.

The J-lens is also not optimized to reconstruct the current next-token
distribution. Anthropic reports poor middle-layer next-token KL as an expected
consequence of its future-summed objective. That is why greedy-token transport
is a supplemental fit-capacity diagnostic here, not the primary definition of
J-lens quality.

## Three Estimator Paths

| Path | What it is | What it is not |
|---|---|---|
| Public `neuronpedia/jacobian-lens` | 1,000-prompt FP16 matrix artifact for Qwen3.6-27B | Fit-time model precision and quantization were not published; it must not be called a verified BF16 or NVFP4 fit |
| Local NF4 | Anthropic future-summed VJP estimator on bitsandbytes NF4 weights, BF16 compute, ten frozen prompts | Not fitted on the deployed ModelOpt NVFP4 graph |
| Local native NVIDIA | Anthropic future-summed estimator with exact deployed NVFP4/FP8 forward values and explicit packed-W4, live-FP8, GDN, and attention input VJPs, ten prompts | The identity STE is not the literal derivative of FP4/FP8 rounding |

The Apple reference `WeZZard/jlens-qwen36` established that this Qwen
architecture could be instrumented and supplied a useful custom GDN backward.
It is not the estimator used here. WeZZard position-averages each layer's local
Jacobian and chain-multiplies those averages; a product of position averages is
not the position average of the complete future-summed suffix Jacobian. Its
readout also differs in treatment of final RMSNorm. The local NVIDIA fitter
implements the Anthropic future-summed VJP directly, while borrowing only
architecture-level lessons from the Apple port.

Artifact identities are frozen in
[`configs/swe_behavioral_readout_protocol.json`](../configs/swe_behavioral_readout_protocol.json):

| Readout | Fit prompts | Artifact SHA-256 |
|---|---:|---|
| Public J-lens | 1,000 | `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1` |
| Local NF4 J-lens | 10 | `54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f` |
| Native NVFP4/FP8-STE J-lens | 10 | `82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057` |

## Frozen Behavioral Protocol

The primary protocol was frozen before official outcome scoring and before the
three-lens N20 analysis. It uses:

1. Uniform, label-independent checkpoint selection: eight probeable request
   prefixes per task.
2. Fixed layers 24-47 and position `-1`, with arithmetic averaging and no
   layer selection.
3. Four next-action classes, each reduced by log-mean-exp over six declared
   single-token vocabulary forms: `inspect`, `edit`, `validate`, `finalize`.
4. Leave-one-repository-out fitting and calibration, so a held-out repository's
   labels never enter its readout fit.
5. Paired hierarchical repository-then-task bootstrap draws. Checkpoint rows
   are not resampled as independent observations.
6. A future-identifier control derived from the agent's generated patch,
   mutation completion, and terminal summary, with visibility rejection and a
   same-task hidden foil.
7. One latest uniform checkpoint per task for the binary official outcome.
   Missing, error, or empty scorer states are never imputed as failures.

The operational support gates are deliberately stronger than merely having 20
tasks. Action needs at least 80% joint numerical coverage and five tasks from
three repositories per class. Official outcome needs 16 jointly certified
tasks and at least eight tasks from four repositories in each class. The
future control needs ten tasks from six repositories.

## Numerical Certification

All three lens reports pair exactly on prompt text and IDs, token metadata,
runtime identity, residual-capture manifests, ordinary logit readouts, and
final reconstruction evidence. Each process completed all 160 checkpoints but
returned status `failed` because the strict adapter gate is conjunctive.

| Per-report check | Passing checkpoints |
|---|---:|
| Greedy top-1 reconstruction | 159/160 |
| Top-5 prefix reconstruction | 159/160 |
| Final norm tolerances | 157/160 |
| Full final-logit tolerances | 70/160 |
| Every strict check | 68/160 |

Ninety checkpoints had a full-logit maximum absolute error of `0.125` rather
than the frozen `0.0625` limit. The same failures occur for the public, NF4,
and native reports before lens-specific scores are compared, so they diagnose
the residual adapter, not one fitted lens. They still make the strict primary
coverage only 42.5%, below its 80% threshold.

## Primary Behavioral Result

The frozen decision is `insufficient_support`, reason
`predeclared_coverage_or_paired_inference_gate_failed`.

| Track | Strict support | Required support | Result |
|---|---|---|---|
| Next action | 66 labeled certified rows, 20 tasks, 11 repositories, 41.25% of all selected rows | 80% coverage and at least 5 tasks / 3 repositories per class | Failed coverage; edit has only 4 tasks |
| Official outcome | 8 certified latest-task rows: 4 success, 4 failure, each across 3 repositories | 16 tasks and at least 8 tasks / 4 repositories per class | Failed |
| Future identifier | 4 target rows from one xarray task and one repository | 10 task-averaged observations across 6 repositories | Failed |

The available action point estimates are not a refit signal. Public J-lens
balanced accuracy exceeds the majority readout by `+0.22396`, paired 95% CI
`[0.04944, 0.41840]`, and ordinary logit by `+0.13542`, CI
`[-0.01250, 0.32283]`. Public minus native is `-0.04167`, CI
`[-0.13214, 0.00000]`: the native lens does not show the preregistered public
advantage of at least `0.10` with a positive lower interval bound. These are
conditional on frozen cross-fit calibrators; the bootstrap does not refit them
inside each draw.

The future-identifier values cannot repair that conclusion. Only one task is
represented. Its public-minus-logit accuracy delta is `0.0`, while public minus
native is `+0.25`; neither is task-level generalization evidence. Official
outcome is similarly sparse after strict certification. The development and
replication subgroup metrics are descriptive only and are not independent
confirmatory experiments.

## Greedy Next-Token Transport Control

This supplemental control asks whether the fitted transport surfaces the
replay main model's own final greedy token earlier in layers 24-47. It uses
normalized rank utility
`log(248320 / rank) / log(248320)`, averages layers within checkpoint,
checkpoints within task, and then weights tasks equally. It never selects a
best layer.

The strict track is still `insufficient_support`: 68/160 checkpoints pass,
versus 128/160 required, and several tasks have fewer than six passing rows.
Its estimates are descriptive only.

After observing only public-report numerical reconstruction diagnostics, but
before inspecting any public, NF4, or native target rank or log probability, a
sensitivity amendment relaxed only the reconstruction bounds to the observed
BF16-scale envelope: full-logit maximum `<=0.125`, RMS `<=0.02`, plus the
existing top-1, top-5, and final-norm checks. The strict primary was unchanged.
The amended track retained 155/160 checkpoints, all 20 tasks, and all 11
repositories.

| Sensitivity method | Rank utility | Geometric mean target rank |
|---|---:|---:|
| Ordinary logit lens | 0.61141 | 124.9 |
| Public J-lens `n=1000` | 0.37692 | 2,298.9 |
| Native NVFP4/FP8-STE J-lens `n=10` | 0.30778 | 5,426.5 |
| NF4 J-lens `n=10` | 0.23769 | 12,962.7 |

Paired equal-task differences are:

| Comparison | Rank-utility delta | Paired 95% CI |
|---|---:|---:|
| Public J minus ordinary logit | -0.23449 | [-0.26190, -0.20507] |
| Public J minus native J | +0.06914 | [+0.06313, +0.07519] |
| Public J minus NF4 J | +0.13924 | [+0.13169, +0.14722] |

The sensitivity classification is
`sensitivity_readout_control_failure`: the public lens fails the required
positive control against ordinary logit. The public artifact's advantage over
both local artifacts therefore cannot, by itself, justify refitting. It is
consistent with a shared `n=10` sample-count, corpus, precision, or derivative
limitation, but the readout target is also misaligned with the J-lens objective.
This control cannot distinguish those explanations.

The timing disclosure and immutable rules are in
[`configs/swe_next_token_transport_protocol.json`](../configs/swe_next_token_transport_protocol.json).

## Learned Action Readout

The frozen band-average readout can discard action-specific layer structure.
The supplemental protocol was frozen after the public and NF4 report files
existed. Action labels and their exact class counts had already been inspected;
the protocol explicitly discloses that public, NF4, and native action scores,
predictions, and metrics had not been inspected. It trains one multinomial
logistic classifier per method on the existing raw class scores. Its 96
features are four class scores at each of the fixed 24 layers. Scaling, class
weights, and regularization are fitted
inside nested leave-one-repository-out cross-validation; no held-out
repository contributes labels to scaling, fitting, or `C` selection. There is
still no best-layer or feature selection.

The strict track has the same 66 rows as the primary action analysis. It
contains every action class, all 20 tasks, and all 11 repositories, and all
nested folds converge with complete predictions. It nevertheless misses the
frozen 128-row minimum, so its classification is
`insufficient_strict_support`. The values below are conditional descriptive
results, not an actionable readout claim.

| Strict feature source | Learned balanced accuracy | Micro accuracy | Edit recall |
|---|---:|---:|---:|
| Ordinary logit lens | 0.80729 | 0.86364 | 0.75 |
| NF4 J-lens | 0.78125 | 0.78788 | 0.75 |
| Native NVFP4/FP8-STE J-lens | 0.75000 | 0.69697 | 0.75 |
| Public J-lens | 0.68229 | 0.74242 | 0.50 |
| Frozen public band-average readout | 0.47396 | 0.75758 | 0.00 |
| Post-hoc checkpoint-ordinal prior | 0.63542 | 0.83333 | 0.00 |

Using a learned multivariate readout of the 96 layer-by-class features raises
public J-lens balanced accuracy over its frozen band-average readout by
`+0.20833`, but its paired 95% CI
`[-0.04666, 0.46348]` includes zero. More importantly, learned public J-lens
trails learned ordinary logit by `-0.12500`, CI
`[-0.33786, -0.00962]`. The learned J-lens readouts do recover examples from
all four classes, including edit, so the token features contain usable phase
information. They do not show a J-lens-specific advantage.

The action sensitivity track is also non-actionable. It retains 149 labeled
rows, every class, every task, and every repository, but
`pytest-dev__pytest-10356` contributes five rows rather than the required six.
Its balanced accuracies are NF4 `0.70992`, ordinary logit `0.66992`, native
`0.64976`, and public `0.63218`; the frozen public readout is `0.42334`.
Learned public minus frozen public is `+0.20884`, CI
`[0.03170, 0.39887]`, while learned public minus learned ordinary logit is
`-0.03774`, CI `[-0.12248, 0.04192]`. This post-public-numerical-diagnostic
track cannot override strict support and still provides no evidence for
refitting the native lens.

The ordinal prior is explicitly post hoc and descriptive. For each held-out
repository it predicts from the empirical action distribution at the same
uniform checkpoint ordinal in the outer training repositories. It uses no
held-out labels, but it exploits a strong phase confound: early uniform
checkpoints are mostly inspect and the last checkpoint is mostly finalize.
Its balanced accuracy is `0.63542` strict and `0.59232` sensitivity. Any useful
learned lens readout must be compared with this progress-only baseline, not
only with an inspect-majority classifier.

This supplement refines the conclusion: descriptive sensitivity evidence
suggests a multivariate readout of layer-resolved features may improve on band
averaging, but strict support is insufficient. The design does not isolate
layer weighting from cross-class mixing or fitted intercepts. Refitting the
J-lens is not the bottleneck demonstrated by these data. Ordinary logit
features are as good or better under the same nested repository-held-out model.
The next readout should model action vocabulary and task progress jointly, with
strictly adequate rows per task and an external semantic target that ordinary
logit does not already solve. The immutable protocol is
[`configs/swe_action_layer_readout_protocol.json`](../configs/swe_action_layer_readout_protocol.json).

## External Positive Control

The lens implementation is not globally broken. On Anthropic's pinned
93-item multihop intermediate set, evaluated on the same fixed layers 24-47,
both the public and native lenses beat ordinary logit lens:

These values are not numerically comparable to the N20 mean-over-layer
estimand. For each intermediate occurrence, the control takes the minimum rank
across the fixed layer band and eligible bare or leading-space token forms,
averages occurrences within each item while counting unscorable forms as
misses, then macro-averages the 93 items.

| Method | Normalized log-rank AUC | Gain over logit | Paired 95% CI | Pass-at-10 | Gain over logit | Paired 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary logit | 0.47073 | - | - | 0.13441 | - | - |
| Public J-lens | 0.62374 | +0.15300 | [0.11490, 0.19011] | 0.29032 | +0.15591 | [0.06452, 0.24731] |
| Native NVFP4/FP8-STE | 0.61902 | +0.14829 | [0.10986, 0.18626] | 0.27957 | +0.14516 | [0.05376, 0.23656] |

That is the method-aligned reproduction outcome: a known future-relevant
single-token concept can be surfaced earlier by J-lens than by ordinary logit
lens. It does not imply the same fixed vocabulary groups will classify an
agent's next tool action or reveal its chain of thought. The N20 study tests
that separate adaptation and currently lacks a valid SWE-specific positive
control.

The multihop comparison is paired descriptive evidence, not a strict adapter
certificate. Public and native reports each passed the full final-logit
tolerance on 72/93 prompts and retain report status `failed`; prompt IDs,
residual manifests, and ordinary logit fields nevertheless match between
methods. Imperfect final-output parity limits causal interpretation but does
not erase the fixed, paired rank-utility result.

## Probe Or Refit?

The next step is **probe and capture refinement, not another lens fit**.

1. Fix the adapter discrepancy so the existing strict `0.0625` maximum-logit
   gate covers at least 80% of checkpoints, or preregister a justified numeric
   contract before another outcome inspection. Do not promote the existing
   post-public sensitivity into a new primary result retroactively.
2. Predeclare a lens-independent task-selection rule that yields hidden future
   identifiers before they appear in the transcript. The current derivation
   found 44 dynamic targets, but visibility and foil constraints left only
   four checkpoint rows from one task.
3. Increase edit and finalize support, and retain at least six numerically
   usable checkpoints per task. Uniform sampling avoided label leakage but did
   not balance rare action transitions.
4. Prefer task-held-out learned readouts over hand-averaged token groups when
   the learned analysis shows out-of-repository signal. Always compare them
   with non-lens baselines, including an ordinal progress prior.
5. Refit only after a public-lens SWE positive control passes and the public
   artifact then beats the native artifact by the frozen material threshold.
   If both local `n=10` lenses trail public in a valid control, investigate fit
   sample count and corpus before assigning the gap specifically to NVFP4 STE.

The public artifact's 1,000 versus 10 fit prompts is currently confounded with
fit corpus, fit precision, quantization, and derivative contract. A larger
native fit may improve transport fidelity, but this experiment does not show
that it would make the SWE semantic probe useful.

## Official Task Outcome

The digest-pinned official SWE-bench 4.1.0 runs completed with zero error or
empty predictions:

| Cohort | Resolved | Unresolved | Total |
|---|---:|---:|---:|
| Development | 5 | 5 | 10 |
| Replication | 4 | 6 | 10 |
| Combined | 9 | 11 | 20 |

This 45% task resolution rate describes the generated patches. It is not a
J-lens metric. The four exact turn-limit tasks were all unresolved, but turn
limit is kept as operational metadata rather than substituted for the official
verdict.

## Reproduction

The combined three-lens replay is automated by:

```bash
OUT_DIR="$PWD/validation/jlens-swe-behavioral-n20-2026-07-18" \
  scripts/run_swe_behavioral_pilot.sh
```

The wrapper materializes and hash-binds the prompt cohort, validates all three
lens artifacts, runs public then NF4 then native replay in separate model
loads, waits for GPU memory to settle between loads, and runs the frozen
behavioral analyzer. The supplemental analyzers consume the same immutable
prompt bundle and reports:

```bash
.venv-vllm/bin/python scripts/analyze_swe_next_token_transport.py --help
.venv-vllm/bin/python scripts/analyze_swe_action_layer_readout.py --help
```

Primary compact evidence is under
[`validation/jlens-swe-behavioral-n20-2026-07-18/publication/`](../validation/jlens-swe-behavioral-n20-2026-07-18/publication/).
The publication manifest binds:

- the 4,042,839-byte frozen primary analysis;
- the exact compact [next-token transport analysis](../validation/jlens-swe-behavioral-n20-2026-07-18/publication/next-token-transport-analysis.json)
  and [learned action-layer analysis](../validation/jlens-swe-behavioral-n20-2026-07-18/publication/action-layer-readout.json);
- the 160-row prompt summary and both campaign evidence records;
- the two official outcome files;
- the exact full prompt bundle and all three full lens reports by SHA-256.

The 86,989,478-byte prompt bundle and three approximately 274 MB lens reports
are intentionally not committed. Each exceeds the compact-publication policy,
and each lens report exceeds GitHub's single-file limit. Their hashes prove
identity and lineage, not public content availability. Independent numerical
reanalysis requires those omitted local raw artifacts. Raw Qwen Code proxy
captures, vLLM logs, SWE-bench harness logs, model checkpoints, and 6.15 GiB
or larger lens tensors are also omitted from Git.

Run the publication verifier for the exact compact-bundle boundary:

```bash
.venv-vllm/bin/python scripts/check_swe_behavioral_n20_publication.py
```

## Limitations

- Twenty tasks are enough for a pipeline study, but strict reconstruction left
  only 42.5% joint checkpoint coverage and unbalanced rare action classes.
- Eight checkpoints from one task are correlated trajectory samples, not eight
  independent tasks. Bootstrap inference resamples repositories and tasks,
  never checkpoint rows.
- The learned classifiers and frozen calibrators are not refit within
  bootstrap draws, so their intervals are conditional on the realized
  out-of-fold predictions and exclude model-fit uncertainty.
- Generated patches define future identifiers without benchmark-gold leakage,
  but most identifiers are already visible in long coding transcripts or lack
  an eligible same-task single-token foil.
- Eager replay is the correct activation-capture path for the accepted main
  model, but it is not a capture of the compiled MTP generation process.
- The public lens fit precision is unknown. The native lens uses an exact
  deployed quantized forward with a declared identity-STE surrogate backward.
  Neither should be described as a literal quantization derivative.
- Vocabulary readouts can reveal token-level features. They cannot establish
  literal hidden English, private chain of thought, or causal necessity without
  separate interventions.

## References

- [Anthropic J-lens method](https://transformer-circuits.pub/2026/workspace/index.html#methods-jlens)
- [Anthropic objective comparison](https://transformer-circuits.pub/2026/workspace/index.html#methods-compare)
- [Anthropic pinned implementation](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e)
- [WeZZard Qwen3.6 Apple reference](https://github.com/WeZZard/jlens-qwen36/tree/d788bc321dc7bad4ed33e1465b306389959b0046)
- [Native NVIDIA fit report](JLENS_NVFP4_STE_EXPERIMENT.md)
- [Preceding N=1 multistage report](JLENS_SWE_MULTISTAGE_2026-07-18.md)
