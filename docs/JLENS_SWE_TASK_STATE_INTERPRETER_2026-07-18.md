# Qwen3.6 NVFP4 J-Lens Dense SWE Task-State Interpreter

**Date:** 2026-07-18
**Execution status:** complete development screen
**Frozen decision:** `development_gates_passed: false`
**Operational reliability claim:** false

## Bottom Line

The lens contains a real but limited signal about the next Qwen Code action.
It is not a reliable task-state interpreter, and it does not reveal hidden
English chain of thought.

This run evaluated every probeable model request across the existing 20-task
SWE-bench Verified trajectories, rather than the earlier eight checkpoints per
task. The production trajectories contained 699 Qwen Code completions. There
were 698 replayable request prefixes; 639 passed both the action-label and
frozen numerical-stability gates. They cover 20 tasks and 11 repositories.

On those 639 repository-held-out predictions:

- the public J-lens alone reached balanced accuracy `0.6680`;
- the ordinary logit lens alone reached `0.6717`;
- J plus causal transcript context reached `0.6752`;
- ordinary logits plus the same context reached `0.6530`;
- the full additive hybrid, J plus ordinary logits plus context, reached
  `0.6494`.

The clean incremental test is the hybrid minus the otherwise identical
ordinary-logit context model. Its balanced-accuracy difference is `-0.0036`,
with conditional hierarchical-bootstrap 95% interval
`[-0.0532, +0.0471]`. No accuracy, calibration, NLL, or Brier comparison
establishes positive J-lens value beyond ordinary logits.

The hybrid's task-equal selective accuracy is `0.6798` at `0.8275` task-equal
coverage. The frozen reliability requirements were `0.90` point accuracy,
`0.80` interval lower bound, and at least `0.50` coverage. The observed
accuracy lower bound is `0.6068`. Only 4 of 11 outer folds found a calibration
threshold meeting the train-only selection floors. This is not close enough
to justify touching the reserved validation cohort.

The practical result is therefore:

1. The middle-layer vocabulary readout carries action/phase information.
2. That information is not uniquely Jacobian-lens information; ordinary
   residual logits carry at least as much.
3. The current target, exact next agent action, is too noisy for a dependable
   interpreter at useful coverage.
4. Another expensive NVFP4 lens fit is not the next experiment. The next
   experiment should refine the behavioral target and sequence readout first,
   then collect fresh tasks only after the development gate is met.

## What Was Predicted

A row ends at the final token of one exact Qwen Code request prefix. The label
is the observable action in the next assistant completion:

- `inspect`: repository search, file read, or other investigation;
- `edit`: a source mutation;
- `validate`: tests or another declared validation command;
- `finalize`: the terminal answer.

This is one supervised prediction per agent-loop completion, not one label per
SWE task and not one token sampled from a hidden narrative. A completion can
contain one or more tool calls. Their results enter the following request.

The 639 eligible labels are:

| Action | Rows |
|---|---:|
| Inspect | 465 |
| Edit | 44 |
| Validate | 115 |
| Finalize | 15 |

Class-balanced fitting makes balanced accuracy the useful headline. A trivial
always-inspect predictor has raw accuracy `0.7277` but balanced accuracy
`0.25`; raw accuracy alone would therefore reward ignoring the three rarer
states.

## Whole-Trajectory Coverage

The dense bundle uses all probeable requests in all 20 recorded tasks:

| Unit | Count |
|---|---:|
| Production Qwen Code completions | 699 |
| Dense replay prefixes | 698 |
| Stable labeled analysis rows | 639 |
| Tasks | 20 |
| Repositories | 11 |
| Numerically unstable exclusions | 47 |
| Missing-action exclusions | 12 |

The earlier N20 report sampled eight prefixes per task. This experiment
instead follows each task from its first request through its last probeable
request. Missing intermediate action labels remain explicit unknowns in causal
history; they are never imputed.

## Generation And Replay

The agent trajectories were generated with the production stack:

- `nvidia/Qwen3.6-27B-NVFP4` at revision
  `0893e1606ff3d5f97a441f405d5fc541a6bdf404`;
- compiled vLLM serving on the RTX 5090;
- one-token MTP speculative decoding;
- Qwen Code in official SWE-bench Verified containers.

The lens replay is a separate deterministic main-model pass. It uses eager
vLLM hooks, layers 24 through 47, the final prompt position, FP8 KV cache, and
native CPU KV offload. MTP is disabled during replay because it is a serving
draft mechanism, not a source layer in the 64-block target model. The target
weights are still the pinned NVIDIA ModelOpt NVFP4/FP8 checkpoint.
The analyzer independently requires the resolved capture and every layer
readout to name token position `len(prompt_token_ids) - 1`; a replay captured
at `-2` or any other position is rejected before feature extraction.

The 698-prefix replay completed in `906.378108` seconds, loaded the model in
`24.202871` seconds, and reserved at most `26,531,069,952` CUDA bytes. The
runner's top-level status is `failed` because 47 rows miss the strict adapter
reconstruction profile; the artifact, model, and lens identity assertions
pass, and the analyzer excludes those rows fail closed.

## Lens Identity

The experiment applies the public `neuronpedia/jacobian-lens` Qwen3.6-27B
artifact:

| Field | Value |
|---|---|
| Revision | `a4114d7752d11eb546e6cf372213d7e75526d3a1` |
| SHA-256 | `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1` |
| Fit prompts | 1,000 |
| Stored tensor dtype | FP16 |
| Stored shape | `[5120, 5120]` per source layer |

The stored matrix dtype does not establish the precision used to fit it. Its
fit-time precision and quantization are unpublished, so this report does not
call it a BF16 lens or an NVFP4 lens. It is a public FP16 artifact applied to
NVFP4/FP8 residuals.

## Readout Variants

All learned variants use nested leave-one-repository-out multinomial logistic
regression. Feature scaling, class weights, regularization selection,
temperature selection, and abstention threshold selection use training data
only. The held-out repository is not used for any of those choices.

| Variant | Feature blocks | Width |
|---|---|---:|
| `progress_only` | request ordinal | 2 |
| `lexical_progress` | exact vocabulary frequency/recency plus progress | 18 |
| `history_context` | prior actions, lexical controls, progress | 32 |
| `ordinary_logit` | 24 layers x 4 action scores | 96 |
| `logit_context` | ordinary logits plus identical context | 128 |
| `public_jacobian` | 24 layers x 4 transported action scores | 96 |
| `jacobian_context` | public J scores plus identical context | 128 |
| `hybrid` | public J, ordinary logits, and identical context | 224 |

The action score at each layer is log-mean-exp over six frozen vocabulary
forms. There is no best-layer search. The readout uses the complete fixed band
24 through 47.

## Full-Cohort Results

`Task selective accuracy` and `task coverage` give each SWE task equal weight.
The selected threshold can differ by held-out repository because it is chosen
inside that fold from inner training predictions.

| Variant | Accuracy | Balanced accuracy | NLL | Brier | Task BA | Task selective accuracy | Task coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Progress only | `0.4695` | `0.4046` | `1.1925` | `0.6120` | `0.3735` | `0.4986` | `1.0000` |
| Lexical + progress | `0.4851` | `0.3806` | `1.2502` | `0.6539` | `0.3865` | `0.4920` | `1.0000` |
| History context | `0.5446` | `0.5279` | `1.1236` | `0.5998` | `0.4988` | `0.5578` | `1.0000` |
| Ordinary logit | `0.6025` | `0.6717` | `0.8800` | `0.5061` | `0.7317` | `0.6182` | `0.9989` |
| Public J | `0.5978` | `0.6680` | `0.9380` | `0.5243` | `0.7342` | `0.6139` | `0.9967` |
| Logit context | `0.6166` | `0.6530` | `0.8851` | `0.4937` | `0.6834` | `0.6312` | `1.0000` |
| J context | `0.6275` | `0.6752` | `0.8951` | `0.4954` | `0.6976` | `0.6599` | `0.9433` |
| Full hybrid | `0.5994` | `0.6494` | `0.8620` | `0.4869` | `0.7016` | `0.6798` | `0.8275` |

Public J alone and ordinary logits alone are effectively tied on balanced
accuracy. Public J minus ordinary is `-0.0037`, 95% interval
`[-0.0541, +0.0426]`. Public J has worse NLL by `+0.0581`, interval
`[+0.0029, +0.1195]`.

Replacing ordinary logits with J in the contextual model gives a small point
gain: J context minus logit context is `+0.0222` balanced accuracy, interval
`[-0.0225, +0.0615]`. The interval includes no effect. Its task selective
accuracy gain is `+0.0287`, interval `[-0.0032, +0.0603]`, while task coverage
drops by `-0.0567`, interval `[-0.1337, 0.0000]`.

## Clean J Ablation

The additive hybrid and logit-context variants differ only by the 96 public-J
features. This is the primary J-specific comparison.

| Hybrid minus logit context | Estimate | Conditional 95% interval |
|---|---:|---:|
| Balanced accuracy | `-0.0036` | `[-0.0532, +0.0471]` |
| Task-macro balanced accuracy | `+0.0182` | `[-0.0280, +0.0804]` |
| Task selective accuracy | `+0.0486` | `[-0.0276, +0.1274]` |
| Task coverage | `-0.1725` | `[-0.3195, -0.0425]` |
| Balanced accepted recall | `-0.0916` | `[-0.2090, -0.0017]` |
| NLL, lower is better | `-0.0231` | `[-0.0608, +0.0290]` |
| Brier, lower is better | `-0.0068` | `[-0.0336, +0.0255]` |

The selective-accuracy point increase is therefore not a dependable gain. It
comes with significantly less coverage and lower balanced accepted recall.

Hybrid does beat history-only balanced accuracy by `+0.1214`, interval
`[+0.0319, +0.2290]`. Ordinary-logit context already reaches `0.6530`, however,
which is essentially the hybrid's `0.6494`. Activations add information beyond
the behavioral history; the experiment does not attribute that information to
the Jacobian transport.

## Per-Class Meaning

The raw J-only recalls are:

| True action | Recall |
|---|---:|
| Inspect | `0.5849` |
| Edit | `0.7045` |
| Validate | `0.5826` |
| Finalize | `0.8000` |

J context recalls `0.6258 / 0.5909 / 0.6174 / 0.8667` for
inspect/edit/validate/finalize. Logit context recalls
`0.6258 / 0.5455 / 0.5739 / 0.8667`. The J path recovers two additional edits
and five additional validations, but those small counts do not produce an
interval-supported task-level advantage.

Only 15 finalize rows exist. A high finalize recall is not evidence of robust
rare-state performance, especially because the same terminal pattern repeats
within a small number of tasks.

## Reliability Decision

Support and outer-fold completion pass. The following substantive gates fail:

- hybrid balanced accuracy is `0.6494`, below `0.70`;
- task selective accuracy is `0.6798`, below `0.90`;
- its interval lower bound is `0.6068`, below `0.80`;
- ECE interval upper bound is `0.1080`, above `0.10`;
- only 4 of 11 calibration folds meet the train-only selection floors;
- the required positive J effects on balanced and selective accuracy fail;
- the required J improvements in NLL and Brier fail.

Coverage itself is adequate: task-equal point coverage `0.8275`, lower bound
`0.6805`, and minimum per-class row coverage `0.7955`. The failure is accuracy
and incremental J value, not an interpreter that merely abstains too often.

The 5,000-draw intervals resample repositories and then tasks. They are
conditional on the frozen out-of-repository predictions. They do not refit
models or repeat regularization, temperature, and threshold selection inside
each draw. They are suitable for this development screen and explicitly do
not prove operational reliability.

## Solver Correction

The first analysis attempt omitted the Pylint and scikit-learn outer folds in
the 224-feature hybrid. One shared inner fit stalled after 4,000 iterations at
gradient infinity norm `1.23777e-5`, just above the frozen `1e-5` tolerance.
Increasing the iteration count did not move the solution.

The root cause was stale L-BFGS inverse-curvature history after rejecting a
new curvature pair below the existing numerical floor. Clearing that history
on rejection converged the exact matrix in 134 iterations, with gradient
`9.68515e-6`. The correction does not change the labels, features, splits,
regularization grid, tolerance, or success gates. A deterministic regression
test covers the rejected-curvature behavior. The entire analysis and all
bootstrap draws were then rerun; all 11 outer folds are now complete.
The generalized solver is isolated in
[`scripts/swe_task_state_readout.py`](../scripts/swe_task_state_readout.py), so
the earlier published N20 action analyzer and its byte-level provenance remain
unchanged.

## Next-Milestone Exploration

The exact next action is partly a tool-granularity target. A development-only,
post-hoc exploration therefore asks a more stable question: after skipping
intervening `inspect` requests, what is the next observed consequential
milestone: `edit`, `validate`, or `finalize`?

The assignment is fail closed. An unknown action before the next milestone or
no observed future milestone censors the row. Of the numerically stable source
rows, 620 receive milestone labels:

| Milestone | Expanded rows | Distinct target events |
|---|---:|---:|
| Edit | 331 | 46 |
| Validate | 248 | 116 |
| Finalize | 41 | 15 |

There are 177 distinct events behind the 620 interval-expanded rows. The
median horizon is two requests, the 90th percentile is 16, and 446 labels
refer to a genuinely future request rather than the immediate action.

This target is more promising but still not reliable:

- J context balanced accuracy: `0.6602`;
- logit context balanced accuracy: `0.6341`;
- J-context minus logit-context: `+0.0261`, interval
  `[-0.0442, +0.0780]`;
- task-macro J-context minus logit-context: `-0.0126`, interval
  `[-0.0842, +0.0456]`;
- selected J-context task accuracy/coverage: `0.6414 / 0.8968`;
- selected logit-context task accuracy/coverage: `0.6570 / 0.9744`.

No frozen fixed threshold achieves at least `0.80` task-equal accuracy at
`0.50` coverage. The best relevant J point is `0.7254` accuracy at `0.5582`
coverage. This exploration identifies a better target for further work, but it
does not authorize fresh validation.

The reusable development-only explorer is
[`scripts/explore_swe_milestone_state.py`](../scripts/explore_swe_milestone_state.py).
Its complete compact output is
[`validation/jlens-swe-milestone-state-exploration-n20-2026-07-18.json`](../validation/jlens-swe-milestone-state-exploration-n20-2026-07-18.json).

## What This Does Not Mean

The J-lens maps transported residuals into vocabulary scores. A classifier can
learn that score patterns correlate with actions. Neither operation decodes a
sentence that the model secretly wrote.

The output supports statements such as:

> At this boundary, the frozen J feature bank is more aligned with the learned
> validation-action pattern than with the learned edit pattern.

It does not support:

> The model is thinking, "I have finished the fix and should run tests now."

Observed transcript history, action labels, and task descriptions remain
separate from lens evidence. No private chain of thought is reconstructed or
claimed.

## Next Decision

Do not refit the 27B NVFP4 lens and do not collect more identical immediate
action rows yet. The current public, NF4, and native artifacts already agree
well enough on prior controls that lens quantization is not the diagnosed
bottleneck.

The next justified ladder is:

1. Freeze a milestone/event-level protocol that gives each distinct transition
   appropriate weight instead of treating 620 overlapping prefixes as 620
   independent events.
2. Use a sequence model or transition-aware calibrator trained only within
   repository folds, while retaining ordinary-logit, history, lexical, and
   progress controls.
3. Require at least `0.80` task-equal selective accuracy at `0.50` coverage on
   development data and a positive J-specific interval before validation.
4. Only then run the already reserved 20-task cohort under a separate locked
   eight-repository validation contract.

The reserved cohort has been selected and hash checked, but no generation,
lens replay, or score inspection was performed after this failed development
screen.

## Reproduction

The fail-closed wrapper materializes the exact 698-prefix bundle, verifies its
SHA-256, confirms the production serving endpoint is offline, runs the public
J replay with the pinned arguments, and analyzes it:

```bash
scripts/run_swe_task_state_interpreter.sh
```

A clean `--prepare-only` verification through the isolated dense adapter
regenerated the 381 MB prompt file with SHA
`c89a8dc95455633d1e25c62c3393c063ead3184e84d33a9a00b55ab390b76c27`
and the byte-identical summary with SHA
`a031cf07f7cb21a93a0391cd3f68f2ebc1f03c7d9a0fdd69024095a67a7eee6f`.

For an already completed wrapper run with valid sidecars:

```bash
scripts/run_swe_task_state_interpreter.sh --analyze-only
```

Reproduce the post-hoc milestone screen from the same dense artifacts:

```bash
.venv-vllm/bin/python scripts/explore_swe_milestone_state.py \
  --prompts .cache/swe_task_state_interpreter/prompts.json \
  --report .cache/swe_task_state_interpreter/public-report.json \
  --protocol configs/swe_task_state_interpreter_protocol.json \
  --behavioral-protocol configs/swe_behavioral_readout_protocol.json \
  --variants jacobian_context logit_context \
  --output .cache/swe_task_state_interpreter/milestone-exploration.json
```

The reserved validation selection can be reproduced without generating tasks:

```bash
scripts/run_swe_task_state_validation.sh --check-only
```

The check binds the exact 68-candidate and eligible-instance sets, not only
their counts, plus the 20 selected task IDs, image digests, campaign bytes,
action protocol, and chat template. Full mode stops after generating and
strictly validating the combined prompt bundle; it does not replay a lens or
score the held-out cohort. Because the development gate failed, full mode was
not invoked.

The validation runner intentionally uses one shared target-model endpoint for
its sequential A/B campaigns, preventing two 27B checkpoints from becoming
resident on the 32 GB GPU. Distinct run roots and proxy ports preserve campaign
separation.

## Evidence And Hashes

| Artifact | SHA-256 |
|---|---|
| Dense prompt bundle | `c89a8dc95455633d1e25c62c3393c063ead3184e84d33a9a00b55ab390b76c27` |
| Public replay report | `e4e849afe234ad38da85278df24e8606f2daa5a1c7da0772b5dc60786a75444a` |
| Frozen protocol | `a6441137828866e8aad9dc547fc0fee37706ece390f503a81fdd9e0f53ed409a` |
| Dense materializer adapter | `a1c9b75b885fd9e50f59363c09d5aa47566eb7bbf7a91fe5aad6a8dccfd1cc5e` |
| Analyzer implementation | `279d0a41742e9feeabd3dbd82b73f609326a4942c983cc2eee7125164ebd0594` |
| Task-state readout solver | `e8015b51a68ed2001f3351e38d6beaccbcf7dbe507eeeb2e2b608c4f276e2e77` |
| Final analysis | `ba70254e023173785f5d3e90f11b09c5c300b3e4a16ba2219f6c6b763fba7204` |
| Milestone explorer | `54c62b1dd105274cdb60174240e90c231cbc0183a9f4bebf18bb1493afecc358` |
| Milestone exploration | `aa36add5aaaa1e53c2a0f4180d58f17b1e1551adfa5cddcac2040468b3c6d9fe` |
| Reserved validation cohort | `f0c2adfb562494362f359a60aa37a63f289af9f1b8b833805c07e2173aea6cbd` |
| Reserved cohort checker | `9cb4a509ab1faa29d2ff0e64336e3fb283f803812037eaf6644f0d8ee25e9d6b` |
| Reserved staging runner | `71f8bf263945a7f616c29b6eb1eaa134d799b58ac348925a78428bfcdad2198f` |

The complete 23.2 MB analysis, including every prediction, outer-fold model
record, calibration sweep, bootstrap interval, and gate, is committed as
[`validation/jlens-swe-task-state-interpreter-n20-2026-07-18.json`](../validation/jlens-swe-task-state-interpreter-n20-2026-07-18.json).
The 1.2 GB raw replay report is not committed; its exact digest, model and lens
pins, prompt payloads, numerical evidence, and reproduction command are bound
by the analysis and wrapper.

The exact publication configs, implementations, tests, report, and compact
artifacts are bound by
[`validation/jlens-swe-task-state-interpreter-source-manifest-2026-07-18.sha256`](../validation/jlens-swe-task-state-interpreter-source-manifest-2026-07-18.sha256).
Verify it from the repository root with:

```bash
sha256sum --check \
  validation/jlens-swe-task-state-interpreter-source-manifest-2026-07-18.sha256
```
