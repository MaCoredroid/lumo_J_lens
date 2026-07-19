# Qwen3.6 NVFP4 J-Lens Binary SWE Phase Interpreter

**Date:** 2026-07-18
**Target:** next consequential Qwen Code phase at each stable request boundary
**Development status:** `15/18` frozen gates passed; J-specific screen failed
**Confirmatory status:** reserved validation intentionally not started

## What The Lens Actually Revealed

The useful result is a behavioral phase forecast, not decoded hidden prose.
At the final prompt token of a Qwen Code request, the layer-resolved public
Jacobian-lens scores contain information about whether the next consequential
agent action will be:

- `edit`: mutate source code; or
- `check_or_finish`: validate the work or return the final response.

Known inspection actions are skipped when assigning the offline target. The
forecast can therefore refer to the current completion or to a later
completion after one or more inspection steps. An unknown action before the
next milestone censors the ground truth. It never suppresses an online
prediction.

The corrected operational development evaluator emitted at 650 stable
boundaries and retained 620 label-ascertained
boundaries representing 177 physical phase events across 20 SWE-bench
Verified tasks and 11 repositories. Its repository-held-out results were:

| Readout | Accuracy | Balanced accuracy | NLL | Brier |
|---|---:|---:|---:|---:|
| Public J compact | `0.8451` | `0.8178` | `0.4262` | `0.2493` |
| Ordinary logit compact | `0.8235` | `0.7950` | `0.4441` | `0.2669` |
| J minus logit | `+0.0216` | `+0.0227` | `-0.0179` | `-0.0176` |

The exact predictions, horizon strata, intervals, and gate records are in the
[development report](../validation/swe-binary-phase-v2-development.json). The
[model manifest](../artifacts/swe-binary-phase-v2.manifest.json) binds the two
serialized estimators, training identities/features, environment, sources,
and canary.

The 10,000-draw paired repository-then-task bootstrap interval for the
accuracy difference was `[-0.0127, +0.0552]`. The NLL-difference interval was
`[-0.0392, +0.0049]`, and the Brier-difference interval was
`[-0.0365, +0.0023]`. All three cross zero. The frozen J-specific accuracy,
NLL, and Brier interval gates therefore fail. These are development
sensitivity values because the architecture and binary target were chosen
after inspecting this same cohort.

An earlier exploratory implementation reported a larger J advantage because
it excluded current-action-unavailable stable boundaries from the sequence
state. That is not how an online interpreter operates. The corrected analyzer
emits and updates its causal state at every stable boundary, regardless of
whether offline truth will later be available. This correction reduced the J
advantage and is the result frozen here.

Forecast distance matters. J balanced accuracy was `0.862` at horizon zero,
`0.789` at horizons one to two, `0.719` at horizons three to five, `0.666` at
horizons six to ten, and `0.600` at horizons eleven or more.
The aggregate result must not be described as uniform reliability throughout
a task.

## What It Does Not Reveal

The readout does not recover a sentence such as "the model thinks the parser
is wrong." It does not identify the bug, intended patch, file, function, or
test. It does not expose chain-of-thought. It predicts one coarse observable
work-cycle outcome from a supervised activation pattern.

The public lens is also not a native NVFP4 fit. The tracked matrices are FP16,
their fit-time precision is unpublished, and this experiment applies them to
residuals from the pinned NVIDIA NVFP4/FP8 checkpoint. The repository's local
ten-prompt native lens is not used for this readout because it is far too small
to replace the public 1,000-prompt artifact.

## Causal Boundary And Denominators

Each input is the exact transcript prefix immediately before one Qwen Code
completion. Features use only that prefix, its final-token residuals, and
causal summaries of earlier stable prefixes in the same task. Future actions
are used only after the trajectory exists to assign evaluation truth.

Every report keeps four denominators separate:

1. all replayed prompts;
2. numerically stable, feature-complete emitted predictions;
3. emitted predictions whose future phase is ascertainable; and
4. accepted predictions.

The frozen confidence threshold is zero, so every stable emitted prediction
is accepted. Label ascertainment is not inference coverage. Accuracy is
identified only on the ascertainable subset, and the report states that
missing-outcome limitation.

## Feature Construction

Both branches use the same 24 source layers (`24..47`), final prompt position,
four action concepts (`inspect`, `edit`, `validate`, `finalize`), causal
context, folds, weights, model family, random seeds, temperature, and
threshold. Only the 96 layer-by-action score block changes:

- `j_compact` uses public Jacobian-transported vocabulary scores;
- `l_compact` uses ordinary residual-to-vocabulary logit scores.

For each current 24 by 4 score matrix, delta from the previous stable matrix,
and deviation from the prior exponential moving average (`alpha=0.5`), the
analyzer computes ten summaries per action: mean, standard deviation, minimum,
maximum, three fixed layer-band means, layer slope, last-minus-first, and
normalized argmax layer. That produces `3 * 40 = 120` features. It appends the
same 32-dimensional causal history/lexical/progress context, log request gap,
and no-previous flag for a total width of 154.

Sequence state updates at every stable emission, including emissions whose
future truth is later censored. No future label, milestone, horizon, physical
event ID, task outcome, or gold patch enters the features.

## Fitted Readout

The frozen classifier is `ExtraTreesClassifier` with 100 trees,
`min_samples_leaf=5`, `max_features=0.5`, and random seed `271828`. All other
parameters are explicit in
[`configs/swe_binary_phase_interpreter_v2.json`](../configs/swe_binary_phase_interpreter_v2.json).
There is no scaler. Temperature is `1.0`, confidence threshold is `0.0`, and
there is no abstention.

Training first gives equal base mass to each task, each physical event within
a task, and each prefix within an event. It then balances the two aggregate
class masses. That final class rebalance means the training weights are no
longer task-equal; the analyzer records pre/post task and class mass ranges and
the exact float64 weight hash. Evaluation retains task-equal,
event-equal-within-task, prefix-equal-within-event weights. Balanced accuracy
is reported separately as macro class recall.

`min_samples_leaf` still counts raw correlated prefix rows, not weighted
physical events. This is a known limitation, not hidden by the event weights.

## Development And Confirmation

Development uses leave-one-repository-out predictions. The direct J-versus-L
comparison is a matched replacement ablation; it does not estimate the
conditional contribution of adding J on top of ordinary logits. Bootstrap
draws resample repositories and then tasks, retain complete task trajectories,
use the same draws for J and L, and do not refit models. The intervals therefore
condition on the frozen predictions and exclude model-selection uncertainty.

The untouched reserved cohort contains 20 tasks from eight repositories in
two pinned campaigns. It was selected without lens outputs, official outcomes,
gold patches, or problem difficulty. It is the first confirmatory test of the
post-development design. The protocol and serialized models must be committed
and pushed before any reserved trajectory is generated.

Reserved provenance uses a two-commit lifecycle. The model commit binds the
full fit-time protocol with a null reserved-summary pin and a core protocol
hash that normalizes only that one field. After legitimate trajectory
generation and strict prompt materialization, but before lens replay or
evaluation, a second commit may replace only the null field with the literal
prompt-summary SHA-256. Evaluation requires the same core hash and verifies
that exact null-to-SHA transition. Any other protocol change fails closed, and
refitting or reserializing the model after materialization is forbidden.

The frozen development go/no-go rule did not pass: 15 of 18 gates passed, but
the paired accuracy lower bound and paired NLL/Brier upper bounds failed.
Accordingly no reserved trajectory was generated, no reserved lens replay was
run, and no confirmatory accuracy was inspected. This preserves that cohort
for a future protocol that first clears an independent development screen.

If a later independent development protocol clears its go gate, the already
frozen validation decision is conjunctive. It requires support and replay
integrity, absolute J accuracy/calibration, and a positive paired advantage
over the ordinary-logit control. The principal floors include:

- at least 400 label-ascertained rows and 120 physical events;
- all 20 tasks and eight repositories, with class support across tasks/repos;
- J accuracy at least `0.80`, bootstrap lower bound at least `0.75`;
- J balanced accuracy at least `0.78`, lower bound at least `0.70`;
- edit/check recalls at least `0.65`/`0.80`, NLL at most `0.50`, ECE at most `0.15`;
- J-minus-logit accuracy at least `+0.02` with paired lower bound above zero;
- paired NLL and Brier interval upper bounds below zero.

All gates are encoded in the protocol. They cannot be relaxed after reserved
generation starts. A paired failure permits only a generic activation-readout
claim. An absolute failure means the readout is not reliable. A support or
integrity failure is inconclusive. Development and validation may not be
pooled to rescue a failed decision.

## Environment Setup

Create the separate small CPU readout environment:

```bash
scripts/setup_readout_v2_env.sh
```

This uses Python 3.12 and the exact versions in
[`requirements-readout-v2.txt`](../requirements-readout-v2.txt). The model
manifest records Python, NumPy, scikit-learn, joblib, SciPy, and threadpoolctl
versions. Evaluation verifies the serialized bundle hash before the joblib
pickle is loaded and then checks a frozen feature/probability canary.

The published joblib SHA-256 is an integrity pin for those exact bytes, not an
expected checksum for a fresh refit. Scikit-learn tree serialization can differ
in non-semantic bytes even when fitted tree arrays and predictions are equal.
Reproduction is therefore checked through the frozen inputs, implementation
and runtime hashes, estimator contract, training identities, feature hash, and
probability canary. Gate evidence is rounded to 14 decimal places before its
canonical decision hash so irrelevant floating-point reduction ULPs do not
change the scientific stop/go record.

## Reproduce The Development Freeze

First reproduce the existing dense development prompt/report pair if the
ignored cache is absent:

```bash
OUT_DIR="$PWD/.cache/swe_task_state_n20_dense" \
  scripts/run_swe_task_state_interpreter.sh
```

Then fit both matched branches and create the manifest and development report:

```bash
mkdir -p artifacts validation
.venv-readout-v2/bin/python scripts/analyze_swe_binary_phase_v2.py fit \
  --prompts .cache/swe_task_state_n20_dense/prompts.json \
  --public-report .cache/swe_task_state_n20_dense/public-report.json \
  --protocol configs/swe_binary_phase_interpreter_v2.json \
  --source-protocol configs/swe_task_state_interpreter_protocol.json \
  --behavioral-protocol configs/swe_behavioral_readout_protocol.json \
  --bundle artifacts/swe-binary-phase-v2.joblib \
  --manifest artifacts/swe-binary-phase-v2.manifest.json \
  --output validation/swe-binary-phase-v2-development.json
```

The input hashes must match the protocol. Fit emits a single joblib dictionary
containing both models. The published bundle SHA-256 is
`39941dd04c9287887b7daa4cb0b1905e231af55b5d8e5fcc2c0e0d1c81f8ed54`.
Never load an untrusted or unverified joblib file.

## Reserved Validation Is Blocked

Do not run the frozen reserved campaigns for this model. The development
report's `development_gates.passed` value is `false`, so the preregistered stop
rule applies before generation. This is a scientific stop, not an execution
failure.

The current `pins.reserved_prompts_summary_sha256` value is intentionally
null, which independently blocks `evaluate`. For a future protocol that first
passes its development go gate, use this order:

1. Commit and push the null-pin protocol, model bundle, manifest, and passing
   development decision.
2. Run `scripts/run_swe_task_state_validation.sh`. This generates the two
   campaigns sequentially, materializes every probeable request, and invokes
   the hash-pinned cohort checker on `prompts.json` and
   `prompts_summary.json`. Do not replay the lens or inspect outcomes yet.
3. Compute `sha256sum .cache/swe_task_state_validation_20260718/prompts_summary.json`
   and replace only `pins.reserved_prompts_summary_sha256` with that lowercase
   digest. Commit and push this provenance-only amendment. Do not refit or
   reserialize the model.
4. Replay the frozen public J and ordinary-logit branches, then call
   `evaluate` with the exact prompts, replay report, and
   `--reserved-prompts-summary` path. Evaluation re-runs the checker, verifies
   the prompt payloads and summary bindings, proves the sole null-to-SHA
   protocol transition, enforces the failed-development stop before joblib
   loading, and only then scores the reserved cohort.

Those steps are a future lifecycle specification, not commands to run for this
failed development model. No reserved prompt bundle or summary was created.

When a future independently developed protocol clears its go gate, generation
should still use the production compiled NVFP4 server with one-token MTP.
Replay should use the same target checkpoint in eager capture mode with MTP
disabled, because the draft block is a serving optimization and is not a
source layer in the target model. Layers `24..47` can be replayed together in
one model load.

The large exact prompts, conversations, and replay reports remain under
`.cache/` and are not published. The compact gate report, model manifest,
checksums, protocol, source, tests, and experiment report are safe tracked
artifacts after a credential scan.
