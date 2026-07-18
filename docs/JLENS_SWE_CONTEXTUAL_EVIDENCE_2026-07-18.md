# Qwen3.6 NVFP4 J-Lens Contextual Evidence Pilot

**Date:** 2026-07-18

**Execution status:** complete

**Frozen decision:** `frozen_directional_point_rule_pass`
**Scientific interpretation:** directional point estimates, not interval-supported or operationally useful

## Bottom Line

This experiment asked a narrower and more useful question than "what sentence
is the model thinking?" For 12 recorded Qwen Code trajectories, it compared
the model immediately before and after one observed task step and measured
whether a predeclared task concept gained more middle-layer support than three
same-task foils and 11 other tasks' targets.

The public J-lens passed the frozen **direction-only point-estimate rule** on the
eight control-matched tasks from seven repositories:

- target-versus-foil update: `+0.321868`;
- public J-lens minus ordinary logit lens on that update: `+0.317214`;
- own-task context selectivity: `+0.177475`;
- public J-lens minus ordinary logit lens on context selectivity: `+0.033281`.

That is the signal obtained from the lens. It is not yet a useful task
interpreter. The 95% confidence-interval lower bound is nonpositive for all
four quantities. The public J-lens retrieved the correct task target from the
12-target pool with MRR `0.305262` and top-1 `0.125`, while exact-token copy
frequency reached `0.681319`/`0.875` and copy recency reached
`0.625694`/`0.625`. The frozen copy diagnostic therefore failed. Only 3 of 12
task-card `WHY` fields passed every per-task guard; nine were withheld.

The local NF4 and native NVFP4/FP8-STE lenses closely tracked the public lens.
Their confidence intervals overlap, every direct public-versus-local interval
includes zero, and all three have top-1 `0.125`. This result does not isolate
NVFP4 fitting as the problem, so another expensive lens refit is not the next
step. The next useful test is a larger, independently preregistered set of
control-matched evidence transitions, followed by a causal swap only if the
readout first beats the copy and ordinary-logit controls.

Nothing in this experiment recovers hidden prose, a complete task plan, or
private chain of thought. It scores declared vocabulary entities at a prompt
boundary. A positive score update means stronger vocabulary alignment under
the fixed J-lens readout, not that the model internally stated the associated
English explanation.

## What Was Measured

One observation is a pair of consecutive Qwen Code request prefixes from one
SWE-bench Verified trajectory:

1. `before` ends immediately before an assistant completion.
2. `after` extends `before` by that completion and its tool result, then ends
   immediately before the next completion.
3. The following assistant completion supplies a target concept that is
   present in its reasoning or visible text; each of three foils is absent.

The materializer proves the two exact-prefix extensions and binds every raw
request by SHA-256. It retains hashes and concept-presence audits for the label
completion, but not its raw reasoning text. Gold patches and official outcomes
are not used to select the target or score the readout.

The 12 pairs contain 24 rendered prompts, 519,016 prompt tokens total. Prompt
length ranges from 13,247 to 37,964 tokens. Each prompt scores the same union
of 90 exact token IDs at the final prompt position across fixed source layers
24 through 47. There is no best-layer selection.

Eight pairs are primary-control eligible:

| Task target | Control match |
|---|---|
| Astropy `where` | target and foils exposed in both states |
| Django `mode` | target and foils exposed in both states |
| Matplotlib `weak` | target and foils newly exposed |
| xarray `squeeze` | target and foils statically exposed |
| Matplotlib `renderer` | target and foils exposed in both states |
| pytest `getattr` | target and foils exposed in both states |
| Requests `caught` | target and foils unexposed in both states |
| Sphinx `masking` | target and foils unexposed in both states |

The other four pairs remain descriptive because target/foil exposure frequency
does not match. Only the last two primary rows test a target that is absent
from both prompt states with equally unexposed foils. Six matched exposed rows
measure contextual reweighting of visible vocabulary, not prediction of a
hidden future entity. Two hidden-target rows are far too few for a novel
inference claim.

This is a development pilot, not an independent confirmation. The existing
N20 reports and earlier generic-action and patch-identifier results predated
the protocol. The contextual boundaries, candidates, controls, layer band,
and decision rule were frozen before any new contextual lens scores were
inspected.

## Relation To The Original J-Lens

This study keeps the core structure of Anthropic's
[Jacobian Lens method](https://transformer-circuits.pub/2026/workspace/index.html#methods-jlens)
and the public
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens)
implementation: a token-vocabulary readout of a transported residual over a
fixed middle-layer workspace band, with no best-layer search. The lens matrix
approximates the average causal Jacobian from a source residual to final-block
residuals at the present and future positions; applying final norm and the LM
head to the transported residual yields vocabulary scores. It is deliberately
not optimized to reproduce the current next-token distribution.

The before/after contrast is inspired by Anthropic's paired-context
evaluations, while the SWE adaptation adds controls that the original concept
benchmark does not need: same-task foils, an identical ordinary-logit readout,
all-task wrong-context retrieval, exact-token copy frequency, and exact-token
copy recency. The form and aggregation rules follow the pinned upstream
[evaluation conventions](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/data/evaluations/README.md)
where applicable.

The Apple-Silicon
[`WeZZard/jlens-qwen36`](https://github.com/WeZZard/jlens-qwen36) port was an
architecture reference, not the runtime or estimator used here. These replays
use raw vLLM forward hooks on the pinned NVIDIA NVFP4/FP8 main model on the RTX
5090; no TransformerLens model wrapper or Apple/Metal kernel is involved.

## Score Definition

For a concept with one or more declared token forms, the analyzer takes
log-mean-exp over forms separately at each layer and then averages that concept
score over the fixed layer band. It pools the three already band-averaged foil
concepts with a second log-mean-exp. This nonlinear reduction order is part of
the frozen contract:

```text
concept_score(state, concept) = mean over layers 24..47 of
                                logmeanexp(form scores at that layer)

foil_pool(state) = logmeanexp(concept_score(state, foil_1),
                              concept_score(state, foil_2),
                              concept_score(state, foil_3))

margin(state) = concept_score(state, target) - foil_pool(state)

target_margin_update = margin(after) - margin(before)
```

The ordinary-logit control uses the same residuals, tokens, positions, layers,
and reductions without the Jacobian transport matrix. `J minus logit` is the
paired difference between those two updates.

The wrong-task control scores every task target on every pair. It ranks the
pair's own target by `after - before` among all 12 targets and defines context
selectivity as the own-target update minus the mean update of the other task
targets. MRR and top-1 summarize that 12-way retrieval task. Exact ties receive
average rank; top-1 passes when no candidate is strictly greater.

Two model-free controls apply the same retrieval calculation to:

- exact token-form frequency, using paired `log1p(count)` updates;
- exact token-form recency, using paired negative `log1p(last distance)`
  updates.

Confidence intervals use 5,000 hierarchical repository-then-task bootstrap
draws with seed `36040`. Rows are never treated as independent bootstrap
observations. Every metric gives each task equal weight.

## Primary Stable Result

The primary profile includes all eight eligible tasks from seven repositories.
Values in brackets are paired hierarchical-bootstrap 95% confidence intervals.

| Readout | Target-margin update | Context selectivity | Own-target MRR | Own-target top-1 |
|---|---:|---:|---:|---:|
| Ordinary logit | `+0.004654` `[-0.226207, 0.247005]` | `+0.144194` `[-0.210274, 0.566474]` | `0.223264` `[0.117460, 0.344753]` | `0.000` `[0.000, 0.000]` |
| Public J, `n=1000` | `+0.321868` `[-0.084974, 0.783097]` | `+0.177475` `[-0.209548, 0.574091]` | `0.305262` `[0.139069, 0.532202]` | `0.125` `[0.000, 0.400]` |
| NF4 J, `n=10` | `+0.401771` `[-0.095678, 0.859420]` | `+0.175471` `[-0.308690, 0.775457]` | `0.305114` `[0.126887, 0.536458]` | `0.125` `[0.000, 0.400]` |
| Native NVFP4/FP8-STE J, `n=10` | `+0.447191` `[-0.040030, 0.898978]` | `+0.184032` `[-0.317517, 0.782977]` | `0.315530` `[0.136068, 0.546245]` | `0.125` `[0.000, 0.400]` |

The public J-lens differences from the ordinary logit lens are:

| Paired difference | Estimate | 95% CI |
|---|---:|---:|
| Target-margin update | `+0.317214` | `[-0.229511, 0.945667]` |
| Context selectivity | `+0.033281` | `[-0.245433, 0.333123]` |
| Own-target MRR | `+0.081999` | `[-0.104241, 0.244829]` |
| Own-target top-1 | `+0.125` | `[0.000, 0.400]` |

All four frozen directional point criteria are positive and the minimum sample
counts are met, so the literal frozen classification is
`frozen_directional_point_rule_pass`. The analyzer explicitly records
`classification_is_not_operational_usefulness: true`. All four primary
decision confidence-interval lower bounds are nonpositive. The direction is
therefore a lead for replication, not evidence that this is a reliable task
readout.

The public-versus-native target-update difference is `-0.125323`, 95% CI
`[-0.438704, 0.112123]`; public-versus-NF4 is `-0.079904`, CI
`[-0.401946, 0.227745]`. Their context-selectivity and MRR differences are
also near zero with intervals spanning zero. There is no measured basis here
for preferring one J-lens artifact.

## Copy Control

| Retrieval source | MRR | Top-1 |
|---|---:|---:|
| Public J-lens | `0.305262` | `0.125` |
| Exact-form copy frequency | `0.681319` | `0.875` |
| Exact-form copy recency | `0.625694` | `0.625` |

Public J minus frequency is `-0.376056` MRR, CI
`[-0.643736, -0.110822]`, and `-0.750` top-1, CI
`[-1.000, -0.428571]`. Public J minus recency is `-0.320432` MRR, CI
`[-0.584801, -0.100676]`, and `-0.500` top-1, CI
`[-0.857143, -0.142857]`. The J-lens loses all four point comparisons and
all four intervals exclude zero against it. This is the main practical reason
not to turn the directional result into a semantic task claim.

## Guarded Task Cards

`EVIDENCE`, `WHERE`, and `NEXT` are observed summaries bound to the original
Qwen Code trajectories; they are not lens output. `WHY` is emitted only when
one task passes all seven frozen guards: stable numerical replay,
control-match eligibility, immediate-next-token exclusion, positive public
target and context effects, and positive public-minus-logit effects for both.

Three `WHY` fields pass:

| Task | Guarded entity-level lens statement |
|---|---|
| `django-10914-mode` | Evidence for `file permission mode` increased against same-task foils, task-external targets, and corresponding logit effects. |
| `pytest-10356-getattr` | Evidence for `getattr MRO lookup` increased under the same controls. |
| `sphinx-8269-masking` | Evidence for `masking the HTTP error` increased under the same controls. |

The other nine `WHY` fields are withheld. Even the three emitted statements
are noncausal, entity-level descriptions. They do not establish the relation
expressed by the human-authored phrase, do not explain why the model acted,
and are not recovered chain of thought. The `masking` case is the only emitted
card whose target and foils were all absent from both prompt states; one case
cannot establish hidden-concept recovery.

## Numerical And Runtime Evidence

The target model is pinned `nvidia/Qwen3.6-27B-NVFP4` revision
`0893e1606ff3d5f97a441f405d5fc541a6bdf404`. All three reports pair exactly
on prompt text and IDs, residual-capture manifests, numerical diagnostics, and
ordinary-logit readouts.

The primary stable adapter profile was frozen from the earlier N20
reconstruction envelope before contextual scores were inspected. It requires
final-norm max/RMS error at most `0.125`/`0.006`, full-logit max/RMS error at
most `0.125`/`0.02`, exact greedy top-1, and an exact top-5 prefix. All 12
before/after pairs pass this profile.

The runner also retains the older strict full-logit tolerances
`0.0625`/`0.01`. Only 5/12 pairs pass that conjunctive profile, including 3/8
primary pairs, so each raw runner report correctly retains `status: failed`.
That status is a legacy numerical-certificate failure, not a missing replay or
an incomplete lens application. All 24 prompts passed final norm, greedy
top-1, and top-5 reconstruction; 14/24 passed the strict full-logit check.

| Lens report | Measured lifecycle | Shell wall time | Peak CUDA allocated / reserved |
|---|---:|---:|---:|
| Public | `115.705 s` | `120 s` | `26,087,818,752 / 26,531,069,952 B` |
| NF4 | `116.838 s` | `120 s` | `26,087,818,752 / 26,531,069,952 B` |
| Native NVFP4/FP8-STE | `120.147 s` | `128 s` | `26,087,818,752 / 26,531,069,952 B` |

These are one-load 24-prompt replays on the local RTX 5090.

## MTP Scope

The original Qwen Code trajectories were generated by the production NVFP4
server with one-token MTP speculative decoding. The lens replay intentionally
uses eager vLLM, `language_model_only=true`, and MTP disabled so forward hooks
can capture the 64-block target model's hidden states. MTP is a serving-time
draft-token optimization, not another source layer in the target model.

The replay reconstructs the exact frozen chat prefixes through the pinned main
model weights; it is not a retrospective capture of the original compiled,
MTP-enabled activations. Disabling MTP for this separate replay does not modify
or invalidate the production server profile. Production MTP serving remains
enabled and separately validated.

## Reproduction

The recommended fail-closed wrapper materializes a fresh immutable attempt,
verifies all model/lens/runner hashes, refuses a busy GPU or live generation
server, runs the three replays sequentially, validates their terminal reports,
and emits the final analysis and cards:

```bash
scripts/run_swe_contextual_evidence_pilot.sh
```

Use `--public-only` for the roughly two-minute public-plus-logit replay or
`--prepare-only` to validate and materialize the 24 prompts without loading the
GPU model. The steps below show the equivalent components explicitly.

Materialize and validate the frozen prompt pairs:

```bash
.venv-vllm/bin/python scripts/materialize_swe_contextual_evidence.py \
  --protocol configs/swe_contextual_evidence_protocol.json \
  --output-prompts .cache/swe_contextual_evidence/prompts.json \
  --output-manifest .cache/swe_contextual_evidence/manifest.json
```

Common replay arguments for all three lenses are:

```bash
COMMON=(
  --prompts-file .cache/swe_contextual_evidence/prompts.json
  --layers 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47
  --positions=-1 --top-k 10
  --max-model-len 65536 --max-num-batched-tokens 4096
  --mamba-block-size 1024 --enable-prefix-caching
  --kv-cache-dtype fp8_e4m3
  --kv-offloading-size 8 --kv-offloading-backend native
  --stream-final-only --gpu-memory-utilization 0.78
)
```

Run the public lens:

```bash
scripts/run_jlens_nvfp4.sh "${COMMON[@]}" \
  --lens-kind public \
  --output .cache/swe_contextual_evidence/reports/public-report.json
```

Run the local NF4 lens:

```bash
scripts/run_jlens_nvfp4.sh "${COMMON[@]}" \
  --lens-kind nf4 \
  --lens-path .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.pt \
  --lens-sha256 54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f \
  --lens-provenance .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.provenance.json \
  --output .cache/swe_contextual_evidence/reports/nf4-report.json
```

Run the native NVFP4/FP8-STE lens:

```bash
scripts/run_jlens_nvfp4.sh "${COMMON[@]}" \
  --lens-kind nvfp4-ste \
  --lens-path .cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt \
  --lens-sha256 82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057 \
  --lens-provenance .cache/nvfp4_ste_fit/final-mean/metadata.json \
  --lens-state .cache/nvfp4_ste_fit/state.json \
  --lens-state-sha256 f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6 \
  --output .cache/swe_contextual_evidence/reports/native-report.json
```

The runner exits nonzero when the retained legacy strict certificate fails;
do not discard a fresh complete report merely because it has `status: failed`.
The analyzer applies the separately frozen stable and strict profiles:

```bash
.venv-vllm/bin/python scripts/analyze_swe_contextual_evidence.py \
  --protocol configs/swe_contextual_evidence_protocol.json \
  --manifest .cache/swe_contextual_evidence/manifest.json \
  --public-report .cache/swe_contextual_evidence/reports/public-report.json \
  --nf4-report .cache/swe_contextual_evidence/reports/nf4-report.json \
  --native-report .cache/swe_contextual_evidence/reports/native-report.json \
  --output .cache/swe_contextual_evidence/analysis.json \
  --cards-output .cache/swe_contextual_evidence/cards.json
```

## Artifact Bindings

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `d2e32b4aa027ed387f1c8105046b2a102c3cfbe51d7879dbe86ebd804b58160a` |
| Materialization manifest | `9f7f0fdd4c90c596147b78cd17d9eb1469922dd6324103f058a1d42986fda01c` |
| Prompt bundle | `0662a327c4d13e4f359935e5766df53803939917095d00f1d5122865b425497d` |
| Public report | `3f166a2c6a84054323c423c121662022694e6154dd2ec991d6b185155ce1018f` |
| NF4 report | `30dc0d7fcaa4951c9c5f2bcb2785b380664e1381fb767aaf73f93b8d042e05ed` |
| Native report | `79ff0cce0c2d51d7091e3e0bb5b3b6ea70d9885170b07251403c13fc92f8f71c` |
| Final analysis | `7ff997dc97c4fdc8d4b66f78ee795cd4a762645852fb977e473646a5a38f511b` |
| Guarded cards | `f9aedb3f8052f47be9b9841b601bc96485a8d56e368ee66ad17a1957f7125cdf` |
| Compact run manifest | `6dd84b8c585ca7182c6474fe177ae5c0f1fa6bd76c4658b27d1a0bd96d3d73ce` |
| Public J-lens | `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1` |
| NF4 J-lens | `54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f` |
| Native NVFP4/FP8-STE J-lens | `82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057` |

The frozen contract is
[`configs/swe_contextual_evidence_protocol.json`](../configs/swe_contextual_evidence_protocol.json).
Committed compact evidence includes the
[analysis](../validation/jlens-swe-contextual-evidence-2026-07-18/publication/analysis.json),
[guarded cards](../validation/jlens-swe-contextual-evidence-2026-07-18/publication/cards.json),
[materialization manifest](../validation/jlens-swe-contextual-evidence-2026-07-18/publication/materialization-manifest.json),
[run manifest](../validation/jlens-swe-contextual-evidence-2026-07-18/publication/run_manifest.json),
and [checksums](../validation/jlens-swe-contextual-evidence-2026-07-18/publication/checksums.sha256).
The large prompt and raw report files remain under
`.cache/swe_contextual_evidence/`; the compact artifacts bind them by the
hashes above.

## Conclusion

The refined probe extracted a small, coherent contextual direction from all
three J-lens artifacts, including the native NVFP4 fit. It did not extract a
reliable task explanation. Sampling uncertainty remains large, trivial lexical
retrieval is much stronger, and only three individual entity statements pass
the frozen guard. Treat those three as candidates for an independently frozen
replication and eventual causal intervention, not as recovered reasoning.
