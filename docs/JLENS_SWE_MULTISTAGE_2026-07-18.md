# Qwen3.6 NVFP4 J-Lens SWE Multistage Pilot

**Date:** 2026-07-18
**Status:** completed N=1 development experiment; partially uncertified replay

## Bottom Line

The RTX 5090 reproduction succeeded at the engineering level:

- Qwen Code 0.19.4 generated a patch for `django__django-13297`; a
  contemporaneous but unsealed server log identifies the intended pinned
  `nvidia/Qwen3.6-27B-NVFP4` checkpoint, compiled vLLM, and one MTP speculative
  token. The sealed generation metadata does not independently prove that
  launch profile.
- The later pinned SWE-bench scorer resolved the exact generated patch, 1/1,
  with one FAIL_TO_PASS and all 52 PASS_TO_PASS tests successful.
- The exact 25-request trajectory was materialized into eight lifecycle
  prefixes and replayed through the pinned NVFP4 checkpoint in eager,
  language-model-only vLLM with MTP disabled. Public, NF4, and native NVFP4-STE
  J-lenses all decoded the paired residual captures.
- The corrected semantic visibility audit finds zero hidden concept rows:
  `SimpleLazyObject` exposes `lazy` from S0 onward. The earlier positive
  `lazy`-versus-`_dims` margins are an invalidated, lexically primed association
  control, not evidence that the lens recovered a hidden thought.

This is not a fully certified reproduction of the generation-time hidden
states. All three replay reports retain `status: failed`: the strict final
logit reconstruction check passed for 6/8 prompts and failed at S5 and S6.
The primary analysis excludes those rows and is explicitly
`descriptive_n1_development`. The action readout is below its majority baseline,
the outcome controls are underidentified, and neither result justifies a lens
refit.

The decisive next step is to freeze the lenses, correct the probe target and
foil, and then run independent SWE-Verified tasks with a task-held-out calibrated
readout before spending GPU time on another lens fit.

## Scope And Task Outcome

The single development trajectory is:

| Item | Value |
|---|---|
| SWE instance | `django__django-13297` |
| Repository | `django/django` |
| Base commit | `8954f255bbf5f4ee997fd6de62cb50fc9b5dd697` |
| Agent | Qwen Code `0.19.4` |
| Model ID | `qwen3.6-27b-nvfp4-mtp::qwen-code-0.19.4` |
| Requests / tool calls | 25 / 28 |
| Input / output / total tokens | 707,727 / 6,455 / 714,182 |
| Agent elapsed time | 164.17 seconds |
| Generation result | parsed CLI success; evaluation intentionally `skip` |
| Later official score | resolved, 1/1 |

The generated patch changes `django/views/generic/base.py`. It passes raw URL
kwargs into `get_context_data()`, then lazily wraps only URL kwargs that survive
in the returned context. The generated patch and officially scored patch are
byte-identical, SHA-256
`d571976cda27c77be7c9dac51a8b475e5262728a38d7a5ae93fb39b4336d4e1f`.
The official instance report records successful patch application, one
FAIL_TO_PASS success, 52 PASS_TO_PASS successes, and no failures.

The generation runner recorded the mutable image reference
`swebench/sweb.eval.x86_64.django_1776_django-13297:latest`. Its digest at
generation time is not proven. The later official score did verify
`swebench/sweb.eval.x86_64.django_1776_django-13297@sha256:0291be700f4db5aa369af9f7106943c9af1e46dcebf313d9999e88d851334bed`.
That distinction is preserved in the trajectory manifest and must not be
collapsed into a claim that the generation container was digest-pinned.

Primary local evidence:

- [generation metadata](../validation/swe-multistage-django-13297/generation/runner_metadata.json)
- [generated patch](../validation/swe-multistage-django-13297/generation/patch.diff)
- [pinned official report](../validation/swe-multistage-django-13297/official_score_pinned/qwen3.6-27b-nvfp4-mtp%3A%3Aqwen-code-0.19.4.lumo_j_lens_swe_multistage_pinned_20260718.json)
- [45-entry evidence ledger](../validation/swe-multistage-django-13297/evidence.sha256)

## Generation And Replay Are Different Runs

The lens replay uses the exact sealed chat prefixes and intended target
checkpoint, but it is not a capture of the generation process. The production
column below is reconstructed from a contemporaneous operational log that was
not included in the sealed evidence ledger.

| Setting | Observed production profile (unsealed) | J-lens replay |
|---|---|---|
| Model | `nvidia/Qwen3.6-27B-NVFP4` at `0893e1606ff3d5f97a441f405d5fc541a6bdf404` | same |
| Quantization | ModelOpt mixed NVFP4/FP8 | same pinned checkpoint |
| vLLM | 0.23.0, compiled, `enforce_eager=false` | 0.23.0, eager |
| Model path | target plus MTP draft | language model only |
| MTP | one speculative token | disabled |
| Maximum model length | 65,536 | 49,152 |
| Maximum batched tokens / sequences | 4,096 / 1 | 4,096 / replayed one at a time |
| GPU memory fraction | 0.85 | 0.78 |
| KV cache | auto resolved FP8 E4M3; 8 GiB native offload | FP8 E4M3; 8 GiB native offload |
| Prefix/chunking | prefix cache and chunked prefill enabled | prefix cache; final chunk retained |
| GDN/Mamba | Triton GDN prefill; Mamba block 1,024 and FP32 SSM cache | Mamba block 1,024 |
| Generation | stochastic request settings and MTP acceptance | one deterministic greedy token per prefix |
| Capture | no retained generation residuals | vLLM `apply_model` forward hooks |

The production profile above comes from the contemporaneous operational
`runs/server.log` around the 06:39:15-06:42:02 generation window. That log is
not checksum-bound by the final validation manifest. The sealed trajectory
does bind the 25 raw request dumps, usage ledger, runner metadata, patch, and
official score, but its model ID is a label rather than a checkpoint-revision
attestation. Therefore the checkpoint and MTP launch profile are operational
observations, not sealed reproduction evidence.

Each production request streamed with thinking enabled, `max_tokens=8192`,
`temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0`, and presence penalty
zero. Request seeds were `880001234..880001258`. The prompt grew to 40,989
tokens, so the 65,536-token production profile is material to this trajectory.

The replay is not an attempt to reproduce the stochastic continuation. It
re-renders the frozen messages with the checkpoint's pinned tokenizer and
chat template, evaluates the exact token IDs through the pinned replay
checkpoint, and captures the last prompt position (`-1`) after source layers
16 through 47. The scored band is fixed at layers 24 through 47. The readout
uses FP32 Jacobian transport and the checkpoint's BF16 final norm and LM head.
The greedy replay token is used only for accepted-token exclusion and numerical
checks; it is not substituted for the original next completion's action or
outcome label.

Hardware and replay software were NVIDIA GeForce RTX 5090 (32,607 MiB, compute
capability 12.0), driver 595.71.05, Python 3.12.13, PyTorch 2.11.0+cu130,
Transformers 5.12.1, Triton 3.6.0, and huggingface-hub 1.21.0. The target has
64 main-model blocks with width 5,120. Its pinned `config.json` and model index
hashes are respectively
`c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338`
and `7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2`.

## Request Definition

`chat_N` is the exact accumulated conversation prefix immediately before
completion N. Completion N contains the assistant reasoning fields, visible
text, and any tool calls. Tool results from completion N enter `chat_(N+1)`.
Thus the latest included completion in request N is completion N-1. The final
response is bound by terminal usage metadata.

For every selected stage, raw messages, normalized messages, rendered prompt,
and tokenizer IDs are chronological prefixes of this definition. Lifecycle
events were selected by the frozen rules in
[the lifecycle protocol](../configs/swe_multistage_protocol.json), with no lens
outputs inspected. Concept visibility is audited over assistant text/reasoning,
tool arguments, tool results, rendered prompt, scored token IDs, and
case-folded identifier segments and aliases. Next completion labels are used
only for the declared action and outcome analysis.

## S0-S7 Mapping

| Stage | Exact boundary | Request | Evidence request | Tokens | Analysis role | Next action |
|---|---|---:|---:|---:|---|---|
| S0 | task start | 1 | - | 12,625 | semantic-exposure control | inspect |
| S1 | first successful repository orientation | 2 | 2 | 13,480 | semantic-exposure control | inspect |
| S2 | first successful oracle source-body read | 5 | 5 | 16,185 | semantic-exposure control | inspect |
| S3 | first diagnosis after source read, before edit | 6 | 6 | 17,451 | semantic-exposure control | inspect |
| S4 | last request before first successful source edit | 13 | 14 | 31,385 | semantic-exposure control | edit |
| S5 | first post-edit prefix | 14 | 14 | 32,809 | semantic-exposure control | inspect |
| S6 | first successful post-edit validation prefix | 16 | 16 | 35,115 | semantic-exposure control | validate |
| S7 | prefix before terminal finalization after validation | 25 | 25 | 40,989 | semantic-exposure control | finalize |

At S4 the edit is issued in completion 13 and its successful result is visible
in request 14, which is why the selected pre-edit request and evidence request
differ. All eight stages were available; none was imputed. The gold target is
`lazy`, with exact one-token forms `lazy` and ` lazy`; its matched cross-task
foil is `_dims`. The old audit treated the lower-case token forms as hidden at
S0-S2, but the task statement already contains `SimpleLazyObject` and
`SimpleLazyObjects`. The corrected case-folded identifier-segment audit marks
`lazy` exposed from S0, so all eight stages are controls and there are zero
hidden-gold rows.

The contrast also does not represent Qwen's selected repair. `lazy` was derived
from the benchmark gold patch; Qwen's resolved patch retains
`SimpleLazyObject` and introduces `_lazy_reify_context_value` rather than
calling `lazy(...)`. `_dims` comes from an unrelated xarray task. This makes the
old contrast especially vulnerable to lexical and repository-domain effects.

The exact stage audit, including raw request and rendered prompt hashes, is in
[prompts_summary.json](../validation/jlens-swe-multistage-2026-07-18/prompts_summary.json).
The source prompt bundle is
`5ac1d1287e502590249b524df713d9850bb68ab4482f18244ea48566ff0ca83b`.

## Lenses And Metric Contract

| Readout | Fit | Artifact SHA-256 |
|---|---|---|
| Ordinary logit lens | no fitted transport | embedded identically in all three reports |
| Public J-lens | 1,000 prompts; published fit precision/quantization unspecified | `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1` |
| Local NF4 J-lens | 10 prompts on `Qwen/Qwen3.6-27B` NF4 | `54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f` |
| Local native J-lens | 10 prompts; exact NVFP4/FP8 forward with identity-STE surrogate backward | `82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057` |

The native identity STE is not the literal derivative of FP4/FP8 rounding. Its
fit contract and limitations are documented in
[JLENS_NVFP4_STE_EXPERIMENT.md](JLENS_NVFP4_STE_EXPERIMENT.md).

The hidden-gold metric would be the same-layer log-probability margin between a
predeclared, visibility-audited target and matched foil, averaged equally over
layers 24-47 and then within task. The corrected audit leaves no eligible rows,
so that metric is unavailable in this pilot. The older minimum-rank utility is
sensitivity-only.

Action classes are `inspect`, `edit`, `validate`, and `finalize`, each with six
predeclared one-token forms. At each layer, the class score is log-mean-exp over
its forms. The primary margin is expected class minus the maximum competing
class. Outcome uses the same construction with six `success` and six `failure`
forms. Transition outcome describes the actual next completion; terminal
outcome uses the official verdict. The official-outcome control maps this
resolved task to success at every stage.

Any target, foil, action, or outcome form equal to the replay's accepted greedy
token is excluded. If a contrast cannot survive that exclusion, the row is
unscorable rather than imputed. Accepted-token exclusion is separate from the
semantic visibility audit and cannot make the exposed `lazy` contrast hidden.
`diagnosis_expressed` is independent regex metadata and is not included in
action accuracy.

The configured bootstrap has 20,000 samples, seed 36,027, and task-level
resampling. With one task it cannot form a confidence interval, so all reported
comparisons have null intervals and descriptive-only inference.

## Numerical Certification

All three report processes exited 1 and retained `status: failed` for one
reason: `all_final_adapter_reconstructions_within_tolerance=false`.

| Check, per report | Passing prompts |
|---|---:|
| Greedy top-1 | 8/8 |
| Final norm | 8/8 |
| Final top-5 prefix | 8/8 |
| Full final logits | 6/8 |

S5 and S6 have final-logit maximum absolute error `0.125`, above the configured
`0.0625` limit. The other six are at or below `0.0625`. Across all rows, final
norm RMS error is at most `0.006`, final-logit RMS error is at most `0.01`, and
top-1/top-5 checks pass. The analysis therefore certifies 18 of 24 report rows:
eight prompts times three lens reports, with S5 and S6 failing in each report.
The ordinary logit baseline is repeated identically in those reports; it is not
a fourth independent capture.

Primary action and outcome metrics use only the six fully certified stages. S5
and S6 appear only in explicitly labeled inclusive sensitivity views. The
hidden-gold metric has zero eligible stages regardless of numerical status.
Prompt tokens, vocabulary metadata, residual manifests, runtime identity, and
fixed-band ordinary logit readouts pair exactly across public, NF4, and native
reports.

## Invalidated Concept Contrast

The corrected analysis has zero hidden-gold rows. For auditability, the table
below preserves the superseded fixed-band `lazy`-minus-`_dims` calculation that
first exposed the probe-design problem. These are semantic-exposure control
values, not hidden-reasoning results.

| Stage | Logit | Public J | NF4 J | Native J |
|---|---:|---:|---:|---:|
| S0 | 2.253526 | 4.936494 | 6.111070 | 5.410237 |
| S1 | -0.442160 | 0.081073 | 2.320210 | 1.974695 |
| S2 | -0.374130 | 0.228727 | 2.177924 | 2.109724 |
| Three-stage mean | 0.479079 | 1.748765 | 3.536401 | 3.164885 |
| Delta from logit | - | +1.269686 | +3.057322 | +2.685807 |

The values show that all J-lenses strengthen an already primed, Django-relevant
`lazy` association relative to the unrelated xarray `_dims` token. Because the
prompt contains `SimpleLazyObject` from S0, and because the target and foil do
not encode Qwen's actual repair alternatives, this table cannot distinguish
transport quality from lexical association or domain relevance. It establishes
neither a hidden thought nor a ranking among the fitted lenses.

## Next-Action Results

Certified primary support is six correlated prefixes: inspect 4, edit 1,
finalize 1, and validate 0. S5 and S6 are excluded, so the certified set lacks
the only validate example.

| Method | Band-correct micro | Observed-class macro | Mean expected-class margin |
|---|---:|---:|---:|
| Logit | 0.333333 | 0.166667 | -0.714452 |
| Public J | 0.500000 | 0.250000 | -1.007943 |
| NF4 J | 0.500000 | 0.250000 | -0.763083 |
| Native J | 0.500000 | 0.250000 | -0.868919 |

The certified majority baseline is 0.666667. The J-lenses classify more
stage-band decisions correctly than logit, but every mean expected-class margin
is negative, and the uncalibrated logit readout has the least-negative margin.
The two metrics therefore do not support a decisive action-prediction claim.

Immediate-next-action classification is also not directly aligned with the
J-lens objective. J-lens transports sensitivity summed over present and future
positions; it is not fitted to identify the next tool call. For example, a
`validate` preference at an inspection boundary can reflect test and
verification content that matters later in the trajectory. An action mismatch
is therefore not, by itself, evidence of a failed lens fit.

The inclusive eight-row sensitivity has micro band accuracy 0.375 for logit
and 0.500 for each J-lens; observed-class macro is 0.350 for logit and 0.400 for
each J-lens. It includes the uncertified S5 and S6 rows and is not primary.

## Outcome Results

In the certified six-row subset, transition and official outcomes are both
success for every row. The one failed transition occurs at uncertified S6.
Consequently the certified outcome task is one-class and non-identifiable as a
predictive evaluation.

| Method | Certified band-correct | Certified expected-score margin |
|---|---:|---:|
| Logit | 1.000000 | 0.719669 |
| Public J | 0.500000 | 1.258292 |
| NF4 J | 0.500000 | 1.041380 |
| Native J | 0.333333 | 0.682602 |

These values mainly expose each readout's global preference among the chosen
success/failure words on one resolved task. They do not measure general outcome
prediction. In the inclusive transition sensitivity, where support is seven
successes and one failure, band accuracy is 0.875, 0.500, 0.500, and 0.375 for
logit, public, NF4, and native respectively. That view includes the uncertified
failure row. Official outcome remains success at all eight stages.

## What This Establishes

The experiment establishes that:

- the pinned NVFP4 model, Qwen Code trajectory, stage materializer, three
  J-lens readouts, numerical audit, and official SWE scorer operate end to end
  on this RTX 5090;
- exact-prefix eager replay can capture paired NVFP4 residuals at 40k-token
  context length with MTP disabled;
- the stricter semantic audit detects the `SimpleLazyObject`/`lazy` exposure and
  removes all purported hidden rows;
- the current uncalibrated action and outcome token-set readouts are not
  reliable evaluation gates.

It does not establish:

- exact equality with generation-time hidden states or MTP-enabled compiled
  execution;
- a passing strict adapter certificate for S5 or S6;
- any hidden-concept recovery result, task generalization, causal mediation,
  stage prediction, or recovery of natural-language thoughts;
- superiority of native NVFP4, NF4, public J-lens, or J-lens generally.

The effective sample is one task, one successful trajectory, zero hidden
targets, and eight correlated exposure controls. The old comparison uses one
gold-patch target and one unrelated cross-task foil; its three early-stage
values are not independent examples. Local lenses use only ten fit prompts,
versus 1,000 for the public lens. The generation image digest and production
checkpoint/MTP profile are unproven by the sealed record. Confidence intervals
are null. These limitations dominate any ordering in the tables.

## Decisive Next Step

Do not refit J-lens yet.

1. Keep the public, NF4, and native lenses frozen. First run the existing
   resolved SymPy trajectory as an `N=2` automation and schema smoke test.
2. Freeze an algorithm that derives each target from the agent's actual later
   diagnosis or edit, without consulting lens output. A benchmark gold-patch
   token that the agent never uses is not sufficient.
3. Require every hidden candidate to be absent from all prior channels under
   exact scored-token IDs, Unicode case folding, snake/camel identifier
   segmentation, and declared semantic aliases. Relabel any failure as an
   exposure control.
4. Pair each retained target with a plausible same-task, same-family alternative
   rather than an identifier from an unrelated repository. Preserve separate
   method-aligned positive controls for known future-relevant concepts.
5. Automate 10-20 independent SWE-Verified tasks using the same request-boundary
   definition and lifecycle selectors, with balanced action checkpoints and both
   successful and failed validation transitions.
6. On disjoint calibration tasks, fit a small action readout over the frozen
   fixed-band class-score vector. A multinomial affine readout with temperature
   calibration is sufficient; freeze it before held-out evaluation.
7. Evaluate that frozen readout on independent held-out tasks with complete
   action-class support and task-cluster bootstrap intervals.
8. Refit a lens only if positive controls pass and calibrated held-out
   comparisons reveal a consistent lens-specific failure that cannot be
   explained by target leakage, foil quality, action vocabulary, checkpoint
   timing, class imbalance, or the readout itself.

This sequence fixes the current bottleneck, probe validity and action
measurement, before incurring another expensive NVFP4/NF4 lens fit.

## Reproduction Entry Point

The final replay was orchestrated by
[run_swe_multistage_pilot.sh](../scripts/run_swe_multistage_pilot.sh). With the
pinned environments and three lens artifacts present, a fresh replay uses:

```bash
OUT_DIR="$PWD/validation/jlens-swe-multistage-2026-07-18" \
  scripts/run_swe_multistage_pilot.sh
```

`--prepare-only` materializes and audits stages without loading the model.
`--reuse-reports` re-materializes inputs and re-runs analysis after validating
existing reports. The semantic-audit correction changes prompt metadata and
therefore cannot reuse the superseded reports: exact metadata pairing forces a
fresh three-lens replay. A fresh run may retain report exit status 1 when the
strict adapter check fails, while still writing the reports and final analysis
through the pilot's explicit failed-report handling.

`run_manifest.fresh-local.json` is the archived pre-correction run manifest. It
records the original measured phase statuses and absolute local paths, but it
also binds the invalidated visibility protocol and must not be used as the
current publication result. Its exact historical input bytes are preserved
under `validation/jlens-swe-multistage-2026-07-18/fresh-sources/`; those copies
are archival, not runnable current sources. The corrected portable
`run_manifest.json` uses repository-relative paths, records
`mode: fresh_replay`, and is the manifest to verify after cloning.

The setup and underlying fit contracts are documented in:

- [Qwen Code SWE experiment](JLENS_SWE_QWEN_CODE_EXPERIMENT.md)
- [NVFP4 STE experiment](JLENS_NVFP4_STE_EXPERIMENT.md)
- [multitask checkpoints](JLENS_SWE_MULTITASK_CHECKPOINTS_2026-07-18.md)

## Final Artifact Bindings

| Artifact | SHA-256 |
|---|---|
| Portable publication manifest | `c977fd5197466eb21b44e509f8c36016df0f3556e3bf30055d890f314f7ca6bd` |
| Archived pre-correction fresh-run local manifest | `104d9ba5d422be3a489f8fc225fce3724791efdfe270231a8b172af973ce689f` |
| Invalidated replay archive manifest | `bf0cb4e9e47c165f7fc5d49e8110fe03303471fb592508a9fea9407ddb0b6503` |
| Lifecycle protocol | `a850972479134e1b796592831ddd0927114253adde0d3f5413a31ee1ba5de5bf` |
| Trajectory manifest | `6161a8a8825676af60c1beb00e3d1f471b8b9f16b1b87b03704801c20bba1d33` |
| Action protocol | `bce204d03608e181456bb5c05a041c4bf4d305f48cb4b4e651ba34460d46d493` |
| Dataset | `f492377f82271dd83cfc45351863c4b2c05f9dab7f2951b6fe0c5977ea2f8697` |
| Evidence ledger | `6ca599fc730e51b45b2d2fc491a341650265b2c528a7811e9aa2bf4c98a47591` |
| Prompt bundle | `5ac1d1287e502590249b524df713d9850bb68ab4482f18244ea48566ff0ca83b` |
| Augmented action bundle | `7a76bb65f921d2d5cd20d818364be0b1d49544cb212425b61aee0e7999d8f719` |
| Public report | `d342eb6560f607a11fb98ba8f9a0a5b0b126c9162c186053f1b7b859a36c491d` |
| NF4 report | `ea2fa71d576ea894762ddd2768e51f4adfb7a98d7f765df48b919d9045fcc28a` |
| Native report | `dbb67f375f8a5f18253989af81a8286cac2c64b6ce8e800183217cf07d6d5712` |
| Final analysis | `174cf6a655076400a0e32d62df6fbcab112908d968f6217bfe535f7cf6f449e7` |
| Official aggregate report | `cd9a51e78b153f365b0c1e740ba9f832e8cb0e7825864a551748aeb98fd40466` |
| Official instance report | `36053d145d504a800648f6a1e981a586d27c5927bc794617a9d0515158a58a25` |

The manifest-linked result files are:

- [run manifest](../validation/jlens-swe-multistage-2026-07-18/run_manifest.json)
- [original fresh-run local manifest](../validation/jlens-swe-multistage-2026-07-18/run_manifest.fresh-local.json)
- [invalidated replay archive manifest](../validation/jlens-swe-multistage-2026-07-18/legacy-lexically-primed-audit/manifest.json)
- [public report](../validation/jlens-swe-multistage-2026-07-18/public-report.json)
- [NF4 report](../validation/jlens-swe-multistage-2026-07-18/nf4-report.json)
- [native report](../validation/jlens-swe-multistage-2026-07-18/native-report.json)
- [final analysis](../validation/jlens-swe-multistage-2026-07-18/analysis.json)

Run `scripts/check_swe_multistage_publication.py` to verify both manifest
sidecars, every current input and artifact, the 45-entry raw evidence ledger,
all eight compressed pre-correction artifacts, and the corrected zero-hidden-row
semantic contract. The same verifier is part of `scripts/check.sh`.

## Pinned Method References

- [Anthropic J-lens method](https://transformer-circuits.pub/2026/workspace/index.html#methods-jlens)
- [Objective comparison](https://transformer-circuits.pub/2026/workspace/index.html#methods-compare)
- [Quantitative comparison](https://transformer-circuits.pub/2026/workspace/index.html#app-quant)
- [Pinned upstream repository](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e)
- [Pinned fitting implementation](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/fitting.py)
- [Pinned evaluation conventions](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/data/evaluations/README.md)
