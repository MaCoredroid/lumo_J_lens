# Qwen3.6 SWE Multitask J-Lens Checkpoints

**Date:** 2026-07-18
**Status:** exploratory C0, capture-matched C0M, and C1 evaluations complete

## Semantic Visibility Erratum

The frozen C0M/C1 artifacts used identifier-boundary and exact scored-token
visibility checks. Those checks missed a semantic case variant: the Django task
contains `SimpleLazyObject` from task start, so the proposed target `lazy` is a
case-folded camel-case identifier segment at both C0M and C1. The corrected
audit finds no other new target or foil exposure in the historical C0M/C1
rendered prompts.

Therefore the historical eight-task, nine-concept cohort contains one exposed
control. Excluding Django leaves seven tasks and eight hidden concept candidates,
but the historical aggregate JSON and tables below were not recomputed. They
remain exact records of the superseded audit and must not be cited as a clean
hidden-concept evaluation. In particular, the Django `lazy` rank movement is a
lexically primed association control, not evidence of hidden reasoning.

## Scope

This historical experiment asked whether a Jacobian lens ranks gold-patch
concept candidates above matched controls during Qwen Code requests. It is a
concept-localization probe, not chain-of-thought recovery, task grading, or
causal evidence. The erratum above supersedes its Django visibility label and
its aggregate hidden-cohort interpretation.

One **request** is one model invocation at a completion boundary. It contains
the conversation accumulated so far and ends immediately before the next
assistant token. One **completion** is the assistant response to that request,
including its reasoning fields, visible text, and any tool calls. Tool results
from that completion become input to the next request.

- **C0 / task start:** the first request for each independent task, before the
  first assistant token and before any tool result.
- **C0M / capture-matched task start:** the exact first request from each
  retained C1 trajectory. Its rendered text and token IDs are an exact prefix
  of C1 and it retains the identical historical concept/foil vocabulary.
- **C1 / post-observation:** the second request, after the first completion and
  at least one genuinely successful repository read or search, but before the
  second assistant token.

Thus the original C0 sample is ten independent SWE tasks with one request each.
It is not ten completions from one agent trajectory. C0M and C1 are paired
boundaries from eight retained trajectories.

## Frozen C0 Protocol

The protocol was frozen without inspecting lens outputs. It selected ten tasks
from ten repositories, twelve unique target concepts, and eleven cross-task
foil occurrences. Target surfaces were derived from non-test Python files in
gold patches and excluded when visible in the task input or generic Qwen Code
template. Each retained bare or leading-space form is an exact, round-tripping
single Qwen token.

The readout uses source layers 16 through 47. For each concept and method, its
rank is the minimum exact rank over its predeclared forms and those 32 layers.
Utility is

`U = log(248320 / minimum_rank) / log(248320)`.

Utility and pass@k aggregation give equal weight to concepts within family,
families within task, and independent tasks overall. The rank min, median,
geometric mean, and max in the result tables are descriptive, unweighted
summaries over retained concepts. Pass@k is reported for
`k = 1, 5, 10, 50, 100, 1000`. Confidence intervals use 20,000 deterministic
paired task-level bootstrap samples. This minimum-over-forms-and-layers
statistic is intentionally sensitive and optimistic.

All runs use `nvidia/Qwen3.6-27B-NVFP4` revision
`0893e1606ff3d5f97a441f405d5fc541a6bdf404`, eager language-model-only vLLM,
MTP disabled, FP8 KV cache, and the final prompt-token position. The compared
readouts are the ordinary logit lens, the public `n=1000` Qwen3.6 lens, and the
local `n=10` native NVFP4/FP8 identity-STE lens.

## C0 Results

Both runner reports passed all ten final adapter checks. Public and native runs
have identical prompts, accepted generated tokens, residuals, and ordinary
logit-lens readouts. No scored form was excluded for matching the accepted
generated token.

| Method | Target U | Min / median / geometric-mean / max rank | Pass@1000 | Target-minus-foil U, 95% CI |
|---|---:|---:|---:|---:|
| Logit lens | 0.335734395448 | 274 / 6959 / 4714.405514 / 90162 | 0.4 | 0.079669565638 `[-0.018657263622, 0.172559063048]` |
| Public Jacobian | 0.312907126634 | 377 / 7102.5 / 8024.105765 / 131488 | 0.2 | 0.069039700070 `[-0.085851421802, 0.222026137642]` |
| Native Jacobian | 0.314525172542 | 258 / 8218 / 7120.447361 / 101026 | 0.3 | 0.071026974262 `[-0.122937761577, 0.258662409930]` |

All three methods have pass@1, pass@5, pass@10, pass@50, and pass@100 equal
to zero.

| Paired comparison | Delta U, 95% CI | Delta pass@1000, 95% CI |
|---|---:|---:|
| Public Jacobian minus logit | -0.022827268814 `[-0.105957193274, 0.061071697396]` | -0.2 `[-0.5, 0.0]` |
| Native Jacobian minus logit | -0.021209222905 `[-0.089765002812, 0.050461609039]` | -0.1 `[-0.4, 0.2]` |
| Native Jacobian minus public Jacobian | 0.001618045909 `[-0.051313681343, 0.048827304199]` | 0.1 `[0.0, 0.3]` |

The C0 result does not show a Jacobian-lens advantage over the logit lens. No
public-versus-native difference is detected at this sample size. This is an
exploratory result: the frozen metadata does not
contain preregistered decision thresholds, and every utility-difference
interval crosses zero.

## C0 Artifact Bindings

| Artifact | SHA-256 |
|---|---|
| Candidate manifest | `45f7ae73c3d207198dd20d29abbeae99725da262547eb181c91b0bd3ab792dea` |
| Frozen protocol | `9a7ac3d1206f83c1c964245f8f3ed2824cbfcd471b17fe81ae87b77c47045cb5` |
| Certified source request report | `94f3963be698b0e6e86d3878ba69efa53a94059e5520f292a97f344d6bed6fab` |
| Tokenizer JSON | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| Materialized prompt bundle | `2b0987ed67d1aa12a1df728d88141de7ed52ea7e6698d11f8e49437689688cda` |
| Prompt summary | `aa184169d035da79b148ee8b4845d00732536197c7cdf3700492186cf3bfd5f1` |
| Public runner report | `dd1a30f23f167c74383cea6176002ebb049393ef2bca9cc49f40b246bf5ca99d` |
| Native runner report | `1d39780fcaca68bf42eb54aad72578194c3d01c0af2824edbd7ccef830542c4b` |
| Analysis | `a51ccc3299fd029858bbbaea9a2dfcea5effe9ce6bdaf02d6ede827e22ccf082` |
| Paired residual-manifest list | `ee6b453640502d5e71cf094cad823e96400428e534f8daf99f8ebe558821570a` |
| Public lens | `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1` |
| Native lens | `82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057` |

The completed local artifacts are under `.cache/swe_multitask_initial/`.
The original C0 runner context predates the v2/v7 C1 captures and uses a
different Qwen Code startup/system prompt. C0 remains a valid frozen
cross-method task-start evaluation, but it is not used for trajectory-change
claims. C0M is the valid baseline for C1.

## Why Probe C1 Before Refitting

C0 is the earliest possible boundary. The model has only the issue and agent
instructions; it has not inspected the repository. A weak result can therefore
mean that patch-localization concepts have not become active yet, rather than
that either fitted lens is miscalibrated.

The public and native lenses behave similarly at C0, while both trail the logit
lens. That pattern does not isolate NVFP4 fit quality as the problem. A native
refit is also expensive and would risk optimizing the same under-activated
boundary. C1 is the cheaper discriminating experiment:

- stronger matched movement by both Jacobian lenses than logit would prioritize
  checkpoint timing over refitting;
- stronger public movement without native movement would prioritize native fit
  or calibration work;
- failure by both after relevant repository evidence would point first to
  concept construction, the reduction statistic, or the fit corpus.

Refitting should follow that diagnosis, not precede it.

## C1 Contract And Results

C1 merges request pairs from the original local v2 and v7 captures. Byte-exact
selected requests and their full usage ledgers are tracked under
`validation/swe-multitask-c1-capture-v2/` and
`validation/swe-multitask-c1-capture-v7/`, so materialization does not depend
on ignored local run directories. Each selected second
request exactly preserves its paired task-start messages, appends the first
assistant completion and paired tool results, and ends before the second
assistant token. A repository observation must be a non-mutating read/search
command with exit code and signal zero, no reported error or missing-path
diagnostic, and nonempty repository output.

Visibility is audited in assistant reasoning/text, tool-call arguments, tool
outputs, the rendered prompt, and the actual scored token IDs. Targets or foils
made visible are excluded before lens evaluation. That historical token-level
check is stricter than identifier-boundary matching but still misses
case-folded semantic segments whose tokenization differs, such as `lazy` inside
`SimpleLazyObject`. The current audit also segments snake-case and camel-case
identifiers and checks declared aliases.

The historical merged sample has eight tasks and nine primary concepts. The
corrected status column below marks the newly detected Django exposure:

| Capture | Task | Targets | Corrected status |
|---|---|---|---|
| v7 | `pydata__xarray-6938` | `equals` | hidden candidate |
| v7 | `scikit-learn__scikit-learn-9288` | `seed` | hidden candidate |
| v7 | `django__django-13297` | `lazy` | exposed control: `SimpleLazyObject` |
| v2 | `sympy__sympy-21847` | `sum` | hidden candidate |
| v2 | `pylint-dev__pylint-4551` | `visit`, `inspector` | hidden candidates |
| v2 | `sphinx-doc__sphinx-8638` | `PyObject` | hidden candidate |
| v2 | `astropy__astropy-14598` | `_split` | hidden candidate |
| v2 | `pytest-dev__pytest-7571` | `_finalize` | hidden candidate |

The xarray `_dims` target and seaborn `spacer` target are observed-token
controls, not hidden targets: `swap_dims` exposes the scored `_dims` token and
returned seaborn source exposes the scored `spacer` token inside `_spacer`.
Matplotlib is excluded because neither capture produced a qualifying first
repository observation.

### Aggregate C1 metrics

| Method | Target U | Min / median / geometric-mean / max rank | Pass@1000 | Target-minus-foil U, 95% CI |
|---|---:|---:|---:|---:|
| Logit lens | 0.257713267849 | 228 / 21830 / 9508.696038 / 196600 | 0.125 | 0.021880109876 `[-0.082606516696, 0.142682473048]` |
| Public Jacobian | 0.262472824386 | 1043 / 8830 / 9165.860811 / 121733 | 0.0 | 0.051614491945 `[-0.022137091060, 0.124689919217]` |
| Native Jacobian | 0.287378549481 | 209 / 8695 / 7276.350077 / 87849 | 0.125 | 0.073264319333 `[-0.038475223982, 0.204744937614]` |

All three methods have pass@1, pass@5, pass@10, pass@50, and pass@100 equal
to zero.

| Paired comparison | Delta U, 95% CI | Delta pass@1000, 95% CI |
|---|---:|---:|
| Public Jacobian minus logit | +0.004759556537 `[-0.123387922416, 0.132766104800]` | -0.125 `[-0.375, 0.0]` |
| Native Jacobian minus logit | +0.029665281632 `[-0.106690070487, 0.181250696305]` | 0.0 `[-0.375, 0.375]` |
| Native Jacobian minus public Jacobian | +0.024905725095 `[-0.010954258715, 0.064002777535]` | +0.125 `[0.0, 0.375]` |

Native has the best aggregate C1 utility and concept-level geometric mean rank,
but its advantage over logit is positive on only three of eight tasks and its
interval crosses zero. No public-versus-logit difference is detected. The
target-minus-foil point estimates are positive for all methods and largest for
native, but all three intervals also cross zero. These are directional
observations, not a claims-gate pass.

For Django's exposed `lazy` control, native ranks the best allowed form 209th in
the fixed middle band, versus 1,043 for public and 44,630 for logit. Native
improves from rank 607 at the matched C0M boundary, while logit worsens from
929. These are real historical ranks, but the prompt is already primed by
`SimpleLazyObject`; the movement cannot establish hidden-concept emergence or
an aggregate stage effect.

Both raw C1 reports preserve `status: failed`: final norm, greedy top-1, and
top-5 parity pass for 8/8 prompts, while the strict full-final-logit tolerance
passes for 6/8. Public and native prompt tokens, generated tokens, residual
manifests, and ordinary logit readouts pair exactly. The paired residual-list
digest is
`bc2ad58710ca41ff59729551944fa0cf0736155b5478293e296a788878d1cd9f`.

### Capture-matched C0M baseline

A final audit found that the original C0 system context differed from the later
v2/v7 captures before any assistant token. The preliminary C0-to-C1 deltas were
therefore confounded and are not published as stage evidence. C0M fixes this by
rendering the exact first request from every selected C1 pair. For all eight
tasks, C0M raw messages, normalized messages, rendered text, and token IDs are
exact C1 prefixes; the same historical nine targets and eight foils are
retained. Under the corrected audit, Django `lazy` is exposed at both boundaries
and the other eight targets have no new semantic identifier hit.

Both C0M runner reports pass every strict reconstruction check for 8/8 prompts.
Their prompt tokens, generated tokens, residual manifests, and ordinary logit
readouts pair exactly. The residual-list digest is
`1b65dc47de31b3cbf60474db1c70a39822e2d23ac9fac2ac696128103abcd82c`.

| Method | Target U | Min / median / geometric-mean / max rank | Pass@1000 | Target-minus-foil U, 95% CI |
|---|---:|---:|---:|---:|
| Logit lens | 0.321156180756 | 363 / 4623 / 5201.513749 / 94409 | 0.375 | 0.040013403256 `[-0.058020875752, 0.123838027388]` |
| Public Jacobian | 0.306427217475 | 882 / 2865 / 7225.743986 / 119966 | 0.125 | 0.085569071482 `[-0.088088861541, 0.254243412642]` |
| Native Jacobian | 0.316591861463 | 607 / 2418 / 5776.816242 / 83024 | 0.25 | 0.062196964957 `[-0.150486090593, 0.273475236764]` |

All three methods have pass@1, pass@5, pass@10, pass@50, and pass@100 equal
to zero.

| Paired comparison | Delta U, 95% CI | Delta pass@1000, 95% CI |
|---|---:|---:|
| Public Jacobian minus logit | -0.014728963281 `[-0.088345724760, 0.069771450054]` | -0.25 `[-0.625, 0.0]` |
| Native Jacobian minus logit | -0.004564319293 `[-0.069353251307, 0.059337962353]` | -0.125 `[-0.5, 0.25]` |
| Native Jacobian minus public Jacobian | +0.010164643987 `[-0.038281911011, 0.047402852501]` | +0.125 `[0.0, 0.375]` |

No C0M method-utility difference is detected.

### Matched C0M-to-C1 change

The comparator requires identical model, protocol, layer, task, concept, and
first/second request-hash bindings. Every method's aggregate utility has a
negative point change after the first generic repository observation:

| Method | C0M U | C1 U | C1 - C0M U, 95% CI | C1/C0M task-macro geometric-rank ratio |
|---|---:|---:|---:|---:|
| Logit lens | 0.321156180756 | 0.257713267849 | -0.063442912907 `[-0.154236120646, 0.020344563260]` | 2.199253 |
| Public Jacobian | 0.306427217475 | 0.262472824386 | -0.043954393090 `[-0.140136093756, 0.055645237661]` | 1.726372 |
| Native Jacobian | 0.316591861463 | 0.287378549481 | -0.029213311982 `[-0.123699169092, 0.055727679758]` | 1.437494 |

Each stage interval crosses zero. Direct difference-in-differences tests ask
whether a J-lens changes more favorably than its control across the same tasks:

| Stage contrast | Delta-of-delta U, 95% CI | Positive / negative tasks |
|---|---:|---:|
| Public Jacobian minus logit | +0.019488519817 `[-0.083924887533, 0.124464845351]` | 5 / 3 |
| Native Jacobian minus logit | +0.034229600925 `[-0.073643003789, 0.158838023190]` | 4 / 4 |
| Native Jacobian minus public Jacobian | +0.014741081108 `[-0.035943119426, 0.066124964127]` | 4 / 4 |

None is detected at this sample size. The point estimates are consistent with
J-lens associations degrading less than logit, but they do not establish
preservation. Target-minus-foil point estimates move from C0M to C1 as
`0.0400 -> 0.0219` for logit, `0.0856 -> 0.0516` for public, and
`0.0622 -> 0.0733` for native; no paired stage interval is claimed for those
descriptive values.

Concept movements are heterogeneous. Public improves `visit` from rank 32,471
to 1,689 and `inspector` from 119,966 to 26,792. Native moves the exposed `lazy`
control from 607 to 209, and improves `visit` from 42,979 to 5,896 and
`_finalize` from 33,301 to 10,667, while `PyObject` worsens from 2,418 to
87,849. A single generic read is not a reliable semantic checkpoint, and the
`lazy` movement is not hidden-concept evidence.

### C0M/C1 artifact bindings

| Artifact | SHA-256 |
|---|---|
| Merged capture manifest | `62fd83a16262845475057f950d2ac32816ad93fc9bd3802b6cdbb4f66319f895` |
| Merged usage-ledger manifest | `eb738f3be2f5537db78ac5fe154aa30d681bc948bf1960da65955ecd3493a967` |
| C0M prompt bundle | `fe822c10acec2f579161685a2c3c5937fa8a0efe90890d193475231633acb657` |
| C0M prompt summary | `d139d162038346d6b74faf5fbfc6423eeecee3877216c1e63a63beb93228a552` |
| C0M public report | `2fcc03ab726afd58d70788b4223bd67452fc527bf0f238bcdcbb789be5e98d71` |
| C0M native report | `39640b1d3f5b99b5d723ed3e12d4b6c612041f93a81c17d08db56fc6c0f380c2` |
| C0M analysis | `6fe987bf2eec089b683483c4c64d6c2782dd7770f8cd19d8b63340e7ebeccb55` |
| C1 prompt bundle | `45a00ae55b61ef2370a88cf2d177b8fa703a70abf04b9c9f4aa0ca10d1ebb5e6` |
| C1 prompt summary | `3dd850e86e50b20ace6156887fa91f1ba1a705f208e482d09ca7016c1ae89e9c` |
| C1 public report | `26b0b3a42c816adb5a6b3cda3e3262c225e3fbc5516ac686e601b320b733715e` |
| C1 native report | `e6619a803d01942101800b54f6de9fa6607cfac1cfdc8c31bf0cf437e533d159` |
| C1 analysis | `42a9e8d9b22cd61eb7bd47f6845936092476af2bf2023941537832af57628dc4` |
| C0M-to-C1 comparison | `3ecba1f2dd8a10e3bfd0c41ac44beb2c56528f5347b166ea8fd50a9f4a8a4692` |
| Evidence checksum manifest | `0d0c5c8733e2d913846a899ae38fc4b23dbf7cf38d4bfa0818a7f0c8ea3ee85b` |

The tracked evidence is under `validation/`. Run the complete C1 replay with:

```bash
scripts/stop_server.sh
scripts/run_swe_multitask_c1_pilot.sh
```

Use `--prepare-only` to audit/materialize both boundaries without loading the
model, `--reuse-c1-reports` to run only C0M, or `--reuse-reports` to recompute
statistics without another GPU replay.

## Next Experiment

Repair the probe before refitting the lens. Preserve C0M and C1, derive a target
from the agent's actual later diagnosis or edit without consulting lens output,
and match it to a plausible same-task, same-family foil. Reject a hidden label
if the target appears in any prior channel under exact token IDs, Unicode case
folding, snake/camel identifier segmentation, or declared semantic aliases.
Then add a semantically aligned **C2** boundary: after Qwen has read the relevant
source file or function, but before its next assistant token and before the
audited target appears. Freeze relevance before inspecting lens output, for
example as the first successful read that returns a non-test file changed by
the subsequent agent patch or its enclosing function. That is an explicitly
future-selected boundary and must be labeled as such. Report how many tasks
never reach it and keep their missingness separate from complete-case
C0M/C1/C2 estimates. Selecting an arbitrary later request would mix different
observation depths, survivorship, and explicit leakage.

The next diagnosis should use matched C0M/C1/C2 task changes, direct changes in
J-minus-logit contrasts, target-minus-foil behavior, and task-bootstrap
uncertainty. The outcomes prioritize follow-up work; they do not by themselves
prove a mechanism:

- if both J-lenses improve more than logit at C2 with aligned target-minus-foil
  movement, expand the trajectory probe to more tasks before making a timing
  claim;
- if public improves relative to logit and native does not, prioritize a native
  refit with substantially more code/agent-context prompts rather than the
  current ten short WikiText items;
- if neither improves relative to logit after relevant source evidence,
  prioritize revising the oracle concept and position objective before spending
  GPU time on a refit.

The current evidence does not support refitting yet: public and native remain
close at both matched checkpoints, every stage and stage-method interval crosses
zero, and a generic first observation is not semantically aligned enough to
separate timing from concept-objective quality.

## Methodological Limits

- C0, C0M, and C1 are associative token-rank probes. They do not expose hidden
  reasoning and do not show that a concept causally affects the completion.
- The historical C0M/C1 audit missed `lazy` inside `SimpleLazyObject`. Its
  aggregate JSON includes that exposed Django row; a corrected seven-task,
  eight-concept aggregate has not been computed.
- Gold-patch identifiers are oracle labels. They test localization and may not
  be the model's preferred semantic representation of a fix.
- Taking the minimum rank over multiple forms and 32 layers creates an
  optimistic multiple-opportunity statistic.
- Ten original C0 tasks and eight paired C0M/C1 tasks provide low task-level
  power. Repository and concept-family coverage remain sparse.
- Cross-task foils are imperfect semantic controls, and one C0 target has no
  assigned foil.
- C1 observations vary from path listings to source excerpts and sometimes
  include more than one parallel tool result. Observation depth is not
  controlled.
- The v2 and v7 captures ran against host workspaces and were intentionally
  stopped after two requests. All campaign entries are marked `skipped`, all
  patches are empty, and no official SWE harness verdict or resolved-task claim
  follows from them.
- Merging v2 and v7 introduces capture-batch and sampling-seed nuisance. The
  frozen model, task contract, tokenizer, visibility rules, and checkpoint
  boundary remain shared.
- The original C0 and captured C0M system contexts differ. C0 is reported only
  as a frozen cross-method task-start evaluation; all trajectory-change claims
  use exact C0M/C1 request bindings.
- Both C1 runner reports retain `status: failed` because strict full-final-logit
  reconstruction tolerance passes only 6/8 prompts, despite 8/8 final-norm,
  greedy-top-1, and top-5 checks. Interpretation is conditional on that
  disclosed adapter mismatch.
- Identifier-boundary and exact-token filtering are insufficient for subword
  models; semantically equivalent case variants can use different token IDs.
  Future probes must also audit case-folded snake/camel identifier segments and
  declared aliases.
