# Validation

## Jacobian Lens Fit And Transfer

Dates: 2026-07-16 to 2026-07-17 (America/Los_Angeles)

The July 16 fresh-fit result is a **completed exact dense NF4 fit**, not a
native NVFP4 fit. The July 17 native NVIDIA implementation separately completed
a strict `n=10` production fit through the pinned deployed ModelOpt NVFP4/FP8
forward on the RTX 5090. Its backward is a declared identity-STE surrogate,
including analytic GDN; it is not the literal derivative of quantization
rounding. MTP was disabled for capture, fitting, and evaluation because its
draft block is outside the accepted main-model lens.

| Claim | Result | Evidence |
|---|---|---|
| Real-model batched-VJP diagnostics (`C=4,8,32`) | PASS; exact equality to sequential autograd on source layers 61/62 | [`c4`](validation/jlens-nf4-diagnostic-c4-2026-07-16.json), [`c8`](validation/jlens-nf4-diagnostic-c8-2026-07-16.json), [`c32`](validation/jlens-nf4-diagnostic-c32-2026-07-16.json) |
| Exact `n=10` NF4 fit, layers `0..62` to target 63 | PASS; status `completed`, `complete=true` | [`fit provenance`](validation/jlens-nf4-fit-provenance-2026-07-16.json) |
| Published local artifact | PASS; 63 finite FP32 `[5120,5120]` matrices | [`artifact verification`](validation/jlens-nf4-artifact-verification-2026-07-16.json) |
| Held-out NF4 readout evaluation | COMPLETED; four prompts, four positions, all 63 layers | [`evaluation`](validation/jlens-nf4-eval-2026-07-16.json) |
| Dense local/public matrix comparison | REPORTED; no post-hoc similarity threshold | [`comparison`](validation/jlens-nf4-vs-public-2026-07-16.json) |
| Local NF4 lens applied to NVFP4, strict paired adapter gate | **FAIL**; cross-application is not certified | [`local lens run`](validation/jlens-nf4-on-nvfp4-2026-07-16.json) |
| Public lens on the same four NVFP4 prompts, control gate | **FAIL** with the same adapter errors | [`public control`](validation/jlens-public-on-nvfp4-heldout-2026-07-16.json) |
| Public lens on the original two semantic prompts | PASS after recertification | [`public baseline`](validation/jlens-nvfp4-2026-07-16.json) |
| Production native compiled baseline/observer proof | PASS for all ten prompts; exact endpoint generation, 688/688 shared tensors bit-exact, 432/432 observer-only boundaries complete, 785/785 replay parameters equal | [`fit state`](validation/jlens-nvfp4-ste-fit-state-2026-07-17.json) |
| Native packed/live VJPs, analytic GDN, and all-layer reverse replay | PASS; all 5,120 rows for source layers `0..62`, target block 63, 20 committed chunks per prompt | [`final metadata`](validation/jlens-nvfp4-ste-final-metadata-2026-07-17.json) |
| Dense native NVFP4/FP8-STE `n=10` artifact | **PASS**; 63 finite FP32 `[5120,5120]` matrices, exact production verifier passed | [`artifact verification`](validation/jlens-nvfp4-ste-artifact-verification-2026-07-17.json) |
| Upstream native artifact load | PASS with `JacobianLens.load` and `JacobianLens.from_pretrained` | [`loader record`](validation/jlens-nvfp4-ste-upstream-load-2026-07-17.json) |
| Native/public dense geometry | REPORTED; global cosine `0.732877`, mean layer cosine `0.822360`; no post-hoc threshold | [`geometry`](validation/jlens-nvfp4-ste-vs-public-2026-07-17.json) |
| Paired native/public held-out NVFP4 readout | REPORTED over 1,008 observations; both independent adapter certificates failed with identical pre-lens evidence | [`paired report`](validation/jlens-nvfp4-ste-vs-public-heldout-2026-07-17.json) |

The production native capture ran the local pinned vLLM/ModelOpt graph; this
RTX 5090 resolved W4 operations to the observed weight-only Marlin fallback and
FP8 operations to Cutlass. The runner discarded the older exploratory captures
and recaptured/reproved all ten prompts under model-identity, metadata, all
three shard hashes, prompt-manifest, and source-contract binding. MTP was
disabled because the proof covers main-model prefill, not speculative
draft/decode. On every prompt, 688 GDN/attention tensors were directly compared
bit-for-bit. The 432 linear/SwiGLU/post-block tensors exist only in the observer
graph and are supported indirectly by exact endpoint generation parity plus
the direct shared-tensor proof; they are not claimed as direct baseline
equality. Real packed-W4 and live-FP8 probes measured relative RMS `1.2549e-7`
and `7.8404e-7` against dense dequantization.

Run `20e4bc8c-9fed-4513-b548-9727f9686222` completed ten frozen 128-token
prompts in 47,577.883 seconds (13:12:57.9). Peak CUDA allocation/reservation was
8,936,882,688/11,404,312,576 bytes. The authoritative mean is 63
little-endian FP32 matrices totaling 6,606,028,800 bytes. Its aggregate layer
SHA-256 is
`a4c2adc7be15232db0e5a8840a6442248caa80a363c0c5239a1ee248f36fb3b4`;
the ten committed prompt records hash to
`a1690ab9e88cff53a2eba407195ced52e6908208fedffed68819ee47c1a888c1`.
The 6,606,046,478-byte exported checkpoint has SHA-256
`82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057`.
The exact verifier checked all 63 exported tensors, all ten prompt commits and
their 20 contiguous chunks, finiteness, checkpoint/source bindings, and model
identity. Upstream `jlens` 0.1.0 at commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e` loaded it through both supported
load APIs.

The state file is 329,400 bytes with SHA-256
`f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6`.
Final metadata is 988,263 bytes with SHA-256
`289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601`.
The frozen contract SHA-256 is
`7944ea163b548edc3372fa67242fbbcfbe0a5abbe95c04ce4a378107ebe03dd0`.
Dense geometry against the public `n=1000` FP16 lens measured global Frobenius
cosine `0.7328770738661481`, mean per-layer cosine `0.8223602375534815`, global
relative Frobenius difference `0.9345964627007955`, and all-row cosine mean
`0.791449281794253`. These are descriptive measurements, not equivalence gates.

The paired schema-3 readout covered four held-out prompts, positions 16, 32,
64, and 96, and all 63 layers: 1,008 observations. Native/public Jacobian
readouts measured target-rank Spearman `0.902843338526047`, top-1 agreement
`0.4126984126984127`, mean top-5 overlap `0.4936507936507937`, and target-score
RMSE `2.780105132813771`. Native target top-1/top-5 rates were
`0.06349206349206349`/`0.11904761904761904`; public rates were
`0.054563492063492064`/`0.1378968253968254`. Both independently executed
adapter certificates failed with the same reconstruction values. All four
residual-capture manifests and every logit-lens baseline field matched exactly,
so adapter status remains lens-independent and is not a lens-quality verdict.

The measured native production and export commands were:

```bash
.venv-vllm/bin/python scripts/run_nvfp4_ste_fit.py \
  --work-dir .cache/nvfp4_ste_fit

# Use this form after an interruption; it rehashes committed state first.
.venv-vllm/bin/python scripts/run_nvfp4_ste_fit.py \
  --work-dir .cache/nvfp4_ste_fit --resume

.venv-vllm/bin/python scripts/export_nvfp4_ste_lens.py \
  --final-mean .cache/nvfp4_ste_fit/final-mean \
  --output .cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt
```

Do not use `--resume` to bypass a contract mismatch. Resume revalidates the
model, sources, prompt corpus, committed chunks, sums, and capture proof before
continuing. The raw 6.15 GiB matrices and exported checkpoint remain under
`.cache`; the compact JSON evidence is committed instead.

The fit used `Qwen/Qwen3.6-27B` revision `6a9e13bd...`, 496
bitsandbytes NF4 linears, double quantization, BF16 compute, a 128-token
sequence, `skip_first=16`, cotangent batch 32, and the full 5,120 rows for all
63 source matrices. The estimator took 6,367.555 seconds; cumulative
in-process invocation time was 6,496.513 seconds (1:48:16.5). The timestamps
span 6,566.933 seconds including the pause before resume. Peak CUDA
allocation/reservation was 23.252/24.201 GiB. The resulting 6,606,048,039-byte
artifact has SHA-256
`54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f`.

The measured fit commands were:

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

The first invocation committed one prompt in 657.335 seconds and stopped by
contract. The resumed invocation completed the other nine in 5,839.178
seconds. Held-out NF4 evaluation then completed in 15.363 seconds and matched
the fit's aggregate NF4 weight hash exactly. The dense comparison with the
public `n=1000` FP16 lens of unpublished fit precision measured global
Frobenius cosine `0.750216`, mean
per-layer cosine `0.820655`, and global relative Frobenius difference
`0.865690`; these are descriptive measurements, not equivalence gates.

The local artifact was successfully loaded and read through every NVFP4 layer,
but the strict certificate ended with status `failed` after 33.328 seconds.
On Wikitext validation rows 3, 18, and 49, full-logit max error was `0.125`,
above the `0.0625` limit; row 42 was exactly at the limit. All four full-logit
RMS errors were below `0.01`, and all four reconstructed top-1 tokens matched
greedy generation. Row 18 also exceeded the final-norm max limit
(`0.25 > 0.125`) and missed the exact top-5 prefix. Repeating the run with the
public lens produced the identical reconstruction errors, demonstrating that
this strict failure is in the residual adapter check rather than the fitted
lens. It does not make the local NVFP4 cross-application certified.

As a descriptive cross-model readout comparison over the same 1,008
layer/position observations, local-lens NF4 versus NVFP4 outputs averaged
`0.769841` top-1 agreement, `0.816468` top-5 overlap, and `0.974066` Spearman
target-rank correlation. The public-lens control measured `0.755952`,
`0.818452`, and `0.979986`. These measurements do not override the failed
adapter certificate.

The original two-prompt public-lens baseline remains a separate passing run.
It was recertified in 16.300 seconds, including an 8.846-second model load,
with 25.618/27.596 GiB peak CUDA allocated/reserved. Both prompts had full-logit
max error `0.0625` at the allowed boundary, RMS errors `0.008794` and
`0.008464`, exact top-5 prefixes, and greedy top-1 agreement.

The detailed method, commands, result interpretation, and limitations are in
[`docs/JLENS_NF4_EXPERIMENT.md`](docs/JLENS_NF4_EXPERIMENT.md). Evidence-file
hashes are pinned in
[`validation/jlens-nf4-evidence.sha256`](validation/jlens-nf4-evidence.sha256).
The corresponding fit, evaluation, runner, checker, test, contract, and freeze
files are pinned in
[`validation/jlens-nf4-source-manifest.sha256`](validation/jlens-nf4-source-manifest.sha256).

## SWE Episode Jacobian-Lens Replay

Date: 2026-07-17 (America/Los_Angeles)

This experiment replays the nine exact next-token contexts from the certified
Qwen Code episode through the pinned deployed NVFP4/FP8 target model. It is not
a retrospective capture of the original hidden states: that server ran the
compiled MTP profile and did not retain activations. Replay uses eager hooks,
one-token greedy generation, and no MTP draft block. Prefix caching, chunked
prefill, FP8 E4M3 KV, and the original 1,024-token Mamba cache block remain
enabled.

The evaluation target follows Anthropic's published objective. J-lens averages
causal transport from a source residual to final-layer residuals at present and
future positions (`t' >= t`); it is not an affine predictor of the current
next-token distribution. Anthropic's
own quantitative comparison reports deliberately poor next-token KL through
most layers. Consequently, exact-rank intermediate recovery is the
method-aligned readout metric here. Dense next-token KL/NLL and semantic
distance to the captured final distribution are calibration diagnostics only,
not J-lens quality or reproduction gates. The pinned sources are the
[method](https://transformer-circuits.pub/2026/workspace/index.html#methods-jlens),
[objective comparison](https://transformer-circuits.pub/2026/workspace/index.html#methods-compare),
[quantitative appendix](https://transformer-circuits.pub/2026/workspace/index.html#app-quant),
and [upstream evaluation conventions](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/data/evaluations/README.md).

| Claim | Result | Evidence |
|---|---|---|
| Exact prompt reconstruction | PASS; all nine template renders match recorded counts `11,861..15,678` | [`provenance`](validation/jlens-swe-qwen-code-prompt-provenance-2026-07-17.json) |
| Longest-context capture preflight | PASS; 15,678 tokens, layers 31/32, strict adapter gate and final top-1 | [`preflight`](validation/jlens-swe-qwen-code-longest-preflight-2026-07-17.json) |
| Native NVFP4/FP8-STE all-layer replay | COMPLETED; nine prompts x 63 layers | [`native report`](validation/jlens-swe-qwen-code-native-nvfp4-ste-2026-07-17.json) |
| Public `n=1000` all-layer control | COMPLETED on independently replayed, byte-identical residual manifests | [`public report`](validation/jlens-swe-qwen-code-public-2026-07-17.json) |
| Paired native/public comparison | REPORTED; 567 paired observations and exact logit-lens identity | [`paired report`](validation/jlens-swe-qwen-code-native-vs-public-2026-07-17.json) |
| Same-context `co` versus `cot` probe | REPORTED for both native and public lenses over all 63 layers | [`native`](validation/jlens-swe-qwen-code-candidate-probe-2026-07-17.json), [`public`](validation/jlens-swe-qwen-code-candidate-probe-public-2026-07-17.json) |
| Pinned Anthropic multihop intermediate recovery | REPORTED; public and native J-lenses both outperform logit lens in the fixed middle band, with paired 95% CIs excluding zero | [`analysis`](validation/jlens-upstream-multihop-control-analysis-2026-07-17.json) |
| Dense 293-state next-token trajectory | CALIBRATION ONLY; J-lens is worse than logit lens on final-output KL/NLL, which is not its fit objective | [`analysis`](validation/jlens-swe-qwen-code-trajectory-calibration-2026-07-17.json) |
| Ten-state, 17-intermediate SWE concept probe | EXPLORATORY; fixed layer band and exact ranks reported, but one task, transcript leakage, and no preregistered claims gate | [`analysis`](validation/jlens-swe-qwen-code-intermediate-analysis-2026-07-17.json) |
| Ten-state semantic final-margin comparison | CALIBRATION ONLY; final-margin closeness is not a J-lens quality gate | [`analysis`](validation/jlens-swe-qwen-code-semantic-calibration-2026-07-17.json) |
| Official task outcome | PASS; `sympy__sympy-13480`, 1/1 resolved, zero errors | [`certified record`](validation/2026-07-15-publication-certified.json), [`patch`](validation/sympy__sympy-13480.patch) |

The method-aligned external control uses the pinned upstream
`lens-eval-multihop.json`: 93 items, 103 intermediate occurrences, 94
single-token-scorable occurrences, and nine exclusions counted as misses.
Over the primary fixed middle band, layers 24 through 47, the public J-lens
normalized exact log-rank AUC was `0.6237377` versus logit-lens `0.4707330`, a
gain of `+0.1530047` with paired item-bootstrap 95% CI
`[0.1148996, 0.1901072]`. Public pass-at-10 was `0.2903226` versus
`0.1344086`, a gain of `+0.1559140` with CI `[0.0645161, 0.2473118]`.
The native lens measured AUC `0.6190194`, gain `+0.1482864` with CI
`[0.1098553, 0.1862643]`, and pass-at-10 `0.2795699`, gain `+0.1451613`
with CI `[0.0537634, 0.2365591]`. The bootstrap resamples upstream items,
20,000 times at seed 36027.

Those multihop comparisons are numerically paired but not strict adapter
certificates. Public and native reports each retain `status: failed`: greedy
final top-1 and final-norm tolerance passed on 90/93 prompts, final top-5
parity on 85/93, and full final-logit tolerance on 72/93. Prompt IDs, token
inputs, residual manifests, and logit-lens fields match between the public and
native reports. The paired intermediate-rank comparison remains descriptive;
imperfect final-output parity limits causal claims.

The one-task SWE adaptation freezes ten exact trajectory points, 17 semantic
intermediates, and contiguous layers 16 through 47 before scoring; the
accepted next token is not a scored intermediate target. Public J-lens AUC was `0.7154340` versus
logit-lens `0.6918246`, a gain of `+0.0236094` with paired item-bootstrap 95%
CI `[-0.0736269, 0.1022612]` (7/10 item gains positive). Native J-lens was
`0.6916674`, a gain of `-0.0001572` with CI
`[-0.0969315, 0.0851788]` (6/10 positive). Public/native pass-at-10 were
`0.35`/`0.25`, versus logit-lens `0.40`. Seven post-tool boundary items mostly
name concepts already present in prior tool output. Their public/native AUC
gains were `+0.1062543`, CI `[0.0833565, 0.1291646]`, and `+0.0784104`, CI
`[0.0230768, 0.1347097]`; these same-task, mostly transcript-explicit items
can measure retention rather than novel inference. At the exact pre-identifier
state, public/native ranks for the two semantic concepts were
`[191,525]`/`[175,787]`, versus logit-lens `[3,463]`. This exact-identifier
weakness agrees with the old `co`/`cot` contrast: the J-lenses usually favored
`co`, but logit lens did so more strongly and both alternatives remained
outside the middle-layer top 10. All ten rows passed model-greedy top-1,
top-5, and final-norm checks, but only 5/10 passed full-final-logit tolerance;
the recorded accepted target matched greedy generation on 9/10 and was not a
scored intermediate. There is no preregistered claims gate and no
benchmark-generalization or latent-reasoning claim.

The dense trajectory makes the objective mismatch concrete. Over its fixed
middle slice, public J-lens final-output KL was `12.22461` versus logit-lens
`8.77098`; native J-lens was `13.09468`. Accepted-target log-probability gains
relative to logit lens were negative for public (`-3.52156`, request-bootstrap
95% CI `[-3.96742, -3.08083]`) and native (`-4.40035`, CI
`[-4.90920, -3.96555]`). These intervals quantify within-episode variation
over nine correlated requests, not task-level uncertainty. Only 192/293 prompts
passed the full strict numerical gate in either report. These values diagnose
output calibration and do not constitute a failed J-lens reproduction. The
earlier final-margin semantic
analysis is labeled the same way: public/native J-lens margins were positive
on 8/10 states, logit lens on 10/10, and J-lens was closer to the final margin
on only 4/10.

At stages 3 and 4, after source inspection and failure reproduction, native
layers 39/40 decode semantically clustered vocabulary such as
`obvious`/`clearly` and `confirm`/`confirming`. Later stages move through
repair/success terms, while the missing-`pytest` stage moves toward
`lack`/`unavailable`. The public control shows the same broad transitions.
These are decoded token distributions, not claims that a hidden state contains
literal English thoughts.

The primary candidate contrast uses an identical 12,775-token context for both
alternatives. It compares `co`, the first BPE token of correct `cothm`, against
`cot`, the first token of buggy `cotm`. Native J-lens margins (`co - cot`) are
`+2.06640625` at layer 31, `+1.9921875` at 32, `+3.0390625` at 35,
`+3.2578125` at 36, `+1.75` at 39, and `+2.3984375` at 40. The public lens
favors `co` at all nine layers in the fixed exploratory reporting slice;
native does so at eight of nine. At layer 62 native/public margins are
`+13.5625`/`+14.25`. The final
model margin is `+13.4999996424`, with `co` rank 1 and `cot` rank 2. Both
candidate runs have identical residual manifests for all five teacher-forced
steps. Both candidate tokens remain outside the top 10 through the middle
slice, whose leading tokens are mostly formatting or discourse vocabulary;
the contrast is a relative preference, not a claim that `cothm` is directly
decoded there. Full-string sequence totals are secondary because `cothm` uses
three BPE tokens while `cotm` uses two.

The fixed-middle native/public mean J-lens margins are `+1.6732` and `+1.5686`
respectively. The ordinary logit lens is positive at all nine layers with a
larger `+3.2440` mean. The correct first-token preference is therefore not a
signal uniquely exposed by J-lens transport.

The paired stage report measures native/public target-rank Spearman
`0.8239592012`, top-1 agreement `0.4268077601`, and mean top-5 overlap
`0.4800705467` over 567 observations. Greedy replay matches the original
temperature-1 sampled first token on eight of nine stages. The strict native
and public stage reports still have status `failed`: five stages have full-logit
maximum error `0.125` above the `0.0625` limit. Every stage passes final-norm
RMS, full-logit RMS, top-5 prefix, and greedy top-1 checks, and all paired
pre-lens residual/logit fields match. This remains descriptive evidence rather
than a passing strict adapter certificate.

The benchmark prompt already identified the undefined `cotm`, its file, and
line number. The run therefore does not demonstrate open-ended fault
localization. The narrower supported observation is that, after reading the
source and seeing `cothm = coth(m)` immediately above the bad reference, the
replay measures a preference for the correct first def-use token. The original
run ultimately emitted the successful one-character fix. Full commands,
artifact hashes, public-checkout prompt extraction, and limitations are in
[`docs/JLENS_SWE_QWEN_CODE_EXPERIMENT.md`](docs/JLENS_SWE_QWEN_CODE_EXPERIMENT.md).

## Multi-Task SWE Checkpoint Probe

Date: 2026-07-18

The task-start C0 protocol freezes ten independent SWE-Verified tasks, twelve
gold-patch concept candidates, eleven matched cross-task foils, and layers 16
through 47 without inspecting lens output. Its historical visibility labels use
the superseded boundary/token audit; the aggregate values below are retained as
historical outputs rather than recomputed. Ordinary logit-lens utility is
`0.335734`; public/native J-lens utilities are `0.312907`/`0.314525`. Their
paired differences from logit are `-0.022827`, 95% CI
`[-0.105957, 0.061072]`, and `-0.021209`, CI
`[-0.089765, 0.050462]`.

C1 is the second request after at least one successful repository read/search
and before the second assistant token. The historical identifier/scored-token
audit retained eight tasks and nine concepts. A corrected case-folded segment
audit finds `lazy` inside `SimpleLazyObject` at both C0M and C1, so the Django
row is an exposure control; the remaining candidate cohort is seven tasks and
eight concepts. No corrected aggregate JSON was computed. The historical C1
utilities (`0.287379` native, `0.262473` public, `0.257713` logit) include that
row and are not a clean hidden-concept evaluation. Native-minus-logit is
`+0.029665`, CI `[-0.106690, 0.181251]`; no method comparison excludes zero.

A final audit rejected the preliminary C0-to-C1 stage comparison because the
startup/system contexts differed. The replacement C0M baseline is the exact
first-request text/token prefix of each C1 trajectory and preserves the same
historical vocabulary. Matched logit/public/native utility changes are
`-0.063443/-0.043954/-0.029213`; all intervals cross zero. Direct
public-minus-logit and native-minus-logit stage contrasts are `+0.019489`, CI
`[-0.083925, 0.124465]`, and `+0.034230`, CI
`[-0.073643, 0.158838]`. These point estimates are consistent with less J-lens
degradation but do not detect preservation or aggregate concept emergence.
The `lazy` rank change is now an exposed lexical-association control, not a
hidden result. The next probe must apply the semantic audit, use agent-chosen
targets with same-task foils, then add an oracle-labeled C2 after relevant
source evidence and compare matched J-minus-logit changes. This result does not
yet motivate another fit.

Both C0M reports pass every strict check for 8/8 prompts. Both C1 reports pair
exact prompts, residual manifests, and ordinary logit readouts, but strict
full-final-logit tolerance passes 6/8, so the C1 reports retain `status: failed`.
The complete protocol, commands, hashes, limitations, and next decision are in
[`docs/JLENS_SWE_MULTITASK_CHECKPOINTS_2026-07-18.md`](docs/JLENS_SWE_MULTITASK_CHECKPOINTS_2026-07-18.md).

## Full-Lifecycle SWE Pilot

Date: 2026-07-18

The Django development trajectory contains 25 Qwen Code requests and an
officially resolved patch (`FAIL_TO_PASS 1/1`, `PASS_TO_PASS 52/52`). Eight
exact request prefixes were selected before lens output was inspected:
`S0..S7 = 1,2,5,6,13,14,16,25`. A corrected case-folded identifier audit finds
that `SimpleLazyObject` exposes the proposed target `lazy` from S0 onward.
Consequently all eight prefixes are semantic-exposure controls and the pilot
contains zero eligible hidden-gold rows. The generated patch and the patch
scored in the digest-pinned official rerun are byte-identical with SHA-256
`d571976cda27c77be7c9dac51a8b475e5262728a38d7a5ae93fb39b4336d4e1f`.
The original generation record contains only a mutable image tag, so the image
digest is proven for the later official score, not retrospectively for
generation.

Public `n=1000`, NF4 `n=10`, and native NVFP4/FP8-STE `n=10` lenses were
replayed on identical eager NVFP4 residuals at layers 16 through 47, with MTP
disabled for residual capture. A contemporaneous but unsealed server log says
the earlier Qwen Code generation used compiled vLLM and one MTP speculative
token; the sealed generation metadata does not independently prove that launch
profile or checkpoint revision. The replay reports have exact prompt,
scored-vocabulary, runtime, residual-manifest, and logit-readout pairing.

The superseded audit reported positive fixed-layer `lazy`-minus-`_dims`
margins. Those values are retained only as an invalidated association control:
`lazy` is semantically present in `SimpleLazyObject`, comes from the benchmark
gold patch rather than Qwen's actual resolved repair, and is compared with an
unrelated xarray identifier. The margins can reflect lexical priming and task
domain, so they do not support hidden-concept transport.

Next-action classification does not beat the majority baseline. On the six
certified rows, public/native/NF4 each reach `0.50` micro accuracy and `0.25`
observed-class macro recall, versus an inspect-majority accuracy of `0.6667`.
The inclusive sensitivity view remains `0.50` for each J-lens versus a
`0.625` majority baseline. Edit and finalize are missed by every J-lens. The
official-outcome control is degenerate (8/8 success), and the transition
control has only one failure, which every method misses.

Each report retains `status: failed` because S5/S6 final-logit maximum error is
`0.125` against a `0.0625` tolerance. All 24 rows pass greedy top-1,
final-norm, and top-five reconstruction; only those six report rows fail the
maximum-logit check, while their full-logit RMS remains below `0.01`. Primary
aggregates exclude them and the inclusive aggregates are sensitivity only.
The analysis status is `descriptive_n1_development` because all eight stages
come from one task.

The evidence does not justify refitting. The next gate is to freeze the current
lenses, derive targets from each agent's actual later diagnosis or edit, require
case-folded token/identifier/alias absence, and use plausible same-task foils.
After an `N=2` pipeline smoke test, automate 10-20 independent tasks with
balanced actions, real success/failure transitions, and a task-held-out
calibrated readout. The complete report, command, and limitations are in
[`docs/JLENS_SWE_MULTISTAGE_2026-07-18.md`](docs/JLENS_SWE_MULTISTAGE_2026-07-18.md);
the machine-readable result is
[`validation/jlens-swe-multistage-2026-07-18/analysis.json`](validation/jlens-swe-multistage-2026-07-18/analysis.json).

## Twenty-Task SWE Behavioral Study

Date: 2026-07-18

The N20 run contains two predeclared ten-task cohorts spanning 11 repositories,
699 Qwen Code requests, 20 nonempty generated patches, and 160 uniform
checkpoint prefixes. Generation used the pinned compiled NVFP4/FP8 server and
one-token MTP; 91,789/105,033 speculative draft tokens were accepted. Replay
used the same pinned main model in eager mode with MTP disabled and evaluated
public `n=1000`, NF4 `n=10`, native NVFP4/FP8-STE `n=10`, and ordinary logit
readouts at position `-1` over fixed layers 24-47.

| Claim | Result | Evidence |
|---|---|---|
| Exact task/request cohort | PASS; 20 tasks, 11 repositories, 699 requests, 8 uniform checkpoints per task | [`prompt summary`](validation/jlens-swe-behavioral-n20-2026-07-18/publication/prompts_summary.json) |
| MTP generation evidence | PASS; combined weighted acceptance `0.873906` | [`campaign evidence`](validation/jlens-swe-behavioral-n20-2026-07-18/publication/campaign_evidence.json) |
| Official SWE-bench score | COMPLETE; 9 resolved, 11 unresolved, zero error/empty | [`development`](validation/jlens-swe-behavioral-n20-2026-07-18/publication/official-outcomes/development.json), [`replication`](validation/jlens-swe-behavioral-n20-2026-07-18/publication/official-outcomes/replication.json) |
| Exact three-report pairing | PASS; prompts, residual manifests, runtime, final reconstruction, and logit readouts match | [`primary analysis`](validation/jlens-swe-behavioral-n20-2026-07-18/publication/analysis.json) |
| Strict numerical coverage | FAIL support gate; 68/160 jointly certified checkpoints versus 128 required | [`primary analysis`](validation/jlens-swe-behavioral-n20-2026-07-18/publication/analysis.json) |
| Preregistered behavioral decision | `insufficient_support`; action, outcome, and future-identifier coverage gates fail | [`protocol`](configs/swe_behavioral_readout_protocol.json), [`analysis`](validation/jlens-swe-behavioral-n20-2026-07-18/publication/analysis.json) |
| Greedy next-token transport sensitivity | SUPPLEMENTAL; 155/160 eligible, but ordinary logit beats public J-lens | [`protocol`](configs/swe_next_token_transport_protocol.json), [`result`](validation/jlens-swe-behavioral-n20-2026-07-18/publication/next-token-transport-analysis.json) |
| Learned action-layer readout | SUPPLEMENTAL; all classes recovered, but strict support is 66 rows and learned ordinary logit beats learned public J-lens | [`protocol`](configs/swe_action_layer_readout_protocol.json), [`result`](validation/jlens-swe-behavioral-n20-2026-07-18/publication/action-layer-readout.json) |

All three replay processes completed their 160 rows and returned report status
`failed` because the strict gate is conjunctive. Per report, 159/160 greedy
top-1 and top-5 checks pass, 157/160 final-norm checks pass, 70/160 full-logit
checks pass, and 68/160 pass every check. The same reconstruction evidence is
present before lens-specific scores are compared. It diagnoses the residual
adapter rather than an individual lens, while still limiting the primary
scientific claim.

The strict behavioral analysis has only 66 action-labeled certified rows, 8
certified latest-task official outcomes, and four future-identifier rows from
one task. Its frozen decision is therefore `insufficient_support` and its next
step is to collect the missing jointly certified task controls without
refitting. A sensitivity amendment was frozen after public numerical
diagnostics but before lens ranks or log probabilities were inspected. It
retains 155/160 rows, yet public J-lens minus ordinary logit normalized
greedy-token rank utility is `-0.23449`, paired 95% CI
`[-0.26190, -0.20507]`. Public minus native is `+0.06914`, CI
`[0.06313, 0.07519]`, and public minus NF4 is `+0.13924`, CI
`[0.13169, 0.14722]`. Because the public positive control fails, those local
deficits do not authorize a refit.

The nested leave-one-repository-out action supplement learns from all 96 fixed
layer-by-class features. On 66 strict rows, balanced accuracy is `0.80729` for
ordinary logit, `0.78125` for NF4 J-lens, `0.75000` for native J-lens, and
`0.68229` for public J-lens. Learned public improves over the frozen public
band-average readout by `+0.20833`, CI `[-0.04666, 0.46348]`, but trails
learned ordinary logit by `-0.12500`, CI `[-0.33786, -0.00962]`. The
149-row sensitivity track misses its six-rows-per-task requirement and remains
descriptive. A post-hoc, outer-held-out checkpoint-ordinal prior reaches
`0.63542` strict balanced accuracy, demonstrating that uniform checkpoint
position itself carries substantial task-phase information. These descriptive
results justify testing a progress-aware action readout with adequate support;
they do not justify a new native lens fit.

The complete accounting, estimator distinctions, supplemental controls, raw
artifact boundary, and next-step decision are in
[`docs/JLENS_SWE_BEHAVIORAL_N20_2026-07-18.md`](docs/JLENS_SWE_BEHAVIORAL_N20_2026-07-18.md).
The compact publication intentionally omits the 86,989,478-byte prompt bundle
and three approximately 274 MB full lens reports; their hashes bind identity
but do not make their content independently available.

## Publication-Certified Run

Date: 2026-07-15 (America/Los_Angeles)

Run: `publication_certified_v2_20260715`

Result: **PASS**

This run validates the extracted repository itself, including the final security
boundary. It closes the historical gap between the July 8 gate, which used Qwen
Code 0.19.2 and a temperature-0.6 envelope, and the final Qwen Code 0.19.4,
temperature-1.0 thinking profile.

| Item | Certified result |
|---|---|
| Model | `nvidia/Qwen3.6-27B-NVFP4` at revision `0893e160...` |
| Server | vLLM 0.23.0, ModelOpt mixed NVFP4, FP8 E4M3 KV |
| MTP | native one-token draft model detected and active |
| Cache | prefix caching, chunked prefill, Mamba `align`/FP32 |
| Agent | Qwen Code 0.19.4, thinking enabled |
| Boundary | isolated HOME/CWD; exact tool schema `[run_shell_command]` |
| Runtime | pinned official task image, container network `none` |
| Task | `sympy__sympy-13480` |
| Qwen Code | exit 0, 9 turns/requests, 32.406 s |
| Patch | 590 bytes; SHA-256 `98e80f91393e...` |
| Official score | 1/1 resolved; zero unresolved, empty, or error |

### Setup and server evidence

- `.venv-vllm` exactly matched the 190-package freeze; `uv pip check` passed.
- `.venv-swe` exactly matched the 72-package freeze; `uv pip check` passed.
- `npm ci` installed Qwen Code 0.19.4 with zero reported vulnerabilities.
- The server passed the exact-name and 32,768-context endpoint gate.
- vLLM resolved `modelopt_fp4` to `modelopt_mixed` and KV to `fp8_e4m3`.
- The engine detected and shared weights with the native MTP draft model.
- With a warm compilation cache, engine initialization took 10.00 seconds,
  including 5.14 seconds compilation.
- GPU KV capacity was 83,012 tokens.
- MTP acceptance samples were 92.8% and 93.3%.
- Prefix-cache hit samples rose from 51.7% to 66.3%.
- Generation-throughput samples were 48.6 and 51.7 tokens/second.
- GPU memory returned to 383 MiB after server shutdown.

### Agent and isolation evidence

All nine forwarded requests declared exactly one well-formed function tool:
`run_shell_command`. The proxy rejects extra, malformed, non-function, or
duplicate entries. The agent executed nine shell calls successfully through the
Docker shim and attempted no other tool. The task container used `--network
none`, a 6 GiB memory ceiling, 8 GiB memory-plus-swap ceiling, and 1,024-PID
ceiling. Qwen ran with a scrubbed environment, fresh HOME, and an isolated CWD
containing only `AGENTS.md`.

The first forwarded envelope was:

```json
{
  "model": "qwen3.6-27b-nvfp4",
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 0.0,
  "seed": 880001234,
  "max_tokens": 8192,
  "chat_template_kwargs": {"enable_thinking": true},
  "tools": ["run_shell_command"]
}
```

Seeds increased monotonically through `880001242`. Aggregate model usage was
124,864 input tokens and 1,439 output tokens.

### Task and scorer evidence

The model corrected the undefined `cotm` reference to the existing `cothm`
variable in `sympy/functions/elementary/hyperbolic.py`. The official SWE-bench
4.1.0 harness, using the same materialized pinned dataset row, reported:

```json
{
  "submitted_instances": 1,
  "completed_instances": 1,
  "resolved_instances": 1,
  "unresolved_instances": 0,
  "empty_patch_instances": 0,
  "error_instances": 0,
  "resolved_ids": ["sympy__sympy-13480"]
}
```

Generation deliberately ran with local evaluation skipped, so its internal
campaign summary contains no resolved verdict. The separate official report
above is the benchmark verdict, as required by the memory-safe lifecycle.

Published evidence:

- `validation/2026-07-15-publication-certified.json`: sanitized machine record
- `validation/first-forwarded-request.json`: sanitized request envelope
- `validation/sympy__sympy-13480.patch`: exact generated patch
- `validation/official-report.json`: exact official aggregate report
- `validation/runtime-source-manifest.sha256`: exact certified runtime sources

Artifact hashes:

| Artifact | SHA-256 |
|---|---|
| Patch | `98e80f91393e76fc6323eeb0a7582aa89f1db717932ea5742eed467d692d024d` |
| Predictions JSONL | `7ea5811e0765855a1a817a06e975893368cc4304e24d093cfb4cc7d857fab4bb` |
| Materialized dataset | `e8d93ec886c5d367e412c2ec04c7a7696af3c821ab9fb3f0e03261a7743445e2` |
| Official report | `6f0e1c2ce449d411df0f2f8ffaf2680f1178ced057cee156c989a53732b0f7f8` |
| Runtime source manifest | `9537252ade61d1c58350b77b182180d8e09ea2a445d42d98218a210641f2bfcd` |

## Static Verification

After the final documentation-alignment changes, `scripts/check.sh` passed:

- all shell scripts passed `bash -n`;
- all Python entry points and tests passed `py_compile`;
- 24 standalone envelope/tool-boundary assertions passed;
- 426 discovered unit tests passed with one intentional skip;
- both Python environments passed compatibility and exact-freeze checks; and
- the installed Qwen Code version was 0.19.4.

## Historical July 8 Gate

The source session's earlier three-gate run established broader serving evidence:

- format: zero schema mismatches and identical `qwen3_xml` structure;
- argument grounding: 63/64 source-verbatim arguments, zero malformed;
- live tasks: 4/4 resolved, including 2/2 official Verified; and
- MTP A/B: 109.4 versus 68.8 tokens/second, 1.59x at 93.02% acceptance.

The historical Verified tasks were `django__django-10914` and
`sympy__sympy-13480`. Those results are not presented as final-stack
certification; the publication-certified run above is that evidence.
