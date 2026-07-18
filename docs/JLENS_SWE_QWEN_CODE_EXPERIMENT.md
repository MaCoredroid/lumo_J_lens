# Jacobian Lens Replay of a Qwen Code SWE-Verified Task

Date: 2026-07-17

## Outcome

The engineering reproduction completed on the local RTX 5090: the public lens
and the locally fitted native NVFP4/FP8-STE lens both load and decode frozen
Qwen Code residuals. A pinned Anthropic multihop control also reproduced the
method-aligned result: both J-lenses recover known intermediate concepts better
than logit lens over the fixed middle band. The single SWE episode is more
limited. It contains semantically relevant decoded readouts and a usually
positive preference for the correct `cothm` definition-use repair, but the
ordinary logit lens is stronger on that exact identifier contrast. This is a
completed exploratory replay, not a claim that hidden states are literal
natural-language thoughts or that J-lens has passed a multi-task SWE gate.

The underlying certified SWE-bench result is independent and unambiguous:
`sympy__sympy-13480` was resolved, 1/1. The patch changes one identifier in
`sympy/functions/elementary/hyperbolic.py`:

```diff
-                    if cotm is S.ComplexInfinity:
+                    if cothm is S.ComplexInfinity:
```

The official scorer recorded the FAIL_TO_PASS `test_coth` and all 43
PASS_TO_PASS tests as successful. The frozen certification record reports one
submitted, one completed, one resolved, zero unresolved, and zero errors.
The raw harness test output reported 45 passed; the additional
`test_cosh_expansion` was not in either scorer classification.

The strict status needs a separate statement. Both nine-context stage replay
reports have `status: failed` because five final-logit maximum absolute errors
were `0.125` rather than the configured `0.0625` limit. Every stage final
top-1 check, top-5 prefix check, final-norm maximum and
RMS check, and final-logit RMS check passed. The decoded lens measurements are
therefore published as descriptive results with the failed adapter certificate
preserved, not relabeled as a passing certificate.

## What Was Replayed

The source episode is the certified Qwen Code 0.19.4 run
`publication_certified_v2_20260715`. Qwen Code made nine forwarded chat
requests with thinking enabled against the pinned NVIDIA model. The original
server used compiled vLLM, one MTP speculative token, temperature 1.0,
`top_p=0.95`, `top_k=20`, and seeds `880001234..880001242`. It emitted 1,439
output tokens in 32.406 seconds and produced the scored patch above.
That server used `max_model_len=32768` and GPU fraction 0.85. The hookable
replay uses eager execution, `max_model_len=16384`, and GPU fraction 0.78;
all frozen prompt lengths fit without truncation.

This experiment did not retain or recover the original run's hidden states.
It deterministically rendered the exact nine request bodies with the exact
chat template and tokenizer, then replayed those frozen token IDs through the
same pinned target-model checkpoint in eager vLLM so hooks could capture the
last prompt-token residual after every main-model block. The nine prompt
lengths were exactly:

```text
11861, 12148, 12743, 13629, 13883, 14522, 15073, 15327, 15678
```

For source block `l`, the native readout is:

```text
transported_l = FP32(h_l) @ FP32(J_l).T
logits_l      = quantized_lm_head(final_rms_norm(BF16(transported_l)))
```

The ordinary logit lens uses the same final RMSNorm and quantized LM head but
does not apply `J_l`. The reports cover source blocks `0..62`; block 63 is the
fitted target and is captured separately as the final-model control. The MTP
draft block is not a source layer.

Long prompts were evaluated with `max_num_batched_tokens=4096`, so vLLM used
chunked prefill. `--stream-final-only` allowed each hook to overwrite its
captured chunk tail and retain only the final chunk's last prompt token.
Prefix caching was enabled to match the server setting, although these unique
request tails do not constitute a cache-equivalence proof for the reused
prefix states that were not captured.

## Task Difficulty And Interpretive Scope

This episode is not evidence of open-ended bug localization. The SWE task
prompt itself names `hyperbolic.py`, line 590, the `NameError`, and the
undefined spelling `cotm`. Requests 1 and 2 mainly inspect the location the
prompt already supplied.

The new program evidence arrives in the tool response before request 3. The
source shows:

```python
cothm = coth(m)
if cotm is S.ComplexInfinity:
```

The supported observation is narrow: after observing that adjacent definition,
the frozen turn-3 context supports the exact def-use correction from `cotm`
to `cothm`. This readout experiment does not establish a causal mechanism, and
it does not show that the model would have discovered the file or line without
the task prompt.

## Evaluation Objective

The J-lens objective is present-and-future-summed causal transport, not
reconstruction of the current next-token distribution. At a source layer and
token state it estimates an average Jacobian from that residual to final-layer
residuals at positions `t' >= t`, then decodes the transported state without
fitting a current-context affine intercept. Schematically:

```text
J_l       = E[d h_final,t' / d h_l,t] over target positions t' >= t
readout_l = softmax(W_U norm(J_l h_l,t))
```

That distinction changes the evaluation. Anthropic explicitly reports that
J-lens has the worst next-token KL through most of the network and describes
this as a feature: the readout emphasizes information that will matter later,
not everything already present in the current output distribution. Dense
next-token KL, accepted-target NLL, and distance to the captured final margin
are therefore useful calibration diagnostics, but they are not J-lens quality
or reproduction gates.

The method-aligned quality check uses known semantic intermediates, verifies
each candidate surface is one tokenizer token, takes the exact minimum rank
over its allowed forms and a fixed layer band, and reports item-macro pass-at-k
and normalized log-rank AUC. The pinned references are:

- [Anthropic J-lens method](https://transformer-circuits.pub/2026/workspace/index.html#methods-jlens)
- [Objective comparison](https://transformer-circuits.pub/2026/workspace/index.html#methods-compare)
- [Quantitative comparison](https://transformer-circuits.pub/2026/workspace/index.html#app-quant)
- [Pinned upstream repository](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e)
- [Pinned fitting implementation](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/fitting.py)
- [Pinned evaluation conventions](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/data/evaluations/README.md)

Intervention-induced KL is a separate, valid causal metric: after ablating or
swapping a concept-selected J-space direction, measure the change in the
model's downstream output. That does not compare a J-lens readout with the
unmodified final next-token distribution and is not the calibration KL above.

## Frozen Pins

### Machine and software

| Item | Value |
|---|---|
| Host | Linux `7.0.0-27-generic`, x86_64, glibc 2.43 |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, 32,607 MiB |
| Driver | `595.71.05` |
| Python | `3.12.13` |
| PyTorch / CUDA | `2.11.0+cu130` / CUDA 13.0 |
| vLLM / Triton | `0.23.0` / `3.6.0` |
| Transformers / huggingface-hub | `5.12.1` / `1.21.0` |
| Qwen Code / SWE-bench | `0.19.4` / `4.1.0` |
| Replay runtime | eager, text-only, FP32 transport, BF16 readout |
| Replay cache | FP8 E4M3 KV cache, prefix caching on, Mamba block 1,024 |
| Replay limits | model length 16,384; prefill budget 4,096; GPU fraction 0.78 |

### Target model

| Item | Value |
|---|---|
| Repository | `nvidia/Qwen3.6-27B-NVFP4` |
| Revision | `0893e1606ff3d5f97a441f405d5fc541a6bdf404` |
| Quantization | ModelOpt mixed NVFP4/FP8 |
| Main model | width 5,120; 64 blocks; 48 GDN and 16 full-attention |
| `config.json` | `c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338` |
| `hf_quant_config.json` | `fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1` |
| `model.safetensors.index.json` | `7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2` |

The strict runner also hashes all model shards:

| File | Bytes | SHA-256 |
|---|---:|---|
| `model-00001-of-00003.safetensors` | 9,965,652,512 | `b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d` |
| `model-00002-of-00003.safetensors` | 9,985,757,032 | `06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d` |
| `model-00003-of-00003.safetensors` | 1,970,287,640 | `e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845` |

### Lens artifacts

| Item | Native NVIDIA lens | Public control lens |
|---|---|---|
| Kind | exact deployed NVFP4/FP8 forward plus identity-STE surrogate backward | public Qwen3.6-27B lens; fit-time precision and quantization unpublished |
| Fit prompts | 10 | 1,000 |
| Stored matrices | 63 FP32 `[5120,5120]` | 63 FP16 `[5120,5120]` |
| Artifact bytes | 6,606,046,478 | 3,303,032,772 |
| Artifact SHA-256 | `82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057` | `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1` |
| Revision / run | run `20e4bc8c-9fed-4513-b548-9727f9686222` | HF revision `a4114d7752d11eb546e6cf372213d7e75526d3a1` |

Additional native bindings are:

| Record | SHA-256 |
|---|---|
| Final-mean provenance | `289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601` |
| Completed fit state | `f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6` |
| Layer aggregate | `a4c2adc7be15232db0e5a8840a6442248caa80a363c0c5239a1ee248f36fb3b4` |
| Committed prompts | `a1690ab9e88cff53a2eba407195ced52e6908208fedffed68819ee47c1a888c1` |
| Fit contract | `7944ea163b548edc3372fa67242fbbcfbe0a5abbe95c04ce4a378107ebe03dd0` |

The native `n=10` lens was fitted on frozen 128-token WikiText prompts. Its
backward is an explicitly declared identity straight-through estimator for
quantized operations, not the literal derivative of FP4/FP8 rounding. See
[`JLENS_NVFP4_STE_EXPERIMENT.md`](JLENS_NVFP4_STE_EXPERIMENT.md) for the full
fit contract.

### Episode and prompt bindings

| Episode item | Value |
|---|---|
| SWE dataset | `princeton-nlp/SWE-bench_Verified` |
| Dataset revision | `c104f840cc67f8b6eec6f759ebc8b2693d585d4a` |
| Task / base commit | `sympy__sympy-13480` / `f57fe3f4b3f2cab225749e1b3b38ae1bf80b62f0` |
| Task image | `swebench/sweb.eval.x86_64.sympy_1776_sympy-13480` |
| Task image digest | `sha256:3a985bfd2f3430af337ad8f34793964d1f8845485fcee129c3f166dfba6c5e43` |

| Record | SHA-256 |
|---|---|
| Certified episode record | `1da4939bd32393423075b2812c8d00d28506dcd5598948caa39e6258bd8332da` |
| Scored patch | `98e80f91393e76fc6323eeb0a7582aa89f1db717932ea5742eed467d692d024d` |
| Frozen one-task dataset | `e8d93ec886c5d367e412c2ec04c7a7696af3c821ab9fb3f0e03261a7743445e2` |
| Submitted predictions | `7ea5811e0765855a1a817a06e975893368cc4304e24d093cfb4cc7d857fab4bb` |
| Official harness summary | `6f0e1c2ce449d411df0f2f8ffaf2680f1178ced057cee156c989a53732b0f7f8` |
| Qwen chat template | `c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da` |
| Qwen trace | `3e39701da94a3f590e62efc9e67aa22220e155b385cceb27cbecc4d9a56d632a` |
| Proxy usage records | `60757ba7d953c5378392bf2a4d3fe8812ef739645a7977a9eeff311f0d1313e9` |
| Prompt source-contract ID | `72da18d1cead29ce7c4fe2627608040599c5f43c9e2b589855f89efa39afe038` |
| Prompt provenance JSON | `8426064cf82c48961433987e04b43cbc9f54375feb8e326c60496215835dfc1e` |
| Nine-stage prompt bundle | `5044dff697ac55971c090ab8fa2b744df912d5acd784e9641d7b800ba9bedf1c` |
| Five-step candidate bundle | `1f8a303c3b02688c0511506ed4d3d80a27bb5708d81ddc8cf106a3ffd93d91ae` |

The renderer uses the checkpoint's `Qwen2Tokenizer` at the same pinned model
revision. Its `tokenizer.json`, `tokenizer_config.json`, and `vocab.json`
hashes are respectively
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`,
`5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b`,
and `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003`.

The candidate strings do not share a BPE length. `cothm` tokenizes as
`[981, 337, 76]` (`co`, `th`, `m`), while `cotm` tokenizes as `[62317, 76]`
(`cot`, `m`). The primary comparison below therefore uses only the alternative
first token at the identical unmodified context.

## Middle-Layer Findings

### Stage-boundary decoded readouts

Layers 31 and 32 straddle the 64-block model's geometric midpoint. The most
coherent natural-language stage readouts appeared slightly later at the
adjacent full-attention/GDN pair 39 and 40. The table shows selected tokens
from the recorded native Jacobian-Lens top 10 at the last prompt position; it
does not paraphrase an unobserved hidden sentence.

| Request state | Original sampled continuation | Layer 39 decoded terms | Layer 40 decoded terms |
|---|---|---|---|
| 3, source inspected | `I can see the bug clearly` | `obvious`, `clearly`, `obviously` | `obvious`, `clearly`, `obviously` |
| 4, failure reproduced | `Bug confirmed` | `confirming`, `confirm`, `confirmed` | `confirm`, `confirming`, `confirmation` |
| 6, smoke fix verified | `The fix works` | `修复` (repair), `好消息` (good news), `confirming` | `修复` (repair), `确认` (confirm), `好消息` (good news) |
| 7, broader values verified | `All values work without error` | `everything`, `confirming`, `nicely`, `confirm` | `confirming`, `everything`, `confirm`, `successfully` |
| 8, pytest unavailable | `pytest is not installed` | `缺乏` (lack), `lack`, `unavailable`, `unfortunately` | `缺乏` (lack), `lack`, `couldn`, `wasn` |

These are semantic category matches, not exact next-token recovery at the
middle layers. For example, at request 3 the native J-lens rank of the actual
first sampled token `I` was 8,185 at layer 39 and 2,134 at layer 40, while the
captured block-63 final readout ranked `I` first. That separation is why the
report describes the intermediate tokens as decoded readouts rather than the
model "saying" a literal sentence.

### Same-context def-use probe

The stronger task-specific probe preserves the certified request-3 context
and its original reasoning up to this exact prefix:

```text
I can see the bug clearly. On line 590, it checks `if cotm is S.ComplexInfinity:` but the variable is actually `
```

At that slot it scores `co` (the first token of correct `cothm`) against `cot`
(the first token of buggy `cotm`). The score is the recorded BF16 output logit
after native Jacobian transport. Positive margin means the correct spelling's
first token is preferred.

| Source readout | Block type | Native `co` / `cot` rank | Native margin | Public margin |
|---:|---|---:|---:|---:|
| 24 | GDN | 34,821 / 25,205 | -0.3125 | +1.3750 |
| 27 | full attention | 62,695 / 86,007 | +0.4688 | +2.0938 |
| 28 | GDN | 30,188 / 41,766 | +0.3984 | +0.4453 |
| 31 | full attention | 10,551 / 42,115 | +2.0664 | +2.7266 |
| 32 | GDN | 8,095 / 36,317 | +1.9922 | +3.1797 |
| 35 | full attention | 2,320 / 40,519 | +3.0391 | +1.2500 |
| 36 | GDN | 3,548 / 74,545 | +3.2578 | +1.0000 |
| 39 | full attention | 19,352 / 83,254 | +1.7500 | +0.7031 |
| 40 | GDN | 17,943 / 114,183 | +2.3984 | +1.3438 |
| 62 | GDN | 1 / 14 | +13.5625 | +14.2500 |
| Captured block 63 final | final-model control | 1 / 2 | +13.5000 | +13.5000 |

Thus the correct def-use spelling already has a positive same-context
first-token margin at the exact midpoint pair, remains positive at layers
35/36 and 39/40, and becomes decisive near the output. Across the fixed
middle slice `24,27,28,31,32,35,36,39,40`, the native lens favored the
correct token at eight of nine layers; layer 24 was the exception at
`-0.3125`. The independently loaded public lens favored the correct token at
all nine. The public and native candidate reports have byte-identical residual
capture manifests, so those lens differences are evaluated on the same
recorded target-model states. On this fixed exploratory nine-layer reporting
slice, the
mean margin was `+1.6732` for native and `+1.5686` for public, and their signs
agreed at eight of nine layers.

The ordinary logit-lens baseline favored `co` at all nine fixed middle layers
with mean margin `+3.2440`, larger than either J-lens mean. The candidate
preference is therefore not uniquely revealed by Jacobian transport; the
J-lens result describes how that already-visible contrast changes under the
fitted residual-to-final transport.

This preference is not asserted across the complete depth. Native was
positive at 44/63 source layers and public at 33/63. The evidence claim is
restricted to the fixed middle slice and the late-output controls shown in the
table, rather than selecting a post-hoc all-depth emergence point.

Teacher-forced full-sequence log probabilities are secondary because the
correct spelling consumes three BPE tokens and the incorrect spelling consumes
two. For completeness, at layer 62 the summed native J-lens log probabilities
were `-0.395038` for `cothm` and `-15.777278` for `cotm`; at the captured final
readout they were `-0.000033` and `-13.513421`. Those totals are consistent
with the late correct preference, but they are not the primary middle-layer
comparison and should not be compared as if token counts were equal.

### Public-lens control

The nine-context native/public comparator paired all nine token boundaries and
all 63 source layers, 567 observations total. It proved the ordinary logit
lens was identical between independent captures. Native versus public
Jacobian-Lens target-rank Spearman was `0.823959` over all observations;
top-1 agreement was `0.426808` and mean top-5 overlap was `0.480071`. These
figures show that the two lens artifacts are related but not interchangeable.
They are descriptive because the public artifact's fit-time precision and
quantization are unpublished and both stage adapter certificates failed in
the same five maximum-error checks.

Restricted to the 81 fixed exploratory middle-layer observations, native/public
top-1 agreement was `40/81` (`0.493827`), mean top-5 overlap was `0.533333`,
and exact top-5-set agreement was `2/81` (`0.024691`). Macro layer-wise
target-rank Spearman was `0.6630`, and target-log-probability mean absolute
difference was `2.3017`. This is moderate agreement, not equivalence.

## Method-Aligned Evaluations

### Pinned Anthropic multihop control

The external control uses Anthropic's pinned
[`lens-eval-multihop.json`](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/data/evaluations/lens-eval-multihop.json).
Its source SHA-256 is
`50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416`.
It contains 93 items and 103 intermediate occurrences. Ninety-four
occurrences have at least one exact single-token form under the pinned Qwen
tokenizer; the other nine are retained as misses rather than silently dropped.
For each occurrence, the metric takes the minimum exact vocabulary rank over
eligible forms and the primary fixed middle band, layers 24 through 47. Scores
are macro-averaged over upstream items. The paired percentile bootstrap
resamples the 93 items 20,000 times with seed 36027.

| Lens artifact | J-lens AUC | Logit-lens AUC | AUC gain, 95% CI | J-lens pass-at-10 | Logit pass-at-10 | Pass-at-10 gain, 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Public `n=1000` | 0.623738 | 0.470733 | +0.153005 `[0.114900, 0.190107]` | 0.290323 | 0.134409 | +0.155914 `[0.064516, 0.247312]` |
| Native NVFP4/FP8-STE `n=10` | 0.619019 | 0.470733 | +0.148286 `[0.109855, 0.186264]` | 0.279570 | 0.134409 | +0.145161 `[0.053763, 0.236559]` |

Public J-lens pass-at-1/5/50/100 was
`0.129032/0.241935/0.460573/0.530466`; native was
`0.134409/0.225806/0.465950/0.546595`. Over all 63 layers, a declared
secondary view, AUC gains remained positive but smaller: public `+0.030925`
with CI `[0.012334, 0.049868]` and native `+0.028842` with CI
`[0.010716, 0.047271]`; the all-layer pass-at-10 gain CIs crossed zero. The
fixed band is the primary comparison, not a post-hoc best-layer selection.

This result supports method reproduction but does not create a strict adapter
pass. Both raw reports retain `status: failed`. Greedy final top-1 and final
norm tolerance passed for 90/93 prompts, final top-5 parity for 85/93, and the
full final-logit tolerance for 72/93. The analyzer verifies paired prompt IDs,
tokens, residual manifests, and all logit-lens fields between public and native
runs. Thus the intermediate comparison is paired on identical residuals, while
imperfect final-output parity limits causal claims. The compact result is
[`validation/jlens-upstream-multihop-control-analysis-2026-07-17.json`](../validation/jlens-upstream-multihop-control-analysis-2026-07-17.json).
The local manifest/public/native/analysis SHA-256 values are respectively
`ca1bc7aa53070cb398d60517c1d1f16e10b17be9c5bcab00dce775f59738791d`,
`e5bbd06335d11b6dfa19bf884d396d3b3b628d84a235a1ca7ddf17cea0cd572e`,
`7786e37a2a39eda351e1edf13a2b4e4bcbba634e524c3033f01c450ca6dd00ae`,
and `ea07313ff6f788d51e90cac00a987881264a69c2c93cc6b67eff64f937e856d0`.

### SWE intermediate-concept probe

The episode-specific probe adapts the same evaluation convention without
scoring the accepted next token. Before any lens output was inspected, it
froze ten exact trajectory states, 17 semantic intermediates with verified
single-token forms, and contiguous layers 16 through 47. The selected states
are request 1 offset 0; request 2 offset 0; request 3 offset 32; requests 4
through 8 offset 0; and request 9 offsets 0 and 60. They are selected from the
293-state teacher-forced trajectory bundle by task text, trace, shell command,
and tool-result evidence only.

The frozen config SHA-256 is
`2cae42b3b3f559209a81ae80d55800ff215be0786a28865b5f95d0a16fdba1cc`.
The materialized prompt bundle SHA-256 is
`fc3293d64323cb25ea4ae1626e5a3508c983bec18a25ff2241ddf74d3a86e9fc`;
its summary file SHA-256 is
`4438fe2326b73262a9cc5a6064c35e9f8d99b5762156d57c007cd3a61d68d427`.
The public and native raw report SHA-256 values are
`16a8c781db6c3dea9dd6602a1e6a113d1a7b29b5ada1d1668168a0ff0d9290b7`
and `a307b236c259bc58703ba1449ecfe404f6387376a1db10d3f675b0c1c21b5068`.

| Method | Pass@1 | Pass@5 | Pass@10 | Pass@50 | Pass@100 | Pass@1000 | Normalized log-rank AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public J-lens | 0.25 | 0.35 | 0.35 | 0.55 | 0.65 | 0.85 | 0.715434 |
| Native J-lens | 0.10 | 0.25 | 0.25 | 0.50 | 0.70 | 0.85 | 0.691667 |
| Logit lens | 0.00 | 0.30 | 0.40 | 0.50 | 0.55 | 0.95 | 0.691825 |

These results are exploratory. With 20,000 deterministic paired item-bootstrap
draws, public AUC gain over logit lens is `+0.023609`, 95% CI
`[-0.073627, 0.102261]` (7/10 item gains positive), and native gain is
`-0.000157`, CI `[-0.096932, 0.085179]` (6/10 positive). The pre-output
nonbaseline subset has eight items: public gain `+0.071442`, CI
`[-0.003289, 0.122190]`, and native gain `+0.045482`, CI
`[-0.036767, 0.118418]`. These intervals cross zero, and the adaptation has no
preregistered claims gate.

The leakage labels are material. The seven post-tool boundary items mostly
test concepts already explicit in earlier tool output. Their public/native AUC
gains are `+0.106254`, CI `[0.083356, 0.129165]` (7/7 positive), and
`+0.078410`, CI `[0.023077, 0.134710]` (6/7 positive), but same-task
transcript-explicit rows are not independent-task evidence and can measure
retention rather than novel inference. The initial task-explicit item is worse
under both J-lenses (`0.284740`/`0.287085`) than logit lens (`0.596228`). The
focused-test state is only an implicit tool-success case, and the final summary
is a teacher-forced explicit positive control.

The exact pre-identifier state remains the clearest weakness. For its
`defined_identifier` and `typographical_error` concepts, public J-lens
band-minimum ranks are `[191,525]`, native ranks `[175,787]`, and logit-lens
ranks `[3,463]`. The older lexical `co` versus `cot` probe is consistent:
native/public favor `co` at 8/9 and 9/9 middle layers, but logit lens favors it
at 9/9 with the larger mean margin (`+3.244` versus `+1.673`/`+1.569`), and
both alternatives remain outside the J-lens middle-layer top 10. The result
does not establish latent reasoning or SWE-bench generalization.

All ten public and native probe rows pass model-greedy final top-1, final top-5,
and final-norm checks. Only 5/10 pass the strict full-final-logit tolerance, so
both raw reports retain `status: failed`. The recorded accepted target matches
greedy generation on 9/10 rows and is not scored as an intermediate. The
compact analysis is
[`validation/jlens-swe-qwen-code-intermediate-analysis-2026-07-17.json`](../validation/jlens-swe-qwen-code-intermediate-analysis-2026-07-17.json).
It is 263,279 bytes with SHA-256
`29f7cb2f1ffe7948f7836c49db46020864fd4e2876ec060ca03637edd5e034db`.

### Output-calibration diagnostics

The dense trajectory evaluates all 293 teacher-forced states, but asks the
non-objective question of how closely each readout predicts the captured final
next-token distribution. Over its fixed middle slice, the public J-lens KL is
`12.224606` versus logit-lens `8.770976`; native J-lens KL is `13.094680`.
Accepted-target log-probability gains relative to logit lens are public
`-3.521563` with request-bootstrap 95% CI `[-3.967421, -3.080830]` and native
`-4.400345` with CI `[-4.909200, -3.965547]`. These intervals quantify
within-episode variation over nine correlated requests, not task-level
uncertainty. Both reports have only 192/293 strictly eligible rows. The poor
output calibration is consistent with the transport objective and Anthropic's
reported behavior; it is not a failed J-lens reproduction. The compact record
is [`validation/jlens-swe-qwen-code-trajectory-calibration-2026-07-17.json`](../validation/jlens-swe-qwen-code-trajectory-calibration-2026-07-17.json).

The older ten-state semantic final-margin analysis is also calibration only.
Public and native J-lens margins are positive on 8/10 states, versus 10/10 for
logit lens; each J-lens is closer to the captured final margin on only 4/10.
Its former final-margin error reductions are negative (`-18.06%` public and
`-24.08%` native) and are not quality gates. Native/public J-lens signs still
agree on all ten states. The compact record is
[`validation/jlens-swe-qwen-code-semantic-calibration-2026-07-17.json`](../validation/jlens-swe-qwen-code-semantic-calibration-2026-07-17.json).

## Next Experiment

1. Freeze method-aligned concept probes across multiple independent SWE tasks
   before inspecting lens outputs. Separate novel inferred concepts from
   task-explicit, tool-explicit, and teacher-forced controls; keep fixed layer
   bands, exact single-token forms, item/task macro weighting, and held-out
   preregistered pass-at-k/AUC claims.
2. If that retrieval gate is met, perform causal ablation and cross-state swap
   interventions on selected J-space concept directions. Measure downstream
   tool/action changes and intervention-induced output KL; do not reinterpret
   readout-to-final calibration KL as a causal result.
3. Refit the native lens with at least 100 prompts only if it systematically
   trails the public lens on the pinned external multihop control and the
   frozen multi-task SWE probes. The current native multihop result nearly
   matches public, while the one-task SWE result is too weak for a refit
   decision, so no refit is justified yet.

## Adapter-Certificate Detail

The native and public stage runs captured identical residual manifests for
each prompt and produced identical ordinary logit-lens outputs. Both reports
failed only these final-logit maximum-absolute checks:

| Request | Observed | Limit |
|---:|---:|---:|
| 2 | 0.125 | 0.0625 |
| 5 | 0.125 | 0.0625 |
| 6 | 0.125 | 0.0625 |
| 7 | 0.125 | 0.0625 |
| 8 | 0.125 | 0.0625 |

All nine stage top-1 identities, top-5 prefixes, final-norm maximum/RMS
thresholds, and final-logit RMS thresholds passed. No tolerance was widened
after observing the values.

The five-step native candidate report also preserves its own failed strict
status. All candidate final top-1 and top-5 checks passed, but
`cothm` step 2 had final-logit maximum error `0.125 > 0.0625`, and `cothm`
step 3 had final-logit RMS error `0.0125369 > 0.01`. Candidate margins remain
descriptive measurements rather than a passing adapter certificate. The
public candidate control failed on the same two checks with the same values;
its residual manifests exactly match the native candidate run. Each candidate
report passed the combined strict adapter gate on three of five steps, while
final-norm, top-5-prefix, and greedy top-1 checks passed on all five.

## Reproduction

### Quick one-load timeline

One Qwen Code request is one model completion inside the agentic loop. Qwen
receives the conversation accumulated so far, emits an assistant completion
(often a tool call), the tool runs, and the enlarged conversation becomes the
next request. The nine requests here are nine completion boundaries within one
SWE task, not nine benchmark tasks.

The quickest replay reuses the public `n=1000` lens and processes all nine
exact token contexts in one model load:

```bash
scripts/quick_swe_jlens.py
```

It reads layers `24,31,32,39,40,62` at the final prompt-token boundary and
writes both the raw runner report and a compact timeline under
`.cache/swe_jlens_quick/`. Select a subset with `--requests 3` or
`--requests 3-6`. `--dry-run` materializes the exact token bundle and prints
the pinned GPU command without loading the model. The compact output names two
different sample sizes: `task_request_count` is the number of evaluated agent
completions, while `lens_fit_prompt_count` is the unrelated background corpus
used once to estimate the lens (`1000` for the public default). The command
returns success after a structurally valid replay even when the report's
strict adapter certificate is `failed`; `--require-strict` instead propagates
that failure as exit status 1.

The final verified reference invocation processed all nine requests in 38.60
seconds wall time on the RTX 5090. Its measured runner lifecycle was 35.381
seconds, including an 8.613-second model load, and it produced all nine
timeline rows. The report retained `failed` strict status because requests 2,
5, 6, 7, and 8 exceeded only the maximum-logit tolerance; all nine passed the
final top-1, top-5-prefix, final-norm RMS, and final-logit RMS checks.

### 1. Set up the pinned model and artifacts

```bash
git clone https://github.com/MaCoredroid/lumo_J_lens.git
cd lumo_J_lens

scripts/setup.sh --download-model
.venv-vllm/bin/python scripts/download_jlens.py
scripts/check.sh
```

The public control artifact is downloaded by `scripts/download_jlens.py`.
The 6.6 GB native artifact is not stored in Git. Produce and verify it using
the workflow in
[`JLENS_NVFP4_STE_EXPERIMENT.md`](JLENS_NVFP4_STE_EXPERIMENT.md), then require
these exact local files:

```text
.cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt
.cache/nvfp4_ste_fit/final-mean/metadata.json
.cache/nvfp4_ste_fit/state.json
```

### 2A. Materialize prompts from the certified run

This stronger path is available to the owner of the original certified
`runs/` directory. The materializer rehashes all nine raw request bodies,
usage records, Qwen trace, template, tokenizer files, and model metadata; it
normalizes stringified JSON tool arguments and requires every rendered prompt
token count to match the certified usage record.

```bash
.venv-vllm/bin/python scripts/materialize_swe_jlens_prompts.py \
  --run-dir runs/publication_certified_v2_20260715 \
  --template configs/qwen3-openai-codex.jinja \
  --output-dir .cache/swe_jlens_prompts

sha256sum \
  .cache/swe_jlens_prompts/swe_jlens_stage_prompts.json \
  .cache/swe_jlens_prompts/swe_jlens_candidate_prompts.json \
  .cache/swe_jlens_prompts/swe_jlens_prompt_provenance.json
```

Expected hashes, in order, are `5044dff6...f1c`, `1f8a303c...91ae`, and
`8426064c...dfc1e`, with full values in the binding table above.

### 2B. Extract exact token inputs from a public checkout

The original `runs/` directory is intentionally untracked. The tracked stage
and candidate reports embed the exact `prompt_token_ids` and metadata needed
for another target-model replay. A public checkout can deterministically make
minimal runner inputs without the private raw request dumps:

```bash
mkdir -p .cache/swe_jlens_prompts

jq -S '[.experiments[] | {
  id: .id,
  token_ids: .prompt_token_ids,
  metadata: .metadata
}]' \
  validation/jlens-swe-qwen-code-native-nvfp4-ste-2026-07-17.json \
  > .cache/swe_jlens_prompts/swe_jlens_stage_prompts.from-report.json

jq -S '[.experiments[] | {
  id: .id,
  token_ids: .prompt_token_ids,
  target_token_id: .target_token_id_override,
  metadata: .metadata
}]' \
  validation/jlens-swe-qwen-code-candidate-probe-2026-07-17.json \
  > .cache/swe_jlens_prompts/swe_jlens_candidate_prompts.from-report.json
```

Set `STAGE_PROMPTS` and `CANDIDATE_PROMPTS` to either the owner-materialized
files or the `.from-report.json` files. The public extraction reproduces the
exact model inputs, candidate targets, and metadata, but it cannot independently
rehash absent raw request/trace files. The extracted files also omit redundant
decoded `text`, so they are not expected to have the owner bundle file hashes.

```bash
STAGE_PROMPTS=.cache/swe_jlens_prompts/swe_jlens_stage_prompts.from-report.json
CANDIDATE_PROMPTS=.cache/swe_jlens_prompts/swe_jlens_candidate_prompts.from-report.json
mkdir -p .cache/swe_jlens_results
```

### 3. Pass the longest-context preflight

The tracked preflight isolates the 15,678-token request 9 and layers 31/32.
It passed the complete strict adapter certificate, including exact final
top-1, matching top-5, and all max/RMS thresholds. This proves an internally
coherent final-state capture and adapter reconstruction under the isolated
request-9 schedule. It does not prove bit-equivalence to the later warmed,
sequential nine-request schedule.

```bash
jq '[.[-1]]' "$STAGE_PROMPTS" \
  > .cache/swe_jlens_prompts/swe_jlens_longest_prompt.from-report.json

scripts/run_jlens_nvfp4.sh \
  --lens-kind public \
  --prompts-file .cache/swe_jlens_prompts/swe_jlens_longest_prompt.from-report.json \
  --layers 31,32 \
  --positions=-1 \
  --top-k 10 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --mamba-block-size 1024 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --stream-final-only \
  --gpu-memory-utilization 0.78 \
  --output .cache/swe_jlens_results/longest-preflight.json
```

Expected structural result: prompt ID `swe-sympy-13480-request-09`, 15,678
prompt tokens, captured position 15,677, layers 31/32, generated token `The`,
`status: passed`, and both top-1 and adapter assertions true.

The preflight is an independent execution of exact request 9. Its residual
manifest hash is `2c8119e4...3ef`, while the later public all-layer request-9
capture is `a78e0255...076`; the analysis binds the exact prompt, metadata,
position, token target, lens, model, and runtime but does not claim those two
independent numerical captures are byte-identical.

### 4. Replay all stage boundaries with the native lens

```bash
scripts/run_jlens_nvfp4.sh \
  --lens-kind nvfp4-ste \
  --lens-path .cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt \
  --lens-sha256 82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057 \
  --lens-provenance .cache/nvfp4_ste_fit/final-mean/metadata.json \
  --lens-state .cache/nvfp4_ste_fit/state.json \
  --lens-state-sha256 f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6 \
  --prompts-file "$STAGE_PROMPTS" \
  --layers all \
  --positions=-1 \
  --top-k 10 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --mamba-block-size 1024 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --stream-final-only \
  --gpu-memory-utilization 0.78 \
  --output .cache/swe_jlens_results/native-stage.json
```

This exact configuration writes a complete report and then exits 1 because
the strict adapter status is failed as documented above. Do not discard the
report or convert that exit to a claimed pass.

### 5. Replay the same stages with the public control

```bash
scripts/run_jlens_nvfp4.sh \
  --lens-kind public \
  --prompts-file "$STAGE_PROMPTS" \
  --layers all \
  --positions=-1 \
  --top-k 10 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --mamba-block-size 1024 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --stream-final-only \
  --gpu-memory-utilization 0.78 \
  --output .cache/swe_jlens_results/public-stage.json
```

This also writes the report and exits 1 for the same five strict maximum-error
bins.

### 6. Pair the independent stage reports

```bash
.venv-vllm/bin/python scripts/compare_jlens_nvfp4_reports.py \
  --native-report .cache/swe_jlens_results/native-stage.json \
  --public-report .cache/swe_jlens_results/public-stage.json \
  --output .cache/swe_jlens_results/native-vs-public-stage.json
```

The comparator requires identical model, host, runtime, prompt-token,
position, residual-manifest, and ordinary-logit-lens identities before it
reports native/public lens metrics.

### 7. Replay the teacher-forced candidate steps

```bash
scripts/run_jlens_nvfp4.sh \
  --lens-kind nvfp4-ste \
  --lens-path .cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt \
  --lens-sha256 82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057 \
  --lens-provenance .cache/nvfp4_ste_fit/final-mean/metadata.json \
  --lens-state .cache/nvfp4_ste_fit/state.json \
  --lens-state-sha256 f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6 \
  --prompts-file "$CANDIDATE_PROMPTS" \
  --layers all \
  --positions=-1 \
  --top-k 10 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --mamba-block-size 1024 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --stream-final-only \
  --gpu-memory-utilization 0.78 \
  --output .cache/swe_jlens_results/native-candidate.json
```

A public-lens candidate control uses the identical target-model inputs and
runtime:

```bash
scripts/run_jlens_nvfp4.sh \
  --lens-kind public \
  --prompts-file "$CANDIDATE_PROMPTS" \
  --layers all \
  --positions=-1 \
  --top-k 10 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --mamba-block-size 1024 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --stream-final-only \
  --gpu-memory-utilization 0.78 \
  --output .cache/swe_jlens_results/public-candidate.json
```

### 8. Verify the tracked compact analysis

```bash
.venv-vllm/bin/python scripts/analyze_swe_jlens_report.py --check
sha256sum --check validation/jlens-swe-qwen-code-evidence-2026-07-17.sha256
sha256sum --check validation/jlens-swe-qwen-code-source-manifest.sha256
```

The analyzer reconstructs the prompt-bundle hashes from the raw reports,
recomputes the paired stage comparison, validates native/public candidate
roles and runtime identity, binds the preflight to request 9, and checks the
official `1/1` record and patch before accepting the compact findings.

### 9. Reproduce the pinned Anthropic multihop control

Materialize the exact upstream commit and verify its source hash and tokenizer
forms:

```bash
.venv-vllm/bin/python scripts/materialize_jlens_upstream_multihop.py \
  --allow-download \
  --output-dir .cache/jlens_upstream_multihop
```

Run the public artifact on all layers. The command writes the report and then
returns 1 because the strict final-state certificate is failed; the shell guard
accepts only that documented exit code and does not change the report status:

```bash
scripts/run_jlens_nvfp4.sh \
  --lens-kind public \
  --prompts-file .cache/jlens_upstream_multihop/prompts.json \
  --layers all --positions=-1 --top-k 10 \
  --max-model-len 256 --max-num-batched-tokens 256 \
  --gpu-memory-utilization 0.82 \
  --output .cache/jlens_upstream_multihop/public-report.json \
  || test $? -eq 1
```

Run the native artifact on the identical prompts and runtime:

```bash
scripts/run_jlens_nvfp4.sh \
  --lens-kind nvfp4-ste \
  --lens-path .cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt \
  --lens-sha256 82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057 \
  --lens-provenance .cache/nvfp4_ste_fit/final-mean/metadata.json \
  --lens-state .cache/nvfp4_ste_fit/state.json \
  --lens-state-sha256 f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6 \
  --prompts-file .cache/jlens_upstream_multihop/prompts.json \
  --layers all --positions=-1 --top-k 10 \
  --max-model-len 256 --max-num-batched-tokens 256 \
  --gpu-memory-utilization 0.82 \
  --output .cache/jlens_upstream_multihop/native-report.json \
  || test $? -eq 1
```

Validate exact ranks, paired residual identity, item-macro metrics, and the
deterministic bootstrap:

```bash
.venv-vllm/bin/python scripts/analyze_jlens_upstream_multihop.py \
  --manifest .cache/jlens_upstream_multihop/manifest.json \
  --report .cache/jlens_upstream_multihop/public-report.json \
  --native-report .cache/jlens_upstream_multihop/native-report.json \
  --output .cache/jlens_upstream_multihop/analysis.json
```

The recorded public/native runner lifecycles were 148.583/135.831 seconds,
including 15.108/10.600 seconds to load the model. Both were eager,
MTP-disabled, prefix-cache-disabled target-model replays.

### 10. Reproduce the SWE intermediate-concept probe

The owner path first materializes the private 293-state trajectory from the
certified raw run. This verifies every request/trace hash and reconstructs the
teacher-forced completion boundaries:

```bash
.venv-vllm/bin/python scripts/materialize_swe_jlens_trajectory.py \
  --run-dir runs/publication_certified_v2_20260715 \
  --template configs/qwen3-openai-codex.jinja \
  --output .cache/swe_jlens_trajectory/swe_jlens_trajectory_prompts.json \
  --summary-output .cache/swe_jlens_trajectory/swe_jlens_trajectory_prompts.summary.json
```

Select the ten predeclared points from that trajectory. The materializer checks
the config and trajectory hashes, request bindings, exact points, Qwen
single-token round trips, and the rule that the accepted target token is not
scored:

```bash
.venv-vllm/bin/python scripts/materialize_swe_intermediate_probes.py \
  --config configs/swe_intermediate_concept_probes.json \
  --trajectory-prompts .cache/swe_jlens_trajectory/swe_jlens_trajectory_prompts.json \
  --output .cache/swe_jlens_intermediate/prompts.json
```

The private 45 MB trajectory is intentionally not tracked. A public checkout
instead uses the exact ten-input bundle and summary committed for replay. These
files have the same hashes as the owner-materialized outputs:

```bash
INTERMEDIATE_PROMPTS=validation/jlens-swe-qwen-code-intermediate-prompts-2026-07-17.json
INTERMEDIATE_SUMMARY=validation/jlens-swe-qwen-code-intermediate-prompts-summary-2026-07-17.json

test "$(sha256sum "$INTERMEDIATE_PROMPTS" | cut -d' ' -f1)" = \
  fc3293d64323cb25ea4ae1626e5a3508c983bec18a25ff2241ddf74d3a86e9fc
test "$(sha256sum "$INTERMEDIATE_SUMMARY" | cut -d' ' -f1)" = \
  4438fe2326b73262a9cc5a6064c35e9f8d99b5762156d57c007cd3a61d68d427
```

For the owner path, point the same variables at the `.cache` outputs instead.
Extract and verify the fixed 58-token scoring vocabulary before either replay:

```bash
SCORE_TOKEN_IDS=$(jq -r '.scored_token_ids | join(",")' "$INTERMEDIATE_SUMMARY")
test "$(jq '.scored_token_ids | length' "$INTERMEDIATE_SUMMARY")" -eq 58
```

Run the public lens on the frozen middle band:

```bash
scripts/run_jlens_nvfp4.sh \
  --lens-kind public \
  --prompts-file "$INTERMEDIATE_PROMPTS" \
  --score-token-ids "$SCORE_TOKEN_IDS" \
  --layers "$(seq -s, 16 47)" --positions=-1 --top-k 10 \
  --max-model-len 16384 --max-num-batched-tokens 4096 \
  --mamba-block-size 1024 --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 --stream-final-only \
  --gpu-memory-utilization 0.78 \
  --output .cache/swe_jlens_intermediate/public-report.json \
  || test $? -eq 1
```

Run the native lens on the identical prompts and residual schedule:

```bash
scripts/run_jlens_nvfp4.sh \
  --lens-kind nvfp4-ste \
  --lens-path .cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt \
  --lens-sha256 82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057 \
  --lens-provenance .cache/nvfp4_ste_fit/final-mean/metadata.json \
  --lens-state .cache/nvfp4_ste_fit/state.json \
  --lens-state-sha256 f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6 \
  --prompts-file "$INTERMEDIATE_PROMPTS" \
  --score-token-ids "$SCORE_TOKEN_IDS" \
  --layers "$(seq -s, 16 47)" --positions=-1 --top-k 10 \
  --max-model-len 16384 --max-num-batched-tokens 4096 \
  --mamba-block-size 1024 --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 --stream-final-only \
  --gpu-memory-utilization 0.78 \
  --output .cache/swe_jlens_intermediate/native-report.json \
  || test $? -eq 1
```

Analyze exact form/layer minima and paired residual identity:

```bash
.venv-vllm/bin/python scripts/analyze_swe_intermediate_probes.py \
  --config configs/swe_intermediate_concept_probes.json \
  --summary "$INTERMEDIATE_SUMMARY" \
  --prompts "$INTERMEDIATE_PROMPTS" \
  --report .cache/swe_jlens_intermediate/public-report.json \
  --native-report .cache/swe_jlens_intermediate/native-report.json \
  --output .cache/swe_jlens_intermediate/analysis.json
```

The recorded public/native runner lifecycles were 50.058/73.291 seconds,
including 12.711/11.453 seconds to load the model. Both were eager and
MTP-disabled. They used prefix caching only for replay efficiency; this does
not reproduce or claim access to the original accepted MTP draft-token trace.

## Recorded Evidence

| File | Bytes | SHA-256 |
|---|---:|---|
| `validation/jlens-swe-qwen-code-native-nvfp4-ste-2026-07-17.json` | 5,769,269 | `94f3963be698b0e6e86d3878ba69efa53a94059e5520f292a97f344d6bed6fab` |
| `validation/jlens-swe-qwen-code-public-2026-07-17.json` | 5,771,118 | `b3a4e43bf379dca4a630002644e824956e1bf6ca5b3533480ff80bd0dc2e2a97` |
| `validation/jlens-swe-qwen-code-native-vs-public-2026-07-17.json` | 1,024,142 | `495b7e043d83b0ce47f81687c763bdee3306b94f897aeea88f50be95d318fe93` |
| `validation/jlens-swe-qwen-code-candidate-probe-2026-07-17.json` | 3,019,146 | `6273844f62b19b0060eff958b920d54cf0d969dea6cb85e999bcb72e14d98edc` |
| `validation/jlens-swe-qwen-code-candidate-probe-public-2026-07-17.json` | 3,018,632 | `cf29a51f6267b0a3692362e3a0a3d860accbe1a4d3bc6c86f3115cac89e05153` |
| `validation/jlens-swe-qwen-code-longest-preflight-2026-07-17.json` | 547,599 | `e6a064ee99177372ac14bd62a33cbd069c210699a45b4845a62522950b5c6886` |
| `validation/jlens-swe-qwen-code-prompt-provenance-2026-07-17.json` | 5,967 | `8426064cf82c48961433987e04b43cbc9f54375feb8e326c60496215835dfc1e` |
| `validation/jlens-swe-qwen-code-analysis-2026-07-17.json` | 104,118 | `739a1963410d4043f95c4c1757cd00d3e743bcaeb59747f9cf4654f84ae91af8` |
| `validation/jlens-upstream-multihop-control-analysis-2026-07-17.json` | 206,885 | `ea07313ff6f788d51e90cac00a987881264a69c2c93cc6b67eff64f937e856d0` |
| `validation/jlens-swe-qwen-code-trajectory-calibration-2026-07-17.json` | 114,825 | `e7c6bffda6c8e7b01e13e1117025cfb1bb71155888243b2ab68c2d0733d02b7d` |
| `validation/jlens-swe-qwen-code-semantic-calibration-2026-07-17.json` | 10,843 | `8b2020c47ede816a85e03d1d1550441642ff8325019fc6cd170a72b20ea9b125` |
| `validation/jlens-swe-qwen-code-intermediate-analysis-2026-07-17.json` | 263,279 | `29f7cb2f1ffe7948f7836c49db46020864fd4e2876ec060ca03637edd5e034db` |
| `validation/jlens-swe-qwen-code-intermediate-prompts-2026-07-17.json` | 1,670,877 | `fc3293d64323cb25ea4ae1626e5a3508c983bec18a25ff2241ddf74d3a86e9fc` |
| `validation/jlens-swe-qwen-code-intermediate-prompts-summary-2026-07-17.json` | 3,017 | `4438fe2326b73262a9cc5a6064c35e9f8d99b5762156d57c007cd3a61d68d427` |

Native stage replay took 51.069 seconds, public stage replay 52.201 seconds,
the five-step native candidate replay 64.265 seconds, the public candidate
replay 48.960 seconds, and the longest-context preflight 43.161 seconds. The
full runs recorded a peak CUDA allocation/reservation of
26,074,220,032/28,290,580,480 bytes.

These report hashes bind the recorded run. A fresh report includes new
timestamps and measured durations, so byte-for-byte report equality is not an
acceptance criterion; the pinned inputs, residual identities, assertions, and
decoded numeric fields are the reproducibility checks.

## Publication Surface

The SWE J-lens reports intentionally retain the full rendered task prompts,
reversible token IDs, original Qwen reasoning text, trace UUIDs, noncredential
local paths, and process metadata needed to audit exact replay inputs. They are
publication-reviewed, not sanitized. A credential-pattern review found no
GitHub tokens, API keys, bearer credentials, email addresses, or public IP
addresses. The trace UUIDs and local paths are provenance labels, not
credentials. Do not publish this full-report format for a private task unless
its prompts have been separately approved for disclosure.

## Interpretation Limits

1. This is exact frozen-context eager replay, not a retrospective capture of
   the original compiled Qwen Code hidden states. The original SSE stream,
   accepted MTP draft-token trace, and residual tensors were not retained.
2. MTP was deliberately disabled in replay. It accelerates decoding through a
   separate draft block; it is not one of the 63 J-lens source matrices. The
   prompt residuals analyzed here belong to the accepted main model.
3. Chunked prefill and hooks change the execution mode from the original
   compiled server. Exact token inputs and checkpoint weights are pinned, but
   the failed BF16 adapter bins prevent a passing numerical-equivalence claim.
4. The native identity-STE lens was fitted on only ten short, 128-token
   WikiText prompts. These 11.9K-15.7K-token code-agent contexts are strongly
   out of distribution for the fit corpus.
5. The public lens has 1,000 fit prompts and FP16 stored matrices, but its
   fit-time model precision, quantization, exact source revision, and full fit
   command are unpublished.
6. Decoded top tokens are readouts under a fitted linear transport and
   unembedding. They are not literal thoughts, explanations, or proof of a
   unique internal causal mechanism.
7. The same-context first-token `co` versus `cot` margin is a lexically
   controlled calibration contrast, not a method-aligned quality metric. Full
   `cothm` versus `cotm` sequence totals use unequal BPE lengths and different
   teacher-forced continuations, so they are secondary.
8. This is one easy, prompt-localized SWE-Verified task. It establishes a
   concrete engineering reproduction and an exploratory task-aligned probe,
   not general SWE-bench interpretability performance.
9. The nine-layer middle-depth slice is a fixed exploratory reporting view
   chosen while analyzing these results, not a preregistered hypothesis or a
   claim that the preference holds across all 63 layers.
10. Seven of ten intermediate-probe states follow tool outputs whose target
    concepts are mostly explicit in the transcript. High ranks there can show
    retention; they do not by themselves show novel latent inference.
11. Both upstream multihop reports fail the strict final-output adapter gate,
    despite identical paired residual manifests and positive intermediate-rank
    gains. Those gains support descriptive method reproduction but not a
    numerical-equivalence or causal certificate.
12. No ablation or state-swap intervention was performed. Readout ranks and
    correlations alone do not identify a unique causal mechanism.
