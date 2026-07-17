# Qwen3.6 Quantized Jacobian Lens Fit Contract

This document defines what counts as fitting a new Jacobian Lens for
Qwen3.6-27B on this host. Applying the public FP16 lens, whose fit-time model
precision and quantization are unpublished, to NVFP4 activations does not
satisfy this contract.

## Measured implementation status

On July 16, 2026, this contract was satisfied at the **minimum usable new
lens** level with a differentiable bitsandbytes NF4 forward:

- Model: `Qwen/Qwen3.6-27B` at revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Quantization: 496 NF4 linears, double quantization, BF16 compute, 64-element
  blocks, and 256-element nested blocks.
- Estimator: exact Anthropic future-summed VJP, ten 128-token prompts,
  `skip_first=16`, cotangent batch 32, source layers `0..62`, target block 63,
  and all 5,120 rows.
- Result: 63 finite FP32 `[5120,5120]` matrices in a
  6,606,048,039-byte artifact with SHA-256
  `54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f`.
- Time: 6,367.555 estimator seconds and 6,496.513 cumulative in-process
  invocation seconds (1:48:16.5). The timestamps span 6,566.933 seconds,
  including the deliberate pause between a one-prompt stop and nine-prompt
  resume.
- Peak CUDA memory: 23.252 GiB allocated and 24.201 GiB reserved.

The complete certificate is
[`validation/jlens-nf4-fit-provenance-2026-07-16.json`](../validation/jlens-nf4-fit-provenance-2026-07-16.json),
and the independent artifact record is
[`validation/jlens-nf4-artifact-verification-2026-07-16.json`](../validation/jlens-nf4-artifact-verification-2026-07-16.json).
The full procedure and evaluation are in
[`JLENS_NF4_EXPERIMENT.md`](JLENS_NF4_EXPERIMENT.md).

This result does **not** satisfy the strict NVFP4 fit contract below. The local
NF4 lens was cross-applied to that checkpoint, but its strict paired adapter
certificate failed; the public-lens control failed with the same adapter
errors. Therefore the cross-application is also not certified.

On July 17, the separate native NVIDIA implementation completed its strict
ten-prompt production run. It uses exact compiled NVFP4/FP8 forward captures,
packed/live input VJPs, identity STE for FP8 activation quantization, and
analytic GDN. Production did not reuse the older exploratory capture: it
recaptured and reproved every prompt under hardened model/shard/source binding.
The result is 63 finite FP32 `[5120,5120]` matrices; its exported checkpoint is
6,606,046,478 bytes with SHA-256
`82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057`.
The exact verifier and both upstream loaders passed. See
[`JLENS_NVFP4_STE_EXPERIMENT.md`](JLENS_NVFP4_STE_EXPERIMENT.md) for its
contract, evidence scope, commands, and measured status. This native result is
an exact deployed quantized forward with a declared identity-STE surrogate
backward; it is not the literal derivative of quantization rounding.

## Reference estimator

The normative implementation is `anthropics/jacobian-lens` at commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`. For a 128-token prompt, source
layer `l`, and target block output 63, it computes

```text
J_l[prompt][i, :] = mean over source positions s=16..126 of
                    d(sum over target positions t=16..126 h_63[t, i])
                    / d h_l[s, :]

J_l = mean over prompts of J_l[prompt]
```

Causality makes terms with `t < s` zero, so this is the paper's
future-summed estimator. It takes one forward pass and
`ceil(5120 / dim_batch)` retained-graph VJPs per prompt. The final RMSNorm is
not part of `J_l`; it is applied once, after transport, during readout:

```text
logits = lm_head(final_norm(h_l @ J_l.T))
```

The public Qwen artifact has matrices for source layers 0 through 62. Its
layer-62 matrix is consistent with this contract: it is close to an identity
plus the last block's branch derivative. It is not a final-norm Jacobian.

## What the Apple reference does differently

`WeZZard/jlens-qwen36` commit
`d788bc321dc7bad4ed33e1465b306389959b0046` is useful architecture and kernel
reference code, but it is not a numerical oracle for the Anthropic estimator:

- It uses 32-token prompts, skips positions 0 through 3, and keeps the tail
  when truncating. The Anthropic Qwen path uses 128 tokens, skips 0 through
  15, and Hugging Face right truncation keeps the prefix.
- It forms position-averaged per-layer matrices and chain-multiplies them.
  A product of position averages is not the position average of the complete
  suffix Jacobian. Its own real-layer comparison reports 1.45% to 2.14%
  relative error before errors are compounded across layers; its toy suffix
  comparison reports 2.6% to 4.8% after fixing the indexing bug.
- It includes the final RMSNorm Jacobian in the chained matrix, while its
  readout path applies final RMSNorm again. That differs from the public Qwen
  artifact and the Anthropic application contract.

A CUDA port may reuse its GDN recurrence equations and tests. A result that
uses its averaged-matrix chain must be labeled an approximate WeZZard-style
lens, not an exact reproduction of the public estimator.

## Quantization derivative contract

The pinned NVIDIA checkpoint is mixed precision:

- MLP projections and the LM head are `W4A16_NVFP4`, group size 16. Their
  activations remain FP16/BF16. With frozen packed weights, an input VJP using
  the corresponding dequantized weight is a meaningful derivative of the
  executed linear map.
- Attention and GDN projections are static FP8 weight-and-activation
  quantized operations. Rounding is discontinuous, with a zero derivative
  almost everywhere and no derivative at bin boundaries.

Consequently, a useful fit through the exact serving checkpoint must declare
the surrogate used for FP8 activation quantization. The implemented production
contract preserves captured quantized forward values and uses an identity
straight-through input VJP through activation quantization with the frozen
post-load FP8 weight. This is an **STE Jacobian of the NVFP4/FP8 forward**, not
the literal mathematical derivative of rounding. A fit that instead executes
all projections with A16 activations must say so and report forward parity
against the serving path.

### NF4 alternative

An NF4 path avoids the packed-kernel backward problem only if the quantized
linear implementation returns input gradients and Qwen's GDN is forced
through differentiable PyTorch operations. Freeze all weights; the fit needs
gradients with respect to residual activations, not weight gradients. Pin the
source BF16 revision, quantizer/library version, compute dtype, double-quant
setting, block size, and the serialized NF4 artifact hashes.

Transformers' quantized single-device loader uses `device_map` and therefore
also requires the pinned Accelerate runtime. The certified fit environment
uses `accelerate==1.14.0`; omitting it fails before model construction.

That failure occurred in the first `C=4` diagnostic attempt. Its state records
`"accelerate": "missing"`, no `model_execution` record, and no committed
work. Running `scripts/setup_fit.sh` installed the pinned
`accelerate==1.14.0`; the diagnostic was then restarted in a fresh work
directory and completed. A stale pre-fix work directory must not be resumed,
because the resume contract intentionally binds package versions.

This produces a lens fitted to an NF4 forward. It is a valid reproduction of
the fitting method on a differentiable 4-bit Qwen3.6 model, but applying that
lens to `nvidia/Qwen3.6-27B-NVFP4` is still cross-quantization. Do not label it
an NVFP4-fitted lens. The native path uses the pinned NVIDIA weights, exact
deployed forward captures, and the declared identity-STE surrogate above.

## Completion levels

These levels must not be conflated.

### Kernel and fitter proof

This is an engineering smoke test, not a usable lens:

- One frozen prompt with at least 18 tokens; use 128 tokens for parity.
- Full dense FP32 matrices for source layers 61 and 62, targeting block 63.
- Layer 61 forces the suffix through GDN block 62 and full-attention block 63;
  layer 62 alone does not test GDN backward.
- All 5120 output basis rows are computed. A few rows or a random sketch does
  not pass this level.

### Minimum usable new lens

The paper reports that the J-lens beats its baselines with as few as ten
prompts. The minimum scientifically useful local artifact is therefore:

- Ten frozen, pretraining-like prompts of exactly 128 model tokens.
- `skip_first=16`, source layers 0 through 62, target block 63.
- 63 full dense `5120 x 5120` matrices, averaged over all ten prompts.
- Readout and held-out comparisons against both the public lens and logit
  lens. The artifact must be labeled `n=10`; it is not paper-scale.

One prompt over all layers is a full-depth fitter proof, but it is not a
usable corpus-averaged lens. Fitting only selected layers is legitimate for a
declared layer-specific experiment, but it is not a reproduction of the
63-layer public artifact.

### Research and paper scale

- About 100 prompts is the upstream recommendation for a usable,
  substantially stabilized lens.
- The paper and public Qwen artifact use 1,000 sequences of 128 tokens.

The canonical corpus candidate is the first qualifying records from
`Salesforce/wikitext`, configuration `wikitext-103-raw-v1`, train split,
with stripped length at least 600 characters. Pin dataset revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, materialize the selected raw
texts, and record their exact token IDs. The public artifact does not expose
its row IDs, tokenizer revision, model revision, or fitter settings, so exact
corpus identity cannot be inferred from its filename alone.

## Resource scaling on this host

For `d_model=5120`:

- One dense FP32 matrix is exactly 100 MiB.
- One dense FP16 matrix is exactly 50 MiB.
- 63 FP32 matrices are 6,300 MiB (6.152 GiB).
- 63 FP16 matrices are 3,150 MiB (3.076 GiB).
- With `dim_batch=32`, fitting performs 160 VJPs per prompt; with
  `dim_batch=8`, 640; with `dim_batch=4`, 1,280; with `dim_batch=1`, it
  performs 5,120. The leading backward FLOPs are similar, but the lower batch
  substantially reduces peak activation and returned-gradient memory. The
  NF4 production contract on this host uses `dim_batch=32`: its real 27B
  diagnostic allocated/reserved 23.21/24.16 GiB and matched sequential VJPs
  exactly for 32 rows at both source layers 61 and 62. The `C=4`, `C=8`, and
  `C=32` diagnostics completed in approximately 23.6, 24.5, and 28.7 seconds.

The upstream in-memory accumulator can simultaneously retain the running
sum, the current prompt matrices, and the final mean. That approaches 18.5
GiB of host tensor data before serialization, against about 25 GiB currently
available on this machine. The production fitter should write row batches to
FP32 memory-mapped temporary matrices, atomically commit a prompt only after
all rows finish, and stream the final mean. Peak transient disk use is about
18.45 GiB: one committed FP32 sum, one transactional FP32 prompt or next sum,
and one FP32 final-mean generation. Publication staging can require additional
space for the serialized checkpoint. This host has ample disk capacity.

Saving FP16 is allowed only after an explicit finite and range check. The
upstream default cast can silently convert finite values above 65,504 to
infinities. Keep the authoritative fit/checkpoint in FP32 and separately
verify any FP16 publication copy.

## Required provenance

The fit certificate must contain all of the following before it can support a
reproduction claim:

1. Model repository, immutable model revision, config and quantization-config
   SHA-256 values, model index SHA-256, and hashes or immutable Hub identities
   for every weight shard.
2. Tokenizer repository/revision, tokenizer file hashes, special-token and
   truncation settings, plus every selected prompt's raw-text SHA-256, exact
   token IDs, and token count.
3. Dataset repository, immutable revision, configuration, split, selection
   rule, selected row indices, and a SHA-256 over the materialized prompt
   manifest.
4. Fitter source commit and source-tree manifest, exact command line,
   `source_layers`, `target_layer`, `max_seq_len`, `skip_first`, `dim_batch`,
   dtypes, quantization derivative contract, seed, and determinism settings.
5. Python and package freeze, PyTorch/CUDA/driver versions, GPU identity, run
   start/end times, elapsed time, and peak host/CUDA memory.
6. For every output layer: shape, storage dtype, finite count, min/max,
   Frobenius norm, trace, and a SHA-256 over canonical little-endian tensor
   bytes. Also record the whole-file SHA-256 and the metadata sidecar SHA-256.
7. Checkpoint `n_done`, `next_idx`, and prompt-manifest hash. Resume must reject
   any model, corpus, layer, estimator, or source mismatch.

`n_prompts`, `d_model`, and layer keys alone are not provenance. The current
public artifact only provides those fields.

## Acceptance gates

### Operation and kernel gates

For each custom VJP, compare against an unfused differentiable reference on
small tensors and real captured activations:

- Forward max and RMS error are reported separately from backward error.
- VJP relative L2 error is at most `1e-3` in FP32 reference tests and at most
  `2e-2` for BF16/quantized real-activation tests.
- GDN tests cover `dq`, `dk`, `dv`, decay-gate `dg`, write-gate `dbeta`, the
  output gate, convolution state, causal future-position flow, saturated
  gates, and sequence lengths crossing CUDA tile boundaries.
- Full-attention tests include Q, K, V, output projection, RoPE, softmax, and
  gating paths.
- NVFP4 and FP8 linear tests validate input VJPs against explicitly
  dequantized weights and document the FP8 STE.

For a random activation direction `v` and cotangent `u`, the adjoint identity
must also hold:

```text
dot(JVP(v), u) == dot(v, VJP(u))
```

### Fitter gates

- Every requested basis row, layer, and prompt is present exactly once.
- All accumulator and final tensors are finite FP32 with shape
  `[5120, 5120]`.
- A resumed run is tensor-identical to an uninterrupted run under deterministic
  settings.
- A two-block direct suffix VJP for layers 61 and 62 matches the fitter's
  matrices within the real-activation tolerance. This catches layer indexing,
  transpose, final-norm, position-mask, and GDN omissions.
- Applying layer `l` uses `residual @ J_l.T`, then the model final norm exactly
  once, then the model LM head.

### Lens comparison gates

Numerical comparison with the public n=1000 FP16 lens is required but cannot
prove equality because its fit-time precision, quantization, and full
provenance are absent and the new fit uses a pinned quantized forward. Report,
per layer:

The acceptance corpus is frozen separately from fitting in
`configs/jlens_nf4_eval_prompts.json`: Wikitext validation rows 3, 18, 42, and
49 at revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Each prompt is
exactly 128 pinned model tokens. Its manifest SHA-256 is
`cd0fe64e800c7b937fcd891196eed6d7c30a8ff1246b9555dc2962bf61c9a56b`, and
the predeclared teacher-forced evaluation positions are 16, 32, 64, and 96.
The fit uses the train split, so no selected evaluation row is a fit row.

- Relative Frobenius difference and Frobenius cosine.
- Trace, norm, and best scalar-identity component.
- Row-wise cosine quantiles.
- Top-1/top-5 agreement and rank correlation on frozen held-out prompts at
  positions 16 or later.
- The same held-out metrics for the logit-lens baseline.

Do not invent a global public-lens similarity threshold after seeing the
result. Kernel equivalence, estimator completeness, artifact provenance, and
held-out behavior are the pass/fail gates; public-lens similarity is a
reported scientific result. The frozen currency and France prompts remain
useful regressions, but two semantic examples alone do not validate a fit.

The completed `n=10` artifact measured global Frobenius cosine `0.750216`,
mean per-layer cosine `0.820655`, and global relative Frobenius difference
`0.865690` against the public `n=1000` lens. Held-out NF4 evaluation completed
over all 63 layers and 16 observations per layer. When cross-applied to NVFP4,
three of four prompts exceeded the strict full-logit max threshold
(`0.125 > 0.0625`), although every full-logit RMS and greedy top-1 check
passed. One prompt also failed final-norm max and top-5 prefix. The public lens
control reproduced those exact adapter errors, so this failure does not reject
the NF4 fitting method, but it does reject certification of the NVFP4
cross-application.

Those values describe the NF4 artifact only. The completed native
NVFP4/FP8-STE `n=10` artifact's separate public-lens comparison measured global
Frobenius cosine `0.732877`, mean per-layer cosine `0.822360`, and global
relative difference `0.934596`. Its 1,008-observation paired held-out comparison
measured target-rank Spearman `0.902843`, top-1 agreement `0.412698`, top-5
overlap `0.493651`, and target-score RMSE `2.780105`. Both native/public adapter
certificates failed with identical pre-lens residual evidence. For either lens,
that certificate is computed before lens application and must remain separate
from lens-quality metrics. The public control stores 63 FP16 matrices fitted
over `n=1000`, but its fit-time model precision and quantization are
unpublished.

## Sketches and low-rank methods

JVP computes columns while VJP computes rows; neither reduces the information
required for an exact dense matrix. Without structural assumptions, a full
`5120 x 5120` matrix needs 5,120 independent directions. Batched VJPs improve
hardware utilization, not asymptotic work.

Rademacher or randomized-SVD fits can provide an unbiased or low-rank
diagnostic with fewer VJPs, but they do not satisfy the dense-artifact gates.
The WeZZard project measured 316% single-prompt Frobenius error with 512
Rademacher probes. An independent randomized-SVD check of the pinned public
artifact on this machine found rank-256 captured approximately 97.3%, 89.3%,
61.3%, 18.2%, and 14.3% of total Frobenius energy at layers 0, 24, 40, 56,
and 62 respectively. A low-rank-only lens is especially unsuitable late in
the stack.

A scalar-identity plus low-rank residual is more compact for late layers: at
layer 62, the best scalar identity accounts for about 87.6% of energy and a
rank-256 correction raises represented energy to about 97.3%. This is useful
for experimental acceleration or compression after the dense fit exists; it
is not a substitute for fitting and publishing the requested dense lens.
