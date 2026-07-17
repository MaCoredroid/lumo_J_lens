# Qwen3.6-27B NVFP4/FP8-STE Jacobian Lens

Date: 2026-07-17

## Status

The native NVIDIA path is implemented and has passed its operation,
late-suffix, and exploratory all-layer hardware gates on the local RTX 5090.
It uses the same pinned ModelOpt checkpoint and compiled vLLM target-model
graph as the serving profile. It does not substitute the BF16 source model,
bitsandbytes NF4, or the Apple MLX port for the fitting forward.

The existing 128-token prompt-0 pair proof and short all-layer proof predate
the hardened `model.identity` field and full shard-hash binding. They validate
the operator math and baseline/observer parity that they measured, but they are
not production capture certificates. The production runner intentionally
rejects reuse of those ignored exploratory artifacts and will recapture and
reprove every prompt under the strict identity contract.

The complete ten-prompt artifact is **pending**. No final `n=10` NVFP4/FP8-STE
lens hash or held-out score is claimed until all ten prompts have committed and
all 63 final matrices have passed their integrity gates. The current evidence
establishes that the operation/replay path is executable and numerically
validated. It does not yet establish a hardened end-to-end production run or a
completed corpus-averaged reproduction.

| Level | Status | Meaning |
|---|---|---|
| Public `n=1000` Qwen lens on NVFP4 residuals | Complete | BF16-fitted lens, cross-precision application |
| WeZZard Apple reference | Reference only | MLX 4-bit forward and custom Metal GDN backward |
| Native NVIDIA operation and capture mechanics | Passed exploratory hardware gates | Exact deployed forward values plus declared surrogate backward; strict production recapture pending |
| All-layer reverse replay | Passed exploratory hardware gate | One real estimator row for each source layer `0..62`, targeting block 63 |
| Dense native `n=10` lens | Pending | Ten prompts, 63 full `5120 x 5120` FP32 means |

## What Is Different From The Public And Apple Paths

The public Qwen3.6 lens distributed in `neuronpedia/jacobian-lens` was fitted
against the differentiable BF16 Qwen model. Its stored matrices are FP16. When
that artifact is read against NVIDIA residuals, the operation is a
cross-precision application; it is not an NVFP4 fit.

`WeZZard/jlens-qwen36` solves a related problem on Apple Silicon. It uses MLX,
Apple 4-bit weights, and a custom Metal backward for Gated DeltaNet. None of
those kernels runs on this host. It is an architecture reference, not the
implementation or numerical oracle for this experiment.

The NVIDIA checkpoint needs a third path because the deployed graph combines:

- ModelOpt `W4A16_NVFP4` MLP and LM-head weights. On this RTX 5090, vLLM 0.23
  reports the weight-only Marlin FP4 fallback after the FlashInfer FP4 kernels
  are disabled.
- Post-load E4M3 FP8 weights and FP8 activation quantization for attention and
  GDN projections. vLLM selects the compiled Cutlass FP8 path.
- 48 Gated DeltaNet blocks and 16 full-attention blocks in the 64-block target
  model.

Those serving kernels do not expose the residual activation backward needed by
`jlens.fit`. This repository therefore captures the actual deployed forward
and supplies explicit input VJPs for its frozen operations.

## Frozen Target

| Item | Value |
|---|---|
| Model | `nvidia/Qwen3.6-27B-NVFP4` |
| Revision | `0893e1606ff3d5f97a441f405d5fc541a6bdf404` |
| Host | Ubuntu 26.04, kernel `7.0.0-27`, x86_64 |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, 33,635,434,496 bytes |
| Driver | `595.71.05` |
| PyTorch / CUDA | `2.11.0+cu130` / CUDA 13.0 |
| vLLM / Triton | `0.23.0` / `3.6.0` |
| Hidden width | 5,120 |
| Main blocks | 64: 48 GDN, 16 full attention |
| Lens sources / target | post-block residuals `0..62` / post-block residual 63 |
| MTP | Disabled for capture and fitting |
| Input modality | Text only |

The checkpoint gate hashes all three metadata files and all three weight
shards:

| File | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 88,567 | `c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338` |
| `hf_quant_config.json` | 54,902 | `fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1` |
| `model.safetensors.index.json` | 214,866 | `7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2` |
| `model-00001-of-00003.safetensors` | 9,965,652,512 | `b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d` |
| `model-00002-of-00003.safetensors` | 9,985,757,032 | `06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d` |
| `model-00003-of-00003.safetensors` | 1,970,287,640 | `e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845` |

These pins are embedded in `scripts/modelopt_checkpoint.py`. The production
runner revalidates the checkpoint immediately before every prompt commit, and
it rejects model-path substitution.

MTP is intentionally outside this contract. The draft block is a serving-time
speculation mechanism, not source layer 64. Captures cover the accepted main
model's compiled prefill with `language_model_only=True`; the already certified
MTP serving and SWE-bench workflow remains a separate experiment.

## Estimator Contract

For prompt `p`, source layer `l`, and output coordinate `i`, the fitter uses the
Anthropic future-summed VJP estimator:

```text
J_l[p][i, :] = mean over source positions s=16..126 of
               d(sum over target positions t=16..126 h_63[t, i])
               / d h_l[s, :]

J_l = mean over ten prompts of J_l[p]
```

The final RMSNorm and LM head are not part of `J_l`. Readout applies the lens,
then final RMSNorm once, then the LM head:

```text
logits = lm_head(final_norm(h_l @ J_l.T))
```

The production corpus is exactly
`configs/jlens_nf4_fit_prompts.json`, SHA-256
`2c36f17dee7287c096f7d1fdb7f8d7ecb8372c6cf8d13d7af8cfaed820439d3b`.
It contains ten 128-token Wikitext train sequences at frozen row indices
`3, 9, 10, 15, 16, 17, 21, 22, 26, 30`. It uses dataset revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, tokenizer
`Qwen/Qwen3.6-27B` revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, right truncation, and
`skip_first=16`.

## Forward And Backward Contract

### Exact forward

Each prompt is run in two isolated compiled vLLM processes:

1. An unmodified compiled baseline records the authoritative generation and
   the capture boundaries naturally visible without the output observer.
2. A compiled-observer process adds opaque post-output copies for the linear,
   SwiGLU, and post-block values needed by reverse replay.

The second graph is instrumented. It is accepted only after the pair verifier
binds both artifacts to the same model, prompt IDs, runtime shape, source code,
and parameter content. Reverse replay substitutes the captured deployed value
at every block boundary, so recomputation cannot silently turn the fitting
forward into an A16 or BF16 model.

The proof scope must be stated precisely:

- 688 shared internal tensors are compared directly by shape, dtype, logical
  byte length, and SHA-256 of row-major bytes. All were bit-exact in the real
  128-token prompt proof. These comprise 624 GDN and 64 full-attention tensors.
- 432 required tensors exist only in the observer artifact: 304 compiled
  linear outputs, 64 SwiGLU outputs, and 64 post-block residuals. They cannot
  be directly compared to absent baseline tensors. Exact generation-record
  parity and the direct shared-tensor proof discharge the observer modification
  indirectly for this pinned prompt and runtime shape.
- All 785 replay parameters match by name, shape, dtype, and content hash
  between the two isolated captures.

This does not prove MTP decode, another prompt, another sequence shape, or an
unmodified observer graph. Production repeats the isolated proof for every fit
prompt and rehashes the retained observer payload immediately before commit.

### Surrogate backward

The backward is deliberately named `NVFP4/FP8-STE`:

- `PackedNvFp4W4Backend` reads raw ModelOpt E2M1 packed weights and their FP8
  block/global scales from the pinned shards. It computes frozen-weight input
  VJPs in bounded tiles without materializing the complete dequantized weight.
- `LiveFp8Backend` consumes the post-load E4M3 weight and scalar scale captured
  from vLLM. It streams the frozen-weight input VJP without materializing a
  full BF16 weight.
- FP8 activation quantization uses the identity straight-through estimator.
  The forward value remains the captured quantized value. The production CLI
  accepts only `--ste-policy identity`.
- Gated DeltaNet uses an analytic causal reverse recurrence, including Q, K,
  V, decay gate, write gate, convolution state, output gate, normalization,
  residual, and MLP paths.
- Full attention replays RMSNorm, QKV, text MRoPE, causal softmax attention,
  output gating/projection, residual, and MLP derivatives.

This is not the literal derivative of FP4 or FP8 rounding. Rounding is
discontinuous, so calling it an exact quantization derivative would be false.
It is an exact deployed quantized forward paired with a declared identity-STE
surrogate backward.

Only text positions are supported. Triplicated equal MRoPE axes are accepted
as text-equivalent input; divergent three-axis multimodal MRoPE positions fail
closed because that derivative path has not been validated.

## Hardware Evidence

The following engineering gates ran on the real local pinned checkpoint and
RTX 5090. The prompt-0 and short all-layer records are exploratory because they
do not contain the later strict `model.identity`/shard binding. Production will
repeat the pair proof for every prompt and will not adopt these artifacts:

| Gate | Result |
|---|---|
| Exploratory compiled baseline/observer endpoint | Exact generated token, text, finish reason, prompt IDs, and top-logprob record |
| Exploratory shared internal boundaries | 688/688 bit-exact; 892,731,392 logical bytes |
| Exploratory observer-only completeness | 432/432 required boundaries present |
| Exploratory replay parameters | 785/785 content hashes equal; 7,266,685,440 logical bytes |
| Packed W4 layer-62 probe | Passed; relative RMS `1.2549e-7` vs dense BF16 dequantization |
| Live FP8 layer-62 probe | Passed; relative RMS `7.8404e-7` vs dense BF16 dequantization |
| Analytic GDN layers 61/62 | Passed; output relative RMS at most `0.002846`, state at most `0.002613` |
| Late suffix rows | Finite, nonzero J61/J62 rows; batched/sequential relative RMS `3.42e-4` / `1.45e-5` |
| Exploratory all-layer estimator row | 63/63 finite, nonzero rows from sources `0..62` |
| Exploratory packed/live vs dense-dequant all-layer row | 63/63 dense certificate hashes matched; worst relative RMS `0.0174684` on the frozen 128-token prompt |

The last row is a real all-depth engineering proof, not a hardened production
capture and not a finished lens. It computes one of 5,120 output rows for every
source matrix. The completed artifact requires fresh strict captures plus all
5,120 rows for all 63 matrices and all ten prompts.

## Setup And Production Run

Set up the same frozen vLLM environment and download the pinned checkpoint:

```bash
git clone https://github.com/MaCoredroid/lumo_J_lens.git
cd lumo_J_lens

scripts/setup.sh --download-model
scripts/check.sh
```

Inspect the complete frozen contract without writing fit state:

```bash
.venv-vllm/bin/python scripts/run_nvfp4_ste_fit.py \
  --work-dir .cache/nvfp4_ste_fit \
  --plan-only
```

Start the ten-prompt production fit:

```bash
.venv-vllm/bin/python scripts/run_nvfp4_ste_fit.py \
  --work-dir .cache/nvfp4_ste_fit
```

Resume after an interruption:

```bash
.venv-vllm/bin/python scripts/run_nvfp4_ste_fit.py \
  --work-dir .cache/nvfp4_ste_fit \
  --resume
```

After the runner reports `status: completed`, export the raw final means to the
four-key checkpoint consumed by upstream `JacobianLens.load` and
`JacobianLens.from_pretrained`:

```bash
.venv-vllm/bin/python scripts/export_nvfp4_ste_lens.py \
  --final-mean .cache/nvfp4_ste_fit/final-mean \
  --output .cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt
```

The exporter revalidates every source matrix before and after serialization,
maps the raw matrices from files instead of first copying 6.15 GiB into
anonymous RAM, verifies the written checkpoint, refuses to overwrite an
existing path, and prints its byte size and SHA-256. The checkpoint keys are
`J`, `n_prompts`, `source_layers`, and `d_model`. Export is a post-completion
step; the command cannot turn an incomplete work directory into a lens.

Apply the exported lens only with the exact completed fit state and externally
recorded hashes:

```bash
LENS=.cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt
STATE=.cache/nvfp4_ste_fit/state.json
LENS_SHA256=$(sha256sum "$LENS" | cut -d' ' -f1)
STATE_SHA256=$(sha256sum "$STATE" | cut -d' ' -f1)

scripts/run_jlens_nvfp4.sh \
  --lens-kind nvfp4-ste \
  --lens-path "$LENS" \
  --lens-sha256 "$LENS_SHA256" \
  --lens-provenance .cache/nvfp4_ste_fit/final-mean/metadata.json \
  --lens-state "$STATE" \
  --lens-state-sha256 "$STATE_SHA256" \
  --prompts-file configs/jlens_prompts.json \
  --layers all --positions=-1 --top-k 10 \
  --output validation/jlens-native-nvfp4-ste-on-nvfp4.json
```

This is intentionally an exact-production-run verifier, not a generic loader
for arbitrary NVFP4-STE fits. It binds the completed `state.json`, final-mean
metadata, prompt commits, compiled-observer proofs, source/model hashes, and
exported tensors. The verifier holds the checked lens inode open through
readout, so replacing the CLI path cannot change the tensors being applied.
Each evaluation also rehashes all three ModelOpt shards before model loading and
after readout.

Do not add a custom model path to a production command. The runner resolves the
exact cached NVIDIA revision and fails if metadata, shard bytes, prompt
manifest, source manifest, estimator settings, or resume state differs.

The default cotangent batch is 256. A real prompt-0 sweep measured:

| Batch | Seconds per batch | Peak allocated | Peak reserved |
|---:|---:|---:|---:|
| 4 | 7.427 | 0.408 GiB | 0.570 GiB |
| 8 | 9.896 | 0.519 GiB | 0.836 GiB |
| 16 | 16.458 | 0.770 GiB | 1.291 GiB |
| 32 | 30.739 | 1.273 GiB | 2.312 GiB |
| 64 | 59.495 | 2.286 GiB | 2.912 GiB |
| 128 | 115.369 | 4.299 GiB | 5.479 GiB |
| 256 | 228.215 | 8.323 GiB | 10.609 GiB |

At batch 256, 20 row batches cover one prompt. The measured estimator rate
projects to about 76.1 minutes per prompt and 12.7 hours for ten prompts.
Isolated baseline/observer captures, content hashing, prompt commits, final
averaging, and publication add wall time beyond that estimator-only estimate.

## State, Resume, And Output

The work directory is transactional:

- `state.json` binds the full run contract and points to the next prompt/row.
- `current/layer-XX.f32` stores the current prompt's row chunks with hashes.
- `sum-NNNNNN/layer-XX.f32` is an immutable, integrity-hashed committed sum.
- `captures/` contains each prompt's isolated capture state and retained
  observer payload. The large uninstrumented baseline payload is deleted only
  after the pair proof records its hash.
- `run-progress.json` records timing and memory telemetry; fit state, not this
  journal, is authoritative for resume.
- `final-mean/layer-XX.f32` and `final-mean/metadata.json` appear only after all
  ten prompts commit and streaming finalization succeeds.
- The optional exported `.pt` is an upstream-compatible publication form; the
  integrity-gated raw `final-mean/` directory remains its authoritative source.

Every matrix is little-endian FP32 with shape `[5120, 5120]`, exactly 100 MiB.
The 63-matrix mean is 6,300 MiB (6.152 GiB). Prompt commits and finalization use
generation-stamped directories and atomic renames. Resume rehashes completed
row prefixes and committed sums before doing new work; contract drift fails
closed.

The finalizer checks every output chunk for finiteness. Its metadata must report
`n_prompts=10`, sources `0..62`, target 63, per-layer shape/dtype/size/content
hashes, an aggregate layer-manifest hash, the prompt/capture bindings,
runtime/source provenance, and the declared forward/backward contract. Until
that metadata exists and validates, the correct result label is **native path
validated, final artifact pending**.

## Limits

1. The fit is an identity-STE surrogate Jacobian, not a literal derivative of
   quantization rounding.
2. The direct observer proof is scoped to each pinned text prompt and runtime
   shape. The 432 observer-only values are supported indirectly, not by a
   direct baseline tensor comparison.
3. MTP and speculative decode are excluded. MTP remains enabled and separately
   validated in the serving/SWE-bench profile.
4. Divergent multimodal MRoPE positions are unsupported.
5. Ten prompts are the minimum usable local scale, not the public artifact's
   `n=1000` scale.
6. No quality, public-lens similarity, or held-out readout conclusion is
   available until the full ten-prompt artifact is finalized and evaluated.
