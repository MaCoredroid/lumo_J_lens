# Jacobian Lens on Qwen3.6-27B NVFP4

## 1. Result

This repository reproduces the forward-only Jacobian Lens on the local RTX
5090 using the exact NVIDIA serving checkpoint already certified by this
project:

```text
nvidia/Qwen3.6-27B-NVFP4
revision: 0893e1606ff3d5f97a441f405d5fc541a6bdf404
```

The successful run occurred on July 16, 2026. It applied a pinned, pre-fitted
1,000-prompt lens to residuals produced by the GPU-resident ModelOpt
NVFP4/FP8 model, decoded both the vanilla logit lens and Jacobian Lens through
the checkpoint's own quantized LM head, and covered every fitted source layer
`0..62`.

The complete machine-readable result is
[`validation/jlens-nvfp4-2026-07-16.json`](../validation/jlens-nvfp4-2026-07-16.json).
Its SHA-256 is:

```text
0a9787534f38f867b6c48e3edfcb9d5ee946dd2113ccbcdfdcdfda7873ed866d
```

[`validation/jlens-source-manifest.sha256`](../validation/jlens-source-manifest.sha256)
ties that result to the exact runner, launcher, prompt set, verifier, tests, and
vLLM dependency freeze that produced and checks it.

This is a **cross-precision application**, not an NVFP4 fit. The public lens
was fitted against differentiable model weights and is applied here to
quantized forward activations. That distinction matters for interpreting
rank and score differences.

## 2. Source Pins

### Reference implementation

The mathematical and API reference is
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e),
Apache-2.0, at commit:

```text
581d398613e5602a5af361e1c34d3a92ea82ba8e
```

Its readout is `unembed(J_l @ h_l)`, where `h_l` is the residual after block
`l`, `J_l` transports it to the final-layer basis, and `unembed` means final
RMSNorm followed by the model's LM head. The upstream Hugging Face adapter
cannot load the NVIDIA checkpoint because Transformers 5.12.1 does not
recognize its `modelopt` quantization type and the on-disk weights are packed.

### Pre-fitted lens

The experiment pins this exact artifact:

| Field | Value |
|---|---|
| Repository | [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens/tree/a4114d7752d11eb546e6cf372213d7e75526d3a1) |
| Revision | `a4114d7752d11eb546e6cf372213d7e75526d3a1` |
| Filename | `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt` |
| Size | 3,303,032,772 bytes |
| SHA-256 | `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1` |
| Metadata | `n_prompts=1000`, `d_model=5120`, source layers `0..62` |
| Tensors | 63 finite FP16 matrices, each `[5120, 5120]` |

Do not use the 381,550,248-byte file named
`Qwen3.6-27B_jacobian_lens.pt`. Its internal archive and dimensions identify
it as a mislabeled GPT-OSS lens (`d_model=2880`, layers `0..22`). The adjacent
historical YAML is also stale GPT-OSS metadata. The downloader therefore
accepts only the n1000 filename and gates it by immutable revision, byte size,
SHA-256, and tensor metadata.

### Apple reference

[`WeZZard/jlens-qwen36`](https://github.com/WeZZard/jlens-qwen36/tree/d788bc321dc7bad4ed33e1465b306389959b0046)
was used only as an architecture and expected-behavior reference. Its MLX
runtime and custom Metal GDN backward kernel are not imported or executed.
The Linux run in this repository is vLLM, PyTorch, CUDA, and ModelOpt.

## 3. Architecture Mapping

Qwen3.6-27B reports the Hugging Face architecture family `qwen3_5` and has:

- 64 main decoder blocks of width 5,120.
- 48 Gated DeltaNet linear-attention blocks.
- 16 full-attention blocks, every fourth layer.
- A 248,320-token vocabulary.
- One separate MTP draft block in the NVIDIA checkpoint.

The lens was fitted for main-model blocks `0..62`, with block 63 as the target.
The MTP draft block is not a 65th source layer and is never passed through the
lens.

vLLM's Qwen block returns two tensors rather than the Hugging Face post-block
residual:

```python
branch_output, residual = block_output
post_block = branch_output + residual
```

The offline runner forces `VLLM_ENABLE_V1_MULTIPROCESSING=0` and eager mode,
then uses vLLM's public `LLM.apply_model()` entry point to install hooks on the
64 main blocks. For one request with one token of generation, each hook copies
only the requested prompt positions to CPU. The runner then computes one
matrix at a time:

```python
transported = post_block.float() @ J_l.float().T
logits = language_model.compute_logits(final_norm(transported.bfloat16()))
```

Loading the artifact with `torch.load(..., weights_only=True, mmap=True)` keeps
the 3.3 GB checkpoint memory-mapped. It avoids upstream `JacobianLens.load()`,
which expands every lens matrix to FP32 and would consume about 6.6 GB of host
RAM before any model work.

## 4. Setup And Run

The lens workflow reuses the frozen vLLM environment and CUDA workarounds from
the serving workflow. No Apple, MLX, TransformerLens, bitsandbytes, AWQ, or
GPTQ dependency is needed.

```bash
git clone https://github.com/MaCoredroid/lumo_J_lens.git
cd lumo_J_lens

# Creates the pinned vLLM environment and downloads the NVIDIA model.
scripts/setup.sh --download-model

# Downloads 3.3 GB and performs the complete hash/shape/dtype/finite gate.
.venv-vllm/bin/python scripts/download_jlens.py

# Runs both frozen prompts, all 63 fitted layers, and the final-position grid.
scripts/run_jlens_nvfp4.sh \
  --prompts-file configs/jlens_prompts.json \
  --layers all \
  --positions=-1 \
  --top-k 10 \
  --output validation/jlens-nvfp4-local.json
```

The run fails closed unless the model and lens are already present at their
pinned Hugging Face revisions. `scripts/download_model.sh` and
`scripts/download_jlens.py` are the explicit network steps.

Useful narrower probes are:

```bash
scripts/run_jlens_nvfp4.sh \
  --prompt "The capital of France is" \
  --layers 0,16,32,48,56,60,62 \
  --positions=-1 \
  --top-k 5
```

The default limits are deliberately diagnostic: one sequence, 256 tokens,
82% GPU utilization, eager execution, language-only loading, and no prefix
cache. Prompts must fit in one scheduled prefill. The runner uses raw
completion text, not a chat template, matching the reference lens regime.

## 5. Acceptance Gates

A run is successful only if all of these hold:

1. The model revision, config hash, and safetensor index hash match the pinned snapshot.
2. The model exposes 64 blocks, width 5,120, 48 GDN layers, and 16 full-attention layers.
3. The lens byte size and SHA-256 match the pin.
4. The lens metadata is exactly 1,000 prompts, width 5,120, and source layers `0..62`.
5. Every requested source block and block 63 is captured at every requested position.
6. Every `h @ J.T` result is finite.
7. Block 63 is reconstructed independently and compared with the real fused
   final path: both top-1 IDs equal vLLM's greedy token, their top-5 IDs match,
   final-norm max/RMS error is at most `0.125`/`0.006`, and full-vocabulary
   logit max/RMS error is at most `0.0625`/`0.01`.
8. GPU state is shut down cleanly and released when the process exits.

The certified result reports all gates true for both prompts. GPU memory
returned to 383 MiB after process exit.

A separate hardware smoke requested only position `-2` at layers 0 and 62.
The runner captured `-1` internally for parity, returned only the requested
layer rows, passed all adapter gates, and again released GPU memory to 383 MiB.

## 6. Measured Experiment

### Machine and timing

| Item | Measured value |
|---|---:|
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0 |
| VRAM | 32,607 MiB |
| Driver | 595.71.05 |
| vLLM / PyTorch | 0.23.0 / 2.11.0+cu130 |
| Model load | 7.439 s |
| Currency prompt generation / all-layer readout | 0.097 s / 1.090 s |
| France prompt generation / all-layer readout | 0.063 s / 0.988 s |
| Artifact gates through readout | 13.967 s |
| Peak CUDA allocated / reserved | 25.62 GiB / 27.60 GiB |

vLLM resolved the checkpoint as ModelOpt mixed quantization. On this consumer
Blackwell GPU, vLLM 0.23 selected Cutlass FP8 linears and the Marlin
weight-only FP4 fallback for the NVFP4 weights.

### Currency prompt

Prompt:

```text
Fact: The currency used in the country shaped like a boot is
```

The immediate greedy token was ` the`. That is syntactically correct but does
not expose the future answer. The Jacobian Lens showed the answer computation
forming earlier:

| Layer | Vanilla top-1 | Jacobian top-1 | Greedy-token rank in J readout |
|---:|---|---|---:|
| 0 | ` ` | `.` | 74 |
| 40 | `...` | ` Italy` | 68,212 |
| 48 | `...` | ` Italy` | 6,312 |
| 52 | `意大利` | ` Italian` | 1,206 |
| 56 | ` called` | ` Italian` | 507 |
| 58 | ` euro` | ` euro` | 368 |
| 60 | ` euro` | ` euro` | 35 |
| 62 | ` the` | ` Euro` | 2 |

This reproduces the expected `Italy`/`Italian` middle-layer transition and
`euro` late-layer transition reported by the Qwen3.6 reference port.

### France prompt

Prompt:

```text
Question: What is the capital of France? Answer: The capital of France is
```

The greedy token was ` Paris`. The Jacobian Lens made ` Paris` top-1 at layer
56 and kept it top-1 through layers 58, 60, and 62. Independently decoding the
reconstructed block-63 residual also returned ` Paris`. Against the real fused
final path, reconstructed final-norm RMS error was `0.00524` (0.274% relative),
full-vocabulary logit RMS error was `0.00846`, and the top five IDs were exact.

## 7. MTP And Serving Context

The production profile in this repository serves the same checkpoint with its
native one-token MTP head, 32,768-token context, chunked prefill, GDN-aware
prefix caching, and Mamba `align` mode. That end-to-end profile remains
documented in [`SPEC.md`](../SPEC.md) and certified in
[`VALIDATION.md`](../VALIDATION.md), including a successful official
SWE-bench Verified task and measured MTP acceptance.

MTP is intentionally disabled in the lens process:

- MTP predicts speculative draft tokens; it does not alter the accepted
  target model's layer definitions or lens matrices.
- The lens must read main-model residuals, not the separate draft block.
- Eager, single-process execution provides deterministic hook boundaries and
  direct access to the quantized target LM head.
- The production MTP server and the lens process must not run concurrently on
  this 32 GB GPU.

Thus MTP is a separately validated serving optimization, while the lens is a
diagnostic readout of the same NVFP4 target model.

## 8. Limits And Next Steps

- The lens file embeds dimensions and prompt count but not its exact source
  model revision or corpus revision. Hash and tensor compatibility are proven;
  full fitting provenance is not recoverable from the artifact alone.
- Quantization transfer can change logits and ranks. The exact final-layer
  parity gate validates the adapter, not equivalence to BF16 activations.
- The vLLM `apply_model()` method is public, but the Qwen module layout and
  tuple output are internal APIs pinned to vLLM 0.23.0.
- This runner reads selected prompt positions. It is not a streaming web grid
  or a steering implementation. Readout is capped at eight positions per run
  to bound the `[readout rows, 248320]` logits allocation; run additional
  position slices as separate processes. The final prompt position is captured
  implicitly for the adapter parity gate when it is not part of the requested
  slice; that extra row is not added to the requested layer grid.
- Fitting a new lens on this machine is not feasible with the exact checkpoint:
  vLLM's packed ModelOpt/Marlin/GDN deployment kernels do not provide the
  activation backward path required by `jlens.fit`, and BF16 model weights
  alone require roughly 54 GB. A fresh fit needs a differentiable BF16 model
  on larger or multi-GPU hardware, or a separately validated CUDA analytic
  GDN backward implementation.

The experiment therefore establishes the practical rung that fits this host:
verified application of the stronger 1,000-prompt lens to the exact local
NVFP4 target model.
