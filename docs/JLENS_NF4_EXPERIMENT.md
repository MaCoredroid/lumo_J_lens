# Qwen3.6-27B Jacobian Lens NF4 Experiment

Date: 2026-07-16

## Outcome

The strict native-NVFP4 goal was **not reproduced**. No Jacobian Lens was
fitted through the packed `nvidia/Qwen3.6-27B-NVFP4` forward. Its ModelOpt
serving graph combines NVFP4 weight-only operations with FP8
weight-and-activation operations, vLLM exposes inference kernels rather than
an autograd graph, and no validated CUDA input VJP plus explicit FP8
straight-through surrogate was implemented. Applying another lens to NVFP4
residuals is not an NVFP4 fit.

The user-authorized fallback did succeed: the official BF16
`Qwen/Qwen3.6-27B` checkpoint was quantized at load time to differentiable
bitsandbytes NF4, and a new `n=10` Jacobian Lens was fitted on this RTX 5090.
The result contains all 63 requested dense `5120 x 5120` FP32 matrices and
passed the finite, metadata, and content-hash checks. Its integrity-gated
held-out NF4 evaluation completed; the reported quality metrics are
descriptive rather than post-hoc pass/fail thresholds.

The NF4-fitted lens was then applied to residuals from the pinned NVIDIA
NVFP4/FP8 model. That is a completed **cross-quantization application**, not a
native NVFP4 fit. Both the local-lens and public-lens held-out NVFP4 reports
have `status: failed` because the strict vLLM capture-adapter reconstruction
certificate failed. The failure values are identical between the two reports
and therefore do not compare the lenses.

## Pinned Configuration

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, 33,635,434,496 bytes |
| Driver / CUDA | 595.71.05 / CUDA 13.0 |
| Fit source model | `Qwen/Qwen3.6-27B` |
| Source revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Fit quantization | bitsandbytes NF4, double quantization, BF16 compute |
| NF4 block sizes | weight 64, nested scale 256 |
| Packed NF4 aggregate | `964aef016bf13e0c68b5322eada1af8036ee56e7b57dfadce09557c9253be0d9` |
| Python / PyTorch | 3.12.13 / `2.11.0+cu130` |
| Transformers / bitsandbytes | 5.12.1 / 0.49.2 |
| Accelerate | 1.14.0 |
| Upstream `jlens` | commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` |
| Local fitter source | commit `352fb83de16771d03d2844eddde43964c3a67b13` |
| Fit source manifest | `ebbb7ef2b6230341feb463889d03060040c1b45f540bf39c5698cccb1b02f505` |
| Determinism | seed 0, deterministic algorithms, TF32 off, cuBLAS `:4096:8` |

The text-only `Qwen3_5ForCausalLM` loader quantized 496 decoder linear
modules. All weights were frozen. Qwen's 48 Gated DeltaNet blocks used the
ordinary differentiable PyTorch implementation; the other 16 blocks used
eager full attention. The untied BF16 LM head remained on CPU and was not
used during fitting.

The ten fit prompts are 128-token prefixes from `Salesforce/wikitext`,
`wikitext-103-raw-v1`, train rows 3, 9, 10, 15, 16, 17, 21, 22, 26, and 30 at
revision `b08601e04326c79dfdd32d625aee71d232d685c3`. The frozen manifest SHA-256
is `2c36f17dee7287c096f7d1fdb7f8d7ecb8372c6cf8d13d7af8cfaed820439d3b`.

The held-out set uses validation rows 3, 18, 42, and 49 from the same dataset
revision. Its manifest SHA-256 is
`cd0fe64e800c7b937fcd891196eed6d7c30a8ff1246b9555dc2962bf61c9a56b`.
Evaluation used teacher-forced next-token targets at positions 16, 32, 64,
and 96, giving 16 observations per layer.

## Estimator

For each prompt and source layer `l` in 0 through 62, the target is the
post-block output of block 63. For output coordinate `i`, the fitted row is:

```text
J_l[prompt][i, :] = mean over source positions s=16..126 of
                    d(sum over target positions t=16..126 h_63[t, i])
                    / d h_l[s, :]

J_l = mean over the 10 prompts of J_l[prompt]
```

The final token, position 127, is excluded. Causality makes the expression the
Anthropic future-summed estimator. Output rows were evaluated in batched
cotangents of 32, so each prompt required 160 retained-graph VJP batches. The
final RMSNorm is not differentiated into `J_l`; application is exactly:

```text
logits = lm_head(final_norm(residual @ J_l.T))
```

The fitter used crash-safe, disk-backed FP32 prompt and running-sum matrices.
A prompt was committed only after all 5,120 rows completed and its content
hash was recorded. Resume checked the model, prompt, estimator, quantization,
runtime, and source identities before continuing.

## Failure And Recovery

The first real C4 diagnostic failed before model construction. The fit
environment omitted Accelerate, while Transformers' quantized single-device
loader uses `device_map` and requires it. Commit `4b7be61` added
`accelerate==1.14.0` to the pinned environment, runtime checks, package
freeze, and contract. The successful diagnostics and production fit all use
that corrected environment.

Three real-model diagnostics compared batched cotangents with sequential
autograd on the same graph at layers 61 and 62. Layer 61 forces the suffix
through a GDN block and the final full-attention block. Every comparison had
exactly zero maximum absolute and relative RMS error:

| Batch / rows | Diagnostic interval | Peak allocated | Peak reserved | Exported lens |
| ---: | ---: | ---: | ---: | --- |
| 4 | 23.633 s | 20.386 GiB | 20.557 GiB | No |
| 8 | 24.541 s | 20.736 GiB | 20.893 GiB | No |
| 32 | 28.671 s | 23.214 GiB | 24.160 GiB | No |

These row-limited runs are vectorization diagnostics, not partial lenses. The
production batch was fixed at 32 only after the C32 check passed.

## Fit Result

The first invocation intentionally committed one prompt and stopped; the
second resumed the same transactional work directory and completed prompts 2
through 10. The recorded invocation times were 657.335 s and 5,839.178 s.
Cumulative in-process invocation time was 6,496.513 s (1 h 48 min 16.5 s), of
which 6,367.555 s was estimator time. The start/completion timestamps span
6,566.933 s (1 h 49 min 26.9 s), including the 70.419-second operator gap
between the paused invocation and resume.

| Result | Measured value |
| --- | --- |
| Prompt count | 10 |
| Matrices | 63, source layers 0 through 62, target block 63 |
| Matrix shape / dtype | `5120 x 5120` / FP32 |
| Artifact size | 6,606,048,039 bytes (6.152 GiB) |
| Cumulative invocation wall | 6,496.513 s |
| Calendar start-to-completion span | 6,566.933 s |
| Peak CUDA allocated | 24,966,626,816 bytes (23.252 GiB) |
| Peak CUDA reserved | 25,985,810,432 bytes (24.201 GiB) |
| Peak host RSS | 26,483,826,688 bytes (24.665 GiB) |
| Artifact SHA-256 | `54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f` |
| Provenance SHA-256 | `08dfe022cc74bab226308dcd77a3901ec18806f8832dc8dec46524374fa3b1e7` |
| Contract SHA-256 | `2a720e5193f0e6cc733521392ee1ce3d38f8afa8425ff954f14cc1d25dd5553d` |

The verifier checked all 1,651,507,200 FP32 elements as finite and matched
each layer's canonical tensor hash against the provenance sidecar.

## Comparison With The Public Lens

The reference is the public `neuronpedia/jacobian-lens` Qwen artifact at
revision `a4114d7752d11eb546e6cf372213d7e75526d3a1`, SHA-256
`1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1`.
It is an FP16 `n=1000` artifact fitted on a BF16 model; its exact prompt rows
and full fitter provenance are not published.

Across the 63 matrix pairs, global Frobenius cosine was 0.750216 and relative
Frobenius difference, normalized by the public norm, was 0.865690. Mean and
median per-layer cosine were 0.820655 and 0.860296. Mean and median per-layer
relative difference were 0.681999 and 0.605761. Across all 322,560 rows, mean
and median row cosine were 0.789686 and 0.833997.

| Layer | Frobenius cosine | Relative difference | Mean row cosine |
| ---: | ---: | ---: | ---: |
| 0 | 0.427758 | 1.652471 | 0.382530 |
| 31 | 0.860296 | 0.605761 | 0.815759 |
| 47 | 0.936135 | 0.398972 | 0.930539 |
| 61 | 0.995200 | 0.098902 | 0.995559 |
| 62 | 0.997619 | 0.069292 | 0.997719 |

Similarity rises sharply toward the output, but these values are descriptive,
not an equivalence certificate. The artifacts differ in fit quantization,
corpus size, and possibly prompt selection.

## Held-Out NF4 Evaluation

The evaluator re-created 496 NF4 modules and required their aggregate hash to
equal the fit aggregate. It applied final RMSNorm exactly once and compared
the local lens, public lens, and ordinary logit lens. The measured evaluation
interval was 15.363 s, with 16.671 GiB peak allocated and 16.984 GiB peak
reserved.

Equal-weight macro means over all 63 layers, derived from the recorded
per-layer summaries, were:

| Comparison to public lens | Top-1 agreement | Top-5 overlap | Target-rank Spearman |
| --- | ---: | ---: | ---: |
| Local `n=10` NF4 J-lens | 0.411706 | 0.494048 | 0.765352 |
| Logit lens | 0.100198 | 0.137302 | 0.096303 |

At layer 61, local/public top-1 agreement was 14/16, top-5 overlap was 0.925,
and target-rank Spearman was 0.977733. At layer 62, they were 13/16, 0.9375,
and 0.996732. At layer 62 the local and public lenses also had identical mean
target rank (2.5), target top-1 rate (11/16), and target top-5 rate (15/16).
This small held-out result shows that the local lens tracks the public lens,
especially late in the network; it is not a paper-scale quality estimate.

## Cross-Application To NVFP4

Both lenses were applied to all 63 post-block residuals from
`nvidia/Qwen3.6-27B-NVFP4` revision
`0893e1606ff3d5f97a441f405d5fc541a6bdf404`. The runtime was vLLM 0.23.0,
eager language-model-only execution, `max_model_len=256`, BF16 readout, FP32
Jacobian transport, and 0.82 GPU-memory utilization. MTP was disabled because
this was an activation-capture diagnostic, not the production serving path.

| Run | Status | Elapsed | Model load | Peak allocated / reserved |
| --- | --- | ---: | ---: | ---: |
| Local NF4 `n=10` lens | Failed certificate | 33.328 s | 12.384 s | 25.857 / 27.916 GiB |
| Public BF16 `n=1000` lens | Failed certificate | 23.291 s | 12.388 s | 25.857 / 27.916 GiB |

The capture-adapter reconstruction fields are exactly identical for all four
prompts in the two reports. Final-logit RMS error passed its 0.01 threshold in
all cases (range 0.007698 to 0.008663), and reconstructed final-layer top-1
matched vLLM greedy output in all 8 report/prompt cases. However, three of
four prompts had final-logit maximum absolute error 0.125 against a 0.0625
limit; only validation row 42 met the limit at 0.0625. Validation row 18 also
had final-norm maximum error 0.25 against a 0.125 limit, despite every
final-norm RMS error passing the 0.006 limit, and its final-logit top-5 prefix
did not match. Therefore the strict adapter certificate correctly failed for
both runs.

This adapter check reconstructs the captured model output and is independent
of which Jacobian is later applied. Its identical failure does not show that
the two lenses are identical, but it prevents either long-prompt NVFP4 report
from being called certified. As a contrast, the current runner's original
two-short-prompt public-lens baseline passed: both final-logit maximum errors
were 0.0625, elapsed time was 16.300 s, model load was 8.846 s, and peak
reserved memory was 29,630,660,608 bytes.

The cross-applied lenses did produce finite readouts. At layer 62 on the 16
held-out position/prompt pairs, local/public top-1 predictions agreed in
12/16 cases. The NF4 lens put the teacher-forced target in top-1 for 10/16 and
top-5 for 15/16; the public lens did so for 12/16 and 15/16. Mean target ranks
were 2.875 and 2.8125. These are diagnostic cross-quantization measurements,
not a successful native-NVFP4 reproduction.

A derived paired comparison used the same 4 prompts, 128 token IDs, 63
layers, and 16 observations per layer in each runtime. For the local lens,
NF4-output versus NVFP4-output macro averages were 0.769841 top-1 agreement,
0.816468 top-5 overlap, and 0.974066 target-rank Spearman. The public-lens
control was 0.755952, 0.818452, and 0.979986. This shows that the cross-model
application produced coherent readouts, but it does not override the failed
NVFP4 capture certificate or turn the local artifact into an NVFP4-fitted
lens.

## Commands

The production invocations below are copied from the fit provenance. The
other commands regenerate the recorded reports with their full non-default
arguments.

```bash
scripts/setup_fit.sh

# C32 vectorization diagnostic; --cotangent-batch defaults to the pinned 32.
.venv-fit/bin/python scripts/fit_jlens_nf4.py \
  --work-dir .cache/jlens-nf4-diagnostic-c32-v2 \
  --output .cache/jlens-nf4-diagnostic-c32-v2.pt \
  --provenance .cache/jlens-nf4-diagnostic-c32-v2.json \
  --row-limit 32 \
  --output-dtype float32

# Commit prompt 1, testing the resumable boundary.
.venv-fit/bin/python scripts/fit_jlens_nf4.py \
  --work-dir .cache/jlens-nf4-production-n10-c32 \
  --output .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.pt \
  --provenance .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.provenance.json \
  --max-prompts 1 \
  --output-dtype float32

# Complete prompts 2 through 10 from the bound state.
.venv-fit/bin/python scripts/fit_jlens_nf4.py \
  --work-dir .cache/jlens-nf4-production-n10-c32 \
  --output .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.pt \
  --provenance .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.provenance.json \
  --resume \
  --output-dtype float32

.venv-fit/bin/python scripts/compare_jlens_artifacts.py \
  --local-path .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.pt \
  --local-sha256 54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f \
  --local-provenance .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.provenance.json \
  --output validation/jlens-nf4-vs-public-2026-07-16.json

.venv-fit/bin/python scripts/evaluate_jlens_nf4.py \
  --local-path .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.pt \
  --local-sha256 54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f \
  --local-provenance .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.provenance.json \
  --layers all \
  --positions 16,32,64,96 \
  --top-k 10 \
  --output validation/jlens-nf4-eval-2026-07-16.json
```

The vLLM runner consumes a list of completion prompts rather than the richer
held-out manifest. Materialize the exact 128-token strings from the frozen
IDs before running it:

```bash
.venv-fit/bin/python - <<'PY'
import json
from pathlib import Path
from transformers import AutoTokenizer

snapshot = Path(
    "/home/mark/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/"
    "snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
manifest = json.loads(Path("configs/jlens_nf4_eval_prompts.json").read_text())
prompts = []
for item in manifest["prompts"]:
    token_ids = item["token_ids"]
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.encode(text, add_special_tokens=True) != token_ids:
        raise RuntimeError(f"decode/re-encode mismatch for row {item['row_index']}")
    prompts.append(
        {"id": f"wikitext-validation-row-{item['row_index']}", "text": text}
    )
Path(".cache/jlens_nf4_eval_prompts_nvfp4.json").write_text(
    json.dumps(prompts, indent=2, ensure_ascii=True) + "\n"
)
PY

scripts/run_jlens_nvfp4.sh \
  --prompts-file .cache/jlens_nf4_eval_prompts_nvfp4.json \
  --layers all \
  --positions 16,32,64,96 \
  --top-k 10 \
  --lens-path .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.pt \
  --lens-sha256 54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f \
  --lens-provenance .cache/Qwen3.6-27B-jlens-nf4-n10-fp32.provenance.json \
  --output validation/jlens-nf4-on-nvfp4-2026-07-16.json

scripts/run_jlens_nvfp4.sh \
  --prompts-file .cache/jlens_nf4_eval_prompts_nvfp4.json \
  --layers all \
  --positions 16,32,64,96 \
  --top-k 10 \
  --output validation/jlens-public-on-nvfp4-heldout-2026-07-16.json
```

Both held-out NVFP4 commands are expected to exit nonzero while preserving
their reports, because the recorded strict adapter certificate fails.

## Evidence Hashes

| Evidence | SHA-256 |
| --- | --- |
| C32 diagnostic | `a0ab4be096f9c83bfa989621793eefc080d087ed028681095e5dfb76c74d3302` |
| Local lens artifact | `54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f` |
| Fit provenance | `08dfe022cc74bab226308dcd77a3901ec18806f8832dc8dec46524374fa3b1e7` |
| Geometry report | `c1328c1142087e6bd063d35764cc0782945a617cf99c32847a896faa400f699e` |
| Held-out NF4 report | `9784988b68cebb4b4db994dcf912480bcd6ae0ace9c9e47c0d3562a1e6547448` |
| NF4 lens on NVFP4 report | `1738bd22501a953e6f36a6577dbd6599e09e4b0d81fe028c90022acd796c5a52` |
| Public lens on held-out NVFP4 report | `abbe9538376775eb3185764180366ed2d0b66b533a421708c74a804c2c538309` |

## Limitations

1. This is not a native NVFP4 fit. A strict reproduction still needs
   validated derivatives for the packed ModelOpt forward, including a
   declared surrogate for FP8 activation rounding.
2. Ten prompts are the project's minimum usable scale, not the paper's
   `n=1000` scale. Sampling variance is visible in the early-layer geometry.
3. The public artifact lacks enough provenance to reproduce its exact corpus
   and fit, so numerical equality is neither expected nor testable.
4. The held-out set has only four prompts and four positions. It measures
   teacher-forced readouts, not steering, causal interventions, or downstream
   task accuracy.
5. The long-prompt NVFP4 reports failed their strict capture-adapter
   certificates. Their lens rows remain useful diagnostics but are not
   certified reproduction evidence.
6. MTP was deliberately disabled for vLLM activation capture. Nothing here
   establishes that lens capture works inside speculative MTP execution.
7. The 6.15 GiB artifact remains under `.cache` and is not intended for Git.
   The committed reports and provenance bind its content by SHA-256.
