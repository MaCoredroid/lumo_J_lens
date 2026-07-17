# Qwen3.6-27B NVFP4/FP8-STE Jacobian Lens

Date: 2026-07-17

## Status

The native NVIDIA production fit completed on the local RTX 5090. It used the
same pinned ModelOpt checkpoint and compiled vLLM target-model graph as the
serving profile. It did not substitute the BF16 source model, bitsandbytes NF4,
or the Apple MLX port for the fitting forward.

Run `20e4bc8c-9fed-4513-b548-9727f9686222` discarded the earlier exploratory
captures, recaptured all ten frozen prompts, and reproved each one under the
hardened `model.identity`, metadata, three-shard, prompt-manifest, runtime, and
source-contract bindings. Every prompt had exact endpoint generation parity,
688/688 shared internal tensors bit-exact, 432/432 observer-only compiled
boundaries present, and 785/785 replay parameters equal by name, shape, dtype,
and content hash. Each prompt committed 20 contiguous estimator chunks.

The run completed all 5,120 rows for every source layer `0..62`, targeting
block 63, and finalized 63 finite little-endian FP32 `[5120,5120]` matrices. It
took 47,577.883 seconds (13:12:57.9) and peaked at
8,936,882,688/11,404,312,576 CUDA bytes allocated/reserved. The exported
6,606,046,478-byte checkpoint has SHA-256
`82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057`
and passed the exact production verifier plus both upstream load APIs.

Dense geometry against the public artifact and paired held-out readout also
completed. Geometry measured global Frobenius cosine `0.732877` and mean
per-layer cosine `0.822360`. Across 1,008 paired readout observations,
native/public target-rank Spearman was `0.902843`, top-1 agreement `0.412698`,
top-5 overlap `0.493651`, and target-score RMSE `2.780105`. Both independently
run adapter certificates failed with identical pre-lens residual manifests and
reconstruction values. Adapter status is therefore reported separately from
lens-quality metrics.

| Level | Status | Meaning |
|---|---|---|
| Public `n=1000` Qwen lens on NVFP4 residuals | Complete | FP16 lens; fit-time precision and quantization unpublished |
| WeZZard Apple reference | Reference only | MLX 4-bit forward and custom Metal GDN backward |
| Native NVIDIA production capture | Passed on all ten prompts | Exact deployed forward values; exact endpoint parity; 688 shared, 432 observer-only, and 785 replay-parameter checks per prompt |
| All-layer reverse replay | Complete | All 5,120 rows for source layers `0..62`, targeting block 63 |
| Dense native `n=10` lens | Complete and verified | 63 full `5120 x 5120` FP32 means; upstream loaders passed |
| Geometry and paired held-out readout | Reported | Descriptive native/public measurements; both independent adapter certificates failed identically |

## What Is Different From The Public And Apple Paths

The public Qwen3.6 lens distributed in `neuronpedia/jacobian-lens` stores FP16
matrices, but does not publish the fit-time model precision or quantization.
An unquantized BF16 source is plausible because the canonical Qwen checkpoint
is BF16, but is not verified. When that artifact is read against NVIDIA
residuals, it is a public control application, not an NVFP4 fit.

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
  byte length, and SHA-256 of row-major bytes. All were bit-exact in every one
  of the ten production 128-token prompt proofs. These comprise 624 GDN and 64
  full-attention tensors per prompt.
- 432 required tensors exist only in the observer artifact: 304 compiled
  linear outputs, 64 SwiGLU outputs, and 64 post-block residuals. They cannot
  be directly compared to absent baseline tensors. Exact generation-record
  parity and the direct shared-tensor proof discharge the observer modification
  indirectly for this pinned prompt and runtime shape.
- All 785 replay parameters match by name, shape, dtype, and content hash
  between the two isolated captures.

This does not prove MTP decode, a prompt outside the frozen corpus, another
sequence shape, or an unmodified observer graph. Production repeated the
isolated proof for every fit prompt and rehashed the retained observer payload
immediately before each commit.

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

The following engineering and production gates ran on the real local pinned
checkpoint and RTX 5090. The prompt-0 probe rows remain useful operator tests;
the completed production rows are the evidence for the final artifact:

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
| Production isolated capture pairs | 10/10 exact generation records; 688/688 shared tensors bit-exact, 432/432 observer-only boundaries complete, and 785/785 replay parameters equal for every prompt |
| Production row commits | 10/10 prompts; 20 contiguous chunks per prompt; all 5,120 rows for all 63 source matrices |
| Production finalization | 63/63 finite FP32 `[5120,5120]` means; aggregate layer SHA-256 `a4c2adc7be15232db0e5a8840a6442248caa80a363c0c5239a1ee248f36fb3b4` |
| Export and load | Exact artifact verifier passed; `JacobianLens.load` and `JacobianLens.from_pretrained` passed |

The exploratory all-layer row computed one of 5,120 output rows for every
source matrix. The production run did not adopt that capture; it performed
fresh strict captures and completed every row for every matrix and prompt.

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

The measured production identities are:

| Record | Bytes | SHA-256 |
|---|---:|---|
| `state.json` | 329,400 | `f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6` |
| `final-mean/metadata.json` | 988,263 | `289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601` |
| Exported `.pt` | 6,606,046,478 | `82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057` |

The raw 63-matrix mean totals 6,606,028,800 logical bytes. Its aggregate layer
SHA-256 is
`a4c2adc7be15232db0e5a8840a6442248caa80a363c0c5239a1ee248f36fb3b4`.
The final cumulative sum hashes to
`0e81bf4b5118f664bbb14f2858f5d654ba780450b20148178e4190d4db0c40e3`;
the ordered committed-prompt records hash to
`a1690ab9e88cff53a2eba407195ced52e6908208fedffed68819ee47c1a888c1`.
The final metadata payload hash is
`7b96a49e209e2e3008531fc7d7ac46de1582eb824cba50c95c3b1d7302bb8b66`.
The frozen fit contract hashes to
`7944ea163b548edc3372fa67242fbbcfbe0a5abbe95c04ce4a378107ebe03dd0`;
its 13 bound source files hash in aggregate to
`dbe1f28bbd829fa30cb48b4c593419de205c440d195c12bed398c0036ed16400`.

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

The committed compact evidence records are:

- `validation/jlens-nvfp4-ste-fit-state-2026-07-17.json`
- `validation/jlens-nvfp4-ste-final-metadata-2026-07-17.json`
- `validation/jlens-nvfp4-ste-run-progress-2026-07-17.json`
- `validation/jlens-nvfp4-ste-export-2026-07-17.json`
- `validation/jlens-nvfp4-ste-artifact-verification-2026-07-17.json`
- `validation/jlens-nvfp4-ste-upstream-load-2026-07-17.json`
- `validation/jlens-nvfp4-ste-vs-public-2026-07-17.json`
- `validation/jlens-native-nvfp4-ste-on-nvfp4-heldout-2026-07-17.json`
- `validation/jlens-public-schema3-on-nvfp4-heldout-2026-07-17.json`
- `validation/jlens-nvfp4-ste-vs-public-heldout-2026-07-17.json`

The exact verifier checked all 63 exported tensors against the raw means, all
ten prompt commits and 20 contiguous chunks per prompt, finiteness, capture
proofs, model/source/contract hashes, and completed state. The upstream smoke
used `jlens` 0.1.0 at commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`; both
`JacobianLens.load` and `JacobianLens.from_pretrained` passed.

Run the same upstream API smoke sequentially so only one 6.15 GiB in-memory
lens is live at a time:

```bash
# One-time setup for the pinned environment that contains upstream jlens.
scripts/setup_fit.sh

LENS="$LENS" .venv-fit/bin/python - <<'PY'
import gc
import os

import torch
from jlens import JacobianLens

path = os.environ["LENS"]
for name, loader in (
    ("JacobianLens.load", lambda: JacobianLens.load(path)),
    (
        "JacobianLens.from_pretrained",
        lambda: JacobianLens.from_pretrained(path),
    ),
):
    lens = loader()
    assert lens.n_prompts == 10
    assert lens.d_model == 5120
    assert lens.source_layers == list(range(63))
    assert all(tuple(lens.jacobians[i].shape) == (5120, 5120) for i in range(63))
    assert all(lens.jacobians[i].dtype == torch.float32 for i in range(63))
    print(f"{name}: passed")
    del lens
    gc.collect()
PY
```

Run the descriptive dense geometry comparison separately:

```bash
.venv-vllm/bin/python scripts/compare_jlens_artifacts.py \
  --local-kind nvfp4-ste \
  --local-path "$LENS" \
  --local-sha256 82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057 \
  --local-provenance .cache/nvfp4_ste_fit/final-mean/metadata.json \
  --local-state "$STATE" \
  --local-state-sha256 f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6 \
  --row-chunk 16 \
  --output validation/jlens-nvfp4-ste-vs-public-2026-07-17.json
```

No global similarity threshold was chosen after seeing the geometry. The
63-layer comparison measured:

| Geometry metric | Value |
|---|---:|
| Global Frobenius cosine | `0.7328770738661481` |
| Mean per-layer Frobenius cosine | `0.8223602375534815` |
| Global relative Frobenius difference | `0.9345964627007955` |
| Mean cosine over all 322,560 rows | `0.791449281794253` |

For held-out readout, the native and public schema-3 commands used identical
frozen prompts, layers, positions, target model, and runtime. The untracked
`$PROMPTS` JSON is materialized from the exact 128-token ID arrays in
`configs/jlens_nf4_eval_prompts.json` using the pinned NVIDIA checkpoint's
tokenizer; each decoded string must re-encode to the original ID array with
`add_special_tokens=True` before evaluation. Run both evaluations and compare
them with:

```bash
PROMPTS=.cache/jlens_nvfp4_heldout_prompts_2026-07-17.json
export PROMPTS

.venv-vllm/bin/python - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

revision = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
snapshot = Path(snapshot_download(
    "nvidia/Qwen3.6-27B-NVFP4",
    revision=revision,
    local_files_only=True,
))
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
manifest = json.loads(Path("configs/jlens_nf4_eval_prompts.json").read_text())
prompts = []
for item in manifest["prompts"]:
    token_ids = item["token_ids"]
    assert len(token_ids) == item["token_count"] == 128
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    assert tokenizer.encode(text, add_special_tokens=True) == token_ids
    prompts.append({
        "id": f"wikitext-validation-row-{item['row_index']}",
        "text": text,
    })

Path(os.environ["PROMPTS"]).write_text(
    json.dumps(prompts, indent=2, ensure_ascii=True) + "\n"
)
PY

NATIVE_RC=0
scripts/run_jlens_nvfp4.sh \
  --lens-kind nvfp4-ste \
  --lens-path "$LENS" \
  --lens-sha256 82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057 \
  --lens-provenance .cache/nvfp4_ste_fit/final-mean/metadata.json \
  --lens-state "$STATE" \
  --lens-state-sha256 f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6 \
  --prompts-file "$PROMPTS" \
  --layers all --positions 16,32,64,96 --top-k 10 \
  --max-model-len 256 --gpu-memory-utilization 0.82 \
  --output validation/jlens-native-nvfp4-ste-on-nvfp4-heldout-2026-07-17.json \
  || NATIVE_RC=$?

PUBLIC_RC=0
scripts/run_jlens_nvfp4.sh \
  --lens-kind public \
  --prompts-file "$PROMPTS" \
  --layers all --positions 16,32,64,96 --top-k 10 \
  --max-model-len 256 --gpu-memory-utilization 0.82 \
  --output validation/jlens-public-schema3-on-nvfp4-heldout-2026-07-17.json \
  || PUBLIC_RC=$?
```

Both commands write complete reports and exit 1 because their independently
derived adapter certificates fail. Do not run this pair under `set -e` without
capturing each status. Preserve the reports, verify that exit 1 agrees with
`status: failed`, and then run the offline comparator:

```bash
.venv-vllm/bin/python scripts/compare_jlens_nvfp4_reports.py \
  --native-report validation/jlens-native-nvfp4-ste-on-nvfp4-heldout-2026-07-17.json \
  --public-report validation/jlens-public-schema3-on-nvfp4-heldout-2026-07-17.json \
  --output validation/jlens-nvfp4-ste-vs-public-heldout-2026-07-17.json
```

The comparison covered Wikitext validation rows 3, 18, 42, and 49 at positions
16, 32, 64, and 96 for every layer `0..62`, or 1,008 observations. It measured:

| Held-out native/public metric | Value |
|---|---:|
| Target-rank Spearman | `0.902843338526047` |
| Top-1 agreement | `0.4126984126984127` |
| Mean top-5 overlap fraction | `0.4936507936507937` |
| Target-score RMSE | `2.780105132813771` |
| Native target top-1 / top-5 rate | `0.06349206349206349` / `0.11904761904761904` |
| Public target top-1 / top-5 rate | `0.054563492063492064` / `0.1378968253968254` |

Both independently executed adapter certificates returned `status: failed`:
rows 3, 18, and 49 had full-logit max error `0.125 > 0.0625`; row 42 was at
the `0.0625` limit. Row 18 also failed final-norm max and top-5 prefix. All
four residual-capture manifests and every logit-lens baseline field matched
exactly across the runs. The pairing command preserves each adapter outcome
separately from the lens readout metrics; this failed adapter certificate is
not a lens-quality verdict.

| Downstream record | Bytes | SHA-256 |
|---|---:|---|
| Dense geometry | 94,335 | `43b7431bdc006c1e097e7c187a23671d95f4807c0fe701078ee7345fafdb6fa2` |
| Native held-out | 2,437,812 | `17ecf282aadd26db281bc2ac5817769ddd1a82ba6b7d0474db386117490e9b90` |
| Public held-out control | 2,435,472 | `fb94cf4f84d110d2b52f473695a675e3e88341836c949658e76fae29a6ebc486` |
| Paired summary | 767,523 | `2fe0d1e6e564119dcb757ec7073da9df051d5c496ed494bb38cd870e32ff6f02` |

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

At batch 256, 20 row batches cover one prompt. The pre-run estimator rate
projected about 76.1 minutes per prompt and 12.7 hours for ten prompts. The
completed production state measured 47,577.883 seconds (13:12:57.9), including
isolated baseline/observer captures, content hashing, 200 row-chunk commits,
ten prompt commits, and streaming finalization. Maximum CUDA allocated/reserved
was 8,936,882,688/11,404,312,576 bytes (8.323/10.621 GiB).

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

The finalizer checked every output chunk for finiteness. Its metadata reports
`n_prompts=10`, sources `0..62`, target 63, per-layer shape/dtype/size/content
hashes, an aggregate layer-manifest hash, the prompt/capture bindings,
runtime/source provenance, and the declared forward/backward contract. That
metadata and the exported artifact passed independent verification, so the
correct result label is **completed native NVFP4/FP8-STE `n=10` fit**.

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
6. Geometry and held-out readout are descriptive comparisons on one four-prompt
   corpus, not broad behavioral-equivalence or downstream-task guarantees.
7. The public control contains 1,000 prompts and FP16 stored matrices, but its
   fit-time model precision and quantization are unpublished. The native lens
   contains ten prompts and FP32 matrices; direct equality is not expected.
8. Residual-adapter certification is independent of the selected lens and must
   not be treated as a lens-quality metric.
