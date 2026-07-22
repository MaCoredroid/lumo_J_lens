#!/usr/bin/env python3
"""Causal activation-patching harness for the J-space swap reproduction (Phase 0+).

Extends the read-only J-lens capture into a WRITE path: a combined forward hook that both
captures the post-block residual at the final prompt position AND, when armed, injects a delta
into that position's residual (returning the modified (branch_output, residual) tuple, which
vLLM's forward-hook contract substitutes downstream). Single-token factual answers are read as
the greedy next token from a max_tokens=1 prefill — the same forward the capture path validates,
so no mid-decode / CUDA-graph intervention is needed.

Phase 0 sanity (this file's main): given a minimal pair P (->C via B) and P' (->C' via B'),
(1) confirm the model does both hops greedily, (2) zero-ablate a mid layer to prove the write
path changes the output, (3) sweep layers transplanting h_{P'}-h_P into P and report which layers
flip the answer C->C'. No science until this passes. Run via run_jlens_patch.sh (CUDA env).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _rj():
    spec = importlib.util.spec_from_file_location("run_jlens_nvfp4", ROOT / "scripts/run_jlens_nvfp4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RJ = _rj()


def load_llm(*, max_model_len: int = 4096):
    from huggingface_hub import snapshot_download
    from vllm import LLM

    model_path = snapshot_download(_RJ.MODEL_REPO, revision=_RJ.MODEL_REVISION)
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        quantization="modelopt_fp4",
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=1,
        enforce_eager=True,
        enable_chunked_prefill=False,  # keep short prompts in ONE prefill -> last row == final position
        enable_prefix_caching=False,
        language_model_only=True,
        gdn_prefill_backend="triton",
        mamba_cache_mode="align",
        mamba_ssm_cache_dtype="float32",
        kv_cache_dtype="fp8_e4m3",
        attention_backend="TRITON_ATTN",
        limit_mm_per_prompt={"image": 0, "video": 0},
        enable_flashinfer_autotune=False,
        async_scheduling=False,
        seed=0,
    )
    llm.apply_model(_install_patch_hooks)
    return llm


def _install_patch_hooks(model: Any) -> None:
    import torch

    torch.set_grad_enabled(False)
    _, text_model = _RJ._text_model_parts(model)
    model._patch_cap = {}
    model._patch_cap_on = False
    model._patch_inj = {}       # layer -> list of (row_index, delta tensor) to add to branch_output
    model._patch_handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any):
            branch_output, residual = output
            if model._patch_cap_on:
                post = branch_output + residual
                model._patch_cap[layer] = post.detach().float().cpu()  # full (all positions)
            edits = model._patch_inj.get(layer)
            if edits:
                bo = branch_output.clone()
                for pos, delta in edits:
                    bo[pos] = bo[pos] + delta.to(dtype=bo.dtype, device=bo.device)
                return (bo, residual)
            return None

        return hook

    for layer in _RJ.CAPTURE_LAYERS:
        model._patch_handles.append(text_model.layers[layer].register_forward_hook(make_hook(layer)))


def _arm_capture(model: Any, on: bool) -> None:
    if on:
        model._patch_cap = {}  # reset only when arming; disarming must preserve captures for readout
    model._patch_cap_on = on


def _read_captures(model: Any) -> dict[int, Any]:
    return {L: t.clone() for L, t in model._patch_cap.items()}


def _arm_inject(model: Any, inj: dict[int, Any]) -> None:
    model._patch_inj = inj


def greedy_token(llm, token_ids: list[int], text: str, *, capture: bool = False,
                 inject: dict[int, Any] | None = None) -> int:
    """Run a max_tokens=1 prefill -> greedy next-token id. inject = {layer: [(pos, delta), ...]}."""
    from vllm import SamplingParams, TokensPrompt

    llm.apply_model(lambda m: _arm_capture(m, capture))
    llm.apply_model(lambda m: _arm_inject(m, inject or {}))
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=token_ids, prompt=text)],
        SamplingParams(max_tokens=1, temperature=0, seed=0),
        use_tqdm=False,
    )
    llm.apply_model(lambda m: _arm_inject(m, {}))
    llm.apply_model(lambda m: _arm_capture(m, False))
    return outputs[0].outputs[0].token_ids[0]


def capture_all_layers(llm, token_ids: list[int], text: str) -> dict[int, Any]:
    """Return {layer: full post-block residual (all positions)} for one prompt."""
    greedy_token(llm, token_ids, text, capture=True)
    return llm.apply_model(_read_captures)[0]


def _encode(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=True)


def phase0(llm, p_text: str, pprime_text: str, *, layers: list[int]) -> dict[str, Any]:
    import torch

    tok = llm.get_tokenizer()
    p_ids, pp_ids = _encode(tok, p_text), _encode(tok, pprime_text)

    tid_C = greedy_token(llm, p_ids, p_text)
    tid_Cp = greedy_token(llm, pp_ids, pprime_text)
    C, Cp = tok.decode([tid_C]), tok.decode([tid_Cp])

    h_p = capture_all_layers(llm, p_ids, p_text)
    h_pp = capture_all_layers(llm, pp_ids, pprime_text)

    # sanity: zero-ablate the final-position residual at a mid layer -> output must change
    midL = layers[len(layers) // 2]
    zero_delta = {midL: (-h_p[midL][0]).to("cuda")}  # post-block final residual -> ~0
    tid_zero = greedy_token(llm, p_ids, p_text, inject=zero_delta)

    # transplant sweep: inject h_pp - h_p at the final position, per layer
    sweep = []
    for L in layers:
        delta = {L: (h_pp[L][0] - h_p[L][0]).to("cuda", dtype=torch.bfloat16)}
        tid = greedy_token(llm, p_ids, p_text, inject=delta)
        sweep.append({"layer": L, "token": tok.decode([tid]), "flipped_to_Cp": tid == tid_Cp, "changed": tid != tid_C})

    return {
        "p_text": p_text, "pprime_text": pprime_text,
        "C": C, "tid_C": tid_C, "Cp": Cp, "tid_Cp": tid_Cp,
        "both_hops_distinct": tid_C != tid_Cp,
        "zero_ablate_layer": midL, "zero_ablate_token": tok.decode([tid_zero]), "zero_ablate_changed": tid_zero != tid_C,
        "transplant_sweep": sweep,
        "n_layers_flipped_to_Cp": sum(1 for s in sweep if s["flipped_to_Cp"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p", default="Fact: The chemical symbol for the element with atomic number 29 is ")
    ap.add_argument("--pprime", default="Fact: The chemical symbol for the element with atomic number 26 is ")
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(10, 48, 2)))
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/jspace-phase0-sanity.json")
    args = ap.parse_args()

    llm = load_llm()
    result = phase0(llm, args.p, args.pprime, layers=args.layers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "transplant_sweep"}, indent=2))
    print("transplant sweep (layer -> token, flipped?):")
    for s in result["transplant_sweep"]:
        mark = " <== FLIP to C'" if s["flipped_to_Cp"] else (" (changed)" if s["changed"] else "")
        print(f"  L{s['layer']:2d} -> {s['token']!r}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
