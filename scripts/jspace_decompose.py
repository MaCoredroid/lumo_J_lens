#!/usr/bin/env python3
"""J-space / non-J-space decomposition of residual directions (Phase 3).

The J-lens reads a residual v as `v @ J[L].T`, i.e. via inner products with the ROWS of J[L].
So the subspace the lens can see ("J-space") is row(J[L]); its orthogonal complement is the
right null space (`J[L] @ v_perp == 0`) — the "non-J-space" that carries most of the residual
variance but is invisible to the lens. This module builds the projector P_J onto row(J[L]) via
SVD, splits a delta into v_J = P_J v and v_perp = v - v_J, norm-matches them for a fair causal
comparison, and generates norm-matched random controls inside row(J) and ker(J). Caches per-layer
projector rank. CPU or GPU torch. No model needed.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def j_checkpoint_path() -> str:
    hits = glob.glob(
        str(Path.home() / ".cache/huggingface/hub/models--neuronpedia--jacobian-lens/**/Qwen3.6-27B_jacobian_lens_n1000.pt"),
        recursive=True,
    )
    if not hits:
        raise FileNotFoundError("jacobian-lens checkpoint not found")
    return sorted(hits)[0]


def load_J(layer: int, *, device: str = "cpu"):
    import torch

    ckpt = torch.load(j_checkpoint_path(), map_location="cpu", weights_only=True, mmap=True)
    return ckpt["J"][layer].to(device=device, dtype=torch.float32)


def row_space_projector(J_L, *, energy: float = 0.99, rel_sigma: float = 1e-2) -> tuple[Any, int, Any]:
    """P_J projects onto row(J_L). Effective rank = min rank hitting `energy` OR sigma>rel_sigma*sigma_max."""
    import torch

    # J_L = U diag(S) Vh ; rows of Vh are the right singular vectors spanning row(J_L)
    U, S, Vh = torch.linalg.svd(J_L, full_matrices=False)
    total = torch.sum(S * S)
    cum = torch.cumsum(S * S, dim=0) / total
    r_energy = int(torch.searchsorted(cum, energy).item()) + 1
    r_sigma = int((S > rel_sigma * S[0]).sum().item())
    r = max(1, min(r_energy, r_sigma))
    Vr = Vh[:r].T.contiguous()          # (d_model, r), columns = right singular vectors
    P_J = Vr @ Vr.T                     # (d_model, d_model) projector onto row(J_L)
    return P_J, r, S


def decompose(v, P_J) -> tuple[Any, Any]:
    """Split v into (v_J in row(J), v_perp in ker(J))."""
    v_J = P_J @ v
    return v_J, v - v_J


def norm_match(a, b) -> tuple[Any, Any]:
    """Rescale both to a common norm (the geometric-mean magnitude) for a fair-magnitude comparison."""
    import torch

    na, nb = torch.linalg.norm(a), torch.linalg.norm(b)
    target = torch.sqrt(na * nb).clamp_min(1e-12)
    return a * (target / na.clamp_min(1e-12)), b * (target / nb.clamp_min(1e-12))


def random_in_subspaces(P_J, ref_norm, *, seed_vec) -> tuple[Any, Any]:
    """A random direction placed in row(J) and one in ker(J), each scaled to ref_norm."""
    import torch

    r = seed_vec  # a deterministic seed vector (e.g. a shuffled delta) to avoid RNG nondeterminism
    r_J = P_J @ r
    r_perp = r - r_J
    out = []
    for x in (r_J, r_perp):
        n = torch.linalg.norm(x).clamp_min(1e-12)
        out.append(x * (ref_norm / n))
    return out[0], out[1]


def verify_perp(J_L, v_perp) -> float:
    """‖J_L @ v_perp‖ should be ~0 if v_perp is truly in the null space."""
    import torch

    return float(torch.linalg.norm(J_L @ v_perp))


def main() -> int:
    import argparse

    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, nargs="+", default=[16, 24, 32, 40, 44, 47])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    print(f"{'layer':>5} {'rank':>6} {'d_model':>7} {'sigma_max':>10} {'sigma_min/max':>13}")
    for L in args.layers:
        J_L = load_J(L, device=args.device)
        P_J, r, S = row_space_projector(J_L)
        # sanity: P_J idempotent + a random v_perp is killed by J_L
        v = torch.randn(J_L.shape[1], device=J_L.device, dtype=J_L.dtype)
        _, v_perp = decompose(v, P_J)
        print(f"{L:>5} {r:>6} {J_L.shape[1]:>7} {float(S[0]):>10.3f} {float(S[-1]/S[0]):>13.2e}  "
              f"‖J·v_perp‖={verify_perp(J_L, v_perp):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
