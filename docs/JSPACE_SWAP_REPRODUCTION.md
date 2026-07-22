# Reproducing the J-space causal-mediation result on two-hop trials (2026-07-21)

**Claim reproduced.** Anthropic's jacobian-lens / global-workspace result: on two-hop trials,
swapping the **J-space** component of an intermediate-concept direction flips the model's answer,
while swapping the **non-J-space** component (which carries most of the residual variance) does
not — i.e. the small-variance J-space component causally mediates the two-hop reasoning (cited
~61% vs ~28%).

**Our result:** on Qwen3.6-27B-NVFP4 with Anthropic's own two-hop data, the norm-matched
**J-space swap flips the answer to the counterfactual target while the norm-matched non-J-space
swap flips it 0%** — the small-variance J-space component carries the entire causal effect. The
qualitative mechanism reproduces robustly (difference CI excludes zero); absolute rates differ.

## Method

- **Data.** Anthropic's `data/evaluations/lens-eval-multihop.json` (93 two-hop items:
  `{prompt, target C, intermediate B}`, e.g. *"…element with atomic number 79 is"* → gold → Au).
  Gated to items the model solves greedily with a single-token answer (49/93). Minimal pairs
  P/P′ (same length, ≤3 differing cue tokens, distinct answers) → **8** clean pairs. Plus an
  **expanded** set of the same atomic-symbol template over 22 two-digit elements → **72** pairs.
- **Concept-swap direction.** Empirical: at each layer L and differing cue position p,
  Δ = h_{P′}[L,p] − h_P[L,p] (the residual difference that carries the B→B′ concept change).
- **J / non-J split.** J-space = row(J[L]) via SVD (99%-energy effective rank: 599/5120 at L8
  growing to 3657/5120 at L47 — a small fraction of the residual early). Δ_J = P_J Δ,
  Δ_⊥ = Δ − Δ_J. **Norm-matched** (‖Δ_J‖=‖Δ_⊥‖) so the comparison is at equal perturbation size —
  the load-bearing control, since J-space is deliberately small-variance.
- **Causal patch.** A new write hook injects the delta at the differing positions across the
  middle-layer band **L16–40** during a `max_tokens=1` prefill; the answer is the greedy next
  token. (Final-position patching does **not** work — the answer is re-derived from the explicit
  cue via attention; subject-position band patching is required. This is the key methodological
  finding, validated in Phase 0.)
- **Controls.** full Δ, raw (non-norm-matched) components, norm-matched random directions inside
  row(J) and ker(J), and no-op.

## Results — strict flip rate (answer → the counterfactual target C′)

| condition | Anthropic data (n=8) | expanded atomic (n=72) |
|---|---:|---:|
| full Δ | 0.375 | 0.083 |
| **Δ_J, norm-matched** | **0.375** | **0.125** |
| **Δ_⊥, norm-matched** | **0.000** | **0.000** |
| Δ_J raw | 0.375 | 0.125 |
| Δ_⊥ raw | 0.000 | 0.000 |
| random in row(J), norm-matched | 0.000 | 0.000 |
| random in ker(J), norm-matched | 0.000 | 0.000 |
| no-op | 0.000 | 0.000 |

Bootstrap 95% CI on (Δ_J − Δ_⊥), strict:
- Anthropic n=8: **[0.125, 0.750]**
- expanded n=72: **[0.056, 0.208]**

Both exclude zero. Per-trial, on the n=8 set the J-space swap flips **exactly the pairs the full
swap flips** (3/8) and no others; Δ_⊥ flips none.

## Reading

- **The J-space component mediates; the non-J-space does not.** Δ_J reproduces the full swap's
  causal effect (equal or greater strict flip rate), while the norm-matched Δ_⊥ — carrying most
  of the residual variance — is causally **inert** (0%, identical to random directions and no-op).
  This is the paper's core mechanism, reproduced on a different model.
- **Absolute rates are lower than the cited 61%/28%.** Expected: different model (Qwen3.6-27B-NVFP4
  vs theirs), a different patching protocol (subject-position band, empirical delta vs their probe
  construction), and answer-space effects — atomic symbols are a large answer space, so landing on
  the *exact* counterfactual symbol is hard (the "loose" ≠C metric saturates at ~0.97: any band
  perturbation disrupts the answer, but rarely onto C′). The month-attribute pairs (small answer
  space) give the higher 0.375.
- **The direction and significance are robust**, not the absolute magnitude.

## Honest limits

- Not a bit-for-bit reproduction: we reverse-engineered the patching protocol (the paper's exact
  positions / probe construction / prompt pairs were not in hand), and the absolute rates differ.
- Small clean-pair yield on Anthropic's exact set (n=8); the expanded set (n=72) is same-template
  instances, not new templates.
- Empirical Δ (P/P′ residual difference) conflates the cue change with the concept; a residual-
  trained concept probe would be cleaner but needs a training pipeline.
- 99%-energy J-space truncation is a hyperparameter; ‖J·Δ_⊥‖ is small-but-nonzero (Δ_⊥ is read
  *weakly*, not exactly zero).

## Artifacts / scripts

`run_jlens_patch.py` (causal write hook), `jspace_decompose.py` (SVD J/non-J split),
`run_jspace_swap_experiment.py` (the swap), `analyze_jspace_swap.py` (bootstrap),
`build_twohop_expanded.py`. Results: `artifacts/jspace-swap-experiment.json` (n=8),
`artifacts/jspace-swap-expanded.json` (n=72), Phase-0 sanity `artifacts/jspace-phase0-sanity.json`.
