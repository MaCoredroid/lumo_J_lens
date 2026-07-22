# J-lens interpretability: from correlational readout to causal test (summary, 2026-07-22)

One coherent arc, three results. We built a concept lens over a Qwen SWE-agent's own residual
stream, asked whether it is *faithful* to the model's reasoning, then whether the concepts it reads
*causally* drive behavior. The headline: **the lens reads reasoning correlationally; those concepts
causally mediate a factual answer but do not steer an agent's action.** Correlational faithfulness ≠
causal control.

---

## Result 1 — a trustworthy *correlational* concept lens

*Does the residual state encode the concept the model's chain-of-thought expresses?*

- **Cohort faithfulness = 0.41** centered (n=532, full coverage), up from 0.31, via three levers:
  read at the late-layer peak (L41: +0.07, the biggest lever — the signal is late-layer localized),
  a richer concept vocab (+0.03), and full 535-turn coverage (bounded context; validated −0.007 vs
  full). Now above the majority-class baseline.
- **Naive per-concept probes fail, but a *learned* linear probe over the lens features works** —
  held-out task_resolution 0→0.80, dependency 0→0.63, source_localization 0.44→0.66. Most fine
  families become trustworthy; repair/edit stay entangled.
- **Trustworthy granularity = 3 concepts** (locate / modify_code / assess): held-out **0.74**,
  per-family recall 0.76 / 0.83 / 0.68. A disagreement detector on top flags candidate
  surface–internal divergences.

**Takeaway:** the lens genuinely reads *what mode of reasoning the agent is in*, trustworthy at a
coarse granularity. This is a **read-out** — it reflects reasoning.
(Reports: `JLENS_COHORT_FAITHFULNESS_REPORT_v4.md`, `JLENS_TRUSTWORTHY_CONCEPT_LENS.md`.)

---

## Result 2 — concepts *causally mediate* a factual answer (Anthropic reproduction)

*On two-hop facts, does swapping the J-space component of the intermediate concept flip the answer?*
(Anthropic's cited result: J-space ~61% vs non-J ~28%.)

Built the missing causal capability — a **write hook** that patches the residual mid-forward — and
split concept directions into **J-space** (row(J), what the lens reads; a small fraction of the
residual early: rank 599/5120 at L8) vs **non-J-space** (the high-variance remainder).

| strict flip → counterfactual answer, norm-matched | Anthropic data (n=8) | expanded (n=72) |
|---|---:|---:|
| **J-space swap** | **0.375** | **0.125** |
| **non-J-space swap** | **0.000** | **0.000** |
| random in row(J)/ker(J), no-op | 0.000 | 0.000 |

Bootstrap 95% CIs on (J − non-J): **[0.125, 0.750]** (n=8), **[0.056, 0.208]** (n=72) — both exclude
zero. The J-space component carries the *entire* causal effect (flips exactly when the full swap
flips); the large-variance non-J remainder is inert. **Anthropic's mechanism reproduced** on a
different model; absolute rates differ from 61/28 (different model, reverse-engineered protocol,
answer-space size). (`JSPACE_SWAP_REPRODUCTION.md`.)

---

## Result 3 — concepts do *not* steer the agent's action (decisive null)

*Does swapping the same J-space concept redirect the SWE agent's next tool/command?*

Two loci (end-of-thinking; the command-choice point), trained steering vectors (CAA + shrinkage-LDA),
and a sensitive logit metric (`logprob(grep) − logprob(sed)`, baseline +9.8 on locate turns).

- **The concept vector does not steer the action.** The CAA J-space vector is **identical, per turn,
  to a random row-space vector** (both −9.81) — no concept-specificity; it merely garbles the output
  (output-sensitive), never biasing toward a modify command. J-space ≈ non-J-space; neither steers.
- The one positive signal (LDA J-space, −16.4) is **task-specific (django only) and overfit** (60
  samples in 5120-dim), not general steering.

**Takeaway:** even a trained vector + sensitive metric confirm the null. Unlike a factual answer —
the immediate, cleanly-mediated next token — the agent's action is **context-anchored** across the
trajectory and not hijackable by a concept vector. (`JSPACE_STATE_ACTION.md`.)

---

## The central finding

| regime | is the concept **encoded**? | does it **causally control** the output? |
|---|---|---|
| factual two-hop | yes (lens reads it) | **yes** — J-space swap flips the answer |
| agentic SWE action | yes (lens reads it, ~0.74) | **no** — no reliable concept-steering of the action |

A fact has a clean concept→answer channel: the intermediate concept *is* the immediate causal
determinant, so a small J-space edit flips it. An agent's action is determined by concrete task
state (which bug, which files, prior tool outputs) spread across the whole trajectory — the concept
lens reads the reasoning mode but is not a lever on behavior. **This is the caution the whole line
delivers: reading a concept faithfully from activations does not license steering behavior through
it.** And it cleanly separates the two regimes where naive intuition would conflate them.

## Infrastructure built (reusable)

- `run_jlens_patch.py` — causal **write hook** (chunk-aware injection, memory-safe for 28k-token
  agent prompts) + capture + next-token logprobs.
- `jspace_decompose.py` — SVD row(J) projector; J-space vs non-J split, norm-match, subspace random
  controls.
- Swap/steer harnesses + analyzers; the concept lens, learned probe, and disagreement detector.

## Honest limits & open threads

- Factual repro is qualitative (mechanism), not the exact 61/28 — needs trained residual concept
  probes + Anthropic's precise protocol.
- The state→action null is "*this* concept vector doesn't steer," not "nothing can": the overfit
  **LDA-django** hint suggests a *task-tuned* direction might steer — one run would characterize it.
- Small n in the causal tests; coarse SWE action space; single-locus prefill injection.
