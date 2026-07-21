# J-lens: a trustworthy concept lens + meaningful disagreement (2026-07-21)

**Goal.** Tighten the weak concept families so the lens is *trustworthy per family* — then a
lens-vs-CoT disagreement becomes a real signal (candidate unfaithfulness / surface–internal
divergence) rather than noise.

**Outcome.** Per-concept probes for the fine families **do not work** — the distinctions aren't
in the residual. But at the granularity the residual *does* support — **3 concepts: locate /
modify_code / assess** — the lens is trustworthy on held-out tasks (per-family recall 0.68–0.83),
and a disagreement detector on top of it surfaces meaningful divergences.

## Finding 1 — per-concept probes fail (the fine taxonomy over-fractions the residual)

Two independent lines of evidence:

- **Confusion (L44):** the weak fine families fail by *adjacent-blob confusion*, not noise.
  source_edit→substitution 0.47, repair→substitution 0.30; `substitution` is a magnet for the
  whole code-modification blob; the check/success families collapse together.
- **Data-driven forms, honest 10/10 train/test task split:** deriving discriminative single-token
  forms per family from the tagged CoT corpus and evaluating on held-out tasks makes it **worse**,
  not better: **0.20 vs the v2 vocab's 0.39** — the discriminative forms wreck the families v2 got
  right (verification 0.59→0.09) while only marginally helping weak ones. Better vocabulary cannot
  recover a distinction the residual does not encode.

So repair-vs-edit-vs-substitution is one thing to the model ("I'm changing this code"); the
14-family taxonomy asks the lens for splits that aren't there.

## Finding 2 — the trustworthy granularity is 3 concepts (held-out 0.74)

Collapsing to the granularity the residual supports, evaluated on **held-out test tasks** at L44:

| granularity | held-out faithfulness | per-family recall |
|---|---:|---|
| 14 fine families | 0.39 | weak families 0.0–0.18 |
| 5 super-concepts | 0.58 | check/success/fault untrustworthy |
| **3 concepts: locate / modify_code / assess** | **0.74** | **locate 0.76 · modify_code 0.83 · assess 0.68** |

Every concept recalls **0.68–0.83 out-of-sample** — the lens is trustworthy per family at this
granularity. The three concepts:
- **locate** — where is the problem (localization, identifier, located-source)
- **modify_code** — change it (edit, repair, substitution)
- **assess** — check / validate / confirm / resolve / diagnose-fault

## The payoff — a meaningful disagreement detector

`concept_disagreement.py` reads each turn's baseline-centered top-1 concept and compares it to the
CoT's concept, ranking disagreements by a confidence margin. On all 532 comparable turns:

- **Agreement 0.72**; 147 disagreements.
- Direction skew: `modify_code` is over-predicted (assess→modify_code 51, locate→modify_code 31) —
  the residual leans toward "changing code."
- **The confident disagreements are inspectable and often real.** The top ones are frequently cases
  where the lens is *right* and the turn's single tag missed it, because the **end-of-thinking state
  anticipates the next action mode**:
  - scikit-12585 t05 — tag `locate`, lens `modify_code`: *"…I need to **add** `or isinstance(estimator, type)` to this."*
  - matplotlib-26291 t19 — tag `assess`, lens `modify_code`: *"…let me use **sed** to make the **change** instead."*
  - xarray-4094 t20 — tag `assess`, lens `modify_code`: *"All test cases pass. Now let me **apply the fix** to the actual source file."*

  In each, the CoT text at the boundary has already turned to *modifying*, and the residual encodes
  that — the lens catches an intent the turn-level label smeared over.

This is the deliverable: at a trustworthy granularity, a confident disagreement is a **meaningful
signal** — a triage queue of turns where the internal state and the stated reasoning diverge, for
review. It is triage, not verdict: the lens still errs ~25%, and some disagreements are label
transitions rather than true divergences.

## Honest limits

- **Triage, not verdict.** ~25% of turns are lens error; a single disagreement is a candidate, not
  proof. Confidence-ranking concentrates the real signal at the top.
- **Boundary effect.** End-of-thinking often captures the *next* mode (locate→modify pivots), so
  some "disagreements" are the label lagging the state, not unfaithfulness. This is itself useful
  (the lens sees the pivot) but must not be read as the model lying.
- **modify_code bias** — the residual over-encodes code-modification; centering mitigates but does
  not remove it.
- Coarser than hoped: we cannot see repair-vs-edit, or the fine assess sub-modes.

## Verdict

The lens is now **trustworthy per family at the 3-concept level (locate / modify_code / assess,
held-out 0.68–0.83)**, and disagreements at that level are meaningful — demonstrated by a detector
whose top hits are genuine boundary-level divergences. Making the *fine* families trustworthy is not
possible with this lens: those distinctions are not represented in the residual.

Artifacts: `cohort-perfamily-trust-eval-L44.json`, `cohort-concept-disagreements-L44.json`,
`cohort-faithfulness-confusion-L44.json`. Scripts: `perfamily_trust_eval.py`,
`concept_disagreement.py`, `datadriven_concept_forms.py`, `superconcept_faithfulness.py`.
Prior: `JLENS_COHORT_FAITHFULNESS_REPORT_v4.md`.
