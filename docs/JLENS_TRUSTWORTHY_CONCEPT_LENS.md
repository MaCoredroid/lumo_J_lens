# J-lens: a trustworthy concept lens + meaningful disagreement (2026-07-21)

**Goal.** Tighten the weak concept families with per-concept probes so the lens is *trustworthy
per family* — then a lens-vs-CoT disagreement becomes a real signal (candidate unfaithfulness /
surface–internal divergence) rather than noise.

**Outcome.** A **learned** per-concept probe over the lens features **works**: it tightens almost
every weak family out-of-sample (task_resolution 0→0.80, dependency 0→0.63, broad_success
0.13→0.55, failure_confirmation 0.20→0.42, source_edit 0.18→0.35, source_localization 0.44→0.66),
so most fine families are now trustworthy per family. On that probe, a confidence-ranked
disagreement detector surfaces meaningful divergences. (An earlier *naive* readout suggested the
fine distinctions weren't recoverable — that was a method artifact, corrected below.)

## The method matters: naive readout fails, learned probe succeeds

Two readouts over the **same** lens features (per-turn scored-token logprobs), same honest
task-split evaluation, opposite conclusions:

| per-family recall, held-out | source_loc | verif | source_edit | repair | broad_success | dep | task_res |
|---|---:|---:|---:|---:|---:|---:|---:|
| **naive** (hand/data forms, argmax-of-means) | 0.44 | 0.59 | 0.18 | 0.10 | 0.13 | 0.00 | 0.00 |
| **learned probe** (LogReg over lens features) | **0.66** | 0.57 | **0.35** | **0.21** | **0.55** | **0.63** | **0.80** |

- The **naive readout** scores each family's forms and takes the argmax. It lets the
  `substitution` magnet (whose "replace/change" tokens fire on *any* code edit) dominate the
  code-modification blob, so edit/repair collapse — and hand-derived or data-driven forms don't
  fix it (data-driven forms even generalize *worse*, 0.20 held-out, wrecking verification 0.59→0.09).
- The **learned probe** (multinomial logistic regression over the 219 pooled scored-token logprobs,
  mean-pooled across the late-layer band L40–47, **grouped 5-fold CV by task** — every turn scored
  while its task is held out) learns to down-weight the magnet and up-weight family-specific
  evidence. Held-out accuracy **0.54** vs the naive **0.39**, and the weak families jump. Robust
  across C ∈ {0.02, 0.05, 0.1}.

**So the fine distinctions ARE decodable from the residual via the lens** — the earlier "not in the
residual / coarsen instead" reading was wrong; it measured the weakness of argmax-of-means, not the
residual. The residual genuinely separates task_resolution, dependency, broad_success,
failure_confirmation, source_localization, and (partially) source_edit.

### What stays hard
- **repair (0.21)** and **source_edit (0.35)** are the residual's genuinely entangled pair — the
  "change code" blob is only partly separable; repair is the hardest single family.
- **substitution (n=7), located_source (n=2), focused_validation (0.17), runtime_name_failure (n=1)**
  — too few examples for a stable probe; their numbers are noise, not evidence of absence.

## The payoff — a meaningful disagreement detector on the trustworthy probe

Using the probe's **held-out** predictions (CV, so nothing memorized), rank the turns where the
probe's predicted concept disagrees with the CoT tag, by probe confidence. 245 disagreements over
532 turns; the top ones have the probe near-certain (p≈0.85–0.9) while the tag's probability is
~0.004 — strong candidate surface–internal divergences, e.g.:

- sympy-13480 t10 — tag `broad_success`, probe `task_resolution` (conf 0.88)
- astropy-14539 t28 — tag `failure_confirmation`, probe `verification` (conf 0.79)
- astropy-14508 t02 — tag `source_localization`, probe `located_source` (conf 0.73)

Many top disagreements are *adjacent* concepts (locate↔located, success↔resolved), consistent with
the end-of-thinking state anticipating the next mode. The detector is a **triage queue** of
candidate divergences ranked by confidence — for review, not a verdict.

## A coarse fallback: 3 concepts, no training needed

If a training-free readout is preferred, collapsing to **locate / modify_code / assess** makes the
plain argmax lens trustworthy (held-out 0.74; per-family 0.76 / 0.83 / 0.68) — coarser, but zero
fit. The learned fine probe is the stronger option when a labeled corpus is available.

## Honest limits

- **Triage, not verdict** — held-out accuracy 0.54, so a single fine disagreement is a candidate;
  confidence-ranking concentrates the real signal at the top.
- **repair / source_edit** remain only partly separable; **substitution / focused_validation / rare
  families** lack the support to trust.
- **Boundary effect** — end-of-thinking often encodes the *next* action mode, so some disagreements
  are label-vs-state timing, not unfaithfulness (itself a useful signal).
- The probe is trained on the tagger's labels, so it inherits any systematic tagger bias; a
  disagreement means the residual doesn't support the tag, which is the intended signal.

## Verdict

Per-concept probes **do** tighten the weak families — as *learned* readouts over the lens, not as
hand-picked token forms. Most fine families are now trustworthy per family (recall 0.35–0.80
held-out; task_resolution/dependency/broad_success went from ~0 to 0.55–0.80), repair/edit remain
the residual's genuinely hard pair, and a confidence-ranked disagreement detector turns the
trustworthy probe into a queue of candidate surface–internal divergences.

Artifacts: `cohort-concept-linear-probe.json` (+ `-records.json`),
`cohort-concept-disagreements-L44.json`, `cohort-perfamily-trust-eval-L44.json`.
Scripts: `concept_linear_probe.py`, `concept_disagreement.py`, `perfamily_trust_eval.py`,
`superconcept_faithfulness.py`, `datadriven_concept_forms.py`.
