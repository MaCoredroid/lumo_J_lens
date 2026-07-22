# Does the J-space concept component drive the SWE agent's ACTION? (2026-07-21)

**Question.** The two-hop factual reproduction showed that swapping the J-space component of an
intermediate concept flips the model's *answer*. Does the same hold for our SWE agent — does
swapping the J-space component of a *reasoning concept* (locate vs modify) redirect the agent's
*action* (its next tool/command)? This would upgrade our correlational faithfulness ("the state
encodes 'locate'") to a causal claim ("the concept drives behavior").

**Answer: NO — a robust null.** Unlike a factual answer, the SWE agent's action is **not** causally
controlled by a swappable concept vector. Injecting the locate→modify concept direction (its
J-space part, its non-J part, or the whole thing), at any tested scale, never redirects the agent
to a modify action — it is either recovered from or garbles generation, but never steers behavior.

## Method (agentic analog of the reproduction)

- **Concept direction.** Δ[L] = mean(residual over `modify` turns) − mean(residual over `locate`
  turns) at the end-of-thinking boundary, from tagged cohort turns (train split; source_localization
  /located_source/defined_identifier = locate, source_edit/repair/substitution = modify).
- **J/non-J split.** Δ_J = row(J[L]) projection, Δ_⊥ = remainder; norm-matched at the concept scale.
  Norms (L16): ‖Δ‖=7.6, ‖Δ_J‖=3.1, ‖Δ_⊥‖=6.9, ‖residual‖=46.5 — the concept is ~16% of the residual
  norm, and mostly in the non-J part (the global-workspace shape).
- **Intervention.** Inject the component across band L16–40 (prefill, chunk-aware) at held-out
  locate turns; read the agent's action. Two loci: (a) end-of-thinking + free generation; (b) the
  forced command-choice point (`…<parameter=command>\n`) reading the immediate command. Scale swept
  α ∈ {3, 8, 20}× the concept magnitude. Controls: full Δ, norm-matched random in row(J)/ker(J), no-op.
- **Action classifier.** locate = grep/find/ls/cat/head/read; modify = sed -i/edit/write/replace.

## Results

**End-of-thinking injection (α≈1–2):** the model **recovers** — it emits the perturbed token, then
`</think>`, then the *same* locate action as baseline. E.g. dJ→"icc", dperp→"Shibboleth", but both
continue to `grep for VLA` / `read_file diff.py` unchanged. All conditions → 0% modify.

**Command-choice injection, α sweep (n=8 held-out locate turns):**

| | modify rate | valid-command rate |
|---|---:|---:|
| baseline (no inject) | 0.00 | 0.25 |
| Δ_J  @ {3,8,20} | **0.00 / 0.00 / 0.00** | 0.00 / 0.00 / 0.00 |
| Δ_⊥  @ {3,8,20} | 0.00 / 0.00 / 0.00 | 0.12 / 0.00 / 0.25 |
| full Δ @ {3,8,20} | 0.00 | 0.00 |
| random row(J)/ker(J) | 0.00 | ≤0.25 |
| no-op | 0.00 | 0.25 |

**`modify` rate is 0.000 in every cell.** No injection, at any scale, ever produces a modify
command. The J-space injection (Δ_J) actually *garbles* generation hardest (valid=0 everywhere) —
consistent with J-space being the output-sensitive channel — but it garbles to **nonsense**, not
to a modify action. Non-J and random sometimes leave a valid *locate* command intact.

## Interpretation

**The naive concept-swap that works for factual two-hop does not transfer to agentic action
control.** For a fact, the intermediate concept is the immediate causal determinant of the next
token, so a J-space swap flips the answer. For the SWE agent, the tool/command is determined by the
concrete task state (which bug, which files, prior tool outputs) — distributed across the whole
trajectory — not by a low-dimensional "locate vs modify" vector that can be hijacked at the decision
point. The model re-derives its action from context and is robust to (or destroyed by) the injection.

**Correlational faithfulness ≠ causal control.** Our earlier lens shows the residual *encodes* the
concept the CoT expresses; this shows that concept, as a vector, does **not** *control* the agent's
behavior. The concept lens reads out reasoning; it is not a steering lever. This sharpens — and
appropriately humbles — the interpretation of the whole faithfulness line, and cleanly contrasts
agentic action (context-anchored, robust) with factual recall (a clean, swappable concept→answer
channel).

## Honest limits (this is "this intervention doesn't steer," not "nothing can")

- The action metric is coarse and noisy (baseline valid-command rate only 0.25 — many commands
  like `sed -n …` / `git …` / `python …` aren't classified locate/modify).
- Δ is a simple mean-difference probe; a *trained* steering vector might steer where this doesn't.
- Single-locus, prefill-only injection across a band may be the wrong intervention; persistent
  decode-time or multi-position steering is untested.
- Small n (8 held-out turns), one concept pair (locate→modify).

So the null is specific: **the mean-difference concept swap that reproduces factual mediation does
not steer the agent's action.** Stronger steering methods remain open.

Scripts: `run_swe_state_action_swap.py` (+ `.sh`), `run_jlens_patch.py` (chunk-aware write hook),
`jspace_decompose.py`. Results: `artifacts/swe-sa-sweep.json`, `swe-sa-diag.json`, `swe-sa-forcecmd.json`.
Contrast: the positive factual result in `docs/JSPACE_SWAP_REPRODUCTION.md`.
