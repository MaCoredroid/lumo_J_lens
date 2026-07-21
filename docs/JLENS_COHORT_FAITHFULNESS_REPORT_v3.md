# J-lens cohort faithfulness — report v3 (2026-07-21)

**Question.** At each turn's end-of-thinking boundary, does the Qwen residual state —
read through the public Jacobian lens, scored over a task-independent concept-form vocab —
encode the concept the model's *own* chain-of-thought expressed (the tagger's label)?

**One-line answer.** Partial and **concept-dependent** faithfulness: the lens genuinely and
strongly encodes *verification* (centered recall 0.78) and *substitution* (0.71), but does
**not** reliably encode the *localize / edit / repair / resolve* family — whose raw "hits"
are base-rate artifacts that vanish under centering.

## Numbers (n = 290 comparable turns)

| metric | value | reading |
|---|---|---|
| faithfulness, **baseline-centered** top-1 | **0.307** | 4.3× the uniform baseline (0.071) |
| faithfulness, **raw** top-1 | 0.448 | ≈ majority-class baseline → mostly base-rate |
| majority-class baseline (always `source_localization`) | 0.493 | trivial; exploits the 49% base rate |
| uniform baseline (1/14) | 0.071 | chance |

The raw number (0.45) essentially equals the majority-class baseline (0.49): raw scoring
mostly reflects family base-rates, not concept tracking. **Centering removes that confound**;
what survives (0.31, 4.3× chance) is genuine signal — but, as the per-family table shows, it
is *concentrated in a few concepts*, not spread evenly. That is why the centered overall
number sits below the base-rate-exploiting majority baseline: the lens is not a base-rate
predictor, it is a *concept-specific* one.

## Per-family recall — where the faithfulness actually lives

| family | n | centered | raw | reading |
|---|---:|---:|---:|---|
| **verification** | 51 | **0.784** | 0.882 | genuine, strong — barely drops under centering |
| **substitution_operation** | 7 | **0.714** | 0.143 | genuine — centering *unmasks* it (was hidden by high-baseline families) |
| focused_validation | 2 | 0.500 | 1.000 | n too small |
| failure_confirmation | 11 | 0.273 | 0.091 | weak-moderate |
| source_localization | 143 | 0.245 | 0.476 | partial real signal; raw is base-rate-inflated |
| dependency_unavailable | 14 | 0.143 | 0.000 | weak |
| broad_success | 7 | 0.143 | 0.000 | weak |
| repair | 19 | 0.053 | 0.053 | ~none |
| **source_edit** | 25 | **0.040** | 0.480 | raw hits are **pure base-rate artifact** — collapse under centering |
| task_resolution | 4 | 0.000 | 0.000 | none |
| defined_identifier / located_source / test_success | 2 each | 0.000 | — | none (tiny n) |
| runtime_name_failure | 1 | 0.000 | — | none |

The `source_edit` row is the cleanest illustration of why centering matters: raw recall 0.48
looks like tracking, but it is entirely base-rate/vocab bleed — corrected, it is 0.04.
`verification` is the opposite: 0.88 raw → 0.78 centered, i.e. the signal is *real*, not a
base-rate effect. Interpreted charitably, the lens reliably represents "am I checking /
validating this" reasoning (26% of all CoT turns) and "replace X with Y" reasoning, but does
not carry a distinct residual signature for "where is the bug / edit this / fix this / done".

## Method

- **Boundary.** Per tagged turn, the exact end-of-thinking token. Prompt reconstructed as:
  fixed agent framing (42.5k-char qwen-code system prompt + the task's initial user message,
  taken from the rerun proxy dumps — run-independent) + the original thinking-on canonical
  trace's turn-by-turn assistant actions and tool results (`tool_use`↔`tool_result` by id),
  with the turn's exact thinking appended so the boundary lands on the last thinking token.
  Prior-turn thinking dropped (the agent strips it — confirmed by the dumps). Rendered via the
  live server `/tokenize`. Turn indices align 1:1 with the 535 tags (0 mismatches / 20 tasks).
- **Lens.** Public Jacobian lens (`neuronpedia/jacobian-lens`, J 5120×5120), VJP over layers
  16–47 at the final position, scored over the 14-family general concept-form vocab
  (`configs/swe_task_state_v4_general_concept_forms.json`, 63 single-token forms).
- **Scoring.** Per family: mean form/layer log-probability; leave-one-out cross-boundary
  baseline-centering (removes each family's frequency bias); centered top-1 = internal concept.
- **Faithfulness.** internal top-1 == tag, over turns with a real tag (`none` excluded).

## Coverage & caveats (do not over-read)

- **Coverage 293/535 turns (290 comparable).** Reconstructions median ~31.9k tokens; the agent
  runs near the 32,768 context limit without heavy compression (confirmed vs real sent-prompts,
  p90 32,129), so 242 long/late-trajectory turns exceed the window and are dropped.
- **Early/mid bias.** Kept turns skew to turn-index p90=21. Late-trajectory concepts are
  under-covered — `task_resolution` 4/15, `focused_validation` 2/12, `broad_success` 7/22,
  `verification` 51/135 — so the resolve/validate end of faithfulness is thinly measured.
- **All-layer averaging** (16–47) may dilute layer-localized concept signals; a per-layer sweep
  could reveal stronger, depth-specific faithfulness (untested here).
- **General vocab** is deliberately task-agnostic and weaker than the hand-tuned single-task
  probe (which reached 0.60 strict on sympy-13480). 0.31 cohort-centered is the honest
  general-vocab number.

## Verdict

Not a clean "the lens is faithful," not null. **The internal state faithfully encodes a
specific, important slice of the agent's reasoning — verification/checking and substitution —
and does not encode the localize/edit/repair/resolve slice as distinct concepts.** After
removing base-rate confounds the lens tracks the CoT 4.3× above chance, but the signal is
concentrated, not uniform.

Artifacts: `artifacts/cohort-faithfulness-v3.json` (+ `-records.json`),
readout `artifacts/cohort-perturn-readout-v3.json`.
