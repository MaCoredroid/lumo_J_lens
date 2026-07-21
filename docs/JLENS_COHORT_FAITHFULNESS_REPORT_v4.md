# J-lens cohort faithfulness — report v4 (2026-07-21)

Supersedes v3's headline. Same question (does the residual state at each turn's end-of-thinking
encode the concept the CoT expressed?), three improvements applied and **cleanly ablated**:
read at the right depth (Lever 1), a richer concept vocab (Lever 2), and full 535-turn coverage
via bounded context (Lever 3).

## Headline

**Faithfulness = 0.41 centered** (n=532 comparable turns, full 535-turn coverage, late-layer
peak, v2 vocab) — up from v3's 0.31 (all-layer average, sparse vocab, 293 early-biased turns).

- Now **above the majority-class baseline** (source_localization 206/532 = 0.387) — v3 was below it.
- **5.7× the uniform baseline** (0.071).
- The peak is a flat late-layer plateau (L39–L44 ≈ 0.40–0.41; best single L44 = 0.412, L41 = 0.410,
  band 39–43 = 0.412), so the number is robust to the exact layer choice.

## The three levers — factorial ablation at L41 (centered)

v1's 63 concept tokens are a strict subset of v2's 126, and the v4 (bounded, 535-turn) readout
scored the full v2 set, so one pair of readouts gives a clean factorial:

| config | n | centered | isolates |
|---|---:|---:|---|
| v3 full context · v1 vocab · 293 | 290 | 0.379 | — |
| v4 bounded · v1 vocab · shared 293 | 290 | 0.372 | context: **bounded costs −0.007** |
| v4 bounded · v2 vocab · shared 293 | 290 | 0.386 | vocab on shared turns |
| v4 bounded · v1 vocab · all 535 | 532 | 0.380 | coverage on v1 |
| **v4 bounded · v2 vocab · all 535** | 532 | **0.410** | full result |

**Attributed gains (vs the v3 0.307 all-layer number):**
- **Depth (Lever 1):** all-layer 0.307 → L41 0.379 — **+0.072, the largest lever.** The concept
  signal is late-layer localized; averaging layers 16–47 diluted it.
- **Vocab (Lever 2):** v1 → v2 at fixed context/coverage: **+0.030** (0.380 → 0.410 on 535).
- **Coverage (Lever 3):** bounding context costs essentially nothing (**−0.007**, so the
  approximation is validated), and adding the 242 late turns *raises* faithfulness
  (**+0.024**: 0.386 → 0.410) — the late turns are not harder, so v3's early-turn bias was
  *understating*, not inflating, the number.

## Per-family recall at the peak (L44, v4 bounded/v2/535)

| family | n | recall (centered) | vs v3 all-layer |
|---|---:|---:|---|
| substitution_operation | 7 | 0.714 | 0.714 |
| verification | 135 | 0.615 | 0.784 |
| focused_validation | 12 | 0.500 | — |
| source_localization | 206 | 0.461 | 0.245 ↑ |
| failure_confirmation | 19 | 0.368 | 0.273 ↑ |
| task_resolution | 15 | 0.333 | 0.000 ↑ |
| source_edit | 51 | 0.235 | 0.040 ↑ |
| repair | 33 | 0.121 | 0.053 |
| dependency_unavailable | 16 | 0.062 | 0.143 |
| broad_success | 22 | 0.045 | 0.143 |
| test_success / defined_identifier / located_source / runtime_name_failure | 2–7 | 0.000 | — |

The concept-*dependent* pattern from v3 **broadens** at the right depth + vocab: source_localization
(0.25→0.46), source_edit (0.04→0.24), task_resolution (0→0.33), failure_confirmation (0.27→0.37)
all come alive; verification (0.62) and substitution (0.71) stay strong. Genuinely weak concepts
remain: repair, broad_success, dependency, test_success. So v3's "only verification is encoded"
read was largely an artifact of layer-dilution + sparse vocab + coverage bias.

## Method notes

- Boundary/reconstruction unchanged from v3 (fixed framing + trace trajectory + exact thinking).
- **Bounded context** (Lever 3): system + task + the most-recent whole turn-blocks fitting a
  30 000-token cap + the current thinking; truncation only at turn boundaries. All 535 turns fit
  (0 dropped), median 28k tokens, median 10 recent blocks kept. Rendered locally (registered the
  codex template's `from_json`/`tojson` filters), verified to match the server `/tokenize`.
- **v2 vocab** (Lever 2): 126 task-independent single-token forms across 14 families (v1 was 63),
  through the same single-token + global-uniqueness filter.
- Same lens (public Jacobian), layers 16–47 captured, per-layer LOO baseline-centering.

## Verdict (revised)

The internal state is **partially faithful, and more broadly than v3 suggested**. Read at the
late-layer peak with a richer vocab and full coverage, it tracks the CoT's concept at **0.41 —
above the majority baseline, 5.7× chance** — with genuine per-concept signal for localization,
verification, substitution, and focused-validation, moderate for edit / resolution /
failure-confirmation, and little for repair / dependency / broad-success / test-success.

Artifacts: `artifacts/cohort-faithfulness-ablation-L41.json`,
`artifacts/cohort-faithfulness-perlayer-v4.json`, readout `cohort-perturn-readout-v4.json`
(gitignored, 1.7 GB). Prior: `JLENS_COHORT_FAITHFULNESS_REPORT_v3.md`.
