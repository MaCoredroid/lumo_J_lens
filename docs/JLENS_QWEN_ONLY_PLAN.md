# J-lens Qwen-only plan + loop runbook

Doc of record for the **Qwen-only** J-lens effort. Supersedes the multi-model
epistemic-chain control-adjudication path, which is **retired** (see below). The
`/loop 10m` supervisor reads this each iteration, advances ONE concrete step,
verifies it, and yields.

## Scope decision (2026-07-20)
J-lens is a lens over the **Qwen** SWE-agent's own trajectory. Nothing else.
- **Retired / parked:** the sealed multi-model control run
  (independent_a=Qwen, independent_b=GPT-OSS, adjudicator=Mistral) and its
  `sealed_control_*`, `sealed_adjudicator_*`, `epistemic_chain_*` code. Left
  committed-but-parked, not deleted. Do NOT resume it.
- Rationale: the two things the lens needs are FREE from the Qwen trajectory
  jsonl, so no second/third model and no annotation are needed. See
  `[[jlens-qwen-only-cot-free]]`.

## What is free from the trajectory (`~/lumo_J_lens/.cache/swe_jlens_trajectory/`)
Verified 2026-07-20 against `swe_jlens_trajectory_prompts.summary.json`:
- **action label** — the agent's own next tool call → the `inspect`/`edit`/
  `check_or_finish` target (self-supervised, non-circular).
- **CoT text** — each completion's `<think>…</think>` block (9/9 requests had one).
- **reasoning-event markers** — already tagged as token offsets: `bug_recognized`,
  `diagnosis_named`, `patch_target_named`, `reproduction_planned`, `fix_working`,
  `task_resolved`, etc. A labeled CoT timeline from Qwen's own output.
- **task outcome** — SWE-verified pass/fail (free from the harness).

## The only thing NOT free (the lens's actual job)
The latent signal BEHIND the tokens: is the stated CoT faithful; how much
confidence / doubt / stress sits behind a claim. That is the sole reason to read
Qwen activations/logits/Jacobians rather than parse the jsonl. It needs NO
external labels — anchor it with the free action/outcome labels and check it
against the captured `<think>` text.

## Existing Qwen-only stack (reuse; do NOT rebuild)
- `swe_task_state_v4_reasoning_trace.py` (+ `.json`): CoT-phase sidecar; already
  emits `normalized_entropy`, `ambivalence_index`, source-disagreement
  (`normalized_js_divergence`) — the confidence/doubt indices.
- `swe_task_state_v4_observable_*`: decoder, events, feature bundle, nested
  inference, bootstrap.
- `swe_task_state_v4_activation_features.py` / `activation_projection.py` /
  `features.py`: Qwen activation feature capture.
- `swe_task_state_v4_calibration.py` / `decision.py` / `evaluator.py` /
  `metrics.py`: fit / calibrate / evaluate.
- action gauge: last at ~0.760 balanced acc, 0.859 accepted acc @ 75.5% coverage.

## Work plan (loop advances one step per iteration; verify each)
**P0 — retire + baseline**
- [x] Retirement of the multi-model control path recorded (this doc + `[[jlens-qwen-only-cot-free]]`).
- [x] Qwen-only test suite baseline (2026-07-20): **197 passed, 1 skipped, 98
      subtests** across the 21 non-multi-model `test_swe_task_state_v4_*` files.
      Stack is healthy.

**Canonical readout test command** (the readout stack needs SciPy/sklearn, which
live in `.venv-readout-v2`; pytest lives in the vllm-side uv archive — combine them):
```
QT=$(ls tests/test_swe_task_state_v4_*.py | grep -viE 'epistemic_chain|sealed|adjudicat|counterfactual|controls_v2|control_executor|native_vllm|native_smoke')
PYTHONPATH=/home/mark/.cache/uv/archive-v0/Yn5oTX4avDWYV_JW/lib/python3.12/site-packages \
  .venv-readout-v2/bin/python -m pytest -q $QT
```
(pytest 9.1.1, scipy 1.18.0, sklearn 1.9.0, numpy 2.5.1 — the pinned readout env.)

**P1 — wire the free CoT into the lens output**
- [x] Trajectory reader `scripts/swe_task_state_v4_trajectory_cot_reader.py`
      (+ test, 6 pass): read-only, extracts per boundary the stage / region /
      reasoning-event markers, and a per-turn epistemic timeline. Verified on the
      captured trajectory (9 turns, 293 boundaries): diagnosis_named ->
      bug_recognized -> correct_identifier_named -> patch_target_named ->
      fix_working -> task_resolved. Evaluation/observation-only; no predictor input.
- [ ] (follow-on) optionally decode the verbatim `<think>` text per turn
      (needs the Qwen tokenizer or `public-report.json`) — deferred; the
      structured event timeline above is the higher-signal, self-labeled part.
- [x] Join module `scripts/swe_task_state_v4_lens_report_join.py` (+ test, 4 pass):
      `attach_reasoning_context(trace_rows, turns)` augments each
      `build_reasoning_trace` output row (aligned by `boundary.request_index`)
      with its turn's observed stage + semantic events, ALONGSIDE the latent
      `proxies` (entropy/ambivalence/source-disagreement). Reasoning_trace left
      untouched; observed context is a report annotation, not a predictor input.
      Full Qwen-only suite: 207 passed, 1 skipped, 98 subtests.

**P1 COMPLETE.** Each lens prediction now carries: predicted action/phase +
calibrated confidence + latent doubt indices + the observed epistemic timeline —
all Qwen-only, all free.

**P2 — faithfulness = surface-vs-internal divergence** (user decision 2026-07-20)
Faithfulness metric = per-boundary `source_disagreement` = normalized
JSD(`sequence_logit_probabilities`, `sequence_j_probabilities`) ∈ [0,1] — where the
public-token story diverges from the internal-J story. Already computed by
`reasoning_trace._source_disagreement`; needs per-boundary diagnostics carrying
BOTH source distributions.
- [x] Located the source: the per-boundary `sequence_logit` and `sequence_j`
      probability distributions are produced by the per-source decoders in
      `swe_task_state_v4_observable_decoder.py` / `..._evaluator.py` (each a 136-dim
      action-word-probe baseline: sequence_logit = ordinary logit, sequence_j =
      public-J). NOT in the report artifacts (those are summaries). Next: run those
      decoders to emit per-boundary per-source probabilities as the divergence input.
- [x] Compute `source_disagreement` per boundary across the N60 cohort. Data was
      already materialized: `observable-action-phase-v1.json` carries
      `variants/sequence_logit` and `variants/sequence_j` OOF probabilities (1570
      rows, identical folds/labels/weights), so no decoder run was needed. Built
      `scripts/swe_task_state_v4_source_divergence.py` (+ test, 7 pass; JSD verified
      to match `reasoning_trace`).
- [x] Eval (b) divergence-flags-error: **RESULT — divergence significantly flags
      lens mispredictions.** On 1570 rows (332 error / 1238 correct of the pooled
      `sequence_logit_j` forecast vs the agent's own next action): divergence mean
      **0.01445 on error vs 0.00908 on correct** (effect +0.00538), **error-detection
      AUC 0.631**, label-permutation **p ≈ 2e-4** (5000 perms). Small absolute JSD
      (sources usually agree) but a real, significant signal. "Error" = pooled
      forecast argmax != free next-action label.
- [ ] (follow-on) Eval (a) event-concentration + (c) region distribution — needs
      per-row semantic events for the 1570 cohort (the CoT reader currently covers
      the single demo trajectory). Check whether cohort-wide event tags exist; if
      not, this is descriptive-only on the demo trajectory. Non-blocking for P3.

**P3 — report** ✅ COMPLETE
- [x] `scripts/swe_task_state_v4_qwen_lens_report.py` (+ test, 4 pass) →
      `artifacts/jlens-qwen-only-report-v1.json`. Synthesizes, all Qwen-only / all
      free: (1) action gauge — weighted accuracy history_only 0.752 →
      sequence_logit 0.776 / sequence_j 0.773 / pooled sequence_logit_j 0.772
      (pooled best AUPRC 0.672 vs 0.477 baseline); (2) faithfulness — divergence
      flags lens error, AUC 0.631, permutation p ≈ 2e-4; (3) the free demo
      epistemic timeline; plus scope (no GPT-OSS/Mistral/human labels) and honest
      limitations (event-concentration not cohort-scale).

**P3 report exists and green (v1). Follow-on faithfulness work below.**

---

## P4 — REAL faithfulness: CoT-event ↔ internal-concept agreement (user, 2026-07-20)
The P2 divergence metric was rescoped as a lens-reliability flag, NOT faithfulness
(both sources are internal probes). Real CoT-faithfulness = does the internal
concept-chain readout encode the concept the CoT claims at that boundary.

**Data (from scoping workflow wf_b875405b-43c):**
- Concept-chain readout: `scripts/swe_task_state_v4_concept_chain.py` ->
  `.cache/swe_task_state_v4_concept_chain/common-ontology-chain.json`. Per boundary
  (`boundary.request_index`+`boundary.offset`): `candidate_rankings.{public_j,native_j,
  ordinary_logit}.top_k` + `selection.selected_concept_id`; 17 families, 14 scorable
  (excluded: typographical_error, repair_success, repair_summary). Also carries
  `evaluation.boundary_rows[i].positive_concept_ids` (human reference labels).
- Free CoT events: `swe_task_state_v4_trajectory_cot_reader` (per turn/offset).
- **Both are the SAME single demo task** (swe-sympy-13480): 10 boundaries, 5
  strict-fidelity, 3 selected. So P4 is DESCRIPTIVE, n=1, uncalibrated — the honest
  hard result (design doc already: 5/10 fidelity, 3/10 selected).

**Event -> concept mapping (a design choice; high-confidence, scorable only):**
diagnosis_named->source_localization, source_location_reaffirmed->located_source,
correct_identifier_named->defined_identifier, failure_confirmed->failure_confirmation,
original_reproduction_passed->verification, broader_values_passed->broad_success,
pytest_unavailable->dependency_unavailable, focused_test_passed->test_success,
task_resolved->task_resolution. (fix_working->repair_success dropped: unscorable.
bug_recognized/patch_target_named/reproduction_planned: medium/low, reported separately.)
The mapping is an explicit, reviewable table in the module — the user can adjust it.

**Preliminary signal (from public_j top1 vs mapping):** MIXED — matches at
defined_identifier / dependency_unavailable / verification / task_resolution; misses at
turn-1 diagnosis (reads focused_validation); public_j has a focused_validation bias (7/10).

**Steps:**
- [x] `swe_task_state_v4_cot_concept_faithfulness.py` (+ test, 4 pass): aligns free
      events to concept boundaries by (request_index, offset), scores internal top-1
      vs the CoT-implied concept, cross-checks free events vs human positive labels.
- [x] Folded into the report (v2) as `cot_faithfulness`, distinct from the
      reliability flag.
- [x] **RESULT (honest): WEAK/PARTIAL.** Internal concept-chain top-1 matches the
      model's own CoT-implied concept only **0.33** of the time (3/9 events; 0.40 on
      strict-fidelity boundaries). The free CoT events DO agree with human labels
      (~1.0), but the internal readout tracks neither reliably, and public_j has a
      focused_validation bias (7/10). i.e. on this task the lens does NOT reliably
      encode what the CoT claims — the falsifying result the design doc foreshadowed.
- [x] (enrichment) uncertain-mapping events reported separately (top-1 0.25, n=4 —
      even weaker); native_j agreement added (0.33, also weak — not a public_j
      artifact); focused_validation-bias quantified (public_j collapses to
      focused_validation on 6/10 boundaries, native_j 3/10 — the degenerate root
      cause of the weak faithfulness). All in report v2. Suite 224 passed.

**P4 enrichments COMPLETE. Only the cohort-scale fork remains → loop STOPPED.**
- [~] cohort-scale fork was surfaced; user chose **"diagnose/fix the probe first"** ->
      superseded by P5.

## P5 — diagnose + fix the concept-probe degeneracy (user, 2026-07-20)
- [x] **Diagnosed:** the collapse is a BASELINE-FREQUENCY artifact, not model
      unfaithfulness. The concept score is an absolute mean log-prob over a family's
      token forms with NO baseline correction, so high-frequency families dominate:
      focused_validation has the highest baseline mean (-12.96) and the CoT-mapped
      families sit near the bottom (source_localization -16.4, located_source -16.9).
- [x] **Fixed (post-hoc re-ranking):** leave-one-out per-family baseline centering in
      `swe_task_state_v4_cot_concept_faithfulness.py::_attach_baseline_centered_top1`.
      Effect: focused_validation top-1 share 6/10 -> 0/10; CoT-faithfulness top-1
      **0.33 -> 0.44 overall, 0.40 -> 0.60 on strict-fidelity boundaries**. So much of
      the "weak faithfulness" WAS a probe artifact; the lens PARTIALLY tracks the CoT
      once corrected. Report v2 reframed accordingly. Suite 224 passed.
- [ ] (follow-on) the fix is a POST-HOC re-rank using an LOO baseline over the 10 eval
      boundaries; a production fix would bake a proper reference/null-context baseline
      into `concept_chain.py` itself (heavier; re-runs the lens). And residual misses
      are partly semantic-adjacency (source_localization vs located_source) + offset
      alignment — a family-group match would raise agreement further.
- [ ] **RE-OPENED FORK (user):** now that the probe fix roughly recovers a moderate
      faithfulness signal, is cohort-scale worth it (bake the baseline fix into
      concept_chain.py + run over N60)? Surface, do not start without the user.

**Loop discipline unchanged:** pause on a NEW design fork (e.g. the mapping being
too ambiguous, or the cohort-scale decision); stop when the faithfulness result is
in the report and green.

## Loop discipline
- Advance concrete build/test/eval steps autonomously; commit + push each step.
- **PAUSE and surface to the user** (stop the loop) on any genuine DESIGN
  decision (e.g., how to define a faithfulness target) — do not guess. This is a
  research effort; design forks are the user's call, mechanical progress is mine.
- **STOP the loop** when P3 (the Qwen-only report) exists and its eval is green.
- Never resume or touch the parked multi-model / sealed-control code.
