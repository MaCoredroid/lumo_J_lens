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
- [ ] Compute `source_disagreement` per boundary across the trajectory; scale to
      the N60 development cohort (the single captured task is not enough for stats).
- [ ] Evaluate whether HIGH divergence flags meaningful structure, using only free
      signals: (a) does it concentrate at epistemic-event boundaries
      (diagnosis/bug/patch/fix milestones via the CoT reader join); (b) does it
      coincide with detours/non-advancing steps; (c) how it distributes across
      regions (reasoning vs visible vs tool). Report effect sizes + a permutation
      null (divergence-vs-random-boundary) so a signal isn't over-claimed.

**P3 — report**
- [ ] One Qwen-only J-lens report: per-boundary action + CoT + latent
      confidence/doubt, with the faithfulness evaluation. This is the deliverable.

## Loop discipline
- Advance concrete build/test/eval steps autonomously; commit + push each step.
- **PAUSE and surface to the user** (stop the loop) on any genuine DESIGN
  decision (e.g., how to define a faithfulness target) — do not guess. This is a
  research effort; design forks are the user's call, mechanical progress is mine.
- **STOP the loop** when P3 (the Qwen-only report) exists and its eval is green.
- Never resume or touch the parked multi-model / sealed-control code.
