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
- [ ] Formalize retirement of the multi-model control path in the design doc.
- [ ] Run the existing Qwen-only test suite (reasoning_trace + observable_* +
      calibration/evaluator) → record what passes; establish the current baseline
      output of the lens on the captured trajectory.

**P1 — wire the free CoT into the lens output**
- [ ] Add a trajectory reader that extracts, per boundary, the `<think>` text +
      the reasoning-event markers + the next-action label from
      `swe_jlens_trajectory` (read-only; no new capture).
- [ ] Extend the reasoning_trace output so each boundary carries the observed CoT
      snippet + events ALONGSIDE the latent indices (entropy/ambivalence/JS).
      Keep labels evaluation-only; never let CoT text feed the predictor.

**P2 — faithfulness / confidence readout (the real lens value)**
- [ ] Define faithfulness/confidence targets from FREE signals only, e.g.:
      does the latent doubt index rise where the CoT hedges; does the stated
      diagnosis boundary coincide with the action-prediction margin; does
      confidence predict eventual task success.
- [ ] Fit/evaluate on held-out with the free action/outcome labels. Report
      calibration + whether the latent signal adds over the CoT text alone.

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
