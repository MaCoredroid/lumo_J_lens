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
## P6 — bake the baseline fix into concept_chain.py (user, 2026-07-20)
Make the lens emit baseline-corrected rankings itself, not just a post-hoc re-rank.
**Feasibility (investigated):** re-runnable from the present intermediate reports
(.cache/swe_jlens_intermediate/{analysis,public-report,native-report,prompts}.json);
the config pins INPUTS not the source, so no source-hash update; 499-line
test_swe_task_state_v4_concept_chain.py must stay green (or be updated for the new
ranking). **Baseline choice:** `_report_scores` emits a raw absolute mean log-prob and
NO null-context/prior baseline exists in the data — a true null-context baseline needs
a NEW reference capture (out of scope now). So bake **cross-boundary background
centering** (each family minus its mean over the run's boundaries) at the ranking step;
this is a legitimate empirical background baseline AT COHORT SCALE (circular only for a
tiny single-run). Document the caveat in-code.
- [ ] Inject family-mean centering between `_report_scores` and `_ranking` in
      `_build_concept_chain` (localized; keep raw scores + fidelity gates intact).
- [ ] Re-run the lens -> regenerate common-ontology-chain.json; verify the
      focused_validation collapse is gone in the LENS's own output and selection
      diversifies. Update the concept_chain tests for the new ranking.
- [ ] Re-point the faithfulness module to the lens's now-corrected top-1 (drop the
      post-hoc re-rank), confirm 0.44/0.60-ish carries through; update report v3.
- [ ] PAUSE if the change would weaken the strict fail-closed selection contract or a
      test reveals an integrity issue — surface, don't force.
- [!] **BLOCKED — integrity finding (2026-07-20):** baking cross-boundary background
      centering into the lens VIOLATES the lens's online/causal contract. The additive
      `baseline_centered_rankings` change was reverted because
      `test_future_score_mutation_cannot_change_earlier_online_nodes` failed: a family
      baseline over ALL boundaries uses FUTURE boundaries, so mutating a future
      boundary changes earlier boundaries' centered rankings. A full-background
      baseline is inherently RETROSPECTIVE; it cannot be a causal lens feature.
      Reframe: baseline-centering belongs in the RETROSPECTIVE faithfulness analysis
      (where it already lives, `swe_task_state_v4_cot_concept_faithfulness.py`), not
      the online lens. Options surfaced to the user:
      (A) keep centering retrospective-only (current post-hoc re-rank) — the lens stays
          causal/raw; recommended;
      (B) causal prior-only/expanding baseline in the lens (uses boundaries <= current;
          noisy early, converges at cohort scale, order-dependent);
      (C) fixed EXTERNAL null-context/prior baseline baked in — needs a NEW reference
          capture (heavier), then it IS causal.
      Then cohort-scale is a separate fork. **Loop paused pending this decision.**
- [x] **RESOLVED (user, 2026-07-20): keep it retrospective (option A).** The lens
      stays causal/raw; baseline-centering remains in the retrospective faithfulness
      analysis, which the failed bake-in proved is the methodologically correct layer.
      Nothing more to build on the lens. Faithfulness thread settled: probe artifact
      diagnosed, retrospective correction recovers a moderate signal (0.44 / 0.60
      strict), lens correctly causal. Docstring updated to reflect this is by design.

## Faithfulness thread: single-task COMPLETE. Cohort-scale = P7.

## P7 — Qwen CoT-event auto-tagger for cohort-scale faithfulness (user, 2026-07-20)
Cohort-scale was blocked: the CoT events, ontology, and human labels are ALL
hand-curated for the one demo task (`materialize_swe_jlens_trajectory.py::SEMANTIC_EVENTS`
is a hand-written {turn: (label, exact-CoT-sentence)} dict; `materialize_swe_intermediate_probes.py`
hard-errors unless `exploratory_one_task_adaptation`). User chose to build an automated
Qwen-only tagger so the faithfulness eval can scale to n>>1.

**Inference path:** serve Qwen via `scripts/start_server.sh` /
`serve_qwen36_27b_nvfp4_mtp.sh` -> OpenAI API on **port 9952**, served `qwen3.6-27b-nvfp4`.
**Tag space = the 14 scorable concept families** (shared with the internal concept
readout, so the comparison is direct; reviewable — some may not fit non-bug tasks).
**Design:** Qwen tags each CoT boundary with a concept family (or abstain); faithfulness
= internal concept-chain top-1 (baseline-centered, retrospective) vs Qwen's CoT tag.
Both Qwen-only; drops the hand-curated event->concept mapping.

**Steps (loop-driven, /loop 20m monitor):**
- [ ] Build `swe_task_state_v4_cot_concept_tagger.py`: family definitions, prompt,
      strict parse/validate (constrained to the 14 families + abstain), server-query
      fn. CPU-testable parsing.
- [x] Server integration fixed: Qwen3.6 is a reasoning model, so tag with
      `chat_template_kwargs={enable_thinking:false}` (answer lands in content). The 9
      clean per-turn CoT blocks come from `runs/.../sympy__sympy-13480/qwen_trace.json`
      -> `message.content[*].thinking` (NOT the probe reports).
- [x] **VALIDATED (gate passed):** `scripts/validate_cot_tagger_demo.py` — Qwen
      auto-tagger vs hand-curated labels = **6/9 = 0.67 strict, 8/9 = 0.89
      adjacency-aware** (1 defensible miss, turn 3). The tagger reliably reproduces the
      hand labels -> trustworthy to scale. Server on 9952 (systemd lumo_j_lens_qwen27b).
- [!] **CAPTURE-COST FORK (paused for user, 2026-07-20).** Assessed the cohort capture:
      - Tagger side is CHEAP: 143 cohort task traces exist under runs/.../per_task/
        (astropy, sympy, ...); the tagger can run cohort-wide (~1300 Qwen calls, ~20 min).
      - Internal side is HEAVY + not done: NO cohort concept-probe capture exists;
        `materialize_swe_intermediate_probes.py` hard-errors off one-task; the
        Jacobian-lens readout capture (swe_task_state_readout.py) must be generalized +
        run over ~15k cohort boundaries (demo = 293 boundaries / 149 MB; cohort ~50x =
        many hours GPU + ~9 GB). This is the "capture cost" pause fork.
- [x] **DECISION (user): do the FULL cohort, optimized for speed** (per the GPU-util
      standard — no low-util captures). **Cost corrected (was over-estimated):** the
      capture is the NVFP4 VJP lens (`run_jlens_nvfp4.py` + `fp8_live_vjp.py` /
      `nvfp4_packed_vjp.py`) via vLLM forward hooks with a PRECOMPUTED J matrix (no live
      backward). Demo = 159s / 293 boundaries (~0.54s/bnd), 10 captured layers, gpu-util
      0.78, fp8 kv. Cohort capturing per-turn boundaries (~60 tasks x ~10) ≈ 30 min–2 hr,
      NOT hours-plural. Feasible + monitorable.

## P7b — full cohort capture, optimized (user, 2026-07-20)
Pipeline to generalize off single-task (each currently hard-wired to swe-sympy-13480):
`materialize_swe_jlens_prompts.py` (trace->prompts) -> `run_jlens_nvfp4.py` (VJP capture)
-> `materialize_swe_intermediate_probes.py` (probe reports) -> `concept_chain.py` (17-family
readout). Plus the Qwen tagger on each task's CoT. Then faithfulness across tasks.
- [x] Robust cohort trace reader `swe_task_state_v4_cohort_traces.py` (+ test):
      reads the qwen-code trace corpus and extracts per-turn CoT. CORRECTION (user
      caught this): the traces are NOT heterogeneous JSON/JSONL — they are ONE clean
      format, a JSON ARRAY of qwen-code stream events per task (qwen3.6-27b via the
      qwen-code CLI). Across ~17 runs: 44 clean json-array traces + 39 EMPTY (0-byte)
      files from failed/superseded runs (swe_multitask_c1_host v1/v2/v5/v7 empty;
      v3/v4 populated). No JSONL exists; the reader's jsonl branch is dead code (harmless). **Survey: 20 usable tasks / 535 turns** (django, matplotlib,
      seaborn, requests, xarray, astropy...) — a real statistical cohort, not n=1. At
      ~0.54s/boundary the 535-boundary capture is ~tens of minutes. NOTE: the pipeline
      is hand-curated deeper than events (per-task hashes/stage-names/counts in
      materialize_swe_jlens_prompts) -> the cohort path is a STREAMLINED capture reusing
      the lens core (VJP + J projection + concept scoring) at AUTO per-turn boundaries,
      not a replay of the hand-curated single-task machinery.
- [x] Tagged all 535 cohort turns (`swe_task_state_v4_cohort_cot_tags.py`) at 99% GPU
      util (2 concurrent workers): 20 tasks / 535 turns / 15 distinct concepts. Overall
      dist: source_localization 206, verification 135, source_edit 51, repair 33, ...
      (`artifacts/cohort-cot-tags-summary-v1.json`; full tags in .cache). Diverse +
      narratively coherent (localize->edit/repair->verify->resolve); source_localization
      +verification = 63% (agents explore+check a lot; mild tagger lean). TAGGER SIDE DONE.
- [x] Internal-capture INPUT path validated: the VJP capture (`run_jlens_nvfp4.py`) takes
      a `--prompts-file` (per-boundary token_ids) + a TASK-INDEPENDENT J checkpoint. Exact
      prompts come FREE from the recorded proxy dumps (`runs/*/proxy_dumps/chat_*.json` =
      exact {messages, tools, chat_template_kwargs} sent to the model) -> POST to the
      server's `/tokenize` -> exact token_ids (validated: 8-msg request -> 13670 tokens). No
      fragile reconstruction, no separate tokenizer. Caveat: proxy dumps exist for some runs
      only + are globally numbered -> need dump->task/turn mapping. `render_request` core is
      generic; the single-task hard-wiring is only validation constants.
- [x] **The multitask capture pipeline ALREADY EXISTS — no from-scratch build.**
      `run_swe_multitask_c1_pilot.sh` -> `materialize_swe_multitask_c1_probes.py` (builds
      the prompts-file from the SAME proxy dumps) -> `run_jlens_nvfp4.sh` (VJP capture) ->
      per-task public/native-report.json. J checkpoint exists + is task-independent
      (`.cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt`). RAN `--prepare-only` (CPU): built
      8 C1 prompts (xarray, scikit-learn, django-13297, ...), 1 boundary/task. Pipeline
      proven. To get per-turn faithfulness: extend to per-turn boundaries (proxy dumps are
      already per-request/turn) at the end-of-thinking, then run the GPU capture
      (coordinate GPU with the tagger server — capture loads its own vLLM).
- [x] **GPU CAPTURE PROVEN END-TO-END on real cohort data.** Stopped the tagger server
      (GPU handoff), ran `run_jlens_nvfp4.sh --lens-kind public` on the 8 C1 prompts:
      8 experiments x 32 layers (16-47) in **54 s**, valid per-layer jacobian_lens/logit_lens
      + scored_vocabulary (status:"failed" is the expected lens-readout marker concept_chain
      requires). The biggest P7b risk (fragile capture generalization) is RETIRED.
- [!] **BLOCKER / DESIGN FORK (paused for user, 2026-07-21).** Both halves' MECHANICS are
      proven (tagger + GPU capture), but the internal concept-chain's VOCAB is hand-curated
      PER TASK: `configs/swe_intermediate_concept_probes.json` is `exploratory_one_task_adaptation`
      for sympy__sympy-13480 — its concept items are that task's specific milestones
      (nameerror_reproduced, defined_identifier_correction, ...) with token forms selected from
      that task's text/trace. So the internal readout does NOT transfer to the cohort as-is.
      Cohort faithfulness needs a GENERAL, task-independent concept-form vocab for the 14
      families (the internal analog of the general auto-tagger). Options for the user:
      (A) build a general concept-form vocab (task-independent single-token surfaces per family;
          I define + verify them in the Qwen vocab) — a design choice on the internal readout;
      (B) reformulate faithfulness (e.g. compare the tagger to a different Qwen-internal signal
          that IS general, like the action-gauge sequence_j readout already at cohort scale);
      (C) accept the strong single-task faithfulness result (0.44/0.60 after the probe fix) +
          the validated cohort auto-tagger as the deliverable; stop the cohort internal capture.
      Per-turn prompts + capture are ready to run once the vocab question is decided.

## P7c — general concept-form vocab (user chose: build general vocab, 2026-07-21)
- [x] `swe_task_state_v4_general_concept_forms.py` (+ test, 4 pass) ->
      `configs/swe_task_state_v4_general_concept_forms.json`: TASK-INDEPENDENT single-token
      surface forms for ALL 14 scorable families (e.g. source_localization=locate/where/
      search/pinpoint; verification=verify/check/validate/confirm; task_resolution=resolved/
      done/complete/solved). Enforces the concept-chain rules: every form a single Qwen token,
      globally unique across families, >=2/family. 14/14 scorable, 0 collisions. This is the
      internal analog of the general auto-tagger.
- [x] General concept scorer `swe_task_state_v4_general_concept_scorer.py` (+ test, 3 pass):
      reads a VJP-lens readout, computes each family's mean form/layer log-prob (concept-chain
      convention) over the general vocab, ranks, LOO baseline-centers -> top-1 per boundary.
      Reuses the concept-chain math; works on ANY task's readout. Full suite: 243 passed.
- [!] **BLOCKER (paused for user, 2026-07-21): no clean per-task per-turn prompt source.**
      Exact per-turn prompts need the exact request messages, but: proxy dumps are globally
      numbered + interleaved + don't self-identify their task (unmappable); the per-task
      qwen-code chats/*.jsonl is incomplete (3 records for django-13297); the trace has the
      conversation but no token counts to validate reconstruction + uncertain full system prompt.
      So an EXACT 535-turn prompts-file is a real data-engineering wall. Everything else is built
      + proven (tagger 535 tags, general vocab 14/14, general scorer, GPU capture). Options:
      (A) BEST-EFFORT per-turn: reconstruct approximate prompts from the trace (concept readout
          is a diagnostic, likely robust to minor prompt diffs) -> n~535, some imprecision;
      (B) CLEAN n=8 via C1: reuse the fully-proven C1 pipeline with the general vocab swapped in
          as score_token_ids -> a clean cohort faithfulness at the C1 boundary (n=8), coarser
          boundary-to-tag alignment;
      (C) accept the deliverable (all components + validated tagger + demo faithfulness 0.44/0.60)
          + this honest finding; stop. + run
      the public+native capture over the cohort; then generalize concept_chain's input
      binding to score the cohort reports; then faithfulness vs the tagger tags.: batch across tasks/boundaries, keep
      GPU util high (profile; target >0.78 effective, fix host-bound stalls), capture only
      the per-turn concept boundaries (not all offsets) to cut cost.
- [ ] Generalize the probe/concept-chain readout per task; tag each task's CoT.
- [ ] Cohort faithfulness (tagged vs internal centered top-1), n = tasks x turns; report v3.
- Monitor loop (/loop 20m): report capture progress each fire; PAUSE only on a real
  generalization blocker or if effective GPU util stays low after a fix.
- PAUSE on genuine design forks (taxonomy fit, poor tagger agreement, capture-cost).
- STOP when cohort faithfulness is computed + in the report, green.

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
