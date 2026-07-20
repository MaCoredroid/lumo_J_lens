# J-lens V4 sealed-control run — supervisor runbook

Operational runbook for the `/loop 10m` supervisor driving the SWE task-state V4
epistemic-chain **sealed control run** to completion. Read this each iteration,
determine current state, advance one step, then yield.

## Goal / stop condition
Complete a full sealed **40-control + 12-fixture** run through all stages and
produce the final **scored public bundle**. STOP the loop (ScheduleWakeup
`stop:true`) once a scorer artifact exists and validates for an active suite.

## Doctrine (do not violate)
- **Fail-closed = immutable.** If any stage errors (nonzero / "FAILED CLOSED"),
  the suite is a preserved failed run. NEVER rerun or overwrite a stage. Diagnose,
  fix the *source* if it's a real serialization/contract bug, then **fork the next
  suite** (r6, r7, …).
- Fixing source changes its sha256 → the old suite's frozen expected hashes will
  no longer match, so a source fix ALWAYS means a fresh suite (new init-suite).
- Generation root (outside repo): `~/.cache/swe_task_state_v4_raw_capture/n60-final/sealed-control-v3-20260720-<tag>`
  Key root (in repo): `~/lumo_J_lens/.cache/swe_task_state_v4_raw_capture/n60-final/sealed-control-v3-20260720-<tag>-keys`

## Stable identity (r6+; adapter repinned after the r5 fix)
- controller `66c46142…`  executor src `2f4a4068…`  executor cfg `cdeee68b…`
- adapter src `77125141…`  adapter cfg `ad818dc8…`  adjudicator src `6a0d840c…`
  (r5 and earlier used adapter src `033252d0…` / cfg `22f22002…`; superseded)
- venv `.venv-vllm/bin/python`; GPU env vars are baked into the driver script.
- If any current source sha differs from the above, a source edit happened →
  you are on a NEW adapter/controller; fork a fresh suite with the *current* shas.

## Stage sequence
Primary lanes (AUTOMATED by `scripts/drive_sealed_primary.sh <tag>`):
1. independent_a: freeze-initial → run-initial → freeze-detail → run-detail → finalize
2. independent_b: same five stages
   Driver is idempotent (skips completed stages) and exits: `0` primaries done,
   `3` a stage failed closed, `2` precondition error.

Adjudicator + scoring (NOT yet automated — drive manually; **never run for real
before, expect novel fail-closed bugs**), via
`scripts/swe_task_state_v4_epistemic_chain_sealed_adjudicator_run_v3.py`:
3. `lock-primaries`  (re-authenticate both primary receipt trees, dual lock)
4. `freeze-verdict`  (freeze the 51-request adjudicator batch = 39 controls + 12 fixtures)
5. `run-round`       (adjudicator initial round on GPU)
6. `freeze-followup` + `run-round`  (repair round, then detail round; may be empty)
7. `finalize-adjudicator`
8. `lock-all`
Then the scorer `scripts/swe_task_state_v4_epistemic_chain_sealed_control_scorer_v3.py`
performs the key-read-last scoring (consumes the one-use nonce) → public bundle.
Inspect each controller's `--help` for exact args; thread each stage's output
sha256 the same way the primary driver does (see its `stage`/`sha` helpers and
the recipe in `docs` / prior commands).

## Each iteration
1. `tag=$(cat ~/lumo_J_lens/.cache/sealed-run-driver-logs/active-tag.txt)` (current suite).
2. Is the primary driver still running?  `pgrep -af drive_sealed_primary` /
   `pgrep -af sealed_control_run_v3.py`.  If yes → it's progressing; tail its log
   (`~/lumo_J_lens/.cache/sealed-run-driver-logs/latest-<tag>.txt` → logfile),
   report briefly, reschedule, done.
3. If not running: read the last driver log + exit state.
   - Primaries not all done and no failure → relaunch `bash scripts/drive_sealed_primary.sh $tag`.
   - Primaries done (`independent_a-primary-receipt.json` and
     `independent_b-primary-receipt.json` both exist) → advance the ADJUDICATOR
     stage that is next (check which artifacts exist), one GPU stage at a time.
   - A stage failed closed → diagnose from the stage log, fix source if warranted,
     fork the next suite (see below), relaunch driver.
4. Scorer artifact exists & validates → **STOP the loop**.

## Forking the next suite (on fail-closed or after a source fix)
```
tag=r6   # next unused
root=~/.cache/swe_task_state_v4_raw_capture/n60-final/sealed-control-v3-20260720-$tag
keys=~/lumo_J_lens/.cache/swe_task_state_v4_raw_capture/n60-final/sealed-control-v3-20260720-$tag-keys
[ -e "$root" ] || [ -e "$keys" ] && { echo "tag taken"; }   # never clobber
mkdir "$root"; mkdir -p "$keys"
# init-suite with the CURRENT executor/controller shas, key-root = ~/lumo_J_lens
# (see the exact init-suite recipe in prior session commands / the primary driver header)
echo "$tag" > ~/lumo_J_lens/.cache/sealed-run-driver-logs/active-tag.txt
bash scripts/drive_sealed_primary.sh $tag
```
The primary driver auto-computes the stable shas from the current source files, so
after `init-suite` it drives the new suite without further hash bookkeeping,
PROVIDED init-suite used those same current shas.

## History
- r1–r4: failed closed (JSON canonicalization order; GPT-OSS Harmony `<|constrain|>`
  sentinel; per-role token budget). All fixes baked into source.
- r5: Primary A LOCKED (22 chains, 1 unknown), then Primary B (GPT-OSS) initial
  run FAILED CLOSED on `validate_sampling_params_against_native_request` — the
  post-engine re-check at adapter `_execute_authorized_production` omitted
  `max_output_tokens=`, so it recomputed the expected cap from the global 768
  instead of the GPT-OSS role cap 2048 (Qwen/Mistral both 768, so only GPT-OSS
  bit). One-line fix: pass the role-resolved `max_output_tokens` (adapter repinned
  033252d0→77125141, cfg 22f22002→ad818dc8). Unit suite green (41+29 subtests).
  r5 preserved as an immutable failed suite.
- r6: forked 2026-07-20 with the fixed adapter; driver relaunched. The GPT-OSS
  Primary B run is the live verification of the r5 fix. **No run has reached the
  adjudicator yet — that remains the frontier after both primaries lock.**
