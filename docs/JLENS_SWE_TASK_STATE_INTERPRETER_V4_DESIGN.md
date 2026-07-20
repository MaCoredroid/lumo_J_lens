# SWE Task-State Interpreter V4 Design

**Status:** pre-freeze design; fixed forecast architecture retained and a
1,794-identity source universe authenticated, but direct transport power and
qualified sparse-repository supply are both NO-GO; not a preregistered protocol,
generation authorization, replay authorization, reliability result, or
validation go decision

**Date:** 2026-07-19

## Purpose

This document records the proposed V4 response to the frozen V3 development
stop. It preserves V3 as an immutable scientific result and separates two uses
of data:

1. the completed V3 N60 development evidence may be used only to design and
   debug V4; and
2. every confirmatory V4 development gate must be evaluated on a newly selected,
   identity-only, disjoint, nonreserved development cohort that has not informed
   V4 target, feature, model, calibration, threshold, or gate choices.

V4 retains the V3 prospective same-request target. It changes the readout from
a current-score classifier into a causal transition-aware procedure, separates
one shared ordinary-logit action policy from branch-specific forecast
calibration and class-specific abstention, and tests only whether a fixed public-J
forecast pool improves proper scores conditional on that unchanged action
policy. This document does not authorize any run. Every unresolved item listed
below must be fixed in a hash-bound V4 protocol before the fresh cohort is
selected or generated.

## Frozen V3 Boundary

V3 remains closed under its existing source and materialization contracts. V4
must not modify a path in
`scripts/check_swe_task_state_v3_development_cohort.py::SOURCE_FREEZE_PATHS`,
the V3 materialization receipt, or any V3 development output. Changed behavior
must be implemented under new V4 filenames and a new V4 output namespace.

The following bindings are inputs to this design and must remain unchanged:

| Artifact | Frozen identity |
|---|---|
| Original V3 source freeze | Git commit `71f6153162627864707cc6f0473cb04f75cadcfd` |
| Post-generation, pre-lens materialization amendment | Git commit `d0ede2f950c34e382060c419e93bee9291fc0ecc` |
| V3 materialization receipt freeze | Git commit `c0643484c094863bae6d8026015ba9a719d37c4a` |
| V3 interpreter protocol | `configs/swe_task_state_interpreter_v3.json`, SHA-256 `9d8b0a7d5c45dc192365429af27c6193de752cc160458eff8e21807d37662b1d` |
| V3 action protocol | `configs/swe_task_state_v3_action_probes.json`, SHA-256 `0ebd258a2b46beb2a9be3d42cab24680803a2f971cb21e96acecb78e19cd81bf` |
| V3 development cohort declaration | `configs/swe_task_state_v3_development_cohort.json`, SHA-256 `d95e40730308be4cb252b7f59923063689b2098cf5b79405729b219557f61e10` |
| V3 analyzer | `scripts/analyze_swe_task_state_v3.py`, SHA-256 `53c7d41688f6c5ab21f7ad029d343af06e9b13c777fd2e5517ff8d5254ad9e6c` |
| V3 materialization receipt | `validation/swe-task-state-v3-development-materialization.json`, SHA-256 `3f7d6dbecb3157badc7e0db09444c8b77bd6f51c8941604b3a5bf81bf30e7812` |
| V3 prompt bundle | SHA-256 `17f664b3029220458ff62b8e80a90ec5e796f0372217f02b727d6589df38d3d0` |
| V3 prompt summary | SHA-256 `b70b3751e9cb5a896bc264d2214b425cc7a690ff8026c6c972d3a21cbd22d44d` |
| V3 replay merge manifest | SHA-256 `302437210d582081e4c343cadb30afecf9e7e0bfb18a8d7b12fdb10c3d782e6f` |
| V3 public replay report | SHA-256 `7c943132163749f69bd35e4fa2e52bcfee2318fe349fa77603324a37ffaabe46` |
| V3 no-bootstrap development diagnostic | SHA-256 `b14016f09a67dfb9ad60f583fff27ba8fac075d97ac6b206b3423c88d6e48ae4` |
| V3 development stop decision | `validation/swe-task-state-v3-development-decision.json`, SHA-256 `2eccd406f7963aef83b764b0bd4d94d93d18e6c16c7fc7ecc24d7686e872711d` |

The bound V3 stop decision is final for V3:

- all predeclared support gates passed;
- the `history_j` point gates failed for balanced accuracy (`0.7016810912`
  versus `0.75`), edit recall (`0.5110350720` versus `0.65`), and selected
  accepted accuracy (`0.8073816159` versus `0.85`);
- a 1,000-draw model-refit bootstrap was not run because intervals cannot
  rescue failed deterministic point gates;
- no V3 model fit is authorized;
- V3 makes no operational reliability claim; and
- reserved validation remains closed.

The V3 N60 evidence is design-only for V4. It must never be counted as an
independent V4 development screen, pooled with the fresh V4 cohort, or used to
rescue a failed fresh-cohort decision.

## Completed Initial B2 Design Screen

The first V4 B2 design screen is complete and recorded at
`.cache/swe_state_interpreter_v4_design/v3-n60-sequence-b2-design.json`,
SHA-256
`8ec24b65b71b831e5206af353857766a58fbc0a343acee6f6ab7071cccb3489e`.
It is a nonconfirmatory design artifact over the already-used V3 N60 evidence,
not a frozen V4 protocol result, fresh-development screen, generation
authorization, reliability result, or validation result. Its support checks
passed, but the proposed operational candidate failed the balanced-accuracy
point gate (`0.7491604610 < 0.75`) and selected accepted-accuracy point gate
(`0.8421313478 < 0.85`). It therefore cannot support promotion or an operational
claim. The dynamic-alpha B2 action/fusion design represented by that artifact is
superseded by the shared-action, fixed-geometric-pool proposal below; the
artifact and its failures remain part of the design history and must not be
overwritten or reinterpreted as confirmatory evidence.

A later approximate fixed-pooling calculation reported candidate-minus-
reference deltas of `-0.004418` for multiclass NLL and `-0.002202` for
multiclass Brier score, with improving signs in `10/10` repository
comparisons. An early conditional all-286 calculation reported joint
proper-score power of `0.925`. That estimate is superseded for planning: it
conditioned on fixed repository effects and did not resample task effects
within the sparse prospective repositories.

## Completed Fixed-Pool Design Screen And Power Decision

The contract-closed fixed-pool screen is complete at
`.cache/swe_state_interpreter_v4_design/v3-n60-geometric-a020-shared-action-contract-closed.json`,
SHA-256
`e1436a9c7fe763a4fab6ae505af17395a4290faefd0734b230d22127bff0cf8c`.
It binds design config raw SHA-256
`cca7687e9c061e7f600469d4db89312e9b55739034bf09fd7a81be2a1f86d04d`
and canonical SHA-256
`6378cc92f78f6e662dc9493756a240ec1304ccbc9735ce17ef5932290e81624d`.
The final runner SHA-256 is
`cd5434f50a131d4bd9e762a1d63c3d86f6be662e045fece8ee6854365e6e0ad3`
and the final evaluator SHA-256 is
`6f3ce5b9baad60fc83f87b0bc29d03b2d07ac0a918afe63404c270647cd548ea`.
All 58 focused V4 tests pass.

An earlier no-clobber fixed-pool artifact at
`.cache/swe_state_interpreter_v4_design/v3-n60-geometric-a020-shared-action-design.json`,
SHA-256
`31df5db7b4e680971f23e76001dacdc4b74a5fb2283047ad877e476cda464533`,
has an exactly identical prediction payload. It is superseded because its
promotion metadata blocked a candidate threshold fallback but did not yet
encode the documented symmetric block for a reference threshold fallback.
The corrected evaluator requires the shared full action rule and both full
threshold branches to pass. Both branches pass in the final artifact.

The final screen independently reproduced all 8,030 stored row-probability
records, every probability and acceptance hash, all metrics, the exact
normalized `0.80`/`0.20` geometric pool, and exact candidate/reference raw
action and decision-distribution identity. Support and all design-only point
checks pass. Candidate and reference action metrics are exactly equal. The
candidate point results include:

| Quantity | Result |
|---|---:|
| Balanced accuracy | `0.7597266130` |
| Inspect recall | `0.7635335704` |
| Edit recall | `0.7127927369` |
| Check-or-finish recall | `0.8028535317` |
| Selected accepted accuracy | `0.8588700875` |
| Selected coverage | `0.7554432902` |
| Multiclass NLL | `0.5248830505` |
| Multiclass Brier | `0.3013183541` |
| Candidate minus reference NLL | `-0.0061785402` |
| Candidate minus reference Brier | `-0.0021992920` |

Both proper-score differences improve in `10/10` repository comparisons. Raw,
uncalibrated pooling also improves both scores in `10/10`, so the direction is
not created by temperature selection. The result is nevertheless design-only
and selection-biased because the fixed `0.20` J weight was chosen using the
same V3 N60 evidence. Operational margins are thin: the full shared action
selection has balanced accuracy `0.7517323014`, only `0.0017323014` above its
inner floor; four of ten outer shared-action selections use explicit
fallbacks; and outer accepted accuracy is only `0.0088700875` above its point
gate. A fixed-prediction N60 hierarchical Bayesian bootstrap puts the balanced-
accuracy lower bound at `0.6945259292`, below the required `0.70`.

The identity-only post-V3 pool contains 286 tasks with this exact allocation:
Astropy 7, Django 177, Matplotlib 16, Flask 1, Xarray 5, pytest 2,
scikit-learn 16, Sphinx 23, and SymPy 39. The reproducible prospective power
screen is recorded at
`.cache/swe_state_interpreter_v4_design/v3-n60-geometric-a020-paired-proper-power-screen.json`,
SHA-256
`863ff5db0399e11581307ca3789e116459092d641aa67c62d741f4c5c57fa6bc`.
It binds power config SHA-256
`d696ca7edd58be17abd29cc4908322d4741a54fee2d90c89a7e05dcf61fa6574`
and analyzer SHA-256
`0a21926c03578490b4423fa07d54ff7120babaee1e007689ad7b161d3c2191d7`.
The screen drew whole task-level paired loss profiles within each matching
repository at those exact counts. Each simulated cohort then used 1,000
independent `Gamma(1)` repository weights and `Gamma(1)` task-within-repository
weights and passed only when both 97.5th-percentile proper-score differences
were strictly below zero. The predeclared decision requires at least `0.80`
joint power in every Flask scenario. Over 10,000 cohorts, joint conditional
proper-score power was only `0.4092` with an exchangeable Flask proxy, `0.3990`
with a neutral Flask J effect, `0.4737` with a Django-domain Flask proxy, and
`0.1772` under a p90-adverse Flask scenario. The resulting decision is
`NO_GO`. These estimates still freeze existing predictions and omit model-refit
variance, so they are optimistic rather than conservative evidence for the
eventual required full-refit procedure.

A separate full-OOF raw-probability sensitivity screen independently reselected
candidate and reference temperatures. On N60 it retained negative conditional
upper bounds and selected the same candidate/reference temperature in `96.65%`
of draws. In the all-286 Django-proxy scenario, however, prospective joint
proper-score power fell from `0.6686` with fixed `T=1` to `0.5490` with
temperature reselection. Selection uncertainty therefore does not repair the
cohort design and can reduce its power even before model refitting.

An additional exploratory 250-cohort screen included the unchanged absolute
point and interval gates using the existing action and acceptance outputs. In
the exchangeable-Flask scenario it estimated paired proper-score power near
`0.36`, absolute-interval power near `0.11`, and joint point/absolute/paired
power near `0.03` (joint Monte Carlo standard error about `0.011`). This small
screen is diagnostic, not a protocol result, but it confirms that the proper-
score shortfall is not the only prospective risk.

The decision is therefore:

- **GO** for the fixed `0.20` geometric J forecast architecture as the current
  V4 design direction;
- **NO-GO** to freeze the complete V4 protocol or select/generate the currently
  available 286-task cohort; and
- **NO-GO** to spend compute on a nominal full-refit power run for that cohort,
  because even the optimistic fixed-prediction proper-score screen is below
  `0.50` joint power.

Repetition within an existing task cannot repair this deficit under the
hierarchical task estimand. Freeze work requires a new identity-only source of
enough independent tasks in the sparse repositories, or an explicitly new
scientific estimand and support contract justified before seeing new outcomes.
Neither change is authorized by this design record. The fresh cohort remains
unselected and ungenerated, and reserved validation remains closed.

## Completed Identity-Only Source-Universe Audit

The broader official SWE-bench test source has now been authenticated without
selecting a task or interpreting task payload. The fail-closed analyzer at
`scripts/analyze_swe_task_state_v4_source_feasibility.py`, SHA-256
`c9235bf6a96888338ecd7b3bc692c1c5b26c1c63b192486e9807240bf541a1c2`,
binds config SHA-256
`165f8c5d179c24855e8edfec30d22c078706b3703969f655af9a062de725afb8`.
It accepts only the exact official Parquet files and calls PyArrow with the
literal projection `instance_id, repo` for both inputs. The full test source is
SWE-bench revision `e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6`, file SHA-256
`db4f70ef735b3162c74801ddcdf8d7bae8d704193788c6d844f898c20b571cbb`.
The exclusion source is the entire SWE-bench Verified test set at revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`, file SHA-256
`a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd`.

The no-clobber aggregate result is
`.cache/swe_state_interpreter_v4_design/swe-task-state-v4-source-feasibility.json`,
SHA-256
`e7b2f6a0fecbc7c4ff10e41463ff0c83898a5a59b070856ac820279b4652be89`.
It proves that all 500 Verified identities are an exact repository-preserving
subset of the 2,294 full-test identities. Subtracting the whole Verified set,
rather than consulting any reserved-stage membership, leaves 1,794 identities
with set SHA-256
`953b83337651cfa8e68f812f30e3ba1394a8a08e1f66980680832a1d6bd02861`:

| Repository | Available identities |
|---|---:|
| Astropy | 73 |
| Django | 619 |
| Matplotlib | 150 |
| Seaborn | 20 |
| Flask | 10 |
| Requests | 36 |
| Xarray | 88 |
| Pylint | 47 |
| pytest | 100 |
| scikit-learn | 197 |
| Sphinx | 143 |
| SymPy | 311 |

All 16 focused source-feasibility tests pass, and all 89 V4 tests present at
that design checkpoint pass together under the pinned readout environment. An independent
contract review verified that no all-column Dataset, Arrow IPC, or post-load
selection path remains. The output contains no raw instance identifiers and
sets task-payload access, cohort selection, generation, reserved-membership
access, reserved-validation access, confirmatory interpretation, and
operational-reliability claims to false.

`source_feasibility_passed` authenticates only this identity-set subtraction.
It does not establish task executability, permit task selection, or repair the
power and unseen-repository transport risks. In particular, the authenticated
source contains only 10 Flask and 20 Seaborn identities. Those repositories
were unseen in the V3 N60 task-loss profiles, so their paired proper-score
effects cannot honestly be replaced by a favorable proxy merely because the
identity supply is now known.

An exploratory identity/infrastructure-only top-up audit did not close that
gap. All counts and intersections below were computed from explicit column
projections; no preview payload informed a count, hash, overlap, or source
decision:

- [SWE-bench-Live](https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live/tree/a637bd46829f3132e12938c8a0ca93173a977b8e)
  revision `a637bd46829f3132e12938c8a0ca93173a977b8e` has 1,887 unique
  full-split identities and contributes three disjoint Flask issue tasks but no
  Seaborn tasks. Its two full shards have SHA-256
  `9371afaef65bb6e21417fb76e5cbd89a5bc5241e197db3ead4eb0353007e740c`
  and `12e4956a1761efaeb121afe13a5371d8d439a7708b4638397d0b3d6cce1e8407`.
  The Parquet schema exposes no task-level image or environment identifier, so
  the project's Docker execution infrastructure still requires a separate
  task-level availability check.
- [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench/tree/89cdfbab4ab1bd8f5a658bb212d1b63624f4f881)
  revision `89cdfbab4ab1bd8f5a658bb212d1b63624f4f881` contributes no Flask
  tasks and 47 disjoint Seaborn issue tasks. All 47 have a nonempty
  `environment_setup_commit` and non-null `install_config`, but none has a
  populated `docker_image` or `image_name`; they are build-configured rather
  than prebuilt-image-certified. Its two test shards have SHA-256
  `39d4791f12cf5ee2a2e56d47eeef559642a800534ff053e1ae3acab0a0c87067`
  and `c50af8bffbfe70fc3a89b2e47825f299bc04c1058318fe2263b7e4857f8193d7`.
- [FEA-Bench](https://huggingface.co/datasets/microsoft/FEA-Bench/tree/55ee2f78126a3ecdeac6f595fa1ba6ae5c600bad)
  revision `55ee2f78126a3ecdeac6f595fa1ba6ae5c600bad` contributes one
  disjoint Flask and five disjoint Seaborn identities with environment commits,
  but these are feature-implementation tasks rather than the same issue-repair
  stratum. Its Parquet SHA-256 is
  `22ebac756180b3dfd63d1c7d57a1ed7cf5707a28c1e4ce59929dc30955cf0836`.
- [SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith/tree/ea6d7173829c7ec8fa16c22055699ff2e9188091)
  revision `ea6d7173829c7ec8fa16c22055699ff2e9188091` has 59,136 synthetic
  tasks across 222 repositories but no canonical Flask or Seaborn task.

The recommended staged design needs five Flask and ten Seaborn tasks for a
transport pilot in addition to an untouched confirmation allocation containing
ten Flask and twenty Seaborn tasks. The official complement therefore needs at
least `+5` Flask and `+10` Seaborn identities. SWE-rebench can cover the
identity-level Seaborn shortfall, but SWE-bench-Live supplies only three Flask
identities; even mixing in FEA-Bench's different task stratum raises the Flask
total to only four. No audited source or defensible same-stratum combination
therefore satisfies the qualified top-up. This audit does not authorize source
mixing, task selection, environment building, or generation.

## Completed Twelve-Repository Transport And Supply Screen

The fail-closed transport planner is complete at
`.cache/swe_state_interpreter_v4_design/swe-task-state-v4-transport-power-screen.json`,
SHA-256
`a58a2cb71a6ebee61cf6bb4dddbc26d3418cabf3551a2029398bf8f7313e7b18`.
It binds config SHA-256
`5fb51e564f927f60474708bc47454fffada7d87e1fd9bd9b5907c19287191525`
and analyzer SHA-256
`4fe20eabd9f86e5bd6794e16dbd7055296b292936b1f3f20b4473fe14a8ceeb1`.
All 15 focused transport tests pass, including a nonuniform golden test that
distinguishes perfectly coupled from independent unseen-repository transport.
The 89-test V4 suite present at that transport checkpoint passes. The artifact contains no raw task identity and
authorizes neither a pilot, confirmation, selection, nor generation.

The prospective confirmation allocation caps each repository at 20 tasks,
subject to the authenticated official supply. It contains 20 tasks from each of
the ten observed repositories, 20 Seaborn tasks, and all 10 Flask tasks, for
`N=230`. The screen uses NumPy 2.5.1, SciPy 1.18.0, six independent PCG64 child
streams from `SeedSequence(20260729)`, 10,000 outer cohorts, 1,000 inner
hierarchical `Gamma(1)` draws, and a linear 97.5th-percentile strict paired NLL
and Brier gate. A scenario passes planning only when its Bonferroni-adjusted
one-sided exact Clopper--Pearson lower bound is at least `0.80`.

| Unseen Flask/Seaborn transport scenario | Joint passes | Power | Exact adjusted lower bound |
|---|---:|---:|---:|
| Perfectly coupled Django-domain proxy | 9,574 / 10,000 | `0.9574` | `0.952651` |
| Perfectly coupled exchangeable known-task proxy | 8,174 / 10,000 | `0.8174` | `0.808574` |
| Neutral zero J effect | 8,899 / 10,000 | `0.8899` | `0.882689` |
| Both unseen repositories at empirical p90 adverse | 1,498 / 10,000 | `0.1498` | `0.141882` |

An independent-latent sensitivity also matters: its exchangeable proxy passes
only 7,492 / 10,000 cohorts, with power `0.7492` and exact lower bound
`0.739343`. Thus even a future pilot that excludes the fixed p90 profile would
still have to resolve whether Flask and Seaborn transport jointly or
independently; a favorable coupled point estimate is insufficient.

The p90 problem is structural rather than a request for more tasks within the
same repositories. The known ten-repository equal-weight point is
`-0.0061785402` NLL and `-0.0021992920` Brier, while the empirical p90 task
profile is `+0.0022797268` and `+0.0031221769`. Giving that profile to both
unseen repositories leaves a favorable infinite-task equal-repository point,
but the repository-Gamma 97.5th-percentile Brier difference is
`+0.00005254`. The strict Brier `< 0` gate therefore fails even after all
within-repository task uncertainty vanishes.

The recommended outcome-informed pilot remains 5 Flask plus 10 Seaborn tasks.
It must be disjoint from, and may never be pooled into, a later confirmation.
Keeping 10 Flask and 20 Seaborn tasks untouched for the `N=230` confirmation
requires total supplies of 15 Flask and 30 Seaborn, so the official complement
is short five and ten respectively. Pilot outcomes may narrow the transport
envelope only through a separately frozen confidence-set gate, followed by a
new prospective power screen. They may not directly authorize confirmation.

The production decision is therefore both
`NO_GO_RETAINED_TRANSPORT_ENVELOPE` and
`NO_GO_QUALIFIED_SOURCE_SHORTFALL`. Reserved validation remains closed. The
next admissible design step is a separate, content-blind curation and
environment-qualification contract for additional same-stratum Flask tasks;
it is not task selection or generation under this screen.

## Data Separation And V4 Lifecycle

Before any confirmatory V4 development gate is evaluated, V4 must use a fresh
development cohort satisfying all of the following:

- selection uses only `instance_id` and `repo` from the pinned official dataset
  revision;
- the cohort is disjoint from every previously used, allocated, or excluded
  task, including the V3 N60 cohort;
- selection does not consult task text, generated trajectories, action labels,
  lens values, model scores, difficulty proxies, or evaluation results;
- cohort derivation, membership, ordering, campaign partition, and image pins
  are reproduced by a checker from frozen inputs; and
- the final V4 protocol, analyzer, tests, requirements, cohort checker, and
  generation inputs are committed before V4 development generation begins.

The reserved validation stage remains closed throughout V4 design, engineering,
fresh development generation, replay, and analysis. A passing V4 development
result would establish only eligibility to propose a separate locked validation
lifecycle. It would not itself open or execute validation.

The design-only V3 artifacts may be consumed read-only for implementation
parity, model-family selection, feature debugging, and power analysis. Any
choice made after examining them must be fixed before fresh-cohort selection.
All V4 writes must use new paths, including a dedicated root such as
`.cache/swe_state_interpreter_v4_development/`; aliases, symlinks, hard-link
aliases, and writes into V3 or validation roots must fail closed.

## V4 Prediction Target

V4 keeps the V3 target unchanged. At every numerically stable,
feature-complete final-prompt boundary, it predicts the observable action in the
ensuing completion of that same request:

1. `inspect`;
2. `edit`; or
3. `check_or_finish`, collapsing the source actions `validate` and `finalize`.

The target is prospective from the final-prompt boundary. Later-request actions
must not define this target. An unknown, missing, or unclassified ensuing action
does not suppress inference: the procedure emits a prediction, records the
unknown status, and excludes that row from fit, calibration, threshold
selection, and label-dependent metrics. Complete consecutive prompt bundles are
required wherever causal state depends on prior requests.

V4 does not claim to reconstruct hidden prose or chain of thought. It forecasts
an observable completion action.

### COT-like observable reasoning-trace sidecar

The V4 scope now includes a separate, post-prediction reasoning-state sidecar.
The sidecar is specified by
`configs/swe_task_state_v4_reasoning_trace.json` and implemented in
`scripts/swe_task_state_v4_reasoning_trace.py`; it does not modify the frozen
V3 paths or the V4 action/forecast procedure. Its output is explicitly a
high-level, COT-like trace of observable action-state evidence, not recovered
private chain-of-thought, subjective emotion, task-specific intent, or hidden
prose.

The only phase aliases supported by the calibrated three-class action target
are:

- `inspect` -> `information_gathering`;
- `edit` -> `implementation`; and
- `check_or_finish` -> `verification_or_completion`.

The sidecar does not split verification from completion and does not emit
`hypothesize`, `stall`, or literal emotional states as calibrated phase labels.
Those distinctions require separately defined, causally trained, held-out
targets. It may emit deterministic transition events such as
`inspection_to_implementation`,
`implementation_to_verification_or_completion`,
`reconsideration_or_rework_like`, `uncertain_transition`, and
`recovery_like`. Specific transition names require an immediately preceding,
consecutive accepted readout boundary; they describe forecast-state changes,
not observed actions or actual rework. Every rationale is produced from a
fixed template and carries
the exact numeric template inputs needed to reproduce it. Free-form rationale
generation, first-person claims, task-specific explanations, and inferred
intent are forbidden. The rendered sentence itself states whether its inputs
are synthetic/unverified or design-only and transport-unconfirmed.

For calibrated forecast `q`, shared decision distribution `d`, and selected
action `a* = argmax(d)`, the sidecar reports these exact quantities:

```text
decision_confidence = q[a*]
forecast_doubt = 1 - decision_confidence
diffuse_uncertainty = entropy(q) / log(3)
ambivalence = 1 - (q_top1 - q_top2) / (q_top1 + q_top2)
forecast_volatility = JSD(q_t, q_previous) / log(2)
source_disagreement = JSD(p_sequence_logit, p_sequence_j) / log(2)
```

`decision_confidence` means estimated correctness of the selected next-action
class; it is not a claim of experienced self-confidence. `forecast_doubt` is
its probability complement, not felt doubt. Entropy, ambivalence, volatility,
and source disagreement are descriptive indices rather than probabilities of
a mental state.

An optional activation-load-like diagnostic combines diffuse uncertainty,
forecast volatility, source disagreement, and robust-scale-bounded ordinary
and public-J activation innovation. It is comparable only when all five
components are present and a training-only empirical reference distribution
maps the raw mean to a percentile. First-boundary volatility is unavailable,
never imputed as zero. The field is explicitly named
`activation_load_like_not_emotion_probability`; it must not be presented as
stress or any other emotion. The current implementation has no authenticated
causal innovation-scale/ECDF artifact, rejects bare caller-supplied innovation
or reference values, and therefore emits only a partial, noncomparable raw
diagnostic with a null percentile.

Hesitation-like, trajectory-continuation, and recovery-within-two probabilities
are present in the schema but remain null with explicit
`unavailable_unfitted_*` statuses. They cannot become available until separate
causal heads pass fresh, nonreserved, out-of-repository calibration and
selective-accuracy gates. `recovery_like` remains unavailable and null in the
current implementation. A later protocol may enable that descriptive event
only after it binds an episode horizon, an authenticated earlier high-load
state, a later accepted lower-load regime, a fixed confidence rise, and
movement to implementation or verification/completion.

The authenticated path is bound to the exact design artifact at
`.cache/swe_state_interpreter_v4_design/v3-n60-geometric-a020-shared-action-contract-closed.json`,
SHA-256
`e1436a9c7fe763a4fab6ae505af17395a4290faefd0734b230d22127bff0cf8c`.
That whole-artifact binding authenticates the primary forecast/calibration,
shared decision, abstention, and `sequence_logit`/`sequence_j` diagnostic
branches. The generic row API is marked synthetic/unverified and cannot emit
calibrated provenance. Bare caller-supplied diagnostics never receive an
authenticated status.

Within either path, the sidecar reads only row/task identity, request index,
`q`, `d`, and the V4 abstention decision. Source probabilities are supplied
through a separate two-field allowlist; in the authenticated path they are
derived internally from the bound artifact. Existing prediction fields
containing current/future labels, auxiliary diagnostics, completion reasoning,
tool outcomes, usage, finish reasons, official outcomes, later actions, or task
text are ignored and covered by mutation-invariance tests. Both row and task
identities are hashed in emitted records. The rendered phase is nulled when the
V4 selective action rule abstains, while its probability vector and explicit
reason are retained.

Current outputs remain design-only and transport-unconfirmed. The sidecar sets
`private_chain_of_thought_reconstructed=false`,
`subjective_emotion_inferred=false`, and
`operational_reliability_claim=false` on every row. Its focused tests, combined
with the V4 suite present at that reasoning-trace checkpoint, pass 121/121 tests without opening
reserved validation. The exact authenticated end-to-end design replay produces
1,606 finite sidecar rows across 60 tasks: 1,202 pass the existing V4 selective
action threshold and 404 explicitly abstain. Every load percentile and recovery
field remains null because the required authenticated training artifact does
not yet exist. Row-level reliability intervals and selective-accuracy/coverage
lower bounds are also null until a fresh transport-confirmatory model-refit
bootstrap exists.

### Ordered semantic concept-chain sidecar

The phase trace above is not the full COT-like target: it renames three
next-action classes and therefore cannot by itself show a chain of task
concepts. V4 now also has a separate semantic-chain sidecar specified by
configs/swe_task_state_v4_concept_chain.json, implemented in
scripts/swe_task_state_v4_concept_chain.py, and covered by
tests/test_swe_task_state_v4_concept_chain.py. It leaves the action trace,
V3 closure, and all reserved validation paths unchanged.

The current authenticated input is the exact permitted one-task intermediate
analysis at .cache/swe_jlens_intermediate/analysis.json, SHA-256
29f7cb2f1ffe7948f7836c49db46020864fd4e2876ec060ca03637edd5e034db,
paired with configs/swe_intermediate_concept_probes.json, SHA-256
2cae42b3b3f559209a81ae80d55800ff215be0786a28865b5f95d0a16fdba1cc.
The sidecar additionally authenticates the exact paired public and native
reports, SHA-256
16a8c781db6c3dea9dd6602a1e6a113d1a7b29b5ada1d1668168a0ff0d9290b7
and
a307b236c259bc58703ba1449ecfe404f6387376a1db10d3f675b0c1c21b5068.
The analysis binds the exact prompt bundle and dense trajectory hashes. The
sidecar revalidates the whole-artifact identities, ten exact-prefix
coordinates, fixed layers 16 through 47, scored-token coverage, public/native
pairing, and per-boundary numerical fidelity before inference.

The first semantic-chain draft was rejected after an independent audit. It
ranked only each boundary's human-specified positive concepts, so its apparently
coherent narrative was a visualization of retrospective labels rather than a
decoded chain. It also upgraded the minimum rank of one generic synonym over 32
layers into proposition-like language. Those paths are no longer used.

Protocol version 2 instead ranks one common concept ontology at every boundary.
The registry contains seventeen concept families. To prevent shared generic
tokens from being attributed to several concepts, every token ID registered to
more than one family is excluded. A family is scorable only with at least two
remaining unique forms. Fourteen families pass; typographical-error has only
one unique form, while repair-success and repair-summary have none, so all three
fail closed. For every scorable family and each source, the score is the
arithmetic mean of the token log probabilities for every unique form over every
fixed layer. The score is not a probability or confidence.

The public-J, native-J, and ordinary-logit branches each rank the same fourteen
families. A source-agreement candidate is emitted only when both paired reports
pass strict adapter fidelity at that boundary and public J and native J have
the same top-one family. Otherwise the semantic readout abstains. Ordinary
logit is retained in every sentence as a baseline but never affects candidate
selection. Human positive labels and leakage classes travel through a separate
evaluation-only argument and cannot affect rankings, selection, relations, or
rendering. A label-permutation test proves that changing all positives changes
only evaluation, never the candidate chain.

Renderer language comes only from conservative names such as
defined-identifier concept family. It does not turn a token association into a
claim that the model located, understood, believed, decided, or felt anything.
Task text, state descriptions, completion prose, tool output, winning token
surfaces, target tokens, future actions, and outcomes are not renderer inputs.
Every sentence retains source scores, the ordinary-logit top family, numerical
eligibility or abstention reason, and evidence status. The sparse chain sentence
retains exact coordinates and the number of intervening abstaining registered
boundaries; later evidence is never smoothed or backdated.

The authenticated result falsifies, rather than establishes, a reliable
concept-chain claim:

- only 5/10 boundaries pass paired strict adapter fidelity;
- public/native J top-one agreement selects 3/10 boundaries;
- 2/3 selected candidates match the retrospective positive labels;
- public-J, native-J, and ordinary-logit top-one accuracy are all 0.60 on the
  same five strict-fidelity boundaries;
- the selected sequence is focused-validation at request 1,
  defined-identifier at request 3, and focused-validation at request 9, with
  one and five intervening abstentions respectively; and
- request 1 is a false positive, no confidence intervals exist, no incremental
  J value is shown, and no operational reliability claim is available.

Every registered boundary is task explicit, tool-output explicit/implicit, or
a teacher-forced lexical positive control. The non-prefix-explicit count is
zero. The ontology, positives, and coordinates were retrospectively derived
from the completed visible trace. Consequently this artifact provides no
evidence of concepts absent from the JSONL-visible prefix. Its mandatory scope
sentences state both facts:

> This sparse sequence is an uncalibrated common-ontology vocabulary-readout
> candidate chain, not private chain-of-thought, hidden content, intent, causal
> explanation, or emotion.

> The ontology, positive evaluation labels, and ten coordinates were
> retrospectively specified from the completed visible trace; they provide no
> evidence of concepts absent from the JSONL-visible prefix.

### Current authenticated development evidence

The raw-residual/public-J capture and label-independent projection described
above now exist for development data. They contain 1,708 authenticated
final-prefix states, of which 1,606 are numerically stable feature rows across
60 tasks and ten repositories. The feature bundle keeps labels, completions,
outcomes, repository/task identity, and future actions out of every numeric
model matrix. Exact-prefix dense states remain teacher-forced replay states,
not captured generation-time hidden states.

COT/COT-like structure and affective-state questions are separate goal lanes:

- **Observable process/COT-like proxy.** On 1,549 available rows, the strongest
  combined vocabulary baseline (`sequence_logit_j`) has NLL `0.394364` and
  Brier `0.247997` for the ensuing visible rationale-language marker. Adding
  current raw activations gives `0.346319` / `0.217629`; adding current
  public-J activations gives `0.337220` / `0.210125`. Both nested candidates
  improve both losses in all ten held-out repositories. These variants were
  added after the initial point results, and the repository intervals are
  fixed-OOF, post-hoc development diagnostics without model-refit or selection
  uncertainty. The target is a visible surface-language regex proxy, not a
  sentence, semantic chain, private COT, hidden thought, or understanding.
- **Prefix-hidden identifier audit.** A separate fixed-choice audit scores 74
  stable future-target instances across 24 target trajectories, 17 tasks, and
  eight repositories after verifying that every target and retained foil
  surface is absent from the complete current prefix. Its candidate set is
  retrospectively supplied by the future trace. Ordinary-logit top-one is
  `0.576968` and public-J top-one is `0.504832`; ordinary-logit NLL is
  `0.981827` and public-J NLL is `1.116754`. The paired intervals include zero,
  and public-J has no advantage. This is negative fixed-choice identifier
  evidence, not decoded concepts, propositions, sentences, or COT.
- **Semantic epistemic-action chain.** The prospective target is an explicit
  observation `E` followed by a supported/refuted/narrowed conclusion `H`
  that motivates an inspection/edit/validation action `A`, with prefix novelty
  adjudicated separately. A hash-bound synthetic codebook freezes positive,
  negative, ambiguity, span, and earliest-chain decisions before labeling.
  Exactly 1,548 blinded completion packets have been
  materialized from 1,549 eligible completions; one malformed-prefix row is a
  frozen unknown rather than a negative. Target annotations and decoder fitting
  have not run, so semantic sentence/chain decoding remains unestablished.
- **Affect, confidence, doubt, and pressure.** The separate randomized protocol
  has selected 120 label-independent boundaries and assigned 1,440 matched
  evidence-by-pressure prompts. Generation, activation capture, and label
  extraction have not started. It can eventually test objective calibration,
  rechecking, pressure sensitivity, and explicit affect language; it cannot
  establish experienced emotion, felt confidence, doubt, or stress.

Accordingly, the development artifacts support a narrow claim that latent
states carry incremental information about an upcoming visible process-language
proxy beyond the strongest current word-probe baseline. Open-ended phrase or
sentence recovery, semantic concept-chain decoding, private COT, hidden
understanding, subjective emotion/stress/confidence/doubt, causal interpretation,
transport confirmation, and operational reliability remain null or unavailable.

The authoritative observable route is recorded by
`.cache/swe_task_state_v4_raw_capture/n60-final/observable-current-artifacts-v1.json`;
older V2 observable bundles and reports are retained but explicitly marked
historical and superseded. At this checkpoint, the complete V4-pattern test
suite passes 252 tests with one expected Torch/vLLM-only skip in the readout
environment, without opening reserved validation.

## Causal State And Feature Contract

All features for request boundary `t` must be computable before the ensuing
completion at `t`. Task state is reset at the first boundary of each task. State
updates occur only after the current prediction has been emitted.

### Common history state

Every branch receives the exact causal prior-action state used by V3, computed
before the current action update. It includes cumulative prior action counts,
the previous known action representation, cumulative and previous unknown-action
indicators, prior edit/validation state, and turns since prior edit/validation.
No current or future action label may enter the feature vector.

### Per-source temporal score state

For each score source independently, V4 proposes the same frozen layers,
concept order, token forms, and forty-value compact layer-shape reduction used
by V3. Each source first yields one raw 96-wide vector: 24 frozen layers by four
source concepts. The sequence block contains three independently compacted
raw-space quantities:

- `compact(raw_current_t)`;
- `compact(raw_current_t - raw_current_(t-1))`, using the immediately previous
  numerically stable, feature-complete raw vector in the same task; and
- `compact(raw_current_t - raw_ema_(t-1))`, where `raw_ema_(t-1)` is based only
  on earlier stable 96-wide vectors.

The raw 96-wide delta and raw 96-wide prior-EMA deviation must be computed
before their separate compact reductions. Subtracting two already compacted
forty-value summaries is forbidden because the nonlinear summaries do not
commute with subtraction.

The proposed EMA coefficient is `0.5`, matching the implemented V2 temporal
feature. After prediction at `t`, the raw state updates as
`raw_ema_t = 0.5 * raw_current_t + 0.5 * raw_ema_(t-1)`. At the first stable
boundary, both raw differences are fixed 96-wide zeros and `raw_ema_t` is
initialized to `raw_current_t`. Sequence variants also carry an explicit
`no_previous_stable_row` indicator and causal `log1p_request_gap`; neither is
added to `history_only`. Every stable, feature-complete boundary updates sensor
state after feature construction, including a boundary whose action label is
unknown. This initialization, missing-boundary behavior, feature order, and
state update order must be covered by exact tests before freeze.

The public-J and ordinary-logit blocks must use identical layer/concept
reductions and temporal update rules. Future trajectory fields, forecast
horizon, current action labels, later actions, evaluation results, and
post-completion text are forbidden from features.

## Candidate Procedures

The confirmatory V4 analysis will expose these four matched base variants in a
fixed order:

| Variant | Width | Feature blocks |
|---|---:|---|
| `history_only` | 14 | common causal history state only |
| `sequence_j` | 136 | history, public-J current/delta/prior-EMA-deviation blocks, and two shared sequence fields |
| `sequence_logit` | 136 | history, ordinary-logit current/delta/prior-EMA-deviation blocks, and two shared sequence fields |
| `sequence_logit_j` | 256 | history, both matched temporal score-source blocks, and two shared sequence fields |

All variants must use the same outer and inner folds, hierarchical base weights,
training-class rebalance, estimator family, seed order, action-offset grids,
temperature grid, threshold grid, and deterministic numerical conventions.
Architecture or hyperparameter asymmetry between score sources is forbidden.

### Fixed J forecast pool with one shared logit action policy

The proposed primary concept is
`j_forecast_geometric_pool_logit_policy`. It has two forecast branches but only
one action policy. Let `p_L` be the strictly positive, row-normalized raw
probability from `sequence_logit`, and let `p_LJ` be the matched raw probability
from `sequence_logit_j`. The reference forecast is fixed as

```text
p_reference = p_L
```

and the candidate forecast is the fixed normalized geometric pool

```text
g_k = p_L,k^0.80 * p_LJ,k^0.20
p_candidate,k = g_k / sum_m(g_m)
```

for each class `k`. The exponents `0.80` and `0.20` are fixed design constants,
not selected parameters. Both inputs have already received the same strictly
positive model probability floor and row normalization. The geometric pool
therefore remains positive, but its displayed normalization is mandatory. No
additional probability floor is applied after pooling.

The action policy is selected once per fit context from `sequence_logit` alone.
Its inspect-fixed edit and check-or-finish offsets and resulting decision
distribution `d` are then shared exactly by candidate and reference. Public-J
probabilities cannot alter the action offsets, `d`, or `argmax(d)`. Candidate
and reference independently select only their scalar forecast temperature and
their predicted-class abstention thresholds from the corresponding forecast
branch. Outer held-out labels may not select model parameters, action offsets,
temperatures, or thresholds. Every hierarchical model-refit bootstrap draw
repeats the model fits and all permitted inner selections, while retaining the
fixed pool exponents and the single shared action-policy construction.

`sequence_j - sequence_logit` remains a matched replacement diagnostic.
`sequence_logit_j - sequence_logit` remains a feature-fusion diagnostic. The
primary J estimand is the candidate forecast from
`j_forecast_geometric_pool_logit_policy` minus the `sequence_logit` reference
forecast, conditional on their exact shared ordinary-logit action policy.

## Raw Probability, Forecast, Decision, And Abstention

V4 keeps forecast calibration mechanically separate from the action decision.
For each row it defines two branch-specific raw forecast probabilities
`p_reference` and `p_candidate`, two branch-specific calibrated forecasts
`q_reference` and `q_candidate`, and exactly one shared offset decision
distribution `d`. Neither a `q` nor `d` is fed back into the model or temporal
state.

### Shared-action, fixed-pool inner selection order

1. **Select the shared action rule once.** On inner out-of-repository
   `sequence_logit` raw probabilities `p_L`, enumerate the two additive decision
   offsets. The inspect offset is fixed at exactly `0`; only edit and
   check-or-finish offsets vary. For each setting, define

   ```text
   d = softmax(log(p_L) + [0, edit_offset, check_offset])
   ```

   A setting is eligible only if all four hierarchically weighted decision
   floors pass conjunctively: inspect recall `>= 0.75`, edit recall `>= 0.65`,
   check-or-finish recall `>= 0.75`, and balanced accuracy `>= 0.75`. Passing
   settings are ranked by maximum accuracy, then maximum balanced accuracy,
   then minimum decision-distribution NLL, then minimum frozen complexity.
   Decision NLL is only a tertiary action-rule tie break; it is not a calibrated
   forecast score. This selection is performed once in the fit context, not
   independently for candidate and reference.
2. **Form the fixed raw forecast branches.** Set `p_reference = p_L`. Form
   `p_candidate` by the exact normalized `0.80`/`0.20` geometric pool of `p_L`
   and `p_LJ` defined above. There is no exponent search, dynamic blend, or
   candidate-specific action construction.
3. **Calibrate each forecast branch independently.** For each branch
   `b in {reference, candidate}`, select its own scalar temperature `T_b` by
   minimum hierarchically weighted multiclass NLL on `p_b`, independently of
   the shared decision offsets, and define

   ```text
   q_b = softmax(log(p_b) / T_b)
   ```

4. **Apply the one shared action policy.** Apply the selected offsets to `p_L`
   only, as in step 1, and set `predicted_class = argmax(d)`. Temperature does
   not enter `d`; public-J probabilities do not enter `d`; and the offsets do
   not enter either `q`.
5. **Select separate predicted-class thresholds.** For each forecast branch
   `b`, confidence for the shared decision is the probability assigned by that
   branch's `q_b` to the class selected by the shared `d`:

   ```text
   decision_confidence_b = q_b[argmax(d)]
   accept_b = decision_confidence_b >= threshold_b[argmax(d)]
   ```

   Candidate and reference select separate threshold vectors on the same inner
   rows and shared predicted classes. Neither branch may use `max(q_b)` at a
   different class to accept the action chosen by `d`.

The proposed frozen action-setting complexity order after decision NLL is
smallest `abs(edit_offset) + abs(check_offset)`, then smallest absolute edit
offset, smallest absolute check offset, and finally the signed edit and check
offsets. If no action-rule setting passes all four decision floors, the frozen
fallback first maximizes the number of floors met, then minimizes total
recall-plus-balanced-accuracy shortfall, then applies the same accuracy,
balanced-accuracy, decision-NLL, and complexity order. The fallback still emits
outer diagnostic predictions and is recorded explicitly as a shared fallback.

Each branch's thresholds are chosen jointly from the same frozen Cartesian
grid. The inner passing rule maximizes hierarchically weighted coverage subject
to accepted accuracy at least `0.86`, coverage at least `0.70`, and a frozen
minimum accepted-row count for every true class, then prefers accepted accuracy
and the lowest deterministic threshold vector. The `0.86` inner selection floor
is a pre-freeze guardband; it does not change the outer absolute
accepted-accuracy gate of `0.85`. If no vector passes, the explicit threshold
fallback maximizes accepted accuracy subject to the coverage floor, then
coverage, then the lowest threshold vector. Outer labels may only score
settings already selected on the corresponding inner predictions.

The full-development shared action-rule selection and both full-development
branch-specific threshold selections must meet their respective inner floors
before model promotion. A full-development action or threshold fallback blocks
promotion. Outer-fold fallbacks remain mandatory, explicit diagnostic parts of
the frozen procedure: their rows are predicted and reported rather than
dropped, and their fallback status is preserved in bootstrap evidence.

Candidate and reference use the exact same stored action-policy settings and
row-level `d`; exact settings, `d`, and `argmax(d)` identity are integrity
requirements. They use the same weights, temperature grid and criterion,
threshold grid and objective, and tie rules, while selecting branch-specific
temperatures and thresholds from their respective `p` and `q`.

### Metric ownership

- Multiclass negative log likelihood and multiclass Brier score use each
  branch's own `q_b`.
- Standard forecast top-label ECE uses `argmax(q_b)`, `max(q_b)`, and the
  correctness of that forecast top label.
- Operational accuracy, balanced accuracy, and per-class recalls use the one
  shared `argmax(d)` and therefore must be exactly identical between candidate
  and reference.
- Branch-specific selected coverage and selected accepted accuracy use
  acceptance from `q_b[argmax(d)]` and correctness of the shared `argmax(d)`.
- Decision-confidence ECE and any binary confidence-versus-decision-correctness
  scores use `q_b[argmax(d)]` against correctness of `argmax(d)`. They remain
  mandatory diagnostics in point reports and bootstrap evidence, but they are
  not absolute or paired inferential gates.

## Nested Development Evaluation

The outer algorithm remains leave-one-repository-out. Within each outer
training set, every fit, one shared ordinary-logit decision-offset choice, both
branch-specific forecast temperatures, and both branch-specific
predicted-class threshold vectors are selected in the order above with
leave-one-repository-out inner predictions. The geometric-pool exponents remain
fixed at `0.80` and `0.20`. At least five inner repositories are required. A
row's outer repository label must not affect any fitted or selected quantity
for that row.

The point estimand remains V3's equal-repository, then equal-known-task within
repository, then equal-known-row within task weighting. Training restricts and
renormalizes those base weights to the current split and applies a split-local
exact three-class rebalance. Shared decision-rule selection, branch-specific
forecast-temperature selection, branch-specific threshold selection, and
evaluation use restricted base weights without class rebalance. Unknown-action
rows receive predictions but no label-dependent weight.

All variants and forecast branches use identical row order, folds, weights, and
random-seed schedule. Per-repository metrics, standard forecast top-label ECE,
diagnostic decision-confidence scores, and the inference, stability,
known-label, branch-specific acceptance, and class-support denominators remain
mandatory. The evidence validator must additionally prove exact shared action
settings, `d`, and predicted-class identity before scoring paired forecast
gates.

## Full-Refit Hierarchical Bootstrap

Confirmatory V4 evidence requires 1,000 complete hierarchical Bayesian
bootstrap draws. Each draw independently samples positive repository weights
and positive task-within-repository weights, retains every original row, and
recomputes the exact row weights. The same draw weights are used for all
variants and all paired differences.

Every draw reruns the full nested procedure, including:

- base-model fitting for all variants;
- inner out-of-repository prediction;
- one inspect-fixed `sequence_logit` decision-offset selection under all four
  decision floors, shared exactly by candidate and reference;
- exact raw reference construction and exact normalized fixed `0.80`/`0.20`
  geometric candidate pooling;
- independent scalar forecast-temperature selection on each branch's raw `p`;
- separate predicted-class threshold selection for each `q` using the shared
  `argmax(d)` and the `0.86` inner accepted-accuracy floor; and
- outer branch-specific `p` and `q`, shared `d`, metric computation, structural
  identity checks, and diagnostic decision-confidence scores.

Intervals that condition on one frozen set of out-of-fold predictions are not
sufficient. All 1,000 refit draws, deterministic per-draw seeds, exact resume
identity, and row-level probability/acceptance evidence are required. Any
missing, inconsistent, silently downgraded, or non-refit bootstrap evidence
fails every interval-bound gate closed.

## Proposed Support And Absolute Reliability Gates

These gates are proposals until the V4 protocol is frozen. The support floors
and every numeric absolute reliability gate from V3 remain unchanged. Absolute
gates apply to the proposed primary
`j_forecast_geometric_pool_logit_policy`; its action metrics come from the
shared ordinary-logit `d`, while its forecast and selective metrics come from
the candidate `q` and candidate thresholds. Changing the primary forecast
branch does not relax any V3 numeric floor. Decision-confidence ECE and related
binary confidence-versus-decision-correctness scores are reported as
diagnostics only and do not add gates.

### Support

| Requirement | Minimum |
|---|---:|
| Stable prediction rows | 1,500 |
| Known-action metric rows | 1,425 |
| Prediction tasks | 55 |
| Prediction repositories | 9 |
| Known-action tasks | 55 |
| Known-action repositories | 9 |
| Hierarchical known-action fraction | 0.95 |
| Known tasks per target class | 10 |
| Known repositories per target class | 6 |
| Numerical stability fraction | 0.90 |
| Stable feature-complete prediction fraction | 0.90 |

### Absolute reliability

| Metric | Bound | Requirement |
|---|---|---:|
| Known-action fraction | Point | `>= 0.95` |
| Balanced accuracy | Point | `>= 0.75` |
| Balanced accuracy | Bootstrap lower | `>= 0.70` |
| Inspect recall | Point | `>= 0.75` |
| Inspect recall | Bootstrap lower | `>= 0.70` |
| Edit recall | Point | `>= 0.65` |
| Edit recall | Bootstrap lower | `>= 0.55` |
| Check-or-finish recall | Point | `>= 0.75` |
| Check-or-finish recall | Bootstrap lower | `>= 0.70` |
| Selected accepted accuracy | Point | `>= 0.85` |
| Selected accepted accuracy | Bootstrap lower | `>= 0.80` |
| Selected coverage | Point | `>= 0.70` |
| Selected coverage | Bootstrap lower | `>= 0.65` |
| Standard forecast top-label ECE on `q` | Bootstrap upper | `<= 0.10` |
| Multiclass negative log likelihood | Point | `<= 0.60` |
| Multiclass negative log likelihood | Bootstrap upper | `<= 0.65` |

All support and absolute gates are conjunctive. Passing calibration or selective
accuracy cannot compensate for failed class recall, balanced accuracy, support,
or interval completeness.

## Proposed J-Specific Paired Gates

V4's J-specific claim is narrower than V3's matched-replacement claim:

> Adding a fixed 20% public-J hybrid contribution through the normalized
> geometric forecast pool improves probabilistic forecasts relative to the
> ordinary-logit forecast, conditional on an exactly shared ordinary-logit
> action policy.

The candidate is the forecast head of
`j_forecast_geometric_pool_logit_policy`; the reference forecast is
`sequence_logit`. Differences are always `candidate - reference` on identical
outer rows, weights, folds, and bootstrap draws. Each branch uses its own
inner-selected scalar temperature. Neither the fixed pool nor the J-specific
claim includes an action-policy difference.

Proper-score superiority is conjunctive:

| Metric difference | Bound | Requirement |
|---|---|---:|
| Multiclass negative log likelihood | Bootstrap upper | `< 0.0` |
| Multiclass Brier score | Bootstrap upper | `< 0.0` |

Both proper-score differences are computed from each procedure's independently
temperature-calibrated forecast `q`.

There are no accuracy or balanced-accuracy noninferiority intervals in the
J-specific inferential gate. Instead, exact structural identity is a fail-closed
integrity precondition: candidate and reference must bind the same
`sequence_logit` action-policy selection record, inspect/edit/check offset
settings, row order, row-level `d`, and `argmax(d)` for every outer prediction
and every bootstrap draw. Any mismatch invalidates the paired comparison rather
than becoming an estimated decision difference. Branch-specific temperatures,
`q` values, and threshold vectors are expected to differ and are not included
in this identity requirement.

Both proper-score gates must pass together with every support, integrity, and
absolute gate. The unchanged absolute action gates protect the one shared
action policy. Decision-confidence ECE and other binary
confidence-versus-shared-decision-correctness scores are descriptive only; they
cannot establish or defeat the J-specific claim. Direct
`sequence_j - sequence_logit` and `sequence_logit_j - sequence_logit` results
remain descriptive and cannot substitute for strict paired proper-score
superiority under the fixed pool and shared action policy.

## Fail-Closed Decision Mapping

- **All support, integrity, absolute, and paired gates pass:** V4 may be reported
  as a reliable fresh-development result and may justify drafting a separate
  validation protocol. Validation remains closed until that separate lifecycle
  is frozen.
- **Support, capture, provenance, or bootstrap-integrity failure:** the V4
  development result is inconclusive. It may not be rescued by pooling data or
  silently changing denominators.
- **Any absolute gate fails:** the V4 operational readout is not reliable on the
  fresh development cohort.
- **The structural shared-action identity check fails:** the paired J forecast
  comparison is invalid, regardless of point metrics or intervals.
- **Absolute gates pass but a J-specific proper-score gate fails:** the shared
  ordinary-logit action readout may still be useful, but positive incremental-J
  forecast value is not established.

No gate, target, feature, model family, geometric-pool exponent, decision-offset
grid, forecast-temperature grid, threshold rule, or decision mapping may be
changed after fresh-cohort selection to rescue the result.

## Required New V4 Namespace

The eventual implementation should add new paths rather than alter V3, for
example:

- `configs/swe_task_state_interpreter_v4.json`;
- `configs/swe_task_state_v4_action_probes.json` only if the unchanged target
  requires a separately pinned copy;
- V4 development cohort, campaign, and image declarations under new names;
- `requirements-v4-state-interpreter.txt`;
- `scripts/analyze_swe_task_state_v4.py`;
- V4 cohort checker, materializer, replay pipeline, wrappers, and tests; and
- `.cache/swe_state_interpreter_v4_development/` for all generated outputs.

V4 may call an unchanged V3-frozen helper only as a hash-pinned dependency. If
the helper must change, V4 must copy or replace it under a new path. In
particular, V4 must not modify shared package locks or runtime scripts included
in the V3 source closure.

## Unresolved Before Protocol Freeze

This document intentionally does not resolve the following. None may remain
open when the confirmatory V4 protocol and cohort are frozen:

1. **Base readout family:** the design screen retains the exact V3 multiseed
   ExtraTrees procedure over causal temporal features. A future protocol must
   pin its exact source, requirements, estimator schedule, and numerical
   execution identity; no further architecture search is authorized on V3 N60.
2. **Feature serialization:** freeze exact temporal feature names, widths,
   float dtype, raw 96-wide subtraction order, separate compact reductions,
   first-boundary encoding, unstable-boundary behavior, and state reset/update
   tests.
3. **Shared action-rule search:** freeze the edit-offset and
   check-or-finish-offset grids for the one `sequence_logit` action policy.
   Inspect offset must remain exactly zero, and the four decision floors,
   ranking, complexity order, fallback, single-selection semantics, and exact
   candidate/reference settings-and-`d` identity checks must be encoded.
4. **Forecast construction and calibration:** freeze the exact normalized
   geometric-pool formula and exponents `0.80` and `0.20`, scalar temperature
   grid, probability validation tolerance, branch-specific weighted-NLL
   selection, and tie break. Temperature must not enter `d`, public-J
   probabilities must not enter `d`, and offsets must not enter either `q`.
5. **Class thresholds:** freeze the Cartesian threshold grid, the `0.86` inner
   accepted-accuracy floor, unchanged `0.85` outer gate, minimum accepted rows
   per true class, selection fallback, separate branch vectors, confidence
   source `q_b[argmax(d)]`, and deterministic vector tie order.
6. **Cohort design:** the official 1,794-identity complement is authenticated,
   but a disjoint 5-Flask/10-Seaborn transport pilot plus untouched cap-20
   confirmation needs five more Flask and ten more Seaborn tasks. SWE-rebench
   can cover Seaborn at the identity/build-config level; no audited
   same-stratum source covers the Flask shortfall. Freeze a separate
   content-blind source-curation, environment-qualification, exclusion, seed,
   ordering, and campaign-partition contract before any task is selected or
   environment is built.
7. **Power:** both the historical all-286 screen and the new twelve-repository
   cap-20 screen are NO-GO. The latter has p90-adverse joint power `0.1498`
   with adjusted exact lower bound `0.141882`; an independent exchangeable
   sensitivity also falls below target. Only a separately frozen,
   outcome-informed transport pilot may narrow this envelope. If its
   confidence-set gate passes, repeat prospective power on the still-untouched
   confirmation allocation and then verify full nested/refit sensitivity
   before freeze. Power analysis may change cohort design before, but never
   after, fresh confirmation selection.
8. **Implementation closure:** decide which unchanged V3 helpers are imported
   read-only and which are forked. Pin every transitive source and runtime byte
   without modifying V3's closure.
9. **Positive controls:** freeze any SWE-specific, lens-independent positive
   control and its role as a diagnostic or gate. It must not alter the
   same-request target or expose future information to V4 features.
10. **Publication surface:** define a compact decision record that binds large
    development artifacts without publishing raw trajectory or replay payloads.

## Freeze Checklist

A future commit may change this pre-freeze design into an executable V4
protocol only after it can answer yes to every item below:

- Is the V3 source closure byte-identical to its frozen commit?
- Does the V4 protocol bind the V3 stop decision and all immutable V3 inputs
  listed above?
- Are the target and label/censor rules exactly specified and same-request
  prospective?
- Are all causal state updates executable without the current or later action?
- Are base models, the one shared ordinary-logit offset selection, fixed
  `0.80`/`0.20` geometric forecast pool, branch-specific temperatures,
  branch-specific thresholds with the `0.86` inner guardband, exact
  `p`/`q`/shared-`d` construction, grids, tie rules, folds, weights, seeds, and
  numerical conventions fixed?
- Does the bootstrap refit and reselect the entire nested procedure in all 1,000
  draws?
- Are every unchanged V3 absolute gate, the standard forecast ECE gate, both
  strict paired proper-score gates, exact shared-action structural checks, and
  diagnostic-only treatment of decision-confidence scores encoded in the
  checker rather than left as prose?
- Is the new cohort selected identity-only and proven disjoint before any
  generation or score access, with qualified Flask/Seaborn transport supply,
  a disjoint pilot, and a post-pilot confirmation power rescreen resolved?
- Do all V4 outputs resolve only beneath a new no-clobber development namespace?
- Does every failure mode preserve the closed validation boundary?

Until those conditions are met and committed, V4 remains a design proposal and
no confirmatory evidence may be claimed.
