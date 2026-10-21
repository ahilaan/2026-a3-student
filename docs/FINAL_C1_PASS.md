# Final optimization and compliance pass - 21 October 2026

Final measured results and verification.

## Decision

Adopt the bounded raw-versus-query-percentile C1 selector; preserve all existing main/neural artifacts and inference.

Initial 39-condition held-out mean gain +0.005710 with positive family means; disjoint MSLR confirmation gain +0.009717. A fixed query-percentile model has slightly higher initial mean (+0.006107) but worse initial worst regression. Raw/rank selection retains a known baseline fallback and conservative inner tie preference. Weak/noisy generated labels regress by 0.008150 on average, so improvement is not universal.

Hidden C1 performance is unknown. No leaderboard submission was made.

## Compliance and current-model reproduction

`scripts/train_model.py --config current-main` orchestrates the existing full
trainer and native-tree blender. The legacy `basic` and `submitted` modes are
unchanged in behavior; `submitted` is explicitly labeled as the historical
200-tree configuration. Outputs under `submission/` are rejected by the new
current-main mode. Fresh fits and retained-component assembly are distinct.

```bash
python scripts/train_model.py --config current-main --train data/train --dev data/dev --use-dev --work-dir runs/current_reproduction --out runs/current_reproduction/model.txt
```

The actual fresh run used `runs/final_c1_pass/fresh` instead of
`runs/current_reproduction`. Only released TRAIN and DEV were read. Source
hashes, raw cached arrays, labels, query IDs, within-query transforms and the
TRAIN-then-DEV combined matrix were audited before fitting. Evidence:
`runs/final_c1_pass/fresh/input_audit.json`.

Architecture: two native LambdaRank components, 2,000 trees/127 leaves/seed 0
and 2,400 trees/63 leaves/seed 23. The blend scales leaf outputs by .65/.35
and concatenates trees. Both fit 958,671 original TRAIN+DEV rows/8,000
queries plus one augmented copy (1,917,342 fit rows/16,000 query blocks).
Their 1,200-tree save/reload boundaries are preserved; continuation resets
training state as in the actual workflow. Hyperparameters: learning rate .05,
minimum leaf 100, L2 10, feature/row fractions .8, bagging every round,
extra randomized splits, 63 bins, deterministic row-wise training, four
threads and all explicit RNG seeds equal to the component seed. No stopping
or gradients use held-out data. Original plus average-tie query percentiles
give 272 features. Augmentation RNG is `76420 + seed`: Gaussian noise .02
times query feature range, per-query log-normal column scaling sigma .5,
and 14-40 dropped columns, followed by the same inference transform.

Retained-component assembly: byte equality **True**;
prediction equality **True**; ranking equality
**True** on 20 TRAIN queries.
Fresh staged refit: byte equality **True**; prediction equality **True**; ranking equality **True**; maximum score difference 0.0; 4400 trees/272 features; 5984.71 seconds.
Exact commands and hashes are in the respective `verification.json` files.
These results apply to local LightGBM 4.7.0; Ubuntu/Python 3.12
cross-platform byte identity was not tested. Matching hyperparameters alone
are not treated as a reproduction result.

README and SUBMISSION_NOTES now state current splits, augmentation, component
settings, tuning exposure, the correct standard entry point and C1 behavior.
C1_REPRODUCIBILITY and BLEND_SELECTION are explicitly historical. The old
public-data study's permission claim is labeled unverified and excluded from
this pass. Known historical TEST-label and public LTR-label usage is preserved
in COMPLIANCE and the original research records. A new refit cannot erase that
selection exposure or establish that the historical process complied. AI and
code-provenance disclosures remain. The existing assignment PDF/report were
not rewritten to conceal history.

## Baseline protocol and results

There are 39 condition/fold pairs: three provided toy sets (one outer split
each), three generated datasets (three outer folds each), and nine MSLR TRAIN
size/feature conditions (three outer folds each). For toy sets, provided TRAIN
is divided into gradient/stopping queries; provided validation is outer-only.
Generated datasets use widths 17/90/300, query counts 240/360/240, densities
1/.1/.02 and labels with 2/3/5 levels, seeds 171/172/173. They include singleton
and zero-label queries. MSLR pools contain 300/900/2,250 complete released TRAIN
queries sampled with seed 193, with leading 40/90/136-column subsets. Outer
grouped folds use seed 41; one fifth of available queries is inner validation
using seed 1200+fold. All models use seed 0. No row-random splitting occurs.
Nested pools and feature variants share queries; these are correlated
conditions, not 39 independent replications or a significance estimate.

Original baseline: LambdaRank, 15 leaves, minimum leaf 50, L2 1, rate .05,
feature/row fractions .8, 255 bins, truncation 30, maximum 800 rounds and
patience 50. All 39 baseline models saved/reloaded with finite predictions and
matched the standard ranking interface. Baseline fitting time total:
244.03 seconds.

| Condition | Folds | Train nDCG | Inner nDCG | Outer nDCG | Outer SD | Mean procedure s |
| --- | --- | --- | --- | --- | --- | --- |
| a_40_features_labels0-4 | 1 | 0.972131 | 0.941850 | 0.936459 | 0.000000 | 0.49 |
| b_90_features_labels0-4 | 1 | 0.945304 | 0.959286 | 0.927711 | 0.000000 | 0.24 |
| c_250_features_labels0-2 | 1 | 0.991603 | 0.949986 | 0.945524 | 0.000000 | 1.25 |
| generated_17 | 3 | 0.930845 | 0.923415 | 0.910524 | 0.032265 | 0.42 |
| generated_90 | 3 | 0.756366 | 0.689964 | 0.689045 | 0.017884 | 0.43 |
| generated_300 | 3 | 0.457183 | 0.462072 | 0.454330 | 0.016262 | 0.21 |
| MSLR_train_q300_d40 | 3 | 0.566448 | 0.397130 | 0.342960 | 0.010073 | 1.62 |
| MSLR_train_q300_d90 | 3 | 0.520144 | 0.391330 | 0.329978 | 0.014645 | 1.08 |
| MSLR_train_q300_d136 | 3 | 0.667176 | 0.464543 | 0.426553 | 0.008578 | 3.71 |
| MSLR_train_q900_d40 | 3 | 0.488286 | 0.381140 | 0.371110 | 0.007032 | 3.80 |
| MSLR_train_q900_d90 | 3 | 0.513879 | 0.388004 | 0.378902 | 0.004008 | 9.47 |
| MSLR_train_q900_d136 | 3 | 0.562496 | 0.451966 | 0.447069 | 0.002471 | 10.20 |
| MSLR_train_q2250_d40 | 3 | 0.447300 | 0.380871 | 0.379497 | 0.003138 | 14.49 |
| MSLR_train_q2250_d90 | 3 | 0.450651 | 0.382707 | 0.381182 | 0.005128 | 15.77 |
| MSLR_train_q2250_d136 | 3 | 0.540646 | 0.461135 | 0.457651 | 0.009846 | 30.97 |

Three full-input synthetic staff-interface runs are separately recorded in
`baseline_api.json`; their supplied validation scores are mechanical checks,
not independent outer scores. Per-fit best iterations, saved trees, model
bytes, matrix bytes, process-memory high-water marks, train/inner/outer scores
and gaps are recorded in `study/fits.json` and `experiments.csv`.

## Staged experiments

Candidates were registered in `study/protocol.json` before fitting. Screening
uses nine representatives: all toy sets, first fold of each generated set,
and first full-width fold at each MSLR size. Practical finalists (baseline,
31 leaves, regularized 31 leaves, fast15) were predetermined and then evaluated
on every condition. The query-transform extension was registered based only
on screening inner scores, before reviewing outer results; its fixed gate is
recorded in `study/transform_extension.json`. Every meaningful override is in
the protocol and each fit's parameter dictionary. No Cartesian search was run.

| Candidate | Fits | Outer mean | Mean paired delta | Worst delta | Mean procedure s |
| --- | --- | --- | --- | --- | --- |
| baseline | 39 | 0.500413 | +0.000000 | +0.000000 | 7.14 |
| leaves7 | 9 | 0.695891 | -0.000845 | -0.032346 | 4.36 |
| leaves31 | 39 | 0.498996 | -0.001417 | -0.014442 | 9.02 |
| leaves63 | 9 | 0.693448 | -0.003288 | -0.017167 | 8.56 |
| regular31 | 39 | 0.495462 | -0.004951 | -0.022694 | 6.88 |
| regular63 | 9 | 0.691806 | -0.004930 | -0.033295 | 4.74 |
| fast15 | 39 | 0.499338 | -0.001074 | -0.017975 | 7.09 |
| trunc10 | 9 | 0.692365 | -0.004371 | -0.030705 | 4.03 |
| trunc13 | 9 | 0.695845 | -0.000891 | -0.008820 | 3.89 |
| rate10 | 9 | 0.694315 | -0.002422 | -0.013101 | 3.58 |
| slow31 | 9 | 0.697965 | +0.001229 | -0.003842 | 7.31 |
| feature07 | 9 | 0.693611 | -0.003125 | -0.018465 | 4.58 |
| feature10 | 9 | 0.692224 | -0.004512 | -0.043442 | 6.78 |
| no_bagging | 9 | 0.696305 | -0.000431 | -0.012758 | 8.95 |
| l1gain | 9 | 0.694759 | -0.001978 | -0.026292 | 8.36 |
| bin127col | 9 | 0.695745 | -0.000991 | -0.010951 | 4.76 |
| xendcg | 9 | 0.694598 | -0.002138 | -0.018495 | 4.40 |
| patience20 | 9 | 0.696470 | -0.000266 | -0.003545 | 4.14 |
| patience35 | 9 | 0.696383 | -0.000354 | -0.001704 | 4.68 |
| patience75 | 9 | 0.695705 | -0.001032 | -0.009285 | 7.06 |
| query_ranks | 39 | 0.506520 | +0.006107 | -0.013138 | 15.51 |
| sparse_csr | 9 | 0.696736 | +0.000000 | +0.000000 | 5.97 |

Candidate means with different fit counts use different subsets and are not
directly comparable; paired deltas use each candidate's matching baseline.
Screening covers 7/15/31/63 leaves, minimum leaves 20/50/100/200, L2 0/1/5/10,
L1/minimum gain, feature fractions .7/.8/1, row sampling, rates .03/.05/.1,
up to 1,200 rounds, patience 20/35/50/75, 63/127/255 bins, LambdaRank
truncations 10/13/20/default30 and diagnostic rank_xendcg. Raw dense and CSR
handling and the existing query-percentile schema are tested. Data statistics
include queries, rows, dimensions, density, label distribution, multigrade
queries, documents/query and constant columns. The simple adaptive rule uses
31 leaves for at least 400 gradient queries, otherwise baseline. No complex
metadata rule is fitted to outer scores.

## Complete strategy comparison

The fixed alternative is chosen using inner scores only. Validation search
compares baseline/31/regular31 and retains the simpler earlier candidate unless
inner nDCG improves by more than .001. The hybrid compares baseline with the
size-rule choice. Raw/rank selection uses the same inner-only tie rule. Strategy
runtime sums the measured constituent procedures; it excludes source loading
and is not a fresh-process API benchmark. The actual end-to-end procedure,
two-candidate selection and 200-tree race/continuation are separately measured
in `profiles/results.json`, including parsing. All outer outcomes measure the
entire corresponding inner-driven procedure, not the best outer candidate.

| Strategy | Outer nDCG | SD | Mean procedure s | Worst delta | Loadable |
| --- | --- | --- | --- | --- | --- |
| Original recipe | 0.500413 | 0.200516 | 7.14 | +0.000000 | 39/39 |
| Improved fixed recipe | 0.500413 | 0.200516 | 7.14 | +0.000000 | 39/39 |
| Adaptive recipe | 0.499932 | 0.200936 | 8.43 | -0.009290 | 39/39 |
| Validation search | 0.499720 | 0.200582 | 23.04 | -0.010273 | 39/39 |
| Hybrid policy | 0.500717 | 0.200463 | 14.95 | -0.003883 | 39/39 |
| Selected final recipe | 0.506122 | 0.198175 | 22.65 | -0.010728 | 39/39 |
| Query-relative fixed | 0.506520 | 0.197295 | 15.51 | -0.013138 | 39/39 |
| Raw/rank validation selection | 0.506122 | 0.198175 | 22.65 | -0.010728 | 39/39 |

Family-specific deltas and selection decisions are in `study/summary.json`
and `decision.json`. Model-selection/stopping never consume outer labels.
Small correlated-fold differences do not establish statistical significance.
No ensemble or dataset-specific clipping/removal schema is deployed without
evidence. LightGBM already filters unsplittable constant columns; keeping their
schema positions preserves inference. The NaN/finite-overflow behavior is
tested using generated input. Explicit infinities/malformed labels fail with
context. Transformations requiring unavailable sidecars are avoided.

### Actual API allocation comparison

| Procedure | Outer conditions | Outer mean | Paired delta | Worst delta | Mean API s |
| --- | --- | --- | --- | --- | --- |
| baseline | 9 | 0.696736 | +0.000000 | +0.000000 | 12.03 |
| two | 9 | 0.696203 | -0.000534 | -0.008321 | 18.12 |
| search | 9 | 0.694717 | -0.002019 | -0.008321 | 25.90 |
| successive | 9 | 0.696013 | -0.000723 | -0.004320 | 20.25 |
| raw_rank | 9 | 0.698231 | +0.001495 | -0.010728 | 19.27 |

These nine representatives are the first fold of each generated condition,
all three toy sets, and the full raw width at each MSLR pool size. They are a
subset of the 39 primary conditions, not additional independent replications.
The two-model race gives each candidate at most 200 trees and continues only
the inner winner. It does not reliably improve outer accuracy. Baseline and
raw/rank chosen artifacts agree byte-for-byte with the in-memory study in all
nine cases. The deployed production implementation was rerun on all nine and
also matches the evaluated selector's chosen model bytes. Normal-case behavior
is therefore verified through the actual parser, training and serialization.
Emergency-budget allocation is tested separately; production gives the raw
baseline its full shared deadline before considering optional features.

### Confirmation before deployment

| Confirmation condition | Outer folds | Mean selector delta | Worst delta | Rank choices |
| --- | --- | --- | --- | --- |
| MSLR_disjoint1800_d136 | 3 | +0.009717 | -0.001910 | 3 |
| heavy_tail | 3 | +0.000000 | +0.000000 | 0 |
| query_shift_scale | 3 | +0.133575 | +0.124125 | 3 |
| weak_sparse | 3 | -0.008150 | -0.015667 | 3 |

These 12 additional outer folds were registered based only on inner screening,
before any outer summaries were reviewed. The 1,800-query MSLR pool is disjoint
from the initial 2,250-query pool: shuffled TRAIN IDs [2250:4050], seed 193.
Outer folds use seed 84. Generated cases use 360 queries each, seeds 471/472/473:
90 columns with label-preserving per-query positive scales and shifts; 40
columns with monotone exponential scales; and 300 sparse columns with random
three-grade labels. The last case intentionally has no reliable feature signal.
The disjoint MSLR mean change is +0.009717. The large scale-shift gain is a
constructed stress condition and is not a prediction of hidden C1 improvement.
Weak/noisy labels regress by 0.008150 on average; worst confirmation change
is -0.015667. No weak-label heuristic was added after viewing those results.
All predeclared initial and confirmation gates passed (decision.json).

## Selected C1 implementation

`submission/recipe.py` retains the original shallow LambdaRank parameters.
It loads TRAIN/validation once, saves the completed raw baseline, and scores
its validation predictions using the staff stable-tie/zero-relevance convention.
Only if sufficient shared time remains does it append complete-query average
percentiles and fit the same ranker again. A validation improvement above .001
is required to atomically replace the raw artifact. No validation gradients,
train+validation refit, learned preprocessing statistics, external files,
dataset identities, hard-coded feature positions or ensembles are used.
Feature names encode the raw/query-percentile schema, already supported by the
unchanged prepare/rerank interface. Only one selected native model is loaded.

The raw baseline retains the entire 960-second soft deadline. Optional work
requires max(30 seconds, three times measured baseline cost) remaining. Query
feature generation checks the shared deadline between original queries while
reserving max(30 seconds, twice baseline cost) for fitting/saving. Abandoning
optional feature generation leaves the raw model intact. Candidate boosting
uses the same shared deadline and public best-iteration stopping mechanism.
The .001 tie threshold and two-candidate policy were frozen before outer
evaluation; they were not tuned against confirmation outcomes.

## Runtime, memory and reliability

Original fixed baseline, before the representation selector:

| TRAIN rows | D | Background density | Recipe s | Peak working set MiB | Trees | Loads |
| --- | --- | --- | --- | --- | --- | --- |
| 500000 | 300 | 1.0 | 554.96 | 1538.8 | 187 | True |
| 500000 | 300 | 0.1 | 109.05 | 1007.3 | 139 | True |

Selected final adaptive policy:

| TRAIN rows | D | Density | Total recipe s | Peak MiB | Chosen D | Candidate fits | Loads |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 500000 | 300 | 1.0 | 669.81 | 3031.4 | 300 | 2 | True |
| 500000 | 300 | 0.1 | 400.50 | 2939.5 | 300 | not instrumented | True |

Stress inputs have five dense signal columns, generated noisy relevance,
100 documents/query, 50,000 separate validation rows, 300 columns and comments.
Both parsing passes, training and saving are timed; generation is excluded.
Peak memory is process working set, not an enforced 8 GB cap. Measurements
were made on Windows, with other local work running; staff machine speed,
memory accounting and distributions may differ. Edge cases additionally cover
5,000-document groups, 5,000 tiny queries, zeros/constants and validation-only
indices, with group sums verified. NaN, malformed/non-finite values,
interleaved IDs, ties, singleton queries and forced-expiry behavior are tested.

The production soft timer starts before import/parsing and targets 960 seconds,
reserving max(15 seconds, twice recent iteration cost). Public LightGBM
EarlyStopException retains the best completed validation iteration. A valid
first-tree fallback is written before further iterations; atomic replacement
keeps it until final serialization succeeds. The raw candidate remains byte-equal
to the original when the timer is inactive. This cannot preempt
parsing, dataset construction or a single native iteration, and does not
guarantee recovery from a killed/failed staff process. Emergency time-based
tree counts depend on machine load; normal fixed-seed runs are repeatable.

The original recipe was also verified to fail on a 10,001-document query
because of LightGBM's native 10,000-document limit. Only oversized queries
are split into contiguous training blocks of at most 10,000 documents, never
mixing original queries. In this case stopping uses a custom metric on whole
original validation queries, including stable ties and zero-relevance queries.
Cross-block training pairs are omitted, a known approximation. Normal query
groups and normal validation behavior remain unchanged. The native failure,
group-boundary invariants, staff metric agreement and successful 10,001-row
training/validation are recorded in native_group_limit.json and targeted tests.

For two-document queries, optional training percentiles use an exact vectorized
equivalent of the existing inference transform. The 5,000-tiny-query workload
fell from 43.41 to 2.88 seconds with identical model bytes. Float32/float64,
ties, missing values and non-finite ordering match the existing transform;
three generated production cases also retain identical model bytes.
Evidence: `small_query_optimization.json`. A minimal staff-module isolation
check confirms the recipe has no dependency on the offline experiment scripts.

## Regression, tests and packaging

All protected artifact/source hashes are unchanged except the intentionally
modified recipe. On 50 authorized TRAIN queries
(4954 documents), main scores/order/IDs and
fusion outputs match exactly. Neural scores match through the staff loader.
Final repeated-seed C1 artifacts match byte-for-byte on all three provided
transfer sets. Two retain the original raw model; the 250-feature set selects
the query-percentile model. These supplied-validation scores are mechanical
selection checks, not outer generalization evidence:

| Provided transfer set | Original vali | Final vali | Final repeat byte equal |
| --- | --- | --- | --- |
| a_40_features_labels0-4 | 0.943638070 | 0.943638070 | True |
| b_90_features_labels0-4 | 0.942557838 | 0.942557838 | True |
| c_250_features_labels0-2 | 0.952455956 | 0.955299679 | True |

Evidence: `regressions.json`, `final_recipe_regression.json`.

Real DEV preflight: main nDCG 0.624297, pointwise
0.424748, pairwise 0.490646,
BM25 0.288487. DEV supplies main/pairwise gradients and
pointwise selection: these are regression/fitting measurements, not new tests.
Largest-query inference 0.584 seconds;
prepare 0.622 seconds; preflight passes
True.

Test logs are in `runs/final_c1_pass/`: actual-artifact suite **126 passed,
1 deselected** (the excluded end-to-end test expects a toy-trained main model),
isolated toy-trained full suite **127 passed**, plus focused reproducibility,
fallback, selector and module-isolation tests.
The isolated copy protects production weights. Packaging verification is
recorded in `packaging.json`; the reviewed ZIP is separate from the original
archive, contains all seven required entries, excludes raw datasets/runs,
and must remain below 50 MB. No automatic submission occurs.
