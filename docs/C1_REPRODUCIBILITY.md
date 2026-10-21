# Historical: C1 reliability and submission-53 model reproduction

**Historical record, superseded for current artifacts.** All hashes, scores,
"current"/"submitted" model references and reproduction results below refer
to submission 53 (18 October), a 200-tree artifact. They do not describe the
present 4,400-tree blend or its neural artifacts. Current configuration is in
`CURRENT_MODEL.json`; current verification is in `FINAL_C1_PASS.md`.
The public-data research cited below is historical and has no verified
official permission amendment; see `COMPLIANCE.md`. It is excluded from this pass.

This records submission 53 and its original checks. The later public-data
C1 study is in `runs/c1_research/REPORT.md`: it retains these ranking
parameters and adds finite-overflow saturation only for transfer training.
The exact main-model reproduction below still applies. The existing ZIP
and portal submission have not been updated with that local parser change.

Ahilaan Saxena, 2023CS10076. Verified 18 October 2026.

## Scope and preserved baseline

The initial Git working tree was clean. Before changes, the existing ZIP,
all submission Python sources and all three trained artifacts were copied to
`runs/c1_repro/baseline/`, with SHA-256 hashes. These backups, all datasets,
portal session information, experimental models and run outputs are excluded
from the submission ZIP. The prior archive remains available in that backup.

The final `submission/model.txt`, `pointwise_model.pt`, `pairwise_model.pt`,
`ranking.py`, `fusion.py`, `pointwise_pairwise.py` and `query_features.py` are
byte-identical to the preserved versions. No fusion, neural, C2 artifact or
inference optimization was performed. No held-out labels or hidden benchmark
identities were sought. All recipe selection experiments used released TRAIN
or generated data; public DEV was not a recipe tuning target.

## C1 experimental design

`python scripts/benchmark_c1.py` performs a resumable limited comparison, with
full parameters and per-fit results in `runs/c1_repro/benchmark.json` and
`benchmark_summary.json`. There were 126 fits, all successfully saved and
reloaded with finite predictions. Total measured fitting time was 235.84 s;
this excludes data loading, scoring, saving and other experiment overhead.

Three query pools of 300, 900 and 2,250 queries were sampled from authorized
MSLR TRAIN using seed 19. Each has three outer query folds (seed 0). Of the
remaining queries in each fold, one fifth is used only for early stopping
(seed 200 + fold); the outer queries do not select the stopping iteration.
Actual gradient-training sizes were 19,118-20,018, 58,195-59,440 and
142,580-145,542 rows, respectively. Outer folds contain 100, 300 and 750
complete queries. The nested pools share queries, so the nine fits are not
nine independent replications. This is a limited comparison, not a significance
claim or an estimate of either undisclosed benchmark's score.

All candidates use LambdaRank, nDCG@10, four threads, deterministic row-wise
training, seed 0 for all explicit RNG seeds, row subsampling 0.8 every round,
at most 800 rounds and validation patience 50. Differences from baseline:

- Baseline: 15 leaves, minimum leaf 50, L2=1, rate 0.05, feature fraction 0.8,
  default 255 bins and truncation 30.
- `leaves31`: 31 leaves; other settings unchanged.
- `regular31`: 31 leaves, minimum leaf 100, L2=5, 63 bins, truncation 13.
- `fast15`: 63 bins and truncation 20; other baseline settings unchanged.
- `regular63`: 63 leaves, minimum leaf 200, L2=10, 63 bins, truncation 13.
- `slow15`: minimum leaf 100, L2=10, feature fraction 0.9, rate 0.03,
  63 bins, truncation 13.

The following entries are outer-fold mean nDCG@10 +/- population standard
deviation. Runtime is mean fitting wall time across the nine MSLR fits,
excluding parsing, prediction and saving. Times are local workstation
measurements, not an evaluation-machine speed guarantee.

| Configuration | 300-query pool | 900-query pool | 2,250-query pool | Mean fit seconds | Reloadable |
|---|---|---|---|---|---|
| Existing C1 recipe | 0.431589 +/- 0.022383 | 0.450340 +/- 0.017569 | 0.454873 +/- 0.007405 | 4.26 | Yes |
| leaves31 | 0.424343 +/- 0.015553 | 0.454569 +/- 0.013484 | 0.459397 +/- 0.011329 | 5.14 | Yes |
| regular31 | 0.423072 +/- 0.023803 | 0.449509 +/- 0.014871 | 0.452548 +/- 0.011416 | 3.45 | Yes |
| fast15 | 0.435122 +/- 0.021855 | 0.447951 +/- 0.021369 | 0.456201 +/- 0.010488 | 3.48 | Yes |
| regular63 | 0.422911 +/- 0.017845 | 0.452444 +/- 0.014444 | 0.449901 +/- 0.011672 | 4.13 | Yes |
| slow15 | 0.422717 +/- 0.015997 | 0.439046 +/- 0.011924 | 0.456029 +/- 0.011605 | 4.95 | Yes |
| Selected: baseline parameters + reliability safeguards | Same baseline predictions when timer is inactive | Same | Same | Timer overhead not separately isolated | Yes |

Additional experiments used the three provided toy transfer sets and generated
17-, 90- and 300-column datasets. Generated datasets vary query sizes (including
singletons), query counts (180/300/120), number of label levels (2/3/5), uniform
queries, and nonzero density (1/0.1/0.02). Generated labels depend on feature
signal plus noise, never on any real hidden labels. Generated scores are means
of three outer query folds with separate stopping queries. For provided toy
sets, the original TRAIN supplies gradient/stopping partitions and the provided
validation set is the outer report; these rows have one outer split each.

| Dataset | Baseline | leaves31 | regular31 | fast15 | regular63 | slow15 |
|---|---|---|---|---|---|---|
| Provided 40 columns | 0.940395 | 0.930845 | 0.936905 | 0.937408 | 0.937831 | 0.929537 |
| Provided 90 columns | 0.935232 | 0.937406 | 0.941332 | 0.927337 | 0.942970 | 0.940580 |
| Provided 250 columns | 0.950923 | 0.950295 | 0.944509 | 0.944387 | 0.949674 | 0.940287 |
| Generated 17 columns, dense | 0.898841 | 0.902837 | 0.892042 | 0.896829 | 0.898334 | 0.889672 |
| Generated 90 columns, 10% nonzero | 0.683750 | 0.683761 | 0.678874 | 0.683142 | 0.673358 | 0.677197 |
| Generated 300 columns, 2% nonzero | 0.470360 | 0.470360 | 0.470360 | 0.470360 | 0.470360 | 0.470360 |

The last generated case is too sparse/small to produce useful splits at the
tested leaf sizes: all candidates save one constant tree. It checks mechanics
and exposes the limit of these settings; it provides no evidence of accuracy
improvement. Matrix byte counts, training nDCG, best iterations and per-fold
scores are recorded in the JSON. Matrix bytes are not peak process memory.

### Selection and adaptation

Retain the existing accuracy parameters. Across the three MSLR pool means,
baseline averages 0.445601, leaves31 0.446103 and fast15 0.446425. Those small
differences accompany reversals across sizes/folds and the synthetic cases.
Larger trees hurt the smallest real pool; stronger regularization did not
consistently improve the larger pools. Choosing tree complexity by query count
is therefore plausible but insufficiently supported here: a threshold fitted
to these three nested pools would need independent confirmation. No arbitrary
adaptive leaf-size rule or in-recipe hyperparameter search was adopted.

The implemented improvements address failure and timeout risk:

- A monotonic timer starts before LightGBM import and input parsing. After
  each completed boosting round, a callback monitors a 960-second total target,
  reserving at least 15 seconds or twice the maximum of the last five observed
  iteration costs. LightGBM's public `EarlyStopException` selects the best
  completed validation iteration and the normal save path writes it.
- TRAIN and validation retain common width, implicit zeros, stable query
  regrouping, trailing-comment support and integer labels in 0..4.
- Float32 overflow and infinite features fail with file/line context. NaN is
  accepted as LightGBM's native missing value. Empty and malformed inputs fail
  explicitly. No clipping or new feature engineering is introduced.

The timer cannot preempt parsing, dataset construction or a single native
iteration. The 16-minute target leaves substantial margin below 20 minutes,
but this is a safeguard, not a hard execution-time proof. When that emergency
timer is inactive, fixed-seed outputs are deterministic; if it fires, the
number of completed trees depends on machine speed/load. A time-based fallback
cannot also promise cross-machine bitwise equality at the timeout boundary.

The preserved baseline and final recipe were compared through the unchanged
staff `run_recipe_and_evaluate()` on all three provided sets, then the final
recipe was repeated with seed 0. All saved model bytes match across old/new
and repeated runs. Full TRAIN + supplied validation stopping gives nDCG@10
0.943638070 / 0.942557838 / 0.952455956, respectively. These are mechanical
validation scores, not unseen hidden scores. Results and timings are in
`runs/c1_repro/recipe_regression.json`.

### Large stress checks

`scripts/stress_c1.py` generated deterministic independent TRAIN/validation
files with noisy relevance signal, 100 documents per query, labels 0..4,
trailing comments and all 300 feature indices. The first five columns carry
dense inputs; other columns follow the listed density. Timing starts at the
actual recipe call, including parsing, LightGBM import/training and saving,
and excludes generating test inputs. Each run uses four LightGBM threads in
a separate Python process. Peak working set is measured with the Windows
process-memory API; no OS memory cap or slow-machine emulation was imposed.

| Workload | TRAIN / validation rows | Feature width | Recipe seconds | Peak working set MiB | Saved trees | Normal-interface load |
|---|---|---|---|---|---|---|
| Sparse, other columns 10% nonzero | 500,000 / 50,000 | 300 | 40.04 | 1,007 | 139 | Pass |
| Dense | 100,000 / 10,000 | 300 | 51.43 | 606 | 254 | Pass |
| Dense, about 2.1 GB input text | 500,000 / 50,000 | 300 | 168.49 | 1,539 | 187 | Pass |

These are full row/width stress cases, but generated relevance and query
distributions need not resemble the hidden benchmarks. Windows workstation
speed and working-set accounting differ from staff Ubuntu hardware. The
results demonstrate substantial local margin, not guaranteed compliance on
every hidden workload. The emergency timer did not fire during these runs;
its stopping/save behavior was tested separately with forced expiry.
Evidence is in `runs/c1_repro/stress_sparse500k.json`,
`stress_dense100k.json` and `stress_dense500k.json`.

## Exact submitted-model reproduction

The standard entry point now supports both its original basic mode and an
explicit `submitted` configuration. Existing defaults remain 100 trees,
15 leaves, minimum leaf size 50, rate 0.05, raw features and TRAIN-only fitting.
`--config submitted` defaults to the current 200-tree configuration and shared
query-relative preprocessing. `--query-features` / `--no-query-features`,
`--trees`, `--seed`, `--out` and the existing `--use-dev` make these choices
explicit. `--use-dev` verifies disjoint TRAIN and DEV query IDs.

Exact measured command:

```bash
python scripts/train_model.py --config submitted --query-features --train data/train --dev data/dev --use-dev --trees 200 --seed 0 --out runs/c1_repro/reproduced_model.txt --report-out runs/c1_repro/reproduction_training.json
```

Inputs are `data/train/train.txt` and `data/dev/vali.txt`, the released Fold1
TRAIN and DEV. **Both supply training gradients**, totaling 958,671 rows /
8,000 queries. DEV is in-training and is not an unseen generalization estimate.
The neural artifacts retain their separately documented TRAIN-only gradient
usage and DEV epoch selection. No submitted model was replaced by this run.

The current model uses 136 raw features followed by their 136 average-tie
within-query percentiles, in that exact order. Constants and single-document
queries get zero relative features. For a nonconstant column with n documents,
the relative feature is its zero-based average tie rank divided by n-1.
The same implementation in `ranking._augment_query` is reused by
`submission.query_features.augment_grouped` in training, CV and evaluation.
There are no label inputs or fitted corpus statistics in this transform.
Float32 conversion and NaN/tie conventions are consequently shared.

The model's exact `raw_0 ... raw_135 qrank_0 ... qrank_135` names declare this
schema. `prepare()` detects those names, rather than inferring a dataset
identity or looking only at feature count. C1 artifacts have ordinary raw
feature names. A new regression explicitly trains a raw 272-column C1 model
and verifies that the shared inference interface does not augment it.
`ranking.py` needed no changes.

Submitted parameters: 200 trees, 31 leaves, rate 0.05, minimum leaf size 100,
L2=5, feature fraction 0.7, bagging fraction 0.8/frequency 1, `extra_trees=True`,
63 bins, deterministic row-wise training and four threads. All explicitly set
RNG seeds are 0; remaining library parameters match the original artifact
(including derived extra-tree seed 11797 and default truncation 30).

Verification used Python 3.13.7, LightGBM 4.7.0 on Windows 11. For exact byte
reproduction, use LightGBM 4.7.0; other allowed library versions/platforms have
not been claimed byte-identical. Reproduction took 189.61 seconds, including
reading, preprocessing, fitting and training-score reporting.

- Original and reproduced SHA-256:
  `09980276dba725e9d24e9e47f68b845cbd7f82c5161c11aa050c50f204aa48a7`.
- Byte equality: yes. Full parameter dictionaries, 272 features and 200 trees
  match exactly.
- All 235,259 DEV rows: maximum absolute prediction difference 0.0.
- All 2,000 DEV query orders: exact agreement.
- Both DEV nDCG@10: 0.49485494386221607, explicitly an in-training metric.

Evidence: `runs/c1_repro/reproduction_training.json` and
`reproduction_verification.json`. Standard library plus existing permitted
NumPy/LightGBM dependencies suffice; no submission dependency was added.

## Regression and test commands

All existing tests and the staff harness were preserved. Baseline testing:
65 passed, with `test_end_to_end_toy` deselected because it requires a toy-trained
tree. Final real-artifact suite: 80 passed, one toy test deselected. Separate
smoke workspace: 74 intended smoke/conformance tests passed, then all 81 tests
passed with its toy-trained tree. The isolated smoke runs the original shell
script's commands using Python directly on Windows; the repository's original
smoke script and CI workflow were not changed. Toy training never touched the
real submission model.

Commands used, with all output artifacts outside `submission/`:

```bash
python -m pytest tests -q -k 'not end_to_end_toy'
python -m pytest tests/test_c1_reliability.py tests/test_training_cli.py -q
python scripts/benchmark_c1.py
python scripts/verify_c1_regression.py
python scripts/stress_c1.py
python scripts/stress_c1.py --rows 100000 --density 1 --name stress_dense100k
python scripts/stress_c1.py --rows 500000 --density 1 --name stress_dense500k
```

In the disposable copy `tmp/c1_smoke/`, the intended smoke commands were run:

```bash
python scripts/make_toy_data.py --out data/toy
python scripts/train_model.py --train data/toy --out submission/model.txt --trees 30
python -m harness.run_harness --features data/toy/vali.txt --model submission/model.txt
python -m pytest tests --ignore=tests/test_student_todos.py -q
python -m pytest tests -q
```

Checks include parser regrouping/alignment, missing split indices, sparse and
variable widths (including raw width 272), label subsets, singleton/uniform
queries, trailing comments, numerical failures, basic-CLI prediction parity,
engineered-CLI inference parity and deadline-best-iteration behavior. A forced
expired deadline produces a loadable one-tree model instead of losing the
artifact. Twenty-five complete authorized TRAIN queries also matched the
preserved inference outputs exactly.

Final real-DEV preflight using the unchanged staff neural loader passes:
tree nDCG@10 0.494854944, pointwise 0.424748287, pairwise 0.445577828 and
BM25 0.288486781. Both learned rankers clear the BM25 floor. Largest-query
reranking took 0.048 s; prepare took 0.016 s. All three artifact hashes and
the inference/fusion/neural source files match the preserved versions.
No hidden C2 score was computed locally.

## Packaging and remaining limits

The builder includes all seven required entries, repository sources/docs,
the assignment specification, `.gitignore` and the original `.github` CI
workflow. It keeps empty dataset directories while excluding all dataset
contents, caches, credentials, experiment outputs and backups. ZIP integrity,
required entries, artifact bytes and the 50 MB size limit are checked before
upload. The final filename is `2023CS10076.zip`.

C1 hidden generalization cannot be measured locally and C1 is graded after
the competition deadline. These changes make no claim of a hidden accuracy
gain. Model reproduction and regression checks are exact for the measured
environment; the staff Ubuntu/Python 3.12 machine is a different environment.
The report and other post-competition Moodle deliverables remain separate
from this leaderboard ZIP, as specified by the assignment.
