# First-pass submission notes

Status as of 16 October 2026: implementation and synthetic checks completed.
Real released `data/train/train.txt` and `data/dev/vali.txt` are absent. Every
saved submission artifact is toy-trained and must be replaced before a real
competition submission. No withheld data or additional relevance labels were
obtained. The final assignment report has not been generated.

## Data and artifact declarations

- `submission/model.txt`: LightGBM text, 136 features, 100 trees, seed 0;
  gradients use only `data/toy/train.txt` (120 queries, 3000 rows). Tree count
  selected using five query folds of that same synthetic TRAIN, without early
  stopping. `data/toy/vali.txt` (40 queries, 1000 rows) used for reporting only.
  This model was explicitly trained AFTER the smoke test's 30-tree toy model.
- `submission/pointwise_model.pt`: toy TRAIN only; toy validation selects the
  best epoch. Ran 20 epochs, restored epoch 20.
- `submission/pairwise_model.pt`: toy TRAIN only, relevance-label preferences;
  toy validation selects the best epoch. Ran 10 epochs, restored epoch 7.
- Transfer recipe: each supplied TRAIN file supplies gradients; the supplied
  validation file selects the best round. There is no train+validation refit.
  Only the three provided synthetic transfer datasets were run locally.
  Their temporary models were loaded and evaluated, then removed by the dry-run
  runner. No real or hidden benchmark tuning was performed.

## Methods and configuration

Fusion implements unweighted textbook RRF, CombSUM and CombMNZ. Default
`fuse_candidates()` uses RRF with c=60. Ranks start at 1, higher scores are
better even when negative, and channel ties retain candidate order. CombSUM
uses per-channel min-max normalization; constant channels contribute zero.
CombMNZ multiplies that sum by the number of strictly positive normalized
scores. No labels or fitted preprocessing enter fusion.

Both neural networks use the staff `Standardizer` fitted only on TRAIN and the
staff `ScoringNetwork`: actual input width -> 64 ReLU -> 1 score. Adam uses
learning rate 0.001, batch size 1024, maximum 20 epochs, patience 3 and seed 0.
Pointwise loss is mean squared error. RankNet is mean softplus of the negative
preferred-minus-other score difference, without weighting or distillation.
Python, NumPy and PyTorch are seeded; deterministic PyTorch operations and at
most four CPU threads are used. Validation scores are evaluated over whole
queries after each epoch; scoring in chunks bounds memory. The best weights
are restored, and artifacts use the unchanged staff serialization format.

Pair sampling is uniform without replacement over all unequal-label pairs in
each query, capped at 300 per epoch. Cartesian label-bucket blocks and sampled
integer offsets avoid constructing all possible document pairs. Preferred
documents always occupy column 0. Queries may have noncontiguous source rows;
all-tied queries contribute no pairs. Pairs are resampled and shuffled each epoch.

LambdaMART uses `lambdarank`, nDCG@10, 15 leaves, learning rate 0.05,
`min_data_in_leaf=50`, `lambda_l2=1`, feature fraction 0.8, bagging fraction 0.8
and bagging frequency 1. Training/data/feature/bagging seeds are 0; deterministic
row-wise mode is enabled. Offline training/CV and per-query predictions use one
thread; the transfer recipe trains with four. No tree standardization or extra
features are used. Shared fitting stably regroups features, labels and query IDs.

Transfer uses one bounded run of at most 200 rounds, with early stopping after
30 non-improving validation rounds, saving the best iteration. A two-pass reader
handles sparse indices and trailing comments, infers a common width from both
supplied files, and fills float32 arrays directly in query order. A 500k x 300
feature matrix uses 600 MB; the implementation avoids retaining feature tuples
for every row. The same reader supports offline CLIs. Maximum-scale runtime and
peak memory have not been measured; only the supplied toy datasets were timed.

Inference loads the artifact once, checks feature width against that loaded
model, deduplicates candidates, predicts with one thread and sorts stably.
Missing/corrupt artifacts raise exceptions and clear any previously loaded
model. Non-finite predictions raise a clear error.

## Recorded synthetic results

These are synthetic validation results, not real DEV or leaderboard scores:

- Default RRF: nDCG@10 0.9341, MAP@10 0.8474.
- Final 100-tree LambdaMART: nDCG@10 0.9456 (TRAIN nDCG@10 0.9917).
- Pointwise: nDCG@10 0.9119.
- RankNet: nDCG@10 0.9358; single whole-document BM25 feature: 0.9287.
- Transfer validation nDCG@10: 40 features/labels 0..4: 0.9436;
  90 features/labels 0..4: 0.9426; 250 features/labels 0..2: 0.9525.
  The recorded JSON dry run took 1.0, 0.6 and 1.6 seconds respectively, well
  below 1200 seconds locally. These timings do not establish maximum-scale speed.

Five-fold grouped toy TRAIN CV, fixed tree counts, no early stopping:

- 50 trees: train mean 0.983244; validation mean 0.941386, population std
  0.003293; uncapped validation/train ratio 0.957429.
- 100 trees: train mean 0.992903; validation mean 0.942880, std 0.005198;
  ratio 0.949619.
- 200 trees: train mean 0.996693; validation mean 0.941207, std 0.004434;
  ratio 0.944330.

The strongest mean selects 100 trees; differences within 0.0001 nDCG prefer the
smaller model. This validates the CV pipeline only and does not select a real
MSLR tree count. `runs/toy_cv.json` preserves every fold's scores and parameters;
`runs/toy_harness.json`, `runs/toy_neural.json`, and `runs/toy_transfer.json`
preserve evaluations. `submission/*_history.json` preserves neural epoch losses
and validation scores. `runs/verification.json` records versions and artifact hashes.

## Reproduction

Use an environment with the declared `requirements.txt`. Local checks ran on
Windows with Python 3.13.7; Python 3.12/Ubuntu has not been exercised here.
The implementation uses Python 3.12-compatible syntax and the allowed packages.
Activate the local `.venv` if using the environment created during this work.

```bash
python -m pytest tests/test_student_todos.py -q
python -m pytest tests --ignore=tests/test_student_todos.py -q
bash scripts/smoke_test.sh
python scripts/cv_lambdamart.py --train data/toy --trees 50 100 200 --report-out runs/toy_cv.json
python scripts/train_model.py --train data/toy --out submission/model.txt --trees 100 --seed 0
python scripts/train_neural.py --kind both --train data/toy --dev data/toy --out-dir submission --seed 0
python -m harness.run_harness --features data/toy/vali.txt --model submission/model.txt --report-out runs/toy_harness.json
python scripts/evaluate_neural.py --features data/toy/vali.txt --report-out runs/toy_neural.json
python scripts/run_recipe.py --report-out runs/toy_transfer.json
```

With authorized released real TRAIN and DEV files in place, rerun the following.
Replace `SELECTED_TREES` with the tree count selected by the real TRAIN CV JSON;
the placeholder is not a literal CLI value. Do not run the toy smoke test after
producing the final real model.

```bash
python scripts/cv_lambdamart.py --train data/train --trees 50 100 200 --report-out runs/mslr_cv.json
python scripts/train_model.py --train data/train --dev data/dev --out submission/model.txt --trees SELECTED_TREES --seed 0
python scripts/train_neural.py --kind both --train data/train --dev data/dev --out-dir submission --seed 0
python -m harness.run_harness --features data/dev/vali.txt --model submission/model.txt --report-out runs/mslr_dev.json
python scripts/evaluate_neural.py --features data/dev/vali.txt --report-out runs/mslr_neural.json
```

Declare neural early stopping on real DEV once performed; real DEV labels should
not supply training gradients. The requested real-data CV, training, metrics and
real BM25 comparison are pending, not inferred from toy results. Verify the
three final artifacts before packaging; exclude environment/cache and data files.

## Next iteration

First obtain the authorized released TRAIN/DEV splits and establish genuine
metrics. Then use TRAIN query folds for modest tree regularization/round-count
changes, inspect neural learning curves and the real BM25 comparison, and measure
transfer resource use at larger synthetic sizes. No LambdaRank weighting,
exotic architectures or exhaustive tuning was added in this pass.

## AI-use disclosure and provenance

Codex assisted with repository inspection, implementation, testing and
documentation. ChatGPT assisted with assignment analysis and preparation of the
implementation prompt. The student must verify and complete all other personal
disclosure details.

Code/formula sources consulted: the supplied `a3-rankfusion-LTR.pdf` (version 1.5),
the [course starter repository](https://github.com/col7364iitd/2026-a3-student),
its `docs/RANKNET_PRIMER.md`, interface/grading/transfer/data documentation, and
the unchanged harness implementations. RRF, min-max CombSUM/CombMNZ, squared
error and RankNet formulas follow those materials. No external implementation
was copied; LightGBM supplies the explicitly permitted LambdaMART objective.
