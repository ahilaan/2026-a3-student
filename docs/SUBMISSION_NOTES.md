# Submission notes

## Current TRAIN+DEV query-relative candidate (18 October 2026)

This section supersedes the historical baseline below. The tree fits released Fold1 TRAIN+DEV labels, explicitly allowed by the spec
when declared: 958,671 rows / 8,000 complete queries. Neural gradients still use
TRAIN only (723,412 rows / 6,000 queries); DEV selects neural epochs. Tree DEV
is now an in-training metric. No held-out file, other fold, hidden benchmark
identity or hidden version construction was consulted.

The current LambdaMART artifact uses 200 trees, 31 leaves, learning rate 0.05,
minimum leaf size 100, L2=5, feature fraction 0.7 and bagging fraction 0.8,
random split thresholds (extra_trees=True), max_bin=63, bagging frequency
1, deterministic row-wise training, four threads, seed 0. Each provided feature
is retained and its within-query average-tie percentile is appended. Constant
columns and single-document queries get zero percentiles. This label-free,
query-local transform fits no corpus statistics; all documents of a query stay
together. The real 136-column input therefore produces 272 tree inputs, but
neither inference nor the feature transform hard-codes that dimensionality.
The model's raw_/qrank_ feature schema explicitly declares preprocessing.
Inference helpers are inside required ranking.py, with no extra runtime file.
Raw-feature models remain supported by their explicit ordinary feature schema.
No dataset name, query ID or input distribution selects the preprocessing.

Completed five-query-fold CV WITHIN the combined TRAIN+DEV data, seed 0,
fixed counts and no early stopping on reported folds (validation mean /
population std / validation-TRAIN ratio):

- 100 trees: 0.461986 / 0.004823 / 0.973882.
- 200 trees: 0.472384 / 0.004721 / 0.957007.

200 trees wins in all five folds. This randomized regularization alternative
has a slightly lower mean but lower fold variability than the ordinary
15-leaf combined-data control (0.472940 mean / 0.006371 std). It is tested as
a separate candidate; the control remains preserved as submission #38, which
passed all 10 correctness checks and reached rank 2 (score 0.5522).
The released TRAIN and DEV query sets
were verified disjoint; all rows and appended features of each query remain in
one fold. The percentile transform fits no corpus statistics. Public DEV is
included in the final tree fit, so it is not an unseen generalization estimate.
The separate TRAIN-only control scored 0.482551 on unseen DEV at 400 trees;
its TRAIN query-fold validation was 0.474710. The combined CV uses a different
query pool and fold assignment. Normal aggregate portal feedback motivates
controlling fit; no held-out labels were used locally.
Full configurations and all five fold scores are in runs/optimization/.

The pointwise artifact is retained from baseline #17: TRAIN-fitted standardizer,
136 -> 64 ReLU -> 1 MLP, MSE, Adam 0.001, batch 1024, best DEV epoch 7 of 10.
The pairwise artifact is a staff-compatible 136 -> 128 ReLU -> 64 ReLU -> 1
MLP, TRAIN relevance-label preferences and TRAIN-fitted standardization only.
DEV selects epochs/configurations. Initial AdamW 0.001, weight decay 0.0001,
batch 2048, at most 400 sampled pairs/query/epoch, seed 0: best epoch 3.
Fine-tuning from there uses learning rate 0.0002, weight decay 0.001,
training-only input dropout 0.03, patience 5: best fine-tuning epoch 2.
Both runs use two CPU threads. Dropout adds no inference layers. No distillation.
The separate optional neural loss is nDCG-weighted RankNet, explicitly permitted
by the assignment: pairs touch the current top 20; weights are absolute swap
delta-nDCG@10 normalized by the query mean, plus 0.1. Graded losses and optimizer
steps retain their textbook definitions and signatures. The artifact format is
the unchanged staff config/state_dict/standardizer MLP format.

Fusion tuning uses TRAIN only: 4,800 tuning and 1,200 reserved confirmation
queries, seed 0. A finite grid compared RRF constants, min-max powers and
nonnegative channel weights. A finer grid selected power 1.5, weights
whole-document BM25=2, title BM25=3, Dirichlet LMIR=3, JM LMIR=0. Tuning nDCG
0.302205 and confirmation 0.292748 passed a predeclared adoption rule before
DEV evaluation: tuning gain >=0.0005 and confirmation improvement over the
initial rule. Subsequent DEV 0.309080 is slightly below the initial rule's
0.309807, within the same leaderboard bands. DEV did not choose fusion settings.
The graded RRF, CombSUM and CombMNZ definitions remain unchanged.

Current real DEV nDCG@10: LambdaMART 0.494854944 (DEV in training), pointwise 0.424748287,
pairwise 0.445577828, BM25 0.288486781; fusion 0.309080471 and MAP 0.129081911.
Fresh staff-loader preflight confirms finite/repeatable scores and runtime.
The largest query takes approximately 0.03 seconds. The transfer recipe remains
a generic raw-feature LambdaMART, 15 leaves, at most 800 rounds and validation
patience 50, using only its two supplied files and four threads. It writes an
ordinary raw-feature schema accepted by the same prepare/rerank functions.
All three provided synthetic transfer dry runs passed; hidden data was not sought.

Upload #23 exposed a bitwise reproducibility issue: inheriting incoming OpenMP
thread state can change weights by about 3e-7 despite an identical seed. The
graded train_model helper now fixes one CPU thread. Regression tests exercise
both pointwise and pairwise training across different incoming thread states.
Offline optimized artifacts retain their explicitly fixed two-thread path.
The correction was uploaded separately as #24 and passed all 10 server checks;
#25 also passed all 10. No graded function was weakened.

Reproduction:

```bash
python scripts/setup_mslr.py
python scripts/optimize_rankers.py cache
python scripts/optimize_rankers.py fusion
python scripts/optimize_rankers.py fusion --fine
python scripts/cv_query_features.py --include-dev --variant extra --counts 100 200
python scripts/cv_query_features.py --include-dev --variant extra --counts 100 200 --full --trees 200
python scripts/optimize_neural.py --name lambda128x64 --threads 2
python scripts/optimize_neural.py --name lambda128x64_finetune --threads 2 --lr 0.0002 --weight-decay 0.001 --input-dropout 0.03 --init data/cache/optimization/lambda128x64/pairwise_model.pt
```

Training outputs live under data/cache/optimization/ and are promoted explicitly
after validation. Run scripts/preflight.py and scripts/build_submission.py on
the promoted artifacts. ZIPs include all seven grader entries, preserve the
repository layout and exclude datasets, caches, run outputs, credentials, Git
metadata and environments. The portal's cooldown uses server time (the local
clock runs ahead); only one team submission is active at once. Provided harness
and tests are unchanged. CI runs all original tests with their intended toy
fixture and also validates the committed real-artifact interface. The separate
post-competition 2-3-page report has not been generated.

Codex assisted with setup, implementation, experiments, testing and documentation;
ChatGPT assisted with assignment analysis and the implementation prompt. The
student must complete any other personal disclosure. Sources: supplied assignment,
starter code/docs and Microsoft dataset page. No external ranking implementation
was copied; LightGBM supplies the explicitly permitted tree ranking objective.

## Historical real-data baseline (#17)


Updated 17 October 2026. All three submitted model artifacts now use actual
MSLR-WEB10K Fold1 TRAIN. This is a properly trained baseline; no advanced
leaderboard optimization, DEV-gradient training or hidden-benchmark tuning was
performed. The final 2-3-page assignment report has not been generated.

## Data provenance and setup

The Microsoft dataset page publishes archive MD5
`97c5d4e7c171e475c91d7031e4fd8e79`. Its current OneDrive endpoint returned HTTP
403, "The request is blocked". The same 1,234,144,912-byte archive was downloaded
from `https://storage.googleapis.com/personalization-takehome/MSLR-WEB10K.zip`.
The full downloaded bytes matched Microsoft's published MD5 before extraction.
Only `Fold1/train.txt` and `Fold1/vali.txt` were opened/extracted. No `test.txt`
or file from another fold was opened, decompressed, processed or used.

`scripts/setup_mslr.py` performs checksum verification, selective extraction and
calls the existing `scripts/prepare_mslr.py`. Prepared data:

- `data/train/train.txt`: 723,412 rows, 6,000 queries, 136 features, labels 0..4.
- `data/dev/vali.txt`: 235,259 rows, 2,000 queries, 136 features, labels 0..4.

`runs/mslr_data_provenance.json` records download provenance, selected archive
entries and input SHA-256 hashes. `runs/mslr_split_stats.json` records aggregate
split counts. Dataset/cache files are Git-ignored and excluded from the ZIP.

## Training declarations and configuration

- `submission/model.txt`: LightGBM text, 200 trees, 15 leaves, seed 0. Gradients
  use full Fold1 TRAIN only. Tree count was selected by five query folds within
  TRAIN; DEV is reporting-only, with no early stopping or gradients on DEV.
  A single full-TRAIN run fit the maximum swept count; CV selected all 200 trees.
- `submission/pointwise_model.pt`: full Fold1 TRAIN relevance labels only;
  Standardizer fitted on TRAIN; DEV selects epoch 7. Ran 10 epochs and restored
  epoch 7 after patience 3.
- `submission/pairwise_model.pt`: full Fold1 TRAIN relevance-label preferences
  only; Standardizer fitted on TRAIN; DEV selects epoch 9. Ran 12 epochs and
  restored epoch 9 after patience 3. No model-score distillation.
- `submission/recipe.py`: staff-supplied TRAIN supplies gradients; supplied
  validation selects the best iteration. No train+validation refit. Only the
  provided synthetic transfer datasets were run locally; hidden data was not
  sought or inspected.

Both neural networks use the unchanged staff MLP/serialization: 136 inputs ->
64 ReLU units -> 1 score, Adam at 0.001, batch size 1024, maximum 20 epochs,
patience 3, seed 0 and at most four CPU threads. Pointwise minimizes mean squared
error; pairwise uses plain mean softplus of the negative preferred-minus-other
score difference. RankNet samples at most 300 distinct unequal-label pairs per
query per epoch, preferred first. Label-bucket Cartesian offsets and Floyd
sampling avoid constructing all possible pairs. Pairs are query-local,
resampled and shuffled each epoch. Python/NumPy/PyTorch seeds are set, and best
validation weights are restored.

LambdaMART uses `lambdarank`, nDCG@10, learning rate 0.05,
`min_data_in_leaf=50`, `lambda_l2=1`, feature fraction 0.8, bagging fraction 0.8
and bagging frequency 1. Training/data/feature/bagging seeds are 0; deterministic
row-wise mode is enabled. Offline tree training uses one CPU thread; transfer
training uses four. No tree standardization or extra features were introduced.
The transfer recipe uses at most 200 rounds and validation patience 30.

Fusion remains hand-written unweighted RRF (default c=60), CombSUM and CombMNZ.
Higher scores are better, ranks start at 1, ties preserve candidate order, and
constant channels normalize to zero. No relevance labels enter fusion.
Inference deduplicates candidates, checks feature width against the loaded
model and ranks stably. Missing/corrupt artifacts and non-finite predictions
raise clear exceptions. The staff harness and provided tests are unchanged.

## Real TRAIN cross-validation

Five query-grouped folds, seed 0, fixed tree counts, no early stopping on the
reported validation folds. All rows of a query stay together; no preprocessing
is fitted across folds. Each tree count was trained independently on four folds.

- 50 trees: train mean 0.450769; validation mean 0.441080, population std 0.007464; uncapped validation/train ratio 0.978506.
- 100 trees: train mean 0.468226; validation mean 0.454500, population std 0.006744; uncapped validation/train ratio 0.970687.
- 200 trees: train mean 0.487034; validation mean 0.464036, population std 0.006981; uncapped validation/train ratio 0.952781.

Selected 200 trees by strongest mean validation nDCG@10; differences within
0.0001 would prefer fewer trees. Complete fold scores and parameters are saved
in `runs/mslr_cv.json`.

## Real DEV results

All results below use the 2,000 Fold1 validation queries, not toy data:

- RRF: nDCG@10 0.299457603; MAP@10 0.124636892.
- LambdaMART: nDCG@10 0.471196142; MAP@10 0.150633605.
- Pointwise: nDCG@10 0.424748287.
- RankNet: nDCG@10 0.436420925.
- Single whole-document BM25 feature: nDCG@10 0.288486781.

Saved-artifact scores reproduced in fresh processes through the unchanged staff
loader. RankNet beats BM25 and fixes the locally reproducible cause of the
previous upload's 9/10 check. Final real-DEV harness and preflight reports are
`runs/mslr_harness.json`, `runs/mslr_neural.json` and `runs/mslr_preflight.json`.
No held-out score was computed locally.

## Verification and the toy-test assumption

- Required handwritten correctness tests: 7 passed.
- Direct full suite with real-trained artifacts: 52 passed, 1 failed. The sole
  failure is the stock `test_end_to_end_toy` assertion that LambdaMART scores
  above 0.5 on synthetic data; the real-trained model scores 0.278058 there.
  This synthetic accuracy assumption does not measure real-DEV model quality.
- In a disposable extracted copy with a separately toy-trained LambdaMART
  fixture, the unchanged Bash smoke test passed 46 tests and the unchanged
  full suite passed all 53. No final real model was overwritten.
- The actual real-data harness passed on all 2,000 queries. Preflight reloaded
  every artifact, checked finite/repeatable scores and ranking contracts, and
  verified both LambdaMART and RankNet beat BM25 on real DEV.
- Both `.pt` files have only staff-compatible config/state_dict/standardizer
  fields and a single hidden layer of width 64.

CI validates committed model interface contracts, then runs all provided tests
with their intended toy fixture, restoring the committed model afterward.
It does not modify or weaken any harness/test assertion. Run the toy smoke test
in a disposable copy, or before final real training: it writes a toy model.

## Reproduction

Use the declared requirements. Local training/evaluation ran on Windows,
Python 3.13.7, NumPy 2.5.4, PyTorch 2.14.1 and LightGBM 4.7.0. The source is
Python 3.12-compatible; CI uses Ubuntu 24.04 / Python 3.12.

```bash
python scripts/setup_mslr.py
python scripts/cv_lambdamart.py --train data/train --trees 50 100 200 --report-out runs/mslr_cv.json
python scripts/train_model.py --train data/train --out submission/model.txt --trees 200 --seed 0
python scripts/train_neural.py --kind both --train data/train --dev data/dev --out-dir submission --seed 0
python -m harness.run_harness --features data/dev/vali.txt --model submission/model.txt --report-out runs/mslr_harness.json
python scripts/evaluate_neural.py --features data/dev/vali.txt --report-out runs/mslr_neural.json
python scripts/preflight.py --features data/dev/vali.txt --report-out runs/mslr_preflight.json
python scripts/build_submission.py
```

The ZIP builder verifies the seven required grader entries, preserves the
starter directory layout and excludes datasets, archive/cache files, login
cookies, Git metadata, environments and experiment outputs. Its size limit is
50 MB. Epoch histories and experiment results remain in the repository for the
later assignment report; they are not required model inputs.

## AI-use disclosure and provenance

Codex assisted with inspection, downloading/setup, implementation, training,
testing and documentation. ChatGPT assisted with assignment analysis and the
implementation prompt. The student must verify any other personal disclosure
details. Formulas/code sources were the supplied assignment PDF, course starter
repository and its documentation/unchanged harness. Microsoft's dataset page
supplied archive identity and split information. No external ranking
implementation was copied; LightGBM supplies the permitted LambdaMART objective.
