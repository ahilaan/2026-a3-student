# Submission notes

## Current optimization candidate (18 October 2026)

This section supersedes the historical baseline descriptions below. The tree
and pointwise artifacts are retained from baseline #17. Their split declarations
and configurations below still apply. The current fusion and pairwise model
have changed as follows. Expanded tree CV is running separately; experimental
models are never loaded implicitly at inference.

Fusion was tuned using TRAIN only: 4,800 tuning and 1,200 reserved confirmation
queries, seed 0, all documents of a query together. A finite grid compared
weighted RRF constants 0/5/20/60/100/300/1000 and weighted min-max CombSUM powers
0.5/1/2. Nonnegative integer weights sum to four. The selected square-root
min-max rule weights whole-document BM25=2, title BM25=1, Dirichlet LMIR=1 and
JM LMIR=0. Tuning nDCG@10=0.301608; confirmation=0.291043. Constants were fixed
before DEV evaluation. Real DEV nDCG@10=0.309807338, MAP@10=0.130272851.
The graded textbook fusion functions retain their definitions.

The current pairwise artifact is a staff-compatible 136 -> 128 ReLU -> 64 ReLU
-> 1 MLP. Full TRAIN relevance-label preferences supply gradients, TRAIN fits
the standardizer, DEV selects neural epochs/configurations. No DEV gradients or
distillation. Initial AdamW run: learning rate 0.001, weight decay 0.0001, batch
2048, maximum 400 sampled pairs/query/epoch, seed 0; best epoch 3. Fine-tuning
starts there with learning rate 0.0002, weight decay 0.001, training-only input
dropout 0.03 and patience 5; the selected checkpoint is fine-tuning epoch 2.
Real DEV nDCG@10=0.445577828, reproduced through the unchanged staff loader.

The separate optional loss in scripts/optimize_neural.py is nDCG-weighted
RankNet, expressly allowed by the assignment. Sampled unequal-label pairs touch
the current top 20; weights are absolute swap delta-nDCG@10 divided by their
query mean, plus 0.1 to retain a lower-rank learning signal. The graded
ranknet_loss, pointwise loss, sampler and optimizer steps remain textbook.
Dropout adds no inference layers; the staff artifact format is unchanged.

Expanded LambdaMART CV compares 15/31/63 leaves and fixed 200/400/800 prefixes of
an 800-tree fit in each TRAIN query fold. No reported fold supplies early
stopping. Results are saved after each fit in runs/optimization/tree_cv.json.
The currently shipped 200-tree model retains its completed baseline CV below.
Each tree fit uses at most four threads; neural fits use two.

Reproduce tuning with scripts/optimize_rankers.py (modes cache/fusion/trees).
The script caches only the explicitly released TRAIN/DEV files, excluded from
the ZIP. Reproduce neural training with:

```bash
python scripts/optimize_neural.py --name lambda128x64 --threads 2
python scripts/optimize_neural.py --name lambda128x64_finetune --threads 2 --lr 0.0002 --weight-decay 0.001 --input-dropout 0.03 --init data/cache/optimization/lambda128x64/pairwise_model.pt
```

Fresh correctness/interface checks pass 55 tests with the stock toy-accuracy
test deselected. CI also runs that unchanged test with its intended toy fixture.
Normal aggregate portal feedback is used to assess submissions, respecting the
10-minute cooldown and one-active-submission limit. No held-out file or hidden
benchmark identity was sought, opened, processed or used. Codex also assisted
with the optimization code and experiments described here.

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
