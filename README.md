# A3 -- Rank Fusion + LTR Arena (student starter)

Fuse four base rankers, then train pointwise / RankNet / LambdaMART rankers on MSLR-WEB10K Fold1 without overfitting. Full spec: the assignment PDF. Held-out data is **not** here and must not be sought (honor code).

Current declarations and results are in [submission notes](docs/SUBMISSION_NOTES.md).
The current tree is a 200-tree/31-leaf randomized LambdaMART with generic raw-plus-query-
percentile features, fitted on **released Fold1 TRAIN+DEV** (958,671 rows / 8,000
queries), as explicitly allowed by the spec when declared. Its DEV nDCG@10
**0.494855 is an in-training metric**. Five query folds within the combined data
give validation nDCG@10 0.472384 (std 0.004721, validation/TRAIN ratio 0.957007).
The regularized alternative uses random split thresholds, minimum leaf size 100,
L2=5, 63 histogram bins and feature fraction 0.7; its learning rate is 0.05.
Neural models still use TRAIN gradients only: label-trained 128/64-unit
LambdaRank MLP DEV 0.445578 and pointwise DEV 0.424748. Fusion is TRAIN-only-tuned
weighted power CombSUM (DEV 0.309080). Held-out labels remain untouched.
The graded neural helper fixes one CPU thread for bitwise seed reproducibility;
optimized neural artifact training uses explicitly fixed two threads.

## Quick start
```
pip install -r requirements.txt
bash scripts/smoke_test.sh                 # toy data, trains, runs the harness, runs tests
python scripts/setup_mslr.py              # checksum-verified archive; extract ONLY Fold1 TRAIN/DEV
python scripts/prepare_mslr.py --fold-dir <MSLR-WEB10K/Fold1>     # real data -> data/train, data/dev
python scripts/run_recipe.py                      # Track C1 dry run on toy data (synthetic transfer sets; see docs/TRANSFER_TRACK.md)
python scripts/cv_lambdamart.py --train data/train --trees 50 100 200 400   # local train-vs-validation gap
python scripts/train_model.py --train data/train --out submission/model.txt --trees 200
python -m harness.run_harness --features data/dev/vali.txt --model submission/model.txt
```

## What you write
| file | what |
|---|---|
| `submission/fusion.py` | Hand-written RRF (default, c=60), CombSUM and CombMNZ, with stable ties |
| `submission/pointwise_pairwise.py` | Seeded pointwise MSE and plain RankNet training, query-local pair sampling, minibatching and best-validation-state early stopping |
| `submission/ranking.py` | the competition entry: keep the shape, improve the internals |
| `scripts/train_model.py` | your reproducible LambdaMART training script |
| `submission/recipe.py` | your training recipe, run by staff on undisclosed benchmarks for Track C1 (working default provided; tune it so it transfers) |

Self-check your parts: `python -m pytest tests/test_student_todos.py`. CI gates the interface (`tests/` minus that file). Additional edge checks live in `tests/test_baseline_edges.py`.

Docs: `docs/SUBMISSION_INTERFACE.md`, `docs/GRADING.md`, `docs/RANKNET_PRIMER.md`, `docs/DATA.md`, `docs/TRANSFER_TRACK.md`, `docs/SUBMISSION_NOTES.md` (fill in and keep as your README section).

## Layout
`harness/` metrics, I/O, CV, leaderboard formulas, runner (do not edit; staff run an identical copy) - `submission/` your entry - `scripts/` tooling - `data/toy/` and `data/toy_transfer/` synthetic data - `tests/`.

The original real-data baseline `submission/model.txt`, `pointwise_model.pt`, and `pairwise_model.pt` are trained on **actual MSLR-WEB10K Fold1 TRAIN only** (723,412 rows / 6,000 queries). LambdaMART uses 200 trees selected by five-fold query-grouped CV within TRAIN. DEV (235,259 rows / 2,000 queries) is reporting-only for LambdaMART and selects the neural early-stopping epoch. No DEV labels supply training gradients. Only Fold1 TRAIN and validation were extracted from the checksum-verified dataset archive; no held-out file was opened.

Real DEV nDCG@10: LambdaMART **0.471196**, pointwise **0.424748**, RankNet **0.436421**, single BM25 feature **0.288487**. RankNet now clears the learning floor that the previous toy-trained upload failed.

The baseline uses all supplied features: an input-dimension -> 64 ReLU -> 1 MLP, Adam at 0.001, batches of 1024, up to 20 epochs and patience 3. RankNet samples up to 300 distinct pairs per query per epoch. LambdaMART uses 15 leaves, learning rate 0.05, minimum leaf size 50, L2=1, and feature/bagging fractions 0.8. The transfer recipe uses the same regularization, at most 200 rounds and validation patience 30, with four threads. CLI readers align sparse feature widths and group queries without retaining a large Python object per feature.

Reproduction commands, split declarations, measured real DEV results and limitations are in [submission notes](docs/SUBMISSION_NOTES.md); JSON results are in `runs/`, and neural epoch histories accompany the artifacts. Run `python scripts/preflight.py` and `python scripts/build_submission.py` to verify the loaded models on real DEV and build the data-free `2023CS10076.zip`. **Run the toy smoke test before final real-data training: it overwrites `submission/model.txt`.** Exclude `.venv/`, caches and data from the submission ZIP.

AI-use disclosure: Codex assisted with implementation, testing and documentation. ChatGPT assisted with assignment analysis and the implementation prompt. The student must verify and complete any other personal disclosure details. Code/formula sources consulted were the supplied assignment, starter repository and its documentation; no external implementation was copied.

CI checks the committed models' interface contracts first, then runs the unchanged smoke/full tests with their toy-trained LambdaMART fixture in a disposable CI working copy. The stock `test_end_to_end_toy` assumes a toy-trained model and fails its synthetic accuracy threshold for the real-trained model; real artifact quality is checked on actual DEV. See the notes for exact test outcomes.
