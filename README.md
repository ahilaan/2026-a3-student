# A3 -- Rank Fusion + LTR Arena (student starter)

Fuse four base rankers, then train pointwise / RankNet / LambdaMART rankers on MSLR-WEB10K Fold1 without overfitting. Full spec: the assignment PDF. Held-out data is **not** here and must not be sought (honor code).

## Quick start
```
pip install -r requirements.txt
bash scripts/smoke_test.sh                 # toy data, trains, runs the harness, runs tests
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

The current `submission/model.txt` / `pointwise_model.pt` / `pairwise_model.pt` are trained on **synthetic toy TRAIN only**, not real MSLR. LambdaMART uses 100 trees selected by grouped CV on toy TRAIN; toy validation is reporting-only. Both neural models use toy validation for early stopping. No real TRAIN/DEV files were available. Replace these artifacts after authorized released data are supplied.

The baseline uses all supplied features: an input-dimension -> 64 ReLU -> 1 MLP, Adam at 0.001, batches of 1024, up to 20 epochs and patience 3. RankNet samples up to 300 distinct pairs per query per epoch. LambdaMART uses 15 leaves, learning rate 0.05, minimum leaf size 50, L2=1, and feature/bagging fractions 0.8. The transfer recipe uses the same regularization, at most 200 rounds and validation patience 30, with four threads. CLI readers align sparse feature widths and group queries without retaining a large Python object per feature.

Reproduction commands, split declarations, measured toy results and limitations are in [submission notes](docs/SUBMISSION_NOTES.md); JSON results are in `runs/`, and neural epoch histories accompany the artifacts. **Run the toy smoke test before final real-data training: it overwrites `submission/model.txt`.** Exclude `.venv/`, caches and data from the submission ZIP.

AI-use disclosure: Codex assisted with implementation, testing and documentation. ChatGPT assisted with assignment analysis and the implementation prompt. The student must verify and complete any other personal disclosure details. Code/formula sources consulted were the supplied assignment, starter repository and its documentation; no external implementation was copied.
