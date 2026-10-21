# COL7364 Assignment 3 - Ahilaan Saxena (2023CS10076)

Competition ZIP: `2023CS10076.zip`, due 22 October 2026 at 11:59 PM.
The report and statements are a separate Moodle submission due 24 October
at 11:59 PM (assignment Version 1.5, pages 1 and 12).
See [package verification](docs/COMPETITION_PACKAGE.md) and
[Moodle deliverables](docs/MOODLE_SUBMISSION.md). Raw datasets are excluded
from the ZIP; local tests require the supplied toy/transfer fixtures.

The current main LambdaMART artifact uses released Fold1 TRAIN+DEV labels:
958,671 original rows / 8,000 queries. DEV scores are trained-on measurements.
TEST is excluded from current fitting and new local evaluation. Historical
TEST-label experiments and model-selection exposure remain disclosed in
[docs/COMPLIANCE.md](docs/COMPLIANCE.md); refitting does not erase that exposure.

The native LightGBM artifact has 4,400 trees and 272 features: 136 raw plus
136 label-free within-query average-tie percentiles. Two augmented components
have 2,000 trees / 127 leaves / seed 0 and 2,400 trees / 63 leaves / seed 23,
weighted 0.65/0.35. Both use LambdaRank, rate .05, minimum leaf 100, L2 10,
extra randomized splits, 63 bins, feature/row fractions .8, row sampling every
round and four threads. There is no early stopping in these fixed-count fits.
Augmentation doubles query blocks using seed 76420 + component seed, noise
.02 times within-query range, log-normal feature scaling (sigma .5), and
14-40 dropped raw columns; transforms are recomputed after augmentation.
TRAIN grouped CV, trained-on DEV and portal feedback informed selection.

Reproduce the current pipeline without replacing submitted artifacts:

```bash
python scripts/train_model.py --config current-main --train data/train --dev data/dev --use-dev --work-dir runs/current_reproduction --out runs/current_reproduction/model.txt
```

This preserves each component's 1,200-tree save/reload boundary before
continuation and verifies output against the submitted model. Use `--dry-run`
to inspect commands. `--assembly-only` verifies retained component receipts
and rebuilds just the blend; those local receipts are not packaged.
`--config submitted` remains the legacy 200-tree configuration for CLI
compatibility; it does not reproduce the current model. Measured reproduction
status and limitations are in [docs/FINAL_C1_PASS.md](docs/FINAL_C1_PASS.md).
Configuration is in [docs/CURRENT_MODEL.json](docs/CURRENT_MODEL.json).

C1 reads only supplied TRAIN/validation and fits the original shallow raw
LambdaRank baseline first (15 leaves, minimum leaf 50, L2 1, rate .05,
feature/row fractions .8, maximum 800 rounds, patience 50, four threads).
It saves a valid raw fallback, then, if time remains, fits the same ranker
with label-free query percentiles appended. Only an inner-validation gain
above .001 selects that alternative; validation never supplies gradients.
The chosen single native artifact encodes its feature schema for inference.
A shared 960-second soft deadline includes parsing and optional transforms;
native calls cannot be preempted. Oversized queries use within-query native
blocks and whole-query validation. The final report records the 39-condition
comparison, 12 confirmation conditions, weak/noisy-data regressions, runtime
limits and stress tests. Hidden C1 accuracy remains unknown.

Fusion is handwritten equal-weight, query-min-max CombSUM over BM25 whole,
BM25 title and LMIR DIR whole (TRAIN-only selection). Pointwise uses TRAIN
MSE gradients, TRAIN standardization and DEV epoch selection. Pairwise is a
plain 256/128 ReLU MLP refined for four fixed epochs on TRAIN+DEV relevance
labels, seed 71, rate .00003, dropout .03 and averaging from epoch 1. It keeps
TRAIN-only standardization and uses no teacher scores. Exact initialization
and history are in docs/CURRENT_NEURAL.json and docs/NEURAL_PROVENANCE.md.

Install requirements.txt. Run `python -m pytest tests -q -k "not test_end_to_end_toy"`
and `python scripts/preflight.py`. The toy quality threshold expects a toy-trained
model; run that smoke workflow in a disposable copy because its training
command overwrites model.txt. The builder excludes raw data and run outputs
and verifies the seven required files and 50 MB ZIP cap. docs/REPORT.pdf is
the existing three-page assignment report; the new pass is documented separately.

AI-use disclosure: Codex assisted with implementation, experiments, testing,
documentation and earlier authorized portal submissions. ChatGPT assisted with
assignment analysis. Code/formula provenance: assignment and starter repository;
no external implementation copied in this pass. The student must verify these
disclosures. This pass does not submit to the leaderboard.
