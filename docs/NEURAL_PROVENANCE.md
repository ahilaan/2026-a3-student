# Verified baseline neural artifacts

This records the baseline initialization. If docs/CURRENT_NEURAL.json is
present, that manifest describes the active neural refinement and its hash;
the baseline below remains its provenance and reproduction reference.

Pointwise SHA-256: `cfbff50b8290bb112912256e29b36dbd5f62669b66d0f69fb44f17c547279495`.
TRAIN-only gradients, TRAIN standardization, DEV early stopping; standard
pointwise MSE training from scripts/train_neural.py.

Pairwise SHA-256: `00568f30451537e0592dcc37d4444a1333382cbd54205d4e7c3bbddd62691140`.
Plain 136->256->128->1 ReLU network. Concatenated hidden layers and block
diagonal connections reproduce a convex sum of two 128->64 MLPs: .3 seed 0,
.7 seed 23. Both constituents use TRAIN+DEV relevance labels only. The original
TRAIN-only standardizer is identical and frozen for both constituents.

The seed-0 constituent warm-started the earlier TRAIN-only 128->64 artifact
and ran exactly 9 epochs: AdamW .0001, decay .001, input dropout .03,
400 label-ordered same-query pairs/query, batch 2048, label-supervised
delta-nDCG pair weights, two threads. The seed-23 constituent starts fresh
and runs exactly 6 epochs: AdamW .0007, decay .001, dropout .05; otherwise
the same settings. No teacher scores supervise either model.

Study fitting excluded 400 DEV queries (permutation seed 764). The .3/.7
weight scored .442399977 there, versus .440312371 for the stronger single
study model. Starting-model epoch selection had already used DEV, and
development selection was repeated. This is not a fresh independent test.
Final refits include all TRAIN+DEV. Fold1 TEST was excluded throughout this
artifact's training and selection. A separate later TEST-trained research
model was never submitted and is not used here.

The optional four-epoch refinement starts from exactly the baseline hash
above, retains its TRAIN standardizer and plain 256/128 hidden layers, and
uses TRAIN+DEV labels with seed 71, learning rate 0.00003, dropout 0.03 and
parameter averaging from epoch 1. No DEV query is reserved in this full fit;
DEV scores are fitting measurements. Its held-out P must be measured by the
portal. Recreate the baseline below at the `args.init` path recorded in
CURRENT_NEURAL.json and verify `initial_artifact_sha256`, then run:

```
python scripts/score_campaign2_neural.py --name campaign4_pair_trusted_refine4 --full --seed 71 --lr 0.00003 --dropout 0.03 --epochs 4 --swa-start 1 --init runs/compliant_campaign/stages/campaign4_extra127075_seedmean025/pairwise_model.pt
```

To reproduce the original TRAIN-only starting checkpoint, run
scripts/optimize_neural.py with --name lambda128x64 --hidden 128 64 --loss lambda
--lr .001 --weight-decay .0001 --epochs 30 --patience 5 --threads 2, then
--name lambda128x64_finetune --hidden 128 64 --loss lambda --lr .0002
--weight-decay .001 --input-dropout .03 --epochs 30 --patience 5 --threads 2
--init data/cache/optimization/lambda128x64/pairwise_model.pt.

Use scripts/score_campaign2_neural.py with explicit --init pointing to that
starting checkpoint. For seed 0 use --name neural_dev_full9 --full --epochs 9;
for seed 23 use --name campaign3_fresh23_full6 --fresh --seed 23 --lr .0007
--dropout .05 --full --epochs 6. For the fresh fit the checkpoint provides
only architecture and original TRAIN standardizer. Combine the final
artifacts with scripts/blend_neural.py's blend(..., first_weight=.3) and
harness.neural.save_scorer. No fitting or distillation occurs during merging.
