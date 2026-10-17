"""Grouped 5-fold cross-validation sweep over the number of trees (assignment Section 4.3).

    python scripts/cv_lambdamart.py --train data/train --trees 25 50 100 200 400

For each tree count we train on 4 folds and score the 5th. NO early stopping is used on the reported fold.
The gap between the train-fold and validation-fold nDCG@10 is your local estimate of how much a model loses on
queries it has not seen: a model whose gap widens with more trees is overfitting. Pick the tree count where validation stops improving.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import json

import numpy as np

from harness.cv import grouped_kfold
from scripts.ltr_data import load_splits
from scripts.lambdamart_utils import default_params, fit, ndcg10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/toy")
    ap.add_argument("--trees", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--num-leaves", type=int, default=15)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-out", help="optional JSON with fold scores and selected tree count")
    args = ap.parse_args()
    path = os.path.join(args.train, "train.txt") if os.path.isdir(args.train) else args.train
    (X, y, q), _ = load_splits(path)
    folds = grouped_kfold(q, k=args.folds, seed=args.seed)
    params = default_params(args.num_leaves, args.learning_rate, args.min_data_in_leaf, seed=args.seed)
    results = []
    print(f"{'trees':>6} {'train':>8} {'val':>8} {'val std':>8} {'val/train':>10}")
    for n in args.trees:
        tr, va = [], []
        for tr_idx, va_idx in folds:
            m = fit(X[tr_idx], y[tr_idx], q[tr_idx], params, n)
            tr.append(ndcg10(m, X[tr_idx], y[tr_idx], q[tr_idx]))
            va.append(ndcg10(m, X[va_idx], y[va_idx], q[va_idx]))
        train_mean, val_mean = float(np.mean(tr)), float(np.mean(va))
        ratio = val_mean / train_mean if train_mean > 0 else None
        std = float(np.std(va))  # population standard deviation across the folds
        results.append({"trees": n, "train_mean": train_mean, "validation_mean": val_mean,
                        "validation_std": std, "validation_train_ratio": ratio,
                        "train_folds": tr, "validation_folds": va})
        ratio_text = f"{ratio:.4f}" if ratio is not None else "undefined"
        print(f"{n:>6} {train_mean:>8.4f} {val_mean:>8.4f} {std:>8.4f} {ratio_text:>10}")
    # Treat differences <= 1e-4 nDCG as near ties; prefer fewer trees.
    best = max(r["validation_mean"] for r in results)
    selected = min(r["trees"] for r in results if r["validation_mean"] >= best - 1e-4)
    print(f"selected trees: {selected} (near-tie tolerance 0.0001)")
    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            json.dump({"train_path": path, "folds": args.folds, "seed": args.seed,
                       "params": params, "results": results, "selected_trees": selected}, f, indent=2)


if __name__ == "__main__":
    main()
