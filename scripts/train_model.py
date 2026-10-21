"""Train the LambdaMART artifact that submission/ranking.py loads (offline; grading never retrains).

    python scripts/train_model.py --train data/train --out submission/model.txt --trees 300
    python scripts/train_model.py --config submitted --query-features --train data/train --dev data/dev --use-dev --trees 200 --seed 0 --out runs/reproduced_model.txt

--train takes a split file or a directory holding train.txt. Pass --use-dev to ALSO train on the dev split;
that raises your dev nDCG (Tracks A and B) without helping on queries the model has not seen, so your public
leaderboard position will overstate where you stand (assignment Section 7). Whatever you do, declare it in your
submission notes (Section 10).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import json
import time
import numpy as np

from scripts.ltr_data import load_splits
from scripts.lambdamart_utils import default_params, fit, ndcg10
from submission.query_features import augment_grouped, feature_names


def _resolve(path, name):
    return os.path.join(path, name) if os.path.isdir(path) else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/toy")
    ap.add_argument("--dev", default=None, help="dev file/dir; used for reporting only unless --use-dev")
    ap.add_argument("--out", default="submission/model.txt")
    ap.add_argument("--config", choices=["basic", "submitted", "current-main"], default="basic",
                    help="submitted: legacy 200-tree fit; current-main: exact 4400-tree pipeline")
    ap.add_argument("--work-dir", default="runs/final_c1_pass/reproduction")
    ap.add_argument("--assembly-only", action="store_true",
                    help="current-main: verify retained components and rebuild only the blend")
    ap.add_argument("--dry-run", action="store_true", help="current-main: print the full training plan")
    ap.add_argument("--query-features", action=argparse.BooleanOptionalAction, default=None,
                    help="append shared query percentiles (default: enabled for submitted)")
    ap.add_argument("--trees", type=int, default=None, help="default: basic 100; submitted 200")
    ap.add_argument("--num-leaves", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-dev", action="store_true", help="also fit DEV labels; must be declared")
    ap.add_argument("--report-out", help="write configuration, split declaration and measurements as JSON")
    args = ap.parse_args()
    if args.config == 'current-main':
        from scripts.reproduce_current import reproduce
        reproduce(args)
        return
    if args.assembly_only or args.dry_run:
        ap.error('--assembly-only and --dry-run require --config current-main')
    started = time.perf_counter()
    optimized = args.config == "submitted"
    args.trees = args.trees if args.trees is not None else (200 if optimized else 100)
    args.num_leaves = args.num_leaves if args.num_leaves is not None else (31 if optimized else 15)
    args.min_data_in_leaf = args.min_data_in_leaf if args.min_data_in_leaf is not None else (100 if optimized else 50)
    relative = args.query_features if args.query_features is not None else optimized
    if args.trees < 1:
        ap.error("--trees must be positive")

    dev_path = _resolve(args.dev, "vali.txt") if args.dev else _resolve(args.train, "vali.txt")
    train, dev = load_splits(_resolve(args.train, "train.txt"), dev_path if os.path.isfile(dev_path) else None)
    if args.use_dev:
        if dev is None:
            raise SystemExit("--use-dev needs a dev file")
        if np.intersect1d(np.unique(train[2]), np.unique(dev[2])).size:
            raise SystemExit("TRAIN and DEV must have disjoint query IDs for --use-dev")
        train_used = tuple(np.concatenate([a, b]) for a, b in zip(train, dev))
    else:
        train_used = train
    X, y, q = train_used
    if relative:
        X = augment_grouped(X, q)
    extra = dict(lambda_l2=5, extra_trees=True, max_bin=63, feature_fraction=0.7, num_threads=4) if optimized else {}
    params = default_params(args.num_leaves, args.learning_rate, args.min_data_in_leaf, seed=args.seed, **extra)
    model = fit(X, y, q, params, args.trees,
                feature_names=feature_names(X.shape[1] // 2) if relative else "auto")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save_model(args.out)
    print(f"trained on {len(set(q))} queries ({'train+dev' if args.use_dev else 'train only'}) -> {args.out}")
    report = {"config": args.config, "params": params, "seed": args.seed,
              "train_path": _resolve(args.train, "train.txt"), "dev_path": dev_path,
              "training_splits": "TRAIN+DEV" if args.use_dev else "TRAIN only",
              "feature_schema": "raw_plus_query_percentiles" if relative else "raw",
              "features": model.num_feature(), "trees": model.num_trees(),
              "rows": len(y), "queries": len(set(q)), "out": args.out,
              "train_ndcg@10": ndcg10(model, X, y, q)}
    print(f"train nDCG@10 = {report['train_ndcg@10']:.6f}")
    if dev is not None and not args.use_dev:
        Xd, yd, qd = dev
        if relative:
            Xd = augment_grouped(Xd, qd)
        report["dev_ndcg@10"] = ndcg10(model, Xd, yd, qd)
        print(f"dev   nDCG@10 = {report['dev_ndcg@10']:.6f}")
    report["seconds"] = time.perf_counter() - started
    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as target:
            json.dump(report, target, indent=2)


if __name__ == "__main__":
    main()
