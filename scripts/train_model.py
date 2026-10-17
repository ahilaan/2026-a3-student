"""Train the LambdaMART artifact that submission/ranking.py loads (offline; grading never retrains).

    python scripts/train_model.py --train data/train --out submission/model.txt --trees 300

--train takes a split file or a directory holding train.txt. Pass --use-dev to ALSO train on the dev split;
that raises your dev nDCG (Tracks A and B) without helping on queries the model has not seen, so your public
leaderboard position will overstate where you stand (assignment Section 7). Whatever you do, declare it in your
submission notes (Section 10).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import numpy as np

from scripts.ltr_data import load_splits
from scripts.lambdamart_utils import default_params, fit, ndcg10


def _resolve(path, name):
    return os.path.join(path, name) if os.path.isdir(path) else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/toy")
    ap.add_argument("--dev", default=None, help="dev file/dir; used for reporting only unless --use-dev")
    ap.add_argument("--out", default="submission/model.txt")
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--num-leaves", type=int, default=15)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-dev", action="store_true", help="also train on dev (lowers Track C; declare it)")
    args = ap.parse_args()

    dev_path = _resolve(args.dev, "vali.txt") if args.dev else _resolve(args.train, "vali.txt")
    train, dev = load_splits(_resolve(args.train, "train.txt"), dev_path if os.path.isfile(dev_path) else None)
    if args.use_dev:
        if dev is None:
            raise SystemExit("--use-dev needs a dev file")
        train_used = tuple(np.concatenate([a, b]) for a, b in zip(train, dev))
    else:
        train_used = train
    X, y, q = train_used
    params = default_params(args.num_leaves, args.learning_rate, args.min_data_in_leaf, seed=args.seed)
    model = fit(X, y, q, params, args.trees)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save_model(args.out)
    print(f"trained on {len(set(q))} queries ({'train+dev' if args.use_dev else 'train only'}) -> {args.out}")
    print(f"train nDCG@10 = {ndcg10(model, X, y, q):.4f}")
    if dev is not None and not args.use_dev:
        Xd, yd, qd = dev
        print(f"dev   nDCG@10 = {ndcg10(model, Xd, yd, qd):.4f}")


if __name__ == "__main__":
    main()
