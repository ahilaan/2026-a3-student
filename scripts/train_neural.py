"""Train the pointwise and/or pairwise (RankNet) MLP scorers and save pointwise_model.pt / pairwise_model.pt.

    python scripts/train_neural.py --kind both --train data/train --dev data/dev --out-dir submission

Calls train_model() from submission/pointwise_pairwise.py. Early stopping on --dev counts as tuning against dev
(it makes your dev score optimistic); declare that in your submission notes (Section 10).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import json

from scripts.ltr_data import load_splits
from submission.pointwise_pairwise import dev_ndcg, make_scorer, save_scorer, train_model


def _resolve(path, name):
    return os.path.join(path, name) if os.path.isdir(path) else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["pointwise", "pairwise", "both"], default="both")
    ap.add_argument("--train", default="data/toy")
    ap.add_argument("--dev", default="data/toy")
    ap.add_argument("--out-dir", default="submission")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pairs-per-query", type=int, default=300)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64],
                    help="hidden layer widths, e.g. --hidden 128 64 (at most 3 layers of at most 256 units)")
    args = ap.parse_args()
    train, dev = load_splits(_resolve(args.train, "train.txt"), _resolve(args.dev, "vali.txt"))
    os.makedirs(args.out_dir, exist_ok=True)
    for kind in (["pointwise", "pairwise"] if args.kind == "both" else [args.kind]):
        net, std, hist = train_model(kind, train, dev, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                                     lr=args.lr, pairs_per_query=args.pairs_per_query, patience=args.patience,
                                     hidden=args.hidden, verbose=True)
        path = os.path.join(args.out_dir, f"{kind}_model.pt")
        save_scorer(path, net, std, args.hidden)
        with open(os.path.join(args.out_dir, f"{kind}_history.json"), "w", encoding="utf-8") as f:
            json.dump({"options": vars(args), "history": hist}, f, indent=2)
        print(f"{kind}: epochs run = {len(hist)}, final dev nDCG@10 = {dev_ndcg(make_scorer(net, std), *dev):.4f} -> {path}")


if __name__ == "__main__":
    main()
