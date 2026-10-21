"""Label-only neural experiments in the unchanged staff MLP artifact format.

The optional LambdaRank loss is separate from the graded textbook RankNet.
TRAIN supplies gradients and standardization; DEV supplies early stopping.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F

from harness.metrics import mean_ndcg_from_scores
from harness.neural import ScoringNetwork, save_scorer, load_scorer
from harness.preprocessing import Standardizer
from submission.pointwise_pairwise import make_pairs, ranknet_loss, set_seed
from scripts.optimize_rankers import cached, CACHE, RUNS, write_json


def weighted_ranknet_loss(preferred, other, weights):
    """Stable per-pair logistic loss with detached nonnegative nDCG weights."""
    return (F.softplus(-(preferred - other)) * weights).sum() / weights.sum().clamp_min(1e-12)


def lambda_pairs(starts, labels, predictions, cap, rng, weight_power=1.):
    """Sample unequal-label pairs touching current top 20, weighted by |dNDCG@10|.

    A small unweighted component supplies a learning signal below rank 10.
    Query-normalized weights keep queries comparable rather than emphasizing
    queries with more documents. Only relevance labels supervise learning.
    An optional power below one smooths extreme weights before restoring each
    query's mean weight; the default preserves the original weighting exactly.
    """
    if not 0 <= weight_power <= 1:
        raise ValueError('pair-weight power must be between zero and one')
    pairs, weights = [], []
    for a, b in zip(starts[:-1], starts[1:]):
        y = labels[a:b]
        n = len(y)
        order = np.argsort(-predictions[a:b], kind="stable")
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)
        hi, lo = np.nonzero((y[:, None] > y[None, :]) &
                            ((ranks[:, None] < 20) | (ranks[None, :] < 20)))
        if len(hi) == 0:
            continue
        if len(hi) > cap:
            chosen = rng.choice(len(hi), cap, replace=False)
            hi, lo = hi[chosen], lo[chosen]
        gains = 2.0 ** y - 1
        discount = np.where(ranks < 10, 1 / np.log2(ranks + 2), 0)
        ideal = (np.sort(gains)[::-1][:10] / np.log2(np.arange(2, min(n, 10) + 2))).sum()
        delta = np.abs((gains[hi] - gains[lo]) * (discount[hi] - discount[lo])) / max(ideal, 1e-12)
        scaled = delta / max(delta.mean(), 1e-12)
        if weight_power != 1:
            scaled = scaled ** weight_power
            scaled = scaled / max(scaled.mean(), 1e-12)
        weight = scaled + 0.1
        pairs.append(np.column_stack((hi + a, lo + a)))
        weights.append(weight.astype(np.float32))
    return np.concatenate(pairs), np.concatenate(weights)


def train(args):
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    X, y, q = cached("train")
    Xd, yd, qd = cached("dev")
    std = Standardizer().fit(X)
    Xt = torch.from_numpy(std.transform(X))
    Xdt = torch.from_numpy(std.transform(Xd))
    network = ScoringNetwork(X.shape[1], args.hidden)
    if args.init:
        blob = torch.load(args.init, weights_only=True)
        if blob["config"] != {"n_features": X.shape[1], "hidden": args.hidden}:
            raise ValueError("warm-start architecture does not match")
        if blob["standardizer"] != std.to_dict():
            raise ValueError("warm-start standardization differs from TRAIN")
        network.load_state_dict(blob["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.RandomState(args.seed)
    starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1, len(q)]
    target = CACHE / args.name
    target.mkdir(parents=True, exist_ok=True)
    result = {"args": vars(args), "gradients": "Fold1 TRAIN relevance labels only",
              "standardizer": "TRAIN only", "selection": "best DEV nDCG@10 epoch", "history": []}
    best, stale, state = -1, 0, None

    def predict(tensor):
        network.eval()
        with torch.no_grad():
            return np.concatenate([network(tensor[i:i + 4096]).numpy() for i in range(0, len(tensor), 4096)])

    if args.init:
        best = mean_ndcg_from_scores(predict(Xdt), yd, qd)
        state = copy.deepcopy(network.state_dict())
        result.update(best_epoch=0, **{"best_dev_ndcg@10": best})
        save_scorer(str(target / "pairwise_model.pt"), network, std)

    for epoch in range(1, args.epochs + 1):
        if args.loss == "lambda":
            pairs, weights = lambda_pairs(starts, y, predict(Xt), args.pairs, rng)
        else:
            pairs = make_pairs(q, y, args.pairs, rng)
            weights = np.ones(len(pairs), np.float32)
        network.train()
        total = 0
        order = rng.permutation(len(pairs))
        for start in range(0, len(order), args.batch_size):
            ix = order[start:start + args.batch_size]
            pair = pairs[ix]
            xhi, xlo = Xt[pair[:, 0]], Xt[pair[:, 1]]
            if args.input_dropout:
                xhi = F.dropout(xhi, args.input_dropout, training=True)
                xlo = F.dropout(xlo, args.input_dropout, training=True)
            optimizer.zero_grad(set_to_none=True)
            shi, slo = network(xhi), network(xlo)
            loss = (weighted_ranknet_loss(shi, slo, torch.from_numpy(weights[ix]))
                    if args.loss == "lambda" else ranknet_loss(shi, slo))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 10)
            optimizer.step()
            total += loss.item() * len(ix)
        ndcg = mean_ndcg_from_scores(predict(Xdt), yd, qd)
        record = {"epoch": epoch, "train_loss": total / len(pairs), "dev_ndcg@10": ndcg,
                  "pairs": len(pairs)}
        result["history"].append(record)
        if ndcg > best:
            best, stale, state = ndcg, 0, copy.deepcopy(network.state_dict())
            result["best_epoch"] = epoch
            result["best_dev_ndcg@10"] = best
            save_scorer(str(target / "pairwise_model.pt"), network, std)
        else:
            stale += 1
        write_json(RUNS / (args.name + ".json"), result)
        print(args.name, record, "best", best, flush=True)
        if stale >= args.patience:
            break
    network.load_state_dict(state)
    scorer = load_scorer(str(target / "pairwise_model.pt"))
    scores = np.concatenate([scorer(Xd[i:i + 4096]) for i in range(0, len(Xd), 4096)])
    result["reloaded_dev_ndcg@10"] = mean_ndcg_from_scores(scores, yd, qd)
    write_json(RUNS / (args.name + ".json"), result)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--loss", choices=["ranknet", "lambda"], default="lambda")
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=0.0001)
    ap.add_argument("--input-dropout", type=float, default=0)
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--init", help="optional prior label-trained scorer of the same architecture")
    train(ap.parse_args())
