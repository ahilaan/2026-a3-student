"""
submission/pointwise_pairwise.py -- the pointwise / pairwise models (assignment Sections 4.1, 4.3 and 7).
Read docs/RANKNET_PRIMER.md first; no prior neural-network experience is assumed.

Both models use the SAME scoring network (harness/neural.py: ScoringNetwork), so the only difference between
them is the loss you write:
  pointwise: squared error between the network's score and the graded label
  pairwise : RankNet -- for a pair (i more relevant than j), loss =
             log(1 + exp(-(s_i - s_j))), i.e. -log sigmoid(s_i - s_j)

PROVIDED : ScoringNetwork, make_scorer, save_scorer, load_scorer (harness/neural.py), dev_ndcg, and the
           overall shape of train_model().
IMPLEMENTED: reproducible seeds, squared error, query-local pair sampling,
            plain RankNet, optimizer steps, minibatching and early stopping.
ALLOWED  : PyTorch tensors, autograd, nn layers, activations, optimizers
           (Adam/SGD), random-seed utilities.
NOT ALLOWED: any ready-made ranking loss or LTR trainer (allRank, TF-Ranking,
           PT-Ranking, RankLib, ...) and LightGBM/XGBoost for these two
           models. torch's MarginRankingLoss is a HINGE loss, not RankNet.

TWO JOBS, KEEP THEM SEPARATE.
  1. Correctness (hand-checked): the functions below are tested by NAME and SIGNATURE (docs/GRADING.md) against
     their textbook definitions. Keep them exactly as declared -- in particular ranknet_loss must stay the plain,
     unweighted RankNet loss.
  2. Track P (leaderboard): your submitted pairwise_model.pt is scored on held-out data. To make it rank better you
     MAY change, in your own extra functions / train_model options: how pairs are sampled (e.g. favour large
     label gaps or top-ranked documents), pairs per query, epochs, learning rate, batch size, the MLP's width and
     depth (at most MAX_HIDDEN_LAYERS hidden layers of at most MAX_WIDTH units: pass e.g. hidden=[128, 64]), and
     per-pair weights such as |delta nDCG| (which turns RankNet into LambdaRank) -- write such weighting as a
     separate function (e.g. weighted_ranknet_loss), never inside ranknet_loss.
     You may NOT: train on anything but relevance labels (no distillation from LambdaMART or any other model's
     scores), or change the artifact format: staff load it with their own copy of harness/neural.py, and an
     artifact that breaks the bounds scores 0 on Track P. Declare the splits you trained and tuned on.

Artifacts to ship in submission/: pointwise_model.pt and pairwise_model.pt (see scripts/train_neural.py).
"""
import copy
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from harness.metrics import mean_ndcg_from_scores
from harness.preprocessing import Standardizer
# PROVIDED, and also what staff use to load your artifacts (assignment Section 7, Track P):
from harness.neural import MAX_HIDDEN_LAYERS, MAX_WIDTH, ScoringNetwork, load_scorer, make_scorer, save_scorer  # noqa: F401


def set_seed(seed: int) -> None:
    """Make training reproducible -- seed Python's `random`, NumPy,
    and torch (torch.manual_seed). Staff check two runs with the same seed
    give identical weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def pointwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predicted scores and graded labels
    (both 1-D tensors of equal length). Returns a scalar tensor.
    Example: scores=[1, 2], labels=[0, 2] -> ((1-0)^2 + (2-2)^2) / 2 = 0.5"""
    return ((scores - labels) ** 2).mean()


def make_pairs(qids, labels, pairs_per_query: int, rng: np.random.RandomState) -> np.ndarray:
    """Build within-query preference pairs.

    qids  : one query id per ROW (any hashable; rows of a query may be anywhere)
    labels: one graded relevance label per row
    Return an int array of shape (P, 2) of ROW INDICES: column 0 is the
    strictly MORE relevant document of the pair, column 1 the less relevant one.
    Rules (all checked by staff): (a) both rows from the SAME query; (b) only
    pairs with different labels (ties carry no preference); (c) oriented
    more-relevant first; (d) at most `pairs_per_query` pairs per query -- sample
    without replacement using `rng` (a NumPy RandomState) so runs are
    reproducible. Never build the pairs of the whole TRAIN split at once.
    Example: qids=[a,a,a,b,b], labels=[2,0,1,1,1], pairs_per_query large ->
    the 3 pairs (0,1), (0,2), (2,1); query b has only ties -> no pairs."""
    labels = np.asarray(labels)
    if len(qids) != len(labels):
        raise ValueError("qids and labels must have the same length")
    if pairs_per_query <= 0:
        return np.empty((0, 2), dtype=np.int64)
    by_query = {}
    for row, qid in enumerate(qids):
        by_query.setdefault(qid, []).append(row)
    sampled = []
    for rows in by_query.values():
        rows = np.asarray(rows, dtype=np.int64)
        buckets = [rows[labels[rows] == grade] for grade in np.unique(labels[rows])]
        # Each higher/lower label block is a Cartesian product, addressed by
        # an integer offset. No quadratic list of document pairs is allocated.
        blocks = [(hi, lo) for i, hi in enumerate(buckets) for lo in buckets[:i]]
        ends = np.cumsum([len(hi) * len(lo) for hi, lo in blocks], dtype=np.int64)
        total = int(ends[-1]) if len(ends) else 0
        count = min(pairs_per_query, total)
        if not count:
            continue
        if count == total:
            offsets = range(total)
        else:
            # Floyd sampling: uniform without replacement, O(count) memory
            # even for a query with millions of possible preference pairs.
            chosen, offsets = set(), []
            for upper in range(total - count, total):
                draw = int(rng.randint(upper + 1))
                value = upper if draw in chosen else draw
                chosen.add(value)
                offsets.append(value)
        for offset in offsets:
            block = int(np.searchsorted(ends, offset, side="right"))
            local = offset - (int(ends[block - 1]) if block else 0)
            hi, lo = blocks[block]
            sampled.append((hi[local // len(lo)], lo[local % len(lo)]))
    return np.asarray(sampled, dtype=np.int64).reshape(-1, 2)


def ranknet_loss(s_preferred: torch.Tensor, s_other: torch.Tensor) -> torch.Tensor:
    """RankNet loss. For each pair, log(1 + exp(-(s_i - s_j))), where
    s_preferred = s_i is the score of the more relevant document. Average over
    the pairs (scalar tensor). Use a numerically stable form -- softplus(-d) or
    torch.logaddexp(0, -d) with d = s_i - s_j -- not a raw exp.
    Example: s_preferred=[2, 0], s_other=[0, 0] ->
    mean(log(1+e^-2), log 2) = mean(0.126928, 0.693147) = 0.410038"""
    return torch.nn.functional.softplus(-(s_preferred - s_other)).mean()


def pointwise_step(network: nn.Module, optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    """ONE optimisation step on a minibatch of documents: zero the
    gradients, forward pass, pointwise_loss, loss.backward(), optimizer.step().
    Return the loss as a float. (Forgetting optimizer.zero_grad() is the classic
    bug: gradients silently accumulate across batches.)"""
    optimizer.zero_grad(set_to_none=True)
    loss = pointwise_loss(network(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def pairwise_step(network: nn.Module, optimizer, x_preferred: torch.Tensor, x_other: torch.Tensor) -> float:
    """ONE optimisation step on a minibatch of PAIRS. x_preferred /
    x_other are (batch, n_features) feature tensors of each pair's more / less
    relevant document. Run both through the SAME network (shared weights),
    ranknet_loss on the two score vectors, backward, step. Return the loss as a
    float. Staff check one step on a pair raises s_preferred - s_other."""
    optimizer.zero_grad(set_to_none=True)
    loss = ranknet_loss(network(x_preferred), network(x_other))
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


# ---- provided helpers -------------------------------------------------------

def dev_ndcg(scorer: Callable[[np.ndarray], np.ndarray], X, y, qids) -> float:
    """PROVIDED. Mean nDCG@10 of `scorer` over the queries of (X, y, qids). It scores
    WHOLE queries and sorts them -- evaluation is not batched like training."""
    return mean_ndcg_from_scores(scorer(X), y, qids, k=10)


# ---- training ---------------------------------------------------------------

def train_model(kind: str, train, dev, *, seed: int = 0, epochs: int = 20, batch_size: int = 1024,
                lr: float = 1e-3, pairs_per_query: int = 300, patience: int = 3, hidden=64,
                verbose: bool = False):
    """Train a 'pointwise' or 'pairwise' scorer. train / dev are (X_raw, y, qids)
    tuples. Returns (network, standardizer, history) with the best-dev-nDCG
    weights restored. Training loss is the example-weighted minibatch mean."""
    if kind not in ("pointwise", "pairwise"):
        raise ValueError("kind must be 'pointwise' or 'pairwise'")
    X, y, q = train
    Xd, yd, qd = dev
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    if len(X) == 0 or len(Xd) == 0:
        raise ValueError("training and validation data must be nonempty")
    if X.ndim != 2 or Xd.ndim != 2 or X.shape[1] != Xd.shape[1]:
        raise ValueError("training and validation feature widths must match")
    if len(y) != len(X) or len(q) != len(X) or len(yd) != len(Xd) or len(qd) != len(Xd):
        raise ValueError("features, labels and query IDs must have matching row counts")
    set_seed(seed)
    # A caller (including another numerical library) can alter OpenMP's current
    # thread count. Inheriting it changes floating-point reduction order and
    # breaks the bitwise same-seed contract. Use one fixed CPU training path.
    torch.set_num_threads(1)
    standardizer = Standardizer().fit(X)              # fit on TRAIN only
    Xt = torch.from_numpy(standardizer.transform(X))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    network = ScoringNetwork(X.shape[1], hidden)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    rng = np.random.RandomState(seed)

    history = []
    best_ndcg, best_state, stale = -float("inf"), None, 0
    scorer = make_scorer(network, standardizer)

    def batched_scorer(raw):
        # Score in chunks, then compute the metric over whole query rankings.
        return np.concatenate([scorer(raw[i:i + batch_size]) for i in range(0, len(raw), batch_size)])

    for epoch in range(1, epochs + 1):
        network.train()
        examples = (make_pairs(q, y, pairs_per_query, rng) if kind == "pairwise"
                    else np.arange(len(X)))
        order = rng.permutation(len(examples))
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            batch = examples[order[start:start + batch_size]]
            if kind == "pairwise":
                loss = pairwise_step(network, optimizer, Xt[batch[:, 0]], Xt[batch[:, 1]])
            else:
                loss = pointwise_step(network, optimizer, Xt[batch], yt[batch])
            loss_sum += loss * len(batch)
        nd = dev_ndcg(batched_scorer, Xd, yd, qd)
        record = {"epoch": epoch, "train_loss": loss_sum / max(1, len(examples)), "dev_ndcg@10": nd}
        history.append(record)
        if verbose:
            print(f"{kind} epoch {epoch}: loss={record['train_loss']:.6f} dev nDCG@10={nd:.6f}")
        if nd > best_ndcg:
            best_ndcg, best_state, stale = nd, copy.deepcopy(network.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    network.load_state_dict(best_state)
    network.eval()
    return network, standardizer, history
