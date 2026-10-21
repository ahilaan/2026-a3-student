"""Relevance-label-only RankNet refinement using extra declared DEV training queries.

Study reserves 400 DEV queries from gradients. Warm starts have used
all DEV for earlier epoch selection. Fresh starts retain only TRAIN statistics.
Repeated selection is development, not a fresh final test.
Final refit uses TRAIN+DEV only.

"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from harness.metrics import mean_ndcg_from_scores
from harness.neural import ScoringNetwork, save_scorer
from harness.preprocessing import Standardizer
from scripts.optimize_rankers import cached
from scripts.optimize_neural import lambda_pairs, weighted_ranknet_loss
from submission.pointwise_pairwise import set_seed


def standardized_tensor(std, *arrays, chunk_rows=8192):
    """Preserve row order while bounding float64 standardization temporaries."""
    width = arrays[0].shape[1]
    if any(x.ndim != 2 or x.shape[1] != width for x in arrays):
        raise ValueError('Feature matrices must have the same width')
    result = torch.empty((sum(len(x) for x in arrays), width), dtype=torch.float32)
    offset = 0
    for x in arrays:
        for start in range(0, len(x), chunk_rows):
            stop = min(start + chunk_rows, len(x))
            result[offset+start:offset+stop].copy_(torch.from_numpy(std.transform(x[start:stop])))
        offset += len(x)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', required=True); ap.add_argument('--full', action='store_true')
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--lr', type=float, default=.0001)
    ap.add_argument('--dropout', type=float, default=.03)
    ap.add_argument('--fresh', action='store_true', help='Random initialization; retain original TRAIN standardization')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--swa-start', type=int, default=0, help='Also evaluate running parameter averages from this epoch')
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--weight-power',type=float,default=1.,help='Temper nDCG pair weights; 1 preserves the original objective')
    ap.add_argument('--init', help='Optional label-trained staff-compatible warm-start artifact')
    ap.add_argument('--hidden', type=int, nargs='+', help='Fresh model architecture, up to 3 hidden layers / 256 units')
    a = ap.parse_args()
    if not 0 <= a.weight_power <= 1:
        ap.error('--weight-power must be in [0,1]')
    set_seed(a.seed); torch.set_num_threads(2)
    out = ROOT / 'runs/compliant_campaign'; target = out / 'models' / a.name
    target.mkdir(parents=True, exist_ok=False)
    initial_path = Path(a.init) if a.init else ROOT / 'submission/pairwise_model.pt'
    blob = torch.load(initial_path, weights_only=True)
    std = Standardizer.from_dict(blob['standardizer'])
    hidden = a.hidden or blob['config']['hidden']
    if a.hidden and not a.fresh:
        raise ValueError('--hidden requires --fresh')
    if not 1 <= len(hidden) <= 3 or not all(1 <= n <= 256 for n in hidden):
        raise ValueError('Architecture exceeds the assignment MLP bounds')
    network = ScoringNetwork(136, hidden)
    if not a.fresh:
        network.load_state_dict(blob['state_dict'], strict=True)
    X, y, q = cached('train'); Xd, yd, qd = cached('dev')
    reserved = np.random.RandomState(764).permutation(np.unique(qd))[:400]
    validation = np.isin(qd, reserved)
    keep = np.ones(len(qd), bool) if a.full else ~validation
    trainy = np.r_[y, yd[keep]]; trainq = np.r_[q, qd[keep]]
    trainX = standardized_tensor(std, X, Xd[keep])
    validX = standardized_tensor(std, Xd[validation]); validy = yd[validation]; validq = qd[validation]
    starts = np.r_[0, np.flatnonzero(trainq[1:] != trainq[:-1]) + 1, len(trainq)]
    optimizer = torch.optim.AdamW(network.parameters(), lr=a.lr, weight_decay=.001)
    rng = np.random.RandomState(a.seed)
    def predict(tensor):
        network.eval()
        with torch.no_grad():
            return np.concatenate([network(tensor[i:i+4096]).numpy() for i in range(0, len(tensor), 4096)])
    initial = mean_ndcg_from_scores(predict(validX), validy, validq)
    warning = ('Final refit trains on all DEV; the reported DEV subset supplies gradients.' if a.full else
               '400 DEV queries have no study gradients; repeated development selection is not a fresh final test.' if a.fresh else
               '400 DEV queries have no study gradients, but the warm-start artifact used all DEV for prior epoch selection.')
    result = {'args': vars(a), 'initial_reserved_dev': initial, 'selection_warning': warning,
              'gradients': 'TRAIN+DEV' if a.full else 'TRAIN+1600 DEV queries',
              'training_rows':len(trainy), 'training_queries':len(np.unique(trainq)),
              'reported_dev_supplied_gradients':bool(a.full),
              'standardization': 'Frozen original TRAIN-only statistics',
              'TEST_used': False, 'history': [], 'best_epoch': 0, 'best_reserved_dev': initial}
    if not a.fresh:
        result['initial_artifact_sha256'] = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    best = initial; stale = 0; begin = time.perf_counter()
    averaged = None; average_count = 0
    save_scorer(str(target / 'pairwise_model.pt'), network, std)
    for epoch in range(1, a.epochs+1):
        pairs, weights = lambda_pairs(starts, trainy, predict(trainX), 400, rng, weight_power=a.weight_power)
        network.train(); order = rng.permutation(len(pairs)); total = 0.
        for start in range(0, len(order), 2048):
            ix = order[start:start+2048]; pair = pairs[ix]
            hi = F.dropout(trainX[pair[:, 0]], a.dropout, training=True)
            lo = F.dropout(trainX[pair[:, 1]], a.dropout, training=True)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_ranknet_loss(network(hi), network(lo), torch.from_numpy(weights[ix]))
            loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(), 10); optimizer.step()
            total += loss.item()*len(ix)
        metric = mean_ndcg_from_scores(predict(validX), validy, validq)
        record = {'epoch': epoch, 'reserved_dev': metric, 'loss': total/len(pairs), 'seconds': time.perf_counter()-begin}
        selected_metric, kind = metric, 'epoch'
        current = None
        if a.swa_start and epoch >= a.swa_start:
            current = {k:v.detach().clone() for k,v in network.state_dict().items()}
            average_count += 1
            if averaged is None: averaged = copy.deepcopy(current)
            else:
                for key in averaged: averaged[key].add_((current[key]-averaged[key])/average_count)
            network.load_state_dict(averaged)
            average_metric = mean_ndcg_from_scores(predict(validX), validy, validq)
            record['averaged_reserved_dev'] = average_metric
            network.load_state_dict(current)
            if average_metric > selected_metric: selected_metric, kind = average_metric, 'swa'
        result['history'].append(record)
        if selected_metric > best:
            best = selected_metric; stale = 0; result.update(best_epoch=epoch, best_reserved_dev=best, best_kind=kind)
            if not a.full:
                if kind == 'swa': network.load_state_dict(averaged)
                save_scorer(str(target / 'pairwise_model.pt'), network, std)
                if kind == 'swa': network.load_state_dict(current)
        else: stale += 1
        if a.full:
            if averaged is not None: network.load_state_dict(averaged)
            save_scorer(str(target / 'pairwise_model.pt'), network, std)
            if averaged is not None: network.load_state_dict(current)
        (out / (a.name+'.json')).write_text(json.dumps(result, indent=2))
        print(record, flush=True)
        if not a.full and stale >= a.patience: break
    if a.full: result['saved_fixed_epoch'] = len(result['history'])
    model_path = target / 'pairwise_model.pt'
    result.update(model=model_path.relative_to(ROOT).as_posix(),
                  sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  hidden=network.hidden_sizes,
                  supervision='Relevance labels only; no teacher scores',
                  standardizer_sha256=hashlib.sha256(json.dumps(blob['standardizer'],sort_keys=True).encode()).hexdigest())
    (out / (a.name+'.json')).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
