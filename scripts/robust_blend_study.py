"""Compare blend weights on queries excluded from both study fits.

Only TRAIN is opened. The reserve is reused development data, not a fresh test.
This is a single-seed robustness proxy for later full-data seed ensembles.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np
from harness.metrics import mean_ndcg_from_scores
from scripts.compliant_train import transform_grouped
from scripts.optimize_rankers import cached
from scripts.score_push_models import OUT


def study_model(receipt):
    record = json.loads(Path(receipt).read_text())
    if (record['args']['mode'] != 'study'
            or record['args']['normalization'] != 'rank'
            or record['training_splits'] != 'Fold1 TRAIN+DEV only'
            or len(record.get('reserved_train_queries', [])) != 600):
        raise ValueError('A rank-feature study excluding 600 TRAIN queries is required')
    path = ROOT / record['model']
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Study model bytes differ from receipt')
    return lgb.Booster(model_file=str(path)), record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--augmented', required=True)
    ap.add_argument('--plain', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--weights', nargs='+', type=float, default=[.6, .7, .8, .9, 1.])
    a = ap.parse_args()
    if not a.name.replace('_', '').isalnum() or not all(0 <= w <= 1 for w in a.weights):
        ap.error('Invalid output name or blend weights')
    target = OUT / (a.name + '.json')
    if target.exists():
        raise FileExistsError(target)
    augmented, ar = study_model(a.augmented)
    plain, pr = study_model(a.plain)
    if set(ar['reserved_train_queries']) != set(pr['reserved_train_queries']):
        raise ValueError('Study models must exclude exactly the same queries')
    X, y, q = cached('train')
    selected = np.isin(q, ar['reserved_train_queries'])
    raw, y, q = np.array(X[selected]), y[selected], q[selected]
    assert len(np.unique(q)) == 600
    rng = np.random.RandomState(123)
    dropped = raw.copy()
    dropped[:, rng.choice(raw.shape[1], 27, replace=False)] = 0
    variants = {'clean': raw,
                'scale': raw * np.exp(rng.normal(0, .5, raw.shape[1])).astype(np.float32),
                'drop20pct': dropped}
    scores = {w: {} for w in a.weights}
    for name, values in variants.items():
        features = transform_grouped(values, q, 'rank')
        aug_scores = augmented.predict(features, num_threads=1)
        plain_scores = plain.predict(features, num_threads=1)
        for weight in a.weights:
            scores[weight][name] = mean_ndcg_from_scores(
                weight * aug_scores + (1-weight) * plain_scores, y, q)
    rows = [{'augmented_weight': w, 'metrics': metrics,
             'mean': float(np.mean(list(metrics.values())))} for w, metrics in scores.items()]
    result = {'args': vars(a), 'evaluation_split': '600 reserved TRAIN queries',
              'TEST_used': False, 'queries': len(np.unique(q)), 'rows': len(y),
              'results': rows, 'highest_proxy_mean': max(rows, key=lambda r: r['mean']),
              'limitation': 'Repeated development comparison; single-seed proxy, not validation of a full-data ensemble or hidden C2.'}
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
