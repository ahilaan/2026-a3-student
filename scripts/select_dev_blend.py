"""Find the most augmented blend reaching a declared trained-on DEV target.

DEV supplied gradients to both fits. These scores are an optimization target,
not a generalization estimate. No TEST data is opened.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
from harness.metrics import mean_ndcg_from_scores
from scripts.compliant_package import verify_model, OUT
from scripts.cv_query_features import data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--augmented', required=True)
    ap.add_argument('--plain', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--target', type=float, default=.5575)
    ap.add_argument('--threads', type=int, choices=[1, 2, 3, 4], default=4)
    ap.add_argument('--weights', nargs='+', type=float,
                    default=[.9, .85, .8, .75, .7, .65, .6])
    a = ap.parse_args()
    if (not a.name.replace('_', '').isalnum() or not 0 <= a.target <= 1
            or not all(0 < w < 1 for w in a.weights)):
        ap.error('Invalid name, target or weights')
    target = OUT / (a.name + '.json')
    if target.exists():
        raise FileExistsError(target)
    sources = [verify_model(p)[0] for p in (a.augmented, a.plain)]
    X, y, q = data('dev')
    models = [lgb.Booster(model_file=str(p)) for p in sources]
    if any(m.num_feature() != X.shape[1] for m in models):
        raise ValueError('Study requires the declared rank-feature schema')
    aug_scores, plain_scores = [m.predict(X, num_threads=a.threads) for m in models]
    rows = [{'augmented_weight': w, 'dev_trained_on_ndcg': mean_ndcg_from_scores(
        w * aug_scores + (1-w) * plain_scores, y, q)} for w in a.weights]
    qualifying = [r for r in rows if r['dev_trained_on_ndcg'] >= a.target]
    selected = max(qualifying, key=lambda r: r['augmented_weight']) if qualifying else None
    result = {'args': vars(a), 'evaluation_split': 'DEV (supplied gradients)',
              'TEST_used': False, 'results': rows, 'selected': selected,
              'selection': 'Largest augmentation weight meeting the fixed DEV target; hidden C2 remains unknown.'}
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
