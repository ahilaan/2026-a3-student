"""Fixed-count grouped five-fold CV on released TRAIN only.

Original and augmented copies are created only after each query split.
DEV and TEST are never opened. Checkpoint selection uses completed folds,
not early stopping on a reported validation fold.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np
from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from scripts.compliant_train import prepare_matrix, training_groups
from scripts.cv_query_features import data
from scripts.lambdamart_utils import default_params
from scripts.optimize_rankers import write_json
from scripts.score_push_models import CACHE, CONFIGS, OUT
from submission.query_features import feature_names


def run(args, X, y, q, out=OUT, cache_dir=CACHE):
    counts = sorted(set(args.counts))
    if min(counts) < 1 or not args.name.replace('_', '').isalnum():
        raise ValueError('Positive counts and an alphanumeric/underscore name required')
    signature = {k: getattr(args, k) for k in
                 ('name', 'config', 'augment', 'rank_only', 'seed')}
    signature['counts'] = counts
    out.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = out / (args.name + '.json')
    result = json.loads(path.read_text()) if path.exists() else {
        'signature': signature, 'split': 'Fold1 TRAIN only',
        'excluded_splits': ['DEV', 'TEST'], 'folds': 5, 'fits': []}
    if result['signature'] != signature:
        raise ValueError('Cannot mix different experiments in one CV report')
    width = X.shape[1] // 2
    params = default_params(**CONFIGS[args.config], seed=args.seed,
                            num_threads=args.threads)
    if args.rank_only:
        params['feature_contri'] = [0.] * width + [1.] * width
    result['params'] = params
    model_dir = out / 'models' / args.name
    model_dir.mkdir(parents=True, exist_ok=True)
    folds = grouped_kfold(q, k=5, seed=0)
    for number in args.folds:
        if any(f['fold'] == number for f in result['fits']):
            continue
        training, validation = folds[number]
        trainq, validq = q[training], q[validation]
        assert not np.intersect1d(trainq, validq).size
        trainX, trainy = np.asarray(X[training]), np.asarray(y[training])
        matrix_path = cache_dir / (args.name + f'_fold{number}_X.npy')
        started = time.perf_counter()
        if args.augment:
            matrix = prepare_matrix(trainX, trainq, matrix_path,
                                    augment=True, seed=args.seed)
            fitX, fity = matrix, np.r_[trainy, trainy]
            originalX = matrix[:len(trainy)]
            del trainX
        else:
            fitX, fity = trainX, trainy
            originalX = trainX
        target = model_dir / f'fold{number}.txt'
        def progress(env):
            count = env.iteration + 1
            if count % 100 == 0:
                print('Fold', number, 'trees', count,
                      'seconds', round(time.perf_counter()-started), flush=True)
            if count in counts:
                env.model.save_model(str(model_dir/f'fold{number}_at{count}.txt'))
        model = lgb.train(params, lgb.Dataset(
            fitX, label=fity, group=training_groups(trainq, args.augment),
            feature_name=feature_names(width)),
            num_boost_round=max(counts), callbacks=[progress])
        model.save_model(str(target))
        validX = np.asarray(X[validation])
        metrics = []
        for count in counts:
            metrics.append({'trees': count,
                'train_ndcg': mean_ndcg_from_scores(model.predict(
                    originalX, num_iteration=count, num_threads=args.threads), trainy, trainq),
                'validation_ndcg': mean_ndcg_from_scores(model.predict(
                    validX, num_iteration=count, num_threads=args.threads), y[validation], validq)})
        result['fits'].append({'fold': number, 'train_queries': len(np.unique(trainq)),
            'validation_queries': np.unique(validq).tolist(), 'metrics': metrics,
            'seconds': time.perf_counter()-started,
            'num_threads': args.threads,
            'model': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        write_json(path, result)
        print('Completed fold', number, metrics, flush=True)
        del model, fitX, fity, originalX, validX
        if not args.augment:
            del trainX
        if args.augment:
            matrix._mmap.close()
            del matrix
            if matrix_path.resolve().parent != cache_dir.resolve():
                raise ValueError('Unexpected generated-cache location')
            matrix_path.unlink()
        gc.collect()
    if len(result['fits']) == 5:
        validation_ids = [v for f in result['fits'] for v in f['validation_queries']]
        assert len(validation_ids) == len(set(validation_ids)) == len(np.unique(q))
        summary = []
        for count in counts:
            scores = [next(r for r in f['metrics'] if r['trees'] == count)
                      for f in result['fits']]
            train_mean = float(np.mean([r['train_ndcg'] for r in scores]))
            values = [r['validation_ndcg'] for r in scores]
            summary.append({'trees': count, 'train_mean': train_mean,
                'validation_mean': float(np.mean(values)),
                'validation_std': float(np.std(values)),
                'validation_train_ratio': float(np.mean(values))/train_mean if train_mean else None})
        result['summary'] = summary
        best = max(r['validation_mean'] for r in summary)
        result['selected_trees'] = min(r['trees'] for r in summary
                                      if r['validation_mean'] >= best-1e-4)
        write_json(path, result)
        print(json.dumps({'summary': summary, 'selected_trees': result['selected_trees']}, indent=2), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', required=True)
    ap.add_argument('--config', choices=list(CONFIGS), default='extra127')
    ap.add_argument('--counts', nargs='+', type=int, default=[300, 600, 900, 1200])
    ap.add_argument('--folds', nargs='+', type=int, choices=range(5), default=list(range(5)))
    ap.add_argument('--augment', action='store_true')
    ap.add_argument('--rank-only', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, choices=[1, 2, 3, 4], default=2)
    args = ap.parse_args()
    run(args, *data('train'), out=OUT, cache_dir=CACHE)


if __name__ == '__main__':
    main()
