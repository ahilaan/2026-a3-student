"""TRAIN+DEV refits and robustness studies reserving TRAIN queries before fitting.

Only allowlisted released TRAIN/DEV loaders are used. Reserved study queries
never supply gradients. Historical main-model checkpoints/caches are not inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np
from harness.metrics import mean_ndcg_from_scores
from scripts.cv_query_features import combined_data, data
from scripts.lambdamart_utils import default_params, group_sizes
from scripts.optimize_rankers import cached
from scripts.score_push_models import CONFIGS, OUT, CACHE
from submission.ranking import _augment_query, _augment_query_minmax
from submission.query_features import feature_names


def transform_grouped(raw, q, kind):
    transform = _augment_query if kind == 'rank' else _augment_query_minmax
    starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1, len(q)]
    return np.concatenate([transform(raw[a:b]) for a, b in zip(starts[:-1], starts[1:])])


def training_groups(q, augmented=False):
    groups = group_sizes(q)
    # Rows contain all original queries followed by all augmented queries.
    # Repeat the list of group lengths; do not multiply individual lengths.
    return groups + groups if augmented else groups


def reproduction_command(args):
    command = (f"python scripts/compliant_train.py {args['mode']} --name {args['name']}"
               f" --config {args['config']} --rounds {args['rounds']}"
               f" --normalization {args['normalization']} --seed {args['seed']}"
               f" --threads {args.get('threads', 4)}")
    for key in ('augment', 'group_dropout', 'rank_only'):
        if args[key]:
            command += ' --' + key.replace('_', '-')
    if args.get('init_receipt'):
        command += ' --init-receipt ' + args['init_receipt']
    if args.get('checkpoints'):
        command += ' --checkpoints ' + ' '.join(map(str, args['checkpoints']))
    if args.get('output_dir'):
        import shlex
        command += ' --output-dir ' + shlex.quote(str(args['output_dir']))
        command += ' --cache-dir ' + shlex.quote(str(args['cache_dir']))
    return command


def continuation_model(receipt, args, names):
    """Continue only a verified full fit with identical data construction."""
    from scripts.compliant_package import verify_model
    path, record = verify_model(receipt)
    if record.get('constituent_receipts'):
        raise ValueError('Continuation requires a single fitted model')
    for key in ('mode', 'config', 'normalization', 'augment', 'group_dropout', 'rank_only', 'seed'):
        if record['args'].get(key) != getattr(args, key):
            raise ValueError('Continuation configuration mismatch: ' + key)
    if args.mode != 'full':
        raise ValueError('Continuation supports full fits only')
    model = lgb.Booster(model_file=str(path))
    if model.feature_name() != names or model.num_trees() != record['args']['rounds']:
        raise ValueError('Continuation schema or tree count mismatch')
    if args.rounds <= model.num_trees():
        raise ValueError('Continuation total rounds must exceed the parent count')
    return model, record


def prepare_matrix(X, q, cache, *, normalization='rank', augment=False,
                   seed=0, group_dropout=False):
    """Construct original and augmented blocks after the caller excludes queries."""
    rows, width = X.shape
    raw_width = width // 2
    m = np.lib.format.open_memmap(cache, mode='w+', dtype=np.float32,
                                 shape=(rows * (2 if augment else 1), width))
    transform = _augment_query if normalization == 'rank' else _augment_query_minmax
    starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1, rows]
    rng = np.random.RandomState(76420 + seed)
    for i, (lo, hi) in enumerate(zip(starts[:-1], starts[1:])):
        m[lo:hi] = X[lo:hi] if normalization == 'rank' else transform(X[lo:hi, :raw_width])
        if augment:
            raw = np.array(X[lo:hi, :raw_width], dtype=np.float64)
            raw += rng.normal(size=raw.shape) * (.02 * np.ptp(raw, axis=0))
            raw *= np.exp(rng.normal(0, .5, raw_width))
            raw[:, rng.choice(raw_width, rng.randint(14, min(41, raw_width+1)), replace=False)] = 0
            if group_dropout:
                if raw_width != 136:
                    raise ValueError('Feature-family augmentation requires the declared MSLR schema')
                if rng.rand() < .5:
                    families = [np.arange(133,136), np.arange(127,133), np.arange(105,125)]
                    raw[:, families[rng.randint(3)]] = 0
            m[rows+lo:rows+hi] = transform(raw.astype(np.float32))
        if i % 2000 == 0:
            print('Prepared query', i, flush=True)
    m.flush()
    return m


def main():
    global OUT, CACHE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['full', 'study'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--config', choices=list(CONFIGS), default='extra63')
    ap.add_argument('--rounds', type=int, default=600)
    ap.add_argument('--normalization', choices=['rank', 'minmax'], default='rank')
    ap.add_argument('--augment', action='store_true')
    ap.add_argument('--group-dropout', action='store_true')
    ap.add_argument('--rank-only', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, choices=[1,2,3,4], default=4)
    ap.add_argument('--checkpoints', nargs='*', type=int, default=[])
    ap.add_argument('--report-fit-dev', action='store_true',
                    help='Also report trained-on DEV for a reserve study; never an unseen estimate')
    ap.add_argument('--keep-matrix-cache', action='store_true', help='Retain the rebuildable augmentation matrix after a successful fit')
    ap.add_argument('--init-receipt', help='Verified full-fit receipt to extend; rounds is the total count')
    ap.add_argument('--output-dir', type=Path, default=OUT,
                    help='Separate experiment directory inside the repository')
    ap.add_argument('--cache-dir', type=Path, default=CACHE)
    args = ap.parse_args()
    OUT, CACHE = args.output_dir.resolve(), args.cache_dir.resolve()
    if not OUT.is_relative_to(ROOT) or not CACHE.is_relative_to(ROOT):
        ap.error('Output and cache directories must be inside the repository')
    # Paths in JSON receipts must stay JSON serializable.
    args.output_dir, args.cache_dir = str(OUT), str(CACHE)
    if args.rounds < 1 or not args.name.replace('_','').isalnum():
        ap.error('positive rounds and an alphanumeric/underscore name required')
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    target = OUT / 'models' / (args.name + '.txt')
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    start = time.perf_counter()
    X, y, q = combined_data()
    reserved = None
    if args.mode == 'study':
        _, _, trainq = cached('train')
        reserved = np.random.RandomState(76421).permutation(np.unique(trainq))[:600]
        valid = np.isin(q, reserved)
        validation = (np.array(X[valid, :X.shape[1]//2]), y[valid], q[valid])
        X, y, q = X[~valid], y[~valid], q[~valid]
        assert not np.intersect1d(q, reserved).size
    rows, width = X.shape; raw_width = width // 2
    names = feature_names(raw_width) if args.normalization == 'rank' else [f'raw_{i}' for i in range(raw_width)] + [f'qmin_{i}' for i in range(raw_width)]
    initial, parent = continuation_model(args.init_receipt, args, names) if args.init_receipt else (None, None)
    initial_trees = initial.num_trees() if initial is not None else 0
    if args.augment or args.normalization == 'minmax':
        cache = CACHE / (args.name + '_X.npy')
        X = prepare_matrix(X, q, cache, normalization=args.normalization,
                           augment=args.augment, seed=args.seed,
                           group_dropout=args.group_dropout)
    groups = training_groups(q, args.augment)
    if args.augment:
        y = np.r_[y,y]
    params = default_params(**CONFIGS[args.config], seed=args.seed, num_threads=args.threads)
    if args.rank_only:
        params['feature_contri'] = [0.] * raw_width + [1.] * raw_width
    print('Training', args.name, X.shape, 'groups', len(groups), flush=True)
    def checkpoint(env):
        count = env.iteration + 1
        if count % 100 == 0:
            print('Trees', count, 'seconds', round(time.perf_counter()-start), flush=True)
        if count in args.checkpoints:
            env.model.save_model(str(target.with_name(args.name + f'_at{count}.txt')))
    model = lgb.train(params, lgb.Dataset(X, label=y, group=groups, feature_name=names),
                      num_boost_round=args.rounds-initial_trees, init_model=initial, callbacks=[checkpoint])
    model.save_model(str(target))
    result = {'args':vars(args), 'training_splits':'Fold1 TRAIN+DEV only',
              'excluded_splits':['Fold1 TEST'], 'original_rows':rows,
              'original_queries':len(np.unique(q)), 'fit_rows':len(y),
              'params':params, 'model':target.relative_to(ROOT).as_posix(),
              'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'seconds':time.perf_counter()-start, 'checkpoints':[]}
    if parent is not None:
        result['continued_from'] = Path(args.init_receipt).resolve().relative_to(ROOT).as_posix()
        result['initial_trees'] = initial_trees
        result['reproduction'] = parent.get('reproduction', [reproduction_command(parent['args'])]) + [reproduction_command(vars(args))]
    if reserved is not None:
        result['reserved_train_queries'] = reserved.tolist()
        raw, vy, vq = validation
        rng = np.random.RandomState(123)
        dropped = raw.copy(); dropped[:,rng.choice(raw_width,27,replace=False)] = 0
        variants = {'clean':raw, 'scale':raw*np.exp(rng.normal(0,.5,raw_width)).astype(np.float32), 'drop20pct':dropped}
        variants = {n:transform_grouped(v,vq,args.normalization) for n,v in variants.items()}
        for count in sorted(set(args.checkpoints + [args.rounds])):
            metrics = {n:mean_ndcg_from_scores(model.predict(v,num_iteration=count,num_threads=4),vy,vq) for n,v in variants.items()}
            result['checkpoints'].append({'trees':count,'metrics':metrics,'mean':float(np.mean(list(metrics.values())))})
    else:
        xd, yd, qd = data('dev')
        if args.normalization == 'minmax':
            xd = transform_grouped(xd[:,:raw_width],qd,'minmax')
        result['dev_trained_on_ndcg'] = mean_ndcg_from_scores(model.predict(xd,num_threads=4),yd,qd)
        result['dev_fixed_counts'] = [{'trees': count, 'ndcg':
            result['dev_trained_on_ndcg'] if count == args.rounds else
            mean_ndcg_from_scores(model.predict(xd, num_iteration=count, num_threads=4), yd, qd)}
            for count in sorted(set(args.checkpoints + [args.rounds]))]
        result['evaluation_warning'] = 'DEV supplies gradients; this is a trained-on measurement.'
    if reserved is not None and args.report_fit_dev:
        xd, yd, qd = data('dev')
        if args.normalization == 'minmax':
            xd = transform_grouped(xd[:,:raw_width], qd, 'minmax')
        result['dev_fixed_counts'] = [{'trees': count, 'ndcg': mean_ndcg_from_scores(
            model.predict(xd, num_iteration=count, num_threads=args.threads), yd, qd)}
            for count in sorted(set(args.checkpoints + [args.rounds]))]
        result['dev_trained_on_ndcg'] = result['dev_fixed_counts'][-1]['ndcg']
        result['evaluation_warning'] = 'Study fits all DEV queries; these DEV scores are trained-on, not validation.'
    (OUT / (args.name + '.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    if (args.augment or args.normalization == 'minmax') and not args.keep_matrix_cache:
        X._mmap.close()
        if cache.resolve().parent != CACHE.resolve():
            raise ValueError('Unexpected matrix-cache location')
        cache.unlink()
        print('Removed rebuildable matrix cache:', cache.name, flush=True)


if __name__ == '__main__':
    main()
