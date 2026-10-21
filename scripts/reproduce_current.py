"""Exact staged current-model workflow, invoked through train_model.py.

Fresh fitting and retained-component assembly are explicitly separate claims.
All generated artifacts live outside submission/. No packaging or submission.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plan(work):
    """Preserve the original 1200-tree save/reload boundaries and RNG resets."""
    commands = []
    for name, config, seed, total in [('extra127', 'extra127', 0, 2000),
                                      ('extra63', 'extra63', 23, 2400)]:
        for rounds in [1200, total]:
            cmd = [sys.executable, 'scripts/compliant_train.py', 'full',
                   '--name', f'{name}_{rounds}', '--config', config, '--rounds', str(rounds),
                   '--normalization', 'rank', '--seed', str(seed), '--threads', '4', '--augment',
                   '--output-dir', str(work), '--cache-dir', str(work / 'cache')]
            if rounds != 1200:
                cmd += ['--init-receipt', str(work / f'{name}_1200.json')]
            commands.append(cmd)
    commands.append([sys.executable, 'scripts/compliant_blend.py',
                     '--first', str(work / 'extra127_2000.json'),
                     '--second', str(work / 'extra63_2400.json'),
                     '--first-weight', '0.65', '--name', 'current_main', '--output-dir', str(work)])
    return commands


def reproduce(args):
    import lightgbm as lgb
    import numpy as np
    from scripts.compliant_package import verify_model
    from scripts.optimize_rankers import cached
    from submission.query_features import augment_grouped, feature_names

    out, work = Path(args.out).resolve(), Path(args.work_dir).resolve()
    report_target = Path(args.report_out).resolve() if args.report_out else work/'verification.json'
    if out.is_relative_to(ROOT / 'submission') or not work.is_relative_to(ROOT / 'runs'):
        raise ValueError('Use a separate output and a work directory under runs/; submission/ is protected')
    if report_target.is_relative_to(ROOT/'submission') or report_target == out:
        raise ValueError('Verification report must be separate from scoring artifacts')
    if out.exists():
        raise FileExistsError(out)
    if not args.use_dev or not args.dev:
        raise ValueError('current-main requires --train data/train --dev data/dev --use-dev')
    if Path(args.train).resolve() not in [ROOT/'data/train', ROOT/'data/train/train.txt'] or Path(args.dev).resolve() not in [ROOT/'data/dev', ROOT/'data/dev/vali.txt']:
        raise ValueError('Current-main reproduction uses exactly the released Fold1 TRAIN/DEV inputs')
    if any(v is not None for v in [args.trees, args.num_leaves, args.min_data_in_leaf, args.query_features]) or args.seed != 0 or args.learning_rate != .05:
        raise ValueError('Current-main fixes component settings; use the legacy modes for experiments')
    commands = plan(work)
    if args.dry_run:
        print(json.dumps({'commands': commands, 'output': str(out)}, indent=2))
        return
    started = time.perf_counter()
    protected = digest(ROOT/'submission/model.txt')
    work.mkdir(parents=True, exist_ok=True)
    if args.assembly_only:
        current = json.loads((ROOT/'docs/CURRENT_MODEL.json').read_text())
        first, second = [verify_model(ROOT/p)[0] for p in current['constituent_receipts']]
        command = commands[-1]
        command[command.index('--first')+1] = str(ROOT/current['constituent_receipts'][0])
        command[command.index('--second')+1] = str(ROOT/current['constituent_receipts'][1])
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        # Audit raw/relative caches against the allowlisted source files first.
        subprocess.run([sys.executable, 'scripts/check_training_inputs.py',
                        '--out', str(work/'input_audit.json')], cwd=ROOT, check=True)
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)
    result_model = work/'models/current_main.txt'
    model = lgb.Booster(model_file=str(result_model))
    reference = lgb.Booster(model_file=str(ROOT/'submission/model.txt'))
    # Complete queries from authorized TRAIN; inference transforms are shared.
    X, _, q = cached('train')
    sample = np.isin(q, np.unique(q)[:20])
    inputs = augment_grouped(X[sample], q[sample])
    actual, expected = model.predict(inputs, num_threads=4), reference.predict(inputs, num_threads=4)
    ranking_equal = all(np.array_equal(np.argsort(-actual[q[sample] == query], kind='stable'),
                                      np.argsort(-expected[q[sample] == query], kind='stable'))
                        for query in np.unique(q[sample]))
    report = {'mode': 'retained-component assembly only' if args.assembly_only else 'fresh staged refit',
              'sha256': digest(result_model), 'reference_sha256': protected,
              'byte_equal': digest(result_model) == protected,
              'prediction_equal': bool(np.array_equal(actual, expected)),
              'maximum_prediction_difference': float(np.max(np.abs(actual-expected))),
              'ranking_equal': ranking_equal, 'comparison_queries': 20,
              'trees': model.num_trees(), 'features': model.num_feature(),
              'schema_equal': model.feature_name() == feature_names(136),
              'lightgbm': lgb.__version__, 'seconds': time.perf_counter()-started,
              'commands': commands, 'output': str(out)}
    if digest(ROOT/'submission/model.txt') != protected:
        raise RuntimeError('Protected model changed during reproduction')
    out.parent.mkdir(parents=True, exist_ok=True)
    if result_model.resolve() != out:
        shutil.copy2(result_model, out)
    target = report_target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
