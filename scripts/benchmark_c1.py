"""Limited, resumable C1 experiments using released TRAIN queries only.

Each reported outer fold has a separate inner stopping split. The supplied toy
sets are mechanical checks, not estimates of hidden benchmark performance.
"""
import argparse
import gc
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
from scripts.lambdamart_utils import default_params, group_sizes
from scripts.ltr_data import load_splits
from scripts.optimize_rankers import cached

CONFIGS = {
    "baseline": {},
    "leaves31": {"num_leaves": 31},
    "regular31": {"num_leaves": 31, "min_data_in_leaf": 100, "lambda_l2": 5,
                  "max_bin": 63, "lambdarank_truncation_level": 13},
    "fast15": {"max_bin": 63, "lambdarank_truncation_level": 20},
    "regular63": {"num_leaves": 63, "min_data_in_leaf": 200, "lambda_l2": 10,
                  "max_bin": 63, "lambdarank_truncation_level": 13},
    "slow15": {"min_data_in_leaf": 100, "lambda_l2": 10, "feature_fraction": .9,
               "learning_rate": .03, "max_bin": 63, "lambdarank_truncation_level": 13},
}
OUT = ROOT / "runs/c1_repro"


def generated(width, queries, density, levels, seed):
    rng = np.random.RandomState(seed)
    sizes = rng.randint(1, 61, queries)
    q = np.repeat(np.arange(queries), sizes)
    X = rng.randn(len(q), width).astype(np.float32)
    X *= rng.rand(*X.shape) < density
    signal = 1.5 * X[:, 0] - X[:, 1] + 0.7 * (X[:, 2] > 0) + 0.8 * rng.randn(len(q))
    y = np.digitize(signal, np.quantile(signal, np.arange(1, levels) / levels)).astype(np.int32)
    # Include entirely uninformative and singleton queries.
    y[q % 19 == 0] = 0
    return X, y, q


def evaluate(name, data, splits, records):
    X, y, q = data
    for fold, (available, outer) in enumerate(splits):
        uq = np.unique(q[available])
        rng = np.random.RandomState(200 + fold)
        stop_queries = rng.permutation(uq)[:max(1, len(uq) // 5)]
        stopping = available[np.isin(q[available], stop_queries)]
        training = available[~np.isin(q[available], stop_queries)]
        for config, extra in CONFIGS.items():
            if any(r["dataset"] == name and r["fold"] == fold and r["config"] == config for r in records):
                continue
            started = time.perf_counter()
            params = default_params(num_threads=4, **extra)
            ds = lgb.Dataset(X[training], label=y[training], group=group_sizes(q[training]))
            dv = lgb.Dataset(X[stopping], label=y[stopping], group=group_sizes(q[stopping]), reference=ds)
            model = lgb.train(params, ds, num_boost_round=800, valid_sets=[dv],
                              callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
            seconds = time.perf_counter() - started
            path = OUT / "benchmark_model.txt"
            model.save_model(str(path))
            loaded = lgb.Booster(model_file=str(path))
            scores = loaded.predict(X[outer], num_threads=4)
            rec = {"dataset": name, "fold": fold, "config": config, "params": params,
                   "train_rows": len(training), "stopping_rows": len(stopping), "outer_rows": len(outer),
                   "features": X.shape[1], "ndcg@10": mean_ndcg_from_scores(scores, y[outer], q[outer]),
                   "train_ndcg@10": mean_ndcg_from_scores(model.predict(X[training], num_threads=4), y[training], q[training]),
                   "fit_seconds": seconds, "best_iteration": model.best_iteration,
                   "loadable": bool(np.isfinite(scores).all()),
                   "matrix_bytes": int(X.nbytes + y.nbytes + q.nbytes)}
            records.append(rec)
            (OUT / "benchmark.json").write_text(json.dumps(records, indent=2))
            print(name, fold, config, round(rec["ndcg@10"], 6), round(seconds, 2), model.best_iteration, flush=True)
            del model, loaded, ds, dv
            gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", choices=["all", "synthetic", "mslr"], default="all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "benchmark.json"
    records = json.loads(path.read_text()) if path.exists() else []
    if args.section in ("all", "synthetic"):
        for d in sorted((ROOT / "data/toy_transfer").iterdir()):
            tr, va = load_splits(d / "train.txt", d / "vali.txt")
            # Retain the supplied validation split only as an outer report.
            X, y, q = (np.concatenate([a, b]) for a, b in zip(tr, va))
            evaluate(d.name, (X, y, q), [(np.arange(len(tr[1])), np.arange(len(tr[1]), len(y)))], records)
        for width, queries, density, levels, seed in [(17, 180, 1., 2, 71), (90, 300, .1, 3, 72), (300, 120, .02, 5, 73)]:
            data = generated(width, queries, density, levels, seed)
            evaluate(f"generated_{width}_{density}", data, grouped_kfold(data[2], 3, 0), records)
    if args.section in ("all", "mslr"):
        X, y, q = cached("train")
        uq = np.unique(q)
        shuffled = np.random.RandomState(19).permutation(uq)
        for n in [300, 900, 2250]:
            rows = np.flatnonzero(np.isin(q, shuffled[:n]))
            subset = X[rows], y[rows], q[rows]
            evaluate(f"MSLR_TRAIN_{n}_queries", subset, grouped_kfold(subset[2], 3, 0), records)
    summary = []
    for dataset in dict.fromkeys(r["dataset"] for r in records):
        for config in CONFIGS:
            values = [r for r in records if r["dataset"] == dataset and r["config"] == config]
            if values:
                scores = [r["ndcg@10"] for r in values]
                summary.append({"dataset": dataset, "config": config, "mean": float(np.mean(scores)),
                                "std": float(np.std(scores)), "seconds_mean": float(np.mean([r["fit_seconds"] for r in values])),
                                "loadable": all(r["loadable"] for r in values)})
    (OUT / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
