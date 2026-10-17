"""Compare a small fixed set of tree averages using five TRAIN query folds.

Recreates missing screening checkpoints deterministically. No DEV file or
held-out partition is used. No early stopping uses a reported fold.
"""
import sys
import argparse
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np

from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from scripts.lambdamart_utils import default_params, group_sizes
from scripts.optimize_rankers import cached, CACHE, RUNS, CONFIGS, write_json


def main(short=False):
    X, y, q = cached("train")
    folds = grouped_kfold(q, k=5, seed=0)
    folder = CACHE / "cv_checkpoints"
    folder.mkdir(exist_ok=True)
    repairs = []
    # Only the first screening fold predated checkpoint persistence.
    for name in ("leaves15", "leaves31"):
        path = folder / f"{name}_fold0.txt"
        if not path.exists():
            tr, va = folds[0]
            params = default_params(**CONFIGS[name], num_threads=4)
            started = time.perf_counter()
            print("Recreating fixed-count checkpoint:", name, 0, flush=True)
            model = lgb.train(params, lgb.Dataset(X[tr], label=y[tr], group=group_sizes(q[tr])),
                              num_boost_round=800)
            model.save_model(str(path))
            repairs.append({"config": name, "fold": 0, "params": params,
                            "seconds": time.perf_counter() - started})
            del model
    configs = {
        "15_800": [("leaves15", 800, 1.0)],
        "31_400": [("leaves31", 400, 1.0)],
        "31_800": [("leaves31", 800, 1.0)],
        "mix15_800_31_400": [("leaves15", 800, 0.5), ("leaves31", 400, 0.5)],
        "mix15_800_31_800": [("leaves15", 800, 0.5), ("leaves31", 800, 0.5)],
    }
    if short:
        configs = {
            "15_100": [("leaves15", 100, 1.0)],
            "15_200": [("leaves15", 200, 1.0)],
            "31_100": [("leaves31", 100, 1.0)],
            "31_200": [("leaves31", 200, 1.0)],
            "mix15_100_31_100": [("leaves15", 100, 0.5), ("leaves31", 100, 0.5)],
            "mix15_200_31_200": [("leaves15", 200, 0.5), ("leaves31", 200, 0.5)],
            "mix15_100_31_200": [("leaves15", 100, 0.5), ("leaves31", 200, 0.5)],
        }
    report_path = RUNS / ("short_blend_cv.json" if short else "blend_cv.json")
    result = {"split": "Fold1 TRAIN only", "folds": 5, "seed": 0,
              "selection": "fixed-count query-grouped CV; no early stopping", "repairs": repairs,
              "configs": configs, "scores": []}
    for f, (tr, va) in enumerate(folds):
        models = {name: lgb.Booster(model_file=str(folder / f"{name}_fold{f}.txt"))
                  for name in ("leaves15", "leaves31")}
        record = {"fold": f, "train": {}, "validation": {}}
        for split, rows in (("train", tr), ("validation", va)):
            requested = {(source, count) for pieces in configs.values() for source, count, weight in pieces}
            predictions = {(name, count): models[name].predict(X[rows], num_iteration=count, num_threads=4)
                           for name, count in sorted(requested)}
            for name, pieces in configs.items():
                score = sum(weight * predictions[(source, count)] for source, count, weight in pieces)
                record[split][name] = mean_ndcg_from_scores(score, y[rows], q[rows])
        result["scores"].append(record)
        write_json(report_path, result)
        print("Blend CV", record, flush=True)
    summary = []
    for name in configs:
        train = np.mean([r["train"][name] for r in result["scores"]])
        vals = [r["validation"][name] for r in result["scores"]]
        summary.append({"config": name, "train_mean": float(train),
                        "validation_mean": float(np.mean(vals)), "validation_std": float(np.std(vals)),
                        "validation_train_ratio": float(np.mean(vals) / train)})
    result["summary"] = summary
    result["selected"] = max(summary, key=lambda r: r["validation_mean"])
    write_json(report_path, result)
    print("Selected", result["selected"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--short", action="store_true")
    main(short=ap.parse_args().short)
