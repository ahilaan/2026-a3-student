"""Fixed-count, query-fold validation of label-free query-relative features."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np

from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from scripts.optimize_rankers import CACHE, RUNS, cached, write_json
from scripts.lambdamart_utils import default_params, group_sizes
from submission.query_features import augment_grouped, feature_names


def data(split):
    X, y, q = cached(split)
    target = CACHE / split / "query_ranks_v1.npy"
    # Source signature is checked by cached(); arrays have fixed schema v1.
    if not target.exists() or target.stat().st_mtime_ns < (CACHE / split / "manifest.json").stat().st_mtime_ns:
        print("Building label-free query features:", split, flush=True)
        np.save(target, augment_grouped(X, q), allow_pickle=False)
    return np.load(target, mmap_mode="r", allow_pickle=False), y, q


def main(args):
    X, y, q = data("train")
    names = feature_names(X.shape[1] // 2)
    params = default_params(num_leaves=15, num_threads=4)
    result_path = RUNS / "query_features_cv.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {
        "split": "Fold1 TRAIN only", "feature_schema": "raw plus per-query average-tie percentiles v1",
        "params": params, "folds": 5, "seed": 0, "fits": []}
    if args.full:
        model = lgb.train(params, lgb.Dataset(X, label=y, group=group_sizes(q), feature_name=names),
                          num_boost_round=args.trees)
        target = CACHE / f"query_features15_{args.trees}"
        target.mkdir(exist_ok=True)
        model.save_model(str(target / "model.txt"))
        Xd, yd, qd = data("dev")
        report = {"trees": args.trees, "params": params, "gradients": "TRAIN only",
                  "dev_ndcg@10": mean_ndcg_from_scores(model.predict(Xd, num_threads=4), yd, qd)}
        write_json(RUNS / (target.name + ".json"), report)
        print(report, flush=True)
        return
    folds = grouped_kfold(q, 5, 0)
    for fold in args.folds:
        if any(r["fold"] == fold for r in result["fits"]):
            continue
        tr, va = folds[fold]
        print("Query-feature fixed-count CV fold", fold, flush=True)
        model = lgb.train(params, lgb.Dataset(X[tr], label=y[tr], group=group_sizes(q[tr]), feature_name=names),
                          num_boost_round=400)
        scores = []
        for count in (100, 200, 400):
            record = {"trees": count}
            for label, rows in (("train", tr), ("validation", va)):
                record[label] = mean_ndcg_from_scores(model.predict(X[rows], num_iteration=count, num_threads=4), y[rows], q[rows])
            scores.append(record)
            print(fold, record, flush=True)
        result["fits"].append({"fold": fold, "scores": scores})
        write_json(result_path, result)
    if len(result["fits"]) == 5:
        result["summary"] = []
        for count in (100, 200, 400):
            scores = [next(s for s in f["scores"] if s["trees"] == count) for f in result["fits"]]
            train = float(np.mean([s["train"] for s in scores]))
            val = [s["validation"] for s in scores]
            result["summary"].append({"trees": count, "train_mean": train,
                                      "validation_mean": float(np.mean(val)), "validation_std": float(np.std(val)),
                                      "validation_train_ratio": float(np.mean(val)) / train})
        write_json(result_path, result)
        print(result["summary"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--trees", type=int, default=200)
    main(ap.parse_args())
