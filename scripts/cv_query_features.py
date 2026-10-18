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


def combined_data(raw=False):
    """Only the two explicitly released splits; query-disjointness is checked."""
    loader = cached if raw else data
    Xt, yt, qt = loader("train")
    Xd, yd, qd = loader("dev")
    if len(np.intersect1d(np.unique(qt), np.unique(qd))):
        raise ValueError("combined split requires disjoint complete query blocks")
    name = "combined_raw_v1" if raw else "combined_query_v1"
    target, manifest = CACHE / (name + ".npy"), CACHE / (name + ".json")
    signature = {s: json.loads((CACHE / s / "manifest.json").read_text()) for s in ("train", "dev")}
    if not target.exists() or not manifest.exists() or json.loads(manifest.read_text()) != signature:
        print("Combining released TRAIN+DEV (labels included, declared)", flush=True)
        temporary = target.with_suffix(".tmp.npy")
        matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                           shape=(len(Xt) + len(Xd), Xt.shape[1]))
        matrix[:len(Xt)] = Xt
        matrix[len(Xt):] = Xd
        matrix.flush()
        del matrix
        temporary.replace(target)
        write_json(manifest, signature)
    return np.load(target, mmap_mode="r", allow_pickle=False), np.concatenate([yt, yd]), np.concatenate([qt, qd])


def main(args):
    X, y, q = combined_data(args.raw) if args.include_dev else (cached("train") if args.raw else data("train"))
    names = "auto" if args.raw else feature_names(X.shape[1] // 2)
    if args.variant == "extra":
        params = default_params(num_leaves=31, min_data_in_leaf=100, lambda_l2=5,
                                extra_trees=True, max_bin=63, feature_fraction=0.7, num_threads=4)
        prefix = "raw_extra31" if args.raw else "query_extra31"
    else:
        params = default_params(num_leaves=15, num_threads=4)
        prefix = "raw_features15" if args.raw else "query_features15"
    stem = "raw" if args.raw else "query"
    report_name = f"{stem}_extra_cv.json" if args.variant == "extra" else f"{stem}_features_cv.json"
    if args.include_dev:
        prefix += "_train_dev"
        report_name = report_name.replace("_cv.json", "_train_dev_cv.json")
    counts = sorted(set(args.counts))
    if not counts or min(counts) < 1:
        raise ValueError("fixed counts must be positive")
    result_path = RUNS / report_name
    result = json.loads(result_path.read_text()) if result_path.exists() else {
        "split": "Fold1 TRAIN+DEV combined" if args.include_dev else "Fold1 TRAIN only",
        "feature_schema": "raw" if args.raw else "raw plus per-query average-tie percentiles v1",
        "params": params, "folds": 5, "seed": 0, "counts": counts, "fits": []}
    if result.get("counts", [100, 200, 400]) != counts:
        raise ValueError("cannot mix fixed-count grids in one report")
    if args.full:
        model = lgb.train(params, lgb.Dataset(X, label=y, group=group_sizes(q), feature_name=names),
                          num_boost_round=args.trees)
        target = CACHE / f"{prefix}_{args.trees}"
        target.mkdir(exist_ok=True)
        model.save_model(str(target / "model.txt"))
        Xd, yd, qd = cached("dev") if args.raw else data("dev")
        report = {"trees": args.trees, "params": params,
                  "gradients": "TRAIN+DEV combined" if args.include_dev else "TRAIN only",
                  "dev_is_training": args.include_dev,
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
                          num_boost_round=max(counts))
        scores = []
        for count in counts:
            record = {"trees": count}
            for label, rows in (("train", tr), ("validation", va)):
                record[label] = mean_ndcg_from_scores(model.predict(X[rows], num_iteration=count, num_threads=4), y[rows], q[rows])
            scores.append(record)
            print(fold, record, flush=True)
        result["fits"].append({"fold": fold, "scores": scores})
        write_json(result_path, result)
    if len(result["fits"]) == 5:
        result["summary"] = []
        for count in counts:
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
    ap.add_argument("--variant", choices=["standard", "extra"], default="standard")
    ap.add_argument("--raw", action="store_true", help="compare the same training rule without relative features")
    ap.add_argument("--include-dev", action="store_true", help="DECLARED TRAIN+DEV fit/CV; DEV becomes in-training")
    ap.add_argument("--counts", type=int, nargs="+", default=[100, 200, 400])
    main(ap.parse_args())
