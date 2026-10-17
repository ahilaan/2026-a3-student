"""Reproducible TRAIN-only tuning; never opens any held-out partition.

Arrays are cached only for the two explicitly released files. Tree selection
uses fixed iteration counts and query folds, never early stopping on a reported
fold. Intermediate results are written after each fit to allow resuming.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from harness.cv import grouped_kfold
from harness.feature_names import BASE_RANKER_CHANNELS
from harness.metrics import mean_ndcg_from_scores
from scripts.ltr_data import load_splits
from scripts.lambdamart_utils import default_params, group_sizes

CACHE = ROOT / "data/cache/optimization"
RUNS = ROOT / "runs/optimization"
COUNTS = [200, 400, 800]
CONFIGS = {
    "leaves15": dict(num_leaves=15, min_data_in_leaf=50, lambda_l2=1),
    "leaves31": dict(num_leaves=31, min_data_in_leaf=50, lambda_l2=1),
    "leaves63reg": dict(num_leaves=63, min_data_in_leaf=100, lambda_l2=5),
}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def cached(split):
    if split not in ("train", "dev"):
        raise ValueError("only released TRAIN/DEV may be cached")
    source = ROOT / ("data/train/train.txt" if split == "train" else "data/dev/vali.txt")
    target = CACHE / split
    stat = source.stat()
    signature = {"source": str(source), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    manifest = target / "manifest.json"
    if not manifest.exists() or json.loads(manifest.read_text()) != signature:
        print(f"Caching released {split}: {source}", flush=True)
        (X, y, q), _ = load_splits(source)
        target.mkdir(parents=True, exist_ok=True)
        for name, array in zip(("X", "y", "q"), (X, y, q.astype(str))):
            np.save(target / (name + ".npy"), array, allow_pickle=False)
        write_json(manifest, signature)
        del X, y, q
    return tuple(np.load(target / (name + ".npy"), mmap_mode="r", allow_pickle=False)
                 for name in ("X", "y", "q"))


def snapshot():
    target = CACHE / "baseline17"
    if target.exists():
        return
    target.mkdir(parents=True)
    files = [ROOT / "2023CS10076.zip"] + list((ROOT / "submission").glob("*.py"))
    files += [ROOT / "submission" / n for n in ("model.txt", "pointwise_model.pt", "pairwise_model.pt")]
    hashes = {}
    for path in files:
        shutil.copy2(path, target / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(target / "sha256.json", hashes)


def trees(config_names, fold_ids):
    import lightgbm as lgb
    X, y, q = cached("train")
    folds = grouped_kfold(q, k=5, seed=0)
    path = RUNS / "tree_cv.json"
    result = json.loads(path.read_text()) if path.exists() else {
        "split": "Fold1 TRAIN only", "folds": 5, "seed": 0,
        "selection": "fixed tree counts; no early stopping on reported folds", "fits": []}
    for name in config_names:
        params = default_params(**CONFIGS[name], num_threads=4)
        for f in fold_ids:
            if any(r["config"] == name and r["fold"] == f for r in result["fits"]):
                continue
            tr, va = folds[f]
            started = time.perf_counter()
            ds = lgb.Dataset(X[tr], label=y[tr], group=group_sizes(q[tr]))
            print(f"Fitting {name} fold {f}, fixed {max(COUNTS)} trees", flush=True)
            model = lgb.train(params, ds, num_boost_round=max(COUNTS))
            checkpoint = CACHE / "cv_checkpoints"
            checkpoint.mkdir(parents=True, exist_ok=True)
            model.save_model(str(checkpoint / f"{name}_fold{f}.txt"))
            scores = []
            for count in COUNTS:
                record = {"trees": count}
                for label, rows in (("train", tr), ("validation", va)):
                    prediction = model.predict(X[rows], num_iteration=count, num_threads=4)
                    record[label] = mean_ndcg_from_scores(prediction, y[rows], q[rows])
                scores.append(record)
                print(name, f, record, flush=True)
            result["fits"].append({"config": name, "params": params, "fold": f,
                                   "seconds": time.perf_counter() - started, "scores": scores})
            write_json(path, result)
            del ds, model
    summary = []
    for name in config_names:
        fits = [r for r in result["fits"] if r["config"] == name]
        if len(fits) == 5:
            for count in COUNTS:
                records = [next(s for s in r["scores"] if s["trees"] == count) for r in fits]
                train_mean = float(np.mean([s["train"] for s in records]))
                val = [s["validation"] for s in records]
                summary.append({"config": name, "trees": count, "train_mean": train_mean,
                                "validation_mean": float(np.mean(val)), "validation_std": float(np.std(val)),
                                "validation_train_ratio": float(np.mean(val)) / train_mean})
    if summary:
        result["complete_summaries"] = summary
        result["selected"] = max(summary, key=lambda s: (s["validation_mean"], -s["trees"]))
        write_json(path, result)
        print("CV selected", result["selected"], flush=True)


def train_tree(name, count):
    import lightgbm as lgb
    X, y, q = cached("train")
    Xd, yd, qd = cached("dev")
    params = default_params(**CONFIGS[name], num_threads=4)
    model = lgb.train(params, lgb.Dataset(X, label=y, group=group_sizes(q)), num_boost_round=count)
    target = CACHE / f"{name}_{count}"
    target.mkdir(parents=True, exist_ok=True)
    model.save_model(str(target / "model.txt"))
    result = {"config": name, "trees": count, "params": params, "fit_split": "TRAIN only",
              "dev_ndcg@10": mean_ndcg_from_scores(model.predict(Xd, num_threads=4), yd, qd),
              "train_ndcg@10": mean_ndcg_from_scores(model.predict(X, num_threads=4), y, q)}
    write_json(RUNS / (target.name + ".json"), result)
    print(json.dumps(result), flush=True)


def fusion(fine=False):
    X, y, q = cached("train")
    starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1, len(q)]
    # Tune on 4800 TRAIN queries; reserve 1200 TRAIN queries for a confirmation.
    order = np.random.RandomState(0).permutation(len(starts) - 1)
    selection = np.zeros(len(order), dtype=bool)
    selection[order[1200:]] = True
    discounts = 1 / np.log2(np.arange(2, 12))
    settings = [("rrf", c) for c in (0, 5, 20, 60, 100, 300, 1000)]
    settings += [("minmax", power) for power in (0.5, 1, 2)]
    if fine:
        settings = [("rrf", c) for c in (20, 100, 1000)]
        settings += [("minmax", power) for power in (0.25, 0.5, 0.75, 1, 1.5, 2)]
    # Integer simplex includes sparse and balanced mixtures; ratios, not scale,
    # determine rankings. No DEV labels used to choose constants or weights.
    units = 8 if fine else 4
    weights = np.array([w for w in itertools.product(range(units + 1), repeat=4) if sum(w) == units], float)
    weights = np.vstack([weights, [1, 1, 1, 1]])
    totals = np.zeros((len(settings), len(weights), 2))
    names = list(BASE_RANKER_CHANNELS)
    for qi, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        n = b - a
        ranks = np.empty((n, 4), float)
        raw = np.empty((n, 4), float)
        for j, col in enumerate(BASE_RANKER_CHANNELS.values()):
            raw[:, j] = X[a:b, col - 1]
            ix = np.argsort(-raw[:, j], kind="stable")
            ranks[ix, j] = np.arange(1, n + 1)
        span = np.ptp(raw, axis=0)
        normalized = (raw - raw.min(axis=0)) / np.where(span > 0, span, 1)
        gain = 2.0 ** y[a:b] - 1
        limit = min(10, n)
        ideal = np.sort(gain)[::-1][:limit] @ discounts[:limit]
        if ideal:
            for ci, (family, value) in enumerate(settings):
                component = 1 / (value + ranks) if family == "rrf" else normalized ** value
                scores = component @ weights.T
                ranked = np.argsort(-scores, axis=0, kind="stable")[:limit]
                ndcg = (gain[ranked] * discounts[:limit, None]).sum(axis=0) / ideal
                totals[ci, :, int(selection[qi])] += ndcg
        if qi % 1000 == 0:
            print(f"Fusion TRAIN queries: {qi}/{len(order)}", flush=True)
    means = totals / np.array([1200, len(order) - 1200])
    ci, wi = np.unravel_index(np.argmax(means[:, :, 1]), means[:, :, 1].shape)
    result = {"fit_split": "TRAIN only", "tune_queries": len(order) - 1200, "confirmation_queries": 1200,
              "family": settings[ci][0], "constant": settings[ci][1],
              "weights": dict(zip(names, weights[wi].tolist())),
              "tune_ndcg@10": means[ci, wi, 1], "confirmation_ndcg@10": means[ci, wi, 0],
              "candidates": [{"family": family, "constant": value, "weights": weights[w].tolist(), "tune": means[i, w, 1],
                               "confirmation": means[i, w, 0]}
                              for i, (family, value) in enumerate(settings) for w in range(len(weights))]}
    if fine:
        baseline = json.loads((RUNS / "fusion.json").read_text())
        result["adopt_by_TRAIN_only"] = bool(
            result["tune_ndcg@10"] >= baseline["tune_ndcg@10"] + 0.0005 and
            result["confirmation_ndcg@10"] >= baseline["confirmation_ndcg@10"])
    write_json(RUNS / ("fusion_fine.json" if fine else "fusion.json"), result)
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["cache", "fusion", "trees", "train-tree"])
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--trees", type=int, default=800)
    ap.add_argument("--fine", action="store_true", help="finer TRAIN-only fusion grid")
    args = ap.parse_args()
    snapshot()
    if args.mode == "cache":
        for split in ("train", "dev"):
            arrays = cached(split)
            print(split, [a.shape for a in arrays], flush=True)
    elif args.mode == "fusion":
        fusion(fine=args.fine)
    elif args.mode == "trees":
        trees(args.configs, args.folds)
    else:
        train_tree(args.configs[0], args.trees)


if __name__ == "__main__":
    main()
