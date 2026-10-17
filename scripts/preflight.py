"""Check the submitted artifacts on released DEV before spending an upload.

Unlike the synthetic unit tests, this includes the staff's real-data learning
floor: the loaded RankNet artifact must beat the single BM25 feature.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import torch

from harness.feature_names import feature_index
from harness.metrics import mean_ndcg_from_scores
from harness.neural import load_scorer
from scripts.ltr_data import load_splits
from submission import ranking


def check(features_path, model_dir, *, data=None):
    torch.set_num_threads(4)
    if data is None:
        (X, y, q), _ = load_splits(features_path)
    else:
        X, y, q = data
    if X.shape[1] != 136:
        raise ValueError(f"MSLR DEV must have 136 features, got {X.shape[1]}")
    paths = [Path(model_dir) / name for name in ("model.txt", "pointwise_model.pt", "pairwise_model.pt")]
    result = {"features": str(features_path), "rows": len(y), "queries": len(set(q)),
              "artifact_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    start = time.perf_counter()
    ranking.prepare(str(paths[0]))
    result["prepare_seconds"] = time.perf_counter() - start
    model = lgb.Booster(model_file=str(paths[0]))
    predicted = model.predict(X, num_threads=4)
    if not np.isfinite(predicted).all():
        raise ValueError("non-finite LambdaMART scores")
    result["ltr_ndcg@10"] = mean_ndcg_from_scores(predicted, y, q)
    for kind, path in zip(("pointwise", "pairwise"), paths[1:]):
        scorer = load_scorer(str(path))  # unchanged staff loader
        scores = np.concatenate([scorer(X[i:i + 4096]) for i in range(0, len(X), 4096)])
        if not np.isfinite(scores).all() or not np.array_equal(scorer(X[:128]), scorer(X[:128])):
            raise ValueError(f"{kind} scores are non-finite or not repeatable")
        result[f"{kind}_ndcg@10"] = mean_ndcg_from_scores(scores, y, q)
    result["bm25_ndcg@10"] = mean_ndcg_from_scores(X[:, feature_index("bm25") - 1], y, q)
    result["pairwise_beats_bm25"] = result["pairwise_ndcg@10"] > result["bm25_ndcg@10"]
    result["ltr_beats_bm25"] = result["ltr_ndcg@10"] > result["bm25_ndcg@10"]

    # Exercise the exact entry point on the largest query, including duplicates.
    unique, counts = np.unique(q, return_counts=True)
    rows = np.flatnonzero(q == unique[np.argmax(counts)])
    ids = [f"candidate-{i}" for i in range(len(rows))]
    features = dict(zip(ids, X[rows]))
    start = time.perf_counter()
    output = ranking.rerank_with_ltr("preflight", ids + ids[:1], features, 10)
    result["largest_query_seconds"] = time.perf_counter() - start
    if output != ranking.rerank_with_ltr("preflight", ids + ids[:1], features, 10):
        raise ValueError("rerank_with_ltr is not repeatable")
    if len(output) > 10 or len({d for d, _ in output}) != len(output):
        raise ValueError("ranking output violates candidate contract")
    result["passes"] = (result["pairwise_beats_bm25"] and result["ltr_beats_bm25"] and result["prepare_seconds"] < 60
                        and result["largest_query_seconds"] < 2)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/dev/vali.txt")
    ap.add_argument("--model-dir", default="submission")
    ap.add_argument("--report-out", default="runs/preflight.json")
    args = ap.parse_args()
    result = check(args.features, args.model_dir)
    os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
    Path(args.report_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passes"]:
        raise SystemExit("Preflight FAILED: do not upload this artifact set")


if __name__ == "__main__":
    main()
