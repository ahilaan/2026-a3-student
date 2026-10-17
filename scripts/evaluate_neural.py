"""Reload neural artifacts, check repeatability, and report whole-query nDCG@10."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from harness.feature_names import N_FEATURES_MSLR, feature_index
from harness.metrics import mean_ndcg_from_scores
from harness.neural import load_scorer
from scripts.ltr_data import load_splits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/toy/vali.txt")
    ap.add_argument("--model-dir", default="submission")
    ap.add_argument("--report-out")
    args = ap.parse_args()
    torch.set_num_threads(4)
    (X, y, q), _ = load_splits(args.features)
    result = {"features": args.features, "num_queries": len(set(q)), "num_features": X.shape[1]}
    for kind in ("pointwise", "pairwise"):
        scorer = load_scorer(os.path.join(args.model_dir, f"{kind}_model.pt"))

        def predict():
            return np.concatenate([scorer(X[i:i + 1024]) for i in range(0, len(X), 1024)])

        scores = predict()
        if not np.isfinite(scores).all() or not np.array_equal(scores, predict()):
            raise ValueError(f"{kind} scores must be finite and deterministic")
        result[f"{kind}_ndcg@10"] = mean_ndcg_from_scores(scores, y, q)
    if X.shape[1] == N_FEATURES_MSLR:
        result["bm25_ndcg@10"] = mean_ndcg_from_scores(X[:, feature_index("bm25") - 1], y, q)
        result["pairwise_beats_bm25"] = result["pairwise_ndcg@10"] > result["bm25_ndcg@10"]
    print(json.dumps(result, indent=2))
    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
