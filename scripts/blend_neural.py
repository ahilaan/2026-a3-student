"""Combine two label-trained MLPs into ONE staff-compatible bounded MLP.

This copies parameters into concatenated hidden layers and block-diagonal
connections. It trains on no model scores and adds no custom inference logic.
Both input standardizers must be identical, and the total widths must satisfy
the ordinary staff bounds. The resulting score is their weighted average.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from harness.neural import ScoringNetwork, load_scorer, save_scorer
from harness.preprocessing import Standardizer
from harness.metrics import mean_ndcg_from_scores
from scripts.optimize_rankers import cached, CACHE, RUNS, write_json


def blend(first, second, first_weight=0.5):
    """Exact two-hidden-layer construction, with all staff width bounds intact."""
    if not 0 <= first_weight <= 1:
        raise ValueError("convex weights required")
    a, b = [torch.load(str(p), weights_only=True) for p in (first, second)]
    if a["standardizer"] != b["standardizer"]:
        raise ValueError("input standardizers must match")
    ha, hb = a["config"]["hidden"], b["config"]["hidden"]
    if len(ha) != 2 or len(hb) != 2 or a["config"]["n_features"] != b["config"]["n_features"]:
        raise ValueError("two-hidden-layer networks of equal input width required")
    network = ScoringNetwork(a["config"]["n_features"], [ha[0] + hb[0], ha[1] + hb[1]])
    sa, sb = a["state_dict"], b["state_dict"]
    with torch.no_grad():
        network.net[0].weight.copy_(torch.cat([sa["net.0.weight"], sb["net.0.weight"]]))
        network.net[0].bias.copy_(torch.cat([sa["net.0.bias"], sb["net.0.bias"]]))
        network.net[2].weight.zero_()
        network.net[2].weight[:ha[1], :ha[0]].copy_(sa["net.2.weight"])
        network.net[2].weight[ha[1]:, ha[0]:].copy_(sb["net.2.weight"])
        network.net[2].bias.copy_(torch.cat([sa["net.2.bias"], sb["net.2.bias"]]))
        network.net[4].weight.copy_(torch.cat([first_weight * sa["net.4.weight"],
                                               (1 - first_weight) * sb["net.4.weight"]], dim=1))
        network.net[4].bias.copy_(first_weight * sa["net.4.bias"] + (1 - first_weight) * sb["net.4.bias"])
    return network.eval(), Standardizer.from_dict(a["standardizer"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first", required=True)
    ap.add_argument("--second", required=True)
    ap.add_argument("--name", default="lambda_blend")
    args = ap.parse_args()
    torch.set_num_threads(2)
    X, y, q = cached("dev")
    target = CACHE / args.name
    target.mkdir(parents=True, exist_ok=True)
    results = {"sources": [args.first, args.second], "selection": "DEV only; no gradient fitting", "scores": []}
    for w in (0.25, 0.5, 0.75):
        network, std = blend(args.first, args.second, w)
        path = target / f"blend_{w}.pt"
        save_scorer(str(path), network, std)
        scorer = load_scorer(str(path))
        pred = np.concatenate([scorer(X[i:i+4096]) for i in range(0, len(X), 4096)])
        record = {"first_weight": w, "dev_ndcg@10": mean_ndcg_from_scores(pred, y, q), "path": str(path)}
        results["scores"].append(record)
        print(record, flush=True)
    write_json(RUNS / (args.name + ".json"), results)
