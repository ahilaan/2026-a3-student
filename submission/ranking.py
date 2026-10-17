"""
submission/ranking.py -- your competition entry (assignment Sections 4.2, 5).

    prepare(model_path)
    fuse_candidates(qid, candidate_doc_ids, base_scores, k)   -> Track A
    rerank_with_ltr(qid, candidate_doc_ids, features, k)      -> Tracks B and C

The entry uses TRAIN-tuned weighted score fusion and a trained LambdaMART
model (submission/model.txt). The harness
never trains anything: prepare() only LOADS the artifact, and must raise if it
is missing (no silent fallback to training or to an untrained model).

Both ranking functions must return at most k (doc_id, score) pairs, best
first, using ONLY doc_ids from `candidate_doc_ids`, with no duplicates.
`base_scores`: {doc_id: {channel: score}} (the 4 named channels).
`features`:    {doc_id: [full feature vector]}.
"""
import os
from typing import Dict, List, Tuple

import numpy as np

from submission.fusion import weighted_power_combsum

_MODEL = None
_QUERY_RANK_FEATURES = False
_RAW_WIDTH = None
_FUSION_WEIGHTS = {"bm25_whole_document": 2.0, "bm25_title": 3.0,
                   "lmir_dir_whole_document": 3.0, "lmir_jm_whole_document": 0.0}
_FUSION_POWER = 1.5


def _query_feature_names(width):
    return [f"raw_{i}" for i in range(width)] + [f"qrank_{i}" for i in range(width)]


def _augment_query(X):
    """Append label-free, average-tie percentiles within this candidate pool."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("query features must be a matrix")
    n, width = X.shape
    result = np.zeros((n, 2 * width), dtype=np.float32)
    result[:, :width] = X
    if n > 1:
        for col in range(width):
            values, inverse, counts = np.unique(X[:, col], return_inverse=True, return_counts=True)
            if len(values) > 1:
                average = np.cumsum(counts) - (counts + 1) / 2
                result[:, width + col] = average[inverse] / (n - 1)
    return result


def prepare(model_path: str) -> None:
    global _MODEL, _QUERY_RANK_FEATURES, _RAW_WIDTH
    _MODEL = None  # a failed load must not leave a previous dataset's model active
    _QUERY_RANK_FEATURES, _RAW_WIDTH = False, None
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"trained model artifact not found at {model_path!r}; run scripts/train_model.py and "
            f"include the artifact in your submission (grading never retrains)"
        )
    import lightgbm as lgb
    try:
        model = lgb.Booster(model_file=model_path)
    except (lgb.basic.LightGBMError, OSError) as exc:
        raise ValueError(f"cannot load LightGBM model artifact {model_path!r}: {exc}") from exc
    # Preprocessing is explicitly declared by the model's feature schema, never
    # inferred from a dataset name, dimension, query ID or feature distribution.
    names = model.feature_name()
    _QUERY_RANK_FEATURES = (len(names) % 2 == 0 and names == _query_feature_names(len(names) // 2))
    _RAW_WIDTH = model.num_feature() // 2 if _QUERY_RANK_FEATURES else model.num_feature()
    _MODEL = model


def _unique(ids: List[str]) -> List[str]:
    return list(dict.fromkeys(ids))


def fuse_candidates(qid: str, candidate_doc_ids: List[str], base_scores: Dict[str, Dict[str, float]],
                    k: int) -> List[Tuple[str, float]]:
    if k <= 0:
        return []
    ids = _unique(candidate_doc_ids)
    fused = weighted_power_combsum({d: base_scores[d] for d in ids}, _FUSION_WEIGHTS, power=_FUSION_POWER)
    ranked = sorted(ids, key=lambda d: -fused[d])  # stable: ties keep candidate order
    return [(d, float(fused[d])) for d in ranked[:k]]


def rerank_with_ltr(qid: str, candidate_doc_ids: List[str], features: Dict[str, List[float]],
                    k: int) -> List[Tuple[str, float]]:
    if _MODEL is None:
        raise RuntimeError("call prepare(model_path) first")
    ids = _unique(candidate_doc_ids)
    if not ids or k <= 0:
        return []
    X = np.asarray([features[d] for d in ids], dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != _RAW_WIDTH:
        raise ValueError(f"expected {_RAW_WIDTH} features per candidate, got shape {X.shape}")
    if _QUERY_RANK_FEATURES:
        X = _augment_query(X)
    preds = _MODEL.predict(X, num_threads=1)
    if not np.isfinite(preds).all():
        raise ValueError("LightGBM returned non-finite ranking scores")
    order = np.argsort(-preds, kind="stable")
    return [(ids[i], float(preds[i])) for i in order[:k]]
