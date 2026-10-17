"""Generic, label-free query-relative features for optional tree models."""
import numpy as np
from submission.ranking import _augment_query as augment_query, _query_feature_names as feature_names


def augment_grouped(X, qids):
    """Transform each complete query independently; no corpus statistics fit."""
    qids = np.asarray(qids)
    if len(X) != len(qids):
        raise ValueError("features and query IDs must have matching rows")
    if len(qids) == 0:
        return np.empty((0, X.shape[1] * 2), dtype=np.float32)
    starts = np.r_[0, np.flatnonzero(qids[1:] != qids[:-1]) + 1, len(qids)]
    if len(np.unique(qids)) != len(starts) - 1:
        raise ValueError("augment_grouped requires contiguous complete queries")
    out = np.empty((len(X), X.shape[1] * 2), dtype=np.float32)
    for a, b in zip(starts[:-1], starts[1:]):
        out[a:b] = augment_query(X[a:b])
    return out
