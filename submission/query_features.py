"""Generic, label-free query-relative features for optional tree models."""
import numpy as np


def feature_names(width):
    return [f"raw_{i}" for i in range(width)] + [f"qrank_{i}" for i in range(width)]


def augment_query(X):
    """Append average-tie ascending percentiles within this candidate query.

    Constant features and single-document queries have percentile zero.
    Equal values receive equal percentiles, independent of candidate order.
    Original values remain available so trees can retain absolute context.
    """
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
