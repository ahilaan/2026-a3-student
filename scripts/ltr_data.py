"""Memory-bounded CLI data loading, sharing the recipe's SVMLight reader."""
import numpy as np

from submission.recipe import _read, _scan


def load_splits(train_path, dev_path=None):
    """Return TRAIN and optional DEV (X, y, qids) with a common feature width.

    Query and document order match harness.feature_io, including interleaved
    input queries, without retaining every parsed feature as a Python tuple.
    """
    tr_counts, tr_width = _scan(train_path)
    dv_counts, dv_width = _scan(dev_path) if dev_path is not None else ({}, 0)
    width = max(tr_width, dv_width)
    if not tr_counts or width == 0:
        raise ValueError("TRAIN must be nonempty with at least one feature")

    def load(path, counts):
        X, y, groups = _read(path, counts, width)
        qids = np.repeat(np.asarray(list(counts), dtype=object), groups)
        return X, y, qids

    return load(train_path, tr_counts), load(dev_path, dv_counts) if dev_path is not None else None
