"""Bound standardization allocations without changing training rows or values."""
import numpy as np
import torch

from harness.preprocessing import Standardizer
from scripts.score_campaign2_neural import standardized_tensor


def test_chunked_training_tensor_preserves_values_order_and_float32():
    raw = np.random.RandomState(71).normal(size=(17, 6)).astype(np.float32)
    std = Standardizer().fit(raw[:10])
    expected = std.transform(raw)
    seen = []
    transform = std.transform

    def measured(block):
        seen.append(len(block))
        return transform(block)

    std.transform = measured
    actual = standardized_tensor(std, raw[:10], raw[10:], chunk_rows=4)
    assert actual.dtype == torch.float32
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert max(seen) <= 4
