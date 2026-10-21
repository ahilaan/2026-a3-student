"""Tempering retains sampled preferences and controls extreme query weights."""
import numpy as np
import pytest

from scripts.optimize_neural import lambda_pairs


def test_tempering_keeps_preferences_and_query_weight_mass():
    labels = np.array([0, 0, 1, 2, 4, 0, 0, 0, 1, 4], dtype=float)
    scores = np.array([1., 0., 2., 4., 3., 4., 3., 2., 1., 0.])
    starts = np.array([0, 5, 10])
    plain, a = lambda_pairs(starts, labels, scores, 100, np.random.RandomState(7))
    tempered, b = lambda_pairs(starts, labels, scores, 100, np.random.RandomState(7), .5)
    np.testing.assert_array_equal(plain, tempered)
    assert np.all(labels[plain[:,0]] > labels[plain[:,1]])
    assert np.all(b >= .1) and np.isfinite(b).all()
    for lo, hi in zip(starts[:-1], starts[1:]):
        use = (plain[:,0] >= lo) & (plain[:,0] < hi)
        assert np.all(plain[use,1] >= lo) and np.all(plain[use,1] < hi)
        assert np.isclose(a[use].mean(), b[use].mean())
        assert b[use].max() < a[use].max()
    again, weights = lambda_pairs(starts, labels, scores, 100, np.random.RandomState(7), 1.)
    np.testing.assert_array_equal(again, plain)
    np.testing.assert_array_equal(weights, a)


def test_invalid_tempering_is_rejected():
    with pytest.raises(ValueError,match='power'):
        lambda_pairs([0,2], np.array([0,1]), np.array([0.,1.]), 2, np.random.RandomState(0), -1.)
