"""Independent numerical checks for optional optimization code."""
import numpy as np
import pytest
import torch

from scripts.optimize_neural import lambda_pairs, weighted_ranknet_loss
from submission.fusion import weighted_power_combsum


def test_weighted_loss_matches_manual_and_has_finite_gradients():
    hi = torch.tensor([0., -1000.], requires_grad=True)
    lo = torch.tensor([0., 1000.])
    value = weighted_ranknet_loss(hi, lo, torch.tensor([3., 1.]))
    assert value.item() == pytest.approx((3 * np.log(2) + 2000) / 4)
    value.backward()
    assert torch.isfinite(hi.grad).all() and (hi.grad < 0).all()


def test_lambda_pairs_are_label_supervised_query_local_and_reproducible():
    starts = [0, 30, 45]
    labels = np.arange(45) % 5
    prediction = np.arange(45)[::-1]
    pairs, weights = lambda_pairs(starts, labels, prediction, 25, np.random.RandomState(5))
    again = lambda_pairs(starts, labels, prediction, 25, np.random.RandomState(5))
    assert np.array_equal(pairs, again[0]) and np.array_equal(weights, again[1])
    assert np.all(labels[pairs[:, 0]] > labels[pairs[:, 1]])
    assert np.all((pairs[:, 0] < 30) == (pairs[:, 1] < 30))
    assert len(pairs) <= 50 and np.all(weights > 0) and np.isfinite(weights).all()


def test_power_fusion_ignores_disabled_channel_and_handles_constants():
    scores = {"a": {"x": 0., "y": -5., "z": 100.},
              "b": {"x": 1., "y": -5., "z": 0.},
              "c": {"x": 4., "y": -5., "z": -100.}}
    assert weighted_power_combsum(scores, {"x": 2., "y": 3., "z": 0.}) == {"a": 0., "b": 1., "c": 2.}
