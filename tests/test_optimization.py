"""Independent numerical checks for optional optimization code."""
import numpy as np
import pytest
import torch

from scripts.optimize_neural import lambda_pairs, weighted_ranknet_loss
from submission.fusion import weighted_power_combsum
from scripts.blend_neural import blend
from harness.neural import ScoringNetwork, make_scorer, save_scorer, load_scorer
from harness.preprocessing import Standardizer
from scripts.blend_trees import blend_models
from submission import pointwise_pairwise as pp


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


def test_blend_staff_loader_reproduces_weighted_original_scores(tmp_path):
    rng = np.random.RandomState(1)
    X = rng.randn(20, 7).astype(np.float32)
    std = Standardizer().fit(X)
    a, b = ScoringNetwork(7, [8, 5]), ScoringNetwork(7, [9, 6])
    pa, pb, pc = [tmp_path / (n + ".pt") for n in ("a", "b", "combined")]
    save_scorer(str(pa), a, std)
    save_scorer(str(pb), b, std)
    network, standardizer = blend(pa, pb, 0.25)
    save_scorer(str(pc), network, standardizer)
    expected = 0.25 * make_scorer(a, std)(X) + 0.75 * make_scorer(b, std)(X)
    np.testing.assert_allclose(load_scorer(str(pc))(X), expected, atol=1e-6, rtol=1e-5)


def test_tree_blend_roundtrip_preserves_convex_scores(tmp_path):
    import lightgbm as lgb
    rng = np.random.RandomState(3)
    X = rng.randn(60, 7)
    y = rng.randint(0, 5, 60)
    paths, models = [], []
    for seed in (0, 1):
        model = lgb.train({"objective": "lambdarank", "verbosity": -1, "num_threads": 2,
                           "num_leaves": 5, "min_data_in_leaf": 2, "feature_fraction": 0.7, "seed": seed},
                          lgb.Dataset(X, label=y, group=[10]*6), num_boost_round=5)
        path = tmp_path / f"tree{seed}.txt"
        model.save_model(str(path))
        paths.append(path)
        models.append(model)
    merged = blend_models(*paths, weight=0.25, first_trees=3, second_trees=4)
    out = tmp_path / "merged.txt"
    merged.save_model(str(out))
    loaded = lgb.Booster(model_file=str(out))
    assert loaded.num_trees() == 7 and loaded.num_feature() == 7
    probe = rng.randn(25, 7)
    expected = 0.25*models[0].predict(probe, num_iteration=3, num_threads=2)
    expected += 0.75*models[1].predict(probe, num_iteration=4, num_threads=2)
    np.testing.assert_allclose(loaded.predict(probe, num_threads=2), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('kind', ['pointwise', 'pairwise'])
def test_same_seed_is_independent_of_incoming_cpu_thread_state(kind):
    from harness.feature_io import read_svmlight_ltr, stack
    train = stack(read_svmlight_ltr('data/toy/train.txt'))
    dev = stack(read_svmlight_ltr('data/toy/vali.txt'))
    networks = []
    for incoming in (1, 4):
        torch.set_num_threads(incoming)
        network, _, _ = pp.train_model(kind, train, dev, seed=3, epochs=2)
        networks.append(network)
    for first, second in zip(networks[0].parameters(), networks[1].parameters()):
        assert torch.equal(first, second)
