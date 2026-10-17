"""Additional baseline checks; the provided tests and harness stay unchanged."""
import itertools

import numpy as np
import pytest
import torch

from submission import fusion, pointwise_pairwise as pp, ranking, recipe
from scripts.lambdamart_utils import default_params, fit
from scripts.ltr_data import load_splits


@pytest.mark.parametrize("method", [fusion.rrf_scores, fusion.combsum_scores, fusion.combmnz_scores])
def test_fusion_empty_and_deterministic(method):
    assert method({}) == {}
    assert method({}, channels=["lm"]) == {}
    base = {"b": {"lm": -2.0, "constant": -5.0}, "a": {"lm": -2.0, "constant": -5.0},
            "c": {"lm": -8.0, "constant": -5.0}}
    assert method(base) == method(base)
    assert list(method(base)) == list(base)
    if method is fusion.rrf_scores:
        assert method(base)["b"] > method(base)["a"] > method(base)["c"]
    else:
        assert method(base) == {"b": 1.0, "a": 1.0, "c": 0.0}
        assert method(base, channels=["constant"]) == {d: 0.0 for d in base}
    assert method(base, channels=[]) == {d: 0.0 for d in base}


def test_fusion_candidate_order_duplicates_and_cutoff():
    base = {"a": {"x": 1.0}, "b": {"x": 1.0}, "foreign": {"x": 100.0}}
    assert [d for d, _ in ranking.fuse_candidates("q", ["b", "a", "b"], base, 10)] == ["b", "a"]
    assert ranking.fuse_candidates("q", [], {}, 10) == []
    assert ranking.fuse_candidates("q", ["b", "a"], base, 0) == []


def test_pairs_interleaved_complete_capped_and_empty():
    q = np.array(list("ababac"))
    y = np.array([0, 4, 2, 1, 3, 1])
    pairs = pp.make_pairs(q, y, 100, np.random.RandomState(5))
    expected = {(i, j) for i, j in itertools.permutations(range(len(q)), 2) if q[i] == q[j] and y[i] > y[j]}
    assert set(map(tuple, pairs)) == expected
    assert pairs.shape == (len(expected), 2) and np.issubdtype(pairs.dtype, np.integer)
    a = pp.make_pairs(q, y, 2, np.random.RandomState(5))
    b = pp.make_pairs(q, y, 2, np.random.RandomState(5))
    assert np.array_equal(a, b)
    assert len(set(map(tuple, a))) == len(a)
    assert all(np.sum(q[a[:, 0]] == query) <= 2 for query in set(q))
    for ids, labels, cap in [([], [], 3), ([1, 1], [2, 2], 3), ([1, 1], [0, 1], 0)]:
        assert pp.make_pairs(ids, labels, cap, np.random.RandomState(0)).shape == (0, 2)


def test_pair_sampling_large_query_without_enumerating_all_pairs():
    labels = np.arange(10000) % 5
    pairs = pp.make_pairs(["q"] * len(labels), labels, 300, np.random.RandomState(0))
    assert pairs.shape == (300, 2)
    assert len(set(map(tuple, pairs))) == 300
    assert np.all(labels[pairs[:, 0]] > labels[pairs[:, 1]])


def test_ranknet_extreme_scores_have_finite_gradients():
    preferred = torch.tensor([-1000.0, 1000.0], requires_grad=True)
    other = -preferred.detach()
    loss = pp.ranknet_loss(preferred, other)
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert torch.isfinite(preferred.grad).all()


@pytest.mark.parametrize("kind", ["pointwise", "pairwise"])
def test_training_restores_best_weights_and_uses_train_statistics(monkeypatch, kind):
    rng = np.random.RandomState(2)
    X = rng.randn(12, 3).astype(np.float32)
    train = (X, np.arange(12) % 3, np.repeat(["a", "b", "c"], 4))
    dev = (X + 50, train[1], train[2])
    monkeypatch.setattr(pp, "dev_ndcg", lambda *args: 0.8)
    first, _, _ = pp.train_model(kind, train, dev, epochs=1, batch_size=5)
    scores = iter([0.8, 0.7, 0.6, 0.5])
    monkeypatch.setattr(pp, "dev_ndcg", lambda *args: next(scores))
    net, std, history = pp.train_model(kind, train, dev, epochs=4, batch_size=5, patience=2)
    assert len(history) == 3
    assert np.allclose(std.mean, X.mean(axis=0))
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), net.parameters()))
    assert all(np.isfinite(record["train_loss"]) for record in history)


@pytest.mark.parametrize("train_feature,vali_feature", [(2, 7), (7, 2)])
def test_recipe_sparse_comments_interleaved_queries_and_width(tmp_path, train_feature, vali_feature):
    paths = [tmp_path / "train.txt", tmp_path / "vali.txt"]
    for path, feature in zip(paths, [train_feature, vali_feature]):
        path.write_text("\n".join(f"{i % 3} qid:{i % 2} 1:{i} {feature}:0.5 # row {i}" for i in range(120)),
                        encoding="utf-8")
    counts, width = recipe._scan(paths[0])
    X, y, groups = recipe._read(paths[0], counts, width)
    assert groups == [60, 60]
    assert np.array_equal(X[:, 0], list(range(0, 120, 2)) + list(range(1, 120, 2)))
    assert np.array_equal(y, X[:, 0].astype(int) % 3)
    train_data, vali_data = load_splits(*paths)
    assert train_data[0].shape == vali_data[0].shape == (120, 7)
    assert list(train_data[2]) == ["0"] * 60 + ["1"] * 60
    model_path = tmp_path / "model.txt"
    recipe.train(str(paths[0]), str(paths[1]), str(model_path), seed=4)
    ranking.prepare(str(model_path))
    assert ranking._MODEL.num_feature() == 7
    features = {"a": [0.0] * 7, "b": [0.0] * 7}
    expected = ranking.rerank_with_ltr("q", ["b", "a", "b"], features, 10)
    assert expected == ranking.rerank_with_ltr("q", ["b", "a", "b"], features, 10)
    assert [d for d, _ in expected] == ["b", "a"]
    assert all(type(score) is float and np.isfinite(score) for _, score in expected)
    assert ranking.rerank_with_ltr("q", [], {}, 10) == []
    with pytest.raises(ValueError, match="features"):
        ranking.rerank_with_ltr("q", ["x"], {"x": [0.0]}, 1)


def test_lambdamart_regroups_features_and_labels_together():
    rng = np.random.RandomState(0)
    X = rng.randn(160, 3).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    q = np.tile(np.arange(4), 40)
    params = default_params(min_data_in_leaf=5)
    interleaved = fit(X, y, q, params, 5)
    order = np.argsort(q, kind="stable")
    grouped = fit(X[order], y[order], q[order], params, 5)
    assert np.array_equal(interleaved.predict(X, num_threads=1), grouped.predict(X, num_threads=1))


def test_prepare_failed_load_clears_previous_model(tmp_path):
    invalid = tmp_path / "bad.txt"
    invalid.write_text("not a LightGBM model", encoding="utf-8")
    ranking._MODEL = object()
    with pytest.raises(ValueError, match="cannot load LightGBM"):
        ranking.prepare(str(invalid))
    assert ranking._MODEL is None
