"""Larger candidate pools use bounded parallel prediction without score drift."""
import lightgbm as lgb
import numpy as np

from submission import ranking


def test_parallel_reranking_matches_serial_predictions(tmp_path, monkeypatch):
    rng = np.random.RandomState(91)
    X = rng.normal(size=(908, 4)).astype(np.float32)
    y = rng.randint(0, 5, len(X))
    model = lgb.train(dict(objective='lambdarank', num_threads=1, verbosity=-1,
                           min_data_in_leaf=2, num_leaves=7),
                      lgb.Dataset(X, label=y, group=[len(X)]), num_boost_round=10)
    path = tmp_path/'model.txt'
    model.save_model(str(path))
    for name in ('_MODEL', '_QUERY_RANK_FEATURES', '_QUERY_MINMAX_FEATURES',
                 '_QUERY_COMBINED_FEATURES', '_RAW_WIDTH'):
        monkeypatch.setattr(ranking, name, getattr(ranking, name))
    ranking.prepare(str(path))
    monkeypatch.setattr(ranking._MODEL, 'num_trees', lambda: 4400)
    serial = ranking._MODEL.predict
    calls = []
    def observed(values, **kwargs):
        calls.append(kwargs['num_threads'])
        return serial(values, **kwargs)
    monkeypatch.setattr(ranking._MODEL, 'predict', observed)
    for count, expected_threads in [(908, 4), (10, 2), (5, 1)]:
        ids = [str(i) for i in range(count)]
        scores = serial(X[:count], num_threads=1)
        expected = [(ids[i], float(scores[i])) for i in np.argsort(-scores, kind='stable')[:10]]
        actual = ranking.rerank_with_ltr('opaque', ids, dict(zip(ids, X[:count])), 10)
        assert calls[-1] == expected_threads
        assert actual == expected
