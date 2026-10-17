import numpy as np
import pytest

from submission.query_features import augment_grouped, augment_query, feature_names
from submission import ranking


def test_average_ties_constants_and_candidate_permutations():
    X = np.array([[1, 5], [1, 5], [3, 5], [7, 5]], dtype=np.float32)
    out = augment_query(X)
    np.testing.assert_array_equal(out[:, :2], X)
    np.testing.assert_allclose(out[:, 2], [1/6, 1/6, 2/3, 1])
    np.testing.assert_array_equal(out[:, 3], 0)
    permutation = [3, 1, 0, 2]
    np.testing.assert_array_equal(augment_query(X[permutation]), out[permutation])
    assert feature_names(2) == ['raw_0', 'raw_1', 'qrank_0', 'qrank_1']


def test_queries_transform_independently_without_corpus_statistics():
    X = np.array([[0], [2], [1000], [1001]], dtype=np.float32)
    out = augment_grouped(X, ['a', 'a', 'b', 'b'])
    np.testing.assert_array_equal(out[:, 1], [0, 1, 0, 1])
    np.testing.assert_array_equal(augment_query(X[:1]), [[0, 0]])
    assert augment_grouped(X[:0], []).shape == (0, 2)
    with pytest.raises(ValueError, match='contiguous'):
        augment_grouped(X, ['a', 'b', 'a', 'b'])


@pytest.mark.parametrize('width', [3, 17])
def test_declared_feature_schema_roundtrips_through_real_entry_point(tmp_path, width):
    import lightgbm as lgb
    rng = np.random.RandomState(4)
    X = rng.randn(60, width).astype(np.float32)
    y = rng.randint(0, 5, 60)
    q = np.repeat(np.arange(6), 10)
    params = {'objective': 'lambdarank', 'num_leaves': 5, 'min_data_in_leaf': 2,
              'num_threads': 1, 'verbosity': -1}
    for engineered in (True, False):
        inputs = augment_grouped(X, q) if engineered else X
        names = feature_names(width) if engineered else 'auto'
        model = lgb.train(params, lgb.Dataset(inputs, label=y, group=[10]*6, feature_name=names), num_boost_round=4)
        path = tmp_path / ('relative.txt' if engineered else 'raw.txt')
        model.save_model(str(path))
        ranking.prepare(str(path))
        assert ranking._QUERY_RANK_FEATURES is engineered and ranking._RAW_WIDTH == width
        ids = [str(i) for i in range(10)]
        rows = dict(zip(ids, X[:10]))
        output = ranking.rerank_with_ltr('query', ids + ids[:1], rows, 10)
        scores = model.predict(inputs[:10], num_threads=1)
        expected = [(ids[i], scores[i]) for i in np.argsort(-scores, kind='stable')]
        assert output == expected
    with pytest.raises(FileNotFoundError):
        ranking.prepare(str(tmp_path / 'missing.txt'))
    assert ranking._MODEL is None and ranking._RAW_WIDTH is None and not ranking._QUERY_RANK_FEATURES
