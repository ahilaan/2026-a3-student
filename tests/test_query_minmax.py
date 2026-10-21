"""The new serialized feature schema must remain generic and match training."""
import numpy as np
import pytest
from submission import ranking
from scripts.score_campaign2_minmax import augment


def test_minmax_missing_constants_affine_and_training_agreement():
    x = np.array([[-5, 3, np.nan], [0, 3, 4], [5, 3, 8]], dtype=np.float32)
    actual = ranking._augment_query_minmax(x)
    np.testing.assert_array_equal(actual[:, :3], x)
    np.testing.assert_allclose(actual[:, 3:], [[0, 0, np.nan], [.5, 0, 0], [1, 0, 1]], equal_nan=True)
    np.testing.assert_array_equal(actual, augment(x))
    np.testing.assert_allclose(actual[:, 3:], ranking._augment_query_minmax(x*2+3)[:, 3:], equal_nan=True)
    assert ranking._augment_query_minmax(x[:0]).shape == (0, 6)


@pytest.mark.parametrize('width', [3, 17])
def test_minmax_model_schema_matches_actual_entrypoint_and_failed_load_resets(tmp_path, width):
    import lightgbm as lgb
    rng = np.random.RandomState(12)
    x = rng.randn(60, width).astype(np.float32)
    transformed = np.concatenate([augment(rows) for rows in x.reshape(6, 10, width)])
    names = [f'raw_{i}' for i in range(width)] + [f'qmin_{i}' for i in range(width)]
    model = lgb.train({'objective': 'lambdarank', 'num_leaves': 5, 'min_data_in_leaf': 2,
                       'num_threads': 1, 'verbosity': -1},
                      lgb.Dataset(transformed, label=rng.randint(0, 5, 60), group=[10]*6, feature_name=names),
                      num_boost_round=4)
    path = tmp_path/'minmax.txt';model.save_model(str(path));ranking.prepare(str(path))
    assert ranking._QUERY_MINMAX_FEATURES and not ranking._QUERY_RANK_FEATURES
    assert ranking._RAW_WIDTH == width
    ids = [str(i) for i in range(10)]
    result = ranking.rerank_with_ltr('any-query', ids+ids[:1], dict(zip(ids, x[:10])), 10)
    scores = model.predict(transformed[:10], num_threads=1)
    assert result == [(ids[i], scores[i]) for i in np.argsort(-scores, kind='stable')]
    with pytest.raises(FileNotFoundError):ranking.prepare(str(tmp_path/'missing.txt'))
    assert ranking._MODEL is None and ranking._RAW_WIDTH is None
    assert not ranking._QUERY_MINMAX_FEATURES and not ranking._QUERY_RANK_FEATURES
