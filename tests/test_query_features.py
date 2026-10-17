import numpy as np
import pytest

from submission.query_features import augment_grouped, augment_query, feature_names


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
