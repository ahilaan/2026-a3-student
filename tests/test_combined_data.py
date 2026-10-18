"""Combined released-split experiments must preserve rows and query isolation."""
import json
import numpy as np
import pytest

import scripts.cv_query_features as experiment


@pytest.mark.parametrize('raw', [False, True])
def test_combined_cache_preserves_rows_and_rejects_query_overlap(tmp_path, monkeypatch, raw):
    splits = {
        'train': (np.array([[1, 2], [3, 4]], np.float32), np.array([0, 1]), np.array(['a', 'a'])),
        'dev': (np.array([[5, 6], [7, 8]], np.float32), np.array([3, 4]), np.array(['b', 'b'])),
    }
    seen = []
    def released_only(split):
        assert split in ('train', 'dev')
        seen.append(split)
        return splits[split]
    for split in splits:
        folder = tmp_path / split
        folder.mkdir()
        (folder / 'manifest.json').write_text(json.dumps({'split': split}))
    monkeypatch.setattr(experiment, 'CACHE', tmp_path)
    monkeypatch.setattr(experiment, 'cached', released_only)
    monkeypatch.setattr(experiment, 'data', released_only)
    X, y, q = experiment.combined_data(raw)
    np.testing.assert_array_equal(X, [[1, 2], [3, 4], [5, 6], [7, 8]])
    np.testing.assert_array_equal(y, [0, 1, 3, 4])
    np.testing.assert_array_equal(q, ['a', 'a', 'b', 'b'])
    assert seen == ['train', 'dev']
    X._mmap.close()
    X2, _, _ = experiment.combined_data(raw)
    np.testing.assert_array_equal(X2, [[1, 2], [3, 4], [5, 6], [7, 8]])
    X2._mmap.close()
    splits['dev'] = (splits['dev'][0], splits['dev'][1], np.array(['a', 'a']))
    with pytest.raises(ValueError, match='disjoint'):
        experiment.combined_data(raw)
