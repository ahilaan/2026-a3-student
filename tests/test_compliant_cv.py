"""Exercise augmentation boundaries and the complete TRAIN-only CV entry point."""
import json
import sys

import numpy as np
from scripts import compliant_cv as cv
from scripts.compliant_train import prepare_matrix
from submission.query_features import augment_grouped


def test_matrix_blocks_retain_original_queries_and_repeat_deterministically(tmp_path):
    q = np.repeat([11, 22, 33], [2, 3, 2])
    raw = np.arange(1, 7*136+1, dtype=np.float32).reshape(7, 136)
    original = augment_grouped(raw, q)
    first = prepare_matrix(original, q, tmp_path/'first.npy', augment=True, seed=9)
    second = prepare_matrix(original, q, tmp_path/'second.npy', augment=True, seed=9)
    assert first.shape == (14, 272)
    np.testing.assert_array_equal(first[:7], original)
    np.testing.assert_array_equal(first, second)
    for lo, hi in [(0, 2), (2, 5), (5, 7)]:
        augmented_raw = first[7+lo:7+hi, :136]
        dropped = np.all(augmented_raw == 0, axis=0)
        assert 14 <= dropped.sum() <= 40
        assert np.isfinite(augmented_raw).all()
        assert np.all((first[7+lo:7+hi, 136:] >= 0) &
                      (first[7+lo:7+hi, 136:] <= 1))
    first._mmap.close()
    second._mmap.close()


def test_cv_main_opens_only_train_and_reports_all_disjoint_folds(tmp_path, monkeypatch):
    q = np.repeat(np.arange(10).astype(str), 5)
    raw = np.random.RandomState(4).normal(size=(len(q), 136)).astype(np.float32)
    X, y = augment_grouped(raw, q), np.tile(np.arange(5), 10)
    calls = []
    def loader(split):
        calls.append(split)
        assert split == 'train'
        return X, y, q
    monkeypatch.setattr(cv, 'data', loader)
    monkeypatch.setattr(cv, 'OUT', tmp_path/'results')
    monkeypatch.setattr(cv, 'CACHE', tmp_path/'cache')
    monkeypatch.setitem(cv.CONFIGS, 'unit_test',
                        dict(num_leaves=3, min_data_in_leaf=1, min_data_in_bin=1))
    monkeypatch.setattr(sys, 'argv', ['compliant_cv', '--name', 'smoke',
        '--config', 'unit_test', '--counts', '1', '2', '--augment', '--rank-only',
        '--threads', '1'])
    cv.main()
    report = json.loads((tmp_path/'results/smoke.json').read_text())
    assert calls == ['train']
    assert report['split'] == 'Fold1 TRAIN only'
    assert len(report['fits']) == 5
    ids = [q for fold in report['fits'] for q in fold['validation_queries']]
    assert len(ids) == len(set(ids)) == 10
    assert report['selected_trees'] in [1, 2]
    assert all(0 <= r['validation_mean'] <= 1 for r in report['summary'])
    assert not list((tmp_path/'cache').glob('*.npy'))
