"""Continuation retains old trees and refuses changed training provenance."""
import argparse
import hashlib
import json

import lightgbm as lgb
import numpy as np
import pytest

from scripts import compliant_package as package
from scripts.compliant_train import continuation_model


def test_continuation_preserves_parent_and_checks_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    X = np.random.RandomState(8).normal(size=(80, 4))
    labels = np.tile(np.arange(4), 20)
    params = dict(objective='lambdarank', num_threads=1, min_data_in_leaf=2,
                  num_leaves=5, verbosity=-1, deterministic=True, seed=0)
    names = ['raw_0', 'raw_1', 'qrank_0', 'qrank_1']
    def dataset():
        return lgb.Dataset(X, label=labels, group=[4]*20, feature_name=names)
    parent = lgb.train(params, dataset(), num_boost_round=3)
    model = tmp_path/'model.txt'
    parent.save_model(str(model))
    args = dict(mode='full', config='extra127', normalization='rank', augment=True,
                group_dropout=False, rank_only=False, seed=0, rounds=3)
    record = dict(model='model.txt', sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  training_splits='Fold1 TRAIN+DEV only', original_rows=958671,
                  original_queries=8000, args=args)
    receipt = tmp_path/'receipt.json'
    receipt.write_text(json.dumps(record))
    requested = argparse.Namespace(**dict(args, rounds=5))
    loaded, _ = continuation_model(receipt, requested, names)
    resumed = lgb.train(params, dataset(), num_boost_round=2, init_model=loaded)
    repeated = lgb.train(params, dataset(), num_boost_round=2, init_model=loaded)
    assert resumed.num_trees() == 5
    np.testing.assert_array_equal(resumed.predict(X, num_iteration=3), parent.predict(X))
    np.testing.assert_array_equal(resumed.predict(X), repeated.predict(X))
    requested.seed = 23
    with pytest.raises(ValueError, match='mismatch: seed'):
        continuation_model(receipt, requested, names)
    requested.seed = 0
    requested.rounds = 3
    with pytest.raises(ValueError, match='exceed'):
        continuation_model(receipt, requested, names)
    requested.rounds = 5
    model.write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes'):
        continuation_model(receipt, requested, names)


def test_package_checks_continuation_ancestry(tmp_path, monkeypatch):
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    model = tmp_path/'model.txt'
    model.write_bytes(b'weights')
    record = dict(model='model.txt', sha256=hashlib.sha256(b'weights').hexdigest(),
                  training_splits='Fold1 TRAIN+DEV only', original_rows=958671,
                  original_queries=8000, args={'mode':'full'})
    parent = tmp_path/'parent.json'
    parent.write_text(json.dumps(dict(record, training_splits='TRAIN+DEV+TEST')))
    child = tmp_path/'child.json'
    child.write_text(json.dumps(dict(record, continued_from='parent.json')))
    with pytest.raises(ValueError, match=r'TRAIN\+DEV fits'):
        package.verify_model(child)
