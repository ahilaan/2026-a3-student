"""A later failure must leave an already loadable fallback, without hiding it."""
from pathlib import Path

import lightgbm as lgb
import pytest

from submission import recipe, ranking


def test_interrupted_native_training_leaves_first_tree(tmp_path,monkeypatch):
    target=tmp_path/'model.txt'
    original=lgb.train
    def train(*args,**kwargs):
        def interrupt(env):
            if env.iteration==1:raise RuntimeError('simulated later native failure')
        interrupt.order=15
        kwargs['callbacks'].append(interrupt)
        return original(*args,**kwargs)
    monkeypatch.setattr(lgb,'train',train)
    with pytest.raises(RuntimeError,match='simulated later'):
        recipe.train('data/toy_transfer/a_40_features_labels0-4/train.txt',
                     'data/toy_transfer/a_40_features_labels0-4/vali.txt',str(target))
    assert lgb.Booster(model_file=str(target)).num_trees()==1
    ranking.prepare(str(target))
    assert len(ranking.rerank_with_ltr('q',['a'],{'a':[0.]*40},1))==1
    assert not list(tmp_path.glob('*.tmp'))
