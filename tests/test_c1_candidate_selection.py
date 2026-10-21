"""Exercise the experimental selector through real LightGBM artifacts."""
import importlib.util
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from scripts import c1_candidate_recipe as selector
from scripts.final_c1_study import choose
from submission import ranking

DATA=Path('data/toy_transfer/a_40_features_labels0-4')


def test_conservative_selection_uses_inner_only():
    rows=[{'config':'baseline','inner_ndcg':.5,'outer_ndcg':.9},
          {'config':'leaves31','inner_ndcg':.5009,'outer_ndcg':.99},
          {'config':'regular31','inner_ndcg':.51,'outer_ndcg':.1}]
    assert choose(rows,['baseline','leaves31']) is rows[0]
    assert choose(rows,['baseline','leaves31','regular31']) is rows[2]


@pytest.mark.parametrize('policy,count',[('search',3),('raw_rank',2)])
def test_actual_search_is_repeatable_and_serializes_selected_model(tmp_path,policy,count):
    paths=[tmp_path/'first.txt',tmp_path/'second.txt']
    reports=[selector.train(str(DATA/'train.txt'),str(DATA/'vali.txt'),str(path),seed=3,rounds=40,policy=policy) for path in paths]
    assert paths[0].read_bytes()==paths[1].read_bytes()
    assert reports[0]['selected']==reports[1]['selected']
    assert len(reports[0]['fits'])==count
    ranking.prepare(str(paths[0]))
    result=ranking.rerank_with_ltr('q',['a','b'],{'a':np.zeros(40),'b':np.ones(40)},2)
    assert len(result)==2 and np.isfinite([s for _,s in result]).all()


def test_expired_budget_keeps_baseline_artifact_and_skips_search(tmp_path):
    path=tmp_path/'fallback.txt'
    result=selector.train(str(DATA/'train.txt'),str(DATA/'vali.txt'),str(path),budget_seconds=-1.)
    assert result['selected']=='baseline' and len(result['fits'])==1
    assert lgb.Booster(model_file=str(path)).num_trees()==1
    assert not list(tmp_path.glob('*.tmp'))
