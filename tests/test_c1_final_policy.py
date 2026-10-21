"""End-to-end equivalence of the deployed policy and the measured experiment."""
from pathlib import Path
import shutil
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pytest

from scripts.c1_candidate_recipe import train as experimental
from submission import recipe, ranking


@pytest.mark.parametrize('name',[
    'a_40_features_labels0-4','b_90_features_labels0-4','c_250_features_labels0-2'])
def test_deployed_policy_matches_measured_selector_bytes(tmp_path,name):
    data=Path('data/toy_transfer')/name
    final,reference=tmp_path/'final.txt',tmp_path/'reference.txt'
    recipe.train(str(data/'train.txt'),str(data/'vali.txt'),str(final),seed=0)
    result=experimental(str(data/'train.txt'),str(data/'vali.txt'),str(reference),seed=0,policy='raw_rank')
    assert final.read_bytes()==reference.read_bytes()
    ranking.prepare(str(final))
    width=int(name.split('_')[1])
    assert ranking._RAW_WIDTH==width
    assert len(ranking.rerank_with_ltr('q',['a'],{'a':[0.]*width},1))==1


def test_expired_shared_budget_skips_optional_transform(tmp_path,monkeypatch):
    monkeypatch.setattr(recipe,'_TIME_BUDGET_SECONDS',0.)
    def forbidden(*args):raise AssertionError('Optional transform started after expiry')
    monkeypatch.setattr(recipe,'_query_rank_features',forbidden)
    out=tmp_path/'fallback.txt'
    recipe.train('data/toy_transfer/a_40_features_labels0-4/train.txt',
                 'data/toy_transfer/a_40_features_labels0-4/vali.txt',str(out))
    assert lgb.Booster(model_file=str(out)).num_trees()==1
    ranking.prepare(str(out))
    assert not ranking._QUERY_RANK_FEATURES


def test_expired_feature_budget_returns_without_altering_raw_values():
    X=np.arange(40,dtype=np.float32).reshape(10,4);original=X.copy()
    assert recipe._query_rank_features(X,[3,7],-1.) is None
    np.testing.assert_array_equal(X,original)


def test_recipe_works_without_offline_scripts_or_optional_source_modules(tmp_path):
    root=Path(__file__).resolve().parents[1]
    package=tmp_path/'submission';package.mkdir()
    for name in ['__init__.py','ranking.py','fusion.py','recipe.py','pointwise_pairwise.py']:
        shutil.copy2(root/'submission'/name,package/name)
    train=root/'data/toy_transfer/a_40_features_labels0-4/train.txt'
    valid=train.with_name('vali.txt')
    program=('import sys; from submission import recipe,ranking; '
             'recipe.train(sys.argv[1],sys.argv[2],"model.txt"); '
             'ranking.prepare("model.txt"); '
             'assert len(ranking.rerank_with_ltr("q",["a"],{"a":[0.]*40},1))==1')
    subprocess.run([sys.executable,'-c',program,str(train),str(valid)],cwd=tmp_path,
                   capture_output=True,text=True,check=True)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_small_query_fast_transform_exactly_matches_existing_inference(dtype):
    import time
    X=np.array([[1,3,np.nan,np.nan,0,1+1e-9,np.inf,-np.inf],
                [1,2,4,np.nan,-0.,1+1e-8,-np.inf,-np.inf]],dtype=dtype)
    actual=recipe._query_rank_features(X,[2],time.monotonic()+10.)
    expected=ranking._augment_query(X)
    np.testing.assert_array_equal(actual,expected)
