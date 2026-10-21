"""The assignment does not cap documents/query; native LambdaRank does."""
import lightgbm as lgb
import numpy as np

from harness.metrics import mean_ndcg_from_scores
from submission import recipe, ranking


def test_bounded_native_blocks_never_cross_original_query():
    groups=[3,10001,2,25000]
    bounded=recipe._bounded_groups(groups)
    q=np.repeat(np.arange(len(groups)),groups)
    assert sum(bounded)==len(q) and max(bounded)<=10000
    bounds=np.r_[0,np.cumsum(bounded)]
    assert all(np.unique(q[a:b]).size==1 for a,b in zip(bounds[:-1],bounds[1:]))
    assert recipe._bounded_groups([1,50,10000])==[1,50,10000]


def test_whole_query_metric_matches_staff_zero_and_tie_conventions():
    y=np.array([0,0,2,1,0,3,1,2],dtype=np.int32)
    groups=[2,3,3];q=np.repeat(np.arange(3),groups)
    scores=np.array([0,0,1,1,2,4,4,0.])
    name,value,higher=recipe._whole_query_metric(y,groups)(scores,None)
    assert abs(value-mean_ndcg_from_scores(scores,y,q))<1e-14 and higher


def test_native_oversized_train_and_validation_queries_save_loadable_model(tmp_path):
    paths=[tmp_path/'train.txt',tmp_path/'vali.txt']
    for path in paths:
        with path.open('w') as stream:
            for i in range(10001):
                stream.write(f'{i%3} qid:large 1:{i%7} 2:{i%3}\n')
    out=tmp_path/'model.txt'
    recipe.train(str(paths[0]),str(paths[1]),str(out))
    model=lgb.Booster(model_file=str(out))
    assert model.num_feature()==2 and model.num_trees()>=1
    ranking.prepare(str(out))
    assert len(ranking.rerank_with_ltr('q',['a','b'],{'a':[0.,0.],'b':[1.,2.]},2))==2
