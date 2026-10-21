import numpy as np
import pytest
from scripts.score_campaign3_union import lift_pair
from submission.ranking import _augment_query, _augment_query_minmax


@pytest.mark.parametrize('width',[3,9,17])
def test_feature_remapping_preserves_both_models_at_digit_boundaries(width,tmp_path):
    import lightgbm as lgb
    rng=np.random.RandomState(7);raw=rng.randn(60,width).astype(np.float32)
    ranks=np.concatenate([_augment_query(x) for x in raw.reshape(6,10,width)])
    mins=np.concatenate([_augment_query_minmax(x) for x in raw.reshape(6,10,width)])
    params={'objective':'lambdarank','num_leaves':5,'min_data_in_leaf':2,'num_threads':1,'verbosity':-1,
            'feature_contri':[0.]*width+[1.]*width}
    models=[]
    for inputs,kind in [(ranks,'qrank'),(mins,'qmin')]:
        names=[f'raw_{i}' for i in range(width)]+[f'{kind}_{i}' for i in range(width)]
        models.append(lgb.train(params,lgb.Dataset(inputs,label=rng.randint(0,5,60),group=[10]*6,feature_name=names),num_boost_round=4))
    lifted=lift_pair(*models);union=np.column_stack([ranks,mins[:,width:]])
    for old,new,inputs in zip(models,lifted,[ranks,mins]):
        assert new.num_feature()==3*width
        np.testing.assert_array_equal(old.predict(inputs,num_threads=1),new.predict(union,num_threads=1))
    from scripts.blend_trees import blend_models
    from submission import ranking
    paths=[tmp_path/'rank.txt',tmp_path/'minmax.txt']
    for model,path in zip(lifted,paths):model.save_model(str(path))
    combined=blend_models(*paths,weight=.4);output=tmp_path/'combined.txt';combined.save_model(str(output))
    ranking.prepare(str(output));assert ranking._QUERY_COMBINED_FEATURES and ranking._RAW_WIDTH==width
    ids=[str(i) for i in range(10)]
    actual=ranking.rerank_with_ltr('unseen-id',ids+ids[:1],dict(zip(ids,raw[:10])),10)
    expected=.4*models[0].predict(ranks[:10],num_threads=1)+.6*models[1].predict(mins[:10],num_threads=1)
    assert [d for d,_ in actual]==[ids[i] for i in np.argsort(-expected,kind='stable')]
    np.testing.assert_allclose([v for _,v in actual],sorted(expected,reverse=True),rtol=0,atol=1e-12)
    with pytest.raises(FileNotFoundError):ranking.prepare(str(tmp_path/'absent.txt'))
    assert not ranking._QUERY_COMBINED_FEATURES and ranking._MODEL is None
