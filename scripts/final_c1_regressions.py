"""Compare preserved and final scorers on authorized complete TRAIN queries."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from harness.feature_names import BASE_RANKER_CHANNELS
from harness.neural import load_scorer
from scripts.optimize_rankers import cached
from submission import ranking


def main():
    directory=ROOT/'runs/final_c1_pass'
    baseline=directory/'baseline'
    hashes=json.loads((baseline/'sha256.json').read_text())
    checks={}
    for name,expected in hashes.items():
        if name.startswith('submission') and not name.endswith('recipe.py'):
            checks[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
    spec=importlib.util.spec_from_file_location('preserved_ranking',baseline/'submission/ranking.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.prepare(str(baseline/'submission/model.txt'));ranking.prepare(str(ROOT/'submission/model.txt'))
    X,y,q=cached('train')
    sample=np.random.RandomState(221).permutation(np.unique(q))[:50]
    count=0
    for query in sample:
        inputs=X[q==query]
        ids=[f'{query}:{i}' for i in range(len(inputs))]
        features=dict(zip(ids,inputs))
        base={doc:{name:float(x[index-1]) for name,index in BASE_RANKER_CHANNELS.items()} for doc,x in zip(ids,inputs)}
        assert old.rerank_with_ltr(str(query),ids,features,len(ids))==ranking.rerank_with_ltr(str(query),ids,features,len(ids))
        assert old.fuse_candidates(str(query),ids,base,len(ids))==ranking.fuse_candidates(str(query),ids,base,len(ids))
        count+=len(ids)
    neural={}
    rows=np.flatnonzero(np.isin(q,sample))[:2000]
    for name in ['pointwise_model.pt','pairwise_model.pt']:
        a,b=load_scorer(str(baseline/'submission'/name)),load_scorer(str(ROOT/'submission'/name))
        neural[name]=bool(np.array_equal(a(X[rows]),b(X[rows])))
    assert all(checks.values()) and all(neural.values())
    result={'unchanged_sha256':checks,'comparison_queries':50,'comparison_documents':count,
            'main_scores_order_ids_identical':True,'fusion_scores_order_ids_identical':True,
            'neural_scores_identical':neural,'source':'released TRAIN only'}
    (directory/'regressions.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
