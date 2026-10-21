"""Actual baseline/search/racing calls on nested query splits, including I/O.

Profiles nine representatives: three toy, three generated, and the full raw
width at each MSLR TRAIN pool size, fold zero. No outer query enters a recipe.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import lightgbm as lgb
import numpy as np
from harness.metrics import mean_ndcg_from_scores
from scripts.c1_candidate_recipe import train as search
from scripts.final_c1_study import conditions,inner_split,save
from scripts.stress_c1 import peak_memory_bytes
from submission import ranking

OUT=ROOT/'runs/final_c1_pass/profiles'


def write(path,X,y,q):
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='ascii') as stream:
        for x,label,query in zip(X,y,q):
            values=' '.join(f'{i+1}:{v:.9g}' for i,v in enumerate(x) if v!=0)
            # Explicit trailing zero declares width even if last column is absent.
            if x[-1]==0:values+=f' {X.shape[1]}:0'
            stream.write(f'{int(label)} qid:{query} {values} # synthetic or released TRAIN partition\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('preserved_recipe',ROOT/'runs/final_c1_pass/baseline/submission/recipe.py')
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    path=OUT/'results.json';records=json.loads(path.read_text()) if path.exists() else []
    for name,data,splits,family in conditions():
        if family=='MSLR_TRAIN' and not name.endswith('d136'):continue
        X,y,q=data;available,outer=splits[0]
        tr,va=inner_split(available,q,0)
        directory=ROOT/'tmp/final_c1_profiles'/name
        train,vali=directory/'train.txt',directory/'vali.txt'
        write(train,X[tr],y[tr],q[tr]);write(vali,X[va],y[va],q[va])
        for policy in ['baseline','two','search','successive','raw_rank']:
            if any(r['dataset']==name and r['policy']==policy for r in records):continue
            model_path=directory/(policy+'.txt')
            start=time.monotonic()
            if policy=='baseline':
                baseline.train(str(train),str(vali),str(model_path),seed=0)
                details={'policy':policy,'selected':'baseline'}
            else:details=search(str(train),str(vali),str(model_path),seed=0,policy=policy)
            seconds=time.monotonic()-start
            model=lgb.Booster(model_file=str(model_path))
            ranking.prepare(str(model_path))
            if ranking._QUERY_RANK_FEATURES:
                from submission.query_features import augment_grouped
                outer_inputs=augment_grouped(X[outer],q[outer])
                inner_inputs=augment_grouped(X[va],q[va])
            else:outer_inputs,inner_inputs=X[outer],X[va]
            predictions=model.predict(outer_inputs,num_threads=4)
            rows=outer[q[outer]==q[outer][0]]
            ids=[str(i) for i in range(len(rows))]
            actual=ranking.rerank_with_ltr('q',ids,dict(zip(ids,X[rows])),10)
            assert [doc for doc,_ in actual]==[ids[i] for i in np.argsort(-predictions[:len(rows)],kind='stable')[:10]]
            rec={'dataset':name,'family':family,'policy':policy,'seconds':seconds,
                 'outer_ndcg':mean_ndcg_from_scores(predictions,y[outer],q[outer]),
                 'inner_ndcg':mean_ndcg_from_scores(model.predict(inner_inputs,num_threads=4),y[va],q[va]),
                 'trees':model.num_trees(),'model_bytes':model_path.stat().st_size,
                 'sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
                 'peak_process_bytes':peak_memory_bytes(),'interface_ok':True,'details':details}
            records.append(rec);save(path,records)
            print(name,policy,'seconds',round(seconds,2),'selected',details['selected'],flush=True)
    print('Completed',len(records),'actual API calls',flush=True)


if __name__=='__main__':main()
