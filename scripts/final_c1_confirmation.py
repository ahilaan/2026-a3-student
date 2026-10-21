"""Additional frozen confirmation: unused TRAIN queries and generated scales.

Registered based only on inner screening; no outer scores used to construct
these conditions. This file never accesses DEV, TEST or public LTR caches.
"""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from harness.cv import grouped_kfold
from scripts import final_c1_study as study
from scripts.benchmark_c1 import generated
from scripts.optimize_rankers import cached

OUT=ROOT/'runs/final_c1_pass/confirmation'


def conditions():
    for name,width,seed,kind in [('query_shift_scale',90,471,'shift'),
                                ('heavy_tail',40,472,'exp'),('weak_sparse',300,473,'noise')]:
        X,y,q=generated(width,360,.05 if kind=='noise' else 1.,2 if kind=='noise' else 5,seed)
        rng=np.random.RandomState(seed+50)
        if kind=='shift':
            for query in np.unique(q):
                rows=q==query
                X[rows]=X[rows]*np.exp(rng.normal(0,1.5,width))+rng.normal(0,4,width)
        if kind=='exp':X=np.exp(np.clip(X*2,-15,15)).astype(np.float32)
        if kind=='noise':y=rng.randint(0,3,len(y)).astype(np.int32)
        data=X,y,q
        yield name,data,grouped_kfold(q,3,84),'generated_confirmation'
    X,y,q=cached('train')
    selected=np.random.RandomState(193).permutation(np.unique(q))[2250:4050]
    rows=np.flatnonzero(np.isin(q,selected))
    data=X[rows],y[rows],q[rows]
    yield 'MSLR_disjoint1800_d136',data,grouped_kfold(data[2],3,84),'MSLR_TRAIN_confirmation'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'configs':['baseline','leaves31','query_ranks'],'outer_seed':84,
              'MSLR_queries':'seed193 shuffled TRAIN IDs [2250:4050], disjoint from initial pools',
              'generated':['90 columns, per-query shift/scale','40 columns, global exp scales','300 sparse columns, random labels'],
              'seeds':[471,472,473],'policies':['baseline','fixed31','fixed_query_ranks','inner raw/rank selection with .001 threshold'],
              'acceptance':'Confirm positive equal-family raw/rank mean delta, no family mean below -.005; initial extension gate must also pass.'}
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Frozen protocol changed')
    study.save(path,protocol)
    study.OUT=OUT
    path=OUT/'fits.json';records=json.loads(path.read_text()) if path.exists() else []
    for name,data,splits,family in conditions():
        for fold,(available,outer) in enumerate(splits):
            for config in protocol['configs']:
                if any((r['dataset'],r['fold'],r['config'])==(name,fold,config) for r in records):continue
                row=study.fit_one(name,data,fold,available,outer,config,family)
                records.append(row);study.save(path,records)
                print(name,fold,config,'inner',round(row['inner_ndcg'],6),flush=True)
    decisions=[]
    for key in sorted(set((r['dataset'],r['fold']) for r in records)):
        rows=[r for r in records if (r['dataset'],r['fold'])==key]
        if len(rows)!=3:continue
        base=next(r for r in rows if r['config']=='baseline')
        for policy,row in [('baseline',base),('fixed31',next(r for r in rows if r['config']=='leaves31')),
                           ('fixed_query_ranks',next(r for r in rows if r['config']=='query_ranks')),
                           ('raw_rank_selection',study.choose(rows,['baseline','query_ranks']))]:
            decisions.append({'dataset':key[0],'fold':key[1],'family':row['family'],'policy':policy,
                              'selected':row['config'],'outer_ndcg':row['outer_ndcg'],
                              'delta':row['outer_ndcg']-base['outer_ndcg'],'inner_ndcg':row['inner_ndcg']})
    study.save(OUT/'decisions.json',decisions)


if __name__=='__main__':main()
