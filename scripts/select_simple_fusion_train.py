"""TRAIN-only one-standard-error simplification of the fine fusion search.

Among textbook equal-weight CombSUM channel subsets, retain the most channels
whose TRAIN nDCG and MAP are both within one query standard error of the fine
TRAIN-selected rule; break ties by .8*nDCG+.2*MAP. No DEV/TEST is loaded.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import itertools
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tune_fusion_train import groups


def main():
    out=ROOT/'runs/compliant_campaign'
    fine=json.loads((out/'fusion_train_selection.json').read_text())
    rules=[fine['selected']]+[
        {'weights':list(map(float,w)),'powers':[1.]*4}
        for w in itertools.product([0,1],repeat=4) if sum(w)>=2]
    data=groups();results=np.zeros((len(data),len(rules),2))
    weights=np.array([r['weights'] for r in rules]);powers=np.array([r['powers'] for r in rules])
    for i,(x,gain,rel,disc,ideal) in enumerate(data):
        scores=np.sum(x[:,None,:]**powers[None,:,:]*weights[None,:,:],axis=2)
        order=np.argsort(-scores,axis=0,kind='stable')[:10]
        results[i,:,0]=(gain[order]*disc[:,None]).sum(axis=0)/ideal
        if rel.sum():
            hit=rel[order];results[i,:,1]=(hit*np.cumsum(hit,axis=0)/np.arange(1,len(disc)+1)[:,None]).sum(axis=0)/rel.sum()
    means=results.mean(axis=0);sem=results[:,0].std(axis=0,ddof=1)/np.sqrt(len(data))
    eligible=[i for i in range(1,len(rules)) if np.all(means[i]>=means[0]-sem)]
    selected=max(eligible,key=lambda i:(sum(np.array(rules[i]['weights'])>0),float(means[i]@np.array([.8,.2])))) if eligible else 0
    report={'protocol':__doc__,'metric_order':['ndcg@10','map@10'],
            'fine_rule':rules[0],'fine_TRAIN':means[0].tolist(),'query_standard_error':sem.tolist(),
            'selected':rules[selected],'selected_TRAIN':means[selected].tolist(),
            'eligible_classic_rules':[dict(rule=rules[i],TRAIN=means[i].tolist()) for i in eligible],
            'DEV_used_for_selection':False,'TEST_used':False}
    (out/'fusion_simple_selection.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
