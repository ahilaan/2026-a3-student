"""Select channel powers/weights on TRAIN; freeze the rule before DEV measurement.

This uses all TRAIN queries for tuning. It is repeated development research,
not an independent confirmation test. Neither DEV nor TEST is loaded here.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from harness.feature_names import BASE_RANKER_CHANNELS
from scripts.optimize_rankers import cached

OUT=ROOT/'runs/compliant_campaign'


def groups():
    X,y,q=cached('train')
    starts=np.r_[0,np.flatnonzero(q[1:]!=q[:-1])+1,len(q)]
    result=[];columns=np.array(list(BASE_RANKER_CHANNELS.values()))-1
    for lo,hi in zip(starts[:-1],starts[1:]):
        raw=np.array(X[lo:hi,columns],dtype=np.float64)
        span=np.ptp(raw,axis=0);norm=(raw-raw.min(axis=0))/np.where(span>0,span,1.)
        gain=2.**y[lo:hi]-1.;rel=y[lo:hi]>0
        disc=1/np.log2(np.arange(2,min(hi-lo,10)+2))
        ideal=np.sort(gain)[::-1][:10]@disc
        result.append((norm,gain,rel,disc,max(ideal,1e-15)))
    return result


def evaluate(data,choices):
    weights=np.array([c['weights'] for c in choices]);powers=np.array([c['powers'] for c in choices])
    totals=np.zeros((len(choices),2))
    for x,gain,rel,disc,ideal in data:
        scores=np.sum(x[:,None,:]**powers[None,:,:]*weights[None,:,:],axis=2)
        order=np.argsort(-scores,axis=0,kind='stable')[:10]
        totals[:,0]+=(gain[order]*disc[:,None]).sum(axis=0)/ideal
        if rel.sum():
            hit=rel[order];totals[:,1]+=(hit*np.cumsum(hit,axis=0)/np.arange(1,len(disc)+1)[:,None]).sum(axis=0)/rel.sum()
    return totals/len(data)


def main():
    data=groups();baseline={'weights':[2.,3.,3.,0.],'powers':[1.5]*4}
    baseline_metrics=evaluate(data,[baseline])[0]
    seeds=[baseline,{'weights':[2.,3.,3.375,0.],'powers':[1.5,1.,.75,1.5]}]
    report={'protocol':__doc__,'baseline':baseline,'baseline_TRAIN':baseline_metrics.tolist(),
            'metric_order':['ndcg@10','map@10'],'selection':'Maximize .8*nDCG+.2*MAP across TRAIN, requiring both TRAIN means to meet the baseline; two predefined starts and two coordinate sweeps. No DEV selection.',
            'steps':[],'selected':baseline,'selected_TRAIN':baseline_metrics.tolist()}
    best_value=float(baseline_metrics@np.array([.8,.2]));best=baseline
    for seed_index,seed in enumerate(seeds):
        chosen=seed
        for sweep in range(2):
            for col in range(3):
                values=sorted(set([chosen['weights'][col]] if col==0 else [0.,.5,1.,1.5,2.,2.5,3.,3.5,4.,5.,6.]+[chosen['weights'][col]*v for v in [.625,.75,.875,1.125,1.25,1.5]]))
                choices=[baseline,chosen]
                for weight in values:
                    for power in [.5,.625,.75,.875,1.,1.125,1.25,1.5,1.75,2.,2.5]:
                        w,p=list(chosen['weights']),list(chosen['powers']);w[col]=weight;p[col]=power
                        choices.append({'weights':w,'powers':p})
                metrics=evaluate(data,choices)
                eligible=(metrics[:,0]>=baseline_metrics[0]-1e-12)&(metrics[:,1]>=baseline_metrics[1]-1e-12)
                objective=metrics@np.array([.8,.2]);objective[~eligible]=-np.inf
                ix=int(np.argmax(objective));chosen=choices[ix]
                if objective[ix]>best_value:
                    best_value=float(objective[ix]);best=chosen
                    report['selected']=best;report['selected_TRAIN']=metrics[ix].tolist()
                row={'start':seed_index,'sweep':sweep,'column':col,'choice':chosen,'TRAIN':metrics[ix].tolist()}
                report['steps'].append(row)
                (OUT/'fusion_train_selection.json').write_text(json.dumps(report,indent=2))
                print(row,flush=True)
    print('Frozen selection:',json.dumps(report['selected']),'TRAIN:',report['selected_TRAIN'],flush=True)


if __name__=='__main__':main()
