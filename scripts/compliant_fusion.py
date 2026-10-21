"""Compare handwritten fusion rules using TRAIN only; freeze before DEV reporting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from harness.feature_names import BASE_RANKER_CHANNELS
from scripts.optimize_rankers import cached


def candidates():
    weights=[[2.,3.,3.,0.],[1.,1.,1.,1.],[1.,1.,1.,0.],[1.,2.,2.,0.],[2.,1.,2.,0.]]
    return [{'kind':'power','parameter':1.5,'weights':weights[0]}]+[
        {'kind':kind,'parameter':p,'weights':w}
        for kind,grid in [('power',[.5,1.,1.5,2.,3.]),('rrf',[5.,20.,60.,1000.]),('mnz',[1.])]
        for p in grid for w in weights]


def evaluate(split,rules):
    X,y,q=cached(split)
    starts=np.r_[0,np.flatnonzero(q[1:]!=q[:-1])+1,len(q)]
    out=np.zeros((len(starts)-1,len(rules),2))
    columns=np.array(list(BASE_RANKER_CHANNELS.values()))-1
    for qi,(lo,hi) in enumerate(zip(starts[:-1],starts[1:])):
        raw=np.array(X[lo:hi,columns],dtype=np.float64)
        span=np.ptp(raw,axis=0);norm=(raw-raw.min(axis=0))/np.where(span>0,span,1.)
        ranks=np.empty_like(raw)
        for col in range(raw.shape[1]):
            order=np.argsort(-raw[:,col],kind='stable');ranks[order,col]=np.arange(1,len(raw)+1)
        scores=[]
        for rule in rules:
            w=np.array(rule['weights']);p=rule['parameter']
            if rule['kind']=='rrf':score=(1/(p+ranks))@w
            else:
                score=(norm**p)@w
                if rule['kind']=='mnz':score*=((norm>0)&(w>0)).sum(axis=1)
            scores.append(score)
        order=np.argsort(-np.column_stack(scores),axis=0,kind='stable')[:10]
        gains=2.**y[lo:hi]-1.;rel=y[lo:hi]>0
        disc=1/np.log2(np.arange(2,len(order)+2));ideal=np.sort(gains)[::-1][:10]@disc
        if ideal>0:out[qi,:,0]=(gains[order]*disc[:,None]).sum(axis=0)/ideal
        if rel.sum():
            hit=rel[order];out[qi,:,1]=(hit*np.cumsum(hit,axis=0)/np.arange(1,len(order)+1)[:,None]).sum(axis=0)/rel.sum()
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--objective',choices=['combined','map'],default='combined')
    args=ap.parse_args()
    rules=candidates();values=evaluate('train',rules)
    indices=np.random.RandomState(21020).permutation(len(values));tune,confirm=indices[:4000],indices[4000:]
    metric=np.array([.8,.2] if args.objective=='combined' else [0.,1.]);means=values[tune].mean(axis=0)
    selected=int(np.argmax(means@metric));delta=(values[confirm,selected]-values[confirm,0])@metric
    rng=np.random.RandomState(0)
    lower=float(np.quantile([delta[rng.randint(len(delta),size=len(delta))].mean() for _ in range(2000)],.025))
    accepted=bool(selected and delta.mean()>.0005 and lower>0)
    if args.objective=='map':
        accepted=accepted and bool(values[confirm,selected,0].mean()>=values[confirm,0,0].mean()-.0015)
    result={'protocol':'4000 TRAIN tuning /2000 TRAIN confirmation seed21020. TRAIN selects and accepts; DEV reporting follows the frozen decision. Repeated TRAIN research is adaptive development.',
            'objective':args.objective,'selected':rules[selected],'accepted':accepted,'metric_order':['ndcg@10','map@10'],
            'train_tuning':[means[0].tolist(),means[selected].tolist()],
            'train_confirmation':values[confirm][:,[0,selected]].mean(axis=0).tolist(),
            'confirmation_objective_gain':float(delta.mean()),'bootstrap_lower':lower,
            'train_rule_metrics':[dict(r,metrics=values[:,i].mean(axis=0).tolist()) for i,r in enumerate(rules)],
            'DEV_reporting_only':evaluate('dev',[rules[0],rules[selected],{'kind':'rrf','parameter':60.,'weights':[1.,1.,1.,1.]}]).mean(axis=0).tolist(),
            'production_changed':False}
    name='fusion_study' if args.objective=='combined' else 'fusion_map_study'
    (ROOT/'runs/compliant_campaign'/(name+'.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='train_rule_metrics'},indent=2),flush=True)


if __name__=='__main__':main()
