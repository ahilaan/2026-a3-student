"""Select a bounded plain-MLP ensemble on gradient-excluded DEV queries.

Both studies used TRAIN+1600 DEV query gradients; all400 reserved queries
are excluded from gradients. Repeated DEV selection is development, not a
fresh final test. Final networks fit all TRAIN+DEV; no TEST labels are read.
"""
from pathlib import Path
import sys
import json
import hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from harness.neural import load_scorer,save_scorer
from harness.metrics import mean_ndcg_from_scores
from scripts.optimize_rankers import cached
from scripts.blend_neural import blend


def scores(path,x):
    scorer=load_scorer(str(path))
    return np.concatenate([scorer(x[i:i+4096]) for i in range(0,len(x),4096)])


def main():
    torch.set_num_threads(2);out=ROOT/'runs/compliant_campaign'
    x,y,q=cached('dev');reserved=np.random.RandomState(764).permutation(np.unique(q))[:400]
    mask=np.isin(q,reserved);x,y,q=np.asarray(x[mask]),y[mask],q[mask]
    studies=[ROOT/'runs/compliant_campaign/models/neural_dev_study/pairwise_model.pt',
             ROOT/'runs/compliant_campaign/models/campaign3_fresh23_study/pairwise_model.pt']
    a,b=[scores(p,x) for p in studies]
    values=[{'old_weight':w,'reserved_DEV':mean_ndcg_from_scores(w*a+(1-w)*b,y,q)} for w in [0.,.05,.1,.2,.3,.4,.5,.6,.75,1.]]
    selected=max(values,key=lambda r:r['reserved_DEV']);baseline=max(values[0]['reserved_DEV'],values[-1]['reserved_DEV'])
    accepted=bool(0<selected['old_weight']<1 and selected['reserved_DEV']>=baseline+.001)
    result={'protocol':__doc__,'studies':[str(p.relative_to(ROOT)) for p in studies],
            'score_std':[float(a.std()),float(b.std())],'candidates':values,'selected':selected,
            'best_single':baseline,'accepted':accepted,'standardization':'Identical frozen original TRAIN-only statistics',
            'method':'Copy label-trained parameters into concatenated hidden layers/block diagonal connections; no distillation or model-score training.'}
    if accepted:
        final=[ROOT/'runs/compliant_campaign/models/neural_dev_full9/pairwise_model.pt',
               ROOT/'runs/compliant_campaign/models/campaign3_fresh23_full6/pairwise_model.pt']
        network,std=blend(*final,first_weight=selected['old_weight'])
        p=out/'models/pairwise_ensemble.pt';save_scorer(str(p),network,std)
        expected=selected['old_weight']*scores(final[0],x)+(1-selected['old_weight'])*scores(final[1],x)
        actual=scores(p,x);assert np.allclose(actual,expected,rtol=1e-5,atol=1e-5)
        result.update(final_sources=[str(p.relative_to(ROOT)) for p in final],model=str(p.relative_to(ROOT)),
                      hidden=network.hidden_sizes,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                      float32_max_score_difference=float(np.max(np.abs(actual-expected))))
    (out/'neural_ensemble.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
