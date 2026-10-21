"""Report the frozen production fusion on DEV; never select or alter a rule."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from harness.metrics import evaluate_run
from harness.feature_names import BASE_RANKER_CHANNELS
from scripts.optimize_rankers import cached
from submission import ranking


def main():
    X,y,q=cached('dev');run={};qrels={}
    starts=np.r_[0,np.flatnonzero(q[1:]!=q[:-1])+1,len(q)]
    for lo,hi in zip(starts[:-1],starts[1:]):
        key=str(q[lo]);ids=[f'{key}:{i}' for i in range(hi-lo)]
        base={doc:{name:float(row[col-1]) for name,col in BASE_RANKER_CHANNELS.items()}
              for doc,row in zip(ids,X[lo:hi])}
        run[key]=ranking.fuse_candidates(key,ids,base,10)
        qrels[key]={doc:int(label) for doc,label in zip(ids,y[lo:hi])}
    result=evaluate_run(run,qrels,k=10)
    (ROOT/'runs/compliant_campaign/fusion_frozen_dev.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='per_query'},indent=2),flush=True)


if __name__=='__main__':main()
