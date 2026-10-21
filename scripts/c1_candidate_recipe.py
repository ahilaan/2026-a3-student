"""Experimental validation-search recipe; not automatically deployed.

Only the two supplied files are read. One shared soft deadline includes
parsing. Baseline is saved before optional alternatives. Time-based stopping
is an emergency guard, not a promise of cross-machine byte equality.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import lightgbm as lgb
import numpy as np
from harness.metrics import mean_ndcg_from_scores
from scripts.lambdamart_utils import default_params
from submission.recipe import _scan, _read, _TrainingDeadline, _bounded_groups, _whole_query_metric
from submission.query_features import augment_grouped, feature_names

CANDIDATES=[('baseline',{}),('leaves31',{'num_leaves':31}),
            ('regular31',{'num_leaves':31,'min_data_in_leaf':100,'lambda_l2':5.,
                          'max_bin':63,'lambdarank_truncation_level':13})]


def _score(model,X,y,groups):
    q=np.repeat(np.arange(len(groups)),groups)
    return mean_ndcg_from_scores(model.predict(X,num_threads=4),y,q)


def _save(model,path):
    temporary=str(path)+'.candidate.tmp'
    model.save_model(temporary,num_iteration=model.best_iteration)
    os.replace(temporary,path)


def train(train_path,vali_path,out_model_path,seed=0,*,policy='search',budget_seconds=960.,rounds=800):
    started=time.monotonic();deadline=started+budget_seconds
    tc,tw=_scan(train_path);vc,vw=_scan(vali_path);width=max(tw,vw)
    if not tc or not vc or width==0:raise ValueError('Nonempty TRAIN/validation and feature schema required')
    X,y,g=_read(train_path,tc,width,saturate=True);Xv,yv,gv=_read(vali_path,vc,width,saturate=True)
    parse_seconds=time.monotonic()-started
    native_g,native_gv=_bounded_groups(g),_bounded_groups(gv)
    oversized=native_g!=g or native_gv!=gv
    evaluation=_whole_query_metric(yv,gv) if oversized else None
    pool=([('baseline',{}),('query_ranks',{})] if policy=='raw_rank' else
          CANDIDATES if policy=='search' else CANDIDATES[:2])
    best=None;best_score=-float('inf');fits=[];selected=None
    for name,overrides in pool:
        now=time.monotonic();remaining=deadline-now
        # Never start another dataset construction with a tiny remaining budget.
        if best is not None and remaining<max(30.,3*fits[0]['seconds']):break
        params=default_params(seed=seed,num_threads=4,**overrides)
        if oversized:params['metric']='None'
        Xt,Xvt,names=X,Xv,'auto'
        if name=='query_ranks':
            Xt=augment_grouped(X,np.repeat(np.arange(len(g)),g))
            Xvt=augment_grouped(Xv,np.repeat(np.arange(len(gv)),gv))
            names=feature_names(width)
        ds=lgb.Dataset(Xt,label=y,group=native_g,feature_name=names)
        dv=lgb.Dataset(Xvt,label=yv,group=native_gv,reference=ds)
        # Give baseline an ample share; it is normally finished by early stopping.
        candidate_deadline=deadline if best is not None else min(deadline,now+max(30.,remaining*.55))
        phase_rounds=min(200,rounds) if policy=='successive' else rounds
        model=lgb.train(params,ds,num_boost_round=phase_rounds,valid_sets=[dv],feval=evaluation,
                        callbacks=[_TrainingDeadline(candidate_deadline),
                                   lgb.early_stopping(50,first_metric_only=True,verbose=False)])
        score=_score(model,Xvt,yv,gv)
        fits.append({'candidate':name,'inner_ndcg':score,'trees':model.num_trees(),
                     'seconds':time.monotonic()-now})
        if best is None or score>best_score+.001:
            best,best_score,selected=model,score,name
            _save(best,out_model_path)
        del ds,dv,Xt,Xvt
    if policy=='successive' and time.monotonic()+max(30.,3*fits[0]['seconds'])<deadline:
        # Continue only the validation winner of the bounded first stage.
        # The initial best is retained if continuation does not improve it.
        overrides=dict(next(p for n,p in CANDIDATES if n==selected))
        params=default_params(seed=seed,num_threads=4,**overrides)
        if oversized:params['metric']='None'
        ds=lgb.Dataset(X,label=y,group=native_g)
        dv=lgb.Dataset(Xv,label=yv,group=native_gv,reference=ds)
        start=time.monotonic()
        extended=lgb.train(params,ds,num_boost_round=max(1,rounds-best.num_trees()),init_model=best,
                            valid_sets=[dv],feval=evaluation,callbacks=[_TrainingDeadline(deadline),
                                    lgb.early_stopping(50,first_metric_only=True,verbose=False)])
        score=_score(extended,Xv,yv,gv)
        fits.append({'candidate':selected+'_continued','inner_ndcg':score,
                     'trees':extended.num_trees(),'seconds':time.monotonic()-start})
        if score>best_score+.001:
            best,best_score=extended,score
            _save(best,out_model_path)
    return {'policy':policy,'selected':selected,'fits':fits,'parse_seconds':parse_seconds,
            'seconds':time.monotonic()-started,'budget_seconds':budget_seconds}


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',required=True);ap.add_argument('--vali',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--report',required=True)
    ap.add_argument('--budget-seconds',type=float,default=960.)
    ap.add_argument('--policy',choices=['search','two','successive','raw_rank'],default='search')
    a=ap.parse_args()
    result=train(a.train,a.vali,a.out,budget_seconds=a.budget_seconds,policy=a.policy)
    Path(a.report).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
