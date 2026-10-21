"""Staged C1 study using ONLY released TRAIN and generated/toy queries.

No public benchmark caches, DEV, TEST or network. Frozen candidates/policies
are assessed on outer queries that never control stopping or candidate choice.
Run baseline, screening, finalists, then summarize, in that order.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np
from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from scripts.benchmark_c1 import generated
from scripts.lambdamart_utils import default_params, group_sizes
from scripts.ltr_data import load_splits
from scripts.optimize_rankers import cached
from scripts.stress_c1 import peak_memory_bytes
from submission import ranking
from submission.query_features import augment_grouped, feature_names

OUT = ROOT/'runs/final_c1_pass/study'
CONFIGS = {
    'baseline': {},
    'leaves7': {'num_leaves': 7, 'min_data_in_leaf': 20, 'lambda_l2': 0., 'feature_fraction': 1.},
    'leaves31': {'num_leaves': 31},
    'leaves63': {'num_leaves': 63},
    'regular31': {'num_leaves': 31, 'min_data_in_leaf': 100, 'lambda_l2': 5., 'max_bin': 63,
                  'lambdarank_truncation_level': 13},
    'regular63': {'num_leaves': 63, 'min_data_in_leaf': 200, 'lambda_l2': 10., 'max_bin': 63,
                  'lambdarank_truncation_level': 13},
    'fast15': {'max_bin': 63, 'lambdarank_truncation_level': 20},
    'trunc10': {'lambdarank_truncation_level': 10},
    'trunc13': {'lambdarank_truncation_level': 13},
    'rate10': {'learning_rate': .1},
    'slow31': {'num_leaves': 31, 'learning_rate': .03, 'rounds': 1200},
    'feature07': {'feature_fraction': .7},
    'feature10': {'feature_fraction': 1.},
    'no_bagging': {'bagging_fraction': 1., 'bagging_freq': 0},
    'l1gain': {'lambda_l1': .1, 'min_gain_to_split': .01, 'lambda_l2': 5.},
    'bin127col': {'max_bin': 127, 'force_row_wise': False, 'force_col_wise': True},
    'xendcg': {'objective': 'rank_xendcg'},  # diagnostic, excluded from final policies
    'patience20': {'patience': 20},
    'patience35': {'patience': 35},
    'patience75': {'patience': 75},
    'query_ranks': {'transform': 'query_ranks'},
    'sparse_csr': {'matrix': 'csr'},
}
# Predetermined practical selection pool, not chosen using any outer result.
FINALISTS = ['baseline', 'leaves31', 'regular31', 'fast15']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def conditions():
    # Three original transfer sets: original TRAIN supplies fit/inner stopping;
    # supplied validation is used ONLY as the separate outer report here.
    for directory in sorted((ROOT/'data/toy_transfer').iterdir()):
        tr, va = load_splits(directory/'train.txt', directory/'vali.txt')
        X, y, q = [np.concatenate([a, b]) for a, b in zip(tr, va)]
        yield directory.name, (X, y, q), [(np.arange(len(tr[1])), np.arange(len(tr[1]), len(y)))], 'toy'
    for width, queries, density, levels, seed in [(17,240,1.,2,171), (90,360,.1,3,172), (300,240,.02,5,173)]:
        data = generated(width, queries, density, levels, seed)
        yield f'generated_{width}', data, grouped_kfold(data[2],3,41), 'generated'
    X, y, q = cached('train')
    selected = np.random.RandomState(193).permutation(np.unique(q))
    # Valid, predetermined leading feature subsets; no label-based selection.
    for queries in [300,900,2250]:
        rows = np.flatnonzero(np.isin(q, selected[:queries]))
        for width in [40,90,136]:
            data = X[rows,:width], y[rows], q[rows]
            yield f'MSLR_train_q{queries}_d{width}', data, grouped_kfold(data[2],3,41), 'MSLR_TRAIN'


def inner_split(available, q, fold):
    uq = np.random.RandomState(1200+fold).permutation(np.unique(q[available]))
    inner = np.isin(q[available], uq[:max(1,len(uq)//5)])
    training, stopping = available[~inner], available[inner]
    assert not np.intersect1d(q[training],q[stopping]).size
    return training, stopping


def meta(X,y,q):
    groups = group_sizes(q)
    starts = np.r_[0,np.cumsum(groups)]
    return {'rows':len(y), 'queries':len(groups), 'features':X.shape[1],
            'mean_docs':float(np.mean(groups)), 'median_docs':float(np.median(groups)),
            'density':float(np.count_nonzero(X)/X.size),
            'label_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(y,return_counts=True))},
            'multigrade_query_fraction':float(np.mean([np.unique(y[a:b]).size>1 for a,b in zip(starts[:-1],starts[1:])])),
            'constant_features':int(np.sum(np.ptp(X,axis=0)==0))}


def fit_one(name, data, fold, available, outer, config, family):
    X,y,q = data
    training, stopping = inner_split(available,q,fold)
    assert not np.intersect1d(q[outer],q[available]).size
    extra = dict(CONFIGS[config])
    patience, rounds = extra.pop('patience',50), extra.pop('rounds',800)
    transform, matrix = extra.pop('transform',None), extra.pop('matrix',None)
    params = default_params(num_threads=4,**extra)
    path = OUT/'models'/f'{name}_fold{fold}_{config}.txt'
    path.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    inputs=[X[rows] for rows in [training,stopping,outer]]
    if transform:
        inputs=[augment_grouped(x,q[rows]) for x,rows in zip(inputs,[training,stopping,outer])]
    if matrix:
        from scipy.sparse import csr_matrix
        inputs=[csr_matrix(x) for x in inputs]
    names=feature_names(X.shape[1]) if transform else 'auto'
    ds=lgb.Dataset(inputs[0],label=y[training],group=group_sizes(q[training]),feature_name=names)
    dv=lgb.Dataset(inputs[1],label=y[stopping],group=group_sizes(q[stopping]),reference=ds)
    model=lgb.train(params,ds,num_boost_round=rounds,valid_sets=[dv],
                    callbacks=[lgb.early_stopping(patience,first_metric_only=True,verbose=False)])
    fit_seconds=time.perf_counter()-start
    model.save_model(str(path), num_iteration=model.best_iteration)
    loaded=lgb.Booster(model_file=str(path))
    predictions=[loaded.predict(x,num_threads=4) for x in inputs]
    metrics=[mean_ndcg_from_scores(p,y[rows],q[rows]) for p,rows in zip(predictions,[training,stopping,outer])]
    # Staff-compatible serialization/interface, every fit, one complete outer query.
    rows=outer[q[outer]==q[outer][0]]
    ids=[f'{q[rows][0]}:{i}' for i in range(len(rows))]
    ranking.prepare(str(path))
    actual=ranking.rerank_with_ltr(str(q[rows][0]),ids,dict(zip(ids,X[rows])),min(10,len(rows)))
    expected_order=np.argsort(-predictions[2][:len(rows)],kind='stable')[:10]
    interface_ok=[i for i,_ in actual]==[ids[i] for i in expected_order]
    assert interface_ok and all(np.isfinite(p).all() for p in predictions)
    rec={'dataset':name,'family':family,'fold':fold,'config':config,'params':params,
         'patience':patience,'max_rounds':rounds,'transform':transform,'matrix':matrix,
         'train_ndcg':metrics[0],'inner_ndcg':metrics[1],'outer_ndcg':metrics[2],
         'inner_outer_gap':metrics[1]-metrics[2], 'fit_seconds':fit_seconds,
         'total_seconds':time.perf_counter()-start,'best_iteration':model.best_iteration,
         'saved_trees':loaded.num_trees(),'model_bytes':path.stat().st_size,
         'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'peak_process_bytes':peak_memory_bytes(),'matrix_bytes':sum(x.nbytes for x in [X,y,q]),
         'train_rows':len(training),'inner_rows':len(stopping),'outer_rows':len(outer),
         'metadata':meta(X[training],y[training],q[training]),
         'loadable':True,'interface_ok':interface_ok,'success':True}
    return rec


def choose(records, pool, threshold=.001):
    # Predeclared conservative tie: retain the first/simpler candidate unless
    # another improves inner validation by more than .001 nDCG.
    selected=next(r for r in records if r['config']==pool[0])
    for config in pool[1:]:
        row=next(r for r in records if r['config']==config)
        if row['inner_ndcg']>selected['inner_ndcg']+threshold:selected=row
    return selected


def summarize(records):
    frozen_path=OUT.parent/'decision.json'
    final_policy=json.loads(frozen_path.read_text()).get('selected_strategy') if frozen_path.exists() else None
    summaries=[]
    complete=[]
    for key in sorted(set((r['dataset'],r['fold']) for r in records)):
        rows=[r for r in records if (r['dataset'],r['fold'])==key]
        if set(FINALISTS)<=set(r['config'] for r in rows):complete.append(rows)
    strategies={name:[] for name in ['Original recipe','Improved fixed recipe','Adaptive recipe','Validation search','Hybrid policy','Selected final recipe']}
    if all(any(r['config']=='query_ranks' for r in rows) for rows in complete) and complete:
        strategies['Query-relative fixed']=[]
        strategies['Raw/rank validation selection']=[]
    # The improved fixed setting is chosen on INNER scores only, equal weight
    # per condition. All alternatives and this rule were declared before fits.
    fixed=max(FINALISTS,key=lambda c:np.mean([next(r for r in rows if r['config']==c)['inner_ndcg'] for rows in complete])) if complete else 'baseline'
    decisions=[]
    for rows in complete:
        baseline=next(r for r in rows if r['config']=='baseline')
        by={r['config']:r for r in rows}
        adaptive='leaves31' if baseline['metadata']['queries']>=400 else 'baseline'
        search=choose(rows,['baseline','leaves31','regular31'])
        hybrid=choose(rows,['baseline',adaptive]) if adaptive!='baseline' else baseline
        choices={'Original recipe':baseline,'Improved fixed recipe':by[fixed],
                 'Adaptive recipe':by[adaptive],'Validation search':search,
                 'Hybrid policy':hybrid,'Selected final recipe':baseline}
        if 'Query-relative fixed' in strategies:
            choices['Query-relative fixed']=by['query_ranks']
            choices['Raw/rank validation selection']=choose(rows,['baseline','query_ranks'])
            if final_policy=='Raw/rank validation selection':
                choices['Selected final recipe']=choices['Raw/rank validation selection']
        for label,row in choices.items():
            cost=row['total_seconds']
            if label=='Validation search':cost=sum(by[c]['total_seconds'] for c in ['baseline','leaves31','regular31'])
            if label=='Hybrid policy':cost=baseline['total_seconds']+(by[adaptive]['total_seconds'] if adaptive!='baseline' else 0)
            if label=='Raw/rank validation selection':cost=baseline['total_seconds']+by['query_ranks']['total_seconds']
            if label=='Selected final recipe' and final_policy=='Raw/rank validation selection':
                cost=baseline['total_seconds']+by['query_ranks']['total_seconds']
            strategies[label].append((row,baseline,cost))
            decisions.append({'dataset':row['dataset'],'fold':row['fold'],'strategy':label,
                              'selected':row['config'],'inner_ndcg':row['inner_ndcg'],'outer_ndcg':row['outer_ndcg'],
                              'baseline_outer':baseline['outer_ndcg'],'delta':row['outer_ndcg']-baseline['outer_ndcg'],
                              'total_seconds':cost})
    for label,items in strategies.items():
        if not items:continue
        values=np.array([r['outer_ndcg'] for r,b,s in items]);deltas=np.array([r['outer_ndcg']-b['outer_ndcg'] for r,b,s in items])
        summaries.append({'strategy':label,'conditions':len(items),'outer_mean':float(values.mean()),
                          'outer_std':float(values.std()),'mean_paired_delta':float(deltas.mean()),
                          'worst_regression':float(deltas.min()),'wins':int(np.sum(deltas>1e-12)),
                          'mean_seconds':float(np.mean([s for r,b,s in items])),
                          'mean_inner':float(np.mean([r['inner_ndcg'] for r,b,s in items])),
                          'mean_train':float(np.mean([r['train_ndcg'] for r,b,s in items])),
                          'successes':sum(r['success'] for r,b,s in items)})
    save(OUT/'summary.json',{'fixed_selected_by_inner':fixed,'strategies':summaries,'decisions':decisions})
    import csv
    if decisions:
        with (OUT/'strategies.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(decisions[0]));writer.writeheader();writer.writerows(decisions)
    print(json.dumps(summaries,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['baseline','screening','finalists','transforms','summarize'])
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'seed':0,'outer_seed':41,'pool_seed':193,'inner_seed':'1200+fold',
              'allowed_sources':['released MSLR TRAIN','supplied synthetic transfer','generated'],
              'configs':CONFIGS,'finalists':FINALISTS,'search':['baseline','leaves31','regular31'],
              'adaptive_rule':'31 leaves for >=400 gradient queries; baseline otherwise',
              'tie_threshold':.001,'lightgbm':lgb.__version__,
              'note':'Outer folds are never inputs to stopping or candidate selection; correlated conditions are not independent replications.'}
    if (OUT/'protocol.json').exists():
        if json.loads((OUT/'protocol.json').read_text())!=protocol:raise ValueError('Protocol changed; use a new output directory')
    else:save(OUT/'protocol.json',protocol)
    path=OUT/'fits.json'
    records=json.loads(path.read_text()) if path.exists() else []
    if args.stage=='summarize':summarize(records);return
    for name,data,splits,family in conditions():
        for fold,(available,outer) in enumerate(splits):
            # Screening subset fixed before any metrics: all toy, first generated
            # fold, and first MSLR fold with full raw width at each query size.
            if args.stage=='screening' and not (family=='toy' or (fold==0 and (family=='generated' or name.endswith('d136')))):continue
            configs=(['baseline'] if args.stage=='baseline' else list(CONFIGS) if args.stage=='screening'
                     else ['query_ranks'] if args.stage=='transforms' else FINALISTS)
            for config in configs:
                if any((r['dataset'],r['fold'],r['config'])==(name,fold,config) for r in records):continue
                rec=fit_one(name,data,fold,available,outer,config,family)
                records.append(rec);save(path,records)
                # Log inner only until all fits/policy rules are frozen.
                print(name,fold,config,'inner',round(rec['inner_ndcg'],6),'seconds',round(rec['total_seconds'],2),flush=True)
                gc.collect()
    if args.stage in ['finalists','transforms']:summarize(records)


if __name__=='__main__':main()
