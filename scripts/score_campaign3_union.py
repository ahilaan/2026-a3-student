"""Exactly combine numeric LightGBM models with two declared query transforms.

Tree feature indices are remapped into raw/qrank/qmin columns. No fitting,
new labels, dataset identity switches or custom model format are introduced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import lightgbm as lgb
import numpy as np
from scripts.blend_trees import blend_models


def _infos(model):
    text=model.model_to_string()
    return dict(zip(model.feature_name(),re.search(r'(?m)^feature_infos=(.*)$',text).group(1).split()))


def _merge_info(first,second):
    if first=='none':return second
    if second=='none':return first
    values=[]
    for value in [first,second]:
        match=re.fullmatch(r'\[([^:]+):([^\]]+)\]',value)
        if not match:raise ValueError('Only numeric feature models are supported')
        values.append(tuple(map(float,match.groups())))
    return f'[{min(v[0] for v in values):.17g}:{max(v[1] for v in values):.17g}]'


def lift(model,names,infos):
    """Rebuild tree byte offsets after an exact numeric feature-index remap."""
    original=model.feature_name();mapping=[names.index(n) for n in original]
    source=model.model_to_string();start=source.index('Tree=0\n');end=source.index('end of trees',start)
    header,tail=source[:start],source[end:]
    blocks=[b for b in re.split(r'(?m)(?=^Tree=\d+\n)',source[start:end]) if b]
    sizes=list(map(int,re.search(r'(?m)^tree_sizes=(.*)$',header).group(1).split()))
    assert sizes==[len(b.encode('utf-8')) for b in blocks]
    rewritten=[]
    for block in blocks:
        block=re.sub(r'(?m)^split_feature=(.*)$',lambda m:'split_feature='+' '.join(str(mapping[int(i)]) for i in m.group(1).split()),block)
        rewritten.append(block)
    replacements={'max_feature_idx':str(len(names)-1),'feature_names':' '.join(names),
                  'feature_infos':' '.join(infos[n] for n in names),
                  'tree_sizes':' '.join(str(len(b.encode('utf-8'))) for b in rewritten)}
    for key,value in replacements.items():header=re.sub(r'(?m)^'+key+r'=.*$',lambda m:key+'='+value,header)
    result=lgb.Booster(model_str=header+''.join(rewritten)+tail)
    assert result.feature_name()==names and result.num_trees()==model.num_trees()
    return result


def lift_pair(rank_model,minmax_model):
    a,b=rank_model.feature_name(),minmax_model.feature_name()
    if len(a)!=len(b) or len(a)%2:raise ValueError('Matching two-block schemas required')
    width=len(a)//2
    raw=[f'raw_{i}' for i in range(width)];rank=[f'qrank_{i}' for i in range(width)];qmin=[f'qmin_{i}' for i in range(width)]
    if a!=raw+rank or b!=raw+qmin:raise ValueError('Expected explicit raw/qrank and raw/qmin models')
    ia,ib=_infos(rank_model),_infos(minmax_model);names=raw+rank+qmin
    infos={n:_merge_info(ia.get(n,'none'),ib.get(n,'none')) for n in names}
    return lift(rank_model,names,infos),lift(minmax_model,names,infos)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rank',required=True);ap.add_argument('--minmax',required=True)
    ap.add_argument('--minmax-weight',type=float,default=.5);ap.add_argument('--out',required=True)
    a=ap.parse_args();target=Path(a.out);target.parent.mkdir(exist_ok=True,parents=True)
    original_rank=lgb.Booster(model_file=a.rank);original_minmax=lgb.Booster(model_file=a.minmax)
    rank,mm=lift_pair(original_rank,original_minmax)
    rp=target.with_suffix('.rank_lifted.txt');mp=target.with_suffix('.minmax_lifted.txt')
    rank.save_model(str(rp));mm.save_model(str(mp))
    merged=blend_models(mp,rp,weight=a.minmax_weight);merged.save_model(str(target))
    from scripts.cv_query_features import data
    from submission.ranking import _augment_query_minmax
    x,y,q=data('dev');starts=np.r_[0,np.flatnonzero(q[1:]!=q[:-1])+1,len(q)]
    limit=starts[min(30,len(starts)-1)];width=x.shape[1]//2
    minmax=np.concatenate([_augment_query_minmax(x[lo:hi,:width]) for lo,hi in zip(starts[:-1],starts[1:]) if hi<=limit])
    union=np.column_stack([x[:limit],minmax[:,width:]])
    predictions=[original_rank.predict(x[:limit],num_threads=2),original_minmax.predict(minmax,num_threads=2)]
    lifting_error=max(float(np.max(np.abs(rank.predict(union,num_threads=2)-predictions[0]))),float(np.max(np.abs(mm.predict(union,num_threads=2)-predictions[1]))))
    merge_error=float(np.max(np.abs(merged.predict(union,num_threads=2)-((1-a.minmax_weight)*predictions[0]+a.minmax_weight*predictions[1]))))
    assert lifting_error==0 and merge_error<1e-12
    report={'rank':a.rank,'minmax':a.minmax,'minmax_weight':a.minmax_weight,'rank_weight':1-a.minmax_weight,
            'features':merged.num_feature(),'trees':merged.num_trees(),'complete_DEV_queries_checked':30,
            'rows_checked':int(limit),'lifting_max_error':lifting_error,'merge_max_error':merge_error,
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'training':'Constituents must be independently verified TRAIN+DEV-only fits; no additional fitting.',
            'runtime_support_required':'Explicit raw/qrank/qmin union schema, dynamic width'}
    target.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
