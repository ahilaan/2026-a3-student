"""Merge verified TRAIN+DEV-only trees and preserve their provenance chain."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import lightgbm as lgb
from scripts.blend_trees import blend_models
from scripts.compliant_package import verify_model,digest,OUT
from scripts.score_campaign3_union import lift_pair


def main():
    global OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--first',required=True,help='First training/blend receipt')
    ap.add_argument('--second',required=True,help='Second training/blend receipt')
    ap.add_argument('--first-weight',type=float,required=True)
    ap.add_argument('--name',required=True)
    ap.add_argument('--union',action='store_true',help='First rank, second minmax; merge feature schemas')
    ap.add_argument('--output-dir',type=Path,default=OUT)
    a=ap.parse_args()
    OUT=a.output_dir.resolve()
    if not OUT.is_relative_to(ROOT):ap.error('Output directory must be inside the repository')
    (OUT/'models').mkdir(parents=True,exist_ok=True)
    first,ra=verify_model(a.first);second,rb=verify_model(a.second)
    target=OUT/'models'/(a.name+'.txt')
    if target.exists():raise FileExistsError(target)
    if a.union:
        models=lift_pair(lgb.Booster(model_file=str(first)),lgb.Booster(model_file=str(second)))
        paths=[target.with_suffix('.rank.txt'),target.with_suffix('.minmax.txt')]
        for m,p in zip(models,paths):m.save_model(str(p))
        first,second=paths
    model=blend_models(first,second,weight=a.first_weight)
    model.save_model(str(target))
    command=f'python scripts/compliant_blend.py --first {a.first} --second {a.second} --first-weight {a.first_weight} --name {a.name}'
    if a.union:command+=' --union'
    import shlex
    command+=' --output-dir '+shlex.quote(str(OUT))
    def reproduction(record):
        if 'reproduction' in record:return record['reproduction']
        b=record['args'];cmd=f"python scripts/compliant_train.py full --name {b['name']} --config {b['config']} --rounds {b['rounds']} --normalization {b['normalization']} --seed {b['seed']}"
        for flag in ['augment','group_dropout','rank_only']:
            if b[flag]:cmd+=' --'+flag.replace('_','-')
        cmd+=' --threads '+str(b.get('threads',4))
        return [cmd]
    commands=list(dict.fromkeys(reproduction(ra)+reproduction(rb)+[command]))
    record={'training_splits':'Fold1 TRAIN+DEV only','model':target.relative_to(ROOT).as_posix(),
            'sha256':digest(target),'constituent_receipts':[Path(p).resolve().relative_to(ROOT).as_posix() for p in [a.first,a.second]],
            'first_weight':a.first_weight,'union':a.union,'trees':model.num_trees(),
            'features':model.num_feature(),'reproduction':commands}
    (OUT/(a.name+'.json')).write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
