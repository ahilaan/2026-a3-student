"""Verify final production artifacts equal the evaluated selector on 9 cases."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from submission import recipe,ranking


def main():
    out=ROOT/'runs/final_c1_pass'
    rows=json.loads((out/'profiles/results.json').read_text())
    records=[]
    for row in rows:
        if row['policy']!='raw_rank':continue
        directory=ROOT/'tmp/final_c1_profiles'/row['dataset']
        model=directory/'production.txt'
        started=time.monotonic()
        recipe.train(str(directory/'train.txt'),str(directory/'vali.txt'),str(model),0)
        digest=hashlib.sha256(model.read_bytes()).hexdigest()
        assert digest==row['sha256'],row['dataset']
        ranking.prepare(str(model))
        width=ranking._RAW_WIDTH
        assert len(ranking.rerank_with_ltr('q',['a'],{'a':[0.]*width},1))==1
        records.append({'dataset':row['dataset'],'byte_equal_to_evaluated_policy':True,
                        'selected':row['details']['selected'],'seconds':time.monotonic()-started,
                        'sha256':digest,'model_features':ranking._MODEL.num_feature()})
        print(json.dumps(records[-1]),flush=True)
    (out/'production_api_agreement.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
