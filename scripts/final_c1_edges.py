"""Deterministic generated boundary workloads, never real evaluation data."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import lightgbm as lgb
import numpy as np
from scripts.stress_c1 import peak_memory_bytes
from submission import recipe,ranking


def main():
    root=ROOT/'tmp/final_c1_edges';root.mkdir(parents=True,exist_ok=True)
    records=[]
    cases=[('large_groups',4,5000,17),('many_tiny_queries',5000,2,300),
           ('zero_constant_features',100,12,90),('validation_only_indices',100,10,300)]
    for name,queries,docs,width in cases:
        rng=np.random.RandomState(217)
        directory=root/name;directory.mkdir(exist_ok=True)
        for split,nqueries in [('train',queries),('vali',max(10,queries//10))]:
            path=directory/(split+'.txt')
            with path.open('w') as stream:
                for query in range(nqueries):
                    for row in range(docs):
                        label=int(row%3)
                        if name=='zero_constant_features':features=f'1:0 {width}:1'
                        elif name=='validation_only_indices':features=f'1:{label+rng.randn():.8g}'+(f' {width}:1' if split=='vali' else '')
                        else:features=f'1:{label+rng.randn():.8g} {width}:{rng.randn():.8g}'
                        stream.write(f'{label} qid:{query*17+9} {features} # generated edge\n')
        counts,tw=recipe._scan(directory/'train.txt')
        vc,vw=recipe._scan(directory/'vali.txt')
        X,y,g=recipe._read(directory/'train.txt',counts,max(tw,vw))
        assert sum(g)==len(y) and len(g)==queries and all(size==docs for size in g)
        start=time.monotonic();path=directory/'model.txt'
        recipe.train(str(directory/'train.txt'),str(directory/'vali.txt'),str(path),0)
        elapsed=time.monotonic()-start
        ranking.prepare(str(path))
        model=lgb.Booster(model_file=str(path))
        result=ranking.rerank_with_ltr('q',['a','b'],{'a':[0.]*width,'b':[1.]*width},2)
        assert len(result)==2 and np.isfinite([s for _,s in result]).all()
        ids=[f'opaque:{i}' for i in range(docs)]
        infer_start=time.monotonic()
        full=ranking.rerank_with_ltr('large',ids,dict(zip(ids,X[:docs])),min(10,docs))
        inference_seconds=time.monotonic()-infer_start
        assert len(full)==min(10,docs) and len(set(doc for doc,_ in full))==len(full)
        records.append({'workload':name,'train_rows':queries*docs,'queries':queries,'max_group':docs,
                        'width':width,'seconds':elapsed,'trees':model.num_trees(),
                        'peak_process_bytes':peak_memory_bytes(),'interface_ok':True,
                        'groups_verified':True,'inference_seconds':inference_seconds,
                        'inference_candidates':docs,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        print(json.dumps(records[-1]),flush=True)
    (ROOT/'runs/final_c1_pass/edges.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
