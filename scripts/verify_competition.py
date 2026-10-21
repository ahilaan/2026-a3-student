"""Build the competition ZIP and verify a disposable extraction; never submit.

The production scoring files and weights are immutable. Only temporary C1 and
toy smoke artifacts are fitted; the expensive main-model refit is not rerun.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_submission import REQUIRED, build, source_files

OUT = ROOT / 'runs/competition_package'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args, cwd, log):
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    with (OUT/log).open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable]+args, cwd=cwd, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{log}: exit {result.returncode}; inspect the log')
    return (OUT/log).read_text(encoding='utf-8')


SMOKE = r'''
import json,math
from pathlib import Path
import numpy as np
import lightgbm as lgb
import submission.ranking as ranking
from harness.neural import load_scorer
from harness.run_harness import check_conformance
from harness.recipe_eval import run_recipe_and_evaluate
assert Path(ranking.__file__).resolve().is_relative_to(Path.cwd())
assert check_conformance(ranking,'submission/model.txt') == []
ranking.prepare('submission/model.txt')
model=lgb.Booster(model_file='submission/model.txt')
assert (model.num_trees(),model.num_feature()) == (4400,272)
X=np.random.RandomState(764).normal(size=(25,136)).astype(np.float32)
ids=[f'opaque-{i}' for i in range(len(X))]
features=dict(zip(ids,X))
def valid(result):
 assert len(result)==10 and len({d for d,s in result})==10
 assert all(d in ids and math.isfinite(s) for d,s in result)
a=ranking.rerank_with_ltr('opaque-query',ids,features,10)
assert a==ranking.rerank_with_ltr('opaque-query',ids,features,10)
valid(a)
base={d:{'bm25':float(i),'bm25_title':float(25-i),'lmir_dir':float(i%3)} for i,d in enumerate(ids)}
b=ranking.fuse_candidates('opaque-query',ids,base,10)
assert b==ranking.fuse_candidates('opaque-query',ids,base,10)
valid(b)
for kind in ('pointwise','pairwise'):
 scorer=load_scorer(f'submission/{kind}_model.pt')
 scores=scorer(X)
 assert scores.shape==(25,) and np.isfinite(scores).all()
 assert np.array_equal(scores,scorer(X))
results=[]
Path('tmp').mkdir(exist_ok=True)
for data in sorted(Path('data/toy_transfer').iterdir()):
 if not data.is_dir():continue
 target=Path('tmp')/f'{data.name}.txt'
 result=run_recipe_and_evaluate(str(data/'train.txt'),str(data/'vali.txt'),str(data/'vali.txt'),str(target),seed=0)
 assert result['ndcg@10']>.5 and result['num_eval_queries']==50
 ranking.prepare(str(target))
 assert lgb.Booster(model_file=str(target)).num_trees()>0
 results.append({'dataset':data.name,**result})
assert len(results)==3
print(json.dumps({'main_trees':4400,'main_features':272,'conformance':True,'neural_artifacts_load':True,
 'deterministic_finite_candidate_restricted_rankings':True,'C1':results},indent=2))
'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_path = OUT/'frozen_submission_sha256.json'
    if not frozen_path.exists():
        frozen_path.write_text(json.dumps({p.relative_to(ROOT).as_posix():digest(p)
                              for p in (ROOT/'submission').iterdir() if p.is_file()},indent=2))
    frozen = json.loads(frozen_path.read_text())
    assert all(digest(ROOT/name)==expected for name,expected in frozen.items())
    fresh_path = ROOT/'runs/final_c1_pass/fresh/verification.json'
    fresh = json.loads(fresh_path.read_text())
    expected = frozen['submission/model.txt']
    assert fresh['mode']=='fresh staged refit' and fresh['sha256']==expected
    assert fresh['byte_equal'] and fresh['prediction_equal'] and fresh['ranking_equal']
    assert digest(ROOT/'runs/final_c1_pass/fresh/model.txt')==expected
    reproduction = {'status':'Prior full fresh fit verified again by hashing its output; no new main fit',
                    'prior_result':fresh,
                    'environment':json.loads((ROOT/'runs/final_c1_pass/environment.json').read_text()),
                    'current_artifact_sha256':expected,
                    'cross_platform_reproduction':'Not verified on staff Ubuntu/Python 3.12'}
    (ROOT/'docs/REPRODUCTION_VERIFICATION.json').write_text(json.dumps(reproduction,indent=2)+'\n')
    dry = run(['scripts/train_model.py','--config','current-main','--train','data/train',
               '--dev','data/dev','--use-dev','--work-dir','runs/current_reproduction',
               '--out','runs/current_reproduction/model.txt','--dry-run'], ROOT, 'reproduction_dry_run.log')
    assert len(json.loads(dry)['commands'])==5
    # Original upload, before implementation and generated research history.
    starter = subprocess.check_output(['git','ls-tree','-r','--name-only','6b6a70b'],cwd=ROOT,text=True).splitlines()
    excluded_starter_data = [n for n in starter if n.startswith('data/') and n!='data/README.md']
    required_starter = sorted(set(starter)-set(excluded_starter_data))
    selected = source_files(ROOT)
    assert set(required_starter)<=set(selected)
    manifest = {'assignment_version':'1.5, dated 15 October 2026',
                'competition_deadline':'2026-10-22 23:59 Asia/Kolkata',
                'moodle_deadline':'2026-10-24 23:59 Asia/Kolkata',
                'required':list(REQUIRED),'frozen_submission_sha256':frozen,
                'starter_upload_commit':'6b6a70b','preserved_starter_files':required_starter,
                'excluded_starter_data_files':excluded_starter_data,
                'report_pdf':'Retained for existing verification guards; Moodle submission is separate',
                'source_sha256':{n:digest(ROOT/n) for n in selected if n!='docs/COMPETITION_MANIFEST.json'}}
    (ROOT/'docs/COMPETITION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    target = ROOT/'2023CS10076.zip'
    if target.exists():
        backup=OUT/f'previous_{digest(target)}.zip'
        if not backup.exists():shutil.copy2(target,backup)
    count=build(ROOT,target)
    secret_patterns = [rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                       rb'\bAKIA[0-9A-Z]{16}\b',rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
                       rb'\bgithub_pat_[A-Za-z0-9_]{40,}\b',rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b']
    with zipfile.ZipFile(target) as package:
        assert package.testzip() is None
        names=package.namelist()
        assert len(names)==len(set(names)) and set(REQUIRED)<=set(names)
        for name in names:
            path=PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and '\\' not in name
            assert not any(p in {'.git','.venv','__pycache__','.pytest_cache','runs','tmp','output',
                                 'storage-audit','submissions','node_modules'} for p in path.parts)
            if name.endswith('/'):continue
            data=package.read(name)
            assert data==(ROOT/name).read_bytes()
            assert path.suffix not in {'.ipynb','.pyc','.zip','.npy','.npz','.log','.pkl','.csv'}
            if name.startswith('data/'):assert name=='data/README.md'
            if path.suffix in {'.py','.md','.json','.yml','.yaml','.txt','.sh','.tex'} and not name.endswith('model.txt'):
                assert not any(re.search(pattern,data) for pattern in secret_patterns),f'Credential pattern: {name}'
        with tempfile.TemporaryDirectory(prefix='competition_extract_',dir=ROOT/'tmp') as directory:
            extract=Path(directory)
            package.extractall(extract)
            assert all(digest(extract/n)==digest(ROOT/n) for n in names if not n.endswith('/'))
            # Fixtures are supplied locally only AFTER archive/data-exclusion verification.
            for folder in ('toy','toy_transfer'):
                shutil.copytree(ROOT/'data'/folder,extract/'data'/folder,dirs_exist_ok=True)
            smoke=run(['-c',SMOKE],extract,'extracted_interfaces.log')
            smoke_result=json.loads(smoke)
            # The only main artifact overwritten is in this disposable extraction.
            run(['scripts/train_model.py','--train','data/toy','--out','submission/model.txt','--trees','30'],
                extract,'isolated_toy_training.log')
            suite=run(['-m','pytest','tests','-q','--basetemp','tmp/pytest'],extract,'extracted_full_tests.log')
            (extract/'submission/model.txt').write_bytes(package.read('submission/model.txt'))
            assert all(digest(extract/n)==digest(ROOT/n) for n in names if not n.endswith('/'))
    assert target.stat().st_size < 50_000_000
    assert all(digest(ROOT/name)==expected for name,expected in frozen.items())
    result={'archive':str(target),'bytes':target.stat().st_size,'MiB':target.stat().st_size/2**20,
            'sha256':digest(target),'files':count,'zip_entries':len(names),'required_files':list(REQUIRED),
            'artifact_sha256':{n:frozen['submission/'+n] for n in ('model.txt','pointwise_model.pt','pairwise_model.pt')},
            'all_submission_files_unchanged':True,'all_packaged_files_equal_sources':True,
            'CRC_integrity':True,'starter_structure_preserved':True,
            'data_exclusion':'Only data/README.md and empty data directories in ZIP',
            'secret_scan':'No prohibited paths or high-confidence credential patterns found; no credentials required',
            'extracted_interfaces':smoke_result,'extracted_full_suite':suite.strip().splitlines()[-1],
            'reproduction':reproduction['status'],'no_portal_submission':True,
            'limitations':['Staff Ubuntu/Python 3.12 execution not tested',
                           'Remote CI status at the 20 October freeze not verified',
                           'Historical TEST/public-data exposure remains disclosed in docs/COMPLIANCE.md']}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
