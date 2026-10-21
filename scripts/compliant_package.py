"""Verify TRAIN/DEV provenance, preflight, and freeze a concrete portal package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_submission import build, REQUIRED
from scripts.optimize_rankers import cached
from scripts.preflight import check

OUT = ROOT / 'runs/compliant_campaign'
TRUSTED_NEURAL = {
    'pairwise_model.pt':'00568f30451537e0592dcc37d4444a1333382cbd54205d4e7c3bbddd62691140',
    'pointwise_model.pt':'cfbff50b8290bb112912256e29b36dbd5f62669b66d0f69fb44f17c547279495',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_model(receipt):
    record = json.loads(Path(receipt).read_text())
    if record['training_splits'] != 'Fold1 TRAIN+DEV only':
        raise ValueError('Only declared TRAIN+DEV fits can be packaged')
    if record.get('args',{}).get('mode','full') != 'full':
        raise ValueError('Study artifact is not a final refit')
    model = ROOT / record['model']
    if digest(model) != record['sha256']:
        raise ValueError('Model bytes do not match training receipt')
    for source in record.get('constituent_receipts',[]):
        verify_model(ROOT / source)
    if record.get('continued_from'):
        verify_model(ROOT / record['continued_from'])
    if not record.get('constituent_receipts'):
        if (record['original_rows'],record['original_queries']) != (958671,8000):
            raise ValueError('Unexpected TRAIN+DEV size')
    return model,record


def verify_neural(receipt):
    import torch
    record=json.loads(Path(receipt).read_text())
    if not record['args'].get('full'):
        raise ValueError('New neural artifact must be a final refit')
    if not record['args'].get('fresh'):
        init = record['args'].get('init')
        trusted = TRUSTED_NEURAL['pairwise_model.pt']
        if (not init or record.get('initial_artifact_sha256') != trusted
                or digest(ROOT / init) != trusted):
            raise ValueError('Neural warm start must match the verified TRAIN+DEV-only baseline')
    if record.get('TEST_used') is not False or record.get('gradients')!='TRAIN+DEV':
        raise ValueError('New neural artifact must exclude TEST')
    model=ROOT/record['model']
    if digest(model)!=record['sha256']:
        raise ValueError('Neural model bytes do not match receipt')
    blob=torch.load(model,weights_only=True)
    reference=torch.load(ROOT/'submission/pointwise_model.pt',weights_only=True)
    if blob['standardizer']!=reference['standardizer']:
        raise ValueError('Neural TRAIN standardization changed')
    hidden=blob['config']['hidden']
    if not 1<=len(hidden)<=3 or not all(1<=n<=256 for n in hidden):
        raise ValueError('Neural architecture exceeds staff bounds')
    return model,record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',required=True)
    ap.add_argument('--receipt',required=True)
    ap.add_argument('--pairwise-receipt',help='Verified TRAIN+DEV neural fit or refinement receipt')
    ap.add_argument('--ready-folder',help='Optional folder name under submissions; ZIP basename stays 2023CS10076.zip')
    ap.add_argument('--cv-report', default='runs/optimization/query_extra_cv.json',
                    help='Completed TRAIN-only five-fold sweep for the report')
    a=ap.parse_args()
    ready_name=a.ready_folder or a.name
    if not ready_name.replace('_','').isalnum():
        ap.error('Ready folder must contain only letters, digits, and underscores')
    ready=ROOT/'submissions'/ready_name
    if ready.exists():
        raise FileExistsError('Refusing to overwrite a frozen candidate: '+str(ready))
    model,record=verify_model(a.receipt)
    stage=OUT/'stages'/a.name;stage.mkdir(parents=True,exist_ok=False)
    shutil.copy2(model,stage/'model.txt')
    neural_record=None
    for name,expected in TRUSTED_NEURAL.items():
        if name=='pairwise_model.pt' and a.pairwise_receipt:
            neural_model,neural_record=verify_neural(a.pairwise_receipt)
            shutil.copy2(neural_model,stage/name)
            continue
        if digest(ROOT/'submission'/name)!=expected:
            raise ValueError('Neural artifact requires a new independently checked provenance declaration: '+name)
        shutil.copy2(ROOT/'submission'/name,stage/name)
    result=check('data/dev/vali.txt',stage,data=cached('dev'))
    (OUT/(a.name+'_preflight.json')).write_text(json.dumps(result,indent=2))
    if not result['passes']:
        raise ValueError('Artifact preflight failed')
    pairwise_description='''Pairwise is a plain 136->256->128->1 ReLU MLP, combining .3 of a seed-0
9-epoch refit and .7 of a fresh seed-23 6-epoch refit. Both use relevance
labels from TRAIN+DEV, identical frozen TRAIN-only standardization, and no
TEST gradients. Weight selection used 400 DEV queries excluded from study
gradients. Repeated DEV selection is development, not a fresh test.
See docs/NEURAL_PROVENANCE.md for settings and reproduction.'''
    if neural_record:
        b=neural_record['args']
        pairwise_description=f'''Pairwise is a fresh plain ReLU MLP with hidden widths {neural_record['hidden']}.
It uses TRAIN+DEV relevance labels, unchanged TRAIN-only standardization,
seed {b['seed']}, fixed {neural_record['saved_fixed_epoch']} epochs, learning
rate {b['lr']} and input dropout {b['dropout']}. Its matched study reserves
400 DEV queries from gradients. No TEST gradients or teacher labels are used.
Exact configuration is in docs/CURRENT_NEURAL.json. Repeated DEV and portal
selection are development comparisons, not fresh final tests.'''
        if not b.get('fresh'):
            pairwise_description=f'''Pairwise refines the verified TRAIN+DEV-only retained plain ReLU MLP
with hidden widths {neural_record['hidden']}, using {neural_record['saved_fixed_epoch']} fixed epochs,
seed {b['seed']}, learning rate {b['lr']} and input dropout {b['dropout']}.
It starts from the retained baseline hash recorded in docs/CURRENT_NEURAL.json.
It uses TRAIN+DEV relevance labels, unchanged TRAIN-only standardization and
no TEST labels or teacher scores. Epoch averaging starts at {b['swa_start']}.
All DEV queries supply gradients; local DEV is a fitting measurement, with
no claim of a fresh local validation set. Portal feedback guides comparison.
Exact settings and initialization hash are recorded in docs/CURRENT_NEURAL.json.'''
    declaration=f'''# A3 compliant submission - Ahilaan Saxena (2023CS10076)

The main model is fitted from scratch on Fold1 TRAIN+DEV only: 958,671
original rows / 8,000 queries. Fold1 TEST is excluded from fitting,
augmentation, preprocessing statistics and local validation. DEV is fitting
data; its reported nDCG is trained-on. Model SHA-256: `{record['sha256']}`.

Model configuration and reproduction are recorded in
`docs/CURRENT_MODEL.json`. Run its `reproduction` commands in order, then
`python scripts/compliant_package.py --name {a.name}_reproduced --receipt {a.receipt} --cv-report {a.cv_report}`.

Inputs comprise raw features and declared within-query transforms derived
without labels. Inference derives feature width from the artifact schema.
The model is a native LightGBM LambdaRank text artifact. C1 uses the generic
recipe in submission/recipe.py. Fusion is handwritten equal-weight CombSUM
over BM25 whole-document, BM25 title and LMIR Dirichlet whole-document;
LMIR JM is disabled. Selection uses TRAIN only; see docs/FUSION_PROVENANCE.md.
Pointwise uses TRAIN gradients and DEV epoch selection.

{pairwise_description}

Earlier experiments used TEST labels despite the assignment's prohibition.
Those main-model weights have been replaced. Current fitting uses TRAIN+DEV
only; historical model-selection exposure and its limitations are documented
in docs/COMPLIANCE.md. Portal feedback informs campaign selection.

Install requirements.txt; use scripts/setup_mslr.py to prepare only TRAIN/DEV.
Run `python -m pytest tests -q -k "not test_end_to_end_toy"` for artifact
regression checks and `python scripts/preflight.py` for the real DEV check.
The required three-page report is docs/REPORT.pdf; its model hash and CV
evidence are recorded in docs/REPORT_MANIFEST.json and docs/CV_RESULTS.json.
For local report regeneration, install reportlab and pypdf; use pymupdf for
rendering QA. The grading entry points use requirements.txt.
The starter toy smoke script overwrites the main model; run it only in a
disposable copy. No data, credentials, local archives or campaign outputs
are included in the ZIP.

AI-use disclosure: Codex assisted with implementation, experiments, testing,
documentation and authorized portal submissions. ChatGPT assisted with
assignment analysis. Code/formula sources: assignment and starter repository;
no external implementation copied. The student must verify these statements.
'''
    reproduction=record.get('reproduction')
    if reproduction is None:
        args=record['args']; command='python scripts/compliant_train.py full --name '+args['name']+' --config '+args['config']+' --rounds '+str(args['rounds'])+' --normalization '+args['normalization']+' --seed '+str(args['seed'])
        for flag in ['augment','group_dropout','rank_only']:
            if args[flag]:command+=' --'+flag.replace('_','-')
        command+=' --threads '+str(args.get('threads',4))
        if args.get('checkpoints'):
            command+=' --checkpoints '+' '.join(map(str, sorted(set(args['checkpoints']))))
        reproduction=[command]
    current=dict(record,reproduction=reproduction,report_cv_receipt=a.cv_report)
    (ROOT/'docs/CURRENT_MODEL.json').write_text(json.dumps(current,indent=2))
    from scripts.make_report import make_report
    make_report(current, result, ROOT/a.cv_report, neural_record=neural_record)
    if neural_record:
        (ROOT/'docs/CURRENT_NEURAL.json').write_text(json.dumps(neural_record,indent=2))
    else:
        (ROOT/'docs/CURRENT_NEURAL.json').unlink(missing_ok=True)
    # The build includes source/docs .md and .py plus an explicit current-model manifest.
    for name in ['README.md','docs/SUBMISSION_NOTES.md']:
        (ROOT/name).write_text(declaration,encoding='utf-8')
    shutil.copy2(stage/'model.txt',ROOT/'submission/model.txt')
    shutil.copy2(stage/'pairwise_model.pt',ROOT/'submission/pairwise_model.pt')
    count=build(ROOT,ROOT/'2023CS10076.zip')
    package=OUT/'packages'/(a.name+'_2023CS10076.zip')
    shutil.copy2(ROOT/'2023CS10076.zip',package)
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        assert all(z.read(n)==(ROOT/n).read_bytes() for n in REQUIRED)
        assert not any(n.startswith(('runs/','tmp/','data/cache/')) for n in z.namelist())
    receipt={'name':a.name,'sha256':digest(package),'bytes':package.stat().st_size,
             'files':count,'package':package.relative_to(ROOT).as_posix(),
             'model_receipt':a.receipt,'required_sha256':{n:digest(ROOT/n) for n in REQUIRED},
             'pairwise_receipt':a.pairwise_receipt,
             'report_manifest':json.loads((ROOT/'docs/REPORT_MANIFEST.json').read_text()),
             'preflight':result}
    ready.mkdir(parents=True,exist_ok=False)
    shutil.copy2(package,ready/'2023CS10076.zip')
    receipt['ready_path']=(ready/'2023CS10076.zip').relative_to(ROOT).as_posix()
    (ready/'verification.json').write_text(json.dumps(receipt,indent=2))
    (OUT/(a.name+'_package.json')).write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
