"""Packaging must reject mislabeled, changed, or noncompliant constituent weights."""
import hashlib
import json
import pytest
from scripts import compliant_package as package


def record(tmp_path, **changes):
    model=tmp_path/'model.txt';model.write_bytes(b'weights')
    value={'model':'model.txt','sha256':hashlib.sha256(b'weights').hexdigest(),
           'training_splits':'Fold1 TRAIN+DEV only','original_rows':958671,
           'original_queries':8000,'args':{'mode':'full'}}
    value.update(changes)
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps(value))
    return receipt,model


def test_valid_refit_and_changed_weights(tmp_path,monkeypatch):
    monkeypatch.setattr(package,'ROOT',tmp_path)
    receipt,model=record(tmp_path)
    assert package.verify_model(receipt)[0]==model
    model.write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes'):
        package.verify_model(receipt)


@pytest.mark.parametrize('changes',[
    {'training_splits':'TRAIN+DEV+TEST'},
    {'original_rows':1200192,'original_queries':10000},
    {'args':{'mode':'study'}},
])
def test_reject_disallowed_refit(tmp_path,monkeypatch,changes):
    monkeypatch.setattr(package,'ROOT',tmp_path)
    receipt,_=record(tmp_path,**changes)
    with pytest.raises(ValueError):package.verify_model(receipt)


def test_reject_noncompliant_blend_constituent(tmp_path,monkeypatch):
    monkeypatch.setattr(package,'ROOT',tmp_path)
    receipt,_=record(tmp_path,training_splits='TRAIN+DEV+TEST')
    child=tmp_path/'child.json';receipt.rename(child)
    receipt,_=record(tmp_path,constituent_receipts=['child.json'])
    with pytest.raises(ValueError,match=r'TRAIN\+DEV fits'):
        package.verify_model(receipt)


def test_new_neural_refit_requires_matching_train_statistics_and_final_fit(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(package,'ROOT',tmp_path)
    (tmp_path/'submission').mkdir()
    blob={'standardizer':{'mean':[1.,2.],'scale':[2.,3.]},'config':{'hidden':[8,4]}}
    torch.save(blob,tmp_path/'submission/pointwise_model.pt')
    model=tmp_path/'neural.pt';torch.save(blob,model)
    value={'model':'neural.pt','sha256':package.digest(model),
           'args':{'full':True,'fresh':True},'TEST_used':False,'gradients':'TRAIN+DEV'}
    receipt=tmp_path/'neural.json';receipt.write_text(json.dumps(value))
    assert package.verify_neural(receipt)[0]==model
    value['args']['full']=False;receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='final refit'):package.verify_neural(receipt)
    value['args']['full']=True
    blob['standardizer']['mean']=[99.,2.];torch.save(blob,model)
    value['sha256']=package.digest(model);receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='standardization'):package.verify_neural(receipt)


def test_warm_neural_refit_requires_exact_verified_baseline_bytes(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(package,'ROOT',tmp_path)
    (tmp_path/'submission').mkdir()
    blob={'standardizer':{'mean':[1.,2.],'scale':[2.,3.]},'config':{'hidden':[8,4]}}
    torch.save(blob,tmp_path/'submission/pointwise_model.pt')
    initial=tmp_path/'initial.pt';torch.save(blob,initial)
    trusted=package.digest(initial)
    monkeypatch.setitem(package.TRUSTED_NEURAL,'pairwise_model.pt',trusted)
    model=tmp_path/'new.pt';torch.save(blob,model)
    record={'model':'new.pt','sha256':package.digest(model),
            'args':{'full':True,'fresh':False,'init':'initial.pt'},
            'initial_artifact_sha256':trusted,'TEST_used':False,'gradients':'TRAIN+DEV'}
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps(record))
    assert package.verify_neural(receipt)[0]==model
    initial.write_bytes(b'changed initialization')
    with pytest.raises(ValueError,match='warm start'):package.verify_neural(receipt)
