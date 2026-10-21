"""Guard against missing grader inputs and accidental dataset/credential uploads."""
from pathlib import Path
import hashlib
import json
import zipfile
import pytest
from scripts import build_submission as packaging

from scripts.build_submission import REQUIRED, build, source_files, verify_report


def test_package_contains_required_files_and_excludes_private_outputs(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    selected = source_files(root)
    assert set(REQUIRED) <= set(selected)
    assert not any(name.startswith(("tmp/", "runs/", ".git/", ".venv/")) for name in selected)
    assert [name for name in selected if name.startswith("data/")] == ["data/README.md"]
    assert not any(name.endswith(("_history.json", ".pyc")) for name in selected)
    archive = tmp_path / "2023CS10076.zip"
    # Archive construction is independent of the report guard tested below.
    # The full toy suite intentionally replaces model.txt with a toy fixture.
    monkeypatch.setattr(packaging, 'verify_report', lambda root: None)
    assert build(root, archive) == len(selected)
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert set(REQUIRED) <= set(package.namelist())
        for name in REQUIRED:
            assert package.read(name) == (root / name).read_bytes()


def report_fixture(root):
    (root/'docs').mkdir(); (root/'submission').mkdir()
    (root/'submission/model.txt').write_bytes(b'model')
    for name in ['pointwise_model.pt', 'pairwise_model.pt']:
        (root/'submission'/name).write_bytes(name.encode())
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
               b'<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>']
    objects += [b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] >>'] * 3
    pdf = bytearray(b'%PDF-1.4\n'); offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f'{number} 0 obj\n'.encode()+obj+b'\nendobj\n')
    xref = len(pdf); pdf.extend(b'xref\n0 6\n0000000000 65535 f \n')
    for offset in offsets[1:]: pdf.extend(f'{offset:010d} 00000 n \n'.encode())
    pdf.extend(f'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode())
    (root/'docs/REPORT.pdf').write_bytes(pdf)
    manifest = {'model_sha256': hashlib.sha256(b'model').hexdigest(),
                'artifact_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (root/'submission').iterdir()},
                'report_sha256': hashlib.sha256((root/'docs/REPORT.pdf').read_bytes()).hexdigest(),
                'cv_source_sha256': 'cv', 'pages': 3}
    (root/'docs/REPORT_MANIFEST.json').write_text(json.dumps(manifest))
    (root/'docs/CV_RESULTS.json').write_text(json.dumps(
        {'split': 'Fold1 TRAIN only', 'folds': 5, 'source_sha256': 'cv'}))


def test_report_cannot_follow_changed_weights(tmp_path):
    report_fixture(tmp_path)
    assert verify_report(tmp_path)['pages'] == 3
    (tmp_path/'submission/model.txt').write_bytes(b'new model')
    with pytest.raises(ValueError, match='different model'): verify_report(tmp_path)


def test_report_pdf_cannot_change_after_verification(tmp_path):
    report_fixture(tmp_path)
    with (tmp_path/'docs/REPORT.pdf').open('ab') as output: output.write(b'changed')
    with pytest.raises(ValueError, match='PDF bytes changed'): verify_report(tmp_path)


def test_report_binds_its_input_audit(tmp_path):
    report_fixture(tmp_path)
    path = tmp_path/'docs/TRAINING_AUDIT.json'
    path.write_bytes(b'audit')
    manifest_path = tmp_path/'docs/REPORT_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['training_audit_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    verify_report(tmp_path)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Input-audit bytes changed'): verify_report(tmp_path)
