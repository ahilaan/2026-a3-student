"""The data audit must check content, including non-full final split batches."""
import hashlib
import json

import numpy as np
import pytest
from scripts import check_training_inputs as audit
from submission.query_features import augment_grouped


def inputs(root, monkeypatch):
    cache, out = root/'cache', root/'runs'
    out.mkdir()
    monkeypatch.setattr(audit, 'ROOT', root)
    monkeypatch.setattr(audit, 'CACHE', cache)
    monkeypatch.setattr(audit, 'OUT', out)
    fixtures = [
        ('train', 'data/train/train.txt', b'3 qid:1 1:1.5 2:2\n1 qid:1 1:3\n0 qid:2 2:-1\n',
         [[1.5, 2], [3, 0], [0, -1]], [3, 1, 0], ['1', '1', '2']),
        ('dev', 'data/dev/vali.txt', b'2 qid:3 1:0.25 2:1\n4 qid:4 1:-2 2:4\n',
         [[.25, 1], [-2, 4]], [2, 4], ['3', '4']),
    ]
    expected, derived = {}, []
    for split, filename, text, raw, labels, ids in fixtures:
        source = root/filename
        source.parent.mkdir(parents=True)
        source.write_bytes(text)
        folder = cache/split
        folder.mkdir(parents=True)
        X, y, q = np.array(raw, dtype=np.float32), np.array(labels), np.array(ids)
        for name, values in [('X', X), ('y', y), ('q', q)]: np.save(folder/(name+'.npy'), values)
        ranks = augment_grouped(X, q)
        np.save(folder/'query_ranks_v1.npy', ranks)
        derived.append(ranks)
        stat = source.stat()
        (folder/'manifest.json').write_text(json.dumps({'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}))
        expected[filename] = {'sha256': hashlib.sha256(text).hexdigest()}
    np.save(cache/'combined_query_v1.npy', np.vstack(derived))
    (out/'input_verification.json').write_text(json.dumps({'verified_inputs': expected}))
    return cache, out


def test_audit_matches_sparse_sources_and_partial_final_split_batch(tmp_path, monkeypatch):
    _, out = inputs(tmp_path, monkeypatch)
    audit.main()
    result = json.loads((out/'training_cache_audit.json').read_text())
    assert result['combined_exactly_train_then_dev']
    assert not result['TEST_opened']
    assert result['verified_splits']['train']['rows'] == 3
    assert result['verified_splits']['dev']['rows'] == 2


@pytest.mark.parametrize('changed', ['y', 'query_ranks_v1'])
def test_audit_rejects_cache_content_changes_even_with_matching_source_manifest(tmp_path, monkeypatch, changed):
    cache, _ = inputs(tmp_path, monkeypatch)
    path = cache/'train'/(changed+'.npy')
    values = np.load(path)
    values.flat[0] = 99
    np.save(path, values)
    with pytest.raises(ValueError, match='cache differs'):
        audit.main()
