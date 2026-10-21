"""Verify cached raw/derived arrays against only the two released source files.

This streams TRAIN/DEV, checks their known SHA-256 values, and never opens TEST.
The audit produces aggregate provenance metadata, not dataset rows or labels.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from submission.query_features import augment_query

CACHE = ROOT/'data/cache/optimization'
OUT = ROOT/'runs/compliant_campaign'


def verify_split(split, source, expected_sha, batch_size=4096):
    folder = CACHE/split
    manifest = json.loads((folder/'manifest.json').read_text())
    stat = source.stat()
    if (manifest['bytes'], manifest['mtime_ns']) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError('Source/cache freshness mismatch: '+split)
    X, y, q = [np.load(folder/(name+'.npy'), mmap_mode='r', allow_pickle=False)
               for name in ['X', 'y', 'q']]
    block = np.zeros((batch_size, X.shape[1]), dtype=np.float32)
    labels, ids, offset, used = [], [], 0, 0
    digest = hashlib.sha256()
    def compare():
        if not np.array_equal(block[:used], X[offset:offset+used], equal_nan=True):
            raise ValueError('Raw feature cache differs from released '+split)
        if not np.array_equal(np.asarray(labels), y[offset:offset+used]):
            raise ValueError('Label cache differs from released '+split)
        if not np.array_equal(np.asarray(ids), q[offset:offset+used]):
            raise ValueError('Query cache differs from released '+split)
    with source.open('rb') as stream:
        for raw_line in stream:
            digest.update(raw_line)
            fields = raw_line.split(b'#', 1)[0].split(None, 2)
            if not fields:
                continue
            labels.append(float(fields[0]))
            if not fields[1].startswith(b'qid:'):
                raise ValueError('Invalid released query token')
            ids.append(fields[1][4:].decode('ascii'))
            if len(fields) == 3:
                values = np.fromstring(fields[2].replace(b':', b' '), sep=' ', dtype=np.float64)
                indices = values[::2].astype(np.int64)-1
                if (len(values) % 2 or np.any(indices < 0) or np.any(indices >= X.shape[1])
                        or np.any(indices+1 != values[::2])):
                    raise ValueError('Invalid released feature index')
                block[used, indices] = values[1::2].astype(np.float32)
            used += 1
            if used == batch_size:
                compare(); offset += used; used = 0
                block.fill(0); labels.clear(); ids.clear()
                if offset % (batch_size*50) == 0:
                    print(split, 'source rows verified', offset, flush=True)
    if used:
        compare(); offset += used
    if offset != len(y) or digest.hexdigest() != expected_sha:
        raise ValueError('Released input row count or SHA mismatch: '+split)
    ranks = np.load(folder/'query_ranks_v1.npy', mmap_mode='r', allow_pickle=False)
    starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1])+1, len(q)]
    for number, (lo, hi) in enumerate(zip(starts[:-1], starts[1:])):
        if not np.array_equal(augment_query(X[lo:hi]), ranks[lo:hi], equal_nan=True):
            raise ValueError('Query-derived cache differs from released '+split)
        if number % 2000 == 0:
            print(split, 'query transforms verified', number, flush=True)
    return {'source_sha256': digest.hexdigest(), 'rows': offset,
            'queries': len(starts)-1, 'raw_labels_queries_match_source': True,
            'query_transforms_recomputed_and_equal': True}, ranks


def main(argv=()):
    import argparse
    from scripts.cv_query_features import combined_data
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT/'training_cache_audit.json')
    args = ap.parse_args(argv)
    declaration = ROOT/'docs/TRAINING_AUDIT.json'
    if declaration.exists():
        expected = json.loads(declaration.read_text())['verified_splits']
    else:
        # Backward-compatible programmatic audit fixtures / historical receipts.
        old = json.loads((OUT/'input_verification.json').read_text())['verified_inputs']
        expected = {split: {'source_sha256': old[filename]['sha256']} for split,filename
                    in [('train','data/train/train.txt'),('dev','data/dev/vali.txt')]}
    # A clean checkout need not contain any pre-existing campaign caches.
    if not (CACHE/'combined_query_v1.npy').exists():
        combined_data()
    results, matrices = {}, []
    for split, filename in [('train', 'data/train/train.txt'), ('dev', 'data/dev/vali.txt')]:
        result, ranks = verify_split(split, ROOT/filename, expected[split]['source_sha256'])
        results[split] = result
        matrices.append(ranks)
    combined = np.load(CACHE/'combined_query_v1.npy', mmap_mode='r', allow_pickle=False)
    offset = 0
    for matrix in matrices:
        for lo in range(0, len(matrix), 4096):
            hi = min(lo+4096, len(matrix))
            if not np.array_equal(matrix[lo:hi], combined[offset+lo:offset+hi], equal_nan=True):
                raise ValueError('Combined cache differs from verified TRAIN/DEV transforms')
        offset += len(matrix)
    if offset != len(combined):
        raise ValueError('Combined cache contains unexpected rows')
    result = {'verified_splits': results, 'combined_exactly_train_then_dev': True,
              'TEST_opened': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
