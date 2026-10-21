"""Training CLI schema and legacy behavior, using separate temporary artifacts."""
import json
from pathlib import Path
import subprocess
import sys

import lightgbm as lgb
import numpy as np

from scripts.ltr_data import load_splits
from scripts.lambdamart_utils import default_params, fit
from submission import ranking
from submission.query_features import augment_grouped
from scripts.compliant_train import training_groups


def test_augmentation_preserves_query_boundaries_in_lightgbm():
    q = np.repeat([11, 22, 33], [2, 3, 2])
    # Originals precede every augmentation, exactly as the full trainer writes rows.
    row_queries = np.concatenate([q, q])
    row_copies = np.repeat([0, 1], len(q))
    groups = training_groups(q, augmented=True)
    assert training_groups(q) == [2, 3, 2]
    assert groups == [2, 3, 2, 2, 3, 2]
    data = lgb.Dataset(np.arange(28).reshape(14, 2),
                       label=np.tile([0, 1, 0, 1, 2, 0, 1], 2), group=groups,
                       params={'min_data_in_bin': 1, 'min_data_in_leaf': 1, 'verbosity': -1})
    data.construct()
    np.testing.assert_array_equal(data.get_group(), groups)
    boundaries = np.r_[0, np.cumsum(data.get_group())]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        assert np.unique(row_queries[lo:hi]).size == 1
        assert np.unique(row_copies[lo:hi]).size == 1


def test_default_cli_preserves_basic_predictions(tmp_path):
    out = tmp_path / "basic.txt"
    subprocess.run([sys.executable, "scripts/train_model.py", "--train", "data/toy",
                    "--trees", "4", "--out", str(out)], check=True, capture_output=True)
    (X, y, q), _ = load_splits("data/toy/train.txt")
    expected = fit(X, y, q, default_params(), 4)
    actual = lgb.Booster(model_file=str(out))
    assert actual.feature_name() == expected.feature_name()
    np.testing.assert_array_equal(actual.predict(X[:30]), expected.predict(X[:30]))


def test_submitted_cli_declares_feature_schema_and_explicit_split(tmp_path):
    out, report = tmp_path / "relative.txt", tmp_path / "report.json"
    subprocess.run([sys.executable, "scripts/train_model.py", "--train", "data/toy",
                    "--config", "submitted", "--query-features", "--use-dev", "--trees", "4",
                    "--seed", "3", "--out", str(out), "--report-out", str(report)],
                   check=True, capture_output=True)
    result = json.loads(report.read_text())
    assert result["training_splits"] == "TRAIN+DEV" and result["features"] == 272
    assert result["params"]["extra_trees"] and result["params"]["seed"] == 3
    ranking.prepare(str(out))
    assert ranking._QUERY_RANK_FEATURES and ranking._RAW_WIDTH == 136
    (X, y, q), _ = load_splits("data/toy/vali.txt")
    rows = np.flatnonzero(q == q[0])
    inputs = augment_grouped(X[rows], q[rows])
    model = lgb.Booster(model_file=str(out))
    scores = model.predict(inputs, num_threads=1)
    ids = [str(i) for i in range(len(rows))]
    actual = ranking.rerank_with_ltr("q", ids, dict(zip(ids, X[rows])), len(ids))
    assert actual == [(ids[i], scores[i]) for i in np.argsort(-scores, kind="stable")]
