"""C1 regrouping, sparse schema, bad input and emergency stopping checks."""
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pytest

from submission import ranking, recipe


def test_sparse_interleaved_queries_split_widths_and_missing_values(tmp_path):
    train, valid, model = (tmp_path / name for name in ["train.txt", "vali.txt", "model.txt"])
    train.write_text("2 qid:b 1:3 # first\n0 qid:a 3:2\n1 qid:b 2:1\n0 qid:c 1:nan\n")
    valid.write_text("1 qid:v 272:1\n1 qid:v 2:5\n0 qid:w # singleton\n")
    counts, width = recipe._scan(train)
    assert list(counts.items()) == [("b", 2), ("a", 1), ("c", 1)]
    X, y, groups = recipe._read(train, counts, 272)
    assert groups == [2, 1, 1] and y.tolist() == [2, 1, 0, 0]
    assert X[0, 0] == 3 and X[1, 1] == 1 and X[2, 2] == 2 and np.isnan(X[3, 0])
    assert np.count_nonzero(X[:, 3:]) == 0
    recipe.train(str(train), str(valid), str(model), 0)
    ranking.prepare(str(model))
    # A raw 272-column model must never be mistaken for engineered MSLR.
    assert ranking._RAW_WIDTH == 272 and not ranking._QUERY_RANK_FEATURES
    result = ranking.rerank_with_ltr("q", ["a", "b"], {"a": [0.] * 272, "b": [1.] * 272}, 2)
    assert len(result) == 2 and np.isfinite([s for _, s in result]).all()


@pytest.mark.parametrize("row", ["", "0 1:2", "0 qid:", "0 qid:a 0:3", "0 qid:a 1:x",
                                "-1 qid:a 1:1", "1.5 qid:a 1:1", "nan qid:a 1:1",
                                "0 qid:a 1:inf"])
def test_malformed_or_empty_inputs_fail_with_context(tmp_path, row):
    train, valid = tmp_path / "train.txt", tmp_path / "vali.txt"
    train.write_text(row + "\n")
    valid.write_text("1 qid:v 1:1\n")
    with pytest.raises(ValueError):
        recipe.train(str(train), str(valid), str(tmp_path / "out.txt"))


def test_finite_overflow_is_valid_transfer_data_but_main_reader_stays_strict(tmp_path):
    train,valid=tmp_path/'train.txt',tmp_path/'vali.txt'
    train.write_text('1 qid:a 1:1e100\n0 qid:a 1:-1.79e308\n')
    valid.write_text('1 qid:v 1:1.79e308\n0 qid:v 1:0\n')
    counts,width=recipe._scan(train)
    with pytest.raises(ValueError):recipe._read(train,counts,width)
    X,_,_=recipe._read(train,counts,width,saturate=True)
    assert np.isfinite(X).all() and X[0,0]==np.finfo(np.float32).max
    assert X[1,0]==-np.finfo(np.float32).max
    target=tmp_path/'model.txt';recipe.train(str(train),str(valid),str(target),0)
    ranking.prepare(str(target))
    with pytest.warns(RuntimeWarning,match='overflow encountered in cast'):
        result=ranking.rerank_with_ltr('q',['a','b'],{'a':[1.79e308],'b':[-1.79e308]},2)
    assert len(result)==2 and np.isfinite([score for _,score in result]).all()


def test_deadline_keeps_best_completed_iteration():
    times = iter([0., 1., 2., 90.])
    stop = recipe._TrainingDeadline(100., clock=lambda: next(times))
    for i, score in enumerate([.6, .8]):
        stop(SimpleNamespace(iteration=i, evaluation_result_list=[("valid", "ndcg@10", score, True)]))
    with pytest.raises(lgb.callback.EarlyStopException) as caught:
        stop(SimpleNamespace(iteration=2, evaluation_result_list=[("valid", "ndcg@10", .7, True)]))
    assert caught.value.best_iteration == 1
    assert caught.value.best_score[0][2] == .8


def test_expired_deadline_still_saves_a_loadable_first_tree(tmp_path, monkeypatch):
    original = recipe._TrainingDeadline
    monkeypatch.setattr(recipe, "_TrainingDeadline", lambda deadline: original(-1.))
    out = tmp_path / "timed.txt"
    recipe.train("data/toy_transfer/a_40_features_labels0-4/train.txt",
                 "data/toy_transfer/a_40_features_labels0-4/vali.txt", str(out), 0)
    model = lgb.Booster(model_file=str(out))
    assert model.num_trees() == 1
    ranking.prepare(str(out))
    assert len(ranking.rerank_with_ltr("q", ["a"], {"a": [0.] * 40}, 1)) == 1
