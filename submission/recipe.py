"""
submission/recipe.py -- your training recipe (assignment Section 4.1, Track C1).

Staff call train() ONCE per hidden benchmark (two undisclosed flat-feature LTR datasets) on that
benchmark's train and vali files, in an isolated directory, then evaluate the model it writes on the benchmark's
held-out split through YOUR prepare() / rerank_with_ltr(). Contract:

  * signature: train(train_path, vali_path, out_model_path, seed=0) -> None
  * the files are flat-feature SVMLight (label qid:N 1:v ... D:v), D unknown (at most ~300), labels integers
    0..4 (possibly fewer levels), at most ~500k training rows; feature meanings are NOT given
  * write a LightGBM text model (or whatever your prepare() loads) to out_model_path
  * 20 minutes wall-clock PER BENCHMARK on 4 CPU threads; no network; read only the two input files; deterministic given seed
  * a recipe that raises, times out, or writes a model prepare() cannot load scores 0 on that benchmark

The provided default is a fixed-hyperparameter LambdaMART. Replace the hyperparameters with choices that
transfer: the point of Track C1 is that settings picked by cross-validation on MSLR TRAIN should carry over to
a different dataset -- e.g. express tree count / leaf size / min_data_in_leaf as functions of the number of
queries and features rather than hard-coding numbers tuned on MSLR, or run a small CV inside the time limit.

Test it locally on the synthetic transfer sets:  python scripts/run_recipe.py
"""
import numpy as np


def _group_sizes(qids):
    sizes, last = [], object()
    for q in qids:
        if q == last:
            sizes[-1] += 1
        else:
            sizes.append(1)
            last = q
    return sizes


def _scan(path):
    """Find width and query counts without retaining Python feature tuples."""
    counts, width = {}, 0
    with open(path, encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            try:
                if len(fields) < 2 or not fields[1].startswith("qid:") or not fields[1][4:]:
                    raise ValueError("expected label qid:<id>")
                qid = fields[1][4:]
                counts[qid] = counts.get(qid, 0) + 1
                for field in fields[2:]:
                    index, sep, value = field.partition(":")
                    index = int(index)
                    if not sep or not value or index < 1:
                        raise ValueError("feature indices must be positive and have values")
                    width = max(width, index)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return counts, width


def _read(path, counts, width):
    """Two-pass dense float32 reader; place interleaved queries in contiguous blocks.

    At 500k x 300 the feature matrix uses 600 MB; unlike retaining all sparse
    tokens as Python tuples, parsing memory does not grow by many GB.
    """
    n_rows = sum(counts.values())
    X, y = np.zeros((n_rows, width), dtype=np.float32), np.empty(n_rows, dtype=np.int32)
    positions, offset = {}, 0
    for qid, count in counts.items():
        positions[qid] = offset
        offset += count
    with open(path, encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            try:
                label = float(fields[0])
                if not label.is_integer() or not 0 <= label <= 4:
                    raise ValueError("labels must be integers in 0..4")
                qid = fields[1][4:]
                row = positions[qid]
                positions[qid] += 1
                y[row] = int(label)
                for field in fields[2:]:
                    index, value = field.split(":")
                    X[row, int(index) - 1] = float(value)
            except (ValueError, IndexError, KeyError) as exc:
                raise ValueError(f"{path}:{line_no}: malformed LTR row ({exc})") from exc
    return X, y, list(counts.values())


def train(train_path: str, vali_path: str, out_model_path: str, seed: int = 0) -> None:
    import lightgbm as lgb

    train_counts, train_width = _scan(train_path)
    vali_counts, vali_width = _scan(vali_path)
    width = max(train_width, vali_width)
    if not train_counts or not vali_counts or width == 0:
        raise ValueError("TRAIN and validation must be nonempty with at least one feature")
    X, y, groups = _read(train_path, train_counts, width)
    Xv, yv, vali_groups = _read(vali_path, vali_counts, width)
    params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
        "num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 50, "lambda_l2": 1.0,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed,
        "data_random_seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed,
        "deterministic": True, "force_row_wise": True, "verbose": -1, "num_threads": 4,
    }
    training = lgb.Dataset(X, label=y, group=groups)
    validation = lgb.Dataset(Xv, label=yv, group=vali_groups, reference=training)
    # A single bounded run, no search or train+validation refit. Only TRAIN
    # labels supply gradients; validation chooses the best iteration.
    booster = lgb.train(params, training, num_boost_round=800, valid_sets=[validation],
                        callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
    booster.save_model(out_model_path)
