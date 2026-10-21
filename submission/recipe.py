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

The selected policy first saves a shallow raw-feature LambdaMART. When the shared deadline allows, it
compares the same ranker with label-free within-query percentiles appended. Validation controls stopping
and selects the representation, with a .001 improvement required to replace the raw baseline. Only one
self-describing native model is saved; the existing inference path derives its transform from feature names.

Test it locally on the synthetic transfer sets:  python scripts/run_recipe.py
"""
import math
import os
import time
import numpy as np

_FLOAT32_MAX = float(np.finfo(np.float32).max)
_TIME_BUDGET_SECONDS = 960.0


def _save_model(booster, path):
    """Replace only after serialization succeeds; retain any valid fallback."""
    temporary = str(path) + '.recipe.tmp'
    booster.save_model(temporary)
    os.replace(temporary, path)


class _FirstTreeFallback:
    """Save the first completed tree before spending the remaining budget.

    Normal completion replaces it with the best validation iteration. This
    cannot rescue a process killed before its first native iteration finishes.
    """
    order = 10
    before_iteration = False

    def __init__(self, path):
        self.path = path
        self.saved = False

    def __call__(self, env):
        if not self.saved:
            _save_model(env.model, self.path)
            self.saved = True


class _TrainingDeadline:
    """Stop at a completed iteration and let LightGBM retain the best model.

    The deadline includes input parsing. The reserve accounts for saving and
    two recent iteration costs; it cannot interrupt parsing or a native call.
    Normal runs retain the exact original fixed-seed/early-stopping behavior.
    """
    order = 20
    before_iteration = False

    def __init__(self, deadline, clock=None):
        self.clock = clock or time.monotonic
        self.deadline = deadline
        self.previous = self.clock()
        self.costs = []
        self.best_score = -float("inf")
        self.best_iteration = 0
        self.best_result = None

    def __call__(self, env):
        from lightgbm.callback import EarlyStopException
        now = self.clock()
        self.costs = (self.costs + [now - self.previous])[-5:]
        self.previous = now
        result = env.evaluation_result_list
        score = result[0][2]  # only validation nDCG@10 is requested
        if self.best_result is None or score > self.best_score:
            self.best_score, self.best_iteration = score, env.iteration
            self.best_result = list(result)
        if now + max(15.0, 2 * max(self.costs)) >= self.deadline:
            raise EarlyStopException(self.best_iteration, self.best_result)


def _group_sizes(qids):
    sizes, last = [], object()
    for q in qids:
        if q == last:
            sizes[-1] += 1
        else:
            sizes.append(1)
            last = q
    return sizes


def _bounded_groups(groups, limit=10000):
    """Native LambdaRank limits each query to 10,000 rows.

    Oversized queries train on contiguous within-query blocks, never mixing
    original queries. Ordinary inputs keep exactly their original grouping.
    Validation must still score complete original queries, not these blocks.
    """
    result = []
    for size in groups:
        while size > limit:
            result.append(limit)
            size -= limit
        result.append(size)
    return result


def _whole_query_metric(labels, groups):
    """Stable-tie nDCG@10, with zero for queries having no relevant document."""
    bounds = np.r_[0, np.cumsum(groups)]
    gains = np.exp2(labels.astype(np.float64)) - 1.
    discount = 1. / np.log2(np.arange(2, 12))
    ideals = [float(np.dot(np.sort(gains[a:b])[::-1][:10], discount[:min(10,b-a)]))
              for a,b in zip(bounds[:-1],bounds[1:])]
    def evaluate(predictions, dataset):
        total = 0.
        for a,b,ideal in zip(bounds[:-1],bounds[1:],ideals):
            if ideal > 0:
                order = np.argsort(-predictions[a:b], kind='stable')[:10]
                total += float(np.dot(gains[a:b][order], discount[:len(order)])) / ideal
        return 'whole_query_ndcg@10', total / len(groups), True
    return evaluate


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


def _read(path, counts, width, *, saturate=False):
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
                    value = float(value)
                    # NaN is LightGBM's native missing value. Reject explicit
                    # infinities; finite overflow requires the transfer opt-in.
                    if math.isinf(value):
                        raise ValueError("feature value must fit float32 or be NaN")
                    if abs(value) > _FLOAT32_MAX:
                        if not saturate:
                            raise ValueError("feature value must fit float32 or be NaN")
                        # Preserve the representable ordering used by float32
                        # inference without guessing feature/sentinel meanings.
                        value = math.copysign(_FLOAT32_MAX, value)
                    X[row, int(index) - 1] = value
            except (ValueError, IndexError, KeyError) as exc:
                raise ValueError(f"{path}:{line_no}: malformed LTR row ({exc})") from exc
    return X, y, list(counts.values())


def _fit_candidate(X, y, groups, Xv, yv, vali_groups, seed, deadline,
                   names='auto', fallback_path=None):
    """Same shallow LambdaMART for both feature representations."""
    import lightgbm as lgb
    params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
        "num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 50, "lambda_l2": 1.0,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed,
        "data_random_seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed,
        "deterministic": True, "force_row_wise": True, "verbose": -1, "num_threads": 4,
    }
    bounded_train, bounded_vali = _bounded_groups(groups), _bounded_groups(vali_groups)
    evaluation = None
    if bounded_train != groups or bounded_vali != vali_groups:
        # Native validation would otherwise average artificial blocks. Use the
        # staff convention on the original complete queries in this edge case.
        params['metric'] = 'None'
        evaluation = _whole_query_metric(yv, vali_groups)
    training = lgb.Dataset(X, label=y, group=bounded_train, feature_name=names)
    validation = lgb.Dataset(Xv, label=yv, group=bounded_vali, reference=training)
    callbacks = [_TrainingDeadline(deadline),
                 lgb.early_stopping(50, first_metric_only=True, verbose=False)]
    if fallback_path is not None:
        callbacks.append(_FirstTreeFallback(fallback_path))
    return lgb.train(params, training, num_boost_round=800, valid_sets=[validation],
                     feval=evaluation, callbacks=callbacks)


def _query_rank_features(X, groups, deadline):
    """Budgeted, label-free transform shared with the existing inference path.

    Returning None intentionally skips optional optimization. A single query
    transform, like a native boosting iteration, cannot be preempted.
    """
    from submission.ranking import _augment_query as augment_query
    if sum(groups) != len(X):
        raise ValueError('Query groups do not cover the feature matrix')
    if time.monotonic() >= deadline:
        return None
    out = np.empty((len(X), 2 * X.shape[1]), dtype=np.float32)
    offset = 0
    for size in groups:
        if time.monotonic() >= deadline:
            return None
        if size == 2:
            # The exact two-document percentile has only 0/1 values; constant
            # columns stay zero. Vectorize this common small-query case while
            # matching inference's NaN ordering and float32 tie semantics.
            raw = np.asarray(X[offset:offset+2], dtype=np.float32)
            missing = np.isnan(raw)
            width = raw.shape[1]
            out[offset:offset+2, :width] = raw
            out[offset, width:] = (raw[0] > raw[1]) | (missing[0] & ~missing[1])
            out[offset+1, width:] = (raw[1] > raw[0]) | (missing[1] & ~missing[0])
        else:
            out[offset:offset+size] = augment_query(X[offset:offset+size])
        offset += size
    return out if time.monotonic() < deadline else None


def train(train_path: str, vali_path: str, out_model_path: str, seed: int = 0) -> None:
    started = time.monotonic()
    deadline = started + _TIME_BUDGET_SECONDS
    import lightgbm as lgb

    train_counts, train_width = _scan(train_path)
    vali_counts, vali_width = _scan(vali_path)
    width = max(train_width, vali_width)
    if not train_counts or not vali_counts or width == 0:
        raise ValueError("TRAIN and validation must be nonempty with at least one feature")
    X, y, groups = _read(train_path, train_counts, width, saturate=True)
    Xv, yv, vali_groups = _read(vali_path, vali_counts, width, saturate=True)
    # Preserve the reliable raw baseline, including its full shared deadline.
    # Only TRAIN supplies gradients; validation selects stopping/representation.
    baseline_started = time.monotonic()
    booster = _fit_candidate(X, y, groups, Xv, yv, vali_groups, seed, deadline,
                             fallback_path=out_model_path)
    _save_model(booster, out_model_path)
    metric = _whole_query_metric(yv, vali_groups)
    baseline_score = metric(booster.predict(Xv, num_threads=4), None)[1]
    baseline_seconds = time.monotonic() - baseline_started
    del booster
    if baseline_score >= .999 or deadline-time.monotonic() < max(30., 3*baseline_seconds):
        return

    # Reserve measured training cost before allocating/constructing the optional
    # representation. Both transforms use complete original queries, even when
    # native training must split an oversized query into blocks.
    transform_deadline = deadline - max(30., 2*baseline_seconds)
    relative = _query_rank_features(X, groups, transform_deadline)
    if relative is None:
        return
    relative_valid = _query_rank_features(Xv, vali_groups, transform_deadline)
    if relative_valid is None:
        return
    X, Xv = relative, relative_valid  # release raw matrices before fitting
    from submission.ranking import _query_feature_names as feature_names
    candidate = _fit_candidate(X, y, groups, Xv, yv, vali_groups, seed, deadline,
                               names=feature_names(width))
    score = metric(candidate.predict(Xv, num_threads=4), None)[1]
    # A fixed conservative tie preference avoids selecting tiny validation noise.
    # Only the winning native model is needed by prepare(), with no sidecars.
    if score > baseline_score + .001:
        _save_model(candidate, out_model_path)
