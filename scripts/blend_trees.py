"""Serialize a convex LambdaMART ensemble as one ordinary LightGBM artifact.

Only models trained on released relevance labels are combined. Leaves are
scaled and trees concatenated with LightGBM's own model-merge API. Inference
remains the standard, dimension-independent Booster.predict path.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
from lightgbm.basic import _LIB, _safe_call


def blend_models(first, second, weight=0.5, first_trees=-1, second_trees=-1):
    if not 0 <= weight <= 1:
        raise ValueError("convex weights required")
    boosters = []
    for path, scale, count in ((first, weight, first_trees), (second, 1 - weight, second_trees)):
        source = lgb.Booster(model_file=str(path))
        model = lgb.Booster(model_str=source.model_to_string(num_iteration=count))
        for tree in model.dump_model()["tree_info"]:
            for leaf in range(tree["num_leaves"]):
                value = model.get_leaf_output(tree["tree_index"], leaf)
                model.set_leaf_output(tree["tree_index"], leaf, value * scale)
        boosters.append(model)
    a, b = boosters
    if a.feature_name() != b.feature_name() or a.num_model_per_iteration() != b.num_model_per_iteration():
        raise ValueError("feature/class dimensions must match")
    # This is the same C API called by LightGBM's own warm-start/refit code.
    # It is needed only offline; the saved file is a normal LightGBM model.
    _safe_call(_LIB.LGBM_BoosterMerge(a._handle, b._handle))
    return a
