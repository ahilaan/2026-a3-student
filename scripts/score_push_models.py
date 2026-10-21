"""TRAIN/DEV-only tree training; prior campaign sources are archived under runs/."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cv_query_features import combined_data
OUT = ROOT / 'runs/compliant_campaign'
CACHE = ROOT / 'data/cache/compliant_campaign'
CONFIGS = {
    'extra31': dict(num_leaves=31, min_data_in_leaf=100, lambda_l2=5, extra_trees=True, max_bin=63, feature_fraction=.7),
    'regular31': dict(num_leaves=31, min_data_in_leaf=100, lambda_l2=5, max_bin=63, feature_fraction=.8),
    'extra63': dict(num_leaves=63, min_data_in_leaf=100, lambda_l2=10, extra_trees=True, max_bin=63, feature_fraction=.8),
    'extra127': dict(num_leaves=127, min_data_in_leaf=100, lambda_l2=10, extra_trees=True, max_bin=63, feature_fraction=.8),
    'regular63': dict(num_leaves=63, min_data_in_leaf=100, lambda_l2=10, max_bin=63, feature_fraction=.8),
}
def matrices():
    """Build exclusively from the allowlisted TRAIN and DEV loaders."""
    return combined_data()
if __name__ == '__main__':
    from scripts.compliant_train import main
    main()
