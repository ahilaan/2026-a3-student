"""Compare preserved and final C1 recipes through the unmodified staff harness."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.recipe_eval import run_recipe_and_evaluate
from submission import recipe


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,default=ROOT/'runs/c1_repro/baseline/recipe.py')
    ap.add_argument('--output',type=Path,default=ROOT/'runs/c1_repro/recipe_regression.json')
    args=ap.parse_args()
    baseline = args.baseline
    spec = importlib.util.spec_from_file_location("preserved_recipe", baseline)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    original = recipe.train
    results = []
    with tempfile.TemporaryDirectory() as directory:
        for data in sorted((ROOT / "data/toy_transfer").iterdir()):
            train, valid = str(data / "train.txt"), str(data / "vali.txt")
            paths = [Path(directory) / f"{kind}.txt" for kind in ["baseline", "final", "repeat"]]
            scores = []
            try:
                for method, path in zip([old.train, original, original], paths):
                    recipe.train = method
                    scores.append(run_recipe_and_evaluate(train, valid, valid, str(path), seed=0))
            finally:
                recipe.train = original
            result = {"dataset": data.name, "baseline": scores[0], "final": scores[1],
                      "baseline_final_byte_equal": paths[0].read_bytes() == paths[1].read_bytes(),
                      "same_seed_repeat_byte_equal": paths[1].read_bytes() == paths[2].read_bytes()}
            assert result["same_seed_repeat_byte_equal"]
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
