"""Generate and train a large, deterministic transfer workload (no real labels).

Default: 500,000 TRAIN rows, 50,000 validation rows, 300 features, about
11.5% nonzero entries. This is a sparse stress check, not a dense worst case.
"""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np
from submission import recipe, ranking


def peak_memory_bytes():
    if sys.platform == "win32":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage"]]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def write(path, rows, width, density, seed):
    rng = np.random.RandomState(seed)
    with path.open("w", encoding="ascii") as target:
        for offset in range(0, rows, 1000):
            X = rng.randn(min(1000, rows - offset), width).astype(np.float32)
            keep = rng.rand(*X.shape) < density
            keep[:, :5] = True
            X *= keep
            y = np.digitize(X[:, 0] - X[:, 1] + 0.4 * X[:, 2] + rng.randn(len(X)) * .8, [-1., -.3, .3, 1.])
            for i, (x, label) in enumerate(zip(X, y)):
                tokens = " ".join(f"{j+1}:{x[j]:.6g}" for j in np.flatnonzero(x))
                target.write(f"{label} qid:{(offset+i)//100} {tokens} # generated stress\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=500000)
    ap.add_argument("--width", type=int, default=300)
    ap.add_argument("--density", type=float, default=.1)
    ap.add_argument("--name", default="stress_sparse500k")
    ap.add_argument("--output-dir", type=Path, default=ROOT/'runs/c1_repro')
    args = ap.parse_args()
    if args.rows < 100 or args.width < 5 or not 0 < args.density <= 1:
        ap.error("need at least 100 rows, 5 features and density in (0, 1]")
    directory = ROOT / "tmp/c1_stress" / args.name
    directory.mkdir(parents=True, exist_ok=True)
    signature = {"rows": args.rows, "width": args.width, "density": args.density, "generator": 1}
    manifest = directory / "inputs.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("stress input parameters changed: choose a new --name")
    manifest.write_text(json.dumps(signature, indent=2))
    for name, rows, seed in [("train", args.rows, 81), ("vali", max(1000, args.rows // 10), 82)]:
        path = directory / (name + ".txt")
        if not path.exists():
            temporary = path.with_suffix(".partial")
            write(temporary, rows, args.width, args.density, seed)
            temporary.replace(path)
    print("Stress input generated", args.name, flush=True)
    started = time.perf_counter()
    out = directory / "model.txt"
    fits=[]
    original_fit=getattr(recipe,'_fit_candidate',None)
    if original_fit is not None:
        def measured_fit(*fit_args,**fit_kwargs):
            begin=time.perf_counter()
            booster=original_fit(*fit_args,**fit_kwargs)
            fits.append({'features':booster.num_feature(),'trees':booster.num_trees(),
                         'seconds':time.perf_counter()-begin})
            return booster
        recipe._fit_candidate=measured_fit
    try:
        recipe.train(str(directory / "train.txt"), str(directory / "vali.txt"), str(out), seed=0)
    finally:
        if original_fit is not None:recipe._fit_candidate=original_fit
    elapsed = time.perf_counter() - started
    ranking.prepare(str(out))
    X = np.zeros((5, args.width), dtype=np.float32)
    result = ranking.rerank_with_ltr("stress", list("abcde"), dict(zip("abcde", X)), 5)
    model = lgb.Booster(model_file=str(out))
    record = {"rows": args.rows, "validation_rows": max(1000, args.rows // 10), "width": args.width,
              "background_density": args.density, "first_five_features_dense": True,
              "seconds": elapsed, "peak_working_set_bytes": peak_memory_bytes(),
              "trees": model.num_trees(), "loadable": len(result) == 5,
              "model_features":model.num_feature(),
              "query_rank_features":ranking._QUERY_RANK_FEATURES,
              "candidate_fits":fits,
              "platform": sys.platform, "threads": 4, "lightgbm": lgb.__version__}
    import hashlib
    record['sha256'] = hashlib.sha256(out.read_bytes()).hexdigest()
    record['model_bytes'] = out.stat().st_size
    args.output_dir.mkdir(parents=True,exist_ok=True)
    target = args.output_dir / (args.name + ".json")
    target.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
