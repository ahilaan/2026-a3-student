"""Build 2023CS10076.zip only after the released-DEV artifact preflight passes."""
import argparse
import os
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.preflight import check

REQUIRED = (
    "submission/ranking.py", "submission/fusion.py", "submission/pointwise_pairwise.py",
    "submission/recipe.py", "submission/model.txt", "submission/pointwise_model.pt",
    "submission/pairwise_model.pt",
)


def source_files(root):
    """Keep starter sources/docs and required weights; exclude data and run outputs."""
    selected = set(REQUIRED)
    selected.update(("README.md", "requirements.txt", "conftest.py", "data/README.md",
                     "a3-rankfusion-LTR.pdf", "a3-rankfusion-LTR.tex"))
    for folder in ("submission", "harness", "scripts", "tests", "docs"):
        for path in (root / folder).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".md", ".sh") and "__pycache__" not in path.parts:
                selected.add(path.relative_to(root).as_posix())
    missing = sorted(name for name in selected if not (root / name).is_file())
    if missing:
        raise FileNotFoundError("Missing submission files: " + ", ".join(missing))
    return sorted(selected)


def build(root, out):
    files = source_files(root)
    temporary = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        # Keep the provided directory layout while excluding all dataset files.
        for folder in ("data/", "data/train/", "data/dev/", "data/toy/", "data/toy_transfer/",
                       "data/toy_transfer/a_40_features_labels0-4/",
                       "data/toy_transfer/b_90_features_labels0-4/",
                       "data/toy_transfer/c_250_features_labels0-2/"):
            package.writestr(folder, b"")
        for name in files:
            package.write(root / name, name)
    with zipfile.ZipFile(temporary) as package:
        if package.testzip() is not None or not set(REQUIRED) <= set(package.namelist()):
            raise ValueError("ZIP integrity or required-file check failed")
        if temporary.stat().st_size > 50 * 1024 * 1024:
            raise ValueError("ZIP exceeds the portal's 50 MB limit")
    os.replace(temporary, out)
    return len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", default="data/dev/vali.txt", help="authorized released MSLR DEV file")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    dev = Path(args.dev).resolve()
    if not dev.is_file():
        raise SystemExit(f"Cannot build a verified submission: released DEV file is missing: {dev}")
    result = check(dev, root / "submission")
    if not result["passes"]:
        raise SystemExit("Cannot build: pairwise learning floor or inference contract failed")
    out = root / "2023CS10076.zip"
    count = build(root, out)
    print(f"Preflight passed; packaged {count} files, including all 7 required entries, into {out}")


if __name__ == "__main__":
    main()
