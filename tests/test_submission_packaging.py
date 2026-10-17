"""Guard against missing grader inputs and accidental dataset/credential uploads."""
from pathlib import Path
import zipfile

from scripts.build_submission import REQUIRED, build, source_files


def test_package_contains_required_files_and_excludes_private_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    selected = source_files(root)
    assert set(REQUIRED) <= set(selected)
    assert not any(name.startswith(("tmp/", "runs/", ".git/", ".venv/")) for name in selected)
    assert [name for name in selected if name.startswith("data/")] == ["data/README.md"]
    assert not any(name.endswith(("_history.json", ".pyc")) for name in selected)
    archive = tmp_path / "2023CS10076.zip"
    assert build(root, archive) == len(selected)
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert set(REQUIRED) <= set(package.namelist())
        for name in REQUIRED:
            assert package.read(name) == (root / name).read_bytes()
