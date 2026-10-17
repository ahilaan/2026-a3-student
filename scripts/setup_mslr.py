"""Download Microsoft's checksum-verified WEB10K archive; extract only Fold1 TRAIN/DEV.

The current Microsoft OneDrive endpoint can block automated downloads. The
fallback serves the identical ZIP, accepted only after the published MD5 matches.
No test file, or file from another fold, is opened or extracted.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://storage.googleapis.com/personalization-takehome/MSLR-WEB10K.zip"
OFFICIAL_PAGE = "https://www.microsoft.com/en-us/research/project/mslr/"
MD5 = "97c5d4e7c171e475c91d7031e4fd8e79"


def checksum(path):
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    cache = ROOT / "data" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "MSLR-WEB10K.zip"
    if not archive.is_file():
        partial = archive.with_suffix(".zip.part")
        start, received, last_report = time.monotonic(), 0, 0
        request = urllib.request.Request(URL, headers={"User-Agent": "MSLR-course-setup/1.0"})
        with urllib.request.urlopen(request, timeout=90) as source, partial.open("wb") as destination:
            expected_bytes = int(source.headers["Content-Length"])
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                destination.write(chunk)
                received += len(chunk)
                if received - last_report >= 64 * 1024 * 1024:
                    print(f"Downloaded {received / 1024**2:.0f}/{expected_bytes / 1024**2:.0f} MiB "
                          f"in {time.monotonic() - start:.1f}s", flush=True)
                    last_report = received
        if received != expected_bytes:
            raise ValueError(f"Incomplete download: {received}/{expected_bytes} bytes")
        partial.replace(archive)
    actual_md5 = checksum(archive)
    if actual_md5 != MD5:
        raise ValueError(f"Archive does not match Microsoft's published MD5: {actual_md5}")
    print(f"Archive matches Microsoft MD5: {actual_md5}", flush=True)

    allowed = {"train.txt", "vali.txt"}
    fold = cache / "Fold1"
    fold.mkdir(exist_ok=True)
    selected = []
    with zipfile.ZipFile(archive) as package:
        # Reading the directory is metadata only. Open exactly these two files.
        for basename in sorted(allowed):
            matches = [info for info in package.infolist()
                       if info.filename.replace("\\", "/").rstrip("/").split("/")[-2:] == ["Fold1", basename]]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one Fold1/{basename} in official archive")
            info = matches[0]
            target = fold / basename
            print(f"Extracting only {info.filename} ({info.file_size} bytes)", flush=True)
            with package.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            selected.append(info.filename)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_mslr.py"),
                    "--fold-dir", str(fold), "--out", str(ROOT / "data")], check=True)
    provenance = {"official_source": OFFICIAL_PAGE, "download_url": URL, "archive_md5": actual_md5,
                  "archive_bytes": archive.stat().st_size, "extracted_entries": selected,
                  "held_out_files_opened": False}
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "mslr_data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("MSLR Fold1 TRAIN and DEV are ready; no held-out files were opened.", flush=True)


if __name__ == "__main__":
    main()
