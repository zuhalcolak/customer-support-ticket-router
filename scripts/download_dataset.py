"""Download MASSIVE 1.1 (Amazon, CC BY 4.0) and extract the en-US and tr-TR files.

Usage: python scripts/download_dataset.py
"""
from __future__ import annotations

import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_RAW, MASSIVE_URL  # noqa: E402

WANTED = ["1.1/data/en-US.jsonl", "1.1/data/tr-TR.jsonl", "1.1/LICENSE", "1.1/NOTICE.md", "1.1/CITATION.md"]


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    archive = DATA_RAW / "massive-1.1.tar.gz"
    if not archive.exists():
        print(f"Downloading {MASSIVE_URL} ...")
        urllib.request.urlretrieve(MASSIVE_URL, archive)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(f"archive: {archive} ({archive.stat().st_size/1e6:.1f} MB) sha256={sha}")
    out = DATA_RAW / "massive"
    out.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for name in WANTED:
            member = tar.getmember(name)
            # only regular files with the expected relative path are extracted
            if not member.isfile() or member.name.startswith("/") or ".." in member.name:
                raise RuntimeError(f"unexpected archive member {member.name}")
            tar.extract(member, out, filter="data")
            print("extracted", out / name)


if __name__ == "__main__":
    main()
