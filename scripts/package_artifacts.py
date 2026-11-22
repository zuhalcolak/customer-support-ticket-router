"""Zip the trained model artifacts for upload (Hugging Face Hub / Google Drive).

Usage: python scripts/package_artifacts.py   ->  artifacts.zip in the project root
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["artifacts/classical", "artifacts/neural", "artifacts/transformer", "artifacts/student",
           "artifacts/quantized"]  # thresholds.json and known_words.txt are tracked in Git
SKIP_NAMES = {"teacher_train_logits.npy"}  # only needed to re-run distillation


def main() -> None:
    out = ROOT / "artifacts.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for inc in INCLUDE:
            p = ROOT / inc
            files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
            for f in files:
                if f.name not in SKIP_NAMES:
                    z.write(f, f.relative_to(ROOT))
    print(f"{out} {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
