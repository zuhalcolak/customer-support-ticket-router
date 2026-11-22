#!/usr/bin/env bash
# Reproduce the whole project from scratch.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/download_dataset.py
python scripts/prepare_dataset.py
python scripts/train_classical.py
python scripts/train_neural.py
python scripts/train_transformer.py
python scripts/distill_student.py
python scripts/quantize_model.py
python scripts/evaluate_all.py
python scripts/make_architecture_diagram.py
python scripts/update_readme_results.py
python -m pytest -q
