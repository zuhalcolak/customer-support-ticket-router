"""Train Model 3: fine-tune XLM-RoBERTa base for intent classification (teacher).

Usage: python scripts/train_transformer.py [--epochs 5]
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data.dataset import load_split  # noqa: E402
from src.evaluation.metrics import compute_metrics, save_predictions, update_threshold  # noqa: E402
from src.transformer.model import (TEACHER_DIR, TEACHER_NAME, HFClassifier, new_classifier,  # noqa: E402
                                   pick_device, save_json, train_classifier)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    train, val, test = (load_split(s) for s in ["train", "val", "test"])
    device = pick_device()
    tok, model = new_classifier(TEACHER_NAME)
    info = train_classifier(tok, model, train.normalized.tolist(), train.y.tolist(),
                            val.normalized.tolist(), val.y.tolist(), epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size, device=device, out_dir=TEACHER_DIR, log_name="teacher")
    best = HFClassifier.load(TEACHER_DIR, device=str(device))
    vp = best.predict_proba_texts(val.normalized.tolist())
    tp = best.predict_proba_texts(test.normalized.tolist())
    # Teacher logits on the training set are needed for distillation
    tr_logits = best.logits_texts(train.normalized.tolist())
    import numpy as np
    np.save(C.ARTIFACTS / "transformer" / "teacher_train_logits.npy", tr_logits)
    save_predictions("transformer", vp, tp)
    thr = update_threshold("transformer", vp, val.y.to_numpy())
    log = {"base_model": TEACHER_NAME, "device": str(device), "torch": torch.__version__,
           "platform": platform.platform(), "processor": platform.processor(),
           "config": {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "max_len": 64,
                      "optimizer": "AdamW", "weight_decay": 0.01, "warmup_ratio": 0.1, "scheduler": "linear"},
           **info, "threshold": thr,
           "val": compute_metrics(vp, val.y.to_numpy(), thr),
           "test": compute_metrics(tp, test.y.to_numpy(), thr, test)}
    save_json(C.REPORTS / "training_logs" / "transformer.json", log)
    print({k: log["test"][k] for k in ["accuracy", "macro_f1", "oos_recall"]})


if __name__ == "__main__":
    main()
