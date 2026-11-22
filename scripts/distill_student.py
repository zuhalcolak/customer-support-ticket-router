"""Knowledge distillation: XLM-R teacher -> multilingual DistilBERT student.

Response-based KD: the student is trained on the teacher's soft labels (temperature T)
plus the gold labels:  loss = alpha*T^2*KL(teacher_T || student_T) + (1-alpha)*CE(student, y).
The teacher logits were computed by scripts/train_transformer.py on the TRAIN split only.
Teacher and student use different tokenizers; this is fine because only output
distributions over the same 19 labels are matched.

Usage: python scripts/distill_student.py [--epochs 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data.dataset import load_split  # noqa: E402
from src.evaluation.metrics import compute_metrics, save_predictions, update_threshold  # noqa: E402
from src.quantization.model import STUDENT_DIR, STUDENT_NAME  # noqa: E402
from src.transformer.model import HFClassifier, new_classifier, pick_device, save_json, train_classifier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--temperature", type=float, default=2.0)
    args = ap.parse_args()
    train, val, test = (load_split(s) for s in ["train", "val", "test"])
    teacher_logits = np.load(C.ARTIFACTS / "transformer" / "teacher_train_logits.npy")
    assert len(teacher_logits) == len(train)
    device = pick_device()
    tok, model = new_classifier(STUDENT_NAME)
    info = train_classifier(tok, model, train.normalized.tolist(), train.y.tolist(),
                            val.normalized.tolist(), val.y.tolist(), epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size, device=device, out_dir=STUDENT_DIR,
                            teacher_logits=teacher_logits, alpha=args.alpha,
                            temperature=args.temperature, log_name="student")
    best = HFClassifier.load(STUDENT_DIR, device=str(device))
    vp = best.predict_proba_texts(val.normalized.tolist())
    tp = best.predict_proba_texts(test.normalized.tolist())
    save_predictions("student_fp32", vp, tp)
    thr = update_threshold("student_fp32", vp, val.y.to_numpy())
    log = {"student_base": STUDENT_NAME, "teacher": "artifacts/transformer (XLM-R base fine-tuned)",
           "device": str(device), "config": vars(args) | {"max_len": 64, "optimizer": "AdamW"},
           **info, "threshold": thr,
           "val": compute_metrics(vp, val.y.to_numpy(), thr),
           "test": compute_metrics(tp, test.y.to_numpy(), thr, test)}
    save_json(C.REPORTS / "training_logs" / "student_distillation.json", log)
    print({k: log["test"][k] for k in ["accuracy", "macro_f1", "oos_recall"]})


if __name__ == "__main__":
    main()
