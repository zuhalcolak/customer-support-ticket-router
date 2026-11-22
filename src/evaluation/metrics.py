"""Classification metrics shared by all models."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src import config as C
from src.routing.router import apply_threshold, tune_threshold

OOS_ID = C.LABEL2ID[C.OOS_LABEL]


def compute_metrics(probs: np.ndarray, y: np.ndarray, threshold: float, meta=None) -> dict:
    pred = apply_threshold(probs, threshold)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(len(C.LABELS))),
                                                 average="macro", zero_division=0)
    oos = y == OOS_ID
    res = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
        "oos_recall": float((pred[oos] == OOS_ID).mean()),
        "oos_precision": float((y[pred == OOS_ID] == OOS_ID).mean()) if (pred == OOS_ID).any() else 0.0,
        "inscope_accuracy": float((pred[~oos] == y[~oos]).mean()),
        "accuracy_no_threshold": float((probs.argmax(1) == y).mean()),
        "oos_recall_no_threshold": float((probs.argmax(1)[oos] == OOS_ID).mean()),
        "n": int(len(y)), "n_oos": int(oos.sum()),
    }
    if meta is not None:
        for col in ["lang", "oos_type", "source"]:
            for v in sorted(meta[col].unique()):
                m = (meta[col] == v).to_numpy()
                key = f"{col}={v}"
                if col == "oos_type" and v != "none":
                    res[f"oos_recall[{v}]"] = float((pred[m] == OOS_ID).mean())
                else:
                    res[f"accuracy[{key}]"] = float((pred[m] == y[m]).mean())
    return res


def confusion(probs: np.ndarray, y: np.ndarray, threshold: float) -> np.ndarray:
    return confusion_matrix(y, apply_threshold(probs, threshold), labels=list(range(len(C.LABELS))))


def save_predictions(name: str, val_probs, test_probs) -> None:
    d = C.ARTIFACTS / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}_val.npy", val_probs)
    np.save(d / f"{name}_test.npy", test_probs)


def update_threshold(name: str, val_probs, val_y) -> float:
    t, acc = tune_threshold(val_probs, val_y)
    path = C.ARTIFACTS / "thresholds.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = {"threshold": t, "val_accuracy_at_threshold": acc}
    path.write_text(json.dumps(data, indent=2))
    return t
