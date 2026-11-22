"""Train Model 1: TF-IDF + Logistic Regression.  Usage: python scripts/train_classical.py"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.classical.model import ClassicalModel, build_pipeline  # noqa: E402
from src.data.dataset import load_split  # noqa: E402
from src.evaluation.metrics import compute_metrics, save_predictions, update_threshold  # noqa: E402


def main() -> None:
    train, val, test = (load_split(s) for s in ["train", "val", "test"])
    t0 = time.time()
    pipe = build_pipeline().fit(train.folded.tolist(), train.y)
    train_time = time.time() - t0
    model = ClassicalModel(pipe)
    model.save()
    vp = model.predict_proba_texts(val.folded.tolist())
    tp = model.predict_proba_texts(test.folded.tolist())
    save_predictions("classical", vp, tp)
    thr = update_threshold("classical", vp, val.y.to_numpy())
    log = {"train_seconds": train_time, "threshold": thr,
           "val": compute_metrics(vp, val.y.to_numpy(), thr),
           "test": compute_metrics(tp, test.y.to_numpy(), thr, test)}
    out = C.REPORTS / "training_logs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "classical.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
