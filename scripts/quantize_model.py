"""Quantize the distilled student to INT8 (dynamic quantization), save it, reload it from
disk and verify that the reloaded INT8 model runs and is evaluated on val/test.

Usage: python scripts/quantize_model.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data.dataset import load_split  # noqa: E402
from src.evaluation.metrics import compute_metrics, save_predictions, update_threshold  # noqa: E402
from src.quantization.model import (QUANT_DIR, QUANT_WEIGHTS, STUDENT_DIR, count_quantized_modules, is_fully_quantized,  # noqa: E402
                                    load_quantized, quantize, save_quantized, set_quant_engine)
from src.transformer.model import save_json  # noqa: E402


def dir_size_mb(path: Path, pattern: str) -> float:
    return sum(p.stat().st_size for p in path.glob(pattern)) / 1e6


def main() -> None:
    warnings.filterwarnings("ignore")
    engine = set_quant_engine()
    tok = AutoTokenizer.from_pretrained(STUDENT_DIR)
    fp32 = AutoModelForSequenceClassification.from_pretrained(STUDENT_DIR).eval()
    q = quantize(fp32)
    modules = count_quantized_modules(q)
    save_quantized(q, tok, {"method": "PyTorch dynamic quantization (torch.ao.quantization.quantize_dynamic)",
                            "linear": "qint8 weights, dynamic per-batch activation quantization",
                            "embedding": "quint8 weight-only, per-channel float qparams",
                            "engine": engine, "torch": torch.__version__, "modules": modules})

    # verify: reload from disk and run
    qclf = load_quantized()
    reloaded = count_quantized_modules(qclf.model)
    assert is_fully_quantized(qclf.model), f"model is not fully quantized: {reloaded}"
    val, test = load_split("val"), load_split("test")
    vp = qclf.predict_proba_texts(val.normalized.tolist())
    tp = qclf.predict_proba_texts(test.normalized.tolist())
    save_predictions("quantized", vp, tp)
    thr = update_threshold("quantized", vp, val.y.to_numpy())
    fp32_test = np.load(C.ARTIFACTS / "predictions" / "student_fp32_test.npy")
    agree = float((fp32_test.argmax(1) == tp.argmax(1)).mean())
    log = {"engine": engine, "quantized_modules": modules, "reloaded_modules": reloaded,
           "student_fp32_weights_mb": dir_size_mb(STUDENT_DIR, "*.safetensors"),
           "int8_weights_mb": QUANT_WEIGHTS.stat().st_size / 1e6,
           "argmax_agreement_with_fp32_student_on_test": agree, "threshold": thr,
           "val": compute_metrics(vp, val.y.to_numpy(), thr),
           "test": compute_metrics(tp, test.y.to_numpy(), thr, test)}
    save_json(C.REPORTS / "training_logs" / "quantization.json", log)
    print({k: log[k] for k in ["student_fp32_weights_mb", "int8_weights_mb", "argmax_agreement_with_fp32_student_on_test"]})
    print({k: log["test"][k] for k in ["accuracy", "macro_f1", "oos_recall"]}, modules)
    print("saved", QUANT_DIR)


if __name__ == "__main__":
    main()
