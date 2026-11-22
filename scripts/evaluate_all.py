"""Evaluate all four models (plus the FP32 student, for the quantization comparison) on the
SAME held-out test set, on CPU, and benchmark size and latency.

Outputs (reports/):
  results.csv / results.json            main results table
  quantization_comparison.csv           teacher vs FP32 student vs INT8 student
  per_class_<model>.csv                 per-intent precision / recall / F1
  failure_examples.csv                  real misclassified test messages
  hardware.json                         hardware / software description
  figures/*.png                         bar charts and confusion matrices

Usage: python scripts/evaluate_all.py [--runs 200] [--threads 4]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import classification_report  # noqa: E402

from src import config as C  # noqa: E402
from src.data.dataset import load_split  # noqa: E402
from src.evaluation.benchmark import benchmark_latency, hardware_info, model_size_mb  # noqa: E402
from src.evaluation.metrics import compute_metrics, confusion  # noqa: E402
from src.pipeline import load_model  # noqa: E402
from src.preprocessing.text import preprocess  # noqa: E402
from src.routing.router import apply_threshold, load_thresholds  # noqa: E402

DISPLAY = dict(C.MODEL_NAMES, student_fp32="Distilled mDistilBERT (FP32)")
ORDER = ["classical", "neural", "transformer", "quantized"]
COLORS = {"classical": "#6b7280", "neural": "#2563eb", "transformer": "#7c3aed", "quantized": "#ea580c",
          "student_fp32": "#f59e0b"}


def load_student_fp32():
    from src.quantization.model import STUDENT_DIR
    from src.transformer.model import HFClassifier
    return HFClassifier.load(STUDENT_DIR, device="cpu")


def bar(df: pd.DataFrame, col: str, title: str, fname: str, fmt: str = "{:.3f}", log=False) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(df.display_name, df[col], color=[COLORS[m] for m in df.model])
    for i, v in enumerate(df[col]):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=12, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / fname, dpi=160)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, title: str, fname: str) -> None:
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(C.LABELS)), C.LABELS, rotation=90, fontsize=7)
    ax.set_yticks(range(len(C.LABELS)), C.LABELS, fontsize=7)
    for i in range(len(C.LABELS)):
        for j in range(len(C.LABELS)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted (after threshold)")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(C.FIGURES / fname, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    torch.set_num_threads(args.threads)
    C.FIGURES.mkdir(parents=True, exist_ok=True)

    test = load_split("test")
    y = test.y.to_numpy()
    thresholds = load_thresholds()
    pre_test = [preprocess(t) for t in test.text]
    rng = random.Random(C.SEED)
    bench_idx = rng.sample(range(len(test)), args.runs)
    bench_inputs = [pre_test[i] for i in bench_idx]

    # preprocessing latency (shared by all models)
    from src.evaluation.benchmark import benchmark_latency as bl
    pre_lat = bl(lambda t: preprocess(t), [test.text[i] for i in bench_idx], args.warmup, args.runs)

    rows, probs_all = [], {}
    for name in ORDER + ["student_fp32"]:
        print("evaluating", name, flush=True)
        model = load_student_fp32() if name == "student_fp32" else load_model(name)
        probs = model.predict_proba(pre_test)  # CPU, batched, full test set
        probs_all[name] = probs
        thr = thresholds[name]
        m = compute_metrics(probs, y, thr, test)
        lat = benchmark_latency(lambda p: model.predict_proba([p]), bench_inputs, args.warmup, args.runs)
        rows.append({"model": name, "display_name": DISPLAY[name], **m, "size_mb": model_size_mb(name), **lat})
        np.save(C.ARTIFACTS / "predictions" / f"{name}_test_cpu.npy", probs)
        rep = classification_report(y, apply_threshold(probs, thr), labels=list(range(len(C.LABELS))),
                                    target_names=C.LABELS, output_dict=True, zero_division=0)
        pd.DataFrame(rep).T.to_csv(C.REPORTS / f"per_class_{name}.csv")
        if name in ORDER:
            plot_confusion(confusion(probs, y, thr), f"Confusion matrix - {DISPLAY[name]}", f"confusion_{name}.png")
        del model

    df = pd.DataFrame(rows)
    main_df = df[df.model.isin(ORDER)].copy()
    cols = ["model", "display_name", "accuracy", "macro_f1", "oos_recall", "size_mb", "latency_mean_ms",
            "latency_median_ms", "latency_p95_ms", "macro_precision", "macro_recall", "inscope_accuracy",
            "oos_precision", "threshold", "accuracy[lang=en]", "accuracy[lang=tr]", "oos_recall[seen]",
            "oos_recall[unseen]", "accuracy[source=manual]", "accuracy[source=massive]",
            "accuracy_no_threshold", "oos_recall_no_threshold", "runs", "warmup", "batch_size", "n", "n_oos"]
    main_df[cols].to_csv(C.REPORTS / "results.csv", index=False)
    q = df[df.model.isin(["transformer", "student_fp32", "quantized"])][
        ["display_name", "accuracy", "macro_f1", "oos_recall", "size_mb", "latency_mean_ms", "latency_median_ms"]]
    q.to_csv(C.REPORTS / "quantization_comparison.csv", index=False)
    hw = hardware_info() | {"benchmark_threads": args.threads}
    (C.REPORTS / "hardware.json").write_text(json.dumps(hw, indent=2))
    (C.REPORTS / "results.json").write_text(json.dumps(
        {"hardware": hw, "preprocessing_latency": pre_lat, "test_set": {"n": int(len(y)), "n_oos": int((y == C.LABEL2ID[C.OOS_LABEL]).sum())},
         "benchmark": {"batch_size": 1, "warmup_runs": args.warmup, "timed_runs": args.runs,
                       "inputs": "random sample (seed 42) of preprocessed test messages",
                       "what_is_timed": "model.predict_proba on one preprocessed message (tokenization/vectorization + forward + softmax)"},
         "models": df.to_dict(orient="records")}, indent=2))

    # failure examples (real test messages)
    pred = {n: apply_threshold(probs_all[n], thresholds[n]) for n in ORDER}
    fails = test[["uid", "lang", "source", "oos_type", "label", "text"]].copy()
    for n in ORDER:
        fails[f"pred_{n}"] = [C.ID2LABEL[i] for i in pred[n]]
        fails[f"conf_{n}"] = probs_all[n].max(1).round(3)
    wrong_q = pred["quantized"] != y
    fails[wrong_q].to_csv(C.REPORTS / "failure_examples.csv", index=False)

    # plots
    for col, title, fname, fmt, log in [("accuracy", "Accuracy (test, after threshold)", "accuracy.png", "{:.3f}", False),
                                        ("macro_f1", "Macro-F1 (test)", "macro_f1.png", "{:.3f}", False),
                                        ("oos_recall", "Out-of-scope recall (test)", "oos_recall.png", "{:.3f}", False),
                                        ("size_mb", "Model size on disk (MB, log scale)", "model_size.png", "{:.1f}", True),
                                        ("latency_mean_ms", "Mean CPU latency per message (ms, batch 1, log scale)", "latency.png", "{:.2f}", True)]:
        bar(main_df, col, title, fname, fmt, log)

    print(main_df[["display_name", "accuracy", "macro_f1", "oos_recall", "size_mb", "latency_mean_ms"]].to_string(index=False))
    print(q.to_string(index=False))
    print("failures (quantized):", int(wrong_q.sum()))


if __name__ == "__main__":
    main()
