"""Draw docs/architecture.png (and .pdf). Every box names the source file that implements it.

Usage: python scripts/make_architecture_diagram.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GREY, BLUE, PURPLE, ORANGE, GREEN, RED, TEAL = ("#e5e7eb", "#dbeafe", "#ede9fe", "#ffedd5", "#dcfce7",
                                                 "#fee2e2", "#ccfbf1")


def box(ax, x, y, w, h, title, body="", color=GREY, edge="#374151", lw=1.2, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=edge, lw=lw))
    ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="top", fontsize=fs, weight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.47, body, ha="center", va="top", fontsize=fs - 1.6, linespacing=1.25)


def arrow(ax, a, b, text="", color="#374151", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, color=color, lw=1.2, linestyle=ls))
    if text:
        ax.text((a[0] + b[0]) / 2 + 0.05, (a[1] + b[1]) / 2, text, fontsize=6.5, color=color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.4)
    ax.axis("off")
    ax.text(0.2, 9.15, "Customer Support Ticket Router – system architecture (inference path runs on CPU)",
            fontsize=13, weight="bold")

    # run-time region
    ax.add_patch(FancyBboxPatch((0.15, 2.55), 15.7, 6.35, boxstyle="round,pad=0.02", fc="none",
                                ec="#9ca3af", ls="--"))
    ax.text(0.3, 8.7, "RUN TIME  ·  Streamlit app, CPU only (torch threads = 4)", fontsize=8, color="#4b5563")

    box(ax, 0.4, 6.9, 2.2, 1.55, "User", "types a Turkish / English\nmessage in the browser", TEAL)
    box(ax, 0.4, 4.9, 2.2, 1.6, "Streamlit UI", "app/main.py\ninput · examples · results\n4-model view · analytics", TEAL)
    box(ax, 3.1, 5.2, 3.0, 3.25, "Preprocessing (classical NLP)",
        "src/preprocessing/text.py\n\n1 language ID (tr / en)\n2 regex normalization:\n   URL / e-mail / phone removal,\n"
        "   Turkish lower-casing (I→ı, İ→i),\n   repeated letters, punctuation\n3 model-code regex  → DW-6534\n"
        "4 edit-distance spelling correction\n   of product terms + category\n5 ASCII folding (classical/neural input)", BLUE)
    box(ax, 3.1, 3.0, 3.0, 1.75, "ELIZA-style rules",
        "src/routing/smalltalk.py\nregex patterns → reply templates\n(greetings, thanks, small talk)\nshort-circuits to an auto-reply", BLUE)

    # models
    mx, mw, mh = 6.7, 4.1, 1.18
    box(ax, mx, 7.35, mw, mh, "① TF-IDF + Logistic Regression",
        "src/classical/model.py · scikit-learn · FP64\nword 1-2g + char 2-5g TF-IDF  (input: folded text)", GREY)
    box(ax, mx, 6.05, mw, mh, "② FastText + Feed-forward NN",
        "src/neural/model.py · gensim + PyTorch · FP32\navg. FastText(150d) → 256 → 128 → softmax (SGD)", BLUE)
    box(ax, mx, 4.75, mw, mh, "③ XLM-RoBERTa base (teacher)",
        "src/transformer/model.py · FP32 · 278M params\nSentencePiece subwords · self-attention · cls head", PURPLE)
    box(ax, mx, 3.3, mw, 1.3, "④ Distilled mDistilBERT – INT8",
        "src/quantization/model.py · 6 layers · INT8 dynamic\nWordPiece · Linear qint8 + Embedding quint8\n"
        "PRIMARY MODEL: makes the routing decision", ORANGE, edge="#ea580c", lw=2.2)

    box(ax, 11.4, 5.35, 2.15, 2.6, "Confidence &\nOOS logic",
        "\n\nsrc/routing/router.py\nsoftmax max-prob\n≥ threshold τ (per model,\ntuned on validation)\n"
        "arg-max = out_of_scope\n→ Human Agent", GREEN)
    box(ax, 11.4, 3.3, 2.15, 1.75, "Department map",
        "src/config.py\nINTENT_TO_DEPARTMENT\n19 labels → 7 targets", GREEN)
    box(ax, 13.85, 6.4, 1.85, 1.55, "Department", "Technical Support\nWarranty & Returns\nBilling · Orders\nSmart Home …", GREEN)
    box(ax, 13.85, 4.55, 1.85, 1.55, "Human Agent", "low confidence or\nout-of-scope", RED)
    box(ax, 13.85, 2.85, 1.85, 1.4, "Auto-reply", "greeting /\nsmall talk", TEAL)
    box(ax, 0.4, 2.85, 2.2, 1.75, "Analytics log",
        "app/components/analytics.py\nartifacts/app_logs/\npredictions.jsonl", TEAL)

    arrow(ax, (1.5, 6.9), (1.5, 6.5), "message")
    arrow(ax, (2.6, 5.9), (3.1, 6.3))
    arrow(ax, (2.6, 5.3), (3.1, 4.2), "raw text")
    for y in (7.94, 6.64, 5.34, 3.95):
        arrow(ax, (6.1, 6.6), (mx, y))
    for y in (7.94, 6.64, 5.34):
        arrow(ax, (mx + mw, y), (11.4, 6.65), color="#6b7280")
    arrow(ax, (mx + mw, 3.95), (11.4, 6.0), color="#ea580c")
    arrow(ax, (12.47, 5.35), (12.47, 5.05))
    arrow(ax, (13.55, 4.6), (13.85, 7.0))
    arrow(ax, (13.55, 6.4), (13.85, 5.3), color="#dc2626")
    arrow(ax, (6.1, 3.5), (13.85, 3.3), "rule fired", color="#0f766e")
    # result path back to the UI, drawn along the bottom edge so it crosses no box
    ax.plot([14.77, 14.77, 0.27, 0.27], [2.85, 2.68, 2.68, 5.7], color="#0f766e", lw=1.2, ls="--")
    arrow(ax, (0.27, 5.7), (0.4, 5.7), color="#0f766e")
    ax.text(6.5, 2.73, "result card + side-by-side comparison of ①–④ (with latency) back to the UI",
            fontsize=7, color="#0f766e")
    arrow(ax, (1.5, 4.9), (1.5, 4.6))

    # offline training region
    ax.add_patch(FancyBboxPatch((0.15, 0.1), 15.7, 2.25, boxstyle="round,pad=0.02", fc="#f9fafb", ec="#9ca3af", ls="--"))
    ax.text(0.3, 2.12, "OFFLINE TRAINING  (scripts/*.py)  ·  ① ② on CPU  ·  ③ ④ trained on Apple M3 GPU (PyTorch MPS)",
            fontsize=8, color="#4b5563")
    steps = [("MASSIVE 1.1 tr-TR + en-US\n+ 100 manual examples", "download_dataset.py\nprepare_dataset.py"),
             ("train / val / test\n(same test for all)", "data/processed/*.csv"),
             ("TF-IDF+LR · FastText+FFNN", "train_classical.py\ntrain_neural.py"),
             ("fine-tune XLM-R (teacher)", "train_transformer.py"),
             ("knowledge distillation\nT=2, α=0.7 → student", "distill_student.py"),
             ("dynamic INT8 quantization\n+ reload check", "quantize_model.py"),
             ("accuracy · OOS recall ·\nsize · latency (CPU)", "evaluate_all.py")]
    w = 2.05
    for i, (t, b) in enumerate(steps):
        x = 0.35 + i * 2.22
        box(ax, x, 0.3, w, 1.6, t, "\n" + b, ORANGE if "INT8" in t else GREY, fs=7.3)
        if i:
            arrow(ax, (x - 0.17, 1.1), (x, 1.1))
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "architecture.png", dpi=170, bbox_inches="tight")
    fig.savefig(out / "architecture.pdf", bbox_inches="tight")
    print("saved", out / "architecture.png")


if __name__ == "__main__":
    main()
