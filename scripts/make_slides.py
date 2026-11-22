"""Build the 11-slide presentation PDF (docs/presentation.pdf) from the measured result files.
All numbers are read from reports/*.csv|json at build time - nothing is typed in by hand.

Usage: python scripts/make_slides.py --name "Your Name" --student-id 123 --repo URL --video URL
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

INK, MUTED, ACCENT, LIGHT = "#111827", "#4b5563", "#ea580c", "#f3f4f6"
W, H = 13.333, 7.5
SECTION_COL = {"Introduction": "#2563eb", "Development": "#7c3aed", "Conclusion": "#059669"}


PNG_DIR = None  # set by --png-dir: also export every slide as PNG (for previewing)
_count = [0]


def save(pdf, fig):
    pdf.savefig(fig)
    _count[0] += 1
    if PNG_DIR:
        fig.savefig(Path(PNG_DIR) / f"slide_{_count[0]:02d}.png", dpi=80)


def new_slide(pdf_pages, title, section=None, num=None):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, H - 0.08), W, 0.08, color=ACCENT))
    if section:
        ax.text(0.6, H - 0.45, section.upper(), fontsize=10, color=SECTION_COL[section], weight="bold")
    ax.text(0.6, H - 0.95, title, fontsize=24, color=INK, weight="bold")
    if num:
        ax.text(W - 0.5, 0.3, str(num), fontsize=9, color=MUTED, ha="right")
        ax.text(0.6, 0.3, "Customer Support Ticket Router · SWE015", fontsize=9, color=MUTED)
    return fig, ax


def bullets(ax, x, y, items, fs=13, width=60, gap=0.42, color=INK):
    for it in items:
        sub = it.startswith("  ")
        txt = it.strip()
        lines = textwrap.wrap(txt, width - (4 if sub else 0))
        ax.text(x + (0.35 if sub else 0), y, "–" if sub else "•", fontsize=fs, color=ACCENT if not sub else MUTED)
        for j, ln in enumerate(lines):
            ax.text(x + (0.35 if sub else 0) + 0.3, y - j * gap * 0.82, ln, fontsize=fs - (1.5 if sub else 0),
                    color=color if not sub else MUTED, va="baseline")
        y -= gap * (0.82 * (len(lines) - 1) + 1) + (0.04 if not sub else 0)
    return y


def table(ax, x, y, df, col_w, fs=10.5, row_h=0.38, header_color="#1f2937", highlight=None):
    cols = list(df.columns)
    cx = x
    for c, w in zip(cols, col_w):
        ax.add_patch(Rectangle((cx, y - row_h), w, row_h, color=header_color))
        ax.text(cx + 0.08, y - row_h / 2, str(c), fontsize=fs, color="white", va="center", weight="bold")
        cx += w
    for i, (_, r) in enumerate(df.iterrows()):
        yy = y - row_h * (i + 2)
        bg = "#fff7ed" if highlight is not None and highlight(r) else (LIGHT if i % 2 == 0 else "white")
        ax.add_patch(Rectangle((x, yy), sum(col_w), row_h, color=bg))
        cx = x
        for c, w in zip(cols, col_w):
            ax.text(cx + 0.08, yy + row_h / 2, str(r[c]), fontsize=fs, color=INK, va="center",
                    weight="bold" if highlight is not None and highlight(r) else "normal")
            cx += w
    return y - row_h * (len(df) + 1)


def card(ax, x, y, w, h, title, body, color, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=color, ec="#d1d5db"))
    ax.text(x + 0.18, y + h - 0.35, title, fontsize=fs + 2, weight="bold", color=INK)
    yy = y + h - 0.75
    for ln in body:
        for k, part in enumerate(textwrap.wrap(ln, int(w * 9.2))):
            ax.text(x + 0.18, yy, part, fontsize=fs, color=INK)
            yy -= 0.3
        yy -= 0.04


def image(fig, path, rect):
    a = fig.add_axes(rect)
    a.imshow(mpimg.imread(path))
    a.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Zuhal Çolak")
    ap.add_argument("--student-id", default="220901606")
    ap.add_argument("--repo", default="https://github.com/zuhalcolak/customer-support-ticket-router")
    ap.add_argument("--video", default="[INSERT DEMO VIDEO URL]")
    ap.add_argument("--png-dir", default=None)
    args = ap.parse_args()
    global PNG_DIR
    PNG_DIR = args.png_dir
    if PNG_DIR:
        Path(PNG_DIR).mkdir(parents=True, exist_ok=True)

    R = pd.read_csv(C.REPORTS / "results.csv")
    Q = pd.read_csv(C.REPORTS / "quantization_comparison.csv")
    J = json.loads((C.REPORTS / "results.json").read_text())
    hw = J["hardware"]
    ds = json.loads((C.REPORTS / "dataset_stats.json").read_text())
    logs = {n: json.loads((C.REPORTS / "training_logs" / f"{f}.json").read_text()) for n, f in
            [("classical", "classical"), ("neural", "neural"), ("transformer", "transformer"),
             ("student", "student_distillation"), ("quant", "quantization")]}
    fails = pd.read_csv(C.REPORTS / "failure_examples.csv", keep_default_na=False)
    r = {row.model: row for row in R.itertuples()}
    out = ROOT / "docs" / "presentation.pdf"
    n = 0

    with PdfPages(out) as pdf:
        # 1 ---------------------------------------------------------------- title
        n += 1
        fig, ax = new_slide(pdf, "", None)
        ax.add_patch(Rectangle((0, 0), 0.25, H, color=ACCENT))
        ax.text(0.9, 5.4, "Customer Support Ticket Router", fontsize=36, weight="bold", color=INK)
        ax.text(0.9, 4.8, "Routing Turkish & English support messages with a four-model ladder:\n"
                "TF-IDF + LogReg → FastText + FFNN → XLM-R → distilled INT8 Transformer",
                fontsize=15, color=MUTED, va="top", linespacing=1.5)
        info = [("Student", args.name), ("Student ID", args.student_id),
                ("Course", "SWE015 – Introduction to Large Language Models"),
                ("Use case", "Use Case 2 – Customer Support Ticket Router"),
                ("Repository", args.repo), ("Demo video", args.video)]
        for i, (k, v) in enumerate(info):
            ax.text(0.9, 3.2 - i * 0.42, k, fontsize=13, color=MUTED)
            url = v if v.startswith("http") else None  # clickable link in the PDF
            ax.text(3.0, 3.2 - i * 0.42, v, fontsize=13, color=INK, weight="bold", url=url)
        save(pdf, fig)
        plt.close(fig)

        # 2 ---------------------------------------------------------------- introduction
        n += 1
        fig, ax = new_slide(pdf, "Problem, users and what success looks like", "Introduction", n)
        bullets(ax, 0.6, 5.9, [
            "A home-appliance brand receives thousands of support messages a day in Turkish and English "
            "(e.g. \"bulaşık makinem su boşaltmıyor, model DW-6534\").",
            "Manual triage is slow; a message sent to the wrong team is re-routed and the customer waits.",
            "Users: the support-centre triage desk and the support agents who receive the routed tickets.",
            "Selected use case: Use Case 2 – Customer Support Ticket Router.",
        ], fs=13, width=58)
        card(ax, 7.4, 2.0, 5.3, 4.2, "Objectives / success criteria", [
            "Free text (tr / en) → normalized → intent + confidence → department.",
            "Product model code extracted (DW-6534).",
            "Low-confidence and out-of-scope messages → Human Agent.",
            "Greetings / small talk → rule-based auto-reply.",
            "4 models compared on the SAME test set; metrics: accuracy and out-of-scope recall, plus size and latency.",
            "The app runs the quantized Transformer on CPU.",
        ], "#fff7ed", fs=11)
        save(pdf, fig)
        plt.close(fig)

        # 3 ---------------------------------------------------------------- data
        n += 1
        fig, ax = new_slide(pdf, "Data: sources, splits, out-of-scope strategy, preprocessing", "Development", n)
        s = ds["split_sizes"]
        bullets(ax, 0.6, 5.95, [
            "MASSIVE 1.1 (Amazon, CC BY 4.0): en-US + tr-TR, 16,521 utterances per locale, 60 intents, "
            "parallel ids → official splits keep translations in the same split.",
            "  In scope (original labels): 9 iot_* (smart home), takeaway_order / _query, general_greet / _joke.",
            "100 hand-written appliance examples (5 intents × 10 tr + 10 en), labelled as authored, not real tickets; "
            "train-only augmentation (+273): product swap, ASCII typing, typos, model codes.",
            "Out-of-scope: 7 MASSIVE scenarios as an explicit out_of_scope class (seen); 8 other scenarios + "
            "general_quirky only in TEST (unseen topics).",
            f"Splits: train {s['train']:,} · val {s['val']:,} · test {s['test']:,} (606 OOS: 306 seen + 300 unseen). "
            f"{ds['removed_for_leakage']} leaking rows + {ds['removed_duplicates']} duplicates removed.",
            "Preprocessing (src/preprocessing/text.py): language ID, Turkish lower-casing (I→ı), regex removal of "
            "URL/e-mail/phone/model code, punctuation, repeated letters, edit-distance correction of product terms, "
            "ASCII folding for TF-IDF/FastText.",
        ], fs=12.5, width=105, gap=0.4)
        save(pdf, fig)
        plt.close(fig)

        # 4 ---------------------------------------------------------------- architecture
        n += 1
        fig, ax = new_slide(pdf, "Architecture (matches the code; every box names its file)", "Development", n)
        image(fig, ROOT / "docs" / "architecture.png", [0.03, 0.06, 0.94, 0.78])
        save(pdf, fig)
        plt.close(fig)

        # 5 ---------------------------------------------------------------- technique map
        n += 1
        fig, ax = new_slide(pdf, "Technique map", "Development", n)
        tm = pd.DataFrame([
            ["A · Neural nets", "FastText embeddings (trained on our train set), averaged", "② src/neural/model.py"],
            ["", "Feed-forward net 150→256→128→19, ReLU, dropout", "② src/neural/model.py"],
            ["", "SGD (momentum 0.9) + backpropagation, early stopping", "② scripts/train_neural.py"],
            ["", "Data augmentation of manual training examples", "scripts/prepare_dataset.py"],
            ["B · Classical NLP", "Regex normalization, Turkish casing, tokenization", "src/preprocessing/text.py"],
            ["", "Regex product model-code extraction (DW-6534)", "src/preprocessing/text.py"],
            ["", "Minimum edit distance spelling correction", "src/preprocessing/text.py"],
            ["", "ELIZA-style regex rules for small talk", "src/routing/smalltalk.py"],
            ["", "TF-IDF (word + char n-grams) + Logistic Regression", "① src/classical/model.py"],
            ["C · Transformers", "XLM-RoBERTa base fine-tuned with classification head", "③ src/transformer/model.py"],
            ["", "Self-attention encoders; SentencePiece / WordPiece subwords", "③ ④ HF tokenizers"],
            ["D · Efficient", "Knowledge distillation (T=2, α=0.7) into mDistilBERT", "④ scripts/distill_student.py"],
            ["", "Dynamic INT8 quantization (Linear qint8, Embedding quint8)", "④ src/quantization/model.py"],
            ["", "Size / latency (≥100 runs) / accuracy benchmarking", "scripts/evaluate_all.py"],
        ], columns=["Area", "Technique", "Where in the application"])
        table(ax, 0.6, 6.15, tm, [2.3, 6.2, 3.6], fs=10.5, row_h=0.37)
        save(pdf, fig)
        plt.close(fig)

        # 6 ---------------------------------------------------------------- models & training
        n += 1
        fig, ax = new_slide(pdf, "Four models and how each was trained", "Development", n)
        lt, ls_, ln = logs["transformer"], logs["student"], logs["neural"]
        md = pd.DataFrame([
            ["① TF-IDF + LogReg", "word 1-2g + char 2-5g TF-IDF, LR (C=10, balanced)", "scikit-learn, CPU",
             f"{logs['classical']['train_seconds']:.1f} s"],
            ["② FastText + FFNN", f"FastText 150d (40 ep.) → FFNN; SGD lr 0.02, mom. 0.9, batch 32, "
             f"{ln['epochs_run']} epochs (early stop)", "PyTorch, CPU", f"{ln['fasttext_seconds'] + ln['train_seconds']:.1f} s"],
            ["③ XLM-R base", f"AdamW lr 3e-5, batch 32, max_len 64, {len(lt['history'])} epochs, linear warm-up 10%",
             "Apple M3 GPU (MPS)", f"{lt['train_seconds'] / 60:.1f} min"],
            ["④ mDistilBERT student", f"KD from ③: α=0.7, T=2; AdamW lr 5e-5, {len(ls_['history'])} epochs → INT8",
             "Apple M3 GPU (MPS)", f"{ls_['train_seconds'] / 60:.1f} min"],
        ], columns=["Model", "Training procedure", "Hardware", "Train time"])
        table(ax, 0.6, 6.1, md, [2.5, 6.6, 1.9, 1.3], fs=10, row_h=0.5)
        bullets(ax, 0.6, 3.25, [
            "All four models predict the same 19 labels and share the same preprocessing and routing code.",
            "Model selection / early stopping and the confidence threshold use the validation split only.",
            "Threshold rule: the largest τ that costs ≤ 2 points of in-scope validation accuracy "
            "(misrouting is worse than a hand-off).",
            "Why XLM-R: multilingual (100 languages incl. Turkish), strong on Turkish morphology; "
            "student uses mDistilBERT (6 layers, 135M) because it is a pretrained small multilingual encoder.",
        ], fs=12, width=110, gap=0.4)
        save(pdf, fig)
        plt.close(fig)

        # 7 ---------------------------------------------------------------- quantization
        n += 1
        fig, ax = new_slide(pdf, "Distillation + quantization method and its effect", "Development", n)
        qd = Q.copy()
        qd.columns = ["Model", "Accuracy", "Macro-F1", "OOS recall", "Size (MB)", "Mean latency (ms)", "Median (ms)"]
        for c in ["Accuracy", "Macro-F1", "OOS recall"]:
            qd[c] = qd[c].map("{:.3f}".format)
        for c in ["Size (MB)", "Mean latency (ms)", "Median (ms)"]:
            qd[c] = qd[c].map("{:.1f}".format)
        table(ax, 0.6, 6.1, qd, [3.4, 1.3, 1.3, 1.3, 1.3, 1.9, 1.4], fs=11, row_h=0.45,
              highlight=lambda row: "INT8" in row["Model"])
        ql = logs["quant"]
        bullets(ax, 0.6, 3.75, [
            "Distillation: student trained on α·T²·KL(teacher_T ‖ student_T) + (1-α)·CE(gold), teacher logits on train only.",
            "Quantization: torch.ao.quantization.quantize_dynamic – all nn.Linear → int8 weights with dynamic "
            "activation quantization; nn.Embedding → int8 weight-only. Engine: " + ql["engine"] + ".",
            f"Verified: model reloaded from disk contains DynamicQuantizedLinear / QuantizedEmbedding modules; "
            f"INT8 vs FP32 student agree on {ql['argmax_agreement_with_fp32_student_on_test']:.1%} of test messages.",
            "Size = weight file on disk; latency = CPU, batch 1, mean of "
            f"{J['benchmark']['timed_runs']} runs after {J['benchmark']['warmup_runs']} warm-up runs.",
        ], fs=12, width=115, gap=0.4)
        save(pdf, fig)
        plt.close(fig)

        # 8 ---------------------------------------------------------------- results table
        n += 1
        fig, ax = new_slide(pdf, "Results – same held-out test set (n = 1,206)", "Development", n)
        rt = R[["display_name", "accuracy", "macro_f1", "oos_recall", "size_mb", "latency_mean_ms"]].copy()
        rt.columns = ["Model", "Accuracy", "Macro-F1", "OOS recall", "Size (MB)", "Latency (ms)"]
        for c in ["Accuracy", "Macro-F1", "OOS recall"]:
            rt[c] = rt[c].map("{:.3f}".format)
        rt["Size (MB)"] = rt["Size (MB)"].map("{:.1f}".format)
        rt["Latency (ms)"] = rt["Latency (ms)"].map("{:.2f}".format)
        table(ax, 0.6, 6.15, rt, [3.6, 1.4, 1.4, 1.4, 1.4, 1.5], fs=11.5, row_h=0.45,
              highlight=lambda row: "INT8" in row["Model"])
        ax.text(0.6, 3.6, textwrap.fill(
            f"Hardware: {hw['cpu']}, {hw['ram_gb']} GB RAM, {hw['os']}, Python {hw['python']}, torch {hw['torch']}. "
            f"Benchmarks on CPU only ({hw['benchmark_threads']} threads), batch size 1, {J['benchmark']['warmup_runs']} "
            f"warm-up + {J['benchmark']['timed_runs']} timed runs on random test messages. Accuracy counts out_of_scope "
            "as a class and applies each model's validation-tuned threshold.", 150), fontsize=9.5, color=MUTED, va="top")
        image(fig, C.FIGURES / "accuracy.png", [0.03, 0.06, 0.31, 0.37])
        image(fig, C.FIGURES / "oos_recall.png", [0.345, 0.06, 0.31, 0.37])
        image(fig, C.FIGURES / "latency.png", [0.66, 0.06, 0.31, 0.37])
        save(pdf, fig)
        plt.close(fig)

        # 9 ---------------------------------------------------------------- findings
        n += 1
        fig, ax = new_slide(pdf, "What the comparison showed", "Conclusion", n)
        best = R.sort_values("accuracy", ascending=False).iloc[0]
        man = R.set_index("model")["accuracy[source=manual]"].to_dict()
        uns = R.set_index("model")["oos_recall[unseen]"]
        tr_acc = R.set_index("model")["accuracy[lang=tr]"]
        mf1 = R.set_index("model")["macro_f1"]
        rc, rn, rt_, rq = r["classical"], r["neural"], r["transformer"], r["quantized"]
        wl = pd.DataFrame([
            ["① TF-IDF + LogReg",
             f"best accuracy {rc.accuracy:.3f}, best OOS recall {rc.oos_recall:.3f} (unseen {uns['classical']:.3f}); 4.8 MB, {rc.latency_mean_ms:.2f} ms",
             f"appliance intents {man['classical']:.2f}: needs exact word overlap"],
            ["② FastText + FFNN",
             f"fastest ({rn.latency_mean_ms:.2f} ms); typo-robust subword vectors",
             f"weakest: accuracy {rn.accuracy:.3f}; averaging loses word order"],
            ["③ XLM-R (FP32)",
             f"best macro-F1 {mf1['transformer']:.3f}; appliance intents {man['transformer']:.2f}",
             f"accuracy {rt_.accuracy:.3f} (< baseline); {rt_.size_mb:.0f} MB, {rt_.latency_mean_ms:.1f} ms"],
            ["④ INT8 student",
             f"{rq.size_mb / rt_.size_mb:.0%} of ③'s size, {rt_.latency_mean_ms / rq.latency_mean_ms:.1f}× faster, "
             f"accuracy {rq.accuracy - rt_.accuracy:+.3f}, OOS recall {rq.oos_recall:.3f}",
             f"macro-F1 {mf1['quantized']:.3f}; appliance intents {man['quantized']:.2f}"],
        ], columns=["Model", "Wins", "Loses"])
        table(ax, 0.6, 6.15, wl, [2.2, 6.0, 4.0], fs=10, row_h=0.5)
        bullets(ax, 0.6, 3.35, [
            "MASSIVE commands are short and formulaic, so word/char n-grams are already very strong: "
            "the Transformer does NOT beat the baseline on overall accuracy.",
            "Transformers win where lexical overlap is low: the hand-written appliance requests "
            "(only 20 test sentences, so high variance) and rare classes (macro-F1).",
            "Distillation + INT8 keeps overall accuracy and OOS recall but loses part of the teacher's gain on "
            "the small appliance classes, which have very little training data for the student.",
        ], fs=12, width=118, gap=0.4)
        save(pdf, fig)
        plt.close(fig)

        # 10 --------------------------------------------------------------- limitations + failures
        n += 1
        fig, ax = new_slide(pdf, "Limitations, real failure examples, next steps", "Conclusion", n)
        fx = fails.copy()
        pick = []
        for cond in [fx.source == "manual", (fx.label == C.OOS_LABEL) & (fx.oos_type == "unseen"), fx.label != C.OOS_LABEL]:
            sub = fx[cond & ~fx.uid.isin([p.uid for p in pick])]
            if len(sub):
                pick.append(sub.iloc[0])
        fdf = pd.DataFrame([{"Message (test set, or live app)": textwrap.shorten(p.text, 46),
                             "True label": p.label, "INT8 prediction": p.pred_quantized,
                             "conf.": f"{p.conf_quantized:.2f}"} for p in pick])
        ax.text(0.6, 6.15, "Failure examples of the deployed INT8 model (reports/failure_examples.csv)",
                fontsize=12, weight="bold", color=INK)
        # one live input, run through the real INT8 model while building the slides
        import warnings
        warnings.filterwarnings("ignore")
        from src.pipeline import TicketRouter
        live_msg = "Bulaşık makinem su boşaltmıyor."
        lr = TicketRouter(["quantized", "transformer"]).route(live_msg)
        lq, lt_ = lr.models["quantized"], lr.models["transformer"]
        fdf.loc[len(fdf)] = [f"LIVE APP: {live_msg}", "appliance_malfunction",
                             f"{lq.intent} (XLM-R: {lt_.intent})", f"{lq.confidence:.2f}"]
        table(ax, 0.6, 6.0, fdf, [4.5, 2.3, 4.7, 0.8], fs=10, row_h=0.42)
        bullets(ax, 0.6, 3.6, [
            "Domain gap: MASSIVE is virtual-assistant data; appliance intents rest on 100 authored sentences "
            "(4 test sentences per intent → high variance). takeaway_* is only a proxy for 'Orders & Delivery'.",
            "OOS threshold is tuned on seen OOS only; unseen topics with appliance-like words can still be misrouted.",
            "Latency measured on one Apple M3 laptop, CPU only; numbers will differ on a server CPU.",
            "Next: collect real anonymised tickets, add more appliance intents, ONNX Runtime export, "
            "vocabulary pruning of the student embeddings, calibration (temperature scaling) before thresholding.",
        ], fs=12, width=115, gap=0.4)
        save(pdf, fig)
        plt.close(fig)

        # 11 --------------------------------------------------------------- AI usage
        n += 1
        fig, ax = new_slide(pdf, "AI usage statement", "Conclusion", n)
        bullets(ax, 0.6, 5.9, [
            "Tool: Claude (Anthropic) through Claude Code, an AI coding assistant in my project folder.",
            "Used for: code generation (src/, scripts/, app/, tests), debugging (e.g. slow MPS training from dynamic "
            "padding, a spelling-correction bug), drafting the 100 authored example sentences (reviewed by me), "
            "documentation (README, dataset report) and drafting these slides.",
            "Not used for results: every number here was produced by running the repository's scripts on my "
            "machine; raw outputs are in reports/ (results.csv, results.json, training_logs/).",
            "No metric was copied from an AI answer, a paper or a model card.",
            "I reviewed the code and can explain every component of the architecture diagram.",
        ], fs=13, width=100, gap=0.45)
        save(pdf, fig)
        plt.close(fig)
    print(f"saved {out} ({n} slides)")


if __name__ == "__main__":
    main()
