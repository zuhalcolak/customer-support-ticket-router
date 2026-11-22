"""Write the measured results table into README.md between the RESULTS markers.

Usage: python scripts/update_readme_results.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402


def main() -> None:
    R = pd.read_csv(C.REPORTS / "results.csv")
    Q = pd.read_csv(C.REPORTS / "quantization_comparison.csv")
    hw = json.loads((C.REPORTS / "hardware.json").read_text())
    J = json.loads((C.REPORTS / "results.json").read_text())
    lines = ["| Model | Accuracy | Macro-F1 | OOS recall | Model size (MB) | Latency (ms, mean) | Latency (ms, p95) |",
             "|---|---|---|---|---|---|---|"]
    for r in R.itertuples():
        lines.append(f"| {r.display_name} | {r.accuracy:.3f} | {r.macro_f1:.3f} | {r.oos_recall:.3f} | "
                     f"{r.size_mb:.1f} | {r.latency_mean_ms:.2f} | {r.latency_p95_ms:.2f} |")
    lines += ["", "Breakdown (same test set):", "",
              "| Model | Acc. en | Acc. tr | In-scope acc. | OOS recall (seen) | OOS recall (unseen) | Acc. manual appliance (n=20) | Threshold τ |",
              "|---|---|---|---|---|---|---|---|"]
    for r in R.to_dict(orient="records"):
        lines.append(f"| {r['display_name']} | {r['accuracy[lang=en]']:.3f} | {r['accuracy[lang=tr]']:.3f} | "
                     f"{r['inscope_accuracy']:.3f} | {r['oos_recall[seen]']:.3f} | {r['oos_recall[unseen]']:.3f} | "
                     f"{r['accuracy[source=manual]']:.2f} | {r['threshold']:.2f} |")
    lines += ["", "Quantization comparison:", "",
              "| Model | Accuracy | Macro-F1 | OOS recall | Size (MB) | Latency mean (ms) | Latency median (ms) |",
              "|---|---|---|---|---|---|---|"]
    for r in Q.itertuples():
        lines.append(f"| {r.display_name} | {r.accuracy:.3f} | {r.macro_f1:.3f} | {r.oos_recall:.3f} | "
                     f"{r.size_mb:.1f} | {r.latency_mean_ms:.2f} | {r.latency_median_ms:.2f} |")
    lines += ["", f"Hardware: {hw['cpu']} ({hw['physical_cores']} cores), {hw['ram_gb']} GB RAM, {hw['os']}, "
              f"Python {hw['python']}, PyTorch {hw['torch']}. Benchmark on CPU only, {hw['benchmark_threads']} threads, "
              f"batch size 1, {J['benchmark']['warmup_runs']} warm-up runs, {J['benchmark']['timed_runs']} timed runs "
              "on randomly sampled test messages (seed 42). Latency = model call on a preprocessed message "
              f"(tokenization/vectorization + forward + softmax). Shared preprocessing adds "
              f"{J['preprocessing_latency']['latency_mean_ms']:.2f} ms on average. Size = weight files on disk. "
              "Transformer and student were trained on the M3 GPU (MPS); GPU not used for benchmarking."]
    block = "\n".join(lines)
    readme = (ROOT / "README.md").read_text()
    readme = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                    f"<!-- RESULTS:START -->\n{block}\n<!-- RESULTS:END -->", readme, flags=re.S)
    (ROOT / "README.md").write_text(readme)
    print(block)


if __name__ == "__main__":
    main()
