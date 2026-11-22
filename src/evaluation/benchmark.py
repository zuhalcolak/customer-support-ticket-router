"""Hardware description, model-size measurement and latency benchmarking."""
from __future__ import annotations

import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import psutil
import torch

from src import config as C


def hardware_info() -> dict:
    cpu = platform.processor()
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    gpu = "none used for benchmarking (CPU only)"
    return {
        "cpu": cpu, "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "os": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(), "benchmark_device": "cpu", "gpu": gpu,
        "training_accelerator": "Apple M3 GPU via PyTorch MPS (Transformer + student training only)",
        "quantized_engine": torch.backends.quantized.engine,
    }


def files_size_mb(paths: list[Path]) -> float:
    return sum(p.stat().st_size for p in paths if p.exists()) / 1e6


MODEL_WEIGHT_FILES = {
    "classical": lambda: [C.ARTIFACTS / "classical" / "tfidf_logreg.joblib"],
    "neural": lambda: [p for p in (C.ARTIFACTS / "neural").glob("*") if p.suffix in {".model", ".npy", ".pt", ".npz"}],
    "transformer": lambda: list((C.ARTIFACTS / "transformer").glob("*.safetensors")),
    "student_fp32": lambda: list((C.ARTIFACTS / "student").glob("*.safetensors")),
    "quantized": lambda: [C.ARTIFACTS / "quantized" / "model_int8.pt"],
}


def model_size_mb(name: str) -> float:
    return files_size_mb(MODEL_WEIGHT_FILES[name]())


def benchmark_latency(predict_one, inputs: list, warmup: int = 10, runs: int = 200) -> dict:
    """Batch size 1. `warmup` untimed calls, then `runs` timed calls cycling over `inputs`."""
    for i in range(warmup):
        predict_one(inputs[i % len(inputs)])
    times = []
    for i in range(runs):
        x = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        predict_one(x)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"latency_mean_ms": statistics.fmean(times), "latency_median_ms": statistics.median(times),
            "latency_p95_ms": times[int(0.95 * len(times)) - 1], "latency_std_ms": statistics.pstdev(times),
            "runs": runs, "warmup": warmup, "batch_size": 1, "pid": os.getpid()}
