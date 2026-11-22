"""Persistent log of messages processed by the app (used by the analytics tab).

Every classified message is appended to artifacts/app_logs/predictions.jsonl, so the
analytics shown in the app are computed from real recorded app usage only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src import config as C

LOG_PATH = C.ARTIFACTS / "app_logs" / "predictions.jsonl"


def log_result(result: dict, path: Path = LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "language": result["language"],
        "final_intent": result["final_intent"],
        "final_department": result["final_department"],
        "routing_reason": result["routing_reason"],
        "confidence": result["confidence"],
        "has_model_code": bool(result["model_codes"]),
    }
    for name, m in result["models"].items():
        row[f"{name}_intent"] = m["final_intent"]
        row[f"{name}_latency_ms"] = m["latency_ms"]
        row[f"{name}_confidence"] = m["confidence"]
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_log(path: Path = LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def clear_log(path: Path = LOG_PATH) -> None:
    if path.exists():
        path.unlink()
