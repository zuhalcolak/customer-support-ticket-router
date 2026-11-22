"""Confidence threshold + out-of-scope detection -> department routing."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from src import config as C

DEFAULT_THRESHOLD = 0.5


def load_thresholds() -> dict[str, float]:
    path = C.ARTIFACTS / "thresholds.json"
    if path.exists():
        return {k: float(v["threshold"]) for k, v in json.loads(path.read_text()).items()}
    return {}


@dataclass
class Decision:
    intent: str               # raw arg-max intent of the model
    confidence: float         # max softmax probability
    final_intent: str         # intent after OOS / threshold logic
    department: str
    routed_to_human: bool
    reason: str               # "confident", "low_confidence", "out_of_scope"
    top3: list[tuple[str, float]]

    def to_dict(self) -> dict:
        return asdict(self)


def decide(probs: np.ndarray, threshold: float) -> Decision:
    order = np.argsort(-probs)
    top = int(order[0])
    intent = C.ID2LABEL[top]
    conf = float(probs[top])
    top3 = [(C.ID2LABEL[int(i)], float(probs[int(i)])) for i in order[:3]]
    if intent == C.OOS_LABEL:
        return Decision(intent, conf, C.OOS_LABEL, C.HUMAN_AGENT, True, "out_of_scope", top3)
    if conf < threshold:
        return Decision(intent, conf, C.OOS_LABEL, C.HUMAN_AGENT, True, "low_confidence", top3)
    return Decision(intent, conf, intent, C.INTENT_TO_DEPARTMENT[intent], False, "confident", top3)


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Vectorised version of `decide` used for evaluation: returns label ids."""
    pred = probs.argmax(1)
    low = probs.max(1) < threshold
    pred = pred.copy()
    pred[low] = C.LABEL2ID[C.OOS_LABEL]
    return pred


MAX_INSCOPE_DROP = 0.02


def tune_threshold(val_probs: np.ndarray, val_y: np.ndarray) -> tuple[float, float]:
    """Choose the confidence threshold on the VALIDATION split only.

    Policy (business rule: a wrong department costs more than a human hand-off):
    take the LARGEST threshold whose in-scope validation accuracy is at most
    MAX_INSCOPE_DROP (2 points) below the in-scope accuracy without a threshold.
    The validation OOS examples come only from 'seen' OOS scenarios, so a plain
    accuracy-maximising threshold would collapse to 0 and never catch unseen topics.
    Returns (threshold, overall validation accuracy at that threshold)."""
    oos_id = int(np.where(np.array(list(C.LABEL2ID)) == C.OOS_LABEL)[0][0])
    ins = val_y != oos_id
    base = float((val_probs[ins].argmax(1) == val_y[ins]).mean())
    best_t = 0.0
    for t in np.round(np.arange(0.0, 0.96, 0.01), 2):
        acc_in = float((apply_threshold(val_probs[ins], t) == val_y[ins]).mean())
        if acc_in >= base - MAX_INSCOPE_DROP:
            best_t = float(t)
    acc = float((apply_threshold(val_probs, best_t) == val_y).mean())
    return best_t, acc
