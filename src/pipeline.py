"""End-to-end ticket routing pipeline used by the Streamlit app, the tests and the benchmarks.

message -> preprocess (language, regex normalization, model code, spelling correction)
        -> ELIZA small-talk rules (short-circuit)
        -> model(s) -> softmax probabilities -> threshold / OOS -> department
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from src import config as C
from src.preprocessing.text import Preprocessed, preprocess
from src.routing.router import DEFAULT_THRESHOLD, decide, load_thresholds
from src.routing.smalltalk import small_talk_reply

PRIMARY_MODEL = "quantized"      # the model that makes the routing decision in the app
MODEL_ORDER = ["classical", "neural", "transformer", "quantized"]


def load_model(name: str):
    if name == "classical":
        from src.classical.model import ClassicalModel
        return ClassicalModel.load()
    if name == "neural":
        from src.neural.model import NeuralModel
        return NeuralModel.load()
    if name == "transformer":
        from src.transformer.model import HFClassifier
        return HFClassifier.load(device="cpu")
    if name == "quantized":
        from src.quantization.model import load_quantized
        return load_quantized()
    raise ValueError(name)


@dataclass
class ModelResult:
    model: str
    display_name: str
    intent: str
    final_intent: str
    confidence: float
    department: str
    routed_to_human: bool
    reason: str
    latency_ms: float
    top3: list


@dataclass
class RoutingResult:
    original: str
    language: str
    normalized: str
    model_codes: list[str]
    product_category: str | None
    corrections: list
    small_talk_reply: str | None
    final_department: str
    final_intent: str
    confidence: float | None
    routing_reason: str
    models: dict[str, ModelResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class TicketRouter:
    def __init__(self, models: list[str] | None = None, num_threads: int = 4):
        torch.set_num_threads(num_threads)
        self.models = {n: load_model(n) for n in (models or MODEL_ORDER)}
        self.thresholds = load_thresholds()

    def threshold(self, name: str) -> float:
        return self.thresholds.get(name, DEFAULT_THRESHOLD)

    def run_model(self, name: str, pre: Preprocessed) -> ModelResult:
        model = self.models[name]
        t0 = time.perf_counter()
        probs = model.predict_proba([pre])[0]
        latency = (time.perf_counter() - t0) * 1000
        d = decide(np.asarray(probs), self.threshold(name))
        return ModelResult(name, C.MODEL_NAMES[name], d.intent, d.final_intent, d.confidence, d.department,
                           d.routed_to_human, d.reason, latency, d.top3)

    def route(self, message: str) -> RoutingResult:
        pre = preprocess(message)
        if pre.is_empty:
            return RoutingResult(message, "unknown", "", [], None, [], None, "—", "empty_input", None,
                                 "empty_input")
        reply = small_talk_reply(message)
        results = {n: self.run_model(n, pre) for n in self.models}
        primary = results.get(PRIMARY_MODEL) or next(iter(results.values()))
        if reply is not None:
            dept, intent, reason = C.SMALL_TALK, "small_talk", "rule_based_small_talk"
        else:
            dept, intent, reason = primary.department, primary.final_intent, primary.reason
            if primary.final_intent in ("general_greet", "general_joke"):
                reply = ("Merhaba! Size nasıl yardımcı olabilirim?" if pre.language == "tr"
                         else "Hello! How can I help you with your appliance today?")
        return RoutingResult(message, pre.language, pre.normalized, pre.model_codes, pre.product_category,
                             pre.corrections, reply, dept, intent, primary.confidence, reason, results)
