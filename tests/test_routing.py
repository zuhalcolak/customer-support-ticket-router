import numpy as np

from src import config as C
from src.routing.router import apply_threshold, decide, tune_threshold
from src.routing.smalltalk import small_talk_reply


def probs_for(label: str, p: float) -> np.ndarray:
    v = np.full(len(C.LABELS), (1 - p) / (len(C.LABELS) - 1))
    v[C.LABEL2ID[label]] = p
    return v


def test_confident_prediction_routes_to_department():
    d = decide(probs_for("appliance_malfunction", 0.9), threshold=0.5)
    assert d.department == "Technical Support" and not d.routed_to_human and d.reason == "confident"


def test_low_confidence_goes_to_human():
    d = decide(probs_for("warranty_inquiry", 0.3), threshold=0.5)
    assert d.routed_to_human and d.department == C.HUMAN_AGENT and d.reason == "low_confidence"
    assert d.intent == "warranty_inquiry" and d.final_intent == C.OOS_LABEL


def test_out_of_scope_goes_to_human():
    d = decide(probs_for(C.OOS_LABEL, 0.95), threshold=0.5)
    assert d.routed_to_human and d.department == C.HUMAN_AGENT and d.reason == "out_of_scope"


def test_apply_threshold_vectorised():
    P = np.stack([probs_for("iot_coffee", 0.9), probs_for("iot_coffee", 0.2)])
    out = apply_threshold(P, 0.5)
    assert out[0] == C.LABEL2ID["iot_coffee"] and out[1] == C.LABEL2ID[C.OOS_LABEL]


def test_tune_threshold_uses_only_given_split():
    P = np.stack([probs_for("iot_coffee", 0.9)] * 10 + [probs_for(C.OOS_LABEL, 0.9)] * 5)
    y = np.array([C.LABEL2ID["iot_coffee"]] * 10 + [C.LABEL2ID[C.OOS_LABEL]] * 5)
    t, acc = tune_threshold(P, y)
    assert 0.0 <= t <= 0.95 and acc == 1.0


def test_every_label_has_a_department():
    assert set(C.LABELS) == set(C.INTENT_TO_DEPARTMENT)


def test_greetings_get_rule_based_reply():
    assert small_talk_reply("Merhaba") is not None
    assert small_talk_reply("selam") is not None
    assert small_talk_reply("Hello!") is not None
    assert small_talk_reply("good morning") is not None


def test_small_talk_reply():
    assert "welcome" in small_talk_reply("thanks a lot").lower()
    assert small_talk_reply("Teşekkürler") is not None
    assert small_talk_reply("how are you?") is not None
    assert "tired" in small_talk_reply("I feel tired")


def test_support_request_with_greeting_is_not_small_talk():
    assert small_talk_reply("Merhaba, bulaşık makinem çalışmıyor") is None
    assert small_talk_reply("Hi, my oven is broken") is None


def test_empty_is_not_small_talk():
    assert small_talk_reply("") is None
