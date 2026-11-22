"""Model-loading and end-to-end tests. They need the trained artifacts (see README);
they are skipped automatically when the artifacts are missing."""
import numpy as np
import pytest

from src import config as C

needs_models = pytest.mark.skipif(
    not (C.ARTIFACTS / "quantized" / "model_int8.pt").exists(), reason="trained artifacts not available")


@pytest.fixture(scope="module")
def router():
    from src.pipeline import MODEL_ORDER, TicketRouter
    return TicketRouter(MODEL_ORDER)


@needs_models
def test_all_models_load_and_return_distributions(router):
    from src.preprocessing.text import preprocess
    pre = [preprocess("My washing machine is not spinning")]
    for name, model in router.models.items():
        p = model.predict_proba(pre)
        assert p.shape == (1, len(C.LABELS)), name
        assert np.isclose(p.sum(), 1.0, atol=1e-3), name


@needs_models
def test_quantized_model_is_really_int8(router):
    from src.quantization.model import count_quantized_modules, is_fully_quantized
    q = router.models["quantized"].model
    c = count_quantized_modules(q)
    assert c["int8_dynamic_linear"] > 0 and c["int8_embedding"] > 0
    assert is_fully_quantized(q)
    assert "DynamicQuantizedLinear" in repr(q)


@needs_models
def test_quantized_model_inference(router):
    r = router.route("Bulaşık makinem su boşaltmıyor.")
    assert "quantized" in r.models
    assert r.models["quantized"].latency_ms > 0
    assert r.final_department == r.models["quantized"].department


@needs_models
def test_turkish_message(router):
    r = router.route("bulaşık makinem su boşaltmıyor model DW-6534")
    assert r.language == "tr" and r.model_codes == ["DW-6534"]
    assert r.final_department in set(C.INTENT_TO_DEPARTMENT.values())


@needs_models
def test_english_message(router):
    r = router.route("My washing machine is not spinning.")
    assert r.language == "en"
    assert set(r.models) == {"classical", "neural", "transformer", "quantized"}


@needs_models
def test_misspelled_message(router):
    r = router.route("my dishwaser wont drain")
    assert ("dishwaser", "dishwasher") in [tuple(c) for c in r.corrections]


@needs_models
def test_empty_input(router):
    r = router.route("   ")
    assert r.routing_reason == "empty_input" and r.models == {}


@needs_models
def test_very_long_message(router):
    r = router.route("My oven is not heating. " * 500)
    assert r.final_department


@needs_models
def test_greeting_routes_to_small_talk(router):
    r = router.route("Merhaba")
    assert r.small_talk_reply and r.final_department == C.SMALL_TALK


@needs_models
def test_out_of_scope_goes_to_human(router):
    # weather is an UNSEEN out-of-scope topic (never in training data)
    r = router.route("What will the weather be like in Istanbul tomorrow?")
    assert r.final_department == C.HUMAN_AGENT
    assert r.routing_reason in ("out_of_scope", "low_confidence")


@needs_models
def test_low_confidence_routes_to_human(router):
    from src.pipeline import PRIMARY_MODEL
    router.thresholds[PRIMARY_MODEL] = 0.999  # force the low-confidence path
    try:
        r = router.route("My washing machine is not spinning.")
        assert r.final_department == C.HUMAN_AGENT
        assert r.routing_reason in ("low_confidence", "out_of_scope")
    finally:
        from src.routing.router import load_thresholds
        router.thresholds = load_thresholds()
