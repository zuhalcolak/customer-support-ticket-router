from src.preprocessing.text import (detect_language, extract_model_codes, fold_ascii, levenshtein, preprocess,
                                    turkish_lower)


def test_language_turkish():
    assert detect_language("Bulaşık makinem su boşaltmıyor.") == "tr"
    assert detect_language("Buzdolabım yeterince soğutmuyor") == "tr"


def test_language_turkish_ascii_typed():
    assert detect_language("bulasik makinem calismiyor") == "tr"


def test_language_english():
    assert detect_language("My dishwasher is not draining water.") == "en"
    assert detect_language("I want to check my warranty.") == "en"


def test_turkish_lowercase_dotted_i():
    assert turkish_lower("IŞIK İADE") == "ışık iade"


def test_fold_ascii():
    assert fold_ascii("çamaşır şğü") == "camasir sgu"


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("dishwaser", "dishwasher") == 1
    assert levenshtein("", "abc") == 3


def test_model_code_extraction():
    assert extract_model_codes("bulaşık makinem su boşaltmıyor model DW-6534") == ["DW-6534"]
    assert extract_model_codes("fridge RF 4410X leaking") == ["RF-4410X"]
    assert extract_model_codes("my dw6534 is broken") == ["DW-6534"]


def test_error_codes_are_not_model_codes():
    assert extract_model_codes("ekranda E15 hata kodu var") == []
    assert extract_model_codes("in 2024 I bought it") == []


def test_preprocess_example_from_assignment():
    p = preprocess("bulaşık makinem su boşaltmıyor model DW-6534")
    assert p.language == "tr"
    assert p.model_codes == ["DW-6534"]
    assert p.product_category == "dishwasher"
    assert "dw" not in p.normalized


def test_misspelled_product_terms_are_corrected():
    p = preprocess("my dishwaser is broken")
    assert ("dishwaser", "dishwasher") in p.corrections
    assert p.product_category == "dishwasher"
    p = preprocess("Bulasik makinem calismiyor")
    assert p.normalized.startswith("bulaşık")
    p = preprocess("refrigirator not cooling")
    assert p.product_category == "refrigerator"


def test_known_words_are_not_overcorrected():
    # "kahvaltı" (breakfast) must not become "kahve..." (coffee)
    p = preprocess("kahvaltı hazırla")
    assert "kahvaltı" in p.normalized


def test_empty_input():
    p = preprocess("   ")
    assert p.is_empty and p.normalized == "" and p.language == "unknown"
    assert preprocess("").is_empty


def test_very_long_input_is_truncated():
    p = preprocess("bulaşık makinem çalışmıyor " * 1000)
    assert len(p.original) > 2000
    assert len(p.normalized) <= 2100


def test_noise_normalization():
    p = preprocess("Fırınım ÇALIŞMIYOR!!!! www.example.com mail@test.com")
    assert "www" not in p.normalized and "@" not in p.normalized
    assert "!" not in p.normalized
    assert "çalışmıyor" in p.normalized
