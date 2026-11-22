"""Classical NLP preprocessing shared by all four models.

Components (all implemented here, no external NLP library):
  * language identification (stop-word / character heuristics)
  * Turkish-aware lower-casing and regex normalization
  * product model-code extraction (regex)
  * product-term detection and edit-distance spelling correction
  * tokenization
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Language identification
# ---------------------------------------------------------------------------
TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
TR_STOP = {
    "ve", "bir", "bu", "da", "de", "mi", "mı", "mu", "mü", "ne", "ile", "için", "ama",
    "çok", "var", "yok", "ben", "sen", "benim", "bana", "beni", "nasıl", "neden",
    "değil", "olarak", "gibi", "daha", "kadar", "istiyorum", "yapmıyor", "çalışmıyor",
    "lütfen", "merhaba", "selam", "teşekkürler", "musunuz", "misiniz", "edebilir",
    "makinem", "makinesi", "makine", "zaman", "hiç", "şu", "şey", "olan", "yapın",
    "yardım", "edin", "sorun", "sorunum", "bozuk", "bozuldu", "arıza", "yok", "neden", "kaç",
}
EN_STOP = {
    "the", "a", "an", "is", "are", "was", "my", "i", "you", "to", "of", "and", "in",
    "it", "for", "on", "not", "can", "what", "how", "do", "does", "please", "with",
    "this", "that", "me", "want", "need", "will", "be", "have", "has", "hello", "hi",
    "thanks", "when", "where", "why", "at", "from", "your", "there", "working",
}


def detect_language(text: str) -> str:
    """Return 'tr' or 'en' using Turkish characters and stop-word overlap."""
    if not text or not text.strip():
        return "unknown"
    tokens = re.findall(r"[^\W\d_]+", text.lower())
    tr_score = sum(ch in TR_CHARS for ch in text) * 1.5
    tr_score += sum(t in TR_STOP or fold_ascii(t) in _TR_STOP_FOLDED for t in tokens)
    en_score = sum(t in EN_STOP for t in tokens)
    # Turkish agglutinative suffix hints on ASCII-typed text
    tr_score += 0.5 * sum(bool(re.search(r"(iyor|uyor|iyo|uyo|lar|ler|si|su|im|in|um|mi|mu|dir|dur|yim|sin)$",
                                         fold_ascii(t))) for t in tokens if len(t) > 3)
    if tr_score == 0 and en_score == 0:
        return "en"
    return "tr" if tr_score > en_score else "en"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
_TR_FOLD = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")


def fold_ascii(text: str) -> str:
    """Map Turkish letters to ASCII (ş->s, ı->i, ...). Used for matching only."""
    text = text.translate(_TR_FOLD)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


_TR_STOP_FOLDED = {fold_ascii(w) for w in TR_STOP}


def turkish_lower(text: str) -> str:
    """Lower-case with Turkish dotted/dotless I rules (I->ı, İ->i)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def smart_lower(text: str, lang: str) -> str:
    return turkish_lower(text) if lang == "tr" else text.lower()


URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s-]{8,}\d)(?!\w)")
REPEAT_RE = re.compile(r"(.)\1{2,}")  # "çooook" -> "çook"
NON_WORD_RE = re.compile(r"[^\w\s<>]")
SPACE_RE = re.compile(r"\s+")

# Product model code: 1-4 letters, optional hyphen/space, 3-5 digits, optional letter
# suffix. Requiring >=3 digits avoids matching error codes such as "E15"/"E24".
MODEL_CODE_RE = re.compile(r"\b([A-Za-z]{1,4})[-\s]?(\d{3,5})([A-Za-z]{0,2})\b")


def extract_model_codes(text: str) -> list[str]:
    codes = []
    for m in MODEL_CODE_RE.finditer(text):
        prefix, digits, suffix = m.groups()
        # Skip a lone word directly followed by a number, e.g. "model 123" or "in 2024"
        if m.group(0).count(" ") and prefix.lower() in {"model", "in", "on", "at", "the", "for", "gün", "yıl"}:
            continue
        if prefix.lower() in {"model", "no", "nr"}:
            continue
        codes.append(f"{prefix.upper()}-{digits}{suffix.upper()}")
    return codes


# ---------------------------------------------------------------------------
# Product lexicon + edit-distance spelling correction
# ---------------------------------------------------------------------------
# canonical form -> product category
PRODUCT_LEXICON = {
    # Turkish
    "bulaşık": "dishwasher", "çamaşır": "washing machine", "buzdolabı": "refrigerator",
    "fırın": "oven", "kurutma": "dryer", "ocak": "cooktop", "klima": "air conditioner",
    "dondurucu": "freezer", "aspiratör": "range hood", "süpürge": "vacuum cleaner",
    "kahve": "coffee machine", "makine": None, "makinesi": None,
    # English
    "dishwasher": "dishwasher", "washing": "washing machine", "washer": "washing machine",
    "refrigerator": "refrigerator", "fridge": "refrigerator", "oven": "oven",
    "dryer": "dryer", "cooktop": "cooktop", "freezer": "freezer",
    "conditioner": "air conditioner", "vacuum": "vacuum cleaner", "machine": None,
    "microwave": "microwave",
}
# Domain terms that are not products but are useful to correct.
DOMAIN_TERMS = ["garanti", "fatura", "kurulum", "iade", "arıza", "teknisyen",
                "warranty", "invoice", "installation", "refund", "technician", "repair"]

CORRECTION_TERMS = list(PRODUCT_LEXICON) + DOMAIN_TERMS
_FOLDED_TERMS = {fold_ascii(t): t for t in CORRECTION_TERMS}


def levenshtein(a: str, b: str) -> int:
    """Minimum edit distance (insert/delete/substitute cost 1), DP with two rows."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _max_dist(term: str) -> int:
    return 1 if len(term) <= 6 else 2


_KNOWN_WORDS: set[str] | None = None


def _known_words() -> set[str]:
    """ASCII-folded words seen >=2 times in the training corpus (built by
    scripts/prepare_dataset.py). Known words are never 'corrected' to a
    different word, which prevents over-correction (e.g. kahvaltı -> kahve)."""
    global _KNOWN_WORDS
    if _KNOWN_WORDS is None:
        from src.config import ARTIFACTS
        path = ARTIFACTS / "known_words.txt"
        _KNOWN_WORDS = set(path.read_text(encoding="utf-8").split()) if path.exists() else set()
    return _KNOWN_WORDS


def set_known_words(words: set[str]) -> None:
    global _KNOWN_WORDS
    _KNOWN_WORDS = set(words)


@lru_cache(maxsize=200_000)
def correct_token(token: str) -> tuple[str, str | None]:
    """Correct one lower-cased token against the product/domain lexicon.

    Handles Turkish suffixes by comparing the token prefix with the term
    ("bulasik" -> "bulaşık", "buzdolabim" -> "buzdolabım", "dishwaser" -> "dishwasher").
    Returns (corrected_token, matched_term or None).
    """
    if len(token) < 4 or not token.isalpha():
        return token, None
    ft = fold_ascii(token)
    known = ft in _known_words()
    best = None
    for fterm, term in _FOLDED_TERMS.items():
        if ft[0] != fterm[0] and ft[1:3] != fterm[1:3]:
            continue
        if len(ft) < len(fterm) - 2 or len(ft) > len(fterm) + 5:
            continue
        k = _max_dist(fterm)
        # whole-token match
        d = levenshtein(ft, fterm)
        cand = (d, 0, term, "")
        # prefix + suffix match (Turkish inflection: makinem, buzdolabım, fırınım)
        for cut in range(max(1, len(fterm) - 1), min(len(ft), len(fterm) + 1) + 1):
            dp = levenshtein(ft[:cut], fterm)
            suffix = token[cut:]
            if len(suffix) <= 5 and (dp, 1) < cand[:2]:
                cand = (dp, 1, term, suffix)
        if cand[0] <= k and (best is None or cand[:2] < best[:2]):
            best = cand
    if best is None:
        return token, None
    d, _, term, suffix = best
    if d == 0 and token == term + suffix:
        return token, term
    if d > 0 and known:
        return token, None  # a real word of the corpus, not a misspelling
    if d > 0 and suffix and fold_ascii(suffix) not in _TR_SUFFIXES:
        return token, None  # e.g. "kahvaltı" is not "kahve" + "ltı"
    corrected = term + suffix if suffix else term
    # Keep the user's exact token if only diacritics differ in the suffix part
    return corrected, term


# Common Turkish inflectional suffixes (ASCII-folded). A misspelled stem may only be
# corrected together with one of these endings.
_TR_SUFFIXES = {
    "m", "im", "um", "n", "in", "un", "i", "u", "e", "a", "y", "yi", "yu", "ye", "ya", "si", "su",
    "de", "da", "te", "ta", "den", "dan", "ten", "tan", "ler", "lar", "leri", "lari", "lerim", "larim",
    "imiz", "umuz", "miz", "muz", "nin", "nun", "inin", "unun", "sini", "sinin", "mi", "mu",
    "s", "es", "ed", "ing", "er", "ers",  # English endings
}


@dataclass
class Preprocessed:
    original: str
    language: str
    normalized: str            # for Transformer models (case kept lower, diacritics kept)
    folded: str                # ASCII-folded, used by TF-IDF / FastText models
    tokens: list[str]
    model_codes: list[str]
    product_category: str | None
    corrections: list[tuple[str, str]] = field(default_factory=list)
    is_empty: bool = False


def tokenize(text: str) -> list[str]:
    return [t for t in SPACE_RE.split(text) if t]


def preprocess(text: str, correct: bool = True) -> Preprocessed:
    """Full pipeline used by every model and by the application."""
    original = text if text is not None else ""
    if not original.strip():
        return Preprocessed(original, "unknown", "", "", [], [], None, [], True)
    text = original[:2000]  # hard cap protects the models from huge inputs
    lang = detect_language(text)
    codes = extract_model_codes(text)
    t = URL_RE.sub(" ", text)
    t = EMAIL_RE.sub(" ", t)
    t = MODEL_CODE_RE.sub(" ", t)
    t = PHONE_RE.sub(" ", t)
    t = smart_lower(t, lang)
    t = REPEAT_RE.sub(r"\1\1", t)
    t = t.replace("'", " ").replace("’", " ")
    t = NON_WORD_RE.sub(" ", t)
    t = SPACE_RE.sub(" ", t).strip()
    tokens = tokenize(t)
    corrections: list[tuple[str, str]] = []
    category = None
    out = []
    for tok in tokens:
        new, term = (correct_token(tok) if correct else (tok, None))
        if term is not None:
            if new != tok:
                corrections.append((tok, new))
            if category is None and PRODUCT_LEXICON.get(term):
                category = PRODUCT_LEXICON[term]
        out.append(new)
    # Two-word product names
    joined = " ".join(out)
    if category is None or category == "washing machine":
        if re.search(r"\b(washing machine|çamaşır makine)", joined):
            category = "washing machine"
    if re.search(r"\bair conditioner\b", joined):
        category = "air conditioner"
    normalized = joined
    folded = fold_ascii(normalized)
    return Preprocessed(original, lang, normalized, folded, out, codes, category, corrections, False)
