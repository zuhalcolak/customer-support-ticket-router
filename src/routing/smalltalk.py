"""ELIZA-style rule-based replies for greetings and small talk (regex pattern -> template).

Like ELIZA, rules are ordered regular expressions; captured fragments are reflected
back into the reply ("I feel tired" -> "Why do you feel tired?").
Rules only fire for short, purely social messages, so a message such as
"Hello, my dishwasher is broken" is NOT treated as small talk.
"""
from __future__ import annotations

import random
import re

from src.preprocessing.text import fold_ascii

REFLECT_EN = {"i": "you", "my": "your", "me": "you", "am": "are", "i'm": "you are", "you": "I", "your": "my"}

# (language, pattern on ASCII-folded lower-case text, reply templates)
RULES: list[tuple[str, re.Pattern, list[str]]] = [
    ("tr", re.compile(r"^(merhaba+|selam+|slm|mrb|iyi (gunler|aksamlar|sabahlar)|gunaydin|hey)\b"),
     ["Merhaba! Size nasıl yardımcı olabilirim? Ürününüzle ilgili sorunu yazabilirsiniz.",
      "Merhaba, hoş geldiniz! Hangi ürünle ilgili destek almak istersiniz?"]),
    ("tr", re.compile(r"\b(nasilsin|nasilsiniz|naber|ne haber)\b"),
     ["Teşekkür ederim, iyiyim! Sizin için ne yapabilirim?"]),
    ("tr", re.compile(r"\b(tesekkur|tesekkurler|sagol|sag ol|eyvallah|mersi)\b"),
     ["Rica ederim! Başka bir konuda yardımcı olabilir miyim?"]),
    ("tr", re.compile(r"^(gorusuruz|hosca kal|bay bay|iyi geceler)\b"),
     ["Görüşmek üzere! İyi günler dileriz."]),
    ("tr", re.compile(r"\b(sen kimsin|adin ne|bot musun|robot musun)\b"),
     ["Ben otomatik destek asistanıyım. Mesajınızı doğru ekibe yönlendiriyorum."]),
    ("tr", re.compile(r"\bkendimi (.+) hissediyorum\b"),
     ["Neden kendinizi {0} hissediyorsunuz? Ürününüzle ilgili bir sorun varsa yardımcı olabilirim."]),
    ("en", re.compile(r"^(hello+|hi+|hey+|good (morning|afternoon|evening)|greetings|howdy)\b"),
     ["Hello! How can I help you today? Please describe the issue with your appliance.",
      "Hi there! Which product do you need help with?"]),
    ("en", re.compile(r"\bhow are you\b|\bhow is it going\b|\bwhat'?s up\b"),
     ["I'm doing well, thank you! What can I do for you?"]),
    ("en", re.compile(r"\b(thank you|thanks|thx|cheers)\b"),
     ["You're welcome! Is there anything else I can help with?"]),
    ("en", re.compile(r"^(bye|goodbye|see you|good night)\b"),
     ["Goodbye! Have a nice day."]),
    ("en", re.compile(r"\b(who are you|what are you|are you a (bot|robot|human))\b"),
     ["I'm the automated support assistant. I route your message to the right team."]),
    ("en", re.compile(r"\bi feel (.+)"),
     ["Why do you feel {0}? If something is wrong with your appliance, I can help."]),
]

# Words that show the message is a real support request, not small talk.
SUPPORT_HINT = re.compile(
    r"(makin|buzdolab|firin|ocak|klima|kurutma|garanti|iade|fatura|odeme|kurulum|siparis|ariza|calismiyor|bozuk|"
    r"dishwash|washing|washer|fridge|refrigerat|oven|dryer|freezer|warranty|refund|return|invoice|payment|"
    r"install|order|broken|not working|repair|charge|lamba|isik|light|coffee|kahve|supurge|vacuum)")


def _reflect(fragment: str) -> str:
    return " ".join(REFLECT_EN.get(w, w) for w in fragment.split())


def small_talk_reply(text: str, rng: random.Random | None = None) -> str | None:
    """Return a rule-based reply, or None when the message is not small talk."""
    if not text or not text.strip():
        return None
    t = fold_ascii(text.lower()).strip()
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t.split()) > 8 or SUPPORT_HINT.search(t):
        return None
    rng = rng or random.Random(0)
    for _, pattern, replies in RULES:
        m = pattern.search(t)
        if m:
            reply = rng.choice(replies)
            if "{0}" in reply:
                reply = reply.format(_reflect(m.group(m.lastindex or 1)))
            return reply
    return None
