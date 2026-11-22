"""Build the bilingual (tr/en) intent dataset with train/val/test splits.

Sources
  * MASSIVE 1.1 en-US + tr-TR (official train/dev/test partitions are kept, and
    MASSIVE ids are parallel across locales, so a translated utterance never
    lands in a different split than its source).
  * 100 hand-written appliance-support examples (data/manual/appliance_examples.csv),
    split 6/2/2 per (intent, language).
  * Augmented copies of the manual TRAIN examples only (product swap + ASCII typing).

Out-of-scope (OOS) strategy
  * "seen" OOS: MASSIVE scenarios in OOS_SEEN_SCENARIOS form an explicit
    `out_of_scope` class in train/val/test.
  * "unseen" OOS: MASSIVE scenarios in OOS_UNSEEN_SCENARIOS (+ general_quirky)
    appear ONLY in the test set, to measure OOS recall on topics never seen.

Usage: python scripts/prepare_dataset.py
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.preprocessing.text import fold_ascii, preprocess, set_known_words  # noqa: E402

PRODUCT_SWAPS = {
    "tr": [("bulaşık makinesi", "çamaşır makinesi"), ("bulaşık makinem", "çamaşır makinem"),
           ("buzdolabı", "derin dondurucu"), ("buzdolabım", "derin dondurucum"),
           ("fırın", "ocak"), ("fırınım", "ocağım"), ("klima", "kurutma makinesi"),
           ("çamaşır makinesi", "kurutma makinesi"), ("çamaşır makinem", "bulaşık makinem")],
    "en": [("dishwasher", "washing machine"), ("washing machine", "dryer"),
           ("refrigerator", "freezer"), ("fridge", "freezer"), ("oven", "cooktop"),
           ("air conditioner", "dishwasher"), ("dryer", "washing machine"), ("freezer", "fridge")],
}


def load_massive() -> pd.DataFrame:
    rows = []
    for lang, locale in C.MASSIVE_LOCALES.items():
        with open(C.MASSIVE_DIR / f"{locale}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append({"massive_id": r["id"], "lang": lang, "partition": r["partition"],
                             "scenario": r["scenario"], "original_intent": r["intent"],
                             "text": r["utt"]})
    df = pd.DataFrame(rows)
    df["split"] = df["partition"].map({"train": "train", "dev": "val", "test": "test"})
    return df


def massive_role(row) -> str:
    if row.original_intent in C.MASSIVE_INSCOPE_INTENTS:
        return "inscope"
    if row.original_intent in C.OOS_UNSEEN_INTENTS or row.scenario in C.OOS_UNSEEN_SCENARIOS:
        return "oos_unseen"
    if row.scenario in C.OOS_SEEN_SCENARIOS:
        return "oos_seen"
    return "unused"


def sample_ids(df: pd.DataFrame, n: int, rng: random.Random) -> set:
    """Sample MASSIVE ids (shared by tr/en) stratified by original intent."""
    ids_by_intent = collections.defaultdict(list)
    for mid, intent in df[["massive_id", "original_intent"]].drop_duplicates().itertuples(index=False):
        ids_by_intent[intent].append(mid)
    total = sum(len(v) for v in ids_by_intent.values())
    chosen = set()
    for intent in sorted(ids_by_intent):
        ids = sorted(ids_by_intent[intent], key=int)
        k = max(1, round(n * len(ids) / total))
        chosen.update(rng.sample(ids, min(k, len(ids))))
    return chosen


def build_massive(rng: random.Random) -> pd.DataFrame:
    df = load_massive()
    df["role"] = df.apply(massive_role, axis=1)
    parts = [df[df.role == "inscope"].assign(label=lambda d: d.original_intent, oos_type="none")]
    caps = {("oos_seen", "train"): C.OOS_TRAIN_CAP_PER_LANG, ("oos_seen", "val"): C.OOS_VAL_CAP_PER_LANG,
            ("oos_seen", "test"): C.OOS_TEST_SEEN_CAP_PER_LANG, ("oos_unseen", "test"): C.OOS_TEST_UNSEEN_CAP_PER_LANG}
    for (role, split), cap in caps.items():
        sub = df[(df.role == role) & (df.split == split)]
        ids = sample_ids(sub, cap, rng)
        parts.append(sub[sub.massive_id.isin(ids)].assign(label=C.OOS_LABEL, oos_type=role.split("_")[1]))
    out = pd.concat(parts, ignore_index=True)
    out["source"] = "massive"
    out["uid"] = "massive-" + out.lang + "-" + out.massive_id
    return out.drop(columns=["partition", "role"])


def build_manual(rng: random.Random) -> pd.DataFrame:
    m = pd.read_csv(C.DATA_MANUAL / "appliance_examples.csv")
    m["split"] = ""
    for (_, _), g in m.groupby(["intent", "lang"]):
        idx = list(g.index)
        rng.shuffle(idx)
        m.loc[idx[:6], "split"] = "train"
        m.loc[idx[6:8], "split"] = "val"
        m.loc[idx[8:], "split"] = "test"
    return pd.DataFrame({"uid": "manual-" + m.id, "massive_id": "", "lang": m.lang, "split": m.split,
                         "scenario": "appliance_support", "original_intent": m.intent, "text": m.text,
                         "label": m.intent, "oos_type": "none", "source": "manual"})


def augment(manual_train: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """Augment manual TRAIN rows only (-> no leakage):
    (1) swap the product noun, (2) ASCII typing without Turkish letters,
    (3) greeting prefix / help-request suffix, (4) appended random model code,
    (5) one adjacent-character swap typo in the longest word."""
    prefixes = {"tr": ["Merhaba, ", "İyi günler, ", "Selam, "], "en": ["Hello, ", "Hi, ", "Good morning, "]}
    suffixes = {"tr": [" Lütfen yardım edin.", " Acil dönüş bekliyorum."], "en": [" Please help.", " I need help asap."]}
    rows = []
    for r in manual_train.itertuples(index=False):
        variants = set()
        variants.add(rng.choice(prefixes[r.lang]) + r.text[0].lower() + r.text[1:])
        variants.add(r.text + rng.choice(suffixes[r.lang]))
        code = f"{rng.choice(['DW', 'WM', 'RF', 'OV', 'AC', 'TD'])}-{rng.randint(1000, 9999)}"
        variants.add(r.text.rstrip(".?!") + f" model {code}")
        words = r.text.split()
        j = max(range(len(words)), key=lambda k: len(words[k]))
        w = words[j]
        if len(w) > 4:
            pos = rng.randrange(1, len(w) - 2)
            words[j] = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
            variants.add(" ".join(words))
        low = r.text.lower() if r.lang == "en" else r.text
        for a, b in PRODUCT_SWAPS[r.lang]:
            if a in low:
                variants.add(low.replace(a, b))
        variants.add(fold_ascii(r.text).lower())
        variants.add(r.text.rstrip(".?!").replace(",", ""))
        for i, v in enumerate(sorted(variants)):
            if v != r.text:
                d = r._asdict()
                d.update(uid=f"{r.uid}-aug{i}", text=v, source="manual_augmented")
                rows.append(d)
    return pd.DataFrame(rows)


def main() -> None:
    rng = random.Random(C.SEED)
    massive = build_massive(rng)
    manual = build_manual(rng)
    aug = augment(manual[manual.split == "train"], rng)
    df = pd.concat([massive, manual, aug], ignore_index=True)

    # Known-word list for the spelling corrector: built from TRAIN text only.
    counts = collections.Counter(w for t in df[df.split == "train"].text
                                 for w in fold_ascii(t.lower()).split())
    known = {w.strip(".,!?;:'\"()") for w, c in counts.items() if c >= 2}
    C.ARTIFACTS.mkdir(exist_ok=True)
    (C.ARTIFACTS / "known_words.txt").write_text("\n".join(sorted(known)), encoding="utf-8")
    set_known_words(known)

    pre = [preprocess(t) for t in df.text]
    df["normalized"] = [p.normalized for p in pre]
    df["folded"] = [p.folded for p in pre]

    # Leakage control: drop TRAIN rows whose normalized text also occurs in val/test,
    # and VAL rows that occur in test.
    test_norm = set(df[df.split == "test"].normalized)
    val_norm = set(df[df.split == "val"].normalized)
    before = len(df)
    leak_train = (df.split == "train") & df.normalized.isin(test_norm | val_norm)
    leak_val = (df.split == "val") & df.normalized.isin(test_norm)
    df = df[~(leak_train | leak_val)].reset_index(drop=True)
    removed = before - len(df)
    # Remove exact duplicates inside each split (same text + label)
    dup = df.duplicated(subset=["split", "lang", "normalized", "label"])
    df = df[~dup].reset_index(drop=True)

    assert set(df.label) <= set(C.LABELS), set(df.label) - set(C.LABELS)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        df[df.split == split].to_csv(C.DATA_PROCESSED / f"{split}.csv", index=False)

    stats = {
        "removed_for_leakage": int(removed),
        "removed_duplicates": int(dup.sum()),
        "split_sizes": df.split.value_counts().to_dict(),
        "by_split_lang": df.groupby(["split", "lang"]).size().unstack().to_dict(),
        "by_split_source": df.groupby(["split", "source"]).size().unstack(fill_value=0).to_dict(),
        "by_split_oos_type": df.groupby(["split", "oos_type"]).size().unstack(fill_value=0).to_dict(),
        "by_label_split": df.groupby(["label", "split"]).size().unstack(fill_value=0).to_dict(),
        "test_oos_original_intents": df[(df.split == "test") & (df.label == C.OOS_LABEL)]
        .groupby("oos_type").original_intent.nunique().to_dict(),
        "known_words": len(known),
    }
    (C.REPORTS).mkdir(exist_ok=True)
    (C.REPORTS / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: stats[k] for k in ["removed_for_leakage", "removed_duplicates", "split_sizes",
                                              "by_split_lang", "by_split_source", "by_split_oos_type"]}, indent=1))


if __name__ == "__main__":
    main()
