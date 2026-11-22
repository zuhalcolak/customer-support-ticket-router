# Dataset Report

This file documents every piece of data used by the Customer Support Ticket Router, how it
was filtered and transformed, and how the train / validation / test splits were built.
All numbers below were produced by `scripts/prepare_dataset.py` and are saved in
`reports/dataset_stats.json`.

## 1. Sources

| | MASSIVE 1.1 (primary) | Manually authored examples |
|---|---|---|
| Origin | Amazon Science, FitzGerald et al. 2022, *MASSIVE: A 1M-Example Multilingual NLU Dataset* (arXiv:2204.08582) | Written for this project |
| URL | https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz (repo: https://github.com/alexa/massive) | `data/manual/appliance_examples.csv` |
| License | CC BY 4.0 (`data/raw/massive/1.1/LICENSE`). MASSIVE is a translation of SLURP, also CC BY 4.0 | Same as this repository (MIT) |
| Languages used | `en-US`, `tr-TR` (51 locales exist) | Turkish (50), English (50) |
| Original size | 16,521 utterances per locale (11,514 train / 2,033 dev / 2,974 test), 18 scenarios, 60 intents | 100 sentences, 5 intents x 2 languages x 10 |
| Labels | Original MASSIVE intents (e.g. `iot_cleaning`, `takeaway_query`) | 5 labels we defined (see below) |

Why MASSIVE and not CLINC150? CLINC150 is English only and has a built-in `oos` class.
MASSIVE has Turkish **and** English with *parallel* utterance ids and official splits.
Parallel ids matter: the Turkish sentence with id 1 is a translation of the English sentence
with id 1 and both are in the same official partition, so a translation of a test sentence can
never leak into training. CLINC150 was not used, so its labels were not mixed with MASSIVE.

### Honest note about domain fit
MASSIVE is a virtual-assistant dataset, not a home-appliance support dataset. No MASSIVE label
says "my dishwasher is broken". We therefore:
* use the MASSIVE intents that are closest to a smart-home appliance company
  (`iot_*`: robot vacuum, coffee machine, lights, smart plugs; `takeaway_*`: orders and order
  status; `general_greet`, `general_joke`: small talk) **with their original names**;
* add **100 hand-written** appliance-support sentences (the assignment allows 50-100) for the
  intents MASSIVE does not have;
* use all other MASSIVE scenarios as **out-of-scope** data.

The `takeaway_*` intents are about food take-away orders in MASSIVE; we use them as a proxy for
the "Orders & Delivery" team. This is a documented approximation, not a claim that MASSIVE
contains appliance orders.

### The manually authored examples
`data/manual/appliance_examples.csv` (columns `id, lang, intent, text`) contains 100 sentences
written for this project (drafted with help from an AI assistant and reviewed by the author; see
`AI_USAGE.md`). They are **not** collected from real customers. Intents:

| Label (ours) | Meaning | Examples |
|---|---|---|
| `appliance_malfunction` | device fault / error | "Bulaşık makinem su boşaltmıyor.", "My oven is not working at all." |
| `installation_request` | installation / set-up appointment | "Klimanın montajı ücretli mi?" |
| `warranty_inquiry` | warranty status / extension | "Ürünümün garantisi ne zaman bitiyor?" |
| `return_refund` | return, exchange, refund | "What is the status of my refund?" |
| `billing_payment` | invoices, charges, installments | "Kartımdan iki kez ücret çekilmiş." |

## 2. Label set and department mapping (19 labels)

| Label | Source | Department |
|---|---|---|
| appliance_malfunction | manual | Technical Support |
| installation_request | manual | Installation & Field Service |
| warranty_inquiry, return_refund | manual | Warranty & Returns |
| billing_payment | manual | Billing & Payments |
| iot_cleaning, iot_coffee, iot_hue_lightchange, iot_hue_lightdim, iot_hue_lightoff, iot_hue_lighton, iot_hue_lightup, iot_wemo_off, iot_wemo_on | MASSIVE | Smart Home Support |
| takeaway_order, takeaway_query | MASSIVE | Orders & Delivery |
| general_greet, general_joke | MASSIVE | Small Talk (auto-reply) |
| out_of_scope | MASSIVE (other scenarios), relabelled | Human Agent |

The mapping lives in `src/config.py` (`INTENT_TO_DEPARTMENT`). Low-confidence predictions are
also sent to *Human Agent* (see `src/routing/router.py`).

## 3. Out-of-scope (OOS) strategy

MASSIVE scenarios that are not in scope are split into two disjoint groups:

| Group | MASSIVE scenarios | Used in |
|---|---|---|
| **seen OOS** | alarm, audio, calendar, music, news, play, social | train + val + test, label `out_of_scope` |
| **unseen OOS** | weather, qa, recommendation, lists, email, datetime, transport, cooking, and the intent `general_quirky` | **test only** |

* Seen OOS gives the models an explicit `out_of_scope` class (like CLINC150's `oos`).
* Unseen OOS measures whether OOS detection generalises to topics never seen in training or
  threshold tuning. The test set contains 22 different seen-OOS intents and 24 unseen-OOS intents.
* OOS sampling is capped per language and stratified by original intent (seed 42):
  train 700, val 120, test 150 seen + 150 unseen. The same MASSIVE ids are drawn for both
  languages.
* In addition, a confidence threshold (tuned on the validation split only) sends uncertain
  predictions to the human agent.

## 4. Splits

* MASSIVE: official partitions (`train` -> train, `dev` -> val, `test` -> test).
* Manual examples: per (intent, language) 6 train / 2 val / 2 test, shuffled with seed 42.
* Augmentation is applied **only to manual training rows** (so no augmented variant of a
  test sentence exists): product-noun swap, ASCII typing without Turkish letters, greeting
  prefix, help-request suffix, appended random model code, one adjacent-letter typo.
  273 augmented rows; they are marked `source = manual_augmented`.

### Leakage control
* 131 rows removed: training rows whose normalized text also appears in val/test, and val rows
  whose normalized text appears in test.
* 171 exact duplicates (same split, language, normalized text, label) removed.
* Thresholds and early stopping use the validation split only; the test split is used once,
  by `scripts/evaluate_all.py`.
* The spelling corrector's known-word list (`artifacts/known_words.txt`, 1,700 words) is built
  from **training** text only.

### Final sizes

| Split | Total | en | tr | MASSIVE | manual | augmented | in-scope | seen OOS | unseen OOS |
|---|---|---|---|---|---|---|---|---|---|
| train | 3,808 | 1,968 | 1,840 | 3,475 | 60 | 273 | 2,414 | 1,394 | 0 |
| val | 609 | 310 | 299 | 589 | 20 | 0 | 368 | 241 | 0 |
| test | 1,206 | 609 | 597 | 1,186 | 20 | 0 | 600 | 306 | 300 |

Per-label counts (test / train / val):

| Label | test | train | val |
|---|---|---|---|
| appliance_malfunction | 4 | 67 | 4 |
| billing_payment | 4 | 64 | 4 |
| general_greet | 2 | 47 | 3 |
| general_joke | 37 | 141 | 30 |
| installation_request | 4 | 67 | 4 |
| iot_cleaning | 48 | 161 | 37 |
| iot_coffee | 72 | 233 | 27 |
| iot_hue_lightchange | 72 | 242 | 43 |
| iot_hue_lightdim | 41 | 130 | 32 |
| iot_hue_lightoff | 81 | 256 | 33 |
| iot_hue_lighton | 6 | 41 | 10 |
| iot_hue_lightup | 54 | 146 | 23 |
| iot_wemo_off | 33 | 86 | 9 |
| iot_wemo_on | 20 | 88 | 13 |
| out_of_scope | 606 | 1,394 | 241 |
| return_refund | 4 | 69 | 4 |
| takeaway_order | 44 | 267 | 40 |
| takeaway_query | 70 | 243 | 48 |
| warranty_inquiry | 4 | 66 | 4 |

**Limitation:** the five appliance intents have only 4 test sentences each (20 in total), so
their per-class scores have high variance. They are reported separately
(`accuracy[source=manual]` in `reports/results.csv`).

## 5. Preprocessing stored with the data
`data/processed/{train,val,test}.csv` contain the raw `text` plus two derived columns produced
by `src/preprocessing/text.py`:
* `normalized` – Turkish-aware lower-casing, URL/e-mail/phone/model-code removal, punctuation
  removal, repeated-letter squashing, product-term spelling correction. Used by the Transformers.
* `folded` – `normalized` mapped to ASCII (ş→s, ı→i, …). Used by TF-IDF and FastText.

## 6. Reproduce
```bash
python scripts/download_dataset.py   # downloads + extracts en-US / tr-TR only
python scripts/prepare_dataset.py    # writes data/processed/*.csv and reports/dataset_stats.json
```

## Citation
FitzGerald, J. et al. (2022). *MASSIVE: A 1M-Example Multilingual Natural Language Understanding
Dataset with 51 Typologically-Diverse Languages.* arXiv:2204.08582.
Bastianelli, E. et al. (2020). *SLURP: A Spoken Language Understanding Resource Package.* EMNLP.
