"""Central configuration: paths, label set, intent-to-department mapping, seeds."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_MANUAL = ROOT / "data" / "manual"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

MASSIVE_DIR = DATA_RAW / "massive" / "1.1" / "data"
MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MASSIVE_LOCALES = {"en": "en-US", "tr": "tr-TR"}

SEED = 42

OOS_LABEL = "out_of_scope"

# ---------------------------------------------------------------------------
# In-scope intents.
#   * MASSIVE intents keep their ORIGINAL label names (they exist in MASSIVE 1.1).
#   * Manual intents are labels we defined ourselves for the 100 hand-written
#     appliance-support examples in data/manual/appliance_examples.csv.
# ---------------------------------------------------------------------------
MASSIVE_INSCOPE_INTENTS = [
    "iot_cleaning",
    "iot_coffee",
    "iot_hue_lightchange",
    "iot_hue_lightdim",
    "iot_hue_lightoff",
    "iot_hue_lighton",
    "iot_hue_lightup",
    "iot_wemo_off",
    "iot_wemo_on",
    "takeaway_order",
    "takeaway_query",
    "general_greet",
    "general_joke",
]

MANUAL_INTENTS = [
    "appliance_malfunction",
    "installation_request",
    "warranty_inquiry",
    "return_refund",
    "billing_payment",
]

# MASSIVE scenarios used as *seen* out-of-scope data (an explicit OOS class in
# training, CLINC150-style). Their val/test splits are also OOS.
OOS_SEEN_SCENARIOS = ["alarm", "audio", "calendar", "music", "news", "play", "social"]

# MASSIVE scenarios/intents that are NEVER seen during training or threshold
# tuning. They appear only in the test set to measure OOS recall on unseen topics.
OOS_UNSEEN_SCENARIOS = ["weather", "qa", "recommendation", "lists", "email",
                        "datetime", "transport", "cooking"]
OOS_UNSEEN_INTENTS = ["general_quirky"]

LABELS = sorted(MASSIVE_INSCOPE_INTENTS + MANUAL_INTENTS) + [OOS_LABEL]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# Application-level routing: intent -> department.
# ---------------------------------------------------------------------------
HUMAN_AGENT = "Human Agent"
SMALL_TALK = "Small Talk (auto-reply)"

INTENT_TO_DEPARTMENT = {
    "appliance_malfunction": "Technical Support",
    "installation_request": "Installation & Field Service",
    "warranty_inquiry": "Warranty & Returns",
    "return_refund": "Warranty & Returns",
    "billing_payment": "Billing & Payments",
    "iot_cleaning": "Smart Home Support",
    "iot_coffee": "Smart Home Support",
    "iot_hue_lightchange": "Smart Home Support",
    "iot_hue_lightdim": "Smart Home Support",
    "iot_hue_lightoff": "Smart Home Support",
    "iot_hue_lighton": "Smart Home Support",
    "iot_hue_lightup": "Smart Home Support",
    "iot_wemo_off": "Smart Home Support",
    "iot_wemo_on": "Smart Home Support",
    "takeaway_order": "Orders & Delivery",
    "takeaway_query": "Orders & Delivery",
    "general_greet": SMALL_TALK,
    "general_joke": SMALL_TALK,
    OOS_LABEL: HUMAN_AGENT,
}

# Sampling caps (per language) to keep the OOS class from dominating.
OOS_TRAIN_CAP_PER_LANG = 700
OOS_VAL_CAP_PER_LANG = 120
OOS_TEST_SEEN_CAP_PER_LANG = 150
OOS_TEST_UNSEEN_CAP_PER_LANG = 150

MODEL_NAMES = {
    "classical": "TF-IDF + LogReg",
    "neural": "FastText + FFNN",
    "transformer": "XLM-R base (FP32)",
    "quantized": "Distilled mDistilBERT (INT8)",
}
