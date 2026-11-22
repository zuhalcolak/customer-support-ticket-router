"""Model 1 - classical baseline: TF-IDF (word + character n-grams) + Logistic Regression."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config as C
from src.preprocessing.text import Preprocessed

PATH = C.ARTIFACTS / "classical" / "tfidf_logreg.joblib"


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                 token_pattern=r"(?u)\b\w+\b")),
        # character n-grams make the baseline robust to Turkish suffixes and typos
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")
    return Pipeline([("tfidf", features), ("clf", clf)])


class ClassicalModel:
    name = "classical"

    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline

    @classmethod
    def load(cls) -> "ClassicalModel":
        return cls(joblib.load(PATH))

    def save(self) -> None:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, PATH)

    def predict_proba_texts(self, folded_texts: list[str]) -> np.ndarray:
        """Return probabilities in the global LABELS order."""
        p = self.pipeline.predict_proba(folded_texts)
        out = np.zeros((len(folded_texts), len(C.LABELS)), dtype=np.float32)
        out[:, self.pipeline.classes_] = p
        return out

    def predict_proba(self, pre: list[Preprocessed]) -> np.ndarray:
        return self.predict_proba_texts([p.folded for p in pre])
