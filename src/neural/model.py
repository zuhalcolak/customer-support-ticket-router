"""Model 2 - FastText embeddings (trained on our training messages only) averaged per
message, fed into a feedforward network trained with SGD + backpropagation (PyTorch)."""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from gensim.models import FastText

from src import config as C
from src.preprocessing.text import Preprocessed

DIR = C.ARTIFACTS / "neural"


def train_fasttext(sentences: list[list[str]], dim: int = 150, seed: int = C.SEED) -> FastText:
    return FastText(sentences=sentences, vector_size=dim, window=5, min_count=1, sg=1,
                    min_n=3, max_n=5, epochs=40, seed=seed, workers=1,
                    bucket=100_000)  # gensim default 2M buckets = 1.2 GB; 100k is plenty for 3.8k sentences


def average_embeddings(ft: FastText, token_lists: list[list[str]]) -> np.ndarray:
    dim = ft.wv.vector_size
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        if toks:
            out[i] = np.mean([ft.wv[t] for t in toks], axis=0)  # subword vectors for OOV tokens
    return out


class FeedForward(nn.Module):
    """input(avg embedding) -> 256 -> 128 -> |labels| ; ReLU, dropout; softmax applied outside."""

    def __init__(self, in_dim: int, n_classes: int, hidden=(256, 128), dropout: float = 0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralModel:
    name = "neural"

    def __init__(self, ft: FastText, net: FeedForward, mean: np.ndarray, std: np.ndarray):
        self.ft, self.net, self.mean, self.std = ft, net.eval(), mean, std

    def features(self, token_lists: list[list[str]]) -> torch.Tensor:
        x = (average_embeddings(self.ft, token_lists) - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32))

    @torch.no_grad()
    def predict_proba_tokens(self, token_lists: list[list[str]]) -> np.ndarray:
        return torch.softmax(self.net(self.features(token_lists)), -1).numpy()

    def predict_proba(self, pre: list[Preprocessed]) -> np.ndarray:
        return self.predict_proba_tokens([p.folded.split() for p in pre])

    def save(self, config: dict) -> None:
        DIR.mkdir(parents=True, exist_ok=True)
        self.ft.save(str(DIR / "fasttext.model"))
        torch.save(self.net.state_dict(), DIR / "ffnn.pt")
        np.savez(DIR / "norm.npz", mean=self.mean, std=self.std)
        (DIR / "config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls) -> "NeuralModel":
        cfg = json.loads((DIR / "config.json").read_text())
        ft = FastText.load(str(DIR / "fasttext.model"))
        net = FeedForward(cfg["in_dim"], cfg["n_classes"], tuple(cfg["hidden"]), cfg["dropout"])
        net.load_state_dict(torch.load(DIR / "ffnn.pt", map_location="cpu"))
        norm = np.load(DIR / "norm.npz")
        return cls(ft, net, norm["mean"], norm["std"])
