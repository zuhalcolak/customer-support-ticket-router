"""Model 3 - fine-tuned multilingual Transformer (XLM-RoBERTa base) with a classification head.
Also contains the shared Hugging Face wrapper and training loop used by the student."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src import config as C
from src.preprocessing.text import Preprocessed

TEACHER_NAME = "FacebookAI/xlm-roberta-base"
TEACHER_DIR = C.ARTIFACTS / "transformer"
MAX_LEN = 64


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def new_classifier(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(C.LABELS), id2label=C.ID2LABEL, label2id=C.LABEL2ID)
    return tok, model


class HFClassifier:
    """Thin inference wrapper: text -> probabilities in LABELS order."""

    def __init__(self, tokenizer, model, device: torch.device | str = "cpu", name: str = "transformer"):
        self.tok, self.model, self.device, self.name = tokenizer, model.eval(), torch.device(device), name
        self.model.to(self.device)

    @classmethod
    def load(cls, path: Path = TEACHER_DIR, device: str = "cpu") -> "HFClassifier":
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return cls(tok, model, device)

    @torch.no_grad()
    def logits_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        outs = []
        for i in range(0, len(texts), batch_size):
            # fixed-length padding on GPU/MPS avoids re-compiling kernels for every new shape
            pad = True if self.device.type == "cpu" else "max_length"
            enc = self.tok(texts[i:i + batch_size], padding=pad, truncation=True,
                           max_length=MAX_LEN, return_tensors="pt").to(self.device)
            outs.append(self.model(**enc).logits.float().cpu())
        return torch.cat(outs).numpy()

    def predict_proba_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return torch.softmax(torch.from_numpy(self.logits_texts(texts, batch_size)), -1).numpy()

    def predict_proba(self, pre: list[Preprocessed]) -> np.ndarray:
        return self.predict_proba_texts([p.normalized for p in pre])


def train_classifier(tok, model, train_texts, train_y, val_texts, val_y, *, epochs, lr, batch_size,
                     device, out_dir: Path, teacher_logits: np.ndarray | None = None,
                     alpha: float = 0.0, temperature: float = 1.0, warmup_ratio: float = 0.1,
                     weight_decay: float = 0.01, seed: int = C.SEED, log_name: str = "model") -> dict:
    """Fine-tune `model`. If `teacher_logits` is given, train with the knowledge-distillation
    loss  alpha * T^2 * KL(softmax(t/T) || softmax(s/T)) + (1 - alpha) * CE(s, y)."""
    torch.manual_seed(seed)
    model.to(device)
    n = len(train_texts)
    steps_per_epoch = math.ceil(n / batch_size)
    total = steps_per_epoch * epochs
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    groups = [
        {"params": [p for nme, p in model.named_parameters() if not any(nd in nme for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for nme, p in model.named_parameters() if any(nd in nme for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=lr)
    sched = get_linear_schedule_with_warmup(opt, int(warmup_ratio * total), total)
    y_all = torch.tensor(np.asarray(train_y))
    t_all = torch.tensor(teacher_logits, dtype=torch.float32) if teacher_logits is not None else None
    history, best_acc = [], -1.0
    g = torch.Generator().manual_seed(seed)
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        run_loss = 0.0
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            enc = tok([train_texts[i] for i in idx], padding="max_length", truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to(device)
            logits = model(**enc).logits
            y = y_all[idx].to(device)
            loss = F.cross_entropy(logits, y)
            if t_all is not None:
                tl = t_all[idx].to(device)
                kd = F.kl_div(F.log_softmax(logits / temperature, -1), F.softmax(tl / temperature, -1),
                              reduction="batchmean") * temperature ** 2
                loss = alpha * kd + (1 - alpha) * loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            run_loss += loss.item()
            if step % 20 == 0:
                print(f"[{log_name}] epoch {epoch} step {step}/{steps_per_epoch} loss={loss.item():.4f} "
                      f"t={time.time() - t_start:.0f}s", flush=True)
        clf = HFClassifier(tok, model, device)
        vp = clf.predict_proba_texts(val_texts)
        vacc = float((vp.argmax(1) == np.asarray(val_y)).mean())
        history.append({"epoch": epoch, "train_loss": run_loss / steps_per_epoch, "val_acc": vacc,
                        "elapsed_s": time.time() - t_start})
        print(f"[{log_name}] epoch {epoch}: loss={run_loss / steps_per_epoch:.4f} val_acc={vacc:.4f}", flush=True)
        if vacc > best_acc:
            best_acc = vacc
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
    return {"history": history, "best_val_acc": best_acc, "train_seconds": time.time() - t_start,
            "steps": total}


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
