"""Model 4 - distilled student (multilingual DistilBERT) quantized to INT8 for CPU.

Quantization method: PyTorch eager-mode *dynamic* quantization
  * every nn.Linear -> int8 weights, activations quantized on the fly per batch
  * nn.Embedding (word/position) -> int8 weight-only quantization (per-row float qparams)
Backend: 'qnnpack' on ARM (Apple Silicon), 'fbgemm' on x86.
"""
from __future__ import annotations

import json
import warnings

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src import config as C
from src.transformer.model import HFClassifier

STUDENT_NAME = "distilbert/distilbert-base-multilingual-cased"
STUDENT_DIR = C.ARTIFACTS / "student"            # distilled FP32 student
QUANT_DIR = C.ARTIFACTS / "quantized"            # INT8 student
QUANT_WEIGHTS = QUANT_DIR / "model_int8.pt"


def set_quant_engine() -> str:
    engines = torch.backends.quantized.supported_engines
    engine = "fbgemm" if "fbgemm" in engines else "qnnpack"
    torch.backends.quantized.engine = engine
    return engine


def quantize(model: nn.Module) -> nn.Module:
    from torch.ao.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig, quantize_dynamic
    set_quant_engine()
    model = model.cpu().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return quantize_dynamic(model, {nn.Linear: default_dynamic_qconfig,
                                        nn.Embedding: float_qparams_weight_only_qconfig}, dtype=torch.qint8)


def save_quantized(qmodel: nn.Module, tokenizer, meta: dict) -> None:
    QUANT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel.state_dict(), QUANT_WEIGHTS)
    qmodel.config.save_pretrained(QUANT_DIR)
    tokenizer.save_pretrained(QUANT_DIR)
    (QUANT_DIR / "quantization.json").write_text(json.dumps(meta, indent=2))


def load_quantized() -> HFClassifier:
    """Rebuild the student architecture, apply the same quantization, load INT8 weights."""
    cfg = AutoConfig.from_pretrained(QUANT_DIR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoModelForSequenceClassification.from_config(cfg)
        q = quantize(model)
        q.load_state_dict(torch.load(QUANT_WEIGHTS, map_location="cpu", weights_only=False))
    tok = AutoTokenizer.from_pretrained(QUANT_DIR)
    clf = HFClassifier(tok, q, "cpu")
    clf.name = "quantized"
    return clf


def count_quantized_modules(model: nn.Module) -> dict:
    """Count float vs quantized Linear/Embedding modules. Note: the quantized classes are
    also *named* `Linear`/`Embedding` (torch.ao.nn.quantized...), so isinstance is used."""
    import torch.ao.nn.quantized as nnq
    import torch.ao.nn.quantized.dynamic as nnqd
    return {
        "int8_dynamic_linear": sum(isinstance(m, nnqd.Linear) for m in model.modules()),
        "int8_embedding": sum(isinstance(m, nnq.Embedding) for m in model.modules()),
        "float_linear": sum(type(m) is nn.Linear for m in model.modules()),
        "float_embedding": sum(type(m) is nn.Embedding for m in model.modules()),
    }


def is_fully_quantized(model: nn.Module) -> bool:
    c = count_quantized_modules(model)
    return c["int8_dynamic_linear"] > 0 and c["float_linear"] == 0 and c["float_embedding"] == 0
