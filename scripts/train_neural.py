"""Train Model 2: FastText + feedforward network (SGD, backpropagation).

Usage: python scripts/train_neural.py
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data.dataset import load_split  # noqa: E402
from src.evaluation.metrics import compute_metrics, save_predictions, update_threshold  # noqa: E402
from src.neural.model import FeedForward, NeuralModel, average_embeddings, train_fasttext  # noqa: E402

CFG = dict(dim=150, fasttext_bucket=100_000, hidden=[256, 128], dropout=0.3, lr=0.02, momentum=0.9, weight_decay=1e-4,
           batch_size=32, max_epochs=400, patience=40, optimizer="SGD (momentum)")


def main() -> None:
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    train, val, test = (load_split(s) for s in ["train", "val", "test"])
    tok = {s: d.folded.str.split().tolist() for s, d in [("train", train), ("val", val), ("test", test)]}

    t0 = time.time()
    ft = train_fasttext(tok["train"], dim=CFG["dim"])  # training messages only
    ft_time = time.time() - t0

    xtr = average_embeddings(ft, tok["train"])
    mean, std = xtr.mean(0), xtr.std(0) + 1e-6
    norm = lambda x: torch.from_numpy(((x - mean) / std).astype(np.float32))  # noqa: E731
    Xtr, Xva = norm(xtr), norm(average_embeddings(ft, tok["val"]))
    ytr, yva = torch.tensor(train.y.values), torch.tensor(val.y.values)

    counts = np.bincount(train.y, minlength=len(C.LABELS)).astype(np.float32)
    weights = torch.tensor(len(train) / (len(C.LABELS) * np.maximum(counts, 1)))
    net = FeedForward(CFG["dim"], len(C.LABELS), tuple(CFG["hidden"]), CFG["dropout"])
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.SGD(net.parameters(), lr=CFG["lr"], momentum=CFG["momentum"],
                          weight_decay=CFG["weight_decay"])

    history, best, best_state, wait = [], float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, CFG["max_epochs"] + 1):
        net.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), CFG["batch_size"]):
            idx = perm[i:i + CFG["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(net(Xtr[idx]), ytr[idx])
            loss.backward()          # backpropagation
            opt.step()               # SGD update
            tot += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            vlogits = net(Xva)
            vloss = loss_fn(vlogits, yva).item()
            vacc = (vlogits.argmax(1) == yva).float().mean().item()
        history.append({"epoch": epoch, "train_loss": tot / len(Xtr), "val_loss": vloss, "val_acc": vacc})
        score = -vacc + 1e-3 * vloss  # early stopping on validation accuracy (loss breaks ties)
        if score < best - 1e-6:
            best, best_state, wait = score, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= CFG["patience"]:
                break
    train_time = time.time() - t0
    net.load_state_dict(best_state)

    model = NeuralModel(ft, net, mean, std)
    model.save({**CFG, "in_dim": CFG["dim"], "n_classes": len(C.LABELS)})
    vp = model.predict_proba_tokens(tok["val"])
    tp = model.predict_proba_tokens(tok["test"])
    save_predictions("neural", vp, tp)
    thr = update_threshold("neural", vp, val.y.to_numpy())
    log = {"config": CFG, "fasttext_seconds": ft_time, "train_seconds": train_time,
           "epochs_run": len(history), "best_val_score": best, "threshold": thr,
           "val": compute_metrics(vp, val.y.to_numpy(), thr),
           "test": compute_metrics(tp, test.y.to_numpy(), thr, test), "history": history}
    out = C.REPORTS / "training_logs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "neural.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: v for k, v in log.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
