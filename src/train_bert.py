"""Transformer (Area C): fine-tune BERTurk with a classification head.

Two pretrained BERTurk checkpoints are fine-tuned with the same recipe; the one with the higher
validation macro-F1 is kept in models/bert (model selection never looks at the test set).
  dbmdz/bert-base-turkish-cased       32k WordPiece vocabulary, 110M parameters
  dbmdz/bert-base-turkish-128k-cased  128k WordPiece vocabulary, 184M parameters
"""
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .data import LABELS, MODEL_DIR, SEED, load_splits, log_training
from .textnorm import normalize

import shutil

CANDIDATES = ["dbmdz/bert-base-turkish-cased", "dbmdz/bert-base-turkish-128k-cased"]
MAX_LEN, BATCH, EPOCHS, LR = 64, 32, 4, 3e-5


def encode(tok, texts):
    return tok([normalize(t, lower=False) for t in texts], truncation=True, max_length=MAX_LEN,
               padding=True, return_tensors="pt")


@torch.no_grad()
def predict(model, tok, texts, device, batch=128):
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = encode(tok, texts[i:i + batch]).to(device)
        out.append(model(**enc).logits.argmax(1).cpu().numpy())
    return np.concatenate(out)


def train_one(base: str, out_dir):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BERT] fine-tuning {base} on {device}")
    train, val, _ = load_splits()
    texts, labels = train["text"].tolist(), torch.tensor(train["label"].values)

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}).to(device)

    steps = EPOCHS * ((len(texts) + BATCH - 1) // BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best_f1, best_epoch, start = -1.0, 0, time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(texts))
        total = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            enc = encode(tok, [texts[j] for j in b]).to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**enc, labels=labels[b].to(device)).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(b)
        f1 = f1_score(val["label"], predict(model, tok, val["text"].tolist(), device), average="macro")
        print(f"[BERT] epoch {epoch}  loss={total / len(perm):.4f}  val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
    print(f"[BERT] {base}: best val macro-F1={best_f1:.4f} (epoch {best_epoch})")
    return {"base": base, "val_f1": best_f1, "best_epoch": best_epoch,
            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6),
            "vocab": tok.vocab_size, "minutes": round((time.time() - start) / 60, 1)}


def main():
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    runs = []
    for i, base in enumerate(CANDIDATES):
        out = MODEL_DIR / f"bert_candidate_{i}"
        runs.append({**train_one(base, out), "dir": out})
        torch.cuda.empty_cache()
    best = max(runs, key=lambda r: r["val_f1"])
    shutil.rmtree(MODEL_DIR / "bert", ignore_errors=True)
    best["dir"].rename(MODEL_DIR / "bert")
    for r in runs:
        shutil.rmtree(r["dir"], ignore_errors=True)
    print(f"[BERT] selected {best['base']} (val macro-F1 {best['val_f1']:.4f})")
    log_training("bert", {**{k: v for k, v in best.items() if k != "dir"}, "device": device,
                          "candidates": [{k: v for k, v in r.items() if k != "dir"} for r in runs]})


if __name__ == "__main__":
    main()
