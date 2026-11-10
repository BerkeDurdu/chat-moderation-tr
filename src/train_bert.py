"""Transformer (Area C): fine-tune BERTurk with a classification head.

Model: dbmdz/bert-base-turkish-cased (WordPiece tokenizer, 12 layers, 110M parameters).
"""
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .data import LABELS, MODEL_DIR, SEED, load_splits, log_training
from .textnorm import normalize

BASE = "dbmdz/bert-base-turkish-cased"
MAX_LEN, BATCH, EPOCHS, LR = 64, 32, 3, 3e-5


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


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BERT] training on {device}")
    train, val, _ = load_splits()
    texts, labels = train["text"].tolist(), torch.tensor(train["label"].values)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=2, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}).to(device)

    steps = EPOCHS * ((len(texts) + BATCH - 1) // BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best_f1, start = -1.0, time.time()
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
            best_f1 = f1
            model.save_pretrained(MODEL_DIR / "bert")
            tok.save_pretrained(MODEL_DIR / "bert")
    print(f"[BERT] best val macro-F1={best_f1:.4f}")
    name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    log_training("bert", {"val_f1": best_f1, "device": name, "minutes": (time.time() - start) / 60})


if __name__ == "__main__":
    main()
