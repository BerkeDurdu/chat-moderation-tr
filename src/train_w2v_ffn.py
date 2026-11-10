"""Neural network (Area A): our own word2vec + a feedforward net trained with SGD and backprop.

word2vec (skip-gram) is trained only on the training messages. A message is represented by
the average of its word vectors and classified by a 2-hidden-layer MLP with dropout.
"""
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.metrics import f1_score

from .data import MODEL_DIR, SEED, load_splits, log_training
from .textnorm import tokenize

DIM, HIDDEN, EPOCHS, BATCH, LR = 200, (256, 64), 40, 64, 0.05


class FFN(nn.Module):
    def __init__(self, dim=DIM, hidden=HIDDEN, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], 2))

    def forward(self, x):
        return self.net(x)


def average_vectors(texts, vocab: dict, emb: np.ndarray) -> np.ndarray:
    out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
    for i, text in enumerate(texts):
        idx = [vocab[t] for t in tokenize(text) if t in vocab]
        if idx:
            out[i] = emb[idx].mean(axis=0)
    return out


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train, val, _ = load_splits()

    sentences = [tokenize(t) for t in train["text"]]
    w2v = Word2Vec(sentences, vector_size=DIM, window=5, min_count=2, sg=1, negative=10,
                   epochs=15, seed=SEED, workers=1)       # workers=1 -> reproducible
    vocab = {w: i for i, w in enumerate(w2v.wv.index_to_key)}
    emb = w2v.wv.vectors.astype(np.float32)
    print(f"[word2vec] vocabulary={len(vocab)}  dim={DIM}")

    x_tr = torch.from_numpy(average_vectors(train["text"], vocab, emb))
    y_tr = torch.tensor(train["label"].values)
    x_va = torch.from_numpy(average_vectors(val["text"], vocab, emb))

    model = FFN()
    counts = torch.bincount(y_tr).float()
    loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts))   # classes are imbalanced
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    best_f1, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(x_tr))
        total = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(x_tr[b]), y_tr[b])
            loss.backward()                                # backpropagation
            opt.step()                                     # SGD update
            total += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(x_va).argmax(1).numpy()
        f1 = f1_score(val["label"], pred, average="macro")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"[FFN] epoch {epoch:2d}  loss={total / len(perm):.4f}  val macro-F1={f1:.4f}")

    print(f"[FFN] best val macro-F1={best_f1:.4f}")
    log_training("w2v", {"val_f1": best_f1, "vocab": len(vocab), "dim": DIM})
    torch.save({"vocab": list(vocab), "emb": torch.from_numpy(emb), "state_dict": best_state},
               MODEL_DIR / "w2v_ffn.pt")


if __name__ == "__main__":
    main()
