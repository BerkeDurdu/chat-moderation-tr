"""Choose each learned model's decision threshold on the VALIDATION split (never the test set).

A message is flagged when P(offensive) >= threshold. Classes are imbalanced, so 0.5 is not
necessarily the best cut-off for macro-F1. The rule baseline keeps its fixed rule (0.5).
Writes models/thresholds.json, read by src/models.py.
"""
import json

import numpy as np
from sklearn.metrics import f1_score

from .data import MODEL_DIR, load_splits, log_training
from .models import REGISTRY

GRID = np.round(np.arange(0.10, 0.91, 0.01), 2)
FIXED = {"lexicon": 0.5}


def main():
    _, val, _ = load_splits()
    out = dict(FIXED)
    for key, cls in REGISTRY.items():
        if key in FIXED:
            continue
        model = cls()
        p = np.array([model.prob(t) for t in val["text"]])
        scores = [f1_score(val["label"], (p >= t).astype(int), average="macro") for t in GRID]
        best = int(np.argmax(scores))
        out[key] = float(GRID[best])
        at_half = f1_score(val["label"], (p >= 0.5).astype(int), average="macro")
        print(f"[threshold] {key:15s} val macro-F1 {at_half:.4f} @0.50 -> {scores[best]:.4f} @{GRID[best]:.2f}")
    (MODEL_DIR / "thresholds.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log_training("thresholds", out)


if __name__ == "__main__":
    main()
