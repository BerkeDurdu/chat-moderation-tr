"""Evaluate every model of the ladder on the same held-out test set (R3, R4).

Writes results/results.json, results/results.md and results/errors.json.
Latency is measured end to end (normalization + tokenization + model) for ONE message at a
time on CPU, averaged over N_LATENCY different test messages after a warm-up.
"""
import json
import os
import platform
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .data import RESULT_DIR, load_splits
from .models import REGISTRY

N_LATENCY, WARMUP, BUDGET_MS = 300, 20, 50.0


def cpu_name() -> str:
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or platform.machine()


def main():
    RESULT_DIR.mkdir(exist_ok=True)
    _, _, test = load_splits()
    texts, gold = test["text"].tolist(), test["label"].values
    rng = np.random.default_rng(0)
    lat_texts = [texts[i] for i in rng.choice(len(texts), N_LATENCY, replace=False)]

    rows, probs = [], {}
    for key, cls in REGISTRY.items():
        model = cls()
        p = np.array([model.prob(t) for t in texts])
        pred = (p >= 0.5).astype(int)
        probs[key] = p
        for t in lat_texts[:WARMUP]:
            model.prob(t)
        times = []
        for t in lat_texts:
            t0 = time.perf_counter()
            model.prob(t)
            times.append((time.perf_counter() - t0) * 1000)
        prec, rec, f1_off, _ = precision_recall_fscore_support(gold, pred, average="binary", zero_division=0)
        rows.append({
            "key": key, "model": model.name, "precision": model.precision,
            "macro_f1": f1_score(gold, pred, average="macro"),
            "accuracy": accuracy_score(gold, pred),
            "off_precision": prec, "off_recall": rec, "off_f1": f1_off,
            "size_mb": model.size_mb,
            "latency_ms_mean": float(np.mean(times)), "latency_ms_p95": float(np.percentile(times, 95)),
            "within_budget": bool(np.percentile(times, 95) <= BUDGET_MS),
        })
        r = rows[-1]
        print(f"{r['model']:36s} macro-F1={r['macro_f1']:.4f}  size={r['size_mb']:8.2f} MB  "
              f"latency={r['latency_ms_mean']:7.2f} ms (p95 {r['latency_ms_p95']:.2f})")
        del model

    meta = {
        "test_set": "OffensEval-TR 2020 official test set", "n_test": len(texts),
        "offensive_share": float(gold.mean()), "latency_runs": N_LATENCY, "budget_ms": BUDGET_MS,
        "hardware": f"{cpu_name()}, {os.cpu_count()} logical cores, CPU only", "os": platform.platform(),
    }
    (RESULT_DIR / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")

    md = ["| Model | Precision | Macro-F1 | Accuracy | Size (MB) | Latency mean (ms) | Latency p95 (ms) | <= 50 ms |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['precision']} | {r['macro_f1']:.3f} | {r['accuracy']:.3f} | "
                  f"{r['size_mb']:.1f} | {r['latency_ms_mean']:.2f} | {r['latency_ms_p95']:.2f} | "
                  f"{'yes' if r['within_budget'] else 'no'} |")
    md += ["", f"Test set: {meta['test_set']} ({meta['n_test']} messages, {meta['offensive_share']:.1%} offensive).",
           f"Latency: one message at a time, mean over {N_LATENCY} messages, on {meta['hardware']}."]
    (RESULT_DIR / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # failure examples of the deployed (quantized) model, most confident mistakes first
    p = probs["bert_int8"]
    wrong = np.where((p >= 0.5).astype(int) != gold)[0]
    wrong = sorted(wrong, key=lambda i: -abs(p[i] - 0.5))[:40]
    errors = [{"text": texts[i], "gold": int(gold[i]), "p_offensive": float(p[i])} for i in wrong]
    agree = float(((probs["bert_int8"] >= 0.5) == (probs["bert_onnx_fp32"] >= 0.5)).mean())
    (RESULT_DIR / "errors.json").write_text(
        json.dumps({"int8_fp32_agreement": agree, "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"INT8 vs FP32 prediction agreement: {agree:.2%}")


if __name__ == "__main__":
    main()
