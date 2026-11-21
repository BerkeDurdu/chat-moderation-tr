"""Reproduce everything: download data, train the four models, quantize, evaluate.

    python train_all.py

A GPU is used for BERT fine-tuning when available (about 25 minutes for both BERTurk candidates on an RTX 4050);
everything else, including all latency measurements, runs on CPU.
"""
from src import evaluate, quantize, train_bert, train_classical, train_w2v_ffn, tune_thresholds

if __name__ == "__main__":
    for step in (train_classical, train_w2v_ffn, train_bert, quantize, tune_thresholds, evaluate):
        print(f"\n=== {step.__name__} ===")
        step.main()
