| Model | Precision | Macro-F1 | Accuracy | Size (MB) | Latency mean (ms) | Latency p95 (ms) | <= 50 ms |
|---|---|---|---|---|---|---|---|
| Lexicon + edit distance (rule) | - | 0.687 | 0.798 | 0.0 | 0.07 | 0.15 | yes |
| tf-idf + logistic regression | FP64 | 0.766 | 0.853 | 2.5 | 2.22 | 4.14 | yes |
| word2vec + feedforward net | FP32 | 0.702 | 0.789 | 24.0 | 0.36 | 0.59 | yes |
| BERTurk fine-tuned (PyTorch) | FP32 | 0.816 | 0.889 | 442.5 | 31.68 | 49.99 | yes |
| BERTurk fine-tuned (ONNX Runtime) | FP32 | 0.816 | 0.889 | 442.7 | 20.29 | 32.20 | yes |
| BERTurk quantized (ONNX Runtime) | INT8 | 0.816 | 0.889 | 111.4 | 11.04 | 18.80 | yes |

Test set: OffensEval-TR 2020 official test set (3528 messages, 20.3% offensive).
Latency: one message at a time, mean over 300 messages, on 13th Gen Intel(R) Core(TM) i5-13500H, 16 logical cores, CPU only.
