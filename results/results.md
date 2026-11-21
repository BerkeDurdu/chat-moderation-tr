| Model | Precision | Macro-F1 | Accuracy | Size (MB) | Latency mean (ms) | Latency p95 (ms) | <= 50 ms |
|---|---|---|---|---|---|---|---|
| Lexicon + edit distance (rule) | - | 0.697 | 0.830 | 0.0 | 0.07 | 0.16 | yes |
| tf-idf + logistic regression | FP64 | 0.765 | 0.853 | 2.5 | 1.37 | 1.75 | yes |
| word2vec + feedforward net | FP32 | 0.708 | 0.802 | 24.0 | 0.20 | 0.33 | yes |
| BERTurk fine-tuned (PyTorch) | FP32 | 0.813 | 0.882 | 442.5 | 28.62 | 44.72 | yes |
| BERTurk fine-tuned (ONNX Runtime) | FP32 | 0.813 | 0.882 | 442.7 | 18.41 | 29.80 | yes |
| BERTurk quantized (ONNX Runtime) | INT8 | 0.822 | 0.889 | 111.4 | 8.47 | 13.97 | yes |

Test set: OffensEval-TR 2020 official test set (3528 messages, 20.3% offensive).
Latency: one message at a time, mean over 300 messages, on 13th Gen Intel(R) Core(TM) i5-13500H, 16 logical cores, CPU only.
