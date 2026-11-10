| Model | Precision | Macro-F1 | Accuracy | Size (MB) | Latency mean (ms) | Latency p95 (ms) | <= 50 ms |
|---|---|---|---|---|---|---|---|
| Lexicon + edit distance (rule) | - | 0.659 | 0.786 | 0.0 | 0.04 | 0.08 | yes |
| tf-idf + logistic regression | FP64 | 0.765 | 0.852 | 2.5 | 1.58 | 2.36 | yes |
| word2vec + feedforward net | FP32 | 0.699 | 0.787 | 24.0 | 0.21 | 0.40 | yes |
| BERTurk fine-tuned (PyTorch) | FP32 | 0.816 | 0.889 | 442.5 | 33.21 | 50.98 | no |
| BERTurk fine-tuned (ONNX Runtime) | FP32 | 0.816 | 0.889 | 442.7 | 23.59 | 37.73 | yes |
| BERTurk quantized (ONNX Runtime) | INT8 | 0.815 | 0.889 | 111.4 | 11.22 | 20.83 | yes |

Test set: OffensEval-TR 2020 official test set (3528 messages, 20.3% offensive).
Latency: one message at a time, mean over 300 messages, on 13th Gen Intel(R) Core(TM) i5-13500H, 16 logical cores, CPU only.
