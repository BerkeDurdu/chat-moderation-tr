# Real-Time Chat Moderation (Turkish)

SWE015 – Introduction to Large Language Models · Homework, **Use Case 4**.

A chat-like application that labels every Turkish message live as **clean** or **offensive**,
masks flagged messages and logs the response time of each message. The same task is solved
with four models of increasing sophistication, and the application serves the last one:

| # | Model | Area |
|---|---|---|
| – | Rule baseline: lexicon mined from the training data + minimum edit distance | B |
| 1 | Classical baseline: tf-idf (word + character n-grams) + logistic regression | B |
| 2 | Neural network: our own word2vec + feedforward net (SGD, backpropagation, dropout) | A (+B) |
| 3 | Transformer: BERTurk fine-tuned for classification (two checkpoints tried, picked on validation) | C |
| 4 | Quantized Transformer: the same BERTurk, dynamic INT8, ONNX Runtime – **runs in the app** | D |

## Setup

Requires Python 3.11 (64-bit). No GPU is needed to run the application.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python download_models.py         # trained weights from the Hugging Face Hub (~1 GB)
```

## Run

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>. Type a Turkish message in the chat box: it is labelled
clean / offensive with a confidence, masked if offensive, and its response time is logged in the
side panel. The side panel also switches the active model; "Show all models side by side" runs
every model on each message. The default model is the INT8-quantized BERTurk.

Try for example: `gg iyi oyundu herkese teşekkürler` (clean), `4ptal mısın sen saaaalak`
(leetspeak + stretched letters, offensive).

### Alternative: Docker (Windows, macOS, Linux)

With [Docker](https://docs.docker.com/get-docker/) installed, no Python setup is needed; the image
installs the pinned packages (CPU-only PyTorch) and downloads the trained weights itself:

```bash
docker build -t chat-moderation-tr .
docker run --rm -p 8501:8501 chat-moderation-tr
```

Then open <http://localhost:8501>. The first build downloads about 2 GB and takes a few minutes.

## Reproduce the results from scratch (optional)

```bash
python train_all.py      # downloads the data, trains all models, quantizes, evaluates
python make_slides.py    # builds slides/presentation.pdf (copy slides_config.example.json to slides_config.json first)
```

BERT fine-tuning takes a few minutes on a GPU (for a CUDA build of PyTorch see pytorch.org);
everything else runs on CPU. Outputs: `models/` (weights), `results/results.md`,
`results/results.json`, `results/training.json`, `results/errors.json`.

## Results

All models are evaluated on the same held-out test set (measured by `src/evaluate.py`):

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

Decision thresholds (`models/thresholds.json`) were chosen on the validation split; the test set
was used only for the numbers above.

Latency budget: p95 ≤ 50 ms per message on CPU. The quantized model is 4× smaller and 3.4× faster than
the PyTorch FP32 model and does not lose macro-F1.

## Architecture

```
user message ──> Streamlit UI (app.py)
                    │
                    ▼
        regex normalization (src/textnorm.py)
        URLs, @mentions, leetspeak, repeated letters, Turkish I/İ,
        abbreviations (orsp → orospu, aq/amk, o.ç)
                    │
                    ▼   selected model (or all, side by side)      src/models.py
   ┌────────────┬───────────────┬─────────────────┬───────────────┬──────────────────────┐
   │ lexicon +  │ tf-idf + LR   │ word2vec avg +  │ BERTurk       │ BERTurk INT8         │
   │ edit dist. │ sklearn, FP64 │ FFN, torch FP32 │ torch, FP32   │ ONNX Runtime (default)│
   └────────────┴───────────────┴─────────────────┴───────────────┴──────────────────────┘
                    │ P(offensive)
                    ▼
        post-processing: threshold (tuned on validation) → label + confidence, mask, stop timer
                    │
                    ▼
        Streamlit UI: chat bubble · response-time log (side panel) · comparison table
```

Everything runs in one Python process on CPU.

## Repository layout

```
app.py                  Streamlit chat application
train_all.py            full pipeline: train → quantize → evaluate
make_slides.py          builds the presentation PDF from the measured results
download_models.py      fetches the weights from the Hugging Face Hub
src/textnorm.py         regex normalization (B)
src/lexicon.py          edit distance + lexicon rule baseline (B)
src/train_classical.py  lexicon mining, tf-idf + logistic regression (B)
src/train_w2v_ffn.py    word2vec + feedforward network (A)
src/train_bert.py       BERTurk fine-tuning (C)
src/quantize.py         ONNX export + dynamic INT8 quantization (D)
src/tune_thresholds.py  decision thresholds chosen on the validation split
src/models.py           inference wrappers shared by app and evaluation
src/evaluate.py         metrics, model size and latency on the held-out test set
results/                measured numbers
slides_config.example.json  template for the presentation text
```

## Data and model sources

* **OffensEval-TR 2020** – Çöltekin, Ç. (2020). *A Corpus of Turkish Offensive Language on
  Social Media.* LREC 2020. <https://coltekin.github.io/offensive-turkish/> (CC-BY 2.0).
  Official training set (31,756 tweets, split 90/10 into train/validation) and official
  test set (3,528 tweets).
* **BERTurk** – Stefan Schweter. Candidates `dbmdz/bert-base-turkish-cased` and
  `dbmdz/bert-base-turkish-128k-cased` (<https://huggingface.co/dbmdz>); the one with the higher
  validation macro-F1 is used.
