"""One inference wrapper per model, shared by the evaluation script and the application.

Every wrapper exposes:  name, precision, size_mb, prob(text) -> P(offensive) for one message.
All of them run on CPU.
"""
import json
from pathlib import Path

import numpy as np

from .data import MODEL_DIR
from .textnorm import normalize


def _mb(*paths: Path) -> float:
    return sum(p.stat().st_size for p in paths) / 1e6


def _softmax_off(logits: np.ndarray) -> float:
    z = logits - logits.max()
    e = np.exp(z)
    return float(e[1] / e.sum())


class LexiconModel:
    name, precision = "Lexicon + edit distance (rule)", "-"

    def __init__(self):
        from .lexicon import LexiconMatcher
        path = MODEL_DIR / "lexicon.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        self.matcher = LexiconMatcher(cfg["words"], cfg["fuzzy_min_len"])
        self.size_mb = _mb(path)

    def prob(self, text: str) -> float:
        return self.matcher.score(text)


class TfidfModel:
    name, precision = "tf-idf + logistic regression", "FP64"

    def __init__(self):
        import joblib
        path = MODEL_DIR / "tfidf_lr.joblib"
        self.pipe = joblib.load(path)
        self.size_mb = _mb(path)

    def prob(self, text: str) -> float:
        return float(self.pipe.predict_proba([text])[0, 1])


class W2vFfnModel:
    name, precision = "word2vec + feedforward net", "FP32"

    def __init__(self):
        import torch
        from .train_w2v_ffn import FFN, average_vectors
        path = MODEL_DIR / "w2v_ffn.pt"
        ckpt = torch.load(path, map_location="cpu")
        self.vocab = {w: i for i, w in enumerate(ckpt["vocab"])}
        self.emb = ckpt["emb"].numpy()
        self.net = FFN()
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self._avg, self._torch = average_vectors, torch
        self.size_mb = _mb(path)

    def prob(self, text: str) -> float:
        x = self._torch.from_numpy(self._avg([text], self.vocab, self.emb))
        with self._torch.no_grad():
            return _softmax_off(self.net(x)[0].numpy())


class BertTorchModel:
    name, precision = "BERTurk fine-tuned (PyTorch)", "FP32"

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = MODEL_DIR / "bert"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        self._torch = torch
        self.size_mb = _mb(path / "model.safetensors")

    def prob(self, text: str) -> float:
        enc = self.tok(normalize(text, lower=False), truncation=True, max_length=64, return_tensors="pt")
        with self._torch.no_grad():
            return _softmax_off(self.model(**enc).logits[0].numpy())


class _BertOnnx:
    file = ""

    def __init__(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        path = MODEL_DIR / "bert_onnx"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.sess = ort.InferenceSession(str(path / self.file), providers=["CPUExecutionProvider"])
        self.size_mb = _mb(path / self.file)

    def prob(self, text: str) -> float:
        enc = self.tok(normalize(text, lower=False), truncation=True, max_length=64, return_tensors="np")
        return _softmax_off(self.sess.run(None, dict(enc))[0][0])


class BertOnnxFp32Model(_BertOnnx):
    name, precision, file = "BERTurk fine-tuned (ONNX Runtime)", "FP32", "model_fp32.onnx"


class BertOnnxInt8Model(_BertOnnx):
    name, precision, file = "BERTurk quantized (ONNX Runtime)", "INT8", "model_int8.onnx"


def thresholds() -> dict:
    """Decision thresholds chosen on the validation split by src/tune_thresholds.py."""
    path = MODEL_DIR / "thresholds.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def get(key: str):
    """Instantiate a model and attach its decision threshold (default 0.5)."""
    model = REGISTRY[key]()
    model.threshold = thresholds().get(key, 0.5)
    return model


# the model ladder, in order; the last one is what the application serves by default
REGISTRY = {
    "lexicon": LexiconModel,
    "tfidf": TfidfModel,
    "w2v_ffn": W2vFfnModel,
    "bert_fp32": BertTorchModel,
    "bert_onnx_fp32": BertOnnxFp32Model,
    "bert_int8": BertOnnxInt8Model,
}
