"""Efficient model (Area D): export the fine-tuned BERTurk to ONNX and quantize it to INT8.

Method: post-training dynamic quantization with ONNX Runtime. The weights of every
MatMul / Gemm (attention + feed-forward layers) are stored as 8-bit integers; activations
are quantized on the fly at inference time, so no calibration data is needed.
"""
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import MODEL_DIR


def main():
    src, dst = MODEL_DIR / "bert", MODEL_DIR / "bert_onnx"
    dst.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    model.config.return_dict = False

    enc = tok("örnek bir mesaj", return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    fp32, int8 = dst / "model_fp32.onnx", dst / "model_int8.onnx"
    torch.onnx.export(
        model, tuple(enc[n] for n in names), str(fp32),
        input_names=names, output_names=["logits"],
        dynamic_axes={**{n: {0: "batch", 1: "seq"} for n in names}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    tok.save_pretrained(dst)
    for p in (fp32, int8):
        print(f"[quantize] {p.name}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
