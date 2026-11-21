"""Build slides/presentation.pdf from results/results.json and slides_config.json.

    python make_slides.py

Every number on the slides is read from the files written by src/evaluate.py, so the
presentation always matches the last real run. Edit slides_config.json for your name,
student ID and links.
"""
import json
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

ROOT = Path(__file__).resolve().parent
W, H = 960, 540
NAVY, TEAL, GREY, LIGHT, RED = (colors.HexColor(c) for c in ("#14213d", "#0f766e", "#5b6472", "#eef2f5", "#b42318"))
PALE_TEAL = colors.HexColor("#d5efec")

_fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("R", str(_fonts / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("B", str(_fonts / "DejaVuSans-Bold.ttf")))

cfg = json.loads((ROOT / "slides_config.json").read_text(encoding="utf-8"))
res = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
err = json.loads((ROOT / "results" / "errors.json").read_text(encoding="utf-8"))
meta, rows = res["meta"], {r["key"]: r for r in res["rows"]}
LADDER = ["tfidf", "w2v_ffn", "bert_fp32", "bert_int8"]
train_info = json.loads((ROOT / "results" / "training.json").read_text(encoding="utf-8"))
BERT = train_info["bert"]
BASE_SHORT = BERT["base"].split("/")[-1]                     # e.g. bert-base-turkish-128k-cased
THR = train_info.get("thresholds", {})


def thr(k):
    return f"{THR.get(k, 0.5):.2f}"

(ROOT / "slides").mkdir(exist_ok=True)
c = canvas.Canvas(str(ROOT / "slides" / "presentation.pdf"), pagesize=(W, H))
c.setTitle("Real-Time Chat Moderation - SWE015 Homework")
c.setAuthor(cfg["name"])
page = 0


def header(section: str, title: str):
    global page
    page += 1
    c.setFillColor(NAVY)
    c.rect(0, H - 8, W, 8, stroke=0, fill=1)
    c.setFillColor(TEAL)
    c.setFont("B", 11)
    c.drawString(48, H - 44, section.upper())
    c.setFillColor(NAVY)
    c.setFont("B", 26)
    c.drawString(48, H - 78, title)
    c.setFillColor(GREY)
    c.setFont("R", 9)
    c.drawString(48, 22, "SWE015 · Use Case 4 – Real-Time Chat Moderation")
    c.drawRightString(W - 48, 22, str(page))


def text(x, y, s, size=14, font="R", color=NAVY, width=None, leading=None):
    """Draw wrapped text, return the y below it."""
    c.setFillColor(color)
    c.setFont(font, size)
    leading = leading or size * 1.38
    for line in (simpleSplit(s, font, size, width) if width else [s]):
        c.drawString(x, y, line)
        y -= leading
    return y


def bullets(x, y, items, size=14, width=860, gap=7):
    for item in items:
        c.setFillColor(TEAL)
        c.circle(x + 4, y + size * 0.33, 2.6, stroke=0, fill=1)
        y = text(x + 18, y, item, size, width=width - 18) - gap
    return y


def box(x, y, w, h, title, lines=(), fill=LIGHT, title_color=NAVY, size=9.5):
    c.setFillColor(fill)
    c.setStrokeColor(colors.HexColor("#c5ced8"))
    c.roundRect(x, y, w, h, 6, stroke=1, fill=1)
    yy = text(x + 8, y + h - 15, title, 10.5, "B", title_color, width=w - 14, leading=13)
    for line in lines:
        yy = text(x + 8, yy - 1, line, size, color=GREY, width=w - 14, leading=size * 1.25)


def arrow(x1, y1, x2, y2):
    c.setStrokeColor(NAVY)
    c.setFillColor(NAVY)
    c.setLineWidth(1.3)
    c.line(x1, y1, x2, y2)
    if x1 == x2:                                    # vertical
        d = -1 if y2 < y1 else 1
        p = c.beginPath(); p.moveTo(x2, y2); p.lineTo(x2 - 4, y2 - 7 * d); p.lineTo(x2 + 4, y2 - 7 * d); p.close()
    else:                                           # horizontal
        d = -1 if x2 < x1 else 1
        p = c.beginPath(); p.moveTo(x2, y2); p.lineTo(x2 - 7 * d, y2 - 4); p.lineTo(x2 - 7 * d, y2 + 4); p.close()
    c.drawPath(p, stroke=0, fill=1)


def table(data, x, y_top, col_widths, size=11, highlight_row=None, row_h=None):
    t = Table(data, colWidths=col_widths, rowHeights=row_h)
    style = [("FONT", (0, 0), (-1, -1), "R", size), ("FONT", (0, 0), (-1, 0), "B", size),
             ("TEXTCOLOR", (0, 0), (-1, -1), NAVY), ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
             ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#c5ced8")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
             ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]
    if highlight_row is not None:
        style += [("BACKGROUND", (0, highlight_row), (-1, highlight_row), PALE_TEAL),
                  ("FONT", (0, highlight_row), (-1, highlight_row), "B", size)]
    t.setStyle(TableStyle(style))
    _, h = t.wrapOn(c, W, H)
    t.drawOn(c, x, y_top - h)
    return y_top - h


def f1(k):  return f"{rows[k]['macro_f1']:.3f}"
def ms(k):  return f"{rows[k]['latency_ms_mean']:.1f}"
def p95(k): return f"{rows[k]['latency_ms_p95']:.2f}"
def mb(k):  return f"{rows[k]['size_mb']:.1f}"


int8, fp32, onnx32 = rows["bert_int8"], rows["bert_fp32"], rows["bert_onnx_fp32"]
size_ratio = fp32["size_mb"] / int8["size_mb"]
speedup = fp32["latency_ms_mean"] / int8["latency_ms_mean"]
f1_drop = fp32["macro_f1"] - int8["macro_f1"]

# ------------------------------------------------------------------ 1 · title
page += 1
c.setFillColor(NAVY)
c.rect(0, 0, W, H, stroke=0, fill=1)
c.setFillColor(colors.HexColor("#5eead4"))
c.setFont("B", 12)
c.drawString(60, 420, "SWE015 – INTRODUCTION TO LARGE LANGUAGE MODELS · HOMEWORK")
text(60, 370, "Real-Time Chat Moderation", 40, "B", colors.white)
text(60, 328, "Flagging offensive Turkish chat messages within 50 ms on a CPU", 18, color=colors.white)
c.setStrokeColor(colors.HexColor("#5eead4"))
c.line(60, 290, 300, 290)
y = 255
for label, value in (("Student", f"{cfg['name']} · {cfg['student_id']}"),
                     ("Use case", "4 – Real-Time Chat Moderation (Turkish, OffensEval-TR 2020)"),
                     ("Repository", cfg["repo_url"]), ("Demo video", cfg["video_url"])):
    text(60, y, label, 12, "B", colors.HexColor("#5eead4"))
    text(170, y, value, 13, color=colors.white)
    y -= 30
c.showPage()

# ------------------------------------------------------------------ 2 · introduction
header("Introduction", "The problem and what success looks like")
y = text(48, H - 120, "Problem", 15, "B", TEAL)
y = bullets(48, y - 4, [
    "A gaming platform must hide offensive chat messages before other players read them. "
    "Moderators cannot read thousands of messages per minute, and the server has no GPU.",
    "Players write noisy Turkish on purpose: leetspeak (s1kt1r), stretched letters (saaalak), missing Turkish "
    "characters (serefsiz). A plain word list is easy to evade.",
], width=860)
y = text(48, y - 10, "Who it is for", 15, "B", TEAL)
y = bullets(48, y - 4, ["Platform moderators (fewer messages to review) and players (a cleaner chat, live)."])
y = text(48, y - 10, "Success criteria", 15, "B", TEAL)
bullets(48, y - 4, [
    "Quality: the highest macro-F1 on the official OffensEval-TR test set (classes are imbalanced, "
    f"{meta['offensive_share']:.0%} offensive, so accuracy alone is misleading).",
    f"Speed: 95% of messages answered in ≤ {meta['budget_ms']:.0f} ms, one message at a time, CPU only.",
    "Delivery: the quantized Transformer runs inside a chat application anyone can try.",
])
c.showPage()

# ------------------------------------------------------------------ 3 · data
header("Development · Data", "Data: OffensEval-TR 2020")
y = bullets(48, H - 122, [
    "Source: Çöltekin (2020), “A Corpus of Turkish Offensive Language on Social Media”, LREC – the Turkish "
    "set of SemEval-2020 Task 12. Public, CC-BY 2.0, downloaded from coltekin.github.io/offensive-turkish.",
    "Turkish tweets labelled NOT / OFF by human annotators. User names are already replaced by @USER.",
], width=860)
t = train_info["splits"]
y = table([["Split", "Messages", "Offensive", "Used for"],
           ["Train", f"{t['train']['n']:,}", f"{t['train']['off']:.1%}", "fitting all four models (and word2vec)"],
           ["Validation", f"{t['val']['n']:,}", f"{t['val']['off']:.1%}", "hyper-parameters, early stopping (stratified 10% of train, seed 42)"],
           ["Test", f"{t['test']['n']:,}", f"{t['test']['off']:.1%}", "official held-out test set – reported numbers only"]],
          48, y - 6, [110, 100, 100, 554], size=11.5)
y = text(48, y - 28, "Preprocessing (regex, src/textnorm.py) – shared by every model, the evaluation and the app", 15, "B", TEAL)
bullets(48, y - 4, [
    "Remove URLs, replace @mentions by a placeholder; Turkish-aware lower-casing (I → ı, İ → i).",
    "De-leetspeak inside words (0→o, 1→i, 3→e, 4→a, 5→s, 7→t, $→s); real numbers such as “100lerce” are kept.",
    "Collapse letters repeated 3+ times (saaaalak → salak). ASCII-folding (ş → s) for the lexicon matcher.",
    "Expand abbreviations: shorthands (aq, amk, o.ç) and vowel-dropped words (orsp → orospu, sktr → siktir).",
    "BERTurk gets the same cleaning but keeps the original case (it is a cased model).",
], size=13)
c.showPage()

# ------------------------------------------------------------------ 4 · architecture
header("Development · Architecture", "Architecture: from a typed message to a moderated chat")
top = H - 150
box(40, top - 20, 120, 62, "User", ["types a chat message in the browser"])
box(190, top - 20, 150, 62, "Streamlit UI", ["app.py · st.chat_input", "model switch (side panel)"])
box(370, top - 20, 300, 62, "Preprocessing (regex)", ["src/textnorm.py · URLs, mentions, leetspeak,", "repeated letters, Turkish I/İ, abbreviations"])
arrow(160, top + 11, 190, top + 11)
arrow(340, top + 11, 370, top + 11)

my, mh, mw = top - 150, 86, 168
models = [("Rule baseline", ["lexicon mined from train +", "edit distance ≤ 1", "src/lexicon.py"], LIGHT),
          ("Classical · tf-idf + LR", ["word 1-2 + char 2-5 grams", "scikit-learn · FP64"], LIGHT),
          ("Neural · word2vec + FFN", ["own skip-gram, averaged →", "MLP 200-256-64-2", "PyTorch · FP32"], LIGHT),
          ("Transformer · BERTurk", [BASE_SHORT, "fine-tuned, WordPiece", "PyTorch · FP32"], LIGHT),
          ("Quantized BERTurk", ["dynamic INT8 weights", "ONNX Runtime · INT8", "DEFAULT IN THE APP"], PALE_TEAL)]
for i, (title, lines, fill) in enumerate(models):
    x = 40 + i * (mw + 10)
    box(x, my, mw, mh, title, lines, fill, TEAL if i == 4 else NAVY)
    arrow(x + mw / 2, my + mh + 22, x + mw / 2, my + mh)
c.setStrokeColor(NAVY)
c.setLineWidth(1.3)
c.line(40 + mw / 2, my + mh + 22, 40 + 4 * (mw + 10) + mw / 2, my + mh + 22)
c.line(520, top - 20, 520, my + mh + 22)
text(532, top - 36, "normalized text → selected model (or all five, side by side)", 9.5, color=GREY)

py = my - 112
c.line(40 + mw / 2, my - 18, 40 + 4 * (mw + 10) + mw / 2, my - 18)
for i in range(5):
    x = 40 + i * (mw + 10) + mw / 2
    c.line(x, my, x, my - 18)
arrow(300, my - 18, 300, py + 62)
text(312, my - 36, "P(offensive)", 9.5, color=GREY)
box(190, py, 220, 62, "Post-processing", ["threshold (tuned on validation) →", "label + confidence · mask · stop timer"])
box(440, py, 250, 62, "Streamlit UI (output)", ["chat bubble: label, confidence, ms", "side panel: response-time log · comparison table"])
arrow(410, py + 31, 440, py + 31)
box(720, py, 200, 62, "Where it runs", ["one Python process, CPU only", meta["hardware"].split(",")[0][:38]], colors.white)
text(40, 62, "Offline (train_all.py): OffensEval-TR → train_classical · train_w2v_ffn · train_bert (GPU) → quantize "
     "(ONNX export → INT8) → tune_thresholds (validation) → evaluate (same test set, CPU).", 10, color=GREY, width=880)
c.showPage()

# ------------------------------------------------------------------ 5 · technique map
header("Development · Technique map", "Technique map: which component uses which technique")
table([["Area", "Technique", "Component in the application", "Code"],
       ["B · Classical NLP", "Regular expressions, text normalization", "Preprocessing of every message (leetspeak,\nrepeated letters, Turkish characters)", "src/textnorm.py"],
       ["B · Classical NLP", "Minimum edit distance", "Rule baseline: lexicon matcher that also catches\nwords one edit away from a lexicon entry", "src/lexicon.py"],
       ["B · Classical NLP", "tf-idf (+ logistic regression)", "Classical baseline of the model ladder", "src/train_classical.py"],
       ["B · Classical NLP", "word2vec embeddings (own, skip-gram)", "Input features of the neural network", "src/train_w2v_ffn.py"],
       ["A · Neural networks", "Feedforward net, SGD + backpropagation,\ndropout", "Neural model of the ladder (word2vec average\n→ MLP)", "src/train_w2v_ffn.py"],
       ["C · Transformers", "Fine-tuning BERT (BERTurk) with a task\nhead; WordPiece tokenizer; attention", "Transformer model of the ladder", "src/train_bert.py"],
       ["D · Efficient models", "Quantization (dynamic INT8), ONNX Runtime", "Model served by default in the chat app", "src/quantize.py"],
       ["D · Efficient models", "Benchmarking size, latency, accuracy", "Results table, live latency log in the side panel", "src/evaluate.py, app.py"]],
      48, H - 108, [160, 240, 300, 164], size=10.5, highlight_row=7)
c.showPage()

# ------------------------------------------------------------------ 6 · models
header("Development · Models", "The four models and how each was trained")
b, w = train_info, 425
box(48, 268, w, 150, "1 · Classical baseline – tf-idf + logistic regression", [
    "Features: word 1–2-gram tf-idf + character 2–5-gram tf-idf (sublinear tf).",
    "Classifier: logistic regression, class_weight = balanced.",
    f"C chosen on validation from {{1, 3, 10}} → C = {b['tfidf']['C']:g}.",
    f"Validation macro-F1: {b['tfidf']['val_f1']:.3f}",
    f"Extra rule baseline: {b['lexicon']['words']} lexicon words mined from train, edit distance ≤ 1 "
    f"(val macro-F1 {b['lexicon']['val_f1']:.3f}).",
], size=10.5)
box(487, 268, w, 150, "2 · Neural network – word2vec + feedforward net", [
    f"word2vec: skip-gram, {b['w2v']['dim']} dimensions, trained by us on the training messages only "
    f"({b['w2v']['vocab']:,} words).",
    "Message vector = average of its word vectors.",
    "MLP 200 → 256 → 64 → 2, ReLU, dropout 0.3.",
    "SGD (momentum 0.9, lr 0.05, cosine decay), 40 epochs, class-weighted cross-entropy, backpropagation; "
    "best epoch kept by validation macro-F1.",
    f"Validation macro-F1: {b['w2v']['val_f1']:.3f}",
], size=10.5)
cands = " vs. ".join(f"{c['base'].split('/')[-1].replace('bert-base-turkish-', '')} {c['val_f1']:.3f}"
                     for c in BERT["candidates"])
box(48, 98, w, 150, "3 · Transformer – BERTurk fine-tuned", [
    f"{BERT['base']}: 12 self-attention layers, {BERT['params_m']} M parameters, "
    f"WordPiece vocabulary of {BERT['vocab'] // 1000}k.",
    "New classification head on [CLS]; all layers fine-tuned. AdamW, lr 3e-5, warm-up 10%, batch 32, "
    "max 64 tokens, mixed precision, up to 4 epochs (best epoch by validation).",
    f"Two BERTurk checkpoints fine-tuned on {BERT['device']}; picked on validation macro-F1 ({cands}).",
    f"Validation macro-F1: {BERT['val_f1']:.3f}",
], size=10.5)
box(487, 98, w, 150, "4 · Quantized Transformer – BERTurk INT8", [
    "The fine-tuned model from step 3, exported to ONNX (opset 17).",
    "Post-training dynamic quantization with ONNX Runtime: weights stored as 8-bit integers.",
    "No retraining and no calibration data.",
    f"{mb('bert_fp32')} MB → {mb('bert_int8')} MB.  This is the model the application runs.",
], PALE_TEAL, TEAL, size=10.5)
text(48, 70, "Decision thresholds were also chosen on the validation split (macro-F1): "
     f"tf-idf {thr('tfidf')}, FFN {thr('w2v_ffn')}, BERTurk {thr('bert_fp32')}, BERTurk INT8 {thr('bert_int8')}; "
     "the rule baseline keeps 0.50. The test set was used only for the final numbers.", 10.5, color=GREY, width=870)
c.showPage()

# ------------------------------------------------------------------ 7 · quantization
header("Development · Quantization", "Quantization: meeting the 50 ms budget on a CPU")
y = bullets(48, H - 122, [
    f"Budget we set: p95 ≤ {meta['budget_ms']:.0f} ms per message, CPU only, one message at a time (chat is not batched).",
    "Method: post-training dynamic quantization (onnxruntime.quantization.quantize_dynamic, QInt8). "
    "The float32 weight matrices of the attention and feed-forward layers become int8 + a scale; "
    "activations are quantized on the fly, so no calibration set is needed.",
    "Pipeline: fine-tuned PyTorch model → ONNX export → INT8 ONNX → ONNX Runtime CPU session in the app.",
], width=860)
y = table([["BERTurk variant", "Macro-F1", "Size (MB)", "Mean (ms)", "p95 (ms)", "≤ 50 ms"],
           ["FP32 · PyTorch", f1("bert_fp32"), mb("bert_fp32"), ms("bert_fp32"), p95("bert_fp32"), "yes" if fp32["within_budget"] else "no"],
           ["FP32 · ONNX Runtime", f1("bert_onnx_fp32"), mb("bert_onnx_fp32"), ms("bert_onnx_fp32"), p95("bert_onnx_fp32"), "yes" if onnx32["within_budget"] else "no"],
           ["INT8 · ONNX Runtime (deployed)", f1("bert_int8"), mb("bert_int8"), ms("bert_int8"), p95("bert_int8"), "yes" if int8["within_budget"] else "no"]],
          48, y - 8, [300, 110, 110, 110, 110, 110], size=12, highlight_row=3)
bullets(48, y - 34, [
    f"Trade-off measured: {size_ratio:.1f}× smaller and {speedup:.1f}× faster than the PyTorch FP32 model, "
    f"for a macro-F1 change of {-f1_drop:+.4f} ({fp32['macro_f1']:.4f} → {int8['macro_f1']:.4f}).",
    f"INT8 and FP32 give the same label on {err['int8_fp32_agreement']:.1%} of the test messages.",
    "The ONNX FP32 row separates the two effects: runtime change (PyTorch → ONNX Runtime) vs. quantization itself.",
], width=860, size=13)
c.showPage()

# ------------------------------------------------------------------ 8 · results
header("Development · Results", "Results: four models, one test set")
data = [["Model", "Precision", "Thresh.", "Macro-F1", "Accuracy", "Size (MB)", "Latency" + chr(10) + "mean (ms)", "Latency" + chr(10) + "p95 (ms)"]]
names = {"lexicon": "Rule: lexicon + edit distance", "tfidf": "1 · Classical: tf-idf + log. regression",
         "w2v_ffn": "2 · Neural net: word2vec + FFN", "bert_fp32": "3 · Transformer: BERTurk (PyTorch)",
         "bert_int8": "4 · Quantized Transformer: BERTurk"}
for k in ["lexicon"] + LADDER:
    r = rows[k]
    data.append([names[k], r["precision"], f"{r.get('threshold', 0.5):.2f}", f"{r['macro_f1']:.3f}", f"{r['accuracy']:.3f}", f"{r['size_mb']:.1f}"
                 if r["size_mb"] >= 0.1 else f"{r['size_mb']:.2f}", f"{r['latency_ms_mean']:.2f}", f"{r['latency_ms_p95']:.2f}"])
y = table(data, 48, H - 112, [270, 76, 70, 82, 84, 84, 104, 104], size=11.5, highlight_row=5)
y = bullets(48, y - 30, [
    f"Test set: {meta['test_set']}, {meta['n_test']:,} messages, {meta['offensive_share']:.1%} offensive – "
    "identical for every row, never used for training or tuning.",
    f"Latency: end to end (normalization + tokenization + model) for one message, mean over {meta['latency_runs']} "
    "different test messages after warm-up.",
    f"Hardware: {meta['hardware']}. No GPU is used at inference.",
    "Size: the file(s) the app has to load (tf-idf: vectorizers + weights; FFN: embeddings + MLP; BERT: weights).",
    "Thresholds were chosen on the validation split. The first row is the extra rule baseline required for "
    "Area B; rows 1–4 are the model ladder.",
], width=860, size=12.5)
c.showPage()

# ------------------------------------------------------------------ 9 · conclusion
header("Conclusion", "What the comparison showed")
bullets(48, H - 124, cfg["conclusions"] if cfg.get("conclusions") else [
    f"Quality: BERTurk wins clearly – macro-F1 {f1('bert_int8')} (INT8) / {f1('bert_fp32')} (FP32) vs. {f1('tfidf')} "
    f"(tf-idf) and {f1('w2v_ffn')} (word2vec + FFN). Context matters: many offensive tweets contain no swear word.",
    f"Speed and size: the small models win by orders of magnitude ({ms('tfidf')} ms / {ms('w2v_ffn')} ms per message, "
    f"{mb('tfidf')} / {mb('w2v_ffn')} MB). If only a tiny, very fast model is allowed, tf-idf is the one to pick.",
    f"The rule baseline is the weakest (macro-F1 {f1('lexicon')}): it only fires on known words.",
    f"Quantization costs no quality here: {size_ratio:.1f}× smaller, {speedup:.1f}× faster, macro-F1 "
    f"{f1('bert_int8')} vs. {f1('bert_fp32')} (each with its own validation-tuned threshold; a gap this small is "
    f"within noise). p95 latency {p95('bert_int8')} ms vs. {p95('bert_fp32')} ms for PyTorch FP32 "
    f"({meta['budget_ms']:.0f} ms budget).",
    f"Model selection on validation: the 32k-vocabulary BERTurk beat the larger 128k one, and a validation gain "
    f"does not always carry over to the test set – tuning must never look at the test set.",
    f"The neural net ({f1('w2v_ffn')}) does not beat tf-idf ({f1('tfidf')}): averaging word2vec vectors loses word order "
    "and sub-word evidence that character n-grams keep – "
    "a neural network is not automatically better than a strong classical baseline.",
], width=860, size=13, gap=9)
c.showPage()

# ------------------------------------------------------------------ 10 · limitations
header("Conclusion", "Limitations, failure examples and next steps")
y = text(48, H - 118, "Failure examples of the deployed INT8 model (real outputs)", 14, "B", TEAL)
data = [["Message", "Gold", "Predicted", "Why it fails"]]
for e in cfg["failure_examples"]:
    data.append([e["text"], e["gold"], e["pred"], e["why"]])
y = table(data, 48, y - 2, [330, 80, 130, 324], size=10)
y = text(48, y - 26, "Limitations", 14, "B", TEAL)
y = bullets(48, y - 4, cfg["limitations"], size=12, width=860, gap=4)
y = text(48, y - 8, "What we would do next", 14, "B", TEAL)
bullets(48, y - 4, cfg["next_steps"], size=12, width=860, gap=4)
c.showPage()

# ------------------------------------------------------------------ 11 · AI usage
header("Conclusion", "AI usage statement")
y = bullets(48, H - 124, cfg["ai_usage"], width=860, size=14, gap=12)
text(48, 70, "Data: Çöltekin, Ç. (2020). A Corpus of Turkish Offensive Language on Social Media. LREC 2020.  "
     f"Model: {BERT['base']} (BERTurk, Stefan Schweter).", 10, color=GREY, width=860)
c.showPage()
c.save()
print(f"wrote slides/presentation.pdf ({page} slides)")
