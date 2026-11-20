"""Real-time chat moderation demo (Use Case 4).

    streamlit run app.py

Every message typed into the chat is normalized, classified (clean / offensive) with a
confidence score, masked if flagged, and its response time is logged in the side panel.
The default model is the INT8-quantized BERTurk running on CPU with ONNX Runtime.
"""
import re
import time

import pandas as pd
import streamlit as st

from src.models import REGISTRY
from src.textnorm import normalize

BUDGET_MS = 50.0
DEFAULT = "bert_int8"
APP_MODELS = ["bert_int8", "bert_fp32", "w2v_ffn", "tfidf", "lexicon"]

st.set_page_config(page_title="Chat Moderation", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load(key: str):
    try:
        model = REGISTRY[key]()
    except (FileNotFoundError, OSError) as exc:
        st.error(f"Model files for **{REGISTRY[key].name}** were not found in `models/`.\n\n"
                 "Run `python download_models.py` (or `python train_all.py`) and restart the app.")
        st.caption(str(exc))
        st.stop()
    model.prob("ısınma mesajı")            # warm-up so the first real message is not slow
    return model


def classify(key: str, text: str):
    model = load(key)
    t0 = time.perf_counter()
    p = model.prob(text)
    ms = (time.perf_counter() - t0) * 1000
    return ("offensive" if p >= 0.5 else "clean"), (p if p >= 0.5 else 1 - p), ms


def mask(text: str) -> str:
    return re.sub(r"\w+", lambda m: m.group(0)[0] + "*" * (len(m.group(0)) - 1), text)


if "chat" not in st.session_state:
    st.session_state.chat, st.session_state.log = [], []

# ---------------------------------------------------------------- side panel
with st.sidebar:
    st.header("Model")
    key = st.selectbox("Active model", APP_MODELS, index=0,
                       format_func=lambda k: f"{REGISTRY[k].name} [{REGISTRY[k].precision}]")
    compare = st.toggle("Show all models side by side", value=True)
    st.caption(f"Latency budget: {BUDGET_MS:.0f} ms per message, CPU only.")
    for k in (APP_MODELS if compare else [key]):     # load before the first message arrives
        load(k)

    st.header("Response time log")
    if st.session_state.log:
        log = pd.DataFrame(st.session_state.log)
        c1, c2 = st.columns(2)
        c1.metric("Last (ms)", f"{log['ms'].iloc[-1]:.1f}")
        c2.metric("Mean (ms)", f"{log['ms'].mean():.1f}")
        st.dataframe(log.iloc[::-1], hide_index=True, use_container_width=True,
                     column_config={"ms": st.column_config.NumberColumn("ms", format="%.1f")})
    else:
        st.caption("No messages yet.")
    if st.button("Clear chat"):
        st.session_state.chat, st.session_state.log = [], []
        st.rerun()

# ---------------------------------------------------------------- chat
st.title("🛡️ Real-Time Chat Moderation")
st.caption("Turkish offensive-language filter · classical baseline → neural net → BERTurk → quantized BERTurk")

for item in st.session_state.chat:
    with st.chat_message("user"):
        if item["label"] == "offensive":
            st.markdown(f"`{mask(item['text'])}`")
            st.error(f"🚫 offensive · confidence {item['conf']:.0%} · {item['ms']:.1f} ms · message masked")
            with st.expander("Show original (moderator view)"):
                st.write(item["text"])
        else:
            st.write(item["text"])
            st.success(f"✅ clean · confidence {item['conf']:.0%} · {item['ms']:.1f} ms")
        if item["compare"] is not None:
            with st.expander("All models on this message", expanded=True):
                st.caption(f"After regex normalization: `{normalize(item['text'])}`")
                st.dataframe(item["compare"], hide_index=True, use_container_width=True)

text = st.chat_input("Type a chat message (Turkish)...")
if text:
    label, conf, ms = classify(key, text)
    table = None
    if compare:
        rows = []
        for k in APP_MODELS:
            l, c, m = (label, conf, ms) if k == key else classify(k, text)
            rows.append({"model": f"{REGISTRY[k].name} [{REGISTRY[k].precision}]", "label": l,
                         "confidence": f"{c:.0%}", "ms": round(m, 1),
                         "≤ 50 ms": "yes" if m <= BUDGET_MS else "no"})
        table = pd.DataFrame(rows)
    st.session_state.chat.append({"text": text, "label": label, "conf": conf, "ms": ms, "compare": table})
    st.session_state.log.append({"#": len(st.session_state.log) + 1, "model": key, "label": label, "ms": ms})
    st.rerun()
