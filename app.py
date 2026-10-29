"""
Review Sentiment Studio
-----------------------
Streamlit front-end for the 3-class (Negative / Neutral / Positive) review
sentiment model: TF-IDF (5000 features) -> PyTorch MLP (5000-512-256-3).

Run from the project root:
    streamlit run app.py
"""

import html
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_mlp.pth"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

LABELS = ["Negative", "Neutral", "Positive"]  # must match label_map in notebook 04
N_FEATURES = 5000

META = {
    "Negative": {"color": "#ff5d73", "emoji": "😞", "tag": "Unhappy customer"},
    "Neutral": {"color": "#ffb84d", "emoji": "😐", "tag": "Mixed or lukewarm"},
    "Positive": {"color": "#2fe6a7", "emoji": "😍", "tag": "Happy customer"},
}

EXAMPLES = {
    "Loved it": "These are the BEST treats for training and rewarding my dog. "
    "He goes crazy for them and the ingredients are great. Will buy again!",
    "Disappointed": "I'm disappointed with the flavor. The chocolate tasted stale and "
    "waxy, and the package arrived half crushed. Not worth the price.",
    "It's okay": "The product is fine. It does what it says, nothing special though. "
    "Shipping was average and the taste is just okay.",
    "Mixed": "Great packaging and fast delivery, but the taste was bland and "
    "the portion is way too small for the price.",
}

st.set_page_config(
    page_title="Review Sentiment Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

:root {
  --bg: #0b0e1a;
  --panel: rgba(255,255,255,0.045);
  --panel-border: rgba(255,255,255,0.09);
  --text: #eef0f8;
  --muted: #98a0b8;
  --accent: #7c8cff;
  --accent2: #b36bff;
}

html, body, [class*="css"], .stApp {
  font-family: 'Inter', sans-serif !important;
  color: var(--text);
}

.stApp {
  background:
    radial-gradient(900px 500px at 8% -10%, rgba(124,140,255,0.20), transparent 60%),
    radial-gradient(800px 500px at 100% 0%, rgba(179,107,255,0.16), transparent 55%),
    radial-gradient(700px 500px at 50% 110%, rgba(47,230,167,0.08), transparent 60%),
    var(--bg);
}

/* hide default chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }

/* hero */
.hero { padding: 6px 0 22px 0; }
.badge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 14px; border-radius: 999px;
  background: rgba(124,140,255,0.12); border: 1px solid rgba(124,140,255,0.35);
  color: #b9c1ff; font-size: 12.5px; font-weight: 600; letter-spacing: .4px;
}
.badge .dot { width: 7px; height: 7px; border-radius: 50%; background: #2fe6a7;
  box-shadow: 0 0 10px #2fe6a7; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
.hero h1 {
  font-size: 46px; line-height: 1.08; font-weight: 800; margin: 16px 0 10px 0;
  letter-spacing: -1.2px;
}
.grad {
  background: linear-gradient(90deg, #8fa0ff 0%, #c58bff 50%, #4ff0b8 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: var(--muted); font-size: 16.5px; max-width: 700px; margin: 0; line-height: 1.6; }

/* cards */
.card {
  background: var(--panel); border: 1px solid var(--panel-border);
  border-radius: 20px; padding: 24px 26px;
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 10px 40px rgba(0,0,0,0.28);
}
.card-title { font-size: 12px; font-weight: 700; letter-spacing: 1.4px;
  text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }

/* result */
.verdict-row { display: flex; align-items: center; gap: 22px; margin-bottom: 22px; }
.donut {
  width: 112px; height: 112px; border-radius: 50%; flex: none;
  display: flex; align-items: center; justify-content: center;
}
.donut-inner {
  width: 84px; height: 84px; border-radius: 50%; background: #121528;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.donut-inner b { font-size: 22px; font-weight: 800; letter-spacing: -.5px; }
.donut-inner span { font-size: 10px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }
.verdict-label { font-size: 32px; font-weight: 800; letter-spacing: -.8px; line-height: 1.1; }
.verdict-sub { color: var(--muted); font-size: 14px; margin-top: 6px; }
.warn {
  display: inline-block; margin-top: 10px; padding: 5px 11px; border-radius: 8px;
  background: rgba(255,184,77,0.12); border: 1px solid rgba(255,184,77,0.35);
  color: #ffcf85; font-size: 12.5px; font-weight: 500;
}

.bar-row { display: flex; align-items: center; gap: 12px; margin: 11px 0; }
.bar-name { width: 78px; font-size: 13.5px; font-weight: 600; }
.bar-track { flex: 1; height: 10px; border-radius: 999px; background: rgba(255,255,255,0.07); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; transition: width .8s ease; }
.bar-val { width: 58px; text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #cfd4e8; }

/* word chips */
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.chip {
  padding: 6px 12px; border-radius: 10px; font-size: 13px; font-weight: 600;
  border: 1px solid; display: inline-flex; gap: 7px; align-items: center;
}
.chip i { font-style: normal; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; opacity: .85; }
.chip.against { color: #b6bdd3; background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.14); }
.legend { color: var(--muted); font-size: 12.5px; margin-top: 12px; line-height: 1.55; }

/* empty state */
.empty { text-align: center; padding: 46px 10px; color: var(--muted); }
.empty .big { font-size: 46px; margin-bottom: 10px; }
.empty b { color: var(--text); display: block; font-size: 17px; margin-bottom: 6px; }

/* stat tiles */
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 6px 0 20px 0; }
.tile { background: var(--panel); border: 1px solid var(--panel-border); border-radius: 16px; padding: 16px 18px; }
.tile .k { font-size: 11.5px; letter-spacing: 1.2px; text-transform: uppercase; color: var(--muted); font-weight: 700; }
.tile .v { font-size: 28px; font-weight: 800; letter-spacing: -.8px; margin-top: 4px; }

/* history */
.hist { display: flex; align-items: center; gap: 12px; padding: 11px 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
.hist:last-child { border-bottom: none; }
.pill { padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; flex: none; border: 1px solid; }
.hist-text { color: #c5cadd; font-size: 13.5px; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-conf { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--muted); }

/* about table */
table.metrics { width: 100%; border-collapse: collapse; font-size: 14px; }
table.metrics th { text-align: right; color: var(--muted); font-weight: 600; font-size: 12px;
  letter-spacing: 1px; text-transform: uppercase; padding: 10px 12px; border-bottom: 1px solid var(--panel-border); }
table.metrics th:first-child, table.metrics td:first-child { text-align: left; }
table.metrics td { text-align: right; padding: 11px 12px; border-bottom: 1px solid rgba(255,255,255,0.05);
  font-family: 'JetBrains Mono', monospace; }
table.metrics td:first-child { font-family: 'Inter', sans-serif; font-weight: 500; }
table.metrics tr:last-child td { border-bottom: none; }
.flow { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 4px; }
.flow .node { padding: 9px 14px; border-radius: 12px; background: rgba(124,140,255,0.12);
  border: 1px solid rgba(124,140,255,0.3); font-size: 13.5px; font-weight: 600; }
.flow .arrow { color: var(--muted); }

/* streamlit widgets */
.stTextArea textarea {
  background: rgba(255,255,255,0.04) !important; border: 1px solid var(--panel-border) !important;
  border-radius: 14px !important; color: var(--text) !important; font-size: 15px !important;
  line-height: 1.6 !important; padding: 16px !important;
}
.stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(124,140,255,0.22) !important; }
.stTextArea label, .stSelectbox label, .stFileUploader label, .stRadio label { color: var(--muted) !important; font-weight: 600 !important; }

.stButton > button, .stDownloadButton > button {
  border-radius: 12px; font-weight: 600; border: 1px solid var(--panel-border);
  background: rgba(255,255,255,0.05); color: var(--text); transition: all .18s ease; padding: 0.55rem 1rem;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--accent); background: rgba(124,140,255,0.14); transform: translateY(-1px); color: #fff;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: linear-gradient(90deg, #6b7bff, #a45cff); border: none; color: #fff;
  box-shadow: 0 8px 24px rgba(124,108,255,0.38);
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.1); transform: translateY(-1px); }

.stTabs [data-baseweb="tab-list"] { gap: 6px; background: rgba(255,255,255,0.04); padding: 5px;
  border-radius: 14px; border: 1px solid var(--panel-border); width: fit-content; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 8px 20px; height: auto; color: var(--muted); font-weight: 600; }
.stTabs [aria-selected="true"] { background: linear-gradient(90deg, rgba(107,123,255,0.28), rgba(164,92,255,0.28)); color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

[data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,0.03); border: 1.5px dashed rgba(124,140,255,0.4); border-radius: 16px; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--panel-border); }

@media (max-width: 800px) {
  .hero h1 { font-size: 32px; }
  .tiles { grid-template-columns: repeat(2, 1fr); }
  .verdict-row { flex-direction: column; align-items: flex-start; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def build_model() -> nn.Sequential:
    """Same architecture as notebook 04 so the state_dict keys match."""
    return nn.Sequential(
        nn.Linear(N_FEATURES, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 3),
    )


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = build_model()
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    return vectorizer, model, feature_names


missing = [p.name for p in (MODEL_PATH, VECTORIZER_PATH) if not p.exists()]
if missing:
    st.error(
        f"Missing file(s) in the `models/` folder: {', '.join(missing)}. "
        "Run notebook 04 first, and start the app from the project root."
    )
    st.stop()

try:
    vectorizer, model, feature_names = load_artifacts()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the model: {exc}")
    st.stop()

if len(feature_names) != N_FEATURES:
    st.error(
        f"The TF-IDF vectorizer has {len(feature_names)} features but the model expects "
        f"{N_FEATURES}. These files come from different training runs."
    )
    st.stop()


# ----------------------------------------------------------------------------
# Inference helpers
# ----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Light cleaning: strip HTML tags / URLs and collapse whitespace.
    TfidfVectorizer already lowercases and tokenises."""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict_proba(texts, batch_size: int = 1024) -> np.ndarray:
    X = vectorizer.transform([clean_text(t) for t in texts])
    chunks = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i : i + batch_size].toarray().astype(np.float32))
            chunks.append(torch.softmax(model(xb), dim=1).numpy())
    return np.vstack(chunks)


def explain(text: str, class_idx: int, top_k: int = 10):
    """Occlusion-based word importance: zero out one word's TF-IDF value at a
    time and measure how much the predicted-class probability drops."""
    x = vectorizer.transform([clean_text(text)])
    idx = x.indices
    if len(idx) == 0:
        return []
    base = x.toarray().astype(np.float32)
    variants = np.repeat(base, len(idx), axis=0)
    for row, col in enumerate(idx):
        variants[row, col] = 0.0
    with torch.no_grad():
        p_base = torch.softmax(model(torch.from_numpy(base)), dim=1)[0, class_idx].item()
        p_var = torch.softmax(model(torch.from_numpy(variants)), dim=1)[:, class_idx].numpy()
    delta = p_base - p_var  # > 0: the word supports the prediction
    order = np.argsort(-np.abs(delta))[:top_k]
    return [(feature_names[idx[i]], float(delta[i])) for i in order]


# ----------------------------------------------------------------------------
# Rendering helpers (HTML strings must not be indented, or markdown treats
# them as code blocks)
# ----------------------------------------------------------------------------
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def result_card(probs: np.ndarray) -> str:
    i = int(np.argmax(probs))
    label = LABELS[i]
    meta = META[label]
    conf = float(probs[i]) * 100
    color = meta["color"]

    warn = ""
    if conf < 55:
        warn = '<div class="warn">⚠ Low confidence: this review has a mixed or ambiguous signal</div>'
    elif label == "Neutral":
        warn = '<div class="warn">Neutral is the hardest class for this model (test F1 ≈ 0.30)</div>'

    bars = ""
    for name, p in zip(LABELS, probs):
        c = META[name]["color"]
        bars += (
            f'<div class="bar-row"><div class="bar-name" style="color:{c}">{name}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{p*100:.1f}%;'
            f'background:linear-gradient(90deg,{hex_to_rgba(c,0.55)},{c})"></div></div>'
            f'<div class="bar-val">{p*100:.1f}%</div></div>'
        )

    return (
        '<div class="card"><div class="card-title">Prediction</div>'
        '<div class="verdict-row">'
        f'<div class="donut" style="background:conic-gradient({color} {conf:.1f}%,rgba(255,255,255,0.08) 0)">'
        f'<div class="donut-inner"><b>{conf:.0f}%</b><span>confidence</span></div></div>'
        f'<div><div class="verdict-label" style="color:{color}">{meta["emoji"]} {label}</div>'
        f'<div class="verdict-sub">{meta["tag"]}</div>{warn}</div></div>'
        f"{bars}</div>"
    )


def words_card(words, label: str) -> str:
    color = META[label]["color"]
    if not words:
        body = '<div class="legend">None of the words in this text are in the model vocabulary.</div>'
    else:
        chips = ""
        for w, d in words:
            sign = "+" if d >= 0 else "−"
            if d >= 0:
                style = (
                    f"color:{color};background:{hex_to_rgba(color,0.12)};"
                    f"border-color:{hex_to_rgba(color,0.4)}"
                )
                chips += f'<span class="chip" style="{style}">{html.escape(w)}<i>{sign}{abs(d)*100:.1f}</i></span>'
            else:
                chips += f'<span class="chip against">{html.escape(w)}<i>{sign}{abs(d)*100:.1f}</i></span>'
        body = (
            f'<div class="chips">{chips}</div>'
            f'<div class="legend">Coloured words push the model <b>toward {label}</b>; grey words pull '
            "it away. Numbers show the change in probability (percentage points) when the word is removed.</div>"
        )
    return f'<div class="card" style="margin-top:18px"><div class="card-title">Why this prediction</div>{body}</div>'


def history_card(items) -> str:
    rows = ""
    for it in items:
        c = META[it["label"]]["color"]
        pill = (
            f"color:{c};background:{hex_to_rgba(c,0.12)};border-color:{hex_to_rgba(c,0.4)}"
        )
        rows += (
            f'<div class="hist"><span class="pill" style="{pill}">{it["label"]}</span>'
            f'<span class="hist-text">{html.escape(it["text"])}</span>'
            f'<span class="hist-conf">{it["conf"]:.0f}%</span></div>'
        )
    return f'<div class="card" style="margin-top:18px"><div class="card-title">Recent analyses</div>{rows}</div>'


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
st.session_state.setdefault("review_text", "")
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_result", None)


def set_example(key: str):
    st.session_state["review_text"] = EXAMPLES[key]


def clear_text():
    st.session_state["review_text"] = ""
    st.session_state["last_result"] = None


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><span class="badge"><span class="dot"></span>MODEL ONLINE · PyTorch MLP + TF-IDF</span>'
    '<h1>Understand what customers <span class="grad">really feel.</span></h1>'
    "<p>Paste a product review and get an instant Negative / Neutral / Positive prediction with "
    "class probabilities and the exact words that drove the decision. Or score thousands of reviews at once.</p></div>",
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_about = st.tabs(["✦  Analyze review", "▤  Batch analysis", "ⓘ  About the model"])

# ----------------------------------------------------------------------------
# Tab 1: single review
# ----------------------------------------------------------------------------
with tab_single:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<div class="card-title" style="margin-top:6px">Try an example</div>', unsafe_allow_html=True)
        ex_cols = st.columns(len(EXAMPLES))
        for col, key in zip(ex_cols, EXAMPLES):
            col.button(key, key=f"ex_{key}", on_click=set_example, args=(key,), use_container_width=True)

        st.text_area(
            "Review text",
            key="review_text",
            height=230,
            placeholder="Paste or type a product review here...",
        )

        b1, b2 = st.columns([2, 1])
        analyze = b1.button("Analyze sentiment", type="primary", use_container_width=True)
        b2.button("Clear", on_click=clear_text, use_container_width=True)

        if analyze:
            text = st.session_state["review_text"].strip()
            if not text:
                st.warning("Please enter a review first.")
            elif len(vectorizer.transform([clean_text(text)]).indices) == 0:
                st.warning(
                    "None of the words in this text are in the model vocabulary, so a prediction "
                    "would be meaningless. Try a longer, English product review."
                )
            else:
                probs = predict_proba([text])[0]
                i = int(np.argmax(probs))
                st.session_state["last_result"] = {
                    "text": text,
                    "probs": probs,
                    "words": explain(text, i),
                }
                st.session_state["history"].insert(
                    0, {"text": text, "label": LABELS[i], "conf": float(probs[i]) * 100}
                )
                st.session_state["history"] = st.session_state["history"][:6]

    with right:
        res = st.session_state["last_result"]
        if res is None:
            st.markdown(
                '<div class="card"><div class="empty"><div class="big">🧠</div>'
                "<b>Your analysis will appear here</b>"
                "Pick an example or paste your own review, then hit Analyze.</div></div>",
                unsafe_allow_html=True,
            )
        else:
            label = LABELS[int(np.argmax(res["probs"]))]
            st.markdown(result_card(res["probs"]), unsafe_allow_html=True)
            st.markdown(words_card(res["words"], label), unsafe_allow_html=True)

    if st.session_state["history"]:
        st.markdown(history_card(st.session_state["history"]), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Tab 2: batch
# ----------------------------------------------------------------------------
with tab_batch:
    st.markdown(
        '<div class="card-title" style="margin-top:6px">Score many reviews at once</div>',
        unsafe_allow_html=True,
    )
    mode = st.radio("Input", ["Upload CSV", "Paste (one review per line)"], horizontal=True, label_visibility="collapsed")

    df_in = None
    text_col = None

    if mode == "Upload CSV":
        up = st.file_uploader("CSV file with a column of review text", type=["csv"])
        if up is not None:
            try:
                df_in = pd.read_csv(up)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read that CSV: {exc}")
            if df_in is not None:
                obj_cols = [c for c in df_in.columns if df_in[c].dtype == object]
                if not obj_cols:
                    st.error("No text columns found in this file.")
                    df_in = None
                else:
                    preferred = next(
                        (c for c in obj_cols if c.lower() in ("text", "review", "reviews", "review_text", "content")),
                        obj_cols[0],
                    )
                    text_col = st.selectbox("Which column has the review text?", obj_cols, index=obj_cols.index(preferred))
    else:
        pasted = st.text_area("Reviews", height=180, placeholder="One review per line...", key="batch_paste")
        lines = [ln.strip() for ln in pasted.splitlines() if ln.strip()]
        if lines:
            df_in = pd.DataFrame({"Text": lines})
            text_col = "Text"

    if df_in is not None and text_col is not None:
        n = len(df_in)
        MAX_ROWS = 50_000
        st.caption(f"{n:,} rows ready" + (f" · only the first {MAX_ROWS:,} will be scored" if n > MAX_ROWS else ""))

        if st.button("Run batch analysis", type="primary"):
            work = df_in.head(MAX_ROWS).copy()
            work[text_col] = work[text_col].fillna("").astype(str)
            progress = st.progress(0.0, text="Scoring reviews...")
            parts = []
            step = 2000
            texts = work[text_col].tolist()
            for s in range(0, len(texts), step):
                parts.append(predict_proba(texts[s : s + step]))
                progress.progress(min((s + step) / len(texts), 1.0), text="Scoring reviews...")
            progress.empty()
            P = np.vstack(parts)
            out = work.copy()
            out["Predicted sentiment"] = [LABELS[i] for i in P.argmax(axis=1)]
            out["Confidence"] = (P.max(axis=1) * 100).round(1)
            for k, name in enumerate(LABELS):
                out[f"P({name})"] = (P[:, k] * 100).round(1)
            st.session_state["batch_out"] = out

    out = st.session_state.get("batch_out")
    if out is not None:
        counts = out["Predicted sentiment"].value_counts()
        total = len(out)
        tiles = f'<div class="tile"><div class="k">Reviews scored</div><div class="v">{total:,}</div></div>'
        for name in LABELS:
            c = int(counts.get(name, 0))
            tiles += (
                f'<div class="tile"><div class="k" style="color:{META[name]["color"]}">{name}</div>'
                f'<div class="v">{c:,} <span style="font-size:14px;color:#98a0b8;font-weight:600">'
                f"({c/total*100:.0f}%)</span></div></div>"
            )
        st.markdown(f'<div class="tiles">{tiles}</div>', unsafe_allow_html=True)

        bars = ""
        for name in LABELS:
            share = counts.get(name, 0) / total * 100
            c = META[name]["color"]
            bars += (
                f'<div class="bar-row"><div class="bar-name" style="color:{c}">{name}</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{share:.1f}%;'
                f'background:linear-gradient(90deg,{hex_to_rgba(c,0.55)},{c})"></div></div>'
                f'<div class="bar-val">{share:.1f}%</div></div>'
            )
        st.markdown(
            f'<div class="card"><div class="card-title">Sentiment distribution</div>{bars}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        flt = st.multiselect("Filter by sentiment", LABELS, default=LABELS)
        view = out[out["Predicted sentiment"].isin(flt)]
        st.dataframe(view, use_container_width=True, height=360)
        st.download_button(
            "Download results as CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_predictions.csv",
            mime="text/csv",
            type="primary",
        )

# ----------------------------------------------------------------------------
# Tab 3: about
# ----------------------------------------------------------------------------
with tab_about:
    a, b = st.columns([1.1, 1], gap="large")
    with a:
        st.markdown(
            '<div class="card"><div class="card-title">Pipeline</div><div class="flow">'
            '<span class="node">Review text</span><span class="arrow">→</span>'
            '<span class="node">Clean</span><span class="arrow">→</span>'
            '<span class="node">TF-IDF · 5,000 features</span><span class="arrow">→</span>'
            '<span class="node">MLP 512 → 256</span><span class="arrow">→</span>'
            '<span class="node">Softmax · 3 classes</span></div>'
            '<div class="legend" style="margin-top:16px">Trained on ~393k product reviews. The Positive class was '
            "down-sampled to 60k for training to soften the class imbalance, and the loss was class-weighted. "
            "Splits are 60 / 20 / 20 (train / validation / test), stratified.</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card" style="margin-top:18px"><div class="card-title">Test-set results (MLP)</div>'
            '<table class="metrics"><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr>'
            '<tr><td style="color:#ff5d73">Negative</td><td>0.58</td><td>0.72</td><td>0.64</td></tr>'
            '<tr><td style="color:#ffb84d">Neutral</td><td>0.22</td><td>0.48</td><td>0.30</td></tr>'
            '<tr><td style="color:#2fe6a7">Positive</td><td>0.96</td><td>0.80</td><td>0.87</td></tr>'
            '<tr><td><b>Overall</b></td><td>acc 0.77</td><td>macro 0.67</td><td>macro 0.61</td></tr></table></div>',
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            '<div class="card"><div class="card-title">Model comparison (validation)</div>'
            '<table class="metrics"><tr><th>Metric</th><th>LogReg</th><th>MLP</th></tr>'
            "<tr><td>Accuracy</td><td>0.86</td><td>0.82</td></tr>"
            "<tr><td>Macro F1</td><td>0.61</td><td>0.63</td></tr>"
            "<tr><td>Neutral F1</td><td>0.22</td><td>0.31</td></tr>"
            "<tr><td>Positive F1</td><td>0.93</td><td>0.91</td></tr></table></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card" style="margin-top:18px"><div class="card-title">Good to know</div>'
            '<div class="legend" style="margin-top:0">• Neutral is the hardest class; reviews that mix praise and '
            "complaints are often uncertain.<br>• TF-IDF ignores word order, so sarcasm and negation "
            "(“not good”) can fool the model.<br>• The model was trained on food-product reviews and works best "
            "on similar English text.</div></div>",
            unsafe_allow_html=True,
        )