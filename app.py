"""
CelebMatch – Celebrity Look-Alike Finder
Streamlit Application

Usage:
    streamlit run app.py
"""

import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CelebMatch – Celebrity Look-Alike Finder",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
}

/* ── Hero title ── */
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #f093fb, #f5576c, #fda085);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0;
    letter-spacing: -1px;
}
.hero-sub {
    text-align: center;
    color: rgba(255,255,255,0.55);
    font-size: 1rem;
    margin-top: 4px;
    margin-bottom: 28px;
}

/* ── Cards ── */
.glass-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

/* ── Result boxes ── */
.result-name {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f093fb, #f5576c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
.result-confidence {
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    color: #43e97b;
}
.result-label {
    text-align: center;
    color: rgba(255,255,255,0.5);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}

/* ── Metric boxes ── */
.metric-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-val {
    font-size: 2rem;
    font-weight: 800;
    color: #f093fb;
}
.metric-lbl {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Divider ── */
hr {border-color: rgba(255,255,255,0.1);}

/* ── Button overrides ── */
.stButton > button {
    border-radius: 10px;
    background: linear-gradient(90deg, #f093fb, #f5576c);
    border: none;
    color: white;
    font-weight: 700;
    padding: 0.45rem 1.4rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Radio ── */
.stRadio > label { color: rgba(255,255,255,0.7); }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
    color: rgba(255,255,255,0.6);
    font-weight: 600;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #f093fb33, #f5576c33);
    border-bottom: 2px solid #f5576c;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ─── Helper: Load assets ──────────────────────────────────────────────────────
DATASET_PATH = "Celebrity Faces Dataset"
MODEL_PATH   = "celebrity_model_best.h5"  # Best checkpoint (highest val accuracy)
INFO_PATH    = "model_info.json"


@st.cache_resource(show_spinner="🔄 Loading model …")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_model_info():
    with open(INFO_PATH, "r") as f:
        return json.load(f)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize, normalize, and expand dims for model input."""
    img = img.convert("RGB")
    img = img.resize((128, 128))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def get_reference_image(celebrity_name: str) -> Image.Image | None:
    """Return the first available image from a celebrity's folder."""
    folder = os.path.join(DATASET_PATH, celebrity_name)
    if os.path.isdir(folder):
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            return Image.open(os.path.join(folder, files[0])).convert("RGB")
    return None


def confidence_color(conf: float) -> str:
    if conf >= 0.75:
        return "#43e97b"
    elif conf >= 0.50:
        return "#f9c74f"
    else:
        return "#f5576c"


# ─── Check files exist ────────────────────────────────────────────────────────
files_ready = os.path.exists(MODEL_PATH) and os.path.exists(INFO_PATH)

# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎬 CelebMatch</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Celebrity Look-Alike Neural Network · 13 Celebrities · CNN</div>',
    unsafe_allow_html=True,
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ About")
    st.markdown("""
    **CelebMatch** uses a Convolutional Neural Network trained on 13 celebrity faces to find your closest look-alike.
    
    **Celebrities:**
    """)
    celebs = sorted(os.listdir(DATASET_PATH)) if os.path.isdir(DATASET_PATH) else []
    for c in celebs:
        st.markdown(f"  • {c}")

    st.markdown("---")
    st.markdown("**Model:** Custom CNN · TensorFlow/Keras")
    st.markdown("**Input:** 128 × 128 px")
    st.markdown("**Classes:** 13 celebrities")

    if files_ready:
        info = load_model_info()
        ep = info["model_params"]["epochs_trained"]
        va = info["evaluation"]["val_accuracy"] * 100
        st.markdown(f"**Epochs trained:** {ep}")
        st.markdown(f"**Val Accuracy:** `{va:.1f}%`")
    else:
        st.warning("Model not trained yet.\nRun `python train_model.py` first.")

# ─── Main Content ─────────────────────────────────────────────────────────────
if not files_ready:
    st.error(
        "⚠️ **Model files not found.**  "
        "Please run **`python train_model.py`** to train the model first, "
        "then relaunch this app."
    )
    st.stop()

model     = load_model()
info      = load_model_info()
CLASS_NAMES = info["class_names"]

tab1, tab2 = st.tabs(["🔍 Find My Look-Alike", "🧠 Model & Neural Network Info"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Inference
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Choose Input Method")
    method = st.radio(
        "",
        ["📁 Upload an Image", "📷 Use Webcam"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image_input = None

    if method == "📁 Upload an Image":
        uploaded = st.file_uploader(
            "Upload a face photo (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded:
            image_input = Image.open(uploaded).convert("RGB")

    else:
        cam = st.camera_input("📸 Capture your face")
        if cam:
            image_input = Image.open(cam).convert("RGB")

    if image_input:
        st.markdown("---")
        col_in, col_vs, col_out = st.columns([1, 0.1, 1])

        with col_in:
            st.markdown('<div class="result-label">Your Photo</div>', unsafe_allow_html=True)
            st.image(image_input, use_container_width=True)

        with col_vs:
            st.markdown(
                "<div style='display:flex;align-items:center;justify-content:center;"
                "height:100%;font-size:2rem;color:rgba(255,255,255,0.3);'>⟷</div>",
                unsafe_allow_html=True,
            )

        # ── Prediction ──────────────────────────────────────────────────────
        with st.spinner("🔮 Analyzing facial features …"):
            processed    = preprocess_image(image_input)
            predictions  = model.predict(processed, verbose=0)[0]
            top_idx      = int(np.argmax(predictions))
            top_name     = CLASS_NAMES[top_idx]
            top_conf     = float(predictions[top_idx])

        ref_img = get_reference_image(top_name)

        with col_out:
            st.markdown('<div class="result-label">Your Celebrity Look-Alike</div>', unsafe_allow_html=True)
            if ref_img:
                st.image(ref_img, use_container_width=True)

        # ── Result Banner ────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            st.markdown(
                f'<div class="glass-card">'
                f'<div class="result-label">You look like</div>'
                f'<div class="result-name">{top_name}</div>'
                f'<br>'
                f'<div class="result-label">Confidence</div>'
                f'<div class="result-confidence" style="color:{confidence_color(top_conf)}">'
                f'{top_conf * 100:.1f}%'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Probability Breakdown ────────────────────────────────────────────
        with st.expander("📊 Full Probability Breakdown"):
            sorted_preds = sorted(
                zip(CLASS_NAMES, predictions), key=lambda x: x[1], reverse=True
            )
            for name, prob in sorted_preds:
                bar_color = "#f093fb" if name == top_name else "#555"
                st.markdown(
                    f"**{name}** — {prob*100:.1f}%",
                    unsafe_allow_html=False,
                )
                st.progress(float(prob))

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Model Info
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    params = info["model_params"]
    hist   = info["training_history"]
    evl    = info["evaluation"]

    st.markdown("### 📐 Neural Network Architecture")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    arch_layers = params.get("architecture", [])
    for i, layer in enumerate(arch_layers):
        st.markdown(f"`{i+1}.` {layer}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### ⚙️ Hyperparameters & Training Setup")
    p1, p2, p3, p4 = st.columns(4)
    meta_items = [
        (p1, "Epochs Trained", str(params["epochs_trained"])),
        (p2, "Batch Size",     str(params["batch_size"])),
        (p3, "Image Size",     f"{params['img_size'][0]} × {params['img_size'][1]}"),
        (p4, "Optimizer",      params["optimizer"]),
    ]
    for col, lbl, val in meta_items:
        with col:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-val">{val}</div>'
                f'<div class="metric-lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("### 🏆 Final Evaluation Metrics")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val">{evl["val_accuracy"]*100:.1f}%</div>'
            f'<div class="metric-lbl">Validation Accuracy</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with e2:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val">{evl["val_loss"]:.4f}</div>'
            f'<div class="metric-lbl">Validation Loss</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with e3:
        best_acc = max(hist["val_accuracy"]) * 100
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val">{best_acc:.1f}%</div>'
            f'<div class="metric-lbl">Best Val Accuracy (any epoch)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 📈 Training Curves")
    ch1, ch2 = st.columns(2)

    epochs_range = list(range(1, len(hist["accuracy"]) + 1))

    with ch1:
        st.markdown("**Accuracy per Epoch**")
        acc_data = {
            "Train Accuracy":      hist["accuracy"],
            "Validation Accuracy": hist["val_accuracy"],
        }
        import pandas as pd
        acc_df = pd.DataFrame(acc_data, index=epochs_range)
        st.line_chart(acc_df, use_container_width=True)

    with ch2:
        st.markdown("**Loss per Epoch**")
        loss_data = {
            "Train Loss":      hist["loss"],
            "Validation Loss": hist["val_loss"],
        }
        loss_df = pd.DataFrame(loss_data, index=epochs_range)
        st.line_chart(loss_df, use_container_width=True)

    st.markdown("### 📋 Epoch-by-Epoch History Table")
    rows = []
    for i, (a, va, l, vl) in enumerate(zip(
        hist["accuracy"], hist["val_accuracy"],
        hist["loss"],     hist["val_loss"]
    )):
        rows.append({
            "Epoch":         i + 1,
            "Train Acc (%)": f"{a*100:.2f}",
            "Val Acc (%)":   f"{va*100:.2f}",
            "Train Loss":    f"{l:.4f}",
            "Val Loss":      f"{vl:.4f}",
        })
    import pandas as pd
    df = pd.DataFrame(rows).set_index("Epoch")
    st.dataframe(df, use_container_width=True)

    st.markdown("### 🎓 Celebrity Classes")
    cols = st.columns(4)
    for i, name in enumerate(CLASS_NAMES):
        with cols[i % 4]:
            ref = get_reference_image(name)
            if ref:
                st.image(ref, caption=name, use_container_width=True)
