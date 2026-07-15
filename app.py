import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CLEAN UI STYLES
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', system-ui, -apple-system, sans-serif;
}

.stApp {
    background: #0c0c10;
    color: #e8e8ed;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 620px !important;
}

/* ── Title ────────────────────────────────── */
.page-title {
    text-align: center;
    margin: 0 0 0.35rem 0;
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #f4f4f5;
    line-height: 1.2;
}
.page-sub {
    text-align: center;
    color: #71717a;
    font-size: 0.95rem;
    margin: 0 0 1.35rem 0;
    line-height: 1.45;
}

/* ── Section label ────────────────────────── */
.section-label {
    text-align: center;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #52525b;
    margin: 0.35rem 0 0.75rem 0;
}

/* ── Canvas ───────────────────────────────── */
iframe[title*="st_canvas"] {
    touch-action: none;
    display: block !important;
    margin: 0 auto !important;
    width: 280px !important;
    height: 280px !important;
    border: 1px solid #26262c !important;
    border-radius: 14px !important;
    background-color: #000000 !important;
    box-shadow: none !important;
}

/* ── Clear button alignment fix ───────────── */
div[data-testid="stButton"] > button {
    background: #101014 !important;
    color: #d4d4d8 !important;
    border: 1px solid #26262c !important;
    border-radius: 12px !important;
    padding: 0.55rem 1.2rem !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    
    /* Lock the exact width to match the canvas and center it */
    width: 280px !important;
    display: block !important;
    margin: 0.25rem auto 0 auto !important;
    
    box-shadow: none !important;
    transition: background 0.12s ease, border-color 0.12s ease, color 0.12s ease !important;
}

div[data-testid="stButton"] > button:hover {
    background: #18181d !important;
    border-color: #3f3f46 !important;
    color: #fafafa !important;
    transform: none !important;
}

/* ── Results section ──────────────────────── */
.results-label {
    text-align: center;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #52525b;
    margin: 1.55rem 0 1rem 0;
}

/* ── Prediction cards ─────────────────────── */
.prediction-card {
    background: #121216;
    border: 1px solid #232329;
    border-radius: 14px;
    padding: 0.95rem 0.7rem 0.85rem;
    text-align: center;
    min-height: 160px;
    box-shadow: none;
}
.prediction-card.perceptron { border-top: 3px solid #38bdf8; }
.prediction-card.ann { border-top: 3px solid #a78bfa; }
.prediction-card.cnn { border-top: 3px solid #fbbf24; }

.model-name {
    text-align: center;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}
.model-name.perceptron { color: #38bdf8; }
.model-name.ann { color: #a78bfa; }
.model-name.cnn { color: #fbbf24; }

.model-digit {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 2.65rem;
    font-weight: 600;
    color: #fafafa;
    line-height: 1.1;
    margin-bottom: 0.6rem;
}

.prog-track {
    background: #1c1c22;
    border-radius: 999px;
    height: 5px;
    overflow: hidden;
}
.prog-fill {
    border-radius: 999px;
    height: 5px;
    transition: width 0.35s ease;
}

.confidence {
    text-align: center;
    font-size: 0.78rem;
    color: #71717a;
    margin-top: 0.4rem;
    font-variant-numeric: tabular-nums;
}
.confidence b {
    color: #d4d4d8;
    font-weight: 600;
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
}

/* ── Empty state ──────────────────────────── */
.empty-state {
    background: #121216;
    border: 1px solid #232329;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    margin-top: 1.1rem;
}
.empty-icon {
    font-size: 2.1rem;
    margin-bottom: 0.5rem;
    opacity: 0.65;
}
.empty-main {
    color: #e4e4e7;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
}
.empty-sub {
    color: #71717a;
    font-size: 0.82rem;
    line-height: 1.45;
}

/* ── Expander ─────────────────────────────── */
details {
    background: #121216 !important;
    border: 1px solid #232329 !important;
    border-radius: 10px !important;
    margin-top: 1.2rem !important;
}
details summary {
    color: #71717a !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}
details[open] summary {
    color: #a1a1aa !important;
}

/* ── Columns ──────────────────────────────── */
[data-testid="column"] {
    display: flex;
    flex-direction: column;
}

/* ── Caption ──────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    text-align: center !important;
    color: #52525b !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<h1 class="page-title">🧠 Digit Classifier</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-sub">Draw a digit (0–9). Three models classify it live.</p>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
def _patch_keras_dense_for_model_loading():
    """Allow newer model snapshots to load on older Keras runtimes."""
    try:
        from keras.src.layers.core.dense import Dense
    except Exception:
        return False

    if getattr(Dense.__init__, "__name__", "") == "_patched_dense_init":
        return True

    original_init = Dense.__init__

    def _patched_dense_init(self, *args, quantization_config=None, **kwargs):
        kwargs.pop("quantization_config", None)
        return original_init(self, *args, **kwargs)

    Dense.__init__ = _patched_dense_init
    return True


@st.cache_resource
def load_models():
    _patch_keras_dense_for_model_loading()
    try:
        m1 = tf.keras.models.load_model("models/perceptron_model.h5")
        m2 = tf.keras.models.load_model("models/ann_model.h5")
        m3 = tf.keras.models.load_model("models/cnn_model.h5")
        return m1, m2, m3
    except Exception as e:
        st.error(
            f"Error loading models. Verify that 'perceptron_model.h5', 'ann_model.h5', "
            f"and 'cnn_model.h5' are placed inside the 'models' folder. \nError: {e}"
        )
        return None, None, None


perceptron, ann, cnn = load_models()

# ─────────────────────────────────────────────
# HELPER TO RENDER CARDS
# ─────────────────────────────────────────────
def render_prediction_card(title, digit, confidence, card_class, name_class, bar_color):
    st.markdown(
        f"""
        <div class="prediction-card {card_class}">
            <div class="model-name {name_class}">{title}</div>
            <div class="model-digit">{digit}</div>
            <div class="prog-track">
                <div class="prog-fill" style="width:{min(confidence, 100):.1f}%; background:{bar_color};"></div>
            </div>
            <div class="confidence"><b>{confidence:.1f}%</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# CANVAS
# ─────────────────────────────────────────────
if "canvas_reset_count" not in st.session_state:
    st.session_state.canvas_reset_count = 0

st.markdown('<p class="section-label">Drawing pad</p>', unsafe_allow_html=True)

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.0)",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_reset_count}",
    display_toolbar=False,
)

# use_container_width=True stretches the invisible wrapper 
# allowing our CSS margin: 0 auto; to snap perfectly to the center
if st.button("Clear canvas", use_container_width=True):
    st.session_state.canvas_reset_count += 1
    st.rerun()

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
if canvas_result.image_data is not None:
    raw_img = canvas_result.image_data

    # FIX: ignore alpha channel so a blank canvas doesn't trigger a false prediction.
    rgb = raw_img[:, :, :3] if raw_img.shape[-1] >= 3 else raw_img
    has_drawing = np.max(rgb) > 20

    if has_drawing:
        st.markdown(
            '<p class="results-label">Predictions</p>',
            unsafe_allow_html=True,
        )

        img = Image.fromarray(raw_img.astype("uint8")).convert("L")
        img_resized = img.resize((28, 28))
        img_array = np.array(img_resized)
        img_normalized = img_array.astype("float32") / 255.0

        col1, col2, col3 = st.columns(3, gap="small")

        # A. Perceptron
        if perceptron:
            prep_percept = img_normalized.reshape(1, 28, 28)
            pred_percept = perceptron.predict(prep_percept, verbose=0)
            digit_percept = int(np.argmax(pred_percept))
            confidence_percept = float(np.max(pred_percept) * 100)

            with col1:
                render_prediction_card(
                    title="Perceptron",
                    digit=digit_percept,
                    confidence=confidence_percept,
                    card_class="perceptron",
                    name_class="perceptron",
                    bar_color="#38bdf8",
                )

        # B. ANN
        if ann:
            prep_ann = img_normalized.reshape(1, 28, 28)
            pred_ann = ann.predict(prep_ann, verbose=0)
            digit_ann = int(np.argmax(pred_ann))
            confidence_ann = float(np.max(pred_ann) * 100)

            with col2:
                render_prediction_card(
                    title="ANN",
                    digit=digit_ann,
                    confidence=confidence_ann,
                    card_class="ann",
                    name_class="ann",
                    bar_color="#a78bfa",
                )

        # C. CNN
        if cnn:
            prep_cnn = img_normalized.reshape(1, 28, 28, 1)
            pred_cnn = cnn.predict(prep_cnn, verbose=0)
            digit_cnn = int(np.argmax(pred_cnn))
            confidence_cnn = float(np.max(pred_cnn) * 100)

            with col3:
                render_prediction_card(
                    title="CNN",
                    digit=digit_cnn,
                    confidence=confidence_cnn,
                    card_class="cnn",
                    name_class="cnn",
                    bar_color="#fbbf24",
                )

        with st.expander("View 28×28 input"):
            st.image(img_resized, width=112)

    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">✏️</div>
                <div class="empty-main">Draw a digit above to run predictions</div>
                <div class="empty-sub">The results will appear here once the canvas contains actual ink.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )