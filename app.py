import base64
import os
import streamlit as st

from model_utils import load_mobilevit_model, pre_process_img_mobilevit
from analysis_utils import get_vlm_explanation

st.set_page_config(
    page_title="AI vs REAL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("./styles/style.css")

mobilevit_model = load_mobilevit_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
  <p class="main-title">Image-Detection</p>
  <p class="sub-title">이미지 위변조 탐지 시스템</p>
</div>
""", unsafe_allow_html=True)

# ── Main: two columns ──────────────────────────────────────────────────────────
col_img, col_result = st.columns([1, 1], gap="large")

with col_img:
    st.markdown('<p class="section-label">Input Image</p>', unsafe_allow_html=True)
    image_placeholder = st.empty()
    image_placeholder.markdown(
        '<div class="image-placeholder"><span>No image uploaded yet</span></div>',
        unsafe_allow_html=True
    )
    user_image = st.file_uploader(
        "Upload an image to analyze",
        ["png", "jpg", "jpeg"],
        label_visibility="visible"
    )

with col_result:
    st.markdown('<p class="section-label">Analysis Result</p>', unsafe_allow_html=True)
    result_placeholder = st.empty()
    result_placeholder.markdown(
        '<div class="result-waiting">Upload an image to see the result</div>',
        unsafe_allow_html=True
    )

# ── Deep Analysis Report ───────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p class="section-header">Deep Analysis Report</p>', unsafe_allow_html=True)

rep_left, rep_right = st.columns([1, 1], gap="medium")

with rep_left:
    st.markdown('<p class="section-label">Visual Heatmap</p>', unsafe_allow_html=True)
    heatmap_placeholder = st.empty()
    heatmap_placeholder.markdown(
        '<div class="analysis-waiting">Awaiting image</div>',
        unsafe_allow_html=True
    )

with rep_right:
    st.markdown('<p class="section-label">AI Reasoning</p>', unsafe_allow_html=True)
    vlm_placeholder = st.empty()
    vlm_placeholder.markdown(
        '<div class="analysis-waiting">Awaiting analysis</div>',
        unsafe_allow_html=True
    )

# ── Logic ──────────────────────────────────────────────────────────────────────
if user_image is not None:
    image_bytes = user_image.read()
    image_b64 = base64.b64encode(image_bytes).decode()

    image_placeholder.markdown(
        f'<div class="image-preview">'
        f'<img src="data:image/jpeg;base64,{image_b64}"/>'
        f'</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Analyzing…"):
        try:
            predictions = pre_process_img_mobilevit(user_image, mobilevit_model)
            prob = float(predictions[0][0])
            is_ai = prob < 0.5
            result_word = "AI Generated" if is_ai else "REAL"
            confidence = 1 - prob if is_ai else prob

            verdict_label = "AI Generated" if is_ai else "Real"
            bar_class = "bar-fill-ai" if is_ai else "bar-fill-real"
            bar_pct = f"{confidence * 100:.1f}"

            result_placeholder.markdown(f"""
            <div class="result-card">
              <p class="verdict-text">{verdict_label}</p>
              <p class="confidence-value">{confidence:.1%}</p>
              <p class="confidence-sub">Confidence</p>
              <div class="bar-track">
                <div class="bar-fill {bar_class}" style="width:{bar_pct}%;"></div>
              </div>
              <p class="result-meta">score {prob:.4f} &nbsp;/&nbsp; threshold 0.50</p>
            </div>
            """, unsafe_allow_html=True)

            heatmap_placeholder.image(user_image, use_container_width=True)

            explanation = get_vlm_explanation(prob, result_word)
            vlm_placeholder.markdown(
                f'<div class="reasoning-box">{explanation}</div>',
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error during analysis: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">MobileViT v2 &nbsp;·&nbsp; Threshold 0.50</div>',
    unsafe_allow_html=True
)
