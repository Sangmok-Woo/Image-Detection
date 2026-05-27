import base64
import io
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

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
  <p class="main-title">Image-Detection</p>
  <p class="sub-title">이미지 위변조 탐지 시스템</p>
</div>
""", unsafe_allow_html=True)

# ── Upload Zone (centered) ───────────────────────────────────────────────────────
_, col_up, _ = st.columns([1, 4, 1])
with col_up:
    user_image = st.file_uploader(
        "이미지를 드래그하거나 클릭하여 업로드",
        ["png", "jpg", "jpeg"],
        label_visibility="visible"
    )

# ── Analysis Flow ────────────────────────────────────────────────────────────────
if user_image is not None:
    image_bytes = user_image.read()
    image_b64 = base64.b64encode(image_bytes).decode()
    file_key = f"{user_image.name}_{len(image_bytes)}"

    if st.session_state.get("file_key") != file_key:
        with st.spinner("Analyzing…"):
            try:
                preds = pre_process_img_mobilevit(io.BytesIO(image_bytes), mobilevit_model)
                st.session_state.update({
                    "file_key": file_key,
                    "prob": float(preds[0][0]),
                    "image_b64": image_b64,
                    "image_bytes": image_bytes,
                })
            except Exception as e:
                st.session_state.pop("file_key", None)
                st.error(f"Error during analysis: {e}")

    if st.session_state.get("file_key") == file_key:
        prob        = st.session_state["prob"]
        image_b64   = st.session_state["image_b64"]
        image_bytes = st.session_state["image_bytes"]

        is_ai       = prob < 0.5
        result_word = "AI Generated" if is_ai else "REAL"
        confidence  = 1 - prob if is_ai else prob
        bar_class   = "bar-fill-ai" if is_ai else "bar-fill-real"
        badge_class = "badge-ai"    if is_ai else "badge-real"
        bar_pct     = f"{confidence * 100:.1f}"

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        # ── Image + Result side-by-side ──────────────────────────────────────────
        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown('<p class="section-label">Input Image</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="image-preview">'
                f'<img src="data:image/jpeg;base64,{image_b64}"/>'
                f'</div>',
                unsafe_allow_html=True
            )

        with col_res:
            st.markdown('<p class="section-label">Analysis Result</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-card">
              <span class="verdict-badge {badge_class}">{result_word}</span>
              <p class="confidence-value">{confidence:.1%}</p>
              <p class="confidence-sub">Confidence</p>
              <div class="bar-track">
                <div class="bar-fill {bar_class}" style="width:{bar_pct}%;"></div>
              </div>
              <p class="result-meta">score {prob:.4f} &nbsp;/&nbsp; threshold 0.50</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Deep Analysis Report (expander) ──────────────────────────────────────
        with st.expander("Deep Analysis Report"):
            exp_left, exp_right = st.columns([1, 1], gap="medium")
            with exp_left:
                st.markdown('<p class="section-label">Visual Heatmap</p>', unsafe_allow_html=True)
                st.image(io.BytesIO(image_bytes), use_container_width=True)
            with exp_right:
                st.markdown('<p class="section-label">AI Reasoning</p>', unsafe_allow_html=True)
                explanation = get_vlm_explanation(prob, result_word)
                st.markdown(
                    f'<div class="reasoning-box">{explanation}</div>',
                    unsafe_allow_html=True
                )

else:
    st.session_state.pop("file_key", None)
    st.markdown(
        '<div class="upload-hint">이미지를 업로드하면 AI 위변조 여부를 분석합니다</div>',
        unsafe_allow_html=True
    )

# ── Footer ───────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">MobileViT v2 &nbsp;·&nbsp; Threshold 0.50</div>',
    unsafe_allow_html=True
)
