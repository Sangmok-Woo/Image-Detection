import os
import sys

try:
    import tensorflow as tf
    if not hasattr(tf, 'compat'):
        print("CRITICAL: TensorFlow가 정상적으로 로드되었으나 'compat' 모듈이 누락되었습니다.", file=sys.stderr)
except ImportError as e:
    print(f"CRITICAL: 가상환경 내에서 tensorflow 임포트 자체를 실패했습니다. 에러명: {e}", file=sys.stderr)

import base64
import io
import streamlit as st
import numpy as np

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tf_keras as keras
sys.modules['keras'] = keras
sys.modules['tensorflow.keras'] = keras

from model_utils import load_mobilevit_model, pre_process_img_mobilevit
from analysis_utils import get_vlm_explanation, generate_gradcam_overlay
from tf_keras.preprocessing.image import load_img, img_to_array

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
  <p class="main-title">AI vs REAL Image Detection</p>
  <p class="sub-title">이미지 위변조 탐지 시스템</p>
</div>
""", unsafe_allow_html=True)

# ── Upload Zone (centered) ───────────────────────────────────────────────────────
_, col_up, _ = st.columns([1, 4, 1])
with col_up:
    user_image = st.file_uploader(
        "이미지를 드래그하거나 클릭하여 업로드 (png, jpg, jpeg)",
        ["png", "jpg", "jpeg"],
        label_visibility="visible"
    )

# ── Analysis Flow ────────────────────────────────────────────────────────────────
if user_image is not None:
    image_bytes = user_image.read()
    image_b64 = base64.b64encode(image_bytes).decode()
    file_key = f"{user_image.name}_{len(image_bytes)}"

    # 세션 상태를 활용해 동일 파일 업로드 시 재연산 방지
    if st.session_state.get("file_key") != file_key:
        with st.spinner("Analyzing…"):
            try:
                preds = pre_process_img_mobilevit(io.BytesIO(image_bytes), mobilevit_model)
                prob_val = float(preds[0][0])
                
                # Grad-Cam 히트맵 생성 및 Base64 인코딩 진행
                heatmap_img = generate_gradcam_overlay(image_bytes, mobilevit_model)
                img_buffer = io.BytesIO()
                heatmap_img.save(img_buffer, format="PNG")
                heatmap_b64 = base64.b64encode(img_buffer.getvalue()).decode()

                st.session_state.update({
                    "file_key": file_key,
                    "prob": prob_val,
                    "image_b64": image_b64,
                    "heatmap_b64": heatmap_b64,
                    "image_bytes": image_bytes,
                })
            except Exception as e:
                st.session_state.pop("file_key", None)
                st.error(f"Error during analysis: {e}")

    if st.session_state.get("file_key") == file_key:
        prob        = st.session_state["prob"]
        image_b64   = st.session_state["image_b64"]
        heatmap_b64 = st.session_state["heatmap_b64"]

        is_ai       = prob < 0.5
        result_word = "AI Generated" if is_ai else "REAL"
        confidence  = 1 - prob if is_ai else prob
        
        bar_class   = "bar-fill-ai" if is_ai else "bar-fill-real"
        badge_class = "badge-ai"    if is_ai else "badge-real"
        bar_pct     = f"{confidence * 100:.1f}"

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        # ── Image + Result side-by-side (상단 레이아웃) ──────────────────────────
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

        # ── Deep Analysis Report (하단 상세 리포트) ──────────────────────────────
        with st.expander("🔬 Deep Analysis Report", expanded=True):
            exp_left, exp_right = st.columns([1, 1], gap="medium")
            
            with exp_left:
                st.markdown('<p class="section-label">🔍 Visual Heatmap</p>', unsafe_allow_html=True)
                # 상대방 코드의 원본 이미지 출력 버그를 상목님의 Grad-Cam 처리 이미지 렌더링으로 전면 수정
                st.markdown(
                    f'<div style="width:100%; text-align:center;" class="image-preview">'
                    f'<img src="data:image/png;base64,{heatmap_b64}" style="width:100%; max-width:700px; height:auto; border-radius:8px;"/>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
            with exp_right:
                st.markdown('<p class="section-label">📝 AI Reasoning</p>', unsafe_allow_html=True)
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