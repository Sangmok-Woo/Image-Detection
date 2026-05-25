import base64
import os
import streamlit as st

from model_utils import load_mobilevit_model, pre_process_img_mobilevit
from analysis_utils import get_vlm_explanation

# --- 설정 및 리소스 로드 ---
st.set_page_config(page_title="AI vs REAL Detector", layout="wide")

def load_local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_local_css("./styles/style.css")
st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">', unsafe_allow_html=True)

mobilevit_model = load_mobilevit_model()

# --- UI 레이아웃 ---
# Title Section
col1, col2, col3, col4, col5 = st.columns([4,1,3,3,1], gap="small")
with col2:
    if os.path.exists("styles/robot.png"): st.image("styles/robot.png")
with col3:
    st.markdown('<p class="title"> AI vs REAL Image Detection </p>', unsafe_allow_html=True)

# Main layout (상단)
main_col_one, main_col_two = st.columns([2,2], gap="large")

with main_col_one:
    image_placeholder = st.empty()

with main_col_two:
    if os.path.exists("styles/detectiveMag.svg"):
        with open("styles/detectiveMag.svg", "r") as file:
            svg_content = file.read()
        c1, c2, c3 = st.columns([4,4,1], gap="small")
        with c2: st.markdown('<p class="upload_line"> Please upload the image </p>', unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='detectiveMag1'>{svg_content}</div>", unsafe_allow_html=True)

    user_image = st.file_uploader("png, jpg, or jpeg image", ['png', 'jpg', 'jpeg'], label_visibility='hidden')
    result_placeholder = st.empty()

# 상세 분석 섹션 (하단)
st.markdown("<br><hr>", unsafe_allow_html=True)
st.subheader("🔬 Deep Analysis Report")
detail_col_left, detail_col_right = st.columns([1, 1], gap="medium")

with detail_col_left:
    st.markdown("#### 🔍 Visual Heatmap")
    heatmap_placeholder = st.empty()

with detail_col_right:
    st.markdown("#### 📝 AI Reasoning")
    vlm_explanation_placeholder = st.empty()

# --- 실행 로직 ---
if user_image is not None:
    # 원본 이미지 표시
    image_bytes = user_image.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    image_placeholder.markdown(
        f'<div style="display: flex; justify-content: center;">'
        f'<img src="data:image/jpeg;base64,{image_base64}" style="max-width:100%; height:auto;"/>'
        f'</div>', unsafe_allow_html=True
    )

    with st.spinner('Analyzing...'):
        try:
            # 1. 모델 분석
            predictions = pre_process_img_mobilevit(user_image, mobilevit_model)
            prob = predictions[0][0]
            result_word = "AI Generated" if prob < 0.5 else "REAL"
            
            # 2. 결과 출력 (상단)
            result_placeholder.markdown(
                f"<div class='result'><span class='prediction'>Confidence: {prob:.2%}</span> <br>"
                f"It is a <span class='resultword'>{result_word}</span> image.</div>", 
                unsafe_allow_html=True
            )
            
            # 3. 상세 리포트 (하단)
            heatmap_placeholder.image(user_image, use_container_width=True) # 히트맵 로직 연결 전 가변 공간
            explanation = get_vlm_explanation(prob, result_word)
            vlm_explanation_placeholder.info(explanation)
            
        except Exception as e:
            st.error(f"Error: {e}")