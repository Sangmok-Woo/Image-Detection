# imports
import base64
import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Keras 2/3 호환성 이슈 해결을 위한 환경 변수 (필요 시 주석 해제)
# os.environ['TF_USE_LEGACY_KERAS'] = '1'

# load css
def load_local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_local_css("./styles/style.css")

# bootstrap
st.markdown(
    """<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">""",
    unsafe_allow_html=True
)

# load model weights
@st.cache_resource
def load_mobilevit_model():
    # 경로와 확장자(.ht)를 실제 파일과 동일하게 설정하세요.
    model_path = 'MobileViT2_Model/MobileViT2_Model.h5'
    model = tf.keras.models.load_model(model_path)
    return model

# Access cached model
mobilevit_model = load_mobilevit_model()

# preprocess image for MobileViT2
def pre_process_img_mobilevit(image):
    target_size = (224, 224) 
    img = load_img(image, target_size=target_size)
    img_arr = img_to_array(img) / 255.0  # 정규화
    img_arr = np.expand_dims(img_arr, axis=0) # 배치 차원 추가
    prediction = mobilevit_model.predict(img_arr)
    return prediction

# --- UI ---

# title section
col1, col2, col3, col4, col5 = st.columns([4,1,3,3,1], gap="small")

with col2:
    if os.path.exists("styles/robot.png"):
        st.image("styles/robot.png")
with col3:
    st.markdown('<p class="title"> AI vs REAL Image Detection </p>', unsafe_allow_html=True)

# Main layout
main_col_one, main_col_two = st.columns([2,2], gap="large")

with main_col_one:
    image_placeholder = st.empty()

with main_col_two:
    # Detective SVG icon
    if os.path.exists("styles/detectiveMag.svg"):
        with open("styles/detectiveMag.svg", "r") as file:
            svg_content = file.read()
        
        c1, c2, c3 = st.columns([4,4,1], gap="small")
        with c2:
            st.markdown('<p class="upload_line"> Please upload the image </p>', unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='detectiveMag1'>{svg_content}</div>", unsafe_allow_html=True)

    # File uploader
    user_image = st.file_uploader("png, jpg, or jpeg image", ['png', 'jpg', 'jpeg'], label_visibility='hidden')
    result_placeholder = st.empty()

# Prediction Logic
if user_image is not None:
    # Display uploaded image
    image_bytes = user_image.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    image_placeholder.markdown(
        f'<div style="display: flex; justify-content: center;">'
        f'<img src="data:image/jpeg;base64,{image_base64}" style="max-width:100%; height:auto;"/>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Single Model Execution (MobileViT2)
    with st.spinner('Analyzing image...'):
        try:
            predictions = pre_process_img_mobilevit(user_image)
            prob = predictions[0][0]
            
            # 결과 판별 (0.5 기준)
            if prob < 0.5:
                result_word = "AI Generated"
            else:
                result_word = "REAL"
            
            result_placeholder.markdown(
                f"<div class='result'>"
                f"<span class='prediction'>Confidence: {prob:.2%}</span> <br>"
                f"It is a <span class='resultword'>{result_word}</span> image."
                f"</div>", 
                unsafe_allow_html=True
            )
            print(f"MobileViT2 Prediction: {prob}")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")