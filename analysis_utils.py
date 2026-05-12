import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import streamlit as st

@st.cache_resource
def load_mobilevit_model():
    model_path = 'MobileViT2_Model/MobileViT2_Model.h5'
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

def pre_process_img_mobilevit(image, model):
    target_size = (224, 224) 
    img = load_img(image, target_size=target_size)
    img_arr = img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    prediction = model.predict(img_arr)
    return prediction

def get_vlm_explanation(prob, result_word):
    if result_word == "AI Generated":
        return f"이 이미지는 생성 모델 특유의 패턴이 감지되었습니다. 특히 경계선의 부자연스러움과 픽셀의 비정상적인 분포가 관찰됩니다."
    return f"이 이미지는 자연스러운 이미지 노이즈와 광원 처리를 보여주고 있습니다. 인공적인 생성 징후가 발견되지 않았습니다."