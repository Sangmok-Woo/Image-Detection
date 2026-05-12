import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import streamlit as st



# 필요한 경우 레거시 Keras 설정을 활성화할 수 있습니다.
# os.environ['TF_USE_LEGACY_KERAS'] = '1'

@st.cache_resource
def load_mobilevit_model():
    """
    저장된 MobileViT2 모델을 로드하고 캐싱합니다.
    경로가 잘못되었을 경우 사용자에게 에러를 알립니다.
    """
    model_path = 'MobileViT2_Model/MobileViT2_Model.h5'
    
    if not os.path.exists(model_path):
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path}\n경로를 다시 확인해주세요.")
        return None
        
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"모델 로드 중 오류가 발생했습니다: {e}")
        return None

def pre_process_img_mobilevit(image, model):
    """
    업로드된 이미지를 모델 규격에 맞춰 전처리하고 예측값을 반환합니다.
    """
    try:
        # 모델의 입력 크기에 맞춰 리사이징 (MobileViT2 기준 224x224)
        target_size = (224, 224) 
        
        # Streamlit의 UploadedFile 객체를 바로 load_img에 전달 가능
        img = load_img(image, target_size=target_size)
        
        # 이미지 배열 변환 및 정규화 (0~1 사이 값)
        img_arr = img_to_array(img) / 255.0
        
        # 배치 차원 추가 (1, 224, 224, 3)
        img_arr = np.expand_dims(img_arr, axis=0)
        
        # 예측 수행
        prediction = model.predict(img_arr)
        return prediction
        
    except Exception as e:
        # 전처리 과정에서 발생할 수 있는 오류를 호출부로 전달
        raise RuntimeError(f"이미지 전처리 중 오류 발생: {e}")