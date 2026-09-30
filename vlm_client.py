"""앱(TensorFlow venv)에서 증류 학생 VLM을 부르는 쪽.

1. prepare: 학습 데이터를 만들 때(distill/build_samples.py)와 똑같이 384 정사각형으로
   통일하고 MobileViT 판정 + Grad-CAM + 크롭을 만든다. 전처리가 다르면 학생이 못 보던
   분포가 들어가므로 함수를 그대로 가져다 쓴다.
2. stream: distill/vlm_server.py에 이미지 3장과 판정을 실어 보내고 평가문을 받아 흘려보낸다.
   이미지를 요청에 직접 싣기 때문에 서버는 로컬이든 Colab·AWS든 상관없다.
"""
import base64
import json
import os
import sys
import tempfile

import numpy as np
import requests
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'distill'))

from build_samples import SIZE, crop_box, gradcam, make_overlay, standardize

URL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vlm_url.txt')


def default_url():
    """서버 주소. 환경변수 VLM_URL > vlm_url.txt > 로컬 순으로 본다.
    Colab 터널 주소는 켤 때마다 바뀌므로 vlm_url.txt 한 줄만 고치면 되게 했다."""
    if os.environ.get('VLM_URL'):
        return os.environ['VLM_URL']
    if os.path.exists(URL_FILE):
        with open(URL_FILE, encoding='utf-8') as f:
            url = f.read().strip()
        if url:
            return url
    return 'http://127.0.0.1:8502'


VLM_URL = default_url()
OUT_ROOT = os.path.join(tempfile.gettempdir(), 'image-detection-vlm')


def _last_conv(model):
    # build_samples.last_conv_name은 tf.keras 클래스로 비교하는데, 앱은 tf_keras로 모델을
    # 올려 isinstance가 빗나간다. 클래스 이름으로 찾는다.
    name = None
    for layer in model.layers:
        if type(layer).__name__ == 'Conv2D':
            name = layer.name
    return name


def prepare(image_bytes, model, key):
    """판정 + 히트맵 + 크롭. (폴더, meta, 오버레이 이미지)를 돌려준다."""
    out = os.path.join(OUT_ROOT, key)
    os.makedirs(out, exist_ok=True)

    img = standardize(image_bytes)
    x = np.asarray(img.resize((224, 224), Image.NEAREST), np.float32)[None] / 255.0
    prob = float(model(x, training=False).numpy()[0, 0])  # 1에 가까울수록 실제 사진
    pred_real = prob >= 0.5

    grad_model = type(model)(model.input, [model.get_layer(_last_conv(model)).output, model.output])
    cam = gradcam(grad_model, tf.constant(x), pred_real)
    overlay, cam_big = make_overlay(img, cam)
    box = crop_box(cam_big)
    crop = img.crop(box).resize((SIZE, SIZE), Image.BICUBIC)

    img.save(f'{out}/orig.jpg', quality=95)
    overlay.save(f'{out}/cam.jpg', quality=90)
    crop.save(f'{out}/crop.jpg', quality=90)
    meta = {'mobilevit_prob_real': round(prob, 4),
            'mobilevit_pred': 'REAL' if pred_real else 'AI',
            'crop_box': box}
    with open(f'{out}/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)
    return out, meta, overlay


def health(url=VLM_URL):
    """서버 상태. 안 떠 있으면 None."""
    try:
        return requests.get(f'{url.rstrip("/")}/health', timeout=5).json()
    except (requests.RequestException, ValueError):
        return None


def stream(out, meta, url=VLM_URL):
    """생성문 조각을 나오는 대로 내놓는다."""
    images = {}
    for k in ('orig', 'cam', 'crop'):
        with open(f'{out}/{k}.jpg', 'rb') as f:
            images[k] = base64.b64encode(f.read()).decode()
    with requests.post(f'{url.rstrip("/")}/generate', json={'images': images, 'meta': meta},
                       stream=True, timeout=(10, 900)) as r:
        r.raise_for_status()
        r.encoding = 'utf-8'
        for piece in r.iter_content(chunk_size=None, decode_unicode=True):
            if piece:
                yield piece
