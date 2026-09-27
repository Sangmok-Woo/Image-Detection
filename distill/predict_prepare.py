"""추론 1단계: 이미지 한 장을 MobileViT로 판정하고 Grad-CAM·크롭을 만든다.

TensorFlow가 있는 venv에서 돈다 (torch venv와 numpy 버전이 달라 환경을 나눴다).
보통은 predict.py가 알아서 불러 쓰고, 직접 쓸 일은 디버깅할 때뿐이다.

    venv\\Scripts\\python.exe distill\\predict_prepare.py <이미지> --out <폴더>

결과 (out/):
    orig.jpg  cam.jpg  crop.jpg  meta.json
학습 데이터를 만들 때(build_samples.py)와 똑같은 전처리를 쓴다. 다르면 학생이 못 보던
분포가 들어가므로 여기서 함수를 재사용한다.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from PIL import Image

from build_samples import (MODEL_PATH, SIZE, crop_box, gradcam, last_conv_name,
                           make_overlay, make_panel, standardize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default=MODEL_PATH)
    ap.add_argument('--panel', action='store_true', help='교사용 3칸 패널도 남긴다')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.image, 'rb') as f:
        img = standardize(f.read())

    model = tf.keras.models.load_model(args.model)
    grad_model = tf.keras.Model(model.input, [model.get_layer(last_conv_name(model)).output, model.output])

    # app.py·build_samples.py와 같은 전처리: 224 nearest, 0~1
    x = np.asarray(img.resize((224, 224), Image.NEAREST), np.float32)[None] / 255.0
    prob = float(model(x, training=False).numpy()[0, 0])  # 1에 가까울수록 실제 사진
    pred_real = prob >= 0.5
    cam = gradcam(grad_model, x, pred_real)
    overlay, cam_big = make_overlay(img, cam)
    box = crop_box(cam_big)
    crop = img.crop(box).resize((SIZE, SIZE), Image.BICUBIC)

    img.save(f'{args.out}/orig.jpg', quality=95)
    overlay.save(f'{args.out}/cam.jpg', quality=90)
    crop.save(f'{args.out}/crop.jpg', quality=90)
    if args.panel:
        make_panel(img, overlay, crop, box).save(f'{args.out}/panel.jpg', quality=85)

    meta = {'source': os.path.abspath(args.image),
            'mobilevit_prob_real': round(prob, 4),
            'mobilevit_pred': 'REAL' if pred_real else 'AI',
            'crop_box': box}
    with open(f'{args.out}/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    main()
