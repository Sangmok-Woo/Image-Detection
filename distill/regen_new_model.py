"""신모델로 판정·Grad-CAM·크롭·패널만 다시 만든다. 원본(std/)은 그대로 쓴다.

사용법:
    venv\\Scripts\\python.exe distill\\regen_new_model.py
입력: distill/work/std/, distill/work/old_model/meta.jsonl (id·정답·생성기·출처)
출력: distill/work/meta.jsonl, cam/, crop/, panel/  (구모델 결과는 old_model/에 백업돼 있다)
"""
import json
import os
import sys

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from PIL import Image

# PC가 버벅이지 않게 CPU 2코어만 쓴다
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

sys.path.insert(0, os.path.dirname(__file__))
from build_samples import MODEL_PATH, SIZE, crop_box, gradcam, last_conv_name, make_overlay, make_panel

WORK = 'distill/work'


def main():
    for sub in ['cam', 'crop', 'panel']:
        os.makedirs(f'{WORK}/{sub}', exist_ok=True)
    old = [json.loads(l) for l in open(f'{WORK}/old_model/meta.jsonl', encoding='utf-8')]

    # 중간에 끊겨도 이어서 돌 수 있게 이미 끝난 id는 건너뛴다
    done = set()
    if os.path.exists(f'{WORK}/meta.jsonl'):
        done = {json.loads(l)['id'] for l in open(f'{WORK}/meta.jsonl', encoding='utf-8') if l.strip()}

    model = tf.keras.models.load_model(MODEL_PATH)
    grad_model = tf.keras.Model(model.input, [model.get_layer(last_conv_name(model)).output, model.output])

    with open(f'{WORK}/meta.jsonl', 'a', encoding='utf-8') as meta:
        for k, m in enumerate(old):
            sid = m['id']
            if sid in done:
                continue
            img = Image.open(f'{WORK}/std/{sid}.jpg').convert('RGB')
            x = np.asarray(img.resize((224, 224), Image.NEAREST), np.float32)[None] / 255.0
            prob = float(model(x, training=False).numpy()[0, 0])  # 신모델도 1=REAL 방향 (비교 결과로 확인)
            pred_real = prob >= 0.5
            cam = gradcam(grad_model, x, pred_real)
            overlay, cam_big = make_overlay(img, cam)
            box = crop_box(cam_big)
            crop = img.crop(box).resize((SIZE, SIZE), Image.BICUBIC)

            overlay.save(f'{WORK}/cam/{sid}.jpg', quality=90)
            crop.save(f'{WORK}/crop/{sid}.jpg', quality=90)
            make_panel(img, overlay, crop, box).save(f'{WORK}/panel/{sid}.jpg', quality=85)
            meta.write(json.dumps({**m, 'mobilevit_prob_real': round(prob, 4),
                                   'mobilevit_pred': 'REAL' if pred_real else 'AI',
                                   'crop_box': box, 'model': 'new'}, ensure_ascii=False) + '\n')
            meta.flush()
            if k % 50 == 0:
                print(f'{k}/{len(old)}', flush=True)
    print('done')


if __name__ == '__main__':
    main()
