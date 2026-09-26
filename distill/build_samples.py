"""1단계: Tiny-GenImage에서 샘플을 뽑아 MobileViT 판정 + Grad-CAM + 크롭을 만든다.

사용법:
    venv\\Scripts\\python.exe distill\\build_samples.py --shards <parquet 파일들> --n 1000

결과 (distill/work/):
    std/{id}.jpg      384x384로 통일한 원본 (학생 VLM 입력 1)
    cam/{id}.jpg      Grad-CAM 오버레이 (입력 2)
    crop/{id}.jpg     히트맵 최고점 주변 크롭, 384로 확대 (입력 3)
    panel/{id}.jpg    위 세 장을 가로로 붙인 것 (교사 Claude가 보는 용도)
    meta.jsonl        id, 정답, 생성기, MobileViT 확률, 크롭 좌표
"""
import argparse
import io
import json
import os
import random

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image, ImageDraw

MODEL_PATH = 'MobileViT2_Model/MobileViT2_Model.h5'
GENERATORS = ['Real', 'ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD14', 'SD15', 'VQDM', 'Wukong']
SIZE = 384
CROP = 128  # 384 기준 1/3 크기 영역을 잘라 확대한다


def standardize(img_bytes):
    # 진짜=JPEG 직사각형, 가짜=PNG 정사각형이라 형식·크기가 곧 정답 힌트가 된다.
    # 짧은 변 기준 리사이즈 → 가운데 정사각형 크롭 → JPEG q90 재인코딩으로 흔적을 지운다.
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    w, h = img.size
    s = SIZE / min(w, h)
    img = img.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    img = img.crop((left, top, left + SIZE, top + SIZE))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=90)
    return Image.open(io.BytesIO(buf.getvalue())).convert('RGB')


def sample_rows(shards, n, seed):
    # 진짜 n/2, 가짜 n/2 (가짜는 생성기별 균등)
    rng = random.Random(seed)
    by_gen = {}
    for path in shards:
        t = pq.read_table(path)
        imgs, gens = t['image'].to_pylist(), t['generator'].to_pylist()
        for i, g in enumerate(gens):
            by_gen.setdefault(g, []).append((os.path.basename(path), i, imgs[i]['bytes']))
    fake_gens = sorted(g for g in by_gen if g != 0)
    picked = [(s, i, b, 0) for s, i, b in rng.sample(by_gen[0], n // 2)]
    per, extra = divmod(n - n // 2, len(fake_gens))
    for k, g in enumerate(fake_gens):
        picked += [(s, i, b, g) for s, i, b in rng.sample(by_gen[g], per + (1 if k < extra else 0))]
    rng.shuffle(picked)
    return picked


def last_conv_name(model):
    name = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            name = layer.name
    return name


def gradcam(grad_model, x, target_real):
    # 예측한 클래스 쪽 점수로 기울기를 잡는다 (REAL이면 p, AI면 1-p)
    with tf.GradientTape() as tape:
        conv, pred = grad_model(x, training=False)
        score = pred[:, 0] if target_real else 1.0 - pred[:, 0]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.squeeze(conv[0] @ weights[..., tf.newaxis])).numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def make_overlay(img, cam):
    cam_big = np.array(Image.fromarray(np.uint8(cam * 255)).resize((SIZE, SIZE), Image.BILINEAR)) / 255.0
    colored = colormaps['jet'](cam_big)[:, :, :3] * 255
    blended = 0.55 * np.asarray(img, np.float32) + 0.45 * colored
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)), cam_big


def crop_box(cam_big):
    y, x = np.unravel_index(np.argmax(cam_big), cam_big.shape)
    left = int(np.clip(x - CROP // 2, 0, SIZE - CROP))
    top = int(np.clip(y - CROP // 2, 0, SIZE - CROP))
    return left, top, left + CROP, top + CROP


def make_panel(img, overlay, crop, box):
    boxed = overlay.copy()
    ImageDraw.Draw(boxed).rectangle(box, outline=(255, 255, 255), width=3)
    panel = Image.new('RGB', (SIZE * 3 + 16, SIZE + 22), (255, 255, 255))
    d = ImageDraw.Draw(panel)
    for k, (im, title) in enumerate([(img, '1 original'), (boxed, '2 grad-cam (box = crop)'), (crop, '3 crop x3')]):
        panel.paste(im, (k * (SIZE + 8), 22))
        d.text((k * (SIZE + 8) + 4, 5), title, fill=(0, 0, 0))
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--out', default='distill/work')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    for sub in ['std', 'cam', 'crop', 'panel']:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    grad_model = tf.keras.Model(model.input, [model.get_layer(last_conv_name(model)).output, model.output])

    rows = sample_rows(args.shards, args.n, args.seed)
    with open(os.path.join(args.out, 'meta.jsonl'), 'w', encoding='utf-8') as meta:
        for k, (shard, idx, b, gen) in enumerate(rows):
            sid = f'tg{k:04d}'
            img = standardize(b)
            # app.py와 같이 224 nearest 리사이즈 후 0~1 정규화
            x = np.asarray(img.resize((224, 224), Image.NEAREST), np.float32)[None] / 255.0
            prob = float(model(x, training=False).numpy()[0, 0])  # 1=REAL
            pred_real = prob >= 0.5
            cam = gradcam(grad_model, x, pred_real)
            overlay, cam_big = make_overlay(img, cam)
            box = crop_box(cam_big)
            crop = img.crop(box).resize((SIZE, SIZE), Image.BICUBIC)

            img.save(f'{args.out}/std/{sid}.jpg', quality=95)
            overlay.save(f'{args.out}/cam/{sid}.jpg', quality=90)
            crop.save(f'{args.out}/crop/{sid}.jpg', quality=90)
            make_panel(img, overlay, crop, box).save(f'{args.out}/panel/{sid}.jpg', quality=85)

            meta.write(json.dumps({
                'id': sid, 'source': f'{shard}#{idx}',
                'label': 'REAL' if gen == 0 else 'AI', 'generator': GENERATORS[gen],
                'mobilevit_prob_real': round(prob, 4),
                'mobilevit_pred': 'REAL' if pred_real else 'AI',
                'crop_box': box,
            }, ensure_ascii=False) + '\n')
            if k % 100 == 0:
                print(f'{k}/{len(rows)}', flush=True)
    print('done')


if __name__ == '__main__':
    main()
