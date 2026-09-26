"""구모델(_old)과 신모델을 distill/work/std의 1000장에 돌려 판정과 Grad-CAM을 비교한다.

사용법:
    venv\\Scripts\\python.exe distill\\compare_models.py
결과: distill/work/compare/summary.json, per_sample.jsonl
"""
import json
import os
from collections import defaultdict

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf

# PC가 버벅이지 않게 CPU 2코어만 쓴다
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
from PIL import Image

WORK = 'distill/work'
MODELS = {'old': 'MobileViT2_Model/MobileViT2_Model_old.h5', 'new': 'MobileViT2_Model/MobileViT2_Model.h5'}
BATCH = 1
N = int(os.environ.get('N', '30'))  # 진짜 절반 + 생성기별 균등으로 N장만 본다


def last_conv_name(model):
    return [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]


def run(path, x):
    model = tf.keras.models.load_model(path)
    grad_model = tf.keras.Model(model.input, [model.get_layer(last_conv_name(model)).output, model.output])
    probs, cams = [], []
    for b in range(0, len(x), BATCH):
        xb = tf.constant(x[b:b + BATCH])
        with tf.GradientTape() as tape:
            conv, pred = grad_model(xb, training=False)
            p = pred[:, 0]
            # build_samples.py와 같이 "예측한 쪽" 점수 기준. 샘플끼리 독립이라 합해서 미분해도 된다
            score = tf.reduce_sum(tf.where(p >= 0.5, p, 1.0 - p))
        grads = tape.gradient(score, conv)
        w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(conv * w, axis=-1)).numpy()
        cam /= np.maximum(cam.max(axis=(1, 2), keepdims=True), 1e-8)
        probs.append(p.numpy())
        cams.append(cam)
    return np.concatenate(probs), np.concatenate(cams)


def peak_box(cam, size=384, crop=128):
    big = np.array(Image.fromarray(np.uint8(cam * 255)).resize((size, size), Image.BILINEAR))
    y, x = np.unravel_index(np.argmax(big), big.shape)
    l, t = int(np.clip(x - crop // 2, 0, size - crop)), int(np.clip(y - crop // 2, 0, size - crop))
    return l, t, l + crop, t + crop


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) * 2 - inter)


def main():
    meta = [json.loads(l) for l in open(f'{WORK}/meta.jsonl', encoding='utf-8')]
    groups = defaultdict(list)
    for m in meta:
        groups[m['generator']].append(m)
    fakes = sorted(g for g in groups if g != 'Real')
    per = max(1, (N // 2) // len(fakes))
    meta = groups['Real'][:N // 2] + [m for g in fakes for m in groups[g][:per]]
    x = np.stack([np.asarray(Image.open(f"{WORK}/std/{m['id']}.jpg").convert('RGB')
                             .resize((224, 224), Image.NEAREST), np.float32) / 255.0 for m in meta])
    res = {k: run(p, x) for k, p in MODELS.items()}

    out = {}
    y_real = np.array([m['label'] == 'REAL' for m in meta])
    gens = np.array([m['generator'] for m in meta])
    meta_prob = np.array([m['mobilevit_prob_real'] for m in meta])
    for k, (p, _) in res.items():
        acc_as_real = (p >= 0.5) == y_real   # 출력이 "진짜일 확률"이라고 볼 때
        out[k] = {
            'acc_if_output_is_P(real)': round(float(acc_as_real.mean()), 4),
            'acc_if_output_is_P(fake)': round(float((~acc_as_real).mean()), 4),
            'mean_prob_real_imgs': round(float(p[y_real].mean()), 4),
            'mean_prob_fake_imgs': round(float(p[~y_real].mean()), 4),
            'per_generator_acc_if_P(real)': {g: round(float(acc_as_real[gens == g].mean()), 3) for g in sorted(set(gens))},
            'max_abs_diff_vs_meta': round(float(np.abs(p - meta_prob).max()), 4),
        }

    po, co = res['old']
    pn, cn = res['new']
    flat_o, flat_n = co.reshape(len(co), -1), cn.reshape(len(cn), -1)
    corr = [float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0 for a, b in zip(flat_o, flat_n)]
    ious = [iou(peak_box(a), peak_box(b)) for a, b in zip(co, cn)]
    out['old_vs_new'] = {
        'pred_agreement_raw': round(float(((po >= 0.5) == (pn >= 0.5)).mean()), 4),
        'heatmap_pearson_mean': round(float(np.mean(corr)), 3),
        'heatmap_pearson_median': round(float(np.median(corr)), 3),
        'crop_box_iou_mean': round(float(np.mean(ious)), 3),
        'crop_box_same_ratio(iou>0.5)': round(float(np.mean(np.array(ious) > 0.5)), 3),
        'cam_grid': list(co.shape[1:]),
    }
    os.makedirs(f'{WORK}/compare', exist_ok=True)
    with open(f'{WORK}/compare/summary.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(f'{WORK}/compare/per_sample.jsonl', 'w', encoding='utf-8') as f:
        for i, m in enumerate(meta):
            f.write(json.dumps({'id': m['id'], 'label': m['label'], 'generator': m['generator'],
                                'old_prob': round(float(po[i]), 4), 'new_prob': round(float(pn[i]), 4),
                                'heatmap_corr': round(corr[i], 3), 'crop_iou': round(ious[i], 3)}) + '\n')
    np.save(f'{WORK}/compare/cams_new.npy', cn)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
