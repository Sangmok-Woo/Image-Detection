"""CIFAKE test 세트로 모델 성능을 측정한다.

사용법:
    venv\Scripts\python.exe evaluate.py --data-dir <test 폴더 경로>

<test 폴더 경로> 아래에 FAKE/ 와 REAL/ 두 하위 폴더가 있어야 한다.
결과는 results/ 폴더에 metrics.json, roc.png, confusion_matrix.png,
worst_mistakes.txt 로 저장된다.
"""
import argparse
import json
import os
import sys

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf

MODEL_PATH = 'MobileViT2_Model/MobileViT2_Model.h5'
IMG_SIZE = (224, 224)


def load_dataset(data_dir, batch_size, limit):
    # FAKE=0, REAL=1. limit이 있으면 두 클래스에서 균등하게 뽑는다.
    file_paths, labels = [], []
    for label, cls in enumerate(['FAKE', 'REAL']):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            sys.exit(f"클래스 폴더가 없습니다: {cls_dir}")
        files = sorted(
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        if limit:
            k = limit // 2
            step = max(1, len(files) // k)
            files = files[::step][:k]
        file_paths += files
        labels += [label] * len(files)

    labels = np.array(labels, dtype=np.float32)

    def load(path, y):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        # app.py의 load_img와 동일하게 nearest 보간으로 리사이즈한다
        img = tf.image.resize(img, IMG_SIZE, method='nearest')
        return tf.cast(img, tf.float32) / 255.0, y

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, file_paths


def roc_auc(y_true, y_score):
    """positive=REAL(1) 기준 ROC 곡선과 AUC를 numpy만으로 계산한다."""
    order = np.argsort(-y_score)
    y = y_true[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / max(tps[-1], 1)
    fpr = fps / max(fps[-1], 1)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='datasets/dataset/test',
                        help='FAKE/, REAL/ 하위 폴더를 가진 test 폴더')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--limit', type=int, default=0,
                        help='빠른 확인용: 이 개수만큼만 평가 (0=전체)')
    parser.add_argument('--out-dir', default='results')
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        sys.exit(f"데이터 폴더가 없습니다: {args.data_dir}")

    model = tf.keras.models.load_model(MODEL_PATH)
    ds, file_paths = load_dataset(args.data_dir, args.batch_size, args.limit)

    probs = model.predict(ds, verbose=1).ravel()  # REAL일 확률
    y_true = np.concatenate([y.numpy().ravel() for _, y in ds]).astype(int)
    y_pred = (probs >= 0.5).astype(int)

    n = len(y_true)
    acc = float((y_pred == y_true).mean())
    # 혼동행렬: 행=실제, 열=예측 (0=FAKE, 1=REAL)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tn, fp = cm[0]   # 실제 FAKE
    fn, tp = cm[1]   # 실제 REAL

    # AI 탐지 관점(positive=FAKE) 지표
    fake_precision = tn / max(tn + fn, 1)
    fake_recall = tn / max(tn + fp, 1)
    real_precision = tp / max(tp + fp, 1)
    real_recall = tp / max(tp + fn, 1)

    fpr, tpr, auc = roc_auc(y_true, probs)

    os.makedirs(args.out_dir, exist_ok=True)

    metrics = {
        'num_images': n,
        'accuracy': round(acc, 4),
        'auc': round(auc, 4),
        'confusion_matrix': {
            'FAKE를 FAKE로': int(tn), 'FAKE를 REAL로': int(fp),
            'REAL을 FAKE로': int(fn), 'REAL을 REAL로': int(tp),
        },
        'fake_precision': round(float(fake_precision), 4),
        'fake_recall': round(float(fake_recall), 4),
        'real_precision': round(float(real_precision), 4),
        'real_recall': round(float(real_recall), 4),
        'data_dir': args.data_dir,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 가장 자신 있게 틀린 사례 (발표용 예시 이미지 후보)
    wrong = np.where(y_pred != y_true)[0]
    confidence_gap = np.abs(probs - 0.5)
    worst = wrong[np.argsort(-confidence_gap[wrong])][:30]
    with open(os.path.join(args.out_dir, 'worst_mistakes.txt'), 'w', encoding='utf-8') as f:
        f.write("실제라벨\t예측확률(REAL)\t파일\n")
        for i in worst:
            label = 'REAL' if y_true[i] == 1 else 'FAKE'
            f.write(f"{label}\t{probs[i]:.4f}\t{file_paths[i]}\n")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (positive = REAL)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'roc.png'), dpi=150)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap='Blues')
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f'{v:,}', ha='center', va='center',
                color='white' if v > cm.max() / 2 else 'black')
    ax.set_xticks([0, 1], ['FAKE', 'REAL'])
    ax.set_yticks([0, 1], ['FAKE', 'REAL'])
    ax.set_xlabel('predicted')
    ax.set_ylabel('actual')
    ax.set_title(f'Confusion Matrix (n={n:,})')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'confusion_matrix.png'), dpi=150)

    print()
    print(f"이미지 수      : {n:,}")
    print(f"정확도         : {acc:.2%}")
    print(f"AUC            : {auc:.4f}")
    print(f"FAKE 정밀/재현 : {fake_precision:.2%} / {fake_recall:.2%}")
    print(f"REAL 정밀/재현 : {real_precision:.2%} / {real_recall:.2%}")
    print(f"결과 저장      : {args.out_dir}/")


if __name__ == '__main__':
    main()
