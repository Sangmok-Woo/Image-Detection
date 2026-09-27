"""이미지 한 장 → MobileViT 판정 + Grad-CAM → 증류된 Qwen2-VL이 한국어 평가문을 쓴다.

torch가 있는 venv에서 돌린다. MobileViT 단계는 TensorFlow가 있는 venv를 subprocess로
불러 쓴다 (numpy 버전이 달라 한 환경에 못 합친다).

    venv-train\\Scripts\\python.exe distill\\predict.py <이미지> [<이미지> ...]

자주 쓰는 옵션:
    --adapter   LoRA 어댑터 (기본 distill/qwen2vl-distill/final)
    --out       중간 산출물을 남길 폴더 (기본 distill/predict_out/<파일명>)
    --tf-python MobileViT를 돌릴 파이썬 (기본 venv/Scripts/python.exe)
    --json      결과를 JSONL로 찍는다
    --keep      orig/cam/crop 이미지를 지우지 않는다 (기본 남김)

프롬프트는 학습 때(export_dataset.py)와 글자 하나까지 같아야 한다. 그래서 user_text를
거기서 그대로 가져다 쓴다.
"""
import argparse
import json
import os
import re
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_dataset import CONF_KO, USEFUL_KO, VERDICT_KO, user_text
from train_qlora import MODEL_ID, SIDE, build_messages, load_base, load_images

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_TF_PY = os.path.join(ROOT, 'venv', 'Scripts', 'python.exe')
DEFAULT_ADAPTER = os.path.join(HERE, 'qwen2vl-distill', 'final')


def prepare(image, out, tf_python, panel=False):
    """MobileViT 판정 + Grad-CAM + 크롭. TF venv를 따로 띄운다."""
    cmd = [tf_python, os.path.join(HERE, 'predict_prepare.py'), image, '--out', out]
    if panel:
        cmd.append('--panel')
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', env=env)
    if r.returncode != 0:
        raise RuntimeError(f'MobileViT 단계 실패 ({tf_python}):\n{r.stderr[-2000:]}')
    with open(f'{out}/meta.json', encoding='utf-8') as f:
        return json.load(f)


def parse(text):
    """생성문에서 판정·확신·히트맵 평가를 뽑는다. 못 뽑으면 None."""
    ko2en = {v: k for k, v in VERDICT_KO.items()}
    m = re.search(r'판정:\s*(실제 사진|AI 생성)', text)
    c = re.search(r'확신:\s*(높음|보통|낮음)', text)
    h = re.search(r'히트맵 평가:\s*([^\n—-]+)', text)
    conf = {v: k for k, v in CONF_KO.items()}
    useful = {v.strip(): k for k, v in USEFUL_KO.items()}
    return {
        'verdict': ko2en.get(m.group(1)) if m else None,
        'confidence': conf.get(c.group(1)) if c else None,
        'heatmap_useful': useful.get(h.group(1).strip()) if h else None,
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+')
    ap.add_argument('--adapter', default=DEFAULT_ADAPTER)
    ap.add_argument('--model', default=MODEL_ID)
    ap.add_argument('--out', default=None)
    ap.add_argument('--tf-python', default=DEFAULT_TF_PY)
    ap.add_argument('--side', type=int, default=SIDE)
    ap.add_argument('--max-new-tokens', type=int, default=1800)
    ap.add_argument('--panel', action='store_true')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    args.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if not os.path.exists(args.tf_python):
        sys.exit(f'TensorFlow용 파이썬이 없다: {args.tf_python}\n--tf-python 으로 지정하라')
    if not os.path.isdir(args.adapter):
        sys.exit(f'어댑터가 없다: {args.adapter}\n학습을 먼저 돌리거나 --adapter 로 지정하라')

    # 1단계: 이미지마다 MobileViT 판정과 히트맵을 만든다 (모델을 한 번만 올리게 먼저 몰아서)
    jobs = []
    for image in args.images:
        stem = os.path.splitext(os.path.basename(image))[0]
        out = args.out or os.path.join(HERE, 'predict_out', stem)
        print(f'[1/2] MobileViT: {os.path.basename(image)}', file=sys.stderr, flush=True)
        meta = prepare(image, out, args.tf_python, args.panel)
        jobs.append((image, out, meta))

    # 2단계: 증류된 학생이 평가문을 쓴다
    print(f'[2/2] Qwen2-VL 로드 중...', file=sys.stderr, flush=True)
    from peft import PeftModel
    model, processor = load_base(args)
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    processor.tokenizer.padding_side = 'left'

    for image, out, meta in jobs:
        ex = {'images': ['orig.jpg', 'cam.jpg', 'crop.jpg'], 'user_text': user_text(meta)}
        prompt = processor.apply_chat_template(build_messages(ex, False), tokenize=False,
                                               add_generation_prompt=True)
        enc = processor(text=[prompt], images=[load_images(ex, out)], return_tensors='pt').to(model.device)
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = processor.decode(gen[0, enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()

        if args.json:
            print(json.dumps({'image': image, **meta, **parse(text), 'text': text}, ensure_ascii=False),
                  flush=True)
        else:
            print(f'\n{"=" * 70}\n{image}')
            print(f'MobileViT: {VERDICT_KO[meta["mobilevit_pred"]]} '
                  f'(실제 사진일 확률 {meta["mobilevit_prob_real"]:.2f})')
            print(f'중간 산출물: {out}')
            print('-' * 70)
            print(text)


if __name__ == '__main__':
    main()
