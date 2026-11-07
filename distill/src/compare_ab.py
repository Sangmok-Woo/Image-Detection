"""9.3절 실험: A안(언어 모델 LoRA)과 B안(언어 모델 + 프로젝터 LoRA)의 test 채점 결과를 비교한다.

    python distill/src/compare_ab.py
    python distill/src/compare_ab.py --a distill/models/qwen2vl-A-lm/test_predictions.jsonl --b distill/models/qwen2vl-B-proj/test_predictions.jsonl
"""
import argparse
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(os.path.dirname(HERE), 'models')  # distill/models


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return {r['id']: r for r in map(json.loads, f) if r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default=os.path.join(MODELS, 'qwen2vl-A-lm', 'test_predictions.jsonl'))
    ap.add_argument('--b', default=os.path.join(MODELS, 'qwen2vl-B-proj', 'test_predictions.jsonl'))
    args = ap.parse_args()
    a, b = read_jsonl(args.a), read_jsonl(args.b)
    ids = sorted(a.keys() & b.keys())

    rows = defaultdict(lambda: [0, 0, 0, 0, 0])  # 장수, A, B, 교사, MobileViT
    for i in ids:
        ra, rb = a[i], b[i]
        g = 'real' if ra['label'] == 'REAL' else ra['generator']
        for k, ok in enumerate((1, ra['student_pred'] == ra['label'], rb['student_pred'] == rb['label'],
                                ra['teacher_verdict'] == ra['label'], ra['mobilevit_pred'] == ra['label'])):
            rows[g][k] += ok
            rows['합계'][k] += ok

    print(f'| 출처 | A안 | B안 | 교사 | MobileViT |\n|---|---|---|---|---|')
    for g in sorted(rows, key=lambda g: (g == '합계', g == 'real', g)):
        n, *c = rows[g]
        print(f'| {g} | ' + ' | '.join(f'{x}/{n}' for x in c) + ' |')

    up = [i for i in ids if a[i]['student_pred'] != a[i]['label'] and b[i]['student_pred'] == b[i]['label']]
    down = [i for i in ids if a[i]['student_pred'] == a[i]['label'] and b[i]['student_pred'] != b[i]['label']]
    print(f'\nA→B 판정 변화: 개선 {len(up)}건, 악화 {len(down)}건')
    tag = lambda xs: ', '.join('%s(%s)' % (i, a[i]['generator']) for i in xs)
    print(f'  개선: {tag(up)}')
    print(f'  악화: {tag(down)}')
    for name, d in (('A', a), ('B', b)):
        preds = [d[i]['student_pred'] for i in ids]
        print(f'{name} 예측 분포: REAL {preds.count("REAL")} / AI {preds.count("AI")} / 파싱 실패 {preds.count(None)}')


if __name__ == '__main__':
    main()
