"""상세 설명용 배치: 기존 교사 판정·근거 + 분류기 결과를 합친다 (정답 라벨은 뺀다).
merge 모드: teacher_detailed/의 상세 설명을 teacher/batch_all.jsonl의 판정·근거와 합쳐
teacher_full/batch_all.jsonl에 쓴다. 기존 teacher/는 건드리지 않는다.

    python distill/make_detailed_batches.py          # 배치 만들기 (50개씩)
    python distill/make_detailed_batches.py merge    # 합치기
"""
import glob
import json
import os
import sys

WORK = 'distill/work'
SIZE = 50
KEEP = ['teacher_verdict', 'teacher_confidence', 'observations']
DETAIL = ['scene', 'evidence_ai', 'evidence_real', 'heatmap_focus_detailed',
          'heatmap_useful', 'heatmap_comment', 'reasoning', 'uncertainty',
          'explanation_detailed']

meta = {}
with open(f'{WORK}/meta.jsonl', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            m = json.loads(line)
            meta[m['id']] = m

base = {}
with open(f'{WORK}/teacher/batch_all.jsonl', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            t = json.loads(line)
            base[t['id']] = t

if len(sys.argv) > 1 and sys.argv[1] == 'merge':
    detail = {}
    for path in sorted(glob.glob(f'{WORK}/teacher_detailed/dbatch_*.jsonl')):
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    detail[d['id']] = d
    os.makedirs(f'{WORK}/teacher_full', exist_ok=True)
    missing, incomplete = [], []
    with open(f'{WORK}/teacher_full/batch_all.jsonl', 'w', encoding='utf-8') as f:
        for i in sorted(meta):
            t, d = base.get(i), detail.get(i)
            if not t or not d:
                missing.append(i)
                continue
            if any(k not in d for k in DETAIL):
                incomplete.append(i)
                continue
            # 짧은 설명(explanation)과 히트맵 유용성은 상세 쪽 값으로 덮지 않고 둘 다 남긴다
            row = {**t, **{k: d[k] for k in DETAIL},
                   'heatmap_useful_detailed': d['heatmap_useful'],
                   'agrees_with_classifier': t['teacher_verdict'] == meta[i]['mobilevit_pred']}
            row['heatmap_useful'] = t['heatmap_useful']
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print('merged', len(meta) - len(missing) - len(incomplete),
          'missing', len(missing), missing[:10],
          'incomplete', len(incomplete), incomplete[:10])
    sys.exit()

os.makedirs(f'{WORK}/dbatches', exist_ok=True)
ids = sorted(meta)
for b in range(0, len(ids), SIZE):
    chunk = ids[b:b + SIZE]
    with open(f'{WORK}/dbatches/dbatch_{b // SIZE:02d}.jsonl', 'w', encoding='utf-8') as f:
        for i in chunk:
            t = base[i]
            f.write(json.dumps({'id': i,
                                'mobilevit_prob_real': meta[i]['mobilevit_prob_real'],
                                'mobilevit_pred': meta[i]['mobilevit_pred'],
                                **{k: t[k] for k in KEEP}}, ensure_ascii=False) + '\n')
print(len(ids), 'rows ->', len(sorted(os.listdir(f'{WORK}/dbatches'))), 'batches')
