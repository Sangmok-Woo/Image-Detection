"""신모델 재생성용 배치: 새 분류기 결과 + 보존할 교사 판정·근거를 합친다 (정답 라벨은 뺀다).
merge 모드: teacher_heatmap/의 새 히트맵 설명을 old_model/teacher의 판정·근거와 합쳐 teacher/에 쓴다.

    python distill/make_heatmap_batches.py          # 배치 만들기 (meta.jsonl에 있는 만큼, 100개씩)
    python distill/make_heatmap_batches.py merge    # 합치기
"""
import glob, json, os, sys
WORK = 'distill/work'
SIZE = 100
old_t = {}
for p in glob.glob(f'{WORK}/old_model/teacher/batch_*.jsonl'):
    for l in open(p, encoding='utf-8'):
        t = json.loads(l); old_t[t['id']] = t
meta = [json.loads(l) for l in open(f'{WORK}/meta.jsonl', encoding='utf-8') if l.strip()]

if len(sys.argv) > 1 and sys.argv[1] == 'merge':
    new_h = {}
    for p in glob.glob(f'{WORK}/teacher_heatmap/hbatch_*.jsonl'):
        for l in open(p, encoding='utf-8'):
            if l.strip():
                h = json.loads(l); new_h[h['id']] = h
    os.makedirs(f'{WORK}/teacher', exist_ok=True)
    for p in glob.glob(f'{WORK}/teacher/*.jsonl'):
        os.remove(p)
    missing = []
    with open(f'{WORK}/teacher/batch_all.jsonl', 'w', encoding='utf-8') as f:
        for m in meta:
            t, h = old_t[m['id']], new_h.get(m['id'])
            if not h:
                missing.append(m['id']); continue
            t = {**t, 'heatmap_focus': h['heatmap_focus'], 'heatmap_useful': h['heatmap_useful'],
                 'explanation': h['explanation'],
                 'agrees_with_classifier': t['teacher_verdict'] == m['mobilevit_pred']}
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    print('merged', len(meta) - len(missing), 'missing', len(missing), missing[:10])
    sys.exit()

os.makedirs(f'{WORK}/hbatches', exist_ok=True)
for b in range(0, len(meta), SIZE):
    chunk = meta[b:b + SIZE]
    if len(chunk) < SIZE and len(meta) < 1000:
        break
    with open(f'{WORK}/hbatches/hbatch_{b // SIZE:02d}.jsonl', 'w', encoding='utf-8') as f:
        for m in chunk:
            t = old_t[m['id']]
            f.write(json.dumps({'id': m['id'], 'mobilevit_prob_real': m['mobilevit_prob_real'],
                                'mobilevit_pred': m['mobilevit_pred'],
                                **{k: t[k] for k in ['teacher_verdict', 'teacher_confidence', 'observations']}},
                               ensure_ascii=False) + '\n')
print(sorted(os.listdir(f'{WORK}/hbatches')))
