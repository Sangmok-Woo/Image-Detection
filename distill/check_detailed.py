"""상세 설명 배치를 검증하고 통과분을 백업한다.

    python distill/check_detailed.py [백업경로]

검사 항목: JSON 파싱 / id 순서가 배치 파일과 일치 / 필수 키 / 금지 키(정답 누출 방지) /
heatmap_useful 값 / 다른 배치와 설명 문자열이 겹치는 교차오염.
병렬 에이전트가 임시 파일명을 공유해 서로 덮어쓴 적이 있어 교차오염 검사를 넣었다.
"""
import collections
import json
import os
import shutil
import sys

WORK = 'distill/work'
REQ = ['id', 'scene', 'evidence_ai', 'evidence_real', 'heatmap_focus_detailed',
       'heatmap_useful', 'heatmap_comment', 'reasoning', 'uncertainty', 'explanation_detailed']
BAN = ['teacher_verdict', 'teacher_confidence', 'observations']
USEFUL = ('yes', 'partial', 'no')

backup = sys.argv[1] if len(sys.argv) > 1 else None
if backup:
    os.makedirs(backup, exist_ok=True)

seen, tot = {}, collections.Counter()
ok, pending, failed = [], [], []
for i in range(20):
    nb = f'{i:02d}'
    path = f'{WORK}/teacher_detailed/dbatch_{nb}.jsonl'
    if not os.path.exists(path):
        pending.append(nb)
        continue
    with open(f'{WORK}/dbatches/dbatch_{nb}.jsonl', encoding='utf-8') as f:
        want = [json.loads(l)['id'] for l in f if l.strip()]
    rows, err = [], 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                err += 1
    if [r['id'] for r in rows] != want:
        (pending if not err and len(rows) < len(want) else failed).append(f'{nb}({len(rows)}줄)')
        continue
    miss = sum(1 for r in rows for k in REQ if k not in r)
    ban = sum(1 for r in rows for k in BAN if k in r)
    bad = sum(1 for r in rows if r.get('heatmap_useful') not in USEFUL)
    # 같은 설명이 다른 배치에도 있으면 임시 파일이 섞인 것이다
    dupx = [r['id'] for r in rows if r['explanation_detailed'] in seen]
    for r in rows:
        seen[r['explanation_detailed']] = r['id']
    if err or miss or ban or bad or dupx:
        failed.append(f'{nb}(오류{err} 누락{miss} 금지{ban} 값{bad} 중복{len(dupx)})')
        continue
    ok.append(nb)
    tot += collections.Counter(r['heatmap_useful'] for r in rows)
    if backup:
        shutil.copyfile(path, f'{backup}/dbatch_{nb}.jsonl')

n = sum(tot.values())
print(f'검증통과 {len(ok)}배치 / {n}장 :', ' '.join(ok))
if pending:
    print('작성중   :', ' '.join(pending))
if failed:
    print('실패     :', ' '.join(failed))
if n:
    print('  히트맵 :', '  '.join(f'{k} {tot[k]} ({tot[k] / n:.1%})' for k in USEFUL))
