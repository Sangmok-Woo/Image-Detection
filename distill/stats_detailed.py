"""상세 설명(teacher_full)의 최종 통계. 발표용 수치를 한 번에 뽑는다.

    python distill/stats_detailed.py

정답 라벨(meta.jsonl)은 여기서만 쓴다. 교사·학생 작업 단계에서는 절대 열지 않는다.
"""
import collections
import json

WORK = 'distill/work'


def load(path, key='id'):
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r[key]] = r
    return out


meta = load(f'{WORK}/meta.jsonl')
short = load(f'{WORK}/teacher/batch_all.jsonl')
full = load(f'{WORK}/teacher_full/batch_all.jsonl')
ids = sorted(full)
print(f'샘플 {len(ids)} / 기존 설명 {len(short)} / 상세 설명 {len(full)}\n')

# --- 설명 길이 ---
a = sum(len(short[i]['explanation']) for i in ids) / len(ids)
b = sum(len(full[i]['explanation_detailed']) for i in ids) / len(ids)
print(f'설명 평균 길이: 기존 {a:.0f}자 -> 상세 {b:.0f}자 ({b / a:.1f}배)\n')

# --- 히트맵 유용성: 기존 평가 vs 상세 평가 ---
print('히트맵 유용성 (같은 이미지·같은 히트맵, 평가 지침만 다름)')
for name, get in [('1차(기존)', lambda i: short[i]['heatmap_useful']),
                  ('2차(상세)', lambda i: full[i]['heatmap_useful_detailed'])]:
    c = collections.Counter(get(i) for i in ids)
    n = sum(c.values())
    print(f'  {name}: ' + '  '.join(f'{k} {c[k]:4d} ({c[k] / n:5.1%})' for k in ('yes', 'partial', 'no')))
flip = collections.Counter((short[i]['heatmap_useful'], full[i]['heatmap_useful_detailed']) for i in ids)
print('  이동(1차->2차) 상위:', ', '.join(f'{a}->{b} {n}' for (a, b), n in flip.most_common(5)))

# --- 히트맵 유용성을 정답·판정별로 쪼갠다 ---
print('\n히트맵 유용성 x 정답 라벨')
for lab in ('REAL', 'AI'):
    sub = [i for i in ids if meta[i]['label'] == lab]
    c = collections.Counter(full[i]['heatmap_useful_detailed'] for i in sub)
    n = max(1, len(sub))
    print(f'  {lab:4s}({n:4d}): ' + '  '.join(f'{k} {c[k] / n:5.1%}' for k in ('yes', 'partial', 'no')))
print('\n히트맵 유용성 x 분류기 적중')
for hit, name in [(True, '분류기 맞음'), (False, '분류기 틀림')]:
    sub = [i for i in ids if (meta[i]['mobilevit_pred'] == meta[i]['label']) == hit]
    c = collections.Counter(full[i]['heatmap_useful_detailed'] for i in sub)
    n = max(1, len(sub))
    print(f'  {name}({n:4d}): ' + '  '.join(f'{k} {c[k] / n:5.1%}' for k in ('yes', 'partial', 'no')))

# --- 근거 쏠림: evidence_ai/real 빈 배열이 한쪽 라벨에 몰리면 학생이 지름길을 배운다 ---
print('\n근거 빈 배열 분포 (학생이 "빈칸=정답" 지름길을 배울 위험)')
for side in ('evidence_ai', 'evidence_real'):
    empty = [i for i in ids if not full[i][side]]
    c = collections.Counter(meta[i]['label'] for i in empty)
    v = collections.Counter(full[i]['teacher_verdict'] for i in empty)
    print(f'  {side:14s} 빈 배열 {len(empty):4d}건  정답 REAL {c["REAL"]} / AI {c["AI"]}'
          f'   교사판정 REAL {v["REAL"]} / AI {v["AI"]}')

# --- 정확도 (정답 대비) ---
print('\n정확도')
for name, pred in [('교사(Claude)', lambda i: full[i]['teacher_verdict']),
                   ('MobileViT', lambda i: meta[i]['mobilevit_pred'])]:
    per = collections.defaultdict(lambda: [0, 0])
    for i in ids:
        g = meta[i]['generator']
        per[g][0] += pred(i) == meta[i]['label']
        per[g][1] += 1
    tot = sum(c for c, _ in per.values())
    print(f'  {name:12s} 전체 {tot}/{len(ids)} ({tot / len(ids):.1%})')
    print('      ' + '  '.join(f'{g} {c}/{n}' for g, (c, n) in sorted(per.items())))

# --- 확신도 분포와 그때의 적중률 ---
print('\n교사 확신도별 적중률')
for conf in ('high', 'medium', 'low'):
    sub = [i for i in ids if full[i]['teacher_confidence'] == conf]
    if not sub:
        continue
    hit = sum(1 for i in sub if full[i]['teacher_verdict'] == meta[i]['label'])
    print(f'  {conf:7s} {len(sub):4d}건  적중 {hit / len(sub):.1%}')
