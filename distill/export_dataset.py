"""3단계: 파이프라인 결과(meta) + 교사 설명(teacher)을 합쳐 QLoRA용 데이터셋으로 내보낸다.

사용법:
    venv\\Scripts\\python.exe distill\\export_dataset.py

결과 (distill/dataset/):
    images/{id}_orig.jpg, _cam.jpg, _crop.jpg
    train.jsonl / val.jsonl / test.jsonl
    stats.json   교사 정확도, 분류기 정확도, 버린 샘플 수
그리고 distill/tiny_genimage_distill.zip (Colab 업로드용)

train/val은 교사 판정이 정답과 맞은 샘플만 남긴다 (틀린 설명을 외우지 않게).
test는 전부 남긴다 (학생 모델을 정답 라벨로 채점하려고).
"""
import argparse
import glob
import json
import os
import random
import shutil
from collections import Counter, defaultdict

VERDICT_KO = {'REAL': '실제 사진', 'AI': 'AI 생성'}
CONF_KO = {'high': '높음', 'medium': '보통', 'low': '낮음'}
USEFUL_KO = {'yes': '판정 근거와 관련 있음', 'partial': '부분적으로 관련 있음', 'no': '판정 근거와 관련 없음'}
REQUIRED = ['id', 'teacher_verdict', 'teacher_confidence', 'observations',
            'heatmap_focus', 'heatmap_useful', 'agrees_with_classifier', 'explanation']


def user_text(m):
    return (
        '첫 번째 이미지는 원본, 두 번째는 분류기의 Grad-CAM 히트맵, '
        '세 번째는 히트맵이 가장 강한 영역을 확대한 크롭입니다.\n'
        f"분류기(MobileViT v2) 판정: {VERDICT_KO[m['mobilevit_pred']]} "
        f"(실제 사진일 확률 {m['mobilevit_prob_real']:.2f})\n"
        '이 이미지가 AI로 생성됐는지 판단하고 근거를 설명하세요.'
    )


def assistant_text(t):
    lines = [f"판정: {VERDICT_KO[t['teacher_verdict']]} (확신: {CONF_KO[t['teacher_confidence']]})", '근거:']
    lines += [f'- {o}' for o in t['observations']]
    lines.append(f"히트맵: {t['heatmap_focus']} ({USEFUL_KO[t['heatmap_useful']]})")
    lines.append('분류기 판정과 일치' if t['agrees_with_classifier'] else '분류기 판정과 불일치')
    lines.append(f"설명: {t['explanation']}")
    return '\n'.join(lines)


DETAIL_KEYS = ['scene', 'evidence_ai', 'evidence_real', 'heatmap_focus_detailed',
               'heatmap_useful_detailed', 'heatmap_comment', 'reasoning',
               'uncertainty', 'explanation_detailed']


def assistant_text_detailed(t):
    # 장면 → 양쪽 근거 → 히트맵 검증 → 저울질 → 결론. 학생이 중간 과정까지 따라 배우게 한다.
    lines = [f"장면: {t['scene']}", '',
             f"판정: {VERDICT_KO[t['teacher_verdict']]} (확신: {CONF_KO[t['teacher_confidence']]})", '']
    lines.append('AI 생성 쪽 근거:')
    lines += [f'- {o}' for o in t['evidence_ai']] or ['- 뚜렷한 것 없음']
    lines.append('실제 촬영 쪽 근거:')
    lines += [f'- {o}' for o in t['evidence_real']] or ['- 뚜렷한 것 없음']
    lines.append('')
    lines.append(f"히트맵: {t['heatmap_focus_detailed']}")
    lines.append(f"히트맵 평가: {USEFUL_KO[t['heatmap_useful_detailed']]} — {t['heatmap_comment']}")
    lines.append('분류기 판정과 일치' if t['agrees_with_classifier'] else '분류기 판정과 불일치')
    lines.append('')
    lines.append(f"판단 과정: {t['reasoning']}")
    if t.get('uncertainty'):
        lines.append(f"불확실한 점: {t['uncertainty']}")
    lines.append('')
    lines.append(f"설명: {t['explanation_detailed']}")
    return '\n'.join(lines)


def valid_detailed(t):
    return (all(k in t for k in DETAIL_KEYS)
            and t['heatmap_useful_detailed'] in USEFUL_KO
            and isinstance(t['evidence_ai'], list) and isinstance(t['evidence_real'], list)
            and isinstance(t['explanation_detailed'], str) and len(t['explanation_detailed']) >= 80)


def valid(t):
    return (all(k in t for k in REQUIRED)
            and t['teacher_verdict'] in VERDICT_KO
            and t['teacher_confidence'] in CONF_KO
            and t['heatmap_useful'] in USEFUL_KO
            and isinstance(t['observations'], list) and 1 <= len(t['observations']) <= 6
            and isinstance(t['explanation'], str) and len(t['explanation']) >= 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='distill/work')
    ap.add_argument('--out', default='distill/dataset')
    ap.add_argument('--n-test', type=int, default=100)
    ap.add_argument('--n-val', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--detailed', action='store_true',
                    help='teacher_full/의 상세 설명을 assistant_text로 쓴다')
    args = ap.parse_args()
    teacher_dir = 'teacher_full' if args.detailed else 'teacher'
    check = (lambda t: valid(t) and valid_detailed(t)) if args.detailed else valid
    render = assistant_text_detailed if args.detailed else assistant_text

    meta = {}
    with open(f'{args.work}/meta.jsonl', encoding='utf-8') as f:
        for line in f:
            m = json.loads(line)
            meta[m['id']] = m

    teacher, bad = {}, []
    for path in sorted(glob.glob(f'{args.work}/{teacher_dir}/batch_*.jsonl')):
        with open(path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                except json.JSONDecodeError:
                    bad.append(line[:60])
                    continue
                if check(t) and t['id'] in meta:
                    t['agrees_with_classifier'] = t['teacher_verdict'] == meta[t['id']]['mobilevit_pred']
                    teacher[t['id']] = t
                else:
                    bad.append(t.get('id', line[:60]))

    missing = sorted(set(meta) - set(teacher))
    ids = sorted(teacher)

    # 생성기별로 test/val을 고르게 떼어낸다
    rng = random.Random(args.seed)
    by_gen = defaultdict(list)
    for i in ids:
        by_gen[meta[i]['generator']].append(i)
    split = {}
    for need, name in [(args.n_test, 'test'), (args.n_val, 'val')]:
        pool = [i for i in ids if i not in split]
        # 진짜 절반, 가짜는 생성기별 균등
        gens = sorted({meta[i]['generator'] for i in pool})
        fake_gens = [g for g in gens if g != 'Real']
        quota = {'Real': need // 2}
        per, extra = divmod(need - need // 2, len(fake_gens))
        for k, g in enumerate(fake_gens):
            quota[g] = per + (1 if k < extra else 0)
        for g, q in quota.items():
            cands = [i for i in by_gen[g] if i not in split]
            for i in rng.sample(cands, min(q, len(cands))):
                split[i] = name
    for i in ids:
        split.setdefault(i, 'train')

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(f'{args.out}/images')

    rows = defaultdict(list)
    dropped_wrong = Counter()
    for i in ids:
        m, t, s = meta[i], teacher[i], split[i]
        correct = t['teacher_verdict'] == m['label']
        if s != 'test' and not correct:
            dropped_wrong[s] += 1
            continue
        imgs = []
        for src, suf in [('std', 'orig'), ('cam', 'cam'), ('crop', 'crop')]:
            shutil.copy(f'{args.work}/{src}/{i}.jpg', f'{args.out}/images/{i}_{suf}.jpg')
            imgs.append(f'images/{i}_{suf}.jpg')
        u, a = user_text(m), render(t)
        rows[s].append({
            'id': i, 'images': imgs,
            'messages': [{'role': 'user', 'content': '<image><image><image>' + u},
                         {'role': 'assistant', 'content': a}],
            'user_text': u, 'assistant_text': a,
            'label': m['label'], 'generator': m['generator'],
            'mobilevit_prob_real': m['mobilevit_prob_real'], 'mobilevit_pred': m['mobilevit_pred'],
            'teacher_verdict': t['teacher_verdict'], 'teacher_correct': correct,
            'heatmap_useful': t['heatmap_useful'],
            **({'heatmap_useful_detailed': t['heatmap_useful_detailed']}
               if 'heatmap_useful_detailed' in t else {}),
        })

    for s in ['train', 'val', 'test']:
        with open(f'{args.out}/{s}.jsonl', 'w', encoding='utf-8') as f:
            for r in rows[s]:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    def acc(key):
        per = defaultdict(lambda: [0, 0])
        for i in ids:
            g = meta[i]['generator']
            pred = teacher[i]['teacher_verdict'] if key == 'teacher' else meta[i]['mobilevit_pred']
            per[g][0] += pred == meta[i]['label']
            per[g][1] += 1
        total = sum(c for c, _ in per.values()) / max(1, len(ids))
        return {'overall': round(total, 4), **{g: round(c / n, 3) for g, (c, n) in sorted(per.items())}}

    stats = {
        'samples_built': len(meta), 'teacher_valid': len(ids),
        'teacher_invalid': len(bad), 'teacher_missing': len(missing),
        'teacher_accuracy': acc('teacher'), 'mobilevit_accuracy': acc('mobilevit'),
        'heatmap_useful': dict(Counter(teacher[i]['heatmap_useful'] for i in ids)),
        **({'heatmap_useful_detailed':
            dict(Counter(teacher[i]['heatmap_useful_detailed'] for i in ids))}
           if args.detailed else {}),
        'dropped_teacher_wrong': dict(dropped_wrong),
        'split_sizes': {s: len(rows[s]) for s in ['train', 'val', 'test']},
    }
    with open(f'{args.out}/stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if missing:
        print('설명 없는 id (앞 20개):', missing[:20])

    zip_base = os.path.join(os.path.dirname(args.out), 'tiny_genimage_distill')
    shutil.make_archive(zip_base, 'zip', os.path.dirname(args.out), os.path.basename(args.out))
    print('zip:', zip_base + '.zip')


if __name__ == '__main__':
    main()
