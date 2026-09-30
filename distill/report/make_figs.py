"""보고서 8장에 넣을 그림을 만든다. 색은 dataviz 기본 팔레트(검증 통과)만 쓴다."""
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

REPO, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
font_manager.fontManager.addfont(r'C:\Windows\Fonts\malgun.ttf')
font_manager.fontManager.addfont(r'C:\Windows\Fonts\malgunbd.ttf')
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 11})

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'


def base(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    return fig, ax


def save(fig, name):
    fig.savefig(f'{OUT}/{name}', facecolor=SURF, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


# ── 그림 1: 파이프라인 ───────────────────────────────────────────────────────────
def pipeline():
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_xlim(0, 92); ax.set_ylim(0, 46); ax.axis('off')

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2',
                                    fc='white', ec=color, lw=1.6))
        ax.text(x + w / 2, y + h - 2.6, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color=INK)
        ax.text(x + w / 2, y + h / 2 - 1.6, sub, ha='center', va='center', fontsize=8.6, color=INK2, linespacing=1.5)

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.3))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.4, label, ha='center', fontsize=8, color=INK2)

    ax.text(1, 44.2, '① 증류 데이터 구축', fontsize=10, fontweight='bold', color=INK2)
    box(1, 32, 19, 10, 'Tiny-GenImage', '1,000장 추출\n384×384 표준화', BLUE)
    box(25, 32, 19, 10, 'MobileViT v2', '판정 확률\nGrad-CAM · 크롭', BLUE)
    box(49, 32, 19, 10, '교사 (Claude)', '정답 비공개\n상세 해설 작성', BLUE)
    box(73, 32, 18, 10, '학습 데이터셋', 'train 722 / val 39\ntest 100', BLUE)
    arrow(20.6, 37, 24.4, 37); arrow(44.6, 37, 48.4, 37); arrow(68.6, 37, 72.4, 37)

    ax.text(1, 28.2, '② 학생 학습', fontsize=10, fontweight='bold', color=INK2)
    box(73, 16.5, 18, 10, 'QLoRA 학습', 'Qwen2-VL-2B (4bit)\nLoRA r=16, 5에포크', ORANGE)
    box(49, 16.5, 19, 10, 'LoRA 어댑터', '1,846만 파라미터\n(전체의 0.83%)', ORANGE)
    arrow(82, 31.4, 82, 27.2); arrow(72.4, 21.5, 68.6, 21.5)

    ax.text(1, 12.7, '③ 서비스', fontsize=10, fontweight='bold', color=INK2)
    box(1, 1, 19, 10, '사용자', '이미지 업로드\n결과 확인', AQUA)
    box(25, 1, 19, 10, 'Streamlit 앱', 'MobileViT v2 판정\nGrad-CAM · 크롭', AQUA)
    box(49, 1, 19, 10, 'VLM 추론 서버', '어댑터 병합 (T4 GPU)\n해설 스트리밍', AQUA)
    arrow(20.6, 6, 24.4, 6)
    arrow(44.6, 7.6, 48.4, 7.6); arrow(48.4, 4.2, 44.6, 4.2)
    ax.text(46.5, 9.3, '3장+판정', ha='center', fontsize=7.4, color=INK2)
    ax.text(46.5, 2.1, '해설', ha='center', fontsize=7.4, color=INK2)
    ax.annotate('', xy=(58.5, 11.7), xytext=(58.5, 15.9), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.3, ls='--'))
    ax.text(60, 13.4, '어댑터 적재', fontsize=7.6, color=INK2)
    fig.savefig(f'{OUT}/fig_pipeline.png', facecolor=SURF, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)


# ── 그림 2: 입력 3장 패널 ────────────────────────────────────────────────────────
def panel(sid, box_xy, name):
    d = f'{REPO}/distill/dataset/images'
    ims = [Image.open(f'{d}/{sid}_{k}.jpg').convert('RGB') for k in ('orig', 'cam', 'crop')]
    ImageDraw.Draw(ims[1]).rectangle(box_xy, outline=(255, 255, 255), width=3)
    f = ImageFont.truetype(r'C:\Windows\Fonts\malgunbd.ttf', 17)
    W = 384 * 3 + 24
    p = Image.new('RGB', (W, 384 + 38), 'white')
    dr = ImageDraw.Draw(p)
    for i, (im, t) in enumerate(zip(ims, ['① 원본 (384×384 표준화)', '② Grad-CAM 히트맵 (흰 박스 = 크롭 위치)', '③ 크롭 3배 확대'])):
        p.paste(im, (i * 396, 38))
        dr.rectangle((i * 396, 38, i * 396 + 383, 38 + 383), outline=(195, 194, 183), width=1)
        dr.text((i * 396 + 4, 8), t, fill=(11, 11, 11), font=f)
    p.save(f'{OUT}/{name}')


# ── 그림 3: 학습 곡선 ────────────────────────────────────────────────────────────
def curve():
    tr, ev = [], []
    for log in ('train.log', 'train6.log'):
        t = open(f'{REPO}/distill/qwen2vl-distill/{log}', encoding='utf-8', errors='ignore').read().replace('\r', '\n')
        for m in re.finditer(r"\{'(loss|eval_loss)': '([\d.]+)'.*?'epoch': '([\d.]+)'\}", t):
            (tr if m.group(1) == 'loss' else ev).append((float(m.group(3)), float(m.group(2))))
    tr = sorted(set(tr)); ev = sorted(set(ev))
    fig, ax = base(7.2, 3.6)
    ax.plot(*zip(*tr), color=BLUE, lw=2, label='학습 손실')
    ax.plot(*zip(*ev), color=ORANGE, lw=2, marker='o', ms=6, mec=SURF, mew=1.5, label='검증 손실')
    best = min(ev, key=lambda p: p[1])
    ax.annotate(f'최저 {best[1]:.4f} (5에포크)', xy=best, xytext=(best[0] - 1.9, best[1] + 0.33), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=1))
    ax.text(tr[-1][0] + 0.08, tr[-1][1], '학습', color=INK2, fontsize=9.5, va='center')
    ax.text(ev[-1][0] + 0.08, ev[-1][1], '검증', color=INK2, fontsize=9.5, va='center')
    ax.set_xlabel('에포크', color=INK2); ax.set_ylabel('손실', color=INK2)
    ax.set_xlim(0, 6.6); ax.set_ylim(0.5, 2.1)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper right', labelcolor=INK2)
    save(fig, 'fig_loss.png')
    return tr, ev


# ── 그림 4: 정확도 비교 ──────────────────────────────────────────────────────────
def accuracy():
    rows = [json.loads(l) for l in open(f'{REPO}/distill/qwen2vl-distill/test_predictions.jsonl', encoding='utf-8')]
    gens = ['Real', 'ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
    tab = {}
    for g in gens:
        rs = [r for r in rows if r['generator'] == g]
        tab[g] = (len(rs), sum(r['teacher_verdict'] == r['label'] for r in rs),
                  sum(r['student_pred'] == r['label'] for r in rs), sum(r['mobilevit_pred'] == r['label'] for r in rs))
    n = len(rows)
    tot = [sum(tab[g][k] for g in gens) / n * 100 for k in (1, 2, 3)]
    fig, ax = base(6.2, 2.6)
    names = ['교사 (Claude)', '학생 (Qwen2-VL-2B + QLoRA)', 'MobileViT v2']
    ys = [2, 1, 0]
    ax.barh(ys, tot, color=[MUTED, BLUE, MUTED], height=0.5)
    for y, v, nm in zip(ys, tot, names):
        ax.text(v + 1.2, y, f'{v:.0f}%', va='center', fontsize=11, color=INK, fontweight='bold')
    ax.set_yticks(ys); ax.set_yticklabels(names, color=INK2)
    ax.set_xlim(0, 100); ax.set_xlabel('평가 세트 100장 판정 정확도 (%)', color=INK2)
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True); ax.spines['left'].set_visible(False)
    save(fig, 'fig_accuracy.png')
    return tab, tot


# ── 그림 5: 히트맵 유용성 ────────────────────────────────────────────────────────
def heatmap_useful():
    fig, ax = base(6.6, 2.5)
    cats = ['분류기 정답 (664장)', '분류기 오답 (336장)']
    data = {'관련 있음': [26.7, 13.7], '부분적 관련': [40.5, 64.6], '관련 없음': [32.8, 21.7]}
    cols = [BLUE, '#9ec5f4', '#d9d8d2']
    left = [0, 0]
    for (k, v), c in zip(data.items(), cols):
        ax.barh([1, 0], v, left=left, color=c, height=0.5, edgecolor=SURF, linewidth=2, label=k)
        for y, l, w in zip([1, 0], left, v):
            ax.text(l + w / 2, y, f'{w:.1f}%', ha='center', va='center', fontsize=9.5, color=INK)
        left = [a + b for a, b in zip(left, v)]
    ax.set_yticks([1, 0]); ax.set_yticklabels(cats, color=INK2)
    ax.set_xlim(0, 100); ax.set_xticks([])
    for s in ('left', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), labelcolor=INK2)
    save(fig, 'fig_heatmap.png')


# ── 그림 6: 지연 시간 ────────────────────────────────────────────────────────────
def latency():
    fig, ax = base(6.6, 2.6)
    names = ['노트북 CPU', 'Colab T4 (어댑터 미병합)', 'Colab T4 (어댑터 병합)']
    vals = [449, 115, 58]
    ys = [2, 1, 0]
    ax.barh(ys, vals, color=[MUTED, MUTED, BLUE], height=0.5)
    for y, v in zip(ys, vals):
        ax.text(v + 6, y, f'{v}초', va='center', fontsize=11, color=INK, fontweight='bold')
    ax.axvspan(60, 120, color='#cde2fb', alpha=0.5, lw=0)
    ax.text(90, 2.62, '목표 1~2분', ha='center', fontsize=8.5, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels(names, color=INK2)
    ax.set_xlim(0, 520); ax.set_ylim(-0.5, 2.9); ax.set_xlabel('해설 한 건 완성까지 걸린 시간 (초)', color=INK2)
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True); ax.spines['left'].set_visible(False)
    save(fig, 'fig_latency.png')


pipeline()
panel('tg0078', (0, 0, 127, 127), 'fig_panel_tg0078.png')
tr, ev = curve()
tab, tot = accuracy()
heatmap_useful()
latency()
json.dump({'train_last': tr[-1], 'eval': ev, 'gen': tab, 'total': tot}, open(f'{OUT}/figdata.json', 'w'), ensure_ascii=False)
print(json.dumps({'train_last': tr[-1], 'eval': ev, 'gen': tab, 'total': tot}, ensure_ascii=False))
