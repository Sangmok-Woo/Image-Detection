"""보고서 9장(프로젝터 어댑터 실험 결과)에 넣을 그림을 만든다. 색·글꼴은 make_figs.py와 같다.

    python distill/src/docs/report/make_figs_ch9.py <저장소> <출력 폴더>
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
font_manager.fontManager.addfont(r'C:\Windows\Fonts\malgun.ttf')
font_manager.fontManager.addfont(r'C:\Windows\Fonts\malgunbd.ttf')
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 11})

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
M = f'{REPO}/distill/models'


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


def eval_losses(*logs):
    ev = {}
    for log in logs:
        t = open(log, encoding='utf-8', errors='ignore').read()
        for m in re.finditer(r"'eval_loss': '([\d.]+)'.*?'epoch': '([\d.]+)'", t):
            ev[float(m.group(2))] = float(m.group(1))
    return sorted(ev.items())


# ── 그림 9-4: 에포크별 검증 손실 ─────────────────────────────────────────────────
def curve():
    a = eval_losses(f'{M}/qwen2vl-A-lm/train.log', f'{M}/qwen2vl-A-lm/train6.log')
    b = eval_losses(f'{M}/qwen2vl-B-proj/train.log')
    fig, ax = base(7.2, 3.4)
    ax.plot(*zip(*a), color=MUTED, lw=2, marker='o', ms=6, mec=SURF, mew=1.5)
    ax.plot(*zip(*b), color=BLUE, lw=2.2, marker='o', ms=6, mec=SURF, mew=1.5)
    ax.text(a[-1][0] + 0.1, a[-1][1], 'A안 (언어 모델)', color=INK2, fontsize=9.5, va='center')
    ax.text(b[-1][0] + 0.1, b[-1][1] - 0.006, 'B안 (+ 프로젝터)', color=BLUE, fontsize=9.5, va='center', fontweight='bold')
    ba, bb = min(a, key=lambda p: p[1]), min(b, key=lambda p: p[1])
    ax.annotate(f'A안 최저 {ba[1]:.3f}', xy=ba, xytext=(ba[0] - 0.4, ba[1] + 0.045), fontsize=9, color=INK2,
                ha='center', arrowprops=dict(arrowstyle='-', color=MUTED, lw=1))
    ax.annotate(f'B안 최저 {bb[1]:.3f}', xy=bb, xytext=(bb[0] - 0.6, bb[1] - 0.04), fontsize=9, color=BLUE,
                ha='center', arrowprops=dict(arrowstyle='-', color=BLUE, lw=1))
    ax.set_xlabel('에포크', color=INK2); ax.set_ylabel('검증 손실', color=INK2)
    ax.set_xlim(0.7, 7.4); ax.set_ylim(0.8, 1.05); ax.set_xticks(range(1, 7))
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    save(fig, 'fig9_loss_ab.png')
    return a, b


# ── 그림 9-5: 출처별 정답률 ──────────────────────────────────────────────────────
def by_source():
    rd = lambda p: [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    A, B = rd(f'{M}/qwen2vl-A-lm/test_predictions.jsonl'), rd(f'{M}/qwen2vl-B-proj/test_predictions.jsonl')
    gens = ['Real', 'ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
    names = ['실제 사진', 'ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD 1.5', 'VQDM', 'Wukong']
    cnt = lambda rows, g: (sum(r['student_pred'] == r['label'] for r in rows if r['generator'] == g),
                           sum(r['generator'] == g for r in rows))
    fig, ax = base(7.6, 3.2)
    w = 0.38
    for i, g in enumerate(gens):
        (ca, n), (cb, _) = cnt(A, g), cnt(B, g)
        ax.bar(i - w / 2, ca / n * 100, w, color='#c9c8c1')
        ax.bar(i + w / 2, cb / n * 100, w, color=BLUE)
        ax.text(i - w / 2, ca / n * 100 + 2, f'{ca}', ha='center', fontsize=8.5, color=INK2)
        ax.text(i + w / 2, cb / n * 100 + 2, f'{cb}', ha='center', fontsize=8.5, color=BLUE, fontweight='bold')
    ax.set_xticks(range(len(gens))); ax.set_xticklabels(names, color=INK2, fontsize=9.5)
    ax.set_ylim(0, 112); ax.set_ylabel('정답률 (%)', color=INK2)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.bar(0, 0, color='#c9c8c1', label='A안'); ax.bar(0, 0, color=BLUE, label='B안')
    ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.12), labelcolor=INK2)
    save(fig, 'fig9_source_ab.png')


if __name__ == '__main__':
    print(curve())
    by_source()
