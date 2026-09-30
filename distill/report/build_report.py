"""1학기 보고서 docx에 8장(자체 VLM 지식 증류)을 넣는다. 그림·코드·데이터 포함판.

    python build_report.py <원본.docx> <결과.docx> <저장소> <그림 폴더> [쪽수 JSON]
"""
import io
import json
import keyword
import os
import re
import sys
import tokenize
import zipfile
from xml.sax.saxutils import escape

from PIL import Image

SRC, DST, REPO, FIG = sys.argv[1:5]
PAGES = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}

zin = zipfile.ZipFile(SRC)
x = zin.read('word/document.xml').decode('utf-8')
rels = zin.read('word/_rels/document.xml.rels').decode('utf-8')
media = {}        # zip 경로 → 바이트
_bid = [9000]
_img = [0]


# ── 문단 조각 ────────────────────────────────────────────────────────────────────
def run(t, bold=False, sz=None, color=None, font=None):
    rpr = ''
    if font:
        rpr += f'<w:rFonts w:ascii="{font}" w:cs="{font}" w:eastAsia="{font}" w:hAnsi="{font}"/>'
    if bold:
        rpr += '<w:b w:val="1"/><w:bCs w:val="1"/>'
    if color:
        rpr += f'<w:color w:val="{color}"/>'
    if sz:
        rpr += f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
    return f'<w:r><w:rPr>{rpr}<w:rtl w:val="0"/></w:rPr><w:t xml:space="preserve">{escape(t)}</w:t></w:r>'


def bookmark(name):
    _bid[0] += 1
    return (f'<w:bookmarkStart w:colFirst="0" w:colLast="0" w:name="{name}" w:id="{_bid[0]}"/>'
            f'<w:bookmarkEnd w:id="{_bid[0]}"/>')


def h2(t, anchor):
    return ('<w:p><w:pPr><w:pStyle w:val="Heading2"/><w:keepNext w:val="0"/><w:pageBreakBefore w:val="1"/>'
            '<w:spacing w:after="80" w:line="360" w:lineRule="auto"/>'
            '<w:rPr><w:b w:val="1"/><w:bCs w:val="1"/><w:sz w:val="34"/><w:szCs w:val="34"/></w:rPr></w:pPr>'
            + bookmark(anchor) + run(t, True, 34) + '</w:p>')


def h3(t, anchor):
    return ('<w:p><w:pPr><w:pStyle w:val="Heading3"/><w:keepNext w:val="1"/>'
            '<w:spacing w:after="80" w:before="280" w:line="360" w:lineRule="auto"/>'
            '<w:ind w:left="0" w:firstLine="0"/>'
            '<w:rPr><w:b w:val="1"/><w:bCs w:val="1"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr></w:pPr>'
            + bookmark(anchor) + run(t, True, 26) + '</w:p>')


def para(t):
    return ('<w:p><w:pPr><w:spacing w:after="240" w:line="360" w:lineRule="auto"/><w:rPr/></w:pPr>'
            + run(t) + '</w:p>')


def bullet(lead, t):
    return ('<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
            '<w:spacing w:after="0" w:afterAutospacing="0" w:line="360" w:lineRule="auto"/>'
            '<w:ind w:left="720" w:hanging="360"/></w:pPr>'
            + run(lead, True) + run(' ' + t) + '</w:p>')


def spacer():
    return '<w:p><w:pPr><w:spacing w:line="360" w:lineRule="auto"/><w:rPr/></w:pPr></w:p>'


def caption(t):
    return ('<w:p><w:pPr><w:spacing w:before="80" w:after="240" w:line="276" w:lineRule="auto"/>'
            '<w:jc w:val="center"/></w:pPr>' + run(t, False, 18) + '</w:p>')


def table(rows, widths, cap, align=None):
    total = sum(widths)
    border = ''.join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
                     for s in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'))
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/><w:jc w:val="center"/>'
           f'<w:tblBorders>{border}</w:tblBorders><w:tblLayout w:type="fixed"/>'
           '<w:tblCellMar><w:left w:w="100" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar>'
           '</w:tblPr><w:tblGrid>' + ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths) + '</w:tblGrid>']
    for i, r in enumerate(rows):
        out.append('<w:tr><w:trPr><w:cantSplit w:val="1"/></w:trPr>')
        for j, (w, c) in enumerate(zip(widths, r)):
            shd = '<w:shd w:val="clear" w:color="auto" w:fill="EDEDED"/>' if i == 0 else ''
            bold = i == 0 or c.startswith('**')
            jc = 'left' if (align and align[j] == 'l' and i > 0) else 'center'
            out.append(f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}<w:vAlign w:val="center"/></w:tcPr>'
                       '<w:p><w:pPr><w:keepNext w:val="1"/><w:spacing w:before="40" w:after="40" w:line="276" '
                       f'w:lineRule="auto"/><w:jc w:val="{jc}"/></w:pPr>' + run(c.strip('*'), bold, 20) + '</w:p></w:tc>')
        out.append('</w:tr>')
    out.append('</w:tbl>')
    return ''.join(out) + caption(cap)


# ── 코드 블록: 기존 보고서와 같은 어두운 배경 + 구문 색 ─────────────────────────────
C_BASE, C_KW, C_FN, C_STR, C_CMT, C_NUM = 'c9d1d9', 'ff7b72', 'd2a8ff', 'a5d6ff', '8b949e', '79c0ff'


def _code_line(parts, last):
    after = 240 if last else 0
    runs = ''.join(run(t, color=c, sz=19, font='Consolas') for t, c in parts) or run(' ', color=C_BASE, sz=19, font='Consolas')
    keep = '' if last else '<w:keepNext w:val="1"/>'
    return ('<w:p><w:pPr>' + keep + '<w:keepLines w:val="1"/><w:shd w:fill="121314" w:val="clear"/>'
            f'<w:spacing w:before="0" w:after="{after}" w:line="264" w:lineRule="auto"/>'
            '<w:ind w:left="120" w:right="120"/></w:pPr>' + runs + '</w:p>')


def code(src):
    src = src.strip('\n')
    lines = src.split('\n')
    colored = [[] for _ in lines]
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src + '\n').readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        toks = []
    spans = {}
    prev = None
    for tk in toks:
        (r1, c1), (r2, c2) = tk.start, tk.end
        col = None
        if tk.type == tokenize.COMMENT:
            col = C_CMT
        elif tk.type == tokenize.STRING or tk.type in (getattr(tokenize, 'FSTRING_START', -1),
                                                       getattr(tokenize, 'FSTRING_MIDDLE', -1),
                                                       getattr(tokenize, 'FSTRING_END', -1)):
            col = C_STR
        elif tk.type == tokenize.NUMBER:
            col = C_NUM
        elif tk.type == tokenize.NAME:
            if keyword.iskeyword(tk.string):
                col = C_KW
            elif prev in ('def', 'class'):
                col = C_FN
        if tk.type == tokenize.NAME:
            prev = tk.string
        if col and r1 == r2:
            spans.setdefault(r1, []).append((c1, c2, col))
        elif col:  # 여러 줄 문자열
            for r in range(r1, r2 + 1):
                a = c1 if r == r1 else 0
                b = c2 if r == r2 else len(lines[r - 1])
                spans.setdefault(r, []).append((a, b, col))
    for i, ln in enumerate(lines, 1):
        pos = 0
        for a, b, col in sorted(spans.get(i, [])):
            if a > pos:
                colored[i - 1].append((ln[pos:a], C_BASE))
            colored[i - 1].append((ln[a:b], col))
            pos = b
        if pos < len(ln):
            colored[i - 1].append((ln[pos:], C_BASE))
    return ''.join(_code_line(p, i == len(lines) - 1) for i, p in enumerate(colored))


def textblock(t, color=C_BASE):
    """프롬프트·출력문 예시. 코드 블록과 같은 상자에 글만 넣는다."""
    lines = t.strip('\n').split('\n')
    out = []
    for i, ln in enumerate(lines):
        after = 240 if i == len(lines) - 1 else 0
        out.append('<w:p><w:pPr><w:shd w:fill="121314" w:val="clear"/>'
                   f'<w:spacing w:before="0" w:after="{after}" w:line="288" w:lineRule="auto"/>'
                   '<w:ind w:left="120" w:right="120"/></w:pPr>'
                   + run(ln or ' ', color=color, sz=19) + '</w:p>')
    return ''.join(out)


# ── 그림 ─────────────────────────────────────────────────────────────────────────
def figure(path, cap, width_in=6.2):
    global rels
    _img[0] += 1
    n = _img[0]
    name = f'vlm_fig{n}{os.path.splitext(path)[1]}'
    media[f'word/media/{name}'] = open(path, 'rb').read()
    rid = f'rIdVlm{n}'
    rels = rels.replace('</Relationships>',
                        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                        f'relationships/image" Target="media/{name}"/></Relationships>')
    w, h = Image.open(path).size
    cx = int(width_in * 914400)
    cy = int(cx * h / w)
    draw = (f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:docPr id="{900 + n}" name="{name}"/><a:graphic><a:graphicData '
            'uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr>'
            f'<pic:cNvPr id="0" name="{name}"/><pic:cNvPicPr preferRelativeResize="0"/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"/><a:ln/></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>')
    return ('<w:p><w:pPr><w:keepNext w:val="1"/><w:spacing w:before="120" w:after="0" w:line="240" w:lineRule="auto"/>'
            f'<w:jc w:val="center"/></w:pPr><w:r>{draw}</w:r></w:p>' + caption(cap))


# ── 저장소에서 실제 코드 발췌 ────────────────────────────────────────────────────
def src_of(relpath, start, end=None, keep_comments=True):
    """start로 시작하는 줄부터, 들여쓰기가 끝나는 곳(또는 end 직전)까지."""
    lines = open(f'{REPO}/{relpath}', encoding='utf-8').read().split('\n')
    i = next(k for k, l in enumerate(lines) if l.startswith(start))
    out = [lines[i]]
    for l in lines[i + 1:]:
        if end and l.startswith(end):
            break
        if not end and l and not l.startswith((' ', '\t')):
            break
        out.append(l)
    while out and not out[-1].strip():
        out.pop()
    return '\n'.join(out)


def dedent_block(relpath, first, last):
    lines = open(f'{REPO}/{relpath}', encoding='utf-8').read().split('\n')
    i = next(k for k, l in enumerate(lines) if first in l)
    j = next(k for k, l in enumerate(lines) if last in l and k >= i)
    blk = lines[i:j + 1]
    ind = min(len(l) - len(l.lstrip()) for l in blk if l.strip())
    return '\n'.join(l[ind:] for l in blk)


D = json.load(open(f'{FIG}/figdata.json', encoding='utf-8'))
G = D['gen']

S = [
    ('8. 자체 전용 VLM 지식 증류 및 로컬 통합', '_heading=h.vlm8'),
    ('8.1. 설계 변경 배경 및 전체 구조', '_heading=h.vlm81'),
    ('8.2. 증류용 데이터셋 구축', '_heading=h.vlm82'),
    ('8.3. 교사 모델 기반 상세 해설 생성', '_heading=h.vlm83'),
    ('8.4. QLoRA 기반 학생 모델 학습', '_heading=h.vlm84'),
    ('8.5. 성능 평가', '_heading=h.vlm85'),
    ('8.6. Grad-CAM 신뢰성 검증', '_heading=h.vlm86'),
    ('8.7. 추론 파이프라인 및 웹 통합', '_heading=h.vlm87'),
    ('8.8. 한계 및 향후 과제', '_heading=h.vlm88'),
]

B = []
# ── 8 ────────────────────────────────────────────────────────────────────────────
B.append(h2(*S[0]))
B.append(para('7.3절에서 제시한 향후 과제에 따라, 2학기에는 상용 Claude API에 의존하던 자연어 해설 기능을 '
              '자체 학습한 전용 VLM으로 대체하는 작업을 수행하였다. 본 장에서는 지식 증류 데이터셋 구축, '
              'QLoRA 기반 학생 모델 학습, 성능 평가, 그리고 웹 서비스 통합과 실측 결과까지의 전 과정을 기술한다.'))

# ── 8.1 ──────────────────────────────────────────────────────────────────────────
B.append(h3(*S[1]))
B.append(para('1학기 계획은 MobileViT v2의 분류 직전 특징 맵((1, 7, 7, 512))을 프로젝션 레이어로 변환하여 소형 '
              '언어 모델의 임베딩 공간에 직접 결합하는 방식이었다. 그러나 이 방식은 시각-언어 정렬 단계에서 수십만 건 '
              '규모의 쌍 데이터가 필요하며, 수천 건 이하의 데이터로는 언어 모델이 시각 토큰을 의미 있는 정보로 '
              '받아들이도록 학습시키기 어렵다는 한계가 확인되었다.'))
B.append(para('이에 따라 본 연구는 이미 시각-언어 정렬이 완료된 사전학습 VLM인 Qwen2-VL-2B-Instruct를 학생 모델로 '
              '채택하고, MobileViT v2의 판단 정보를 특징 텐서가 아닌 이미지와 텍스트 형태로 전달하는 방식으로 설계를 '
              '변경하였다. 학생 모델에는 원본 이미지, Grad-CAM 히트맵, 히트맵 최고 활성 영역의 확대 크롭 3장과 '
              'MobileViT v2의 판정 및 확률값이 함께 입력된다. 전체 파이프라인은 다음 세 단계로 구성된다.'))
B.append(bullet('교사 해설 생성:', 'Claude가 3장의 이미지를 직접 관찰하고 정답을 모르는 상태에서 판정과 근거를 서술한다.'))
B.append(bullet('학생 증류 학습:', 'Qwen2-VL-2B가 동일한 입력으로부터 교사의 해설을 재현하도록 QLoRA로 미세조정한다.'))
B.append(bullet('서비스 통합:', '학습된 어댑터를 추론 서버에 적재하고 Streamlit 웹 서비스와 연결하여 상용 API 없이 판정과 해설을 출력한다.'))
B.append(spacer())
B.append(figure(f'{FIG}/fig_pipeline.png', '[그림 8-1] 지식 증류 및 서비스 파이프라인 전체 구조'))

# ── 8.2 ──────────────────────────────────────────────────────────────────────────
B.append(h3(*S[2]))
B.append(para('증류 데이터는 공개 데이터셋 Tiny-GenImage에서 실제 사진 500장과 7종 생성기(ADM, BigGAN, GLIDE, '
              'Midjourney, SD 1.5, VQDM, Wukong)의 생성 이미지 500장을 생성기별로 균등하게 추출하여 총 '
              '1,000장으로 구성하였다. 판별 엔진은 Tiny-GenImage로 재학습한 MobileViT v2 신모델을 사용하였다.'))
B.append(table([
    ['구분', 'Real', 'ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD 1.5', 'VQDM', 'Wukong', '합계'],
    ['장수', '500', '72', '72', '72', '71', '71', '71', '71', '**1,000'],
], [840, 740, 720, 860, 760, 1340, 780, 780, 1020, 800], '[표 8-1] 증류 데이터 1,000장의 출처별 구성'))
B.append(para('원본 데이터에서는 실제 사진이 JPEG 직사각형, 생성 이미지가 PNG 정사각형으로 저장되어 있어 파일 형식과 '
              '크기만으로 정답이 드러나는 형식 누수가 존재하였다. 이를 차단하기 위해 모든 이미지를 짧은 변 기준으로 '
              '리사이즈한 뒤 중앙을 384x384로 잘라내고 JPEG로 재인코딩하는 표준화 함수 standardize()를 구현하였다.'))
B.append(code(src_of('distill/build_samples.py', 'def standardize')))
B.append(para('표준화된 이미지마다 MobileViT v2의 판정 확률을 구하고, 예측한 클래스 방향의 점수로 기울기를 계산하는 '
              'gradcam() 함수로 히트맵을 생성하였다. 이어 crop_box() 함수가 히트맵의 최고점을 중심으로 128x128 영역을 '
              '잡아 384로 3배 확대한 크롭을 만든다. 학생 모델은 이 크롭을 통해 분류기가 가장 강하게 반응한 지점을 '
              '세밀하게 관찰할 수 있다.'))
B.append(code(src_of('distill/build_samples.py', 'def gradcam') + '\n\n\n' + src_of('distill/build_samples.py', 'def crop_box')))
B.append(para('그 결과 샘플 하나는 [그림 8-2]와 같이 원본, 히트맵 오버레이, 크롭의 3장과 메타데이터 한 줄로 구성된다. '
              '아래 예시는 Midjourney로 생성된 이미지(tg0078)이며, MobileViT v2는 이를 실제 사진일 확률 0.89로 오판하였다.'))
B.append(figure(f'{FIG}/fig_panel_tg0078.png', '[그림 8-2] 학생 모델 입력 3장 예시 (tg0078, Midjourney 생성 이미지)'))
B.append(code('{"id": "tg0078", "source": "train-00000.parquet#1757", "label": "AI", "generator": "Midjourney",\n'
              ' "mobilevit_prob_real": 0.8938, "mobilevit_pred": "REAL", "crop_box": [0, 0, 128, 128]}'))
B.append(para('교사 해설 가운데 판정이 정답과 일치한 샘플만 학습(722장)과 검증(39장) 세트에 포함하여 학생이 틀린 '
              '설명을 학습하지 않도록 하였으며, 이 과정에서 교사가 오판한 139장(학습 128장, 검증 11장)이 제외되었다. '
              '평가 세트 100장(실제 50장, 생성 50장)은 정답 라벨로 채점하기 위해 교사 판정과 무관하게 전부 유지하였다.'))

# ── 8.3 ──────────────────────────────────────────────────────────────────────────
B.append(h3(*S[3]))
B.append(para('교사 해설은 Claude Code의 서브에이전트 20개가 50장씩 나누어 작성하였다. 각 에이전트는 원본, 히트맵, '
              '크롭을 한 장에 배치한 패널 이미지를 직접 관찰하였으며, 정답 라벨과 메타데이터 파일에는 접근하지 못하도록 '
              '제한하여 교사가 정답을 참고해 설명을 맞추는 경로를 원천 차단하였다. 교사와 학생에게 주어지는 질문은 '
              'user_text() 함수가 생성하며, 분류기의 판정과 확률이 질문에 포함된다.'))
B.append(code(src_of('distill/export_dataset.py', 'def user_text')))
B.append(para('초기 해설은 평균 201자 분량의 3~5문장이어서 학생이 판단의 중간 과정을 학습하기 어려웠다. 이에 장면 '
              '묘사, AI 생성 쪽 근거, 실제 촬영 쪽 근거, 히트맵 검증, 판단 과정, 불확실성, 결론의 순서로 추론 전 과정을 '
              '서술하는 상세 형식을 설계하였으며, 평균 분량은 712자로 약 3.5배 증가하였다. [그림 8-2]의 이미지에 대해 '
              '교사가 작성한 상세 해설의 앞부분은 다음과 같다.'))
B.append(textblock('''장면: 완전히 흰 배경 위에 연한 갈색 치와와 한 마리가 네 다리로 서서 정면을 보고 있는 스튜디오 사진이다. 큰 귀가 양옆으로 벌어져 있고 꼬리는 위로 살짝 말려 올라가 있다.

판정: AI 생성 (확신: 보통)

AI 생성 쪽 근거:
- 털이 전체적으로 매끈한 그림처럼 표현되어 털 한 올의 결이 거의 드러나지 않고 몸의 색이 부드러운 면으로만 이어진다.
- 두 눈이 유리구슬처럼 똑같은 모양으로 반짝이고 표정이 지나치게 정돈되어 있다.
- 흰 배경과 발 밑 그림자가 광고용으로 다듬은 것처럼 깔끔해 촬영 현장의 흔적이 전혀 없다.
- 귀 안쪽의 잔털과 혈관 같은 세부가 없이 연한 분홍 면으로만 채워져 있다.
실제 촬영 쪽 근거:
- 네 다리의 길이와 굵기, 발가락의 형태가 비교적 자연스럽고 좌우가 어긋나지 않는다.
- 몸의 비례와 서 있는 자세가 실제 치와와와 맞는다.

(히트맵 묘사 중략)
히트맵 평가: 판정 근거와 관련 없음 — 최고점이 화면 가장자리이고 크롭도 거의 비어 있어 분류기의 근거를 읽어낼 수 없다. 눈과 털의 표면 처리를 봤어야 한다.
분류기 판정과 불일치
(이하 판단 과정, 불확실한 점, 설명 생략)'''))
B.append(para('병렬 작업 중 일부 에이전트가 동일한 임시 파일명을 사용하여 서로의 결과를 덮어쓰는 문제가 발생하였다. '
              '이를 검출하기 위해 id 순서 대조와 배치 간 교차오염 검사를 수행하는 검증 스크립트를 작성하였고, 최종 '
              '1,000건 중 오염된 항목은 0건임을 확인하였다. 교사의 자기 확신도는 [표 8-2]와 같이 실제 적중률과 잘 '
              '부합하여, 교사가 스스로 불확실하다고 판단한 샘플에서 실제로 오답이 많았다.'))
B.append(table([
    ['교사 확신도', '건수', '적중률'],
    ['높음', '254', '97.6%'],
    ['보통', '545', '86.2%'],
    ['낮음', '201', '63.7%'],
], [2600, 2000, 2000], '[표 8-2] 교사 확신도별 판정 적중률 (1,000장)'))

# ── 8.4 ──────────────────────────────────────────────────────────────────────────
B.append(h3(*S[4]))
B.append(para('학생 모델은 Qwen2-VL-2B-Instruct를 4bit NF4로 양자화한 뒤, 비전 인코더를 제외한 언어 모델의 어텐션 및 '
              'MLP 투영 레이어에만 LoRA를 결합하였다. 대상 모듈은 정규식으로 지정하여 이름에 visual이 포함된 비전 '
              '인코더 쪽 모듈이 선택되지 않도록 하였다.'))
B.append(code(open(f'{REPO}/distill/train_qlora.py', encoding='utf-8').read().split('\n')[25] + '\n'
              + open(f'{REPO}/distill/train_qlora.py', encoding='utf-8').read().split('\n')[26] + '\n\n'
              + dedent_block('distill/train_qlora.py', 'model = prepare_model_for_kbit_training', 'model.print_trainable_parameters()')))
B.append(table([
    ['항목', '설정값'],
    ['베이스 모델', 'Qwen2-VL-2B-Instruct (4bit NF4, 이중 양자화)'],
    ['LoRA', 'rank 16, alpha 32, dropout 0.05'],
    ['학습 파라미터', '18,464,768 / 2,227,450,368 (0.83%)'],
    ['배치', '1 × 그래디언트 누적 8'],
    ['학습률', '2e-4, 코사인 스케줄'],
    ['입력 이미지', '한 변 336px, 샘플당 3장'],
    ['시퀀스 길이', '2,100~2,300 토큰 (학습 대상 1,540~1,592)'],
    ['학습 장비', 'RTX 4060 Ti (VRAM 8GB)'],
], [2400, 5000], '[표 8-3] QLoRA 학습 설정', align='ll'))
B.append(para('학습은 1학기와 동일한 RTX 4060 Ti(VRAM 8GB)에서 수행하였다. 사전 측정 결과 VRAM 여유가 거의 없었으며, '
              '[표 8-4]와 같이 해상도를 392px에서 280px로 낮추어도 0.9GB만 감소하였다. 메모리의 주된 소비원은 이미지 '
              '토큰이 아니라 15.2만 개 어휘에 대한 로짓 텐서로, fp32 변환과 그래디언트를 합쳐 2.7GB 이상을 차지하였다.'))
B.append(table([
    ['이미지 한 변', '392px', '336px (채택)', '280px'],
    ['최대 VRAM', '7.95GB', '**7.46GB', '7.05GB'],
], [2000, 1700, 1900, 1700], '[표 8-4] 입력 해상도별 최대 VRAM 사용량'))
B.append(figure(f'{FIG}/fig_loss.png', '[그림 8-3] 학습 손실과 검증 손실의 변화 (6에포크)', 5.6))
ev = {int(e): v for e, v in D['eval']}
B.append(para(f'[그림 8-3]과 같이 3에포크 시점까지 검증 손실이 단조 감소하여 체크포인트에서 이어 6에포크까지 학습하였다. '
              f'검증 손실은 1에포크 {ev[1]:.4f}에서 5에포크 {ev[5]:.4f}까지 낮아진 뒤 6에포크에서 {ev[6]:.4f}로 반등하였다. '
              f'같은 기간 학습 손실은 {D["train_last"][1]:.2f} 수준까지 계속 하락하여 두 곡선의 간격이 벌어졌으며, 이를 과적합 '
              '시작 지점으로 판단하여 5에포크 가중치를 최적 어댑터로 선정하였다.'))

# ── 8.5 ──────────────────────────────────────────────────────────────────────────
B.append(h3(*S[5]))
B.append(para('평가 세트 100장에 대해 교사, 학생, MobileViT v2의 판정 정확도를 비교하였다. 학생 모델은 출력 형식 준수율 '
              '99%(파싱 실패 1건)를 보였으며, [그림 8-4]와 같이 판정 정확도 78%로 MobileViT v2 단독 대비 11%p 향상되었다.'))
B.append(figure(f'{FIG}/fig_accuracy.png', '[그림 8-4] 평가 세트 100장 판정 정확도', 5.2))
order = ['Real', 'ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
ko = {'Real': '실제 사진', 'SD15': 'SD 1.5'}
rows = [['구분', '장수', '교사', '학생', 'MobileViT v2']]
for g in order:
    n, t, s, m = G[g]
    rows.append([ko.get(g, g), str(n), f'{t}/{n}', f'{s}/{n}', f'{m}/{n}'])
rows.append(['**합계', '**100', f'**{sum(G[g][1] for g in order)}/100', f'**{sum(G[g][2] for g in order)}/100',
             f'**{sum(G[g][3] for g in order)}/100'])
B.append(table(rows, [1900, 1100, 1400, 1400, 1700], '[표 8-5] 출처별 정답 수 (평가 세트, 5에포크 어댑터 기준)'))
B.append(para(f'출처별 결과를 분석한 결과, 학생 모델은 교사를 단순히 복제하지 않았다. MobileViT v2는 ADM, BigGAN, GLIDE를 '
              f'모두 맞히지만 실제 사진은 50장 중 {G["Real"][3]}장만 맞힌 반면, 교사는 실제 사진 {G["Real"][1]}장을 맞히고 '
              f'ADM에서는 8장 중 {G["ADM"][1]}장에 그쳤다. 학생은 입력으로 함께 주어진 분류기 판정을 활용하여 분류기가 강한 '
              f'ADM에서는 {G["ADM"][2]}장을 모두 맞혔고, 분류기가 취약한 실제 사진에서는 교사 쪽으로 기울어 {G["Real"][2]}장을 '
              '맞혔다. 두 판정 근거를 상황에 따라 저울질하는 앙상블 동작이 학습을 통해 나타난 것이다. 반면 GLIDE와 SD 1.5에서는 '
              '교사와 분류기가 모두 맞힌 샘플을 학생이 틀리는 경우가 있어, 저울질이 항상 옳은 방향으로 작동하지는 않았다.'))
B.append(para('한편 3에포크 모델과 5에포크 모델은 검증 손실이 0.910에서 0.881로 개선되었음에도 정확도는 78%로 동일하였다. '
              '검증 손실은 해설 전체의 토큰 확률을 측정하는 반면 정확도는 판정 한 줄만을 반영하므로, 해설이 교사를 더 '
              '닮아가는 것과 판정이 정확해지는 것은 별개의 문제임을 확인하였다.'))

# ── 8.6 ──────────────────────────────────────────────────────────────────────────
B.append(h3(*S[6]))
B.append(para('교사 해설 과정에서 각 샘플의 히트맵이 판정 근거와 관련 있는지를 함께 평가하도록 하여, 1학기에 구현한 '
              'Grad-CAM 모듈의 신뢰성을 정량적으로 검증하였다. 그 결과 히트맵이 판정 근거를 정확히 가리킨 경우는 '
              '1,000장 중 223장(22.3%)에 불과하였고, 위치만 참고할 수 있는 경우가 48.6%, 전혀 관련 없는 경우가 29.1%였다.'))
B.append(figure(f'{FIG}/fig_heatmap.png', '[그림 8-5] 분류기 정오답에 따른 히트맵 유용성 평가 분포', 5.6))
B.append(bullet('오분류 시 신뢰도 저하:', '분류기가 정답을 맞힌 샘플에서는 히트맵 유용 비율이 26.7%였으나, 틀린 샘플에서는 13.7%로 절반 수준이었다. '
                '설명이 가장 필요한 순간에 히트맵의 신뢰도가 가장 낮아진다.'))
B.append(bullet('모서리 편향:', '[표 8-6]과 같이 히트맵 최고점이 이미지 네 모서리에 위치한 비율은 균등분포 기댓값의 2.6배에 달하였다. '
                '[그림 8-2]의 히트맵에서도 열이 피사체가 아닌 화면 네 변을 따라 분포하는 것을 확인할 수 있다.'))
B.append(bullet('평가 기준 민감도:', '동일한 히트맵이라도 억지로 의미를 부여하지 말라는 지침의 유무에 따라 유용 비율이 42.4%에서 22.3%로 달라졌다. '
                '따라서 단일 수치보다 두 값을 함께 제시하는 것이 타당하다.'))
B.append(spacer())
B.append(table([
    ['크롭 위치', '관측 비율', '균등분포 기댓값', '배율'],
    ['한 변에 닿음', '60.1%', '55.9%', '1.1배'],
    ['**모서리 (두 축 모두)', '**29.2%', '**11.3%', '**2.6배'],
], [2600, 1600, 2000, 1200], '[표 8-6] 히트맵 최고점의 위치 편향 (1,000장)'))
B.append(para('이 결과는 히트맵만으로 판별 근거를 제시하는 방식의 한계를 수치로 보여주며, 학생 VLM이 히트맵을 그대로 '
              '따르지 않고 원본과 크롭을 함께 관찰하여 히트맵의 유용성 자체를 평가하도록 학습시킨 설계의 근거가 된다.'))

# ── 8.7 ──────────────────────────────────────────────────────────────────────────
B.append(h3(*S[7]))
B.append(para('MobileViT v2는 TensorFlow 2.15 기반으로 numpy 1.x를 요구하고, Qwen2-VL은 PyTorch 및 최신 transformers '
              '기반으로 numpy 2.x를 요구하여 단일 가상환경에서 공존할 수 없었다. 이를 해결하기 위해 두 모델을 별도의 '
              '가상환경으로 분리하고, 학생 VLM을 HTTP 추론 서버로 상주시키는 구조를 설계하였다.'))
B.append(bullet('판별 단계 (Streamlit, TensorFlow 환경):', '업로드된 이미지를 학습 데이터와 동일한 384 표준화 과정으로 '
                '전처리한 뒤 MobileViT v2 판정, Grad-CAM 오버레이, 확대 크롭을 생성한다. 학습 시와 전처리가 다르면 학생이 '
                '보지 못한 분포가 입력되므로 8.2절의 데이터셋 생성 함수를 그대로 재사용하였다.'))
B.append(bullet('해설 단계 (추론 서버, PyTorch 환경):', '서버 기동 시 베이스 모델과 LoRA 어댑터를 한 번만 적재하고, '
                '요청이 들어오면 3장의 이미지와 판정 정보로 해설을 생성하여 토큰 단위로 스트리밍 전송한다.'))
B.append(bullet('웹 화면:', '상단에는 MobileViT v2의 판정과 신뢰도를, 하단 이중 검증 리포트의 좌측에는 Grad-CAM 히트맵을, '
                '우측에는 학생 VLM의 해설을 생성되는 즉시 순차적으로 출력한다.'))
B.append(spacer())
B.append(para('앱은 이미지 3장을 Base64로 인코딩하여 요청 본문에 직접 담아 전송한다. 파일 경로가 아닌 이미지 자체를 '
              '전달하므로 추론 서버는 같은 PC뿐 아니라 별도의 GPU 장비에서도 구동할 수 있다.'))
B.append(code(src_of('vlm_client.py', 'def stream')))
B.append(para('서버는 어댑터를 적재한 직후 merge_and_unload()로 LoRA 가중치를 베이스 가중치에 병합한다. 병합하지 않으면 '
              '토큰을 하나 생성할 때마다 196개의 어댑터 모듈을 별도로 거쳐야 하기 때문이다. 생성된 토큰은 '
              'TextIteratorStreamer를 통해 조각 단위로 받아 HTTP chunked 전송으로 즉시 내보낸다.'))
B.append(code(dedent_block('distill/vlm_server.py', 'model = PeftModel.from_pretrained(model, adapter)', 'model = model.eval()')
              + '\n\n' + dedent_block('distill/vlm_server.py', 'for piece in streamer:', 'stop.flag = True')))
if os.path.exists(f'{FIG}/fig_app.png'):
    B.append(figure(f'{FIG}/fig_app.png', '[그림 8-6] 웹 서비스 실행 화면 (MobileViT v2 판정, Grad-CAM 히트맵, 학생 VLM 해설)'))
B.append(para('[그림 8-2]의 이미지를 웹 서비스에 입력한 실제 결과는 다음과 같다. MobileViT v2는 실제 사진(확률 0.8964)으로 '
              '오판하였으나, 학생 VLM은 AI 생성으로 판정하여 정답을 맞혔다.'))
B.append(textblock('''[입력] 분류기(MobileViT v2) 판정: 실제 사진 (실제 사진일 확률 0.90)

[학생 VLM 출력]
장면: 완전히 흰 배경 위에 갈색과 흰색이 섞인 작은 개 한 마리가 옆모습으로 서 있다. 뒤쪽에는 뾰족한 턱지가 뻗어 있다.

판정: AI 생성 (확신: 높음)

AI 생성 쪽 근거:
- 뒤쪽 뾰족한 턱지가 뻗어 나온 모양이 실제 개의 턱지와 크게 다르고, 끝이 뭉툭하게 떨어져 어디까지가 지나가야 하는지 구분되지 않는다.
- 눈이 눈꺼풀이 뭉개진 형태로 놓여 있고 눈 주변의 털이 뭉개져 눈의 위치와 그 주변 털의 경계가 흐릿하다.
- 몸통의 털이 한 장씩의 결 없이 매끈하게 이어져 있어, 실제 개의 털이 가지런히 뻗어 있는 모습과 다르다.
실제 촬영 쪽 근거:
- 앞발의 발가락이 개의 발 구조와 맞고 발가락이 뻗어 나간 각도가 자연스럽다.
(이하 히트맵 평가, 판단 과정, 설명 생략. 전체 1,655자)'''))
B.append(para('학생의 해설은 8.3절의 교사 해설과 동일한 구조를 따르며, 털의 결이 드러나지 않고 매끈하게 이어진다는 핵심 '
              '근거도 교사와 일치한다. 다만 꼬리를 가리키는 자리에 존재하지 않는 단어를 사용하는 등 2B 규모 모델의 어휘 '
              '오류가 함께 관찰되었다.'))
B.append(para('GPU가 없는 노트북(Ryzen 5 7530U)에서 추론 서버를 CPU로 구동한 결과, 해설 한 건을 완성하는 데 약 7분 30초가 '
              '소요되어 시연에 적합하지 않았다. 이에 판별 단계는 노트북에 그대로 두고 해설 단계만 Google Colab의 T4 GPU로 '
              '분리하였으며, 두 구간은 HTTPS 터널로 연결하였다. 동일한 이미지 1장에 대한 실측 결과는 [표 8-7]과 같다.'))
B.append(table([
    ['실행 환경', '첫 글자 출력', '해설 완료'],
    ['노트북 CPU (bf16, 6스레드)', '약 2분', '약 7분 30초'],
    ['Colab T4 GPU, 어댑터 미병합', '5.5초', '115초'],
    ['**Colab T4 GPU, 어댑터 병합', '**2.4초', '**58초'],
], [4200, 1800, 1800], '[표 8-7] 실행 환경별 해설 생성 지연 시간 (약 1,650자 분량 기준)'))
B.append(figure(f'{FIG}/fig_latency.png', '[그림 8-7] 실행 환경별 해설 완성 시간', 5.4))
B.append(para('계산 정밀도를 bf16에서 fp16으로 바꾸는 것은 지연 시간에 영향을 주지 않았으며(115초로 동일), 어댑터 병합만으로 '
              '소요 시간이 절반으로 줄었다. 웹 화면 기준으로도 업로드 후 4.5초에 첫 문장이 출력되고 약 60초에 해설이 '
              '완성되어, 목표로 설정한 1~2분 이내의 응답 시간을 달성하였다. 동일한 입력에 대해서는 매번 같은 해설이 '
              '생성되도록 탐욕적 디코딩을 사용하였으며, 반복 측정에서도 56초에 글자 수까지 동일한 결과를 확인하였다.'))
B.append(para('이로써 7.3절에서 제기한 상용 API의 한계 가운데 호출 비용과 외부 모델 의존성 문제가 해소되었으며, 판별, 시각적 '
              '근거 제시, 자연어 해설이 자체 학습한 모델만으로 완결되는 구조를 달성하였다. GPU를 갖춘 장비에서는 전 과정이 '
              '단일 시스템 안에서 구동되고, GPU가 없는 장비에서는 해설 단계에 한해 이미지가 팀이 직접 띄운 추론 서버로 '
              '전송된다. 시연 환경에는 3에포크 시점의 어댑터를 탑재하였으며, 이는 평가 세트 정확도가 5에포크 어댑터와 '
              '동일한 78%이다.'))

# ── 8.8 ──────────────────────────────────────────────────────────────────────────
B.append(h3(*S[8]))
B.append(bullet('형식 누수의 잔존:', '384 표준화로 파일 형식과 크기 차이는 제거하였으나, 원본에 포함된 테두리, 레터박스, 회전 보정 '
                '여백은 픽셀에 남아 재인코딩으로 제거되지 않는다. 8.6절의 모서리 편향 2.6배가 분류기가 이를 활용하고 있다는 '
                '간접 증거이다.'))
B.append(bullet('평가 데이터 중복 가능성:', 'MobileViT v2 신모델을 Tiny-GenImage 학습 분할로 학습하였고 증류 샘플도 같은 분할에서 '
                '추출하였으므로, 분류기 정확도가 실제보다 높게 측정되었을 가능성이 있다.'))
B.append(bullet('해설 순서와 환각:', '학습 해설이 판정을 근거보다 먼저 서술하는 구조여서 추론 과정 학습 효과가 제한될 수 있으며, '
                '2B 규모의 학생 모델은 이미지에 없는 장면 요소나 존재하지 않는 단어를 지어내는 경우가 관찰되었다. 근거를 판정 '
                '앞에 배치한 형식으로의 재학습과 7B급 학생 모델 적용을 후속 과제로 설정한다.'))
B.append(bullet('추론 속도와 구동 환경:', 'GPU가 없는 환경에서는 해설 한 건에 수 분이 소요되어 원격 GPU에 의존해야 하며, 무료 Colab '
                '세션은 일정 시간 사용하지 않으면 종료되고 터널 주소도 매번 바뀐다. 해설 길이 단축과 판정 경계 근처 이미지에만 '
                '해설을 생성하는 선택적 호출 전략으로 개선할 수 있다.'))
B.append(spacer())

NEW = ''.join(B)

# ── 본문 삽입 + Roles 번호 8 → 9 ─────────────────────────────────────────────────
marker = x.index('w:name="_heading=h.gna0rxudwgcs"')
pstart = x.rfind('<w:p ', 0, marker)
x = x[:pstart] + NEW + x[pstart:]
x = x.replace('8. Roles &amp; Responsibilities', '9. Roles &amp; Responsibilities')

# ── 목차 ─────────────────────────────────────────────────────────────────────────
roles_toc = x.index('w:anchor="_heading=h.bn7wxgmt9ry1"')
toc_p_start = x.rfind('<w:p ', 0, roles_toc)
toc_h2 = x[toc_p_start:x.index('</w:p>', roles_toc) + 6]
sub_anchor = x.index('w:anchor="_heading=h.xjei0hnflq"')
toc_h3 = x[x.rfind('<w:p ', 0, sub_anchor):x.index('</w:p>', sub_anchor) + 6]


def toc_entry(tpl, old_anchor, old_text, anchor, text, page):
    e = tpl.replace(f'w:anchor="{old_anchor}"', f'w:anchor="{anchor}"')
    e = e.replace(f'>{old_text}</w:t>', f'>{escape(text)}</w:t>')
    e = re.sub(r'<w:tab/><w:t xml:space="preserve">\d+</w:t>', f'<w:tab/><w:t xml:space="preserve">{page}</w:t>', e)
    return re.sub(r' w14:paraId="[^"]+"', '', e)


pg = lambda a, d: PAGES.get(a, d)
entries = [toc_entry(toc_h2, '_heading=h.bn7wxgmt9ry1', '9. Roles &amp; Responsibilities', S[0][1], S[0][0], pg(S[0][1], 28))]
for t, a in S[1:]:
    entries.append(toc_entry(toc_h3, '_heading=h.xjei0hnflq', '7.3. 종합 결론 및 향후 과제', a, t, pg(a, 28)))
x = x[:toc_p_start] + ''.join(entries) + x[toc_p_start:]

roles_toc = x.index('w:anchor="_heading=h.bn7wxgmt9ry1"')
tail_start = x.rfind('<w:p ', 0, roles_toc)
toc_end = x.index('w:name="_heading=', tail_start)
seg = x[tail_start:toc_end].replace('<w:tab/><w:t xml:space="preserve">27</w:t>',
                                    f'<w:tab/><w:t xml:space="preserve">{pg("roles", 27)}</w:t>')
x = x[:tail_start] + seg + x[toc_end:]

# ── 다시 묶기 ────────────────────────────────────────────────────────────────────
with zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED) as z:
    for i in zin.infolist():
        if i.filename == 'word/document.xml':
            z.writestr(i, x.encode('utf-8'))
        elif i.filename == 'word/_rels/document.xml.rels':
            z.writestr(i, rels.encode('utf-8'))
        else:
            z.writestr(i, zin.read(i.filename))
    for name, data in media.items():
        z.writestr(name, data)
print('ok', len(NEW), 'figures', _img[0])
