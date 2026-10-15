"""Colab용 VLM 서버 노트북을 만든다. 서버 코드와 어댑터는 공개 저장소 main에서 받는다."""
import json
import os
import sys

repo = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(True)}


def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': s.strip('\n').splitlines(True)}


RAW = 'https://raw.githubusercontent.com/Sangmok-Woo/Image-Detection/main'
LFS = 'https://media.githubusercontent.com/media/Sangmok-Woo/Image-Detection/main'

cells = [
    md('''
# Qwen2-VL 증류 학생 — Colab GPU 서버

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sangmok-Woo/Image-Detection/blob/main/distill/colab_vlm_server.ipynb)

노트북 PC의 Streamlit 앱(MobileViT 판정·히트맵)은 그대로 두고, **해설을 쓰는 Qwen2-VL만 Colab GPU에서** 돌린다.

1. 런타임 유형이 **T4 GPU** 인지 확인한다 (이 노트북은 T4로 저장돼 있다)
2. **런타임 → 모두 실행** (Ctrl+F9). 올릴 파일은 없다
3. 1번 셀 끝에 나오는 `https://....trycloudflare.com` 주소를 저장소 루트 `vlm_url.txt` 에 적거나 앱 사이드바에 붙여 넣는다

서버 코드와 어댑터는 공개 저장소 main에서 받는다. 서버 코드를 고쳤으면 main에 올린 뒤 실행할 것.
무료 Colab은 한동안 아무것도 안 하면 끊긴다. 발표 직전에 켜고, 2번 셀로 한 번 워밍업해 둘 것.
'''),
    md('## 1. 준비 + 서버 + 터널\n모델 다운로드와 서버 기동을 겹쳐 돌린다. 올릴 파일은 없다.'),
    code('''
import hashlib, os, re, subprocess, time
t_start = time.time()
REPO, D = '/content/repo', '/content/repo/distill'
ADAPTER = f'{D}/qwen2vl-distill/final-3ep'
SHA = '1f4d48ef800abd4614150e1decf38fc709f3db7053dcd16cf908264767a89bf4'
os.makedirs(ADAPTER, exist_ok=True)
subprocess.run('pkill -f vlm_server.py; pkill -f cloudflared', shell=True)

def bg(cmd, log):
    return subprocess.Popen(f'{cmd} > /content/{log}.log 2>&1', shell=True)

# 모델(4.2GB)은 바로 받기 시작한다. 서버가 torch를 처음 import하는 동안(새 VM에서 1분 넘게) 같이 받는다
RAW = \'''' + RAW + '''\'
LFS = \'''' + LFS + '''\'  # LFS 파일(어댑터 가중치)은 raw 주소로 받으면 포인터만 온다
wget = lambda pairs: ' && '.join(f'wget -q -O {dst} {src}' for dst, src in pairs)
jobs = {
    'model': bg("python -c \\"from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2-VL-2B-Instruct')\\"", 'model'),
    'adapter': bg(wget([(f'{ADAPTER}/adapter_model.safetensors', f'{LFS}/distill/qwen2vl-distill/final-3ep/adapter_model.safetensors'),
                        (f'{ADAPTER}/adapter_config.json', f'{RAW}/distill/qwen2vl-distill/final-3ep/adapter_config.json')]), 'adapter'),
    'code': bg(wget([(f'{D}/{f}', f'{RAW}/distill/{f}') for f in ('vlm_server.py', 'export_dataset.py', 'train_qlora.py')]), 'code'),
}
if not os.path.exists('/content/cloudflared'):
    jobs['cloudflared'] = bg('wget -q -O /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'
                             ' && chmod +x /content/cloudflared', 'cloudflared')

# Colab에 이미 맞는 버전이 깔려 있으면 설치를 건너뛴다 (매번 -U 하면 40초).
# 버전은 import 없이 메타데이터로만 본다. import로 확인하면 새 VM에서 그것만 2분 넘게 걸렸다
from importlib.metadata import version, PackageNotFoundError
def ver(pkg):
    try:
        return tuple(int(x) for x in version(pkg).split('.')[:2])
    except PackageNotFoundError:
        return None
need = (ver('transformers') or (0,)) < (4, 49) or not ver('peft') or not ver('accelerate')
if need:
    !pip install -q -U "transformers>=4.49" peft accelerate
# Colab에 깔린 구버전 torchao 때문에 PEFT가 어댑터를 붙이다 죽는다. 서버는 torchao를 쓰지 않는다.
if ver('torchao'):
    !pip uninstall -y -q torchao
print(f'설치 {"함" if need else "건너뜀"}: {time.time() - t_start:.0f}초')

# 서버는 코드·어댑터(수 초)만 오면 띄운다. 모델 다운로드는 기다리지 않는다
for name in ('code', 'adapter'):
    if jobs[name].wait() != 0:
        print(open(f'/content/{name}.log').read()[-2000:])
        raise RuntimeError(f'{name} 다운로드 실패')
h = hashlib.sha256(open(f'{ADAPTER}/adapter_model.safetensors', 'rb').read()).hexdigest()
assert h == SHA, f'어댑터 해시 불일치: {h}'

import requests
# T4 실측(2026-09-30): fp16 + 어댑터 병합으로 해설 한 건 약 58초. 병합 전에는 115초였다.
server = subprocess.Popen('exec python distill/vlm_server.py --host 127.0.0.1 --port 8502 --dtype fp16 '
                          '--adapter distill/qwen2vl-distill/final-3ep > /content/server.log 2>&1',
                          shell=True, cwd=REPO)
# 터널은 모델이 올라가는 동안 같이 연다
if 'cloudflared' in jobs and jobs['cloudflared'].wait() != 0:
    raise RuntimeError('cloudflared 다운로드 실패')
open('/content/tunnel.log', 'w').close()
tunnel = subprocess.Popen('/content/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8502 >> /content/tunnel.log 2>&1',
                          shell=True)
for _ in range(240):
    if server.poll() is not None:  # 서버가 죽었으면 기다리지 말고 로그를 보여 준다
        print(open('/content/server.log').read()[-3000:])
        raise RuntimeError('서버가 뜨지 못했다. 위 로그를 볼 것')
    if jobs['model'].poll() not in (None, 0):
        print(open('/content/model.log').read()[-2000:])
        raise RuntimeError('모델 다운로드 실패')
    try:
        if requests.get('http://127.0.0.1:8502/health', timeout=2).json().get('ready'):
            break
    except Exception:
        pass
    time.sleep(2)
print(f'서버 준비: {time.time() - t_start:.0f}초', requests.get('http://127.0.0.1:8502/health').json())

URL = None
for _ in range(60):
    m = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', open('/content/tunnel.log').read())
    if m:
        URL = m.group(0); break
    time.sleep(1)
# 주소가 나온 직후엔 DNS가 아직 안 퍼져서 바로 요청하면 실패한다 (2번 셀이 그렇게 죽었다)
for _ in range(60):
    try:
        if requests.get(f'{URL}/health', timeout=5).ok:
            break
    except Exception:
        pass
    time.sleep(2)
print(f'\\n전체 {time.time() - t_start:.0f}초. 앱에 넣을 주소:\\n', URL)
'''),
    md('## 2. 워밍업 + 속도 측정\n회색 더미 이미지로 한 번 생성해 본다. 첫 요청은 CUDA 초기화 때문에 조금 더 느리다.'),
    code('''
import base64, io
from PIL import Image
def b64(color):
    buf = io.BytesIO(); Image.new('RGB', (384, 384), color).save(buf, 'JPEG'); return base64.b64encode(buf.getvalue()).decode()
req = {'images': {'orig': b64((128, 128, 128)), 'cam': b64((90, 90, 200)), 'crop': b64((140, 140, 140))},
       'meta': {'mobilevit_pred': 'AI', 'mobilevit_prob_real': 0.3, 'crop_box': [0, 0, 128, 128]}}
t0 = time.time(); first = None; text = ''
with requests.post(f'{URL}/generate', json=req, stream=True, timeout=600) as r:
    r.encoding = 'utf-8'
    for p in r.iter_content(chunk_size=None, decode_unicode=True):
        first = first or time.time() - t0; text += p
print(text[:300], '...')
print(f'\\n터널 경유 — 첫 글자 {first:.1f}초, 전체 {time.time() - t0:.1f}초, {len(text)}자')
print('서버 측정:', requests.get(f'{URL}/health').json()['last'])
'''),
    md('## 3. 로그 보기 (문제가 생겼을 때)'),
    code('''
!tail -20 /content/server.log
!tail -5 /content/tunnel.log
'''),
]

nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                                   'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 0}
out = f'{repo}/distill/colab_vlm_server.ipynb'
json.dump(nb, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(out)
