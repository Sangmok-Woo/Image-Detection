"""Colab용 VLM 서버 노트북을 만든다. vlm_server.py는 저장소에 아직 안 올라갔으므로 본문에 싣는다."""
import json
import os
import sys

repo = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
server = open(f'{repo}/distill/vlm_server.py', encoding='utf-8').read()


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(True)}


def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': s.strip('\n').splitlines(True)}


cells = [
    md('''
# Qwen2-VL 증류 학생 — Colab GPU 서버

노트북 PC의 Streamlit 앱(MobileViT 판정·히트맵)은 그대로 두고, **해설을 쓰는 Qwen2-VL만 Colab GPU에서** 돌린다.

1. 메뉴 **런타임 → 런타임 유형 변경 → T4 GPU** 를 먼저 고른다
2. 위에서부터 셀을 차례로 실행한다 (**런타임 → 모두 실행** 도 된다)
3. 마지막에 나오는 `https://....trycloudflare.com` 주소를 앱 왼쪽 사이드바의 **VLM 서버 주소**에 붙여 넣는다

무료 Colab은 한동안 아무것도 안 하면 끊긴다. 발표 직전에 켜고, 6번 셀로 한 번 워밍업해 둘 것.
'''),
    md('## 1. GPU 확인'),
    code('''
!nvidia-smi --query-gpu=name,memory.total --format=csv
'''),
    md('## 2. 설치'),
    code('''
!pip install -q -U "transformers>=4.49" peft accelerate
# Colab에 깔린 구버전 torchao 때문에 PEFT가 어댑터를 붙이다 죽는다. 서버는 torchao를 쓰지 않는다.
!pip uninstall -y -q torchao
# clone 하지 않는다. 서버에 필요한 파일 3개는 아래 셀에 들어 있다.
!mkdir -p /content/repo/distill
%cd /content/repo
'''),
    md('## 3. 서버 코드\n저장소의 `distill/` 에서 서버가 쓰는 파일 3개를 그대로 옮겨 적는다.'),
    code('%%writefile /content/repo/distill/export_dataset.py\n' + open(f'{repo}/distill/export_dataset.py', encoding='utf-8').read()),
    code('%%writefile /content/repo/distill/train_qlora.py\n' + open(f'{repo}/distill/train_qlora.py', encoding='utf-8').read()),
    code('%%writefile /content/repo/distill/vlm_server.py\n' + server),
    md('''
## 4. LoRA 어댑터

5에포크 어댑터는 GitHub LFS에 없어서 **3에포크판(test 정확도 동일 78%)** 을 쓴다.

**이 셀을 돌리기 전에** 왼쪽 파일 패널(폴더 아이콘)의 업로드 버튼으로 PC의
`Image-Detection/distill/adapter-3ep.zip` 을 `/content` 에 올려 둔다. 없으면 파일 선택 창이 뜬다.
(조각 `adapter.partNN` 9개와 `adapter_config.json` 을 나눠 올려도 된다. 다 모이면 자동으로 합친다.)
'''),
    code('''
import glob, hashlib, os, shutil, zipfile
ADAPTER = '/content/repo/distill/qwen2vl-distill/final-3ep'
SHA = '1f4d48ef800abd4614150e1decf38fc709f3db7053dcd16cf908264767a89bf4'
os.makedirs(ADAPTER, exist_ok=True)
os.makedirs('/content/up', exist_ok=True)
have = lambda: all(os.path.exists(f'{ADAPTER}/{n}') for n in ('adapter_model.safetensors', 'adapter_config.json'))

def collect():
    # 파일 패널로 올린 것(/content)과 업로드 창으로 올린 것(/content/up)을 모두 본다
    found = lambda pat: sorted(glob.glob(f'/content/{pat}') + glob.glob(f'/content/up/{pat}'))
    for z in found('adapter*.zip'):
        with zipfile.ZipFile(z) as f:
            f.extractall('/content/repo/distill/qwen2vl-distill')
    for c in found('adapter_config*.json')[:1]:
        shutil.copy(c, f'{ADAPTER}/adapter_config.json')
    parts = found('adapter.part??')
    if parts:
        print('조각', len(parts), '/ 9')
    if len(parts) == 9:
        with open(f'{ADAPTER}/adapter_model.safetensors', 'wb') as out:
            for p in parts:
                out.write(open(p, 'rb').read())

collect()
while not have():
    from google.colab import files
    os.chdir('/content/up')
    files.upload()
    collect()
os.chdir('/content/repo')
h = hashlib.sha256(open(f'{ADAPTER}/adapter_model.safetensors', 'rb').read()).hexdigest()
print('어댑터 해시', '일치' if h == SHA else f'불일치! {h}')
'''),
    md('## 5. 서버 + 터널 실행\n모델 로드에 1~2분 걸린다. 마지막 줄의 주소를 앱에 붙여 넣는다.'),
    code('''
import subprocess, time, re, requests
subprocess.run('pkill -f vlm_server.py; pkill -f cloudflared', shell=True)
time.sleep(2)
# T4 실측(2026-09-30): fp16 + 어댑터 병합으로 해설 한 건 약 58초. 병합 전에는 115초였다.
server = subprocess.Popen('exec python distill/vlm_server.py --host 127.0.0.1 --port 8502 --dtype fp16 '
                          '--adapter distill/qwen2vl-distill/final-3ep > /content/server.log 2>&1',
                          shell=True, cwd='/content/repo')
for _ in range(120):
    if server.poll() is not None:  # 서버가 죽었으면 기다리지 말고 로그를 보여 준다
        print(open('/content/server.log').read()[-3000:])
        raise RuntimeError('서버가 뜨지 못했다. 위 로그를 볼 것')
    try:
        if requests.get('http://127.0.0.1:8502/health', timeout=2).json().get('ready'):
            break
    except Exception:
        pass
    time.sleep(5)
print(requests.get('http://127.0.0.1:8502/health').json())

if not os.path.exists('/content/cloudflared'):
    !wget -q -O /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /content/cloudflared
open('/content/tunnel.log', 'w').close()  # 터널이 쓰기 전에 읽어도 죽지 않게 빈 파일을 먼저 만든다
tunnel = subprocess.Popen('/content/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8502 >> /content/tunnel.log 2>&1',
                          shell=True)
URL = None
for _ in range(60):
    m = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', open('/content/tunnel.log').read())
    if m:
        URL = m.group(0); break
    time.sleep(1)
print('\\n앱 사이드바에 붙여 넣을 주소:\\n', URL)
'''),
    md('## 6. 워밍업 + 속도 측정\n회색 더미 이미지로 한 번 생성해 본다. 첫 요청은 CUDA 초기화 때문에 조금 더 느리다.'),
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
    md('## 7. 로그 보기 (문제가 생겼을 때)'),
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
