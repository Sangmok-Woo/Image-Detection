"""증류된 Qwen2-VL 학생을 상주시켜 Streamlit 앱에 평가문을 스트리밍한다.

torch 환경(로컬 venv-train 또는 Colab)에서 돈다. 앱은 TensorFlow venv라 모델을 직접 못
올리므로 HTTP로 붙는다. 모델은 서버가 뜰 때 한 번만 올린다.

    venv-train\\Scripts\\python.exe distill\\vlm_server.py [--port 8502] [--quant auto|4bit|none]

GET  /health    {"ready": bool, "device": ..., "adapter": ..., "last": {...}}
POST /generate  {"images": {"orig": <b64>, "cam": <b64>, "crop": <b64>}, "meta": {...}}
                → 생성문을 글자가 나오는 대로 흘려보낸다 (text/plain, chunked)

이미지를 요청에 직접 싣기 때문에 서버가 다른 컴퓨터(Colab, AWS)에 있어도 된다.

정밀도(--quant auto): GPU 메모리가 12GB 이상이면 양자화 없이 fp16/bf16(작은 배치에선
4비트보다 빠르다), 그보다 작으면 학습 때와 같은 4비트, GPU가 없으면 CPU bf16.
QLoRA 어댑터는 양자화하지 않은 베이스에 붙여도 그대로 쓸 수 있다.
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_dataset import user_text
from train_qlora import MODEL_ID, build_messages

HERE = os.path.dirname(os.path.abspath(__file__))
# 5에포크판(final)이 없으면 3에포크판을 쓴다. test 정확도는 둘 다 78%로 같다.
ADAPTERS = [os.path.join(HERE, 'qwen2vl-distill', d) for d in ('final', 'final-3ep')]
# 학습 때 이미지 한 변. 다르면 학생이 못 보던 토큰 수가 들어간다.
SIDE = 336
# 학습 데이터와 같은 순서: 원본, 히트맵, 크롭
ORDER = ('orig', 'cam', 'crop')

state = {'ready': False, 'device': None, 'adapter': None, 'last': None}
lock = threading.Lock()  # 한 번에 한 장만 생성한다


def pick_adapter(explicit=None):
    for d in ([explicit] if explicit else ADAPTERS):
        f = os.path.join(d, 'adapter_model.safetensors')
        # LFS를 못 받으면 130바이트짜리 포인터 파일만 남는다
        if os.path.exists(f) and os.path.getsize(f) > 1_000_000:
            return d
    sys.exit(f'어댑터가 없다: {explicit or ADAPTERS}')


def load(args):
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    if torch.cuda.is_available():
        dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(
            args.dtype, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        quant = args.quant if args.quant != 'auto' else ('none' if vram >= 12 else '4bit')
        name = torch.cuda.get_device_name(0)
        if quant == '4bit':
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID, quantization_config=bnb, dtype=dtype, device_map='auto')
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype, device_map='cuda')
        state['device'] = f'{name} {quant if quant == "4bit" else str(dtype).split(".")[-1]}'
    else:
        torch.set_num_threads(args.threads)
        model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        state['device'] = f'cpu bf16 ({args.threads} threads)'
        quant = 'none'

    adapter = pick_adapter(args.adapter)
    model = PeftModel.from_pretrained(model, adapter)
    if quant != '4bit':
        # 어댑터를 가중치에 합친다. 안 합치면 토큰마다 LoRA 모듈 196개를 따로 거쳐 느리다.
        model = model.merge_and_unload()
    model = model.eval()
    px = SIDE * SIDE
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=px, max_pixels=px)
    processor.tokenizer.padding_side = 'left'
    state['adapter'] = os.path.basename(adapter.rstrip('/\\'))
    return model, processor


def decode_images(images):
    return [Image.open(io.BytesIO(base64.b64decode(images[k]))).convert('RGB') for k in ORDER]


class Stop:
    """클라이언트가 끊으면 생성을 멈춘다 (Streamlit rerun 등)."""
    def __init__(self):
        self.flag = False

    def __call__(self, *a, **k):
        return self.flag


def make_handler(model, processor, max_new_tokens):
    from transformers import StoppingCriteriaList, TextIteratorStreamer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'  # chunked 전송은 1.1에서만 유효하다

        def log_message(self, fmt, *a):
            print(f'[vlm] {fmt % a}', file=sys.stderr, flush=True)

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                self._json(200, state)
            else:
                self._json(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/generate':
                return self._json(404, {'error': 'not found'})
            req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            ex = {'images': list(ORDER), 'user_text': user_text(req['meta'])}
            prompt = processor.apply_chat_template(build_messages(ex, False), tokenize=False,
                                                   add_generation_prompt=True)
            enc = processor(text=[prompt], images=[decode_images(req['images'])],
                            return_tensors='pt').to(model.device)

            streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True,
                                            skip_special_tokens=True)
            stop = Stop()
            with lock:
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.send_header('Transfer-Encoding', 'chunked')
                # 터널(cloudflared)이 모아서 보내지 않게 한다
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Accel-Buffering', 'no')
                self.end_headers()
                t0, first, n = time.time(), None, 0
                t = threading.Thread(target=self._generate, args=(enc, streamer, stop))
                t.start()
                for piece in streamer:
                    if not piece or stop.flag:
                        continue
                    first = first or time.time() - t0
                    n += len(piece)
                    data = piece.encode('utf-8')
                    try:
                        self.wfile.write(f'{len(data):X}\r\n'.encode() + data + b'\r\n')
                        self.wfile.flush()
                    except OSError:
                        stop.flag = True  # 앱이 연결을 끊었다
                t.join()
                if not stop.flag:
                    self.wfile.write(b'0\r\n\r\n')
                total = time.time() - t0
                state['last'] = {'first_s': round(first or 0, 1), 'total_s': round(total, 1), 'chars': n}
                print(f'[vlm] 생성 완료: 첫 글자 {first or 0:.1f}초, 전체 {total:.1f}초, {n}자',
                      file=sys.stderr, flush=True)

        @torch.no_grad()
        def _generate(self, enc, streamer, stop):
            try:
                model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                               streamer=streamer, stopping_criteria=StoppingCriteriaList([stop]))
            except Exception as e:  # 스트리머가 영원히 기다리지 않게 끝을 알린다
                print(f'[vlm] 생성 실패: {e!r}', file=sys.stderr, flush=True)
                streamer.text_queue.put(f'\n[생성 실패: {e}]')
                streamer.end()

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8502)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--adapter', default=None, help='어댑터 폴더 (기본: final → final-3ep)')
    ap.add_argument('--quant', choices=['auto', '4bit', 'none'], default='auto')
    ap.add_argument('--dtype', choices=['auto', 'bf16', 'fp16'], default='auto',
                    help='GPU 계산 정밀도. T4는 bf16을 에뮬레이션하므로 fp16과 비교해 볼 것')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('VLM_THREADS', 6)))
    ap.add_argument('--max-new-tokens', type=int, default=1800)
    args = ap.parse_args()

    print('[vlm] Qwen2-VL + LoRA 로드 중...', file=sys.stderr, flush=True)
    model, processor = load(args)
    state['ready'] = True
    print(f'[vlm] 준비 완료: {state["device"]}, 어댑터 {state["adapter"]}, '
          f'http://{args.host}:{args.port}', file=sys.stderr, flush=True)
    ThreadingHTTPServer((args.host, args.port),
                        make_handler(model, processor, args.max_new_tokens)).serve_forever()


if __name__ == '__main__':
    main()
