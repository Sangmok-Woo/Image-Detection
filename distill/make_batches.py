"""2단계 준비: meta.jsonl에서 정답 라벨을 뺀 배치 파일(50개씩)을 만든다. 교사는 이 파일만 본다."""
import json, os, sys
work = sys.argv[1] if len(sys.argv) > 1 else 'distill/work'
size = 50
rows = [json.loads(l) for l in open(f'{work}/meta.jsonl', encoding='utf-8')]
os.makedirs(f'{work}/batches', exist_ok=True)
for b in range(0, len(rows), size):
    path = f'{work}/batches/batch_{b // size:02d}.jsonl'
    chunk = rows[b:b + size]
    if len(chunk) < size and len(rows) < 1000:
        break  # 아직 파이프라인이 도는 중이면 덜 찬 배치는 만들지 않는다
    with open(path, 'w', encoding='utf-8') as f:
        for m in chunk:
            f.write(json.dumps({k: m[k] for k in ['id', 'mobilevit_prob_real', 'mobilevit_pred']}) + '\n')
print(sorted(os.listdir(f'{work}/batches')))
