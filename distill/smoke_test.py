"""학습 전 점검: 모델이 4비트로 올라가는지, 실제 샘플 한 건의 forward+backward가
이 GPU에 들어가는지, 시퀀스가 몇 토큰인지 잰다.

    venv-train\\Scripts\\python.exe distill\\smoke_test.py --data distill/dataset [--side 392]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_qlora as T


def gb(x):
    return f'{x / 1024 ** 3:.2f}GB'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='distill/dataset')
    ap.add_argument('--model', default=T.MODEL_ID)
    ap.add_argument('--side', type=int, default=T.SIDE)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--n', type=int, default=3, help='몇 건을 흘려볼지')
    args = ap.parse_args()
    args.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    free, total = torch.cuda.mem_get_info()
    print(f'GPU {torch.cuda.get_device_name(0)}  전체 {gb(total)}  여유 {gb(free)}')
    print(f'dtype {args.dtype}  side {args.side}px\n')

    print('모델 로드 중...', flush=True)
    model, processor = T.load_base(args)
    torch.cuda.synchronize()
    print(f'  로드 후 점유 {gb(torch.cuda.memory_allocated())}'
          f'  여유 {gb(torch.cuda.mem_get_info()[0])}')

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=T.TARGETS, task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()

    rows = T.read_jsonl(f'{args.data}/train.jsonl')
    # 가장 긴 것부터 본다. 여기서 통과하면 나머지는 통과한다
    rows.sort(key=lambda r: -len(r['assistant_text']))
    coll = T.Collator(processor, args.data)

    torch.cuda.reset_peak_memory_stats()
    times = []
    for i, ex in enumerate(rows[:args.n]):
        t0 = time.time()
        batch = coll([ex])
        n_tok = batch['input_ids'].shape[1]
        n_sup = int((batch['labels'] != -100).sum())
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)
        print(f'  [{i + 1}] {ex["id"]}  {n_tok}토큰 (학습대상 {n_sup})  '
              f'loss {out.loss.item():.3f}  peak {gb(torch.cuda.max_memory_allocated())}  {dt:.2f}s')

    steady = times[2:] or times          # 앞 두 건은 워밍업이라 뺀다
    per = sum(steady) / len(steady)
    print(f'\n최대 점유 {gb(torch.cuda.max_memory_allocated())}'
          f'  남은 여유 {gb(torch.cuda.mem_get_info()[0])}')
    print(f'샘플당 {per:.2f}s  ->  722건 x 3epoch = {per * 722 * 3 / 60:.0f}분')


if __name__ == '__main__':
    main()
