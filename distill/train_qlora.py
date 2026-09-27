"""4단계: Qwen2-VL-2B-Instruct에 QLoRA로 교사 설명을 증류한다. (Colab T4 이상에서 실행)

Colab 셀 예시:
    !pip install -q "transformers>=4.49" peft bitsandbytes accelerate pillow
    !unzip -q /content/drive/MyDrive/tiny_genimage_distill.zip -d /content
    !python train_qlora.py --data /content/dataset --out /content/drive/MyDrive/qwen2vl-distill
    !python train_qlora.py --data /content/dataset --out /content/drive/MyDrive/qwen2vl-distill --eval-only

학습: train.jsonl, 검증 손실: val.jsonl
평가(--eval-only): test.jsonl에 대해 설명을 생성하고 "판정:" 줄을 정답 라벨과 비교한다.
"""
import argparse
import json
import os
import re

import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration,
                          Trainer, TrainingArguments)

MODEL_ID = 'Qwen/Qwen2-VL-2B-Instruct'
# 384px 이미지가 장당 약 196토큰이 되도록 고정한다 (--side 로 낮추면 토큰·메모리가 준다)
SIDE = 392
# 비전 인코더는 건드리지 않고 언어 모델 쪽만 LoRA를 붙인다
TARGETS = r'^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def build_messages(ex, with_answer=True):
    content = [{'type': 'image'}] * len(ex['images']) + [{'type': 'text', 'text': ex['user_text']}]
    msgs = [{'role': 'user', 'content': content}]
    if with_answer:
        msgs.append({'role': 'assistant', 'content': [{'type': 'text', 'text': ex['assistant_text']}]})
    return msgs


def load_images(ex, root):
    return [Image.open(os.path.join(root, p)).convert('RGB') for p in ex['images']]


class Collator:
    def __init__(self, processor, root):
        self.p, self.root = processor, root
        self.head = processor.tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)

    def __call__(self, batch):
        texts = [self.p.apply_chat_template(build_messages(ex), tokenize=False) for ex in batch]
        images = [load_images(ex, self.root) for ex in batch]
        enc = self.p(text=texts, images=images, return_tensors='pt', padding=True)
        labels = enc['input_ids'].clone()
        labels[enc['attention_mask'] == 0] = -100
        # 답변(assistant) 부분만 손실에 넣는다
        n = len(self.head)
        for row in range(labels.size(0)):
            ids = enc['input_ids'][row].tolist()
            start = max(i for i in range(len(ids) - n + 1) if ids[i:i + n] == self.head) + n
            labels[row, :start] = -100
        enc['labels'] = labels
        return enc


def load_base(args):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=args.dtype)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, quantization_config=bnb, torch_dtype=args.dtype, device_map='auto')
    px = args.side * args.side
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=px, max_pixels=px)
    processor.tokenizer.padding_side = 'right'
    return model, processor


def train(args):
    train_rows = read_jsonl(f'{args.data}/train.jsonl')
    # transformers 5.x는 warmup_ratio를 없애고 warmup_steps만 받는다
    total_steps = max(1, int(len(train_rows) * args.epochs / 8))
    warmup = max(1, round(total_steps * 0.05))
    print(f'총 {total_steps}스텝 / 워밍업 {warmup}스텝', flush=True)
    model, processor = load_base(args)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=TARGETS, task_type='CAUSAL_LM'))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
        lr_scheduler_type='cosine', warmup_steps=warmup, logging_steps=10,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='eval_loss',
        bf16=args.dtype == torch.bfloat16, fp16=args.dtype == torch.float16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        remove_unused_columns=False, dataloader_num_workers=args.workers, report_to='none')
    trainer = Trainer(model=model, args=targs, data_collator=Collator(processor, args.data),
                      train_dataset=train_rows, eval_dataset=read_jsonl(f'{args.data}/val.jsonl'))
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(f'{args.out}/final')
    processor.save_pretrained(f'{args.out}/final')


@torch.no_grad()
def evaluate(args):
    model, processor = load_base(args)
    model = PeftModel.from_pretrained(model, f'{args.out}/final').eval()
    processor.tokenizer.padding_side = 'left'
    test = read_jsonl(f'{args.data}/test.jsonl')
    results, correct, parsed = [], 0, 0
    for ex in test:
        prompt = processor.apply_chat_template(build_messages(ex, False), tokenize=False, add_generation_prompt=True)
        enc = processor(text=[prompt], images=[load_images(ex, args.data)], return_tensors='pt').to(model.device)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = processor.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        m = re.search(r'판정:\s*(AI 생성|실제 사진)', text)
        pred = None if not m else ('AI' if m.group(1) == 'AI 생성' else 'REAL')
        parsed += pred is not None
        correct += pred == ex['label']
        results.append({'id': ex['id'], 'label': ex['label'], 'generator': ex['generator'],
                        'student_pred': pred, 'mobilevit_pred': ex['mobilevit_pred'],
                        'teacher_verdict': ex['teacher_verdict'], 'student_text': text})
        print(ex['id'], ex['label'], pred, flush=True)
    n = len(test)
    summary = {
        'n': n, 'format_ok': parsed / n,
        'student_acc': correct / n,
        'teacher_acc': sum(r['teacher_verdict'] == r['label'] for r in results) / n,
        'mobilevit_acc': sum(r['mobilevit_pred'] == r['label'] for r in results) / n,
    }
    with open(f'{args.out}/test_predictions.jsonl', 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(f'{args.out}/test_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(summary)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default=MODEL_ID)
    ap.add_argument('--epochs', type=float, default=3)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--resume', nargs='?', const=True, default=None,
                    help='체크포인트에서 이어서 학습한다. 경로를 주거나, 빈 값이면 최신 것')
    ap.add_argument('--side', type=int, default=SIDE,
                    help='이미지 한 변(px). 392=장당 196토큰. 낮추면 VRAM이 준다')
    ap.add_argument('--workers', type=int, default=0 if os.name == 'nt' else 2,
                    help='DataLoader 워커. 윈도우는 spawn 비용 때문에 0이 낫다')
    ap.add_argument('--max-new-tokens', type=int, default=1800,
                    help='상세 설명은 1200~1600토큰이라 400이면 잘린다')
    args = ap.parse_args()
    # T4는 bf16을 지원하지 않는다
    args.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    evaluate(args) if args.eval_only else train(args)
