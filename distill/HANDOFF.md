# 다른 기기로 작업 옮기기

데스크탑(RTX 4060 Ti 8GB)에서 무거운 작업은 다 끝냈다. 노트북에서 이어받는 절차다.

## 무엇을 옮기고 무엇을 다시 만드나

옮길 것은 **261MB**뿐이다. 나머지는 저장소나 인터넷에서 다시 받으면 된다.

| 옮긴다 (Drive) | 크기 | 왜 |
|---|---|---|
| `adapter-5ep/` | 82MB | **학습된 LoRA 어댑터.** 다시 만들려면 GPU로 4시간 |
| `adapter-3ep/` | 82MB | 3에포크판. 비교용이라 없어도 된다 |
| `tiny_genimage_distill.zip` | 98MB | 데이터셋 전체 (이미지 + train/val/test) |
| `results/` | 1MB | 채점 결과·학습 로그 |

| 안 옮긴다 | 크기 | 대신 |
|---|---|---|
| `venv`, `venv-train` | 6.8GB | pip로 재설치 (아래) |
| `data/*.parquet` | 909MB | Hugging Face에서 재다운로드 |
| `checkpoint-455`, `checkpoint-546` | 424MB | `final/`에 최적 가중치가 이미 있다. 학습을 더 이을 때만 필요 |
| `work/{std,cam,crop,panel}` | 213MB | zip 안 `dataset/images/`로 충분. 다시 만들려면 아래 3단계 |
| `.git` | 127MB | `git clone` |

`adapter-5ep`는 `distill/qwen2vl-distill/final/`을 그대로 복사한 것이다.
`checkpoint-455`와 가중치 해시가 같다 — 5에포크가 최적점이었고 6에포크는 과적합이다.

## 노트북에서

### 1. 저장소

```
git clone -b feature/claude-image-analysis https://github.com/Sangmok-Woo/claude-image-detection.git
cd claude-image-detection
```

`MobileViT2_Model.h5`(62MB)는 Git LFS다. `git lfs install`이 안 돼 있으면 포인터 파일만
내려오니 `head -c 4 MobileViT2_Model/MobileViT2_Model.h5`가 `\x89HDF`인지 확인한다.

### 2. Drive에서 받은 것 제자리에

```
mkdir -p distill/qwen2vl-distill
cp -r <Drive>/adapter-5ep  distill/qwen2vl-distill/final
cp -r <Drive>/adapter-3ep  distill/qwen2vl-distill/final-3ep
cp    <Drive>/tiny_genimage_distill.zip distill/
cd distill && unzip -q tiny_genimage_distill.zip && cd ..   # → distill/dataset/
```

### 3. 환경 — 두 개로 나뉜다

TensorFlow는 numpy<2, torch 쪽은 numpy 2.x라 **한 환경에 못 합친다.**

```
# MobileViT용 (판정 + Grad-CAM)
py -3.11 -m venv venv
./venv/Scripts/python -m pip install -r requirements.txt pyarrow huggingface_hub

# Qwen용 (학습 + 추론)
py -3.11 -m venv venv-train
./venv-train/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
./venv-train/Scripts/python -m pip install "transformers>=4.49" peft bitsandbytes accelerate pillow
```

GPU가 없거나 8GB 미만이면 추론도 버겁다. **VRAM 실측 7.05~7.95GB**(`--side`에 따라).
CPU로도 돌긴 하지만 한 장에 수 분 걸린다.

### 4. 되는지 확인

```
./venv-train/Scripts/python distill/predict.py <아무 이미지>
```

이미지 → MobileViT 판정 + Grad-CAM → 학생이 한국어 평가문. 이게 되면 옮기기 성공이다.

## 다시 만들어야 할 때

### 이미지 1000장 (`work/`)

```
./venv/Scripts/python -c "
from huggingface_hub import hf_hub_download; import shutil
for i in (0,1):
    p = hf_hub_download('TheKernel01/Tiny-GenImage', f'data/train-{i:05d}-of-00014.parquet', repo_type='dataset')
    shutil.copyfile(p, f'data/train-{i:05d}.parquet')"
./venv/Scripts/python distill/build_samples.py --shards data/train-0000{0,1}.parquet --n 1000
rm distill/work/meta.jsonl
./venv/Scripts/python distill/regen_new_model.py
```

**두 번째 단계를 빼먹으면 안 된다.** `build_samples.py`는 메모리상 q90 이미지로 확률을 재고
`std/`에는 q95로 저장하는데, `regen_new_model.py`는 저장된 q95를 다시 읽는다. 커밋된
`meta.jsonl`은 후자의 산출물이라, 이어 돌려야 `pred`·`crop_box`가 1000/1000 일치한다.
CPU로 약 1시간 + 40분.

### 데이터셋 다시 내보내기

```
./venv/Scripts/python distill/export_dataset.py --detailed
```

교사 설명(`work/teacher_full/`)은 저장소에 있으니 `work/` 이미지만 있으면 된다.

### 학습 다시/더 돌리기

```
./venv-train/Scripts/python distill/smoke_test.py --data distill/dataset      # VRAM 먼저 확인
./venv-train/Scripts/python distill/train_qlora.py --data distill/dataset --out distill/qwen2vl-distill --side 336
```

이어서 돌리려면 `--resume <checkpoint>` + `--epochs <더 큰 수>`. 단 **체크포인트를 안 옮겼으면
처음부터**다. 어차피 6에포크가 3에포크보다 정확도가 낫지 않았으니 더 돌릴 이유는 적다.

## 상태 요약 (2026-09-27)

끝난 것: 이미지 1000장, 교사 상세 설명 1000장, QLoRA 학습·채점, 추론 파이프라인.

| | test 100장 정확도 |
|---|---|
| 교사(Claude) | 85% |
| 학생(Qwen2-VL-2B) | **78%** |
| MobileViT | 67% |

자세한 내용과 발표용 수치는 [REPORT.md](REPORT.md)에 있다.

남은 것으로 적어둔 것:
- 근거를 판정보다 **앞에** 두는 `assistant_text` 순서로 재학습해 정확도가 달라지는지
- 원본에서 테두리를 잘라낸 변형으로 분류기 정확도가 얼마나 떨어지는지 (모서리 집중 2.6배의 직접 증명)
- 2B 학생이 장면 묘사를 지어내는 문제 — 더 큰 학생(7B)이나 근거 검증 단계 추가

## 데스크탑 정리

노트북으로 옮긴 뒤 데스크탑 공간을 비우려면:

```
rm -rf venv venv-train                          # 6.8GB, pip로 재생성
rm -rf data                                     # 909MB, HF에서 재다운로드
rm -rf distill/qwen2vl-distill/checkpoint-*     # 424MB, final/에 최적 가중치가 있다
rm -rf distill/work/{cam,crop,panel}            # 158MB, regen_new_model.py로 재생성
```

`distill/qwen2vl-distill/final*`과 `distill/tiny_genimage_distill.zip`은 **지우지 말 것.**
Drive에 올렸더라도 원본은 남겨두는 편이 안전하다.
