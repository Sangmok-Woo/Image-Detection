# Claude → Qwen2-VL 설명 증류

MobileViT 판정 + Grad-CAM 히트맵 + 히트맵 크롭을 Claude(Opus 5)가 보고 쓴 한국어 설명을
Qwen2-VL-2B에 QLoRA로 옮겨 담는다. 데이터: Hugging Face `TheKernel01/Tiny-GenImage` (CC BY-NC-SA 4.0)

## 흐름
| 단계 | 파일 | 어디서 |
|---|---|---|
| 1. 샘플 1000장 뽑기 + 판정/히트맵/크롭 | `build_samples.py` | 로컬 venv (CPU, 약 40분) |
| 2. 라벨 뺀 배치 만들기 → Claude Code 서브에이전트가 패널을 보고 설명 작성 | `make_batches.py`, `teacher_prompt.md` | Claude Code |
| 3. 합치고 검증·분할해 내보내기 | `export_dataset.py` | 로컬 |
| 4. QLoRA 학습 / test 채점 | `train_qlora.py` | Colab (T4 이상) |

## 설계에서 정한 것
- **형식 누수 제거**: 원본은 진짜=JPEG 직사각형, 가짜=PNG 정사각형이다. 그대로 두면 "파일 형식"으로 정답을 외우므로 전부 384×384 JPEG로 통일했다.
  단, BigGAN(128px)은 확대돼 흐릿해지는 흔적이 남는다. 발표 때 한계로 적을 것.
- **교사에게 정답을 주지 않는다**: 정답을 알고 쓰면 그럴듯한 끼워맞추기 설명이 나온다. 교사는 눈으로 판단하고,
  train/val에는 교사가 맞힌 것만 남긴다. test는 전부 남겨 학생·교사·MobileViT 정확도를 같은 기준으로 비교한다.
- **Grad-CAM은 예측한 클래스 기준**으로 계산한다 (앱의 기존 코드는 항상 REAL 점수 기준).
- 샘플 구성: 진짜 500 + 가짜 500 (ADM, BigGAN, GLIDE, Midjourney, SD15, VQDM, Wukong 균등. 이 조각들엔 SD14가 없다).

## 학습 데이터 한 줄
`images`(원본·히트맵·크롭 3장), `messages`(LLaMA-Factory 호환), `user_text`/`assistant_text`, 정답·생성기·MobileViT·교사 판정.

## 2026-09-17 1차 결과 (구모델, 폐기 — old_model/에 백업) (`dataset/stats.json`)
| | 교사(Claude Opus 5) | MobileViT |
|---|---|---|
| 전체 1000장 | **84.6%** | 47.4% |
| 진짜 사진 | 95.2% | **1.6%** — 거의 모든 사진을 "AI"라고 찍는다 |
| ADM | **36.1%** — 교사의 약점, 진짜로 착각 | 90.3% |
| BigGAN / VQDM / GLIDE | 65~69% | 89~93% |
| SD15 / Wukong / Midjourney | 89~99% | 92~100% |

- 분할: train 722 / val 39 (교사가 틀린 139장 제외) / test 100 (진짜 50 + 생성기별 7~8, 전부 포함)
- train 치우침: 진짜 403 vs AI 319, **ADM은 24장뿐**이다. 학생도 ADM에 약할 것으로 예상.
- 히트맵 유용성(교사 평가): yes 214 / partial 446 / **no 340**. 3분의 1은 히트맵이 엉뚱한 곳을 가리킨다. "Grad-CAM이 거짓말하는지" 발표 자료로 쓸 수 있다.
- zip 98MB → Google Drive에 올리고 `train_qlora.py` 맨 위 Colab 셀 예시대로 실행.

## 2026-09-17 2차 — 신모델(Tiny-GenImage로 재학습된 MobileViT)로 교체 ✅ 현재 데이터셋
pull이 덜 돼 1차는 구모델(CIFAKE)로 만들었다. 두 모델의 히트맵 상관은 0.05, 크롭 겹침은 17%로 사실상 딴판이라 다시 했다.
- **보존**: 384 원본, 교사 판정·확신·근거(`observations`)
- **재생성**: 확률·판정·Grad-CAM·크롭·패널(`regen_new_model.py`), 교사의 히트맵 설명·유용성·최종 설명(`teacher_heatmap_prompt.md`, `make_heatmap_batches.py`)
- 교사 정확도·분할은 1차와 동일(판정을 보존했으므로)

| | 구모델 | 신모델 |
|---|---|---|
| 전체 정확도 | 47.4% | **66.4%** |
| 진짜 사진 | 1.6% | 47.4% |
| Midjourney / Wukong | 91% / 100% | **45% / 75%** |
| 히트맵 유용 yes / partial / no | 214 / 446 / 340 | **424 / 461 / 115** |

주의: 신모델은 Tiny-GenImage train으로 학습했는데 우리 1000장도 train 조각에서 뽑았다. 신모델이 본 이미지일 수 있어 66%도 후하게 나온 숫자일 수 있다.
