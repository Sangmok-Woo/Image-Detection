# 교사(Claude) 히트맵 부분 재작성 지침

분류기 모델이 새 버전으로 바뀌어 히트맵·크롭·분류기 판정이 달라졌다.
원본 이미지를 보고 쓴 **판정과 근거는 그대로 두고**, 모델이 관여하는 부분만 다시 쓴다.

각 샘플마다 `distill/work/panel/{id}.jpg` 한 장을 본다 (1 원본 | 2 새 Grad-CAM, 흰 박스 = 크롭 위치 | 3 크롭 3배 확대).
배치 파일 한 줄에는 다음이 있다.
- `mobilevit_prob_real`, `mobilevit_pred`: 새 분류기 확률(1에 가까울수록 실제 사진)과 판정
- `teacher_verdict`, `teacher_confidence`, `observations`: 이전에 원본을 보고 내린 판정·확신·근거 (**바꾸지 않는다**)

## 다시 쓸 것
- `heatmap_focus`: 새 히트맵과 크롭이 가리키는 곳에 무엇이 있는지 한두 문장. 크롭에서 실제로 보이는 것을 구체적으로.
- `heatmap_useful`: `yes | partial | no`. 히트맵이 `teacher_verdict`의 근거가 되는 곳을 짚었는지. 배경이나 의미 없는 곳이면 솔직하게 no.
- `explanation`: 최종 설명 3~5문장. `teacher_verdict`와 `observations`를 바탕으로 쓰되, 새 분류기 판정과 같은지 다른지, 새 히트맵이 근거를 제대로 짚었는지를 반영한다. `teacher_confidence`에 맞춰 확신 강도를 조절한다.

## 규칙
- 판정·확신·근거를 바꾸지 않는다. 설명이 그것과 모순되면 안 된다.
- 파일 형식, 해상도, 압축 품질은 근거로 쓰지 않는다.
- 한국어, 일반인이 이해할 말로. 지어내지 않는다.

## 출력 (한 줄에 JSON 하나, JSONL)
```json
{"id": "tg0000", "heatmap_focus": "...", "heatmap_useful": "yes | partial | no", "explanation": "..."}
```
