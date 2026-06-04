# 🛡️ AI vs REAL: 이미지 위변조 탐지 시스템  
### Generative Image Forgery Detection with MobileViT v2

이 프로젝트는 **MobileViT v2** 아키텍처를 기반으로 AI 생성 이미지(Fake)와 실제 촬영 이미지(Real)를 판별하는 **Streamlit 웹 애플리케이션**입니다.  
경량 비전 트랜스포머를 활용하여 효율적인 자원 사용과 높은 탐지 성능을 동시에 제공합니다.

---

# 📌 주요 기능

- **실시간 탐지:** 이미지 업로드 즉시 AI 위변조 여부 분석
- **최적화된 추론:** MobileViT v2 기반의 경량화된 추론 엔진 탑재
- **직관적 UI:** 분석 확률(Confidence)과 판별 결과를 시각적으로 시각화

---

# 🛠️ 설치 및 실행 (Windows)

본 프로젝트는 **Python 3.10 / 3.11** 환경에서 가장 안정적으로 작동합니다.

## 1️⃣ 가상 환경 설정

py -3.11 -m venv venv
.\venv\Scripts\activate

---

## 2️⃣ 필수 패키지 설치

python -m pip install --upgrade pip
pip install -r requirements.txt


---

## 3️⃣ 앱 실행

python -m streamlit run app.py

---

## TensorFlow 오류시

1. 기존에 설치가 꼬인 텐서플로우 관련 패키지들을 강제로 완전히 삭제합니다.
pip uninstall -y tensorflow tensorflow-intel tf_keras keras

2. 캐시를 무시하고 텐서플로우 2.15.0 버전을 깨끗하게 다시 설치합니다.
pip install --no-cache-dir tensorflow==2.15.0 tf_keras==2.15.0

---

# ⚙️ 모델 사양 및 판별 기준

| 항목 | 내용 |
|---|---|
| 모델 구조 | MobileViT v2 |
| 입력 해상도 | 224 × 224 (RGB) |
| 판별 임계값 (Threshold) | 0.5 |

### 판별 기준

- **0.5 미만 → AI Generated (위조 이미지)**
- **0.5 이상 → REAL (실제 이미지)**