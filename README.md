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

```powershell
# 가상 환경 생성
py -3.11 -m venv venv

# 가상 환경 활성화
.\venv\Scripts\activate
```

---

## 2️⃣ 필수 패키지 설치

```powershell
pip install --upgrade pip
pip install streamlit tensorflow==2.15.0 numpy Pillow
```

---

## 3️⃣ 앱 실행

```powershell
python -m streamlit run app.py
```

---

# 📂 프로젝트 구조

```plaintext
GP/
├── MobileViT2_Model/
│   └── MobileViT2_Model.ht      # 핵심 탐지 모델 (Input: 224x224)
├── styles/
│   ├── style.css                # 커스텀 UI 디자인
│   ├── robot.png                # 상단 로고
│   └── detectiveMag.svg         # 돋보기 아이콘
├── app.py                       # 서비스 메인 로직
└── README.md                    # 프로젝트 문서
```

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

---

# ⚠️ 실행 시 참고사항

## DLL 오류 발생 시

Microsoft Visual C++ 재배포 패키지를 설치한 후 재부팅하세요.

## 모델 경로 확인

다음 경로에 모델 파일이 반드시 존재해야 합니다.

```plaintext
MobileViT2_Model/MobileViT2_Model.ht
```

---

# 🚀 기술 스택

- Python
- TensorFlow 2.15
- Streamlit
- NumPy
- Pillow
- MobileViT v2

---

# 📷 시스템 동작 흐름

1. 사용자가 이미지를 업로드
2. 이미지 전처리 (224×224 리사이즈)
3. MobileViT v2 모델 추론 수행
4. AI 생성 여부 확률 계산
5. 결과 및 Confidence 출력

---

# 📄 라이선스

본 프로젝트는 연구 및 학습 목적으로 제작되었습니다.
```