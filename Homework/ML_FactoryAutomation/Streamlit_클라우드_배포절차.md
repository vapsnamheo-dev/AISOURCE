# Streamlit Cloud 배포 절차 — PdM-Guard

## 사전 준비 (완료)

| 항목 | 상태 |
|---|---|
| GitHub 저장소 | `vapsnamheo-dev/AISOURCE` (main 브랜치) |
| 앱 파일 | `Homework/ML_FactoryAutomation/app/streamlit_app.py` |
| 패키지 목록 | `Homework/ML_FactoryAutomation/app/requirements.txt` |
| 학습 모델 | `Homework/ML_FactoryAutomation/model/*.pkl` |
| 소스 코드 | `Homework/ML_FactoryAutomation/src/*.py` |

---

## 배포 절차

### 1단계. Streamlit Cloud 로그인

1. 브라우저에서 `https://share.streamlit.io` 접속
2. **GitHub로 계속하기** 클릭 → GitHub 계정으로 로그인

---

### 2단계. 앱 생성 시작

1. 우측 상단 **Create app** 버튼 클릭
2. **Deploy a public app from GitHub** 선택 (첫 번째 옵션)

---

### 3단계. 배포 정보 입력

아래 값을 그대로 입력합니다.

| 항목 | 입력값 |
|---|---|
| **Repository** | `vapsnamheo-dev/AISOURCE` |
| **Branch** | `main` |
| **Main file path** | `Homework/ML_FactoryAutomation/app/streamlit_app.py` |
| **App URL** | 원하는 이름 입력 (예: `pdmguard`) |

> App URL 옆에 **Domain is available** 이라고 표시되면 사용 가능한 이름입니다.

**Deploy** 버튼 클릭 → 패키지 설치 및 앱 기동 (약 2~5분 소요)

---

### 4단계. Python 버전 설정

배포 완료 후 아래 순서로 Python 버전을 맞춥니다.

1. 우측 하단 **Manage app** 버튼 클릭
2. 점 3개(`⋮`) 클릭 → **Settings** 클릭
3. **Python version** → `3.10` 선택
4. **Save changes** 클릭 → 앱 자동 재시작

---

### 5단계. 배포 확인

- 배포 완료 URL 예시: `https://pdmguard.streamlit.app`
- 이 URL을 누구에게나 공유하면 브라우저에서 바로 접속 가능

---

## 배포 후 관리

| 작업 | 방법 |
|---|---|
| **앱 재시작** | Manage app → 점 3개 → Reboot app |
| **앱 삭제** | Manage app → 점 3개 → Delete app |
| **로그 확인** | Manage app 패널 하단 로그 영역 |
| **코드 업데이트** | GitHub main 브랜치에 push하면 자동 반영 |

---

## requirements.txt 위치 규칙

Streamlit Cloud는 아래 두 위치에서만 `requirements.txt`를 탐색합니다.

- **앱 파일과 같은 디렉토리** → `app/requirements.txt` ← 현재 프로젝트 적용 위치
- **저장소 루트** → `AISOURCE/requirements.txt`

현재 프로젝트의 `app/requirements.txt` 내용:

```
streamlit>=1.30
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
sqlalchemy>=2.0
joblib>=1.3
```
