# PdM-Guard — 설비 고장 예측 시스템 🛠️

설비 센서 데이터(온도·회전속도·토크·공구마모)를 학습하여 **고장 발생 여부를 예측**하는 머신러닝 분석 + Streamlit 웹앱 프로젝트입니다. 해석 가능한 베이스라인(Logistic Regression)과 고성능 모델(XGBoost)을 비교합니다.

> ⚠️ 본 README는 **초안**입니다. 소스 생성 후 실제 명령/경로에 맞게 갱신됩니다.

---

## 📌 주요 특징
- 이진 분류: 정상(0) / 고장(1), 클래스 불균형(고장 3.4%) 대응
- 모델 비교: LogisticRegression vs XGBoost
- 불균형 친화 지표: ROC-AUC, Recall, F1, 혼동행렬
- 예측 웹앱(Streamlit) + 예측 이력 DB 로깅
- CI(GitHub Actions) + Streamlit Community Cloud 배포

---

## 🗂️ 디렉터리 구조
```
pdm-guard/
├── data/
│   └── predictive_maintenance.csv
├── src/
│   ├── config.py          # 경로·하이퍼파라미터·환경변수
│   ├── data_loader.py     # CSV 로드·검증
│   ├── eda.py             # 분포·상관·이상치
│   ├── preprocess.py      # 인코딩·스케일·분리
│   ├── train.py           # LogReg / XGBoost 학습
│   ├── evaluate.py        # 지표·모델비교·SHAP
│   ├── db.py              # SQLAlchemy 모델·세션
│   └── predict.py         # 추론 + DB 로깅
├── app/
│   └── streamlit_app.py   # 예측 UI
├── models/                # 학습된 모델·스케일러(joblib)
├── notebooks/
│   └── eda.ipynb
├── tests/                 # 단위·통합 테스트(pytest)
├── .github/workflows/ci.yml
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```

---

## ⚙️ 기술 스택
| 영역 | 스택 |
|---|---|
| 언어 | Python 3.11 |
| ML | scikit-learn, xgboost, shap |
| 데이터 | pandas, numpy |
| 시각화 | matplotlib, seaborn |
| 앱 | streamlit |
| DB | SQLAlchemy + SQLite(로컬) / Postgres·MariaDB(클라우드) |
| 테스트 | pytest |
| CI/CD | GitHub Actions → Streamlit Community Cloud |

---

## 🚀 빠른 시작 (로컬)
```bash
# 1. 가상환경
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. 의존성
pip install -r requirements.txt

# 3. 모델 학습
python -m src.train

# 4. 평가
python -m src.evaluate

# 5. 앱 실행
streamlit run app/streamlit_app.py
```

---

## 🗄️ 데이터베이스
- 기본값: 로컬 SQLite (`sqlite:///pdm.db`)
- 클라우드: 환경변수 `DATABASE_URL` 로 교체 (예: Supabase Postgres)
```bash
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
```
- 스키마: `04_DB설계_논리_물리_ERD.md` 참조 (machine / sensor_reading / prediction / model_registry / failure_type)

---

## 🧪 테스트
```bash
pytest -v            # 단위 + 통합
ruff check .         # 린트
```

---

## ☁️ 배포 (Streamlit Community Cloud)
1. GitHub 저장소 push
2. https://share.streamlit.io 접속 → 저장소·브랜치·엔트리포인트(`app/streamlit_app.py`) 지정
3. `Secrets` 에 `DATABASE_URL` 등록
4. Deploy → 배포 URL 동작 확인

> 대안(전체 백엔드+DB 일체형): **CloudType** 으로 Dockerfile 기반 배포 가능. 검토 후 선택.

---

## 📊 결과 요약 (학습 후 채움)
| 모델 | Accuracy | Recall(고장) | F1 | ROC-AUC |
|---|---|---|---|---|
| LogisticRegression | - | - | - | - |
| XGBoost | - | - | - | - |

---

## 📝 라이선스 / 출처
- 데이터: AI4I 2020 Predictive Maintenance Dataset (CC BY 4.0)
- 본 프로젝트: 교육/캡스톤 목적

---
*문서 버전 v0.1 · 검토자: Daniel*

## 데이터 확장 — 동일 물리규칙 생성 비교 (머신러닝 전용)
- `data/ai4i_physics_100k.csv`: AI4I의 공개 물리규칙(HDF·PWF·OSF·TWF·RNF)으로 생성한 100,000행(딥러닝 미사용).
- 생성: `python -m src.synth_ai4i 100000`
- 비교/검증: `python -m src.compare_physics` (동일 분포 비교 + 실제 데이터 검증 A/B/C)

## 분류 모델 종류별 전수 비교 (머신러닝 8종)
- 실행: `python -m src.compare_classifiers`
- LogReg·KNN·DecisionTree·RandomForest·GradientBoosting·SVC·GaussianNB·XGBoost 비교.
- MLPClassifier(신경망=딥러닝)는 머신러닝 전용 원칙상 제외, Naive Bayes는 연속형에 맞는 GaussianNB 사용.
