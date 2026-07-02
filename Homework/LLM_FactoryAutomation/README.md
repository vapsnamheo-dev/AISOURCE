# DL_FactoryAutomation — 딥러닝 기반 설비 고장 예측

## ML → DL 확장 배경

[ML_FactoryAutomation](../ML_FactoryAutomation) 프로젝트(XGBoost, 표 데이터)의 딥러닝 후속입니다.
PDF 수행내역서 6.4절 로드맵 ("딥러닝: 시계열 센서 신호에 1D-CNN/LSTM 적용") 구현체입니다.

| 항목 | ML 프로젝트 | DL 프로젝트 (본) |
|------|------------|-----------------|
| 모델 | XGBoost | 1D-CNN / LSTM / CNN+LSTM |
| 입력 | 표 데이터 (1행 = 1스냅샷) | 시계열 시퀀스 (50 타임스텝) |
| 피처 | 동일 (AI4I 2020 기반) | 동일 + 시계열 파생 |
| 데이터 | AI4I 2020 (10k행) | 합성 시계열 (6,000 시퀀스) |
| Recall | 80.9% | 목표 ≥ 82% |

## 피처 (ML과 동일)

| 피처 | 설명 | ML 대응 |
|------|------|---------|
| `air_temp_k` | 공기 온도 (K) | Air temperature [K] |
| `process_temp_k` | 공정 온도 (K) | Process temperature [K] |
| `rotational_speed_rpm` | 회전속도 | Rotational speed [rpm] |
| `torque_nm` | 토크 (Nm) | Torque [Nm] |
| `tool_wear_min` | 공구 마모 (min) | Tool wear [min] |
| `type_encoded` | 설비 등급 (0=L/1=M/2=H) | Type |
| `power_w` | 파생: 소비전력 | Power [W] |
| `overstrain_minnm` | 파생: 과부하 지수 | Overstrain [minNm] |
| `temp_diff_k` | 파생: 온도 차이 | Temp diff [K] |

## 빠른 시작

```bash
# 1. 환경 설치
pip install -r requirements.txt

# 2. 시계열 데이터 생성 (train 4800개 / test 1200개 / demo 200개)
python src/generate_ts_data.py

# 3. CNN+LSTM 모델 학습 (CV=3, 80:20 분할)
python src/train.py

# 4. Streamlit 앱 실행
streamlit run app/streamlit_app.py
```

## 프로젝트 구조

```
DL_FactoryAutomation/
├── src/
│   ├── config.py            # 설정 (시퀀스 길이, 피처 목록, 경로)
│   ├── generate_ts_data.py  # AI4I 물리 규칙 기반 시계열 생성
│   ├── data_loader.py       # numpy 배열 변환 + CV 분할
│   ├── dl_model.py          # 1D-CNN / LSTM / CNN+LSTM 정의
│   ├── train.py             # 학습 파이프라인 (CV=3, 80:20)
│   └── predict.py           # 추론 (단건 시퀀스)
├── data/
│   ├── train/               # 학습 데이터 (4,800 시퀀스)
│   ├── test/                # 테스트 데이터 (1,200 시퀀스)
│   └── demo/                # 데모 전용 (200 시퀀스, train/test 미포함)
├── app/streamlit_app.py     # 웹 UI
├── tests/test_generate.py   # 단위 테스트
└── .github/workflows/ci.yml # GitHub Actions CI
```

## 데이터 생성 방법론

AI4I 2020 동일 물리 규칙으로 시계열 시퀀스를 합성합니다:

```
고장 규칙 (ML과 동일):
  TWF: tool_wear > 200 AND torque > 60
  HDF: temp_diff < 8.6K AND rpm < 1380
  PWF: power ∉ [3500, 9000] W
  OSF: overstrain > 장비등급별 임계값
  RNF: 0.1% 확률 무작위

각 시퀀스: 50 타임스텝 × 9 피처
  → 정상 시퀀스: 50스텝 동안 물리 범위 유지
  → 고장 시퀀스: 마지막 10~20스텝에 임계값 초과 누적
```

데이터 분할:
- 전체 6,000 시퀀스 → **train 4,800 (80%) / test 1,200 (20%)**
- **CV=3** Stratified K-Fold (클래스 비율 유지)
- **demo 200개**: 완전히 별도 시드(9999)로 생성, train/test와 겹치지 않음
