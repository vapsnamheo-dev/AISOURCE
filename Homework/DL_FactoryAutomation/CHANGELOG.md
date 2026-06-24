# 변경 이력 (CHANGELOG)

## v0.2.0 — 2026-06-24 (예정)
- 데이터2(v2): 60,000 시퀀스 + 7.5% 가우시안 노이즈
- CNN+LSTM 재학습 예정

## v0.1.0 — 2026-06-24
### 초기 릴리스
- CNN+LSTM 이진 분류 모델 구현 (src/dl_model.py)
- AI4I 물리 규칙 기반 합성 시계열 생성기 (src/generate_ts_data.py)
- 데이터1(v1): 6,000 시퀀스 (train 4,800 / test 1,200), 35% 고장률, 노이즈 없음
- CV=3 K-Fold 학습 파이프라인 (src/train.py)
- Streamlit 데모 앱 (app/streamlit_app.py)

### 성능 (데이터1 기준)
| 지표 | CV 평균 | Test Set |
|------|---------|---------|
| Accuracy | 99.98% | 100.00% |
| F1 Score | - | 1.00 |
| ROC-AUC | - | 약 1.00 |
| Threshold | 0.75 | 0.75 |

> 데이터1은 노이즈 없는 이상적 합성 데이터로 100% 달성. 현실적 성능은 데이터2(노이즈 포함) 결과 참조.
