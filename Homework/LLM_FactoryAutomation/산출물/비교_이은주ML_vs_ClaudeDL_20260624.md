# 이은주 ML 보고서 vs Claude DL 프로젝트 비교

**작성일**: 2026-06-24  
**목적**: 이은주(2026) ML 보고서와 본 Claude DL 프로젝트의 접근 방식·성능 비교  
**주의**: 이은주 보고서는 **AI4I 2020 정형 데이터**, Claude FEMTO DL은 **FEMTO-ST 시계열 데이터** — 데이터셋이 다르므로 성능 수치 직접 비교 불가. 방법론 발전 방향 비교용.

---

## 1. 이은주 ML 성능 참고값 (AI4I 2020 정형 데이터 기준)

| 모델 | Accuracy | Recall | F1 | ROC-AUC |
|------|----------|--------|----|---------|
| RandomForest (이은주) | — | **1.000** | **0.996** | — |

> 출처: 이은주 (2026). 머신러닝 기반 설비 고장 예측 (AI4I 2020 데이터셋).
> AI4I는 이진 고장 플래그가 명확한 정형 데이터 — FEMTO 시계열과 난이도·구조 모두 다름.

---

## 2. Claude ML (AI4I 동일 데이터) 학습 결과 — ML_FactoryAutomation

이은주 보고서와 **동일한 AI4I 2020** 데이터셋에서 Claude가 학습한 결과:

| 모델 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|----------|-----------|--------|----|---------|
| XGBoost | 0.979 | 0.655 | 0.809 | 0.724 | 0.971 |
| RandomForest | 0.983 | 0.827 | 0.632 | 0.717 | 0.961 |
| LogisticReg | 0.825 | 0.142 | 0.824 | 0.242 | 0.907 |

> 출처: `ML_FactoryAutomation/reports/model_comparison.csv`

**이은주 RF vs Claude RF (AI4I 동일 데이터 직접 비교)**:

| 항목 | 이은주 RF | Claude RF | 비고 |
|------|----------|-----------|------|
| Recall | **1.000** | 0.632 | 이은주 우위 |
| F1 | **0.996** | 0.717 | 이은주 우위 |
| ROC-AUC | — | 0.961 | — |

> 동일 데이터에서 이은주 RF가 Recall·F1 우위. 하이퍼파라미터 튜닝 조건 차이로 추정.

---

## 3. Claude DL (FEMTO-ST 시계열) 학습 결과 — DL_FactoryAutomation

FEMTO-ST 데이터(이은주 보고서와 **다른 데이터셋**)에서 Claude ML+DL 통합 결과:

### 3.1 ML 분류 (Claude — FEMTO-ST)

| 모델 | Accuracy | Recall | F1 | ROC-AUC |
|------|----------|--------|----|---------|
| LogisticRegression | 0.9871 | **0.8897** | 0.8171 | 0.9969 |
| RandomForest | 0.9839 | 0.5048 | 0.6701 | 0.9204 |
| XGBoost | 0.9841 | 0.5126 | 0.6769 | 0.7582 |

> 출처: `models/femto_ml_results.json` OOS 평가. GroupKFold(베어링 단위) 적용.

### 3.2 DL RUL 예측 (Claude — FEMTO-ST)

| 방법 | RMSE (분) | MAE (분) | 비고 |
|------|-----------|----------|------|
| RF 베이스라인 (ML) | 954.5 | 757.7 | 시계열 의존성 없음 |
| LSTM | 934.0 | 767.1 | 개선률 2.15% |
| **GRU+BN+LN** | **810.4** | **665.6** | **개선률 16.75%** ← 최종 채택 |

> 출처: `models/femto_rul_results.json`, `femto_dl_bn_compare_results.json`

---

## 4. 이은주 ML 한계 vs Claude DL 해결 방안

| 이은주 ML 한계 | Claude DL 해결책 |
|--------------|----------------|
| ① 단일 데이터셋 (AI4I만) | FEMTO-ST PRONOSTIA 추가 적용 |
| ② 시계열 미활용 | GRU 슬라이딩 윈도우(20분) + BN+LN 적용 |
| ③ 언제 교체할지 모름 | RUL 분 단위 예측으로 교체 타이밍 정량화 |
| ④ 고장 직전 데이터만 | 수명 전 구간(정상→열화) 연속 학습 |
| ⑤ 분류만 가능 | ML(분류) + DL(회귀 RUL) 이중 판단 |

---

## 5. 방법론 비교 (이은주 방식 vs Claude AI4I ML vs Claude FEMTO DL)

| 요소 | 이은주 방식 | Claude AI4I ML | Claude FEMTO DL |
|------|------------|---------------|----------------|
| 데이터셋 | AI4I 2020 정형 | AI4I 2020 정형 | FEMTO-ST 시계열 |
| 피처 수 | h_rms 등 4개 | 공정파라미터+파생 11개 | h/v 8개 + 온도 = 9개 |
| VIF 분석 | 미수행 | statsmodels VIF | statsmodels VIF |
| 모델 | RF 위주 | LR/RF/XGB 3종 | LR/RF/XGB + GRU |
| 교차검증 | 미표기 | Train/Test 80:20 | GroupKFold (베어링 단위) |
| GridSearch | 미수행 | 수행 | 수행 |
| 라벨 기준 | 고장 플래그 | 고장 플래그 | h_rms × 2.5 임계값 |
| 예측 종류 | 이진 분류 | 이진 분류 | 분류 + RUL 회귀 |

---

## 6. 결론

- AI4I 동일 데이터에서 이은주 RF가 Claude RF보다 Recall·F1 우위 (튜닝 조건 차이)
- FEMTO 시계열로 확장 시 Claude DL(GRU+BN+LN)이 ML 베이스라인 대비 RMSE 16.75% 개선
- 데이터셋이 달라 직접 성능 비교 불가하나, DL이 시계열 구조 학습에서 명확한 강점 확인

---

> 이 파일은 이은주 보고서와의 비교 전용 문서입니다.
> Claude ML vs Claude DL 순수 비교: [보고서_ML_DL_비교_20260624.md](보고서_ML_DL_비교_20260624.md), [보고서_FEMTO_RUL_20260624.md](보고서_FEMTO_RUL_20260624.md)
