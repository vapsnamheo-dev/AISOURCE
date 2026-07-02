# GRU 전체 구조 한눈에 보기 — 상세 설명

**참고 다이어그램**: 전체구조.png (batch_size=16, seq_length=8, GRU(64)→GRU(32)→Dense(1))  
**작성일**: 2026-06-29

---

## 1. 전체 흐름 요약

```
전체 학습 데이터
      ↓ (배치 단위로 분할)
[배치 1 — 16개 시퀀스]  [배치 2]  ...  [배치 10(마지막)]
      ↓
  1 Epoch 완료 → 가중치 1회 업데이트
      ↓ (다음 Epoch 반복)
  EarlyStopping 조건 충족 시 종료
```

**핵심 개념 3가지**:

| 개념 | 정의 | 예시 |
| ---- | ---- | ---- |
| **Epoch** | 전체 데이터셋을 한 번 완전히 학습하는 단위 | 50 Epoch = 50번 반복 |
| **Batch** | 한 번에 처리하는 시퀀스 묶음 | batch_size=16 → 16개 동시 처리 |
| **Iteration** | 1 Epoch 안의 배치 처리 횟수 | 전체 160샘플 ÷ 16 = 10 iterations |

---

## 2. 배치 1개의 내부 구조 (batch_size=16, seq_length=8)

### 2.1 입력 텐서 형태

```
입력 shape: (batch_size=16, seq_length=8, features=1)

          ← 시간(날짜), 8 타임스텝 →
          Day1  Day2  Day3  Day4  Day5  Day6  Day7  Day8
시퀀스  1: x1,1  x1,2  x1,3  x1,4  x1,5  x1,6  x1,7  x1,8
시퀀스  2: x2,1  x2,2  ...                           x2,8
   ...
시퀀스 16: x16,1  ...                           x16,8
```

- **행(row)** = 배치 내 시퀀스 (16개가 병렬 처리)
- **열(column)** = 타임스텝 (Day1→Day8, 순서가 핵심)
- **깊이** = 피처 수 (count 1개)

### 2.2 배치 크기 vs look_back vs filter_size 구분

| 용어 | 값 | 의미 | 조정 효과 |
| ---- | ---- | ---- | ---- |
| **batch_size** | 16 (다이어그램) / 32 (bike) | 병렬 처리 시퀀스 수 | 크면 빠름·메모리 증가 |
| **seq_length (look_back)** | 8 (다이어그램) / 24 (bike) | 과거 몇 스텝을 볼지 | 크면 장기패턴 학습 |
| **filter_size (CNN 전용)** | 3, 5, 7 | 합성곱 커널 크기 | GRU에는 해당 없음 |

> **batch_size ≠ look_back ≠ filter_size** — 세 개념은 완전히 독립적.

---

## 3. GRU(64) — return_sequences=True

첫 번째 GRU는 **매 타임스텝마다 hidden state 출력**.  
입력 (16, 8, 1) → 출력 (16, 8, 64)

```
1개 시퀀스 기준, 타임스텝 t에서:

Update Gate:  z_t = σ(W_z · [h_{t-1}, x_t])
Reset Gate:   r_t = σ(W_r · [h_{t-1}, x_t])
Candidate:    h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t])
Hidden State: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

- **Update Gate (z)**: 이전 기억을 얼마나 유지할지 (0=완전 교체, 1=완전 유지)
- **Reset Gate (r)**: 이전 hidden state를 후보 계산에 얼마나 반영할지
- **h_t**: 현재 타임스텝의 기억 벡터 → 다음 타임스텝으로 전달

```
Day1 →  GRU(64)  →  h¹₁  (64차원)
Day2 →  GRU(64)  →  h¹₂  (h¹₁ 기억 포함)
...
Day8 →  GRU(64)  →  h¹₈  (Day1~Day8 전체 압축)
```

출력: **(16, 8, 64)**

---

## 4. GRU(32) — return_sequences=False

두 번째 GRU는 **마지막 타임스텝(Day8)의 hidden state만 출력**.  
입력 (16, 8, 64) → 출력 (16, 32) — 시계열 → 고정 벡터 압축

```
GRU(32) 최종 출력:
  s₁ = f(s₇, h¹₈)   ← Day1~Day8 전체 정보가 32차원에 압축됨
  ...
  s₁₆ = f(...)
```

출력: **(16, 32)**

---

## 5. Dense(1) — 예측 출력

```
입력 (16, 32) → Dense(1) → 출력 (16, 1)

ŷ₁ = W·s¹ + b    (시퀀스 1의 예측)
ŷ₂ = W·s² + b
...
ŷ₁₆ = W·s¹⁶ + b
```

활성화 함수 없음 (linear) → 회귀 출력

---

## 6. Loss 계산 및 역전파

```
배치 평균 MSE Loss:
  L = (1/16) × Σᵢ₌₁¹⁶ (ŷᵢ - yᵢ)²

역전파 (BPTT):
  ∂L/∂W → Adam → 가중치 업데이트

1 Epoch = 전체 샘플 ÷ batch_size 번 반복
```

---

## 7. bike.csv 예제 실제 설정값 대응

| 항목 | 다이어그램 | bike.csv 실제 |
| ---- | ---- | ---- |
| batch_size | 16 | **32** |
| seq_length (look_back) | 8일 | **24시간** |
| 피처 수 | 1 | **1 (count)** |
| GRU 1st | GRU(64) | **GRU(64)** |
| GRU 2nd | GRU(32) | **GRU(32)** |
| Dense | Dense(1) | **Dense(1)** |
| 체크포인트 | — | **best_bike_gru_model.keras** |
| EarlyStopping | — | **patience=5** |

---

## 8. 전처리 → 학습 → 역변환 전체 파이프라인

```
[bike.csv]  count: 왜도=1.24 (right-skewed)
      ↓  (Case 2) np.log1p(count)
[log_count]  왜도≈0.1 (정규분포 근접)
      ↓  MinMaxScaler → [0, 1]
[scaled]  shape: (N, 1)
      ↓  create_dataset(look_back=24)
[X, y]  X shape: (N, 24, 1)
      ↓  train/test split 80/20 (shuffle=False)
[X_train, X_test]
      ↓  model.fit(callbacks=[checkpoint, early_stopping])
[학습 완료 — best_bike_gru_model.keras]
      ↓  model.predict(X_test)
[y_pred_scaled]
      ↓  scaler.inverse_transform()
[y_pred_log]
      ↓  np.expm1()
[y_pred_orig]
      ↓  np.maximum(y_pred_orig, 0)   ← 음수 보정
[최종 예측값 (대여 건수)]
```

---

## 9. 핵심 용어 사전

| 용어 | 정의 |
| ---- | ---- |
| Epoch | 전체 데이터셋 1회 완전 학습 |
| Batch | 1회 forward+backward pass의 샘플 묶음 |
| Iteration | 1 Epoch 내 배치 처리 횟수 |
| Hidden State | GRU의 "기억" 벡터 (타임스텝 간 전달) |
| return_sequences=True | 모든 타임스텝 hidden state 출력 |
| return_sequences=False | 마지막 타임스텝만 출력 |
| look_back | 입력 시퀀스 길이 (몇 스텝 과거 참조) |
| log1p / expm1 | 왜도 제거 로그 변환 쌍 |
| np.maximum(x, 0) | 음수 예측값 클리핑 |
| ModelCheckpoint | val_loss 최솟값 시점 자동 저장 |
| EarlyStopping | patience 초과 시 학습 자동 종료 |

---

*관련 소스: `c:\AISOURCE\classExample\dl.ipynb` (GRU 자전거 수요 예측 셀 5개)*
