# DL 모델 개발 트러블슈팅 가이드

**작성일**: 2026-06-29  
**프로젝트**: DL_FactoryAutomation — FEMTO-ST 베어링 RUL 예측 + 딥러닝 수업 예제

---

## 1. 모델 저장 형식 — `.h5` vs `.keras`

### 문제

```python
# 구 방식 — 경고 또는 호환 문제 발생 가능
model.save('./model/best_bike_gru_model.h5')
checkpoint_path = './model/best_bike_gru_model.h5'
```

```
UserWarning: You are saving your model as an HDF5 file via `model.save()` or
`keras.saving.save_model(model)`. This file format is considered legacy.
We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
```

### 해결

```python
# 신규 방식 — Keras v3 공식 표준
checkpoint_path = './model/best_bike_gru_model.keras'
model.save('./model/best_bike_gru_model.keras')
```

### 배경

| 항목 | `.keras` (권장) | `.h5` (레거시) |
| ---- | --------------- | -------------- |
| 도입 버전 | Keras v3 | Keras v1/v2 HDF5 |
| 직렬화 방식 | 네이티브 Keras JSON + 가중치 | HDF5 바이너리 |
| 커스텀 레이어 | 완전 지원 | 일부 미지원 |
| 권장 여부 | 신규 프로젝트 모두 | 레거시 유지 목적만 |

> 기존 `.h5` 파일 변환: `model = load_model('old.h5'); model.save('new.keras')`

---

## 2. ModelCheckpoint 콜백

### 개요

학습 중 **검증 손실(val_loss)이 최소인 시점의 모델**을 자동으로 저장하는 콜백.  
마지막 에폭이 아닌 **가장 성능이 좋은 체크포인트**를 보존하여 과적합 구간 이후 모델 사용을 방지.

### 기본 사용법

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    './model/best_bike_gru_model.keras',  # 저장 경로 (.keras 권장)
    monitor='val_loss',                   # 기준 지표
    save_best_only=True,                  # 최적값 갱신 시에만 저장
    mode='min',                           # val_loss 최솟값 기준
    verbose=1                             # 저장 이벤트 로그 출력
)
```

### 주요 파라미터

| 파라미터 | 옵션 | 설명 |
| ---- | ---- | ---- |
| `monitor` | `'val_loss'`, `'val_mae'`, `'val_accuracy'` | 기준 지표 (회귀=val_loss) |
| `save_best_only` | `True` / `False` | True: 최적 갱신 시만 저장 |
| `mode` | `'min'`, `'max'`, `'auto'` | min: 낮을수록 좋음 |
| `verbose` | `0`, `1` | 1: 저장 이벤트 출력 |

### 저장 시 출력 예시

```
Epoch 12/50
Epoch 00012: val_loss improved from 0.0234 to 0.0198, saving model to ./model/best_bike_gru_model.keras
Epoch 13/50
Epoch 00013: val_loss did not improve from 0.0198
```

### 저장된 모델 로드

```python
from tensorflow.keras.models import load_model

model = load_model('./model/best_bike_gru_model.keras')
y_pred = model.predict(X_test)
```

---

## 3. EarlyStopping 콜백

### 개요

검증 손실이 **patience 에폭 동안 개선되지 않으면 학습을 자동 조기 종료**하는 콜백.  
불필요한 학습 시간을 줄이고, 과적합(overfitting) 구간으로의 진입을 방지.

### 기본 사용법

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',          # 기준 지표
    patience=5,                  # 5 에폭 연속 미개선 시 중단
    restore_best_weights=True    # 중단 시 최적 가중치 자동 복원
)
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
| ---- | ---- | ---- |
| `monitor` | `'val_loss'` | 모니터링 지표 |
| `patience` | `0` | 개선 없이 허용할 에폭 수 (보통 3~10) |
| `restore_best_weights` | `False` | True 권장 — 조기 종료 시 최적 가중치 복원 |
| `min_delta` | `0` | 개선으로 인정할 최소 변화량 |

### `restore_best_weights=True` vs `False`

```
학습 과정:
Epoch 10: val_loss = 0.0198  ← 최적 (checkpoint 저장)
Epoch 11~15: val_loss 계속 악화 (patience=5 초과, EarlyStopping 발동)

restore_best_weights=True  → Epoch 10 가중치로 복원
restore_best_weights=False → Epoch 15 (과적합) 가중치 유지
```

---

## 4. ModelCheckpoint + EarlyStopping 조합

두 콜백을 함께 사용하면 **최적 모델 자동 저장 + 과적합 방지**를 동시에 달성.

```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

os.makedirs('./model', exist_ok=True)

checkpoint = ModelCheckpoint(
    './model/best_bike_gru_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopping],  # 두 콜백 동시 적용
    shuffle=False                             # 시계열 데이터 → shuffle 금지
)
```

### 역할 분담

| 역할 | ModelCheckpoint | EarlyStopping |
| ---- | --------------- | ------------- |
| 최적 모델 저장 | 디스크에 .keras 파일로 저장 | 없음 |
| 과적합 방지 | 없음 | patience 초과 시 중단 |
| 최적 가중치 메모리 복원 | 없음 | restore_best_weights=True |
| 안전망 | 디스크 영구 보존 | 런타임 메모리 복원 |

---

## 5. FEMTO RUL 실험 결과 (EarlyStopping 효과)

| 모델 버전 | 콜백 구성 | 학습 에폭 | OOS RMSE | OOS MAE |
| ---- | ---- | ---- | ---- | ---- |
| GRU v1 | Dropout만 | 54 에폭 | 973.47분 | 806.26분 |
| GRU v2 | BN + LN + EarlyStopping | 33 에폭 | **810.38분** | **665.63분** |
| 개선 | — | **38.9% 빠름** | **16.75%↓** | **17.44%↓** |

> 출처: `models/femto_dl_bn_compare_results.json`

---

## 6. 흔한 오류 및 해결책

### 6.1 모델 저장 경로 없음

```
FileNotFoundError: [Errno 2] No such file or directory: './model/best_bike_gru_model.keras'
```

**해결**: 저장 전 디렉터리 생성

```python
import os
os.makedirs('./model', exist_ok=True)
```

### 6.2 `.h5`로 저장된 모델 `.keras`로 마이그레이션

```python
from tensorflow.keras.models import load_model

old_model = load_model('./model/old_model.h5')
old_model.save('./model/new_model.keras')
```

### 6.3 `shuffle=False` 누락 (시계열 데이터)

```python
# 시계열 데이터에서 shuffle=True는 시간 순서를 파괴
# 반드시 shuffle=False 사용
history = model.fit(X_train, y_train, epochs=50, shuffle=False)
```

### 6.4 EarlyStopping이 너무 일찍 종료

**증상**: 학습 곡선이 아직 수렴하지 않았는데 조기 종료

**해결**: `patience` 값 증가 또는 `min_delta` 추가

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,        # 5 → 10으로 증가
    min_delta=1e-4,     # 0.0001 이상 개선되어야 인정
    restore_best_weights=True
)
```

---

*참고 파일: `src/femto_dl_bn_compare.py` (GRU v2 BN+LN 학습), `src/femto_dl_rul.py` (LSTM baseline)*
