"""
Source 3: NASA Milling - ML 베이스라인 + DL 1D-CNN 비교 학습
ML: RandomForest + 통계/FFT 핸드크래프트 피처
DL: 멀티채널 1D-CNN (원시 신호 직접 처리)
"""
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy import stats as sp_stats
import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DATA_DIR = Path("c:/AISOURCE/Homework/DL_FactoryAutomation/data")
MDL_DIR  = Path("c:/AISOURCE/Homework/DL_FactoryAutomation/models")
MDL_DIR.mkdir(exist_ok=True)
THRESHOLD = 0.75

# ── 데이터 로드 ─────────────────────────────────────────────────────────────
tr = np.load(DATA_DIR / "milling_train.npz")
te = np.load(DATA_DIR / "milling_test.npz")
X_tr, y_tr = tr["X"].astype("float32"), tr["y"].astype("int32")
X_te, y_te = te["X"].astype("float32"), te["y"].astype("int32")
print(f"Train: {X_tr.shape} | Test: {X_te.shape}")
print(f"  마모 비율 - Train: {y_tr.mean()*100:.1f}% | Test: {y_te.mean()*100:.1f}%\n")

# ══════════════════════════════════════════════════════════════════
# 1. ML 베이스라인 - RandomForest + 핸드크래프트 피처
# ══════════════════════════════════════════════════════════════════
def extract_features(X):
    """(N, 512, 6) → (N, 48) 통계+스펙트럼 피처"""
    N, T, C = X.shape
    feats = []
    for i in range(N):
        row = []
        for c in range(C):
            sig = X[i, :, c]
            # 시계열 통계 (6종)
            row += [
                np.sqrt(np.mean(sig**2)),      # RMS
                np.max(np.abs(sig)),            # 피크
                sig.std(),                      # 표준편차
                float(sp_stats.kurtosis(sig)),  # 첨도
                float(sp_stats.skew(sig)),      # 왜도
                np.percentile(sig, 75) - np.percentile(sig, 25),  # IQR
                # FFT 스펙트럼 (2종)
                np.argmax(np.abs(np.fft.rfft(sig))),  # 주 주파수 빈
                np.mean(np.abs(np.fft.rfft(sig))),    # 평균 스펙트럼 크기
            ]
        feats.append(row)
    return np.array(feats, dtype="float32")  # (N, 48)

print("=== ML 베이스라인 (RandomForest + 48 핸드크래프트 피처) ===")
F_tr = extract_features(X_tr)
F_te = extract_features(X_te)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=2026, n_jobs=-1)
rf.fit(F_tr, y_tr)

ml_prob  = rf.predict_proba(F_te)[:, 1]
ml_pred  = (ml_prob >= 0.5).astype(int)
ml_auc   = roc_auc_score(y_te, ml_prob)
ml_rep   = classification_report(y_te, ml_pred, target_names=["Normal","Worn"], output_dict=True)
print(classification_report(y_te, ml_pred, target_names=["Normal","Worn"]))
print(f"ROC-AUC: {ml_auc:.4f}\n")

import pickle
with open(MDL_DIR / "milling_rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

# ══════════════════════════════════════════════════════════════════
# 2. DL - 멀티채널 1D-CNN
# ══════════════════════════════════════════════════════════════════
print("=== DL - 멀티채널 1D-CNN (원시신호 직접 처리) ===")
import tensorflow as tf
from tensorflow import keras

def build_cnn1d_milling(seq_len=512, n_ch=6):
    inp = keras.Input(shape=(seq_len, n_ch))
    x = keras.layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model

model = build_cnn1d_milling()
model.summary()

cw = {0: y_tr.mean(), 1: 1 - y_tr.mean()}  # 역가중치 (마모 클래스 강조)
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
]
history = model.fit(
    X_tr, y_tr,
    validation_split=0.15,
    epochs=50, batch_size=64,
    class_weight=cw,
    callbacks=callbacks,
    verbose=1,
)

# 평가
dl_prob = model.predict(X_te, verbose=0).flatten()
dl_pred = (dl_prob >= THRESHOLD).astype(int)
dl_auc  = roc_auc_score(y_te, dl_prob)
dl_rep  = classification_report(y_te, dl_pred, target_names=["Normal","Worn"], output_dict=True)
print(f"\n[Test - T*={THRESHOLD}]")
print(classification_report(y_te, dl_pred, target_names=["Normal","Worn"]))
print(f"ROC-AUC: {dl_auc:.4f}")
cm = confusion_matrix(y_te, dl_pred)
print(f"Confusion Matrix:\n{cm}")

model.save(str(MDL_DIR / "milling_cnn1d.keras"))

# ── 비교 요약 ──────────────────────────────────────────────────────────────
result = {
    "ML_RandomForest": {
        "accuracy":  ml_rep["accuracy"],
        "precision": ml_rep["Worn"]["precision"],
        "recall":    ml_rep["Worn"]["recall"],
        "f1":        ml_rep["Worn"]["f1-score"],
        "roc_auc":   ml_auc,
        "method":    "RF + 48개 핸드크래프트 피처 (통계+FFT)",
    },
    "DL_1DCNN": {
        "accuracy":  dl_rep["accuracy"],
        "precision": dl_rep["Worn"]["precision"],
        "recall":    dl_rep["Worn"]["recall"],
        "f1":        dl_rep["Worn"]["f1-score"],
        "roc_auc":   dl_auc,
        "method":    "멀티채널 1D-CNN 원시신호 직접 처리",
        "threshold": THRESHOLD,
    },
}
with open(MDL_DIR / "milling_comparison.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("ML vs DL 직접 비교 (NASA Milling - Source 3)")
print("="*60)
for name, r in result.items():
    print(f"\n{name} ({r['method']})")
    print(f"  Accuracy: {r['accuracy']:.4f} | F1: {r['f1']:.4f} | ROC-AUC: {r['roc_auc']:.4f}")
    print(f"  Recall: {r['recall']:.4f} | Precision: {r['precision']:.4f}")
print("\n저장: models/milling_rf_model.pkl, milling_cnn1d.keras, milling_comparison.json")

