"""
Source 3: NASA Milling - ML vs DL 비교 (v2 개선판)
v1 실패 원인: RF 피처를 정규화된 데이터에서 추출 -> RMS/std 모두 ~1 로 수렴
개선사항:
  1. RF: 원시신호(역정규화)에서 피처 추출 + StandardScaler
  2. CNN: 세그먼트별 독립 z-score + Focal Loss + GlobalMaxPooling
  3. 평가 임계값 0.5 통일
"""
from __future__ import annotations
import numpy as np
import json, pickle, os, warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONIOENCODING"] = "utf-8"

DATA_DIR = Path("c:/AISOURCE/Homework/DL_FactoryAutomation/data")
MDL_DIR  = Path("c:/AISOURCE/Homework/DL_FactoryAutomation/models")
MDL_DIR.mkdir(exist_ok=True)
THRESHOLD = 0.5

# -- 데이터 로드 + 원시신호 역복원 -----------------------------------------
tr   = np.load(DATA_DIR / "milling_train.npz")
te   = np.load(DATA_DIR / "milling_test.npz")
norm = np.load(DATA_DIR / "milling_norm.npz")

X_tr_n, y_tr = tr["X"].astype("float32"), tr["y"].astype("int32")
X_te_n, y_te = te["X"].astype("float32"), te["y"].astype("int32")
mean_ch = norm["mean"].astype("float32")   # (1,1,6)
std_ch  = norm["std"].astype("float32")    # (1,1,6)

X_tr_raw = X_tr_n * std_ch + mean_ch      # 원시신호 복원
X_te_raw = X_te_n * std_ch + mean_ch

print(f"Train: {X_tr_raw.shape} | Test: {X_te_raw.shape}")
print(f"  마모 비율 - Train: {y_tr.mean()*100:.1f}% | Test: {y_te.mean()*100:.1f}%\n")

# == 1. ML: RF + 원시신호 피처 + StandardScaler =============================
def extract_features(X):
    """(N, 512, 6) -> (N, 48) 원시신호 기반 피처"""
    N, T, C = X.shape
    feats = []
    for i in range(N):
        row = []
        for c in range(C):
            sig = X[i, :, c]
            fft_mag = np.abs(np.fft.rfft(sig))
            row += [
                float(np.sqrt(np.mean(sig**2))),
                float(np.max(np.abs(sig))),
                float(sig.std()),
                float(sp_stats.kurtosis(sig)),
                float(sp_stats.skew(sig)),
                float(np.percentile(sig, 75) - np.percentile(sig, 25)),
                float(np.argmax(fft_mag)),
                float(np.mean(fft_mag)),
            ]
        feats.append(row)
    return np.array(feats, dtype="float32")

print("=== ML v2: RandomForest + 원시신호 피처 + StandardScaler ===")
F_tr = extract_features(X_tr_raw)
F_te = extract_features(X_te_raw)

for ch_name, idx in [("smcAC RMS", 0), ("vib_table RMS", 16), ("AE_spindle RMS", 40)]:
    n_mean = F_tr[y_tr == 0, idx].mean()
    w_mean = F_tr[y_tr == 1, idx].mean()
    print(f"  [{ch_name}] Normal={n_mean:.4f} | Worn={w_mean:.4f} | diff={abs(w_mean-n_mean):.4f}")
print()

scaler = StandardScaler()
F_tr_s = scaler.fit_transform(F_tr)
F_te_s = scaler.transform(F_te)

rf = RandomForestClassifier(
    n_estimators=500, class_weight="balanced",
    random_state=2026, n_jobs=-1,
    min_samples_leaf=2, max_features="sqrt",
)
rf.fit(F_tr_s, y_tr)

ml_prob = rf.predict_proba(F_te_s)[:, 1]
ml_pred = (ml_prob >= THRESHOLD).astype(int)
ml_auc  = roc_auc_score(y_te, ml_prob)
ml_rep  = classification_report(y_te, ml_pred, target_names=["Normal","Worn"], output_dict=True)
print(classification_report(y_te, ml_pred, target_names=["Normal","Worn"]))
print(f"ROC-AUC: {ml_auc:.4f}")
print(f"Confusion:\n{confusion_matrix(y_te, ml_pred)}\n")

with open(MDL_DIR / "milling_rf_model.pkl", "wb") as f:
    pickle.dump((rf, scaler, mean_ch, std_ch), f)

# == 2. DL: 1D-CNN + per-segment norm + Focal Loss =========================
print("=== DL v2: 1D-CNN + 세그먼트별 정규화 + Focal Loss ===")
import tensorflow as tf
from tensorflow import keras

def normalize_per_segment(X):
    """각 윈도우 독립 z-score: 채널간 상대 진폭 유지, 전역 편향 제거"""
    mu  = X.mean(axis=1, keepdims=True)
    sig = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype("float32")

X_tr_ps = normalize_per_segment(X_tr_raw)
X_te_ps = normalize_per_segment(X_te_raw)

def focal_loss(gamma=2.0, alpha=0.65):
    """Focal Loss - 경계 케이스 집중"""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred)
                + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        at = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        return tf.reduce_mean(at * tf.pow(1.0 - pt, gamma) * bce)
    return loss

def build_cnn1d_v2(seq_len=512, n_ch=6):
    inp = keras.Input(shape=(seq_len, n_ch))
    x = keras.layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=focal_loss(gamma=2.0, alpha=0.65),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return m

model = build_cnn1d_v2()
model.summary()

n_normal = int((y_tr == 0).sum())
n_worn   = int((y_tr == 1).sum())
n_total  = len(y_tr)
cw = {0: n_total / (2 * n_normal), 1: n_total / (2 * n_worn)}
print(f"\nClass weights: Normal(0)={cw[0]:.3f}, Worn(1)={cw[1]:.3f}\n")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_auc", patience=10,
        restore_best_weights=True, mode="max",
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc", factor=0.5,
        patience=5, min_lr=1e-6, mode="max",
    ),
]
model.fit(
    X_tr_ps, y_tr,
    validation_split=0.15,
    epochs=60, batch_size=64,
    class_weight=cw,
    callbacks=callbacks,
    verbose=1,
)

dl_prob = model.predict(X_te_ps, verbose=0).flatten()
dl_pred = (dl_prob >= THRESHOLD).astype(int)
dl_auc  = roc_auc_score(y_te, dl_prob)
dl_rep  = classification_report(y_te, dl_pred, target_names=["Normal","Worn"], output_dict=True)
print(f"\n[Test T={THRESHOLD}]")
print(classification_report(y_te, dl_pred, target_names=["Normal","Worn"]))
print(f"ROC-AUC: {dl_auc:.4f}")
print(f"Confusion:\n{confusion_matrix(y_te, dl_pred)}\n")

model.save(str(MDL_DIR / "milling_cnn1d.keras"))

# -- 결과 저장 ---------------------------------------------------------------
result = {
    "version": "v2_improved",
    "preprocessing": "raw_features_for_rf + per_segment_norm_for_cnn + focal_loss",
    "ML_RandomForest": {
        "accuracy":  float(ml_rep["accuracy"]),
        "precision": float(ml_rep["Worn"]["precision"]),
        "recall":    float(ml_rep["Worn"]["recall"]),
        "f1":        float(ml_rep["Worn"]["f1-score"]),
        "roc_auc":   float(ml_auc),
        "method":    "RF(n=500,balanced) + 48 raw features + StandardScaler",
    },
    "DL_1DCNN": {
        "accuracy":  float(dl_rep["accuracy"]),
        "precision": float(dl_rep["Worn"]["precision"]),
        "recall":    float(dl_rep["Worn"]["recall"]),
        "f1":        float(dl_rep["Worn"]["f1-score"]),
        "roc_auc":   float(dl_auc),
        "method":    "1D-CNN(GlobalMaxPool) + per-segment norm + Focal Loss",
        "threshold": THRESHOLD,
    },
}
with open(MDL_DIR / "milling_comparison.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("=" * 60)
print("ML vs DL 직접 비교 v2 (NASA Milling - Source 3)")
print("=" * 60)
for name, r in result.items():
    if isinstance(r, dict) and "accuracy" in r:
        print(f"\n{name}")
        print(f"  Accuracy: {r['accuracy']:.4f}  F1: {r['f1']:.4f}  ROC-AUC: {r['roc_auc']:.4f}")
        print(f"  Recall:   {r['recall']:.4f}  Precision: {r['precision']:.4f}")
print("\n저장 완료: milling_rf_model.pkl  milling_cnn1d.keras  milling_comparison.json")
