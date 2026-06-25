# ════════════════════════════════════════════════════════════════════
# [역할] FEMTO-ST RUL — DL 아키텍처 5종 비교 (ML 프로젝트 8분류 비교 방식 준용)
# [단계] [4] 모델 비교 — LSTM / GRU / BiLSTM / 1D-CNN / CNN-LSTM
# [작업 메모] GroupKFold(train 3베어링) → OOS(Full_Test_Set 10베어링) 평가
# ════════════════════════════════════════════════════════════════════
"""FEMTO-ST 잔여수명(RUL) DL 아키텍처 비교.

실행:
    python -m src.femto_dl_compare

출력:
    models/femto_dl_compare_results.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "FEMTO_processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 30
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 7


# ── 데이터 로딩 ────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, list[str]]:
    feat_path = PROCESSED_DIR / "femto_features.csv"
    sel_path = PROCESSED_DIR / "selected_features.csv"

    if not feat_path.exists():
        from src.femto_preprocess import run as preprocess_run
        preprocess_run()

    df = pd.read_csv(feat_path)
    if sel_path.exists():
        features = pd.read_csv(sel_path)["feature"].tolist()
    else:
        features = [
            "h_rms", "h_kurt", "h_skew", "h_crest",
            "v_rms", "v_kurt", "v_skew", "v_crest", "temp_mean",
        ]
    return df, features


# ── 시퀀스 생성 ────────────────────────────────────────────────────────────────

def make_sequences(
    df: pd.DataFrame,
    features: list[str],
    window: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list, y_list, g_list = [], [], []
    le = LabelEncoder()
    df = df.copy()
    df["group_id"] = le.fit_transform(df["bearing"])

    for bearing, bdf in df.groupby("bearing"):
        bdf = bdf.sort_values("minute").reset_index(drop=True)
        feat_frame = bdf[features].copy()
        for c in features:
            med = feat_frame[c].median()
            feat_frame[c] = feat_frame[c].fillna(med if np.isfinite(med) else 0.0)
        feat_vals = feat_frame.values.astype(np.float64)
        rul_vals = bdf["rul"].values
        gid = bdf["group_id"].iloc[0]

        for i in range(len(bdf) - window):
            X_list.append(feat_vals[i: i + window])
            y_list.append(rul_vals[i + window])
            g_list.append(gid)

    if not X_list:
        return np.empty((0, window, len(features))), np.empty(0), np.empty(0)
    return np.array(X_list), np.array(y_list, dtype=float), np.array(g_list)


# ── 모델 빌더 정의 ─────────────────────────────────────────────────────────────

def _build_lstm(window: int, n_feat: int, units: int = 64, dropout: float = 0.2):
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, n_feat)),
        tf.keras.layers.LSTM(units, return_sequences=True),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(units // 2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="LSTM")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def _build_gru(window: int, n_feat: int, units: int = 64, dropout: float = 0.2):
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, n_feat)),
        tf.keras.layers.GRU(units, return_sequences=True),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.GRU(units // 2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="GRU")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def _build_bilstm(window: int, n_feat: int, units: int = 64, dropout: float = 0.2):
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, n_feat)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units // 2)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="BiLSTM")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def _build_cnn1d(window: int, n_feat: int, units: int = 64, dropout: float = 0.2):
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, n_feat)),
        tf.keras.layers.Conv1D(units, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(units // 2, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="CNN1D")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def _build_cnn_lstm(window: int, n_feat: int, units: int = 64, dropout: float = 0.2):
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, n_feat)),
        tf.keras.layers.Conv1D(units, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(units // 2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="CNN_LSTM")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


MODEL_BUILDERS: dict[str, Callable] = {
    "LSTM":     _build_lstm,
    "GRU":      _build_gru,
    "BiLSTM":   _build_bilstm,
    "1D-CNN":   _build_cnn1d,
    "CNN-LSTM": _build_cnn_lstm,
}


# ── 단일 모델 학습·평가 ────────────────────────────────────────────────────────

def train_and_evaluate_model(
    name: str,
    builder: Callable,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    X_te: np.ndarray,
    y_te_orig: np.ndarray,
    y_scaler: MinMaxScaler,
    y_range: float,
) -> dict:
    import tensorflow as tf

    n_feat = X_tr.shape[2]
    window = X_tr.shape[1]
    cv = GroupKFold(n_splits=min(3, len(np.unique(groups_tr))))
    y_pred_cv = np.zeros(len(y_tr))

    print(f"\n[{name}] GroupKFold CV 학습 중...")
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_tr, y_tr, groups_tr)):
        model = builder(window, n_feat)
        cb = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        )]
        model.fit(
            X_tr[tr_idx], y_tr[tr_idx],
            validation_data=(X_tr[val_idx], y_tr[val_idx]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=0,
        )
        y_pred_cv[val_idx] = model.predict(X_tr[val_idx], verbose=0).flatten()
        print(f"  Fold {fold+1}/{cv.n_splits} 완료")

    cv_rmse = float(np.sqrt(mean_squared_error(y_tr, y_pred_cv)) * y_range)
    cv_mae  = float(mean_absolute_error(y_tr, y_pred_cv) * y_range)

    # 전체 train으로 최종 모델 재학습
    print(f"  [{name}] 최종 모델 재학습 중...")
    final_model = builder(window, n_feat)
    n_val = max(1, int(len(X_tr) * 0.1))
    final_model.fit(
        X_tr[:-n_val], y_tr[:-n_val],
        validation_data=(X_tr[-n_val:], y_tr[-n_val:]),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)],
        verbose=0,
    )

    # OOS 평가
    if len(X_te):
        oos_sc = final_model.predict(X_te, verbose=0).flatten()
        oos_orig = y_scaler.inverse_transform(oos_sc.reshape(-1, 1)).flatten()
        oos_rmse = float(np.sqrt(mean_squared_error(y_te_orig, oos_orig)))
        oos_mae  = float(mean_absolute_error(y_te_orig, oos_orig))
    else:
        oos_rmse, oos_mae = cv_rmse, cv_mae

    print(f"  [{name}] CV RMSE={cv_rmse:.1f}  OOS RMSE={oos_rmse:.1f} 스냅샷")

    return {
        "cv_rmse":  round(cv_rmse, 2),
        "cv_mae":   round(cv_mae, 2),
        "oos_rmse": round(oos_rmse, 2),
        "oos_mae":  round(oos_mae, 2),
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────

def run() -> None:
    print("=" * 60)
    print("FEMTO-ST DL 아키텍처 5종 비교")
    print("=" * 60)

    try:
        import tensorflow as tf
        print(f"[TensorFlow] v{tf.__version__}")
    except ImportError:
        print("[오류] TensorFlow 미설치 -> 종료")
        return

    df, features = load_data()
    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()
    print(f"[분리] train={df_train['bearing'].nunique()}개  test={df_test['bearing'].nunique()}개")
    print(f"[피처] {features}")

    X_tr, y_tr, groups_tr = make_sequences(df_train, features, WINDOW_SIZE)
    X_te, y_te, _         = make_sequences(df_test,  features, WINDOW_SIZE)
    print(f"[시퀀스] train={len(X_tr)}  test={len(X_te)}")

    n_feat = X_tr.shape[2]

    seq_scaler = MinMaxScaler()
    X_tr_sc = seq_scaler.fit_transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
    X_te_sc = seq_scaler.transform(X_te.reshape(-1, n_feat)).reshape(X_te.shape) if len(X_te) else X_te

    y_scaler = MinMaxScaler()
    y_tr_sc = y_scaler.fit_transform(y_tr.reshape(-1, 1)).flatten()
    y_range  = float(y_tr.max() - y_tr.min()) if len(y_tr) else 1.0

    y_te_orig = y_scaler.inverse_transform(
        y_scaler.transform(y_te.reshape(-1, 1))
    ).flatten() if len(y_te) else np.array([])

    results: dict[str, dict] = {}
    for model_name, builder in MODEL_BUILDERS.items():
        try:
            r = train_and_evaluate_model(
                model_name, builder,
                X_tr_sc, y_tr_sc, groups_tr,
                X_te_sc, y_te_orig, y_scaler, y_range,
            )
            results[model_name] = r
        except Exception as e:
            print(f"  [{model_name}] 오류: {e}")
            results[model_name] = {"oos_rmse": float("nan"), "error": str(e)}

    valid = {k: v for k, v in results.items() if np.isfinite(v.get("oos_rmse", float("nan")))}
    best_model = min(valid, key=lambda k: valid[k]["oos_rmse"]) if valid else None
    results["_best_model"] = best_model

    print("\n[아키텍처 비교 결과 - OOS RMSE (스냅샷 단위)]")
    print(f"{'모델':<12} {'CV RMSE':>10} {'OOS RMSE':>10} {'OOS MAE':>10}")
    print("-" * 44)
    for name, r in results.items():
        if name.startswith("_"):
            continue
        marker = " *" if name == best_model else ""
        cv_r  = r.get("cv_rmse", float("nan"))
        oos_r = r.get("oos_rmse", float("nan"))
        oos_m = r.get("oos_mae", float("nan"))
        print(f"{name:<12} {cv_r:>10.1f} {oos_r:>10.1f} {oos_m:>10.1f}{marker}")

    if best_model:
        print(f"\n[최적 모델] {best_model}  OOS RMSE={valid[best_model]['oos_rmse']:.1f}")

    out = MODEL_DIR / "femto_dl_compare_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out}")
    print("=" * 60)
    print("DL 비교 완료")


if __name__ == "__main__":
    run()
