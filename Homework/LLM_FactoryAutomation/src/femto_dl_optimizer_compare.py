# -*- coding: utf-8 -*-
"""FEMTO-ST GRU optimizer 비교 실험 (Adam / Nadam / AdamW).

실행:
    python -m src.femto_dl_optimizer_compare

출력:
    models/femto_optimizer_results.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

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
UNITS = 64
DROPOUT = 0.2
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 7


def load_data():
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


def make_sequences(df, features, window=WINDOW_SIZE):
    X_list, y_list, g_list = [], [], []
    le = LabelEncoder()
    df = df.copy()
    df["group_id"] = le.fit_transform(df["bearing"])

    for bearing, bdf in df.groupby("bearing"):
        bdf = bdf.sort_values("minute").reset_index(drop=True)
        feat_frame = bdf[features].copy()
        # 선형 보간 → 끝단 forward/backfill → 그래도 남으면 0
        for c in features:
            feat_frame[c] = (
                feat_frame[c]
                .interpolate(method="linear")
                .bfill()
                .ffill()
                .fillna(0.0)
            )
        feat_vals = feat_frame.values.astype(np.float64)
        rul_vals = bdf["rul"].values
        gid = bdf["group_id"].iloc[0]

        for i in range(len(bdf) - window):
            X_list.append(feat_vals[i: i + window])
            y_list.append(rul_vals[i + window])
            g_list.append(gid)

    return np.array(X_list), np.array(y_list, dtype=float), np.array(g_list)


def build_gru(window, n_feat, opt_name):
    import tensorflow as tf
    opts = {
        "Adam":  tf.keras.optimizers.Adam(learning_rate=1e-3),
        "Nadam": tf.keras.optimizers.Nadam(learning_rate=1e-3),
        "AdamW": tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    }
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, n_feat)),
        tf.keras.layers.GRU(UNITS, return_sequences=True),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.GRU(UNITS // 2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name=f"GRU_{opt_name}")
    m.compile(optimizer=opts[opt_name], loss="mse", metrics=["mae"])
    return m


def train_and_evaluate(opt_name, X_tr, y_tr, groups_tr, X_te, y_te_orig, y_scaler, y_range):
    import tensorflow as tf

    n_feat = X_tr.shape[2]
    window = X_tr.shape[1]
    cv = GroupKFold(n_splits=min(3, len(np.unique(groups_tr))))
    y_pred_cv = np.zeros(len(y_tr))

    print(f"\n[GRU-{opt_name}] CV 학습 중...")
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_tr, y_tr, groups_tr)):
        model = build_gru(window, n_feat, opt_name)
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

    print(f"  [GRU-{opt_name}] 최종 모델 재학습 중...")
    final_model = build_gru(window, n_feat, opt_name)
    n_val = max(1, int(len(X_tr) * 0.1))
    final_model.fit(
        X_tr[:-n_val], y_tr[:-n_val],
        validation_data=(X_tr[-n_val:], y_tr[-n_val:]),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)],
        verbose=0,
    )

    if len(X_te):
        oos_sc = final_model.predict(X_te, verbose=0).flatten()
        oos_orig = y_scaler.inverse_transform(oos_sc.reshape(-1, 1)).flatten()
        oos_rmse = float(np.sqrt(mean_squared_error(y_te_orig, oos_orig)))
        oos_mae  = float(mean_absolute_error(y_te_orig, oos_orig))
    else:
        oos_rmse, oos_mae = cv_rmse, cv_mae

    print(f"  [GRU-{opt_name}] CV RMSE={cv_rmse:.1f}  OOS RMSE={oos_rmse:.1f}")
    return {
        "cv_rmse":  round(cv_rmse, 2),
        "cv_mae":   round(cv_mae, 2),
        "oos_rmse": round(oos_rmse, 2),
        "oos_mae":  round(oos_mae, 2),
    }


def run():
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    print("=== FEMTO-ST GRU Optimizer 비교 실험 (Adam / Nadam / AdamW) ===")
    df, features = load_data()

    mask_train = df["split"] == "train"
    df_tr = df[mask_train].copy()
    df_te = df[~mask_train].copy()

    y_scaler = MinMaxScaler()
    y_all = df_tr["rul"].values.reshape(-1, 1)
    y_scaler.fit(y_all)
    y_range = float(y_scaler.data_max_[0] - y_scaler.data_min_[0])

    df_tr = df_tr.copy()
    df_tr["rul"] = y_scaler.transform(df_tr["rul"].values.reshape(-1, 1)).flatten()

    X_tr, y_tr, groups_tr = make_sequences(df_tr, features, WINDOW_SIZE)

    if len(df_te):
        y_te_orig = df_te["rul"].values
        df_te_sc = df_te.copy()
        df_te_sc["rul"] = y_scaler.transform(df_te_sc["rul"].values.reshape(-1, 1)).flatten()
        X_te, _, _ = make_sequences(df_te_sc, features, WINDOW_SIZE)
        y_te_aligned = y_te_orig[-len(X_te):] if len(X_te) <= len(y_te_orig) else y_te_orig
    else:
        X_te = np.empty((0, WINDOW_SIZE, len(features)))
        y_te_aligned = np.empty(0)

    print(f"Train: {X_tr.shape}, OOS: {X_te.shape}")

    results = {}
    for opt_name in ["Adam", "Nadam", "AdamW"]:
        results[opt_name] = train_and_evaluate(
            opt_name, X_tr, y_tr, groups_tr,
            X_te, y_te_aligned, y_scaler, y_range
        )

    out_path = MODEL_DIR / "femto_optimizer_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


if __name__ == "__main__":
    run()
