# ════════════════════════════════════════════════════════════════════
# [역할] FEMTO-ST RUL — 최적 DL 모델 하이퍼파라미터 그리드서치 튜닝
# [단계] [5] 모델 선정 후 하이퍼파라미터 튜닝 (ML 프로젝트 GridSearchCV 방식 준용)
# [작업 메모] window_size x units x dropout 그리드로 OOS RMSE 최적화
# ════════════════════════════════════════════════════════════════════
"""FEMTO-ST RUL 최적 DL 모델 하이퍼파라미터 튜닝.

femto_dl_compare.py 실행 후 선정된 최적 모델을 대상으로
window_size / units / dropout 조합의 그리드서치를 수행한다.

실행:
    python -m src.femto_dl_tune

출력:
    models/femto_dl_tune_results.json
    models/femto_best_dl_tuned.keras
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from itertools import product

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

PARAM_GRID = {
    "window_size": [20, 30, 50],
    "units":       [32, 64, 128],
    "dropout":     [0.1, 0.2, 0.3],
}

EPOCHS     = 50
BATCH_SIZE = 32
PATIENCE   = 7


# ── 최적 모델명 조회 ──────────────────────────────────────────────────────────

def get_best_model_name() -> str:
    compare_path = MODEL_DIR / "femto_dl_compare_results.json"
    if compare_path.exists():
        with open(compare_path, encoding="utf-8") as f:
            data = json.load(f)
        best = data.get("_best_model")
        if best:
            print(f"[최적 모델] 비교 결과 로딩: {best}")
            return best
    print("[알림] 비교 결과 없음 -> 기본값 LSTM 사용")
    return "LSTM"


# ── 모델 빌더 ─────────────────────────────────────────────────────────────────

def build_model(name: str, window: int, n_feat: int, units: int, dropout: float):
    import tensorflow as tf

    if name == "LSTM":
        m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(window, n_feat)),
            tf.keras.layers.LSTM(units, return_sequences=True),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.LSTM(units // 2),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1),
        ])
    elif name == "GRU":
        m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(window, n_feat)),
            tf.keras.layers.GRU(units, return_sequences=True),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.GRU(units // 2),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1),
        ])
    elif name == "BiLSTM":
        m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(window, n_feat)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units // 2)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1),
        ])
    elif name == "1D-CNN":
        m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(window, n_feat)),
            tf.keras.layers.Conv1D(units, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(units // 2, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1),
        ])
    else:  # CNN-LSTM
        m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(window, n_feat)),
            tf.keras.layers.Conv1D(units, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.LSTM(units // 2),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1),
        ])

    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


# ── 데이터 유틸 ────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, list[str]]:
    feat_path = PROCESSED_DIR / "femto_features.csv"
    sel_path  = PROCESSED_DIR / "selected_features.csv"
    if not feat_path.exists():
        from src.femto_preprocess import run as preprocess_run
        preprocess_run()
    df = pd.read_csv(feat_path)
    if sel_path.exists():
        features = pd.read_csv(sel_path)["feature"].tolist()
    else:
        features = ["h_rms", "h_kurt", "h_skew", "h_crest",
                    "v_rms", "v_kurt", "v_skew", "v_crest", "temp_mean"]
    return df, features


def make_sequences(df: pd.DataFrame, features: list[str], window: int):
    X_list, y_list, g_list = [], [], []
    le = LabelEncoder()
    df = df.copy()
    df["group_id"] = le.fit_transform(df["bearing"])
    for _, bdf in df.groupby("bearing"):
        bdf = bdf.sort_values("minute").reset_index(drop=True)
        feat_frame = bdf[features].copy()
        for c in features:
            med = feat_frame[c].median()
            feat_frame[c] = feat_frame[c].fillna(med if np.isfinite(med) else 0.0)
        feat_vals = feat_frame.values.astype(np.float64)
        rul_vals  = bdf["rul"].values
        gid       = bdf["group_id"].iloc[0]
        for i in range(len(bdf) - window):
            X_list.append(feat_vals[i: i + window])
            y_list.append(rul_vals[i + window])
            g_list.append(gid)
    if not X_list:
        return np.empty((0, window, len(features))), np.empty(0), np.empty(0)
    return np.array(X_list), np.array(y_list, dtype=float), np.array(g_list)


# ── 단일 파라미터 조합 평가 ────────────────────────────────────────────────────

def evaluate_params(
    model_name: str,
    window: int,
    units: int,
    dropout: float,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
) -> tuple[dict, object]:
    import tensorflow as tf

    X_tr, y_tr, groups_tr = make_sequences(df_train, features, window)
    X_te, y_te, _         = make_sequences(df_test,  features, window)

    n_feat = X_tr.shape[2]

    seq_scaler = MinMaxScaler()
    X_tr_sc = seq_scaler.fit_transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
    X_te_sc = seq_scaler.transform(X_te.reshape(-1, n_feat)).reshape(X_te.shape) if len(X_te) else X_te

    y_scaler = MinMaxScaler()
    y_tr_sc  = y_scaler.fit_transform(y_tr.reshape(-1, 1)).flatten()
    y_range  = float(y_tr.max() - y_tr.min()) if len(y_tr) else 1.0

    cv = GroupKFold(n_splits=min(3, len(np.unique(groups_tr))))
    y_pred_cv = np.zeros(len(y_tr_sc))
    for _, (tr_idx, val_idx) in enumerate(cv.split(X_tr_sc, y_tr_sc, groups_tr)):
        m = build_model(model_name, window, n_feat, units, dropout)
        cb = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        )]
        m.fit(X_tr_sc[tr_idx], y_tr_sc[tr_idx],
              validation_data=(X_tr_sc[val_idx], y_tr_sc[val_idx]),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=0)
        y_pred_cv[val_idx] = m.predict(X_tr_sc[val_idx], verbose=0).flatten()

    cv_rmse = float(np.sqrt(mean_squared_error(y_tr_sc, y_pred_cv)) * y_range)

    final_m = build_model(model_name, window, n_feat, units, dropout)
    n_val = max(1, int(len(X_tr_sc) * 0.1))
    final_m.fit(
        X_tr_sc[:-n_val], y_tr_sc[:-n_val],
        validation_data=(X_tr_sc[-n_val:], y_tr_sc[-n_val:]),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)],
        verbose=0,
    )

    if len(X_te_sc):
        oos_sc    = final_m.predict(X_te_sc, verbose=0).flatten()
        oos_orig  = y_scaler.inverse_transform(oos_sc.reshape(-1, 1)).flatten()
        y_te_orig = y_scaler.inverse_transform(
            y_scaler.transform(y_te.reshape(-1, 1))
        ).flatten()
        oos_rmse = float(np.sqrt(mean_squared_error(y_te_orig, oos_orig)))
        oos_mae  = float(mean_absolute_error(y_te_orig, oos_orig))
    else:
        oos_rmse = oos_mae = cv_rmse

    result = {
        "window_size": window,
        "units":       units,
        "dropout":     dropout,
        "cv_rmse":     round(cv_rmse, 2),
        "oos_rmse":    round(oos_rmse, 2),
        "oos_mae":     round(oos_mae, 2),
    }
    return result, final_m


# ── 메인 ─────────────────────────────────────────────────────────────────────

def run() -> None:
    print("=" * 60)
    print("FEMTO-ST DL 하이퍼파라미터 그리드서치 튜닝")
    print("=" * 60)

    try:
        import tensorflow as tf
        print(f"[TensorFlow] v{tf.__version__}")
    except ImportError:
        print("[오류] TensorFlow 미설치 -> 종료")
        return

    model_name = get_best_model_name()
    df, features = load_data()
    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()
    print(f"[모델] {model_name}  train={df_train['bearing'].nunique()}개  "
          f"test={df_test['bearing'].nunique()}개")
    print(f"[피처] {features}")

    windows  = PARAM_GRID["window_size"]
    units_l  = PARAM_GRID["units"]
    dropouts = PARAM_GRID["dropout"]
    total    = len(windows) * len(units_l) * len(dropouts)
    print(f"\n[그리드] window={windows}  units={units_l}  dropout={dropouts}  총 {total}조합")

    grid_results = []
    best_oos      = float("inf")
    best_model_obj = None
    best_params    = {}

    for idx, (w, u, d) in enumerate(product(windows, units_l, dropouts), 1):
        print(f"\n[{idx:>2}/{total}] window={w}  units={u}  dropout={d}")
        try:
            r, model_obj = evaluate_params(model_name, w, u, d, df_train, df_test, features)
            print(f"  CV RMSE={r['cv_rmse']:.1f}  OOS RMSE={r['oos_rmse']:.1f}")
            grid_results.append(r)
            if r["oos_rmse"] < best_oos:
                best_oos        = r["oos_rmse"]
                best_model_obj  = model_obj
                best_params     = {"window_size": w, "units": u, "dropout": d}
        except Exception as e:
            print(f"  오류: {e}")
            grid_results.append({
                "window_size": w, "units": u, "dropout": d,
                "oos_rmse": None, "error": str(e),
            })

    valid = [r for r in grid_results if r.get("oos_rmse") is not None]
    valid.sort(key=lambda r: r["oos_rmse"])

    print("\n[그리드서치 결과 - OOS RMSE 상위 10개]")
    print(f"{'window':>8} {'units':>6} {'dropout':>8} {'CV RMSE':>10} {'OOS RMSE':>10}")
    print("-" * 46)
    for r in valid[:10]:
        is_best = (r["window_size"] == best_params.get("window_size") and
                   r["units"]       == best_params.get("units") and
                   r["dropout"]     == best_params.get("dropout"))
        marker = " *" if is_best else ""
        print(f"{r['window_size']:>8} {r['units']:>6} {r['dropout']:>8.1f} "
              f"{r.get('cv_rmse', float('nan')):>10.1f} {r['oos_rmse']:>10.1f}{marker}")

    print(f"\n[최적 파라미터] {best_params}  OOS RMSE={best_oos:.1f}")

    if best_model_obj is not None:
        try:
            best_model_obj.save(MODEL_DIR / "femto_best_dl_tuned.keras")
            print("[저장] femto_best_dl_tuned.keras")
        except Exception as e:
            print(f"[경고] 모델 저장 실패: {e}")

    tune_results = {
        "model_name":    model_name,
        "best_params":   best_params,
        "best_oos_rmse": round(best_oos, 2),
        "grid_results":  grid_results,
    }
    out = MODEL_DIR / "femto_dl_tune_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tune_results, f, ensure_ascii=False, indent=2)
    print(f"[저장] {out}")
    print("=" * 60)
    print("하이퍼파라미터 튜닝 완료")


if __name__ == "__main__":
    run()
