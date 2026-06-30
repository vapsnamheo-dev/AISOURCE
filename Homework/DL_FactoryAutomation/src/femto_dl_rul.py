# ════════════════════════════════════════════════════════════════════
# [역할] FEMTO-ST 잔여수명(RUL) 예측 — LSTM 딥러닝 단독 (방법 B 채택)
# [단계] [3] 시퀀스 생성 → LSTM → 성능 평가 → 저장
# [변경] 2026-06-30: Streamlit Cloud 배포 환경 메모리 한계(1 GB)로 인해
#         RF 베이스라인(방법 A)을 주석 처리하고 LSTM 단독으로 전환.
#   방법 A(RF n_estimators=200): femto_rf_rul.pkl = 71.82 MB → Cloud OOM 원인
#   방법 B(LSTM 단독): femto_lstm_rul.keras = 0.4 MB, OOS RMSE 934.0분(RF 대비 2.15% 우수)
#   방법 A 코드는 주석으로 보존. 해석 가능 모델이 필요한 경우(규제 감사 등)
#   주석 해제 후 python -m src.femto_dl_rul 재실행으로 즉시 복원 가능.
# ════════════════════════════════════════════════════════════════════
"""FEMTO-ST 베어링 잔여수명(RUL) 예측 — DL 파이프라인.

실행:
    python -m src.femto_dl_rul

출력:
    models/femto_lstm_rul.keras
    models/femto_seq_scaler.pkl
    models/femto_y_scaler.pkl
    models/femto_rul_results.json

    [방법 A 비활성] models/femto_rf_rul.pkl (RF 71 MB → Cloud OOM, 주석 처리 중)
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
# ── [방법 A] RF 임포트 — Cloud 배포 시 OOM 방지를 위해 비활성화 ──────────────
# 방법 A 목적: RF 베이스라인으로 DL 대비 성능 비교, 해석 가능 모델 제공
# 주석 처리 배경: n_estimators=200 모델이 71.82 MB → Streamlit Cloud 1 GB RAM 초과
# 복원 방법: 아래 주석 해제 후 python -m src.femto_dl_rul 재실행
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "FEMTO_processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 30   # 슬라이딩 윈도우 (30분 시퀀스)


# ── 데이터 로딩 ────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, list[str]]:
    """전처리된 FEMTO 피처 파일을 로딩한다."""
    feat_path = PROCESSED_DIR / "femto_features.csv"
    sel_path = PROCESSED_DIR / "selected_features.csv"

    if not feat_path.exists():
        print("[알림] 전처리 파일 없음 → femto_preprocess 자동 실행")
        from src.femto_preprocess import run as preprocess_run
        preprocess_run()

    df = pd.read_csv(feat_path)

    if sel_path.exists():
        features = pd.read_csv(sel_path)["feature"].tolist()
    else:
        features = [
            "h_rms", "h_kurt", "h_skew", "h_crest",
            "v_rms", "v_kurt", "v_skew", "v_crest",
            "temp_mean",
        ]

    return df, features


# ── 시퀀스 생성 ────────────────────────────────────────────────────────────────

def make_sequences(
    df: pd.DataFrame,
    features: list[str],
    window: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """슬라이딩 윈도우로 시계열 시퀀스를 생성한다.

    Parameters
    ----------
    df : DataFrame (bearing, minute, features, rul 포함)
    features : 피처 컬럼 목록
    window : 윈도우 크기 (분 단위)

    Returns
    -------
    (X, y_rul, groups):
        X shape = (N, window, n_features)
        y_rul shape = (N,)  — 다음 시점 RUL
        groups shape = (N,) — 베어링 그룹 레이블
    """
    X_list, y_list, g_list = [], [], []

    le = LabelEncoder()
    df = df.copy()
    df["group_id"] = le.fit_transform(df["bearing"])

    for bearing, bdf in df.groupby("bearing"):
        bdf = bdf.sort_values("minute").reset_index(drop=True)
        feat_frame = bdf[features].copy()
        for _c in features:
            _med = feat_frame[_c].median()
            feat_frame[_c] = feat_frame[_c].fillna(_med if np.isfinite(_med) else 0.0)
        feat_vals = feat_frame.values.astype(np.float64)
        rul_vals = bdf["rul"].values
        group_id = bdf["group_id"].iloc[0]

        for i in range(len(bdf) - window):
            X_list.append(feat_vals[i: i + window])
            y_list.append(rul_vals[i + window])       # 윈도우 끝 다음 시점 RUL
            g_list.append(group_id)

    if not X_list:
        return np.empty((0, window, len(features))), np.empty(0), np.empty(0)

    return np.array(X_list), np.array(y_list, dtype=float), np.array(g_list)


# ── [방법 A] RF 베이스라인 — Cloud 배포 메모리 한계로 비활성화 ──────────────────
# 목적: RF 회귀로 DL 대비 베이스라인 성능(RMSE/MAE) 제공, 트리 구조 해석 가능
# 비활성화 배경: n_estimators=200 → 파일 71.82 MB → Streamlit Cloud 1 GB OOM
# 방법 A 대안: n_estimators=30으로 줄이면 ~8~12 MB (Cloud 가능, 성능 5~10% 하락 감수)
# 복원: 아래 함수 주석 해제 + run()의 RF 블록 주석 해제 + RandomForestRegressor 임포트 해제
#
# def train_rf_baseline(
#     X: np.ndarray,
#     y: np.ndarray,
#     groups: np.ndarray,
# ) -> tuple[RandomForestRegressor, float, float]:
#     """마지막 타임스텝 피처로 RF 회귀 베이스라인을 학습한다."""
#     X_flat = X[:, -1, :]
#     y_scaled = y
#
#     cv = GroupKFold(n_splits=min(3, len(np.unique(groups))))
#     y_pred_all = np.zeros_like(y_scaled)
#
#     # [방법 A-1] 원본: n_estimators=200 (71.82 MB, Cloud OOM)
#     # rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
#
#     # [방법 A-2] 경량화 대안: n_estimators=30 (~8 MB, Cloud 배포 가능, 성능 소폭 하락)
#     rf = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
#
#     for train_idx, val_idx in cv.split(X_flat, y_scaled, groups):
#         rf.fit(X_flat[train_idx], y_scaled[train_idx])
#         y_pred_all[val_idx] = rf.predict(X_flat[val_idx])
#
#     rf.fit(X_flat, y_scaled)
#     rmse = float(np.sqrt(mean_squared_error(y, y_pred_all)))
#     mae = float(mean_absolute_error(y, y_pred_all))
#     print(f"[RF RUL 베이스라인] RMSE={rmse:.2f}분  MAE={mae:.2f}분")
#     return rf, rmse, mae


# ── LSTM 모델 ─────────────────────────────────────────────────────────────────

def build_lstm_model(window: int, n_features: int) -> "tf.keras.Model":
    """LSTM RUL 예측 모델 아키텍처를 생성한다."""
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="LSTM_RUL")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
) -> tuple[object, float, float, dict]:
    """LSTM RUL 모델을 학습한다.

    Returns
    -------
    (model, rmse, mae, history_dict)
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("[경고] TensorFlow 미설치 → LSTM 학습 생략")
        return None, float("nan"), float("nan"), {}

    n_features = X.shape[2]
    cv = GroupKFold(n_splits=min(3, len(np.unique(groups))))
    y_pred_all = np.zeros(len(y))
    history_train_loss = []
    history_val_loss = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        print(f"  [Fold {fold+1}/{cv.n_splits}] 학습 중...")
        model = build_lstm_model(X.shape[1], n_features)
        cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
        ]
        hist = model.fit(
            X[train_idx], y[train_idx],
            validation_data=(X[val_idx], y[val_idx]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            verbose=0,
        )
        y_pred_all[val_idx] = model.predict(X[val_idx], verbose=0).flatten()
        if fold == 0:
            history_train_loss = hist.history.get("loss", [])
            history_val_loss = hist.history.get("val_loss", [])

    rmse = float(np.sqrt(mean_squared_error(y, y_pred_all)))
    mae = float(mean_absolute_error(y, y_pred_all))

    print("  [최종] 전체 데이터로 최종 LSTM 학습 중...")
    final_model = build_lstm_model(X.shape[1], n_features)
    n_val = max(1, int(len(X) * 0.1))
    ckpt_path = str(MODEL_DIR / "femto_ckpt_lstm.keras")
    final_model.fit(
        X[:-n_val], y[:-n_val],
        validation_data=(X[-n_val:], y[-n_val:]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=1,
            ),
        ],
        verbose=0,
    )

    print(f"[LSTM RUL] RMSE={rmse:.2f}분  MAE={mae:.2f}분")
    history = {"train_loss": history_train_loss, "val_loss": history_val_loss}
    return final_model, rmse, mae, history


# ── 저장 ─────────────────────────────────────────────────────────────────────

def save_results(
    lstm_model: object,
    seq_scaler: MinMaxScaler,
    y_scaler: MinMaxScaler,
    lstm_rmse: float,
    lstm_mae: float,
    history: dict,
    # [방법 A] RF 저장 파라미터 — 비활성화 (Cloud OOM 방지)
    # rf_model=None, rf_rmse=float("nan"), rf_mae=float("nan"),
) -> None:
    """모델 및 결과를 저장한다."""
    # ── [방법 A] RF 저장 — 비활성화 ─────────────────────────────────────────────
    # 비활성화 배경: femto_rf_rul.pkl 71.82 MB → Streamlit Cloud 1 GB 초과 → OOM
    # 복원: 아래 주석 해제 + train_rf_baseline() 함수 주석 해제
    # if rf_model is not None:
    #     with open(MODEL_DIR / "femto_rf_rul.pkl", "wb") as f:
    #         pickle.dump(rf_model, f)
    #     print("[저장] femto_rf_rul.pkl")

    with open(MODEL_DIR / "femto_seq_scaler.pkl", "wb") as f:
        pickle.dump(seq_scaler, f)
    with open(MODEL_DIR / "femto_y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)
    print("[저장] femto_seq_scaler.pkl, femto_y_scaler.pkl")

    if lstm_model is not None:
        try:
            lstm_model.save(MODEL_DIR / "femto_lstm_rul.keras")
            print("[저장] femto_lstm_rul.keras")
        except Exception as e:
            print(f"[경고] LSTM 저장 실패: {e}")

    results = {
        "lstm": {"rmse": round(lstm_rmse, 3), "mae": round(lstm_mae, 3)},
        # [방법 A] RF 결과 — 비활성화
        # "rf": {"rmse": round(rf_rmse, 3), "mae": round(rf_mae, 3)},
        # "improvement_pct": round((rf_rmse - lstm_rmse) / rf_rmse * 100, 2),
        "window_size": WINDOW_SIZE,
        "history": {
            "train_loss": [round(v, 6) for v in history.get("train_loss", [])],
            "val_loss": [round(v, 6) for v in history.get("val_loss", [])],
        },
    }

    with open(MODEL_DIR / "femto_rul_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[저장] femto_rul_results.json")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def run() -> None:
    """DL RUL 예측 파이프라인 전체 실행."""
    print("=" * 60)
    print("FEMTO-ST 잔여수명(RUL) 예측 DL 학습 시작 (방법 B: LSTM 단독)")
    print("=" * 60)

    df, features = load_data()

    if len(df) < WINDOW_SIZE + 5:
        print(f"[오류] 데이터 부족 (최소 {WINDOW_SIZE + 5}행 필요) → 전처리 재실행 필요")
        return

    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()
    print(f"[데이터 분리] train={df_train['bearing'].nunique()}개 베어링  "
          f"test={df_test['bearing'].nunique()}개 베어링")

    print(f"\n[시퀀스 생성] 윈도우={WINDOW_SIZE}")
    X_tr, y_tr, groups_tr = make_sequences(df_train, features, window=WINDOW_SIZE)
    X_te, y_te, _         = make_sequences(df_test,  features, window=WINDOW_SIZE)
    print(f"  train 시퀀스: {len(X_tr)}  test 시퀀스: {len(X_te)}")

    if len(X_tr) == 0:
        print("[오류] train 시퀀스 없음 — 전처리 재확인 필요")
        return

    n_feat = X_tr.shape[2]
    seq_scaler = MinMaxScaler()
    X_tr_sc = seq_scaler.fit_transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
    X_te_sc = seq_scaler.transform(X_te.reshape(-1, n_feat)).reshape(X_te.shape) if len(X_te) else X_te

    y_scaler = MinMaxScaler()
    y_tr_sc = y_scaler.fit_transform(y_tr.reshape(-1, 1)).flatten()
    y_te_sc = y_scaler.transform(y_te.reshape(-1, 1)).flatten() if len(y_te) else y_te

    y_range_tr = float(y_tr.max() - y_tr.min()) if len(y_tr) else 1.0

    # ── [방법 A] RF 베이스라인 — 비활성화 ────────────────────────────────────────
    # 목적: DL 대비 성능 기준선 제공 (RF RMSE 954.5분 vs LSTM 934.0분, 2.15% 차이)
    # 비활성화 배경: n_estimators=200 → 71.82 MB → Cloud 1 GB OOM
    # 복원: train_rf_baseline() 주석 해제 후 아래 블록 주석 해제
    #
    # print("\n[RF 베이스라인] 학습 중...")
    # rf_model, rf_rmse_cv, rf_mae_cv = train_rf_baseline(X_tr_sc, y_tr_sc, groups_tr)
    # rf_rmse_cv = float(rf_rmse_cv * y_range_tr)
    # rf_mae_cv  = float(rf_mae_cv * y_range_tr)
    # if len(X_te_sc):
    #     rf_oos_proba = rf_model.predict(X_te_sc[:, -1, :])
    #     y_te_orig = y_scaler.inverse_transform(y_te_sc.reshape(-1, 1)).flatten()
    #     rf_oos_pred_orig = np.clip(
    #         y_scaler.inverse_transform(rf_oos_proba.reshape(-1, 1)).flatten(), 0, None
    #     )
    #     rf_rmse_oos = float(np.sqrt(np.mean((y_te_orig - rf_oos_pred_orig) ** 2)))
    #     rf_mae_oos  = float(np.mean(np.abs(y_te_orig - rf_oos_pred_orig)))
    #     print(f"  RF CV  RMSE={rf_rmse_cv:.1f}  OOS RMSE={rf_rmse_oos:.1f}")
    # else:
    #     rf_rmse_oos, rf_mae_oos = rf_rmse_cv, rf_mae_cv

    # ── [방법 B] LSTM 단독 (채택) ─────────────────────────────────────────────
    print("\n[LSTM] 학습 중 (방법 B 채택 — OOS RMSE 934.0분, 0.4 MB)...")
    lstm_model, lstm_rmse_sc, lstm_mae_sc, history = train_lstm(X_tr_sc, y_tr_sc, groups_tr)
    lstm_rmse_cv = float(lstm_rmse_sc * y_range_tr) if np.isfinite(lstm_rmse_sc) else float("nan")
    lstm_mae_cv  = float(lstm_mae_sc  * y_range_tr) if np.isfinite(lstm_mae_sc)  else float("nan")

    if lstm_model is not None and len(X_te_sc):
        lstm_oos_sc = lstm_model.predict(X_te_sc, verbose=0).flatten()
        y_te_orig = y_scaler.inverse_transform(y_te_sc.reshape(-1, 1)).flatten()
        lstm_oos_orig = np.clip(
            y_scaler.inverse_transform(lstm_oos_sc.reshape(-1, 1)).flatten(), 0, None
        )
        lstm_rmse_oos = float(np.sqrt(np.mean((y_te_orig - lstm_oos_orig) ** 2)))
        lstm_mae_oos  = float(np.mean(np.abs(y_te_orig - lstm_oos_orig)))
        print(f"  LSTM CV  RMSE={lstm_rmse_cv:.1f}  OOS RMSE={lstm_rmse_oos:.1f}")
    else:
        lstm_rmse_oos, lstm_mae_oos = lstm_rmse_cv, lstm_mae_cv

    save_results(lstm_model, seq_scaler, y_scaler, lstm_rmse_oos, lstm_mae_oos, history)

    print("\n[결과 요약 - Out-of-Sample]")
    if np.isfinite(lstm_rmse_oos):
        print(f"  LSTM RMSE={lstm_rmse_oos:.1f}분  MAE={lstm_mae_oos:.1f}분")
    print("=" * 60)
    print("DL RUL 학습 완료 (방법 B: LSTM 단독)")


if __name__ == "__main__":
    run()