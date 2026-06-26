"""
1. 호출자: 독립 실행 (python -m src.femto_dl_bn_compare) - import 없음
2. 중복 없음: femto_dl_compare=5모델 아키텍처, femto_dl_tune=그리드서치, femto_dl_optimizer_compare=옵티마이저 비교 — BN/LN 효과 전용 없음
3. 입출력: 읽음=data/FEMTO_processed/femto_features.csv(bearing,minute,rul,split,h_rms,h_kurt,h_skew,h_crest,v_rms,v_kurt,v_skew,v_crest,temp_mean), 씀=models/femto_dl_bn_compare_results.json + models/femto_best_dl_GRU_v2.keras
4. 사용자 지시(원문): "성능개선 위해 배치 정규화, dropout도 실시했지? 안 했으면 추가해서 성능비교해죠."

GRU v1(기존: Dropout만) vs v2(LayerNorm+BN+Dropout 강화) 성능 비교.
최적 파라미터(window=20, units=32, dropout=0.1, batch=16) 고정 후 정규화 레이어 효과만 측정.

실행:
    python -m src.femto_dl_bn_compare

출력:
    models/femto_dl_bn_compare_results.json
    models/femto_best_dl_GRU_v2.keras  (개선된 경우에만)
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

# 그리드서치 + 배치 튜닝으로 확정된 최적 파라미터
WINDOW_SIZE = 20
UNITS       = 32
DROPOUT     = 0.1
BATCH_SIZE  = 16   # 배치 튜닝에서 OOS RMSE 836.6분 달성
EPOCHS      = 80
PATIENCE    = 10


# ── 데이터 로드 및 시퀀스 생성 ─────────────────────────────────────────────────

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


# ── 모델 빌더 ─────────────────────────────────────────────────────────────────

def build_gru_v1(window: int, n_feat: int):
    """v1 기존 구조: GRU → Dropout → GRU → Dense(16) → Dense(1)."""
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, n_feat)),
        tf.keras.layers.GRU(UNITS, return_sequences=True),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.GRU(UNITS // 2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="GRU_v1_baseline")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def build_gru_v2(window: int, n_feat: int):
    """v2 개선 구조: GRU → LayerNorm → Dropout → GRU → LayerNorm → BN → Dense(32) → Dropout → Dense(1).

    - LayerNormalization: GRU 직후 — 배치 통계 대신 피처 축 정규화 (시퀀스에 안정적)
    - BatchNormalization: Dense 앞 — 시간 독립 구간, 안정적 수렴에 효과적
    - Dense 뉴런 수 16→32: 표현력 확장
    - Dense 뒤 Dropout 추가: 과적합 억제
    """
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, n_feat)),
        tf.keras.layers.GRU(UNITS, return_sequences=True),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.GRU(UNITS // 2),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1),
    ], name="GRU_v2_BN_LN")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


# ── 학습·평가 공통 함수 ────────────────────────────────────────────────────────

def train_and_evaluate(
    label: str,
    builder,
    X_tr_sc: np.ndarray,
    y_tr_sc: np.ndarray,
    groups_tr: np.ndarray,
    X_te_sc: np.ndarray,
    y_te_orig: np.ndarray,
    y_scaler: MinMaxScaler,
    y_range: float,
) -> tuple[dict, object]:
    import tensorflow as tf

    window = X_tr_sc.shape[1]
    n_feat = X_tr_sc.shape[2]
    n_splits = min(3, len(np.unique(groups_tr)))
    cv = GroupKFold(n_splits=n_splits)
    y_pred_cv = np.zeros(len(y_tr_sc))

    print(f"\n[{label}] GroupKFold(k={n_splits}) CV...")
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_tr_sc, y_tr_sc, groups_tr)):
        m = builder(window, n_feat)
        cb = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        )]
        m.fit(
            X_tr_sc[tr_idx], y_tr_sc[tr_idx],
            validation_data=(X_tr_sc[val_idx], y_tr_sc[val_idx]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=0,
        )
        y_pred_cv[val_idx] = m.predict(X_tr_sc[val_idx], verbose=0).flatten()
        print(f"  Fold {fold+1}/{n_splits} 완료")

    cv_rmse = float(np.sqrt(mean_squared_error(y_tr_sc, y_pred_cv)) * y_range)
    cv_mae  = float(mean_absolute_error(y_tr_sc, y_pred_cv) * y_range)

    print(f"  [{label}] 최종 모델 재학습 (전체 train)...")
    final_m = builder(window, n_feat)
    n_val = max(1, int(len(X_tr_sc) * 0.1))
    hist = final_m.fit(
        X_tr_sc[:-n_val], y_tr_sc[:-n_val],
        validation_data=(X_tr_sc[-n_val:], y_tr_sc[-n_val:]),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        )],
        verbose=0,
    )
    actual_epochs = len(hist.history["loss"])

    # OOS 평가 — y_scaler.inverse_transform으로 분 단위 복원 후 RMSE 계산
    if len(X_te_sc):
        oos_sc   = final_m.predict(X_te_sc, verbose=0).flatten()
        oos_orig = y_scaler.inverse_transform(oos_sc.reshape(-1, 1)).flatten()
        oos_rmse = float(np.sqrt(mean_squared_error(y_te_orig, oos_orig)))
        oos_mae  = float(mean_absolute_error(y_te_orig, oos_orig))
    else:
        oos_rmse = oos_mae = cv_rmse

    print(
        f"  [{label}] 에폭={actual_epochs:3d}  "
        f"CV RMSE={cv_rmse:7.1f}분  "
        f"OOS RMSE={oos_rmse:7.1f}분  "
        f"OOS MAE={oos_mae:7.1f}분"
    )

    return {
        "cv_rmse":       round(cv_rmse, 2),
        "cv_mae":        round(cv_mae, 2),
        "oos_rmse":      round(oos_rmse, 2),
        "oos_mae":       round(oos_mae, 2),
        "actual_epochs": actual_epochs,
    }, final_m


# ── 메인 ─────────────────────────────────────────────────────────────────────

def run() -> None:
    print("=" * 65)
    print("GRU v1(기존) vs v2(LayerNorm+BN) 성능 비교")
    print(f"  window={WINDOW_SIZE}, units={UNITS}, dropout={DROPOUT}, batch={BATCH_SIZE}")
    print("=" * 65)

    try:
        import tensorflow as tf
        print(f"[TensorFlow] v{tf.__version__}")
    except ImportError:
        print("[오류] TensorFlow 미설치"); return

    df, features = load_data()
    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()
    print(f"[분리] train={df_train['bearing'].nunique()}개  "
          f"test={df_test['bearing'].nunique()}개")
    print(f"[피처({len(features)}개)] {features}")

    X_tr, y_tr, groups_tr = make_sequences(df_train, features, WINDOW_SIZE)
    X_te, y_te, _         = make_sequences(df_test,  features, WINDOW_SIZE)
    print(f"[시퀀스] train={len(X_tr):,}  test={len(X_te):,}")

    n_feat = X_tr.shape[2]
    seq_scaler = MinMaxScaler()
    X_tr_sc = seq_scaler.fit_transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
    X_te_sc = (seq_scaler.transform(X_te.reshape(-1, n_feat)).reshape(X_te.shape)
               if len(X_te) else X_te)

    y_scaler = MinMaxScaler()
    y_tr_sc  = y_scaler.fit_transform(y_tr.reshape(-1, 1)).flatten()
    y_range  = float(y_tr.max() - y_tr.min()) if len(y_tr) else 1.0
    y_te_orig = (y_scaler.inverse_transform(
                     y_scaler.transform(y_te.reshape(-1, 1))).flatten()
                 if len(y_te) else np.array([]))

    r_v1, mdl_v1 = train_and_evaluate(
        "GRU_v1(기존)",
        build_gru_v1,
        X_tr_sc, y_tr_sc, groups_tr,
        X_te_sc, y_te_orig, y_scaler, y_range,
    )

    r_v2, mdl_v2 = train_and_evaluate(
        "GRU_v2(BN+LN)",
        build_gru_v2,
        X_tr_sc, y_tr_sc, groups_tr,
        X_te_sc, y_te_orig, y_scaler, y_range,
    )

    improvement = (r_v1["oos_rmse"] - r_v2["oos_rmse"]) / max(r_v1["oos_rmse"], 1e-9) * 100
    improved    = r_v2["oos_rmse"] < r_v1["oos_rmse"]
    winner      = "GRU_v2(BN+LN)" if improved else "GRU_v1(기존)"

    print("\n" + "=" * 65)
    print("[최종 비교 결과]")
    print(f"{'버전':<22} {'CV RMSE':>10} {'OOS RMSE':>10} {'OOS MAE':>10} {'에폭':>6}")
    print("-" * 62)
    print(f"{'GRU_v1(기존)':<22} {r_v1['cv_rmse']:>10.1f} {r_v1['oos_rmse']:>10.1f} "
          f"{r_v1['oos_mae']:>10.1f} {r_v1['actual_epochs']:>6}")
    print(f"{'GRU_v2(BN+LN)':<22} {r_v2['cv_rmse']:>10.1f} {r_v2['oos_rmse']:>10.1f} "
          f"{r_v2['oos_mae']:>10.1f} {r_v2['actual_epochs']:>6}")
    print(f"\n[개선률] {improvement:+.1f}%  →  우승: {winner}")

    if improved:
        save_path = MODEL_DIR / "femto_best_dl_GRU_v2.keras"
        try:
            mdl_v2.save(str(save_path))
            print(f"[모델저장] {save_path.name}")
        except Exception as e:
            print(f"[경고] 모델 저장 실패: {e}")

    out = {
        "GRU_v1_baseline": r_v1,
        "GRU_v2_BN_LN":    r_v2,
        "improvement_pct": round(improvement, 2),
        "winner":          winner,
        "improved":        improved,
        "settings": {
            "window": WINDOW_SIZE, "units": UNITS,
            "dropout": DROPOUT, "batch": BATCH_SIZE,
        },
    }
    json_path = MODEL_DIR / "femto_dl_bn_compare_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[저장] {json_path.name}")
    print("=" * 65)


if __name__ == "__main__":
    run()
