# ════════════════════════════════════════════════════════════════════
# [역할] FEMTO-ST RUL 예측 — MLflow 하이퍼파라미터 튜닝 (로컬 SQLite)
# [실행] python -m src.femto_mlflow_tune
# [UI]   mlflow ui --backend-store-uri sqlite:///mlflow.db
#         → http://localhost:5000 에서 실험 결과 비교
# [출력] mlflow.db (SQLite), models/femto_best_mlflow.keras
# ════════════════════════════════════════════════════════════════════
"""FEMTO-ST RUL 예측 MLflow 하이퍼파라미터 튜닝.

탐색 공간:
    model_type : GRU | LSTM
    units_1    : 32 | 64
    units_2    : 16 | 32
    dropout    : 0.1 | 0.2 | 0.3
    batch_size : 16 | 32
    epochs     : 30 (EarlyStopping patience=5 로 조기 종료)

SQLAlchemy + SQLite 백엔드로 모든 run 파라미터·메트릭을 저장한다.
최적 run(val_rmse 기준)의 모델을 models/femto_best_mlflow.keras 에 저장한다.
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import mlflow
import mlflow.keras

# ── 경로 & MLflow 설정 ────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODEL_DIR  = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# SQLite 백엔드 — sqlalchemy 드라이버 사용
DB_PATH    = ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{DB_PATH}"
EXPERIMENT   = "FEMTO_RUL_HyperparamSearch"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

# ── 탐색 공간 정의 ────────────────────────────────────────────────────────────
PARAM_GRID = {
    "model_type" : ["GRU", "LSTM"],
    "units_1"    : [32, 64],
    "units_2"    : [16, 32],
    "dropout"    : [0.1, 0.2, 0.3],
    "batch_size" : [16, 32],
    "epochs"     : [30],          # EarlyStopping patience=5 로 실제 에폭 조기 종료
}

WINDOW_SIZE = 30


# ── 데이터 로딩 (femto_dl_rul 파이프라인 재사용) ──────────────────────────────

def load_sequences() -> tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray, object, object]:
    """전처리된 FEMTO 피처에서 스케일된 시퀀스를 반환한다."""
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from src.femto_dl_rul import load_data, make_sequences

    df, features = load_data()
    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()

    X_tr, y_tr, groups_tr = make_sequences(df_train, features, WINDOW_SIZE)
    X_te, y_te, _         = make_sequences(df_test,  features, WINDOW_SIZE)

    n_feat = X_tr.shape[2]
    seq_sc = MinMaxScaler()
    X_tr_sc = seq_sc.fit_transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
    X_te_sc = seq_sc.transform(X_te.reshape(-1, n_feat)).reshape(X_te.shape) if len(X_te) else X_te

    y_sc = MinMaxScaler()
    y_tr_sc = y_sc.fit_transform(y_tr.reshape(-1, 1)).flatten()

    return X_tr_sc, y_tr_sc, groups_tr, X_te_sc, y_te, seq_sc, y_sc


# ── 모델 생성 ─────────────────────────────────────────────────────────────────

def build_model(
    model_type: str,
    window: int,
    n_features: int,
    units_1: int,
    units_2: int,
    dropout: float,
) -> "tf.keras.Model":
    """GRU 또는 LSTM 모델을 생성한다."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, LSTM, Dropout, Dense, Input

    layer_cls = GRU if model_type == "GRU" else LSTM

    model = Sequential([
        Input(shape=(window, n_features)),
        layer_cls(units_1, return_sequences=True),
        Dropout(dropout),
        layer_cls(units_2),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1),
    ], name=f"{model_type}_RUL")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ── 단일 run 실행 ─────────────────────────────────────────────────────────────

def run_one(
    params: dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    y_sc: object,
) -> dict:
    """MLflow run 하나를 실행하고 메트릭 dict를 반환한다."""
    import tensorflow as tf
    from sklearn.model_selection import GroupKFold

    n_feat = X_tr.shape[2]
    window = X_tr.shape[1]

    run_name = (
        f"{params['model_type']}_u{params['units_1']}_{params['units_2']}"
        f"_d{params['dropout']}_b{params['batch_size']}"
    )

    with mlflow.start_run(run_name=run_name):
        # 파라미터 로깅
        for k, v in params.items():
            mlflow.log_param(k, v)

        cv = GroupKFold(n_splits=min(3, len(np.unique(groups))))
        y_pred_cv = np.zeros(len(y_tr))
        fold_val_rmse = []

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_tr, y_tr, groups)):
            model = build_model(
                params["model_type"], window, n_feat,
                params["units_1"], params["units_2"], params["dropout"],
            )
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                ),
            ]
            history = model.fit(
                X_tr[tr_idx], y_tr[tr_idx],
                validation_data=(X_tr[val_idx], y_tr[val_idx]),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                callbacks=callbacks,
                verbose=0,
            )
            y_pred_cv[val_idx] = model.predict(X_tr[val_idx], verbose=0).flatten()

            # 폴드별 val_loss 마지막 값 로깅
            fold_val_loss = history.history["val_loss"][-1]
            mlflow.log_metric(f"fold{fold+1}_val_loss", round(float(fold_val_loss), 6))
            fold_val_rmse.append(float(fold_val_loss))

        # CV RMSE (스케일 역변환)
        y_range = float(y_sc.data_max_[0] - y_sc.data_min_[0])
        cv_rmse = float(np.sqrt(np.mean((y_tr - y_pred_cv) ** 2))) * y_range
        mlflow.log_metric("cv_rmse", round(cv_rmse, 2))

        # OOS (test set) 평가
        oos_rmse, oos_mae = float("nan"), float("nan")
        if len(X_te) and len(y_te):
            # 전체 train으로 최종 모델 재학습
            final_model = build_model(
                params["model_type"], window, n_feat,
                params["units_1"], params["units_2"], params["dropout"],
            )
            n_val = max(1, int(len(X_tr) * 0.1))
            final_model.fit(
                X_tr[:-n_val], y_tr[:-n_val],
                validation_data=(X_tr[-n_val:], y_tr[-n_val:]),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=5, restore_best_weights=True
                    ),
                ],
                verbose=0,
            )
            y_te_pred_sc = final_model.predict(X_te, verbose=0).flatten()
            y_te_orig    = y_sc.inverse_transform(y_te.reshape(-1, 1)).flatten()
            y_te_pred    = np.clip(
                y_sc.inverse_transform(y_te_pred_sc.reshape(-1, 1)).flatten(), 0, None
            )
            oos_rmse = float(np.sqrt(np.mean((y_te_orig - y_te_pred) ** 2)))
            oos_mae  = float(np.mean(np.abs(y_te_orig - y_te_pred)))
            mlflow.log_metric("oos_rmse", round(oos_rmse, 2))
            mlflow.log_metric("oos_mae",  round(oos_mae,  2))

            # 모델을 MLflow artifact로도 저장
            mlflow.keras.log_model(final_model, artifact_path="model")

        print(
            f"  [{run_name}] CV_RMSE={cv_rmse:.1f}  "
            f"OOS_RMSE={oos_rmse:.1f}  OOS_MAE={oos_mae:.1f}"
        )

    return {
        "run_name": run_name,
        "params": params,
        "cv_rmse": cv_rmse,
        "oos_rmse": oos_rmse,
        "oos_mae": oos_mae,
    }


# ── 메인: 그리드 서치 ─────────────────────────────────────────────────────────

def run() -> None:
    print("=" * 65)
    print(f"FEMTO RUL MLflow 하이퍼파라미터 탐색 시작")
    print(f"  추적 URI : {TRACKING_URI}")
    print(f"  실험명   : {EXPERIMENT}")
    print("=" * 65)

    # 데이터 준비
    X_tr, y_tr, groups, X_te, y_te, seq_sc, y_sc = load_sequences()
    print(f"[데이터] train={len(X_tr)} 시퀀스  test={len(X_te)} 시퀀스\n")

    # 파라미터 조합 생성
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    total  = len(combos)
    print(f"[탐색] 총 {total}개 조합\n")

    results = []
    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        print(f"[{i:>3}/{total}] {params}")
        try:
            r = run_one(params, X_tr, y_tr, groups, X_te, y_te, y_sc)
            results.append(r)
        except Exception as e:
            print(f"  → 실패: {e}")

    if not results:
        print("[오류] 완료된 run 없음")
        return

    # 최적 run 선택 (OOS RMSE 기준, NaN은 제외)
    valid = [r for r in results if np.isfinite(r["oos_rmse"])]
    if not valid:
        valid = [r for r in results if np.isfinite(r["cv_rmse"])]
        key = "cv_rmse"
    else:
        key = "oos_rmse"

    best = min(valid, key=lambda r: r[key])

    # 최적 모델 재학습 후 저장
    print(f"\n[최적 run] {best['run_name']}  {key.upper()}={best[key]:.1f}분")
    print("  → 최적 파라미터로 최종 모델 재학습 중...")

    import tensorflow as tf
    p = best["params"]
    n_val = max(1, int(len(X_tr) * 0.1))
    best_model = build_model(
        p["model_type"], X_tr.shape[1], X_tr.shape[2],
        p["units_1"], p["units_2"], p["dropout"],
    )
    best_model.fit(
        X_tr[:-n_val], y_tr[:-n_val],
        validation_data=(X_tr[-n_val:], y_tr[-n_val:]),
        epochs=p["epochs"],
        batch_size=p["batch_size"],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
        ],
        verbose=1,
    )
    out_path = MODEL_DIR / "femto_best_mlflow.keras"
    best_model.save(out_path)

    # 결과 JSON 저장
    summary = {
        "best_run"  : best["run_name"],
        "best_params": best["params"],
        "oos_rmse"  : best["oos_rmse"] if np.isfinite(best["oos_rmse"]) else None,
        "oos_mae"   : best["oos_mae"]  if np.isfinite(best["oos_mae"])  else None,
        "cv_rmse"   : best["cv_rmse"],
        "all_results": [
            {k: (v if not isinstance(v, float) or np.isfinite(v) else None)
             for k, v in r.items() if k != "params"}
            | {"params": r["params"]}
            for r in results
        ],
    }
    summary_path = MODEL_DIR / "femto_mlflow_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[저장] {out_path.name}")
    print(f"[저장] {summary_path.name}")
    print(f"\n[MLflow UI 실행 명령]")
    print(f"  mlflow ui --backend-store-uri {TRACKING_URI}")
    print(f"  → 브라우저 http://localhost:5000 에서 실험 비교 가능")
    print("=" * 65)
    print("MLflow 탐색 완료")


if __name__ == "__main__":
    run()