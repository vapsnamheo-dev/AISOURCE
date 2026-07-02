# ════════════════════════════════════════════════════════════════════
# [역할] FEMTO-ST 베어링 열화 분류 — ML 3종 비교 (LogReg/RF/XGB)
# [단계] [2] 모델 학습·평가·저장
# [작업 메모] ML만 적용시 기준 + GroupKFold·VIF·3종 모델 최적 결합 방식 적용.
#   GroupKFold(베어링 단위)로 시계열 누수 방지. VIF 선택 피처 사용.
# ════════════════════════════════════════════════════════════════════
"""FEMTO-ST 베어링 열화 분류 — 머신러닝 파이프라인.

실행:
    python -m src.femto_ml

출력:
    models/femto_rf.pkl
    models/femto_xgb.pkl
    models/femto_best_clf.pkl
    models/femto_scaler.pkl
    models/femto_ml_results.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[경고] xgboost 미설치 → XGBoost 모델 생략")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "FEMTO_processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── 데이터 로딩 ────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, list[str]]:
    """전처리된 FEMTO 피처 파일을 로딩한다.

    Returns
    -------
    (df, features): DataFrame과 선택된 피처 목록
    """
    feat_path = PROCESSED_DIR / "femto_features.csv"
    sel_path = PROCESSED_DIR / "selected_features.csv"

    if not feat_path.exists():
        print("[알림] 전처리 파일 없음 → femto_preprocess 자동 실행")
        from src.femto_preprocess import run as preprocess_run
        preprocess_run()

    df = pd.read_csv(feat_path)

    if sel_path.exists():
        selected = pd.read_csv(sel_path)["feature"].tolist()
    else:
        # VIF 선택 파일 없으면 기본 피처 사용
        selected = [
            "h_rms", "h_kurt", "h_skew", "h_crest",
            "v_rms", "v_kurt", "v_skew", "v_crest",
            "temp_mean",
        ]

    print(f"[로딩] {len(df)}행, 피처: {selected}")
    return df, selected


# ── 모델 정의 ─────────────────────────────────────────────────────────────────

def build_models_and_params() -> list[tuple[str, object, dict]]:
    """(이름, 모델, GridSearch 파라미터) 튜플 리스트를 반환한다."""
    models = [
        (
            "LogisticRegression",
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            {"C": [0.01, 0.1, 1.0, 10.0]},
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                class_weight="balanced_subsample", random_state=42, n_jobs=-1
            ),
            {
                "n_estimators": [100, 300],
                "max_depth": [None, 10],
                "min_samples_leaf": [1, 2],
            },
        ),
    ]

    if HAS_XGB:
        models.append((
            "XGBoost",
            XGBClassifier(
                eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0
            ),
            {
                "n_estimators": [100, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "scale_pos_weight": [1, 3],
            },
        ))

    return models


# ── 학습 및 평가 ──────────────────────────────────────────────────────────────

def train_and_evaluate(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[dict, dict]:
    """GroupKFold + GridSearchCV로 모델을 학습하고 평가 결과를 반환한다.

    train split으로 GroupKFold CV, test split으로 out-of-sample 최종 평가.

    Returns
    -------
    (results, best_models, scaler)
    """
    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()

    print(f"[데이터 분리] train={len(df_train)}행({df_train['bearing'].nunique()}개 베어링)  "
          f"test={len(df_test)}행({df_test['bearing'].nunique()}개 베어링)")

    le = LabelEncoder()
    groups_tr = le.fit_transform(df_train["bearing"])

    def _safe_fill(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
        out = frame[cols].copy()
        for c in cols:
            med = out[c].median()
            out[c] = out[c].fillna(med if np.isfinite(med) else 0.0)
        return out.values.astype(np.float64)

    X_tr = _safe_fill(df_train, features)
    y_tr = df_train["label"].values
    X_te = _safe_fill(df_test, features)
    y_te = df_test["label"].values

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    n_splits = min(3, len(df_train["bearing"].unique()))
    cv = GroupKFold(n_splits=n_splits)
    results = {}
    best_models = {}

    for name, model, param_grid in build_models_and_params():
        print(f"\n[학습] {name} GridSearchCV 중...")
        X_tr_in = X_tr_sc if name == "LogisticRegression" else X_tr
        X_te_in = X_te_sc if name == "LogisticRegression" else X_te

        grid = GridSearchCV(
            model, param_grid,
            cv=cv, scoring="recall", n_jobs=-1, refit=True,
        )
        grid.fit(X_tr_in, y_tr, groups=groups_tr)
        best_model = grid.best_estimator_

        # CV 성능 (train 내부)
        y_cv_proba = np.zeros(len(y_tr))
        y_cv_pred  = np.zeros(len(y_tr), dtype=int)
        best_cls = best_model.__class__
        best_params = grid.best_params_
        for tr_idx, val_idx in cv.split(X_tr_in, y_tr, groups_tr):
            m = best_cls(**best_params)
            m.fit(X_tr_in[tr_idx], y_tr[tr_idx])
            proba = m.predict_proba(X_tr_in[val_idx])[:, 1]
            y_cv_proba[val_idx] = proba
            y_cv_pred[val_idx] = (proba >= 0.5).astype(int)

        # Out-of-sample 성능 (test 베어링)
        oos_proba = best_model.predict_proba(X_te_in)[:, 1]
        oos_pred  = (oos_proba >= 0.5).astype(int)

        def _metrics(y_true, y_pred, y_proba):
            try:
                auc = float(roc_auc_score(y_true, y_proba))
            except Exception:
                auc = float("nan")
            return {
                "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                "roc_auc":   round(auc, 4),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            }

        cv_metrics  = _metrics(y_tr, y_cv_pred, y_cv_proba)
        oos_metrics = _metrics(y_te, oos_pred,  oos_proba)
        oos_metrics["best_params"] = grid.best_params_

        results[name] = {
            "cv":  cv_metrics,
            "oos": oos_metrics,
            # 최종 보고 지표 = OOS (out-of-sample)
            "recall":  oos_metrics["recall"],
            "f1":      oos_metrics["f1"],
            "roc_auc": oos_metrics["roc_auc"],
            "best_params": grid.best_params_,
        }
        best_models[name] = best_model

        print(f"  최적 파라미터: {grid.best_params_}")
        print(f"  CV  → Recall={cv_metrics['recall']:.4f}  F1={cv_metrics['f1']:.4f}  AUC={cv_metrics['roc_auc']:.4f}")
        print(f"  OOS → Recall={oos_metrics['recall']:.4f}  F1={oos_metrics['f1']:.4f}  AUC={oos_metrics['roc_auc']:.4f}")

        if hasattr(best_model, "feature_importances_"):
            imp = best_model.feature_importances_
            results[name]["feature_importance"] = {f: round(float(v), 6) for f, v in zip(features, imp)}

    return results, best_models, scaler


# ── 저장 ─────────────────────────────────────────────────────────────────────

def save_models(
    best_models: dict,
    scaler: StandardScaler,
    results: dict,
) -> None:
    """모델, 스케일러, 결과를 저장한다."""
    # 개별 모델 저장
    model_map = {
        "RandomForest": "femto_rf.pkl",
        "XGBoost": "femto_xgb.pkl",
    }
    for name, fname in model_map.items():
        if name in best_models:
            with open(MODEL_DIR / fname, "wb") as f:
                pickle.dump(best_models[name], f)
            print(f"[저장] {fname}")

    # 최고 모델 선택 (Recall 기준)
    best_name = max(results, key=lambda n: results[n].get("recall", 0))
    with open(MODEL_DIR / "femto_best_clf.pkl", "wb") as f:
        pickle.dump(best_models[best_name], f)
    print(f"[저장] femto_best_clf.pkl → {best_name} (Recall={results[best_name]['recall']:.4f})")

    # 스케일러 저장
    with open(MODEL_DIR / "femto_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("[저장] femto_scaler.pkl")

    # JSON 결과 저장
    results["_best_model"] = best_name
    json_path = MODEL_DIR / "femto_ml_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[저장] femto_ml_results.json")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def run() -> None:
    """ML 파이프라인 전체 실행."""
    print("=" * 60)
    print("FEMTO-ST 베어링 열화 분류 ML 학습 시작")
    print("=" * 60)

    df, features = load_data()

    if len(df) < 30:
        print("[오류] 데이터 부족 (30행 미만) → 전처리 재실행 필요")
        return

    results, best_models, scaler = train_and_evaluate(df, features)
    save_models(best_models, scaler, results)

    print("\n[결과 요약]")
    print(f"{'모델':<20} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 46)
    for name, r in results.items():
        if name.startswith("_"):
            continue
        print(f"{name:<20} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>8.4f}")

    print("=" * 60)
    print("ML 학습 완료")


if __name__ == "__main__":
    run()
