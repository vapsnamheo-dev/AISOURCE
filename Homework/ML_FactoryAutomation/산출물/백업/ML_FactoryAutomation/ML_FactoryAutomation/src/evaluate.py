"""[6] 평가 · [7] 특성 중요도 · [8] 불필요 특성 제거 후 재평가 · [9] 모델 비교 · [10] 시각화."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
from src import config, train as train_mod


def _metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return _metrics(y_test, pred, proba), pred, proba


def feature_importance_logreg(model, cols) -> pd.DataFrame:
    imp = np.abs(model.coef_[0])
    return (pd.DataFrame({"feature": cols, "importance": imp})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def drop_low_importance_and_refit(X_train, X_test, y_train, y_test, cols, drop_n=2):
    """[8] 중요도 낮은 특성 제거 후 LogReg 재학습/재평가."""
    base = LogisticRegression(class_weight="balanced", max_iter=1000,
                              random_state=config.RANDOM_STATE).fit(X_train, y_train)
    fi = feature_importance_logreg(base, cols)
    drop_feats = fi.tail(drop_n)["feature"].tolist()
    keep_idx = [i for i, c in enumerate(cols) if c not in drop_feats]
    Xtr, Xte = X_train[:, keep_idx], X_test[:, keep_idx]
    reduced = LogisticRegression(class_weight="balanced", max_iter=1000,
                                 random_state=config.RANDOM_STATE).fit(Xtr, y_train)
    m_red, _, _ = evaluate_model(reduced, Xte, y_test)
    return drop_feats, m_red


def _save_confusion(y_test, pred, name):
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Failure"], yticklabels=["Normal", "Failure"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / f"cm_{name}.png", dpi=120)
    plt.close()


def _save_roc(curves):
    plt.figure(figsize=(6, 5))
    for name, (y_test, proba) in curves.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "roc_curve.png", dpi=120)
    plt.close()


def _save_importance(fi: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.barplot(x="importance", y="feature", data=fi, color="#1f4e79")
    plt.title("Feature Importance (LogReg |coef|)")
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "feature_importance.png", dpi=120)
    plt.close()


def run():
    out = train_mod.train(save=True)
    logreg, rf, xgb, cols = out["logreg"], out["rf"], out["xgb"], out["columns"]
    X_train, X_test, y_train, y_test = out["splits"]

    m_log, pred_log, proba_log = evaluate_model(logreg, X_test, y_test)
    m_rf, pred_rf, proba_rf = evaluate_model(rf, X_test, y_test)
    m_xgb, pred_xgb, proba_xgb = evaluate_model(xgb, X_test, y_test)

    comparison = pd.DataFrame({
        "LogisticRegression": m_log,
        "RandomForest": m_rf,
        "XGBoost": m_xgb,
    }).T
    fi = feature_importance_logreg(logreg, cols)
    drop_feats, m_reduced = drop_low_importance_and_refit(
        X_train, X_test, y_train, y_test, cols)

    # 시각화 저장
    _save_confusion(y_test, pred_log, "LogReg")
    _save_confusion(y_test, pred_rf, "RandomForest")
    _save_confusion(y_test, pred_xgb, "XGBoost")
    _save_roc({"LogReg": (y_test, proba_log), "RandomForest": (y_test, proba_rf), "XGBoost": (y_test, proba_xgb)})
    _save_importance(fi)
    comparison.to_csv(config.REPORTS_DIR / "model_comparison.csv")

    print("\n=== [6][9] 모델 비교 ===")
    print(comparison.to_string())
    print("\n=== [7] 특성 중요도 (LogReg) ===")
    print(fi.to_string(index=False))
    print(f"\n=== [8] 중요도 낮은 특성 {drop_feats} 제거 후 재평가 ===")
    print(m_reduced)
    print("\n리포트 저장:", [p.name for p in config.REPORTS_DIR.glob('*')])
    return comparison, fi, m_reduced


if __name__ == "__main__":
    run()
