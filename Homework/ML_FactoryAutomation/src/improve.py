# ════════════════════════════════════════════════════════════════════
# [역할] 성능 개선 실험 — 다항 차수(degree) 비교 + XGBoost 하이퍼파라미터 튜닝(전/후).
# [단계] 성능 개선 (보고서 5.1·5.2)
# [작업 메모] 작업: RandomizedSearchCV 25조합×cv3=75 fits 반복 튜닝.
# (이 헤더는 이전/이번 작업 맥락을 요약한 주석입니다 — 기능에는 영향 없음)
# ════════════════════════════════════════════════════════════════════
"""성능 개선 실험: (1) 다항 차수(degree) 비교  (2) 하이퍼파라미터 튜닝 전/후."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBClassifier
from src import config, data_loader, preprocess


def _metrics(y, pred, proba):
    return {
        "Accuracy": round(accuracy_score(y, pred), 4),
        "Precision": round(precision_score(y, pred, zero_division=0), 4),
        "Recall": round(recall_score(y, pred, zero_division=0), 4),
        "F1": round(f1_score(y, pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y, proba), 4),
    }


def _split():
    df = data_loader.load_data()
    X, y = preprocess.build_features(df)
    return train_test_split(X, y, test_size=config.TEST_SIZE,
                            random_state=config.RANDOM_STATE, stratify=y)


# ---------- (1) 다항 차수 비교 ----------
def degree_experiment(degrees=(1, 2, 3)):
    Xtr, Xte, ytr, yte = _split()
    rows = {}
    for d in degrees:
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=d, include_bias=False)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=3000,
                                       random_state=config.RANDOM_STATE)),
        ])
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        n_feat = pipe.named_steps["poly"].n_output_features_
        m = _metrics(yte, pred, proba)
        m["#Features"] = n_feat
        rows[f"degree={d}"] = m
    return pd.DataFrame(rows).T


# ---------- (2) XGBoost 튜닝 전/후 ----------
def tune_xgboost(n_iter=25):
    Xtr, Xte, ytr, yte, scaler, cols = preprocess.split_and_scale(*_get_xy())
    neg, pos = int((ytr == 0).sum()), int((ytr == 1).sum())
    spw = neg / pos

    base = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                         subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
                         eval_metric="logloss", random_state=config.RANDOM_STATE, n_jobs=2)
    base.fit(Xtr, ytr)
    pb = base.predict_proba(Xte)[:, 1]
    m_before = _metrics(yte, (pb >= 0.5).astype(int), pb)

    space = {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.5, 1, 2],
    }
    est = XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                        random_state=config.RANDOM_STATE, n_jobs=2)
    search = RandomizedSearchCV(est, space, n_iter=n_iter, scoring="f1", cv=3,
                                random_state=config.RANDOM_STATE, n_jobs=2, verbose=0)
    search.fit(Xtr, ytr)
    best = search.best_estimator_
    pa = best.predict_proba(Xte)[:, 1]
    m_after = _metrics(yte, (pa >= 0.5).astype(int), pa)
    return m_before, m_after, search.best_params_


def _get_xy():
    df = data_loader.load_data()
    return preprocess.build_features(df)


def _bar_degree(dfm):
    deg = list(dfm.index)
    f1 = dfm["F1"].values
    auc = dfm["ROC-AUC"].values
    x = np.arange(len(deg))
    plt.figure(figsize=(6.2, 4))
    plt.bar(x - 0.2, f1, 0.4, label="F1", color="#1F4E79")
    plt.bar(x + 0.2, auc, 0.4, label="ROC-AUC", color="#F59E0B")
    plt.xticks(x, deg)
    plt.ylim(0, 1.0)
    for i, (a, b) in enumerate(zip(f1, auc)):
        plt.text(i - 0.2, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
        plt.text(i + 0.2, b + 0.02, f"{b:.3f}", ha="center", fontsize=9)
    plt.title("LogReg Performance by Polynomial Degree")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "degree_comparison.png", dpi=120)
    plt.close()


def _bar_tuning(before, after):
    keys = ["Precision", "Recall", "F1", "ROC-AUC"]
    b = [before[k] for k in keys]
    a = [after[k] for k in keys]
    x = np.arange(len(keys))
    plt.figure(figsize=(6.2, 4))
    plt.bar(x - 0.2, b, 0.4, label="튜닝 전", color="#9CA3AF")
    plt.bar(x + 0.2, a, 0.4, label="튜닝 후", color="#1F4E79")
    plt.xticks(x, keys)
    plt.ylim(0, 1.0)
    for i, (bb, aa) in enumerate(zip(b, a)):
        plt.text(i - 0.2, bb + 0.02, f"{bb:.3f}", ha="center", fontsize=9)
        plt.text(i + 0.2, aa + 0.02, f"{aa:.3f}", ha="center", fontsize=9)
    plt.title("XGBoost: Before vs After Tuning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "tuning_comparison.png", dpi=120)
    plt.close()


def run():
    print("===== (1) 다항 차수(degree) 비교 — LogisticRegression =====")
    dfm = degree_experiment()
    print(dfm.to_string())
    _bar_degree(dfm)
    dfm.to_csv(config.REPORTS_DIR / "degree_comparison.csv")

    print("\n===== (2) XGBoost 하이퍼파라미터 튜닝 전/후 =====")
    before, after, best = tune_xgboost()
    cmp = pd.DataFrame({"튜닝 전": before, "튜닝 후": after}).T
    print(cmp.to_string())
    print("\nbest_params:", best)
    _bar_tuning(before, after)
    cmp.to_csv(config.REPORTS_DIR / "tuning_comparison.csv")
    import json
    (config.REPORTS_DIR / "best_params.json").write_text(json.dumps(best, indent=2))
    return dfm, cmp, best


if __name__ == "__main__":
    run()
