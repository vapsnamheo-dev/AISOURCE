# ════════════════════════════════════════════════════════════════════
# [역할] RandomForest 하이퍼파라미터 교차검증(GridSearchCV, scoring=F1).
# [단계] 성능 개선 (보고서 5.6·5.7.3)
# [작업 메모] 작업: 36 후보×cv3=108 fits, CV F1 0.362→0.626.
# (이 헤더는 이전/이번 작업 맥락을 요약한 주석입니다 — 기능에는 영향 없음)
# ════════════════════════════════════════════════════════════════════
"""RandomForest 하이퍼파라미터 교차검증(GridSearchCV).

첨부 예제 방식: param_grid 격자 탐색 + cv=3 교차검증으로 최적 조합 탐색.
불균형 데이터이므로 scoring은 accuracy 대신 f1(소수 클래스 반영)을 사용한다.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split  # noqa: E402

from src import config, data_loader, preprocess  # noqa: E402

PARAM_GRID = {
    "n_estimators": [50, 100, 200],   # 트리 개수
    "max_depth": [4, 6, 8],           # 트리 최대 깊이
    "min_samples_split": [2, 4],      # 노드 분할 최소 샘플 수
    "min_samples_leaf": [1, 2],       # 리프 노드 최소 샘플 수
}


def _metrics(m, Xte, yte) -> dict:
    p = m.predict(Xte)
    pr = m.predict_proba(Xte)[:, 1]
    return {
        "Accuracy": round(accuracy_score(yte, p), 4),
        "Precision": round(precision_score(yte, p, zero_division=0), 4),
        "Recall": round(recall_score(yte, p), 4),
        "F1": round(f1_score(yte, p), 4),
        "ROC_AUC": round(roc_auc_score(yte, pr), 4),
    }


def save_visuals(best_model, cols, Xte, yte):
    """RandomForest 혼동행렬·특성중요도 시각화 저장(첨부 예제 방식)."""
    pred = best_model.predict(Xte)
    # 1) 혼동행렬 (sns.heatmap)
    cm = confusion_matrix(yte, pred)
    plt.figure(figsize=(4.8, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal(0)", "Failure(1)"],
                yticklabels=["Normal(0)", "Failure(1)"])
    plt.title("Confusion Matrix - RandomForest (tuned)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "cm_RandomForest.png", dpi=120)
    plt.close()
    # 2) 특성 중요도 (feature_importances_)
    fi = (pd.DataFrame({"feature": cols, "importance": best_model.feature_importances_})
          .sort_values("importance", ascending=False))
    plt.figure(figsize=(8, 5))
    sns.barplot(x="importance", y="feature", data=fi, color="#1f4e79")
    plt.title("Feature Importance - RandomForest")
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "rf_feature_importance.png", dpi=120)
    plt.close()
    return fi


def save_analysis_visuals(grid, base_fit, Xte, yte, X, y, cols):
    """오분류 개선(혼동행렬 전후)·특성 축소·HP 반복 튜닝 차트 저장."""
    bp = grid.best_params_
    best = grid.best_estimator_
    # 1) 튜닝 전후 혼동행렬 (오분류=비대각선 빨간 박스 강조)
    cm_b = confusion_matrix(yte, base_fit.predict(Xte))
    cm_t = confusion_matrix(yte, best.predict(Xte))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    for a, (cm, ttl) in zip(ax, [(cm_b, f"Default RF (FN={cm_b[1, 0]} missed)"),
                                  (cm_t, f"Tuned RF (FN={cm_t[1, 0]} missed)")]):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Normal(0)", "Failure(1)"],
                    yticklabels=["Normal(0)", "Failure(1)"], ax=a)
        a.set_title(ttl)
        a.set_xlabel("Predicted")
        a.set_ylabel("True")
        for (r, c) in [(0, 1), (1, 0)]:
            a.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="red", lw=2.5))
    fig.suptitle("Confusion Matrix: misclassification reduced by tuning")
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "rf_cm_beforeafter.png", dpi=120)
    plt.close()
    # 2) 특성 중요도 기반 특성 축소
    Xtr_d, Xte_d, ytr_d, yte_d = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    order = pd.Series(best.feature_importances_, index=cols).sort_values(ascending=False).index.tolist()
    ks, f1s, aucs = [], [], []
    for k in range(len(order), 1, -1):
        feats = order[:k]
        m = RandomForestClassifier(random_state=42, class_weight="balanced_subsample",
                                   **bp).fit(Xtr_d[feats], ytr_d)
        pr = m.predict_proba(Xte_d[feats])[:, 1]
        ks.append(k)
        f1s.append(f1_score(yte_d, (pr >= 0.5).astype(int)))
        aucs.append(roc_auc_score(yte_d, pr))
    plt.figure(figsize=(7.4, 4.3))
    plt.plot(ks, f1s, "o-", color="#1F4E79", lw=2.2, label="F1")
    plt.plot(ks, aucs, "s-", color="#F59E0B", lw=2, label="ROC-AUC")
    plt.gca().invert_xaxis()
    plt.xlabel("number of features kept (by importance)")
    plt.ylabel("score")
    plt.title("Feature reduction by importance (RandomForest)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "rf_feature_reduction.png", dpi=120)
    plt.close()
    # 3) 하이퍼파라미터 반복 튜닝 (누적 best)
    scores = grid.cv_results_["mean_test_score"]
    cummax = np.maximum.accumulate(scores)
    x = np.arange(1, len(scores) + 1)
    plt.figure(figsize=(7.6, 4.3))
    plt.plot(x, scores, "o", color="#9CA3AF", ms=4, alpha=0.6, label="each candidate CV F1")
    plt.plot(x, cummax, "-", color="#1F4E79", lw=2.4, label="best so far (cumulative)")
    plt.xlabel("hyperparameter candidate (GridSearchCV iteration)")
    plt.ylabel("CV F1")
    plt.title("HP tuning: CV F1 improves over iterations")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "rf_tuning_iterations.png", dpi=120)
    plt.close()


def run():
    df = data_loader.load_data()
    X, y = preprocess.build_features(df)
    Xtr, Xte, ytr, yte, _sc, cols = preprocess.split_and_scale(X, y)

    base = RandomForestClassifier(random_state=42, class_weight="balanced_subsample")
    grid = GridSearchCV(base, PARAM_GRID, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(Xtr, ytr)

    fi = save_visuals(grid.best_estimator_, cols, Xte, yte)
    base_fit = RandomForestClassifier(
        random_state=42, class_weight="balanced_subsample").fit(Xtr, ytr)
    save_analysis_visuals(grid, base_fit, Xte, yte, X, y, cols)

    return {
        "best_params": grid.best_params_,
        "best_cv_f1": round(grid.best_score_, 4),
        "n_candidates": len(grid.cv_results_["params"]),
        "before": _metrics(base_fit, Xte, yte),
        "after": _metrics(grid.best_estimator_, Xte, yte),
        "importance": fi.to_dict("records"),
    }


def main():
    r = run()
    print("탐색 조합 수:", r["n_candidates"], "| cv=3 → 총", r["n_candidates"] * 3, "회 학습")
    print("Best Params:", r["best_params"])
    print("Best CV F1 :", r["best_cv_f1"])
    print("기본 RF (테스트):", r["before"])
    print("튜닝 RF (테스트):", r["after"])


if __name__ == "__main__":
    main()
