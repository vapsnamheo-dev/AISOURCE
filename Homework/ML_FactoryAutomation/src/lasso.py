"""Lasso(L1 정규화) 로지스틱 회귀: 기여 낮은 feature 가중치를 0으로 만드는 특성 선택."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from src import config, data_loader, preprocess


def _m(model, X, y):
    pr = model.predict_proba(X)[:, 1]
    pred = (pr >= 0.5).astype(int)
    return {"Accuracy": round(accuracy_score(y, pred), 4),
            "Recall": round(recall_score(y, pred, zero_division=0), 4),
            "F1": round(f1_score(y, pred, zero_division=0), 4),
            "PR-AUC": round(average_precision_score(y, pr), 4),
            "ROC-AUC": round(roc_auc_score(y, pr), 4)}


def run(Cs=(0.01, 0.05, 0.2, 1.0)):
    df = data_loader.load_data()
    X, y = preprocess.build_features(df)
    Xtr, Xte, ytr, yte, sc, cols = preprocess.split_and_scale(X, y)

    # L2 베이스라인
    l2 = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000,
                            random_state=42).fit(Xtr, ytr)

    # L1 (Lasso) — C(=1/규제강도) 변화에 따른 특성 선택
    coef_rows, perf_rows = {}, {}
    coef_rows["L2 (참고)"] = dict(zip(cols, np.round(l2.coef_[0], 3)))
    for C in Cs:
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                               class_weight="balanced", max_iter=2000, random_state=42).fit(Xtr, ytr)
        coef_rows[f"L1 C={C}"] = dict(zip(cols, np.round(m.coef_[0], 3)))
        nz = int((m.coef_[0] != 0).sum())
        met = _m(m, Xte, yte)
        met["#nonzero"] = nz
        perf_rows[f"Lasso C={C}"] = met
    perf_rows["L2 baseline"] = {**_m(l2, Xte, yte), "#nonzero": int((l2.coef_[0] != 0).sum())}

    coef = pd.DataFrame(coef_rows).T
    perf = pd.DataFrame(perf_rows).T

    print("===== Lasso(L1) 계수: 0이 되는 feature 확인 =====")
    print(coef.to_string())
    print("\n===== 성능 비교 (테스트) =====")
    print(perf.to_string())

    coef.to_csv(config.REPORTS_DIR / "lasso_coef.csv")
    perf.to_csv(config.REPORTS_DIR / "lasso_perf.csv")
    _plot(coef, cols, Cs)
    return coef, perf


def _plot(coef, cols, Cs):
    # C 변화에 따른 비0 계수 개수
    counts = [(coef.loc[f"L1 C={C}"] != 0).sum() for C in Cs]
    plt.figure(figsize=(6.2, 4))
    plt.plot([str(c) for c in Cs], counts, "o-", color="#1F4E79")
    for x, c in zip([str(c) for c in Cs], counts):
        plt.text(x, c + 0.15, str(int(c)), ha="center", fontsize=9)
    plt.xlabel("C  (작을수록 규제 강함)")
    plt.ylabel("# nonzero features")
    plt.ylim(0, len(cols) + 1)
    plt.title("Lasso(L1): Feature Selection by Regularization")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "lasso_selection.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    run()
