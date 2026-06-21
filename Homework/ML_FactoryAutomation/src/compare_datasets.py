"""신규 대형 데이터(S3E17) 전체 과정 실행 + 기존(AI4I 10k) 대비 비교."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from xgboost import XGBClassifier
from src import config, data_loader, preprocess, bigdata

BIG = config.BASE_DIR / "data" / "train_s3e17.csv"


def _split(df):
    X, y = preprocess.build_features(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte, list(X.columns)


def _models(ytr):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    return {
        "LogReg": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="logloss",
            random_state=42, n_jobs=-1),
    }


def _m(model, X, y):
    pr = model.predict_proba(X)[:, 1]
    pred = (pr >= 0.5).astype(int)
    return dict(Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred, zero_division=0), F1=f1_score(y, pred, zero_division=0),
                PR_AUC=average_precision_score(y, pr), ROC_AUC=roc_auc_score(y, pr))


def compare_models(df):
    Xtr, Xte, ytr, yte, cols = _split(df)
    perf, gap = {}, {}
    for n, m in _models(ytr).items():
        m.fit(Xtr, ytr)
        te, tr = _m(m, Xte, yte), _m(m, Xtr, ytr)
        perf[n] = {k: round(v, 4) for k, v in te.items()}
        gap[n] = {"Train AUC": round(tr["ROC_AUC"], 4), "Test AUC": round(te["ROC_AUC"], 4),
                  "Gap": round(tr["ROC_AUC"] - te["ROC_AUC"], 4), "Test F1": round(te["F1"], 4)}
    return pd.DataFrame(perf).T, pd.DataFrame(gap).T


def degree_exp(df, degrees=(1, 2, 3)):
    X, y = preprocess.build_features(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rows = {}
    for d in degrees:
        pipe = Pipeline([("poly", PolynomialFeatures(degree=d, include_bias=False)),
                         ("sc", StandardScaler()),
                         ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42))])
        pipe.fit(Xtr, ytr)
        pr = pipe.predict_proba(Xte)[:, 1]
        rows[f"degree={d}"] = {"#Feat": pipe.named_steps["poly"].n_output_features_,
                               "F1": round(f1_score(yte, (pr >= .5).astype(int), zero_division=0), 4),
                               "ROC-AUC": round(roc_auc_score(yte, pr), 4)}
    return pd.DataFrame(rows).T


def learning_curve_big(df, fracs=(0.05, 0.1, 0.25, 0.5, 1.0)):
    X, y = preprocess.build_features(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    rng = np.random.RandomState(42)
    rows = []
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    for fr in fracs:
        n = int(len(Xtr) * fr)
        sub = rng.choice(len(Xtr), n, replace=False)
        m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                          colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="logloss",
                          random_state=42, n_jobs=-1).fit(Xtr_s[sub], ytr.iloc[sub])
        pr = m.predict_proba(Xte_s)[:, 1]
        rows.append({"train_n": n, "F1": round(f1_score(yte, (pr >= .5).astype(int), zero_division=0), 4),
                     "ROC-AUC": round(roc_auc_score(yte, pr), 4)})
    return pd.DataFrame(rows)


def run():
    small = data_loader.load_data()
    big = bigdata.load_external(BIG)
    print(f"기존 AI4I: {small.shape[0]:,}행, 고장률 {small['Target'].mean()*100:.2f}%")
    print(f"신규 S3E17: {big.shape[0]:,}행, 고장률 {big['Target'].mean()*100:.2f}%\n")

    print("===== 3개 모델 비교 — 기존(10k) =====")
    p_s, g_s = compare_models(small)
    print(p_s.to_string())
    print("\n===== 3개 모델 비교 — 신규(136k) =====")
    p_b, g_b = compare_models(big)
    print(p_b.to_string())

    print("\n===== 과적합 Gap — 기존(10k) =====")
    print(g_s.to_string())
    print("\n===== 과적합 Gap — 신규(136k) =====")
    print(g_b.to_string())

    print("\n===== 다항 차수 — 신규(136k) =====")
    print(degree_exp(big).to_string())

    print("\n===== 학습곡선 — 신규(136k) =====")
    lc = learning_curve_big(big)
    print(lc.to_string(index=False))

    # save artifacts
    for name, obj in [("cmp_small", p_s), ("cmp_big", p_b), ("gap_small", g_s),
                      ("gap_big", g_b), ("lc_big", lc)]:
        obj.to_csv(config.REPORTS_DIR / f"{name}.csv")
    _plot_xgb_compare(p_s, p_b)
    _plot_lc_big(lc)
    return p_s, p_b, g_s, g_b, lc


def _plot_xgb_compare(ps, pb):
    keys = ["Recall", "F1", "PR_AUC", "ROC_AUC"]
    s = [ps.loc["XGBoost", k] for k in keys]
    b = [pb.loc["XGBoost", k] for k in keys]
    x = np.arange(len(keys))
    plt.figure(figsize=(6.4, 4))
    plt.bar(x - 0.2, s, 0.4, label="기존 10k", color="#9CA3AF")
    plt.bar(x + 0.2, b, 0.4, label="신규 136k", color="#1F4E79")
    for i, (a, c) in enumerate(zip(s, b)):
        plt.text(i - 0.2, a + 0.02, f"{a:.3f}", ha="center", fontsize=8)
        plt.text(i + 0.2, c + 0.02, f"{c:.3f}", ha="center", fontsize=8)
    plt.xticks(x, ["Recall", "F1", "PR-AUC", "ROC-AUC"])
    plt.ylim(0, 1.05)
    plt.title("XGBoost: 10k vs 136k (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "data_size_compare.png", dpi=120)
    plt.close()


def _plot_lc_big(lc):
    plt.figure(figsize=(6.4, 4))
    plt.plot(lc["train_n"], lc["ROC-AUC"], "s-", color="#F59E0B", label="ROC-AUC")
    plt.plot(lc["train_n"], lc["F1"], "o-", color="#1F4E79", label="F1")
    plt.xlabel("Training samples")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("Learning Curve on 136k (XGBoost)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "lc_big.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    run()
