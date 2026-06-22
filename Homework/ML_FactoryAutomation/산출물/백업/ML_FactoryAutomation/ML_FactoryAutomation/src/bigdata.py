"""데이터 확대 검증 + 이상치 처리.
(1) 외부 대형 데이터(외부 대형 CSV 등, 동일 feature) 로더 — 컬럼 자동 정규화
(2) 이상치 식별(IQR) → 평균(또는 0) 대치
(3) 학습곡선: 학습 데이터량 증가에 따른 성능 변화
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score
from xgboost import XGBClassifier
from src import config, data_loader, preprocess

NUM = config.NUMERIC_FEATURES


def load_external(path: str) -> pd.DataFrame:
    """대형 데이터(예: Playground 외부 대형 CSV train.csv)를 AI4I 스키마로 정규화."""
    df = pd.read_csv(path)
    ren = {"Machine failure": "Target"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    drop = [c for c in ["id", "UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF",
                        "Failure Type"] if c in df.columns]
    df = df.drop(columns=drop)
    keep = ["Type"] + NUM + ["Target"]
    return df[[c for c in keep if c in df.columns]]


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in NUM:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append({"feature": c, "n_outliers": n, "pct": round(100 * n / len(df), 2)})
    return pd.DataFrame(rows)


def handle_outliers(df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """IQR 기준 이상치를 평균(mean) 또는 0(zero)으로 대치."""
    df = df.copy()
    for c in NUM:
        df[c] = df[c].astype(float)
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[c] < lo) | (df[c] > hi)
        df.loc[mask, c] = df[c].mean() if method == "mean" else 0
    return df


def _eval_xgb(Xtr, Xte, ytr, yte):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                      subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
                      eval_metric="logloss", random_state=config.RANDOM_STATE, n_jobs=2)
    m.fit(Xtr, ytr)
    pr = m.predict_proba(Xte)[:, 1]
    pred = (pr >= 0.5).astype(int)
    return {"Accuracy": round(accuracy_score(yte, pred), 4),
            "Recall": round(recall_score(yte, pred, zero_division=0), 4),
            "F1": round(f1_score(yte, pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(yte, pr), 4)}


def outlier_effect(df: pd.DataFrame | None = None):
    df = df if df is not None else data_loader.load_data()
    rep = outlier_report(df)
    out = {}
    for label, d in [("이상치 처리 전", df), ("이상치 처리 후(평균 대치)", handle_outliers(df, "mean"))]:
        X, y = preprocess.build_features(d)
        Xtr, Xte, ytr, yte, sc, cols = preprocess.split_and_scale(X, y)
        out[label] = _eval_xgb(Xtr, Xte, ytr, yte)
    return rep, pd.DataFrame(out).T


def learning_curve(df: pd.DataFrame | None = None,
                   fracs=(0.1, 0.25, 0.5, 0.75, 1.0)):
    df = df if df is not None else data_loader.load_data()
    X, y = preprocess.build_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    Xtr_s, Xte_s = scaler.transform(X_tr), scaler.transform(X_te)
    rows = []
    for fr in fracs:
        n = int(len(X_tr) * fr)
        idx = np.arange(len(X_tr))
        rng = np.random.RandomState(config.RANDOM_STATE)
        sub = rng.choice(idx, n, replace=False)
        m = _eval_xgb(Xtr_s[sub], Xte_s, y_tr.iloc[sub], y_te)
        m["train_n"] = n
        rows.append(m)
    return pd.DataFrame(rows)


def _plot_lc(lc):
    plt.figure(figsize=(6.4, 4))
    plt.plot(lc["train_n"], lc["F1"], "o-", color="#1F4E79", label="F1")
    plt.plot(lc["train_n"], lc["ROC-AUC"], "s-", color="#F59E0B", label="ROC-AUC")
    for _, r in lc.iterrows():
        plt.text(r["train_n"], r["F1"] - 0.04, f"{r['F1']:.3f}", ha="center", fontsize=8)
    plt.xlabel("Training samples")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("Learning Curve (XGBoost, test=2000)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "learning_curve.png", dpi=120)
    plt.close()


def run():
    print("===== 이상치 식별(IQR) =====")
    rep, eff = outlier_effect()
    print(rep.to_string(index=False))
    print("\n===== 이상치 처리 전/후 (XGBoost) =====")
    print(eff.to_string())
    print("\n===== 학습곡선: 데이터량 증가 효과 =====")
    lc = learning_curve()
    print(lc.to_string(index=False))
    _plot_lc(lc)
    rep.to_csv(config.REPORTS_DIR / "outlier_report.csv", index=False)
    eff.to_csv(config.REPORTS_DIR / "outlier_effect.csv")
    lc.to_csv(config.REPORTS_DIR / "learning_curve.csv", index=False)
    return rep, eff, lc


if __name__ == "__main__":
    run()
