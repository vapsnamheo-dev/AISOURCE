"""과적합 진단 및 테스트 성능 개선.
(1) 훈련 vs 테스트 비교(Gap) — 3개 모델
(2) 정규화로 과적합 완화 → 테스트 개선
(3) 다항 차수 train/test gap — 특성 수 증가의 과적합 효과
(4) 노이즈 특성 주입 — 'feature 많으면 노이즈'를 직접 실증
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from src import config, data_loader, preprocess


def _scores(model, X, y):
    pr = model.predict_proba(X)[:, 1]
    pred = (pr >= 0.5).astype(int)
    return {"Acc": round(accuracy_score(y, pred), 4),
            "F1": round(f1_score(y, pred, zero_division=0), 4),
            "AUC": round(roc_auc_score(y, pr), 4)}


def _data():
    df = data_loader.load_data()
    X, y = preprocess.build_features(df)
    return preprocess.split_and_scale(X, y)


def _spw(y):
    return (y == 0).sum() / max((y == 1).sum(), 1)


# (1) 훈련 vs 테스트 Gap
def train_test_gap():
    Xtr, Xte, ytr, yte, sc, cols = _data()
    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "RandomForest(max_depth=None)": RandomForestClassifier(n_estimators=300, max_depth=None,
            min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=2),
        "XGBoost(depth=4)": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=_spw(ytr),
            eval_metric="logloss", random_state=42, n_jobs=2),
    }
    rows = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        tr, te = _scores(m, Xtr, ytr), _scores(m, Xte, yte)
        rows[name] = {"Train AUC": tr["AUC"], "Test AUC": te["AUC"],
                      "AUC Gap": round(tr["AUC"] - te["AUC"], 4),
                      "Train F1": tr["F1"], "Test F1": te["F1"]}
    return pd.DataFrame(rows).T


# (2) 정규화로 과적합 완화
def regularize():
    Xtr, Xte, ytr, yte, sc, cols = _data()
    spw = _spw(ytr)
    pairs = {
        "RF 과적합(depth=None)": RandomForestClassifier(n_estimators=300, max_depth=None,
            min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=2),
        "RF 정규화(depth=8,leaf=10)": RandomForestClassifier(n_estimators=300, max_depth=8,
            min_samples_leaf=10, max_features="sqrt", class_weight="balanced_subsample",
            random_state=42, n_jobs=2),
        "XGB 기본": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
            eval_metric="logloss", random_state=42, n_jobs=2),
        "XGB 정규화(L1/L2)": XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=3.0, gamma=1.0,
            min_child_weight=5, scale_pos_weight=spw, eval_metric="logloss",
            random_state=42, n_jobs=2),
    }
    rows = {}
    for name, m in pairs.items():
        m.fit(Xtr, ytr)
        tr, te = _scores(m, Xtr, ytr), _scores(m, Xte, yte)
        rows[name] = {"Train AUC": tr["AUC"], "Test AUC": te["AUC"],
                      "AUC Gap": round(tr["AUC"] - te["AUC"], 4), "Test F1": te["F1"]}
    return pd.DataFrame(rows).T


# (3) 다항 차수 train/test
def degree_train_test(degrees=(1, 2, 3)):
    df = data_loader.load_data()
    X, y = preprocess.build_features(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rows = {}
    for d in degrees:
        pipe = Pipeline([("poly", PolynomialFeatures(degree=d, include_bias=False)),
                         ("sc", StandardScaler()),
                         ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42))])
        pipe.fit(Xtr, ytr)
        nf = pipe.named_steps["poly"].n_output_features_
        tr, te = _scores(pipe, Xtr, ytr), _scores(pipe, Xte, yte)
        rows[f"degree={d}"] = {"#Feat": nf, "Train AUC": tr["AUC"], "Test AUC": te["AUC"],
                               "AUC Gap": round(tr["AUC"] - te["AUC"], 4)}
    return pd.DataFrame(rows).T


# (4) 노이즈 특성 주입
def noise_experiment(ks=(0, 10, 30, 60)):
    df = data_loader.load_data()
    X, y = preprocess.build_features(df)
    Xtr0, Xte0, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rng = np.random.RandomState(42)
    rows = []
    for k in ks:
        Xtr = Xtr0.copy()
        Xte = Xte0.copy()
        for j in range(k):
            Xtr[f"noise_{j}"] = rng.normal(size=len(Xtr))
            Xte[f"noise_{j}"] = rng.normal(size=len(Xte))
        sc = StandardScaler().fit(Xtr)
        Xtrs, Xtes = sc.transform(Xtr), sc.transform(Xte)
        m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                          colsample_bytree=0.9, scale_pos_weight=_spw(ytr),
                          eval_metric="logloss", random_state=42, n_jobs=2).fit(Xtrs, ytr)
        te = _scores(m, Xtes, yte)
        rows.append({"노이즈 특성 수": k, "Test AUC": te["AUC"], "Test F1": te["F1"]})
    return pd.DataFrame(rows)


def _plot_gap(df):
    names = ["LogReg", "RF(None)", "XGB(4)"]
    tr = df["Train AUC"].values
    te = df["Test AUC"].values
    x = np.arange(3)
    plt.figure(figsize=(6.2, 4))
    plt.bar(x - 0.2, tr, 0.4, label="Train AUC", color="#9CA3AF")
    plt.bar(x + 0.2, te, 0.4, label="Test AUC", color="#1F4E79")
    for i, (a, b) in enumerate(zip(tr, te)):
        plt.text(i - 0.2, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
        plt.text(i + 0.2, b + 0.005, f"{b:.3f}", ha="center", fontsize=8)
    plt.xticks(x, names)
    plt.ylim(0.85, 1.02)
    plt.title("Train vs Test AUC (Overfitting Check)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "overfit_gap.png", dpi=120)
    plt.close()


def _plot_noise(df):
    plt.figure(figsize=(6.2, 4))
    plt.plot(df["노이즈 특성 수"], df["Test AUC"], "o-", color="#1F4E79", label="Test AUC")
    plt.plot(df["노이즈 특성 수"], df["Test F1"], "s-", color="#F59E0B", label="Test F1")
    for _, r in df.iterrows():
        plt.text(r["노이즈 특성 수"], r["Test F1"] - 0.03, f"{r['Test F1']:.3f}", ha="center", fontsize=8)
    plt.xlabel("# Random noise features added")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("Effect of Noise Features on Test Performance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "noise_effect.png", dpi=120)
    plt.close()


def run():
    print("===== (1) 훈련 vs 테스트 (과적합 진단) =====")
    g = train_test_gap()
    print(g.to_string())
    _plot_gap(g)
    print("\n===== (2) 정규화로 과적합 완화 → 테스트 개선 =====")
    r = regularize()
    print(r.to_string())
    print("\n===== (3) 다항 차수 train/test (특성 수↑ 과적합) =====")
    d = degree_train_test()
    print(d.to_string())
    print("\n===== (4) 노이즈 특성 주입 (feature 많으면 노이즈) =====")
    n = noise_experiment()
    print(n.to_string(index=False))
    _plot_noise(n)
    for name, obj in [("overfit_gap", g), ("regularize", r), ("degree_traintest", d), ("noise_effect", n)]:
        obj.to_csv(config.REPORTS_DIR / f"{name}.csv")
    return g, r, d, n


if __name__ == "__main__":
    run()
