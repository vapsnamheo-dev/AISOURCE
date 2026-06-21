# ════════════════════════════════════════════════════════════════════
# [역할] 분류기 8종 전수 비교(머신러닝 전용 — MLP/딥러닝 제외, NB는 GaussianNB).
# [단계] 모델 비교 (보고서 4.6)
# [작업 메모] 부스팅·앙상블 1~3위 확인.
# (이 헤더는 이전/이번 작업 맥락을 요약한 주석입니다 — 기능에는 영향 없음)
# ════════════════════════════════════════════════════════════════════
"""분류 모델 종류별 전수 비교 (머신러닝 전용).

이미지의 분류 모델 중 MLPClassifier(신경망=딥러닝)는 '머신러닝만 사용' 원칙에 따라 제외.
Naive Bayes는 MultinomialNB(음수 불가·빈도형) 대신 연속형에 맞는 GaussianNB 사용.
"""
from __future__ import annotations
import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef)
from xgboost import XGBClassifier
from src import config, data_loader, preprocess

warnings.filterwarnings("ignore")


def build_models(spw: float) -> dict:
    return {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "SVC": SVC(class_weight="balanced", probability=True, random_state=42),
        "GaussianNB": GaussianNB(),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="logloss",
            random_state=42, n_jobs=-1),
    }


def run() -> pd.DataFrame:
    df = data_loader.load_data()
    X, y = preprocess.build_features(df)
    Xtr, Xte, ytr, yte, _sc, _cols = preprocess.split_and_scale(X, y)
    spw = (ytr == 0).sum() / (ytr == 1).sum()

    rows = {}
    for name, m in build_models(spw).items():
        m.fit(Xtr, ytr)
        pr = m.predict_proba(Xte)[:, 1]
        pred = (pr >= 0.5).astype(int)
        rows[name] = {
            "Accuracy": round(accuracy_score(yte, pred), 4),
            "Precision": round(precision_score(yte, pred, zero_division=0), 4),
            "Recall": round(recall_score(yte, pred, zero_division=0), 4),
            "F1": round(f1_score(yte, pred, zero_division=0), 4),
            "MCC": round(matthews_corrcoef(yte, pred), 4),
            "PR_AUC": round(average_precision_score(yte, pr), 4),
            "ROC_AUC": round(roc_auc_score(yte, pr), 4),
        }
    res = pd.DataFrame(rows).T.sort_values("ROC_AUC", ascending=False)
    res.to_csv(config.REPORTS_DIR / "classifier_compare.csv")
    print(res.to_string())
    print("\n[제외] MLPClassifier = 신경망(딥러닝) → 머신러닝 전용 원칙상 제외")
    return res


if __name__ == "__main__":
    run()
