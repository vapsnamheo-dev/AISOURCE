# ════════════════════════════════════════════════════════════════════
# [역할] 원본(10k) vs 물리규칙(100k) 분포·성능 비교 + 실데이터 검증.
# [단계] 데이터 확장 (보고서 6.1·6.2)
# [작업 메모] 1번 vs 5번 동일 분포대 확인.
# (이 헤더는 이전/이번 작업 맥락을 요약한 주석입니다 — 기능에는 영향 없음)
# ════════════════════════════════════════════════════════════════════
"""원본(AI4I 10k) vs 동일 물리규칙 생성(100k) 비교 + 실제 데이터 검증.

본 프로젝트는 머신러닝만 사용한다(딥러닝 미사용).
물리 데이터는 src.synth_ai4i 의 공개 물리규칙으로 생성된 ai4i_physics_100k.csv.
"""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from src import config, data_loader, preprocess

PHYS = config.BASE_DIR / "data" / "ai4i_physics_100k.csv"


def _xgb(ytr):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                         colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="logloss",
                         random_state=42, n_jobs=-1)


def _metrics(m, X, y):
    pr = m.predict_proba(X)[:, 1]
    pred = (pr >= 0.5).astype(int)
    return {"Recall": round(recall_score(y, pred, zero_division=0), 4),
            "F1": round(f1_score(y, pred, zero_division=0), 4),
            "PR_AUC": round(average_precision_score(y, pr), 4),
            "ROC_AUC": round(roc_auc_score(y, pr), 4)}


def _split(df):
    X, y = preprocess.build_features(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte


def same_distribution(df, label):
    Xtr, Xte, ytr, yte = _split(df)
    m = _xgb(ytr).fit(Xtr, ytr)
    print(f"[{label}] {len(df):,}행, 고장률 {df['Target'].mean()*100:.2f}% → {_metrics(m, Xte, yte)}")


def real_validation():
    """학습 데이터만 바꾸고 시험은 모두 '실제 원본' 시험셋으로 고정."""
    real = data_loader.load_data()
    Xr, yr = preprocess.build_features(real)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=42, stratify=yr)
    phys = pd.read_csv(PHYS)
    Xp, yp = preprocess.build_features(phys)

    def run(Xtr, ytr, tag):
        sc = StandardScaler().fit(Xtr)
        m = _xgb(ytr).fit(sc.transform(Xtr), ytr)
        print(f"  {tag:28s} {_metrics(m, sc.transform(Xr_te), yr_te)}")

    print("[실제 시험셋(2,000행) 검증]")
    run(Xr_tr, yr_tr, "A) 실제 10k 학습")
    run(Xp, yp, "B) 물리 100k 학습")
    run(pd.concat([Xr_tr, Xp]), pd.concat([yr_tr, yp]), "C) 실제8k + 물리100k")


def run():
    real = data_loader.load_data()
    phys = pd.read_csv(PHYS)
    print("===== 동일 분포 비교 (XGBoost) =====")
    same_distribution(real, "AI4I 원본 10k")
    same_distribution(phys, "물리규칙 100k")
    print("\n===== 현장 검증 =====")
    real_validation()


if __name__ == "__main__":
    run()
