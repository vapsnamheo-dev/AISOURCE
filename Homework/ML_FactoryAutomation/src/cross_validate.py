# ════════════════════════════════════════════════════════════════════
# [역할] 최종 모델 5-fold 교차검증(cross_val_score) — 단일 분할 결과의 안정성 확인.
# [단계] 검증 (보고서 5.9)
# [작업 메모] 누수 방지: Pipeline(StandardScaler+XGBoost)로 각 fold에서 스케일러를 따로 적합.
# ════════════════════════════════════════════════════════════════════
"""최종 XGBoost(11 feature) 5-fold 교차검증.

cross_val_score를 누수 방지 파이프라인(StandardScaler→XGBoost)으로 실행한다.
RandomizedSearchCV/GridSearchCV가 '튜닝+교차검증'이라면, 본 스크립트는 '고정 파라미터의
순수 교차검증'으로 단일 8:2 분할 결과가 우연이 아님을 확인한다.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from src import config, data_loader, preprocess


def run(n_splits: int = 5):
    df = data_loader.load_data()
    X, y = preprocess.build_features(df)
    Xv, yv = X.values, y.values  # XGBoost는 대괄호 컬럼명을 거부하므로 ndarray 사용
    spw = int((yv == 0).sum()) / max(int((yv == 1).sum()), 1)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
            eval_metric="logloss", random_state=config.RANDOM_STATE, n_jobs=2)),
    ])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    results = {}
    for metric in ["f1", "average_precision", "roc_auc", "recall"]:
        scores = cross_val_score(pipe, Xv, yv, cv=skf, scoring=metric, n_jobs=2)
        results[metric] = (float(scores.mean()), float(scores.std()))
        print(f"{metric:18s} {scores.mean():.3f} ± {scores.std():.3f}  "
              f"(folds {np.round(scores, 3).tolist()})")
    return results


if __name__ == "__main__":
    run()
