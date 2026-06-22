"""[5] 모델 학습: 베이스라인 LogisticRegression + 비교 모델 XGBoost."""
from __future__ import annotations
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src import config, data_loader, preprocess


def build_models(y_train):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)
    logreg = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=config.RANDOM_STATE, n_jobs=2,
    )
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=config.RANDOM_STATE, n_jobs=2,
    )
    return logreg, rf, xgb


def train(save: bool = True):
    df = data_loader.load_data()
    X, y = preprocess.build_features(df)
    X_train, X_test, y_train, y_test, scaler, cols = preprocess.split_and_scale(X, y)
    logreg, rf, xgb = build_models(y_train)
    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    if save:
        joblib.dump(logreg, config.MODEL_PATHS["logreg"])
        joblib.dump(rf, config.MODEL_PATHS["rf"])
        joblib.dump(xgb, config.MODEL_PATHS["xgb"])
        joblib.dump(scaler, config.MODEL_PATHS["scaler"])
        joblib.dump(cols, config.MODEL_PATHS["columns"])
    return {
        "logreg": logreg, "rf": rf, "xgb": xgb, "scaler": scaler, "columns": cols,
        "splits": (X_train, X_test, y_train, y_test),
    }


if __name__ == "__main__":
    out = train()
    print("학습 완료. 저장 모델:", [p.name for p in config.MODEL_PATHS.values()])
    print("피처:", out["columns"])
