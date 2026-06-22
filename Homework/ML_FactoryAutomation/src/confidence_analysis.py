# ════════════════════════════════════════════════════════════════════
# [역할] 예측 신뢰도 분석 — max(predict_proba) 분포로 모델 확신도/오답 상관 확인.
# [단계] 검증 (보고서 5.10)
# [작업 메모] 신뢰도가 낮은 예측에 오답이 몰리는지 확인(휴먼인더루프 근거).
# ════════════════════════════════════════════════════════════════════
"""예측 신뢰도(max predict_proba) 분포 분석.

신뢰도가 1에 가까울수록 확신, 0.5에 가까울수록 불확실하다.
정답/오답별 신뢰도를 비교해, 신뢰도 기반 임계값·휴먼인더루프 운영 근거를 만든다.
"""
from __future__ import annotations
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from src import config, data_loader, preprocess


def run(which: str = "xgb"):
    df = data_loader.load_data()
    X, y = preprocess.build_features(df)
    cols = list(joblib.load(config.MODEL_PATHS["columns"]))
    scaler = joblib.load(config.MODEL_PATHS["scaler"])
    model = joblib.load(config.MODEL_PATHS[which])
    _, X_test, _, y_test = train_test_split(
        X.values, y.values, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y.values)
    proba = model.predict_proba(scaler.transform(X_test))
    conf = proba.max(axis=1)                 # 예측 신뢰도 = 최대 클래스 확률
    pred = proba[:, 1] >= 0.5
    correct = pred == y_test
    stats = {
        "mean": round(float(conf.mean()), 3),
        "median": round(float(np.median(conf)), 3),
        "high_conf_ratio(>=0.9)": round(float((conf >= 0.9).mean()), 3),
        "mean_conf_correct": round(float(conf[correct].mean()), 3),
        "mean_conf_wrong": round(float(conf[~correct].mean()), 3),
    }
    for k, v in stats.items():
        print(f"{k:24s} {v}")
    return conf, correct, stats


if __name__ == "__main__":
    run()
