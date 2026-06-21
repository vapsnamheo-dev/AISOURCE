# ════════════════════════════════════════════════════════════════════
# [역할] 전처리 — X/y 분리, Type 원-핫, 8:2 분할, StandardScaler.
# [단계] [3-4] 전처리 (보고서 2장)
# [작업 메모] 원-핫 후 8 feature 생성·표준화. 원본 CSV 불변(변환본만 생성).
# (이 헤더는 이전/이번 작업 맥락을 요약한 주석입니다 — 기능에는 영향 없음)
# ════════════════════════════════════════════════════════════════════
"""[3] X/y 분리 · [4] 분할 · 스케일링."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src import config


def build_features(df: pd.DataFrame):
    """식별자/누수 컬럼 제거 → Type 원-핫 인코딩 → X, y 반환."""
    df = df.copy()
    drop = [c for c in config.DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)
    df = pd.get_dummies(df, columns=config.CATEGORICAL_FEATURES, drop_first=False)
    y = df[config.TARGET].astype(int)
    X = df.drop(columns=[config.TARGET])
    # bool(dummies) -> int
    X = X.astype({c: "int" for c in X.columns if X[c].dtype == "bool"})
    return X, y


def split_and_scale(X, y, scaler: StandardScaler | None = None):
    """stratify 분할(불균형 대응) 후 StandardScaler 적용."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    if scaler is None:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
    else:
        X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler, list(X.columns)
