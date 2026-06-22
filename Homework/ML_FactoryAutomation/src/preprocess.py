# ════════════════════════════════════════════════════════════════════
# [역할] 전처리 — X/y 분리, Type 원-핫, 8:2 분할, StandardScaler.
# [단계] [3-4] 전처리 (보고서 2장)
# [작업 메모] 원-핫 후 8 feature 생성·표준화. 원본 CSV 불변(변환본만 생성).
# (이 헤더는 이전/이번 작업 맥락을 요약한 주석입니다 — 기능에는 영향 없음)
# ════════════════════════════════════════════════════════════════════
"""[3] X/y 분리 · [4] 분할 · 스케일링.

인코딩: 범주형(Type)은 sklearn OneHotEncoder(handle_unknown='ignore')로 처리한다.
학습 시 인코더를 models/encoder.joblib 에 저장하고, 추론 시 이를 로드해 사용하므로
학습에 없던 새로운 범주가 들어와도 모든 더미가 0이 되어 에러 없이 동작한다.
출력 컬럼 순서(수치형 5 + Type_H/L/M)는 기존 get_dummies 결과와 동일하다.
"""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import config


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """물리 기반 파생 feature 추가(고장 규칙 대응). 원본 컬럼은 보존."""
    df = df.copy()
    df["Power [W]"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    df["Overstrain [minNm]"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    df["Temp diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    return df


def fit_encoder(df: pd.DataFrame) -> OneHotEncoder:
    """범주형(Type)에 OneHotEncoder(handle_unknown='ignore')를 적합·저장."""
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(df[config.CATEGORICAL_FEATURES])
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(enc, config.MODEL_PATHS["encoder"])
    return enc


def load_encoder() -> OneHotEncoder | None:
    """저장된 인코더 로드(models/encoder.joblib → model/encoder.pkl 순). 없으면 None."""
    p = config.MODEL_PATHS["encoder"]
    if p.exists():
        return joblib.load(p)
    pkl = config.BASE_DIR / "model" / "encoder.pkl"
    if pkl.exists():
        import pickle
        with open(pkl, "rb") as f:
            return pickle.load(f)
    return None


def _ohe_frame(df: pd.DataFrame, enc: OneHotEncoder) -> pd.DataFrame:
    """Type → Type_H/Type_L/Type_M (0/1) DataFrame. 미정의 범주는 handle_unknown='ignore'로 전부 0."""
    arr = enc.transform(df[config.CATEGORICAL_FEATURES]).astype(int)
    cols = list(enc.get_feature_names_out(config.CATEGORICAL_FEATURES))
    return pd.DataFrame(arr, columns=cols, index=df.index)


def build_features(df: pd.DataFrame, encoder: OneHotEncoder | None = None):
    """식별자/누수 컬럼 제거 → Type 원-핫(OneHotEncoder) → X, y 반환.

    encoder=None 이면 저장된 인코더를 로드하고(없으면 새로 적합·저장),
    추론 시에는 학습 때 저장된 인코더가 사용되어 미정의 범주를 안전하게 0 처리한다.
    """
    df = df.copy()
    drop = [c for c in config.DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)
    df = engineer(df)  # 물리 기반 파생 feature 추가
    enc = encoder or load_encoder() or fit_encoder(df)
    ohe = _ohe_frame(df, enc)
    y = df[config.TARGET].astype(int).reset_index(drop=True)
    # 출력 순서: 수치형 5 + 파생 3 + Type_H/L/M
    num = df[config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES].reset_index(drop=True)
    X = pd.concat([num, ohe.reset_index(drop=True)], axis=1)
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
