"""predict.py — 추론 (단건 시퀀스 또는 CSV 파일 배치)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import FEATURE_COLS, SEQ_LEN, N_FEATURES, PRED_THRESHOLD, MODEL_DIR


def load_model_and_norm(model_name: str = "cnn_lstm", suffix: str = "binary"):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("tensorflow 설치 필요")

    model_path = MODEL_DIR / f"{model_name}_{suffix}.keras"
    norm_path  = MODEL_DIR / f"{model_name}_{suffix}_norm.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {model_path} — 먼저 train.py 실행")

    model = tf.keras.models.load_model(str(model_path))
    norm = np.load(str(norm_path))
    return model, norm["mean"], norm["std"]


def predict_sequence(seq_df: pd.DataFrame, model=None, mean=None, std=None,
                     model_name: str = "cnn_lstm") -> dict:
    """
    단건 시퀀스 예측.
    seq_df: DataFrame (SEQ_LEN rows × FEATURE_COLS)
    returns: {prob, label, risk_level}
    """
    if model is None:
        model, mean, std = load_model_and_norm(model_name)

    X = seq_df[FEATURE_COLS].values.astype(np.float32)
    if X.shape[0] != SEQ_LEN:
        raise ValueError(f"시퀀스 길이 {X.shape[0]} ≠ {SEQ_LEN}")

    X_norm = (X - mean) / (std + 1e-8)
    X_input = X_norm[np.newaxis]  # (1, SEQ_LEN, N_FEATURES)

    prob = float(model.predict(X_input, verbose=0)[0][0])
    label = int(prob >= PRED_THRESHOLD)
    risk = _risk_level(prob)

    return {"prob": round(prob * 100, 1), "label": label, "risk_level": risk}


def _risk_level(prob: float) -> str:
    if prob < 0.3:
        return "정상"
    if prob < 0.55:
        return "주의"
    if prob < 0.75:
        return "위험"
    return "긴급"
