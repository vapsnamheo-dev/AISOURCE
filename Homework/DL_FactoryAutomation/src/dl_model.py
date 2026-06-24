"""
dl_model.py — 1D-CNN / LSTM 모델 정의
ML 프로젝트(XGBoost, tabular)의 DL 후속: 시계열 센서 데이터로 고장 예측

구조:
  CNN1D  : 로컬 패턴(단기 이상신호) 추출 → 글로벌 컨텍스트 풀링 → Dense
  LSTM   : 장기 의존성(누적 열화 추세) 학습 → 최종 상태 → Dense
  CNN_LSTM: Conv 피처 추출 → LSTM 시퀀스 모델링 (권장)
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import SEQ_LEN, N_FEATURES


def build_cnn1d(n_classes: int = 1, dropout: float = 0.3):
    """
    1D-CNN 고장 분류기.
    n_classes=1  → 이진분류 (sigmoid)
    n_classes>1  → 다중분류 (softmax)
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("tensorflow 설치 필요: pip install tensorflow")

    inp = tf.keras.Input(shape=(SEQ_LEN, N_FEATURES), name="sensor_seq")

    # Block 1: 단기 패턴 (kernel=3)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Block 2: 중기 패턴 (kernel=5)
    x = tf.keras.layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Block 3: 장기 트렌드 (kernel=7)
    x = tf.keras.layers.Conv1D(64, kernel_size=7, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    if n_classes == 1:
        out = tf.keras.layers.Dense(1, activation="sigmoid", name="failure_prob")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="recall")]
    else:
        out = tf.keras.layers.Dense(n_classes, activation="softmax", name="failure_type")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model = tf.keras.Model(inp, out, name="CNN1D_Failure")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss, metrics=metrics,
    )
    return model


def build_lstm(n_classes: int = 1, dropout: float = 0.3):
    """LSTM 고장 분류기 — 장기 의존성 패턴 포착."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("tensorflow 설치 필요: pip install tensorflow")

    inp = tf.keras.Input(shape=(SEQ_LEN, N_FEATURES), name="sensor_seq")

    x = tf.keras.layers.LSTM(128, return_sequences=True)(inp)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    if n_classes == 1:
        out = tf.keras.layers.Dense(1, activation="sigmoid", name="failure_prob")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="recall")]
    else:
        out = tf.keras.layers.Dense(n_classes, activation="softmax", name="failure_type")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model = tf.keras.Model(inp, out, name="LSTM_Failure")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss, metrics=metrics,
    )
    return model


def build_cnn_lstm(n_classes: int = 1, dropout: float = 0.3):
    """
    CNN + LSTM 하이브리드 (권장 모델).
    CNN으로 로컬 피처 추출 후 LSTM으로 시퀀스 모델링.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("tensorflow 설치 필요: pip install tensorflow")

    inp = tf.keras.Input(shape=(SEQ_LEN, N_FEATURES), name="sensor_seq")

    # CNN 피처 추출
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    # LSTM 시퀀스 학습
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    if n_classes == 1:
        out = tf.keras.layers.Dense(1, activation="sigmoid", name="failure_prob")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="recall")]
    else:
        out = tf.keras.layers.Dense(n_classes, activation="softmax", name="failure_type")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model = tf.keras.Model(inp, out, name="CNNLSTM_Failure")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss, metrics=metrics,
    )
    return model


MODEL_REGISTRY = {
    "cnn1d": build_cnn1d,
    "lstm": build_lstm,
    "cnn_lstm": build_cnn_lstm,
}
