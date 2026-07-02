"""시계열 데이터 로더 — NPZ 통합 파일 + CV 지원."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from typing import Iterator
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    FEATURE_COLS, TARGET_COL, FAILURE_TYPE_COL,
    DATA_DIR, DEMO_DIR, SEQ_LEN, CV_FOLDS,
)


def _load_npz(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """train_data.npz / test_data.npz 로드."""
    path = DATA_DIR / f"{split}_data.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — python src/generate_ts_data.py 먼저 실행")
    d = np.load(str(path))
    return d["X"].astype(np.float32), d["y_bin"].astype(np.int32), d["y_multi"].astype(np.int32)


def load_train() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _load_npz("train")


def load_test() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _load_npz("test")


def load_demo() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """demo 폴더의 개별 CSV 로드 (Streamlit 시연용)."""
    files = sorted(DEMO_DIR.glob("demo_*.csv"))
    if not files:
        raise FileNotFoundError(f"{DEMO_DIR} 에 demo_*.csv 없음")
    X_list, y_bin_list, y_multi_list = [], [], []
    for f in files:
        df = pd.read_csv(f)
        if df.shape[0] != SEQ_LEN:
            continue
        X_list.append(df[FEATURE_COLS].values.astype(np.float32))
        y_bin_list.append(int(df[TARGET_COL].iloc[-1]))
        y_multi_list.append(int(df[FAILURE_TYPE_COL].iloc[-1]))
    return np.stack(X_list), np.array(y_bin_list, np.int32), np.array(y_multi_list, np.int32)


def normalize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """train 기준 표준화."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def kfold_splits(X: np.ndarray, y: np.ndarray) -> Iterator[tuple]:
    """Stratified K-Fold (CV_FOLDS=3)."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=2026)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        yield fold, X[train_idx], X[val_idx], y[train_idx], y[val_idx]
