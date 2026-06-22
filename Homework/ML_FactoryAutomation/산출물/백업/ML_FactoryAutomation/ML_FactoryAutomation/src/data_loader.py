"""[1] 데이터 로드 · [2] 데이터 확인."""
from __future__ import annotations
import pandas as pd
from src import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "n_missing": int(df.isnull().sum().sum()),
        "target_dist": df[config.TARGET].value_counts().to_dict(),
        "columns": list(df.columns),
    }


if __name__ == "__main__":
    d = load_data()
    s = summarize(d)
    print("데이터셋 크기:", s["shape"])
    print("결측치 합계:", s["n_missing"])
    print("Target 분포:", s["target_dist"])
    print("\n처음 5개 행:\n", d.head())
    print("\n기술통계:\n", d.describe())
