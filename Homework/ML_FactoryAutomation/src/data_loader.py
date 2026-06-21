# ════════════════════════════════════════════════════════════════════
# [역할] 데이터 로드 및 기본 요약(행수·결측·분포).
# [단계] [1] 데이터 수집 (보고서 1장)
# [작업 메모] AI4I 10k CSV 로드·검증.
# (이 헤더는 이전/이번 작업 맥락을 요약한 주석입니다 — 기능에는 영향 없음)
# ════════════════════════════════════════════════════════════════════
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
