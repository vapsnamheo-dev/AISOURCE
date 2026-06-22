# ════════════════════════════════════════════════════════════════════
# [역할] 탐색적 데이터 분석 — 분포·상관(heatmap)·클래스 불균형 확인.
# [단계] [2] EDA (보고서 3장)
# [작업 메모] 불균형(고장 3.4%) 시각 확인.
# (이 헤더는 이전/이번 작업 맥락을 요약한 주석입니다 — 기능에는 영향 없음)
# ════════════════════════════════════════════════════════════════════
"""[2] EDA: 분포 · 상관관계(heatmap) · 불균형 확인 (샘플 PDF EDA 흐름 대응)."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src import config, data_loader


def run():
    df = data_loader.load_data()
    num = config.NUMERIC_FEATURES  # heatmap은 수치형만 (범주형·Target 제외)

    # 분포
    df[config.NUMERIC_FEATURES].hist(figsize=(11, 6), bins=30)
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "eda_distributions.png", dpi=120)
    plt.close()

    # 상관관계 heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "eda_heatmap.png", dpi=120)
    plt.close()

    dist = df[config.TARGET].value_counts().to_dict()
    print("Target 분포(불균형):", dist, "| 고장비율:", round(dist.get(1, 0) / len(df), 4))
    print("EDA 리포트 저장: eda_distributions.png, eda_heatmap.png")


if __name__ == "__main__":
    run()
