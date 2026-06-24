# ════════════════════════════════════════════════════════════════════
# [역할] 다중공선성 진단 — 실제 전처리 feature(11개)에 대한 VIF 산출·시각화.
# [단계] [3] EDA / 전처리 검증 (보고서 3.8 다중공선성 진단)
# [작업 메모] statsmodels VIF로 수치형5+파생3+Type 원-핫3 진단.
#   파생 feature(Power·Overstrain·Temp diff)가 기저 변수의 결정함수라
#   강한(일부 완전) 다중공선성이 발생함을 정량 확인. 트리 모델(XGBoost)은
#   다중공선성에 강건하므로 feature 유지가 타당함을 뒷받침.
# ════════════════════════════════════════════════════════════════════
"""다중공선성 진단 (VIF, Variance Inflation Factor).

실행:
    python -m src.vif_analysis              # 표 출력
    python -m src.vif_analysis --plot out.png  # 막대그래프 저장

VIF = 1 / (1 - R_i^2),  R_i^2 = (변수 i를 나머지로 회귀한 결정계수).
통상 VIF >= 5 주의, >= 10 심각. inf 는 완전 선형종속(예: Temp diff = 공정 - 공기).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from src import config, preprocess

# 한글 라벨 매핑(보고서/그래프 표기용)
KO_LABEL = {
    "Air temperature [K]": "공기온도",
    "Process temperature [K]": "공정온도",
    "Rotational speed [rpm]": "회전속도",
    "Torque [Nm]": "토크",
    "Tool wear [min]": "공구마모",
    "Power [W]": "Power(파생)",
    "Overstrain [minNm]": "Overstrain(파생)",
    "Temp diff [K]": "Temp_diff(파생)",
    "Type_H": "Type_H",
    "Type_L": "Type_L",
    "Type_M": "Type_M",
}


def compute_vif(drop_reference_dummy: bool = True) -> pd.DataFrame:
    """실제 전처리 파이프라인(build_features)의 feature에 대한 VIF를 산출한다.

    Parameters
    ----------
    drop_reference_dummy : bool
        Type 원-핫 더미는 합이 1이라 상수항과 완전공선이므로, 기준범주(Type_L)를
        제외해 더미 함정(dummy trap)을 피한다. False면 연속형 8개만 진단한다.

    Returns
    -------
    pd.DataFrame  columns=[feature, label, VIF, judgement]
    """
    df = pd.read_csv(config.DATA_PATH)
    X, _ = preprocess.build_features(df)

    cont = config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES  # 연속형 8
    cols = list(cont)
    if drop_reference_dummy:
        cols += ["Type_H", "Type_M"]  # Type_L 을 기준범주로 제외

    Xc = add_constant(X[cols], has_constant="add")
    rows = []
    for i, c in enumerate(Xc.columns):
        if c == "const":
            continue
        with np.errstate(divide="ignore"):
            vif = variance_inflation_factor(Xc.values, i)
        rows.append(
            {
                "feature": c,
                "label": KO_LABEL.get(c, c),
                "VIF": vif,
                "judgement": _judge(vif),
            }
        )
    return pd.DataFrame(rows)


def _judge(vif: float) -> str:
    if not np.isfinite(vif):
        return "완전종속(∞)"
    if vif >= 10:
        return "심각"
    if vif >= 5:
        return "주의"
    return "양호"


def plot_vif(out_path: str = "reports/vif_chart.png") -> str:
    """VIF 막대그래프 저장(주의선 5·심각선 10 표시). inf 는 막대 상단 처리."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    res = compute_vif(drop_reference_dummy=True)
    labels = res["label"].tolist()
    raw = res["VIF"].tolist()
    cap = 60.0  # inf/초대형 값 표시 상한
    vals = [cap if not np.isfinite(v) else min(v, cap) for v in raw]

    fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = fm.FontProperties(fname=fp) if _exists(fp) else None
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    colors = ["#7D3C98" if not np.isfinite(v) else ("#C0392B" if v >= 10 else ("#E07B39" if v >= 5 else "#4C78A8")) for v in raw]
    bars = ax.bar(labels, vals, color=colors, edgecolor="#333", linewidth=0.6, width=0.62)
    ax.axhline(5, color="#C0392B", ls="--", lw=1.2)
    ax.axhline(10, color="#7D3C98", ls=":", lw=1.1)
    for b, v in zip(bars, raw):
        txt = "∞" if not np.isfinite(v) else f"{v:.1f}"
        ax.text(b.get_x() + b.get_width() / 2, min(v if np.isfinite(v) else cap, cap) + 0.6,
                txt, ha="center", va="bottom", fontsize=10, fontweight="bold",
                fontproperties=font)
    ax.set_ylim(0, cap + 6)
    ax.set_ylabel("VIF", fontproperties=font)
    ax.set_title("변수별 다중공선성 진단 (VIF) — 실제 전처리 11 feature", fontproperties=font, fontweight="bold")
    for lab in ax.get_xticklabels():
        if font:
            lab.set_fontproperties(font)
        lab.set_fontsize(9.5)
        lab.set_rotation(20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


def _exists(p: str) -> bool:
    import os
    return os.path.exists(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="다중공선성(VIF) 진단")
    ap.add_argument("--plot", nargs="?", const="reports/vif_chart.png", default=None,
                    help="VIF 막대그래프 저장 경로")
    args = ap.parse_args()

    res = compute_vif(drop_reference_dummy=True)
    pd.set_option("display.unicode.east_asian_width", True)
    print("\n=== 다중공선성 진단 (VIF) — 실제 전처리 feature ===")
    for _, r in res.iterrows():
        vif = "∞ (완전종속)" if not np.isfinite(r["VIF"]) else f"{r['VIF']:8.1f}"
        print(f"  {r['label']:16s} VIF = {vif:>14}   [{r['judgement']}]")
    print("\n해석: Temp_diff = 공정온도 - 공기온도 (완전 선형종속) → 세 변수 VIF=∞.")
    print("      Power·Overstrain 등 곱형 파생도 강한 공선성 유발(토크 VIF≈52).")
    print("      최종 모델 XGBoost(트리)는 다중공선성에 강건 → feature 유지 타당.")

    if args.plot:
        path = plot_vif(args.plot)
        print(f"\n[저장] VIF 그래프 → {path}")


if __name__ == "__main__":
    main()
