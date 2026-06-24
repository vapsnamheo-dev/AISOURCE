# ════════════════════════════════════════════════════════════════════
# [역할] FEMTO-ST 베어링 데이터 전처리 — 피처 추출, VIF 분석, 열화 라벨 생성
# [단계] [1] 데이터 로딩 → 피처 추출 → VIF 분석 → 라벨 생성 → 저장
# [작업 메모] ML만 적용시 기준(h_rms × 2.5 임계값) + VIF 기반 피처 선택 적용.
#   데이터 없을 시 generate_synthetic_femto() 로 demo mode 자동 전환.
# ════════════════════════════════════════════════════════════════════
"""FEMTO-ST PRONOSTIA 베어링 예지보전 — 전처리 모듈.

실행:
    python -m src.femto_preprocess

출력:
    data/FEMTO_processed/femto_features.csv   (전체 피처)
    data/FEMTO_processed/selected_features.csv (VIF 선택 피처 목록)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FEMTO_DIR = ROOT / "data" / "FEMTO"
PROCESSED_DIR = ROOT / "data" / "FEMTO_processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# FEMTO 베어링 목록 (학습/테스트 분리)
TRAIN_BEARINGS = [
    ("Learning_set", "Bearing1_1", "train"),
    ("Learning_set", "Bearing1_2", "train"),
    ("Learning_set", "Bearing1_3", "train"),
]
TEST_BEARINGS = [
    ("Test_set", "Bearing1_4", "test"),
    ("Test_set", "Bearing1_5", "test"),
    ("Test_set", "Bearing1_6", "test"),
    ("Test_set", "Bearing1_7", "test"),
]

# 피처 목록 (수평+수직 채널)
FEATURE_COLS = [
    "h_rms", "h_kurt", "h_skew", "h_crest",
    "v_rms", "v_kurt", "v_skew", "v_crest",
    "temp_mean",
]


# ── 피처 추출 함수 ─────────────────────────────────────────────────────────────

def _extract_features_from_signal(signal: np.ndarray) -> dict:
    """1D 진동 신호 배열에서 시간영역 피처를 추출한다."""
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    kurt = float(stats.kurtosis(signal, fisher=True) + 3)   # excess → normal kurtosis
    skew = float(stats.skew(signal))
    crest = float(peak / (rms + 1e-10))
    return {"rms": rms, "kurt": kurt, "skew": skew, "crest": crest}


def _load_acc_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """acc_XXXXX.csv 로딩 — 2열(h_acc, v_acc) 또는 6열(+시간 4열) 자동 감지."""
    try:
        df = pd.read_csv(filepath, header=None)
        if df.shape[1] == 2:
            df.columns = ["h_acc", "v_acc"]
        elif df.shape[1] >= 6:
            df = df.iloc[:, -2:]
            df.columns = ["h_acc", "v_acc"]
        else:
            return None
        return df.astype(float)
    except Exception:
        return None


def _load_temp_csv(filepath: Path) -> Optional[float]:
    """temp_XXXXX.csv 에서 평균 온도(°C)를 반환한다."""
    try:
        df = pd.read_csv(filepath, header=None)
        if df.shape[1] == 2:
            temp = df.iloc[:, 0].mean()
        elif df.shape[1] >= 6:
            temp = df.iloc[:, -2].mean()
        else:
            return None
        return float(temp)
    except Exception:
        return None


def extract_bearing_features(bearing_dir: Path) -> pd.DataFrame:
    """베어링 디렉토리에서 스냅샷별 피처 DataFrame을 추출한다."""
    acc_files = sorted(bearing_dir.glob("acc_*.csv"))
    if not acc_files:
        return pd.DataFrame()

    rows = []
    for i, acc_path in enumerate(acc_files):
        df_acc = _load_acc_csv(acc_path)
        if df_acc is None:
            continue

        h_feat = _extract_features_from_signal(df_acc["h_acc"].values)
        v_feat = _extract_features_from_signal(df_acc["v_acc"].values)

        # 온도 파일 매칭 (acc_00001.csv → temp_00001.csv)
        temp_path = acc_path.parent / acc_path.name.replace("acc_", "temp_")
        temp_val = _load_temp_csv(temp_path) if temp_path.exists() else np.nan

        rows.append({
            "minute": i + 1,
            "h_rms": h_feat["rms"],
            "h_kurt": h_feat["kurt"],
            "h_skew": h_feat["skew"],
            "h_crest": h_feat["crest"],
            "v_rms": v_feat["rms"],
            "v_kurt": v_feat["kurt"],
            "v_skew": v_feat["skew"],
            "v_crest": v_feat["crest"],
            "temp_mean": temp_val,
        })

    return pd.DataFrame(rows)


# ── VIF 분석 ───────────────────────────────────────────────────────────────────

def compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """전체 피처에 대해 VIF를 계산하여 DataFrame으로 반환한다."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant

        X_sub = df[features].dropna()
        if len(X_sub) < 10:
            return pd.DataFrame({"feature": features, "VIF": [np.nan] * len(features),
                                 "judgement": ["데이터 부족"] * len(features)})

        X_const = add_constant(X_sub, has_constant="add")
        rows = []
        for i, col in enumerate(X_const.columns):
            if col == "const":
                continue
            with np.errstate(divide="ignore"):
                vif = variance_inflation_factor(X_const.values, i)
            rows.append({
                "feature": col,
                "VIF": round(float(vif), 2),
                "judgement": _vif_judge(vif),
            })
        return pd.DataFrame(rows)

    except ImportError:
        print("[경고] statsmodels 미설치 → VIF 분석 생략")
        return pd.DataFrame({"feature": features, "VIF": [np.nan] * len(features),
                             "judgement": ["미분석"] * len(features)})


def _vif_judge(vif: float) -> str:
    if not np.isfinite(vif):
        return "완전종속"
    if vif >= 10:
        return "심각"
    if vif >= 5:
        return "주의"
    return "양호"


def select_features_by_vif(vif_df: pd.DataFrame, threshold: float = 10.0) -> list[str]:
    """VIF < threshold 인 피처만 선택한다."""
    selected = vif_df[vif_df["VIF"].fillna(threshold + 1) < threshold]["feature"].tolist()
    if not selected:
        # 모두 제거되면 전체 유지 (안전장치)
        selected = vif_df["feature"].tolist()
        print("[경고] VIF 기준으로 모든 피처가 제거됨 → 전체 피처 유지")
    return selected


# ── 합성 데이터 생성 (Demo Mode) ───────────────────────────────────────────────

def generate_synthetic_femto(n_bearings: int = 3, minutes_per_bearing: int = 100) -> pd.DataFrame:
    """FEMTO-ST 유사 합성 데이터 생성 (demo용).

    Parameters
    ----------
    n_bearings : int
        생성할 베어링 수 (기본 3개: train 2개 + test 1개).
    minutes_per_bearing : int
        베어링당 평균 수명 (분 단위, ±20 랜덤).
    """
    np.random.seed(42)
    rows = []
    for i in range(n_bearings):
        total = minutes_per_bearing + np.random.randint(-20, 20)
        baseline_rms = np.random.uniform(0.3, 0.6)
        # 합성 데이터: 초기 10분 평균에 맞춰 임계값 설정
        # 후반 30% 구간에서 degradation=5*(0.3)^2=0.45 → rms_max ≈ baseline*1.45
        # threshold를 1.2×로 낮춰서 열화 라벨이 실제로 생성되도록 함
        threshold = baseline_rms * 1.2
        split = "test" if i >= n_bearings - 1 else "train"
        bearing_name = f"SynBearing1_{i + 1}"

        for t in range(1, total + 1):
            progress = t / total
            # 열화 진행 곡선: 초반 평탄 → 70% 이후 급상승 (최대 약 +45%)
            degradation = 5 * max(0.0, progress - 0.7) ** 2
            rms = baseline_rms * (1 + degradation + np.random.normal(0, 0.02))
            rms = max(rms, 0.01)

            h_rms = rms
            v_rms = rms * np.random.uniform(0.8, 1.0)
            h_kurt = np.random.normal(3, 0.5) + progress * 2
            v_kurt = np.random.normal(3, 0.5) + progress * 1.5
            h_skew = np.random.normal(0, 0.2)
            v_skew = np.random.normal(0, 0.2)
            h_crest = np.random.uniform(3, 6) + progress * 2
            v_crest = np.random.uniform(3, 5) + progress * 1.5
            temp = 25 + progress * 15 + np.random.normal(0, 0.5)

            rows.append({
                "bearing": bearing_name,
                "minute": t,
                "h_rms": round(h_rms, 6),
                "h_kurt": round(h_kurt, 4),
                "h_skew": round(h_skew, 4),
                "h_crest": round(h_crest, 4),
                "v_rms": round(v_rms, 6),
                "v_kurt": round(v_kurt, 4),
                "v_skew": round(v_skew, 4),
                "v_crest": round(v_crest, 4),
                "temp_mean": round(temp, 2),
                "rul": total - t,
                "rul_pct": round((total - t) / total, 4),
                "label": int(h_rms > threshold),
                "threshold": round(threshold, 6),
                "split": split,
            })

    df = pd.DataFrame(rows)
    print(f"[합성 데이터] {n_bearings}개 베어링, {len(df)}행 생성 (demo mode)")
    return df


# ── 실제 FEMTO 데이터 로딩 ────────────────────────────────────────────────────

def load_femto_data() -> tuple[pd.DataFrame, bool]:
    """FEMTO-ST 데이터를 로딩한다. 없으면 합성 데이터 반환 (demo mode).

    Returns
    -------
    (DataFrame, is_synthetic)
    """
    all_bearings = TRAIN_BEARINGS + TEST_BEARINGS
    rows = []

    for subset, bearing_name, split in all_bearings:
        bearing_dir = FEMTO_DIR / subset / bearing_name
        if not bearing_dir.exists():
            continue

        df_feat = extract_bearing_features(bearing_dir)
        if df_feat.empty:
            continue

        df_feat["bearing"] = bearing_name
        df_feat["split"] = split

        # ML만 적용시 기준: 초기 10분 h_rms 평균 × 2.5 = 열화 임계값
        init_10 = df_feat["h_rms"].iloc[:10].mean()
        threshold = init_10 * 2.5
        df_feat["threshold"] = threshold
        df_feat["label"] = (df_feat["h_rms"] > threshold).astype(int)

        # RUL 계산
        total_min = len(df_feat)
        df_feat["rul"] = total_min - df_feat["minute"]
        df_feat["rul_pct"] = (df_feat["rul"] / total_min).clip(0, 1)

        rows.append(df_feat)

    if not rows:
        print("[알림] FEMTO 데이터 없음 → 합성 데이터(demo mode) 사용")
        df_syn = generate_synthetic_femto(n_bearings=6, minutes_per_bearing=100)
        return df_syn, True

    df = pd.concat(rows, ignore_index=True)
    print(f"[FEMTO 실데이터] {df['bearing'].nunique()}개 베어링, {len(df)}행 로딩 완료")
    return df, False


# ── 메인 파이프라인 ────────────────────────────────────────────────────────────

def run() -> None:
    """전처리 파이프라인 실행."""
    print("=" * 60)
    print("FEMTO-ST 베어링 전처리 시작")
    print("=" * 60)

    # 1. 데이터 로딩
    df, is_synthetic = load_femto_data()

    # 2. 온도 결측 처리 (중앙값 대체)
    if df["temp_mean"].isna().any():
        df["temp_mean"] = df["temp_mean"].fillna(df["temp_mean"].median())

    # 3. VIF 분석
    print("\n[VIF 분석] 다중공선성 진단 중...")
    vif_df = compute_vif(df, FEATURE_COLS)
    print(vif_df.to_string(index=False))

    # 4. VIF 기반 피처 선택
    selected = select_features_by_vif(vif_df, threshold=10.0)
    print(f"\n[피처 선택] VIF<10 유지: {selected}")

    # 5. 저장
    out_path = PROCESSED_DIR / "femto_features.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[저장] 전체 피처 → {out_path} ({len(df)}행)")

    sel_path = PROCESSED_DIR / "selected_features.csv"
    pd.DataFrame({"feature": selected}).to_csv(sel_path, index=False, encoding="utf-8")
    print(f"[저장] 선택 피처 → {sel_path}")

    vif_path = PROCESSED_DIR / "vif_results.csv"
    vif_df.to_csv(vif_path, index=False, encoding="utf-8")
    print(f"[저장] VIF 결과 → {vif_path}")

    # 6. 요약
    print("\n[요약]")
    print(f"  베어링 수: {df['bearing'].nunique()}")
    print(f"  전체 스냅샷: {len(df)}")
    print(f"  열화(label=1): {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"  데이터 모드: {'합성(Demo)' if is_synthetic else '실제 FEMTO-ST'}")
    print("=" * 60)


if __name__ == "__main__":
    run()
