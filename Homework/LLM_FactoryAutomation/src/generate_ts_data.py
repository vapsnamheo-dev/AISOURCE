"""
시계열 합성 데이터 생성기 — AI4I 물리 규칙 기반 (개선판)
ML 프로젝트(AI4I 2020)와 동일한 피처로 시계열 시퀀스를 생성합니다.

각 시퀀스 = 한 대의 장비가 가동 시작부터 고장(또는 정상 종료)까지의 센서 기록
  - 정상 시퀀스: 50스텝 동안 물리 규칙 범위 유지
  - 고장 시퀀스: 마지막 15스텝에 특정 고장 패턴이 누적되어 임계값 초과
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    SEQ_LEN, SYNTH_TOTAL_RUNS, DEMO_RUNS, SYNTH_FAILURE_RATE,
    TRAIN_DIR, TEST_DIR, DEMO_DIR, TEST_RATIO,
    FEATURE_COLS, TARGET_COL, FAILURE_TYPE_COL,
    DATA_DIR, NOISE_LEVEL, DATA_VERSION,
)

RNG = np.random.default_rng(2026)
TYPE_MAP = {"L": 0, "M": 1, "H": 2}
OSF_THRESHOLD = {"L": 11_000, "M": 12_000, "H": 13_000}
FAILURE_TYPES = {0: "Normal", 1: "TWF", 2: "HDF", 3: "PWF", 4: "OSF", 5: "RNF"}

# 고장 유형별 생성 비율 (AI4I 파레토와 유사하게)
FAIL_TYPE_PROBS = [0.30, 0.25, 0.28, 0.12, 0.05]  # HDF, PWF, OSF, TWF, RNF


def _make_normal_sequence(equip_type: str) -> pd.DataFrame:
    """정상 운전 시퀀스: 50스텝 동안 물리 규칙 내 유지."""
    rows = []
    tool_wear = float(RNG.uniform(5, 80))
    for _ in range(SEQ_LEN):
        rpm = float(np.clip(RNG.normal(1550, 70), 1400, 2800))
        torque = float(np.clip(RNG.normal(38, 5), 10, 55))
        air_temp = float(np.clip(RNG.normal(300, 1.5), 296, 304))
        heat_diff = float(RNG.uniform(9.0, 11.0))
        process_temp = air_temp + heat_diff
        tool_wear = min(tool_wear + RNG.uniform(0.3, 1.0), 180)
        omega = rpm * 2 * np.pi / 60
        power_w = torque * omega
        overstrain = tool_wear * torque
        temp_diff = process_temp - air_temp
        rows.append({
            "air_temp_k": air_temp, "process_temp_k": process_temp,
            "rotational_speed_rpm": rpm, "torque_nm": torque,
            "tool_wear_min": tool_wear, "power_w": power_w,
            "overstrain_minnm": overstrain, "temp_diff_k": temp_diff,
        })
    df = pd.DataFrame(rows)
    df["type_encoded"] = TYPE_MAP[equip_type]
    df["failure"] = 0
    df["failure_type"] = 0
    df["equip_type"] = equip_type
    return df


def _make_failure_sequence(equip_type: str, fail_type: int) -> pd.DataFrame:
    """
    고장 시퀀스: 처음 35스텝은 정상, 마지막 15스텝에 해당 고장 패턴 주입.
    fail_type: 1=TWF, 2=HDF, 3=PWF, 4=OSF, 5=RNF
    """
    normal_steps = SEQ_LEN - 15
    rows = []
    tool_wear = float(RNG.uniform(60, 120))

    # 정상 구간
    for _ in range(normal_steps):
        rpm = float(np.clip(RNG.normal(1550, 70), 1400, 2800))
        torque = float(np.clip(RNG.normal(38, 5), 10, 55))
        air_temp = float(np.clip(RNG.normal(300, 1.5), 296, 304))
        process_temp = air_temp + float(RNG.uniform(9.0, 11.0))
        tool_wear = min(tool_wear + RNG.uniform(0.5, 1.2), 180)
        omega = rpm * 2 * np.pi / 60
        rows.append({
            "air_temp_k": air_temp, "process_temp_k": process_temp,
            "rotational_speed_rpm": rpm, "torque_nm": torque,
            "tool_wear_min": tool_wear, "power_w": torque * omega,
            "overstrain_minnm": tool_wear * torque,
            "temp_diff_k": process_temp - air_temp,
        })

    # 고장 패턴 구간 (15스텝 — 점진적 악화)
    for step in range(15):
        ratio = (step + 1) / 15  # 0→1 점진 증가

        if fail_type == 1:  # TWF: 공구 마모 급증 + 토크 급증
            tool_wear = min(tool_wear + RNG.uniform(3, 6), 260)
            torque = float(np.clip(RNG.normal(62 + ratio * 15, 3), 55, 80))
            rpm = float(np.clip(RNG.normal(1500, 60), 1200, 2000))
            air_temp = float(np.clip(RNG.normal(300, 1.5), 296, 304))
            process_temp = air_temp + float(RNG.uniform(9.0, 11.0))

        elif fail_type == 2:  # HDF: 냉각 불량 (temp_diff 감소) + rpm 저하
            tool_wear = min(tool_wear + RNG.uniform(0.5, 1.0), 180)
            rpm = float(np.clip(RNG.normal(1380 - ratio * 100, 40), 1100, 1400))
            torque = float(np.clip(RNG.normal(40, 5), 10, 60))
            air_temp = float(np.clip(RNG.normal(302, 1.5), 296, 305))
            # 냉각 저하: temp_diff가 8.6 이하로 하강
            heat_diff = max(9.5 - ratio * 2.0, 5.5)
            process_temp = air_temp + heat_diff

        elif fail_type == 3:  # PWF: 전력 범위 이탈 (너무 낮거나 너무 높음)
            tool_wear = min(tool_wear + RNG.uniform(0.5, 1.0), 180)
            # power < 3500 케이스: rpm 과도 저하
            rpm = float(np.clip(RNG.normal(1100 - ratio * 200, 50), 800, 1200))
            torque = float(np.clip(RNG.normal(30, 5), 5, 40))
            air_temp = float(np.clip(RNG.normal(300, 1.5), 296, 304))
            process_temp = air_temp + float(RNG.uniform(9.0, 11.0))

        elif fail_type == 4:  # OSF: 과부하 (tool_wear*torque 급증)
            thresh = OSF_THRESHOLD[equip_type]
            tool_wear = min(tool_wear + RNG.uniform(2, 4), 240)
            torque = float(np.clip(tool_wear * (1 + ratio) + RNG.normal(0, 5), 55, 80))
            # overstrain ≈ tool_wear * torque → thresh 초과 유도
            if tool_wear * torque < thresh * 0.9:
                torque = float(min(torque + thresh * 0.1 / max(tool_wear, 1), 80))
            rpm = float(np.clip(RNG.normal(1500, 60), 1200, 2000))
            air_temp = float(np.clip(RNG.normal(300, 1.5), 296, 304))
            process_temp = air_temp + float(RNG.uniform(9.0, 11.0))

        else:  # RNF: 랜덤 (임의 센서 스파이크)
            tool_wear = min(tool_wear + RNG.uniform(0.5, 1.0), 180)
            rpm = float(np.clip(RNG.normal(1500 + RNG.normal(0, 200), 80), 1100, 2900))
            torque = float(np.clip(RNG.normal(40 + RNG.normal(0, 10), 5), 5, 75))
            air_temp = float(np.clip(RNG.normal(300, 2), 296, 304))
            process_temp = air_temp + float(RNG.uniform(8.0, 12.0))

        omega = rpm * 2 * np.pi / 60
        rows.append({
            "air_temp_k": air_temp, "process_temp_k": process_temp,
            "rotational_speed_rpm": rpm, "torque_nm": torque,
            "tool_wear_min": tool_wear, "power_w": torque * omega,
            "overstrain_minnm": tool_wear * torque,
            "temp_diff_k": process_temp - air_temp,
        })

    df = pd.DataFrame(rows)
    df["type_encoded"] = TYPE_MAP[equip_type]
    df["failure"] = 1
    df["failure_type"] = fail_type
    df["equip_type"] = equip_type
    return df


def generate_run(equip_type=None, force_failure: bool = False) -> pd.DataFrame:
    if equip_type is None:
        equip_type = str(RNG.choice(["L", "L", "L", "M", "M", "H"]))
    if force_failure:
        ft = int(RNG.choice([1, 2, 3, 4, 5], p=FAIL_TYPE_PROBS))
        return _make_failure_sequence(equip_type, ft)
    return _make_normal_sequence(equip_type)


def generate_dataset(n_runs: int, seed_offset: int = 0) -> list[pd.DataFrame]:
    global RNG
    RNG = np.random.default_rng(2026 + seed_offset)
    n_failure = int(n_runs * SYNTH_FAILURE_RATE)
    n_normal = n_runs - n_failure
    runs = []
    for _ in range(n_failure):
        et = str(RNG.choice(["L", "L", "L", "M", "M", "H"]))
        runs.append(generate_run(equip_type=et, force_failure=True))
    for _ in range(n_normal):
        et = str(RNG.choice(["L", "L", "L", "M", "M", "H"]))
        runs.append(generate_run(equip_type=et, force_failure=False))
    idx = list(range(len(runs)))
    RNG.shuffle(idx)
    return [runs[i] for i in idx]


def save_sequences(runs: list, out_dir: Path, prefix: str = "run"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i, df in enumerate(runs):
        df.to_csv(Path(out_dir) / f"{prefix}_{i:05d}.csv", index=False)


_NOISE_COLS = [
    "air_temp_k", "process_temp_k", "rotational_speed_rpm",
    "torque_nm", "tool_wear_min", "power_w", "overstrain_minnm", "temp_diff_k",
]


def _apply_noise(df: pd.DataFrame, noise_level: float) -> pd.DataFrame:
    """각 타임스텝 센서값에 가우시안 계측 노이즈 추가 (noise_level=표준편차 비율)."""
    if noise_level <= 0:
        return df
    df = df.copy()
    noise = RNG.normal(1.0, noise_level, size=(len(df), len(_NOISE_COLS)))
    df[_NOISE_COLS] = df[_NOISE_COLS].values * noise
    return df


def _runs_to_arrays(runs: list) -> tuple:
    """시퀀스 리스트 → (X, y_bin, y_multi) 배열 변환."""
    X, y_bin, y_multi = [], [], []
    for df in runs:
        X.append(df[FEATURE_COLS].values.astype("float32"))
        y_bin.append(int(df[TARGET_COL].iloc[-1]))
        y_multi.append(int(df[FAILURE_TYPE_COL].iloc[-1]))
    return (
        np.stack(X).astype("float32"),
        np.array(y_bin, dtype="int32"),
        np.array(y_multi, dtype="int32"),
    )


def _print_summary(name: str, runs: list):
    n_fail = sum(int(r["failure"].iloc[-1]) for r in runs)
    print(f"  [{name}] {len(runs)}개 시퀀스 | 고장: {n_fail} ({n_fail/len(runs)*100:.1f}%)")


def build_and_split():
    noise_tag = f"노이즈 {NOISE_LEVEL*100:.0f}%" if NOISE_LEVEL > 0 else "노이즈 없음"
    print(f"[{DATA_VERSION}] 총 {SYNTH_TOTAL_RUNS}개 시퀀스 생성 중 ({noise_tag})...")

    all_runs = generate_dataset(SYNTH_TOTAL_RUNS, seed_offset=0)
    if NOISE_LEVEL > 0:
        all_runs = [_apply_noise(r, NOISE_LEVEL) for r in all_runs]

    n_test = int(SYNTH_TOTAL_RUNS * TEST_RATIO)
    train_runs = all_runs[n_test:]
    test_runs  = all_runs[:n_test]
    print(f"  → train: {len(train_runs)}개 (80%), test: {len(test_runs)}개 (20%)")

    # NPZ로 직접 저장 (중간 CSV 없이, 60,000개 대용량 대응)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_tr, y_tr_bin, y_tr_multi = _runs_to_arrays(train_runs)
    np.savez_compressed(str(DATA_DIR / "train_data.npz"),
                        X=X_tr, y_bin=y_tr_bin, y_multi=y_tr_multi)
    X_te, y_te_bin, y_te_multi = _runs_to_arrays(test_runs)
    np.savez_compressed(str(DATA_DIR / "test_data.npz"),
                        X=X_te, y_bin=y_te_bin, y_multi=y_te_multi)

    # 데모는 Streamlit용으로 개별 CSV 유지
    print(f"데모 데이터 {DEMO_RUNS}개 생성 중 (시드 9999, train/test와 무관)...")
    demo_runs = generate_dataset(DEMO_RUNS, seed_offset=9999)
    if NOISE_LEVEL > 0:
        demo_runs = [_apply_noise(r, NOISE_LEVEL) for r in demo_runs]
    save_sequences(demo_runs, DEMO_DIR, prefix="demo")

    _print_summary("Train", train_runs)
    _print_summary("Test", test_runs)
    _print_summary("Demo", demo_runs)
    print(f"\n완료: data/train_data.npz, data/test_data.npz, data/demo/ 저장됨 [{DATA_VERSION}]")


if __name__ == "__main__":
    build_and_split()
