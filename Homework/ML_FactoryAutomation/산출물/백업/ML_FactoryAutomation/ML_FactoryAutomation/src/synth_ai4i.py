"""AI4I 2020의 공개된 '물리 고장 규칙'을 그대로 적용해 더 큰 원본급 데이터를 생성.

- 딥러닝 생성(2차 합성)이 아니라, 원본과 동일한 1차 물리 규칙으로 라벨 부여.
- 특성은 실제 AI4I 분포를 부트스트랩 + 소량 지터로 새 점을 만들어 분포를 보존.
규칙(Matzka 2020):
  HDF: (process_temp - air_temp) < 8.6 K AND rotational_speed < 1380 rpm
  PWF: power = torque * (rpm * 2π/60); power < 3500 W OR > 9000 W
  OSF: tool_wear * torque > 11000(L)/12000(M)/13000(H) minNm
  TWF: tool_wear 가 200~240 구간이면 확률적 고장(약 51/120)
  RNF: 0.1% 무작위
  Target = 위 다섯 모드 중 하나라도 발생
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src import config

NUM = config.NUMERIC_FEATURES
OSF_THR = {"L": 11000, "M": 12000, "H": 13000}


def _apply_rules(df: pd.DataFrame, rng: np.random.RandomState) -> np.ndarray:
    air = df["Air temperature [K]"].values
    proc = df["Process temperature [K]"].values
    rpm = df["Rotational speed [rpm]"].values
    tau = df["Torque [Nm]"].values
    wear = df["Tool wear [min]"].values
    typ = df["Type"].values
    n = len(df)

    hdf = ((proc - air) < 8.6) & (rpm < 1380)
    power = tau * (rpm * 2 * np.pi / 60.0)
    pwf = (power < 3500) | (power > 9000)
    thr = np.array([OSF_THR[t] for t in typ])
    osf = (wear * tau) > thr
    twf_zone = (wear >= rng.uniform(200, 240, n))
    twf = twf_zone & (rng.random(n) < 51.0 / 120.0)
    rnf = rng.random(n) < 0.001
    return (hdf | pwf | osf | twf | rnf).astype(int)


def generate(n: int = 100000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    real = pd.read_csv(config.DATA_PATH)
    cols = ["Type"] + NUM
    base = real[cols].sample(n, replace=True, random_state=seed).reset_index(drop=True)
    # 소량 지터로 '새로운' 물리적 점 생성 (분포 보존)
    base["Air temperature [K]"] += rng.normal(0, 0.3, n)
    base["Process temperature [K]"] += rng.normal(0, 0.3, n)
    base["Rotational speed [rpm]"] = (base["Rotational speed [rpm]"] + rng.normal(0, 25, n)).round().clip(1160, 2900).astype(int)
    base["Torque [Nm]"] = (base["Torque [Nm]"] + rng.normal(0, 1.0, n)).clip(3, 80).round(1)
    base["Tool wear [min]"] = (base["Tool wear [min]"] + rng.randint(-3, 4, n)).clip(0, 253).astype(int)
    base["Air temperature [K]"] = base["Air temperature [K]"].round(1)
    base["Process temperature [K]"] = base["Process temperature [K]"].round(1)
    base["Target"] = _apply_rules(base, rng)
    return base


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    df = generate(n)
    out = config.BASE_DIR / "data" / "ai4i_physics_100k.csv"
    df.to_csv(out, index=False)
    print(f"생성 완료: {out.name}  {len(df):,}행  고장 {int(df['Target'].sum()):,} ({df['Target'].mean()*100:.2f}%)")


def actual_label(type_, air, proc, rpm, torque, wear) -> int:
    """단건 입력에 대한 물리규칙 기반 실제(정답) 라벨 — 결정적(RNF 제외).

    HDF/PWF/OSF는 결정적, TWF는 마모 220min 이상을 고장으로 근사.
    """
    hdf = ((proc - air) < 8.6) and (rpm < 1380)
    power = torque * (rpm * 2 * np.pi / 60.0)
    pwf = (power < 3500) or (power > 9000)
    osf = (wear * torque) > OSF_THR.get(type_, 12000)
    twf = wear >= 220
    return int(bool(hdf or pwf or osf or twf))
