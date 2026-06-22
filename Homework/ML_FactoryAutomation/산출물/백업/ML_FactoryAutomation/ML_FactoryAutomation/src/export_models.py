"""학습 모델을 model/ 폴더에 pickle(.pkl)로 저장하고, 모델 정보 메타데이터를 생성한다.

저장 대상:
  - 각 모델 pickle: xgb_model.pkl / rf_model.pkl / logreg_model.pkl
  - 전처리: scaler.pkl / feature_columns.pkl
  - model_info.json : 사용 모델·하이퍼파라미터, 로지스틱회귀 가중치(계수)·절편, 지표
  - model_card.md   : 사람이 읽기 쉬운 요약
"""
from __future__ import annotations

import datetime
import json
import pickle

import joblib
import numpy as np
import pandas as pd

from src import config

OUT = config.BASE_DIR / "model"
LABELS = {"xgb": "XGBoost", "rf": "RandomForest", "logreg": "LogisticRegression"}


def _metrics(which: str) -> dict:
    """reports/model_comparison.csv 에서 해당 모델 지표 로드(있으면)."""
    f = config.REPORTS_DIR / "model_comparison.csv"
    name_map = {"xgb": "XGBoost", "rf": "RandomForest", "logreg": "LogisticRegression"}
    if not f.exists():
        return {}
    df = pd.read_csv(f, index_col=0)
    key = name_map.get(which, which)
    if key not in df.index:
        return {}
    return {k: round(float(v), 4) for k, v in df.loc[key].items()}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    scaler = joblib.load(config.MODEL_PATHS["scaler"])
    cols = list(joblib.load(config.MODEL_PATHS["columns"]))

    with open(OUT / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(OUT / "feature_columns.pkl", "wb") as f:
        pickle.dump(cols, f)

    info = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": "AI4I 2020 Predictive Maintenance (10,000 rows, 8:2 train/test)",
        "target": {"0": "정상(normal)", "1": "고장(failure)"},
        "n_features": len(cols),
        "features": cols,
        "scaler": {
            "type": "StandardScaler",
            "mean": dict(zip(cols, np.round(scaler.mean_, 4).tolist())),
            "scale": dict(zip(cols, np.round(scaler.scale_, 4).tolist())),
        },
        "models": {},
    }

    for key, label in LABELS.items():
        model = joblib.load(config.MODEL_PATHS[key])
        fp = OUT / f"{key}_model.pkl"
        with open(fp, "wb") as f:
            pickle.dump(model, f)
        entry = {
            "file": fp.name,
            "algorithm": type(model).__name__,
            "hyperparameters": model.get_params(),
            "metrics": _metrics(key),
        }
        if key == "logreg":
            w_std = model.coef_[0]
            b_std = float(model.intercept_[0])
            # 원본 feature 공간 환산: x'=(x-mean)/scale 이므로 w_orig = w_std/scale
            w_orig = w_std / scaler.scale_
            b_orig = b_std - float(np.sum(w_std * scaler.mean_ / scaler.scale_))
            entry["weights_standardized"] = dict(zip(cols, np.round(w_std, 4).tolist()))
            entry["intercept_standardized"] = round(b_std, 4)
            entry["weights_original_space"] = dict(zip(cols, np.round(w_orig, 6).tolist()))
            entry["intercept_original_space"] = round(b_orig, 4)
            entry["note"] = "모델은 StandardScaler 표준화 입력을 사용. weights_standardized가 실제 학습 가중치이며, weights_original_space는 원본 단위 환산값."
        info["models"][label] = entry

    with open(OUT / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=str)

    _write_card(info)
    print("모델 저장 완료 →", OUT)
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size:,} bytes)")


def _write_card(info: dict) -> None:
    lines = [
        "# 모델 카드 (Model Card) — 설비 고장 예측",
        "",
        f"- 생성: {info['created_at']}",
        f"- 데이터: {info['dataset']}",
        "- 타깃: 0=정상(normal), 1=고장(failure)",
        f"- Feature({info['n_features']}개): {', '.join(info['features'])}",
        "- 스케일러: StandardScaler",
        "",
        "## 모델별 하이퍼파라미터·지표",
    ]
    for label, e in info["models"].items():
        lines.append(f"\n### {label} ({e['algorithm']}) — `{e['file']}`")
        m = e.get("metrics") or {}
        if m:
            lines.append("- 지표: " + ", ".join(f"{k} {v}" for k, v in m.items()))
        hp = {k: v for k, v in e["hyperparameters"].items() if v is not None}
        lines.append("- 주요 하이퍼파라미터: " + ", ".join(f"{k}={v}" for k, v in list(hp.items())[:12]))
        if label == "LogisticRegression":
            lines.append(f"- 절편(표준화): {e['intercept_standardized']}")
            lines.append("- 가중치(표준화 공간):")
            for fname, w in e["weights_standardized"].items():
                lines.append(f"    - {fname}: {w}")
    (OUT / "model_card.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
