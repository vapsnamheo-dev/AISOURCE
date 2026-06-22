"""추론 서비스 + DB 로깅."""
from __future__ import annotations
import pickle
import joblib
import pandas as pd
from src import config, db


def load_artifacts(which: str = "xgb"):
    """학습된 모델을 로드(재학습 없음).

    우선순위: model/{which}_model.pkl (GitHub 업로드용 피클) → 없으면 models/*.joblib.
    시연 시 1·5·6번 데이터를 다시 학습하지 않고 저장된 모델만 읽어온다.
    """
    pkl_dir = config.BASE_DIR / "model"
    pkl = pkl_dir / f"{which}_model.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            model = pickle.load(f)
        with open(pkl_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(pkl_dir / "feature_columns.pkl", "rb") as f:
            cols = pickle.load(f)
        return model, scaler, cols
    model = joblib.load(config.MODEL_PATHS[which])
    scaler = joblib.load(config.MODEL_PATHS["scaler"])
    cols = joblib.load(config.MODEL_PATHS["columns"])
    return model, scaler, cols


def make_feature_row(inp: dict, cols: list[str]) -> pd.DataFrame:
    """입력 dict -> 학습 피처 순서에 맞는 1행 DataFrame."""
    row = {c: 0 for c in cols}
    mapping = {
        "Air temperature [K]": inp["air_temperature"],
        "Process temperature [K]": inp["process_temperature"],
        "Rotational speed [rpm]": inp["rotational_speed"],
        "Torque [Nm]": inp["torque"],
        "Tool wear [min]": inp["tool_wear"],
    }
    for k, v in mapping.items():
        if k in row:
            row[k] = v
    type_col = f"Type_{inp.get('type', 'M')}"
    if type_col in row:
        row[type_col] = 1
    return pd.DataFrame([row])[cols]


def predict_one(inp: dict, which: str = "xgb", artifacts=None) -> dict:
    model, scaler, cols = artifacts or load_artifacts(which)
    X = make_feature_row(inp, cols)
    Xs = scaler.transform(X)
    proba = float(model.predict_proba(Xs)[0, 1])
    label = int(proba >= 0.5)
    return {"pred_label": label, "pred_proba": round(proba, 4)}


def predict_and_log(inp: dict, session, which: str = "xgb", artifacts=None) -> dict:
    result = predict_one(inp, which=which, artifacts=artifacts)
    machine = db.Machine(product_id=inp.get("product_id", "DEMO"), type=inp.get("type", "M"))
    session.add(machine)
    session.flush()
    reading = db.SensorReading(
        machine_id=machine.machine_id,
        air_temperature=inp["air_temperature"],
        process_temperature=inp["process_temperature"],
        rotational_speed=inp["rotational_speed"],
        torque=inp["torque"],
        tool_wear=inp["tool_wear"],
    )
    session.add(reading)
    session.flush()
    pred = db.Prediction(
        reading_id=reading.reading_id,
        pred_label=result["pred_label"],
        pred_proba=result["pred_proba"],
    )
    session.add(pred)
    session.commit()
    result["prediction_id"] = pred.prediction_id
    return result
