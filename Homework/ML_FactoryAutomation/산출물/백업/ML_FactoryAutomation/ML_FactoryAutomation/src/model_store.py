"""학습한 모델을 DB에 저장(BLOB)하고, 예측 시 DB에서 읽어와 바로 응답하는 모델 스토어.

흐름:
  학습(train) → 모델 번들(model+scaler+columns) 직렬화 → DB(model_registry.artifact) 저장
  사용자 예측 요청 → DB에서 활성 모델 로드(메모리 캐시) → 즉시 추론·응답
"""
from __future__ import annotations
import io
import joblib
import pandas as pd
from sqlalchemy import select, desc
from src import config, db
from src.predict import make_feature_row, load_artifacts

# 프로세스 메모리 캐시: (model_id, bundle) — DB 재조회 없이 즉시 응답
_CACHE: dict[str, tuple[int, dict]] = {}


def _serialize(model, scaler, cols, threshold: float, metrics: dict) -> bytes:
    buf = io.BytesIO()
    joblib.dump({"model": model, "scaler": scaler, "columns": list(cols),
                 "threshold": float(threshold), "metrics": metrics}, buf)
    return buf.getvalue()


def _deserialize(blob: bytes) -> dict:
    return joblib.load(io.BytesIO(blob))


def save_model_to_db(which: str = "xgb", *, name: str = "pdm-failure",
                     version: str = "v1", threshold: float = 0.5,
                     metrics: dict | None = None, session=None) -> int:
    """디스크의 학습 산출물을 직렬화하여 DB에 저장하고 model_id 반환.

    동일 name 의 기존 모델은 비활성화(is_active=0) 후 새 모델을 활성으로 등록.
    """
    own = session is None
    session = session or db.get_session()
    try:
        model, scaler, cols = load_artifacts(which)  # model/*.pkl 우선, 없으면 joblib
        metrics = metrics or {}
        blob = _serialize(model, scaler, cols, threshold, metrics)

        # 기존 동일 이름 모델 비활성화
        for old in session.execute(select(db.ModelRegistry).where(
                db.ModelRegistry.name == name, db.ModelRegistry.is_active == 1)).scalars():
            old.is_active = 0

        reg = db.ModelRegistry(
            name=name, version=version, algorithm=which,
            accuracy=metrics.get("Accuracy"), f1_score=metrics.get("F1"),
            roc_auc=metrics.get("ROC_AUC"), pr_auc=metrics.get("PR_AUC"),
            threshold=threshold, is_active=1, artifact=blob)
        session.add(reg)
        session.commit()
        _CACHE.pop(name, None)  # 캐시 무효화
        return reg.model_id
    finally:
        if own:
            session.close()


def load_model_from_db(name: str = "pdm-failure", session=None) -> tuple[int, dict]:
    """DB에서 활성 모델을 로드. 메모리에 캐시하여 재요청 시 즉시 반환."""
    own = session is None
    session = session or db.get_session()
    try:
        reg = session.execute(
            select(db.ModelRegistry)
            .where(db.ModelRegistry.name == name, db.ModelRegistry.is_active == 1)
            .order_by(desc(db.ModelRegistry.trained_at))).scalars().first()
        if reg is None:
            raise ValueError(f"DB에 활성 모델이 없습니다: name={name}. 먼저 save_model_to_db 실행 필요.")
        cached = _CACHE.get(name)
        if cached and cached[0] == reg.model_id:
            return cached
        bundle = _deserialize(reg.artifact)
        bundle["model_id"] = reg.model_id
        _CACHE[name] = (reg.model_id, bundle)
        return reg.model_id, bundle
    finally:
        if own:
            session.close()


def predict_from_db(inp: dict, name: str = "pdm-failure", session=None) -> dict:
    """예측 요청 → DB(또는 캐시)에서 학습 모델 로드 → 즉시 추론 응답."""
    model_id, bundle = load_model_from_db(name=name, session=session)
    X = make_feature_row(inp, bundle["columns"])
    Xs = bundle["scaler"].transform(X)
    proba = float(bundle["model"].predict_proba(Xs)[0, 1])
    th = float(bundle.get("threshold", 0.5))
    return {"pred_label": int(proba >= th), "pred_proba": round(proba, 4),
            "threshold": th, "model_id": model_id, "source": "db"}


def _load_metrics(which: str) -> dict:
    """reports/model_comparison.csv 에서 해당 모델 지표 로드(있으면)."""
    f = config.REPORTS_DIR / "model_comparison.csv"
    name_map = {"xgb": "XGBoost", "rf": "RandomForest", "logreg": "LogReg"}
    if not f.exists():
        return {}
    df = pd.read_csv(f, index_col=0)
    key = name_map.get(which, which)
    if key not in df.index:
        return {}
    r = df.loc[key]
    out = {}
    for src_col, dst in [("ROC-AUC", "ROC_AUC"), ("ROC_AUC", "ROC_AUC"),
                         ("F1", "F1"), ("Accuracy", "Accuracy"), ("PR-AUC", "PR_AUC"),
                         ("PR_AUC", "PR_AUC")]:
        if src_col in r.index:
            out[dst] = float(r[src_col])
    return out


if __name__ == "__main__":
    db.init_db()
    mid = save_model_to_db(which="xgb", threshold=0.5, metrics=_load_metrics("xgb"))
    print(f"DB 저장 완료: model_id={mid}")
    demo = {"type": "L", "air_temperature": 300.0, "process_temperature": 310.0,
            "rotational_speed": 1500, "torque": 40.0, "tool_wear": 120}
    print("DB 기반 예측:", predict_from_db(demo))


def predict_dataframe_from_db(df, name: str = "pdm-failure", session=None):
    """CSV/DataFrame 일괄 예측 (DB의 활성 모델 사용).

    df 는 Type + 5개 수치 feature(+선택적 Target 실제라벨)를 포함한다.
    반환: (결과 DataFrame, model_id, has_actual)
      결과에는 pred_proba/pred_label 가 추가되고, Target 이 있으면 actual/match 도 추가.
    """
    from src import preprocess
    model_id, bundle = load_model_from_db(name=name, session=session)
    has_actual = "Target" in df.columns
    work = df.copy()
    if not has_actual:
        work = work.assign(Target=0)
    X, y = preprocess.build_features(work)
    X = X.reindex(columns=bundle["columns"], fill_value=0)
    Xs = bundle["scaler"].transform(X)
    proba = bundle["model"].predict_proba(Xs)[:, 1]
    th = float(bundle.get("threshold", 0.5))
    pred = (proba >= th).astype(int)
    out = df.copy()
    out["pred_proba"] = proba.round(4)
    out["pred_label"] = pred
    if has_actual:
        out["actual"] = y.values
        out["match"] = (pred == y.values)
    return out, model_id, has_actual


def ensure_model_in_db(which: str = "xgb", name: str = "pdm-failure", session=None) -> int:
    """DB에 활성 모델이 없으면 디스크 산출물을 저장하고 model_id 반환."""
    try:
        mid, _ = load_model_from_db(name=name, session=session)
        return mid
    except Exception:
        return save_model_to_db(which=which, name=name, session=session)
